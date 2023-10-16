'''
Author: niko
Date: 2021-09-21 15:54:02
克隆代码一致性修改需求预测第二版
输入: 合并相同子图之后的克隆对PDG
'''
# https://blog.csdn.net/beilizhang/article/details/111936390
import argparse
import csv
import re
import os
import codecs
import json
import time
import datetime
import random
import math
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dgl.data.utils import save_graphs, load_graphs, load_labels
from sklearn.metrics import precision_score, recall_score

codeTokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codeBertModel = AutoModel.from_pretrained("microsoft/codebert-base")


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        '''
            in_feats: 输入向量的维度
            hid_feats: 隐藏层的维度
            out_feats: 输出向量的维度
            rel_names: 边名字列表
        '''
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats, hid_feats)
             for rel in rel_names},
            aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hid_feats, out_feats)
             for rel in rel_names},
            aggregate='sum')

    def forward(self, graph, inputs):
        '''
            graph: 图
            inputs: 节点特征向量tensor
        '''
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        '''
            in_dim: 输入向量的维度
            hidden_dim: 隐藏层的维度
            n_classes: 分类的数量，对应输出向量的维度
            rel_names: 边名字列表
        '''
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        '''
            graph: 图
        '''
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)


class Node:
    def __init__(self, text):
        '''
            节点包含三个标签
            id：节点id
            label：节点包含的文本，pdg的每个节点对应一个代码表达式
            modify：该节点是否是修改的节点
        '''
        self.id, self.label, self.modify = self.extractNodeInfo(text)

    def extractNodeInfo(self, text):
        nodeId = text.split(" ")[0]
        labelPa = re.compile(r'\"(.*)\",')
        label = labelPa.findall(text)
        modifyPa = re.compile(r'AAA(.*)AAA')
        modify = modifyPa.findall(text)
        return nodeId, label[0], modify[0]


class Edge:
    def __init__(self, text):
        '''
            src: 源节点的id
            target：目标节点的id
            label：数据流边上有变量名作为label
            relation：边的类型名
        '''
        self.src, self.target, self.label, self.relation = self.extractEdgeInfo(
            text)

    def extractEdgeInfo(self, text):
        edge = text.split(" ")[0]
        ids = edge.split("->")
        src = ids[0]
        target = ids[1]

        labelPa = re.compile(r'\"(.*)\"')
        label = labelPa.findall(text)

        relation = text.split(" ")[3].replace(",", "")
        # 用label 0 | 1 来区分完全相同或者差异对等节点
        # if((label[0] == "0" or label[0] == "1") and relation == "dashed"):
        #     relation += label[0]

        return src, target, label[0], relation


def constructGraph(path):
    '''
        解析dot文件，构建dgl图对象
    '''
    nodeDict = dict()
    edgeDict = dict()
    edgeMap = {
        "solid": 'dataFlow',
        "dotted": 'controlFlow',
        "bold": 'executionFlow',
        "dashed": 'change'
    }
    with open(path, "r") as f:
        for line in f:
            if ("[" not in line):
                continue
            if ("->" in line):
                edge = Edge(line)
                if (edge.relation in edgeDict):
                    edgeDict[edge.relation].append(edge)
                else:
                    edgeDict[edge.relation] = list()
                    edgeDict[edge.relation].append(edge)
            else:
                node = Node(line)
                nodeDict[node.id] = node

    edgeInputDict = dict()
    existedEdge = set()
    for key in edgeDict:
        # print(key)
        edgeList = edgeDict[key]
        srcNodeList = list()
        targetNodeList = list()
        for edge in edgeList:
            srcNodeList.append(int(edge.src))
            targetNodeList.append(int(edge.target))

        edgeNameTuple = ('node', edgeMap[key], 'node')
        # print(edgeNameTuple)
        edgeValueTuple = (torch.tensor(srcNodeList),
                          torch.tensor(targetNodeList))
        edgeInputDict[edgeNameTuple] = edgeValueTuple
        existedEdge.add(edgeNameTuple)

    # 单一节点运行会报错，这里给每个graph加一个入口节点  &&  运行发现dgl.batch要求所有的graph 的边 都对应，因此对所有图都加上所有类型边
    entryTuple = ('node', 'entry', 'entryNode')
    edgeInputDict[entryTuple] = (torch.tensor([0]), torch.tensor([0]))
    # {"solid": 'dataFlow', "dotted": 'controlFlow', "bold": 'executionFlow', "dashed": 'equivalent'}

    edgeInputDict = addLeftEdge(edgeInputDict, existedEdge)
    clonePdgGraph = dgl.heterograph(edgeInputDict)

    clonePdgGraph.nodes['node'].data['feat'] = getNodeVec(nodeDict)
    clonePdgGraph.nodes['entryNode'].data['feat'] = getEntryEmbedding()
    # print(getNodeVec(nodeDict))
    return clonePdgGraph


def addLeftEdge(edgeInputDict, existedEdge):
    '''
        原始的图样本中，有些图会缺少一些边，这导致没法进行批处理
        每个图增加一个entry节点，并且对它添加各种类型的边，保证每个图里都有所有类型的边
    '''
    dataFlowTuple = ('node', 'dataFlow', 'node')
    controlFlowTuple = ('node', 'controlFlow', 'node')
    executionFlowTuple = ('node', 'executionFlow', 'node')
    changeTuple = ('node', 'change', 'node')

    if (dataFlowTuple not in existedEdge):
        edgeInputDict[dataFlowTuple] = (torch.tensor([0]), torch.tensor([0]))
    if (controlFlowTuple not in existedEdge):
        edgeInputDict[controlFlowTuple] = (torch.tensor([0]), torch.tensor([0
                                                                            ]))
    if (executionFlowTuple not in existedEdge):
        edgeInputDict[executionFlowTuple] = (torch.tensor([0]),
                                             torch.tensor([0]))
    if (changeTuple not in existedEdge):
        edgeInputDict[changeTuple] = (torch.tensor([0]), torch.tensor([0]))

    return edgeInputDict


def getNodeDisVec(label):
    '''
        节点位置信息向量，（1, 0, 0）表示该节点只存在于克隆实例C1中
    '''
    distribution = label[-13:-1]
    disArr = distribution.split(",")
    vec = [int(item.replace("'", "")) for item in disArr]
    return vec


def getNodeVec(nodeDict):
    '''
        获取所有节点的embedding，codebert输出 + 位置向量
    '''
    nodeVec = list()
    nodeSize = len(nodeDict)
    for i in range(nodeSize):
        node = nodeDict[str(i)]
        label = node.label
        label = label.split("@@@")[0]
        modify = int(node.modify)
        tokens = [codeTokenizer.cls_token] + codeTokenizer.tokenize(label)
        tokens_ids = codeTokenizer.convert_tokens_to_ids(tokens)
        distributionVec = getNodeDisVec(node.label)
        # modifyList = [modify]
        # distributionVec.extend(modifyList)
        nodeVec.append(getFeatureEmbedding(tokens_ids, distributionVec))
    # print(th.tensor(nodeVec).shape)
    return torch.tensor(nodeVec)


def arrageVec(tokenIds, dim, distributionVec):
    '''
        拼接位置向量，现已废弃
    '''
    tokenNum = len(tokenIds)
    arrangedVec = list()
    if (tokenNum >= dim):
        arrangedVec = tokenIds[:dim]
    else:
        leftNum = dim - tokenNum
        arrangedVec = tokenIds
        restVec = [0 for i in range(leftNum)]
        arrangedVec.extend(restVec)
    arrangedVec.extend(distributionVec)
    return arrangedVec


def getEntryEmbedding():
    '''
        entry节点的embedding
    '''
    ids = [codeTokenizer.cls_token] + [codeTokenizer.sep_token]
    tokens_ids = codeTokenizer.convert_tokens_to_ids(ids)
    inputEmbedding = torch.tensor(tokens_ids)[None, :]
    context_embeddings = codeBertModel(inputEmbedding)[0]
    context_embeddings = context_embeddings.tolist()[0] + [1, 1, 1]
    # return torch.tensor([context_embeddings])
    return torch.tensor([[0] * 771])


def getFeatureEmbedding(tokenIds, distributionVec):
    '''
        每个节点包含一个表达式，基于codebert计算该节点的embedding
    '''
    inputEmbedding = torch.tensor(tokenIds)[None, :]
    context_embeddings = codeBertModel(inputEmbedding)[0][0]
    context_embeddings = context_embeddings.tolist()[0] + distributionVec
    return context_embeddings


def getPredictResult(logits, args):
    '''
        判断预测结果，0为非一致性修改，1为一致性修改
    '''
    # 非一致性概率
    ip = F.softmax(logits, dim=1)[0][0].item()
    # 一致性概率
    cp = F.softmax(logits, dim=1)[0][1].item()

    if ip > args.softmax_threhold:
        return 0
    if cp > args.softmax_threhold:
        return 1
    return 0


def trans(x):
    if x == 0:
        return 1
    else:
        return 0


def testRecall(model, testDataset, args):
    '''
        r: 一致性修改的召回率
        ir：非一致性修改的召回率
    '''
    true_list, pred_list = [], []
    for graph, labels in testDataset:
        graph = graph.to("cuda")
        logits = model(graph)
        predLabel = getPredictResult(logits, args)
        if args.do_embedding:
            true_list.append(labels.item())
        else:
            true_list.append(labels)
        pred_list.append(predLabel)

    r = recall_score(true_list, pred_list, zero_division=0)
    true_list = [trans(x) for x in true_list]
    pred_list = [trans(x) for x in pred_list]
    ir = recall_score(true_list, pred_list, zero_division=0)

    return r, ir


def testPrecision(model, testDataset, args):
    '''
        p: 一致性修改的精确率
        ip：非一致性修改的精确率
    '''
    true_list, pred_list = [], []
    for graph, labels in testDataset:
        graph = graph.to("cuda")
        logits = model(graph)
        predLabel = getPredictResult(logits, args)
        if args.do_embedding:
            true_list.append(labels.item())
        else:
            true_list.append(labels)
        pred_list.append(predLabel)

    p = precision_score(true_list, pred_list, zero_division=0)
    true_list = [trans(x) for x in true_list]
    pred_list = [trans(x) for x in pred_list]
    ip = precision_score(true_list, pred_list, zero_division=0)

    return p, ip

def eachProjectPrecisionAndRecall(model, testDataset, args):
    projectList = readProjectInfo(args.project_test_dataset_file)
    del(projectList[0])
    projectTestList = list()
    for project in projectList:

        start = int(project[1])
        length = int(project[2])
        projectTestDataset = testDataset[start: (start + length)]
        p, ip = testPrecision(model, projectTestDataset, args)
        r, ir = testRecall(model, projectTestDataset, args)
        projectTestList.append((project[0], p, ip, r, ir))
    return projectTestList



def loadDataset(args):
    '''
        加载数据集
        如果有生成好的图数据，则直接加载
        没有生成好的数据，则根据dot文件来生成graph 【生成graph很慢，需要先执行run_embedding来生成并保留图数据】
    '''

    if not args.do_embedding:
        return load_local_graph_data(args)

    pdgPath = args.pdg_path
    pdgFileList = os.listdir(pdgPath)
    consistData = list()
    inConsistData = list()
    data = list()

    pdgFileList.sort()
    normalFileList = list()

    for pdgFile in tqdm(pdgFileList):
        try:
            consistLabel, graphLabel = extractGraphLabel(pdgFile)
            graphLabel = torch.LongTensor(graphLabel)
            pdgGraph = constructGraph(pdgPath + "/" + pdgFile)
            # # print(consistLabel)
            if consistLabel == '1' or consistLabel == '0':
                data.append((pdgGraph, graphLabel))
                normalFileList.append(pdgFile)
                # consistData.append((pdgGraph, graphLabel))
            # else:
            #     inConsistData.append((pdgGraph, graphLabel))
            #     data.append((pdgGraph, graphLabel))

        except:
            print(pdgPath+"/"+pdgFile)
            print("dot format error")
            pass

    # dataSize = min(len(consistData), len(inConsistData))
    # print("datasize",dataSize)
    # random.shuffle(consistData[:dataSize])
    # random.shuffle(inConsistData[:dataSize])
    # print(consistData, inConsistData)

    # consistData.extend(inConsistData)

    # random.shuffle(consistData)

    # print(consistData, file=open('graph.txt', 'w', encoding='utf-8'))

    # label_list = [l.item() for _, l in consistData]
    # graph_list = [g for g, _ in consistData]
    # graph_labels = {"Labels": torch.tensor(label_list)}

    label_list = [l.item() for _, l in data]
    graph_list = [g for g, _ in data]
    graph_labels = {"Labels": torch.tensor(label_list)}

    # for g in graph_list:
    #     print(g.nodes['node'].data['feat'])

    save_graphs(args.graph_data_file, graph_list, graph_labels)
    extractGraphProjectName(normalFileList, args)
    print("embedding finish")
    print("dataLength: ", len(data))
    # random.shuffle(consistData)
    # return consistData
    return data


def load_local_graph_data(args):
    '''
        加载图数据
    '''
    # 将
    graph_list = load_graphs(args.graph_data_file)
    label_dict = load_labels(args.graph_data_file)
    graph_list = list(graph_list[0])
    label_dict = label_dict["Labels"].numpy().tolist()
    data_len = len(label_dict)
    dataset = []
    print("graph_list[0]: ", graph_list[0])
    print("label_dict[0]: ", label_dict[0])

    for i in tqdm(range(data_len)):
        # print(graph_list[i].nodes['node'].data['feat'])
        dataset.append((graph_list[i], label_dict[i]))
    # print(dataset, file=open('graph1.txt', 'w', encoding='utf-8'))

    print("datalength: ", len(dataset))
    random.shuffle(dataset)
    return dataset


def spliceDataset(dataset, ratio):
    '''
        切分数据集
    '''
    datasetSize = len(dataset)
    spliceIndex = math.floor(datasetSize * ratio)
    return dataset[0:(spliceIndex - 1)], dataset[spliceIndex:]


# 按照每个项目8:1:1分割数据集
def spliceDatasetNew(dataset, trainRation, testRation, args):
    projectList = readProjectInfo(args.project_info_file)
    trainDataset = list()
    evaDataset = list()
    testDataset = list()
    testList = list()
    testIndex = 0

    for project in projectList:
        start = int(project[1])
        length = int(project[2])
        projectData = dataset[start: (start + length)]
        random.shuffle(projectData)

        spliceIndex1 = math.floor(length * trainRation)
        spliceIndex2 = math.floor(spliceIndex1 + length * testRation)

        trainDataset.extend(projectData[0:spliceIndex1])
        evaDataset.extend(projectData[spliceIndex1:spliceIndex2])
        testDataset.extend(projectData[spliceIndex2:])

        testList.append((project[0], testIndex, (length-spliceIndex2)))
        testIndex += length-spliceIndex2

    print("splice finish")
    print(len(trainDataset))
    print(len(evaDataset))
    print(len(testDataset))
    if not os.path.exists(args.project_test_dataset_file):
        # 如果文件存在，删除文件
        # os.remove(args.project_test_dataset_file)
        with open(args.project_test_dataset_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(("projectName", "startLine", "length"))
            for row in testList:
                writer.writerow(row)

    return trainDataset, evaDataset, testDataset


def extractGraphLabel(fileName):
    '''
        提取图标签【0为非一致性修改，1为一致性修改】
    '''
    labelPa = re.compile(r'[0-9]')
    label = labelPa.findall(fileName)
    return label[-1], np.array([int(label[-1])])


def extractGraphProjectName(fileList, args):
    # pdgFileList = os.listdir(directory)
    fileList.sort()
    print("pdgFileListLength: ", len(fileList))
    projectName = fileList[0][:fileList[0].find("-")]
    start = 0
    end = 0
    projectList = list()
    for pdgFile in tqdm(fileList):
        curProjectName = pdgFile[:pdgFile.find("-")]
        if (curProjectName != projectName):
            projectList.append((projectName, start, end - start + 1))
            projectName = curProjectName
            start = end + 1
        end += 1
    projectList.append((projectName, start, end - start))
    print("projectSize: ", len(projectList))
    if os.path.exists(args.project_info_file):
        # 如果文件存在，删除文件
        os.remove(args.project_info_file)
    with open(args.project_info_file, "w", newline='') as f:
        writer = csv.writer(f)
        for row in projectList:
            writer.writerow(row)


def readProjectInfo(filename):
    projectInfoList = list()
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            projectInfoList.append(row)
    return projectInfoList


# 计算准确率
def binary_acc(preds, y):
    '''
        准确率计算，用于训练过程中批处理
    '''
    # 过了一道 softmax 的激励函数后的最大概率才是预测值
    preds = torch.max(F.softmax(preds, dim=1), 1)[1]
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)  # 预测中有多少和真实值一样

    return acc


def collate(samples):
    '''
        划分数据集batch
    '''
    # graphs, labels = map(list, zip(*samples))
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs).to("cuda")
    batched_labels = torch.tensor(labels).cuda()
    return batched_graph, batched_labels


# 评估函数
def evaluate(model, evaDataset, opt, args):
    '''
        评估模型
    '''
    avg_loss = []
    avg_acc = []
    model.eval()  # 表示进入测试模式

    # evaProjectList = list()
    # avg_loss_list = list()
    # evaProjectList = readProjectInfo("./evaList.csv")
    # for evaProject in evaProjectList:
    #     projectName = evaProject[0]
    #     start = int(evaProject[1])
    #     length = int(evaProject[2])
    #     evaProjectData = evaDataset[start: (start + length)]

    trainDataLoader = DataLoader(evaDataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=True,
                                 shuffle=True)

    for batched_graph, labels in trainDataLoader:
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels.squeeze(-1))

        acc = binary_acc(logits, labels).item()  # 计算每个batch的准确率

        avg_loss.append(loss.item())
        avg_acc.append(acc)

        opt.zero_grad()
        loss.backward()
        opt.step()

    avg_acc = 0 if len(avg_acc) == 0 else np.array(avg_acc).mean()
    avg_loss = 0 if len(avg_loss) == 0 else np.array(avg_loss).mean()
    return avg_loss, avg_acc


def train(model, trainDataset, evaDataset, opt, args):
    '''
        训练模型
    '''
    model.train()
    avg_loss = []
    avg_acc = []
    trainDataLoader = DataLoader(trainDataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=True,
                                 shuffle=True)

    for batched_graph, labels in trainDataLoader:
        # print(batched_graph, labels)
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels.squeeze(-1))

        acc = binary_acc(logits, labels).item()  # 计算每个batch的准确率

        avg_loss.append(loss.item())
        avg_acc.append(acc)

        opt.zero_grad()
        loss.backward()
        opt.step()

    avg_acc = 0 if len(avg_acc) == 0 else np.array(avg_acc).mean()
    avg_loss = 0 if len(avg_loss) == 0 else np.array(avg_loss).mean()

    evaLoss, evaAcc = evaluate(model, evaDataset, opt, args)
    return avg_loss, avg_acc, evaLoss, evaAcc


def pipline():
    '''
        程序运行入口方法
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--do_embedding",
        action="store_true",
        help="Whether to run embedding."
    )

    parser.add_argument(
        "--hidden_dim",
        default=350,
        type=int,
        required=True,
        help="The Dimension of Hidden Layer"
    )

    parser.add_argument(
        "--pdg_path",
        default=None,
        type=str,
        required=True,
        help="The pdg folder path.",
    )

    parser.add_argument(
        "--model_save_dir",
        default=None,
        type=str,
        required=True,
        help="The model save dir.",
    )

    parser.add_argument(
        "--graph_data_file",
        default=None,
        type=str,
        required=True,
        help="The embedding graph data to be saved.",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--softmax_threhold",
        default=0.7,
        type=float,
        help="The softmax threhold.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=15,
        help="Num train epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size."
    )

    parser.add_argument(
        "--project_info_file",
        default=None,
        type=str,
        required=True,
        help="Project info store file"
    )

    parser.add_argument(
        "--project_test_dataset_file",
        default=None,
        type=str,
        required=True,
        help="Project test dataset info"
    )

    args = parser.parse_args()
    print("args ", args)

    dataset = loadDataset(args)
    trainDataset, evaDataset, testDataset = spliceDatasetNew(dataset, 0.8, 0.1, args)

    print(len(dataset))

    # trainDataset, testDataset = spliceDataset(dataset, 0.9)
    # trainDataset, evaDataset = spliceDataset(trainDataset, 0.9)

    etypes = ['dataFlow', 'controlFlow', 'executionFlow', 'change', 'entry']
    print(etypes)
    print("total data: ", len(trainDataset), len(evaDataset))
    model = HeteroClassifier(771, args.hidden_dim, 2, etypes).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    opt1 = torch.optim.Adam(model.parameters(), lr=1e-6)

    for epoch in tqdm(range(args.num_train_epochs)):
        start_time = time.time()
        if epoch == 60:
            opt = opt1
        train_loss, train_acc, dev_loss, dev_acc = train(model, trainDataset, evaDataset, opt, args)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(
            f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s'
        )
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%'
        )
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc * 100:.2f}%')

    p, ip = testPrecision(model, testDataset, args)
    r, ir = testRecall(model, testDataset, args)

    project_test_list = eachProjectPrecisionAndRecall(model, testDataset, args)

    model_to_save = model
    dt = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    pr = "--P--%.2f--R--%.2f--IP%.2f--IR%.2f" % (p, r, ip, ir)
    output_dir = os.path.join(args.model_save_dir, str(dt) + pr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_info = {
        "consistentPrecision": format(p.item(), '.3f'),
        "consistentRecall": format(r.item(), '.3f'),
        "inconsistentPrecision": format(ip.item(), '.3f'),
        "inconsistentRecall": format(ir.item(), '.3f'),
        "hiddenDim": args.hidden_dim,
        "learningRate": args.learning_rate,
        "epochs": args.num_train_epochs,
        "batchSize": args.batch_size,
        "projectTestResult": [],
    }

    for project_test_res in project_test_list:
        model_info["projectTestResult"].append({
            "projectName": project_test_res[0],
            "consistentPrecision": format(project_test_res[1].item(), '.3f'),
            "consistentRecall": format(project_test_res[2].item(), '.3f'),
            "inconsistentPrecision": format(project_test_res[3].item(), '.3f'),
            "inconsistentRecall": format(project_test_res[4].item(), '.3f')})

    model_info_path = os.path.join(output_dir, "modelInfo.json")
    model_save_dir = os.path.join(output_dir, "model.bin")
    torch.save(model_to_save.state_dict(), model_save_dir)
    json_data = json.dumps(model_info, sort_keys=False, indent=4, separators=(',', ':'))
    with codecs.open(model_info_path, 'a+', encoding='utf-8') as f:
        f.write(json_data)

    print("Saving model checkpoint to %s" % output_dir)


if __name__ == '__main__':
    pipline()
