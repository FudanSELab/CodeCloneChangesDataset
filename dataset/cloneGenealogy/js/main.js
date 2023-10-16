var proList = ["spark","groovy","iceberg","beam","canal","tomcat","pinot","dbeaver","rxjava","jabref","mybatis3","commonstext","dnsjava","maven","lucene","netbeans","curator","zookeeper","hudi","kafka","camel","subversion","druid","zipkin","shardingsphere","ant","dolphinscheduler","cloudstack","helix","skywalking","juneau","gobblin","hadoop","poi","springboot","geode","jenkins","protobuf","calcite","guava","cassandra","pulsar","isis","solr","jfreechart","incubatordoris","mina","flink","elasticsearch","apollo","argouml"];
var proPath = "/res/" + proList[0];
var colors = ['orchid', 'purple', 'cornflowerblue', 'blue', 'darkcyan', 'turquoise', 'springgreen', 'green', 'darkkhaki', 'gold'];
var colorIndex = 0;

function getNextColor() {
    if (colorIndex === colors.length){
        return 'black';
    }
    return colors[colorIndex++];
}

function onSearchClick(e) {
    colorIndex = 0;
    var id = $('#projectName').val();
//    var proPath = $('#proPath').val();
    console.log(proPath, id);
    mainEntry(proPath + "/" + id + ".json")
}

$(document).ready(function () {
    $("#projectName").on("keydown", function (e) {
        if (e.keyCode === 13) {
            //to-do when click Enter Key
            console.log('enter keydown');
            onSearchClick(e);
        }
    });
});


function contains(string, substr, isIgnoreCase) {
    if (isIgnoreCase) {
        string = string.toLowerCase();
        substr = substr.toLowerCase();
    }

    var startChar = substr.substring(0, 1);
    var strLen = substr.length;

    for (var j = 0; j < string.length - strLen + 1; j++) {
        if (string.charAt(j) == startChar)  //如果匹配起始字符,开始查找
        {
            if (string.substring(j, j + strLen) == substr)  //如果从j开始的字符与str匹配，那ok
            {
                return true;
            }
        }
    }
    return false;
}

function isBugCommit(commitMessage) {
    if (contains(commitMessage, "bug", true))
        return true;
    if (contains(commitMessage, "fix", true))
        return true;
    if (contains(commitMessage, "issue", true))
        return true;
//    if(contains(commitMessage, "#", true))
//        return true;
    return false;
}

function clickMessage(e) {
//    alert(e.getAttribute('title'));
//    console.log(e.getAttribute('title'))
    prompt("", e.getAttribute('title'));
}



function addReleaseDiv() {
    var releaseOption = document.getElementById("proPath");
    var content = "";

    for (var i = 0; i < proList.length; i++) {
        content += "<option>" + proList[i] + "</option>";
    }
    releaseOption.innerHTML = content;
    console.log(content)
}


function onSelect() {
    console.log("click option");
    var options = document.getElementById("proPath");
    var selectedIndex = options.selectedIndex;
    var text = options.options[selectedIndex].text//获取option的文本内容

    proPath = "/res/" + text + "";
}

function mainEntry(resUrl) {
    console.log(resUrl)
    // var container = d3.select(".diffContainer");
    var container = document.getElementsByClassName("diffContainer")[0];
    var content = "";
    var genealogyLength = 0;
    var pathList = [];
    console.log(container);

    content += "<table style = 'width:100%'>";

    var summaryInfo = new Map();
    d3.json(resUrl, function (data) {
        genealogyLength = data.length;
        var index = 1;
        content += "<tr>";
//        var curlength = data[0]["codes"].length;
//        for(var i = 0; i < curlength; i++){
//            console.log(i, "hello")
//            summaryInfo.set(i, 0);
//            console.log("set", i, "-->" ,summaryInfo.get(i));
        var instanceNum = data[0]["codes"].length;
        var differentPaths = [];
        var pathColorMap = new Map();

        for (var i = 0; i < instanceNum; i++) {
            if (differentPaths.indexOf(data[0]["codes"][i]["realPath"]) < 0){
                console.log('push ' + data[0]["codes"][i]["realPath"]);
                differentPaths.push(data[0]["codes"][i]["realPath"]);
                pathColorMap[data[0]["codes"][i]["realPath"]] = getNextColor();
            }
            console.log(i, "hello");
            summaryInfo.set(i, 0);
            console.log("set", i, "-->", summaryInfo.get(i));
            content += "<td>";
//                pathList[i] = data[0]["codes"][i]["path"];
            content += "<p style='color:red'>" + data[0]["codes"][i]["repoName"] + "<p style='color: " + pathColorMap[data[0]["codes"][i]["realPath"]] +"'>" + data[0]["codes"][i]["realPath"];
            content += "</td>";
        }
//        }
        content += "</tr>";
//        content += "<tr><td>1</td><td>2</td></tr>";
        for (var curClone of data) {
            var equalsNum = 0;
            var instanceNum = curClone["codes"].length;
            console.log(instanceNum);
            var commitMessageOri = curClone["commitMessage"];
//            var commitMessageOri = "default";
            var commitMessage = "";
            for (var char of commitMessageOri) {
                if (char == "'")
                    commitMessage += "*";
                else
                    commitMessage += char;
            }
//            var isBug = isBugCommit(commitMessage);
            var isBug = isBugCommit("default");
            var messageClass = "commitMessage";
            if (isBug)
                messageClass = "bugCommit";

            content += "<tr><td><p class = 'commitTime' style = 'float:left; margin-left:20px'>" + curClone['date'] + " <---> " + curClone['commitId'] + " <---> Submitter:" + curClone["submitter"] + "<--->ModifiedFilesCount:" + curClone['modifiedFileCount'] + "</p><p  class = " + messageClass + " title='" + commitMessage + "'onclick=clickMessage(this)>Commit Message</p></td></tr>";

            content += "<tr>";

            for (var i = 0; i < instanceNum; i++) {
                var curIdL = "compare" + index;
                var codeLeftPre = curClone["codes"][i]["preCode"];
                var codeLeftCur = curClone["codes"][i]["curCode"];
                var status = curClone["codes"][i]["status"];
                /* 
                if (status == "N") {
//                    content += "<td><div class='compare' id ='" + curIdL + "''></div></td>";
                    content += "<td><div>" + "startEndLine:  " + curClone["codes"][i]["startLine"] + "," + curClone["codes"][i]["endLine"] + ". Modify Type:" + curClone["codes"][i]["status"] + ". Group Id:" + curClone["codes"][i]["groupId"] + " id:" + curClone["codes"][i]["id"] + "</div><div class='compare' id ='" + curIdL + "''></div></td>";
                    index += 1;
                    continue;
                }*/
                content += "<td><div>" + "startEndLine:  " + curClone["codes"][i]["startLine"] + "," + curClone["codes"][i]["endLine"] + ".   Modify Type:" + curClone["codes"][i]["status"] + ". Group Id:" + curClone["codes"][i]["groupId"] + " id:" + curClone["codes"][i]["id"] + "</div><div class='compare' id ='" + curIdL + "''></div></td>";
//                if(codeLeftCur != ""){
                if (status == "M") {
                    summaryInfo.set(i, summaryInfo.get(i) + 1);
                    console.log(i, "-->", summaryInfo.get(i));
                }
//                }
                index += 1;
            }

            content += "</tr>";
        }
        content += "</table>";

        var info = "";
        for (i = 0; i < summaryInfo.size; i++) {
            info += ("-" + summaryInfo.get(i));
        }

        contentHeader = "<h2>Clone modification times are: " + info + "</h2>";
        contentHeader += "<h3>Different file count: " + differentPaths.length + "</h3>";
        content = contentHeader + content;
        container.innerHTML = content;

        index = 1;
        for (var curClone of data) {
            var equalsNum = 0;
            var instanceNum = curClone["codes"].length;

            for (var i = 0; i < instanceNum; i++) {
                var codeLeftPre = curClone["codes"][i]["preCode"];
                var codeLeftCur = curClone["codes"][i]["curCode"];
                var codeStatus = curClone["codes"][i]["status"];
                var curIdL = "compare" + index;
                console.log(codeStatus);
                if (codeStatus == "" || codeStatus == "NULL") {     //|| codeStatus == "N" 
                    index++;
                    continue;
                }
//                if(codeStatus != "" || codeStatus != "N"){
                $('#' + curIdL).mergely({
                    width: 750,
                    height: 220,
                    cmsettings: {
                        readOnly: false,
                        lineWrapping: true,
                    }
                });
                $('#' + curIdL).mergely('lhs', codeLeftPre);
                $('#' + curIdL).mergely('rhs', codeLeftCur);
//                }
                index++;
            }
        }
    });
}

window.onload = function () {
    console.log("load")
    addReleaseDiv();
};
