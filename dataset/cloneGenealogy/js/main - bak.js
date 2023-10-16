
function onSearchClick(e){
    var id = $('#projectName').val();
    console.log(id);
    mainEntry("res/" + id + ".json")
}

/ ** string:原始字符串 substr:子字符串 isIgnoreCase:忽略大小写 */
function contains(string, substr, isIgnoreCase)
{
    if (isIgnoreCase)
    {
         string = string.toLowerCase();
         substr = substr.toLowerCase();
    }

    var startChar = substr.substring(0, 1);
    var strLen = substr.length;

    for (var j = 0; j<string.length - strLen + 1; j++)
    {
         if (string.charAt(j) == startChar)  //如果匹配起始字符,开始查找
         {
             if (string.substring(j, j+strLen) == substr)  //如果从j开始的字符与str匹配，那ok
             {
                 return true;
             }   
         }
    }
    return false;
}

function isBugCommit(commitMessage){
    if(contains(commitMessage, "bug", true))
        return true;
    if(contains(commitMessage, "fix", true))
        return true;
    if(contains(commitMessage, "issue", true))
        return true;
//    if(contains(commitMessage, "#", true))
//        return true;
    return false;
}

function mainEntry(resUrl){
    // var container = d3.select(".diffContainer");
    var container = document.getElementsByClassName("diffContainer")[0];
    var content = "";
    var genealogyLength = 0;
    var pathList = [];
    console.log(container);

    content += "<table style = 'width:100%'>";

    var summaryInfo = new Map();
    d3.json(resUrl, function(data) {
        genealogyLength = data.length;
        var index = 1;
        content += "<tr>";
//        var curlength = data[0]["codes"].length;
//        for(var i = 0; i < curlength; i++){
//            console.log(i, "hello")
//            summaryInfo.set(i, 0);
//            console.log("set", i, "-->" ,summaryInfo.get(i));
            var instanceNum = data[0]["codes"].length;
            for(var i = 0; i < instanceNum; i++){
                console.log(i, "hello");
            summaryInfo.set(i, 0);
            console.log("set", i, "-->" ,summaryInfo.get(i));
                content += "<td>";
//                pathList[i] = data[0]["codes"][i]["path"];
                content += "<p style='color:red'>" + data[0]["codes"][i]["repoName"] + "<p>" + data[0]["codes"][i]["realPath"];
                content += "</td>";
            }
//        }
        content += "</tr>";
//        content += "<tr><td>1</td><td>2</td></tr>";
        for(var curClone of data){
            var equalsNum = 0;
            var instanceNum = curClone["codes"].length;
            console.log(instanceNum);
            var commitMessageOri = curClone["commitMessage"];
            var commitMessage = "";
            for(var char of commitMessageOri){
               if(char == "'")
                   commitMessage += "*";
               else
                   commitMessage += char;
            }
            var isBug = isBugCommit(commitMessage);
            var messageClass = "commitMessage";
            if(isBug)
                messageClass = "bugCommit";

            content += "<tr><td><p class = 'commitTime' style = 'float:left; margin-left:20px'>" + curClone['date'] + " <---> " + curClone['commitId'] + " <---> Submitter:" + curClone["submitter"] + "</p><p  class = " + messageClass + " title='" + commitMessage +"'>Commit Message</p></td></tr>";
            
            content += "<tr>";

            for(var i = 0; i < instanceNum; i++){
                var curIdL = "compare" + index;
                var codeLeftPre = curClone["codes"][i]["preCode"];
                var codeLeftCur = curClone["codes"][i]["curCode"];
                var status = curClone["codes"][i]["status"];
                if(status == "N"){
//                    content += "<td><div class='compare' id ='" + curIdL + "''></div></td>";
                    content += "<td><div>" + "startEndLine:  " + curClone["codes"][i]["startLine"] + "," + curClone["codes"][i]["endLine"] + ". Modify Type:" + curClone["codes"][i]["status"] + ". Group Id:" + curClone["codes"][i]["groupId"] + " id:" + curClone["codes"][i]["id"] + "</div><div class='compare' id ='" + curIdL + "''></div></td>";
                    index += 1;
                    continue;
                }
                content += "<td><div>" + "startEndLine:  " + curClone["codes"][i]["startLine"] + "," + curClone["codes"][i]["endLine"] + ".   Modify Type:" + curClone["codes"][i]["status"] + ". Group Id:" + curClone["codes"][i]["groupId"] + " id:" + curClone["codes"][i]["id"] + "</div><div class='compare' id ='" + curIdL + "''></div></td>";
//                if(codeLeftCur != ""){
                if(status == "M"){
                    summaryInfo.set(i, summaryInfo.get(i) + 1);
                    console.log(i, "-->" ,summaryInfo.get(i));
                }   
//                }
                index += 1;
            }

            content += "</tr>";
        }
        content += "</table>";

        var info = ""
        for (i = 0; i < summaryInfo.size; i++) {
            info +=  ("-" + summaryInfo.get(i));
        }
        
        contentHeader = "<h2>Clone modification times are:" + info + "</h2>";
        content = contentHeader + content;

        container.innerHTML = content;

        index = 1;
        for(var curClone of data){
            var equalsNum = 0;
            var instanceNum = curClone["codes"].length;
            
            for(var i = 0; i < instanceNum; i++){
                var codeLeftPre = curClone["codes"][i]["preCode"];
                var codeLeftCur = curClone["codes"][i]["curCode"];
                var codeStatus = curClone["codes"][i]["status"];
                var curIdL = "compare" + index;
                console.log(codeStatus);
                if(codeStatus == "" || codeStatus == "N" || codeStatus == "NULL"){
                    index++;
                    continue;
                }
//                if(codeStatus != "" || codeStatus != "N"){
                    $('#'+curIdL).mergely({
                        width: 750,
                        height: 220,
                        cmsettings: {
                            readOnly: false, 
                            lineWrapping: true,
                        }
                    });
                    $('#'+curIdL).mergely('lhs', codeLeftPre);
                    $('#'+curIdL).mergely('rhs', codeLeftCur);
//                }
                index ++;
            }
        }
    });
}

$(document).ready(function(){
    // mainEntry("res/result.json")
})