#!/bin/bash

echo "第一个参数为: $1"
echo "第二个参数为: $2"
echo "脚本名称为: $0"
echo "脚本接受参数总数为: $#"

curl -I baidu.com
echo "运行命令的状态为:$?"

echo "脚本的ID为:$$"

echo "\$*的结果为:$*"
echo "\$@的结果为:$@"

for i in "$*";
do
        echo $i
done

for j in "$@";
do 
        echo $j
done
