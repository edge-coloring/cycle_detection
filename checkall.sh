#!/bin/bash
set -euo pipefail

while IFS="," read -r filePath r c conts
do
    if (( c >= 5 ));
    then
        echo "Checking ${filePath} with conts ${conts}"
        fileName=`basename ${filePath%.*}`
        fileBase=${fileName%.*}
        python3 check_safe.py ${1}/conf/${fileBase}.conf `echo "${conts}" | tr -d '"'`
    fi
done < "${1}/summary.csv"