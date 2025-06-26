@echo off
echo Starting Git history cleanup...
echo Creating backup branch...
git branch -f backup-master master

echo Removing large file from Git history...
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch \"AIComplianceMonitoring/data/5000000 HRA Records.csv\"" --prune-empty --tag-name-filter cat -- --all

echo Cleaning up refs...
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo Done! Now try pushing with: git push origin master --force
