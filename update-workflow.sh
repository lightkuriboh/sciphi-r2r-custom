git clone {THIS_REMOTE_REPO} ../work-on-private-B
cd ../work-on-private-B
git remote add upstream git@github.com:SciPhi-AI/R2R.git
git fetch upstream
git checkout -b update_stream main
git merge upstream/main

echo "Resolve conflicts now!"

# git checkout main
# git merge update_stream --no-ff
# git push origin main
# git branch -d update_stream




