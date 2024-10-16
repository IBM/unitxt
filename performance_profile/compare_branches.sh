# generate a safe
safe=`date +"%H_%M_%S_%N"`
mkdir $safe
# copy the updated card_prifller, potentially updated by this current branch to the safe, 
# to be accessible to another branch
cp performance_profile/card_profiler.py $safe
# record the name of this current branch
current_branch=`git branch --show-current`
# check out any other branch against which you want to compare the current branch
# e.g. here: main
git checkout main  
# prepare the dirs for performance evaluation on main
# before that - determine the way to get things back in place, so can checkout 
# from main
if [ -e performance_profile ]; then
    end_performance_profile="echo directory performance_profile was here before and it stays"
    if [ -e performance_profile/logs ]; then
        end_performance_profile_logs="echo directory performance_profile/logs was here before and it stays"
        if [ -e performance_profile/logs/cards_benchmark.prof ]; then
            end_cards_benchmark_prof="git restore performance_profile/logs/cards_benchmark.prof"
        else 
            end_cards_benchmark_prof="rm performance_profile/logs/cards_benchmark.prof"
        fi
        if [ -e performance_profile/logs/cards_benchmark.json ]; then
            end_cards_benchmark_json="git restore performance_profile/logs/cards_benchmark.json"
        else 
            end_cards_benchmark_json="rm performance_profile/logs/cards_benchmark.json"
        fi        
    else 
        end_performance_profile_logs="rm -rf performance_profile/logs"
    fi
    if [ -e performance_profile/card_profiler.py ]; then
        end_performance_profile_card_profiler="git restore performance_profile/card_profiler.py"
    else 
        end_performance_profile_card_profiler="rm performance_profile/card_profiler.py"
    fi  
else 
    end_performance_profile="rm -rf performance_profile"
fi

mkdir -p performance_profile
mkdir -p performance_profile/logs
# copy out card_profiler from the safe
cp $safe/card_profiler.py  performance_profile/
# run performance on main branch and save result in safe
python -m performance_profile.card_profiler
cp performance_profile/logs/cards_benchmark.json $safe/main_cards_benchmark.json
# delete all new files that may have counterparts in current branch, and prevent the git checkout back
eval "$end_performance_profile_card_profiler"
eval "$end_cards_benchmark_prof"
eval "$end_cards_benchmark_json"
eval "$end_performance_profile_logs"
eval "$end_performance_profile"
# checkout back to current branch
git checkout $current_branch
mkdir -p performance_profile/logs
# Run performance on PR branch
python -m performance_profile.card_profiler
cp performance_profile/logs/cards_benchmark.json performance_profile/logs/pr_cards_benchmark.json
# Download main performance result from the safe
cp $safe/main_cards_benchmark.json performance_profile/logs/
# compare main and PR performance results
python -m performance_profile.compare_performance_results
