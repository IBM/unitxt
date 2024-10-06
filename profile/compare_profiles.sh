# run this script from the root directory of the new branch
# (sh profile/compare_profiles.sh)
# it will print out the net time (overall time minus loading time) of the 
# new branch, the net time of main, and the ratio of: 
# net time of new branch divided by the net time of main.
python profile/card_profiler.py
RC1=$?
mv profile/logs/benchmark_cards.prof profile/logs/benchmark_cards_new_branch.prof
git checkout main
python profile/card_profiler.py
RC2=$?
echo "Main branch net runtime: $RC2"
echo "New branch net runtime: ${RC1}"
ratio=$(echo "scale = 3; $RC1/$RC2" | bc)
echo "ratio of net times:  new_branch / main_branch is: ${ratio}"
#  echo "scale = 3; $RC1/$RC2" | bc
mv profile/logs/benchmark_cards.prof profile/logs/benchmark_cards_main.prof
