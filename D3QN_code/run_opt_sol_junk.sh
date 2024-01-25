for ten_times_inst_id in 0 #$(seq 0 0 0)
do
	for gr_id in 1 #2 3 4 5
	do
		for inst_id_within in $(seq 1 1 1)
		do
			python3 main_run_opt.py ${gr_id} $(((${ten_times_inst_id} * 10) + ${inst_id_within})) &
		done
	wait
	done
done
