for nodes in 10 #15 20 25

do
	sed -i "s/^NUM_NODES = .*/NUM_NODES = ${nodes}/" ./data_file.py;


	for ten_times_inst_id in $(seq 0 1 4)
	do
		for gr_id in 1 #2 3 #4 5
		do
			for inst_id_within in $(seq 1 1 10)
			do
				python3 main_run_opt.py ${gr_id} $(((${ten_times_inst_id} * 10) + ${inst_id_within})) &
			done
		wait
		done
	done
done
