for nodes in 10 15 20 25

do
	sed -i "s/^NUM_NODES = .*/NUM_NODES = ${nodes}/" ./data_file.py;

	for gr_id in 1 2 3 #4 5
	do
		python3 gen_base_inst.py ${gr_id}
	done
done
