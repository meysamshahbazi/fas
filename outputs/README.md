output dir for saving result of each run

output/
	path = namespace.dataset + '_' + namespace.model + '_' + namespace.criterion + '_' + namespace.optimizer
	checkpoints/ ep_nb.pt
	eer_figs/ep_nb.png
	loss_fig
	logs.txt 
