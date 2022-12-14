all: other_extension cutlass-conv cutlass-linear pool cuda-q cudnn-conv pre last
simu: other_extension cutlass-linear pre last

pre:
	python3 target_conf_gen.py --funct conf
	cd 3rd_configs; bash mv_file.sh; cd ..;
	cd 3rd_party/transformers; pip3 install .; cd ../..;
	cd 3rd_party/dpro; pip3 install .; cd ../..;

cutlass-conv:
	cd pytorch/cutlass-conv; pip3 install .

cudnn-conv:
	cd pytorch/cudnn_conv; pip3 install .

cuda-q:
	cd pytorch/quantization; pip3 install .

cutlass-linear:
	cd pytorch/cutlass-linear; pip3 install .

pool:
	cd pytorch/int8pool-extension; pip3 install .

ssdc:
	cd pytorch/ssdc; pip3 install .

other_extension:
	cd pytorch/other_extension; pip3 install .


clean:
	pip3 uninstall cutlassconv_cuda cutlasslinear_cuda int8pool actnn cuda-q
	cd pytorch/cutlass-conv; rm -rf build; rm -rf cutlassconv_cuda.egg-info;
	cd pytorch/cudnn_conv; rm -rf build; rm -rf cudnn_conv_cuda.egg-info;
	cd pytorch/quantization; rm -rf build; rm -rf cuda_quantization.egg-info;
	cd pytorch/other_extension; rm -rf build; rm -rf *.egg-info;

clean-conv:
	rm -rf pytorch/cutlass-conv/build; rm -rf qsync_niti_based/pytorch/cutlass-conv/cutlassconv_cuda.egg-info;
	rm -rf pytorch/cutlass-conv/cutlassconv_cuda.egg-info;

clean-li:
	rm -rf pytorch/cutlass-linear/build; rm -rf qsync_niti_based/pytorch/cutlass-linear/cutlasslinear_cuda.egg-info;
	rm -rf pytorch/cutlass-linear/cutlasslinear_cuda.egg-info;

last:
	python3 target_conf_gen.py --funct qopt
	python3 target_conf_gen.py --funct tops --ops \
 nn.Conv2d nn.ReLU nn.BatchNorm2d nn.LayerNorm nn.ReLU nn.Linear nn.Embedding nn.MaxPool2d nn.Softmax nn.Dropout TransGELU
