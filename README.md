# autoVB

## install
```
conda create -n autovb-test python=3.11 -y
conda activate autovb-test
conda install pyscf -c conda-forge
conda install mokit -c mokit -c conda-forge
pip install autovb-0.1.0-py3-none-any.whl
```

run 
```
pip install -e . 
```
to install

包括 autovb-nbo autovb-xmi 两个命令可以直接调用