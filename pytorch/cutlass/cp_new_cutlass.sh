CUTLASS_DIR='/home/ubuntu/cutlass'
rm -rf include
rm -rf util
cp -r $CUTLASS_DIR/include .
cp -r $CUTLASS_DIR/tools/util .