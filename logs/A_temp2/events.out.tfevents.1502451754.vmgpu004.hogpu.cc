       �K"	  �
fc�Abrain.Event:2J��g3�     �̘.	�2�
fc�A"��
^
dataPlaceholder*
shape: *
dtype0*/
_output_shapes
:���������
W
labelPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
h
conv2d_1_inputPlaceholder*/
_output_shapes
:���������*
shape: *
dtype0
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *�x�*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
valueB
 *�x=*
_output_shapes
: *
dtype0
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2�Ѿ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0
�
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:@*
T0
�
conv2d_1/kernel
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
[
conv2d_1/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
y
conv2d_1/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
`
conv2d_2/random_uniform/minConst*
valueB
 *�\1�*
dtype0*
_output_shapes
: 
`
conv2d_2/random_uniform/maxConst*
valueB
 *�\1=*
dtype0*
_output_shapes
: 
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2��%*
T0*
seed���)*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 
�
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:@@
�
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0
�
conv2d_2/kernel
VariableV2*&
_output_shapes
:@@*
	container *
shape:@@*
dtype0*
shared_name 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
[
conv2d_2/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
s
conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
data_formatNHWC*
strides

�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
a
dropout_1/keras_learning_phasePlaceholder*
shape: *
dtype0
*
_output_shapes
:
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
:
e
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2��+
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*/
_output_shapes
:���������@*
T0
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
flatten_1/stack/0Const*
valueB :
���������*
_output_shapes
: *
dtype0
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�3z�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
T0*
seed���)*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*!
_output_shapes
:���
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*!
_output_shapes
:���*
	container *
shape:���*
dtype0*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
~
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
z
dense_1/bias
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
]
activation_3/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
�
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
T0
*
_output_shapes
:
]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
:*
T0

e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
valueB"�   
   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *̈́U�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *̈́U>*
_output_shapes
: *
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	�
*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�

�
dense_2/kernel
VariableV2*
_output_shapes
:	�
*
	container *
shape:	�
*
dtype0*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

Z
dense_2/ConstConst*
valueB
*    *
_output_shapes
:
*
dtype0
x
dense_2/bias
VariableV2*
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0
�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
�
'sequential_1/conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:���������@*
use_cudnn_on_gpu(
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
_output_shapes
:*
T0

r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2ԗ�*
T0*
seed���)*
dtype0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
2sequential_1/dropout_1/cond/dropout/random_uniformAdd6sequential_1/dropout_1/cond/dropout/random_uniform/mul6sequential_1/dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
�
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
t
*sequential_1/flatten_1/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes
:*
ellipsis_mask *

begin_mask 
f
sequential_1/flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
sequential_1/flatten_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
�
sequential_1/dense_1/MatMulMatMulsequential_1/flatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
T0
*
_output_shapes
:
w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��>*
T0*
seed���)*
dtype0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
T0*
N**
_output_shapes
:����������: 
�
sequential_1/dense_2/MatMulMatMul!sequential_1/dropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
[
num_inst/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
l
num_inst
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
_class
loc:@num_inst*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
a
num_inst/readIdentitynum_inst*
T0*
_class
loc:@num_inst*
_output_shapes
: 
^
num_correct/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
o
num_correct
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
num_correct/readIdentitynum_correct*
_class
loc:@num_correct*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
e
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
S
ToFloatCastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
X
SumSumToFloatConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
Const_1Const*
valueB
 *  �B*
dtype0*
_output_shapes
: 
z
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_class
loc:@num_inst*
_output_shapes
: 
~
AssignAdd_1	AssignAddnum_correctSum*
use_locking( *
T0*
_class
loc:@num_correct*
_output_shapes
: 
L
Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
AssignAssignnum_instConst_2*
use_locking(*
T0*
_class
loc:@num_inst*
validate_shape(*
_output_shapes
: 
L
Const_3Const*
valueB
 *    *
_output_shapes
: *
dtype0
�
Assign_1Assignnum_correctConst_3*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
J
add/yConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
A
addAddnum_inst/readadd/y*
_output_shapes
: *
T0
F
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
value	B :*
_output_shapes
: *
dtype0
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
T0*
out_type0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
T0*

axis *
N*
_output_shapes
:
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
Tshape0*0
_output_shapes
:������������������*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*

axis *
_output_shapes
:*
T0*
N
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
Tshape0*#
_output_shapes
:���������*
T0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes
: *
T0
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
T0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
N
PlaceholderPlaceholder*
_output_shapes
:*
shape: *
dtype0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
_output_shapes
: *
T0
�
6gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/Greatergradients/Fill:gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
�
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
�
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: 
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: 
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
_output_shapes
: *
dtype0
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
out_type0*
_output_shapes
:*
T0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������*
T0
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*
T0*#
_output_shapes
:���������
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:������������������
�
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
_
gradients/div_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_1_grad/RealDivRealDiv9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapediv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������

�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������
*
T0
�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
�
1gradients/sequential_1/dense_2/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*(
_output_shapes
:����������*
transpose_a( *
T0
�
3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1MatMul!sequential_1/dropout_2/cond/MergeDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	�
*
transpose_a(*
T0
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�
*
T0
�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������
�
Agradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*(
_output_shapes
:����������*
T0
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:����������: 
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*(
_output_shapes
:����������*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Lgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
T0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*(
_output_shapes
:����������
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
T0*
N**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*(
_output_shapes
:����������*
T0*
N
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������*
T0
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
1gradients/sequential_1/dense_1/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*)
_output_shapes
:�����������*
transpose_a( 
�
3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1MatMulsequential_1/flatten_1/ReshapeDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *(
_output_shapes
:����������*
transpose_a(*
T0
�
;gradients/sequential_1/dense_1/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_1/MatMul_grad/MatMul4^gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*)
_output_shapes
:�����������
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*!
_output_shapes
:���*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@*
T0
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*/
_output_shapes
:���������@
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
T0*
N*1
_output_shapes
:���������@: 
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
out_type0*
_output_shapes
:*
T0
�
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
Tshape0*/
_output_shapes
:���������@*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:���������@
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*/
_output_shapes
:���������@
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
T0*
N*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
N*/
_output_shapes
:���������@
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
T0
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:@@*
data_formatNHWC*
strides

�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������@
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@*
T0
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
�
=gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_1/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
T0
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
strides

�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
dtype0*
_output_shapes
:
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@
�
beta1_power/initial_valueConst*
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta1_power
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
shape: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
n
beta1_power/readIdentitybeta1_power*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
beta2_power/initial_valueConst*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
j
zerosConst*%
valueB@*    *&
_output_shapes
:@*
dtype0
�
conv2d_1/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:@*
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
l
zeros_1Const*%
valueB@*    *&
_output_shapes
:@*
dtype0
�
conv2d_1/kernel/Adam_1
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:@*
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
T
zeros_2Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
T
zeros_3Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam_1
VariableV2* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
l
zeros_4Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0
�
conv2d_2/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
l
zeros_5Const*%
valueB@@*    *&
_output_shapes
:@@*
dtype0
�
conv2d_2/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
shape:@@*
shared_name 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
T
zeros_6Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_2/bias/Adam
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
T
zeros_7Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_2/bias/Adam_1
VariableV2*
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
�
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
b
zeros_8Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape:���*
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
b
zeros_9Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam_1
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape:���*
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
W
zeros_10Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
dense_1/bias/Adam
VariableV2*
shape:�*
_output_shapes	
:�*
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container 
�
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0
W
zeros_11Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/bias/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
shape:�*
shared_name 
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0
_
zeros_12Const*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
shape:	�
*
shared_name 
�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
validate_shape(*
use_locking(
�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
_
zeros_13Const*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam_1
VariableV2*
shared_name *!
_class
loc:@dense_2/kernel*
	container *
shape:	�
*
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
validate_shape(*
use_locking(
�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

U
zeros_14Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
:
*
shape:
*
shared_name 
�
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

U
zeros_15Const*
valueB
*    *
_output_shapes
:
*
dtype0
�
dense_2/bias/Adam_1
VariableV2*
_class
loc:@dense_2/bias*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
use_locking( 
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
use_locking( 
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"��|��     ����	{.�
fc�AJ��
�%�$
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
5
PreventGradient

input"T
output"T"	
Ttype
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��
^
dataPlaceholder*/
_output_shapes
:���������*
shape: *
dtype0
W
labelPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������

h
conv2d_1_inputPlaceholder*
shape: *
dtype0*/
_output_shapes
:���������
v
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
valueB
 *�x�*
_output_shapes
: *
dtype0
`
conv2d_1/random_uniform/maxConst*
valueB
 *�x=*
_output_shapes
: *
dtype0
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2�Ѿ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
_output_shapes
: *
T0
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
:@
�
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
:@
�
conv2d_1/kernel
VariableV2*&
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
[
conv2d_1/ConstConst*
valueB@*    *
_output_shapes
:@*
dtype0
y
conv2d_1/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
dtype0*
shared_name 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
s
conv2d_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
v
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*
seed���)*
T0*&
_output_shapes
:@@*
seed2��%
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 
�
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*&
_output_shapes
:@@*
T0
�
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@
�
conv2d_2/kernel
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
[
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_2/bias
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
s
conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
dtype0
*
shape: *
_output_shapes
:
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
_output_shapes
:*
T0

e
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2��+
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
�
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
_output_shapes
:*
end_mask*
T0*
Index0*
shrink_axis_mask *
new_axis_mask 
Y
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB" d  �   
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�3z�
_
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed���)*
T0*!
_output_shapes
:���*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*!
_output_shapes
:���*
T0
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
~
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
_output_shapes	
:�*
dtype0*
valueB�*    
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
data_formatNHWC*
T0
]
activation_3/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
�
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
_
dropout_2/cond/switch_tIdentitydropout_2/cond/Switch:1*
_output_shapes
:*
T0

]
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
_output_shapes
:*
T0

e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2���
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*(
_output_shapes
:����������*
T0
�
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
t
dropout_2/cond/dropout/FloorFloordropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/divRealDivdropout_2/cond/mul dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   
   
_
dense_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U�
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U>
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2���*
dtype0*
T0*
seed���)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	�
*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�
*
T0
�
dense_2/kernel
VariableV2*
shared_name *
dtype0*
shape:	�
*
_output_shapes
:	�
*
	container 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
|
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
Z
dense_2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_2/bias
VariableV2*
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
�
'sequential_1/conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2ԗ�
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
�
2sequential_1/dropout_1/cond/dropout/random_uniformAdd6sequential_1/dropout_1/cond/dropout/random_uniform/mul6sequential_1/dropout_1/cond/dropout/random_uniform/min*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
�
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
t
*sequential_1/flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
f
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
sequential_1/flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 
�
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
T0
�
sequential_1/dense_1/MatMulMatMulsequential_1/flatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*
T0*(
_output_shapes
:����������
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_2/cond/switch_tIdentity$sequential_1/dropout_2/cond/Switch:1*
_output_shapes
:*
T0

w
$sequential_1/dropout_2/cond/switch_fIdentity"sequential_1/dropout_2/cond/Switch*
_output_shapes
:*
T0

r
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
T0
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2��>
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
�
sequential_1/dense_2/MatMulMatMul!sequential_1/dropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
data_formatNHWC*
T0
b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
[
num_inst/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
num_inst
VariableV2*
shared_name *
dtype0*
shape: *
_output_shapes
: *
	container 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
a
num_inst/readIdentitynum_inst*
_output_shapes
: *
_class
loc:@num_inst*
T0
^
num_correct/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
o
num_correct
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_correct
j
num_correct/readIdentitynum_correct*
T0*
_output_shapes
: *
_class
loc:@num_correct
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
g
ArgMax_1ArgMaxlabelArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
S
ToFloatCastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
X
SumSumToFloatConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
_output_shapes
: *
_class
loc:@num_inst*
T0*
use_locking( 
~
AssignAdd_1	AssignAddnum_correctSum*
_output_shapes
: *
_class
loc:@num_correct*
T0*
use_locking( 
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
�
AssignAssignnum_instConst_2*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
L
Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
�
Assign_1Assignnum_correctConst_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_correct
J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.
A
addAddnum_inst/readadd/y*
_output_shapes
: *
T0
F
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
L
div_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
_output_shapes
:*
out_type0*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
_output_shapes
:*
N*

axis *
T0
o
%softmax_cross_entropy_loss/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
h
&softmax_cross_entropy_loss/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*0
_output_shapes
:������������������*
Tshape0*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:

,softmax_cross_entropy_loss/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
j
(softmax_cross_entropy_loss/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
d
"softmax_cross_entropy_loss/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
_output_shapes
:*
N*

axis *
T0
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
T0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
_output_shapes
: *
T0
�
6gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/Greatergradients/Fill:gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
T0*
_output_shapes
: 
�
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
�
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
_output_shapes
: *
Tshape0*
T0
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*
_output_shapes
: *H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
T0
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
T0
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
dtype0*
_output_shapes
: *
valueB 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
_output_shapes
:*
out_type0*
T0
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
T0
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
_output_shapes
:*
out_type0*
T0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*
T0*#
_output_shapes
:���������
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
T0*
_output_shapes
:*
out_type0
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:������������������*
T0
�
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
_
gradients/div_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/div_1_grad/RealDivRealDiv9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapediv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape*
T0
�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
_output_shapes
: *1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape*
T0
�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:
*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
1gradients/sequential_1/dense_2/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1MatMul!sequential_1/dropout_2/cond/MergeDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	�
*
transpose_a(*
T0
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Agradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
_output_shapes
:*
out_type0
Z
gradients/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
N*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:����������*
Tshape0*
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Lgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
N*
T0**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
N*
T0*(
_output_shapes
:����������*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
T0
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
1gradients/sequential_1/dense_1/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*)
_output_shapes
:�����������*
transpose_a( *
T0
�
3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1MatMulsequential_1/flatten_1/ReshapeDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a(
�
;gradients/sequential_1/dense_1/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_1/MatMul_grad/MatMul4^gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Agradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_depsNoOp;^gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad
�
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
T0
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
e
gradients/Shape_3Shapegradients/Switch_2:1*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*/
_output_shapes
:���������@
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*1
_output_shapes
:���������@: *
N*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
�
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*/
_output_shapes
:���������@*
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*
T0*/
_output_shapes
:���������@
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*
T0
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
c
gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������@*
T0
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*/
_output_shapes
:���������@*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
T0
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
_output_shapes
:*
out_type0
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:@@
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_1/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:@
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0
�
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel
�
beta1_power
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta1_power/readIdentitybeta1_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel
�
beta2_power
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
j
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    
�
conv2d_1/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
l
zeros_1Const*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *&
_output_shapes
:@*
shape:@
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
T
zeros_2Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_1/bias/Adam
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
shared_name *
_output_shapes
:@*
shape:@
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
T
zeros_3Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_1/bias/Adam_1
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
shared_name *
_output_shapes
:@*
shape:@
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
l
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam
VariableV2*&
_output_shapes
:@@*
dtype0*
shape:@@*
	container *"
_class
loc:@conv2d_2/kernel*
shared_name 
�
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
l
zeros_5Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_2/kernel*
shared_name *&
_output_shapes
:@@*
shape:@@
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
T
zeros_6Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam
VariableV2*
	container *
dtype0* 
_class
loc:@conv2d_2/bias*
shared_name *
_output_shapes
:@*
shape:@
�
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
T
zeros_7Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam_1
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_2/bias*
shared_name 
�
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
b
zeros_8Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
�
dense_1/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
b
zeros_9Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
�
dense_1/kernel/Adam_1
VariableV2*!
_output_shapes
:���*
dtype0*
shape:���*
	container *!
_class
loc:@dense_1/kernel*
shared_name 
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
W
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
shared_name *
_output_shapes	
:�*
shape:�
�
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
W
zeros_11Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
dense_1/bias/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *
_class
loc:@dense_1/bias*
shared_name 
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
_
zeros_12Const*
dtype0*
_output_shapes
:	�
*
valueB	�
*    
�
dense_2/kernel/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@dense_2/kernel*
shared_name *
_output_shapes
:	�
*
shape:	�

�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
_
zeros_13Const*
dtype0*
_output_shapes
:	�
*
valueB	�
*    
�
dense_2/kernel/Adam_1
VariableV2*
_output_shapes
:	�
*
dtype0*
shape:	�
*
	container *!
_class
loc:@dense_2/kernel*
shared_name 
�
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
_output_shapes
:	�
*
validate_shape(*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
U
zeros_14Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *
_class
loc:@dense_2/bias*
shared_name 
�
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0
U
zeros_15Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
_class
loc:@dense_2/bias
�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias

dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0*
use_locking( 
�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking( 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""
train_op

Adam"�&
cond_context�&�&
�
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *�
activation_2/Relu:0
dropout_1/cond/dropout/Floor:0
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/add:0
dropout_1/cond/dropout/div:0
"dropout_1/cond/dropout/keep_prob:0
dropout_1/cond/dropout/mul:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/mul/Switch:1
dropout_1/cond/mul/y:0
dropout_1/cond/mul:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:02
activation_2/Relu:0dropout_1/cond/mul/Switch:1
�
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*�
activation_2/Relu:0
dropout_1/cond/Switch_1:0
dropout_1/cond/Switch_1:1
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:00
activation_2/Relu:0dropout_1/cond/Switch_1:0
�
dropout_2/cond/cond_textdropout_2/cond/pred_id:0dropout_2/cond/switch_t:0 *�
activation_3/Relu:0
dropout_2/cond/dropout/Floor:0
dropout_2/cond/dropout/Shape:0
dropout_2/cond/dropout/add:0
dropout_2/cond/dropout/div:0
"dropout_2/cond/dropout/keep_prob:0
dropout_2/cond/dropout/mul:0
5dropout_2/cond/dropout/random_uniform/RandomUniform:0
+dropout_2/cond/dropout/random_uniform/max:0
+dropout_2/cond/dropout/random_uniform/min:0
+dropout_2/cond/dropout/random_uniform/mul:0
+dropout_2/cond/dropout/random_uniform/sub:0
'dropout_2/cond/dropout/random_uniform:0
dropout_2/cond/mul/Switch:1
dropout_2/cond/mul/y:0
dropout_2/cond/mul:0
dropout_2/cond/pred_id:0
dropout_2/cond/switch_t:02
activation_3/Relu:0dropout_2/cond/mul/Switch:1
�
dropout_2/cond/cond_text_1dropout_2/cond/pred_id:0dropout_2/cond/switch_f:0*�
activation_3/Relu:0
dropout_2/cond/Switch_1:0
dropout_2/cond/Switch_1:1
dropout_2/cond/pred_id:0
dropout_2/cond/switch_f:00
activation_3/Relu:0dropout_2/cond/Switch_1:0
�
%sequential_1/dropout_1/cond/cond_text%sequential_1/dropout_1/cond/pred_id:0&sequential_1/dropout_1/cond/switch_t:0 *�
 sequential_1/activation_2/Relu:0
+sequential_1/dropout_1/cond/dropout/Floor:0
+sequential_1/dropout_1/cond/dropout/Shape:0
)sequential_1/dropout_1/cond/dropout/add:0
)sequential_1/dropout_1/cond/dropout/div:0
/sequential_1/dropout_1/cond/dropout/keep_prob:0
)sequential_1/dropout_1/cond/dropout/mul:0
Bsequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform:0
8sequential_1/dropout_1/cond/dropout/random_uniform/max:0
8sequential_1/dropout_1/cond/dropout/random_uniform/min:0
8sequential_1/dropout_1/cond/dropout/random_uniform/mul:0
8sequential_1/dropout_1/cond/dropout/random_uniform/sub:0
4sequential_1/dropout_1/cond/dropout/random_uniform:0
(sequential_1/dropout_1/cond/mul/Switch:1
#sequential_1/dropout_1/cond/mul/y:0
!sequential_1/dropout_1/cond/mul:0
%sequential_1/dropout_1/cond/pred_id:0
&sequential_1/dropout_1/cond/switch_t:0L
 sequential_1/activation_2/Relu:0(sequential_1/dropout_1/cond/mul/Switch:1
�
'sequential_1/dropout_1/cond/cond_text_1%sequential_1/dropout_1/cond/pred_id:0&sequential_1/dropout_1/cond/switch_f:0*�
 sequential_1/activation_2/Relu:0
&sequential_1/dropout_1/cond/Switch_1:0
&sequential_1/dropout_1/cond/Switch_1:1
%sequential_1/dropout_1/cond/pred_id:0
&sequential_1/dropout_1/cond/switch_f:0J
 sequential_1/activation_2/Relu:0&sequential_1/dropout_1/cond/Switch_1:0
�
%sequential_1/dropout_2/cond/cond_text%sequential_1/dropout_2/cond/pred_id:0&sequential_1/dropout_2/cond/switch_t:0 *�
 sequential_1/activation_3/Relu:0
+sequential_1/dropout_2/cond/dropout/Floor:0
+sequential_1/dropout_2/cond/dropout/Shape:0
)sequential_1/dropout_2/cond/dropout/add:0
)sequential_1/dropout_2/cond/dropout/div:0
/sequential_1/dropout_2/cond/dropout/keep_prob:0
)sequential_1/dropout_2/cond/dropout/mul:0
Bsequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform:0
8sequential_1/dropout_2/cond/dropout/random_uniform/max:0
8sequential_1/dropout_2/cond/dropout/random_uniform/min:0
8sequential_1/dropout_2/cond/dropout/random_uniform/mul:0
8sequential_1/dropout_2/cond/dropout/random_uniform/sub:0
4sequential_1/dropout_2/cond/dropout/random_uniform:0
(sequential_1/dropout_2/cond/mul/Switch:1
#sequential_1/dropout_2/cond/mul/y:0
!sequential_1/dropout_2/cond/mul:0
%sequential_1/dropout_2/cond/pred_id:0
&sequential_1/dropout_2/cond/switch_t:0L
 sequential_1/activation_3/Relu:0(sequential_1/dropout_2/cond/mul/Switch:1
�
'sequential_1/dropout_2/cond/cond_text_1%sequential_1/dropout_2/cond/pred_id:0&sequential_1/dropout_2/cond/switch_f:0*�
 sequential_1/activation_3/Relu:0
&sequential_1/dropout_2/cond/Switch_1:0
&sequential_1/dropout_2/cond/Switch_1:1
%sequential_1/dropout_2/cond/pred_id:0
&sequential_1/dropout_2/cond/switch_f:0J
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"
	summaries


loss:0"�
trainable_variables��
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"�
	variables��
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
C
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:0
=
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
.

num_inst:0num_inst/Assignnum_inst/read:0
7
num_correct:0num_correct/Assignnum_correct/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
R
conv2d_1/kernel/Adam:0conv2d_1/kernel/Adam/Assignconv2d_1/kernel/Adam/read:0
X
conv2d_1/kernel/Adam_1:0conv2d_1/kernel/Adam_1/Assignconv2d_1/kernel/Adam_1/read:0
L
conv2d_1/bias/Adam:0conv2d_1/bias/Adam/Assignconv2d_1/bias/Adam/read:0
R
conv2d_1/bias/Adam_1:0conv2d_1/bias/Adam_1/Assignconv2d_1/bias/Adam_1/read:0
R
conv2d_2/kernel/Adam:0conv2d_2/kernel/Adam/Assignconv2d_2/kernel/Adam/read:0
X
conv2d_2/kernel/Adam_1:0conv2d_2/kernel/Adam_1/Assignconv2d_2/kernel/Adam_1/read:0
L
conv2d_2/bias/Adam:0conv2d_2/bias/Adam/Assignconv2d_2/bias/Adam/read:0
R
conv2d_2/bias/Adam_1:0conv2d_2/bias/Adam_1/Assignconv2d_2/bias/Adam_1/read:0
O
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:0
U
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:0
I
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:0
O
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:0
O
dense_2/kernel/Adam:0dense_2/kernel/Adam/Assigndense_2/kernel/Adam/read:0
U
dense_2/kernel/Adam_1:0dense_2/kernel/Adam_1/Assigndense_2/kernel/Adam_1/read:0
I
dense_2/bias/Adam:0dense_2/bias/Adam/Assigndense_2/bias/Adam/read:0
O
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0Ӻ-E       ��-	/��
fc�A*

loss/]@�,��       ��-	��
fc�A*

lossw@ԶH�       ��-	,e�
fc�A*

lossP�@m��       ��-	b�
fc�A*

losss��?7/��       ��-	4��
fc�A*

lossZ'�?�Ww        ��-	�c�
fc�A*

loss��?���       ��-	�
fc�A*

loss���?�pL       ��-	���
fc�A*

lossq��?UC/$       ��-	�S�
fc�A	*

loss��? sf       ��-	���
fc�A
*

loss,�u?'��}       ��-	R��
fc�A*

loss�W?K�P       ��-	�w�
fc�A*

loss�1'?}K.       ��-	��
fc�A*

lossq[?���       ��-	Ul�
fc�A*

loss�J?��?       ��-	p�
fc�A*

losss�a?�kVG       ��-	H��
fc�A*

loss.��>�t�       ��-	�B�
fc�A*

loss3?���C       ��-	���
fc�A*

loss��?�hY�       ��-	;��
fc�A*

loss�+3?�{��       ��-	�:�
fc�A*

lossO%2?�%       ��-	]��
fc�A*

loss��-?c'iu       ��-	���
fc�A*

loss�%?,e�       ��-	x)�
fc�A*

loss�JX?�#��       ��-	���
fc�A*

loss�.?�{�N       ��-	qv�
fc�A*

loss7��>��uq       ��-	T�
fc�A*

loss��?~!}       ��-	���
fc�A*

loss���>�:q       ��-	�_�
fc�A*

loss��?@G��       ��-	�	�
fc�A*

loss�?�z��       ��-	��
fc�A*

loss�2?��#�       ��-	G�
fc�A*

lossH��>:�0>       ��-	2��
fc�A *

loss��>AT~�       ��-	���
fc�A!*

loss%�2?�/�!       ��-	��
fc�A"*

loss��>�5	       ��-	��
fc�A#*

lossW>?G��       ��-	K9fc�A$*

loss;כ>�J?0       ��-	��fc�A%*

loss_�/?�=~       ��-	zfc�A&*

loss��1?��@S       ��-	�lfc�A'*

loss��?0&�       ��-	=fc�A(*

loss�]?��*�       ��-	عfc�A)*

loss��>���       ��-	�zfc�A**

loss$-�>�`z�       ��-	t&fc�A+*

loss�b�>�3       ��-	��fc�A,*

loss��>�D       ��-	�ufc�A-*

loss
�?_�S!       ��-	fc�A.*

lossAM�>����       ��-	�fc�A/*

loss��>F���       ��-	;T	fc�A0*

lossz��>�]�       ��-	��	fc�A1*

lossG�>�޺�       ��-	c�
fc�A2*

loss���>i��k       ��-	:;fc�A3*

loss8�>|��u       ��-	|�fc�A4*

loss�q�>+���       ��-	�fc�A5*

loss.M?��a       ��-	n1fc�A6*

loss+��>}�8�       ��-	"�fc�A7*

lossa�I>J��Q       ��-	]�fc�A8*

loss�dG>=���       ��-	 8fc�A9*

loss�N>Z^��       ��-	��fc�A:*

lossѥ?�=9�       ��-	�tfc�A;*

loss��>�V1       ��-	fc�A<*

losso�>*g�       ��-	��fc�A=*

loss�<>�S�m       ��-	�efc�A>*

loss��>X��       ��-	�fc�A?*

loss�T�>��y       ��-	�fc�A@*

losseJ�>��       ��-	�Zfc�AA*

loss��>B�b�       ��-	�fc�AB*

loss}u�>#I       ��-	��fc�AC*

loss���>,I[S       ��-	y]fc�AD*

lossl��>�+�       ��-	3�fc�AE*

loss�V�>� �z       ��-	i�fc�AF*

loss���>��I�       ��-	�Xfc�AG*

loss���>ˁ��       ��-	L�fc�AH*

loss�ˊ>���	       ��-	��fc�AI*

loss�s�>���       ��-	�Bfc�AJ*

loss�%�>h�1H       ��-	%�fc�AK*

loss�X�>�M�0       ��-	U�fc�AL*

loss�f�>|kW�       ��-	*fc�AM*

loss�	?ĝ=#       ��-	��fc�AN*

loss�_>�Z�,       ��-	��fc�AO*

losspS>��>1       ��-	hAfc�AP*

lossD�>���N       ��-	L�fc�AQ*

loss��><6�       ��-	9fc�AR*

loss�=�>T~&Z       ��-	�) fc�AS*

loss?�A>x{n2       ��-	"� fc�AT*

lossno�>�?h2       ��-	�^!fc�AU*

loss�)�>m��       ��-	��!fc�AV*

loss! j>��f�       ��-	O�"fc�AW*

loss�N�>uW       ��-	�+#fc�AX*

lossIO�>�)�o       ��-	 �#fc�AY*

loss�g>(�[�       ��-	an$fc�AZ*

loss��>	'�5       ��-	c%fc�A[*

loss�IC>��r�       ��-	��%fc�A\*

loss���>A`�       ��-	�&fc�A]*

loss*1�>{��       ��-	�'fc�A^*

loss�v�>�md�       ��-	ٱ'fc�A_*

loss�92>˾��       ��-	��(fc�A`*

loss	}�>���       ��-	��)fc�Aa*

loss*r�>Ϥ��       ��-	^*fc�Ab*

loss�D�>�UF�       ��-	a7+fc�Ac*

lossҪ�>Z~ڡ       ��-	4,fc�Ad*

loss��+>�8�[       ��-	)�,fc�Ae*

loss��=����       ��-	��-fc�Af*

loss�+�>����       ��-	�=.fc�Ag*

loss�>>+#�       ��-	�.fc�Ah*

lossXr>�8�       ��-	��/fc�Ai*

loss{I0>O�|�       ��-	800fc�Aj*

loss-\�=����       ��-	��0fc�Ak*

loss��=� ��       ��-	�^1fc�Al*

lossa��>�)`       ��-	��1fc�Am*

lossMߙ>U��9       ��-	i�2fc�An*

loss.}�>9��]       ��-	-C3fc�Ao*

loss>��ǻ       ��-	w�3fc�Ap*

loss��#>C���       ��-	n4fc�Aq*

loss��=��k       ��-	�5fc�Ar*

loss��%>,�       ��-	�5fc�As*

loss��>h�"�       ��-	e46fc�At*

lossx��=׶�C       ��-	��6fc�Au*

loss�Nw>����       ��-	�Y7fc�Av*

loss]>h2q�       ��-	��7fc�Aw*

lossJwn>s^%�       ��-	r�8fc�Ax*

loss!�J>��k       ��-	�#9fc�Ay*

loss�f_>���p       ��-	��9fc�Az*

loss�4>���       ��-	[^:fc�A{*

loss��m>��S@       ��-	�;fc�A|*

lossIa�=�R       ��-	�;fc�A}*

loss�Y>P*�c       ��-	TS<fc�A~*

lossʗ�>|'t�       ��-	_�<fc�A*

lossJ�?>*?m       �	�=fc�A�*

loss�r>�W4       �	wI>fc�A�*

loss��>�M^a       �	��>fc�A�*

loss�Z>D�m       �	��?fc�A�*

losse�8>ae       �	�Afc�A�*

losss��=���       �	�Afc�A�*

loss��>�r�       �	2�Bfc�A�*

loss�k>�U�       �	*9Cfc�A�*

loss>D�v�       �	��Cfc�A�*

lossJ�Z>��)       �	tDfc�A�*

loss7�6>)�C�       �	�Efc�A�*

loss-l�=q�o        �	��Efc�A�*

loss4��=����       �	DOFfc�A�*

lossn��=o��       �	��Ffc�A�*

loss/�/>TY�       �	m�Gfc�A�*

loss�>�Z�N       �	N+Hfc�A�*

loss�
�>oG;�       �	Y�Hfc�A�*

lossʂ>(��       �	\Ifc�A�*

loss��=�w>       �	Jfc�A�*

loss�X�>S�O�       �	�Jfc�A�*

loss�
�=#\��       �	b.Kfc�A�*

loss��>�       �	��Kfc�A�*

lossf$>���       �	 oLfc�A�*

loss�d>��O�       �	�Mfc�A�*

lossI�>��       �	T�Mfc�A�*

loss&|A>� �)       �	�HNfc�A�*

losscS�>>*Ro       �	.�Nfc�A�*

loss��I>+��       �	g}Ofc�A�*

loss�0�=zЅ�       �	/Pfc�A�*

loss��=r(��       �	)�Pfc�A�*

loss��!>��7@       �	�eQfc�A�*

loss\�|>�<       �	hRfc�A�*

lossY��>���A       �	D�Rfc�A�*

lossib>���~       �	�ASfc�A�*

loss�fM>��;k       �	j�Sfc�A�*

lossI�A>R��       �	�}Tfc�A�*

lossn��=���l       �	]Ufc�A�*

loss8�>W�Z       �	b�Ufc�A�*

lossxK>$dI       �	qVVfc�A�*

loss�$_>��^$       �	�Vfc�A�*

lossLS�=���h       �	C�Wfc�A�*

loss��>��c       �	�$Xfc�A�*

loss(�>����       �	��Xfc�A�*

loss�->���a       �	[\Yfc�A�*

loss���=�S5M       �	� Zfc�A�*

loss{�>����       �	�Zfc�A�*

loss�t�>��4       �	�?[fc�A�*

lossn̋=͍Z�       �	Z�[fc�A�*

lossL�>-k�m       �	�m\fc�A�*

lossi�> 	�p       �	� ]fc�A�*

lossw�>�8��       �	��]fc�A�*

loss�I,>f�h�       �	�-^fc�A�*

lossFF�=�E\a       �	ܽ^fc�A�*

loss��=�.g       �	~S_fc�A�*

loss`��>h�2       �	m�_fc�A�*

loss|u>
�%w       �	`fc�A�*

loss� �=,�1        �	�afc�A�*

loss$�->�g�J       �		�afc�A�*

loss�1>�,�       �	�Cbfc�A�*

loss?>\��       �	��bfc�A�*

loss��y>�B9       �	ncfc�A�*

lossD�$>���       �	Ndfc�A�*

loss�n>3��       �	=�dfc�A�*

loss3f�>�T�!       �	eefc�A�*

lossԫD>7۞       �	cffc�A�*

loss�|>�N       �	>�ffc�A�*

loss��c>���       �	�[gfc�A�*

loss���=k�O       �	��gfc�A�*

loss/4P>0�+�       �	��hfc�A�*

loss�~>���       �	�Iifc�A�*

loss/�*>|��       �	(jfc�A�*

loss��>�z(�       �	'kfc�A�*

loss@=6>�s�y       �	�lfc�A�*

lossi><�3�       �	Rmfc�A�*

lossH��=��%�       �	�mfc�A�*

loss&U�=A�       �	��nfc�A�*

loss��>>��;       �	�-ofc�A�*

loss8c�=�?>:       �	G�ofc�A�*

loss͹�=�_�       �	dpfc�A�*

loss��#>"�E       �	w�pfc�A�*

loss�#�=5��       �	��qfc�A�*

lossܞ�=�$�        �	brfc�A�*

loss?�>��       �	J
sfc�A�*

loss�I>��       �	s�sfc�A�*

loss3B�=�2�       �	3mtfc�A�*

loss,��=ǉ�       �	;7ufc�A�*

loss�Z�=/ߡ�       �	��ufc�A�*

loss�:)>#a       �	%�vfc�A�*

lossTb >4X�       �	wNwfc�A�*

loss�~>HU�       �	C�wfc�A�*

loss��N>��8�       �	�xfc�A�*

loss�E{>'ּ       �	)!yfc�A�*

loss�5�>�1d�       �	�yfc�A�*

loss�Z==�O�D       �	yZzfc�A�*

loss��>�Zf       �	��zfc�A�*

loss'�>�*�G       �	u�{fc�A�*

loss�a�>�b}�       �	�9|fc�A�*

lossЮ�=����       �	��|fc�A�*

loss-��=6"��       �	k�}fc�A�*

loss1c!>p��i       �	�~fc�A�*

loss�V$>��k       �	{�~fc�A�*

loss�'�=��c       �	�Xfc�A�*

loss�+>��w       �	M�fc�A�*

lossߨv=7�V       �	S�fc�A�*

loss��=�Y�       �	~��fc�A�*

loss�K->�t�       �	)$�fc�A�*

loss�
>�k�       �	���fc�A�*

lossۭ5>Z�o�       �	�Q�fc�A�*

lossL�>OxW       �	�fc�A�*

lossT:>@�;       �	[z�fc�A�*

loss�,>���       �	��fc�A�*

loss3�&>v��       �	l��fc�A�*

lossXů=��ad       �	R�fc�A�*

loss�(H>R2'       �	�fc�A�*

loss|E�=B��*       �	���fc�A�*

loss�@r>bO�       �	P�fc�A�*

loss#�@>��$�       �	���fc�A�*

loss|I�>�5�       �	uZ�fc�A�*

loss��@>~�       �	���fc�A�*

loss3F�=�6       �	j��fc�A�*

loss�9v=^۳�       �	�(�fc�A�*

lossgS >���       �	*ǋfc�A�*

loss�80>��       �	�p�fc�A�*

loss$qU>��       �	�
�fc�A�*

lossaj�>����       �	���fc�A�*

loss�Y>���*       �	RE�fc�A�*

loss r�=�A�       �	"��fc�A�*

loss�p�=f�7�       �	��fc�A�*

loss��=`&w[       �	d!�fc�A�*

loss 5>���c       �	|��fc�A�*

loss�>EP��       �	�O�fc�A�*

loss��=�B       �	��fc�A�*

loss�i6>P��g       �	-��fc�A�*

loss�>���       �	�/�fc�A�*

loss.>��(V       �	�ʓfc�A�*

loss�Z�>���       �	�o�fc�A�*

loss���=�gm       �	��fc�A�*

loss ��>���       �	!�fc�A�*

loss>�S       �	��fc�A�*

loss�@->-�0       �	X�fc�A�*

loss+�">���       �	̳�fc�A�*

loss(��>kw"W       �	�J�fc�A�*

loss�>L��       �	7�fc�A�*

lossJ��=; �       �	W{�fc�A�*

loss�.>Q� �       �	�fc�A�*

lossn3�=C.6       �		��fc�A�*

loss@�->��Ƥ       �	1A�fc�A�*

loss�w >�U�}       �	�S�fc�A�*

loss�?�>��       �	��fc�A�*

loss:CA> <B�       �	3��fc�A�*

loss�1�=$4q�       �	�D�fc�A�*

loss8-�=U�>       �	3ݞfc�A�*

loss.7->Hc��       �	܃�fc�A�*

lossZ��<{%       �	� �fc�A�*

loss桩=7�u       �	�àfc�A�*

loss��i>��9o       �	,a�fc�A�*

loss�H�=��O�       �	P��fc�A�*

loss�M>~�       �	���fc�A�*

loss�?�=�F       �	�7�fc�A�*

loss�I�=�j�b       �	�ϣfc�A�*

loss6t=Zb��       �	���fc�A�*

loss�)�=��7       �	d�fc�A�*

lossI!=��uI       �	Է�fc�A�*

loss�Z=z:Q;       �	YN�fc�A�*

loss�>��@       �	�fc�A�*

loss]��=��[�       �	D��fc�A�*

loss�G=i'W�       �	�#�fc�A�*

loss���=�!�       �	��fc�A�*

loss�z0>��u       �	L��fc�A�*

lossV��=���       �	���fc�A�*

loss�7�>�)�       �	�ګfc�A�*

loss�.>b�ڞ       �	5��fc�A�*

loss��Q>�˗�       �	Fҭfc�A�*

lossܹ�=��
�       �	���fc�A�*

loss��=˸_       �	s/�fc�A�*

loss���=����       �	��fc�A�*

lossHC�=K�y�       �	�Ȱfc�A�*

loss_��=;n�7       �	7��fc�A�*

loss��B>����       �	0Բfc�A�*

loss�8D=�5�$       �	L��fc�A�*

lossz�=�CQZ       �	�M�fc�A�*

loss|d�=_�L�       �	���fc�A�*

loss�E�=n���       �	 S�fc�A�*

lossn>���       �	��fc�A�*

loss�_>�s�       �	5Ϸfc�A�*

loss$ >̬       �	p�fc�A�*

loss���=�a�       �	+��fc�A�*

lossh��=O#        �	��fc�A�*

lossć>0���       �	���fc�A�*

loss��==!�        �	�r�fc�A�*

loss� �=���       �	��fc�A�*

lossr\>���R       �	&��fc�A�*

loss9v�=�H       �	�@�fc�A�*

loss�=�i�
       �	z߾fc�A�*

loss$�<>b��       �	�fc�A�*

lossq~�=�ɾ�       �	&�fc�A�*

loss��=K�.\       �	�)�fc�A�*

loss��=�~�       �	��fc�A�*

loss���>ɖ�       �	�b�fc�A�*

loss+O�=x��b       �	�)�fc�A�*

loss&�=3�0       �	���fc�A�*

loss�"Z>�"��       �	zo�fc�A�*

loss1�3>IJ�       �	��fc�A�*

loss�g >]>z       �	���fc�A�*

loss�h>5:H       �	AF�fc�A�*

loss��=���       �	���fc�A�*

loss�~Q=|z�       �	�z�fc�A�*

loss��z>tۮ�       �	�fc�A�*

loss.ٚ=�P�       �	?��fc�A�*

lossڸW>)2(6       �	'J�fc�A�*

loss�J>��UZ       �	H�fc�A�*

loss*�=�?��       �	ȴ�fc�A�*

loss�Vl<��8T       �	O�fc�A�*

lossT��=&M�       �	���fc�A�*

loss�>=&]��       �	���fc�A�*

loss�3�=�M�       �	Q-�fc�A�*

loss��=)�IA       �	y��fc�A�*

loss_r`>�[�       �	Pm�fc�A�*

loss�b.>�Mi       �	d�fc�A�*

loss��@=z'��       �	���fc�A�*

loss��->����       �	�>�fc�A�*

loss܊=u5�       �	8��fc�A�*

loss�=���       �	.v�fc�A�*

loss>��-       �	��fc�A�*

lossT�l=D&O�       �	���fc�A�*

loss/��=���       �	�>�fc�A�*

loss���=v<D       �	<��fc�A�*

lossE�>7o�       �	:u�fc�A�*

loss7�7=�x�p       �	��fc�A�*

loss���<M7#       �	���fc�A�*

lossV2@=�R       �	�]�fc�A�*

loss�f5>FI�S       �	��fc�A�*

loss|!y=;:;2       �	��fc�A�*

loss?�=a���       �	M�fc�A�*

loss:��=�]       �	���fc�A�*

loss֖�={t�$       �	Ύ�fc�A�*

loss�ix>ݒ��       �	e8�fc�A�*

loss��C>�{K       �	���fc�A�*

loss�9�=O5n�       �	Á�fc�A�*

lossR�>��4E       �	�'�fc�A�*

lossy3>}�Q�       �	T��fc�A�*

losssh�=`I	       �	mu�fc�A�*

loss�׏=���       �	��fc�A�*

loss���=R�ż       �	���fc�A�*

loss��=3LLJ       �	|B�fc�A�*

loss���=�G�
       �	���fc�A�*

loss�]	>Y��~       �	o��fc�A�*

lossܥ
>��"%       �	�/�fc�A�*

loss�aH>b �       �	$��fc�A�*

loss8] >UwmP       �	�x�fc�A�*

loss,�2>g�:4       �	��fc�A�*

loss6�=�bl       �	���fc�A�*

loss�܈>Bq�       �	�h�fc�A�*

loss�z>u�<:       �	��fc�A�*

losslQ!>0�ڌ       �	���fc�A�*

loss��->�e       �	IL�fc�A�*

lossrd=��3�       �	/��fc�A�*

lossnF=��       �	���fc�A�*

loss4�+>��Mf       �	���fc�A�*

loss1U2>m[       �	1x�fc�A�*

loss�>�N�       �	d��fc�A�*

lossd��=�N��       �	�C�fc�A�*

lossD>E�       �	��fc�A�*

loss�K>Z{�w       �	/�fc�A�*

loss���=�[)�       �	���fc�A�*

loss	>L�	       �	0K�fc�A�*

loss�?>wv+       �	X��fc�A�*

loss͵>��D       �	z�fc�A�*

loss_c_>�m�       �	��fc�A�*

loss=�=��       �	7��fc�A�*

loss?�R=�oR�       �	cD�fc�A�*

loss��=�ʀ/       �	���fc�A�*

loss�N#>���       �	`s�fc�A�*

loss��=%LJ]       �	��fc�A�*

loss��b>�B�       �	d��fc�A�*

loss��D>�'D       �	�L�fc�A�*

lossTe�=����       �	���fc�A�*

loss��=�-w?       �	��fc�A�*

loss�G >���       �	ZH�fc�A�*

lossw�>ƃ��       �	4��fc�A�*

loss�=�`�       �	)��fc�A�*

loss��=���8       �	�)�fc�A�*

loss�j�>�ɺ�       �	���fc�A�*

loss���=���       �	�d�fc�A�*

lossڹS>�99�       �	;��fc�A�*

loss�Z�=���       �	��fc�A�*

loss�(�>��u       �	�a�fc�A�*

lossۨA>ٹ��       �	� �fc�A�*

loss��#>����       �	x��fc�A�*

loss�l=�7�       �	+4�fc�A�*

loss��=.b�H       �	���fc�A�*

loss���=��5�       �	�w fc�A�*

loss�=	>�`�{       �	fc�A�*

loss�uK=;�.
       �	�)fc�A�*

loss{��=+�#�       �	Z�fc�A�*

loss���=qIд       �	��fc�A�*

lossȦm>�ޔ       �	34fc�A�*

lossb>2���       �	Ҧfc�A�*

loss$�,>��D�       �	ǽfc�A�*

loss��>�x�       �	�bfc�A�*

loss��)>uKA�       �	�	fc�A�*

loss�ҫ=;|Sx       �	o�	fc�A�*

lossP�=}�F�       �	�
fc�A�*

lossCG>w��       �	bfc�A�*

loss�8(>���       �	�fc�A�*

loss<E�=����       �	$�fc�A�*

lossq2!>%W?       �	�tfc�A�*

lossڌ�=��k�       �	�fc�A�*

loss�0=�6N       �	��fc�A�*

loss]3�=N��       �	�Hfc�A�*

loss�F�=ز��       �	~�fc�A�*

loss�5z=F61P       �	`xfc�A�*

loss ��=�\x       �	g&fc�A�*

loss՚�>#�q       �	k�fc�A�*

lossx;->N��*       �	�Sfc�A�*

lossC��=4g�       �	�fc�A�*

loss��=��Da       �	!�fc�A�*

loss�T�<K�f       �	-fc�A�*

loss�M.>�th�       �	n�fc�A�*

loss7��>�8u       �	<fc�A�*

loss� �=��+       �	|�fc�A�*

loss���=oNKb       �	�dfc�A�*

losso�a>�C       �	Q�fc�A�*

loss�5&>��e       �	_�fc�A�*

loss|�=�]q       �	/fc�A�*

loss��=�nm�       �	;�fc�A�*

lossm�.>�@,       �	�Wfc�A�*

loss.�8>���       �	F�fc�A�*

loss�1>Ro8t       �	Krfc�A�*

loss=�p>�LG9       �	�.fc�A�*

loss���=�:k       �	��fc�A�*

lossR��=�,�       �	rjfc�A�*

lossH��=����       �	�fc�A�*

lossq��=�m�       �	� fc�A�*

loss-U=��       �	̘ fc�A�*

loss���=T       �	�*!fc�A�*

loss��>�FhZ       �	��!fc�A�*

lossJ�X=�IA�       �	�Q"fc�A�*

lossi�B=��<       �	��"fc�A�*

loss:�>M=7�       �	��#fc�A�*

lossh�=�z       �	�+$fc�A�*

loss��=Fe��       �	E�$fc�A�*

lossa%=��       �	~W%fc�A�*

loss�Q>B(p:       �	c�%fc�A�*

loss�!�=�L��       �	��&fc�A�*

loss+7>� �u       �	'fc�A�*

loss3�`>�֐       �	��'fc�A�*

loss��=�%y       �	l�(fc�A�*

loss% T=FF�       �	��)fc�A�*

lossw=ưU�       �	T�*fc�A�*

lossDo�=��       �	��+fc�A�*

loss��>%�LE       �	��,fc�A�*

lossH�*>���       �	�'.fc�A�*

loss;��><�.       �	<./fc�A�*

lossC��=�ڣG       �	�0fc�A�*

loss�`�=���V       �	11fc�A�*

lossJ50>��|�       �	�b2fc�A�*

loss|R�=d�h       �	�3fc�A�*

loss�[l={,e�       �	��3fc�A�*

loss��=����       �	�4fc�A�*

loss��6>q���       �	.�6fc�A�*

loss乁=�b�       �	K;7fc�A�*

lossC�'>Ǜ?�       �	�7fc�A�*

loss]��=ސ��       �	؀8fc�A�*

loss)Z�=T<B�       �	h�9fc�A�*

loss[��<a n�       �	UP:fc�A�*

loss&th=VvP�       �	��:fc�A�*

loss�=�V       �	r�;fc�A�*

lossHr�<d$�       �	�%<fc�A�*

loss�?>�zf       �	�<fc�A�*

loss���<�W��       �	CX=fc�A�*

lossS��=l�J�       �	��=fc�A�*

loss|�L>�Yn�       �	.�>fc�A�*

loss��_>���       �	�2?fc�A�*

loss�*"=���F       �	�?fc�A�*

loss4=��
�       �	�i@fc�A�*

loss���<T=��       �	voAfc�A�*

lossq��=��l3       �	�Bfc�A�*

loss�'T=,a�W       �	"�Bfc�A�*

loss&/>���       �	�CCfc�A�*

loss��=:��!       �	��Cfc�A�*

loss��P>��:l       �	�vDfc�A�*

loss���=<��J       �	�Efc�A�*

loss��e=�}       �	-�Efc�A�*

loss$T�=uz0�       �	�|Ffc�A�*

loss�j�<:*�:       �	�Gfc�A�*

lossv��=W,b�       �	�&Hfc�A�*

loss��=����       �	+�Hfc�A�*

loss�ر=Y�b�       �	��Ifc�A�*

loss�g�=zFɦ       �	KJfc�A�*

lossF�=���       �	��Jfc�A�*

loss��0=�S�       �	�_Lfc�A�*

lossn��=�C3�       �	��Lfc�A�*

loss�]8=@��w       �	B�Mfc�A�*

loss��s=�bF�       �	0Nfc�A�*

loss���<G	�       �	��Nfc�A�*

loss��D=Hɝ       �	=dOfc�A�*

loss}�=)
�       �	��Ofc�A�*

loss���=l�       �	��Pfc�A�*

loss�&l=���       �	{3Qfc�A�*

loss�3>���       �	��Qfc�A�*

loss^�">�M��       �	nRfc�A�*

loss �a=.k�       �	�Sfc�A�*

loss��=�~S       �	r�Sfc�A�*

loss�">��Vz       �	O?Tfc�A�*

loss�=�T��       �	��Tfc�A�*

loss ��<���       �	�}Ufc�A�*

loss�؟<�U
\       �	�Vfc�A�*

lossZ�=�E�P       �	�Vfc�A�*

losszs�<#�@       �	�TWfc�A�*

loss<~=��=}       �	I�Wfc�A�*

lossv�t<yym       �	C�Xfc�A�*

loss�	>�b7v       �	�3Yfc�A�*

lossˋ;���T       �	t�Yfc�A�*

loss�*<k�       �	*qZfc�A�*

loss �;q���       �	�[fc�A�*

lossJ��=R��       �	�[fc�A�*

loss�	;>��       �	�G\fc�A�*

loss��=3��       �	��\fc�A�*

losssY;�ٖ�       �	�{]fc�A�*

lossݨ�=c:�       �	;^fc�A�*

loss��>Jp�d       �	E�^fc�A�*

lossϳ�;����       �	*S_fc�A�*

loss�E@>�p�P       �	]�_fc�A�*

lossS�=�p+       �	��`fc�A�*

loss��|=��B       �	�7afc�A�*

loss���=�dC       �	8�afc�A�*

loss�G%=��)�       �	Nbfc�A�*

loss4@�=0��       �	icfc�A�*

loss���=	��       �	ͮcfc�A�*

loss�t>G �-       �	�Odfc�A�*

lossԻ�=�)'Z       �	��dfc�A�*

loss��=��       �	��efc�A�*

loss��M>�օ(       �	�4ffc�A�*

lossL��=H8m�       �	�ffc�A�*

loss��=B}\�       �	jjgfc�A�*

loss �>>��k       �	2hfc�A�*

lossγ>X�<�       �	��hfc�A�*

loss�՝=��:�       �	JEifc�A�*

lossCm�=�$�       �	e�ifc�A�*

loss���=��       �	�rjfc�A�*

loss�5=#���       �	�kfc�A�*

loss�s;=|��!       �	i�kfc�A�*

loss튦=�n='       �		Olfc�A�*

loss�<�>��	n       �	��lfc�A�*

lossZ�)=��h�       �	��mfc�A�*

lossTH=)3��       �	#nfc�A�*

lossL�<�D �       �	;�nfc�A�*

loss@P:>[��       �	,gofc�A�*

lossݺn=���       �	�ofc�A�*

loss|��=+I
�       �	?�pfc�A�*

loss�_>:�q       �	� qfc�A�*

loss�n%=�n|�       �	��qfc�A�*

lossQ�h=����       �	�Drfc�A�*

loss���=|�*/       �	 �rfc�A�*

loss���<�
a       �	�vsfc�A�*

loss�t=�d�~       �		tfc�A�*

loss�)>���       �	z�tfc�A�*

loss2�	=+���       �	cEufc�A�*

loss��k=�n��       �	�ufc�A�*

loss��$>�w��       �	i�vfc�A�*

loss��>?Mĭ       �	/6wfc�A�*

lossZ�G=}�{�       �	�wfc�A�*

lossJ�=-�       �	@hxfc�A�*

loss|T<=���q       �	lyfc�A�*

lossn��='��       �	|�yfc�A�*

loss�&�=�L��       �	=zfc�A�*

loss��=�:G�       �	�zfc�A�*

loss[*>?�Z       �	,d{fc�A�*

lossP�=����       �	��{fc�A�*

loss�@:>�cf�       �	��|fc�A�*

loss���<n�~(       �	D1}fc�A�*

loss��=���       �	}�}fc�A�*

loss���=����       �	�љfc�A�*

lossJρ=�Z�       �	�b�fc�A�*

loss<mg>�       �	`�fc�A�*

loss��=��˭       �	ԛ�fc�A�*

lossn2�=��W       �	�.�fc�A�*

loss�`=`�
�       �	U��fc�A�*

loss��=��h;       �	�v�fc�A�*

loss��9>?�?b       �	9�fc�A�*

loss;��=�ڑY       �	즞fc�A�*

lossd��=�8}       �	�H�fc�A�*

lossK�<���8       �	@�fc�A�*

loss��=�#g       �	S��fc�A�*

lossk�>l�km       �	�J�fc�A�*

loss���=m��m       �	��fc�A�*

loss�õ=xw�       �	�}�fc�A�*

loss{�>��A       �	�fc�A�*

loss@=,jH       �	V��fc�A�*

loss��>8��       �	�$�fc�A�*

lossePo=�`�       �	�ѧfc�A�*

loss跍>�p5
       �	zn�fc�A�*

lossaB�=�оV       �	#�fc�A�*

lossH��=.��       �	� �fc�A�*

loss:�=Nu��       �	VG�fc�A�*

lossj�>y7�       �	3k�fc�A�*

lossf��=h��\       �	2�fc�A�*

loss��=]�J       �	�l�fc�A�*

loss�(>8��       �	��fc�A�*

loss�:U=д@|       �	ޯfc�A�*

loss��=�5}       �	���fc�A�*

loss�\>�M�       �	��fc�A�*

loss�kl=m���       �	��fc�A�*

lossWLF=�_)       �	㨲fc�A�*

loss�D�=��       �	�C�fc�A�*

lossD8>���       �	V׳fc�A�*

loss�4f=%O�       �	�l�fc�A�*

lossZt�=`��c       �	��fc�A�*

losso�q<��{h       �	[��fc�A�*

lossAI'>�|N       �	�1�fc�A�*

lossI�>����       �	&ƶfc�A�*

losss�>�Qg�       �	5_�fc�A�*

loss��(>��s       �	���fc�A�*

loss-إ=4h��       �	���fc�A�*

losssr<��m(       �	�'�fc�A�*

loss���=�J	       �	���fc�A�*

loss���=��       �	�T�fc�A�*

lossr�k>Qd�       �	 ��fc�A�*

lossy={�3�       �	���fc�A�*

loss�I�<k��-       �	��fc�A�*

loss;\/=5-       �	k��fc�A�*

loss2u�<񽋘       �	/M�fc�A�*

loss���<�u       �	��fc�A�*

loss�&->c��=       �	.��fc�A�*

loss���=��       �	% �fc�A�*

loss�a�>�޶       �	��fc�A�*

loss���<"g       �	�g�fc�A�*

loss8��<o���       �	��fc�A�*

loss��H<��h�       �	���fc�A�*

lossgz<ٞG       �	DP�fc�A�*

lossqb�=7�B�       �	���fc�A�*

loss��>��       �	�{�fc�A�*

loss�1>2<4?       �	k�fc�A�*

loss��=&r[       �	���fc�A�*

loss�f�<O��       �	���fc�A�*

loss�?�=��9       �	ȕ�fc�A�*

loss���=�С�       �	�,�fc�A�*

loss6�8=~9�       �	���fc�A�*

loss4r�=��`       �	�Z�fc�A�*

loss��=��hI       �	���fc�A�*

lossJ��=H8��       �	@��fc�A�*

lossB�={�v       �	�c�fc�A�*

loss? �=�A&]       �	���fc�A�*

loss�v/>i���       �	ٖ�fc�A�*

lossD[=�*�T       �	�1�fc�A�*

loss�"�=;1�<       �	6��fc�A�*

lossWG�=D�Ȱ       �	�g�fc�A�*

loss�͟=�_+�       �	E�fc�A�*

loss�=@Յ-       �	6��fc�A�*

loss1�>=>�       �	YO�fc�A�*

lossxG�=	7       �	���fc�A�*

loss�=��       �	[}�fc�A�*

loss,j�=:,�       �	��fc�A�*

lossr�=�       �	�d�fc�A�*

lossㄾ=p�U�       �	��fc�A�*

loss�-�;�
�       �	K��fc�A�*

loss���=�G|�       �	�C�fc�A�*

loss�]�=nU�       �	��fc�A�*

loss� >(<X�       �	�p�fc�A�*

loss���=�e�g       �	D�fc�A�*

loss�ߣ=��I       �	9��fc�A�*

loss�&=A�63       �	�\�fc�A�*

loss൶=su7�       �	�1�fc�A�*

loss�G�="��%       �	W��fc�A�*

lossW�=��`�       �	%u�fc�A�*

lossZ�F>�Ε        �	��fc�A�*

loss�>���       �	���fc�A�*

lossiS�<�BŚ       �	�O�fc�A�*

loss�e�=�3��       �	C��fc�A�*

loss��=s��(       �	���fc�A�*

loss���=��o-       �	�3�fc�A�*

lossE�#>����       �	���fc�A�*

loss�G�=Ԧ�F       �	�k�fc�A�*

loss��.=i��       �	l�fc�A�*

lossMA�=���5       �	8��fc�A�*

lossJ��=
|�       �	�=�fc�A�*

loss=S�=��UY       �	
��fc�A�*

lossl� >�_C�       �	���fc�A�*

loss��=��       �	�C�fc�A�*

loss�w=�,��       �	���fc�A�*

loss��(>=� �       �	|�fc�A�*

loss�_=��FS       �	��fc�A�*

loss�b>|:�y       �	���fc�A�*

lossnؼ<|��       �	�b�fc�A�*

loss��=z�}>       �	+��fc�A�*

loss�73<����       �	���fc�A�*

loss(��=v_��       �	�9�fc�A�*

loss�F�=�|:�       �	���fc�A�*

loss��=ȴ%�       �	�x�fc�A�*

loss���=�XL       �	
�fc�A�*

loss�-�=GN��       �	���fc�A�*

loss�n
=�D��       �	9C�fc�A�*

loss��=�`B       �	r��fc�A�*

loss�<�=��4       �	v�fc�A�*

loss�¿=�u[       �	��fc�A�*

loss�">�	`       �	Y��fc�A�*

loss�O�<k���       �	�C�fc�A�*

loss�,�=��<       �	���fc�A�*

loss`9�=4`��       �	S{�fc�A�*

loss@Z�<DJ"k       �	��fc�A�*

loss���=�Ŵ       �	\��fc�A�*

loss�ć=0e�       �	�E�fc�A�*

loss��>T<?�       �	��fc�A�*

loss���=i8�       �	~�fc�A�*

loss�F= ��L       �	�fc�A�*

loss6��=l�	�       �	���fc�A�*

lossm�1>��k       �	Y�fc�A�*

loss
w=�]I       �	��fc�A�*

loss�2=��4�       �	o��fc�A�*

loss,�=�9*�       �	V�fc�A�*

loss��<=�U��       �	{��fc�A�*

loss&W=;,�       �	K9�fc�A�*

loss�V=k�M       �	���fc�A�*

loss��<ZE��       �	!u�fc�A�*

lossn��=eh�       �	h	�fc�A�*

loss|��=����       �	j��fc�A�*

loss�+>�f��       �	�4�fc�A�*

loss��e=ei̥       �	8��fc�A�*

loss�P=>yQ�       �	�o�fc�A�*

loss�T<�b3k       �	�fc�A�*

loss]��<eA       �	{��fc�A�*

loss���=�<�       �	�H�fc�A�*

loss��^=%B�       �	���fc�A�*

loss���=���       �	s��fc�A�*

loss��>�KA{       �	��fc�A�*

loss���=UŰ       �	?��fc�A�*

loss�]�=ckX       �	og�fc�A�*

loss_R�<�=.�       �	� fc�A�*

lossR-�=��       �	o� fc�A�*

loss���=��9       �	�1fc�A�*

loss3[>�늪       �	��fc�A�*

loss-`=-�K�       �	S]fc�A�*

lossܝ�=�B��       �	��fc�A�*

loss��=�>�       �	��fc�A�*

loss�NG=���       �	� fc�A�*

loss�%=�]�       �		�fc�A�*

loss�=Jʹ�       �	d]fc�A�*

loss�c�=���A       �	e�fc�A�*

loss��!=m�       �	>�fc�A�*

lossF0>^���       �	�9fc�A�*

loss�.�=���       �	��fc�A�*

loss�^x=��8�       �	w�fc�A�*

loss|�=g���       �	G	fc�A�*

loss�}<�z       �	`�	fc�A�*

loss�u=�ƴZ       �	
fc�A�*

lossa�=�<=       �	�fc�A�*

loss\]	=��{       �	��fc�A�*

lossTd=Ө�       �	p[fc�A�*

loss��>��?       �	��fc�A�*

loss��=�?       �	��fc�A�*

loss�	�=��bO       �	\fc�A�*

loss��F=S       �	x�fc�A�*

loss�j�<s)�       �	Jfc�A�*

loss11�=K2�       �	��fc�A�*

lossJ�=�E��       �	�ufc�A�*

loss��v<�:E       �	Ffc�A�*

loss�)=��       �	Ԟfc�A�*

loss,}=�hu       �	�Qfc�A�*

loss�0=��b�       �	u�fc�A�*

loss#4�<�L=       �	�fc�A�*

lossa_(=���       �	�fc�A�*

lossf��=�mC4       �	Ѱfc�A�*

loss~�>z`Q       �	Ifc�A�*

lossv�<�       �	��fc�A�*

loss�xn=1�       �	��fc�A�*

loss�<ߧ��       �	1(fc�A�*

loss �<=���       �	��fc�A�*

lossj�v=�@       �	�cfc�A�*

lossM*�<DI�       �	��fc�A�*

lossE��=K�P�       �	�fc�A�*

lossx�>�	�=       �	n�fc�A�*

loss
p�<����       �	o�fc�A�*

loss=yF=Q�.^       �	"ofc�A�*

loss��=�~��       �	�fc�A�*

loss��I=p]�z       �	��fc�A�*

loss#�=��2       �	Afc�A�*

loss(�=��=       �	#�fc�A�*

loss�ȿ=��?�       �	Tqfc�A�*

loss��=dP��       �	� fc�A�*

loss�1�=G�       �	�� fc�A�*

lossXk$=̮z�       �	2<!fc�A�*

loss�:�=`CԸ       �	
�!fc�A�*

loss�yK=���
       �	+l"fc�A�*

loss+=m��       �	�#fc�A�*

loss��<���       �	
�#fc�A�*

loss�f�<��{x       �	VD$fc�A�*

loss*�=ǍD�       �	�$fc�A�*

loss�t�=���N       �	�p%fc�A�*

loss2�V>?!�
       �	�&fc�A�*

loss�*>N�H�       �	U�&fc�A�*

loss�+Y>��o       �	�:'fc�A�*

loss��=��.       �	��'fc�A�*

loss��=�J|       �	Rd(fc�A�*

loss�C3=��η       �	z )fc�A�*

loss̾<�t�D       �	%�)fc�A�*

loss�מ=(�+a       �	^.*fc�A�*

lossD�<5�       �	<g+fc�A�*

loss��=O/E�       �	,fc�A�*

loss��=Qa�T       �	��,fc�A�*

lossl�=�b�       �	AG-fc�A�*

loss�[=��[       �	D�-fc�A�*

lossg�=���$       �	֐.fc�A�*

loss��<��Դ       �	j0/fc�A�*

loss��<�       �	Y�/fc�A�*

loss$��=�tG�       �	}�0fc�A�*

loss;�_=��       �	.1fc�A�*

loss��=kl��       �	.�1fc�A�*

losss�F=K�.�       �	�f2fc�A�*

loss|�I=n=3       �	l	3fc�A�*

loss͉=}��       �	I�3fc�A�*

loss+9>R罣       �	�U4fc�A�*

loss[�5<�rZ�       �	��4fc�A�*

loss�X=m��k       �	5fc�A�*

loss�E�=��       �	�b6fc�A�*

loss�I�=!�Ze       �	��6fc�A�*

lossAZ�=�K�D       �	X�7fc�A�*

loss���=hݷz       �	Z-8fc�A�*

loss��N=���       �	�9fc�A�*

loss��<��       �	ߣ9fc�A�*

lossIh�=����       �	P:fc�A�*

loss��=�T:       �	� ;fc�A�*

loss��<ё�6       �	t�;fc�A�*

loss��>=�j�a       �	�2<fc�A�*

lossHZ�=��Gl       �	d�<fc�A�*

loss�O=>�$       �	i=fc�A�*

loss$��<m�C�       �	d>fc�A�*

loss��<�r�,       �	|�>fc�A�*

loss`�_=��K�       �	74?fc�A�*

loss�T=�b�       �	��?fc�A�*

loss�6�=���       �	�`@fc�A�*

loss��F=���       �	<�@fc�A�*

lossDG�=L��#       �	��Afc�A�*

loss�e�=�9�`       �	�2Bfc�A�*

loss�~�<ݕ�x       �	�(Cfc�A�*

loss3��=添l       �	��Cfc�A�*

loss�F=`�MI       �	DiDfc�A�*

loss�Ę=�]�       �	� Efc�A�*

lossW�D=6G��       �	8�Efc�A�*

loss>�n�j       �	N}Ffc�A�*

loss6�=u�       �	+Gfc�A�*

loss=�>12��       �	�
Hfc�A�*

loss$�u=��e8       �	��Hfc�A�*

loss�])=2�L        �	�`Ifc�A�*

lossv֫=R8#       �	)Jfc�A�*

loss�7~=$�       �	�Jfc�A�*

loss�>�I�#       �	�>Kfc�A�*

lossE�A>Z�;       �	=�Kfc�A�*

loss>��T       �	�oLfc�A�*

loss#1=:Hh       �	|eMfc�A�*

lossDw<#^�"       �	��Mfc�A�*

loss�ُ=���       �	6�Nfc�A�*

loss���=��6�       �	�gOfc�A�*

lossn(�;D�dF       �	��Ofc�A�*

loss���<��<       �	�Pfc�A�*

loss-\�=d`e       �	9Qfc�A�*

loss5=.��       �	w�Qfc�A�*

lossZ��=&�       �	�mRfc�A�*

loss�q=m��       �	�Sfc�A�*

lossE�=�9.       �	a�Sfc�A�*

loss��<�*��       �	NCTfc�A�*

loss��=2�a       �	��Tfc�A�*

loss���<�sz�       �	\wUfc�A�*

loss�]z<�v@       �	
Vfc�A�*

loss~P=���:       �	&�Vfc�A�*

lossҨq<C�$R       �	�DWfc�A�*

loss΅B=�Akp       �	��Wfc�A�*

loss�Ј=�4�_       �	]oXfc�A�*

loss8m�="���       �	�Yfc�A�*

loss�&V=��B�       �	��Yfc�A�*

loss�{�=����       �	6Zfc�A�*

loss��=�<��       �	=�Zfc�A�*

lossX�=u�Zp       �	�r[fc�A�*

loss	�X=�BT�       �	�\fc�A�*

loss��B=�~�       �	�\fc�A�*

lossJ��=���d       �	B]fc�A�*

loss��:>���       �	��]fc�A�*

loss�l=�g       �	��^fc�A�*

lossjT4>Η�S       �	@1_fc�A�*

lossJ��<����       �	��_fc�A�*

loss	��=�S�k       �	�m`fc�A�*

loss��5=	9�4       �	�afc�A�*

loss�.s=C˽T       �	��afc�A�*

loss��#=�C��       �	�Nbfc�A�*

loss[�=`9�       �	��bfc�A�*

loss�Ӓ=5�j/       �	 �cfc�A�*

loss꽙=4vW�       �	�Edfc�A�*

lossi��<���       �	��dfc�A�*

lossjG1=� l       �	Ӆefc�A�*

loss2�<K!��       �	�!ffc�A�*

loss?�<�I��       �	I�ffc�A�*

lossl�=ӻ�o       �	�Ygfc�A�*

loss�\�=�z�       �	��gfc�A�*

loss�5�<k]z�       �	��hfc�A�*

loss��?>�G�       �	Cifc�A�*

loss ��=TW(�       �	[�ifc�A�*

loss8��<ё�`       �	sLjfc�A�*

loss��>�+       �	S�jfc�A�*

loss��'>�2       �	M�kfc�A�*

loss#=%'=�       �	3Nlfc�A�*

loss_�O=t�$�       �	��lfc�A�*

loss�@}=�ia       �	1�mfc�A�*

loss6->^")?       �	�nfc�A�*

loss;�=����       �	|ofc�A�*

lossm��<ɀ�       �	]pfc�A�*

losseW�<ܰ       �	��pfc�A�*

loss��=����       �	=rfc�A�*

lossb�=Y�       �	'�rfc�A�*

loss�J�=;�Ù       �	Ŏsfc�A�*

loss��;=e�       �	kItfc�A�*

loss(H>��.       �	�ufc�A�*

loss���=���       �	��vfc�A�*

lossW�<�s��       �	;Vwfc�A�*

loss�"<:_�       �	�xxfc�A�*

loss�7=u��       �	/yfc�A�*

loss�s�=Q�        �	.�yfc�A�*

lossM.�=��       �	<�zfc�A�*

loss<�>H��       �	�W{fc�A�*

loss}?�=J4*       �	4�{fc�A�*

lossX�=<fX�       �	Œ|fc�A�*

loss�� >ޝ�$       �	�:}fc�A�*

loss�ʂ=�G��       �	u�}fc�A�*

loss#3�=��#�       �	K�~fc�A�*

lossxN=N��       �	�/fc�A�*

loss?0=[���       �	K�fc�A�*

loss��="��       �	�g�fc�A�*

loss!mQ=o�`�       �	��fc�A�*

loss���=���       �	m��fc�A�*

lossG�=��`y       �	[C�fc�A�*

loss �I<r��       �	<ۂfc�A�*

loss|��<�0�$       �	��fc�A�*

loss�M�=�@�r       �	�'�fc�A�*

loss;��=I�w       �	vÄfc�A�*

loss䦮=���r       �	_�fc�A�*

lossO^�<Л�       �	T��fc�A�*

loss�=�Dvl       �	��fc�A�*

loss�>u��y       �	�7�fc�A�*

loss���=���       �	pчfc�A�*

loss�V�=T�       �	q�fc�A�*

loss�s=�wX       �	� �fc�A�*

lossV�g<��9       �	̉fc�A�*

loss�!�<�j�h       �	�v�fc�A�*

losswͣ=Q�I       �	��fc�A�*

loss�>
l�Q       �	���fc�A�*

loss���=3���       �	)\�fc�A�*

lossT9=�	�       �	��fc�A�*

loss{b&=�~Q3       �	���fc�A�*

loss��9=Ӕ��       �	�3�fc�A�*

lossЛ�=H~       �	tЎfc�A�*

loss/��=)�n       �	3o�fc�A�*

loss�C�=��N       �	$	�fc�A�*

lossr�Q=�UMr       �	ʥ�fc�A�*

loss��=N�X       �	�^�fc�A�*

lossl��=l=%�       �	v��fc�A�*

loss���=�.�Z       �	���fc�A�*

loss-�=��}       �	�(�fc�A�*

loss��=h��,       �	���fc�A�*

lossͽF<�ق!       �	\V�fc�A�*

loss�7>�>��       �	z��fc�A�*

loss��]=���       �	y��fc�A�*

lossC�=��ƃ       �	�(�fc�A�*

loss&b=��       �	ǽ�fc�A�*

lossʀ�=�8KN       �	d�fc�A�*

loss}E=�0�       �	�fc�A�*

loss�˸=��.       �	���fc�A�*

loss�$=.���       �	�D�fc�A�*

loss��a=�|��       �	���fc�A�*

loss�]q=��       �	� �fc�A�*

loss��=Uq       �	˼�fc�A�*

loss*�=�y�       �	XT�fc�A�*

lossR`=|��       �	��fc�A�*

loss��=�qx�       �	ƅ�fc�A�*

loss_��=���       �	�#�fc�A�*

lossq�6=ȍc�       �	��fc�A�*

loss� n=MN       �	�Y�fc�A�*

loss&�=l(,�       �	M��fc�A�*

loss���<wc|       �	X��fc�A�*

loss?�=OԳ       �	�*�fc�A�*

loss�-S=�L*�       �	U��fc�A�*

loss��==l5�       �	`�fc�A�*

loss�z�==�       �	���fc�A�*

lossF��=�]�       �	B��fc�A�*

loss��+>zċ       �	S<�fc�A�*

loss�`�=OC��       �	٤fc�A�*

loss��=�#�;       �	�u�fc�A�*

loss��<=����       �	��fc�A�*

lossn0�=�T��       �	���fc�A�*

lossmEx=�0}�       �	�A�fc�A�*

loss)Q=_CŪ       �	��fc�A�*

lossd� <b��       �	���fc�A�*

loss��8<{e>y       �	�.�fc�A�*

loss%�=";c�       �	8�fc�A�*

loss�H�<?@B�       �	4٪fc�A�*

lossm��<-�q       �	���fc�A�*

loss�>��)O       �	lB�fc�A�*

loss��9=�˔       �	��fc�A�*

loss�*=>l8�       �	�ǭfc�A�*

loss�?�=�s�       �	�l�fc�A�*

loss(>�=��       �	�l�fc�A�*

lossJ*=]�Y�       �	k��fc�A�*

loss�>��       �	1@�fc�A�*

loss���=TA       �	Tr�fc�A�*

loss�0�=Z�&       �	h#�fc�A�*

lossQ�=��5       �	�-�fc�A�*

loss 3=� �<       �	=ִfc�A�*

loss�0=�O��       �	G�fc�A�*

loss^Z�={�)       �	�a�fc�A�*

lossr%p=~���       �	�fc�A�*

loss��=?�i       �	 ��fc�A�*

loss)��<����       �	�]�fc�A�*

lossw4=f8M       �	~�fc�A�*

loss�vj=]�nP       �	Χ�fc�A�*

loss�<�3�       �	�G�fc�A�*

loss\j�>2`��       �	-�fc�A�*

loss<><�       �	���fc�A�*

loss:�g>�,       �	�'�fc�A�*

loss���=P�A_       �	���fc�A�*

loss��b<;��R       �	]�fc�A�*

loss;]�=lK
�       �	���fc�A�*

lossv�[>;��R       �	��fc�A�*

loss!A=�4��       �	z9�fc�A�*

lossZ2�=g�#�       �	 ��fc�A�*

loss|<�=APj       �	�H�fc�A�*

lossl��=����       �	���fc�A�*

loss��6=jυ       �	ς�fc�A�*

loss�<6�       �	3�fc�A�*

lossi��=e��U       �	���fc�A�*

losse`&>=@�       �	�Q�fc�A�*

loss���=c(5K       �	��fc�A�*

loss�|�=�̑�       �	/��fc�A�*

loss I�=��a       �	y$�fc�A�*

loss���<���       �	���fc�A�*

loss͞=���A       �	_\�fc�A�*

lossڽ|<���       �	u�fc�A�*

loss��}</��       �	��fc�A�*

loss3X�<VV       �	NE�fc�A�*

lossC� >�K�       �	C��fc�A�*

lossџU=ٌu       �	��fc�A�*

lossë�=(^@?       �	� �fc�A�*

loss^l�=(�w�       �	���fc�A�*

lossJ�*<vy        �	V�fc�A�*

lossU+�=t�}       �	~��fc�A�*

loss�F^<34J[       �	Փ�fc�A�*

lossR�>L*�       �	�0�fc�A�*

lossl�@=x��       �	X��fc�A�*

loss�i�=-͒J       �	�[�fc�A�*

loss�V<>��J       �	U��fc�A�*

loss��T=dY�       �	؞�fc�A�*

loss��|=��N�       �	 E�fc�A�*

loss��&<��5       �	��fc�A�*

loss�EG=xԠ�       �	��fc�A�*

lossl"Q='
՚       �	} �fc�A�*

loss��>F�A%       �	ϼ�fc�A�*

loss;�`>��D�       �	[�fc�A�*

loss���=��       �	���fc�A�*

lossIC�<�;       �	���fc�A�*

loss���=��	       �	�1�fc�A�*

loss�Ti=#D�\       �	B��fc�A�*

loss��=��̯       �	�j�fc�A�*

loss��6=��H2       �	��fc�A�*

loss��=j@�       �	,��fc�A�*

loss���<Q�a^       �	*�fc�A�*

loss��=VY�       �	,��fc�A�*

loss?U�=u�S�       �	�6�fc�A�*

loss���=cG�[       �	���fc�A�*

loss�P�<�A�       �	0��fc�A�*

loss��C=!�2       �	Ō�fc�A�*

loss���<D���       �	_(�fc�A�*

loss���<='�       �	���fc�A�*

loss�k�=��i�       �	T�fc�A�*

loss���<�F��       �	��fc�A�*

loss*)B=Dq�S       �	C��fc�A�*

loss�Xe=:���       �	�w�fc�A�*

lossIL�=�ӷ       �	Q�fc�A�*

lossc��<޳��       �	��fc�A�*

loss��'=���       �	�]�fc�A�*

loss|��<^)�       �	���fc�A�*

loss �>=�:I       �	�fc�A�*

loss�E=Ě'd       �	 �fc�A�*

loss�c=����       �	��fc�A�*

loss��`=q�       �	J�fc�A�*

loss.�=��       �	���fc�A�*

loss��<U��	       �	��fc�A�*

lossO{�<���n       �	�0�fc�A�*

lossN��=`�        �	���fc�A�*

loss6<�^�       �	�r�fc�A�*

loss�}a=zBø       �	��fc�A�*

lossx�Y<"���       �	���fc�A�*

loss�{>�S�       �	Ov�fc�A�*

loss�q�=�~��       �	R��fc�A�*

loss�83=�C       �	�j�fc�A�*

loss��=�M��       �	ҩ�fc�A�*

loss�a�=N֡[       �	��fc�A�*

loss` �<8��r       �	�Y�fc�A�*

loss�Ɇ<B��       �	��fc�A�*

lossfD�;�4�       �	���fc�A�*

loss<W<�ݿt       �	��fc�A�*

loss�0=��       �	i��fc�A�*

loss`�<��       �	���fc�A�*

loss��<4��M       �	�=�fc�A�*

lossV�s=�#�       �	���fc�A�*

loss�x>���       �	$��fc�A�*

loss�;�<�G��       �	�D�fc�A�*

losss� = h��       �	;��fc�A�*

lossn�1=C�       �	���fc�A�*

loss�j<6#�       �	�0�fc�A�*

loss��<��F�       �	��fc�A�*

lossX;X=i��       �	���fc�A�*

loss*�*=��ݽ       �	�c�fc�A�*

loss��A;�U       �	K�fc�A�*

loss��d=6�
       �	��fc�A�*

losscL�<��AA       �	��fc�A�*

loss �<5A�       �	qW fc�A�*

loss*4:v��       �	� fc�A�*

loss���:���       �	 �fc�A�*

loss��1:���e       �	��fc�A�*

loss8�D=C�eY       �	�Hfc�A�*

loss��=hG�       �	�~fc�A�*

loss�&M=���       �	1?fc�A�*

loss�9�;L��^       �	�,fc�A�*

loss�Y-=t�9       �	�,fc�A�*

loss$9�>=8�       �	B�fc�A�*

loss`8�<�0�c       �	-
	fc�A�*

loss��>�N1�       �	}�	fc�A�*

loss7!D=M��$       �	�R
fc�A�	*

lossqS�=�-�6       �	Yfc�A�	*

loss�`=mcp       �	��fc�A�	*

loss�#=y��y       �	�fc�A�	*

loss3��=V��*       �	��fc�A�	*

loss6�=߁�       �	x�fc�A�	*

loss3g�=���       �	�fc�A�	*

loss���=�9       �	uXfc�A�	*

loss��<�k�       �	f�fc�A�	*

loss��=�wH       �	մfc�A�	*

lossO]�=���G       �	�Mfc�A�	*

loss �Z=�v��       �	�wfc�A�	*

loss�'>�|'�       �	fc�A�	*

loss�߶=���=       �	Efc�A�	*

loss�/=���       �	��fc�A�	*

loss鮿<�d��       �	�jfc�A�	*

loss+�>~�j�       �	{fc�A�	*

lossc�=G��       �	܁fc�A�	*

loss���=y$��       �	k(fc�A�	*

loss�0�=N��"       �	��fc�A�	*

lossVA=��"�       �	��fc�A�	*

loss�=�<���       �	�sfc�A�	*

loss�.�<39�       �	fc�A�	*

lossI��<��t       �	��fc�A�	*

loss�0�=ٵv�       �	�Dfc�A�	*

loss�fD=L,UU       �	z�fc�A�	*

lossB�=�/M�       �	�| fc�A�	*

loss�ދ=�W>       �	v!fc�A�	*

losswI=�� ,       �	��!fc�A�	*

loss�=��v       �	�_"fc�A�	*

loss�"�<oYm�       �	#fc�A�	*

loss�9{; uD�       �	=�#fc�A�	*

lossj��=t*X�       �	�V$fc�A�	*

loss�Q�<�t�       �	��$fc�A�	*

loss�T<ḥ       �	��%fc�A�	*

loss%G�=+F4�       �	��&fc�A�	*

loss�&�=�Ϲ       �	�E'fc�A�	*

loss}�=�[%       �	d�'fc�A�	*

loss
7�<�       �	��(fc�A�	*

lossv�V=|>��       �	�4)fc�A�	*

lossX1�=�*x       �	+fc�A�	*

loss�kC=L��       �	��+fc�A�	*

loss6�<�(x       �	��,fc�A�	*

loss��K=�#\S       �	f-fc�A�	*

loss45>Ǵb       �	�	.fc�A�	*

loss�Ö<�W��       �	�.fc�A�	*

loss�� =M�f       �	�/fc�A�	*

losss*�<J��k       �	8�0fc�A�	*

loss�vE=� �0       �	U1fc�A�	*

lossO��=ћ��       �	��Ifc�A�	*

loss�a>�	       �	�tJfc�A�	*

lossȍ	>;��        �	RKfc�A�	*

lossf�
>Zt�r       �	D�Lfc�A�	*

losslv�=��\l       �	EMfc�A�	*

loss��=�\��       �	�[Nfc�A�	*

lossȂ=��k       �	��Nfc�A�	*

loss��=۫��       �	g�Ofc�A�	*

loss�� >�^\�       �	EPfc�A�	*

loss��T=����       �	[�Pfc�A�	*

loss(bo<#)7v       �	d�Qfc�A�	*

lossM�3=��aD       �	�>Rfc�A�	*

loss��=�g9�       �	@LSfc�A�	*

lossc�=Ռ�J       �	��Sfc�A�	*

loss���<۲�       �	F�Tfc�A�	*

loss��D=-G`�       �	v�Ufc�A�	*

lossO�(<����       �	�-Vfc�A�	*

loss#y�<�A�
       �	-�Vfc�A�	*

lossm��=EL/       �	�nWfc�A�	*

lossHH>�{�       �	EXfc�A�	*

loss��=~�       �	��Xfc�A�	*

lossL��=�gx�       �	�SYfc�A�	*

loss|�=���       �	��Yfc�A�	*

loss��@=��oO       �	�Zfc�A�	*

loss��=�J��       �	1([fc�A�	*

lossȇ�<�H�k       �	��[fc�A�	*

loss7N3=գy�       �	L�\fc�A�	*

lossĲZ=n �       �	xB]fc�A�	*

loss���=k�C       �	��]fc�A�	*

loss�C�=��K�       �	�{^fc�A�	*

loss,S*=6��       �	\!_fc�A�	*

lossd{R=W;�       �	Թ_fc�A�	*

lossf:8=�D       �	�Q`fc�A�	*

losscɥ=��1       �	?�afc�A�	*

loss�q=o	�       �	>Ybfc�A�	*

lossؼ�=#@       �	f�bfc�A�	*

lossFΆ<�ڮ�       �	cfc�A�	*

loss�{y=!BEv       �	�$dfc�A�	*

lossf>�36�       �	��dfc�A�	*

losst��=<!Ǉ       �	�Yefc�A�	*

loss!=��,       �	t�efc�A�	*

loss\QP=�Ϛe       �	��ffc�A�	*

loss}0<���Z       �	G!gfc�A�	*

loss\ޜ=}���       �	йgfc�A�	*

losssܩ=���3       �	+Phfc�A�	*

loss��=�c_�       �	�hfc�A�	*

loss��H=��g�       �	�ifc�A�	*

loss�Ѭ<ș��       �	�jfc�A�	*

loss��=1e�       �	V�jfc�A�	*

lossip)<�1�A       �	$dkfc�A�	*

loss��<QTl       �	��kfc�A�	*

loss
M�=��9\       �	M�lfc�A�	*

lossB<�4R'       �	omfc�A�	*

lossu*�>�m�(       �	 �nfc�A�	*

loss �Z=���}       �	�ofc�A�	*

loss��8<VlcY       �	��ofc�A�	*

lossͱ�;���        �	�|pfc�A�	*

lossQ<��c       �	w�qfc�A�	*

lossz�<�߾�       �	��rfc�A�	*

loss�D�=�Bp       �	��sfc�A�	*

loss@�>吞       �	@�tfc�A�	*

lossI1=�'�|       �	�ufc�A�	*

loss2(<[�       �	*�vfc�A�	*

loss�g=��&�       �	K�wfc�A�	*

losse�G<��       �	d>xfc�A�	*

loss���<^�{       �	�<yfc�A�	*

loss�z�=�ظ�       �	E�yfc�A�	*

loss�V,=��r�       �	uzfc�A�	*

lossd��=�2!q       �	H{fc�A�	*

loss���=�� ]       �	�{fc�A�	*

loss��
=2��       �	xb|fc�A�	*

loss�=Z68       �	�}fc�A�	*

loss��=CrT�       �	��}fc�A�	*

loss�H�=0RU�       �	�~fc�A�	*

loss?�=d�i�       �	�Afc�A�	*

loss�K=.��       �	��fc�A�	*

loss�$=l��D       �	3�fc�A�	*

lossa�=�յ�       �	�8�fc�A�	*

loss-�T<#�L�       �	�L�fc�A�	*

loss#T=2�<+       �	
h�fc�A�
*

loss��=Ї˧       �	j�fc�A�
*

lossτ�=�6�/       �	b�fc�A�
*

loss!p�<	q��       �	���fc�A�
*

loss��;�נ'       �	Ӈfc�A�
*

lossآj=��^       �	�o�fc�A�
*

lossD�,=�t9�       �	e�fc�A�
*

loss�F�<�v       �	���fc�A�
*

loss�Y=2f��       �	�S�fc�A�
*

lossuV<%�^       �	W�fc�A�
*

loss��=�R�       �	D��fc�A�
*

loss��2=TMo�       �	��fc�A�
*

loss�u�=��6E       �	�P�fc�A�
*

lossC�K=	[�E       �	s�fc�A�
*

loss�'=�',�       �	D��fc�A�
*

loss>��y�       �	�@�fc�A�
*

loss>�=ӱO       �	��fc�A�
*

lossf K=/� n       �	ڌ�fc�A�
*

lossn��<�aH�       �	�2�fc�A�
*

loss{�o>���_       �	�ܑfc�A�
*

loss��=P��       �	�z�fc�A�
*

lossA�0=��V�       �	7�fc�A�
*

loss��=�s11       �	V��fc�A�
*

loss��p=����       �	�\�fc�A�
*

loss��&=�t�e       �	 �fc�A�
*

lossv�<����       �	=��fc�A�
*

loss��=;�ѫ       �	R�fc�A�
*

loss�ʭ;,��v       �	���fc�A�
*

loss���<�䶙       �	Ę�fc�A�
*

lossS�j=A-1.       �	�2�fc�A�
*

loss�/�=�hݭ       �	=՘fc�A�
*

loss��>�e�?       �	p�fc�A�
*

loss7:=��}�       �	�O�fc�A�
*

loss7�=����       �	��fc�A�
*

loss���<��       �	���fc�A�
*

lossVj�<��>       �	>&�fc�A�
*

loss),=F���       �	@Üfc�A�
*

lossj_=�f�       �	�W�fc�A�
*

loss�8�=�땄       �	��fc�A�
*

lossMd =b�       �	���fc�A�
*

loss}��="/i�       �	4�fc�A�
*

lossd/2=�r       �	&ȟfc�A�
*

losst_�=c��m       �	=d�fc�A�
*

loss�p=lw�       �	��fc�A�
*

loss`HD=Xp�       �	0��fc�A�
*

loss;�e=iY�=       �	�M�fc�A�
*

lossm=��h5       �	��fc�A�
*

lossW�=���=       �	���fc�A�
*

loss��<|�1       �	k�fc�A�
*

loss��=�n-D       �	��fc�A�
*

lossw�=�$.V       �	S��fc�A�
*

loss��<�c       �	�X�fc�A�
*

loss2�=�WC       �	��fc�A�
*

loss�x@=۪W       �	��fc�A�
*

loss4�= [�       �	^G�fc�A�
*

loss_�=RME       �	�ݨfc�A�
*

loss��;��~�       �	[Ωfc�A�
*

loss)k=���       �	�q�fc�A�
*

loss�k�<���       �	
�fc�A�
*

loss<�<��22       �	C˫fc�A�
*

lossVU�<(Ǎ       �	��fc�A�
*

loss?��;��H       �	�~�fc�A�
*

loss�Y<,���       �	a�fc�A�
*

lossŗu=��0�       �	s��fc�A�
*

loss��<�       �	�X�fc�A�
*

loss�F;=���       �	���fc�A�
*

lossl �<�f�       �	���fc�A�
*

loss�O�<��`F       �	W$�fc�A�
*

lossZ�;��m=       �	�ñfc�A�
*

loss�Q�<:��       �	Na�fc�A�
*

loss��<�1'u       �	EH�fc�A�
*

loss�;1�b       �	�fc�A�
*

loss&l
=-�x       �	�x�fc�A�
*

loss���<��'4       �	E�fc�A�
*

loss��=�r{       �	fc�A�
*

lossJ�=E/��       �	�S�fc�A�
*

loss͒k=�9�R       �	���fc�A�
*

lossrE<�<       �	���fc�A�
*

loss���<�"��       �	.7�fc�A�
*

loss�Њ=�tl       �	Ըfc�A�
*

loss�4�;�%�G       �	2w�fc�A�
*

loss!��<c�4i       �	X �fc�A�
*

loss(��<�7�!       �	f��fc�A�
*

lossPC=��r       �	t]�fc�A�
*

loss�:�2�       �	��fc�A�
*

loss�� =��T�       �	�{�fc�A�
*

lossO�Q=�R��       �	��fc�A�
*

loss|�<��       �	��fc�A�
*

lossU�=?Ƃ*       �	�U�fc�A�
*

loss��h<�ڠ8       �	|��fc�A�
*

loss7��=i��h       �	'��fc�A�
*

loss�i=�'�_       �	WC�fc�A�
*

loss�|X<��w�       �	H��fc�A�
*

loss�آ=��       �	���fc�A�
*

loss�٢=p���       �	�+�fc�A�
*

loss�b�<�.�       �	���fc�A�
*

loss�p�;G��1       �	5��fc�A�
*

loss�΄=6���       �	W{�fc�A�
*

loss.�=|1z       �	P9�fc�A�
*

loss�<\���       �	���fc�A�
*

loss)%�<	z#�       �	8L�fc�A�
*

loss6P.;$���       �	���fc�A�
*

loss:,�<��       �	���fc�A�
*

loss���=n���       �	���fc�A�
*

loss=#�;�� =       �	��fc�A�
*

loss��
=l�[       �	���fc�A�
*

loss�05<%�
       �	�<�fc�A�
*

loss��g=凎�       �	�fc�A�
*

loss���<M�B       �	V��fc�A�
*

losse
E<ۂb"       �	X�fc�A�
*

loss��1< �go       �	��fc�A�
*

loss���=���       �	���fc�A�
*

loss��<5*%�       �	�z�fc�A�
*

lossI�!<?f�q       �	��fc�A�
*

loss\M�=�b�       �	ͮ�fc�A�
*

loss2��;���       �	I�fc�A�
*

loss
{�<��@       �	���fc�A�
*

loss%>�=�q�O       �	�x�fc�A�
*

loss&)>|W��       �	V�fc�A�
*

loss�U�=�l;       �	-��fc�A�
*

loss)=}�]       �	�K�fc�A�
*

lossq�p=��7�       �	��fc�A�
*

loss�}G={��C       �	���fc�A�
*

loss��=����       �	��fc�A�
*

loss�&�=?��K       �	GU�fc�A�
*

loss���<�x l       �	���fc�A�
*

lossWzZ=x��       �	Ҏ�fc�A�
*

loss]=/�Ma       �	�#�fc�A�
*

loss��<��Y       �	^��fc�A�*

loss�u>=�}       �		S�fc�A�*

loss�X>S��       �	���fc�A�*

loss�<T=��W       �	O��fc�A�*

loss=
=nC�       �	+2�fc�A�*

loss�^�<�K�       �	���fc�A�*

lossYM<���e       �	?r�fc�A�*

loss�)�="�7�       �	 �fc�A�*

loss1=ꝟ�       �	���fc�A�*

loss:J5> ���       �	�>�fc�A�*

lossx�.>A��c       �	���fc�A�*

lossJ�>1�J�       �	���fc�A�*

loss_&%>����       �	Y0�fc�A�*

loss��r<��LZ       �	m��fc�A�*

loss��<l�_       �	0f�fc�A�*

loss�چ=�8�       �	��fc�A�*

lossw�S=�w       �	��fc�A�*

loss�,C<��f=       �	�r�fc�A�*

lossEI�=�H�       �	��fc�A�*

loss�XY=7���       �	t��fc�A�*

lossH��<"�       �	�3�fc�A�*

losswja<r���       �	���fc�A�*

lossׄ=H,�       �	�z�fc�A�*

loss��b<�C�       �	p�fc�A�*

loss���<���t       �	M��fc�A�*

loss`�I=Z"�       �	;�fc�A�*

loss�wT=��H       �	q��fc�A�*

loss#��=+�Z       �	��fc�A�*

lossˎ=�(��       �	�"�fc�A�*

loss�]�<:&�       �	½�fc�A�*

loss?A�=�Q�k       �	0��fc�A�*

loss׸�=o�7�       �	 ��fc�A�*

loss�&H<m�D       �	�i�fc�A�*

loss���<}D��       �	$b�fc�A�*

lossѕ�=/�U       �	�v�fc�A�*

loss�5K=Ȳv       �	v�fc�A�*

lossM�>��z       �	u��fc�A�*

loss�V=�7z�       �	�j�fc�A�*

loss6RW=J�s�       �	)�fc�A�*

loss��<�i�       �	���fc�A�*

loss���=G2�9       �	h=�fc�A�*

loss�@k=XY�       �	3��fc�A�*

loss)K�=��IB       �	$~�fc�A�*

loss��0='8F�       �	��fc�A�*

loss��?=����       �	��fc�A�*

loss�b='�(�       �	��fc�A�*

lossqQ^<�U��       �	V(�fc�A�*

lossb�;13no       �	���fc�A�*

loss�=�<�x��       �	^�fc�A�*

loss3L8=˸�.       �	��fc�A�*

loss��A=� ��       �	��fc�A�*

loss���<O��       �	J'�fc�A�*

loss���=93�!       �	D��fc�A�*

losswP�=+��       �	�d�fc�A�*

loss3o={5b        �	�
 fc�A�*

loss�3%<�/E       �	�� fc�A�*

loss���<5�2�       �	r5fc�A�*

loss� =���       �	 �fc�A�*

loss��<��       �	�]fc�A�*

loss���<o���       �	��fc�A�*

loss@t=��       �	߉fc�A�*

lossV�7>���L       �	 *fc�A�*

loss*2`=�l/       �	X�fc�A�*

loss��(=�
.�       �	ifc�A�*

loss�=0�       �	xfc�A�*

loss�i�<=U
Q       �	�fc�A�*

loss1��<��c�       �	~Ufc�A�*

loss�M�=rްa       �	�fc�A�*

lossa��=���       �	4�fc�A�*

loss,=#<o��       �	� 	fc�A�*

lossl�;Ҁ�	       �	5�	fc�A�*

loss|3*=|5λ       �	�o
fc�A�*

loss}��<�j#x       �	�fc�A�*

lossN�;�.X�       �	x�fc�A�*

loss���;�~       �	NGfc�A�*

lossmW�;z�8�       �	�fc�A�*

loss�~�<�YM�       �	xfc�A�*

loss���=�q       �	�	fc�A�*

lossM�=w��I       �	��fc�A�*

loss*�X=6���       �	Ufc�A�*

loss×<}V9�       �	b�fc�A�*

loss���<�G|       �	��fc�A�*

loss���;Ǿ0(       �	3fc�A�*

loss�YA<ٻ��       �	�fc�A�*

lossz��=R?��       �	x`fc�A�*

loss�3�;Q'?       �	�fc�A�*

lossd��;K֖       �	ٳfc�A�*

lossl��<�s��       �	Lfc�A�*

loss�Й=�"��       �	>�fc�A�*

lossH*�<*�."       �	fc�A�*

loss�(>>Q��       �	zfc�A�*

loss��>Y�I#       �	��fc�A�*

loss�6�=�/EQ       �	@fc�A�*

loss��~<W       �	b�fc�A�*

lossW��<� �       �		mfc�A�*

loss#�;���       �	� fc�A�*

loss�/�=,�4z       �	+�fc�A�*

losstU�=�И       �	�Lfc�A�*

loss�D�=T}n       �	k�fc�A�*

loss�:�;`�S�       �	;�fc�A�*

loss}��=Շ�!       �	.fc�A�*

loss�D=nZ �       �	�fc�A�*

lossĚ�<�9��       �	�efc�A�*

lossaH�=�Y��       �	�fc�A�*

loss��Y=۳X       �	W�fc�A�*

loss�=g��       �	�3fc�A�*

loss�ڔ<��\9       �	G�fc�A�*

loss�ȍ<�!�       �	` fc�A�*

loss��>t9��       �	j� fc�A�*

loss��S<ur0       �	��!fc�A�*

loss���< ���       �	c*"fc�A�*

lossqI'=_��       �	<�"fc�A�*

loss .4<��@�       �	�\#fc�A�*

lossM#N<H�Y       �	��#fc�A�*

loss;�>���/       �	��$fc�A�*

lossj`=�!

       �	*%fc�A�*

loss��<��i       �	��%fc�A�*

loss	s=m=�       �	O&fc�A�*

loss��=]��       �	|�&fc�A�*

loss(pM=]��       �	ۈ'fc�A�*

lossox�=���       �	~(fc�A�*

losstj`=���       �	6�(fc�A�*

loss+\=÷��       �	FB)fc�A�*

loss��<Jg�       �	��)fc�A�*

loss��<;��#       �	�v*fc�A�*

loss$�0;b        �	�+fc�A�*

lossڬ=-5�       �	^�+fc�A�*

loss;�=���T       �	n1,fc�A�*

lossfG�<�	�       �	�,fc�A�*

loss	-<��w1       �	�e-fc�A�*

loss,Jb<a�       �	~�-fc�A�*

loss��=c���       �	*�.fc�A�*

loss��X<arH�       �	�//fc�A�*

loss�hw<�lc$       �	�/fc�A�*

loss�I=pu�       �	�a0fc�A�*

loss�&&=I��a       �	�1fc�A�*

loss�;�<��Ć       �	і1fc�A�*

loss��=�ҏ       �	^/2fc�A�*

loss�3�=F��       �	��2fc�A�*

loss�uA<�kO�       �	�\3fc�A�*

loss-+5=E�,�       �	�4fc�A�*

loss �!=�tC       �	��4fc�A�*

loss	�<�       �	�]5fc�A�*

loss�H�;�&�       �	��5fc�A�*

loss��<k�{9       �	s�6fc�A�*

lossŒ<X:�       �	�67fc�A�*

loss�<�Ґ       �	|�7fc�A�*

loss���=V�       �	h8fc�A�*

loss��E;p���       �	q9fc�A�*

loss�*�<�.?       �	R�9fc�A�*

loss��=�        �	e5:fc�A�*

loss�	<��Y[       �	��:fc�A�*

loss�
�=u�2�       �	�i;fc�A�*

loss��=Ũ�1       �	C<fc�A�*

loss��<��       �	,�<fc�A�*

lossx0=	lM�       �	�7=fc�A�*

loss#��=\��       �	^�=fc�A�*

loss��=y��       �	x|>fc�A�*

loss��=d���       �	�K?fc�A�*

loss�>`�+�       �	��?fc�A�*

loss�<L�|U       �	�@fc�A�*

lossQn�<�`I�       �	3Afc�A�*

loss�qz<�x�       �	!�Afc�A�*

lossփ�</ �       �	�KBfc�A�*

loss���=��|       �	��Bfc�A�*

loss��=���M       �	��Cfc�A�*

lossNg�<ܘ�       �	#Dfc�A�*

loss�o)=�aJ       �	i�Dfc�A�*

loss
6>�2|u       �	hEfc�A�*

loss�8�<P���       �	<�Efc�A�*

loss�.�<�<'       �	��Ffc�A�*

lossC�=�s�       �	�Gfc�A�*

loss�M�=�R(+       �	��Gfc�A�*

loss�
F=���       �	�FHfc�A�*

lossD_=�x�       �	��Hfc�A�*

loss�d�=�!eF       �	�iIfc�A�*

loss�^e<��%�       �	��Ifc�A�*

loss�P<MsWu       �	��Jfc�A�*

loss���=ӱ��       �	�iKfc�A�*

lossY�=�}       �	yLfc�A�*

lossԾK=S7�4       �	��Lfc�A�*

loss�Y =L*W       �	�1Mfc�A�*

loss[�><4&*�       �	�Mfc�A�*

loss�*=�vUF       �	8�Nfc�A�*

loss�}1=�Y_�       �	�<Ofc�A�*

lossQk�=���N       �	��Ofc�A�*

loss�(=����       �	kPfc�A�*

lossØ�<�F��       �	t
Qfc�A�*

losst�>��<       �	�eRfc�A�*

lossOJ<-�       �	�Rfc�A�*

loss�I�<�Q       �	��Sfc�A�*

lossm!�<��p       �	�6Tfc�A�*

loss�t=���       �	��Tfc�A�*

loss�M�;!=��       �	jUfc�A�*

loss!ݢ<�l$       �	�	Vfc�A�*

loss�X�=��B       �	��Vfc�A�*

loss
�;��<�       �	 9Wfc�A�*

loss�� >���       �	M�Xfc�A�*

loss�ct<�g2       �	xcYfc�A�*

loss|q=�M�e       �	�HZfc�A�*

lossэ�=�r�I       �	��Zfc�A�*

loss�C�=�=c       �	u�[fc�A�*

lossp�=�m�}       �	6u\fc�A�*

loss��$=˱�       �	b�]fc�A�*

lossw�>s&�       �	$^fc�A�*

loss�=��Ĝ       �	]�^fc�A�*

lossx"�=�4t       �	�^_fc�A�*

lossf-�=���       �	�_fc�A�*

loss��>�vh�       �	c�`fc�A�*

loss`�<K��       �	�>afc�A�*

loss]%�<�U       �	%�afc�A�*

loss3}s=��       �	�bfc�A�*

loss�e=��yw       �	�*cfc�A�*

loss���<L���       �	!�cfc�A�*

loss|8�=��n�       �	�jdfc�A�*

loss� �<�%       �	�efc�A�*

loss��>o��       �	��efc�A�*

lossJ�=ĕ�       �	Wffc�A�*

loss��!=_׷�       �	��ffc�A�*

loss��<���       �	�gfc�A�*

lossLe�=�o�@       �	]2hfc�A�*

loss��F=�A?,       �	d�hfc�A�*

lossu��<E�Ք       �	�bifc�A�*

lossz�<�Y��       �	*�ifc�A�*

loss���<���       �	"�jfc�A�*

loss8 Q<Aئ       �	Gkfc�A�*

lossAa�<$y��       �	j�kfc�A�*

loss>k=N�:       �	tylfc�A�*

loss!�<B"�t       �	mfc�A�*

loss��<\M��       �	g�nfc�A�*

lossE4�<DG6       �	�ofc�A�*

loss�;r^�7       �	�Fpfc�A�*

losss�z<p-�8       �	Z�pfc�A�*

loss%M=�Xv�       �	��qfc�A�*

loss��<����       �	�Hrfc�A�*

loss���=43+-       �	b�rfc�A�*

loss  �=T���       �	��sfc�A�*

lossI&�;��r       �	q9tfc�A�*

loss���=��Z       �	X�tfc�A�*

loss:>�<����       �	P�ufc�A�*

loss�҅<F_vx       �	�(vfc�A�*

loss�n<�       �	k*wfc�A�*

loss��9==�T       �	��wfc�A�*

loss��L=�.Z       �	�xfc�A�*

lossVާ;��Uh       �	�,yfc�A�*

loss�b;�1�	       �	��yfc�A�*

lossyg=L���       �	r�{fc�A�*

loss�(>@�o�       �	�$|fc�A�*

lossk�=��5       �	E�|fc�A�*

lossE�<+;B�       �	�V}fc�A�*

loss�=Y$�r       �	$�}fc�A�*

loss�Ѹ;\4�       �	)�~fc�A�*

loss�9�<��j       �	�/fc�A�*

loss%�1<�,�D       �	�fc�A�*

loss�l�;���5       �	�m�fc�A�*

loss�m.<Ɣj       �	��fc�A�*

lossI�=-֘`       �	��fc�A�*

loss�D;G�8�       �	#N�fc�A�*

loss�+u<��       �	��fc�A�*

lossh�<b��       �	'��fc�A�*

lossw�=�8       �	�!�fc�A�*

loss}�>�4�H       �	���fc�A�*

loss�^�:o*       �	���fc�A�*

loss�>�oq[       �	�5�fc�A�*

loss��=�:6�       �	Άfc�A�*

loss���=����       �	bf�fc�A�*

loss
�)=ZN       �	�fc�A�*

loss�$<�<�       �	���fc�A�*

loss�l <       �	|�fc�A�*

loss�)<b��O       �	���fc�A�*

loss�n�;���       �	0�fc�A�*

loss�1�<�p�p       �	$ԋfc�A�*

lossH�_>}*�       �	/m�fc�A�*

loss� >��       �	 
�fc�A�*

losst��=��w�       �	�ڎfc�A�*

loss��p<L��Z       �	�|�fc�A�*

lossr��=����       �	��fc�A�*

lossM��=l��       �	���fc�A�*

loss�y�<3�\k       �	�`�fc�A�*

loss�A;<��5       �	��fc�A�*

loss2��</�q>       �	���fc�A�*

loss���<��T       �	�,�fc�A�*

loss��V=Uʽ�       �	Гfc�A�*

loss���<�L�_       �	Kv�fc�A�*

loss���=w!�       �	��fc�A�*

lossV<��       �	ȴ�fc�A�*

lossV�=T���       �	�O�fc�A�*

loss5<{��       �	(�fc�A�*

loss ��<�Ϯ       �	C��fc�A�*

loss��=�V�       �	+/�fc�A�*

loss�u�;���       �	y̘fc�A�*

loss8�A=��h�       �	�b�fc�A�*

loss=�>>�1       �	��fc�A�*

loss�(=���t       �	>��fc�A�*

loss7^<��=       �	�I�fc�A�*

loss8��<�$��       �	�O�fc�A�*

loss,@�;�5�       �	l�fc�A�*

loss6[5=��w       �	��fc�A�*

loss�X<p�
A       �	��fc�A�*

loss�D2=Pճ�       �	k(�fc�A�*

lossm�<.��       �	�fc�A�*

loss�vs=\e�Z       �	���fc�A�*

loss�U�=�
��       �	���fc�A�*

loss��(<F��       �	�S�fc�A�*

loss�[3=N�Y       �	��fc�A�*

loss �<�c�       �	 ��fc�A�*

loss���=B)�       �	s/�fc�A�*

loss��<�	��       �	dʤfc�A�*

loss���=�B.�       �	�l�fc�A�*

loss��=V��|       �	\�fc�A�*

lossEx�<q�d       �	���fc�A�*

loss���<踉1       �	�Z�fc�A�*

loss� �=���.       �	���fc�A�*

loss[-�<�]x       �	���fc�A�*

loss���<��YB       �	���fc�A�*

loss1	<b�G       �	b1�fc�A�*

loss=.�<�5�       �	!̪fc�A�*

loss�t
=W���       �	�c�fc�A�*

loss���<�]7�       �	�fc�A�*

loss,��<���       �	젬fc�A�*

loss��<��]#       �	�8�fc�A�*

loss��=�g        �	�y�fc�A�*

loss-�$<�O�I       �	��fc�A�*

lossif�<47�       �	�D�fc�A�*

loss�!Q=�ہ       �	G�fc�A�*

loss3ܛ;OsJ       �	��fc�A�*

loss���9l��\       �	òfc�A�*

loss�%�<�$�2       �	ٳfc�A�*

loss��t:�)Q�       �	� �fc�A�*

loss�?�:ܥ�       �	���fc�A�*

loss"�<MH�       �	���fc�A�*

lossR7�;O��i       �	9&�fc�A�*

loss}U�;Ga!       �	|�fc�A�*

lossҭi;���H       �	�(�fc�A�*

loss��9���i       �	���fc�A�*

loss�ͣ9� t       �	WZ�fc�A�*

lossbw
<�M�       �	��fc�A�*

loss�Si<]�rN       �	���fc�A�*

loss*��;X		$       �	4�fc�A�*

loss���;@}
�       �	�μfc�A�*

lossf��<k�1       �	�h�fc�A�*

lossedj>u��d       �	��fc�A�*

lossq <�>]�       �	й�fc�A�*

loss��f>��'�       �	6V�fc�A�*

loss�(�=��O       �	a��fc�A�*

loss�=I�       �	���fc�A�*

loss�:�;�Ŧ�       �	�:�fc�A�*

loss�=� 1�       �	���fc�A�*

loss�4�=��F       �	~p�fc�A�*

lossc΁=Q
k�       �	D�fc�A�*

lossi�%=/{       �	���fc�A�*

loss��=K;)�       �	�M�fc�A�*

lossC=�N�       �	[��fc�A�*

lossz�=�o@       �	��fc�A�*

lossWU�<S�s�       �	�@�fc�A�*

loss��)=|�/       �	���fc�A�*

loss?�`=�s0       �	���fc�A�*

loss&�k=��_       �	��fc�A�*

lossHA�=>7�c       �	p��fc�A�*

loss{�<N�Q�       �	DO�fc�A�*

loss�-M=���       �	�!�fc�A�*

loss��P<d��       �	���fc�A�*

loss��<�_�       �	�N�fc�A�*

loss-zb<�J2�       �	�fc�A�*

loss[(h=��G       �	��fc�A�*

loss��;�H*!       �	.9�fc�A�*

loss�K�<���       �	��fc�A�*

lossjf< 4�       �	Dk�fc�A�*

lossӇ==����       �	��fc�A�*

lossjZ�<쀘R       �	J��fc�A�*

loss��=����       �	eQ�fc�A�*

lossb�>��p        �	���fc�A�*

loss
�<Vq��       �	���fc�A�*

loss�8�</���       �	R��fc�A�*

loss�t~=p�q       �	u��fc�A�*

loss���;^�w       �	�	�fc�A�*

lossȧ+<,�       �	��fc�A�*

lossv><���       �	�P�fc�A�*

loss�>=��9       �	�[�fc�A�*

lossf�<;͑�       �	!�fc�A�*

loss��+=�|h�       �	ߦ�fc�A�*

loss��q=< �       �	JE�fc�A�*

loss���<PF�J       �	���fc�A�*

loss�*�<���       �	��fc�A�*

loss�t�<���       �	��fc�A�*

loss��=����       �	���fc�A�*

loss��<X�}        �	�R�fc�A�*

loss���=)w        �	���fc�A�*

loss�Y>����       �	΋�fc�A�*

loss,M =}Q5�       �	)#�fc�A�*

loss�=�MO       �	#��fc�A�*

lossT8n<����       �	�S�fc�A�*

loss�m<�&�       �	���fc�A�*

loss(E�<(:�       �	JA�fc�A�*

loss�@�<uҘ�       �	���fc�A�*

loss�TM=h�ڸ       �	Pm�fc�A�*

loss���<[{��       �	��fc�A�*

lossJ.=Fa${       �	���fc�A�*

lossn��<�ӄ       �	E+�fc�A�*

loss@�k=�?��       �	���fc�A�*

lossM��<�jq       �	�`�fc�A�*

loss�i�=x/o       �	8��fc�A�*

loss{��=[g�       �	a��fc�A�*

lossjG<��       �	_%�fc�A�*

loss׀@<$��       �	���fc�A�*

loss]�>��=�       �	Q�fc�A�*

loss��=���       �	���fc�A�*

loss�K�<�Wr=       �	Fz fc�A�*

loss��)=$�z       �	fc�A�*

loss.Z;�]��       �		�fc�A�*

loss)=֟�       �	�Gfc�A�*

lossl1P<�p�       �	%�fc�A�*

loss轟=���       �	�fc�A�*

loss��<�FM       �	fc�A�*

lossj��=wɸ       �	�fc�A�*

loss�)n<�ܸa       �	QMfc�A�*

loss�d_=u�f       �	=�fc�A�*

loss�i@=j��       �	�fc�A�*

loss3�<X8U�       �	R(fc�A�*

loss�|S<�v�6       �	��fc�A�*

loss���<'r�        �	�Wfc�A�*

loss41U<f*��       �	�fc�A�*

loss�q=ړ�       �	
fc�A�*

loss�l�<�dk       �	��
fc�A�*

lossN)�<��       �	�cfc�A�*

loss�g�<)$z�       �	Bfc�A�*

loss�#�=Nu0C       �	3�fc�A�*

loss�=∼�       �	�fc�A�*

loss�ƛ=��Ƞ       �	�,fc�A�*

loss��<6+g       �	��fc�A�*

loss�4�=2D��       �	OZfc�A�*

lossXb�=��       �	��fc�A�*

lossf�>���       �	w�fc�A�*

lossC�;c>�*       �	�*fc�A�*

loss��2=o���       �	1�fc�A�*

loss-
e<zS�       �	Byfc�A�*

loss���<��m�       �	�fc�A�*

lossa�=��
�       �	��fc�A�*

loss�=F/a�       �	B^fc�A�*

loss���;� ��       �	�fc�A�*

loss_j�=���       �	��fc�A�*

loss�"[;�߃�       �	,)fc�A�*

loss ~�;��	       �	Y0fc�A�*

loss�%�<��8P       �	G�fc�A�*

loss��^=�آm       �	4ffc�A�*

loss��K<t�       �	�fc�A�*

loss�w�=�y D       �	��fc�A�*

loss��D=?�9       �	�bfc�A�*

loss	�8;d��       �	��fc�A�*

loss��u;��:�       �	Ύfc�A�*

lossx�E;G��]       �	(fc�A�*

lossZ�=HNW�       �	Ӽfc�A�*

lossĂ�=r�8�       �	�Qfc�A�*

loss&��=�m�       �	�fc�A�*

loss�U2=5��`       �	!fc�A�*

loss��;�G�       �	�fc�A�*

lossj�V=����       �	M fc�A�*

lossӱ<��=       �	V� fc�A�*

loss�И< 7[       �	֏!fc�A�*

lossH&=ߦX)       �	)&"fc�A�*

loss�M=��!       �	�"fc�A�*

lossr�V=�s       �	W#fc�A�*

lossI��=�)��       �	�#fc�A�*

lossMV�<�b�       �	ܠ$fc�A�*

loss��=���Y       �	�:%fc�A�*

loss<5!<%�f       �	F�%fc�A�*

lossS�=���       �	�o&fc�A�*

lossJ��<�~F�       �	�'fc�A�*

loss�h<hw��       �	X�'fc�A�*

loss�@�<��|       �	|D(fc�A�*

loss���<��`l       �	��(fc�A�*

loss���<G��T       �	�|)fc�A�*

lossJ��<�P�       �	+1*fc�A�*

losst��=w���       �	��*fc�A�*

loss�l�=]�,       �	��,fc�A�*

loss���=�<d�       �	<.-fc�A�*

lossF�;cֈ�       �	�.fc�A�*

loss�M�<�/��       �	h�.fc�A�*

loss��<��2       �	W�/fc�A�*

loss���=�1J\       �	�0fc�A�*

lossjVM<�Ѿ�       �	ͫ1fc�A�*

loss��0<}@��       �	��2fc�A�*

loss�f�;} 1/       �	��3fc�A�*

loss���<� ^       �	�t4fc�A�*

loss���=��       �	�5fc�A�*

loss���=C�       �	�	6fc�A�*

loss�l(<�p��       �	
7fc�A�*

loss�s<<`n�       �	�7fc�A�*

loss��u;ܞb�       �	�|8fc�A�*

loss�4�=_J       �	�9fc�A�*

loss��<<���       �	��9fc�A�*

loss�n�=z\��       �	�G:fc�A�*

loss	P�<R�'W       �	��:fc�A�*

lossrW�;8�2       �	�t;fc�A�*

loss=R <��w�       �	I<fc�A�*

loss��=-%]       �	ձ<fc�A�*

loss��X=���l       �	jK=fc�A�*

loss���<7["       �	��=fc�A�*

loss�n~=] Xb       �	N�>fc�A�*

loss@O�<��ڍ       �	j?fc�A�*

loss���<� 'l       �	.�?fc�A�*

loss�Ā<�%`       �	�@fc�A�*

loss��\<�]^A       �	lCAfc�A�*

loss'�<�㍭       �	�Bfc�A�*

loss|K�<SI��       �	�JCfc�A�*

losslz�<��#       �	��Cfc�A�*

loss 3;꣆�       �	�Dfc�A�*

loss��]=��r       �	�Efc�A�*

loss�I<ti��       �	 �Efc�A�*

loss��d<�ݤ�       �	PRFfc�A�*

loss<;=j�-       �	��Ffc�A�*

lossf�=��?�       �	�Gfc�A�*

lossA �<��       �	w.Hfc�A�*

loss���<��O�       �	$�Hfc�A�*

loss��=B{��       �	)vIfc�A�*

loss�==3���       �	kJfc�A�*

loss@�M=v�       �	��Jfc�A�*

loss/��< �"�       �	HKfc�A�*

loss�+=���u       �	m�Kfc�A�*

lossvh5<k2�       �	N{Lfc�A�*

loss0<p��L       �	�Mfc�A�*

lossލ<z}�8       �	�Mfc�A�*

loss1�=���       �	:ZNfc�A�*

loss?4<�,��       �	l�Ofc�A�*

loss���<X��       �	(*Pfc�A�*

loss��=ߩ�       �	��Pfc�A�*

lossx�< v��       �	�fQfc�A�*

loss�Dv=\\��       �	5'Rfc�A�*

loss���<�E2�       �	S�Rfc�A�*

loss�'=����       �	nSfc�A�*

loss{7t<�>4�       �	�Tfc�A�*

loss��R<uT��       �	,�Tfc�A�*

losseʢ:'�       �	:\Ufc�A�*

loss��K<��K�       �	z�Ufc�A�*

loss%��<��`.       �	M�Vfc�A�*

loss�y�<��H       �	�?Wfc�A�*

loss�N<��J
       �	B�Wfc�A�*

loss(��<C�E       �	֏Xfc�A�*

loss�y<��Յ       �	8,Yfc�A�*

loss?��< "�       �	��Yfc�A�*

lossJ@�<ɣ"       �	�{Zfc�A�*

loss(G';��Ϳ       �	�\[fc�A�*

loss$~=��6       �	��[fc�A�*

lossRw;�       �	J�\fc�A�*

loss�q�<����       �	��]fc�A�*

loss�9=!�       �	C7^fc�A�*

loss�I< !A�       �	��^fc�A�*

loss}�=0�?�       �	r_fc�A�*

lossW��;�Z�d       �	`fc�A�*

loss��;?��       �	S�`fc�A�*

loss7�=&)�h       �	�Zafc�A�*

loss3��=��ϵ       �	9
bfc�A�*

loss��r<���i       �	W�bfc�A�*

loss;^�;)�-       �	�Rcfc�A�*

loss
�:��"       �	��cfc�A�*

lossʹ=�g       �	ˡdfc�A�*

loss��9<�W�       �	Defc�A�*

loss#��=��dN       �	`�efc�A�*

lossɦ�<�ٙJ       �	�ffc�A�*

loss��=:8:�       �	f/gfc�A�*

loss~=�ƣ       �	��gfc�A�*

loss�s<|D       �	�ohfc�A�*

loss��5<��/�       �	ifc�A�*

loss��=�+y       �	o�ifc�A�*

loss3Xp;y3��       �	;jfc�A�*

loss� <��.       �	9�jfc�A�*

loss���=̠Ժ       �	�okfc�A�*

loss}�
=��g�       �	�lfc�A�*

lossO�*=[F6=       �	v�lfc�A�*

loss�x�=�|a�       �	 Emfc�A�*

lossQ��=�3�       �	��mfc�A�*

loss���=wD�       �	ynfc�A�*

loss�1�;e���       �	�ofc�A�*

lossڤT;c	?$       �	��ofc�A�*

loss�Y=#Op�       �	?pfc�A�*

loss(֫;��+       �	�pfc�A�*

loss��<��l�       �	�tqfc�A�*

loss�/
<})vQ       �	�
rfc�A�*

loss}x�;�>OA       �	I�rfc�A�*

loss���<��uB       �	�Esfc�A�*

loss$�"=����       �	��sfc�A�*

loss��=i�D       �	.�tfc�A�*

loss �<�ߠR       �	�6ufc�A�*

loss���=�&�       �	��ufc�A�*

loss}<��       �	A}vfc�A�*

loss��/<\(U%       �	�wfc�A�*

loss�8�;�$Wq       �	��wfc�A�*

loss3��;���       �	�oxfc�A�*

loss�J�;�\�       �	�yfc�A�*

loss��*>���y       �	��yfc�A�*

loss�H9=���m       �	[zfc�A�*

loss�?==�A?+       �	�{fc�A�*

loss�Z�<��S�       �	Χ{fc�A�*

loss��u=��,M       �	}X|fc�A�*

loss�;8_��       �	�}fc�A�*

loss
��;,��       �	P�}fc�A�*

loss�{=�6��       �	"T~fc�A�*

loss �};�`��       �	��~fc�A�*

loss�e�<���       �	>�fc�A�*

loss�R�=��       �	.X�fc�A�*

loss�;4=��6�       �	Z��fc�A�*

loss��=\�Į       �	���fc�A�*

loss�P�<����       �	�:�fc�A�*

lossaҼ<��uL       �	߂fc�A�*

lossFs�;��u�       �	�{�fc�A�*

lossHk=+p�       �	��fc�A�*

loss�;�棹       �	V��fc�A�*

lossr�0</U�       �	�W�fc�A�*

lossi��=�ɐ       �	G��fc�A�*

lossػ>B��       �	멆fc�A�*

loss���=��9"       �	�G�fc�A�*

loss�5�=�9�p       �	��fc�A�*

loss�"z=��X�       �	E��fc�A�*

loss�^s<��H�       �	�&�fc�A�*

loss��V<1l�       �	�؉fc�A�*

loss�/<���       �	�v�fc�A�*

lossc�=��?�       �	��fc�A�*

loss3��;Z���       �	=��fc�A�*

loss���<�yb       �	�p�fc�A�*

loss6v�=K��       �	
�fc�A�*

loss�y�=v��j       �	��fc�A�*

loss�K)=��       �	�<�fc�A�*

lossiC=��y�       �	�ێfc�A�*

loss$��;�Rl�       �	&p�fc�A�*

loss��j<����       �	��fc�A�*

loss!d=��#       �	���fc�A�*

loss;L=���#       �	a3�fc�A�*

loss�<	0,       �	�֑fc�A�*

loss���<it       �	�p�fc�A�*

loss���< �H�       �	��fc�A�*

loss��E=o��       �	���fc�A�*

loss ��=�KX;       �	�2�fc�A�*

loss�.�<���       �	.Ȕfc�A�*

loss�u<YzK       �	Tq�fc�A�*

loss�^S=���       �	�
�fc�A�*

loss��<�Jy       �	j��fc�A�*

loss8�=�)��       �	�?�fc�A�*

loss��A=`�q       �	�ܗfc�A�*

loss���<�P��       �	yu�fc�A�*

loss)�;�#G>       �	��fc�A�*

loss,��;9(��       �	Ĵ�fc�A�*

lossT��<����       �	wM�fc�A�*

losss�
=�>�b       �	�fc�A�*

loss�=�#        �	�|�fc�A�*

lossO�=jILF       �	��fc�A�*

lossJ4e<+�4       �	Ͻ�fc�A�*

loss	�9�,��       �	\T�fc�A�*

loss�:�;8p��       �	��fc�A�*

lossX'u<o�;�       �	���fc�A�*

loss���<����       �	��fc�A�*

loss`ڴ;���       �	��fc�A�*

loss��">Mͨ�       �	�G�fc�A�*

loss֕=��y�       �	'ܠfc�A�*

loss��N=&�p.       �	2r�fc�A�*

loss��e<�a=J       �	�	�fc�A�*

loss�j=��RS       �	|��fc�A�*

loss_�;���J       �	�6�fc�A�*

loss��%=��J       �	<ݣfc�A�*

loss�)=�@:�       �	u�fc�A�*

loss�=�OR       �	JC�fc�A�*

lossĽ�<)`%n       �	��fc�A�*

loss
�]=C*       �	���fc�A�*

loss!O�=O��       �	*�fc�A�*

lossSw=I���       �	Yħfc�A�*

loss�k<��I3       �	�f�fc�A�*

loss��=����       �	 �fc�A�*

loss���=����       �	��fc�A�*

loss|��=���       �	2�fc�A�*

loss��=���       �	:˪fc�A�*

loss�D<zF�j       �	J_�fc�A�*

losscV�<� '       �	4��fc�A�*

loss�6N=M��S       �	N��fc�A�*

loss�W�=eG       �	�;�fc�A�*

losssk�:�H�B       �	w��fc�A�*

loss.��;s�K       �	���fc�A�*

loss �-=�[�       �	0�fc�A�*

lossB<Q���       �	ڎ�fc�A�*

lossB:	<��       �	�U�fc�A�*

loss͓�=�~Nr       �	W�fc�A�*

loss,�=��*I       �	�$�fc�A�*

loss8�<mw�W       �	j��fc�A�*

loss���<��V{       �	���fc�A�*

loss*j<ވ��       �	9&�fc�A�*

loss#��:�q�       �	ǽ�fc�A�*

loss8�0=b�v�       �	�^�fc�A�*

loss�<�a*�       �	"��fc�A�*

loss8];]��       �	�-�fc�A�*

lossӱ<�n�       �		ĸfc�A�*

loss5��=�o�g       �	O\�fc�A�*

loss���<$j.       �	+��fc�A�*

loss�#�=M���       �	�fc�A�*

loss�<���_       �	�<�fc�A�*

loss�:=D�t�       �	�ڻfc�A�*

lossX!=T�j3       �	*r�fc�A�*

lossF�<a��@       �	��fc�A�*

loss	��<��d�       �	G��fc�A�*

loss�4�=��Z       �	��fc�A�*

loss�=��Q_       �	�!�fc�A�*

loss���=p�"T       �	Oʿfc�A�*

lossD�;�ۇ       �	�k�fc�A�*

lossBf
=�1       �	��fc�A�*

lossE��<�d��       �	9��fc�A�*

loss��=M>V       �	`[�fc�A�*

loss�<V'I       �	'j�fc�A�*

loss��<;�Wk       �	�fc�A�*

loss6�=�`�       �	&��fc�A�*

loss� �<FsU�       �	AH�fc�A�*

loss�N<��       �	���fc�A�*

loss�!�=WPu\       �	Ҍ�fc�A�*

loss}�<^YB�       �	J$�fc�A�*

lossfW=��w�       �	~��fc�A�*

lossA��=�pDn       �	���fc�A�*

loss�]<k��^       �	�e�fc�A�*

lossϧB<�X�       �	��fc�A�*

lossV#�=Jy!�       �	��fc�A�*

loss�ѐ<4*�j       �	]�fc�A�*

loss֠-=BN       �	@O�fc�A�*

lossq<�C�<       �	m��fc�A�*

loss�g�=��+�       �	]��fc�A�*

loss�{x=6�       �	�=�fc�A�*

loss��=U8�       �	���fc�A�*

loss-�=��"       �	���fc�A�*

loss#p,=w�R�       �	�G�fc�A�*

loss�=g�W       �		�fc�A�*

loss4�t<14C�       �	k��fc�A�*

lossWw�<M�       �	zq�fc�A�*

loss?��;^a��       �	�	�fc�A�*

lossF�=,�6       �	Z��fc�A�*

loss
/A<�%z       �	�H�fc�A�*

loss�W�<=F��       �	���fc�A�*

lossA�< �Mx       �	�'�fc�A�*

loss���=IJ��       �	���fc�A�*

lossE:�:	)�O       �	'l�fc�A�*

loss�a�;fdsL       �	S>�fc�A�*

loss�(=��ܫ       �	���fc�A�*

loss=+�=�@       �	Y��fc�A�*

lossZ� =M��       �	�'�fc�A�*

lossL��=��6�       �	���fc�A�*

lossE��=��y�       �	O]�fc�A�*

loss�Z`<>25       �	�o�fc�A�*

loss)��=���       �	O�fc�A�*

loss�s=�[�n       �	���fc�A�*

loss���=�B>�       �	<�fc�A�*

loss��<��       �	V��fc�A�*

loss��<��w       �	�l�fc�A�*

loss�ڝ=���       �	��fc�A�*

lossO[�<�V_}       �	I��fc�A�*

lossn��<�˛�       �	�;�fc�A�*

loss��<14@j       �	���fc�A�*

lossqu<U�u�       �	�k�fc�A�*

loss�'�<$/       �	�q�fc�A�*

loss�^p;\��m       �	��fc�A�*

lossW�e=w���       �	���fc�A�*

loss��C=d8"       �	r6�fc�A�*

loss1��=R1H�       �	���fc�A�*

loss�n<��K�       �	d�fc�A�*

loss�i�=���       �	? �fc�A�*

loss�m�=x'�       �	D��fc�A�*

loss
��=-�6       �	�8�fc�A�*

loss�A�<��e�       �	���fc�A�*

lossw�<��l�       �	)v�fc�A�*

loss5�<�+z       �	��fc�A�*

loss�=�K�       �	��fc�A�*

losstA�=0׾�       �	�9�fc�A�*

loss�O<�z�       �	���fc�A�*

loss��e=F �       �	e�fc�A�*

loss��a<�t>       �	� �fc�A�*

lossR�<��?�       �	��fc�A�*

lossKm=(	?       �	~�fc�A�*

loss!ʮ<��b6       �	�/�fc�A�*

loss��M=��hD       �	}�fc�A�*

loss#	&=����       �	{��fc�A�*

lossi��;�5��       �	nh�fc�A�*

loss�N�=�<�       �	�fc�A�*

loss��=��       �	�U�fc�A�*

lossD�=]��r       �	f2�fc�A�*

lossMp�<t�g�       �	}	�fc�A�*

loss��<?<��       �	���fc�A�*

loss͗�=��P       �	��fc�A�*

loss*��=>Q)�       �	�]�fc�A�*

loss��<ԕ��       �	�>�fc�A�*

loss\~'=~y�X       �	�,�fc�A�*

lossE�;�f�Q       �	���fc�A�*

lossjh�<oF�n       �	ٔ�fc�A�*

loss�i�<�Q"       �	'.�fc�A�*

loss�(�;i��"       �	���fc�A�*

loss�
Z<E�       �	Yn�fc�A�*

losssL=<R�       �	��fc�A�*

loss��=�x��       �	ޮ�fc�A�*

lossO$a:�o��       �	J�fc�A�*

loss�r;�F��       �	���fc�A�*

lossJd<�ѭ       �	U. fc�A�*

lossf�<�`D�       �	�� fc�A�*

loss�<7��R       �	Ύfc�A�*

loss.�	<�z�       �	��fc�A�*

losscܩ=�ה�       �	,Ifc�A�*

loss:�p<z�0       �	��fc�A�*

loss���=ni2\       �	!wfc�A�*

loss& �<�mU0       �	�fc�A�*

loss�2=�|�       �	��fc�A�*

loss!�d;�S       �	?;fc�A�*

loss{��<�qE       �	��fc�A�*

lossq�I=Q��       �	�}fc�A�*

loss.9=�       �	�fc�A�*

loss�*c=A��+       �	5�fc�A�*

loss��<UT&?       �	�J	fc�A�*

loss&��=c��       �	��	fc�A�*

loss�=��D       �	��
fc�A�*

loss�+*=ZV�|       �	�rfc�A�*

loss��J<6rN�       �	tfc�A�*

lossr״<C0]�       �	Ҧfc�A�*

loss��=|H_�       �	�Ufc�A�*

losstn�=L�       �	�fc�A�*

loss�;F�}p       �	�fc�A�*

loss��>�Gg�       �	f�fc�A�*

loss�;ߌ��       �	��fc�A�*

loss��$>�l<O       �	�sfc�A�*

lossܜ0=�e       �	�fc�A�*

loss�0=D�`       �	&�fc�A�*

loss�=X�w#       �	LRfc�A�*

losshJ�=y`��       �	��fc�A�*

loss���<�}�       �	��fc�A�*

lossEd�=wϝP       �	�&fc�A�*

loss8�e=\�       �	��fc�A�*

loss�
=��ZO       �	qfc�A�*

loss݃<�g�t       �	�fc�A�*

loss�=j=��O       �	Ӿfc�A�*

loss��<��I�       �	dfc�A�*

loss���=U�       �	�fc�A�*

loss�Ib<Ey�K       �	�fc�A�*

loss�<��A       �	�2fc�A�*

loss��=VЄ�       �	��fc�A�*

loss���<v�'�       �	�cfc�A�*

loss��=e=<�       �	6fc�A�*

loss�~=��9�       �	��fc�A�*

loss$�=��ð       �	�/fc�A�*

lossz
�=*���       �	��fc�A�*

loss���<,?r       �	�sfc�A�*

loss��e=��A       �	�fc�A�*

loss�G>d�+�       �	��fc�A�*

lossԟ�<ũiR       �	Q fc�A�*

loss�z<ᢿ�       �	�� fc�A�*

lossz=]<�K��       �	~!fc�A�*

loss��<Jp��       �	�"fc�A�*

loss�%=�`       �	��"fc�A�*

loss��
=Ö�       �	�N#fc�A�*

loss�<��d�       �	,�#fc�A�*

loss�ۗ=�}%�       �	�$fc�A�*

loss\�A='���       �	4,%fc�A�*

loss���=Y9�]       �	��%fc�A�*

loss�)�;���       �	�`&fc�A�*

loss�<W<�b�       �	��&fc�A�*

loss!`�=Rͅ       �	�'fc�A�*

loss�zB<y��       �	�'(fc�A�*

loss,�<�浈       �	d�(fc�A�*

loss�!<���       �	$_)fc�A�*

loss��=�`       �	�*fc�A�*

loss�yK<8C\�       �	�*fc�A�*

loss�h	=�@��       �	�`+fc�A�*

loss}�K<Wp�s       �	��+fc�A�*

loss��;u{qT       �	'�,fc�A�*

loss�lZ<ba{       �	2>-fc�A�*

lossʬ�<A\�       �	��-fc�A�*

loss���={��       �	�.fc�A�*

lossܥ<��f       �	�#0fc�A�*

loss]��=�ۤ       �	��0fc�A�*

loss���=�]��       �	��1fc�A�*

loss��<���       �	Ɗ2fc�A�*

lossz.�<]�B�       �	��3fc�A�*

loss)\�:R�*1       �	�F4fc�A�*

loss�#=y�X       �	��4fc�A�*

lossH�#=��S       �	E�5fc�A�*

loss�2�=F-�       �	�6fc�A�*

lossAZ#>�
w�       �	 7fc�A�*

lossڅ=&?       �	��7fc�A�*

loss���;����       �	�e8fc�A�*

loss�|�=�S       �	�<9fc�A�*

losst�=�n�       �	��9fc�A�*

loss��<<�H�'       �	�s:fc�A�*

loss-ݡ<g�x�       �	T;fc�A�*

loss=]=��N[       �	�<fc�A�*

loss�q<:��       �	��=fc�A�*

loss�{�=�o�X       �	I�>fc�A�*

loss�<,P_�       �	�O?fc�A�*

loss�k�=�2X�       �	��?fc�A�*

loss�%2=g.`�       �	��@fc�A�*

lossH�m=�NS�       �	bAfc�A�*

loss��=�p�3       �	#,Bfc�A�*

loss!`=�H��       �	�Cfc�A�*

lossh�=���       �	�,Dfc�A�*

loss�=<C��       �	��Dfc�A�*

loss2/=�C       �	��Efc�A�*

loss��=�<�c       �	�xFfc�A�*

loss_;�={ފm       �	�Gfc�A�*

loss%&<3���       �	��Gfc�A�*

loss�g.<}ZA�       �	��Hfc�A�*

loss�]&<UP��       �	~UJfc�A�*

loss�<>c�b�       �	[BKfc�A�*

loss���=�G       �	��Kfc�A�*

lossc}<��O�       �	=�Lfc�A�*

lossm��;�5|�       �	�iMfc�A�*

loss�\�=_K�       �	�5Nfc�A�*

loss��<�k�^       �	i�Nfc�A�*

loss�n�;��k�       �	AOfc�A�*

loss��=�H�g       �	-Pfc�A�*

loss��;�z�M       �		�Pfc�A�*

loss%� =c�x.       �	�Qfc�A�*

loss��A=�yw�       �	[Rfc�A�*

lossG=�:�       �	@�Rfc�A�*

loss)%c=/�:        �	��Sfc�A�*

loss�E�<�2s       �	3NTfc�A�*

loss��<���       �	Z�Tfc�A�*

loss���<R�O�       �	��Ufc�A�*

loss!dO=�t��       �	]2Wfc�A�*

loss��=�Q��       �	O�Wfc�A�*

loss�;Ud,�       �	�dXfc�A�*

loss�1f==�ߛ       �	`Yfc�A�*

loss!�=�;׌       �	��Yfc�A�*

loss�*R=�/       �	29Zfc�A�*

loss��g=]�"       �	��Zfc�A�*

loss��<j��       �	m[fc�A�*

lossy�=���       �	�\fc�A�*

loss�D�=�X��       �	*�\fc�A�*

loss)��;�˄H       �	B]fc�A�*

loss��<X�0       �	��]fc�A�*

loss��W<�=�`       �	j�^fc�A�*

loss܊3;�[2�       �	�_fc�A�*

loss�N�<���F       �	�_fc�A�*

loss��V<�1�2       �	
�`fc�A�*

losszh;�       �	#afc�A�*

loss@l�<=��       �	��afc�A�*

loss�5�<$R�m       �	KVbfc�A�*

lossd��;��\       �	-�bfc�A�*

lossv�9?n"^       �	i�cfc�A�*

lossr��9���       �	|*dfc�A�*

lossVT�9f)��       �	��dfc�A�*

loss\".;�5�       �	t_efc�A�*

lossW�;<K�q�       �	��efc�A�*

lossm�G<���       �	5�ffc�A�*

loss��:�e        �	�/gfc�A�*

lossn�:����       �	��gfc�A�*

loss;�=�ER       �	�hhfc�A�*

loss�b-;�:       �	�ifc�A�*

loss���=�M =       �	2tjfc�A�*

loss��^=|Ŝ�       �	�kfc�A�*

loss�9�=Q��       �	��kfc�A�*

loss�B�<KT_�       �	�lfc�A�*

lossf#;��b�       �	U.mfc�A�*

loss��l=�$2       �	��mfc�A�*

loss�rk=���       �	�enfc�A�*

loss��m=3�N�       �	��nfc�A�*

loss�+8=R�̟       �	P5pfc�A�*

loss���<�R��       �	;�pfc�A�*

loss�7=����       �	z�qfc�A�*

loss��=PoĻ       �	�Brfc�A�*

loss��=p��&       �	6xsfc�A�*

loss|��<$@qv       �	�Xtfc�A�*

loss�s><���2       �	�vufc�A�*

lossln=m�2�       �	��vfc�A�*

loss��=Y��       �	@Nwfc�A�*

loss��=�h       �	�uxfc�A�*

loss���;Ħ}~       �	�Syfc�A�*

loss� �;��o�       �	k�yfc�A�*

loss�?�<�K�E       �	��zfc�A�*

lossKN�<%5"e       �	C�{fc�A�*

loss���<��<       �	�}fc�A�*

losshq�:���       �	ݘ}fc�A�*

loss�o�;�(;�       �	�6~fc�A�*

lossAȏ=~��       �	4�~fc�A�*

loss��\<�ѻO       �	Z�fc�A�*

loss׽�=
��       �	�T�fc�A�*

loss2��=paj�       �	��fc�A�*

loss�=�2j       �	]��fc�A�*

lossu�<�*�       �	K!�fc�A�*

loss��<�8��       �	=��fc�A�*

loss���;E��       �	���fc�A�*

loss�ĝ;�q��       �	�+�fc�A�*

loss�=�<�O+�       �	rfc�A�*

losslT<d��{       �	X�fc�A�*

loss��=9���       �	_�fc�A�*

loss��-=�8N       �	���fc�A�*

lossa��<��¢       �	��fc�A�*

loss`&;�N�       �	���fc�A�*

loss�(o<*G�       �	0H�fc�A�*

loss�\�<��1Y       �	!�fc�A�*

losse��<�{�~       �	M��fc�A�*

loss
��;�]hY       �	29�fc�A�*

loss�s�;�|e       �	y�fc�A�*

loss�>B=͞       �	|{�fc�A�*

lossp�<DT�X       �	��fc�A�*

lossH��;��C�       �	ŭ�fc�A�*

loss�Gk<��Υ       �	���fc�A�*

loss~=Cb       �	ࢎfc�A�*

lossX�<̅Pw       �	3��fc�A�*

loss���<�w�       �	�#�fc�A�*

loss���=��Y�       �	
��fc�A�*

loss�cg<���{       �	LT�fc�A�*

lossd<�
%       �	��fc�A�*

loss輶;d�?�       �	A�fc�A�*

loss���<�}�<       �	��fc�A�*

loss���<'�ԋ       �	���fc�A�*

loss]��=��5       �	�;�fc�A�*

lossv�<��C�       �	�ݫfc�A�*

loss��;}V//       �	u �fc�A�*

loss��"=���l       �	B��fc�A�*

lossE=�wڻ       �	M.�fc�A�*

loss�\=-o�       �	*Ǯfc�A�*

loss��B=f��f       �	Jb�fc�A�*

loss(�<�       �	p��fc�A�*

loss@�q;��       �	f��fc�A�*

loss�=G���       �	(d�fc�A�*

loss��Y<�(iT       �	���fc�A�*

loss�Ѳ<eQu       �	ŏ�fc�A�*

loss`��<N�       �	</�fc�A�*

loss��i=pb�       �	�ƴfc�A�*

loss��G<���       �	�`�fc�A�*

loss,�-=�5�       �	=�fc�A�*

lossA�<��YY       �	/��fc�A�*

loss�
�<,!�       �	X�fc�A�*

lossm��=���       �	ٰ�fc�A�*

losscO�<|&�C       �	�G�fc�A�*

lossf==7N       �	#۸fc�A�*

loss�@=g��       �	"q�fc�A�*

loss���;:���       �	B�fc�A�*

loss�<�]}^       �	���fc�A�*

loss�?*;�3��       �	7�fc�A�*

loss�7= .��       �	ʻfc�A�*

loss4&�:��       �	䟽fc�A�*

losst��;���       �	�:�fc�A�*

lossʲ;D��       �	�ξfc�A�*

loss]$:<��       �	�e�fc�A�*

lossr`�=鏯�       �	2��fc�A�*

loss�s	=(���       �	���fc�A�*

lossqm<���       �	"6�fc�A�*

loss)}Z=���       �	��fc�A�*

lossC�	;��6�       �	�c�fc�A�*

loss�|�<�`�       �	f3�fc�A�*

loss�t=p:y        �	���fc�A�*

loss%Q<?�_�       �	�|�fc�A�*

lossWny=�*�       �	j0�fc�A�*

loss�l<�6��       �	-��fc�A�*

loss��g;�Ν       �	�e�fc�A�*

loss_2S:
%�p       �	u�fc�A�*

loss1ȶ;5���       �	��fc�A�*

loss���<h�       �	�6�fc�A�*

loss�g<���q       �	��fc�A�*

lossM|7>����       �	g�fc�A�*

loss�g<|�0�       �	�fc�A�*

loss\e>:���S       �	��fc�A�*

loss��]:o�̆       �	�)�fc�A�*

lossV�9v�3�       �	r��fc�A�*

loss�P<<��P�       �	�~�fc�A�*

loss*��=	���       �	��fc�A�*

loss�1�=�9�       �	n��fc�A�*

loss[s�<PƐ'       �	<�fc�A�*

loss|��;S��]       �	���fc�A�*

loss�"*=�H��       �	�y�fc�A�*

loss�1j<�7��       �	�fc�A�*

lossgՇ=�?fb       �	��fc�A�*

lossi-�=���!       �	T5�fc�A�*

lossx!=��B       �	���fc�A�*

loss�u�=�'=       �	�X�fc�A�*

loss(��=��       �	���fc�A�*

lossC�v<g�l^       �	o~�fc�A�*

loss��;�b]       �	��fc�A�*

lossš�<gS(       �	���fc�A�*

loss��<��	�       �	AH�fc�A�*

loss�oW<E�5       �	��fc�A�*

loss��=�k       �	}u�fc�A�*

loss��; r0       �	,�fc�A�*

lossdJ<UG��       �	���fc�A�*

loss	`|<��T       �	wd�fc�A�*

lossj��<��e�       �	��fc�A�*

loss(&=�[�Y       �	��fc�A�*

loss��Z<�e�V       �	/3�fc�A�*

loss�=U�Q       �	���fc�A�*

loss�{<y8�       �	�o�fc�A�*

lossWA�<Y�f^       �	��fc�A�*

loss��<;��       �	���fc�A�*

loss�ק;�Y�f       �	�H�fc�A�*

loss���<�.       �	W��fc�A�*

losss[#=�,k       �	���fc�A�*

loss��Z<���v       �	'K�fc�A�*

loss,��<����       �	��fc�A�*

loss�!=���       �	=��fc�A�*

loss��u;���       �	z4�fc�A�*

loss���=e�(�       �	���fc�A�*

loss��5<j�"�       �	���fc�A�*

loss�� ;�Քg       �	K�fc�A�*

loss��=V���       �	`��fc�A�*

lossiۣ<S��       �	�^�fc�A�*

loss�d�=�e�       �	8��fc�A�*

loss�m;���'       �	��fc�A�*

lossS|�;ym�       �	�i�fc�A�*

loss���;
�t        �	���fc�A�*

loss㯿=�r�       �	F��fc�A�*

loss�ͳ<;=��       �	�(�fc�A�*

losswEG;-庯       �	���fc�A�*

loss���=����       �	�S�fc�A�*

loss�v=��[=       �	���fc�A�*

loss��<jh&�       �	���fc�A�*

loss���=�L�       �	,-�fc�A�*

loss���<H���       �	���fc�A�*

loss�F=l%4R       �	7p�fc�A�*

loss�A=�� �       �	$�fc�A�*

loss&�X=�;�+       �	ũ�fc�A�*

loss���:u�B       �	�G�fc�A�*

loss��U</t��       �	%��fc�A�*

loss7��:q D`       �	R��fc�A�*

loss�+�=|`�%       �	�fc�A�*

loss���=q�"�       �	x��fc�A�*

loss�%5=M���       �	�O�fc�A�*

loss�&	=I;0        �	���fc�A�*

lossRa�<�w       �	�~�fc�A�*

loss�D�=�4�9       �	-`�fc�A�*

loss-�=�F�       �	���fc�A�*

loss)G�<���       �	��fc�A�*

lossD�<���       �	�-�fc�A�*

loss|�b<[���       �	v��fc�A�*

loss�O=m�       �	J`�fc�A�*

loss@z<��l       �	���fc�A�*

loss��?=g.A�       �	��fc�A�*

loss�=�܂       �	�B�fc�A�*

loss��<�ƫW       �	/��fc�A�*

loss���<�8�k       �	|�fc�A�*

loss;TZ=�Ta�       �	j�fc�A�*

loss2;<H��       �	��fc�A�*

lossL��=-#�L       �	n�fc�A�*

lossM�D=�s�]       �	�fc�A�*

lossD�=�n       �	��fc�A�*

lossd�<8|��       �	`>�fc�A�*

loss�.4=����       �	J��fc�A�*

loss��:���       �	
i�fc�A�*

loss�	�;���       �	���fc�A�*

loss�� <���       �	O� fc�A�*

lossZW=�ږ�       �	Z+fc�A�*

loss�=����       �	��fc�A�*

loss�R�<��5       �	�yfc�A�*

loss��}<�s       �	&fc�A�*

loss0��<Ь��       �	u�fc�A�*

loss��5<g?�       �	RGfc�A�*

loss��:��!�       �	�fc�A�*

loss���;�4g       �	�wfc�A�*

lossq�m<�T�X       �	�fc�A�*

loss!�\=vu�       �	�fc�A�*

lossN�=����       �	Afc�A�*

losss��=~���       �	��fc�A�*

loss�KU=7M       �	�ifc�A�*

loss6�:��O�       �	:@	fc�A�*

loss��; 
��       �	��	fc�A�*

loss�3c<��]�       �	;o
fc�A�*

loss�S�=o�j       �	�fc�A�*

loss�%b=d֤       �	��fc�A�*

loss*>=J�)        �	�5fc�A�*

loss�f<g���       �	\�fc�A�*

loss�E5=�'@�       �	�ufc�A�*

lossة;k���       �	�fc�A�*

losshܡ=�+|c       �	ѯfc�A�*

loss�I�;����       �	��fc�A�*

lossR2S;3�x�       �	2<fc�A�*

losst�<Q���       �	��fc�A�*

lossS��<�y�       �	˿fc�A�*

loss�8>�<��       �		nfc�A�*

loss���=n�t�       �	�fc�A�*

loss.Ј;I���       �	�fc�A�*

loss��D<�{��       �	��fc�A�*

loss�*=o�E       �	Ffc�A�*

loss��<�H��       �	�"fc�A�*

lossm�X<��9<       �	,�fc�A�*

loss-�=qh�=       �	M�fc�A�*

lossnΡ<x$s       �	('fc�A�*

loss�5=,?�\       �	`tfc�A�*

loss��!=D<��       �	*Sfc�A�*

lossv�_</`�       �	�fc�A�*

loss_�<��T
       �	f�fc�A�*

loss%K�=ϧ        �	�fc�A�*

loss@��<��'       �	&fc�A�*

loss���=�?��       �	�fc�A�*

loss%	P<��&�       �	ߋfc�A�*

loss��"=
_�       �	D0fc�A�*

loss�ݠ<v(�       �	#�fc�A�*

loss�M=�       �	s� fc�A�*

loss�.�<�jX�       �	�^!fc�A�*

loss7Z=��u�       �	��!fc�A�*

lossT=���D       �	)�"fc�A�*

loss\��;�-)�       �	�3#fc�A�*

loss!m"=j��       �	�#fc�A�*

lossa�<��ё       �	ca$fc�A�*

lossw<+�;<       �	��$fc�A�*

loss��E<&>�       �	+�%fc�A�*

loss��h=M?I       �	�(&fc�A�*

loss�=�S��       �	��&fc�A�*

loss�J�;9)       �	B^'fc�A�*

loss,ٱ=}%%       �	��'fc�A�*

lossC�<��4�       �	�(fc�A�*

loss2�<4�nK       �	0))fc�A�*

loss�N3<�	       �	��)fc�A�*

loss�y�<;%TS       �	Hp*fc�A�*

loss���;s��       �	�+fc�A�*

loss�s�=m�:�       �	�+fc�A�*

loss��=I�V�       �	kf,fc�A�*

loss�:�<ޙ��       �	c-fc�A�*

loss8�<-���       �	��-fc�A�*

lossM5<�/��       �	�R.fc�A�*

lossA\=4�+       �	��.fc�A�*

loss�~?<����       �	��/fc�A�*

loss&]�:��$       �	jM1fc�A�*

losstn�<p        �	V�1fc�A�*

loss���<��H�       �	��2fc�A�*

lossW#�=F       �	�4fc�A�*

loss� =�6�       �	X6fc�A�*

loss�<�Gu�       �	%�6fc�A�*

loss;��=���       �	��7fc�A�*

loss��=M|       �	(e8fc�A�*

loss@�;��2�       �	�9fc�A�*

loss�_<�5o�       �	��9fc�A�*

loss�(�<�մ�       �	�H:fc�A�*

loss�y-<�`X�       �	-�:fc�A�*

loss3�<�`       �	`�;fc�A�*

lossλu<��͋       �	U/<fc�A�*

loss�)=��	�       �	��<fc�A�*

lossm�<���       �	v=fc�A�*

lossZ=�o�G       �	�>fc�A�*

loss=�U;+G5�       �	:�>fc�A�*

loss��=^	�       �	RF?fc�A�*

lossK�=ˇ�       �	��?fc�A�*

loss�o�<�~TY       �	=�@fc�A�*

loss;��=E
D�       �	�%Afc�A�*

loss�ʫ;ђ-x       �	��Afc�A�*

loss��i<MT��       �	AdBfc�A�*

loss��=���#       �	/�Bfc�A�*

loss�Q�<��       �	�Cfc�A�*

loss��!<��^�       �	0-Dfc�A�*

loss�ѕ<W�:u       �	��Dfc�A�*

lossQ7=�%��       �	4�Efc�A�*

loss�FO=����       �	!Ffc�A�*

loss�p�=RK�       �	��Ffc�A�*

loss���<1%(}       �	wgGfc�A�*

loss�+=��s       �	Hfc�A�*

loss4��;��.       �	 �Hfc�A�*

lossle-<کU�       �	�7Ifc�A�*

loss���;���       �	��Ifc�A�*

loss�>�<��z�       �	~Jfc�A�*

loss��=���       �	�Kfc�A�*

loss��>��a       �	ͱKfc�A�*

loss���=�v`�       �	HLfc�A�*

lossj%�<[a�       �	a�Lfc�A�*

lossv'|;0`$       �	�wMfc�A�*

lossC�N=�"}       �	,Nfc�A�*

lossV��<�3�9       �	��Nfc�A�*

loss�/!=��u       �	�dOfc�A�*

loss�� =d�rl       �	m�Ofc�A�*

loss�v=h�m+       �	؞Pfc�A�*

loss.$=��CG       �	<Qfc�A�*

lossғi;��j       �	o�Qfc�A�*

loss���<�-y       �	�mRfc�A�*

loss8��<A�|:       �	uSfc�A�*

loss�:�<3}w�       �	]�Sfc�A�*

lossSn�;�/(        �	��Tfc�A�*

loss�>p=���       �	! Ufc�A�*

loss��<ъ7�       �	��Ufc�A�*

lossM0=E�f       �	�OVfc�A�*

loss%��<�v�C       �	�3Wfc�A�*

loss:�G=`%�,       �	��Wfc�A�*

lossT�=�ˍ       �	wXfc�A�*

loss�*�<�$�u       �	;Yfc�A�*

loss��<V���       �	(�Yfc�A�*

lossQ�=�6�l       �	jmZfc�A�*

loss��8=�
��       �	[fc�A�*

lossMн;����       �	�[fc�A�*

loss{�B=[�>�       �	�:\fc�A�*

loss-R,<��Vo       �	��\fc�A�*

loss8`�=�\%k       �	fg]fc�A�*

loss��J<'CA       �	��]fc�A�*

loss�1�<��       �	Gt_fc�A�*

loss\	<Q�        �	M`fc�A�*

loss!\n=u�O
       �	��`fc�A�*

lossL�=W�H       �	׉afc�A�*

loss.�><�(ݼ       �	"bfc�A�*

loss�ͽ<5�       �	�cfc�A�*

loss���;cp�;       �	��cfc�A�*

loss:�;W�w       �	�5dfc�A�*

loss� ;�+�U       �	m�dfc�A�*

loss���;�ajj       �	
�efc�A�*

loss��<��y�       �	�ffc�A�*

lossx;ţ��       �	�ffc�A�*

loss�m<���       �	 Rgfc�A�*

loss��;}8Y�       �	��gfc�A�*

loss[�'=���M       �	�hfc�A�*

loss4��;R��       �	vifc�A�*

loss,P�<�:��       �	ܽifc�A�*

loss{6�=��-&       �	�Zjfc�A�*

loss̔`=�P��       �	_�jfc�A�*

loss�Q=Uƥ�       �	K�kfc�A�*

loss��<�p�       �	 )lfc�A�*

loss/u<H���       �	��lfc�A�*

loss�+�=ñ�       �	�amfc�A�*

loss=)�<8�o;       �	��mfc�A�*

lossJ��=����       �	��nfc�A�*

lossk';�l}       �	�<ofc�A�*

loss 3�=�ec�       �	��ofc�A�*

lossE��;��       �	��pfc�A�*

loss
�<�y�       �	��qfc�A�*

loss��<�D7-       �	�rfc�A�*

lossE�j=̾�	       �	�`sfc�A�*

loss��(=��Q       �	
+tfc�A�*

loss/2�;js�       �	��tfc�A�*

loss�~�<4�O       �	��ufc�A�*

loss��u=Q��       �	p>vfc�A�*

loss�,�;�j�6       �	�awfc�A�*

loss��<p       �	h%xfc�A�*

lossR0;=��l;       �	�yfc�A�*

loss/�9=��7�       �	$�yfc�A�*

lossV|�<����       �	r{fc�A�*

loss�=l2\       �	/M|fc�A�*

loss�D =���       �	�}fc�A�*

loss�Rw;�2\Y       �	��~fc�A�*

loss�~<?X�/       �	s�fc�A�*

loss�I=�vn�       �	Lo�fc�A�*

lossۑ	<h�        �	t
�fc�A�*

loss3�=&Qė       �	:��fc�A�*

loss*�g=�*�)       �	,F�fc�A�*

losss&�=�B�<       �	�܂fc�A�*

loss��C<2b�       �	t�fc�A�*

loss8A�<C��.       �	��fc�A�*

loss�>;���       �	���fc�A�*

loss���;��X�       �	���fc�A�*

loss.��<R���       �	��fc�A�*

loss�V�<y�       �	�8�fc�A�*

loss>��=0�?       �	߇fc�A�*

loss]Z�<���       �	e��fc�A�*

loss�P�=����       �	�,�fc�A�*

lossȃ�;m�Tj       �	�ȉfc�A�*

loss�Y�<F˔�       �	^f�fc�A�*

loss��<$p�:       �	�fc�A�*

loss�3�<w$�       �	l��fc�A�*

loss˦<�ݎ       �	�R�fc�A�*

loss���=1��I       �	J�fc�A�*

lossMP�=�AG@       �	���fc�A�*

lossi��<�ǁq       �	�,�fc�A�*

loss|��=�3�       �	�Îfc�A�*

loss�]a=\���       �	Y�fc�A�*

loss�� <I�8�       �	��fc�A�*

lossS�:F!��       �	�fc�A�*

loss詍;;�X       �	��fc�A�*

loss��=��d       �	��fc�A�*

loss�H<���#       �	�Ēfc�A�*

loss_~I<�A-�       �	�X�fc�A�*

loss� E<9���       �	p�fc�A�*

loss��;� 1�       �	D��fc�A�*

loss�.�<�(       �	�!�fc�A�*

lossc�m=	ٲ       �	���fc�A�*

loss��<#;t�       �	�P�fc�A�*

lossDM
<~p�$       �	C�fc�A�*

loss�Y)<
n��       �	�u�fc�A�*

loss�*u<���       �	��fc�A�*

loss<�<�w~X       �	���fc�A�*

loss��<�)Z_       �	�/�fc�A�*

lossj��=��n�       �		ƙfc�A�*

loss�M=+���       �	ρ�fc�A�*

lossF�<^8�       �	��fc�A�*

lossy<u�F�       �	ٴ�fc�A�*

loss��=����       �	�X�fc�A�*

loss�� =�/z�       �	I�fc�A�*

loss�T�<x��       �	l��fc�A�*

loss�!�<��_       �	�?�fc�A�*

lossã�;Qoi       �	X�fc�A�*

loss��<�m�;       �	؂�fc�A�*

loss���<_�`       �	��fc�A�*

loss�=�&��       �	��fc�A�*

loss��<���       �	��fc�A�*

loss��=`n�       �	�M�fc�A�*

loss7wo=0>PA       �	h�fc�A�*

loss��;�jY�       �	��fc�A�*

loss
=����       �	B!�fc�A�*

loss/��=�@�       �	ᷥfc�A�*

lossQ�"=�ADv       �	�O�fc�A�*

lossф�<�ғ\       �	���fc�A�*

loss��=���       �	G��fc�A�*

loss.�a<i`��       �	�(�fc�A�*

loss}b�:H'�       �	���fc�A�*

loss��<:�@�       �	�W�fc�A�*

loss*<��l       �	��fc�A�*

loss6z�;�00c       �	���fc�A�*

lossd9�<�`��       �	��fc�A�*

loss8��;����       �	 ��fc�A�*

loss��=�h
       �	�C�fc�A�*

lossN�$<�xϚ       �	��fc�A�*

loss%��=&��)       �	)z�fc�A�*

lossR2�;�4�Z       �	y �fc�A�*

loss��=���O       �	:ʮfc�A�*

loss=��;$b�       �	�j�fc�A�*

loss��<[S�K       �	W�fc�A�*

lossث�<E�φ       �	�>�fc�A�*

loss�_S=�+j�       �	#�fc�A�*

loss�?
<�'<l       �	��fc�A�*

loss[��<�%�5       �	eP�fc�A�*

loss��=(V�e       �	��fc�A�*

loss	��=��B*       �	&��fc�A�*

loss7��<���       �	���fc�A�*

loss��@=_`�       �	ݱ�fc�A�*

loss��<�T�       �	�J�fc�A�*

loss)��<[)�e       �	(d�fc�A�*

lossd,=E6��       �	� �fc�A�*

loss��,=?��i       �	���fc�A�*

loss���;���j       �	�&�fc�A�*

loss|<W���       �	&��fc�A�*

loss#c�<6�-       �	�7�fc�A�*

lossw7�<_vs�       �	�ٽfc�A�*

loss=G;���d       �	5z�fc�A�*

loss��<�Mǉ       �	f�fc�A�*

loss_�*=B�}�       �	&��fc�A�*

lossa�=���       �	���fc�A�*

loss�S�;��̳       �	�A�fc�A�*

loss{e=b�@       �	���fc�A�*

loss`�J;�0�       �	�|�fc�A�*

loss
�=�LG       �	��fc�A�*

loss��;�M�       �	��fc�A�*

loss��&>B�9M       �	X�fc�A�*

loss]a<�r�f       �	��fc�A�*

loss�2�<1�2�       �	���fc�A�*

loss�<э8C       �	�'�fc�A�*

loss,u�<'RS       �	L��fc�A�*

loss���<�UA�       �	Y�fc�A�*

lossr� =~hJ       �	c��fc�A�*

loss�p�;)�[Y       �	���fc�A�*

loss.D$="rO       �	d�fc�A�*

loss]|�<�G8%       �	}��fc�A�*

loss�y =�S�       �	L�fc�A�*

loss_'<΢��       �	F��fc�A�*

loss��<-���       �	��fc�A�*

loss~�!=0���       �	�!�fc�A�*

loss�w<�_q;       �	L��fc�A�*

loss���<P       �	sh�fc�A�*

loss�W�<gX�/       �	:"�fc�A�*

lossd�<,/       �	��fc�A�*

loss.+�=��       �	�T�fc�A�*

lossš:����       �	���fc�A�*

lossΧ.=�`\       �	
��fc�A�*

loss0�=��Y"       �	�,�fc�A�*

loss=ٓf�       �	���fc�A�*

loss�«<�mf       �	
��fc�A�*

loss�-=�UcC       �	��fc�A�*

loss�&=+�Ԑ       �	�+�fc�A�*

loss\0�;�Q�       �	���fc�A�*

loss< �<��3�       �	&n�fc�A�*

loss�r>Q A        �	 �fc�A�*

loss�`�;p       �	��fc�A�*

loss���<��(�       �	82�fc�A�*

loss�;.;��Y       �	���fc�A�*

loss��<����       �	e�fc�A�*

loss{�@<�Z�o       �	^�fc�A�*

loss��<1��n       �	`�fc�A�*

loss!�d<2�>1       �	2��fc�A�*

loss�8�<�K�       �	wM�fc�A�*

loss���;�Ц�       �	���fc�A�*

loss�YH=O�_       �	F~�fc�A�*

loss�]F=Z\��       �	��fc�A�*

loss:�,<?>r       �	���fc�A�*

loss�Cf=�       �	]Q�fc�A�*

loss$�<���       �	2>�fc�A�*

lossl��:y�JI       �	a��fc�A�*

loss{_;���       �	�u�fc�A�*

loss更=��+       �	.;�fc�A�*

lossA�<y�n�       �	V��fc�A�*

loss���<B�       �	�n�fc�A�*

loss�P�<w���       �	��fc�A�*

losse��<1ѫj       �	8��fc�A�*

lossDb*=4�t       �	)=�fc�A�*

loss�:)<��       �	E��fc�A�*

loss��<�ȗ       �	vk�fc�A�*

loss��x<5X]B       �	 �fc�A�*

loss���=s�C�       �	���fc�A�*

loss���<���       �	E/�fc�A�*

lossAj�=� 8       �	��fc�A�*

loss���=)~5B       �	1_�fc�A�*

loss	~�;��V�       �	�]�fc�A�*

lossoݘ=���       �	M��fc�A�*

loss��w<��	�       �	ׇ�fc�A�*

loss�[;C��$       �	��fc�A�*

loss/=�\�       �	T��fc�A�*

loss�&;1�h�       �	�F�fc�A�*

loss�N�=�!^       �	���fc�A�*

loss��w=,��       �	k�fc�A�*

loss��<J�Q�       �	�Z�fc�A�*

loss@.�:���n       �	��fc�A�*

loss���<m0�       �	���fc�A�*

lossc3�;���       �	�/�fc�A�*

loss�f�<羚�       �	���fc�A�*

loss��=�I�       �	���fc�A�*

loss%m�=��U       �	IJ�fc�A�*

loss���<oΊ       �	���fc�A�*

loss���=n[J       �	���fc�A�*

loss�x�;!>��       �	���fc�A�*

lossD~u:��~       �	�k�fc�A�*

loss1��<>�       �	��fc�A�*

loss��=ű��       �	��fc�A�*

loss��<-���       �	�U�fc�A�*

lossj��;1Y;�       �	8��fc�A�*

loss��<9Y/~       �	7��fc�A�*

lossTrZ;&c�       �	=+�fc�A�*

loss��^<�Wr       �	7��fc�A�*

loss3��=Njb       �	�Z�fc�A�*

loss�B =��N�       �	t��fc�A�*

loss]��<�(�       �	c��fc�A�*

lossö<����       �	`;�fc�A�*

loss��Z=�S&�       �	c��fc�A�*

loss�<�
��       �	'��fc�A�*

loss�!!=%
�p       �	*��fc�A�*

lossF�=H��       �	��fc�A�*

loss<�<���       �	���fc�A�*

loss���<�B�       �	�W fc�A�*

loss:+�<�8�o       �	� fc�A�*

loss��6<42�       �	��fc�A�*

loss�g0=m�lY       �	4fc�A�*

loss]�::��q2       �	>�fc�A�*

loss��<���W       �	=afc�A�*

loss8�<�}ՠ       �	��fc�A�*

loss}w�<h��       �	R�fc�A�*

loss�(;ƥ k       �	Ifc�A�*

loss�i <>;/�       �	A�fc�A�*

loss�v�=#H�u       �	ݙfc�A�*

loss�=�;��>       �	G;fc�A�*

loss�78<J�       �	��fc�A�*

loss�G�=�%�       �	�vfc�A�*

loss�g�:*���       �	�<	fc�A�*

loss1��:qT��       �	�	fc�A�*

lossm��<ږ��       �	�r
fc�A�*

loss���;�"�       �	h	fc�A�*

loss���;B�       �	j�fc�A�*

lossڕ	<�?ގ       �	�?fc�A�*

lossxs;5 >�       �	��fc�A�*

lossy=��K�       �	*pfc�A�*

loss�f�<���       �	�	fc�A�*

lossE��:|(�       �	'�fc�A�*

lossE�T;g�q       �	�Mfc�A�*

lossH��;W�*       �	�fc�A�*

loss6]=Uc�]       �	V�fc�A�*

losszT<<n�G       �	�!fc�A�*

lossRB/:��9       �	w�fc�A�*

loss��Y;C��7       �	~qfc�A�*

loss�A>��l       �	�fc�A�*

lossd��;2��       �	e�fc�A�*

loss� >�m0?       �	��fc�A�*

loss9��<�~4�       �	�(fc�A�*

loss�iz=e�s�       �	��fc�A�*

loss2�;R�L�       �	��fc�A�*

lossdJX<�3�       �	fc�A�*

loss�_m=�q�       �	�fc�A�*

loss3DM=dY*       �	�fc�A�*

loss�"=ﶀ       �	�8fc�A�*

loss��X<�S�%       �	@�fc�A�*

loss���<*��       �	�fc�A�*

lossM�1=���K       �	*fc�A�*

losshZ=����       �	��fc�A�*

loss)~�<��]*       �	�afc�A�*

loss��K=+P\       �	�fc�A�*

lossJ�1=���       �	��fc�A�*

loss�o�<�r�`       �	Vd fc�A�*

loss�<>��`       �	  !fc�A�*

loss��=NJ�~       �	9�!fc�A�*

loss�[�<�n4�       �	+2"fc�A�*

loss��:^�xl       �	�#fc�A�*

loss��=�26       �	�!$fc�A�*

loss9��=�B�       �	��$fc�A�*

loss:��<�H �       �	Ӣ%fc�A�*

loss�_)<���       �	;:&fc�A�*

lossI<U���       �	r�&fc�A�*

loss�2i=re�^       �	�v'fc�A�*

loss��8=�zG       �	�(fc�A�*

lossÙB=RU       �	��(fc�A�*

loss��S=���       �	h[)fc�A�*

loss;d<�A*�       �	��)fc�A�*

lossMb�<�?�       �	��*fc�A�*

loss��<��       �	6+fc�A�*

loss֊�;m�(k       �	*�+fc�A�*

loss��<�a6�       �	cb,fc�A�*

loss��;c$�       �	��,fc�A�*

loss��i;��       �	�-fc�A�*

loss�7=%�|       �	�2.fc�A�*

loss#�=��[�       �	=�.fc�A�*

losss�e<��܆       �	ro/fc�A�*

loss�(�;K�       �	�0fc�A�*

lossԫ<�˱�       �	�0fc�A�*

lossy=��~�       �	E1fc�A�*

loss��=���       �	��1fc�A�*

lossq�;�óf       �	 93fc�A�*

loss�V�;�O#       �	��3fc�A�*

loss��~=!O��       �	��4fc�A�*

lossV�<���       �	�	6fc�A�*

loss�==@6.�       �	O�6fc�A�*

loss��:�cU�       �	��7fc�A�*

lossi5<=0$�       �	�58fc�A�*

losst�"<�Z-�       �	�Vfc�A�*

losst�<b�       �	'�Vfc�A�*

lossه=�j��       �	q�Wfc�A�*

loss�H=*��|       �	KsXfc�A�*

loss�=$< �=       �		Yfc�A�*

loss��;&�=       �	��Yfc�A�*

loss�{0;����       �	�QZfc�A�*

loss���<��e5       �	��Zfc�A�*

lossd�u<!�L�       �	Z[fc�A�*

loss���<��       �	�\fc�A�*

loss==�;e�9       �	��\fc�A�*

lossS�<Hr1Q       �	G]fc�A�*

lossX<t=T�K       �	�]fc�A�*

lossn]�<��j�       �	Z�^fc�A�*

loss���<z2)�       �	_fc�A�*

loss:�1=���G       �	C�_fc�A�*

loss�$u;��8�       �	}@`fc�A�*

loss��H<���A       �	3Nafc�A�*

loss,�
<�Jo�       �	��afc�A�*

loss<�F=�V=6       �	o�bfc�A�*

lossax�<m�J�       �	W!cfc�A�*

loss�5Q=��       �	i�cfc�A�*

loss�r;a�sf       �	�adfc�A�*

loss=�=�9�       �	��dfc�A�*

loss�}�:�<�"       �	w�efc�A�*

loss#�<�,p�       �	�;ffc�A�*

loss=�=k��"       �	�ffc�A�*

loss=��;"M'       �	Ktgfc�A�*

loss�I=���H       �	'hfc�A�*

loss/ϡ=(v�       �	q�hfc�A�*

loss��=1,       �	DMifc�A�*

lossZ�<,��O       �	��ifc�A�*

lossn<[\�b       �	j�jfc�A�*

loss���="L�       �	�$kfc�A�*

loss]��<�|�       �	��kfc�A�*

lossVB�=Ҋ;       �	qYlfc�A�*

loss�jU<`       �	�lfc�A�*

loss�1�<�,"R       �	�mfc�A�*

loss��=E]�=       �	
,nfc�A�*

lossCK\=,Jo#       �	2�nfc�A�*

loss��<h���       �	[aofc�A�*

loss܎?:RGϋ       �	H�ofc�A�*

lossԬ1;(�u�       �	̖pfc�A�*

loss�g7<���%       �	I,qfc�A�*

loss��<�y�       �	 �qfc�A�*

loss��=��3�       �	|&sfc�A�*

loss��&<�E�       �	p]tfc�A�*

losse��<�<�       �	�ufc�A�*

loss��<.�g       �	�vfc�A�*

loss��9��)�       �	�vfc�A�*

loss��:i��}       �	U�wfc�A�*

loss�A<8c�       �	�Exfc�A�*

loss�&�;3�`       �	]�xfc�A�*

loss[_�=�D]}       �	~yfc�A�*

loss���<��u       �	F�zfc�A�*

loss�<�:N�U       �	.�{fc�A�*

loss ��;���       �	�<|fc�A�*

loss�H:e�O�       �	R�|fc�A�*

lossK�	<u��       �	�
~fc�A�*

loss}�=����       �	�"�fc�A�*

loss<�=<���       �	$_�fc�A�*

loss�N;���       �	x�fc�A�*

loss�<JO       �	i��fc�A�*

lossIN�=�R�:       �	^H�fc�A�*

loss�w= :�M       �	t�fc�A�*

lossz��<�M�       �	ׅ�fc�A�*

loss�,]=�eY�       �	�%�fc�A�*

loss�U�<n�DR       �	VՅfc�A�*

loss�É=�g:�       �	���fc�A�*

loss�A=��        �	�*�fc�A�*

lossf(<�x�       �	pχfc�A�*

lossb��<���t       �	`w�fc�A�*

loss�6�<���       �	��fc�A�*

loss "=�<[�       �	���fc�A�*

lossޗ<A|i�       �	gd�fc�A�*

lossx��<�k�       �	)�fc�A�*

loss��+=q{��       �	���fc�A�*

loss���;�,kw       �	�O�fc�A�*

losse�(;p�c	       �	�fc�A�*

loss�2�=%��       �	2��fc�A�*

losse�<����       �	mV�fc�A�*

loss�#G<vǣ�       �	
�fc�A�*

loss�<c�D       �	"��fc�A�*

loss=B<���J       �	)&�fc�A�*

loss#=��=       �	�ʐfc�A�*

loss�n�<��5�       �	�d�fc�A�*

loss�E=r-��       �	2�fc�A�*

loss4�<>>��       �	ퟒfc�A�*

loss�Z<}"	�       �	�C�fc�A�*

loss7\�<�r�       �	��fc�A�*

lossE��;=#�,       �	G��fc�A�*

lossZ�=�8       �	�M�fc�A�*

lossTB�;*!;G       �	���fc�A�*

loss�2�<��       �	���fc�A�*

loss94�;�fu       �	=C�fc�A�*

lossZ
<����       �	~�fc�A�*

loss}m�<ņ�^       �	n��fc�A�*

lossy�<�kTa       �	Z/�fc�A�*

loss�H�=�2�       �	�̙fc�A�*

lossL�<=�[       �	�m�fc�A�*

loss��:���       �	\�fc�A�*

losst�;=���       �	���fc�A�*

loss�<;ȟE�       �	y]�fc�A�*

loss�]�<�fV�       �	z��fc�A�*

lossѨ�;sj�       �	Ϣ�fc�A�*

loss��<sq�6       �	O;�fc�A�*

lossi�?;G<�       �	;��fc�A�*

loss!�;Ʋ�S       �	�z�fc�A�*

loss=G=�V`�       �	�fc�A�*

lossʦt<)�P       �	��fc�A�*

loss?&�=��       �	�Y�fc�A�*

loss��]=����       �	�`�fc�A�*

loss/�=�,�       �	+��fc�A�*

loss�;�r�L       �	R��fc�A�*

loss\��;�n��       �	�=�fc�A�*

lossC7�:v�2       �	��fc�A�*

loss;ں;��7�       �	��fc�A�*

loss�H�<��K?       �	�h�fc�A�*

loss�W(<��       �	(�fc�A�*

loss�F<�"T�       �	l�fc�A�*

loss���<�	%�       �	��fc�A�*

loss��<آg�       �	���fc�A�*

loss�a�;e	�       �	CU�fc�A�*

loss�=w��j       �	z��fc�A�*

lossvt�;{�ȧ       �	.�fc�A�*

loss���<S��w       �	��fc�A�*

lossӗ;l���       �	ZE�fc�A�*

lossZBV<W��l       �	wܮfc�A�*

loss��=���       �	�u�fc�A�*

loss�?=_��       �	 �fc�A�*

loss��G<��p       �	��fc�A�*

loss�B�=�~�       �	�=�fc�A�*

lossm)={*��       �	�`�fc�A�*

lossUI;���       �	�k�fc�A�*

loss���<����       �	���fc�A�*

loss)�<���       �	�ʵfc�A�*

loss�@ <�]A       �	^��fc�A�*

loss��<��!       �	iƷfc�A�*

loss�nN;s%{a       �	�U�fc�A�*

lossX:�<b5��       �	�%�fc�A�*

loss�<�Cz�       �	���fc�A�*

loss
}�<���       �	ݻfc�A�*

lossHGu;�J6+       �	r��fc�A�*

loss��<W��I       �	qZ�fc�A�*

loss,��<k�f       �	�M�fc�A�*

loss��{=m+�       �	��fc�A�*

lossMe<d�H|       �	%��fc�A�*

loss�z�<�j�n       �	�L�fc�A�*

loss]�:o1��       �	�b�fc�A�*

loss��?<�x�       �	ǡ�fc�A�*

loss�L�;��G�       �	�l�fc�A�*

loss3<���`       �	��fc�A�*

loss�F�;̴�f       �	Eg�fc�A�*

loss�==i       �	�,�fc�A�*

lossô)=0�f/       �	�	�fc�A�*

lossA�9K67I       �	���fc�A�*

loss�[%;wҡ.       �	���fc�A�*

loss��t<|��R       �	 ��fc�A�*

losso|�<�;�       �	@�fc�A�*

lossZ�?=���       �	�S�fc�A�*

loss��;Eb��       �	U��fc�A�*

lossJ�<�	Y�       �	y�fc�A�*

loss?�X=s{�       �	V��fc�A�*

loss�i=:�[�       �	�e�fc�A�*

loss��;���       �	dW�fc�A�*

loss�~�:ܠ�
       �	Q��fc�A�*

lossICH<�<5       �	���fc�A�*

loss�gg< D7       �	�o�fc�A�*

lossI|�;s��       �	��fc�A�*

loss��h9ðw       �	���fc�A�*

lossCA�<�[�       �	��fc�A�*

loss�vu;��T�       �	=��fc�A�*

lossH_;�|��       �	���fc�A�*

loss��<D       �	�)�fc�A�*

lossw�94��       �	���fc�A�*

lossڿ�=}�       �	���fc�A�*

loss��<���       �	�f�fc�A�*

lossX� =$\Zl       �	�
�fc�A�*

loss�=_9�%       �	��fc�A�*

loss8A:�p�       �	�W�fc�A�*

loss�4P;��r~       �	��fc�A�*

loss[t�<�x��       �	x��fc�A�*

lossT�'<W���       �	f��fc�A�*

loss\�B=�BŜ       �	���fc�A�*

loss�5;I �       �	�}�fc�A�*

loss��;��P       �	&�fc�A�*

loss�^=KR�*       �	���fc�A�*

loss,�<�|i       �	4h�fc�A�*

loss�K�=���'       �	i�fc�A�*

loss4`P=��F       �	���fc�A�*

lossmF�=Ύ�       �	�l�fc�A�*

loss��(<�H       �	I�fc�A�*

loss�d;�A�       �	���fc�A�*

loss-�G<mR�       �	[_�fc�A�*

loss�V�;��       �	O�fc�A�*

lossno�:�$�1       �	���fc�A�*

lossZ�<(h[,       �	&W�fc�A�*

lossɣ=���       �	���fc�A�*

loss7�s=��+       �	���fc�A�*

lossT#�<L(Q�       �	$|�fc�A�*

loss�c�<�$5       �	�(�fc�A�*

loss��<�5��       �	���fc�A�*

loss1ɮ<(�9$       �	G��fc�A�*

loss�T$=WS       �	�X�fc�A�*

lossC��<D#%       �	��fc�A�*

loss�׫;[Ͱ       �	�Z�fc�A�*

lossn��<}ٌ)       �	��fc�A�*

loss@��<���       �		��fc�A�*

lossf��;��:2       �	1�fc�A�*

lossҼ�;|u�$       �	U�fc�A�*

loss�0h<���{       �	) �fc�A�*

lossE��;�V��       �	���fc�A�*

loss%�<Ak�7       �	_�fc�A�*

loss��;Edb�       �	u�fc�A�*

loss�]4=����       �	���fc�A�*

loss��<��ּ       �	H6�fc�A�*

loss|:>3Ӂ�       �	��fc�A�*

loss�7�<[��|       �	��fc�A�*

loss�e6=)��       �	�+�fc�A�*

loss1o�<=:�z       �	���fc�A�*

loss�	q=�J��       �	�{�fc�A�*

loss�)(<OtF       �	g,�fc�A�*

loss��*<�k��       �	��fc�A�*

loss�l=,��       �	�{�fc�A�*

loss��;�z}�       �	��fc�A�*

lossxY�<v�j'       �	���fc�A�*

loss�=k='I       �	Dl�fc�A�*

lossx�]=_��E       �	��fc�A�*

lossC�<ʊ��       �	���fc�A�*

loss�h9<�ؘs       �	j fc�A�*

loss��r;�Z��       �	�� fc�A�*

loss}Ŵ:nL�       �	�fc�A�*

lossE%B=���'       �	)fc�A�*

loss�B&<���O       �	��fc�A�*

loss?��<g+�       �	PTfc�A�*

loss<��;����       �	��fc�A�*

loss�߅<���       �	�fc�A�*

loss�N>=ZJM�       �	#fc�A�*

loss=�=�A%�       �	��fc�A�*

loss�N�;�Wp       �	�Bfc�A�*

loss�F=�;P       �	B'fc�A�*

loss=�<���O       �	+�fc�A�*

loss`&&<��(       �	��fc�A�*

loss_u	<��	�       �	�,	fc�A�*

loss��<�m5[       �	��	fc�A�*

loss�u�<\���       �	�g
fc�A�*

loss� <,��u       �	fc�A�*

lossF�<��?       �	Ūfc�A�*

loss��%=[�tq       �	�Ofc�A�*

loss�R�<$��'       �	��fc�A�*

loss�c;�FY2       �	��fc�A�*

lossl�=n�@�       �	�.fc�A�*

loss$b�<g�       �	>�fc�A�*

lossJ{�;�m=�       �	�hfc�A�*

loss�5M;���       �	dfc�A�*

lossl�;Ͽo]       �	Ԟfc�A�*

loss��=̳E       �	C:fc�A�*

loss���<����       �	��fc�A�*

loss�*=�l�       �	apfc�A�*

loss��=����       �	fc�A�*

loss p<G[�       �	�fc�A�*

lossq�<:)S       �	�9fc�A�*

loss`]?=�&�       �	4�fc�A�*

loss:�=�3�       �	Cufc�A�*

loss"�<mgm�       �	9%fc�A�*

loss�7;9ל�       �	��fc�A�*

losst�=�J��       �	�fc�A�*

loss���<�h�       �	v�fc�A�*

loss�h�=t�_�       �	U�fc�A�*

lossL�>=��       �	hfc�A�*

loss��\<����       �	~fc�A�*

loss��<Cm�0       �	3fc�A�*

loss�7&<�#��       �	��fc�A�*

loss�c�;��/\       �	�tfc�A�*

loss͚�<�+p       �	)fc�A�*

lossRIM<��.)       �	Лfc�A�*

loss�G�<�a��       �	%; fc�A�*

loss�T�;0܅�       �	�� fc�A�*

loss�f�<X'o       �	5~!fc�A�*

loss�8=���       �	Q"fc�A�*

lossԀO9��/9       �	��"fc�A�*

loss��<b\OB       �	<K#fc�A�*

loss�_�<}�A�       �	��#fc�A�*

lossy.;='X       �	��$fc�A�*

loss���;�$�E       �	�;%fc�A�*

lossX1�<�l�       �	i�%fc�A�*

loss���<���;       �	�&fc�A�*

loss�o;�=C       �	�8'fc�A�*

losstj<L8��       �	rM(fc�A�*

loss]=�<n��k       �	��(fc�A�*

loss�� :�I6_       �	ސ)fc�A�*

loss� =<m�       �	�0*fc�A�*

loss�/=U
!       �	��*fc�A�*

loss.�;:)S�{       �	�i+fc�A�*

loss(�5<<�p       �	,fc�A�*

loss��<�#��       �	��,fc�A�*

loss�3<m�}       �	}<-fc�A�*

loss��<3��Y       �	��-fc�A�*

loss���=�"`�       �	en.fc�A�*

lossZ�"<��$0       �	:/fc�A�*

loss=G8��       �	ԙ/fc�A�*

lossXش<��Z       �	0,0fc�A�*

loss���;ʰ�e       �	�0fc�A�*

loss�&,=�3٧       �	Y1fc�A�*

loss��g<����       �	��1fc�A�*

loss_Z=��V^       �	ɓ2fc�A�*

loss+�<!�]�       �	�3fc�A�*

loss��<�;       �	]j4fc�A�*

lossʋ�;NJ�       �	��5fc�A�*

loss/��;��       �	CV6fc�A�*

loss��?<8$�       �	q�7fc�A�*

lossE�:ˍ��       �	��8fc�A�*

loss���<h,xK       �	P�9fc�A�*

lossf�<���       �	��:fc�A�*

loss�i�<闥       �	I;fc�A�*

loss��n=wD       �	p<fc�A�*

lossi��;�k8       �	.Y=fc�A�*

loss�@�<G�d�       �	�>fc�A�*

loss�5�<���g       �	��>fc�A�*

loss�rO=ܽ�       �	(�?fc�A�*

loss/@<i�       �	<@fc�A�*

loss��=Y �t       �	�@fc�A�*

lossN�<ۀ��       �	��Afc�A�*

lossK<����       �	ɯBfc�A�*

loss<:�=QF�x       �	[[Cfc�A�*

loss�=g��        �	�vDfc�A�*

loss
J�<��?�       �	�Efc�A�*

loss�$=<�(       �	��Efc�A�*

loss��<�v(       �	_Ffc�A�*

lossM�<�*�u       �	�Ffc�A�*

loss�=�;>M�       �	L�Gfc�A�*

loss1 �<����       �	�AHfc�A�*

lossD��:H$d:       �	��Hfc�A�*

loss��<˷gp       �	4�Ifc�A�*

loss�(=�'��       �	c)Jfc�A�*

loss8Z=.d       �	;�Jfc�A�*

loss3�;���       �	2�Kfc�A�*

loss_m<�$��       �	�<Lfc�A�*

loss�*=���&       �	��Lfc�A�*

loss���;)��       �	k�Mfc�A�*

loss��;��0�       �	6Nfc�A�*

lossz_%=4��&       �	P�Nfc�A�*

loss�:�b�       �	K[Ofc�A�*

loss��:!U       �	B�Ofc�A�*

loss�r4=���       �	�Pfc�A�*

loss�<hO��       �	#,Qfc�A�*

loss��b;�@�/       �	��Qfc�A�*

loss;ڠ<,9�       �	SRfc�A�*

loss���<��       �	��Rfc�A�*

loss��<�S�@       �	�ySfc�A�*

loss;�:P�h       �	�Tfc�A�*

loss[��;lY�       �	��Tfc�A�*

loss�=w@�       �	�4Ufc�A�*

loss@�E;\�2       �	X�Ufc�A�*

loss�h^=�`�w       �	|cVfc�A�*

lossq��;��       �	Y�Vfc�A�*

loss���;�F�       �	�Wfc�A�*

loss�O�;��~       �	XYfc�A�*

loss�<08(1       �	�Yfc�A�*

loss��
<���       �	�~Zfc�A�*

loss���<k(a       �	�[fc�A�*

loss��;��!�       �	�N\fc�A�*

loss��<����       �	�]fc�A�*

loss\�=�N�+       �	��]fc�A�*

loss��,=��@,       �	�I^fc�A�*

lossj��;�`       �	��^fc�A�*

loss�)<��e       �	�_fc�A�*

loss#��<�0�       �	� `fc�A�*

loss�n<t.�       �	=�`fc�A�*

loss���<���       �	AJafc�A�*

loss�/<6\Y�       �	��afc�A�*

loss$}�;�^Q�       �	�bfc�A�*

loss�]�<���       �	�"cfc�A�*

lossF#�<ox       �		�cfc�A�*

loss�Օ<j&�	       �	�edfc�A�*

loss�)E= T{k       �	_efc�A�*

loss�S<��g       �	�efc�A�*

loss�a�<
�Y�       �	�Iffc�A�*

loss3&];qH`�       �	��ffc�A�*

loss�v�:�y�       �	��gfc�A�*

loss��o<<c}�       �	Khfc�A�*

loss� =�~�       �	��hfc�A�*

lossv|�=�c�:       �	܀ifc�A�*

lossc�;���       �	yjfc�A�*

loss1l�<��,       �	o�jfc�A�*

loss�
�<5��       �	9_kfc�A�*

loss#�";�q(       �	�	lfc�A�*

loss좤;�7E�       �	�lfc�A�*

loss�Ԇ<vm��       �	aQmfc�A�*

loss�e{;�m��       �	�nfc�A�*

lossÍ{<����       �	I�nfc�A�*

lossϬB<wY��       �	�ofc�A�*

lossO�<2k       �	�pfc�A�*

loss��N=a�XO       �	��pfc�A�*

loss�r=�E�n       �	$aqfc�A�*

loss2	'>M�g:       �	��qfc�A�*

loss_�f<	�L        �	��rfc�A�*

loss�;��lc       �	�Csfc�A�*

loss��;��v       �	��sfc�A�*

loss��<�N�       �	��tfc�A�*

loss�A]<s�E       �	�!ufc�A�*

loss.��;�3X�       �	�ufc�A�*

loss_s=�0       �	rPvfc�A�*

loss&��;q	hw       �	5�vfc�A�*

loss�M�=���k       �	s�wfc�A�*

loss��<�ø�       �	�xfc�A�*

loss}<���       �	ުxfc�A�*

loss�`=��fg       �	bIyfc�A�*

lossm�;a��       �	��yfc�A�*

loss���<�=Y�       �	$~zfc�A�*

loss�?7<)�}3       �	D{fc�A�*

loss�i&=Ѝ�h       �	#�{fc�A�*

loss ��<@z��       �	�Q|fc�A�*

loss��=�2       �	��|fc�A�*

loss�Ԉ<���h       �	q�}fc�A�*

loss!F<fUA�       �	�%~fc�A�*

loss�y:���/       �	��~fc�A�*

lossW�w;�q�z       �	)Zfc�A�*

loss�8B<�"��       �	?�fc�A�*

lossi��;,�a�       �	g��fc�A�*

loss��<nu`       �	�7�fc�A�*

loss���<�/S�       �	-ρfc�A�*

loss;`6��       �	r�fc�A�*

loss�i�<p���       �	��fc�A�*

loss�"�;\?��       �	פ�fc�A�*

loss���;-:�N       �	�9�fc�A�*

lossx�<�]]?       �	(ӄfc�A�*

loss��=�_s,       �	�d�fc�A�*

lossFe�<��@�       �	r��fc�A�*

loss�~�:W=;       �	��fc�A�*

loss��6=�׊�       �	�,�fc�A�*

loss�<x:�       �	���fc�A�*

loss�M�;��QN       �	�W�fc�A�*

loss��;���       �	c�fc�A�*

lossD��< q�u       �	��fc�A�*

lossi�7=@�9       �	��fc�A�*

loss"�<��?K       �	⭊fc�A�*

lossb<K#^       �	F@�fc�A�*

loss��<(�1       �	�Ӌfc�A�*

loss[�Q<h9��       �	�z�fc�A�*

loss��=i�       �	-#�fc�A�*

loss��=dZ<�       �	ػ�fc�A�*

loss�<�^u�       �	�d�fc�A�*

loss<$ =�4Qd       �		��fc�A�*

loss(�H<�*�       �	���fc�A�*

loss�˕<�cW-       �	B�fc�A�*

loss���=��       �	��fc�A�*

lossc�;���       �	χ�fc�A�*

lossR�!=ĳ�       �	�/�fc�A�*

loss��<��"�       �	F͒fc�A�*

lossX��=YI��       �	]o�fc�A�*

loss@�S:V�_|       �	h��fc�A�*

loss:��:�� �       �	d�fc�A�*

loss�6<���       �	���fc�A�*

loss���<P��s       �	,��fc�A�*

loss���<�4��       �	|E�fc�A�*

loss�H=o��       �	*��fc�A�*

loss�<R>H       �	���fc�A�*

loss�c<۬�       �	훚fc�A�*

loss���<&wo�       �	;�fc�A�*

loss�!;(�*       �	F�fc�A�*

lossTz�<�ǈ.       �	���fc�A�*

loss\��;��       �	�fc�A�*

loss��!>��х       �	P6�fc�A�*

loss썊<���0       �	R*�fc�A�*

loss�=��       �	#�fc�A�*

loss/J?<���       �	ۉ�fc�A�*

loss��S;uD4�       �	)�fc�A�*

loss%�N<c%�       �	�¢fc�A�*

loss���:*q��       �	�Y�fc�A�*

lossdI�=�&�       �	t�fc�A�*

losskc<J.�u       �	��fc�A�*

loss���=$.Ad       �	��fc�A�*

loss���<����       �	,��fc�A�*

loss}��:t$t�       �	l��fc�A�*

loss���<�(3       �	Z+�fc�A�*

loss�;M��       �	�\�fc�A�*

loss^��<"+6�       �	�p�fc�A�*

loss&jP;+��q       �	
�fc�A�*

loss��=����       �	���fc�A�*

loss��W=����       �	[A�fc�A�*

loss�s�<���       �	+߫fc�A�*

loss�=ۮ�       �	���fc�A�*

loss��=�)�       �	��fc�A�*

loss�I�=;E�>       �	,��fc�A�*

loss,];2���       �	\�fc�A�*

loss�1;�̒i       �	�fc�A�*

lossS}\=ؽ��       �	w��fc�A�*

loss�=��T       �	�E�fc�A�*

loss�b=fK�X       �	��fc�A�*

lossX!�<��Y�       �	ˁ�fc�A�*

lossnK<S��       �	X�fc�A�*

lossx��;�=��       �	J��fc�A�*

loss�,�<�2�       �	UM�fc�A�*

loss�%�;`�Z2       �	��fc�A�*

loss�x�;:�       �	��fc�A�*

loss!I=�e�-       �	��fc�A�*

lossO<�tX%       �	&�fc�A�*

loss�=�v�       �	̳�fc�A�*

losso�-=d��       �	�M�fc�A�*

loss��<3�       �	{L�fc�A�*

losss)<�7�$       �	l�fc�A�*

loss��=����       �	���fc�A�*

loss��K<�<Oq       �	&�fc�A�*

lossL6=��k       �	u��fc�A�*

loss�;q�EE       �	�H�fc�A�*

loss��=�0/�       �	��fc�A�*

loss��;��J�       �	r��fc�A�*

loss�+_<uAH       �	�;�fc�A�*

loss�}V=~}>�       �	�۽fc�A�*

loss>�!;�_��       �	�y�fc�A�*

lossWY�<ҕS�       �	��fc�A�*

loss��;��j       �	}��fc�A�*

loss�ő;���       �	�V�fc�A�*

lossC�<�$��       �	@��fc�A�*

loss��d;��h       �	E��fc�A�*

lossqW=��       �	�?�fc�A�*

loss�Y�;�0x       �	L��fc�A�*

loss�w6=�|2�       �	���fc�A�*

lossɇ<�B�=       �	��fc�A�*

loss�(5;�%I       �	f��fc�A�*

loss�H�;�BM}       �	<i�fc�A�*

loss�];�T&�       �	��fc�A�*

loss<㪢�       �	��fc�A�*

loss=w\�       �	ML�fc�A�*

loss��<���       �	�7�fc�A�*

lossr�W=;G�       �	���fc�A�*

loss�X;�_3�       �	�i�fc�A�*

loss&h�=9SX       �	v��fc�A�*

loss��=�	E�       �	u��fc�A�*

lossO<5=C�.        �	9'�fc�A�*

loss�< E       �	��fc�A�*

loss\��:�X��       �	U�fc�A�*

lossT>�9͍��       �	,��fc�A�*

loss��y<N���       �	j��fc�A�*

loss*^�:X��}       �	4H�fc�A�*

loss�ܡ:q���       �	6��fc�A�*

lossd�;tW7�       �	=��fc�A�*

loss�0:J�:�       �	�V�fc�A�*

loss���<�ꗉ       �	[�fc�A�*

loss���9��)       �	���fc�A�*

lossx�	9���@       �	i�fc�A�*

lossm.�8��Ĵ       �	�fc�A�*

loss�<=�"A�       �	i��fc�A�*

loss�<�<�7g       �	RI�fc�A�*

loss��$<��H       �	���fc�A�*

lossw�j:4��       �	k�fc�A�*

loss�o2<��c@       �	'�fc�A�*

loss�1�=�͂�       �	��fc�A�*

lossf[D:�y��       �	[a�fc�A�*

loss&�^>Tz�       �	j��fc�A�*

loss�Om<��q       �	f��fc�A�*

loss��'<֣e�       �	��fc�A�*

losszy�;���       �	.��fc�A�*

loss��j;�`�       �	gD�fc�A�*

loss��>=��8�       �	4��fc�A�*

loss~X<9!�       �	�l�fc�A�*

loss���;�]�p       �	���fc�A�*

loss��X=p�_�       �	}��fc�A�*

loss��;�J��       �	^-�fc�A�*

lossH^�=ZXf�       �	L��fc�A�*

loss�y�<R��       �	\X�fc�A�*

loss#=�"G�       �	}<�fc�A�*

loss�=�OX       �	s��fc�A�*

lossFE<"�m�       �	ލ�fc�A�*

loss1.�;��ճ       �	80�fc�A�*

loss���=�r�       �	��fc�A�*

lossi�P=a�-J       �	?t�fc�A�*

loss�Q�;�       �	j�fc�A�*

loss�;�	��       �	��fc�A�*

loss��{<0(9       �	��fc�A�*

loss�/n=UHB�       �	Ͱ�fc�A�*

loss ڬ:K�o�       �	��fc�A�*

losscY�;��\       �	�e�fc�A�*

loss<��:��5       �	� �fc�A�*

loss`�<�|�       �	��fc�A�*

loss��O=��٢       �	t�fc�A�*

loss�T=I�<�       �	8�fc�A�*

loss��<��)       �	���fc�A�*

loss���=��O0       �	PP�fc�A�*

loss:=�u�       �	���fc�A�*

loss��,<u�       �	���fc�A�*

loss@�q:,�q       �	�!�fc�A�*

loss!�=�p?       �	�Z�fc�A�*

loss�ŋ<F��       �	���fc�A�*

loss�D�<w       �	��fc�A�*

loss(�=aP�       �	��fc�A�*

lossT�<sh	e       �	�p�fc�A�*

loss.Jo<��u       �	K�fc�A�*

loss���:��u�       �	 ��fc�A�*

loss�(<��       �	�d�fc�A�*

loss3�<��f       �	g�fc�A�*

loss̪�<^P       �	i��fc�A�*

loss��;���       �	d�fc�A�*

loss��E=��5]       �	��fc�A�*

loss��=$ʼ�       �	:��fc�A�*

loss
W;�o�N       �	�1�fc�A�*

loss���<+2�d       �	���fc�A�*

loss/�8;K��       �	m�fc�A�*

loss��:����       �	 	�fc�A�*

lossz7-<�2�F       �	��fc�A�*

loss\&�<�~��       �	�fc�A�*

lossS
=|��       �	B{fc�A�*

loss�=�`�W       �	�fc�A�*

lossο�<U��       �	��fc�A�*

loss���;�       �	�Lfc�A�*

lossts�<+�	�       �	t�fc�A�*

loss�Ԗ<i�ۣ       �	�fc�A�*

loss�8=�С�       �	�%fc�A�*

loss1=@���       �	��fc�A�*

loss�T<kK��       �	�dfc�A�*

loss��<ÌG�       �	��fc�A�*

loss��
=�`2�       �	��fc�A�*

loss���<0Ϗ       �	ofc�A�*

lossƒ\<
Q�       �	fc�A�*

loss�z<I��       �	��fc�A�*

lossdaG<�{       �	�fc�A�*

loss4i�<�Y�,       �	[zfc�A�*

lossM=Qpt       �	yy fc�A�*

loss��P<�<�       �	q!fc�A�*

loss2�<��?       �	'�!fc�A�*

loss�U�=�o�       �	�\"fc�A�*

lossD��:�U~^       �	a�"fc�A�*

loss��=�Ns�       �	8�#fc�A�*

lossC/:�uf-       �	�B$fc�A�*

lossx��:��       �	��$fc�A�*

loss��;:-.!       �	4�%fc�A�*

loss��;!�{�       �	�&fc�A�*

loss���;�n}3       �	-�&fc�A�*

loss���;J�6�       �	@O'fc�A�*

lossD��;6�C       �	��'fc�A�*

loss�]<����       �	�(fc�A�*

loss�>=r���       �	�*)fc�A�*

loss��'<)�^�       �	��)fc�A�*

lossh�;�'�Q       �	�j*fc�A�*

loss��<AJ��       �	�+fc�A�*

loss!m�:4�'�       �	�+fc�A�*

lossng7<��ַ       �	J,fc�A�*

loss���=�1�3       �	�,fc�A�*

loss�3M<�,N�       �	܄-fc�A�*

lossQ�<��cJ       �	2.fc�A�*

lossq�=;���       �	M�.fc�A�*

loss��;&��       �	Y/fc�A�*

loss1ui<���       �	�/fc�A�*

lossF�<����       �	ڍ0fc�A�*

loss�Ē;�g��       �	>$1fc�A�*

lossz��<�q�       �	P�1fc�A�*

loss+'= �ܫ       �	�\2fc�A�*

loss���;���       �	��3fc�A�*

lossry;V��       �	��4fc�A�*

loss�I�:I^�       �	��5fc�A�*

lossF� =irc       �	�6fc�A�*

loss�Q=�b^�       �	��7fc�A�*

loss̓�=�Ek       �	�8fc�A�*

loss��f=���B       �	Tq9fc�A�*

loss�8�:�+\H       �	K":fc�A�*

lossy;�O       �	(;fc�A�*

loss�z:$��       �	QN<fc�A�*

loss�XK:=:_       �	6�<fc�A�*

loss~�<�I�       �	�>fc�A�*

lossκ�={_n       �	�>fc�A�*

loss�`�;X�=       �	r�?fc�A�*

loss^�<�5t�       �	�'@fc�A�*

lossa�);.�(       �	%!Afc�A�*

loss[�:���       �	FBfc�A�*

losstT<�ǘF       �	��Bfc�A�*

loss��<��       �	�Cfc�A�*

loss��<?��       �	+2Dfc�A�*

loss5�=����       �	$�Dfc�A�*

loss�y�<V�L       �	UmEfc�A�*

loss��=��
       �	�Ffc�A�*

loss��<@���       �	ʧFfc�A�*

loss��/<���       �	7�Gfc�A�*

loss�/=#3[4       �	VFHfc�A�*

loss�5<�b��       �	P�Hfc�A�*

loss�$:� �<       �	�vIfc�A�*

loss��4;D�W�       �	�Jfc�A�*

loss�I�:l�       �	BKfc�A�*

loss�&�9k�R�       �	�Kfc�A�*

loss�F=WM��       �	�7Lfc�A�*

lossC=r��B       �	�^Mfc�A�*

loss�Ь<�i�       �	�Nfc�A�*

loss��	;�E�       �	�Nfc�A�*

loss�<ȕ�p       �	�?Ofc�A�*

lossh��:M$�       �	�Ofc�A�*

loss\Տ<ƻK�       �	ŎPfc�A�*

loss�|�;:���       �	�)Qfc�A�*

loss�@=��C       �	��Qfc�A�*

loss��;#��b       �	�^Rfc�A�*

lossz�;�꼓       �	AJSfc�A�*

lossz�<�ʯ
       �	/�Sfc�A�*

losse�=�c�#       �	�wTfc�A�*

lossV�_;�;��       �	cUfc�A�*

loss8��;[r�R       �	�Vfc�A�*

loss��'=�ni       �	0�Vfc�A�*

loss-�d;GVB       �	q=Wfc�A�*

lossX�=�>��       �	��Wfc�A�*

loss1/<e�}       �	��Xfc�A�*

loss��K;2Ad�       �	Yfc�A�*

loss���:�pv       �	r�Yfc�A�*

lossn�:�*��       �	��Zfc�A�*

loss��[;G���       �	�2[fc�A�*

loss���<�>��       �	g�[fc�A�*

loss&��;�k��       �	7o\fc�A�*

loss���<`Aj�       �	k]fc�A�*

loss���;���       �	M�]fc�A�*

loss���:C~6�       �	ڐ^fc�A�*

loss��:����       �	�-_fc�A�*

loss�=ɖ�       �	��_fc�A�*

lossb�:�k�       �	�o`fc�A�*

loss��=*�x:       �	�	afc�A�*

loss�5<���[       �	.�afc�A�*

lossa�<��;�       �	JFbfc�A�*

loss�*<��4       �	��bfc�A�*

loss{��<Ϻ�       �	^�cfc�A�*

loss�j�<5�$       �	�dfc�A�*

loss���<�J�       �	�dfc�A�*

loss�U<��2J       �	RFefc�A�*

loss��=<��6�       �	a�efc�A�*

loss�x ;���       �	�qffc�A�*

loss@g�;Qi�m       �	*gfc�A�*

lossm�
<^1�       �	_�gfc�A�*

lossl@<�l�.       �	k)hfc�A�*

losst��<��UU       �	
�hfc�A�*

loss�g :(��       �	�\ifc�A�*

loss�X>�Vw-       �	�7jfc�A�*

loss1�@<[�+       �	��jfc�A�*

lossm=�}O       �	_]lfc�A�*

lossͽ=A��       �	a�lfc�A�*

loss���;A`�G       �	Z�mfc�A�*

lossZ�;6�       �	!;nfc�A�*

loss�`�;s ��       �	$ofc�A�*

lossK�;�u9       �	��ofc�A�*

loss8DW<��*�       �	�\pfc�A�*

loss%t<�o�       �	��pfc�A�*

loss���<���S       �	h�qfc�A�*

loss��_;%R%       �	�*rfc�A�*

loss�gO:�[>�       �	Y�rfc�A�*

lossҫ�:�       �	]�sfc�A�*

loss��<����       �	irtfc�A�*

loss_Q�<@�(6       �	�vufc�A�*

loss(!>m1@�       �	��vfc�A�*

loss�M�<88&       �	��wfc�A�*

loss�	<���       �	3Rxfc�A�*

loss�
=���       �	�Kyfc�A�*

loss*�X=�+��       �	�Dzfc�A�*

loss�<��/�       �	�{fc�A�*

loss���:m]z#       �	��{fc�A�*

loss洌:�&e       �	$H|fc�A�*

loss��<G��8       �	�|fc�A�*

loss�0�;�&       �	R�}fc�A�*

loss�&�;Io2�       �	hB~fc�A�*

loss�p�;�}�&       �	X�~fc�A�*

loss�<����       �	�fc�A�*

lossC�=Q�gM       �	�4�fc�A�*

loss�c;�c�.       �	cՀfc�A�*

loss\C=.w�       �	Kx�fc�A�*

loss/j�<9�h       �	z�fc�A�*

lossqʧ<e�b�       �	(��fc�A�*

losshb�:���       �	eS�fc�A�*

loss�r=�<L�       �	��fc�A�*

loss���<�       �	ҍ�fc�A�*

loss���<,�H�       �	�#�fc�A�*

loss��:j���       �	ü�fc�A�*

lossQ�=k�e2       �	6W�fc�A�*

loss�7<��\k       �	��fc�A�*

loss�sE<��<       �	��fc�A�*

loss���=���       �	Q.�fc�A�*

loss�<�u�&       �	Xʈfc�A�*

loss�&�;��_        �	#g�fc�A�*

loss���<���i       �	��fc�A�*

loss��:}���       �	L��fc�A�*

loss]��:�;�g       �	�@�fc�A�*

loss@cE=�B��       �	��fc�A�*

loss��:�ӆ8       �	��fc�A�*

loss#�;<�l�       �	�)�fc�A�*

loss��^;~��=       �	��fc�A�*

loss*�-=r��       �	���fc�A�*

loss �l<���N       �	j0�fc�A�*

lossӠ<i��)       �	�ɏfc�A�*

loss�b�:���       �	kd�fc�A�*

loss�}�=�_k       �	l�fc�A�*

loss�e;��       �	T��fc�A�*

loss`B�:� �       �	$E�fc�A�*

lossu1;
U�w       �	��fc�A�*

loss���;���r       �	��fc�A�*

loss�/�<r�b�       �	��fc�A�*

loss\�<��%`       �	鶔fc�A�*

loss��5=J�P       �	"T�fc�A�*

loss��.<㝛|       �	\�fc�A�*

lossh�H=v{?�       �	��fc�A�*

loss��q<���       �	D��fc�A�*

loss@��;�Q�       �	�'�fc�A�*

loss��;��X6       �	���fc�A�*

loss��[;}0�       �	)Z�fc�A�*

loss8��;�;�N       �	��fc�A�*

losss��<���w       �	b��fc�A�*

loss.��;ǓF       �	��fc�A�*

loss!(�<�q�       �	���fc�A�*

loss�#<z���       �	�^�fc�A�*

loss���<E�#       �	g�fc�A�*

lossѨ<W��$       �	��fc�A�*

lossDZ<x|L.       �	��fc�A�*

loss!R;Q=#\       �	)��fc�A�*

loss��
<�w`       �	�A�fc�A�*

loss���<Ƨ	�       �	�՟fc�A�*

loss�0=	+�       �	"q�fc�A�*

loss���=b�Mj       �	m�fc�A�*

loss�<
,9Y       �	xϡfc�A�*

loss��<�oe       �	�m�fc�A�*

loss{;�Z�z       �	�	�fc�A�*

loss�#�:�eg�       �	Q��fc�A�*

loss��;��W?       �	�>�fc�A�*

loss��c<^a$v       �	�ؤfc�A�*

loss���<�P@�       �	�y�fc�A�*

lossu�='        �	��fc�A�*

loss�=�1�\       �	��fc�A�*

loss0/�=	�0�       �	2W�fc�A�*

loss��*=6|�K       �	9�fc�A�*

loss�O�<~�=       �	㉨fc�A�*

loss
$�=�˖�       �	%�fc�A�*

loss
i;�޳       �	Gǩfc�A�*

loss�1<J\�       �	�^�fc�A�*

lossRO�=z��3       �	��fc�A�*

loss�F�;�%N       �	5��fc�A�*

lossIW�:��b�       �	�-�fc�A�*

loss�U>f���       �	���fc�A�*

loss�l�<BA��       �	K��fc�A�*

lossd�<�YD�       �	R)�fc�A�*

loss��;��ڈ       �	v®fc�A�*

loss�b�<�	H       �	"U�fc�A�*

lossc��:�DoC       �	�Ͱfc�A�*

losseL[;�Gf       �	�g�fc�A�*

loss��4;9/�       �	�fc�A�*

loss��<�oA�       �	���fc�A�*

loss�x<�rڡ       �	v7�fc�A�*

loss<,�<��a       �	�fc�A�*

lossl=N9R�       �	�дfc�A�*

loss�a�=�,�4       �	��fc�A�*

loss�&�:(d�*       �	��fc�A�*

loss��;�I       �	�8�fc�A�*

lossT>+<ZltC       �	`�fc�A�*

loss<�<Z��       �	��fc�A�*

loss�C�<�?�       �	�˹fc�A�*

loss��<'i��       �	a�fc�A�*

loss���<���       �	>�fc�A�*

loss�FK=���=       �	��fc�A�*

lossjD�:�u4       �	� �fc�A�*

loss�;Ռ�       �	��fc�A�*

loss���<���       �	j�fc�A�*

loss��>�h��       �	���fc�A�*

loss��=ۉ%�       �	�2�fc�A�*

loss��m=���C       �	��fc�A�*

loss $�:>r�]       �	�l�fc�A�*

loss�1:�`�       �	��fc�A�*

loss���</�ˍ       �	=��fc�A�*

loss�%�;�2Z       �	�5�fc�A�*

loss��<B-@�       �	G��fc�A�*

lossInY=3W�]       �	_^�fc�A�*

loss�H�<v�GC       �	b��fc�A�*

loss��=�50       �	!��fc�A�*

loss��:��Х       �	�0�fc�A�*

loss�;�_��       �	���fc�A�*

losst�<�h�       �	@i�fc�A�*

lossI�<�G��       �	:�fc�A�*

loss��
=r�<       �	=��fc�A�*

loss���=�;�       �	v5�fc�A�*

loss�<����       �	���fc�A�*

loss,w(<�>�}       �	�r�fc�A�*

loss�k�<�l�$       �	m�fc�A�*

loss$��;���       �	$��fc�A�*

loss�<qg�       �	&Q�fc�A�*

loss��c=H_�       �	���fc�A�*

loss�m�<��       �	���fc�A�*

loss���=�I�       �	]1�fc�A�*

loss�Į<Ӊ~5       �	d��fc�A�*

lossEZ`;ۉ|�       �	�a�fc�A�*

lossR;M�n       �	���fc�A�*

loss�(=�훊       �	���fc�A�*

lossZ*�<��_+       �	&:�fc�A�*

lossف:�h��       �	 ��fc�A�*

loss���;yl��       �	Dm�fc�A�*

lossW�<fM       �	��fc�A�*

lossa��;}�|�       �	���fc�A�*

loss��<@�rO       �	�@�fc�A�*

loss/u�;8b/�       �	I��fc�A�*

loss��=�ކ       �	7n�fc�A�*

lossz�m;XOE�       �	��fc�A�*

loss��;9��h       �	E��fc�A�*

loss�#y<j��G       �	�9�fc�A�*

lossO�B:	Z��       �	��fc�A�*

loss���<�>�       �	ji�fc�A�*

loss���<����       �	D��fc�A�*

loss�6j;ڝx�       �	��fc�A�*

lossMq�< �       �	�+�fc�A�*

loss#��;t5�       �	���fc�A�*

loss�k�<�GE�       �	���fc�A�*

loss�-�<Êw)       �	Y1�fc�A�*

loss	�<�7�       �	���fc�A�*

loss\��<��       �	jg�fc�A�*

lossG)=`m�       �	C��fc�A�*

loss��<��       �	З�fc�A�*

losson�:��~b       �	79�fc�A�*

lossA�	<��>|       �	���fc�A�*

loss��<�;�s       �	�y�fc�A�*

loss�9�=F��       �	w�fc�A�*

loss�<=�|�       �	s��fc�A�*

loss8�>"m��       �	�u�fc�A�*

loss���9mw�       �	��fc�A�*

lossܱ�;%uр       �	ؼ�fc�A�*

loss2�%<{Mm�       �	�`�fc�A�*

loss6l8<X�Y�       �	O�fc�A�*

loss��/<����       �	J��fc�A�*

loss >�<XjU        �	z5�fc�A�*

loss`R�<;� 8       �	b��fc�A�*

lossU*=�O�       �	w�fc�A�*

loss��:c�A�       �	��fc�A�*

loss��)<��W�       �	���fc�A�*

loss!�=��l�       �	�;�fc�A�*

lossmD=G<�       �	���fc�A�*

loss�|<�a�k       �	�n�fc�A�*

loss��a=�	�U       �	�fc�A�*

lossI�<���       �	���fc�A�*

loss_�;`���       �	�>�fc�A�*

loss���;=���       �	���fc�A�*

loss���=[xZ       �	~n�fc�A�*

loss��<��~J       �	��fc�A�*

loss��X=QCY�       �	s��fc�A�*

loss1�g=����       �	�3�fc�A�*

lossl��<����       �	:��fc�A�*

loss*�7;���h       �	j�fc�A�*

loss��;%Jh       �	��fc�A�*

loss��:�$�       �	���fc�A�*

lossu =�[(       �	7R�fc�A�*

loss�V<^C|�       �	V��fc�A�*

loss8�z;/q1       �	���fc�A�*

loss<m�<?��       �	�>�fc�A�*

loss��F=�&V�       �	��fc�A�*

loss���=E'eA       �	$&�fc�A�*

loss�D�;�\�&       �	B
�fc�A�*

loss�O:j���       �	z��fc�A�*

loss4�=<�L�       �	M�fc�A�*

loss��J=� U       �	d��fc�A�*

loss��=3��N       �	Y��fc�A�*

loss�<=��)�       �	�(�fc�A�*

lossr%�<>�2       �	 ��fc�A�*

lossm�R<2}�       �	)��fc�A�*

lossOp�<2k��       �	{I�fc�A�*

lossPܠ<�è�       �	��fc�A�*

loss�\<��w�       �	���fc�A�*

loss�[U;�+       �	y]�fc�A�*

loss^�<�[�Q       �	���fc�A�*

loss�[=%��       �	���fc�A�*

loss`J�;�*A       �	�+ fc�A�*

loss]{>l\[7       �	-� fc�A�*

loss���;롡       �	lfc�A�*

lossa��;V3E       �	(fc�A�*

loss�;��E       �	Y�fc�A�*

loss,��;�X�       �	�Dfc�A�*

lossH��:(���       �		�fc�A�*

loss�Y�<F���       �	�wfc�A�*

loss#�	=t�Z�       �	Rfc�A�*

loss���<	]_i       �	��fc�A�*

loss��	=���       �	l?fc�A�*

loss��=�p~9       �	�fc�A�*

loss)4%=�2��       �	��fc�A�*

lossp�<���       �	h%fc�A�*

loss��<�P�       �	��fc�A�*

loss,h <c�^�       �	ŏ	fc�A�*

loss]�<���       �	�4
fc�A�*

loss�ӓ;\�I       �	�
fc�A�*

lossI��;=�
�       �	infc�A�*

lossگ�<��B�       �	�fc�A�*

lossI��;�D       �	��fc�A�*

loss���<D�^�       �	�Mfc�A�*

loss�F<��#�       �	C�fc�A�*

loss��<�V<{       �	֋fc�A�*

losscS<�<�       �	b0fc�A�*

losseZ�<��	       �	{�fc�A�*

loss��;� �z       �	�zfc�A�*

loss8�<p��       �	q<fc�A�*

loss2=��,       �	��fc�A�*

lossv�t=���       �	"�fc�A�*

lossZʡ<��#       �	�(fc�A�*

loss�*;�"{h       �	�fc�A�*

lossI�:<���       �	�ffc�A�*

loss�{�=&�9       �	!fc�A�*

loss���<Q�z^       �	f�fc�A�*

lossN��:�gyM       �	�fc�A�*

loss��	<���a       �	�^fc�A�*

loss�#�;0���       �	^�fc�A�*

loss�u/<e�X       �	a�fc�A�*

loss,s�<UbA;       �	�!fc�A�*

loss���<C�&       �	��fc�A�*

loss��'=*i�       �	�Wfc�A�*

loss�Q�;˷1�       �	��fc�A�*

loss7�@<��8�       �	�fc�A�*

loss��O=p��^       �	�$fc�A�*

loss
V�;� K       �	j�fc�A�*

loss��<}C��       �	�[fc�A�*

lossi';��$       �	��fc�A�*

loss[SY;��Z�       �	v�fc�A�*

loss&y�;�
x�       �	+/fc�A�*

loss���<'�bz       �	p�fc�A�*

loss��`<UHQ�       �	�s fc�A�*

loss٪�<�b��       �	�!fc�A�*

loss
�(<��       �	6�!fc�A�*

loss�q�<�8
1       �	nL"fc�A�*

loss{�{<�|[�       �	]�"fc�A�*

loss��;��$       �	w�#fc�A�*

loss4�<l	�=       �	�$fc�A�*

loss�:�=�a�       �	�$fc�A�*

losse��=�(�       �	�F%fc�A�*

losssV�=v�       �	��%fc�A�*

loss��;w�mK       �	ǃ&fc�A�*

loss��/=�2�Y       �	�#'fc�A�*

lossa~<z��       �	5�'fc�A�*

loss��;"Rpd       �	aT(fc�A�*

loss��<r�@F       �	:)fc�A�*

loss\�P<�1er       �	��)fc�A�*

loss-=�QU       �	%<*fc�A�*

lossMO6<��W�       �	+�*fc�A�*

lossJJ$=�]ͨ       �	�z+fc�A�*

lossn\=���2       �	U,fc�A�*

loss)1><��7&       �	��,fc�A�*

losso=!Zt       �	zT-fc�A�*

loss���<�S�       �	@.fc�A�*

lossN�<ɔ=U       �	78/fc�A�*

loss^6�;#�       �	��/fc�A�*

loss��%=��       �	Hn0fc�A�*

loss�n�<�_�       �	^1fc�A�*

loss�[<�p�       �	�1fc�A�*

losst��;_�[�       �	ZI2fc�A�*

loss;y2;8�$       �	)�2fc�A�*

lossJu<���R       �	��3fc�A�*

loss�>�=|�       �	�+4fc�A�*

lossIUW;_> �       �	��4fc�A�*

loss��<Y��I       �	Z�5fc�A�*

loss�n�<���       �	��6fc�A�*

loss�_�<�?C�       �	n7fc�A�*

loss1��;�W�8       �	�W8fc�A�*

loss��
=G�0       �	v�8fc�A�*

lossz.	=�~�       �	�:fc�A�*

loss?(�=1*(       �	F�:fc�A�*

loss�s";�ܕ        �	8J;fc�A�*

loss�:t<�-@�       �	��;fc�A�*

loss䌏<c�W�       �	�==fc�A�*

loss=w�<� ��       �	��=fc�A�*

loss��<;By       �	,�>fc�A�*

loss��<��|�       �	��?fc�A�*

loss�1=@sl�       �	 s@fc�A�*

loss3�G:�p       �	Afc�A�*

loss�y�<E	Ȯ       �	˿Afc�A�*

loss�r�;^��       �	�uBfc�A�*

lossƘ\=moRf       �	�Cfc�A�*

loss3�;�DO       �	��Dfc�A�*

loss�&�;Ӿ�       �	�sEfc�A�*

loss�@�<�]D       �	�mFfc�A�*

loss�<�5G       �	s�Gfc�A�*

loss-=�F��       �	>�Hfc�A�*

loss�,�<~o�N       �	��Ifc�A�*

loss�n(<O�       �	=cJfc�A�*

loss�մ:{��3       �	NKfc�A�*

loss��2>��;       �	�Lfc�A�*

lossa��;���P       �	�:Mfc�A�*

lossh��;=�       �	9�Mfc�A�*

losso�<�8��       �	JOfc�A�*

loss:#;��%       �	%yPfc�A�*

loss�=���       �	yQfc�A�*

lossA¶9qOĉ       �	��Qfc�A�*

loss�e�==���       �	�zRfc�A�*

losss�;���M       �	�9Sfc�A�*

loss���<>Ba�       �	��Sfc�A�*

loss���<V:�       �	��Tfc�A�*

loss�E:�;K       �	�3Ufc�A�*

loss�L�:p��       �		�Ufc�A�*

loss�dM;DX�       �	�Vfc�A�*

lossƃ�:����       �	~Wfc�A�*

loss��<0쾱       �	��Wfc�A�*

loss�7:<h7�       �	h^Xfc�A�*

loss�>��p       �	�Xfc�A�*

loss�u<��"�       �	 �Yfc�A�*

loss_4^<�͈i       �	�6Zfc�A�*

lossr4�;�b       �	��Zfc�A�*

loss.�~<q�\�       �	q[fc�A�*

loss��&;;�       �	�\fc�A�*

loss��1<��       �	��\fc�A�*

lossU�<#_�#       �	��]fc�A�*

loss;Rf;k���       �	�*^fc�A�*

losss�y;���       �	��^fc�A�*

loss���<z���       �	^_fc�A�*

loss��<Av�       �	�`fc�A�*

loss;��;EU��       �	f�`fc�A�*

loss�}<"<��       �	�<afc�A�*

loss��<�t��       �	��afc�A�*

lossX�<4T[�       �	Ήbfc�A� *

loss?�;�_�       �	Tcfc�A� *

lossE�m:��       �	>�cfc�A� *

loss o><6߁�       �	�Edfc�A� *

loss&�j<���       �	��dfc�A� *

lossT@�<��p`       �	��efc�A� *

lossi��<5�<       �	nkffc�A� *

lossU�:�       �	ygfc�A� *

lossvN�9�X�       �	d�gfc�A� *

lossZ�<(�d       �	�hhfc�A� *

loss�\�;��?       �	Lifc�A� *

loss|ԅ:�Qx       �	ܹifc�A� *

lossA�*;�d�*       �	+ljfc�A� *

loss�1=���F       �	kfc�A� *

loss |�;n��       �	��kfc�A� *

loss�:�L �       �	?qlfc�A� *

lossn��<c[O       �	]mfc�A� *

lossy�<�V��       �	�mfc�A� *

loss׷�<]_	       �	�]nfc�A� *

loss��;����       �	T�nfc�A� *

loss��]=�I�=       �	�ofc�A� *

lossSE=�M"i       �	RHpfc�A� *

losslW�:A��       �	|~qfc�A� *

lossE�&:���       �	�rfc�A� *

loss��k<���       �	<�rfc�A� *

loss�F�;��j�       �	�Wsfc�A� *

loss�6�<p�$	       �	j�sfc�A� *

lossc�m:tq/�       �	�tfc�A� *

loss$߃;G���       �	�ufc�A� *

loss�[�<��F�       �	��vfc�A� *

loss��:ko�J       �	��wfc�A� *

loss,�?;�K�[       �	Mxfc�A� *

loss��<��R       �	t	yfc�A� *

lossa�=��T�       �	G9zfc�A� *

lossiς:�ΐ       �	*�zfc�A� *

loss��r:�[��       �	�z{fc�A� *

loss W=XM�       �	M|fc�A� *

lossO�;��       �	�|fc�A� *

loss�iH;�.��       �	$B}fc�A� *

lossh�;��X�       �	��}fc�A� *

loss��;�=�       �	s~fc�A� *

lossMε:pB�       �	�Efc�A� *

lossH�x<L�}       �	�fc�A� *

loss�9�}�       �	Fz�fc�A� *

loss�c�:i��;       �	��fc�A� *

loss&K�9�FK�       �	���fc�A� *

loss]M:��*�       �	sL�fc�A� *

loss\�e8����       �	X�fc�A� *

loss&};k&       �	�փfc�A� *

loss���</=�       �	p�fc�A� *

lossU;ڊ�       �	F'�fc�A� *

loss|��7����       �	��fc�A� *

lossJ �;�y1       �	Q��fc�A� *

loss춫=Yʊr       �	g)�fc�A� *

loss;��9 .9       �	}·fc�A� *

loss��	>xI�       �	Jz�fc�A� *

loss>֣=�g�b       �	��fc�A� *

loss�5�=8���       �	崉fc�A� *

lossM#�;�^R�       �	�O�fc�A� *

loss`�;��i       �	��fc�A� *

loss� �<�Ϋ�       �	^��fc�A� *

loss@c<Po       �	!�fc�A� *

loss��/<	0�N       �	+��fc�A� *

loss��<�	�q       �	|`�fc�A� *

loss�P<i5O�       �	y�fc�A� *

loss�=���,       �	;��fc�A� *

lossFɜ;Lr��       �	�F�fc�A� *

loss��;B��       �	��fc�A� *

loss��=	�P�       �	���fc�A� *

loss���=(�!       �	�fc�A� *

loss��<�6��       �	���fc�A� *

lossDo'<�}       �	�Q�fc�A� *

loss��=}��Q       �	��fc�A� *

loss���<�D��       �	}�fc�A� *

loss�q�:�;��       �	=��fc�A� *

loss�d=9�!�       �	&5�fc�A� *

loss���<p�C�       �	�fc�A� *

loss\p;� �5       �	D6�fc�A� *

loss���;�;�h       �	 �fc�A� *

loss� �;s�qk       �	��fc�A� *

loss�0<B��       �	�K�fc�A� *

lossF��<���       �	~s�fc�A� *

loss��.;��0V       �	#�fc�A� *

loss�i�=�\Ʈ       �	���fc�A� *

loss6
�;U�7�       �		R�fc�A� *

lossdG=��J       �	E��fc�A� *

loss|��;�*�       �	���fc�A� *

loss���<�12�       �	f.�fc�A� *

loss;�c<_��       �	�ʞfc�A� *

loss�@w<��       �	�h�fc�A� *

loss�GH=�ƆC       �	N	�fc�A� *

loss��=�!�       �	���fc�A� *

loss��\=�oX       �	�M�fc�A� *

loss8��:+�S^       �	W�fc�A� *

loss�d�<���       �	�fc�A� *

loss_ʜ;.`S�       �	/2�fc�A� *

loss�#;��&�       �	�[�fc�A� *

loss�VE<w�H�       �	f��fc�A� *

loss,>=�s�c       �	.��fc�A� *

lossF2=U�       �	�H�fc�A� *

lossH�R=��j�       �	�fc�A� *

loss/��:\8eB       �	U��fc�A� *

lossʭ�;DAjq       �	�%�fc�A� *

loss� �;����       �	Xɨfc�A� *

loss���:� "*       �	'g�fc�A� *

lossn �<5��       �	���fc�A� *

loss��<T��       �	�N�fc�A� *

lossO
k=�_�S       �	d:�fc�A� *

loss3��:�g��       �	���fc�A� *

loss�=�;AX)       �	Ό�fc�A� *

losso�<)P�       �	<0�fc�A� *

loss!��;��       �	���fc�A� *

lossq��<&�       �	7��fc�A� *

loss@�H=O�s       �	rQ�fc�A� *

loss�_";��]       �	��fc�A� *

loss��;��R       �	f��fc�A� *

loss,�"=W<�e       �	��fc�A� *

loss^�;񯱸       �	#��fc�A� *

loss���;��=u       �	�]�fc�A� *

loss!��;�D�       �	��fc�A� *

loss=��<Y�       �	?�fc�A� *

loss���9$L�'       �	��fc�A� *

lossr��:��       �	�L�fc�A� *

loss�'�;L���       �	�fc�A� *

loss���<� 7�       �	g��fc�A� *

loss*^e<���       �	 T�fc�A� *

loss��=��r�       �	���fc�A� *

loss�\�9��en       �	�~�fc�A� *

loss��N=�8�       �	��fc�A�!*

lossL��:��mO       �	z��fc�A�!*

loss�;���       �	d>�fc�A�!*

loss!*;D&�       �	f��fc�A�!*

loss�<X<OTs^       �	�p�fc�A�!*

loss`X�;��o�       �	|�fc�A�!*

lossw�q;ƥ:       �	���fc�A�!*

lossE��;s<<#       �	�>�fc�A�!*

loss��<�z�       �	���fc�A�!*

loss��@<Ҧ�       �	�o�fc�A�!*

loss�\O=�"v       �	 �fc�A�!*

losss,m<���       �	w��fc�A�!*

lossbJ>㆘=       �	"6�fc�A�!*

loss��8;�t	]       �	��fc�A�!*

loss$�<��.       �	�f�fc�A�!*

loss_H�;D&L       �	N�fc�A�!*

loss��=�&       �	���fc�A�!*

loss,+�:��1�       �	vQ�fc�A�!*

loss ��;�c3       �	���fc�A�!*

loss�w�;Ҕ�       �	O��fc�A�!*

loss#A;�/�*       �	���fc�A�!*

loss��<F���       �	*�fc�A�!*

loss�Σ;㛝       �	p��fc�A�!*

loss��I;��m{       �	�L�fc�A�!*

loss��z</�ݎ       �	���fc�A�!*

lossi
<��A       �	n��fc�A�!*

lossx��:F
�+       �	s+�fc�A�!*

loss/�;2��       �	��fc�A�!*

loss`�<�D�       �	�[�fc�A�!*

loss��o;8�Ä       �	���fc�A�!*

loss��=��kd       �	���fc�A�!*

loss�0�<Λ�!       �	��fc�A�!*

loss���;(���       �	���fc�A�!*

loss@�a:z�       �	`w�fc�A�!*

loss���:�J$�       �	�fc�A�!*

loss��<"�V;       �	���fc�A�!*

lossl8�;�L�       �	8L�fc�A�!*

lossѡ�<�,;�       �	���fc�A�!*

loss}-0=7�^/       �	���fc�A�!*

loss�b�:<a�1       �	 a�fc�A�!*

loss�<<R�w�       �	m��fc�A�!*

lossc�k;��(       �	��fc�A�!*

loss�B�:�k[x       �	6�fc�A�!*

loss$W	<3��y       �	���fc�A�!*

lossQ�<�i�b       �	ۊ�fc�A�!*

loss<O�<��;�       �	�fc�A�!*

loss F�<�`�       �	ܷ�fc�A�!*

loss��`<8��       �	wM�fc�A�!*

loss��;�o4�       �	���fc�A�!*

loss��=�ѵ�       �	|�fc�A�!*

losszH< ŵ/       �	��fc�A�!*

loss�\�<��J�       �	���fc�A�!*

loss���98�Rh       �	�O�fc�A�!*

loss�'�:n�IK       �	���fc�A�!*

lossǾ<�L��       �	���fc�A�!*

loss��9X
��       �	_&�fc�A�!*

loss7�:���{       �	��fc�A�!*

loss�9�;�y��       �	�[�fc�A�!*

loss��"=�K       �	���fc�A�!*

loss)۽<ä��       �	���fc�A�!*

loss��
<Y�6c       �	& fc�A�!*

loss��R;OU       �	L� fc�A�!*

loss �N:�0?�       �	9dfc�A�!*

lossα;��1�       �	e�fc�A�!*

loss�ҷ<�3��       �	��fc�A�!*

loss���<�~e�       �	Q/fc�A�!*

loss��9<�a�       �	�fc�A�!*

loss�ơ;��/       �	jifc�A�!*

loss��;Cn7       �	��fc�A�!*

lossÆ`;��ٙ       �	��fc�A�!*

lossD";�g�       �	0Kfc�A�!*

loss��t<���?       �	$�fc�A�!*

loss!v;V��k       �	��fc�A�!*

lossf*~=��p�       �	�#fc�A�!*

loss!y�<X���       �	��fc�A�!*

loss��;B�       �	�T	fc�A�!*

loss���;�~]b       �	��	fc�A�!*

loss��?:E�I       �	�|
fc�A�!*

loss&9965 �       �	�fc�A�!*

loss`��;�F!       �	˼fc�A�!*

lossT�<r�a       �	�^fc�A�!*

lossz��:��?R       �	�fc�A�!*

lossș9+���       �	��fc�A�!*

loss���;��]b       �	�@fc�A�!*

loss��=;�+�       �	zfc�A�!*

lossO{�<�;��       �	�!fc�A�!*

lossx��:h!�(       �	Y�fc�A�!*

loss�O�:»e�       �	�Zfc�A�!*

loss���;Vy��       �	Z�fc�A�!*

loss�ɜ;ZM[       �	2�fc�A�!*

loss�Y<X6�       �	k+fc�A�!*

loss�d�;���       �	��fc�A�!*

lossA�;��w       �	!sfc�A�!*

lossO��:O�D3       �	R
fc�A�!*

lossd��<ua�t       �	�fc�A�!*

loss_}�=�h       �	kIfc�A�!*

loss��P=lW�       �	��fc�A�!*

loss߳:@�&       �	#�fc�A�!*

loss�*�<}5�y       �	�}fc�A�!*

loss2�8<��ݦ       �	�fc�A�!*

loss�*�:)�U       �	̲fc�A�!*

loss��:%�_       �	�Ifc�A�!*

loss10=�KJ�       �	��fc�A�!*

loss���<�ͣ3       �	��fc�A�!*

lossV��;���Y       �	��fc�A�!*

loss��b<��       �	�*fc�A�!*

lossoJ\<+�M       �	��fc�A�!*

lossϫ-:w�Q3       �	ufc�A�!*

loss�z�<z��       �	� fc�A�!*

loss���<�p��       �	� fc�A�!*

loss�� ;o/
       �	�J!fc�A�!*

loss��Z=5��o       �	��!fc�A�!*

loss�>�<.=�.       �	�"fc�A�!*

loss�JB<D)�       �	$#fc�A�!*

lossd3�<�       �	�#fc�A�!*

loss���<_�߳       �	�X$fc�A�!*

lossn�T<�W�       �	��$fc�A�!*

lossqaW;�ɔ       �	P�%fc�A�!*

loss�>�<(��       �	�*&fc�A�!*

loss��<Q��^       �	��&fc�A�!*

loss1P<�Ս       �	F^'fc�A�!*

loss]:�<[�Ky       �	E�'fc�A�!*

loss���;�A7�       �	�(fc�A�!*

loss[�<�sG       �	)%)fc�A�!*

loss���;	���       �	b�)fc�A�!*

loss�K�9�[�       �	�`*fc�A�!*

loss�9=;o̩�       �	m +fc�A�!*

loss���;�H�       �	I�+fc�A�!*

loss�.+=Gb�       �	�4,fc�A�"*

loss�}3<�vY�       �	%�,fc�A�"*

lossZ��;R�m       �	�b-fc�A�"*

loss ;T��       �	�.fc�A�"*

loss/i;�}��       �	؛.fc�A�"*

loss��;�N��       �	!:/fc�A�"*

loss�S<�~�	       �	8�/fc�A�"*

lossW6c=�B�W       �	io0fc�A�"*

loss��?<�(�7       �	s1fc�A�"*

lossay�<U��       �	�1fc�A�"*

loss\&�;�|I       �	hX2fc�A�"*

loss	�<j�t�       �	�2fc�A�"*

loss6��9
O�       �	[�3fc�A�"*

loss�1=<�Z�       �	=4fc�A�"*

loss�%�;��Y�       �	R�4fc�A�"*

loss��<�2��       �	{h5fc�A�"*

loss9��;AV)       �	�6fc�A�"*

loss¿;Ki,       �	o�6fc�A�"*

loss�s<�H��       �	�77fc�A�"*

losse�<��       �	_�7fc�A�"*

lossZ�;C��J       �	/k8fc�A�"*

lossx:�       �	� 9fc�A�"*

lossu̔;|E��       �	��9fc�A�"*

loss;b�       �	(C:fc�A�"*

loss�]<>`�,       �	��:fc�A�"*

lossd�=��L       �	�m;fc�A�"*

lossI�<h ��       �	�<fc�A�"*

loss�7�<H#�       �	y�<fc�A�"*

loss]��<���       �	[C=fc�A�"*

lossc�;dj�       �	�=fc�A�"*

loss@6D;yx�       �	zp>fc�A�"*

loss4:=C
�%       �	d?fc�A�"*

loss��:��Q       �	��?fc�A�"*

loss���9��*F       �	��@fc�A�"*

loss6��:7Seg       �	7Afc�A�"*

loss�u?;pr       �	�Afc�A�"*

loss��;��z�       �	�MBfc�A�"*

loss�L�<�o2       �	f�Bfc�A�"*

loss��U;�P6�       �	�Cfc�A�"*

loss䩠<�1��       �	#/Dfc�A�"*

loss�;��f        �	��Dfc�A�"*

loss惡<A-q       �	�XEfc�A�"*

lossx <^��       �	Q�Efc�A�"*

loss��n:%��       �	��Ffc�A�"*

lossR�A:��T       �	>Gfc�A�"*

loss��F<9���       �	��Gfc�A�"*

loss,#?<7,�       �	�|Hfc�A�"*

lossL�<����       �	�Ifc�A�"*

loss�h <e�]       �	�Ifc�A�"*

loss���<h7l�       �	�RJfc�A�"*

lossx��<�c�       �	��Jfc�A�"*

lossN��:@[��       �	0�Kfc�A�"*

lossp��;ɮ�       �	�Lfc�A�"*

loss���9��3t       �	��Lfc�A�"*

loss��;c�       �	�FMfc�A�"*

losswʝ=���       �	��Mfc�A�"*

loss�]<&T�<       �	=�Nfc�A�"*

loss�f<�,��       �	�Ofc�A�"*

loss��T=�M7�       �	��Ofc�A�"*

loss���<��J       �	�SPfc�A�"*

losso�&;�Z+�       �	�Pfc�A�"*

loss��;4�x       �	|Qfc�A�"*

lossìa9���       �	Rfc�A�"*

lossj�=�k/�       �	�Rfc�A�"*

lossn��<����       �	�mSfc�A�"*

lossRA!>�Na�       �	�Tfc�A�"*

loss7E�;9SS       �	ߤTfc�A�"*

loss��<7|�       �	6<Ufc�A�"*

loss<J=�       �	��Ufc�A�"*

loss�x�<L�TY       �	�fVfc�A�"*

loss@}�:� �       �	��Vfc�A�"*

lossB��;��+       �	��Wfc�A�"*

loss�Mn=_$d       �	9%Xfc�A�"*

loss��:E�K       �	��Xfc�A�"*

lossEz�=��:       �	�aYfc�A�"*

losseL�<��^       �	-Zfc�A�"*

loss�(�;�� c       �	E�Zfc�A�"*

loss��=�:�       �	�1[fc�A�"*

lossm�;>J5�       �	@m\fc�A�"*

loss-�{:�qv�       �	�^]fc�A�"*

loss�"�;8�0       �	�^fc�A�"*

lossqyj<)s}z       �	If_fc�A�"*

lossN��<�Z��       �	O`fc�A�"*

loss2�*<2��-       �	x�`fc�A�"*

loss�O�<��ؒ       �	�0afc�A�"*

lossQ�<�X|       �	Q�afc�A�"*

loss�<�ʵ       �	{�bfc�A�"*

loss�{�<��J       �	Y3cfc�A�"*

loss�='p|�       �	
�cfc�A�"*

loss_�<�� &       �	��dfc�A�"*

loss�|�;�D��       �	�efc�A�"*

loss <P<l���       �	�Iffc�A�"*

loss��<[��       �	�ffc�A�"*

loss� =�F�1       �	��gfc�A�"*

loss�o;��!�       �	0hfc�A�"*

loss#Š:Ɯ��       �	��hfc�A�"*

lossT�<9�H       �	$aifc�A�"*

loss���<� �       �	�jfc�A�"*

loss�%D<�j�       �	��jfc�A�"*

lossA��=�G��       �	�5kfc�A�"*

loss,C�=S��       �	1�kfc�A�"*

loss��Q=���        �	�klfc�A�"*

loss��m;���       �	\mfc�A�"*

loss��<�7��       �	��mfc�A�"*

lossZ)�;A��y       �	/nfc�A�"*

lossO�<c�bg       �	��nfc�A�"*

loss�V;���q       �	W[ofc�A�"*

loss��R<�a       �	3�ofc�A�"*

loss���;EP�       �	��pfc�A�"*

loss���;���       �	�(qfc�A�"*

loss;�=�C��       �	�qfc�A�"*

loss�+
<M.{       �	�Orfc�A�"*

lossc �<]��k       �	q�rfc�A�"*

lossh�;�%�       �	�usfc�A�"*

loss��;�4t�       �	�tfc�A�"*

loss{��;l�;       �	��tfc�A�"*

loss��<�S�       �	$Hufc�A�"*

loss�?R=]�>�       �	��ufc�A�"*

lossSݒ<dSM       �	�xvfc�A�"*

loss�2�9e�5       �	c}wfc�A�"*

loss��v<s���       �	�4xfc�A�"*

loss�;)�e       �	��xfc�A�"*

lossMdo<���+       �	�lyfc�A�"*

loss��<��s�       �	�zfc�A�"*

loss��=��       �	Лzfc�A�"*

loss��=:uG�       �	�5{fc�A�"*

loss��Y;ϔW       �	��{fc�A�"*

lossս=�|l�       �	p|fc�A�"*

loss�	�;`��       �	�"}fc�A�#*

loss�;=��       �	��}fc�A�#*

loss-�4<X�S�       �	��~fc�A�#*

loss��;����       �	�fc�A�#*

lossk=�s�=       �	��fc�A�#*

lossE�<��"�       �	Q�fc�A�#*

loss��x<�y�9       �	��fc�A�#*

loss��<␰o       �	Z��fc�A�#*

loss���:��d       �	T�fc�A�#*

loss�<l�q-       �	��fc�A�#*

lossM�L;�_�       �	?��fc�A�#*

loss���:��У       �	U/�fc�A�#*

loss�{b<OX�f       �	�˄fc�A�#*

loss�٧:��y       �	si�fc�A�#*

loss��;�ӣ�       �	��fc�A�#*

loss3�<7m�       �	���fc�A�#*

loss�4=t���       �	�-�fc�A�#*

loss,� =�Π�       �	�ʇfc�A�#*

loss4�<�U�       �	^�fc�A�#*

loss���<s{w�       �	��fc�A�#*

loss}�>;��K)       �	V��fc�A�#*

loss%P<�D       �	.�fc�A�#*

lossf~�;�s�       �	���fc�A�#*

loss��9�U       �	�=�fc�A�#*

loss�)=	�5       �	 �fc�A�#*

loss��-;q��q       �	�r�fc�A�#*

loss���<;�T�       �	C�fc�A�#*

loss�w�9�8��       �	�fc�A�#*

lossZ	�<8X       �	^0�fc�A�#*

loss���;*!       �	�Ďfc�A�#*

loss��!<�[Vj       �	jg�fc�A�#*

loss�`�;���       �	�fc�A�#*

loss[�<���R       �	���fc�A�#*

loss��1;��[       �	�;�fc�A�#*

lossR�&<�4ړ       �	ߑfc�A�#*

lossg�<E��       �	s�fc�A�#*

loss�YF<�M�       �	��fc�A�#*

lossee�:s��[       �	(��fc�A�#*

loss=w{<Q�h�       �	?S�fc�A�#*

lossb�<��?       �	(�fc�A�#*

loss}�!=�2+       �	���fc�A�#*

loss�@;�B�T       �	y#�fc�A�#*

lossoWm=+��       �	8��fc�A�#*

loss���<9�E�       �	�T�fc�A�#*

lossR%�:�̐�       �	��fc�A�#*

loss؅Y;wQ+�       �	=��fc�A�#*

loss��"<�S�O       �	�/�fc�A�#*

loss�[a<��;�       �	2əfc�A�#*

loss�-<6��y       �	]�fc�A�#*

loss3=��       �	V�fc�A�#*

loss���<@�       �	���fc�A�#*

loss��;r�6�       �	a�fc�A�#*

loss��";0 L       �	S��fc�A�#*

loss��;s�w_       �	jK�fc�A�#*

lossq:�:��6�       �	7ߝfc�A�#*

loss�FO=g�8       �	�q�fc�A�#*

lossAͦ:�Gn�       �	/�fc�A�#*

loss�V�=Y��c       �	l�fc�A�#*

lossj};ia_       �	V��fc�A�#*

loss���<�>��       �	b1�fc�A�#*

loss�\�9�x�4       �	��fc�A�#*

loss��;��5[       �	��fc�A�#*

loss:�:�~g�       �	a8�fc�A�#*

loss3b�<�=�       �	�ˣfc�A�#*

lossz��;ؗĲ       �	�k�fc�A�#*

loss���=��m�       �	4�fc�A�#*

loss{�J<AF'�       �	�ѥfc�A�#*

lossm�:	)�       �	=e�fc�A�#*

loss�i=o�F�       �	e��fc�A�#*

lossñ�<��       �	�4�fc�A�#*

loss��=Z'D�       �	Ѩfc�A�#*

loss��<��TQ       �	p�fc�A�#*

loss|/.=�}�u       �	�fc�A�#*

loss.<r:�~       �	���fc�A�#*

lossA�=E�(�       �	�H�fc�A�#*

loss ��=2��       �	-�fc�A�#*

loss�T;i�}2       �	D��fc�A�#*

loss4L_;	Ζ�       �	�.�fc�A�#*

loss�9�<��(       �	�ͭfc�A�#*

loss_c�9�*]       �	]l�fc�A�#*

loss}=sb��       �	��fc�A�#*

loss�i\=��}       �	���fc�A�#*

loss�|�;F��       �	,I�fc�A�#*

loss���<�UJ       �	��fc�A�#*

loss���=�       �	0��fc�A�#*

loss��C=��˦       �	��fc�A�#*

loss�y<R^�[       �	#��fc�A�#*

lossd��=S׷�       �	TS�fc�A�#*

loss�m=`�w       �	���fc�A�#*

loss���;DM_       �	h��fc�A�#*

loss���;�,�       �	�2�fc�A�#*

loss`zr;�&6=       �	jٵfc�A�#*

lossD�:����       �	�q�fc�A�#*

loss��;,�2�       �	�	�fc�A�#*

loss�ˌ<��!�       �	oF�fc�A�#*

lossԇ<�G�       �	�E�fc�A�#*

loss���<��b�       �	� �fc�A�#*

losse�O=Ta�Z       �	m�fc�A�#*

lossi��<0��       �	�fc�A�#*

loss��u<l�;(       �	���fc�A�#*

lossϯ�:˰��       �	�-�fc�A�#*

loss_!�;6h�       �	Tǽfc�A�#*

loss�q
<��Z       �	�a�fc�A�#*

loss��=Չ�7       �	@��fc�A�#*

loss���<%�q'       �	h��fc�A�#*

loss�؍<�&v�       �	�7�fc�A�#*

loss_<�rѫ       �	Q��fc�A�#*

loss���;Ӎ�^       �	B|�fc�A�#*

lossO3 =��*�       �	��fc�A�#*

loss� ;`s_       �	1��fc�A�#*

loss&a*;T�z       �	qV�fc�A�#*

loss# ;��T�       �	��fc�A�#*

losseJT<P!��       �	ė�fc�A�#*

loss�;\(�       �	\<�fc�A�#*

loss�\k<abA=       �	V��fc�A�#*

loss��=¼��       �	�m�fc�A�#*

loss�a<O)       �	�fc�A�#*

loss�\A9 4��       �	J��fc�A�#*

losss<��¥       �	�2�fc�A�#*

loss�
�:{t�       �	c��fc�A�#*

loss�H�<�́r       �	\t�fc�A�#*

lossERS;a��       �	�fc�A�#*

lossIr;�~4l       �	��fc�A�#*

loss/T�<C"ex       �	B�fc�A�#*

loss�Q�;4�o,       �	s��fc�A�#*

lossD~^=j��'       �	!x�fc�A�#*

loss��H;���       �	��fc�A�#*

loss��;�I�       �	���fc�A�#*

loss�9 <�/9       �	�A�fc�A�$*

loss�X7=���       �	���fc�A�$*

loss�p�=Z#�c       �	��fc�A�$*

loss�t�=��4g       �	�"�fc�A�$*

lossR��<��a       �	Z��fc�A�$*

loss.;�;�l       �	���fc�A�$*

loss�"?< �gU       �	�F�fc�A�$*

lossq�Y<9�+�       �	���fc�A�$*

loss���;��I       �	g��fc�A�$*

loss��9���       �	7�fc�A�$*

lossgM�<��Wd       �	�W�fc�A�$*

losse��;���_       �	���fc�A�$*

loss߉={ڝE       �	��fc�A�$*

loss8�_:̫`�       �	f��fc�A�$*

loss��r<U��       �	�&�fc�A�$*

loss�ey:�-3R       �	���fc�A�$*

losszҔ=��       �	P�fc�A�$*

loss���;)C�       �	�+�fc�A�$*

loss�U;��;       �	e��fc�A�$*

loss�4�<�.       �	W�fc�A�$*

loss2�i<��̓       �	/��fc�A�$*

loss�P;q�       �	���fc�A�$*

loss!�K;�� �       �	vT�fc�A�$*

lossM2�<`�|�       �	�{�fc�A�$*

loss��d<u]R�       �	_B�fc�A�$*

loss��:;��D       �	���fc�A�$*

loss�dc;�bl�       �	�u�fc�A�$*

loss�P; �a       �	�fc�A�$*

lossA`:�x�       �	���fc�A�$*

loss���;�ߠ�       �	`r�fc�A�$*

loss_�;�()       �	��fc�A�$*

loss�P:�zo`       �	h��fc�A�$*

lossE��:�Q       �	Tr�fc�A�$*

lossCn�</��*       �	��fc�A�$*

losso�	=43�       �	ޮ�fc�A�$*

loss;�p<�5��       �	�F�fc�A�$*

loss�y=%��       �	���fc�A�$*

loss��;a�yc       �	��fc�A�$*

loss���:���       �	z��fc�A�$*

loss/<i0��       �	@/�fc�A�$*

lossڈ=;%T�2       �	$��fc�A�$*

loss��</�2       �	6t�fc�A�$*

loss���;���       �	��fc�A�$*

loss�Ա<�he�       �	и�fc�A�$*

loss	?�:�Hj�       �	�]�fc�A�$*

lossv2�:=m       �	U��fc�A�$*

lossa�;OqW~       �	���fc�A�$*

lossT2�<osC       �	�:�fc�A�$*

lossc2<�P�       �	���fc�A�$*

loss�<��QU       �		q�fc�A�$*

lossc�<�9��       �	��fc�A�$*

loss�{";7���       �	��fc�A�$*

loss��;U�D       �	k~�fc�A�$*

loss�S�;nB\�       �	��fc�A�$*

loss��;��nh       �	���fc�A�$*

lossH֮:��$r       �	�\�fc�A�$*

loss\�<=@C�       �	���fc�A�$*

loss �<�'y�       �	,��fc�A�$*

loss��9����       �	�3�fc�A�$*

loss�M=��[_       �	��fc�A�$*

loss��:����       �	�r�fc�A�$*

loss� �;4�       �	��fc�A�$*

loss�i�;�ЭT       �	C��fc�A�$*

losscp=����       �	��fc�A�$*

loss?��9�R�z       �	�N�fc�A�$*

loss�T�;�v�       �	_��fc�A�$*

loss�QL=HC�l       �	s��fc�A�$*

lossQY;o���       �	(�fc�A�$*

lossM#<���/       �	��fc�A�$*

loss�M�;9\�)       �	2 �fc�A�$*

loss�B`=��       �	���fc�A�$*

loss�U<�bC       �	b��fc�A�$*

loss_��=n֛�       �	["�fc�A�$*

loss��>T�X;       �	���fc�A�$*

loss�� =�{2�       �	Z fc�A�$*

loss�k:��L$       �	M� fc�A�$*

loss�[�;u�z�       �	��fc�A�$*

loss ��<ڡ0�       �	�kfc�A�$*

loss��_;Z�p~       �	Bfc�A�$*

loss�V)<��7       �	��fc�A�$*

loss|QV=Q�<       �	�pfc�A�$*

losso�?=.55�       �	fc�A�$*

loss�p�:wCK�       �	8�fc�A�$*

loss�R�<�ā       �	�7fc�A�$*

loss4De<j9��       �	1�fc�A�$*

loss��;�k�       �	�bfc�A�$*

loss�fH=��\       �	�fc�A�$*

lossb;��d2       �	��fc�A�$*

loss��`=1ը       �	#.	fc�A�$*

loss�sn<#�YO       �	u�	fc�A�$*

loss���<@�u�       �	�f
fc�A�$*

loss�(<t}       �	�fc�A�$*

loss�c�;Ou�       �	3�fc�A�$*

loss��<��K�       �	Mfc�A�$*

loss�+);�a�       �	��fc�A�$*

lossH*S<�|��       �	��fc�A�$*

loss���<T��       �	�!fc�A�$*

loss��T<nrA       �	�fc�A�$*

loss<��:�ߘ       �	�\fc�A�$*

loss�	�< B�       �	��fc�A�$*

loss�E6;*=��       �	��fc�A�$*

loss�+�<�!�N       �	0fc�A�$*

loss6��;��       �	�&fc�A�$*

loss�^l=��~!       �	��fc�A�$*

loss�q=WOJ        �	}zfc�A�$*

loss�h<��       �	wfc�A�$*

loss?��<\��       �	��fc�A�$*

loss94;�-��       �	�Hfc�A�$*

loss�d�;F�       �	��fc�A�$*

loss�\=�-�!       �	wfc�A�$*

loss&�=;��!       �	fc�A�$*

loss
�:�=%       �	��fc�A�$*

loss�r <�>Q�       �	�>fc�A�$*

loss��E;��&       �	��fc�A�$*

loss{��;,ע�       �	׆fc�A�$*

losst�8:��4�       �	��fc�A�$*

loss	�;�\�       �	��fc�A�$*

loss�&-=��1�       �	`wfc�A�$*

loss�>=Qbb       �	�fc�A�$*

lossN9�<���c       �	MIfc�A�$*

loss���=a��       �	��fc�A�$*

loss</�=ۯ�u       �	n�fc�A�$*

loss�;1#V       �	
. fc�A�$*

lossr��:���l       �	N� fc�A�$*

lossNM=��O�       �	lw!fc�A�$*

lossz@x<"Z"h       �	�0"fc�A�$*

loss#��8 g�K       �	��"fc�A�$*

loss�/X;~nE       �	~t#fc�A�$*

loss$k�:ͺ]       �	I$fc�A�%*

loss�L�;w"�       �	�%fc�A�%*

loss�n(<(��]       �	��%fc�A�%*

loss�5�9��s       �	pC&fc�A�%*

loss��< @h       �	�'fc�A�%*

loss��9��K�       �	D�'fc�A�%*

lossQ�8��Z       �	�>(fc�A�%*

lossT�}7� ��       �	4�(fc�A�%*

lossL+)=ǩv       �	�})fc�A�%*

loss$��<�<�B       �	]*fc�A�%*

loss��=�]�W       �	N�*fc�A�%*

lossn�9�4Ӿ       �	oJ+fc�A�%*

loss�W�:�7       �	�+fc�A�%*

loss`|e<��,       �	_y,fc�A�%*

loss8[=:~I8e       �	0d-fc�A�%*

loss�d�= X�`       �	.fc�A�%*

loss,ݏ<�wSp       �	�.fc�A�%*

loss#?�=\��       �	37/fc�A�%*

loss��;j�Q�       �	:�/fc�A�%*

loss�w<L�M*       �	��0fc�A�%*

loss͘�<	f3       �	��1fc�A�%*

loss�w�<*��       �	�.2fc�A�%*

loss���;���o       �	F�2fc�A�%*

loss��<�ls�       �	er3fc�A�%*

loss2&;�`�       �	k-4fc�A�%*

lossc�r;|���       �	`�4fc�A�%*

loss�ߒ<��AS       �	�u5fc�A�%*

loss��:�uM�       �	�6fc�A�%*

lossA�*<O&�       �	E�6fc�A�%*

lossןA=+gD�       �	�7fc�A�%*

loss,�~<��e�       �	��8fc�A�%*

loss�s�;F��       �	,E9fc�A�%*

loss/�^<+�N       �	��:fc�A�%*

loss$��9��t!       �	"p;fc�A�%*

loss �;��O       �	\<fc�A�%*

loss�F?<vS-�       �	D=fc�A�%*

lossV��<��       �	�f>fc�A�%*

loss4D<5!��       �	f�>fc�A�%*

losszC<��!       �	ɰ?fc�A�%*

loss�#�9��~�       �	��@fc�A�%*

losspT�<��g       �	X�Afc�A�%*

loss	7<1���       �	.YBfc�A�%*

loss�u;�-̀       �	i7Dfc�A�%*

loss�8=�o$�       �	�Efc�A�%*

loss*O�:����       �	�Efc�A�%*

loss9A=�oG�       �	p
Gfc�A�%*

loss��+=��       �	�BHfc�A�%*

loss�\<�=��       �	��Hfc�A�%*

lossV�F:��       �	0�Ifc�A�%*

lossn0,;��Х       �	�KJfc�A�%*

loss���:��I       �	Kfc�A�%*

losswe:;	�       �	��Kfc�A�%*

loss��=�!D       �	��Lfc�A�%*

loss��M<6�TR       �	�+Mfc�A�%*

loss�p�;��       �	i�Mfc�A�%*

loss�:"<ѭ�       �	scNfc�A�%*

lossq+<���       �	�Ofc�A�%*

losse�<$S��       �	�Ofc�A�%*

loss|�U;��=s       �	.VPfc�A�%*

loss�r"<D2�g       �	��Pfc�A�%*

loss��;+e�Z       �	��Qfc�A�%*

lossQ��;����       �	� Rfc�A�%*

lossI'�<�y�0       �	��Rfc�A�%*

loss7�
<t]?�       �	�TSfc�A�%*

loss���<�Ώ�       �	$�Sfc�A�%*

loss�r<V���       �	S�lfc�A�%*

loss�*�<C�3       �	�ymfc�A�%*

loss�9b<D�F       �	�"nfc�A�%*

loss���<���       �	{�nfc�A�%*

loss��<��ɍ       �	beofc�A�%*

lossH�=�I@{       �	Cpfc�A�%*

lossmX(=7��=       �	�pfc�A�%*

lossR'�;Y3�*       �	?7qfc�A�%*

loss@�$;"�u       �	��qfc�A�%*

losscQU=%��
       �	c~rfc�A�%*

lossC�:qy�?       �	?sfc�A�%*

lossn{=� �       �	ӿsfc�A�%*

loss��<��x~       �	gtfc�A�%*

lossMo =u�Ǣ       �	�ufc�A�%*

losse;ы�]       �	|�ufc�A�%*

loss��;�z$�       �	�avfc�A�%*

loss��9DC��       �	`wfc�A�%*

loss��<��       �	E�wfc�A�%*

loss��<�F�q       �	�zxfc�A�%*

lossD>�;s0�       �	#yfc�A�%*

loss]�;Jް�       �	��zfc�A�%*

loss���=�#       �	m�{fc�A�%*

loss�cS<�uE       �	U|fc�A�%*

loss��_=���       �	)~fc�A�%*

loss��<���       �	s�~fc�A�%*

loss.d�:�~�a       �	�tfc�A�%*

loss�d�<G]^       �	s�fc�A�%*

loss��:�:�m       �	��fc�A�%*

loss��F=�?�       �	�G�fc�A�%*

loss��{=�(W2       �	��fc�A�%*

loss��<��       �	��fc�A�%*

lossq"�924�       �	
J�fc�A�%*

loss���;�p/       �	��fc�A�%*

loss	��<x��9       �	,��fc�A�%*

loss3Es<.B>�       �	��fc�A�%*

lossÜ�<c_qB       �	مfc�A�%*

loss}<��{       �	�y�fc�A�%*

loss�M==�       �	��fc�A�%*

loss�j�<�x��       �	���fc�A�%*

loss�a�=`��       �	�P�fc�A�%*

lossd�]=� =�       �	��fc�A�%*

lossd�$<�h�       �	�d�fc�A�%*

loss���:��       �	�fc�A�%*

loss�Z:�       �	���fc�A�%*

loss��<;�       �	�O�fc�A�%*

losso�;P��3       �	��fc�A�%*

loss�0�;�{G�       �	��fc�A�%*

loss�7=<��       �	p��fc�A�%*

loss���<��;       �	<3�fc�A�%*

lossӅ�:mU�       �	(яfc�A�%*

loss7��8�"       �	m�fc�A�%*

loss�s=36��       �	 �fc�A�%*

lossB;j���       �	T��fc�A�%*

loss%�k=}��5       �	�R�fc�A�%*

lossrz;y�T�       �	�fc�A�%*

loss��':��B       �	��fc�A�%*

loss��9�i�0       �	�*�fc�A�%*

loss#߯9� �a       �	�Ȕfc�A�%*

lossL5�;&��       �	Di�fc�A�%*

loss�%�=�
T2       �	��fc�A�%*

loss��=
��       �	���fc�A�%*

losss��;��o       �	�ޗfc�A�%*

loss*��;�o	       �	�y�fc�A�%*

loss;��<w5��       �	u�fc�A�&*

loss !=A�}�       �	��fc�A�&*

loss�%<;�''       �	���fc�A�&*

loss��<W��s       �	>�fc�A�&*

lossǥ<� '       �	�ݛfc�A�&*

lossi)�<��       �	<��fc�A�&*

loss�	?=�=q*       �	I0�fc�A�&*

loss���;͑�p       �	 ǝfc�A�&*

lossw
�<��=�       �	B]�fc�A�&*

loss���=�i�       �	���fc�A�&*

loss�{<K*�#       �	GU�fc�A�&*

lossL;j���       �	��fc�A�&*

losscL�<��4H       �	Ӈ�fc�A�&*

loss��p<���       �	v6�fc�A�&*

loss�ܻ<�d��       �	=բfc�A�&*

loss1��;E�v�       �	�|�fc�A�&*

loss��=�ȯ.       �	�fc�A�&*

loss��;!aD       �	XǤfc�A�&*

loss�*O<d       �	{ڥfc�A�&*

loss4=>Q��       �	M��fc�A�&*

loss��V;�>�>       �	/�fc�A�&*

loss�e�<��       �	oקfc�A�&*

loss��-<Ш�       �	���fc�A�&*

loss��^<o��       �	�(�fc�A�&*

loss�V�:�K4�       �	Jҩfc�A�&*

loss:�K;<�P�       �	�y�fc�A�&*

loss���;PL7)       �	�(�fc�A�&*

loss[5�<zI       �	�E�fc�A�&*

loss\#4;�6l�       �	6�fc�A�&*

lossӻ�;i��T       �	��fc�A�&*

loss�H;ʍ�_       �	�7�fc�A�&*

loss��<�@�U       �	��fc�A�&*

loss��;��l       �	���fc�A�&*

lossܙ<<�)�       �	X8�fc�A�&*

loss`;�<Q�S�       �	Dݰfc�A�&*

loss!f�=�M�U       �	7��fc�A�&*

loss�Y<a](�       �	�9�fc�A�&*

loss ��;�n�       �	�޲fc�A�&*

loss��;�M��       �	��fc�A�&*

loss�X<���*       �	N%�fc�A�&*

loss?��;@sC       �	�ֵfc�A�&*

loss��S;R���       �	�{�fc�A�&*

loss�H:�\��       �	g)�fc�A�&*

loss$��:��&�       �	>ʷfc�A�&*

loss#0O;�+�k       �	N�fc�A�&*

loss$oy<3u]       �	��fc�A�&*

loss��!<nͯ       �	��fc�A�&*

loss�r<��g�       �	���fc�A�&*

loss�$<�Ŵ�       �	ke�fc�A�&*

loss@�D;��K*       �	�G�fc�A�&*

loss,��:�t�       �	m�fc�A�&*

loss�x<6��(       �	�/�fc�A�&*

loss�G�;���V       �	�޿fc�A�&*

loss�a<:�       �	O��fc�A�&*

loss���<ϒ�       �	>�fc�A�&*

loss?��;�"�       �	���fc�A�&*

loss���9��6�       �	��fc�A�&*

lossI�<��$+       �	wH�fc�A�&*

loss��@<;��       �	���fc�A�&*

loss�<<(       �	'��fc�A�&*

loss���<"�       �	/M�fc�A�&*

loss
Ո;W��       �	��fc�A�&*

loss��8=�W�n       �	���fc�A�&*

loss	�:3W'�       �	�d�fc�A�&*

loss���<w�A5       �	��fc�A�&*

loss�U�<��       �	���fc�A�&*

loss�<v��       �	�r�fc�A�&*

lossf	=�y]�       �	*p�fc�A�&*

loss ��:F	��       �	$�fc�A�&*

lossQ�<�c{=       �	��fc�A�&*

loss�*w<ؙ�       �	���fc�A�&*

lossƚ�<>e��       �	�6�fc�A�&*

loss8ܐ;e#U       �	:��fc�A�&*

loss��e<�O       �	\��fc�A�&*

loss�EX;���       �	#��fc�A�&*

lossI��:F��       �	PR�fc�A�&*

lossH�<8_�       �	V��fc�A�&*

loss�O�9Zs�       �	���fc�A�&*

loss��;A��6       �	N�fc�A�&*

loss*9<?���       �	���fc�A�&*

loss,a�;�w�       �	���fc�A�&*

lossE�<��v       �	�^�fc�A�&*

lossx;�;Y��        �	��fc�A�&*

loss���:��m       �	0��fc�A�&*

lossҨr:i��<       �	A��fc�A�&*

loss��Q=����       �	�(�fc�A�&*

lossft�9)��       �	���fc�A�&*

loss��:��       �	��fc�A�&*

loss���<�B�\       �	S��fc�A�&*

loss�=:�:       �	�m�fc�A�&*

lossW%�<�_       �	M�fc�A�&*

loss�6<繇       �	��fc�A�&*

loss��J9���       �	�k�fc�A�&*

lossڿI:ޯ�       �	<�fc�A�&*

lossk<-�ׂ       �	���fc�A�&*

lossۃ�<��r:       �	�s�fc�A�&*

loss��=�Ka�       �	u�fc�A�&*

loss�^�<u��       �	e��fc�A�&*

loss���;����       �	�m�fc�A�&*

loss���=T�6�       �	v�fc�A�&*

losst� <E�>�       �	`��fc�A�&*

lossObN<#���       �	r�fc�A�&*

loss<a�A       �	%"�fc�A�&*

loss�E1:�6��       �	#��fc�A�&*

loss�Ό:���+       �	M��fc�A�&*

loss��=;Oa�I       �	A*�fc�A�&*

lossH�;!U�       �	q��fc�A�&*

loss��;���       �	�y�fc�A�&*

loss�p9�%>�       �	��fc�A�&*

lossj�J<�]
       �	���fc�A�&*

loss �-<^o:�       �	v�fc�A�&*

loss���9}�)       �	��fc�A�&*

loss
�?=���t       �	�b�fc�A�&*

lossW��;��^       �	a�fc�A�&*

loss��s;<�k       �	���fc�A�&*

lossv��;�1       �	z��fc�A�&*

loss@e�;�l�       �	���fc�A�&*

loss�3;��NA       �	�V�fc�A�&*

loss8��;�1�       �	�fc�A�&*

loss{j%:����       �	կ�fc�A�&*

loss�);ӊ7�       �	%Z�fc�A�&*

loss�"�9���       �	%�fc�A�&*

loss�7<��+^       �	���fc�A�&*

losszh<�8#       �		Q�fc�A�&*

loss��<��Z       �	��fc�A�&*

loss���;��a�       �	���fc�A�&*

loss��{:�0��       �	oG�fc�A�&*

loss@�J;�nd�       �	h��fc�A�&*

loss. 0=[.��       �	:��fc�A�'*

loss �?;�c�p       �	AE�fc�A�'*

lossÃ?:!��I       �	>�fc�A�'*

lossm�j=�[�       �	��fc�A�'*

loss�xX;+n��       �	���fc�A�'*

loss((;��.c       �	���fc�A�'*

loss��<h�X�       �	&r�fc�A�'*

loss,.�9��       �	��fc�A�'*

loss&�H;�|�#       �	j��fc�A�'*

loss��d=kg�       �	kd�fc�A�'*

lossc��8k���       �	��fc�A�'*

lossg�<��       �	[��fc�A�'*

lossE
=�M��       �	�S fc�A�'*

loss�(a<%       �	9� fc�A�'*

loss��<���       �	�fc�A�'*

loss �<�&�k       �	P4fc�A�'*

losst5�<�i@�       �	=�fc�A�'*

loss�/y=>3�       �	�lfc�A�'*

loss,��;��w       �	pfc�A�'*

lossrF+;X�A(       �	�fc�A�'*

loss-��:����       �	8Nfc�A�'*

loss�x;oZ�%       �	��fc�A�'*

loss嫮:_��       �	��fc�A�'*

loss8�<��Jz       �	Jfc�A�'*

lossi&_<?]       �	.�fc�A�'*

loss���<��U       �	D�fc�A�'*

loss�-#=o��       �	F&	fc�A�'*

loss�l�;�EV       �	��	fc�A�'*

lossJQ3<��l       �	�f
fc�A�'*

loss���9��	       �	�fc�A�'*

loss7nU;!r0       �	�fc�A�'*

loss�� ;CM&�       �	�Ifc�A�'*

loss���<���       �	��fc�A�'*

loss�=h�       �	s�fc�A�'*

lossJ�H<���       �	2:fc�A�'*

lossTA:;�}��       �	��fc�A�'*

loss�X�<7�b|       �	Úfc�A�'*

lossM;��#P       �	�>fc�A�'*

loss��B<��       �	1�fc�A�'*

loss��:xfEZ       �	[�fc�A�'*

lossh�;a�       �	�Bfc�A�'*

loss��<<�;�       �	%�fc�A�'*

loss�_<��6�       �	9�fc�A�'*

lossW��:k#��       �	�?fc�A�'*

loss�@�:R#	       �	��fc�A�'*

lossZ?2<Q�HJ       �	Ύfc�A�'*

loss�f;b�K�       �	7fc�A�'*

losso�?:|���       �	��fc�A�'*

loss79�Y�        �	�zfc�A�'*

loss!��;�<        �	�bfc�A�'*

lossi��;"�֌       �	�fc�A�'*

lossH�8;�f�V       �	�fc�A�'*

loss	Dl=	�g       �	�Qfc�A�'*

loss{t�;���       �	��fc�A�'*

loss���<j�5�       �	�fc�A�'*

loss�p:��h       �	\:fc�A�'*

loss��<�X�       �	ffc�A�'*

loss��<z0�2       �	:�fc�A�'*

lossM�o:T�^�       �	Ofc�A�'*

loss��N<�|J�       �	��fc�A�'*

lossD�;!5��       �	��fc�A�'*

lossl�D<�m��       �	�/ fc�A�'*

loss�d�:�͢�       �	N� fc�A�'*

loss��-8��.       �	�k!fc�A�'*

loss��9����       �	 "fc�A�'*

loss�.<*���       �	9�"fc�A�'*

loss�70<��P�       �	 9#fc�A�'*

lossq��<�4"�       �	��#fc�A�'*

loss3�;R���       �	"o$fc�A�'*

loss�u3<m�T       �	M%fc�A�'*

loss#�<M�       �	�%fc�A�'*

loss{x:�)k       �	�K&fc�A�'*

lossY90��       �	C�&fc�A�'*

lossH�d=<��x       �	N%(fc�A�'*

lossV�:
�S�       �	]�(fc�A�'*

lossJ=v��       �	gd)fc�A�'*

lossf=�ҍ^       �	�*fc�A�'*

loss���=�N)�       �	K�*fc�A�'*

losszA#=-��;       �	�+fc�A�'*

loss;�i<O�       �	G>,fc�A�'*

lossC*<8ˤ�       �	��,fc�A�'*

loss��:�&�       �	ap-fc�A�'*

loss�=Q;χ`�       �	�	.fc�A�'*

loss���<��I       �	Ǡ.fc�A�'*

loss��<ٓj�       �	�6/fc�A�'*

loss&��;f��|       �	�/fc�A�'*

loss-�<E���       �	�f0fc�A�'*

loss6�;h;�       �	��0fc�A�'*

loss�Uu<�;�E       �	K�1fc�A�'*

loss�:U�ɱ       �	9(2fc�A�'*

loss4p�<}X�       �	��2fc�A�'*

losslۢ:ջ>I       �	�N3fc�A�'*

loss���=���       �	e�3fc�A�'*

loss��=Ȅ�6       �	�s4fc�A�'*

loss�<3��       �	�
5fc�A�'*

losss��<��d�       �	s�5fc�A�'*

loss_�:O��       �	WC6fc�A�'*

lossM�:Q��       �	7�6fc�A�'*

lossN<(�J�       �	��7fc�A�'*

loss�0;�H-�       �	v8fc�A�'*

loss�k�:��       �	�)9fc�A�'*

lossX�;4�       �	L�9fc�A�'*

loss��F9X�p�       �	*8;fc�A�'*

lossx%<3!�|       �	'�;fc�A�'*

lossl#�<�^+       �	8�<fc�A�'*

lossp�;3g�z       �	Ve=fc�A�'*

lossvK<�=�       �	��>fc�A�'*

loss3(3=�'.[       �	�q?fc�A�'*

loss �;���       �	�"@fc�A�'*

loss���;��:�       �	��@fc�A�'*

loss��9K&��       �	4KAfc�A�'*

loss��:N�M       �	L3Bfc�A�'*

loss�H=����       �	
�Bfc�A�'*

loss=�<f
1�       �	moCfc�A�'*

loss�0�=��0�       �	�Dfc�A�'*

loss�Ԗ<G2|~       �	B�Dfc�A�'*

loss�e="+        �	�)Efc�A�'*

loss�;��u`       �	o�Efc�A�'*

loss���=�b��       �	y]Ffc�A�'*

loss��;�Z�|       �	��Ffc�A�'*

loss���:���       �	��Gfc�A�'*

lossX�<��-b       �	78Hfc�A�'*

loss;1W:jN͋       �	p�Hfc�A�'*

lossZ�]<� �       �	�gIfc�A�'*

loss�İ=Wc(b       �	�Jfc�A�'*

loss�g�;��~.       �	�Jfc�A�'*

loss�;�<o��&       �	��Kfc�A�'*

losszZN=.]d       �	�Lfc�A�'*

loss�b�;��J       �	�6Mfc�A�(*

loss��;�
~u       �	F�Mfc�A�(*

loss��_;�*(�       �	hNfc�A�(*

loss&w�:t�<�       �	 Ofc�A�(*

loss���<L��       �	�Ofc�A�(*

loss`��;�)�       �	|*Pfc�A�(*

loss_�=���       �	��Pfc�A�(*

loss�0=h_GY       �	 qQfc�A�(*

loss�><5�4       �	�Rfc�A�(*

loss��<�+HV       �	��Rfc�A�(*

loss� �<��Ap       �	M2Sfc�A�(*

loss�a=͉u�       �	��Sfc�A�(*

loss�i�;|���       �	�pTfc�A�(*

lossC"�:	_�       �	kUfc�A�(*

lossZ�z;�4��       �	/�Ufc�A�(*

lossڽ�<x�}       �	�IVfc�A�(*

lossX�=�X�       �	�Vfc�A�(*

lossb!<��b>       �	�Wfc�A�(*

loss�Ś<i���       �	�Xfc�A�(*

loss�I�<c��%       �	(�Xfc�A�(*

loss��:�m�S       �	<PYfc�A�(*

loss�)*=����       �	��Yfc�A�(*

loss�\�9j+y�       �	�&[fc�A�(*

loss���9R�       �	��[fc�A�(*

loss��:*�       �	Z\fc�A�(*

loss=9K<:���       �	^�\fc�A�(*

loss�;��       �	�]fc�A�(*

loss���:5�E|       �	o/^fc�A�(*

lossH�<���       �	�^fc�A�(*

lossxTv;��4h       �	_{_fc�A�(*

loss_,<F���       �	�`fc�A�(*

losst��;�Yy       �	�`fc�A�(*

loss��;���       �	jKafc�A�(*

loss��$=ze       �	��afc�A�(*

lossms<���]       �	�qbfc�A�(*

loss}g/=6�Y       �	cfc�A�(*

loss!XV;|Z��       �	z�cfc�A�(*

loss��6:�u       �	!@dfc�A�(*

loss��;�A7"       �	��dfc�A�(*

loss.Hx;��       �	�jefc�A�(*

loss��<�y�)       �	��efc�A�(*

loss[;=h���       �	��ffc�A�(*

loss�(�:�>�       �	u9gfc�A�(*

loss(�;�	�       �	\�gfc�A�(*

loss ��=4c&�       �	��ifc�A�(*

loss��<���       �	�Qjfc�A�(*

lossF�=O���       �	�jfc�A�(*

loss6J�=�G�       �	ۆkfc�A�(*

losso�w;s�H�       �	Clfc�A�(*

loss���;bF�       �	�lfc�A�(*

loss��u<:v��       �	�^mfc�A�(*

loss%O�<�b٘       �	��mfc�A�(*

lossE�7<E��?       �	��nfc�A�(*

loss���9�I�       �	HRofc�A�(*

loss#�;���       �	I�ofc�A�(*

loss��o;�l�!       �	��pfc�A�(*

lossh�;?�O       �	�*qfc�A�(*

loss���; l�4       �		�qfc�A�(*

loss�Pc=F���       �	�_rfc�A�(*

loss�B<A�f�       �	��rfc�A�(*

lossv�;cԼn       �	ޏsfc�A�(*

loss��#=�b��       �	{1tfc�A�(*

lossZ�Q=�c�,       �	��tfc�A�(*

loss�}=7��       �	�\ufc�A�(*

loss��;tV�       �	 vfc�A�(*

lossrq/;��:�       �	|�vfc�A�(*

loss���=����       �	P:wfc�A�(*

loss���;�ūu       �	��wfc�A�(*

loss�j�<+|�       �	ofxfc�A�(*

loss��
:n�B�       �	��xfc�A�(*

loss� ;���u       �	��yfc�A�(*

loss�E<۵�       �	�-zfc�A�(*

loss��;] ��       �	
i{fc�A�(*

loss��K;�>/-       �	�g|fc�A�(*

loss&�;<��       �	;}fc�A�(*

loss%=�;����       �	 �}fc�A�(*

loss��:=���       �	��~fc�A�(*

loss)��:��       �	t�fc�A�(*

loss���:�ud�       �	Y4�fc�A�(*

loss6�9��~+       �	Wрfc�A�(*

lossӏ,=��A       �	us�fc�A�(*

loss�}=��B�       �	�fc�A�(*

loss�f�9@;��       �	�Ղfc�A�(*

loss�_�<j��#       �	��fc�A�(*

loss�7=��       �	��fc�A�(*

loss�Y�<mOb�       �	 ��fc�A�(*

loss�;=8(�#       �	#M�fc�A�(*

loss���;���       �	��fc�A�(*

loss�<ĵS�       �	���fc�A�(*

loss�g�:���O       �	�B�fc�A�(*

loss��e:�l�       �	��fc�A�(*

lossP@<S��       �	��fc�A�(*

loss���<�cE!       �	i�fc�A�(*

loss��<���       �	|��fc�A�(*

loss�=Q�#�       �	7R�fc�A�(*

loss��O=�00m       �	s�fc�A�(*

loss��4;� �S       �	��fc�A�(*

lossW*c:�h�       �	[$�fc�A�(*

loss,�+;�       �	^��fc�A�(*

loss���<�       �	SY�fc�A�(*

loss�l�<Sã�       �	���fc�A�(*

loss
�:��\�       �	���fc�A�(*

lossl8�<��       �	�:�fc�A�(*

lossX�:λ�]       �	^؏fc�A�(*

lossr)<�Қ       �	`v�fc�A�(*

loss�$g=�p       �	��fc�A�(*

loss�G8=�K7,       �	���fc�A�(*

loss}�q<���       �	pD�fc�A�(*

loss���<Et]       �	"ޒfc�A�(*

loss���;�uX       �	�t�fc�A�(*

loss�=IW(�       �	��fc�A�(*

loss���<`:�       �	X��fc�A�(*

loss(�<וr�       �	�F�fc�A�(*

loss��x<��A       �	/ߕfc�A�(*

loss��I<�̳       �	�s�fc�A�(*

lossW(<[��       �	��fc�A�(*

loss.Sx;c�."       �	�ԗfc�A�(*

loss\AA<��`       �	�v�fc�A�(*

loss֐�;� D       �	��fc�A�(*

losss7<c�       �	|��fc�A�(*

loss$*�;'�       �	DN�fc�A�(*

loss�;�<h�v{       �	��fc�A�(*

loss�=bH��       �	#��fc�A�(*

lossf�c<l�       �	�*�fc�A�(*

loss,Pw=�.i�       �	eĜfc�A�(*

loss��8;�Z��       �	�{�fc�A�(*

loss� <J��       �	!#�fc�A�(*

loss�P+>��I       �	]��fc�A�(*

loss��#<�t�       �	��fc�A�)*

loss	E�:#
p       �	���fc�A�)*

loss7r:}貋       �	N&�fc�A�)*

loss_�==�5��       �	Lšfc�A�)*

loss��<0��       �	5c�fc�A�)*

lossV-;�bX�       �	��fc�A�)*

lossƐ;��4�       �	s��fc�A�)*

lossӊ�<p�[       �	YR�fc�A�)*

loss�t�<����       �	���fc�A�)*

loss
;-��h       �	���fc�A�)*

lossv�)< )C       �	�.�fc�A�)*

loss��d;���B       �	$Ѧfc�A�)*

loss�<�ABv       �	�j�fc�A�)*

loss>o�=k���       �	:w�fc�A�)*

loss,�;%��n       �	��fc�A�)*

lossD�;h]��       �	4��fc�A�)*

loss|l�<���2       �	�Y�fc�A�)*

losss_�:��	�       �	��fc�A�)*

loss�G�<��(�       �	\��fc�A�)*

loss�Ld:�DW�       �	�&�fc�A�)*

loss1h<�0�       �	�­fc�A�)*

loss��\</i�       �	K\�fc�A�)*

lossiK�;��       �	���fc�A�)*

loss�^�=�t       �	���fc�A�)*

loss���;�c�       �	�'�fc�A�)*

lossO�;��@�       �	���fc�A�)*

loss��=���4       �	�\�fc�A�)*

loss|ڶ;v;�       �	���fc�A�)*

lossI�<am�       �	e��fc�A�)*

loss+��;aNI       �	�.�fc�A�)*

loss��;�e��       �	&ȳfc�A�)*

loss�֕;ً��       �	eq�fc�A�)*

loss�=�QPY       �	��fc�A�)*

lossJzd=�E+v       �	\��fc�A�)*

loss�v1<�43       �	�O�fc�A�)*

loss���<HO�       �	k�fc�A�)*

lossb{:~�Q       �	��fc�A�)*

loss�F�;:Z.�       �	QK�fc�A�)*

loss/l�:�-��       �	��fc�A�)*

loss�"0:�B��       �	撹fc�A�)*

loss�;5<CM�       �	�I�fc�A�)*

loss(�e<=*g�       �	]��fc�A�)*

lossMY;G�y       �	B?�fc�A�)*

lossʪ+<9�ݨ       �	��fc�A�)*

loss���:��%       �	W�fc�A�)*

lossc/1:PC��       �	&4�fc�A�)*

loss�V<c��Z       �	���fc�A�)*

lossv(;T�@s       �	���fc�A�)*

loss�O�<.{�k       �	�=�fc�A�)*

lossuw;}��W       �	V~�fc�A�)*

loss:��;���$       �	=C�fc�A�)*

loss�o;���       �	���fc�A�)*

loss�J;�;p       �	���fc�A�)*

loss�:�;��?�       �	S��fc�A�)*

lossn�:�>       �	�v�fc�A�)*

loss�<��,P       �	/�fc�A�)*

loss���;��       �	���fc�A�)*

loss�q0<=u�       �	�L�fc�A�)*

lossZN�<��       �	���fc�A�)*

lossR��;"���       �	$}�fc�A�)*

loss�f;<���       �	��fc�A�)*

loss�J;��	       �	��fc�A�)*

loss{̱<�~�       �	�F�fc�A�)*

loss�;gC�o       �	M��fc�A�)*

loss��:���       �	�r�fc�A�)*

loss��q:�7��       �	��fc�A�)*

lossS�;�K5�       �	v��fc�A�)*

lossX�x<Bc�       �	T:�fc�A�)*

loss<�<|*9       �	���fc�A�)*

loss9�=���       �	�b�fc�A�)*

lossM9��m       �	���fc�A�)*

lossѩ�<���       �	��fc�A�)*

loss�c�<�@;�       �	29�fc�A�)*

loss�,B:.|�Y       �	���fc�A�)*

loss4�;)kN{       �	Ym�fc�A�)*

loss!��984�V       �	]�fc�A�)*

loss6�9v�	�       �	���fc�A�)*

loss(�w:�-Kr       �	�e�fc�A�)*

lossXo�< q�|       �	���fc�A�)*

loss��-<�B	�       �	G��fc�A�)*

loss�)<Ej�       �	w��fc�A�)*

lossv��<c�mN       �	B!�fc�A�)*

lossy#;��M�       �	Թ�fc�A�)*

loss�=G:�5��       �	`�fc�A�)*

lossa_�=����       �	���fc�A�)*

lossfE�9#�:6       �	ɓ�fc�A�)*

lossX7o:q       �	�,�fc�A�)*

lossJض:P�'       �	��fc�A�)*

loss�=�;¨       �	OX�fc�A�)*

loss\L�;gr��       �	���fc�A�)*

loss�xj;5H��       �	���fc�A�)*

loss�!`8�y��       �	E��fc�A�)*

loss��<��#       �	���fc�A�)*

lossm��8G���       �	�^�fc�A�)*

lossx��7D��       �	���fc�A�)*

loss?*�7�^�z       �	U��fc�A�)*

loss�4j8g
\9       �	8�fc�A�)*

loss*��<ֳL3       �	��fc�A�)*

lossۏ:Q/~       �	�a�fc�A�)*

loss3�
:5K+�       �	���fc�A�)*

lossr�:_Π:       �	͓�fc�A�)*

loss)<H=i��6       �	�;�fc�A�)*

loss}��9�2�p       �	���fc�A�)*

loss�Ę=����       �	�r�fc�A�)*

losstI�<�k�k       �	�y�fc�A�)*

loss:�<�}�       �	&�fc�A�)*

loss�ܥ;�9i       �	��fc�A�)*

loss��:�GEc       �	ݚ�fc�A�)*

loss�=y%�>       �	m7�fc�A�)*

loss춓<�*>       �	���fc�A�)*

lossk]	:��c       �	���fc�A�)*

loss�3f;4��       �	�$�fc�A�)*

loss��P:=yP�       �	Z��fc�A�)*

loss��	;}c$0       �	!V�fc�A�)*

loss�v3:�(�       �	Y��fc�A�)*

loss��];UFΝ       �	,��fc�A�)*

loss��;�R�       �	5�fc�A�)*

lossoF�;�F��       �	5�fc�A�)*

loss��<.��       �	��fc�A�)*

loss4��<25,       �	�d�fc�A�)*

lossځ<Rl��       �	���fc�A�)*

losstU;�.�       �	6��fc�A�)*

loss��=;m�*�       �	�*�fc�A�)*

loss��<�e�R       �	�fc�A�)*

lossȴ�<I�R�       �	��fc�A�)*

loss��Q<�\?*       �	�=�fc�A�)*

loss��P:��T`       �	���fc�A�)*

loss�/>;����       �	ŭ�fc�A�)*

losss<;b���       �	:@�fc�A�**

loss��2<8Ks       �	���fc�A�**

loss�r�=�	       �	�e�fc�A�**

lossTb;�       �	;��fc�A�**

lossi8�:�sdi       �	���fc�A�**

loss�F<;q�       �	�(�fc�A�**

lossHX$<���       �	j��fc�A�**

loss2�:|�hB       �	6X�fc�A�**

loss�;�<��[�       �	�o�fc�A�**

losso�:I�o       �	:�fc�A�**

loss�;��@       �	1��fc�A�**

loss� <��&;       �	P5�fc�A�**

loss�%�<-�d�       �	!��fc�A�**

losslub=ޮ�!       �	b�fc�A�**

loss\�[:��_�       �	���fc�A�**

lossF{k:��D       �	ʈ fc�A�**

loss��=<Ҽ�       �	�fc�A�**

loss�ha;ܵ��       �	�fc�A�**

lossa�3<av�       �	Qfc�A�**

loss>�<��       �	��fc�A�**

lossO'�<k�       �	$�fc�A�**

loss�.<9k	�       �	�fc�A�**

loss%�L=gZ4�       �	�fc�A�**

loss���:�F)�       �	�Pfc�A�**

losshӾ<Ώ       �	��fc�A�**

loss��:���       �	ճ fc�A�**

loss���<�\�       �	�J!fc�A�**

loss���=<K��       �	��!fc�A�**

loss��;�Eb       �	:z"fc�A�**

loss�%�:�8�       �	�#fc�A�**

loss��:���       �	.�#fc�A�**

lossN�-<�T�       �	JA$fc�A�**

lossr�^<��4�       �	��$fc�A�**

loss�t�=�2H       �	�e%fc�A�**

loss� =׾0       �	�&fc�A�**

loss�	x:�*X       �	��&fc�A�**

loss6�;;݋�e       �	�3'fc�A�**

loss��;9W�       �	��'fc�A�**

loss���;m�V�       �	�b(fc�A�**

loss�"D=�G�       �	��(fc�A�**

loss�q�;y��       �	G�)fc�A�**

lossD&9��<       �	�*fc�A�**

lossJ��:>�e       �	s�*fc�A�**

loss��<=�i,P       �	KV+fc�A�**

loss���<���       �	�",fc�A�**

loss�4=*aM�       �	r�,fc�A�**

loss�Xa<Z��       �	 `-fc�A�**

loss�N;�,/}       �	r�-fc�A�**

loss�3B<$�        �	��.fc�A�**

lossڄ�:��d)       �	�#0fc�A�**

loss�uq;�g>�       �	��0fc�A�**

loss��s;�/       �	]S1fc�A�**

loss$��;����       �	��1fc�A�**

loss�κ<��9�       �	E�3fc�A�**

loss�l;�L�5       �	I-4fc�A�**

lossg�;���       �	��4fc�A�**

loss܆�<��c$       �	S^5fc�A�**

loss��P<�#7       �	Q�5fc�A�**

loss1Z_=7���       �	ӄ6fc�A�**

lossW��:ظ�Z       �	L7fc�A�**

lossn�8=(��       �	��7fc�A�**

loss�=:F��       �	TS8fc�A�**

loss|P;�aS0       �	�8fc�A�**

loss�0<,E2       �	Z�9fc�A�**

loss�*�;X��       �	�:fc�A�**

loss�
9=���       �	��:fc�A�**

loss�;
��       �	z;fc�A�**

loss�l�:�Q�       �	o<fc�A�**

lossq�;�H�       �	6�<fc�A�**

loss��#=��?S       �	sL=fc�A�**

loss�J#=+qRH       �	��=fc�A�**

loss�{�;��2x       �	�|>fc�A�**

lossqi;<�2��       �	�?fc�A�**

loss�a�<&$�g       �	�?fc�A�**

loss�>�9�(�       �	�D@fc�A�**

loss]\;�׳       �	�@fc�A�**

loss�h�<��U       �	�tAfc�A�**

loss��;u~b       �	�
Bfc�A�**

loss��=X��U       �	�Bfc�A�**

loss�;��&V       �	�;Cfc�A�**

lossI�;�6
        �	#�Cfc�A�**

loss�9;�Jx       �	�oDfc�A�**

loss�>9� q       �	�	Efc�A�**

loss��;��c       �		�Efc�A�**

loss k�=CW��       �	:>Ffc�A�**

loss)?�=��1�       �	@�Ffc�A�**

loss��e: P       �	:tGfc�A�**

lossX�:�#�       �	��Ifc�A�**

loss\?�<��|       �	�Jfc�A�**

lossn��9����       �	@�Lfc�A�**

loss-|]<�       �	��Mfc�A�**

loss��W:�I       �	�*Nfc�A�**

loss:G;��       �	�YOfc�A�**

lossN�=�e��       �	��Ofc�A�**

lossZ/X;M�,       �	��Pfc�A�**

loss���:��ՙ       �	]3Qfc�A�**

loss>��<}�X�       �	�Qfc�A�**

loss�#;#�y-       �	��Rfc�A�**

loss�mI;��~       �	Q�Sfc�A�**

loss{&);i���       �	�Tfc�A�**

loss�=�8t       �	E�Tfc�A�**

loss�h;t��       �	�LUfc�A�**

loss�>:K�4       �	��Ufc�A�**

lossI:�:�bi=       �	yVfc�A�**

lossv�<ҫ%       �	��Wfc�A�**

loss�B8:1�|       �	�#Xfc�A�**

lossF�J<��x2       �	��Xfc�A�**

loss�;Sz       �	+KYfc�A�**

lossM�9^��       �	��Yfc�A�**

loss�:�	��       �	�sZfc�A�**

loss�g�<D��       �	%[fc�A�**

loss�d�;�׷�       �	¢[fc�A�**

loss�<�g|       �	9G\fc�A�**

lossm_;h��       �	"]fc�A�**

loss/�6<�s\       �	u�]fc�A�**

lossQxF=�g�/       �	_\^fc�A�**

loss/��:�=\       �	�^fc�A�**

lossA�<�@��       �	hz_fc�A�**

loss�� =�9�       �	�`fc�A�**

lossŻ�=�n�r       �	Y�`fc�A�**

loss��;��1       �	�6afc�A�**

lossC�<s���       �	��afc�A�**

loss7G<}*�?       �	�dbfc�A�**

loss�`<r�!       �	�bfc�A�**

lossCR:�Lo�       �	6�cfc�A�**

loss��-:T��w       �	5$dfc�A�**

loss4?�;2�m�       �	x�dfc�A�**

loss*�;����       �	
Mefc�A�**

loss2��;��g�       �	v�efc�A�+*

loss7r='�        �	�offc�A�+*

loss�d=Y��       �	�gfc�A�+*

lossTj�9��$       �	�gfc�A�+*

loss���<޷C�       �	)hfc�A�+*

loss���=��3�       �	E�hfc�A�+*

loss%a�<�E#       �	Kifc�A�+*

loss:T{<��o       �	�ifc�A�+*

loss��=��|�       �	�rjfc�A�+*

loss�-=��fX       �	�kfc�A�+*

loss �:~Q��       �	g�kfc�A�+*

loss��=U��       �	i;lfc�A�+*

loss���;/��       �	N�lfc�A�+*

loss��+;N�tP       �	�vmfc�A�+*

loss��a=���       �	nnfc�A�+*

lossn��;����       �	únfc�A�+*

loss� �<�r�H       �	�Uofc�A�+*

loss�-�:���A       �	��ofc�A�+*

loss��;	���       �	��pfc�A�+*

loss:>=<^       �	Xqfc�A�+*

loss�H�<\Z       �	��qfc�A�+*

lossY�;۽�M       �	�Qrfc�A�+*

loss���;t�       �	~�rfc�A�+*

loss�*�;9��!       �	�~sfc�A�+*

loss�̮<l���       �	�tfc�A�+*

loss1B�<�+��       �	Ψtfc�A�+*

lossX�<h�`�       �	�8ufc�A�+*

loss��R<��Z�       �	��ufc�A�+*

loss�0;^�       �	�hvfc�A�+*

loss!1M<���       �	�wfc�A�+*

loss@}�<U�[�       �	��wfc�A�+*

loss2f<�u^+       �	(,xfc�A�+*

lossC\J:{�٥       �	��xfc�A�+*

loss�==p%�*       �	Zyfc�A�+*

loss ��;�k�       �	p�yfc�A�+*

loss�K5;�-K       �	Q�zfc�A�+*

loss,޷;�	       �	Yi{fc�A�+*

loss��9���       �	`�|fc�A�+*

loss[��<[g 0       �	L�}fc�A�+*

loss؊<{1r       �	 �~fc�A�+*

lossz��;rړ-       �	uZfc�A�+*

lossݪ�9|e_�       �	m�fc�A�+*

loss(��;\��       �		��fc�A�+*

loss���:�7m>       �	5D�fc�A�+*

loss��:��b       �	�݁fc�A�+*

loss#��:�w�       �	�v�fc�A�+*

loss��;R��l       �	�
�fc�A�+*

loss��;G���       �		��fc�A�+*

loss��Q=��       �	�;�fc�A�+*

loss��=�-�O       �	k҄fc�A�+*

lossF�_=wBN�       �	�2�fc�A�+*

loss�N;9y�       �	Uކfc�A�+*

loss<	;lMR       �	��fc�A�+*

loss?�S;�%$�       �	��fc�A�+*

loss���;�i�       �	Ĵ�fc�A�+*

loss�/ :�G       �	�W�fc�A�+*

lossQ�;~�z�       �	\�fc�A�+*

loss�� <��5       �	���fc�A�+*

lossCc�8b��}       �	*:�fc�A�+*

loss���:�n7L       �	�Ћfc�A�+*

loss��[:(e       �	j�fc�A�+*

loss��<LC�       �	e��fc�A�+*

loss6��:�>�       �	՗�fc�A�+*

loss���;��T       �	�0�fc�A�+*

lossȷ =����       �	jߎfc�A�+*

loss��h:��       �	�x�fc�A�+*

loss�2;�L�,       �	a�fc�A�+*

loss�V6<w��       �	�`�fc�A�+*

loss�$y;���       �	�fc�A�+*

loss�܇:�g�r       �	�	�fc�A�+*

loss_�2=��w       �	���fc�A�+*

loss�H,:48h       �	�<�fc�A�+*

loss�}<��l�       �	I.�fc�A�+*

loss��/<b�_�       �	�ٕfc�A�+*

loss�!<��tI       �	���fc�A�+*

loss@m�;T�       �	�"�fc�A�+*

loss�s�:��,�       �	�ʗfc�A�+*

loss}��8h�a       �	kc�fc�A�+*

loss�s�<����       �	� �fc�A�+*

lossԝ1:ip2�       �	���fc�A�+*

loss#�#;وl*       �	�F�fc�A�+*

losspI9���,       �	i �fc�A�+*

loss&);��ء       �	��fc�A�+*

lossH�=1�uV       �	�I�fc�A�+*

loss�}�;�6o�       �	�fc�A�+*

lossH	;+ rp       �	���fc�A�+*

loss��3<�c��       �	)@�fc�A�+*

loss.|/=�Ile       �	Lޞfc�A�+*

loss/��:}��       �	�v�fc�A�+*

loss�;�l�n       �	\ �fc�A�+*

loss|0�;ɩ:f       �	ǹ�fc�A�+*

lossC"9�d}�       �	Gt�fc�A�+*

lossi��;s$��       �	��fc�A�+*

loss���;�Z �       �	L��fc�A�+*

loss�xC;�9��       �	>�fc�A�+*

lossA �:Ϸ(n       �	�ۣfc�A�+*

lossM��;���       �	�n�fc�A�+*

loss�/�<:���       �	� �fc�A�+*

loss\��:a���       �	���fc�A�+*

loss`I7<9��       �	%?�fc�A�+*

losszb�:"��n       �	4צfc�A�+*

loss�u�:_�D       �	�h�fc�A�+*

loss�=퇹�       �	>&�fc�A�+*

lossJ�s;\�       �	^��fc�A�+*

loss ٳ;?Z��       �	ӡ�fc�A�+*

lossF� =�mY       �	�6�fc�A�+*

loss9Z;����       �	�ƪfc�A�+*

loss��;���       �	]�fc�A�+*

lossqt�:ȸ	�       �	��fc�A�+*

loss8O$<Q�U�       �	ˁ�fc�A�+*

loss��;{.F&       �	^�fc�A�+*

loss\)�;��       �	\��fc�A�+*

loss���<�>�       �	u��fc�A�+*

loss��>��=�       �	�r�fc�A�+*

lossH��=t7       �	��fc�A�+*

lossl!g=ʞqD       �	�Ȱfc�A�+*

lossh)�<ρ��       �	_`�fc�A�+*

lossz�<���       �	���fc�A�+*

loss:�N<#��S       �	r��fc�A�+*

losszeS;}��       �	�׳fc�A�+*

lossOp;#z��       �	Dl�fc�A�+*

loss&�d<ga�       �	�fc�A�+*

lossT�>:Sl       �	���fc�A�+*

loss�<�;.D       �	�F�fc�A�+*

loss�}�;�iݴ       �	�ܶfc�A�+*

losshy6;S&�       �	mq�fc�A�+*

loss�3�:5��V       �	,.�fc�A�+*

loss�%�:3f�       �	øfc�A�+*

lossv]O:^�|�       �	�Y�fc�A�,*

loss{��<?g�       �	��fc�A�,*

loss*͆;��P       �	���fc�A�,*

loss#:	<�_R�       �	�'�fc�A�,*

loss6xV;�o��       �	)]�fc�A�,*

loss�U_<��       �	���fc�A�,*

losswGO=%�R       �	�P�fc�A�,*

loss��I<K�{       �	�fc�A�,*

lossTn:j"6       �	���fc�A�,*

loss�<�k�H       �	�U�fc�A�,*

lossr:�<J��       �	�>�fc�A�,*

lossV�;:���       �	���fc�A�,*

loss�?�<7<e       �	��fc�A�,*

loss��=��ȱ       �	%��fc�A�,*

loss��;T�!       �	I��fc�A�,*

loss��R;��}�       �	GW�fc�A�,*

loss��:n��       �	���fc�A�,*

loss�B>@ՠ       �	��fc�A�,*

loss �;^ߔY       �	Q3�fc�A�,*

loss�'�<x���       �	5��fc�A�,*

loss
�6;q���       �	Q��fc�A�,*

lossdvA<���       �	`!�fc�A�,*

lossx��;y~��       �	���fc�A�,*

loss{;����       �	Hk�fc�A�,*

loss��S9�y�D       �	A�fc�A�,*

lossd��;7�s�       �	ձ�fc�A�,*

loss|�0;a���       �	^M�fc�A�,*

loss<d:�ލ�       �	}��fc�A�,*

loss	��<�\��       �	}�fc�A�,*

lossR��<�_$I       �	��fc�A�,*

loss2�7:X��k       �	���fc�A�,*

loss��"=/��4       �	4K�fc�A�,*

lossK�<����       �	���fc�A�,*

loss\G�;��a�       �	p|�fc�A�,*

lossA� :�3       �	#�fc�A�,*

lossN��<L��       �	N��fc�A�,*

lossQ�=��;�       �	��fc�A�,*

loss��=�9��       �	���fc�A�,*

lossxS�;�Pq       �	�W�fc�A�,*

loss���:��Ow       �	��fc�A�,*

loss!��;��a       �	c��fc�A�,*

loss�wz;�       �	�.�fc�A�,*

loss� =2��       �	���fc�A�,*

loss�+�<��V       �	,b�fc�A�,*

loss�'<����       �	��fc�A�,*

loss�~�:�X�       �	Z��fc�A�,*

lossQ�;����       �	35�fc�A�,*

loss���<l�       �	���fc�A�,*

loss�F'<�Dk       �	�l�fc�A�,*

loss�`R8���       �	��fc�A�,*

loss��"<��<]       �	���fc�A�,*

loss��;3?�       �	K<�fc�A�,*

loss�=�j�;       �	���fc�A�,*

loss���;��       �	�r�fc�A�,*

loss�<����       �	 `�fc�A�,*

loss��<��E�       �	��fc�A�,*

lossʈ�<x`��       �	���fc�A�,*

loss�g�:Y�
�       �	�1�fc�A�,*

loss۠�<��j       �	���fc�A�,*

loss��;�06q       �	>z�fc�A�,*

lossm�'<��I       �	� �fc�A�,*

lossS��9k/��       �	���fc�A�,*

loss��90���       �	�Y�fc�A�,*

loss ��:��Q|       �	��fc�A�,*

lossƛ�<])��       �	���fc�A�,*

lossa&A<��g       �	�fc�A�,*

lossC<�=��9�       �	;��fc�A�,*

lossd�`<4�=�       �	=H�fc�A�,*

loss`=t{Ke       �	f��fc�A�,*

loss��;s�~H       �	�v�fc�A�,*

loss���;G�";       �	��fc�A�,*

lossq�}:�'a�       �	��fc�A�,*

loss!K;�T�       �	�F�fc�A�,*

lossP<�g�r       �	���fc�A�,*

loss�8�=P
lK       �	:x�fc�A�,*

loss��9�႑       �	1�fc�A�,*

loss=�A=V��.       �	���fc�A�,*

loss�J�:'�H       �	���fc�A�,*

loss;��:xY�G       �	uw�fc�A�,*

loss�a�;-���       �	��fc�A�,*

loss���:)���       �	��fc�A�,*

loss���=�6       �	�[�fc�A�,*

loss�s�:��=       �	���fc�A�,*

loss_J=� �       �	g��fc�A�,*

lossM�J=��       �	V,�fc�A�,*

lossj�s9� ��       �	h��fc�A�,*

loss1{;�i�       �	�n�fc�A�,*

loss��C;?F�?       �	0�fc�A�,*

loss��|<�V�E       �	���fc�A�,*

loss1<���       �	�Q�fc�A�,*

loss2%&<i�%�       �	t��fc�A�,*

loss���;����       �	���fc�A�,*

losshȯ:/���       �	If�fc�A�,*

loss��<��rW       �	�fc�A�,*

loss��9<b�.K       �	k��fc�A�,*

loss��<OOz       �	�?�fc�A�,*

loss37c=
�t�       �	j��fc�A�,*

loss�:�+P        �	%y�fc�A�,*

loss� =��`L       �	)��fc�A�,*

lossZt0<�m       �	q��fc�A�,*

loss䛛;MĲ       �	>��fc�A�,*

lossC�:���       �	���fc�A�,*

lossa�;��#@       �	ǂ fc�A�,*

lossVG=���v       �	�)fc�A�,*

lossMq
:L�]       �	+�fc�A�,*

loss2)=�3       �	s�fc�A�,*

lossܦ�;�Ca       �	mfc�A�,*

lossV�=���       �	�Mfc�A�,*

lossQ��8�l       �	��fc�A�,*

lossc�G<�گ^       �	��fc�A�,*

lossCN�;`b@       �	5�fc�A�,*

losst�;���f       �	y�fc�A�,*

loss�!�;*Nf       �	 'fc�A�,*

loss�=hR6       �	<	fc�A�,*

loss��?=�dь       �	`!
fc�A�,*

loss��f;yp��       �	��
fc�A�,*

loss��;�kv�       �	 �fc�A�,*

loss� �<v�       �	J�fc�A�,*

lossR��:\%�       �	�Cfc�A�,*

loss�k<X��       �	>Yfc�A�,*

lossLi1;;Ӏ�       �	͐fc�A�,*

loss�z+=Y�l       �	?�fc�A�,*

loss�:�r�       �	d�fc�A�,*

loss۬�;�N��       �	I0fc�A�,*

loss+;N�̯       �	x�fc�A�,*

lossV�?<"�a#       �	xzfc�A�,*

loss*�:�i�       �	�fc�A�,*

loss�R�;Ȭ�       �	P�fc�A�,*

loss-��=���       �	fc�A�-*

loss�x�:�s       �	�$fc�A�-*

loss��;�*��       �	��fc�A�-*

lossF�B<U	�t       �	]lfc�A�-*

loss�)�<n�n@       �	�fc�A�-*

loss��;���       �	�fc�A�-*

loss|#�<�"g�       �	Efc�A�-*

lossE��<^a��       �	"�fc�A�-*

loss��;?I9       �	�xfc�A�-*

loss�8;g6��       �	�fc�A�-*

loss�g<�� &       �	Ӿfc�A�-*

loss���;�2��       �	�ffc�A�-*

lossO��<S6M       �	�fc�A�-*

lossb%;���       �	��fc�A�-*

loss�E�:�	!�       �	nLfc�A�-*

loss�f=e��b       �	��fc�A�-*

loss71;�YA       �	��fc�A�-*

loss�x�=�n�       �	?� fc�A�-*

loss:R�;	��!       �	�E!fc�A�-*

loss�	�;Xd�r       �	��!fc�A�-*

loss�n�=��ދ       �	I�"fc�A�-*

loss�V�:j��`       �	�#fc�A�-*

loss)C7=�5��       �	d�#fc�A�-*

lossڱ�<g�/�       �	�M$fc�A�-*

loss��<���       �	��$fc�A�-*

loss=|�:mL��       �	��%fc�A�-*

lossi	=
R=       �	z�&fc�A�-*

loss��&<^*�       �	u!'fc�A�-*

lossӋ�;=ê       �	ܷ'fc�A�-*

loss��;\Kl       �	XW(fc�A�-*

loss!'4<D;�R       �	��(fc�A�-*

lossE��;�R�       �	J�)fc�A�-*

loss<��<7�}?       �	�.*fc�A�-*

loss�(�<����       �	��*fc�A�-*

loss��A<��Њ       �	�p+fc�A�-*

loss�)�<���?       �	c,fc�A�-*

loss4�<B��.       �	"�,fc�A�-*

loss!F9�k�)       �	�?-fc�A�-*

loss�:m�l/       �	^�-fc�A�-*

lossmz�<�?��       �	�s.fc�A�-*

loss:��;�=��       �	�/fc�A�-*

loss��,<���4       �	&�/fc�A�-*

loss��<H�u�       �	�>0fc�A�-*

loss��<�I��       �	0�0fc�A�-*

loss;�;���       �	Tq1fc�A�-*

lossL]�<ԆHt       �	R2fc�A�-*

loss:@F< Z�U       �	u�2fc�A�-*

loss6�< ~��       �	�J3fc�A�-*

loss�l<��       �	a�3fc�A�-*

loss��;�%��       �	A~4fc�A�-*

loss6��;���6       �	�5fc�A�-*

loss	�
=ށ]       �	�5fc�A�-*

loss7I#<�~V       �	�D6fc�A�-*

loss;�m<`ǃ       �	��6fc�A�-*

loss���=V��M       �	w7fc�A�-*

loss���;��$       �	�8fc�A�-*

loss��x;"� b       �	@�8fc�A�-*

loss$f�;�{S)       �	~;9fc�A�-*

loss�B�:XS��       �	��9fc�A�-*

loss�rl<�<��       �	Yk:fc�A�-*

loss��<ųɴ       �	�;fc�A�-*

lossJ˙;�9�C       �	k�;fc�A�-*

loss|�W=0�A6       �	��<fc�A�-*

loss-��8σd�       �	��=fc�A�-*

loss�V�<����       �	'�>fc�A�-*

lossz��<nD(       �	��?fc�A�-*

lossG�;��\�       �	Afc�A�-*

loss�)�<}?�       �	��Afc�A�-*

loss]Q'=;a�k       �	W�Bfc�A�-*

loss���;7^�       �	Dfc�A�-*

lossI�<y9       �	�Efc�A�-*

loss��E:�o�       �	�Efc�A�-*

loss��:�Y�       �	�Gfc�A�-*

loss-;�O�8       �	_�Gfc�A�-*

loss�=;)���       �	NbHfc�A�-*

loss:�;�x�       �	'lIfc�A�-*

lossA��:v\h       �	�rJfc�A�-*

loss���;��c�       �	�Kfc�A�-*

loss��=Gr0�       �	_�Kfc�A�-*

loss@^�::�{�       �	��Lfc�A�-*

loss*��;lܶ�       �	\vMfc�A�-*

lossC8�<L��b       �	�Nfc�A�-*

loss�ڐ:vv       �	z�Nfc�A�-*

loss�;���       �	�Ofc�A�-*

loss���<6��       �	^0Pfc�A�-*

loss�:��u       �	��Pfc�A�-*

lossN<�%��       �	_^Qfc�A�-*

loss|��<('�j       �	H�Qfc�A�-*

loss�f<՞1       �	�Rfc�A�-*

loss�T=���       �	�&Sfc�A�-*

loss�);� B       �	ϻSfc�A�-*

loss���<�="�       �	vRTfc�A�-*

loss�
;��r	       �	X�Tfc�A�-*

loss��0<���       �	g{Ufc�A�-*

loss��n:`�q\       �	MVfc�A�-*

loss���<��       �	ۦVfc�A�-*

lossR`1=��%�       �	=Wfc�A�-*

lossIFF;�Nu       �	$�Wfc�A�-*

loss`qc<Ʀ�       �	alXfc�A�-*

loss�a:<�T       �	"�Xfc�A�-*

loss��=�#�       �	��Yfc�A�-*

loss�<�P7H       �	z7Zfc�A�-*

loss�EP:+�T       �	��Zfc�A�-*

loss�W�:J        �	�w[fc�A�-*

lossf��=T�       �	\fc�A�-*

lossI�<!f�       �	�\fc�A�-*

loss��:]��       �	�]]fc�A�-*

loss$=T��       �	��]fc�A�-*

loss�a�=w5,{       �	=�^fc�A�-*

lossv��;�d��       �	�:_fc�A�-*

loss�<����       �	��_fc�A�-*

loss�[�=J�y�       �	��`fc�A�-*

loss�el=� �V       �	LUafc�A�-*

loss�T;�%�       �	��afc�A�-*

loss���<���8       �	��bfc�A�-*

losss ;#�B        �	�cfc�A�-*

lossd�r:J�x�       �	��cfc�A�-*

loss��;�5��       �	Gdfc�A�-*

lossN?k= c^       �	��dfc�A�-*

lossq�2:x�       �	�oefc�A�-*

loss�)�=�%2       �	yffc�A�-*

loss�7I=J	e�       �	8�ffc�A�-*

loss��H<�A��       �	�;gfc�A�-*

loss�p:�'@�       �	��gfc�A�-*

loss�N0:�o�       �	c�hfc�A�-*

loss�R'=�ۗE       �	�ifc�A�-*

loss,F;7�H       �	�ifc�A�-*

loss��:�D��       �	Jjfc�A�-*

lossJ�;����       �	&�kfc�A�.*

loss�~;�>�?       �	_Blfc�A�.*

loss{�=.�Y�       �	��lfc�A�.*

loss���;0G��       �	�umfc�A�.*

loss+,�<�
��       �	^nfc�A�.*

loss���:T���       �	��nfc�A�.*

loss��<���4       �	�Vofc�A�.*

loss83<�h�       �	f�ofc�A�.*

loss��z<h�x       �	��qfc�A�.*

loss�+�;x���       �	*Urfc�A�.*

loss��=6Z�Q       �	�rfc�A�.*

loss溶;jD�       �	P�sfc�A�.*

loss��<�RS�       �	�-tfc�A�.*

loss�6c<�=Vm       �	y�tfc�A�.*

lossE:;}��y       �	�fufc�A�.*

loss��;�4        �	�	vfc�A�.*

loss�4�:���       �	P�vfc�A�.*

lossZ=iwr       �	k�wfc�A�.*

loss��
<(�P�       �	�Axfc�A�.*

loss�v<��A?       �	��xfc�A�.*

loss8ȑ<���       �	$�yfc�A�.*

loss)v)="�8�       �	�zfc�A�.*

loss���<8�l       �	��zfc�A�.*

loss p�:q��       �	�g{fc�A�.*

lossiѧ<�|U�       �	��|fc�A�.*

loss�E<����       �	��}fc�A�.*

loss䎍;WnR�       �	u�~fc�A�.*

lossSw<<dN]�       �	k(�fc�A�.*

loss�;��C�       �	�ʀfc�A�.*

loss�&�<�,j�       �	ڐ�fc�A�.*

loss��1<炐I       �	V�fc�A�.*

loss�
K95$ǉ       �	��fc�A�.*

loss�F�<�؋�       �	�M�fc�A�.*

loss�u�8�Y�c       �	[#�fc�A�.*

loss�1�<�}i_       �	�fc�A�.*

loss ��9[}�       �	� �fc�A�.*

loss�G:!��q       �	%ɇfc�A�.*

loss�:0<�v��       �	���fc�A�.*

loss,��<jA�       �	|,�fc�A�.*

lossu�:��#       �	�҉fc�A�.*

loss�E;R���       �	���fc�A�.*

loss��%<Ǻ�|       �	Ҍ�fc�A�.*

lossWq�;|~��       �	/�fc�A�.*

loss!<�1�       �	�ʌfc�A�.*

loss'X<��i       �	9{�fc�A�.*

loss��9���       �	���fc�A�.*

loss���8�ɴ       �	u��fc�A�.*

loss8��9AǇ�       �	]��fc�A�.*

loss�:��\       �	�J�fc�A�.*

losswI;��Y       �	f��fc�A�.*

loss|q�<~�N�       �	��fc�A�.*

loss܏�:���O       �	�j�fc�A�.*

loss�&�<.��/       �	�fc�A�.*

loss-�^9��z       �	�%�fc�A�.*

loss��m6��:       �	�G�fc�A�.*

loss�ϭ9���       �	'��fc�A�.*

losso�:�q�$       �	k��fc�A�.*

loss��y:Be�F       �	�V�fc�A�.*

loss�M�:�(/(       �	z��fc�A�.*

loss��
:�5>J       �	�H�fc�A�.*

loss�|�<qR�       �	�:�fc�A�.*

loss��<`�JZ       �	i��fc�A�.*

loss1�9�tz       �	힜fc�A�.*

losse�=�r�[       �	aS�fc�A�.*

loss�w<����       �	k��fc�A�.*

loss�u�;��U�       �	��fc�A�.*

losső;�w�%       �	�%�fc�A�.*

loss q�<r�%�       �	�Ġfc�A�.*

lossik<[�V*       �	$��fc�A�.*

loss�M\;{|[�       �	-\�fc�A�.*

loss6�<q<��       �	;��fc�A�.*

lossZ��:��       �	���fc�A�.*

loss�^�;S�g&       �	�Q�fc�A�.*

lossc�<���       �	���fc�A�.*

loss��<�`�       �	��fc�A�.*

loss��W:�^��       �	�0�fc�A�.*

loss��6<b'�       �	=+�fc�A�.*

lossqm$=-�g�       �	Tɧfc�A�.*

loss
�<<єv�       �	Ab�fc�A�.*

loss�c;Rw�\       �	��fc�A�.*

loss���<�)#D       �	�@�fc�A�.*

loss�Ý;�}�0       �	\W�fc�A�.*

loss/+$:Vn��       �	A��fc�A�.*

loss� ;"ݩ       �	�fc�A�.*

loss~]�;ސ�(       �	&5�fc�A�.*

loss�#:�Gi       �	�׭fc�A�.*

loss��9vp�K       �	�k�fc�A�.*

lossω�:��       �	�fc�A�.*

lossq��9��,�       �	힯fc�A�.*

loss���:���       �	�V�fc�A�.*

loss��z=�0��       �	�fc�A�.*

loss�\�=I�ط       �	ĵ�fc�A�.*

loss隦:V[       �	j�fc�A�.*

loss
:c��O       �	� �fc�A�.*

loss!�:I�        �	1��fc�A�.*

loss��<��I       �	�(�fc�A�.*

loss��<��       �	^�fc�A�.*

lossxn;[K-       �	���fc�A�.*

lossc�	;�izj       �	��fc�A�.*

loss��M;#߳K       �	��fc�A�.*

loss�A7:SÂG       �	�?�fc�A�.*

loss($:����       �	�
�fc�A�.*

loss��O;t~̒       �	^��fc�A�.*

loss��6<�.uQ       �	80�fc�A�.*

loss�1�<"���       �	.ɹfc�A�.*

loss���9c�       �	-]�fc�A�.*

loss*e�;����       �	���fc�A�.*

loss��K;� �       �	���fc�A�.*

loss�g#=��E       �	�̼fc�A�.*

loss�Al;{�]       �	�	�fc�A�.*

loss��<v�V       �	k��fc�A�.*

lossƕ�;,       �	���fc�A�.*

loss(�:�{ځ       �	�I�fc�A�.*

loss7#4;.�I7       �	XS�fc�A�.*

loss� �;���       �	���fc�A�.*

loss�T}=�^;�       �	P��fc�A�.*

loss�X�;,T*�       �	&�fc�A�.*

lossq>�;�u�       �	2��fc�A�.*

losst:�;��n       �	Ag�fc�A�.*

loss��;�7�       �	5	�fc�A�.*

loss]s9:]���       �	���fc�A�.*

loss;;y3��       �	C�fc�A�.*

loss`�;sD�M       �	��fc�A�.*

lossH�I:'% �       �	?r�fc�A�.*

lossK]�:���[       �	��fc�A�.*

loss�W$;�wU       �	���fc�A�.*

loss�y="��       �	�<�fc�A�.*

loss,aM;��       �	V�fc�A�.*

lossXG:��n       �	ū�fc�A�/*

loss� �8 �       �	�E�fc�A�/*

lossŴ<0lcl       �	%��fc�A�/*

lossi��;�A�       �	]��fc�A�/*

loss�<_7       �	��fc�A�/*

loss�R*<�i�       �	u��fc�A�/*

lossqk�<w��       �		��fc�A�/*

lossÿ:���       �	���fc�A�/*

loss��<a���       �	�u�fc�A�/*

loss�W;�G��       �	C �fc�A�/*

loss�Z><��Y0       �	+��fc�A�/*

lossi�w:-�L       �	u �fc�A�/*

loss̿�:_+NX       �	X��fc�A�/*

lossꖉ:?�n;       �	�i�fc�A�/*

loss[�L<v�1�       �	��fc�A�/*

lossza�<ft��       �	��fc�A�/*

loss��;3?��       �	�a�fc�A�/*

loss�<<\�U       �	`�fc�A�/*

lossWo[<ʒ�@       �	8��fc�A�/*

loss�H;�H��       �	�G�fc�A�/*

loss�E_=xd,�       �	���fc�A�/*

loss׆9S��       �	xz�fc�A�/*

loss
�;�:       �	��fc�A�/*

loss��<s�J       �	c��fc�A�/*

lossZ	^<�P��       �	Y�fc�A�/*

loss�#�<��       �	���fc�A�/*

loss��:@�p       �	Й�fc�A�/*

loss���;�VP�       �	�4�fc�A�/*

lossJdf:��P�       �	���fc�A�/*

loss��::�Ic       �	c�fc�A�/*

loss��>=W�       �	F	�fc�A�/*

loss��;��       �	���fc�A�/*

loss.D�<���7       �	�:�fc�A�/*

loss Ze<����       �	w��fc�A�/*

losscaw;ߩ+�       �	���fc�A�/*

loss�b';J�9;       �	~;�fc�A�/*

loss�o;��       �	��fc�A�/*

loss�Z�:�B1V       �	G��fc�A�/*

lossJ�=�e%�       �	�i fc�A�/*

loss��9<n��       �	j4fc�A�/*

loss���:���-       �	�2fc�A�/*

loss�:�9HNʸ       �	�fc�A�/*

loss��;:{(�       �	<�fc�A�/*

loss�F�<}�|        �	�|fc�A�/*

loss�b-=)��       �	3fc�A�/*

losso�=c!^�       �	J�fc�A�/*

loss9z�<]P�       �	l{fc�A�/*

loss��9���i       �	�@fc�A�/*

loss�ZY<��P�       �	��fc�A�/*

lossJ+;p�X       �	�sfc�A�/*

loss�;�Y�0       �		fc�A�/*

loss��=���       �	
�	fc�A�/*

loss�'Z<�Y       �	�^
fc�A�/*

loss��=���       �	��
fc�A�/*

lossh�&=�?       �	��fc�A�/*

loss�6�<��       �	�%fc�A�/*

lossR\7=��"�       �	�fc�A�/*

loss��:�!�7       �	dWfc�A�/*

loss�^i;����       �	_�fc�A�/*

loss1� =��	       �	^�fc�A�/*

lossH};+a�       �	'fc�A�/*

loss�	�;A�       �	��fc�A�/*

lossdo�9���       �	�cfc�A�/*

lossϘ<?���       �	�fc�A�/*

loss���;����       �	؟fc�A�/*

lossa�P<��e       �	d?fc�A�/*

loss߽�;�P�       �	��fc�A�/*

lossa' <���I       �	�~fc�A�/*

loss�@�:�E`|       �	�fc�A�/*

lossV9�:qh       �	�fc�A�/*

loss6C�;�t'S       �	�[fc�A�/*

lossl��;�|l       �	�fc�A�/*

lossof=��d�       �	}�fc�A�/*

loss�s�<*�       �	H3fc�A�/*

loss��85�b       �	��fc�A�/*

loss���=uk�       �	�tfc�A�/*

loss��;L�@�       �	�fc�A�/*

loss*�y:�ߘ       �	�fc�A�/*

lossT@�;����       �	Dfc�A�/*

loss��3=a���       �	 �fc�A�/*

loss�M;?L�s       �	��fc�A�/*

loss���;��ZN       �	�$fc�A�/*

loss%��:�üf       �	�fc�A�/*

lossA�<BHR�       �	�Vfc�A�/*

loss*e�;�         �	��fc�A�/*

loss$z;�"�-       �	�fc�A�/*

loss�Jq9U�L       �	`9fc�A�/*

loss��Q=�1�       �	��fc�A�/*

loss�N;�Xa:       �	F~ fc�A�/*

loss�%�<2��h       �	�!fc�A�/*

loss��=s��       �	��"fc�A�/*

loss��:��As       �	=(#fc�A�/*

loss���8葎�       �	B^$fc�A�/*

loss��;��G�       �	�%fc�A�/*

loss���<:�       �	��%fc�A�/*

loss=re<��       �	h&fc�A�/*

loss�'�;FV	       �	�@'fc�A�/*

lossM�=`��       �	�(fc�A�/*

lossö�<�8�       �	ץ(fc�A�/*

loss�{�;F��       �	�)fc�A�/*

loss.I�:^_>       �	�*fc�A�/*

loss��&<�i�B       �	6�*fc�A�/*

loss �5;�-�       �	-C+fc�A�/*

lossZy<-�2�       �	��+fc�A�/*

loss�N;�ߒ�       �	�q,fc�A�/*

loss�9;���       �	-C-fc�A�/*

loss}x�<FJ��       �	Ov.fc�A�/*

loss��9]��       �	��/fc�A�/*

lossh�:>$��       �	u�0fc�A�/*

lossR#+;yX�g       �	�D1fc�A�/*

lossR�<P�$(       �	Y�1fc�A�/*

lossqe^<��
!       �	�s2fc�A�/*

loss��;\�̠       �	�3fc�A�/*

lossRJ<6 	�       �	A�3fc�A�/*

loss gB;�7��       �	�]4fc�A�/*

loss��8��\�       �	 5fc�A�/*

loss��;����       �	d�5fc�A�/*

loss�R�:�2       �	�-6fc�A�/*

lossߔ�:rц|       �	5�6fc�A�/*

lossһ2<s�Tz       �	�|7fc�A�/*

loss�::��F�       �	�+8fc�A�/*

lossQ)<c]�       �	��8fc�A�/*

lossci<�%��       �	�]9fc�A�/*

loss|;`v~       �	f�9fc�A�/*

loss��;�K�}       �	��:fc�A�/*

lossί:����       �	S>;fc�A�/*

loss
�;� w\       �	�;fc�A�/*

lossB�;�k(4       �	dt<fc�A�/*

loss�(�<�C�%       �	�=fc�A�0*

lossվ<��       �	3�=fc�A�0*

losst�:<gZGg       �	�>fc�A�0*

loss��<:��N       �	�?fc�A�0*

loss�I�:����       �	`"@fc�A�0*

loss �9�=�       �	��@fc�A�0*

lossl�=U�T|       �	0�Afc�A�0*

lossv��;�]�.       �	F�Bfc�A�0*

loss!�<�2�!       �	��Cfc�A�0*

loss%RU<��y9       �	�NDfc�A�0*

losst�R<���p       �	gEfc�A�0*

loss!:��,�       �	��Efc�A�0*

loss��:�	��       �	W�Ffc�A�0*

lossSc=r&�       �	��Gfc�A�0*

losslm%;�]       �	�DHfc�A�0*

lossLA<�gw       �	�7Ifc�A�0*

loss)�:�.W�       �	�Jfc�A�0*

loss�1g<a�-�       �	b�Jfc�A�0*

lossvp�:��.       �	��Kfc�A�0*

lossz�;/ۊ       �	4gLfc�A�0*

lossI!<qwM       �	MMfc�A�0*

loss��79��       �	��Mfc�A�0*

loss���:x�k>       �	�{Nfc�A�0*

lossr^F<� �       �	Ofc�A�0*

loss.;�2��       �	�Pfc�A�0*

lossL$�9����       �	��Pfc�A�0*

lossJ�;�J�       �	�:Qfc�A�0*

loss�3=��ڝ       �	��Qfc�A�0*

lossC0�<��}�       �	:tRfc�A�0*

lossm��:�xcs       �	Sfc�A�0*

lossIF�<�S��       �	�Sfc�A�0*

lossJ��<�R��       �	�QTfc�A�0*

lossH+&<4��       �	�Tfc�A�0*

loss%H=���       �	�Ufc�A�0*

lossXE=/"       �	J)Vfc�A�0*

lossĝB=�'��       �	:�Vfc�A�0*

lossmM;�n��       �	gWfc�A�0*

loss��:[��.       �	%Xfc�A�0*

losswT�<�	�       �	��Xfc�A�0*

loss8��;�<�       �	�.Yfc�A�0*

loss��<s��       �	��Yfc�A�0*

lossRk�;��l       �	�rZfc�A�0*

lossj�;o       �	�[fc�A�0*

loss�"<Yϭ       �	��[fc�A�0*

loss�,%;���I       �	lC\fc�A�0*

lossF��<�2o�       �	}�\fc�A�0*

loss��}<,-��       �	��]fc�A�0*

lossF��;��۪       �	5^fc�A�0*

lossw��:���       �	��^fc�A�0*

loss=�9AO�       �	�g_fc�A�0*

loss�<ճ9       �	�_fc�A�0*

loss�+9U�Uh       �	5�`fc�A�0*

loss4>=����       �	�3afc�A�0*

loss�Ul:�T�       �	��afc�A�0*

loss��!=
�\�       �	�abfc�A�0*

loss�} =���       �	Vcfc�A�0*

lossE�'<�wHK       �	l�cfc�A�0*

loss��=ٙ,�       �	cbefc�A�0*

lossX��9�10T       �	-	ffc�A�0*

loss�m�=Y��T       �	WCgfc�A�0*

loss��s;�~��       �	՗hfc�A�0*

loss�=ג�       �	Aifc�A�0*

loss�Z9�<]       �	=,jfc�A�0*

loss��<�	�}       �	��jfc�A�0*

lossm�_=�S�       �	�fkfc�A�0*

lossq�<��       �	�lfc�A�0*

lossU=]� K       �	�lfc�A�0*

lossnú:����       �	�1mfc�A�0*

loss��]<'�vz       �	]�mfc�A�0*

loss��<��       �	�nnfc�A�0*

loss
��;ȫ\�       �	ofc�A�0*

loss�F0=Dx2       �	ūofc�A�0*

lossɗa=Nݥ\       �	AGpfc�A�0*

loss:�<SZ�       �	L�pfc�A�0*

loss�d?=�?	/       �	{qfc�A�0*

loss��<����       �	.rfc�A�0*

loss�e�<�c#k       �	иrfc�A�0*

loss��;1���       �	rQsfc�A�0*

loss�؀;�ϥe       �	��sfc�A�0*

lossTc�:�}{�       �	��tfc�A�0*

loss:�;���       �	�ufc�A�0*

loss�8c;ޖ��       �	 �ufc�A�0*

loss�7<R0K       �	�lvfc�A�0*

loss�V6=;#       �	�wfc�A�0*

lossX�v:M�9�       �	]�wfc�A�0*

lossm_�:�L�y       �	�Qxfc�A�0*

loss`�=�h��       �	��xfc�A�0*

loss4��;�%@�       �	��yfc�A�0*

loss�;b�H-       �	#2zfc�A�0*

loss�P�;����       �	��zfc�A�0*

lossL��<}p�W       �	�m{fc�A�0*

lossfM<;�       �	m |fc�A�0*

loss�EV=�=�       �	�|fc�A�0*

loss}�<��_       �	��}fc�A�0*

loss�K=�t�       �	G�~fc�A�0*

lossV�w<�sq       �	�Rfc�A�0*

lossŊ�:����       �	|�fc�A�0*

loss�J�:��b       �	5��fc�A�0*

lossa�;r�p�       �	
/�fc�A�0*

loss�e�<�X��       �	�Łfc�A�0*

loss/�<v�Q�       �	`[�fc�A�0*

lossd�!;E���       �	�fc�A�0*

loss�!�:?T�       �	�fc�A�0*

loss�89P-       �	�!�fc�A�0*

loss($:J�V:       �	���fc�A�0*

loss4��=��#?       �		O�fc�A�0*

loss]�l<f�	       �	��fc�A�0*

loss:�:ށi6       �	�u�fc�A�0*

loss���9BF�       �	-	�fc�A�0*

loss&�;��ݔ       �	s��fc�A�0*

losssD�:��Aq       �	�J�fc�A�0*

loss2AH:�D��       �	1�fc�A�0*

loss<���       �	f��fc�A�0*

loss̗<��@       �	�'�fc�A�0*

lossw�=<Ԝ��       �	ʊfc�A�0*

loss3	;���       �	h�fc�A�0*

loss��;E��8       �	��fc�A�0*

loss�U�;�7��       �	O�fc�A�0*

loss���;ӭ       �	���fc�A�0*

lossib�<�Z^L       �	���fc�A�0*

loss?��:��D�       �	�7�fc�A�0*

loss�{�;��u�       �	���fc�A�0*

loss0c�9@��       �	ƨ�fc�A�0*

loss#�b=Q��y       �	N�fc�A�0*

lossճ:����       �	��fc�A�0*

loss�=p=@�m#       �	͕�fc�A�0*

loss�P;�1��       �	�{�fc�A�0*

lossɋM<q�d�       �	�fc�A�0*

loss�<�m�       �	Ѱ�fc�A�1*

loss-K:��]       �	1E�fc�A�1*

loss�$:�bT�       �	��fc�A�1*

loss�d�=oZk       �	닖fc�A�1*

loss);�<'b�0       �	?�fc�A�1*

loss&�<�)K       �	F��fc�A�1*

losst��;|xe�       �	VE�fc�A�1*

loss��%<�02�       �	��fc�A�1*

lossJnv<>z��       �	�|�fc�A�1*

lossEer:����       �	w�fc�A�1*

loss*bU;u�        �	欚fc�A�1*

lossv�:�/H       �	B�fc�A�1*

lossF.=~"��       �	ޛfc�A�1*

loss_=�}Z       �	s0�fc�A�1*

loss�"�:T]%       �	=֝fc�A�1*

loss`�4=P���       �	�s�fc�A�1*

loss�5;�5�       �	s�fc�A�1*

loss�{�=B���       �	X��fc�A�1*

loss`��;�.��       �	hA�fc�A�1*

loss	;��kw       �	'ڠfc�A�1*

loss4l�<����       �	\s�fc�A�1*

lossj�9T2]J       �	'�fc�A�1*

loss��9��1       �	oF�fc�A�1*

loss
XT;��N       �	:]�fc�A�1*

loss<==���       �	���fc�A�1*

lossF�:���       �	ᚥfc�A�1*

lossf��=�M�       �	�4�fc�A�1*

loss��Z=��       �	�ͦfc�A�1*

loss�;gu~4       �	bg�fc�A�1*

loss���;�D�       �	��fc�A�1*

loss=H�;�p       �	���fc�A�1*

loss�j�;0�#]       �	�<�fc�A�1*

loss.D=<��P�       �	[�fc�A�1*

loss�`�<t�!�       �	��fc�A�1*

lossq8�;Y��       �	%�fc�A�1*

loss@M;��"g       �	*ʫfc�A�1*

loss&�<_�       �	�d�fc�A�1*

loss
�k<�UE�       �	��fc�A�1*

loss���<Q��       �	ɭ�fc�A�1*

loss_Ҍ;#v�,       �	LS�fc�A�1*

loss�(k;�EƤ       �	���fc�A�1*

loss2ݱ:��u�       �	ޒ�fc�A�1*

loss�A�;@�A       �	�/�fc�A�1*

loss�MI;o�+       �	�ϰfc�A�1*

lossz��<��T�       �	�u�fc�A�1*

loss�.<�W*�       �	��fc�A�1*

loss��A<.[w       �	���fc�A�1*

loss�F�<� 0       �	Q�fc�A�1*

lossۋ<X	V�       �	)�fc�A�1*

lossaJ�;p�a�       �	�~�fc�A�1*

loss�Cs=9�!       �	I�fc�A�1*

loss���;-��       �	���fc�A�1*

loss/��:y'�       �	�<�fc�A�1*

loss��;�GgU       �	�Ҷfc�A�1*

loss�4:���       �	sg�fc�A�1*

loss��: ���       �	D��fc�A�1*

loss��U=�9�       �	t��fc�A�1*

loss\vG=�q�       �	c+�fc�A�1*

loss��:�2H�       �	"Ĺfc�A�1*

lossS�=YZ�       �	6[�fc�A�1*

loss�$�:�u8       �	��fc�A�1*

lossl�;Fq�       �	8��fc�A�1*

loss��V<3�~       �	� �fc�A�1*

loss�	<���y       �	U��fc�A�1*

loss�
�:+�b5       �	�U�fc�A�1*

loss/��:���       �	��fc�A�1*

loss���:9n�@       �	X��fc�A�1*

loss&�<����       �	RC�fc�A�1*

loss�u:}�+O       �	�ڿfc�A�1*

loss�M<���       �	�~�fc�A�1*

lossv!9A
a       �	�!�fc�A�1*

loss�ӗ:G5Kw       �	A��fc�A�1*

loss���:	�g�       �	�]�fc�A�1*

loss��=J��?       �	���fc�A�1*

lossf�<{��       �	ș�fc�A�1*

loss�Z;�(s       �	�6�fc�A�1*

loss�N�;�0�       �	|��fc�A�1*

loss���<�h�       �	�{�fc�A�1*

loss�<���       �	�fc�A�1*

lossZ4�9�Z�       �	Q��fc�A�1*

loss-�L<HZv�       �	�X�fc�A�1*

loss���<	
A�       �	Q��fc�A�1*

loss�"<o�-�       �	���fc�A�1*

lossx.�;�c�^       �	,�fc�A�1*

loss��l<�(�       �	X��fc�A�1*

loss��G:n���       �	�c�fc�A�1*

loss�:N=�l��       �	���fc�A�1*

lossq0
:۰�b       �	���fc�A�1*

losssj =�X�A       �	�2�fc�A�1*

lossi��<���       �	h��fc�A�1*

lossR��<Ķ|�       �	0g�fc�A�1*

lossj��;yu��       �	���fc�A�1*

loss	� =�Bw       �	���fc�A�1*

lossߨ�<	"�v       �	o.�fc�A�1*

loss�f�:�c�i       �	���fc�A�1*

loss���<�o       �	.Y�fc�A�1*

lossVu�;�7w�       �	���fc�A�1*

loss{;�:�y�       �	���fc�A�1*

loss���;�&MT       �	��fc�A�1*

loss8cv;��z       �	��fc�A�1*

lossM��:nW�Q       �	�V�fc�A�1*

loss�<�a��       �	6��fc�A�1*

loss�K;����       �	~��fc�A�1*

loss��:X�[       �	(�fc�A�1*

loss�G�:��       �	���fc�A�1*

loss��;�6'�       �	hZ�fc�A�1*

loss���=��D�       �	���fc�A�1*

loss��K<�{<�       �	���fc�A�1*

loss��<Kr�       �	o*�fc�A�1*

lossg;��W       �	���fc�A�1*

loss�ܬ<�6��       �	�c�fc�A�1*

loss<��<T�3       �	��fc�A�1*

loss���;��Z�       �	���fc�A�1*

lossF�;��       �	�A�fc�A�1*

loss�|<ׅ�       �	=��fc�A�1*

loss[��<��T�       �	n��fc�A�1*

loss���:^�       �	m�fc�A�1*

loss[�<����       �	��fc�A�1*

lossww;�I       �	R�fc�A�1*

loss�Ǆ:�8UM       �	
��fc�A�1*

loss͎�<�f�       �	���fc�A�1*

loss�Ŗ;  %.       �	�,�fc�A�1*

loss��<҈ʔ       �	���fc�A�1*

loss�c<�zY�       �	�^�fc�A�1*

loss�w�;A�B�       �		��fc�A�1*

loss[N{9d���       �	��fc�A�1*

loss�/�9]?�       �	�%�fc�A�1*

lossH��<�2b       �	��fc�A�1*

loss�o�;^n��       �	�c�fc�A�2*

loss�Y;�Y�       �	!�fc�A�2*

loss�t�;X	�l       �	8��fc�A�2*

loss13=^;�       �	�?�fc�A�2*

loss���<�؝�       �	#��fc�A�2*

loss�\<[I2       �	�o�fc�A�2*

loss��;��dD       �	��fc�A�2*

loss`��;��E1       �	#��fc�A�2*

loss� �=��9�       �	�6�fc�A�2*

loss`�<!0       �	���fc�A�2*

loss��;Z7y�       �	�^�fc�A�2*

loss1�;g2�       �	��fc�A�2*

loss��=��_}       �	���fc�A�2*

loss;<���       �	i;�fc�A�2*

loss�P(=~$��       �	��fc�A�2*

lossW��;�,%�       �	ip�fc�A�2*

loss�f�;���E       �	��fc�A�2*

loss��<����       �	ʨ�fc�A�2*

loss/�r;�HE�       �	cD�fc�A�2*

loss��;}1��       �	@��fc�A�2*

loss�n<q̜�       �	�o�fc�A�2*

lossZ�:~�       �	_�fc�A�2*

loss��<s��:       �	���fc�A�2*

loss1<~��1       �	�M�fc�A�2*

loss�F�<��?       �	���fc�A�2*

loss��;š)�       �	��fc�A�2*

lossN�
=��x       �	�1�fc�A�2*

loss��<<�L�!       �	��fc�A�2*

loss 
�;�       �	�[�fc�A�2*

loss�+'<`�,       �	���fc�A�2*

loss(�<�Y�7       �	���fc�A�2*

loss�p�<>�v       �	�$�fc�A�2*

loss6�;
�p       �	��fc�A�2*

lossHh;�;K       �	�T�fc�A�2*

lossv��;��L�       �	���fc�A�2*

loss�uD<�cn�       �	~�fc�A�2*

loss�I;�1�       �	��fc�A�2*

loss� �;�@�       �	a��fc�A�2*

loss�O<p���       �	=�fc�A�2*

loss�fV:�r}e       �	���fc�A�2*

loss.��:�9*[       �	�|�fc�A�2*

lossS=� �       �	n�fc�A�2*

loss�v>=�^��       �	S��fc�A�2*

loss;�;(�       �	���fc�A�2*

loss��"<*�s{       �	ge fc�A�2*

loss�vS:j��       �	4.fc�A�2*

loss��<n�C�       �	&�fc�A�2*

lossZ:J;�I��       �	�vfc�A�2*

loss7y<$5G       �	�fc�A�2*

loss�:J��       �	�fc�A�2*

lossTR/;��7       �	�Xfc�A�2*

loss;��<���       �	�fc�A�2*

loss�{m:θ��       �	��fc�A�2*

loss�k0=��#       �	�5fc�A�2*

loss�� <���0       �	 �fc�A�2*

loss*L<�J'       �	�qfc�A�2*

lossuz;u��       �	ofc�A�2*

lossd�;��       �	�fc�A�2*

loss��9�
       �	�Q	fc�A�2*

lossh&$;�p�       �	��	fc�A�2*

loss�ǿ:��]�       �	b�
fc�A�2*

loss�3::�0�       �	x*fc�A�2*

loss�q;���       �	��fc�A�2*

loss
k�;?�8       �	�gfc�A�2*

loss�%/>l�7       �	�fc�A�2*

loss ��<\�B�       �	H�fc�A�2*

loss���:]�       �	�Afc�A�2*

loss�P�<T[��       �	^�fc�A�2*

loss���9z	�a       �	�vfc�A�2*

lossz�<��V�       �	fc�A�2*

loss��;��/�       �	��fc�A�2*

loss�2h=���       �	wNfc�A�2*

loss0�<ص�x       �	��fc�A�2*

loss��:�h�       �	�fc�A�2*

lossCT�;MQD       �	=*fc�A�2*

loss��<��+�       �	�fc�A�2*

loss`|�;�s��       �	m�fc�A�2*

lossV�;Θ       �	�7fc�A�2*

loss%<P<~�c~       �	�fc�A�2*

loss��<��z�       �	�sfc�A�2*

loss!s=�Pݶ       �	�Cfc�A�2*

loss��=kˈ�       �	-�fc�A�2*

loss
"<�Ɉ�       �	�fc�A�2*

loss�=
[\       �	3fc�A�2*

loss�i%<�a�Q       �	��fc�A�2*

loss���;��w�       �	0hfc�A�2*

loss���:�u��       �	fc�A�2*

lossi�Z:���       �	!�fc�A�2*

loss� 6<gJ�       �	�Rfc�A�2*

lossC�:��N�       �	�fc�A�2*

loss/ =c��       �	��fc�A�2*

loss�4=Q|�}       �	7fc�A�2*

loss��=�Ճd       �	�fc�A�2*

loss��<��&�       �	
�fc�A�2*

loss8��;�g��       �	�) fc�A�2*

loss�/�:���l       �	�� fc�A�2*

loss{�;ԇ��       �	�m!fc�A�2*

loss_w�;�b��       �	 "fc�A�2*

loss&�;~\�       �	�"fc�A�2*

loss��;�Lm>       �	X7#fc�A�2*

loss͓�9?�ZE       �	`%fc�A�2*

loss���;6�{       �	�&fc�A�2*

loss�e�:23�       �	��&fc�A�2*

lossv�<���       �	�G'fc�A�2*

lossa`<w�o�       �	�(fc�A�2*

loss��<��xT       �	��(fc�A�2*

loss@q�:���"       �	�\)fc�A�2*

lossre�<r[?8       �	�*fc�A�2*

loss髓<<ptk       �	��*fc�A�2*

loss�?	<��T       �	#N+fc�A�2*

loss�];k�{P       �	��+fc�A�2*

lossl��9�J��       �	Ŏ,fc�A�2*

loss��<����       �	3-fc�A�2*

loss���:�H�       �	��-fc�A�2*

lossQ�m<	gT�       �	�p.fc�A�2*

lossq>�;z8�       �	�,/fc�A�2*

loss�J�:w��'       �	o�/fc�A�2*

loss�e<�c�/       �	o0fc�A�2*

lossב]:9��#       �	'1fc�A�2*

loss�S�;a/+G       �	��1fc�A�2*

lossE��;fS/       �	�P2fc�A�2*

lossD��;<��w       �	��2fc�A�2*

loss���:u��       �	M�3fc�A�2*

loss,τ:�:        �	��4fc�A�2*

loss���:�~A       �	�(5fc�A�2*

lossi��;���Z       �	��5fc�A�2*

loss��4:)�Y�       �	p6fc�A�2*

lossA�E;���]       �	�	7fc�A�2*

lossn��:���       �	�7fc�A�3*

loss={�:�/��       �	P8fc�A�3*

loss
(:�H       �	�8fc�A�3*

loss�^�9�a�       �	��9fc�A�3*

loss�@E;	Ӥ�       �	�R:fc�A�3*

lossN&E;���       �	��:fc�A�3*

lossx,;B~{       �	��;fc�A�3*

loss��:��5j       �	X�<fc�A�3*

loss�.18q�,       �	k�=fc�A�3*

lossE^�:�2f�       �	�>fc�A�3*

lossc�b:����       �	O@fc�A�3*

loss�BJ:F��       �	��@fc�A�3*

loss_~�<�C�^       �	��Afc�A�3*

loss'u9(�M       �	�<Bfc�A�3*

lossf:27���/       �	��Bfc�A�3*

loss�85�gw/       �	'�Cfc�A�3*

loss��9���q       �	TDfc�A�3*

loss��<����       �	��Dfc�A�3*

loss��;txG�       �	]Efc�A�3*

loss�=;?��       �	�Ffc�A�3*

lossA�<�{H       �	3�Ffc�A�3*

loss���:amo       �	�JGfc�A�3*

lossDo�8�':       �	!�Gfc�A�3*

loss���=�oxN       �	b�Hfc�A�3*

lossa�$=GE       �	HIfc�A�3*

loss#��:΅�       �	V�Ifc�A�3*

loss�ϟ8!�c}       �	�LJfc�A�3*

loss��<�-n�       �	2�Jfc�A�3*

losso7<��5       �	FyKfc�A�3*

loss�]:�+�[       �	�Lfc�A�3*

loss>;�\�       �	��Lfc�A�3*

lossn�:��       �	bKMfc�A�3*

loss6�;��ځ       �	��Mfc�A�3*

loss���=a&��       �	�Nfc�A�3*

loss3�(<�Qd�       �	�!Ofc�A�3*

lossF��;84��       �	��Ofc�A�3*

loss$=c�8       �	�bPfc�A�3*

loss��;�-�;       �	��Pfc�A�3*

loss$��<�o��       �	G�Qfc�A�3*

lossl�=`�       �	�&Rfc�A�3*

lossVK�;5�c&       �	@�Rfc�A�3*

loss�'4<��?       �	�ZSfc�A�3*

loss�>;���       �	�Sfc�A�3*

loss���;��˛       �	��Tfc�A�3*

loss�� <�S�       �	2:Ufc�A�3*

loss̍:�X�S       �	x�Ufc�A�3*

loss���9*��       �	�nVfc�A�3*

lossU�:_y��       �	VWfc�A�3*

lossL�=W{CC       �	�Wfc�A�3*

loss�aA=��       �	�VXfc�A�3*

loss �,="��R       �	 Yfc�A�3*

loss�dg<�d��       �	ǼYfc�A�3*

loss|%�90
E       �	6\Zfc�A�3*

loss3�2:�p9       �	m�Zfc�A�3*

loss�+:�7�I       �	��[fc�A�3*

loss�%�:MU�M       �	�7\fc�A�3*

lossaX:�I�       �	f�\fc�A�3*

loss���8�<�       �	��]fc�A�3*

lossȠD:Y�       �	�^fc�A�3*

loss`;�@`&       �	�,_fc�A�3*

loss��Q<��Q�       �	��_fc�A�3*

loss���;e�m       �	�b`fc�A�3*

loss*l�98�z�       �	�`fc�A�3*

loss&/x:t�{       �	9�afc�A�3*

loss�\	:�}�       �	0bfc�A�3*

loss�W;$��       �	��bfc�A�3*

lossT�8=�ˏ<       �	�dcfc�A�3*

loss�q�;�.:�       �	 dfc�A�3*

loss�p�=h7�       �	�efc�A�3*

loss���8pFx       �	z�efc�A�3*

loss��;+�       �	��ffc�A�3*

loss�A:��]�       �	M.gfc�A�3*

losso[�;�Gc       �	G:ifc�A�3*

lossa+y:�5�       �	�3�fc�A�3*

lossx�<7��       �	�ςfc�A�3*

loss�X<Y�C�       �	#h�fc�A�3*

lossb��<��ɂ       �	&��fc�A�3*

loss{�<��t       �	���fc�A�3*

loss�L<pk       �	�+�fc�A�3*

loss��<c��       �	�ͅfc�A�3*

loss�4<�n       �	�f�fc�A�3*

lossT��;�Tv       �	��fc�A�3*

loss{a�<w6
       �	ࠇfc�A�3*

lossG:P�q       �	�=�fc�A�3*

lossp�<�T       �	nڈfc�A�3*

loss�Hq;�ف>       �	-w�fc�A�3*

loss��9:>��       �	��fc�A�3*

lossy$=G� z       �	~��fc�A�3*

loss���:� 1G       �	�>�fc�A�3*

loss��8D�F       �	��fc�A�3*

loss��z;�-��       �	���fc�A�3*

loss7��<D~�       �	^0�fc�A�3*

loss�&<d�@�       �	�ƍfc�A�3*

loss���:�s��       �	_^�fc�A�3*

lossV,r=7�       �	D��fc�A�3*

losso�:so��       �	��fc�A�3*

loss3U=
�        �	�+�fc�A�3*

loss�a�:�"       �	�̐fc�A�3*

lossͨ;��4       �	g�fc�A�3*

lossS�7;@$�       �	;��fc�A�3*

loss�=8��7       �	���fc�A�3*

loss���<�Ӫ       �	�E�fc�A�3*

loss��M;p�       �	�ܓfc�A�3*

loss6"=Ez�       �	_~�fc�A�3*

loss���;���       �	h �fc�A�3*

lossNH;G(�A       �	Z��fc�A�3*

loss(�<FM�)       �	�Z�fc�A�3*

loss�X�9^��       �	���fc�A�3*

loss�hz;��}       �	��fc�A�3*

lossd�n:v���       �	�4�fc�A�3*

loss�=���       �	Ҙfc�A�3*

loss	��;��g       �	�k�fc�A�3*

loss���;<I��       �	��fc�A�3*

loss!��:$�g�       �	ٗ�fc�A�3*

lossCp;lD$�       �	�0�fc�A�3*

loss���8�z��       �	țfc�A�3*

lossIaH<�r՛       �	Qj�fc�A�3*

loss9<2�dK       �	��fc�A�3*

lossWK�<��       �	���fc�A�3*

loss�0f;1�H       �	LĞfc�A�3*

loss�?3=}/5�       �	�&�fc�A�3*

loss*��:�j�_       �	�Ǡfc�A�3*

loss�:qGo�       �	�f�fc�A�3*

loss\�T:+��5       �	u�fc�A�3*

loss��C=^��       �	ס�fc�A�3*

loss�3;��od       �	?6�fc�A�3*

loss	G�=�^M       �	���fc�A�3*

loss(�9���%       �	���fc�A�3*

loss��;�@�i       �	�5�fc�A�4*

loss�z:!���       �	JΥfc�A�4*

lossOv�9�CFt       �	fi�fc�A�4*

loss��;�JV       �		�fc�A�4*

loss��<��       �	�\�fc�A�4*

loss��=��Y�       �	��fc�A�4*

lossn�;�6�       �	˜�fc�A�4*

loss�e=:S�gl       �	#תfc�A�4*

lossÿ�;9 s�       �	��fc�A�4*

loss���;Y�X       �	]4�fc�A�4*

loss�/k;�:,�       �	BϬfc�A�4*

loss�0�:_���       �	�r�fc�A�4*

lossO{�;4��D       �	�]�fc�A�4*

loss2��;�'��       �	���fc�A�4*

loss 45;�<�       �	�[�fc�A�4*

loss��G;�h�Q       �	��fc�A�4*

loss���:!���       �	���fc�A�4*

losst�];��(�       �	�Ҳfc�A�4*

lossm��<t��^       �	d��fc�A�4*

lossN5�:dW�       �	oH�fc�A�4*

loss���9�$܇       �	��fc�A�4*

loss��-;`�xb       �	%#�fc�A�4*

lossQL;�ʕ       �	H��fc�A�4*

loss\�;K�c�       �	p[�fc�A�4*

loss@e�;��u\       �	���fc�A�4*

lossݠ�<yj	&       �	���fc�A�4*

loss�@<��
       �	�*�fc�A�4*

loss�u<��       �	�̹fc�A�4*

loss�	;>+[�       �	�fc�A�4*

loss��;T���       �	�P�fc�A�4*

loss���:)P�       �	���fc�A�4*

loss�R<�{��       �	��fc�A�4*

loss�M$<��       �	�7�fc�A�4*

loss�*�:���m       �	�ҽfc�A�4*

loss[�};�,�       �	m�fc�A�4*

loss)N;S!��       �	�!�fc�A�4*

loss6p@<<i�2       �	���fc�A�4*

loss}�";:v��       �	j�fc�A�4*

lossH�s=g3�	       �	��fc�A�4*

loss�Q�<L�}T       �	3��fc�A�4*

loss �#9���       �	���fc�A�4*

loss��R;�O��       �	nm�fc�A�4*

lossrN�9�N       �	0��fc�A�4*

loss�~�<W��       �	�9�fc�A�4*

loss�N:V1*�       �	A��fc�A�4*

loss 8;��g.       �	b��fc�A�4*

lossɓd9����       �	�\�fc�A�4*

loss��=�y�p       �	9%�fc�A�4*

lossW�j;ז�U       �	�~�fc�A�4*

loss��=<y��       �	[��fc�A�4*

loss�q�:��rL       �	aO�fc�A�4*

loss�� <�!�7       �	���fc�A�4*

lossv�.:7�D�       �	��fc�A�4*

loss(�1;ˑu�       �	���fc�A�4*

loss��;=��       �	z��fc�A�4*

loss�'�<�'��       �	��fc�A�4*

loss��;��       �	�>�fc�A�4*

lossm�V<A�       �	9�fc�A�4*

loss�Y�8Rpc       �	���fc�A�4*

loss��;�9�H       �	���fc�A�4*

lossv=�?'�       �	Xu�fc�A�4*

losst=	;��ns       �	Xp�fc�A�4*

lossx�=�u<�       �	qu�fc�A�4*

lossۨx:f�VA       �	
�fc�A�4*

loss�9�9���       �	ʩ�fc�A�4*

loss8�;f��       �	l��fc�A�4*

loss���;��s�       �	�(�fc�A�4*

loss���:�d�       �	�>�fc�A�4*

loss� �;}O�       �	���fc�A�4*

loss���:P8B       �	zm�fc�A�4*

lossn�<��t       �	6�fc�A�4*

lossX��;���       �	ʤ�fc�A�4*

loss\l<e�	�       �	�8�fc�A�4*

loss�Z"=�ОT       �	#�fc�A�4*

loss�:�;�̡       �	���fc�A�4*

loss m�:ď�Z       �	���fc�A�4*

loss��:��BX       �	=F�fc�A�4*

loss2>4;�س�       �	���fc�A�4*

lossZJa:,�"�       �	ƅ�fc�A�4*

loss�M<pg��       �	6!�fc�A�4*

loss��}<=A�       �	Ͼ�fc�A�4*

loss�#�;=��       �	h]�fc�A�4*

loss~v:����       �	���fc�A�4*

loss��;<XO��       �	��fc�A�4*

loss���8�>�       �	�G�fc�A�4*

lossn�;�go       �	���fc�A�4*

loss���<���       �	���fc�A�4*

loss$G;d'�       �	���fc�A�4*

loss�U:�I�'       �	0b�fc�A�4*

loss ڗ<篴#       �	[`�fc�A�4*

lossZ��:�|�       �	��fc�A�4*

loss��.<ϬB�       �	q��fc�A�4*

loss*�s8jK�        �	G�fc�A�4*

losso+;;�n       �	-��fc�A�4*

loss�{(;����       �	��fc�A�4*

loss��7:19)
       �	1�fc�A�4*

loss�#�<XhU       �	��fc�A�4*

lossű�< %d       �	�r�fc�A�4*

loss�zO=vFhx       �	��fc�A�4*

loss/z<�J�p       �	���fc�A�4*

losst�<$�QN       �	
K�fc�A�4*

lossQp;:j�       �	3��fc�A�4*

loss	�_<��b       �	G8�fc�A�4*

loss^�=��       �	���fc�A�4*

loss�<h<���       �	Vd�fc�A�4*

loss";���       �	��fc�A�4*

loss�60;����       �	e��fc�A�4*

loss ��<|�       �	PV�fc�A�4*

loss�5�:;T�       �	B��fc�A�4*

loss�9�:~�R0       �	���fc�A�4*

loss�v�;��Nb       �	�'�fc�A�4*

loss)dq9n_ݶ       �	��fc�A�4*

loss�=���c       �	mX�fc�A�4*

lossA;n��I       �	�;�fc�A�4*

loss�;"��       �	
J�fc�A�4*

lossi�82�*@       �	���fc�A�4*

loss��V;�jA�       �	���fc�A�4*

lossl_;���       �	#2�fc�A�4*

loss���<�]��       �	���fc�A�4*

loss��9����       �	���fc�A�4*

loss�)�<Y0�       �	Y�fc�A�4*

lossR<�::�c       �	s�fc�A�4*

loss�ZS<��6q       �	l��fc�A�4*

loss,��;c�B       �	[y�fc�A�4*

lossK8�:4��       �	� fc�A�4*

loss�Of;Ƕ�?       �	�3fc�A�4*

loss��<�#�e       �	{�fc�A�4*

loss,�p:9�X;       �	��fc�A�4*

losslO;W�       �	�2fc�A�5*

losss�<	��G       �	�Mfc�A�5*

loss��9��       �	=fc�A�5*

lossmx�;(ͯ       �	��fc�A�5*

loss6=9Ԯ.       �	*�fc�A�5*

loss�8B:�@�\       �	5]fc�A�5*

loss��;-�       �	Ukfc�A�5*

loss!�;t�3e       �	�?	fc�A�5*

loss�5�;�N       �	��	fc�A�5*

loss6'�<[8�       �	�m
fc�A�5*

loss�1;:��6       �	fc�A�5*

lossJT97(�P       �	Q�fc�A�5*

loss4t�;z�Rc       �	�=fc�A�5*

loss.�4=�r�p       �	��fc�A�5*

lossWo<6X       �	�pfc�A�5*

lossѶ=�6��       �	�fc�A�5*

lossM�_:є��       �	8�fc�A�5*

loss&!;DV�v       �	�;fc�A�5*

loss�l<�0+       �	�fc�A�5*

loss�0:bb��       �	*ofc�A�5*

loss0��<R�
       �	�fc�A�5*

loss��=���       �	{�fc�A�5*

loss��:?��       �	)<fc�A�5*

loss&�<�8!}       �	��fc�A�5*

loss׹:�#i�       �	�pfc�A�5*

lossm:�9
UU�       �	1	fc�A�5*

loss$�<٣�        �	��fc�A�5*

loss�eX:J* �       �		2fc�A�5*

lossf=q:I�       �	��fc�A�5*

loss��9���       �	�ofc�A�5*

loss;��:�^�C       �	�fc�A�5*

lossx��:�`v       �	ŭfc�A�5*

loss��;�(��       �	�Efc�A�5*

lossx��<�Mč       �	��fc�A�5*

lossh `=�`x�       �	�}fc�A�5*

lossӪ�<�I�       �	�fc�A�5*

loss��?:Ӹ�       �	�fc�A�5*

loss0�<�{�       �	�Mfc�A�5*

loss��=�E��       �	N�fc�A�5*

loss��o9q7*       �	q�fc�A�5*

lossF��<\���       �	n3fc�A�5*

loss�9�^�       �	I�fc�A�5*

loss�<	ý�       �	Ttfc�A�5*

loss�9:���       �	Xfc�A�5*

loss��:����       �	#�fc�A�5*

lossz�;1(       �	�g fc�A�5*

lossrC:�:�v       �	!fc�A�5*

loss\<���       �	ߥ!fc�A�5*

loss&o-;62��       �	S?"fc�A�5*

lossT�=x��       �	M�"fc�A�5*

loss���;�5�       �	��$fc�A�5*

loss2D�;2R       �	�c%fc�A�5*

loss���;�Q�	       �	�&fc�A�5*

loss(m,;�w
       �	��&fc�A�5*

loss�t�;@$�       �	�D'fc�A�5*

loss�>1=��       �	$�)fc�A�5*

loss8�&;�       �	+fc�A�5*

lossti�;��>j       �	T�+fc�A�5*

loss���<e��       �	cC,fc�A�5*

loss��}9�9�?       �	��,fc�A�5*

loss���:Ԓv       �	��-fc�A�5*

loss-Y�;���       �	2.fc�A�5*

loss��<�Ʃ�       �	��.fc�A�5*

loss��-:���       �	�d/fc�A�5*

loss�(�:�F�       �	�0fc�A�5*

lossa��<3�Z       �	I�0fc�A�5*

loss;�=3h�3       �	�:1fc�A�5*

loss���=�zX!       �	�1fc�A�5*

lossѝ=���       �	��2fc�A�5*

loss�N>:��       �	/3fc�A�5*

loss�N+9��d�       �	ڪ3fc�A�5*

loss\�9~,       �	D4fc�A�5*

loss�MJ9�/?�       �	��4fc�A�5*

loss��f;"Y�       �	�s5fc�A�5*

loss�:�M�{       �	6fc�A�5*

lossj�p;�s5       �	��6fc�A�5*

lossf}1=�ޡ       �	P77fc�A�5*

loss*	%=2��       �	��7fc�A�5*

loss��:'¥       �	g_8fc�A�5*

lossH��<���V       �	��8fc�A�5*

loss��x:�|?S       �	��9fc�A�5*

loss3� <y���       �	9%:fc�A�5*

lossV|P;B�~       �	��:fc�A�5*

loss��<#t�[       �	.T;fc�A�5*

loss�6�<�^�       �	t�;fc�A�5*

loss4}=���p       �	`�<fc�A�5*

lossN;P���       �	s,=fc�A�5*

loss[X;=� 1.       �	��=fc�A�5*

lossh.�9U)[�       �	�]>fc�A�5*

loss��;k%�       �	�>fc�A�5*

loss ��::�Ӕ       �	L�?fc�A�5*

loss�,<E6�       �	�V@fc�A�5*

loss��<z�z^       �	�@fc�A�5*

loss1"<[�z�       �	P�Afc�A�5*

loss�L9�]�d       �	T�Bfc�A�5*

loss.�.;
i�H       �	�GCfc�A�5*

loss�dM=b���       �	Dfc�A�5*

loss?�9{&h       �	ؚDfc�A�5*

loss��t<ư�       �	��Efc�A�5*

loss� +;ݮ�       �	��Ffc�A�5*

loss�e�:�(w       �	-AHfc�A�5*

lossNP<���4       �	�;Ifc�A�5*

loss'��:�Q5       �	��Ifc�A�5*

loss`�J=�%�       �	�rJfc�A�5*

loss\T9
��       �	5Kfc�A�5*

loss]D�;�8�       �	
�Kfc�A�5*

loss	��:�_��       �	YLfc�A�5*

loss��91�i�       �	
�Lfc�A�5*

loss�#�;B��       �	c�Mfc�A�5*

loss?�1;;�<n       �	�3Nfc�A�5*

loss��:���       �	'�Nfc�A�5*

loss���<���       �	υOfc�A�5*

lossJۈ<�*l�       �	�!Pfc�A�5*

loss5�<�l�       �	9�Pfc�A�5*

loss�$�:&��       �	�IQfc�A�5*

lossLС;�9�w       �	�Qfc�A�5*

loss��q=��b       �	9Rfc�A�5*

loss�~;���       �	�Sfc�A�5*

lossI�;�~��       �	�Sfc�A�5*

loss�|:J        �	�MTfc�A�5*

loss��<3Vk       �	H�Tfc�A�5*

loss�'�<R4C�       �	p{Ufc�A�5*

loss-G6=�G��       �	�Vfc�A�5*

loss�$=K�Ta       �	r�Vfc�A�5*

loss�	;\q       �	a4Wfc�A�5*

loss�!<M��       �	��Wfc�A�5*

loss�u�<@"K       �	kbXfc�A�5*

lossFE�:����       �	Z�Xfc�A�5*

loss{�i:��=�       �	��Yfc�A�6*

loss�}Y;�r�       �	Zfc�A�6*

loss���:�_��       �	6�Zfc�A�6*

loss�R�<6K��       �	�F[fc�A�6*

lossXǊ;%((       �	@�[fc�A�6*

loss�qd:����       �	�s\fc�A�6*

lossav�;�Fbl       �	-]fc�A�6*

loss��;�m�       �	�]fc�A�6*

loss�\;���       �	կ_fc�A�6*

lossR,l;e���       �	�C`fc�A�6*

loss�A�;&�       �	��`fc�A�6*

loss�n9<F\��       �	�{afc�A�6*

lossr�x<#ld       �	bfc�A�6*

loss|!�;L��       �	1�bfc�A�6*

loss���;��/       �	�Qcfc�A�6*

loss��Q;%<{�       �	>�cfc�A�6*

loss�Ɋ:Q�:�       �	�dfc�A�6*

loss;3�;\��       �	efc�A�6*

loss��=T�P�       �	_�efc�A�6*

loss�;����       �	Mffc�A�6*

loss$;�3��       �	B�ffc�A�6*

loss�j28ޔl       �	��gfc�A�6*

loss\��;�<m!       �	B%hfc�A�6*

loss��<8��       �	{�hfc�A�6*

loss�<�a0+       �	�~ifc�A�6*

lossN�/=.��r       �	�jfc�A�6*

loss%��;FO�c       �	sIkfc�A�6*

loss.�s<�B3       �	�kfc�A�6*

loss��9�=?�       �	�2mfc�A�6*

lossR.�;���O       �	�ofc�A�6*

loss&�e8��       �	�Fpfc�A�6*

loss���;�!L�       �	X�qfc�A�6*

loss>8<M�v       �	E.rfc�A�6*

loss�Ͷ<����       �	�sfc�A�6*

lossR��;zP�       �	
�sfc�A�6*

loss��<�&D       �	��tfc�A�6*

losst��<��)       �	�ufc�A�6*

lossvf�:z#j�       �	F�vfc�A�6*

loss�=�;��       �	�hxfc�A�6*

loss�Xd9ZY��       �	�yfc�A�6*

loss}nM;���!       �	ݶyfc�A�6*

loss�+�;��d       �	v�zfc�A�6*

loss��<�S�h       �	��{fc�A�6*

lossJ�:��1�       �	`Y|fc�A�6*

loss��:!Y��       �	2}fc�A�6*

lossV�p<"�E       �	T�}fc�A�6*

loss��H;#\��       �	�D~fc�A�6*

loss격8�\�       �	�rfc�A�6*

loss��;E��       �	f�fc�A�6*

loss5��<+q�       �	��fc�A�6*

loss���< ��       �	Ѳ�fc�A�6*

lossh
=�+R�       �	�_�fc�A�6*

loss��];��*_       �	̴�fc�A�6*

loss�<���       �	@N�fc�A�6*

loss�в=���       �	��fc�A�6*

loss�{<�U�(       �	D��fc�A�6*

loss�:u9O]       �	�*�fc�A�6*

lossZ�W;U#"�       �	HĆfc�A�6*

loss(�<N+��       �	6Y�fc�A�6*

loss��:<��       �	��fc�A�6*

loss1[�;�T�w       �	��fc�A�6*

loss ��;�y�/       �	�#�fc�A�6*

losst)�8u���       �	Q��fc�A�6*

lossb:���       �	�Z�fc�A�6*

loss�E{;���       �	���fc�A�6*

loss-:��       �	���fc�A�6*

loss��:�B�)       �	c*�fc�A�6*

loss3�;s�Y\       �	�ˌfc�A�6*

lossR�;�pq       �	Di�fc�A�6*

loss�܋<�f�       �	��fc�A�6*

loss.��:��       �	蠎fc�A�6*

lossc{�; �,�       �	F@�fc�A�6*

loss��<\t*       �	O�fc�A�6*

lossqI0:�7s       �	���fc�A�6*

loss�^�<H�Ou       �	��fc�A�6*

lossOf=e�       �	���fc�A�6*

loss_c�<�,       �	�T�fc�A�6*

loss,��:<0�Z       �	˻�fc�A�6*

loss�;/r�T       �	�R�fc�A�6*

loss�<����       �	e��fc�A�6*

loss�[�;���       �	N��fc�A�6*

loss�R�=&З�       �	gG�fc�A�6*

loss/�l;E:��       �	X�fc�A�6*

loss�|9�2�        �	V��fc�A�6*

loss�O�=�       �	,J�fc�A�6*

loss�+\:�O��       �	f��fc�A�6*

lossT�94�{       �	䠙fc�A�6*

loss��9���       �	IM�fc�A�6*

loss��.;�yd�       �	���fc�A�6*

lossxQ�:N{�       �	 ��fc�A�6*

loss~~�: G7       �	#��fc�A�6*

loss��u;u'�{       �	斝fc�A�6*

lossnϻ9h���       �	�1�fc�A�6*

loss,T�<1	��       �	ٞfc�A�6*

lossO��<���       �	A�fc�A�6*

loss�z�<�)��       �	F��fc�A�6*

lossk
�:�/�       �	K�fc�A�6*

loss1;��#       �	��fc�A�6*

loss�<�;2�}�       �	*��fc�A�6*

lossP'<�~��       �	8/�fc�A�6*

loss���=,�?�       �	�ѣfc�A�6*

loss�]�;"�       �	�q�fc�A�6*

loss̘V<o:)'       �	"�fc�A�6*

loss�<G-H       �	��fc�A�6*

lossZ��;��j�       �	u[�fc�A�6*

loss|U�;��QT       �	���fc�A�6*

loss�q�;�φa       �	ٙ�fc�A�6*

loss�J�:���M       �	�6�fc�A�6*

loss2�-<�;[�       �	�ިfc�A�6*

loss��9\p��       �	Wv�fc�A�6*

lossF�<���       �	��fc�A�6*

loss�=W9N��       �	��fc�A�6*

loss�K�<����       �	�H�fc�A�6*

loss��;�{q       �	�ܫfc�A�6*

loss��;����       �	Pq�fc�A�6*

lossJp;:��'       �	Z�fc�A�6*

loss�.�;ഈ@       �	���fc�A�6*

lossh7�:d��       �	T:�fc�A�6*

loss���;չ�
       �	xѮfc�A�6*

loss�>:�S�       �	�j�fc�A�6*

loss��;���       �	K�fc�A�6*

loss!\r<Z        �	ᛰfc�A�6*

loss�q�;��       �	I0�fc�A�6*

loss�<[	�&       �	Ǳfc�A�6*

loss?Ɲ;K��       �	�a�fc�A�6*

lossZ�<;/Cj       �	���fc�A�6*

lossh�T<��$8       �	{��fc�A�6*

loss�˂9���       �	�9�fc�A�6*

loss��/<���?       �	e�fc�A�7*

loss��3;�%�       �	&��fc�A�7*

loss�:=n7l       �	�E�fc�A�7*

loss�U<O��       �	>�fc�A�7*

loss:��;��

       �	ڌ�fc�A�7*

loss\�:*bI�       �	#�fc�A�7*

loss�5<��CS       �	���fc�A�7*

lossQ�=��~       �	�[�fc�A�7*

lossz��;N`�l       �	D��fc�A�7*

loss��<$R��       �	���fc�A�7*

loss���=� Y,       �	=�fc�A�7*

loss*.;��L�       �	\�fc�A�7*

loss��<-΢R       �	'��fc�A�7*

loss�׹;�'"       �	A-�fc�A�7*

loss�N�9��ٟ       �	�нfc�A�7*

loss�d <�/A�       �	g�fc�A�7*

loss���=��W       �	��fc�A�7*

loss��F;����       �	%��fc�A�7*

loss��<p���       �	'N�fc�A�7*

loss�IC<���       �	4��fc�A�7*

lossV�=�iY�       �	���fc�A�7*

loss?C�:��(       �	+/�fc�A�7*

loss���:F]>$       �	���fc�A�7*

loss�V(:��:       �	y�fc�A�7*

lossh�C<L�       �	��fc�A�7*

lossr�8��^       �	��fc�A�7*

lossװ�9B���       �	�V�fc�A�7*

loss�2I:/��       �	<��fc�A�7*

loss*;�P��       �	���fc�A�7*

lossڧc<Н_�       �	2�fc�A�7*

loss,�<C���       �	>��fc�A�7*

loss��<(���       �	/j�fc�A�7*

loss��<t���       �	!��fc�A�7*

loss�C�9ا�       �	���fc�A�7*

loss�Ʉ;f~�       �	tB�fc�A�7*

loss; c<�sQ       �	���fc�A�7*

loss�z�9       �	��fc�A�7*

losslg�<����       �	K"�fc�A�7*

loss���9\0�       �	Ӿ�fc�A�7*

loss(u+<���       �	¾�fc�A�7*

loss�3�:�_��       �	�Y�fc�A�7*

loss�W�;}|�       �	��fc�A�7*

loss��v<����       �	��fc�A�7*

loss�{9�?rn       �	RC�fc�A�7*

loss�T;"_�       �	��fc�A�7*

loss��<
�:       �	`��fc�A�7*

loss�;ϣ�       �	8K�fc�A�7*

lossȻ�:!C��       �	���fc�A�7*

lossh�P=��-|       �	v��fc�A�7*

lossVB <CHOH       �	8-�fc�A�7*

loss4!�=��]       �	��fc�A�7*

lossmE�;���       �	
g�fc�A�7*

loss^0�=9/L       �	Y��fc�A�7*

lossF�9�D�       �	R��fc�A�7*

loss��!<t�S        �	�3�fc�A�7*

loss
�:H-�_       �	���fc�A�7*

loss�d�;���       �	`�fc�A�7*

loss�
�<.��       �	���fc�A�7*

loss��G<       �	���fc�A�7*

lossQ��<��Y       �	m�fc�A�7*

lossZ�\9�B�       �	؀�fc�A�7*

loss�I=�P�       �	�*�fc�A�7*

lossR-s:&	�       �	&��fc�A�7*

loss��;�t;
       �	Z�fc�A�7*

loss�=#��       �	���fc�A�7*

loss��<�>       �	���fc�A�7*

loss�!:,q�       �	�-�fc�A�7*

lossJ�<��0h       �	y��fc�A�7*

loss;Z>:G�]�       �	Re�fc�A�7*

loss�-�=p伏       �	���fc�A�7*

loss8�`;	q�       �	��fc�A�7*

loss(��:B^�_       �	�0�fc�A�7*

lossNq�;[�F       �	T��fc�A�7*

loss�cU;�)�       �	_�fc�A�7*

loss��:���       �	���fc�A�7*

lossf�=C�1       �	���fc�A�7*

losse~<NW�?       �	]p�fc�A�7*

loss�}�9��U       �	M�fc�A�7*

lossx89f �]       �	Ů�fc�A�7*

loss�gr;�צx       �	�I�fc�A�7*

lossf��<�4�       �	3��fc�A�7*

loss�L;[U�       �	w�fc�A�7*

loss��:�n�%       �	��fc�A�7*

loss$��9���       �		8�fc�A�7*

loss�X�<m��       �	��fc�A�7*

loss�E�<U.@�       �	�c�fc�A�7*

loss���<,q�       �	��fc�A�7*

lossh�<�dx�       �	O��fc�A�7*

loss��d<{N��       �	a�fc�A�7*

lossG�=�@�       �	b��fc�A�7*

loss���:��       �	p��fc�A�7*

lossᎾ;J(�       �	M-�fc�A�7*

loss�-�;m�i       �	z��fc�A�7*

loss���:��/�       �	hy�fc�A�7*

loss-��8��9       �	9�fc�A�7*

loss���9b�w       �	ʣ�fc�A�7*

loss�<�:��>       �	�6�fc�A�7*

loss�t :`v�W       �	���fc�A�7*

loss��;-%�       �	Ze�fc�A�7*

lossy%8[�WA       �		��fc�A�7*

loss��#<뮌6       �	���fc�A�7*

loss�W�9��\�       �	�~�fc�A�7*

loss�qR:!�c       �	X��fc�A�7*

loss�:y8��O�       �	���fc�A�7*

loss�2�;�
X       �	��fc�A�7*

loss�^[<'�	       �	ĵ�fc�A�7*

loss=��:�S*       �	QN�fc�A�7*

loss�
:,>Qt       �	u��fc�A�7*

loss-;k9w       �	>{�fc�A�7*

loss*��;uߌ)       �	A�fc�A�7*

loss���<��       �	��fc�A�7*

lossr�x=Ɖ��       �	�>�fc�A�7*

lossg;3Z��       �	8��fc�A�7*

loss;��;�٘�       �	�x�fc�A�7*

loss�`�<�5�       �	I fc�A�7*

loss�},9���       �	� fc�A�7*

loss� &<���       �	Abfc�A�7*

loss+؋=��vR       �	'�fc�A�7*

lossqmS;�8�#       �	W�fc�A�7*

loss9#�:��D�       �	�/fc�A�7*

loss�=�9�\       �	6�fc�A�7*

losswP<�8��       �	tbfc�A�7*

loss@J%=&[�%       �	v�fc�A�7*

loss�ag9���       �	��fc�A�7*

loss&\�;��ڲ       �	[%fc�A�7*

loss	��:@Πb       �	3�fc�A�7*

lossf�<����       �	Aafc�A�7*

loss��L<�Ǹ3       �	�fc�A�7*

loss�Y�:�r�
       �	B�fc�A�8*

loss	2�<�aI�       �	�[	fc�A�8*

loss�;���       �	]�	fc�A�8*

loss�/U9U���       �	��
fc�A�8*

loss.��;����       �	�Efc�A�8*

lossn�<�mʷ       �	��fc�A�8*

lossl��:ɔ�h       �	E�fc�A�8*

loss���;	���       �	�fc�A�8*

loss��u;��/1       �	�fc�A�8*

loss�@G<F�8�       �	]Qfc�A�8*

loss�X9-�T�       �	��fc�A�8*

loss/�<:
�_�       �	҉fc�A�8*

lossj�;		3(       �	�fc�A�8*

lossLjt:�K�       �	x�fc�A�8*

lossÆ�<d~�y       �	&Wfc�A�8*

loss��m:-x�       �	��fc�A�8*

lossD�<�n�V       �	Y�fc�A�8*

loss�:�J       �	�%fc�A�8*

loss���:��gG       �	��fc�A�8*

lossH��;|���       �	t�fc�A�8*

loss*r�=Z��       �	Rfc�A�8*

loss!�=Ǉ�	       �	�fc�A�8*

loss��<�'��       �	
�fc�A�8*

loss�`p=���       �	dXfc�A�8*

loss�
�:�K��       �	8�fc�A�8*

losscc�=��W!       �	��fc�A�8*

loss���:P���       �	�1fc�A�8*

lossc�5:N��       �	9�fc�A�8*

losst-a<�gG�       �	dtfc�A�8*

lossM�;l$�b       �	�fc�A�8*

lossNY;p�
       �	�:fc�A�8*

loss}A;0��X       �	M�fc�A�8*

lossP��:,�w@       �	qsfc�A�8*

loss2T;�ya�