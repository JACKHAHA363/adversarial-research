       �K"	   NYc�Abrain.Event:2ʯ�3�     �̘.	�fNYc�A"��
^
dataPlaceholder*
shape: *
dtype0*/
_output_shapes
:���������
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
valueB"         @   *
_output_shapes
:*
dtype0
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
 *�x=*
dtype0*
_output_shapes
: 
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
T0*
seed���)*
dtype0
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
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_1/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
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
valueB"         @   *
_output_shapes
:*
dtype0
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
seed2ٳ*
T0*
seed���)*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
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
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
shape: *
dtype0
*
_output_shapes
:
�
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

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
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2Ş*
T0*
seed���)*
dtype0
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
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
T0*
N
c
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes
:*

begin_mask *
ellipsis_mask 
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
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
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
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
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
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
\
dense_1/ConstConst*
valueB�*    *
_output_shapes	
:�*
dtype0
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
dtype0*
shared_name 
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
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
]
activation_3/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
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
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
:
e
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
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
dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
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
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��*
T0*
seed���)*
dtype0
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
%dropout_2/cond/dropout/random_uniformAdd)dropout_2/cond/dropout/random_uniform/mul)dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
T0*
N
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
 *̈́U>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2��+*
T0*
seed���)*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�
*
T0
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
*    *
dtype0*
_output_shapes
:

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
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
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
valueB"         @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:���������@*
use_cudnn_on_gpu(
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
�
'sequential_1/conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
strides
*
data_formatNHWC
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
�
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
y
$sequential_1/dropout_1/cond/switch_tIdentity$sequential_1/dropout_1/cond/Switch:1*
T0
*
_output_shapes
:
w
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
_output_shapes
:*
T0

r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
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
seed2���*
T0*
seed���)*
dtype0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
�
2sequential_1/dropout_1/cond/dropout/random_uniformAdd6sequential_1/dropout_1/cond/dropout/random_uniform/mul6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*/
_output_shapes
:���������@*
T0
�
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*/
_output_shapes
:���������@*
T0
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
T0*
N
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
valueB:*
dtype0*
_output_shapes
:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
f
sequential_1/flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
�
sequential_1/dense_1/MatMulMatMulsequential_1/flatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*(
_output_shapes
:����������*
T0
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
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
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
 *    *
_output_shapes
: *
dtype0
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
:����������*
seed2���*
T0*
seed���)*
dtype0
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
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*(
_output_shapes
:����������*
T0
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*
T0*(
_output_shapes
:����������
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
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

[
num_inst/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
num_inst
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
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
num_inst/readIdentitynum_inst*
_class
loc:@num_inst*
_output_shapes
: *
T0
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
num_correct/AssignAssignnum_correctnum_correct/initial_value*
use_locking(*
T0*
_class
loc:@num_correct*
validate_shape(*
_output_shapes
: 
j
num_correct/readIdentitynum_correct*
T0*
_class
loc:@num_correct*
_output_shapes
: 
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
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
S
ToFloatCastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
	AssignAdd	AssignAddnum_instConst_1*
_class
loc:@num_inst*
_output_shapes
: *
T0*
use_locking( 
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
AssignAssignnum_instConst_2*
_class
loc:@num_inst*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
L
Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
 *���.*
_output_shapes
: *
dtype0
A
addAddnum_inst/readadd/y*
T0*
_output_shapes
: 
F
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
valueB
 *   B*
_output_shapes
: *
dtype0
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
 softmax_cross_entropy_loss/ShapeShapediv_1*
T0*
out_type0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
out_type0*
_output_shapes
:*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0
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
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
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
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
out_type0*
_output_shapes
:
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
valueB:*
_output_shapes
:*
dtype0
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
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
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*

axis *
_output_shapes
:*
T0*
N
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
valueB *
_output_shapes
: *
dtype0
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
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
valueB: *
_output_shapes
:*
dtype0
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
_output_shapes
: *
T0
�
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
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
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
 *    *
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
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
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
:
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
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
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
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
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
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
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
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
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
out_type0*
_output_shapes
:*
T0
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
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
valueB *
_output_shapes
: *
dtype0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
valueB *
_output_shapes
: *
dtype0
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
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
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
valueB: *
dtype0*
_output_shapes
:
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0
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

Tdim0*
T0*'
_output_shapes
:���������
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
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

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
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������
*
T0
�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
_output_shapes
:
*
T0*
data_formatNHWC
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������

�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
1gradients/sequential_1/dense_2/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1MatMul!sequential_1/dropout_2/cond/MergeDgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�
*
transpose_a(
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�

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
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
_output_shapes
:*
T0
Z
gradients/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
T0*
N
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
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
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*(
_output_shapes
:����������*
T0
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
T0*
Tshape0*
_output_shapes
: 
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*(
_output_shapes
:����������*
T0
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
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*(
_output_shapes
:����������*
T0
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: 
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1**
_output_shapes
:����������: *
T0*
N
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
N*(
_output_shapes
:����������
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*
T0*(
_output_shapes
:����������
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
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
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
1gradients/sequential_1/dense_1/MatMul_grad/MatMulMatMulDgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*)
_output_shapes
:�����������*
transpose_a( 
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
T0*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*)
_output_shapes
:�����������
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*!
_output_shapes
:���
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
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
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@
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
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*1
_output_shapes
:���������@: *
T0*
N
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
out_type0*
_output_shapes
:*
T0
�
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������@
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
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
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
out_type0*
_output_shapes
:*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
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
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*/
_output_shapes
:���������@*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*/
_output_shapes
:���������@
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
_output_shapes
: 
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
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������@*
T0
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
T0*
N*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*/
_output_shapes
:���������@*
T0*
N
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_2/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
out_type0*
_output_shapes
:*
T0
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4������������������������������������
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
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*
T0*/
_output_shapes
:���������@
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
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
dtype0*
_output_shapes
:
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
VariableV2*
shape: *
_output_shapes
: *
shared_name *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
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
beta1_power/readIdentitybeta1_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta2_power
VariableV2*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
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
VariableV2*
shape:@*&
_output_shapes
:@*
shared_name *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
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
VariableV2*
shape:@*&
_output_shapes
:@*
shared_name *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
T
zeros_2Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_1/bias/Adam
VariableV2*
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_1/bias*
dtype0*
	container 
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
l
zeros_4Const*%
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
T
zeros_6Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam
VariableV2* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
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
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
T
zeros_7Const*
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam_1
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
b
zeros_8Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam
VariableV2*
shape:���*!
_output_shapes
:���*
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container 
�
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
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
VariableV2*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
shape:���*
dtype0*
shared_name *
	container 
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
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense_1/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
shape:�*
dtype0*
shared_name *
	container 
�
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
�
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
_
zeros_12Const*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam
VariableV2*
shape:	�
*
_output_shapes
:	�
*
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
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
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

_
zeros_13Const*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
dense_2/kernel/Adam_1
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
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

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
*    *
_output_shapes
:
*
dtype0
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
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0
U
zeros_15Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 
�
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

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
 *w�?*
dtype0*
_output_shapes
: 
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
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
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
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
use_locking( 
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
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
use_locking( 
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
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
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
N"��z���     ����	�NYc�AJ��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
T0*
seed���)*
dtype0
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 
�
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*&
_output_shapes
:@*
T0
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
valueB@*    *
dtype0*
_output_shapes
:@
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
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
data_formatNHWC*
strides

�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
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
 *�\1=*
_output_shapes
: *
dtype0
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2ٳ*
T0*
seed���)*
dtype0
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
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0
�
conv2d_2/kernel
VariableV2*
shape:@@*
shared_name *
dtype0*&
_output_shapes
:@@*
	container 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
[
conv2d_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
conv2d_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
s
conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
paddingVALID*
T0*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
use_cudnn_on_gpu(
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
_output_shapes
:*
shape: *
dtype0

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
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

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
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2Ş*
T0*
seed���)*
dtype0
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
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/dropout/addAdd dropout_1/cond/dropout/keep_prob%dropout_1/cond/dropout/random_uniform*
T0*/
_output_shapes
:���������@
{
dropout_1/cond/dropout/FloorFloordropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
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
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
T0*
Index0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
Y
flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
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
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
_output_shapes
:*
dtype0
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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*!
_output_shapes
:���*
T0
�
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*!
_output_shapes
:���
�
dense_1/kernel
VariableV2*
shape:���*
shared_name *
dtype0*!
_output_shapes
:���*
	container 
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
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
\
dense_1/ConstConst*
valueB�*    *
_output_shapes	
:�*
dtype0
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
dtype0*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
r
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0
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
:����������*
T0*
data_formatNHWC
]
activation_3/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
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
dropout_2/cond/switch_fIdentitydropout_2/cond/Switch*
T0
*
_output_shapes
:
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
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2��
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_2/cond/dropout/random_uniform/mulMul3dropout_2/cond/dropout/random_uniform/RandomUniform)dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
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
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
T0*
N
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
 *̈́U�*
_output_shapes
: *
dtype0
_
dense_2/random_uniform/maxConst*
valueB
 *̈́U>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2��+*
T0*
seed���)*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�


dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�
*
T0
�
dense_2/kernel
VariableV2*
shape:	�
*
shared_name *
dtype0*
_output_shapes
:	�
*
	container 
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
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
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
valueB"         @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
strides
*
data_formatNHWC
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
�
'sequential_1/conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:���������@*
strides
*
data_formatNHWC
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC

sequential_1/activation_2/ReluRelusequential_1/conv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
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
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/mulMul@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:���������@
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
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
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
valueB: *
_output_shapes
:*
dtype0
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
T0*
Index0*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask 
f
sequential_1/flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
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
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
w
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*(
_output_shapes
:����������*
T0
�
"sequential_1/dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

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
 *  �?*
_output_shapes
: *
dtype0
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������*
T0
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
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
'sequential_1/dropout_2/cond/dropout/addAdd-sequential_1/dropout_2/cond/dropout/keep_prob2sequential_1/dropout_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:����������
�
)sequential_1/dropout_2/cond/dropout/FloorFloor'sequential_1/dropout_2/cond/dropout/add*
T0*(
_output_shapes
:����������
�
'sequential_1/dropout_2/cond/dropout/divRealDivsequential_1/dropout_2/cond/mul-sequential_1/dropout_2/cond/dropout/keep_prob*(
_output_shapes
:����������*
T0
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
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
T0*
N
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
 *    *
dtype0*
_output_shapes
: 
l
num_inst
VariableV2*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
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
 *    *
dtype0*
_output_shapes
: 
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
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
ToFloatCastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
X
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@num_inst
~
AssignAdd_1	AssignAddnum_correctSum*
_output_shapes
: *
_class
loc:@num_correct*
T0*
use_locking( 
L
Const_2Const*
dtype0*
_output_shapes
: *
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
Assign_1Assignnum_correctConst_3*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
use_locking(
J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.
A
addAddnum_inst/readadd/y*
T0*
_output_shapes
: 
F
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*
T0*'
_output_shapes
:���������

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
 softmax_cross_entropy_loss/Sub/yConst*
dtype0*
_output_shapes
: *
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
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
dtype0*
_output_shapes
:*
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
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
c
!softmax_cross_entropy_loss/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
_output_shapes
:*
out_type0*
T0
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
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
_output_shapes
:*
N*

axis *
T0
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
j
(softmax_cross_entropy_loss/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
d
"softmax_cross_entropy_loss/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
dtype0*
_output_shapes
:*
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
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*#
_output_shapes
:���������*
Tshape0*
T0
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
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
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
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
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
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
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
PlaceholderPlaceholder*
_output_shapes
:*
dtype0*
shape: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
T0
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
_output_shapes
: *H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
T0
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
T0
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
_output_shapes
: *
T0
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
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
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
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
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
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
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
T0*
_output_shapes
:*
out_type0
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
_output_shapes
:*
out_type0*
T0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
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
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*#
_output_shapes
:���������*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
T0
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
T0*
_output_shapes
: *
Tshape0
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
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
_output_shapes
:*
out_type0*
T0
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
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:������������������
�
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
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
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
T0*
_output_shapes
:*
out_type0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
_
gradients/div_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_1_grad/RealDivRealDiv9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapediv_1/y*
T0*'
_output_shapes
:���������

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
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*'
_output_shapes
:���������
*
T0
�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/div_1_grad/Reshape_1
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
_output_shapes
:
*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape
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
transpose_b( *
T0*
_output_shapes
:	�
*
transpose_a(
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
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*
T0
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
_output_shapes
:*
out_type0*
T0
�
Lgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulMulKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_2/cond/dropout/divKgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*(
_output_shapes
:����������*
Tshape0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape
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
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Dgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_2/cond/mul_grad/Shape6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
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
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
T0
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
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
�
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
N*
T0*1
_output_shapes
:���������@: 
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
T0
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
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*/
_output_shapes
:���������@*
T0
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape
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
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Mul(sequential_1/dropout_1/cond/mul/Switch:1Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency*
T0*/
_output_shapes
:���������@
�
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
?gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
T0
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
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
N*
T0
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
N*
T0*/
_output_shapes
:���������@*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
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
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"      @   @   
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@@*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*
T0
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
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
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         @   
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@*
data_formatNHWC*
strides
*
T0*
paddingVALID
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*
T0
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*&
_output_shapes
:@*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0
�
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta1_power/readIdentitybeta1_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
�
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel
�
beta2_power
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *
_output_shapes
: *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
j
zerosConst*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam
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
zeros_1Const*&
_output_shapes
:@*
dtype0*%
valueB@*    
�
conv2d_1/kernel/Adam_1
VariableV2*&
_output_shapes
:@*
dtype0*
shape:@*
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
T
zeros_2Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_1/bias/Adam
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_1/bias*
shared_name 
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
T
zeros_3Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_1/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
�
conv2d_2/kernel/Adam
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
l
zeros_5Const*&
_output_shapes
:@@*
dtype0*%
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
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
T
zeros_6Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_2/bias/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
zeros_7Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_2/bias/Adam_1
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
b
zeros_9Const*!
_output_shapes
:���*
dtype0* 
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
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
dtype0*
	container 
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
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
_
zeros_12Const*
_output_shapes
:	�
*
dtype0*
valueB	�
*    
�
dense_2/kernel/Adam
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
_output_shapes
:	�
*
dtype0*
valueB	�
*    
�
dense_2/kernel/Adam_1
VariableV2*
shared_name *
shape:	�
*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
dtype0*
	container 
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
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
U
zeros_14Const*
dtype0*
_output_shapes
:
*
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
zeros_15Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
dense_2/bias/Adam_1
VariableV2*
shared_name *
shape:
*
_output_shapes
:
*
_class
loc:@dense_2/bias*
dtype0*
	container 
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
dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0*
use_locking( 
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
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"
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
"softmax_cross_entropy_loss/value:0g]��       ��-	��MNYc�A*

lossi`@Ü�T       ��-	u�NNYc�A*

loss�@�%��       ��-	IONYc�A*

loss��@���       ��-	��ONYc�A*

loss(�@���       ��-	z�PNYc�A*

loss�@�B*�       ��-	s/QNYc�A*

loss�E	@C�ݿ       ��-	��QNYc�A*

lossq@)أ}       ��-	�mRNYc�A*

loss��@`�p       ��-	h	SNYc�A	*

loss ��?��
       ��-	*�SNYc�A
*

loss���?A�?       ��-	�HTNYc�A*

loss� �?b�$       ��-	i�TNYc�A*

loss���?!���       ��-	ڑUNYc�A*

lossE&�?�Zu       ��-	�)VNYc�A*

lossaE�?�	��       ��-	w�VNYc�A*

loss��?�t�       ��-	�qWNYc�A*

lossem?)��       ��-	�XNYc�A*

loss*�?Y'��       ��-	�XNYc�A*

lossx�`?p�ij       ��-	�fYNYc�A*

loss5c�?�n=       ��-	i�YNYc�A*

loss�Æ?齏�       ��-	
�ZNYc�A*

loss��?�9aU       ��-	hB[NYc�A*

loss/�c?K��       ��-	��[NYc�A*

loss
	�?��       ��-	�\NYc�A*

loss)hi?�~=�       ��-	 ]NYc�A*

loss�K?��$6       ��-	8�]NYc�A*

loss.G?D8�       ��-	�]^NYc�A*

loss�
?^��       ��-	T _NYc�A*

loss�J�?͜K       ��-	z�_NYc�A*

lossȝv?�A       ��-	B`NYc�A*

loss4B9?�J"�       ��-	��`NYc�A*

loss#�%?���z       ��-	-waNYc�A *

lossJ�Q?����       ��-	
bNYc�A!*

loss��z?_��       ��-	B�bNYc�A"*

loss,?�2h�       ��-	�FcNYc�A#*

loss�nY?$�>O       ��-	?�cNYc�A$*

loss�,%?`��       ��-	܃dNYc�A%*

lossq�g?�j       ��-	S#eNYc�A&*

lossa�?r*f       ��-	��eNYc�A'*

loss$b?R���       ��-	ffNYc�A(*

loss��%?Z�D^       ��-	SgNYc�A)*

loss=�
?_���       ��-	�gNYc�A**

loss�%?���p       ��-	�GhNYc�A+*

loss�
/?�;��       ��-	��hNYc�A,*

lossL�=?��#       ��-	6�iNYc�A-*

loss[9?�5�:       ��-	�=jNYc�A.*

loss��#?Lq�       ��-	��jNYc�A/*

loss��?�N�=       ��-	�|kNYc�A0*

loss�\'?'�s�       ��-	�lNYc�A1*

lossr��>'�       ��-	ĲlNYc�A2*

loss�7�>�{��       ��-	�JmNYc�A3*

loss+?NP�D       ��-	��mNYc�A4*

loss�z?�2�%       ��-	�}nNYc�A5*

loss(�J?\�0       ��-	XoNYc�A6*

loss��>11,t       ��-	�oNYc�A7*

loss1B�>¶D�       ��-	papNYc�A8*

loss�8�>���8       ��-	p	qNYc�A9*

lossq��>�9�g       ��-	n�qNYc�A:*

loss��7?h:�w       ��-	\<rNYc�A;*

loss���>�p�       ��-	��rNYc�A<*

lossJ�?��#2       ��-	ssNYc�A=*

loss���>����       ��-	% tNYc�A>*

loss���>���       ��-	A�tNYc�A?*

lossw��>K���       ��-	ZguNYc�A@*

loss:�?�.8'       ��-	�vNYc�AA*

loss�F�>�y�       ��-	H�vNYc�AB*

lossS�6?Y��v       ��-	IcwNYc�AC*

loss�e?��       ��-	G�wNYc�AD*

loss3�B?�s:�       ��-	?�xNYc�AE*

loss1b?�D�       ��-	�SyNYc�AF*

loss�V�>Dm�l       ��-	4�yNYc�AG*

loss-)?�!       ��-	�zNYc�AH*

loss���>c|Y       ��-	�2{NYc�AI*

loss�?y"�       ��-	��{NYc�AJ*

loss�u8?f��       ��-	A}|NYc�AK*

loss�"?Eq�       ��-	}NYc�AL*

loss�d4?�g�       ��-	'�}NYc�AM*

loss��J?w%HQ       ��-	�j~NYc�AN*

loss�?�7       ��-	�NYc�AO*

loss���>ɪ7       ��-	ǺNYc�AP*

loss�T�>��.�       ��-	d�NYc�AQ*

lossf(?#@�       ��-	N�NYc�AR*

loss�>?u��p       ��-	���NYc�AS*

lossTD�>F6)       ��-	U�NYc�AT*

loss���>�B��       ��-	K�NYc�AU*

loss� �>$�wj       ��-	蟃NYc�AV*

loss[H�>�(��       ��-	�@�NYc�AW*

lossq5?��N       ��-	i�NYc�AX*

lossŶ?�t��       ��-	���NYc�AY*

loss�>e�9       ��-	T�NYc�AZ*

loss�z�>�zy       ��-	g��NYc�A[*

lossm��>�'       ��-	�J�NYc�A\*

lossw?H��w       ��-	a�NYc�A]*

loss�C?I�q�       ��-	�x�NYc�A^*

loss_�?�y�       ��-	,�NYc�A_*

loss��>F��       ��-	8��NYc�A`*

lossʂ�>P ��       ��-	�:�NYc�Aa*

loss�K'?�f�       ��-	܊NYc�Ab*

loss��R?uֹ	       ��-	�v�NYc�Ac*

loss�F?B0�@       ��-	�NYc�Ad*

loss��>纱�       ��-	ۥ�NYc�Ae*

loss4�>���_       ��-	O��NYc�Af*

lossF�?�#��       ��-	�D�NYc�Ag*

loss��>28}       ��-	��NYc�Ah*

loss�>J�2�       ��-	0��NYc�Ai*

loss7��>g�t       ��-	�T�NYc�Aj*

loss���>�D��       ��-	`�NYc�Ak*

loss?��>	g@X       ��-	r��NYc�Al*

loss��??�[�       ��-	���NYc�Am*

loss���>�ș�       ��-	N�NYc�An*

loss��?e�R       ��-	\��NYc�Ao*

loss�v ?}�e       ��-	�F�NYc�Ap*

loss��>�#��       ��-	ܕNYc�Aq*

loss�M>�8,u       ��-	��NYc�Ar*

loss���>D)^�       ��-	9��NYc�As*

loss��>>�       ��-	"�NYc�At*

loss�k�>u�       ��-	�NYc�Au*

loss�;�>B���       ��-	~�NYc�Av*

loss,.�>V	��       ��-	�y�NYc�Aw*

loss$��>�P�#       ��-	W`�NYc�Ax*

loss}��>CT��       ��-	�]�NYc�Ay*

loss�%�>G0�'       ��-	�&�NYc�Az*

loss�^�>�T��       ��-	'l�NYc�A{*

lossL �>�ƹ       ��-	��NYc�A|*

loss. >̯�       ��-	��NYc�A}*

loss�D�>2wr       ��-	0�NYc�A~*

lossw��>�:C�       ��-	��NYc�A*

loss��>g�G�       �	c��NYc�A�*

loss�޷> :�       �	3��NYc�A�*

lossֹ�>x�PE       �	l^�NYc�A�*

lossT�s>���       �	���NYc�A�*

loss<{�>>�3       �	f��NYc�A�*

loss�;>E�y       �	�u�NYc�A�*

lossDcS>S{Kn       �	��NYc�A�*

loss�Bh>~�]�       �	uȨNYc�A�*

lossص�>bK��       �	��NYc�A�*

lossO�>�8{       �	ʤ�NYc�A�*

loss�y> ��       �	�B�NYc�A�*

loss��s>���       �	<ݫNYc�A�*

lossH~`>w���       �	x�NYc�A�*

loss�>>n�0       �	��NYc�A�*

loss��>Xh�@       �	���NYc�A�*

lossf��>��X�       �	�L�NYc�A�*

loss3��>Ɋ�       �	�NYc�A�*

loss���>U���       �	�}�NYc�A�*

loss��A>���A       �	��NYc�A�*

loss���>d.�       �	�ʰNYc�A�*

lossV>n*4       �	d�NYc�A�*

loss�|>>�A_       �	;�NYc�A�*

loss_S>m       �	ꓲNYc�A�*

loss�!�>�La�       �	$'�NYc�A�*

loss�_�>-wF>       �	ü�NYc�A�*

lossߟ�>�J8�       �	�]�NYc�A�*

loss:��>����       �	���NYc�A�*

loss4Z�>!�DT       �	ۋ�NYc�A�*

loss3�C>F�(�       �	:$�NYc�A�*

loss$`�>�%@J       �	�NYc�A�*

loss�N�>����       �	�O�NYc�A�*

lossW+?X �       �	s��NYc�A�*

lossQF�>�uL       �	���NYc�A�*

losst��>�.�       �	�G�NYc�A�*

lossL�>x�6c       �	0�NYc�A�*

lossN�v>~�<       �	:��NYc�A�*

loss$:1>n:�,       �	�9�NYc�A�*

lossr�>_��       �	��NYc�A�*

loss��>���       �	�NYc�A�*

loss�{�>����       �	B'�NYc�A�*

lossr�>-wb�       �	�ŽNYc�A�*

lossz/=>h<%       �	�]�NYc�A�*

loss��A>��Vw       �	���NYc�A�*

loss�}>�(       �	���NYc�A�*

loss,>'N�W       �	�#�NYc�A�*

loss��>��       �	���NYc�A�*

loss/O�>��\I       �	�b�NYc�A�*

loss�I>�|       �	1
�NYc�A�*

loss=Y�>"��?       �	��NYc�A�*

lossJ�	?�@[�       �	�U�NYc�A�*

loss��>����       �	��NYc�A�*

lossQn>T��>       �	w�NYc�A�*

loss�V>=�(       �	5	�NYc�A�*

loss|8�=@פ$       �	��NYc�A�*

loss�}�>F}��       �	>�NYc�A�*

loss�M>ɑ��       �	��NYc�A�*

loss�$>Tq�/       �	]l�NYc�A�*

lossmY>�ʥ       �	��NYc�A�*

lossA�>= tu       �	Ω�NYc�A�*

loss���>���       �	�H�NYc�A�*

loss�G�>�_�       �	���NYc�A�*

loss�5z>���       �	I��NYc�A�*

losst_s>�~�       �	�"�NYc�A�*

loss���>k`t       �	I��NYc�A�*

loss2ǜ>_��       �	|`�NYc�A�*

loss�Z/>
C       �	Y��NYc�A�*

loss���>�t       �	4��NYc�A�*

loss�>�_�o       �	��NYc�A�*

loss�>M_^~       �	�&�NYc�A�*

lossTRr>��>       �	�G�NYc�A�*

loss)��>�J�Q       �	�:�NYc�A�*

loss���>���       �	���NYc�A�*

loss��>�mʌ       �	n��NYc�A�*

loss�(�>�q18       �	$%�NYc�A�*

loss��>;%͹       �	���NYc�A�*

loss��~>��%�       �	$_�NYc�A�*

lossQ=/>u�*e       �	���NYc�A�*

lossʹ7>.(��       �	���NYc�A�*

loss�?>2bc�       �	�Z�NYc�A�*

loss>�>��h       �	
��NYc�A�*

loss$�T>�ܹ�       �	;��NYc�A�*

loss�[:>�m<�       �	#�NYc�A�*

losswݲ>�a��       �	���NYc�A�*

loss��O>���       �	�r�NYc�A�*

lossXHe>�+�       �	��NYc�A�*

loss 7�=$��       �	̳�NYc�A�*

lossqĂ>��}I       �	W�NYc�A�*

loss-��>�r�       �	��NYc�A�*

loss���>��       �	��NYc�A�*

loss��>�       �	�.�NYc�A�*

loss���>n�h�       �	���NYc�A�*

loss{��>{�D       �	1]�NYc�A�*

lossq��>Q�7       �	���NYc�A�*

lossx�=7�]       �	���NYc�A�*

loss�[(>S�e�       �	�a�NYc�A�*

lossoG�>pY'�       �	��NYc�A�*

loss�/h>�1��       �	���NYc�A�*

lossP�>��uF       �	��NYc�A�*

loss��[>�9>       �	���NYc�A�*

loss��D>ڀr�       �	=I�NYc�A�*

loss���>l�#       �	��NYc�A�*

loss���>aݝ       �	�~�NYc�A�*

loss}xX>x&O       �	���NYc�A�*

loss�8>�K�u       �	�"�NYc�A�*

loss(��=g�       �	
��NYc�A�*

loss��>^��       �	h]�NYc�A�*

loss�$>^�!�       �	D��NYc�A�*

loss��>���       �	��NYc�A�*

lossݝF>E�       �	7�NYc�A�*

loss�LO>��E�       �	���NYc�A�*

loss\�> *ۚ       �	�l�NYc�A�*

loss�}>��+�       �	��NYc�A�*

lossTi%>Q�v       �	I��NYc�A�*

loss��d>�`�       �	�^�NYc�A�*

loss�L>��N       �	(
�NYc�A�*

loss���>�	�       �	4��NYc�A�*

lossa7a>Z3��       �	�e�NYc�A�*

loss&��>oF       �	l�NYc�A�*

loss}�1>K��+       �	���NYc�A�*

lossʩ$>���       �	�6�NYc�A�*

loss�a&>{\v�       �	���NYc�A�*

lossL�W>k�Ne       �	i�NYc�A�*

loss�R�>����       �	� �NYc�A�*

lossbǉ>���{       �	���NYc�A�*

losss��>#߷�       �	$B�NYc�A�*

loss]��>���8       �	���NYc�A�*

loss�(
>&<       �	˄�NYc�A�*

loss.h�=᫕�       �	��NYc�A�*

loss�B>|9�       �	���NYc�A�*

loss�y�>�H��       �	�S�NYc�A�*

lossjd�>	*q�       �	��NYc�A�*

loss�::>S��       �	���NYc�A�*

loss*��>�U�5       �	�&�NYc�A�*

lossOڀ>�b��       �	P��NYc�A�*

loss<�>L�,�       �	%]�NYc�A�*

loss��>�2r       �	��NYc�A�*

loss��o>	�HU       �	-��NYc�A�*

lossM�>�Ub�       �	�/�NYc�A�*

loss�֓>����       �	���NYc�A�*

lossan�>L��       �	�n�NYc�A�*

loss��h>��U       �	 OYc�A�*

losst�>�K��       �	4� OYc�A�*

loss1n>�v��       �	�\OYc�A�*

lossJ�D>�w       �	��OYc�A�*

loss)h>�K       �	��OYc�A�*

loss�M\>�%       �	�OYc�A�*

loss1�B>K�       �	ظOYc�A�*

loss��>��       �	eQOYc�A�*

loss���>yn��       �	��OYc�A�*

loss��>�H�       �	8�OYc�A�*

loss�� >��\�       �	rOYc�A�*

loss���>.�*       �	�OYc�A�*

lossO�C>��Yc       �	[[OYc�A�*

loss�
�=z0��       �	�OYc�A�*

lossj>{>D�       �	��OYc�A�*

loss�>M�;B       �	�0	OYc�A�*

loss
�F>ѕ"L       �	S�	OYc�A�*

lossa��>�$�`       �	�d
OYc�A�*

lossϼf>��M       �	`OYc�A�*

lossAT>��C       �	P�OYc�A�*

lossߋ#>0j/v       �	�YOYc�A�*

loss��y>�3�%       �	�OYc�A�*

loss�q�=j^��       �	¤OYc�A�*

loss8�=�n�P       �	�XOYc�A�*

lossNT<>o�(�       �	�9OYc�A�*

lossE��='�E�       �	�OYc�A�*

loss��>M�p       �	�yOYc�A�*

loss,}$>�u(�       �	G;OYc�A�*

loss�>���n       �	��OYc�A�*

loss�ä>d(kB       �	o�OYc�A�*

loss,��>�#0       �	�COYc�A�*

lossR��>W&�h       �	��OYc�A�*

lossd�Z>�lL�       �	
�OYc�A�*

loss�X>'yO/       �	qOYc�A�*

loss6�>s�w�       �	�OYc�A�*

loss�s'>
��s       �	GsOYc�A�*

lossi�r>��(>       �	LOYc�A�*

lossM�=kmi       �	��OYc�A�*

loss!'�>=��       �	VfOYc�A�*

loss���=׍��       �	�OYc�A�*

lossa�`>���       �	s�OYc�A�*

lossM��=zp�P       �	>[OYc�A�*

lossOR]>U��       �	b,OYc�A�*

loss���>`P%�       �	��OYc�A�*

loss��z>���       �	Q�OYc�A�*

loss�9/>��/       �	� OYc�A�*

loss�>Q�P�       �	/�OYc�A�*

loss�>$>#��y       �	�YOYc�A�*

loss��>>�V       �	Q�OYc�A�*

loss�rW>53<W       �	��OYc�A�*

lossH}�=S�y�       �	�& OYc�A�*

loss��y>��|[       �	'� OYc�A�*

loss�Z>�X�o       �	��!OYc�A�*

loss�s�={&�       �	�7"OYc�A�*

loss��>Vf��       �	(#OYc�A�*

loss�#t>l�y       �	�#OYc�A�*

lossXH>>���       �	ȱ$OYc�A�*

loss�>�0�6       �	V%OYc�A�*

loss$�?(���       �	U�%OYc�A�*

loss��[>þ�       �	��&OYc�A�*

loss��->�v��       �	OX'OYc�A�*

loss�E�>���       �	��'OYc�A�*

lossc�>A�x       �	<�(OYc�A�*

loss��^>A̽�       �	�=)OYc�A�*

loss&>�Q�       �	��)OYc�A�*

loss�>����       �	l*OYc�A�*

loss��>>29_i       �	�+OYc�A�*

losss=�>��        �	]�+OYc�A�*

loss�*%>�3�       �	q;,OYc�A�*

loss	�p>��U�       �	�,OYc�A�*

loss}ӂ>���       �	�w-OYc�A�*

loss��O>�X�       �	`#.OYc�A�*

lossP-�=8-�       �	�]/OYc�A�*

loss
�">��U       �	#�/OYc�A�*

loss]�=+��       �	a�0OYc�A�*

loss�,W>���D       �	K#1OYc�A�*

loss<�?>/"�T       �	��1OYc�A�*

lossV ?�|۞       �	Q2OYc�A�*

loss*ރ>���D       �	�2OYc�A�*

loss��=���V       �	-{3OYc�A�*

loss��y>��       �	/4OYc�A�*

losss9>��i       �	��4OYc�A�*

loss��>�.��       �	]5OYc�A�*

lossZ5>'��       �	(�5OYc�A�*

lossI%>���       �	��6OYc�A�*

loss�r#>:�9�       �	�27OYc�A�*

loss��C>�$��       �	q�7OYc�A�*

loss*�">�y��       �	�_8OYc�A�*

loss��=�Aޅ       �	��8OYc�A�*

loss��&>m�AO       �	9OYc�A�*

loss��*>v}�       �	�:OYc�A�*

lossc.>���       �	��:OYc�A�*

loss�i	>]tr       �	sL;OYc�A�*

loss5�>��S�       �	C�;OYc�A�*

loss��a>�3��       �	�}<OYc�A�*

loss���=�)S       �	�=OYc�A�*

lossN��>�Ձ�       �	��=OYc�A�*

lossZ1I>�6�       �	�N>OYc�A�*

lossMy�>4]y�       �	q�>OYc�A�*

loss�>#>D��       �	tz?OYc�A�*

loss��=�죪       �	�@OYc�A�*

lossM�0>֠}-       �	��@OYc�A�*

loss�>�0κ       �	�WAOYc�A�*

loss�	>p��       �	��AOYc�A�*

loss�'>��(       �	Z�BOYc�A�*

loss��%>7�Ƒ       �	�COYc�A�*

loss�='>B�j       �	ղCOYc�A�*

loss8e>n@G�       �	fIDOYc�A�*

loss�#�>� �       �	��DOYc�A�*

loss�ъ>���P       �	xEOYc�A�*

lossC4>��F�       �	�FOYc�A�*

loss�(>�-;�       �	�FOYc�A�*

loss=dy>��\       �	�OGOYc�A�*

loss��>=�X2       �	��GOYc�A�*

loss�5�>\Υ�       �	�HOYc�A�*

losshc�=;��f       �	�-IOYc�A�*

loss#�=9eà       �	�IOYc�A�*

lossߡb=�wm       �	�_JOYc�A�*

loss��=d�-0       �	n�JOYc�A�*

loss2�a>�J��       �	�KOYc�A�*

losscHJ>�j��       �	�XLOYc�A�*

loss�}>�KX       �	�LOYc�A�*

lossw�8>�79�       �	S�MOYc�A�*

loss��E>����       �	v�NOYc�A�*

loss�x>̎       �	c�OOYc�A�*

loss�FM>��2�       �	ݘPOYc�A�*

loss��> uX�       �	�CQOYc�A�*

loss�>>�s       �	�aROYc�A�*

loss0�>I�/       �	gSOYc�A�*

loss�9�="Y�       �	~�SOYc�A�*

lossD��=����       �	�YTOYc�A�*

loss��>��m�       �	^�TOYc�A�*

loss#�i>�i��       �	�UOYc�A�*

loss�)>8�v�       �	�WVOYc�A�*

loss4��>=̝       �	_�VOYc�A�*

lossIP&>9���       �	A�WOYc�A�*

lossn�=�/P�       �	�XOYc�A�*

lossԙ�=X�v�       �	��XOYc�A�*

loss�M5>��')       �	�\YOYc�A�*

loss]kN>�-�O       �	��YOYc�A�*

loss�3>��       �	͒ZOYc�A�*

loss!�!>�n�K       �	M0[OYc�A�*

loss���> 05       �	\�[OYc�A�*

lossr�>�4��       �	R�\OYc�A�*

lossW�g>"��3       �	]OYc�A�*

loss�">^0�       �	z�]OYc�A�*

loss��>x�Ty       �	?^OYc�A�*

loss�UP>v�+       �	f�^OYc�A�*

loss�>�6�6       �	3l_OYc�A�*

lossH�z=d��       �	`OYc�A�*

loss�#�=�6Ÿ       �	�`OYc�A�*

loss��=>*8��       �	KaOYc�A�*

lossܿ>�y��       �	q�aOYc�A�*

loss��=FOlE       �	��bOYc�A�*

loss�
G>9��       �	b�cOYc�A�*

loss�F>��^       �	�0dOYc�A�*

loss �v>���       �	�dOYc�A�*

loss�8?>��O       �	�`eOYc�A�*

loss�>��;�       �	��eOYc�A�*

loss�^>5�2�       �	0�fOYc�A�*

lossS��>�K��       �	��gOYc�A�*

loss�!D>�|)       �	ihOYc�A�*

loss�0
>��3�       �	�^iOYc�A�*

lossH�j>T8        �	j2jOYc�A�*

loss��>���       �	:�jOYc�A�*

loss�l�=�S[�       �	k�kOYc�A�*

lossf+�=���j       �	U�lOYc�A�*

loss�l>���       �	�%mOYc�A�*

loss엔=~	�l       �	��mOYc�A�*

loss�>�8��       �	l\nOYc�A�*

loss�//>=��R       �	H�nOYc�A�*

loss��= ⧮       �	ޒoOYc�A�*

loss3��=�Q�p       �	R&pOYc�A�*

loss��>sM��       �	�pOYc�A�*

loss�Pk>����       �	�tqOYc�A�*

loss��>�X̐       �	�rOYc�A�*

loss��>Z�       �	ӣrOYc�A�*

loss�k�=�6�r       �	q=sOYc�A�*

loss,>���7       �	�sOYc�A�*

loss��?L��       �	�ntOYc�A�*

loss�a�>���       �	$uOYc�A�*

lossz�=��`�       �	�uOYc�A�*

loss&H�>{Up>       �	:vOYc�A�*

lossGя>;�3�       �	�vOYc�A�*

lossq��=Š��       �	�iwOYc�A�*

loss��=@d��       �	��wOYc�A�*

loss��>bEo       �	��xOYc�A�*

losshN�>3��       �	('yOYc�A�*

losso�~>��       �	��yOYc�A�*

loss�8z>�U��       �	__zOYc�A�*

loss+�>@K`�       �	.{OYc�A�*

loss�g1>�tv       �	��{OYc�A�*

loss�3>�1��       �	�6|OYc�A�*

lossc�>����       �	>�|OYc�A�*

loss�V�=\I       �	k}OYc�A�*

losss.�={)�+       �	�~OYc�A�*

loss;(O> �y�       �	�~OYc�A�*

lossA/�=G,�       �	�COYc�A�*

loss��=6�        �	W�OYc�A�*

loss� �=��       �	���OYc�A�*

loss�9�=�^�'       �	�)�OYc�A�*

lossoX>�Q�       �	�сOYc�A�*

loss���=���b       �	�u�OYc�A�*

loss��9>�y�       �	��OYc�A�*

loss�D->+k�`       �	���OYc�A�*

losss�>����       �	�I�OYc�A�*

lossϠn>e�       �	��OYc�A�*

losse�=�Z-�       �	�OYc�A�*

loss�f�='��o       �	.�OYc�A�*

losss(I=�;vP       �	�ˆOYc�A�*

loss\^�=Ȑ��       �	xc�OYc�A�*

loss��2>�[+>       �	���OYc�A�*

loss�r>��_       �	0��OYc�A�*

lossA�>f�?G       �	�?�OYc�A�*

lossx�=����       �	�݉OYc�A�*

lossJ�!>�L��       �	�v�OYc�A�*

loss� 1>����       �	��OYc�A�*

loss��>!�\       �	��OYc�A�*

lossj�=�E�       �	;V�OYc�A�*

loss$E)>��i�       �	c(�OYc�A�*

loss��D>d,�*       �	���OYc�A�*

loss�|u=���c       �	LߎOYc�A�*

loss��>�čZ       �	���OYc�A�*

loss�r*>�j       �	��OYc�A�*

loss:>=>�[�x       �	�ΐOYc�A�*

loss�x�=d�)l       �	 �OYc�A�*

lossV	�=�m��       �	���OYc�A�*

loss��=��A�       �	sK�OYc�A�*

loss�e�=��       �	\�OYc�A�*

loss�$L>2���       �	e�OYc�A�*

loss��<q��       �	#��OYc�A�*

loss���=���T       �	VE�OYc�A�*

loss;B`>u���       �	��OYc�A�*

loss�i,>�R��       �	���OYc�A�*

loss\��=�Ea        �	�?�OYc�A�*

loss�w>Zl*�       �		ݘOYc�A�*

lossf?>H~��       �	py�OYc�A�*

loss��z>�e       �	W"�OYc�A�*

loss-s�=3       �	�ȚOYc�A�*

lossMd�>���y       �	#j�OYc�A�*

lossf�#>25q       �	��OYc�A�*

lossݛ7>�_2       �	���OYc�A�*

loss	��=����       �	�V�OYc�A�*

loss[�=�H7�       �	��OYc�A�*

loss�&�=[.�       �	OYc�A�*

lossJ�B=��E       �	O@�OYc�A�*

loss�>*�_�       �	��OYc�A�*

loss-#�=q���       �	���OYc�A�*

lossj�8>z�o�       �	y;�OYc�A�*

loss��=MB�+       �	!�OYc�A�*

loss��2>U�#�       �	׈�OYc�A�*

loss�
�=1%�4       �	�1�OYc�A�*

lossΡ><���       �	kףOYc�A�*

lossE��=)ſz       �	ˁ�OYc�A�*

loss�>���Q       �	0c�OYc�A�*

loss,�=b��       �	 ��OYc�A�*

loss���=C�b       �	4G�OYc�A�*

loss�� >jǮq       �	���OYc�A�*

loss8�=KL�       �	���OYc�A�*

loss��>��'       �	�{�OYc�A�*

lossoP>�{�       �	�U�OYc�A�*

lossͪW>�a,H       �	�	�OYc�A�*

lossÊ*=��       �	���OYc�A�*

loss0��=��|)       �	m��OYc�A�*

loss�>lh�       �	:X�OYc�A�*

loss��=*�
       �	�OYc�A�*

loss�[%=U-�7       �	ٯ�OYc�A�*

loss<̓=x�6&       �	0d�OYc�A�*

loss���=���       �	�OYc�A�*

lossH��=�K+Q       �	%˰OYc�A�*

loss�r=�0�       �	�{�OYc�A�*

loss��A=�+��       �	�x�OYc�A�*

loss@f>s�?       �	��OYc�A�*

lossr�-<���       �	 гOYc�A�*

loss!u�;�@��       �	�OYc�A�*

loss��<�p       �	�2�OYc�A�*

lossRR�=S^r�       �	�ٵOYc�A�*

lossR�>�	       �		��OYc�A�*

loss(͠=��`D       �	�@�OYc�A�*

lossW��<if\       �	�OYc�A�*

losso:�=F'r�       �	|ָOYc�A�*

lossns�>�t�F       �	0��OYc�A�*

lossU@<0��       �	r1�OYc�A�*

loss�=�>֢        �	�պOYc�A�*

loss�X>oil       �	˄�OYc�A�*

lossۤ>��\       �	�'�OYc�A�*

lossY�=r���       �	l˼OYc�A�*

loss��`=       �	6x�OYc�A�*

lossі=��F�       �	�*�OYc�A�*

loss��=>��ݝ       �	0پOYc�A�*

loss�9">LF�<       �	�y�OYc�A�*

loss�	>(�       �	3�OYc�A�*

loss�]g> �9�       �	���OYc�A�*

loss��>;��       �	T�OYc�A�*

loss8A4>[���       �	���OYc�A�*

loss?>�80       �	Q��OYc�A�*

loss���>�e       �	�G�OYc�A�*

loss��>'ѿ4       �	Y��OYc�A�*

loss���=z%�w       �	z�OYc�A�*

loss���=pM��       �	�OYc�A�*

loss1��=�       �	_��OYc�A�*

loss�'�=�^�       �	�d�OYc�A�*

lossf�=�Z*u       �	L��OYc�A�*

lossf]>/�F�       �	Y��OYc�A�*

loss��B>�e�       �	@�OYc�A�*

loss*��=��O�       �	_B�OYc�A�*

lossN[�=3ۄ�       �	���OYc�A�*

lossMڞ=���       �	���OYc�A�*

loss�N?> <       �	�,�OYc�A�*

loss�C�=�l       �	��OYc�A�*

loss$>>U�^�       �	v��OYc�A�*

loss}�:>��       �	H�OYc�A�*

loss��=4��       �	z��OYc�A�*

loss��=��ؾ       �	v�OYc�A�*

loss��P>ñ       �	��OYc�A�*

loss�l8=��Y       �	W��OYc�A�*

loss�>��q}       �	�K�OYc�A�*

loss;+�=~�i.       �	9��OYc�A�*

loss_a�=)       �	���OYc�A�*

loss��=����       �	�l�OYc�A�*

loss�(>����       �	}�OYc�A�*

lossq��=s76        �	9��OYc�A�*

loss3��=�E"       �	�1�OYc�A�*

loss�yJ=�`ޑ       �	���OYc�A�*

lossXچ=�H�       �	$d�OYc�A�*

loss2�>LTE       �	�1�OYc�A�*

loss �=8k�       �	��OYc�A�*

loss٘=�R��       �	t��OYc�A�*

loss*1>���)       �	)Y�OYc�A�*

lossZ:[=�@��       �	���OYc�A�*

loss���=�d��       �	F��OYc�A�*

lossrd�=ΊeS       �	�5�OYc�A�*

loss���=5L�       �	���OYc�A�*

loss89>�(       �	/j�OYc�A�*

loss#�>quc�       �	E-�OYc�A�*

loss��>\���       �	���OYc�A�*

loss�&7>��b       �	�]�OYc�A�*

loss.�=SQ�       �	���OYc�A�*

loss�>�x�       �	��OYc�A�*

lossH��=�c��       �	l"�OYc�A�*

losssX�=b�q�       �	k��OYc�A�*

lossn�+>*Y�f       �	c}�OYc�A�*

lossx��=��}       �	��OYc�A�*

loss:��=�x��       �	=��OYc�A�*

loss���=��u+       �	Mj�OYc�A�*

loss��=f�p(       �	-�OYc�A�*

lossz>Ún       �	E��OYc�A�*

loss��#>�	Ӏ       �	�8�OYc�A�*

loss� .>�
4�       �	� PYc�A�*

loss�5=���       �	�� PYc�A�*

loss�{�=�22       �	�hPYc�A�*

loss��=�p�U       �	�PYc�A�*

loss
�>���k       �	X�PYc�A�*

loss�>�=j�]       �	�OPYc�A�*

lossN�>O��M       �	{PYc�A�*

loss���=�\$�       �	�PYc�A�*

losshK>�%��       �	�CPYc�A�*

loss,T7>�>i�       �	��PYc�A�*

loss&<�=�Jr       �	xPYc�A�*

loss΄>~��       �	� PYc�A�*

loss)�>��T       �	��PYc�A�*

lossC�>���B       �	'gPYc�A�*

lossEEd>�/��       �	E	PYc�A�*

loss�>�uE       �	��	PYc�A�*

lossv$�=�s�       �	�
PYc�A�*

lossF�=��       �	dPYc�A�*

loss1G�>���       �	��PYc�A�*

loss�ym=�a��       �	0�PYc�A�*

lossZ�_>�ϱM       �	� PYc�A�*

lossI�/=ƴw�       �	��PYc�A�*

loss�W�=��s       �	OPYc�A�*

lossTA�>�{_x       �	~�PYc�A�*

lossizR>�mN       �	qvPYc�A�*

loss��>"�`�       �	jPYc�A�*

loss�61>	��#       �	��PYc�A�*

loss?��=���       �	��PYc�A�*

lossY><��3       �	�SPYc�A�*

lossqc�=�I�:       �	l�PYc�A�*

lossd��=�l�L       �	~�PYc�A�*

lossV��=�s�M       �	��PYc�A�*

loss��=~�0        �	Q�PYc�A�*

loss]�= ��\       �	�PYc�A�*

lossX�`=���       �	ձPYc�A�*

loss�=��|       �	�KPYc�A�*

loss��%>��+       �	�PYc�A�*

loss�d�=7O       �	0-PYc�A�*

loss��>�(bv       �	~�PYc�A�*

loss�j=�"@Y       �	ZPYc�A�*

loss��6=�m,�       �	��PYc�A�*

loss��=,ZS�       �	��PYc�A�*

loss�g�=cc��       �	�PYc�A�*

lossZ
>~Ge�       �	d�PYc�A�*

loss]n>���S       �	�GPYc�A�*

loss� >�p0       �	�PYc�A�*

loss��m=�V]�       �	�wPYc�A�*

loss��=���       �	�PYc�A�*

losss�J>)A#z       �	��PYc�A�*

lossh�>M�R       �	� PYc�A�*

loss�>��x       �	G!PYc�A�*

loss��8>g�u�       �	�!PYc�A�*

loss�S>�#�g       �	�s"PYc�A�*

loss>+�>h�DB       �	�#PYc�A�*

loss���=߉3.       �	P�#PYc�A�*

lossv÷=?V�       �	�C$PYc�A�*

loss���=�-:       �	/�$PYc�A�*

lossU�=�>��       �	9�%PYc�A�*

loss�pK>���       �	�N&PYc�A�*

loss]:1>n�       �	3N'PYc�A�*

loss��=�+��       �	O�'PYc�A�*

loss�I>�٢H       �	A(PYc�A�*

loss��1>���       �	)PYc�A�*

loss���=���	       �	$�)PYc�A�*

loss�ʥ=H��;       �	O*PYc�A�*

loss�W�=��n�       �	��*PYc�A�*

loss�Te>8v*�       �	S�+PYc�A�*

lossfY�=���Z       �	�1,PYc�A�*

loss�Z�=��I       �	9�,PYc�A�*

loss�b�="'[k       �	�s-PYc�A�*

loss���=��!I       �	�.PYc�A�*

loss_l_>�D�       �	��.PYc�A�*

lossÙ�=��v�       �	�K/PYc�A�*

loss]@�=�@�       �	K�/PYc�A�*

lossDG�=$R!g       �	j�0PYc�A�*

loss/n>���&       �	�1PYc�A�*

loss� >N��        �	�1PYc�A�*

lossr�=���a       �	w�2PYc�A�*

loss��&>#��Z       �	-&3PYc�A�*

lossi�D>���       �	��3PYc�A�*

lossL��="�       �	�N4PYc�A�*

loss��W>�SAR       �	1�4PYc�A�*

lossc
>���       �	�5PYc�A�*

loss	n�>���       �	t)6PYc�A�*

loss��D>q�{       �	ǽ6PYc�A�*

lossG\="�9�       �	�U7PYc�A�*

loss�S�=ٺ�       �	g�7PYc�A�*

loss�7u>�Y�       �	<�8PYc�A�*

loss18�=�Q��       �	7�9PYc�A�*

loss
�!>�8@�       �	|D:PYc�A�*

lossa��=�ee(       �	 B;PYc�A�*

loss:�P=%/ѩ       �	�;PYc�A�*

loss/�=�eu�       �	ˁ<PYc�A�*

loss��g>��|       �	�!=PYc�A�*

loss�a>;U�       �	��=PYc�A�*

loss�|>��#       �	vp>PYc�A�*

loss���=3�S       �	=?PYc�A�*

loss|N>�/�       �	ʨ?PYc�A�*

loss�p=�浌       �	@P@PYc�A�*

loss���=1��       �	D�@PYc�A�*

lossaw�=����       �	
�APYc�A�*

loss�L>r>Qo       �	�9BPYc�A�*

loss�l>��S       �	��BPYc�A�*

losss�%>���G       �	XrCPYc�A�*

lossTK�=�f8       �	�DPYc�A�*

loss���=�y��       �	ٳDPYc�A�*

loss���=`��q       �	�OEPYc�A�*

loss��/>���       �	p�EPYc�A�*

loss���=��ա       �	��FPYc�A�*

loss���<�[�       �	d GPYc�A�*

loss��C>s���       �	1�GPYc�A�*

loss�s(>):l       �	HSHPYc�A�*

lossl >6�3�       �	��HPYc�A�*

loss��>���       �	{�IPYc�A�*

loss��>Ю��       �	vJPYc�A�*

loss0>���       �	A�JPYc�A�*

loss&�>Q��       �	JbKPYc�A�*

loss�]�=z*e[       �	+LPYc�A�*

loss ��<�B       �	��LPYc�A�*

loss�E >�"A�       �	�cMPYc�A�*

loss��=a\+=       �	�NPYc�A�*

lossѴ�=F&?
       �	�NPYc�A�*

loss���<�*�       �	bOPYc�A�*

loss�ݲ=����       �	�PPYc�A�*

loss�3�<��       �	HSQPYc�A�*

loss��=�h       �	�_RPYc�A�*

lossD9�=��$g       �	]5SPYc�A�*

loss&�>�@�t       �	/�SPYc�A�*

loss�(>G�F�       �	�TPYc�A�*

lossR�>p�,e       �	�UPYc�A�*

loss��!=�El�       �	�aVPYc�A�*

loss��=��0       �	g
WPYc�A�*

loss��x=,\&       �	��WPYc�A�*

loss�+=�@�       �	�XPYc�A�*

loss!��=Id�E       �	ZYPYc�A�*

loss80-=@r_       �	AZPYc�A�*

loss,W>��or       �	^�ZPYc�A�*

loss#�>� �~       �	��[PYc�A�*

loss���=J��       �	�:\PYc�A�*

loss��=�ݞ       �	�[]PYc�A�*

loss���<��       �	 	^PYc�A�*

loss&p�=� ��       �	�_PYc�A�*

loss�h=>L��       �	q`PYc�A�*

lossd�>\       �	Χ`PYc�A�*

lossfق=��>       �	�ZaPYc�A�*

loss��=�e�|       �	&bPYc�A�*

loss�Q�=5a}       �	�bPYc�A�*

loss�-�=r��X       �	�YcPYc�A�*

lossc��<w�.,       �	��cPYc�A�*

loss��=�3|�       �	�dPYc�A�*

loss��=��       �	�ePYc�A�*

loss�=F���       �	��fPYc�A�*

lossO�>�X�^       �	)vgPYc�A�*

loss$��==�M       �	hPYc�A�*

loss4Ne=���^       �	�ViPYc�A�*

loss�� >�>�x       �	3RjPYc�A�*

loss��=6��       �	F&kPYc�A�*

lossׇ�=�7�       �	"�kPYc�A�*

loss�'>v�0       �	�VlPYc�A�*

loss�$<=z��       �	��lPYc�A�*

loss	�=G�f       �	L�mPYc�A�*

loss��>�p��       �	�"nPYc�A�*

loss�0>�1�X       �	��nPYc�A�*

loss��r=liތ       �	�QoPYc�A�*

lossHHO=2ap�       �	�oPYc�A�*

loss��< ݎ       �	�pPYc�A�*

loss<��=$̿c       �	�%qPYc�A�*

loss��=
ܾ�       �	�qPYc�A�*

loss�
T=��JT       �	�rPYc�A�*

loss#I>��:       �	�lsPYc�A�*

loss�/
>ޑCG       �	!tPYc�A�*

lossm��=a0
       �	�tPYc�A�*

lossʰ�=�qt%       �	�EuPYc�A�*

loss,g�=Q�q       �	��uPYc�A�*

loss "�=?��       �	ovPYc�A�*

loss�e@>yj       �	� wPYc�A�*

loss��v=R�       �	B�wPYc�A�*

losse%�=c��       �	�.xPYc�A�*

loss~H�=v�؝       �	��xPYc�A�*

loss�M�=�|4�       �	�uyPYc�A�*

lossdH�=i��       �	�)zPYc�A�*

loss饫=}Y�       �	v�zPYc�A�*

loss���=��3       �	�i{PYc�A�*

loss��a>�E�       �	�|PYc�A�*

lossoZ�=w=�       �	V�|PYc�A�*

loss�'�=�X��       �	�|}PYc�A�*

loss�}�<���
       �	b~PYc�A�*

loss��=�u       �	��~PYc�A�*

loss�ކ=WQ	�       �	[PYc�A�*

loss�b=����       �	x�PYc�A�*

loss��9=,���       �	U��PYc�A�*

loss
t5>9��c       �	+�PYc�A�*

loss�=���       �	ρPYc�A�*

loss���=Pw�\       �	g�PYc�A�*

lossd>�=�-��       �	��PYc�A�*

loss_��=�5�       �	P��PYc�A�*

lossd>YPmg       �	$��PYc�A�*

loss$aV=M��       �	�3�PYc�A�*

loss��= Y�       �	lͅPYc�A�*

loss�&�=�s�       �	4i�PYc�A�*

loss���=WH,n       �	$�PYc�A�*

lossdq�>&7�       �	2��PYc�A�*

loss�N�>�R��       �	eP�PYc�A�*

loss��>�5�       �	R�PYc�A�*

loss�@d>�{�       �	Y��PYc�A�*

loss�6G= e��       �	� �PYc�A�*

loss#d�<I�9
       �	4��PYc�A�*

loss��>���       �	X�PYc�A�*

loss��]>�+d_       �	f��PYc�A�*

lossR�<��=�       �	o��PYc�A�*

loss6>�=�0�       �	X9�PYc�A�*

loss��?>�P�l       �	�ۍPYc�A�*

lossO#>Vۗ       �	���PYc�A�*

lossI^�=ʫ�&       �	V*�PYc�A�*

loss���=25T�       �	c%�PYc�A�*

lossʹ(=尉i       �	ƐPYc�A�*

lossܞ5=��\O       �	�k�PYc�A�*

loss4f>/�)�       �	��PYc�A�*

loss�r�=rR��       �	_��PYc�A�*

loss&�^>�b�~       �	�Z�PYc�A�*

loss^>0C��       �	���PYc�A�*

loss�u�=L�       �	Ĕ�PYc�A�*

loss�9,>�M$       �	a2�PYc�A�*

loss��$>~�L{       �	�PYc�A�*

lossHe�=�t�       �	p��PYc�A�*

loss���=?V�       �	=I�PYc�A�*

loss�h�=�f�       �	�PYc�A�*

loss��>A̺�       �	��PYc�A�*

loss���=ѩz�       �	v�PYc�A�*

loss�F2>��ʳ       �	���PYc�A�*

loss�.F>�#i~       �	al�PYc�A�*

loss{g=���:       �	=�PYc�A�*

lossS=�"       �	+��PYc�A�*

loss��=<��U       �	�I�PYc�A�*

loss�_�=�v�       �	��PYc�A�*

loss���>�G]�       �	���PYc�A�*

lossMt>yU�W       �	�#�PYc�A�*

loss�`�=(c�!       �	�PYc�A�*

loss:�[=&���       �	�^�PYc�A�*

loss��=̸�       �	���PYc�A�*

lossx�=6�X       �	٘�PYc�A�*

loss�Z>��(�       �	m:�PYc�A�*

lossf�>�ψv       �	��PYc�A�*

loss�=�&�       �	��PYc�A�*

loss&>�g�       �	�3�PYc�A�*

loss�p�=�h�       �	�̣PYc�A�*

loss�:�=^{��       �	�i�PYc�A�*

loss�=>�       �	S�PYc�A�*

loss�h=�g��       �	)��PYc�A�*

loss�.>|C�0       �	���PYc�A�*

loss6X�=n�x.       �	���PYc�A�*

loss�E>2f=s       �	�7�PYc�A�*

lossF]>��       �	FͨPYc�A�*

loss�\m>���[       �	vm�PYc�A�*

loss��=���d       �	�PYc�A�*

loss.z�=cyl�       �	�ʪPYc�A�*

losseS�=�r��       �	h�PYc�A�*

loss��=��T       �	���PYc�A�*

loss��>n}�a       �	䣬PYc�A�*

lossm>q��       �	�F�PYc�A�*

loss��Q>a�       �	��PYc�A�*

loss���=J¤�       �	���PYc�A�*

loss�w=����       �	�Q�PYc�A�*

loss��=BQg       �	��PYc�A�*

loss
'>i:�       �	K��PYc�A�*

loss��=I�w�       �	*�PYc�A�*

lossE�M=��5~       �	"²PYc�A�*

lossi\�=HUr       �	��PYc�A�*

lossS��=��1�       �	W'�PYc�A�*

loss0>be�       �	�ɴPYc�A�*

loss ��= �Y�       �	vm�PYc�A�*

loss��=��^�       �	�PYc�A�*

loss
W=t�       �	:��PYc�A�*

loss�J�=hG!=       �	#.�PYc�A�*

loss(��<���       �	YķPYc�A�*

loss�� =�![       �	c`�PYc�A�*

loss�j�=��,\       �	`W�PYc�A�*

loss6�Y=���J       �	��PYc�A�*

loss��c=/���       �	���PYc�A�*

lossi�B=;Ih�       �	�)�PYc�A�*

lossMG.>�R��       �	���PYc�A�*

losssR�=� (       �	�X�PYc�A�*

loss��+>���       �	��PYc�A�*

lossܵ4>��       �	U��PYc�A�*

loss0>MC2*       �	�PYc�A�*

loss���=j��\       �	`��PYc�A�*

loss7Y8=�mV       �	t�PYc�A�*

loss�H�=D��       �	4�PYc�A�*

loss��=Y�}�       �	֨�PYc�A�*

loss�J�=��       �	2;�PYc�A�*

loss��/>.O�+       �	���PYc�A�*

loss
=�:l�       �	\r�PYc�A�*

loss
'D>^��       �	D�PYc�A�*

lossC�=ǀ/	       �	̴�PYc�A�*

loss�<>�L�       �	B_�PYc�A�*

lossT>�E�"       �	j��PYc�A�*

loss�}'>��;       �	v��PYc�A�*

lossZ�>���>       �	�-�PYc�A�*

loss�s=Lu�       �	���PYc�A�*

loss�,=&�)N       �	qY�PYc�A�*

loss�X>��}�       �	���PYc�A�*

loss<<�=��͡       �	���PYc�A�*

loss�o=/�.t       �	M/�PYc�A�*

loss��>�FTd       �	���PYc�A�*

loss�,�=�Gu       �	K\�PYc�A�*

loss)�w=p[�       �	P �PYc�A�*

loss��`>�-��       �	���PYc�A�*

loss���=4{       �	c+�PYc�A�*

loss�ԡ=7u�q       �	��PYc�A�*

lossi2>Ҡ^�       �	�X�PYc�A�*

loss?{�>�{       �	���PYc�A�*

loss�\�=z��       �	v��PYc�A�*

loss��B=�~:�       �	%�PYc�A�*

loss�\�=�T�V       �	���PYc�A�*

loss�r>>���       �	���PYc�A�*

loss�>�r�#       �	�{�PYc�A�*

loss��/=��_�       �	PR�PYc�A�*

loss�^=Ƴj+       �	���PYc�A�*

loss�I=�r       �	��PYc�A�*

loss]�r>���C       �	��PYc�A�*

loss���=���       �	��PYc�A�*

loss�U�=Xp#       �	���PYc�A�*

lossT>��!�       �	>[�PYc�A�*

losstt>a��'       �	��PYc�A�*

loss�}=&�*       �	d��PYc�A�*

loss�Wb=9+�       �	�k�PYc�A�*

lossֲ�=��}�       �	P�PYc�A�*

loss��>���       �	���PYc�A�*

loss���=��R�       �	�i�PYc�A�*

loss�f�>vM*Q       �	��PYc�A�*

lossl%x>ye�       �	���PYc�A�*

loss��+=�{�C       �	�O�PYc�A�*

loss��e>��       �	E��PYc�A�*

loss%d�=�yۧ       �	ɓ�PYc�A�*

loss��=�K}       �	|*�PYc�A�*

lossTP�<"��       �	;��PYc�A�*

loss�ڤ=��r$       �	�k�PYc�A�*

losshU�=��Fk       �	A�PYc�A�*

lossһ�=�*�       �	%��PYc�A�*

lossi��=���       �	�J�PYc�A�*

loss��=0�ǔ       �	���PYc�A�*

loss�JO=K�e�       �	��PYc�A�*

loss��=��A�       �	�V�PYc�A�*

loss=��=��B�       �	���PYc�A�*

loss&�=yl�       �	ԝ�PYc�A�*

losss�> �5,       �	;�PYc�A�*

lossh�x=Vu       �	���PYc�A�*

loss�"�=��[1       �	S!�PYc�A�*

lossv�>����       �	���PYc�A�*

loss �\>P���       �	��PYc�A�*

loss��>���6       �	���PYc�A�*

loss��=��c       �	�3�PYc�A�*

losssK�=�-k'       �	��PYc�A�*

loss��4=:j       �	�{�PYc�A�*

loss8\=e_ƕ       �	%�PYc�A�*

loss�!Y=<�Ʋ       �	��PYc�A�*

loss�>��Ln       �	�p�PYc�A�*

lossÎb=<Z��       �	��PYc�A�*

loss��=�d'�       �	���PYc�A�*

loss<$.>M�1�       �	h�PYc�A�*

lossMm�=x���       �	��PYc�A�*

loss;�R>aR��       �	ڪ�PYc�A�*

loss,� >�t��       �	\u�PYc�A�*

loss�{�=�;�       �	��PYc�A�*

loss��>�/�       �	��PYc�A�*

loss�=�=_w�       �	9b�PYc�A�*

loss��=�Q�       �	T�PYc�A�*

lossH�>��$�       �	���PYc�A�*

loss4��=8~J�       �	�A�PYc�A�*

loss(�= fQ�       �	��PYc�A�*

loss���=�;�       �	/��PYc�A�*

loss\/�=r��       �	�"�PYc�A�*

loss���=�U�       �	���PYc�A�*

lossc/j=\��       �	���PYc�A�*

loss��'>�|6       �	�6�PYc�A�*

loss�E
>٩g�       �	���PYc�A�*

loss\��=���       �	er�PYc�A�*

loss�E�==N�       �	�
�PYc�A�*

loss��=�x B       �	ٲ�PYc�A�*

loss�}=���!       �	nO�PYc�A�*

lossi-7>}D        �	T��PYc�A�*

lossE8p=�L)       �	��PYc�A�*

loss�}z=�6#K       �	8/�PYc�A�*

loss�X={��       �	"��PYc�A�*

loss��=����       �	)^�PYc�A�*

loss��<=��       �	���PYc�A�*

loss">0�?@       �	χ QYc�A�*

loss��>5��       �	mQYc�A�*

loss�̰=D!       �	�QYc�A�*

loss��d=s0�       �	jPQYc�A�*

lossԡD=x"yQ       �	��QYc�A�*

loss��
>�D�       �	�QYc�A�*

loss
��=�<p�       �	�,QYc�A�*

loss��}=I�!�       �	m�QYc�A�*

loss�3�= z       �	
dQYc�A�*

loss�=Fܑ�       �	.QYc�A�*

loss�+>��       �	��QYc�A�*

lossa=�=�A       �	�;QYc�A�*

lossܪG>�uD�       �	��QYc�A�*

lossE�B><>4A       �	r�QYc�A�*

loss��>p�TE       �	�$	QYc�A�*

loss�#	=b_5�       �	Y�	QYc�A�*

loss��<��       �	�t
QYc�A�*

loss��>��P�       �	�QYc�A�*

lossX<">?2Ae       �	E/QYc�A�*

loss�a=*Cd       �	S�QYc�A�*

lossZ�=;�0       �	znQYc�A�*

loss��=6��M       �	�QYc�A�*

loss��=>�=?       �	^�QYc�A�*

loss�o7>da��       �	�?QYc�A�*

lossÍI>���       �	��QYc�A�*

loss��=ws�       �	U�QYc�A�*

lossM�>d�7       �	x�QYc�A�*

loss��==����       �	+mQYc�A�*

loss�k=��qC       �	�FQYc�A�*

loss�3>����       �	�!QYc�A�*

loss��=|M��       �	 �QYc�A�*

lossl�=�H:�       �	}uQYc�A�*

loss��=�Q�       �	QYc�A�*

loss-��=N&%f       �	��QYc�A�*

lossQ
=���       �	�cQYc�A�*

loss7r�=I�4       �	`QYc�A�*

loss��I=Y���       �	��QYc�A�*

loss�R=s�~       �	*TQYc�A�*

losst�=K�|&       �	��QYc�A�*

loss|�0>F=�g       �	L�QYc�A�*

loss)1i>�K_�       �	mWQYc�A�*

loss{�1=���:       �	z�QYc�A�*

loss�0>���m       �	¥QYc�A�*

lossYJ�=��>       �	�VQYc�A�*

lossb�=�@�U       �	��QYc�A�*

lossjO�>'s?x       �	 �QYc�A�*

loss��='�"       �	�9QYc�A�*

lossϮ=4>>       �	��QYc�A�*

loss��Q>��
�       �	�� QYc�A�*

lossz�>�Fձ       �	�8!QYc�A�*

loss�v=T- �       �	Q�!QYc�A�*

loss�wj=\�       �	Ί"QYc�A�*

lossiG�=#�n�       �	cA#QYc�A�*

loss�L>@��       �	�#QYc�A�*

loss(�>f�=%       �	�$QYc�A�*

loss�>?>��Z       �	K:%QYc�A�*

loss��c=p���       �	��%QYc�A�*

loss2�=��K       �	:�&QYc�A�*

loss;��=����       �	��'QYc�A�*

lossL-t=���M       �	�k(QYc�A�*

loss�P$=��!�       �	0�)QYc�A�*

loss�y�=�o�       �	�*QYc�A�*

lossSE>X��       �	�v+QYc�A�*

loss@��=�[�       �	�8,QYc�A�*

loss�c#=�_��       �	k�,QYc�A�*

loss�|(=�_�U       �	��-QYc�A�*

loss�n�=$�       �	�!.QYc�A�*

loss즄=Bh��       �	��.QYc�A�*

loss��<�s�       �	�K/QYc�A�*

lossf�J>��\       �	J�/QYc�A�*

loss���=a��N       �	��0QYc�A�*

loss}�>đ6�       �	1QYc�A�*

loss?��=)-�V       �	��1QYc�A�*

loss+�=a�F�       �	��2QYc�A�*

loss�l�<�ӱ       �	��3QYc�A�*

lossg��=�CU^       �	�4QYc�A�*

loss���=k�       �	b�4QYc�A�*

loss��h=?w2�       �	�^5QYc�A�*

loss�R>R�P�       �	 6QYc�A�*

loss*�>�Jk�       �	s�6QYc�A�*

loss)c�=� E�       �	�V7QYc�A�*

loss��=yUa`       �	��7QYc�A�*

lossw!�=���       �	��8QYc�A�*

loss(��=Dɴ�       �	�!9QYc�A�*

loss��,=B'��       �	,�9QYc�A�*

loss�MW=�u��       �	є:QYc�A�*

loss->�_�       �	�*;QYc�A�*

loss�͇=(a�       �	�<QYc�A�*

loss��A>���       �	|�<QYc�A�*

loss�D�=����       �	D5=QYc�A�*

loss��>�l��       �	��=QYc�A�*

loss;Z�<~dW        �	9c>QYc�A�*

lossZ =��0Y       �	�>QYc�A�*

loss}ab=g��       �	�?QYc�A�*

loss�T=�ꩢ       �	L@QYc�A�*

loss��>����       �	`�@QYc�A�*

loss�G=�9�<       �	x}AQYc�A�*

loss`�=���\       �	BQYc�A�*

loss���=��b       �	�BQYc�A�*

loss
;�=��b       �	�VCQYc�A�*

loss��=�p��       �	��CQYc�A�*

lossZp)=Fu       �	��DQYc�A�*

loss�$=�|)�       �	6?EQYc�A�*

loss.�$>���W       �	�EQYc�A�*

loss 4�=�F�       �	}FQYc�A�*

loss�(�=DUN9       �	&GQYc�A�*

loss!?T=uD�7       �	S�GQYc�A�*

loss[>��/�       �	�MHQYc�A�*

lossT��=3TK       �	g�HQYc�A�*

loss���=cs       �	.�IQYc�A�*

loss`q�=$Sc�       �	�4JQYc�A�*

loss��<\���       �	6�JQYc�A�*

loss��='ῇ       �	�bKQYc�A�*

loss-�=X���       �	��KQYc�A�*

loss��=Ca�t       �	��LQYc�A�*

loss�=@��U       �	o-MQYc�A�*

loss���=Хˡ       �	�MQYc�A�*

loss�ʂ=mSB�       �	�ZNQYc�A�*

loss)��=ix݇       �	��NQYc�A�*

loss�$�=�4�       �	��OQYc�A�*

loss/�=T�)�       �	�!PQYc�A�*

loss���<��4�       �	��PQYc�A�*

loss��=���       �	\YQQYc�A�*

loss$��=D<��       �	��QQYc�A�*

loss��,=In��       �	0�RQYc�A�*

lossɔ�=��       �	�=SQYc�A�*

loss�>�I9�       �	��SQYc�A�*

lossx�2>���       �	'�TQYc�A�*

loss�� =M���       �	b1UQYc�A�*

loss�h<MD��       �	��UQYc�A�*

loss IM>�d@�       �	�vVQYc�A�*

losse4N=���+       �	VWQYc�A�*

lossR<d���       �	��WQYc�A�*

loss�%7<.u&       �	�NXQYc�A�*

loss��9=����       �	��XQYc�A�*

lossm0<LT.       �	�YQYc�A�*

loss8��<W�u�       �	p%ZQYc�A�*

loss��=�"�       �	��ZQYc�A�*

loss"܇>Y�k�       �	�W[QYc�A�*

loss��;*7��       �	��[QYc�A�*

lossz�<��8       �	��\QYc�A�*

loss2��;�{��       �	
,]QYc�A�*

loss���=?2�R       �	F�]QYc�A�*

losso)>���)       �	fk^QYc�A�*

loss!�A=�b�*       �	G_QYc�A�*

loss:<d��V       �	B�_QYc�A�*

loss���=ڟlk       �	�8`QYc�A�*

loss���>�&�5       �	<�`QYc�A�*

loss$�<�1�J       �	CsaQYc�A�*

loss&98>g�O       �	�!bQYc�A�*

loss$��=�1x       �	=�bQYc�A�	*

loss{0J>��m[       �		RcQYc�A�	*

loss��=��[�       �	p�cQYc�A�	*

loss=�<=���h       �	U�dQYc�A�	*

lossv�`=M��       �	�&eQYc�A�	*

loss�[�="��       �	.�eQYc�A�	*

loss�}�=+Q;�       �	�afQYc�A�	*

loss���=��U7       �	��fQYc�A�	*

lossq*�=B�]       �	\�gQYc�A�	*

loss�>%�C�       �	4hQYc�A�	*

lossn�7>m�-z       �	�hQYc�A�	*

loss;�>Nnw       �	�niQYc�A�	*

loss��=
�y�       �	jQYc�A�	*

loss*�>���G       �	a�jQYc�A�	*

loss$G=M��(       �	�AkQYc�A�	*

loss�=׹�       �	�kQYc�A�	*

loss.��=�Iw�       �	�tlQYc�A�	*

loss�Z�=���[       �	�mQYc�A�	*

lossA,�<cG{%       �	v�mQYc�A�	*

loss���=L
�b       �	&QnQYc�A�	*

lossW��=��       �	��nQYc�A�	*

loss��<�%��       �	N�oQYc�A�	*

loss�O�<��"�       �	)@pQYc�A�	*

lossQ�<0�t�       �	��pQYc�A�	*

loss���=���a       �	qQYc�A�	*

loss�6/=1VE�       �	 rQYc�A�	*

loss���=�~�       �	b�rQYc�A�	*

loss� �=HT�-       �	�dsQYc�A�	*

loss:!�<� 2       �	6;tQYc�A�	*

loss�D�=�       �	��tQYc�A�	*

loss���=Pz�       �	e�uQYc�A�	*

loss���<�*?       �	�+vQYc�A�	*

loss
?D=S�n       �	��vQYc�A�	*

lossLmg=����       �	��wQYc�A�	*

lossPK=U>�       �	7�xQYc�A�	*

loss2p0=DZPL       �	b/yQYc�A�	*

loss���=yI�       �	Y�yQYc�A�	*

lossFp�=`:a(       �	zQYc�A�	*

loss���<��9'       �	�.{QYc�A�	*

loss���<v�=       �	D�{QYc�A�	*

loss�T�=GX>G       �	W�|QYc�A�	*

loss�e=��K�       �	�G}QYc�A�	*

lossz�=~{<       �	��}QYc�A�	*

loss��`=�O[U       �	<�~QYc�A�	*

lossOX�=(�       �	�ĀQYc�A�	*

loss�e�<�/�+       �	�f�QYc�A�	*

loss� >�e�!       �	d�QYc�A�	*

loss=Jh=��E       �	���QYc�A�	*

loss��M=H]�       �	�{�QYc�A�	*

loss>��       �	�\�QYc�A�	*

loss�t�=�Ԏ#       �	���QYc�A�	*

lossP�>��#�       �	S��QYc�A�	*

lossFF�=�)��       �	�5�QYc�A�	*

loss�_�={�       �	,מQYc�A�	*

lossRV�=�i{f       �	mt�QYc�A�	*

loss���=�6��       �	��QYc�A�	*

loss�R�=eD�s       �	�۠QYc�A�	*

loss��=0�W�       �	�v�QYc�A�	*

loss�7�=�C	�       �	4g�QYc�A�	*

loss�T�<QA�M       �	\�QYc�A�	*

lossȃ!=�~)6       �	���QYc�A�	*

lossJ��=#�       �	q9�QYc�A�	*

loss'={�       �	�ӤQYc�A�	*

lossmq=cp��       �	Z��QYc�A�	*

loss �">��V       �	��QYc�A�	*

loss�
A<<��       �	�V�QYc�A�	*

lossx��=���w       �	���QYc�A�	*

loss��<�H�y       �	N��QYc�A�	*

loss�>5r��       �	�M�QYc�A�	*

loss,,�=g��       �	���QYc�A�	*

loss�$m>�`M�       �	���QYc�A�	*

loss?��=�!d�       �	oG�QYc�A�	*

lossq~�=�|f       �	��QYc�A�	*

loss&��=p��2       �	���QYc�A�	*

loss4�=(rj�       �	�'�QYc�A�	*

loss&Q�=�X�J       �	�ƭQYc�A�	*

loss��<,<       �	�t�QYc�A�	*

loss�X�=K{�       �	;7�QYc�A�	*

lossh�=[�x       �	[ӯQYc�A�	*

losswG�=	TS       �	~p�QYc�A�	*

loss΁�<��@       �	�!�QYc�A�	*

lossRM�=T�       �	�ͱQYc�A�	*

loss���=����       �	 o�QYc�A�	*

loss�"�=]ְ�       �	0�QYc�A�	*

lossZ`8>C��5       �	>��QYc�A�	*

lossk�=� �y       �	�U�QYc�A�	*

loss���=�~I       �	/��QYc�A�	*

lossv!�=(��t       �	��QYc�A�	*

loss���=�(Xg       �	䠶QYc�A�	*

loss�C�=}�       �	PR�QYc�A�	*

loss4��=
�T9       �	e��QYc�A�	*

losswՅ<��U       �	��QYc�A�	*

loss5�=/y݌       �	.X�QYc�A�	*

loss�y>��       �	W�QYc�A�	*

lossq_>��       �	���QYc�A�	*

loss��=Vxuc       �	�g�QYc�A�	*

lossmmZ=L�       �	��QYc�A�	*

loss^�=Z�       �	׽�QYc�A�	*

loss;��<"]f�       �	�f�QYc�A�	*

loss)=$#�       �	1�QYc�A�	*

loss�E�=B-a6       �	���QYc�A�	*

lossJ �=��       �	�]�QYc�A�	*

loss���>k�c       �	=+�QYc�A�	*

loss�;c=l� �       �	>��QYc�A�	*

loss6S�<�w��       �	�r�QYc�A�	*

loss�t�<ݡ�       �	��QYc�A�	*

loss���<&��       �	���QYc�A�	*

loss7=�̹       �	�a�QYc�A�	*

loss��R=�>I       �	3��QYc�A�	*

loss��=��B�       �	Й�QYc�A�	*

loss���=]�[�       �	C8�QYc�A�	*

loss*�+=tĖ�       �	��QYc�A�	*

loss��>�H�P       �	 }�QYc�A�	*

loss-�=���x       �	:!�QYc�A�	*

losssHQ=�rQ       �	/��QYc�A�	*

loss<r�=�U�q       �	se�QYc�A�	*

lossӉ�=*<��       �	�QYc�A�	*

lossM!�==B�       �	t��QYc�A�	*

loss�B>@7�       �	-_�QYc�A�	*

loss`�|=	:��       �	%�QYc�A�	*

loss+2�=�!Y       �	F"�QYc�A�	*

loss� [=r�ץ       �	���QYc�A�	*

loss�Jg=>{v       �	���QYc�A�	*

loss�[.=���K       �	��QYc�A�	*

loss���=�HJ       �	V�QYc�A�	*

loss��>U'��       �	��QYc�A�	*

loss4��=�โ       �	(��QYc�A�	*

loss8�=�y�       �	6V�QYc�A�	*

loss[�=���L       �	9&�QYc�A�
*

loss���=�Ju�       �	���QYc�A�
*

loss�Uh>%�       �	Ii�QYc�A�
*

loss���=�\˦       �	��QYc�A�
*

loss��=QZ԰       �	w��QYc�A�
*

loss���=rb>�       �	�4�QYc�A�
*

loss��=h�/       �	���QYc�A�
*

lossa�R=�gn�       �	�g�QYc�A�
*

loss�0[=r�4       �	��QYc�A�
*

losshյ=�<�M       �	0��QYc�A�
*

lossO:�=QA7f       �	T9�QYc�A�
*

lossŌ�=A��       �	���QYc�A�
*

lossJl�=��2       �	�h�QYc�A�
*

loss�(=��       �	O�QYc�A�
*

loss��=[[ݹ       �	N��QYc�A�
*

loss���=�c       �	�0�QYc�A�
*

loss��@=h�b�       �	���QYc�A�
*

lossMHh=<���       �	�b�QYc�A�
*

loss]ȅ=���F       �		��QYc�A�
*

loss�D�>��       �	)��QYc�A�
*

lossT��=i��       �	�L�QYc�A�
*

lossd=��)       �	� �QYc�A�
*

loss}=�{�       �	Y��QYc�A�
*

lossq�=IC�       �	�B�QYc�A�
*

loss{�<�*$�       �	��QYc�A�
*

loss��=x{%       �	���QYc�A�
*

lossa�w>�]�n       �	1#�QYc�A�
*

loss	�<�9f2       �	 ��QYc�A�
*

loss���=Bh5       �	�g�QYc�A�
*

loss��?>�i�J       �	��QYc�A�
*

lossѦ�=�Vl�       �	���QYc�A�
*

loss�'>,�Q5       �	�8�QYc�A�
*

loss;4=�N"�       �	[��QYc�A�
*

loss��=5�ȳ       �	�k�QYc�A�
*

loss{��<Ȼ�u       �	C�QYc�A�
*

lossJ$�=�       �	f��QYc�A�
*

loss�d�=6.!#       �	�;�QYc�A�
*

loss��=���u       �	$��QYc�A�
*

loss*�4>�+*Q       �	�m�QYc�A�
*

loss>|�       �	�)�QYc�A�
*

lossA�<R5�       �	`��QYc�A�
*

loss�֟=h7�h       �	�a�QYc�A�
*

lossX�=^.vM       �	s,�QYc�A�
*

loss_�=.���       �	���QYc�A�
*

loss<��=��$n       �	�a�QYc�A�
*

lossq`<�`�       �	���QYc�A�
*

loss̷3=�>��       �	X��QYc�A�
*

loss��>���       �	�A�QYc�A�
*

loss}��=��)�       �	���QYc�A�
*

lossv��=��B�       �	#��QYc�A�
*

lossA+�=����       �	`<�QYc�A�
*

loss3�=����       �	���QYc�A�
*

lossT
�=k��       �	�k�QYc�A�
*

loss���=A涑       �	�J�QYc�A�
*

loss�b=�3x�       �	C��QYc�A�
*

loss��3>:��	       �	���QYc�A�
*

lossT�=]u�A       �	�#�QYc�A�
*

lossfn�=�J�M       �	<��QYc�A�
*

loss�){=/ѡ�       �	ٔ�QYc�A�
*

lossº<�]@2       �	�:�QYc�A�
*

loss��<��e       �	���QYc�A�
*

lossF�R<O~}       �	�r�QYc�A�
*

lossLU�=��#       �	V�QYc�A�
*

loss̱�=�4�       �	ͫ�QYc�A�
*

loss�>9�9�       �	�I�QYc�A�
*

loss�>:��       �	���QYc�A�
*

loss�"�<}YU�       �	���QYc�A�
*

loss47�=���)       �	E.�QYc�A�
*

lossxs<��       �	=��QYc�A�
*

loss80�<��(`       �	�l�QYc�A�
*

loss;��=�V#�       �	h�QYc�A�
*

loss{�<k�^       �	��QYc�A�
*

loss��>�f       �	���QYc�A�
*

loss�s�=գ       �	W> RYc�A�
*

loss�-�=���#       �	$RYc�A�
*

loss�W�=z�nG       �	�RYc�A�
*

lossԖ�<��-       �	9`RYc�A�
*

loss _�=߹{�       �	RRYc�A�
*

loss7�=anh       �	��RYc�A�
*

loss��^>���B       �	ׅRYc�A�
*

loss�g<MLD�       �	�&RYc�A�
*

lossww�=�7D�       �	��RYc�A�
*

loss�x\=�:�f       �	�kRYc�A�
*

lossfqR=�+0�       �	 RYc�A�
*

loss���<��       �	(�RYc�A�
*

loss�O=?�X�       �	�9RYc�A�
*

lossl&�<�.��       �	�RYc�A�
*

loss��<Ԃ\�       �	Do	RYc�A�
*

loss�f�=��
�       �	�
RYc�A�
*

loss߇�<�S       �	�
RYc�A�
*

lossc�=Jߙ       �	�[RYc�A�
*

loss���=CQ։       �	RYc�A�
*

loss29=��M8       �	��RYc�A�
*

loss-h�=A���       �	�?RYc�A�
*

loss��=�ö       �	��RYc�A�
*

loss���<�'�2       �	�RYc�A�
*

loss���=B�j'       �	m�RYc�A�
*

lossh��=6�w       �	�DRYc�A�
*

loss�.>�W       �	Q�RYc�A�
*

lossn(�=
-       �	�vRYc�A�
*

loss�<+"P*       �	�$RYc�A�
*

loss���;�T��       �	3�RYc�A�
*

loss=�^=���       �	˝RYc�A�
*

loss�<=����       �	n�RYc�A�
*

loss*�<n� �       �	WRYc�A�
*

loss]fb<(`       �	Y�RYc�A�
*

loss���<p�       �	��RYc�A�
*

loss�з=X�d�       �	y�RYc�A�
*

lossv�v=��w       �	�*RYc�A�
*

lossL/'=�6�;       �	�%RYc�A�
*

lossh�T=�tQ�       �	&URYc�A�
*

loss���=f<ֽ       �	�
RYc�A�
*

loss}�<RѮ�       �	r�RYc�A�
*

lossҁ�<t�'U       �	CtRYc�A�
*

loss�d�</��       �	
HRYc�A�
*

lossA��<A�	:       �	c%RYc�A�
*

loss 8=��.       �	�RYc�A�
*

loss)��=�       �	�RYc�A�
*

loss�]}=G�{:       �	a� RYc�A�
*

loss��=���W       �	-�!RYc�A�
*

loss��=ӽ�~       �	�"RYc�A�
*

loss���=�k�u       �	�#RYc�A�
*

loss��+=�2P'       �	i$RYc�A�
*

loss�q=߼D)       �	J`%RYc�A�
*

loss�R=(��       �	��'RYc�A�
*

loss�Z=F�N"       �	�v(RYc�A�
*

loss׿�=�CȀ       �	�)RYc�A�
*

lossa��=��       �	��)RYc�A�
*

loss�3�=����       �	M*RYc�A�*

loss
�=-�       �	��*RYc�A�*

loss�M�=l�-�       �	?�+RYc�A�*

loss�S=��V�       �	�,,RYc�A�*

loss1X�=<Y��       �	�".RYc�A�*

loss�q=�ov       �	��.RYc�A�*

lossRY�<j�(�       �	�d/RYc�A�*

lossT�=ԡ�       �	��/RYc�A�*

loss�(�=���*       �	0�0RYc�A�*

lossH=V>��C�       �	3R1RYc�A�*

lossj�>lޙ=       �	y2RYc�A�*

loss]�4>���       �	J�2RYc�A�*

lossq��>�΋       �	ݴ3RYc�A�*

lossȩ�=^{��       �	�[4RYc�A�*

loss��5=�ݒ       �	��4RYc�A�*

loss��>��y'       �	��5RYc�A�*

loss_�>�v��       �	U16RYc�A�*

lossO]�=z�#�       �	O�6RYc�A�*

loss6!Z=��>�       �	�i7RYc�A�*

lossn��=u���       �	8RYc�A�*

lossA�=����       �	��8RYc�A�*

loss(��=b��       �	�L9RYc�A�*

loss�R$>��My       �	��9RYc�A�*

loss� =����       �	��:RYc�A�*

loss��<`R�       �	F#;RYc�A�*

lossM'>Ex       �	Ӽ;RYc�A�*

loss:��=�E
�       �	�V<RYc�A�*

loss��=|��i       �	:�<RYc�A�*

lossPǆ=��U       �	��=RYc�A�*

loss\g	>�!�       �	�>RYc�A�*

losss�=#f�       �	��>RYc�A�*

lossX�=�/       �	M?RYc�A�*

lossؓ"=��u�       �	��?RYc�A�*

loss�-=d�ˍ       �	4�@RYc�A�*

lossF"`=6Y��       �	4/ARYc�A�*

loss���=.�|9       �	2�ARYc�A�*

lossd�A=N��       �	��BRYc�A�*

loss@�=��       �	�@CRYc�A�*

loss��=��G       �	��CRYc�A�*

loss��v<�y��       �	��DRYc�A�*

lossԙq=��\       �	�ERYc�A�*

loss���=r��l       �	R�ERYc�A�*

losss�=�؆       �	�NFRYc�A�*

loss �=��6�       �	��FRYc�A�*

loss��=�sj;       �	ՓGRYc�A�*

loss��=��?T       �	I.HRYc�A�*

loss���<;��       �	~�HRYc�A�*

loss�<}_��       �	�YIRYc�A�*

loss3�/=�şr       �	��IRYc�A�*

loss��=��ؠ       �	��JRYc�A�*

loss�
�=�t{�       �	l$KRYc�A�*

loss��=n6       �	u�KRYc�A�*

loss1��=��i.       �	geLRYc�A�*

loss��=��6�       �	��LRYc�A�*

loss��u=a�:�       �	p�MRYc�A�*

lossެ>c��v       �	<NRYc�A�*

lossr��=��I�       �	��NRYc�A�*

loss��h=M�&       �	"qORYc�A�*

loss��=eR(�       �	B
PRYc�A�*

loss���=iLv[       �	��PRYc�A�*

loss;�=� �y       �	)@QRYc�A�*

lossL�4>����       �	��QRYc�A�*

loss���=!B�       �	n�RRYc�A�*

loss� ==;�h        �	Y6SRYc�A�*

lossRR~=�o       �	��SRYc�A�*

loss͓=:�       �	yxTRYc�A�*

loss��=I2�d       �	;:URYc�A�*

loss�6>�K�3       �	M�URYc�A�*

loss˰=x�y7       �	�rVRYc�A�*

loss~א=ͅ�       �	�
WRYc�A�*

loss ��=��)       �	D�WRYc�A�*

loss��=�<��       �	�?XRYc�A�*

lossO(�<I�\�       �	�XRYc�A�*

loss2�;����       �	<�YRYc�A�*

loss�P=��q�       �	�%ZRYc�A�*

loss{��=4m
�       �	z�ZRYc�A�*

loss��=�;!�       �	�[[RYc�A�*

loss���=v��       �	Q�[RYc�A�*

loss		�=��c       �	��\RYc�A�*

loss��=����       �	28]RYc�A�*

loss��=n��       �	��]RYc�A�*

loss7�s=L^�       �	u^RYc�A�*

loss�=D�j6       �	U_RYc�A�*

loss싕<�o�       �	Ӽ_RYc�A�*

lossYġ=Fdd�       �	mW`RYc�A�*

loss�@�<�Ж?       �	K�`RYc�A�*

lossl:=]֊        �	 }aRYc�A�*

loss6oS=0���       �	~bRYc�A�*

loss���=P#>       �	x�bRYc�A�*

loss��=���~       �	�_cRYc�A�*

loss��=D�N       �	�dRYc�A�*

loss�X/> A��       �	��dRYc�A�*

loss��.>�U]M       �	75eRYc�A�*

loss:�'=R�
|       �	M�eRYc�A�*

loss
�=�k9       �	�rfRYc�A�*

loss�s�=�-7:       �	gRYc�A�*

loss;�>�4{�       �	`�gRYc�A�*

loss�R�=�k�a       �	�NhRYc�A�*

loss�+.>خ3�       �	��hRYc�A�*

loss���;H���       �	=�iRYc�A�*

loss$z�=���P       �	7�jRYc�A�*

loss�K=�R8�       �	�$kRYc�A�*

lossa?�=R�       �	�kRYc�A�*

loss(R�=
���       �	�RlRYc�A�*

lossM�k=j:G�       �	�lRYc�A�*

loss�_�=�:��       �	b�mRYc�A�*

lossZ��<��9       �	�EnRYc�A�*

loss(��<���h       �	��nRYc�A�*

loss�0>�Xhd       �	1�oRYc�A�*

lossfV=9p�e       �	�epRYc�A�*

loss��<��fK       �	~�pRYc�A�*

loss���=�J��       �	5�qRYc�A�*

loss�k�=�7�Q       �	r6rRYc�A�*

loss��<P+��       �	��rRYc�A�*

losst7,>�(j       �	?rsRYc�A�*

loss�)w=<�       �	ltRYc�A�*

loss?j�=BQ��       �	��tRYc�A�*

loss޹=�l�       �	�3uRYc�A�*

loss�>���       �	��uRYc�A�*

lossF	�=g�"       �	�cvRYc�A�*

loss��=h�)       �	�:wRYc�A�*

loss],~=��u       �	��wRYc�A�*

lossT�@>Lp�\       �	;qxRYc�A�*

lossV��=3a0+       �	@yRYc�A�*

loss{@=���       �	4�yRYc�A�*

loss�m�<����       �	;mzRYc�A�*

loss.�=�b       �	�{RYc�A�*

lossƩ#>��C       �	��{RYc�A�*

loss"S�=:2�M       �	DO|RYc�A�*

loss�p�=���       �	��~RYc�A�*

lossV2=�f�       �	% RYc�A�*

loss!`=}	TK       �	ܼRYc�A�*

loss��J=��ķ       �	~S�RYc�A�*

lossd�K=�
       �	d�RYc�A�*

loss?�D=�ӭR       �	��RYc�A�*

loss
&�=K�       �	r3�RYc�A�*

loss�}==3�Y       �	�؂RYc�A�*

lossC�,>q;�       �	�ރRYc�A�*

loss�3�==i�d       �	t�RYc�A�*

lossFD=n9!       �	��RYc�A�*

loss�'U>�d\       �	t��RYc�A�*

loss�\{=.�C^       �	��RYc�A�*

loss�F^=��]       �	�<�RYc�A�*

loss@j�<��        �	��RYc�A�*

loss���=��L,       �	�}�RYc�A�*

loss��=Yk1(       �	�!�RYc�A�*

loss�	=`$}U       �	̷�RYc�A�*

loss�#>I�u       �	�Y�RYc�A�*

loss�=*��B       �	���RYc�A�*

loss��=Z�.�       �	y��RYc�A�*

loss�bF=�U�4       �	�/�RYc�A�*

loss�f>���       �	�ЍRYc�A�*

loss��=�S��       �	�g�RYc�A�*

lossF7=���^       �	���RYc�A�*

loss��^=�̂�       �	3��RYc�A�*

losslR�<�6��       �	�C�RYc�A�*

loss��*>i��       �	��RYc�A�*

loss��>%-�       �	��RYc�A�*

loss�@->���       �		n�RYc�A�*

losse*�=^�h�       �	6#�RYc�A�*

loss{�<�M�E       �	��RYc�A�*

loss��<	tO       �	#��RYc�A�*

loss(.=�r�       �	�RYc�A�*

loss��t=u���       �	��RYc�A�*

loss/ˉ='�V       �	W{�RYc�A�*

loss��c=��}       �	�7�RYc�A�*

loss�T\=:hu�       �	CT�RYc�A�*

loss���=~(.�       �	�RYc�A�*

loss�)�=8���       �	^��RYc�A�*

loss���=[��       �	�1�RYc�A�*

loss���=̕γ       �	珝RYc�A�*

loss��T={L�       �	�)�RYc�A�*

loss�h�=&���       �	nRYc�A�*

loss��=�S�)       �	Ie�RYc�A�*

loss�%�=�<.�       �	�%�RYc�A�*

loss���=_\#       �	���RYc�A�*

loss��j=�=��       �	���RYc�A�*

lossE�X<Ñ�s       �	��RYc�A�*

loss4��=5`�n       �	ܻ�RYc�A�*

loss���=N�ܢ       �	��RYc�A�*

loss�wz=Ɍ�Z       �	��RYc�A�*

loss}��=Ԍ۹       �	gd�RYc�A�*

loss6FL=��ty       �	���RYc�A�*

loss��>IKC�       �	,��RYc�A�*

loss�t�=#pP�       �	>�RYc�A�*

loss��n=���       �	�ۧRYc�A�*

lossf�k=��       �	6w�RYc�A�*

loss��*=�7x       �	Y�RYc�A�*

loss$��=�'�7       �	���RYc�A�*

loss�n�<�P��       �	��RYc�A�*

loss�=S�4       �	/5�RYc�A�*

loss��B=G�       �	�ЫRYc�A�*

loss_��='p�`       �	�g�RYc�A�*

loss&��<Pa�F       �	ϡ�RYc�A�*

loss��r=o'       �	�o�RYc�A�*

loss�G�=��"/       �	�	�RYc�A�*

loss�<�"�       �	P��RYc�A�*

lossds�<���       �	�ҰRYc�A�*

lossݐ+=-8��       �	Ww�RYc�A�*

loss� >���       �	��RYc�A�*

loss�P�=�F�       �	g��RYc�A�*

loss �@=_�H�       �	�[�RYc�A�*

loss��>��Oy       �	��RYc�A�*

loss�1>g�Ľ       �	���RYc�A�*

loss�\>�֡       �	�+�RYc�A�*

lossD=��       �	�ͶRYc�A�*

losst>!>ʶ��       �	vT�RYc�A�*

loss�J�=�잠       �	z5�RYc�A�*

loss��=���`       �	�عRYc�A�*

lossf��<�vH       �	zǺRYc�A�*

loss M8=
3d�       �	*p�RYc�A�*

loss��=���       �	�RYc�A�*

lossW�[=�/��       �	帼RYc�A�*

loss_�	=j�	�       �	�c�RYc�A�*

loss�F�=8j-�       �	��RYc�A�*

lossv�x==���       �	n��RYc�A�*

loss��>D���       �	(D�RYc�A�*

lossr�=�TkK       �	��RYc�A�*

lossֺC>��l�       �	ƅ�RYc�A�*

loss��<U��       �	�"�RYc�A�*

losse�%>�yu�       �	��RYc�A�*

loss��P=G�}�       �	�W�RYc�A�*

loss��>���       �	��RYc�A�*

loss;t�=(ܠ�       �	���RYc�A�*

lossA�=��/       �	�-�RYc�A�*

loss��=l�Դ       �	�
�RYc�A�*

loss"@�=9\j�       �	n��RYc�A�*

loss4�=Tr��       �	�{�RYc�A�*

loss��=�&6,       �	g'�RYc�A�*

lossai�<49�       �	���RYc�A�*

loss�r�<�Cq�       �	`t�RYc�A�*

losst�c<+8`�       �	��RYc�A�*

lossO��<b_5K       �	���RYc�A�*

loss�1�=-���       �	 S�RYc�A�*

loss#�6>�/Zw       �	'��RYc�A�*

loss܈=����       �	]��RYc�A�*

loss���=��M�       �	�T�RYc�A�*

lossZ'�<�8M�       �	���RYc�A�*

lossme=A�.T       �	��RYc�A�*

loss,�A>f3�K       �	}A�RYc�A�*

loss�'�<�S�~       �	��RYc�A�*

losst�=K�n�       �	C��RYc�A�*

loss���=E6�       �	V,�RYc�A�*

loss�ɀ=O~A       �	���RYc�A�*

loss�g<oWH�       �	2q�RYc�A�*

lossf2�<2}�$       �	9�RYc�A�*

loss�l�=YG��       �	���RYc�A�*

loss�>٢�       �	�}�RYc�A�*

loss&,�=���       �	/m�RYc�A�*

loss�~�=�Ҡ       �	D4�RYc�A�*

loss��(=��Z       �	���RYc�A�*

loss��=;��       �	�c�RYc�A�*

loss��=@C}�       �	��RYc�A�*

loss�l=R�Uj       �	Χ�RYc�A�*

loss��=�K��       �	�d�RYc�A�*

loss]r�<\��       �	��RYc�A�*

loss���= H;�       �	���RYc�A�*

loss��	=���       �	�(�RYc�A�*

lossi=9��       �	<��RYc�A�*

loss�Y�=�?��       �	~��RYc�A�*

loss�u�=��`5       �	f��RYc�A�*

loss\R>�m7J       �	��RYc�A�*

loss���<JtO�       �	0�RYc�A�*

loss�1^>���       �	���RYc�A�*

lossfq=�c�       �	��RYc�A�*

lossR/�='��       �	O�RYc�A�*

loss}��=BFV       �	���RYc�A�*

loss��`=����       �	 <�RYc�A�*

loss�>4=+�       �	A��RYc�A�*

loss�%=(,�       �	�l�RYc�A�*

loss z=M��3       �	f�RYc�A�*

loss`�0=m���       �	��RYc�A�*

loss���=o��       �	�]�RYc�A�*

loss�k>�E��       �	;��RYc�A�*

loss�
=پ'x       �	���RYc�A�*

loss��X=��_       �	�M�RYc�A�*

loss�I>��
�       �	���RYc�A�*

loss�]=o�!f       �	��RYc�A�*

loss/�{<J�]       �	�5�RYc�A�*

loss�W=��       �	���RYc�A�*

losso>L*�       �	9d�RYc�A�*

loss'�
=��7�       �	��RYc�A�*

loss��=�3��       �	���RYc�A�*

lossQ
a=i�C�       �	l%�RYc�A�*

loss�.�=��{�       �	j��RYc�A�*

loss.l�<��;�       �	iS�RYc�A�*

lossϯ<b0��       �	\��RYc�A�*

loss�_�<���
       �	�|�RYc�A�*

loss/�<�Rk�       �	��RYc�A�*

loss,>@D�P       �	���RYc�A�*

loss{<�<iv��       �	v�RYc�A�*

loss�Gi=� a       �	��RYc�A�*

loss��=m�̋       �	o��RYc�A�*

lossID=`(�       �	}X�RYc�A�*

loss�"�<����       �	t��RYc�A�*

lossvM�<�b        �	���RYc�A�*

loss_�u=Lh��       �	�!�RYc�A�*

loss<)a=Mс$       �	0�RYc�A�*

loss{x=�D�U       �	P��RYc�A�*

loss�n=���       �	�<�RYc�A�*

lossd*�=٣�o       �	c��RYc�A�*

loss�)�=e�'0       �	|�RYc�A�*

loss+�=B:�       �	��RYc�A�*

loss{��<�#!       �	F��RYc�A�*

loss�ٺ=WޘY       �	1a�RYc�A�*

loss�N=O��_       �	?�RYc�A�*

loss%5�=ti�K       �	���RYc�A�*

losssD=��
�       �	t�RYc�A�*

loss���<W�NH       �	��RYc�A�*

lossJG=�̟�       �	j��RYc�A�*

loss���=p�ć       �	�\�RYc�A�*

loss��<�ԋ�       �	���RYc�A�*

loss��=��ρ       �	؝�RYc�A�*

loss���<Q"�       �	�A SYc�A�*

loss��8=z�!       �	D� SYc�A�*

loss$�<$R|�       �	�|SYc�A�*

loss�4;=�mP       �	�SYc�A�*

loss��Y=(�	�       �	
�SYc�A�*

loss�P�<��;�       �	�qSYc�A�*

lossm2^=���U       �	eSYc�A�*

loss1we=�-�k       �	)�SYc�A�*

lossd_>A��Y       �	�SSYc�A�*

loss�$�<�D�       �	@�SYc�A�*

loss�6�<�fn       �	�SYc�A�*

lossJg�=#��       �	r6SYc�A�*

loss31#=�6
       �	RSYc�A�*

loss�P2<��P�       �	��SYc�A�*

loss�d<Ah@�       �	T	SYc�A�*

lossp�<�*\       �	��	SYc�A�*

lossFm�<�cpb       �	��
SYc�A�*

loss�%<ߞ�}       �	r1SYc�A�*

loss�-�;$�Uz       �	[�SYc�A�*

loss���=���0       �	)wSYc�A�*

loss�o+;�ŶF       �	?SYc�A�*

loss�\!;u�@       �	��SYc�A�*

lossZ�;�˧(       �	2ZSYc�A�*

loss��<� b       �	 SYc�A�*

loss��R=_V�b       �	T�SYc�A�*

loss[@�<mpɅ       �	��SYc�A�*

lossk;�u�Z       �	7RSYc�A�*

lossQ�=\��W       �	�SYc�A�*

loss�o�>�qp�       �	�SYc�A�*

loss[��<���X       �	.vSYc�A�*

loss4�>�$L�       �	]�SYc�A�*

loss~��=�Ȏ�       �	(,SYc�A�*

loss�=�ʺ�       �	��SYc�A�*

loss恔<��+G       �	LpSYc�A�*

loss�ψ<�,�       �	�SYc�A�*

loss�j�=˄��       �	7�SYc�A�*

lossݔ�=;��       �	pASYc�A�*

loss��=��6�       �	�SYc�A�*

loss��=9(�       �	tzSYc�A�*

lossT�<@��       �	VSYc�A�*

loss�a>r)@        �	�SYc�A�*

loss�a>̍��       �	�:SYc�A�*

loss�Q.=��R�       �	��SYc�A�*

loss1�=(I�K       �	�hSYc�A�*

loss��>�t�\       �	SYc�A�*

loss�~=֙�F       �	��SYc�A�*

loss��=�
$�       �	8NSYc�A�*

loss�&V>��       �	X�SYc�A�*

loss�:�=�-�       �	�~SYc�A�*

loss�?�<����       �		 SYc�A�*

lossq�2=j�       �	bM!SYc�A�*

loss��=��       �	.�!SYc�A�*

loss8
d<P�n5       �	�}"SYc�A�*

lossQ�S<H"       �	�#SYc�A�*

loss���<7!�       �	�#SYc�A�*

lossT�@=�)c�       �	MJ$SYc�A�*

loss�$~=)�z1       �	?�$SYc�A�*

loss���=���       �	<�%SYc�A�*

lossH;�=U)]�       �	&SYc�A�*

lossL�<ZsH�       �	S�&SYc�A�*

loss?lT=��o       �	�F'SYc�A�*

loss��M=5��       �	��'SYc�A�*

loss$�=��A*       �	�n(SYc�A�*

lossx�<�}t       �	�)SYc�A�*

loss�h=�̅�       �	�)SYc�A�*

loss���<�Jm       �	aQ*SYc�A�*

loss��=�|e�       �	��*SYc�A�*

loss�z�=��D       �	o�+SYc�A�*

loss��=��       �	�f,SYc�A�*

lossk<�A$       �	� -SYc�A�*

loss��<�a�       �	�.SYc�A�*

loss��=�L|       �	��.SYc�A�*

loss��=�O�u       �	=�/SYc�A�*

loss���<�ؗ�       �	��0SYc�A�*

loss6Ʒ=FU �       �	�T1SYc�A�*

loss&5�=�W�N       �	��1SYc�A�*

loss,��<d���       �	��2SYc�A�*

loss��:=l8f�       �	�?3SYc�A�*

loss*[�< ̭�       �	f�4SYc�A�*

lossy�<0 ��       �	�95SYc�A�*

lossi͖=Ջ�       �	�QSYc�A�*

lossN�>r��       �	3�QSYc�A�*

lossҡ�=��[       �	�ARSYc�A�*

loss���=�H�q       �	b�RSYc�A�*

loss��t=q�l       �	�SSYc�A�*

lossxr/=�֚       �	�sTSYc�A�*

loss�\�<~Cr`       �	�4USYc�A�*

loss)S�=���       �	mVSYc�A�*

loss�M>����       �	��VSYc�A�*

loss�0>Nd�       �	�|WSYc�A�*

lossĕ�<�nT�       �	�&XSYc�A�*

loss��>=�X�       �	�AYSYc�A�*

loss���=.�       �	υZSYc�A�*

loss��{=)��       �	DL[SYc�A�*

losso�p=��
[       �	"�[SYc�A�*

loss��Y=��eL       �	Л\SYc�A�*

lossR�[<t���       �	�5]SYc�A�*

loss�^�=�1Q       �	��]SYc�A�*

loss �=@�_�       �	f�^SYc�A�*

loss̞~>�qk�       �	�H_SYc�A�*

loss,^Y=�Iy       �	��_SYc�A�*

loss���=7w�c       �	И`SYc�A�*

loss&�=�A2�       �	GaSYc�A�*

loss�`	>��"       �	��aSYc�A�*

loss�?=, ��       �	H�bSYc�A�*

loss==s7,       �	�RcSYc�A�*

loss���=��\
       �	r�cSYc�A�*

loss���<�"�G       �	G�dSYc�A�*

loss^==�"��       �	.TeSYc�A�*

lossyC�<���       �	�fSYc�A�*

loss�T�=��       �	K�fSYc�A�*

loss�*P<x&       �	�\gSYc�A�*

loss���<n�37       �	�QhSYc�A�*

loss��>w) �       �	CiSYc�A�*

loss��L=3b�b       �	�iSYc�A�*

loss�� >��n�       �	�`jSYc�A�*

loss&�	<���       �	�kSYc�A�*

loss�p�=�Nk       �	��kSYc�A�*

lossϩ>���
       �	��lSYc�A�*

loss���=<8ק       �	+KmSYc�A�*

lossA�=�b�       �	m�mSYc�A�*

loss3gr=�3       �	q�nSYc�A�*

loss��?<*�       �	�oSYc�A�*

loss{�j=iAX]       �	�>pSYc�A�*

loss�)=K�}�       �	��pSYc�A�*

loss�|>U��       �	��qSYc�A�*

lossj{b=Fg$       �	�6rSYc�A�*

loss���=���       �	]�rSYc�A�*

loss-�3=�-�O       �	��sSYc�A�*

loss�q?<�h!�       �	��tSYc�A�*

loss���<x�i       �	]1uSYc�A�*

loss��=`���       �	E�uSYc�A�*

loss�^=���       �	�uvSYc�A�*

loss��>U���       �	�wSYc�A�*

lossn;K=�%�       �	�wSYc�A�*

loss��<�>�       �	�ZxSYc�A�*

loss�uB<t�q       �	�ySYc�A�*

loss{#<���c       �	V�ySYc�A�*

lossqMR=eO��       �	[zSYc�A�*

loss��=h��       �	U�zSYc�A�*

lossÅ>���       �	��{SYc�A�*

lossL�B=�ޞ       �	G|SYc�A�*

loss�=P��~       �	��|SYc�A�*

loss1�e=P�؄       �	/�}SYc�A�*

loss?��<��<       �	n0~SYc�A�*

lossm�=����       �	`�~SYc�A�*

loss!��=�|��       �	#�SYc�A�*

lossS�W=ʹl�       �	h �SYc�A�*

loss(s>���       �	{��SYc�A�*

loss#s�=���       �	-[�SYc�A�*

loss[-�<؉�       �	��SYc�A�*

loss6�&=6f��       �	␂SYc�A�*

lossq\=�v�D       �	B��SYc�A�*

lossT`=>7$�       �	%?�SYc�A�*

loss(�=
µ�       �	��SYc�A�*

loss��=��~�       �	u��SYc�A�*

loss�2=����       �	�5�SYc�A�*

loss��2=�o       �		�SYc�A�*

loss3=d�Ja       �	��SYc�A�*

lossf�=+6��       �	;��SYc�A�*

loss��g=(&�       �	
I�SYc�A�*

loss|��=+�.�       �	:�SYc�A�*

loss#+=��Zu       �	���SYc�A�*

loss��<E�.�       �	�͋SYc�A�*

loss��<B%�8       �	�n�SYc�A�*

losss�;=�C       �	4�SYc�A�*

lossJ�]=�E��       �	���SYc�A�*

loss��<> @�       �	MH�SYc�A�*

lossq�<iy�       �	$'�SYc�A�*

loss��I<���       �	���SYc�A�*

loss�Ii=��Un       �	�z�SYc�A�*

loss�a>��}j       �	��SYc�A�*

lossa�=��Z       �	9��SYc�A�*

loss�=���u       �	Y�SYc�A�*

loss��D=�7ֆ       �	��SYc�A�*

losscB�<.'�>       �	���SYc�A�*

loss�)/=�#V�       �	tДSYc�A�*

loss��;=M˷�       �	�n�SYc�A�*

loss�i,>V��G       �	��SYc�A�*

loss��=�3X       �	�ΖSYc�A�*

loss�<ud5�       �	L��SYc�A�*

loss͞8<I��       �	f�SYc�A�*

loss�p=k;<       �	�B�SYc�A�*

lossW�<���'       �	� �SYc�A�*

loss`܁=�-L       �	t��SYc�A�*

loss�>��\-       �	r��SYc�A�*

loss�Gi<��u�       �	5��SYc�A�*

loss�`X<sе�       �	�6�SYc�A�*

losso9>?�At       �	��SYc�A�*

loss�X(=�Ry       �	a��SYc�A�*

loss�kw=��E5       �	j4�SYc�A�*

lossn�<x��       �	oӟSYc�A�*

lossԜ==�ϱ~       �	6r�SYc�A�*

loss��<�Z_       �	��SYc�A�*

loss}��<Eo�       �	!ʡSYc�A�*

loss�E<��/�       �	�c�SYc�A�*

lossq��<<A�Y       �	a��SYc�A�*

loss�j�=��D�       �	��SYc�A�*

lossSE�==�s7       �	�y�SYc�A�*

loss t=5�A       �	��SYc�A�*

loss�f�=���       �	ũ�SYc�A�*

loss��=FAa�       �	9D�SYc�A�*

lossm�
>]܊�       �	v��SYc�A�*

lossİ >ց�       �	Ș�SYc�A�*

lossŃ�=dOZt       �	/�SYc�A�*

lossFa�=N�	       �	�ĨSYc�A�*

loss�a�=ё        �	�g�SYc�A�*

loss��=���[       �	��SYc�A�*

lossT��=)�       �	�תSYc�A�*

loss=*�=��9       �	it�SYc�A�*

loss��=�nY^       �	��SYc�A�*

loss��=~��P       �	I-�SYc�A�*

lossH��<x]0�       �	ǭSYc�A�*

loss��<Χ�       �	_�SYc�A�*

loss`�={��       �	���SYc�A�*

loss��=�W�       �	u��SYc�A�*

loss��S=�[Ǜ       �	f3�SYc�A�*

loss�D?=�b��       �	�ȰSYc�A�*

loss�r�<-���       �	�c�SYc�A�*

loss(�<h��       �	���SYc�A�*

loss��<8\�       �	�0�SYc�A�*

loss�^=���       �	0״SYc�A�*

lossJL�=;NU       �	A~�SYc�A�*

loss�N�=�Vn}       �	�&�SYc�A�*

loss�>kY��       �	�ضSYc�A�*

losswd~=S�>_       �	�u�SYc�A�*

lossd�;=�Q��       �	Z�SYc�A�*

loss�6�<�j��       �	=��SYc�A�*

loss���<(��       �	�W�SYc�A�*

loss���<�89�       �	��SYc�A�*

loss�� =z���       �	�G�SYc�A�*

lossO�w=���       �	M�SYc�A�*

loss���=߶k6       �	��SYc�A�*

loss��_=R#J-       �	�+�SYc�A�*

loss+ׇ=���       �	�ƽSYc�A�*

loss��a<���)       �	_�SYc�A�*

loss��=Z���       �	��SYc�A�*

losswǵ=
�N0       �	7��SYc�A�*

lossH�*>;:       �	}$�SYc�A�*

lossؘ�<�)       �	���SYc�A�*

loss� >a�B�       �	l\�SYc�A�*

loss�R�<�7�       �	���SYc�A�*

lossZ�=A�t@       �	��SYc�A�*

loss�<�:��       �	N+�SYc�A�*

loss��O=w��       �	���SYc�A�*

lossv��<��_       �	$^�SYc�A�*

loss0��<��9�       �	���SYc�A�*

lossi77<I�[�       �	���SYc�A�*

loss��=��c       �	�T�SYc�A�*

loss|u�<��       �	���SYc�A�*

loss�� =�'�}       �	���SYc�A�*

loss[��<��       �	�-�SYc�A�*

loss�xE=nc�K       �	���SYc�A�*

loss�`�=�F�       �	j�SYc�A�*

loss��<F|7{       �	H�SYc�A�*

loss6g�<��U�       �	���SYc�A�*

lossR�$>��0�       �	sI�SYc�A�*

loss�F�=��Y�       �	���SYc�A�*

loss-�=[|�       �	��SYc�A�*

loss\�<�Zx       �	�7�SYc�A�*

loss7o�;A�e       �	p��SYc�A�*

loss�=��       �	�m�SYc�A�*

lossw�Y<�yHl       �	��SYc�A�*

lossN�^<���        �	ղ�SYc�A�*

lossA��<���.       �	�R�SYc�A�*

lossS�`=����       �	���SYc�A�*

loss�M=�j�       �	��SYc�A�*

loss��=���b       �	A)�SYc�A�*

lossH{=�       �	���SYc�A�*

loss��=�iw       �	�|�SYc�A�*

loss�X
>�v�       �	��SYc�A�*

loss�-=��M�       �	ܷ�SYc�A�*

loss�`G=Q9f       �	iU�SYc�A�*

lossxמ=ka�.       �	���SYc�A�*

lossdqH<�'��       �	 ��SYc�A�*

loss�W=!8F�       �	�&�SYc�A�*

loss�/�<�!2�       �	���SYc�A�*

loss�=l�Z       �	6Y�SYc�A�*

lossa݉=�X�U       �	���SYc�A�*

lossJX<q,�,       �	[��SYc�A�*

loss��=����       �	�7�SYc�A�*

loss��=<ܥ�       �	��SYc�A�*

loss:hJ=���       �	ds�SYc�A�*

lossۆk=I�ߜ       �	0�SYc�A�*

loss-��<&�K)       �	]��SYc�A�*

lossȿ5=�d	       �	�m�SYc�A�*

loss:�d=Qn7       �	��SYc�A�*

loss}1
>�;
�       �	y��SYc�A�*

loss1z=��       �	I�SYc�A�*

loss:=�E;       �	&��SYc�A�*

loss���=�w�,       �	~�SYc�A�*

loss�Q�<J��e       �	��SYc�A�*

loss�v=��z       �	���SYc�A�*

loss���<���       �	�V�SYc�A�*

losshT�=Qk�Y       �	?��SYc�A�*

loss�n�=��-       �	v��SYc�A�*

loss��{>S��       �	���SYc�A�*

loss��a>MmN       �	��SYc�A�*

lossq�>��2D       �	k)�SYc�A�*

loss��>�S[       �	���SYc�A�*

loss�Ƶ<:��       �	�h�SYc�A�*

loss O�<@�       �	^�SYc�A�*

loss��I=�^�O       �	h��SYc�A�*

loss��=k���       �	DN�SYc�A�*

loss��)<dU��       �	���SYc�A�*

loss���=�I�o       �	���SYc�A�*

loss��=s|>       �	i9�SYc�A�*

loss�-�=�:�       �	��SYc�A�*

loss4�-=��b�       �	�{�SYc�A�*

losst>��eF       �	
��SYc�A�*

lossOP=d#Y�       �	�b�SYc�A�*

loss��$=�5K       �	I��SYc�A�*

losssWb=`�M       �	V~�SYc�A�*

loss�-�<��2       �	6�SYc�A�*

lossa��=�-��       �	���SYc�A�*

loss���=�7       �	ͫ�SYc�A�*

lossA�=Ls	       �	�F�SYc�A�*

loss���<`���       �	c��SYc�A�*

lossk��=�K�       �	���SYc�A�*

lossZ�S=y�n       �	*�SYc�A�*

loss�D=j=�       �	ı�SYc�A�*

lossZ�==~_�       �	�I�SYc�A�*

loss���=��\�       �	���SYc�A�*

loss�;�=��       �	~�SYc�A�*

loss�=uA�       �	��SYc�A�*

loss1̞=`��F       �	#��SYc�A�*

loss�ŕ=8:       �	�T�SYc�A�*

loss��=jח:       �	B��SYc�A�*

loss$��=�7u�       �	@��SYc�A�*

lossp=��       �	��SYc�A�*

loss"=&�v       �	��SYc�A�*

lossTJ�=�U7�       �	�g�SYc�A�*

loss��= �T       �	=�SYc�A�*

loss��U<�$؁       �	���SYc�A�*

lossT�<�#>�       �	$G TYc�A�*

loss��#=�a��       �	*� TYc�A�*

loss�"=�b��       �	4�TYc�A�*

loss$͝=����       �	KTYc�A�*

lossc�-=��5       �	��TYc�A�*

lossJ�,=L�#�       �	�[TYc�A�*

loss���<y&�       �	��TYc�A�*

loss��=�&�       �	�TYc�A�*

losss��=�}<=       �	,bTYc�A�*

loss�j�=iu=�       �	+�TYc�A�*

loss�[F=�H�       �	��TYc�A�*

loss��<��D7       �	�8TYc�A�*

loss���=�+��       �	��TYc�A�*

loss,�+=����       �	�vTYc�A�*

lossp!>�9N�       �		TYc�A�*

loss��=��/�       �	lA
TYc�A�*

lossV�[=nDi       �	+�
TYc�A�*

loss��=j��       �	�qTYc�A�*

loss�GA=ٕ�R       �	bTYc�A�*

loss��w=�X�       �	��TYc�A�*

loss#�A>���       �	�KTYc�A�*

loss���=w��$       �	-�TYc�A�*

loss��<Yw�i       �	;�TYc�A�*

loss��2<v%ک       �	�$TYc�A�*

lossF��=�Ȩ�       �	��TYc�A�*

loss�h�=V�       �	w�TYc�A�*

lossjl�<�\�O       �	(CTYc�A�*

loss�߀<j�ў       �	��TYc�A�*

loss=s�<H�s�       �	�vTYc�A�*

loss���<���H       �	�TYc�A�*

loss�k0>?0Z       �	��TYc�A�*

loss��O=n��f       �	�sTYc�A�*

loss�.Y=�g�       �	�DTYc�A�*

lossR�<K�N�       �	O�TYc�A�*

loss��=�5�D       �	> TYc�A�*

loss��'=��m�       �	�WTYc�A�*

loss}*<y f�       �	!TYc�A�*

loss��=�+�       �	��TYc�A�*

loss1�<��7       �	��TYc�A�*

loss��7<�1Qn       �	<OTYc�A�*

loss��<�Yn       �	�PTYc�A�*

loss�C�=�;~d       �	�TYc�A�*

loss��}=�9b       �	��TYc�A�*

loss��=p�*�       �	XpTYc�A�*

loss��=���       �	�&TYc�A�*

loss�G�=�
�       �	oc TYc�A�*

loss�Y�==�3       �	�� TYc�A�*

loss��<�m��       �	,�!TYc�A�*

lossD��<Ѿ5       �	�2"TYc�A�*

lossQW�<Aœ       �	��"TYc�A�*

lossD�=$�&�       �	�a#TYc�A�*

lossF�>>���       �	��#TYc�A�*

loss�b�<��P       �	��$TYc�A�*

lossC�<=�T�       �	�%TYc�A�*

losszU|=�t�S       �	��%TYc�A�*

loss�0=%���       �	sd&TYc�A�*

loss��[=���g       �	� 'TYc�A�*

loss��<]��`       �	ȗ'TYc�A�*

lossl�)>mh��       �	I+(TYc�A�*

loss���<�橞       �	��(TYc�A�*

loss(|=��       �	�k)TYc�A�*

loss��=�yE�       �	�)TYc�A�*

losszY�<����       �	#�*TYc�A�*

loss�=��C       �	a4+TYc�A�*

loss�O=�"�!       �	X�+TYc�A�*

loss�v�<�z�s       �	�f,TYc�A�*

loss݉�<���       �	�I-TYc�A�*

loss��=ו�w       �	��-TYc�A�*

loss�+Z=v�)B       �	$�.TYc�A�*

loss,�1=�3Ӽ       �	�/TYc�A�*

loss�;6=3'��       �	 �/TYc�A�*

loss\��=��       �	?S0TYc�A�*

loss.��<I]+j       �	;1TYc�A�*

lossQ�J=5�+
       �	��1TYc�A�*

loss�X�<��p       �	=b2TYc�A�*

loss�>�y�P       �	��2TYc�A�*

loss��<}<(A       �	�3TYc�A�*

loss�E%=��8       �	�&4TYc�A�*

loss��<�$U�       �	;�4TYc�A�*

loss �?<��$h       �	�_5TYc�A�*

lossWXA>>:�R       �	��5TYc�A�*

loss*�<(��R       �	7�6TYc�A�*

loss��=����       �	�#7TYc�A�*

loss�@=c�*       �	�{8TYc�A�*

loss�]�=j��[       �	�9TYc�A�*

loss�o<|U�Z       �	��9TYc�A�*

loss��<�M�!       �	6V:TYc�A�*

loss���=]�?`       �	�:TYc�A�*

loss�=�ޔ�       �	��;TYc�A�*

loss�X=�wK�       �	�Y<TYc�A�*

losss�*>�P*�       �	��<TYc�A�*

loss���=c�       �	n�=TYc�A�*

loss�z=�9g       �	C>TYc�A�*

loss�1=��       �	n�>TYc�A�*

loss1{�=c�#       �	؁?TYc�A�*

lossx =O       �	@TYc�A�*

lossjz�<O�1�       �	��@TYc�A�*

loss��<��TG       �	{MATYc�A�*

loss��3=!�       �	��ATYc�A�*

loss�z�<�(�E       �	�BTYc�A�*

loss���=���       �	�HCTYc�A�*

lossT"|<       �	��CTYc�A�*

loss��R=#��c       �	ׇDTYc�A�*

loss�h<��        �	dETYc�A�*

loss_F�=k�E�       �	��ETYc�A�*

loss��<=���4       �	�lFTYc�A�*

losse�"=�X��       �	GTYc�A�*

loss�{ =�"�       �	ΩGTYc�A�*

lossR�=x       �	�?HTYc�A�*

loss�0>xy       �	��HTYc�A�*

loss��=0��z       �	�ITYc�A�*

loss�Ұ=�p�       �	�,JTYc�A�*

loss'l=׭R       �	o�JTYc�A�*

loss��<eY�       �	|KTYc�A�*

loss(�<�Twp       �	�JLTYc�A�*

loss|��<�/�       �	��LTYc�A�*

lossnb�=BŨo       �	f�MTYc�A�*

losst՗=�)�       �	�NTYc�A�*

loss��=�G�       �	N�NTYc�A�*

loss}"3=�i       �	YQOTYc�A�*

lossR�<U��       �	��OTYc�A�*

loss��U>n?��       �	A}PTYc�A�*

loss�v=C��t       �	QTYc�A�*

lossYm=<z2       �	��QTYc�A�*

loss��0=���       �	mSRTYc�A�*

lossl5�=ys       �	��RTYc�A�*

loss{j=v�'�       �	^�STYc�A�*

loss��B=���d       �	�TTYc�A�*

loss��=�Df�       �	��TTYc�A�*

loss��B=}��H       �	O�UTYc�A�*

loss6�U<m;�       �	oGVTYc�A�*

loss��=����       �	@�VTYc�A�*

loss��=5j�       �	PqWTYc�A�*

loss�l=���!       �	p	XTYc�A�*

loss���<�xE       �	��XTYc�A�*

loss�C=�a       �	�4YTYc�A�*

loss6A�=>aQL       �	�YTYc�A�*

loss��<T8>       �	�kZTYc�A�*

lossH=��       �	\�ZTYc�A�*

loss�}=^]�V       �	��[TYc�A�*

lossv-=���       �	�5\TYc�A�*

lossS�0>��       �	�\TYc�A�*

loss�A�<Ay�	       �	�m]TYc�A�*

loss�W=���       �	�^TYc�A�*

lossok�<^w�r       �	�^TYc�A�*

loss��=й6�       �	�8_TYc�A�*

lossNؒ<�Wk�       �	��_TYc�A�*

lossO�e<3��?       �	i`TYc�A�*

lossor�=$��       �	��`TYc�A�*

loss�f�<eYD�       �	�aTYc�A�*

loss9��=��}�       �	I�cTYc�A�*

loss�T1=�T��       �	�kdTYc�A�*

loss?7�=��ΰ       �	"eTYc�A�*

lossƛ�=�qSB       �	1�eTYc�A�*

loss�}�<B�1�       �	�OfTYc�A�*

loss [=p;��       �	��fTYc�A�*

lossiX�=5(��       �	'�gTYc�A�*

loss��=��       �	�%hTYc�A�*

loss%*=|e�w       �	r�hTYc�A�*

loss�<>�6`       �	�^iTYc�A�*

loss=��=9��       �	��iTYc�A�*

loss䆪=^E       �	��jTYc�A�*

loss�O<��@�       �	3kTYc�A�*

loss_�<ëtw       �	��kTYc�A�*

loss/��=���       �	sclTYc�A�*

loss�v=�e1u       �	j�lTYc�A�*

loss4o�<��r       �	C�mTYc�A�*

loss�ؽ=77.       �	inTYc�A�*

loss�ZP=}�o       �	oTYc�A�*

loss&Q>Q"P�       �	?�oTYc�A�*

loss/�=���A       �	�CpTYc�A�*

loss��=x3H       �	��pTYc�A�*

lossN�A=��       �	�|qTYc�A�*

loss��=H*|O       �	'rTYc�A�*

loss;X6=��Ȧ       �	��rTYc�A�*

loss�<e\)       �	*XsTYc�A�*

loss��==��(�       �	��sTYc�A�*

loss/��<���       �	��tTYc�A�*

losse�<"8��       �	�4uTYc�A�*

loss8�{<��o       �	��uTYc�A�*

loss?�T=���       �	��vTYc�A�*

loss_�I=�]��       �	jwTYc�A�*

loss���<�p(       �	\�wTYc�A�*

lossd<B�o�       �	AExTYc�A�*

loss��<ܝ�J       �	��xTYc�A�*

lossFE*=m��       �	*�yTYc�A�*

lossդ>.���       �	,zTYc�A�*

loss���=f���       �	_�zTYc�A�*

loss��t<�[(       �	&s{TYc�A�*

loss�`�=�N�?       �	�
|TYc�A�*

loss�g<�D-�       �	�|TYc�A�*

loss��<X��S       �	=E}TYc�A�*

lossagS>�)B       �	4�}TYc�A�*

loss?=�mX       �	�r~TYc�A�*

loss�=��l       �	�TYc�A�*

loss�O�="[�       �	=�TYc�A�*

loss3
�=�f�W       �	T5�TYc�A�*

loss(�=���       �	�ˀTYc�A�*

loss�=n껭       �	y]�TYc�A�*

lossc�=�٥1       �	��TYc�A�*

loss�ޘ=�EP       �	m��TYc�A�*

lossFj+=�fS�       �	�'�TYc�A�*

loss,�=�T��       �	�ŃTYc�A�*

loss@1=D��       �	���TYc�A�*

lossh�=��}�       �	ND�TYc�A�*

loss��i=�v��       �	g��TYc�A�*

loss�"�<���       �	:��TYc�A�*

loss��<�^�+       �	�-�TYc�A�*

loss�2�<o�)       �	�ƇTYc�A�*

loss��>jk        �	R`�TYc�A�*

loss,#2=�       �	��TYc�A�*

lossU�=6 �s       �	2��TYc�A�*

lossO_/=����       �	�'�TYc�A�*

loss�� <C[;}       �	��TYc�A�*

loss���=ћ��       �	eV�TYc�A�*

lossf1(<5]Ӯ       �	��TYc�A�*

loss�k9>t�       �	��TYc�A�*

lossq�m=���y       �	��TYc�A�*

lossT�=p	=       �	���TYc�A�*

loss���=kH       �	�?�TYc�A�*

loss��&="N)       �	�֎TYc�A�*

loss��<���       �	���TYc�A�*

loss(%<6�t       �	=��TYc�A�*

loss�V�=����       �	�V�TYc�A�*

lossω�<�oa       �	$�TYc�A�*

loss��=�)�n       �	���TYc�A�*

loss�J�=�d(+       �	�'�TYc�A�*

lossѸ�=�oX�       �	l̓TYc�A�*

loss�#=Y=��       �	Ho�TYc�A�*

loss�H3>f�/M       �	Y�TYc�A�*

loss���=�~Si       �	/��TYc�A�*

loss� �<>0��       �	y?�TYc�A�*

loss7�<�
3�       �	�זTYc�A�*

loss�j�=���'       �	�m�TYc�A�*

lossN�<
��       �	C�TYc�A�*

loss�%�=�[f       �	5��TYc�A�*

lossE<=j�F�       �	�1�TYc�A�*

loss�@�=.0�       �	���TYc�A�*

lossW��<E�M�       �	���TYc�A�*

lossaK=�A�-       �	3�TYc�A�*

loss�=lrd�       �	ƛTYc�A�*

loss�e�=����       �	!Z�TYc�A�*

loss�J�=Z�       �	9�TYc�A�*

lossf}'<���       �	w��TYc�A�*

loss�ɂ=j�I       �	��TYc�A�*

loss�*�=x_�       �	2��TYc�A�*

loss;0=��D�       �	BB�TYc�A�*

lossM;m<�/��       �	4֟TYc�A�*

loss�j<x�ʴ       �	�n�TYc�A�*

loss@=�<�P�)       �	�TYc�A�*

lossdX�=O��\       �	��TYc�A�*

loss�p�<b��       �	�;�TYc�A�*

loss8Ed=�Gg       �	oעTYc�A�*

loss�_<s9L�       �	y�TYc�A�*

lossŀG=�A��       �	E�TYc�A�*

loss%D=�        �	o��TYc�A�*

loss��<���       �	DP�TYc�A�*

loss�=j?�t       �	��TYc�A�*

lossI��<�F       �	�s�TYc�A�*

lossl~�=�y��       �	��TYc�A�*

loss��<5Ee       �	��TYc�A�*

loss)��=����       �	�I�TYc�A�*

loss�=l��Y       �	S�TYc�A�*

loss�
�<F��       �	J~�TYc�A�*

loss���<g�
       �	��TYc�A�*

loss���=�X3       �	ެ�TYc�A�*

loss�<)K�       �	�C�TYc�A�*

lossCt7=|���       �	�׫TYc�A�*

lossR��;0�       �	I��TYc�A�*

lossH"=��       �	�8�TYc�A�*

lossWF-=++R�       �	�խTYc�A�*

loss6�;=�Ŧ       �	A~�TYc�A�*

lossS��<��T       �	�TYc�A�*

loss3�&=��m       �	j��TYc�A�*

loss7G=vK�]       �	�ذTYc�A�*

loss�� <���       �	~��TYc�A�*

loss�ś=W��       �	8g�TYc�A�*

loss8Ż=��ʸ       �	y�TYc�A�*

loss^��<��#0       �	䣳TYc�A�*

lossw1:�k       �	�ϴTYc�A�*

loss�[�<�Z�c       �	�g�TYc�A�*

loss���;�gj�       �	T �TYc�A�*

loss�N�;�/�       �	�TYc�A�*

loss�?{<�?��       �	�Y�TYc�A�*

loss}�X;�/��       �	/��TYc�A�*

loss�˒<�`�       �	���TYc�A�*

loss�:��S       �	=I�TYc�A�*

loss�+o;܈	s       �	o�TYc�A�*

loss�@:+R��       �	���TYc�A�*

lossˈ<r'�       �	�J�TYc�A�*

loss�f=D�P�       �	��TYc�A�*

loss`��;�{��       �	�y�TYc�A�*

loss7Ve:Q<��       �	('�TYc�A�*

loss�>=�8�       �		½TYc�A�*

loss;\e>KU��       �	�k�TYc�A�*

loss/L�;su�       �	K �TYc�A�*

loss@�>3���       �	l��TYc�A�*

loss_l�<8'�/       �	�*�TYc�A�*

loss7В=��T0       �	�;�TYc�A�*

lossʰ5=8�=       �	���TYc�A�*

loss��;-5       �	b�TYc�A�*

loss2->\��       �	��TYc�A�*

loss=��=���R       �	���TYc�A�*

loss�T=��)H       �	�$�TYc�A�*

loss���=�K��       �	|��TYc�A�*

loss�=Q��       �	>[�TYc�A�*

loss=
>t8��       �	p��TYc�A�*

loss�;�=[�       �	>�TYc�A�*

lossT=���x       �	��TYc�A�*

loss�d�=h�	       �	Q3�TYc�A�*

loss�P=#��       �	P��TYc�A�*

lossf@=3	n       �	�Y�TYc�A�*

loss	�<^҇�       �	���TYc�A�*

lossL�=�       �	r��TYc�A�*

loss���=g��       �	�"�TYc�A�*

loss�[H<e(ڋ       �	��TYc�A�*

loss��_=�_j       �	�Q�TYc�A�*

loss;k�<��       �	���TYc�A�*

loss�<wS�       �	V}�TYc�A�*

loss�S<��*       �	#�TYc�A�*

loss��"=z%�[       �	N��TYc�A�*

loss�b<|�       �	{N�TYc�A�*

loss���=$�2�       �	���TYc�A�*

loss�+�=���       �	֐�TYc�A�*

loss8_=$�g�       �	�2�TYc�A�*

lossg��=:VQ�       �	���TYc�A�*

loss	 ==.P�       �	0e�TYc�A�*

loss��+=��p�       �	��TYc�A�*

lossd];j-�       �	S��TYc�A�*

loss���<
���       �	A-�TYc�A�*

lossm7=��c       �	���TYc�A�*

losse�<A�       �	w1�TYc�A�*

loss)��<��^�       �	C�TYc�A�*

losseZd=����       �	W��TYc�A�*

lossh·=��	       �	CW�TYc�A�*

loss��X<gE�       �	si�TYc�A�*

lossCK<����       �	
�TYc�A�*

loss�7�<���!       �	��TYc�A�*

loss��E=���!       �	[]�TYc�A�*

loss��<���L       �	���TYc�A�*

loss�zh=���       �	��TYc�A�*

lossJ��=���       �	�8�TYc�A�*

lossF/<'�       �	 ��TYc�A�*

loss$��<~u��       �	�q�TYc�A�*

lossd�u<�c}�       �	d��TYc�A�*

loss�:B;��m"       �	v4�TYc�A�*

lossQ�=C��       �	^f�TYc�A�*

loss��=�_WC       �	���TYc�A�*

loss��=lcv       �	��TYc�A�*

loss
��=Z���       �	\�TYc�A�*

loss�!o=.�!y       �	|��TYc�A�*

loss��<����       �	M�TYc�A�*

loss��0<i%��       �	i��TYc�A�*

loss��=�r��       �	�}�TYc�A�*

loss/�=Fe�*       �	|E�TYc�A�*

loss�y�=��Sv       �	,��TYc�A�*

loss�p�<Ξ��       �	��TYc�A�*

lossͤ<=`�	       �	~8�TYc�A�*

lossq�<��/y       �	���TYc�A�*

loss[��<9�t       �	�u UYc�A�*

lossVu5<Ң�       �	MUYc�A�*

loss�n=��       �	��UYc�A�*

loss�4;��>G       �	�JUYc�A�*

loss�0=�]�       �	�UYc�A�*

loss��=A
�       �	J~UYc�A�*

losso�> Ц       �	aUYc�A�*

loss%�<
�H       �	
�UYc�A�*

loss��>��l
       �	�pUYc�A�*

loss�<�3X�       �	�UYc�A�*

loss�B�=.r�       �	��UYc�A�*

loss?o�<���       �	�BUYc�A�*

loss(þ<�_�       �	�UYc�A�*

loss1m�<�L��       �	U�UYc�A�*

lossM2�<!Y�j       �	"	UYc�A�*

loss!��=��x�       �	'�	UYc�A�*

loss5�<�v       �	�X
UYc�A�*

loss�K�=r��~       �	�UYc�A�*

loss��k=q�H       �	��UYc�A�*

loss�7o=����       �	�<UYc�A�*

lossh�>X�l       �	[�UYc�A�*

lossv�=�`��       �	r�UYc�A�*

loss�|=���       �	iUYc�A�*

loss.�;2��       �	�UYc�A�*

loss�I�=���@       �	5bUYc�A�*

lossF�>���       �	.UYc�A�*

loss�?=I�7       �	ЙUYc�A�*

lossf�a=����       �	r7UYc�A�*

loss}� =!�p�       �	�UYc�A�*

loss���<�X��       �	��UYc�A�*

loss�@=�P�P       �	�dUYc�A�*

loss���=eB�       �	uUYc�A�*

loss�=}��/       �	q�UYc�A�*

loss2�?=���       �	�lUYc�A�*

loss��<�_t       �	�UYc�A�*

loss�!�<���       �	�UYc�A�*

loss�7�;O*m       �	*�UYc�A�*

loss���<��Zq       �	ӡUYc�A�*

loss�"=��[       �	�_UYc�A�*

loss�S*=r�       �	:UYc�A�*

loss}��>�_c�       �	��UYc�A�*

loss��=la       �	��UYc�A�*

lossXg�;j�?�       �	�%UYc�A�*

loss���<Q7�       �	��UYc�A�*

loss�z*<ğK'       �	fiUYc�A�*

loss�:7=B�       �	UYc�A�*

loss-��=�ݘ�       �	~�UYc�A�*

loss�S�=jAK�       �	KUYc�A�*

loss=Ld=�,(        �	�UYc�A�*

loss�U=���       �	�~ UYc�A�*

loss��=k9y�       �	�!UYc�A�*

loss�?<=�t$       �	��!UYc�A�*

loss���<5X�       �	kb"UYc�A�*

lossh��=��I       �	��"UYc�A�*

loss��#=���y       �	�#UYc�A�*

lossr�q=��       �	��$UYc�A�*

loss��>�<�       �	#i%UYc�A�*

loss��<�s�&       �	z6&UYc�A�*

loss6&�<�-c       �	��&UYc�A�*

loss�GF=��m       �	Yn'UYc�A�*

loss��n=��>a       �	(UYc�A�*

lossR�=��       �	;�(UYc�A�*

loss���<X�ml       �	�H)UYc�A�*

loss;	d=��f       �	�)UYc�A�*

loss/��=)�5}       �	�{*UYc�A�*

loss��'=ӊ�       �	e+UYc�A�*

loss�=�9�       �	w�+UYc�A�*

loss&!=��}�       �	Y,UYc�A�*

loss��I=��$2       �	��,UYc�A�*

lossm�Z=��4^       �	��-UYc�A�*

loss� <y(-       �	�.UYc�A�*

loss���<��Uo       �	Y6/UYc�A�*

loss�!�<��g<       �	F�/UYc�A�*

loss���<綑�       �	�0UYc�A�*

loss�!m=�r|8       �	�!1UYc�A�*

loss�Z<R�1�       �	��1UYc�A�*

lossH�<�i~       �	�^2UYc�A�*

loss���=u�3�       �	��2UYc�A�*

loss�J�=�;��       �	?�3UYc�A�*

loss{�;=X:d       �	R�4UYc�A�*

loss�M=/�;�       �	Bx5UYc�A�*

loss�S�=�^q�       �	�6UYc�A�*

lossܓ�=��u�       �	��6UYc�A�*

loss���=`Lܓ       �	�D7UYc�A�*

loss� =�/2E       �	��7UYc�A�*

loss���=L��       �	`�8UYc�A�*

loss�x3=&!f�       �	U39UYc�A�*

loss/t<7�*       �	��9UYc�A�*

lossh� =�3z       �	wd:UYc�A�*

lossք�<
k�       �	��:UYc�A�*

loss���<W6,�       �	��;UYc�A�*

losswVr=O��       �	�4<UYc�A�*

loss��>٣�z       �	�<UYc�A�*

lossx�<p��       �	c�=UYc�A�*

loss��[<���z       �	�|>UYc�A�*

loss��6>�#^�       �	M?UYc�A�*

loss�8=��;       �	+�?UYc�A�*

loss���=ETB�       �	�w@UYc�A�*

loss�M�<��       �	�AUYc�A�*

loss�u)=�0�B       �	��AUYc�A�*

loss�+"<��y       �	�JBUYc�A�*

loss>=���       �	��BUYc�A�*

loss�:<m�+;       �	��CUYc�A�*

lossqb=hm�       �	�"DUYc�A�*

loss�,E=�&��       �	��DUYc�A�*

loss��%=��[       �	�aEUYc�A�*

lossͷ)=ʧ2Y       �	��EUYc�A�*

loss��=B�]�       �	��FUYc�A�*

loss�ڿ<���f       �	C9GUYc�A�*

loss?�=�'q       �	k�GUYc�A�*

loss���=r A       �	�pHUYc�A�*

loss܅u<�20b       �	9IUYc�A�*

loss��<����       �	HKUYc�A�*

loss
�o=O��       �	��KUYc�A�*

loss	�<�U�a       �	��LUYc�A�*

loss��=�
]2       �	9MUYc�A�*

lossH��=۝��       �	�MUYc�A�*

loss��=�s        �	ΥNUYc�A�*

loss{*H=�;��       �	q=OUYc�A�*

loss"Ϡ<�iK       �	��OUYc�A�*

loss��<���       �	'�PUYc�A�*

loss��	=�<�       �	'NQUYc�A�*

loss�f�<ץI       �	7 RUYc�A�*

loss���=d�E       �	�RUYc�A�*

lossR�<��NN       �	�MSUYc�A�*

loss�~�;PMC�       �	��SUYc�A�*

loss�/�;��U$       �	¦TUYc�A�*

loss�*<|�       �	@NUUYc�A�*

lossw�<�:~       �	�UUYc�A�*

loss�|p=XG�\       �	��VUYc�A�*

loss<"�=�.��       �	0�WUYc�A�*

loss-!>͛*.       �	�FXUYc�A�*

lossa��<*���       �	��XUYc�A�*

loss`�Q=��G'       �	4�YUYc�A�*

loss;j�<W�J�       �	!ZUYc�A�*

lossr��<[͢�       �	z�ZUYc�A�*

lossl�-=�Ora       �	_[UYc�A�*

loss/c�<5>�6       �	��[UYc�A�*

loss�/.=Xsx       �	�\UYc�A�*

loss�A:=���.       �	!:]UYc�A�*

loss�A�<�wJ�       �	��]UYc�A�*

loss�Ϛ=��^�       �	ߌ^UYc�A�*

loss{��<9)�[       �	v6_UYc�A�*

loss�D�<��       �	��_UYc�A�*

lossjr{=w��#       �	�w`UYc�A�*

loss6��=����       �	�'aUYc�A�*

loss�i�<k��<       �	;�aUYc�A�*

loss;�u=��3       �	3obUYc�A�*

loss|�=���a       �	�cUYc�A�*

loss�:w<I�7]       �	��cUYc�A�*

loss�D]<��       �	�IdUYc�A�*

loss��l=��a\       �	o�dUYc�A�*

loss�( =�^Z�       �	�eUYc�A�*

loss)�B<�o�       �	�DfUYc�A�*

loss�ȧ=W2s�       �	��fUYc�A�*

loss�Ռ=L3/       �	<�gUYc�A�*

loss��7=	�s�       �	1(hUYc�A�*

loss6��=*���       �	��hUYc�A�*

lossΐ<ʢEz       �	�oiUYc�A�*

losss�A<�m`h       �	)jUYc�A�*

loss4��=�.�N       �	9�jUYc�A�*

loss\4&<TRW>       �	�pkUYc�A�*

loss���=h!NM       �	VlUYc�A�*

loss���=��o�       �	=�lUYc�A�*

lossto�=4L�       �	j�mUYc�A�*

loss�r�<hI��       �	gGnUYc�A�*

loss�z�<�^�       �	�nUYc�A�*

loss; <��]       �	oUYc�A�*

loss��<�V�       �	 (pUYc�A�*

loss`�=z��e       �	X�pUYc�A�*

loss��<��]       �	�qUYc�A�*

loss���<_���       �	�MrUYc�A�*

loss�<2=���       �	sUYc�A�*

lossA��<E��       �	O�sUYc�A�*

loss�4�<�H       �	�]tUYc�A�*

loss�=�k��       �	�uUYc�A�*

loss�_x=/#l�       �	�uUYc�A�*

lossI�^=�˺d       �	΍vUYc�A�*

loss�Pl<P��       �	z5wUYc�A�*

loss;a<���       �	��wUYc�A�*

lossz
=��       �	�mxUYc�A�*

loss��h;X��B       �	�	yUYc�A�*

loss��<�9�       �	ӤyUYc�A�*

loss�2a<H\�P       �	,DzUYc�A�*

lossv�,=��~1       �	��zUYc�A�*

losss��=��e�       �	�{UYc�A�*

lossx�z<<       �	G|UYc�A�*

loss?"�<� j"       �	G�|UYc�A�*

lossQ�<Z�d:       �	_�}UYc�A�*

loss�"�<2e�       �	K9~UYc�A�*

lossq��=Ǌ�       �	��~UYc�A�*

loss��2=�7@�       �	�oUYc�A�*

loss;�Z=�f�}       �	x�UYc�A�*

loss[J�=���       �	C��UYc�A�*

lossϘ�<�?x?       �	1E�UYc�A�*

lossz"=p36       �	L�UYc�A�*

loss�%={�       �	�~�UYc�A�*

loss�a�<�'r       �	�UYc�A�*

loss���;�M��       �	D��UYc�A�*

loss��<�R��       �	�[�UYc�A�*

loss➜<�ό�       �	v��UYc�A�*

loss�û=g���       �	���UYc�A�*

lossq�=�piE       �	�7�UYc�A�*

loss��U>�وo       �	�ՆUYc�A�*

loss�G*>��c�       �	T��UYc�A�*

lossQ��=���k       �	P�UYc�A�*

lossM��<����       �	��UYc�A�*

loss�~�<�� �       �	���UYc�A�*

loss�?a<x��>       �	�F�UYc�A�*

loss�]p=��1       �	F�UYc�A�*

loss��=l$�       �	���UYc�A�*

loss^<8�F�       �	81�UYc�A�*

lossn�=ɛ�Q       �	R֌UYc�A�*

loss�M	=��       �	�|�UYc�A�*

loss)t<�-�       �	��UYc�A�*

loss�%=�>#       �	��UYc�A�*

loss��-= ';       �	�Y�UYc�A�*

loss�EX<�.��       �	2�UYc�A�*

lossx#K<1�[�       �	ٳ�UYc�A�*

lossO0c=<�O�       �	�Z�UYc�A�*

loss澨=oY�       �	I��UYc�A�*

lossw�=���       �	[��UYc�A�*

loss���<w��       �	�6�UYc�A�*

loss2�<����       �	�דUYc�A�*

loss�g#=��ʼ       �	y�UYc�A�*

loss�}�=)@8�       �	@�UYc�A�*

loss��<�-       �	ꯕUYc�A�*

loss�*,=[@�       �	�M�UYc�A�*

loss6�8<�T�       �	���UYc�A�*

loss���=t���       �	���UYc�A�*

loss��'=`>�       �	�P�UYc�A�*

loss[�=/�(�       �	,�UYc�A�*

loss�["=�A�       �	C��UYc�A�*

lossS̄=* �R       �	�,�UYc�A�*

loss�=�Az+       �	zŚUYc�A�*

loss�;V=E#��       �	d�UYc�A�*

loss�H =.�7       �	=�UYc�A�*

loss���=Bu]C       �	P��UYc�A�*

lossI �=*m!j       �	�T�UYc�A�*

lossR=u<n�x       �	���UYc�A�*

lossowQ<}��>       �	��UYc�A�*

loss�U9;��'h       �	Y0�UYc�A�*

loss]hP<kW/�       �	&�UYc�A�*

loss\k�<%��D       �	R�UYc�A�*

loss���=�f�       �	N(�UYc�A�*

lossS&�<�H       �	l̡UYc�A�*

loss�Jb=Ϻ��       �	o�UYc�A�*

loss=H#=�Q       �	u>�UYc�A�*

loss�3=!=��       �	uu�UYc�A�*

loss�L>��       �	��UYc�A�*

loss$�<��!       �	�6�UYc�A�*

loss'=3_R       �	ަUYc�A�*

loss�|=�v7       �	��UYc�A�*

loss� J=hC�       �	S"�UYc�A�*

loss@F�=�62�       �	�ҨUYc�A�*

loss�r= �j�       �	3p�UYc�A�*

loss͞�=,�H�       �	��UYc�A�*

loss݄d<O�Ő       �	���UYc�A�*

loss���<p'       �	���UYc�A�*

loss z�<1�M       �	�%�UYc�A�*

losse�=6r�F       �	�ˬUYc�A�*

loss�;W=SV݌       �	Ae�UYc�A�*

loss�Z�=���0       �	���UYc�A�*

lossv�<�L�       �	��UYc�A�*

loss�b=�LM�       �	=�UYc�A�*

loss���=zlD       �	�ԯUYc�A�*

loss��0=J]ʾ       �	{��UYc�A�*

lossjf;�MG�       �	��UYc�A�*

loss�8<�B�       �	&�UYc�A�*

loss�[�<��m�       �	�̲UYc�A�*

loss��'<6�lQ       �	�q�UYc�A�*

loss��J=�F�       �	��UYc�A�*

loss�=��       �	���UYc�A�*

loss�/y=Rx        �	�M�UYc�A�*

loss�!�<�[�       �	��UYc�A�*

loss,�<"6       �	���UYc�A�*

lossqq�;��5/       �	ߤ�UYc�A�*

loss�K�<5g�       �	2<�UYc�A�*

loss|�J=^��       �	ظUYc�A�*

lossj�;�<B�       �	 s�UYc�A�*

loss��<-���       �	��UYc�A�*

loss=��=�[�       �	F��UYc�A�*

loss���=��       �	�c�UYc�A�*

losssdC<��?!       �	��UYc�A�*

loss@��=�Iӯ       �	���UYc�A�*

loss
��=�c��       �	H�UYc�A�*

loss��=��j       �	�޽UYc�A�*

loss�N=��^�       �	�v�UYc�A�*

loss�!�<<2�       �	��UYc�A�*

lossfL�;V��       �	�UYc�A�*

loss��=�F�       �	HP�UYc�A�*

loss|2=)(�P       �	��UYc�A�*

loss,��=���q       �	���UYc�A�*

loss|{l<��r�       �	�_�UYc�A�*

loss���=�:�d       �	V�UYc�A�*

losssP�<RpT       �	*��UYc�A�*

loss֧
= ���       �	,G�UYc�A�*

loss[Y,=Y��       �	x%�UYc�A�*

loss��=��<�       �	���UYc�A�*

lossH��=38E       �	Ee�UYc�A�*

loss&��<:DU       �	>	�UYc�A�*

loss%L;���1       �	C��UYc�A�*

lossk#=���g       �	�?�UYc�A�*

loss<�=ۭ��       �	C��UYc�A�*

lossL�=z       �	i��UYc�A�*

loss���=߫~�       �	-$�UYc�A�*

lossM��<ݻ��       �	���UYc�A�*

loss���<�|�       �	^f�UYc�A�*

loss$�c=9�       �	7��UYc�A�*

loss�<�H��       �	ӣ�UYc�A�*

loss$qV=Jj[�       �	�<�UYc�A�*

loss{�=N�qb       �	���UYc�A�*

loss:�+>�1�       �	�m�UYc�A�*

lossA��<���       �	.�UYc�A�*

loss�]:<��<�       �	f��UYc�A�*

lossv?�=��Ί       �	�C�UYc�A�*

loss��>4��       �	��UYc�A�*

lossO��<yj�       �	���UYc�A�*

lossj8�<��       �	X�UYc�A�*

lossS<|��\       �	���UYc�A�*

lossG)=`�ۏ       �	$_�UYc�A�*

loss��>�0�       �	 ��UYc�A�*

loss�\F=��%=       �	|��UYc�A�*

lossŮ[=R�r       �	�F�UYc�A�*

loss�&=��       �	���UYc�A�*

loss�1M=�^%�       �	c��UYc�A�*

loss�T;�J<�       �	�%�UYc�A�*

loss��$<�/��       �	���UYc�A�*

lossH�=b��       �	�c�UYc�A�*

losso�z=[��       �	���UYc�A�*

loss&Ѯ=����       �	���UYc�A�*

loss�S�=ѩ��       �	3�UYc�A�*

loss}��=!�6�       �	���UYc�A�*

loss͕= !�       �	�h�UYc�A�*

loss�=F=�'c       �	�UYc�A�*

loss�:�=df��       �	b��UYc�A�*

loss�&�=Z��       �	���UYc�A�*

loss��*=�"f�       �	i7�UYc�A�*

loss��<b��       �	8��UYc�A�*

loss}��=?V|y       �	�y�UYc�A�*

loss��<==.       �	�UYc�A�*

loss���=�R�       �	%��UYc�A�*

loss��<7Ա       �	K�UYc�A�*

loss��a<i���       �	���UYc�A�*

loss b�<����       �	���UYc�A�*

loss�=TM��       �	Q-�UYc�A�*

loss�	e=�r�B       �	C��UYc�A�*

loss���<���       �	bh�UYc�A�*

loss6/~=V �i       �	>�UYc�A�*

lossEaR=�6       �	]��UYc�A�*

loss�L�=a��       �	�L�UYc�A�*

lossOq�=�VM       �	d��UYc�A�*

loss��9=���       �	>��UYc�A�*

loss�޸=Q��       �	2�UYc�A�*

loss��;���|       �	��UYc�A�*

loss��r<����       �	��UYc�A�*

loss��3=��V       �	Ǽ�UYc�A�*

loss���<�`       �	Hl�UYc�A�*

loss�Z�<et�       �	��UYc�A�*

lossZm^=鸺\       �	1��UYc�A�*

loss��=熫       �	J]�UYc�A�*

loss{<��;       �	a��UYc�A�*

loss�yO=��+	       �	��UYc�A�*

loss��=��a       �	�G�UYc�A�*

loss���<�,w,       �	�O�UYc�A�*

loss��=����       �	���UYc�A�*

loss�V=�5wC       �	���UYc�A�*

loss?�=���       �	i�UYc�A�*

loss k�<�M��       �	$��UYc�A�*

lossm��=3v��       �	�J�UYc�A�*

loss��<Ӱ�       �	���UYc�A�*

loss<�pڷ       �	Ox�UYc�A�*

loss���=	N�/       �	f�UYc�A�*

loss&"�=��A�       �	E��UYc�A�*

lossQ�<M���       �		o�UYc�A�*

loss���<�QɆ       �	��UYc�A�*

loss���<���       �	�V�UYc�A�*

loss�7�=z�       �	���UYc�A�*

loss��=F�E       �	Y��UYc�A�*

loss��'=֭�b       �	�C�UYc�A�*

loss�f=,x��       �	���UYc�A�*

lossl̷<h���       �	ڭ�UYc�A�*

loss���=',B�       �	_D�UYc�A�*

loss]l*<�|�       �	���UYc�A�*

lossX�]=M��       �	i��UYc�A�*

loss��<R\G       �	A(�UYc�A�*

loss;FM=tT:       �	o��UYc�A�*

lossj�4=���       �	ao�UYc�A�*

loss��<�$��       �	� VYc�A�*

loss�6�=Aq��       �	ף VYc�A�*

loss�x=��-�       �	G=VYc�A�*

loss*�=YE̗       �	jVYc�A�*

loss-��<�>��       �	A�VYc�A�*

loss�I�=˔te       �	�WVYc�A�*

losspE�=1��j       �	��VYc�A�*

lossۻy=?�%�       �	��VYc�A�*

lossD=쥥       �	�:VYc�A�*

loss7�<fV\_       �	��VYc�A�*

loss��=��<�       �	�mVYc�A�*

loss�ܼ<k���       �	5
VYc�A�*

loss$��= �2<       �	@�VYc�A�*

loss�A�=�IL�       �	u>VYc�A�*

loss\M>Vɏ�       �	t%	VYc�A�*

loss��<J��B       �	�	VYc�A�*

loss�|�<��s       �	�V
VYc�A�*

loss��[=��       �	o�
VYc�A�*

loss�ڕ=��       �	΍VYc�A�*

loss!R=9gl        �	�AVYc�A�*

loss�J=e��I       �	g�VYc�A�*

loss|'<���z       �	?�VYc�A�*

lossآ�=ٿ`�       �	�3VYc�A�*

lossV�=���       �	B�VYc�A�*

loss��r=w�JR       �	�hVYc�A�*

loss[}B<��       �	�VYc�A�*

loss�^!=jP       �	�VYc�A�*

loss��X=��+.       �	~8VYc�A�*

loss�$=����       �	��VYc�A�*

lossQς=�T�       �	UmVYc�A�*

loss���<Q�V�       �	.VYc�A�*

loss�!=W��8       �	��VYc�A�*

loss���<�       �	29VYc�A�*

loss�	=M�
       �	0�VYc�A�*

loss�=�a�.       �	TrVYc�A�*

loss��<���       �	0VYc�A�*

loss]W�<��,�       �	��VYc�A�*

lossC�<<&6 �       �	�SVYc�A�*

loss�w=��L       �	rVYc�A�*

loss���=��V       �	ϽVYc�A�*

loss�2�=��h;       �	<jVYc�A�*

loss ��=��0       �	�PVYc�A�*

loss�6=���g       �	��VYc�A�*

lossт�;nQ+R       �	kcVYc�A�*

loss!�<Il�       �	9	VYc�A�*

loss$��=t���       �	I�VYc�A�*

loss��=bh�       �	TRVYc�A�*

loss$r;=y%V�       �	�VYc�A�*

loss�.�<�m��       �	�VYc�A�*

loss�=�=>(�       �	�k VYc�A�*

loss?"�<�8ǁ       �	�$!VYc�A�*

loss�Q�;2��       �	��!VYc�A�*

loss�Ǭ=ԫ�]       �	ʍ"VYc�A�*

lossF!�=V�V       �	�:#VYc�A�*

loss���=\`��       �	2�#VYc�A�*

loss��+=|D	Z       �	�$VYc�A�*

losso"@=}n�n       �	3T%VYc�A�*

loss$�b<$��       �	?&VYc�A�*

loss*�<�2M>       �	Q�&VYc�A�*

loss�r=��̈       �	cd'VYc�A�*

loss�Q<d��X       �	-(VYc�A�*

loss�s=_qGD       �	d�(VYc�A�*

loss:��<H1�       �	�[)VYc�A�*

loss$�=�p��       �	�)VYc�A�*

loss��W<r �Z       �	�*VYc�A�*

loss�O[=�F�       �	�?+VYc�A�*

loss�nu<��^�       �	��+VYc�A�*

lossN�=m�O       �	��,VYc�A�*

lossq��;�V%�       �	�-VYc�A�*

loss�t�=�B��       �	
�-VYc�A�*

lossq��;��}       �	c.VYc�A�*

loss�$[=�X�u       �	�/VYc�A�*

loss��=��e�       �	�/VYc�A�*

loss*��;i�d�       �	WC0VYc�A�*

loss�	-<�~       �	�B1VYc�A�*

loss}d�<J��       �	��1VYc�A�*

loss|Y�<p�       �	��2VYc�A�*

loss�
	<#��[       �	�43VYc�A�*

losssS�=��%�       �	S�3VYc�A�*

loss��_>�d��       �	Lp4VYc�A�*

loss6�U=� �%       �	�65VYc�A�*

loss��;��}�       �	��5VYc�A�*

lossڡW=g�{       �	�l6VYc�A�*

loss��%=V�"       �	�7VYc�A�*

loss���<	��       �	��7VYc�A�*

lossф�<HD�~       �	ZE8VYc�A�*

lossdߗ=2���       �	��8VYc�A�*

loss�M=E@�       �	�z9VYc�A�*

loss���<��ѝ       �	a:VYc�A�*

loss)�<w9�       �	��:VYc�A�*

loss�=@χ�       �	�d;VYc�A�*

loss�,�;���y       �	� <VYc�A�*

lossЙ=�-�       �	��<VYc�A�*

loss#�<�K��       �	�G=VYc�A�*

loss�v<8ւ�       �	m�=VYc�A�*

lossX�j=�t6       �	z�>VYc�A�*

loss$=�<"�8       �	�??VYc�A�*

loss���<	b'       �	��?VYc�A�*

loss���=�]�N       �	r@VYc�A�*

lossm��=A�K;       �	�AVYc�A�*

loss��;�<R;       �	��AVYc�A�*

loss4x;��*�       �	�fBVYc�A�*

loss�h=)�       �	�CVYc�A�*

lossV�=|���       �	�CVYc�A�*

loss$!�<�AOV       �	�.DVYc�A�*

losst׶<��       �	��DVYc�A�*

loss_<�L       �	N�EVYc�A�*

loss[f
=Zʻ�       �	�%FVYc�A�*

lossj�=Pxr       �	�FVYc�A�*

loss�_�;����       �	�RGVYc�A�*

lossX5�=gT �       �	dxHVYc�A�*

loss}�<9���       �	XIVYc�A�*

loss,\�<.���       �	��IVYc�A�*

losswއ<�+�       �	�nJVYc�A�*

lossx�L=j�a5       �	�sKVYc�A�*

loss�}=�Ұ�       �	ILVYc�A�*

loss6��<�+��       �	��LVYc�A�*

loss��<W�8�       �	)YMVYc�A�*

lossh��<��0�       �	��MVYc�A�*

loss=H�<(��       �	�NVYc�A�*

loss��<x�_�       �	P8OVYc�A�*

lossπ�:x~>>       �	x�OVYc�A�*

loss&ga=��(�       �	uPVYc�A�*

lossOܓ={�~       �	t(QVYc�A�*

loss#�8=�7��       �	X�QVYc�A�*

lossrv�<�^��       �	�jRVYc�A�*

loss��<���       �	��RVYc�A�*

loss�0�=�l�a       �	��SVYc�A�*

lossTk�;eh)=       �	|,TVYc�A�*

loss<��<$�!       �	�TVYc�A�*

loss,x-=��O�       �	�VVYc�A�*

loss�,<n���       �	ٰVVYc�A�*

loss;�9�Ē�       �	�@XVYc�A�*

loss;��;��o�       �	h�XVYc�A�*

loss��J<���       �	�YVYc�A�*

lossa8r<�ܟ�       �	P8ZVYc�A�*

loss�u�;|��       �	.�ZVYc�A�*

loss�tA:/7d�       �	8�[VYc�A�*

loss��;�i�       �	�0\VYc�A�*

loss�z�9�g^j       �	��\VYc�A�*

loss(ң:�W}�       �	�w]VYc�A�*

loss���9��       �	^VYc�A�*

loss �<���       �	��^VYc�A�*

loss��{=��Z       �	�[_VYc�A�*

loss0q<�0�       �	Y�_VYc�A�*

loss�O;?�       �	ٔ`VYc�A�*

losst��<@�~�       �	�BaVYc�A�*

loss	�>O{�U       �	b�aVYc�A�*

loss{�;0|�I       �	{�bVYc�A�*

loss��a>�4f�       �	�ScVYc�A�*

loss�e=��       �	}�cVYc�A�*

loss"ӏ=�6��       �	,�dVYc�A�*

loss��l=!�('       �	reVYc�A�*

loss͊<�!�T       �	t�eVYc�A�*

loss��==L�f�       �	FafVYc�A�*

loss��(=���       �	��fVYc�A�*

lossQ�~=�H��       �	�gVYc�A�*

lossZ)�<��)       �	�)hVYc�A�*

loss���<(��       �	M�hVYc�A�*

loss�;p=n� 1       �	dWiVYc�A�*

loss��=?1�       �	[�iVYc�A�*

lossD�<E�*W       �	��jVYc�A�*

lossv�<��       �	ρkVYc�A�*

loss7��=�4E       �	�lVYc�A�*

loss�?=#.�U       �	�lVYc�A�*

loss���<��       �	�RmVYc�A�*

loss�@=jф�       �	��mVYc�A�*

lossJ�c=�ֶ]       �	�nVYc�A�*

loss�V<
���       �	oVYc�A�*

lossǞ=��]�       �	��oVYc�A�*

loss�&g<�       �	^pVYc�A�*

loss�C<�:d       �	�pVYc�A�*

loss"t=�*;       �	��qVYc�A�*

loss�D<��<       �	E�rVYc�A�*

loss10�<{�       �	i�sVYc�A�*

loss�9�<ޢ       �	^�tVYc�A�*

loss�y�=����       �	/4uVYc�A�*

loss'H�=)6       �	��uVYc�A�*

loss��g;<��/       �	�vVYc�A�*

lossm��<7X�       �	TwVYc�A�*

loss�<ُ��       �	��wVYc�A�*

loss*&x<fH+I       �	\�xVYc�A�*

loss<;=%�8       �	|'yVYc�A�*

lossn�1<PVX       �	�yVYc�A�*

loss�g�;��0       �	�UzVYc�A�*

loss�)}=�-i�       �	Z~{VYc�A�*

lossr��=X       �	X|VYc�A�*

loss��=���       �	�|VYc�A�*

loss���<�o��       �	�V}VYc�A�*

lossi?<WL�       �	��}VYc�A�*

lossZŒ<)L�       �	��~VYc�A�*

loss �a<�RK       �	�9VYc�A�*

loss�~=@#H"       �	=�VYc�A�*

loss��<j�+       �	�k�VYc�A�*

lossܽ=��4�       �	J
�VYc�A�*

lossms<�;       �	ޮ�VYc�A�*

losspa={�	�       �	�A�VYc�A�*

lossI��;��       �	;��VYc�A�*

loss�<Ww?       �	K��VYc�A�*

loss���<�w)       �	� �VYc�A�*

loss�?j=i`Y�       �	���VYc�A�*

loss�qX=)�p�       �	�P�VYc�A�*

lossJ��=WY@�       �	:�VYc�A�*

lossj��<��       �	���VYc�A�*

lossaR�<;T�       �	K;�VYc�A�*

lossԽB=U�l       �	RӠVYc�A�*

loss�=�5A       �	0��VYc�A�*

loss�!�=���       �	�9�VYc�A�*

loss�&b=g�H       �	_ԢVYc�A�*

lossB��<�m (       �	�n�VYc�A�*

loss-�Z=���       �	��VYc�A�*

loss��=��d       �	��VYc�A�*

lossrf=����       �	@��VYc�A�*

loss��=�Ӻg       �	��VYc�A�*

loss��8=ȚC'       �	�ĦVYc�A�*

loss���:G$�       �	�c�VYc�A�*

loss3N<�?       �	~�VYc�A�*

loss���<H��h       �	
��VYc�A�*

loss1��=���       �	�E�VYc�A�*

loss*W=R@�F       �	=�VYc�A�*

loss�>�?35       �	H��VYc�A�*

loss�#�<}!7�       �	�+�VYc�A�*

loss%o{=LQ	       �	)̫VYc�A�*

loss �	=[H       �	9b�VYc�A�*

loss�,=�̱@       �	�VYc�A�*

loss�?b=�ς       �	���VYc�A�*

losso5�<%Du:       �	;Q�VYc�A�*

lossa�<�@H       �	��VYc�A�*

loss2��<Œn�       �	���VYc�A�*

loss��g=@�\:       �	~�VYc�A�*

loss���<Ο       �	���VYc�A�*

loss1��<9.�       �	�O�VYc�A�*

lossw�>f1N�       �	m�VYc�A�*

loss1!�<�װ�       �	�{�VYc�A�*

loss4�T=ٟ��       �	{�VYc�A�*

loss7��;B�       �		��VYc�A�*

loss��<0z��       �	>�VYc�A�*

loss��!>���n       �	�ִVYc�A�*

loss�9�=���H       �	�i�VYc�A�*

lossA�=@�o�       �	T �VYc�A�*

loss�7=���       �	x��VYc�A�*

loss�)�<��m       �	�1�VYc�A�*

loss�'=f��L       �	�̷VYc�A�*

loss�|�=|;�P       �	�e�VYc�A�*

loss�h�<T�       �	���VYc�A�*

lossZ�	=�1��       �	���VYc�A�*

loss Y=�q       �	c{�VYc�A�*

loss��<F       �	�e�VYc�A�*

lossW	�<�؅?       �	\�VYc�A�*

losso��<�$�       �	���VYc�A�*

loss�{c=Vt�       �	;:�VYc�A�*

loss��8<�#*�       �	X�VYc�A�*

loss}��=��A       �	F|�VYc�A�*

loss��<��܇       �	��VYc�A�*

loss�}R;�$�       �	L��VYc�A�*

loss���;���       �	�=�VYc�A�*

loss��;�8(�       �	���VYc�A�*

loss�
<���:       �	�h�VYc�A�*

loss�}�=�d+�       �	���VYc�A�*

loss��=T��       �	��VYc�A�*

lossȦ�=��Z       �	�7�VYc�A�*

lossV-(;���       �	_y�VYc�A�*

lossV�i<�w�       �	��VYc�A�*

lossO <�5��       �	c��VYc�A�*

loss<�C̔       �	8i�VYc�A�*

lossDI<��/�       �	��VYc�A�*

loss@�<8t��       �	��VYc�A�*

loss�L=���       �	�^�VYc�A�*

loss��>���h       �	��VYc�A�*

loss=�@       �	��VYc�A�*

loss��<�x�-       �	�S�VYc�A�*

loss�\<a���       �	4��VYc�A�*

loss��=����       �	:��VYc�A�*

loss�E�<���`       �	!>�VYc�A�*

lossX<�$��       �	��VYc�A�*

loss`�=R9]�       �	�n�VYc�A�*

loss1N�<#��       �	�
�VYc�A�*

loss���<>`�`       �	��VYc�A�*

lossnI�<�n�       �	r7�VYc�A�*

loss�?=q���       �	���VYc�A�*

loss��=�gVt       �	#f�VYc�A�*

loss*�<&��/       �	Q��VYc�A�*

loss��$=p]��       �	G��VYc�A�*

loss�J�<��-       �	�J�VYc�A�*

lossj��<�4��       �	���VYc�A�*

loss8;,<��ǲ       �	�y�VYc�A�*

lossY�<XRK       �	a�VYc�A�*

lossX�<���8       �	��VYc�A�*

loss��=�0�       �	wf�VYc�A�*

loss��:=à!       �	"��VYc�A�*

lossĉ�=��c       �	`��VYc�A�*

lossz��<�5I       �	�M�VYc�A�*

lossf�= ���       �	y��VYc�A�*

loss��=�       �	 }�VYc�A�*

loss��t<v���       �	Y�VYc�A�*

loss��=c���       �	��VYc�A�*

loss���<p�/d       �	C��VYc�A�*

loss%=�c��       �	"��VYc�A�*

loss��<晱>       �	��VYc�A�*

lossNq�<�R�m       �	�u�VYc�A�*

loss���;܉P       �	V(�VYc�A�*

lossD�J=>���       �	�P�VYc�A�*

loss-��;�3[�       �	�VYc�A�*

loss��a=FZ��       �	���VYc�A�*

loss�H�<�υ       �	\U�VYc�A�*

lossl�;�n�       �	1��VYc�A�*

lossd�<��F0       �	���VYc�A�*

lossJ��=�2��       �	|&�VYc�A�*

loss7��;����       �	���VYc�A�*

loss��H<w�|       �	�[�VYc�A�*

loss��.=�7��       �	���VYc�A�*

loss_D�<f�D�       �	͒�VYc�A�*

loss�;�À�       �	�)�VYc�A�*

loss�h�;�IM<       �	;��VYc�A�*

lossOF�;��A�       �	GZ�VYc�A�*

loss���=��0�       �	���VYc�A�*

loss@��=�):�       �	p��VYc�A�*

loss�*�=�G�       �	/Q�VYc�A�*

loss��=>��>       �	*��VYc�A�*

loss��h<����       �	+��VYc�A�*

lossŨ�<4�(0       �	K"�VYc�A�*

loss��U<���9       �	U��VYc�A�*

loss�:�=�x��       �	\�VYc�A�*

loss�1�;�~^6       �	{��VYc�A�*

loss6ܫ<�T�A       �	%��VYc�A�*

loss ��<+�i2       �	�W�VYc�A�*

loss�.^;LZ       �	���VYc�A�*

loss=ls=Gi�Y       �	v��VYc�A�*

loss��={N�       �	0)�VYc�A�*

loss�a!=�s       �	���VYc�A�*

loss�k=�T�C       �	�g�VYc�A�*

lossHa9<'xT�       �	)�VYc�A�*

loss�݃<�>
       �	��VYc�A�*

loss��=rV�       �	$D�VYc�A�*

loss��<���u       �	0��VYc�A�*

loss��=���$       �	?s�VYc�A�*

lossx5�<���       �	��VYc�A�*

loss��;�Ew       �	ʩ�VYc�A�*

loss�D<��E�       �	��VYc�A�*

loss�VM<�pG       �	}#�VYc�A�*

lossc�<� �       �	'��VYc�A�*

loss�[k=(�       �	~W�VYc�A�*

loss�}�=����       �	��VYc�A�*

loss��]=c�       �	�VYc�A�*

loss��;�\�C       �	$�VYc�A�*

loss��'=�nw%       �	V��VYc�A�*

loss�^�;Ej�:       �	�V�VYc�A�*

loss��;����       �	���VYc�A�*

loss�=�7�J       �	��VYc�A�*

loss��+<��+�       �	&�VYc�A�*

loss�*2=,tӭ       �	���VYc�A�*

lossa|0=l,�       �	�p WYc�A�*

loss{�H=�V��       �	�WYc�A�*

loss��^=��{x       �	D�WYc�A�*

loss���:��I       �	AWYc�A�*

loss��<5w       �	^�WYc�A�*

loss%Y�=�_       �	��WYc�A�*

loss;��=��       �	�:WYc�A�*

lossJ��<U\f�       �	 �WYc�A�*

lossH =�j       �	�WYc�A�*

loss�{�<���       �	/WYc�A�*

loss�p<)P�/       �	��WYc�A�*

lossMX;�sy�       �	O�WYc�A�*

loss ��<C���       �	�>WYc�A�*

lossO�=�8�       �	/�WYc�A�*

loss_7�<��t�       �	��	WYc�A�*

loss9%�<R�v�       �	d
WYc�A�*

loss�Ő<7��       �	M�
WYc�A�*

loss�z<Qc��       �	�WWYc�A�*

lossA&2=rG6�       �	�WYc�A�*

loss�Y�<�@b�       �	ÛWYc�A�*

loss�d�;4
�       �	8WYc�A�*

loss�D�=�ԩ�       �	��WYc�A�*

loss,g�:��|       �	0hWYc�A�*

loss��=S|j       �	�WYc�A�*

lossa�$=c�a�       �	��WYc�A�*

loss���=�Ǵ�       �	�>WYc�A�*

losst�@=9
�       �	��WYc�A�*

loss E<��       �	�oWYc�A�*

loss�P=���       �	�WYc�A�*

loss [7=i��I       �	��WYc�A�*

loss��:=�	��       �	�:WYc�A�*

loss���<Id��       �	��WYc�A�*

lossl<���       �	lWYc�A�*

loss@�f;%�t�       �	�EWYc�A�*

loss3�=݆��       �	��WYc�A�*

loss_��;yo9�       �	P�WYc�A�*

loss���=Z#�m       �	 WYc�A�*

loss��=���       �	�WYc�A�*

loss�y�<����       �	�QWYc�A�*

loss��$=;d��       �	G�WYc�A�*

loss�!<d�,t       �	�|WYc�A�*

lossAp3<)�N       �	�WYc�A�*

losse{$;״�       �	��WYc�A�*

loss���;�__�       �	�RWYc�A�*

lossHN�<��       �	Z�WYc�A�*

loss�jF=�̻�       �	YWYc�A�*

loss*�s=��:�       �	��WYc�A�*

loss�6�<��Z       �	�JWYc�A�*

loss��<�tu2       �	��WYc�A�*

loss��q<�m!h       �	hzWYc�A�*

lossx�B=��14       �	c� WYc�A�*

loss���=؄��       �	�4!WYc�A�*

loss��=	g       �	��!WYc�A�*

loss2|=��5�       �	�{"WYc�A�*

loss���=�qF       �	a#WYc�A�*

loss�Q�< �`1       �	ٵ#WYc�A�*

loss��<<_�C�       �	'L$WYc�A�*

loss契<^3y�       �	��$WYc�A�*

loss<��<�w�w       �	v%WYc�A�*

lossʝ_=���m       �	=&WYc�A�*

loss���<g�`�       �	]�&WYc�A�*

loss���;Q�/       �	��'WYc�A�*

loss��=        �	s*(WYc�A�*

loss!m�=�[�       �	��(WYc�A�*

lossPt>��}       �	�c)WYc�A�*

loss�H>t�k       �	L�)WYc�A�*

loss��>�p�L       �	��*WYc�A�*

loss=X�=5 ��       �	�:+WYc�A�*

loss[ �<��       �	n�+WYc�A�*

lossh,3=�J�R       �	�y,WYc�A�*

lossHՍ<��;       �	F%-WYc�A�*

lossڧB=K��       �	��-WYc�A�*

loss}�;��       �	�_.WYc�A�*

lossf�=|�       �	� /WYc�A�*

lossz^q=�cӝ       �	b�/WYc�A�*

loss:_<QӰ<       �	�H0WYc�A�*

lossv�"=��T�       �	K�0WYc�A�*

loss);=��I       �	 ~1WYc�A�*

loss�\3=�A^�       �	�2WYc�A�*

loss1�="�       �	�q3WYc�A�*

lossf9�<��       �	�4WYc�A�*

loss�ձ<��os       �	I�5WYc�A�*

loss;T)=�~��       �	�17WYc�A�*

loss$��=t��       �	D�8WYc�A�*

loss�i�=��7       �	��9WYc�A�*

loss�x<a]j#       �	��:WYc�A�*

loss���<^�Wd       �	�1;WYc�A�*

lossDQ<�e��       �	ē<WYc�A�*

loss6U<�x�       �	>Y=WYc�A�*

loss��=wwG       �	>WYc�A�*

loss��3=9�6�       �	�>WYc�A�*

loss%��=.�W       �	�q?WYc�A�*

loss@�d=�y)<       �	�@WYc�A�*

loss�<�xw]       �	+�@WYc�A�*

loss�2�=+h�y       �	�?AWYc�A�*

loss�3=�m(�       �	f�AWYc�A�*

loss/�&=U       �	�rBWYc�A�*

loss�t�<-��       �	WCWYc�A�*

loss��=f ��       �	r�CWYc�A�*

loss/��=���       �	�XDWYc�A�*

loss.�1=}��j       �	��DWYc�A�*

lossIQ�;b��0       �	��EWYc�A�*

loss�Ε<.H4B       �	ګFWYc�A�*

loss�sk<���/       �	�`GWYc�A�*

loss�ܪ<^ׅW       �	L�GWYc�A�*

loss�'�<+�ޭ       �	��HWYc�A�*

loss`��<_��       �	j3IWYc�A�*

loss�R�<�2       �	��IWYc�A�*

loss��n<|��}       �	U�JWYc�A�*

losstZ�=�7\       �	�9KWYc�A�*

loss���=i�ϟ       �	7�KWYc�A�*

loss��=��Y       �	ڒLWYc�A�*

loss&��<P�Rj       �	`YMWYc�A�*

loss.v�<�	       �	D�MWYc�A�*

lossۦs=��0       �	x�NWYc�A�*

loss<]�       �	�>OWYc�A�*

lossM��=�N��       �	��OWYc�A�*

loss�rg=.T�       �	�wPWYc�A�*

loss���<u�	�       �	QWYc�A�*

loss�i�<�3��       �	��QWYc�A�*

loss��==��A�       �	[DRWYc�A�*

loss)$�<US�       �	j�RWYc�A�*

lossUF=�<#       �	�xSWYc�A�*

lossuN=���       �	TWYc�A�*

lossN<��3       �	6�TWYc�A�*

loss�^=zZ�s       �	�EUWYc�A�*

lossӺ]=���       �	�UWYc�A�*

losshR	=��Q�       �	ByVWYc�A�*

lossc�<SP/       �	+WWYc�A�*

loss� R<�ō�       �	کWWYc�A�*

lossL��=V0�       �	�BXWYc�A�*

loss�3�<Wd_       �	M�XWYc�A�*

loss�E�=�e       �	�qYWYc�A�*

loss�n=ꈘ�       �	��ZWYc�A�*

loss��=��#�       �	�\WYc�A�*

loss�
</	&&       �	%�\WYc�A�*

lossr�
<���h       �	5�]WYc�A�*

loss�2<�x�       �	�^WYc�A�*

lossڎ�;��jo       �	�W_WYc�A�*

loss3�m=Oх�       �	��_WYc�A�*

loss��;��B       �	\aWYc�A�*

lossH�b<>�*0       �	��aWYc�A�*

loss
��<䤶       �	��bWYc�A�*

loss�y�=��]       �	йcWYc�A�*

loss��<a��&       �	}^dWYc�A�*

loss�+x=V+1�       �	geWYc�A�*

loss��G=�1       �	��eWYc�A�*

loss���=���m       �	�\fWYc�A�*

loss��=W1Ç       �	�	gWYc�A�*

loss��)=1`�[       �	�gWYc�A�*

lossG�=�+ &       �	�bhWYc�A�*

loss
,�=����       �	�iWYc�A�*

lossc��=+��&       �	��iWYc�A�*

loss�l�=�4�       �	�qjWYc�A�*

loss�#�<�X��       �	gkWYc�A�*

loss�=�=�r       �	֨kWYc�A�*

lossm�=�ʑ       �	�GlWYc�A�*

loss�?b=�/CG       �	�lWYc�A�*

lossT��<��}�       �	
�mWYc�A�*

loss��f<�*�       �	��nWYc�A�*

lossZ�=��       �	�FoWYc�A�*

lossz'j<�S�       �	��oWYc�A�*

loss�Z�<�u       �	�ypWYc�A�*

loss^؃=R�       �	�qWYc�A�*

loss�d<?�Vr       �	+�qWYc�A�*

loss�=�^�8       �	grWYc�A�*

loss��n<����       �	��rWYc�A�*

loss �=�3�9       �	ɐsWYc�A�*

loss<9=0��M       �	�?tWYc�A�*

loss�==��8       �	GuWYc�A�*

loss#UP=�_       �	Q�uWYc�A�*

loss��<<x&\�       �	�ovWYc�A�*

loss8\=��",       �	�wWYc�A�*

loss�?=�"I�       �	V�wWYc�A�*

lossҽ5<����       �	.xWYc�A�*

lossVT=��#�       �	�yWYc�A�*

loss�M�<T8��       �	X�yWYc�A�*

loss&G0=���       �	�yzWYc�A�*

loss]�<��D       �	�{WYc�A�*

lossd��;�\A       �	��{WYc�A�*

loss��;Y46�       �	X8|WYc�A�*

lossTr�<���<       �	��|WYc�A�*

loss7�=Ʈ�       �	�c}WYc�A�*

loss~v<)}Y       �	>	~WYc�A�*

lossjv�<�2x       �	��~WYc�A�*

loss�}:=ȢnN       �	�BWYc�A�*

loss�=��G       �	�WYc�A�*

loss4;7��z       �	�z�WYc�A�*

lossCJw<]Ȁ�       �	'h�WYc�A�*

loss��<vT�       �	3��WYc�A�*

loss�h =��(       �	���WYc�A�*

loss&��<*���       �	)�WYc�A�*

loss@��=�^�       �	��WYc�A�*

lossc�k=i�T       �	��WYc�A�*

loss���;i�!Q       �	(�WYc�A�*

loss��=�>g�       �	�ąWYc�A�*

loss��<��(�       �	Dm�WYc�A�*

loss�S�<��+       �	F	�WYc�A�*

loss
��;���g       �	)��WYc�A�*

lossx�D<�k��       �	(I�WYc�A�*

loss�/<��-       �	�ވWYc�A�*

loss˲=�<&       �	�q�WYc�A�*

loss/Ƙ=��
�       �	��WYc�A�*

loss�?</}r�       �	���WYc�A�*

loss�I9<{�6�       �	=+�WYc�A�*

loss�<ʺ�       �	�ËWYc�A�*

loss�Z=k��:       �	dZ�WYc�A�*

loss��=�D�&       �	��WYc�A�*

loss��==���       �	s��WYc�A�*

loss襦;�7k!       �	�WYc�A�*

lossĴ�<��.       �	e��WYc�A�*

loss7��=�o�       �	>�WYc�A�*

lossd=E��       �	r��WYc�A�*

loss� =Ц�       �	�r�WYc�A�*

loss2��=P�?�       �	��WYc�A�*

loss�EK<�4e�       �	Փ�WYc�A�*

loss���<#�b       �	.�WYc�A�*

loss�N = �M       �	ĒWYc�A�*

loss�=\Ԩ       �	�]�WYc�A�*

loss�z�=�)z�       �	U��WYc�A�*

loss%!x=>�{�       �	}��WYc�A�*

loss��<��       �	�,�WYc�A�*

loss(\�=���       �	zWYc�A�*

loss���=�x"_       �	�Z�WYc�A�*

lossf�g=��n�       �	�WYc�A�*

loss���<�@�       �	Ƌ�WYc�A�*

loss�<�KĢ       �	O!�WYc�A�*

losss��<(+�M       �	ܷ�WYc�A�*

losst��=Pp�       �	�M�WYc�A�*

loss��=tݺ�       �	o�WYc�A�*

loss���=em�4       �	숚WYc�A�*

losseP�<.��%       �	/6�WYc�A�*

loss*�];�Ԓ       �	1|�WYc�A�*

loss�Z==�\�       �	$�WYc�A�*

loss�H=7!j�       �	H��WYc�A�*

loss�!|=\9        �	���WYc�A�*

loss��;P���       �	=�WYc�A�*

loss�,�;�f�       �	�נWYc�A�*

loss&j=�7*�       �	dw�WYc�A�*

lossw�;=>�+       �	��WYc�A�*

loss���<�Kz�       �	��WYc�A�*

loss+�=��:E       �	���WYc�A�*

lossf��;�<       �	.�WYc�A�*

loss@>�=�z5       �	��WYc�A�*

loss-��<+.�{       �	�e�WYc�A�*

loss*5<,#q	       �	z��WYc�A�*

loss(J`<��.=       �	���WYc�A�*

loss��"=��w       �	�*�WYc�A�*

loss�<p�-�       �	3çWYc�A�*

loss|,z<��a       �	iV�WYc�A�*

lossԠ_=Q���       �	��WYc�A�*

loss$�C<�o��       �	���WYc�A�*

loss���=? �       �	M�WYc�A�*

loss�2�<��1^       �	��WYc�A�*

loss��=iY}�       �	�~�WYc�A�*

loss.4B='�s�       �	v�WYc�A�*

losst==&       �	E��WYc�A�*

loss��<	��       �	jO�WYc�A�*

loss��=OɊ       �	~�WYc�A�*

loss�l=a�d*       �	À�WYc�A�*

loss|�<�r?t       �	m�WYc�A�*

loss��~=�� �       �	���WYc�A�*

lossX�<K�       �	XV�WYc�A�*

loss�
�=�#��       �	��WYc�A�*

loss���; r       �	���WYc�A�*

loss@t�<��kh       �	�(�WYc�A�*

lossz�	=׌c       �	4��WYc�A�*

loss���=�!2       �	Q�WYc�A�*

loss�~�<����       �	t�WYc�A�*

loss%�<�t��       �	���WYc�A�*

lossh�=1���       �	i8�WYc�A�*

loss�D�=9��#       �	YڵWYc�A�*

loss�^�=`�
�       �	�W�WYc�A�*

loss)b�=r�m       �	���WYc�A�*

loss��H<��       �	@��WYc�A�*

loss��]=��:�       �	^L�WYc�A�*

loss�W�<����       �	���WYc�A�*

lossA<��D�       �	^��WYc�A�*

lossl�<p0=       �	�F�WYc�A�*

loss���<V��        �	u�WYc�A�*

loss��<őu�       �	JB�WYc�A�*

loss�4=���       �	�#�WYc�A�*

loss���<)�5�       �	�ľWYc�A�*

loss�<�A       �	i�WYc�A�*

lossOR}<��/�       �	!�WYc�A�*

loss�Ψ;ͥ�       �	ӥ�WYc�A�*

loss-s�<\��       �	�A�WYc�A�*

loss�
H<��Z�       �	��WYc�A�*

lossԿ=�őj       �	6t�WYc�A�*

lossO�=^đ}       �	�WYc�A�*

lossdK`=�8�o       �	]��WYc�A�*

loss�=��r�       �	{I�WYc�A�*

loss���;+t�^       �	��WYc�A�*

loss�P�<�Q\       �	���WYc�A�*

loss�ۧ=���c       �	�j�WYc�A�*

lossd�;       �	v��WYc�A�*

loss1=�mc       �	͐�WYc�A�*

loss��=r7��       �	t(�WYc�A�*

loss]��=�?�D       �	��WYc�A�*

lossx��<��]       �	�a�WYc�A�*

lossN�+=I�       �	* �WYc�A�*

loss��f=疰�       �	R��WYc�A�*

loss��>�       �	,F�WYc�A�*

loss]�=����       �	��WYc�A�*

loss���<�) 0       �	G��WYc�A�*

losscs=�Zm       �	�E�WYc�A�*

loss=i�a        �		��WYc�A�*

loss��X=L��_       �	By�WYc�A�*

loss���<��       �	��WYc�A�*

loss,�=>�&       �	��WYc�A�*

lossڒ�;��\�       �	�c�WYc�A�*

loss�P>�6�       �	 �WYc�A�*

loss���<�q�       �	��WYc�A�*

lossݳ�<@��+       �	�Q�WYc�A�*

loss���:�       �	�9�WYc�A�*

loss
�8=|T�       �	��WYc�A�*

loss�,=���       �	L��WYc�A�*

loss4S�<�á�       �	0�WYc�A�*

lossL��=�R�       �	���WYc�A�*

loss��7=�ʳ       �	\q�WYc�A�*

loss��<��[�       �	��WYc�A�*

lossd(g=�Бe       �	x��WYc�A�*

loss��
<��       �	�Y�WYc�A�*

lossa$�<��D       �	���WYc�A�*

loss���<b�`�       �	R��WYc�A�*

loss-�=ܹiF       �	�4�WYc�A�*

loss�/<<�HB       �	��WYc�A�*

lossz��=�j��       �	E-�WYc�A�*

lossɰ>>:�       �	Q��WYc�A�*

loss��h=�5�V       �	H��WYc�A�*

lossŝ�<sv3d       �	g��WYc�A�*

lossz��=�S       �	��WYc�A�*

loss
"~=R{1�       �	��WYc�A�*

loss/�g<+Q��       �	���WYc�A�*

lossԚ=ij�       �	)�WYc�A�*

loss��=���       �	���WYc�A�*

loss��@<ԦM       �	���WYc�A�*

loss4�G=��       �	�/�WYc�A�*

loss�]<=�/k       �	���WYc�A�*

lossa�<X*�J       �	o�WYc�A�*

loss��I<t�?       �	)%�WYc�A�*

loss8=�x��       �	:��WYc�A�*

lossdy�;�:��       �	�~�WYc�A�*

lossNh�;<k#       �	�C�WYc�A�*

loss�t�=;j�       �	2��WYc�A�*

loss��;�}@       �	���WYc�A�*

loss�º<Sy*�       �	Su�WYc�A�*

loss��R=�P��       �	��WYc�A�*

loss�� =Y��e       �	[��WYc�A�*

loss�g@<�8M�       �	�N�WYc�A�*

loss�#�<J�       �	���WYc�A�*

loss��^<v{��       �	~�WYc�A�*

loss�Uw=���       �	+�WYc�A�*

loss��l<d�       �	p��WYc�A�*

loss��|=f��       �	�Q�WYc�A�*

lossx}<ȑ��       �	P�WYc�A�*

loss6}�<gA�       �	���WYc�A�*

loss��=T}       �	b�WYc�A�*

loss��;3���       �	���WYc�A�*

loss2=d��       �	��WYc�A�*

loss�6�<Osc       �	oF�WYc�A�*

lossH��=�ޣ�       �	��WYc�A�*

loss�p7<��H       �	���WYc�A�*

loss�@=�O�#       �	e6�WYc�A�*

loss�#�=���*       �	���WYc�A�*

loss4Y<��6�       �	 }�WYc�A�*

loss��<��ɤ       �	
�WYc�A�*

lossݼ;=!��       �	d��WYc�A�*

loss�-�<09�6       �	�F�WYc�A�*

loss{��=>��       �	8�WYc�A�*

losss�X;־H       �	w��WYc�A�*

loss��<�dV�       �	�n�WYc�A�*

loss� =����       �	��WYc�A�*

loss\5�<d�+�       �	���WYc�A�*

lossO"�;���       �	a�WYc�A�*

loss��=F;u�       �	d;�WYc�A�*

lossoYT=u)�F       �	���WYc�A�*

loss��	<g�zv       �	�b�WYc�A�*

loss�� <��8       �	���WYc�A�*

loss6��<u�Zh       �	ɐ�WYc�A�*

lossdy�;'O�t       �	l& XYc�A�*

loss�)N9g��       �	s� XYc�A�*

loss�;�%f       �	UXYc�A�*

loss؉6<��       �	>�XYc�A�*

loss�=H<��T�       �	��XYc�A�*

loss���:9e��       �	�UXYc�A�*

lossX��:A� M       �	ͮXYc�A�*

lossTSH=y[�r       �	�RXYc�A�*

loss�ˤ:� �       �	^�XYc�A�*

loss��Y::�ݘ       �	I�XYc�A�*

loss�o�9���]       �	UXYc�A�*

loss�,V;���       �	VXYc�A�*

loss�*=��-E       �	�XYc�A�*

loss���<��       �	YM	XYc�A�*

lossW't:
q��       �	��	XYc�A�*

loss.�=^#�       �	:�
XYc�A�*

loss���=F���       �	�;XYc�A�*

loss;�<2�       �	��XYc�A�*

lossm�>c�5n       �	urXYc�A�*

loss#��<E�&m       �	�XYc�A�*

loss��]=�n6�       �	�XYc�A�*

lossH��;R�>c       �	&8XYc�A�*

lossi (<;A�       �	O�XYc�A�*

loss��=��"       �	�fXYc�A�*

loss��=a��q       �	��XYc�A�*

lossO��<���G       �	3�XYc�A�*

loss�=S�-       �	�EXYc�A�*

lossAX�<��        �	��XYc�A�*

loss�E=�,�       �	�XYc�A�*

lossvVF=~�7       �	�0XYc�A�*

loss�D�<���_       �	(�XYc�A�*

loss3� =��ȫ       �	�xXYc�A�*

loss��=[��       �	>&XYc�A�*

loss�y�=�`��       �	��XYc�A�*

lossW�K<�H^>       �	�XYc�A�*

loss�F2=0�ّ       �	�jXYc�A�*

loss�~<�fTO       �	� XYc�A�*

loss��@;��D�       �	�XYc�A�*

loss1*�<L�By       �	8/XYc�A�*

lossʋ==
���       �	a�XYc�A�*

lossP]<�l        �	�WXYc�A�*

loss�;�'c`       �	�GXYc�A�*

loss)�H<ɉ�X       �	y�XYc�A�*

losss�=��b�       �	�XYc�A�*

lossf��<���8       �	�JXYc�A�*

loss�K=~(��       �	AfXYc�A�*

lossj��=���       �	�8 XYc�A�*

loss�&�;�UI       �	n� XYc�A�*

loss�B'=G]       �	��!XYc�A�*

loss�x=�NҸ       �	]R"XYc�A�*

loss�<<;3/       �	�#XYc�A�*

loss�;�;�B�_       �	0�#XYc�A�*

lossRf<�鼢       �	�q$XYc�A�*

loss��<p s-       �	�5%XYc�A�*

lossB؄=�e��       �	v�%XYc�A�*

losse��=ׂ�       �	��&XYc�A�*

loss-��<����       �	�H'XYc�A�*

lossF�<�i:       �	��'XYc�A�*

loss���<h��o       �	�(XYc�A�*

loss��a=�[�       �	fh)XYc�A�*

loss���<��y       �	� *XYc�A�*

loss���<��#+       �	R�*XYc�A�*

loss��
<���       �	f+XYc�A�*

loss��=4       �	 ,XYc�A�*

lossF<!��       �	��,XYc�A�*

loss;�<��G       �	y<-XYc�A�*

lossŻ�;�?�       �	4�-XYc�A�*

loss&��;9`?�       �	�k.XYc�A�*

lossoB=bX~�       �	�GXYc�A�*

loss��=ق��       �	w-HXYc�A�*

loss��=�k^       �	��HXYc�A�*

loss��=���V       �	ǀIXYc�A�*

loss���<f��       �	aJXYc�A�*

loss���<��        �	)�JXYc�A�*

loss�ܔ<��[�       �	�IKXYc�A�*

loss�h�<���       �	:�KXYc�A�*

loss��=;�(       �	I�LXYc�A�*

lossM`�<1�B�       �	�MXYc�A�*

loss�W�;+�y�       �	"�MXYc�A�*

loss�R�<wAk�       �	_BNXYc�A�*

loss�r�<Z3{       �	Q�NXYc�A�*

loss$�D<7a��       �	�tOXYc�A�*

loss�K�<�!�       �	�%PXYc�A�*

lossN��<I&�>       �	��PXYc�A�*

loss1��:W�/       �	X�QXYc�A�*

lossA��<�-z�       �	M�RXYc�A�*

loss�{<V�       �	�RSXYc�A�*

loss7�>=���       �	��SXYc�A�*

lossR�~<9m�       �	�TXYc�A�*

loss���=��ݪ       �	6:UXYc�A�*

loss�,x<���       �	��UXYc�A�*

lossm��=�U       �	#�VXYc�A�*

loss��<��ϩ       �	��WXYc�A�*

loss �<vLm       �	]1XXYc�A�*

loss�H�<'�H       �	�XXYc�A�*

lossX��;d��       �	�]YXYc�A�*

losse<wBO$       �	��YXYc�A�*

lossv-=��B       �	;�ZXYc�A�*

loss	h�<�>R       �	��[XYc�A�*

loss<����       �	��\XYc�A�*

loss���=z�4       �	zl]XYc�A�*

lossX�P="3GL       �	�
^XYc�A�*

lossTV=c�Z�       �	C�^XYc�A�*

loss��=�d�V       �	K_XYc�A�*

loss;[)<q��K       �	��_XYc�A�*

lossب*=HN��       �	�`XYc�A�*

loss��=6>1       �	�]aXYc�A�*

lossЉ=|���       �	��aXYc�A�*

loss�l�;c�Q       �	�bXYc�A�*

loss�Z�<��u       �	#,cXYc�A�*

lossjv�;걅�       �	��cXYc�A�*

loss
l�< 9       �	Z�dXYc�A�*

lossc?=�2lt       �	reXYc�A�*

loss�*�=nUk       �	�eXYc�A�*

lossq��<�a	       �	wMfXYc�A�*

loss]"$=&�V       �	e�fXYc�A�*

loss�=;��rH       �	��gXYc�A�*

lossង;7D��       �	ehXYc�A�*

loss̭�<SG<Q       �	�iXYc�A�*

losse
K=f=�<       �	��iXYc�A�*

lossd��;����       �	�ujXYc�A�*

loss`�=�
O       �	GkXYc�A�*

lossB�<�8�H       �	��kXYc�A�*

loss#�>;�YG�       �	�NlXYc�A�*

loss|`�;<�       �	� mXYc�A�*

loss8�;P�0o       �	�mXYc�A�*

lossWË=!�v       �	�;nXYc�A�*

losstbB=��]�       �	!�nXYc�A�*

loss�ׂ=��e�       �	�zoXYc�A�*

loss�Q�:3,r�       �	zTpXYc�A�*

loss��4<�(N1       �	��pXYc�A�*

loss{�d=��       �	(qXYc�A�*

lossT�K<���>       �	�rXYc�A�*

loss��;�4��       �	s�rXYc�A�*

losss=-B��       �	ePsXYc�A�*

loss$E=Ű�       �	6�sXYc�A�*

loss�_�<�O�       �	��tXYc�A�*

loss�v(<�	�       �	�uXYc�A�*

lossX��<��I       �	p�uXYc�A�*

losskg=�BJ       �	�HvXYc�A�*

loss�<��       �	��vXYc�A�*

loss$&4=���       �	%wwXYc�A�*

loss�=�\�       �	jxXYc�A�*

loss\ �<�1       �	6�xXYc�A�*

loss�i;���       �	�@yXYc�A�*

loss��<a�       �	��yXYc�A�*

loss�!�;^g��       �	`rzXYc�A�*

loss%��=�"�       �	�
{XYc�A�*

loss���=�ٟ�       �	ճ{XYc�A�*

loss�G.<���       �	J|XYc�A�*

loss�ȍ=���M       �	-�|XYc�A�*

loss&<���       �	7�}XYc�A�*

lossqf�<4�N=       �	$'~XYc�A�*

loss�z:<lN�d       �	�~XYc�A�*

loss��<!�
9       �	�\XYc�A�*

loss�5'=اv(       �	&:�XYc�A�*

lossߌD<^��D       �	UـXYc�A�*

loss�0^<���       �	�{�XYc�A�*

loss�6=���1       �	��XYc�A�*

loss@��<%�%�       �	]XYc�A�*

loss�=��]       �	ke�XYc�A�*

loss�<�?�W       �	n�XYc�A�*

loss/<1=�?��       �	ڭ�XYc�A�*

loss��;F#�       �	�O�XYc�A�*

losscŃ<�V܀       �	�XYc�A�*

loss}��=�)       �	��XYc�A�*

lossKݜ= 8�       �	�8�XYc�A�*

lossZQk<h�/T       �	=ԇXYc�A�*

loss��-;|�T       �	�x�XYc�A�*

loss,�$;/��t       �	7�XYc�A�*

loss�b%=�J%�       �	Ӽ�XYc�A�*

loss��<&ù       �	�_�XYc�A�*

loss�w=I���       �	*�XYc�A�*

loss�.�=4���       �	ᖋXYc�A�*

loss:�*=a���       �	P7�XYc�A�*

loss/��;^�H       �	�ҌXYc�A�*

lossH�=6���       �	�o�XYc�A�*

loss;]<�lz       �	�XYc�A�*

lossNZ="4�
       �	q��XYc�A�*

loss�]�;���       �	F�XYc�A�*

loss�,�=� �I       �	���XYc�A�*

loss��:G���       �	�z�XYc�A�*

loss���;W       �	2�XYc�A�*

loss�p�;6u       �	᷑XYc�A�*

loss��_<��y       �	&T�XYc�A�*

lossC=�y�       �	8��XYc�A�*

loss��]<���r       �	X��XYc�A�*

loss��
<L��>       �	�<�XYc�A�*

loss� >=E       �	�۔XYc�A�*

loss�Am<,��       �	���XYc�A�*

loss)�<���`       �	F&�XYc�A�*

loss��<���{       �	�ÖXYc�A�*

lossc�B=5��       �	�\�XYc�A�*

loss�o?=]n��       �	4�XYc�A�*

loss��<o
       �	���XYc�A�*

loss�6�<T~c>       �	�J�XYc�A�*

loss�=D=��ד       �	8�XYc�A�*

loss�Rp=ߊ'�       �	��XYc�A�*

loss���;Wo�/       �	��XYc�A�*

loss�$�;��θ       �	b��XYc�A�*

losso =�Oɝ       �	�7�XYc�A�*

loss`V�<�,/       �	� �XYc�A�*

loss6�S=�E)       �	l̞XYc�A�*

loss݉�;}�'�       �	��XYc�A�*

loss��<��~       �	D6�XYc�A�*

lossb�;���       �	v�XYc�A�*

loss��;:��       �	v�XYc�A�*

loss$8!<'=��       �	���XYc�A�*

lossV��;>�[       �	�o�XYc�A�*

loss�=�6M�       �	y��XYc�A�*

lossL=�t5       �	_C�XYc�A�*

loss[�>=�@!�       �	w�XYc�A�*

loss7!=e��       �	3�XYc�A�*

loss6Z�<J`x       �	�ӧXYc�A�*

loss��
<��ߎ       �	Tp�XYc�A�*

loss6�i:B~$       �	Q��XYc�A�*

loss.]%<�yB�       �	+4�XYc�A�*

loss�d�<��`�       �	�
�XYc�A�*

loss*��;����       �	}�XYc�A�*

loss�"^=O�;�       �	aެXYc�A�*

loss�B<�]e�       �	F��XYc�A�*

loss�{�<�$�S       �	nR�XYc�A�*

lossm6�<$���       �	���XYc�A�*

loss�9�;��I       �	f�XYc�A�*

loss�\<�c�       �	,�XYc�A�*

loss$�<����       �	���XYc�A�*

lossh9=K��       �	n��XYc�A�*

loss"<ޯny       �	�H�XYc�A�*

lossv�2=�SǪ       �	-�XYc�A�*

lossny�<#��,       �	���XYc�A�*

lossV�D=�Y       �	�XYc�A�*

loss>;zÜ       �	�9�XYc�A�*

loss�H~=� �U       �	��XYc�A�*

loss�g�<�*��       �	j޷XYc�A�*

loss��{;���       �	|�XYc�A�*

loss��;<a9u       �	���XYc�A�*

loss��==�i+�       �	&Q�XYc�A�*

lossa�e;�#;%       �	��XYc�A�*

lossN�<��ii       �	~��XYc�A�*

lossJ|;�H)!       �	t'�XYc�A�*

losss�;�|�       �	¼XYc�A�*

loss�^(=�Ѵ       �	)Y�XYc�A�*

loss�D�;편T       �	��XYc�A�*

loss�1$=*��       �	n��XYc�A�*

lossh��=��ic       �	��XYc�A�*

loss-�=����       �	�ؿXYc�A�*

loss���:���       �	Wx�XYc�A�*

loss��&<y�~T       �	��XYc�A�*

lossI�f<d��       �	���XYc�A�*

loss��=|�8�       �	5c�XYc�A�*

loss5�=���.       �	���XYc�A�*

loss-��;����       �	���XYc�A�*

loss�L�=m��       �	U4�XYc�A�*

loss��t<����       �	���XYc�A�*

lossa
�=�]�       �	�]�XYc�A�*

loss
�<6��       �	��XYc�A�*

loss���<�W�       �	���XYc�A�*

loss�� >_�!       �	��XYc�A�*

loss~�=���       �	y��XYc�A�*

loss��<6�3�       �	�A�XYc�A�*

loss�;�;�)T       �	s��XYc�A�*

lossq�b<�p0h       �	�i�XYc�A�*

loss�
=��MK       �	r��XYc�A�*

loss�ҡ;57y�       �	���XYc�A�*

loss�S�<w�/\       �	)�XYc�A�*

loss1e�;K�t�       �	w��XYc�A�*

loss���=clx-       �	�P�XYc�A�*

lossr�=ʀ)\       �	���XYc�A�*

loss �z=JqQ       �	D��XYc�A�*

loss���<����       �	U�XYc�A�*

loss�M�<<�^�       �	���XYc�A�*

loss��<�L"�       �	8K�XYc�A�*

loss�]�;��<_       �	���XYc�A�*

loss�&<�q�       �	�q�XYc�A�*

loss�n^=x�'�       �	[�XYc�A�*

loss,>rh�       �	��XYc�A�*

loss���<��u2       �	
/�XYc�A�*

loss1�{<"]�       �	���XYc�A�*

loss`��<��u       �	�X�XYc�A�*

lossE�;Jw�9       �	B��XYc�A�*

losss�{<���b       �	���XYc�A�*

lossf]b=ľ=�       �	��XYc�A�*

loss��&=fP�W       �	Թ�XYc�A�*

loss_1{=��f�       �	x_�XYc�A�*

loss���=����       �	b��XYc�A�*

loss&s:>���       �	��XYc�A�*

loss���=OQ�       �	��XYc�A�*

loss���;P��       �	��XYc�A�*

loss%=ώ:�       �	���XYc�A�*

loss�x�<r�t       �	C��XYc�A�*

loss�=�P�       �	�#�XYc�A�*

loss	gU=;	��       �	���XYc�A�*

loss�	z;���k       �	lw�XYc�A�*

lossI�S<i
F�       �	*�XYc�A�*

loss�Y=���c       �	��XYc�A�*

loss�4�=4U)H       �	If�XYc�A�*

lossmI<����       �	�j�XYc�A�*

lossta$=�Z�G       �	w/�XYc�A�*

lossZ'<��C�       �	���XYc�A�*

loss���;���       �	�$�XYc�A�*

lossC�=���       �	t��XYc�A�*

loss}	<��}W       �	�q�XYc�A�*

lossi(`<{��       �	��XYc�A�*

loss���=
�d-       �	ݘ�XYc�A�*

loss�)/=����       �	�>�XYc�A�*

loss9�<I��?       �	��XYc�A�*

loss�~=�`�,       �	�p�XYc�A�*

loss��D<a*g�       �	@�XYc�A�*

loss�7i;8� {       �	a��XYc�A�*

loss��7<b�x�       �	�p�XYc�A�*

loss���<o�       �	r�XYc�A�*

lossi��<�:(       �	ݱ�XYc�A�*

loss;�<�b��       �	�[�XYc�A�*

loss���<"4�       �	���XYc�A�*

lossCmz<���       �	���XYc�A�*

lossM��;H@'       �	�J�XYc�A�*

loss��b=5u�       �	5��XYc�A�*

loss��o<�e�1       �	H��XYc�A�*

loss�`<���\       �	�(�XYc�A�*

lossd��<��%p       �	���XYc�A�*

lossR�<Z�O�       �	�p�XYc�A�*

loss��;�t�j       �	�XYc�A�*

loss#��;��K       �	���XYc�A�*

lossc(<܂��       �	x��XYc�A�*

loss�H~=��       �	�.�XYc�A�*

loss��<6kc�       �	���XYc�A�*

lossi<�1       �	!\�XYc�A�*

loss�f�=힚�       �	<��XYc�A�*

loss���<x��       �	��XYc�A�*

loss5<�8T�       �	�!�XYc�A�*

loss-%"=���       �	,��XYc�A�*

loss(g=H��        �	���XYc�A�*

loss�/�<.��}       �	�_�XYc�A�*

loss�e�;E��       �	#��XYc�A�*

loss�q=���P       �	q��XYc�A�*

loss�T�=�O       �	�7�XYc�A�*

lossvq�=�A+       �	��XYc�A�*

loss�V=���       �	Y��XYc�A�*

loss=�E�3       �	v��XYc�A�*

loss�7�<�}��       �	Fz�XYc�A�*

losssz�<x�       �	��XYc�A�*

lossS�=�6��       �	���XYc�A�*

loss���=���F       �	���XYc�A�*

lossF��<��8�       �	�1�XYc�A�*

loss�;z<���D       �	1��XYc�A�*

loss\̏;�.��       �	�8YYc�A�*

loss�=����       �	�YYc�A�*

losss"=���       �	�{YYc�A�*

lossAv;h��l       �	YYc�A�*

loss�ŝ<�{I       �	��YYc�A�*

loss?x�<�к       �	�vYYc�A�*

loss��9= �[z       �	4YYc�A�*

loss�{7=�^�!       �	B�YYc�A�*

loss?��<����       �	4JYYc�A�*

lossOf=f�        �	`�YYc�A�*

loss�w%<��       �	ǺYYc�A�*

loss"�<>[/C       �	�MYYc�A�*

loss��#<���.       �	I�YYc�A�*

loss�+�<�,�       �	O�	YYc�A�*

loss�@�<zz�6       �	+1
YYc�A�*

loss�L:4��       �	a�
YYc�A�*

loss&��;�sv�       �	�ZYYc�A�*

loss��<W`��       �	��YYc�A�*

loss��=z�?�       �	��YYc�A�*

loss�?�<L���       �	�+YYc�A�*

loss;.8=�c       �	�YYc�A�*

loss��0=o\��       �	�ZYYc�A�*

loss�=I�V�       �	��YYc�A�*

loss��<c��       �	��YYc�A�*

loss�_=l�       �	/YYc�A�*

loss���;�(	�       �	.�YYc�A�*

loss�t�<qM�>       �	jjYYc�A�*

loss�`H<i���       �	�YYc�A�*

loss�>=�W�       �	��YYc�A�*

lossxF<��oY       �	CYYc�A�*

loss��T=��       �	�YYc�A�*

lossW�<g���       �	�{YYc�A�*

loss���<s90       �	�YYc�A�*

loss�<4s�k       �	�YYc�A�*

lossiR�;�p       �	e�YYc�A�*

loss?�/=oC��       �	fJYYc�A�*

loss�LG<�d{       �	��YYc�A�*

loss��<�ZT�       �	D�YYc�A�*

loss��d=�g5       �	8IYYc�A�*

loss���;�-�       �	)�YYc�A�*

loss�8�<)�R	       �	��YYc�A�*

loss�mA=��_       �	�)YYc�A�*

loss�#=�x�^       �	[�YYc�A�*

loss��I;{�       �	o�YYc�A�*

loss��Q=ae��       �	F\YYc�A�*

loss�%M=����       �	�YYc�A�*

loss��;�g-       �	�YYc�A�*

loss�},<cyB�       �	��YYc�A�*

loss��<��l       �	�0 YYc�A�*

loss?�}=
q�       �	�[!YYc�A�*

lossQ`�<W�n       �	�"YYc�A�*

loss�1K<�-o       �	
�"YYc�A�*

loss�$=�bX�       �	�<#YYc�A�*

loss�B�<O�W       �	��#YYc�A�*

losss�;��J�       �	�x$YYc�A�*

loss�u;5���       �	� %YYc�A�*

loss��	=b	�'       �	��%YYc�A�*

lossv��=R���       �	�S&YYc�A�*

loss�c�=�{�       �	r�&YYc�A�*

lossX�=���       �	b�'YYc�A�*

loss���<�U=�       �	�=(YYc�A�*

loss$%K=��       �	��(YYc�A�*

loss�;B��S       �	ρ)YYc�A�*

loss-�;�/)       �	�W*YYc�A�*

loss���<@�R�       �	L+YYc�A�*

lossv��<��qB       �	�+YYc�A�*

loss�m~=^���       �	$�,YYc�A�*

loss��->3�<%       �	C-YYc�A�*

loss��"=�[�       �	��-YYc�A�*

loss�N�<2��u       �	��.YYc�A�*

loss!�%=fE��       �	�/YYc�A�*

loss@��<	�M�       �	�/YYc�A�*

loss���=R �       �	�0YYc�A�*

loss#�<�+��       �	\81YYc�A�*

loss鱪<u���       �	I�1YYc�A�*

loss��_=�U       �	�y2YYc�A�*

loss@�=n9       �	3YYc�A�*

loss��=P��       �	��3YYc�A�*

loss�-�<���Y       �	U4YYc�A�*

lossHo*<�'�       �	�4YYc�A�*

loss0@=��!       �	6�5YYc�A�*

lossN/�=+��       �	�D6YYc�A�*

lossZ(?=��2G       �	e�6YYc�A�*

loss�9R=�M�       �	U�7YYc�A�*

loss���<]n�T       �	@48YYc�A�*

loss�I
=�vB�       �	x�8YYc�A�*

loss�=�S       �	�m9YYc�A�*

loss�U>�@fq       �	::YYc�A�*

loss&��=���       �	�:YYc�A�*

loss�<��d       �	�H;YYc�A�*

lossP�;I/�       �	�;YYc�A�*

loss%[�=��       �	ܛ<YYc�A�*

loss5�=7�0�       �	~5=YYc�A�*

lossF�=g���       �	X7>YYc�A�*

lossXo<|p@       �	��>YYc�A�*

loss,��<��       �	.u?YYc�A�*

loss��;���;       �	|&@YYc�A�*

loss�3�<�n��       �	��@YYc�A�*

lossC��=#�AN       �	kAYYc�A�*

loss�>^=���&       �	.BYYc�A�*

loss�k<��       �	
�BYYc�A�*

loss?E�=<K       �	F}CYYc�A�*

lossO�=8�_�       �	�(DYYc�A�*

lossr&<Y��F       �	��DYYc�A�*

loss��b=)ô�       �	huEYYc�A�*

loss� >ɯ��       �	�(FYYc�A�*

loss�0?=��       �	��FYYc�A�*

loss���<GϬ�       �	ĘGYYc�A�*

loss�W�=(�O�       �	1BHYYc�A�*

loss�f=]\       �	��HYYc�A�*

loss�\=Tn��       �	T�IYYc�A�*

loss��o<S�;�       �	L7JYYc�A�*

lossS�<���       �	��JYYc�A�*

loss`D�<ѡ�       �	�KYYc�A�*

loss{�=���       �	� LYYc�A�*

losse=k��       �	u�LYYc�A�*

loss�9<�H]3       �	lMYYc�A�*

loss�|�<�5{�       �	B	NYYc�A�*

loss\�=R��=       �	7�NYYc�A�*

loss`vG<���       �	<POYYc�A�*

lossfTo=�`�1       �	Z�OYYc�A�*

lossė�<��6       �	��PYYc�A�*

lossC!f<&�4�       �	�3QYYc�A�*

losst�<���       �	#�QYYc�A�*

loss�ֶ<��       �	l{RYYc�A�*

loss��-=F�0�       �	zSYYc�A�*

losse��<�s�       �	��SYYc�A�*

loss܊�=�B       �	$_TYYc�A�*

lossf�X=��k�       �	yUYYc�A�*

loss!�=�!�J       �	�UYYc�A�*

loss�H�<vB8�       �	[CVYYc�A�*

loss�p�<���k       �	L�VYYc�A�*

lossĨ�=3TGs       �	ÀWYYc�A�*

loss���=փ$�       �	�XYYc�A�*

loss��>۵@�       �	ԸXYYc�A�*

loss�H�=0��        �	;SYYYc�A�*

lossY�<���h       �	�YYYc�A�*

loss��=�ڽ�       �	L�ZYYc�A�*

loss�ݜ=���3       �	�%[YYc�A�*

lossO��<p��       �	?�[YYc�A�*

loss���;AI�i       �	�^\YYc�A�*

loss	3�<so��       �	��\YYc�A�*

lossG�=�l`�       �	i�]YYc�A�*

loss���;���       �	�S^YYc�A�*

loss�:�=?�)�       �	�_YYc�A�*

lossQc�;��/       �	�_YYc�A�*

lossА>ѣ��       �	M`YYc�A�*

loss=�$=�":"       �	��`YYc�A�*

lossM0-=P&f       �	�aYYc�A�*

lossI\,=Ȧ��       �	�"bYYc�A�*

lossRF�<\���       �	��bYYc�A�*

loss���<�~t�       �	�\cYYc�A�*

loss��
<����       �	/�cYYc�A�*

loss��=څ��       �	��dYYc�A�*

loss\�<|�:
       �	75eYYc�A�*

loss(y<���@       �	J�eYYc�A�*

loss�x�<r X       �	~fYYc�A�*

loss&I�=�d�       �	HgYYc�A�*

loss2��<�AI2       �	��gYYc�A�*

loss&^<���       �	�whYYc�A�*

lossc]<��L�       �	�iYYc�A�*

loss  <q�?       �	��iYYc�A�*

loss���<Vr>J       �	�TjYYc�A�*

loss��Y=}yTv       �	R�jYYc�A�*

loss�e=����       �	h�kYYc�A�*

loss�|�=}۾�       �	MlYYc�A�*

loss3��<gq�e       �	��lYYc�A�*

loss,;F<_��       �	��mYYc�A�*

loss�w<`q�       �	�8nYYc�A�*

loss���=�0k�       �	��nYYc�A�*

loss���<N�       �	#�oYYc�A�*

loss��=��^       �	�EpYYc�A�*

loss�3�<�N�6       �	�pYYc�A�*

loss��=���       �	�qYYc�A�*

loss�=<�>�       �	HPrYYc�A�*

lossO��<�       �	��rYYc�A�*

loss�H�=��^�       �	��sYYc�A�*

lossߴ�=�
~       �	�WtYYc�A�*

lossυe<bi7       �	�uYYc�A�*

lossAĥ<���+       �	0�uYYc�A�*

loss1�<�ra�       �	�{vYYc�A�*

loss`h�;�O"L       �	>#wYYc�A�*

loss%՟<��C�       �	y�wYYc�A�*

lossX�u<d�/<       �	�kxYYc�A�*

loss���;�Q�)       �	�yYYc�A�*

lossܧ�;W���       �	��yYYc�A�*

lossVI>h�1J       �	�PzYYc�A�*

loss�/<dB0�       �	��zYYc�A�*

loss}�/<��q�       �	��{YYc�A�*

lossr��;���       �	*U|YYc�A�*

loss��<D˿       �	^�|YYc�A�*

loss܄";�c�       �	��}YYc�A�*

loss}�;����       �	hZ~YYc�A�*

loss��=�I�f       �	��~YYc�A�*

loss#2<掹       �	�YYc�A�*

loss�+M=_���       �	�:�YYc�A�*

loss��=h�I�       �	�ԀYYc�A�*

loss�X=����       �	�s�YYc�A�*

loss���<�z�       �	��YYc�A�*

lossȷ�:>5�       �	迂YYc�A�*

loss���=�%�       �	�`�YYc�A�*

loss}�;bU9�       �	6�YYc�A�*

loss,m�=�F�       �	U��YYc�A�*

loss<c>��P       �	�@�YYc�A�*

loss:�<X�f�       �	�܅YYc�A�*

loss%%�;���u       �	z�YYc�A�*

lossl�I=r�t       �	��YYc�A�*

lossZ��<u�%_       �	�ćYYc�A�*

loss��a<�7~�       �	$a�YYc�A�*

lossH0*<��A]       �	 �YYc�A�*

loss��:=cz7       �	���YYc�A�*

loss��m<.f�       �	t@�YYc�A�*

loss)"=`�i       �	]܊YYc�A�*

loss!{�<:&At       �	,�YYc�A�*

loss��4=��=8       �	�!�YYc�A�*

loss)H3;	"       �	�ČYYc�A�*

loss�ܐ=�C�^       �	|b�YYc�A�*

loss��<���       �	��YYc�A�*

loss�Ŀ<3|�k       �	���YYc�A� *

loss:�<�if�       �	:?�YYc�A� *

loss��
<���       �	LݏYYc�A� *

loss�%�<�l�       �	�~�YYc�A� *

loss6f=�Zww       �	m�YYc�A� *

loss;C]=��k       �	���YYc�A� *

loss<a��       �	�Y�YYc�A� *

lossD��;/�
�       �	J�YYc�A� *

loss��;� 1N       �	��YYc�A� *

loss���=���7       �	�A�YYc�A� *

loss���;�OM       �	�YYc�A� *

lossʽ�<8�       �	���YYc�A� *

loss(g�;h��       �	F"�YYc�A� *

lossz#-<r'$       �	~ĖYYc�A� *

loss-^�<�O�       �	�g�YYc�A� *

loss;�X;i�ua       �	��YYc�A� *

loss���<�T�       �	��YYc�A� *

loss�7�;��:       �	�N�YYc�A� *

loss�U\=/��       �	���YYc�A� *

lossiRh=�U       �	���YYc�A� *

loss;^�=I/�       �	�<�YYc�A� *

loss��==ok�       �	+ߛYYc�A� *

lossz3F<kG��       �	c}�YYc�A� *

lossH[<�I�       �	�[�YYc�A� *

loss�_�<����       �	�d�YYc�A� *

loss�;�       �	Ѐ�YYc�A� *

loss($�<��8K       �	|*�YYc�A� *

lossDK:Ғ\�       �	�ˠYYc�A� *

lossd�<Ov�       �	�e�YYc�A� *

loss�L�<�_Ă       �	��YYc�A� *

loss2)<P��a       �	��YYc�A� *

lossܯ[<���
       �	�K�YYc�A� *

loss���=Pu�       �	��YYc�A� *

loss��7=/Lv       �	���YYc�A� *

loss�L�;0
       �	�A�YYc�A� *

loss;P�<�W�       �	>�YYc�A� *

losso=��]�       �	���YYc�A� *

loss,�:QH`�       �	��YYc�A� *

loss.��;���o       �	!˧YYc�A� *

loss��d;�yF       �	�m�YYc�A� *

loss�1�:P�"       �	 ��YYc�A� *

loss��w;�v�^       �	t$�YYc�A� *

loss�#E=�XY       �	�תYYc�A� *

loss
��=��,       �	���YYc�A� *

loss*�/;n돏       �	��YYc�A� *

lossh�V;�,I       �	���YYc�A� *

loss:+��       �	�Y�YYc�A� *

loss���9ZD       �	�+�YYc�A� *

loss�&=fb0�       �	��YYc�A� *

loss�|�=�A]       �	�YYc�A� *

loss��<��       �	�'�YYc�A� *

loss��x;���       �	�;�YYc�A� *

loss�f.<�*�       �	�ױYYc�A� *

loss!��=~�eg       �	 r�YYc�A� *

loss���:�L]�       �	N�YYc�A� *

loss
o>V��g       �	޳YYc�A� *

lossO��;�O�       �	.s�YYc�A� *

loss&�=�<�(       �	R�YYc�A� *

loss��=(dy�       �	��YYc�A� *

loss�W<�H�       �	�H�YYc�A� *

loss$?O<���       �	��YYc�A� *

lossq,T=��F�       �	*��YYc�A� *

loss�M<�r��       �	<3�YYc�A� *

loss���<+��       �	�θYYc�A� *

loss��S</c�       �	�g�YYc�A� *

lossݴ�;�lD�       �	��YYc�A� *

loss~r�=M��'       �	��YYc�A� *

loss�}<����       �	꒻YYc�A� *

loss �=���       �	�;�YYc�A� *

loss��Z<�4P       �	�v�YYc�A� *

loss\ބ=�>       �	,�YYc�A� *

lossIm+=���       �	���YYc�A� *

loss�el=W׵       �	�M�YYc�A� *

loss7_�<?�2�       �	��YYc�A� *

loss�i<9>�       �	���YYc�A� *

loss`I�<�JbA       �	�*�YYc�A� *

lossڳ�<��(       �	���YYc�A� *

loss�D;%�C       �	�^�YYc�A� *

loss<�m;�o'�       �	��YYc�A� *

loss�T�;�l�E       �	Q��YYc�A� *

loss�9
=���       �	�g�YYc�A� *

loss�M<�+x       �	?��YYc�A� *

loss#�m=i��V       �	U��YYc�A� *

loss��=�%�       �	�8�YYc�A� *

lossLu<�f�F       �	,��YYc�A� *

losslJ�=�{\�       �	l�YYc�A� *

loss���<o��d       �	��YYc�A� *

lossxt:;��]�       �	���YYc�A� *

loss�< G��       �	2�YYc�A� *

loss��x<鬉�       �	���YYc�A� *

lossP_=��9�       �	���YYc�A� *

loss7_=y���       �	��YYc�A� *

lossf�
=��7�       �	A��YYc�A� *

lossh�T=Z�c�       �	]N�YYc�A� *

loss�^=�@��       �	O��YYc�A� *

lossPh�<8c�       �	��YYc�A� *

loss�=��P       �	\<�YYc�A� *

loss{��<� 0t       �	<��YYc�A� *

loss�{a;�Ư�       �	�y�YYc�A� *

lossaeZ<��       �	�YYc�A� *

losshk�=�@\       �	���YYc�A� *

loss��<����       �	a�YYc�A� *

loss�R=�Dd       �	]��YYc�A� *

loss��;�C=%       �	���YYc�A� *

loss*3<�t�H       �	/�YYc�A� *

loss݈=<x��       �	��YYc�A� *

lossq C=o�r�       �	��YYc�A� *

lossҙ�=�(�.       �	u"�YYc�A� *

loss�T�=tfj�       �	U��YYc�A� *

loss&�<��
5       �	�X�YYc�A� *

lossѨ6=dh�       �	L6�YYc�A� *

loss��t=n�*�       �	���YYc�A� *

loss;s<���       �	�}�YYc�A� *

loss���<��S       �	��YYc�A� *

loss!pg=͒^       �	��YYc�A� *

loss��(<&�24       �	aO�YYc�A� *

loss�a<ۀ<�       �	���YYc�A� *

losse��<��v�       �	ۅ�YYc�A� *

loss�=���       �	 �YYc�A� *

lossM��;M��       �	���YYc�A� *

loss~��=�q��       �	}��YYc�A� *

lossW��;!�;       �	N*�YYc�A� *

loss���<�G@�       �	���YYc�A� *

lossV%<�,+       �	�]�YYc�A� *

loss�j�=
�G       �	���YYc�A� *

losst�p<��>t       �	���YYc�A� *

loss#�=Aq$       �	�9�YYc�A� *

loss�-�<BL�        �	���YYc�A� *

loss�͜=�Y�       �	2t�YYc�A�!*

loss=��<<���       �	�  ZYc�A�!*

loss[<��v       �	C� ZYc�A�!*

loss�Y=?��       �	{hZYc�A�!*

lossCS�<:��6       �	�ZYc�A�!*

loss���<I0�[       �	E�ZYc�A�!*

lossOݮ<�QLx       �	TZYc�A�!*

lossn=u�e�       �	��ZYc�A�!*

loss�<<�B��       �	��ZYc�A�!*

loss�H�<���J       �	l>ZYc�A�!*

loss�+o=̹��       �	<�ZYc�A�!*

loss��;�}       �	9�ZYc�A�!*

loss��=��       �	�GZYc�A�!*

lossG=u�Q�       �	W�ZYc�A�!*

loss��<����       �	��ZYc�A�!*

loss��=�z       �	.7	ZYc�A�!*

lossx3�<؍�       �	��	ZYc�A�!*

loss�s�<���1       �	Ӈ
ZYc�A�!*

loss�T�<�y؈       �	�"ZYc�A�!*

lossގ;H��       �	x�ZYc�A�!*

loss��<�A�       �	�LZYc�A�!*

loss��=AR�       �	o�ZYc�A�!*

loss@_H=bn�       �	�ZYc�A�!*

loss�7�=U��       �	1ZYc�A�!*

loss��;�!�       �	L�ZYc�A�!*

loss�w<p�|�       �	)]ZYc�A�!*

loss϶K;t6ih       �	��ZYc�A�!*

losss<�+�       �	ƈZYc�A�!*

loss��|<d�y       �	�<ZYc�A�!*

losssj�<�-��       �	�ZYc�A�!*

loss�m�=n%�       �	7�ZYc�A�!*

loss�=�u5�       �	�WZYc�A�!*

loss��D;SBXE       �	�ZYc�A�!*

loss15�;yh9`       �	��ZYc�A�!*

lossH�;k�6       �	A�ZYc�A�!*

loss�8i=|�@�       �	h#ZYc�A�!*

loss@��<�0��       �		�ZYc�A�!*

loss��=Q���       �	2UZYc�A�!*

loss���<G��B       �	l�ZYc�A�!*

lossvE=Z���       �	��ZYc�A�!*

lossqf�<���       �	�ZYc�A�!*

loss�:<��e`       �	�ZYc�A�!*

loss��<`� �       �	�ZYc�A�!*

loss,�<O�	�       �	�0ZYc�A�!*

loss��*<T��@       �	 �ZYc�A�!*

loss��F=����       �	�`ZYc�A�!*

lossa�U=U�L       �	{�ZYc�A�!*

lossV:`=�'��       �	��ZYc�A�!*

loss��<��Im       �	+2ZYc�A�!*

loss��=�&��       �	��ZYc�A�!*

loss��.<+���       �	N�ZYc�A�!*

loss
@D=[M�       �	|, ZYc�A�!*

loss�o�<��x       �	7� ZYc�A�!*

loss�J=j��f       �	�k!ZYc�A�!*

loss1{<����       �	�"ZYc�A�!*

loss�S=��7G       �	ٖ"ZYc�A�!*

loss�(�<%�=�       �	�X#ZYc�A�!*

loss d�<洲P       �	��#ZYc�A�!*

loss)8�<i�/R       �	;�$ZYc�A�!*

loss�<�D#       �	��%ZYc�A�!*

loss\�<ʔ�t       �	<&ZYc�A�!*

loss2ʫ<n�       �	��&ZYc�A�!*

lossC�<n�H�       �	`'ZYc�A�!*

lossJ��<I��d       �	��'ZYc�A�!*

loss��<�j�       �	��(ZYc�A�!*

loss���<O���       �	c))ZYc�A�!*

loss��=���o       �	��)ZYc�A�!*

loss!h�<�        �	o�*ZYc�A�!*

lossf�g=� {�       �	�@+ZYc�A�!*

loss��/<�7��       �	G�+ZYc�A�!*

lossf�=�       �	5�,ZYc�A�!*

loss�z�<��       �	h>-ZYc�A�!*

lossr��;�=��       �	r�-ZYc�A�!*

lossH��=B�+{       �	Ō.ZYc�A�!*

loss�n=����       �	R,/ZYc�A�!*

loss�2f<R�m       �	2�/ZYc�A�!*

lossC�M=I\��       �	kc0ZYc�A�!*

lossS��;���       �	=
1ZYc�A�!*

loss�f�:�4�3       �	{�1ZYc�A�!*

lossI�<+��       �	^2ZYc�A�!*

lossA��<�{��       �	Y�2ZYc�A�!*

loss�1�<���W       �	X�3ZYc�A�!*

loss��<�B�       �	�U4ZYc�A�!*

loss�_6<-��       �	��4ZYc�A�!*

lossSX<�j&N       �	ڎ5ZYc�A�!*

lossd��<�Tl|       �	�)6ZYc�A�!*

loss#�<$�]       �	��6ZYc�A�!*

loss�"�<_[8q       �	Nc7ZYc�A�!*

lossɷt<,��H       �	�8ZYc�A�!*

loss�D�=ꚷK       �	;�8ZYc�A�!*

lossX��<�%�       �	�D9ZYc�A�!*

loss�H�<v�i       �	��9ZYc�A�!*

loss2Z�;_�N2       �	pw:ZYc�A�!*

loss ��<��y�       �	�;ZYc�A�!*

lossn��<~�^�       �	(G<ZYc�A�!*

loss)e�<�iZ       �	��<ZYc�A�!*

loss�b1<�E       �	 t=ZYc�A�!*

loss���;Ǝw�       �	�
>ZYc�A�!*

loss��<||��       �	�>ZYc�A�!*

loss�5_=�..6       �	=_?ZYc�A�!*

loss\�<��       �	�@ZYc�A�!*

lossAd�<�d�       �	X�@ZYc�A�!*

lossE.=ǁ
H       �	3lAZYc�A�!*

loss��G=eؤz       �	4BZYc�A�!*

loss�l0<s`H�       �	ٴBZYc�A�!*

loss��<P��       �	<lCZYc�A�!*

loss۬�<�R�       �	cDZYc�A�!*

loss���;6��-       �	�DZYc�A�!*

loss&��<�'j       �	/kEZYc�A�!*

loss�r�=�T�]       �	!FZYc�A�!*

loss��<���       �	ŪFZYc�A�!*

loss���=q�?O       �	�KGZYc�A�!*

loss�hE;��       �	�EHZYc�A�!*

lossz��;����       �	&�HZYc�A�!*

loss�l;�E�       �	�zIZYc�A�!*

lossv�<}�       �	�JZYc�A�!*

loss��;��}�       �	X�JZYc�A�!*

loss�;x^J;       �	/NKZYc�A�!*

loss;|�</2��       �	T�KZYc�A�!*

lossvK�<���e       �	܁LZYc�A�!*

loss�@�<��        �	�!MZYc�A�!*

loss�U.=�w|�       �	��MZYc�A�!*

loss���<���       �	ӈNZYc�A�!*

lossϐL<���       �	�OZYc�A�!*

loss["�;V��       �	I�OZYc�A�!*

loss��:<m���       �	%YPZYc�A�!*

loss�?/<�n��       �	b�PZYc�A�!*

loss=�(=�)9L       �	m�QZYc�A�!*

loss�<ӱ�       �	A*RZYc�A�"*

lossO�+=���P       �	i�RZYc�A�"*

loss���=��ҿ       �	wgSZYc�A�"*

loss��%=�QJN       �	z�SZYc�A�"*

loss��;,��       �	[�TZYc�A�"*

loss�h�:_��       �	E/UZYc�A�"*

lossLa�=�T^>       �	;�UZYc�A�"*

loss�{K=�3�       �	�uVZYc�A�"*

loss��<,U��       �	WZYc�A�"*

loss��}<�S!;       �	��WZYc�A�"*

loss��u<�%AW       �	�kXZYc�A�"*

loss!	g<�I       �	�YZYc�A�"*

loss��G:���       �	��YZYc�A�"*

loss��=C��       �	SAZZYc�A�"*

lossn'�;�F�5       �	��ZZYc�A�"*

loss�-<цA       �	u[ZYc�A�"*

loss��!=SԸ�       �	o\ZYc�A�"*

loss��U<��       �	�\ZYc�A�"*

lossF��<��P�       �	�]ZYc�A�"*

loss�A,>8>��       �	4^ZYc�A�"*

loss���;����       �	��^ZYc�A�"*

loss�@(<q�H`       �	��_ZYc�A�"*

loss\��<����       �	�2`ZYc�A�"*

loss��\=4�1       �	
MaZYc�A�"*

loss�);<=�       �	��aZYc�A�"*

loss��/=H���       �	��bZYc�A�"*

loss,9I=��`�       �	 |cZYc�A�"*

loss���=n�A       �	��dZYc�A�"*

loss��<
Z6W       �	KVeZYc�A�"*

loss}M:����       �	r�eZYc�A�"*

loss���<���M       �	��fZYc�A�"*

loss.]T;,Ȕ�       �	�gZYc�A�"*

loss�9�;0�X       �	��hZYc�A�"*

loss�i<����       �	��iZYc�A�"*

lossӒ;�jN�       �	 �jZYc�A�"*

loss-0X=+�BA       �	̷kZYc�A�"*

loss��5=;��[       �	4elZYc�A�"*

loss���<�,�       �	37mZYc�A�"*

lossnS�<RQBs       �	�KnZYc�A�"*

lossf\e=���       �	ӄoZYc�A�"*

lossxL+=��S-       �	�KpZYc�A�"*

loss�so;���       �	�pZYc�A�"*

lossS�<�_��       �	`�qZYc�A�"*

lossIf[<8�5`       �	��rZYc�A�"*

loss���;��a       �	sZYc�A�"*

loss�	<-��i       �	aptZYc�A�"*

loss/�+=��       �	X;uZYc�A�"*

lossD�^=Ow�       �	�vZYc�A�"*

loss*�;H=�       �	@�vZYc�A�"*

loss(4#=��       �	1}wZYc�A�"*

loss���;
�M       �	�xZYc�A�"*

loss�C�<R��       �	�;yZYc�A�"*

lossq��;�֖       �	�zZYc�A�"*

loss)n=ӑf�       �	f{ZYc�A�"*

loss��P=��Ԕ       �	R|ZYc�A�"*

lossa�=�f       �	��|ZYc�A�"*

loss�+3=���       �	�a}ZYc�A�"*

loss�jJ<O��       �	�}ZYc�A�"*

loss=�<����       �	PZYc�A�"*

lossc!L<�I1        �	JD�ZYc�A�"*

loss;��;�z�       �	�|�ZYc�A�"*

loss�R�<�;Æ       �	B�ZYc�A�"*

lossL�n;љ��       �	��ZYc�A�"*

loss�D�<�<^�       �	��ZYc�A�"*

lossɓW<�I       �	b��ZYc�A�"*

loss�!�=��:Z       �	�҅ZYc�A�"*

loss�T5=���       �	��ZYc�A�"*

loss �(=����       �	�ÇZYc�A�"*

loss�\{<~=�&       �	���ZYc�A�"*

loss$�F<�j�       �	�͉ZYc�A�"*

loss4�=F���       �	Bw�ZYc�A�"*

loss�9=�.       �	��ZYc�A�"*

loss�?K=��0�       �	|��ZYc�A�"*

loss�,;��%�       �	�Q�ZYc�A�"*

loss�u�<�]       �	�ZYc�A�"*

loss��<k�       �	�ZYc�A�"*

loss	�M<�,�       �	J&�ZYc�A�"*

loss	�;��u       �	^��ZYc�A�"*

loss��<��        �	Z�ZYc�A�"*

loss;��;76�U       �	�ZYc�A�"*

lossI6�<\�"       �	L��ZYc�A�"*

loss��Y;�½       �	5*�ZYc�A�"*

loss�~<��I2       �	�őZYc�A�"*

loss��a<�c.       �	)^�ZYc�A�"*

loss��#<�y�o       �	� �ZYc�A�"*

loss6I#<�G:�       �	ԙ�ZYc�A�"*

loss(�s=݃��       �	�-�ZYc�A�"*

loss%�D=�z�       �	�ŔZYc�A�"*

loss��&<�h�n       �	�Z�ZYc�A�"*

loss=�<�BY�       �	��ZYc�A�"*

loss):=c1        �	���ZYc�A�"*

loss�Q=R��       �	���ZYc�A�"*

loss��<����       �	e�ZYc�A�"*

loss���<��S�       �	j��ZYc�A�"*

loss'P=.��       �	�V�ZYc�A�"*

loss��;$�n       �	��ZYc�A�"*

lossWB<li�%       �	�ZYc�A�"*

lossd�]<,�K       �	<�ZYc�A�"*

lossD�<��t�       �	&��ZYc�A�"*

loss�/�<O�Ae       �	h<�ZYc�A�"*

loss}up=���       �	ҜZYc�A�"*

loss��=�/ �       �	�l�ZYc�A�"*

loss|%<y5+&       �	�ZYc�A�"*

lossA�
;~n!+       �	��ZYc�A�"*

loss��<�_א       �	���ZYc�A�"*

loss��x;�t       �	-B�ZYc�A�"*

loss���;)Ll       �	�n�ZYc�A�"*

loss}�)<F}Co       �	(e�ZYc�A�"*

lossm�=�q��       �	��ZYc�A�"*

loss3C�=�CO       �	½�ZYc�A�"*

loss�C]<=Y=T       �	I��ZYc�A�"*

loss(�X<�&��       �	<j�ZYc�A�"*

lossJ�<�;^       �	~�ZYc�A�"*

lossԀ=��6       �	ٵ�ZYc�A�"*

loss�{{;#q˖       �	�Q�ZYc�A�"*

loss��=�|�       �	��ZYc�A�"*

loss�>`<0sb�       �	��ZYc�A�"*

loss5r=R�y�       �	��ZYc�A�"*

loss�m=U�1�       �	���ZYc�A�"*

lossx�K<�+�       �	�Q�ZYc�A�"*

loss�N<ň�Y       �	%�ZYc�A�"*

loss	=p>       �	�~�ZYc�A�"*

loss 7�;|���       �	{�ZYc�A�"*

loss�l�=�g�       �	P��ZYc�A�"*

loss�:8=�l�       �	�F�ZYc�A�"*

loss�=�tg       �	߭ZYc�A�"*

loss��;3{_^       �	���ZYc�A�"*

loss(��=nxb�       �	�)�ZYc�A�"*

lossl�=��j�       �	{��ZYc�A�#*

loss],�;��T�       �	U�ZYc�A�#*

loss#��<�m'�       �	��ZYc�A�#*

loss �];~v       �	���ZYc�A�#*

loss���;��       �	;5�ZYc�A�#*

lossv�7=�t�       �	�ɲZYc�A�#*

loss4ؤ<�<�       �	�]�ZYc�A�#*

loss�x�<���       �	4��ZYc�A�#*

lossH�5;u'�       �	\9�ZYc�A�#*

loss}K�;�;�       �	�εZYc�A�#*

loss��;AK�       �	|b�ZYc�A�#*

loss/��:�W|�       �	��ZYc�A�#*

loss&��<�'?�       �	���ZYc�A�#*

loss8ͪ:ۢ       �	
0�ZYc�A�#*

loss�� :H;bS       �	�ƸZYc�A�#*

loss ˻;o�'       �	�Z�ZYc�A�#*

lossB�=�!��       �	���ZYc�A�#*

loss�C<A�eF       �	v��ZYc�A�#*

loss?A<v��       �	�6�ZYc�A�#*

loss�M�=�
�       �	%�ZYc�A�#*

loss�ĉ;�b�       �	��ZYc�A�#*

lossפ�<�Uu       �	Q�ZYc�A�#*

lossؒ'<���       �	�ZYc�A�#*

loss��L;���:       �	8I�ZYc�A�#*

loss��<����       �	�ݿZYc�A�#*

lossIf2=(��       �	���ZYc�A�#*

loss�G�=��>e       �	ƥ�ZYc�A�#*

loss{�;��V       �	�C�ZYc�A�#*

loss���<3��D       �	��ZYc�A�#*

loss�O�<~�       �	rj�ZYc�A�#*

loss�w�<{�=�       �	��ZYc�A�#*

loss�u;�C�       �	���ZYc�A�#*

loss�><a�#       �	�W�ZYc�A�#*

loss�=��]�       �	]��ZYc�A�#*

loss���;�[��       �	U1�ZYc�A�#*

loss}@Z<3��       �	D��ZYc�A�#*

loss�=��G       �	���ZYc�A�#*

lossxK�=���       �	N*�ZYc�A�#*

loss@��;�nM       �	���ZYc�A�#*

loss���<,@w�       �	sd�ZYc�A�#*

lossIs==��       �	��ZYc�A�#*

loss;��<Є=       �	���ZYc�A�#*

loss�
I<�И/       �	2;�ZYc�A�#*

loss�R=;�;       �	-��ZYc�A�#*

loss�Cd;�D]|       �	b�ZYc�A�#*

loss��=��'[       �	���ZYc�A�#*

loss��}=v�>       �	"��ZYc�A�#*

loss��=����       �	8��ZYc�A�#*

loss_�=�
��       �	���ZYc�A�#*

loss;�&=��̷       �	��ZYc�A�#*

loss[�=��       �	A(�ZYc�A�#*

lossMF�<	��~       �	~��ZYc�A�#*

loss&o�<��       �	�z�ZYc�A�#*

loss���<��=       �	��ZYc�A�#*

loss=�;���       �	��ZYc�A�#*

loss �3=!�       �	�A�ZYc�A�#*

loss���='h�d       �	���ZYc�A�#*

loss@�<���       �	f�ZYc�A�#*

loss �%<D1D�       �	7��ZYc�A�#*

loss�==ܧ\       �	���ZYc�A�#*

loss�j�;��Y�       �	�5�ZYc�A�#*

losstV=����       �	C<�ZYc�A�#*

loss(��<J�	       �	���ZYc�A�#*

loss},�;�,T�       �	�g�ZYc�A�#*

loss-�=d�(       �	�ZYc�A�#*

lossQ�9=�!       �	���ZYc�A�#*

loss�Z�<�wb�       �	y]�ZYc�A�#*

lossݡd;��K       �	��ZYc�A�#*

loss`�`=��       �	+��ZYc�A�#*

lossM�s=dW�e       �	�=�ZYc�A�#*

lossY=h�       �	���ZYc�A�#*

loss��;�b<�       �	�k�ZYc�A�#*

loss��;�       �	u�ZYc�A�#*

loss�=��(3       �	���ZYc�A�#*

loss�a�;��
       �	S<�ZYc�A�#*

loss���="��       �	��ZYc�A�#*

loss7��;���       �	4��ZYc�A�#*

loss�;<^�       �	�V�ZYc�A�#*

loss�]<��q#       �	0��ZYc�A�#*

lossO�<*Xf       �	i��ZYc�A�#*

loss�.=��_�       �	$(�ZYc�A�#*

loss/.=/U�       �	{��ZYc�A�#*

loss��=�S�       �	^�ZYc�A�#*

loss��,=�Fk:       �	���ZYc�A�#*

loss���=���p       �	=��ZYc�A�#*

loss��=��Y       �	*6�ZYc�A�#*

loss�Z�=Z+�       �	���ZYc�A�#*

loss���=���{       �	�g�ZYc�A�#*

loss�X<	�t�       �	���ZYc�A�#*

loss�n;�x�^       �	p��ZYc�A�#*

lossv{�<���f       �	P9�ZYc�A�#*

loss_;5=��з       �	>��ZYc�A�#*

loss�Ʉ<��ʮ       �	��ZYc�A�#*

loss���<-�e       �	�^�ZYc�A�#*

lossZñ<�rr       �	��ZYc�A�#*

loss�!�<F�Ӫ       �	/��ZYc�A�#*

lossW�u=$h��       �	/M�ZYc�A�#*

lossT�9=�yi�       �	���ZYc�A�#*

loss��8=�4�g       �	5~�ZYc�A�#*

lossn��<��(       �	6>�ZYc�A�#*

loss O�=j{�       �	���ZYc�A�#*

loss�:�;Wx�       �	Ql�ZYc�A�#*

loss-Ӟ<�+�%       �	�ZYc�A�#*

loss��:=G_<e       �	��ZYc�A�#*

loss�e�<�!       �	�<�ZYc�A�#*

loss��;���       �	��ZYc�A�#*

lossn��=����       �	�{�ZYc�A�#*

loss#�=�W��       �	C�ZYc�A�#*

loss���=S,��       �	z��ZYc�A�#*

loss�ߊ<�%�7       �	Nc�ZYc�A�#*

lossҤ'<	u��       �	?��ZYc�A�#*

lossn��<c���       �	@��ZYc�A�#*

loss�$j=A(:       �	Xr�ZYc�A�#*

loss��<h��~       �	��ZYc�A�#*

loss�#�<�L�V       �	�u�ZYc�A�#*

loss�0#=�ǚ       �	> [Yc�A�#*

loss�{>7'j       �	E� [Yc�A�#*

loss�$;
3�~       �	�K[Yc�A�#*

loss�.}<Ւ�       �	:�[Yc�A�#*

loss��%<PXL�       �	hw[Yc�A�#*

lossq��<.�H�       �	�
[Yc�A�#*

lossS�5<��f�       �	��[Yc�A�#*

lossM�;�m�w       �	{�[Yc�A�#*

lossg=�T�       �	P6[Yc�A�#*

lossd�j;�q�'       �	}�[Yc�A�#*

loss�=�|zo       �	�e[Yc�A�#*

loss�Ӑ<$���       �	k[Yc�A�#*

lossil�<s�џ       �	ܝ[Yc�A�#*

losszy;=*z'&       �	�3[Yc�A�$*

loss\��;��       �	��[Yc�A�$*

loss�?/=�4R�       �	3�	[Yc�A�$*

loss,��</L��       �	K
[Yc�A�$*

loss�S�=ݤ�       �	��
[Yc�A�$*

lossH�<�1�       �	�x[Yc�A�$*

lossr� <���L       �	�[Yc�A�$*

loss�'0<�Pa       �	��[Yc�A�$*

lossײ<a��9       �	~[Yc�A�$*

loss�vd;�� .       �	�[Yc�A�$*

loss!�H< 8R�       �	�[Yc�A�$*

loss�C�<�B�B       �	�K[Yc�A�$*

loss�C=���       �	��[Yc�A�$*

loss��;Z�E�       �	 s[Yc�A�$*

loss�X$=\$�t       �	[Yc�A�$*

loss]�P<�o{       �	J�[Yc�A�$*

loss���=at�"       �	3T[Yc�A�$*

lossx>�<���_       �	p�[Yc�A�$*

lossq�#=:/�       �	��[Yc�A�$*

loss��'<8��       �	%[Yc�A�$*

lossS�=)�S       �	e[Yc�A�$*

loss6.Z=#m�#       �	��[Yc�A�$*

loss<�=��5�       �	!�[Yc�A�$*

lossWV=��ζ       �	X�[Yc�A�$*

loss;��<�e       �	m [Yc�A�$*

loss&7�;��k1       �	1�[Yc�A�$*

loss/b-<W�A%       �	qT[Yc�A�$*

loss�y=%C��       �	H�[Yc�A�$*

loss�	=�@b�       �	��[Yc�A�$*

loss�i{;q��       �	�,[Yc�A�$*

lossP<��Ζ       �	%�[Yc�A�$*

loss���;����       �	�b[Yc�A�$*

loss��<:&�{       �	��[Yc�A�$*

loss��=�W>\       �	�[Yc�A�$*

loss�[�<|?�W       �	l?[Yc�A�$*

loss�xx=�       �	O�[Yc�A�$*

lossz�<���       �	��[Yc�A�$*

loss�=�=��       �	�% [Yc�A�$*

loss��?<�       �	�� [Yc�A�$*

loss ?�=T�       �	)Y![Yc�A�$*

loss[�=k�ip       �	R�![Yc�A�$*

lossV��<Mʬ       �	��"[Yc�A�$*

lossVvO<ϑ��       �	�P#[Yc�A�$*

loss��A=���        �	?�#[Yc�A�$*

lossϱ=x�4-       �	J|$[Yc�A�$*

loss@�h<�m��       �	f%[Yc�A�$*

loss�+�<;�Y�       �	v�%[Yc�A�$*

loss!s�=2hG       �	d=&[Yc�A�$*

lossR��<3{>       �	��&[Yc�A�$*

loss��`=�#�       �	|'[Yc�A�$*

lossQ�<:�D       �	�([Yc�A�$*

loss��=5�U�       �	ծ([Yc�A�$*

loss�b�<&�D       �	|D)[Yc�A�$*

loss$=%=zל       �	��)[Yc�A�$*

loss֗�;��S�       �	mt*[Yc�A�$*

lossx�<�4       �	�+[Yc�A�$*

loss4>1�       �	��+[Yc�A�$*

loss��e<�c))       �	m;,[Yc�A�$*

lossS,�<�yT�       �	9�,[Yc�A�$*

loss�-==�n_       �	n-[Yc�A�$*

loss��5<I[L�       �	.[Yc�A�$*

lossa�6<JnKr       �	u�.[Yc�A�$*

loss��<�|�        �	�D/[Yc�A�$*

lossl��=��
�       �	��/[Yc�A�$*

loss:0;
�#�       �	�0[Yc�A�$*

loss�i�<��]       �	n41[Yc�A�$*

loss�=c��       �	��1[Yc�A�$*

loss'<��       �	;r2[Yc�A�$*

loss_A�<���       �	�3[Yc�A�$*

loss�{�;����       �	:�3[Yc�A�$*

lossO�)=l��       �	#M4[Yc�A�$*

loss�ѽ<,e�       �	��4[Yc�A�$*

loss.v>�F��       �	��5[Yc�A�$*

loss�:=��d       �	�*6[Yc�A�$*

loss�M�<���       �	��6[Yc�A�$*

loss�u�;8Y�p       �	�g7[Yc�A�$*

loss�QJ=`�XM       �	��7[Yc�A�$*

loss6
�<gp5�       �	��8[Yc�A�$*

loss��<��85       �	\9[Yc�A�$*

loss�J�;�q(�       �	*�9[Yc�A�$*

lossk�=���       �	A�:[Yc�A�$*

loss���;��-�       �	6;[Yc�A�$*

lossWC0<���       �	��;[Yc�A�$*

lossO��;����       �	�n<[Yc�A�$*

loss��'=U�.G       �	�=[Yc�A�$*

loss��g<��8;       �	w�=[Yc�A�$*

loss�#�=��!�       �	�\>[Yc�A�$*

loss�j;w�+       �	�>[Yc�A�$*

loss6vc<�1��       �	��?[Yc�A�$*

loss��+<4�&�       �	�N@[Yc�A�$*

lossv͊<���       �	�@[Yc�A�$*

losssDI<>25�       �	8�A[Yc�A�$*

loss���=� �       �	GB[Yc�A�$*

loss��<��yC       �	��B[Yc�A�$*

loss{�><vitq       �	CXC[Yc�A�$*

lossq�<�;��       �	#�C[Yc�A�$*

loss=�<��        �	T�D[Yc�A�$*

losshIn=����       �	�0E[Yc�A�$*

loss�X<��w%       �	��E[Yc�A�$*

loss�f�<Yk��       �	OWF[Yc�A�$*

loss��w;�?�       �	]�F[Yc�A�$*

loss�Gh<) _e       �	��G[Yc�A�$*

loss|J<����       �	0H[Yc�A�$*

lossLY�;Ā~       �	��H[Yc�A�$*

loss/)<�0�+       �	��I[Yc�A�$*

lossΘ�<����       �	J[Yc�A�$*

loss�=��|       �	��J[Yc�A�$*

loss��
=ӵ       �	�gK[Yc�A�$*

loss�A<ꍇ       �	)L[Yc�A�$*

loss>={k��       �	�M[Yc�A�$*

loss��=�'��       �	ڬM[Yc�A�$*

loss��;�=)       �	VJN[Yc�A�$*

lossF4�<t�|       �	b�N[Yc�A�$*

lossj
=ƭjs       �	�IP[Yc�A�$*

loss
�<a�w�       �	p�P[Yc�A�$*

loss�5;N�       �	b�Q[Yc�A�$*

loss�/�<.A"       �	�#R[Yc�A�$*

lossX�<���9       �	�R[Yc�A�$*

lossA�i;%�i~       �	y\S[Yc�A�$*

losst�;)�?�       �	��S[Yc�A�$*

loss�?<<��8       �	C�T[Yc�A�$*

loss�u�<�A��       �	�BU[Yc�A�$*

loss@�m;"@B�       �	��U[Yc�A�$*

loss=�%<�w��       �	�yV[Yc�A�$*

loss&ݍ<�       �	)uW[Yc�A�$*

loss��e:*� �       �	�X[Yc�A�$*

loss-�:ޥ��       �	P�X[Yc�A�$*

loss�^e<�a�       �	�OY[Yc�A�$*

loss+$<5#{       �	x�Y[Yc�A�%*

loss�O�;���       �	Y�Z[Yc�A�%*

loss�+<�*       �	�![[Yc�A�%*

loss��;(avU       �	ϼ[[Yc�A�%*

loss�=�8�       �	;V\[Yc�A�%*

loss�{�:O��x       �	��\[Yc�A�%*

loss<S:�?m       �	�][Yc�A�%*

loss�U4:�ܕU       �	 &^[Yc�A�%*

loss��:4I׻       �	N�^[Yc�A�%*

lossnY<���(       �	�_[Yc�A�%*

loss���;O,��       �	�7`[Yc�A�%*

loss�uC:�dv�       �	�`[Yc�A�%*

lossj�:E��;       �	�b[Yc�A�%*

loss&u�=��y       �	��b[Yc�A�%*

loss��:g�/�       �	�Cc[Yc�A�%*

lossi�l=��X�       �	��c[Yc�A�%*

loss�m=���       �	׆d[Yc�A�%*

lossE�8<	K%H       �	�&e[Yc�A�%*

loss}g�;�q�r       �	�e[Yc�A�%*

loss��#<6��       �	5f[Yc�A�%*

loss�K=�-��       �	�"g[Yc�A�%*

loss�=4O��       �	74h[Yc�A�%*

loss��;}9��       �	��h[Yc�A�%*

lossO-�<7.�       �	P�i[Yc�A�%*

loss�Q�<���       �	A+j[Yc�A�%*

loss�6|=�߯�       �	��j[Yc�A�%*

loss�p�<P��       �	�sk[Yc�A�%*

loss�b�<�I]       �	�#l[Yc�A�%*

loss��~=�z�c       �	Q�l[Yc�A�%*

lossi_�<�G�:       �	�m[Yc�A�%*

loss�`"=v�I/       �	��n[Yc�A�%*

loss��%<�"�       �	�No[Yc�A�%*

lossft�=�Fk       �	f�o[Yc�A�%*

loss�H
=���       �	��p[Yc�A�%*

loss��<��_       �	��q[Yc�A�%*

loss���;��3�       �	lAr[Yc�A�%*

loss�K<=�`�       �	��r[Yc�A�%*

loss�CH:��4       �	�ss[Yc�A�%*

lossl�;cҬ�       �	t[Yc�A�%*

losssa2;�f       �	u�t[Yc�A�%*

lossi\m=��/       �	gdu[Yc�A�%*

loss�uN;���l       �	/�u[Yc�A�%*

loss��=�,x       �		�v[Yc�A�%*

loss)�A=Rr�       �	�>w[Yc�A�%*

lossx?�<H�lA       �	x[Yc�A�%*

lossŰX<�>       �	ˢx[Yc�A�%*

loss=T�;�iu5       �	�8y[Yc�A�%*

loss�E�:Tk�       �	��y[Yc�A�%*

loss���<�H;       �	J_z[Yc�A�%*

loss�Us<S?�W       �	��z[Yc�A�%*

loss�mM<D�:�       �	��{[Yc�A�%*

loss�<��Z       �	zR|[Yc�A�%*

loss�(�=oQs       �	�|[Yc�A�%*

loss5�!=zM��       �	�}[Yc�A�%*

loss�C<��\Z       �	�O~[Yc�A�%*

lossH��<����       �	�~[Yc�A�%*

loss_6<1z�U       �	3�[Yc�A�%*

loss$	�<'�       �	?o�[Yc�A�%*

loss65<'k0K       �	) �[Yc�A�%*

lossĺ�<���       �	��[Yc�A�%*

loss��Y=�@:8       �	Ps�[Yc�A�%*

loss��;2���       �	cG�[Yc�A�%*

loss4�=SZ�       �	��[Yc�A�%*

loss��;�3��       �	@��[Yc�A�%*

loss4�W:q�       �	5(�[Yc�A�%*

loss�]<gw=�       �	�K�[Yc�A�%*

loss��+<l`�\       �	.�[Yc�A�%*

loss���=���       �	��[Yc�A�%*

loss���="s5       �	��[Yc�A�%*

loss�>=<��z       �	�5�[Yc�A�%*

loss��<B
�?       �	!ͣ[Yc�A�%*

loss=H�8       �	=e�[Yc�A�%*

loss�k#;t���       �	U��[Yc�A�%*

lossR0<=��K       �	oե[Yc�A�%*

lossp�<���c       �	k�[Yc�A�%*

loss��9;T�       �	���[Yc�A�%*

loss�=�Ԛ	       �	���[Yc�A�%*

loss�<Q���       �	J�[Yc�A�%*

loss���<�:��       �	q�[Yc�A�%*

lossxp=�Z�c       �	(|�[Yc�A�%*

loss�	=�g��       �	W"�[Yc�A�%*

loss�x:�)�}       �	iê[Yc�A�%*

loss��;�-��       �	�[�[Yc�A�%*

loss3A$<$e�?       �	�[Yc�A�%*

loss��\<����       �	��[Yc�A�%*

loss,yl;aW0�       �	/��[Yc�A�%*

loss��!>Vޞ+       �	�"�[Yc�A�%*

loss9��< dE�       �	���[Yc�A�%*

losso0�=od1�       �	PQ�[Yc�A�%*

loss���;=5o       �	��[Yc�A�%*

lossX��<��       �	ӟ�[Yc�A�%*

lossf�y<�v��       �	�E�[Yc�A�%*

loss��i;n���       �	�ݱ[Yc�A�%*

loss	ځ<)�&       �	!t�[Yc�A�%*

loss��=ؔ�U       �	��[Yc�A�%*

loss}B�=i���       �	���[Yc�A�%*

loss1�<�*       �	l�[Yc�A�%*

loss�ڃ<�v�       �	�#�[Yc�A�%*

loss�$�=B��W       �	׽�[Yc�A�%*

loss8u�;'��       �	�R�[Yc�A�%*

loss_��<���       �	�[Yc�A�%*

loss ��:U7�;       �	���[Yc�A�%*

loss�bu<��)�       �	@4�[Yc�A�%*

loss�T�=��       �	�Ÿ[Yc�A�%*

loss�%�<��%�       �	�Z�[Yc�A�%*

loss�8�;(i       �	�[Yc�A�%*

loss��{<7.��       �	M��[Yc�A�%*

loss�M�<V��*       �	 6�[Yc�A�%*

losshE=�)
z       �	ͻ[Yc�A�%*

loss�v�=�H       �	4e�[Yc�A�%*

loss��r=��O�       �	/��[Yc�A�%*

lossI*�<�Җ�       �	ꔽ[Yc�A�%*

loss�;�<���       �	�5�[Yc�A�%*

loss:��;!1�}       �	�վ[Yc�A�%*

loss��;)���       �	��[Yc�A�%*

loss*Й;��n�       �	��[Yc�A�%*

loss�=ϧ(�       �	+�[Yc�A�%*

loss��;B�.�       �	ګ�[Yc�A�%*

loss[�	>h;A�       �	h@�[Yc�A�%*

loss
u�<h,�       �	���[Yc�A�%*

loss�:2n o       �	X��[Yc�A�%*

loss�|�;��6�       �	�F�[Yc�A�%*

lossL%�;���       �	w��[Yc�A�%*

loss��<�?�o       �	?p�[Yc�A�%*

loss\%9=�,�8       �	�[Yc�A�%*

loss��=N��       �	��[Yc�A�%*

loss�vi<�
o       �	Qf�[Yc�A�%*

loss�d=S�        �	���[Yc�A�%*

loss)-e=X%��       �	X��[Yc�A�&*

loss�
<2�       �	#g�[Yc�A�&*

loss6��=�{ߘ       �	L6�[Yc�A�&*

lossI��<!�`�       �	y��[Yc�A�&*

loss���;����       �	�a�[Yc�A�&*

loss�ߙ<�S�       �	D��[Yc�A�&*

loss��D<o��e       �	v��[Yc�A�&*

lossM��; �O       �	�+�[Yc�A�&*

loss�F5=
T;%       �	��[Yc�A�&*

loss���<��D�       �	sd�[Yc�A�&*

loss�.<�S)�       �	 �[Yc�A�&*

loss��&=�[�       �	$��[Yc�A�&*

lossr��=9�c*       �	�4�[Yc�A�&*

loss�@?;|��       �	���[Yc�A�&*

lossCn�<��F       �	g�[Yc�A�&*

loss��;?�;       �	�[Yc�A�&*

losszO-=��a
       �	���[Yc�A�&*

loss�-�<3��       �	���[Yc�A�&*

loss��Z=zg�       �	�'�[Yc�A�&*

losshU�<�y+�       �	���[Yc�A�&*

lossK�;��`�       �	[^�[Yc�A�&*

loss9)=dճ       �	���[Yc�A�&*

loss�u~<o�7>       �	���[Yc�A�&*

loss��Z<P��1       �	^e�[Yc�A�&*

loss�`�<�Cog       �	n��[Yc�A�&*

loss���<O�[       �	u��[Yc�A�&*

loss��<k��       �	�H�[Yc�A�&*

loss2�+<��n       �	d��[Yc�A�&*

loss��B=��H       �	${�[Yc�A�&*

lossA<Y�}       �	8�[Yc�A�&*

loss�<(�       �	���[Yc�A�&*

loss�&�<^ d�       �	��[Yc�A�&*

loss��<a�N       �	��[Yc�A�&*

loss���=f       �	���[Yc�A�&*

loss�w5<����       �	 y�[Yc�A�&*

lossQ*=�aP        �	�c�[Yc�A�&*

loss3<yNA�       �	a��[Yc�A�&*

loss\�<)�f�       �	7��[Yc�A�&*

losszv:�>!�       �	�h�[Yc�A�&*

losshG�<�\A       �	d�[Yc�A�&*

loss&�1;�]�       �	^��[Yc�A�&*

loss�c5=�n�       �	5�[Yc�A�&*

loss��S=:���       �	���[Yc�A�&*

loss� �<��Y�       �	�f�[Yc�A�&*

lossڟ�;|U�       �	K�[Yc�A�&*

loss�2=4r�       �	T��[Yc�A�&*

loss�mg<k�       �	�N�[Yc�A�&*

loss�<���       �	���[Yc�A�&*

loss�ߧ<�] 9       �	��[Yc�A�&*

loss��Q=�9~       �	!�[Yc�A�&*

lossu:�ԫ       �	��[Yc�A�&*

loss@F=%��       �	�M�[Yc�A�&*

lossD�:Fo�       �	���[Yc�A�&*

loss21s;�VQ�       �	��[Yc�A�&*

lossc�l=��       �	�<�[Yc�A�&*

lossD`<��h�       �	|��[Yc�A�&*

loss�]�<��pK       �	�i�[Yc�A�&*

loss��=<g�.�       �	�!�[Yc�A�&*

loss1��;s<�       �		��[Yc�A�&*

loss�J<�|�       �	�b�[Yc�A�&*

lossA�=�N��       �	L��[Yc�A�&*

loss���;è��       �	���[Yc�A�&*

lossa&%=��       �	
/�[Yc�A�&*

loss�c�<}��       �	���[Yc�A�&*

loss�~�;O��H       �	�m�[Yc�A�&*

lossEZ�<�Fym       �	�W�[Yc�A�&*

lossA=�<���       �	��[Yc�A�&*

losski=c�wD       �	��[Yc�A�&*

loss��:<؊�m       �	�9�[Yc�A�&*

lossf<�K�       �	4��[Yc�A�&*

lossS֗:(	�       �	�z�[Yc�A�&*

loss��=0#J       �	��[Yc�A�&*

loss��6<:���       �	���[Yc�A�&*

loss��<2��       �	r�[Yc�A�&*

loss�-_<!p��       �	��[Yc�A�&*

lossT�;�W�       �	���[Yc�A�&*

loss	�K;�3Ե       �	�E�[Yc�A�&*

lossl�<ꂡi       �	���[Yc�A�&*

loss�,�<�Nn�       �	:��[Yc�A�&*

loss�,+<z�V       �	�G�[Yc�A�&*

loss��=�]�       �	f��[Yc�A�&*

loss�6<RO�E       �	ޑ�[Yc�A�&*

loss�]�<��>       �	�7�[Yc�A�&*

loss��<2���       �	��[Yc�A�&*

loss��H:��=�       �	ē \Yc�A�&*

loss��<)�       �	�Q\Yc�A�&*

loss�oz;-��q       �	�>\Yc�A�&*

loss}f�<��       �	#�\Yc�A�&*

lossź�:c���       �	σ\Yc�A�&*

loss���<�<�	       �	1%\Yc�A�&*

loss�85<�{�       �	��\Yc�A�&*

loss��=Z�        �	�h\Yc�A�&*

loss��;|��       �	�?\Yc�A�&*

loss!<<K<�       �	K�\Yc�A�&*

loss1��=�l�       �	w�\Yc�A�&*

loss-�}<���       �	�\Yc�A�&*

loss�U;�~�       �	̸\Yc�A�&*

lossw��<;C?       �	6Y	\Yc�A�&*

lossH�;�/�_       �	D�	\Yc�A�&*

loss<�=���k       �	��
\Yc�A�&*

loss��/;����       �	$b\Yc�A�&*

lossMr�< fzd       �	\Yc�A�&*

lossAh�;�a0�       �	}�\Yc�A�&*

lossh�<�й�       �	 �\Yc�A�&*

lossO��<z��       �	�M\Yc�A�&*

lossG<<ǭ��       �	=�\Yc�A�&*

loss�W�;�٬;       �	ڒ\Yc�A�&*

loss6H�=�*��       �	�5\Yc�A�&*

loss�)�;pܖ�       �	a�\Yc�A�&*

loss��;�v�       �	�{\Yc�A�&*

loss�3�=���5       �	�\Yc�A�&*

loss��<��L       �	��\Yc�A�&*

lossW+�<pj�$       �	�X\Yc�A�&*

lossn�<}<��       �	��\Yc�A�&*

loss��=z:�       �	ܞ\Yc�A�&*

loss�r�;�Ӛ�       �	!@\Yc�A�&*

losshD�;@X �       �	0�\Yc�A�&*

loss/D|:_-       �	�y\Yc�A�&*

loss��<��       �	�\Yc�A�&*

loss F�;-�}�       �	�\Yc�A�&*

loss�G$;`(y�       �	'O\Yc�A�&*

loss��=<����       �	l�\Yc�A�&*

lossd{�<t���       �	��\Yc�A�&*

loss-B�=]o�|       �	@\Yc�A�&*

lossTj�;b껊       �	e�\Yc�A�&*

loss��<_�4       �	�{\Yc�A�&*

lossN��<_fP�       �	�#\Yc�A�&*

loss�Q=z�a       �	��\Yc�A�&*

lossz�T<�J%       �	)_\Yc�A�'*

loss�R�;�0�       �	�\Yc�A�'*

loss���;X�!       �	��\Yc�A�'*

loss�Bv;��       �	�9\Yc�A�'*

loss>��<�U�       �	v \Yc�A�'*

loss�O�<2|�       �	�� \Yc�A�'*

loss�R�<�V�       �	^�!\Yc�A�'*

loss�*W=h��Q       �	�D"\Yc�A�'*

loss�;����       �	��#\Yc�A�'*

lossj�%=$���       �	=b$\Yc�A�'*

losse.+=�(H�       �	t(%\Yc�A�'*

lossW;��n�       �	�%\Yc�A�'*

loss��<t!��       �	4�&\Yc�A�'*

loss_�P;����       �	�'\Yc�A�'*

loss?�E;=���       �	{/(\Yc�A�'*

loss?r=��I�       �	�(\Yc�A�'*

loss]�7<\}��       �	e *\Yc�A�'*

lossO�;����       �	F?+\Yc�A�'*

loss�D)<e�+Y       �	�F,\Yc�A�'*

lossV��<���       �	d:-\Yc�A�'*

loss���<[(�r       �	��-\Yc�A�'*

loss�(<Ue�       �	 
/\Yc�A�'*

loss_�.<���X       �	#I0\Yc�A�'*

loss�1=�Hl       �	\�0\Yc�A�'*

loss�_=���       �	�}1\Yc�A�'*

lossrͣ=n��       �	Y2\Yc�A�'*

loss=ۃ=g��s       �	 �2\Yc�A�'*

loss;2�<�E�       �	�"4\Yc�A�'*

losst�<��F�       �	�4\Yc�A�'*

loss#�=�(�x       �	?�5\Yc�A�'*

loss��m;�/�3       �	�N6\Yc�A�'*

loss��<�P��       �	��6\Yc�A�'*

loss��<jA��       �	u�7\Yc�A�'*

loss6��<Vg��       �	/8\Yc�A�'*

loss�	�<Y�P       �	��8\Yc�A�'*

loss�7�=��9Z       �	��9\Yc�A�'*

loss*$= O�       �	p":\Yc�A�'*

lossf<9<���       �	b�:\Yc�A�'*

loss�ׇ=���M       �	��;\Yc�A�'*

lossJ�=/��       �	�v<\Yc�A�'*

loss�<���       �	k=\Yc�A�'*

loss�/=r�@       �	�=\Yc�A�'*

loss�]><`+g       �	�?>\Yc�A�'*

loss�=�<Z��       �	��>\Yc�A�'*

loss���<�rj�       �	�u?\Yc�A�'*

loss���;�ܺ�       �	�@\Yc�A�'*

loss��e;�`�       �	f�@\Yc�A�'*

lossaZ=k�0�       �	�YA\Yc�A�'*

loss��<)�)       �	��A\Yc�A�'*

lossq�;]���       �	��B\Yc�A�'*

loss�|<
��       �	6C\Yc�A�'*

loss�u�< (�]       �	��C\Yc�A�'*

loss<�8,       �	�zD\Yc�A�'*

loss�12=�7�/       �	@E\Yc�A�'*

loss3v{<�s��       �	�RF\Yc�A�'*

loss9�;���       �	p�F\Yc�A�'*

lossA)<zl       �	ʉG\Yc�A�'*

loss�\=_��f       �	�&H\Yc�A�'*

loss�>�<]�[       �	��H\Yc�A�'*

loss9k=/��u       �	NcI\Yc�A�'*

lossL��<���       �	uJ\Yc�A�'*

losse�,<��V       �	�J\Yc�A�'*

lossձ�:��`       �	�GK\Yc�A�'*

loss���:"��A       �	��K\Yc�A�'*

loss!<��       �	$�L\Yc�A�'*

loss��<섘�       �	�:M\Yc�A�'*

lossW�{;����       �	a�M\Yc�A�'*

loss��<6k�       �	�N\Yc�A�'*

loss��=���!       �	�[O\Yc�A�'*

loss�;�c�       �	��O\Yc�A�'*

loss�� =`�:F       �	НP\Yc�A�'*

lossl�0=��\�       �	�FQ\Yc�A�'*

loss}Y�;�I*w       �	&�Q\Yc�A�'*

loss,e�;�s)�       �	b�R\Yc�A�'*

loss�Y;�f��       �	�$S\Yc�A�'*

lossr�"=�C�<       �	s�S\Yc�A�'*

loss�I�=L��       �	�[T\Yc�A�'*

loss�1U<��       �	\�T\Yc�A�'*

loss�9<�]KM       �	S�U\Yc�A�'*

lossRћ;�W       �	�mV\Yc�A�'*

loss�CS=h}�b       �	�W\Yc�A�'*

loss�f=y
��       �	��W\Yc�A�'*

loss�Ƥ<p�+       �	�LX\Yc�A�'*

loss<��=jKg�       �	��X\Yc�A�'*

loss��=y�-[       �	��Y\Yc�A�'*

loss:*�:�n|�       �	r7Z\Yc�A�'*

loss�Է;Ć�       �	M�Z\Yc�A�'*

loss��]=]�{�       �	J{[\Yc�A�'*

loss7u/<ͻF>       �	~\\Yc�A�'*

loss#��;}�       �	��\\Yc�A�'*

loss���;nx�S       �	�Z]\Yc�A�'*

lossELC<�\�       �	#�]\Yc�A�'*

loss�\=��       �	�^\Yc�A�'*

loss��<��1U       �	�=_\Yc�A�'*

lossvKc<�Ҭ%       �	��_\Yc�A�'*

loss��(=�pK�       �	��`\Yc�A�'*

loss���<0�d       �	q�a\Yc�A�'*

loss�j;=c��       �	[�b\Yc�A�'*

loss]�a;gq4       �	��c\Yc�A�'*

lossc�;V��       �	��d\Yc�A�'*

lossy==�0y�       �	��e\Yc�A�'*

loss���:��       �	��f\Yc�A�'*

loss�e:���_       �	�g\Yc�A�'*

loss��=��N       �	��h\Yc�A�'*

lossꛗ=>y��       �	�si\Yc�A�'*

loss�@<�,�       �	2�j\Yc�A�'*

loss�F=�T�       �	�qk\Yc�A�'*

loss��<�_�F       �	�sl\Yc�A�'*

lossZ��<�ο]       �	�Sm\Yc�A�'*

lossE��<2;H�       �	�|n\Yc�A�'*

lossz=�MA�       �	��o\Yc�A�'*

loss��><�]-       �	~Tp\Yc�A�'*

loss(�=��w       �	�p\Yc�A�'*

loss�f�=� �       �	��q\Yc�A�'*

loss� �=�s�       �	OZr\Yc�A�'*

loss.��;&%��       �	2s\Yc�A�'*

loss�4_=�B�k       �	��s\Yc�A�'*

loss�y�<��A�       �	?�t\Yc�A�'*

loss��;\�n�       �	R}u\Yc�A�'*

lossM<:ya       �	ev\Yc�A�'*

loss�� =�ykg       �	�v\Yc�A�'*

loss\B�=[Y�       �	EHw\Yc�A�'*

loss.�z<Q�{       �	a�w\Yc�A�'*

loss�9<.�1       �	yx\Yc�A�'*

loss�^r=��:       �	�y\Yc�A�'*

losso#�;;sWN       �	ۥy\Yc�A�'*

loss���;«��       �	K<z\Yc�A�'*

loss��<�J9�       �	�z\Yc�A�'*

loss?�/<1s͑       �	i{\Yc�A�(*

loss��<r��       �	��{\Yc�A�(*

loss!tD=���       �	h�|\Yc�A�(*

loss���<z��=       �	�T}\Yc�A�(*

loss/";3�Jf       �	n�}\Yc�A�(*

loss=�k=���       �	>�~\Yc�A�(*

loss�P�<K"�4       �	.\Yc�A�(*

loss�Q�<i���       �	��\Yc�A�(*

loss��T<m	�h       �	�_�\Yc�A�(*

lossF�C=#Z�       �	��\Yc�A�(*

loss1�=��q@       �	k��\Yc�A�(*

lossԷi;����       �	D�\Yc�A�(*

loss(/�;��@       �	�؂\Yc�A�(*

lossrAq;*8��       �	�m�\Yc�A�(*

lossM�a;)�ih       �	_�\Yc�A�(*

lossH�_<���w       �	f��\Yc�A�(*

lossmA�<�U(       �	2<�\Yc�A�(*

loss�s�<89JX       �	0ԅ\Yc�A�(*

loss�]�<���o       �	�n�\Yc�A�(*

loss8��<���	       �	�
�\Yc�A�(*

lossט�:_���       �	���\Yc�A�(*

loss
�=�SE�       �	�;�\Yc�A�(*

lossX�T<C$@       �	�Ԉ\Yc�A�(*

loss)K�=��X]       �	�l�\Yc�A�(*

lossMl�;>��p       �	��\Yc�A�(*

loss�r�=�i>�       �	[��\Yc�A�(*

lossI$<=os2Y       �	1�\Yc�A�(*

loss�W
<{*��       �	�Ƌ\Yc�A�(*

loss��<�C�       �	J]�\Yc�A�(*

loss�d�<*6�       �	���\Yc�A�(*

loss))b<J�s       �	���\Yc�A�(*

loss`�<��       �	�*�\Yc�A�(*

loss_�z<f5F�       �	L\Yc�A�(*

loss �3<�vo       �	\W�\Yc�A�(*

loss�yH<���2       �	��\Yc�A�(*

loss�=�Y�       �	@��\Yc�A�(*

loss��:� �       �	K�\Yc�A�(*

loss�/<����       �	��\Yc�A�(*

loss��<-�X+       �	�T�\Yc�A�(*

lossJ <�4�       �	*��\Yc�A�(*

loss�(*<�֐3       �	s��\Yc�A�(*

loss�,�<J�H       �	�A�\Yc�A�(*

loss�u;��       �	<ڔ\Yc�A�(*

loss��;���q       �	M��\Yc�A�(*

lossDn�=u5       �	�S�\Yc�A�(*

loss�d<�2�       �	���\Yc�A�(*

loss�Ӧ<�a}       �	���\Yc�A�(*

loss�<�<b�wg       �	y=�\Yc�A�(*

lossr��<�~��       �	�ט\Yc�A�(*

loss݄�<#*d       �	�u�\Yc�A�(*

lossi�+<�"��       �	��\Yc�A�(*

loss6{�<^��       �	(��\Yc�A�(*

loss�;e=�6       �	�Y�\Yc�A�(*

lossŕ�<+�)       �	���\Yc�A�(*

loss�l;��f�       �	���\Yc�A�(*

lossԪ�<�OK-       �	i7�\Yc�A�(*

loss	��<��F�       �	Qޝ\Yc�A�(*

loss0	�=�G��       �	�~�\Yc�A�(*

loss��<�U��       �	��\Yc�A�(*

loss��;�<O^       �	!ɟ\Yc�A�(*

loss(!�<6!��       �	��\Yc�A�(*

loss3��=􈨅       �	�M�\Yc�A�(*

loss�~&=���Y       �	���\Yc�A�(*

loss�ӹ=��K       �	w��\Yc�A�(*

loss�u<{25]       �	%�\Yc�A�(*

loss��;��v       �	�ߣ\Yc�A�(*

loss���=]=�X       �	��\Yc�A�(*

loss7��<|�0       �	�!�\Yc�A�(*

loss�W�<���
       �	I��\Yc�A�(*

loss=j�<M��       �	�V�\Yc�A�(*

loss
�<gg��       �	��\Yc�A�(*

loss�ܐ<��E�       �	���\Yc�A�(*

loss���;�mU�       �	�G�\Yc�A�(*

lossms<�ܧ�       �	��\Yc�A�(*

loss��<��g�       �	��\Yc�A�(*

loss*��;��       �	�"�\Yc�A�(*

loss���<D��Y       �	�Ī\Yc�A�(*

loss�!�:a�D       �	��\Yc�A�(*

losss]<R�EM       �	4I�\Yc�A�(*

losse<�G       �	�\Yc�A�(*

loss��;
(5       �	��\Yc�A�(*

loss6�&<U(��       �	�"�\Yc�A�(*

loss�ņ<S��       �	Q��\Yc�A�(*

loss�'�=�+       �	F\�\Yc�A�(*

loss��<�y�       �	t�\Yc�A�(*

loss⊛<@9H       �	O��\Yc�A�(*

loss{�<�h�       �	K�\Yc�A�(*

loss�Ly=�ZM�       �	��\Yc�A�(*

loss���;)gݖ       �	ӄ�\Yc�A�(*

loss�D�<l�       �	+/�\Yc�A�(*

loss�+&=�hڹ       �	�ϳ\Yc�A�(*

loss�v=��j       �	s��\Yc�A�(*

loss���=�s       �	�/�\Yc�A�(*

lossc��<�2l}       �	[ҵ\Yc�A�(*

loss�߭<�L`       �	�o�\Yc�A�(*

lossR�="��       �	�\Yc�A�(*

loss�M�=>Z��       �	���\Yc�A�(*

loss4��;��7�       �	�J�\Yc�A�(*

loss��X<c�>       �	�\Yc�A�(*

loss�=;S��       �	���\Yc�A�(*

loss�m&=�:O�       �	�=�\Yc�A�(*

loss��,;p�s        �	�պ\Yc�A�(*

loss`�k=��(       �	�o�\Yc�A�(*

loss/V�;��       �	l�\Yc�A�(*

loss�W�=%��@       �	0��\Yc�A�(*

lossi�e=���~       �	�{�\Yc�A�(*

loss�"o=Wf�       �	��\Yc�A�(*

loss���;:��f       �	Χ�\Yc�A�(*

loss�[�<6�:       �	�G�\Yc�A�(*

loss�><Z�%�       �	�\Yc�A�(*

loss���;�O��       �	׉�\Yc�A�(*

loss���<�D3�       �	�!�\Yc�A�(*

loss2�M<eر�       �	���\Yc�A�(*

loss[�H<+���       �	jm�\Yc�A�(*

loss� =�       �	�
�\Yc�A�(*

lossf��<�l�       �	̵�\Yc�A�(*

loss��<x�T�       �	�O�\Yc�A�(*

loss疐;�Ǝ<       �	��\Yc�A�(*

loss6�)<���       �	4��\Yc�A�(*

loss�:A;���f       �	S�\Yc�A�(*

loss�
=���       �	���\Yc�A�(*

loss�-�<��v�       �	���\Yc�A�(*

losss7�=P�_       �	�2�\Yc�A�(*

loss�F�<��T       �	R�\Yc�A�(*

loss@�==��	�       �	_��\Yc�A�(*

loss�<Ǜ1�       �	#K�\Yc�A�(*

loss�<<$��E       �	n��\Yc�A�(*

lossf��<����       �	`v�\Yc�A�(*

loss{J<����       �	��\Yc�A�)*

loss�#=ޣ4�       �	h��\Yc�A�)*

loss�z�<�k�       �	܀�\Yc�A�)*

lossv�.<�'��       �	��\Yc�A�)*

loss�@�;ǥs       �	\��\Yc�A�)*

loss�i�<0�d'       �	�M�\Yc�A�)*

loss�C<Ac�       �	���\Yc�A�)*

lossL�=|Ԃ�       �	���\Yc�A�)*

loss�u<Oi�       �	�7�\Yc�A�)*

loss�3�<J}D       �	���\Yc�A�)*

loss��:<�j�       �	r�\Yc�A�)*

lossP�=J�D       �	�M�\Yc�A�)*

loss�M'<�ܶ       �	���\Yc�A�)*

lossv�;zN$�       �	A��\Yc�A�)*

loss
/�:�,�       �	�A�\Yc�A�)*

loss}�;��P/       �	���\Yc�A�)*

loss���=��W       �	A�\Yc�A�)*

loss�,O;�;�       �	�"�\Yc�A�)*

loss�E<x�)�       �	��\Yc�A�)*

loss��?=U�ҿ       �	�l�\Yc�A�)*

loss��;�Бx       �	��\Yc�A�)*

lossNU;��       �	���\Yc�A�)*

loss{�s:}3U       �	�]�\Yc�A�)*

loss3>ӑ�:       �	��\Yc�A�)*

loss�>�<��       �	��\Yc�A�)*

loss�.�<�փ       �	�K�\Yc�A�)*

loss:Ë;k�"       �	S��\Yc�A�)*

loss�I=e�k�       �	��\Yc�A�)*

loss���:o       �	�(�\Yc�A�)*

loss��:}kx~       �	���\Yc�A�)*

loss#�;[�       �	e�\Yc�A�)*

loss�Ҁ;0$?       �	�(�\Yc�A�)*

loss�Z�=R���       �	��\Yc�A�)*

lossJ� =�0?       �	���\Yc�A�)*

lossL�=p��c       �	P��\Yc�A�)*

lossi�<׸�       �	s��\Yc�A�)*

loss�Đ<ӗ/�       �	(�\Yc�A�)*

loss��<-�(�       �	��\Yc�A�)*

lossl��<�N1�       �	ZI�\Yc�A�)*

loss��o<9�m�       �	���\Yc�A�)*

loss�ˆ=�f$�       �	���\Yc�A�)*

loss / <Cޣ�       �	�?�\Yc�A�)*

lossi�H=���c       �	���\Yc�A�)*

lossi�4<ޚ�        �	m��\Yc�A�)*

lossw�6=\]}�       �	B�\Yc�A�)*

loss.,(;�T[       �	��\Yc�A�)*

lossF�e=�j�l       �	���\Yc�A�)*

loss�T];�G       �	�q�\Yc�A�)*

lossTz�<hU��       �	g�\Yc�A�)*

loss���<�t]       �	d��\Yc�A�)*

loss���;ę�       �	�M�\Yc�A�)*

loss�==�}�u       �	�\Yc�A�)*

loss%��<A�ś       �	���\Yc�A�)*

loss� �<?��{       �	e5�\Yc�A�)*

loss�A�:~���       �	���\Yc�A�)*

loss���;�L       �	�h�\Yc�A�)*

loss��@<$8�M       �	��\Yc�A�)*

loss�@�<Z�}m       �	���\Yc�A�)*

loss%��;X�       �	n��\Yc�A�)*

loss��<�T��       �	��\Yc�A�)*

lossm�;����       �	R��\Yc�A�)*

lossT(�;-1�%       �	~��\Yc�A�)*

loss8�9=�E        �	�'�\Yc�A�)*

loss�
�:7kyE       �	���\Yc�A�)*

loss`=N���       �	v5�\Yc�A�)*

loss7�;�KZ�       �	���\Yc�A�)*

loss��[=s�k�       �	"o�\Yc�A�)*

loss�^�;���!       �	��\Yc�A�)*

loss��[<�z�'       �	}��\Yc�A�)*

loss$�L=B2       �	�U�\Yc�A�)*

lossWK�<+Ah�       �	���\Yc�A�)*

loss���<�1��       �	���\Yc�A�)*

lossqk4=y���       �	���\Yc�A�)*

loss!^�;��g�       �	M�\Yc�A�)*

loss���<5uj=       �	>��\Yc�A�)*

loss�iz<M�6�       �	��\Yc�A�)*

loss�<��U       �	G8 ]Yc�A�)*

lossz�<p��       �	�]Yc�A�)*

losss�'=W       �	��]Yc�A�)*

lossWԤ;k�qg       �	�c]Yc�A�)*

loss��*<��V�       �	� ]Yc�A�)*

lossO�<7$N�       �	L�]Yc�A�)*

loss��<�3�~       �	o]Yc�A�)*

loss�X�;�W�       �	;6]Yc�A�)*

loss��?=C���       �	w�]Yc�A�)*

loss-�;ui$       �	hu]Yc�A�)*

loss7�9��N       �	�]Yc�A�)*

loss])y<�O��       �	��]Yc�A�)*

loss6R=@��       �	�U	]Yc�A�)*

loss��;^0]       �	k�	]Yc�A�)*

loss��:8�u       �	�
]Yc�A�)*

loss�:W��       �	F[]Yc�A�)*

loss�r�<�g�       �	��]Yc�A�)*

loss
{[:}��%       �	��]Yc�A�)*

loss�h9��y�       �	�/]Yc�A�)*

loss`':~�       �	��]Yc�A�)*

lossv�:�Y<3       �	��]Yc�A�)*

loss�-�<�KΚ       �	WB]Yc�A�)*

loss�b[<�3hx       �	��]Yc�A�)*

lossro:lEa`       �	�~]Yc�A�)*

loss��/;��c�       �	�]Yc�A�)*

lossC�=8v��       �	ػ]Yc�A�)*

loss7_a:�s�p       �	 U]Yc�A�)*

loss��=��n�       �	��]Yc�A�)*

loss���<Q(h�       �	��]Yc�A�)*

loss�u<�>u�       �	�x]Yc�A�)*

loss7f�<�Fl�       �	�]Yc�A�)*

loss��9;��G�       �	�]Yc�A�)*

lossa� =���       �	�R]Yc�A�)*

losst$
=�U�/       �	��]Yc�A�)*

loss��:R;/v       �	�]Yc�A�)*

lossӥ<��p�       �	�!]Yc�A�)*

loss�Cn<
q)d       �	�]Yc�A�)*

lossizW=����       �	y\]Yc�A�)*

lossHv=J"       �	 ]Yc�A�)*

loss��k;�Ooy       �	�]Yc�A�)*

loss@��<3�d       �	u<]Yc�A�)*

loss�z=j�>       �	�]Yc�A�)*

lossT�;����       �	u]Yc�A�)*

loss��}<*�8�       �	�]Yc�A�)*

loss�Xd=�{       �	G�]Yc�A�)*

lossȝ�;9�E       �	H]Yc�A�)*

loss�)0;͘j?       �	��]Yc�A�)*

loss�O<::��       �		�]Yc�A�)*

loss%{<hO��       �	J% ]Yc�A�)*

loss%S<է�       �	@� ]Yc�A�)*

loss��;��       �	�^!]Yc�A�)*

loss@��;��|       �	��!]Yc�A�)*

loss���<�4��       �	��"]Yc�A�**

loss�mi=+�0�       �	�E#]Yc�A�**

lossý�<����       �	��#]Yc�A�**

loss]��<�k��       �	�%]Yc�A�**

loss�̞;�6"-       �	�%]Yc�A�**

loss��I= ���       �	C=&]Yc�A�**

loss�{1=�Y�       �	��&]Yc�A�**

lossf1�9���       �	:x']Yc�A�**

loss��Q<�7��       �	(]Yc�A�**

lossE{�;z��t       �	��(]Yc�A�**

loss��f<�[o       �	�i)]Yc�A�**

loss���;�|�       �	A*]Yc�A�**

lossD�=����       �	�*]Yc�A�**

loss���<#�o)       �	�A+]Yc�A�**

loss`5�;	⒕       �	��+]Yc�A�**

loss�Dd<�>�       �	��,]Yc�A�**

loss6��;Kw       �	.-]Yc�A�**

loss��;�8       �	�-]Yc�A�**

loss8�:�˓�       �	u�.]Yc�A�**

lossd�<<;��       �	Q�/]Yc�A�**

loss�y=aHxl       �	�"0]Yc�A�**

loss{T@;�E�O       �	#�0]Yc�A�**

lossᶣ<�B       �	�a1]Yc�A�**

loss�ă<bN�       �	/�1]Yc�A�**

lossn��;��a�       �	��2]Yc�A�**

loss(��<�� �       �	ۦL]Yc�A�**

loss/��<"�o�       �	:M]Yc�A�**

lossܵ.<���N       �	[�M]Yc�A�**

lossR�5=6�֤       �	�fN]Yc�A�**

losss��<����       �		�N]Yc�A�**

loss�<���m       �	#�O]Yc�A�**

lossgQ<5�       �	�8P]Yc�A�**

loss֜ =��a       �	`�P]Yc�A�**

loss�v8=o���       �	�`Q]Yc�A�**

loss[$L=���4       �	'R]Yc�A�**

loss6�J<��;       �	�R]Yc�A�**

loss�Bl;Q�C�       �	WS]Yc�A�**

lossk��<��@�       �	qT]Yc�A�**

loss���;�޿       �	ѲT]Yc�A�**

lossa؈;WF?V       �	4HU]Yc�A�**

loss̐[<�h��       �	/�U]Yc�A�**

loss<��;�,��       �	\tV]Yc�A�**

loss�Y�;& \c       �	�MW]Yc�A�**

loss��;��       �	�W]Yc�A�**

loss��=�S       �	�yX]Yc�A�**

lossO@�;FŔ�       �	Y]Yc�A�**

lossV��=���W       �	�Y]Yc�A�**

lossD�:��ד       �	�EZ]Yc�A�**

loss�<I]�       �	��Z]Yc�A�**

loss�<K4�       �	�[]Yc�A�**

lossߊq;�b�v       �	u9\]Yc�A�**

loss[+�;/��       �	��\]Yc�A�**

lossPÒ<^��<       �	4f]]Yc�A�**

loss���<n�Xu       �	��]]Yc�A�**

loss�h�<����       �	V�^]Yc�A�**

loss���<^��B       �	�2_]Yc�A�**

loss�h=za       �	��_]Yc�A�**

loss�;J�i�       �	1\`]Yc�A�**

loss9=�d�s       �	��`]Yc�A�**

loss�=��B       �	��a]Yc�A�**

loss�F�<�<�       �	�hb]Yc�A�**

loss���; ��       �	� c]Yc�A�**

loss|��<�6Nz       �	�c]Yc�A�**

loss��G=�|�       �	0)d]Yc�A�**

loss�<�ZP       �	m�d]Yc�A�**

losss�u<2@�!       �	�`e]Yc�A�**

loss��==`�m       �	I�e]Yc�A�**

loss�U8;���o       �	c�f]Yc�A�**

lossP�;�T�       �	�5g]Yc�A�**

loss��#<H&(�       �	�i]Yc�A�**

loss��<�%�g       �	�i]Yc�A�**

lossߒ	=u]T�       �	�Gj]Yc�A�**

loss,��:���       �	��j]Yc�A�**

lossqD<�{N�       �	ޑk]Yc�A�**

losssn�:�FS�       �	(l]Yc�A�**

loss�;��       �	J�l]Yc�A�**

loss
�;4a��       �	Dmm]Yc�A�**

lossS9�<�p�       �	Ln]Yc�A�**

loss��=�|�l       �	_�n]Yc�A�**

lossѰ�;�L�       �	:�o]Yc�A�**

loss,KU:w�       �	�2p]Yc�A�**

loss}��9p�
�       �	m�p]Yc�A�**

loss��V:%d�       �	<�q]Yc�A�**

loss?l<�m�       �	�r]Yc�A�**

loss��=���       �	I�r]Yc�A�**

loss��=���       �	�Qs]Yc�A�**

loss1'�:����       �	��s]Yc�A�**

loss-D;1�-       �	��t]Yc�A�**

loss��<d۠�       �	u]Yc�A�**

loss)<< ���       �	��u]Yc�A�**

loss�!<�3��       �	�Cv]Yc�A�**

loss�=I�o       �	��v]Yc�A�**

loss]ԥ<U���       �	�hw]Yc�A�**

loss\r�<c�G       �	�w]Yc�A�**

loss�)�<)t�:       �	�x]Yc�A�**

loss&ژ=�e �       �	�:y]Yc�A�**

loss��
=�V�       �	��y]Yc�A�**

loss�B�<dce       �	2qz]Yc�A�**

loss�od;�l~       �	Q{]Yc�A�**

loss���<�l��       �	��{]Yc�A�**

loss���;��]�       �	RH|]Yc�A�**

lossn�:�$��       �	��|]Yc�A�**

loss�<1-��       �	h�}]Yc�A�**

lossnr�<��       �	+~]Yc�A�**

loss�u<�Zb^       �	1�~]Yc�A�**

lossq�?;��o�       �	L�]Yc�A�**

loss�v:<��b       �	]�]Yc�A�**

lossI�<�t�Q       �	,�]Yc�A�**

loss�`�;)	e       �	���]Yc�A�**

lossdY=�+�1       �	�,�]Yc�A�**

loss܎�:�8��       �	��]Yc�A�**

loss��A<�9u       �	߃]Yc�A�**

loss��<q�;Q       �	��]Yc�A�**

lossDDP=ބ��       �	]��]Yc�A�**

loss..=m�@       �	��]Yc�A�**

loss��t=�Ԋ�       �	�x�]Yc�A�**

lossE�D;��
       �	��]Yc�A�**

loss���<X��       �	j��]Yc�A�**

loss�m�<��;�       �	�B�]Yc�A�**

loss|�=AksF       �	ۊ]Yc�A�**

loss�"�<\L�j       �	<݋]Yc�A�**

lossb�;���       �	t�]Yc�A�**

loss�T�;a��       �	c�]Yc�A�**

loss�=���       �	��]Yc�A�**

loss���;���'       �	LO�]Yc�A�**

loss
�;GQP�       �	���]Yc�A�**

loss� �<�R��       �	�q�]Yc�A�**

loss?�q=u�^�       �	:�]Yc�A�**

loss��T;�P_�       �	K�]Yc�A�+*

loss�/�;P�l       �	���]Yc�A�+*

loss��k=Y���       �	�-�]Yc�A�+*

loss?
�;2[Z0       �	�Ē]Yc�A�+*

loss&4=��Z       �	\�]Yc�A�+*

loss�x�<�O&       �	g�]Yc�A�+*

lossH$�<8�)�       �	��]Yc�A�+*

loss��=1��u       �	['�]Yc�A�+*

lossaU=˼��       �	�ŕ]Yc�A�+*

loss\w<%�j       �	h[�]Yc�A�+*

lossa�o;)ư+       �	\�]Yc�A�+*

loss�C�;y�\�       �	���]Yc�A�+*

lossr�6<��Ѽ       �	�?�]Yc�A�+*

loss�":<����       �	�Ә]Yc�A�+*

loss{�V<i|	L       �	Mg�]Yc�A�+*

lossq�u<{�8       �	r��]Yc�A�+*

loss�u=��       �	���]Yc�A�+*

lossID^<(�'       �	ZI�]Yc�A�+*

loss!�|<�ˍ�       �	?�]Yc�A�+*

loss���<�l7�       �	j��]Yc�A�+*

lossJy<�c$�       �	7�]Yc�A�+*

loss-�x<�2�       �	�ҝ]Yc�A�+*

loss�{�<�#       �	�n�]Yc�A�+*

loss��<�7A3       �	��]Yc�A�+*

loss���:ӵ�^       �	�ğ]Yc�A�+*

loss���;؆�i       �	9^�]Yc�A�+*

loss��<f7�W       �	?�]Yc�A�+*

lossa�x<h*��       �	���]Yc�A�+*

loss U�<ՙ��       �	w��]Yc�A�+*

loss�\;"�"       �	�G�]Yc�A�+*

loss��e;�#�        �	9*�]Yc�A�+*

loss1��<�z��       �	{/�]Yc�A�+*

lossA�<t9�       �	�Х]Yc�A�+*

loss�	k<�&��       �	�m�]Yc�A�+*

loss��=R=�       �	lx�]Yc�A�+*

loss�:<b�i&       �	�]Yc�A�+*

loss�ޤ;ԧ�X       �	O�]Yc�A�+*

loss�j�;���       �	z��]Yc�A�+*

loss�� =;��       �	+��]Yc�A�+*

loss 9==�z�,       �	,��]Yc�A�+*

loss�$=�&��       �	=e�]Yc�A�+*

lossV��;���       �	�!�]Yc�A�+*

lossJ�<,�*W       �	2�]Yc�A�+*

loss�2<��x�       �	V�]Yc�A�+*

loss�=�:A�Z�       �	�#�]Yc�A�+*

loss��<#��        �	>[�]Yc�A�+*

lossM3<Г��       �	I�]Yc�A�+*

lossf�|<k+T       �	 �]Yc�A�+*

loss�EG<�_�e       �	��]Yc�A�+*

losst=�Ln�       �	j��]Yc�A�+*

loss�\6=.�9       �	���]Yc�A�+*

loss��\=�A,       �	�϶]Yc�A�+*

loss?:ѽ�a       �	@m�]Yc�A�+*

lossW%;�a�       �	��]Yc�A�+*

losso]<<[�\       �	���]Yc�A�+*

loss���<lr��       �	 E�]Yc�A�+*

loss36�;�_!�       �	
0�]Yc�A�+*

lossz"<co��       �	̺]Yc�A�+*

loss! �:)�h�       �	�h�]Yc�A�+*

loss�=j;��       �	l�]Yc�A�+*

lossJv�9Ӯv�       �	ע�]Yc�A�+*

lossԛ;w��       �	?;�]Yc�A�+*

loss�Md<���H       �	�ѽ]Yc�A�+*

lossŠM<7˄       �	Ed�]Yc�A�+*

loss���<�"�       �	]�]Yc�A�+*

loss�IQ<�$t�       �	��]Yc�A�+*

loss���;W!�*       �	�N�]Yc�A�+*

loss Zk= �x�       �	>��]Yc�A�+*

loss��;ۋ�Y       �	���]Yc�A�+*

loss�K<q��       �	%"�]Yc�A�+*

lossW,�<Nɚ�       �	@��]Yc�A�+*

loss�f�9w�L�       �	:Y�]Yc�A�+*

loss��<�ܼ       �		��]Yc�A�+*

lossѹ<)r��       �	���]Yc�A�+*

lossQ8/<Nb       �	,�]Yc�A�+*

loss��;���<       �	���]Yc�A�+*

loss�
v<��o0       �	f��]Yc�A�+*

loss�|&9
A�       �	m�]Yc�A�+*

loss-�d;
᛺       �	Ϻ�]Yc�A�+*

lossJ�#<c6�n       �	��]Yc�A�+*

loss8&l; ��%       �	���]Yc�A�+*

loss��;N,[b       �	��]Yc�A�+*

loss��F<S�       �	RE�]Yc�A�+*

loss�b=3��        �	:��]Yc�A�+*

loss��;�P9�       �	��]Yc�A�+*

loss<�t=�_S�       �	��]Yc�A�+*

loss��(<EB��       �	���]Yc�A�+*

loss��1=�7�       �	�E�]Yc�A�+*

loss��c<�3\F       �	U��]Yc�A�+*

loss5;
;6��e       �	�u�]Yc�A�+*

losslԱ;5�*I       �	�]Yc�A�+*

loss�?$;�+�       �	]��]Yc�A�+*

loss��o=����       �	oF�]Yc�A�+*

loss�-�<��        �	���]Yc�A�+*

loss�I;<� �       �	>{�]Yc�A�+*

loss�=��Պ       �	/�]Yc�A�+*

loss���;�Zۛ       �	[��]Yc�A�+*

loss@��::|,       �	{K�]Yc�A�+*

loss�b�:�=e       �	_��]Yc�A�+*

loss͐�=M�
       �	+��]Yc�A�+*

loss�<	_�       �	��]Yc�A�+*

lossd
�;<K)�       �	l��]Yc�A�+*

loss��<@��~       �	@O�]Yc�A�+*

loss��l=e�!G       �	���]Yc�A�+*

loss�5=,fO       �	���]Yc�A�+*

loss�� <)�gl       �	��]Yc�A�+*

loss,|�=���       �	`��]Yc�A�+*

loss�ő<	C�       �	G�]Yc�A�+*

loss�5�<5NDh       �	���]Yc�A�+*

loss3�;"�ͣ       �	�o�]Yc�A�+*

loss+�=����       �	j�]Yc�A�+*

lossҝ�<`�t       �	��]Yc�A�+*

loss�\=ʅ[3       �	�?�]Yc�A�+*

lossT	�=}��       �	:��]Yc�A�+*

lossF={%�       �	��]Yc�A�+*

loss��=�3�       �	O#�]Yc�A�+*

lossj=m��R       �	7��]Yc�A�+*

loss��`<���       �	p\�]Yc�A�+*

loss�;�;	ȷX       �	��]Yc�A�+*

loss.��<�(��       �	�1�]Yc�A�+*

lossE��;��+)       �	S��]Yc�A�+*

loss�J;|:h       �	ys�]Yc�A�+*

loss�<:��9       �	c�]Yc�A�+*

loss��4=+��       �	���]Yc�A�+*

loss��<<]+�       �	nL�]Yc�A�+*

loss�q<# �/       �	���]Yc�A�+*

lossv5<DC�@       �	c~�]Yc�A�+*

loss�d�:�:�c       �	��]Yc�A�+*

loss΁�:"�V�       �	���]Yc�A�,*

loss`��< #�       �	�`�]Yc�A�,*

loss��;�i       �	(�]Yc�A�,*

lossȄ<��Dy       �	Q��]Yc�A�,*

loss��2<�       �	ob�]Yc�A�,*

loss�3=;;{M�       �	�5�]Yc�A�,*

loss*PW<�,	�       �	���]Yc�A�,*

lossx`�=��f       �	$d�]Yc�A�,*

lossM�p;a�=Y       �	���]Yc�A�,*

loss��=~.�       �	���]Yc�A�,*

loss�=N�:�       �	*V�]Yc�A�,*

loss�L�<���"       �	���]Yc�A�,*

loss<O�<
w��       �	R��]Yc�A�,*

loss�D<<�\?
       �	;T�]Yc�A�,*

loss-';_��       �	���]Yc�A�,*

loss��<\�}4       �	 ��]Yc�A�,*

lossd�<����       �	�<�]Yc�A�,*

loss!�f<�x��       �	1��]Yc�A�,*

loss	�<X��       �	�i�]Yc�A�,*

lossOb�<�Ô       �	 �]Yc�A�,*

lossL)�<��T       �	ѕ�]Yc�A�,*

lossZ]�<�6�Z       �	)�]Yc�A�,*

loss��o<�F��       �	���]Yc�A�,*

loss{D�<(�o-       �	�a�]Yc�A�,*

lossq'C<bx
�       �	��]Yc�A�,*

loss�3<����       �	W��]Yc�A�,*

loss`��;G$��       �	c*�]Yc�A�,*

loss��;�U3       �	@��]Yc�A�,*

losso��<>��e       �	�n�]Yc�A�,*

lossB�<�,�       �	��]Yc�A�,*

loss��:E5       �	��]Yc�A�,*

loss�ّ<��ڍ       �	�F�]Yc�A�,*

loss��;m��       �	<��]Yc�A�,*

lossx=A<;T��       �	2u�]Yc�A�,*

lossÛ_:��˰       �	�0�]Yc�A�,*

loss��(=i��q       �	� ^Yc�A�,*

loss=[�;e��       �	� ^Yc�A�,*

loss�I=`߿�       �	�N^Yc�A�,*

lossx�=�e��       �	��^Yc�A�,*

loss��.=|��       �	�|^Yc�A�,*

lossɹ8<�6N@       �	�^Yc�A�,*

loss\Gb<ޠ��       �	O�^Yc�A�,*

loss�<)���       �	?R^Yc�A�,*

loss��w=�L�       �	;T^Yc�A�,*

loss�=��:       �	��^Yc�A�,*

loss��=����       �	4^Yc�A�,*

loss�"^:���       �	�	^Yc�A�,*

loss\�<�j8�       �	@�	^Yc�A�,*

loss��G=�       �	�?
^Yc�A�,*

loss<ǯ:Ї�       �	˝^Yc�A�,*

loss^�;A΅#       �	�i^Yc�A�,*

loss:��:���.       �	z�^Yc�A�,*

lossa{�<x���       �	S�^Yc�A�,*

loss2�V=��       �	0^Yc�A�,*

loss��U<5�       �	��^Yc�A�,*

lossO��<��>       �	��^Yc�A�,*

loss4�<cC�       �	�^Yc�A�,*

loss�y�<8�*I       �	IL^Yc�A�,*

lossz�9:�Vΰ       �	�^Yc�A�,*

lossL�9=���       �	ӣ^Yc�A�,*

loss���;��;�       �	�=^Yc�A�,*

lossZ@�:ҩ1|       �	��^Yc�A�,*

lossJ"�9����       �	�e^Yc�A�,*

loss�$:��       �	@�^Yc�A�,*

loss�2*<K/��       �	��^Yc�A�,*

lossC=� �       �	�3^Yc�A�,*

lossA=�4��       �	5�^Yc�A�,*

loss{=�(0       �	�t^Yc�A�,*

loss�<�>��       �	'^Yc�A�,*

loss�jG=y���       �	��^Yc�A�,*

loss��<<���`       �	Tp^Yc�A�,*

loss?-<��?       �	�^Yc�A�,*

loss
m�=�w�8       �	�^Yc�A�,*

loss��=��,       �	`^Yc�A�,*

loss�e)=���:       �	b�^Yc�A�,*

loss��<�N׃       �	��^Yc�A�,*

lossHk�<��q       �	m^Yc�A�,*

loss���;D���       �	��^Yc�A�,*

loss��]<�ᛮ       �	�T^Yc�A�,*

loss�
�<N9o�       �	#�^Yc�A�,*

loss�.U<�?       �	�^Yc�A�,*

lossn�+=d7       �	�) ^Yc�A�,*

loss��r;�.��       �	�� ^Yc�A�,*

lossx��;jˌ       �	�Z!^Yc�A�,*

lossh��<y�I+       �	9�!^Yc�A�,*

lossC
�:�E�       �	3�"^Yc�A�,*

loss
V=@/!       �	J�#^Yc�A�,*

lossX�<n�       �	tC$^Yc�A�,*

loss��<@H�       �	�P%^Yc�A�,*

lossm�z<��\       �	��&^Yc�A�,*

loss�`�<�� 
       �	GV'^Yc�A�,*

loss�%<��       �	��'^Yc�A�,*

loss�j�;�E�5       �	(�(^Yc�A�,*

loss�^=<)�T       �	�n)^Yc�A�,*

loss���<>KS`       �	F"*^Yc�A�,*

loss���<�(�c       �	Fy+^Yc�A�,*

loss��<Ga#�       �	�),^Yc�A�,*

loss$�<�3��       �	��,^Yc�A�,*

loss6*3=���X       �	��-^Yc�A�,*

lossX�<��[       �	��.^Yc�A�,*

loss���:�r       �	��/^Yc�A�,*

loss-�;��       �	��0^Yc�A�,*

loss�:�;�u�i       �	n�1^Yc�A�,*

loss f�=_��       �	�w2^Yc�A�,*

loss���9�(x�       �	�V3^Yc�A�,*

loss�>�<e�<�       �	�(4^Yc�A�,*

loss�=<�w       �	�4^Yc�A�,*

loss��=�!gz       �	�96^Yc�A�,*

loss�I�:�^        �	��6^Yc�A�,*

lossؙ�:���       �	��7^Yc�A�,*

loss��<��r       �	1�8^Yc�A�,*

loss��:;�]��       �	�[9^Yc�A�,*

lossE�<#���       �	�9^Yc�A�,*

loss2\�<M�U       �	�:^Yc�A�,*

loss72=� �	       �	�-;^Yc�A�,*

loss/�:h��E       �	ms<^Yc�A�,*

loss��<r�c       �	LU=^Yc�A�,*

loss͵�<,��       �	>^Yc�A�,*

loss�c;��t       �	�>^Yc�A�,*

loss:+~:R��       �	�w@^Yc�A�,*

loss%;�:��?V       �	rA^Yc�A�,*

loss���<���'       �	�A^Yc�A�,*

loss���<{��       �	ˁB^Yc�A�,*

loss�x=�*��       �	�C^Yc�A�,*

loss�:?<8i=J       �	ŭC^Yc�A�,*

loss��<zٗ�       �	5CD^Yc�A�,*

loss�O ;e~��       �	�D^Yc�A�,*

lossow�;�<�;       �	|E^Yc�A�,*

loss��>;�E�       �	�F^Yc�A�-*

lossW�k=zY?       �	��F^Yc�A�-*

loss�p�<5Lz�       �	��G^Yc�A�-*

loss���;�}       �	/H^Yc�A�-*

loss�=z�       �	ޮH^Yc�A�-*

losse{�=�NTH       �	xI^Yc�A�-*

loss<8=�L�r       �	�!J^Yc�A�-*

loss%�e=Z�:�       �	��J^Yc�A�-*

loss�߈;MAp(       �	PK^Yc�A�-*

loss���:ݦ�)       �	}�K^Yc�A�-*

loss&��=�plz       �	�L^Yc�A�-*

loss��<�.l�       �	>\M^Yc�A�-*

loss��\=       �	��M^Yc�A�-*

loss��@<lw�A       �	}�N^Yc�A�-*

loss�X`=� �A       �	�%O^Yc�A�-*

lossq�<�s_�       �	o�O^Yc�A�-*

loss��=4�N�       �	 WP^Yc�A�-*

lossI�<��       �	�P^Yc�A�-*

lossцw<V�6�       �	�Q^Yc�A�-*

loss���<��)\       �	�gR^Yc�A�-*

loss�� =L
�       �	�S^Yc�A�-*

loss�J1<p���       �	��S^Yc�A�-*

lossU=0��       �	�FT^Yc�A�-*

lossn��=�܏�       �	��T^Yc�A�-*

loss��<��       �	��U^Yc�A�-*

loss�<x�=       �	0V^Yc�A�-*

loss���=�l�?       �	��V^Yc�A�-*

loss=�<sP&       �	�W^Yc�A�-*

lossq��;P���       �	�X^Yc�A�-*

loss*�:A5�j       �	��X^Yc�A�-*

loss��X;�*<�       �	�HY^Yc�A�-*

loss�
<����       �	-�Y^Yc�A�-*

loss��<G��.       �	+�Z^Yc�A�-*

loss�D=-o�       �	E/[^Yc�A�-*

loss�A=���[       �	;�[^Yc�A�-*

loss^�=�#�       �	GY\^Yc�A�-*

loss<AT=�Q>       �	��\^Yc�A�-*

loss�?E9��H       �	�]^Yc�A�-*

lossB�"<('��       �	o^^Yc�A�-*

loss�V <�Z�       �	�_^Yc�A�-*

loss;�'<�,�       �	N'`^Yc�A�-*

lossg
=Y�       �	��`^Yc�A�-*

loss�D�<|�|O       �	CYa^Yc�A�-*

lossot!<d:�x       �	}b^Yc�A�-*

lossK<��?m       �	��b^Yc�A�-*

loss�-d=�R�       �	�ic^Yc�A�-*

loss�N�<���g       �	eUd^Yc�A�-*

loss	r=o�x       �	N�d^Yc�A�-*

loss�W�<K�Q       �	�e^Yc�A�-*

loss���;o       �	"9f^Yc�A�-*

loss�&R=<ki�       �	��f^Yc�A�-*

loss%��;�~��       �	[ag^Yc�A�-*

loss���<�A:       �	�5h^Yc�A�-*

lossv\<
�{       �	��h^Yc�A�-*

loss�V<m~�Q       �	Fai^Yc�A�-*

loss1e=xSm�       �	��i^Yc�A�-*

lossF�<��       �	}�j^Yc�A�-*

losss��<���       �	wHk^Yc�A�-*

loss�K�;;��\       �	+0l^Yc�A�-*

lossx9<�)�       �	��l^Yc�A�-*

lossN,�<l�$       �	F\m^Yc�A�-*

lossOI�:����       �	in^Yc�A�-*

loss���<�JP9       �	Ǹn^Yc�A�-*

loss�A';2��M       �	�ao^Yc�A�-*

loss�ʁ=�       �	��o^Yc�A�-*

loss�U!<F߰       �	˝p^Yc�A�-*

lossD�9=]ۆ       �	l?q^Yc�A�-*

loss]�<-���       �	k�q^Yc�A�-*

lossN�<��B�       �	nlr^Yc�A�-*

loss�� <(���       �	Bs^Yc�A�-*

loss�<?W�       �	��s^Yc�A�-*

loss�`<�2�       �	�7t^Yc�A�-*

loss^:;�       �	+�t^Yc�A�-*

loss?�<{�       �	B{u^Yc�A�-*

loss$`I<*?n       �	�@v^Yc�A�-*

loss8�.;&>�       �	��v^Yc�A�-*

loss���<�G�t       �	�mw^Yc�A�-*

loss�|<��cm       �	�y^Yc�A�-*

loss�֋;�ER       �	��y^Yc�A�-*

loss�V�;��$W       �	�0z^Yc�A�-*

lossը;�ȋ       �	��z^Yc�A�-*

loss�<w&��       �	[`{^Yc�A�-*

loss<�=���1       �	|^Yc�A�-*

loss��<�z��       �	��|^Yc�A�-*

lossO�=z��       �	�I}^Yc�A�-*

loss㳆;��TZ       �	��}^Yc�A�-*

loss���;_3�       �	�~^Yc�A�-*

lossQ=<*�        �	� ^Yc�A�-*

loss�N�;J��.       �	
�^Yc�A�-*

loss(1=�a�       �	�a�^Yc�A�-*

lossF��<��D\       �	M��^Yc�A�-*

loss�P=��W�       �	�^Yc�A�-*

loss!�V;Z#       �	�^Yc�A�-*

lossڡ<^�e       �	h?�^Yc�A�-*

lossn�=<X�       �	Mۃ^Yc�A�-*

lossp=%f��       �	s�^Yc�A�-*

loss8��<���       �	`#�^Yc�A�-*

loss3c�<`�        �	幅^Yc�A�-*

loss|�=���       �	%Y�^Yc�A�-*

loss���:L�       �	���^Yc�A�-*

loss�W�<���       �	��^Yc�A�-*

loss���<Qw�       �	�D�^Yc�A�-*

loss�G ;�/�i       �	�݈^Yc�A�-*

loss�;:W�c�       �	�v�^Yc�A�-*

loss��=i���       �	��^Yc�A�-*

loss4�<)?=       �	��^Yc�A�-*

loss��<3ɞ       �	�4�^Yc�A�-*

loss��'<�]�3       �	#I�^Yc�A�-*

loss�Km<K�       �	�ߌ^Yc�A�-*

loss	�=Ҿ=�       �	�v�^Yc�A�-*

loss ��:7^s�       �	��^Yc�A�-*

loss��=_^]       �	���^Yc�A�-*

loss���;z{n       �	J�^Yc�A�-*

losso�=S$^u       �	ݏ^Yc�A�-*

loss��=��Y       �	#��^Yc�A�-*

loss���;__�C       �	e�^Yc�A�-*

losso��;��<8       �	���^Yc�A�-*

loss�<���I       �	���^Yc�A�-*

loss�p<�U4       �	�^Yc�A�-*

lossl\=K�N�       �	���^Yc�A�-*

loss#0=���y       �	W_�^Yc�A�-*

loss��=�N�x       �	���^Yc�A�-*

loss�y<���       �	���^Yc�A�-*

loss�e=8|�       �	�X�^Yc�A�-*

lossA��;4�Z       �	F�^Yc�A�-*

loss#B=����       �	k��^Yc�A�-*

loss�v=���H       �	��^Yc�A�-*

losss��;\�E�       �	���^Yc�A�-*

loss�:?=1Rs       �	�Q�^Yc�A�.*

loss�><��?�       �	�^Yc�A�.*

lossь�;��PW       �	���^Yc�A�.*

loss�
<���       �	��^Yc�A�.*

loss��&=����       �	B��^Yc�A�.*

lossa��:q��       �	kH�^Yc�A�.*

loss�0?=���       �	ݜ^Yc�A�.*

loss(�:�GZQ       �	�t�^Yc�A�.*

lossw��<V|Ѯ       �	��^Yc�A�.*

loss{Y�;O�       �	D��^Yc�A�.*

loss��*<"��       �	�E�^Yc�A�.*

lossZ#�;q{��       �	@ޟ^Yc�A�.*

lossnz�<����       �	���^Yc�A�.*

loss�\<�v�       �	�f�^Yc�A�.*

loss��l:�{'
       �	���^Yc�A�.*

loss�V;��`�       �	���^Yc�A�.*

loss��;�[��       �	�5�^Yc�A�.*

lossڲ<�k�       �	���^Yc�A�.*

lossF.<���       �	l��^Yc�A�.*

loss��[;�       �	�.�^Yc�A�.*

lossH̍;�=k&       �	�å^Yc�A�.*

loss�7�<k�#A       �	�Z�^Yc�A�.*

loss8�/<�4       �	��^Yc�A�.*

loss7FF<��E�       �	j��^Yc�A�.*

loss-�<�̌       �	�$�^Yc�A�.*

lossM�;kI��       �	Ը�^Yc�A�.*

loss��;BOwg       �	l�^Yc�A�.*

lossJ��<J�       �	��^Yc�A�.*

lossị;�̋       �	G�^Yc�A�.*

lossזH<xI�}       �	���^Yc�A�.*

loss�� =X�v       �	� �^Yc�A�.*

loss=\�<�*�       �	�Ӭ^Yc�A�.*

loss	n=@j�       �	ji�^Yc�A�.*

loss�Q�<����       �	�^Yc�A�.*

lossz�4=�r��       �	ع�^Yc�A�.*

lossZ;�T�j       �	�N�^Yc�A�.*

loss�z<�^��       �	)�^Yc�A�.*

loss��<:�Ө       �	؀�^Yc�A�.*

loss���<��?       �	q�^Yc�A�.*

loss�%�;N!Q6       �	���^Yc�A�.*

lossO�1<"6F3       �	��^Yc�A�.*

loss�Ë<��^       �	%Y�^Yc�A�.*

loss�m�;sC�       �	��^Yc�A�.*

loss���<�r5       �	��^Yc�A�.*

loss���<5��Q       �	N��^Yc�A�.*

loss
(�:g9�r       �	�<�^Yc�A�.*

loss���9T�6_       �	�ֶ^Yc�A�.*

lossl�<n��5       �	-x�^Yc�A�.*

loss-π:g��       �	��^Yc�A�.*

lossD*A<����       �	~��^Yc�A�.*

loss��=`�f�       �	kI�^Yc�A�.*

loss$9l<�       �	��^Yc�A�.*

lossZ�`;zx��       �	<��^Yc�A�.*

loss�Z�9�3�Q       �	��^Yc�A�.*

loss�9��       �	���^Yc�A�.*

lossO�8����       �	V�^Yc�A�.*

loss�
;�N@       �	�^Yc�A�.*

loss�<G���       �	⑽^Yc�A�.*

loss��@<-�h       �	5)�^Yc�A�.*

loss)½8�8�       �	H¾^Yc�A�.*

loss��<����       �	�p�^Yc�A�.*

loss��>�HL       �	4�^Yc�A�.*

loss��Q:OO�&       �	:��^Yc�A�.*

lossj�c>fU��       �	,D�^Yc�A�.*

loss.��;�CJ       �	M��^Yc�A�.*

lossT��<�Qm�       �	�q�^Yc�A�.*

loss/T�;>�{h       �	M�^Yc�A�.*

loss\"�=�� %       �	&��^Yc�A�.*

loss��B=)V�S       �	9B�^Yc�A�.*

loss�Կ<X�7�       �	��^Yc�A�.*

lossdފ;R��       �	�^Yc�A�.*

lossl�b<�y�       �	D4�^Yc�A�.*

loss�,�<��Q�       �	���^Yc�A�.*

loss��=���       �	E��^Yc�A�.*

loss�]�<�6
=       �	�|�^Yc�A�.*

lossHo;T�ܽ       �	Q�^Yc�A�.*

loss�.�<����       �	A��^Yc�A�.*

loss�7�<?�s�       �	ߌ�^Yc�A�.*

lossL4=�;�e       �	���^Yc�A�.*

loss�CF<��       �	�_�^Yc�A�.*

lossX��<���:       �	���^Yc�A�.*

lossS;h��       �	���^Yc�A�.*

lossj!<t"
C       �	�2�^Yc�A�.*

lossw�<��q�       �	���^Yc�A�.*

loss6��<ψ2V       �	�^�^Yc�A�.*

loss�2�<+*5�       �	p
�^Yc�A�.*

loss�WZ<�ֵ       �	���^Yc�A�.*

loss���<�f�(       �	�<�^Yc�A�.*

loss�Qs<�[l[       �	� �^Yc�A�.*

loss}ʨ:=�u       �	��^Yc�A�.*

loss��=Q� �       �	�+�^Yc�A�.*

loss�	-=V�x�       �	}��^Yc�A�.*

lossG˄<n �       �	2��^Yc�A�.*

loss�5�;�ꢆ       �	�,�^Yc�A�.*

loss��:�P�       �	[��^Yc�A�.*

loss�_�:�T�	       �	�h�^Yc�A�.*

loss��;��GX       �	�c�^Yc�A�.*

loss
�l;@Qx       �	5	�^Yc�A�.*

loss�<��U�       �	���^Yc�A�.*

loss�5�< �!       �	�K�^Yc�A�.*

loss.��<J�X;       �	W��^Yc�A�.*

loss�'d<5 �       �	:��^Yc�A�.*

loss��:����       �	@2�^Yc�A�.*

loss��7:$�o       �	���^Yc�A�.*

loss:�X<�#V       �	�e�^Yc�A�.*

loss�L�< ��       �	:�^Yc�A�.*

loss�'�;|�&�       �	��^Yc�A�.*

loss�^<@��       �	f�^Yc�A�.*

lossMN�<�v�       �	� �^Yc�A�.*

loss���:�b       �	���^Yc�A�.*

loss1�<4���       �	�A�^Yc�A�.*

loss2Sq; @v       �	���^Yc�A�.*

loss$r�:6~�       �	�~�^Yc�A�.*

lossF�;��	�       �		�
_Yc�A�.*

lossf�<c�S3       �	v2_Yc�A�.*

loss�1m<�ù       �	,c_Yc�A�.*

lossKj<��ͅ       �	�'_Yc�A�.*

loss��%=�3_�       �	_�_Yc�A�.*

lossI�;byq       �	l�_Yc�A�.*

loss ��:�-U       �	�b_Yc�A�.*

loss&.�=o��H       �	�_Yc�A�.*

lossc��=�
�G       �	R�_Yc�A�.*

loss�gL=A\2       �	٘_Yc�A�.*

lossTh;��Dn       �	ZE_Yc�A�.*

loss��D=�p��       �	�_Yc�A�.*

loss���:��       �	��_Yc�A�.*

lossl�#<~��0       �	K[_Yc�A�.*

loss��<��+       �	�_Yc�A�.*

lossl$�;et�L       �	֩_Yc�A�/*

lossc�9��8�       �	%�_Yc�A�/*

loss���=�[�       �	��_Yc�A�/*

loss�C?<2�H       �	!�_Yc�A�/*

loss��=��~       �	İ_Yc�A�/*

lossj��<�i�       �	�_Yc�A�/*

loss�=D�       �	�4_Yc�A�/*

loss���:6z��       �	=�_Yc�A�/*

loss���<�^�       �	��_Yc�A�/*

loss�V�;h�ڛ       �	CX_Yc�A�/*

loss�=SU�r       �	�L_Yc�A�/*

loss��=�w̛       �	�?_Yc�A�/*

loss#K&<ski       �	g} _Yc�A�/*

loss��<Fd&~       �	�(!_Yc�A�/*

loss��6<��n       �	!�!_Yc�A�/*

loss�J?<�9<       �	R�"_Yc�A�/*

loss�;K4ƺ       �	�Z#_Yc�A�/*

loss�]_<�՗9       �	:$_Yc�A�/*

loss�p&=�+6|       �	{�$_Yc�A�/*

loss$Ҽ<�_/�       �	;R%_Yc�A�/*

loss�w�=7a�J       �	��%_Yc�A�/*

lossS/;ͯZ1       �	��&_Yc�A�/*

loss�"=D�o       �	|''_Yc�A�/*

loss!2�=5p��       �	��'_Yc�A�/*

loss##�=���       �	�P(_Yc�A�/*

loss]0�;r6W�       �	��(_Yc�A�/*

lossr�<B�U,       �	�)_Yc�A�/*

loss�K�;"�qT       �	UK*_Yc�A�/*

loss�:�;!eR�       �	��*_Yc�A�/*

loss�g<G�
8       �	[y+_Yc�A�/*

lossH�<ۼf       �	�U,_Yc�A�/*

loss��;C���       �	&�,_Yc�A�/*

loss�y<��{       �	5�-_Yc�A�/*

lossU�;Ҥg�       �	zS._Yc�A�/*

lossz�h:/�       �	��._Yc�A�/*

lossS7:�ˇ       �	w�/_Yc�A�/*

loss%�=�)q�       �	2!0_Yc�A�/*

loss1)�<F�j�       �	�0_Yc�A�/*

loss�z=ڲ��       �	��1_Yc�A�/*

lossOț<czn       �	[A2_Yc�A�/*

lossl;:�B(       �	#�2_Yc�A�/*

loss��<�o��       �	�s3_Yc�A�/*

lossr�:ZG�9       �	A4_Yc�A�/*

loss9�= U�g       �	�4_Yc�A�/*

lossd�C<7f��       �	�?5_Yc�A�/*

loss�y�=����       �	-�5_Yc�A�/*

lossm��<�E�       �	�p6_Yc�A�/*

loss��<Î�O       �	y7_Yc�A�/*

loss�w�;�
       �	B�7_Yc�A�/*

lossK�<���_       �	�^8_Yc�A�/*

loss�E�:���       �	��8_Yc�A�/*

loss]��<�qQi       �	��9_Yc�A�/*

lossd�<m�N�       �	�-:_Yc�A�/*

loss�lk=�{�Z       �	��:_Yc�A�/*

loss��;qA�       �	%x;_Yc�A�/*

lossF�E;���       �	l
<_Yc�A�/*

loss�d<&�I�       �	��<_Yc�A�/*

loss�<��
        �	�2=_Yc�A�/*

loss�/=,e�G       �	7�=_Yc�A�/*

loss���<����       �	�W>_Yc�A�/*

loss d�<��yt       �	��>_Yc�A�/*

loss}H�:�j,       �	m�?_Yc�A�/*

loss��J<���       �	(@_Yc�A�/*

loss�)�;��       �	�A_Yc�A�/*

loss���;���       �	̷A_Yc�A�/*

loss䲷<����       �	��B_Yc�A�/*

loss?&�<�6+�       �	�(C_Yc�A�/*

loss�B=es,�       �	��C_Yc�A�/*

loss =6%�e       �	�hD_Yc�A�/*

loss��p;�$%�       �	AFE_Yc�A�/*

loss���;���h       �	e�E_Yc�A�/*

loss�<8�(�       �	�zF_Yc�A�/*

loss�s�;��(       �	�G_Yc�A�/*

loss��z<�~h       �	ԹG_Yc�A�/*

loss�c�=��a�       �	MH_Yc�A�/*

loss�6=�]c       �	�H_Yc�A�/*

lossGo=dj�       �	+�I_Yc�A�/*

loss:{r=����       �	.XJ_Yc�A�/*

loss%{<�KVQ       �	�J_Yc�A�/*

loss �`=���       �	��K_Yc�A�/*

loss}�5;U��J       �	�oL_Yc�A�/*

loss�?`=�n�1       �	{�M_Yc�A�/*

loss�>�<I&M       �	*N_Yc�A�/*

loss�f�;�י       �	��N_Yc�A�/*

loss���<�S�       �	"PO_Yc�A�/*

lossC�D;���       �	x%P_Yc�A�/*

loss��\;y׋�       �	��P_Yc�A�/*

loss��*=T��,       �	YQ_Yc�A�/*

loss��<=���       �	��Q_Yc�A�/*

loss�K=�Q�]       �	ҏR_Yc�A�/*

loss�v�=ꢢw       �	�"S_Yc�A�/*

lossw��<�*�       �	��S_Yc�A�/*

loss�ۺ<ֹ@Z       �	NT_Yc�A�/*

loss���=��5I       �	2�T_Yc�A�/*

loss�N�<��K�       �	�zU_Yc�A�/*

loss͛N=�&
�       �	@V_Yc�A�/*

loss4�;���p       �	V�V_Yc�A�/*

loss2�L<����       �	F]W_Yc�A�/*

loss�Y�:V�u�       �	�X_Yc�A�/*

loss���;}�E�       �	ϡX_Yc�A�/*

loss �
;���       �	FCY_Yc�A�/*

loss?<jf�       �	"�Y_Yc�A�/*

loss&�=g	&       �	�Z_Yc�A�/*

loss��*<^eK=       �	�$[_Yc�A�/*

loss�<�^�       �	��[_Yc�A�/*

loss���<Z�eG       �	�m\_Yc�A�/*

loss%<�~�       �	�]_Yc�A�/*

loss��<4@��       �	g�]_Yc�A�/*

loss��<,�X�       �	DP^_Yc�A�/*

loss�d,;.��x       �	�^_Yc�A�/*

lossԮ<�[L       �	��__Yc�A�/*

lossvt�<��t�       �	�.`_Yc�A�/*

loss�H8<��       �	��`_Yc�A�/*

loss�f�<�x5       �	3ja_Yc�A�/*

losss�1<BX��       �	�b_Yc�A�/*

loss/�z<�
a       �	��b_Yc�A�/*

loss1as<[�7�       �	�cc_Yc�A�/*

lossvo�;Jpy*       �	�d_Yc�A�/*

loss�M<{)y       �	V�d_Yc�A�/*

loss�s;��ҥ       �	�f_Yc�A�/*

lossl��:��c�       �	��f_Yc�A�/*

loss|�$;���%       �	�>g_Yc�A�/*

loss���<���       �	��g_Yc�A�/*

loss�*;~�=�       �	�h_Yc�A�/*

lossq%H;UL       �	.i_Yc�A�/*

loss���;|��#       �	r�i_Yc�A�/*

loss��<$X}�       �	�Wj_Yc�A�/*

loss}�H<M�W       �	��j_Yc�A�/*

loss(TB<]�H       �	ρk_Yc�A�0*

loss�w�<��=       �	,)l_Yc�A�0*

loss;��<	�$:       �	q�l_Yc�A�0*

loss���<���       �	=�m_Yc�A�0*

loss��U;��       �	e6n_Yc�A�0*

loss�x�;BI�       �	%�n_Yc�A�0*

loss��;,���       �	�fo_Yc�A�0*

lossu�;gܲ�       �	� p_Yc�A�0*

loss�M}=c"�       �	$�p_Yc�A�0*

loss2;=�ٖ       �	L7q_Yc�A�0*

lossʂ�;
L�'       �	}�q_Yc�A�0*

losst*�<�ӝ3       �	cr_Yc�A�0*

lossQn:�ѪA       �	�s_Yc�A�0*

loss�Cc;_�ȩ       �	E�s_Yc�A�0*

loss~ś<�'�l       �	h\t_Yc�A�0*

loss�= ��       �	5�t_Yc�A�0*

lossv*�;v�JU       �	#�u_Yc�A�0*

loss$��<~�k�       �	�v_Yc�A�0*

loss$�;�	>�       �	մv_Yc�A�0*

lossK<xٯd       �	Kw_Yc�A�0*

loss��f9d|n5       �	��w_Yc�A�0*

loss��;���       �	�rx_Yc�A�0*

lossCٔ;�P?       �	1y_Yc�A�0*

loss��:�Z�       �	��y_Yc�A�0*

loss#�3<�ڕ>       �	�/z_Yc�A�0*

lossn0�<�U�       �	a�z_Yc�A�0*

loss g�:z�C       �	�j{_Yc�A�0*

loss&��<P�y�       �	�|_Yc�A�0*

loss� :��       �	��|_Yc�A�0*

lossk�:�\R       �	�E}_Yc�A�0*

loss���;���       �	��}_Yc�A�0*

loss�<	9�5]       �	��~_Yc�A�0*

loss��;����       �	�%_Yc�A�0*

loss�kW;���j       �	��_Yc�A�0*

loss2�8=�gu*       �	mU�_Yc�A�0*

loss/�L<B�*)       �	��_Yc�A�0*

loss�1�;��Dd       �	؁�_Yc�A�0*

lossώ�9�A       �	�/�_Yc�A�0*

loss�?�<�r%�       �	�Ȃ_Yc�A�0*

loss���;��1�       �	�`�_Yc�A�0*

loss�Ø:5��       �	�_Yc�A�0*

loss{�;3��       �	���_Yc�A�0*

lossT��;���       �	,c�_Yc�A�0*

loss�2<�B       �	��_Yc�A�0*

lossUf=�~B       �	���_Yc�A�0*

loss�Fs<�G�       �	\�_Yc�A�0*

lossP��=B��       �	�_Yc�A�0*

loss<��<�$�       �	iR�_Yc�A�0*

loss=��;QU#       �	]��_Yc�A�0*

lossdH�:��>�       �	P��_Yc�A�0*

loss�30=1�g�       �	kH�_Yc�A�0*

loss�-
=@�A�       �	��_Yc�A�0*

loss/,=�P%5       �	敌_Yc�A�0*

loss2j�:ٻ��       �	�5�_Yc�A�0*

loss���;��jB       �	* �_Yc�A�0*

loss�;O*	�       �	|��_Yc�A�0*

loss�+�<&�       �	�7�_Yc�A�0*

lossf��;�W�       �	fݏ_Yc�A�0*

losscHv;���+       �	s��_Yc�A�0*

loss�8�;�rw       �	�%�_Yc�A�0*

loss���=u��a       �	쿑_Yc�A�0*

loss��n;FZ_       �	�]�_Yc�A�0*

lossQ�=A�m       �	���_Yc�A�0*

loss$=��e       �	猓_Yc�A�0*

losszp�<"��       �	�7�_Yc�A�0*

loss�
<���       �	�ϔ_Yc�A�0*

loss�/P;vV1       �	&��_Yc�A�0*

loss%�3=Ve�(       �	O#�_Yc�A�0*

loss�;����       �	g��_Yc�A�0*

loss�2�<�1       �	'O�_Yc�A�0*

loss�qC<��4       �	"�_Yc�A�0*

lossю�=j��I       �	]��_Yc�A�0*

loss=�=�Yz<       �	m �_Yc�A�0*

loss�K=R���       �	���_Yc�A�0*

lossQ�z=C-       �	=F�_Yc�A�0*

loss���<� ԋ       �	'ݚ_Yc�A�0*

loss�b<rv(�       �	�t�_Yc�A�0*

loss0z�<�O�       �	�
�_Yc�A�0*

lossJ�6;DQPa       �	���_Yc�A�0*

lossmj�;s�Ґ       �		3�_Yc�A�0*

loss���<�eڶ       �	.ʝ_Yc�A�0*

loss�;�U�       �	�r�_Yc�A�0*

lossJ��<��es       �	d�_Yc�A�0*

loss{0�;Z�F       �	靟_Yc�A�0*

loss2"=�"��       �	�5�_Yc�A�0*

loss�c;?�`       �	4�_Yc�A�0*

loss#
�;�)F       �	���_Yc�A�0*

lossa�<w�j6       �	�#�_Yc�A�0*

lossĿG<�c{       �	V��_Yc�A�0*

loss!�]<�e�       �	vQ�_Yc�A�0*

loss��;Ѽb�       �	%�_Yc�A�0*

losshoY<�z?�       �	�ʤ_Yc�A�0*

loss�$];*��       �	8إ_Yc�A�0*

lossvW	=��Z�       �	�z�_Yc�A�0*

loss-�;���       �	�Q�_Yc�A�0*

loss}�<,~       �	m�_Yc�A�0*

lossdqY<s�j/       �	X�_Yc�A�0*

loss;�:x���       �	A��_Yc�A�0*

lossmZ	=r�F�       �	:[�_Yc�A�0*

lossLZ�<�b:�       �	��_Yc�A�0*

loss�Y�<G��>       �	���_Yc�A�0*

loss��=c�a�       �	�*�_Yc�A�0*

loss>�;�]�p       �	�ʬ_Yc�A�0*

loss��w<��       �	�ʭ_Yc�A�0*

loss1��<��}�       �	Di�_Yc�A�0*

loss6�w<B�0       �	��_Yc�A�0*

lossT�@<P0��       �	���_Yc�A�0*

loss8�=�bڔ       �	�A�_Yc�A�0*

loss�m?=`q{�       �	<ٰ_Yc�A�0*

loss)�t<X��       �	Cu�_Yc�A�0*

loss�H�;@$a       �	�_Yc�A�0*

loss���:ܷ�f       �	&��_Yc�A�0*

loss/<��A�       �	:=�_Yc�A�0*

loss�;���:       �	�Գ_Yc�A�0*

loss���:_�a�       �	0f�_Yc�A�0*

loss,\L<ef-       �	���_Yc�A�0*

lossZ-=�`��       �	���_Yc�A�0*

loss�c�=*Rq       �	�(�_Yc�A�0*

loss.>�;'ZP�       �	��_Yc�A�0*

loss�l=�lH       �	��_Yc�A�0*

losse)=5��d       �	B"�_Yc�A�0*

loss1��<�*
�       �	���_Yc�A�0*

lossxr?;��F�       �	0d�_Yc�A�0*

lossŒ�=~���       �	]S�_Yc�A�0*

lossV�;�;I%       �	��_Yc�A�0*

lossN�<�~       �	b��_Yc�A�0*

loss�a=�4N       �	j�_Yc�A�0*

losskO"<�\C       �	���_Yc�A�0*

loss�?�<NP       �	�F�_Yc�A�1*

lossZ*�<��z�       �	��_Yc�A�1*

loss��=o��       �	~p�_Yc�A�1*

loss��<)���       �	��_Yc�A�1*

lossOwS<��@�       �	/��_Yc�A�1*

loss$~;8�-M       �	�R�_Yc�A�1*

lossJ^9<X�l�       �	���_Yc�A�1*

loss��H=�       �	���_Yc�A�1*

lossZy�<�2�       �	��_Yc�A�1*

loss2�F;��,�       �	��_Yc�A�1*

loss�d<)[��       �	R��_Yc�A�1*

loss�sQ<v��       �	�;�_Yc�A�1*

loss�:<?��:       �	���_Yc�A�1*

lossX��<q>��       �	�b�_Yc�A�1*

loss��L=If+�       �	A��_Yc�A�1*

loss};W=�R0       �	��_Yc�A�1*

loss�?<��.�       �	-�_Yc�A�1*

lossj�g<=�       �	���_Yc�A�1*

loss�v=��Y�       �	��_Yc�A�1*

loss8J�<i �       �	rP�_Yc�A�1*

loss���<uɬL       �	��_Yc�A�1*

loss<o}<�\�E       �	���_Yc�A�1*

loss(��;����       �	��_Yc�A�1*

loss��:3<D       �	A��_Yc�A�1*

lossMj�<�M       �	iS�_Yc�A�1*

loss��<Y�O       �	���_Yc�A�1*

loss}�=yb�       �	���_Yc�A�1*

loss��<@$       �	�`�_Yc�A�1*

loss�%u<���       �	���_Yc�A�1*

lossa�<�6��       �	\��_Yc�A�1*

loss
&=kH��       �	�%�_Yc�A�1*

loss\)=#y�W       �	���_Yc�A�1*

loss�~�<mޠ       �	�c�_Yc�A�1*

loss�O�;��_       �	D��_Yc�A�1*

loss�g�=[��l       �	��_Yc�A�1*

lossA<�h�       �	�5�_Yc�A�1*

loss�I=Z��       �	���_Yc�A�1*

loss,F�<�x       �	�t�_Yc�A�1*

loss�<���       �	��_Yc�A�1*

loss�K�;��7]       �	ݲ�_Yc�A�1*

loss��<�9�\       �	��_Yc�A�1*

loss��<I��       �	�]�_Yc�A�1*

loss�� ;X��0       �	��_Yc�A�1*

loss��:y_-�       �	U��_Yc�A�1*

lossO�;5G#       �	�?�_Yc�A�1*

lossJ�/=�\W       �	���_Yc�A�1*

loss(q;�=�       �	�}�_Yc�A�1*

loss�@�<���C       �	u�_Yc�A�1*

loss�UQ<���       �	��_Yc�A�1*

lossH;2[n+       �	Tq�_Yc�A�1*

lossŒ@={`d       �	�	�_Yc�A�1*

loss��< �s       �	ʣ�_Yc�A�1*

lossF ;��R�       �	
K�_Yc�A�1*

loss��=��''       �	u��_Yc�A�1*

lossQ�l;��2       �	I��_Yc�A�1*

lossV�<��       �	�_Yc�A�1*

lossd��;�Md       �	l��_Yc�A�1*

loss�q<�]vG       �	�K�_Yc�A�1*

loss]�	=%56�       �	���_Yc�A�1*

loss��[=W��       �	ׄ�_Yc�A�1*

lossPG;?���       �	. �_Yc�A�1*

loss� <�x0�       �	��_Yc�A�1*

loss6'<8�a�       �	�]�_Yc�A�1*

lossR��<���       �	Ɔ�_Yc�A�1*

lossvN�:�Kn�       �	`�_Yc�A�1*

loss���<�!X�       �	t��_Yc�A�1*

loss1a�;5�5�       �	,H�_Yc�A�1*

loss���;T       �	/��_Yc�A�1*

lossj}�9�]f�       �	�t�_Yc�A�1*

loss�G8;:T��       �	=�_Yc�A�1*

loss�pt<%�K�       �	0��_Yc�A�1*

loss�.�=Q�a�       �	_C�_Yc�A�1*

loss�?=��1       �	u��_Yc�A�1*

lossf<�=G<�       �	���_Yc�A�1*

loss�ɂ<7��       �	D5�_Yc�A�1*

loss���:�E̙       �	;��_Yc�A�1*

loss�ǳ<D�       �	���_Yc�A�1*

loss��4<	X�	       �	�8�_Yc�A�1*

loss��<���       �	��_Yc�A�1*

lossW�;��       �	'��_Yc�A�1*

loss��o;_�i       �	�,�_Yc�A�1*

loss�-�=m��       �	���_Yc�A�1*

loss�T�<��       �	���_Yc�A�1*

loss�?=\l(�       �	~p�_Yc�A�1*

loss,W�<��fw       �	/�_Yc�A�1*

loss�~ <p��3       �	K�_Yc�A�1*

loss]=��       �	���_Yc�A�1*

loss&V�<��l        �	^��_Yc�A�1*

lossA�;U��4       �	��_Yc�A�1*

loss�̕<�!d�       �	ձ�_Yc�A�1*

lossW�m<!8^�       �	YR�_Yc�A�1*

loss�q�<�C       �	,��_Yc�A�1*

loss��$=��       �	���_Yc�A�1*

lossr��<I��j       �	�2�_Yc�A�1*

loss�̃<�~{       �	A��_Yc�A�1*

loss�}P<~�8X       �	2s�_Yc�A�1*

loss�-n<��       �	 �_Yc�A�1*

loss,Z�;�|�       �	���_Yc�A�1*

loss�<�"��       �	�f�_Yc�A�1*

lossc�=<��B�       �	��_Yc�A�1*

loss�P: 4��       �	ǜ�_Yc�A�1*

loss�'m<��:�       �	5�_Yc�A�1*

lossM�3;$�r�       �	���_Yc�A�1*

loss�B=�'5i       �	b�_Yc�A�1*

loss���=��       �	r��_Yc�A�1*

loss�t=t~L�       �	p� `Yc�A�1*

loss=�9<��       �	n2`Yc�A�1*

loss��f=¯�S       �	y�`Yc�A�1*

lossmO8<˵+       �	�h`Yc�A�1*

lossYl<�1x�       �	� `Yc�A�1*

loss�C�<;<L�       �	��`Yc�A�1*

loss�zg=�*�       �	�`Yc�A�1*

loss��<h|�       �	�>`Yc�A�1*

loss;�+;pm�       �	��`Yc�A�1*

lossx�S=|��       �	��`Yc�A�1*

loss��<l캱       �	-`Yc�A�1*

lossh�O=�vES       �	D�`Yc�A�1*

loss��l=(� \       �	�a`Yc�A�1*

loss���;f�(       �	��`Yc�A�1*

lossx-�;�@       �	��	`Yc�A�1*

loss�o^=hv       �	�[
`Yc�A�1*

loss��<l�+0       �	��
`Yc�A�1*

loss���<�!�7       �	~�`Yc�A�1*

loss�l=���       �	%`Yc�A�1*

loss�7<�=3~       �	��`Yc�A�1*

loss;���       �	l�`Yc�A�1*

lossO]T;��m       �	@`Yc�A�1*

lossE�<�~�       �	�`Yc�A�1*

lossxn�<b|��       �	�``Yc�A�2*

loss�MF<�|�b       �	��`Yc�A�2*

loss�f=���       �	Eg`Yc�A�2*

loss$�a<:�W"       �	�`Yc�A�2*

loss@)�<�h2�       �	��`Yc�A�2*

loss[�=�um�       �	�=`Yc�A�2*

losso�;��A       �	��`Yc�A�2*

loss��<�~I<       �	�`Yc�A�2*

lossR��<��n       �	�`Yc�A�2*

loss-��<KU*�       �	�`Yc�A�2*

lossX�<�je       �	__`Yc�A�2*

loss�L<�$s       �	��`Yc�A�2*

loss )=��Ŵ       �	Z�`Yc�A�2*

loss�p�<x�2       �	)u`Yc�A�2*

loss���<��;       �	`Yc�A�2*

lossb=7�:       �	8�`Yc�A�2*

loss���<�K��       �	cG`Yc�A�2*

loss�� ;�G0       �	��`Yc�A�2*

loss;O;8�9Y       �	Oy`Yc�A�2*

loss+��<�CQ	       �	�O`Yc�A�2*

loss�<�f~       �	��`Yc�A�2*

loss�,=�A�       �	��`Yc�A�2*

loss��;��;E       �	bI`Yc�A�2*

lossܫ�;��Jg       �	��`Yc�A�2*

loss:rc=Aa��       �	ޓ `Yc�A�2*

loss\9=7ћ'       �	9*!`Yc�A�2*

loss��M=\{�       �	2�!`Yc�A�2*

loss6�)<a��       �	3k"`Yc�A�2*

loss�Pk=tG='       �	 #`Yc�A�2*

loss�Z ;Qt��       �	s�#`Yc�A�2*

lossE�
=��/t       �	>$`Yc�A�2*

loss4�$<A���       �	��$`Yc�A�2*

loss���;����       �	ݚ%`Yc�A�2*

loss�Y�;fX�       �	w�&`Yc�A�2*

loss��<C��c       �	qU'`Yc�A�2*

loss���<ۻ��       �	��'`Yc�A�2*

loss{mm<�U       �	��(`Yc�A�2*

loss�B<6j(r       �	��)`Yc�A�2*

loss,�;���       �	�h*`Yc�A�2*

loss� e:Y�       �	�+`Yc�A�2*

lossX2;���       �	@,`Yc�A�2*

loss��p=gD�{       �	�-`Yc�A�2*

loss&s�=�ҧ�       �	�-`Yc�A�2*

lossG=Ňǎ       �	��.`Yc�A�2*

lossF�=����       �	A�/`Yc�A�2*

lossӹ�;���	       �	�%0`Yc�A�2*

loss�r�<�       �	s�0`Yc�A�2*

lossd�K<Vt��       �	�1`Yc�A�2*

loss���<���L       �	T2`Yc�A�2*

loss���;E�X	       �	M3`Yc�A�2*

loss��^=�!��       �	g�3`Yc�A�2*

loss��>=LBqM       �	�T4`Yc�A�2*

lossJz�;F       �	�4`Yc�A�2*

loss8z�<��p`       �	��5`Yc�A�2*

loss��)<�M��       �	UK6`Yc�A�2*

loss���<+���       �	��6`Yc�A�2*

loss@/I=�H�F       �	�7`Yc�A�2*

loss\��=�`��       �	�68`Yc�A�2*

loss;ؑ<�ŧ       �	��8`Yc�A�2*

loss�[<J�b�       �	�v9`Yc�A�2*

loss�H�;�q|�       �	":`Yc�A�2*

lossȹ^:L�       �	��:`Yc�A�2*

loss��;s9��       �	�_;`Yc�A�2*

loss�Ձ:k�`�       �	?<`Yc�A�2*

loss\��=Ķ#       �	��<`Yc�A�2*

loss�T;\UhA       �	2:=`Yc�A�2*

loss�M�;�ndb       �	/�=`Yc�A�2*

loss��<�IR7       �	��>`Yc�A�2*

lossuY;s���       �	LS?`Yc�A�2*

loss�\h<#�+h       �	s�?`Yc�A�2*

loss�/:���k       �	�@`Yc�A�2*

lossŉ�=�Ug\       �	�aA`Yc�A�2*

lossqc�<nq��       �	�B`Yc�A�2*

loss(=>��N       �	i�B`Yc�A�2*

losss��<�8��       �	�CC`Yc�A�2*

loss��<M\�1       �	~�C`Yc�A�2*

lossA�<P�B       �	�uD`Yc�A�2*

lossj��;n49       �	� E`Yc�A�2*

loss��<JR,n       �	��E`Yc�A�2*

loss���:9C       �	?nF`Yc�A�2*

lossx�A=���\       �	�G`Yc�A�2*

loss=C���       �	:�G`Yc�A�2*

lossd��<�ΐ       �	TH`Yc�A�2*

lossH�;ܥ�<       �	�3I`Yc�A�2*

loss�)&<iz�=       �	�I`Yc�A�2*

lossѰ<T)�I       �	o~J`Yc�A�2*

loss��;_�4E       �	� K`Yc�A�2*

loss�"w;dy       �	��K`Yc�A�2*

loss4B=�'b�       �	��L`Yc�A�2*

loss��i;�'BJ       �	��M`Yc�A�2*

loss���<�_       �	��N`Yc�A�2*

lossؗ�<�E�a       �	Z�O`Yc�A�2*

loss֢�;��%�       �	8NP`Yc�A�2*

loss��\=���       �	��P`Yc�A�2*

loss�/�<���       �	��Q`Yc�A�2*

loss1�t;[�߉       �	�>R`Yc�A�2*

losse@;ہ$       �	��R`Yc�A�2*

loss�F=M�V       �	��S`Yc�A�2*

losso��<���"       �	7TT`Yc�A�2*

loss�!�;��T       �	��T`Yc�A�2*

lossn��=
��P       �	�V`Yc�A�2*

loss�� <�Ր�       �	j�V`Yc�A�2*

loss�T<bXQ       �	�>W`Yc�A�2*

loss�8;C��       �	��W`Yc�A�2*

loss�E�;��"       �	o�X`Yc�A�2*

loss�!=�r��       �	xcY`Yc�A�2*

lossI��;ȕ��       �	G�Y`Yc�A�2*

lossce<&<,V       �	��Z`Yc�A�2*

loss�$�;tp4       �	�{[`Yc�A�2*

losssr=<�]X�       �	�\`Yc�A�2*

loss}އ<>tK�       �	�\`Yc�A�2*

loss36&<�J��       �	n]`Yc�A�2*

loss{jA<�W�       �	r^`Yc�A�2*

loss�@<E?�*       �	��^`Yc�A�2*

loss�5f<0];6       �	�V_`Yc�A�2*

loss�D�;��@�       �	��_`Yc�A�2*

loss�;v.�*       �	��``Yc�A�2*

loss!�T<&Wt�       �	Aa`Yc�A�2*

loss��=��D       �	�a`Yc�A�2*

loss�H<�K#       �	�b`Yc�A�2*

loss���<p%Y�       �	�8c`Yc�A�2*

loss�8:�m�       �	�c`Yc�A�2*

loss��X;]y�G       �	I�d`Yc�A�2*

loss�=;�V��       �	��e`Yc�A�2*

loss��d<��W�       �	Wf`Yc�A�2*

lossS&F<��0)       �	xg`Yc�A�2*

loss Q�;�`T       �	��h`Yc�A�2*

loss�T�;3��       �	�,i`Yc�A�2*

loss<�8 E       �	�i`Yc�A�3*

loss n�<�%�        �	_zj`Yc�A�3*

loss2˖<�5�       �	�k`Yc�A�3*

loss1�;2C>�       �	˽k`Yc�A�3*

loss�!�<$���       �	Aal`Yc�A�3*

loss1 ;���       �	i m`Yc�A�3*

loss �d<��>�       �	��m`Yc�A�3*

lossw�9.�       �	Min`Yc�A�3*

loss��9���h       �	�o`Yc�A�3*

loss3��;��t       �	��o`Yc�A�3*

loss�w<l�c�       �	�p`Yc�A�3*

loss�=�ȀR       �	�_q`Yc�A�3*

loss��<���       �	�r`Yc�A�3*

losso֝:-�nL       �	ßr`Yc�A�3*

loss	91#�       �	Us`Yc�A�3*

loss��a8�<��       �	oHt`Yc�A�3*

losstm�;�~��       �	%�t`Yc�A�3*

loss�A�;f�       �	��u`Yc�A�3*

lossC4{;�?I       �	Fv`Yc�A�3*

loss�P�9�yk?       �	��v`Yc�A�3*

loss��r:�m       �	�w`Yc�A�3*

loss���=l7�       �	->x`Yc�A�3*

lossh�<���d       �	�x`Yc�A�3*

loss�)>8�       �	Ւy`Yc�A�3*

lossF&�=VXB@       �	�z`Yc�A�3*

loss��<v�L:       �	9%{`Yc�A�3*

loss$�;��X�       �	_�{`Yc�A�3*

losszgJ;�[t�       �	�n|`Yc�A�3*

loss3;�@nh       �	�}`Yc�A�3*

loss>+=�       �	��}`Yc�A�3*

lossh�Q=�ƨ�       �	Yk~`Yc�A�3*

loss!Y=L�.�       �	�`Yc�A�3*

loss:v8<�V<�       �	�`Yc�A�3*

loss�� <����       �	�O�`Yc�A�3*

loss��f=�#�       �	���`Yc�A�3*

lossO�o<���?       �	Ō�`Yc�A�3*

losswn<A���       �	�,�`Yc�A�3*

loss!F�;bc;�       �	�˂`Yc�A�3*

loss��<���       �	�h�`Yc�A�3*

loss*�<|~�?       �	 �`Yc�A�3*

loss���;�z^�       �	e`Yc�A�3*

loss���;�'�       �	�h�`Yc�A�3*

loss�H=)�       �	!�`Yc�A�3*

loss8��;~�%�       �	2Ɔ`Yc�A�3*

loss���<M��       �	�q�`Yc�A�3*

loss&�N;���       �	��`Yc�A�3*

loss�&�:dH=       �	t��`Yc�A�3*

loss(iT:�~h7       �	X�`Yc�A�3*

loss�ڭ=��;       �	.7�`Yc�A�3*

lossB�;"1�       �	��`Yc�A�3*

loss�J=\�       �	u��`Yc�A�3*

loss�v,=��       �	2<�`Yc�A�3*

loss!��:��g       �	�+�`Yc�A�3*

loss��<佐�       �	�ڍ`Yc�A�3*

loss��_<\Ux�       �	���`Yc�A�3*

loss�k];�q p       �	@��`Yc�A�3*

loss�� <YϹ       �	b��`Yc�A�3*

loss�v<�̩�       �	6"�`Yc�A�3*

loss��	<�]O�       �	q̑`Yc�A�3*

lossZa<HDH�       �	�j�`Yc�A�3*

loss�N<��t�       �	L�`Yc�A�3*

loss1�&<��H�       �	p��`Yc�A�3*

loss/�:�C�       �	�]�`Yc�A�3*

loss�ަ:˟Xa       �	��`Yc�A�3*

loss���<*hT�       �	<��`Yc�A�3*

lossj��<�<�       �	���`Yc�A�3*

loss!�c;{�0s       �	wM�`Yc�A�3*

loss��	=BI�8       �	��`Yc�A�3*

lossC�S=ti��       �	휘`Yc�A�3*

loss��;Z�       �	�@�`Yc�A�3*

loss��d<�p��       �	m��`Yc�A�3*

lossT�E<2 ��       �	ؚ�`Yc�A�3*

loss���:��0{       �	)@�`Yc�A�3*

loss�@�;��;       �	Է�`Yc�A�3*

loss	�>=.�V       �	�ؽ`Yc�A�3*

loss�4<�K��       �	D��`Yc�A�3*

loss�o?;�R�'       �	'��`Yc�A�3*

loss��S<̢�+       �	8��`Yc�A�3*

loss�up<��       �	���`Yc�A�3*

loss�EV<�N��       �	�m�`Yc�A�3*

lossx��<lNh�       �	��`Yc�A�3*

loss �=t*":       �	D��`Yc�A�3*

loss_B;���       �	e��`Yc�A�3*

loss!k�;��       �	�$�`Yc�A�3*

loss2r�;��=       �	��`Yc�A�3*

lossV"<�_'       �	�k�`Yc�A�3*

loss�~6;\�'�       �	��`Yc�A�3*

loss�C�=��[�       �	���`Yc�A�3*

lossܜs</Ŏ       �	\�`Yc�A�3*

loss�T�:M��       �	���`Yc�A�3*

lossP��;)n       �	ӥ�`Yc�A�3*

loss�;:iC%       �	�E�`Yc�A�3*

lossj=Y<�Zζ       �	���`Yc�A�3*

lossI�<|>�       �	|��`Yc�A�3*

loss�)�<�.��       �	�4�`Yc�A�3*

loss
y�:�I       �	���`Yc�A�3*

loss,�"=Xx\d       �	_�`Yc�A�3*

loss&�!:0h�R       �	t(�`Yc�A�3*

loss��C=�E�       �	s��`Yc�A�3*

loss���<Hy_       �	"O�`Yc�A�3*

lossat�:I\�Z       �	���`Yc�A�3*

loss�
V<b|��       �	ۇ�`Yc�A�3*

losst3�;���Q       �	O�`Yc�A�3*

loss��R;��8�       �	��`Yc�A�3*

loss���;�݋�       �	YL�`Yc�A�3*

lossR�J<�ܕ�       �	���`Yc�A�3*

loss�G�<���       �	7��`Yc�A�3*

loss�`�:���       �	=+�`Yc�A�3*

loss�M�;j��       �	Q��`Yc�A�3*

loss��d;ښ�       �	O�`Yc�A�3*

loss��;�/�       �	:��`Yc�A�3*

loss��<�O��       �	�x�`Yc�A�3*

lossk=Ԟ�       �	�`Yc�A�3*

loss���;��<       �	��`Yc�A�3*

lossߚ3:r+|�       �	�B�`Yc�A�3*

loss��t:KIH�       �	���`Yc�A�3*

loss-
H<���J       �	f��`Yc�A�3*

lossc%�;���C       �	��`Yc�A�3*

loss�BD;KzK�       �	:��`Yc�A�3*

loss�.=y��       �	�F�`Yc�A�3*

loss��<)!Q       �	���`Yc�A�3*

lossSܽ:��u       �	�u�`Yc�A�3*

loss���:��5�       �	��`Yc�A�3*

loss�G;�"'�       �	ݴ�`Yc�A�3*

lossH�/<cr6C       �	�U�`Yc�A�3*

loss`hI:-[�x       �	���`Yc�A�3*

loss�!=t�a       �	T��`Yc�A�3*

loss�*:=�R;       �	�*�`Yc�A�3*

loss��&97��       �	���`Yc�A�4*

loss��:Oz��       �	�e�`Yc�A�4*

lossR�<��C       �	-
�`Yc�A�4*

loss�ʋ:H��       �	���`Yc�A�4*

loss�<GAg�       �	�M�`Yc�A�4*

loss�l�=q�.p       �	���`Yc�A�4*

loss��H=���       �	��`Yc�A�4*

loss��;�6��       �	�!�`Yc�A�4*

lossq�b<4f��       �	1��`Yc�A�4*

loss�M	;�7E$       �	
��`Yc�A�4*

lossC�;:f���       �	�g�`Yc�A�4*

loss���<��*5       �	i�`Yc�A�4*

loss�MO:�2�K       �	���`Yc�A�4*

loss2��=���       �	]P�`Yc�A�4*

loss�;~=,��D       �	�`Yc�A�4*

loss&u�;��       �	n��`Yc�A�4*

loss��p:�6)�       �	}@�`Yc�A�4*

loss��<f��J       �	V��`Yc�A�4*

lossс;��y1       �	6��`Yc�A�4*

loss�_=.l�#       �	I�`Yc�A�4*

lossTo<�&=�       �	@��`Yc�A�4*

loss;k'<�5J>       �	ƈ�`Yc�A�4*

lossd�&<�0G       �	�%�`Yc�A�4*

loss�<s��W       �	'��`Yc�A�4*

lossF�<�o�       �	KX�`Yc�A�4*

loss�J=<��I       �	o��`Yc�A�4*

loss{�y<j�n�       �	?��`Yc�A�4*

lossM_�;���       �	�2�`Yc�A�4*

loss�PZ<=��       �	���`Yc�A�4*

lossq�<z�i       �	2��`Yc�A�4*

loss{��;��N�       �	�3�`Yc�A�4*

loss�L`<�Gz-       �	<��`Yc�A�4*

lossġ<�i�       �	�y�`Yc�A�4*

loss�d;�F�       �	� �`Yc�A�4*

lossd�;��0�       �	���`Yc�A�4*

loss�U<�O?�       �	pa�`Yc�A�4*

lossm<��Ny       �	���`Yc�A�4*

loss��<ދ�       �	6��`Yc�A�4*

loss�m;ϵ�       �	�-�`Yc�A�4*

loss��<z��1       �	���`Yc�A�4*

loss�e�:�"�       �	�[�`Yc�A�4*

loss��=���,       �	R��`Yc�A�4*

loss�
:��1       �	P��`Yc�A�4*

loss�<=���       �	�J�`Yc�A�4*

lossu&
;��       �	/��`Yc�A�4*

loss��=|�       �	,��`Yc�A�4*

loss�ȍ;��u       �	=*�`Yc�A�4*

loss�փ<��       �	���`Yc�A�4*

lossx�	<�L��       �	�x�`Yc�A�4*

lossfa�:E��F       �	� aYc�A�4*

lossh\m<_�!P       �	�� aYc�A�4*

lossM�<���       �	 EaYc�A�4*

losst��;;%U�       �	��aYc�A�4*

loss�Y<#$��       �	Z�aYc�A�4*

lossCS<���4       �	� aYc�A�4*

loss� <-d�+       �	�aYc�A�4*

lossJ�<�%��       �	=�aYc�A�4*

loss���<���\       �	9^aYc�A�4*

lossMP7:�T�]       �	T�aYc�A�4*

lossѩQ<���       �	�aYc�A�4*

lossl�;�u       �	ߧaYc�A�4*

loss#�<&��@       �	�IaYc�A�4*

loss�.=/'�       �	��aYc�A�4*

loss�Ic<R���       �	t�	aYc�A�4*

loss�=#;^^Cr       �	(C
aYc�A�4*

loss�N�;�4       �	��
aYc�A�4*

lossW�<����       �	�aYc�A�4*

loss���;i��J       �	�+aYc�A�4*

loss���;=C��       �	��aYc�A�4*

lossa<(�Q       �	kaYc�A�4*

loss�B=�l5�       �	aYc�A�4*

lossy�<����       �	��aYc�A�4*

loss���<rJ#�       �	�EaYc�A�4*

loss���;WFQ�       �	��aYc�A�4*

loss��"=��.       �	�aYc�A�4*

lossI$�<#@�       �	 aYc�A�4*

loss��a=�cu�       �	��aYc�A�4*

lossH�< ���       �	�aaYc�A�4*

lossp<��B�       �	�aYc�A�4*

loss��N=��@�       �	�aYc�A�4*

loss)G�:���       �	/�aYc�A�4*

loss8�i<��k
       �	6"aYc�A�4*

loss��-:R_�{       �	e�aYc�A�4*

losstĔ;�z4�       �	baYc�A�4*

loss|'f:zA��       �	caaYc�A�4*

loss�R;�r�       �	�aYc�A�4*

loss�~;W�g�       �	̚aYc�A�4*

loss& �<�8�       �	�0aYc�A�4*

loss���:�^e�       �	��aYc�A�4*

loss��<Ȍ�-       �	!�aYc�A�4*

loss�s;7"Vj       �	�aYc�A�4*

lossv<��u       �	�(aYc�A�4*

loss3;%�2       �	��aYc�A�4*

loss���9Ʊ��       �	\qaYc�A�4*

loss���;�        �	VaYc�A�4*

loss�<<�E��       �	M�aYc�A�4*

lossQr%=�\S       �	�JaYc�A�4*

loss�Y};��6       �	��aYc�A�4*

losss�P<��       �	�v aYc�A�4*

loss!�<J��)       �	!aYc�A�4*

loss�uf9!q90       �	£!aYc�A�4*

loss��<.C       �	�#aYc�A�4*

lossz?<�{�       �	û#aYc�A�4*

lossA�=v"       �	�P$aYc�A�4*

lossϗ <�yib       �	y�$aYc�A�4*

loss�0<���W       �	�~%aYc�A�4*

loss��:�Gj�       �	�&aYc�A�4*

loss�"�;��E�       �	j�&aYc�A�4*

loss�Q�:z�       �	{�'aYc�A�4*

loss,ƕ<0<�       �	3�(aYc�A�4*

loss��P9G�       �	}w)aYc�A�4*

loss.��<r�C       �	^*aYc�A�4*

loss�~E;��4�       �	k�*aYc�A�4*

loss�^�:�Û       �	Â+aYc�A�4*

loss"��;��am       �	{�,aYc�A�4*

loss[7M>TK�       �	CV-aYc�A�4*

lossOq�;���       �	p�-aYc�A�4*

loss9  <�"�       �	��.aYc�A�4*

loss�9=���       �	�\/aYc�A�4*

loss�:D<D�S�       �	�/aYc�A�4*

lossOQ�<G�q\       �	.�0aYc�A�4*

loss��I<߰�       �	l%1aYc�A�4*

loss%�<"x�<       �	k�1aYc�A�4*

loss�/<r0�;       �	�M2aYc�A�4*

lossE�;����       �	}�2aYc�A�4*

losse��9���       �	-�3aYc�A�4*

lossS�<�zUY       �	;T4aYc�A�4*

loss��v;�lC�       �	��4aYc�A�4*

loss���:�Y�Y       �	��5aYc�A�5*

loss��9�ّ       �	�+6aYc�A�5*

loss7�u:'�;A       �	��6aYc�A�5*

lossR�N=+y��       �	��7aYc�A�5*

lossc;�qo@       �	�68aYc�A�5*

loss���<pR.G       �	J�8aYc�A�5*

loss�S7;~�e�       �	�o9aYc�A�5*

loss/��<,
7       �	�	:aYc�A�5*

loss-_<#x+�       �	��:aYc�A�5*

loss��:�>��       �	p@;aYc�A�5*

lossZ��;�//       �	L�;aYc�A�5*

lossx�j;\�-W       �	��<aYc�A�5*

loss�J;�@k?       �	�X=aYc�A�5*

lossޥ
<�g��       �	��=aYc�A�5*

loss�hA<wU��       �	��>aYc�A�5*

loss��K<�uU       �	K!?aYc�A�5*

loss� <��       �	Թ?aYc�A�5*

loss���<�//�       �	/N@aYc�A�5*

loss��:����       �	��@aYc�A�5*

loss� <���       �	�xAaYc�A�5*

loss&��:�v�       �	+BaYc�A�5*

lossi�n<���       �	c�BaYc�A�5*

loss���;�
�       �	QCaYc�A�5*

loss&4=����       �	+DaYc�A�5*

loss���;���       �	i�DaYc�A�5*

loss��=��p       �	�JEaYc�A�5*

lossu=N/&3       �	��EaYc�A�5*

loss��;|�;       �	OxFaYc�A�5*

lossvP�<�"	7       �	�GaYc�A�5*

loss�ʭ:ݯM�       �	O�GaYc�A�5*

loss�%5<�       �	q�HaYc�A�5*

lossY�=hѸ�       �	�'IaYc�A�5*

loss�;�ۄ       �	8�IaYc�A�5*

loss���=��g       �	��JaYc�A�5*

loss��/=+       �	�[KaYc�A�5*

loss�'�<�6�@       �	.�KaYc�A�5*

loss[�I<'       �	V�LaYc�A�5*

lossSDA<(��\       �	4MaYc�A�5*

loss8]b;�5       �	��MaYc�A�5*

loss]��<��)~       �	��NaYc�A�5*

loss%��;��e�       �	�iOaYc�A�5*

loss?J�:�W��       �	�PaYc�A�5*

lossM�R=��k       �	x�PaYc�A�5*

loss���<�tIb       �	R�QaYc�A�5*

loss�;���!       �	��RaYc�A�5*

loss;1$<��VK       �	�0SaYc�A�5*

loss�J;��ߵ       �	��SaYc�A�5*

loss��x<���0       �	�[TaYc�A�5*

loss=ޫ:����       �	��TaYc�A�5*

lossӊ|:̆�       �	��UaYc�A�5*

loss��;M��       �	W=VaYc�A�5*

lossRb�;��ED       �	�VaYc�A�5*

loss8�<���       �	��WaYc�A�5*

loss̦X<�S)~       �	�'XaYc�A�5*

loss��=��:       �	��XaYc�A�5*

loss�@<{��6       �	׽YaYc�A�5*

loss�C�;k��       �	��ZaYc�A�5*

lossCS�9����       �	�j\aYc�A�5*

lossh9�<���       �	�]aYc�A�5*

lossW��<��2j       �	P�]aYc�A�5*

loss���<wR�       �	wH^aYc�A�5*

loss�X=�ÎG       �	��^aYc�A�5*

loss�f=���.       �	{�_aYc�A�5*

loss���<��^P       �	 `aYc�A�5*

loss1�H<0�<       �	N�`aYc�A�5*

loss��=�5�       �	baaYc�A�5*

loss`0�:��4s       �	�aaYc�A�5*

lossa^V=�)M       �	��baYc�A�5*

loss�~=�W�w       �	�CcaYc�A�5*

loss:�r<�(%       �	V-daYc�A�5*

lossoP�;u�oM       �	��daYc�A�5*

loss��:0Q       �	_eaYc�A�5*

loss�I�<[1�       �	��eaYc�A�5*

loss�82<6��       �	��faYc�A�5*

loss-��<�x'�       �	B�gaYc�A�5*

loss���:�v��       �	VJhaYc�A�5*

loss���;��+o       �	��haYc�A�5*

loss%[,=���       �	uiaYc�A�5*

loss�"u;F2       �	jaYc�A�5*

loss�5�; ��       �	�kaYc�A�5*

loss�_};9��?       �	+�laYc�A�5*

losstG=3.23       �	�6maYc�A�5*

loss+�;<�t�       �	E�maYc�A�5*

lossRFi<i$��       �	jnaYc�A�5*

loss���<��g       �	� oaYc�A�5*

loss<�<�E\       �	��oaYc�A�5*

loss=/<#���       �	�<paYc�A�5*

loss�2�<����       �	k�paYc�A�5*

loss]j�;'��       �	�lqaYc�A�5*

loss��;ϗ�       �	XraYc�A�5*

loss��;<J���       �	5�raYc�A�5*

lossO'�=�y       �	�BsaYc�A�5*

loss���<E߅/       �	#�saYc�A�5*

loss�ɗ<c,       �	
jtaYc�A�5*

lossn�w;���s       �	�uaYc�A�5*

loss�H=��u       �	>�uaYc�A�5*

loss��C<Bٙ       �	�(vaYc�A�5*

loss��:���q       �	�vaYc�A�5*

loss]q�<��>6       �	vRwaYc�A�5*

loss=�N;3��9       �	!�waYc�A�5*

lossF	F<G�r       �	n�xaYc�A�5*

loss���;��
       �	yaYc�A�5*

loss�;���       �	%�yaYc�A�5*

loss�n	=� �       �	FzaYc�A�5*

loss�lu;���       �	��zaYc�A�5*

loss>�;�̢e       �	\t{aYc�A�5*

loss$�:�Q��       �	�|aYc�A�5*

loss�.�:�       �	�|aYc�A�5*

loss��*;��p�       �	�>}aYc�A�5*

loss��Q<JG�       �	Y�}aYc�A�5*

loss\�\:^P��       �	_{~aYc�A�5*

loss�^;��t       �	aYc�A�5*

lossH	;uo       �	ӠaYc�A�5*

loss�O<<��M�       �	�I�aYc�A�5*

loss�]�;@b�y       �	��aYc�A�5*

loss��=����       �	ޒ�aYc�A�5*

loss�48<�N       �	j1�aYc�A�5*

loss4�B=�c(�       �	�ՂaYc�A�5*

loss�::W�e�       �	�q�aYc�A�5*

loss��<)8Z       �	��aYc�A�5*

loss�&= ֜N       �	�aYc�A�5*

loss��=!���       �	�d�aYc�A�5*

losshs�=�=_       �	��aYc�A�5*

loss7׾:1��U       �	|��aYc�A�5*

loss&i�<��4       �	m7�aYc�A�5*

lossl�C<��?       �	$ЇaYc�A�5*

loss��&<�	       �	�d�aYc�A�5*

lossΦ^;�\4\       �	��aYc�A�5*

lossA�k<KJ��       �	0��aYc�A�6*

loss�P9<
Н       �	'3�aYc�A�6*

loss�:��       �	CȊaYc�A�6*

loss;�C<���~       �	�h�aYc�A�6*

loss6E!<O,`C       �	]��aYc�A�6*

loss	�%</K&t       �	���aYc�A�6*

loss�T$<��=       �	:>�aYc�A�6*

lossN�<��RU       �	�ՍaYc�A�6*

loss��U<4;B<       �	;r�aYc�A�6*

loss���;[��i       �	S�aYc�A�6*

loss	=C�m       �	E��aYc�A�6*

loss�K�<�q�       �	1�aYc�A�6*

loss�6�;T�@       �	�ǐaYc�A�6*

loss��W=��       �	�\�aYc�A�6*

loss%4=�[��       �	��aYc�A�6*

lossz;Ԕ�       �	�6�aYc�A�6*

lossH�A;��       �	ӓaYc�A�6*

lossD�	=��^!       �	�j�aYc�A�6*

loss=؇=��%       �	���aYc�A�6*

loss��; ��       �	��aYc�A�6*

loss�g&<��Wg       �	l��aYc�A�6*

loss4R;�P}�       �	w�aYc�A�6*

loss�'&=�@�       �	�r�aYc�A�6*

lossR�T<^D+       �	�	�aYc�A�6*

loss�CS<	�
�       �	?�aYc�A�6*

loss��;�i�       �	�w�aYc�A�6*

lossY�< ��       �	�:�aYc�A�6*

loss���='R4�       �	wܛaYc�A�6*

loss�8:q�n       �	D��aYc�A�6*

loss�<b��       �	C�aYc�A�6*

loss�[�;���       �	f��aYc�A�6*

loss)9L�W       �	p^�aYc�A�6*

loss���<t$�       �	^�aYc�A�6*

loss�F�=��,x       �	���aYc�A�6*

loss�Q�<`�       �	x_�aYc�A�6*

loss�<�	�S       �	���aYc�A�6*

loss��;��*r       �	���aYc�A�6*

loss3	�<�       �	fI�aYc�A�6*

loss�t|=�V�       �	��aYc�A�6*

lossZx<�4i�       �	���aYc�A�6*

loss�}<s�T       �	MJ�aYc�A�6*

lossv�<RO^�       �	��aYc�A�6*

loss�"p<�b�I       �	�u�aYc�A�6*

lossʶ=;��K       �	��aYc�A�6*

loss@f+=A��       �	k��aYc�A�6*

lossD�n<��3       �	GʧaYc�A�6*

lossh�;�{�L       �	���aYc�A�6*

loss�/<�hR]       �	�ߩaYc�A�6*

loss��<�`��       �	��aYc�A�6*

loss�z�<�\�       �	�8�aYc�A�6*

loss�;�:��       �	��aYc�A�6*

loss�q;�Gb1       �	c��aYc�A�6*

loss��<�B3{       �	V�aYc�A�6*

loss ~8<kKػ       �	��aYc�A�6*

loss�9<�P;       �	���aYc�A�6*

loss�/n;�Q:}       �	`<�aYc�A�6*

loss�I�;D��y       �	hаaYc�A�6*

loss�O�;��q       �	�d�aYc�A�6*

lossQ�;�{       �	�
�aYc�A�6*

lossl);���       �	���aYc�A�6*

loss�(p;�/�       �	T8�aYc�A�6*

loss�	�:)���       �	�γaYc�A�6*

lossJ�:���       �	8f�aYc�A�6*

lossnb1;~���       �	Y��aYc�A�6*

loss	�;�B�       �	��aYc�A�6*

loss3S.<<G       �	�!�aYc�A�6*

loss�m�;�͖       �	[��aYc�A�6*

loss؟<r�0�       �	�H�aYc�A�6*

loss�=s��F       �	
۷aYc�A�6*

lossv9�;���       �	Ct�aYc�A�6*

loss��<�/�       �	�aYc�A�6*

lossR�*=
��g       �	��aYc�A�6*

loss���;V�-	       �	IL�aYc�A�6*

loss��,<��O6       �	�aYc�A�6*

loss�d�=�/2�       �	�z�aYc�A�6*

lossZ<<�M�       �	,�aYc�A�6*

loss &<z*�       �	'��aYc�A�6*

loss��<fF�4       �	4�aYc�A�6*

losshpy;�2       �	PȽaYc�A�6*

loss�W�<�N2�       �	�[�aYc�A�6*

lossz�Q<�M�/       �	9�aYc�A�6*

loss��Q=�3n"       �	��aYc�A�6*

loss;<V;vV.       �	�)�aYc�A�6*

loss��*<[���       �	<��aYc�A�6*

lossE�!=$��5       �	eQ�aYc�A�6*

loss�2�<���       �	���aYc�A�6*

loss{<��t�       �	tz�aYc�A�6*

loss�,�;bĂ       �	�aYc�A�6*

lossL��;�:^       �	H��aYc�A�6*

lossW�<jQ9       �	�D�aYc�A�6*

lossH�;⾗v       �	L��aYc�A�6*

loss��=�Q       �	)u�aYc�A�6*

loss8ѓ:�ހw       �	��aYc�A�6*

loss kJ<�ì'       �	���aYc�A�6*

loss;�#<T�C�       �	�>�aYc�A�6*

loss���<v���       �	���aYc�A�6*

lossC��<Iײ�       �	lx�aYc�A�6*

loss�ف:P���       �	��aYc�A�6*

lossѧq<��p       �	Ƣ�aYc�A�6*

loss��T<sH��       �	"6�aYc�A�6*

loss*!=�N*       �	B��aYc�A�6*

lossߜZ<�=��       �	�s�aYc�A�6*

lossat<���       �	��aYc�A�6*

loss=H\<���k       �	R��aYc�A�6*

loss�^c=��=       �	ʈ�aYc�A�6*

loss\n3<]yu       �	�aYc�A�6*

loss-�L<�+(�       �	2��aYc�A�6*

loss$s�<(fM�       �	K=�aYc�A�6*

loss=�<�^�4       �	l��aYc�A�6*

loss�'�;��-�       �	Xq�aYc�A�6*

losse0=I�       �	^�aYc�A�6*

lossƦ�;�N�4       �	���aYc�A�6*

loss-��=-k�       �	.;�aYc�A�6*

loss��<8���       �	O��aYc�A�6*

lossJ�A<����       �	���aYc�A�6*

loss��
<���       �	x%�aYc�A�6*

lossO%>V�U       �	��aYc�A�6*

lossc�<g��9       �	�O�aYc�A�6*

loss,��;�p       �	G��aYc�A�6*

loss��0<�o��       �	�}�aYc�A�6*

lossne�;��ܲ       �	��aYc�A�6*

loss�I�;�d��       �	���aYc�A�6*

lossW@<s��       �	�U�aYc�A�6*

loss,¢;C��       �	���aYc�A�6*

lossQ�<���0       �	<��aYc�A�6*

loss
m<#��       �	[&�aYc�A�6*

loss�ѱ<N��       �	���aYc�A�6*

lossRY6<8%s{       �	F_�aYc�A�6*

loss| <>�I       �	8��aYc�A�7*

loss���<�x       �	q��aYc�A�7*

loss�¯<i)�=       �	A)�aYc�A�7*

loss(�Y=ڣ�q       �	���aYc�A�7*

loss�l
<���       �	�h�aYc�A�7*

loss/;e��       �	K �aYc�A�7*

loss�SK<����       �	W��aYc�A�7*

loss�.�<N�       �	#g�aYc�A�7*

loss4 �<��Е       �	2�aYc�A�7*

loss3�;	�       �	ș�aYc�A�7*

loss�_�;�Ӕ�       �	��aYc�A�7*

loss�O<�Q	�       �	�_�aYc�A�7*

loss�x
;q;nv       �	���aYc�A�7*

lossH�:&Wg       �	���aYc�A�7*

loss{�g<�{*       �	� �aYc�A�7*

loss
a�=!%��       �	��aYc�A�7*

loss��;�smj       �	�I�aYc�A�7*

lossm�<Lp�       �	�aYc�A�7*

lossx�:�h�       �	���aYc�A�7*

loss��l<Ow�\       �	���aYc�A�7*

loss�!y=�{��       �	3��aYc�A�7*

loss��:ي       �	0d�aYc�A�7*

loss�C<sy��       �	���aYc�A�7*

loss���;&d��       �	,��aYc�A�7*

loss?S>���       �	�>�aYc�A�7*

losscн<�w�       �	���aYc�A�7*

loss���<�N�       �	Ȗ�aYc�A�7*

loss��=�#       �	/�aYc�A�7*

lossfs=�P�+       �	���aYc�A�7*

loss���:B��r       �	"l�aYc�A�7*

loss���;��"�       �	K�aYc�A�7*

loss��F=7���       �	���aYc�A�7*

lossXI�;��Ԟ       �	[�aYc�A�7*

loss�8=����       �	x�aYc�A�7*

loss��%<�7�       �	j��aYc�A�7*

lossT��;�N�       �	UL�aYc�A�7*

loss��=����       �	���aYc�A�7*

loss�1;�c�h       �	���aYc�A�7*

lossh)�<P�       �	�.�aYc�A�7*

lossO�<� O�       �	���aYc�A�7*

lossLz<����       �	���aYc�A�7*

loss�w=x�        �	�>�aYc�A�7*

loss��;�sJ�       �	��aYc�A�7*

loss�{~;�J�[       �	��aYc�A�7*

lossrm�:���B       �	q �aYc�A�7*

lossP8<���F       �	���aYc�A�7*

loss]5;��]p       �	=C�aYc�A�7*

lossi�1;00�*       �	2��aYc�A�7*

loss)�X=�\�D       �	@��aYc�A�7*

loss��S;���       �	+�aYc�A�7*

loss�Z`<T��y       �	a��aYc�A�7*

loss.t<<���       �	���aYc�A�7*

lossvS4<�iP�       �	�D�aYc�A�7*

lossq7;;xN|V       �	���aYc�A�7*

loss���<���       �	�� bYc�A�7*

loss'��<Z@q~       �	�2bYc�A�7*

loss�Qz<��i       �	��bYc�A�7*

loss�y�<{�Q6       �	��bYc�A�7*

loss{�l<_��b       �	�zbYc�A�7*

loss���;�Q�       �	GbYc�A�7*

loss�u4=W��       �	V�bYc�A�7*

loss]�<�ֹ@       �	!�bYc�A�7*

loss_I�:p���       �	�3bYc�A�7*

lossڵV<ٷ�"       �	��bYc�A�7*

lossn��<+yq       �	�bYc�A�7*

loss���;�I
       �	�;bYc�A�7*

loss��=�Љ       �	;�bYc�A�7*

loss���<�:z)       �	Q�	bYc�A�7*

loss�Ņ;�R       �	�p
bYc�A�7*

loss��;<
&(       �	kbYc�A�7*

loss���<�u��       �	��bYc�A�7*

loss�::$��       �	�MbYc�A�7*

loss?�f<.�p�       �	��bYc�A�7*

loss��/;���       �	j�bYc�A�7*

loss���;��cr       �	�dbYc�A�7*

lossڷ�<��d       �	8hbYc�A�7*

loss1ׂ<ĿG       �	�bYc�A�7*

lossȖC=T5�       �	T�bYc�A�7*

loss�� <x콋       �	�FbYc�A�7*

loss)�6;���/       �	r5bYc�A�7*

loss���<�ࢤ       �	��bYc�A�7*

loss/O�;���       �	B|bYc�A�7*

loss�d!<�S~�       �	bYc�A�7*

loss�9$��       �	2�bYc�A�7*

loss��=;�Z�       �	�wbYc�A�7*

loss=�;ZWy�       �	�!bYc�A�7*

lossf;�٧�       �	�bYc�A�7*

loss)˛<�##�       �	 �bYc�A�7*

loss$ �;ѻ_       �	ݘbYc�A�7*

lossӈ=fR�       �	�4bYc�A�7*

lossX};�       �	B
bYc�A�7*

loss]�>;�R|�       �	ÞbYc�A�7*

loss�>�<���       �	/3bYc�A�7*

loss��<��e�       �	��bYc�A�7*

lossz�?9��R�       �	DibYc�A�7*

lossD e;�ȅ       �	F
bYc�A�7*

loss��;�,       �	�bYc�A�7*

loss,[<���       �	�:bYc�A�7*

loss�+<�V�       �	��bYc�A�7*

loss8�8=���       �	�ubYc�A�7*

loss�&�;�FQ�       �	^ bYc�A�7*

loss6�:� 8�       �	� bYc�A�7*

loss�.
:���i       �	��!bYc�A�7*

loss��;����       �	Ts"bYc�A�7*

loss�n�;SD��       �	$#bYc�A�7*

loss���<0���       �	v�#bYc�A�7*

loss�;��H�       �	�9$bYc�A�7*

loss��88��       �	��$bYc�A�7*

loss\d�:��X       �	{�%bYc�A�7*

lossJ��<�q8�       �	=I&bYc�A�7*

loss3b�:���       �	J�&bYc�A�7*

loss�(>����       �	�(bYc�A�7*

loss�j�<�n�       �	�>)bYc�A�7*

loss��U;�v�Z       �	��)bYc�A�7*

losse�x;�,2�       �	�n*bYc�A�7*

loss.ֺ<�4�       �	p[+bYc�A�7*

loss]9�<�e"f       �	;�+bYc�A�7*

loss��
<G�       �	l�,bYc�A�7*

loss
�<��P       �	�j-bYc�A�7*

loss���;ӥ��       �	�.bYc�A�7*

loss���<��dj       �	��.bYc�A�7*

loss�~�<1YE�       �	fl/bYc�A�7*

loss���<ְ��       �	
0bYc�A�7*

lossZ�<����       �	�0bYc�A�7*

lossR<z/L�       �	1@1bYc�A�7*

loss�U�<-�[       �	�2bYc�A�7*

loss�=r0_�       �	�3bYc�A�7*

loss��;���       �	)�3bYc�A�7*

loss�>/<Ac       �	e5bYc�A�8*

loss��1< s       �	��5bYc�A�8*

loss�&;�ˏ        �	eQ6bYc�A�8*

loss���;I��3       �	��6bYc�A�8*

loss��&;y�i�       �	��7bYc�A�8*

losszN;���       �	'.8bYc�A�8*

losso9�9�&
       �	.;9bYc�A�8*

lossm��<Ͳ;<       �	��9bYc�A�8*

lossuИ=n�h@       �	h:bYc�A�8*

loss/�Q<9c�l       �	\;bYc�A�8*

loss��/<���       �	��;bYc�A�8*

lossΥ�<�?<�       �	3<bYc�A�8*

loss�ѝ:>V(       �	��<bYc�A�8*

lossA�;�        �	�i=bYc�A�8*

loss{��:* �       �	�>bYc�A�8*

loss3��;��ge       �	ګ>bYc�A�8*

loss�:
<�#ڪ       �	�B?bYc�A�8*

lossl��:E1�       �	8�?bYc�A�8*

lossb�	<nG��       �	}u@bYc�A�8*

loss�"O9��G9       �	�AbYc�A�8*

loss���=�|S       �	��AbYc�A�8*

loss|�	<�0�0       �	��BbYc�A�8*

loss%?:��A       �	�)CbYc�A�8*

loss~�;�Ά|       �	S�CbYc�A�8*

loss}�<����       �	�yDbYc�A�8*

loss���<�w}       �	��EbYc�A�8*

loss!<����       �	�WFbYc�A�8*

loss���; D�       �	��FbYc�A�8*

loss�^1=�k�       �	_�GbYc�A�8*

loss��,<,"L�       �	�5HbYc�A�8*

loss��X;�g4       �	��HbYc�A�8*

loss��p<bܱ       �	�iIbYc�A�8*

loss�=�w�       �	�JbYc�A�8*

loss��.;���