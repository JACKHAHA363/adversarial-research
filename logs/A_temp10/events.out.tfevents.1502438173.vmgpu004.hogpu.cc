       �K"	  @�Xc�Abrain.Event:2<mBg5�     Or�	W]�Xc�A"��
^
dataPlaceholder*/
_output_shapes
:���������*
shape: *
dtype0
W
labelPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
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
seed2ܘ�
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 
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
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2���
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
_output_shapes
: *
T0
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
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
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
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
:
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
 *  �?*
dtype0*
_output_shapes
: 
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@
�
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
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
valueB:*
_output_shapes
:*
dtype0
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
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
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
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
seed2�ڒ
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
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��&*
T0*
seed���)*
dtype0
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
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
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
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
   *
_output_shapes
:*
dtype0
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
seed2���
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
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0*
validate_shape(*
use_locking(
|
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
Z
dense_2/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

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
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout_2/cond/Mergedense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
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
valueB"      *
dtype0*
_output_shapes
:
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
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
valueB"      *
_output_shapes
:*
dtype0
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
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
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
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
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2��
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
)sequential_1/dropout_1/cond/dropout/FloorFloor'sequential_1/dropout_1/cond/dropout/add*
T0*/
_output_shapes
:���������@
�
'sequential_1/dropout_1/cond/dropout/divRealDivsequential_1/dropout_1/cond/mul-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
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
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
T0*
N
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
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
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
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
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*
T0*(
_output_shapes
:����������
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
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
�
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
'sequential_1/dropout_2/cond/dropout/mulMul'sequential_1/dropout_2/cond/dropout/div)sequential_1/dropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
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
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
T0*
_class
loc:@num_inst*
validate_shape(*
_output_shapes
: 
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
num_correct/readIdentitynum_correct*
_class
loc:@num_correct*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
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
ConstConst*
valueB: *
_output_shapes
:*
dtype0
X
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
Assign_1Assignnum_correctConst_3*
use_locking(*
T0*
_class
loc:@num_correct*
validate_shape(*
_output_shapes
: 
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
 *   A*
dtype0*
_output_shapes
: 
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*
T0*'
_output_shapes
:���������

a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
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
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*

axis *
_output_shapes
:*
T0*
N
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
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
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
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
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
T0*

axis *
N*
_output_shapes
:
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
���������*
dtype0*
_output_shapes
:
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
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
_output_shapes
: *
T0
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*

axis *
_output_shapes
:*
T0*
N
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
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
 *  �?*
_output_shapes
: *
dtype0
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
 *    *
_output_shapes
: *
dtype0
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
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
 *  �?*
_output_shapes
: *
dtype0
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
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
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
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
N
PlaceholderPlaceholder*
_output_shapes
:*
shape: *
dtype0
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
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
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
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: *
T0
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
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
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: 
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
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
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0
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
valueB *
dtype0*
_output_shapes
: 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
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
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
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
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
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
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
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
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
���������*
dtype0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
T0*
out_type0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/RealDiv_2RealDivgradients/div_1_grad/RealDiv_1div_1/y*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*
T0*'
_output_shapes
:���������

�
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: 
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
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
_output_shapes
:���������
*
T0
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
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�

�
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������*
T0
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
:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
Tshape0*(
_output_shapes
:����������*
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
valueB *
dtype0*
_output_shapes
: 
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
T0*
Tshape0*(
_output_shapes
:����������
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
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: 
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_2Shapegradients/Switch_1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
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
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������
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
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
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
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
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
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*/
_output_shapes
:���������@*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
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
T0*
Tshape0*
_output_shapes
: 
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
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
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
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
T0
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
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
strides

�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:@@*
use_cudnn_on_gpu(
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
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
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
valueB"         @   *
_output_shapes
:*
dtype0
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
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
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
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
n
beta2_power/readIdentitybeta2_power*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
j
zerosConst*%
valueB@*    *
dtype0*&
_output_shapes
:@
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
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
l
zeros_1Const*%
valueB@*    *
dtype0*&
_output_shapes
:@
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
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
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
VariableV2*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
shape:@@*
dtype0*
shared_name *
	container 
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
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_2/bias/Adam
VariableV2*
shared_name * 
_class
loc:@conv2d_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam_1
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
b
zeros_8Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
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
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
b
zeros_9Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam_1
VariableV2*
	container *
dtype0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
shape:���*
shared_name 
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
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
W
zeros_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
�
dense_1/bias/Adam
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
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
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam
VariableV2*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
shape:	�
*
dtype0*
shared_name *
	container 
�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

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
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam_1
VariableV2*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
shape:	�
*
dtype0*
shared_name *
	container 
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
*    *
_output_shapes
:
*
dtype0
�
dense_2/bias/Adam_1
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
 *fff?*
_output_shapes
: *
dtype0
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
Adam/mul_1*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
�
AdamNoOp&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
N
	loss/tagsConst*
valueB
 Bloss*
_output_shapes
: *
dtype0
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"��j��     �>�	Wy`�Xc�AJ��
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
seed2ܘ�
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
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
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
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
v
conv2d_2/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0
`
conv2d_2/random_uniform/minConst*
valueB
 *�\1�*
_output_shapes
: *
dtype0
`
conv2d_2/random_uniform/maxConst*
valueB
 *�\1=*
_output_shapes
: *
dtype0
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2���
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
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@
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
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
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
:���������@*
seed2���*
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
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
valueB" d  �   *
_output_shapes
:*
dtype0
_
dense_1/random_uniform/minConst*
valueB
 *�3z�*
_output_shapes
: *
dtype0
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
seed2�ڒ*
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
VariableV2*
shape:���*
shared_name *
dtype0*!
_output_shapes
:���*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��&*
T0*
seed���)*
dtype0
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
 *̈́U�*
_output_shapes
: *
dtype0
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
seed2���
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
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
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
sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
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
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@
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
$sequential_1/dropout_1/cond/switch_fIdentity"sequential_1/dropout_1/cond/Switch*
T0
*
_output_shapes
:
r
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2��
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
'sequential_1/dropout_1/cond/dropout/addAdd-sequential_1/dropout_1/cond/dropout/keep_prob2sequential_1/dropout_1/cond/dropout/random_uniform*/
_output_shapes
:���������@*
T0
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
shrink_axis_mask *
Index0*
T0*
end_mask*
_output_shapes
:*

begin_mask *
ellipsis_mask 
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
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
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
sequential_1/activation_3/ReluRelusequential_1/dense_1/BiasAdd*
T0*(
_output_shapes
:����������
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
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
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
seed2���*
T0*
seed���)*
dtype0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
6sequential_1/dropout_2/cond/dropout/random_uniform/mulMul@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniform6sequential_1/dropout_2/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
2sequential_1/dropout_2/cond/dropout/random_uniformAdd6sequential_1/dropout_2/cond/dropout/random_uniform/mul6sequential_1/dropout_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*1
_class'
%#loc:@sequential_1/activation_3/Relu*<
_output_shapes*
(:����������:����������
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
 *    *
dtype0*
_output_shapes
: 
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
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
T0*
_class
loc:@num_inst*
validate_shape(*
_output_shapes
: 
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
ArgMax_1ArgMaxlabelArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
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
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
Const_1Const*
valueB
 *  �B*
_output_shapes
: *
dtype0
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
 *    *
_output_shapes
: *
dtype0
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
 *   A*
dtype0*
_output_shapes
: 
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*
T0*'
_output_shapes
:���������

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
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
T0*
out_type0*
_output_shapes
:
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
value	B :*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
T0*

axis *
N*
_output_shapes
:
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
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
value	B : *
dtype0*
_output_shapes
: 
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
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
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
value	B :*
_output_shapes
: *
dtype0
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
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
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
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
value	B : *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
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
 *  �?*
_output_shapes
: *
dtype0
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
valueB *
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
N
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
:
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
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
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: 
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: 
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
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0
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
valueB *
dtype0*
_output_shapes
: 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
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
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
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
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
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
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
Bgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradientPreventGradient%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:������������������
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
valueB *
_output_shapes
: *
dtype0
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
gradients/div_1_grad/SumSumgradients/div_1_grad/RealDiv*gradients/div_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: 
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:

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
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������*
T0
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
out_type0*
_output_shapes
:*
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:����������
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/NegNegsequential_1/dropout_2/cond/mul*(
_output_shapes
:����������*
T0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Neg-sequential_1/dropout_2/cond/dropout/keep_prob*
T0*(
_output_shapes
:����������
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
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_2/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
2gradients/sequential_1/dropout_2/cond/mul_grad/mulMulOgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
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
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a(
�
;gradients/sequential_1/dense_1/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_1/MatMul_grad/MatMul4^gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*)
_output_shapes
:�����������*
T0
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
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0
�
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
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
Lgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivRealDivOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*
T0*/
_output_shapes
:���������@
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1Sum:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulNgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
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
T0*
Tshape0*/
_output_shapes
:���������@
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
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
�
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_3/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
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
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*
T0*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
N*/
_output_shapes
:���������@
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
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
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@
�
Agradients/sequential_1/conv2d_2/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������@*
T0
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
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
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
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
valueB"         @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:@*
data_formatNHWC*
strides

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
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
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
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
shape:@*
shared_name 
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
l
zeros_1Const*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam_1
VariableV2*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
shape:@*
dtype0*
shared_name *
	container 
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
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
T
zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/bias/Adam_1
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
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
VariableV2*
shared_name * 
_class
loc:@conv2d_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
VariableV2*
shared_name * 
_class
loc:@conv2d_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
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
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
b
zeros_8Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
validate_shape(*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
b
zeros_9Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
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
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*!
_output_shapes
:���
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0
W
zeros_11Const*
valueB�*    *
_output_shapes	
:�*
dtype0
�
dense_1/bias/Adam_1
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
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
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
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam
VariableV2*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
shape:	�
*
dtype0*
shared_name *
	container 
�
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�

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
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
T0
U
zeros_14Const*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_2/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense_2/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

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
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:


dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
use_locking( 
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0n�       ��-	e���Xc�A*

loss7M@�p�       ��-	����Xc�A*

lossı@$��'       ��-	���Xc�A*

loss�r@�FE�       ��-	)<��Xc�A*

loss#�@�`�2       ��-	���Xc�A*

lossx��?��d       ��-	���Xc�A*

loss?��?���       ��-	q��Xc�A*

loss���?Ec�       ��-	f���Xc�A*

loss�B�?��Z       ��-	t���Xc�A	*

loss�9�?Cү       ��-	eP��Xc�A
*

loss
�?ԛeB       ��-	5��Xc�A*

loss��?T Q       ��-	@���Xc�A*

loss�f?z��       ��-	 C��Xc�A*

loss
J�?�s�       ��-	���Xc�A*

loss�\?�_�       ��-	���Xc�A*

loss�b?�9�       ��-	�)��Xc�A*

loss7?���       ��-	�/��Xc�A*

loss�1?�       ��-	�ʜ�Xc�A*

losse&?d���       ��-	�d��Xc�A*

lossdy[?�1!       ��-	?��Xc�A*

loss��p?�rؙ       ��-	��Xc�A*

loss�+T? S��       ��-	����Xc�A*

loss�	*?��7       ��-	�9��Xc�A*

lossðv?��y>       ��-	�ԡ�Xc�A*

lossm�D?;,�       ��-	zp��Xc�A*

loss��5?�c�       ��-	��Xc�A*

loss�?7*�a       ��-	ܻ��Xc�A*

lossÀ)?zz       ��-	�a��Xc�A*

loss�8L?"�       ��-	��Xc�A*

loss_�I?sȐ       ��-	z���Xc�A*

loss�A?�@m       ��-	���Xc�A*

loss��?�ĩ�       ��-	�7��Xc�A *

loss�*?���       ��-	\��Xc�A!*

loss�yH?���(       ��-	����Xc�A"*

lossj�?¸ ?       ��-	�-��Xc�A#*

loss��E?d�,       ��-	�ȩ�Xc�A$*

loss��?X���       ��-	Nb��Xc�A%*

lossӞB?x�S       ��-	����Xc�A&*

loss�`?@��W       ��-	����Xc�A'*

loss�<?_+qs       ��-	L6��Xc�A(*

lossg�#?�Q�@       ��-	�٬�Xc�A)*

lossO�>5BÅ       ��-	�|��Xc�A**

loss��/?�ky�       ��-	&��Xc�A+*

loss �?9Zȿ       ��-	o���Xc�A,*

loss� )?o�_�       ��-	�P��Xc�A-*

loss�o%?b�r       ��-	G��Xc�A.*

lossE�)?M��1       ��-	r���Xc�A/*

lossx�>����       ��-	(��Xc�A0*

loss���>��9       ��-	����Xc�A1*

loss3��>A  *       ��-	qU��Xc�A2*

lossV[�>���       ��-	���Xc�A3*

loss|��>�Wo�       ��-	N���Xc�A4*

loss��>L�,       ��-	%=��Xc�A5*

loss��.?�%ܵ       ��-	��Xc�A6*

loss(v�>�e�       ��-	ގ��Xc�A7*

loss�!�>���       ��-	)��Xc�A8*

loss͚Q>����       ��-	����Xc�A9*

lossdv�>|���       ��-	�U��Xc�A:*

loss ��>��       ��-	d��Xc�A;*

loss18�>7��       ��-	����Xc�A<*

losss��>BQE       ��-	� ��Xc�A=*

loss���>����       ��-	E���Xc�A>*

loss�H�>S��       ��-	<O��Xc�A?*

loss���>���       ��-	p��Xc�A@*

loss��>ZI��       ��-	o���Xc�AA*

loss`��>pIui       ��-	y#��Xc�AB*

lossI_?R��       ��-	�ü�Xc�AC*

loss��?*�       ��-	<f��Xc�AD*

loss��>]X�x       ��-	���Xc�AE*

loss.��>?$3!       ��-	Ʀ��Xc�AF*

lossxt�>���       ��-	�A��Xc�AG*

loss��>wG��       ��-	���Xc�AH*

loss2��>RLU       ��-	���Xc�AI*

lossr\�>��p       ��-	���Xc�AJ*

lossd�?ʞ��       ��-	����Xc�AK*

lossɨ?�r�'       ��-	�W��Xc�AL*

lossY?ב�!       ��-	
���Xc�AM*

lossd@?�Vȍ       ��-	���Xc�AN*

loss��>Xwm       ��-	}��Xc�AO*

loss�B�>09c�       ��-	
���Xc�AP*

loss|��>
�Et       ��-	aQ��Xc�AQ*

lossm	?t�)       ��-	5���Xc�AR*

lossX��>m�\�       ��-	����Xc�AS*

loss��x>���c       ��-	!��Xc�AT*

lossf��>��	       ��-	����Xc�AU*

lossLt]>-��       ��-	�Q��Xc�AV*

loss{d>̅1�       ��-	y���Xc�AW*

loss���>��\       ��-	E���Xc�AX*

loss���>�&�       ��-	���Xc�AY*

loss���>��7�       ��-	����Xc�AZ*

loss�\�>�pg4       ��-	�e��Xc�A[*

loss$��>��S       ��-	+���Xc�A\*

loss7)�>,1       ��-	���Xc�A]*

loss���>v�%�       ��-	fl��Xc�A^*

lossW��>�F�        ��-	���Xc�A_*

loss8�D>@�	�       ��-	D���Xc�A`*

loss�q�>TK�       ��-	T���Xc�Aa*

losso��>�[H�       ��-	�[��Xc�Ab*

loss�J@?�~��       ��-	|��Xc�Ac*

loss���>]<�       ��-	a���Xc�Ad*

loss&�>X[m�       ��-	�D��Xc�Ae*

loss�t(>A�       ��-	O���Xc�Af*

loss�ɿ>
�%'       ��-	���Xc�Ag*

lossءn>�ۅ	       ��-	���Xc�Ah*

loss�y�>��       ��-	����Xc�Ai*

loss��>�$y       ��-	���Xc�Aj*

loss�w�>T���       ��-	ެ��Xc�Ak*

loss(�C>�>Δ       ��-	�A��Xc�Al*

loss���>���       ��-	����Xc�Am*

loss�֢>��v�       ��-	Fy��Xc�An*

loss���>Af�       ��-	#��Xc�Ao*

loss�^�>y(�       ��-	���Xc�Ap*

loss�y�>N�X�       ��-	VF��Xc�Aq*

loss��X>~8��       ��-	��Xc�Ar*

loss�)>/�C       ��-	M���Xc�As*

lossA�>�
`z       ��-	4��Xc�At*

lossQ>jN��       ��-	(���Xc�Au*

loss���>�h�<       ��-	�n��Xc�Av*

loss\��>�#I       ��-	���Xc�Aw*

loss�|�>�F�L       ��-	s���Xc�Ax*

loss�*�>���S       ��-	�0��Xc�Ay*

lossK�>�E�       ��-	!���Xc�Az*

loss��>|�Y�       ��-	i��Xc�A{*

loss=	}>a���       ��-	?���Xc�A|*

lossC\�=hxg�       ��-	���Xc�A}*

loss7�>�'t       ��-	�'��Xc�A~*

lossƻ�>B`�       ��-	ܹ��Xc�A*

loss�t>=d�}       �	M��Xc�A�*

loss���>�A�       �	7���Xc�A�*

loss�ã>%Br       �	.v��Xc�A�*

loss�f>��H       �	)?��Xc�A�*

lossk�>�|{|       �	����Xc�A�*

loss��X>߅�       �	o��Xc�A�*

loss7/&>���       �	���Xc�A�*

loss8U>"d�9       �	����Xc�A�*

loss�m>IS�       �	�)��Xc�A�*

lossq�:>(���       �	4���Xc�A�*

loss��U>2ò�       �	E��Xc�A�*

loss�=P>�W��       �	'���Xc�A�*

loss��P>=�       �	�p��Xc�A�*

loss�S�=�f��       �	���Xc�A�*

loss?"e>�h�       �	����Xc�A�*

lossOf�>�� �       �	[?��Xc�A�*

loss�Ai>`x�G       �	(���Xc�A�*

loss��.>\ԝ�       �	Փ��Xc�A�*

loss��>r�M       �	_'��Xc�A�*

loss��>�l�       �	����Xc�A�*

lossS��=���       �	�T��Xc�A�*

loss$>�tD�       �	}���Xc�A�*

loss��H>wc�       �	I���Xc�A�*

lossmPC>~T��       �	3��Xc�A�*

loss(��>xm       �	����Xc�A�*

loss6�>x(O�       �	 a��Xc�A�*

loss���>b"��       �	���Xc�A�*

loss��=��l{       �	h���Xc�A�*

loss*Q�=�Џ       �	^.��Xc�A�*

lossxRY>"hr}       �	����Xc�A�*

loss;�d>!���       �	vn��Xc�A�*

loss}��>�0       �	H1��Xc�A�*

loss�Ə>R&-       �	����Xc�A�*

loss��>ǜ�1       �	e��Xc�A�*

loss�ŗ>�.k       �	e���Xc�A�*

loss�
t>Bm�       �	$���Xc�A�*

loss���=3���       �	�3��Xc�A�*

loss
�>|T(�       �	c���Xc�A�*

loss�|>e��[       �	�x��Xc�A�*

loss�gJ>}�	       �	���Xc�A�*

lossm�@>�kY�       �	���Xc�A�*

loss�1�=�� �       �	�I��Xc�A�*

loss{��=I�Z       �	���Xc�A�*

loss):>4�1       �	���Xc�A�*

loss�a�=�Ww+       �	q: �Xc�A�*

loss4�9>9�{       �	�� �Xc�A�*

lossɮ�>/�       �	 o�Xc�A�*

loss%�.>���       �	�Xc�A�*

loss��>���       �	ۦ�Xc�A�*

loss\Q�>y���       �	NA�Xc�A�*

loss���>t�d�       �	��Xc�A�*

loss*8j>U��       �	E��Xc�A�*

loss��
> �W�       �	A�Xc�A�*

loss��=�9�       �	���Xc�A�*

loss1��>.h�o       �	��Xc�A�*

loss=��|1       �	c*�Xc�A�*

loss��=l�,       �	���Xc�A�*

loss��4>�P�       �	�x	�Xc�A�*

lossp'>�0�`       �	#
�Xc�A�*

loss�7>H       �	�
�Xc�A�*

loss. R>�4�        �	"Q�Xc�A�*

loss;y1>R�C       �	|��Xc�A�*

loss�uA>���}       �	 ��Xc�A�*

losssj�>���       �	TT�Xc�A�*

loss	O>�Ǫ�       �	���Xc�A�*

loss�k>��g�       �	΍�Xc�A�*

losscX<>��u       �	�o�Xc�A�*

loss� >��%�       �	G�Xc�A�*

loss;B�>hd�O       �	=��Xc�A�*

losst͎>���       �	�>�Xc�A�*

lossh">��h       �	���Xc�A�*

loss=B>��e�       �	e�Xc�A�*

lossk�>�n+�       �	���Xc�A�*

loss�,>���H       �	��Xc�A�*

loss�1,>[�7�       �	�7�Xc�A�*

lossmF2>�)��       �	���Xc�A�*

loss��>�͛�       �	�\�Xc�A�*

loss�`>7T�       �	(��Xc�A�*

lossr�=l~-�       �	k��Xc�A�*

loss��P>JfR�       �	B!�Xc�A�*

lossK�>�k]�       �	9��Xc�A�*

lossJ(>{��       �	mr�Xc�A�*

lossx�p>���-       �	)�Xc�A�*

loss��>>��       �	&��Xc�A�*

loss!g@>�H`F       �	Q�Xc�A�*

loss�Ҿ=Y�%x       �	��Xc�A�*

loss�^'>�y�,       �	���Xc�A�*

loss�pq>Q�(       �	�N�Xc�A�*

lossj�I>ݚ��       �	.��Xc�A�*

lossO��>���       �	ׅ�Xc�A�*

lossLG�>B�m+       �	��Xc�A�*

loss)�>����       �	^��Xc�A�*

loss�w>-���       �		��Xc�A�*

loss��=G�       �	�; �Xc�A�*

lossT�=�#I�       �	�� �Xc�A�*

loss<j>;�X       �	�z!�Xc�A�*

loss���>B�V�       �	�"�Xc�A�*

loss�=׏?�       �	�"�Xc�A�*

loss��=�Z^q       �	tC#�Xc�A�*

loss��=� ��       �	��#�Xc�A�*

lossT+z>�Θ�       �	�|$�Xc�A�*

loss���>e�Z~       �	�%�Xc�A�*

loss8>��       �	�%�Xc�A�*

losstY>�)��       �	�]&�Xc�A�*

lossE��=��׸       �	n'�Xc�A�*

loss�C>�a\�       �	�'�Xc�A�*

loss��2>Wl��       �	�~(�Xc�A�*

lossz(�>��w�       �	�)�Xc�A�*

loss%��=aW��       �	��)�Xc�A�*

loss��>io�4       �	�Z*�Xc�A�*

loss��>,��       �	�+�Xc�A�*

loss��>�"�)       �	9�+�Xc�A�*

loss�C>E��]       �	�m,�Xc�A�*

lossFE>�M�       �	6-�Xc�A�*

loss��<>U`^�       �	�.�Xc�A�*

loss�r�>Or_       �	-$/�Xc�A�*

loss$�>��D       �	��/�Xc�A�*

lossH�C>���       �	��0�Xc�A�*

lossAT�>�me�       �	9'1�Xc�A�*

loss��=�x��       �	8�1�Xc�A�*

loss�Ol>2
�       �	?�2�Xc�A�*

loss�4>�0��       �	�3�Xc�A�*

loss��C>���       �	��3�Xc�A�*

loss8�b>�Iy�       �	�G4�Xc�A�*

loss��>?1��       �	�4�Xc�A�*

loss��K>����       �	�y5�Xc�A�*

loss�՗=e��       �	�6�Xc�A�*

loss h�=9E\�       �	Y�6�Xc�A�*

loss:�8>%-yi       �	�C7�Xc�A�*

loss�u�>q�f;       �	�7�Xc�A�*

loss`�>�E]       �	R~8�Xc�A�*

loss��>C�u       �	�9�Xc�A�*

loss}�u>�(��       �	��9�Xc�A�*

loss�e>7;b       �	\U:�Xc�A�*

loss�=b���       �	k�:�Xc�A�*

loss=�>�ѵ�       �	�;�Xc�A�*

loss��>L�m       �	O!<�Xc�A�*

loss�V�>��_%       �	B�<�Xc�A�*

lossn�'>[A��       �	�J=�Xc�A�*

loss�%;>��5�       �	�)>�Xc�A�*

loss��j>�RY�       �	��>�Xc�A�*

loss��>YƝ�       �	-Z?�Xc�A�*

loss��>���       �	��?�Xc�A�*

lossV��=���       �	͐@�Xc�A�*

lossJ�o>��M       �	9'A�Xc�A�*

loss,=>1��z       �	�A�Xc�A�*

lossF> ���       �	�jB�Xc�A�*

loss�*>{���       �	�C�Xc�A�*

loss�>�,�       �	�C�Xc�A�*

loss��%>+�Ί       �	�RD�Xc�A�*

loss���=�\       �	q�D�Xc�A�*

loss}0�>Z�l       �	|E�Xc�A�*

loss��>�r�       �	sF�Xc�A�*

loss�M=�D��       �	%�F�Xc�A�*

loss���=^�y       �	�[G�Xc�A�*

loss�>�Y9�       �	�G�Xc�A�*

lossz�1>�ݴa       �	C�H�Xc�A�*

lossw�j>�K6       �	O"I�Xc�A�*

loss�)>u�       �	��I�Xc�A�*

loss��K>�(�       �	�TJ�Xc�A�*

loss��=� ��       �	c�J�Xc�A�*

lossf��=}���       �	�K�Xc�A�*

loss�,"=���'       �	�L�Xc�A�*

loss�g�=�l_�       �	��L�Xc�A�*

lossbA>du       �	V�M�Xc�A�*

loss�;={�AD       �	� N�Xc�A�*

loss�g>T��       �	-�N�Xc�A�*

loss�X>N��       �	�HO�Xc�A�*

loss3�M>�jn       �	��O�Xc�A�*

losse�c>X3&w       �	L�P�Xc�A�*

losso�*>�yϘ       �	G�Q�Xc�A�*

loss�M>�>�       �	��R�Xc�A�*

lossq�W>+yF�       �	kcS�Xc�A�*

loss�@ >�D��       �	cT�Xc�A�*

loss���=R�=Y       �	w�T�Xc�A�*

loss�NJ>~�>�       �	WV�Xc�A�*

losse1>![��       �	g�V�Xc�A�*

loss�#>{f��       �	�3W�Xc�A�*

loss��6>+�D�       �	��W�Xc�A�*

lossQ�B=��qn       �	�hX�Xc�A�*

loss�`3>�k�        �	��X�Xc�A�*

lossԙ=q��p       �	>�Y�Xc�A�*

loss;�'>Gb�6       �	o*Z�Xc�A�*

loss�G�=�i$T       �	+�Z�Xc�A�*

loss��`>�r�       �	/O[�Xc�A�*

loss :D>C���       �	]�[�Xc�A�*

loss� >ߧ-       �	�u\�Xc�A�*

loss3^=���       �	]�Xc�A�*

loss�}h>m֌[       �	�]�Xc�A�*

loss��=��õ       �	�>^�Xc�A�*

lossd�"=yvp�       �	 �^�Xc�A�*

lossV%>�>��       �	�e_�Xc�A�*

loss�>椰       �	f�_�Xc�A�*

lossOʿ=��\C       �	��`�Xc�A�*

losss�v>]��       �	-\a�Xc�A�*

loss\>�=3       �	9�a�Xc�A�*

loss�-�=���       �	E�b�Xc�A�*

loss�J>��*       �	'c�Xc�A�*

loss��>lZ        �	V�c�Xc�A�*

loss�s�=Dhr       �	jKd�Xc�A�*

loss�!9>GG�I       �	��d�Xc�A�*

loss�+l>:�e�       �	#�e�Xc�A�*

lossUq�>���v       �	�f�Xc�A�*

loss��b>�
��       �	ĵf�Xc�A�*

lossn��==H\#       �	@Ng�Xc�A�*

loss1n�=Gz�       �	)�g�Xc�A�*

loss��=
��       �	(�h�Xc�A�*

lossA��>��D�       �	bi�Xc�A�*

loss(=m=��h       �	�i�Xc�A�*

loss�N>{�5       �	_Cj�Xc�A�*

loss�>����       �	�j�Xc�A�*

loss�C�=x���       �	"ok�Xc�A�*

loss�cn=��       �	�l�Xc�A�*

lossf�=��$       �	�l�Xc�A�*

loss�w=�E�       �	>m�Xc�A�*

loss��D>+��1       �	��m�Xc�A�*

loss3v> �       �	�ln�Xc�A�*

lossZh�>�6GN       �	go�Xc�A�*

loss�@>���       �	��o�Xc�A�*

lossj�=`�6       �	�Pp�Xc�A�*

loss���>�~6        �	��p�Xc�A�*

loss��>���       �	��q�Xc�A�*

lossk(>cV�n       �	�'r�Xc�A�*

losss`�=Ჱ�       �	��r�Xc�A�*

loss��=�s��       �	�Zs�Xc�A�*

loss��>�qr�       �	=�s�Xc�A�*

loss\�N>��1       �	�t�Xc�A�*

loss>	��)       �	�#u�Xc�A�*

loss�
Q=�x��       �	MIv�Xc�A�*

loss�>� �<       �	O�v�Xc�A�*

losst~�=����       �	��w�Xc�A�*

loss�* >9�֥       �	�#x�Xc�A�*

loss��=��       �	U�x�Xc�A�*

loss=#>`n-       �	d\y�Xc�A�*

loss2|�=��       �	w�y�Xc�A�*

loss��F>�Y)       �	\�z�Xc�A�*

lossq�>��<       �	n{�Xc�A�*

loss�0<>��7       �	|�Xc�A�*

loss��>����       �	��|�Xc�A�*

lossO�>0d��       �	(~}�Xc�A�*

loss.�w=|�       �	v~�Xc�A�*

loss_�=!ε       �	��~�Xc�A�*

lossi3>;�_       �	�N�Xc�A�*

loss��=�Q�       �	���Xc�A�*

loss��>W�υ       �	����Xc�A�*

lossH�#>�1�       �	}#��Xc�A�*

loss��=��j�       �	����Xc�A�*

loss�m>Cc�	       �	�j��Xc�A�*

loss�>^���       �	���Xc�A�*

lossoA9>C�       �	%���Xc�A�*

loss���=�\v       �	�Y��Xc�A�*

loss��=��/C       �	���Xc�A�*

loss�N>���       �	+���Xc�A�*

loss�jF>.��        �	� ��Xc�A�*

loss`P>�耂       �	$���Xc�A�*

loss�=�f>�       �	�u��Xc�A�*

loss ��=�.�       �	r��Xc�A�*

loss�p�=�s־       �	[���Xc�A�*

loss2J>wR�j       �	�^��Xc�A�*

loss��>�vi�       �	���Xc�A�*

loss|d1>1AZ       �	���Xc�A�*

loss?��=�!��       �	
1��Xc�A�*

loss�>��       �	�ԋ�Xc�A�*

loss2G>����       �	�m��Xc�A�*

loss��4>��       �	���Xc�A�*

lossf�$>�_��       �	����Xc�A�*

loss̠\>�9;m       �	
/��Xc�A�*

lossj�=R�=�       �	6Ȏ�Xc�A�*

loss��>ϫS9       �	[^��Xc�A�*

loss�8=8^       �	�N��Xc�A�*

loss���=%G�       �	6��Xc�A�*

lossW��=ռ�       �	���Xc�A�*

loss�Eg>S�Bd       �	�Ԓ�Xc�A�*

lossrY>�@�       �	A���Xc�A�*

lossR�:>T�t�       �	7��Xc�A�*

loss]��>$۝p       �	5��Xc�A�*

loss(��=����       �	ǝ��Xc�A�*

loss�M�=G05�       �	?R��Xc�A�*

loss',>w}�V       �	F@��Xc�A�*

loss�a>�lf�       �	4���Xc�A�*

lossJ1>�ͭ�       �	՗��Xc�A�*

loss[z&>-�       �	Iۙ�Xc�A�*

loss;�W>0�y�       �	uu��Xc�A�*

loss�3>'�       �	Z��Xc�A�*

loss��{>�g5       �	,՛�Xc�A�*

lossx�=���       �	����Xc�A�*

loss#��>
�8�       �	���Xc�A�*

loss�O>����       �	�E��Xc�A�*

lossd�6>����       �	g&��Xc�A�*

loss�T=2�1�       �	tC��Xc�A�*

loss��=$g�K       �	�ܡ�Xc�A�*

lossʌ>s��B       �	���Xc�A�*

loss�<>{W�6       �	s.��Xc�A�*

loss��=��"�       �	`:��Xc�A�*

loss��4>�ir       �	o��Xc�A�*

loss��7>a��{       �	I���Xc�A�*

loss���>M ��       �	����Xc�A�*

loss�!>EQ�M       �	N'��Xc�A�*

loss({�>J��       �	%��Xc�A�*

lossjb>E`j[       �	��Xc�A�*

loss8],>��~�       �	����Xc�A�*

loss�L�=��O       �	6��Xc�A�*

loss&�=x�u       �	Iت�Xc�A�*

loss�\>�b{�       �	p��Xc�A�*

loss�P�=�0N�       �	n��Xc�A�*

lossS��=�9�       �	����Xc�A�*

loss���=�Zh       �	$F��Xc�A�*

loss�,>�;       �	���Xc�A�*

lossy(�=o|[       �	\v��Xc�A�*

loss��=�9��       �	���Xc�A�*

loss=P�=��޿       �	ݙ��Xc�A�*

lossӶ�=�X�       �	�3��Xc�A�*

lossҪ�=��dE       �	�ư�Xc�A�*

loss��>��I$       �	Ja��Xc�A�*

lossIut>�r�       �	Z���Xc�A�*

lossx��=��F�       �	����Xc�A�*

loss$nH>��ň       �	2!��Xc�A�*

loss9=s�v       �	s���Xc�A�*

loss�l>�,{X       �	�Q��Xc�A�*

loss���>�-�+       �	��Xc�A�*

loss��,>l���       �	�{��Xc�A�*

lossg9�=S���       �	���Xc�A�*

lossh=>�l
       �	����Xc�A�*

loss�%
>�J1�       �	AI��Xc�A�*

loss�^�=ҫR�       �	I���Xc�A�*

loss�D�=f�\X       �	J���Xc�A�*

loss�m�=� �|       �	%;��Xc�A�*

loss��}>(c�       �	�׹�Xc�A�*

loss 6S>��`�       �	yu��Xc�A�*

lossb�>�_��       �	��Xc�A�*

loss�h>���       �	2���Xc�A�*

loss��>��0�       �	�^��Xc�A�*

loss��>.!:       �	P ��Xc�A�*

lossz��=yy�       �	����Xc�A�*

lossj#�=���x       �	�-��Xc�A�*

lossv��=RV
       �	+þ�Xc�A�*

loss�;o> h[�       �	�_��Xc�A�*

loss�T=b	K       �	����Xc�A�*

loss.��=��2P       �	؟��Xc�A�*

loss�r�=��$�       �	;6��Xc�A�*

loss;D�=P��       �	���Xc�A�*

lossHh�=��       �	l^��Xc�A�*

loss1��<�M{       �	k���Xc�A�*

loss�L\>/�A�       �	Ί��Xc�A�*

loss2��=-$%�       �	�"��Xc�A�*

loss(n�=q���       �	���Xc�A�*

loss�C;>�	        �	]R��Xc�A�*

loss���=6P/�       �	����Xc�A�*

lossLF=��w       �	C���Xc�A�*

lossZ�=;���       �	|,��Xc�A�*

loss)>/��       �	v���Xc�A�*

lossA�=�mD�       �	�V��Xc�A�*

loss�>xG��       �	���Xc�A�*

loss6��>j	k�       �	J���Xc�A�*

lossڹ�=S�LI       �	�/��Xc�A�*

loss;�=Z�       �	����Xc�A�*

loss,�|>	��J       �	`u��Xc�A�*

loss�K�=��x�       �	���Xc�A�*

loss�RZ=P�'F       �	����Xc�A�*

loss�V�=��3       �	�j��Xc�A�*

loss� >��D       �	���Xc�A�*

loss�=��0       �	(���Xc�A�*

lossX�>W�O�       �	�Z��Xc�A�*

loss���=���       �	���Xc�A�*

loss�Nt>`)+       �	���Xc�A�*

loss�I=� �3       �	�R��Xc�A�*

lossAv=���^       �	F&��Xc�A�*

lossE�=�'��       �	�X��Xc�A�*

loss� 9=��d.       �	\��Xc�A�*

lossE��=�O�       �	4���Xc�A�*

loss�Ф=#�J�       �	���Xc�A�*

lossܐ�=`�       �	�5��Xc�A�*

loss_�0>[s�       �	!��Xc�A�*

loss�>��[       �	7���Xc�A�*

loss�=,L�h       �	�E��Xc�A�*

loss�7�<c���       �	v���Xc�A�*

loss���=����       �	a���Xc�A�*

loss��=  �       �	���Xc�A�*

loss��w=޸�g       �	���Xc�A�*

loss��$>�cX,       �	>!��Xc�A�*

loss�ѷ=d��       �	���Xc�A�*

loss[�c>�Ȋ�       �	���Xc�A�*

loss#��=ɏ�u       �	�o��Xc�A�*

loss�5i=,f��       �	�c��Xc�A�*

loss��$>�[�       �	6w��Xc�A�*

loss��=(�2�       �	�\��Xc�A�*

loss�>��oI       �	0���Xc�A�*

loss���=�A�N       �	����Xc�A�*

lossl<l>s�7       �	@���Xc�A�*

loss�`=QTw
       �	�K��Xc�A�*

loss���=a�-�       �	���Xc�A�*

loss���=��i       �	]���Xc�A�*

loss>��|o       �	����Xc�A�*

loss�'.=s���       �	����Xc�A�*

loss->��       �	ɮ��Xc�A�*

lossjQ�<';�       �	.W��Xc�A�*

loss#=d*`,       �	c*��Xc�A�*

loss�Ұ=�,�       �	�;��Xc�A�*

loss�=?��       �	���Xc�A�*

lossܙ�=B"�S       �	����Xc�A�*

loss37>0	"�       �	�|��Xc�A�*

loss�� >݄i>       �	�/��Xc�A�*

lossD1=E�7�       �	����Xc�A�*

loss͠�=���       �	F���Xc�A�*

loss���=CS�       �	����Xc�A�*

lossHy�=0J
       �	|��Xc�A�*

loss�0<���       �	��Xc�A�*

loss:�!=.���       �	ٲ��Xc�A�*

loss�7�=bu2       �	iQ��Xc�A�*

loss��K<zk�       �	����Xc�A�*

loss�Z=#?"�       �	
���Xc�A�*

loss@�=͠O�       �	a��Xc�A�*

loss�;>�*�       �	����Xc�A�*

loss�Ae<'	�       �	���Xc�A�*

loss65C<� �       �	���Xc�A�*

loss�w�;B0�r       �	����Xc�A�*

loss@&>F��       �	yW��Xc�A�*

loss�/:>s"c�       �	����Xc�A�*

loss��=}\       �	���Xc�A�*

loss}	<:��       �	�&��Xc�A�*

loss��x=��L�       �	ܼ��Xc�A�*

loss��?PL�i       �	vP��Xc�A�*

loss�1�<���       �	����Xc�A�*

loss嬅>��x       �	N}��Xc�A�*

loss�@�=�[��       �	E��Xc�A�*

loss<\>�9��       �	����Xc�A�*

loss��=�       �	�D��Xc�A�*

lossC=8]]E       �	^���Xc�A�*

lossM�t>��:       �	-z �Xc�A�*

loss2f�=�R�k       �	�Xc�A�*

loss
4%>x3��       �	 ��Xc�A�*

loss-�>�K��       �	>�Xc�A�*

loss.�=��       �	���Xc�A�*

loss�T>�2p       �	si�Xc�A�*

loss`�K>j�$       �	��Xc�A�*

lossH>�s�       �	���Xc�A�*

loss)L">��       �	�4�Xc�A�*

lossx6T>`'�       �	}��Xc�A�*

loss�ϥ=ч�j       �	�d�Xc�A�*

loss��=�֍�       �	{��Xc�A�*

loss�Z>C-��       �	���Xc�A�*

loss&z�=	w�k       �	C��Xc�A�*

loss8�==��c       �	pC	�Xc�A�*

loss�h�=�i~l       �	��	�Xc�A�*

loss���=M�}       �	Kv
�Xc�A�*

lossvΖ<���       �	��Xc�A�*

loss�;=���       �	���Xc�A�*

loss�b=�!��       �	dW�Xc�A�*

loss���=�\r       �	j��Xc�A�*

lossm�T=�5}@       �	���Xc�A�*

loss�T>	"�       �	�X�Xc�A�*

loss�F>��       �	���Xc�A�*

loss�&{<�3��       �	/��Xc�A�*

lossj�=C���       �	�C�Xc�A�*

loss��z=Wz       �	J��Xc�A�*

lossHx�<^Q��       �	��Xc�A�*

loss�ɞ=��       �	���Xc�A�*

loss	�='O��       �	}�Xc�A�*

loss<�4=��i�       �	�!�Xc�A�*

loss;��=���       �	��Xc�A�*

loss��(>}_        �	ʤ�Xc�A�*

loss��>R<��       �	�H�Xc�A�*

lossR��=v�؁       �	���Xc�A�*

lossd+ =O_E�       �	��Xc�A�*

loss@%�=���       �	:u�Xc�A�*

lossF�=h�j       �	�#�Xc�A�*

loss�J�=��       �	G�Xc�A�*

loss B>ޠL       �	��Xc�A�*

lossߋ�=J_k�       �	qX�Xc�A�*

loss��r=�@�       �	Y��Xc�A�*

loss��=��u       �	���Xc�A�*

loss[!�=T�       �	�c�Xc�A�*

loss�H�=O�.       �	��Xc�A�*

loss�2>G
       �	�8�Xc�A�*

loss֮�=}E�g       �	 �8�Xc�A�*

loss-��=�7�	       �	.9�Xc�A�*

loss�Ö=�#o       �	�9�Xc�A�*

loss��5>�Ԓ�       �	Ln:�Xc�A�*

loss�/;>����       �	|;�Xc�A�*

loss�G>SB�       �	)�;�Xc�A�*

loss��=k�?       �	�E<�Xc�A�*

loss��^=&ziW       �	��<�Xc�A�*

lossɪl>t3�       �	�v=�Xc�A�*

lossW��=�m*n       �	�>�Xc�A�*

lossc��=h��M       �	Է>�Xc�A�*

lossI�
>z���       �	,c?�Xc�A�*

loss�m=�<}?       �	c@�Xc�A�*

loss�7�<��t       �	��@�Xc�A�*

loss{>�;�       �	LA�Xc�A�*

lossS8�<��p�       �	�A�Xc�A�*

loss���=�?\       �	3�B�Xc�A�*

loss�p�=ڏg       �	x'C�Xc�A�*

loss���>�L7       �	y�C�Xc�A�*

loss�e>�j�I       �	�fD�Xc�A�*

loss���=+R��       �	��D�Xc�A�*

loss�d�=�?A�       �	��E�Xc�A�*

lossW�F>�q�       �	:;F�Xc�A�*

loss&��=�A�!       �	@�F�Xc�A�*

loss���=�؝�       �	uvG�Xc�A�*

loss���=��f�       �	�H�Xc�A�*

loss�-�=��u       �	>�H�Xc�A�*

loss -8>0cY�       �	<gI�Xc�A�*

loss��>~��y       �	�J�Xc�A�*

loss.>���t       �	�J�Xc�A�*

loss��>ݻ��       �	@0K�Xc�A�*

lossVW>��^       �	��K�Xc�A�*

loss�>"�L�       �	7�L�Xc�A�*

loss��D=:�A�       �	�M�Xc�A�*

loss�%>i1��       �	�9N�Xc�A�*

loss��=\��p       �	��N�Xc�A�*

loss��D>W�
�       �	�aO�Xc�A�*

loss���>.Cg       �	P�Xc�A�*

lossf��=��       �	�P�Xc�A�*

losss�,>�~v       �	?;Q�Xc�A�*

loss�>T��Z       �	�`R�Xc�A�*

loss(�=ww       �	!S�Xc�A�*

lossF�>Ú:M       �	:T�Xc�A�*

loss���=�H�       �	��T�Xc�A�*

loss�/>��J�       �	��U�Xc�A�*

loss��=>*3       �	k+V�Xc�A�*

loss.h�=��L�       �	z�V�Xc�A�*

loss�Y�=I��$       �	kW�Xc�A�*

loss�;�<9Gb       �	kX�Xc�A�*

loss,�<�L�	       �	�X�Xc�A�*

loss�I�=b�e�       �	�\Y�Xc�A�*

lossA�l=����       �	o�Y�Xc�A�*

loss/k�>�.�]       �	�6[�Xc�A�*

loss���<|�O       �	�[�Xc�A�*

loss��b<#�V       �	f\�Xc�A�*

loss(�=>!�s       �	�]�Xc�A�*

loss4��<�.�       �	˞]�Xc�A�*

loss�;
>9۽�       �	�8^�Xc�A�*

loss�؝=ZkJI       �	��^�Xc�A�*

loss�l>4�x�       �	`w_�Xc�A�*

loss���=�*	       �	Y`�Xc�A�*

loss�=���       �	��`�Xc�A�*

loss���=����       �	�{a�Xc�A�*

loss�\�=��͌       �	�Ab�Xc�A�*

loss�[=�t�_       �	��b�Xc�A�*

loss�k>�3��       �	��c�Xc�A�*

loss���=� �       �	WCd�Xc�A�*

loss�Dx>�*�       �	��d�Xc�A�*

loss7$->^CT�       �	`�e�Xc�A�*

loss��=���       �	 Bf�Xc�A�*

loss�2>g=�       �	u�f�Xc�A�*

loss��=+DTG       �	K�g�Xc�A�*

losshQ�=�๑       �	E�h�Xc�A�*

loss��=1��       �	YMi�Xc�A�*

losso��=k�       �	v�i�Xc�A�*

loss��!>��V�       �	L�j�Xc�A�*

loss�d>x!��       �	Kk�Xc�A�*

lossd�=$�Q�       �	|�k�Xc�A�*

loss|`r=��'       �	�l�Xc�A�*

loss �^>��Y�       �	�Em�Xc�A�*

loss��4>�a�       �	K�m�Xc�A�*

loss,��=���       �	X�n�Xc�A�*

loss_^�=��       �	�1o�Xc�A�*

loss���=b4F�       �	Z�o�Xc�A�*

lossD`�=d��       �	2vp�Xc�A�*

lossaG'>���       �	mq�Xc�A�*

loss�R�=��~�       �	��q�Xc�A�*

loss[�+=�%2�       �	br�Xc�A�*

loss�`k=��~       �	s�Xc�A�*

loss�� >���_       �	��s�Xc�A�*

loss�)>��       �	�Tt�Xc�A�*

lossv-�=���5       �	��t�Xc�A�*

loss�k�=�d��       �	��u�Xc�A�*

lossQf>�g%       �	�Ev�Xc�A�*

lossR�D=�s!       �	k�v�Xc�A�*

loss���=1l�T       �	�w�Xc�A�*

loss�G=�`�       �	.<x�Xc�A�*

lossM�=>j���       �	��x�Xc�A�*

lossڿ�=�{       �	�y�Xc�A�*

loss�!=O�Y       �	�Oz�Xc�A�*

loss�"�=4�>(       �	J�z�Xc�A�*

lossA#>I�B%       �	��{�Xc�A�*

lossΗ=xb��       �	>?|�Xc�A�*

loss�
�=�U~Z       �	/}�Xc�A�*

loss}��=5��       �	��}�Xc�A�*

loss���<qQ�       �	Nz~�Xc�A�*

loss�j=�˚�       �	�Xc�A�*

lossz;>#W:       �	y��Xc�A�*

loss�v�=*-�V       �	P��Xc�A�*

lossv~d>�"��       �	����Xc�A�*

loss�D�=���       �	s���Xc�A�*

loss\�K=>�       �	�@��Xc�A�*

loss���<L��`       �	�ۂ�Xc�A�*

loss{�y=4�;�       �	��Xc�A�*

loss�<�̀�       �	MJ��Xc�A�*

lossCG�=TB       �	g��Xc�A�*

loss��>?���       �	����Xc�A�*

loss�%�=ё�       �	�P��Xc�A�*

lossz��=�Qr�       �	����Xc�A�*

loss$��=��o       �	�)��Xc�A�*

loss��=��7       �	�ш�Xc�A�*

loss3�U>�P�x       �	�s��Xc�A�*

loss���=S�       �	���Xc�A�*

loss�<�5Tt       �	Ѳ��Xc�A�*

loss#��=�:O�       �	^M��Xc�A�*

loss���=I|       �	���Xc�A�*

loss�I�=bkL�       �	D���Xc�A�*

loss��%>ԑ[        �	]R��Xc�A�*

loss���=�=`       �	���Xc�A�*

loss�2�=I�       �	)$��Xc�A�*

loss�o�=���l       �	Ï�Xc�A�*

loss���=Ĭ\y       �	�f��Xc�A�*

loss=�Z�u       �	M��Xc�A�*

loss��>.��       �	����Xc�A�*

loss� �=J�Sf       �	�n��Xc�A�*

lossd��=�-�       �	_$��Xc�A�*

loss�0�=O0E       �	E��Xc�A�*

loss��<�`�       �	�ϔ�Xc�A�*

loss��!= �&�       �	Ʀ��Xc�A�*

loss��<t.)E       �	
K��Xc�A�*

loss��m=���       �	���Xc�A�*

loss�9�=v���       �	I���Xc�A�*

loss���=���       �	�'��Xc�A�*

lossE	�=�ݞA       �	n�Xc�A�*

loss��=�m��       �	�Z��Xc�A�*

lossx�=� ~       �	���Xc�A�*

loss���<��#       �	|���Xc�A�*

loss�cb=i	�|       �	>��Xc�A�*

loss�5=���	       �	���Xc�A�*

loss�fR=^h��       �	8��Xc�A�*

loss*f>c���       �	���Xc�A�*

loss�9�=�/�_       �	ǻ��Xc�A�*

loss�W�=1�
�       �	�b��Xc�A�*

loss��=���       �	���Xc�A�*

lossOǻ<�c/       �	���Xc�A�*

loss�&�=-�^�       �	�@��Xc�A�*

loss�O�=@uE�       �	�ӡ�Xc�A�*

loss�D8>i�	�       �	Tr��Xc�A�*

loss,�c=i��       �	���Xc�A�*

lossE�=�ь       �	�ϣ�Xc�A�*

loss�<�=d�+l       �	�m��Xc�A�*

loss�K�=\�]o       �	=
��Xc�A�*

lossrF�<�	D�       �	U���Xc�A�*

loss�ke=��       �	�@��Xc�A�*

lossH�=��t�       �	�֦�Xc�A�*

loss�I�<�8�C       �	�p��Xc�A�*

lossv��=��1       �	_E��Xc�A�*

lossq*�<{EC�       �	�ި�Xc�A�*

loss�V�=�*/�       �	n���Xc�A�*

lossńT=�yѪ       �	_C��Xc�A�*

loss ��<�y��       �	���Xc�A�*

losss��=���       �	����Xc�A�*

loss�r >ytC�       �	�B��Xc�A�*

loss��<	�       �	A���Xc�A�*

loss�s�=g�A�       �	����Xc�A�*

loss���=Rn�       �	fM��Xc�A�*

loss��>��2A       �	����Xc�A�*

lossj�=ъ��       �	����Xc�A�*

loss��=�uB5       �	UO��Xc�A�*

lossqH�<W�s       �	C��Xc�A�*

lossd�>G�K       �	�ױ�Xc�A�*

lossEI�=fu       �	����Xc�A�*

loss�8#=�[�       �	�-��Xc�A�*

loss�/4=J�V#       �	ٳ�Xc�A�*

lossV�=h��       �	ʋ��Xc�A�*

loss�,>�|�       �	5��Xc�A�*

lossvj�=�Ó       �	 ��Xc�A�*

loss�=N@       �	����Xc�A�*

loss���=Ņ]�       �	�@��Xc�A�*

loss&4>���       �	���Xc�A�*

loss���=u�3Q       �	����Xc�A�*

loss�S=$>�       �	>]��Xc�A�*

loss|�=���2       �	���Xc�A�*

lossl��<0ٜ       �	Cʺ�Xc�A�*

loss�{�=z�b�       �	�z��Xc�A�*

loss��
=�u�       �	
.��Xc�A�*

lossR��=Ck�       �	���Xc�A�*

loss���=AO�       �	����Xc�A�*

loss��!=^�-�       �	�n��Xc�A�*

loss���=���       �	���Xc�A�*

loss]�=��3�       �	�п�Xc�A�*

lossd�|=�w_�       �	�x��Xc�A�*

loss���=*?nd       �	@��Xc�A�*

loss_�3=ؙ8/       �	����Xc�A�*

loss\��=� �"       �	����Xc�A�*

lossrV�=�595       �	�q��Xc�A�*

loss�?�=�5��       �	���Xc�A�*

loss��=G^�/       �	x���Xc�A�*

loss(.�=H���       �	���Xc�A�*

lossк=*&��       �	j2��Xc�A�*

loss���=���6       �	����Xc�A�*

lossJ�=��r       �	Ō��Xc�A�*

loss!=���\       �	X8��Xc�A�*

loss��=#�/�       �	����Xc�A�*

loss!B�=7zs�       �	͏��Xc�A�*

loss�R>�q��       �	vQ��Xc�A�*

loss�D�>�q�       �	����Xc�A�*

loss��@>��       �	Ú��Xc�A�*

loss�>;W       �	�E��Xc�A�*

loss��=���       �	l���Xc�A�*

lossq��=��&u       �	B���Xc�A�*

lossf�>w�we       �	�H��Xc�A�*

loss��:>:!O       �	H���Xc�A�*

lossn�3=�_�       �	����Xc�A�*

loss퓞=B(s]       �	Zg��Xc�A�*

loss�ޞ=�<�-       �	�)��Xc�A�*

lossVG�=j�^�       �	K���Xc�A�*

lossF��=*&�       �	����Xc�A�*

loss=K�=����       �	*���Xc�A�*

lossl��=U�`�       �	�R��Xc�A�*

loss�;�<��       �	
1��Xc�A�*

loss(>s��       �	����Xc�A�*

lossV�#>�<��       �	����Xc�A�*

lossn~�=U4�       �	i��Xc�A�*

lossDl[=~�+�       �	x��Xc�A�*

loss �>�j��       �	и��Xc�A�*

loss�^�=X��       �	�c��Xc�A�*

lossR�$>�Q�       �	]��Xc�A�*

loss�l=��41       �	����Xc�A�*

loss搴=*@�h       �	Sz��Xc�A�*

loss�>'���       �	[%��Xc�A�*

lossߞ�=t�	       �	���Xc�A�*

losso��=�Z��       �	�T��Xc�A�*

loss-R�=�       �	����Xc�A�*

loss��=x�>       �	,���Xc�A�*

loss���=j}�K       �	�H��Xc�A�*

loss7=��$�       �	����Xc�A�*

lossA}�=R
B�       �	���Xc�A�*

lossۑX=s'��       �	N��Xc�A�*

loss��=�L�       �	����Xc�A�*

loss���=�Vd       �	ٙ��Xc�A�*

lossc��=��^�       �	5��Xc�A�*

loss��	=�=��       �	����Xc�A�*

loss=�Q!;       �	y���Xc�A�*

loss���=�㫉       �	�G��Xc�A�*

loss��K=�3�       �	)?��Xc�A�*

loss���=5��       �	����Xc�A�*

lossa�j=� ��       �	���Xc�A�*

loss�/�=��8�       �	�E��Xc�A�*

loss�x=Ѭ�       �	����Xc�A�*

loss��=l,       �	Kx��Xc�A�*

loss��I>S9B       �	o��Xc�A�*

loss^G=c��-       �	"���Xc�A�*

loss��=@�;�       �	q9��Xc�A�*

loss�4i=���q       �	t���Xc�A�*

loss%�>uo�V       �	j��Xc�A�*

loss�Y�=�]�S       �	���Xc�A�*

loss��?>9w�       �	a���Xc�A�*

loss�>��K       �	`?��Xc�A�*

loss=�<"��       �	����Xc�A�*

loss4j�=�9�H       �	�m��Xc�A�*

loss�=$���       �	���Xc�A�*

loss��=x"c       �	����Xc�A�*

loss���=b:*       �	6V��Xc�A�*

loss���=���W       �	����Xc�A�*

losszl�=���       �	����Xc�A�*

loss���=�T]s       �	F'��Xc�A�*

loss6�2>KDP\       �	з��Xc�A�*

loss�Z:= ��Y       �	�N��Xc�A�*

lossp��<kW_6       �	G���Xc�A�*

lossA�=��G       �	���Xc�A�*

loss?˸=�XRc       �	�/��Xc�A�*

loss�FZ=�~@�       �	����Xc�A�*

loss�;�=A3	o       �	�g��Xc�A�*

lossD6�=ڟ#�       �	W��Xc�A�*

lossVϹ=���       �	����Xc�A�*

lossV;=+PUZ       �	QO��Xc�A�*

lossVߡ=�       �	����Xc�A�*

loss3��<���_       �	���Xc�A�*

loss�(�<r�j       �	T��Xc�A�*

lossɺ�=��H!       �	����Xc�A�*

lossjĥ<�z9e       �	ɐ��Xc�A�*

lossO�F=���n       �	�<��Xc�A�*

loss�j=�_�       �	����Xc�A�*

lossr��=����       �	�q��Xc�A�*

loss;��=��M�       �	n �Xc�A�*

lossE�Z>s��       �	�� �Xc�A�*

loss���=�ʷb       �	G�Xc�A�*

loss�E�=�l�W       �	���Xc�A�*

loss�W=��.>       �	�}�Xc�A�*

loss��<��ɵ       �	��Xc�A�*

loss��=!w;U       �	p>�Xc�A�*

loss���=�,�       �	2��Xc�A�*

loss���=��R�       �	��Xc�A�*

loss�>��S       �	�4�Xc�A�*

loss#�<��       �	8��Xc�A�*

loss���=�p�G       �	�~�Xc�A�*

loss�=&=�|       �	��Xc�A�*

loss�y(=o��X       �	O��Xc�A�*

loss��=�i�Y       �	�F	�Xc�A�*

loss��=���       �	L�	�Xc�A�*

lossō=���       �	�w
�Xc�A�*

loss�׊=D�       �	S@�Xc�A�*

loss��=	Ն�       �	���Xc�A�*

lossHE�=r�;o       �	�m�Xc�A�*

lossi,6=���R       �	RE�Xc�A�*

loss�==u��       �	y��Xc�A�*

lossK >9�]       �	��Xc�A�*

loss-#�=k��$       �	�&�Xc�A�*

loss��s=3��@       �	��Xc�A�*

loss�>��lj       �	Cq�Xc�A�*

loss�}�=���       �	�(�Xc�A�*

lossd��=��!h       �	��Xc�A�*

lossmp�=�_�       �	5A�Xc�A�*

loss��X>z��E       �	��Xc�A�*

loss��=��n~       �	mt�Xc�A�*

lossm��=�y�c       �	��Xc�A�*

loss��=6\��       �	��Xc�A�*

loss�?>��       �	�S�Xc�A�*

loss�׭=f�       �	R��Xc�A�*

loss�B= �I�       �	���Xc�A�*

lossc�=y}mb       �	V*�Xc�A�*

loss�0=�k       �	���Xc�A�*

loss�W]>���       �	al�Xc�A�*

loss15�=���       �	��Xc�A�*

loss�^>
��d       �	���Xc�A�*

loss`�>�c>B       �	�F�Xc�A�*

losso8�=&���       �	��Xc�A�*

loss���;���       �	�t�Xc�A�*

lossC�W=:q1�       �	��Xc�A�*

loss8
=����       �	f��Xc�A�*

lossT�%>C�f#       �	~9�Xc�A�*

lossWI�=%W��       �	J��Xc�A�*

loss ��>+�~       �	0b�Xc�A�*

loss��=����       �	 �Xc�A�*

loss'�!=}��I       �	ǜ �Xc�A�*

lossE�>>���X       �	�3!�Xc�A�*

lossB=�o"�       �	��!�Xc�A�*

loss{�^=��       �	�y"�Xc�A�*

lossv��=l�B       �	�#�Xc�A�*

loss4?K=�.�M       �	�#�Xc�A�*

loss��=rQ�       �	cE$�Xc�A�*

loss�i�=FA'       �	�$�Xc�A�*

loss���=گD�       �	Cs%�Xc�A�*

loss!\�<�='       �	-	&�Xc�A�*

lossd-�=b�F�       �	�&�Xc�A�*

loss(2=�O       �	='�Xc�A�*

loss48�=0�*c       �	�'�Xc�A�*

loss�;S=��t       �	�u(�Xc�A�*

loss&��=��wP       �	
)�Xc�A�*

loss%��=q�E       �	0�)�Xc�A�*

loss��}=��~�       �	�d*�Xc�A�*

loss�8�=l        �	� +�Xc�A�*

loss7y�=ԟY�       �	�+�Xc�A�*

loss�>�ݜ       �	O<,�Xc�A�*

loss}�>�D�P       �	^�,�Xc�A�*

loss�i=��7�       �	�s-�Xc�A�*

loss�}�<�z8       �	�.�Xc�A�*

lossM �=��Q       �	]�.�Xc�A�*

losse��=KEv�       �	�S/�Xc�A�*

loss4,�=�Sa       �	�0�Xc�A�*

loss��=(p�       �	��0�Xc�A�*

loss$P=����       �	m;1�Xc�A�*

lossJǠ=
!`7       �	��1�Xc�A�*

loss?�>��(7       �	�]2�Xc�A�*

loss�ln>�R�       �	��2�Xc�A�*

loss�4*=@��       �	{�3�Xc�A�*

loss��=.d        �	z4�Xc�A�*

loss�R�=�m�       �	-�4�Xc�A�*

loss���=��Z       �	�D5�Xc�A�*

loss�(>�T�       �	{�5�Xc�A�*

loss]��=1@�       �	�x6�Xc�A�*

loss2@#=s]�R       �	�7�Xc�A�*

lossۆ�<<��k       �	��7�Xc�A�*

loss�Y>��B�       �	6Y8�Xc�A�*

loss�r�=,��       �	��8�Xc�A�*

loss��=����       �	��9�Xc�A�*

loss�_=���       �	�.:�Xc�A�*

loss���=����       �	��:�Xc�A�*

loss��=^�       �	��;�Xc�A�*

loss�b=7E�       �	��<�Xc�A�*

losso�>~}��       �	4=�Xc�A�*

lossMm,>l��       �	�5>�Xc�A�*

loss�q�=	�,z       �	��>�Xc�A�*

losskE>(���       �	Ot?�Xc�A�*

lossc0�<��       �	�@�Xc�A�*

loss�<q=��O       �	�@�Xc�A�*

lossFˇ=r ��       �	�fA�Xc�A�*

loss$��=W���       �	��A�Xc�A�*

loss�4�=�b4i       �	�B�Xc�A�*

losst��=5B1B       �	-ZC�Xc�A�*

loss.C>�+��       �	��C�Xc�A�*

loss�=;Ǜ�       �	��D�Xc�A�*

loss$��=���       �	�>E�Xc�A�*

lossVn�=QS	�       �	��E�Xc�A�*

lossȿ�=Q�?Y       �	ōF�Xc�A�*

loss�b�=���       �	Q/G�Xc�A�*

loss��=Aq0�       �	7�G�Xc�A�*

loss��>>�#�       �	bfH�Xc�A�*

loss���=�G��       �	�I�Xc�A�*

lossF��=�*��       �	�I�Xc�A�*

loss(w�=gծ�       �	�QJ�Xc�A�*

loss{�A>�:t       �	P�J�Xc�A�*

loss���=|p&�       �	��K�Xc�A�*

loss� '>�*�       �	�<L�Xc�A�*

loss�<aD��       �	
�L�Xc�A�*

loss�=�Ak       �	:yM�Xc�A�*

lossi)>^`�	       �	�N�Xc�A�*

loss|P�=���~       �	�N�Xc�A�*

loss�5�=l�N       �	,O�Xc�A�*

lossT>��5       �	 P�Xc�A�*

loss_��=��d�       �	�Q�Xc�A�*

loss��E>\M"       �	��Q�Xc�A�*

loss�L�=F���       �	�S�Xc�A�*

loss��%>t�3�       �	>BT�Xc�A�*

lossn��=��{       �	n�T�Xc�A�*

loss�!�=��o       �	��U�Xc�A�*

lossD�!=u��:       �	�aV�Xc�A�*

loss�o�=H�*       �	�W�Xc�A�*

loss��=]��v       �	�W�Xc�A�*

loss<%`=J�`       �	��X�Xc�A�*

loss�5<=���       �	��Y�Xc�A�*

loss�<=6��       �	t%Z�Xc�A�*

loss��=a8��       �	��[�Xc�A�*

loss�=߼�       �	�\�Xc�A�*

lossw9=1z�{       �	��]�Xc�A�*

lossʲ�<n���       �	��^�Xc�A�*

lossc]4<�o�       �	^0_�Xc�A�*

loss�Q=}J٤       �	�_�Xc�A�*

loss��
>GsBM       �	Ou`�Xc�A�*

loss�J�=���       �	�a�Xc�A�*

loss�i�=,?Q4       �	��a�Xc�A�*

loss&h!>$�V       �	�Qb�Xc�A�*

loss?�b<Q��|       �	��b�Xc�A�*

lossh�=�pA       �	U�c�Xc�A�*

lossh>�Ύ]       �	�*d�Xc�A�*

loss1B�=مH       �	*�d�Xc�A�*

loss��=3��!       �	Vbe�Xc�A�*

lossE��=(�'       �	��e�Xc�A�*

loss���=�'@t       �	�f�Xc�A�*

loss�=y~�       �	�Cg�Xc�A�*

loss�tU=._/       �	��g�Xc�A�*

loss���=$��       �	rh�Xc�A�*

lossT�>���       �	|i�Xc�A�*

loss�\�=���z       �	3�i�Xc�A�*

loss�׃=Z6�2       �	�jj�Xc�A�*

loss��G=���       �	qk�Xc�A�*

loss�4=�t�       �	J�k�Xc�A�*

lossn`u=�˖       �	U/l�Xc�A�*

loss�i�<hֱ       �	��l�Xc�A�*

lossb�<����       �	J�m�Xc�A�*

loss�#�<�x�8       �	�8n�Xc�A�*

loss�b>�P       �	s�n�Xc�A�*

lossF0�=�ƍ       �	Ouo�Xc�A�*

loss�4=H��       �	�p�Xc�A�*

loss���=>���       �	��p�Xc�A�*

loss�=��`b       �	��q�Xc�A�*

lossx�>��E       �	w/r�Xc�A�*

loss���<�7t       �	��r�Xc�A�*

loss >��       �	gs�Xc�A�*

loss�ib=�
       �	��s�Xc�A�*

loss���=��@�       �	_�t�Xc�A�*

lossJBE>��<       �	Q0u�Xc�A�*

lossJ`�<OK�       �	�u�Xc�A�*

loss��0<�8:F       �	�ev�Xc�A�*

loss.�<�       �	�Iw�Xc�A�*

loss�&�=.URN       �	��w�Xc�A�*

loss_=�Ȥ       �	^�x�Xc�A�*

loss��C>�Ҧ�       �	W$y�Xc�A�*

lossz@j>Iy�       �	#�y�Xc�A�*

lossR^�=P���       �	�\z�Xc�A�*

loss,#=�s4a       �	��z�Xc�A�*

loss���=�/�D       �	�{�Xc�A�*

loss��=����       �	=d|�Xc�A�*

loss�d4=so�       �	:}�Xc�A�*

loss���<�p       �	M�}�Xc�A�*

loss	��=�N��       �	�;~�Xc�A�*

loss���=�a0;       �	��~�Xc�A�*

lossˋ>��q�       �	�t�Xc�A�*

loss��|= ��       �	���Xc�A�*

loss�+�=`�'�       �	)���Xc�A�*

loss��C={�(       �	M��Xc�A�*

lossѫe=�##       �	1��Xc�A�*

lossr(=l�d8       �	����Xc�A�*

loss<�='^�{       �	)[��Xc�A�*

loss}n�=K�;h       �	���Xc�A�*

loss|%=��       �	���Xc�A�*

loss��=��       �	N*��Xc�A�*

loss�_�=sh�       �	eŅ�Xc�A�*

lossqi�=}�R9       �	Uk��Xc�A�*

loss���<9��       �	���Xc�A�*

loss���=N1�       �	Ԛ��Xc�A�*

lossa�1=B��}       �	S<��Xc�A�*

loss��=��l�       �	L݈�Xc�A�*

loss��=:�?�       �	�}��Xc�A�*

loss��=Q�b6       �	���Xc�A�*

lossn��<!�Kb       �	����Xc�A�*

loss��>���       �	�R��Xc�A�*

lossr��=�G�
       �	��Xc�A�*

loss+��<+���       �	�Ռ�Xc�A�*

lossQ�E=�Z��       �	�s��Xc�A�*

loss�q�=Lx�2       �	�Q��Xc�A�*

loss���=%��       �	S>��Xc�A�*

loss�5=���       �	�ݏ�Xc�A�*

loss*v�=���       �	���Xc�A�*

lossV�=37�2       �	W'��Xc�A�*

lossd8�<���!       �	̖��Xc�A�*

loss�=��j       �	����Xc�A�*

loss$O8=g�X       �	�V��Xc�A�*

lossq}�<��1       �	D���Xc�A�*

lossߍ�=G�F�       �	˕�Xc�A�*

loss���<�˟=       �	�Ö�Xc�A�*

loss�K=h�A       �	����Xc�A�*

lossa14=XA�       �	!<��Xc�A�*

loss5�<��V       �	f���Xc�A�*

loss�J=��       �	���Xc�A�*

lossJL�=��-�       �	�B��Xc�A�*

loss#n>�$       �	�&��Xc�A�*

loss*��<KՃ+       �	����Xc�A�*

loss3�< o��       �	�U��Xc�A�*

loss1�#=�v�9       �	�O��Xc�A�*

lossA��<]y��       �	Ll��Xc�A�*

loss0�:��<�       �	p`��Xc�A�*

lossx�<�[�o       �	6��Xc�A�*

lossoeG<fT�&       �	�ԡ�Xc�A�*

loss�;�;��       �	W���Xc�A�*

loss��<�^�       �	ۊ��Xc�A�*

loss\�="+�_       �	�N��Xc�A�*

loss�u=�S4l       �	��Xc�A�*

loss��;��       �	�ե�Xc�A�*

lossT��:���       �	ɦ�Xc�A�*

loss�݀<�+       �	ˁ��Xc�A�*

lossR^=~ �       �	�Ȩ�Xc�A�*

lossݬ�=K.(       �	y��Xc�A�*

lossF1=�ՙ'       �	�O��Xc�A�*

lossō�:��"0       �	+K��Xc�A�*

loss6>=\��[       �	N��Xc�A�*

loss��>�'�g       �	�Ь�Xc�A�*

loss�*F<<줯       �	4g��Xc�A�*

loss�>n1�^       �	3��Xc�A�*

lossh�I=l��       �	�®�Xc�A�	*

loss.]�=Dx��       �	�k��Xc�A�	*

loss�=֬�       �	����Xc�A�	*

loss(a=��0       �	����Xc�A�	*

lossb�	>�ۮ       �	By��Xc�A�	*

lossҾe=�A�~       �	F#��Xc�A�	*

lossE�>Ö�e       �	����Xc�A�	*

loss�z>�pa�       �	[���Xc�A�	*

loss�ޙ=+'4?       �	C��Xc�A�	*

loss�}�=�q       �	���Xc�A�	*

loss�6 >�E�H       �	J���Xc�A�	*

losshk�=�b�       �	xG��Xc�A�	*

loss>���       �	����Xc�A�	*

loss�5�=�l|e       �	����Xc�A�	*

loss�A=�]�       �	G��Xc�A�	*

losslӜ=���       �	Qi��Xc�A�	*

loss�Ǡ=X薷       �	���Xc�A�	*

loss5��=��y       �	ȼ�Xc�A�	*

loss!�<
�3       �	�|��Xc�A�	*

loss�G2=%��Y       �	/��Xc�A�	*

loss��=�S��       �	d=��Xc�A�	*

loss��<�Ni       �	ٿ�Xc�A�	*

loss(��<h#��       �	�~��Xc�A�	*

loss*S<��'�       �	"��Xc�A�	*

lossl@=/� H       �	2���Xc�A�	*

lossر�=�i��       �	�k��Xc�A�	*

loss��>]�x�       �	|��Xc�A�	*

losss��=F]�       �	���Xc�A�	*

loss$<���       �	�K��Xc�A�	*

loss��.=ۡ*�       �	G���Xc�A�	*

lossqx7=o˧l       �	y���Xc�A�	*

loss�>;����       �	�2��Xc�A�	*

loss�/5=7Rt)       �	����Xc�A�	*

loss�=I�$       �	f���Xc�A�	*

lossA3=�I       �	g*��Xc�A�	*

loss��= a�       �	����Xc�A�	*

lossx�>�zN�       �	\w��Xc�A�	*

lossd�X=�i<       �	���Xc�A�	*

loss�u\<Rh�5       �	E���Xc�A�	*

loss:�<��i/       �	�X��Xc�A�	*

loss�3=��O       �	0���Xc�A�	*

lossA�
=��6       �	����Xc�A�	*

loss_�=0�y�       �	 *��Xc�A�	*

loss��=�1Y       �	m���Xc�A�	*

loss�	>�p��       �	�R��Xc�A�	*

loss%h=9f        �	����Xc�A�	*

lossea�=#a&       �	���Xc�A�	*

lossi��<$�?x       �	�7��Xc�A�	*

lossZ��=b�s       �	����Xc�A�	*

lossF��=S�4V       �	Su��Xc�A�	*

loss�>?0�       �	0��Xc�A�	*

loss� &>��       �	!���Xc�A�	*

loss�E�=Ϗ#=       �	�S��Xc�A�	*

loss7ō=Oa)�       �	x���Xc�A�	*

lossF�=���*       �	ɬ��Xc�A�	*

loss1�=P$�(       �	<���Xc�A�	*

loss�x[=��R       �	+Q��Xc�A�	*

lossYE>GYn       �	���Xc�A�	*

loss�D�=���       �	����Xc�A�	*

lossx�=Wǌ]       �	Dm��Xc�A�	*

lossE"�=�ަ�       �	V)��Xc�A�	*

loss��L>r@c�       �	2���Xc�A�	*

loss�X�=_J�8       �	�e��Xc�A�	*

loss���=Z=�       �	�-��Xc�A�	*

loss� >�Yg       �	����Xc�A�	*

loss4�<��q�       �	�i��Xc�A�	*

loss���<����       �	 ��Xc�A�	*

lossFjy= f�l       �	Ǻ��Xc�A�	*

loss86>��       �	�Z��Xc�A�	*

loss�D�=(���       �	���Xc�A�	*

loss��>��:       �	����Xc�A�	*

loss�7�=����       �	|B��Xc�A�	*

lossQȵ=Y�%       �	����Xc�A�	*

loss��~=6z$�       �	4���Xc�A�	*

loss^.=�zx�       �	��Xc�A�	*

lossO�?=�c��       �	����Xc�A�	*

loss$>=�\       �	qr��Xc�A�	*

loss;y=a��\       �	��Xc�A�	*

loss�<K=��"       �	����Xc�A�	*

loss^�=���       �	^I �Xc�A�	*

loss�G�=���q       �	�� �Xc�A�	*

loss:��=��g_       �	���Xc�A�	*

loss>81�       �	�S�Xc�A�	*

loss=:=�A��       �	l��Xc�A�	*

loss��=��b       �	��Xc�A�	*

loss3��<\w�       �	�"�Xc�A�	*

lossQ��=�q3       �	���Xc�A�	*

lossT:>g��
       �	�o�Xc�A�	*

loss��~=��v:       �	�k�Xc�A�	*

loss�?H=��%`       �	�Xc�A�	*

loss�'�=��       �	
��Xc�A�	*

lossD�=)M�       �	�<�Xc�A�	*

lossҰy=��       �	���Xc�A�	*

loss�w=����       �	%�	�Xc�A�	*

lossq�>nã�       �	=+
�Xc�A�	*

loss�&=xi~       �	 �
�Xc�A�	*

loss�>�=9ba�       �	�h�Xc�A�	*

loss�^=��U       �	��Xc�A�	*

loss��<D�l       �	���Xc�A�	*

loss;d�<�>�       �	�.�Xc�A�	*

loss�˓=��}�       �	T��Xc�A�	*

loss��<m�6       �	%]�Xc�A�	*

loss�(s>�*/       �	��Xc�A�	*

lossQ;=�Ӿ%       �	q��Xc�A�	*

lossy 	<D%�       �	a�Xc�A�	*

loss�%$<"8j�       �	m��Xc�A�	*

loss|��;	3�       �	��Xc�A�	*

loss�VP=dm��       �	#2�Xc�A�	*

loss���=EL       �	���Xc�A�	*

lossAQP>f[�o       �	�u�Xc�A�	*

loss�!>�}lo       �	��Xc�A�	*

loss�R�<�Xy       �	e�Xc�A�	*

loss��=ћ��       �	!?�Xc�A�	*

lossJ�=m��       �	��Xc�A�	*

lossS�d=�s�       �	���Xc�A�	*

lossE�=�R�       �	�0�Xc�A�	*

loss]��=mQo%       �	=��Xc�A�	*

loss1�=���       �	ע�Xc�A�	*

lossJK�=��A�       �	�s�Xc�A�	*

loss
�@=�JQ�       �	J�Xc�A�	*

loss)XF=�z�       �	��Xc�A�	*

loss��@=�P�M       �	�r�Xc�A�	*

lossc��=)�$a       �	9�Xc�A�	*

loss�=�F�	       �	�Xc�A�	*

loss�l�<q��       �	���Xc�A�	*

loss{�7>dvwI       �	ؼ �Xc�A�	*

lossf�=��Լ       �	�r!�Xc�A�	*

loss l�<ϊU�       �	}"�Xc�A�	*

lossÐ�=U�u�       �	e�"�Xc�A�
*

loss��K=��$�       �	Dj#�Xc�A�
*

lossּ�=JtB       �	(*$�Xc�A�
*

loss��=!�w�       �	��$�Xc�A�
*

loss�w<����       �	;�%�Xc�A�
*

loss��=rX       �	��&�Xc�A�
*

loss��/=�w;�       �	�G'�Xc�A�
*

loss�D�<����       �	��'�Xc�A�
*

loss��<���       �	Tr(�Xc�A�
*

loss��=��4U       �	�/)�Xc�A�
*

loss�>=4w'       �	�!*�Xc�A�
*

loss��=^�0[       �	�*�Xc�A�
*

lossB�=K$k�       �	v�+�Xc�A�
*

loss��=�O�       �	��,�Xc�A�
*

lossi~�=��>�       �	O-�Xc�A�
*

loss�.>�蔣       �	/�-�Xc�A�
*

lossȤ�<f��       �	��.�Xc�A�
*

loss�/�<K�w�       �	?�/�Xc�A�
*

loss�Ҍ=g��d       �	�70�Xc�A�
*

loss�G>	��       �	��0�Xc�A�
*

loss�V\=>��       �	k�1�Xc�A�
*

loss�B=<�d�       �	 62�Xc�A�
*

lossT�=��y�       �	��2�Xc�A�
*

losso�<˴��       �	�n3�Xc�A�
*

loss��<���h       �		4�Xc�A�
*

loss���<IpIM       �	��4�Xc�A�
*

loss͋�=���T       �	J5�Xc�A�
*

lossi�<��m^       �	�6�Xc�A�
*

loss��=-�<|       �	��6�Xc�A�
*

loss��B>�ָv       �	H7�Xc�A�
*

loss�=��b       �	��7�Xc�A�
*

loss���=PV�       �	��8�Xc�A�
*

loss?�@<]�޸       �	mX9�Xc�A�
*

loss�&=�>�$       �	�c:�Xc�A�
*

loss�� =��Is       �	�;�Xc�A�
*

loss��8=�l�9       �	��;�Xc�A�
*

losso�D=f���       �	�@<�Xc�A�
*

loss�W=�®       �	��<�Xc�A�
*

loss���=����       �	�n=�Xc�A�
*

loss�I>q���       �	�>�Xc�A�
*

loss��=��C       �	Ԝ>�Xc�A�
*

lossC�%=J@ �       �	v5?�Xc�A�
*

loss���<�V       �	:�?�Xc�A�
*

loss��n=��3{       �	�h@�Xc�A�
*

loss4Q�=��Ԍ       �	5A�Xc�A�
*

loss! K<����       �	i�A�Xc�A�
*

loss{}=r�Jc       �	�`B�Xc�A�
*

loss���=iS�       �	��B�Xc�A�
*

lossŰf=\Ґ
       �	q�C�Xc�A�
*

loss�2U=1c��       �	 *D�Xc�A�
*

loss�@�=��       �	��D�Xc�A�
*

loss�zo=��Jw       �	bE�Xc�A�
*

loss⩠=̡��       �	`F�Xc�A�
*

loss.y�<�e��       �	@�F�Xc�A�
*

loss�)<�'�       �	�oG�Xc�A�
*

loss�׻=����       �	H�Xc�A�
*

loss��b=��Oz       �	E�H�Xc�A�
*

loss�`=eAY       �	�5I�Xc�A�
*

losso�}<ĵ��       �	}�I�Xc�A�
*

loss7��<֧(X       �	�cJ�Xc�A�
*

lossi��<Tő�       �	��J�Xc�A�
*

lossɍ<�`vE       �	1�K�Xc�A�
*

loss��<=�e       �	,+L�Xc�A�
*

loss[�O=��:8       �	��L�Xc�A�
*

loss7�=M�o       �	.VM�Xc�A�
*

loss�}K=E�4�       �	D�M�Xc�A�
*

lossX��<�=@I       �	ڏN�Xc�A�
*

lossf1�<��^       �	�!O�Xc�A�
*

loss���<��1�       �	=�O�Xc�A�
*

loss��<�m7       �	�nP�Xc�A�
*

loss�ݓ=J-�       �	yQ�Xc�A�
*

loss��W=�:       �	R�Q�Xc�A�
*

loss�߳=.C,       �	v6R�Xc�A�
*

loss��@=/7       �	Q�R�Xc�A�
*

loss6��=���       �	[{S�Xc�A�
*

loss�'�=����       �	@T�Xc�A�
*

loss[R5;<_a?       �	%�T�Xc�A�
*

loss�=HMG       �	kdU�Xc�A�
*

loss���=a�	z       �	oV�Xc�A�
*

lossZO�=��rI       �	�V�Xc�A�
*

loss��)<v���       �	�GW�Xc�A�
*

loss��`=�>��       �	��W�Xc�A�
*

loss�&=�(�E       �	�X�Xc�A�
*

loss!��=n�K       �	^+Y�Xc�A�
*

lossE<`ZT�       �	6�Y�Xc�A�
*

loss��B=�{W�       �	��Z�Xc�A�
*

lossh��<��h       �	'.[�Xc�A�
*

loss���<ʍ�D       �	 �[�Xc�A�
*

loss[�=R(u�       �	�g\�Xc�A�
*

loss���<-`�j       �	��\�Xc�A�
*

loss$�N=�sL       �	��]�Xc�A�
*

loss�UF=/��9       �	�6^�Xc�A�
*

loss�6<����       �	��^�Xc�A�
*

lossZ�v=����       �	q_�Xc�A�
*

lossZ�=����       �	�`�Xc�A�
*

lossEt�;f)D       �	Ü`�Xc�A�
*

lossa�v=�BFh       �	�6a�Xc�A�
*

loss$X	>�B��       �	h�a�Xc�A�
*

loss�:�=� &       �	�jb�Xc�A�
*

loss�f�=��-       �	�c�Xc�A�
*

loss���<���       �	��c�Xc�A�
*

loss�W<�⬑       �	Qd�Xc�A�
*

loss�"�=��Э       �	�d�Xc�A�
*

loss1 G=��?	       �	ސe�Xc�A�
*

loss�9<6 ��       �	nlf�Xc�A�
*

loss��	<泧,       �	Gg�Xc�A�
*

loss�t�<%�ܫ       �	��g�Xc�A�
*

lossl��<�!�       �	�Mh�Xc�A�
*

lossQb�<���2       �	P�h�Xc�A�
*

lossu�=���w       �	�i�Xc�A�
*

loss�OW=:�G�       �	@1j�Xc�A�
*

loss�y=���       �	��j�Xc�A�
*

lossOf=2�T�       �	Hlk�Xc�A�
*

loss##=�h��       �	��l�Xc�A�
*

loss?=��v       �	qm�Xc�A�
*

lossA̫<�Ҕ�       �	Jn�Xc�A�
*

loss���<i5��       �	o�Xc�A�
*

loss�D= �O]       �	w�o�Xc�A�
*

loss�q�<�[��       �	RCp�Xc�A�
*

loss)�6>g�       �	c�p�Xc�A�
*

lossa�=t$v�       �	��q�Xc�A�
*

lossV"g=6��       �	%r�Xc�A�
*

loss�;�<#���       �	h�r�Xc�A�
*

loss�{=��p       �	�cs�Xc�A�
*

lossA�=��y|       �	%;t�Xc�A�
*

lossS��<%��)       �	�u�Xc�A�
*

lossmi�<���       �	��u�Xc�A�
*

loss�(�= ��&       �	�v�Xc�A�
*

lossl�=�៹       �	�)w�Xc�A�*

loss��=�7�       �	o�w�Xc�A�*

loss�>:z�:       �	1_x�Xc�A�*

loss$��=�?`       �	��x�Xc�A�*

loss1��=6���       �	��y�Xc�A�*

loss8q<b�<�       �	�Az�Xc�A�*

loss��<X^�d       �	4�z�Xc�A�*

lossZs�=���       �	3l{�Xc�A�*

lossH�2=LM,_       �	�|�Xc�A�*

loss�Q�>{��       �	��|�Xc�A�*

loss�>b�PA       �	<0}�Xc�A�*

loss�h>�;$       �	C�}�Xc�A�*

loss��=ۋ��       �	l_~�Xc�A�*

lossi��=��:�       �	�6�Xc�A�*

loss�+�<����       �	���Xc�A�*

loss��=Y3       �	>{��Xc�A�*

lossi�=7
�2       �	���Xc�A�*

lossͧ�<p~�U       �	p���Xc�A�*

loss�=���       �	�R��Xc�A�*

loss��=�D        �	���Xc�A�*

loss ��=E��u       �	����Xc�A�*

loss��F=. |�       �	['��Xc�A�*

loss���<�*       �	����Xc�A�*

loss��	=�R�u       �	?R��Xc�A�*

loss�j�=%�dd       �	i��Xc�A�*

loss/=���       �	Kv��Xc�A�*

loss��=��]�       �	���Xc�A�*

loss��
=��k�       �	u���Xc�A�*

lossͭ=~��P       �	cD��Xc�A�*

loss߅�=��       �	�܈�Xc�A�*

loss�]=��q�       �	j���Xc�A�*

loss6�=be�:       �	L8��Xc�A�*

loss���<��e       �	�ʊ�Xc�A�*

loss7�<���       �	Nd��Xc�A�*

loss��=�F�       �	���Xc�A�*

loss�C�=�Zx       �	����Xc�A�*

loss��@=�y.�       �	�E��Xc�A�*

loss���=i��*       �	���Xc�A�*

loss�Fh=�HT       �	(|��Xc�A�*

loss�F�=A�U       �	���Xc�A�*

loss�Y=����       �	N���Xc�A�*

loss��=e��       �	�G��Xc�A�*

loss��<O��T       �	�ݐ�Xc�A�*

loss�e!=�{x*       �	�s��Xc�A�*

lossRg�=��ͳ       �	�f��Xc�A�*

loss8�O=���       �	����Xc�A�*

losse��;��b�       �	nޓ�Xc�A�*

loss×�<J�       �	�u��Xc�A�*

lossf@=:V�0       �	n��Xc�A�*

loss��;=}�+#       �	�ƕ�Xc�A�*

loss���< ��Y       �	�Xc�A�*

loss��X=4b�       �	�E��Xc�A�*

loss({�=w�=�       �	�i��Xc�A�*

lossd��=6#P       �	���Xc�A�*

loss��=I�,.       �	����Xc�A�*

loss��=�X0�       �	�5��Xc�A�*

lossTrC=A	Rc       �	�֚�Xc�A�*

loss,��=�?�       �	y��Xc�A�*

lossZ�<���:       �	0��Xc�A�*

loss��	>���q       �	q���Xc�A�*

loss�z=�G͞       �	�I��Xc�A�*

loss�q>}��g       �	���Xc�A�*

loss���=/F       �	�v��Xc�A�*

loss=�<pV`�       �	mV��Xc�A�*

loss���<�5�s       �	>��Xc�A�*

loss�!�='�7       �	����Xc�A�*

loss�q=�       �	+��Xc�A�*

loss�X>>73r�       �	D���Xc�A�*

loss�=�=����       �	�>��Xc�A�*

loss\�<�x��       �	��Xc�A�*

lossڡu==̧O       �	|��Xc�A�*

loss�O>�<�       �	YR��Xc�A�*

loss?��=&%�       �	��Xc�A�*

lossZ��<���       �	G���Xc�A�*

loss��{<���Y       �	4G��Xc�A�*

loss]�=z1�       �	����Xc�A�*

loss7>=�&P_       �	����Xc�A�*

loss�P/=�R�       �	e6��Xc�A�*

lossf�=���       �	 Ѩ�Xc�A�*

lossx��=U�$�       �	[a��Xc�A�*

loss�J�<{��       �	����Xc�A�*

lossfZ=�$]       �	�ͪ�Xc�A�*

loss&�=�ؖ       �	+k��Xc�A�*

loss�� <�	.       �	W��Xc�A�*

lossnB�=�߲0       �	���Xc�A�*

loss�~;s�4�       �	���Xc�A�*

lossD4X< �0       �	�:��Xc�A�*

loss$��<�c�       �	(Ӯ�Xc�A�*

loss)�=�M+/       �	�j��Xc�A�*

loss�S=�nO       �	����Xc�A�*

loss*֥=oo�`       �	���Xc�A�*

loss�=�b��       �	*9��Xc�A�*

loss�ie=�h��       �	���Xc�A�*

loss�#=�Þg       �	ˠ��Xc�A�*

loss���<z0�[       �	d<��Xc�A�*

loss�"�<�>��       �	�س�Xc�A�*

loss�X�=��"       �	�w��Xc�A�*

loss]2�=_�"       �	ٵ��Xc�A�*

loss�6�=w�[       �	]P��Xc�A�*

lossfQ�;ܛ��       �	2��Xc�A�*

loss�#=G��       �	<���Xc�A�*

lossC�R=.[�J       �	zr��Xc�A�*

loss��\=�X�       �	k��Xc�A�*

loss��F=0�#       �	���Xc�A�*

loss��t=���       �	1C��Xc�A�*

loss��=tR4T       �	kֺ�Xc�A�*

loss��(=��)�       �	�l��Xc�A�*

loss&((=f��q       �	\���Xc�A�*

loss�L=-
�       �	����Xc�A�*

lossϔu=8<�       �	4��Xc�A�*

loss�uS=FP��       �	W$��Xc�A�*

loss �c=���       �	/���Xc�A�*

loss��<b"�       �	yX��Xc�A�*

loss-H�<T�G�       �	���Xc�A�*

loss@~->��.>       �	ܡ��Xc�A�*

loss�\=��Y�       �	m9��Xc�A�*

loss�Y=��       �	����Xc�A�*

lossK?=�r�=       �	vm��Xc�A�*

loss��=���       �	k+��Xc�A�*

loss��=c$9�       �	���Xc�A�*

losss
=��s�       �	�O��Xc�A�*

loss9��=��       �	���Xc�A�*

loss�A$>>.@�       �	����Xc�A�*

lossvJ�<P&o�       �	ly��Xc�A�*

loss�T�<�!
x       �	��Xc�A�*

loss|>V<U���       �	���Xc�A�*

lossx�<�9�D       �	�P��Xc�A�*

lossF7t=u�S       �	����Xc�A�*

loss�i�<}���       �	�{��Xc�A�*

loss�m�=��       �	���Xc�A�*

lossk�=���       �	���Xc�A�*

loss&�s=o��!       �	�G��Xc�A�*

loss
�R;���R       �	����Xc�A�*

loss) �<V�Xs       �	����Xc�A�*

loss��<���(       �	C��Xc�A�*

lossx�=�C
�       �	���Xc�A�*

lossM]�<��'�       �	�J��Xc�A�*

lossa�>�"3!       �	����Xc�A�*

loss���=� �       �	�z��Xc�A�*

loss�u'=X�0�       �	E��Xc�A�*

loss�>����       �	����Xc�A�*

losslh=�Ub       �	�<��Xc�A�*

loss6}=wr�       �	����Xc�A�*

loss���<�+�       �	�o��Xc�A�*

loss�f=��u�       �	���Xc�A�*

loss�0�=��jv       �		���Xc�A�*

loss,/�<��F       �	�9��Xc�A�*

lossc�=Yܳ       �	����Xc�A�*

lossc�<GLk�       �	���Xc�A�*

loss6�U<�Iec       �	�'��Xc�A�*

loss��<͋��       �	K���Xc�A�*

lossvUg=��Ab       �	����Xc�A�*

loss�=��;       �	u��Xc�A�*

loss�?=�оe       �	F���Xc�A�*

loss�D=b	U3       �	�Z��Xc�A�*

lossM.=��F�       �	� ��Xc�A�*

loss{��=��v�       �	����Xc�A�*

lossW��=���       �	�P��Xc�A�*

loss{�<�=�       �	V���Xc�A�*

lossAU�=0���       �	̗��Xc�A�*

loss��<��q+       �	;9��Xc�A�*

loss�`<�:�       �	4���Xc�A�*

loss6��=ac�       �	�w��Xc�A�*

lossd|==�       �	��Xc�A�*

loss�G�<�V��       �	�<��Xc�A�*

lossxG�<��dd       �	x���Xc�A�*

lossh[=8        �	.t��Xc�A�*

loss��<�.�x       �	���Xc�A�*

loss(�3=*c�       �	�&��Xc�A�*

loss��=od��       �	����Xc�A�*

loss��~=�|�=       �	zm��Xc�A�*

losstaC=�J�       �	\��Xc�A�*

loss�P>�/l       �	_���Xc�A�*

lossh��<i��j       �	*7��Xc�A�*

loss��=�s�U       �	���Xc�A�*

loss��4>��"       �	�{��Xc�A�*

loss��"=�"�       �	<��Xc�A�*

loss���;Ɂ�)       �	���Xc�A�*

lossQT�=j�d�       �	�b��Xc�A�*

lossaa�=Odb       �	M���Xc�A�*

loss��=�n�.       �	$���Xc�A�*

lossz�=ު#�       �	�}��Xc�A�*

loss2+Z<�@U�       �	;:��Xc�A�*

lossA��=�nG�       �	����Xc�A�*

loss�T�=�T�       �	�y��Xc�A�*

losspc�=�F��       �	���Xc�A�*

lossV�=�/       �	F���Xc�A�*

loss4�W=\3�       �	LS��Xc�A�*

loss��W>vAh       �	 ���Xc�A�*

loss%U�<�A{8       �	H���Xc�A�*

lossT�<E��       �	%��Xc�A�*

lossi��<�~�g       �	����Xc�A�*

loss_rI=|�!|       �	r���Xc�A�*

lossU<?��v       �	�?��Xc�A�*

lossli=@�5       �	����Xc�A�*

lossJ�>06       �	�k��Xc�A�*

loss�=TjY�       �	v���Xc�A�*

loss(��<���&       �	���Xc�A�*

loss�=[��       �	�*��Xc�A�*

losseƐ=̒��       �	e���Xc�A�*

lossđ>�Y�`       �	V��Xc�A�*

loss9=��_�       �	����Xc�A�*

loss�ɪ=k���       �	b���Xc�A�*

loss��#=�p��       �	���Xc�A�*

loss��=	Zk       �	>���Xc�A�*

loss�b`<x��A       �	qZ��Xc�A�*

lossX�>��v�       �	{���Xc�A�*

loss=A�=���       �	���Xc�A�*

loss�r>�9$�       �	q<��Xc�A�*

loss��t<���       �	o���Xc�A�*

loss���<���\       �	�l��Xc�A�*

lossc=Ǥ       �	� �Xc�A�*

loss��=�s[�       �	�� �Xc�A�*

loss���<:��       �	wI�Xc�A�*

loss���=e���       �		��Xc�A�*

loss�Y�<�+�       �	\s�Xc�A�*

lossȈ�=��.�       �	Q�Xc�A�*

loss�Ʊ=����       �	d��Xc�A�*

loss���=+�8V       �	�N�Xc�A�*

loss���<K��X       �	��Xc�A�*

loss$K�<(�TU       �	%��Xc�A�*

loss�M�<B�i6       �	�/�Xc�A�*

lossC��<�@�       �	 ��Xc�A�*

lossN��<��       �	�Y�Xc�A�*

lossaf'=�Os       �	5��Xc�A�*

loss���;i"��       �	���Xc�A�*

loss`�1=å%�       �	;	�Xc�A�*

lossv�y=��CM       �	��	�Xc�A�*

loss�o�<C�	T       �	bH
�Xc�A�*

loss�F�<�D�@       �	��
�Xc�A�*

loss��V<<ҷs       �	yy�Xc�A�*

loss�;H?��       �	n�Xc�A�*

lossi�=�>.       �	���Xc�A�*

lossQOk>P�       �	YP�Xc�A�*

loss���=�$�       �	-��Xc�A�*

loss�0>�B��       �	��Xc�A�*

lossG2>��4       �	�*�Xc�A�*

loss ��<[�A        �	g��Xc�A�*

loss�yt=��E�       �	�k�Xc�A�*

loss�Z>���       �	��Xc�A�*

loss}|�<�	�<       �	���Xc�A�*

lossQ�=[")       �	�?�Xc�A�*

lossj"=�ͤv       �	3��Xc�A�*

lossz�j=��O�       �	ˁ�Xc�A�*

loss�ٌ<P�n.       �	�$�Xc�A�*

loss�c=v��v       �	7��Xc�A�*

lossM�=�i�P       �	f��Xc�A�*

loss
T�=��}       �	�W�Xc�A�*

loss�Ӫ=(U�       �	���Xc�A�*

loss��=��A       �	L��Xc�A�*

loss	�0=���f       �	>#�Xc�A�*

loss�� =�#�v       �	���Xc�A�*

loss&B�<���5       �	�a�Xc�A�*

loss�0�<mX^       �	d �Xc�A�*

loss8�<r��6       �	���Xc�A�*

loss�F�<q�;       �	�y�Xc�A�*

loss�K>��]]       �	��Xc�A�*

lossͫ�=p�\       �	��Xc�A�*

lossdK=	Ϳ       �	W{�Xc�A�*

loss�f=�y�       �	o�Xc�A�*

loss�v�=Q�d       �	&��Xc�A�*

loss�^�<�%�Q       �	�E�Xc�A�*

loss�}�;���%       �	���Xc�A�*

loss�eN>��5       �	�� �Xc�A�*

loss���<��       �	�!�Xc�A�*

loss� �=�еw       �	��!�Xc�A�*

loss��x=SP��       �	5]"�Xc�A�*

loss�7�<�K-�       �	Z#�Xc�A�*

loss[%=�jas       �	�#�Xc�A�*

loss�!=b�kk       �	��$�Xc�A�*

loss8�<=��       �	'J%�Xc�A�*

lossX<F`&)       �	l�%�Xc�A�*

loss��K>Z�       �	b�&�Xc�A�*

loss��=��6       �	�'�Xc�A�*

loss#�H=��g�       �	��'�Xc�A�*

loss�F�<Y41�       �	�T(�Xc�A�*

loss�j�=e�,       �	��(�Xc�A�*

loss�(>�9N%       �	��)�Xc�A�*

loss�v=�o�       �	�*�Xc�A�*

loss� =7��       �	"�*�Xc�A�*

loss�
�=$���       �	[+�Xc�A�*

loss�Qh<��G       �	{�+�Xc�A�*

losse>����       �	ڏ,�Xc�A�*

loss$u5=���       �	�B-�Xc�A�*

loss,�==V�       �	4�-�Xc�A�*

lossr{<aS�       �	�o.�Xc�A�*

loss��<��q�       �	�/�Xc�A�*

loss��W<�/       �	�/�Xc�A�*

lossx��<�n��       �	~Q0�Xc�A�*

loss�b>�Ջ�       �	��0�Xc�A�*

lossm��<bT�       �	Q�1�Xc�A�*

loss}�=Ĭ��       �	�2�Xc�A�*

lossȳ�=86@�       �	��2�Xc�A�*

loss#��=A*r�       �	!X3�Xc�A�*

loss�;`�Pn       �	e�3�Xc�A�*

loss؆�;6k�       �	��4�Xc�A�*

loss)��;��Y>       �	5F5�Xc�A�*

loss}-�=�"K1       �	u�5�Xc�A�*

lossXv=�y��       �	��6�Xc�A�*

loss�J�=�i       �	()7�Xc�A�*

lossJA�<̝��       �	��7�Xc�A�*

loss�]}=KXp�       �	�q8�Xc�A�*

lossӦS=���       �	=9�Xc�A�*

lossU�<��s�       �	�9�Xc�A�*

loss\��=���       �	�@:�Xc�A�*

loss�\9<
��       �	g�:�Xc�A�*

loss�!�=T�PP       �	�q;�Xc�A�*

loss*�N=6���       �	�<�Xc�A�*

loss�`�=5#j�       �	��<�Xc�A�*

loss�0�=}�       �	`=�Xc�A�*

loss~�="y6�       �	�.>�Xc�A�*

lossR�=A�y�       �	Q�>�Xc�A�*

loss��v=�*~�       �	�|?�Xc�A�*

loss�:
=��*Z       �	i@�Xc�A�*

loss�"=���       �	��@�Xc�A�*

loss�-1;��D�       �	�sA�Xc�A�*

loss��=a��c       �	�B�Xc�A�*

loss %�<��Re       �	�B�Xc�A�*

loss_�%<x9�       �	ZdC�Xc�A�*

loss3'0= ^�       �	:D�Xc�A�*

loss�wd=kZ�       �	ΩD�Xc�A�*

lossS�=�3       �	#LE�Xc�A�*

loss��s<z���       �	��E�Xc�A�*

loss2��<7��       �	f�F�Xc�A�*

lossSy=�g�       �	�!G�Xc�A�*

loss֔U<v�y       �	��G�Xc�A�*

loss��;K#�       �	�uH�Xc�A�*

loss�Q�<n��L       �	�I�Xc�A�*

loss���;F��       �	+�I�Xc�A�*

loss��7<&��       �	�oJ�Xc�A�*

loss�yf<���=       �	�3K�Xc�A�*

lossW1:�c�h       �	��K�Xc�A�*

lossP�<�u�       �	BvL�Xc�A�*

loss��;��]�       �	�M�Xc�A�*

loss#�:6@��       �	E�M�Xc�A�*

lossoQ;��       �	%]N�Xc�A�*

loss��D<l��.       �	Y�N�Xc�A�*

loss��=-]E�       �	[�O�Xc�A�*

loss��<�z?%       �	D1P�Xc�A�*

loss� ;y;7�       �	�P�Xc�A�*

losszlt<�W=�       �	�lQ�Xc�A�*

loss��r>]���       �	R�Xc�A�*

lossޭ<J��r       �	w�R�Xc�A�*

loss��u>��p       �	e7S�Xc�A�*

loss�T>o@�       �	�T�Xc�A�*

loss��=��Q�       �	˝T�Xc�A�*

losslCW=
��       �	v8U�Xc�A�*

loss���<x�/�       �	��U�Xc�A�*

loss�Ջ=��       �	$�V�Xc�A�*

loss�=�S}l       �	��W�Xc�A�*

lossN�<=���i       �	f2X�Xc�A�*

loss�:I=��?�       �	�X�Xc�A�*

loss��= 2�       �	��Y�Xc�A�*

loss���=�m<       �	jgZ�Xc�A�*

lossQ��=;E]       �	��Z�Xc�A�*

loss#2�=Fp�>       �	\�[�Xc�A�*

loss�ɓ=3�zk       �	P4\�Xc�A�*

loss�5="Є6       �	��\�Xc�A�*

loss�gK=�S�        �	jm]�Xc�A�*

loss�8`<�>�       �	�^�Xc�A�*

loss�;�=�y       �	��^�Xc�A�*

loss}R!=N�j$       �	oK_�Xc�A�*

lossMW�<��       �	X�_�Xc�A�*

loss� v=�mx�       �	dw`�Xc�A�*

loss�fM=`��G       �	�
a�Xc�A�*

loss7|�<��?�       �	r�a�Xc�A�*

lossrUR<n�6�       �	�9b�Xc�A�*

loss]�<�:�       �	�b�Xc�A�*

loss��=����       �	dc�Xc�A�*

loss8t=K N>       �	j�c�Xc�A�*

loss�j=j�V)       �	i�d�Xc�A�*

loss߄=��|       �	�_e�Xc�A�*

loss�2/<،�       �	��e�Xc�A�*

loss���=����       �	L�f�Xc�A�*

loss=�;K�2       �	#2g�Xc�A�*

loss���<���       �	u�g�Xc�A�*

loss�l�<lH't       �	�\h�Xc�A�*

lossW� =�I��       �	��h�Xc�A�*

loss��<;'�W       �	G�i�Xc�A�*

loss�C�=#��       �	�&j�Xc�A�*

loss?��="y�       �	��j�Xc�A�*

loss��=5V��       �	GXk�Xc�A�*

lossw��<˦�       �	��k�Xc�A�*

loss���=���S       �	y�l�Xc�A�*

loss�ܺ<:G�U       �	cm�Xc�A�*

loss]j=��v       �	X�m�Xc�A�*

loss�A�<u7��       �	��n�Xc�A�*

loss��g=�ȇ�       �	T9o�Xc�A�*

lossa��=3�n�       �	��o�Xc�A�*

lossM�?<סή       �	b�p�Xc�A�*

lossON=��)       �	Kq�Xc�A�*

loss�=̍�"       �	��q�Xc�A�*

lossV�(<e���       �	Qr�Xc�A�*

lossLʔ=k�B       �	���Xc�A�*

lossɊ==�`�       �	a���Xc�A�*

loss;��=��1       �	%@��Xc�A�*

lossi��=�n;�       �	Ԋ�Xc�A�*

loss;�Z=��       �	{k��Xc�A�*

loss��w=�s�       �	���Xc�A�*

loss3�=�iq       �	-���Xc�A�*

lossD��=�ݻ        �	x|��Xc�A�*

loss,X�=	       �	���Xc�A�*

loss�~�<

�6       �	פ��Xc�A�*

losshO=�ˣx       �	e7��Xc�A�*

loss�C<�N�/       �	y̏�Xc�A�*

loss}�U=��*M       �	Zd��Xc�A�*

loss�I<_(l�       �	����Xc�A�*

loss��=�^PR       �	ɏ��Xc�A�*

loss�sx=J�h       �	�%��Xc�A�*

loss��;��;�       �	���Xc�A�*

loss�]a=���f       �	@O��Xc�A�*

loss�pA<�(�       �	T��Xc�A�*

loss*��=�יn       �	����Xc�A�*

lossQ)=���       �	�ɕ�Xc�A�*

loss,)�=��k�       �	j��Xc�A�*

lossSw�<���       �	"U��Xc�A�*

loss��>񼾱       �	���Xc�A�*

loss�.W=�ݒ        �	�N��Xc�A�*

loss�<~FL�       �	<��Xc�A�*

loss��=t(�^       �	i���Xc�A�*

loss��<@A�       �	gb��Xc�A�*

loss8�=+h/       �	���Xc�A�*

loss��<`�^       �	�_��Xc�A�*

loss��=j5       �	�'��Xc�A�*

lossh\�=魂C       �	軞�Xc�A�*

lossu�<�Bb       �	n���Xc�A�*

loss�z->�+��       �	�7��Xc�A�*

lossq5�<�1^       �	���Xc�A�*

lossɋ=���       �	�ӡ�Xc�A�*

loss�j�<�]�.       �	qq��Xc�A�*

loss��=�˽       �	}��Xc�A�*

lossJ&r=n�       �	���Xc�A�*

loss�Y�=��#�       �	<��Xc�A�*

loss!�A=Fq�~       �	,ؤ�Xc�A�*

loss$�h=��       �	�o��Xc�A�*

losso	;<��<�       �	�	��Xc�A�*

loss��<��?       �	��Xc�A�*

loss��>=�.��       �	�B��Xc�A�*

loss��=�Eu$       �	�ӧ�Xc�A�*

loss �=����       �	Vf��Xc�A�*

loss�`=#�#@       �	����Xc�A�*

loss�1�=�cbo       �	���Xc�A�*

loss��z;�u;:       �	�0��Xc�A�*

loss��<pB��       �	˪�Xc�A�*

lossŨ=�8R       �	8f��Xc�A�*

loss�<�<8İ�       �	t��Xc�A�*

lossX�>�W�       �	����Xc�A�*

loss	�S=�uܒ       �	yX��Xc�A�*

loss3k�;�舮       �	����Xc�A�*

loss!�;l��       �	2���Xc�A�*

loss�;�       �	�V��Xc�A�*

lossz=���       �	Z��Xc�A�*

loss�;{=v8p       �	����Xc�A�*

lossL>�Ԟ�       �	}[��Xc�A�*

loss�/>�w��       �	K��Xc�A�*

loss?�;��       �	����Xc�A�*

loss��<=����       �	h��Xc�A�*

loss堸<�_Q�       �	9��Xc�A�*

lossw<�&y       �	´�Xc�A�*

loss;=;ޫ�       �	__��Xc�A�*

loss��=�Ř�       �	q��Xc�A�*

loss���=��       �	~���Xc�A�*

loss��t=@�.:       �	�W��Xc�A�*

losso�P=�sڳ       �	b���Xc�A�*

loss�Y=�/s�       �	c	��Xc�A�*

loss/�<�l       �	I���Xc�A�*

loss�_=�J*m       �	����Xc�A�*

loss���=m/��       �	E��Xc�A�*

loss�OJ=t;\�       �	|���Xc�A�*

lossa�*<�&]�       �	����Xc�A�*

loss�}X=U�
f       �	�<��Xc�A�*

loss�5 =�yg       �	h��Xc�A�*

loss�(=7�i       �	����Xc�A�*

loss=处A       �	�7��Xc�A�*

lossӆ=�s       �	���Xc�A�*

lossv��<גQ�       �	׉��Xc�A�*

lossm{<d��       �	&���Xc�A�*

loss��	=���       �	����Xc�A�*

loss�-�<�VHC       �	^I��Xc�A�*

loss_��<�Ts?       �	l���Xc�A�*

loss�޷<�ɢ       �	/���Xc�A�*

loss�6S=>f��       �	4��Xc�A�*

loss�L�<�vf       �	X���Xc�A�*

loss8i�=Dw       �	#���Xc�A�*

loss�0�=�8       �	�4��Xc�A�*

loss��=Y��       �	����Xc�A�*

loss�NV=>�       �	���Xc�A�*

lossߡ�=Sؔ�       �	�+��Xc�A�*

loss<��<`^S�       �	���Xc�A�*

lossxRk=H��       �	���Xc�A�*

loss�,#=9�79       �	�1��Xc�A�*

loss$�p=��Q3       �	����Xc�A�*

lossJ
=d�       �	����Xc�A�*

loss�Җ;�a�n       �	�&��Xc�A�*

loss�;̀��       �	����Xc�A�*

loss�4�=t=o       �	����Xc�A�*

loss�e(<���       �	a7��Xc�A�*

loss#�</H�Y       �	����Xc�A�*

loss�>���       �	X���Xc�A�*

loss�G<��Wb       �	_A��Xc�A�*

loss��=�u�       �	|���Xc�A�*

loss\��=<�       �	����Xc�A�*

lossO� =bK�       �	�?��Xc�A�*

loss��>�rߋ       �	����Xc�A�*

loss6��<Њ�g       �	#���Xc�A�*

lossn�{=b{L       �	iQ��Xc�A�*

loss-�;R( i       �	�d��Xc�A�*

loss��I=Lo�?       �	&��Xc�A�*

loss-�=Ԇ=�       �	����Xc�A�*

lossj�
=�c       �	I���Xc�A�*

loss�=�z�m       �	���Xc�A�*

loss�fr=�7�       �	
j��Xc�A�*

loss��=����       �	hX��Xc�A�*

lossx��<��       �	���Xc�A�*

loss��6=��       �	c���Xc�A�*

loss� ]=<+`       �	�|��Xc�A�*

loss�c=Ƒ�L       �	:x��Xc�A�*

loss���;��E�       �	����Xc�A�*

loss-�v=��)P       �	�{��Xc�A�*

loss
�=�M        �	����Xc�A�*

loss᰷<OѼd       �	�a��Xc�A�*

lossi�Z=�^!       �	�=��Xc�A�*

loss���=��       �	?���Xc�A�*

loss]-�<�`(F       �	"���Xc�A�*

loss҃�<.u#       �	�i��Xc�A�*

loss�'f=G{o�       �	y��Xc�A�*

loss�;PMd�       �	=��Xc�A�*

loss�}=),�       �	���Xc�A�*

lossu�=Q>V       �	�^��Xc�A�*

loss�(�=���       �	���Xc�A�*

loss��2=�3�       �	l?��Xc�A�*

lossq,�<J>�O       �	���Xc�A�*

loss!9<wks�       �	����Xc�A�*

lossh��;((S.       �	����Xc�A�*

loss�x�=��       �	�{��Xc�A�*

lossț�=jJ�       �	.S��Xc�A�*

loss�R=J`�)       �	���Xc�A�*

loss�;=�_B       �	����Xc�A�*

loss�k�<D�e       �	N��Xc�A�*

loss�<��l       �	e��Xc�A�*

loss/͏<��P       �	����Xc�A�*

loss
��<#�       �	����Xc�A�*

lossڌ�<ͬ�       �	*T��Xc�A�*

loss���<1Ő       �	Ő��Xc�A�*

lossݡ8=i_�
       �	3R��Xc�A�*

loss�1=����       �	�D��Xc�A�*

lossη�<��}       �	���Xc�A�*

loss� n=���$       �	1���Xc�A�*

loss6��:�=R�       �	����Xc�A�*

loss@�b=�L��       �	�j��Xc�A�*

loss=�l=]]��       �	�G��Xc�A�*

loss=z�=f�V       �	6��Xc�A�*

loss��<�+�       �	D���Xc�A�*

loss,K=�^Æ       �	$���Xc�A�*

loss�I�=�@T%       �	^���Xc�A�*

loss��!="��}       �	!=��Xc�A�*

loss��;��~s       �	� �Xc�A�*

loss%�=ӭm�       �	�� �Xc�A�*

lossI&<��w�       �	�>�Xc�A�*

loss��<UR��       �	���Xc�A�*

lossG=�	<       �	6v�Xc�A�*

lossn=��P�       �	��Xc�A�*

lossc7=�5E       �	2��Xc�A�*

lossMQ�=K�       �	�S�Xc�A�*

lossV�<F	L�       �	0��Xc�A�*

loss(,T<���       �	���Xc�A�*

loss�
=����       �	�5�Xc�A�*

loss�=����       �	8��Xc�A�*

loss�4= �b       �	z�Xc�A�*

loss6K3=@���       �	~�Xc�A�*

loss|�=��v�       �	���Xc�A�*

loss�:<��       �	zT	�Xc�A�*

lossT��;d�O       �	
�Xc�A�*

loss��;�E��       �	{�
�Xc�A�*

loss�Z=*P>^       �	�A�Xc�A�*

loss���<|�       �	���Xc�A�*

lossx,=C>q�       �	-{�Xc�A�*

loss���<�g       �	�Xc�A�*

loss6��<#$�       �	���Xc�A�*

loss�z�=�U�?       �	tz�Xc�A�*

loss���<�OA�       �	�Xc�A�*

lossBF=�H�f       �	-��Xc�A�*

loss3u�=����       �	�M�Xc�A�*

loss�C>����       �	y��Xc�A�*

loss1Fi=>�       �	?��Xc�A�*

loss���;p9�       �	b.�Xc�A�*

loss�hZ<�/��       �	���Xc�A�*

lossNQ<z9�i       �	Bx�Xc�A�*

lossI�7<�T��       �	�2�Xc�A�*

loss���<�       �	���Xc�A�*

lossn�@=kV/�       �	�r�Xc�A�*

lossn��=��o       �	��Xc�A�*

loss#��<�@ek       �	���Xc�A�*

loss���=,M[       �	W^�Xc�A�*

loss�O
=��e       �	;��Xc�A�*

loss�s�<��e�       �	˟�Xc�A�*

loss���<p�6       �	�D�Xc�A�*

loss�߀<Ѱ��       �	���Xc�A�*

loss���<RZ�I       �	M��Xc�A�*

loss*�=`�={       �	�"�Xc�A�*

loss�w=Ԃo�       �	�
�Xc�A�*

loss� =:��       �	I��Xc�A�*

lossX�Y=��R�       �	zV�Xc�A�*

loss��=�'Tl       �	^��Xc�A�*

loss좭=�0�m       �	[��Xc�A�*

lossWd,<��-       �	�i�Xc�A�*

loss�$<�o}       �	b �Xc�A�*

loss��5=Eb$T       �	� �Xc�A�*

loss�M�==
       �	#N!�Xc�A�*

losslb>wXƢ       �	��!�Xc�A�*

loss�t�== �u       �	t}"�Xc�A�*

loss��)><��P       �	7#�Xc�A�*

loss��&>���"       �	l�#�Xc�A�*

lossv�P<�0J>       �	^K$�Xc�A�*

loss�U@=ɿ?�       �	U�$�Xc�A�*

loss�@+=f@�       �	:�%�Xc�A�*

loss%ȇ=�4=       �	;:&�Xc�A�*

loss��;l;�       �	[�&�Xc�A�*

loss5D=N'�U       �	k�'�Xc�A�*

loss���=���       �	�J(�Xc�A�*

loss�{%=����       �	��(�Xc�A�*

loss�/a<�4�x       �	��)�Xc�A�*

lossW�=/,�
       �	�D*�Xc�A�*

loss�p�<�Md       �	��*�Xc�A�*

lossi��;ZS}�       �	ʨ+�Xc�A�*

loss��`=~"��       �	�c,�Xc�A�*

loss�z�<Q�t�       �	M-�Xc�A�*

loss|2\=7�T�       �	��-�Xc�A�*

loss�{b={���       �	�z.�Xc�A�*

loss�&}<c
[�       �	�'/�Xc�A�*

loss��=,���       �	%�/�Xc�A�*

loss��V=2���       �	&s0�Xc�A�*

loss#v�<�l�       �	�1�Xc�A�*

loss��r<=ے       �	�1�Xc�A�*

lossT3=4�]�       �	��2�Xc�A�*

lossTXp=e%X�       �	�3�Xc�A�*

loss�� =�'       �	�74�Xc�A�*

loss�\=柍t       �	��4�Xc�A�*

loss:�:=�/~�       �	'�5�Xc�A�*

loss3�!<j܄�       �	{j6�Xc�A�*

loss>V<�T]	       �	�7�Xc�A�*

lossْ=���7       �	��7�Xc�A�*

loss
�<���(       �	*S8�Xc�A�*

loss�7= ��'       �	��8�Xc�A�*

loss��=�K��       �	W�9�Xc�A�*

loss�b�<@7'       �	i7:�Xc�A�*

loss�4=��؏       �	��:�Xc�A�*

loss���;�^Q]       �	 s;�Xc�A�*

loss���<�*-       �	n<�Xc�A�*

loss�w�<8�B       �	��<�Xc�A�*

lossJm�<T#[�       �	~V=�Xc�A�*

loss���=|Z]�       �	H�=�Xc�A�*

loss
��<u܄�       �	R�>�Xc�A�*

lossJo<S�A�       �	�:?�Xc�A�*

lossʜY=����       �	I�?�Xc�A�*

loss��Y=��!�       �	9�@�Xc�A�*

loss*��<t��<       �	AGA�Xc�A�*

loss�2=U^�       �	h�A�Xc�A�*

loss��_<^���       �	��B�Xc�A�*

loss�=�n�a       �	�9C�Xc�A�*

loss*j=Mwl�       �	��C�Xc�A�*

lossߕ�=�~9       �	��D�Xc�A�*

loss;��=Q�       �	�"E�Xc�A�*

loss�P�<�
VO       �	��E�Xc�A�*

loss��<ly
       �	|eF�Xc�A�*

lossCV�<M��       �	G�Xc�A�*

loss%�<��%�       �	צG�Xc�A�*

loss�5�=d�~       �	xGH�Xc�A�*

loss��>0�1�       �	!�H�Xc�A�*

loss��|<�8e       �	��I�Xc�A�*

loss*��;saK       �	�J�Xc�A�*

loss�Y=����       �	�J�Xc�A�*

lossd�'=���=       �	�VK�Xc�A�*

losso

<�o��       �	�K�Xc�A�*

loss�O=F�P       �	�#M�Xc�A�*

loss�N�<noV       �	��M�Xc�A�*

lossmG�;�k�       �	�nN�Xc�A�*

loss���=���_       �	�O�Xc�A�*

loss��$>�p�       �	H�O�Xc�A�*

loss��2=x�qJ       �	�mP�Xc�A�*

loss2u<<�?��       �	UQ�Xc�A�*

lossja�<�-��       �	1�Q�Xc�A�*

loss(�<��Ow       �	�MR�Xc�A�*

loss���;$$t�       �	��R�Xc�A�*

loss�@=����       �	�S�Xc�A�*

loss�5J;_8�       �	`=T�Xc�A�*

loss�D�:��^v       �	��T�Xc�A�*

loss�p�;�o�       �	ߣU�Xc�A�*

lossV[�=��&�       �	�rV�Xc�A�*

loss��M<:��       �	�vW�Xc�A�*

lossf�5>�E�       �	/X�Xc�A�*

loss�v�=��گ       �	��X�Xc�A�*

loss1j�=�       �	�qY�Xc�A�*

lossw�4=��O       �	Z�Xc�A�*

lossű.=;i�~       �	 �Z�Xc�A�*

loss7si=���;       �	�B[�Xc�A�*

loss.\�<TH�#       �	�\�Xc�A�*

loss�1z=�ӯ�       �	��\�Xc�A�*

lossqɁ=����       �	�U]�Xc�A�*

loss8+<�|��       �	r�]�Xc�A�*

loss���=>%�m       �	��^�Xc�A�*

loss�=v��]       �	�>_�Xc�A�*

lossM��<VG 7       �	4�_�Xc�A�*

loss�Y�<�.ͧ       �	8�`�Xc�A�*

loss<�=�5B       �	Aa�Xc�A�*

loss���=v	       �	��a�Xc�A�*

lossp9<�S?       �	G�b�Xc�A�*

loss���<_)\       �	Y0c�Xc�A�*

lossE��=jw��       �	-�c�Xc�A�*

lossr�<l\��       �	�pd�Xc�A�*

loss[b=�.�       �	8e�Xc�A�*

lossE�=x���       �	5�e�Xc�A�*

loss��<}�ު       �	�Zf�Xc�A�*

loss!s�<ۛj        �	�qg�Xc�A�*

loss[HQ=Ej�a       �	�h�Xc�A�*

loss3�.=H       �	��h�Xc�A�*

loss�;�<�W��       �	ui�Xc�A�*

lossM� =$)5c       �	�j�Xc�A�*

loss�[u=�|��       �	�j�Xc�A�*

loss��-=��B�       �	kk�Xc�A�*

loss��B<�1       �	��l�Xc�A�*

loss6*�<����       �	�,m�Xc�A�*

loss��=�e,�       �	��m�Xc�A�*

loss�g�<��        �	V}n�Xc�A�*

lossT<��<b       �	�)o�Xc�A�*

loss[\	<�b�       �	��o�Xc�A�*

lossJ��=Yۘ�       �	��p�Xc�A�*

loss�\�=���l       �	��q�Xc�A�*

loss���=�}�       �	�Zr�Xc�A�*

loss��=a�:�       �	 �s�Xc�A�*

loss=V6�w       �	�Wt�Xc�A�*

lossJ)u=�7��       �	�u�Xc�A�*

lossv6�</rM       �	��u�Xc�A�*

loss6�M<�C-�       �	O�v�Xc�A�*

loss�e�<ݚ,�       �	�kw�Xc�A�*

lossk�=>T͚       �	�>x�Xc�A�*

lossJ9>=��r       �	��x�Xc�A�*

loss��.>�?<�       �	6�y�Xc�A�*

lossXm=���B       �	v8z�Xc�A�*

lossNVJ;�G{�       �	G�z�Xc�A�*

loss��=��(m       �	:�{�Xc�A�*

loss�Ҋ=F���       �	�o|�Xc�A�*

loss8�;��<       �	&}�Xc�A�*

loss�k=|��       �	��}�Xc�A�*

loss�f1=K���       �	,�~�Xc�A�*

loss�> =�D�       �	�.�Xc�A�*

loss��+=u���       �	@��Xc�A�*

loss��q=�Z�       �	R���Xc�A�*

loss3	�<]"�       �	�.��Xc�A�*

lossd�;<L���       �	7߁�Xc�A�*

loss��<糉7       �	����Xc�A�*

loss�<H(�       �	s+��Xc�A�*

loss��k<�[a       �	׃�Xc�A�*

loss�W[=z߼Q       �	����Xc�A�*

lossC�a<`�4       �	� ��Xc�A�*

lossQ��<����       �	 ȅ�Xc�A�*

loss�1�=2W}2       �	�i��Xc�A�*

lossƪ�=��?�       �		k��Xc�A�*

loss%��<�u��       �	���Xc�A�*

lossT2�=)'�k       �	Q���Xc�A�*

loss�ݒ=�Y�=       �	���Xc�A�*

lossd�;�X�c       �	���Xc�A�*

loss=��^       �	�g��Xc�A�*

loss3�<�5       �	E��Xc�A�*

lossc�<�r;g       �	����Xc�A�*

loss1��<h��       �	�o��Xc�A�*

lossw��<}L,       �	��Xc�A�*

loss�+�<�'��       �	����Xc�A�*

loss�ň=2t]       �	�o��Xc�A�*

loss}�>��#	       �	��Xc�A�*

loss� =W�"       �	հ��Xc�A�*

lossq��<��       �	�N��Xc�A�*

loss�>�=v�A8       �	=��Xc�A�*

loss�|=Z5�9       �	΋��Xc�A�*

lossJ=��F       �	�1��Xc�A�*

loss��=y~�@       �	tϓ�Xc�A�*

lossɁ�;P2��       �	3o��Xc�A�*

lossϝK;��;       �	M��Xc�A�*

loss�ht=R뇁       �	����Xc�A�*

loss�7{=w�"       �	�N��Xc�A�*

loss�2�=��7       �	>��Xc�A�*

lossӛ�;Tqj�       �	��Xc�A�*

lossy�; m�       �	�)��Xc�A�*

lossE��<�p{#       �	�Ę�Xc�A�*

loss6��<�c       �	�a��Xc�A�*

lossC<Ԁ�       �	����Xc�A�*

lossW�<���M       �	����Xc�A�*

lossm�N=J�
       �	�'��Xc�A�*

loss
XH=�-�i       �	����Xc�A�*

loss�*�<:��       �	�W��Xc�A�*

loss�~<�\Z       �	���Xc�A�*

lossE�<���z       �	����Xc�A�*

lossc��=��~�       �	N)��Xc�A�*

loss��<{��W       �	���Xc�A�*

loss6e<���       �	yW��Xc�A�*

loss_��=�+��       �	��Xc�A�*

loss��<M�I�       �	����Xc�A�*

loss��J=v�E�       �	�:��Xc�A�*

loss���<4�r       �	ݡ�Xc�A�*

loss��g=��le       �	�z��Xc�A�*

losscc�=^;[^       �	� ��Xc�A�*

lossD�=���:       �	���Xc�A�*

lossO�6=T[$       �	�X��Xc�A�*

loss��^=�ߪX       �	����Xc�A�*

loss�?�=��l       �	����Xc�A�*

loss�L%=	@/8       �	i��Xc�A�*

loss�=ebF       �	��Xc�A�*

lossӵ�=4�~       �	�ϧ�Xc�A�*

loss��=�u V       �	���Xc�A�*

lossX��;mq�j       �	�2��Xc�A�*

loss��I<��       �	���Xc�A�*

lossmj�=֦��       �	t���Xc�A�*

losswЖ=<�       �	Hl��Xc�A�*

lossX؅<�c�}       �	��Xc�A�*

lossw5�=���&       �	�¬�Xc�A�*

loss���<�h��       �	�g��Xc�A�*

loss�(�=�Z-�       �	���Xc�A�*

loss��=/0�e       �	���Xc�A�*

loss�>�Mv{       �	O[��Xc�A�*

loss}ˈ=��       �	���Xc�A�*

loss_�T=&>��       �	����Xc�A�*

lossKB#=��       �	�m��Xc�A�*

loss��=���       �	{��Xc�A�*

loss��B=��k�       �	���Xc�A�*

loss �=d��       �	�e��Xc�A�*

loss�:O=����       �	 V��Xc�A�*

loss4*=�a�       �	jk��Xc�A�*

loss��4=���>       �	����Xc�A�*

lossZ��<"��       �	���Xc�A�*

loss��<�kVb       �	EH��Xc�A�*

lossl�;<�M�F       �	����Xc�A�*

loss;֮<v��D       �	���Xc�A�*

loss�>�<c9q�       �	�?��Xc�A�*

loss�=.w��       �	�̻�Xc�A�*

loss�=�[        �	7p��Xc�A�*

loss�X=aZys       �	�"��Xc�A�*

loss��>�#[�       �	�ѽ�Xc�A�*

lossf��<�?       �	���Xc�A�*

loss��T=>Օ�       �	3��Xc�A�*

loss�s�=��w       �	�ݿ�Xc�A�*

loss��1=�j��       �	����Xc�A�*

loss,��<[�X9       �	�F��Xc�A�*

loss�	�<mHX�       �	3���Xc�A�*

loss�A�=���       �	���Xc�A�*

lossYL<.!��       �	BZ��Xc�A�*

loss��=<�P;       �	��Xc�A�*

loss<ʻ=de"�       �	@���Xc�A�*

loss���=$�P       �	-_��Xc�A�*

lossȣ�<WY       �	y��Xc�A�*

loss5�<�f�       �	����Xc�A�*

lossO_[=Ǣ[       �	�h��Xc�A�*

loss���<�9�       �	���Xc�A�*

loss�|+=�VF~       �	2���Xc�A�*

lossR�<3���       �	�K��Xc�A�*

loss�`\<eW�       �	����Xc�A�*

lossz�;u��-       �	΍��Xc�A�*

losss f=����       �	�2��Xc�A�*

loss�<Z��       �	����Xc�A�*

loss
�<mO^�       �	����Xc�A�*

loss��y=�1n>       �	���Xc�A�*

loss`�=k\�       �	y���Xc�A�*

loss�<�N       �	h��Xc�A�*

lossV%�<���.       �	���Xc�A�*

loss�ٲ=�QU       �	����Xc�A�*

lossSĠ<Zǆ       �	Yi��Xc�A�*

loss���=յ��       �	���Xc�A�*

lossdo�=�D؁       �	k���Xc�A�*

loss/=i,p\       �	ǂ��Xc�A�*

loss8S@=��
       �	&9��Xc�A�*

loss�,
<��vK       �	����Xc�A�*

loss�Lr=�3��       �	[���Xc�A�*

loss���;uk*�       �	�G��Xc�A�*

loss�6�>�Î�       �	����Xc�A�*

lossR��=42_�       �	����Xc�A�*

loss�=��U�       �	����Xc�A�*

loss�i*<bS/u       �	c���Xc�A�*

losspo=c��O       �	kf��Xc�A�*

loss1ɀ=iM��       �	�C��Xc�A�*

loss��6=��>�       �	CT��Xc�A�*

loss�E<�{{       �	Y���Xc�A�*

loss���=���       �	�[��Xc�A�*

loss���;O�H�       �	�W��Xc�A�*

loss��= �[       �	A��Xc�A�*

lossnϫ<���        �	h���Xc�A�*

loss�ܣ=�hm       �	=e��Xc�A�*

loss���<��w�       �	>��Xc�A�*

loss͕M=*       �	����Xc�A�*

loss 7m<7���       �	���Xc�A�*

loss��'=kb�       �	�H��Xc�A�*

loss�_=���       �	����Xc�A�*

loss�s�<ҏ4+       �	����Xc�A�*

loss�<�<Z	g�       �	�;��Xc�A�*

loss�,#=vdc       �	H���Xc�A�*

loss:B=.�c	       �	g���Xc�A�*

loss��<���v       �	G ��Xc�A�*

loss�O<d��k       �	����Xc�A�*

loss�$<0�>       �	ke��Xc�A�*

lossפ�=�'�       �	���Xc�A�*

loss��=�5       �	r���Xc�A�*

loss�c�=*Cߜ       �	VI��Xc�A�*

loss<<�p       �	a���Xc�A�*

lossr%�<�X�       �	�{��Xc�A�*

loss`�=���-       �	���Xc�A�*

loss��t<��*       �	з��Xc�A�*

loss>S#=�*1]       �	#M��Xc�A�*

loss���<�!q       �	�4��Xc�A�*

loss;��<��U       �	K���Xc�A�*

loss�n�<�޶       �	�d��Xc�A�*

loss�t�<�d��       �	���Xc�A�*

loss��=H��       �	'���Xc�A�*

loss|b�<yMA        �	e7��Xc�A�*

loss��<eFw�       �	����Xc�A�*

loss.�/=R�zb       �	>w��Xc�A�*

loss��<Yܲ�       �	`��Xc�A�*

loss3O<�}]f       �	A���Xc�A�*

loss��:�>�       �	�^��Xc�A�*

loss?�M<�y\3       �	jg��Xc�A�*

loss:�=S���       �	c���Xc�A�*

loss%��<��n�       �	ZG��Xc�A�*

loss�.=~��!       �	a���Xc�A�*

loss�R�<$�,q       �	)v��Xc�A�*

lossVG�=Z���       �	M��Xc�A�*

lossq�<�˹       �	1���Xc�A�*

loss�M�;-w�P       �	����Xc�A�*

loss��.=
O�       �	'��Xc�A�*

loss�q�;]%8       �	7���Xc�A�*

loss竊;!?m}       �	F^��Xc�A�*

loss�ģ;N��S       �	����Xc�A�*

lossoD$<ǣt`       �	����Xc�A�*

loss��<��V[       �	*��Xc�A�*

lossSI+=bL       �	���Xc�A�*

loss��:����       �	J_��Xc�A�*

loss�Q�<�f�       �	����Xc�A�*

loss��93�	       �	D� �Xc�A�*

loss՜9�x �       �	9*�Xc�A�*

loss3;���J       �	��Xc�A�*

loss��w<�z)�       �	�R�Xc�A�*

loss恇=.3x�       �	5��Xc�A�*

loss�R!< ���       �	���Xc�A�*

loss��5:k�S       �	N(�Xc�A�*

loss_h!=S�R       �	>��Xc�A�*

loss�0,>�ۏ       �	��Xc�A�*

loss_� ;��Zx       �	^*�Xc�A�*

loss2)#>HX�       �	:��Xc�A�*

lossn0�=�D��       �	j�Xc�A�*

loss:��=��e�       �	�Xc�A�*

losso�<�>�       �	���Xc�A�*

loss���;B��       �	�_	�Xc�A�*

loss���=,�v       �	��	�Xc�A�*

loss�L�=`��       �	�
�Xc�A�*

loss���<5�V{       �	G8�Xc�A�*

lossw1=��6�       �	��Xc�A�*

loss_��<�=]       �	vk�Xc�A�*

loss�yF=��r�       �	��Xc�A�*

loss�ٵ=�U��       �	��Xc�A�*

loss�{	=6���       �	n3�Xc�A�*

loss�5k=J�ם       �	���Xc�A�*

loss��C=}(�       �	
g�Xc�A�*

loss�Ȩ=��6       �	��Xc�A�*

loss��=���       �	���Xc�A�*

loss���=\�g       �	�`�Xc�A�*

loss��V=6X       �	��Xc�A�*

loss�5�=�_�       �	W��Xc�A�*

lossʖ=���P       �	|*�Xc�A�*

loss;�<��=�       �	���Xc�A�*

loss�z�<���       �	p��Xc�A�*

loss:.$;�4u�       �	,�Xc�A�*

lossB7�<��n�       �	���Xc�A�*

loss���=Wl�       �	�x�Xc�A�*

lossXy�<7K&�       �	�2�Xc�A�*

loss6Qa=�q�d       �	���Xc�A�*

loss]6K=�5�       �	̚�Xc�A�*

lossl�=`���       �	1�Xc�A�*

loss�"=U��v       �	K��Xc�A�*

loss?�<T��U       �	7q�Xc�A�*

lossH�;ZL�>       �	"�Xc�A�*

loss�$�<�)8�       �	-��Xc�A�*

losst�f<���       �	�Xc�A�*

loss��<����       �	)��Xc�A�*

loss�ǌ=�;|       �	�}�Xc�A�*

loss�gz=����       �	<�Xc�A�*

loss��<��F�       �	��!�Xc�A�*

loss�l�;�̻       �	Â"�Xc�A�*

loss�c=�	��       �	�S#�Xc�A�*

lossH��<��eE       �	��#�Xc�A�*

loss~�=��U�       �	�$�Xc�A�*

losshY�<!�       �	�2%�Xc�A�*

loss��<	,K�       �	6�%�Xc�A�*

loss�!=�\��       �	�e&�Xc�A�*

loss,Hu<��z       �	 �&�Xc�A�*

loss��<�m��       �	��'�Xc�A�*

loss_F�;�)Ȏ       �	O;(�Xc�A�*

loss?=�>�N       �	��(�Xc�A�*

lossU�<���M       �	��A�Xc�A�*

loss�Z�<��kw       �	�B�Xc�A�*

loss��=Z�A�       �	}�B�Xc�A�*

loss�|=�n�       �	<KC�Xc�A�*

loss�1�<�Q��       �	sD�Xc�A�*

lossWAe<Ƞ��       �	ߤD�Xc�A�*

lossj��<�}��       �	�=E�Xc�A�*

loss��\=0%�       �	p�E�Xc�A�*

loss���=��       �	y�F�Xc�A�*

loss�C�<��l       �	�BG�Xc�A�*

loss���<s��Q       �	��G�Xc�A�*

loss,=<%�K       �	7�H�Xc�A�*

loss�>򞀣       �	�BI�Xc�A�*

loss;�e<ʚ�       �	��I�Xc�A�*

loss�<�<�)�       �	�|J�Xc�A�*

loss	��<� �-       �	�K�Xc�A�*

loss(26<Z;ݟ       �	=�K�Xc�A�*

lossw�<��;n       �	�NL�Xc�A�*

losstv�<�[q�       �	dM�Xc�A�*

loss	��=bgF       �	��M�Xc�A�*

loss#P�<%�1�       �	=EN�Xc�A�*

loss�>��ߺ       �	�LO�Xc�A�*

loss���;Z؂       �	��O�Xc�A�*

loss&B:=�H��       �	M�P�Xc�A�*

loss�Z�<~�@       �	2=Q�Xc�A�*

loss��<(��       �	8�Q�Xc�A�*

loss�&=&�ʽ       �	kR�Xc�A�*

loss�O=�}��       �	-"S�Xc�A�*

lossrZ:=KMN       �	&�S�Xc�A�*

lossjC�=.�k�       �	��T�Xc�A�*

loss�mY=�^b�       �	��U�Xc�A�*

loss�h�=N!       �	+iV�Xc�A�*

loss��<3�F       �	�W�Xc�A�*

lossz,�=ط�I       �	��W�Xc�A�*

loss|=        �	9EX�Xc�A�*

loss̩I=�'�$       �	��X�Xc�A�*

loss��O;0�3       �	1~Y�Xc�A�*

loss�TZ=���k       �	�Z�Xc�A�*

loss?�>|[b       �	��Z�Xc�A�*

loss�=T=�p�       �	��[�Xc�A�*

loss1`@=�`��       �	~U\�Xc�A�*

lossz]=���       �	=�\�Xc�A�*

loss�6N<p5�       �	�]�Xc�A�*

loss��<�s1w       �	J'^�Xc�A�*

loss�1�<����       �	}�^�Xc�A�*

loss�#�=�� �       �	<g_�Xc�A�*

lossϺ�<"�[       �	`�Xc�A�*

loss�aG=D��       �	Ω`�Xc�A�*

loss��%<�9�_       �	fOa�Xc�A�*

loss� �;~�q�       �	��a�Xc�A�*

loss.�;�x��       �	k�b�Xc�A�*

lossK=
A*       �	�;c�Xc�A�*

loss[[<5�       �	��c�Xc�A�*

loss2i>�>Tz       �	;�d�Xc�A�*

loss
��<-D�p       �	�^e�Xc�A�*

loss��;��W       �	�f�Xc�A�*

loss���;��e       �	�f�Xc�A�*

loss<H;�a�       �	�sg�Xc�A�*

loss"�<��Z�       �	�h�Xc�A�*

loss�U=�(4�       �	�h�Xc�A�*

loss�S8>Ѫ�       �	�Ni�Xc�A�*

lossm��<T��@       �	�i�Xc�A�*

loss���<�aq�       �	[xj�Xc�A�*

loss��;=
qٓ       �	�k�Xc�A�*

loss;��<#X�       �	��k�Xc�A�*

lossm��;�2d>       �	�Vl�Xc�A�*

lossH�c=`�N       �	�l�Xc�A�*

loss�2�<eˏ       �	��m�Xc�A�*

lossW��=�5H8       �	�\n�Xc�A�*

loss_X�<�|�
       �	@�n�Xc�A�*

loss6�=B&�       �	n�o�Xc�A�*

loss0=��       �	9�p�Xc�A�*

loss �w=+-m       �	�3q�Xc�A�*

losszj�<��a       �	P�q�Xc�A�*

loss}YQ=���       �	H�r�Xc�A�*

loss�w�<��1�       �	�1s�Xc�A�*

lossNO�<�Eb       �	�s�Xc�A�*

loss�7h<G8	       �	��t�Xc�A�*

loss�dv<�O       �	�cu�Xc�A�*

loss��K=�p��       �	v�Xc�A�*

loss7�v<f�%�       �	�v�Xc�A�*

loss&��<���       �	k�w�Xc�A�*

losshp;=Nv��       �	�;x�Xc�A�*

loss��;s}��       �	��x�Xc�A�*

loss��;�9V�       �	g~y�Xc�A�*

loss��)<��6       �	� z�Xc�A�*

lossf�<~�g�       �	��z�Xc�A�*

loss1�J=����       �	�]{�Xc�A�*

lossL<�6ʏ       �	�{�Xc�A�*

lossM�<L�       �	)�|�Xc�A�*

lossV �<b��       �	/N}�Xc�A�*

loss�;5=���"       �	d�}�Xc�A�*

loss�G[= �T       �	ؚ~�Xc�A�*

lossf�><��|       �	�7�Xc�A�*

loss��=��	       �	���Xc�A�*

loss�O�;�X�       �	�r��Xc�A�*

loss�z@=6 Ġ       �	���Xc�A�*

loss�=�<*�       �	���Xc�A�*

loss�G�<3�G       �	v���Xc�A�*

lossi�1<���       �	\ ��Xc�A�*

loss͡<��v       �	Ժ��Xc�A�*

loss��;��[       �	>^��Xc�A�*

loss���=�xj�       �	Q���Xc�A�*

loss��x<\	�       �	z���Xc�A�*

loss�=ބ��       �	�{��Xc�A�*

loss@JR<�>2y       �	���Xc�A�*

loss�{�<3iI�       �	/Ç�Xc�A�*

loss:��<J6|�       �	d\��Xc�A�*

loss�`�=:���       �	��Xc�A�*

loss㥅<�]�       �	����Xc�A�*

loss��=��V@       �	����Xc�A�*

loss�E�<�|�       �	P���Xc�A�*

loss/<j1�j       �	����Xc�A�*

loss�^8;�D�~       �	�0��Xc�A�*

loss�`�<���       �	pэ�Xc�A�*

lossaW�<e���       �	�z��Xc�A�*

loss��w<���_       �	\ ��Xc�A�*

loss�/=���       �	�ȏ�Xc�A�*

loss�\
<�:�,       �	g��Xc�A�*

loss�7<�N�       �	���Xc�A�*

loss��<Ј��       �	U���Xc�A�*

loss���<}�¼       �	�B��Xc�A�*

lossz�;��^'       �	���Xc�A�*

lossv�z=�	C       �	;���Xc�A�*

loss�>�;u��        �	f.��Xc�A�*

loss��.=L
�       �	Nє�Xc�A�*

loss�l�<c�e�       �	�h��Xc�A�*

loss�R=6�S       �	�	��Xc�A�*

loss.)�<4I|*       �	r���Xc�A�*

loss���=#��t       �	 B��Xc�A�*

loss�=�;	^9       �	�ڗ�Xc�A�*

lossa��=z��!       �	Kw��Xc�A�*

loss���<��̈       �	C��Xc�A�*

loss�ؚ<�w{       �	���Xc�A�*

lossl�"=��
P       �	����Xc�A�*

loss*__;ȷ�L       �	��Xc�A�*

lossDw�<d�<�       �	~ě�Xc�A�*

loss�$�<�&��       �	�]��Xc�A�*

loss��;��A       �	c��Xc�A�*

loss��H<tb�       �	����Xc�A�*

lossN�;�U��       �	<.��Xc�A�*

loss���=��Z�       �	�Ğ�Xc�A�*

loss[�'=�s�X       �	����Xc�A�*

loss_Q�<� S       �	��Xc�A�*

loss�t�<����       �	����Xc�A�*

loss�:�<��ig       �	3P��Xc�A�*

losslFq=R��W       �	x��Xc�A�*

loss��<{%uJ       �	(���Xc�A�*

loss"&<b,�       �	J��Xc�A�*

loss�]< k��       �	���Xc�A�*

losss�=O�^�       �	�x��Xc�A�*

loss���;�W{       �	���Xc�A�*

loss�c�=���       �	Y���Xc�A�*

lossGؠ=����       �	<��Xc�A�*

loss���=���c       �	�Ѧ�Xc�A�*

loss���;(���       �	4e��Xc�A�*

loss�
�;:��m       �	���Xc�A�*

loss`�h=�q        �	����Xc�A�*

loss;(A=U
��       �	�&��Xc�A�*

loss���;���       �	���Xc�A�*

loss`0=�B�       �	����Xc�A�*

loss��=�g��       �	�I��Xc�A�*

loss��<^γ/       �	���Xc�A�*

lossg�;S?�i       �	|���Xc�A�*

loss8��=�D(�       �	9&��Xc�A�*

lossQZi<ӿ3#       �	hή�Xc�A�*

loss��r<\��       �	?r��Xc�A�*

lossA��=�r�       �	���Xc�A�*

lossox�<��'�       �	c+��Xc�A�*

loss/X=���       �	yͱ�Xc�A�*

loss�hT=�X��       �	�a��Xc�A�*

loss�,<����       �	����Xc�A�*

loss�2<B�\!       �	����Xc�A�*

loss��~=�,M)       �	T��Xc�A�*

losst'�<���       �	|���Xc�A�*

loss�I=���`       �	�J��Xc�A�*

loss��=1T�l       �	���Xc�A�*

loss�>M=J���       �	����Xc�A�*

lossi=�z�       �	p?��Xc�A�*

loss'O<jtF       �	#k��Xc�A�*

loss�f1<�%       �	���Xc�A�*

loss\�N=�H��       �	e���Xc�A�*

loss�=*���       �	II��Xc�A�*

loss�A<� 29       �	���Xc�A�*

loss�_�;S��       �	3���Xc�A�*

loss�"<�$��       �	�+��Xc�A�*

loss��,<�t�F       �	����Xc�A�*

loss�v�=��l       �	�U��Xc�A�*

lossW�<%=�T       �	2 ��Xc�A�*

loss�6�=��s       �	x���Xc�A�*

loss:��=(�$�       �	�7��Xc�A�*

loss1��< $��       �	ٿ�Xc�A�*

lossS�=�M�       �	=���Xc�A�*

lossR��<�s{"       �	%!��Xc�A�*

loss�t:�\)�       �	����Xc�A�*

loss��2=U�y       �	�W��Xc�A�*

losstU�<��^�       �	����Xc�A�*

loss�y=� �       �	.���Xc�A�*

loss�Hh=���I       �	|*��Xc�A�*

loss�k;|~fk       �	���Xc�A�*

loss��<��
B       �	�c��Xc�A�*

loss� a<�sP'       �	U���Xc�A�*

loss��J<�F�d       �	���Xc�A�*

loss��=�l�       �	%��Xc�A�*

loss�-=���e       �	���Xc�A�*

lossH��<��b�       �	�P��Xc�A�*

loss���=ۂ-       �	����Xc�A�*

loss��=�V�       �	M���Xc�A�*

loss&�;� �R       �	���Xc�A�*

loss��2<UOlW       �	���Xc�A�*

loss��8=Ҩ       �	K��Xc�A�*

loss6�<�K!       �	"���Xc�A�*

loss�'<E�       �	�o��Xc�A�*

losso\/=���?       �	���Xc�A�*

lossӨ�<���X       �	Z���Xc�A�*

loss���=�       �	�?��Xc�A�*

loss��>�vU�       �	4���Xc�A�*

loss8;�≠~�       �	�l��Xc�A�*

loss���=�J�       �	���Xc�A�*

lossƹ�=�]�       �	'���Xc�A�*

lossQ�v=�r�       �	�8��Xc�A�*

loss�u�;HD�!       �	����Xc�A�*

loss	��<�       �	V���Xc�A�*

lossg3 =N�IE       �	�%��Xc�A�*

lossҏI;F��       �	���Xc�A�*

loss�W�<6�es       �	�U��Xc�A�*

lossJt�<X��       �	[���Xc�A�*

loss�u	=���       �	���Xc�A�*

loss�76<EL?�       �	g'��Xc�A�*

losso�<�t�       �	����Xc�A�*

lossM��;}�Ò       �	uW��Xc�A�*

loss7_�<J��       �	����Xc�A�*

lossP�<��X�       �	&���Xc�A�*

loss���=��       �	�"��Xc�A�*

loss�q�=��K       �	9���Xc�A�*

loss�<����       �	����Xc�A�*

loss��F=�jI�       �	q��Xc�A�*

loss���=�T8*       �	��Xc�A�*

loss-�=!Y)g       �	ٯ��Xc�A�*

lossn�<��       �	�I��Xc�A�*

lossx�;���       �	h���Xc�A�*

loss{��<w 	       �	���Xc�A�*

loss�jY<t��       �	+��Xc�A�*

loss�f=�e�       �	����Xc�A�*

lossdN=r�
       �	J��Xc�A�*

loss1k�<:]�       �	����Xc�A�*

lossxX�;�m�       �	Z���Xc�A�*

loss1�G<��ai       �	�N��Xc�A�*

lossR��<aU��       �	����Xc�A�*

lossN�=�x �       �	����Xc�A�*

lossɽ�<UM�-       �	�-��Xc�A�*

loss��=3��       �	v���Xc�A�*

lossAb�<���       �	�T��Xc�A�*

lossp��<��K�       �	����Xc�A�*

loss�	=KV��       �	x��Xc�A�*

lossR��;%��       �	d#��Xc�A�*

loss �<����       �	����Xc�A�*

loss�s�;�bf�       �	'O��Xc�A�*

loss���;����       �	�%��Xc�A�*

loss���=����       �	����Xc�A�*

lossX��<`Q�       �	�i��Xc�A�*

lossc=�<�4�        �	C��Xc�A�*

loss�/{<�k#       �	���Xc�A�*

loss��<P�z       �	v6��Xc�A�*

loss� =٩�O       �	N���Xc�A�*

loss��/<��"       �	�|��Xc�A�*

loss��=.0q�       �	�!��Xc�A�*

loss<�OI�       �	n���Xc�A�*

lossJ=	`V�       �	hX��Xc�A�*

loss��=���X       �	��Xc�A�*

loss��;<�o`�       �	����Xc�A�*

lossT��;��!        �	iQ��Xc�A�*

loss2�*=Y�4�       �	���Xc�A�*

loss���<�Q��       �	���Xc�A�*

lossEq�=��f�       �	����Xc�A�*

lossp=M�        �	IJ��Xc�A�*

loss�l�<? ��       �	,���Xc�A�*

loss��E<�X�\       �	����Xc�A�*

loss^��=��       �	����Xc�A�*

loss$�X=�)o$       �	��Xc�A�*

lossE 0<���I       �	x)��Xc�A�*

loss*�<꬙k       �	����Xc�A�*

loss�S�<���       �	�r��Xc�A�*

loss�U�<�I�        �	�"��Xc�A�*

loss�%�<��b�       �	8���Xc�A�*

loss)C�=�S��       �	�V��Xc�A�*

loss�=%��/       �	����Xc�A�*

loss�{;���       �	�3��Xc�A�*

loss~i=�VX       �	����Xc�A�*

lossC�};�<��       �	�x��Xc�A�*

loss� -;Y&�       �	�  �Xc�A�*

loss�j=0�c�       �	�� �Xc�A�*

loss]f:<��?�       �	� �Xc�A�*

loss|�2:��       �	(��Xc�A�*

loss�� =Aq��       �	�1�Xc�A�*

loss=�s=��J       �	���Xc�A�*

lossz�0<��!�       �	@h�Xc�A�*

loss�?=e��       �	O�Xc�A�*

loss��=Ku6       �	���Xc�A�*

loss�>�=�׼C       �	�?�Xc�A�*

loss @�=���@       �	f��Xc�A�*

lossd�==��6       �	A��Xc�A�*

loss]<En�*       �	�&�Xc�A�*

loss6'�=�p�       �	���Xc�A�*

loss�y=b�Ѕ       �	V	�Xc�A�*

loss��>g���       �	��	�Xc�A�*

loss&�<�w��       �	ٳ
�Xc�A�*

lossa��=�+KT       �	wH�Xc�A�*

loss�}?< �j       �	���Xc�A�*

lossW�<�c �       �	�?�Xc�A�*

loss��=5�Ӡ       �	���Xc�A�*

lossa3=�e�!       �	${�Xc�A�*

loss���=��       �	��Xc�A�*

loss�<�wq       �	��Xc�A�*

loss�0�;i�uJ       �	�G�Xc�A�*

loss�&X=�bS�       �	��Xc�A�*

lossH��<r$}�       �	{�Xc�A�*

loss��<w���       �	��Xc�A�*

lossk̆=��E       �	��Xc�A�*

loss��<��s
       �	�I�Xc�A�*

loss�v�<��=       �	���Xc�A�*

loss���=��JD       �	��Xc�A�*

loss<�<j��t       �	d;�Xc�A�*

lossE�;<V�|       �	n��Xc�A�*

lossQm�<4L�       �	x{�Xc�A�*

loss�2;=M�e�       �	��Xc�A�*

lossb�=���f       �	*��Xc�A�*

lossV�7=�n�P       �	�D�Xc�A�*

loss���<ANj       �	��Xc�A�*

loss�A1=u"c       �	���Xc�A�*

loss�<��`-       �	�,�Xc�A�*

loss��<�9�       �	���Xc�A�*

loss�*�;�''       �	�r�Xc�A�*

loss�N=���@       �	|�Xc�A�*

loss�/�=��p       �	���Xc�A�*

loss��=��@       �	G�Xc�A�*

loss��[=���       �	v��Xc�A�*

loss�¥<����       �	$}�Xc�A�*

loss
��<�n~�       �	T�Xc�A�*

loss��:�3       �	] �Xc�A�*

loss���;�6�       �	#� �Xc�A�*

loss�<;a$       �	
�!�Xc�A�*

loss�U=d(>       �	T�"�Xc�A�*

loss�$	<�*��       �	$#�Xc�A�*

lossJ��=p��       �	�#�Xc�A�*

loss�<���       �	B^$�Xc�A�*

lossR�d<E
�H       �	S%�Xc�A�*

lossd_�=.:�       �	1�%�Xc�A�*

lossS�=���       �	P&�Xc�A�*

loss$��=x��0       �	I�&�Xc�A�*

loss�9=��<�       �	�'�Xc�A�*

loss�T�<ꖍ�       �	�4(�Xc�A�*

lossqD=����       �	��(�Xc�A�*

loss�\�;u1}�       �	~n)�Xc�A�*

loss�$�=#�}�       �	�*�Xc�A�*

loss���<gF��       �	�*�Xc�A�*

loss|v�;�H$       �	m7+�Xc�A�*

loss��f;Eѫ�       �	O�+�Xc�A�*

losszId=�2N�       �	��,�Xc�A�*

loss^�=���Y       �	p_-�Xc�A�*

loss�i\=sU��       �	��-�Xc�A�*

loss ��<*ZtE       �	P�.�Xc�A�*

lossm<e�%       �	�P/�Xc�A�*

loss�P�=��2�       �	\�/�Xc�A�*

loss.�X=Km�P       �	��0�Xc�A�*

loss
�=�T��       �	GU1�Xc�A�*

loss��=�[��       �	2�Xc�A�*

loss��P<��f�       �	��2�Xc�A�*

loss8��<�8�       �	�<3�Xc�A�*

loss6�=
�|       �	�3�Xc�A�*

lossJ7=�UY�       �	�4�Xc�A�*

loss,�m=9���       �	6 5�Xc�A�*

lossW�<r4�       �	/�5�Xc�A�*

loss��p;�uA�       �	�X6�Xc�A�*

lossZ�< �       �	�-7�Xc�A�*

loss��<m��^       �	��7�Xc�A�*

lossAz�=I���       �	�p8�Xc�A�*

loss)&=�Z��       �	79�Xc�A�*

loss��h=�XfP       �	ø9�Xc�A�*

lossq�>�v�       �	�`:�Xc�A�*

lossw�g<e�       �	�	;�Xc�A�*

loss�AZ<_��       �	`�;�Xc�A�*

lossqv�=�D5       �	AI<�Xc�A�*

loss��]<��eX       �	K�<�Xc�A�*

loss<P<&��       �	�|=�Xc�A�*

lossiA]=tbLL       �	a>�Xc�A�*

lossM�<��e,       �	��>�Xc�A�*

loss��+=���W       �	�M?�Xc�A�*

loss2�<��w�       �	`�?�Xc�A�*

loss �:=s�G�       �	܀@�Xc�A�*

loss�<��C�       �	O$A�Xc�A�*

loss�,�<�B�       �	�A�Xc�A�*

loss�T�<��=�       �	m�B�Xc�A�*

lossLH�=��       �	�%C�Xc�A�*

loss�~�<L,�       �	b�C�Xc�A�*

lossS��={S�       �	KVD�Xc�A�*

loss,�<_��       �	�D�Xc�A�*

loss�+\=2�ӝ       �	��E�Xc�A�*

loss�.�<Rn��       �	MeF�Xc�A�*

loss7�8=��W       �	�G�Xc�A�*

loss��e<� )�       �	�G�Xc�A�*

loss��r<��fa       �	�>H�Xc�A�*

loss�	I=���
       �	6�H�Xc�A�*

lossA#<���       �	ՑI�Xc�A�*

lossC�=Y�c       �	9)J�Xc�A�*

loss�ڥ<�       �	�J�Xc�A�*

loss��=!�5�       �	�cK�Xc�A�*

loss!�<aW�5       �	lL�Xc�A�*

loss��<��(�       �	;�L�Xc�A�*

loss�'�=����       �	fM�Xc�A�*

loss��%=XRR�       �	��M�Xc�A�*

loss�{�=�Ҥ�       �	/�N�Xc�A�*

lossr�P=��k'       �	�BO�Xc�A�*

loss��C=����       �	n�O�Xc�A�*

loss�8?=J�M�       �	|P�Xc�A�*

loss{��<b"Wm       �	�Q�Xc�A�*

loss�|r;�?       �	��Q�Xc�A�*

loss�Ţ<��       �	�IR�Xc�A�*

loss`Ь=��g       �	r�R�Xc�A�*

lossz�=����       �	�vS�Xc�A�*

lossT4�<� �       �	�T�Xc�A�*

loss$��=i�+       �	��T�Xc�A�*

loss���;z�y       �	�KU�Xc�A�*

loss{Y>JZ�       �	T�U�Xc�A�*

lossx�=�#e       �	b�V�Xc�A�*

loss=�(�       �	�W�Xc�A�*

loss��<�=P       �	$�W�Xc�A�*

loss��=�<�       �	�SX�Xc�A�*

loss�C�<x��       �	��X�Xc�A�*

loss���<��       �	G�Y�Xc�A�*

loss2�a=O�K*       �	Z�Xc�A�*

lossϢ'=U��       �	q[�Xc�A�*

loss��;<	��       �	�[�Xc�A�*

loss?�J=�#�       �	V\�Xc�A�*

loss�'g<��-       �	��\�Xc�A�*

lossn�#=U��       �	��]�Xc�A�*

lossN=�
�       �	0�^�Xc�A�*

lossqg�<����       �	�._�Xc�A�*

lossҴ�;����       �	7�_�Xc�A�*

loss 4V<8~y�       �	�]`�Xc�A�*

loss�y=iF��       �	@�`�Xc�A�*

loss��<)�C       �	 �a�Xc�A�*

loss�uR=�&       �	�kb�Xc�A�*

lossc�0=b!�a       �	c�Xc�A�*

loss�Ej;紜�       �	�c�Xc�A�*

loss�we<��|       �	czd�Xc�A�*

loss�k�=f�G-       �	we�Xc�A�*

loss�C�<Y��$       �	��e�Xc�A�*

loss�Ȣ<�P�       �	_f�Xc�A�*

loss&C=.W�       �	�g�Xc�A�*

lossͻy=�Lg.       �	,�g�Xc�A�*

loss��v<�eX       �	�rh�Xc�A�*

loss�4�<�'��       �	�i�Xc�A�*

loss��=�l�n       �	��i�Xc�A�*

loss7��<����       �	�Pj�Xc�A�*

lossd�=�9)       �	S�j�Xc�A�*

loss2<1=3�TX       �	��k�Xc�A�*

lossno
=���x       �	<l�Xc�A�*

lossh�=%��       �	Q�l�Xc�A�*

loss��=��-�       �	f�m�Xc�A�*

loss���<�3��       �	c(n�Xc�A�*

lossB�<���       �	��n�Xc�A�*

loss�M�;�d%�       �	qo�Xc�A�*

losso��=!���       �	|p�Xc�A�*

loss�a8<(��v       �	y�p�Xc�A�*

loss%ĺ;7��R       �	HOq�Xc�A�*

loss�_�;W��       �	��q�Xc�A�*

lossv�<H>X\       �	��r�Xc�A�*

loss�g�=�QMk       �	�+s�Xc�A�*

loss���:�I�       �	��s�Xc�A�*

lossC{�=�:�`       �	mt�Xc�A�*

loss_C<Ӌ�]       �	�Eu�Xc�A�*

loss&I�<9�x       �	��u�Xc�A�*

loss��!<�{H:       �	3w�Xc�A�*

loss|��;/��r       �	%�w�Xc�A�*

loss2Ӿ;����       �	�hx�Xc�A�*

loss��;;e�       �	�y�Xc�A�*

loss�<b(��       �	��y�Xc�A�*

loss6y�;ggk       �	$Ez�Xc�A�*

lossf/>�To       �	�z�Xc�A�*

loss�=��j       �	8�{�Xc�A�*

lossL�\=�~Q       �	5%|�Xc�A�*

loss���=�/l�       �	ܻ|�Xc�A�*

lossϔ�=��5�       �	�R}�Xc�A�*

loss,�=��
m       �	G~�Xc�A�*

loss$^<qww�       �	;�~�Xc�A�*

loss��<��       �	�`�Xc�A�*

loss��=Ҁ�       �	L��Xc�A�*

loss_GP<ݟ       �	����Xc�A�*

loss��*=z��       �	d=��Xc�A�*

loss��4=+�1       �	����Xc�A�*

loss&L�<T�9       �	����Xc�A�*

loss���<ax�S       �	Q3��Xc�A�*

lossO'�='�{       �	tσ�Xc�A�*

loss{��<�� �       �	�i��Xc�A�*

loss%\
<�pc       �	��Xc�A�*

loss�2=��'�       �	@���Xc�A�*

loss�<To�+       �	�E��Xc�A�*

loss><%�%       �	t%��Xc�A�*

loss[�&=�PL�       �	�Ƈ�Xc�A�*

loss��W<չ@�       �	Ug��Xc�A�*

loss�+�;�fe�       �	c��Xc�A�*

lossO�<Dy       �	ũ��Xc�A�*

lossҜ;���       �	jP��Xc�A�*

loss��B=�hI       �	$��Xc�A�*

lossґ1<4B@       �	����Xc�A�*

loss�4�=tBw       �	C ��Xc�A�*

loss� �<�7/       �	0���Xc�A�*

loss�4�=��Q       �	N��Xc�A�*

loss��=^>��       �	���Xc�A�*

loss�S=4�u5       �	T���Xc�A�*

loss�<�=�=       �	s*��Xc�A�*

loss�8=@#       �	�я�Xc�A�*

loss��6<s���       �	^i��Xc�A�*

lossa?=MN�@       �	r6��Xc�A�*

lossM�h=%L��       �	���Xc�A�*

loss��<덙:       �	㦒�Xc�A�*

loss��=ђ�       �	DQ��Xc�A�*

loss;�;�9ż       �	j���Xc�A�*

loss��'=��1�       �	D���Xc�A�*

loss+=ѫ)>       �	���Xc�A�*

loss�fz=�!       �	U���Xc�A�*

loss�N5;��t�       �	׊��Xc�A�*

loss�=��5=       �	�%��Xc�A�*

loss�&=^��\       �	�˘�Xc�A�*

lossw^�<���       �	x~��Xc�A�*

lossv'�<���       �	y=��Xc�A�*

loss���<|V˷       �	�ؚ�Xc�A�*

loss&��<�Ф�       �	�I��Xc�A�*

lossM�<�{(       �	7��Xc�A�*

loss�
�;��:       �	Z��Xc�A�*

loss��`=��f�       �	���Xc�A�*

loss�x;ͬ�       �	���Xc�A�*

loss��B:��W       �	]R��Xc�A�*

loss;=� o&       �	���Xc�A�*

loss�~�<�H�T       �	����Xc�A�*

loss}њ<@���       �	r5��Xc�A�*

lossq#�;����       �	���Xc�A�*

loss��,;"��~       �	���Xc�A�*

lossl��:�b�       �	C��Xc�A�*

loss���;�
^       �	ܣ�Xc�A�*

losshf;{o�$       �	����Xc�A�*

lossr6�:�AY�       �	�!��Xc�A�*

loss�Qc<�ז;       �	S��Xc�A�*

lossa <��*       �	/ߦ�Xc�A�*

loss�k`<�d�W       �	����Xc�A�*

lossTm:�5Ӗ       �	d]��Xc�A�*

lossS��;h�       �	���Xc�A�*

loss� 5>��W�       �	����Xc�A�*

loss�^�<a�D_       �	�X��Xc�A�*

loss��?>y&<       �	����Xc�A�*

lossq�0<�闺       �	z���Xc�A�*

lossi��=��4�       �	wL��Xc�A�*

loss�r#<V�>�       �	
���Xc�A�*

lossH��<��>       �	8��Xc�A�*

loss#8�=EYZ�       �	��Xc�A�*

loss5<=:k>�       �	ߌ��Xc�A�*

loss|��<b*t�       �	�4��Xc�A�*

loss���<1T       �	L��Xc�A�*

lossl�B=.���       �	����Xc�A�*

loss��6=R���       �	C7��Xc�A�*

loss�e�=H�x�       �	�߲�Xc�A�*

lossoe�<ѕ�7       �	/���Xc�A�*

loss�Ӏ=����       �	;5��Xc�A�*

lossB� =3��       �	�ٴ�Xc�A�*

lossJ�=� ��       �	�z��Xc�A�*

loss�JH=Jes       �	���Xc�A�*

loss�u=b{9k       �	�϶�Xc�A�*

loss4�=��n       �	����Xc�A�*

loss_��;аX       �	�L��Xc�A�*

loss��	<�ȍ       �	���Xc�A�*

loss%)=��%�       �	��Xc�A�*

loss0<�9�       �	����Xc�A�*

loss?J<.��       �	�!��Xc�A�*

lossEL�;	��s       �	�»�Xc�A�*

loss��f<͙�z       �	�Y��Xc�A�*

loss�=$�d       �	����Xc�A�*

loss;XI=����       �	����Xc�A�*

loss$�=���L       �	�4��Xc�A�*

lossQ=�;��J       �	S;�Xc�A�*

loss�w=�-�t       �	j��Xc�A�*

loss,[�=w_V<       �	��Xc�A�*

loss?(;�j2�       �	+���Xc�A�*

loss+U�<S8�	       �	�A��Xc�A�*

loss���<)�       �	M0��Xc�A�*

loss�7�;;���       �	���Xc�A�*

lossoy=�4�       �	����Xc�A�*

lossd��=�^�z       �	K=��Xc�A�*

loss��*<bY�       �	����Xc�A�*

loss;zA;
�X       �	6s��Xc�A�*

lossEe�;��v%       �	���Xc�A�*

loss� Y<{e�       �	a���Xc�A�*

loss��<��1�       �	�I��Xc�A�*

loss@�4=鮰�       �	����Xc�A�*

loss��#="�Y%       �	?���Xc�A�*

loss��S=:W̿       �	w+��Xc�A�*

lossX5><��        �	����Xc�A�*

loss�G=&�KZ       �	b��Xc�A�*

lossJ��;���|       �	����Xc�A�*

loss�A<Y&%�       �	���Xc�A�*

lossw:�<&���       �	r���Xc�A�*

loss�*�=�3       �	���Xc�A�*

loss/{2=`�ȉ       �	j���Xc�A�*

loss:�.==���       �	����Xc�A�*

loss	U=��,j       �	6��Xc�A�*

loss��=�]�       �	����Xc�A�*

lossc��<��       �	u��Xc�A�*

loss��<r5@
       �	j���Xc�A�*

loss�{=�OY       �	l_��Xc�A�*

lossH�t=��}�       �	y��Xc�A�*

loss�l�;^h%H       �	���Xc�A�*

loss�	C=g\�       �	�f��Xc�A�*

loss{�D=G�4       �	���Xc�A�*

loss��<'&L�       �	����Xc�A�*

lossh�T='k"/       �	�D��Xc�A�*

loss�g�=���       �	����Xc�A�*

loss�:+�^       �	���Xc�A�*

lossV�;<�Y��       �	1��Xc�A�*

loss%N�<��e�       �	����Xc�A�*

loss[]=w6��       �	�x��Xc�A�*

loss��=@��A       �	���Xc�A�*

lossv<r=���       �	o���Xc�A�*

loss}̉<��F       �	1]��Xc�A�*

lossAf�=���       �	���Xc�A�*

loss���<�&       �	/���Xc�A�*

loss��C<�"0�       �	�E��Xc�A�*

loss��l<�ۯ       �	����Xc�A�*

lossX�0=����       �	����Xc�A�*

loss�u�<6�u       �	�2��Xc�A�*

loss(O=ҪmM       �	����Xc�A�*

loss�t+=�b{7       �	a��Xc�A�*

loss���;Na�i       �	D���Xc�A�*

lossR��<m�x       �	����Xc�A�*

loss:'C=7�ֆ       �	���Xc�A�*

loss_,G;I�$A       �	���Xc�A�*

loss�4=���;       �	M��Xc�A�*

loss���;S��       �	����Xc�A�*

loss��G<��]�       �	���Xc�A�*

loss�Ѡ=�@��       �	r4��Xc�A�*

lossf��=��r�       �	����Xc�A�*

lossZ�=�3��       �	�t��Xc�A�*

lossN)�<�Cf�       �	!��Xc�A�*

loss3�;P��       �	d���Xc�A�*

loss8��<��r�       �	*p �Xc�A�*

loss�@{=z�4�       �	��Xc�A�*

loss�N�=.���       �	���Xc�A�*

lossO�=�T�{       �	K�Xc�A�*

loss (=�"�H       �	���Xc�A�*

loss�<���       �	���Xc�A�*

loss�xM;�O/       �	�!�Xc�A�*

loss��<�U�       �	���Xc�A�*

lossI�="�(5       �	�[�Xc�A�*

loss�G�<} ��       �	��Xc�A�*

loss{c'=$D<       �	���Xc�A�*

loss�<�g$F       �	�0�Xc�A�*

loss.z�;����       �	���Xc�A�*

lossl�;��       �	
e�Xc�A�*

loss��6;Q�
       �	���Xc�A�*

loss0�<�h��       �	^�	�Xc�A�*

lossnj�<H�O3       �	�|
�Xc�A�*

loss}0�=���       �	� �Xc�A�*

lossX-<�D�       �	'��Xc�A�*

loss:>.<b�Q       �	:[�Xc�A�*

loss?�!=ר��       �	���Xc�A�*

lossJ~m<��       �	��Xc�A�*

lossP,=�,[f       �	K#�Xc�A�*

loss�h!=¬��       �	��Xc�A�*

loss�w�<؃�       �	֪�Xc�A�*

lossz�<���       �	�H�Xc�A�*

lossO��<�$-N       �	���Xc�A�*

loss&�<�:�       �	g��Xc�A�*

loss��6=G^?�       �	_D�Xc�A�*

loss�<�^b�       �	H��Xc�A�*

loss{��<�5�       �	{��Xc�A�*

loss(�<��z       �	6#�Xc�A�*

loss6��;��K�       �	���Xc�A�*

lossłT<�a6       �	�z�Xc�A�*

loss�L=���       �	n�Xc�A�*

lossH�"<+��       �	ҫ�Xc�A�*

loss:ޓ<M�5       �	I�Xc�A�*

lossH"/<��a       �	g��Xc�A�*

loss(- =r�;�       �	��Xc�A�*

loss|H<�ՓR       �	�.�Xc�A�*

loss�B,<�g�d       �	��Xc�A�*

loss��;�+��       �	Ӿ�Xc�A�*

loss;h�<k��       �	%��Xc�A�*

lossX� <��l�       �	?�Xc�A�*

losse��<�;       �	��Xc�A�*

loss��="��       �	���Xc�A�*

loss���;�       �	��Xc�A�*

loss��="�W       �	�Xc�A�*

lossd�z="�_�       �	c��Xc�A�*

lossz�;J�-7       �	D� �Xc�A�*

loss��<���       �	��!�Xc�A�*

loss�J�;��W�       �	ݔ"�Xc�A�*

lossH@�;��R�       �	6#�Xc�A�*

losstP=ݤTl       �	�!$�Xc�A�*

loss��<�Y�       �	%�Xc�A�*

loss���=F���       �	r�%�Xc�A�*

loss9�<"�B�       �	c&�Xc�A�*

loss��@;Z(�       �	H�&�Xc�A�*

lossRw�9��       �	B�'�Xc�A�*

lossi>�=��՜       �	�-(�Xc�A�*

loss��w<����       �	i�(�Xc�A�*

loss6��<
���       �	=`)�Xc�A�*

loss�M�=�q�       �	��)�Xc�A�*

lossTu�<�6�       �	?�*�Xc�A�*

lossz۴;c�&       �	�(+�Xc�A�*

loss�ͷ=�;$       �	�+�Xc�A�*

lossz�<���       �	�X,�Xc�A�*

lossڱ�<��T>       �	�,�Xc�A�*

loss3�B=�
,�       �	�-�Xc�A�*

loss�]=��       �	�&.�Xc�A�*

lossJ<��       �	��.�Xc�A�*

loss�v�= ���       �	�/�Xc�A�*

losse�;>�       �	�&0�Xc�A�*

lossD#�;�0�       �	<�0�Xc�A�*

loss�S�=}�B       �	p^1�Xc�A�*

lossH��<�ڛx       �	w�1�Xc�A�*

lossʫ�=U�*�       �	Ԝ2�Xc�A�*

lossr��<���n       �	�<3�Xc�A�*

loss�}
=��B       �	.�3�Xc�A�*

loss䝤;U�L&       �	��4�Xc�A�*

loss?�5=�W       �	�,5�Xc�A�*

lossH�$<��U       �	��5�Xc�A�*

loss���<�e��       �	>Z6�Xc�A�*

lossڞ�<f0�       �	#�6�Xc�A�*

loss��<l9��       �	C�7�Xc�A�*

loss#_=���w       �	�"8�Xc�A�*

loss횛<H,ߨ       �	ϻ8�Xc�A�*

loss#7b<�huL       �	�T9�Xc�A�*

loss�1=�֓�       �	��9�Xc�A�*

lossi��=M��$       �	^�:�Xc�A�*

loss�`�<�UBe       �	�W;�Xc�A�*

loss�P_=N�J�       �	�a<�Xc�A�*

loss�*�<�[/       �	��<�Xc�A�*

loss$e<�D       �	2�=�Xc�A�*

loss0-<πxa       �	9*>�Xc�A�*

loss��8<�C�`       �	@�>�Xc�A�*

lossN�;-[�       �	6v?�Xc�A�*

lossv�<�_�w       �	)%@�Xc�A�*

loss��<�;4�       �	��@�Xc�A�*

loss��`=)*�       �	�YA�Xc�A�*

loss;F�<{$<�       �	�B�Xc�A�*

loss��F=�k��       �	?�B�Xc�A�*

loss��}<q��       �	�_C�Xc�A�*

lossc�=��[       �	b�C�Xc�A�*

lossü�;m�X       �	�D�Xc�A�*

loss�Z�:��]�       �	4hE�Xc�A�*

lossr�<͑hT       �	�F�Xc�A�*

lossxA�<'z�       �	��F�Xc�A�*

loss�
�=NT�v       �	�nG�Xc�A�*

lossEnK=���*       �	�H�Xc�A�*

loss	!�=�+�x       �	��H�Xc�A�*

loss�*=Wmer       �	.9I�Xc�A�*

lossE-�:Pc       �	��I�Xc�A�*

loss��<���       �	)uJ�Xc�A�*

loss�=f[d       �	�*K�Xc�A�*

loss[��<q�
       �	'�K�Xc�A�*

losseA<rإ9       �	gdL�Xc�A�*

loss�ր=��       �	��L�Xc�A�*

loss,?|;�01       �	��M�Xc�A�*

loss�	H=�?       �	�RN�Xc�A�*

loss���;��        �	��O�Xc�A�*

loss
��<�ca8       �	��P�Xc�A�*

loss� �;�w       �	=Q�Xc�A�*

loss�&A<ǝL       �	@�Q�Xc�A�*

loss;7�<x��o       �	�tR�Xc�A�*

loss��<}jo�       �	 S�Xc�A�*

loss!'B<®��       �	\�S�Xc�A�*

loss�i;=���       �	bKT�Xc�A�*

loss$�L<�~�       �	��T�Xc�A�*

lossW�D;��       �	�U�Xc�A�*

lossr$�<
OD�       �	�V�Xc�A�*

lossP�;���U       �	}�V�Xc�A�*

loss)wT<�Ǭ/       �	jKW�Xc�A�*

loss�M�=�
QL       �	{�W�Xc�A�*

lossd6{=%�]�       �	 �X�Xc�A�*

loss��d<h       �	�&Y�Xc�A�*

loss��<n�R�       �	M�Y�Xc�A�*

loss�8;�`1       �	�RZ�Xc�A�*

loss.xP<����       �	Y�[�Xc�A�*

loss��Z=D��       �	�/\�Xc�A�*

lossJ'�:�{j?       �	��\�Xc�A�*

lossWZ<L�$)       �	�^�Xc�A�*

lossBt=��       �	��^�Xc�A�*

loss���=W!��       �	1�_�Xc�A�*

loss�ӱ<��       �	�o`�Xc�A�*

loss�<J��       �	d�a�Xc�A�*

lossg�<�$G'       �	Мb�Xc�A�*

loss��=-ѣv       �	�=c�Xc�A�*

loss���<��-       �	��c�Xc�A�*

loss�"�<N���       �	_�d�Xc�A�*

loss�ʕ;.=F       �	O]f�Xc�A�*

loss��<�g�{       �	;g�Xc�A�*

loss$X; �7�       �	�h�Xc�A�*

loss�j�<N�       �	
�i�Xc�A�*

loss���=�D3�       �	�k�Xc�A�*

loss���<i m>       �	�l�Xc�A�*

loss���;�[       �	
�l�Xc�A�*

loss(=�=_ ��       �	��m�Xc�A�*

lossן�:]q       �	�n�Xc�A�*

loss���;�X�       �	~p�Xc�A�*

lossH``<q�\�       �	�
q�Xc�A�*

losso�.<�z	$       �	�r�Xc�A�*

loss���=���       �	y�r�Xc�A�*

loss��5=T,�       �	�~s�Xc�A�*

lossҁ�<��t�       �	�#t�Xc�A�*

loss��<�F�V       �	��t�Xc�A�*

lossT�2=�9��       �	p�u�Xc�A�*

loss!< /
o       �	rjv�Xc�A�*

loss�<T��       �	0w�Xc�A�*

loss��F<�?��       �	h#x�Xc�A�*

loss.��;�.Q^       �	�x�Xc�A�*

lossf2!=/&<       �	�]y�Xc�A�*

loss��=thJ�       �	��y�Xc�A�*

loss�H�=$�c       �	�z�Xc�A�*

loss�ST=�&ߏ       �	�R{�Xc�A�*

loss �%=鼿�       �	I�{�Xc�A�*

lossX�{<@o /       �	��|�Xc�A�*

lossl��<�0��       �	�8}�Xc�A�*

loss&�*;f��       �	|�}�Xc�A�*

loss8"=<ms��       �	�x~�Xc�A�*

lossix<���       �		�Xc�A�*

lossG)<S!Ry       �	1��Xc�A�*

loss �h=��       �	J��Xc�A�*

loss�V�;��9R       �	f��Xc�A�*

loss��T<�ߔ       �	���Xc�A�*

loss�D<z`�J       �	pa��Xc�A�*

loss�@O=�/�       �	r��Xc�A�*

loss_�; 1�       �	����Xc�A�*

losso]N;C�2�       �	_��Xc�A�*

lossQ,�; ���       �	��Xc�A�*

loss���;Q���       �	禅�Xc�A�*

lossҰ.=����       �	W��Xc�A�*

loss��#;���       �	v���Xc�A�*

loss��<c���       �	׈�Xc�A�*

lossx ,<6p-�       �	�o��Xc�A�*

loss,M]<$<�5       �	��Xc�A�*

loss��;`��k       �	����Xc�A�*

lossʹ=#|�       �	�Z��Xc�A�*

loss�4<��B�       �	s���Xc�A�*

lossA<�p�       �	W���Xc�A�*

loss��=5�\�       �	����Xc�A�*

loss�)<��R�       �	�1��Xc�A�*

loss��O<["�:       �	���Xc�A�*

loss}�<�έ       �	ꕏ�Xc�A�*

lossSs)=G6       �	&9��Xc�A�*

loss%�<"-�a       �	Ԑ�Xc�A�*

lossi<;�$k�       �	����Xc�A�*

lossu=n2(       �	@L��Xc�A�*

lossj.=|.J�       �	j���Xc�A�*

loss�-�<{�Y       �	����Xc�A�*

loss3_:�H#       �	w.��Xc�A�*

loss[�C;l"[       �	Ŕ�Xc�A�*

lossf�<eѝ       �	�^��Xc�A�*

loss\�=2˿       �	e���Xc�A�*

loss2�S<L�x�       �	���Xc�A�*

loss�w�<Ǫ��       �		2��Xc�A�*

loss�3�=�zGN       �	�Η�Xc�A�*

loss*�=�g       �	0h��Xc�A�*

loss��<nL9�       �	m��Xc�A�*

loss�=Ľw�       �	����Xc�A�*

loss�3�<[Q       �	J@��Xc�A�*

loss�b#=���b       �	Uۚ�Xc�A�*

loss]�;_/�       �		���Xc�A�*

loss�r
=
>!       �	W&��Xc�A�*

loss�%=�^��       �	����Xc�A�*

loss�P=ʦ^K       �	[��Xc�A�*

loss���<-�ǰ       �	@���Xc�A�*

lossJ��<��͆       �	����Xc�A�*

loss��&=l��K       �	9��Xc�A�*

loss���<�6�(       �	�ԟ�Xc�A�*

lossXi�<,N]�       �	2s��Xc�A�*

lossc��=(V.z       �	���Xc�A�*

lossW}�=�P�       �	����Xc�A�*

loss�@�;�M�       �	�R��Xc�A�*

loss�<Ȑ�`       �	"���Xc�A�*

losss�=�R��       �	_���Xc�A�*

loss�1)<����       �	]2��Xc�A�*

loss(�b;� �       �	tϤ�Xc�A�*

loss���;c3�       �	�h��Xc�A�*

lossL�<,�w�       �	
��Xc�A�*

loss
K<����       �	���Xc�A�*

lossf�=l��       �	�o��Xc�A�*

loss�6";�v��       �	���Xc�A�*

loss�M�<�;       �	>���Xc�A�*

loss�É<����       �	R��Xc�A�*

lossR/�;�ͩ�       �	���Xc�A�*

lossm<�oh�       �	����Xc�A�*

loss�<�;���;       �	I/��Xc�A�*

loss3<Bbr�       �	�ȫ�Xc�A�*

loss��<���       �	�q��Xc�A�*

loss}@�:?�       �	���Xc�A�*

loss֝;h�K$       �	г��Xc�A�*

lossDtB=,6�V       �	���Xc�A�*

loss�'_=�8�       �	g���Xc�A�*

loss�MV=�L��       �	��Xc�A�*

loss�5~=�N�       �	O���Xc�A�*

loss6��<N��        �	^L��Xc�A�*

loss�("<��c�       �	 ��Xc�A�*

loss�W�<R��V       �	,��Xc�A�*

loss��Q<���       �	u!��Xc�A�*

loss�m�<Ve�       �	M���Xc�A�*

lossh)=��*o       �	t_��Xc�A�*

loss��=�/��       �	����Xc�A�*

lossX;�7t�       �	����Xc�A�*

loss���<Nh��       �	�*��Xc�A�*

loss*�o<:�Om       �	�¶�Xc�A�*

loss�<7       �	��Xc�A�*

lossF�;3{�       �	h���Xc�A�*

loss�:�<:�g�       �	-��Xc�A�*

lossQ��=�yV�       �	�ǹ�Xc�A�*

lossr2<�w��       �	�c��Xc�A�*

loss��x;!��       �	]���Xc�A�*

loss�JX=�c1       �	���Xc�A�*

loss�\q:��(       �	en��Xc�A�*

lossWP<l��R       �	�
��Xc�A�*

loss�^<WICh       �	����Xc�A�*

loss$��;�9��       �	�D��Xc�A�*

loss��<���       �	޾�Xc�A�*

loss6Ǣ=!4�       �	�r��Xc�A�*

loss])<�AQ�       �	���Xc�A�*

loss�d�;���       �	Z���Xc�A�*

lossl�<��       �	�1��Xc�A�*

loss_�,<�<��       �	����Xc�A�*

loss�UC<J��X       �	rk��Xc�A�*

loss�;Mǅ}       �	� ��Xc�A�*

loss8��<R�I       �	����Xc�A�*

loss���<Wd�H       �	����Xc�A�*

loss� <�&uy       �	�E��Xc�A�*

loss� <�G3       �	Q���Xc�A�*

lossT�;��       �	u��Xc�A�*

loss��<4��       �	A��Xc�A�*

loss�I=fUa�       �	¢��Xc�A�*

loss�<�l��       �	�h��Xc�A�*

loss l=�i5�       �	���Xc�A�*

lossd��<��W       �	���Xc�A�*

loss��=���[       �	p?��Xc�A�*

lossE�:r7m       �	����Xc�A�*

losss��;�\m       �	�t��Xc�A�*

loss��=�^       �	���Xc�A�*

loss�'�;�*       �	����Xc�A�*

loss��;1���       �	�H��Xc�A�*

loss�I=�hx�       �	?��Xc�A�*

loss��d=8�4       �	����Xc�A�*

loss��;���       �	TT��Xc�A�*

lossj�9=ɧ       �	����Xc�A�*

lossŚ=<�R       �	����Xc�A�*

loss�^�:J� f       �	>$��Xc�A�*

loss�H:�}O       �	����Xc�A�*

loss��0;_�	       �	�P��Xc�A�*

loss���<���]       �	!���Xc�A�*

loss#DQ=!�l       �	F~��Xc�A�*

loss���=>*�       �	� ��Xc�A�*

loss�!�:+f5�       �	o���Xc�A�*

loss.�t;�Ze�       �	A���Xc�A�*

loss<�(<�V��       �	:?��Xc�A�*

loss���=Y�|�       �	R���Xc�A�*

loss�)<.rx�       �	�i��Xc�A�*

loss`x@<�녓       �	��Xc�A�*

lossc�K<�_g       �	g���Xc�A�*

loss��E<yV�       �	�.��Xc�A�*

loss63=�,ȯ       �	����Xc�A�*

loss��<�[4�       �	7p��Xc�A�*

loss��<���       �	�P��Xc�A�*

loss�lo=��2�       �	���Xc�A�*

loss���<��       �	dv��Xc�A�*

lossx�;4\�       �	�K��Xc�A�*

loss1E=�h       �	.���Xc�A�*

loss�ʄ<"{�       �	i���Xc�A�*

loss�=<!!k�       �	H��Xc�A�*

lossOի<�ı�       �	����Xc�A�*

loss�&<�m �       �	d���Xc�A�*

loss!��<���       �	M���Xc�A�*

loss�U�<#B&�       �	CW��Xc�A�*

lossw>�1>2       �	���Xc�A�*

loss�ң<��I       �	����Xc�A�*

loss��<�YQ       �	'���Xc�A�*

loss.J�:� �       �	�r��Xc�A�*

loss\��;���       �	c���Xc�A�*

lossN)�<�늸       �	�q��Xc�A�*

loss(Ȯ=s���       �	tB��Xc�A�*

loss��=�Ig       �	)?��Xc�A�*

lossZ9<;��?(       �	����Xc�A�*

loss*�=֓Td       �	#���Xc�A�*

lossD/=q;��       �	jh��Xc�A�*

lossw�X=X�&O       �	'��Xc�A�*

loss6z/;�ra<       �	���Xc�A�*

loss�D;t^6       �	����Xc�A�*

loss<l=1ƴ       �	&R��Xc�A�*

loss��;�x       �	.���Xc�A�*

lossj�<��       �	���Xc�A�*

loss�wp=�z�(       �	�T��Xc�A�*

loss�A=?%*       �	)&��Xc�A�*

lossz+�=�gb       �	���Xc�A�*

lossج�;7       �	7q��Xc�A�*

loss���<c'�       �	�V��Xc�A�*

loss�> <�C       �	���Xc�A�*

loss^�=|�       �	���Xc�A�*

loss��5:=s.�       �	�:��Xc�A�*

loss���;Ĕ��       �	����Xc�A�*

loss��=����       �	�p��Xc�A�*

loss�<c	�:       �	����Xc�A�*

lossŒ�=���       �	�1��Xc�A�*

loss��%<�K��       �	���Xc�A�*

loss�~K<oe��       �	3���Xc�A�*

loss�JP=\�4�       �	H3��Xc�A�*

lossL�<b&�u       �	����Xc�A�*

loss.�@<>Y��       �	l���Xc�A�*

loss	�=ZU^       �	�3��Xc�A�*

lossi��=8��"       �	����Xc�A�*

lossIr>=����       �	�o��Xc�A�*

lossc��<�u֡       �	(��Xc�A�*

loss.�c<
��       �	����Xc�A�*

loss�w<��S�       �	�� �Xc�A�*

loss@q�; _�       �	�2�Xc�A�*

lossۅ;�=�       �	���Xc�A�*

loss
�+=�bt       �	��Xc�A�*

loss�Jm=��u3       �	�}�Xc�A�*

loss� =lv��       �	�(�Xc�A�*

loss,�<+��4       �	z �Xc�A�*

loss!�Y;#�Z�       �	���Xc�A�*

loss���=Qc�       �	�J�Xc�A�*

loss��=vG�       �	���Xc�A�*

loss���<�
h       �	}��Xc�A�*

loss�=y*��       �	37�Xc�A�*

loss��u=��w       �	��Xc�A�*

lossq�H<P��T       �	*�	�Xc�A�*

loss�=�`��       �	�O
�Xc�A�*

loss��3=4�l�       �	��
�Xc�A�*

loss��<����       �	H��Xc�A�*

lossv�p<8�/�       �	4�Xc�A�*

loss��~<G��       �	���Xc�A�*

lossRE}<M��J       �	�q�Xc�A�*

loss�[�<�?�       �	��Xc�A�*

loss%A�;��o       �	���Xc�A�*

loss�ق<!�       �	.U�Xc�A�*

loss�;����       �	8��Xc�A�*

loss�e;��kR       �	��Xc�A�*

loss��Z=�;��       �	o/�Xc�A�*

lossC\ =��D=       �	���Xc�A�*

loss��X<G`�"       �	�_�Xc�A�*

loss?�=��z       �	"��Xc�A�*

lossJ��;�V~~       �	|��Xc�A�*

loss�><����       �	LP�Xc�A�*

loss��=��:       �	���Xc�A�*

loss?��;b	��       �	ѐ�Xc�A�*

loss3<BU�Y       �	�'�Xc�A�*

loss��<���       �	 �Xc�A�*

loss��!=C��Q       �	k��Xc�A�*

lossRE<T�z�       �	35�Xc�A�*

loss͆J<�4�+       �	j��Xc�A�*

loss�4=�r#�       �	3��Xc�A�*

loss+J�=��<       �	�(�Xc�A�*

loss�Ӄ<�(�       �	0��Xc�A�*

lossK?<��f�       �	P��Xc�A�*

loss�(6<ղ�       �	�V�Xc�A�*

loss݊.<�ta       �	���Xc�A�*

loss䔽;Qs�       �	��Xc�A�*

lossf��<ML�       �	y �Xc�A�*

loss��<�pO/       �	ǹ�Xc�A�*

lossӣ�<�� �       �	xb�Xc�A�*

loss<q=߭�c       �	9 �Xc�A�*

lossR9<�#        �	n� �Xc�A�*

loss�]j;7c+�       �	�;!�Xc�A�*

lossH<�;V=.       �	&�!�Xc�A�*

loss�z�<�G�       �	��"�Xc�A�*

loss��<Gl��       �	T#�Xc�A�*

loss,�<<x%P       �	��#�Xc�A�*

lossċ�=��P�       �	�]$�Xc�A�*

loss(�=���       �	��$�Xc�A�*

loss�H�=��(�       �	£%�Xc�A�*

loss�]'=�ώ�       �	�@&�Xc�A�*

lossҝ;�=3F       �	{�&�Xc�A�*

loss;i<��       �	��'�Xc�A�*

lossM�]:����       �	)@(�Xc�A�*

lossP�;O�       �	u�(�Xc�A�*

loss$u:�>��       �	0�)�Xc�A�*

loss&0==����       �	�*�Xc�A�*

losse!<7�U8       �	��*�Xc�A�*

loss���=�8�r       �	�z+�Xc�A�*

loss�G�<���       �	,,�Xc�A�*

loss�o
=ڨ�a       �	[�,�Xc�A�*

loss���=b{`�       �	&s-�Xc�A�*

loss��=��       �	�D.�Xc�A�*

loss���;g/�p       �	�.�Xc�A�*

loss�k=eFA{       �	��/�Xc�A�*

loss�}�;h9F       �	�00�Xc�A�*

loss��=�^'�       �	�0�Xc�A�*

lossq�#<Q�       �	�o1�Xc�A�*

lossi��<�:H�       �	R2�Xc�A�*

loss=��:�o�{       �	.�2�Xc�A�*

loss�?f=���*       �	ND3�Xc�A�*

loss�Y<�nN�       �	��3�Xc�A�*

lossD=��X       �	��4�Xc�A�*

loss��8=���c       �	��5�Xc�A�*

loss�&<lh��       �	Z.6�Xc�A�*

loss���<}Q7�       �	9�6�Xc�A�*

loss�D=��A       �	Wx7�Xc�A�*

lossMA?=��x       �	}$8�Xc�A�*

loss�P;)#x�       �	��8�Xc�A�*

loss�b�:ea\�       �	��9�Xc�A�*

loss/ϡ;�{ӹ       �	G:�Xc�A�*

loss��0=����       �	�:�Xc�A�*

loss�h�;a9M�       �	m�;�Xc�A�*

loss6�=��       �	mV<�Xc�A�*

loss%�;�4�L       �	�
=�Xc�A�*

lossZX<f�[�       �	��=�Xc�A�*

loss���<YYu!       �	�r>�Xc�A�*

loss��<_�       �	p(?�Xc�A�*

losshD�=b�S�       �	��?�Xc�A�*

loss`��<���       �	ٱ@�Xc�A�*

loss�<<=Y�       �	�aA�Xc�A�*

losst <�׈�       �	{B�Xc�A�*

loss5�<be��       �	��B�Xc�A�*

loss7�8=c�       �	DmC�Xc�A�*

lossQU!;����       �	3D�Xc�A�*

lossg<��{       �	�D�Xc�A�*

lossC��<��o       �	lvE�Xc�A�*

loss��3=qց&       �	$&F�Xc�A�*

loss�	t<c�oF       �	��F�Xc�A�*

losslU�;�Ԟ>       �	��G�Xc�A�*

lossj	=�2;       �	�vH�Xc�A�*

loss�Z�<_7�       �	�KI�Xc�A�*

loss�~�;�ܨ\       �	W	J�Xc�A�*

loss\�=b*��       �	�J�Xc�A�*

loss[�n=kI�       �	�ZK�Xc�A�*

loss��=�R�5       �	L�Xc�A�*

loss�;�؟       �	v�L�Xc�A�*

lossX��=L��       �	�GM�Xc�A�*

loss��=�!ө       �	8�M�Xc�A�*

lossMB�;���f       �	��N�Xc�A�*

lossT��:P^�~       �	�4O�Xc�A�*

loss���:�N�       �	��O�Xc�A�*

loss���<2�9�       �	I�P�Xc�A�*

lossX�c<��       �	�4Q�Xc�A�*

loss*�p;~��O       �	; R�Xc�A�*

loss�?*:���!       �	�S�Xc�A�*

loss� �;�WD�       �	��S�Xc�A�*

loss)�K<�Pw�       �	�dT�Xc�A�*

loss�ZJ:�#       �	e U�Xc�A�*

lossx�i:G2�W       �	ԛU�Xc�A�*

loss{ͪ;�>A
       �	�@V�Xc�A�*

loss/��<c��v       �	��V�Xc�A�*

loss�N<JF�
       �	�W�Xc�A�*

loss)[9: ��       �	�%X�Xc�A�*

loss��8==�       �	�X�Xc�A�*

loss&�6=�
�       �	t\Y�Xc�A�*

lossƾ�:�{�       �	Z�Xc�A�*

loss`�;>p�|Z       �	��Z�Xc�A�*

loss�V<��2w       �	'j[�Xc�A�*

loss̠v=R���       �	\�Xc�A�*

loss���<W�       �	n5]�Xc�A�*

loss3�;;�*�U       �	_�]�Xc�A�*

loss��#=��7       �	��^�Xc�A�*

loss��=k�*�       �	ض_�Xc�A�*

loss�}�<���       �	�`�Xc�A�*

loss-��<��R\       �	�8a�Xc�A�*

loss��l;-�       �	Xb�Xc�A�*

loss��0=Ec0�       �	��b�Xc�A�*

loss�A=�N?Y       �	5�c�Xc�A�*

loss��=(�        �	G>d�Xc�A�*

lossM: =mP/       �	��d�Xc�A�*

loss-Ha<�n�,       �	�e�Xc�A�*

loss���<p�       �	�)f�Xc�A�*

loss��&<8I��       �	��f�Xc�A�*

loss�<ĝ       �	ig�Xc�A�*

lossD�<	^��       �	dh�Xc�A�*

lossT��<C
e       �	��h�Xc�A�*

loss�dd<2���       �	6i�Xc�A�*

loss��<c��       �	��i�Xc�A�*

loss/�<-��0       �		nj�Xc�A�*

loss�w�;����       �	Kk�Xc�A�*

loss%��;�a	       �	0�k�Xc�A�*

loss��<M,�#       �	�2l�Xc�A�*

loss�-�<un�       �	)�l�Xc�A�*

loss�=�<д\       �	�cm�Xc�A�*

lossL�q=�X��       �	��m�Xc�A�*

loss�0<�Rw�       �	�n�Xc�A�*

loss�<� �       �	W@o�Xc�A�*

loss�Y"=K`��       �	�o�Xc�A�*

loss�i�<�ɽp       �	��p�Xc�A�*

loss���;m�~       �	�0q�Xc�A�*

loss�;y{�       �	n�q�Xc�A�*

loss�p�;��j       �	s�r�Xc�A�*

loss��$=���k       �	�'s�Xc�A�*

loss` �<�S�       �	6�s�Xc�A�*

loss۩3<��\       �	4dt�Xc�A�*

loss�E=�Y(>       �	c
u�Xc�A�*

loss�R;��.       �	Y�u�Xc�A�*

loss�$H<C�j       �	2<v�Xc�A�*

loss��_=�w:       �		�v�Xc�A�*

loss�*R;u�(       �	�vw�Xc�A�*

loss�i�<w�       �	�x�Xc�A�*

loss�7�=80��       �	p�x�Xc�A�*

loss�;��r       �	+ny�Xc�A�*

loss==f<]/��       �	�z�Xc�A�*

loss{4 <��y�       �	�z�Xc�A�*

lossw#=>�F       �	�Y{�Xc�A�*

lossϼ�=�ef�       �	%Y��Xc�A�*

loss�-=��t�       �	O��Xc�A�*

loss�x�=���       �	����Xc�A�*

loss�<b;�G       �	1B��Xc�A�*

loss���<N;$       �	 ��Xc�A�*

loss�=s
�       �	���Xc�A�*

loss�>=KJ��       �	�1��Xc�A�*

loss�=��x>       �	8ڗ�Xc�A�*

loss��(=n���       �	g}��Xc�A�*

loss	�r=oD��       �	���Xc�A�*

lossڱ;�ۆ!       �	����Xc�A�*

loss��<����       �	Ef��Xc�A�*

loss�{.=��       �	t	��Xc�A�*

loss��*=�#�3       �	X���Xc�A�*

loss�t=���       �	8���Xc�A�*

loss��==��.       �	�"��Xc�A�*

loss*��;�urw       �	Ͼ��Xc�A�*

lossw��;���       �	Ac��Xc�A�*

loss�+i<d'^�       �	����Xc�A�*

loss��3=$*��       �	����Xc�A�*

loss�Ĥ<x�~o       �	�1��Xc�A�*

loss���=�       �	�ؠ�Xc�A�*

loss���;��J�       �	{��Xc�A�*

lossG3�=�2�       �	�"��Xc�A�*

loss#+6<����       �	�Ȣ�Xc�A�*

loss���<p��       �	�m��Xc�A�*

lossF�;!~+�       �	N��Xc�A�*

lossT�;Wȣy       �	����Xc�A�*

loss�<�C��       �	0I��Xc�A�*

loss��<c�>�       �	���Xc�A�*

loss��<q8��       �	K���Xc�A�*

loss14=�
��       �	w,��Xc�A�*

loss�<gu��       �	3ħ�Xc�A�*

loss��=����       �	Qi��Xc�A�*

loss�&=-�[       �	,(��Xc�A�*

loss���<���        �	���Xc�A�*

lossa�[<���       �	GZ��Xc�A�*

lossF&=m��       �	K��Xc�A�*

lossT��=�I�       �	p���Xc�A�*

loss�;=mЅI       �	w/��Xc�A�*

lossn�3=�y�       �	�Ĭ�Xc�A�*

loss�m<Η       �	�i��Xc�A�*

loss�U;�q&o       �	���Xc�A�*

lossI�<���s       �	����Xc�A�*

loss
X(=,�*n       �	���Xc�A�*

loss���<�I��       �	ob��Xc�A�*

lossHd�<�|��       �	e ��Xc�A�*

loss�uF=�<�?       �	¤��Xc�A�*

lossF=����       �	�:��Xc�A�*

loss\ �:V        �	�ղ�Xc�A�*

loss$qi=T���       �	jj��Xc�A�*

loss��w<����       �	���Xc�A�*

loss͹l<��}       �	���Xc�A�*

loss\>�e�3       �	P:��Xc�A�*

lossZګ<AD       �	�ε�Xc�A�*

loss�n�;�= �       �	�c��Xc�A�*

lossA��:(�+       �	W��Xc�A�*

loss�:*HL        �	E���Xc�A�*

loss췃;�=�       �	@3��Xc�A�*

loss�d.=�;�       �	˸�Xc�A�*

loss,"�=օJ�       �	+n��Xc�A�*

loss�p�;f�R�       �	W��Xc�A�*

loss2E <�m�       �	衺�Xc�A�*

loss�z=�nJI       �	�@��Xc�A�*

lossZ�=���       �	_��Xc�A�*

loss �<���       �	+���Xc�A�*

loss���<4^32       �	ᗽ�Xc�A�*

lossJ$�<]���       �	A.��Xc�A�*

loss!�.=��~4       �	�˾�Xc�A�*

lossZ��<��H@       �	�b��Xc�A�*

loss�\<;vݷ       �	���Xc�A�*

loss=�=���X       �	-���Xc�A�*

lossTv<��6�       �	[D��Xc�A�*

loss�
T=J�       �	U���Xc�A�*

loss�I<���       �	�t��Xc�A�*

loss8r�<L��       �	l!��Xc�A�*

loss�,;�0G       �	���Xc�A�*

loss��9;�m       �	����Xc�A�*

loss_��<�h�6       �	���Xc�A�*

lossA�<��y�       �	v���Xc�A�*

lossw�?<�|��       �	=`��Xc�A�*

loss!��<�6r       �	���Xc�A�*

loss�u�<����       �	���Xc�A�*

loss�w<A"�2       �	�T��Xc�A�*

loss���<:�K       �	w���Xc�A�*

loss%7;���O       �	ۊ��Xc�A�*

loss���;�&�       �	�&��Xc�A�*

lossԫ�=،<�       �	���Xc�A�*

loss�5;��       �	�^��Xc�A�*

loss|��;B�r�       �	����Xc�A�*

lossZ.=�       �	���Xc�A�*

loss])<�W,�       �	E+��Xc�A�*

lossX�<<�ٱa       �	;���Xc�A�*

loss�چ<(�       �	"p��Xc�A�*

loss�<6Z��       �	���Xc�A�*

loss�]�;��2s       �	���Xc�A�*

loss:�9=8��       �	�D��Xc�A�*

loss?�<���       �	����Xc�A�*

lossd�V=��~       �	~s��Xc�A�*

lossF�<����       �	Z��Xc�A�*

loss�
�=FRC       �	l���Xc�A�*

loss�'�; A	y       �	�G��Xc�A�*

loss{�#<���       �	{���Xc�A�*

loss/&�<n˫y       �	g{��Xc�A�*

loss�D�<��O       �	���Xc�A�*

lossϗ>=�t       �	L���Xc�A�*

loss���;AA�0       �	c��Xc�A�*

lossmw\;���       �	���Xc�A�*

loss��><bsJ�       �	ѭ��Xc�A�*

loss7x8<��N       �	�X��Xc�A�*

loss�m�<�9�c       �	��Xc�A�*

losspm�<㛬U       �	Q���Xc�A�*

loss�%�<r��I       �	�?��Xc�A�*

loss��<����       �	����Xc�A�*

loss��<�M_       �	9{��Xc�A�*

loss��+=V���       �	���Xc�A�*

loss�<�̣       �	���Xc�A�*

lossz~(=`���       �	u��Xc�A�*

loss(�=��D�       �		���Xc�A�*

loss��<�L٭       �	Dn��Xc�A�*

loss��;]�F�       �	�@��Xc�A�*

loss��;k��N       �	Q���Xc�A�*

loss8�%<���n       �	�Y��Xc�A�*

loss���;McW       �	}��Xc�A�*

loss3T�;���"       �	����Xc�A�*

loss��p=��\       �	dw��Xc�A�*

loss��<&*�       �	���Xc�A�*

loss�+�<i��       �	���Xc�A�*

lossL�R<���9       �	+Q��Xc�A�*

loss��<x�{       �	S���Xc�A�*

loss	�;�W�       �	{���Xc�A�*

loss,��:\y�       �	�"��Xc�A�*

loss���<��LP       �	����Xc�A�*

loss�r�;|�       �	)\��Xc�A�*

lossަ�<���       �	����Xc�A�*

loss�װ<L8G�       �	���Xc�A�*

loss#��<��J       �	�-��Xc�A�*

loss,P<�K�       �	2���Xc�A�*

loss�<09�       �	>[��Xc�A�*

loss-;=��Z       �	f���Xc�A�*

loss��:���m       �	����Xc�A�*

loss��c=���       �	� ��Xc�A�*

lossT��<��
       �	����Xc�A�*

loss.�=p�U       �	�M��Xc�A�*

loss%;=F���       �	����Xc�A�*

loss1�e<��-       �	:y��Xc�A�*

lossW3=����       �	���Xc�A�*

loss6;�:�b�       �	����Xc�A�*

lossJ�;�%�       �	�6��Xc�A�*

lossL��;�܃       �	p���Xc�A�*

loss$UH:\�       �	*s��Xc�A�*

lossm��<��ʣ       �	h��Xc�A�*

lossz��<~
�       �	p���Xc�A�*

lossV�<��Z       �	;��Xc�A�*

loss�\=�jZ�       �	����Xc�A�*

loss�[;}�mi       �	�p��Xc�A�*

lossj�;�l�0       �	���Xc�A�*

loss��j=��'       �	2���Xc�A�*

loss=k�;#َr       �	�>��Xc�A�*

lossX��:��a       �	����Xc�A�*

loss��<-hR{       �	qs��Xc�A�*

loss���;/�7       �		��Xc�A�*

loss�=��O       �	����Xc�A�*

loss�<G�\       �	���Xc�A�*

losss�$<����       �	L4��Xc�A�*

lossϟ=T8�       �	_���Xc�A�*

lossX�);�[Ҳ       �	ni��Xc�A�*

loss8o=B�E       �	���Xc�A�*

loss���:W��       �	����Xc�A�*

lossw-<���!       �	�b �Xc�A�*

loss�<��c�       �	�� �Xc�A�*

lossj�;S/�J       �	���Xc�A�*

loss��+;��M�       �	�9�Xc�A�*

loss�{D<i)�Q       �	��Xc�A�*

loss�?<��       �	�s�Xc�A�*

loss��=��#       �	�Xc�A�*

loss�J�<Y�       �	��Xc�A�*

lossi�<�O�       �	C�Xc�A�*

loss���=$��_       �	��Xc�A�*

lossc�m:n���       �	�r�Xc�A�*

loss�:�\       �	��Xc�A�*

loss�V<�brK       �	���Xc�A�*

loss�e";�6F�       �	��Xc�A�*

lossN��:�@f�       �	E�	�Xc�A�*

loss���<�g�}       �	oJ
�Xc�A�*

loss�	=c��D       �	��
�Xc�A�*

loss�F%<"��       �	���Xc�A�*

loss�YE<�/s/       �	 7�Xc�A�*

loss�\p=���G       �	���Xc�A�*

lossOP�;����       �	w�Xc�A�*

loss�<��ѕ       �	��Xc�A�*

lossF�<�;^L       �	���Xc�A�*

lossq�O;6T�S       �	�m�Xc�A�*

loss���<c�o       �	P�Xc�A�*

lossnw�:֪�A       �	���Xc�A�*

loss�z<�Y�6       �	�\�Xc�A�*

loss�:<
x0       �	�Xc�A�*

loss���;~��       �	���Xc�A�*

loss�v�<>3�R       �	�=�Xc�A�*

loss  :87�       �	���Xc�A�*

loss6n�=�g�\       �	���Xc�A�*

lossjS�<��       �	;�Xc�A�*

loss�h�<Y�3       �	ٲ�Xc�A�*

lossJ6�<���;       �	�d�Xc�A�*

loss���;�Lʠ       �	���Xc�A�*

loss�ӻ;���       �	,��Xc�A�*

loss�h�<E�7d       �	�0�Xc�A�*

loss��4=�w5�       �	���Xc�A�*

loss��"=[Ի�       �	�]�Xc�A�*

lossnsJ=e��       �	M��Xc�A�*

loss�;�;�]��       �	���Xc�A�*

loss_k~;����       �	�0�Xc�A�*

lossy;B�p       �	�5�Xc�A�*

lossɄ<�j$       �	���Xc�A�*

loss�'�<�]��       �	G��Xc�A�*

loss��R<R��I       �	�|�Xc�A�*

loss��<:8�       �	w��Xc�A�*

loss�&\=K&       �	2< �Xc�A�*

loss!C�=5x       �	�!�Xc�A�*

loss3�<EAbH       �	)""�Xc�A�*

lossm�O;��j       �	��"�Xc�A�*

loss���<���       �	E�#�Xc�A�*

lossZ��<of�       �	"P$�Xc�A�*

loss�F<�}�       �	`�$�Xc�A�*

loss�;���       �	�
&�Xc�A�*

lossК�<�pҟ       �	��&�Xc�A�*

lossq�	=N��       �	��'�Xc�A�*

loss�O<<��       �	a(�Xc�A�*

loss�o0<%�       �	��(�Xc�A�*

loss8��<����       �	�T)�Xc�A�*

loss�m8:Q�1       �	��)�Xc�A�*

loss���:�@[G       �	��*�Xc�A�*

loss%�=^V�       �	6 +�Xc�A�*

loss�zI=Y
�E       �	��+�Xc�A�*

lossqx�<?H��       �	ۅ,�Xc�A�*

loss��:W��       �	�o-�Xc�A�*

loss��J= �S�       �	�.�Xc�A�*

loss�Q=���       �	��.�Xc�A�*

loss��g<���L       �	 �/�Xc�A�*

loss�<��(       �	I+0�Xc�A�*

loss�]�:94C�       �	��0�Xc�A�*

lossd�<<��i�       �	��1�Xc�A�*

loss<�=T�I;       �	�12�Xc�A�*

loss5d�=K��       �	�2�Xc�A�*

loss[��<�̦P       �	�z3�Xc�A�*

loss�V<�Y�       �	
4�Xc�A�*

lossd��9��       �	��4�Xc�A�*

loss��/;��Ok       �	�R5�Xc�A�*

loss�H�<�n`u       �	6�Xc�A�*

loss$�S<=��m       �	A�6�Xc�A�*

lossJv)=d�~�       �	�r8�Xc�A�*

loss��=�WD+       �	 9�Xc�A�*

loss��Q<���J       �	[�9�Xc�A�*

lossԝ:�D       �	r�:�Xc�A�*

loss�&;0z�       �	`:;�Xc�A�*

loss���;A��       �	��;�Xc�A�*

loss��;����       �	�m<�Xc�A�*

loss�.=�me�       �		=�Xc�A�*

loss�?;$�ͪ       �	��=�Xc�A�*

losst��;'J.�       �	i�>�Xc�A�*

loss�s�<��H       �	�H?�Xc�A�*

lossj]M=�\�       �	/@�Xc�A�*

loss#F�<{�>�       �	��@�Xc�A�*

lossA@�<iZ�       �	�KA�Xc�A�*

lossd|{<G;.g       �	T�A�Xc�A�*

lossz�< ���       �	WwB�Xc�A�*

loss�m;=i��:       �	jC�Xc�A�*

loss�v�<���       �	T�C�Xc�A�*

loss�=I2�       �	RCD�Xc�A�*

loss\��<}��       �	��D�Xc�A�*

loss_C�<(	�!       �	�rE�Xc�A�*

loss	�I<�I�       �	�	F�Xc�A�*

lossL��</�L[       �	r�F�Xc�A�*

loss���<��        �	y:G�Xc�A�*

loss@,�=��3M       �	�H�Xc�A�*

loss��<(��9       �	I�H�Xc�A�*

loss�tA<�q       �	�3I�Xc�A�*

loss��:�l>       �	W�I�Xc�A�*

loss�q=L��Y       �	;sJ�Xc�A�*

loss��=W�5       �	�K�Xc�A�*

losso�|:k��m       �	��K�Xc�A�*

loss/�<i�͗       �	�XM�Xc�A�*

loss��:���>       �	^�M�Xc�A�*

loss,��<՝�d       �	��N�Xc�A�*

loss_�N=�u�       �	܄O�Xc�A�*

loss�N8=�_"       �	dP�Xc�A�*

loss��==�m��       �	B�P�Xc�A�*

loss~��;tu��       �	JQ�Xc�A�*

loss�#8=���       �	��Q�Xc�A�*

lossh�	;���       �	R�Xc�A�*

loss��:W4/K       �	PS�Xc�A�*

loss�G�=&�o       �	mT�Xc�A�*

lossv�3<�\��       �	I�T�Xc�A�*

loss�;���       �	L7U�Xc�A�*

losst#�<��2s       �	`�U�Xc�A�*

lossT�D<��%�       �	bV�Xc�A�*

loss$�z;��%�       �	�V�Xc�A�*

loss�h�=v�o       �	?�W�Xc�A�*

loss;;=�VA�       �	A(X�Xc�A�*

loss�o=b��F       �	Y�X�Xc�A�*

loss���<+�       �	0LY�Xc�A�*

lossح2=@+       �	i�Y�Xc�A�*

loss�c)<�Z�       �	&sZ�Xc�A�*

loss�O
<) �Z       �	�![�Xc�A�*

loss\�'=}`�       �	,�[�Xc�A�*

loss��=�+�       �	�P\�Xc�A�*

loss��\;A,K       �	��\�Xc�A�*

loss�C<sAP       �	�]�Xc�A�*

loss���;���       �	��^�Xc�A�*

loss�`�=���       �	�k_�Xc�A�*

loss�I<#��       �	��_�Xc�A�*

loss���=�f       �	��`�Xc�A�*

lossOx=���       �	�Ia�Xc�A�*

loss���<��z       �	��a�Xc�A�*

lossE�<Qֳ�       �	��b�Xc�A�*

loss��?=s�4       �	,Dc�Xc�A�*

loss��<=*:S       �	+/d�Xc�A�*

loss�J�=��Q       �	'�d�Xc�A�*

loss�H =����       �	��e�Xc�A�*

loss&h<�x��       �	Y�f�Xc�A�*

loss��<@�	       �	'.g�Xc�A�*

loss���<�/�       �	E�g�Xc�A�*

loss�(�<���A       �	�wh�Xc�A�*

loss��);]|`�       �	��i�Xc�A�*

loss@y�<�Cs�       �	�$j�Xc�A�*

loss`ȗ<��[       �	W�j�Xc�A�*

loss�o/<a3�0       �	[|k�Xc�A�*

loss��t<�Ho|       �	Pl�Xc�A�*

loss�ڳ<1M       �	��l�Xc�A�*

loss;�= � �       �	�tm�Xc�A�*

loss\��;��S       �	�n�Xc�A�*

loss[eb<;��h       �	4�n�Xc�A�*

loss�4o:ĥ�       �	�_o�Xc�A�*

loss���<�<�[       �	�p�Xc�A�*

lossq�K=�D��       �	L�p�Xc�A�*

loss�T5=�o�       �	�Kq�Xc�A�*

loss8\�=
�Y`       �	 �q�Xc�A�*

lossq�\<wsD       �	`�r�Xc�A�*

loss�`Q=R�       �	"8s�Xc�A�*

loss:��;s��       �	s�s�Xc�A�*

loss_/;�
m�       �	��t�Xc�A�*

lossƻ�<���       �	4.u�Xc�A�*

loss$"$<��H       �	M�u�Xc�A�*

lossD�<H\`       �	2wv�Xc�A�*

losshI�=��ʺ       �	�w�Xc�A�*

loss�<�:       �	,�x�Xc�A�*

lossfE�;u��       �	��y�Xc�A�*

lossA{�<9Sh       �	�%z�Xc�A�*

loss�պ<$�p�       �	2�z�Xc�A�*

losspW�;ԕ�       �	!s{�Xc�A�*

lossm2<��        �	V|�Xc�A�*

lossO��:���       �	;�|�Xc�A�*

loss�Q=�$2       �	�@}�Xc�A�*

loss��;�@r       �	��}�Xc�A�*

loss�}�<�U�E       �	v~�Xc�A�*

loss�Q<���b       �	�Xc�A�*

loss{K�<T�]u       �	֪�Xc�A�*

loss�N�;�yU�       �	{K��Xc�A�*

lossJ;����       �	K=��Xc�A�*

loss,�:�3�       �	����Xc�A�*

loss#�[=�;?�       �	є��Xc�A�*

loss��
=��|C       �	]5��Xc�A�*

loss��=Z��>       �	.˃�Xc�A�*

loss�4>=�+p       �	�c��Xc�A�*

lossa|V<C+�       �	����Xc�A�*

loss*�<"��       �	���Xc�A�*

loss���<d�       �	 9��Xc�A�*

loss��<�:5�       �	U݆�Xc�A�*

loss�g�;��_�       �	Ot��Xc�A�*

lossW��;�	�       �	���Xc�A�*

loss �5<oSY~       �	����Xc�A�*

loss���<�)       �	�J��Xc�A�*

loss� =)�       �	����Xc�A�*

loss!Ӆ<8�N�       �	�{��Xc�A�*

loss�ɜ<+�       �	���Xc�A�*

loss�i�<�Ҧ�       �	a���Xc�A�*

lossx�<��{�       �	�b��Xc�A�*

lossŠ�;b� �       �	����Xc�A�*

loss�\><)<�U       �	Ō��Xc�A�*

loss��<�8Z       �	u!��Xc�A�*

loss� <�ލ�       �	���Xc�A�*

loss�.=�	��       �	�X��Xc�A�*

loss%<>s{M�       �	x��Xc�A�*

loss�#�;��%       �	a���Xc�A�*

loss���:zT��       �	)$��Xc�A�*

loss�#=��U>       �	%ɑ�Xc�A�*

losswf�<Jt߲       �	/n��Xc�A�*

loss��;j��       �	(��Xc�A�*

loss=�<6r�       �	/���Xc�A�*

loss�)<�O>�       �	�>��Xc�A�*

lossA1�<N	�\       �	�֔�Xc�A�*

loss���< K�       �	�s��Xc�A�*

loss|�[<¬�x       �	*��Xc�A�*

loss�I="K�       �	ݴ��Xc�A�*

loss��0<`%��       �	�S��Xc�A�*

loss���<,�\       �	��Xc�A�*

loss�M<<΃       �	E���Xc�A�*

loss���;n!��       �	Hޙ�Xc�A�*

loss��<��       �	�y��Xc�A�*

loss�Z;<�SX       �	)$��Xc�A�*

loss�6R:}?<Y       �	1Λ�Xc�A�*

loss�(<�w�	       �	�t��Xc�A�*

loss���<w��       �	�%��Xc�A�*

loss�Hr=�%S?       �	Pm��Xc�A�*

loss�Oj=����       �	����Xc�A�*

loss���<<�nF       �	����Xc�A�*

loss�=AE�       �	Uj��Xc�A�*

loss���<�ϰ�       �	�	��Xc�A�*

losso=y-��       �	�Ԣ�Xc�A�*

loss,~�<,ȣ_       �	�x��Xc�A�*

loss��<$�       �	*X��Xc�A�*

loss%p�=*� O       �	���Xc�A�*

loss Lm<�fv4       �	 ���Xc�A�*

loss?��<��ę       �	����Xc�A�*

lossΧx=_�       �	�ݧ�Xc�A�*

lossʍ�<����       �	W���Xc�A�*

lossh�;p�H�       �	�ũ�Xc�A�*

loss�1�:z���       �	�[��Xc�A�*

loss��o=�1�B       �	b��Xc�A�*

loss�Մ<�)��       �	ǟ��Xc�A�*

lossA+�;F�       �	�w��Xc�A�*

loss��:<NN��       �	�:��Xc�A�*

lossU�;h�g�       �	���Xc�A�*

lossܦ=���1       �	�C��Xc�A�*

loss�x�<m�S       �	 ��Xc�A�*

loss���<�$��       �	����Xc�A�*

losss�<4��B       �	uZ��Xc�A�*

loss�߂=h�       �	;���Xc�A�*

lossE��;����       �	}?��Xc�A�*

loss||�:��`9       �	$��Xc�A�*

lossM[<_�       �	n���Xc�A�*

loss� �;��h�       �	�@��Xc�A�*

loss�S;�П�       �	�۶�Xc�A�*

loss�wd<��-�       �	�p��Xc�A�*

loss���<�ۥ�       �	���Xc�A�*

loss�*n<g       �	���Xc�A�*

loss�N�<V/#�       �	A��Xc�A�*

loss�8<a�}       �	Qۺ�Xc�A�*

lossWg�;:���       �	ly��Xc�A�*

loss�%�;�w�       �	���Xc�A�*

lossC�T=L�       �	����Xc�A�*

loss���=� ��       �	�V��Xc�A�*

loss���<W1.[       �	N��Xc�A�*

loss� =/ 7       �	����Xc�A�*

loss���;E��e       �	Q-��Xc�A�*

lossq\{<�G��       �	�W��Xc�A�*

loss��=��g4       �	���Xc�A�*

lossEpT;�s
       �	"���Xc�A�*

loss�)u<`��       �	�=��Xc�A�*

loss��0=<       �	����Xc�A�*

lossl�c=�<2       �	����Xc�A�*

loss��:��y�       �	w/��Xc�A�*

lossT�8;es^{       �	����Xc�A�*

lossR��:~E��       �	m��Xc�A�*

loss�|>W;V�       �	���Xc�A�*

loss��I=,��@       �	o���Xc�A�*

loss�(�;߃��       �	�=��Xc�A�*

lossδ=���       �	���Xc�A�*

lossa��;�>�+       �	bf��Xc�A�*

loss���<˙�       �	����Xc�A�*

lossn��<n���       �	ݵ��Xc�A�*

lossz��<�g�       �	Q��Xc�A�*

loss:�<^{�       �	����Xc�A�*

loss��=��?�       �	8���Xc�A�*

loss�94=|��]       �	:!��Xc�A�*

loss��n=���       �	4���Xc�A�*

loss��<ˁ�l       �	CY��Xc�A�*

loss͍�<y�.3       �	����Xc�A�*

loss$B�;���       �	e���Xc�A�*

loss�U<�K�}       �	� ��Xc�A�*

loss���=��b�       �	���Xc�A�*

loss]��<M>C7       �	���Xc�A�*

lossL��<e{�       �	$��Xc�A�*

lossv��<�1;�       �	����Xc�A�*

loss%$�;�e       �	�h��Xc�A�*

loss2�<Z��T       �	[��Xc�A�*

loss���<�͝�       �	����Xc�A�*

losss1=]�Cz       �	QJ��Xc�A�*

loss���<��M�       �	P���Xc�A�*

loss@Ng=��`       �	�|��Xc�A�*

loss��0=��       �	N&��Xc�A�*

loss'=l�]�       �	j���Xc�A�*

loss��/;���       �	�W��Xc�A�*

lossN�<�ňa       �	����Xc�A�*

losslK=�6�       �	}��Xc�A�*

loss��;n���       �	���Xc�A�*

lossԼ=M%       �	ϼ��Xc�A�*

loss|
=ʞ��       �	�X��Xc�A�*

loss�:8;����       �	����Xc�A�*

loss�w�<jSb       �	����Xc�A�*

lossD�2=���:       �	���Xc�A�*

loss� <`@       �	կ��Xc�A�*

loss��<;�a�p       �	$F��Xc�A�*

loss��Y;ٚ�       �	����Xc�A�*

loss�O�=듍z       �	����Xc�A�*

lossZ-�<�l��       �	�_��Xc�A� *

lossn �;�>2�       �	d��Xc�A� *

loss-��;�0�       �	�"��Xc�A� *

lossvEm<��J�       �	c���Xc�A� *

loss��<����       �	����Xc�A� *

loss�f�<��A�       �	y���Xc�A� *

lossR	�:��L       �	y���Xc�A� *

lossԶ�;
}�T       �	-%��Xc�A� *

loss6[-<��s�       �		l��Xc�A� *

loss��=��)�       �	���Xc�A� *

loss6Ȥ=�l�3       �	����Xc�A� *

loss��<;��}       �	����Xc�A� *

loss�W<Z�#�       �	v���Xc�A� *

loss9�=x�a(       �	<���Xc�A� *

loss��k<b���       �	�k��Xc�A� *

loss:o�9z���       �	��Xc�A� *

loss��"<,�*d       �	;7��Xc�A� *

loss���=�E�F       �	an��Xc�A� *

loss���=�5�       �	-A��Xc�A� *

loss�L=�tz       �	�v��Xc�A� *

loss:�<��JX       �	-���Xc�A� *

loss�:�;T�J/       �	�P��Xc�A� *

lossz��;�y��       �	����Xc�A� *

loss^�;����       �	����Xc�A� *

losso�=[z��       �	�$��Xc�A� *

loss1c�<W�~       �	�Z��Xc�A� *

loss��;M��       �	+O��Xc�A� *

loss-t�:��>�       �	1A��Xc�A� *

loss �R;ow��       �	�K��Xc�A� *

loss�8�<e��       �	$���Xc�A� *

lossD��<uI��       �	̸��Xc�A� *

loss�^�;+��       �	�U��Xc�A� *

loss�J<�:�U       �	����Xc�A� *

loss�=��       �	$C��Xc�A� *

lossc�;�w�       �	E��Xc�A� *

loss:�T;i!�;       �	���Xc�A� *

loss���<g��       �	� �Xc�A� *

loss�W�;��!�       �	z� �Xc�A� *

loss#ַ9�j�        �	�H�Xc�A� *

lossʷ�:���       �	���Xc�A� *

loss�p#<	e�r       �	��Xc�A� *

loss�1;�Ağ       �	���Xc�A� *

loss/�;�9�       �	���Xc�A� *

loss3(5;oZp|       �	p@�Xc�A� *

lossm� ;4�X�       �	g��Xc�A� *

loss���9�x?       �	b��Xc�A� *

loss�/:D#s�       �	�6�Xc�A� *

loss��9����       �	.�Xc�A� *

lossxO
<�f�       �	���Xc�A� *

loss�U�;9�x�       �	��	�Xc�A� *

loss�;R��       �	�u
�Xc�A� *

loss�Լ:��L       �	�d�Xc�A� *

lossOB`;Q-�b       �	�l�Xc�A� *

loss�+>�;V�       �	�D�Xc�A� *

loss5C�<r���       �	�9�Xc�A� *

loss��=XA�n       �	�X�Xc�A� *

lossj�<�u�}       �	���Xc�A� *

loss��v=��B�       �	
0�Xc�A� *

loss~=5�up       �	���Xc�A� *

loss�_�;{}       �	���Xc�A� *

lossW#�=��X�       �	���Xc�A� *

loss�x5<g}j�       �	ϣ�Xc�A� *

loss���=��       �	�B�Xc�A� *

lossC�6<�8       �	��Xc�A� *

lossh=A�r{       �	��Xc�A� *

loss=�m=gS�4       �	Q��Xc�A� *

loss�Q?=�5}�       �	�?�Xc�A� *

loss=�=�r��       �	�B�Xc�A� *

lossjU<A�<       �	I��Xc�A� *

loss/>x<.U��       �	q�Xc�A� *

loss�u�=H=y       �	/��Xc�A� *

lossU�<_�       �	�[�Xc�A� *

loss��Y=\�       �	{�Xc�A� *

loss�=4>��       �	���Xc�A� *

loss�و:��z�       �	���Xc�A� *

loss�<2��       �	�� �Xc�A� *

loss[i�;�S6m       �	�x!�Xc�A� *

loss�;Ց��       �	�"�Xc�A� *

loss�8�<��=�       �	W#�Xc�A� *

lossI��;/j?�       �	$�#�Xc�A� *

lossW�~;)=G7       �	j�$�Xc�A� *

loss�'O<w{�       �	Y%�Xc�A� *

loss��<4aw�       �	��%�Xc�A� *

loss���=pJ�       �	��&�Xc�A� *

loss�%�;�Y��       �	RD'�Xc�A� *

loss���<�|W�       �	^�'�Xc�A� *

lossRg�:V�%�       �	�i(�Xc�A� *

loss"��:K$��       �	$	)�Xc�A� *

loss���<E�~W       �	P�)�Xc�A� *

loss� ;����       �	Zd*�Xc�A� *

loss�i=pNc�       �	W+�Xc�A� *

loss�8�<��L       �	#�+�Xc�A� *

loss\S�<���Z       �	k�,�Xc�A� *

loss&�<<1$��       �	}A-�Xc�A� *

loss��;�'�       �	��-�Xc�A� *

lossS=1�ۢ       �	�r.�Xc�A� *

lossn�f=�_,#       �	s/�Xc�A� *

loss��=r�{       �	��/�Xc�A� *

loss���<Ez�[       �	�K0�Xc�A� *

loss��<nqt�       �	E�0�Xc�A� *

loss��= �͸       �	&�1�Xc�A� *

lossz�;ʮ�O       �	�$2�Xc�A� *

lossC[�<<�vM       �	��2�Xc�A� *

loss�;�;U5       �	�W3�Xc�A� *

lossHB�<#q�       �	N�3�Xc�A� *

loss��T<���       �	FN�Xc�A� *

loss��<K���       �	4�O�Xc�A� *

loss[��<��è       �	�XP�Xc�A� *

loss��<�ɮ�       �	��P�Xc�A� *

loss_C*=�}e6       �	JyQ�Xc�A� *

loss��{<G�J       �	VR�Xc�A� *

loss�zz=�s��       �	��R�Xc�A� *

loss)b�;�	�
       �	�5S�Xc�A� *

loss(3<���=       �	�"T�Xc�A� *

loss�k�<0V^=       �	��T�Xc�A� *

loss�͒;7w2�       �	:uU�Xc�A� *

loss 4�</���       �	�V�Xc�A� *

lossi�<�?>W       �	r�V�Xc�A� *

loss�ܡ<	A {       �	�;W�Xc�A� *

loss�3X=4MR       �	��W�Xc�A� *

loss�hB<�*j�       �	�oX�Xc�A� *

loss�q�:ʗ�       �	�Y�Xc�A� *

loss��'=W�*       �	I�Y�Xc�A� *

loss�c)<$�7       �	2;Z�Xc�A� *

loss�5�<�Bl        �	�Z�Xc�A� *

loss��k<OV�       �	�w[�Xc�A� *

loss �D=�͵�       �	�\�Xc�A� *

loss��$;��f       �	�\�Xc�A� *

loss��=P�k       �	"5]�Xc�A�!*

lossm߽;���       �	�g^�Xc�A�!*

loss_-^;�m       �	��_�Xc�A�!*

loss�h<���       �	%�`�Xc�A�!*

loss�w�;ogHi       �	*Va�Xc�A�!*

loss"�=<P�       �	x�a�Xc�A�!*

loss���<u��       �	��b�Xc�A�!*

loss�A�;��[%       �	�'c�Xc�A�!*

lossx2<��       �	��c�Xc�A�!*

loss��;=A�@g       �	��d�Xc�A�!*

loss���<���       �	<e�Xc�A�!*

loss8^�<#�W       �	��e�Xc�A�!*

loss-/�<�9�,       �	I�f�Xc�A�!*

lossE�3<�_,�       �	{/g�Xc�A�!*

loss�^�<F	       �	c�g�Xc�A�!*

loss�y=�˝M       �	 ~h�Xc�A�!*

loss���<�w��       �	�#i�Xc�A�!*

lossH��;$�&       �	�i�Xc�A�!*

loss�	�=�ڭZ       �	jj�Xc�A�!*

loss�a�;L6��       �	�k�Xc�A�!*

loss��/;'�v�       �	~�k�Xc�A�!*

loss̩�<`B�       �	Pl�Xc�A�!*

loss�9�<3jO       �	��l�Xc�A�!*

loss|�R;K�7|       �	Am�Xc�A�!*

loss�� ;aڋ�       �	Pn�Xc�A�!*

loss3P�<y�ػ       �	��n�Xc�A�!*

loss�T�9;��       �	.So�Xc�A�!*

loss���<�F�       �	x�o�Xc�A�!*

loss��n=��       �	]�p�Xc�A�!*

loss��)<���       �	� q�Xc�A�!*

lossvT*=' a       �	�q�Xc�A�!*

lossa�<�:��       �	hr�Xc�A�!*

loss���;ž�       �	��r�Xc�A�!*

loss�x:8�Q	       �	��s�Xc�A�!*

loss�,%;��       �	�"t�Xc�A�!*

loss�m�<h�>%       �	иt�Xc�A�!*

loss��Y=5`9;       �	�Ou�Xc�A�!*

loss�J�=N�|�       �	��u�Xc�A�!*

loss���;��"�       �	��v�Xc�A�!*

loss�=�\AK       �	iw�Xc�A�!*

loss��%<��ǹ       �	
�w�Xc�A�!*

loss��j<�̱       �	*Sx�Xc�A�!*

loss���;���       �	��x�Xc�A�!*

loss�DI;���       �	�y�Xc�A�!*

loss|=E=�H@�       �	�0z�Xc�A�!*

lossF�=R��i       �	7�z�Xc�A�!*

lossw�S<�ʲ       �	�h{�Xc�A�!*

loss��<mW�       �	|�Xc�A�!*

loss"�<��>�       �	G�|�Xc�A�!*

loss*�I<�<C2       �	�f}�Xc�A�!*

loss�z=�\*       �	�~�Xc�A�!*

lossm�<<v<|-       �	��~�Xc�A�!*

lossnH�;�λ       �	�6�Xc�A�!*

loss��Q<�G�f       �	;��Xc�A�!*

loss��;d��       �	x_��Xc�A�!*

loss���<p�w�       �	���Xc�A�!*

loss(��<�sb�       �	r���Xc�A�!*

loss场;?�'       �	�!��Xc�A�!*

losse�<o��<       �	F���Xc�A�!*

loss��i<�T;C       �	�I��Xc�A�!*

loss��; C�       �	�܃�Xc�A�!*

loss��	<:&       �	S���Xc�A�!*

loss:F�:�l�u       �	E��Xc�A�!*

loss�]�<�B��       �	�ׅ�Xc�A�!*

loss�*<J��       �	Di��Xc�A�!*

lossJ8<��-�       �	{���Xc�A�!*

loss�J^<�9�4       �	����Xc�A�!*

lossL<�.       �	�#��Xc�A�!*

loss���<�a�       �	^���Xc�A�!*

lossR>�;�k3�       �	�c��Xc�A�!*

lossEo�;��       �	/���Xc�A�!*

loss�W�;�EZ]       �	֋��Xc�A�!*

loss;��<~h�       �	���Xc�A�!*

loss��+=�#��       �	����Xc�A�!*

loss
�=�`��       �	�[��Xc�A�!*

lossR6�=ז+8       �	���Xc�A�!*

lossJ*�9�/�       �	9���Xc�A�!*

lossx�;=-��       �	�'��Xc�A�!*

loss�c;ж�S       �	�Ȏ�Xc�A�!*

loss�d�;���"       �	�^��Xc�A�!*

loss�;��N       �	"���Xc�A�!*

loss&��:�f�|       �	���Xc�A�!*

loss
<<�2       �	sK��Xc�A�!*

loss��9�.�       �	��Xc�A�!*

loss�n1<=��        �	g~��Xc�A�!*

loss�<�Cr%       �	Q��Xc�A�!*

loss���<��=�       �	���Xc�A�!*

loss�0�<�� �       �	�B��Xc�A�!*

loss�}�<tB�m       �	jڔ�Xc�A�!*

loss�U=�bf       �	er��Xc�A�!*

loss�yX:¼��       �	B��Xc�A�!*

loss8a3;_�       �	����Xc�A�!*

loss.<����       �	<��Xc�A�!*

loss֋�<E?�       �	�˗�Xc�A�!*

loss�9=�!��       �	J]��Xc�A�!*

lossJ�;S%�       �	����Xc�A�!*

loss��:�[K"       �	����Xc�A�!*

loss�	�;���       �	�(��Xc�A�!*

lossC��<B;       �	����Xc�A�!*

lossܳ�;��@�       �	P��Xc�A�!*

loss[�<h�	k       �	\��Xc�A�!*

lossAG*=�n�*       �	t{��Xc�A�!*

loss��=?>��       �	r��Xc�A�!*

lossX�.<����       �	O���Xc�A�!*

loss���;Vb�       �	�A��Xc�A�!*

loss�1<|�]�       �	ڞ�Xc�A�!*

loss�';4[�       �	���Xc�A�!*

loss/��;a�2�       �	\���Xc�A�!*

loss@��<5���       �	UL��Xc�A�!*

lossT�<mm��       �	���Xc�A�!*

lossC�:Q&C�       �	,���Xc�A�!*

loss�*=o�9s       �	#I��Xc�A�!*

loss$H+=�       �	d��Xc�A�!*

loss���;.Y Z       �	T���Xc�A�!*

loss%�Q<���M       �	�3��Xc�A�!*

loss�d;���       �	5��Xc�A�!*

loss��;�ݥ       �	gҦ�Xc�A�!*

loss%�:��G       �	^e��Xc�A�!*

loss�j�:�q�       �	����Xc�A�!*

loss�e<�h��       �	����Xc�A�!*

loss�#�=@D       �	���Xc�A�!*

loss�>q?       �	�i��Xc�A�!*

loss�Z ;)���       �	Ra��Xc�A�!*

lossEc�<��&       �	�3��Xc�A�!*

loss]h�;u��       �	�9��Xc�A�!*

loss���;�݋�       �	,��Xc�A�!*

loss� �=�]�       �	���Xc�A�!*

lossR�:Re�]       �	W?��Xc�A�!*

lossf�C<^&�y       �	���Xc�A�"*

loss�l
<��R�       �	侮�Xc�A�"*

loss���;أ'       �	G̱�Xc�A�"*

loss<L<�y�       �	���Xc�A�"*

lossр�:07       �	����Xc�A�"*

loss�;��       �	�>��Xc�A�"*

loss���;øG       �	���Xc�A�"*

loss.�<���       �	���Xc�A�"*

loss��;u�Ip       �	�M��Xc�A�"*

lossZ��<��(�       �	%��Xc�A�"*

loss�)v<��       �	����Xc�A�"*

loss��<ba       �	�4��Xc�A�"*

loss�nh<����       �	oӸ�Xc�A�"*

loss�(�=D5�       �	�j��Xc�A�"*

loss�/&<H��       �	x��Xc�A�"*

lossz�<�5{       �	�ú�Xc�A�"*

loss)��;Π=�       �	p]��Xc�A�"*

loss�I�<hᎺ       �	���Xc�A�"*

lossȽ�;�4~�       �	Έ��Xc�A�"*

loss�k%<���       �	�C��Xc�A�"*

lossBˡ;�}��       �	�߽�Xc�A�"*

lossa�p;��(�       �	w��Xc�A�"*

loss�a<7A�       �	�!��Xc�A�"*

lossЬ9���       �	˹��Xc�A�"*

lossqY'=��q       �	O��Xc�A�"*

lossS̎;�X       �	����Xc�A�"*

loss��!<�e��       �	$���Xc�A�"*

loss�J�<Q�k{       �	p��Xc�A�"*

lossI�?=Wј�       �	���Xc�A�"*

lossq-:��q�       �	���Xc�A�"*

loss���;OB)�       �	�c��Xc�A�"*

loss�+�:7��X       �	� ��Xc�A�"*

loss�>�:p�       �	���Xc�A�"*

loss�S:��=�       �	*��Xc�A�"*

lossN=��       �	����Xc�A�"*

loss��<V5|       �	�T��Xc�A�"*

lossIFj;y}_       �	����Xc�A�"*

lossO��<?�u       �	���Xc�A�"*

loss�+�:Io��       �	�M��Xc�A�"*

loss��<��|       �	����Xc�A�"*

loss%Ò=4xW       �	 ~��Xc�A�"*

loss�l :�u�       �	�0��Xc�A�"*

loss$�;���W       �	���Xc�A�"*

loss6�0<CZ��       �	��Xc�A�"*

lossD�;ʄZ�       �	i��Xc�A�"*

loss�c�<M>R�       �	���Xc�A�"*

lossw�<�2�       �	N~��Xc�A�"*

lossvX=�}��       �	���Xc�A�"*

loss�w�:*+D�       �	���Xc�A�"*

loss��;%Z7�       �	�U��Xc�A�"*

loss�(�<�9��       �	����Xc�A�"*

loss�J	=�k�7       �	����Xc�A�"*

loss��;���       �	���Xc�A�"*

loss��<��B�       �	����Xc�A�"*

lossG% ;���       �	���Xc�A�"*

loss�<�=WI�s       �	�+��Xc�A�"*

loss��`=>f9       �	����Xc�A�"*

loss��6<�'H4       �	�\��Xc�A�"*

loss��=�ho       �	(���Xc�A�"*

loss
;N�       �	����Xc�A�"*

loss�ֽ<�҇.       �	�-��Xc�A�"*

loss��c<j�d�       �	C���Xc�A�"*

loss�=;ٯL       �	^d��Xc�A�"*

loss<��<��,�       �	 ���Xc�A�"*

lossO0�<��       �	����Xc�A�"*

loss}�>��CQ       �	{-��Xc�A�"*

lossn�=%��       �	m��Xc�A�"*

loss9Ȕ<H�G       �	à��Xc�A�"*

loss��<&E+       �	2>��Xc�A�"*

lossZ&@;+��       �	����Xc�A�"*

loss6u�:(/��       �	Pp��Xc�A�"*

loss��\;�U�       �	`��Xc�A�"*

loss��0=s��\       �	����Xc�A�"*

loss���;�Z*�       �	�V��Xc�A�"*

loss�[T<mc�       �	 ���Xc�A�"*

loss�|K<�f�)       �	"q��Xc�A�"*

loss/�;�3�^       �	D��Xc�A�"*

loss���:�W��       �	9���Xc�A�"*

loss@L�;��       �	\V��Xc�A�"*

loss<ڼ;���       �	����Xc�A�"*

lossՆ;��g@       �	����Xc�A�"*

loss4�D<K��~       �	�G��Xc�A�"*

loss�7�<b)O�       �	���Xc�A�"*

loss��<VƤz       �	A��Xc�A�"*

loss&�<�m$�       �	} ��Xc�A�"*

loss6�x={<��       �	����Xc�A�"*

loss{7<:��v       �	^���Xc�A�"*

lossԖ�:��       �	"��Xc�A�"*

lossM_942?�       �	���Xc�A�"*

lossBf<�y%�       �	jN��Xc�A�"*

lossv~�;5�(       �	����Xc�A�"*

loss2Py<[R==       �	J~��Xc�A�"*

loss�(a=ِ�       �	� ��Xc�A�"*

loss%)�<g��D       �	����Xc�A�"*

loss�,�:{"~       �	jN��Xc�A�"*

lossX�;�a-�       �	���Xc�A�"*

loss|g=�W��       �	χ��Xc�A�"*

loss�,<GAe       �	1(��Xc�A�"*

loss�<�J0�       �	����Xc�A�"*

loss���<�Q�c       �	k��Xc�A�"*

lossJY?;��$�       �	���Xc�A�"*

loss,�=�#H       �	����Xc�A�"*

loss�!%:�b6       �	�p��Xc�A�"*

lossJ�:b�K~       �	���Xc�A�"*

lossۅx<i�P       �	à��Xc�A�"*

loss�1<����       �	8��Xc�A�"*

loss�	;:��       �	����Xc�A�"*

loss��`<t(��       �	����Xc�A�"*

lossOI=<7�       �	\ ��Xc�A�"*

loss�Op<-��P       �	e���Xc�A�"*

lossH�M<1%�       �	�d��Xc�A�"*

loss.��<k}�+       �	�
��Xc�A�"*

loss�L�;��@�       �	���Xc�A�"*

loss���;�;k       �	F��Xc�A�"*

lossC��:�<��       �	���Xc�A�"*

loss:u�<�%n       �	>x��Xc�A�"*

loss��;��N\       �	���Xc�A�"*

loss�s;=�l��       �	��Xc�A�"*

loss1�;�ZyG       �	�)�Xc�A�"*

loss��<iq       �	��Xc�A�"*

loss�f�<,�u\       �	�e�Xc�A�"*

loss^;Cl�       �	��Xc�A�"*

loss�.;���<       �	I��Xc�A�"*

loss̞<P�|       �	�5�Xc�A�"*

lossX��=Rx�       �	q��Xc�A�"*

lossw��;���       �	�i�Xc�A�"*

loss��:wg��       �	��Xc�A�"*

lossj�<��	       �	v��Xc�A�"*

lossl��<t%4       �	A}	�Xc�A�#*

loss���:�Ž       �	n2
�Xc�A�#*

loss��<�47       �	X�
�Xc�A�#*

loss�<�b��       �	�{�Xc�A�#*

loss�6�<����       �	��Xc�A�#*

loss�
�;�i��       �	K��Xc�A�#*

loss��<��       �	�H�Xc�A�#*

loss�M�=,m�Y       �	7��Xc�A�#*

loss�`	<�aU       �	���Xc�A�#*

loss;�<��n       �	#/�Xc�A�#*

lossz��:l刧       �	���Xc�A�#*

losse �9*)ہ       �	ji�Xc�A�#*

loss�xS=ALrI       �	��Xc�A�#*

losst�B;�)\�       �	x��Xc�A�#*

loss�1/;L.��       �	+L�Xc�A�#*

loss�s�;]�ׯ       �	S�Xc�A�#*

loss�'=z�D�       �	��Xc�A�#*

lossz�k<��V�       �	8/�Xc�A�#*

lossRV7<�"�       �	v��Xc�A�#*

lossD|=��$�       �	wg�Xc�A�#*

loss�=���       �	��Xc�A�#*

loss=�0<�*Ao       �	���Xc�A�#*

loss6.�;#gl       �	%t�Xc�A�#*

loss-8<�[zo       �	�Xc�A�#*

loss}��<�BH       �	v��Xc�A�#*

loss���;	Kt�       �	�E�Xc�A�#*

loss1Z=�V��       �	*��Xc�A�#*

loss,�@:�3y       �	���Xc�A�#*

lossTc�<Ad��       �	�)�Xc�A�#*

loss��%;���*       �	$��Xc�A�#*

loss�~�;H�n       �	�p�Xc�A�#*

loss�H�;��u�       �	�G�Xc�A�#*

lossD!�;8e�       �	���Xc�A�#*

loss?"=�z�       �	D��Xc�A�#*

loss`�':����       �	+N�Xc�A�#*

loss��;��F�       �	 �Xc�A�#*

loss���:}�k�       �	r� �Xc�A�#*

loss��B;">�       �	˼!�Xc�A�#*

loss��<�K5�       �	�Y"�Xc�A�#*

loss�@=�"��       �	6x#�Xc�A�#*

loss,9<���y       �	�$�Xc�A�#*

lossʄ�:�z��       �	3�$�Xc�A�#*

loss4�;:�D       �	@%�Xc�A�#*

loss��=fk       �	R�%�Xc�A�#*

lossy�;��֊       �	j&�Xc�A�#*

lossŇ$=[���       �	&'�Xc�A�#*

losssː=����       �	|�'�Xc�A�#*

loss�.�<RԾ;       �	�w(�Xc�A�#*

lossJr <�0��       �	+)�Xc�A�#*

loss}�\;�q       �	�)�Xc�A�#*

loss��<�hY�       �	%�*�Xc�A�#*

lossj�;�nA       �	-?+�Xc�A�#*

loss�]"<��       �	��+�Xc�A�#*

loss�i\<����       �	�,�Xc�A�#*

loss�+<f�	.       �	�R-�Xc�A�#*

loss��<�7#�       �	��-�Xc�A�#*

loss�	<�'!v       �	�}.�Xc�A�#*

loss �1<5�]�       �	�#/�Xc�A�#*

lossF!*<���       �	��/�Xc�A�#*

loss��G<���       �	�~0�Xc�A�#*

loss�r;��<       �	�1�Xc�A�#*

lossf�9yK��       �	m�1�Xc�A�#*

loss�H,=pg�^       �	�D2�Xc�A�#*

loss�i=x�S       �	=�2�Xc�A�#*

loss7��< �KS       �	r�3�Xc�A�#*

losso_=��x       �	S#4�Xc�A�#*

loss�d<,��       �	o�4�Xc�A�#*

loss��;�:��       �	�K5�Xc�A�#*

loss
hp;�Q.<       �	/�5�Xc�A�#*

loss�u&=v�/       �	�t6�Xc�A�#*

loss�&O<"�]K       �	�7�Xc�A�#*

loss.�l<_۟�       �	�8�Xc�A�#*

loss��;&�A       �	��8�Xc�A�#*

loss���<˗{       �	�@9�Xc�A�#*

loss4��;A���       �	�9�Xc�A�#*

loss	��;��$�       �	r:�Xc�A�#*

loss��;-D.Z       �	[;�Xc�A�#*

loss;m
<��GA       �	K�;�Xc�A�#*

lossr(�=8R4       �	�E<�Xc�A�#*

loss�e�<�G'�       �	��<�Xc�A�#*

loss���;G�?       �	�p=�Xc�A�#*

loss��U;z��       �	y>�Xc�A�#*

loss"�;'A�@       �	�>�Xc�A�#*

loss��L=IC�J       �	pD?�Xc�A�#*

loss
B�<�       �	#�?�Xc�A�#*

loss���<�I�       �	�l@�Xc�A�#*

lossS�=���       �	+A�Xc�A�#*

loss=C<�8��       �	\�A�Xc�A�#*

loss�EP=l�Ɠ       �	�OB�Xc�A�#*

loss1U�:�"|�       �	��B�Xc�A�#*

lossc<}>A[       �	��C�Xc�A�#*

loss�D<'钒       �	$dD�Xc�A�#*

lossY<�|1�       �	�E�Xc�A�#*

loss�D�<�N!       �	��E�Xc�A�#*

loss尥<����       �	lF�Xc�A�#*

loss1�<L��}       �	`G�Xc�A�#*

lossn\�;�~_�       �	8�G�Xc�A�#*

loss]}r=��`]       �	�9H�Xc�A�#*

loss8]v=7*%b       �	�H�Xc�A�#*

loss��	<�P�       �	k�I�Xc�A�#*

loss�9e;1�z�       �	}@J�Xc�A�#*

loss��;��
@       �	��J�Xc�A�#*

loss���;b�h�       �	�jK�Xc�A�#*

loss�Q�=4��*       �	�L�Xc�A�#*

loss�̅<� �       �	��L�Xc�A�#*

lossE�W;�j��       �	�gM�Xc�A�#*

loss�H�<�y       �	�N�Xc�A�#*

loss�ƪ<!_qN       �	c�N�Xc�A�#*

loss�5t<�<�@       �	�XO�Xc�A�#*

lossQ�:��       �	��O�Xc�A�#*

loss��W;�뵘       �	�P�Xc�A�#*

loss���<0�v       �	�8Q�Xc�A�#*

loss��B<�(<�       �	��Q�Xc�A�#*

loss&'#<��p       �	1~R�Xc�A�#*

loss�4�<U��6       �	�7S�Xc�A�#*

loss�}d<�ki       �	��S�Xc�A�#*

lossE�=�Z�5       �	�nT�Xc�A�#*

loss��B9� $:       �	�U�Xc�A�#*

loss}�r;��=       �	]�U�Xc�A�#*

loss�K�;���       �	h?V�Xc�A�#*

loss�
d<r�c�       �	��V�Xc�A�#*

loss�7;7��9       �	�tW�Xc�A�#*

loss`�R:H	��       �	�NX�Xc�A�#*

loss�w�;Uņ�       �	��X�Xc�A�#*

loss��<����       �	.�Y�Xc�A�#*

loss���<��H       �	x*Z�Xc�A�#*

lossh��;�s       �	��Z�Xc�A�#*

lossQ��;�/�#       �	6Z[�Xc�A�#*

lossTCY<�t��       �	0,\�Xc�A�$*

lossA��<�=ǂ       �	��\�Xc�A�$*

loss�S�<�:�       �	�]]�Xc�A�$*

loss �l<�
�X       �	P�]�Xc�A�$*

loss-��=.��       �	�^�Xc�A�$*

loss���<���       �	+1_�Xc�A�$*

loss,=l��/       �	��_�Xc�A�$*

lossq�"=I�\       �	'�`�Xc�A�$*

loss�<���       �		�a�Xc�A�$*

loss�,�;��}       �	hvb�Xc�A�$*

loss�&;��f       �	c�Xc�A�$*

loss�8=?�2�       �	y�c�Xc�A�$*

loss`H�<4i�*       �	��d�Xc�A�$*

lossa�<��       �	�`e�Xc�A�$*

loss�P�<P��}       �	�f�Xc�A�$*

loss��9��       �	Ŭf�Xc�A�$*

loss���<�a�Z       �	Ag�Xc�A�$*

lossM� =	�;       �	f�g�Xc�A�$*

loss`=�<���       �	�th�Xc�A�$*

loss��z;���       �	J
i�Xc�A�$*

lossh�n;�Ƹ/       �	Y�i�Xc�A�$*

loss�'<�Ν�       �	�<j�Xc�A�$*

loss��9���1       �	��j�Xc�A�$*

loss��"<|��       �	3�k�Xc�A�$*

loss���<��,o       �		l�Xc�A�$*

loss��;���!       �	l�l�Xc�A�$*

loss�!	<�I�p       �	�Em�Xc�A�$*

loss�"�;���e       �	��m�Xc�A�$*

loss���<��.�       �	��n�Xc�A�$*

lossr�<Ӂm       �	co�Xc�A�$*

loss�V�;Ξ�z       �	��o�Xc�A�$*

loss���;ׅX�       �	��p�Xc�A�$*

lossԎ<_��       �	A,q�Xc�A�$*

loss�f>m#|       �	��q�Xc�A�$*

lossEk�<e��u       �	.Wr�Xc�A�$*

loss
;�<NU~G       �	��r�Xc�A�$*

losss؃=y�8|       �	"�s�Xc�A�$*

lossB><f��}       �	�Mt�Xc�A�$*

loss!ՙ;�Fh       �	]�t�Xc�A�$*

loss�;�=s�m       �	$�u�Xc�A�$*

loss�� <��w<       �	�v�Xc�A�$*

lossؗB<�m�       �	ܸv�Xc�A�$*

loss�e�<�.X       �	�Ow�Xc�A�$*

loss��<f#�K       �	�.x�Xc�A�$*

loss�j�:���}       �	��x�Xc�A�$*

lossĮ�:�m��       �	�ly�Xc�A�$*

loss�a�<Wp�       �	z�Xc�A�$*

lossS�<6�C�       �	ȴz�Xc�A�$*

loss��<�z�       �	<P{�Xc�A�$*

loss5�;��_       �	�{�Xc�A�$*

loss��*<�?,�       �	A�|�Xc�A�$*

loss�5#<GWV�       �	?}�Xc�A�$*

loss΄�;S%��       �	ݴ}�Xc�A�$*

loss��<u�dd       �	�K~�Xc�A�$*

loss]�0<��XN       �	��~�Xc�A�$*

loss��{<si[�       �	�x�Xc�A�$*

loss�Є=V��N       �	#��Xc�A�$*

loss�_�<WjU       �	����Xc�A�$*

lossLh�;۴q�       �	�A��Xc�A�$*

loss�@�<�ٷ       �	�ف�Xc�A�$*

lossw��<'Me~       �	����Xc�A�$*

lossd!s;C���       �	GZ��Xc�A�$*

lossuo;����       �	���Xc�A�$*

loss_1>��Z        �	t'��Xc�A�$*

lossv�:�*��       �	ǻ��Xc�A�$*

loss �r<�om       �	fO��Xc�A�$*

loss�c�=��sV       �	��Xc�A�$*

loss-��<of-       �	����Xc�A�$*

lossө�;<%0�       �	V��Xc�A�$*

lossL8�9�ˊ-       �	���Xc�A�$*

lossn�=��Ѹ       �	���Xc�A�$*

lossb��;K�!�       �	�H��Xc�A�$*

lossd,�=�a�H       �	�o��Xc�A�$*

loss?��=���9       �	M��Xc�A�$*

loss_��<W�u       �	���Xc�A�$*

loss�>�;��G+       �	���Xc�A�$*

loss�Z,=[_��       �	���Xc�A�$*

loss���<�;b       �	����Xc�A�$*

lossr8�<��o       �	�֐�Xc�A�$*

lossA�=�g�i       �	\u��Xc�A�$*

loss��=��       �	Z��Xc�A�$*

loss"�; ��q       �	����Xc�A�$*

loss�q�:��F       �	
H��Xc�A�$*

losshQc<�U!       �	���Xc�A�$*

loss�/�<��!       �	ҋ��Xc�A�$*

loss�EP:뗤�       �	�=��Xc�A�$*

lossF�<�`g�       �	�ܕ�Xc�A�$*

loss3B;*h#       �	�v��Xc�A�$*

loss%x<��Y�       �	��Xc�A�$*

loss�<�v�       �	1���Xc�A�$*

loss�UB=����       �		R��Xc�A�$*

loss�C�<)&$7       �	���Xc�A�$*

loss(�Y<TU       �	ׄ��Xc�A�$*

loss�<���H       �	C��Xc�A�$*

loss���;-�-       �	O��Xc�A�$*

loss���<+x�       �	����Xc�A�$*

loss�:~;ިU�       �	�U��Xc�A�$*

loss��
<1id3       �	����Xc�A�$*

loss��=Z�*B       �	X���Xc�A�$*

loss)�;�'Ǉ       �	O��Xc�A�$*

loss�E;��       �	W��Xc�A�$*

loss�a�;ms�       �	Z���Xc�A�$*

loss�Ĕ;(W�e       �	G;��Xc�A�$*

loss��;t�L#       �	�%��Xc�A�$*

lossLS�<E�G       �	�ԡ�Xc�A�$*

lossc��;./�       �	����Xc�A�$*

loss�W�;�N�       �	%���Xc�A�$*

loss���<�GM�       �	���Xc�A�$*

lossd�<��       �	K\��Xc�A�$*

loss&	=z��       �	�	��Xc�A�$*

loss�fM<�h��       �	<���Xc�A�$*

loss|��;#*C       �	���Xc�A�$*

lossFr�<���       �	^d��Xc�A�$*

loss��::���       �	�g��Xc�A�$*

loss��;,�*:       �	���Xc�A�$*

loss��q:��       �	ת�Xc�A�$*

loss�+�;^���       �	zq��Xc�A�$*

lossEE�<����       �	V��Xc�A�$*

loss�)p;�^*       �	|���Xc�A�$*

loss(��:*i2       �	�l��Xc�A�$*

lossD�+<K�M@       �	�"��Xc�A�$*

loss%��<�N       �	|ѯ�Xc�A�$*

loss�;�:Ǡ       �	���Xc�A�$*

loss�1W<q#�       �	'���Xc�A�$*

lossf�R=*��       �	�7��Xc�A�$*

loss�n;c*,       �	tӲ�Xc�A�$*

lossE��9ӟm�       �	�k��Xc�A�$*

loss*B�;L�G�       �	���Xc�A�$*

loss4��:��M~       �	}���Xc�A�%*

lossa;��j       �	J��Xc�A�%*

lossm)�;�M�:       �	���Xc�A�%*

loss��:|N��       �	�W��Xc�A�%*

lossA�(=ٓ       �	����Xc�A�%*

loss��W:Y~�       �	����Xc�A�%*

loss,n�8��P       �	gD��Xc�A�%*

loss#WB:�9       �	wٹ�Xc�A�%*

loss�U;A��       �	k���Xc�A�%*

loss=�/;֐\       �	0��Xc�A�%*

lossW*�;
�ξ       �	oG��Xc�A�%*

loss%P8�I��       �	���Xc�A�%*

loss+�;��       �	5���Xc�A�%*

loss���=�'��       �	�e��Xc�A�%*

loss#;on�B       �	���Xc�A�%*

loss
=�Hs       �	���Xc�A�%*

loss@��<�K��       �	�Y��Xc�A�%*

loss�E=ݙl       �	����Xc�A�%*

losslJ=;�x        �	���Xc�A�%*

lossr�<�Y��       �	5B��Xc�A�%*

loss���<��6       �	L���Xc�A�%*

loss)�;%n\�       �	�w��Xc�A�%*

loss��=����       �	���Xc�A�%*

loss/��;�>�       �	���Xc�A�%*

loss4/�;G�Z�       �	)��Xc�A�%*

loss��<xG��       �	����Xc�A�%*

loss�Џ<�p��       �	oe��Xc�A�%*

lossv�=+s�       �	���Xc�A�%*

loss���<{(��       �	C���Xc�A�%*

loss��H=p��[       �	���Xc�A�%*

loss/=�U�J       �	�4��Xc�A�%*

loss#�3=�e�       �	!���Xc�A�%*

loss|>o=��$k       �	Hn��Xc�A�%*

loss��;��-       �	���Xc�A�%*

lossΛ�;��m       �	i���Xc�A�%*

loss�'=Pbӹ       �	R��Xc�A�%*

lossw��<�|ü       �	���Xc�A�%*

loss|L�;����       �	�N��Xc�A�%*

loss��P;	��       �	>���Xc�A�%*

lossS�79��1�       �	����Xc�A�%*

loss$&;<;^vc       �	�S��Xc�A�%*

loss@
I<��%       �	���Xc�A�%*

lossG̏=�[�*       �	����Xc�A�%*

lossjc�=G~#�       �	xF��Xc�A�%*

loss�+�<�c       �	����Xc�A�%*

loss��;-"Қ       �	v��Xc�A�%*

loss![�:ib4�       �	f��Xc�A�%*

lossd:�D�P       �	ګ��Xc�A�%*

loss�PW;�2X       �	&Q��Xc�A�%*

loss,��:�q�       �	* ��Xc�A�%*

loss��O< x��       �	J���Xc�A�%*

loss���<g�D�       �	�A��Xc�A�%*

loss��<��Q       �	����Xc�A�%*

loss��L=_R       �	�s��Xc�A�%*

lossҨG;���A       �	��Xc�A�%*

loss7�Q:�hd�       �	u���Xc�A�%*

losse�<��>       �	�}��Xc�A�%*

loss`9�<�       �	 ��Xc�A�%*

lossT�;�0E>       �	c���Xc�A�%*

loss3�;�N��       �	�a��Xc�A�%*

loss3f5<g%$�       �	����Xc�A�%*

loss�^�;���v       �	����Xc�A�%*

lossi*�<�_��       �	{g��Xc�A�%*

loss�H�:���       �	��Xc�A�%*

lossX?�:�[_�       �	̚��Xc�A�%*

loss`�1=*�D�       �		���Xc�A�%*

loss�q�=n��       �	�\ �Xc�A�%*

losslW�==c��       �	Q� �Xc�A�%*

loss<S;yd)�       �	��Xc�A�%*

loss�<<���       �	Z-�Xc�A�%*

loss�.�<�U�R       �	��Xc�A�%*

lossW�/;�@�       �	Z�Xc�A�%*

loss2�;4G��       �	��Xc�A�%*

loss�@�=�D�;       �	*��Xc�A�%*

loss.E=�L       �	F�Xc�A�%*

losst},9(9�       �		��Xc�A�%*

loss�r$<d��z       �	���Xc�A�%*

loss�G)<\6`�       �	-@�Xc�A�%*

lossd	�<�յ�       �	��Xc�A�%*

loss���;g��r       �	du�Xc�A�%*

loss�\;�"�       �	O<	�Xc�A�%*

lossx` ;�#�       �	��	�Xc�A�%*

loss�#;�R��       �	ak
�Xc�A�%*

loss,�y=Q|
       �	 ��Xc�A�%*

lossߣ�<��       �	e�Xc�A�%*

lossV�<D��       �	���Xc�A�%*

loss�'�=FKP       �	��Xc�A�%*

loss���;L�'       �	M2�Xc�A�%*

lossW�<�'u,       �	��Xc�A�%*

loss66<�
��       �	io�Xc�A�%*

lossa?�;Ҭs@       �	��Xc�A�%*

loss[�<wn/       �	c��Xc�A�%*

loss�N�;Ș$       �	s0�Xc�A�%*

loss���<D       �	���Xc�A�%*

loss)�;gHо       �	�c�Xc�A�%*

lossA��<�E!�       �	���Xc�A�%*

loss�Ib;\��\       �	���Xc�A�%*

loss��k;X@#�       �	t&�Xc�A�%*

lossᘽ<R��       �	w��Xc�A�%*

lossVS<=��~       �	�_�Xc�A�%*

loss �=�ə�       �	��Xc�A�%*

loss���:�o%�       �	��Xc�A�%*

loss3�=Wя       �	E/�Xc�A�%*

loss?&�=����       �	���Xc�A�%*

loss�=L�       �	Ԝ�Xc�A�%*

loss�R�;$T�       �	�2�Xc�A�%*

loss�d&<�lw�       �	���Xc�A�%*

loss��:l���       �	Yn�Xc�A�%*

loss���<���       �	p�Xc�A�%*

loss�HM<��       �	���Xc�A�%*

loss!�[=��       �	rm�Xc�A�%*

loss�<�<����       �	O�Xc�A�%*

loss� @<`��       �	��Xc�A�%*

loss<�= B�       �	D�Xc�A�%*

loss�K#:���       �	���Xc�A�%*

lossS3`:̶�/       �	0��Xc�A�%*

lossZԨ<���       �	� �Xc�A�%*

loss��;8�U       �	/� �Xc�A�%*

loss��<`�}�       �	�o!�Xc�A�%*

loss�^=~Gt=       �	�""�Xc�A�%*

loss�@i;��-       �	)�"�Xc�A�%*

loss6�<@A�U       �	'j#�Xc�A�%*

loss�֪:�{�#       �	J$�Xc�A�%*

loss87};�#h       �	*�$�Xc�A�%*

loss��<i[��       �	�O%�Xc�A�%*

loss�F>���A       �	��%�Xc�A�%*

loss<M%<��       �	K '�Xc�A�%*

loss��;;�ݦ       �	ܜ'�Xc�A�%*

lossT�<Q�P       �	�6(�Xc�A�&*

loss�@<�c       �	")�Xc�A�&*

loss�4�;.       �	��)�Xc�A�&*

loss{\Q;n0�M       �	AH*�Xc�A�&*

loss�<����       �	�*�Xc�A�&*

loss�+<        �	>x+�Xc�A�&*

lossY\<�)�g       �	^,�Xc�A�&*

loss��5<=x�P       �	��,�Xc�A�&*

loss ۨ=�0��       �	gD-�Xc�A�&*

lossI�=��i$       �	��-�Xc�A�&*

loss� ;���       �	@�.�Xc�A�&*

loss��<�^Ԉ       �	0*/�Xc�A�&*

lossC��;���       �	e�/�Xc�A�&*

loss�n|;��&       �	�d0�Xc�A�&*

loss�w:<edv       �	�1�Xc�A�&*

loss�	<O�]�       �	��1�Xc�A�&*

loss�� =�7{�       �	�K2�Xc�A�&*

loss�9�:/Ҿ�       �	}�2�Xc�A�&*

lossη%=O���       �	ۇ3�Xc�A�&*

lossJI*=�jZ[       �	1%4�Xc�A�&*

loss�-h<�͆�       �	]�4�Xc�A�&*

loss�s$<���       �	�]5�Xc�A�&*

loss���<�Ej�       �	��5�Xc�A�&*

lossV�=���       �	��6�Xc�A�&*

lossf��<�2&�       �	�77�Xc�A�&*

loss���;_X��       �	�7�Xc�A�&*

loss��;2��       �	��8�Xc�A�&*

loss��6<�#�       �	�+9�Xc�A�&*

loss˓<�;��       �	��9�Xc�A�&*

loss�)�:���d       �	�i:�Xc�A�&*

loss�:�;�W�s       �	�;�Xc�A�&*

loss*=h ��       �	��;�Xc�A�&*

losst��:",5�       �	�j<�Xc�A�&*

loss%4+=���       �	=�Xc�A�&*

loss�]�<k�mq       �	��=�Xc�A�&*

loss��E=g�#�       �	
�>�Xc�A�&*

loss���<�'�       �	J?�Xc�A�&*

loss_]�<�^G       �	��?�Xc�A�&*

loss&t];�á+       �	i�@�Xc�A�&*

lossl�<�(       �	V)A�Xc�A�&*

lossc8�;-}�       �	�A�Xc�A�&*

loss͛�<���       �	�oB�Xc�A�&*

loss�m�;��       �	�C�Xc�A�&*

loss��<ip)T       �	L�C�Xc�A�&*

loss�'�;�#X�       �	5ED�Xc�A�&*

losss��<?wT�       �	��D�Xc�A�&*

loss�|7<V��T       �	�|E�Xc�A�&*

loss��}=��Y       �	+LF�Xc�A�&*

lossA;�S�       �	��F�Xc�A�&*

loss��$=��2       �	4�G�Xc�A�&*

lossa��:�@�N       �	!H�Xc�A�&*

loss���<B�
�       �	��H�Xc�A�&*

loss�c;Qs[D       �	�dI�Xc�A�&*

loss�%�<�i�       �	�J�Xc�A�&*

loss�,=[ud�       �	��J�Xc�A�&*

loss@�]<���       �	9K�Xc�A�&*

loss_��;�]Q       �	=�K�Xc�A�&*

loss��;ا$       �	!�L�Xc�A�&*

loss�<#��       �	A+M�Xc�A�&*

loss��=���       �	�M�Xc�A�&*

loss�_U<3��       �	�oN�Xc�A�&*

loss���;�       �	�O�Xc�A�&*

loss�r�<��O       �	`�O�Xc�A�&*

lossS�<��       �	3NP�Xc�A�&*

loss�8;`�       �	1�P�Xc�A�&*

loss!��<q�6       �	ʋQ�Xc�A�&*

loss�	�;XL"�       �	�*R�Xc�A�&*

loss��+:s<��       �	��R�Xc�A�&*

losss;��k�       �	�\S�Xc�A�&*

loss�PN<�u�       �	��S�Xc�A�&*

loss��;=9�(       �	��T�Xc�A�&*

loss�v;�J��       �	�3U�Xc�A�&*

loss��:{���       �	��U�Xc�A�&*

loss�;7;t�       �	+lV�Xc�A�&*

loss\�
<TV;�       �	�
W�Xc�A�&*

loss�<�I1�       �	�W�Xc�A�&*

loss%`;�Q!�       �	�AX�Xc�A�&*

loss	0�:���       �	Q�X�Xc�A�&*

loss,��<6��       �	lvY�Xc�A�&*

loss}m]<b/       �	#Z�Xc�A�&*

loss=��:�.i$       �	ڬZ�Xc�A�&*

loss��2=����       �	�F[�Xc�A�&*

loss�]=�       �	��[�Xc�A�&*

loss�_�:&b�k       �	�u\�Xc�A�&*

loss�r<�$'       �	J]�Xc�A�&*

loss�\<X���       �	l�]�Xc�A�&*

loss�� ;P�+�       �	�M^�Xc�A�&*

loss�A�<���P       �	��^�Xc�A�&*

loss=�<Bj��       �	�_�Xc�A�&*

loss��<�q�       �	�aa�Xc�A�&*

loss���<���       �	�7b�Xc�A�&*

loss��#=z�y       �	J�b�Xc�A�&*

loss��;:�)�|       �	��c�Xc�A�&*

loss��:���       �	�Hd�Xc�A�&*

loss-R=�Ί�       �	�d�Xc�A�&*

loss�2�<����       �	D�e�Xc�A�&*

lossmD<e�        �	�&f�Xc�A�&*

lossb=	3S�       �	/�f�Xc�A�&*

lossx.�;��?�       �	$cg�Xc�A�&*

loss��=Þ�       �	��g�Xc�A�&*

lossut�;d���       �	Ԝh�Xc�A�&*

loss��L<�ZX1       �	35i�Xc�A�&*

loss$�<Q��5       �	V�i�Xc�A�&*

loss�g:tgz       �	H�j�Xc�A�&*

lossjN�:����       �	�>k�Xc�A�&*

loss�$b<���       �	��k�Xc�A�&*

loss���:�A�       �	��l�Xc�A�&*

lossb�=$�       �	X<m�Xc�A�&*

loss��;�       �	k)n�Xc�A�&*

lossJ��<�:1�       �	X�n�Xc�A�&*

loss���<<�s�       �	#ko�Xc�A�&*

lossh�9�}�o       �	�p�Xc�A�&*

lossn7<       �	 �p�Xc�A�&*

loss�5<���J       �	4Lq�Xc�A�&*

loss�M}<,�|       �	��q�Xc�A�&*

loss߳;��K       �	��r�Xc�A�&*

lossڛ?<�/�       �	z6s�Xc�A�&*

loss��;g���       �	��s�Xc�A�&*

loss���<���       �	�rt�Xc�A�&*

loss=�6<j�.       �	�u�Xc�A�&*

loss�R3<A~Ix       �	5�u�Xc�A�&*

loss�<�<��O�       �	�Pv�Xc�A�&*

loss�� :�C��       �	1�v�Xc�A�&*

loss�>�;���1       �	P�w�Xc�A�&*

loss[�;�!Ä       �	�"x�Xc�A�&*

loss�X
=��       �	��x�Xc�A�&*

lossv	=��E�       �	�`y�Xc�A�&*

loss%Z�<��9�       �	��y�Xc�A�&*

lossV`:�p�       �	6�z�Xc�A�'*

loss��:u��       �	�x{�Xc�A�'*

loss�S�;i�       �	{|�Xc�A�'*

lossn�:<EW       �	��|�Xc�A�'*

loss3�=��       �	�O}�Xc�A�'*

loss(�G;Jy3(       �	9�}�Xc�A�'*

loss�w=�;=�       �	��~�Xc�A�'*

loss\G=�}��       �	ס�Xc�A�'*

loss4z:G��       �	�;��Xc�A�'*

lossRg&;�5�       �	ڀ�Xc�A�'*

loss���:���       �	����Xc�A�'*

loss<�@=�x
        �	"��Xc�A�'*

loss;�p:{t��       �	�ق�Xc�A�'*

lossN%R=���       �	�{��Xc�A�'*

loss�<�^��       �	X��Xc�A�'*

loss	��<i�D�       �	��Xc�A�'*

lossn4�<��O�       �	�e��Xc�A�'*

loss��R;D�L�       �	��Xc�A�'*

loss�k�<���6       �	֫��Xc�A�'*

loss}��<����       �	�F��Xc�A�'*

lossJ��;�C��       �	���Xc�A�'*

lossmޟ<�'�&       �	ӆ��Xc�A�'*

loss)� =$	�       �	l"��Xc�A�'*

loss�D=�td�       �	����Xc�A�'*

loss�5�<TV��       �	���Xc�A�'*

loss$˨=0W�       �	�%��Xc�A�'*

loss�{=7]~W       �	Yċ�Xc�A�'*

loss}DO=��|       �	a��Xc�A�'*

loss���;`�p       �	D��Xc�A�'*

lossxeB=���       �	����Xc�A�'*

loss��<"�@=       �	�C��Xc�A�'*

loss��;��       �	�ߎ�Xc�A�'*

loss�m9<3�}       �	�{��Xc�A�'*

loss��;�92O       �	���Xc�A�'*

loss!��<��       �	)���Xc�A�'*

loss,�h<�ˏ�       �	�[��Xc�A�'*

loss���<�9       �	e���Xc�A�'*

loss��=n\�       �	Œ��Xc�A�'*

lossn�:��       �	e4��Xc�A�'*

loss �6<��u�       �	�͓�Xc�A�'*

loss߉=Q3��       �	,f��Xc�A�'*

loss��<0fm�       �	����Xc�A�'*

loss�*;=�a       �	8���Xc�A�'*

lossNBX<iI�A       �	�:��Xc�A�'*

loss��=����       �	���Xc�A�'*

loss��<%Wp�       �	�ԗ�Xc�A�'*

loss�&<"�       �	Bv��Xc�A�'*

loss�=AP��       �	_��Xc�A�'*

lossr��:n�2L       �	Χ��Xc�A�'*

loss8m�;*�U*       �	5A��Xc�A�'*

loss���;��'C       �	wښ�Xc�A�'*

loss[��<:���       �	�v��Xc�A�'*

loss�|�=X�d       �	�"��Xc�A�'*

loss�/=�z�       �	�Ŝ�Xc�A�'*

loss
��:��8{       �	�m��Xc�A�'*

loss5��<@ĵ�       �	���Xc�A�'*

lossW�x;�pY       �	ᵞ�Xc�A�'*

loss�}7< ].�       �	iQ��Xc�A�'*

loss���;yXF       �	�M��Xc�A�'*

loss�='�T�       �	O��Xc�A�'*

loss-W=���       �	Dġ�Xc�A�'*

loss��a<Y;�t       �	�u��Xc�A�'*

loss�#;6���       �	��Xc�A�'*

loss�-7:�I��       �	�N��Xc�A�'*

loss��<��       �	���Xc�A�'*

lossay�<��       �	a���Xc�A�'*

loss�R<Y�c�       �	=,��Xc�A�'*

loss[��;ֺį       �	�զ�Xc�A�'*

lossXG�<Q��       �	�n��Xc�A�'*

loss��;�W�7       �	���Xc�A�'*

lossn�	=�\z}       �	���Xc�A�'*

loss1��<@|��       �	Cr��Xc�A�'*

lossJu|:����       �	(��Xc�A�'*

loss�G�<+��B       �	����Xc�A�'*

loss<I�c�       �	�_��Xc�A�'*

loss!�<�
       �	+���Xc�A�'*

loss}؛<~;��       �	p���Xc�A�'*

loss/��=�Dg�       �	V��Xc�A�'*

loss?�<���       �	��Xc�A�'*

loss�ʦ:��z       �	e���Xc�A�'*

loss���<&2*       �	t)��Xc�A�'*

loss��<�ҖJ       �	�ů�Xc�A�'*

loss߃�;�[��       �	�d��Xc�A�'*

loss��;���       �	>$��Xc�A�'*

loss��<�<�       �	����Xc�A�'*

loss��o<},�K       �	Nd��Xc�A�'*

loss�l�;6F&'       �	B
��Xc�A�'*

loss�=TB7       �	£��Xc�A�'*

loss�J�<7�.       �	�=��Xc�A�'*

loss��;\�E       �	^ٴ�Xc�A�'*

lossq��;k&>T       �	�t��Xc�A�'*

lossz�};�7��       �	�L��Xc�A�'*

lossO�<kĴ       �	!��Xc�A�'*

loss��=܂U4       �	���Xc�A�'*

loss̃�<�+cl       �	#��Xc�A�'*

loss���=�7�~       �	$���Xc�A�'*

lossƲ�=��}�       �	�]��Xc�A�'*

loss
��;�*	M       �	����Xc�A�'*

loss��:&y��       �	���Xc�A�'*

loss�7�<�z��       �	����Xc�A�'*

loss�	=+�g       �	�:��Xc�A�'*

loss���:鋘�       �	��Xc�A�'*

loss�M�;�L�@       �	���Xc�A�'*

lossl\=;��       �	ޭ��Xc�A�'*

loss?�=�i�x       �	kI��Xc�A�'*

loss�{�<(�       �	!��Xc�A�'*

loss2�S;��<>       �	����Xc�A�'*

lossH_;R��       �	('��Xc�A�'*

loss�h=�
4       �	����Xc�A�'*

loss�\�;�\D�       �	�^��Xc�A�'*

loss<��<*2}       �	���Xc�A�'*

loss���;)֛�       �	����Xc�A�'*

loss���<[�q�       �	V��Xc�A�'*

loss��<�@��       �	����Xc�A�'*

loss�Î=7��       �	����Xc�A�'*

loss�t�;���       �	&��Xc�A�'*

lossM�b=��*       �	���Xc�A�'*

loss�|;��Ʃ       �	9^��Xc�A�'*

loss�#�<�I&�       �	����Xc�A�'*

lossc�L=K�K�       �	����Xc�A�'*

lossf#!<qxs�       �	�9��Xc�A�'*

loss��=���L       �	���Xc�A�'*

loss"�<S&       �	6v��Xc�A�'*

loss�<1�|L       �	���Xc�A�'*

lossԧ�<MBm       �	6���Xc�A�'*

loss�'�=�D�       �	sK��Xc�A�'*

loss*<!��,       �	����Xc�A�'*

loss؉`<��       �	t���Xc�A�'*

loss�3�<x��       �	.��Xc�A�(*

loss�|(<� G�       �	����Xc�A�(*

loss��==Q�b       �	�`��Xc�A�(*

loss�6A<��\�       �	����Xc�A�(*

loss�%�;���       �	����Xc�A�(*

loss�ek=|AV�       �	�4��Xc�A�(*

loss��<��%       �	����Xc�A�(*

loss���;پv       �	`s��Xc�A�(*

loss�i�<�F�       �	�L��Xc�A�(*

loss|��<���A       �	���Xc�A�(*

loss6��=�P&u       �	���Xc�A�(*

lossl�;"�T�       �	 ��Xc�A�(*

lossz�;Lo�{       �	���Xc�A�(*

lossfS�;o[�]       �	�X��Xc�A�(*

loss��,;���       �	����Xc�A�(*

loss�&=���       �	ҋ��Xc�A�(*

loss2H�<
�3�       �	�9��Xc�A�(*

loss��<�)�M       �	���Xc�A�(*

lossDҖ;�)ox       �	����Xc�A�(*

loss )m<� W       �	�"��Xc�A�(*

loss�­;�_K       �	 	��Xc�A�(*

loss��:��       �	b���Xc�A�(*

loss�&s<!o�       �	����Xc�A�(*

loss
�g<�Y{�       �	$��Xc�A�(*

loss]2T<���       �	����Xc�A�(*

loss���=��N       �	�Y��Xc�A�(*

loss��=����       �	����Xc�A�(*

loss�(�;�dn�       �	���Xc�A�(*

loss�<�<V��*       �	(��Xc�A�(*

loss�`f<-�       �	���Xc�A�(*

loss=!�:3��       �	V��Xc�A�(*

loss��5;��\�       �	����Xc�A�(*

lossn�:Ė�1       �	����Xc�A�(*

loss
b�<�Jr       �	�)��Xc�A�(*

lossmTW<�MJL       �	����Xc�A�(*

loss�y�<�1U       �	���Xc�A�(*

loss�8q<�*�+       �	�#��Xc�A�(*

loss�C<ڱk       �	r1��Xc�A�(*

lossj�;<���{       �	p���Xc�A�(*

loss��<;���       �	�k��Xc�A�(*

loss��="�       �	���Xc�A�(*

loss���;���       �	���Xc�A�(*

loss���=�ZRr       �	�A��Xc�A�(*

loss�{;t�(       �	����Xc�A�(*

loss�]!=��       �	`r��Xc�A�(*

losseIn<��3�       �	1��Xc�A�(*

loss���;K�L�       �	<���Xc�A�(*

losss%�<`¡�       �	sG��Xc�A�(*

loss��<�Y�I       �	����Xc�A�(*

lossUo<l��       �	�}��Xc�A�(*

loss&q�<�L_�       �	!��Xc�A�(*

loss6/�;��(�       �	,���Xc�A�(*

loss�7;����       �	�v��Xc�A�(*

loss8!�:���       �	'��Xc�A�(*

loss�=��       �	���Xc�A�(*

loss�n<�Å�       �	5a��Xc�A�(*

loss!(=���G       �	����Xc�A�(*

loss<`�<��       �	̙��Xc�A�(*

lossHk;j�g       �	35��Xc�A�(*

lossqԭ;r\R�       �	����Xc�A�(*

lossx0=1��       �	�j��Xc�A�(*

loss8�G<���       �	���Xc�A�(*

loss��	=q�]+       �	8���Xc�A�(*

lossfi�=��X       �	m;��Xc�A�(*

loss\�e<<V�I       �	b���Xc�A�(*

loss8P�;f�d�       �	us��Xc�A�(*

loss�-=���       �	���Xc�A�(*

loss��<��f       �	a���Xc�A�(*

loss�T�<�r-M       �	�?��Xc�A�(*

loss�U�9Z1/#       �	��Xc�A�(*

loss�:ᵶ�       �	:���Xc�A�(*

loss�Y<����       �	Ef��Xc�A�(*

loss×�;�V       �	����Xc�A�(*

loss
!t;�,�       �	TV��Xc�A�(*

loss��/<�Ƈ�       �	I���Xc�A�(*

loss|�,<�IS       �	���Xc�A�(*

loss���<Y��       �	H2��Xc�A�(*

loss�U_:�*       �	E���Xc�A�(*

lossL=E�ڙ       �	� �Xc�A�(*

loss鱷; �?�       �	H�Xc�A�(*

loss�sz;X�n�       �	���Xc�A�(*

loss![�<� pJ       �	%]�Xc�A�(*

loss[� =�3�       �	� �Xc�A�(*

lossve�;�Z)m       �	�;�Xc�A�(*

loss�Y<��]�       �	���Xc�A�(*

loss].�=���       �	�U�Xc�A�(*

loss]|�<}�C~       �	��Xc�A�(*

loss��}<���       �	���Xc�A�(*

loss�<��?!       �	[x�Xc�A�(*

loss劬;� ��       �	a	�Xc�A�(*

lossù=ꃇN       �	ͯ	�Xc�A�(*

loss��<��       �	�L
�Xc�A�(*

loss&i@=m��       �	m �Xc�A�(*

loss�� =M�       �	ԛ�Xc�A�(*

lossE\�<�TY�       �		2�Xc�A�(*

loss[�W=����       �	��Xc�A�(*

loss��#<�e;       �	�q�Xc�A�(*

lossq�'=�`�       �	x
�Xc�A�(*

loss��<�M�       �	ۤ�Xc�A�(*

loss�<�*�]       �	h>�Xc�A�(*

losss�<��u�       �	���Xc�A�(*

loss4��;J�	:       �	���Xc�A�(*

loss�=BW�       �	�;�Xc�A�(*

loss)�n<�\       �	O[�Xc�A�(*

lossN�=XS�       �	J��Xc�A�(*

loss�� <Pkj�       �	8��Xc�A�(*

lossE�n=���       �	Y�Xc�A�(*

loss�8�;S#/       �	!��Xc�A�(*

loss<@]U�       �	�E�Xc�A�(*

loss	��;�.B       �	���Xc�A�(*

loss\ �:��<       �	���Xc�A�(*

loss;�3<9�Y*       �	)&�Xc�A�(*

loss�X�;q@��       �	Ӽ�Xc�A�(*

loss��k;De r       �	�U�Xc�A�(*

lossI�;#)��       �	���Xc�A�(*

lossa�=#"X�       �	Z��Xc�A�(*

loss�f	;o�
�       �	��Xc�A�(*

lossi�c<փ)"       �	p��Xc�A�(*

loss��;B��       �	eT�Xc�A�(*

loss�Ʋ;�0k�       �	^��Xc�A�(*

loss �;ʼ��       �	���Xc�A�(*

loss�F<2/w+       �	�$�Xc�A�(*

loss$�=�Bj�       �	8��Xc�A�(*

loss��;���       �	�Q�Xc�A�(*

loss��=��!U       �	���Xc�A�(*

loss�J<P}�       �	M��Xc�A�(*

loss�;a��       �	@ �Xc�A�(*

loss��=<�j       �	� �Xc�A�(*

loss��<�tB\       �	{K!�Xc�A�)*

loss*� ;$       �	�Y"�Xc�A�)*

loss|h�<���M       �	2#�Xc�A�)*

lossn�9<G6e       �	��#�Xc�A�)*

loss��^;��i�       �	p$�Xc�A�)*

loss?�:z��l       �	4/%�Xc�A�)*

lossl:'=�*c;       �	��%�Xc�A�)*

loss���=��       �	}&�Xc�A�)*

lossג�<�y�{       �	$)'�Xc�A�)*

loss�z�<2g��       �	��'�Xc�A�)*

loss��K=kʢ�       �	��(�Xc�A�)*

lossYv<����       �	�*)�Xc�A�)*

loss�� =+ػ^       �	��)�Xc�A�)*

loss��:�r�       �	]*�Xc�A�)*

lossz� =�k4�       �	j�*�Xc�A�)*

lossmq;�X�F       �	m�+�Xc�A�)*

lossJ%�=<<�       �	&,�Xc�A�)*

loss��r<�i�/       �	˾,�Xc�A�)*

loss��#;M3;       �	�T-�Xc�A�)*

loss�C;�%R       �	W�-�Xc�A�)*

lossUB<�ݫ�       �	!�.�Xc�A�)*

loss��:t�t�       �	�&/�Xc�A�)*

loss�Xb<��3�       �	f�/�Xc�A�)*

loss�w�=���       �	LS0�Xc�A�)*

loss
�<\Y%�       �	u�0�Xc�A�)*

loss��<��A       �	��1�Xc�A�)*

loss�7�;Z��?       �	% 2�Xc�A�)*

loss�;{�       �	E�2�Xc�A�)*

loss�:A�>3       �	�M3�Xc�A�)*

loss�2;FK|       �	��3�Xc�A�)*

loss���=�@L       �	c|4�Xc�A�)*

losst~A;��,       �	{5�Xc�A�)*

loss��2=�J=       �	�5�Xc�A�)*

loss�<mhg       �	�E6�Xc�A�)*

loss�!=X�b       �	��6�Xc�A�)*

loss ��;�{�5       �	;q7�Xc�A�)*

loss��:�*	�       �	n8�Xc�A�)*

loss2�Z= �u       �	��8�Xc�A�)*

lossIw;b3�       �	�B9�Xc�A�)*

loss3�;6W       �	��9�Xc�A�)*

loss�8P=���D       �	�m:�Xc�A�)*

loss	;9��7       �	S;�Xc�A�)*

loss,�G<@�       �	��;�Xc�A�)*

loss��<��)�       �	�6<�Xc�A�)*

loss1��;^��Z       �	��<�Xc�A�)*

loss��;ԅ�|       �	�=�Xc�A�)*

loss#A�<�x�X       �	��>�Xc�A�)*

loss�F�:�p~       �	C7?�Xc�A�)*

lossh.\;�y�       �	I�?�Xc�A�)*

loss�S<o�m�       �	��@�Xc�A�)*

loss�/�;ɐ�       �	CA�Xc�A�)*

loss�L�<��3�       �	��A�Xc�A�)*

loss\Q�<�$.       �	�tB�Xc�A�)*

lossLMR;�       �	�C�Xc�A�)*

loss�@
;:\9r       �	��C�Xc�A�)*

loss��;[0�       �	�FD�Xc�A�)*

loss�H�;��:       �	��D�Xc�A�)*

loss�t�<�L��       �	%vE�Xc�A�)*

loss�;;<MrE       �	�F�Xc�A�)*

loss2�=���<       �	�F�Xc�A�)*

loss�5=l�h,       �	��G�Xc�A�)*

loss
M�<,�̹       �	��H�Xc�A�)*

loss��s;�2�       �	+jI�Xc�A�)*

loss��;�#@       �	��I�Xc�A�)*

loss��<:<�       �	O�J�Xc�A�)*

loss*%I;6���       �	�%K�Xc�A�)*

loss��&=>3i�       �	��K�Xc�A�)*

loss��<ʓ�<       �	�TL�Xc�A�)*

loss��<=>u       �	��L�Xc�A�)*

lossc��=㘽�       �	b�M�Xc�A�)*

loss/=�`v       �	N�Xc�A�)*

loss��?;�O0�       �	-�N�Xc�A�)*

loss�Ys<�l       �	kEO�Xc�A�)*

loss��;����       �	��O�Xc�A�)*

loss�:x<�w       �	�tP�Xc�A�)*

lossV��:f�xn       �	=Q�Xc�A�)*

loss� :l��X       �	ץQ�Xc�A�)*

loss6]<5HА       �	�9R�Xc�A�)*

loss� <<L���       �	��R�Xc�A�)*

loss`�@<%3�=       �	�xS�Xc�A�)*

loss�Ʌ;�_�K       �	�T�Xc�A�)*

loss�	=���       �	P�T�Xc�A�)*

loss�Z<�o.       �	}AU�Xc�A�)*

loss#@<�$�       �	o�U�Xc�A�)*

loss��=�	L9       �	�lV�Xc�A�)*

loss��':^�4a       �	��V�Xc�A�)*

lossw��9�@�       �	J�W�Xc�A�)*

loss7��;S�@       �	�&X�Xc�A�)*

loss��:^YM       �	8�X�Xc�A�)*

loss{��:�pV       �	�QY�Xc�A�)*

loss|�X;ݰ,�       �	�Y�Xc�A�)*

loss�q;h&:D       �	Z~Z�Xc�A�)*

loss_�<�       �	@[�Xc�A�)*

loss*�r8���>       �	�[�Xc�A�)*

loss�s:����       �	:\�Xc�A�)*

lossQ";�w��       �	}�\�Xc�A�)*

lossp(;a��&       �	h]�Xc�A�)*

loss|�a< �k       �	� ^�Xc�A�)*

loss��;�*AS       �	h�^�Xc�A�)*

loss��8Y

P       �	�;_�Xc�A�)*

loss{w9C���       �	�_�Xc�A�)*

loss9a�=g�6�       �	
f`�Xc�A�)*

lossܴ�:0a�       �	��`�Xc�A�)*

loss
%>!1h�       �	G�a�Xc�A�)*

loss�";I�r�       �	&b�Xc�A�)*

lossS�<,<�       �	��b�Xc�A�)*

loss�h�<��       �	�nc�Xc�A�)*

lossS�;��       �	�d�Xc�A�)*

loss/�=$�x       �	Y�d�Xc�A�)*

loss|c�=h��       �	Tte�Xc�A�)*

lossWl�;D�S�       �	�
f�Xc�A�)*

loss��;%ǖ�       �	U�f�Xc�A�)*

loss��:�C�t       �	�Sg�Xc�A�)*

loss��l<R��       �	��g�Xc�A�)*

losswR;<�4       �	V�h�Xc�A�)*

loss��9;����       �	�i�Xc�A�)*

loss��=�ۛ       �	��i�Xc�A�)*

losse�*<�I��       �	�kj�Xc�A�)*

loss$�2;s�t�       �	Wk�Xc�A�)*

loss</��       �	֩k�Xc�A�)*

loss�~�<��ˁ       �	Hl�Xc�A�)*

loss b<N       �	��l�Xc�A�)*

loss?��:��r�       �	Ֆm�Xc�A�)*

loss���;K�X�       �	ʍn�Xc�A�)*

loss���;<){�       �	�Oo�Xc�A�)*

loss��;[��P       �	d�o�Xc�A�)*

lossvu�:{
�       �	�|p�Xc�A�)*

loss���9��S       �	aq�Xc�A�)*

loss@ |<��       �	?�q�Xc�A�**

loss���;�`I^       �	�?r�Xc�A�**

loss��}<��f�       �	��r�Xc�A�**

loss-��< 4	�       �	�hs�Xc�A�**

lossI�$;�X#�       �	Ct�Xc�A�**

loss��
=)�f4       �	̗t�Xc�A�**

loss��y:L!L�       �	8-u�Xc�A�**

loss�<<�`��       �	��u�Xc�A�**

lossM.<��m�       �	�Wv�Xc�A�**

loss3j><ʞ�       �	�v�Xc�A�**

loss�~z:��J5       �	՗w�Xc�A�**

lossڒ�:���       �	�0x�Xc�A�**

loss|��<S�aK       �	�x�Xc�A�**

loss�<D;�       �	�cy�Xc�A�**

loss=5"<辞�       �	� z�Xc�A�**

loss�S_;�s�       �	��z�Xc�A�**

loss�5�<��z       �	&Q{�Xc�A�**

loss���<���       �	��{�Xc�A�**

loss�P�<���       �	W�|�Xc�A�**

loss:�<l��       �	�1}�Xc�A�**

loss�.�<��%�       �	��}�Xc�A�**

loss��:Z�F�       �	�k~�Xc�A�**

loss�F�<M���       �	��Xc�A�**

lossl��9��I�       �	
��Xc�A�**

loss�|;(5ʇ       �	+4��Xc�A�**

loss��;��(       �	����Xc�A�**

lossj�<?�       �	�%��Xc�A�**

loss}g<5[�       �	ۿ��Xc�A�**

loss%<0t�       �	t`��Xc�A�**

loss�`>;?�x       �	f���Xc�A�**

loss���;ǐ��       �	 ���Xc�A�**

loss̖�;O��       �	�D��Xc�A�**

loss8��;�
D-       �	���Xc�A�**

loss�K;����       �	P���Xc�A�**

lossI�<�9       �	�%��Xc�A�**

lossM�_<��i       �	����Xc�A�**

loss�c<�L5�       �	F���Xc�A�**

loss��u=S��;       �	�.��Xc�A�**

loss�h<`F�       �	8���Xc�A�**

lossCG�<�f�       �	�ƥ�Xc�A�**

loss{��;�+�       �	�ʦ�Xc�A�**

loss�);Y|F,       �	r���Xc�A�**

loss1�;�b��       �	�$��Xc�A�**

loss�͎<Q��r       �	����Xc�A�**

loss���<r
       �	(I��Xc�A�**

loss��:�M�+       �	����Xc�A�**

lossFb�=��       �	4���Xc�A�**

lossJ4B<���D       �	�5��Xc�A�**

loss�I�<W��~       �	1ҫ�Xc�A�**

lossJ;d;:-��       �	�r��Xc�A�**

loss�#<!��       �	%��Xc�A�**

loss�[�<`]_       �	/���Xc�A�**

loss�|@<-޹       �	O@��Xc�A�**

loss�[�;�R�       �	�׮�Xc�A�**

loss��<(�       �	rp��Xc�A�**

lossڴ�:��a       �	���Xc�A�**

loss2�:�lެ       �	^���Xc�A�**

loss���<���       �	�+��Xc�A�**

loss}��<u�MM       �	3���Xc�A�**

loss��P:����       �	4d��Xc�A�**

loss
�0=�-:       �	�ʳ�Xc�A�**

lossaZ�:@�d�       �	�c��Xc�A�**

loss�C4=�H�       �	����Xc�A�**

loss�0n;e��M       �	�V��Xc�A�**

loss���;�3�9       �	���Xc�A�**

loss��i:~��~       �	v���Xc�A�**

loss�V:�R       �	�#��Xc�A�**

loss��,;_�       �	�ȸ�Xc�A�**

loss> =��c       �	j��Xc�A�**

loss��I;�$U�       �	���Xc�A�**

lossvP<q�5       �	/���Xc�A�**

loss���;J�@       �	!<��Xc�A�**

loss��<�CC       �	���Xc�A�**

loss|��<;X�S       �	ˁ��Xc�A�**

lossO��9/�Zr       �	L6��Xc�A�**

loss�2�:
��Z       �	Bн�Xc�A�**

loss��]=�D��       �	�e��Xc�A�**

loss�'�<�.�m       �	 ��Xc�A�**

loss��
=����       �	 ���Xc�A�**

lossq�!<x��<       �	1��Xc�A�**

lossnY�:��[y       �	����Xc�A�**

loss �:����       �	Me��Xc�A�**

loss$}D:�2�\       �	���Xc�A�**

loss(3t=Mp��       �	d@��Xc�A�**

loss��u<�R�r       �	���Xc�A�**

losse��=�k{�       �	~q��Xc�A�**

lossOę:�t1�       �	���Xc�A�**

loss��U;��L       �	����Xc�A�**

lossj<�O>Y       �	 p��Xc�A�**

lossW��;�4j*       �	���Xc�A�**

loss�==�p�J       �	K���Xc�A�**

lossdcy<?�(       �	ND��Xc�A�**

loss?��<_P=�       �	����Xc�A�**

loss�M<�*5       �	j���Xc�A�**

loss2�<�T��       �	l��Xc�A�**

lossaɔ<3x�w       �	����Xc�A�**

loss%S�;��       �	,c��Xc�A�**

loss�;�<g��       �		���Xc�A�**

loss���<�Z��       �	_���Xc�A�**

loss��:ߎ�+       �	O��Xc�A�**

loss;D�=����       �	����Xc�A�**

loss�f&<�a�       �	x}��Xc�A�**

loss�db=�Kdr       �	���Xc�A�**

lossJ��;G�4       �	����Xc�A�**

lossv}M<�o)       �	�R��Xc�A�**

loss��;&�~�       �	����Xc�A�**

loss�5=%J�z       �	����Xc�A�**

lossx�;�#w       �	���Xc�A�**

lossf�s;Cq�       �	ȴ��Xc�A�**

loss���<�}�       �	DP��Xc�A�**

loss1��<#�l=       �	����Xc�A�**

loss�R<���       �	b���Xc�A�**

loss��:�^zU       �	��Xc�A�**

loss�}�<��0�       �	m���Xc�A�**

lossf?�<�4�*       �	eP��Xc�A�**

loss���:����       �	����Xc�A�**

loss��v<<vI�       �	����Xc�A�**

loss�? <W_r�       �	�5��Xc�A�**

loss�a�<?��       �	-���Xc�A�**

loss�+�;"�_�       �	~p��Xc�A�**

loss��;,U �       �	���Xc�A�**

lossa#�=�"X�       �	���Xc�A�**

loss�E�<f       �	cE��Xc�A�**

loss\�;x�FP       �	>���Xc�A�**

loss�J<<S�N       �	Ӄ��Xc�A�**

loss�8�;�h�       �	� ��Xc�A�**

loss�ܱ<O#       �	����Xc�A�**

loss��j:�=       �	bf��Xc�A�**

loss�;�:�'II       �	���Xc�A�+*

loss�p.<.��}       �	����Xc�A�+*

loss��}<��-�       �	�Z��Xc�A�+*

loss���;z�WM       �	X8��Xc�A�+*

loss�>X=B5�H       �	]��Xc�A�+*

loss|d�:.�       �	�'��Xc�A�+*

lossƚ<9�F�       �	M��Xc�A�+*

loss��+=m�	�       �	����Xc�A�+*

loss@=����       �	=���Xc�A�+*

loss.�=I
X�       �	-[��Xc�A�+*

lossFJ:N"z�       �	�5��Xc�A�+*

loss�]�<�
λ       �	����Xc�A�+*

loss�;4��       �	�p��Xc�A�+*

loss��;�%B       �	���Xc�A�+*

lossl<�:�?H�       �	;8��Xc�A�+*

loss6zC<@��n       �	����Xc�A�+*

losse�<��C�       �	�a��Xc�A�+*

loss���;��^�       �	Q���Xc�A�+*

loss��/=Yh�       �	����Xc�A�+*

loss�;��       �	�!��Xc�A�+*

lossQz�<���*       �	l���Xc�A�+*

loss���;-H '       �	4d��Xc�A�+*

loss��#<@l\�       �	��Xc�A�+*

lossݗz<-#�       �	����Xc�A�+*

loss{�d:��       �	X:��Xc�A�+*

loss�.�<tCig       �	l���Xc�A�+*

loss)�H=�~       �	S���Xc�A�+*

lossoQ(=�N�:       �	�F��Xc�A�+*

lossE"#;���       �	h���Xc�A�+*

loss���<x8�       �	���Xc�A�+*

loss���<�(        �	g'��Xc�A�+*

loss��o=��        �	���Xc�A�+*

loss��n;�I�B       �	P��Xc�A�+*

loss
��;�w6�       �	����Xc�A�+*

loss�)F;\k&S       �	�|��Xc�A�+*

loss�ۮ<9"�x       �	Y��Xc�A�+*

lossm~%;�
       �	г��Xc�A�+*

loss�:�An�       �	�I��Xc�A�+*

lossHV`<�1�       �	����Xc�A�+*

lossWm�<Q,!�       �	����Xc�A�+*

lossJ�L;UK�       �	�0��Xc�A�+*

loss��<���       �	���Xc�A�+*

loss/�;R���       �	�d��Xc�A�+*

loss�&[<z�       �	2!��Xc�A�+*

loss��$;"�N�       �	����Xc�A�+*

loss,/�<]�2       �	�P��Xc�A�+*

loss�:!tR�       �	\���Xc�A�+*

loss�p<l�        �	�{��Xc�A�+*

lossi&;<��       �	�7 �Xc�A�+*

lossF��;j.��       �	�� �Xc�A�+*

loss<��=�Mw       �	�l�Xc�A�+*

loss�w<)��       �	�9�Xc�A�+*

lossVz�9؇�       �	��Xc�A�+*

loss?'><#	       �	h�Xc�A�+*

loss���<�w�J       �	���Xc�A�+*

loss��"<%x)]       �	��Xc�A�+*

loss��<�`�       �	�4�Xc�A�+*

loss|iJ=^ܖ`       �	���Xc�A�+*

loss��;�勸       �	&m�Xc�A�+*

loss��:<c���       �	��Xc�A�+*

loss,�];�Wl6       �	�U�Xc�A�+*

loss�=�<�5cV       �	���Xc�A�+*

loss���9� �X       �	��	�Xc�A�+*

loss';���       �	�C
�Xc�A�+*

loss��;�vr�       �	E�
�Xc�A�+*

lossr�_9�r��       �	�o�Xc�A�+*

loss��;���s       �	�Xc�A�+*

loss]5�<�NKU       �	B��Xc�A�+*

loss�Q�:;�       �	�5�Xc�A�+*

loss��:�☶       �	���Xc�A�+*

lossB�<�
T�       �	k�Xc�A�+*

lossə�<o3+       �	���Xc�A�+*

loss�/<�x�R       �	6��Xc�A�+*

loss���;-��(       �	g+�Xc�A�+*

loss,T=�T=�       �	��Xc�A�+*

lossp�=<�j3       �	�Q�Xc�A�+*

loss��;JS�       �	���Xc�A�+*

loss��:pH       �	y�Xc�A�+*

loss�]�;�e�g       �	��Xc�A�+*

loss��<��D�       �	��Xc�A�+*

loss��:���<       �	�5�Xc�A�+*

loss��|9�^�a       �	��Xc�A�+*

lossv:Χ�       �	N��Xc�A�+*

loss�%=�(Zd       �	�Xc�A�+*

loss��=<��UZ       �	;��Xc�A�+*

loss-*{<��w0       �	�?�Xc�A�+*

lossTD�<0vCR       �	��Xc�A�+*

lossO�=�s�       �	���Xc�A�+*

lossƇ<=!��C       �	�3�Xc�A�+*

loss�L�;���(       �	e��Xc�A�+*

loss��a9g�       �	+i�Xc�A�+*

loss@�Q9K�&�       �	T��Xc�A�+*

lossC/v<Ee�       �	���Xc�A�+*

loss���;����       �	I+�Xc�A�+*

lossA�:<qQm       �	v��Xc�A�+*

loss��=\M�       �	V�Xc�A�+*

loss�z�;*Ums       �	���Xc�A�+*

loss���=�]��       �	���Xc�A�+*

loss��.<��'�       �	��Xc�A�+*

loss�Z�:[�L�       �	� �Xc�A�+*

loss`f�<����       �	�� �Xc�A�+*

lossFw�<��{�       �	W=!�Xc�A�+*

loss}:�6�:       �	c�!�Xc�A�+*

lossX�v<@�<�       �	Mj"�Xc�A�+*

loss*�
<G�:g       �	�"�Xc�A�+*

loss�{�<g�`       �	q�#�Xc�A�+*

loss4��<�tl       �	P$�Xc�A�+*

loss= <�ձ]       �	�$�Xc�A�+*

loss�n:2u�{       �	Ӈ%�Xc�A�+*

lossm�*;����       �	&�Xc�A�+*

loss2�(<�E)�       �	�&�Xc�A�+*

loss�FB<����       �	�G'�Xc�A�+*

loss��*=hww       �	|�'�Xc�A�+*

lossR*>�l�       �	��(�Xc�A�+*

loss$�<=�k1&       �	� )�Xc�A�+*

loss�Q�;�D�r       �	��)�Xc�A�+*

loss� =�b�       �	�[*�Xc�A�+*

loss�� =�Q�       �	V�*�Xc�A�+*

loss ��:�蓷       �	+�Xc�A�+*

loss,�:�	W       �	�,�Xc�A�+*

loss�i�;�.*�       �	�&-�Xc�A�+*

loss��:�ї�       �	=�-�Xc�A�+*

loss�<k�H       �	QO.�Xc�A�+*

loss���<�T��       �	�.�Xc�A�+*

loss ��:���       �	�/�Xc�A�+*

loss�9l;�       �	�0�Xc�A�+*

lossJr�;�z�[       �	��0�Xc�A�+*

lossZX<[><D       �	�O1�Xc�A�+*

loss��:Vō3       �	G�1�Xc�A�,*

loss&w�<)       �	��2�Xc�A�,*

loss��<W�       �	؀3�Xc�A�,*

losslI�<��5,       �	4�Xc�A�,*

loss�؏<�:,�       �	��4�Xc�A�,*

loss
�2= ��]       �	��5�Xc�A�,*

loss�*<���E       �	%6�Xc�A�,*

losst	=�q�       �	��6�Xc�A�,*

loss�#�;��J�       �	�j7�Xc�A�,*

lossmx4<�bi       �	��7�Xc�A�,*

loss��
=�O        �	��8�Xc�A�,*

losss�:)�T�       �	�89�Xc�A�,*

loss:R�=���       �	��9�Xc�A�,*

loss6o�<�#       �	��:�Xc�A�,*

lossB�=��7�       �	��;�Xc�A�,*

loss��:��S       �	-"<�Xc�A�,*

lossƄ`<��=       �	g�<�Xc�A�,*

loss�r=I�}'       �	UM=�Xc�A�,*

loss��;[��f       �	G�=�Xc�A�,*

loss��=�-I       �	Ӄ>�Xc�A�,*

loss�
s=��x       �	?�Xc�A�,*

lossjF�;��-7       �	�?�Xc�A�,*

loss�@<�e̤       �	(D@�Xc�A�,*

loss��:} ��       �	{A�Xc�A�,*

loss*2�;]s�       �	��A�Xc�A�,*

lossh�2=)�w       �	�:B�Xc�A�,*

lossgԘ:=v��       �	��B�Xc�A�,*

lossAJ<��U       �	TpC�Xc�A�,*

loss�0#<ڣ��       �	%D�Xc�A�,*

loss��F=_�E�       �	,�D�Xc�A�,*

lossݻ<>aI       �	f/E�Xc�A�,*

loss�h<�<�       �	&�E�Xc�A�,*

loss���;��       �	[F�Xc�A�,*

loss��<<�S       �	G�Xc�A�,*

loss�y�;��ǹ       �	d�G�Xc�A�,*

lossώ�<y(�       �	KH�Xc�A�,*

loss��<�b�X       �	��H�Xc�A�,*

lossfJ5<[0�       �	�|I�Xc�A�,*

lossĽ;<iF^       �	jJ�Xc�A�,*

loss�rl<�hŹ       �	t�J�Xc�A�,*

loss�B�;(}�       �	�VK�Xc�A�,*

lossٸ�;���d       �	]�K�Xc�A�,*

lossZ�=��a�       �	�L�Xc�A�,*

lossJ�/=��M�       �	�:M�Xc�A�,*

loss��=Ϊ'(       �	�M�Xc�A�,*

lossM��:�EC       �	sfN�Xc�A�,*

loss{�_;�!��       �	{�O�Xc�A�,*

lossldr;�l,�       �	6:P�Xc�A�,*

lossLY�<����       �	x�P�Xc�A�,*

loss�Fv;�*Oe       �	�mQ�Xc�A�,*

loss,�;�}       �	|R�Xc�A�,*

loss��:?V       �	
�R�Xc�A�,*

lossRm�:���V       �	~�S�Xc�A�,*

loss�	s;G](       �	�8T�Xc�A�,*

loss�!h=�z,       �	:�T�Xc�A�,*

loss[ɦ<?W7�       �	��U�Xc�A�,*

loss�h�<��Ԍ       �	c*V�Xc�A�,*

loss��<F�{�       �	7�V�Xc�A�,*

loss�d~;8�%       �	mXW�Xc�A�,*

lossщl;��0       �	��W�Xc�A�,*

loss ��<4�<)       �	��X�Xc�A�,*

loss��<��N�       �	�:Y�Xc�A�,*

loss=�j:���       �	]�Y�Xc�A�,*

loss'@�;V���       �	��Z�Xc�A�,*

loss�$�;�F7       �	�B[�Xc�A�,*

lossfN�:��f�       �	y�[�Xc�A�,*

loss��x=B���       �	A~\�Xc�A�,*

loss��G<���?       �	�]�Xc�A�,*

loss)�-<��       �	u�]�Xc�A�,*

loss�};<���9       �	�Q^�Xc�A�,*

loss@��:�y<       �	��^�Xc�A�,*

loss��<V�       �	��_�Xc�A�,*

lossf@�=�>�%       �	2`�Xc�A�,*

loss\"=P�(       �	F�`�Xc�A�,*

lossF	<��Ӷ       �	�La�Xc�A�,*

loss|2�:O4�       �	��a�Xc�A�,*

loss�K<��b       �	O�b�Xc�A�,*

loss:D�;��3       �	�-c�Xc�A�,*

loss�%<� �f       �	\d�Xc�A�,*

lossE�;�oN       �	w�d�Xc�A�,*

loss!�<x%       �	�me�Xc�A�,*

loss�<�nѢ       �	�
f�Xc�A�,*

loss�m;�;/       �	��f�Xc�A�,*

loss�8�9%*�       �	�Tg�Xc�A�,*

loss�0�;��]       �	�?h�Xc�A�,*

lossC�a:��L       �	n�h�Xc�A�,*

loss�[�;��Qe       �	Ϡi�Xc�A�,*

loss
�<���       �	dj�Xc�A�,*

loss 
�;�א\       �	B	k�Xc�A�,*

loss���;'�       �	Q�k�Xc�A�,*

loss�Y0=��i:       �	�l�Xc�A�,*

lossin=�J�       �	x�m�Xc�A�,*

loss���95�
�       �	j�n�Xc�A�,*

loss��:]a�S       �	PQo�Xc�A�,*

loss��<��B�       �	nNp�Xc�A�,*

loss�=��a�       �	wdq�Xc�A�,*

loss�7:=MK�A       �	�(r�Xc�A�,*

lossp�<I�º       �	��s�Xc�A�,*

loss��=���       �	��t�Xc�A�,*

lossm��<��n       �	�cu�Xc�A�,*

loss���:��ڧ       �	4+v�Xc�A�,*

loss�O;	�f�       �	l�v�Xc�A�,*

loss�|:%�	�       �	Qkw�Xc�A�,*

loss4�);F��       �	�x�Xc�A�,*

loss;�K:�`�?       �	z�x�Xc�A�,*

loss�k=�8�       �	ZIy�Xc�A�,*

loss�F=;�qF�       �	�8z�Xc�A�,*

lossbJ=��4�       �	�{�Xc�A�,*

loss�92�ׁ       �	��{�Xc�A�,*

loss��;�.��       �	`!}�Xc�A�,*

loss|��<��w�       �	f�}�Xc�A�,*

lossWD�=���       �	�R~�Xc�A�,*

loss|�y:Ďr�       �	�Xc�A�,*

lossd��;��I�       �	���Xc�A�,*

loss~p�;˩.�       �	�[��Xc�A�,*

loss��<f7��       �	Pm��Xc�A�,*

loss"Ҝ<_�p       �	6��Xc�A�,*

loss��<	��       �	����Xc�A�,*

lossh�<(�Bn       �	I+��Xc�A�,*

lossFK�:���i       �	M���Xc�A�,*

loss���<f�e%       �	�T��Xc�A�,*

loss�e�<9��       �	��Xc�A�,*

loss�
�:��¨       �	^���Xc�A�,*

loss�<9X\�       �	�L��Xc�A�,*

loss}�;H�~       �	���Xc�A�,*

loss�$�<�h�       �	�z��Xc�A�,*

lossT�<����       �	n��Xc�A�,*

lossNX;b8��       �	����Xc�A�,*

loss�4�9�)�       �	CT��Xc�A�-*

loss�]<{�a�       �	���Xc�A�-*

loss{5H<j2��       �	T��Xc�A�-*

lossto�;c�ō       �	����Xc�A�-*

lossdw=�U�       �	O@��Xc�A�-*

loss��>Hvvy       �	�܌�Xc�A�-*

lossHzL=Nꌹ       �	����Xc�A�-*

lossiqp;$��       �	b/��Xc�A�-*

lossD�:0N�       �		��Xc�A�-*

loss�uJ;t�c�       �	}z��Xc�A�-*

lossq|<��       �	���Xc�A�-*

loss���;�L2�       �	ͭ��Xc�A�-*

loss�;�`!�       �	�I��Xc�A�-*

loss��:�֭�       �	ߑ�Xc�A�-*

loss��S<Om�K       �	&q��Xc�A�-*

lossXt�:XD�       �	���Xc�A�-*

loss��6=xh�3       �	Ը��Xc�A�-*

lossʾ=���       �	�O��Xc�A�-*

loss���<��!�       �	���Xc�A�-*

loss �;��       �	A���Xc�A�-*

loss�Q<�F�       �	e��Xc�A�-*

lossl5=����       �	ɫ��Xc�A�-*

loss��;�O��       �	AD��Xc�A�-*

lossrr==e9�2       �	4ؗ�Xc�A�-*

loss#=E<=�k�       �	�o��Xc�A�-*

loss�?<�*       �	���Xc�A�-*

lossx/!=�.�       �	���Xc�A�-*

lossoYY=�"�;       �	9��Xc�A�-*

loss���;�}��       �	�њ�Xc�A�-*

lossV�;���       �	Ug��Xc�A�-*

loss#�;t+~�       �	z���Xc�A�-*

loss�==��       �	���Xc�A�-*

loss-ܽ;�:U       �	1��Xc�A�-*

loss��;"��       �	�˝�Xc�A�-*

loss��=���       �	%t��Xc�A�-*

loss���;+4       �	 )��Xc�A�-*

loss��<�	�)       �	`��Xc�A�-*

loss���9�r/       �	���Xc�A�-*

loss���;@�E�       �	1%��Xc�A�-*

loss�Lx:��1       �	����Xc�A�-*

loss�0�<q��       �	:Y��Xc�A�-*

loss� r;�ͱ�       �	8���Xc�A�-*

losse� ;��s       �	Cʣ�Xc�A�-*

loss�X=��0�       �	vm��Xc�A�-*

lossD��<��H�       �	���Xc�A�-*

loss���<Õ�\       �	ݶ��Xc�A�-*

loss�&-<֚�       �	X��Xc�A�-*

loss//�<���#       �	@���Xc�A�-*

loss�#�<�rI       �	���Xc�A�-*

loss�Y<�D=$       �	QO��Xc�A�-*

loss1^9<I��       �	l��Xc�A�-*

loss|�=�v�[       �	o��Xc�A�-*

loss6w=���       �	0-��Xc�A�-*

loss�a9<��RI       �	�ɪ�Xc�A�-*

loss�OJ=W�+       �	b��Xc�A�-*

loss:��<���       �	����Xc�A�-*

loss��s<^��n       �	���Xc�A�-*

loss�=�:���       �	�2��Xc�A�-*

loss\%X<K��       �	$	��Xc�A�-*

loss���<ܙ�       �	 Ϯ�Xc�A�-*

loss��*<�W��       �	�c��Xc�A�-*

loss�!<��9�       �	&���Xc�A�-*

loss�79=cN_       �	�ذ�Xc�A�-*

loss&�*;��٫       �	%���Xc�A�-*

loss1�=�`)       �	���Xc�A�-*

lossM_�;���       �	�G��Xc�A�-*

loss��;��SI       �	E��Xc�A�-*

loss!� =3��       �	 ��Xc�A�-*

loss2V >7-�       �	�{��Xc�A�-*

loss	!{;ɱ�i       �	���Xc�A�-*

lossd�<)D1Q       �	ͮ��Xc�A�-*

loss�#<u�C       �	B��Xc�A�-*

loss)Y�<��W�       �	�շ�Xc�A�-*

lossJ<{\��       �	 q��Xc�A�-*

loss��^;7�+       �	�	��Xc�A�-*

loss��=�UW�       �	����Xc�A�-*

loss���<��0�       �	�1��Xc�A�-*

lossS{;V�ɺ       �	ʺ�Xc�A�-*

loss��<>_��       �	^��Xc�A�-*

loss�d<<C0�       �	
���Xc�A�-*

lossV��<:Z�Q       �	ސ��Xc�A�-*

loss�΁<�k�Z       �	�(��Xc�A�-*

loss�x=
"�       �	����Xc�A�-*

loss��<�t��       �	�m��Xc�A�-*

loss88�;~�O+       �	V��Xc�A�-*

loss<U�;$�v/       �	���Xc�A�-*

loss�5<�D>G       �	�i��Xc�A�-*

lossf9�<*�       �	B��Xc�A�-*

loss�<vJb       �	����Xc�A�-*

loss���<zuy       �	����Xc�A�-*

loss�^�;e��       �	^��Xc�A�-*

loss��_=0[��       �	���Xc�A�-*

lossew�<�ߨ       �	���Xc�A�-*

losse�<C �       �	f2��Xc�A�-*

loss6�; J{�       �	����Xc�A�-*

lossDr<���       �	wj��Xc�A�-*

loss�E<6�iY       �	���Xc�A�-*

loss*��;� ��       �	����Xc�A�-*

lossZ�=uԕ<       �	�E��Xc�A�-*

loss���;�W��       �	����Xc�A�-*

loss6�I<���[       �	]���Xc�A�-*

lossS�;!�2       �	Ü��Xc�A�-*

lossj��9�"��       �	�2��Xc�A�-*

loss!ݠ;�.XK       �	%���Xc�A�-*

loss%`�=��P�       �	f��Xc�A�-*

loss�j�9�o��       �	r���Xc�A�-*

lossR��;��7�       �	����Xc�A�-*

loss�':O�2       �	�*��Xc�A�-*

losss�:�i�s       �	����Xc�A�-*

loss���;M�        �	�c��Xc�A�-*

lossq��:�_>       �	����Xc�A�-*

loss�BT=�Fb�       �	l���Xc�A�-*

loss�[4<W�ˑ       �	�.��Xc�A�-*

loss���<�b��       �	����Xc�A�-*

lossC}9=*�&�       �	�b��Xc�A�-*

loss�H;���%       �	h��Xc�A�-*

loss)�;<���       �	y��Xc�A�-*

lossB�;>E��       �	���Xc�A�-*

loss��<$Dw�       �	a4��Xc�A�-*

loss��s<���       �	C���Xc�A�-*

loss��<gHO       �	f��Xc�A�-*

lossI�a=����       �	���Xc�A�-*

loss�w�;?��       �	_���Xc�A�-*

loss��;-��       �	M.��Xc�A�-*

loss�Q�=3�z�       �	����Xc�A�-*

loss{�8<��z       �	�i��Xc�A�-*

lossqۀ<[���       �	����Xc�A�-*

loss]��;�B�       �	\���Xc�A�-*

loss�R8=?ƃj       �	�5��Xc�A�.*

loss݆�;-:�}       �	����Xc�A�.*

loss�
=��        �	\v��Xc�A�.*

loss��<�k�]       �	���Xc�A�.*

lossDn�:��       �	���Xc�A�.*

lossO+�:���       �	O��Xc�A�.*

loss�ݎ;��[       �	-���Xc�A�.*

loss�}�;i
]�       �	����Xc�A�.*

lossTj{<���O       �	)#��Xc�A�.*

loss���;����       �	]���Xc�A�.*

loss?ߌ;��f{       �	X��Xc�A�.*

lossb �;� z/       �	����Xc�A�.*

lossA_b<�i��       �	ׅ��Xc�A�.*

loss���;���       �	�!��Xc�A�.*

loss�\�:�#�       �	���Xc�A�.*

lossr��<!��       �	XU��Xc�A�.*

loss�;_*��       �	j���Xc�A�.*

lossx��<�:ũ       �	u���Xc�A�.*

lossD�	==aEM       �	�9��Xc�A�.*

loss";�'�       �	�r��Xc�A�.*

loss��;����       �	:\��Xc�A�.*

loss���;֖4       �	�%��Xc�A�.*

loss�e�<T�o       �	o���Xc�A�.*

loss���<���       �	UO��Xc�A�.*

loss}�t<�:�       �	���Xc�A�.*

lossl��;(�2`       �	^���Xc�A�.*

loss��<z�"B       �	+��Xc�A�.*

loss��T=���       �	P���Xc�A�.*

loss3,;��;       �	lB��Xc�A�.*

loss��=y��       �	����Xc�A�.*

loss�O=;�i��       �	!���Xc�A�.*

loss�~:+���       �	����Xc�A�.*

loss��<aG�7       �	v��Xc�A�.*

lossf[�:�0��       �	����Xc�A�.*

loss���<iu��       �	�H��Xc�A�.*

lossbB�9���       �	���Xc�A�.*

lossie�;��
�       �	����Xc�A�.*

loss17�<��B       �	�y��Xc�A�.*

loss�=*�_/       �	���Xc�A�.*

lossV@K;���        �	����Xc�A�.*

loss5E:Fq�g       �	Ic��Xc�A�.*

loss��=��*(       �	;���Xc�A�.*

loss�<^��%       �	���Xc�A�.*

loss���<�Up       �	�I��Xc�A�.*

loss	{�<���       �	����Xc�A�.*

lossoR�<�H�T       �	1���Xc�A�.*

lossȅ�<���
       �	i7��Xc�A�.*

loss}�&:���       �	���Xc�A�.*

loss�"�:��X�       �	�u��Xc�A�.*

lossO��;;�uG       �	���Xc�A�.*

loss��t<��       �	>���Xc�A�.*

loss�ކ9�2�       �	�b��Xc�A�.*

loss��D:�       �	����Xc�A�.*

loss���8G��:       �	T���Xc�A�.*

loss�!.8��y       �	=H��Xc�A�.*

loss�;8>`?       �	����Xc�A�.*

lossXq�:�˴       �	7���Xc�A�.*

loss�V�<�К       �	:< �Xc�A�.*

loss*{�;7�v       �	T8�Xc�A�.*

loss���7�>b       �	��Xc�A�.*

loss���;[YN�       �	b��Xc�A�.*

loss���<�*_�       �	bJ�Xc�A�.*

loss|��9;%,m       �	���Xc�A�.*

loss���=��       �	�Xc�A�.*

lossL�%<y�q       �	#�Xc�A�.*

lossʧ�=�+_       �	U��Xc�A�.*

losst�N;{L�<       �	n�Xc�A�.*

loss��=	{sV       �	$�Xc�A�.*

loss��5;8eā       �	���Xc�A�.*

loss��T<F�W1       �	���Xc�A�.*

loss�)^;P�x/       �	�a	�Xc�A�.*

loss�!
<i�xP       �	r�	�Xc�A�.*

loss�kt;�	��       �	b�
�Xc�A�.*

loss���<�a�       �	@K�Xc�A�.*

loss�8=����       �	���Xc�A�.*

loss��b<TA��       �	��Xc�A�.*

lossQ�]=C�_�       �	�M�Xc�A�.*

loss-	D<t|�A       �	���Xc�A�.*

lossRL�;�w�k       �	��Xc�A�.*

lossI;C�OM       �	�:�Xc�A�.*

lossΪ�<��6       �	B��Xc�A�.*

loss���9����       �	H6�Xc�A�.*

loss�+;���D       �	���Xc�A�.*

loss�Ur;i�       �	��Xc�A�.*

lossh=���       �	�`�Xc�A�.*

lossAI�:���       �	� �Xc�A�.*

loss��%;���       �	��Xc�A�.*

loss?�;s*�       �	"3�Xc�A�.*

loss��:�Vƙ       �	���Xc�A�.*

loss͖�<)��*       �	�p�Xc�A�.*

lossg��<����       �	P�Xc�A�.*

loss?>�<��       �	���Xc�A�.*

lossh��:�~��       �	g_�Xc�A�.*

loss�;\WWc       �	��Xc�A�.*

loss/��:m���       �	{��Xc�A�.*

loss�;D�le       �	:�Xc�A�.*

lossl·:���       �	���Xc�A�.*

loss��<��b       �	yt�Xc�A�.*

loss<,��       �	k�Xc�A�.*

loss[:�<�q��       �	���Xc�A�.*

loss��<k���       �	`[�Xc�A�.*

loss�ۄ;X�Z`       �	m��Xc�A�.*

lossT-G:XV�x       �	"��Xc�A�.*

loss�4$<���       �	G�Xc�A�.*

lossI�/<�r�       �	O��Xc�A�.*

loss�u<`��       �	�� �Xc�A�.*

lossO��;:jN�       �	o+!�Xc�A�.*

loss�=$��       �	�!�Xc�A�.*

loss!J=E��W       �	�r"�Xc�A�.*

loss��X:A�       �	Z#�Xc�A�.*

loss��<�64       �	��#�Xc�A�.*

loss�C;K�i�       �	 U$�Xc�A�.*

loss��G<%Hh       �	�$�Xc�A�.*

loss�e�:�^2       �	�?�Xc�A�.*

loss���<r�5       �	��?�Xc�A�.*

lossT�|<�N�       �	i6@�Xc�A�.*

loss}�0<��NZ       �	�@�Xc�A�.*

loss��;��       �	�mA�Xc�A�.*

loss�?o<��[�       �	�B�Xc�A�.*

loss�f�<�}�j       �	H�B�Xc�A�.*

loss8��<���H       �	��C�Xc�A�.*

lossG�;3��       �	�gD�Xc�A�.*

loss�@%<���       �	��D�Xc�A�.*

loss��;�Y�       �	h�E�Xc�A�.*

loss���<�Ԇ1       �	0-F�Xc�A�.*

loss��:~Z�       �	�1G�Xc�A�.*

loss"��:0M�       �	��G�Xc�A�.*

lossC?)=��       �	R}H�Xc�A�.*

loss��'=��bV       �	1?I�Xc�A�/*

loss�.u9E���       �	��I�Xc�A�/*

loss��.:8ֿ       �	.�J�Xc�A�/*

loss�8�<P��       �	�&K�Xc�A�/*

loss�g>=L�4       �	�K�Xc�A�/*

loss�{B;��$�       �	p�L�Xc�A�/*

loss���;7�`q       �	�<M�Xc�A�/*

loss�e�;b�E       �	P�M�Xc�A�/*

lossl�=�(�       �	˅N�Xc�A�/*

loss�<���_       �	$O�Xc�A�/*

loss5�:�ȧ]       �	z�O�Xc�A�/*

loss���<��|�       �	ͯP�Xc�A�/*

loss	L<�S�~       �	�JQ�Xc�A�/*

lossC؏=Q�       �	>#R�Xc�A�/*

loss��X=��m�       �	�S�Xc�A�/*

loss	��9ش1�       �	�S�Xc�A�/*

loss}
�;m=�       �	�RU�Xc�A�/*

loss�9�;簻�       �	��U�Xc�A�/*

loss���<�㹗       �	<�V�Xc�A�/*

loss��;��]�       �	�(W�Xc�A�/*

lossE�O;����       �	��W�Xc�A�/*

loss�=-;,���       �	�iX�Xc�A�/*

loss?�|<�%�l       �	\Y�Xc�A�/*

loss	P�;��s       �	��Y�Xc�A�/*

loss��;��       �	�HZ�Xc�A�/*

lossL�9<}��       �	��Z�Xc�A�/*

loss�×<�1Q       �	Wz[�Xc�A�/*

loss��:o�       �	�\�Xc�A�/*

loss��<��/�       �	�\�Xc�A�/*

lossvV@<����       �	�G]�Xc�A�/*

loss�9i;���T       �		�^�Xc�A�/*

loss��:�Be       �	��_�Xc�A�/*

loss��<��A\       �	w`�Xc�A�/*

loss��<��f�       �	^a�Xc�A�/*

lossO�;q�       �	��a�Xc�A�/*

loss	P�;f�Y       �	�Sb�Xc�A�/*

loss��<����       �	��b�Xc�A�/*

loss�/=J;$       �	�d�Xc�A�/*

loss(�<Q�t       �	I�d�Xc�A�/*

loss���;��L       �	�<e�Xc�A�/*

loss�q3<zm�L       �	g�e�Xc�A�/*

loss�+a:3B�?       �	�gf�Xc�A�/*

lossM\	9G�=�       �	�@g�Xc�A�/*

loss�U�:�j�       �	M�g�Xc�A�/*

lossR̀:�nD6       �	�vh�Xc�A�/*

loss�>A��       �	�i�Xc�A�/*

lossݸv<�C       �	y<j�Xc�A�/*

loss�b�;���       �	s�j�Xc�A�/*

lossT�=�=(�       �	��k�Xc�A�/*

loss�Q:j��       �	�.l�Xc�A�/*

loss���:�P�       �	��l�Xc�A�/*

lossӪr;�n�_       �	�wm�Xc�A�/*

loss!�:5N�c       �	�Mn�Xc�A�/*

loss7��<Ѓ�e       �	��n�Xc�A�/*

lossQ��;�]�       �	��o�Xc�A�/*

loss�^;�C�5       �	�)p�Xc�A�/*

loss_�|<�D�       �	��p�Xc�A�/*

loss���;�P��       �	�[q�Xc�A�/*

loss��<�l�       �	� r�Xc�A�/*

loss���;��b       �	��r�Xc�A�/*

lossA��;��	�       �	Jbs�Xc�A�/*

loss���;O�y,       �	?�s�Xc�A�/*

loss�d�;\�,       �	��t�Xc�A�/*

lossf�b<��2Z       �	k,u�Xc�A�/*

loss���<Z$��       �	��u�Xc�A�/*

loss�n�<�ޖ       �	ytv�Xc�A�/*

loss�^�;��X�       �	7w�Xc�A�/*

loss�ȴ<-@��       �	��w�Xc�A�/*

lossO��9��a       �	�x�Xc�A�/*

loss���;]�/�       �	$(y�Xc�A�/*

loss��=�p��       �	��y�Xc�A�/*

lossv�:ZP��       �	Llz�Xc�A�/*

lossz��<h�Y�       �	�{�Xc�A�/*

lossL>�;�.*�       �	��{�Xc�A�/*

loss��<8oE       �	ҧ|�Xc�A�/*

loss<*<�| �       �	kF}�Xc�A�/*

loss���<,�Y]       �	��}�Xc�A�/*

loss/��;�1h�       �	�~~�Xc�A�/*

loss���;�:ՙ       �	~�Xc�A�/*

lossI�J;�,       �	��Xc�A�/*

loss�<;{��;       �	�g��Xc�A�/*

loss�5z=~�-       �	���Xc�A�/*

loss�";�l��       �	���Xc�A�/*

loss���=3'�b       �	�F��Xc�A�/*

losswx�<���       �	���Xc�A�/*

loss���;�/��       �	/��Xc�A�/*

loss���<�{j�       �	�j��Xc�A�/*

loss,=l2x6       �	Z��Xc�A�/*

loss֊<Cț@       �	c���Xc�A�/*

loss��^;��Lq       �	l\��Xc�A�/*

loss��<QV\       �	S��Xc�A�/*

loss8�;��ސ       �	���Xc�A�/*

lossLcv<!���       �	�T��Xc�A�/*

loss�#�<߁�       �	u���Xc�A�/*

lossx�";��       �	9��Xc�A�/*

loss�1�=��#%       �	]��Xc�A�/*

loss=�<<�r       �	���Xc�A�/*

loss�+W<�&��       �	A���Xc�A�/*

loss��:$���       �	4-��Xc�A�/*

loss�R�<���j       �	�Ҏ�Xc�A�/*

lossQ`=� �|       �	o��Xc�A�/*

loss�D�;���       �	$��Xc�A�/*

loss���<�Ѻ�       �	3�Xc�A�/*

lossiz=
J8       �	t^��Xc�A�/*

loss���;��m�       �	���Xc�A�/*

loss�B5=R?o�       �	먒�Xc�A�/*

loss��u<}�j(       �	�J��Xc�A�/*

loss}�|;W$:8       �	���Xc�A�/*

loss�#U:����       �	N��Xc�A�/*

loss��"<#        �	����Xc�A�/*

loss�;�8�&       �	�(��Xc�A�/*

lossz��=(���       �	�Ė�Xc�A�/*

losss i;�?       �	Y��Xc�A�/*

loss�tf;�%��       �	����Xc�A�/*

loss�rn<M�7Q       �	^���Xc�A�/*

loss꒻;ϼ(W       �	���Xc�A�/*

lossPo<�ƀ�       �	�"��Xc�A�/*

lossꔖ<_L�       �	����Xc�A�/*

loss��e:x���       �	�j��Xc�A�/*

loss׌B=e�       �	�/��Xc�A�/*

lossO<QDL       �	-"��Xc�A�/*

lossA,<���h       �	鷝�Xc�A�/*

loss���<�Z[�       �	�r��Xc�A�/*

loss
|<�`�       �	�
��Xc�A�/*

loss�p�<��FV       �	����Xc�A�/*

loss�ɕ;�`#;       �	K��Xc�A�/*

loss�f�;9���       �	>��Xc�A�/*

loss��=���       �	ĕ��Xc�A�/*

loss��i=����       �	j4��Xc�A�0*

loss�� =�I�K       �	�ڢ�Xc�A�0*

loss�8�<f�#�       �	0��Xc�A�0*

loss�@;�HLB       �	�!��Xc�A�0*

loss�@.;��N�       �	PŤ�Xc�A�0*

loss�?`<_�Վ       �	�e��Xc�A�0*

loss��:�T�       �	���Xc�A�0*

loss��;�+       �	�Ǧ�Xc�A�0*

lossA]=����       �	;q��Xc�A�0*

loss�r�;��K�       �	���Xc�A�0*

loss}�N<T�p�       �	:Ϩ�Xc�A�0*

lossws�:KbF       �	@j��Xc�A�0*

loss�b�;h8"k       �	i���Xc�A�0*

loss��:���p       �	T���Xc�A�0*

loss�8;$�f�       �	N��Xc�A�0*

loss,�:=�r:;       �	��Xc�A�0*

lossn��:ޕh@       �	敬�Xc�A�0*

loss3�<>
��       �	�?��Xc�A�0*

lossW2�8ze       �	���Xc�A�0*

loss�A�;Z���       �	����Xc�A�0*

loss��9�yH�       �	I.��Xc�A�0*

lossM��;��*y       �	�ϯ�Xc�A�0*

loss#�;EG#�       �	vo��Xc�A�0*

loss�Q�:�;       �	f��Xc�A�0*

loss�==g�       �	"���Xc�A�0*

loss���;����       �	I��Xc�A�0*

loss��	;ʺg       �	��Xc�A�0*

loss�X?<{�x       �	/���Xc�A�0*

loss�J:�~�1       �	[&��Xc�A�0*

lossݦ[:�i       �	{���Xc�A�0*

lossi��:o��       �	�V��Xc�A�0*

lossѹ�:��       �	���Xc�A�0*

loss��<�܃<       �	ެ��Xc�A�0*

lossda�;��       �	�P��Xc�A�0*

lossN�<�c��       �	���Xc�A�0*

loss��=���       �	����Xc�A�0*

loss-dD<t��2       �	�d��Xc�A�0*

loss��9�b�       �	!��Xc�A�0*

lossp�<���       �	i���Xc�A�0*

loss��9s�Gm       �	:<��Xc�A�0*

lossj&�:�MI       �	�Ի�Xc�A�0*

loss�|�9\���       �	�i��Xc�A�0*

lossc��;��M       �	2��Xc�A�0*

lossd�~=��%@       �	Ȗ��Xc�A�0*

lossl�;@�M       �	�,��Xc�A�0*

loss�3<��       �	�Ⱦ�Xc�A�0*

loss�a	;���<       �	�a��Xc�A�0*

loss�j=7U��       �	<��Xc�A�0*

loss��5<O��
       �	>���Xc�A�0*

lossJ�;yG�V       �	�O��Xc�A�0*

loss�1�<�i5.       �	
���Xc�A�0*

loss���;�,H�       �	|���Xc�A�0*

loss\;�H       �	p|��Xc�A�0*

loss��>5˥N       �	��Xc�A�0*

lossm��<Y�/0       �	g���Xc�A�0*

loss6m\;&�2!       �	%���Xc�A�0*

loss���;��cE       �	Á��Xc�A�0*

loss�ɮ:[��       �	���Xc�A�0*

loss�|�=ч#�       �	g���Xc�A�0*

loss"<=2E&       �	8L��Xc�A�0*

loss��;�;}�       �	1���Xc�A�0*

loss���;��{�       �		���Xc�A�0*

loss%;�!��       �	�7��Xc�A�0*

loss��S=�       �	���Xc�A�0*

loss)u�;ч��       �	����Xc�A�0*

loss�֯;�j
A       �	y>��Xc�A�0*

loss}3�:X9ܴ       �	C���Xc�A�0*

loss��^:S2�       �	�x��Xc�A�0*

loss�!�:� W       �	���Xc�A�0*

loss��:��z?       �	C���Xc�A�0*

lossa�;/��       �	�G��Xc�A�0*

loss�=��0f       �	���Xc�A�0*

loss
Fi=F�ۤ       �	�q��Xc�A�0*

loss��:=�xK       �	 $��Xc�A�0*

lossh.A<�t�       �	����Xc�A�0*

loss�j=����       �	.���Xc�A�0*

loss���<�۩�       �	tA��Xc�A�0*

loss^;pӜC       �	���Xc�A�0*

loss�s8;!yg�       �	g{��Xc�A�0*

loss4<4�N�       �	���Xc�A�0*

lossî1<{ ߨ       �	����Xc�A�0*

loss7��<��S^       �	�W��Xc�A�0*

loss
��;��Ť       �	����Xc�A�0*

loss͵�:����       �	]���Xc�A�0*

loss���="��       �	%"��Xc�A�0*

loss��;�k-!       �	A���Xc�A�0*

loss�[�;�s�U       �	7T��Xc�A�0*

losst�z:��.�       �	Y���Xc�A�0*

loss�R:���       �	C���Xc�A�0*

loss�&=ӧ7       �		2��Xc�A�0*

loss��<����       �	����Xc�A�0*

loss��;P�q       �	�^��Xc�A�0*

loss���:�IX0       �	
���Xc�A�0*

loss <�:�� !       �	e���Xc�A�0*

lossZ��=Vq�
       �	s/��Xc�A�0*

loss}Ɣ;9�o�       �	����Xc�A�0*

loss��9J�ԗ       �	p]��Xc�A�0*

loss�� ;s�5       �	����Xc�A�0*

loss�ז;S���       �	���Xc�A�0*

lossCl�=&�s�       �	)^��Xc�A�0*

lossE�F<�V8w       �	� ��Xc�A�0*

loss5e�;M�%`       �	D���Xc�A�0*

lossʘ�<��ı       �	Q���Xc�A�0*

loss<�-<c3�       �	�E��Xc�A�0*

loss�\�<�$�       �	E���Xc�A�0*

losscE"=C^�f       �	2���Xc�A�0*

lossc �<���w       �	6<��Xc�A�0*

lossf<�:Z�I       �	"q��Xc�A�0*

loss��(<D��       �	�,��Xc�A�0*

loss$=��i       �	���Xc�A�0*

loss�9n"u,       �	I���Xc�A�0*

lossNc:X��k       �		q��Xc�A�0*

loss�؀<��Rt       �	'J��Xc�A�0*

loss��; �HI       �	�M��Xc�A�0*

lossҿU=�{�       �	����Xc�A�0*

loss!�y:��S       �	����Xc�A�0*

loss�z�<_]S�       �	0���Xc�A�0*

loss�D<�p�       �	�l��Xc�A�0*

loss�q^;Q���       �	�.��Xc�A�0*

loss���;{�F       �	D���Xc�A�0*

lossM�	<%|!6       �	"��Xc�A�0*

loss�jb<��Ov       �	_���Xc�A�0*

loss��<}%�       �	sL��Xc�A�0*

loss�X�<�mU       �	B	��Xc�A�0*

loss}|�;�&�       �	����Xc�A�0*

lossJB<$$�       �	a���Xc�A�0*

lossl�@<�s       �	����Xc�A�0*

loss�D�<�}1       �	����Xc�A�0*

loss��<�(       �	����Xc�A�1*

loss�ڽ<u�}       �	���Xc�A�1*

loss㮼<��U�       �	����Xc�A�1*

loss�_>�W�-       �	d\��Xc�A�1*

lossD<�TN       �	���Xc�A�1*

loss�:Q;i�]       �	b���Xc�A�1*

loss��:�8�?       �	���Xc�A�1*

loss|�=T��       �	����Xc�A�1*

loss���<�       �	�M��Xc�A�1*

lossüD:v�|       �	����Xc�A�1*

loss�&�< �&       �	J��Xc�A�1*

loss:�-<㗑-       �	���Xc�A�1*

loss�<�L�       �	���Xc�A�1*

loss�<�Z       �	�i �Xc�A�1*

loss�;�9�uv�       �	� �Xc�A�1*

loss�3=�G�        �	ܠ�Xc�A�1*

loss/z�<.��       �	�7�Xc�A�1*

loss��:�G/       �	��Xc�A�1*

lossVX4<2�q^       �	_��Xc�A�1*

loss V�:qS       �	c*�Xc�A�1*

loss_Q;j��       �		P�Xc�A�1*

lossvA�9�D�       �	q��Xc�A�1*

loss��9�o��       �	f��Xc�A�1*

loss#&�:
?b�       �	Q�Xc�A�1*

loss�;���       �	_��Xc�A�1*

lossðE<����       �	U��Xc�A�1*

lossM8k=�I��       �	�	�Xc�A�1*

lossvך;���#       �	��	�Xc�A�1*

loss*�?=2t�J       �	�b
�Xc�A�1*

loss%or=�\@�       �	��
�Xc�A�1*

loss�<{h�       �	��Xc�A�1*

loss�{\;��w       �	f1�Xc�A�1*

lossA�h=��5�       �	G��Xc�A�1*

loss�׺:���3       �	h^�Xc�A�1*

loss�|A<D*��       �	���Xc�A�1*

loss(E-<��       �	Z��Xc�A�1*

loss��:vz�       �	3�Xc�A�1*

loss�b�<���       �	R��Xc�A�1*

loss�p<Α�       �	��Xc�A�1*

loss��	;�J�       �	�+�Xc�A�1*

loss�̫<hPĦ       �	���Xc�A�1*

loss}�<w9�       �	s�Xc�A�1*

lossF'�;��0�       �	{�Xc�A�1*

loss;A�<i�E`       �	!��Xc�A�1*

loss��'=��a       �	�K�Xc�A�1*

loss
dY;�i^       �	���Xc�A�1*

loss&n<�$       �	ۅ�Xc�A�1*

lossN�P<؂�*       �	R,�Xc�A�1*

loss��=�C��       �	@��Xc�A�1*

lossRY;ThE�       �	�g�Xc�A�1*

lossԓ�;!���       �	��Xc�A�1*

loss-�<�4L�       �	���Xc�A�1*

loss��N;�\��       �	L�Xc�A�1*

loss݆�<@���       �	�"�Xc�A�1*

lossZ��=I�֭       �	b��Xc�A�1*

loss��;.���       �	�O�Xc�A�1*

lossC�=�R�+       �	���Xc�A�1*

loss��<�ab�       �	���Xc�A�1*

loss40=O��       �	%#�Xc�A�1*

lossI�b;Z3�W       �	K��Xc�A�1*

loss��i=ǔ0       �	=`�Xc�A�1*

loss���:� �G       �	0�Xc�A�1*

loss�ZA<�p�       �	a��Xc�A�1*

loss7C6=���       �	e �Xc�A�1*

lossQ��<��h       �	K !�Xc�A�1*

loss1�N;���t       �	ؚ!�Xc�A�1*

loss�t�;��!�       �	�9"�Xc�A�1*

loss��=�(:       �	��"�Xc�A�1*

lossT�;J�bQ       �	�i#�Xc�A�1*

loss�/�:�}l       �	��#�Xc�A�1*

loss���;q��4       �	l�$�Xc�A�1*

loss�<����       �	0*%�Xc�A�1*

loss�=���@       �	��%�Xc�A�1*

loss$4�<ܢ*j       �	 �&�Xc�A�1*

lossq�0=�Щ�       �	��'�Xc�A�1*

loss:�<(�=�       �	�(�Xc�A�1*

lossT��<�0	       �	ǹ(�Xc�A�1*

loss��
=���       �	>u)�Xc�A�1*

loss?K�<;�W       �	)	*�Xc�A�1*

loss���:�.�       �	��*�Xc�A�1*

lossv>;�I�       �	�3+�Xc�A�1*

loss\�<�)�I       �	��+�Xc�A�1*

lossZ9);�Е[       �	�m,�Xc�A�1*

loss X�=��:�       �	-�Xc�A�1*

loss� =ѽ�	       �	M�-�Xc�A�1*

loss�C�:X�č       �	�3.�Xc�A�1*

loss@P�;u�r=       �	4�.�Xc�A�1*

loss���;y�       �	;m/�Xc�A�1*

loss$=�;gK~�       �	0�Xc�A�1*

loss�@=>fM�       �	��0�Xc�A�1*

loss�b;ÌG$       �		71�Xc�A�1*

loss$�<����       �	��1�Xc�A�1*

loss��=q�       �	�n2�Xc�A�1*

lossW��;
{��       �	k3�Xc�A�1*

loss��<߄�,       �	��3�Xc�A�1*

loss�2G=V]�	       �	'L4�Xc�A�1*

loss O�<=G�       �	��4�Xc�A�1*

lossXX�9�x��       �	�5�Xc�A�1*

lossE3�;�K3       �	�"6�Xc�A�1*

loss��;=y�	�       �	��6�Xc�A�1*

loss8��9j       �	�o7�Xc�A�1*

lossVj�:�       �	G 8�Xc�A�1*

loss<$:��8       �	I+:�Xc�A�1*

loss�X�;;�       �	��:�Xc�A�1*

loss��_<H���       �	l;�Xc�A�1*

loss�x<��5�       �	�Y<�Xc�A�1*

loss��(=	Cn�       �	��<�Xc�A�1*

loss���;�{t�       �	��=�Xc�A�1*

lossܯ�;6���       �	}%>�Xc�A�1*

loss�#5<T��       �	�?�Xc�A�1*

loss2t7;Y{��       �	��?�Xc�A�1*

loss��=�Fŝ       �	��@�Xc�A�1*

loss�E/:K�       �	AA�Xc�A�1*

loss��:���       �	��A�Xc�A�1*

loss�=�;�-�       �	��B�Xc�A�1*

lossRC�;S��       �	�RC�Xc�A�1*

loss�y�<����       �	c_D�Xc�A�1*

loss�h:��7�       �	z�D�Xc�A�1*

loss�Q[:�R��       �	G�E�Xc�A�1*

lossc3�;Sl֋       �	�{F�Xc�A�1*

loss;Ų<���       �	sG�Xc�A�1*

losss�</���       �	��G�Xc�A�1*

loss��
<��vq       �	#eH�Xc�A�1*

loss�;i�98       �	D�H�Xc�A�1*

loss��j=�V�w       �	�I�Xc�A�1*

loss�4�:+m/       �	�.J�Xc�A�1*

loss�~�;d$�       �	rK�Xc�A�1*

losst�7;j���       �	�
L�Xc�A�1*

loss�Q�<�v��       �	2�L�Xc�A�2*

losspz�<pkNb       �	�_M�Xc�A�2*

loss��	<�E �       �	��M�Xc�A�2*

lossM�<=�0       �	;�N�Xc�A�2*

loss��D<�/�q       �	$O�Xc�A�2*

loss籑<�ON�       �	L�O�Xc�A�2*

loss��B;�,^�       �	x�P�Xc�A�2*

loss�XV<�P�^       �	b,Q�Xc�A�2*

loss�(�=�2)       �	b�Q�Xc�A�2*

loss�t�;0���       �	W[R�Xc�A�2*

lossJ��:̄��       �	|�R�Xc�A�2*

loss%�};���       �	��S�Xc�A�2*

loss�O.=rJh       �	a4T�Xc�A�2*

loss|�0;�	�       �	�U�Xc�A�2*

loss��<���_       �	��U�Xc�A�2*

loss=}<,J       �	+hV�Xc�A�2*

lossMiq=PZ �       �	W�Xc�A�2*

lossD��;znD       �	h�W�Xc�A�2*

lossU��;#V�b       �	gX�Xc�A�2*

lossM�;b�b>       �	Y�Xc�A�2*

loss(D;<t?��       �	��Y�Xc�A�2*

lossŬH;�D"�       �	�@Z�Xc�A�2*

lossAv�<��       �	+�Z�Xc�A�2*

loss�;��m�       �	�{[�Xc�A�2*

loss��=���	       �	'\�Xc�A�2*

loss�e�<�ONn       �	f�\�Xc�A�2*

lossl$�<�_Dp       �	W]�Xc�A�2*

loss�q</�^0       �	��]�Xc�A�2*

loss*��<}P��       �	I�^�Xc�A�2*

lossA�;[z�       �	�_�Xc�A�2*

lossL;ؾ��       �	�_�Xc�A�2*

loss�i�;.��       �	��`�Xc�A�2*

loss*��<@��       �	+1a�Xc�A�2*

lossv~�<�ŏ       �	�a�Xc�A�2*

loss)��;�ewR       �	^gb�Xc�A�2*

loss�3�;%5�        �	T�b�Xc�A�2*

loss���;P�dX       �	�c�Xc�A�2*

loss�<9��       �	8d�Xc�A�2*

lossv�<~��       �	S�d�Xc�A�2*

lossC�<��~       �	�ie�Xc�A�2*

lossȂO<F{�       �	Qf�Xc�A�2*

loss��&<�gYV       �	�f�Xc�A�2*

loss��$<�J�       �	�}g�Xc�A�2*

loss���;I�cm       �	�yh�Xc�A�2*

loss�� =��g�       �	�i�Xc�A�2*

lossx#@9�:c       �	��i�Xc�A�2*

loss�yK;�6ig       �	QNj�Xc�A�2*

loss��B=���1       �	��j�Xc�A�2*

loss�pM<:R�       �	��k�Xc�A�2*

lossӌi<\D/Y       �	��l�Xc�A�2*

loss�j�;�֑;       �	i�m�Xc�A�2*

lossɔ=,0�       �	�tn�Xc�A�2*

loss��;"|ݑ       �	�/o�Xc�A�2*

loss��L;��(�       �	�o�Xc�A�2*

loss �;��       �	{p�Xc�A�2*

loss�1N=s"��       �	@q�Xc�A�2*

loss(�:HZ�       �	�r�Xc�A�2*

lossc�;	(ч       �	��r�Xc�A�2*

lossx;�KTm       �	ds�Xc�A�2*

lossG�:o*�&       �	�^t�Xc�A�2*

loss:�<��        �	MMu�Xc�A�2*

loss���:迏A       �	��u�Xc�A�2*

lossa�:>��B       �	��v�Xc�A�2*

loss�h�;��+       �	��w�Xc�A�2*

loss�Å=GGC       �	�x�Xc�A�2*

loss=�/;$f��       �	��y�Xc�A�2*

loss�L;�V^C       �	Áz�Xc�A�2*

loss��:Z��&       �	�*{�Xc�A�2*

loss�ѣ:�G�       �	D�{�Xc�A�2*

loss
Z�:���       �	��|�Xc�A�2*

loss��9i�       �	�N}�Xc�A�2*

loss�ݼ=Tut�       �	�~�Xc�A�2*

lossi~�;ho�~       �	ץ~�Xc�A�2*

loss�R= ��       �	�H�Xc�A�2*

loss0^=���$       �	���Xc�A�2*

lossă:�U�7       �	ƅ��Xc�A�2*

loss���;�r�       �	�4��Xc�A�2*

loss|�F;�u]�       �	�ԁ�Xc�A�2*

loss�L�<�ײj       �	���Xc�A�2*

loss�F;Q��       �	�O��Xc�A�2*

loss���<Ғ�       �	���Xc�A�2*

lossz�<�֧       �	6���Xc�A�2*

loss�1<���       �	@0��Xc�A�2*

loss��0:6�Z�       �	���Xc�A�2*

loss�RV;\)�       �	����Xc�A�2*

loss�h�<�cA4       �	*��Xc�A�2*

lossp��;OZCE       �	u���Xc�A�2*

loss=��;�        �	F��Xc�A�2*

loss=+�<ʁ{       �	T��Xc�A�2*

losse�L;h�1�       �	�׊�Xc�A�2*

lossS&+:�!��       �	����Xc�A�2*

loss�Z6;lW�       �	�2��Xc�A�2*

loss�e;���>       �	�Ɍ�Xc�A�2*

loss$4 :���5       �	k��Xc�A�2*

loss|ź<�ԟ�       �	P��Xc�A�2*

loss�-�9Q.�)       �	����Xc�A�2*

lossfK<`v0       �	-��Xc�A�2*

loss�&/=$�ox       �	���Xc�A�2*

loss�;�9��       �	}���Xc�A�2*

loss��<Ϛ:7       �	�/��Xc�A�2*

loss�0r<�.       �	�ő�Xc�A�2*

loss(�<yÌG       �	P���Xc�A�2*

loss�Z�9��)�       �	j���Xc�A�2*

losss�; ��       �	vT��Xc�A�2*

lossM�9-!�3       �	���Xc�A�2*

loss��;�*�       �	���Xc�A�2*

loss�k=�wX       �	�"��Xc�A�2*

loss���:'Wo�       �	���Xc�A�2*

lossT�;��       �	^��Xc�A�2*

loss�ݡ<��N       �	���Xc�A�2*

lossqe%<t��       �	���Xc�A�2*

loss�s7;
l��       �	JD��Xc�A�2*

loss&ܪ;�hX�       �	���Xc�A�2*

loss搶9϶�       �	y���Xc�A�2*

loss��;�x@�       �	o*��Xc�A�2*

loss�~�:��g       �	����Xc�A�2*

lossc�<���       �	�V��Xc�A�2*

loss1�o<�뀤       �	���Xc�A�2*

loss���;�|�       �	�|��Xc�A�2*

loss�<{y\�       �	*��Xc�A�2*

loss�@�<۲@       �	���Xc�A�2*

loss��<o�u_       �	{J��Xc�A�2*

losst��95O7�       �	X��Xc�A�2*

loss;�z;��f�       �	���Xc�A�2*

loss�\�8Zw�U       �	/��Xc�A�2*

loss�2;�T�i       �	����Xc�A�2*

lossh��<1N/       �	�`��Xc�A�2*

losstZ�;�7       �	�.��Xc�A�2*

lossXr,<����       �	iȣ�Xc�A�3*

lossL��=�h�       �	]��Xc�A�3*

loss��<v�ݍ       �	����Xc�A�3*

loss/�Q<%�p8       �	ݙ��Xc�A�3*

loss:��<����       �	<0��Xc�A�3*

lossT��9�w|�       �	&Ħ�Xc�A�3*

loss+�:N=?       �	bi��Xc�A�3*

loss�"n:��i�       �	�7��Xc�A�3*

loss`sX;�]��       �	c��Xc�A�3*

loss�ɍ:��       �	rܩ�Xc�A�3*

loss鴴:
fM       �	l���Xc�A�3*

loss�8NolO       �	���Xc�A�3*

loss��p;Puo       �	#��Xc�A�3*

loss��9����       �	���Xc�A�3*

loss,m9\��       �	�6��Xc�A�3*

lossnϫ9�:��       �	�Я�Xc�A�3*

loss��;|C��       �	ظ��Xc�A�3*

loss�C;�v�       �	ca��Xc�A�3*

loss�!1;�B��       �	���Xc�A�3*

loss�¨8f�T       �	圲�Xc�A�3*

lossj*�9��o       �	5��Xc�A�3*

lossS	 =��Y       �	Aӳ�Xc�A�3*

loss 2�9���       �	����Xc�A�3*

loss)��=�|`       �	5*��Xc�A�3*

lossl~�<F�       �	]���Xc�A�3*

loss���:I�8�       �	TW��Xc�A�3*

lossX��<��6C       �	$��Xc�A�3*

loss�;(�b�       �	�·�Xc�A�3*

loss6��<VoM       �	�b��Xc�A�3*

loss��^<��[*       �	�$��Xc�A�3*

loss3:�;&�*       �	����Xc�A�3*

loss�^2=+&�       �	uZ��Xc�A�3*

loss�͏;JkO�       �	���Xc�A�3*

lossWb�=/U�       �	���Xc�A�3*

loss�ȗ<�O�       �	j.��Xc�A�3*

loss�g�;�u%5       �	�ټ�Xc�A�3*

loss=�<Y�z�       �	�w��Xc�A�3*

loss�]�<:k �       �	���Xc�A�3*

loss�G�<�U?<       �	z���Xc�A�3*

lossR�!=��ݨ       �	u=��Xc�A�3*

lossx 3=�z�       �	8ؿ�Xc�A�3*

loss��<2�ϟ       �	�q��Xc�A�3*

lossC:�sf�       �	A��Xc�A�3*

loss���<I}�       �	ߤ��Xc�A�3*

loss��<�p�       �	�9��Xc�A�3*

loss#�;�߲       �	����Xc�A�3*

loss���</Q�       �	Eh��Xc�A�3*

loss4]�:v�~<       �	����Xc�A�3*

loss�G�<Hg(�       �	h���Xc�A�3*

loss�lq<���S       �	,��Xc�A�3*

losscF8<ԏn�       �	`���Xc�A�3*

lossR��<����       �	N^��Xc�A�3*

loss�J�:���       �	4e��Xc�A�3*

lossܕ<S��       �	���Xc�A�3*

loss���:��<�       �	7���Xc�A�3*

loss��<��A�       �	_\��Xc�A�3*

lossȴ<4;�;       �	����Xc�A�3*

lossO��;���       �	����Xc�A�3*

loss���<��4�       �	BZ��Xc�A�3*

loss�+/<!� �       �	& ��Xc�A�3*

loss���<4K�       �	o���Xc�A�3*

lossOa.<[%}       �	;Q��Xc�A�3*

loss�Ya;�Q2       �	b���Xc�A�3*

loss��<�>C�       �	E���Xc�A�3*

lossR��;��_�       �	�4��Xc�A�3*

lossM?;k�۵       �	����Xc�A�3*

lossE��;��Ic       �	�v��Xc�A�3*

loss��;�"W�       �	7��Xc�A�3*

loss���<7M       �	����Xc�A�3*

lossL�<sZR       �	\���Xc�A�3*

loss���;�R�       �	�*��Xc�A�3*

lossFC�9���       �	S���Xc�A�3*

loss��<�E�       �	�\��Xc�A�3*

loss)
�:�mZ^       �	@���Xc�A�3*

loss$�<&�       �	�&��Xc�A�3*

lossa��<3@�       �	����Xc�A�3*

loss�^-=�v��       �	sh��Xc�A�3*

loss�ov;>w�       �	���Xc�A�3*

loss��0;�        �	A���Xc�A�3*

loss$�}</Y�5       �	�V��Xc�A�3*

loss��;g9�3       �	���Xc�A�3*

loss?�D<��+       �	����Xc�A�3*

loss0�<�       �	W"��Xc�A�3*

loss��<��ݣ       �	s���Xc�A�3*

loss�{:�"\       �	�R��Xc�A�3*

lossmoI=���N       �	����Xc�A�3*

loss���;���       �	����Xc�A�3*

loss��;��n�       �	�'��Xc�A�3*

lossnx�<�0��       �	^���Xc�A�3*

loss��V8a�ja       �	����Xc�A�3*

lossH�<�x�       �	P��Xc�A�3*

loss�@>;3�x(       �	f���Xc�A�3*

loss��;Z���       �	S��Xc�A�3*

loss�LP<��ߢ       �	����Xc�A�3*

loss1�*;zٽ       �	�w��Xc�A�3*

lossvK;����       �		��Xc�A�3*

loss�f3=�Nm       �	����Xc�A�3*

loss#��:�,       �	�9��Xc�A�3*

loss��;#��8       �	t���Xc�A�3*

loss���<�~�       �	Rd��Xc�A�3*

loss7��;;���       �	���Xc�A�3*

lossR��<e�z       �	*���Xc�A�3*

loss�ы;$�Ǿ       �	�E��Xc�A�3*

loss
1�:<u:       �	����Xc�A�3*

lossf��<����       �	�o��Xc�A�3*

loss,��;�1��       �	� �Xc�A�3*

lossO�%<'�1=       �	נ �Xc�A�3*

lossk};ǜ1+       �	�3�Xc�A�3*

lossʜ�<AAJ       �	���Xc�A�3*

loss�5;��<�       �	7o�Xc�A�3*

loss㮈;.��       �	��Xc�A�3*

loss���<�Y/       �	���Xc�A�3*

lossNO�;���       �	7�Xc�A�3*

loss���;��5       �	.��Xc�A�3*

loss`xu=q�t       �	j�Xc�A�3*

lossR:e_S       �	}�Xc�A�3*

lossO:<�6�(       �	a��Xc�A�3*

loss��D<܇�       �	$H�Xc�A�3*

loss#Gb=�u�       �	���Xc�A�3*

lossd�<�~�       �	�w�Xc�A�3*

loss*�<2�	        �	�	�Xc�A�3*

lossߎx:f�k�       �	ʣ	�Xc�A�3*

losszW'9yя4       �	77
�Xc�A�3*

loss�.9ikw|       �	��
�Xc�A�3*

loss}]t;G&Z       �	�o�Xc�A�3*

loss�V;Pn�       �	�I�Xc�A�3*

lossx�>=N�ɚ       �	n��Xc�A�3*

loss*�:{���       �	��Xc�A�3*

loss{�[9��$Z       �	�"�Xc�A�4*

loss�iL;�	�       �	\��Xc�A�4*

lossD�9�<.�       �	�~�Xc�A�4*

loss�¸;��y�       �	�"�Xc�A�4*

loss@��<�؂       �	��Xc�A�4*

lossI�<ɏ��       �	�Q�Xc�A�4*

lossڒ�<��g        �	���Xc�A�4*

lossT�8%+:       �	w��Xc�A�4*

loss(é;!��       �	29�Xc�A�4*

lossC��:�Y�       �	���Xc�A�4*

loss��_:r��       �	{��Xc�A�4*

loss�1�:(y�       �	S"�Xc�A�4*

losss�;Ŗ�       �	��Xc�A�4*

loss�-]<�       �	g_�Xc�A�4*

loss�<��@�       �	/��Xc�A�4*

lossȽ�<8�       �	A��Xc�A�4*

loss2`}<hǇ�       �	�t�Xc�A�4*

loss��i<�U!       �	V*�Xc�A�4*

lossė�<�81�       �	���Xc�A�4*

loss_[�:H�8j       �	5b�Xc�A�4*

loss��l;0��       �	��Xc�A�4*

lossF&<Vt��       �	ϡ�Xc�A�4*

loss�$;J+�X       �	�8�Xc�A�4*

loss�x8:���       �	
��Xc�A�4*

loss��R=J�tz       �	�o�Xc�A�4*

loss$"�<��Ӏ       �	S�Xc�A�4*

loss�u�<�~�       �	=��Xc�A�4*

loss���;y7l�       �	I��Xc�A�4*

loss���:Ŋz       �	U. �Xc�A�4*

loss�x:��       �	�� �Xc�A�4*

loss��;��W�       �	�d!�Xc�A�4*

loss�b+<���       �	�"�Xc�A�4*

loss�L�;���       �	`�"�Xc�A�4*

loss��<��ޥ       �	IH#�Xc�A�4*

loss��:�\~1       �	��#�Xc�A�4*

loss1#�<�G5       �	��$�Xc�A�4*

loss.��<*ke�       �	�R%�Xc�A�4*

lossC�r;�r��       �	f�%�Xc�A�4*

loss,�;QQk       �	��&�Xc�A�4*

loss;�`<�F�p       �	�7'�Xc�A�4*

loss�l�;�7;       �	��'�Xc�A�4*

lossڦ�<%K�       �	�j(�Xc�A�4*

lossV;c�cq       �	G)�Xc�A�4*

loss&��=       �	��)�Xc�A�4*

loss�jb;���       �	�*�Xc�A�4*

loss=�:��?       �	�3+�Xc�A�4*

loss���9��        �	��,�Xc�A�4*

lossfK�<�6��       �	�--�Xc�A�4*

loss�+!;�-d�       �	>�-�Xc�A�4*

loss��5=�8�       �	du.�Xc�A�4*

lossx>;q{j�       �	j/�Xc�A�4*

lossv,�:�6�I       �	и/�Xc�A�4*

lossO�$<s}n�       �	<N0�Xc�A�4*

loss&=z�Z4       �	��0�Xc�A�4*

lossJ�;99�O       �	��1�Xc�A�4*

loss�<&<��       �	�/2�Xc�A�4*

loss�o�:�ɼ       �	��2�Xc�A�4*

loss��&;M�9�       �	�y3�Xc�A�4*

loss���:�ǈ*       �	�4�Xc�A�4*

loss�Y�:Lp��       �	�5�Xc�A�4*

lossJ�u;�fA�       �	]�5�Xc�A�4*

loss�6X<�]�       �	|B6�Xc�A�4*

lossy�<�?o�       �	��6�Xc�A�4*

lossQ-�<�\5�       �	��7�Xc�A�4*

loss���;�!\       �	7�8�Xc�A�4*

loss��=��!       �	p>9�Xc�A�4*

loss�I�;j��)       �	��9�Xc�A�4*

loss$�:�e�Y       �	Ts:�Xc�A�4*

loss���:1�       �	-	;�Xc�A�4*

losst��:j��R       �	!�;�Xc�A�4*

loss��e=�x�       �	�f<�Xc�A�4*

loss��;���       �	P�<�Xc�A�4*

loss���:/�       �	�=�Xc�A�4*

loss�؈<X��       �	�/>�Xc�A�4*

loss��=��Y�       �	�>�Xc�A�4*

loss|�;=�s�       �	�g?�Xc�A�4*

loss_��<���k       �	��?�Xc�A�4*

loss$� =��       �	.�@�Xc�A�4*

loss_8=GB��       �	�(A�Xc�A�4*

lossf�<��h       �	׿A�Xc�A�4*

lossmi;����       �	}C�Xc�A�4*

loss|�:1�m�       �	!�C�Xc�A�4*

loss 8^;?��       �	˄D�Xc�A�4*

loss�Y:��]       �	�E�Xc�A�4*

loss_:M���       �	6�E�Xc�A�4*

losslc<�H�$       �	�MF�Xc�A�4*

lossb�;��Q�       �	��F�Xc�A�4*

loss�T�<U�K       �	{�G�Xc�A�4*

loss3\u;��-�       �	GH�Xc�A�4*

loss1D<lذ�       �	�H�Xc�A�4*

loss#�+;�tp�       �	�[I�Xc�A�4*

loss��G:�2|�       �	�I�Xc�A�4*

loss���;z��       �	<�J�Xc�A�4*

lossiD�:'�}       �	!<K�Xc�A�4*

lossH�;�DZ�       �	��K�Xc�A�4*

loss�P�;����       �	��M�Xc�A�4*

lossq�;�l3J       �	x�N�Xc�A�4*

loss�)3=��.�       �	75O�Xc�A�4*

loss��=�E�       �	!�O�Xc�A�4*

losso3<y��K       �	��P�Xc�A�4*

loss-}:>�ƥ       �	�BQ�Xc�A�4*

lossʻ�9��(       �	��Q�Xc�A�4*

loss#:i=C7��       �	��R�Xc�A�4*

lossh�<\��       �	i7S�Xc�A�4*

lossHw�;&d<�       �	�S�Xc�A�4*

lossH��<S)       �	~nT�Xc�A�4*

loss?.8;K�/       �	�CU�Xc�A�4*

loss�3<�U�	       �	y�U�Xc�A�4*

loss���7�$�       �	�V�Xc�A�4*

lossqj�<8[k       �	}W�Xc�A�4*

loss�2;;ڷ�Z       �	C�W�Xc�A�4*

loss]c�9N��       �	�dX�Xc�A�4*

loss�0�<�c�       �	Y�Xc�A�4*

loss�:^�       �	�Y�Xc�A�4*

loss4�+;|.�       �	�RZ�Xc�A�4*

loss�R�<1��w       �	��Z�Xc�A�4*

loss
�x;�[g       �	f�[�Xc�A�4*

loss�z:�֍�       �	�&\�Xc�A�4*

loss|Ʈ;�QS       �	��\�Xc�A�4*

loss���;�5{P       �	�|]�Xc�A�4*

loss;C�;q��b       �	a^�Xc�A�4*

loss��&<��vN       �	��^�Xc�A�4*

lossq�q<��@�       �	jP_�Xc�A�4*

loss=��<d#Sr       �	�_�Xc�A�4*

loss3Yy;_��D       �	N}`�Xc�A�4*

lossP� ;��1�       �	�a�Xc�A�4*

lossj+<X��u       �	�a�Xc�A�4*

loss(��=dSH       �	}<b�Xc�A�4*

loss��<�^�m       �	��b�Xc�A�5*

loss���:��6       �	�jc�Xc�A�5*

loss��;�x�       �	� d�Xc�A�5*

loss%�B:�S$       �	��d�Xc�A�5*

loss#Z�:g���       �	�fe�Xc�A�5*

loss?�g=5�I�       �	.f�Xc�A�5*

lossh@<�`�o       �	3�f�Xc�A�5*

loss*i;��c       �	>g�Xc�A�5*

loss��<;BxnN       �	o�g�Xc�A�5*

lossn�;�Cn�       �	�h�Xc�A�5*

loss_Y�;�?x�       �	�}i�Xc�A�5*

loss�= ;���       �	^j�Xc�A�5*

lossc0 : ##I       �	�j�Xc�A�5*

loss��+=&z>�       �	qk�Xc�A�5*

lossx��:g��F       �	�l�Xc�A�5*

loss�Y�;�f�       �	=�l�Xc�A�5*

loss�7�<�l�       �	Rm�Xc�A�5*

loss�<(;��       �	��m�Xc�A�5*

loss&mj;��;       �	��n�Xc�A�5*

loss1�@:�A�       �	 8o�Xc�A�5*

loss@s,:���       �	��o�Xc�A�5*

loss�q!;tAJ�       �	��p�Xc�A�5*

loss��w<ŧ�$       �	�$q�Xc�A�5*

loss! �<''k       �	j�q�Xc�A�5*

loss��;��       �	Pmr�Xc�A�5*

loss�-�;�-�~       �	0s�Xc�A�5*

lossXڛ<j���       �	?�s�Xc�A�5*

loss�֓<�,�)       �	p`t�Xc�A�5*

loss�q :�d�       �	&u�Xc�A�5*

loss۫9�ׅ�       �	��u�Xc�A�5*

lossV��:8�ѝ       �	�4v�Xc�A�5*

loss�p=�LQ       �	��v�Xc�A�5*

lossP<3o        �	�cw�Xc�A�5*

lossf�=��c       �	��w�Xc�A�5*

loss�w�<le"T       �	�x�Xc�A�5*

lossS�<L.�       �	�6y�Xc�A�5*

loss�{<Z��       �	��y�Xc�A�5*

lossE�:Jp       �	�hz�Xc�A�5*

lossm��:�߀�       �	��z�Xc�A�5*

loss�=u7f0       �	h�{�Xc�A�5*

loss��l;��Lj       �	'.|�Xc�A�5*

loss��8;̌4       �	��|�Xc�A�5*

lossZY�;�DZ�       �	y}�Xc�A�5*

loss�"<=��i       �	�~�Xc�A�5*

losse� :�{6�       �	F�~�Xc�A�5*

loss@Ca=�e�E       �	�N�Xc�A�5*

loss�+<�I�j       �	���Xc�A�5*

lossM�;�f
H       �	���Xc�A�5*

loss-�:�+7�       �	(-��Xc�A�5*

lossR�=�P�j       �	zȁ�Xc�A�5*

loss;�<�c!       �	b��Xc�A�5*

loss�'�<�v       �	���Xc�A�5*

loss���=I���       �	l���Xc�A�5*

lossoƙ9Xz»       �	,��Xc�A�5*

loss�4:�w�&       �	����Xc�A�5*

loss��x;�%�       �	(`��Xc�A�5*

lossyc�9���       �	���Xc�A�5*

loss�2�;~��       �	%���Xc�A�5*

lossZ��<|#       �	�]��Xc�A�5*

loss�,W:j_]       �	0F��Xc�A�5*

loss&*�<��Y�       �	���Xc�A�5*

loss�]�<!Г�       �	�}��Xc�A�5*

loss�}B;�*��       �	���Xc�A�5*

loss��=|Wf�       �	A���Xc�A�5*

lossR�):Bv       �	�V��Xc�A�5*

lossj<>:V�       �	���Xc�A�5*

lossi6�;#�ߧ       �	<���Xc�A�5*

loss*(=�/       �	���Xc�A�5*

loss�7�9�K��       �	ɰ��Xc�A�5*

loss_�{;m\�       �	J��Xc�A�5*

loss�mH;ĝ�       �	d��Xc�A�5*

loss��r;-�v       �	x��Xc�A�5*

loss<��;���       �	��Xc�A�5*

loss]ƾ<���5       �	iÐ�Xc�A�5*

loss�i_:�|�f       �	j��Xc�A�5*

loss��0;�D       �	���Xc�A�5*

loss�	=e�       �	a���Xc�A�5*

loss�:50��       �	�F��Xc�A�5*

loss�ߞ:�%�       �	���Xc�A�5*

lossPa�;_%�!       �	o���Xc�A�5*

loss��:{�       �	w+��Xc�A�5*

lossaA�<�׊       �	�Ε�Xc�A�5*

loss�w�<cdA�       �	u��Xc�A�5*

loss���<���       �	���Xc�A�5*

lossv�<�bM�       �	Yڗ�Xc�A�5*

loss��"<F��       �	�y��Xc�A�5*

loss-�\<��?       �	�!��Xc�A�5*

lossXr�;��N�       �	�Xc�A�5*

loss��<w�r       �	�i��Xc�A�5*

loss5�<��d       �	:��Xc�A�5*

loss��;R���       �	W���Xc�A�5*

lossvg�=剐5       �	QL��Xc�A�5*

lossx�)< �*       �	R��Xc�A�5*

lossԈ*:C3u       �	z���Xc�A�5*

lossV��9����       �	�*��Xc�A�5*

lossX�;�ߛ�       �	?ƞ�Xc�A�5*

loss~�;<�d       �	�i��Xc�A�5*

loss�<�<AKH       �	k��Xc�A�5*

loss��<*l+       �	.���Xc�A�5*

loss�O;?��       �	�G��Xc�A�5*

loss�i�:��+       �	@ݡ�Xc�A�5*

lossݺ�<۪i?       �	)u��Xc�A�5*

lossS�=���       �	� ��Xc�A�5*

lossO��<��qE       �	�ʣ�Xc�A�5*

loss��"<���       �	�i��Xc�A�5*

loss���;3 <S       �	���Xc�A�5*

lossGғ9f6�G       �	���Xc�A�5*

loss)[�<�m9       �	KV��Xc�A�5*

loss4��;��(       �	����Xc�A�5*

lossIN)9�3��       �	ލ��Xc�A�5*

lossϑ9����       �	�4��Xc�A�5*

loss��<�       �	G=��Xc�A�5*

loss_SK<
�Y       �	ө�Xc�A�5*

loss���;!y�@       �	���Xc�A�5*

loss�<u_�       �	۫�Xc�A�5*

loss�@�=>O,       �	Ou��Xc�A�5*

loss��;�Z�2       �	ē��Xc�A�5*

loss��w;Ki*>       �	PS��Xc�A�5*

loss��;��Q�       �	���Xc�A�5*

loss��C=����       �	����Xc�A�5*

loss���:i{�       �	÷��Xc�A�5*

loss���;�)�       �	�Y��Xc�A�5*

loss���<k�8       �	���Xc�A�5*

loss]v�:�V|�       �	����Xc�A�5*

loss�i;+��B       �	�8��Xc�A�5*

loss�m�;c �       �	���Xc�A�5*

loss���:yb �       �	t���Xc�A�5*

loss��L;�-       �	7���Xc�A�5*

loss�;)��.       �	(��Xc�A�6*

loss�<xd�       �	nö�Xc�A�6*

loss�c 9�S��       �	�o��Xc�A�6*

lossi��:̧�:       �	l��Xc�A�6*

loss�E`<�Kr       �	����Xc�A�6*

loss�6�:���       �	q;��Xc�A�6*

loss���9�ce�       �	�ѹ�Xc�A�6*

loss.J�;��       �	険�Xc�A�6*

loss';����       �	1?��Xc�A�6*

loss�B�<�i<       �	,ֻ�Xc�A�6*

loss;�9<`Ms(       �	�w��Xc�A�6*

loss��<J��9       �	$��Xc�A�6*

loss��
;�c�e       �	�Ƚ�Xc�A�6*

lossΔ<�+       �	mo��Xc�A�6*

loss�~�<�׏o       �	�
��Xc�A�6*

loss��<'}�       �	����Xc�A�6*

loss�OV<�v��       �	s���Xc�A�6*

loss���:�6(       �	�5��Xc�A�6*

loss�:�;��6D       �	����Xc�A�6*

loss�Nd;P�/�       �	*���Xc�A�6*

loss�5�;�h�       �	q<��Xc�A�6*

loss˽�;k�G       �	D��Xc�A�6*

loss�3�;��t       �	����Xc�A�6*

loss�=2��@       �	@m��Xc�A�6*

loss$)<*s�        �	C��Xc�A�6*

loss%bW=��       �	ʧ��Xc�A�6*

loss|5�<g       �	�?��Xc�A�6*

loss�f<�4��       �	����Xc�A�6*

loss9O:���       �	|��Xc�A�6*

loss*T;��b       �	���Xc�A�6*

loss$g�9�a��       �	U���Xc�A�6*

loss�Z�:IZ�       �	1���Xc�A�6*

lossFR�:��9�       �	5a��Xc�A�6*

lossx�=���       �	{���Xc�A�6*

loss��!=�K�6       �	a���Xc�A�6*

loss6�T<��       �	hx��Xc�A�6*

loss[�x:.�	q       �	���Xc�A�6*

loss��<w`�       �	����Xc�A�6*

loss��{<n�x�       �	�k��Xc�A�6*

lossi�,8p���       �	���Xc�A�6*

loss���<*��       �	����Xc�A�6*

lossR�:�k��       �	M��Xc�A�6*

loss7�;u�@�       �	����Xc�A�6*

lossq��<N^       �	P���Xc�A�6*

loss�s<��Jn       �	��Xc�A�6*

loss�l;*:k�       �	ٱ��Xc�A�6*

loss�*:j��}       �	F��Xc�A�6*

loss�}�;�²�       �	#���Xc�A�6*

lossN�~;��"B       �	����Xc�A�6*

loss4��<���       �	bH��Xc�A�6*

loss��>;~�V�       �	?���Xc�A�6*

loss�v<Ļ�       �	���Xc�A�6*

loss���;�޵,       �	���Xc�A�6*

loss��t;���       �	_���Xc�A�6*

loss�ke;�3��       �	�U��Xc�A�6*

loss:C=��       �	���Xc�A�6*

loss���;�t       �	����Xc�A�6*

loss}j9��W�       �	�"��Xc�A�6*

lossl�:i&�       �	����Xc�A�6*

loss�?b<_Vj       �	"P��Xc�A�6*

loss�n<��       �	����Xc�A�6*

loss��9q�P�       �	�{��Xc�A�6*

lossT\M:#&N4       �	���Xc�A�6*

lossO'	<��!�       �	q���Xc�A�6*

lossR��:7>       �	M��Xc�A�6*

loss�)=���&       �	����Xc�A�6*

loss�y�;�H�@       �	}y��Xc�A�6*

loss41;�*	[       �	���Xc�A�6*

loss[�'<E*��       �	D���Xc�A�6*

loss11�;#��,       �	�F��Xc�A�6*

lossr��<���       �	����Xc�A�6*

loss�]�=A���       �	�w��Xc�A�6*

loss��;�@J1       �	��Xc�A�6*

loss��n:�K�       �	����Xc�A�6*

lossѕ];��-�       �	c��Xc�A�6*

loss�tG;+��       �	�0��Xc�A�6*

lossD�?;��͋       �	����Xc�A�6*

loss��;T�       �	Dm��Xc�A�6*

loss��:_I       �	�%��Xc�A�6*

lossr�9�[��       �	!\��Xc�A�6*

loss���:*�i�       �	9���Xc�A�6*

loss��Y;���       �	ע��Xc�A�6*

loss�Y�=C�3
       �	PV��Xc�A�6*

loss�k=*gTo       �	����Xc�A�6*

loss�';?��<       �	����Xc�A�6*

loss9u;���       �	 c��Xc�A�6*

lossT:�b       �	����Xc�A�6*

lossN��9,��       �	I��Xc�A�6*

loss_%W<��F�       �	q���Xc�A�6*

loss���:�b0�       �	�z��Xc�A�6*

lossvv[<���       �	m��Xc�A�6*

lossK�=Wcc]       �	����Xc�A�6*

loss�Q�9C�u       �	�X��Xc�A�6*

loss{�%<_Q�       �	����Xc�A�6*

lossSi�;����       �	�}��Xc�A�6*

loss�F;=�Ղ�       �	���Xc�A�6*

loss-�N<�d�       �	���Xc�A�6*

lossx�m;Ih�i       �	�j��Xc�A�6*

lossJb<<�i1�       �	����Xc�A�6*

lossY<�/W       �	X���Xc�A�6*

loss��=5�K       �	�'��Xc�A�6*

loss�C�<Y�q>       �	����Xc�A�6*

lossʣ=�O)       �	|e��Xc�A�6*

loss�[�=C>�       �	z���Xc�A�6*

loss�`�;���_       �	���Xc�A�6*

lossS�9:�.A�       �	fh��Xc�A�6*

loss�Ƌ:�w5       �	���Xc�A�6*

loss��<��.G       �	p���Xc�A�6*

lossߺ�:ރ�x       �	�:��Xc�A�6*

loss�=	:�       �	���Xc�A�6*

loss<<�=��h?       �	wd��Xc�A�6*

loss]Q:��?       �	����Xc�A�6*

lossQ�<�"�       �	����Xc�A�6*

loss���;����       �	�P �Xc�A�6*

loss��=�e��       �	�� �Xc�A�6*

loss��K;q~M       �	���Xc�A�6*

loss���<��8�       �	r1�Xc�A�6*

loss�<3yJ�       �	���Xc�A�6*

loss���;ɞ�q       �	�s�Xc�A�6*

lossO"t=E��       �	��Xc�A�6*

loss���:�5.       �	m �Xc�A�6*

loss�A�:L���       �	3��Xc�A�6*

loss�sD:����       �	�C�Xc�A�6*

lossr�='5       �	���Xc�A�6*

loss��@<�or�       �	�x�Xc�A�6*

loss�c�:�4�g       �	O�Xc�A�6*

loss�"<~A�       �	`>	�Xc�A�6*

loss�n<[��       �	Q�	�Xc�A�6*

loss��=��kQ       �	�u
�Xc�A�7*

loss!�<�e       �	
�Xc�A�7*

lossua=:�:(       �	���Xc�A�7*

loss�?�;g
e       �	�^�Xc�A�7*

lossf�>�*p       �	���Xc�A�7*

loss8��<տ>y       �	1��Xc�A�7*

lossϒV;�z�       �	�*�Xc�A�7*

loss#R$<���       �	߿�Xc�A�7*

loss�8*;�S?	       �	�T�Xc�A�7*

lossl�!<X�>�       �	���Xc�A�7*

loss��-;jii�       �	��Xc�A�7*

lossl"�<ǀc       �	~�Xc�A�7*

lossX��<�Xq       �	\��Xc�A�7*

loss��f;Gv�       �	�s�Xc�A�7*

loss�]�:w}�       �	S%�Xc�A�7*

loss�9�=���       �	���Xc�A�7*

loss:2�<0�T       �	���Xc�A�7*

loss3i�;PL��       �	�m�Xc�A�7*

lossV<�#       �	�O�Xc�A�7*

loss�:(;'�-       �	7��Xc�A�7*

loss?��;���       �	-y�Xc�A�7*

loss�ҭ<�G,       �	�Xc�A�7*

loss��=#�/H       �	���Xc�A�7*

loss��t<v`       �	�E�Xc�A�7*

loss��=��Y       �	/��Xc�A�7*

lossE��<�=3       �	�q�Xc�A�7*

loss�b^;�{M�       �	��Xc�A�7*

loss<'V:X��       �	��Xc�A�7*

loss�<�W��       �	�Y�Xc�A�7*

loss�n�<w���       �	d��Xc�A�7*

lossh�09���       �	E�Xc�A�7*

loss�� =L+       �	~�Xc�A�7*

loss[��:̼Z       �	O��Xc�A�7*

loss���;,1�       �	�� �Xc�A�7*

loss�|=����       �	9!�Xc�A�7*

lossל;���       �	��!�Xc�A�7*

loss�B�:���       �	N|"�Xc�A�7*

lossq>9岴�       �	�#�Xc�A�7*

loss��;���       �	��#�Xc�A�7*

loss��B;=8       �	�Q$�Xc�A�7*

lossh)p=�'       �	��$�Xc�A�7*

loss!�=�z��       �	�%�Xc�A�7*

loss���<F��$       �	fO&�Xc�A�7*

losszx;�J+       �	��&�Xc�A�7*

loss�f�<���C       �	�'�Xc�A�7*

loss���<��       �	&(�Xc�A�7*

lossp�<d�       �	�<)�Xc�A�7*

loss��f<0��       �	t�)�Xc�A�7*

loss�
(=-o�       �	U�*�Xc�A�7*

loss#;����       �	��+�Xc�A�7*

loss<%<u��c       �	O;,�Xc�A�7*

loss���;d���       �	��,�Xc�A�7*

loss�c;{�Z�       �	ׄ-�Xc�A�7*

loss�3<+r�       �	�!.�Xc�A�7*

lossx:�<�|�       �	��.�Xc�A�7*

loss���9v��       �	�S/�Xc�A�7*

loss'=�/?       �	�0�Xc�A�7*

loss�$�;
J��       �	�1�Xc�A�7*

loss�<�ņ�       �	\<2�Xc�A�7*

lossx��=GH_       �	*�2�Xc�A�7*

loss�kd;�ׄ�       �	R�3�Xc�A�7*

losse�<<\�}       �	K#4�Xc�A�7*

loss�
;��i�       �	�4�Xc�A�7*

loss`�x;S.��       �	V5�Xc�A�7*

lossV�>;]���       �	��5�Xc�A�7*

loss!�<�t�       �	�6�Xc�A�7*

loss�`�<��W�       �	�87�Xc�A�7*

loss��\;o�vB       �	f�7�Xc�A�7*

loss.?];�(�       �	I9�Xc�A�7*

loss���;�D��       �	X:�Xc�A�7*

loss3��;�g-       �	��:�Xc�A�7*

loss�)k:��[%       �	�B;�Xc�A�7*

losss�=�vZ       �	��;�Xc�A�7*

lossgo�<��ؚ       �	�s<�Xc�A�7*

losse��:U7X&       �	�=�Xc�A�7*

lossaĒ;�a�       �	��=�Xc�A�7*

losst;e��       �	@1>�Xc�A�7*

lossZ(�<�C��       �	��>�Xc�A�7*

losslA�;r*gK       �	@l?�Xc�A�7*

loss荡9|y�	       �	W@�Xc�A�7*

loss�w�9D�k       �	q�@�Xc�A�7*

loss/`z:L�F       �	�>A�Xc�A�7*

lossa��;��#P       �	�>B�Xc�A�7*

lossJk�9X\&       �	��B�Xc�A�7*

loss�+:��a�       �	K�C�Xc�A�7*

loss�ޚ;+?=�       �	 D�Xc�A�7*

loss�3
;����       �	V�D�Xc�A�7*

loss��;Rh�       �	�JE�Xc�A�7*

losso��;���       �	��E�Xc�A�7*

loss�*<��`       �	��F�Xc�A�7*

loss���;=;�5       �	a6G�Xc�A�7*

lossj�e:��"       �	1�G�Xc�A�7*

loss�
�;*�+P       �	3jH�Xc�A�7*

lossS��:/��       �	��I�Xc�A�7*

lossH��9����       �	�J�Xc�A�7*

lossϙ�;A=�       �	[K�Xc�A�7*

loss.N.;)e�        �	��L�Xc�A�7*

loss&m%;�2�       �	�;M�Xc�A�7*

loss�B�;^ň       �	�M�Xc�A�7*

loss��R8�S@�       �	�jN�Xc�A�7*

loss|-A<�0QZ       �	�O�Xc�A�7*

loss��7�6YU       �	�O�Xc�A�7*

loss�9�e*l       �	�7P�Xc�A�7*

losscLi:��Ϝ       �	%�P�Xc�A�7*

loss��;+S��       �	<fQ�Xc�A�7*

loss��:�h{�       �	
R�Xc�A�7*

loss���<����       �	��R�Xc�A�7*

loss���7���a       �	�NS�Xc�A�7*

loss$�B=�eb`       �	��S�Xc�A�7*

loss�i;���H       �	��T�Xc�A�7*

lossF;�:1b�       �		5U�Xc�A�7*

lossa�=:z�R       �	��U�Xc�A�7*

loss�$@=3�X9       �	M�V�Xc�A�7*

loss ��=���       �	V)W�Xc�A�7*

loss;.�<a�$�       �	s�W�Xc�A�7*

losss@r:yC       �	S]X�Xc�A�7*

loss�&<3aU       �	��X�Xc�A�7*

loss=��:�Z��       �	ѓY�Xc�A�7*

lossê;p��       �	i8Z�Xc�A�7*

loss�``;2c]       �	��Z�Xc�A�7*

loss�q|:��?       �	!t[�Xc�A�7*

loss8�:=Bg�H       �	�>\�Xc�A�7*

loss��C<S�@�       �	��\�Xc�A�7*

loss}��<���m       �	9{]�Xc�A�7*

loss�&�<��;       �	�^�Xc�A�7*

loss��<.�L�       �	T�^�Xc�A�7*

loss&�Y;�ۄ       �	�_�Xc�A�7*

losss�A:�f��       �	1~`�Xc�A�7*

lossf(<N5�E       �	"a�Xc�A�8*

loss��*<�̍       �	d�a�Xc�A�8*

loss�'�9�H��       �	O[b�Xc�A�8*

loss�:b5��       �	V�b�Xc�A�8*

loss��<�)       �	;�c�Xc�A�8*

lossJ�@;y�w       �	�Hd�Xc�A�8*

loss�!�;�Z       �	(�d�Xc�A�8*

lossK)�:�V�       �	=�e�Xc�A�8*

loss���;/�       �	<f�Xc�A�8*

loss�.:�(�       �	w�f�Xc�A�8*

loss��~<;
��       �	Tqg�Xc�A�8*

loss���<�       �	<h�Xc�A�8*

loss�DU:���       �	��h�Xc�A�8*

loss�&�<H\��       �	��i�Xc�A�8*

loss}!�<�`I�       �	�_j�Xc�A�8*

loss�՜;�7��       �	�ek�Xc�A�8*

loss
��;��       �	�l�Xc�A�8*

loss��L<���       �	{�m�Xc�A�8*

loss�!=��U�       �	�Pn�Xc�A�8*

loss���;�=:s       �	^�n�Xc�A�8*

lossA��:$�^�       �	W�o�Xc�A�8*

loss�m
=nt3       �	�3p�Xc�A�8*

lossS6	:��2�       �	�Nq�Xc�A�8*

loss�p�:|Xڐ       �	Or�Xc�A�8*

loss:�;[c�       �	C s�Xc�A�8*

loss^@<���       �	~�s�Xc�A�8*

loss�z:l�1�       �	��t�Xc�A�8*

lossK�;I�=�       �	�u�Xc�A�8*

lossql�;��$�       �	�}v�Xc�A�8*

loss�*�9A�       �	S w�Xc�A�8*

loss�$�<���       �	��w�Xc�A�8*

loss�e�;L�Na       �	C�x�Xc�A�8*

loss}	�9�^��       �	>$y�Xc�A�8*

lossH�4<��