       �K"	  ��Xc�Abrain.Event:2�z�2�     �^�	uȈ�Xc�A"��
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
:@*
seed2��
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
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
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
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
seed2��*
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
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:@@
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
valueB"      @   @   *
_output_shapes
:*
dtype0
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

_
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
:
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
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
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
dropout_1/cond/dropout/divRealDivdropout_1/cond/mul dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
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
valueB:*
_output_shapes
:*
dtype0
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
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
valueB" d  �   *
dtype0*
_output_shapes
:
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
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2��r*
T0*
seed���)*
dtype0
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
VariableV2*!
_output_shapes
:���*
	container *
shape:���*
dtype0*
shared_name 
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
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
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
seed2���
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
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*$
_class
loc:@activation_3/Relu*<
_output_shapes*
(:����������:����������
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
 *̈́U>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2���*
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
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��3*
T0*
seed���)*
dtype0
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
valueB:*
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
�
6sequential_1/dropout_2/cond/dropout/random_uniform/subSub6sequential_1/dropout_2/cond/dropout/random_uniform/max6sequential_1/dropout_2/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
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
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

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
value	B :*
dtype0*
_output_shapes
: 
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
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
	AssignAdd	AssignAddnum_instConst_1*
_class
loc:@num_inst*
_output_shapes
: *
T0*
use_locking( 
~
AssignAdd_1	AssignAddnum_correctSum*
_class
loc:@num_correct*
_output_shapes
: *
T0*
use_locking( 
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
 *���.*
dtype0*
_output_shapes
: 
A
addAddnum_inst/readadd/y*
T0*
_output_shapes
: 
F
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
L
div_1/yConst*
valueB
 *  �A*
_output_shapes
: *
dtype0
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
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0
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
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
_output_shapes
:*
Index0*
T0
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
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
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
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
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
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
 *  �?*
_output_shapes
: *
dtype0
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
 *    *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
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
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
_output_shapes
: *
T0
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
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
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
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
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
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
valueB:*
_output_shapes
:*
dtype0
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
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
valueB:*
dtype0*
_output_shapes
:
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
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
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
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
valueB: *
dtype0*
_output_shapes
:
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
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
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
valueB *
_output_shapes
: *
dtype0
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
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
transpose_b( *(
_output_shapes
:����������*
transpose_a(*
T0
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
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
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
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*/
_output_shapes
:���������@*
T0
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
gradients/Switch_3Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*
T0
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
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
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
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
n
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
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
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
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
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam
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
valueB���*    *!
_output_shapes
:���*
dtype0
�
dense_1/kernel/Adam_1
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
 *w�+2*
_output_shapes
: *
dtype0
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
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
 Bloss*
_output_shapes
: *
dtype0
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "������     o���	���Xc�AJ��
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
:@*
seed2��
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
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
:@*
T0
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
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
valueB"      *
_output_shapes
:*
dtype0
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
:@@*
seed2��
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
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
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
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
:
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
 *  @?*
dtype0*
_output_shapes
: 
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
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2���*
T0*
seed���)*
dtype0
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
�
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*/
_output_shapes
:���������@*
T0
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
 *�3z�*
_output_shapes
: *
dtype0
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
:���*
seed2��r
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
valueB�*    *
dtype0*
_output_shapes	
:�
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
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
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
 *̈́U>*
dtype0*
_output_shapes
: 
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
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
T0*
data_formatNHWC
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��3*
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
valueB:*
dtype0*
_output_shapes
:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
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
 *   ?*
dtype0*
_output_shapes
: 
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
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
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
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
ToFloatCastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
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
AssignAdd_1	AssignAddnum_correctSum*
_class
loc:@num_correct*
_output_shapes
: *
T0*
use_locking( 
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
 *  �A*
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
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
out_type0*
_output_shapes
:*
T0
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
valueB:*
dtype0*
_output_shapes
:
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
value	B : *
_output_shapes
: *
dtype0
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
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
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
value	B : *
dtype0*
_output_shapes
: 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
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
 *  �?*
dtype0*
_output_shapes
: 
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
value	B :*
dtype0*
_output_shapes
: 
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
 *  �?*
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
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
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 
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
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
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
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
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
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�
*
T0
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
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
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
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0
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
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
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
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
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
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
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
 *    *
_output_shapes
: *
dtype0
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*
Tshape0*/
_output_shapes
:���������@*
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
valueB *
_output_shapes
: *
dtype0
�
Dgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/sequential_1/dropout_1/cond/mul_grad/Shape6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/sequential_1/dropout_1/cond/mul_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_1/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*/
_output_shapes
:���������@*
T0*
N
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
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
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@@*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
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
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
T0
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
out_type0*
_output_shapes
:
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*%
valueB"         @   *
_output_shapes
:*
dtype0
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:@*
use_cudnn_on_gpu(
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
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
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
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0
�
beta2_power
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
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
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
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
valueB@*    *
dtype0*
_output_shapes
:@
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
valueB@@*    *
dtype0*&
_output_shapes
:@@
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
valueB@*    *
dtype0*
_output_shapes
:@
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
valueB�*    *
_output_shapes	
:�*
dtype0
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
_output_shapes
:	�
*
dtype0
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
*    *
_output_shapes
:
*
dtype0
�
dense_2/bias/Adam
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
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

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
 *w�+2*
_output_shapes
: *
dtype0
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
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_2/bias*
_output_shapes
:
*
T0*
use_locking( 
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
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"0
losses&
$
"softmax_cross_entropy_loss/value:0��fy       ��-	:���Xc�A*

lossM@}e�       ��-	�[��Xc�A*

loss2�@�A       ��-	���Xc�A*

lossU�@���       ��-	;ö�Xc�A*

loss�@��7       ��-	Yi��Xc�A*

losstD@b��       ��-	���Xc�A*

loss4u�?]�       ��-	����Xc�A*

loss�a�?L�       ��-	�]��Xc�A*

lossa�?�U�       ��-	���Xc�A	*

loss���?�q�       ��-	���Xc�A
*

loss��?b��       ��-	�G��Xc�A*

loss)X�?Y�P       ��-	���Xc�A*

loss��x?V��       ��-	:���Xc�A*

loss��?/m,w       ��-	�,��Xc�A*

lossTD}?�.�Q       ��-	�ɽ�Xc�A*

loss��|?�$m�       ��-	|d��Xc�A*

loss��B?��:�       ��-	���Xc�A*

lossAW?ޗ��       ��-	禿�Xc�A*

lossq0?5�r$       ��-	PU��Xc�A*

loss
�r?2v�e       ��-	_���Xc�A*

loss���?cH"�       ��-	^���Xc�A*

loss��V?��	       ��-	9��Xc�A*

loss��T?V�N�       ��-	o���Xc�A*

loss�t}?+"T       ��-	�l��Xc�A*

loss@�b?��aC       ��-	
��Xc�A*

loss��+?lӘ       ��-	����Xc�A*

loss�U?cT�       ��-	�M��Xc�A*

lossZ=A?&�9�       ��-	{���Xc�A*

lossj-k?����       ��-	*���Xc�A*

losss�`?�L�       ��-	,��Xc�A*

loss�<??e�F       ��-	& ��Xc�A*

loss��?�"�       ��-	؛��Xc�A *

loss��2?c�B�       ��-	N��Xc�A!*

loss&v?����       ��-	J���Xc�A"*

lossjm?��af       ��-	z���Xc�A#*

loss28?#�       ��-	�`��Xc�A$*

loss1�?m��       ��-	�J��Xc�A%*

loss��>?G��       ��-	����Xc�A&*

loss��^?�l7?       ��-	|��Xc�A'*

loss�S?1^�K       ��-	A��Xc�A(*

loss�|?���       ��-	����Xc�A)*

lossԩ�>��       ��-	����Xc�A**

lossAN?�X3�       ��-	A~��Xc�A+*

lossu�?��^I       ��-	s��Xc�A,*

loss_�?�*+       ��-	�/��Xc�A-*

lossɥ?���       ��-	����Xc�A.*

lossz�?V�       ��-	�p��Xc�A/*

lossA	�>�Wo~       ��-	)��Xc�A0*

loss�W?��       ��-	����Xc�A1*

lossȾ>6�E       ��-	�1��Xc�A2*

loss}!�>�T1       ��-	���Xc�A3*

loss���>%~       ��-	wf��Xc�A4*

loss��>w7U       ��-	d��Xc�A5*

loss�?).�       ��-	���Xc�A6*

loss��>2�A       ��-	�k��Xc�A7*

lossm �>�#�       ��-	/��Xc�A8*

loss��>�T�       ��-	����Xc�A9*

lossܩ�>�j        ��-	�R��Xc�A:*

lossL%?,��       ��-	����Xc�A;*

loss���>ZO�D       ��-	S���Xc�A<*

losst��>�<w�       ��-	>?��Xc�A=*

loss�>�>t�+3       ��-	����Xc�A>*

loss�j�>��S�       ��-	[}��Xc�A?*

loss���>>�2�       ��-	i��Xc�A@*

lossr�>��M}       ��-	���Xc�AA*

loss�P�>�[��       ��-	�L��Xc�AB*

loss.#,?+��       ��-	���Xc�AC*

loss�C	?��,       ��-	B|��Xc�AD*

loss�'?g�9       ��-	���Xc�AE*

loss�y�>���@       ��-	ѯ��Xc�AF*

loss)�>16�       ��-	ZH��Xc�AG*

loss$��> �v�       ��-	m���Xc�AH*

lossS�>�f+}       ��-	}��Xc�AI*

loss�5?Y� S       ��-	���Xc�AJ*

loss��?�y_Q       ��-	����Xc�AK*

loss]6?�0��       ��-	ZJ��Xc�AL*

loss�#?v@#|       ��-	���Xc�AM*

loss�yG?�Y5       ��-	k���Xc�AN*

loss���>��c       ��-	�*��Xc�AO*

loss��>�ڶ       ��-	���Xc�AP*

loss��?�3�       ��-	���Xc�AQ*

loss4�?4bى       ��-	����Xc�AR*

lossS�?���?       ��-	l_��Xc�AS*

loss���>v)�       ��-	���Xc�AT*

loss��>&(�       ��-	����Xc�AU*

lossO�>�V1�       ��-	VI��Xc�AV*

loss���>�U�       ��-	l���Xc�AW*

lossI�>.O��       ��-	H���Xc�AX*

loss,��>�o��       ��-	'��Xc�AY*

loss��>q,�       ��-	&���Xc�AZ*

lossH��>�Ž�       ��-	Lm��Xc�A[*

loss,Ʀ>o��E       ��-	����Xc�A\*

loss��>	��       ��-	���Xc�A]*

lossW�>�!d�       ��-	ė��Xc�A^*

loss,y?۝)       ��-	�0��Xc�A_*

lossu��>��       ��-	���Xc�A`*

lossh�>DoW�       ��-	�j��Xc�Aa*

loss�?���       ��-	���Xc�Ab*

loss�e[?�lߏ       ��-	F���Xc�Ac*

lossXi�>�E�       ��-	<2��Xc�Ad*

lossr �>�-       ��-	&���Xc�Ae*

loss�U>Βj�       ��-	e���Xc�Af*

loss8��>o���       ��-	<0��Xc�Ag*

loss�=�>�79       ��-	!���Xc�Ah*

lossf��>�.%       ��-	?���Xc�Ai*

loss#��>U<��       ��-	�]��Xc�Aj*

loss\i�>��V�       ��-	���Xc�Ak*

lossQ�>̻�e       ��-	���Xc�Al*

loss��,?�V��       ��-	Fz��Xc�Am*

lossR,?�}zG       ��-	� �Xc�An*

loss�z?m�|       ��-	��Xc�Ao*

loss���>3n_       ��-	���Xc�Ap*

loss���>���       ��-	J�Xc�Aq*

loss�K|>Q�ʏ       ��-	%��Xc�Ar*

loss���>��sj       ��-	C��Xc�As*

loss��>^��R       ��-	,�Xc�At*

loss�7�>qť�       ��-	i��Xc�Au*

loss���>o��       ��-	�e�Xc�Av*

lossT��>#j>O       ��-	<�Xc�Aw*

lossu�>Y�"[       ��-	���Xc�Ax*

loss�I�>\�o�       ��-	�G�Xc�Ay*

loss}	�>H"�       ��-	��Xc�Az*

lossE�}>(bcv       ��-	{��Xc�A{*

loss�b�>�0�       ��-	Lq	�Xc�A|*

lossQ>�{�v       ��-	�
�Xc�A}*

loss�!�>㚜�       ��-	��
�Xc�A~*

loss�$�>4��       ��-	d��Xc�A*

lossA~q>6=W       �	�F�Xc�A�*

loss	��>�v��       �	���Xc�A�*

lossi�>M�v       �	z��Xc�A�*

loss�mQ>�|�       �	/��Xc�A�*

loss菭>�       �	�(�Xc�A�*

loss�o>�f�       �	���Xc�A�*

loss���=�1T�       �	�j�Xc�A�*

loss�Gm>��1�       �	#�Xc�A�*

loss<��>�1�Z       �	���Xc�A�*

loss}M�>���       �	�R�Xc�A�*

loss\->B	�       �	���Xc�A�*

loss��S>ĉ�       �	C��Xc�A�*

lossͱ6>�T�       �	80�Xc�A�*

loss|��=�s:       �	���Xc�A�*

lossh0�>�%�w       �	��Xc�A�*

loss��>��       �	~�Xc�A�*

lossZc>lV	       �	��Xc�A�*

loss�W>�X��       �	z��Xc�A�*

loss6[>f9��       �	5@�Xc�A�*

loss�i>!�m       �	���Xc�A�*

loss��=*e�x       �	ly�Xc�A�*

loss؆>�.\�       �	z4�Xc�A�*

lossr�y>G��9       �	���Xc�A�*

loss��i>0*       �	�i�Xc�A�*

loss8��>��Uw       �	��Xc�A�*

loss?7�>R}#X       �	r��Xc�A�*

loss���>����       �	C=�Xc�A�*

loss=ۆ>���6       �	s��Xc�A�*

loss�b�=3�sk       �	�y�Xc�A�*

loss�>�>M�{       �	�Xc�A�*

loss��>��       �	���Xc�A�*

loss;?�>����       �	�A �Xc�A�*

loss�_�>��	j       �	�� �Xc�A�*

loss$h�>��Q       �	Lo!�Xc�A�*

lossV7�>ѕ��       �	�"�Xc�A�*

loss��>���        �	|�"�Xc�A�*

loss}>�\��       �	�0#�Xc�A�*

loss��>PKi`       �	y�#�Xc�A�*

loss{�.>�\�       �	�a$�Xc�A�*

lossÐ&>O{G       �	D�$�Xc�A�*

loss��H>�0ޖ       �	��%�Xc�A�*

loss�$)>�S{`       �	�(&�Xc�A�*

loss��>A�l       �	˽&�Xc�A�*

loss$:`>kc\�       �	BZ'�Xc�A�*

losse��=T�W3       �	x�'�Xc�A�*

loss@6�>i|�J       �	��(�Xc�A�*

loss8�_>�ڷ]       �	$)�Xc�A�*

loss���=�f�       �	Q�)�Xc�A�*

loss��>�q�       �	.U*�Xc�A�*

lossc?v���       �	��*�Xc�A�*

loss$�>���       �	H�+�Xc�A�*

loss�Z>"n�       �	,�Xc�A�*

loss��.>���       �	��,�Xc�A�*

loss�>FJ�       �	�Z-�Xc�A�*

loss�ʏ>� o       �	�-�Xc�A�*

lossk�>��P�       �	�.�Xc�A�*

loss���=��S�       �	?6/�Xc�A�*

lossR|>�n\       �	��/�Xc�A�*

loss�0�>����       �	:z0�Xc�A�*

loss��I>Xo��       �	�1�Xc�A�*

losso&�>��'^       �	c�1�Xc�A�*

loss�g>'JZ       �	U2�Xc�A�*

loss�=�>�b�_       �	��2�Xc�A�*

loss�>ߒR       �	��3�Xc�A�*

loss�S>�Y��       �	�4�Xc�A�*

loss��8>�<��       �	h�4�Xc�A�*

lossl�>�h�       �	�b5�Xc�A�*

loss
�;>+a�e       �	��5�Xc�A�*

loss4ژ>�Z6       �	��6�Xc�A�*

loss6��>�6)�       �	�W7�Xc�A�*

loss}�u>k�_       �	�7�Xc�A�*

loss�.�>ۉ}�       �	ʨ8�Xc�A�*

lossN/&>��/�       �	�G9�Xc�A�*

loss�+Q>���       �	��9�Xc�A�*

loss��=�ʚ�       �	�:�Xc�A�*

loss|�)>ȍ��       �	'.;�Xc�A�*

lossoh>\Ԅ�       �	��;�Xc�A�*

lossq0>�U��       �	n<�Xc�A�*

loss4N�=�Aj       �	��=�Xc�A�*

loss O5>�m�       �	�&>�Xc�A�*

lossM��=����       �	��>�Xc�A�*

loss��'>��       �	�^?�Xc�A�*

loss?؀>ŭ�&       �	��?�Xc�A�*

loss�S<>§y�       �	�@�Xc�A�*

loss�'.>��iv       �	�=A�Xc�A�*

loss.�=B!r       �	P�A�Xc�A�*

losso��=Ӻ��       �	{B�Xc�A�*

losssj�>��4       �	=C�Xc�A�*

loss��>���       �	��C�Xc�A�*

loss�zj>9nK       �	)>D�Xc�A�*

lossq@�>����       �	��D�Xc�A�*

loss��>2d       �	�iE�Xc�A�*

loss�E�>���       �	�F�Xc�A�*

loss[��=Uc�i       �	�F�Xc�A�*

loss�>d�       �	�PG�Xc�A�*

lossd �>��&�       �	V�G�Xc�A�*

loss ��>M£g       �	S�H�Xc�A�*

lossF�=�XO�       �	)?I�Xc�A�*

losst�->.�-�       �	u�I�Xc�A�*

loss�>7>�f       �	ӥJ�Xc�A�*

loss��N>o�8�       �	�IK�Xc�A�*

loss�"&>��'u       �	A�K�Xc�A�*

loss�\�>Ɠ��       �	��L�Xc�A�*

lossJa�=���'       �	�>M�Xc�A�*

loss�p>�̽=       �	}�M�Xc�A�*

losst3K>��R       �	A�N�Xc�A�*

lossԢF>e�×       �	�MO�Xc�A�*

loss,��>�*��       �	E�O�Xc�A�*

loss�2>PŌ�       �	˻P�Xc�A�*

loss=X&>=!�       �	�^Q�Xc�A�*

loss_��>�j�       �	��Q�Xc�A�*

lossہ >���:       �	�R�Xc�A�*

loss�G>�R��       �	FBS�Xc�A�*

loss�<�=���       �	�S�Xc�A�*

loss_�c>�iZ       �	�T�Xc�A�*

loss东>��|       �	�U�Xc�A�*

loss2�]>�}�       �	s�U�Xc�A�*

lossH��>@ӭ�       �	�QV�Xc�A�*

loss!�>����       �	-�V�Xc�A�*

loss&� >��e�       �	��W�Xc�A�*

lossE>׼@�       �	�vX�Xc�A�*

lossB�>���       �	kY�Xc�A�*

loss��g>��       �	`�Y�Xc�A�*

lossxhW>A��       �	�GZ�Xc�A�*

loss!,6>@��w       �	P�Z�Xc�A�*

loss��Z>.
�\       �	}x[�Xc�A�*

loss$=�=�aW       �	�\�Xc�A�*

loss��=�#��       �	��\�Xc�A�*

loss�Jo>��!-       �	@]�Xc�A�*

loss��W>p�YI       �	��^�Xc�A�*

loss@��>�!��       �	�_�Xc�A�*

loss3(>,pNu       �	$`�Xc�A�*

loss2��>�o�\       �	��`�Xc�A�*

loss9.>4VD       �	�Ma�Xc�A�*

loss�*�=�6�        �	�a�Xc�A�*

loss�w�>#�I9       �	�zb�Xc�A�*

loss.�=>= t�       �	�c�Xc�A�*

loss�B�>��KI       �	İc�Xc�A�*

loss��=B �       �	�Od�Xc�A�*

loss��J>,���       �	��d�Xc�A�*

lossHw�>�}1u       �	χe�Xc�A�*

lossR��>B��       �	. f�Xc�A�*

lossM4O>���       �	B�f�Xc�A�*

loss�'(>��%I       �	Lg�Xc�A�*

loss��>�auE       �	��g�Xc�A�*

loss�G>^���       �	��h�Xc�A�*

loss��)>y?�P       �	i�Xc�A�*

loss��>"�@�       �	�i�Xc�A�*

losse�|>��k[       �	`j�Xc�A�*

lossA�5>�8'�       �	�j�Xc�A�*

loss��4>�<�D       �	G�k�Xc�A�*

loss&-O>5]n       �	P5l�Xc�A�*

loss�>�s��       �	��l�Xc�A�*

loss-b�=wH
e       �	�bm�Xc�A�*

loss��6>�X�_       �	Xn�Xc�A�*

lossRW]>2���       �	��n�Xc�A�*

loss%��=�;�       �	�Uo�Xc�A�*

lossi��>�y��       �	��o�Xc�A�*

loss�5*>�}P       �	@�p�Xc�A�*

loss�h>���       �	E,q�Xc�A�*

loss�B�=qb�       �	r�q�Xc�A�*

loss��>՝�       �	Rdr�Xc�A�*

loss��=%3�u       �	Ls�Xc�A�*

lossTT�=)j�       �	��s�Xc�A�*

loss��">��       �	Vt�Xc�A�*

loss�K�=T���       �	�u�Xc�A�*

lossf��= �       �	:v�Xc�A�*

loss}>q�U<       �	{�v�Xc�A�*

loss{,>|��       �	�vw�Xc�A�*

lossw$0>X�       �	;�x�Xc�A�*

lossN�%>Pn        �	��y�Xc�A�*

lossl)>5�N7       �	�ez�Xc�A�*

loss�!;>41^       �	E{�Xc�A�*

loss��>�ٕ5       �	f�{�Xc�A�*

loss ��=a�_       �	�a|�Xc�A�*

loss��>���       �	�}�Xc�A�*

lossj!>�L
�       �	e�}�Xc�A�*

loss��=��8�       �	$B~�Xc�A�*

loss��a>�cUZ       �	��~�Xc�A�*

loss�?�=��r\       �	A}�Xc�A�*

loss�K>8���       �	]��Xc�A�*

loss[�=��AX       �	����Xc�A�*

loss�iu>��5       �	�_��Xc�A�*

loss��=�N�       �	����Xc�A�*

loss��>�R�       �	����Xc�A�*

loss�\>Ü�u       �	�*��Xc�A�*

loss���=M�.)       �	y̓�Xc�A�*

loss�U�=�
�       �	,c��Xc�A�*

lossc�q>��*D       �	3��Xc�A�*

loss �3>��yS       �	ʅ�Xc�A�*

lossv��=�$��       �	=d��Xc�A�*

loss�*>��0�       �	���Xc�A�*

loss�;>e�@7       �	�Xc�A�*

loss>(�=o�.�       �	�O��Xc�A�*

loss��j> |�       �	5��Xc�A�*

loss��4>{�`�       �	M���Xc�A�*

loss=y%>M=��       �	¤��Xc�A�*

loss-��>~��(       �	�H��Xc�A�*

loss���>+nL�       �	���Xc�A�*

loss_zj>�w��       �	����Xc�A�*

loss�]�=V)�l       �	���Xc�A�*

loss{)j>���       �	!���Xc�A�*

lossF>0vEl       �	J��Xc�A�*

loss��.>�ѳ�       �	3ߎ�Xc�A�*

lossŉ�=���       �	�x��Xc�A�*

losso�?>��h       �	��Xc�A�*

loss�\>]Y�T       �	[���Xc�A�*

loss�/X>o�%       �	�R��Xc�A�*

loss�p�=�!6(       �	`��Xc�A�*

loss��R>~*`a       �	���Xc�A�*

loss��;>q{�       �	J���Xc�A�*

loss�>&���       �	p��Xc�A�*

loss޵�=��       �	���Xc�A�*

loss�)>V�q@       �	���Xc�A�*

loss%��=��       �	����Xc�A�*

loss�c0>Z��       �	F|��Xc�A�*

loss��
>W�1       �	g���Xc�A�*

loss�?�>���       �	�Q��Xc�A�*

lossr�!>���       �	���Xc�A�*

loss&�=|���       �	����Xc�A�*

loss��>A�k       �	l>��Xc�A�*

loss_t+>��Wp       �	�֛�Xc�A�*

loss��$>Z-��       �	�u��Xc�A�*

loss���=�1�       �	
��Xc�A�*

loss,t�=+,�g       �	O���Xc�A�*

lossF�	>����       �	�G��Xc�A�*

loss�H#>r+�       �	�ߞ�Xc�A�*

lossS�,>B���       �	�t��Xc�A�*

lossiC*=}�B       �	��Xc�A�*

loss(�>T}#�       �	z���Xc�A�*

loss��>?��       �	�8��Xc�A�*

loss��4>�=H�       �	͡�Xc�A�*

lossT��= o��       �	^i��Xc�A�*

loss�>�W��       �	F?��Xc�A�*

loss��=���Z       �	;��Xc�A�*

loss�3�=p}%W       �	㈤�Xc�A�*

loss܋�>����       �	S#��Xc�A�*

lossw�v>Kz��       �	ȥ�Xc�A�*

loss F8>B��       �	�r��Xc�A�*

loss��=ρ.:       �	3��Xc�A�*

lossh�=2=�9       �	��Xc�A�*

loss���=�?�>       �	e��Xc�A�*

loss ��=SQW�       �	���Xc�A�*

loss�O>�\8       �	����Xc�A�*

loss��>�3�.       �	�W��Xc�A�*

loss�K>S`�Q       �	I���Xc�A�*

loss���=ngE�       �	Ֆ��Xc�A�*

loss=!^>���       �	P6��Xc�A�*

loss��>����       �	(Ԭ�Xc�A�*

loss��e>l�q�       �	x��Xc�A�*

loss��>:��       �	���Xc�A�*

loss���=��       �	����Xc�A�*

lossA?O>~=^       �	�@��Xc�A�*

lossJ�F>cl"�       �	�ӯ�Xc�A�*

loss@�&>J-�       �	�i��Xc�A�*

lossJg2>��A       �	����Xc�A�*

lossɕ�=���2       �	����Xc�A�*

loss�>�=>��       �	[(��Xc�A�*

loss|:�=]4�       �	ܲ�Xc�A�*

loss��=�P�V       �	�r��Xc�A�*

lossJ��=��(        �	�	��Xc�A�*

loss2�=Ц�       �	b���Xc�A�*

lossH�Q>���       �	kD��Xc�A�*

loss�v>0� �       �	���Xc�A�*

lossf+�=�}��       �		���Xc�A�*

loss�`:>��VQ       �	ϣ��Xc�A�*

lossϩ4>J�-       �	����Xc�A�*

loss�>T:�       �	�ʹ�Xc�A�*

lossF��>��       �	U���Xc�A�*

loss�u,=�tZO       �	Ü��Xc�A�*

loss$ʳ=�ǫ�       �	/O��Xc�A�*

lossN�=�g       �	_��Xc�A�*

loss� �>7�X�       �	I��Xc�A�*

loss���=Ʌ�       �	�O��Xc�A�*

loss�Gk>I���       �	&s��Xc�A�*

loss��W>tठ       �	�
��Xc�A�*

loss,v�=KB�$       �	)���Xc�A�*

loss�;�=��M�       �	�A��Xc�A�*

lossx/X>�t�*       �	���Xc�A�*

loss��>���       �	���Xc�A�*

loss}tS>���       �	u��Xc�A�*

loss�+P>]B��       �	
���Xc�A�*

lossTK�>d�kd       �	&W��Xc�A�*

loss��>2%�        �	l���Xc�A�*

lossWug>/��G       �	����Xc�A�*

loss�>��       �	�-��Xc�A�*

loss�>��ҁ       �		���Xc�A�*

loss��;>�
-       �	Mh��Xc�A�*

loss�{�>�l8       �	��Xc�A�*

loss(!=����       �	���Xc�A�*

loss���=��u�       �	T6��Xc�A�*

loss��M>�_�E       �	����Xc�A�*

loss�+*>��:�       �	�a��Xc�A�*

loss,��=]���       �	a���Xc�A�*

loss1�=M���       �	C���Xc�A�*

lossHb�=��P&       �	�$��Xc�A�*

loss.�>���       �	u���Xc�A�*

loss�q>驗       �	d��Xc�A�*

loss�݃>y���       �	���Xc�A�*

loss�.>�5��       �	�J��Xc�A�*

loss@�?>��q�       �	Y���Xc�A�*

lossmH�=����       �	w��Xc�A�*

loss��>��?       �	$	��Xc�A�*

lossʈL>}��/       �	T���Xc�A�*

lossjy�=��C�       �	�]��Xc�A�*

lossE��=�@�m       �	����Xc�A�*

loss/�=?��3       �	M���Xc�A�*

loss�TD>/�T       �	i��Xc�A�*

loss�[=��,       �	����Xc�A�*

lossh��=���b       �	�I��Xc�A�*

lossS�>}��3       �	i���Xc�A�*

lossP6=z�Fy       �	us��Xc�A�*

loss�4�=9�h�       �	���Xc�A�*

loss�5�>�.9       �	����Xc�A�*

loss.�>��v       �	�6��Xc�A�*

loss���=�Ҿf       �	����Xc�A�*

loss(�(>�E�       �	�u��Xc�A�*

loss�(=n$=       �	N��Xc�A�*

loss�AM>�R�'       �	N���Xc�A�*

loss���>�#��       �	�O��Xc�A�*

loss`�!>j�f       �	����Xc�A�*

loss���=]=l       �	O���Xc�A�*

lossK�>����       �	+��Xc�A�*

loss�4>M"L        �	����Xc�A�*

loss���=E���       �	U��Xc�A�*

lossל=�6�G       �	}���Xc�A�*

loss]0b>���       �	_~��Xc�A�*

lossڿ�>J>�]       �	���Xc�A�*

lossOb>�G��       �	`���Xc�A�*

loss�Q>�*A       �	�D��Xc�A�*

loss�\>� /�       �	{���Xc�A�*

loss�99>Y�B       �	n��Xc�A�*

loss, >��6�       �	h��Xc�A�*

loss$6u=Ή��       �	{���Xc�A�*

loss�҄=�u�{       �	�6��Xc�A�*

loss��=b.�       �	Q���Xc�A�*

loss�B>�y�       �	�p��Xc�A�*

loss��J=�\n       �	p��Xc�A�*

lossɚ�=6'       �	����Xc�A�*

loss|K�=�?T3       �	�2��Xc�A�*

lossݵ�=�5k"       �	����Xc�A�*

loss��=.�Q       �	�\��Xc�A�*

loss��<=��z�       �	����Xc�A�*

loss蛃>�N��       �	v���Xc�A�*

loss���=�-       �	('��Xc�A�*

loss�>�&       �	���Xc�A�*

loss��?>�pf�       �	$b��Xc�A�*

loss䨾=�0��       �	����Xc�A�*

loss83&=q��       �	m���Xc�A�*

loss2/=�f�I       �	�A��Xc�A�*

loss�1�=�<}[       �	n���Xc�A�*

lossV&�=���       �	7p��Xc�A�*

loss��.>�x�)       �	���Xc�A�*

loss7��>d�7�       �	r���Xc�A�*

losse�=Y"��       �	�D��Xc�A�*

loss]=Y��       �	W���Xc�A�*

loss�U>]�-�       �	�}��Xc�A�*

loss���=���c       �	\��Xc�A�*

loss/�]=[�o       �	����Xc�A�*

loss�SO>N�k       �	�h��Xc�A�*

loss�x> ���       �	�1��Xc�A�*

loss�.�=Zl�       �	����Xc�A�*

loss��>����       �	���Xc�A�*

loss�}3>�3-       �	�N��Xc�A�*

loss֗>M�l'       �	����Xc�A�*

loss�^J=o���       �	����Xc�A�*

loss�s�=h��       �	����Xc�A�*

lossre�=C�       �	|*��Xc�A�*

loss��R=*'z�       �	���Xc�A�*

loss�O	>,�A.       �	?���Xc�A�*

loss��#=��A<       �	�[��Xc�A�*

lossqñ=�"�`       �	#���Xc�A�*

loss�c>j�[       �	���Xc�A�*

loss�h0>�y�       �	�2��Xc�A�*

loss�߉=�چ       �	eP��Xc�A�*

loss]7:=s��	       �	����Xc�A�*

loss���=':       �	ʩ �Xc�A�*

loss#�>+Mڧ       �	�L�Xc�A�*

lossd�I=͘lC       �	z�Xc�A�*

loss4aI>�G�       �	���Xc�A�*

lossY�=?�Q�       �	�a�Xc�A�*

loss�YE>����       �	n��Xc�A�*

loss�8>�?��       �	��Xc�A�*

loss�=^�.�       �	�S�Xc�A�*

loss�a=?A֢       �	���Xc�A�*

loss��5=���       �	A��Xc�A�*

loss�$1>�	��       �	`?�Xc�A�*

loss�b�=��G�       �	���Xc�A�*

loss�EQ>㥕^       �	J��Xc�A�*

loss�ޖ=����       �	�	�Xc�A�*

loss�6�=����       �	�
�Xc�A�*

loss6��=mvс       �	�-�Xc�A�*

lossE� >�y�       �	'�Xc�A�*

lossN�3=�FZt       �	���Xc�A�*

loss���=��g�       �	��Xc�A�*

loss��<(���       �	Zb�Xc�A�*

lossTe�=s�Ǘ       �	�\�Xc�A�*

loss�V=a��       �	&p�Xc�A�*

loss!·=^B�J       �	�:�Xc�A�*

loss#��=�Ǟ�       �	���Xc�A�*

loss`�=���        �	ѕ�Xc�A�*

loss�8\>��5K       �	Q1�Xc�A�*

loss�]�=�X��       �	���Xc�A�*

loss���=P/f�       �	��Xc�A�*

lossT7@>A]�       �	���Xc�A�*

lossP=�SR*       �	�U�Xc�A�*

lossn�<[�K9       �	��Xc�A�*

loss�e=���       �	���Xc�A�*

loss��=��Z�       �	h�Xc�A�*

lossv2�<��"       �	���Xc�A�*

loss��1=�	�\       �	Ww�Xc�A�*

loss���=w���       �	>>�Xc�A�*

losswk1>���       �	�-�Xc�A�*

lossf<���g       �		�Xc�A�*

loss4C<�A�T       �	L��Xc�A�*

loss�<���       �	6x�Xc�A�*

lossE�=)�t       �	�a�Xc�A�*

loss�C>��~�       �	! �Xc�A�*

lossΏ�=�&       �	S� �Xc�A�*

loss��=s6K�       �	�w!�Xc�A�*

loss,��=��ϟ       �	j2"�Xc�A�*

loss��>X��       �	��"�Xc�A�*

loss���<��.       �	Ͻ#�Xc�A�*

loss:�y>1�       �	(�$�Xc�A�*

lossP�=��       �	U%�Xc�A�*

loss!U%>F�       �	��%�Xc�A�*

lossAP>����       �	e�&�Xc�A�*

lossZ�}=� -9       �	�''�Xc�A�*

loss�a>Q��j       �	I�'�Xc�A�*

loss�=>XS	C       �	�Q(�Xc�A�*

loss��_>���       �	F�(�Xc�A�*

loss(O>7C�       �	Ǆ)�Xc�A�*

loss�),>�aI       �	�*�Xc�A�*

loss?R�>M�       �	��*�Xc�A�*

loss�[@>H �       �	|a+�Xc�A�*

lossXJ�=�1&�       �	�+�Xc�A�*

loss�^>d㑊       �	�,�Xc�A�*

lossC*S>9z*       �	�1-�Xc�A�*

loss�(�=͹7|       �	k�-�Xc�A�*

loss� >�h       �	W�.�Xc�A�*

loss�� >����       �	�//�Xc�A�*

loss��q=�U�{       �	T�/�Xc�A�*

loss7 �=��.M       �	|d0�Xc�A�*

loss�"�=7��3       �	*1�Xc�A�*

loss���=_�X�       �	t�1�Xc�A�*

loss{��<pQ�=       �	g2�Xc�A�*

loss���=��       �	 3�Xc�A�*

loss#!=�+Hp       �	
�3�Xc�A�*

loss��'><zi       �	u94�Xc�A�*

loss���<r)'�       �	��4�Xc�A�*

loss��5>s��-       �	fk5�Xc�A�*

loss�J>�<�0       �		6�Xc�A�*

loss��=�yl       �	�6�Xc�A�*

loss���=�9�       �	��7�Xc�A�*

loss��>^���       �	f�8�Xc�A�*

loss�V=���       �	B�9�Xc�A�*

loss�+�=��:�       �	��:�Xc�A�*

loss�ԝ=�d{       �	�;�Xc�A�*

loss<}=;��E       �	-�;�Xc�A�*

loss@
>�_��       �	�~<�Xc�A�*

loss�_>x�<       �	{-=�Xc�A�*

loss�:>��P       �	��=�Xc�A�*

loss�Vh=k�       �	k�>�Xc�A�*

losslg2=Q�ޫ       �	�b?�Xc�A�*

loss���=4B��       �	�@�Xc�A�*

loss��>l؁       �	Y4A�Xc�A�*

loss�o�=�K��       �	N�A�Xc�A�*

lossX��=N3W�       �	:tB�Xc�A�*

lossh->�       �	˝C�Xc�A�*

loss�+Z=ɗGT       �	QiD�Xc�A�*

lossA��=���       �	�E�Xc�A�*

loss�"�=���;       �	H�E�Xc�A�*

loss$�7=�õ       �	��F�Xc�A�*

lossۓ	>����       �	,Jc�Xc�A�*

lossVW�=s��       �	��c�Xc�A�*

lossZCU>t/��       �	�zd�Xc�A�*

loss_��=�`F       �	'e�Xc�A�*

loss�'>Z̀�       �	H�e�Xc�A�*

loss��;=EKS�       �	�8f�Xc�A�*

loss
E�=z���       �	�f�Xc�A�*

loss�=�yμ       �	�sg�Xc�A�*

lossDZ>L�'{       �	J
h�Xc�A�*

loss!$>-�*       �	N�h�Xc�A�*

loss���=�Y��       �	vPi�Xc�A�*

loss��z=�Bh       �	��i�Xc�A�*

loss�">�s�       �	�}j�Xc�A�*

loss���=Ϟ�;       �	?k�Xc�A�*

loss���=ӏ�u       �	��k�Xc�A�*

loss[�h>?S1       �	�Kl�Xc�A�*

loss{�L=ua,h       �	��l�Xc�A�*

loss}&>�_·       �	yum�Xc�A�*

loss�{%>���       �	�n�Xc�A�*

loss�6�>�kU       �	Y�n�Xc�A�*

loss��=n��0       �	l>o�Xc�A�*

loss��j>��       �	��o�Xc�A�*

loss���=m+��       �	s�p�Xc�A�*

loss,�F>Ykg�       �	j3q�Xc�A�*

loss���=����       �	��q�Xc�A�*

loss�Õ=9�e�       �	dr�Xc�A�*

losswD�=��N�       �	��r�Xc�A�*

loss�q�=AU�k       �	��s�Xc�A�*

loss[)>����       �	t)t�Xc�A�*

loss��>��p�       �	Ӽt�Xc�A�*

lossn�>ҧ�       �	�Qu�Xc�A�*

lossIi�=@��       �	��u�Xc�A�*

loss���=��"�       �	Ҭv�Xc�A�*

lossQr[>,,��       �	ܸw�Xc�A�*

loss�e�=���       �	�Yx�Xc�A�*

loss]B>��%�       �	��x�Xc�A�*

loss���<p�
       �	d�y�Xc�A�*

lossED>�[d       �	��z�Xc�A�*

loss�Z�>'I�       �	D{�Xc�A�*

loss	�J>Ӣ�#       �	R�{�Xc�A�*

lossl>3�r1       �	$}�Xc�A�*

lossj��=��O�       �	Ȗ~�Xc�A�*

loss���=ڍ��       �	?q�Xc�A�*

loss��>���s       �	L5��Xc�A�*

loss�s(>���       �	����Xc�A�*

loss(�(>öM�       �	�L��Xc�A�*

loss���=���[       �	J��Xc�A�*

loss��=���X       �	���Xc�A�*

lossAe�=��.�       �	xD��Xc�A�*

lossW�=$v.#       �	���Xc�A�*

loss��=a       �	+���Xc�A�*

losst �>le�       �	.��Xc�A�*

loss���=�\��       �	Bχ�Xc�A�*

lossH��>_���       �	�i��Xc�A�*

loss��4=��Yp       �	X9��Xc�A�*

loss�M=��6       �	z��Xc�A�*

loss��(=���2       �	S���Xc�A�*

lossOn�<�Ja}       �	.=��Xc�A�*

loss���=���C       �	�،�Xc�A�*

loss���=���$       �	�s��Xc�A�*

loss��>p���       �	�=��Xc�A�*

loss,��=�lG$       �	9Վ�Xc�A�*

loss���=\6�       �	c|��Xc�A�*

loss��=��n+       �	�z��Xc�A�*

loss��=ʞ�9       �	���Xc�A�*

loss�[K=E�h�       �	���Xc�A�*

loss���=�f�q       �	�T��Xc�A�*

loss�1>�Yw       �	���Xc�A�*

loss���>��=       �	���Xc�A�*

loss<+>��,       �	�}��Xc�A�*

loss$��=s       �	���Xc�A�*

loss7�=wh]       �	'��Xc�A�*

loss��Z=x�       �	&���Xc�A�*

loss}��= 	e       �	xE��Xc�A�*

loss_�>�� �       �	H���Xc�A�*

loss�R�=_�|�       �	G���Xc�A�*

lossmR�>5���       �	�-��Xc�A�*

loss �o>��       �	�ř�Xc�A�*

loss�=j=�       �	j��Xc�A�*

loss!��=@lx       �	k��Xc�A�*

lossO�>���       �	_���Xc�A�*

lossz�4>�ƨ       �	�p��Xc�A�*

lossR�o=����       �	�
��Xc�A�*

loss�w�<���       �	����Xc�A�*

loss
�=�`�       �	�J��Xc�A�*

lossa1�=�P�n       �	!��Xc�A�*

loss}]�=�ɛp       �	�~��Xc�A�*

loss�ɟ=�@=       �	���Xc�A�*

loss��=Э2�       �	����Xc�A�*

loss���=��Kk       �	[A��Xc�A�*

loss.
>�       �	�ء�Xc�A�*

loss���=��ǜ       �	�n��Xc�A�*

loss ��=⮎�       �	g
��Xc�A�*

loss�>)>�u:$       �	Q���Xc�A�*

loss��}>i���       �	<��Xc�A�*

loss}�E=g��       �	���Xc�A�*

loss�J>�5T�       �	I���Xc�A�*

loss���=�Cl�       �	���Xc�A�*

loss�G_>���|       �	(���Xc�A�*

loss�~>.'u{       �	{J��Xc�A�*

loss/�D=Ӱ=z       �	���Xc�A�*

loss�=�a{       �	䂨�Xc�A�*

loss3;,>n�ZZ       �	X��Xc�A�*

loss|�"=��1�       �	`��Xc�A�*

loss$<7>��|[       �	�{��Xc�A�*

loss�>��|?       �	P��Xc�A�*

loss��<F��       �	���Xc�A�*

loss��<����       �	�H��Xc�A�*

loss�$>Dx�       �	��Xc�A�*

loss�Ŋ=�a��       �	�v��Xc�A�*

loss(��=�\       �	���Xc�A�*

loss���=ާ�       �	���Xc�A�*

lossLO�=N1rg       �	�q��Xc�A�*

loss���<���k       �	���Xc�A�*

lossdش=F$�       �		���Xc�A�*

loss`+=�D�-       �	 E��Xc�A�*

lossD��=�Ȥm       �	ޱ�Xc�A�*

loss��>ef�>       �	����Xc�A�*

loss���=�k       �	�+��Xc�A�*

loss���=�E�       �	ĳ�Xc�A�*

loss�=��vP       �	$^��Xc�A�*

loss���=^���       �	r���Xc�A�*

loss�&>3�eQ       �	����Xc�A�*

lossw��=>��)       �	E���Xc�A�*

loss/U�<-�=�       �	�b��Xc�A�*

loss�,�=��<       �	���Xc�A�*

loss��=���~       �	F���Xc�A�*

lossR=rP�       �	GZ��Xc�A�*

lossﵚ=_n��       �	Y���Xc�A�*

loss��(>mf�       �	��Xc�A�*

loss���=$u�q       �	�7��Xc�A�*

loss۷>7�i�       �	�м�Xc�A�*

loss�>�=�wZ       �	�g��Xc�A�*

loss.4=k1a
       �	L���Xc�A�*

loss2d">�l�B       �	5��Xc�A�*

loss_�=��0       �	"���Xc�A�*

loss��>�&s       �	#��Xc�A�*

loss[�U=k�o�       �	���Xc�A�*

loss�<�<�d�       �	�U��Xc�A�*

lossJ>*=�4�       �	����Xc�A�*

loss�G>=��"       �	ѭ��Xc�A�*

loss�ۏ=�>��       �	(C��Xc�A�*

loss�B>�\�@       �	����Xc�A�*

loss��>��       �	0g��Xc�A�*

lossdb">2C7�       �	u��Xc�A�*

loss�4=ݥ��       �	,���Xc�A�*

lossR�>A^       �	�@��Xc�A�*

lossL[<��uD       �	A���Xc�A�*

loss���=����       �	n��Xc�A�*

loss)�=+^�r       �	���Xc�A�*

loss�=�"       �	Ǡ��Xc�A�*

loss�]>��[       �	`>��Xc�A�*

lossx6&>�?��       �	����Xc�A�*

lossS��=��]       �	�o��Xc�A�*

loss*$�= 1r�       �	)��Xc�A�*

lossa^!=P}�       �	����Xc�A�*

loss���=Y��*       �	C;��Xc�A�*

loss�E>�i��       �	e���Xc�A�*

loss�)>�$�       �	����Xc�A�*

loss25=!�(       �	7��Xc�A�*

loss�m>�@�       �	����Xc�A�*

losse��<����       �	{K��Xc�A�*

loss�~�=:��R       �	����Xc�A�*

losst�I=����       �	�z��Xc�A�*

loss)��=-�       �	���Xc�A�*

loss���=����       �	2���Xc�A�*

lossԍ�=c�       �	VG��Xc�A�*

loss�^t=���       �	;���Xc�A�*

lossj��=�jl       �	�x��Xc�A�*

loss�d�=ĕ��       �	_&��Xc�A�*

loss��>xA��       �	=���Xc�A�*

lossۻ�<����       �	�K��Xc�A�*

loss���=r�`       �	����Xc�A�*

lossm�>��/�       �	�v��Xc�A�*

loss\Κ<!H�s       �	�$��Xc�A�*

loss7�=���.       �	���Xc�A�*

loss���=g̱       �	�[��Xc�A�*

loss ��=�!       �	����Xc�A�*

loss�u�<Yh�       �	U���Xc�A�*

loss�~�<��       �	5%��Xc�A�*

lossZ*�<'?"�       �	X���Xc�A�*

losszV>��       �	Qf��Xc�A�*

loss��=�y�       �	���Xc�A�*

loss�/=R-       �	���Xc�A�*

loss�w�<���       �	IM��Xc�A�*

lossS��=��ŵ       �	����Xc�A�*

loss@�=�*+/       �	}��Xc�A�*

loss�=�S�       �	j��Xc�A�*

loss�[�=��I�       �	����Xc�A�*

losse�V=&;�t       �	�@��Xc�A�*

loss�2>�J�       �	8���Xc�A�*

loss��=n��       �	�p��Xc�A�*

loss�1$=�RP       �	b��Xc�A�*

lossT�m=~�G       �	m���Xc�A�*

loss8�=*� �       �	BB��Xc�A�*

loss8��=C��       �	����Xc�A�*

loss�&�=�;��       �	�t��Xc�A�*

loss��=�1��       �	���Xc�A�*

lossE�4>����       �	3���Xc�A�*

loss!ML=*�	�       �	�@��Xc�A�*

loss:,>q�       �	0���Xc�A�*

loss��=��>       �	rn��Xc�A�*

loss� �=�l�       �	���Xc�A�*

losso(�=����       �	���Xc�A�*

loss�~�<�f       �	aR��Xc�A�*

loss�"x=����       �	p���Xc�A�*

loss��>bw��       �	G���Xc�A�*

loss\A>��,       �	.��Xc�A�*

lossm�x=�|o       �	e���Xc�A�*

loss�>̾��       �	f��Xc�A�*

loss%��=���       �	����Xc�A�*

loss�[='c�S       �	����Xc�A�*

loss�g=r{}�       �	5��Xc�A�*

loss�f=�u       �	����Xc�A�*

loss3��=N���       �	U���Xc�A�*

lossd�y=�4��       �	���Xc�A�*

loss�j>�\       �	9���Xc�A�*

lossq�>[;�       �	�O��Xc�A�*

loss��><��       �	����Xc�A�*

losshu6>��Z�       �	���Xc�A�*

loss�Q	=Q;3f       �	���Xc�A�*

loss��=��H�       �	l���Xc�A�*

lossۮ�=��$i       �	�J��Xc�A�*

loss�>F�       �	 ���Xc�A�*

loss��n=��V�       �	$|��Xc�A�*

loss��r=�W�       �	,��Xc�A�*

loss��=��R�       �	��Xc�A�*

lossOn�=��h       �	 ���Xc�A�*

loss���=a��h       �	L���Xc�A�*

lossߔ�=��       �	wM��Xc�A�*

loss�=�5��       �	����Xc�A�*

loss���<�r       �	����Xc�A�*

loss� 7>�	�L       �	n���Xc�A�*

loss���=��K       �	����Xc�A�*

loss1\�=Ňʯ       �	�r��Xc�A�*

loss(u�=�F��       �	%]��Xc�A�*

loss(^�=�
       �	����Xc�A�*

lossh|>k��K       �	����Xc�A�*

loss�j>�h0       �	'� �Xc�A�*

lossd�K=�݆       �	l��Xc�A�*

loss�iV=%�R       �	�D�Xc�A�*

lossp��=tJ�-       �	�+�Xc�A�*

loss1�=�� �       �	:��Xc�A�*

lossN�P=�L�d       �	n��Xc�A�*

lossJ��=`-��       �	�@�Xc�A�*

loss��=�Q��       �	�	�Xc�A�*

loss��>e��       �	�/�Xc�A�*

loss��=�Auw       �	8��Xc�A�*

loss�9>���       �	��Xc�A�*

loss�3�=k��       �	�	�Xc�A�*

lossD��=_t�_       �	oG
�Xc�A�*

loss�p >�X��       �	��
�Xc�A�*

loss�,�=M�E       �	"��Xc�A�*

lossAk�<��٘       �	{��Xc�A�*

loss�RS=��zf       �	-A�Xc�A�*

loss�W�=.Y,       �	Y�Xc�A�*

loss�>�k%�       �	*�Xc�A�*

loss*h>WT�<       �	���Xc�A�*

lossk�=�Yc       �	E��Xc�A�*

loss	��=8��       �	p>�Xc�A�*

loss{UO=�@�       �	���Xc�A�*

loss�ǌ=��e       �	0��Xc�A�*

loss8L>w��/       �	0d�Xc�A�*

loss�;�=*`=       �	h�Xc�A�*

loss� �=�h
P       �	
��Xc�A�*

loss�S*=��M       �	k}�Xc�A�*

loss�=�9��       �	�[�Xc�A�*

loss}>c�}�       �	 �Xc�A�*

loss�i>�
t�       �	���Xc�A�*

lossaÎ=��`�       �	uv�Xc�A�*

lossG��=hy34       �	?��Xc�A�*

loss���=Κ��       �	�(�Xc�A�*

loss�8�=pq��       �	���Xc�A�*

loss8!N=��m       �	��Xc�A�*

lossyr>���       �	���Xc�A�*

loss&�O>��P�       �	m�Xc�A�*

lossò�=4�=       �	�u�Xc�A�*

loss��<pI"       �	��Xc�A�*

loss z*>��U       �	�5 �Xc�A�*

lossL��=5�i        �	�� �Xc�A�*

loss��=�D�K       �	|!�Xc�A�*

loss��9=����       �	�"�Xc�A�*

loss��=} ]�       �	�"�Xc�A�*

loss4��<xh�       �	W#�Xc�A�*

lossk�>���        �	]$�Xc�A�*

loss�n�=��a�       �	��$�Xc�A�*

loss��^=���       �	�O%�Xc�A�*

lossm.8=�5bY       �	��%�Xc�A�*

loss�>�=�x�:       �	��&�Xc�A�*

loss��<��<A       �	A'�Xc�A�*

loss��=��       �	��'�Xc�A�*

lossN�=���       �	b�(�Xc�A�*

lossR\b<^YŎ       �	�7)�Xc�A�*

loss $�=5���       �	��)�Xc�A�*

loss��1=̡�!       �	<l*�Xc�A�*

lossw)�=F�?       �	�+�Xc�A�*

lossm�,=4}�       �	��+�Xc�A�*

loss�QA>{�N       �	�I,�Xc�A�*

loss6!%>2ni�       �	|-�Xc�A�*

lossg� >]-}	       �	��-�Xc�A�*

lossC��=Gd��       �	
g.�Xc�A�*

lossx��=>�P       �	�/�Xc�A�*

loss��=u       �	Ŭ/�Xc�A�*

losszd=�       �	PT0�Xc�A�*

loss��=B��       �	�1�Xc�A�*

loss p>jVĹ       �	M�1�Xc�A�*

lossh��<����       �	�C2�Xc�A�*

loss���==�_p       �	��2�Xc�A�*

lossʰ4=�6m?       �	��3�Xc�A�*

loss�W=�`�       �	�-4�Xc�A�*

lossѴ�=OP       �	s�4�Xc�A�*

lossD�=s��*       �	�l5�Xc�A�*

lossH�E>��"�       �	16�Xc�A�*

lossI�M=20O�       �	�7�Xc�A�*

loss,
�=a��s       �	��7�Xc�A�*

losss/�=�f�       �	і8�Xc�A�*

loss�_=a~w(       �	329�Xc�A�*

loss��	=g��3       �	��9�Xc�A�*

lossi;>���       �	�:�Xc�A�*

loss�(=RE��       �	�o;�Xc�A�*

loss���<3�       �	�<�Xc�A�*

loss��g>g.��       �	��<�Xc�A�*

loss�3�=�-Y�       �	Ee=�Xc�A�*

loss1E=|\k       �	��>�Xc�A�*

loss���=���y       �	kG?�Xc�A�*

loss��>��@�       �	��?�Xc�A�*

loss:��=�ՏU       �	$�@�Xc�A�*

lossF�=4�k}       �	tA�Xc�A�*

losss��=O�u       �	�B�Xc�A�*

lossƹ">㕳<       �	N�B�Xc�A�*

loss�(�=����       �	`tC�Xc�A�*

loss�k:=b�6       �	�D�Xc�A�*

lossTj�<�k]       �	6�D�Xc�A�*

losss�<촢�       �	iE�Xc�A�*

loss�>"�3F       �	�F�Xc�A�*

lossҰ�=�B       �	.�F�Xc�A�*

loss���=c1�H       �	�^G�Xc�A�*

loss��=�xl�       �	+�H�Xc�A�*

loss�]�=i��       �	ΩI�Xc�A�*

losssp�<J�?       �	1[J�Xc�A�*

loss�/ =�l_�       �	K�Xc�A�*

lossS�B=`G��       �	��K�Xc�A�*

loss@��=&@�       �	oKL�Xc�A�*

loss��n=
�^�       �	|�L�Xc�A�*

loss� �>h �
       �	��M�Xc�A�*

loss���=�Uj�       �	�DN�Xc�A�*

loss��=�&�       �	��N�Xc�A�*

lossd�<>�SШ       �	f�O�Xc�A�*

loss�`a=:;0`       �	cFP�Xc�A�*

lossN�=g'S       �	��P�Xc�A�*

lossL}7=RN2       �	��Q�Xc�A�*

lossM��=��       �	o)R�Xc�A�*

lossalj=h#�C       �	��R�Xc�A�*

loss��=zL�       �	$dS�Xc�A�*

loss39�=��[       �	a�S�Xc�A�*

loss�-�<�᭡       �	�T�Xc�A�*

lossN�=����       �	}�U�Xc�A�*

lossܾ�=���       �	��V�Xc�A�*

loss�<�=���3       �	�IW�Xc�A�*

loss���<����       �	��W�Xc�A�*

lossj��=���       �	M�Y�Xc�A�*

loss=+�a�       �	bMZ�Xc�A�*

loss8��=}�P       �	�[�Xc�A�*

loss�9>6m�       �	��[�Xc�A�*

lossX�>ڍS�       �	Y\�Xc�A�*

loss.��=֘!�       �	n�\�Xc�A�*

loss(i>B��T       �	��]�Xc�A�*

loss[o�<��r�       �	T6^�Xc�A�*

loss���=J<Y�       �	��^�Xc�A�*

loss�I_=�G�       �	eq_�Xc�A�*

loss2�=��cJ       �	�`�Xc�A�*

lossq =T�
       �	Y�`�Xc�A�*

loss�V�=p<{�       �	�ca�Xc�A�*

loss��=�8"m       �	��b�Xc�A�*

lossC3�=ȏ�Z       �	�Gc�Xc�A�*

loss�b�=��օ       �	W�c�Xc�A�*

loss��R>�[E       �	��d�Xc�A�*

loss73*=h�       �	�3e�Xc�A�*

loss��=��       �	!�e�Xc�A�*

lossM�#>�I�       �	sf�Xc�A�*

loss�(�=���       �	�g�Xc�A�*

losst�>){�       �	��g�Xc�A�*

loss���=��s       �	�Ih�Xc�A�*

loss�~g=��*�       �	��h�Xc�A�*

loss��b=	��       �	|i�Xc�A�*

loss��=*Mt�       �	�%j�Xc�A�*

lossA�=5�v�       �	��j�Xc�A�*

loss`�=��       �	�Yk�Xc�A�*

loss]8�=&�ʗ       �	��k�Xc�A�*

loss;�f=��{�       �	&�l�Xc�A�*

loss�=�2a�       �	�+m�Xc�A�*

loss@�W=���       �	u�m�Xc�A�*

loss��=*�r       �	nn�Xc�A�*

loss,>���       �	5o�Xc�A�*

loss쾥=���       �	
�o�Xc�A�*

loss�?>,U$�       �	H4p�Xc�A�*

loss�S�<�Lp       �	��p�Xc�A�*

loss�~=5S!L       �	�kq�Xc�A�*

lossw*=�ڬ       �	�lr�Xc�A�*

loss��=�B�Q       �	s�Xc�A�*

loss� =�s>!       �	/�s�Xc�A�*

loss}�{=K       �	[Ct�Xc�A�*

loss�>��P       �	��t�Xc�A�*

loss��=�~%       �	�u�Xc�A�*

loss
d=mc��       �	�"v�Xc�A�*

lossR]�=˵	�       �	��v�Xc�A�*

loss���=	���       �	�w�Xc�A�*

loss/`=9�O       �	��x�Xc�A�*

lossX�]=j�V       �	� z�Xc�A�*

lossr�z>:@��       �	�{�Xc�A�*

loss%$�=��`�       �	r|�Xc�A�*

lossT�>��[6       �	ӽ|�Xc�A�*

loss�Q=A�       �	�h}�Xc�A�*

loss(Jf>~u�        �	�k~�Xc�A�*

lossB�=��r#       �	<1�Xc�A�*

loss�0>K4�)       �	���Xc�A�*

lossѫA<u��       �	�Ā�Xc�A�*

lossR�N=�#��       �	%w��Xc�A�*

lossHs�=b�L       �	����Xc�A�*

lossOS�=�t       �	l|��Xc�A�*

loss��=�e�       �	��Xc�A�*

loss	 �=�P�x       �	���Xc�A�*

loss;(�=n�       �	.���Xc�A�*

lossXI>'�9�       �	U���Xc�A�*

loss���=�W�       �	5`��Xc�A�*

loss,��=��Kk       �	&:��Xc�A�*

loss���=8i       �	p��Xc�A�*

loss��=���       �	����Xc�A�*

loss��E=E�n�       �	���Xc�A�*

loss�w=���       �	ˢ��Xc�A�*

loss���=�r�5       �	���Xc�A�*

loss�A�=4�t$       �	ԙ��Xc�A�*

loss�=#=��       �	��Xc�A�*

losss��=���g       �	���Xc�A�*

loss�w_=�L��       �	O��Xc�A�*

lossi�=�A(�       �	E���Xc�A�*

loss�!	=�.�       �	k��Xc�A�*

loss��<t�,       �	M/��Xc�A�*

loss;�v<}͢�       �	���Xc�A�*

loss5�=�n!�       �	���Xc�A�*

loss�[a>^Fm       �	����Xc�A�*

lossAC|>��|       �	�X��Xc�A�*

lossw�=�9�7       �	~q��Xc�A�*

loss܆,>0��F       �	�%��Xc�A�*

loss6�{=��[(       �	����Xc�A�*

loss�S<=h5�H       �	����Xc�A�*

loss�z>:H�       �	���Xc�A�*

lossJ�=��       �	���Xc�A�*

loss��%=G��       �	}v��Xc�A�*

lossCf$>8�T�       �	���Xc�A�*

losse>l��       �	����Xc�A�*

loss#�=�;:M       �	�>��Xc�A�*

loss���<|<�       �	�ҝ�Xc�A�*

loss[R`>��(       �	4���Xc�A�*

lossN~>Μ�'       �	&��Xc�A�*

loss���=L{�c       �	ڭ��Xc�A�*

lossP`#>��
       �	�?��Xc�A�*

loss2'�=��-       �	�Ӡ�Xc�A�*

loss�
�=��       �	e��Xc�A�*

loss6�>�듎       �	u��Xc�A�*

loss�=�x��       �	����Xc�A�*

loss��:=���       �	.:��Xc�A�*

lossn�)=Z�!)       �	�Σ�Xc�A�*

lossM�P>��*       �	�`��Xc�A�*

loss�.#=��       �	�&��Xc�A�*

loss�-?=G�d       �	˹��Xc�A�*

loss��a=�b(�       �	R��Xc�A�*

loss��= ��       �	�;��Xc�A�*

lossd<(>gP+       �	�ק�Xc�A�*

lossL"=<�       �	�r��Xc�A�*

lossa��=^lM�       �	-
��Xc�A�*

lossq�=u�+�       �	����Xc�A�*

loss���=\��%       �	=E��Xc�A�*

loss%v>:3�       �	jު�Xc�A�*

loss�OT=�Ū       �	�u��Xc�A�*

loss "A=u���       �	���Xc�A�*

loss3z�<w�/D       �	m���Xc�A�*

lossU�=�jm�       �	�M��Xc�A�*

losse�<�͔       �	���Xc�A�*

loss��)>d�Fy       �	����Xc�A�*

loss&�_>~o��       �	�:��Xc�A�*

loss&r�=y�Vd       �	}Я�Xc�A�*

lossqƠ=#v�       �	`r��Xc�A�*

loss�5�=� pY       �	K��Xc�A�*

loss�>�`�       �	8���Xc�A�*

loss�X	=p�:       �	vq��Xc�A�*

lossɁ=�Ix       �	��Xc�A�*

lossx��=��D       �	5���Xc�A�*

loss�	=���       �	eU��Xc�A�*

lossq��==�w       �	����Xc�A�*

lossW0\=ԙ�`       �	8���Xc�A�*

loss3>P       �	�E��Xc�A�*

lossփ�<�͢�       �	�9��Xc�A�*

lossJ-]=�K��       �	���Xc�A�*

loss&/4=q4�       �	Z���Xc�A�*

loss�3=H��&       �	�c��Xc�A�*

loss1'�=3�;       �	�{��Xc�A�*

loss���<>t�       �	�)��Xc�A�*

loss�/=u*�       �	CŻ�Xc�A�*

loss$��=�f/)       �	�d��Xc�A�*

lossEp�=�Ѕ       �	���Xc�A�*

loss�\=BQ�       �	u���Xc�A�*

loss��<)�c(       �	�P��Xc�A�*

lossXjE=A��       �	"���Xc�A�*

lossd��=�
 �       �	r���Xc�A�*

loss��;=���       �	Tq��Xc�A�*

loss
�>��       �	T��Xc�A�*

lossZ�\=ߋ~�       �	L���Xc�A�*

loss��>�6@X       �	Zd��Xc�A�*

loss��b=�C+       �	����Xc�A�*

loss}� =�ڄ�       �	Ӡ��Xc�A�*

loss�_�=,���       �	+L��Xc�A�*

lossj��<s�"�       �	����Xc�A�*

loss��=e ��       �	s���Xc�A�*

lossz��=F�ݣ       �	���Xc�A�*

loss?�!>T��e       �	p���Xc�A�*

lossÝ�=�	��       �	�M��Xc�A�*

lossi:=(���       �	�)��Xc�A�*

lossgd=R��       �	+���Xc�A�*

loss��=�;��       �	&W��Xc�A�*

loss�~;=�j�F       �	1��Xc�A�*

loss�ۇ=�h~%       �	;���Xc�A�*

loss'f#= �zS       �	\��Xc�A�*

loss�)�<�U       �	���Xc�A�*

lossME5=�*�o       �	����Xc�A�*

loss==��t�       �	U��Xc�A�*

lossۯ�<�)�       �	O��Xc�A�*

lossXI>�q�(       �	���Xc�A�*

loss���=��M       �	(}��Xc�A�*

lossǦ=�0,�       �	x'��Xc�A�*

lossnW,=��-/       �	K���Xc�A�*

loss��<�l�       �	Vc��Xc�A�*

loss*�<h�,       �	 ��Xc�A�*

lossJ�Y<��(�       �	:#��Xc�A�*

loss�r=�k �       �	����Xc�A�*

lossd��;��l       �	�U��Xc�A�*

lossq�<ם5C       �	����Xc�A�*

loss#��;j:ȕ       �	$���Xc�A�*

loss؁ <u���       �	�A��Xc�A�*

loss�}�=,�w       �	��Xc�A�*

loss��;S%��       �	/���Xc�A�*

lossdCw;%�C�       �	t@��Xc�A�*

loss��;IO4       �	M���Xc�A�*

losseK�<Q��N       �	�q��Xc�A�*

loss8��=�<�       �	
��Xc�A�*

loss��=f5b�       �	���Xc�A�*

loss|�1<�s��       �	ob��Xc�A�*

lossİ�=0W�R       �	P���Xc�A�*

loss���>GbM�       �	����Xc�A�*

loss�Vr<.��       �	�A��Xc�A�*

loss�E�>���       �	�$��Xc�A�*

loss�>"�'�       �	�0��Xc�A�	*

loss�>>��1�       �	N���Xc�A�	*

loss���=%a�|       �	:w��Xc�A�	*

lossAg5=�X�       �	���Xc�A�	*

lossC��=�{�       �	����Xc�A�	*

lossA��=_       �	o��Xc�A�	*

loss��=[��       �	P��Xc�A�	*

loss��=�4;�       �	����Xc�A�	*

loss׼�=X,#�       �	����Xc�A�	*

loss��>�!�6       �	�3��Xc�A�	*

lossڵ>��ϩ       �	s���Xc�A�	*

loss���=zGE�       �	ly��Xc�A�	*

loss�0�=)�X!       �	���Xc�A�	*

loss�>]q@�       �	X���Xc�A�	*

lossj
�=�X�       �	�K��Xc�A�	*

lossΪ�=3
e�       �	����Xc�A�	*

loss(�>*g�%       �	4���Xc�A�	*

loss_7y=���       �	�*��Xc�A�	*

loss@�7=VZ��       �	����Xc�A�	*

loss=vR=uL��       �	Mf��Xc�A�	*

lossBa�=
��?       �	���Xc�A�	*

lossaP=d@��       �	e���Xc�A�	*

loss��+=<Η�       �	{K��Xc�A�	*

loss	:=��M       �	S!��Xc�A�	*

loss�.>�.:~       �	z���Xc�A�	*

loss�N=9�|�       �	"l��Xc�A�	*

loss�>���       �	��Xc�A�	*

loss���=��<       �	����Xc�A�	*

loss�M=!�X�       �	�Q��Xc�A�	*

lossMɌ=t�$g       �	8���Xc�A�	*

lossw��=�l<�       �	e���Xc�A�	*

loss�Z<�[$       �	0��Xc�A�	*

loss	z�=�Lz�       �	����Xc�A�	*

loss���<�rԶ       �	���Xc�A�	*

loss��s<�G�7       �	K ��Xc�A�	*

loss�`!=Cc�       �	]���Xc�A�	*

loss���=�fw�       �	���Xc�A�	*

loss��=l:�       �	�^��Xc�A�	*

loss�U�<C�,�       �	���Xc�A�	*

loss��{=�h       �	����Xc�A�	*

loss30�=>��       �	@P��Xc�A�	*

loss��=���       �	>$��Xc�A�	*

loss��=$��'       �	o��Xc�A�	*

loss=J�=ml       �	1A��Xc�A�	*

loss(O�=��H)       �	����Xc�A�	*

loss���<���&       �	z���Xc�A�	*

loss�Z�=!ڑ       �	�]��Xc�A�	*

loss�=tL�       �	���Xc�A�	*

loss�7=#ں       �	7���Xc�A�	*

loss�xt=x ^       �	˄�Xc�A�	*

loss��>����       �	%�Xc�A�	*

loss��=t׫       �	��Xc�A�	*

lossMB�=@��       �	jN�Xc�A�	*

lossah�=�v��       �	��Xc�A�	*

loss��<h��S       �	���Xc�A�	*

lossR|d=�K�A       �	� �Xc�A�	*

loss�3T=���       �	z� �Xc�A�	*

loss<��=- ƾ       �	B^!�Xc�A�	*

lossU��=�BS       �	p"�Xc�A�	*

loss�
�<epG       �	M�"�Xc�A�	*

loss���={��       �	�b#�Xc�A�	*

loss��=�M�       �	�#�Xc�A�	*

lossi|�=�_b       �	��$�Xc�A�	*

lossD_U=_m��       �	�7%�Xc�A�	*

loss�@>҇j�       �	P�%�Xc�A�	*

loss:��;��       �	�~&�Xc�A�	*

loss���=�F�       �	�''�Xc�A�	*

loss�ܣ=t�GI       �	e�'�Xc�A�	*

loss*x�>
c�<       �	�`(�Xc�A�	*

loss��w=��       �	��(�Xc�A�	*

loss*Ȉ>�D�       �	��)�Xc�A�	*

loss�/=��I�       �	Z/*�Xc�A�	*

loss?�	>~�6+       �	��*�Xc�A�	*

loss��=>       �	�,�Xc�A�	*

loss�V�<�m�<       �	z�,�Xc�A�	*

loss��=�<t�       �	�R-�Xc�A�	*

loss(2"=���       �	 �-�Xc�A�	*

loss�=�a�       �	��.�Xc�A�	*

loss�\r=�H�w       �	K?/�Xc�A�	*

loss�_�=�Κ�       �	00�Xc�A�	*

loss�d�=��8�       �	F�0�Xc�A�	*

lossxn�<3��       �	�]1�Xc�A�	*

lossSN,>��^�       �	%2�Xc�A�	*

loss���=���5       �	��2�Xc�A�	*

loss!t�=譗�       �	k3�Xc�A�	*

loss���<�*b       �	�4�Xc�A�	*

loss�R^=պ��       �	��4�Xc�A�	*

loss_4g>D�;�       �	�]5�Xc�A�	*

loss2��=��p[       �	6�Xc�A�	*

lossO3=Ε�       �	֧6�Xc�A�	*

loss�z�=�)P�       �	4e7�Xc�A�	*

loss��=O��       �	�8�Xc�A�	*

loss�U>N�R       �	�9�Xc�A�	*

loss�G�=y�"       �	$�9�Xc�A�	*

lossA[>��F       �	�u:�Xc�A�	*

loss�)�<^��'       �	];�Xc�A�	*

loss�?o=P��       �	f<�Xc�A�	*

loss�y�=N�q�       �	o�<�Xc�A�	*

loss|�<69�       �	c=�Xc�A�	*

loss!fL<i��2       �	�N>�Xc�A�	*

loss�}�=3�=       �	�M?�Xc�A�	*

lossc�'=��w       �	��?�Xc�A�	*

losszd�>��'�       �	��@�Xc�A�	*

loss�8S=�a       �	H3A�Xc�A�	*

losszhB<�'       �	D�B�Xc�A�	*

loss��<��#�       �	F#C�Xc�A�	*

loss2U�<��j5       �	��C�Xc�A�	*

loss��=]��       �	�kD�Xc�A�	*

lossO�=)^�       �	cE�Xc�A�	*

loss�D0>��A       �	��E�Xc�A�	*

loss��O=.���       �	'JF�Xc�A�	*

lossLa=�2�&       �	��F�Xc�A�	*

loss��=l,Q~       �	��G�Xc�A�	*

loss��j=�N��       �	�*H�Xc�A�	*

lossd��=�6�       �	��H�Xc�A�	*

loss���=���       �	�kI�Xc�A�	*

loss:X;=��[�       �	IJ�Xc�A�	*

loss��=Ce��       �	s�J�Xc�A�	*

loss��>���       �	�dK�Xc�A�	*

loss���=w�ϙ       �	L�Xc�A�	*

loss�S\=�fۥ       �	S�L�Xc�A�	*

loss��=�       �	�RM�Xc�A�	*

loss�e�<��h       �	
�M�Xc�A�	*

lossM�=�y�       �	�N�Xc�A�	*

loss,]�<�1c=       �	28O�Xc�A�	*

loss�X=w�0	       �	��O�Xc�A�	*

loss}p>��Ԫ       �	�|P�Xc�A�	*

loss�UT='��V       �	Q�Xc�A�	*

loss9��=��$       �	8�Q�Xc�A�
*

loss$ֻ=U���       �	�aR�Xc�A�
*

lossf�>���       �	��R�Xc�A�
*

loss���=����       �	�S�Xc�A�
*

lossn�Q=��q       �	�BT�Xc�A�
*

lossvT�=��Ms       �	�T�Xc�A�
*

loss.'s=�y�       �	��U�Xc�A�
*

lossw�<��!       �	V�Xc�A�
*

loss�u�=i��       �	Z�V�Xc�A�
*

loss܆�<�� �       �	�VW�Xc�A�
*

lossq��<���       �	M�W�Xc�A�
*

loss�	[=?'��       �	m�X�Xc�A�
*

loss�u�=�~G}       �	�NY�Xc�A�
*

loss�=�H�       �	��Y�Xc�A�
*

loss�l=�|�F       �	,�Z�Xc�A�
*

loss�=je�h       �	2[�Xc�A�
*

loss�h�<
Ր       �	o�[�Xc�A�
*

loss �[=���       �	�n\�Xc�A�
*

loss��u=        �	'2]�Xc�A�
*

lossi��=m�m6       �	>�]�Xc�A�
*

loss���=���I       �	�l^�Xc�A�
*

loss]�9=�v>�       �	�	_�Xc�A�
*

loss6G<��u�       �	H`�Xc�A�
*

lossL��=d��r       �	h�`�Xc�A�
*

loss�@�<��C�       �	B�a�Xc�A�
*

loss�F�=o\�       �	,b�Xc�A�
*

loss�D�=��zN       �	�c�Xc�A�
*

loss��=�1��       �	��c�Xc�A�
*

lossȭ�=1�ʏ       �	/kd�Xc�A�
*

loss��i>�.��       �	4e�Xc�A�
*

loss5�=��F       �	Ӽe�Xc�A�
*

losse��=�H,       �	�[f�Xc�A�
*

lossuE�=3L��       �	��f�Xc�A�
*

loss��=����       �	ȕg�Xc�A�
*

losse��<#�R       �	�>h�Xc�A�
*

loss{gr=���       �	@�h�Xc�A�
*

loss�"`=��}�       �	�i�Xc�A�
*

lossף(=��	       �	;�j�Xc�A�
*

loss���=��Ҟ       �	!tk�Xc�A�
*

loss�Ի=X�!       �		l�Xc�A�
*

loss�OU<4bY�       �	�l�Xc�A�
*

loss�&�<�9�       �	�bm�Xc�A�
*

lossr��<��L�       �	�m�Xc�A�
*

losstm�=%93�       �	Q�n�Xc�A�
*

lossI�P=2��W       �	�:o�Xc�A�
*

loss�@<ːȒ       �	��o�Xc�A�
*

loss���=5��       �	:�p�Xc�A�
*

loss���=�)a�       �	u:q�Xc�A�
*

lossvM�<g/��       �	��q�Xc�A�
*

loss8��=s	1g       �	ULs�Xc�A�
*

loss��=�8T\       �	��s�Xc�A�
*

loss�]=�x��       �	��t�Xc�A�
*

loss���=�r�       �	�Wu�Xc�A�
*

lossv�_=P���       �	��u�Xc�A�
*

loss�A�<a
��       �	��v�Xc�A�
*

loss��>���9       �	�8w�Xc�A�
*

loss]bl=��E`       �	��w�Xc�A�
*

loss˲=�%O�       �	#�x�Xc�A�
*

loss&�=��,       �	Ǆy�Xc�A�
*

lossl!=�i��       �	�Vz�Xc�A�
*

lossDR�<М�       �	
{�Xc�A�
*

lossD�/<-4�H       �	Zg|�Xc�A�
*

loss��_=��       �	��|�Xc�A�
*

loss�w%=G&       �	��}�Xc�A�
*

loss��>U       �	
+~�Xc�A�
*

loss#!>�s��       �	;�~�Xc�A�
*

loss��<�%�       �	!\�Xc�A�
*

loss�$�=	!&�       �	���Xc�A�
*

loss��v<?��       �	L���Xc�A�
*

loss2�:=�Q�       �	K?��Xc�A�
*

loss!��=#��S       �	����Xc�A�
*

lossQ,v=p`       �	�w��Xc�A�
*

losstۙ=p\�       �	���Xc�A�
*

loss��=��#s       �	Cƃ�Xc�A�
*

loss]=��X       �	*o��Xc�A�
*

loss�d�=-�P�       �	��Xc�A�
*

lossB<�Q�'       �	���Xc�A�
*

loss��M=si       �	0��Xc�A�
*

loss�[="�f�       �	�ņ�Xc�A�
*

lossj;>��1p       �	0d��Xc�A�
*

lossa��<K)y�       �	��Xc�A�
*

loss��>�ey�       �	Ϊ��Xc�A�
*

loss�y6=-��       �	�W��Xc�A�
*

loss��7=R*�       �	����Xc�A�
*

loss8	J<�Ci       �	����Xc�A�
*

loss�}=4��n       �	zQ��Xc�A�
*

lossC��<G�nD       �	���Xc�A�
*

lossX��<�1g�       �	A��Xc�A�
*

loss$Ց=�Xt       �	u��Xc�A�
*

loss83=�W	       �	���Xc�A�
*

lossܚ�=i~q�       �	p{��Xc�A�
*

loss�T>\���       �	b��Xc�A�
*

losso/S<s���       �	U���Xc�A�
*

loss\��=�*�y       �	�B��Xc�A�
*

loss�>��{       �	bא�Xc�A�
*

lossE�<���       �	t��Xc�A�
*

loss81=����       �	���Xc�A�
*

lossd��=]Z�       �	���Xc�A�
*

loss�g>:<�`       �	�=��Xc�A�
*

loss]��<Gk��       �	���Xc�A�
*

lossVA�<(	��       �	����Xc�A�
*

loss|��<wb��       �	�>��Xc�A�
*

loss��=t�v(       �	�ו�Xc�A�
*

losse��<%       �	Su��Xc�A�
*

loss菷<�`�       �	��Xc�A�
*

loss��<3/İ       �	u���Xc�A�
*

loss�V<�X{:       �	Nb��Xc�A�
*

loss�5f=��v       �	h��Xc�A�
*

lossN�<��4Q       �	8���Xc�A�
*

loss�3$=�[�       �	�B��Xc�A�
*

loss�=���       �	���Xc�A�
*

loss�/�<��Q       �	]���Xc�A�
*

loss��J=r��       �	-%��Xc�A�
*

loss��=$T-5       �	�ɜ�Xc�A�
*

loss�=���       �	�a��Xc�A�
*

lossi�;=���,       �	D���Xc�A�
*

loss̰a=W��#       �	$���Xc�A�
*

lossR�=C���       �	<��Xc�A�
*

loss�-�=���       �	d;��Xc�A�
*

loss�K >�>�^       �	ՠ�Xc�A�
*

loss�=�<���       �	�ߡ�Xc�A�
*

lossd	�=s��       �	-|��Xc�A�
*

loss��*<�8�       �	6��Xc�A�
*

lossH�K=���       �	����Xc�A�
*

lossT�s<���       �	�X��Xc�A�
*

loss��c<"��(       �	����Xc�A�
*

loss� j=C���       �	���Xc�A�
*

lossc��=�|��       �	�(��Xc�A�
*

loss�;�=���k       �	{ݦ�Xc�A�*

loss�=4*Ey       �	�{��Xc�A�*

loss`�=�
       �	���Xc�A�*

loss?�=�[�       �	����Xc�A�*

lossc�=��r       �	�V��Xc�A�*

loss2�f<���M       �	'���Xc�A�*

loss��=J��       �	6��Xc�A�*

loss�h=�df�       �	囫�Xc�A�*

loss��:=��!�       �	�=��Xc�A�*

loss�5Z>?��       �	�׬�Xc�A�*

loss̃	>���       �	���Xc�A�*

loss~�>�P{�       �	����Xc�A�*

lossZx>P���       �	C���Xc�A�*

loss��G=� �       �	�2��Xc�A�*

loss2	=@)�       �	�Ѱ�Xc�A�*

loss6��=�6       �	Tq��Xc�A�*

loss#t7>�ڕx       �	���Xc�A�*

loss�C�<à��       �	����Xc�A�*

loss��P=��Q�       �	�E��Xc�A�*

loss��=�vrS       �	���Xc�A�*

loss�!�=�I       �	g|��Xc�A�*

loss��|=�#�       �	�"��Xc�A�*

loss��w=\�       �	Թ��Xc�A�*

loss�|�<�f>�       �	6X��Xc�A�*

loss�Z=a=�t       �	����Xc�A�*

loss3�'=a�       �	���Xc�A�*

loss���<���u       �	3��Xc�A�*

loss u=�.S�       �	��Xc�A�*

loss��=.       �	h���Xc�A�*

lossK==kޟ�       �	�~��Xc�A�*

lossM�V=e�K)       �	�v��Xc�A�*

loss�B�=�{C       �	g���Xc�A�*

loss@Ol=ٷ��       �	U��Xc�A�*

loss�J�=����       �	��Xc�A�*

lossO��=JXL       �	�I��Xc�A�*

lossm�m=�]�       �	���Xc�A�*

lossH\=Bg��       �	�v��Xc�A�*

loss6ţ=��       �	C8��Xc�A�*

loss���=Z�       �	����Xc�A�*

loss��y=R�tT       �	,f��Xc�A�*

loss���<�O1&       �	v���Xc�A�*

loss��<+E       �	���Xc�A�*

loss��=���a       �	���Xc�A�*

loss�z>���E       �	���Xc�A�*

loss&�*=)�|i       �	F���Xc�A�*

loss�,(>3���       �	�J��Xc�A�*

loss��(<�I]       �	����Xc�A�*

loss��<���       �	x~��Xc�A�*

loss��=5#�       �	���Xc�A�*

loss��`=�INx       �	����Xc�A�*

losstr�=�g�       �	�_��Xc�A�*

lossw�=�޶       �	{���Xc�A�*

loss|\�=X��       �	&���Xc�A�*

loss(0=^ J�       �	@��Xc�A�*

loss��I=�!�       �	|���Xc�A�*

loss���=)��       �	Ѐ��Xc�A�*

loss��+=4,�       �	e��Xc�A�*

loss ��=��`�       �	!���Xc�A�*

lossD��<%,0�       �	kJ��Xc�A�*

loss_r=���\       �	���Xc�A�*

loss���=n��#       �	8���Xc�A�*

loss���=I�Ct       �	��Xc�A�*

lossW��=$KSs       �	����Xc�A�*

loss�=���       �	�L��Xc�A�*

loss��=�g5        �	#���Xc�A�*

loss�:G=RZ��       �	p���Xc�A�*

loss�}=hK$�       �	f-��Xc�A�*

lossd�=$�6D       �	"���Xc�A�*

loss���=����       �	�W��Xc�A�*

loss_-'=[Ya�       �	���Xc�A�*

loss��<"���       �	����Xc�A�*

losss0�=2hE�       �	�-��Xc�A�*

loss"�=�-<o       �	���Xc�A�*

loss��< ^��       �	����Xc�A�*

loss�� =N�       �	EH��Xc�A�*

loss�ޖ=�D%-       �	���Xc�A�*

lossQ��<���       �	�{��Xc�A�*

loss,�=>�֗       �	�!��Xc�A�*

loss�X�=��ϰ       �	����Xc�A�*

loss��=/�N5       �	�Q��Xc�A�*

loss7m@=�ܲ�       �	e��Xc�A�*

loss
b�<ܖ�j       �	ɰ��Xc�A�*

loss$��;�#�       �	#N��Xc�A�*

loss��;���o       �	���Xc�A�*

loss��=���.       �	����Xc�A�*

loss�2	<��m       �	/��Xc�A�*

loss�|�<��Yt       �	�_��Xc�A�*

lossA�w<#�{       �	]���Xc�A�*

loss�N�=��X#       �	���Xc�A�*

loss���<�'�8       �	�(��Xc�A�*

loss̧�=���`       �	���Xc�A�*

loss_.>e㋆       �	SZ��Xc�A�*

loss��=���       �	�<��Xc�A�*

loss�=	��G       �	B���Xc�A�*

loss?2�=�       �	�h��Xc�A�*

loss��<�c��       �	 ��Xc�A�*

loss:��=ڻ_       �	6���Xc�A�*

loss4�x=ĳV�       �		4��Xc�A�*

loss�|�=Z=�       �	����Xc�A�*

losss�<�k       �	�r��Xc�A�*

loss��=��       �	=��Xc�A�*

loss=R��       �	���Xc�A�*

loss��o=4י)       �	zT��Xc�A�*

lossf�=��*Q       �	@���Xc�A�*

loss/��<q���       �	l���Xc�A�*

lossd�=�V�       �	�A��Xc�A�*

loss�[4=���
       �	����Xc�A�*

loss͈�<#��       �	����Xc�A�*

loss�m�=m|Ƽ       �	y>��Xc�A�*

loss
��;���       �	<���Xc�A�*

loss�;=+H�z       �	����Xc�A�*

loss3gi=IS       �	�G��Xc�A�*

loss�9=>#�       �	C���Xc�A�*

lossn�V=V��       �	����Xc�A�*

loss�K�=�<��       �	�)��Xc�A�*

lossh�;=:���       �	m���Xc�A�*

loss$k�<sfpT       �	8j��Xc�A�*

loss;�f=���       �	���Xc�A�*

loss�_�=���       �	c���Xc�A�*

loss�<l==F�V       �	�S��Xc�A�*

loss`��=*겡       �	����Xc�A�*

loss���=���K       �	I���Xc�A�*

loss=j>-k�-       �	i��Xc�A�*

loss��=����       �	ݴ��Xc�A�*

loss��,=�'�       �	@L��Xc�A�*

loss���<U���       �	����Xc�A�*

loss/G=�͈>       �	>{��Xc�A�*

loss���=~��       �	���Xc�A�*

loss��i=����       �	����Xc�A�*

loss}�>R�_d       �	����Xc�A�*

loss��]=@��q       �	����Xc�A�*

losssu{=e="�       �	�a��Xc�A�*

loss�KD<B��&       �	l��Xc�A�*

loss�9�=?�v�       �	 ���Xc�A�*

loss/49=�ՖG       �	:���Xc�A�*

loss�K�=��d       �	�j��Xc�A�*

loss_�=�H��       �	Pq �Xc�A�*

loss�2>���       �	=�Xc�A�*

lossm��=�"       �	/��Xc�A�*

loss�=�̵�       �	��Xc�A�*

loss���=��p�       �	�b�Xc�A�*

loss�&=����       �	���Xc�A�*

losss=�e�       �	v��Xc�A�*

lossë�<���7       �	�E�Xc�A�*

loss���<rL��       �	t��Xc�A�*

loss�v�=,	C       �	���Xc�A�*

lossy�<���       �	P��Xc�A�*

loss��=3Y       �	���Xc�A�*

loss]��<���       �	�;	�Xc�A�*

lossLy=� ��       �	��	�Xc�A�*

lossL �<��k       �	\s
�Xc�A�*

loss
d=�Q�/       �	�E�Xc�A�*

lossN�=���%       �	���Xc�A�*

loss�*�=���       �	�z�Xc�A�*

loss�t=���       �	m�Xc�A�*

loss1;=5��0       �	7��Xc�A�*

loss��3>�y       �	��Xc�A�*

loss?��=:v�       �	}�Xc�A�*

loss�Gl=�'[<       �	��Xc�A�*

loss0X�=�y��       �	���Xc�A�*

loss/�(=dq�1       �	F�Xc�A�*

loss�;rn]       �	k��Xc�A�*

loss�M=(@M#       �	�l�Xc�A�*

loss���=M]w[       �	 �Xc�A�*

loss�P�=���2       �	��Xc�A�*

loss_!=;�       �	�Z�Xc�A�*

lossf6=$��h       �	\�Xc�A�*

lossh��<nb�!       �	���Xc�A�*

lossŰ�=��7�       �	���Xc�A�*

loss�]>y	/�       �	R,�Xc�A�*

losspc=_&�       �	���Xc�A�*

lossMF=��#�       �	�r�Xc�A�*

lossv�Q=e᠕       �	�K�Xc�A�*

lossNrS=�¸       �	��Xc�A�*

loss�=ܾ�T       �	]��Xc�A�*

loss3�>��mI       �	X��Xc�A�*

loss���<�5iq       �	3�Xc�A�*

loss��<��{O       �	���Xc�A�*

lossH��=�G�6       �	�k�Xc�A�*

lossNa.=
��       �	��Xc�A�*

loss�Ӈ=�}�       �	���Xc�A�*

losst�	=�       �	�`�Xc�A�*

loss�v<��^�       �	�6 �Xc�A�*

loss�6=���       �	~� �Xc�A�*

loss�]=��!!       �	t`!�Xc�A�*

loss��=2�]D       �	"�Xc�A�*

lossxO�='-D       �	,�"�Xc�A�*

loss��=����       �	�8#�Xc�A�*

loss���=Uyg       �	 �#�Xc�A�*

loss%�<��3       �	�d$�Xc�A�*

loss���<�_��       �	1%�Xc�A�*

loss���<����       �	ظ%�Xc�A�*

loss���=l�&       �	,b&�Xc�A�*

losso��<w蓶       �	K'�Xc�A�*

loss�N=��       �	]�'�Xc�A�*

lossml�=�f�       �	�H(�Xc�A�*

loss��=s{x       �	��(�Xc�A�*

loss��=��o       �	�)�Xc�A�*

loss�O=��Z       �	M�*�Xc�A�*

loss$s�=�u*       �	9�+�Xc�A�*

loss[�^=�mh       �	�G,�Xc�A�*

lossv��=����       �	[�,�Xc�A�*

lossO��=��~�       �	,�-�Xc�A�*

loss@�E=U�/K       �	�n.�Xc�A�*

loss%d�=3� �       �	I/�Xc�A�*

loss�=�)?I       �	!�/�Xc�A�*

lossh�+>q�Z       �	�I0�Xc�A�*

lossd��=c-�       �	q�0�Xc�A�*

loss8��=��R�       �	M�1�Xc�A�*

lossO�<ah�       �	�!2�Xc�A�*

loss���<4��q       �	0�2�Xc�A�*

loss>.��6       �	�p4�Xc�A�*

losst/�=�H�       �	�5�Xc�A�*

lossi�8=wUN       �	��5�Xc�A�*

loss`�6=��|       �	��6�Xc�A�*

loss�-�<��2�       �	l%7�Xc�A�*

loss)�>���>       �	@�7�Xc�A�*

loss&0�= �y       �	$`8�Xc�A�*

loss�V�=K'(�       �	��8�Xc�A�*

lossI�=H9
�       �	��9�Xc�A�*

lossW�O=N&h�       �	�R:�Xc�A�*

loss�Ͽ<�11�       �	�:�Xc�A�*

loss=�Z=��!       �	��;�Xc�A�*

loss��a=UH�Z       �	uZ<�Xc�A�*

loss�9=����       �	=�Xc�A�*

loss��Q=!�UX       �	�=�Xc�A�*

loss�Z[=؞       �	n>�Xc�A�*

loss.F,=�Oށ       �	6?�Xc�A�*

losst7X=kA��       �	��?�Xc�A�*

lossm`)=���i       �	NG@�Xc�A�*

lossq�=M(��       �	X�@�Xc�A�*

loss	j�<���j       �	�xA�Xc�A�*

lossTf<�H�<       �	wB�Xc�A�*

lossg}>a���       �	�B�Xc�A�*

lossx5�=~��0       �	KC�Xc�A�*

loss��=/��       �	1D�Xc�A�*

loss\��=����       �	{�D�Xc�A�*

loss��B<���]       �	�CE�Xc�A�*

loss/Y�=�aG�       �	��E�Xc�A�*

lossi��=e�[V       �	.�F�Xc�A�*

loss�+=�'H       �	&9G�Xc�A�*

loss�<peW�       �	��G�Xc�A�*

loss��s=Ij��       �	׈H�Xc�A�*

loss�)�=#�ՙ       �	
1I�Xc�A�*

loss���<�)       �	��I�Xc�A�*

losseX�<��I       �	+�J�Xc�A�*

loss��>�Q�M       �	�K�Xc�A�*

lossZ>� �a       �	R�K�Xc�A�*

lossZF�=yO�        �	NL�Xc�A�*

loss���=b���       �	X�L�Xc�A�*

loss��&=/�s�       �	;�M�Xc�A�*

losss�<�_K       �	�"N�Xc�A�*

loss�V�=���e       �	��N�Xc�A�*

loss .q=�W��       �	�OO�Xc�A�*

loss;d�<�-�       �	��O�Xc�A�*

loss+@<P/��       �	�zP�Xc�A�*

loss�8�=-c       �	�!Q�Xc�A�*

loss�1Y=�q�0       �	�Q�Xc�A�*

loss�+=��+`       �	�OR�Xc�A�*

loss$�=���       �	��R�Xc�A�*

loss��<�8t       �	5|S�Xc�A�*

loss�S�<[s�       �	�<T�Xc�A�*

loss��;��s       �	C�T�Xc�A�*

loss�\�=�2�,       �	��U�Xc�A�*

loss�3=��v�       �	� V�Xc�A�*

loss}�u=V�K�       �	V�V�Xc�A�*

loss� L>�.�       �	�RW�Xc�A�*

loss�c<�X7G       �	��W�Xc�A�*

loss
�<GJ       �	�X�Xc�A�*

loss1�b=�!0;       �	vY�Xc�A�*

loss�!r=c{Va       �	�dZ�Xc�A�*

loss�Fx<)u       �	�8[�Xc�A�*

lossn=i��       �	\�Xc�A�*

lossL�:>�(�l       �	,�\�Xc�A�*

loss�˓=�@�o       �	�]�Xc�A�*

loss��=t�T�       �	F�^�Xc�A�*

loss�B�=`�7       �	pB_�Xc�A�*

loss�C�=3�       �	��_�Xc�A�*

lossp*
= ���       �	b�`�Xc�A�*

loss��<�A       �	�a�Xc�A�*

lossW��=u���       �	mb�Xc�A�*

lossռ<�*       �	�Rc�Xc�A�*

loss�Ki=���       �	��c�Xc�A�*

lossߵ�=9h,�       �	_�d�Xc�A�*

loss�=^-f�       �	�Pe�Xc�A�*

lossf1<ᢖ�       �	�e�Xc�A�*

loss�=���       �	W�f�Xc�A�*

loss�h~<��'Q       �	�3g�Xc�A�*

lossC��<vi�       �	��g�Xc�A�*

loss� >�F��       �	�{h�Xc�A�*

loss-n"<���)       �	Ci�Xc�A�*

loss.~=��R       �	�i�Xc�A�*

lossH�>�#xP       �	H�j�Xc�A�*

loss�ɋ=\��       �	5bk�Xc�A�*

loss4A�<z�r       �	��k�Xc�A�*

loss�g,<�H�       �	��l�Xc�A�*

loss���<��o       �	NBm�Xc�A�*

loss��=�(V�       �	��m�Xc�A�*

loss1�=�w       �	�n�Xc�A�*

loss��3=�kVR       �	oGo�Xc�A�*

lossR�E=c%�(       �	��o�Xc�A�*

lossV�p=���_       �	�p�Xc�A�*

loss�Yt=�Wj       �	�q�Xc�A�*

loss
.%<do�E       �	V�q�Xc�A�*

loss�=�O�       �	�Rr�Xc�A�*

loss�\�<y]N       �	�s�Xc�A�*

loss�N=�+P�       �	?�s�Xc�A�*

loss�mO=��G       �	Et�Xc�A�*

loss@O%=(�       �	��t�Xc�A�*

loss�Q{=ڭ(       �	 �u�Xc�A�*

loss=�=Df��       �	�Ov�Xc�A�*

loss2��<M,φ       �	��w�Xc�A�*

loss�/�=s��       �	bjx�Xc�A�*

lossk�=�Y�       �	�y�Xc�A�*

loss�	=I��Q       �	ȱy�Xc�A�*

loss�;���       �	P{�Xc�A�*

loss�(=���       �	c�{�Xc�A�*

loss�zc=��z       �	MN|�Xc�A�*

lossᘄ<-Z�1       �	(
}�Xc�A�*

loss�lH=-~�       �	��}�Xc�A�*

loss88z=�       �	�E~�Xc�A�*

loss��=D�2�       �	���Xc�A�*

loss� �=�k|       �	����Xc�A�*

loss�X8=Y#fz       �	]6��Xc�A�*

lossXD�=��;�       �	�>��Xc�A�*

loss	n^<�]l       �	�*��Xc�A�*

loss(4<��٢       �	�̓�Xc�A�*

loss|�J=%�5�       �	,f��Xc�A�*

loss㪝<�IL�       �	��Xc�A�*

loss�9�<�2�       �	����Xc�A�*

lossO�<�s�       �	�7��Xc�A�*

loss ��<�ܙ       �	v݆�Xc�A�*

loss-�'=@��}       �	����Xc�A�*

loss֙�<*��       �	m9��Xc�A�*

loss�	:���       �	Uڈ�Xc�A�*

loss2*�:Բ��       �	:���Xc�A�*

loss`�<N`       �	Ԋ�Xc�A�*

loss��w=S�;       �	$���Xc�A�*

loss�b�<���       �	o��Xc�A�*

lossw�T;��ɰ       �	���Xc�A�*

loss�7=��|       �	먍�Xc�A�*

lossR�x>�j�l       �	<��Xc�A�*

loss\H<�w��       �	�؎�Xc�A�*

lossK/�>˕�k       �	w��Xc�A�*

loss&�=E$U�       �	���Xc�A�*

loss��t=���n       �	U���Xc�A�*

lossf��<x%{       �	9d��Xc�A�*

loss���<5K       �	���Xc�A�*

loss=ID=��f�       �	ﮒ�Xc�A�*

loss_��=�54@       �	aP��Xc�A�*

loss?��=)��       �	���Xc�A�*

loss40?=��*       �	����Xc�A�*

lossI�x=��`       �	�4��Xc�A�*

loss�u>�p       �	lѕ�Xc�A�*

lossI=��6       �	hy��Xc�A�*

loss�*~=��{�       �	S!��Xc�A�*

loss��=r�J       �	�ŗ�Xc�A�*

loss�A�=�m�K       �	�l��Xc�A�*

loss��=j�ų       �	|��Xc�A�*

loss��=hXo�       �	���Xc�A�*

loss�#>/v)       �	nR��Xc�A�*

losst�a=��Β       �	���Xc�A�*

loss2�;�#�       �	,��Xc�A�*

loss_ �<Z�#�       �	~���Xc�A�*

loss�^�=B�L       �	%]��Xc�A�*

loss�W3=%Il       �	i���Xc�A�*

loss��D=Q�7E       �	g���Xc�A�*

loss�3�;H�e�       �	�1��Xc�A�*

lossJ�=A�&N       �	�˟�Xc�A�*

loss2��<�o1       �	#e��Xc�A�*

loss�ơ=�ͷ�       �	���Xc�A�*

loss|��=͠w       �	䢡�Xc�A�*

loss7�&<��U�       �	�I��Xc�A�*

loss}��<忭       �	�c��Xc�A�*

loss�(�=�Qƞ       �	¤�Xc�A�*

lossW��;�V]�       �	W]��Xc�A�*

loss��u<8G�.       �	����Xc�A�*

loss�0�<ʛ�T       �	����Xc�A�*

loss��<#��       �	�-��Xc�A�*

lossZ6�=J���       �	|ӧ�Xc�A�*

lossQ�=U���       �	,���Xc�A�*

losssɮ=y��H       �	m��Xc�A�*

loss�I{<+�5�       �	{.��Xc�A�*

loss�z=��8       �	���Xc�A�*

loss=�,�S       �	/���Xc�A�*

lossԊ�=���!       �	�C��Xc�A�*

loss� < �       �	��Xc�A�*

lossڽ$=Ƒ�       �	�}��Xc�A�*

loss
۸=A2       �	Y��Xc�A�*

loss��<[3��       �	 ���Xc�A�*

loss��=��5       �	�E��Xc�A�*

loss�|=�U7       �	A��Xc�A�*

loss��<��]       �	����Xc�A�*

losst�=����       �	8���Xc�A�*

loss]��=��q       �	����Xc�A�*

loss�@�=��3       �	�g��Xc�A�*

loss�UO=�G�\       �	5��Xc�A�*

loss$�=$m�       �	����Xc�A�*

loss�T�<�JW�       �	'K��Xc�A�*

loss6=2���       �	����Xc�A�*

loss-=C��       �	���Xc�A�*

loss���=�+       �	��Xc�A�*

loss�U�=eR�n       �	����Xc�A�*

loss}L�<?���       �	�W��Xc�A�*

loss\�H=��$$       �	����Xc�A�*

lossK�=�|s[       �	����Xc�A�*

loss�:�=�[��       �	�1��Xc�A�*

lossέ=���       �	O���Xc�A�*

lossw=o�       �	we��Xc�A�*

loss��T<��l�       �	;���Xc�A�*

loss��<�k       �	���Xc�A�*

loss�=t���       �	M-��Xc�A�*

loss��>]!�C       �	;���Xc�A�*

lossW;Y=A_f�       �	�\��Xc�A�*

lossJ��=ƕ��       �	����Xc�A�*

loss�t=�?#+       �	-���Xc�A�*

loss�*�=+�       �	|F��Xc�A�*

loss6�=���       �	����Xc�A�*

loss Ҏ<���        �	�z��Xc�A�*

loss��I=
j�n       �	���Xc�A�*

loss�<a<�}�A       �	ѯ��Xc�A�*

lossOS�<w���       �	�F��Xc�A�*

loss\j�<���       �	����Xc�A�*

loss`�=J���       �	n��Xc�A�*

loss<��=��]       �	6��Xc�A�*

loss�Ѫ<ʅ-       �	s���Xc�A�*

loss���=���z       �	T8��Xc�A�*

loss��=�T�       �	����Xc�A�*

loss�Z=��&�       �	�y��Xc�A�*

lossM�=H\�	       �	
��Xc�A�*

lossP��=����       �	����Xc�A�*

loss��x>�]ޑ       �	^G��Xc�A�*

loss��=٠��       �	����Xc�A�*

loss��;=�y�       �	����Xc�A�*

loss +�<W��*       �	��Xc�A�*

loss���<[���       �	?���Xc�A�*

loss�s]=�u�H       �	�K��Xc�A�*

lossƞ�=�J�       �	1	��Xc�A�*

loss�=�^('       �	^���Xc�A�*

loss	3=<��r       �	3��Xc�A�*

loss\��<v�IT       �	2���Xc�A�*

loss=�+	       �	�e��Xc�A�*

loss�m'<6eS�       �	����Xc�A�*

loss�G�<v`�[       �	h���Xc�A�*

loss��"=��(       �	S?��Xc�A�*

loss��<��J       �	����Xc�A�*

loss�>{       �	�w��Xc�A�*

loss�=���       �	���Xc�A�*

lossOYT<0��       �	>���Xc�A�*

loss�"�;�eOL       �	�H��Xc�A�*

loss�n�;uV��       �	����Xc�A�*

loss�D�=�D	       �	!u��Xc�A�*

loss᭔=^��       �	]��Xc�A�*

loss�Y>���       �	����Xc�A�*

loss!'�<�5       �	
M��Xc�A�*

loss�pi;�h�       �	v���Xc�A�*

loss�[p=��       �	����Xc�A�*

loss*��<��Hs       �	3��Xc�A�*

loss��=��k`       �	h#��Xc�A�*

lossR�W=��N�       �	~p��Xc�A�*

loss�V=�F�       �	���Xc�A�*

loss��=��H       �	X���Xc�A�*

loss��=��.       �	����Xc�A�*

losse.='�M       �	���Xc�A�*

loss+�='�1�       �	����Xc�A�*

loss��=t��       �	8e �Xc�A�*

loss_p�<��       �	�~�Xc�A�*

lossRE<�GM       �	,e�Xc�A�*

lossU�=d;�       �	��Xc�A�*

lossC�=
�Ĉ       �	���Xc�A�*

lossݤh=�cu&       �	���Xc�A�*

lossAY=Po�q       �	�_�Xc�A�*

loss��=�ќ�       �	��Xc�A�*

lossT(q=���b       �	���Xc�A�*

loss�d�=O@�?       �	���Xc�A�*

loss��X=�c+�       �	R��Xc�A�*

lossJ�<"[ȓ       �	L	�Xc�A�*

lossO<��       �	�	�Xc�A�*

loss�y�<���E       �	�>
�Xc�A�*

loss�)�=s֋�       �	��
�Xc�A�*

loss` G=��9h       �	���Xc�A�*

loss&f
=���       �	�M�Xc�A�*

loss���<ͦ5'       �	��Xc�A�*

loss���<ʏC       �	^��Xc�A�*

loss�^�<��~       �	�)�Xc�A�*

loss���=#�b�       �	:�Xc�A�*

lossF*�=&=       �	6��Xc�A�*

loss�n�=���d       �	fK�Xc�A�*

lossL�\<�FЁ       �	B��Xc�A�*

loss쬒=��.�       �	+��Xc�A�*

loss��<*ɝ       �	P�Xc�A�*

lossq>���_       �	���Xc�A�*

loss�
=���       �	�L�Xc�A�*

loss�A�<�7�       �	���Xc�A�*

loss��]=̍�       �	9��Xc�A�*

loss�w=,�P�       �	�4�Xc�A�*

loss�ӛ<��       �	'��Xc�A�*

loss<�=�R�       �	�r�Xc�A�*

loss@�P>[���       �	#�Xc�A�*

loss���<K�M9       �	��Xc�A�*

lossQO=�K�       �	f�Xc�A�*

loss�֗= ��*       �	�Xc�A�*

loss���<��P�       �	c��Xc�A�*

loss�]�=�WU       �	;��Xc�A�*

loss�=K<5��"       �	"�Xc�A�*

loss��<$V       �	x��Xc�A�*

loss� �<k��P       �	Q�Xc�A�*

loss�5=�G       �	&��Xc�A�*

loss�ب=�/}       �	)��Xc�A�*

loss�<�l|       �	A*�Xc�A�*

loss8ԓ=~�@�       �	��Xc�A�*

loss��y=N�hR       �	���Xc�A�*

loss;=�%       �	�& �Xc�A�*

loss���=�C �       �	�� �Xc�A�*

loss~�=��p       �	�U!�Xc�A�*

loss��=C�       �	��!�Xc�A�*

lossI��<�	�@       �	3�"�Xc�A�*

loss�G<����       �	>#�Xc�A�*

loss���=�/nd       �	o�#�Xc�A�*

loss�W�=Lï�       �	m$�Xc�A�*

lossa�=ٵ�D       �	p	%�Xc�A�*

loss�F=���       �	��%�Xc�A�*

loss*��=-_!�       �	QO&�Xc�A�*

loss�S�<w�hV       �	(�&�Xc�A�*

loss͜u=�W�z       �	��'�Xc�A�*

loss��=A��       �	�U(�Xc�A�*

loss��<���       �	>�(�Xc�A�*

loss��=��B       �	 �)�Xc�A�*

loss��k=�ho       �	�:*�Xc�A�*

lossr�>t�-       �	��*�Xc�A�*

lossn��<S7Gp       �	Su+�Xc�A�*

loss�V<kf.�       �	,�Xc�A�*

loss)Z�<�~��       �	��,�Xc�A�*

loss�3=����       �	�W-�Xc�A�*

loss��Z<Km�       �	z�-�Xc�A�*

lossg��=Z���       �	�.�Xc�A�*

loss���=�F�4       �	�,/�Xc�A�*

lossx��=j��       �	��/�Xc�A�*

lossK�<&��       �	�h0�Xc�A�*

loss8�A=�V3e       �	_1�Xc�A�*

lossآ�;�؃�       �	+�1�Xc�A�*

lossa�C<S� �       �	�?2�Xc�A�*

loss�=�>��       �	��2�Xc�A�*

loss��<���       �	{3�Xc�A�*

loss���<�Z��       �	�4�Xc�A�*

loss�'x=�8�;       �	Ժ4�Xc�A�*

loss�@�=C��)       �	h]5�Xc�A�*

loss$1q=<2��       �	M�5�Xc�A�*

loss���<.nY       �	g�6�Xc�A�*

lossio=s�!a       �	�/7�Xc�A�*

loss�̖=b�=7       �	��7�Xc�A�*

loss,*k="��       �	�f8�Xc�A�*

loss6�h<�K�       �	/�8�Xc�A�*

loss&�==�)j       �	u�9�Xc�A�*

loss�ݮ<9z�l       �	�%:�Xc�A�*

lossf�g=Lǥ       �	3�:�Xc�A�*

losss!U;	:#       �	�;�Xc�A�*

loss䩕=��       �	
H<�Xc�A�*

loss���<���       �	;�<�Xc�A�*

lossE}�<�<m       �	��=�Xc�A�*

loss%-�<��-       �	u>�Xc�A�*

lossvw�<WR�       �	��>�Xc�A�*

losshL=��       �	K?�Xc�A�*

loss���=�8�4       �	��?�Xc�A�*

lossim�<�٠�       �	�@�Xc�A�*

loss��<o��y       �	zoA�Xc�A�*

loss���=.�'       �	B�Xc�A�*

lossK�;[�Y�       �	d�B�Xc�A�*

loss_l?=f�Ҁ       �	VGC�Xc�A�*

loss���=��l}       �	'�C�Xc�A�*

loss�&�=���       �	��D�Xc�A�*

losszWO<d�g�       �	E�Xc�A�*

lossm��<�ǌ�       �	�E�Xc�A�*

losso��<���X       �	~UF�Xc�A�*

loss�%=�$�       �	��F�Xc�A�*

loss��<)�       �	��G�Xc�A�*

loss���<��       �	�3H�Xc�A�*

loss���<z��       �	+�H�Xc�A�*

loss;<4=O�o�       �	�I�Xc�A�*

loss�ƿ<w�\�       �	W=J�Xc�A�*

loss-5<|��N       �	��J�Xc�A�*

loss��[=�bǝ       �	L�Xc�A�*

loss%*e=�Ku�       �	MM�Xc�A�*

loss8��=b�       �	�M�Xc�A�*

loss�uT=ߧ��       �	�cN�Xc�A�*

lossH%*=�+&�       �	�O�Xc�A�*

loss�}�<�       �	��O�Xc�A�*

loss�;�;��p       �	�TP�Xc�A�*

lossF+�<)(�       �	��P�Xc�A�*

loss��>=[       �	u�Q�Xc�A�*

loss�h�= $�       �	4R�Xc�A�*

loss�>e�2       �	6�R�Xc�A�*

loss�5!<�B:       �	�pS�Xc�A�*

loss�I=� n       �	DT�Xc�A�*

loss`�3=��t       �	s�T�Xc�A�*

loss���<<�g       �	mU�Xc�A�*

loss�z�=�`��       �	�V�Xc�A�*

losswa=9�o`       �	�W�Xc�A�*

loss#|=��b       �	��W�Xc�A�*

loss|q�=�       �	$aX�Xc�A�*

lossfA�<oZ;       �	�$Y�Xc�A�*

loss��=HoII       �	r�Y�Xc�A�*

loss�n�<�Q�       �	�{Z�Xc�A�*

loss6a�<H��z       �	=[�Xc�A�*

lossj_=�RŠ       �	��[�Xc�A�*

lossE��<N�ù       �	{\�Xc�A�*

loss���<8���       �	]�Xc�A�*

loss*��=��_�       �	/�]�Xc�A�*

loss���=Tҟ�       �	�^�Xc�A�*

loss�,>�^       �	$+_�Xc�A�*

loss�>�8t\       �	Y�_�Xc�A�*

loss���=�#�       �	|e`�Xc�A�*

lossJ��=�J�       �	n�`�Xc�A�*

loss�1=�P}�       �	<�a�Xc�A�*

lossib�='9�i       �	e8b�Xc�A�*

loss
Bt=��v       �	F�b�Xc�A�*

loss;zf=�{��       �	q�c�Xc�A�*

loss��;;zs�       �	�^d�Xc�A�*

loss��=�ړp       �	Y�d�Xc�A�*

loss�ʝ=��       �	��e�Xc�A�*

loss���=:!�       �	�0f�Xc�A�*

loss��< ��v       �	G�f�Xc�A�*

loss�g=�&�       �	�rg�Xc�A�*

loss�
+<��u#       �	�h�Xc�A�*

lossn��;�N�?       �	O�h�Xc�A�*

loss�h=�mz1       �	�Ei�Xc�A�*

loss��!=�l�       �	3�i�Xc�A�*

loss��d=��.l       �	��j�Xc�A�*

loss�I>8�'       �	 k�Xc�A�*

loss���=�+       �	��k�Xc�A�*

loss5܃=U�O       �	�Ol�Xc�A�*

loss\>�5E�       �	�l�Xc�A�*

loss&1<s:       �	�m�Xc�A�*

loss��=�d�       �	�+n�Xc�A�*

lossi��<6���       �	��n�Xc�A�*

loss���=�v��       �	�jo�Xc�A�*

loss�(�=���       �	�p�Xc�A�*

loss��=����       �	-�p�Xc�A�*

lossvC=8U؋       �	�eq�Xc�A�*

loss���;z<�       �	Sr�Xc�A�*

loss ��<�GC�       �	�r�Xc�A�*

loss#�=�h'       �	�9s�Xc�A�*

loss�0�=����       �	�s�Xc�A�*

loss��=�V�$       �	?�t�Xc�A�*

lossl�>s�R�       �	�#u�Xc�A�*

loss�=�=6�B�       �	��u�Xc�A�*

lossVP=r֜-       �	�rv�Xc�A�*

loss@*i;33       �	�w�Xc�A�*

loss}J�<$s"�       �	o�w�Xc�A�*

loss;��<��w       �	�ax�Xc�A�*

loss+�<VRdq       �	��x�Xc�A�*

lossf=�;�       �	��y�Xc�A�*

loss��X=��b6       �	�pz�Xc�A�*

lossa3�<�u�        �	J{�Xc�A�*

loss�~�=js�       �	��{�Xc�A�*

lossF��=}D�`       �	�_|�Xc�A�*

lossS�9=�훐       �	��|�Xc�A�*

loss��%=�o�N       �	�}�Xc�A�*

loss�X�;��q       �	}A~�Xc�A�*

lossl��=��s�       �	��~�Xc�A�*

lossl�R=նid       �	j��Xc�A�*

loss
�=q�e�       �	���Xc�A�*

lossW�g=���       �	����Xc�A�*

loss��<%�       �	�[��Xc�A�*

loss��g=jU�       �	@���Xc�A�*

loss�;�=;JP�       �	����Xc�A�*

lossj-�<(ν�       �	�?��Xc�A�*

loss�=QF�.       �	_��Xc�A�*

loss��;=�|�       �	����Xc�A�*

loss�-�<ȿQ�       �	�D��Xc�A�*

lossab�;�M�       �	^م�Xc�A�*

loss�С=>/��       �	ut��Xc�A�*

loss,հ=�;       �	��Xc�A�*

lossLA�;"��C       �	����Xc�A�*

loss`T�<i�Ӕ       �	I��Xc�A�*

loss�ʞ<C:+       �	���Xc�A�*

loss��	=,�l$       �	���Xc�A�*

loss�?=upX       �	���Xc�A�*

loss=M�=��#       �	Զ��Xc�A�*

loss)K<="",       �	;T��Xc�A�*

loss���<���       �	[��Xc�A�*

loss�˞=r0��       �	b���Xc�A�*

loss���<��       �	���Xc�A�*

loss��W;�h�q       �	g���Xc�A�*

loss9A=OaU^       �	^L��Xc�A�*

lossZ3b<G\k       �	e��Xc�A�*

loss���;>ɏ�       �	5���Xc�A�*

loss�;q=�-�       �	oG��Xc�A�*

loss��P=ȕ�~       �	�ߐ�Xc�A�*

loss���;`��       �	�x��Xc�A�*

loss
yr=+�iW       �	,��Xc�A�*

loss���=�/�(       �	C���Xc�A�*

loss�ئ=ٲ�I       �	O��Xc�A�*

loss��=�5       �	a���Xc�A�*

loss`+�<�pN       �	ݳ��Xc�A�*

loss��K<��B�       �	W��Xc�A�*

lossz:c=�T�       �	]���Xc�A�*

losso�<4�!�       �	ß��Xc�A�*

loss&�2>�@Iw       �	�<��Xc�A�*

loss�b�;�q�       �	�ԗ�Xc�A�*

loss5�=U�zU       �	�l��Xc�A�*

loss6�r<��_       �	i;��Xc�A�*

loss�Ah=��h       �	Mؙ�Xc�A�*

loss�A�<���       �	�Ú�Xc�A�*

loss�Q=,D�J       �	)[��Xc�A�*

loss�F=l=}       �	`��Xc�A�*

lossX;�<���       �	���Xc�A�*

loss�u)=e|       �	�>��Xc�A�*

lossY�=N��       �	��Xc�A�*

loss�r=�qd       �	Z���Xc�A�*

lossnh<���       �	:��Xc�A�*

loss��=��[       �	����Xc�A�*

lossvf�<��"       �	�S��Xc�A�*

lossqw=�gA       �	>��Xc�A�*

loss�&>8��j       �	���Xc�A�*

lossHY"=�]�m       �	�R��Xc�A�*

loss�I�;�+@       �	���Xc�A�*

lossO��<�sU       �	į��Xc�A�*

loss�7�=
��       �	QK��Xc�A�*

lossw(!=Rh>       �	 ��Xc�A�*

losso=?=B��       �	P���Xc�A�*

loss��=Z3O�       �	+��Xc�A�*

loss�k>���       �	$Ѧ�Xc�A�*

lossڌ�<c��       �	6s��Xc�A�*

lossD0<9�)7       �	t��Xc�A�*

lossCqp<ɀ       �	.���Xc�A�*

loss�=c��       �	�E��Xc�A�*

loss�>��(�       �	��Xc�A�*

loss�v=��,       �	=���Xc�A�*

lossX��=�+�K       �	�=��Xc�A�*

loss�33=~;�       �	�ܫ�Xc�A�*

loss�[�=�yn<       �	`r��Xc�A�*

losspS<�7�       �	E��Xc�A�*

loss��<��b       �	z���Xc�A�*

loss�=��       �	u?��Xc�A�*

loss}�<���"       �	�׮�Xc�A�*

lossZF{<����       �	1���Xc�A�*

loss|�'>n�       �	�Y��Xc�A�*

lossN��=��)       �	���Xc�A�*

lossFf�<-�x8       �	ؚ��Xc�A�*

loss��=:�~�       �	�<��Xc�A�*

lossl�~=Blv"       �	ײ�Xc�A�*

loss��=��!�       �	�~��Xc�A�*

losso6�<��"�       �		��Xc�A�*

loss|*6=����       �	մ��Xc�A�*

loss��/=ji^�       �	�N��Xc�A�*

loss]�,=��G       �	f��Xc�A�*

loss;:�=�cP       �	g���Xc�A�*

lossօ�<)���       �	wM��Xc�A�*

loss��;�#I�       �	B���Xc�A�*

loss�W=�b�b       �	����Xc�A�*

lossD��=�ʋ       �	�.��Xc�A�*

loss��=��A       �	iȹ�Xc�A�*

loss��<U9U       �	�h��Xc�A�*

lossc��<C��k       �	X��Xc�A�*

loss�I=��By       �	9һ�Xc�A�*

loss=d�=0��       �	ݴ��Xc�A�*

lossP�>�H��       �		��Xc�A�*

lossV��<�eڞ       �	����Xc�A�*

loss�j�<�ΊU       �	�P��Xc�A�*

loss&2�<P!]|       �	���Xc�A�*

lossQ�z<f�5       �	Y���Xc�A�*

loss�ĺ<���       �	J%��Xc�A�*

loss�J=��PP       �	P���Xc�A�*

loss�Q<�f9\       �	WZ��Xc�A�*

lossh��=�t       �	^���Xc�A�*

loss��=7J�       �	���Xc�A�*

loss3rr=͢s       �	-��Xc�A�*

loss�?�=��>       �	���Xc�A�*

lossz�|=>�F�       �	���Xc�A�*

loss��t=\u�=       �	#-��Xc�A�*

loss@�\=	[c�       �	1���Xc�A�*

loss�s�=�ޓD       �	�|��Xc�A�*

loss�J)=X;�R       �	���Xc�A�*

loss�=��(�       �	y���Xc�A�*

lossȏ=~���       �	�M��Xc�A�*

loss�Y<{,�5       �	����Xc�A�*

loss���;GCe�       �	���Xc�A�*

lossEU�<���d       �	A,��Xc�A�*

loss�
H=��Y       �	"���Xc�A�*

loss�Ot=V�       �	E���Xc�A�*

loss$-=9�d       �	r1��Xc�A�*

loss�==y5��       �	����Xc�A�*

loss�=�Gi�       �	�z��Xc�A�*

loss��U=���F       �	W���Xc�A�*

lossCa<[�x�       �	�\��Xc�A�*

loss(�<=�,�       �	���Xc�A�*

loss�\/=CO��       �	d>��Xc�A�*

loss���=�6d*       �	���Xc�A�*

lossst�;��       �	����Xc�A�*

loss_��<
+��       �	���Xc�A�*

lossu�<�d�       �	�h��Xc�A�*

loss=U�Z       �	5��Xc�A�*

loss�<f�       �	����Xc�A�*

loss�=Gx0�       �	:X��Xc�A�*

loss�;�=��B�       �	D���Xc�A�*

loss��T<�CK_       �	���Xc�A�*

loss��=�*1�       �	�5��Xc�A�*

loss"��=LF��       �	$���Xc�A�*

loss�d�=�}�4       �	�x��Xc�A�*

loss�h(=�˳G       �	'��Xc�A�*

loss��=�       �	ޫ��Xc�A�*

loss��=�b�       �	$H��Xc�A�*

loss��G=���       �	%��Xc�A�*

loss��y=����       �	���Xc�A�*

loss= �<LA�       �	:��Xc�A�*

loss�|=zZB�       �	����Xc�A�*

loss��=��ST       �	|��Xc�A�*

loss]Kr=&��       �	v���Xc�A�*

loss[q�<���       �	ZK��Xc�A�*

loss���;ߨ`g       �	����Xc�A�*

loss^B�=
Kt       �	���Xc�A�*

lossQ/;=i�/       �	�)��Xc�A�*

loss�F�<�`4       �	����Xc�A�*

loss!߉=���       �	fh��Xc�A�*

lossN��<�+�       �	u��Xc�A�*

loss�u(>>1�0       �	����Xc�A�*

lossv�=�~��       �	�E��Xc�A�*

loss���=��[�       �	L���Xc�A�*

lossl|=o�q       �	�~��Xc�A�*

lossQ=,�       �	.��Xc�A�*

loss#'9=߮�O       �	����Xc�A�*

loss�R�<ȅ��       �	�U��Xc�A�*

lossa�2=R�x�       �	r���Xc�A�*

loss��D=c�+       �	a���Xc�A�*

lossh9*=m��       �	S=��Xc�A�*

losst!�<�/�       �	����Xc�A�*

loss!�W=A�I�       �	�y��Xc�A�*

loss#��<E�       �	��Xc�A�*

loss��=�ҋ       �	����Xc�A�*

loss�m�;��E       �	}@��Xc�A�*

loss�</��       �	����Xc�A�*

loss
�<� ^^       �	qv��Xc�A�*

loss��=}��       �	���Xc�A�*

loss�LY=e3�H       �	���Xc�A�*

loss�<����       �	����Xc�A�*

lossX>%�       �	�%��Xc�A�*

lossy�<���}       �	����Xc�A�*

loss�[�<"D��       �	__��Xc�A�*

loss.S�>���N       �	���Xc�A�*

loss܉=f �Q       �	Ĕ��Xc�A�*

lossCR�<�d�       �	�>��Xc�A�*

lossx�<0zV�       �	����Xc�A�*

loss��=�e�x       �	`���Xc�A�*

loss[��;�9�       �	RH��Xc�A�*

loss�A<�/�       �	7���Xc�A�*

loss�2�=�{��       �	W{��Xc�A�*

loss��=���       �	Y��Xc�A�*

loss���=PC�p       �	����Xc�A�*

loss43]<��q�       �	����Xc�A�*

loss���<Bw�`       �	U���Xc�A�*

loss�޴<��lV       �	���Xc�A�*

loss�F�=��h�       �	/1��Xc�A�*

lossf�<�햻       �	1���Xc�A�*

loss��<���m       �	�]�Xc�A�*

loss��<�n�H       �	�Xc�A�*

loss�4>��)       �	���Xc�A�*

loss;�<�2C       �	�Q�Xc�A�*

loss���<�E+�       �	���Xc�A�*

loss�==�B-       �	m��Xc�A�*

loss�<���W       �	r2�Xc�A�*

loss�$&==ꃡ       �	s��Xc�A�*

loss�@v;�c��       �	�r�Xc�A�*

lossr��=�z�       �	x�Xc�A�*

lossJ��<w�=�       �	K��Xc�A�*

lossv�h=��O�       �	�a�Xc�A�*

loss��=7檠       �	��Xc�A�*

loss�)^=��       �	a�	�Xc�A�*

loss �
=���       �	>
�Xc�A�*

lossUt;Y��)       �	��
�Xc�A�*

loss�9=�H^       �	�r�Xc�A�*

lossc�E;`CV�       �	@�Xc�A�*

loss�7>����       �	���Xc�A�*

loss��=KO4       �	T�Xc�A�*

loss�~�=;S�       �	^��Xc�A�*

loss2�w<���       �	��Xc�A�*

loss���=<<�       �	t��Xc�A�*

loss>c�=�&�       �	�-�Xc�A�*

loss)�<z.z       �	���Xc�A�*

loss@c<�\s�       �	�s�Xc�A�*

loss���=�Θ�       �	��Xc�A�*

loss֮a<l='       �	��Xc�A�*

lossHA=�_=       �	�\�Xc�A�*

lossɛ0=I5�       �	���Xc�A�*

loss@+�=)j!O       �	���Xc�A�*

loss���<��{       �	�6�Xc�A�*

lossZ'�<��N       �	p��Xc�A�*

loss6I=B}�       �	�i�Xc�A�*

loss.�J=�0�       �	��Xc�A�*

loss��=�0q       �	'��Xc�A�*

loss}�;�"��       �	�E�Xc�A�*

lossJ=ԻWO       �	���Xc�A�*

loss��]=+M3@       �	.��Xc�A�*

loss��R=�^��       �	�&�Xc�A�*

loss�5�<^iI�       �		��Xc�A�*

loss�r;�{�       �	`�Xc�A�*

loss�P4<��۳       �	���Xc�A�*

loss;ua=Fk       �	���Xc�A�*

loss�+=���=       �	4��Xc�A�*

loss`��=l�       �	S<�Xc�A�*

lossmKW=#c       �	���Xc�A�*

lossȗ=d��;       �	��Xc�A�*

loss&S2=�qV       �	z �Xc�A�*

loss�� <kNa�       �	�"�Xc�A�*

loss��y=u+ެ       �	Ů"�Xc�A�*

loss��<U:��       �	'P#�Xc�A�*

loss�<Js�Q       �	{�#�Xc�A�*

loss�J<�a�M       �	��$�Xc�A�*

loss/�=����       �	7Q%�Xc�A�*

lossVGI=`��       �	s�%�Xc�A�*

lossW��<����       �	��&�Xc�A�*

loss�=�M��       �	��'�Xc�A�*

loss#�<_��       �	�K(�Xc�A�*

loss&��<\�D5       �	��(�Xc�A�*

loss��=ͯ �       �	>�)�Xc�A�*

loss��;gt@�       �	{1*�Xc�A�*

lossd�|<]-B�       �	��*�Xc�A�*

loss�=̃��       �	0c+�Xc�A�*

loss�?=���       �	��+�Xc�A�*

loss�c�<Ť[       �	��,�Xc�A�*

loss^H=i�P�       �	�c-�Xc�A�*

loss�B�=�|��       �	��-�Xc�A�*

loss��^=	߮       �	h�.�Xc�A�*

loss�>P=F%k�       �	H4/�Xc�A�*

loss[��=�2�W       �	��/�Xc�A�*

losse�M<�- 	       �	�d0�Xc�A�*

lossɐ&=;�       �	�1�Xc�A�*

lossq�q<��|       �	N�1�Xc�A�*

losszZ�;�qya       �	32�Xc�A�*

losstNI;Le�]       �	��2�Xc�A�*

lossj!<����       �	ܛ3�Xc�A�*

loss�j�=���       �	P44�Xc�A�*

loss�_;��F       �	��4�Xc�A�*

loss_wI;%DH       �	zm5�Xc�A�*

lossM�3:��Й       �	.6�Xc�A�*

loss봕9�H�-       �	��6�Xc�A�*

loss==
V��       �	IJ7�Xc�A�*

losshXL=���       �	��7�Xc�A�*

loss,��;b ��       �	��8�Xc�A�*

loss9.<.���       �	�P9�Xc�A�*

loss��=�yd       �	B�9�Xc�A�*

loss\�I>���       �	
�:�Xc�A�*

loss�<_�s�       �	;�Xc�A�*

loss�>l� }       �	��;�Xc�A�*

loss��<���       �	��<�Xc�A�*

loss���=�m�       �	7n=�Xc�A�*

loss�,y=��[       �	g>�Xc�A�*

lossω5=��       �	d\?�Xc�A�*

loss��=2�};       �	�-@�Xc�A�*

loss�ٚ=͛?�       �	u�@�Xc�A�*

loss�=B=)�t       �	W{A�Xc�A�*

loss�x�=�F�_       �	}#B�Xc�A�*

loss��
=�^ۡ       �	T�B�Xc�A�*

loss��V=Ԓ��       �	^cC�Xc�A�*

loss�z=Վ�S       �	�	D�Xc�A�*

loss&ZQ=�w�@       �	�D�Xc�A�*

loss)�q=a�I~       �	6=E�Xc�A�*

loss�m�=Q�F4       �	��E�Xc�A�*

lossq�<��7z       �	PsF�Xc�A�*

loss(��<\�Ke       �	oG�Xc�A�*

loss]�-=�i�3       �	e�G�Xc�A�*

loss��=� �3       �	ADH�Xc�A�*

loss�h�<a�%        �	��H�Xc�A�*

loss��=v [       �	�vI�Xc�A�*

lossS� >]K5G       �	AJ�Xc�A�*

loss�<�6�o       �	�J�Xc�A�*

loss��<?�UZ       �	�IK�Xc�A�*

loss�R<��D       �	�K�Xc�A�*

loss� �<mK�       �	�|L�Xc�A�*

loss�^�<Փkw       �	� M�Xc�A�*

lossB��=V#9       �	�M�Xc�A�*

lossm�0=�:�       �	�PN�Xc�A�*

lossE<c}b�       �	u�N�Xc�A�*

loss�Ԝ=z	aD       �	�}O�Xc�A�*

losseH�<��[       �	PP�Xc�A�*

loss�\;lǬ�       �	A�P�Xc�A�*

loss䭝<O��?       �	PPQ�Xc�A�*

loss1��<��jJ       �	��Q�Xc�A�*

loss���<C��       �	�R�Xc�A�*

loss�:h=q&�       �	V-S�Xc�A�*

loss$��=����       �	�T�Xc�A�*

lossN�`=��N       �	��T�Xc�A�*

lossF�<�R�       �	�EU�Xc�A�*

lossWb)<׊�       �	��U�Xc�A�*

loss��=��4       �	�qV�Xc�A�*

loss$]="g�&       �	0W�Xc�A�*

lossjʦ<�-�       �	B�W�Xc�A�*

loss���=��'�       �	�gX�Xc�A�*

loss]�=Ga��       �	��X�Xc�A�*

losso��<��       �	=�Y�Xc�A�*

loss�=�m��       �	l?Z�Xc�A�*

loss���<�q��       �	��Z�Xc�A�*

loss���<	J       �	N[�Xc�A�*

lossVЙ=ލ�        �	��r�Xc�A�*

loss���=6ɏ�       �	8.s�Xc�A�*

loss��=^�D       �	��s�Xc�A�*

lossE��<��s       �	�[t�Xc�A�*

loss}�=ޙ��       �	o�t�Xc�A�*

loss�u<(,       �	��u�Xc�A�*

loss�ǲ<V�       �	�Kv�Xc�A�*

loss��<�       �	`�v�Xc�A�*

lossڝ�=�d��       �	�w�Xc�A�*

loss�Y�=��7�       �	x�Xc�A�*

lossH(�<�@,       �	-�x�Xc�A�*

lossc�=��xl       �	�Py�Xc�A�*

loss_��<�ܙ�       �	� z�Xc�A�*

loss*�<m �       �	�z�Xc�A�*

loss�Qd<����       �	eV{�Xc�A�*

lossr��=K8ĩ       �	��{�Xc�A�*

loss�k.;9)�e       �	N�|�Xc�A�*

loss8)=�i�g       �	�N}�Xc�A�*

loss�ӡ<~�<�       �	�}�Xc�A�*

loss���=��pZ       �	�~�Xc�A�*

lossU��<���       �	��Xc�A�*

loss�"�=�9�       �	z�Xc�A�*

loss,>�;l��       �	����Xc�A�*

loss��=�;kP       �	�L��Xc�A�*

loss��<�ڿ�       �	9��Xc�A�*

loss�x�<�b�       �	L���Xc�A�*

lossN=-�x3       �	�-��Xc�A�*

loss�\=y@�       �	�Ǆ�Xc�A�*

loss�>�<`�       �	}��Xc�A�*

loss��Z<U�:�       �	���Xc�A�*

loss��<� �       �	r���Xc�A�*

lossG�=���       �	XV��Xc�A�*

loss<$=9�k�       �	���Xc�A�*

loss��=�5��       �	솈�Xc�A�*

loss���=i��       �	�T��Xc�A�*

loss�f�<g,O�       �	����Xc�A�*

loss��V<HFC�       �	����Xc�A�*

loss�=+]�        �	a6��Xc�A�*

loss��=6>0�       �	{��Xc�A�*

loss�C�=�Z8�       �	����Xc�A�*

loss�9�<.	�       �	�N��Xc�A�*

loss���<��7�       �	9��Xc�A�*

loss��:<%9��       �	���Xc�A�*

loss�c�<�QlZ       �	�Xc�A�*

loss#�<2��       �	�D��Xc�A�*

lossϏ�=)S�       �	�ݐ�Xc�A�*

loss�j=�Uy�       �	ut��Xc�A�*

loss��=��X       �	���Xc�A�*

loss�($=�f��       �	S���Xc�A�*

loss�$b;kO��       �	�I��Xc�A�*

loss���;�	��       �	&��Xc�A�*

loss��=/��       �	����Xc�A�*

loss�i�<�x^�       �	�?��Xc�A�*

loss8i#>��t       �	�ؕ�Xc�A�*

lossnz�<.&�       �	�q��Xc�A�*

lossl�;�]       �	�	��Xc�A�*

loss��1<�NL       �	ﭗ�Xc�A�*

loss���;�F�       �	�D��Xc�A�*

loss�n�;� �       �	e��Xc�A�*

losswܬ=���       �	5~��Xc�A�*

loss���=��+�       �	��Xc�A�*

loss��6=~u��       �	�7��Xc�A�*

loss*�*;G�7       �	�Λ�Xc�A�*

lossƷp=>fTe       �	�m��Xc�A�*

loss���<J�7C       �	�$��Xc�A�*

loss��;d���       �	�ʝ�Xc�A�*

loss�<<�D6'       �	����Xc�A�*

loss5�<Ͽ�(       �	�`��Xc�A�*

lossh��=��-�       �	���Xc�A�*

lossMy`=���       �	_Ҡ�Xc�A�*

loss���<�5��       �	"n��Xc�A�*

lossT�=DβC       �	5���Xc�A�*

loss�Z�<JHC�       �	6X��Xc�A�*

lossl]�<��)�       �	��Xc�A�*

loss�=���       �	��Xc�A�*

losstf<a�,       �	N���Xc�A�*

loss��;qg8u       �	*��Xc�A�*

loss�i�<"<�       �	pҦ�Xc�A�*

loss��u<_|+�       �	:t��Xc�A�*

loss�p=Xl]        �	���Xc�A�*

loss*ҁ<����       �	Lè�Xc�A�*

loss���<0��       �	@j��Xc�A�*

lossH��<&��       �	���Xc�A�*

loss|��;��.       �	v���Xc�A�*

lossJV�<ʇ8�       �	�E��Xc�A�*

loss<9=\�       �	[��Xc�A�*

loss�d�; v`       �	����Xc�A�*

loss6�&=A��)       �	qX��Xc�A�*

lossO�a<�AI       �	����Xc�A�*

losst<=~`?4       �	L���Xc�A�*

loss��q<ͤ �       �	M.��Xc�A�*

loss�͏=g;�l       �	�ȯ�Xc�A�*

loss)�2=v��h       �	`��Xc�A�*

loss{��<���       �	; ��Xc�A�*

lossū8=�!-V       �	c���Xc�A�*

lossܬ�;UT��       �	&7��Xc�A�*

loss�k�=�:ac       �	]��Xc�A�*

loss��6<Xq6�       �	�y��Xc�A�*

loss� =��z�       �	�B��Xc�A�*

loss+P�=��:       �	;V��Xc�A�*

loss�{�;��o       �	V��Xc�A�*

loss�=Z�,       �	����Xc�A�*

loss��m=-z       �	�7��Xc�A�*

loss��;���       �	�׷�Xc�A�*

loss�W�;���X       �	�u��Xc�A�*

loss}�<���       �	|��Xc�A�*

loss��><���       �	����Xc�A�*

loss& =���       �	�@��Xc�A�*

lossi�s=�(x       �	Iۺ�Xc�A�*

loss]��<=,2�       �	�r��Xc�A�*

lossnn`<�ӽ       �	B
��Xc�A�*

loss���=P`�       �	�#��Xc�A�*

lossCxZ=
%6�       �	i��Xc�A�*

loss=T�;WgT       �	�ľ�Xc�A�*

loss���<>>�       �	흿�Xc�A�*

loss�0g=���Z       �	m6��Xc�A�*

loss��,=�	��       �	h���Xc�A�*

lossx�=��ֻ       �	���Xc�A�*

loss���<�[        �	`���Xc�A�*

loss눎=8��       �	�r��Xc�A�*

lossI��<R=�{       �	�C��Xc�A�*

loss��e=���x       �	����Xc�A�*

loss�+Y<%I       �	����Xc�A�*

loss�	=�=�       �	�%��Xc�A�*

loss`}<z�<P       �	z���Xc�A�*

loss���=sG��       �	g��Xc�A�*

loss�@�=��f�       �	*��Xc�A�*

losspm=K���       �	c���Xc�A�*

loss��@=6�v       �	�0��Xc�A�*

lossC��=l���       �	d��Xc�A�*

loss��=�>�       �	����Xc�A�*

loss��B=��r       �	b���Xc�A�*

loss�q�<IT�	       �	���Xc�A�*

loss�'<tO�       �	����Xc�A�*

loss��=�a�       �	lw��Xc�A�*

loss���<F��~       �	I��Xc�A�*

loss�<�PjN       �	���Xc�A�*

loss��<Q���       �	'K��Xc�A�*

loss�d�<7��9       �	e���Xc�A�*

loss��B<�V�)       �	�|��Xc�A�*

lossh/�;ͽ�       �	z��Xc�A�*

loss&�+=@]�h       �	���Xc�A�*

loss4��<�e       �	of��Xc�A�*

loss��M=\n�       �	����Xc�A�*

loss���<�0@�       �	���Xc�A�*

loss�h�<���x       �	|`��Xc�A�*

lossu�<r��       �	
���Xc�A�*

loss�Ǐ;.��f       �	:���Xc�A�*

loss��H<��%�       �	/0��Xc�A�*

lossI�=x-��       �	��Xc�A�*

loss%��<X�?�       �	����Xc�A�*

loss�.�<Sqÿ       �	�E��Xc�A�*

loss��=
�n�       �	����Xc�A�*

loss���=�c       �	���Xc�A�*

loss���=��K       �	5��Xc�A�*

loss�;�8a)       �	.���Xc�A�*

lossj�<�޲�       �	p_��Xc�A�*

lossX�<��a�       �	&���Xc�A�*

loss|*"=�ַ�       �	V���Xc�A�*

lossC\= �FK       �	/��Xc�A�*

lossc�=@�5       �	���Xc�A�*

lossSs<*��       �	���Xc�A�*

loss,�<p�D�       �	�7��Xc�A�*

lossJu{;�c�       �	n���Xc�A�*

loss-�f=ϝ�       �	E���Xc�A�*

loss��#=k���       �	���Xc�A�*

lossx�<�Pp�       �	Y���Xc�A�*

loss��<@��       �	�^��Xc�A�*

loss��*=2�I       �	����Xc�A�*

lossA�=�@�       �	ɏ��Xc�A�*

lossVu�<�/�       �	P7��Xc�A�*

losslS5<�E��       �	���Xc�A�*

lossq��<o$r       �	nl��Xc�A�*

lossc�<��\m       �	���Xc�A�*

loss��<�7��       �		���Xc�A�*

loss�� =x��&       �	JD��Xc�A�*

lossta=�֓�       �	2���Xc�A�*

loss�=X�       �	 ��Xc�A�*

lossWh$<��W       �	���Xc�A�*

loss�&f<��?       �	���Xc�A�*

loss�g�;�� F       �	2v��Xc�A�*

lossn_s<���       �	���Xc�A�*

loss_�q<m5��       �	5���Xc�A�*

loss�:<��Z�       �	U��Xc�A�*

losslpa<��c�       �	����Xc�A�*

loss2C<WՂ        �	j���Xc�A�*

loss��=Q�o�       �	GW��Xc�A�*

loss��<�>��       �	` ��Xc�A�*

lossV�>=iq��       �	���Xc�A�*

loss@�<=T��       �	|B��Xc�A�*

loss�Q=�q�       �	����Xc�A�*

loss|M=�5F       �	a���Xc�A�*

lossƎ2=��?�       �	�,��Xc�A�*

loss
=� �       �	u���Xc�A�*

loss�l^<r[¬       �	�q��Xc�A�*

loss�<o�       �	���Xc�A�*

loss\].<%��       �	���Xc�A�*

losse��<�u�5       �	]Q��Xc�A�*

loss䷢=]��       �	 ���Xc�A�*

losssU�<�       �	����Xc�A�*

loss��=�"�       �	�5��Xc�A�*

loss�q�=��0�       �	����Xc�A�*

lossa��<Gw�       �	t��Xc�A�*

loss�Xo=mY�B       �	���Xc�A�*

lossɱ�;���@       �	����Xc�A�*

loss�j�;/�E       �	ds��Xc�A�*

loss�]�=p��       �	+��Xc�A�*

loss��c<���=       �	����Xc�A�*

loss�� =\       �	�d��Xc�A�*

loss:��=��D]       �	� �Xc�A�*

lossM6�=ߴ|       �	� �Xc�A�*

lossQR
<H%�       �	YL�Xc�A�*

lossn��;�N�       �	��Xc�A�*

loss��<����       �	+��Xc�A�*

loss�?%=KT*�       �	�N�Xc�A�*

loss���=�DL�       �	���Xc�A�*

loss��1>ʯ�:       �	��Xc�A�*

loss�tF=h`�       �	IH�Xc�A�*

loss�1�=˗d4       �	���Xc�A�*

loss�y�=S���       �	���Xc�A�*

loss`=^Q��       �	ҋ�Xc�A�*

loss�z=����       �	=�Xc�A�*

loss���<�s�-       �	�>	�Xc�A�*

loss��=$h�       �	��	�Xc�A�*

loss'�<ؑg       �	
�
�Xc�A�*

loss���<Q,XP       �	!�Xc�A�*

lossYh=W��S       �	I��Xc�A�*

loss9:=�2��       �	V��Xc�A�*

lossH�<�b|Y       �	F$�Xc�A�*

loss{r<4��A       �	���Xc�A�*

loss��b=���J       �	�o�Xc�A�*

loss��<��X}       �	��Xc�A�*

loss}(~=� ��       �	���Xc�A�*

loss[$f=i���       �	�7�Xc�A�*

loss�t=H�!       �	��Xc�A�*

loss�G=3�y       �	�d�Xc�A�*

loss�\=�h��       �	���Xc�A�*

lossqv�=U|Fn       �	��Xc�A�*

loss��y=n�       �	�8�Xc�A�*

loss�fS=N���       �	���Xc�A�*

loss8�<&���       �	i�Xc�A�*

loss7�p=Z��       �	x�Xc�A�*

loss�V=)�x       �	��Xc�A�*

loss].=���       �	�<�Xc�A�*

loss4v1=�85�       �	���Xc�A�*

loss�#	=�}�G       �	�h�Xc�A�*

loss�"T<���\       �	��Xc�A�*

loss�N�<,A�       �	ܞ�Xc�A�*

lossT {=����       �	r6�Xc�A�*

losshً<���V       �	���Xc�A�*

loss3�=����       �	{�Xc�A�*

lossݦ�=C��       �	�'�Xc�A�*

lossf=��       �	��Xc�A�*

loss���<�5�       �	�h�Xc�A�*

loss�5�;�^�e       �	� �Xc�A�*

lossñ�<ä�;       �	��Xc�A�*

loss,ӧ<ӣ�       �	�H�Xc�A�*

loss��U<�{<d       �	&��Xc�A�*

lossl��<�p       �	��Xc�A�*

loss��i=�q�       �	�X �Xc�A�*

loss;��<���       �	@� �Xc�A�*

loss\�Z<���       �	�!�Xc�A�*

loss���=4���       �	O?"�Xc�A�*

loss��&=�ن9       �	��"�Xc�A�*

loss��R=H��       �	E�#�Xc�A�*

loss̼/<i%F�       �	�!$�Xc�A�*

lossj�3=G	��       �	��$�Xc�A�*

loss@Ǥ=*�߽       �	�`%�Xc�A�*

loss�
=Z�~       �	��%�Xc�A�*

loss44=x.t�       �	ݕ&�Xc�A�*

loss��<�1�       �	�5'�Xc�A�*

loss���=.��       �	x�'�Xc�A�*

loss�L=n�G�       �	en(�Xc�A�*

loss0�< �*       �	�)�Xc�A�*

lossjE�=B��       �	V�)�Xc�A�*

lossL`=��       �	y]*�Xc�A�*

loss�k=���       �	�+�Xc�A�*

loss�<�)�       �	7�+�Xc�A�*

loss_�|=�ga�       �	sL,�Xc�A�*

loss,�5=��^*       �	0�,�Xc�A�*

loss�0G<�fj�       �	$�-�Xc�A�*

losss	%<'_��       �	��.�Xc�A�*

lossIE=���X       �	�E/�Xc�A�*

loss�D�<��c2       �	��/�Xc�A�*

loss`�9=(w�       �	�0�Xc�A�*

loss���;��*�       �	*71�Xc�A�*

loss�3q=�۵       �	��1�Xc�A�*

lossgҊ<���Z       �	�2�Xc�A�*

loss.{<,���       �	X3�Xc�A�*

loss��;�=ȍ       �	m�3�Xc�A�*

loss�Β;�yC       �	�4�Xc�A�*

lossf��=�e�       �	~55�Xc�A�*

loss�G<��U�       �	��5�Xc�A�*

loss�<Z=��       �	�6�Xc�A�*

loss@��;�}e       �	�(7�Xc�A�*

loss�K�<����       �	ۿ7�Xc�A�*

loss$iE=Cd1�       �	�i8�Xc�A�*

loss��L=8�       �	�9�Xc�A�*

loss�h�=ɰ��       �	�9�Xc�A�*

loss65�=��g�       �	CU:�Xc�A�*

lossNr<���l       �	�:�Xc�A�*

loss�V=� B�       �	ѓ;�Xc�A�*

loss���<�eF       �	31<�Xc�A�*

loss��=��       �	W�<�Xc�A�*

loss,=�g/�       �	�}=�Xc�A�*

loss�i�=l4��       �	$F>�Xc�A�*

loss���;2us<       �	G ?�Xc�A�*

loss��=l$�       �	ݗ?�Xc�A�*

lossO��<�&�!       �	�z@�Xc�A�*

loss3�,<c{_�       �	�A�Xc�A�*

lossV�D=�ʨs       �	��B�Xc�A�*

loss�L�=d���       �	hZC�Xc�A�*

loss4�&=c��       �	��C�Xc�A�*

lossf�<��z�       �	��D�Xc�A�*

loss��o<���       �	nE�Xc�A�*

loss�GT=-�       �	>F�Xc�A�*

loss��-<����       �	÷F�Xc�A�*

loss��(=����       �	�QG�Xc�A�*

lossRKJ=���       �	��G�Xc�A�*

loss6�=�%       �	��H�Xc�A�*

lossDq=;��       �	B&I�Xc�A�*

loss�6>�Ƀ�       �	e�I�Xc�A�*

lossp�<�_�       �	ZgJ�Xc�A�*

loss%�<�#��       �	}K�Xc�A�*

loss�a7=1��       �	��K�Xc�A�*

lossQh�=�c�       �	IL�Xc�A�*

loss�A�<I��       �	��L�Xc�A�*

loss��1<��{�       �	��M�Xc�A�*

lossY	={�D       �	;N�Xc�A�*

loss��=`��       �	�N�Xc�A�*

loss0?=@�       �	jO�Xc�A�*

loss7@�<J�p       �	�>P�Xc�A�*

loss;�<TO��       �	��P�Xc�A�*

loss�:�<c�       �	�uQ�Xc�A�*

loss
;�=��x�       �	�R�Xc�A�*

loss�*<�LǠ       �	˽R�Xc�A�*

loss��k=�Ǣ�       �	YRS�Xc�A�*

loss4�\<��D       �	��S�Xc�A�*

losst�g=X��       �	��T�Xc�A�*

lossd}<�#�>       �	`"U�Xc�A�*

loss�C<֖��       �	M�U�Xc�A�*

loss��=��D       �	G�V�Xc�A�*

lossȂ�;���       �	�*W�Xc�A�*

lossخ=s��       �	z�W�Xc�A�*

loss�P�=#憵       �	�nX�Xc�A�*

loss�%�=�VaK       �	G8Y�Xc�A�*

lossI�<Oe�       �	��Y�Xc�A�*

loss,�h=i�w�       �	uxZ�Xc�A�*

loss��g=o:2       �	�[�Xc�A�*

loss��="��l       �	k�[�Xc�A�*

loss�t�<���       �	�N\�Xc�A�*

loss�ye<�!�       �	k�\�Xc�A�*

loss'��<�W3)       �	�]�Xc�A�*

loss���<�=5>       �	�;^�Xc�A�*

loss�Z�=V��       �	�_�Xc�A�*

loss��%<Dg	�       �	�_�Xc�A�*

lossL�#<\���       �	MJ`�Xc�A�*

loss�)�<ǉX�       �	��`�Xc�A�*

loss�+�<�s��       �	��a�Xc�A�*

loss�շ<R�m+       �	� b�Xc�A�*

lossx%%=����       �	��b�Xc�A�*

loss��
=��,�       �	�rc�Xc�A�*

loss)�<X
q       �	�d�Xc�A�*

loss��>�R+�       �	ʣd�Xc�A�*

loss�ґ<�D˅       �	:e�Xc�A�*

losso6�=�a��       �	�e�Xc�A�*

loss��4=�^?�       �	{kf�Xc�A�*

loss�U�<��{\       �	�g�Xc�A�*

loss�h<uV�7       �	c�g�Xc�A�*

lossJ�=}Ya       �	w0h�Xc�A�*

loss�OE=���       �	��i�Xc�A�*

loss��=I_�0       �	�j�Xc�A�*

lossꕊ=U'��       �	 �j�Xc�A�*

loss���<ץ?Z       �	�Pk�Xc�A�*

loss�t=Q�Y�       �	�k�Xc�A�*

loss=��=�'��       �	k}l�Xc�A�*

loss$i�=@�B�       �	�m�Xc�A�*

loss�ȫ=��3�       �	��m�Xc�A�*

loss,>=Y8�B       �	Hn�Xc�A�*

loss��=~У�       �	�n�Xc�A�*

loss�Zv<K��)       �	ɐo�Xc�A�*

lossm�<�Ly       �	,p�Xc�A�*

loss��n=w6�M       �	��p�Xc�A�*

loss��"=�bŇ       �	5{q�Xc�A�*

loss�M9<�Q�       �	�r�Xc�A�*

lossh�;>���       �	�r�Xc�A�*

lossH=o=�:S�       �	&Qs�Xc�A�*

loss�;b=U�9       �	k�s�Xc�A�*

loss�B=�Ǿ       �	S�t�Xc�A�*

loss_�2=y���       �	�3u�Xc�A�*

loss�݅<[$e       �	5�u�Xc�A�*

loss:93<�A�       �	{kv�Xc�A�*

loss�s�=��;       �	�
w�Xc�A�*

loss�(=�3�l       �	-�w�Xc�A�*

loss_�<H ~�       �	~Ux�Xc�A�*

loss�x�='n�       �	o�x�Xc�A�*

lossv�.<[6��       �	O�y�Xc�A�*

loss3��:9]Z�       �	X<z�Xc�A�*

loss�n<�[]       �	�z�Xc�A�*

loss�^I=�~       �	Mj{�Xc�A�*

loss8��<�Y       �	|�Xc�A�*

loss��Z<#�O}       �	ף|�Xc�A�*

loss��=.T9�       �	�T}�Xc�A�*

loss3��<�wel       �	��}�Xc�A�*

loss$��=G�S�       �	�~�Xc�A�*

loss��<kuSU       �	#�Xc�A�*

loss�9=�F
       �	���Xc�A�*

lossvw�=d�       �	���Xc�A�*

loss;B�=~/!       �	���Xc�A�*

lossVe�=�b�       �	7Á�Xc�A�*

loss8@X=�P��       �	Vb��Xc�A�*

loss�݋=*t�/       �	���Xc�A�*

lossC�<�*ۦ       �	��Xc�A�*

loss�]�<n�       �	�\��Xc�A�*

loss=�=X�iq       �	��Xc�A�*

lossB�=~XXN       �	����Xc�A�*

lossO*�;#�)T       �	�P��Xc�A�*

loss�<�2(       �	��Xc�A�*

lossE3+=�Im       �	����Xc�A�*

loss��;=��(&       �	�4��Xc�A�*

loss�K<��܎       �	�ˈ�Xc�A�*

loss�pT=��jr       �	zp��Xc�A�*

loss�±;7��y       �	W#��Xc�A�*

loss��=����       �	eŊ�Xc�A�*

losst�=_JLe       �	Dj��Xc�A�*

loss��5=�R@
       �	̸��Xc�A�*

lossWF�=5���       �	X��Xc�A�*

loss�=�.�q       �	����Xc�A�*

loss�<(f       �	9���Xc�A�*

lossڑ�<8-�b       �	�;��Xc�A�*

loss�<�=�cvZ       �	|֏�Xc�A�*

loss�
#=j�x       �	�m��Xc�A�*

loss���<Ev�       �	��Xc�A�*

lossL�=
�f       �	�Xc�A�*

loss�L�<ZzC�       �	�L��Xc�A�*

lossD��;���       �	���Xc�A�*

loss�[=s�-�       �	<���Xc�A�*

loss��C=��B�       �	 ��Xc�A�*

loss��;�23x       �	���Xc�A�*

loss�;<fڄ5       �	�k��Xc�A�*

loss͚h=N52�       �	�K��Xc�A�*

loss�[>`y_       �	�6��Xc�A�*

lossTȮ=�y��       �	�ӗ�Xc�A�*

lossx�=���	       �	�p��Xc�A�*

loss��<D�&       �	��Xc�A�*

lossZ�<��W       �	K���Xc�A�*

loss�2=����       �	Y���Xc�A�*

lossP�=��Qs       �	�#��Xc�A�*

losslFx=����       �	���Xc�A�*

loss%��<4e.$       �	�U��Xc�A�*

loss��O=5Fdq       �	���Xc�A�*

loss���;Y�ӵ       �	����Xc�A�*

loss�Z�<y:d4       �	S��Xc�A�*

loss�k�=ኙ0       �		��Xc�A�*

loss_U�=�\��       �	l���Xc�A�*

loss}�=�K�d       �	�M��Xc�A�*

lossZ�=%       �	����Xc�A�*

loss�'=Ŗ�<       �	F���Xc�A�*

loss���=��#Y       �	����Xc�A�*

losss��=�;�2       �	)&��Xc�A�*

loss�=h��&       �	Hã�Xc�A�*

loss�P�<�T       �	|c��Xc�A�*

loss��6<����       �	� ��Xc�A�*

loss�^�=���       �	����Xc�A�*

loss�?=�'b/       �	�4��Xc�A�*

loss�b=	_�        �	'٦�Xc�A�*

lossၣ<�@�       �	�~��Xc�A�*

loss-��<:�p       �	d��Xc�A�*

lossX�5=x�`        �	���Xc�A�*

loss.r<�5��       �	XV��Xc�A�*

lossWs�=��4       �	���Xc�A�*

loss��<<<qX       �	����Xc�A�*

loss<=�Ŕ       �	�B��Xc�A�*

loss흵<	R:       �	"��Xc�A�*

loss�:<MΈN       �	||��Xc�A�*

loss��j<�r       �	���Xc�A�*

loss��O;�6��       �	sڭ�Xc�A�*

loss��<�c�.       �	0���Xc�A�*

loss�<�f�       �	���Xc�A�*

loss�)P>R�V�       �	=���Xc�A�*

loss�z�=R�)       �	�W��Xc�A�*

lossl<&m�       �	j���Xc�A�*

lossx4�;���g       �	p���Xc�A�*

loss͢�=��9�       �	i6��Xc�A�*

loss�Id=�ԯ       �	$Ӳ�Xc�A�*

loss���<ls2       �	�u��Xc�A�*

loss@;F�U[       �	s��Xc�A�*

loss��=��͛       �	꯴�Xc�A�*

loss��<C��       �	fL��Xc�A�*

loss���=�{j       �	"��Xc�A�*

loss��>="�       �	ڌ��Xc�A�*

lossT|C=�4'=       �	:#��Xc�A�*

loss��<$�yz       �	����Xc�A�*

loss�>�=p���       �	KX��Xc�A�*

loss��;ڱ��       �	���Xc�A�*

loss�Z�<�T7�       �	M���Xc�A�*

lossb�=���e       �	V*��Xc�A�*

loss}f(<r8�I       �	ӿ��Xc�A�*

loss��<����       �	���Xc�A�*

loss)5�=ta�b       �	����Xc�A�*

loss���=�{�A       �	Z���Xc�A�*

lossH�3<Mȹ       �	�߾�Xc�A�*

loss���<X��       �	����Xc�A�*

lossI%<��       �	 '��Xc�A�*

loss��}=%>^       �	����Xc�A�*

loss�,�<�[#       �	$d��Xc�A�*

loss��<�7��       �	G��Xc�A�*

loss��<����       �	ޭ��Xc�A�*

loss\MX=�iB       �	�q��Xc�A�*

loss��=M`G}       �	0��Xc�A�*

loss�L=��^b       �	}���Xc�A�*

lossvO={0�8       �	I���Xc�A�*

loss���<n�ܬ       �	�N��Xc�A�*

loss�K!=8\8�       �	����Xc�A�*

lossAQ�<��P�       �	5���Xc�A�*

loss�7=�/��       �	M2��Xc�A�*

loss	Dy=����       �	h���Xc�A�*

loss%�/<���2       �	ė��Xc�A�*

loss;��<?f��       �	�?��Xc�A�*

lossq�=���       �	.���Xc�A�*

loss�)�=��        �	\���Xc�A�*

lossJa=ʨ:�       �	�*��Xc�A�*

loss��<To       �	G���Xc�A�*

loss�N�=3       �	�c��Xc�A�*

loss�1�<�h�I       �	a���Xc�A�*

lossk�!<��^{       �	����Xc�A�*

lossC�V<���        �	�?��Xc�A�*

loss{�<���       �	����Xc�A�*

loss�s�=~�       �	o���Xc�A�*

loss�]�<���       �	A(��Xc�A�*

loss�z�;T��       �	l���Xc�A�*

loss|��<;P�       �	Hj��Xc�A�*

loss�[<��b       �	f��Xc�A�*

lossM![9R__       �	����Xc�A�*

loss5B<`p�<       �	-[��Xc�A�*

loss��C;�\O%       �	j���Xc�A�*

loss�*�;Y
\       �	w���Xc�A�*

loss �5<�       �	p`��Xc�A�*

loss�B�;	�17       �	����Xc�A�*

lossC!n;mݯ       �	����Xc�A�*

lossa�<��9�       �	�N��Xc�A�*

lossP:"1�       �	����Xc�A�*

loss�P	:p���       �	V���Xc�A�*

loss �;��       �	�y��Xc�A�*

loss��%=xAy�       �	"��Xc�A�*

loss
;�;���       �	����Xc�A�*

loss�¾9��Ƅ       �	)^��Xc�A�*

loss���<{"��       �	_	��Xc�A�*

loss`�U>�5Jn       �	����Xc�A�*

lossߢ�;Brc�       �	�`��Xc�A�*

loss�l>�ud       �	���Xc�A�*

loss
P=&���       �	p#��Xc�A�*

loss�R�<�!�       �	���Xc�A�*

loss4�<$��       �	+l��Xc�A�*

loss�n�<�Ԝ       �	2���Xc�A�*

lossA}�=�
+       �	ϡ��Xc�A�*

loss�~.=I��       �	6>��Xc�A�*

loss��=	%��       �	r��Xc�A�*

loss.��<���       �	 ���Xc�A�*

loss��=�X4Q       �	N��Xc�A�*

loss���=�F�       �	b���Xc�A�*

loss2�S=4�6g       �	���Xc�A�*

loss��<���       �	���Xc�A�*

loss�d�<k�Ճ       �	�r��Xc�A�*

loss
#�<\F�*       �	x��Xc�A�*

loss��<�=[�       �	д��Xc�A�*

loss3��<�PW       �	�U��Xc�A�*

lossq�;=j��       �	����Xc�A�*

loss��<�_�       �	u���Xc�A�*

loss:S�;�R��       �	k-��Xc�A�*

loss3?,=�s        �	1���Xc�A�*

loss�I�<:�9�       �	@i��Xc�A�*

lossQ��;���       �	��Xc�A�*

loss���<����       �	����Xc�A�*

loss��<R��       �	EI��Xc�A�*

lossa��=���       �	$���Xc�A�*

loss�B�<L ��       �	7���Xc�A�*

lossſP=t���       �	�'��Xc�A�*

loss
�=����       �	����Xc�A�*

loss%J�<m��       �	qY��Xc�A�*

loss�w<���       �	Z���Xc�A�*

loss�o&<��/       �	����Xc�A�*

loss.d<u�~�       �	K���Xc�A�*

loss�T=)R�       �	P8��Xc�A�*

loss��;lF��       �	S���Xc�A�*

loss ��;�.�&       �	����Xc�A�*

loss�}J=���J       �	3��Xc�A�*

loss1Q=����       �	]���Xc�A�*

loss/�<���
       �	p{��Xc�A�*

loss��<֜9�       �	��Xc�A�*

loss��K<�6Ƭ       �	Z���Xc�A�*

loss�f�<utBK       �	YP��Xc�A�*

loss�t=T;�`       �	��Xc�A�*

loss;<@
Tk       �	&���Xc�A�*

loss���<�       �	����Xc�A�*

loss8��=z]       �	�� �Xc�A�*

lossfI=�e��       �	���Xc�A�*

loss,9>=� (m       �	,�Xc�A�*

loss��;�&?[       �	`v�Xc�A�*

loss[��<�4'�       �	w�Xc�A�*

loss|�;��S       �	~��Xc�A�*

loss (�<$��       �	���Xc�A�*

loss\P=���       �	�G�Xc�A�*

loss��#<�{�       �	���Xc�A�*

loss�x�=ml�       �	���Xc�A�*

lossF'=�|z       �	zl �Xc�A�*

loss=g�h       �	5!�Xc�A�*

loss�1=�洯       �	O�!�Xc�A�*

loss�@b=-��       �	vP"�Xc�A�*

loss��:<S��b       �	�#�Xc�A�*

loss��G<}��       �	`$�Xc�A�*

lossce�<�Rk�       �	�%�Xc�A�*

loss4�L=>A'.       �	F&�Xc�A�*

loss(Q=�`��       �	k�&�Xc�A�*

lossl�;�i1+       �	��'�Xc�A�*

loss���<d.�g       �	�(�Xc�A�*

loss�A�;1\a       �	v�)�Xc�A�*

lossd��;�.��       �	�~*�Xc�A�*

loss���<]�U�       �	B&+�Xc�A�*

loss;��=B�3       �	X�+�Xc�A�*

loss��<c�       �	y,�Xc�A�*

loss�[�=6G��       �	Ag-�Xc�A�*

loss�ձ<�w��       �	�8.�Xc�A�*

loss,O�=�{I�       �	�9/�Xc�A�*

loss�=��#'       �	��/�Xc�A�*

loss���<�y֮       �	c�0�Xc�A�*

loss�q<�S��       �	*61�Xc�A�*

loss��;� ��       �	hv2�Xc�A�*

loss]�1=�8�       �	�%3�Xc�A�*

loss!n~=�S0�       �	�+4�Xc�A�*

loss66='@�       �	��4�Xc�A�*

loss��=�B}T       �	zo5�Xc�A�*

loss���<���       �	�Y6�Xc�A�*

loss?�=y��       �	��6�Xc�A�*

loss;�?<��3�       �	U�7�Xc�A�*

lossq3�=.���       �	�@8�Xc�A�*

lossw_�<H�u�       �	��8�Xc�A�*

losse^=��K       �	�9�Xc�A�*

lossX�">!�qE       �	@2:�Xc�A�*

loss��+=       �	��:�Xc�A�*

lossM?�<��՛       �	~t;�Xc�A�*

loss�N<q'�X       �	�<�Xc�A�*

lossr�<��g       �	��<�Xc�A�*

loss	�<�K+       �	>=�Xc�A�*

lossqA=~ ��       �	(�=�Xc�A�*

loss�s�<{�L�       �	��>�Xc�A�*

loss)	�<����       �	:;?�Xc�A�*

loss6=�ɖ�       �	\�?�Xc�A�*

loss�6�=�sN�       �	�@�Xc�A�*

loss.Z;SSx�       �	1|A�Xc�A�*

loss�`�;�js�       �	�B�Xc�A�*

lossf��=.�*       �	�+C�Xc�A�*

loss�0<\'s       �	�[D�Xc�A�*

lossd�r=L��5       �	8�D�Xc�A�*

loss���<�T6t       �	��E�Xc�A�*

loss�.�;�#U�       �	J�F�Xc�A�*

lossh��:j>h>       �	W�G�Xc�A�*

loss�'�<)�Y       �	�H�Xc�A�*

loss��7<@!��       �	��I�Xc�A�*

lossC��=��i�       �	��J�Xc�A�*

loss8t>3�F�       �	��K�Xc�A�*

loss�?/<����       �	*sL�Xc�A�*

loss��<4"{�       �	h M�Xc�A�*

lossC߅=��3�       �	��M�Xc�A�*

loss�?�<5@_�       �	h�N�Xc�A�*

loss��=E��       �	�O�Xc�A�*

lossh�=M���       �	ȘP�Xc�A�*

loss[R=nB14       �	�pQ�Xc�A�*

loss �=Y*\       �		R�Xc�A�*

loss��S<�TO       �	d�R�Xc�A�*

loss;��<���       �	qYS�Xc�A�*

loss%)I<���       �	}T�Xc�A�*

loss8�}<��       �	��T�Xc�A�*

lossV6�<�}�       �	GVU�Xc�A�*

loss�D=���       �	P�U�Xc�A�*

loss�B=L3�       �	��V�Xc�A�*

lossVʠ<��        �	�9W�Xc�A�*

loss�{�=m���       �	U�W�Xc�A�*

loss�!�<uh       �	�|X�Xc�A�*

lossʞ�=�dH       �	�Y�Xc�A�*

loss.	�<�ye       �	z�Y�Xc�A�*

loss���=3�       �	JZ�Xc�A�*

loss��<�׾R       �	v[�Xc�A�*

loss��;�U�F       �	j�[�Xc�A�*

loss(1<�E��       �	�`\�Xc�A�*

loss�ݳ<p�&�       �	��\�Xc�A�*

lossߵb<4���       �	l�]�Xc�A�*

loss���<Ӝ��       �	l�^�Xc�A�*

loss�_<�x�[       �	f1_�Xc�A�*

loss�C�<�       �	C�_�Xc�A�*

lossH�h<��e�       �	�j`�Xc�A�*

loss�3�< }F       �	�	a�Xc�A�*

loss��;�l�       �	��a�Xc�A�*

loss�B�=��Q�       �	<gb�Xc�A�*

loss���<����       �	��b�Xc�A�*

loss�#<���J       �	��c�Xc�A�*

loss�T6=p&L�       �	��d�Xc�A�*

losss</�~�       �	e�Xc�A�*

loss���=l��M       �	B�e�Xc�A�*

loss*�m<�"��       �	Nf�Xc�A�*

loss��S<W�7       �	�f�Xc�A�*

loss�*;��       �	��g�Xc�A�*

lossO�<����       �	�/h�Xc�A�*

loss=p�<��q       �	qi�Xc�A�*

loss�Y=c��#       �	�i�Xc�A�*

loss��y<єd       �	W�j�Xc�A�*

loss��<��	�       �	q<k�Xc�A�*

loss��9<+��       �	��k�Xc�A�*

loss�R�=�C�;       �	��l�Xc�A�*

lossr��<���       �	�m�Xc�A�*

loss:�<| S       �	N�m�Xc�A�*

lossT/M=Bϧ       �	[n�Xc�A�*

lossa�<O���       �	do�Xc�A�*

lossլ;�s�`       �	Y�o�Xc�A�*

lossh<�3E       �	�Gp�Xc�A�*

loss��f;����       �	��p�Xc�A�*

lossb9<iԼ       �	�}q�Xc�A�*

lossv�C=����       �	�r�Xc�A�*

lossR��<u��       �	�r�Xc�A�*

loss��<��8n       �	3Ps�Xc�A�*

loss���<�@�       �	��s�Xc�A�*

loss�g�<u���       �	R�t�Xc�A�*

lossO1<�1i�       �	�0u�Xc�A�*

lossM��<X�C       �	��u�Xc�A�*

loss��<B�       �	lv�Xc�A�*

loss�o==EA�~       �		w�Xc�A�*

loss���<a=�5       �	�w�Xc�A�*

loss*��<k���       �	>Ax�Xc�A�*

loss)�>=�
��       �	��x�Xc�A�*

loss��+=�=c�       �	�y�Xc�A�*

loss�w�<T��       �	)z�Xc�A�*

loss�i,=���Q       �	�z�Xc�A�*

losse7=>=�       �	�i{�Xc�A�*

loss�Q<�<ޒ       �	q|�Xc�A�*

lossPĔ=.��0       �	��|�Xc�A�*

loss�S9<R�	�       �	v2}�Xc�A�*

loss{��<�b�       �	��}�Xc�A�*

loss�L�;*͎�       �	yx~�Xc�A�*

loss�0=�5       �	U�Xc�A�*

loss3^Y;"�X�       �	m��Xc�A�*

loss���<)�       �	j�Xc�A�*

loss��;���       �	4ց�Xc�A�*

loss�R�<T�       �	]���Xc�A�*

loss��w=-�       �	�ԃ�Xc�A�*

lossfɃ=.�d       �	o��Xc�A�*

loss���;�k��       �	���Xc�A�*

lossg�=�r��       �	"��Xc�A�*

loss�8�;���       �	�ˆ�Xc�A�*

loss3<=J4\�       �	4���Xc�A�*

loss[�v< ��;       �	z8��Xc�A�*

loss�h�;����       �	��Xc�A�*

loss�Hp=�X4�       �	-���Xc�A�*

lossRO�<ϕy�       �	�K��Xc�A�*

lossW�q<^�       �	���Xc�A�*

loss��v=�4:�       �	5��Xc�A�*

loss̍�:�O{�       �	X��Xc�A�*

loss�2<we�       �	�Ќ�Xc�A�*

loss��= �       �	od��Xc�A�*

loss?�<u�       �	b���Xc�A�*

loss6_;�d�        �	ڏ��Xc�A�*

loss���<��A       �	�T��Xc�A�*

loss���<0o
t       �	`��Xc�A�*

loss�S3=�o��       �	E��Xc�A�*

lossA:��-'       �	���Xc�A�*

lossX�<��/       �	u���Xc�A�*

loss]�=�qu�       �	�M��Xc�A�*

loss��:����       �	���Xc�A�*

loss撆=���       �	&���Xc�A�*

loss]+ =��30       �	�g��Xc�A�*

lossRĹ;IWN�       �	����Xc�A�*

loss�O�<�6u�       �	����Xc�A�*

loss�3<��fh       �	�!��Xc�A�*

lossnU8=�k�       �	!Ȗ�Xc�A�*

lossw��<���7       �	 ]��Xc�A�*

lossW�"=%`4�       �	����Xc�A�*

lossR=]=B       �	����Xc�A�*

loss(�k=�6��       �	@k��Xc�A�*

loss�E=7�       �	2��Xc�A�*

lossZ�=���       �	����Xc�A�*

lossf;�;g�/       �	�:��Xc�A�*

loss�ŧ:A2��       �	�Л�Xc�A�*

lossJ�=i4�       �	�k��Xc�A�*

lossr1�<1�;       �	K��Xc�A�*

loss3.�<T/,�       �	@���Xc�A�*

lossf7C;��       �	m7��Xc�A�*

loss}S�;��/�       �	�Ϟ�Xc�A�*

loss�H<~�
�       �	Dk��Xc�A�*

loss18s<����       �	�l��Xc�A�*

loss�!v<I�Y3       �	h��Xc�A�*

loss�N?=�B��       �	����Xc�A�*

loss��T=�~��       �	�Z��Xc�A�*

loss#w�;k4j`       �	���Xc�A�*

loss/)�<K�y       �	����Xc�A�*

loss[�:<���       �	�2��Xc�A�*

lossb�9%���       �	�T��Xc�A�*

lossQ�j<�J��       �	c��Xc�A�*

loss�=:�͗       �	n���Xc�A�*

loss��	=��~�       �	!>��Xc�A�*

loss>�=[�P�       �	�ۧ�Xc�A�*

loss���;��6       �	s���Xc�A�*

lossd�=�3,       �	n���Xc�A�*

loss�i�;�w��       �	zp��Xc�A�*

loss�x�<�t       �	F��Xc�A�*

loss:Qq<e �}       �	o���Xc�A�*

loss��;�5�Y       �	�9��Xc�A�*

loss���;6WM�       �	�Ӭ�Xc�A�*

loss��=�g�Q       �	�v��Xc�A�*

loss���=���       �	� ��Xc�A�*

loss�m@=Vf�=       �	+®�Xc�A�*

loss���<��	       �	�`��Xc�A�*

loss_j�;u��       �	��Xc�A�*

loss��=�#_g       �	����Xc�A�*

loss���;�m�       �	�<��Xc�A�*

lossL�Z;5��m       �	X��Xc�A�*

loss�<�vw       �	,~��Xc�A�*

loss�qr=��C       �	���Xc�A�*

loss���=�@c�       �	��Xc�A�*

loss�]5=SG�       �	֐��Xc�A�*

loss-��=���       �	m:��Xc�A�*

loss(2m=�r�o       �	(׵�Xc�A�*

lossv��<t<�       �	����Xc�A�*

lossN��<�m7P       �	�)��Xc�A�*

loss�H<ʿ*V       �	yɷ�Xc�A�*

loss�@=�ih�       �	g��Xc�A�*

lossO�;Fݕ�       �	��Xc�A�*

loss�P�<��:       �	����Xc�A�*

loss��=%[�       �	A���Xc�A�*

loss�,.<<r1'       �	-��Xc�A�*

loss���<P&o       �	�ֻ�Xc�A�*

loss'</�6�       �	�o��Xc�A�*

lossFQY<�d�       �	���Xc�A�*

loss��<���       �	>���Xc�A�*

loss��=�R��       �	2V��Xc�A�*

loss�]�<�-h9       �	��Xc�A�*

loss�T=$D       �	����Xc�A�*

loss)l�;IA��       �	�I��Xc�A�*

loss��><m�       �	����Xc�A�*

lossl�v=4X�       �	����Xc�A�*

loss`��<7u��       �	���Xc�A�*

loss�v�;���h       �	���Xc�A�*

lossv�<*�tK       �	�O��Xc�A�*

loss���<��D�       �	����Xc�A�*

loss�=�Y�v       �	���Xc�A�*

loss�[3=W��       �	�.��Xc�A�*

loss��_=_��2       �	{���Xc�A�*

loss�y�<����       �	�v��Xc�A�*

loss%<�c       �	���Xc�A�*

loss�6�=�       �	ü��Xc�A�*

loss�. =��       �	�W��Xc�A�*

lossv'8=A��       �	���Xc�A�*

loss�F�;�(�       �	����Xc�A�*

loss=��<�0d       �	zQ��Xc�A�*

loss�<Z��=       �	]���Xc�A�*

loss��	;�E�       �	����Xc�A�*

loss�~;`0��       �	�A��Xc�A�*

loss"�<��`       �	����Xc�A�*

loss���<��Y�       �	#���Xc�A�*

loss?�;>�#3       �	=)��Xc�A�*

loss�
�=k�Q       �	?���Xc�A�*

lossެ<N�S�       �	�`��Xc�A�*

lossc�<m���       �	 ��Xc�A�*

loss�@y<� �t       �	)���Xc�A�*

loss��!=��>�       �	�0��Xc�A�*

loss�x<�;Ip       �	y��Xc�A�*

lossm�=W��       �	���Xc�A�*

loss�_<��       �	;6��Xc�A�*

loss#2[<�`g3       �	)���Xc�A�*

loss
e�<pk�R       �	�j��Xc�A�*

lossܱG=-�#t       �	���Xc�A�*

loss�h�<��=       �	D���Xc�A�*

lossO�(=t�5�       �	�~��Xc�A�*

lossQ��<M@5�       �	���Xc�A�*

loss�R�;c�,�       �	����Xc�A�*

lossEѺ=/���       �	YP��Xc�A�*

loss���=2�?�       �	����Xc�A�*

lossZ6\=(���       �	����Xc�A�*

loss��<< ���       �	�u��Xc�A�*

loss�Z�:h1�$       �	���Xc�A�*

loss��:=���Z       �	����Xc�A�*

lossL�:=����       �	tC��Xc�A�*

lossM@�;�y�       �	����Xc�A�*

loss!<�<�]��       �	 t��Xc�A�*

loss��<��$       �	5��Xc�A�*

loss�[=���       �	t���Xc�A�*

loss�D,=�6!�       �	|d��Xc�A�*

loss,��=eyC�       �	W��Xc�A�*

loss�=��h       �	Z���Xc�A�*

loss��=�yx�       �	�Z��Xc�A�*

loss��<=,��       �	}��Xc�A�*

loss�Ef<53�       �	W���Xc�A�*

loss�֢:�-Rx       �	�E��Xc�A�*

loss�"=�(       �	M.��Xc�A�*

loss�=;�\�R       �	���Xc�A�*

lossݙ�:�\l�       �	�s��Xc�A�*

loss�1�<��v�       �	�)��Xc�A�*

loss$�m=��       �	����Xc�A�*

loss�m=<+ӭ       �	����Xc�A�*

loss���=�iLU       �	�8��Xc�A�*

loss?�=��       �	����Xc�A�*

loss <�<�ß�       �	n���Xc�A�*

lossm��<<c�$       �	.8��Xc�A�*

loss_�=�J�"       �	����Xc�A�*

loss?�;���       �	����Xc�A�*

loss��v<���       �	Aa��Xc�A�*

loss3CF=���       �	���Xc�A�*

loss���=U���       �	/���Xc�A�*

loss�
�;4���       �	P9��Xc�A�*

loss��<�"       �	���Xc�A�*

lossU�<��[�       �	�v��Xc�A�*

lossq/�<ĝ�       �	T��Xc�A�*

lossZ<B�q:       �	� ��Xc�A�*

lossܟ�<@��|       �	'���Xc�A�*

loss=���       �	hB��Xc�A�*

loss��[<-G�       �	#���Xc�A�*

loss��<�iH�       �	5}��Xc�A�*

lossOU�=��3�       �	���Xc�A�*

lossϴp;S��|       �	���Xc�A�*

loss�$�<���       �	zV��Xc�A�*

losst�<�t��       �	E���Xc�A�*

loss�<�<��       �	j���Xc�A�*

loss�Й<laU       �	���Xc�A�*

loss��=3�#	       �	���Xc�A�*

loss�s�<�~       �	�^��Xc�A�*

lossC��;��-�       �	���Xc�A�*

loss;��=o&�       �	ԙ��Xc�A�*

loss��=O2^�       �	6��Xc�A�*

loss߆�<���       �	����Xc�A�*

loss���<�Jw       �	�l��Xc�A�*

loss|G�<A�?j       �	��Xc�A�*

loss�::=��       �	D���Xc�A�*

lossmz"<�[       �	�B��Xc�A�*

loss7<><!�       �	����Xc�A�*

loss\?�;��k�       �	�z��Xc�A�*

loss�t=q��       �	�&��Xc�A�*

lossML+=!���       �	����Xc�A�*

loss?%�<.K�       �	�� �Xc�A�*

loss���<V��'       �	?�Xc�A�*

loss��;p�,;       �	
��Xc�A�*

loss�WZ=_?�$       �	�n�Xc�A�*

lossl�
;���Z       �	E�Xc�A�*

loss�D�;9�cE       �	���Xc�A�*

loss�b'<8��n       �	�:�Xc�A�*

loss�{{=Xo
       �	V��Xc�A�*

loss��p=��M       �	.r�Xc�A�*

lossD=��9       �	��Xc�A�*

lossT�7<
.�9       �	O��Xc�A�*

loss���;�p�       �	�w�Xc�A�*

lossL�8=���       �	��Xc�A�*

loss��K=3xt       �	֭�Xc�A�*

loss\ą<�f�/       �	jP	�Xc�A�*

loss�z�;?�	@       �	��	�Xc�A�*

loss�9<���       �	��
�Xc�A�*

loss�Ƕ<D�'�       �	�!�Xc�A�*

losss%<�l�	       �	���Xc�A�*

loss��z<�*�       �	_�Xc�A�*

lossi
�<c���       �	i��Xc�A�*

lossea<�;|�       �	���Xc�A�*

loss��=�7b       �	a4�Xc�A�*

loss׃�<g��$       �	���Xc�A�*

loss$��<"�<�       �	?p�Xc�A�*

lossx�L<�ߢ,       �	J�Xc�A�*

lossQ(=��h       �	���Xc�A�*

loss��E<
8�       �	LU�Xc�A�*

loss.��=մQJ       �	���Xc�A�*

loss`��=���       �	��Xc�A�*

loss�[�<u��7       �	5E�Xc�A�*

lossv;�<o��g       �	D��Xc�A�*

lossS�=Lj��       �	��Xc�A�*

loss_��:ˍ�       �	� �Xc�A�*

loss�8�<�BH-       �	
��Xc�A�*

loss�=ŨT�       �	�Y�Xc�A�*

loss�z8<��vV       �	c�Xc�A�*

loss�N&<(��       �	
��Xc�A�*

loss�U�;0��       �	P9�Xc�A�*

loss|vH;��|       �	4��Xc�A�*

loss�q�<Mf��       �	�o�Xc�A�*

loss8�<k�~�       �	��Xc�A�*

loss�M8=�w��       �	:��Xc�A�*

loss�1�<�f"       �	�K�Xc�A�*

loss�!�=����       �	M��Xc�A�*

lossl� <o���       �	��Xc�A�*

loss5<�9��       �	�'�Xc�A�*

loss&%�=?�V�       �	C��Xc�A�*

loss�ŕ;��QQ       �	���Xc�A�*

lossQJ�<��y       �	.V�Xc�A�*

lossT�=lVr�       �	� �Xc�A�*

losst�=�C�/       �	�� �Xc�A�*

loss2��=�ae�       �	�7!�Xc�A�*

loss�O1<K:�1       �	�A"�Xc�A�*

lossI�8;��=4       �	]�"�Xc�A�*

loss9=!<��\H       �	�}#�Xc�A�*

loss��=���       �	�E$�Xc�A�*

loss[=y<��@l       �	Z%�Xc�A�*

loss��:=��aG       �	T�%�Xc�A�*

loss��<���E       �	Nb&�Xc�A�*

loss�>����       �	�'�Xc�A�*

loss�:<kl[5       �	��'�Xc�A�*

loss8*�<���       �	�(�Xc�A�*

loss�4H<8*|�       �	f.)�Xc�A�*

losso�L<eM!�       �	e�)�Xc�A�*

loss��=�̽Z       �	�]*�Xc�A�*

loss8�=�M�_       �	�|+�Xc�A�*

loss�,�=�b�C       �	�,�Xc�A�*

loss�Ԛ=e��"       �	+�,�Xc�A�*

loss�	�=�#�       �	�d-�Xc�A�*

loss�+=�	]2       �	�.�Xc�A�*

loss��<b���       �	�.�Xc�A�*

lossa5=Ȋ��       �	K:/�Xc�A�*

lossH	=��h       �	#�/�Xc�A�*

loss제<�	Ų       �	�s0�Xc�A�*

loss��C=�\t�       �	,1�Xc�A�*

loss�/�=�풟       �	�1�Xc�A�*

loss��<w6�]       �	�@2�Xc�A�*

losse�=-�t       �	��2�Xc�A�*

loss7��<::��       �	�q3�Xc�A�*

lossa�:= \��       �	�4�Xc�A�*

lossQW ;�'�       �	ǟ4�Xc�A�*

loss�ʜ<� �       �	.75�Xc�A�*

lossra=��X�       �	�5�Xc�A�*

loss��<g�iX       �	��6�Xc�A�*

loss]P[<��"G       �	��7�Xc�A�*

loss�?�=X��       �	VJ8�Xc�A�*

loss(��;1�i       �	��8�Xc�A�*

loss�Q�=ڛ�5       �	�9�Xc�A�*

loss��k<�p�2       �	Kt:�Xc�A�*

loss?��=#��       �		;�Xc�A�*

loss��"<ZCs       �	��;�Xc�A�*

loss��(<a���       �	�d<�Xc�A�*

loss� <�~[<       �	�=�Xc�A�*

lossH
�<�璳       �	s�=�Xc�A�*

loss���<^�?       �	&9>�Xc�A�*

loss;�`<@��       �	��>�Xc�A�*

lossOVy<�=�       �	�o?�Xc�A�*

lossT�=��       �	�@�Xc�A�*

loss�܌=I���       �	�IB�Xc�A�*

loss�<=	슄       �	Z�B�Xc�A�*

loss��;r �Z       �	\D�Xc�A�*

loss�c<
 �       �	qE�Xc�A�*

loss��<
��Y       �	șE�Xc�A�*

loss�8W<��v       �	YNF�Xc�A�*

loss�:=8�Y       �	�F�Xc�A�*

loss!|Q=�g{K       �	��G�Xc�A�*

loss�/J<��       �	�'H�Xc�A�*

lossIգ=��       �	��H�Xc�A�*

loss�D�;���       �	�^I�Xc�A�*

loss��<�M.�       �	��I�Xc�A�*

loss��;=ʪ��       �	�J�Xc�A�*

loss��<^�(       �	�#K�Xc�A�*

lossq:y<��       �	��K�Xc�A�*

lossC!=ᒱ`       �	�NL�Xc�A�*

loss��=���       �	��L�Xc�A�*

loss;�;��Iy       �	AM�Xc�A�*

loss���:VW�       �	�N�Xc�A�*

loss�,�=�΂�       �	�N�Xc�A�*

loss�7Y=B.�       �	UO�Xc�A�*

loss�z =���       �	��O�Xc�A�*

lossF)�<���       �	=�P�Xc�A�*

losstŕ<6s�       �	+0Q�Xc�A�*

loss. H<�A�       �	%�Q�Xc�A�*

loss1Y�=�䲒       �	ZfR�Xc�A�*

lossh�<tAC       �	��R�Xc�A�*

loss�@�;��=�       �	t�S�Xc�A�*

loss�0;<%!       �	+/T�Xc�A�*

loss�V�=����       �	1�T�Xc�A�*

lossR,g<�       �	jlU�Xc�A�*

lossFM�<M�6�       �	�V�Xc�A�*

loss�b�<ٓq:       �	ʤV�Xc�A�*

loss��<�[�w       �	G;W�Xc�A�*

lossJl�=���P       �	��W�Xc�A�*

loss��;;�_       �	�X�Xc�A�*

loss�R<>�صj       �	'Y�Xc�A�*

lossSk3<*�)S       �	ܽY�Xc�A�*

loss���<�d�       �	�RZ�Xc�A�*

loss�J2=��n`       �	R�Z�Xc�A�*

lossq�;ܙ��       �	�[�Xc�A�*

lossj�|<0YA       �	s-\�Xc�A�*

loss��<Iv��       �	��\�Xc�A�*

loss���<�=W�       �	:w]�Xc�A�*

loss$#�;�ut       �	o^�Xc�A�*

loss�0�=�       �	o�^�Xc�A�*

loss�P>@�#/       �	Sz_�Xc�A�*

loss�F�=M}��       �	�`�Xc�A�*

losshP�<����       �	�`�Xc�A�*

loss=uZ=0�       �	�Fa�Xc�A�*

loss��=�>�G       �	obb�Xc�A�*

loss$0�;2��{       �	� c�Xc�A�*

loss�}<���       �	��c�Xc�A�*

loss���=��%	       �	�zd�Xc�A�*

loss�F=S)�       �	fe�Xc�A�*

loss.K�<�u��       �	Q�e�Xc�A�*

loss��_=�gM�       �	�=f�Xc�A�*

loss-�r=��       �	�g�Xc�A�*

loss]U+< ���       �	��g�Xc�A�*

lossC��=S��       �	��h�Xc�A�*

loss��.<j!��       �	P:i�Xc�A�*

loss�Ó<��o�       �	��i�Xc�A�*

loss�P�<{��W       �	�k�Xc�A�*

loss��=��0�       �	H�k�Xc�A�*

loss&��<"���       �	�nl�Xc�A�*

lossZ@<��[/       �	h%m�Xc�A�*

lossDP=!L��       �	s�m�Xc�A�*

loss8�9<��[�       �	�Zn�Xc�A�*

lossD�
<�V}�       �	��n�Xc�A�*

loss2�h<�Ԓ       �	Ůo�Xc�A�*

lossf4�<��:o       �	�Up�Xc�A�*

loss�� =��       �	�p�Xc�A�*

loss%�<�̄�       �	��q�Xc�A�*

lossfb<��       �	'gr�Xc�A�*

loss`�=b��       �	��r�Xc�A�*

loss)T<O�f�       �	k�s�Xc�A�*

lossv��;��Q�       �	�3t�Xc�A�*

loss:�=�[�       �	��t�Xc�A�*

lossa�<�1}�       �	cu�Xc�A�*

loss��=���       �	��u�Xc�A�*

loss��}<a��.       �	�v�Xc�A�*

loss��<MQ	�       �	�1w�Xc�A�*

loss�=h�       �	��w�Xc�A�*

loss] �<��}P       �	Z�x�Xc�A�*

lossE��<��t       �	>y�Xc�A�*

loss���<8���       �	i�y�Xc�A�*

lossE �;7"��       �	N�z�Xc�A�*

lossBy;';�y       �	�&{�Xc�A�*

loss�q�:�}>Y       �	+�{�Xc�A�*

loss.�:;Ғj	       �	�v|�Xc�A�*

loss y�<vd��       �	}�Xc�A�*

loss�t<�*�5       �	S<~�Xc�A�*

loss��<y7�       �	^�~�Xc�A�*

loss��;|�f       �	�n�Xc�A�*

loss�\=x�q       �	���Xc�A�*

lossi;���       �	7���Xc�A�*

lossr��=��{       �	Eh��Xc�A�*

loss!#>�9e       �	�m��Xc�A�*

loss[b�;�I��       �	���Xc�A�*

losso�y:P���       �	�̃�Xc�A�*

loss��b;�I��       �	����Xc�A�*

loss,H*;��       �	�?��Xc�A�*

losshҕ<�O��       �	[��Xc�A�*

loss�،<�
�#       �	����Xc�A�*

loss��;�e^�       �	ʉ��Xc�A�*

lossfc�<�7��       �	� ��Xc�A�*

loss��:^���       �	8׈�Xc�A�*

losss;:s��       �	�|��Xc�A�*

loss��9��W0       �	d]��Xc�A�*

lossĞ�;�O       �	���Xc�A�*

loss�P�<��d�       �	�ы�Xc�A�*

loss�8$<Fs       �	ۉ��Xc�A�*

loss��;&�       �	�<��Xc�A�*

lossF�Z<���       �	�2��Xc�A�*

loss�V >Rt��       �	�n��Xc�A�*

loss��:���o       �	q��Xc�A�*

loss�>�ED�       �	X��Xc�A�*

loss(��=%&5�       �	9��Xc�A�*

loss��=��&       �	Χ��Xc�A�*

loss/
Z<IB�#       �	R��Xc�A�*

loss�l*=~��       �	T���Xc�A�*

loss)�=�y�`       �	���Xc�A�*

loss���<�}ˊ       �	76��Xc�A�*

loss}��<�|x       �	9ѕ�Xc�A�*

loss|7=�8'       �	i��Xc�A�*

loss�o�;��2       �	���Xc�A�*

loss�4�=���       �	=֗�Xc�A�*

loss)�e<��[       �	\t��Xc�A�*

lossOk,<`���       �	���Xc�A�*

loss�v-=�3�A       �	����Xc�A�*

loss*.=�p2f       �	����Xc�A�*

loss��^=�,��       �	c+��Xc�A�*

loss��|<E^6       �	�̛�Xc�A�*

loss��=�<�       �	7o��Xc�A�*

loss�ޚ;%�:�       �	��Xc�A�*

loss�#&<5���       �	�ǝ�Xc�A�*

lossM:L<��8�       �	�f��Xc�A�*

loss��<�?nk       �	C��Xc�A�*

loss��<ł�a       �	ʥ��Xc�A�*

loss��<��dp       �	$Ԡ�Xc�A�*

loss/#.;��X       �	�{��Xc�A�*

loss�E=�:d�       �	�$��Xc�A�*

loss�2;��/�       �	M���Xc�A�*

loss^m�<Ĺ�#       �	���Xc�A�*

loss-=�F�       �	{���Xc�A�*

loss�#~<`?�G       �	�B��Xc�A�*

loss���<�4�       �	���Xc�A�*

loss:9<�C:�       �	P���Xc�A�*

lossN.<���M       �	*ʧ�Xc�A�*

lossĦ�;��       �	�d��Xc�A�*

loss��:�^�       �	���Xc�A�*

lossom�;���       �	
���Xc�A�*

lossv<��I�       �	l���Xc�A�*

loss��=5 ,       �	&5��Xc�A�*

loss�k=���Q       �	Jά�Xc�A�*

loss��H<J�>       �	�f��Xc�A�*

loss��$=]��       �	]j��Xc�A�*

loss�/�;�a�p       �	#���Xc�A�*

loss�D�<xU+e       �	؞��Xc�A�*

loss�=w���       �	4��Xc�A�*

loss���=�*       �	�ʱ�Xc�A�*

lossD�=�8��       �	�a��Xc�A�*

loss���<}�#�       �	����Xc�A�*

loss4[�<7"wx       �	V���Xc�A�*

loss%|�;���l       �	�@��Xc�A�*

lossr�T=�H�       �	8ڴ�Xc�A�*

lossEJF<�!S�       �	J���Xc�A�*

loss&��<���k       �	q���Xc�A�*

loss���=p�
       �	&7��Xc�A�*

loss(�<S�+�       �	����Xc�A�*

loss�w�<|f�m       �	���Xc�A�*

loss���;��ĸ       �	A���Xc�A�*

lossc��<�!=       �		p��Xc�A�*

loss[e�;��,A       �	5
��Xc�A�*

lossl7�=J۬�       �	����Xc�A�*

loss�w�<45�P       �	;:��Xc�A�*

lossT�\<g�Fr       �	����Xc�A�*

loss��;S�y       �	�c��Xc�A�*

losslt=��P�       �	'���Xc�A�*

loss�g�<�(}       �	Ȕ��Xc�A�*

losst<�<y��       �	�+��Xc�A�*

loss=��=�       �	i���Xc�A�*

loss���:���       �	5b��Xc�A�*

loss�@�;���       �	d"��Xc�A�*

lossܝ�;����       �		���Xc�A�*

loss�;:=���6       �	�c��Xc�A�*

lossd��;7�~7       �	����Xc�A�*

loss�=.!h�       �	`���Xc�A�*

loss�s�:����       �	����Xc�A�*

loss_ً<Q���       �	of��Xc�A�*

loss=�*=<Q�       �	��Xc�A�*

loss��<���       �	&���Xc�A�*

lossn+=r潏       �	?X��Xc�A�*

loss��<۩�       �	4���Xc�A�*

loss���;@��       �	֍��Xc�A�*

loss��:�b       �	Z+��Xc�A�*

lossfdk<�(�F       �	����Xc�A�*

loss��<&PL�       �	i��Xc�A�*

loss|�H=0|^�       �	���Xc�A�*

loss�=F���       �	ɬ��Xc�A�*

loss��B<yP~�       �	=G��Xc�A�*

loss���<�vG�       �	D���Xc�A�*

loss�F^<a��!       �	ds��Xc�A�*

loss�b+<��)       �	F	��Xc�A�*

lossb�>D���       �	���Xc�A�*

loss���<ࠞ�       �	;9��Xc�A�*

lossIB)<1�!       �	���Xc�A�*

lossE��<\�Cb       �	�i��Xc�A�*

loss��;P^��       �	?��Xc�A�*

lossH��;$�FO       �	*���Xc�A�*

loss8_�<�/w|       �	K���Xc�A�*

lossw��<�|�       �	�H��Xc�A�*

loss[��<�Nu�       �	����Xc�A�*

lossi��<�        �	_{��Xc�A�*

loss��<{���       �	@��Xc�A�*

loss��s;CY4;       �	#���Xc�A�*

loss�qI<��&�       �	z���Xc�A�*

loss�Z=� ��       �	-��Xc�A�*

loss�-<NC��       �	����Xc�A�*

loss��v=�L�#       �	o��Xc�A�*

loss�a=�D       �	h��Xc�A�*

loss�P<��Y�       �	���Xc�A�*

loss�8]<H���       �	A��Xc�A�*

loss��:�v�       �	����Xc�A�*

loss6��;�?h       �	R���Xc�A�*

loss�f=/^b�       �	�G��Xc�A�*

loss?�>:Iv5       �	����Xc�A�*

loss��f;P[�       �	����Xc�A�*

lossJ�;�h�;       �	,.��Xc�A�*

lossQ��<���E       �	`���Xc�A�*

lossC��;���       �	����Xc�A�*

loss,�Q<�,Yp       �	G��Xc�A�*

lossF�w<S�{�       �	���Xc�A�*

loss��;4a�       �	ˁ �Xc�A�*

loss̉�< ���       �	G�Xc�A�*

loss�	=�a��       �	�V�Xc�A�*

loss4��<�d��       �	��Xc�A�*

loss��h<\�t�       �	���Xc�A�*

loss���<�ǚ�       �	fL�Xc�A�*

loss�[�;Oh       �	���Xc�A�*

loss�j�;$���       �	`��Xc�A�*

loss� <,�x       �	�4�Xc�A�*

loss��
<D���       �	���Xc�A�*

lossM�;��V       �	Ę�Xc�A�*

losss�;1��r       �	O>�Xc�A�*

losseXp<u�A�       �	���Xc�A�*

loss3<r5�       �	�	�Xc�A�*

lossZ�=����       �	-B
�Xc�A�*

loss!6	=wm��       �	��
�Xc�A�*

loss���;8�n       �	5��Xc�A�*

lossc�<���       �	fM�Xc�A�*

loss<�	       �	��Xc�A�*

loss�V�<�F�       �	���Xc�A�*

loss�/@=�f��       �	�@�Xc�A�*

loss�Jr<��o       �	>��Xc�A�*

loss�)�=^}R�       �	s��Xc�A�*

lossC7�=����       �	'�Xc�A�*

loss8E:=U       �	��Xc�A�*

loss\�<'ux       �	�q�Xc�A�*

loss�VH<����       �	��Xc�A�*

loss��y=F�1q       �	E��Xc�A�*

loss��J=z�Y�       �	 ��Xc�A�*

loss�:=#���       �	�T�Xc�A�*

loss(Ӫ:�/�<       �	7��Xc�A�*

loss�@�=]���       �	���Xc�A�*

loss���<x��       �	{�Xc�A�*

loss�%=��a�       �	�Xc�A�*

loss�v<	�-)       �	Ĵ�Xc�A�*

losst6�;f�I�       �	�P�Xc�A�*

loss!��<dk��       �	P��Xc�A�*

loss�;<[vke       �	���Xc�A�*

lossv�=4T;       �	�0�Xc�A�*

loss�;����       �	*��Xc�A�*

loss�2�<\v�!       �	R_�Xc�A�*

loss=�=��	       �	��Xc�A�*

lossO��<����       �	���Xc�A�*

loss���<*��e       �	]2�Xc�A�*

loss�G<d$;       �	@��Xc�A�*

loss �J=_�       �	��Xc�A�*

lossw�;Z�e       �	�4�Xc�A�*

loss'k�<���2       �	���Xc�A�*

loss�(<��       �	q� �Xc�A�*

loss.h�<`dk~       �	�~!�Xc�A�*

loss:?_=���       �	�s"�Xc�A�*

lossE��<�vn|       �	�
#�Xc�A�*

loss&��;����       �	�#�Xc�A�*

loss�4=dy��       �	�L$�Xc�A�*

lossh��<̐�       �	v�$�Xc�A�*

loss�ɟ;�]H=       �	�%�Xc�A�*

loss=�,<��n       �	ѕ&�Xc�A�*

loss
G�;u�+       �	�-'�Xc�A�*

loss>7<��O       �	��'�Xc�A�*

loss�*<�}       �	�](�Xc�A�*

loss�<�^       �	��(�Xc�A�*

lossS�=s�@       �	t�)�Xc�A�*

loss��q=��       �	}�*�Xc�A�*

loss-3�;}��u       �	>+�Xc�A�*

lossȾ<���`       �	��+�Xc�A�*

loss {;<��	�       �	�,�Xc�A�*

loss��;�	�W       �	 b-�Xc�A�*

loss���<=��       �	�	.�Xc�A�*

lossY <*��       �	[�.�Xc�A�*

loss��;i�h�       �	T/�Xc�A�*

loss��<��       �	��/�Xc�A�*

losse�;�3&,       �	�0�Xc�A�*

loss�K�<�I�       �	;41�Xc�A�*

loss1��:�?P�       �	��1�Xc�A�*

lossq؝;���       �	�}2�Xc�A�*

lossHm=j���       �	N)3�Xc�A�*

loss$d_=d�       �	��3�Xc�A�*

loss�n�<��ď       �	�W4�Xc�A�*

loss<���L       �	-5�Xc�A�*

loss,��<?<�       �	�5�Xc�A�*

loss�ن<�#&�       �	�D6�Xc�A�*

loss&�:;?^�*       �	��6�Xc�A�*

loss�=����       �	yx7�Xc�A�*

loss��<PI/       �	\8�Xc�A�*

loss�f�<ta��       �	O�8�Xc�A�*

loss=��=�wz       �	��9�Xc�A�*

loss^�=HZjp       �	�b:�Xc�A�*

loss;�T=q�J       �	�:�Xc�A�*

loss���;�!��       �	��;�Xc�A�*

loss[=�@�       �	�=<�Xc�A�*

loss���=�A�       �	��<�Xc�A�*

loss�1= ��w       �	yv=�Xc�A�*

loss@�<�a       �	�>�Xc�A�*

loss6;�<�gp       �	��>�Xc�A�*

loss�u�=7��j       �	�I?�Xc�A�*

loss��;��+�       �	&�?�Xc�A�*

lossZ#[<>6�r       �	�}@�Xc�A�*

loss̩�< ��       �	�A�Xc�A�*

loss�]�;�4�       �	��A�Xc�A�*

lossܐ�;�l[�       �	ƦB�Xc�A�*

loss5<���       �	qC�Xc�A�*

loss!r�<�^T�       �	]4D�Xc�A�*

loss�.<Q,��       �	�D�Xc�A�*

loss$=k=d�P�       �	^gE�Xc�A�*

loss�a�;7��P       �	�oF�Xc�A�*

lossv%;v%-       �	�3G�Xc�A�*

loss[1z<W�       �	�G�Xc�A�*

lossJ��;2Y��       �	�H�Xc�A�*

loss*��<L��       �	.TI�Xc�A�*

loss�.�<�x�4       �	�J�Xc�A�*

loss#f�=��       �	�J�Xc�A�*

loss��<��*�       �	S�K�Xc�A�*

loss�1Y<��ō       �	<�L�Xc�A�*

loss��;��7�       �	^�M�Xc�A�*

loss��:l�'       �	�pN�Xc�A�*

lossfؼ:;��J       �	BO�Xc�A�*

lossd3�;�=�_       �	P�Xc�A�*

loss�y�<��       �	~�P�Xc�A�*

lossں�;m��:       �	�Q�Xc�A�*

loss|�=�ߢ       �	P:S�Xc�A�*

loss8�)<��R�       �	��S�Xc�A�*

loss�^=�ŕ�       �	�T�Xc�A�*

lossە�;�r��       �	<NU�Xc�A�*

loss_��<J�k       �	��U�Xc�A�*

loss�<<��S       �	��V�Xc�A�*

loss�5
=�v�       �	�pW�Xc�A�*

loss��<��B�       �	X�Xc�A�*

loss��;oB�       �	2�X�Xc�A�*

loss���<a�s       �	�lY�Xc�A�*

loss��=v���       �	�	Z�Xc�A�*

loss��<.�/'       �	ߦZ�Xc�A�*

losseIT=�v��       �	>w[�Xc�A�*

loss�^�;�r��       �	�\�Xc�A�*

lossZb�<��       �	��\�Xc�A�*

losst�I<��^o       �	8L]�Xc�A�*

lossW��<�$       �	^�Xc�A�*

loss���<J��       �	D�^�Xc�A�*

loss��/<�dM~       �	W\_�Xc�A�*

loss�)F=��JB       �	%`�Xc�A�*

loss�K6=c��       �	7�`�Xc�A�*

losss��=$	P�       �	'La�Xc�A�*

loss4�?;����       �	�a�Xc�A�*

loss��&=�g�       �	B�b�Xc�A�*

lossL��<��C�       �	�c�Xc�A�*

lossn^<.�<@       �	�_d�Xc�A�*

loss�G�<�#N~       �	��d�Xc�A�*

loss�3<���       �	��e�Xc�A�*

loss2�2=X?��       �	/f�Xc�A�*

loss�j=d�J�       �	c�f�Xc�A�*

lossi2>�i$       �	*pg�Xc�A�*

loss��=����       �	wh�Xc�A�*

loss~R�=��       �	!�h�Xc�A�*

lossZQ<��]P       �	�hi�Xc�A�*

losssWg;��Dc       �	�j�Xc�A�*

lossi��<�[3�       �	H�j�Xc�A�*

loss��<�T��       �	�Ek�Xc�A�*

loss�IM=F�|       �	V�k�Xc�A�*

loss*! <)^       �	W�l�Xc�A�*

loss8M=�;�3       �	6m�Xc�A�*

loss�e{<���O       �	��m�Xc�A�*

loss�-<��l       �	�yn�Xc�A�*

loss��;\���       �	�o�Xc�A�*

lossF�=/A�'       �	öo�Xc�A�*

loss_��:=F{�       �	*Up�Xc�A�*

lossy$:"W��       �	:�p�Xc�A�*

loss��0<�HP       �	��q�Xc�A�*

loss�N�=qz�       �	$r�Xc�A�*

loss��<�h,       �	=�r�Xc�A�*

lossO�=<#Q��       �	.Ss�Xc�A�*

loss���;�6NV       �	h�s�Xc�A�*

loss�K�=�/�       �	�t�Xc�A�*

loss�i=��<j       �	�u�Xc�A�*

loss@�y<\�       �	ȵu�Xc�A�*

lossv��<�x�l       �	�Yv�Xc�A�*

loss���<9�       �	@�v�Xc�A�*

lossa,�;��b*       �	�w�Xc�A�*

lossN1g=��L�       �	n3x�Xc�A�*

lossn�F<�m@       �	�x�Xc�A�*

loss�Ke=�)R�       �	�fy�Xc�A�*

loss)6<��K�       �	tz�Xc�A�*

losss}<�3A�       �	�z�Xc�A�*

loss��<!��       �	c�{�Xc�A�*

loss�+�<G��       �	�|�Xc�A�*

loss���<�̃�       �	�|�Xc�A�*

loss��=���       �	�U}�Xc�A�*

loss�ּ<P��       �	��}�Xc�A�*

loss�٣;���3       �	R�~�Xc�A�*

loss���;�#y       �	�y�Xc�A�*

loss4;�f��       �	@��Xc�A�*

loss�7�<�'e�       �	���Xc�A�*

loss~ <�Vx       �	���Xc�A�*

loss32<q��       �	����Xc�A�*

loss�)<����       �	����Xc�A�*

loss8U�;�!#m       �	rO��Xc�A�*

loss���;t1I�       �	����Xc�A�*

loss�5�<�ye       �	���Xc�A�*

loss��;I��       �	����Xc�A�*

loss�=�E��       �	���Xc�A�*

loss��F<��mu       �	Z+��Xc�A�*

loss8k>=.\s$       �	w��Xc�A�*

loss���<��,�       �	���Xc�A�*

loss-��=C6>       �	����Xc�A�*

loss��=ʳ��       �	�Z��Xc�A�*

loss�˵=�"m&       �	���Xc�A�*

loss�:w<x=SL       �	Mۍ�Xc�A�*

loss��=��S       �	�ǎ�Xc�A�*

loss{�=<�2       �	Œ��Xc�A�*

lossl\=e�       �	�֐�Xc�A�*

lossﾖ=K�e       �	x��Xc�A�*

lossE$�;��       �	)��Xc�A�*

loss �;XQ{       �	Ò�Xc�A�*

loss��*=�o��       �	5^��Xc�A�*

loss� �;�F3�       �	�k��Xc�A�*

loss��Q<��ry       �	���Xc�A�*

loss�b;��|       �	����Xc�A�*

loss�m�<�2#&       �	K̖�Xc�A�*

loss��;f�K       �	�ȗ�Xc�A�*

loss!�.=k4oo       �	�a��Xc�A�*

loss�=��       �	)$��Xc�A�*

loss��<J��       �	Y���Xc�A�*

loss@CM<�i�        �	�i��Xc�A�*

loss���;�@7�       �	���Xc�A�*

loss���;3�!�       �	稛�Xc�A�*

loss7h�;���       �	�@��Xc�A�*

lossH�;��a       �	v���Xc�A�*

lossrO<���[       �	ꖝ�Xc�A�*

loss�<���       �	�B��Xc�A�*

loss
Α<�z�       �	]���Xc�A�*

loss�_�<���       �	^���Xc�A�*

lossl��<��q       �	Й��Xc�A�*

lossl�=&��       �	
/��Xc�A�*

loss(ۍ=!Ӂ�       �	���Xc�A�*

lossŢ�< Hn	       �	B���Xc�A�*

lossjq;s4R'       �	�Q��Xc�A�*

loss�S�;E��e       �	#���Xc�A�*

loss��&<o��       �	f���Xc�A�*

loss�=Ѿci       �	B��Xc�A�*

loss��c=��\i       �	)��Xc�A�*

loss�rm=���       �	���Xc�A�*

lossD^;t�h�       �	=+��Xc�A�*

loss���=��JP       �	.ɧ�Xc�A�*

loss�tJ=$ux       �	�g��Xc�A�*

loss;�3;���j       �	:��Xc�A�*

lossž+<�-��       �	���Xc�A�*

loss�� = ͏1       �	W_��Xc�A�*

lossV��<��       �	l	��Xc�A�*

lossT��<�{�       �	����Xc�A�*

loss�b <�u�"       �	�G��Xc�A�*

loss;��=J� w       �	����Xc�A�*

loss��z;�ͫ       �	���Xc�A�*

losss:<��#�       �	O=��Xc�A�*

lossH�<,P��       �	���Xc�A�*

loss���;����       �	����Xc�A�*

loss���;>��       �	6#��Xc�A�*

loss�G�<V��+       �	�°�Xc�A�*

lossʠD=Ŋ&7       �	ro��Xc�A�*

loss��<t�       �	Q��Xc�A�*

loss��<QZ*�       �	?���Xc�A�*

loss7jc<�(�       �	�Q��Xc�A�*

loss��<�@"�       �	���Xc�A�*

loss��3<�6h�       �	����Xc�A�*

loss��#=�u�7       �	�/��Xc�A�*

loss+q=��*U       �	�ϵ�Xc�A�*

lossz<��B�       �	Ox��Xc�A�*

loss2�<=��       �	8��Xc�A�*

loss�·;J�Xi       �	 ���Xc�A�*

lossn��<2��b       �	GZ��Xc�A�*

loss�(4=��%�       �	<���Xc�A�*

loss��<� ��       �	V���Xc�A�*

loss�޸=` J�       �	�@��Xc�A�*

loss�;<��2       �	���Xc�A�*

loss�=��=       �	d���Xc�A�*

losse[~;�� -       �	m9��Xc�A�*

loss	<w��       �	߼�Xc�A�*

lossk0�<E�AJ       �	0���Xc�A�*

loss䍑<�|��       �	���Xc�A�*

loss�~W<�d]       �	b���Xc�A�*

lossa�M=~i�	       �	t_��Xc�A�*

loss݋ =�B�.       �	&���Xc�A�*

loss.��;2Z۰       �	O���Xc�A�*

loss�#�=��-:       �	"T��Xc�A�*

lossCtS=�<Ҫ       �	����Xc�A�*

loss��9<�u:�       �	q���Xc�A�*

losss�2<�*�d       �	� ��Xc�A�*

lossìB<I/       �	_'��Xc�A�*

lossLH<8͊�       �	����Xc�A�*

loss��5=n�Ɣ       �	aq��Xc�A�*

loss���<���       �	9��Xc�A�*

lossc:p<�<g�       �	J���Xc�A�*

loss��=	c�]       �	����Xc�A�*

loss���<I,��       �	���Xc�A�*

loss��<�=x       �	����Xc�A�*

lossH�=�*�       �	����Xc�A�*

loss�!�;��q-       �	q���Xc�A�*

loss�c`<�lZ       �	����Xc�A�*

loss_�A=_��       �	$���Xc�A�*

loss��>�P*`       �	?9��Xc�A�*

loss��<3�N       �	�3��Xc�A�*

loss!��<����       �	�3��Xc�A�*

loss��<0H       �	����Xc�A�*

lossH�=��_       �	]j��Xc�A�*

lossH�
<�ak�       �	N
��Xc�A�*

lossO�<���       �	����Xc�A�*

loss>�<��=        �	d=��Xc�A�*

lossvE�;J�n       �	1	��Xc�A�*

loss́:<���       �	���Xc�A�*

lossw-�;�y�       �	�E��Xc�A�*

loss�:�<��[�       �	"���Xc�A�*

lossk�#=旤       �	�{��Xc�A�*

loss�yn<�7�       �	m��Xc�A�*

loss3��; xW�       �	4���Xc�A�*

loss��m<S�߀       �	W��Xc�A�*

lossd�<��j0       �	����Xc�A�*

lossqhK<����       �	����Xc�A�*

loss���<����       �	1��Xc�A�*

loss��!>U|�2       �	{���Xc�A�*

loss�T�;a��       �	o~��Xc�A�*

loss\t�<Ǚ�       �	`!��Xc�A�*

loss@_�<�k�1       �	����Xc�A�*

loss�y�<E;��       �	2q��Xc�A�*

loss?ؿ=�Z��       �	���Xc�A�*

loss�`;�M�u       �	����Xc�A�*

loss3�Z<S=�       �	�`��Xc�A�*

loss�
�;L�)�       �	��Xc�A�*

loss�/}<���       �	���Xc�A�*

lossd��<ɽr       �	yY��Xc�A�*

loss|`�<����       �	���Xc�A�*

loss�]<Y�{�       �	����Xc�A�*

loss�Ù=�m��       �	�W��Xc�A�*

loss4�<΂:       �	���Xc�A�*

loss�r�;���       �	+P��Xc�A�*

loss���;�R��       �	���Xc�A�*

lossh�#<�] u       �	����Xc�A�*

lossR��;�L       �	�`��Xc�A�*

loss%e�;-/�       �	��Xc�A�*

lossh�;�J��       �	�B��Xc�A�*

loss/n;J��)       �	����Xc�A�*

loss��.=�|m�       �	^���Xc�A�*

loss�8�;FMX�       �	����Xc�A�*

loss���<R�c�       �	Έ��Xc�A�*

loss:��=N}       �	�3��Xc�A�*

loss�H<�k;�       �	 ���Xc�A�*

loss䖖<�۾�       �	����Xc�A�*

loss���<{;�n       �	�F��Xc�A�*

loss��=?��;       �	����Xc�A�*

loss?�<h'�U       �	g���Xc�A�*

loss��=���       �	@h��Xc�A�*

loss��w<�*�s       �	$`��Xc�A�*

loss��]=��M�       �	m���Xc�A�*

losslܟ;��dr       �	:���Xc�A�*

lossSڷ:a}��       �	o,��Xc�A�*

loss2�J=˯=       �	`���Xc�A�*

lossI:"={���       �	-z��Xc�A�*

loss�V<-f:       �	���Xc�A�*

lossh��<}+H       �	m���Xc�A�*

lossC|<�ՙr       �	���Xc�A�*

loss4�=+�D�       �	hu��Xc�A�*

loss݅=2�       �	���Xc�A�*

losslg�<n�H       �	j���Xc�A�*

lossG<�
>       �	\Z��Xc�A�*

lossMʲ=�;#�       �	I���Xc�A�*

loss���<����       �	����Xc�A�*

loss��i<֭�+       �	8��Xc�A�*

loss�M�<P@��       �	����Xc�A�*

loss<!�E�       �	}u��Xc�A�*

loss��;M�I       �	���Xc�A�*

loss]X�<�p/�       �	����Xc�A�*

loss�<鮿�       �	�@ �Xc�A�*

loss��<M^�       �	�� �Xc�A�*

loss�;�:|w߃       �	z�Xc�A�*

loss�X<l/       �	b�Xc�A�*

loss�ѫ<�4�       �	���Xc�A�*

lossq�C<���I       �	�f�Xc�A�*

loss�s=��NS       �	5�Xc�A�*

loss@?=,��       �	;��Xc�A�*

loss���=�ˎ�       �	���Xc�A�*

loss�7�=��ܤ       �	n�Xc�A�*

lossF4':�)E       �	��Xc�A�*

loss8��<)��       �	��Xc�A�*

loss�=,��       �	ݖ�Xc�A�*

loss`_;�2��       �	+2	�Xc�A�*

loss�ڦ<0���       �	��	�Xc�A�*

loss�=�'4       �	�j
�Xc�A�*

loss��0=�s�@       �	�D�Xc�A�*

loss��<c�~        �	���Xc�A�*

loss7)t;�=�       �	�Xc�A�*

loss�~'<����       �	��Xc�A�*

lossp�<b�:       �	���Xc�A�*

lossi�]<�O �       �	�R�Xc�A�*

loss,��<K�%�       �	���Xc�A�*

loss��<���       �	��Xc�A�*

loss8�<ܺ�       �	V,�Xc�A�*

lossw;�<�Kh       �	��Xc�A�*

lossH{�<@X��       �	���Xc�A�*

lossA�:;{0�M       �	u9�Xc�A�*

loss_�.;��C       �	.��Xc�A�*

loss(F�= �c�       �	3��Xc�A�*

lossq3�<��^y       �	j0�Xc�A�*

loss��<>�       �	���Xc�A�*

loss���;]\�A       �	r�Xc�A�*

lossc�/<'�7�       �	_�Xc�A�*

loss�!<��Z       �	���Xc�A�*

loss���;.b       �	e�Xc�A�*

lossd�>pN�       �	��Xc�A�*

loss���;��
�       �	*��Xc�A�*

lossF"<1�1�       �	fN�Xc�A�*

loss��9=IO��       �	���Xc�A�*

lossq>�:9�:       �	��Xc�A�*

loss��f:���       �	"6�Xc�A�*

loss .C:��zi       �	�K�Xc�A�*

loss#�k<�q�       �	��Xc�A�*

loss!ܕ;���-       �	��Xc�A�*

loss|$�=d>;       �	+4�Xc�A�*

loss���=����       �	^��Xc�A�*

loss�](=ʖ-w       �	�v�Xc�A�*

loss���;*{�       �	u �Xc�A�*

loss2m=N$
       �	˽ �Xc�A�*

loss(,>ICY�       �	�]!�Xc�A�*

loss�W6<Q�õ       �	� "�Xc�A�*

loss�?�;!�|�       �	n�"�Xc�A�*

loss��=�h�       �	�H#�Xc�A�*

loss́F<J
o       �	F�#�Xc�A�*

loss�Ȳ<��       �	!�$�Xc�A�*

loss6�%<\@       �	@2%�Xc�A�*

loss��;��       �	�%�Xc�A�*

loss��;0{%       �	��&�Xc�A�*

loss֕�<K��       �	�j'�Xc�A�*

loss�G<8�xl       �	�	(�Xc�A�*

loss6�=!��       �	��(�Xc�A� *

loss؛�<ǯ�P       �	k~)�Xc�A� *

loss%�x<���%       �	n0*�Xc�A� *

loss�L�<>���       �	��*�Xc�A� *

loss�h�=��0�       �	��+�Xc�A� *

loss+=�X�?       �	h,�Xc�A� *

loss��<Y���       �	�-�Xc�A� *

loss���<-��s       �	u�-�Xc�A� *

loss�x�<'�3       �	�J.�Xc�A� *

loss��<�uAQ       �	*/�Xc�A� *

lossH۱<w
*�       �	D�/�Xc�A� *

loss4�2=�WT3       �	K\0�Xc�A� *

loss�f <�/xN       �	?1�Xc�A� *

loss&T=9V2       �	�1�Xc�A� *

loss�=LB:�       �	8h2�Xc�A� *

loss$��;�Y�h       �	/3�Xc�A� *

lossQ�Z=~�(�       �	�3�Xc�A� *

loss���<�}6       �	�4�Xc�A� *

loss�� =�%$�       �	�<5�Xc�A� *

loss��;��~4       �	�6�Xc�A� *

lossZ�%=3]ո       �	P�6�Xc�A� *

loss�b=K�g       �	K�7�Xc�A� *

loss���<��?/       �	mU8�Xc�A� *

loss��_;��
       �	N9�Xc�A� *

lossEF=a(x6       �	�9�Xc�A� *

loss}/3<�       �	�_:�Xc�A� *

loss���;�b5�       �	c);�Xc�A� *

loss���;O�1       �	�;�Xc�A� *

loss��<��A       �	�r<�Xc�A� *

loss`D�<,��A       �	h"=�Xc�A� *

loss�?�;�&�#       �	��=�Xc�A� *

lossj<o�H       �	Vc>�Xc�A� *

loss�:�<��F       �	�?�Xc�A� *

loss��=z��       �	�?�Xc�A� *

loss�O<���       �	�D@�Xc�A� *

loss��;D�h       �	��@�Xc�A� *

loss��<�g�       �	�A�Xc�A� *

loss�׋;�R�       �	�	C�Xc�A� *

loss4�C;q>��       �	��C�Xc�A� *

loss�]�:{(	1       �	9{D�Xc�A� *

losse�M<�ʉ       �	�<E�Xc�A� *

loss�&�;h�N       �	F�Xc�A� *

losshuH<:���       �	9G�Xc�A� *

loss��I;%^(0       �	��G�Xc�A� *

loss.��<T\.       �	5EH�Xc�A� *

lossX$G;r�:�       �	��H�Xc�A� *

loss��9��W       �	��I�Xc�A� *

losst�;��r       �	�J�Xc�A� *

lossn=�:p�Qt       �	"K�Xc�A� *

loss��)=Y�&       �	��K�Xc�A� *

loss&��;@�I�       �	�OL�Xc�A� *

loss��9���       �	}�L�Xc�A� *

loss��;�kn       �	�yM�Xc�A� *

loss֣=ԺD�       �	=N�Xc�A� *

loss��]:�E5�       �	Y�N�Xc�A� *

lossxU/>���k       �	�;O�Xc�A� *

lossn��<��E�       �	��O�Xc�A� *

loss���<C��       �	RbP�Xc�A� *

loss�I�<NnX       �	KQ�Xc�A� *

loss\�<4�'�       �	��Q�Xc�A� *

loss��=n#G�       �	�;R�Xc�A� *

lossChK=*��       �	��R�Xc�A� *

loss�G�;�At       �	�yS�Xc�A� *

loss3�;��XV       �	�U�Xc�A� *

loss��=�?�       �	4�U�Xc�A� *

loss ��<�m�       �	�AV�Xc�A� *

loss�3:=�K       �	s�V�Xc�A� *

loss�L=~;�       �	@jW�Xc�A� *

loss �6=���       �	&�W�Xc�A� *

lossŀ�<��ُ       �	��X�Xc�A� *

loss�k<� Cg       �	�UY�Xc�A� *

loss�5�;�>wM       �	�Y�Xc�A� *

lossz�<Y�w       �	[�Z�Xc�A� *

loss���;�W�       �	�,[�Xc�A� *

loss,V<���P       �	��[�Xc�A� *

loss��;~�'       �	��\�Xc�A� *

loss��;���       �	�3]�Xc�A� *

loss+M;`�=�       �	�]�Xc�A� *

loss���:��        �	Lp^�Xc�A� *

loss�C�;W�Od       �		_�Xc�A� *

loss	�=���       �	S�_�Xc�A� *

loss��=��       �	�`�Xc�A� *

lossV��<q8       �	qra�Xc�A� *

lossV�i=�       �	�Fb�Xc�A� *

loss�<�;��       �	��b�Xc�A� *

loss�<�<�|p       �	ƈc�Xc�A� *

loss �><���f       �	!d�Xc�A� *

loss�t/;��       �	~�d�Xc�A� *

loss�|�<�=�(       �	�^e�Xc�A� *

loss��<�v�8       �	�f�Xc�A� *

lossH%O=8*�w       �	�f�Xc�A� *

loss���<,�+	       �	$Fg�Xc�A� *

loss��t=�{       �	K�g�Xc�A� *

loss|:=\*f�       �	l�h�Xc�A� *

loss=8;����       �	<.i�Xc�A� *

loss�`s;�
�       �	��i�Xc�A� *

loss��7<�N �       �	}k�Xc�A� *

loss��|=�b��       �	'�k�Xc�A� *

losss\�<�m�       �	�Il�Xc�A� *

lossbL=l �}       �	��l�Xc�A� *

loss��=�^C�       �	�m�Xc�A� *

loss)Z;H�b-       �	�Nn�Xc�A� *

loss\�<�oB�       �	o�Xc�A� *

loss��i;;>��       �	��o�Xc�A� *

loss���;!�+�       �	�Vp�Xc�A� *

loss�`�<�$�       �	ޭ��Xc�A� *

lossR<��<u       �	8K��Xc�A� *

loss$5=�T�       �	���Xc�A� *

loss�'=�I�B       �	���Xc�A� *

losshIw=��cJ       �	�j��Xc�A� *

loss�{=��TO       �	-��Xc�A� *

loss���<�=mq       �	���Xc�A� *

loss��<�?B�       �	#���Xc�A� *

loss
/n=)�A5       �	rm��Xc�A� *

loss���<�@e       �	b��Xc�A� *

loss���;���_       �	m��Xc�A� *

loss�k<h�       �	����Xc�A� *

loss��<���       �	�Д�Xc�A� *

lossrwq<�&��       �	��Xc�A� *

loss��{<�ym�       �	����Xc�A� *

lossQ]=�Xp�       �	#��Xc�A� *

loss��0:a�c       �	��Xc�A� *

loss��<i?�       �	[x��Xc�A� *

loss�k]<0���       �	�U��Xc�A� *

loss/�<�b��       �	�q��Xc�A� *

loss4��<�8       �	����Xc�A� *

loss��=�V|4       �	�L��Xc�A� *

lossn�a<K��&       �	` ��Xc�A� *

loss���=6���       �	����Xc�A�!*

loss�EH<[mr�       �	Lo��Xc�A�!*

loss�ְ;�       �	�B��Xc�A�!*

loss��Z<6��       �		ߟ�Xc�A�!*

loss�,�;��u       �	w��Xc�A�!*

lossʱ!=��       �	�=��Xc�A�!*

loss�,�<2��5       �	Jҡ�Xc�A�!*

loss�Z<o�{       �	�h��Xc�A�!*

loss#<0��       �	X���Xc�A�!*

loss���;؛�       �	@���Xc�A�!*

loss�B<?��       �	u9��Xc�A�!*

lossF�;����       �	5Ԥ�Xc�A�!*

losso��<�$2�       �	?s��Xc�A�!*

loss�Fh;:�	B       �	�!��Xc�A�!*

loss
� =�'�!       �	�2��Xc�A�!*

loss��=�s)       �	�է�Xc�A�!*

loss�B=Bd�       �	-z��Xc�A�!*

loss�(�=���       �	^��Xc�A�!*

loss�B1<���/       �	ĵ��Xc�A�!*

loss��:�i�       �	PT��Xc�A�!*

loss�T=�Hl       �	���Xc�A�!*

loss�U�<J��        �	ۉ��Xc�A�!*

loss���<H90�       �	=*��Xc�A�!*

lossW%=�ɩ�       �	�Ŭ�Xc�A�!*

loss�.�<����       �	�]��Xc�A�!*

loss��=<��E       �	����Xc�A�!*

loss�d�9U}W}       �	���Xc�A�!*

loss�W�;"[`<       �	O=��Xc�A�!*

loss:=���       �	�د�Xc�A�!*

loss�T< �;�       �	�r��Xc�A�!*

loss�Ґ=5       �	E��Xc�A�!*

loss�YT<𱲮       �	����Xc�A�!*

lossNq�;4	       �	F?��Xc�A�!*

loss�j";��
�       �	���Xc�A�!*

loss�<;����       �	���Xc�A�!*

lossߠS<�58/       �	G��Xc�A�!*

loss��*=��/'       �	S���Xc�A�!*

loss6�=��.       �	�U��Xc�A�!*

lossO�;l�H�       �	���Xc�A�!*

lossV];�U�       �	����Xc�A�!*

loss��m<��       �	���Xc�A�!*

loss��i<ۖZ�       �	����Xc�A�!*

loss��V;����       �	D��Xc�A�!*

lossZ�
=�f��       �	L��Xc�A�!*

lossa�<���W       �	�{��Xc�A�!*

loss�)h<�/`       �	���Xc�A�!*

lossW��<�y�       �	d���Xc�A�!*

lossČ�;�x��       �	vQ��Xc�A�!*

loss4��<�<á       �	���Xc�A�!*

lossj1'=�h�       �	����Xc�A�!*

loss�0;K�       �	9&��Xc�A�!*

loss��<^af�       �	�Ƚ�Xc�A�!*

loss�g;�y�       �	c��Xc�A�!*

loss\�=]c�       �	����Xc�A�!*

loss̍A<_^��       �	��Xc�A�!*

loss��<9�ģ       �	v���Xc�A�!*

loss�f�<|��S       �	�>��Xc�A�!*

loss_�<�K'       �	@���Xc�A�!*

loss�5�<�n�8       �	`r��Xc�A�!*

loss��<O���       �	o��Xc�A�!*

loss1E�9��"_       �	{���Xc�A�!*

loss-1`<���       �	 <��Xc�A�!*

lossD+L=�3}       �	����Xc�A�!*

loss�=�:9�
{       �	[���Xc�A�!*

loss_�G<�u       �	d:��Xc�A�!*

lossz�<�c�6       �	����Xc�A�!*

loss|�<����       �	�h��Xc�A�!*

loss,�V=�7��       �	���Xc�A�!*

loss���<7#.       �	̙��Xc�A�!*

loss��:5TN�       �	�5��Xc�A�!*

loss��;sW��       �	���Xc�A�!*

loss�P�<Nd@�       �	�i��Xc�A�!*

lossh��;C;m�       �	���Xc�A�!*

loss.��<�u�       �	���Xc�A�!*

loss���<����       �	M��Xc�A�!*

loss�=T��~       �	����Xc�A�!*

loss<��:ʑ��       �	/���Xc�A�!*

lossd�H;9mI       �	�&��Xc�A�!*

loss�g�<��')       �	����Xc�A�!*

lossH�#=T�r$       �	e��Xc�A�!*

loss��;F�Z�       �	;���Xc�A�!*

lossf/�:�pYz       �	S{��Xc�A�!*

loss�*�=W��       �	���Xc�A�!*

loss���;M���       �	����Xc�A�!*

lossa�B;jQ�       �	�m��Xc�A�!*

lossǒ=	�1�       �	� ��Xc�A�!*

loss��<#�+       �	����Xc�A�!*

lossz�<d@�       �	�Z��Xc�A�!*

lossύ�<�L�       �	����Xc�A�!*

loss���<&�YE       �	n���Xc�A�!*

lossҝ�:���]       �	�;��Xc�A�!*

loss�<y@0       �	:���Xc�A�!*

loss{��:Ŭ]t       �	����Xc�A�!*

lossu�:@�2�       �	�7��Xc�A�!*

loss�R�=�o]	       �	����Xc�A�!*

loss�7y;�5T       �	����Xc�A�!*

loss-r�;B�)       �	;8��Xc�A�!*

loss�<�N�&       �	A���Xc�A�!*

loss=�c       �	v���Xc�A�!*

loss� �:%f�       �	�,��Xc�A�!*

loss�o:�#�       �	}���Xc�A�!*

loss�r�;XB�       �	�g��Xc�A�!*

loss?SS<�Kd       �	�	��Xc�A�!*

lossA�=��`r       �	+���Xc�A�!*

loss�g=Ԫ��       �	�E��Xc�A�!*

loss��4=�`F       �	O���Xc�A�!*

loss��]<���1       �	�y��Xc�A�!*

loss��m=.ꨫ       �	��Xc�A�!*

loss_�C=�q�       �	@���Xc�A�!*

loss$��<��xX       �	l\��Xc�A�!*

lossm�;��Y�       �	(���Xc�A�!*

loss@->=T�P�       �	���Xc�A�!*

lossOH�;I���       �	D1��Xc�A�!*

loss��x<0O�       �	����Xc�A�!*

loss1��<����       �	܁��Xc�A�!*

loss�a';��.T       �	���Xc�A�!*

lossTi!;�~��       �	����Xc�A�!*

loss��;�o�       �	{L��Xc�A�!*

loss��C<^��	       �	���Xc�A�!*

loss +=(�2�       �	���Xc�A�!*

loss{�;�0       �	</��Xc�A�!*

loss�>P�q       �	����Xc�A�!*

loss`�<Z��d       �	=}��Xc�A�!*

loss���;X-p       �	z���Xc�A�!*

lossl��:-�z       �	�_��Xc�A�!*

loss���<��;       �	����Xc�A�!*

losso�;�1�       �	q���Xc�A�!*

loss�xQ<4^�       �	e9��Xc�A�!*

loss�);0_�       �	c���Xc�A�"*

lossFހ<��N       �	�{��Xc�A�"*

loss�P<lm;#       �	���Xc�A�"*

loss.�<��7�       �	����Xc�A�"*

lossi�W<��$       �	Z��Xc�A�"*

lossɲ�=o+��       �	����Xc�A�"*

loss�ԕ<��]�       �	˝��Xc�A�"*

loss�n]<^�5k       �	 9��Xc�A�"*

loss�՛<�4!       �	�5��Xc�A�"*

loss�#<�D(       �	����Xc�A�"*

loss��;�F�2       �	�s��Xc�A�"*

loss�;� ��       �	���Xc�A�"*

loss�v ;)V�&       �	����Xc�A�"*

loss�<b��       �	:\��Xc�A�"*

loss�c<���       �	@���Xc�A�"*

loss��:C7H       �	����Xc�A�"*

loss�PV<�p��       �	z7��Xc�A�"*

loss!M�;���       �	����Xc�A�"*

loss�m�<�>�"       �	b��Xc�A�"*

loss=�=WY�       �	j���Xc�A�"*

loss�=�;J��       �	���Xc�A�"*

lossOvb:�Б�       �	�)��Xc�A�"*

lossj��<��       �	>���Xc�A�"*

loss�%t;4�M*       �	���Xc�A�"*

lossN�x<%��       �	^��Xc�A�"*

loss|�=5�;       �	��  Yc�A�"*

loss�i�=��.       �	�( Yc�A�"*

loss��<���Z       �	f� Yc�A�"*

lossT�<��_       �	�V Yc�A�"*

loss=��:S���       �	�� Yc�A�"*

loss��s=�<��       �	�� Yc�A�"*

loss�2"=x{׿       �	�# Yc�A�"*

loss,�;=fG       �	
� Yc�A�"*

lossC"�:�x`       �	�g Yc�A�"*

loss֎�:����       �	W Yc�A�"*

loss|��<Z���       �		� Yc�A�"*

loss� <kM��       �	ǁ Yc�A�"*

lossn�<<5�       �	W Yc�A�"*

lossVA�<�Ѳ       �	�E	 Yc�A�"*

losshUU=&7�       �	�)
 Yc�A�"*

loss�~;L�O       �	w Yc�A�"*

loss�[;z=>       �	�� Yc�A�"*

loss��{<�,�       �	S@ Yc�A�"*

loss�aB=�p       �	{� Yc�A�"*

loss��E<��.       �	ur Yc�A�"*

lossa�;7�_       �	� Yc�A�"*

loss�ȷ=
a�n       �	�� Yc�A�"*

loss�%/<h�C       �	�> Yc�A�"*

loss��<��X       �	�� Yc�A�"*

loss���<�t�       �	%w Yc�A�"*

loss�I;��pE       �	6 Yc�A�"*

loss���;� ^       �	n� Yc�A�"*

loss��=�SE       �	GW Yc�A�"*

loss�7e=�uh�       �	'� Yc�A�"*

loss�0�;W��/       �	�� Yc�A�"*

loss2�<4�3$       �	-> Yc�A�"*

loss�>�<���V       �	� Yc�A�"*

loss�T�;%3(       �	Wy Yc�A�"*

loss��|<��C       �	Y Yc�A�"*

loss=�&<���       �	� Yc�A�"*

loss�T�;0Q��       �	χ Yc�A�"*

loss�6�;��       �	�# Yc�A�"*

loss��*<�iL       �	�� Yc�A�"*

loss&�H<�k��       �	dY Yc�A�"*

loss�y8<RfA#       �	�� Yc�A�"*

loss�w�=���       �	�� Yc�A�"*

loss�OB=r#D@       �	+ Yc�A�"*

loss|o�<��       �	 Yc�A�"*

lossd�=��        �	/� Yc�A�"*

loss�)�<�!�       �	>x Yc�A�"*

lossJ�<�:�E       �	� Yc�A�"*

loss���<���       �	� Yc�A�"*

loss���<�M��       �	1z Yc�A�"*

loss>�<���y       �	{  Yc�A�"*

losstÕ<�!l�       �	?�  Yc�A�"*

loss�v;�	"�       �	C! Yc�A�"*

loss�^�<3Pl       �	�! Yc�A�"*

losse�h=c��       �	" Yc�A�"*

loss6�s;��'�       �	z# Yc�A�"*

loss*M<R
$S       �	c�# Yc�A�"*

loss;��;U�V�       �	b�$ Yc�A�"*

loss��<3q�       �	�% Yc�A�"*

loss&��<~�9�       �	& Yc�A�"*

loss�"=e���       �	w�& Yc�A�"*

lossWek<b�0j       �	�8' Yc�A�"*

lossˠ=o|       �	��' Yc�A�"*

loss	�?=�T       �	.s( Yc�A�"*

loss���=Ea,       �	�) Yc�A�"*

loss,<%���       �	4�) Yc�A�"*

loss���:q~       �	E* Yc�A�"*

loss�jG=؁�       �	��* Yc�A�"*

loss�z<�U�I       �	�|+ Yc�A�"*

loss���<�ڀ�       �	, Yc�A�"*

loss���<I�[       �	]�, Yc�A�"*

loss/�<C��B       �	�}- Yc�A�"*

loss�f!<O�SF       �	�!. Yc�A�"*

loss�E?;��       �	^�. Yc�A�"*

loss	=���       �	Y/ Yc�A�"*

loss�k�;<��       �	0 Yc�A�"*

loss��;c�>       �	��0 Yc�A�"*

lossF=�<��6�       �	in1 Yc�A�"*

loss��<�^_�       �	�2 Yc�A�"*

loss̖�:�`��       �	~�2 Yc�A�"*

loss�[<K�_�       �	9C3 Yc�A�"*

lossQ�O;e4�J       �	��3 Yc�A�"*

lossLG�;&G%�       �	;r4 Yc�A�"*

loss��<���       �	Z5 Yc�A�"*

lossQ,<!��       �	��5 Yc�A�"*

lossMy�<�AF       �	NB6 Yc�A�"*

lossص<��$/       �	��6 Yc�A�"*

loss�i�;��`       �	�q7 Yc�A�"*

lossۗ=ͮ�v       �	�8 Yc�A�"*

lossJp�<U*�o       �	?�8 Yc�A�"*

lossm�`=�6�e       �	�9 Yc�A�"*

lossL�#<��;       �	L: Yc�A�"*

loss-�=��.9       �	s�: Yc�A�"*

loss�?<z�       �	)]; Yc�A�"*

loss.��<.��:       �	�< Yc�A�"*

loss�=|kr�       �	��< Yc�A�"*

loss�?[<$C��       �	&V= Yc�A�"*

loss���<"��9       �	��= Yc�A�"*

loss��=��       �	�> Yc�A�"*

loss_sv<�'�       �	�4? Yc�A�"*

loss�4�=}+�;       �	��? Yc�A�"*

loss��<Έ+]       �	o@ Yc�A�"*

loss(�;���       �	�	A Yc�A�"*

loss��9�C�       �	�A Yc�A�"*

losshfA;�P       �	 FB Yc�A�"*

loss��H<O!-       �	"�B Yc�A�#*

loss
�<��~I       �	��C Yc�A�#*

lossHf<!a�       �	��D Yc�A�#*

loss�^�<� �       �	�cE Yc�A�#*

loss�+=� w       �	&�E Yc�A�#*

loss
�;ۃ9�       �	p�F Yc�A�#*

lossl�h;�i�       �	VdG Yc�A�#*

loss\B�<��y4       �	�H Yc�A�#*

lossy�;�2kE       �	ũH Yc�A�#*

loss!*`;�*��       �	��I Yc�A�#*

lossWJ;B��5       �	z6K Yc�A�#*

loss�K:1Z��       �	�!L Yc�A�#*

loss��=����       �	P�L Yc�A�#*

loss*�M:2��       �	>yM Yc�A�#*

lossTL;Ad�o       �	{�N Yc�A�#*

loss��2<�/X       �	�O Yc�A�#*

loss��=8�7�       �	d#P Yc�A�#*

loss���;�W�       �	�WQ Yc�A�#*

loss���;#i��       �	�Q Yc�A�#*

loss�(=&���       �	b�R Yc�A�#*

loss��	=9D��       �	S Yc�A�#*

loss??<H�2�       �	��S Yc�A�#*

loss��C<��+1       �	�CT Yc�A�#*

lossV6=p�]i       �	��T Yc�A�#*

loss3�=��c�       �	�nU Yc�A�#*

loss�<��       �	�V Yc�A�#*

loss�Ʋ=�->�       �	��V Yc�A�#*

loss �w;۞#s       �	�X Yc�A�#*

loss�ʅ<����       �	7�X Yc�A�#*

loss8*<GC�       �	>=Y Yc�A�#*

loss�f=lʻ       �	��Y Yc�A�#*

loss|�;�+�       �	�gZ Yc�A�#*

loss�0B=C%��       �	�[ Yc�A�#*

loss���='Ъ       �	��[ Yc�A�#*

lossQE/<�w       �	�D\ Yc�A�#*

loss��<��W�       �	��\ Yc�A�#*

lossO?�=r{�I       �	�] Yc�A�#*

loss���<��`�       �	/^ Yc�A�#*

loss�G<��       �	��^ Yc�A�#*

loss"8�=��@q       �	�\_ Yc�A�#*

loss_0�<� �       �	��_ Yc�A�#*

loss/y�;��       �	ۢ` Yc�A�#*

loss�=�#��       �	�;a Yc�A�#*

loss��;����       �	c�a Yc�A�#*

loss �$<��
       �	jkb Yc�A�#*

loss���<�g��       �	c Yc�A�#*

lossIV�<���r       �	0�c Yc�A�#*

loss�.�<�N;]       �	�7d Yc�A�#*

loss[5<~��       �	~�d Yc�A�#*

loss	^W<���)       �	~e Yc�A�#*

loss��>��c       �	
Hf Yc�A�#*

loss�<r��       �	6�f Yc�A�#*

loss�&]<j��       �	�g Yc�A�#*

loss�7�;n͉�       �	�3h Yc�A�#*

loss���<���       �	�h Yc�A�#*

loss�D�<��W�       �	w�i Yc�A�#*

loss�i�<���       �	GVj Yc�A�#*

loss��{<;�	       �	6!k Yc�A�#*

lossX{�;�]E�       �	��k Yc�A�#*

lossR�<<󩲣       �	�ol Yc�A�#*

lossҞ�;�Hl�       �	�m Yc�A�#*

lossT�;�K=�       �	4�m Yc�A�#*

loss���;�^��       �	�Vn Yc�A�#*

loss�s6<���       �	��n Yc�A�#*

loss	��;9�jr       �	v�o Yc�A�#*

loss3}�=yܱr       �	/p Yc�A�#*

loss �<*=��       �	=�p Yc�A�#*

lossxZ<�        �	�rq Yc�A�#*

loss!�2=�}d       �	?r Yc�A�#*

loss&�C=��       �	}�r Yc�A�#*

loss��<��ԏ       �	Ͽs Yc�A�#*

loss�Ċ;%Œ�       �	�u Yc�A�#*

lossȗ�;���       �	��u Yc�A�#*

lossp^=���        �	�jv Yc�A�#*

loss���<H�{       �	�w Yc�A�#*

lossi�<�m��       �	�w Yc�A�#*

loss6�<2���       �	{ix Yc�A�#*

loss*��;��I       �	@y Yc�A�#*

lossn�;�;c       �	�y Yc�A�#*

loss�Zy<x�m       �	zmz Yc�A�#*

loss�L;��       �	�{ Yc�A�#*

loss��K=�v�g       �	��{ Yc�A�#*

loss�;z��       �	�l| Yc�A�#*

loss�<       �	�} Yc�A�#*

lossK�=��8       �	B�} Yc�A�#*

loss�jr<D1!�       �	~ Yc�A�#*

loss���<�sF       �	}" Yc�A�#*

lossX%>�*��       �	�� Yc�A�#*

loss��<�_n       �	&r� Yc�A�#*

loss�3�<�e�       �	_� Yc�A�#*

loss�<���       �	]�� Yc�A�#*

lossF}�<��?       �	�A� Yc�A�#*

loss;5�<��U�       �	�܂ Yc�A�#*

loss�[�;ڭ��       �	W{� Yc�A�#*

loss���;tc*Z       �	u!� Yc�A�#*

loss��<~��       �	ʄ Yc�A�#*

losso�=9f(�       �	.u� Yc�A�#*

loss$��=��       �	M� Yc�A�#*

lossXu�;�1�&       �	�0� Yc�A�#*

loss�<�<���       �	�� Yc�A�#*

loss� <�;*       �	r� Yc�A�#*

lossϳ�;�5x       �	�׊ Yc�A�#*

loss�v�<�^       �	$�� Yc�A�#*

loss.�=���?       �	~�� Yc�A�#*

loss8�=�J��       �	5|� Yc�A�#*

loss$=��m       �	
�� Yc�A�#*

loss1�=�]�       �	�p� Yc�A�#*

loss��<(�	       �	ƨ� Yc�A�#*

loss[{<�'�       �	){� Yc�A�#*

loss�~�:+�q       �	Gs� Yc�A�#*

losse.R<�{s       �	|�� Yc�A�#*

loss���<
)Ш       �	�I� Yc�A�#*

loss���<]sjr       �	�� Yc�A�#*

loss�6<BM:(       �	�� Yc�A�#*

loss�<����       �	�M� Yc�A�#*

loss��1=�~�       �	�� Yc�A�#*

loss��$=�?       �	�� Yc�A�#*

loss�cK;**w�       �	� Yc�A�#*

lossw{�;Sn�       �	��� Yc�A�#*

loss��;�<h�       �	�>� Yc�A�#*

loss�*�;���       �	�љ Yc�A�#*

lossVԃ;5u       �	�g� Yc�A�#*

loss�v�<�H�       �	6� Yc�A�#*

loss��<e��       �	 �� Yc�A�#*

loss��1<s�       �	�+� Yc�A�#*

loss�=Pb�^       �	��� Yc�A�#*

lossqT�<�D�       �	֎� Yc�A�#*

loss��<0P��       �	)#� Yc�A�#*

loss�� =�9�       �	鹞 Yc�A�$*

loss��;� ��       �	jP� Yc�A�$*

loss��_<?�       �	q� Yc�A�$*

loss��<1�˪       �	>{� Yc�A�$*

lossV��=@�8�       �	#� Yc�A�$*

lossT�!<�$�F       �	��� Yc�A�$*

loss�@C=�=�       �	ZE� Yc�A�$*

lossٚ<�Uһ       �	i� Yc�A�$*

loss�p�<8�e�       �	�|� Yc�A�$*

lossۭN;���       �	&� Yc�A�$*

lossQ% =z+�       �	��� Yc�A�$*

loss�='�@       �	�\� Yc�A�$*

loss� 9=�&]       �	R� Yc�A�$*

loss�#�;�=��       �	�ۦ Yc�A�$*

loss�;�!��       �	��� Yc�A�$*

lossv��;b�       �	�� Yc�A�$*

lossqw=��&       �	%�� Yc�A�$*

lossZ�<"���       �	L� Yc�A�$*

loss��<�~�       �	u� Yc�A�$*

loss���<���       �	Ѐ� Yc�A�$*

loss�
�<���d       �	�� Yc�A�$*

lossi#�<cω       �	�ͫ Yc�A�$*

loss`�K<K	�       �	�j� Yc�A�$*

loss"B<&T�       �	?�� Yc�A�$*

loss_��<���       �	R�� Yc�A�$*

loss��<��"�       �	!<� Yc�A�$*

loss��<;��       �	�Ԯ Yc�A�$*

lossH�<;�        �	(� Yc�A�$*

loss\	�;|�v�       �	�'� Yc�A�$*

loss#�<3n�       �	�° Yc�A�$*

loss���<�+�       �	��� Yc�A�$*

loss���;�nUf       �	�@� Yc�A�$*

loss\Ŝ<B	�w       �	.<� Yc�A�$*

loss�\
=�|Dc       �	L�� Yc�A�$*

loss��=�T��       �	K�� Yc�A�$*

lossQc�<r�\b       �	�M� Yc�A�$*

loss@͌=t�9)       �	�� Yc�A�$*

lossf�!:�/.       �	L�� Yc�A�$*

loss� �<�i6g       �	�,� Yc�A�$*

loss��=�       �	�˷ Yc�A�$*

loss��;���       �	l� Yc�A�$*

lossQ=X�.�       �	�
� Yc�A�$*

loss���<�1�       �	v�� Yc�A�$*

loss��<��.       �	9E� Yc�A�$*

lossvb	<�`�       �	jߺ Yc�A�$*

loss�D�;nV[Z       �	��� Yc�A�$*

loss��;CEB�       �	0J� Yc�A�$*

loss.�!> �E�       �	�� Yc�A�$*

lossw��<NK       �	9~� Yc�A�$*

lossԁ;�5�       �	�� Yc�A�$*

loss+�<M6�       �	�о Yc�A�$*

lossd]U;�eS       �	�i� Yc�A�$*

loss?2<j���       �	�� Yc�A�$*

loss*��;�n��       �	��� Yc�A�$*

loss�p�;�%O#       �	n0� Yc�A�$*

loss3=u<,Fp�       �	��� Yc�A�$*

loss��@=np�       �	d� Yc�A�$*

lossղ<y���       �	�� Yc�A�$*

lossAPh<�L��       �	Ԟ� Yc�A�$*

loss�MB=@�'2       �	S�� Yc�A�$*

lossa�><�B�.       �	R� Yc�A�$*

lossŏy<�#�       �	��� Yc�A�$*

loss6�<F{��       �	��� Yc�A�$*

losso>�e�;       �	�X� Yc�A�$*

loss�Ls<2(��       �	`� Yc�A�$*

loss�<Y.�       �	�� Yc�A�$*

loss*�=9j�(       �	��� Yc�A�$*

loss��;���       �	*R� Yc�A�$*

loss3�<;Vt8       �	� � Yc�A�$*

loss�;�C�       �	"�� Yc�A�$*

loss.�r=�]Z�       �	xA� Yc�A�$*

lossd_E<�;!       �	��� Yc�A�$*

loss��>Ib�{       �	�~� Yc�A�$*

loss�S�=����       �	�� Yc�A�$*

loss�w=���       �	>�� Yc�A�$*

lossx�$;��-       �	�J� Yc�A�$*

loss�e�<j��@       �	��� Yc�A�$*

loss�;�=�I�       �	�v� Yc�A�$*

loss,�W;���f       �	=� Yc�A�$*

loss�(�;z�2       �	@�� Yc�A�$*

loss�$�<���8       �	�E� Yc�A�$*

loss҇P<�S�       �	0�� Yc�A�$*

loss1v<�h�G       �	Û� Yc�A�$*

loss�A<ڢ/j       �	�@� Yc�A�$*

loss��!=�k��       �	|�� Yc�A�$*

loss_[�;1+]�       �	?�� Yc�A�$*

loss�<6x[o       �	0�� Yc�A�$*

loss��;�T�       �	N� Yc�A�$*

loss���<ϗ2�       �	.�� Yc�A�$*

lossq� =˟�       �	F}� Yc�A�$*

lossa��;�˨�       �	�� Yc�A�$*

lossL�;0�	E       �	x�� Yc�A�$*

loss���<~®       �	�L� Yc�A�$*

loss&�=cG7J       �	��� Yc�A�$*

loss]�;>���       �	�� Yc�A�$*

loss?D7=��r       �	E-� Yc�A�$*

loss n�;ޠ�C       �	�� Yc�A�$*

loss�� =xT       �	xc� Yc�A�$*

loss��F<��j       �	� Yc�A�$*

lossط<�&��       �	-�� Yc�A�$*

loss
]�;*9E�       �	�L� Yc�A�$*

loss�D=�V;�       �	��� Yc�A�$*

loss�Qb=#k�       �	G�� Yc�A�$*

lossa�2:P���       �	#� Yc�A�$*

loss�$I<A�/T       �	��� Yc�A�$*

lossVGn<h?PX       �	�T� Yc�A�$*

loss�f"=/G�1       �	��� Yc�A�$*

lossA9+=�28       �	�� Yc�A�$*

loss��\<��
g       �	/� Yc�A�$*

loss�̼<T4�       �	G�� Yc�A�$*

lossp<2nN0       �	�[� Yc�A�$*

lossw�;1��P       �	��� Yc�A�$*

loss��;����       �	��� Yc�A�$*

loss�z:<x��       �	�b� Yc�A�$*

loss�'y<nh�       �	��� Yc�A�$*

loss��^<Hꁒ       �	z�� Yc�A�$*

lossvZX;d���       �	�&� Yc�A�$*

loss�=!I       �	�� Yc�A�$*

loss�N�<4A7~       �	'f� Yc�A�$*

loss�
�;]럧       �	�� Yc�A�$*

loss�}�<��\?       �	/�� Yc�A�$*

loss��=T�G       �	�?� Yc�A�$*

loss��;<�\KO       �	o�� Yc�A�$*

loss,�<[�       �	��� Yc�A�$*

loss��=���5       �	|e� Yc�A�$*

loss,�<?-Hr       �	K� Yc�A�$*

lossJ��7�V�       �	��� Yc�A�$*

loss���:�+m�       �	#I� Yc�A�$*

loss���<k��       �	��� Yc�A�%*

loss4(;q���       �	}�� Yc�A�%*

loss��;��#�       �	*� Yc�A�%*

loss��W:��       �	׿� Yc�A�%*

lossC�J;�I�       �	�W� Yc�A�%*

loss��8���       �	�;� Yc�A�%*

lossZ4�:	S�3       �	��� Yc�A�%*

loss�� :%��'       �	m� Yc�A�%*

loss�o!;!pԻ       �	�� Yc�A�%*

lossv�+<m���       �	)�� Yc�A�%*

loss3;8Z       �	wH� Yc�A�%*

loss��B:{mB�       �	��� Yc�A�%*

lossH�o<D�X�       �	��� Yc�A�%*

loss��/=�R       �	.� Yc�A�%*

lossZY�:脷Z       �	ݶ� Yc�A�%*

loss��O>��E       �	F`� Yc�A�%*

loss��+<�8�       �	c� Yc�A�%*

lossmG==tj��       �	��� Yc�A�%*

loss�4�;�q��       �	�W� Yc�A�%*

loss��;7<¹       �	��� Yc�A�%*

loss&}<K���       �	�� Yc�A�%*

loss}�?=pI       �	�*� Yc�A�%*

loss���;s�U       �	��� Yc�A�%*

loss��d=<;�       �	�a� Yc�A�%*

loss$��;F�H6       �	7�� Yc�A�%*

loss�>S<�,       �	ٕ Yc�A�%*

loss�ʉ=��       �	SYYc�A�%*

loss��<@���       �	_�Yc�A�%*

loss���;��;        �	�Yc�A�%*

loss-[�<�Y�j       �	�UYc�A�%*

loss�}'<G�[       �	��Yc�A�%*

lossߩ_<�6��       �	�{Yc�A�%*

loss*�<�@P�       �	EYc�A�%*

lossČ�<B
0�       �	��Yc�A�%*

lossґ�;3*��       �	x{Yc�A�%*

loss�a=<�_t       �	�CYc�A�%*

loss�&�;D�>       �	H�Yc�A�%*

loss^�:�q@�       �	��	Yc�A�%*

loss
�=C��       �	�&
Yc�A�%*

loss��<7�b       �	��
Yc�A�%*

lossʛ�=.`��       �	>[Yc�A�%*

loss8^"=3~�       �	@�Yc�A�%*

lossH��;:#ς       �	;�Yc�A�%*

loss��,=�|x       �	ADYc�A�%*

loss[r�;�Y�       �	��Yc�A�%*

lossS��;�\:�       �	
�Yc�A�%*

loss\�<�$á       �	uYc�A�%*

loss�v:B�q�       �	��Yc�A�%*

loss�"e<sX�U       �	�YYc�A�%*

loss���<�u��       �	>�Yc�A�%*

lossHO<���       �	��Yc�A�%*

loss��;ǡ)�       �	�:Yc�A�%*

losso�>=0�2�       �	g�Yc�A�%*

loss�W�<����       �	quYc�A�%*

lossZ�y<�M�       �	~Yc�A�%*

loss	�~<�+       �	,�Yc�A�%*

loss�C<���       �	SYc�A�%*

loss$*�<��֡       �	�Yc�A�%*

loss��!=��bu       �	��Yc�A�%*

loss�Y�<���       �	jPYc�A�%*

loss�
Z=�`�       �	��Yc�A�%*

lossڕ
<���g       �	ЗYc�A�%*

loss(��<e���       �	�4Yc�A�%*

loss�;c<n���       �	�Yc�A�%*

loss.�;���       �	NbYc�A�%*

loss��,<�ax       �	t�6Yc�A�%*

lossx��<�)K�       �	17Yc�A�%*

loss.�<&�Wh       �	��7Yc�A�%*

lossa�0=���y       �	)u8Yc�A�%*

lossZO�<�G�        �	_`9Yc�A�%*

loss�\�;�'&B       �	�:Yc�A�%*

loss� �;����       �	�:Yc�A�%*

loss2�;�y��       �	�@;Yc�A�%*

loss���<հ2�       �	��;Yc�A�%*

loss|�S=l��L       �	�j=Yc�A�%*

lossi��;����       �	�k>Yc�A�%*

loss@��;��b       �	 ?Yc�A�%*

loss��=�kI�       �	˼?Yc�A�%*

lossj�`=� g�       �	_@Yc�A�%*

loss��;;����       �	�XAYc�A�%*

loss��<9�'       �	" BYc�A�%*

loss�I�:�慝       �	 �BYc�A�%*

loss��e;����       �	�DCYc�A�%*

lossY&�;a:       �	��CYc�A�%*

losso=<�eU�       �	ՔDYc�A�%*

loss]�;�7h�       �	2:EYc�A�%*

loss�/�=�N��       �	��EYc�A�%*

lossZ�;�p�       �	�FYc�A�%*

loss���=��       �	~RGYc�A�%*

loss7E;|��\       �	HYc�A�%*

loss�1<�}�       �	��HYc�A�%*

loss��|;1YM�       �	UIYc�A�%*

loss@A�;�       �	�@JYc�A�%*

lossx�f<�SȆ       �	cKYc�A�%*

loss}�Z<o���       �	��KYc�A�%*

loss���;�\�       �	�LYc�A�%*

loss�
�;�§�       �	�}MYc�A�%*

lossR{o<�Kn(       �	V+NYc�A�%*

loss[}4=��`       �	��NYc�A�%*

loss��<��_       �	�}OYc�A�%*

loss��;���f       �	��PYc�A�%*

loss��/;�       �	��QYc�A�%*

loss�)=��=�       �	x�RYc�A�%*

loss\�>����       �	��SYc�A�%*

loss���<Qu#�       �	��TYc�A�%*

loss&�6;# �B       �	s�UYc�A�%*

loss�cx; uԖ       �	��VYc�A�%*

loss(u�:|��       �	�%WYc�A�%*

loss�s<-�       �	a�WYc�A�%*

loss$�S<lt�       �	gaXYc�A�%*

loss��:v��       �	�"YYc�A�%*

loss
Zo<ƈ�.       �	A�YYc�A�%*

loss t�;�"Ʋ       �	�ZYc�A�%*

loss��G=9J�       �	|�[Yc�A�%*

lossw�:�%       �	k}\Yc�A�%*

loss�;��+�       �	�]Yc�A�%*

lossC��<�̿m       �	S[^Yc�A�%*

loss!�;|:       �	q�_Yc�A�%*

lossc��=l�ƍ       �	~aYc�A�%*

loss/�;��P�       �	ΦbYc�A�%*

loss�R�:�B9,       �	�UcYc�A�%*

lossOT�;"��        �	8�cYc�A�%*

loss�[w<�x��       �	�dYc�A�%*

loss��<�)#.       �	LeYc�A�%*

losso��<ܲC�       �	��eYc�A�%*

loss��=$u1�       �	�fYc�A�%*

loss[��<���       �	�NgYc�A�%*

loss���;G��       �	�1hYc�A�%*

loss�RE;3H�       �	��hYc�A�&*

lossu;�+"�       �	�iYc�A�&*

loss8G=dPXY       �	�6jYc�A�&*

loss�2];j��o       �	�jYc�A�&*

lossR�1=Lѯ       �	_�kYc�A�&*

loss���=o��       �	�GlYc�A�&*

loss��z<�dA       �	��lYc�A�&*

loss��<��>       �	�mYc�A�&*

lossR�z<"��       �	0bnYc�A�&*

lossq��<�߮�       �	�oYc�A�&*

loss>�<FmL       �	�oYc�A�&*

lossΜ<���       �	�ypYc�A�&*

loss1��;�jg~       �	�7qYc�A�&*

loss<�=>%�       �	��qYc�A�&*

lossjR ;��K       �	h�rYc�A�&*

loss��6<�Y�       �	�EsYc�A�&*

loss��=l@,1       �	��sYc�A�&*

loss*;< j��       �	�tYc�A�&*

loss7}=���       �	�DuYc�A�&*

loss�<��e8       �	1�uYc�A�&*

loss��<Ӆvs       �	�vYc�A�&*

loss\Y=���       �	�IwYc�A�&*

loss~J<�1�       �	xYc�A�&*

lossf)<!�E       �	մxYc�A�&*

loss26S<���       �	�[yYc�A�&*

loss��<i���       �	KzYc�A�&*

lossݛ�<�a�       �	ΥzYc�A�&*

loss���;���=       �	(E{Yc�A�&*

loss���<0�j       �	�{Yc�A�&*

lossH�*<�`�}       �	w�|Yc�A�&*

loss/�=��[       �	�#}Yc�A�&*

loss�j<�U�       �	ؼ}Yc�A�&*

loss�;R,�       �	�f~Yc�A�&*

lossx=b�,i       �	\Yc�A�&*

loss���<��o�       �	%�Yc�A�&*

loss���<!X�       �	�_�Yc�A�&*

loss��B;n�       �	}�Yc�A�&*

loss��<Fr�       �	���Yc�A�&*

lossȊ;R��       �	%��Yc�A�&*

loss�_�<uj#_       �	TW�Yc�A�&*

loss$�O=��w�       �	z��Yc�A�&*

loss!� =S�       �	전Yc�A�&*

lossQ�I<.(�       �	�q�Yc�A�&*

loss�:Y=���0       �	s�Yc�A�&*

loss��;X��[       �	���Yc�A�&*

loss3�<7-�       �	���Yc�A�&*

lossטu;8$e�       �	3�Yc�A�&*

loss��f=E�       �	���Yc�A�&*

loss��<��7�       �	[��Yc�A�&*

lossE=�I       �	�/�Yc�A�&*

lossS:B��P       �	&�Yc�A�&*

loss��<�ŕ       �	z�Yc�A�&*

loss�<b�a       �	��Yc�A�&*

loss��R< ݐ�       �	m��Yc�A�&*

lossA��<�       �	=C�Yc�A�&*

loss�+]<�'3�       �	<ڍYc�A�&*

loss�/;<����       �	ms�Yc�A�&*

loss�)]<#}��       �	��Yc�A�&*

loss��;Y��       �	���Yc�A�&*

lossvY:]���       �	�D�Yc�A�&*

losse�=�Y�       �	-�Yc�A�&*

loss��=;�4��       �	Á�Yc�A�&*

lossRfJ=�>�d       �	j0�Yc�A�&*

loss�_z=xA��       �	@ޒYc�A�&*

lossT��;cJgw       �	�Yc�A�&*

lossw�;Ʈ.}       �	-$�Yc�A�&*

loss���=m'K        �	qǔYc�A�&*

loss���;P�+       �	<h�Yc�A�&*

lossl�;<Ҧ       �	��Yc�A�&*

loss�<#_��       �	
��Yc�A�&*

loss,�;���       �	�@�Yc�A�&*

loss H=1*s�       �	�ٗYc�A�&*

lossq0P<t'�       �	���Yc�A�&*

lossp;��       �	7�Yc�A�&*

loss�L;j[�       �	�֙Yc�A�&*

loss���;
��D       �	�s�Yc�A�&*

loss8�;��F       �	��Yc�A�&*

lossV��;.���       �	���Yc�A�&*

loss�K�;`���       �	�D�Yc�A�&*

lossd8<�ݫ       �	��Yc�A�&*

loss�m';p��       �	ݙ�Yc�A�&*

loss(=:>��       �	�5�Yc�A�&*

loss xY<�P��       �	�ޞYc�A�&*

loss�o�:�K"       �	Yc�A�&*

lossZ$<j�t�       �	�>�Yc�A�&*

losso�<<v)y       �	��Yc�A�&*

loss#m�;�E��       �	���Yc�A�&*

loss���;���       �	N�Yc�A�&*

loss���<  ��       �	���Yc�A�&*

loss�\=���       �	Ǟ�Yc�A�&*

lossh:!<�s�       �	fL�Yc�A�&*

loss�� <��EG       �	s��Yc�A�&*

loss�q�:��x:       �	*��Yc�A�&*

lossc��:)��       �	6W�Yc�A�&*

loss�G�8���       �	W	�Yc�A�&*

lossY~�=i��x       �	���Yc�A�&*

loss��;���1       �	m�Yc�A�&*

loss�e�<��/       �	y!�Yc�A�&*

lossfM�<H��       �	��Yc�A�&*

loss���<qđ       �	A��Yc�A�&*

lossi�W:�b�       �	m�Yc�A�&*

loss�I=��u       �	��Yc�A�&*

loss,�:B��`       �	��Yc�A�&*

loss�c2;ox^       �	��Yc�A�&*

loss��<���        �	��Yc�A�&*

loss�T�:ku       �	fg�Yc�A�&*

loss��<a4�h       �	k�Yc�A�&*

loss_\�<�mt�       �	|�Yc�A�&*

loss��=�m�=       �	뫱Yc�A�&*

loss��9"V       �	���Yc�A�&*

lossrq=m�       �	�W�Yc�A�&*

lossZ̊:	4�       �	��Yc�A�&*

loss]c�<}6��       �	�ǴYc�A�&*

loss�t>=X�m       �	y�Yc�A�&*

loss���<$/       �	�,�Yc�A�&*

lossC/�<I&~S       �	ݶYc�A�&*

loss#n�;��       �	ס�Yc�A�&*

loss��;�1x       �	�B�Yc�A�&*

losss�=�q       �	�Yc�A�&*

loss��;�/       �	���Yc�A�&*

loss�(�;��       �	�G�Yc�A�&*

lossh��;Ȁ��       �	K�Yc�A�&*

loss{9N;��A!       �	Y޻Yc�A�&*

losslI�<����       �	擼Yc�A�&*

lossӶ
=��wC       �	>>�Yc�A�&*

lossz�=Y�m!       �	}�Yc�A�&*

loss��e=��l       �	�Yc�A�&*

loss!�=r ԩ       �	z7�Yc�A�&*

lossHn�<~���       �	Q޿Yc�A�'*

loss��=�ri�       �	�~�Yc�A�'*

loss� �<��
+       �	B'�Yc�A�'*

loss�';Mg6R       �	K��Yc�A�'*

lossC��;i;u       �	�n�Yc�A�'*

losse�%<�K��       �	V�Yc�A�'*

loss]�U<1��       �	���Yc�A�'*

lossj.x<.b�       �	+P�Yc�A�'*

loss�;G`�       �	���Yc�A�'*

lossJ��:�       �	B	�Yc�A�'*

loss�3�;����       �	}��Yc�A�'*

loss�Ht;�?��       �	�Yc�A�'*

loss�ya;���       �	\��Yc�A�'*

lossO��:�e       �	�S�Yc�A�'*

loss|d�;ڦ��       �	��Yc�A�'*

loss��*=�;`�       �	�L�Yc�A�'*

loss�a�=]�       �	hu�Yc�A�'*

loss=�=��S�       �	��Yc�A�'*

loss3�=,5]       �	�`�Yc�A�'*

loss.�0<�mH�       �	b�Yc�A�'*

loss�s<!�I�       �	Ů�Yc�A�'*

loss�̉=�3g       �	X��Yc�A�'*

loss�g�<5:       �	Jy�Yc�A�'*

loss��>V�~U       �	H�Yc�A�'*

loss�ԑ<t��       �	H��Yc�A�'*

loss�$&>��\       �	���Yc�A�'*

loss��=�D��       �	�#�Yc�A�'*

lossh�<�bz�       �	�?�Yc�A�'*

loss=�=s�A�       �	���Yc�A�'*

lossv3�</.�       �	̚�Yc�A�'*

lossz}�; S%�       �	F@�Yc�A�'*

loss��<	QI       �	���Yc�A�'*

loss��<(�d�       �	ȳ�Yc�A�'*

lossw��:���O       �	SX�Yc�A�'*

loss��<��D       �	���Yc�A�'*

lossv=P<�_��       �	���Yc�A�'*

loss�Ɠ:��<       �	X�Yc�A�'*

loss�}d<w�7	       �	 �Yc�A�'*

losso�<��!       �	���Yc�A�'*

loss�5�;�I%)       �	(H�Yc�A�'*

loss�0F<���       �	���Yc�A�'*

loss�3?<2NQv       �	D��Yc�A�'*

loss�=<�?       �	R,�Yc�A�'*

loss߅<��#       �	��Yc�A�'*

loss�<���       �	ur�Yc�A�'*

loss���<p?�       �	��Yc�A�'*

loss鮦<�{��       �	���Yc�A�'*

lossF�=�u�/       �	�^�Yc�A�'*

loss�s�;�=qX       �	_�Yc�A�'*

loss��(<p�^       �	���Yc�A�'*

loss�
�;�O��       �	ō�Yc�A�'*

loss 0<P�6�       �	4,�Yc�A�'*

loss�3�;�fb�       �	���Yc�A�'*

loss��<��|       �	=��Yc�A�'*

loss١=p<�d       �	�"�Yc�A�'*

lossW�?<e5�Y       �	���Yc�A�'*

lossNԍ;Eߝl       �	�i�Yc�A�'*

loss�]�;��OA       �	��Yc�A�'*

losshd�;#R       �	/��Yc�A�'*

loss�k�<Ug��       �	!<�Yc�A�'*

loss��=ڲeB       �	���Yc�A�'*

loss�)�<!%�       �	�`�Yc�A�'*

loss��;}�       �	���Yc�A�'*

lossj�;���       �	��Yc�A�'*

loss�v�<�ψh       �	�y�Yc�A�'*

loss��r;���       �	�.�Yc�A�'*

lossӔ�;�|3�       �	���Yc�A�'*

loss��=�2�       �	^�Yc�A�'*

loss�)<qP*�       �	W�Yc�A�'*

loss=��:��       �	�+�Yc�A�'*

loss!L�;�1*/       �	���Yc�A�'*

lossװ<õ��       �	�^�Yc�A�'*

loss�j"<�-��       �	�!�Yc�A�'*

loss19�<}c�`       �	���Yc�A�'*

loss7�u<���       �	KX�Yc�A�'*

loss�F<K�N�       �	���Yc�A�'*

lossM\[<�6j       �	p��Yc�A�'*

loss�*e<���d       �	s.�Yc�A�'*

lossL��;���       �	���Yc�A�'*

loss <�:��	       �	�l�Yc�A�'*

losss�<�$��       �	��Yc�A�'*

loss
ϫ<��z�       �	���Yc�A�'*

loss�U�<ʯ�K       �	�I�Yc�A�'*

lossH��=e|       �	���Yc�A�'*

loss]�=v7�       �	�~�Yc�A�'*

loss�F= \�       �	��Yc�A�'*

loss���:��!�       �	:��Yc�A�'*

loss���<�O@       �	�T�Yc�A�'*

loss6<�^v�       �	���Yc�A�'*

loss��<G�{�       �	+��Yc�A�'*

loss6��<���%       �	�X�Yc�A�'*

loss�K;`o1       �	� Yc�A�'*

loss6��;E�s�       �	Н Yc�A�'*

loss-W
<�i��       �	�:Yc�A�'*

lossjH�<��s�       �	��Yc�A�'*

loss@��<�2��       �	�mYc�A�'*

lossġ�;q�e       �	#Yc�A�'*

loss(Վ<m�v�       �	ڨYc�A�'*

loss�W�;z먥       �	WCYc�A�'*

lossZ;'}�       �	�Yc�A�'*

loss�z�<�u��       �	�oYc�A�'*

loss��9��K       �	JYc�A�'*

loss��;��6�       �	��Yc�A�'*

loss��<��'       �	h?Yc�A�'*

loss�u<�6l       �	Yc�A�'*

loss�,<�f       �	Y�Yc�A�'*

loss��1=~�۪       �	<�	Yc�A�'*

loss�	�=�*��       �	 b
Yc�A�'*

loss%x�<L���       �	JYc�A�'*

lossS�G<�f��       �	��Yc�A�'*

loss��<�Ɉ�       �	6?Yc�A�'*

loss�A�;���       �	4�Yc�A�'*

loss��=*[=9       �	�pYc�A�'*

loss���<�@}�       �	�Yc�A�'*

loss�<�={gc       �	L�Yc�A�'*

lossX�<��       �	h@Yc�A�'*

loss��1=lg.�       �	��Yc�A�'*

loss�4<#�.�       �	PqYc�A�'*

loss�	<�{��       �	e9Yc�A�'*

loss3�;���,       �	��Yc�A�'*

lossҙ�;���       �	!vYc�A�'*

loss�^=��A       �	ZYc�A�'*

lossM"<;�a�       �	3�Yc�A�'*

loss�B5;O�F�       �	6?Yc�A�'*

loss��<2��       �	��Yc�A�'*

loss�<-�xs       �	qYc�A�'*

loss%w>;'�R#       �	�Yc�A�'*

loss��<H�]G       �	��Yc�A�'*

loss�)�;s+2�       �	FYc�A�(*

lossn#�;��       �	��Yc�A�(*

loss��=Uf�]       �	?�Yc�A�(*

loss�F�<�6       �	�^Yc�A�(*

loss�Z�<��3       �	4�Yc�A�(*

lossF��<,�Bl       �	�Yc�A�(*

loss��<G�ք       �	*WYc�A�(*

loss#5=�qM�       �	Yc�A�(*

lossJ|s<���	       �	[�Yc�A�(*

lossm/f<�y�       �	DQYc�A�(*

lossH�%=�_D�       �	��Yc�A�(*

loss�y�<� ��       �	�Yc�A�(*

loss�39;`�s~       �	�(Yc�A�(*

loss��<�Ќ       �	�Yc�A�(*

lossoD<<3�v       �	dZ Yc�A�(*

loss� +=�a��       �	�� Yc�A�(*

loss=�;�u��       �	��!Yc�A�(*

loss�I�;�0       �	'"Yc�A�(*

loss�m�;�f�C       �	�"Yc�A�(*

loss�=0��       �	׊#Yc�A�(*

loss82;�7'�       �	JD$Yc�A�(*

loss?	�<ڴ#�       �	�%Yc�A�(*

lossyd�<�bEC       �	�%Yc�A�(*

loss1B=��>�       �	tA&Yc�A�(*

loss�5:*ӝ�       �	�'Yc�A�(*

loss�� >|�       �	�(Yc�A�(*

loss$�<ҷ�       �	�(Yc�A�(*

loss-�\;M_1�       �	ZE)Yc�A�(*

loss'*=)��       �	Y�)Yc�A�(*

loss-�(<���       �	�y*Yc�A�(*

loss�Δ;����       �	M+Yc�A�(*

loss�:kd{�       �	��+Yc�A�(*

loss�CI<i��       �	?,Yc�A�(*

loss�=��       �	�,Yc�A�(*

loss�5z<�Z       �	s-Yc�A�(*

loss_i�<��'m       �	�.Yc�A�(*

loss��J<�~mX       �	%�.Yc�A�(*

loss]^<}�X�       �	�P/Yc�A�(*

loss8�-<��yq       �	��/Yc�A�(*

lossiQ�;�*	�       �	��0Yc�A�(*

loss�<;�P       �	�I1Yc�A�(*

lossoo?<+��       �	�1Yc�A�(*

loss�=��M�       �	�2Yc�A�(*

loss���<ʴi�       �	Z3Yc�A�(*

loss�[=�_�e       �	�3Yc�A�(*

lossI�-<��|b       �	��4Yc�A�(*

loss_�=�>       �	y$5Yc�A�(*

loss��>=�Ѯ*       �	��5Yc�A�(*

loss�X�<�N@       �	�U6Yc�A�(*

loss�eO=��       �	Y�6Yc�A�(*

loss@��<L`��       �	�7Yc�A�(*

lossx�`<F֬�       �	4�8Yc�A�(*

loss�%=;h�4#       �	�\9Yc�A�(*

loss�F�<����       �	I�9Yc�A�(*

loss��<~bK       �	S�:Yc�A�(*

loss�w<ɀ��       �	!;;Yc�A�(*

lossJ�O<�K.       �	�;Yc�A�(*

loss]E<���       �	>{<Yc�A�(*

lossA�<��_�       �	 =Yc�A�(*

loss�V<4�{A       �	=�=Yc�A�(*

loss�T =3`�:       �	_b>Yc�A�(*

loss���;�^�       �	��>Yc�A�(*

loss[�<��4�       �	��?Yc�A�(*

loss=j=.��R       �	�4@Yc�A�(*

loss�3<Yj        �	'�@Yc�A�(*

loss��: i�       �	BxAYc�A�(*

loss��>=ۓ��       �	HBYc�A�(*

loss$��<r���       �	E�BYc�A�(*

lossۂ�;�[       �	�VCYc�A�(*

losscq�<���       �	8�CYc�A�(*

lossI(O;��-n       �	r�DYc�A�(*

loss&�^<�eg.       �	�!EYc�A�(*

loss�F�=N�z�       �	�EYc�A�(*

loss�nA<Ɣ0�       �	�NFYc�A�(*

loss� �=�]�z       �	GYc�A�(*

loss��<)�>�       �	��GYc�A�(*

loss��<�;w       �	^HYc�A�(*

lossR:�T�j       �	,�HYc�A�(*

loss��_;�ԝW       �	��IYc�A�(*

loss��<qE��       �	JYc�A�(*

loss�;�<�WF       �	��JYc�A�(*

loss�d�:q6       �	�jKYc�A�(*

loss��<�6�       �	WLYc�A�(*

loss�*
>���a       �	��LYc�A�(*

loss�|�;���       �	gDMYc�A�(*

loss�1=��>e       �	(�MYc�A�(*

loss��;xc �       �	}xNYc�A�(*

losso�Y=	�v�       �	OYc�A�(*

lossL/<��       �	|�OYc�A�(*

loss�;���       �	OPYc�A�(*

loss�5=��       �	�PYc�A�(*

loss}�K<JOa�       �	�QYc�A�(*

lossL:�<��%s       �	�(RYc�A�(*

losso*�<�!��       �	��RYc�A�(*

loss#��<���       �	<hSYc�A�(*

loss6R�<gJ0       �	VTYc�A�(*

loss��<����       �	��TYc�A�(*

loss��<]{�       �	EKUYc�A�(*

loss�]%<+[�       �	��UYc�A�(*

lossE�<`/,       �	H�VYc�A�(*

loss2�(=�7M       �	�!WYc�A�(*

loss&��;d�/       �	��WYc�A�(*

loss\�<�B       �	"OXYc�A�(*

lossm.;6F��       �	|�XYc�A�(*

loss��=��8       �	�YYc�A�(*

loss4�"=GR�       �	�(ZYc�A�(*

loss�Q�=Z��       �	˿ZYc�A�(*

lossn�7<����       �	�a[Yc�A�(*

loss�6�=�]V�       �	�\Yc�A�(*

loss���<?���       �	2�\Yc�A�(*

loss�¬;�>�H       �	�J]Yc�A�(*

loss� e<ŝ)�       �	��]Yc�A�(*

loss-�<���       �	f�^Yc�A�(*

loss�H�<\�#X       �	G_Yc�A�(*

loss�]U<�J��       �	�_Yc�A�(*

loss���;ܷ�       �	�X`Yc�A�(*

loss�|�<4}"       �	�`Yc�A�(*

lossҡ,;y[]       �	!�aYc�A�(*

loss��;�Y       �	�ebYc�A�(*

loss�p�;���       �	 cYc�A�(*

loss#�l;Y�z       �	úcYc�A�(*

loss�fR=�Hn!       �	�mdYc�A�(*

loss���<|\�#       �	�eYc�A�(*

loss��<�e�U       �	��eYc�A�(*

lossA/X<k�p       �	��fYc�A�(*

loss#� ;թ/D       �	�MgYc�A�(*

loss���<<�X�       �	��gYc�A�(*

loss��v<h�*4       �	s�hYc�A�(*

lossWw=��"       �	�"iYc�A�)*

loss�#"<�=�       �	H�iYc�A�)*

lossj�<��c       �	FkYc�A�)*

lossC�=���       �	B�kYc�A�)*

loss��_;��       �	�lYc�A�)*

lossإ<�v+k       �	�#mYc�A�)*

loss���<HX��       �	N�mYc�A�)*

lossC=z'�*       �	F�nYc�A�)*

loss}�=,�Lo       �	�:oYc�A�)*

loss*c�;�J7q       �	�oYc�A�)*

lossc�.<HEM       �	n�pYc�A�)*

lossQ�;�,A�       �	&qYc�A�)*

loss8<�;0���       �	��qYc�A�)*

loss��H;&Ե�       �	�qrYc�A�)*

loss_b;䈌N       �	#sYc�A�)*

loss��i<#ꐁ       �	ۿsYc�A�)*

loss]r�=}}Y       �	{htYc�A�)*

loss�� <����       �	w�uYc�A�)*

loss�3<��-�       �	JbvYc�A�)*

lossv)�<z'u�       �	�'wYc�A�)*

lossJ�;ZAF       �	O�wYc�A�)*

loss�f<r��!       �	�exYc�A�)*

lossz�;���>       �	?yYc�A�)*

lossʆ#=)�@�       �	��yYc�A�)*

loss��v<+�!K       �	�_zYc�A�)*

loss���<޺4T       �	P�zYc�A�)*

losscx~=F���       �	��{Yc�A�)*

lossm�;!��q       �	�<|Yc�A�)*

lossqJ�;bܡ       �	K�|Yc�A�)*

loss�Y<�=��       �	��}Yc�A�)*

loss&(=�ɡ�       �	�2~Yc�A�)*

loss�S_;6X@�       �	��~Yc�A�)*

losst��=�h�       �	�jYc�A�)*

loss�ߢ=,RJf       �	��Yc�A�)*

lossq-<�       �	ɪ�Yc�A�)*

lossz�;��~       �	�B�Yc�A�)*

lossTG<�F��       �	��Yc�A�)*

loss��<!^�       �	S��Yc�A�)*

loss��;=���       �	�V�Yc�A�)*

loss�;���9       �	���Yc�A�)*

loss��6=����       �	���Yc�A�)*

loss��\;�gJ�       �	K>�Yc�A�)*

lossG=�w�=       �	�օYc�A�)*

loss��<1��s       �	zm�Yc�A�)*

loss��=�&�&       �	��Yc�A�)*

loss�Ɂ;i;�T       �	���Yc�A�)*

loss4�=�>�       �	^d�Yc�A�)*

loss�l;����       �	w�Yc�A�)*

loss�[1<����       �	��Yc�A�)*

lossؓn<R*G&       �	s��Yc�A�)*

loss�p<�3       �	窋Yc�A�)*

loss�lA<�6��       �	�\�Yc�A�)*

loss�a<zX��       �	�Yc�A�)*

loss�x�<�p/�       �	ŭ�Yc�A�)*

loss��;�?�       �	ZH�Yc�A�)*

loss�~=�)h�       �	ގYc�A�)*

lossEnt;A�f�       �	���Yc�A�)*

lossc��<���       �	�L�Yc�A�)*

loss��<X��       �	��Yc�A�)*

loss(��<F0Y       �	��Yc�A�)*

loss� <�D�,       �	�*�Yc�A�)*

losst* ;G}�       �	���Yc�A�)*

loss��X<x�YG       �	��Yc�A�)*

loss��E:�"2v       �	�K�Yc�A�)*

lossV�=Ag*�       �	j��Yc�A�)*

loss���<����       �	��Yc�A�)*

loss�~=ś
       �	�V�Yc�A�)*

loss�H<��-�       �	�Yc�A�)*

lossF�<��<�       �	�ɗYc�A�)*

loss���<y��       �	Pm�Yc�A�)*

loss�'<ӛ       �	�Yc�A�)*

loss� �<-,��       �	[��Yc�A�)*

loss;.p;�JRD       �	�f�Yc�A�)*

loss�<�#�       �	��Yc�A�)*

loss�J;.Y@�       �	6��Yc�A�)*

loss��';t6�       �	�Q�Yc�A�)*

loss��N<
F�v       �	���Yc�A�)*

lossNm�<N�       �	T��Yc�A�)*

lossM�;�#0�       �	C�Yc�A�)*

loss{j*<[��       �	7ޞYc�A�)*

loss,�.=73��       �	�x�Yc�A�)*

loss�Hn=z��       �	��Yc�A�)*

loss��<M�14       �	 ��Yc�A�)*

loss�{�<�"�       �	L�Yc�A�)*

loss���=��L       �	h�Yc�A�)*

loss�<y�`�       �	���Yc�A�)*

losse'M9wu�G       �	c+�Yc�A�)*

loss�k<�p       �	ģYc�A�)*

loss���:X)�F       �	hZ�Yc�A�)*

loss,H�;O".       �	��Yc�A�)*

loss�,�9T	�       �	ލ�Yc�A�)*

loss;:����       �	�%�Yc�A�)*

loss�;;g`�       �	���Yc�A�)*

lossi<S;�ϳ�       �	�i�Yc�A�)*

loss<��9�s�3       �	��Yc�A�)*

lossc��;M��       �	D��Yc�A�)*

loss�U�:�u       �	MK�Yc�A�)*

loss���<0��       �	1�Yc�A�)*

loss1��;�ڮ       �	n�Yc�A�)*

loss���9[�ӑ       �	�x�Yc�A�)*

loss��L:�gFX       �	*T�Yc�A�)*

loss��e<E Q�       �	���Yc�A�)*

loss�`�:u��       �	=��Yc�A�)*

lossS��=���Z       �	 D�Yc�A�)*

loss�9�;t7�$       �	߯Yc�A�)*

loss��6<0�=�       �	�u�Yc�A�)*

loss]�)<9\0�       �	d�Yc�A�)*

lossM�;=�e�       �	j�Yc�A�)*

loss�<����       �	d��Yc�A�)*

loss]��;����       �	oH�Yc�A�)*

loss�/V;�Z�       �	�߳Yc�A�)*

losseky<����       �	Jz�Yc�A�)*

loss)65:2Ќ�       �	'�Yc�A�)*

loss�%�==ɏ       �	��Yc�A�)*

loss\Q!=�,��       �	�Z�Yc�A�)*

loss��<AS@m       �	��Yc�A�)*

loss���<"5º       �	)�Yc�A�)*

loss�c =�F��       �	z��Yc�A�)*

loss 8h;Լ��       �	I.�Yc�A�)*

loss�+w<��       �	1кYc�A�)*

loss�=beL       �		��Yc�A�)*

loss;0��6       �	�$�Yc�A�)*

loss4�^;���       �	kռYc�A�)*

loss��<Y�Y       �	?n�Yc�A�)*

lossMk=1r�}       �	��Yc�A�)*

lossR�1;Uk��       �	ͬ�Yc�A�)*

loss�><u�"       �	w��Yc�A�)*

loss��s<��
�       �	9'�Yc�A�)*

loss�0�=����       �	���Yc�A�**

loss���:\�VC       �	_�Yc�A�**

loss��=D��       �	���Yc�A�**

loss7�<�X]       �	���Yc�A�**

loss=�ش�       �	�;�Yc�A�**

loss�f!<Wn\�       �	��Yc�A�**

loss�uj;@��       �	S��Yc�A�**

loss��;d�	L       �	T�Yc�A�**

loss#�;�X�       �	f��Yc�A�**

loss�Į;�j�p       �	Փ�Yc�A�**

losse�=} mr       �	F(�Yc�A�**

loss4�<[僒       �	j��Yc�A�**

loss8��;�5:       �	Ҏ�Yc�A�**

loss.�!<_d       �	#,�Yc�A�**

loss\�=�Y�       �	z��Yc�A�**

loss���:^~       �	}Z�Yc�A�**

loss�|;= ¶       �	Y��Yc�A�**

loss{)�<ZMo       �	X��Yc�A�**

loss_D�;#r��       �	o�Yc�A�**

loss�c�<���F       �	��Yc�A�**

lossw�<Ȳ�"       �	��Yc�A�**

loss�|%;��9f       �	^,�Yc�A�**

lossD�;Q�X       �	���Yc�A�**

lossxw�:L��       �	V�Yc�A�**

loss8�9v���       �	���Yc�A�**

lossc
;=�R       �	�^�Yc�A�**

losss��<���       �	��Yc�A�**

loss��I=���       �	o��Yc�A�**

lossI�w=���       �	�L�Yc�A�**

loss��<���       �	I��Yc�A�**

loss��4< ��F       �	ޒ�Yc�A�**

loss��@<����       �	c`�Yc�A�**

loss?W�<����       �	Z��Yc�A�**

loss��<r�\�       �	���Yc�A�**

loss�~)=�8�T       �	&�Yc�A�**

loss�X;�ť       �	���Yc�A�**

loss��W<^q�8       �	��Yc�A�**

lossoxk<��	       �	�*�Yc�A�**

loss�oc;�FO�       �	���Yc�A�**

loss?�<�c�1       �	1^�Yc�A�**

loss�,�;�BƸ       �	���Yc�A�**

loss��9��       �	��Yc�A�**

loss��z<���|       �	��Yc�A�**

loss��E<�ܢ�       �	E��Yc�A�**

loss4�z<��L       �	�O�Yc�A�**

loss�'�;vUp�       �	G>�Yc�A�**

loss� �=�k��       �	��Yc�A�**

lossm);����       �	rj�Yc�A�**

loss�Ɠ=��       �	��Yc�A�**

loss��<�j<       �	��Yc�A�**

loss��H<�.��       �	�.�Yc�A�**

loss���<       �	���Yc�A�**

loss�31;�u�[       �	�\�Yc�A�**

loss��=�X��       �	��Yc�A�**

loss�4�:�Eh�       �	���Yc�A�**

loss�<]�rt       �	�?�Yc�A�**

loss��=�J       �	���Yc�A�**

loss���;k�72       �	o�Yc�A�**

lossK��<
       �	�3�Yc�A�**

loss�%;Y�*s       �	���Yc�A�**

lossf�y=]jU       �	o��Yc�A�**

loss���;���       �	�"�Yc�A�**

loss�zC<��=        �	M��Yc�A�**

loss`^#=�ά�       �	�x�Yc�A�**

lossiO�<bF{�       �	/ Yc�A�**

loss��;
$�       �	�� Yc�A�**

loss/D=�	Ob       �	^Yc�A�**

loss[�(<�K5U       �	��Yc�A�**

lossh �;6��       �	��Yc�A�**

loss=�<{b       �	�KYc�A�**

loss��x<��F�       �	��Yc�A�**

lossq��<��4�       �	r�Yc�A�**

loss��E<aY�L       �	�%Yc�A�**

loss���:�ëo       �	]�Yc�A�**

loss�v�:�V��       �	�UYc�A�**

loss�;�! �       �	�8Yc�A�**

lossC�#=Q��#       �	��Yc�A�**

loss��=;l¼8       �	�cYc�A�**

loss{vu=.�zm       �	�+	Yc�A�**

loss��;����       �	Z�	Yc�A�**

lossV��<�ݔ�       �	��
Yc�A�**

loss���:�4Z       �	rSYc�A�**

loss2�,<���`       �	wYc�A�**

loss�F<�:�       �	NBYc�A�**

loss��< � R       �	^�Yc�A�**

lossw�C>�QK>       �	��Yc�A�**

losssU;�1,       �	6Yc�A�**

loss�l�:ՏU       �	R�Yc�A�**

loss~�<O��5       �	�uYc�A�**

lossD�5<�Bbu       �	�IYc�A�**

lossQf<p.�*       �	��Yc�A�**

loss�3�;�y��       �	��Yc�A�**

loss��p;�o��       �	EbYc�A�**

lossM_�<D{�^       �	�
Yc�A�**

loss,�<�"l       �	�Yc�A�**

loss�=�S       �	[�Yc�A�**

loss�ǡ<�S�       �	�?Yc�A�**

loss��<�9�       �	N�Yc�A�**

lossmx(<s��d       �	�aYc�A�**

loss8��<9Jܺ       �	(�Yc�A�**

loss��;�	u.       �	~�Yc�A�**

losszEI<І۾       �	fIYc�A�**

loss

 :h34�       �	Y�Yc�A�**

loss�9�<��       �	�yYc�A�**

loss�m0=9�q�       �	aYc�A�**

loss�U�<��Rz       �	��Yc�A�**

loss��w=�lFX       �	�MYc�A�**

loss|��</5;       �	�'Yc�A�**

loss(�	;I`�B       �	��Yc�A�**

loss�$�;yPt�       �	GvYc�A�**

loss�[9=��2�       �	�Yc�A�**

lossZ{�;�-�       �	��Yc�A�**

loss�۬=w�       �	�= Yc�A�**

loss-�k=�k��       �	� Yc�A�**

lossZrA<��p�       �	vl!Yc�A�**

loss�?�<8��       �	�#"Yc�A�**

loss��f<���P       �	��"Yc�A�**

loss�=��       �	�W#Yc�A�**

loss��<"�x0       �	�#Yc�A�**

lossט�<����       �	q�$Yc�A�**

loss�;FzQ�       �	�%Yc�A�**

lossW.�<-��       �	��%Yc�A�**

losshط=�Vq{       �	{�&Yc�A�**

lossCX0<�랍       �	�'Yc�A�**

loss��3=���(       �	�E(Yc�A�**

loss�&�;��h�       �	��(Yc�A�**

loss��<aX       �	J�)Yc�A�**

loss�v7=�-�R       �	]6*Yc�A�**

loss�&<a~�       �	x�*Yc�A�+*

lossi��<����       �	m+Yc�A�+*

loss�X<k�f       �	f,Yc�A�+*

loss׏�;�~U�       �	�,Yc�A�+*

loss��.<j�Z4       �	�X-Yc�A�+*

loss �<@H�5       �	\�-Yc�A�+*

loss,>�;���       �	�.Yc�A�+*

lossi� =6g�p       �	�7/Yc�A�+*

loss� =r�       �	��/Yc�A�+*

lossCU�=�ː}       �	 r0Yc�A�+*

loss�g�;n�z�       �	�1Yc�A�+*

loss��<ιX�       �	��1Yc�A�+*

lossDe:�41       �	O?2Yc�A�+*

loss��H<���       �	@�2Yc�A�+*

loss��'=C�       �	.q3Yc�A�+*

loss��9=r'L       �	�4Yc�A�+*

loss{Pa<'�BO       �	Û4Yc�A�+*

loss�|E<0=�z       �	�?5Yc�A�+*

loss�,�<5t`       �	v�5Yc�A�+*

loss���;�        �	ʈ6Yc�A�+*

loss�:�;�j�:       �	�*7Yc�A�+*

loss�z�:dY�       �	8Yc�A�+*

loss&��;�<`�       �	s�8Yc�A�+*

loss\[<
(Z        �	�C9Yc�A�+*

loss�$;�#�       �	��9Yc�A�+*

lossz�=Gh�       �	��:Yc�A�+*

lossM	�;���       �	 %;Yc�A�+*

lossI
+<�% *       �	��;Yc�A�+*

loss�6/=xڡ       �	�j<Yc�A�+*

loss輔<�$       �	�=Yc�A�+*

loss��:i��       �	N�=Yc�A�+*

loss�X<Z�5q       �	x>Yc�A�+*

lossl!w;3�       �	w?Yc�A�+*

loss/Fo<6�
�       �	�?Yc�A�+*

loss�	};���       �	��@Yc�A�+*

lossw�>;��[�       �	�-AYc�A�+*

loss�C=�       �	��AYc�A�+*

loss���:I4|�       �	��BYc�A�+*

loss�*=]	�7       �	��CYc�A�+*

loss�ّ<;�U       �	"DYc�A�+*

loss��;M-@�       �	�DYc�A�+*

loss�U =�es       �	�\EYc�A�+*

lossH$�;Rݶ�       �	��EYc�A�+*

loss�ώ<�8�'       �	ԚFYc�A�+*

loss��9E�u�       �	QKGYc�A�+*

lossD
:+�ѩ       �	~�GYc�A�+*

lossX�;�#��       �	��HYc�A�+*

lossb{:68�       �	�AIYc�A�+*

loss3M+<�t�	       �	�IYc�A�+*

lossj�<�       �	�JYc�A�+*

loss1�4=?�qj       �	�KYc�A�+*

lossn�I<ET�_       �	.qLYc�A�+*

loss�U�:�G_       �	8MYc�A�+*

lossm��;��       �	rNYc�A�+*

loss���<RĬ       �	�NYc�A�+*

loss��,=���       �	��OYc�A�+*

loss�Rq;��$�       �	�zPYc�A�+*

loss�?@;SI��       �	�QYc�A�+*

losscg;��^       �	�QYc�A�+*

loss6u�<T��       �	TSRYc�A�+*

loss���9���       �	��SYc�A�+*

loss�/><�bv~       �	�&TYc�A�+*

lossr6;O"�       �	,�TYc�A�+*

lossf5�;!��       �	ӅUYc�A�+*

loss�<JM�       �	�VYc�A�+*

loss��P;:��       �	h?WYc�A�+*

loss���<�l       �	F$XYc�A�+*

loss�&�;V/N       �	C�XYc�A�+*

loss�H[;86t�       �	�ZYc�A�+*

lossN��<u�c�       �	�ZYc�A�+*

lossM�?<��h       �	c�[Yc�A�+*

loss;2:Կ/�       �	v4\Yc�A�+*

loss�?;f��s       �	�Q]Yc�A�+*

loss-B�;���       �	*^Yc�A�+*

lossa|c<V��       �	��^Yc�A�+*

loss�l�;��0�       �	8�_Yc�A�+*

loss@N;����       �	�KaYc�A�+*

loss:]9.m"�       �	��aYc�A�+*

loss��<PnnK       �	8�bYc�A�+*

lossX�;�4       �	OcYc�A�+*

lossn�";��>�       �	��cYc�A�+*

loss���:��5       �	�`dYc�A�+*

lossfy5:Ȣ��       �	��dYc�A�+*

losso[5<t�+u       �	F�eYc�A�+*

loss�lG<�$9�       �	�RfYc�A�+*

loss�X�=�o��       �	qgYc�A�+*

lossfv$<����       �	7�gYc�A�+*

loss�_�=��       �	�BhYc�A�+*

lossC�:+}�R       �	��hYc�A�+*

loss��W:����       �	}xiYc�A�+*

loss�<F�{�       �	�jYc�A�+*

loss�E�:��4       �	1kYc�A�+*

loss��#;tT       �	�kYc�A�+*

lossNm�<U�k#       �	Y6lYc�A�+*

loss�6H<z�e       �	��lYc�A�+*

loss	�<
��f       �	��mYc�A�+*

loss�_:�<	       �	c}nYc�A�+*

loss3rm;!��       �	IoYc�A�+*

loss�7�:#�O       �	�oYc�A�+*

loss�Y�;�Z(       �	�QpYc�A�+*

loss��;'��t       �	��pYc�A�+*

loss��c<�z�C       �	z�qYc�A�+*

loss{��:z�c�       �	�$rYc�A�+*

loss�V%<x%�       �	��rYc�A�+*

loss�>�<C�6�       �	�^sYc�A�+*

loss�c�9Mvi|       �	�sYc�A�+*

loss� =�!�K       �	L�tYc�A�+*

loss�Z;����       �	5%uYc�A�+*

loss.�<�       �	�uYc�A�+*

lossE�;x��Y       �	�QvYc�A�+*

loss`ݬ;+�g       �	��vYc�A�+*

loss�w<�9�?       �	��wYc�A�+*

loss��=�k�f       �	�"xYc�A�+*

loss�Ѐ=$!�       �	'�xYc�A�+*

lossr=b0`       �	�^yYc�A�+*

loss$u=��d       �	#�yYc�A�+*

loss�]K=��na       �		�zYc�A�+*

loss�W�<��g       �	B#{Yc�A�+*

loss��;�\r[       �	��{Yc�A�+*

loss_Z�;_{�k       �	mV|Yc�A�+*

loss}1�;�J�       �	��|Yc�A�+*

loss�e�9bp\       �	ƅ}Yc�A�+*

loss|�S;��35       �	~Yc�A�+*

loss���;�.�       �	c�~Yc�A�+*

loss�H�<wv       �	�NYc�A�+*

loss�|r;(�5�       �	��Yc�A�+*

lossӷ�;J`��       �	��Yc�A�+*

loss�t<���       �	�$�Yc�A�+*

loss�^�;Y}4       �	��Yc�A�,*

loss��T<�މ�       �	\�Yc�A�,*

loss��;�HZD       �	��Yc�A�,*

lossTI=dxC�       �	��Yc�A�,*

loss��h;3!�U       �	�6�Yc�A�,*

loss��< V�V       �	�̄Yc�A�,*

loss<M�
R       �	�h�Yc�A�,*

loss���<���       �	_�Yc�A�,*

loss�"�;�m#N       �	���Yc�A�,*

lossᩧ:M��       �	�=�Yc�A�,*

loss���;��       �	ևYc�A�,*

loss��^<:�U       �	�u�Yc�A�,*

loss�^�<w,�       �	��Yc�A�,*

loss���;�
�f       �	��Yc�A�,*

loss�i=ǱD       �	o�Yc�A�,*

loss���<�K�       �	��Yc�A�,*

lossMU=�;��       �	N�Yc�A�,*

loss�=��       �	�ӌYc�A�,*

lossAP�<a�O       �	8j�Yc�A�,*

lossIt<¥
�       �	�5�Yc�A�,*

loss��=;�D��       �	��Yc�A�,*

loss��<h��9       �	t�Yc�A�,*

loss#�;òOp       �	���Yc�A�,*

lossY�9��f�       �	�>�Yc�A�,*

loss*�<��L5       �	�ՑYc�A�,*

loss��<	�       �	m�Yc�A�,*

loss1�<��S       �	9�Yc�A�,*

loss�_�:����       �	@��Yc�A�,*

loss���<V�/       �	qU�Yc�A�,*

loss��=Ԁ��       �	�Yc�A�,*

loss��"<�WQ       �	��Yc�A�,*

loss�Y�<}Y��       �	q<�Yc�A�,*

loss��:�$�       �	�ݖYc�A�,*

loss�,�;kh       �	�r�Yc�A�,*

loss���9~       �	.Y�Yc�A�,*

losss�S;�Hh       �	���Yc�A�,*

loss�<B=��Fg       �	y��Yc�A�,*

loss�p�;�v.       �	�Yc�A�,*

loss�5=<�T       �	I�Yc�A�,*

lossD�;	7#       �	��Yc�A�,*

lossS�;���9       �	�z�Yc�A�,*

loss��:�ӕ�       �	�K�Yc�A�,*

loss�;��        �	X�Yc�A�,*

loss>B�<�E�j       �	[|�Yc�A�,*

loss��D<�A�       �	��Yc�A�,*

loss���:~��       �	��Yc�A�,*

loss\�<:�a#�       �	k��Yc�A�,*

loss@
�<�_6o       �	�1�Yc�A�,*

loss�M�;+-:�       �	�ơYc�A�,*

loss���9w#d       �	`�Yc�A�,*

loss�3�9��k       �	��Yc�A�,*

loss��=;0q|       �	"��Yc�A�,*

loss�9�<b�@�       �	[?�Yc�A�,*

lossvn=��       �	MڤYc�A�,*

loss��:>g�^       �	�s�Yc�A�,*

loss��V=IZ��       �	�*�Yc�A�,*

loss���:1-�       �	�ƦYc�A�,*

loss��z=��2(       �	o�Yc�A�,*

loss���;�T�       �	�Yc�A�,*

loss<Y9��1f       �	��Yc�A�,*

loss�nf;R��       �	돩Yc�A�,*

loss�j�9���       �	4�Yc�A�,*

loss�U9&�|�       �	�ҪYc�A�,*

loss��g:�~�       �	Cu�Yc�A�,*

loss)�T=6��9       �	�Yc�A�,*

loss�]<�}�
       �	�ĬYc�A�,*

loss/��<͉��       �	�ǭYc�A�,*

loss�V�<�w�W       �	��Yc�A�,*

loss/ �<��       �	�*�Yc�A�,*

lossŅE<Da       �	nïYc�A�,*

loss�D<cs�i       �	���Yc�A�,*

loss�E6<0}��       �	s��Yc�A�,*

loss�c<:�PW       �	C<�Yc�A�,*

loss�2�<��}       �	�Yc�A�,*

loss��<:ؼ�       �	��Yc�A�,*

lossX�=�pW=       �	�#�Yc�A�,*

loss�D�;r�       �	LǴYc�A�,*

loss��:�J�       �	�]�Yc�A�,*

loss3��9[�B{       �	^��Yc�A�,*

lossħ�<�e��       �	Y��Yc�A�,*

loss�n;.��       �	�B�Yc�A�,*

loss?�<MO*�       �	���Yc�A�,*

loss���:Z�:c       �	Ú�Yc�A�,*

loss�[�<ef       �	�f�Yc�A�,*

loss {W=�B#_       �	�Yc�A�,*

loss:��<�1�       �	3��Yc�A�,*

loss�K;���Z       �	EI�Yc�A�,*

lossj�2<"7�       �	��Yc�A�,*

lossυ<�﮼       �	���Yc�A�,*

loss�x�<��       �	j1�Yc�A�,*

loss�\�<l�#       �	�ʽYc�A�,*

lossڭt<���r       �	�f�Yc�A�,*

loss}N�:gx�(       �	��Yc�A�,*

loss4)o;��_�       �	��Yc�A�,*

loss�:8=���       �	]��Yc�A�,*

loss
�<���       �	�.�Yc�A�,*

lossh)%<��ث       �	���Yc�A�,*

lossY�<�h        �	`�Yc�A�,*

losso0=.���       �	]��Yc�A�,*

lossD2<����       �	��Yc�A�,*

loss.E�:��       �	e6�Yc�A�,*

lossQ?�:�w��       �	���Yc�A�,*

loss�ݏ<pd�       �	@h�Yc�A�,*

loss|((>��z       �	��Yc�A�,*

loss ւ<v�qQ       �	o��Yc�A�,*

lossR�M<�=�?       �	�1�Yc�A�,*

lossW@�< �x;       �	��Yc�A�,*

lossI�<n�v        �	�]�Yc�A�,*

loss���9�g(       �	s��Yc�A�,*

loss[�S<{��       �	��Yc�A�,*

loss��<"N�       �	:"�Yc�A�,*

loss�s�<��"       �	7�Yc�A�,*

loss;��<�|        �	��Yc�A�,*

lossj��=yF��       �	h�Yc�A�,*

lossX��<�-&       �	h�Yc�A�,*

loss|G_<�}2�       �	_��Yc�A�,*

loss�l=^���       �	7R�Yc�A�,*

losso=����       �	n��Yc�A�,*

loss�.;t[�       �	���Yc�A�,*

loss�48;R�       �	I�Yc�A�,*

loss1�:L
@>       �	}��Yc�A�,*

lossrh�<��4#       �	��Yc�A�,*

loss/W�:�U�
       �	�#�Yc�A�,*

loss���;����       �	!��Yc�A�,*

loss���:P!U�       �	{i�Yc�A�,*

loss�|�;�J       �	�Yc�A�,*

loss�%;{j|       �	0��Yc�A�,*

loss$3�<��i�       �	HR�Yc�A�,*

loss�>=]w�       �	9��Yc�A�-*

loss�U�<F�ߩ       �	��Yc�A�-*

lossj��;�Hz$       �	{-�Yc�A�-*

loss��w;u�       �	���Yc�A�-*

loss��<���       �	�^�Yc�A�-*

loss"X=6��       �	��Yc�A�-*

lossf3�=��v|       �	���Yc�A�-*

loss�.P<�O       �	�G�Yc�A�-*

loss�H�;'O��       �	��Yc�A�-*

loss:A<˭�R       �	���Yc�A�-*

loss-�<�k�       �	�(�Yc�A�-*

lossf��;Ŀ#        �	��Yc�A�-*

loss_P�;�=��       �	=_�Yc�A�-*

lossm�	;�f9       �	L��Yc�A�-*

loss��:w�+&       �	���Yc�A�-*

loss�s1<wLZ�       �	�.�Yc�A�-*

loss�7b=4?       �	���Yc�A�-*

loss*(�<��Q�       �	�v�Yc�A�-*

loss�t�<� `       �	d!�Yc�A�-*

loss�N=?�.-       �	^��Yc�A�-*

lossoYK<�;K       �	�R�Yc�A�-*

lossCS<\y=       �	���Yc�A�-*

loss�\=	��       �	Ǆ�Yc�A�-*

loss&n�<�Y�       �	��Yc�A�-*

loss��=7��I       �	&��Yc�A�-*

loss�4�<X���       �	�`�Yc�A�-*

loss}O+>1�2       �	W�Yc�A�-*

loss�I�<�1m�       �	t��Yc�A�-*

loss��<H���       �	�_�Yc�A�-*

loss��:5xe       �	��Yc�A�-*

loss��=<�T       �	���Yc�A�-*

loss��<~�;(       �	/k�Yc�A�-*

loss-�<����       �	k�Yc�A�-*

loss���;�D7       �	/��Yc�A�-*

loss$��<�oR       �	h=�Yc�A�-*

lossGV�;.8�       �	��Yc�A�-*

loss�d�=�-�       �	�t�Yc�A�-*

loss���:��{       �	��Yc�A�-*

loss��;]�/       �	հ�Yc�A�-*

loss��
<n�R>       �	Ih�Yc�A�-*

loss���<���       �	"��Yc�A�-*

loss�%<�k�}       �	)]�Yc�A�-*

loss��;G�_       �	j��Yc�A�-*

loss�i�<��:�       �	w��Yc�A�-*

lossC&�<�I�^       �	���Yc�A�-*

loss\,�< �       �	 ��Yc�A�-*

loss�GH<�h�       �	�9�Yc�A�-*

loss2B�=7 �C       �	��Yc�A�-*

loss�9<1��       �	R��Yc�A�-*

lossͅ�<�x(�       �	�o�Yc�A�-*

loss^};.<��       �	�0�Yc�A�-*

loss���<��S       �	��Yc�A�-*

loss��=���       �	@h�Yc�A�-*

loss�7<<���       �	�Yc�A�-*

lossY�<O��       �	o��Yc�A�-*

loss��<B��       �	O;�Yc�A�-*

loss �/=l��h       �	h��Yc�A�-*

loss�k{:B�       �	͓�Yc�A�-*

loss��;"返       �	�*�Yc�A�-*

loss�Z�<��E       �	B��Yc�A�-*

lossW8�<�Gm�       �		l�Yc�A�-*

loss��<P��j       �	* �Yc�A�-*

loss>ۓ<kΉ       �	���Yc�A�-*

loss|� =�&"K       �	bi�Yc�A�-*

loss��=����       �	c Yc�A�-*

loss�]�<G�l       �	�� Yc�A�-*

loss��$=Ɲ�        �	+NYc�A�-*

loss���<�n�       �	��Yc�A�-*

losso��;��y!       �	Y�Yc�A�-*

loss���<��;       �	�%Yc�A�-*

loss\XC;����       �	��Yc�A�-*

loss׈�<R>d       �	0cYc�A�-*

lossܚ=�a�       �	]�Yc�A�-*

lossQ��;��       �	�Yc�A�-*

loss�&<��)�       �	�AYc�A�-*

loss 0<nV�       �	{�Yc�A�-*

lossc�<�i�d       �	|Yc�A�-*

lossj�<��       �	 Yc�A�-*

loss�@�<3��       �	�Yc�A�-*

loss]�==��       �	a	Yc�A�-*

loss�n�<V#�H       �	7 
Yc�A�-*

loss�"=���n       �	I�
Yc�A�-*

loss�T�=8�Ӧ       �	�DYc�A�-*

loss3JP<�(T1       �	:�Yc�A�-*

lossh<�@L�       �	�Yc�A�-*

loss2f+<dsHz       �	�&Yc�A�-*

lossj�;�8�       �	׾Yc�A�-*

loss�*?=6�2*       �	�\Yc�A�-*

losst�u<        �	��Yc�A�-*

loss�4�<Az9�       �	�Yc�A�-*

lossa9<��       �	�BYc�A�-*

lossX&�<J*ܐ       �	+�Yc�A�-*

loss$��;�ʐ�       �	pYc�A�-*

loss��c;\�en       �	�Yc�A�-*

loss��3<n�Z�       �	��Yc�A�-*

loss���<p�       �	�BYc�A�-*

loss��s=�i       �	�Yc�A�-*

lossT[H<��i]       �	�oYc�A�-*

loss���:l�r       �	L7Yc�A�-*

loss�k�;MɊ�       �	F�Yc�A�-*

loss�U2<©       �	�eYc�A�-*

loss��;
��0       �	��Yc�A�-*

loss��*;�9�       �	p�Yc�A�-*

loss�!<���^       �	]1Yc�A�-*

loss�7�=��Y       �	��Yc�A�-*

loss��F=�<��       �	܂Yc�A�-*

loss//#<�F��       �	Yc�A�-*

loss!ʄ<*�C       �	�Yc�A�-*

lossu�<p�x�       �	�SYc�A�-*

loss�,�<��       �	��Yc�A�-*

loss;�y:?њ�       �	�Yc�A�-*

lossi�?=N�l�       �	�)Yc�A�-*

loss��<q�       �	v�Yc�A�-*

loss�;���J       �	W`Yc�A�-*

loss��x=Lv��       �	7 Yc�A�-*

loss��E;��       �	��Yc�A�-*

lossT��<�W       �	>= Yc�A�-*

loss1�<_�#�       �	M� Yc�A�-*

loss�z�<�-,�       �	�z!Yc�A�-*

loss�X�;x��-       �	�"Yc�A�-*

loss��=áM�       �	�"Yc�A�-*

loss��<�r�       �	zQ#Yc�A�-*

loss��R<���M       �	��#Yc�A�-*

loss�_H<Qx��       �	�~$Yc�A�-*

loss�D@=y�+       �	%Yc�A�-*

loss��K<z���       �	j�%Yc�A�-*

loss긖<!��       �	�X&Yc�A�-*

lossL.j=�ֻW       �	��&Yc�A�-*

loss.l�<h7K       �	_�'Yc�A�.*

loss�@�<��F       �	�(Yc�A�.*

loss6�A;�rdL       �	#0)Yc�A�.*

lossX��<��%�       �	��)Yc�A�.*

lossF��<���       �	��*Yc�A�.*

loss\�<�v�       �	�+Yc�A�.*

loss��W<�(�       �	��+Yc�A�.*

loss�;p�:       �	�],Yc�A�.*

loss@�<�9�i       �	�-Yc�A�.*

lossm�<�~�O       �	��-Yc�A�.*

loss}�%;�>�       �	��.Yc�A�.*

loss!��<�k       �	jK/Yc�A�.*

loss�;�;J       �	��/Yc�A�.*

loss�q�<�`(�       �	��0Yc�A�.*

loss@;���       �	hB1Yc�A�.*

loss���<���B       �	��1Yc�A�.*

lossQrb=����       �	ڍ2Yc�A�.*

loss�"�<1A7       �	I/3Yc�A�.*

lossۙ�;D_H�       �	��3Yc�A�.*

lossI�<*]
F       �	܃4Yc�A�.*

lossr��;6���       �	�b5Yc�A�.*

loss��Y;ը�J       �	�6Yc�A�.*

loss��;���U       �	��6Yc�A�.*

loss�E;���       �	"O7Yc�A�.*

loss�C�<�K       �	5�7Yc�A�.*

lossD��;K$!       �	C�8Yc�A�.*

loss��=,ŷ�       �	�59Yc�A�.*

loss)��;[V��       �	��9Yc�A�.*

loss��;�!       �	�~:Yc�A�.*

loss`k=�Pm       �	?;Yc�A�.*

loss:�L;)�p�       �	�;Yc�A�.*

loss�'�:E�       �	Zc<Yc�A�.*

loss��2<me�       �	v�<Yc�A�.*

loss}:��Z;       �	�=Yc�A�.*

loss�/(<2��       �	�G>Yc�A�.*

loss�q;P��       �	7�>Yc�A�.*

loss��;�N�       �	c�?Yc�A�.*

loss8B<��&�       �	�@Yc�A�.*

lossEx_<�a>�       �	��@Yc�A�.*

loss*qA;G'��       �	�IAYc�A�.*

loss;A�<)p�       �	��AYc�A�.*

loss�o�=���       �	H�BYc�A�.*

loss|m�<�e�       �	p#CYc�A�.*

loss��9�t��       �	DYc�A�.*

loss�8�<�T        �	�DYc�A�.*

loss�C�;m�`1       �	-�EYc�A�.*

loss%b�9����       �	3FYc�A�.*

lossޕ:Q��_       �	��FYc�A�.*

lossS\�<n���       �	�_GYc�A�.*

lossQ1�:*�o�       �	4IHYc�A�.*

lossE��<�B�I       �	��HYc�A�.*

loss/�8��P�       �	ǠIYc�A�.*

loss�	�<j��       �	`:JYc�A�.*

lossi7U9��֚       �	��JYc�A�.*

loss�:h8�ϖM       �	�KYc�A�.*

lossx�8����       �	� LYc�A�.*

loss2
�:��sb       �	��LYc�A�.*

loss�=�)�       �	hZMYc�A�.*

loss:�=;�p       �	��MYc�A�.*

loss.�I;���7       �	ҎNYc�A�.*

loss�Q�;led       �	A-OYc�A�.*

lossi>W�%       �	��OYc�A�.*

loss�n;�ey�       �	#jPYc�A�.*

loss��=R��p       �	1QYc�A�.*

loss��;U�       �	��QYc�A�.*

loss��==|)N       �	^�RYc�A�.*

loss�;�:��       �	-SYc�A�.*

loss���;���       �	L�SYc�A�.*

lossf��<u���       �	gdTYc�A�.*

loss���;�҉�       �	��UYc�A�.*

loss���<vt�       �	3�VYc�A�.*

lossq2j<g�;�       �	�#WYc�A�.*

loss�I�9�H+       �	@�WYc�A�.*

loss�i+=8���       �	�xXYc�A�.*

loss�<	�4T       �	^YYc�A�.*

loss�m;����       �	ѭYYc�A�.*

loss�<s�[�       �	�GZYc�A�.*

loss
�	<eiAJ       �	:�ZYc�A�.*

loss/+x=����       �	w�[Yc�A�.*

lossv^;� �       �	_(\Yc�A�.*

loss\�5=6A�`       �	��\Yc�A�.*

lossiy<Kë@       �	�i]Yc�A�.*

loss�r<>&9       �	& ^Yc�A�.*

loss��L=,)��       �	��^Yc�A�.*

loss�tJ=�r��       �	�,_Yc�A�.*

lossO�<&�q�       �	]�_Yc�A�.*

loss�I�9��}�       �	�\`Yc�A�.*

loss��<���h       �	4�`Yc�A�.*

lossT��<W\��       �	�aYc�A�.*

lossA��<�.       �	eSbYc�A�.*

loss�!=I�Z       �	S�bYc�A�.*

loss	f�<�f�       �	ӇcYc�A�.*

loss���;L+�       �	�0dYc�A�.*

loss9<Xl��       �	
�dYc�A�.*

loss;��!       �	�teYc�A�.*

lossd.=���       �	fYc�A�.*

loss�b8<�-�R       �	��fYc�A�.*

lossz�6;��W       �	^HgYc�A�.*

loss߸\:nGՐ       �	rhYc�A�.*

loss��*<��`]       �	�hYc�A�.*

loss=��A�       �	
HiYc�A�.*

lossi�;X�n       �	+�iYc�A�.*

lossf��:iU       �	%yjYc�A�.*

loss�az;��h�       �	bkYc�A�.*

loss�ȃ=`��       �	֭kYc�A�.*

loss�*=��2       �	AFlYc�A�.*

loss�k;��1       �	��lYc�A�.*

loss^�;�+��       �	>ymYc�A�.*

lossH+G=��       �	�nYc�A�.*

loss|T:}�7z       �	6�nYc�A�.*

loss<��<^�A�       �	AHoYc�A�.*

loss��
;��^       �	��oYc�A�.*

loss\zg<�,~[       �	�~pYc�A�.*

loss7�>;m���       �	L�Yc�A�.*

loss	��;�&       �	�ǌYc�A�.*

loss�^=l��-       �	�d�Yc�A�.*

loss��H<dN"       �		��Yc�A�.*

loss�*�;�ܷ6       �	��Yc�A�.*

loss�z�<�j�t       �	cc�Yc�A�.*

loss��<��Ծ       �	���Yc�A�.*

loss�S[<�q       �	ު�Yc�A�.*

loss;�N<�W`�       �	��Yc�A�.*

loss�6^<Ơ��       �	���Yc�A�.*

loss'�:<��x       �	W[�Yc�A�.*

loss�RD;NUb�       �	��Yc�A�.*

lossT��;�[u�       �	���Yc�A�.*

loss���; "�	       �	��Yc�A�.*

lossL�@<M���       �	�*�Yc�A�.*

loss�E|:�r��       �	�ÖYc�A�/*

loss�Q9?s*       �	�`�Yc�A�/*

loss�'�:mD       �	��Yc�A�/*

loss&��;zD�       �	��Yc�A�/*

loss�SJ<$�e       �	M,�Yc�A�/*

loss��0=�B�       �	Yc�A�/*

loss��=F�{       �	�]�Yc�A�/*

lossδH<|�׏       �	���Yc�A�/*

lossp�<����       �	���Yc�A�/*

loss��:;*��l       �	p$�Yc�A�/*

lossa�;��Q�       �	���Yc�A�/*

loss�R<b p/       �		S�Yc�A�/*

loss��;7'N�       �	��Yc�A�/*

loss�<��
       �	��Yc�A�/*

loss�:�e�       �	��Yc�A�/*

loss��9N���       �	-��Yc�A�/*

loss
'�;t��2       �	�J�Yc�A�/*

loss��9k̼       �	��Yc�A�/*

loss� �=H�%       �	׉�Yc�A�/*

lossڶ�<�G��       �	*�Yc�A�/*

loss&m�<�f       �	���Yc�A�/*

loss7/:���       �	t_�Yc�A�/*

lossO�<l�A       �	��Yc�A�/*

loss!�<��}       �	���Yc�A�/*

loss_��<I�b       �	�/�Yc�A�/*

lossh��<>;-�       �	�ƥYc�A�/*

loss{��:W^�       �	xc�Yc�A�/*

loss�	\:F��       �	��Yc�A�/*

lossA�<<�8^       �	B��Yc�A�/*

loss�y=]�#       �	�.�Yc�A�/*

loss�N�=^u�       �	YĨYc�A�/*

loss��R<�s       �	>Z�Yc�A�/*

loss��=�`c2       �	R�Yc�A�/*

loss��:�Gl�       �	y��Yc�A�/*

lossjO\:j/7       �	�8�Yc�A�/*

loss��<`�lQ       �	9իYc�A�/*

loss" <�I�M       �	�i�Yc�A�/*

loss:�;D���       �	}�Yc�A�/*

loss� �<qo�0       �	���Yc�A�/*

loss��;�9��       �		8�Yc�A�/*

loss�*:;SP�$       �	$ӮYc�A�/*

loss_+<j�M�       �	�ȯYc�A�/*

loss·;���       �	�c�Yc�A�/*

loss� ;����       �	��Yc�A�/*

loss�A<�7       �	��Yc�A�/*

lossS��=7���       �	O��Yc�A�/*

loss[�>�_"�       �	X�Yc�A�/*

lossc�=��_       �	��Yc�A�/*

losst �:ڳ5       �	��Yc�A�/*

loss?X�;k�       �	�^�Yc�A�/*

loss�a<�3#�       �	��Yc�A�/*

loss�_<�!��       �	��Yc�A�/*

loss�1�;��6       �	���Yc�A�/*

loss���<�^Q       �	l	�Yc�A�/*

loss��<�+L       �	)��Yc�A�/*

loss�<3,�K       �	S�Yc�A�/*

loss\D�;�r�       �	U��Yc�A�/*

loss�;P_rU       �	���Yc�A�/*

lossF]�<'���       �	5C�Yc�A�/*

lossZ�=ԣ��       �	�޽Yc�A�/*

loss(=�ȅ�       �	�{�Yc�A�/*

loss�3<:�        �	��Yc�A�/*

loss���:��F�       �	���Yc�A�/*

loss�x<PЉ`       �	�M�Yc�A�/*

loss��	<�]��       �	���Yc�A�/*

loss���<���       �	O��Yc�A�/*

loss��;	�       �	�(�Yc�A�/*

loss���;�	��       �	��Yc�A�/*

loss0�:�v0�       �	f�Yc�A�/*

loss��=�I>�       �	�#�Yc�A�/*

loss�T�;�(f�       �	��Yc�A�/*

loss��;A��       �	�M�Yc�A�/*

loss~��<ߧ�       �	���Yc�A�/*

loss[��;��u0       �		��Yc�A�/*

loss�N;�J�       �	&�Yc�A�/*

lossD2<e�ut       �	6��Yc�A�/*

loss�z�=j{ő       �	Ra�Yc�A�/*

loss��:X"H�       �	���Yc�A�/*

loss�%�<5�x�       �	-��Yc�A�/*

lossω<�Č7       �	�.�Yc�A�/*

loss��:�c�       �	\��Yc�A�/*

lossv�B=~���       �	�_�Yc�A�/*

loss���<;���       �	���Yc�A�/*

loss�r1=��#       �	���Yc�A�/*

loss���;����       �	H�Yc�A�/*

loss�;�x��       �	���Yc�A�/*

loss+�;��,�       �	�s�Yc�A�/*

loss�;@��9       �	��Yc�A�/*

loss|�;�m��       �	i��Yc�A�/*

loss�};P���       �	p@�Yc�A�/*

lossrZ�;N�8       �	���Yc�A�/*

loss�a�9��h�       �	��Yc�A�/*

loss�*�:�,d�       �	F��Yc�A�/*

loss�E&=�KU�       �	YQ�Yc�A�/*

loss�p;Ҥ�       �	���Yc�A�/*

loss��w<��o�       �	R��Yc�A�/*

loss�k<�
�;       �	�k�Yc�A�/*

loss7�0=�>�       �	\�Yc�A�/*

loss��:�X       �	���Yc�A�/*

loss��;ZnU�       �	�f�Yc�A�/*

lossXL�;�UuY       �	��Yc�A�/*

loss��;\:��       �	ҧ�Yc�A�/*

loss/� =qc&K       �	>B�Yc�A�/*

loss� F;�	�       �	���Yc�A�/*

lossm�=B9g'       �	py�Yc�A�/*

loss�P�;�.Ai       �	<�Yc�A�/*

lossx��<��V       �	��Yc�A�/*

losst��:�5n       �	i��Yc�A�/*

loss�=�ϰ�       �	�(�Yc�A�/*

loss3^�<�S��       �	���Yc�A�/*

loss�=�1�3       �	a�Yc�A�/*

lossh��;��j       �	/��Yc�A�/*

losss�_<L>�B       �	%��Yc�A�/*

lossz =?��       �	�5�Yc�A�/*

lossC#�;kE7�       �	��Yc�A�/*

lossi�:��t       �	�e�Yc�A�/*

loss�� <�2       �	���Yc�A�/*

loss=�=b[O       �	֎�Yc�A�/*

loss�!�:��x�       �	�&�Yc�A�/*

loss�V<����       �	���Yc�A�/*

lossL+	<h��       �	�Q�Yc�A�/*

lossaW^=`��       �	��Yc�A�/*

loss���:��A�       �	X��Yc�A�/*

loss�;n11        �	�%�Yc�A�/*

lossE�_:9	       �	0��Yc�A�/*

losst�:�"�       �	&T�Yc�A�/*

lossó�<��t�       �	���Yc�A�/*

loss�d<T��       �	\��Yc�A�/*

loss�\<P��       �	"�Yc�A�0*

loss��<h��*       �	���Yc�A�0*

loss)bX<F*_       �	J`�Yc�A�0*

lossA*�<DЗ�       �	3��Yc�A�0*

loss�;(�s       �	͏�Yc�A�0*

loss�91<�ϗ>       �	�,�Yc�A�0*

loss#�y<T�       �		��Yc�A�0*

loss�6$<�,:-       �	�Y�Yc�A�0*

loss�%�;�5�       �	���Yc�A�0*

loss�"�:N�.       �	��Yc�A�0*

lossI�v<1�}       �	P8�Yc�A�0*

loss�/�<��k       �	b��Yc�A�0*

loss� 9]�E�       �	r�Yc�A�0*

lossQ�:َC*       �	��Yc�A�0*

loss��=�J�       �	&��Yc�A�0*

loss��D;SC�.       �	u?�Yc�A�0*

loss���<>�E9       �	�Yc�A�0*

loss�0<���       �	L��Yc�A�0*

loss3}P;��K�       �	�f�Yc�A�0*

losscp�<G��       �	 �Yc�A�0*

loss��9�n.�       �	���Yc�A�0*

losst��:	�Q       �	cA�Yc�A�0*

lossqv�:Qd�w       �	���Yc�A�0*

loss�;g��       �	A��Yc�A�0*

lossT�@<��
r       �	�Yc�A�0*

loss;a<�@-       �	%��Yc�A�0*

loss���9_&,       �	�M�Yc�A�0*

lossd}=�⏥       �	���Yc�A�0*

loss� �<��       �	���Yc�A�0*

loss�R�<���       �	\�Yc�A�0*

loss��l<�E       �	̵�Yc�A�0*

loss�L6;��ä       �	�j�Yc�A�0*

loss
l�<�Rc       �	��Yc�A�0*

loss���;+��       �	��Yc�A�0*

lossZ<SV`       �	G<�Yc�A�0*

lossO�{=�J�       �	��Yc�A�0*

loss�I�<�B�i       �	Xr�Yc�A�0*

loss$�9Z�?�       �	J	 Yc�A�0*

loss�O<{��       �	,� Yc�A�0*

loss�4�:x��       �	75Yc�A�0*

loss~�<��hZ       �	��Yc�A�0*

loss��x=c��       �	^�Yc�A�0*

loss)��<b�u�       �	dYc�A�0*

loss���;��/j       �	R�Yc�A�0*

loss.��;�1�       �	�PYc�A�0*

loss�1�<��M       �	d�Yc�A�0*

loss�<H���       �	�zYc�A�0*

loss�-�;)x�W       �	�Yc�A�0*

loss�5<�.�n       �	֬Yc�A�0*

loss��;��       �	�EYc�A�0*

loss�W;���       �	��Yc�A�0*

loss[?|:���       �	�vYc�A�0*

lossZ`m<�|��       �	9	Yc�A�0*

loss�&;�}%       �	8�	Yc�A�0*

lossډ�:_�ۭ       �	�:
Yc�A�0*

lossU�=t ��       �	F�
Yc�A�0*

loss,�;��5�       �	�iYc�A�0*

lossq=���e       �	7lYc�A�0*

lossð�;0*��       �	� Yc�A�0*

loss���9����       �	!Yc�A�0*

lossw�<Ļ5       �	�?Yc�A�0*

lossա<��!u       �	N'Yc�A�0*

loss�<:<�6S`       �	��Yc�A�0*

lossIwB=ۉQ       �	��Yc�A�0*

loss�G=� �L       �	��Yc�A�0*

loss��2<�-Q       �	�tYc�A�0*

loss��<���       �	�hYc�A�0*

loss��1;
�A3       �	qYc�A�0*

loss	�;�P�       �	ԙYc�A�0*

loss��s:�=h        �	�8Yc�A�0*

loss��;ׅ       �	R�Yc�A�0*

loss���;TS��       �	
�Yc�A�0*

loss�F=f�^�       �	W!Yc�A�0*

losszw�=����       �	�Yc�A�0*

loss��=FΛ�       �	ǟYc�A�0*

loss?��<�7©       �	�@Yc�A�0*

loss`��<���       �	��Yc�A�0*

loss@��;+qP|       �	��Yc�A�0*

lossʪ�:��@       �	3Yc�A�0*

loss|'j=�I       �	Z�Yc�A�0*

loss�4<��       �	iQYc�A�0*

loss�:{�U�       �	��Yc�A�0*

loss(�;#!ܖ       �	��Yc�A�0*

loss��<K!yO       �	�$Yc�A�0*

loss��r=�^�s       �	��Yc�A�0*

loss��:�{       �	^ Yc�A�0*

loss�!U<PM�       �	�� Yc�A�0*

loss`�<	?(�       �	��!Yc�A�0*

losslCn9��       �	"Yc�A�0*

loss �<�'W#       �	`�"Yc�A�0*

lossF)<�ߛC       �	DN#Yc�A�0*

lossAя<�43t       �	��#Yc�A�0*

lossji�;�z\�       �	Q�$Yc�A�0*

loss Y�<�B�       �	�}%Yc�A�0*

loss+�<�!       �	�&Yc�A�0*

loss��B;�V�       �	I�&Yc�A�0*

loss6+p9�n�       �	�X'Yc�A�0*

loss���;�hLV       �	I�'Yc�A�0*

loss��;:��:       �	��(Yc�A�0*

loss��<��0{       �	�?)Yc�A�0*

loss
�{=𑘭       �	��)Yc�A�0*

loss]=�ٹ       �	��*Yc�A�0*

loss64 <5@Q       �	�B+Yc�A�0*

loss7)=p�h       �	~�+Yc�A�0*

loss�[);i*i       �	��,Yc�A�0*

loss�Õ<�VF3       �	�*-Yc�A�0*

loss)�$<��j       �	B�-Yc�A�0*

loss��=n5gf       �	>v.Yc�A�0*

lossb%<;NU�       �	/Yc�A�0*

lossm�<| G       �	ɰ/Yc�A�0*

loss��:�O�       �	mT0Yc�A�0*

loss��M;���]       �	E-1Yc�A�0*

lossF(#=)��       �	��1Yc�A�0*

loss���<�)R�       �	e�2Yc�A�0*

loss*Xa;�\��       �	�y3Yc�A�0*

loss�d`<	_       �	4Yc�A�0*

lossH{s<Ϟ�       �	ک4Yc�A�0*

lossH��<�{��       �	q6Yc�A�0*

lossE*�;c�       �	��6Yc�A�0*

loss~�<�K�G       �	%X7Yc�A�0*

lossoi�;�>��       �	(�7Yc�A�0*

loss�P�;kTK       �	��8Yc�A�0*

loss��f<sa�(       �	d9Yc�A�0*

lossϏ=O0�       �	�9Yc�A�0*

loss$�O;*@e       �	T�:Yc�A�0*

loss��};�       �	�y;Yc�A�0*

loss�\<"��$       �	�<Yc�A�0*

loss(�n:�5��       �	��<Yc�A�0*

loss�j�<YqGF       �	�G=Yc�A�1*

loss�.=�Q�       �	e�=Yc�A�1*

loss&t,<S� �       �	yw>Yc�A�1*

loss��=��l�       �	�?Yc�A�1*

loss�I =�{��       �	��?Yc�A�1*

loss[�'<wyq       �	|d@Yc�A�1*

loss�:%*�K       �	.�@Yc�A�1*

loss�G�:ME�P       �	�AYc�A�1*

lossqۜ=<��       �	#/BYc�A�1*

losse�;m]q       �	��BYc�A�1*

loss���<e#z�       �	B_CYc�A�1*

lossZV	;(ۓ       �	0�CYc�A�1*

loss"�;�{��       �	d�DYc�A�1*

loss ��<�32       �	h%EYc�A�1*

loss,�-=F��4       �	��EYc�A�1*

loss�*e=|� �       �	�aFYc�A�1*

loss���:#��S       �	\ GYc�A�1*

loss��B;�Ӓ�       �	E�GYc�A�1*

loss!�`<�\/�       �	�wHYc�A�1*

loss�k	;`r�       �	IYc�A�1*

loss�/�<r^       �	��IYc�A�1*

loss��p;�MX7       �	pAJYc�A�1*

losssZ�9�"�       �	r�JYc�A�1*

loss�<wn��       �	~sKYc�A�1*

loss)�<�1�N       �	�LYc�A�1*

loss.�;m�P       �	v�LYc�A�1*

loss�}5=�(xh       �	�>MYc�A�1*

loss�!<i��u       �	C�MYc�A�1*

loss�hp=_	E�       �	8�NYc�A�1*

loss��;�[��       �	�OYc�A�1*

loss ]�:p(VV       �	ݵOYc�A�1*

loss�i:�k�       �	�JPYc�A�1*

loss ��;�.R�       �	�iQYc�A�1*

loss²<)'ǋ       �	qRYc�A�1*

loss
i�<�p3�       �	�&SYc�A�1*

lossoe�9���       �	q�SYc�A�1*

loss��s<���       �	VdTYc�A�1*

loss��:�'��       �	:UYc�A�1*

lossn��<߈U       �	B�UYc�A�1*

loss;�T<r�{       �	�qVYc�A�1*

loss�߈:Q�?       �	VWYc�A�1*

loss���<��)       �	c�WYc�A�1*

lossV�c<�YT�       �	�KXYc�A�1*

lossoU;@�}�       �	&�XYc�A�1*

lossL-<�%�       �	��YYc�A�1*

loss$|�;�i �       �	�IZYc�A�1*

loss��<pAI�       �	��ZYc�A�1*

lossx�5=�%c       �	�[Yc�A�1*

lossi~�;(\iJ       �	: \Yc�A�1*

loss�:;]��t       �	m�\Yc�A�1*

lossR�Y=h	��       �	W^]Yc�A�1*

loss���;���       �	��]Yc�A�1*

loss��<��q       �	��^Yc�A�1*

lossi�Y=�_�       �	Y_Yc�A�1*

loss�=�t��       �	��_Yc�A�1*

loss��B=�C       �	��`Yc�A�1*

loss�� =�7I       �	=*aYc�A�1*

loss�9�<f}j       �	��aYc�A�1*

loss.�=��`�       �	XbYc�A�1*

loss��+<'��       �	1�bYc�A�1*

lossF�
<'<�%       �	M�cYc�A�1*

loss�#�<�U��       �	�dYc�A�1*

loss��u:�
       �	ɮdYc�A�1*

lossZba<:_<�       �	�oeYc�A�1*

loss�T;� ��       �	 fYc�A�1*

loss��;�2�       �	�fYc�A�1*

loss�-=ʟ��       �	�0gYc�A�1*

lossV�o=^��I       �	��gYc�A�1*

lossۦ(<����       �	OWhYc�A�1*

loss�iE;���       �	��hYc�A�1*

lossȱ�:��@       �	'�iYc�A�1*

loss�R=��       �	7jYc�A�1*

loss��d;3/�       �	!�jYc�A�1*

loss��<=b�?u       �	�kYc�A�1*

loss���<���M       �	HlYc�A�1*

loss;9a�c       �	��lYc�A�1*

loss�<���       �	�umYc�A�1*

loss�
�<9�.       �	�2nYc�A�1*

loss%J<��2       �	G pYc�A�1*

loss�?S:�/�!       �	�pYc�A�1*

loss�;���x       �	�dqYc�A�1*

lossxH]<�E�       �	W?rYc�A�1*

loss�Н;ҢV       �	��rYc�A�1*

lossW��;*�|�       �	�xsYc�A�1*

loss(|�;ވE�       �	0tYc�A�1*

loss�4�:uS       �	j�tYc�A�1*

lossR��:�Z�       �	"vYc�A�1*

loss$F<*x       �	��vYc�A�1*

loss1�j;�B��       �	�wYc�A�1*

loss���<066[       �	S!xYc�A�1*

lossQA�: �       �	��xYc�A�1*

lossS��;��q8       �	�yYc�A�1*

lossidP=��Rr       �	}vzYc�A�1*

lossG�<�@C�       �	�F{Yc�A�1*

loss��Z<}^Q       �	|Yc�A�1*

loss��d=Y�P@       �	ؼ|Yc�A�1*

loss�#�;�N6�       �	�b}Yc�A�1*

loss\��;��!�       �	�%~Yc�A�1*

loss3+�;�zx�       �	��~Yc�A�1*

lossj�`==�,       �	~Yc�A�1*

loss%E;��       �	�$�Yc�A�1*

lossEΆ;֤��       �	`�Yc�A�1*

lossF8;T�x       �	6��Yc�A�1*

loss�3<��Y       �	r7�Yc�A�1*

loss�LG;�M�       �	�ЂYc�A�1*

loss|7�=
ذ�       �	�m�Yc�A�1*

lossSZ�<�զW       �	��Yc�A�1*

loss���;�q�1       �	n��Yc�A�1*

loss>6=OvE       �	-B�Yc�A�1*

loss�Є;�_j�       �	1�Yc�A�1*

loss`i�=ZN�!       �	K��Yc�A�1*

losss�U=�lӨ       �	_'�Yc�A�1*

loss>h<3i�        �	�ׇYc�A�1*

loss�;�Ո       �	:z�Yc�A�1*

lossA�<=&] m       �	0�Yc�A�1*

loss�\�;�i�       �	9��Yc�A�1*

loss���<l�       �	�V�Yc�A�1*

loss�<&��       �	���Yc�A�1*

lossO�;4�.K       �	HߋYc�A�1*

loss���:(��Q       �	+��Yc�A�1*

loss��;6�       �	�G�Yc�A�1*

loss�/"=;��       �	lB�Yc�A�1*

loss:[=����       �	��Yc�A�1*

lossZ��;��=       �	�+�Yc�A�1*

loss�V<�S&       �	�"�Yc�A�1*

loss��h;�|�       �	�V�Yc�A�1*

loss"�=�jj�       �	��Yc�A�1*

loss-_�;W�`s       �	p��Yc�A�1*

lossr�<A�mM       �	ٖ�Yc�A�2*

loss�x=<k�Q       �	�o�Yc�A�2*

loss���;ŕ��       �	��Yc�A�2*

loss���<NC��       �	[ϖYc�A�2*

lossω:P�	y       �	�i�Yc�A�2*

loss�"=����       �	:�Yc�A�2*

loss�[<=��!       �	㧘Yc�A�2*

loss)�;x       �	xC�Yc�A�2*

lossa�<b��       �	EיYc�A�2*

lossh��<���       �	al�Yc�A�2*

loss}P�;�K�       �	� �Yc�A�2*

lossC=4��       �	pΛYc�A�2*

loss���<��Z       �	d�Yc�A�2*

loss|Uo<iv@�       �	���Yc�A�2*

lossnr4;����       �	6��Yc�A�2*

loss��<�=@       �	9E�Yc�A�2*

lossf.�<P�$       �	�ٞYc�A�2*

loss8��:�X�       �	��Yc�A�2*

loss&�;C�:�       �	�Yc�A�2*

loss2�<��yh       �	���Yc�A�2*

loss�{�<�%�:       �	ND�Yc�A�2*

lossi/�;�i�       �	��Yc�A�2*

lossd�O=��GN       �	��Yc�A�2*

loss2�*:�k�       �	AG�Yc�A�2*

loss��v=���K       �	�ܣYc�A�2*

loss��{;Q8�       �		��Yc�A�2*

loss���<1OpR       �	��Yc�A�2*

loss�D;K��|       �	4��Yc�A�2*

lossj��<>���       �	h\�Yc�A�2*

loss[ذ<@	       �	���Yc�A�2*

loss8�<<�)~H       �	 ��Yc�A�2*

loss�Sh=>M�`       �	-$�Yc�A�2*

lossȗ�;�I#       �	qȨYc�A�2*

lossR4<���x       �	[a�Yc�A�2*

loss�<f��       �	��Yc�A�2*

loss�q<�;7�       �	��Yc�A�2*

loss8��<��D       �	8��Yc�A�2*

loss��;�C'V       �	�F�Yc�A�2*

loss1)<�914       �	�ܬYc�A�2*

lossZ`�;���       �	���Yc�A�2*

lossv�b;s�       �	N)�Yc�A�2*

loss<=���       �	�ήYc�A�2*

loss�3%<�w)�       �	us�Yc�A�2*

lossۜ�<��ؾ       �	��Yc�A�2*

loss�v<�{t       �	mȰYc�A�2*

loss���;�Aq�       �	�l�Yc�A�2*

loss� �=�_=       �	��Yc�A�2*

loss�<E=�͍       �	L�Yc�A�2*

lossS�<S/,       �	�ĳYc�A�2*

loss;��;&���       �	��Yc�A�2*

loss���<O�       �	���Yc�A�2*

loss��f="0�       �	:��Yc�A�2*

loss�f�:(��       �	�,�Yc�A�2*

lossN�:O-\�       �	�÷Yc�A�2*

lossW,<�,�       �	�f�Yc�A�2*

loss���<��=�       �	��Yc�A�2*

lossJ�=�fM+       �	a��Yc�A�2*

loss��;����       �	�l�Yc�A�2*

loss3�f<��5       �	�Yc�A�2*

loss}R�;�r�       �	%��Yc�A�2*

loss���<2�2       �	�T�Yc�A�2*

loss�c�<�9�       �	���Yc�A�2*

loss��: <�5       �	ҏ�Yc�A�2*

loss��;����       �	cB�Yc�A�2*

lossl�:=�F       �	0�Yc�A�2*

loss�a�9��.�       �	̙�Yc�A�2*

loss�0}<��s�       �	�;�Yc�A�2*

loss:5M;�M�       �	���Yc�A�2*

lossZ�3;4h��       �	�f�Yc�A�2*

lossz><�G�W       �	;��Yc�A�2*

loss*Z]<��       �	���Yc�A�2*

loss��<� ��       �	�)�Yc�A�2*

lossw,�;����       �	@��Yc�A�2*

loss���:�y       �	Y�Yc�A�2*

loss_@i=���*       �	��Yc�A�2*

lossc�;z��q       �	h��Yc�A�2*

loss�߻:��y{       �	�1�Yc�A�2*

lossn��:�-       �	���Yc�A�2*

loss4�2=� 3�       �	���Yc�A�2*

loss=P.;�d�7       �	#2�Yc�A�2*

loss�u<|7�5       �	q��Yc�A�2*

loss��=2m+C       �	jl�Yc�A�2*

loss[�;�V�A       �	��Yc�A�2*

losse�;�zX�       �	���Yc�A�2*

lossbi=Z~��       �	�N�Yc�A�2*

losstF;���+       �	!��Yc�A�2*

loss��;����       �	���Yc�A�2*

loss��O<4�       �	�H�Yc�A�2*

loss�.�=����       �	7��Yc�A�2*

loss3��;H7��       �	-��Yc�A�2*

loss�g)<�/�w       �	I+�Yc�A�2*

lossw�y=��p       �	@��Yc�A�2*

loss���<R���       �	�\�Yc�A�2*

lossF>0<�FA       �	��Yc�A�2*

loss�2=�E�T       �	(+�Yc�A�2*

loss���8��b       �	R��Yc�A�2*

lossc4�<w��       �	p��Yc�A�2*

loss�\6=U��       �	z5�Yc�A�2*

loss��;��:�       �	��Yc�A�2*

lossoj<@"��       �	�q�Yc�A�2*

loss��<�;cZ       �	��Yc�A�2*

lossa\<��       �	��Yc�A�2*

loss�E�:��SO       �	�z�Yc�A�2*

loss%-<��N.       �	�+�Yc�A�2*

lossC�P;����       �	���Yc�A�2*

lossW5\;����       �	3l�Yc�A�2*

loss��3;��Q�       �	�.�Yc�A�2*

lossw(X< �1       �	���Yc�A�2*

loss� �<���       �	�\�Yc�A�2*

loss���;��+       �	���Yc�A�2*

loss���<�6s       �	@��Yc�A�2*

loss�^�;�jAU       �	� �Yc�A�2*

loss	�<FV�1       �	��Yc�A�2*

lossJ)%;��y       �	/i�Yc�A�2*

loss���=�&�       �	��Yc�A�2*

loss�bW<t���       �	���Yc�A�2*

lossQ�M=��4v       �	U�Yc�A�2*

loss�2�<x�       �	V��Yc�A�2*

loss��\;8q��       �		��Yc�A�2*

loss�xm9�E��       �	p$�Yc�A�2*

loss��;�ӽ2       �	˹�Yc�A�2*

loss��,;"�       �	aR�Yc�A�2*

lossO�<���i       �	I��Yc�A�2*

loss �s;/O��       �	���Yc�A�2*

lossJ��<����       �	n�Yc�A�2*

loss��:r��       �	���Yc�A�2*

loss��9�@�A       �	�v�Yc�A�2*

loss�
.:J|2       �	��Yc�A�2*

loss1aC;� j{       �	R��Yc�A�3*

loss��U=-�>�       �	�R�Yc�A�3*

lossr��::�¤       �	���Yc�A�3*

lossN�=� ��       �	y�Yc�A�3*

loss\�o;9;�       �	/�Yc�A�3*

lossݍ<�g�        �	���Yc�A�3*

loss��;x��       �	q<�Yc�A�3*

lossn��;��1       �	"��Yc�A�3*

lossh�#;���       �	���Yc�A�3*

loss���;���       �	��Yc�A�3*

loss�[T;<�3�       �	�Yc�A�3*

loss4}�;_��^       �	���Yc�A�3*

loss�+;� b?       �	bH�Yc�A�3*

losswJ�7�1s�       �	�m�Yc�A�3*

loss�J<9���       �	��Yc�A�3*

lossī:O�V       �	���Yc�A�3*

loss��'<3�*       �	�>�Yc�A�3*

loss�ʴ<�|\       �	���Yc�A�3*

loss�;e:       �	Z��Yc�A�3*

lossr��:у:�       �	�`�Yc�A�3*

lossa�<���8       �	%�Yc�A�3*

loss8��=����       �	���Yc�A�3*

loss�o�9�_��       �	Ύ�Yc�A�3*

loss�O�=xTH       �	A*�Yc�A�3*

loss��<V�1       �	���Yc�A�3*

loss�y!;ѧ�       �	f�Yc�A�3*

losse�;;���       �	���Yc�A�3*

loss\�=:c���       �	z��Yc�A�3*

lossIv�=���       �	 A�Yc�A�3*

losswP�<l��       �	��Yc�A�3*

loss�O<刪�       �	��Yc�A�3*

loss �[<��_       �	LU�Yc�A�3*

loss�&<OO�g       �	���Yc�A�3*

lossX	�;$vE#       �	���Yc�A�3*

lossO1==�l�j       �	x& Yc�A�3*

loss��<�!��       �	�� Yc�A�3*

loss*��<�hY�       �	�~Yc�A�3*

loss��;Y��"       �	�)Yc�A�3*

loss�U�=�i�8       �	9�Yc�A�3*

loss�d=�cp�       �	�yYc�A�3*

loss%�b=�kWA       �	�!Yc�A�3*

lossܗ�<!Eʁ       �	��Yc�A�3*

loss��:1��       �	�kYc�A�3*

loss�$=�y i       �	�Yc�A�3*

loss��<y8�Z       �	ΥYc�A�3*

loss�c*:Z/3_       �	�AYc�A�3*

loss���;�%��       �	�Yc�A�3*

lossѼ[;&l�L       �	�s	Yc�A�3*

loss��
=�e�       �	�
Yc�A�3*

loss͛�;���t       �	��
Yc�A�3*

loss���:K��       �	(HYc�A�3*

loss�\�<�z�e       �	e�Yc�A�3*

loss;`<"&M       �	��Yc�A�3*

loss�F�;׫o�       �	��Yc�A�3*

loss��<����       �	�Yc�A�3*

loss|�=<@E�       �	��Yc�A�3*

loss�'	=�m�v       �	��Yc�A�3*

loss���;����       �	 Yc�A�3*

loss*��<��       �	��Yc�A�3*

loss��<�*BC       �	�yYc�A�3*

loss7_�<+       �	;9Yc�A�3*

loss`Oc<1�l�       �	y�Yc�A�3*

loss�s
<�^c�       �	�%Yc�A�3*

loss��/<��l       �	3�Yc�A�3*

loss��<�7��       �	dyYc�A�3*

loss��`<�/>U       �	�QYc�A�3*

loss�<S$�       �	�Yc�A�3*

loss���<~FQ�       �	��Yc�A�3*

loss5=(a�J       �	��Yc�A�3*

loss���:�9��       �	�QYc�A�3*

loss��U<���S       �	��Yc�A�3*

loss��K;h$s�       �	6�Yc�A�3*

loss|��;Ң��       �	��Yc�A�3*

loss��<W�^�       �	�5Yc�A�3*

loss��3=G�ن       �	=6Yc�A�3*

loss��=�	��       �	6�6Yc�A�3*

loss�J;B�M       �	�8Yc�A�3*

loss�c�<7@��       �	��8Yc�A�3*

loss��0<�X%       �	��9Yc�A�3*

loss�/;
P�       �	A*:Yc�A�3*

loss��=�b��       �	��:Yc�A�3*

loss|%<IA�       �	�\<Yc�A�3*

lossԳ;i�e�       �	\=Yc�A�3*

lossa�X;�>�       �	a�=Yc�A�3*

loss�Y<È�S       �	5A>Yc�A�3*

lossA!p;�뭈       �	�>Yc�A�3*

loss��<$<�2       �	�?Yc�A�3*

lossF�<���       �	B[@Yc�A�3*

lossi��=pX��       �	|�@Yc�A�3*

loss�� ;�a��       �	��AYc�A�3*

loss��,;v�<y       �	�ABYc�A�3*

loss�W�9��TQ       �	��BYc�A�3*

lossfh�<�P�       �	NzCYc�A�3*

loss��`;r��v       �	}�DYc�A�3*

loss�ce=���f       �	dEYc�A�3*

loss!�:�Ą7       �	*FYc�A�3*

loss��y;���M       �	��FYc�A�3*

loss��z;D�F�       �	Q2GYc�A�3*

loss!��;����       �	!�GYc�A�3*

lossŅ$<M�Gw       �	nHYc�A�3*

loss�j.=Л#&       �	MIYc�A�3*

lossd	=�x��       �	��IYc�A�3*

loss;�:��XT       �	jPJYc�A�3*

loss! v;̨&       �	D�JYc�A�3*

loss�f{<�{u�       �	��KYc�A�3*

losst;�;0d\       �	$'LYc�A�3*

loss)|�<ƠZ"       �	��LYc�A�3*

loss��;T�=�       �	gMYc�A�3*

loss��;^SI�       �	_NYc�A�3*

loss��O9���%       �	�?OYc�A�3*

loss�#=A��1       �	�vPYc�A�3*

lossW��<��f�       �	=cQYc�A�3*

loss���;��       �	�RYc�A�3*

loss3�.:�v�       �	��RYc�A�3*

lossI�3<+;��       �	+OSYc�A�3*

loss=��:�q��       �	�UYc�A�3*

loss�t4<٭$0       �	
�UYc�A�3*

loss���:v;��       �	��VYc�A�3*

lossIa<���       �	�WYc�A�3*

losse��<�`�~       �	�'XYc�A�3*

loss-x&;�A.�       �	��XYc�A�3*

losslH�:K��/       �	gYYc�A�3*

loss�;>;C���       �	ZYc�A�3*

loss3�:�Ԭ�       �	C�ZYc�A�3*

loss:��;�B�       �	�N[Yc�A�3*

loss��=�pC       �	��[Yc�A�3*

loss��<���       �	��\Yc�A�3*

loss���<#��o       �	�6]Yc�A�3*

loss�oV:��\       �	t�]Yc�A�4*

lossj��9��2�       �	t^Yc�A�4*

loss.,l95��       �	_Yc�A�4*

loss��d:����       �	:�_Yc�A�4*

lossHW=^��A       �	;o`Yc�A�4*

loss��]=:�       �	�aYc�A�4*

loss�:�m       �	�aYc�A�4*

loss��%;5A�=       �	UPbYc�A�4*

loss$�~;`��       �	J�bYc�A�4*

loss:��<�<6K       �	U�cYc�A�4*

loss��:�lG       �	�9dYc�A�4*

lossF$c:����       �	c�dYc�A�4*

loss<��<�-!       �	vpeYc�A�4*

lossD*�=K=@       �	YfYc�A�4*

loss4�w;�&�       �	'�fYc�A�4*

loss�:]ߓ       �	{fgYc�A�4*

loss�T-<y���       �	�hYc�A�4*

loss��<�4^�       �	��hYc�A�4*

loss�}2;'�S       �	EhiYc�A�4*

lossiLh='F�O       �	�jYc�A�4*

loss��T;8a�r       �	ؼjYc�A�4*

loss�ҭ;ܘ��       �	�kYc�A�4*

loss
�;ܴ^�       �	bflYc�A�4*

loss��#<�ؔ       �	�mYc�A�4*

losst+�<�#��       �	
�mYc�A�4*

loss\B=R3nu       �	�snYc�A�4*

loss�(k<���{       �	�'oYc�A�4*

loss?;�;�]�~       �	,�oYc�A�4*

loss���:��d�       �	{�pYc�A�4*

loss7�:�L��       �	�$qYc�A�4*

loss2�:���Y       �	��qYc�A�4*

loss��J<��       �	�ZrYc�A�4*

loss\z�<PI�       �	�rYc�A�4*

loss2|�<���       �	ȘsYc�A�4*

loss�R:;v�84       �	*6tYc�A�4*

loss���;`���       �	OYuYc�A�4*

lossa��;!�i�       �	��uYc�A�4*

lossZ�B;L�       �	#�vYc�A�4*

loss���:�H�8       �	�JwYc�A�4*

loss�><���       �	�wYc�A�4*

loss���9�P�$       �	��xYc�A�4*

loss}5<x��       �	�1yYc�A�4*

loss́�;�3�p       �	Q�yYc�A�4*

lossa��=�I�       �	��zYc�A�4*

lossR��;�f��       �	�-{Yc�A�4*

lossϖ�<H��       �	��{Yc�A�4*

loss(��8���       �	i�|Yc�A�4*

lossZ��:��N       �	�,}Yc�A�4*

loss��	<��U4       �	��}Yc�A�4*

loss	��:�s�       �	)�~Yc�A�4*

loss,p;�I�       �	�bYc�A�4*

lossAx�;~*��       �	k�Yc�A�4*

loss
;��       �	��Yc�A�4*

loss!3�<��r�       �	T7�Yc�A�4*

loss�7";�!�       �	��Yc�A�4*

loss�$?;f���       �	�ÂYc�A�4*

loss�;T��W       �	�n�Yc�A�4*

loss�o<S��       �	�Yc�A�4*

losshG9�oy-       �	��Yc�A�4*

loss-��<���       �	�`�Yc�A�4*

loss ;�⽋       �	��Yc�A�4*

loss�B;{�~b       �	z��Yc�A�4*

loss�=Pv8�       �	���Yc�A�4*

loss��=�d1�       �	��Yc�A�4*

loss�)e9�ƾ�       �	~ÈYc�A�4*

loss��;E���       �	\�Yc�A�4*

loss�X<���       �	P��Yc�A�4*

lossWs�:R�ѫ       �	
��Yc�A�4*

loss`�!=n�G       �	�9�Yc�A�4*

lossQ��:f�-�       �	=ԋYc�A�4*

lossҚ0;h?H�       �	��Yc�A�4*

loss���:ǋ��       �	J$�Yc�A�4*

loss.�':ղ�        �	���Yc�A�4*

loss���<�>�       �	1��Yc�A�4*

loss�M�;��w�       �	�1�Yc�A�4*

loss�9����       �	`ɏYc�A�4*

loss�S<�m�       �	�g�Yc�A�4*

loss`��;�9       �	���Yc�A�4*

loss��;����       �	Ӡ�Yc�A�4*

loss1�:�I+�       �	!?�Yc�A�4*

loss���:<P�       �	,֒Yc�A�4*

lossX�:P��       �	[z�Yc�A�4*

loss֡@<�P��       �	2U�Yc�A�4*

loss@m�:���(       �	R�Yc�A�4*

loss��h:��0       �	Ό�Yc�A�4*

loss��:��       �	R)�Yc�A�4*

loss��Y;wT       �	�ΖYc�A�4*

loss-P�<)�ם       �	�j�Yc�A�4*

loss-��;��[Q       �	��Yc�A�4*

loss���;#��K       �	[��Yc�A�4*

loss<��;���       �	dv�Yc�A�4*

lossS��9(�       �	m�Yc�A�4*

loss�J�:kz"       �	�ȚYc�A�4*

loss�A`;.g�       �	_a�Yc�A�4*

lossRzD=��=j       �	�Yc�A�4*

loss���;:�       �	d��Yc�A�4*

loss�K�;5z��       �	�L�Yc�A�4*

loss��<=�"       �	��Yc�A�4*

loss�1<r5H       �	Á�Yc�A�4*

loss�wi:ѽ�H       �	u�Yc�A�4*

loss?��:4	�       �	���Yc�A�4*

loss��89�3˕       �	�Z�Yc�A�4*

lossܾ�<�} �       �	�,�Yc�A�4*

loss��q<k7v�       �	qơYc�A�4*

loss�Is<�*n*       �	�c�Yc�A�4*

lossQ�=�i�       �	:�Yc�A�4*

loss��O=r<�1       �	��Yc�A�4*

loss�K�<��
       �	�6�Yc�A�4*

loss���9;\�       �	�ԤYc�A�4*

lossh��;�yK       �	�m�Yc�A�4*

loss�<�خ-       �	M�Yc�A�4*

loss��:;1        �	.�Yc�A�4*

loss���<h��       �	�|�Yc�A�4*

loss,�:�1��       �	��Yc�A�4*

loss��e:�uow       �	N��Yc�A�4*

loss���;��7       �	EK�Yc�A�4*

loss���;L'�       �	�(�Yc�A�4*

loss\��;)��       �	���Yc�A�4*

lossi�;m�       �	�T�Yc�A�4*

loss�v<uƋ�       �	���Yc�A�4*

lossqW =o�@�       �	h��Yc�A�4*

loss���; e��       �	�3�Yc�A�4*

lossz�\<�jfJ       �	�ϭYc�A�4*

loss�=�<�&       �	.q�Yc�A�4*

loss�{�9 ���       �	
H�Yc�A�4*

loss��:9'�       �	jݯYc�A�4*

losso�[=b��       �	ds�Yc�A�4*

loss��;G�       �	��Yc�A�4*

loss�Tu;�p�       �	��Yc�A�5*

loss�H=�e�       �	າYc�A�5*

loss��=��       �	 R�Yc�A�5*

loss�Qw<��!D       �	��Yc�A�5*

loss��<�jv       �	���Yc�A�5*

lossH��:{�@       �	?U�Yc�A�5*

loss�13=�!c       �	P��Yc�A�5*

loss�<��       �	��Yc�A�5*

lossƍ�<��\       �	b��Yc�A�5*

lossc��;�z@$       �	q;�Yc�A�5*

loss_t0< V�       �	ӸYc�A�5*

loss�29G��i       �	���Yc�A�5*

lossT�Y;ϱ.       �	�`�Yc�A�5*

loss�h:�^�L       �	���Yc�A�5*

lossf��<���       �	Ӣ�Yc�A�5*

loss$��;�N9�       �	�h�Yc�A�5*

loss�(<ǼD       �	�Yc�A�5*

lossڲu<~��       �	q��Yc�A�5*

loss�|9ҷ|T       �	�B�Yc�A�5*

loss���:i��*       �	���Yc�A�5*

loss�]�<�kR       �	���Yc�A�5*

loss���<@-��       �	�*�Yc�A�5*

lossz��;D��       �	���Yc�A�5*

lossￎ=e?.�       �	~W�Yc�A�5*

losss�=�Ӌ�       �	���Yc�A�5*

loss
0�:�!��       �	Y��Yc�A�5*

losseʗ=��D�       �	&�Yc�A�5*

loss��<Uٖ       �	a��Yc�A�5*

lossfw;k8�       �	]�Yc�A�5*

loss�B�;����       �	���Yc�A�5*

loss6bZ;��       �	a��Yc�A�5*

loss���;a&0       �	�'�Yc�A�5*

loss�<׆�8       �	��Yc�A�5*

loss��=���       �	�W�Yc�A�5*

loss�`?<ڛ|H       �	���Yc�A�5*

loss= x<B��N       �	+��Yc�A�5*

loss ��<Vi1_       �	��Yc�A�5*

loss�V7="QT�       �	%��Yc�A�5*

loss���:}�4_       �	�L�Yc�A�5*

loss]�<��       �	���Yc�A�5*

loss:cG=|�9�       �	��Yc�A�5*

lossI�3;���+       �	j�Yc�A�5*

loss�d�;F�e�       �	L��Yc�A�5*

loss�8�<���       �	��Yc�A�5*

loss���;��+t       �	e9�Yc�A�5*

lossx�2<��,       �	��Yc�A�5*

loss�,�;3��       �	T��Yc�A�5*

loss�A�;�i��       �	0/�Yc�A�5*

loss���;���O       �	��Yc�A�5*

lossR��;�w��       �	Ow�Yc�A�5*

loss ~�<k���       �	f�Yc�A�5*

loss���;��_       �	���Yc�A�5*

loss(��;Aiz       �	I�Yc�A�5*

loss]@=�^y       �	U��Yc�A�5*

loss�;����       �	���Yc�A�5*

loss��;F�b�       �	�^�Yc�A�5*

lossmw/<̍�       �	��Yc�A�5*

lossh�=��3�       �	���Yc�A�5*

lossԍ�;3�b       �	�"�Yc�A�5*

lossG֙;�U�       �	��Yc�A�5*

loss��=�m�       �	O�Yc�A�5*

loss&�Q=�i�J       �	���Yc�A�5*

loss.��:R�J�       �	s��Yc�A�5*

loss��
;ǥG@       �	��Yc�A�5*

loss�c;����       �	?��Yc�A�5*

loss	��<��;�       �	!x�Yc�A�5*

lossw3�<v��       �	 �Yc�A�5*

loss_B<^=�       �	@��Yc�A�5*

lossF��=��ǭ       �	�C�Yc�A�5*

loss9=�g       �	
��Yc�A�5*

lossZ*�:]9��       �	�o�Yc�A�5*

loss���:�] &       �	)�Yc�A�5*

loss��&=%z<       �	��Yc�A�5*

loss��);�k�H       �	C9�Yc�A�5*

lossc.�<�If       �	���Yc�A�5*

lossNS�;�ݓ�       �	wd�Yc�A�5*

loss�= <j���       �	T��Yc�A�5*

lossfk�:	��       �	�S�Yc�A�5*

loss�T�:U�/       �	���Yc�A�5*

lossif <�#@       �	��Yc�A�5*

lossƵ�;LG�       �	�2�Yc�A�5*

losszR�;*c       �	��Yc�A�5*

lossS��;Ui�       �	��Yc�A�5*

lossڼ�<��:�       �	�E�Yc�A�5*

lossH��<�jA       �	/��Yc�A�5*

loss�x(=��l       �	`u�Yc�A�5*

loss:��;���5       �	g�Yc�A�5*

lossj�`=c�U       �	���Yc�A�5*

loss�';,y�       �	�F�Yc�A�5*

loss�<���       �	f��Yc�A�5*

loss
��<�'ަ       �	x�Yc�A�5*

loss:=W"U�       �	���Yc�A�5*

loss҂�<'��       �	Z�Yc�A�5*

loss*�;V2��       �	���Yc�A�5*

loss���8�
)       �	���Yc�A�5*

loss�H?;6�f�       �	�0�Yc�A�5*

losso��<Ռ�=       �	B
�Yc�A�5*

lossq��;��       �	���Yc�A�5*

loss�|�;�-_       �	���Yc�A�5*

losssAB:�ԙ       �	��Yc�A�5*

lossT��;��       �	���Yc�A�5*

loss =�;�in�       �	��Yc�A�5*

lossU�=>�       �	�V�Yc�A�5*

loss��<���       �	K �Yc�A�5*

loss �:���       �	Ԝ�Yc�A�5*

loss� �:+��       �	r7�Yc�A�5*

loss�u:�G�       �	o��Yc�A�5*

loss/�q:O5�       �	؟�Yc�A�5*

loss�>�;�'5�       �	E�Yc�A�5*

lossq��;��إ       �	O�Yc�A�5*

loss˝:
U�q       �	���Yc�A�5*

losstb�;?e�z       �	@��Yc�A�5*

lossҟ�;3>�o       �	E�Yc�A�5*

loss���;T{�n       �	���Yc�A�5*

loss�%<�       �	��Yc�A�5*

loss�#N<N�S       �	��Yc�A�5*

lossĘ<��)�       �	;��Yc�A�5*

loss�<gǻ�       �	�b�Yc�A�5*

loss��`<����       �	8 Yc�A�5*

lossh�;t>��       �	H� Yc�A�5*

loss���;���       �	�@Yc�A�5*

losso��;$        �	"�Yc�A�5*

loss��,=w6��       �	ۢYc�A�5*

loss$��:S�x       �	*;Yc�A�5*

loss�Z<z���       �	Z�Yc�A�5*

loss�Ց<2�+�       �	�iYc�A�5*

losstК;�k`�       �	�0Yc�A�5*

loss#�=(���       �	��Yc�A�5*

loss�K<�'e       �	c^Yc�A�6*

loss2�=�ǁ&       �	��Yc�A�6*

lossD�;��U       �	F�Yc�A�6*

loss�:#�U       �	�3Yc�A�6*

loss�	=�y�       �	l	Yc�A�6*

loss��:�=�A       �	R�	Yc�A�6*

loss� �<F�7�       �	�7
Yc�A�6*

loss$��;�3u�       �	B�
Yc�A�6*

lossR�<���       �	�|Yc�A�6*

loss:��<�J;       �	sYc�A�6*

lossOy,<�.�*       �	��Yc�A�6*

loss.�;�~�g       �	ՔYc�A�6*

loss�:)�3       �	Z�Yc�A�6*

loss{3�;B�       �	�AYc�A�6*

loss�x�<A	o       �	��Yc�A�6*

loss���;�� y       �	�uYc�A�6*

loss
�U=b�=       �	�2Yc�A�6*

loss*-�::``6       �	`�Yc�A�6*

lossE��<�w%       �	��Yc�A�6*

lossĺ:d
X       �	��Yc�A�6*

loss�&�:�4�       �	\Yc�A�6*

lossxnG;�8�       �	8Yc�A�6*

loss�H5<�!h       �	8�Yc�A�6*

loss��;�'H       �	4�Yc�A�6*

lossӈ�<�f<T       �	tAYc�A�6*

loss2�C<���       �	0�Yc�A�6*

loss��<��sC       �	ٲYc�A�6*

lossZ�2<�y�       �	P�Yc�A�6*

lossJ�:5{V�       �	oEYc�A�6*

loss�;?9�Bb�       �	�Yc�A�6*

lossLGg:*ou�       �	�Yc�A�6*

lossM5�;*攐       �	e�Yc�A�6*

loss���:��JZ       �	�SYc�A�6*

loss$�u=�(:4       �	�Yc�A�6*

loss��9=y��       �	:�Yc�A�6*

lossT"<6��b       �	�jYc�A�6*

loss-0�;#       �	��Yc�A�6*

loss��[=�SW�       �	$� Yc�A�6*

loss%�t;�       �	nl!Yc�A�6*

loss�^�;��       �	"Yc�A�6*

loss�D<y�       �	E�"Yc�A�6*

loss�(E;�l�p       �	n2#Yc�A�6*

loss�F
=I�"h       �	��#Yc�A�6*

loss�a�<����       �	��$Yc�A�6*

loss]��<1�/       �	�*%Yc�A�6*

loss�]R;��J�       �	X�%Yc�A�6*

loss�$=<��S�       �	׊&Yc�A�6*

lossh�F<��P�       �	`"'Yc�A�6*

loss�<��.[       �	Q�'Yc�A�6*

loss���<���       �	��(Yc�A�6*

loss��<\�4�       �	�o)Yc�A�6*

loss@��<˜;�       �	�Z*Yc�A�6*

loss��>.��       �	"+Yc�A�6*

loss�=� .�       �	׾+Yc�A�6*

loss�=~��       �	2U,Yc�A�6*

loss�y�<�I��       �	�-Yc�A�6*

losseZ<J��@       �	�-Yc�A�6*

loss��b:�#�I       �	�>.Yc�A�6*

loss��;����       �	J�.Yc�A�6*

loss}��:�	��       �	�q/Yc�A�6*

loss4Z;P��\       �	M0Yc�A�6*

lossN�<���y       �	��0Yc�A�6*

lossW��:�[r       �	Ӆ1Yc�A�6*

loss���;���       �	�2Yc�A�6*

loss_2X=�+,d       �	@�2Yc�A�6*

loss��!<�       �	��3Yc�A�6*

loss�?|;�%a7       �	�+4Yc�A�6*

loss=�'=�A�       �	�75Yc�A�6*

lossn�;�`�R       �	|�5Yc�A�6*

loss!��<)�'�       �	#�6Yc�A�6*

loss��;
 ��       �	 |7Yc�A�6*

loss!"�;�B4�       �	�B8Yc�A�6*

loss��6;ez4K       �	��8Yc�A�6*

loss�AY9$��       �	/�9Yc�A�6*

loss %;B+�p       �	y#:Yc�A�6*

lossAS�<k�6       �	ܽ:Yc�A�6*

loss�=���       �	�;Yc�A�6*

lossS��<��N�       �	d@<Yc�A�6*

loss� �9{{��       �	(�<Yc�A�6*

loss>�;��ǽ       �	�p=Yc�A�6*

loss���:�"?�       �	>Yc�A�6*

lossX�f<g0�B       �	��>Yc�A�6*

lossR4B<�ڭk       �	�G?Yc�A�6*

loss�e�;��       �	��?Yc�A�6*

lossf�M<vH�       �	��@Yc�A�6*

loss�)�9|O�       �	HAYc�A�6*

loss�M�:��w       �	q�AYc�A�6*

loss�;��       �	�bBYc�A�6*

losscu�;��y�       �	CYc�A�6*

loss�w;?p�*       �	��CYc�A�6*

loss�6*<�Ά       �	�:DYc�A�6*

lossS>=��8V       �	�DYc�A�6*

loss#��;k	O�       �	�gEYc�A�6*

loss6a�<onx�       �	�FYc�A�6*

loss�Q<˰h~       �	�FYc�A�6*

loss�<}F�}       �	g_GYc�A�6*

loss��;��mu       �	��GYc�A�6*

lossH��;ŵ�i       �	�HYc�A�6*

lossz�<��\       �	�1IYc�A�6*

lossԿ<��*P       �	��IYc�A�6*

lossT�_=)�Y       �	�fJYc�A�6*

loss;�;�       �	��KYc�A�6*

loss#�<�٧a       �	�LYc�A�6*

lossl�;Z�{       �	�wMYc�A�6*

loss�(;���       �	�NYc�A�6*

lossư�<6�       �	� OYc�A�6*

lossC��;����       �	z�OYc�A�6*

lossLI�<����       �	�hPYc�A�6*

loss��=��       �	�-QYc�A�6*

loss-;5j�       �	I�QYc�A�6*

loss*��<YNcp       �	:�RYc�A�6*

loss%;� �D       �	�ISYc�A�6*

loss��<P�Ȁ       �	��SYc�A�6*

loss�00<�fI       �	��TYc�A�6*

lossE�}=:�=q       �	�%UYc�A�6*

loss���<�U��       �	��UYc�A�6*

loss�r=�|=�       �	�cVYc�A�6*

loss\�= ���       �	~WYc�A�6*

loss�o:Aۿ       �	c�WYc�A�6*

loss�a<c�K       �	�,XYc�A�6*

loss7��;n�3�       �	�$YYc�A�6*

loss��L;���       �	E�YYc�A�6*

lossdh<� R�       �	+QZYc�A�6*

loss��M<=���       �	t�ZYc�A�6*

losss[+;3x�q       �	�[Yc�A�6*

loss���<,O�H       �	z\Yc�A�6*

loss�W;�r��       �	�]Yc�A�6*

loss�S<o:w�       �	�]Yc�A�6*

loss�;����       �	GY^Yc�A�7*

loss$��;��@�       �	��^Yc�A�7*

lossa�`<nJ�       �	��_Yc�A�7*

loss�ut<~��       �	�7`Yc�A�7*

loss�9�<��e       �	��`Yc�A�7*

loss��;|�9�       �	�kaYc�A�7*

loss�e�<�n��       �	WbYc�A�7*

loss*�=fϓ�       �	��bYc�A�7*

loss* N;���\       �	e4cYc�A�7*

loss�<~S��       �	��cYc�A�7*

loss2C�;�Y�       �	vdYc�A�7*

loss�S�<���d       �	xeYc�A�7*

loss��;Yt(�       �	3�eYc�A�7*

loss'��9N��*       �	l>fYc�A�7*

lossV7�;gc�)       �	�fYc�A�7*

loss��;���'       �	�ogYc�A�7*

loss�C=�XO       �	�hYc�A�7*

loss�=8�H       �	&�hYc�A�7*

loss��;*Ł,       �	�CiYc�A�7*

lossя<m2��       �	��iYc�A�7*

lossD, =B=��       �	6tjYc�A�7*

loss��6<6\�       �	�	kYc�A�7*

loss��E;R�^{       �	��kYc�A�7*

loss�#:
�c       �	�;lYc�A�7*

loss[�=��4+       �	OmYc�A�7*

lossf��;v�M�       �	b�mYc�A�7*

loss�;R�_-       �	�5nYc�A�7*

loss*-<�c��       �	��nYc�A�7*

loss�H;���1       �	O�oYc�A�7*

loss�x<�?9�       �	�?pYc�A�7*

loss���:�4�H       �	��pYc�A�7*

loss�4=z1�       �	�qYc�A�7*

loss���<��~       �	�HrYc�A�7*

lossD=��$�       �	�rYc�A�7*

loss?].<W���       �	��sYc�A�7*

loss��;~��       �	)]tYc�A�7*

lossD��:B�       �	z�tYc�A�7*

loss<��:���       �	��uYc�A�7*

loss}�=ZY�J       �	�KvYc�A�7*

loss�g�;����       �	B&wYc�A�7*

lossb͡<�XkK       �	��wYc�A�7*

lossW��=sӯI       �	�cxYc�A�7*

lossH��<�.�       �	��yYc�A�7*

loss`Q;�_2       �	�CzYc�A�7*

loss��;ſ`�       �	��zYc�A�7*

losstz=�IV�       �	�{Yc�A�7*

loss쏓:��ϼ       �	V|Yc�A�7*

loss�^<�T#�       �	^�|Yc�A�7*

loss���<�<�2       �	ע}Yc�A�7*

loss�? <��u�       �	�B~Yc�A�7*

loss
�O=��F�       �	�Yc�A�7*

loss���:J�V�       �	8�Yc�A�7*

losse�;��_       �	�f�Yc�A�7*

loss�<Nl�       �	���Yc�A�7*

loss�p<�m}       �	)��Yc�A�7*

loss���;��        �	�/�Yc�A�7*

loss� "<I5w3       �	��Yc�A�7*

lossM�G;ΐ|�       �	{�Yc�A�7*

loss�\D;�mW       �	X��Yc�A�7*

loss��=�N       �	7��Yc�A�7*

loss���;�R�L       �	�B�Yc�A�7*

lossq�t<9ѿ�       �	�Yc�A�7*

loss�'�9���       �	���Yc�A�7*

loss�Y�;�ő�       �	!�Yc�A�7*

loss�� ;��       �	�Yc�A�7*

loss�%U=��~�       �	�b�Yc�A�7*

losso; %��       �	��Yc�A�7*

loss�=<n�       �	�Yc�A�7*

lossQ3�;���       �	zo�Yc�A�7*

loss�$�<�O-�       �	�$�Yc�A�7*

loss�*3<�U       �	�ȌYc�A�7*

loss#��:(�H[       �	e�Yc�A�7*

loss�u<?��j       �	�a�Yc�A�7*

loss�F�<}��        �	�\�Yc�A�7*

loss�;Uُ�       �	[(�Yc�A�7*

loss&k <��l�       �	�`�Yc�A�7*

loss@��;r'�       �	�?�Yc�A�7*

losso�<��=       �	��Yc�A�7*

lossib�;~�_K       �	���Yc�A�7*

loss�<³/�       �	DO�Yc�A�7*

loss4A=ϙ�b       �	E�Yc�A�7*

lossȹ�8��g�       �	��Yc�A�7*

loss�D(=m<zo       �	�D�Yc�A�7*

lossȔ;:t�J&       �	�K�Yc�A�7*

loss���9��q       �	,��Yc�A�7*

loss�#�<��       �	:��Yc�A�7*

loss?]=;�f��       �	�S�Yc�A�7*

lossZ�I:�	       �	���Yc�A�7*

lossWH�:�e�       �	���Yc�A�7*

loss�EE<�s       �	8�Yc�A�7*

loss�8�;
s�(       �	 қYc�A�7*

loss�%<+�       �	w��Yc�A�7*

loss�� =i�       �	B$�Yc�A�7*

loss�-p:�|uo       �	"��Yc�A�7*

loss,H9@jb�       �	t`�Yc�A�7*

loss�(�:k��       �	�Yc�A�7*

loss�A<7"�5       �	Z��Yc�A�7*

lossD=7:Hj��       �	�A�Yc�A�7*

loss��?;6u�+       �	Q٠Yc�A�7*

loss��8XʈU       �	Н�Yc�A�7*

loss��k:�m�       �	tC�Yc�A�7*

loss��#:?�M�       �	"�Yc�A�7*

lossD�&86e7Q       �	肣Yc�A�7*

loss#9�J�       �	x&�Yc�A�7*

loss���:&�]       �	�դYc�A�7*

loss~=�:       �	c~�Yc�A�7*

loss8߆:�wF�       �	��Yc�A�7*

loss��8ܥ.�       �	���Yc�A�7*

loss�B:�q�       �	2Z�Yc�A�7*

loss�0�;b8\       �	��Yc�A�7*

lossob�:�(��       �	ؚ�Yc�A�7*

loss	��=��>g       �	�7�Yc�A�7*

loss1�;����       �	ͩYc�A�7*

loss��<��O        �	�d�Yc�A�7*

loss�<퇢�       �	���Yc�A�7*

loss���;h6�       �	���Yc�A�7*

losst12<�u��       �	()�Yc�A�7*

loss1=�ʥ       �	���Yc�A�7*

loss�?�:ܲ`       �	���Yc�A�7*

lossW4U<�`_�       �	�:�Yc�A�7*

lossz��;Tg��       �	ѮYc�A�7*

lossY�;#q�       �	wi�Yc�A�7*

loss��,=#��       �	d�Yc�A�7*

loss?;_&       �	�˰Yc�A�7*

lossW4�;����       �	���Yc�A�7*

lossn�;�+V       �	q��Yc�A�7*

lossV�|;Ë�b       �	|H�Yc�A�7*

loss��2<0�x       �	�߳Yc�A�7*

loss��;�a       �	�w�Yc�A�8*

loss�B_;�*��       �	��Yc�A�8*

loss��u9��       �	��Yc�A�8*

lossٺ<	x�\       �	�[�Yc�A�8*

loss�:H��       �	���Yc�A�8*

loss��9*���       �	'�Yc�A�8*

loss���:y}�       �	���Yc�A�8*

loss!u�<�=�       �	GT�Yc�A�8*

loss��;�ǐ�       �	5��Yc�A�8*

loss}:�}u�       �	֏�Yc�A�8*

loss:�:��       �	<.�Yc�A�8*

loss$<M}yF       �	�мYc�A�8*

loss�&�<r�H[       �	*r�Yc�A�8*

lossᱜ<�M�       �	E�Yc�A�8*

loss��Q:0'�       �	⬾Yc�A�8*

lossO��<'[       �	dW�Yc�A�8*

loss��;���D       �	��Yc�A�8*

loss!�{;]���       �	��Yc�A�8*

lossƨ^<f�(       �	>��Yc�A�8*

loss��6=k��       �	J�Yc�A�8*

lossovW<\��       �	��Yc�A�8*

loss�� =��1f       �	5|�Yc�A�8*

loss�k<S�أ       �	<�Yc�A�8*

loss]�}:�q       �	3��Yc�A�8*

loss�(F;O(M       �	�\�Yc�A�8*

loss��k<�wNJ       �	i�Yc�A�8*

loss��<G�i�       �	j��Yc�A�8*

loss�@l;ӛ=D       �	�G�Yc�A�8*

loss�4=a�^g       �	���Yc�A�8*

loss�9�~*�       �	�u�Yc�A�8*

loss�$�;�z�       �	V�Yc�A�8*

loss
�:)�'a       �	U��Yc�A�8*

lossa�;��3       �	�=�Yc�A�8*

loss���<y�