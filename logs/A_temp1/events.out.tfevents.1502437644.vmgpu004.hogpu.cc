       �K"	   CXc�Abrain.Event:2�j&6�     �	o-CXc�A"��
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
 *�x=*
dtype0*
_output_shapes
: 
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2���
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
valueB"      *
_output_shapes
:*
dtype0
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
 *�\1=*
dtype0*
_output_shapes
: 
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2���*
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
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
Index0*
T0*
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
:���*
seed2���
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
seed2��*
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
seed2���
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
'sequential_1/dropout_1/cond/dropout/mulMul'sequential_1/dropout_1/cond/dropout/div)sequential_1/dropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
shrink_axis_mask *
_output_shapes
:*
Index0*
T0*
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
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
�
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
 *  �B*
_output_shapes
: *
dtype0
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
 *  �?*
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
value	B :*
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
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
valueB:*
_output_shapes
:*
dtype0
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
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
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
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
value	B : *
_output_shapes
: *
dtype0
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
1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes
: *
T0
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
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
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
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
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
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

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
 *    *
dtype0*
_output_shapes
: 
t
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:����������: 
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
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
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
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*!
_output_shapes
:���*
T0
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
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
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
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
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
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
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
valueB@*    *
dtype0*&
_output_shapes
:@
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
shape:@*
_output_shapes
:@*
shared_name * 
_class
loc:@conv2d_1/bias*
dtype0*
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
VariableV2*
shape:@@*&
_output_shapes
:@@*
shared_name *"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
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
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam_1
VariableV2*
shape:@@*&
_output_shapes
:@@*
shared_name *"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
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
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
W
zeros_10Const*
valueB�*    *
_output_shapes	
:�*
dtype0
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
 Bloss*
dtype0*
_output_shapes
: 
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "�n0N��     >�	�CXc�AJ��
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
seed2���
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
valueB"      *
_output_shapes
:*
dtype0
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
paddingVALID*
T0*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
use_cudnn_on_gpu(
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
 *�\1�*
_output_shapes
: *
dtype0
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
seed2���
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
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
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
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
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
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
Index0*
T0*
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
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
T0*
seed���)*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
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
dropout_2/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
T0
*
_output_shapes

::
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
dropout_2/cond/dropout/addAdd dropout_2/cond/dropout/keep_prob%dropout_2/cond/dropout/random_uniform*(
_output_shapes
:����������*
T0
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
seed2��
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
valueB"      *
_output_shapes
:*
dtype0
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
seed2���
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
���������*
_output_shapes
: *
dtype0
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
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
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
transpose_b( *'
_output_shapes
:���������
*
transpose_a( *
T0
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
 *  �?*
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
value	B : *
_output_shapes
: *
dtype0
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
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
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
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
valueB: *
dtype0*
_output_shapes
:
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
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
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
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: 
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
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
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
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
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
T0*
out_type0*
_output_shapes
:
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
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
out_type0*
_output_shapes
:*
T0
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
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Mul(sequential_1/dropout_2/cond/mul/Switch:1Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
_output_shapes
: *
T0
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
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
Tshape0*/
_output_shapes
:���������@*
T0
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
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
T0*
N
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
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������@*
T0
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
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
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
valueB@*    *
dtype0*
_output_shapes
:@
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
valueB@@*    *&
_output_shapes
:@@*
dtype0
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
valueB@*    *
_output_shapes
:@*
dtype0
�
conv2d_2/bias/Adam
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
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0�4T�       ��-	/�FCXc�A*

loss�@� ^�       ��-	�GCXc�A*

loss�@���       ��-	C�HCXc�A*

lossv�@dkJh       ��-	��ICXc�A*

loss� �?����       ��-	q8JCXc�A*

loss8��?4L2Y       ��-	��JCXc�A*

loss�[�?\���       ��-	�LCXc�A*

lossZј?��U�       ��-	��LCXc�A*

loss�;�?dؔp       ��-	W�MCXc�A	*

loss��?��P�       ��-	ioNCXc�A
*

loss�Aa?K	&�       ��-	/OCXc�A*

loss�ul?;G       ��-	��OCXc�A*

loss{q0?�=I�       ��-	(�PCXc�A*

loss�C??Pwk^       ��-	�QQCXc�A*

loss�@?._:�       ��-	�RCXc�A*

lossnua?{�[�       ��-	p�RCXc�A*

loss�>/ˊ�       ��-	�]SCXc�A*

loss2�%?�h�       ��-	UCXc�A*

loss�?�+��       ��-	K�UCXc�A*

losssl?y�\�       ��-	9bVCXc�A*

lossoID?���       ��-	�	WCXc�A*

loss�Y?ٌO       ��-	9�WCXc�A*

loss�E(?��8�       ��-	�iXCXc�A*

loss��L?AG�       ��-	�YCXc�A*

loss�?���       ��-	��YCXc�A*

loss�%?����       ��-	�}ZCXc�A*

loss/�?$;X       ��-	U/[CXc�A*

loss�<?�8Z�       ��-	P�[CXc�A*

lossέ?�yv�       ��-	X�\CXc�A*

loss�|�>S:&�       ��-	�W]CXc�A*

loss��
?R#z�       ��-	�u^CXc�A*

lossJջ>,U7�       ��-	� _CXc�A *

loss�^?7�       ��-	W�_CXc�A!*

loss� +?�58�       ��-	�v`CXc�A"*

loss$��>xME�       ��-	�}aCXc�A#*

loss�?ָ��       ��-	�hbCXc�A$*

loss�:�>m���       ��-	�cCXc�A%*

loss��&?��|�       ��-	_�cCXc�A&*

loss�C?�N}1       ��-	�]dCXc�A'*

lossc^	?���|       ��-	ReCXc�A(*

loss7_�>�/�       ��-	ԺeCXc�A)*

loss��>y���       ��-	�ffCXc�A**

loss{�>7�r�       ��-	gCXc�A+*

loss/��>[���       ��-	�gCXc�A,*

lossC��>y�-n       ��-	�ahCXc�A-*

lossC&
?ޭ��       ��-	�	iCXc�A.*

loss�ʹ>T6K4       ��-	)�iCXc�A/*

loss�m�>�C<�       ��-	�VjCXc�A0*

loss���>�k9       ��-	��jCXc�A1*

loss�[�>��^�       ��-	�kCXc�A2*

lossiY<>�C"W       ��-	@KlCXc�A3*

loss�$�>j�(       ��-	�lCXc�A4*

loss��Z>�ϑ       ��-	�mCXc�A5*

loss�?��a       ��-	fMnCXc�A6*

lossƭ�=ˑ
�       ��-	^�nCXc�A7*

loss/ɕ>ՓL�       ��-	�oCXc�A8*

loss�E>��        ��-	�?pCXc�A9*

loss��_>��ѷ       ��-	��pCXc�A:*

lossf�>@R       ��-	xqCXc�A;*

loss��>�ⳇ       ��-	rrCXc�A<*

lossSʸ>�E1�       ��-	osCXc�A=*

loss�Cv>ǝ_       ��-	��sCXc�A>*

loss��b>]|       ��-	��tCXc�A?*

loss���>\)��       ��-	ϠuCXc�A@*

loss�ބ>cCF       ��-	RFvCXc�AA*

loss�"�>��x�       ��-	��vCXc�AB*

loss�Y�>�ƺ�       ��-	ŏwCXc�AC*

lossF:�>{�d       ��-	�+xCXc�AD*

loss?��>�P�E       ��-	��xCXc�AE*

lossP�>�>�l       ��-	�kyCXc�AF*

loss(��>��F�       ��-	>zCXc�AG*

loss:��>1���       ��-	;�zCXc�AH*

loss���>�B�       ��-	B@{CXc�AI*

loss�.�>(��       ��-	k|CXc�AJ*

loss�>ռ�       ��-	��|CXc�AK*

losss�>k
Ք       ��-	�@}CXc�AL*

losso��>s2H       ��-	o�}CXc�AM*

loss�?�|l       ��-	�j~CXc�AN*

lossx��>����       ��-	!CXc�AO*

loss�d>��       ��-	ϤCXc�AP*

loss�f�>����       ��-	AD�CXc�AQ*

lossf��>���v       ��-	
ۀCXc�AR*

lossq�>�]�       ��-	\s�CXc�AS*

loss��=�H$r       ��-	��CXc�AT*

loss��>;>_^       ��-	Ü�CXc�AU*

loss�c�>c�       ��-	35�CXc�AV*

loss1�h>�P�       ��-	tσCXc�AW*

loss:��>����       ��-	g�CXc�AX*

lossX~�>�@sC       ��-	���CXc�AY*

loss�7E>�3p�       ��-	d��CXc�AZ*

losse�>~�?V       ��-	��CXc�A[*

loss�/�>9+       ��-	��CXc�A\*

loss��R>�q7�       ��-	� �CXc�A]*

loss�ܤ>�2��       ��-	Ӿ�CXc�A^*

loss!��>$S'�       ��-	ԁ�CXc�A_*

loss*�u>U\CF       ��-	#�CXc�A`*

loss1��>ڝ8C       ��-	ǹ�CXc�Aa*

lossŤ�>��;       ��-	�R�CXc�Ab*

loss0�?�x}�       ��-	N�CXc�Ac*

losse�>b#�       ��-	v��CXc�Ad*

loss8�/>���       ��-	�CXc�Ae*

loss}��=�m�       ��-	�CXc�Af*

loss�uf>Q�K�       ��-	 C�CXc�Ag*

loss�X9>��r       ��-	�ۏCXc�Ah*

loss��5>�olL       ��-	���CXc�Ai*

loss��>0���       ��-	1�CXc�Aj*

lossn�#>T���       ��-	��CXc�Ak*

loss��>�
ŧ       ��-	f��CXc�Al*

loss7s�>K﫿       ��-	�X�CXc�Am*

loss��}>���       ��-	���CXc�An*

lossr��>��~�       ��-	���CXc�Ao*

loss4��>)l��       ��-	i:�CXc�Ap*

loss 1�=ia�       ��-	RԖCXc�Aq*

loss���=�@       ��-	�r�CXc�Ar*

loss=0f=�t��       ��-	_�CXc�As*

lossӍi>D�       ��-	��CXc�At*

loss�`>ڽ@�       ��-	/M�CXc�Au*

loss�"�>���"       ��-	��CXc�Av*

loss�I�><�       ��-	Z�CXc�Aw*

losss8n>|��       ��-	�"�CXc�Ax*

loss{D>�p�       ��-	Ǽ�CXc�Ay*

loss�Aq>]��>       ��-	�T�CXc�Az*

lossXRe>z���       ��-	��CXc�A{*

lossVl>E�j�       ��-	��CXc�A|*

loss���=�¼�       ��-	�>�CXc�A}*

loss��>��7`       ��-	�ڞCXc�A~*

loss8(�>f�?       ��-	�y�CXc�A*

loss�a�=���h       �	��CXc�A�*

loss#pu>z9�       �	���CXc�A�*

loss�q>��rO       �	�|�CXc�A�*

lossX{c>P��       �	��CXc�A�*

loss�T�>H�/       �	���CXc�A�*

loss�c>���       �	_]�CXc�A�*

loss]>��       �	C�CXc�A�*

loss�Ab>����       �	M��CXc�A�*

loss߇6>��N8       �	29�CXc�A�*

loss�{2>�c�       �	�ҥCXc�A�*

lossj�>�:�       �	�t�CXc�A�*

lossi�>�       �	��CXc�A�*

loss��=΢�m       �	_��CXc�A�*

loss_��=)���       �	�P�CXc�A�*

lossO�>z/e       �	�CXc�A�*

loss�-I>48��       �	T��CXc�A�*

loss��|>{��]       �	�*�CXc�A�*

lossZt�>mwʭ       �	��CXc�A�*

loss֏z=AYf       �	���CXc�A�*

loss�}C>����       �	�%�CXc�A�*

lossVM�=g&�       �	CʬCXc�A�*

loss�>#>�*D�       �	Rb�CXc�A�*

loss	�=�4��       �	v��CXc�A�*

loss�I>t6�*       �	��CXc�A�*

lossdxw>��p7       �	�>�CXc�A�*

loss1�3>0Вb       �	��CXc�A�*

loss�(�>nt��       �	肰CXc�A�*

lossC�>d*�       �	�$�CXc�A�*

losscU=p�%       �	�±CXc�A�*

loss r�=L��       �	|b�CXc�A�*

loss �m>c#��       �	��CXc�A�*

loss�>��q�       �	<��CXc�A�*

loss;>e��       �	�>�CXc�A�*

loss4��>s�       �	#ݴCXc�A�*

lossF?>���M       �	���CXc�A�*

loss�>����       �	�CXc�A�*

lossj	>��s       �	���CXc�A�*

loss- >����       �	�V�CXc�A�*

loss�})>-�/        �	��CXc�A�*

loss���=��+       �	P��CXc�A�*

loss��>�$a�       �	�#�CXc�A�*

loss1)K>"�F       �	��CXc�A�*

loss�O>�!N       �	HN�CXc�A�*

loss�>��       �	�CXc�A�*

loss�-=�l>       �	'��CXc�A�*

loss o�=^k�       �	l!�CXc�A�*

loss�jL>)EI�       �	ȷ�CXc�A�*

loss̴�=���       �	UL�CXc�A�*

lossy�>k�`       �	�CXc�A�*

loss��>\��v       �	�t�CXc�A�*

loss��2>B�+t       �	��CXc�A�*

lossc�>��       �	���CXc�A�*

loss��4=�u�       �	�C�CXc�A�*

lossv�>=+��a       �	7��CXc�A�*

loss�Y>��b       �	�w�CXc�A�*

loss�>u��       �	�
�CXc�A�*

loss���=b���       �	���CXc�A�*

lossx�#>��)�       �	�<�CXc�A�*

loss�)�=���\       �	���CXc�A�*

loss���=���       �	"o�CXc�A�*

loss�Q>JjQ       �	*�CXc�A�*

loss�L>1�H       �	���CXc�A�*

loss��R>O��
       �	M1�CXc�A�*

lossTE>��x       �	���CXc�A�*

loss�P>��       �	�V�CXc�A�*

loss�>B>�       �	���CXc�A�*

loss��>��       �	N~�CXc�A�*

loss��=w�%       �	�CXc�A�*

loss�;y>��       �	���CXc�A�*

loss-q.>��       �	\�CXc�A�*

loss��=lAM       �	� �CXc�A�*

loss���>z�;�       �	��CXc�A�*

loss-�>�P	       �	�3�CXc�A�*

loss�,>�v�r       �	S��CXc�A�*

lossm��=�X�       �	#i�CXc�A�*

loss��=��       �	`�CXc�A�*

lossE�3>?��       �	+��CXc�A�*

lossv,$>R*2       �	�;�CXc�A�*

lossӡ=y��       �	���CXc�A�*

loss���=�d��       �	�v�CXc�A�*

loss��>��       �	^�CXc�A�*

loss�>is�       �	ʨ�CXc�A�*

loss�J�=!��<       �	?�CXc�A�*

lossR�h>�rB       �	J��CXc�A�*

loss�">�A��       �	uw�CXc�A�*

loss�n�=�T��       �	�CXc�A�*

losso,�=����       �	���CXc�A�*

lossx�0>UEx       �	�d�CXc�A�*

lossH�=�e�`       �	���CXc�A�*

loss_�W>qy<�       �	��CXc�A�*

lossF˒>j>�)       �	��CXc�A�*

loss�0x>���       �	½�CXc�A�*

loss���>��b4       �	KW�CXc�A�*

losst��=!Vky       �	J��CXc�A�*

loss ;�=��-       �	���CXc�A�*

lossƌ>���       �	�.�CXc�A�*

loss��z>���       �	]��CXc�A�*

lossSd=�#�;       �	�f�CXc�A�*

loss�>���:       �	=�CXc�A�*

loss�>SzO       �	5��CXc�A�*

lossl�7>u�       �	�q�CXc�A�*

lossA*>��Ɵ       �	m�CXc�A�*

loss��;>��Od       �	#��CXc�A�*

loss�G=W]��       �	�T�CXc�A�*

loss钘=���       �	d@�CXc�A�*

loss=lC>���       �	bL�CXc�A�*

loss���>2��K       �	���CXc�A�*

lossr>V,~       �	���CXc�A�*

loss�B?>�ť       �	%��CXc�A�*

loss�N$>��wq       �	�n�CXc�A�*

lossM��>��:�       �	�d�CXc�A�*

losse�T>��@�       �	���CXc�A�*

loss��W=$��`       �	%��CXc�A�*

lossX�>��?       �	�+�CXc�A�*

loss4K>�=��       �	���CXc�A�*

loss��d>zC�O       �	Ih�CXc�A�*

losst�>i9o       �	S�CXc�A�*

loss��~>w02�       �	��CXc�A�*

lossW!>j�t       �	u:�CXc�A�*

loss|��=u]�A       �	���CXc�A�*

loss�A�=3Q7|       �	�r�CXc�A�*

loss��(>��8�       �	�CXc�A�*

loss�>j�p�       �	���CXc�A�*

loss�	>)�4�       �	~9�CXc�A�*

lossTA>��       �	P��CXc�A�*

loss��> �K1       �	9~�CXc�A�*

lossDr=��i       �	�CXc�A�*

loss6T:=�[�y       �	��CXc�A�*

loss���=��.       �	y��CXc�A�*

loss� �=��4       �	zn�CXc�A�*

loss�I)>n�{       �	R�CXc�A�*

loss���=ZT��       �	y��CXc�A�*

loss�~�==�        �	SX�CXc�A�*

loss��>z�ˢ       �	���CXc�A�*

lossC��<��#       �	V��CXc�A�*

loss��>b�-�       �	�:�CXc�A�*

loss)��=ڴ�       �	}��CXc�A�*

loss	��>폌J       �	|��CXc�A�*

loss�߆=���       �	S$�CXc�A�*

losst�>��
�       �	��CXc�A�*

loss�.>�M[�       �	�a�CXc�A�*

lossį>1U �       �	��CXc�A�*

loss���=L��\       �	���CXc�A�*

loss:j[>.��       �	�S�CXc�A�*

loss[��=y"]x       �	b��CXc�A�*

loss��1>���C       �	���CXc�A�*

loss�)�=�)X�       �	O@�CXc�A�*

loss_=�=���       �	��CXc�A�*

loss W>��}       �	���CXc�A�*

losse�=G9��       �	6��CXc�A�*

loss#-y=N�U�       �	�3 DXc�A�*

loss�?>���p       �	�� DXc�A�*

loss\��=2R�       �	OxDXc�A�*

loss�E2=GGۗ       �	�DXc�A�*

loss	{=�!W�       �	�iDXc�A�*

lossv��=��~       �	�DXc�A�*

loss�m�=��       �	��DXc�A�*

loss��3>�8�u       �	�DDXc�A�*

loss/N">95�7       �	q�DXc�A�*

loss��>�5Y|       �	~DXc�A�*

loss�I=980       �	��DXc�A�*

loss�
>K�[-       �	muDXc�A�*

loss��.=���       �	�	DXc�A�*

lossa#�<�p�       �	I�	DXc�A�*

loss�~>�1-�       �	D�
DXc�A�*

lossX=�m%.       �	�.DXc�A�*

loss�!�=%�,�       �	��DXc�A�*

loss�g�=>�       �		�DXc�A�*

lossl�>���       �	��DXc�A�*

loss��=g�ݹ       �	��DXc�A�*

lossTd�>��z       �	\XDXc�A�*

loss(�O>}`_�       �	�*DXc�A�*

loss��/>�n��       �	F\DXc�A�*

loss�i>�q҉       �	��DXc�A�*

loss��>~ �       �	�DXc�A�*

lossHx�=.��       �	�=DXc�A�*

loss��>MT�G       �	��DXc�A�*

loss��=4�!�       �	qtDXc�A�*

loss�6I>�D�#       �	 DXc�A�*

loss`;=��S       �	k�DXc�A�*

loss�w>,G�       �	1�DXc�A�*

loss��c=�L;|       �	�DDXc�A�*

lossLU>g�        �	�DDXc�A�*

loss�F>�Fa       �	8�DXc�A�*

loss��>>=��X       �	w�DXc�A�*

loss9>��)       �	�MDXc�A�*

loss���=�{��       �	v�DXc�A�*

loss�~=���e       �	yDXc�A�*

loss Z>A�}8       �	�DXc�A�*

lossNa*>�'�*       �	TDXc�A�*

loss�V=�Қ       �	��DXc�A�*

losseZ1>v��       �	`WDXc�A�*

loss���=
��+       �	�DXc�A�*

loss`�c=>t&       �	0� DXc�A�*

loss0e>��\       �	eR!DXc�A�*

loss%�%>> ޚ       �	��!DXc�A�*

loss��=ݨ��       �	�"DXc�A�*

loss��=�`       �	B"#DXc�A�*

loss꿅>3*(�       �	��#DXc�A�*

losssĭ=�ҙ�       �	R$DXc�A�*

loss�z�=�fd       �	��$DXc�A�*

loss�>{��M       �	}%DXc�A�*

loss���>g>�       �	e&DXc�A�*

loss���=[|5�       �	
�&DXc�A�*

lossc��= ���       �	�Y'DXc�A�*

loss��=;�t       �	f�'DXc�A�*

lossA7h=���       �	9�(DXc�A�*

loss}�->���       �	N)DXc�A�*

loss?�=����       �	��)DXc�A�*

loss�	>?�b%       �	�*DXc�A�*

loss�cS>��q       �	�1+DXc�A�*

loss]E>�G5�       �	��+DXc�A�*

loss��=���       �	�a,DXc�A�*

loss��=Ұ��       �	��,DXc�A�*

lossP�<��1�       �	O#.DXc�A�*

lossM5�=�.v       �	�.DXc�A�*

loss���=��s       �	iW/DXc�A�*

lossaW�>�ߪ�       �	��/DXc�A�*

loss��0>�e       �	��0DXc�A�*

loss�4�=�O�       �	�1DXc�A�*

loss�i'>�G�R       �	�1DXc�A�*

loss�>��       �	S2DXc�A�*

loss}K�=wOF�       �	M�2DXc�A�*

loss ��=#�1       �	R�3DXc�A�*

losst��=E~��       �	B�5DXc�A�*

loss���=��G       �	�`6DXc�A�*

loss��=����       �	�	7DXc�A�*

loss�F>d�?�       �	��7DXc�A�*

losso�k=Ϸ��       �	GZ9DXc�A�*

loss�=�O�!       �	4�:DXc�A�*

loss�\�= B7�       �	�\;DXc�A�*

loss���=�0�       �	X<DXc�A�*

lossHΛ=�t,{       �	��<DXc�A�*

loss��=�ۡe       �	I=DXc�A�*

lossl�q=��x       �	��=DXc�A�*

loss0�=&��       �	d�>DXc�A�*

loss�>�e��       �	I-?DXc�A�*

loss��e>3���       �	��?DXc�A�*

loss!	>S�dh       �	j�@DXc�A�*

loss��=_�W       �	s-ADXc�A�*

loss�G>�K�       �	r�ADXc�A�*

lossci=W� n       �	ՓBDXc�A�*

loss�a�=}E�\       �	�<CDXc�A�*

loss��v=h�B       �	��CDXc�A�*

loss�P�=��^R       �	�DDXc�A�*

loss1��=i�!       �	�6EDXc�A�*

loss��=J��       �	��EDXc�A�*

loss�2�=l_h�       �	L�FDXc�A�*

lossh>f��       �	�]GDXc�A�*

loss�>�<A�       �	bjHDXc�A�*

lossZ�g=a��       �	(,IDXc�A�*

loss�ٴ=�VDf       �	{�IDXc�A�*

loss]2a>'U0�       �	��JDXc�A�*

lossVQ5>	�;�       �	BxKDXc�A�*

loss~#>~+�       �	�LDXc�A�*

loss�l>���a       �	��MDXc�A�*

loss��^=����       �	"ONDXc�A�*

loss2=�~I�       �	��NDXc�A�*

lossnG>�3L       �	��ODXc�A�*

loss���=]&A�       �	B?PDXc�A�*

loss�S>�N-�       �	�PDXc�A�*

losst� >ה�z       �	4�QDXc�A�*

loss�TN>W@/        �	�"RDXc�A�*

lossm%>�qI       �	��RDXc�A�*

loss@�>r%�       �	�oSDXc�A�*

lossqQ>��       �	�TDXc�A�*

loss�˩=T�<X       �	ҫTDXc�A�*

loss&>΀�}       �	LUDXc�A�*

loss+��>��?P       �	}�UDXc�A�*

loss��E=�ڪ�       �	��VDXc�A�*

loss(49=?V�]       �	�WDXc�A�*

loss��c=�##�       �	fYDXc�A�*

loss�S >��>I       �	-ZDXc�A�*

loss���=�s��       �	�ZDXc�A�*

loss1B$>�!��       �	-�[DXc�A�*

loss?9Q>�z�P       �	M2\DXc�A�*

lossX��< �!4       �	�\DXc�A�*

loss�B�=�       �	v]DXc�A�*

loss�;>�q       �	�^DXc�A�*

loss�E>�(�       �	��^DXc�A�*

lossEv/>�p	       �	�\_DXc�A�*

loss�'�=��       �	��_DXc�A�*

lossܢ,>N�\�       �	��`DXc�A�*

loss!��=����       �	�$aDXc�A�*

loss��:>B	_�       �	3�aDXc�A�*

loss�q>�vm�       �	#hbDXc�A�*

loss7�Q>����       �	�cDXc�A�*

loss���=)�-       �	��cDXc�A�*

loss��=b��J       �	>>dDXc�A�*

loss�]U=�r<0       �	��dDXc�A�*

loss\�=�b�6       �	�~eDXc�A�*

lossT�>R$b�       �	*fDXc�A�*

loss&J�=�b�       �	��fDXc�A�*

lossFa=��#       �	�QgDXc�A�*

lossZ�>��       �	 �gDXc�A�*

loss��o=ȏ`	       �	8�hDXc�A�*

loss�ۅ>�U       �	miDXc�A�*

loss}��=&ˆ       �	_�iDXc�A�*

loss�Ԉ><Q��       �	�YjDXc�A�*

lossM�=$��       �	,�jDXc�A�*

loss<�>	�w       �	��kDXc�A�*

loss@�3=���       �	]1lDXc�A�*

lossj$�=6=6M       �	��lDXc�A�*

loss,>Ϊ±       �	�lmDXc�A�*

lossY�=u��       �	�nDXc�A�*

loss�=帀�       �	p�nDXc�A�*

lossm��=n���       �	�,oDXc�A�*

loss� >�T�       �	��oDXc�A�*

loss:P=���       �	qTpDXc�A�*

losson�=x-��       �	�qDXc�A�*

loss]�3=v�       �	W�qDXc�A�*

loss8�%=8DC       �	�/rDXc�A�*

loss8�=��
       �	&�rDXc�A�*

lossK��>F�G�       �	dsDXc�A�*

loss!�2>���J       �	�sDXc�A�*

loss��=�a*       �	��tDXc�A�*

loss)A�=�R��       �	|)uDXc�A�*

loss��m<s슻       �	��uDXc�A�*

loss(K�=�op       �	3RvDXc�A�*

lossOҊ>��       �	�vDXc�A�*

loss!W>>�e       �	ӇwDXc�A�*

loss�=Q�Ni       �	�'xDXc�A�*

lossE7>5��       �	��xDXc�A�*

loss��F>j�l�       �	>ZyDXc�A�*

loss�<=��?�       �	�yDXc�A�*

loss��=�@�0       �	��zDXc�A�*

loss�>ւ�       �	{DXc�A�*

loss�'>p��       �	[�{DXc�A�*

loss�+>�!�}       �	RI|DXc�A�*

loss�Ĭ=ƭEI       �	��|DXc�A�*

loss��>��y�       �	�v}DXc�A�*

loss�X�=���       �	(~DXc�A�*

loss�E->��M+       �	ˠ~DXc�A�*

loss�β=󦙡       �	p?DXc�A�*

loss��l=h`j       �	��DXc�A�*

lossr=.���       �	�r�DXc�A�*

lossf��>�r��       �	��DXc�A�*

loss+�=B(       �	뫁DXc�A�*

loss!��=�4��       �	a�DXc�A�*

loss�~�=9�n       �	�DXc�A�*

loss��=���       �	u��DXc�A�*

losss�>�$       �	�T�DXc�A�*

loss��)<i�       �	s��DXc�A�*

loss�/,>�V�       �	@��DXc�A�*

lossl��=�w��       �	L�DXc�A�*

lossα2>i�²       �	���DXc�A�*

loss͊�> `�p       �	���DXc�A�*

loss,��=��       �	�=�DXc�A�*

lossS	=�%n�       �	��DXc�A�*

loss��=q�N�       �	^ՉDXc�A�*

loss.i=�<��       �	N�DXc�A�*

loss1�=��       �	�/�DXc�A�*

loss�*O>`Gt}       �	��DXc�A�*

loss�A�>��D       �	���DXc�A�*

loss@7�=r��p       �	rn�DXc�A�*

losss��=�p�       �	�d�DXc�A�*

lossJ">���       �	%A�DXc�A�*

lossD��=#i��       �	!��DXc�A�*

loss���=�R       �	�2�DXc�A�*

loss\��=�a       �	�a�DXc�A�*

lossz�>8��       �	J
�DXc�A�*

loss�e=Hq2�       �	i��DXc�A�*

loss+/>5|>       �	�Q�DXc�A�*

loss��>Y�9o       �	o+�DXc�A�*

loss��#>�B�       �	~ȖDXc�A�*

loss1��<��Q�       �	�f�DXc�A�*

lossh7�<5��       �	0�DXc�A�*

loss�O=�x�       �	���DXc�A�*

loss�m#=�Yo,       �	�P�DXc�A�*

loss ��=b�?�       �	��DXc�A�*

lossI	=�h��       �	��DXc�A�*

loss7b�=���       �	�(�DXc�A�*

loss��!>�4.       �	�ěDXc�A�*

loss��=�	LY       �	ɜDXc�A�*

loss=�i�       �	w��DXc�A�*

loss�v2= ��^       �	�"�DXc�A�*

loss
ߣ=����       �	R��DXc�A�*

loss��>���t       �	�şDXc�A�*

loss|K=`q{        �	F`�DXc�A�*

lossLZ>���Z       �	��DXc�A�*

loss��p=�ף       �	�)�DXc�A�*

loss�S�=&��       �	n��DXc�A�*

loss�ݮ=��3       �	�ۤDXc�A�*

loss�l=�>��       �	�w�DXc�A�*

loss� W=��        �	j�DXc�A�*

lossJ�<�_�       �	S��DXc�A�*

loss�\�=ܩX       �	���DXc�A�*

lossv��=`�       �	�"�DXc�A�*

lossf�>~QB       �	���DXc�A�*

lossw��=�s�       �	�Z�DXc�A�*

lossӔ�=��p�       �	U��DXc�A�*

lossR!K=�Wl>       �		�DXc�A�*

loss{��=�W�I       �	���DXc�A�*

lossX��<�9�       �	MM�DXc�A�*

loss��D=�L       �	�DXc�A�*

lossIm<��       �	G��DXc�A�*

loss<=����       �	�9�DXc�A�*

loss߮�=m��       �	�ٮDXc�A�*

loss�#=a��       �	M��DXc�A�*

lossLc+>���       �	�-�DXc�A�*

lossVۆ=(m�       �	�аDXc�A�*

lossf�>5��       �	yw�DXc�A�*

loss��C=��?�       �	��DXc�A�*

loss�?=��5�       �	TòDXc�A�*

loss|	>*��       �	�d�DXc�A�*

loss�U�<8��^       �	���DXc�A�*

loss&z�<��       �	К�DXc�A�*

loss	�<ύ�       �	�6�DXc�A�*

lossi�G<?�>2       �	tԵDXc�A�*

loss-��<zJPw       �	�v�DXc�A�*

loss�n�<*�       �	$�DXc�A�*

lossf��;D�tQ       �	�ɷDXc�A�*

loss:��=ԏ�       �	�r�DXc�A�*

loss�4�;����       �	��DXc�A�*

lossƙ&;�Ttu       �	O��DXc�A�*

lossB�<�Kz�       �	�M�DXc�A�*

lossˎ�="�H�       �	$�DXc�A�*

lossO>[�       �	��DXc�A�*

loss���=���       �	1(�DXc�A�*

lossנ<ח��       �	QټDXc�A�*

lossY�<�       �	�v�DXc�A�*

loss�G ?z";�       �	��DXc�A�*

loss�>�<F��       �	��DXc�A�*

lossƀ>cLS�       �	�S�DXc�A�*

loss���=&��       �	<��DXc�A�*

loss�:>�&�       �	q��DXc�A�*

loss��=A9�       �	�-�DXc�A�*

loss#u=@)��       �	���DXc�A�*

loss�J>Z+�%       �	�Y�DXc�A�*

loss~�=��D�       �	c��DXc�A�*

lossl*4>�բy       �	^��DXc�A�*

loss��=}�۾       �	)�DXc�A�*

loss�!O>Z�v       �	���DXc�A�*

loss��">� ~�       �		n�DXc�A�*

lossR3O>{��Y       �	��DXc�A�*

loss��= D�]       �	���DXc�A�*

loss�V>���       �	G;�DXc�A�*

lossZH)>rI�       �	���DXc�A�*

loss+�=��O       �	D�DXc�A�*

lossm�=�;Wi       �	� �DXc�A�*

loss��=ۘ�       �	> �DXc�A�*

loss��=���       �	A��DXc�A�*

loss�_`=�>
{       �	vO�DXc�A�*

lossĄ>��Z6       �	t��DXc�A�*

lossv3> ���       �	���DXc�A�*

lossnI�<��b1       �	��DXc�A�*

loss6��<��)�       �	��DXc�A�*

loss�/=�D�       �	(I�DXc�A�*

loss���=�]��       �	L��DXc�A�*

lossvV�<��L       �	�x�DXc�A�*

loss�>$�!       �	��DXc�A�*

loss? >˶=�       �	���DXc�A�*

loss]`==ct�       �	^J�DXc�A�*

loss�ԡ=��5       �	n��DXc�A�*

loss��w=*�^       �	�t�DXc�A�*

loss�H�<��Y       �	A�DXc�A�*

loss6n�=�.z�       �	���DXc�A�*

loss�ʮ==��       �	�H�DXc�A�*

loss|�4=q�`�       �	&��DXc�A�*

loss�q=F��H       �	�u�DXc�A�*

loss���=ڢy.       �	+�DXc�A�*

lossN�#>�{I>       �	��DXc�A�*

loss�F	=�xtD       �	�C�DXc�A�*

lossy3�=���       �	�j�DXc�A�*

loss�aG=G��       �	6!�DXc�A�*

loss}��=6��       �	0��DXc�A�*

loss�=���       �	�`�DXc�A�*

loss�=H       �	���DXc�A�*

losshq>SR�       �	ڐ�DXc�A�*

loss3VK=��       �	�)�DXc�A�*

loss�>���W       �	��DXc�A�*

lossW�<��C�       �	`�DXc�A�*

loss�=�y
�       �	���DXc�A�*

lossm�= ��:       �	���DXc�A�*

loss1&�=zϛn       �	�1�DXc�A�*

lossE�6>4*~       �	*��DXc�A�*

loss?s$>_�J       �	p\�DXc�A�*

loss�A(>RE&2       �	j��DXc�A�*

loss��$>$	�       �	Ɗ�DXc�A�*

loss<'�=NR��       �	m�DXc�A�*

loss/�\=!~�       �		��DXc�A�*

lossģ>t��x       �	�X�DXc�A�*

loss���=)&       �	V��DXc�A�*

lossn�O=m~�       �	��DXc�A�*

loss�N�=Nw       �	�=�DXc�A�*

loss�D>�g�       �	N��DXc�A�*

lossu�>����       �	�d EXc�A�*

loss���=�4`       �	3� EXc�A�*

loss�l�=�p       �	c�EXc�A�*

loss�=_�       �	4EXc�A�*

loss���=���M       �	8�EXc�A�*

loss��>*��       �	tEXc�A�*

loss���>��;g       �	� EXc�A�*

lossp�=$�h       �	��EXc�A�*

lossMq>�cp.       �	��EXc�A�*

loss	.�=b�       �	�EXc�A�*

loss2�O>NI~�       �	_�EXc�A�*

loss�A:=3��        �	<JEXc�A�*

loss�a=���       �	�EXc�A�*

lossAO�=5n��       �	�pEXc�A�*

loss�%-=P���       �	�	EXc�A�*

lossL�>0�y�       �	7�	EXc�A�*

loss���=�7�q       �	�X
EXc�A�*

loss8�=�{�       �	[�
EXc�A�*

lossv�=����       �	H�EXc�A�*

loss��=?��       �	!EXc�A�*

loss��>�W��       �	�EXc�A�*

loss�z�=351�       �	kIEXc�A�*

loss�W�=��d�       �	w�EXc�A�*

loss�j�<ǚF.       �	vEXc�A�*

loss�o�={�94       �	EXc�A�*

loss)\�>F��       �	�EXc�A�*

lossl>,��N       �	*9EXc�A�*

loss��A=�0        �	_�EXc�A�*

loss3��<٪[�       �	&mEXc�A�*

lossHj�<�/ 2       �	�EXc�A�*

loss, ">�       �	j�EXc�A�*

loss=��=��6a       �	 <EXc�A�*

lossV>�7��       �	��EXc�A�*

lossܰ�=b���       �	�rEXc�A�*

loss��=P� �       �	wEXc�A�*

lossW��=��5w       �	@�EXc�A�*

loss}:�<S �S       �	:EXc�A�*

loss��
=�n�P       �	��EXc�A�*

lossI��=$#��       �	��EXc�A�*

loss��=g�}�       �	SEXc�A�*

lossj`�>&�       �	<�EXc�A�*

lossŧ�<�B%^       �	w�EXc�A�*

loss=�O<-�lr       �	CEXc�A�*

loss)o=�̼�       �	M�EXc�A�*

loss�<yUsF       �	�yEXc�A�*

loss�@�=�ҟ[       �	�CEXc�A�*

loss=P>T`��       �	:�EXc�A�*

loss,��=$�p�       �	�xEXc�A�*

loss���=�+{�       �	�EXc�A�*

lossz��=�Y��       �	j�EXc�A�*

loss%d�==d$�       �		5 EXc�A�*

lossa��=����       �	�� EXc�A�*

loss��<��U%       �	z�!EXc�A�*

loss&H~=�3�       �	 7"EXc�A�*

lossw�=����       �	R�"EXc�A�*

lossV!>�J$�       �	Ot#EXc�A�*

lossܾ�=�qڑ       �	.$EXc�A�*

loss���<~'�       �	�5%EXc�A�*

loss�oV=Gy��       �	^�%EXc�A�*

losss�=�
�_       �	�}&EXc�A�*

loss�h�=Z�       �	2'EXc�A�*

loss��t={wZ'       �	˻'EXc�A�*

lossSE�=M�J�       �	�_(EXc�A�*

loss O6>�]j       �	{�(EXc�A�*

loss�'�=i��       �	��)EXc�A�*

loss ۇ=�XhB       �	�5*EXc�A�*

loss�6�=?�lz       �	%�*EXc�A�*

loss�>y�n�       �	[`+EXc�A�*

loss��>*鿰       �	s�+EXc�A�*

lossR��=R�M�       �	�-EXc�A�*

loss��Z<t0_       �	#�-EXc�A�*

loss��E=C��       �	�9.EXc�A�*

loss�l�=exBs       �	�.EXc�A�*

loss�^=�P��       �	�j/EXc�A�*

losszR�=��M1       �	;�/EXc�A�*

lossZ�=��f       �	M�0EXc�A�*

lossTL\=\�       �	�O1EXc�A�*

loss�%�=cQJ�       �	��1EXc�A�*

loss���=t��P       �	ux2EXc�A�*

loss4��=�`�x       �	3EXc�A�*

lossW�v=����       �	��3EXc�A�*

loss�~Q>����       �	�24EXc�A�*

loss�I�<�JV�       �	��4EXc�A�*

loss�	�=*m��       �	�[5EXc�A�*

loss,��<�*ޜ       �	B�5EXc�A�*

loss��>����       �	,6EXc�A�*

lossx��=|���       �	<7EXc�A�*

loss>=��e�       �	��7EXc�A�*

loss�=3�R       �	�L8EXc�A�*

loss���=���"       �	(�8EXc�A�*

loss�	�=Y�>�       �	�9EXc�A�*

loss=��=��u�       �	>&:EXc�A�*

loss4ǋ=L���       �	ؼ:EXc�A�*

lossjBP<��:       �	�[;EXc�A�*

loss���<a���       �	o�;EXc�A�*

loss	�0>���{       �	Ė<EXc�A�*

loss���=�P�@       �	�2=EXc�A�*

loss��=��$�       �	 �=EXc�A�*

loss�ǔ=�F�{       �	a?EXc�A�*

loss\�W=�YI�       �	"�?EXc�A�*

loss)��<�mZ       �	�n@EXc�A�*

loss��<a��q       �	.AEXc�A�*

loss�<`=c�1�       �	=�AEXc�A�*

lossc�=C1�       �	�HBEXc�A�*

loss�!�=!��i       �	>�BEXc�A�*

loss{5">���f       �	U�CEXc�A�*

loss��,=[�l5       �	�BDEXc�A�*

loss&I�=b+{X       �	��DEXc�A�*

loss��J=oN2�       �	{EEXc�A�*

loss#��=��       �	�FEXc�A�*

loss.�e=>�6       �	��FEXc�A�*

loss�0<�'       �	�GGEXc�A�*

loss�ԗ=�HA�       �	u�GEXc�A�*

loss���=4�w       �	D�HEXc�A�*

loss��=�Ͱ       �	�>IEXc�A�*

loss�FE>Гj       �	(-JEXc�A�*

loss��=l_       �	8�JEXc�A�*

loss6�>ی�       �	�KEXc�A�*

loss��`=���       �	KLEXc�A�*

loss�=�T��       �	>BMEXc�A�*

loss�m=h��       �	e�MEXc�A�*

loss�A	>u�g       �	9{NEXc�A�*

lossgC={�       �	�OEXc�A�*

loss�Xp=�
�       �	��OEXc�A�*

loss[�n=����       �	�vPEXc�A�*

lossxy�=U��       �	�QEXc�A�*

loss��e<�.��       �	D�QEXc�A�*

loss2F�=�r�       �	cREXc�A�*

loss3��=�[��       �	D�REXc�A�*

loss��=I-�       �	R�SEXc�A�*

loss4'>g�`�       �	4TEXc�A�*

loss���=7��       �	��TEXc�A�*

losso|�<E���       �	rUEXc�A�*

loss�>�n9�       �	�VEXc�A�*

loss.!=�?�       �	ץVEXc�A�*

lossd�<V��t       �	�?WEXc�A�*

loss�v>��	       �	0�WEXc�A�*

loss\EW=��K       �	qXEXc�A�*

lossɕ7>���       �	W
YEXc�A�*

loss�S6=����       �	��YEXc�A�*

loss�t}=���S       �	�AZEXc�A�*

loss�C�=�5c�       �	�ZEXc�A�*

loss ��<-�J�       �	�p[EXc�A�*

loss�Q=/;�4       �	6\EXc�A�*

lossWѾ=p,�       �	 �\EXc�A�*

loss��=�̺�       �	;7]EXc�A�*

loss�=�Y�       �	��]EXc�A�*

loss�9�=ZUz       �	�m^EXc�A�*

loss?��=�f�#       �	p_EXc�A�*

loss��=t=��       �	?�_EXc�A�*

lossjR�<��C�       �	6>`EXc�A�*

loss�v�=��ZE       �	��`EXc�A�*

loss��=�h�       �	�saEXc�A�*

lossi�=�(�       �	1bEXc�A�*

loss��=-))       �	b�bEXc�A�*

loss?z�=n��k       �	�:cEXc�A�*

loss���=�j�       �	�cEXc�A�*

lossd	>�]�<       �	A�dEXc�A�*

lossxt�<N5�O       �	��eEXc�A�*

loss��=m��       �	9*fEXc�A�*

lossq�>v<�)       �	��fEXc�A�*

loss���<�۲G       �	�_gEXc�A�*

loss$*=j��       �	~�gEXc�A�*

lossܸ�=#��       �	��hEXc�A�*

loss���=�w�       �	H6iEXc�A�*

loss(I�=�㕖       �	b�iEXc�A�*

loss�[�<-
       �	�qjEXc�A�*

loss=;;�6t�       �	1kEXc�A�*

loss�?�=��W       �	Q�kEXc�A�*

lossa��=T�w�       �	�>lEXc�A�*

loss��;qy�       �	��lEXc�A�*

loss��<�h<       �	��mEXc�A�*

lossT��<�0�3       �	�'nEXc�A�*

loss��=D5�       �	��nEXc�A�*

lossQ�=�sb_       �	�foEXc�A�*

lossL�q=cx��       �	WpEXc�A�*

loss��=ٵ�       �	8�pEXc�A�*

lossVܷ=] vc       �	�3qEXc�A�*

loss	�=��       �	��qEXc�A�*

loss6��<
�       �	�zrEXc�A�*

loss%��<�g�       �	�sEXc�A�*

loss4�N<8T       �	$�sEXc�A�*

loss4=�	=       �	�XtEXc�A�*

loss��X=�M       �	��tEXc�A�*

loss <=e�/�       �	��uEXc�A�*

loss�M>E#       �	j2vEXc�A�*

lossa�U=��2       �	)�vEXc�A�*

loss��<	�       �	ewEXc�A�*

losso��=�O�%       �	�wEXc�A�*

lossZ3�<���       �	��xEXc�A�*

lossj\�=���       �	�5yEXc�A�*

loss7I�<�(��       �	.�yEXc�A�*

lossz�=X�X7       �	�dzEXc�A�*

loss�>��       �	M�zEXc�A�*

loss��=���       �	z�{EXc�A�*

loss�i=OI��       �	�D|EXc�A�*

loss���=j�o�       �	�|EXc�A�*

loss�M=���@       �	*t}EXc�A�*

lossĒ<��A.       �	�~EXc�A�*

loss���<0;V�       �	��~EXc�A�*

lossI��<���       �	�6EXc�A�*

lossa7�=*7�       �	$�EXc�A�*

loss&��=�!15       �	�i�EXc�A�*

lossM�T>��{       �	��EXc�A�*

losspd�>���       �	s��EXc�A�*

loss�>�|�       �	�A�EXc�A�*

losscb�=�[�       �	߂EXc�A�*

lossՇ�=DE�u       �	�x�EXc�A�*

loss;Ah=N�q       �	��EXc�A�*

loss��=H4��       �	G��EXc�A�*

loss�&�=���       �	 T�EXc�A�*

loss��=�J'       �	,��EXc�A�*

loss6a�=馬�       �	Ҍ�EXc�A�*

loss���<j	ם       �	\�EXc�A�*

loss(8�=P�X�       �	aއEXc�A�*

lossR�E=���       �	qq�EXc�A�*

loss��+=�!}       �	��EXc�A�*

loss_��<h.�       �	�ΉEXc�A�*

loss�/`=��S�       �	��EXc�A�*

loss�H�=L�߻       �	�o�EXc�A�*

lossYj=����       �	rR�EXc�A�*

loss���=P�c�       �	)�EXc�A�*

loss�Š=:��2       �	<ގEXc�A�*

loss��=H�d�       �	Ϥ�EXc�A�*

loss���=報!       �	9~�EXc�A�*

loss�>�t*�       �	
+�EXc�A�*

lossn��<�!M<       �	f�EXc�A�*

loss��<���!       �	���EXc�A�*

loss���<g ��       �	G�EXc�A�*

lossS�>J�_o       �	�ݓEXc�A�*

loss?OR=�E>�       �	#��EXc�A�*

loss1i�=|+�       �	�~�EXc�A�*

loss���=#��       �	cD�EXc�A�*

loss�1�<G��       �	�ܖEXc�A�*

loss?��<�6       �	?��EXc�A�*

loss�ߋ=�^@        �	[x�EXc�A�*

loss�Mr=0�(:       �	w�EXc�A�*

loss���=H�u       �	��EXc�A�*

lossx�Q=��6       �	nܚEXc�A�*

loss՘=Y���       �	��EXc�A�*

loss��J=^[��       �	̵�EXc�A�*

loss�e�<)��c       �	�Q�EXc�A�*

loss8=�pe_       �	/�EXc�A�*

loss-&=�i�^       �	�ܞEXc�A�*

loss���<=ݤ#       �	�EXc�A�*

loss�#=��R}       �	���EXc�A�*

lossř�=o��       �	���EXc�A�*

lossCi�<�
u>       �	ѢEXc�A�*

loss�vH=y.3       �	>��EXc�A�*

loss!�=�ć       �	=|�EXc�A�*

lossx_�<�r=I       �	�U�EXc�A�*

loss�͇=����       �	��EXc�A�*

loss�/k=���       �	�ݦEXc�A�*

loss�\�=(�;       �	���EXc�A�*

loss��A=���       �	�C�EXc�A�*

loss#_>����       �	bI�EXc�A�*

loss\��=���       �	���EXc�A�*

lossD*�=�_�y       �	��EXc�A�*

loss9T�=]�/'       �	���EXc�A�*

loss�l�=+�X       �	x�EXc�A�*

loss
Q=(�5G       �	}[�EXc�A�*

loss�w>>ԭ       �	]�EXc�A�*

lossv9�=+`�       �	(�EXc�A�*

loss�e=,	�?       �	C��EXc�A�*

loss�Te<Ĉ�V       �	6��EXc�A�*

lossO�<���U       �	�N�EXc�A�*

loss-V�=f�@       �	� �EXc�A�*

loss�m�;"1       �	\�EXc�A�*

loss�_�<�>�!       �	И�EXc�A�*

loss ��<tZ�       �	�9�EXc�A�*

loss�>:=�V�6       �	�J�EXc�A�*

loss8 >�8.       �	#ܵEXc�A�*

lossa��=�xA�       �	u˶EXc�A�*

loss��G=��,       �	���EXc�A�*

lossA�g<	c2�       �	4��EXc�A�*

losse0A=��N�       �	��EXc�A�*

lossA��<8B�       �	��EXc�A�*

losss��;�l       �	�-�EXc�A�*

loss�d�=ׄ�       �	�ҼEXc�A�*

loss���<p�7Z       �	���EXc�A�*

loss)��<:���       �	�&�EXc�A�*

loss��=��L~       �	軾EXc�A�*

loss�{�=70Q       �	uX�EXc�A�*

loss&h=/�       �	{��EXc�A�*

loss���=���=       �	���EXc�A�*

lossJ��=�       �	p$�EXc�A�*

loss�~�=]-�       �	M��EXc�A�*

loss=�2=<���       �	�U�EXc�A�*

loss���=�}٨       �	���EXc�A�*

lossn�=�)��       �	���EXc�A�*

lossV�=����       �	L�EXc�A�*

losst+�=�2�       �	���EXc�A�*

lossl#(>c��+       �	�B�EXc�A�*

loss;-=h��X       �	 ��EXc�A�*

loss�՚=��H       �	�s�EXc�A�*

losso�[=èc�       �	!�EXc�A�*

losst��=���       �	Օ�EXc�A�*

losstc�<�C��       �	M0�EXc�A�*

loss�Qk<���9       �	 ��EXc�A�*

lossx�>�)�/       �	%Z�EXc�A�*

lossA��<�e1�       �	79�EXc�A�*

lossX�@=�+U       �	��EXc�A�*

loss�d�=:X��       �	���EXc�A�*

loss#= V�;       �	�G�EXc�A�*

loss��=���       �	���EXc�A�*

loss:_�=��	       �	 }�EXc�A�*

loss?�=��S       �	��EXc�A�*

loss.2�<mL�       �	0��EXc�A�*

loss�`q>#\.?       �	Ct�EXc�A�*

lossEaG=K�,)       �	N�EXc�A�*

loss�y{<|h;       �	���EXc�A�*

loss	r?=��{       �	}A�EXc�A�*

loss�j> 5�       �	w��EXc�A�*

lossW=Q��       �	�v�EXc�A�*

loss�8=j|K>       �	��EXc�A�*

lossE|)>���       �	���EXc�A�*

loss�*�=��_�       �	�V�EXc�A�*

loss��<w�b�       �	���EXc�A�*

loss�� =�?RI       �	���EXc�A�*

lossϞ�<"U�       �	�<�EXc�A�*

lossj��=f�	v       �	���EXc�A�*

loss]"	>��6       �	s�EXc�A�*

lossi�e=,6�V       �	�	�EXc�A�*

loss���=���       �	;��EXc�A�*

loss&�=�_g       �	�<�EXc�A�*

loss��_=󊎍       �	p��EXc�A�*

loss���<@N�       �	�c�EXc�A�*

loss�n�<oD��       �	���EXc�A�*

loss�12=�J��       �	��EXc�A�*

loss�f�=Vjf�       �	�&�EXc�A�*

loss|�S=��3j       �	���EXc�A�*

loss��{>���       �	���EXc�A�*

lossֈV=:wM       �	2�EXc�A�*

lossCAX=��       �	5��EXc�A�*

lossJK�=v��k       �	7��EXc�A�*

loss;��<J��       �	\Z�EXc�A�*

loss��=�6�5       �	y��EXc�A�*

loss{�M={�D�       �	:��EXc�A�*

lossf��==�ʛ       �		p�EXc�A�*

loss«>�*	2       �	�4�EXc�A�*

loss�>�=J�       �	��EXc�A�*

loss���=�u�       �	�h�EXc�A�*

losshP=�Z�G       �	�Q�EXc�A�*

lossl�3=���i       �	�S�EXc�A�*

loss�U�=�q]X       �	>��EXc�A�*

loss��=(�s�       �	���EXc�A�*

lossI@=w���       �	Q1�EXc�A�*

loss��>��tZ       �	���EXc�A�*

loss�n�<�J�       �	�u�EXc�A�*

loss1I�<�3�       �	��EXc�A�*

loss�(7>t�R       �	U��EXc�A�*

loss��>�be       �	hA�EXc�A�*

loss��%=F���       �	+��EXc�A�*

losslb>S*s�       �	�r�EXc�A�*

loss�"G=[�       �	E�EXc�A�*

loss�=��1       �	3��EXc�A�*

loss��=�N�       �	9�EXc�A�*

loss_E�=�T�       �	���EXc�A�*

loss��<]�Y       �	Qj�EXc�A�*

loss��>Q6�       �	a��EXc�A�*

loss��7=�	��       �	;��EXc�A�*

loss���=P��       �	�1�EXc�A�*

loss ��=Ni       �	��EXc�A�*

loss*:>�R�(       �	�d�EXc�A�*

loss�3�=�~�G       �	� �EXc�A�*

loss��J=H���       �	��EXc�A�*

loss7�=�Z�       �	�(�EXc�A�*

loss{Wg=��       �	��EXc�A�*

lossM�=��x�       �	\�EXc�A�*

losso>TQ�       �	��EXc�A�*

loss��=��@%       �	��EXc�A�*

loss���<k�w`       �	�EXc�A�*

loss���<t�t       �	S��EXc�A�*

loss.��=C�       �	�K�EXc�A�*

lossO�>��4#       �	d��EXc�A�*

lossp[<��y       �	���EXc�A�*

loss��=Z�&�       �	�&�EXc�A�*

loss���=�Y�       �	���EXc�A�*

loss�R4=�0��       �	p\�EXc�A�*

loss���=w�       �	a��EXc�A�*

loss�o=�6�F       �	%��EXc�A�*

loss}k,=�<ƴ       �	�,�EXc�A�*

loss6�>&D       �	���EXc�A�*

loss�"�<G��       �	�R�EXc�A�*

lossn��;@�'�       �	���EXc�A�*

losslq�<�=�       �	��EXc�A�*

lossS;/=�z a       �	 FXc�A�*

lossm��<N{�       �	� FXc�A�*

loss��=���       �	�OFXc�A�*

loss��>�_�&       �	d�FXc�A�*

loss.ZO=�z��       �	�}FXc�A�*

lossh�7=�S�D       �	fFXc�A�*

loss`��<�ٟD       �	l�FXc�A�*

loss��=�z�1       �	�HFXc�A�*

loss�X2>�Eu�       �	f�FXc�A�*

lossS9�=c�f       �	rnFXc�A�*

loss�=ߐ�       �	�	FXc�A�*

loss��T=l�+       �	6�FXc�A�*

loss
�6>SJ       �	FFXc�A�*

lossc��<�k�c       �	i�FXc�A�*

loss%��=�h&�       �	�|FXc�A�*

loss_��=����       �	H	FXc�A�*

lossN��=	�i       �	B�	FXc�A�*

loss�H<�S�       �	w�
FXc�A�*

loss�7�<�2��       �	�FXc�A�*

loss]��=�]��       �	�FXc�A�*

loss���=�,c+       �	�PFXc�A�*

loss�Ƞ<N^��       �	��FXc�A�*

loss�t�=5<�       �	Y�FXc�A�*

loss䴬=�g��       �	l?FXc�A�*

loss6��=�Jǎ       �	��FXc�A�*

loss��=P��       �	�FXc�A�*

loss�->r�       �	�)FXc�A�*

loss���=9��       �	|�FXc�A�*

loss���=�|K       �	��FXc�A�*

loss��\=q��       �	�FXc�A�*

loss4-j=J�MV       �	��FXc�A�*

loss})�=l-�       �	�RFXc�A�*

loss���=G�r�       �	��FXc�A�*

lossi#M<�3GS       �	�vFXc�A�*

loss�g*>nv       �	g
FXc�A�*

lossӉR=��k�       �	�FXc�A�*

loss��<�5P       �	�:FXc�A�*

loss�J�<}�?       �	c�FXc�A�*

loss��<'�       �	�kFXc�A�*

loss�	=y�S       �	�FXc�A�*

loss�x�<Z��       �	�FXc�A�*

loss�>��8�       �	�-FXc�A�*

lossFF�=�_2       �	`�FXc�A�*

loss%�&>�2�       �	�aFXc�A�*

loss���=S`\       �	Z�FXc�A�*

loss�P�<�G��       �	*�FXc�A�*

loss�ԁ=�8@z       �	)FXc�A�*

lossFC>��٣       �	x�FXc�A�*

loss Vm=�Pb�       �	�`FXc�A�*

loss
 2=�z�z       �	��FXc�A�*

loss���<n�l       �	&�FXc�A�*

loss(�=1>,       �	�LFXc�A�*

lossJ=�S0�       �	��FXc�A�*

loss��<��n�       �	�u FXc�A�*

loss�Q>�1n�       �	;!FXc�A�*

lossao>ڼ/�       �	��!FXc�A�*

loss���=��       �	ݔ"FXc�A�*

lossr�=^(��       �	k)#FXc�A�*

loss.��=��.       �	D�#FXc�A�*

lossY��<ܥ       �	�W$FXc�A�*

loss?A�=�ҁ       �	%FXc�A�*

loss#�<�H�k       �	��%FXc�A�*

loss2Q.<n��       �	�C&FXc�A�*

loss�O�<I�|�       �	��&FXc�A�*

loss��=a�v       �	2q'FXc�A�*

loss�N=8��       �	�(FXc�A�*

loss��C=���       �	��(FXc�A�*

lossF�o=�� #       �	�2)FXc�A�*

loss�:�=�-l,       �	_�)FXc�A�*

losss��=�f�       �	�d*FXc�A�*

lossco�;��&�       �	� +FXc�A�*

lossdL+>,v��       �	��+FXc�A�*

loss�k�=#       �	2,FXc�A�*

loss���=s��       �	6�,FXc�A�*

loss�=(Ne       �	�e-FXc�A�*

lossaX�<*�y�       �	��-FXc�A�*

loss��<�결       �	��.FXc�A�*

loss��=_У�       �	0*/FXc�A�*

loss���<c� �       �	H�/FXc�A�*

loss���<���       �	�c0FXc�A�*

losse'b>	W?�       �	 
1FXc�A�*

loss�"5>b)��       �	�1FXc�A�*

loss���=�1[/       �	cA2FXc�A�*

loss��=�3~�       �	a�2FXc�A�*

loss#d>�g�       �	)w3FXc�A�*

lossTv�=�y�       �	4FXc�A�*

lossڴ-=)��:       �	��4FXc�A�*

lossM��<UU��       �	<O5FXc�A�*

loss?��=7z�W       �	��5FXc�A�*

loss#=��/       �	.�6FXc�A�*

loss��s=B���       �	�/7FXc�A�*

loss���=�Zƨ       �	�7FXc�A�*

loss��=���x       �	x`8FXc�A�*

losss�=Z���       �	��8FXc�A�*

loss�ES=��z�       �	��9FXc�A�*

loss]E<-D�       �		2:FXc�A�*

lossM�=��       �	K�:FXc�A�*

lossr��=�']       �	�_;FXc�A�*

loss���<1��       �	
�;FXc�A�*

lossF~�='�.�       �	7�<FXc�A�*

lossqO�=Vј�       �	g)=FXc�A�*

loss��b=��:'       �	�=FXc�A�*

loss�EG<���       �	d>FXc�A�*

lossj�8=Ƈ�t       �	 ?FXc�A�*

losso~<��vQ       �	Ʀ?FXc�A�*

loss�؁=���       �	D@FXc�A�*

loss�J=��`?       �	<�@FXc�A�*

lossؠ�=D�Y�       �	rAFXc�A�*

lossή=�+�       �	%BFXc�A�*

loss��> ��       �	կBFXc�A�*

loss�!�=�x�       �	FCFXc�A�*

loss�*�<���       �	��CFXc�A�*

loss	6�=���       �	�qDFXc�A�*

loss!2�<6�       �	B
EFXc�A�*

lossz�"=ka�       �	��EFXc�A�*

loss*�<��9       �	HpFFXc�A�*

lossz4e=�B�,       �	�GFXc�A�*

loss
7o=Zs?�       �	�GFXc�A�*

loss-"I=��B�       �	�5HFXc�A�*

loss��<�� �       �	��HFXc�A�*

loss�5�=ڳ�       �	N_IFXc�A�*

loss4��<��.       �	N	JFXc�A�*

loss���<���\       �	��JFXc�A�*

loss��%;�[8       �	?9KFXc�A�*

lossv�=���q       �	�MLFXc�A�*

loss�PX=Dg��       �	��LFXc�A�*

lossJ�=���       �	�MFXc�A�*

lossc�=ߍ>C       �	rNFXc�A�*

lossM�=O� �       �	M�NFXc�A�*

loss$�>
�O_       �	LQOFXc�A�*

loss�j�<�}$       �	S�OFXc�A�*

loss��=�       �	�PFXc�A�*

loss�7>R��F       �	oQFXc�A�*

loss2�<h�3�       �	ǹQFXc�A�*

loss
�:@��       �	�ORFXc�A�*

lossQy�<��       �	X�RFXc�A�*

loss�B%=��\�       �	)vSFXc�A�*

loss��l<F���       �	?TFXc�A�*

loss���<DϷ�       �	ѯTFXc�A�*

loss}�%=��W�       �	�CUFXc�A�*

loss��&=.R��       �	'�UFXc�A�*

loss�X�;��i       �	kVFXc�A�*

loss��<��       �	��VFXc�A�*

loss&l�:�+uL       �	��WFXc�A�*

lossvH<;k��       �	�)XFXc�A�*

loss���=
'&       �	��XFXc�A�*

lossP��=�臮       �	qTYFXc�A�*

lossَ�<��       �	��YFXc�A�*

loss���=3�/�       �	�}ZFXc�A�*

lossLQ�>I��       �	[FXc�A�*

lossuu<�ϧR       �	3�[FXc�A�*

loss4>�@*�       �	�N\FXc�A�*

loss��=���       �	��\FXc�A�	*

loss�>����       �	߈]FXc�A�	*

loss��J=�2��       �	H1^FXc�A�	*

loss	�<�p�c       �	�T_FXc�A�	*

loss_>��       �	EH`FXc�A�	*

loss��=�[�       �	PRaFXc�A�	*

loss.�=��-.       �	Z�bFXc�A�	*

loss�ab=�|C�       �	�hcFXc�A�	*

lossoz�=��'F       �	L�cFXc�A�	*

lossE>AUM       �	�dFXc�A�	*

loss�.�=�'f5       �	�LeFXc�A�	*

loss�Zj=�XEq       �	J�eFXc�A�	*

lossD�=TӫR       �	Z�fFXc�A�	*

lossj��=F���       �	2 gFXc�A�	*

loss6�4=,ф�       �		�gFXc�A�	*

loss�q=�Em       �	jhFXc�A�	*

loss��>��W�       �	 iFXc�A�	*

loss1k�<f�/�       �	_�iFXc�A�	*

loss���<����       �	��jFXc�A�	*

loss&_�=�ĵ�       �	U1kFXc�A�	*

loss��.=�h̩       �	X�kFXc�A�	*

loss��U="ݏ       �	rllFXc�A�	*

loss�F<��'�       �	�mFXc�A�	*

loss�n�<�a?Y       �	��mFXc�A�	*

loss���=�Xa�       �	NnFXc�A�	*

loss�c
=A#�3       �	jpFXc�A�	*

loss�6�=6���       �	�\qFXc�A�	*

loss��=L埦       �	�qFXc�A�	*

loss�=��CV       �	m�rFXc�A�	*

lossA};=nq��       �	X9sFXc�A�	*

lossꊵ<+�<�       �	r�sFXc�A�	*

lossb<�d�       �	�}tFXc�A�	*

loss=L�<M�)       �	m uFXc�A�	*

loss!��<B�9       �	��uFXc�A�	*

lossݴ�<��+       �	]pvFXc�A�	*

loss�ϱ=�P�       �	EwFXc�A�	*

lossK�=��N�       �	u�wFXc�A�	*

loss��-=�1\       �	�GxFXc�A�	*

lossp�<e1(e       �	W�xFXc�A�	*

loss/ET=s"g�       �	L�yFXc�A�	*

loss�A	>B�T_       �	�&zFXc�A�	*

loss:Ue=6z%       �	�zFXc�A�	*

loss#΋=�o       �	�\{FXc�A�	*

lossA��=nYj�       �	��{FXc�A�	*

loss�^�=ɱzi       �	#�|FXc�A�	*

loss+�<�b=�       �	JE}FXc�A�	*

loss�xV=�½B       �	��}FXc�A�	*

loss턭<��g       �	�~FXc�A�	*

loss�%�=r�D^       �	!FXc�A�	*

lossO".=R˹�       �	=�FXc�A�	*

losssƉ=�+i�       �	_ϢFXc�A�	*

loss�ɛ=`={�       �	�e�FXc�A�	*

loss��=��?       �	#��FXc�A�	*

loss��.= ��       �	ѐ�FXc�A�	*

lossh_�=<��       �	='�FXc�A�	*

loss�?=��r^       �	R��FXc�A�	*

loss�0=����       �	~��FXc�A�	*

lossN.M>B���       �	 �FXc�A�	*

loss�z�=k�ݏ       �	峧FXc�A�	*

loss�s\=�4V�       �	(I�FXc�A�	*

loss �=%B�w       �	�ۨFXc�A�	*

loss��->GJ�9       �	eo�FXc�A�	*

lossSl�=����       �	2�FXc�A�	*

loss��B=�m�       �	͔�FXc�A�	*

loss۝�=�y       �	g,�FXc�A�	*

loss<��(       �	۾�FXc�A�	*

loss4�I<=!Bq       �	�`�FXc�A�	*

loss��^=9��d       �	F_�FXc�A�	*

loss��
>�       �	��FXc�A�	*

losswd=L�        �	���FXc�A�	*

loss=�V>j�b�       �	m�FXc�A�	*

loss��b=���       �	R��FXc�A�	*

lossa)�=���       �	@Q�FXc�A�	*

loss��#=�+E�       �	��FXc�A�	*

loss��=Ǒ�#       �	�v�FXc�A�	*

loss�̂=��\       �	�	�FXc�A�	*

loss�<�<����       �	��FXc�A�	*

loss���=�\86       �	?�FXc�A�	*

loss+=�m<e       �	�ٳFXc�A�	*

loss�͋=��_       �	Wy�FXc�A�	*

lossC=�"       �	��FXc�A�	*

loss�:=@��Y       �	���FXc�A�	*

lossV�=�9       �	�6�FXc�A�	*

loss?��<嗇s       �	�ζFXc�A�	*

lossr͝=x�?t       �	�d�FXc�A�	*

loss��<W��       �	a��FXc�A�	*

losst�`=�7�       �	=��FXc�A�	*

loss6�6>8:       �	�/�FXc�A�	*

loss�|7=̇�       �	�ȹFXc�A�	*

lossg3=I�v       �	�]�FXc�A�	*

loss�]�=�y:�       �	���FXc�A�	*

loss�Tv=�X�l       �	ۢ�FXc�A�	*

loss��=9�"       �	�7�FXc�A�	*

loss��.=�'�]       �	�ϼFXc�A�	*

lossO�>DHg       �	wd�FXc�A�	*

lossoN�<�I~�       �	��FXc�A�	*

loss�'�=����       �	���FXc�A�	*

loss��^=>F�       �	�E�FXc�A�	*

lossݭ�;5���       �	j޿FXc�A�	*

lossb]<)4�       �	D��FXc�A�	*

loss���=͋��       �	H7�FXc�A�	*

loss-�<i�]:       �	oF�FXc�A�	*

loss���=8��       �	,��FXc�A�	*

lossl�^<��       �	���FXc�A�	*

loss=�c<��"       �	S[�FXc�A�	*

loss}2�:S�En       �	���FXc�A�	*

lossz��;.�+       �	|~�FXc�A�	*

loss3�<�Ӎ�       �	��FXc�A�	*

loss
��=kR8       �	��FXc�A�	*

loss�=sO       �	!v�FXc�A�	*

loss(��<K�;5       �	J�FXc�A�	*

loss,D�<�+N       �	���FXc�A�	*

loss���=���p       �	���FXc�A�	*

lossL4=Q��       �	��FXc�A�	*

loss$��=*T�k       �	=��FXc�A�	*

loss�F=�K       �	�[�FXc�A�	*

loss�܀=���       �	���FXc�A�	*

loss�N�=�O3�       �	���FXc�A�	*

loss�c�=�#ݫ       �	�-�FXc�A�	*

loss%\�<�6#T       �	���FXc�A�	*

loss>h�=��V       �	
j�FXc�A�	*

loss }5=����       �	���FXc�A�	*

lossaMm>��ʆ       �	��FXc�A�	*

lossf�<䀹�       �	�+�FXc�A�	*

loss#��<]O�       �	;��FXc�A�	*

loss�^=-:W@       �	-Z�FXc�A�	*

loss#��=�jw�       �	���FXc�A�	*

loss.q�;���s       �	@��FXc�A�	*

loss:�Q<e�Z       �	��FXc�A�
*

lossj͌=p2ۢ       �	���FXc�A�
*

loss�G�=!�L       �		n�FXc�A�
*

loss��@=vZ�`       �	#�FXc�A�
*

loss�I<�U��       �	��FXc�A�
*

lossRb=��       �	�B�FXc�A�
*

loss�`�<�=�       �	���FXc�A�
*

loss謙<<_R�       �	ǀ�FXc�A�
*

loss��=�,l�       �	��FXc�A�
*

loss��=�?�       �	?��FXc�A�
*

lossw��;^J��       �	�9�FXc�A�
*

loss1\�<w}�       �	���FXc�A�
*

loss2�=��K       �	�m�FXc�A�
*

loss�A<�-��       �	��FXc�A�
*

lossSq=�xK�       �	��FXc�A�
*

loss�:J=[��       �	�S�FXc�A�
*

loss̾�<��       �	[��FXc�A�
*

loss8��=�O[�       �	k�FXc�A�
*

loss��=�O�B       �	��FXc�A�
*

loss,_j>��Q       �	���FXc�A�
*

loss;�u</��       �	)^�FXc�A�
*

lossnZ~<�_`<       �	74�FXc�A�
*

loss	�=���$       �	���FXc�A�
*

losshI�=��M       �	���FXc�A�
*

loss�Ԣ<�=�.       �	�$�FXc�A�
*

lossL�=�8       �	���FXc�A�
*

loss���=��e       �	L�FXc�A�
*

loss���<�]��       �	��FXc�A�
*

loss�e�<W��       �	lx�FXc�A�
*

loss�e=�[��       �	.�FXc�A�
*

loss��5=��Q�       �	��FXc�A�
*

loss��^=�"#�       �	kI�FXc�A�
*

loss��R<p��       �	�Z�FXc�A�
*

loss\�9=c��H       �	�E�FXc�A�
*

loss��;�d��       �	��FXc�A�
*

loss5t�< 
�e       �	z�FXc�A�
*

loss�k<<��W       �	�FXc�A�
*

loss ��<!G@�       �	:��FXc�A�
*

lossðm=s���       �	�Q�FXc�A�
*

loss�_�=�<��       �	���FXc�A�
*

loss��>V��       �	��FXc�A�
*

loss<�2=���Z       �	��FXc�A�
*

loss�N=@�2�       �	��FXc�A�
*

lossc�=��       �	�L�FXc�A�
*

lossaJ�=�c�3       �	���FXc�A�
*

lossg�";+��       �	�"�FXc�A�
*

loss���<�� �       �	���FXc�A�
*

loss>�=�Ǡ�       �		��FXc�A�
*

losshe<vQ:�       �	%#�FXc�A�
*

loss�a=�L       �	���FXc�A�
*

lossh|=L�T       �	�`�FXc�A�
*

loss�4=V�*A       �	� �FXc�A�
*

loss1�<�@!�       �	���FXc�A�
*

lossOtE=H���       �	:=�FXc�A�
*

loss�͢<�VZ�       �	���FXc�A�
*

loss ��=�}B       �	VH�FXc�A�
*

lossTx<�(�'       �	a��FXc�A�
*

loss�m�;%��       �	�s�FXc�A�
*

loss8A<
���       �	h�FXc�A�
*

loss��<��4�       �	��FXc�A�
*

loss@��<�F�6       �	�*�FXc�A�
*

lossSF#;
ڸT       �	���FXc�A�
*

loss4p<t��       �	U�FXc�A�
*

loss2y�=�Ǭ�       �	)��FXc�A�
*

loss��=���       �	��FXc�A�
*

loss==�       �	��FXc�A�
*

loss-`=�V��       �	��FXc�A�
*

loss��u=3��       �	�9 GXc�A�
*

loss?ױ<��dq       �	�� GXc�A�
*

loss$��<\�|       �	dGXc�A�
*

loss��<�ct       �	��GXc�A�
*

loss#x=�{�`       �	G�GXc�A�
*

loss1�=�T�       �	�'GXc�A�
*

loss���=���]       �	��GXc�A�
*

loss,=X"�       �	lAGXc�A�
*

loss�F�=����       �	�(GXc�A�
*

lossj��<�6�Q       �	��GXc�A�
*

loss�<�<�z�       �	�[GXc�A�
*

loss
�<��ό       �	��GXc�A�
*

lossΰ?>lբ�       �	КGXc�A�
*

lossq�<�~�       �	{3	GXc�A�
*

lossh�g= ��(       �	��	GXc�A�
*

loss�j�<�/s�       �	r
GXc�A�
*

loss��=�a'       �	�|GXc�A�
*

loss�a\;Xx!�       �	8�GXc�A�
*

lossV�<�V*p       �	��GXc�A�
*

loss��v<�       �	ĲGXc�A�
*

loss: �<[B�g       �	�WGXc�A�
*

lossW��<k�       �	� GXc�A�
*

loss�'�<���       �	�GXc�A�
*

loss�H=���"       �	PRGXc�A�
*

loss���=�b?�       �	/�GXc�A�
*

loss��;����       �	�GXc�A�
*

loss���<�0�       �	�,GXc�A�
*

loss�P�=�uЛ       �	]�GXc�A�
*

loss]i�;+ ;E       �	��GXc�A�
*

lossn��=�u��       �	�FGXc�A�
*

loss��=y���       �	�GXc�A�
*

loss%�k=����       �	��GXc�A�
*

loss��j="O�       �	tDGXc�A�
*

loss�2�;uq       �	��GXc�A�
*

loss%4�:d��       �	�zGXc�A�
*

loss��=�>�       �	�GXc�A�
*

lossiA;=���=       �	ݴGXc�A�
*

loss�n�;�q��       �	�RGXc�A�
*

lossA�<B2�/       �	c�GXc�A�
*

loss��b<dXY�       �	�GXc�A�
*

loss('=�mE       �	�'GXc�A�
*

loss"�<��6�       �	��GXc�A�
*

loss��x=�`�       �	[GXc�A�
*

lossq��<�q֑       �	�GXc�A�
*

loss* >���       �	#�GXc�A�
*

loss_�=����       �	qTGXc�A�
*

loss���;46�       �	5�GXc�A�
*

loss�Y=7��       �	�� GXc�A�
*

loss38.;�Έ�       �	�[!GXc�A�
*

loss~z<��i       �	��!GXc�A�
*

loss�/=t��Q       �	��#GXc�A�
*

loss�"=�N��       �	�,$GXc�A�
*

loss�T>5[,@       �	��$GXc�A�
*

loss@�<�k��       �	oe%GXc�A�
*

loss�%�=^b��       �	 &GXc�A�
*

lossv��<�)r�       �	��&GXc�A�
*

lossck)=��?U       �	.W'GXc�A�
*

loss�As=d5�       �	3�'GXc�A�
*

loss�f<�r�       �	 �(GXc�A�
*

lossLÖ<$�>)       �	M1)GXc�A�
*

loss��>�ޗ�       �	��)GXc�A�
*

loss�=���       �	yZ*GXc�A�*

loss��I<��g�       �	�*GXc�A�*

lossjh
>͙)       �	�+GXc�A�*

lossx:"=[e4S       �	�1,GXc�A�*

loss�p�<�&u�       �	��,GXc�A�*

loss��P=L�5       �	;q-GXc�A�*

loss�^d<�5��       �	.GXc�A�*

loss�7�=�H�       �	��.GXc�A�*

lossԌ=��       �	xG/GXc�A�*

loss�_>^�FX       �	��/GXc�A�*

loss8 >j�_s       �	Bx0GXc�A�*

loss�#=��aw       �	�1GXc�A�*

loss���=�N<       �	;�1GXc�A�*

loss�w-<����       �	?2GXc�A�*

loss�v�<����       �	g�2GXc�A�*

lossV��<:��       �	�s3GXc�A�*

lossR";=Ž�\       �	�4GXc�A�*

loss}z=��V�       �	G�4GXc�A�*

loss�E�=lD{       �	�G5GXc�A�*

loss�=tA�       �	4�5GXc�A�*

loss��=�Y�5       �	�p6GXc�A�*

loss�=V�       �	7GXc�A�*

loss�+=J��       �	�7GXc�A�*

loss�M�<�G�       �	PV8GXc�A�*

lossÏ_=�si�       �	��8GXc�A�*

loss��=ʀ��       �	��9GXc�A�*

lossW��<�%\       �	�):GXc�A�*

loss
j�=��m       �	�	;GXc�A�*

loss��a=e)��       �	ץ;GXc�A�*

loss���=P���       �	�@<GXc�A�*

loss��*>��       �	��<GXc�A�*

loss�d�=)Ѵ�       �	A}=GXc�A�*

loss��<<�.�       �	@>GXc�A�*

losst�x<�CS       �	;�>GXc�A�*

loss���<��
       �	�B?GXc�A�*

lossJ�=h���       �	��?GXc�A�*

loss-��=E�G       �	"q@GXc�A�*

lossW��=؞vg       �	�
AGXc�A�*

loss��U=�xݓ       �	�AGXc�A�*

loss6Y�<V�"�       �	gbBGXc�A�*

loss��<��б       �	@�BGXc�A�*

loss��=�X       �	�CGXc�A�*

loss 4=,:�       �	�*DGXc�A�*

loss*M"=�a�(       �	��DGXc�A�*

lossF��=�K�a       �	LREGXc�A�*

loss��==}<׭       �	�EGXc�A�*

loss*�<���A       �	1xFGXc�A�*

loss}!�;�R�5       �	kGGXc�A�*

loss8�<ug۶       �	:�GGXc�A�*

loss�6.=�2��       �	KYHGXc�A�*

loss�<�gCl       �	��HGXc�A�*

lossz��=S��t       �	��IGXc�A�*

loss�I=����       �	�JGXc�A�*

loss$��="k��       �	�6KGXc�A�*

lossd(�<�_��       �	G�KGXc�A�*

loss �a=Yyt       �	�gLGXc�A�*

loss�܃=��\B       �	>MGXc�A�*

loss��=%�E�       �	&�MGXc�A�*

lossq5=ܧ1$       �	xBNGXc�A�*

loss��=��p       �	��NGXc�A�*

lossH�}=�]�       �	��OGXc�A�*

lossT�<>Ja8       �	G PGXc�A�*

loss�>$=�c�       �	��PGXc�A�*

loss���=HX�       �	�}QGXc�A�*

loss6��<b�
q       �	5RGXc�A�*

loss. =V9Գ       �	��RGXc�A�*

loss/s ==�W       �	�SGXc�A�*

loss��=o߶\       �	d TGXc�A�*

loss�~S=�pi�       �	��TGXc�A�*

loss��=��       �	��UGXc�A�*

loss�&�<�"�H       �	�VGXc�A�*

loss`�V>��qy       �	,fWGXc�A�*

loss��	>w��       �	
XGXc�A�*

lossS�= �q+       �	f�XGXc�A�*

loss,�9=���       �	�dYGXc�A�*

loss���<����       �	�ZGXc�A�*

loss*�<�n�       �	�ZGXc�A�*

loss\ub=X��       �	?U[GXc�A�*

loss�� >�l�5       �	��[GXc�A�*

lossQ1=��޾       �	��\GXc�A�*

loss�t2=�T��       �	�F]GXc�A�*

lossfl.=����       �	��^GXc�A�*

loss?9<���       �	�6_GXc�A�*

loss� Q;�^�l       �	R�_GXc�A�*

loss��v<��<<       �	0�`GXc�A�*

lossag;sd��       �	�%aGXc�A�*

loss/`�;Mٞ8       �	��aGXc�A�*

losst{=Wf�       �	�bGXc�A�*

loss�=��:!       �	0/cGXc�A�*

loss��.<���       �	��cGXc�A�*

loss���=�LX�       �	:zdGXc�A�*

loss���=N��l       �	*eGXc�A�*

loss!��=�B�       �	O�eGXc�A�*

loss�]�<_h4       �	4ifGXc�A�*

lossrn=�r>       �	�gGXc�A�*

loss��;��QT       �	��gGXc�A�*

loss}:�=�+2�       �	�_hGXc�A�*

loss*�<+���       �	�iGXc�A�*

lossT8�=��P�       �	�iGXc�A�*

lossF��;{5       �	JbjGXc�A�*

loss&RX=��Nq       �	|kGXc�A�*

loss%�!=���
       �	s�kGXc�A�*

loss���<�tX�       �	�glGXc�A�*

loss�<���       �	�mGXc�A�*

lossv�S=�Gf�       �	��mGXc�A�*

loss��=�D��       �	wnGXc�A�*

loss|��<��K       �	.oGXc�A�*

loss)0;sF��       �	��oGXc�A�*

loss�-�=� �       �	opGXc�A�*

loss�Z<?���       �	*qGXc�A�*

loss!�=5�,       �	X�qGXc�A�*

lossw�=تN�       �	�orGXc�A�*

loss&��<���       �	�sGXc�A�*

losst٩<�F½       �	��sGXc�A�*

loss�,�=v���       �	0ftGXc�A�*

loss�t=��ʅ       �	�uGXc�A�*

loss�4!<-7�       �	d�uGXc�A�*

loss�q]=�r��       �	�ZvGXc�A�*

loss���=�iH�       �	H�vGXc�A�*

loss� =q��!       �	�wGXc�A�*

loss]v�=���D       �	�CxGXc�A�*

lossOLX=�!M�       �	S�xGXc�A�*

lossr^�=)��       �	>�yGXc�A�*

loss���<�~p       �	�4zGXc�A�*

lossQ�C=y��y       �	��zGXc�A�*

loss,<<B!       �	�{GXc�A�*

loss�E@=���T       �	*9|GXc�A�*

loss���=�O$�       �	�}GXc�A�*

loss��=ɬ��       �	~�}GXc�A�*

loss%��=^r       �	Q~GXc�A�*

lossCW�=1��       �	��~GXc�A�*

loss?��=+�/�       �	��GXc�A�*

loss��<����       �	�C�GXc�A�*

loss}W2<�#!f       �	��GXc�A�*

loss(�!=|_m       �	G��GXc�A�*

loss��=����       �	w-�GXc�A�*

loss�k�=��l       �	�؂GXc�A�*

loss�z[>���       �	|�GXc�A�*

loss\n=���g       �	�$�GXc�A�*

loss�T=�t�       �	.ʄGXc�A�*

lossZ�=MYS�       �	6r�GXc�A�*

loss.�Y=���z       �	Y�GXc�A�*

loss��=�a�       �	&��GXc�A�*

loss�m�=cQE=       �	(E�GXc�A�*

loss2[�<*\ J       �	7߇GXc�A�*

lossst�=t�S       �	�y�GXc�A�*

loss/��<n�T       �	,�GXc�A�*

loss�?�=C�h\       �	_��GXc�A�*

loss%�<���       �	�]�GXc�A�*

loss(�=��       �	{��GXc�A�*

lossq��<D>�p       �	d�GXc�A�*

loss{)+=_��0       �	1��GXc�A�*

loss$�<$�(�       �	uW�GXc�A�*

loss��<9��       �	
�GXc�A�*

loss��<.	2       �	:�GXc�A�*

loss;��=e��\       �	5��GXc�A�*

loss�[�=�F       �	�i�GXc�A�*

loss�ף=���       �	i �GXc�A�*

loss���=�L��       �	���GXc�A�*

loss#�1>`�S�       �	l^�GXc�A�*

loss�N�<��e       �	� �GXc�A�*

loss�Wu=�/       �	q��GXc�A�*

lossM_�=��͵       �	�(�GXc�A�*

loss�^�=�î�       �	?ŔGXc�A�*

loss���:�5       �	���GXc�A�*

loss�W=����       �	HQ�GXc�A�*

loss�} <N�       �	�_�GXc�A�*

lossk�=a\�T       �	���GXc�A�*

loss h�=��       �	Q3�GXc�A�*

lossQ��=���       �	�͙GXc�A�*

loss�Jx==�       �	:��GXc�A�*

loss�A�=��9U       �	sK�GXc�A�*

loss���=��V       �	��GXc�A�*

loss�!=0*�       �	߿�GXc�A�*

lossj�<*�l�       �	�g�GXc�A�*

lossxZ�=�2       �	'��GXc�A�*

loss�>�=�H       �	���GXc�A�*

loss��<_)       �	zT�GXc�A�*

loss��=w�A�       �	e��GXc�A�*

loss��@=�SpC       �	��GXc�A�*

loss=�16h       �	J�GXc�A�*

loss-%�<b�N       �	��GXc�A�*

loss&m=et��       �	ڐ�GXc�A�*

lossAF�<`�	�       �	:A�GXc�A�*

lossȚ�=�X2       �	��GXc�A�*

loss� �<A��Q       �	���GXc�A�*

loss��`=I�?'       �	7�GXc�A�*

loss�!=d��       �	��GXc�A�*

lossܒ�=G�8/       �	ǃ�GXc�A�*

loss�Qy<���       �	#�GXc�A�*

loss�0<{��K       �	=��GXc�A�*

loss��<NQ*       �	�Y�GXc�A�*

lossS�7=`�       �	��GXc�A�*

loss���<�oU�       �	'��GXc�A�*

loss.p�=�F�       �	��GXc�A�*

lossσ�=�C(       �	�ªGXc�A�*

loss��<<��       �	�W�GXc�A�*

loss�P�=>W9       �	�GXc�A�*

loss�@�<F���       �	��GXc�A�*

loss�=��	       �	��GXc�A�*

loss4|%=u,*       �	�GXc�A�*

loss3=����       �	E�GXc�A�*

loss���=8�L�       �	7ݮGXc�A�*

loss�Ŏ=�34       �	�v�GXc�A�*

loss��7=^���       �	�GXc�A�*

lossS�^=���>       �	x��GXc�A�*

loss�Z>�f��       �	�R�GXc�A�*

lossK�=M���       �	t�GXc�A�*

lossR k=����       �	#��GXc�A�*

loss��F<n���       �	;�GXc�A�*

loss舢<��4�       �	Ȳ�GXc�A�*

loss�ƈ=���D       �	�F�GXc�A�*

loss��=�G�T       �	�ߴGXc�A�*

lossL�3<`-��       �	9�GXc�A�*

lossW�/=	��m       �	��GXc�A�*

losszH�<��$W       �	���GXc�A�*

loss#�=�#	�       �	�Q�GXc�A�*

loss�$�=���       �	�GXc�A�*

loss���=4�4�       �	U��GXc�A�*

loss���<#���       �	N%�GXc�A�*

lossά�=c�       �	׽�GXc�A�*

loss�++<��O�       �	�Q�GXc�A�*

loss�h�<�s��       �	��GXc�A�*

loss^=n��       �	,�GXc�A�*

loss�#6=]��'       �	M�GXc�A�*

loss8�<ZAK       �	?��GXc�A�*

lossv��<Mg��       �	A�GXc�A�*

loss��a=� �       �	�;�GXc�A�*

losst#G=KeJ       �	9ԿGXc�A�*

loss� 0=̧�       �	�r�GXc�A�*

loss�X<�f�       �	o�GXc�A�*

loss��<�@�       �	���GXc�A�*

loss��<+��
       �	�G�GXc�A�*

loss(�=*Y��       �	l��GXc�A�*

lossJ`(=5K�       �	���GXc�A�*

loss��=����       �	P4�GXc�A�*

loss�q�= ��       �	���GXc�A�*

loss�t�<�q:       �	Xu�GXc�A�*

lossҭ3=��       �	��GXc�A�*

loss�e�=Eb@�       �	|��GXc�A�*

loss!ɓ<�X�       �	�Z�GXc�A�*

loss�V=����       �	�GXc�A�*

loss%-�<i�C       �	���GXc�A�*

loss���=@��       �	�O�GXc�A�*

loss��<��F�       �	|��GXc�A�*

loss�о<dP�.       �	���GXc�A�*

lossR7=�*Mv       �	K�GXc�A�*

loss��>��m0       �	H��GXc�A�*

lossNT`=m�N�       �	X��GXc�A�*

loss?M+=�b�       �	�G�GXc�A�*

loss��=g^ /       �	���GXc�A�*

loss�<�)�       �	��GXc�A�*

loss�,=*Y�U       �	�&�GXc�A�*

loss���<A+&�       �	x��GXc�A�*

lossHP�;��       �	�N�GXc�A�*

loss4]=�<3�       �	\��GXc�A�*

lossRc�=+{�       �	��GXc�A�*

loss��t=V%�       �	,+�GXc�A�*

loss�Q�<?�j�       �	���GXc�A�*

loss�&�;9��o       �	k�GXc�A�*

loss��<Δ�2       �	 �GXc�A�*

loss�i<�iu       �	Q��GXc�A�*

loss,.$;�~�       �	�J�GXc�A�*

loss�l>E���       �	���GXc�A�*

loss��u<����       �	T��GXc�A�*

lossE=�4�       �	)"�GXc�A�*

loss@��=S(�       �	Q��GXc�A�*

loss)`�;&���       �	�}�GXc�A�*

loss��[=x(g�       �	��GXc�A�*

loss��^=tHP?       �	o��GXc�A�*

loss��=���       �	d�GXc�A�*

loss�{�<���.       �	���GXc�A�*

loss��
>BQz       �	k��GXc�A�*

loss�=����       �	�9�GXc�A�*

loss�t=h]       �	���GXc�A�*

loss�5y<�U�       �	f��GXc�A�*

lossT&E=��-�       �	B]�GXc�A�*

lossoz*>~iM       �	r��GXc�A�*

lossJ�m<v_       �	��GXc�A�*

loss�-=Hs�       �	�.�GXc�A�*

loss�is=�6�,       �	���GXc�A�*

lossC�<�M)       �	���GXc�A�*

lossi1=Z.��       �	�Z�GXc�A�*

loss �<�EA"       �	���GXc�A�*

loss��=�F�u       �	t��GXc�A�*

loss��F<�f{       �	/1�GXc�A�*

loss�j3=dS�T       �	}��GXc�A�*

loss�<�#.�       �	���GXc�A�*

lossa�a='���       �	�*�GXc�A�*

lossꒄ<X5.�       �	��GXc�A�*

loss��|=�4�"       �	W!�GXc�A�*

lossꟑ<7���       �	���GXc�A�*

lossl�T=>��       �	��GXc�A�*

lossh��<&��&       �	�4�GXc�A�*

loss�q�<��ؙ       �	���GXc�A�*

loss�N=�2�$       �		q�GXc�A�*

loss�Q�=�V8       �	h�GXc�A�*

loss�<F=�L�-       �	���GXc�A�*

lossF�c<AM��       �	�:�GXc�A�*

loss���<?[Oy       �	/��GXc�A�*

loss͆L<z���       �	�r�GXc�A�*

loss�^r=���       �	��GXc�A�*

loss�FA=��x       �	���GXc�A�*

loss��<3Ԣ       �	�<�GXc�A�*

loss��<О��       �	��GXc�A�*

lossE��;���       �	�u�GXc�A�*

loss��=p�>�       �	��GXc�A�*

loss�<�~�       �	&8�GXc�A�*

loss8$�=�ʶ�       �	���GXc�A�*

lossx�<�L�3       �	(|�GXc�A�*

loss@#N<aT��       �	��GXc�A�*

lossF	�<��-�       �	���GXc�A�*

loss�:=�K5       �	o~�GXc�A�*

loss]��<���       �	P�GXc�A�*

loss)�H<�Ms       �	v��GXc�A�*

loss��l:��*�       �	�_�GXc�A�*

loss�<4�;:       �	���GXc�A�*

loss��<k/~       �	���GXc�A�*

loss���;7 �l       �	�4�GXc�A�*

loss�R�<uL��       �	p��GXc�A�*

loss�ݏ=�,(�       �	�h�GXc�A�*

loss?�7>x�       �	�C�GXc�A�*

loss�:<�W�|       �	���GXc�A�*

loss$8<:��       �	�o�GXc�A�*

loss�A�<I�.�       �	�GXc�A�*

loss�4�;���       �	P��GXc�A�*

loss��:o�:�       �	�= HXc�A�*

loss��y;��       �	�� HXc�A�*

lossKނ<=���       �	lHXc�A�*

lossM�6<�X��       �	�4HXc�A�*

loss��=,<�M       �	�HXc�A�*

lossO~s=ն>�       �	�aHXc�A�*

lossM��=����       �	��HXc�A�*

loss��8�j��       �	�HXc�A�*

lossTC�:�gO�       �	�HXc�A�*

loss�4�:.��j       �	��HXc�A�*

loss���<�?��       �	}?HXc�A�*

loss ��=���       �	��HXc�A�*

loss{Ud<����       �	8�HXc�A�*

loss�G:����       �	4,HXc�A�*

loss��=�#�$       �	h�HXc�A�*

loss�G.>+@       �	en	HXc�A�*

lossɟ=!{�       �	N
HXc�A�*

lossȓ+>�/�2       �	2�
HXc�A�*

lossM��<�c)�       �	�KHXc�A�*

loss|Հ=!�>d       �	�HXc�A�*

lossEՔ<����       �	��HXc�A�*

loss�/%<1(�       �	Y�HXc�A�*

loss���=���       �	�yHXc�A�*

loss#��=��n       �	+HXc�A�*

lossRط<é��       �	T�HXc�A�*

loss���<L�¶       �	�AHXc�A�*

loss��r<!P�o       �		�HXc�A�*

loss/�1=�4(       �	_{HXc�A�*

loss��B=����       �	fHXc�A�*

losso�<W���       �	`�HXc�A�*

lossX"l<�V�       �	�RHXc�A�*

loss#ԣ=:c��       �	QHXc�A�*

loss�^�=���       �	x�HXc�A�*

loss��t=QS^       �	$HXc�A�*

loss3�=lG��       �	T�HXc�A�*

lossm=���u       �	4�HXc�A�*

loss��#<x���       �	�HXc�A�*

loss͓)=�1zX       �	l�HXc�A�*

loss��H<*e�       �	�vHXc�A�*

loss��:w��)       �	RHXc�A�*

loss�+^<�y�?       �	r�HXc�A�*

lossz��<R��       �	�5HXc�A�*

loss���<�G��       �	P�HXc�A�*

loss.ap=�m�       �	9|HXc�A�*

loss
 o=mt6�       �	�HXc�A�*

loss��=r�T�       �	�HXc�A�*

lossH��<����       �	�SHXc�A�*

loss�26=���       �	F�HXc�A�*

losst8<����       �	2�HXc�A�*

loss�ZB<z��       �	T9 HXc�A�*

loss�< =��E       �	R� HXc�A�*

loss��z;�;�|       �	�f!HXc�A�*

loss��<�j%�       �	�S"HXc�A�*

loss�pn=-i�       �	8N#HXc�A�*

loss�ڒ=T�0�       �	�#HXc�A�*

loss��w=YQ�6       �	<�$HXc�A�*

loss���<�GW       �	#I%HXc�A�*

loss��<b��;       �	�%HXc�A�*

lossE�<�!=�       �	��&HXc�A�*

loss S�<TmX       �	:>)HXc�A�*

loss��<F�
#       �	��)HXc�A�*

losswi�=����       �	��*HXc�A�*

loss��=ٿ��       �	%+HXc�A�*

loss�}=ک��       �	��+HXc�A�*

loss�%5=�8        �	Ɗ,HXc�A�*

loss�eH;��       �	�.-HXc�A�*

loss��<=~%       �	�-HXc�A�*

loss��=����       �	�GHXc�A�*

loss�a=�u4       �	W�GHXc�A�*

lossII=�!D�       �	-HHXc�A�*

loss�#=�ˁ�       �	��HHXc�A�*

loss][�<3���       �	�_IHXc�A�*

loss�ޯ<��       �	
�IHXc�A�*

loss���<s3�W       �	]�JHXc�A�*

lossi�
=#@�       �	�KHXc�A�*

loss��>��F       �	�KHXc�A�*

losse�<s��v       �	�LLHXc�A�*

loss���;�#��       �	�LHXc�A�*

loss)�<D�E       �	T�MHXc�A�*

loss�-�<|��'       �	4-NHXc�A�*

lossڒ�<1_F�       �	z�NHXc�A�*

loss0=F"�       �	5_OHXc�A�*

loss}�b=��E       �	r�OHXc�A�*

loss��;=ѧ�       �	�PHXc�A�*

lossNY�<�O       �	�3QHXc�A�*

loss��<6�       �	6�QHXc�A�*

loss E�<����       �	�cRHXc�A�*

loss��E=+7�9       �	��RHXc�A�*

loss�=�>�8�       �	��SHXc�A�*

loss�lX<W6j�       �	�(THXc�A�*

lossE8�=�ҷ\       �	�THXc�A�*

loss?�`<�@��       �	XSUHXc�A�*

lossq��<��M7       �	l�UHXc�A�*

loss��=>f       �	sVHXc�A�*

lossI�=�l       �	�WHXc�A�*

loss�X=���       �	ԵWHXc�A�*

lossl�<�ĭ�       �	�IXHXc�A�*

loss��<ٌ9�       �	��XHXc�A�*

loss���<�:�        �	�YHXc�A�*

loss  �<�ٞ       �	�1ZHXc�A�*

loss%*^=D�Po       �	��ZHXc�A�*

lossԩ�<tj��       �	�d[HXc�A�*

loss�g�=q-/       �	�!\HXc�A�*

loss:e;��W�       �	��\HXc�A�*

loss*��=%�8F       �	�m]HXc�A�*

losst	>ST2�       �	�^HXc�A�*

loss�<�=cO�       �	��^HXc�A�*

loss�t=��l%       �	�N_HXc�A�*

loss�`d<?3       �	�_HXc�A�*

lossqP�;{�)       �	��`HXc�A�*

lossӽ=��_       �	g*aHXc�A�*

loss�{!=�{�       �	h�aHXc�A�*

loss,9�=p�s�       �	4fbHXc�A�*

loss���<��:�       �	�cHXc�A�*

loss`=<��       �	�cHXc�A�*

lossA19<\�!+       �	vdHXc�A�*

lossYz�:�y       �	�^eHXc�A�*

losseb<�Q��       �	�7fHXc�A�*

lossI�=E���       �	p�fHXc�A�*

loss��;��Y�       �	J~gHXc�A�*

loss��>�K       �	/hHXc�A�*

loss��<��n_       �	;�hHXc�A�*

loss�`�;s[E       �	niHXc�A�*

loss���;����       �	�	jHXc�A�*

lossL��<x�Q�       �	��jHXc�A�*

loss��2=dF(       �	�4kHXc�A�*

loss*��=���E       �	��kHXc�A�*

loss69>A�       �	ƣlHXc�A�*

loss�� =����       �	VmHXc�A�*

loss�+�;m�-Z       �	��mHXc�A�*

loss�4=I�\       �	:�nHXc�A�*

lossc��<T���       �	-oHXc�A�*

loss2n�<�D~�       �	��oHXc�A�*

loss<d�<��       �	AfpHXc�A�*

lossm\N=�}       �	;�pHXc�A�*

loss��>�!�       �	ϽqHXc�A�*

loss��=��+y       �	�TrHXc�A�*

loss�7=��#       �	��rHXc�A�*

lossm��<ϥ�       �	�sHXc�A�*

loss|H�<�U?       �	%;tHXc�A�*

losst`&=�ke;       �	��tHXc�A�*

lossc>]=^��R       �	�yuHXc�A�*

lossɕ�<V� j       �	nvHXc�A�*

loss���=����       �	E�vHXc�A�*

loss}>�=��       �	NwHXc�A�*

loss=�K<�`C�       �	��wHXc�A�*

lossx�,=�TA       �	.txHXc�A�*

losssy�=�1�g       �	#yHXc�A�*

loss �=���X       �	��yHXc�A�*

loss�=C���       �	�AzHXc�A�*

loss՝�;;�       �	(�zHXc�A�*

loss��7<*��       �	�m{HXc�A�*

lossؼ�<�tp�       �	|HXc�A�*

loss���<��&       �	��|HXc�A�*

loss��=x��       �	1}HXc�A�*

loss�=v��       �	H�}HXc�A�*

loss�VV<��       �	�~HXc�A�*

loss�]u=�U��       �	�'HXc�A�*

loss H�< 9^�       �	��HXc�A�*

loss%�k<<�       �	�f�HXc�A�*

loss8v�<��       �	��HXc�A�*

loss C>�yy       �	^��HXc�A�*

loss�59<�c��       �	׆�HXc�A�*

lossM�<���       �	2�HXc�A�*

loss$N=3d��       �	���HXc�A�*

lossGL>� ߒ       �	]Q�HXc�A�*

loss�Gn<SuJ       �	d�HXc�A�*

loss	��<6Y�       �	_}�HXc�A�*

loss8%�;i���       �	�HXc�A�*

loss� �=ǰ�@       �	!��HXc�A�*

loss֝3;s<��       �	"P�HXc�A�*

loss�f�<��il       �	��HXc�A�*

loss�CG=r�)       �	���HXc�A�*

lossvR<�Q|       �	V(�HXc�A�*

lossƹY<](�        �	Y��HXc�A�*

lossB%>Cr�N       �	V�HXc�A�*

lossS �<
�0�       �	�HXc�A�*

loss3��=��.9       �	k}�HXc�A�*

loss���<�KEY       �	��HXc�A�*

lossW5�<]�J?       �	� �HXc�A�*

lossx�^<���       �	���HXc�A�*

loss폈=a��       �	J�HXc�A�*

loss���;�Z�        �	�ގHXc�A�*

lossᰘ<�-�b       �	?t�HXc�A�*

loss�J4=hq       �	x
�HXc�A�*

loss?._=(��t       �	��HXc�A�*

loss�|=�A&       �	�;�HXc�A�*

loss6�,=�@�       �	�ӒHXc�A�*

loss�=�x�)       �	�i�HXc�A�*

lossi�M=#�>�       �	}�HXc�A�*

loss}�=ehq�       �	穀HXc�A�*

loss�^�;pX       �	�G�HXc�A�*

lossW�,=���       �	i�HXc�A�*

loss%��;C�^�       �	1z�HXc�A�*

losse@@<ۗ�       �	
�HXc�A�*

loss=f�=�m�       �	a��HXc�A�*

loss�=���m       �	GX�HXc�A�*

loss��<w�+[       �	0�HXc�A�*

lossK�<%#6�       �	���HXc�A�*

loss	~�<���       �	�)�HXc�A�*

loss�$�<�}m       �	�ĚHXc�A�*

loss#ZY=��>�       �	]�HXc�A�*

loss�u<V}_       �	���HXc�A�*

loss�/�=t��I       �	��HXc�A�*

lossxp�<E���       �	q9�HXc�A�*

loss `;���       �	�ߝHXc�A�*

loss�c�<��,/       �	8��HXc�A�*

loss6!U;���5       �	� �HXc�A�*

loss�)�<я�?       �	$��HXc�A�*

lossM7\=��fZ       �	nM�HXc�A�*

loss?�=i��       �	W�HXc�A�*

loss�?�;��3S       �	ԁ�HXc�A�*

loss� =n�z�       �	i�HXc�A�*

lossT�`<r�N       �	$��HXc�A�*

loss};�� �       �	MM�HXc�A�*

loss"�<w㶴       �	 �HXc�A�*

loss�} <�\��       �	��HXc�A�*

loss�|7;����       �	�&�HXc�A�*

lossƊ!=���
       �	�˥HXc�A�*

loss��<ض�}       �	n�HXc�A�*

loss��r=DU�Q       �	?��HXc�A�*

lossx��=����       �	ࡧHXc�A�*

loss��;	Y.       �	?�HXc�A�*

loss�q,=x`�       �	s֨HXc�A�*

loss��]=3%�       �	�p�HXc�A�*

lossT��=�\O�       �	�	�HXc�A�*

loss�X<9���       �	欪HXc�A�*

loss-!=(���       �	_F�HXc�A�*

loss�ɗ<�,�)       �	7߫HXc�A�*

loss���<��       �	z�HXc�A�*

loss�Y;��F�       �	j�HXc�A�*

lossv�=�es       �	X��HXc�A�*

loss�Q<���       �	�C�HXc�A�*

lossv�Q<mĞ�       �	+ٮHXc�A�*

lossS�<�d$       �	7n�HXc�A�*

loss�l==��c�       �	��HXc�A�*

loss��"=�L8	       �	�HXc�A�*

lossn�|=67       �	x)�HXc�A�*

loss�ٴ;S*�       �	���HXc�A�*

loss��;�Bu�       �	dZ�HXc�A�*

lossĝ�<!�       �	��HXc�A�*

lossf�/<����       �	N��HXc�A�*

loss*�|<0cR�       �	�-�HXc�A�*

loss�}={�       �	��HXc�A�*

loss<c=���       �	Y�HXc�A�*

loss/@�<�D�       �	��HXc�A�*

loss��5;�5�\       �	:��HXc�A�*

loss���9�a"�       �	�%�HXc�A�*

lossR�@=��%�       �	���HXc�A�*

loss���<;��       �	P�HXc�A�*

loss
�D<Eh@       �	��HXc�A�*

losslOv=]�(�       �	��HXc�A�*

lossֺ�;]�
       �	�&�HXc�A�*

loss{=Ƶ�2       �	2ɺHXc�A�*

loss�";��\       �	�c�HXc�A�*

lossO�9=@�|       �	��HXc�A�*

loss��^; }C�       �	B��HXc�A�*

loss-�=��c       �		3�HXc�A�*

loss(<@=���       �	yɽHXc�A�*

lossʋ�<q�       �	�p�HXc�A�*

loss�!b=?u�W       �	�
�HXc�A�*

loss�G�;,��       �	£�HXc�A�*

loss�z<�B�       �	C�HXc�A�*

loss"(�=oRwG       �	M��HXc�A�*

lossD.<�mr�       �	p�HXc�A�*

loss��=�[x       �	��HXc�A�*

loss�p�<W"��       �	Z��HXc�A�*

lossӂZ<]7k�       �	�7�HXc�A�*

loss�r�=1p��       �	F��HXc�A�*

loss�"=.��^       �	�j�HXc�A�*

lossf�7=�#�       �	*��HXc�A�*

loss��<A0ġ       �	%��HXc�A�*

loss�^�=���       �	#0�HXc�A�*

loss��==m�       �	h��HXc�A�*

loss��>�9'�       �	h�HXc�A�*

lossEY�<g{{s       �	*��HXc�A�*

loss��~=G�       �	;��HXc�A�*

loss!~$<r��       �	-&�HXc�A�*

loss�3�<�9�       �	���HXc�A�*

loss�Ք<��dW       �	�Z�HXc�A�*

lossج�<
q0�       �	���HXc�A�*

loss,C�<���       �	\��HXc�A�*

loss�1L=X��[       �	$*�HXc�A�*

lossJ�>Wzo       �	���HXc�A�*

lossL�	>F35o       �	q�HXc�A�*

lossϟ�=K���       �	M�HXc�A�*

loss��.>�%       �	��HXc�A�*

loss���<��\�       �	�Q�HXc�A�*

loss�_�<�&À       �	���HXc�A�*

lossZ�<�ì�       �	��HXc�A�*

loss�E�=�]ċ       �	2:�HXc�A�*

lossN.�<F�       �	w��HXc�A�*

loss�]�=`�       �	�k�HXc�A�*

loss��<6��E       �	:�HXc�A�*

lossɫ;�Uy'       �	_��HXc�A�*

loss�1=��M       �	�,�HXc�A�*

loss�0�<�	L�       �	���HXc�A�*

lossIz�;�l�       �	�R�HXc�A�*

loss�<��       �	���HXc�A�*

loss%ec<h)�G       �	_}�HXc�A�*

loss���<'s97       �	�HXc�A�*

loss֗o=�zV2       �	ʤ�HXc�A�*

loss���<cآQ       �	�I�HXc�A�*

lossf=G�c�       �	���HXc�A�*

loss��t=�[��       �	*r�HXc�A�*

loss���<�]�+       �	S�HXc�A�*

loss�|�;���       �	��HXc�A�*

loss�b�;ל��       �	�-�HXc�A�*

lossX2=�*`�       �	���HXc�A�*

loss�R�=sj��       �	|c�HXc�A�*

lossu�<BjJ       �	���HXc�A�*

loss�N�<lXDI       �	���HXc�A�*

lossj=ȝXz       �	�#�HXc�A�*

lossJz<���       �	���HXc�A�*

loss�o)=���       �	�_�HXc�A�*

loss�^�=9��       �	a��HXc�A�*

loss�:=I~�       �	��HXc�A�*

lossA(=��9       �	�;�HXc�A�*

loss�a�=w���       �	���HXc�A�*

lossT~�<�eH�       �	Ow�HXc�A�*

loss2X�:~�       �	m�HXc�A�*

loss2��;��h       �	>��HXc�A�*

loss�KW<ې5:       �	�P�HXc�A�*

loss\}<5���       �	S��HXc�A�*

loss��I=&�ۜ       �	�~�HXc�A�*

loss�q<$Ә�       �	��HXc�A�*

loss�o2=	I�x       �	 ��HXc�A�*

loss(��=�@�       �	kE�HXc�A�*

loss�Ū<'_V       �	��HXc�A�*

losso��=�]       �	p�HXc�A�*

loss�Z=�y��       �	�	�HXc�A�*

loss�R�<�m>[       �	��HXc�A�*

lossc,�<
f��       �	>�HXc�A�*

lossD��=�6u       �	k��HXc�A�*

lossw� <1�       �	�l�HXc�A�*

loss�X�<6y֒       �	 �HXc�A�*

loss��8=�Q��       �	���HXc�A�*

loss��<y�       �	A.�HXc�A�*

loss�\�<���p       �	��HXc�A�*

loss͔�<[m|       �	�T�HXc�A�*

loss;U�=s�Y       �	��HXc�A�*

loss�@=���V       �	�{�HXc�A�*

loss&��<��\       �	��HXc�A�*

loss��p=��+�       �	c��HXc�A�*

lossaP�;.��       �	�]�HXc�A�*

loss ��<V�(       �	6�HXc�A�*

loss�=~��       �	���HXc�A�*

lossxH�;�ǜH       �	�D�HXc�A�*

loss*��<\��       �	C��HXc�A�*

loss1~�=2텫       �	���HXc�A�*

loss�r><����       �	�!�HXc�A�*

lossw;�=�p��       �	˻�HXc�A�*

lossJ`�<3�~       �	U�HXc�A�*

loss�'n=�b�       �	@��HXc�A�*

loss�<-��       �	s��HXc�A�*

loss,�<>��t       �	C�HXc�A�*

lossTHV<�<0:       �	0�HXc�A�*

loss[�	<O��       �	���HXc�A�*

loss�\=���`       �	�p�HXc�A�*

loss�x�<�6Y�       �	��HXc�A�*

lossZ�q;4�#�       �	��HXc�A�*

loss�߄<���(       �	O?�HXc�A�*

loss���=���       �	���HXc�A�*

loss�~~<.�j�       �	l�HXc�A�*

loss��=b%�       �	��HXc�A�*

loss^O�=:�%       �	,��HXc�A�*

loss�T
>��x8       �	*9 IXc�A�*

loss=�<��^       �	 � IXc�A�*

loss�$=���       �	��IXc�A�*

loss�ȏ;���       �	�+IXc�A�*

lossV>W\�       �	�IXc�A�*

loss���=Ժ�       �	PsIXc�A�*

lossN��=%��j       �	�IXc�A�*

loss�;���       �	)�IXc�A�*

lossv�T<p�է       �	�SIXc�A�*

lossQG�<-V[p       �	��IXc�A�*

loss��=�O�       �	ΥIXc�A�*

loss���;mT��       �	�SIXc�A�*

lossd
�<�	�       �	��IXc�A�*

lossL=���       �	
�IXc�A�*

loss�#}<m��       �	W"	IXc�A�*

loss{�;���       �	I�	IXc�A�*

loss\�=�rQ�       �	�O
IXc�A�*

loss�?<�ԃ1       �	��
IXc�A�*

loss��=�P�M       �	�tIXc�A�*

loss�A�<i�       �	W
IXc�A�*

loss%�<)��       �	�IXc�A�*

lossϥ�<ʻ��       �	��IXc�A�*

lossà=��X)       �	�*IXc�A�*

lossVƴ<�r��       �	�IXc�A�*

lossck<e���       �	_IXc�A�*

loss�Z�<v䈌       �	��IXc�A�*

loss�q=��J       �	��IXc�A�*

loss .�=�w�       �	�%IXc�A�*

loss$��<L�*`       �	v�IXc�A�*

loss�/�<��J       �	iIXc�A�*

losstq�=:Y��       �	{IXc�A�*

loss�L=T!\r       �	��IXc�A�*

lossH�=��       �	�RIXc�A�*

loss��<0놺       �	�IXc�A�*

loss��[<��       �	<IXc�A�*

lossa^�=NB��       �	ȴIXc�A�*

loss��=�rJ�       �	�PIXc�A�*

loss�F[=�̖       �	��IXc�A�*

loss߰�=�Ya       �	��IXc�A�*

loss��%=,d�       �	f0IXc�A�*

losso�u;H���       �	��IXc�A�*

loss�.�<a�u       �	x{IXc�A�*

loss��<���       �		IXc�A�*

loss�{�=����       �	N�IXc�A�*

loss85�=��K       �	�OIXc�A�*

loss �$>���p       �	�IXc�A�*

loss�$l=�-��       �	܂IXc�A�*

loss�*<%���       �	�IXc�A�*

lossM<�=����       �	6�IXc�A�*

lossjL=�S�       �	Aa IXc�A�*

lossS�=,�       �	�� IXc�A�*

loss`�H=���       �	��!IXc�A�*

loss��<��w`       �	)#"IXc�A�*

loss]�>�;       �	�"IXc�A�*

lossMz�<��2�       �	�z#IXc�A�*

lossG=�1�'       �	�$IXc�A�*

loss�8
<1��       �	�%IXc�A�*

loss2�P<��6       �	��%IXc�A�*

lossC�;���       �	��&IXc�A�*

lossl�,=�~�       �	��'IXc�A�*

lossf�<��       �	'J(IXc�A�*

lossf��=J�[B       �	�r)IXc�A�*

loss�W_<��λ       �	c**IXc�A�*

loss�t[=�fbU       �	��*IXc�A�*

loss�,>�GT       �	�+IXc�A�*

loss~�=:��x       �	�z,IXc�A�*

loss��<x
2&       �	�-IXc�A�*

losst�=���       �	F�-IXc�A�*

lossj�<��       �	��.IXc�A�*

loss�O�<oUN       �	��/IXc�A�*

loss�;�<��       �	�E0IXc�A�*

loss�5�<	�Z       �	;1IXc�A�*

losstJ�<�^�        �	%�1IXc�A�*

loss��u=��       �	Fa2IXc�A�*

loss���;R�       �	U�2IXc�A�*

loss��2<KC\@       �	��3IXc�A�*

loss��y=���       �	�o4IXc�A�*

loss�	(=
d��       �	�5IXc�A�*

loss��=�I�       �	�5IXc�A�*

loss.a�<)�"       �	U26IXc�A�*

lossER�=��t       �	��6IXc�A�*

loss/�<u:��       �	��7IXc�A�*

loss�n�<r$N�       �	�48IXc�A�*

loss�D�=T"Y�       �	��8IXc�A�*

loss���;!���       �	�a9IXc�A�*

loss�;�H�       �	A�9IXc�A�*

loss���=zO�4       �	��:IXc�A�*

lossD8=��6�       �	�$;IXc�A�*

loss!��=����       �	��;IXc�A�*

loss7��;?�Ǐ       �	�^<IXc�A�*

loss���<�"�       �	� =IXc�A�*

loss��0<�$Q3       �	��=IXc�A�*

loss��l<*�d�       �	�:>IXc�A�*

lossf��;�P       �	A�>IXc�A�*

loss�p�=o/       �	&n?IXc�A�*

loss8�2=����       �	�@IXc�A�*

loss�6�=L��%       �	��@IXc�A�*

loss~�:��n       �	^LAIXc�A�*

loss�8<��       �	��AIXc�A�*

loss�G}<�ح�       �	n�BIXc�A�*

loss�
+=)��0       �	y$CIXc�A�*

loss�*=�!T       �	��CIXc�A�*

loss-��<Y��       �	�kDIXc�A�*

loss$�)=�*       �	\EIXc�A�*

loss���<
�
       �	#FIXc�A�*

loss.��=s�"       �	e�FIXc�A�*

loss�7=�D1       �	1^GIXc�A�*

loss���<o6Dw       �	(�GIXc�A�*

losss��<~��       �	 �HIXc�A�*

loss���=Z��F       �	
1IIXc�A�*

loss��K=���       �	�IIXc�A�*

loss(�M=eq�s       �	PpJIXc�A�*

loss=�t��       �	�	KIXc�A�*

loss[pd=�չl       �	��KIXc�A�*

loss�=�%8       �	v5LIXc�A�*

loss}�m=�R�       �	B#MIXc�A�*

loss�2�=��       �	�NIXc�A�*

loss6��<-Ғe       �	��OIXc�A�*

loss�<���       �	�[PIXc�A�*

loss��*=��       �	QIXc�A�*

loss.�<\�sx       �	�QIXc�A�*

loss�٤;6�e       �	aORIXc�A�*

loss<�=}}֫       �	��RIXc�A�*

loss@x�<��{�       �	E�SIXc�A�*

lossm�R>Ξд       �	�TIXc�A�*

loss�==��}       �	��TIXc�A�*

lossI�s=�-�       �	�fUIXc�A�*

lossqK�<^?K�       �	dVIXc�A�*

loss#v=�:��       �	��VIXc�A�*

loss�߰<���       �		4WIXc�A�*

loss���<.,��       �	�WIXc�A�*

lossx�W=��'p       �	�^XIXc�A�*

lossDȪ<���       �	�	YIXc�A�*

loss؁<{��t       �	��YIXc�A�*

loss$Ib=�/]�       �	��ZIXc�A�*

loss-\<�=�s       �	�)[IXc�A�*

lossC-9<f�F       �	s�[IXc�A�*

loss�<o� �       �	�x\IXc�A�*

lossqj�;�E�D       �	�$]IXc�A�*

loss��<��@       �	��]IXc�A�*

loss�X�;���%       �	Hk^IXc�A�*

loss�Ă<��       �	h	_IXc�A�*

lossL�>.��)       �	��_IXc�A�*

lossZ�i=�z65       �	�R`IXc�A�*

loss#�=<~n
       �	��`IXc�A�*

lossa�R<���[       �	��aIXc�A�*

loss�t�<�}/\       �	J'bIXc�A�*

loss<�W=�c       �	cIXc�A�*

lossj9�<4E�s       �	ȳcIXc�A�*

loss��X=����       �	kJdIXc�A�*

loss�==�'D�       �	|�dIXc�A�*

loss
D�=L��w       �	5�eIXc�A�*

lossS��;j�C&       �	�lfIXc�A�*

loss��;(��       �	F]gIXc�A�*

loss[�=x!!}       �	��gIXc�A�*

loss���=n���       �	P�hIXc�A�*

loss}��<�1V�       �	zViIXc�A�*

loss_��<1I       �	s�iIXc�A�*

loss�R=�tw^       �	��jIXc�A�*

loss�V�;�L�       �	okIXc�A�*

loss͗=6���       �	x
lIXc�A�*

loss��=��G�       �	�lIXc�A�*

loss�;FnE       �	BmIXc�A�*

loss�>;�AUG       �	[	nIXc�A�*

loss�s�=H՝T       �	��nIXc�A�*

loss5w�<��G       �	q;oIXc�A�*

loss��;.A�       �	=�oIXc�A�*

lossEe;�!T`       �	�hpIXc�A�*

lossʃ�<��~'       �	�qIXc�A�*

loss��?;J��       �	��qIXc�A�*

loss�_�:O,Wu       �	y:rIXc�A�*

lossYc>-��@       �	s�rIXc�A�*

loss�M�<9�E       �	apsIXc�A�*

lossD=���       �	-tIXc�A�*

lossi�&>� e*       �	o�tIXc�A�*

loss��<j��       �	){uIXc�A�*

lossC�<2օ
       �	�vIXc�A�*

loss+�;��C�       �	�vIXc�A�*

loss r<���z       �	�OwIXc�A�*

loss�0�<���       �	��wIXc�A�*

lossM^)>mB�       �	��xIXc�A�*

loss���=����       �	�4yIXc�A�*

loss�,A=�Q��       �	��yIXc�A�*

loss��=�ڭ�       �	�czIXc�A�*

loss��\<�|�       �	��zIXc�A�*

loss,
�<�z�       �	C�{IXc�A�*

loss�� =�*��       �	�F|IXc�A�*

loss��*=4�H�       �	�|IXc�A�*

loss}�=�Ԋ�       �	$|}IXc�A�*

loss1�)<�� g       �	�~IXc�A�*

loss�/�=���i       �	�IXc�A�*

loss��<�S2�       �	ǽIXc�A�*

loss�+<��       �	W]�IXc�A�*

losso��;I��A       �	�IXc�A�*

lossLCN='���       �	fN�IXc�A�*

loss+�=;I[       �	���IXc�A�*

loss��<��*       �	��IXc�A�*

loss���</o��       �	]ĄIXc�A�*

loss�P�;8.��       �	jj�IXc�A�*

lossTJl=fCѢ       �	.�IXc�A�*

loss���=X��       �	8��IXc�A�*

loss��<��o       �	lB�IXc�A�*

loss!`<��<       �	��IXc�A�*

loss_�<��S�       �	�x�IXc�A�*

lossˊ<	��.       �	�!�IXc�A�*

loss-�}<�<�       �	��IXc�A�*

loss]�;��N�       �	�a�IXc�A�*

loss�Z�<�?V�       �	�IXc�A�*

losse��;��$U       �	;��IXc�A�*

loss��W<;Z?�       �	�A�IXc�A�*

loss�=�+g�       �	��IXc�A�*

loss��"=R�       �	ׅ�IXc�A�*

loss�"=���z       �	��IXc�A�*

loss�R=D9��       �	�%�IXc�A�*

loss[X�<e�b�       �	s��IXc�A�*

loss42�<۲;       �	�P�IXc�A�*

loss��Q=H�       �	2�IXc�A�*

loss}�`=�C�       �	a��IXc�A�*

loss��.<0�       �	�"�IXc�A�*

loss���<����       �	e��IXc�A�*

loss�h�<e��       �	�G�IXc�A�*

loss3;X;Q�       �	1�IXc�A�*

loss\P�<л�       �	��IXc�A�*

loss�c�:�H��       �	�(�IXc�A�*

loss���;{$�       �	�ƖIXc�A�*

loss�@�<����       �	#g�IXc�A�*

lossx��<,�w�       �	��IXc�A�*

losss�<-eL%       �	'��IXc�A�*

loss{OX=��*       �	�M�IXc�A�*

loss-��=6�h       �	���IXc�A�*

loss@q�;�x~       �	e��IXc�A�*

lossԓ<Jf�.       �	�%�IXc�A�*

loss�F>Cj       �	�IXc�A�*

loss}1�:��p       �	�_�IXc�A�*

lossD'�;cMO       �	}�IXc�A�*

loss
e�:
n�       �	襝IXc�A�*

losst��<e�;�       �	�B�IXc�A�*

lossB�!;���       �	�ߞIXc�A�*

loss�$�:���       �	���IXc�A�*

lossI�:>�       �	��IXc�A�*

loss��<ή�       �	W'�IXc�A�*

loss���<FY��       �	OɡIXc�A�*

lossh:%;) V�       �	�_�IXc�A�*

loss�/:��r�       �	��IXc�A�*

loss�$s<m���       �	���IXc�A�*

lossѳ=GXU�       �	v6�IXc�A�*

losse8#=ԤC�       �	�ۤIXc�A�*

loss�$�:=^�       �	�u�IXc�A�*

loss�{=���       �	#�IXc�A�*

lossIT+>&oɶ       �	���IXc�A�*

losss�;D�f�       �	\X�IXc�A�*

loss��4>��i       �	��IXc�A�*

losswOy=:��       �	7¨IXc�A�*

loss�t
<���       �	W]�IXc�A�*

loss%��=_��       �	��IXc�A�*

loss@��<�B�8       �	���IXc�A�*

loss���=�&��       �	-#�IXc�A�*

lossvn<�]�R       �	j��IXc�A�*

loss{�(=����       �	]�IXc�A�*

loss?�<ND�c       �	��IXc�A�*

lossu�=
�       �	���IXc�A�*

loss�A�=Yi�6       �	 :�IXc�A�*

loss]n}=���       �	�ծIXc�A�*

loss�S<��|       �	�y�IXc�A�*

loss<$�<�ב,       �	��IXc�A�*

loss��<5R       �	F��IXc�A�*

loss��m=宭!       �	HN�IXc�A�*

loss=0<��
       �	��IXc�A�*

loss��<M��       �	���IXc�A�*

lossfÌ<3��;       �	��IXc�A�*

loss3�<�d6       �	p��IXc�A�*

lossс=a9�       �	�N�IXc�A�*

lossCg=��       �	��IXc�A�*

loss;<.-B�       �	�IXc�A�*

loss���;2�[       �	��IXc�A�*

lossf�;�hE�       �	|��IXc�A�*

loss�e�<U�       �	�W�IXc�A�*

loss{�<�G5$       �	p��IXc�A�*

loss<]=���       �	���IXc�A�*

loss��t=��F       �	z�IXc�A�*

lossl��<����       �	#��IXc�A�*

loss�4"<��՘       �	�\�IXc�A�*

loss}�$=���P       �	;��IXc�A�*

loss1�h< @X       �	���IXc�A�*

loss<��<��       �	�<�IXc�A�*

loss:L!<ÙA|       �	�IXc�A�*

loss�Yr<�4�n       �	擽IXc�A�*

loss�]=�zo       �	.=�IXc�A�*

loss+�=˦O       �	��IXc�A�*

loss��<5�s       �	�}�IXc�A�*

loss�d�;8lBf       �	��IXc�A�*

loss��;=@EJN       �	���IXc�A�*

loss���<^��O       �	�P�IXc�A�*

loss@��<�Y�M       �	}��IXc�A�*

loss�3�<)b{       �	#��IXc�A�*

loss��>�kY�       �	d!�IXc�A�*

losse��=ձ�       �	M��IXc�A�*

loss�N<n<X�       �	\q�IXc�A�*

loss��<!�s#       �	9�IXc�A�*

lossh}<V�z*       �	��IXc�A�*

loss8* <?�X0       �	�G�IXc�A�*

loss髀;��p       �	�f�IXc�A�*

loss��<��Sw       �	X �IXc�A�*

loss��<���       �	���IXc�A�*

loss�`a;��j�       �	m7�IXc�A�*

loss��=����       �	���IXc�A�*

loss��Y<�;#�       �	�t�IXc�A�*

loss�v�<��$�       �	r�IXc�A�*

lossȽ;����       �	���IXc�A�*

loss�'�<"\�t       �	�U�IXc�A�*

loss�(f=�|x�       �	���IXc�A�*

loss���;I��       �	���IXc�A�*

loss�B+=&Ҝd       �	H4�IXc�A�*

lossӀ_<	�D@       �	���IXc�A�*

loss�	<w}       �	�j�IXc�A�*

loss��<�h�       �	��IXc�A�*

loss�;�<�7�#       �	7��IXc�A�*

loss|Fw:�        �	>z�IXc�A�*

lossx�;y���       �	��IXc�A�*

lossfD�;��       �	7��IXc�A�*

loss�r�<D���       �	�>�IXc�A�*

loss�G<���V       �	���IXc�A�*

loss6Y >L�c'       �	
d�IXc�A�*

loss\`�;�餖       �	��IXc�A�*

loss3��=�z��       �	˟�IXc�A�*

loss�J;���       �	n5�IXc�A�*

lossmA�<ܽ��       �	d��IXc�A�*

loss�Q<�Ta`       �	k�IXc�A�*

loss2Q;��:�       �	�IXc�A�*

lossA�5<�5��       �	p��IXc�A�*

lossC �<
��>       �	(e�IXc�A�*

loss;	�=С�g       �	��IXc�A�*

lossA�=�۽�       �	���IXc�A�*

lossc�
=�.5�       �	�;�IXc�A�*

loss�lU=���       �	0�IXc�A�*

loss�0�;�S       �	:��IXc�A�*

loss?�W=4�H       �	�N�IXc�A�*

lossi0�;��;       �	��IXc�A�*

lossm��;K�7       �	�|�IXc�A�*

loss#�=h�9�       �	��IXc�A�*

loss;V�=��u%       �	��IXc�A�*

loss2"�<Yw�       �	ZG�IXc�A�*

loss�A=U��B       �	t��IXc�A�*

loss�t�;S�       �	&��IXc�A�*

lossV�;�h�^       �	�%�IXc�A�*

loss�=�J�       �	i��IXc�A�*

loss��M=j�r�       �	$d�IXc�A�*

loss)�<;{�}       �	���IXc�A�*

lossQ��<��>       �	��IXc�A�*

lossC�;��       �	-A�IXc�A�*

lossY��<�]��       �	���IXc�A�*

lossH(�;�9��       �	o�IXc�A�*

loss}�<-S��       �	��IXc�A�*

loss)�^<a*g�       �	H��IXc�A�*

loss��=�%��       �	�> JXc�A�*

loss&a<L<��       �	�� JXc�A�*

losso��;F���       �	AfJXc�A�*

lossO�d;�1��       �	D�JXc�A�*

lossؔ+:�y       �	�JXc�A�*

loss�ŉ;�[�        �	JXc�A�*

loss�5�<�+;       �	S�JXc�A�*

loss,�=��W�       �	˟JXc�A�*

loss{�<�뷺       �	�7JXc�A�*

loss��;z/��       �	x�JXc�A�*

loss��=�޶�       �	{iJXc�A�*

loss�(<
o�X       �	�JXc�A�*

loss_�</1%       �	�JXc�A�*

loss)�4=�0Rs       �	�FJXc�A�*

lossy�;^�tv       �	��JXc�A�*

loss�F�=gd'w       �	�~	JXc�A�*

lossʟ�<��W�       �	�
JXc�A�*

loss�T<�t�       �	.�
JXc�A�*

loss�*b=���E       �		QJXc�A�*

lossZ��=mf"       �	��JXc�A�*

lossچu;'���       �	�JXc�A�*

loss1<�CI,       �	�5JXc�A�*

lossa�;��?J       �	�JXc�A�*

loss�V<��`       �	��JXc�A�*

loss�'/=�;<6       �	��JXc�A�*

loss��-;z��       �	�_JXc�A�*

loss���<◍�       �	2<JXc�A�*

losse�<��ǩ       �	�oJXc�A�*

loss���=�M��       �	�JXc�A�*

losss��;���x       �	/�JXc�A�*

loss��=P{��       �	�JXc�A�*

loss��<e��0       �	�DJXc�A�*

lossW��<�:n       �	��JXc�A�*

loss:�M<�� �       �	�lJXc�A�*

loss<�<ݷ4�       �	�JXc�A�*

loss�]<(�       �	z�JXc�A�*

loss��;<��P       �	?9JXc�A�*

loss�X</E�       �	�JXc�A�*

lossgx�=c�o       �	ѯJXc�A�*

loss�h.<骟�       �	(CJXc�A�*

lossJ��<�d��       �	=�JXc�A�*

loss_G>=*y�-       �	�xJXc�A�*

loss�_=b/�C       �	VJXc�A�*

loss�	>�&�       �	��JXc�A�*

loss�.�;C�Vn       �	U0JXc�A�*

lossf(�=6��       �	
�JXc�A�*

lossE�'<\H�>       �	DmJXc�A�*

lossc�9;�Tm       �	@�JXc�A�*

loss��:CP4�       �	3�JXc�A�*

lossO?�<WF1'       �	�  JXc�A�*

loss��9;QQ       �	$� JXc�A�*

lossTL	=���H       �	wL!JXc�A�*

loss-��<��E�       �	��!JXc�A�*

loss���;����       �	�s"JXc�A�*

loss;fK=pu�       �	�#JXc�A�*

lossg5=839       �	ԝ#JXc�A�*

lossi�><}�p�       �	2$JXc�A�*

loss�Z>.       �	��$JXc�A�*

loss��*<�΍       �	�c%JXc�A�*

loss咠<⏝�       �	�%JXc�A�*

loss��<�|       �	K�&JXc�A�*

loss`�1=�w�"       �	8I'JXc�A�*

lossJ>);|?       �	�'JXc�A�*

loss�-<[��       �	τ(JXc�A�*

loss.�=���       �	d#)JXc�A�*

loss��<���       �	��)JXc�A�*

loss!A�;�G�2       �	[[*JXc�A�*

loss�+6=�e [       �	{�*JXc�A�*

loss��<�F��       �	��+JXc�A�*

loss<��=I���       �	�,JXc�A�*

loss�_H=�XG�       �	�8-JXc�A�*

losst�<Un�        �	��-JXc�A�*

lossS=���b       �	J`.JXc�A�*

losseƮ=/>@�       �	��/JXc�A�*

lossm��<��R�       �	�t0JXc�A�*

loss�R�<j)-       �	N	1JXc�A�*

loss�n	<�܋�       �	+�1JXc�A�*

loss���<�ː�       �	u�2JXc�A�*

loss�-=
�v�       �	�&3JXc�A�*

lossxM=&�7       �	f�3JXc�A�*

loss��&=~�{�       �	jP4JXc�A�*

loss�>���       �	�4JXc�A�*

lossZ��<X       �	�t5JXc�A�*

lossi�,=a�:       �	[
6JXc�A�*

loss Ě:A3<�       �	4�6JXc�A�*

loss��A<q7w       �	�-7JXc�A�*

loss}��:�k�       �	r�7JXc�A�*

loss�C^;�Ȧ�       �	3R8JXc�A�*

lossR��<���c       �	��8JXc�A�*

lossۨ<�c       �	�z9JXc�A�*

lossJA=W�m       �	I:JXc�A�*

loss�3�=���E       �		�:JXc�A�*

loss�A=�+��       �	D6;JXc�A�*

loss�H�<���       �	6�;JXc�A�*

loss���;�:T       �	{g<JXc�A�*

loss:K�<��UW       �	��<JXc�A�*

loss�۩=�g       �	%�=JXc�A�*

loss[�}<�{1       �	E)>JXc�A�*

loss#ђ<gZ       �	U�>JXc�A�*

loss�~p<I��       �	�N?JXc�A�*

loss!� >16       �	��?JXc�A�*

loss���<S�I       �	"p@JXc�A�*

lossT �:���,       �	�AJXc�A�*

loss���;NT�<       �	��AJXc�A�*

loss`��<�
�       �	)BJXc�A�*

lossdN�=~N        �	'�BJXc�A�*

lossZت<�F       �	�CJXc�A�*

loss�߲=�˃       �	�xDJXc�A�*

loss1��:=�       �	W"EJXc�A�*

loss�gn=^�?�       �	��EJXc�A�*

loss�D:;n5��       �	�HFJXc�A�*

loss��<ml       �	��FJXc�A�*

lossh�=�!�q       �	JzGJXc�A�*

lossln<k�u@       �	�!HJXc�A�*

loss��<�{�       �	��HJXc�A�*

losst*9=����       �	PUIJXc�A�*

lossz�=�e
       �	)�IJXc�A�*

loss=�E=�)p}       �	V�JJXc�A�*

loss�^�;{bo       �	�KJXc�A�*

lossnj<��v       �	��KJXc�A�*

loss\��<=���       �	~9LJXc�A�*

loss�E;<�/       �	��LJXc�A�*

lossEƝ=��̃       �	�_MJXc�A�*

lossw��=>�}b       �	{�MJXc�A�*

loss�>�<"k�       �	�NJXc�A�*

loss�(�=�`��       �	��OJXc�A�*

lossO-�;�ܰQ       �	�=PJXc�A�*

loss��;��.       �	��PJXc�A�*

loss��=�2
�       �	V�QJXc�A�*

loss7	={���       �	�SJXc�A�*

loss�;<�e�       �	^TJXc�A�*

loss%<���       �	 UJXc�A�*

loss��<��%�       �	��UJXc�A�*

loss8��;�q�r       �	�SVJXc�A�*

lossX�o=e�0       �	�VJXc�A�*

loss��<�v�y       �	�WJXc�A�*

loss��<6�Ŕ       �	�@XJXc�A�*

lossHzH>�u�P       �	"�XJXc�A�*

lossE4A<��       �	�~YJXc�A�*

loss�m<;�H��       �	�XZJXc�A�*

loss�-�;8�[�       �	��ZJXc�A�*

lossҋ�:5$       �	)�[JXc�A�*

loss��;���       �	�+\JXc�A�*

loss��)=����       �	L�\JXc�A�*

lossh�"<�5�
       �	d]]JXc�A�*

loss��f=�e"O       �	��]JXc�A�*

loss|r
;p��y       �	��^JXc�A�*

loss
}�<(�       �	&_JXc�A�*

loss1-=��       �	֭_JXc�A�*

loss��B<�Y�E       �	�`JXc�A�*

loss��=)���       �	�YaJXc�A�*

loss� <��]       �	��aJXc�A�*

lossnh�<1��_       �	<�bJXc�A�*

lossN�e=3��n       �	�cJXc�A�*

lossL��=|���       �	=�cJXc�A�*

loss�?�<���       �	2VdJXc�A�*

lossM��<"��       �	5�dJXc�A�*

lossN�<ѧ �       �	$eJXc�A�*

lossLƊ=�A�%       �	7fJXc�A�*

loss��=���Z       �	�fJXc�A�*

loss$�/;��       �	xFgJXc�A�*

loss��=%rJ�       �	+�gJXc�A�*

loss���=�0I4       �	�}hJXc�A�*

loss��=�o�        �	*iJXc�A�*

lossV��=�x73       �	��iJXc�A�*

loss���=D��       �	NdjJXc�A�*

loss��=\^qZ       �	��jJXc�A�*

loss�= B�       �	֌kJXc�A�*

lossܚ#;���f       �	-lJXc�A�*

loss�3=�C�8       �	>�lJXc�A�*

lossԸ.=E\5�       �	ZemJXc�A�*

loss̻;T��       �	�nJXc�A�*

loss�LA=�>K       �	�nJXc�A�*

loss[�<cZ �       �	�6oJXc�A�*

lossn*H<ω       �	A�oJXc�A�*

loss�M�;HO�2       �	spJXc�A�*

loss�l)<�ݑ�       �	0qJXc�A�*

lossdQU;��[       �	.�qJXc�A�*

loss�N�:���x       �	ErJXc�A�*

loss�<=��Ac       �	4�rJXc�A�*

loss���;
��       �	"osJXc�A�*

loss �
=��e       �	�tJXc�A�*

loss��g;DK       �	��tJXc�A�*

loss�\<ԟc       �	�@uJXc�A�*

lossH��<�b�I       �	��uJXc�A�*

loss*i�<7�9       �	�nvJXc�A�*

lossX�<�f�e       �	�wJXc�A�*

loss�1=<��w�       �	!�wJXc�A�*

loss.�<Xw�       �	t%xJXc�A�*

loss7�I<A�T       �	��xJXc�A�*

loss��<zh�^       �	!vyJXc�A�*

loss��u=�L       �	�zJXc�A�*

loss1)�<CA�#       �	4�zJXc�A�*

loss A;�x�a       �	a8{JXc�A�*

loss�=�=� s�       �	V�{JXc�A�*

loss��$=ǻ�       �	�m|JXc�A�*

loss~��<��5       �	�}JXc�A�*

losswPv<=R�       �	��}JXc�A�*

lossC��=� /�       �	q<~JXc�A�*

loss�C�<�1��       �	A�~JXc�A�*

losswn*;�8�h       �	�sJXc�A�*

loss���;϶�       �	��JXc�A�*

loss��(=W��       �	���JXc�A�*

loss�u<�.V       �	�L�JXc�A�*

losst�(=��s�       �	��JXc�A�*

loss��=1�g�       �	���JXc�A�*

loss%��=�lL�       �	��JXc�A�*

loss.X;'��       �	Q��JXc�A�*

loss�s<Vt7       �	?W�JXc�A�*

loss�Xg=�̐       �	��JXc�A�*

lossZ�<�:�       �	���JXc�A�*

loss�f�<O��       �	�&�JXc�A�*

lossF�q;.�x;       �	/JXc�A�*

loss��o=m�@T       �	�a�JXc�A�*

lossE7;O���       �	y�JXc�A�*

loss��#=;kii       �	Q��JXc�A�*

loss��{<� �7       �	%>�JXc�A�*

loss��:iV�7       �	�ӉJXc�A�*

loss�S"<�[[�       �	�j�JXc�A�*

loss%Q�=fz��       �	]��JXc�A�*

lossi2<�Sh       �	6��JXc�A�*

lossa�=t�.�       �	�&�JXc�A�*

loss��m=r���       �	s��JXc�A�*

loss�E ;1j=�       �	"O�JXc�A�*

loss-��;����       �	��JXc�A�*

loss��;��\�       �	7��JXc�A�*

loss�lA<]eR�       �	�̏JXc�A�*

loss�h�:�iC�       �	h�JXc�A�*

loss�$(<�&S�       �	j��JXc�A�*

loss_�;���       �	��JXc�A�*

loss�zI;-t�,       �	us�JXc�A�*

loss<�G<3>K       �	?6�JXc�A�*

loss���</}�       �	BΓJXc�A�*

loss��#=z*4h       �	�^�JXc�A�*

lossF�==ҏ�       �	V�JXc�A�*

lossJ��<�!߾       �	�&�JXc�A�*

loss;�&��       �	и�JXc�A�*

lossR�u<����       �	�O�JXc�A�*

loss�Z=4���       �	Q��JXc�A�*

lossfн; ���       �	&��JXc�A�*

losso�;2#>       �	,-�JXc�A�*

loss�<���^       �	�řJXc�A�*

loss�cN<�y       �	h\�JXc�A�*

loss �<bwr       �	��JXc�A�*

loss�Cr=W���       �	��JXc�A�*

loss#��=�%�#       �	&�JXc�A�*

lossA]=r ��       �	�ɜJXc�A�*

lossJww<na��       �	�v�JXc�A�*

loss�p�<z��b       �	L�JXc�A�*

loss^A�<2Hv�       �	���JXc�A�*

loss��>=�`�       �	�b�JXc�A�*

lossc-=#:<�       �	}�JXc�A�*

loss{@�=N
       �	m��JXc�A�*

loss�	�;�T�       �	ӟ�JXc�A�*

loss1��<���       �	_F�JXc�A�*

loss7��<�m       �	.�JXc�A�*

lossh�<C�e�       �	㎣JXc�A�*

loss���:��׆       �	8/�JXc�A�*

loss�(=�Ϙ%       �	gѤJXc�A�*

loss�t�<��Ne       �	&o�JXc�A�*

loss�z<��sO       �	�8�JXc�A�*

loss-Q)<H���       �	@�JXc�A�*

loss��U=��î       �	�JXc�A�*

loss��H:>�&�       �	,��JXc�A�*

loss]��<g !}       �	eo�JXc�A�*

loss��=ng�       �	��JXc�A�*

loss�b�;��Ӱ       �	�ƪJXc�A�*

loss=����       �	Cu�JXc�A�*

loss�=:r<�       �	�JXc�A�*

loss4�Q=��o4       �	��JXc�A�*

lossL��;bo�       �	�S�JXc�A�*

lossA�G<3�A�       �	�JXc�A�*

loss�W�=���o       �	E֯JXc�A�*

loss*��<x�Y,       �	W��JXc�A�*

loss���<M	�_       �	�J�JXc�A�*

lossF�=�F�       �	Z�JXc�A�*

lossç�=���       �	���JXc�A�*

loss}��<n�I       �	�I�JXc�A�*

losss�<Lm��       �	{��JXc�A�*

loss��Y;�9       �	���JXc�A�*

loss��<��%�       �	.�JXc�A�*

loss[H�<��E~       �	˵JXc�A�*

loss ��;���       �	/i�JXc�A�*

lossŮ_=q	b       �	�JXc�A�*

loss�{<+N�       �	���JXc�A�*

loss�P�<0*'       �	8�JXc�A�*

loss/t�;{D��       �	۸JXc�A�*

loss7l<^;�       �	�x�JXc�A�*

loss0�<�At       �	��JXc�A�*

loss��;;n���       �	B��JXc�A�*

loss���<��GW       �	�M�JXc�A�*

loss�P�=��m�       �	S�JXc�A�*

loss���=�ܚ       �	���JXc�A�*

loss�ؑ<Y��W       �	T�JXc�A�*

lossdO	=�ݮ       �	���JXc�A�*

loss��:=ᶿ�       �	�K�JXc�A�*

loss@s;I���       �	O�JXc�A�*

loss�V1<�?/�       �	9}�JXc�A�*

lossַb;���       �	<�JXc�A�*

loss8Uq=��i       �	���JXc�A�*

loss]�&<s� �       �	�@�JXc�A�*

loss�
�=أ��       �	���JXc�A�*

loss�l`<^l>       �	'h�JXc�A�*

loss���<)u�       �	���JXc�A�*

loss	�(=��Va       �	=��JXc�A�*

loss�c�<�NN,       �	�0�JXc�A�*

loss��;�;NT       �	6��JXc�A�*

loss�a�;f���       �	�`�JXc�A�*

losszwJ=�_�       �	��JXc�A�*

lossxz9<�u�       �	���JXc�A�*

loss_�m=O�t�       �	1�JXc�A�*

lossd��=����       �	q��JXc�A�*

loss�Ua=��       �	�^�JXc�A�*

loss�ls=���L       �	���JXc�A�*

loss�j�;�D�h       �	���JXc�A�*

loss	<g.;�       �	�#�JXc�A�*

lossڍ�<:+��       �	0��JXc�A�*

loss[ +=j��       �	�?�JXc�A�*

lossV�%;�!��       �	���JXc�A�*

loss)A<6�<�       �	B��JXc�A�*

loss��a<x��       �	�4�JXc�A�*

loss�.�;ԇsg       �	���JXc�A�*

lossփ�;?�S       �	���JXc�A�*

loss&��<i��       �	C��JXc�A�*

loss/eu=�Qٛ       �	�b�JXc�A�*

loss%$�<r       �	SZ�JXc�A�*

loss��*<t���       �	���JXc�A�*

loss�a0;S��       �	���JXc�A�*

loss��|;r�7       �	�J�JXc�A�*

loss��=�[`       �	p��JXc�A�*

loss���<�q       �	{��JXc�A�*

loss�K�:9Y!       �	v�JXc�A�*

loss�[�=|t       �	��JXc�A�*

loss�_�<�y�       �	a��JXc�A�*

losse@z;+��R       �	�I�JXc�A�*

loss`�;�F��       �	h��JXc�A�*

loss��;�wo�       �	{��JXc�A�*

loss@Y&=��+�       �	�f�JXc�A�*

loss=|;<�a��       �	�
�JXc�A�*

loss_��<`��       �	��JXc�A�*

lossa�?<�mP�       �	�I�JXc�A�*

loss��N<�W/       �	)��JXc�A�*

loss=x=4}       �	���JXc�A�*

loss�E<z �       �	J&�JXc�A�*

lossͭ�<�Tm�       �	-��JXc�A�*

lossc��:`�f2       �	p}�JXc�A�*

loss���;q�w       �	��JXc�A�*

loss$.:����       �	qu�JXc�A�*

loss�D�;�-       �	��JXc�A�*

loss��=R�       �	��JXc�A�*

lossM<|=ǥs       �	-]�JXc�A�*

loss���={�.       �	���JXc�A�*

loss1�a<4>�7       �	��JXc�A�*

loss7��<m�_       �	ҋ�JXc�A�*

loss���<]�B;       �	l]�JXc�A�*

loss�
<t^��       �	�?�JXc�A�*

loss���<b�
S       �	h��JXc�A�*

loss���=�l��       �	�JXc�A�*

loss�=�]�P       �	��JXc�A�*

loss���=���       �	��JXc�A�*

lossZ�=afE�       �	��JXc�A�*

loss��G=�6��       �	G<�JXc�A�*

loss�ܝ;|�>F       �	��JXc�A�*

loss��;>���       �	�o�JXc�A�*

loss��O;ya��       �	_�JXc�A�*

lossNb=o��       �	M��JXc�A�*

loss#�<G��       �	8L�JXc�A�*

loss3�[<�m��       �	���JXc�A�*

lossm��=���       �	���JXc�A�*

loss$Z�;�2C�       �	,,�JXc�A�*

losszG�=9�       �	T��JXc�A�*

loss2�O=�a[       �	]�JXc�A�*

loss�e=��       �	r��JXc�A�*

loss{�<�=6       �	!��JXc�A�*

loss��=:��       �	�*�JXc�A�*

lossq�%<ܽ}�       �	���JXc�A�*

loss�K�<l�~       �	KW�JXc�A�*

loss�n:=m�=       �	���JXc�A�*

loss�
�<UaM       �	��JXc�A�*

loss��/<�C�       �	5)�JXc�A�*

loss���;��[�       �	^��JXc�A�*

loss��;<����       �	X�JXc�A�*

loss�-=��-       �	9��JXc�A�*

loss�s�;m�       �	ۊ�JXc�A�*

loss�=����       �	�%�JXc�A�*

loss�Vt;���       �	���JXc�A�*

lossW<��2�       �	5^�JXc�A�*

loss�oY=-��R       �	���JXc�A�*

loss��=��-       �	ݕ�JXc�A�*

lossEm)=Xft       �	/2�JXc�A�*

lossD�>�Vd�       �	6��JXc�A�*

loss��\;��J=       �	O��JXc�A�*

loss<ɘ<�r_       �	��JXc�A�*

lossX��=c�\       �	w,�JXc�A�*

loss|�A=Q)�       �	k��JXc�A�*

loss;A,<o�$G       �	�w KXc�A�*

loss��%<�=f�       �	�KXc�A�*

loss_H�<���       �	E�KXc�A�*

loss��m<Pgn       �	p[KXc�A�*

loss��;�F�       �	s�KXc�A�*

loss(��<N|զ       �	ڏKXc�A�*

loss��v=�c��       �	s-KXc�A�*

loss|�=�E�       �	�KXc�A�*

loss�'`<Ϝ�J       �	@kKXc�A�*

loss~=�)[3       �	�KXc�A�*

loss���<�N�       �	1�KXc�A�*

loss �=t��I       �	�.KXc�A�*

loss{��<l�       �	Q�KXc�A�*

loss���<F�w       �	fiKXc�A�*

loss�#1;$h^m       �	D�KXc�A�*

loss@j�=}�"       �	��	KXc�A�*

loss���<��%       �	~;
KXc�A�*

lossv�T;�       �	��
KXc�A�*

loss�]�;�       �	dKXc�A�*

loss���<X�Q"       �	�KXc�A�*

loss�f=ڂ�       �	��KXc�A�*

loss� �;�IL       �	�%KXc�A�*

loss�`�=n���       �	�KXc�A�*

loss
J�;��A�       �	iUKXc�A�*

loss���<�~       �	��KXc�A�*

lossq�v=�\��       �	�KXc�A�*

loss���;y?�       �	 $KXc�A�*

lossi��<f���       �	)<KXc�A�*

loss���:�s�       �	�qKXc�A�*

loss��=1
[       �	EKXc�A�*

lossFp=��s�       �	%�KXc�A�*

loss�: >d4       �	9DKXc�A�*

loss�FM=��.       �	O�KXc�A�*

lossZz=�)��       �	~�KXc�A�*

loss14�<���       �	�!KXc�A�*

lossN6'=�V��       �	#�KXc�A�*

loss��=l�0�       �		�KXc�A�*

lossmh�<�4@       �	�KXc�A�*

loss%�<�?�       �	� KXc�A�*

loss��=-�[�       �	1}KXc�A�*

loss �4=��_       �	�kKXc�A�*

lossJ�=�v�       �	BvKXc�A�*

loss�u�<P���       �	eKXc�A�*

loss�I<���J       �	wNKXc�A�*

loss��7<��       �	��KXc�A�*

lossêp<-h�E       �	�� KXc�A�*

loss��b<���       �	$}!KXc�A�*

loss,ς<���       �	�"KXc�A�*

loss�R�<��K�       �	K�"KXc�A�*

loss(.z<�HE�       �	�#KXc�A�*

loss��:=G���       �	5�$KXc�A�*

loss��5=U�m       �	�!&KXc�A�*

loss�6=�6�h       �	D'KXc�A�*

loss�+�;Ӡgx       �	Ǽ'KXc�A�*

loss���<��ҋ       �	tb(KXc�A�*

loss�<��t=       �	��(KXc�A�*

lossE�<��       �	�)KXc�A�*

lossa't<d��       �	��*KXc�A�*

loss��< }�       �	��+KXc�A�*

loss���<+�d       �	�F,KXc�A�*

loss�<��       �	�,KXc�A�*

loss�?�;j�kT       �	_y-KXc�A�*

losslQ�;"���       �	�.KXc�A�*

loss�|<Ϯ�       �	��.KXc�A�*

loss5�;)灠       �	G/KXc�A�*

loss/X=�[o�       �	��/KXc�A�*

loss�S ;���j       �	tz0KXc�A�*

loss���<�N�=       �	�1KXc�A�*

lossl6z=�`�       �	T�1KXc�A�*

loss���;�ǜj       �	pC2KXc�A�*

lossv��:��P_       �	��2KXc�A�*

loss�Ũ=c�|       �	�z3KXc�A�*

loss&Mq;?�2       �	�4KXc�A�*

loss!�<�:1�       �	��4KXc�A�*

loss}d�:��V?       �	�F5KXc�A�*

lossl.�;E�#~       �	��5KXc�A�*

loss
8�; ��       �	�v6KXc�A�*

loss�b2<+X�       �	�7KXc�A�*

lossfw�;+��.       �	ݳ7KXc�A�*

loss%�f<s�%       �	YO8KXc�A�*

loss�I�<�=(       �	=�8KXc�A�*

lossj�h<�>�       �	&�9KXc�A�*

loss`�<��0       �	�):KXc�A�*

loss��b="��       �	m�:KXc�A�*

lossR�I=����       �	Lk;KXc�A�*

loss��<;��uq       �	2<KXc�A�*

loss$��9���       �	�<KXc�A�*

loss��<���       �	�8=KXc�A�*

loss���; �z        �	��=KXc�A�*

loss��y<��`       �	t>KXc�A�*

loss��:��)       �	E?KXc�A�*

loss�t;èF       �	�?KXc�A�*

loss,�59��֍       �	�M@KXc�A�*

lossϘ�7�u��       �	��@KXc�A�*

loss��h;��{x       �	|�AKXc�A�*

loss�dL<��l       �	)!BKXc�A�*

loss�0<�_A�       �	��BKXc�A�*

loss/`�;b�O�       �	�_CKXc�A�*

loss���9�^�       �	*�CKXc�A�*

loss���;K�       �	�DKXc�A�*

loss�i�=c�       �	-?EKXc�A�*

loss��:P`�       �	��EKXc�A�*

loss�{�=�/ӽ       �	�zFKXc�A�*

lossq>_=��]�       �	0GKXc�A�*

lossn>)�       �	֧GKXc�A�*

loss6�<)dQk       �	�IHKXc�A�*

loss�c�=�2��       �	*�HKXc�A�*

loss|��<V7O       �	(}IKXc�A�*

loss�r<o�       �	�1JKXc�A�*

loss1�2<v�S       �	G�JKXc�A�*

loss3$�=�M�       �	\KKXc�A�*

loss��<��ֽ       �	,�KKXc�A�*

loss晓=,)�       �	��LKXc�A�*

loss\�=�wbu       �	�MKXc�A�*

loss3�=���       �	ڮMKXc�A�*

loss�}�<�Z�v       �	�DNKXc�A�*

loss��k<�)�s       �	f�NKXc�A�*

loss/B7=mϡ�       �	Q�OKXc�A�*

lossr�<�#�       �	b�PKXc�A�*

lossg=�J��       �	`QKXc�A�*

lossز<�       �	��QKXc�A�*

loss�a;�e+       �	�FRKXc�A�*

loss��<ث��       �	��RKXc�A�*

loss;�Z<7��       �	B�SKXc�A�*

lossT�b<�y       �	�$TKXc�A�*

lossM <��[       �	>�TKXc�A�*

lossL�.;Q��+       �	�tUKXc�A�*

loss�R�=�o!�       �	�VKXc�A�*

loss�Ϥ;���	       �	�VKXc�A�*

loss�d=��       �	�7WKXc�A�*

lossOy=���       �	�WKXc�A�*

loss�f<��<u       �	�iXKXc�A�*

lossZ =���?       �	K YKXc�A�*

loss�2�;��9s       �	��YKXc�A�*

loss��;�I}N       �	� ZKXc�A�*

loss;�<2g�       �	��ZKXc�A�*

loss���<�+�       �	\Z[KXc�A�*

lossZ�=E	�/       �	_�[KXc�A�*

loss�hT=ɿ�       �	��\KXc�A�*

loss�{�<�	�       �	r]KXc�A�*

lossJp<a�	       �	�]KXc�A�*

loss�[B;�W_$       �	JF^KXc�A�*

loss5�<�V�       �	n�^KXc�A�*

loss?�W<e8�w       �	�l_KXc�A�*

loss�9=Q�Ȥ       �	�`KXc�A�*

loss��<�� �       �	t�`KXc�A�*

loss��<�0,�       �	-aKXc�A�*

lossT	�=��X       �	 �aKXc�A�*

loss/J<�39l       �	�cbKXc�A�*

lossH�]=�]U-       �	+�bKXc�A�*

loss���;8��p       �	��cKXc�A�*

loss#Q1;v)�[       �	�dKXc�A�*

loss�;=@A_�       �	*�|KXc�A�*

lossd�+=JV�A       �	K!}KXc�A�*

lossWB=|/$�       �	��}KXc�A�*

loss?�<D��       �	.v~KXc�A�*

lossP}=H&�       �	NKXc�A�*

loss��<V�"       �	��KXc�A�*

loss���;��       �	y:�KXc�A�*

loss*��<�B��       �	[ӀKXc�A�*

lossD=��S�       �	Af�KXc�A�*

lossE�<���       �	��KXc�A�*

loss���;���       �	ߥ�KXc�A�*

loss#P�<��N�       �	;�KXc�A�*

loss\��<�v��       �	FЄKXc�A�*

loss蟁< *�       �	*;�KXc�A�*

loss�N<I��t       �	�ՆKXc�A�*

loss�;�˜�       �	ur�KXc�A�*

loss��R:����       �	�'�KXc�A�*

loss�t�<�h       �	]KXc�A�*

loss��<ą-�       �	.Y�KXc�A�*

losspV=)��R       �	���KXc�A�*

loss��Z<#�Y�       �	4��KXc�A�*

loss��)=RhI�       �	e�KXc�A�*

loss�~�;g$V       �	\��KXc�A�*

loss��<w�       �	�C�KXc�A�*

lossJ�7=��D       �	�،KXc�A�*

loss!;^�(�       �	?q�KXc�A�*

loss�XI=����       �	g�KXc�A�*

loss���<�f6^       �	*��KXc�A�*

loss�/=iK�       �	�E�KXc�A�*

loss7�:;��4       �	�؏KXc�A�*

loss��<8z��       �	$��KXc�A�*

loss���;)(P       �	�;�KXc�A�*

loss(��<�j��       �	5ӑKXc�A�*

lossl=u�       �	Zh�KXc�A�*

loss	��<�'�X       �	��KXc�A�*

loss�6<��%�       �	J��KXc�A�*

loss�e�<��g       �	|+�KXc�A�*

loss?��<��Q�       �	輔KXc�A�*

lossA<�=҄܎       �	N�KXc�A�*

loss�}�;��       �	nݕKXc�A�*

loss���<A�o       �	en�KXc�A�*

lossB=�V��       �	��KXc�A�*

loss�<���f       �	���KXc�A�*

loss��<'�       �	K>�KXc�A�*

loss���<�m71       �	�ܘKXc�A�*

loss���=u׸T       �	p�KXc�A�*

loss\?�<ΪĚ       �	G�KXc�A�*

loss-$=5���       �	)��KXc�A�*

loss���<�"��       �	<0�KXc�A�*

loss�K;��        �	ǛKXc�A�*

loss�y:t-OV       �	N_�KXc�A�*

loss�'<�p�N       �	B�KXc�A�*

lossF�=���       �	���KXc�A�*

loss�#�=�B�^       �	iQ�KXc�A�*

loss�<�9�       �	���KXc�A�*

lossz��;�:�	       �	���KXc�A�*

loss1�<�X�       �	=�KXc�A�*

loss�AZ<�|~2       �	5ѠKXc�A�*

loss�B;%~�I       �	)v�KXc�A�*

loss�==�`��       �	��KXc�A�*

loss�'>\��       �	���KXc�A�*

loss`N;Љ}q       �	�N�KXc�A�*

loss���<>�`       �	G�KXc�A�*

loss�=뒣�       �		��KXc�A�*

lossfE*;}���       �	,�KXc�A�*

loss��U<LH�4       �	�ƥKXc�A�*

loss�Z�;2��       �	�d�KXc�A�*

loss��<�(l�       �	�KXc�A�*

loss�	>D@�t       �	흧KXc�A�*

loss�*<{y�v       �	�F�KXc�A�*

lossRp�<�H�       �	��KXc�A�*

loss�3$<�x��       �	w�KXc�A�*

loss��;����       �	��KXc�A�*

loss���:��.       �	ƪKXc�A�*

loss��<�W�       �	x^�KXc�A�*

lossoL<aZ/       �	���KXc�A�*

lossD�<����       �	���KXc�A�*

loss#lv;��Z       �	W>�KXc�A�*

loss�<>��       �	�ҭKXc�A�*

loss�}�=�1��       �	{�KXc�A�*

lossV<=Sa�`       �	{�KXc�A�*

lossA�=QA��       �	.ɯKXc�A�*

loss�9�;�6�       �	�l�KXc�A�*

loss�n�:�
��       �	 �KXc�A�*

loss���:�e�*       �	̲�KXc�A�*

loss��<�p��       �	oF�KXc�A�*

losss�3=�Ї�       �	ٲKXc�A�*

lossm�=�ݫ       �	�ųKXc�A�*

lossd�%<P�}�       �	eS�KXc�A�*

lossF@�<���z       �	m�KXc�A�*

loss@=����       �	�t�KXc�A�*

loss�=@��o       �		�KXc�A�*

lossߗ!;Q���       �	U��KXc�A�*

loss?4�:a�       �	?;�KXc�A�*

loss� =�OF�       �	�ַKXc�A�*

loss=�R;�       �	�o�KXc�A�*

lossj�l=���       �	E*�KXc�A�*

loss�<�yT       �	�ҺKXc�A�*

loss̜(=P�/3       �	n�KXc�A�*

loss�<]i�W       �	�KXc�A�*

loss��9C�v�       �	��KXc�A�*

loss5W: �       �	ձ�KXc�A�*

loss�j=[=w�       �	�I�KXc�A�*

lossH�;	�XG       �	ܾKXc�A�*

lossV�:���        �	2w�KXc�A�*

loss�8=Ow�       �	#�KXc�A�*

loss�0 <"(Q       �	D��KXc�A�*

loss��:���5       �	=�KXc�A�*

loss��=Ҳ��       �	V��KXc�A�*

loss���;ZϦ�       �	�t�KXc�A�*

lossT�|=�1cx       �	�KXc�A�*

loss�<�9��       �	h��KXc�A�*

loss��=>���       �	�L�KXc�A�*

loss��;�CF       �	���KXc�A�*

loss`*7<]���       �	n��KXc�A�*

lossL�=z�s       �	�O�KXc�A�*

loss8�;A�u�       �	
��KXc�A�*

loss�3�=�>��       �	��KXc�A�*

lossݥ=j�5�       �	^+�KXc�A�*

loss,��:�3�       �	��KXc�A�*

loss
�q<�삼       �	`�KXc�A�*

loss3Y�<E	P       �	���KXc�A�*

loss��#<Vf       �	C8�KXc�A�*

loss5:�<��R�       �	��KXc�A�*

lossڋ�;����       �	�6�KXc�A�*

loss��=ZF��       �	���KXc�A�*

loss�c==�       �	t'�KXc�A�*

loss{�r;��       �	���KXc�A�*

loss%3y=���       �	��KXc�A�*

loss!��<�Ӧ�       �	��KXc�A�*

loss,�:�=Ⱦ       �	c�KXc�A�*

loss�I<�}�N       �	@��KXc�A�*

lossI�<d��       �	W��KXc�A�*

loss�Ex= ��       �	��KXc�A�*

loss�ɬ<��[�       �	��KXc�A�*

loss�?D<�R       �	ZJ�KXc�A�*

loss��A<���       �	��KXc�A�*

loss�:�UA�       �	�t�KXc�A�*

loss�$<U���       �	M�KXc�A�*

lossJ�<s@i       �	���KXc�A�*

loss�^;%/�%       �	#J�KXc�A�*

lossSɗ<���       �	!��KXc�A�*

loss���;o��       �	9��KXc�A�*

loss�H�<'��       �	��KXc�A�*

lossl�=6Y�_       �	}��KXc�A�*

loss
p�;�Q��       �	�C�KXc�A�*

lossz;�lh       �	Q��KXc�A�*

loss�\P;C       �	Tq�KXc�A�*

loss�{:о�       �	��KXc�A�*

loss%�<�~�       �	���KXc�A�*

lossx��<5�od       �	sJ�KXc�A�*

loss�;�<���       �	��KXc�A�*

loss��$=��5       �	��KXc�A�*

lossh��<L�n�       �	w1�KXc�A�*

lossy�=b�B        �	���KXc�A�*

lossFu�;M��       �	�u�KXc�A�*

loss�};
���       �	�C�KXc�A�*

loss)�=H��        �	3��KXc�A�*

loss1>=M+?�       �	2U�KXc�A�*

loss���:h��Q       �	}��KXc�A�*

lossܴ:=&�       �	��KXc�A�*

loss@,<���       �	�.�KXc�A�*

loss\:�;�ٖ�       �	�h�KXc�A�*

loss�q�<�`��       �	�!�KXc�A�*

loss��Q<3gq5       �	��KXc�A�*

lossC��<��6       �	�I�KXc�A�*

loss[9�;�*s�       �	��KXc�A�*

lossc�q<��rV       �	4��KXc�A�*

lossE�P=^+       �	al�KXc�A�*

loss���<�       �	W�KXc�A�*

lossۓ�;����       �	���KXc�A�*

lossn�4<�'�       �	���KXc�A�*

lossG1�<k��       �	�j�KXc�A�*

loss늍;ݭɨ       �	P�KXc�A�*

loss�uR<|���       �	���KXc�A�*

loss+�;lLx;       �	�B�KXc�A�*

lossi�E<�8�       �	���KXc�A�*

lossv=�� �       �	�h�KXc�A�*

loss���=���       �	��KXc�A�*

lossJ�;�G�       �	0��KXc�A�*

loss�1F;�'��       �	�8�KXc�A�*

loss��%=_��6       �	���KXc�A�*

loss�d�<7��       �	t�KXc�A�*

lossR}8:�jH�       �	A�KXc�A�*

lossMx�:�7@�       �	~��KXc�A�*

lossf��<��@       �	-@�KXc�A�*

loss��=xAJ�       �	4��KXc�A�*

lossi��:�It       �	{j�KXc�A�*

loss���<��       �	�'�KXc�A�*

loss��=���b       �	���KXc�A�*

loss�/=r��       �	�i�KXc�A�*

losss��;���#       �	��KXc�A�*

loss���;�c�       �	���KXc�A�*

lossA�<s�JC       �	Y1�KXc�A�*

loss�>�:JGp�       �	���KXc�A�*

loss�3;�W�       �	��KXc�A�*

loss��<�vA�       �	&�KXc�A�*

loss�p�=�;��       �	���KXc�A�*

loss�b�<��k9       �	�E LXc�A�*

loss��=�=�       �	�� LXc�A�*

loss22^=փ�       �	�lLXc�A�*

loss� >;�>�o       �	��LXc�A�*

loss�I�;����       �	y�LXc�A�*

loss��<au��       �	)LXc�A�*

lossŴ:�f&       �	E�LXc�A�*

loss���;���       �	�OLXc�A�*

loss<�fÎ       �	/�LXc�A�*

loss�~=�b�       �	�rLXc�A�*

loss�6<�~�       �	�LXc�A�*

losse�w;��m\       �	w�LXc�A�*

loss)��:}˻j       �	�2LXc�A�*

loss���<�Ze       �	M�LXc�A�*

loss�xU<d�       �	�oLXc�A�*

loss��C<���?       �	�	LXc�A�*

loss���=���K       �	@�	LXc�A�*

lossl�=�1       �	�9
LXc�A�*

loss>Xu>�       �	\�
LXc�A�*

loss��<��h^       �	�aLXc�A�*

lossɞ�=�e�       �	�LXc�A�*

lossQ�<0���       �	y�LXc�A�*

losss`�=렽       �	�#LXc�A�*

lossd�I;�A�       �	�LXc�A�*

loss{E<�p��       �	PLXc�A�*

loss?�=-=o�       �	*�LXc�A�*

loss�m;�A��       �	�}LXc�A�*

loss}��<�9S       �	LXc�A�*

lossH��;3#��       �	v�LXc�A�*

lossv��;�r��       �	�NLXc�A�*

loss��!<��a|       �	��LXc�A�*

lossn��<S���       �	ÝLXc�A�*

loss4N�<>�!       �	�ZLXc�A�*

loss���<*0�       �	a�LXc�A�*

loss1ö<n/��       �	�&LXc�A�*

loss:�M=%M�       �	BLXc�A�*

loss,�+<���       �	�#LXc�A�*

loss���<��C�       �	>\LXc�A�*

loss�>t;���       �	��LXc�A�*

loss�;<L�A       �	I�LXc�A�*

loss�W�<�%�?       �	LXc�A�*

loss��m;,l�o       �	��LXc�A�*

loss�:�,6�       �	�JLXc�A�*

loss�.~<o6�       �	��LXc�A�*

loss���=�
�       �	�tLXc�A�*

loss1Q=S�i       �	$
LXc�A�*

loss3v�<ͧ:�       �	f�LXc�A�*

loss@�;�
��       �	�:LXc�A�*

loss��A:�x'       �	��LXc�A�*

loss��4=A}U�       �	d LXc�A�*

loss�µ=����       �	�� LXc�A�*

loss�	�;��       �	��!LXc�A�*

loss�h�<%���       �	�"LXc�A�*

loss�^�<��#�       �	��"LXc�A�*

loss�=���*       �	YL#LXc�A�*

loss��<""~B       �	e�#LXc�A�*

loss�?t<���$       �	�w$LXc�A�*

loss(kM<�L�#       �	#%LXc�A�*

loss!5<,=U�       �	ڬ%LXc�A�*

loss���;��4       �	�>&LXc�A�*

loss��H=����       �	��&LXc�A�*

lossJr=��2       �	i'LXc�A�*

loss�p�=��       �	y(LXc�A�*

loss�^<�qd       �	��(LXc�A�*

lossc�:އ!�       �	=*LXc�A�*

loss�<��_       �	t�*LXc�A�*

loss>�=t�-�       �	f+LXc�A�*

loss�4�<ңsn       �	Z�,LXc�A�*

loss���=�XY       �	{.-LXc�A�*

lossWh�< gp�       �	�-LXc�A�*

losstC<i� �       �	Y.LXc�A�*

lossâ�=e�       �	=�.LXc�A�*

loss��E;H���       �	�/LXc�A�*

lossR�= ��9       �	>0LXc�A�*

lossA<�O;�       �	N�0LXc�A�*

lossRA�;f"��       �	�l1LXc�A�*

loss�r>�+�z       �	�2LXc�A�*

loss��= ,�       �	��2LXc�A�*

loss�j<3��       �	d=3LXc�A�*

loss�+�;�б�       �	8�3LXc�A�*

loss��+=���       �	8j4LXc�A�*

lossw�D=p��       �	.5LXc�A�*

lossq&:.b       �	E�5LXc�A�*

loss�<S�G�       �	�Q6LXc�A�*

lossdZ(<E/��       �	��6LXc�A�*

loss�:<Rge       �	(�7LXc�A�*

lossێ�=�ƀ�       �	s,8LXc�A�*

lossjg�=^Z       �	�8LXc�A�*

loss�(<3]�       �	�V9LXc�A�*

loss$~�:0��       �	?�9LXc�A�*

loss�N�<T��"       �	�}:LXc�A�*

lossr
�;����       �	�;LXc�A�*

loss��O;��V�       �	*�;LXc�A�*

lossC��<��d       �	�E<LXc�A�*

loss$);DJ�w       �	v5=LXc�A�*

loss�]�;p`�       �	��=LXc�A�*

lossA�<��X[       �	�m>LXc�A�*

lossA<�_�       �	t	?LXc�A�*

loss�v=�`�       �	��?LXc�A�*

losse��=��"       �	O;@LXc�A�*

loss�Gx=9���       �	��@LXc�A�*

loss�<�l�&       �	�kALXc�A�*

loss
<I�!       �	�BLXc�A�*

losss�<�7��       �	�BLXc�A�*

loss�+<㊦\       �	�/CLXc�A�*

lossa�&=p���       �	��CLXc�A�*

losst�<��8�       �	o�DLXc�A�*

lossX�W=$Gm       �	+ELXc�A�*

loss�.B:k�eP       �	\�ELXc�A�*

lossSz�=V���       �	0cFLXc�A�*

lossE)�=f�+y       �	��FLXc�A�*

loss�99^��       �	C�GLXc�A�*

loss���;��m       �	�(HLXc�A�*

loss�y�;PC�N       �	�HLXc�A�*

loss���<��       �	�OILXc�A�*

loss��k<��       �	�ILXc�A�*

loss,�<&A�       �	{�JLXc�A�*

loss4Ul<P�       �	� KLXc�A�*

loss��;乞�       �	'�KLXc�A�*

lossL��;���       �	�SLLXc�A�*

loss1[�<�C;�       �	��LLXc�A�*

loss���<ٗ|�       �	tzMLXc�A�*

loss\�a<V~�       �	VNLXc�A�*

lossg�<�~�{       �	�NLXc�A�*

loss�Y
=B�%�       �	_EOLXc�A�*

loss(9;5D)<       �	��OLXc�A�*

loss�;�C��       �	��PLXc�A�*

loss�3=���{       �	}QLXc�A�*

loss� =����       �	ZRLXc�A�*

loss���<f��       �	�RLXc�A�*

loss���<���L       �	�JSLXc�A�*

loss-�</j�       �	�ULXc�A�*

loss���<0�k�       �	��ULXc�A�*

loss��d=P�       �	XTVLXc�A�*

loss��;u"��       �	%�VLXc�A�*

lossk�=x�ڝ       �	��WLXc�A�*

loss�H=�s       �	�XLXc�A�*

loss�.<�s       �	�XLXc�A�*

loss�3!=1��       �	�?YLXc�A�*

lossӥ!=���H       �	��YLXc�A�*

losskP=ގ        �	�hZLXc�A�*

loss���9��#,       �	��ZLXc�A�*

loss�g;�j��       �	�[LXc�A�*

loss���<��p       �	?5\LXc�A�*

loss�_�=;��B       �	��\LXc�A�*

lossa��;*�(�       �	b]LXc�A�*

loss�!F=��       �	�^LXc�A�*

loss�=�,[�       �	�^LXc�A�*

loss��<?��       �	1@_LXc�A�*

loss+.=k�-       �	��_LXc�A�*

loss���<�Tnq       �	�z`LXc�A�*

loss�X�=�H��       �	�aLXc�A�*

loss�@	<a"��       �	�aLXc�A�*

loss
Q�;^5��       �	�HbLXc�A�*

lossd��=y4@<       �	��bLXc�A�*

loss��<~NkR       �	GucLXc�A�*

loss{�A=�2)�       �	uVdLXc�A�*

lossEKz;ј N       �	��dLXc�A�*

loss��;=h�\       �	��eLXc�A�*

loss��<�t��       �	�fLXc�A�*

loss���:���j       �	A�fLXc�A�*

loss�B�<i�.v       �	�HgLXc�A�*

loss>�=��J       �	�ShLXc�A�*

loss:8
<O͎!       �	��hLXc�A�*

loss{pp;ʲ'       �	ӥiLXc�A�*

loss|A=Ƒ��       �	uYjLXc�A�*

loss��<vf�       �	r�jLXc�A�*

loss�0n<�⅀       �	^�kLXc�A�*

loss��=hZ1�       �	��lLXc�A�*

loss<@5<�v       �	znmLXc�A�*

loss3��;Ԇ��       �	mnLXc�A�*

loss��G<j��<       �	R�nLXc�A�*

lossc�S<j���       �	T7oLXc�A�*

loss�Q�;P�d�       �	o�oLXc�A�*

loss3!=��V       �	7�pLXc�A�*

loss@�9��b       �	nnqLXc�A�*

lossr��<��M�       �	�rLXc�A�*

lossһN<gM��       �	s�rLXc�A�*

loss��=Z���       �	ysLXc�A�*

lossDn=�6
       �	: tLXc�A�*

loss�&<5�A�       �	V�tLXc�A�*

loss\��<'��h       �	�TuLXc�A�*

loss��F<���       �	0�uLXc�A�*

loss?n�<��j       �	;�vLXc�A�*

loss�u�=E6ɮ       �	�$wLXc�A�*

lossDi�<.G��       �	4�wLXc�A�*

loss��1=荫}       �	exLXc�A�*

lossc��=��X       �	�yLXc�A�*

lossC��<K@ɔ       �	�yLXc�A�*

loss}"
=+�9       �	�,zLXc�A�*

loss�ɜ<E�sc       �	��zLXc�A�*

loss�z�:{�{V       �	[{LXc�A�*

loss�A�;^�h�       �	0�{LXc�A�*

lossΜ<�>�i       �	�|LXc�A�*

lossX��;��V�       �	�(}LXc�A�*

loss[��<6���       �	��}LXc�A�*

lossx��<����       �	bK~LXc�A�*

loss�@�<�?X�       �	��~LXc�A�*

loss�y�:9g5        �	�qLXc�A�*

loss�w�<��Is       �	h�LXc�A�*

loss�=0��       �	c��LXc�A�*

loss�$j;���       �	Y2�LXc�A�*

loss��;���       �	7āLXc�A�*

loss��^<��@�       �	�b�LXc�A�*

loss(�)<�&��       �	b��LXc�A�*

loss��:���x       �	��LXc�A�*

loss��=Һ8�       �	�$�LXc�A�*

loss��"=�j�       �	�u�LXc�A�*

loss���;c17       �	�	�LXc�A�*

loss�ܩ<���%       �	��LXc�A�*

lossx�`<B���       �	v7�LXc�A�*

lossc	<���r       �	hЇLXc�A�*

loss���;O\�       �	�f�LXc�A�*

loss�8Z=@S�       �	+��LXc�A�*

loss8�l=�C�       �	�LXc�A�*

loss��K=	Je�       �	�.�LXc�A�*

loss�D�;�u�       �	+ÊLXc�A�*

loss��=�l�z       �	hX�LXc�A�*

lossz�,;kx       �	��LXc�A�*

loss�,�;�*h�       �	ƅ�LXc�A�*

loss��=�4]       �	��LXc�A�*

loss
\m=?%�!       �	$��LXc�A�*

loss��V;��H       �	�O�LXc�A�*

loss��<-���       �	)�LXc�A�*

loss
��;:1�_       �	'��LXc�A�*

loss�g�=}�J�       �	}"�LXc�A�*

loss�1=6�R�       �	@��LXc�A�*

loss�n�=.��i       �	�u�LXc�A�*

lossX�n<��%       �	�LXc�A�*

loss{<9�S�       �	%��LXc�A�*

lossI�)<�|�       �	
��LXc�A�*

loss�gd;ĳ�*       �	�\�LXc�A�*

loss�|�<���       �	S�LXc�A�*

loss��c;���y       �	XǕLXc�A�*

loss�\B<���       �	֋�LXc�A�*

loss)��<#��K       �	\=�LXc�A�*

loss�@�<��H       �	`�LXc�A�*

loss�V�<�a�r       �	���LXc�A�*

loss;B?<�T�@       �	�J�LXc�A�*

loss��<H       �	��LXc�A�*

loss֔E<& T       �	0՚LXc�A�*

loss,�N;J|n4       �	L�LXc�A�*

loss./�<�PF       �	���LXc�A�*

lossR��=5���       �	$�LXc�A�*

loss��<	�
�       �	��LXc�A�*

losso�=�hxk       �	�M�LXc�A�*

loss�r�:�̦s       �	�LXc�A�*

lossQ�<�9�7       �	D��LXc�A�*

loss]�-=��       �	2!�LXc�A�*

loss�j~:Lm�d       �	ѠLXc�A�*

loss�O�<s�KL       �	�n�LXc�A�*

loss�rE=(!��       �	s�LXc�A�*

loss�5F<��       �	���LXc�A�*

lossY�;��b       �	;�LXc�A�*

loss��c<�~�       �	�֣LXc�A�*

loss�c=�I�&       �	�j�LXc�A�*

loss�=���       �	��LXc�A�*

losss_)=�L       �	ۧ�LXc�A�*

loss�tK<���       �	NF�LXc�A�*

loss�K�<�,R       �	s٦LXc�A�*

loss�Э<�0       �	�t�LXc�A�*

loss�x|=��F       �	�
�LXc�A�*

lossQ�;�Z��       �	Ǡ�LXc�A�*

loss�d�;8^R       �	�ީLXc�A�*

lossl�~<�Z�       �	t�LXc�A�*

loss
=�MO�       �	��LXc�A�*

loss��<�e-M       �	�ȫLXc�A�*

loss��:;�_o       �	�m�LXc�A�*

loss�<���U       �	v�LXc�A�*

loss��x;@�Mh       �	!��LXc�A�*

loss�Ǎ<�I;       �	�N�LXc�A�*

lossC/:���       �	 �LXc�A�*

loss,]=H�j�       �	¦�LXc�A�*

loss�$�9}d9	       �	�9�LXc�A�*

loss�c<�(�u       �	q̱LXc�A�*

lossǁ=��]�       �	�l�LXc�A�*

loss���:7
z       �	�LXc�A�*

loss��M<R���       �	s��LXc�A�*

lossv�y:n�#�       �	�4�LXc�A�*

lossC��;R�;�       �	�ѴLXc�A�*

loss�z�<�X�       �	,d�LXc�A�*

loss
c>�3G       �	+��LXc�A�*

loss���=���I       �	ҍ�LXc�A�*

lossAS=٘�A       �	"�LXc�A�*

loss(�;z�r       �	J��LXc�A�*

losst�=T��       �	�V�LXc�A�*

loss%CE=<�       �	��LXc�A�*

loss3�;���       �	X��LXc�A�*

loss��":�t��       �	�O�LXc�A�*

loss؎<�?��       �	��LXc�A�*

loss�<,d��       �	#��LXc�A�*

loss���<��T�       �	��LXc�A�*

lossm�<��kZ       �	�LXc�A�*

lossL��<iQ�r       �	&T�LXc�A�*

loss;&$:z&��       �	��LXc�A�*

lossM&!=+)&�       �	���LXc�A�*

loss��=�
��       �	|&�LXc�A�*

loss��;��a       �	;��LXc�A�*

loss���<����       �	}X�LXc�A�*

loss_��;<4       �	���LXc�A�*

loss�=�&��       �	���LXc�A�*

lossV��=u��       �	�:�LXc�A�*

loss�%�<�@       �	���LXc�A�*

loss]<�B|       �	n�LXc�A�*

lossI��<�ZW        �	(�LXc�A�*

loss.<.���       �	`��LXc�A�*

loss�<�Lqc       �	�N�LXc�A�*

loss�y=z���       �	J��LXc�A�*

loss<h       �	H��LXc�A�*

loss��<c���       �	�)�LXc�A�*

loss���<R>1�       �	���LXc�A�*

loss&�D<�kɖ       �	~n�LXc�A�*

loss�f�;��p       �	�	�LXc�A�*

loss91=$�_       �	��LXc�A�*

loss��:i�+�       �	�M�LXc�A�*

losst�J<���       �	k��LXc�A�*

lossj�<[�K�       �	d��LXc�A�*

loss@Ζ=i�n       �	2�LXc�A�*

loss��"=*�͕       �	��LXc�A�*

loss@��<��'K       �	U��LXc�A�*

loss�\�<���       �	�B�LXc�A�*

loss;�=6`��       �	��LXc�A�*

loss��<54��       �	J��LXc�A�*

loss���;���6       �	�R�LXc�A�*

loss���:[b��       �	9��LXc�A�*

lossn�^<�.�       �	���LXc�A�*

loss]
=l�        �	\�LXc�A�*

loss t<��A�       �	���LXc�A�*

loss��(;��C       �	,��LXc�A�*

loss��=��5~       �	.��LXc�A�*

lossLf�;?G.       �	H��LXc�A�*

loss�_F<�xG�       �	�=�LXc�A�*

loss�	W<�B�       �	�?�LXc�A�*

loss:ʴ<��Z       �	O��LXc�A�*

loss�m;r�'       �	���LXc�A�*

loss��8H�B�       �	�t�LXc�A�*

loss�x<�i�F       �	a�LXc�A�*

loss�':�M�       �	�,�LXc�A�*

loss-?;0��:       �	���LXc�A�*

lossE�s=`%
5       �	�r�LXc�A�*

loss��3:E1�       �	�	�LXc�A�*

lossry<J�S       �	<��LXc�A�*

lossRq=:�%�       �	x��LXc�A�*

lossc7�sg       �	,��LXc�A�*

loss�b:��?       �	Vc�LXc�A�*

lossV�=X6D�       �	5&�LXc�A�*

loss-��=Q��       �	A�LXc�A�*

loss�Y=O��Z       �	���LXc�A�*

loss��D9�|�v       �	��LXc�A�*

loss���;�4	Q       �	 b�LXc�A�*

lossX��=�T�       �	�Y�LXc�A�*

loss��;�`�y       �	�M�LXc�A�*

lossy� >�W;�       �	�D�LXc�A�*

loss�=.��       �	R��LXc�A�*

loss�>{��X       �	`��LXc�A�*

loss���<{��       �	���LXc�A�*

loss�t�;�mI�       �	��LXc�A�*

lossq��<���       �	B��LXc�A�*

loss`<PS+i       �	N`�LXc�A�*

loss�=�       �	o~�LXc�A�*

loss�q�=#<�G       �	�6�LXc�A�*

loss�><M�"�       �	3P�LXc�A�*

loss!�+=٧{       �	�0�LXc�A�*

loss;vm=]��3       �	���LXc�A�*

loss
3�;��U�       �	6s�LXc�A�*

loss�f�=�g�"       �	��LXc�A�*

loss���=O�|V       �	���LXc�A�*

loss��3=���P       �	�Y�LXc�A�*

lossU6�<$n�       �	Y��LXc�A�*

loss�j9=��_       �	-��LXc�A�*

loss�u<Ϸa       �	�%�LXc�A�*

lossr<=�T       �	½�LXc�A�*

loss�Q=���       �	�Z�LXc�A�*

loss��*=��7.       �	]��LXc�A�*

loss��>;��F4       �	>��LXc�A�*

loss��r:��kI       �	)�LXc�A�*

loss��;�  r       �	��LXc�A�*

loss��=�-2       �	���LXc�A�*

loss��%=o��       �	 s�LXc�A�*

loss-0<Vy�|       �	��LXc�A�*

loss|�k=|@!       �	ع�LXc�A�*

loss��2=�V@r       �	�]�LXc�A�*

loss���<���       �	��LXc�A�*

lossZ�;pԍ�       �	֎�LXc�A�*

lossm�K<M�S�       �	�&�LXc�A�*

losss;���       �	��LXc�A�*

loss���<(��       �	�L�LXc�A�*

loss��[<\K�7       �	b��LXc�A�*

loss��<U{�       �	ǜ MXc�A�*

loss8Y�=�u�       �	�3MXc�A�*

loss���;��%�       �	�MXc�A�*

lossl�:�D��       �	�`MXc�A�*

lossֿQ=%� �       �	��MXc�A�*

loss���<uY��       �	o�MXc�A�*

loss4�C=Ia�       �	GMXc�A�*

loss���<|�       �	z�MXc�A�*

loss�'=('�=       �	V�MXc�A�*

lossm/�=[��       �	�+MXc�A�*

loss< <l�       �	��MXc�A�*

loss��<[_�J       �	�tMXc�A�*

loss,��<�P:�       �	IMXc�A�*

loss�>]<�|w�       �	z�MXc�A�*

losss�T;�       �	}�!MXc�A�*

loss��<��       �	ZE"MXc�A�*

loss��]=�Ȑ       �	�"MXc�A�*

loss��<���       �	�x#MXc�A�*

loss.U�<HV�       �	<$MXc�A�*

loss� ;�IJ�       �	u�$MXc�A�*

lossi5<�P_�       �	�@%MXc�A�*

loss��<��F�       �	��%MXc�A�*

losst�=���       �	Ou&MXc�A�*

loss���<�[�D       �	�'MXc�A�*

loss0T�;6�_�       �	+�'MXc�A�*

loss���<�G       �	WA(MXc�A�*

lossή�;;�J       �	|�(MXc�A�*

loss��[<V�2/       �	Pp)MXc�A�*

loss6�=Rp��       �	�*MXc�A�*

lossčt=�e�       �	F�*MXc�A�*

loss��:�D��       �	�]+MXc�A�*

loss;��<Ҳ)"       �	��,MXc�A�*

loss��=V       �	du-MXc�A�*

loss ��<�&�       �	.MXc�A�*

lossH�T<\|��       �	O/MXc�A�*

loss��N=���       �	�/MXc�A�*

loss���;A�i       �	5]0MXc�A�*

loss��=�Sg�       �	Z1MXc�A�*

loss1s�;�%       �	z�1MXc�A�*

lossҷ�;n��       �	�>2MXc�A�*

loss��=]�M�       �	9�2MXc�A�*

loss���:�^��       �	�v3MXc�A�*

lossl�H<sqPo       �	?4MXc�A�*

loss �=�'g       �	U5MXc�A�*

lossi��;l?A�       �	!6MXc�A�*

loss�U4;\�%�       �	˺6MXc�A�*

loss��<U�8       �	P7MXc�A�*

lossp�;��4       �	e�7MXc�A�*

loss�);=�3       �	��8MXc�A�*

loss>;<��qP       �	Q9MXc�A�*

loss|�1<?���       �	��9MXc�A�*

loss��g=�Gan       �	φ:MXc�A�*

loss���=���>       �	8;MXc�A�*

loss�Ӏ=��%       �	��;MXc�A�*

loss�<��M;       �	��<MXc�A�*

lossO7�<xn�        �	S$=MXc�A�*

loss5��<V���       �	��=MXc�A�*

loss�2Y;v⁣       �	-Z>MXc�A�*

lossB�"=��F�       �	� ?MXc�A�*

loss�*=r���       �	4�?MXc�A�*

loss�P=����       �	78@MXc�A�*

lossE�{;?�       �	�@MXc�A�*

loss��i:��C<       �	mqAMXc�A�*

loss���<�3\B       �	�BMXc�A�*

loss�:s}�       �	HCMXc�A�*

loss�=U�u       �	=�CMXc�A�*

loss��2:���1       �	�cDMXc�A�*

lossqU�=�Ĭ�       �	�EMXc�A�*

loss�$<�_+=       �	�EMXc�A�*

lossQ�l;��"�       �	�KFMXc�A�*

losscF�<��       �	82GMXc�A�*

loss�;	,Ti       �	��GMXc�A�*

lossz�S;8NQ�       �	roHMXc�A�*

loss�=#=ct~       �	�IMXc�A�*

lossb�>!�C�       �	��IMXc�A�*

loss;n.;A��       �	E�JMXc�A�*

loss`=�;]AO       �	�`KMXc�A�*

losss��<��ŗ       �	��KMXc�A�*

loss)�<�w��       �	�LMXc�A�*

lossIg;U�"�       �	:=MMXc�A�*

loss�7=�x�M       �	j�MMXc�A�*

loss-��;�Ou�       �	(|NMXc�A�*

loss�A�<�\��       �	�OMXc�A�*

loss�(�=�Q�_       �	��OMXc�A�*

loss���;Ų��       �	�ZPMXc�A�*

loss���;Pl�       �	�&QMXc�A�*

loss��r;@A?\       �	�QMXc�A�*

loss��V=�r��       �	W`RMXc�A�*

lossV��<�n!O       �	 
SMXc�A�*

loss��o;����       �	y�SMXc�A�*

loss	A�;G5��       �	&mTMXc�A�*

loss�a=d*��       �	_`UMXc�A�*

loss��<����       �	�	VMXc�A�*

loss,�;�j[�       �	��VMXc�A�*

loss�9L<���:       �	aWMXc�A�*

loss�=(�U�       �	�XMXc�A�*

loss�4�<7R�&       �	��XMXc�A�*

loss<}�:��$       �	W@YMXc�A�*

loss��T<cq�l       �	��YMXc�A�*

losss�<�w�        �	�ZMXc�A�*

loss�l�<VK�B       �	�![MXc�A�*

loss\} <�<�K       �	1�[MXc�A�*

losst�<��K       �	�G\MXc�A�*

lossj=Q=����       �	��\MXc�A�*

loss���<�R       �	|]MXc�A�*

loss�ge<>�u       �	�^MXc�A�*

loss���<_��       �	�^MXc�A�*

loss2ܙ=-�`       �	bL_MXc�A�*

loss���=Y���       �	n�_MXc�A�*

loss�7n<- X       �	�x`MXc�A�*

loss8�-=3���       �	caMXc�A�*

loss���;*�~m       �	��aMXc�A�*

lossܘ�<�Y��       �	�ObMXc�A�*

loss�Ѣ:bG��       �	I�bMXc�A�*

lossv�'<
�=�       �	H�cMXc�A�*

loss\�{=�n�       �	!dMXc�A�*

lossE6;�]       �	J�dMXc�A�*

lossA&�:���2       �	�oeMXc�A�*

loss�|*<�9�B       �	�fMXc�A�*

lossѡ<��!�       �	R�fMXc�A�*

lossx�:$'       �	.8gMXc�A�*

loss��:}p}       �	��gMXc�A�*

loss�(�<r}&�       �	�vhMXc�A�*

loss���<�=�       �	�iMXc�A�*

lossJV�=�Oq~       �	�iMXc�A�*

loss��I=�(       �	�IjMXc�A�*

loss���;"�܊       �	�EkMXc�A�*

loss��;��5       �	��kMXc�A�*

loss��:pV��       �	�ylMXc�A�*

loss��;$?Y�       �	�mMXc�A�*

loss6<�E#r       �	��mMXc�A�*

lossj4<T. f       �	canMXc�A�*

loss��;�?�       �	oMXc�A�*

loss�l�;A&{�       �	ǟoMXc�A�*

loss�R<�b�)       �	�@pMXc�A�*

lossN}�;s��O       �	��pMXc�A�*

lossV�;��}       �	��qMXc�A�*

loss�=�<��O       �	TrMXc�A�*

loss#��:�d�^       �	£sMXc�A�*

loss�+�<5�ܐ       �	��tMXc�A�*

lossQ�A:x�-�       �	��uMXc�A�*

loss�r�:�ݻ�       �	�(vMXc�A�*

loss��&=P	Q�       �	e�vMXc�A�*

loss�R�<8���       �	�ewMXc�A�*

loss�T�93Mt       �	-xMXc�A�*

loss��;0=��       �	^�xMXc�A�*

lossJ��<�y(^       �	xFyMXc�A�*

loss��=J�+�       �	��yMXc�A�*

loss�J<�#4?       �	�tzMXc�A�*

lossv[*;>@       �	�{MXc�A�*

loss_c!=tޞ�       �	Q�{MXc�A�*

loss�;Cz       �	�E|MXc�A�*

loss|2;��'       �	0�|MXc�A�*

loss��:���       �	�u}MXc�A�*

loss`��:V�W�       �	#~MXc�A�*

loss�# <� �[       �	ު~MXc�A�*

loss��=��R       �	GMXc�A�*

loss���<y�[       �	��MXc�A�*

loss��=��|�       �	Kw�MXc�A�*

loss��`;�e�<       �	E�MXc�A�*

loss�2_;����       �	���MXc�A�*

loss��9=I<o�       �	�D�MXc�A�*

loss ��:M��       �	 �MXc�A�*

loss�x<<�       �	1y�MXc�A�*

loss$��:9a�e       �	��MXc�A�*

lossd��;�(�       �	���MXc�A�*

loss�d�;H��       �	.<�MXc�A�*

loss��<S�       �	�υMXc�A�*

loss$�<�o�+       �	�h�MXc�A�*

loss�e:�`       �	� �MXc�A�*

loss��0<2!p       �	A��MXc�A�*

loss�A�<ߍt�       �	�9�MXc�A�*

loss�6�<h�"N       �	�ӈMXc�A�*

loss�j�<S��       �	+i�MXc�A�*

loss��=��z'       �	��MXc�A�*

loss�F=<M]�       �	6��MXc�A�*

loss$�;,˭]       �	�*�MXc�A�*

loss�iq:�� �       �	f��MXc�A�*

loss�g;?<��       �	�L�MXc�A�*

loss�`�:��x       �	[�MXc�A�*

loss.�4;��       �	���MXc�A�*

loss��w=O�˟       �	�(�MXc�A�*

loss�'�;�jZo       �	ܻ�MXc�A�*

loss��-;�8D        �	N�MXc�A�*

loss�i.=�e��       �	��MXc�A�*

loss�t=:m���       �	؁�MXc�A�*

loss�ը<u��>       �		�MXc�A�*

loss� Q=��A$       �	��MXc�A�*

lossݾ�:�b�7       �	D�MXc�A�*

lossW��;�,nC       �	�֒MXc�A�*

loss�g'<?�_       �	��MXc�A�*

loss��<p�5�       �	2u�MXc�A�*

lossq<��m�       �	���MXc�A�*

loss=�D\�       �	}�MXc�A�*

loss� :;~���       �	��MXc�A�*

losso��<��       �	��MXc�A�*

lossR�;ں       �	%�MXc�A�*

loss!L<:B���       �	.��MXc�A�*

loss
�L<xOș       �	�=�MXc�A�*

loss�-�:����       �	��MXc�A�*

loss�H�<� $s       �	��MXc�A�*

loss?lH<% ,<       �	dϜMXc�A�*

loss=�fg�       �	�r�MXc�A�*

loss�<;�;��       �	 �MXc�A�*

lossJ�7<�A       �	f��MXc�A�*

lossjk<��7       �	�6�MXc�A�*

loss�i�:�U��       �	ПMXc�A�*

lossN#3;�2�       �	�l�MXc�A�*

losscIt;��)       �	?�MXc�A�*

loss6<H�;[       �	Ü�MXc�A�*

loss�`=<�P?       �	cB�MXc�A�*

loss���<:���       �	bܢMXc�A�*

loss��<e��       �	�v�MXc�A�*

loss���<��5�       �	$�MXc�A�*

loss&-<Y?��       �	M��MXc�A�*

lossE�#=�5p       �	&6�MXc�A�*

loss���<z%X
       �	7��MXc�A�*

lossL& =�A%�       �	` �MXc�A�*

loss��y<|٠�       �	��MXc�A�*

loss]^;g��$       �	�]�MXc�A�*

loss>8;�H"�       �	���MXc�A�*

loss�8:oZ��       �	��MXc�A�*

loss��=�d(�       �	w/�MXc�A�*

loss�R`=|"�       �	rĪMXc�A�*

loss��;D%�`       �	�`�MXc�A�*

lossI�f<�D��       �	���MXc�A�*

loss��;�#�F       �	H��MXc�A�*

loss�O�<���6       �	�`�MXc�A�*

loss���;�(h�       �	���MXc�A�*

lossϖ�<YsJ�       �	6��MXc�A�*

loss�� >>�4I       �	�.�MXc�A�*

lossFۙ;�g��       �	�įMXc�A�*

loss�<����       �	�^�MXc�A�*

loss3<LOm       �	��MXc�A�*

loss<vQ<*�ss       �	$��MXc�A�*

loss0�;�9�       �	2�MXc�A�*

loss�T8<j;�       �	eòMXc�A�*

loss��=b7}       �	W]�MXc�A�*

losscc�:�)�f       �	���MXc�A�*

lossR��;��'       �	���MXc�A�*

loss h=�X��       �	,�MXc�A�*

loss�X;`��       �	�еMXc�A�*

lossث=�+�]       �	�u�MXc�A�*

loss�4�;G�V3       �	�,�MXc�A�*

loss
j�;f       �	�ͷMXc�A�*

lossj�;��E       �	+h�MXc�A�*

lossV�<�5X       �	��MXc�A�*

loss�. <�O��       �	$��MXc�A�*

loss;���u       �	i7�MXc�A�*

loss���<d��       �	�̺MXc�A�*

lossVA=A<       �	|b�MXc�A�*

loss�=�/�K       �	���MXc�A�*

loss���<�y�       �	m��MXc�A�*

loss�:`���       �	�$�MXc�A�*

losss�-<�l�       �	���MXc�A�*

loss�d<��       �	CT�MXc�A�*

loss�3=�J��       �	u�MXc�A�*

lossiY�<SH&i       �	�{�MXc�A�*

loss|�M=���       �	��MXc�A�*

loss��q;�d�       �	���MXc�A�*

loss�2=�2       �	fM�MXc�A�*

loss�q�<���       �	���MXc�A�*

loss��f=�Ak�       �	b��MXc�A�*

loss��<+��       �	��MXc�A�*

loss�4�;fǨ~       �	h��MXc�A�*

loss���=�t�       �	�G�MXc�A�*

loss��=��Җ       �	��MXc�A�*

loss�s�:'fe�       �	���MXc�A�*

loss瞇;W�1       �	e�MXc�A�*

loss6-�9�D
       �	���MXc�A�*

loss��~<5a=       �	W`�MXc�A�*

lossfŕ<��5�       �	��MXc�A�*

loss{�=^��       �	���MXc�A�*

loss���;εq+       �	a5�MXc�A�*

loss��<O��       �	$��MXc�A�*

loss��<t��       �	��MXc�A�*

loss+k<7Jh       �	�?�MXc�A�*

loss8;�~�       �	+��MXc�A�*

lossAm_<��t�       �	�|�MXc�A�*

loss{��9��O6       �	>&�MXc�A�*

lossl�j<n��       �	��MXc�A�*

loss��;vӆ4       �	�l�MXc�A�*

lossX4@<שF       �	N	�MXc�A�*

loss�Fm<�	&       �	��MXc�A�*

loss��p<2,W       �	jM�MXc�A�*

loss�J<'�A�       �	q��MXc�A�*

loss@�<��2$       �	���MXc�A�*

loss�p�;����       �	��MXc�A�*

lossXk>=��       �	k��MXc�A�*

loss���=aY8�       �	mS�MXc�A�*

losslA�;��ޥ       �	��MXc�A�*

loss�L:��"�       �	���MXc�A�*

loss|ˁ=�l�T       �	![�MXc�A�*

loss�)�<H�!k       �	H��MXc�A�*

loss��;IZ��       �	���MXc�A�*

losst%$=IS�       �	�'�MXc�A�*

loss:�U;�O�       �	ƿ�MXc�A�*

loss�=GGkT       �	�X�MXc�A�*

loss��<��2�       �	���MXc�A�*

loss��<��xN       �	À�MXc�A�*

lossX��=>�%       �	a�MXc�A�*

loss�?1;��7       �	G��MXc�A�*

loss��;B�Sl       �	�=�MXc�A�*

loss���:�e�       �	k��MXc�A�*

lossȑ!;As-       �	bg�MXc�A�*

loss409<O�3       �	���MXc�A�*

lossֻ;���       �	��MXc�A�*

loss��?:h3�A       �	W%�MXc�A�*

lossu�<�qL�       �	g��MXc�A�*

loss��=5�#�       �	bJ�MXc�A�*

loss��0<�       �	���MXc�A�*

loss�N�=?�q�       �	yu�MXc�A�*

lossv�;�4��       �	1�MXc�A�*

lossOF_<=c��       �	b��MXc�A�*

loss�%-=���       �	�/�MXc�A�*

losse��;}���       �	n��MXc�A�*

lossi =� �H       �	LT�MXc�A�*

loss7E	=F��5       �	���MXc�A�*

loss�ST=�G�y       �	���MXc�A�*

loss���<ģ.:       �	%�MXc�A�*

loss�>H<���       �	���MXc�A�*

loss���<�c�       �	R�MXc�A�*

lossߚ�<T۽3       �	���MXc�A�*

loss��<jx�       �	ɑ�MXc�A�*

lossH�;Hp�       �	+�MXc�A�*

loss#<���F       �	G��MXc�A�*

loss�˨=	��       �	d]�MXc�A�*

lossex�:�	�       �	.��MXc�A�*

loss�<O�Au       �	A��MXc�A�*

loss�F=;�P       �	�7�MXc�A�*

loss��^:u�       �	A��MXc�A�*

loss��<���       �	:��MXc�A�*

loss*]�<4���       �	'�MXc�A�*

lossȃd;�XN�       �	���MXc�A�*

loss1b*=Q0^       �	l�MXc�A�*

loss<��<�cQ       �	��MXc�A�*

loss)�+=����       �	U�MXc�A�*

lossLy�;�,P�       �	V��MXc�A�*

lossj�;=�`       �	���MXc�A�*

lossS,=�{�       �	�x�MXc�A�*

loss�='r|9       �	���MXc�A�*

loss;�;>,K�       �	� �MXc�A�*

loss��6=Z�d       �	���MXc�A�*

loss�P�=U�E�       �	�Q�MXc�A�*

loss���<�;��       �	C��MXc�A�*

loss@1�<X�x�       �	Fx�MXc�A�*

lossh0�;����       �	�
�MXc�A�*

loss��:�}#       �	���MXc�A�*

lossC�9=
*       �	�5�MXc�A�*

lossԻ�;2���       �	f�MXc�A�*

loss��j<g:�       �	���MXc�A�*

lossS�<��4�       �	R�MXc�A�*

loss[��<��0       �	.��MXc�A�*

lossHI*<�$�v       �	�y�MXc�A�*

loss<;��]�       �	��MXc�A�*

loss���:��~�       �	��MXc�A�*

loss���;Ns       �	$F�MXc�A�*

loss�ɝ;.�9G       �	���MXc�A�*

loss][�={J��       �	�}�MXc�A�*

loss
#�<7f�       �	a�MXc�A�*

loss�}�<a�P{       �	 ��MXc�A�*

lossjW=ė�~       �	�E NXc�A�*

loss��[<�QB�       �	�� NXc�A�*

lossi5:<�K�X       �	�~NXc�A�*

loss���;\l��       �	�NXc�A�*

loss�0;�ڐ       �	��NXc�A�*

lossc:�=�Q�       �	1ENXc�A�*

lossZC9<�F��       �	��NXc�A�*

loss�3<�Xړ       �	�mNXc�A�*

lossmj=%6       �	lNXc�A�*

loss��;�iM�       �	�NXc�A�*

loss�`�<��>       �	v6NXc�A�*

loss$c=�|�       �	��NXc�A�*

lossÔd=Pc2`       �	��NXc�A�*

loss��R=W�       �	k)NXc�A�*

losse6�<�m�       �	��NXc�A�*

loss��t;���(       �	se	NXc�A�*

lossp�=>K��       �	\
NXc�A�*

lossi"[=eϖ       �	8�
NXc�A�*

lossk@�=Fa�N       �	K<NXc�A�*

loss�ZO<۠�p       �	8�NXc�A�*

loss?|�;��k       �	�NXc�A�*

loss8{�;�p�.       �	�NXc�A�*

losso5<���       �	 �NXc�A�*

loss��V<����       �	�PNXc�A�*

loss���:f
N       �	�NXc�A�*

lossR9�;�Z�3       �	��NXc�A�*

loss���;3D�       �	GNXc�A�*

lossvy�<@Aw       �	{�NXc�A�*

lossj4�<Ĳ[}       �	�bNXc�A�*

loss= =�b�       �	7�NXc�A�*

loss���<Fjh�       �	=�NXc�A�*

loss7><ġ*�       �	�5NXc�A�*

loss�#?;ofq       �	�NXc�A�*

loss��p;��h�       �	P�NXc�A�*

loss[Kj<0��G       �	�-NXc�A�*

loss�tJ=1��       �	��NXc�A�*

loss,<����       �	�hNXc�A�*

loss�^:CA��       �	��NXc�A�*

loss) �<�rt       �	��NXc�A�*

loss	e=�y�       �	�7NXc�A�*

loss�P=�d��       �	��NXc�A�*

lossw 2=��e�       �	yNXc�A�*

loss���<�6�Q       �	|NXc�A�*

loss��;DU��       �	z�NXc�A�*

lossc#*=�"?       �	�NXc�A�*

loss���:�
�A       �	ȴNXc�A�*

loss��;�3       �	�ZNXc�A�*

loss1o�:]��5       �	#�NXc�A�*

loss1�=��Sz       �	�NXc�A�*

lossȄ�9|��6       �	
�NXc�A�*

loss��<.]�       �	� NXc�A�*

loss�<���       �	�� NXc�A�*

loss�U;@u�       �	0c!NXc�A�*

lossh�:Ѳ�       �	�
"NXc�A�*

lossL<�;3�#       �	�#NXc�A�*

loss�`�<�2��       �	#�#NXc�A�*

lossD�;&�%U       �	�$NXc�A�*

loss<ǰ=�w�L       �	�-%NXc�A�*

losshw<����       �	\�%NXc�A�*

loss7��:��8       �	��&NXc�A�*

loss&��<�֧       �	@'NXc�A�*

loss�X<��z       �	��'NXc�A�*

losslt<�S�       �	�t(NXc�A�*

loss}�C=�R2       �	J)NXc�A�*

loss	_K=;��       �	��)NXc�A�*

loss{);��k       �	�7*NXc�A�*

loss�\=/��E       �	t�*NXc�A�*

loss�4=9���       �	�t+NXc�A�*

loss��6;�F       �	 ,NXc�A�*

loss��:v[�=       �	4�,NXc�A�*

loss�Zc<�e�       �	�-NXc�A�*

loss��=<Q       �	r7.NXc�A�*

loss�O�;��       �	�/NXc�A�*

loss��5<�8       �	Z�/NXc�A�*

lossLQ�<���       �	3S0NXc�A�*

loss|L_;&�       �	��0NXc�A�*

loss��b=}��       �	�/2NXc�A�*

loss�P<��       �	)!3NXc�A�*

loss��+=Vi�@       �	��3NXc�A�*

loss��<��d�       �	&V4NXc�A�*

loss�$l<��U�       �	/�5NXc�A�*

loss/
P<��       �	�6NXc�A�*

loss��<ؒ(�       �	`7NXc�A�*

lossR�|<ō��       �	�a9NXc�A�*

loss㠺:kv�s       �	�:NXc�A�*

loss��.;��l       �	N�:NXc�A�*

loss�Vt;]��       �	2�;NXc�A�*

loss�Q<�;|p       �	��<NXc�A�*

loss�m�<�U&�       �	#h=NXc�A�*

loss8�;�ϥ�       �	�?NXc�A�*

lossQ�A<G�y       �	�,@NXc�A�*

loss̽�:N�i�       �	�@NXc�A�*

loss�y�:$�c�       �	�|ANXc�A�*

loss���<J	�       �	�BNXc�A�*

lossŇ<έK       �	Z�BNXc�A�*

loss֯�;1��7       �	�aCNXc�A�*

lossNQ=�L�       �	� DNXc�A�*

loss�D=\�k       �	��DNXc�A�*

loss7��<Ry1       �	�;ENXc�A�*

loss��6=ȯ��       �	�ENXc�A�*

loss�EU<XU
�       �	FxFNXc�A�*

loss��=��>       �	�GNXc�A�*

loss���<1���       �	~�GNXc�A�*

loss�=��O�       �	IHHNXc�A�*

loss�.(;�Y`�       �	��HNXc�A�*

loss*�;�͛�       �	ʊINXc�A�*

loss���;Niz       �	M,JNXc�A�*

loss#��=�"K�       �	��JNXc�A�*

loss,�=���       �	1_KNXc�A�*

loss�=%[Ʃ       �	@�KNXc�A�*

loss�1w:�f�       �	��LNXc�A�*

loss_p�;��t�       �	�6MNXc�A�*

loss��;,�[T       �	2�MNXc�A�*

loss��U=K_��       �	�ONXc�A�*

loss�CF=z"       �	��ONXc�A�*

lossQ��;l�L       �	� QNXc�A�*

loss��=?��       �	w�QNXc�A�*

loss��.=���       �	6?RNXc�A�*

loss�4d<���       �	��RNXc�A�*

loss�H�<W���       �	WySNXc�A�*

loss)��;��v�       �	Q-TNXc�A�*

lossO�$=qR$�       �	#�TNXc�A�*

loss��<\>       �	%�UNXc�A�*

loss�?A=�ݵ�       �	YVNXc�A�*

loss��#<��ǒ       �	�)WNXc�A�*

loss#	�<�=s@       �	a�WNXc�A�*

loss�%"<zw�       �	�vXNXc�A�*

loss�,=�ZIW       �	7YNXc�A�*

loss�֔:��       �	��YNXc�A�*

loss`*`:��t       �	̵ZNXc�A�*

losst�B=�X�       �	K[NXc�A�*

lossD��<��       �	��[NXc�A�*

loss�	P=���       �	8�\NXc�A�*

loss��<$�@       �	F"]NXc�A�*

loss
��<Y�       �	r�]NXc�A�*

loss�<���       �	�`^NXc�A�*

loss���=8~��       �	��^NXc�A�*

loss�ʃ<�^�       �	؛_NXc�A�*

loss(�;��s�       �	H7`NXc�A�*

loss�d�;"��       �	��`NXc�A�*

loss-H�<j��       �	�maNXc�A�*

loss)I�<�L�       �	SbNXc�A�*

lossPW�<�1�       �	ܛbNXc�A�*

loss8�?=��o1       �	�>cNXc�A�*

lossx�v<�)Rx       �	��cNXc�A�*

lossxg�:���       �	kdNXc�A�*

lossŐ�<n�~#       �	2eNXc�A�*

loss�)%<�̎�       �	)�eNXc�A�*

loss��>����       �	�+fNXc�A� *

loss��~<qn��       �	��fNXc�A� *

loss�b�<�-��       �	�VgNXc�A� *

loss�-�<Rf��       �	�gNXc�A� *

loss1�=#��)       �	�hNXc�A� *

loss%��<��p�       �	�>iNXc�A� *

loss&SW;�Rb       �	{�iNXc�A� *

loss��-=�۲�       �	x{jNXc�A� *

loss�''<vK{k       �	ikNXc�A� *

loss��Z<y�P�       �	�kNXc�A� *

loss�:4=?޻�       �	XlNXc�A� *

loss�;n���       �	<�lNXc�A� *

loss1!<�1�       �	h�mNXc�A� *

loss��<q���       �	4,nNXc�A� *

loss�H�;YK(s       �	z�nNXc�A� *

losse�6:y%�e       �	�[oNXc�A� *

loss�f=OL��       �	�oNXc�A� *

loss";�{rC       �	��pNXc�A� *

loss:�<S��       �	�%qNXc�A� *

loss`~=+��       �	ϻqNXc�A� *

loss�<��ag       �	�NrNXc�A� *

lossn��=�&!       �	��rNXc�A� *

loss�+�;,:��       �	<�sNXc�A� *

lossH];��#       �	V)tNXc�A� *

loss��;� �       �	_�tNXc�A� *

lossH�<
��J       �	�uNXc�A� *

loss�^�:q+�|       �	wJvNXc�A� *

loss<�<oz�K       �	S�vNXc�A� *

loss��=|�ڵ       �	��wNXc�A� *

loss!�<B$B�       �	7xNXc�A� *

loss���<2��e       �	[�xNXc�A� *

loss{��<`��
       �	muyNXc�A� *

lossF�:��o�       �	RzNXc�A� *

lossaG�=� �       �	ͬzNXc�A� *

loss��6:4�=       �	fJ{NXc�A� *

loss�*;$��+       �	��{NXc�A� *

loss�=q��       �	||NXc�A� *

loss���;.�d       �	�}NXc�A� *

loss
\G99;?2       �	A�}NXc�A� *

loss()�;�Rl       �	�Y~NXc�A� *

loss�;::��       �	��~NXc�A� *

loss���<;�!       �	E�NXc�A� *

loss�+�:BP       �	��NXc�A� *

lossž�:B&�9       �	;��NXc�A� *

lossM"`=`���       �	�C�NXc�A� *

loss��9Kc(       �	u�NXc�A� *

lossc��8�1�_       �	5~�NXc�A� *

loss�x*8�8u�       �	��NXc�A� *

loss��w;}�78       �	ު�NXc�A� *

losscD#<2�#       �	F�NXc�A� *

loss*�;�]��       �	��NXc�A� *

loss���9DC��       �	�|�NXc�A� *

lossX�2=K�^V       �	��NXc�A� *

lossa�=�z��       �	G��NXc�A� *

lossd.�:G��       �	�D�NXc�A� *

lossXP
>��       �	
ڇNXc�A� *

loss��<��8�       �	zn�NXc�A� *

loss�cg=_��v       �	�NXc�A� *

lossQ�<Fz6�       �	%��NXc�A� *

loss/�;(@h�       �	�B�NXc�A� *

lossj;/��
       �	FӊNXc�A� *

loss|�<�Ɨ�       �	�u�NXc�A� *

lossʋ�; g��       �	=�NXc�A� *

loss���;ż"       �	#��NXc�A� *

lossػn<:N��       �	<�NXc�A� *

lossME=���@       �	�ЍNXc�A� *

loss}#=l�h�       �	n�NXc�A� *

loss��=�7Y�       �	��NXc�A� *

lossn2�<}5ף       �	Ӥ�NXc�A� *

loss��=X��       �	pA�NXc�A� *

loss3��:ei�       �	bڐNXc�A� *

lossq;�_�       �	�r�NXc�A� *

loss��'=��J�       �	�
�NXc�A� *

loss�Ea<�p       �	���NXc�A� *

loss���:V�,	       �	�O�NXc�A� *

lossa^�;ԟ:M       �	��NXc�A� *

lossҺ%<��=[       �	�~�NXc�A� *

lossFt;�x�0       �	N��NXc�A� *

lossd]S;��       �	�X�NXc�A� *

loss��<��]       �	�!�NXc�A� *

loss��=J+Qz       �	���NXc�A� *

loss��;oiy       �	-�NXc�A� *

loss*�=n�a�       �	獙NXc�A� *

loss#�=�fj       �	�W�NXc�A� *

lossv��;RŶ       �	=�NXc�A� *

loss.��<$f�       �	P�NXc�A� *

lossr��;�ם^       �	'��NXc�A� *

loss	�.95L�!       �	S^�NXc�A� *

loss�=���       �	8��NXc�A� *

loss�P�;�V�Q       �	��NXc�A� *

lossF�;nQ�       �	��NXc�A� *

loss�
<W��h       �	{��NXc�A� *

loss
N<��W       �	��NXc�A� *

loss��<L�Q       �	�!�NXc�A� *

losssp�:����       �	���NXc�A� *

loss�[�<9C�       �	@N�NXc�A� *

loss��
<	��M       �	2�NXc�A� *

loss$p�;)�       �	�v�NXc�A� *

lossWΉ;G��I       �	rM�NXc�A� *

loss�^�<k�3�       �	��NXc�A� *

lossZ=o���       �	/��NXc�A� *

lossR�3:3y��       �	�*�NXc�A� *

loss��c=b�{�       �	�¦NXc�A� *

loss��	;s�H       �	g�NXc�A� *

losse�3<��U       �	��NXc�A� *

loss얊;�d��       �	�J�NXc�A� *

loss�(�<�/�       �	��NXc�A� *

loss�=��       �	�r�NXc�A� *

loss�t�;�jT�       �		�NXc�A� *

loss&�/;#�K       �	���NXc�A� *

loss�i;���b       �	%@�NXc�A� *

loss3��<|H�       �	=��NXc�A� *

loss�sI=��"       �	�f�NXc�A� *

loss�K�<#�\�       �	\��NXc�A� *

loss���;���       �	���NXc�A� *

loss�Z�:�G<       �	�,�NXc�A� *

loss�d<_�       �	���NXc�A� *

lossDF<eĉ       �	�h�NXc�A� *

loss_�=�"��       �	��NXc�A� *

loss]��<L��       �	���NXc�A� *

loss_s�<i;%�       �	)?�NXc�A� *

loss�i�:/�6�       �	���NXc�A� *

lossk�<�kPw       �	6t�NXc�A� *

loss��!;�U�l       �	��NXc�A� *

loss��<7v�B       �	���NXc�A� *

loss�5s;�       �	�9�NXc�A� *

loss�>�=�]_�       �	���NXc�A� *

loss(��<���       �	�e�NXc�A� *

loss��:<ت�        �	���NXc�A�!*

loss&�<�-�!       �	͔�NXc�A�!*

loss��z<��#y       �	+�NXc�A�!*

loss:s�;�1S�       �	���NXc�A�!*

loss�\�<ґ
       �	N�NXc�A�!*

lossz"; �X+       �	���NXc�A�!*

loss�=�EKw       �	Ou�NXc�A�!*

loss��;�Ɨg       �	S	�NXc�A�!*

lossD�:�
-�       �	k��NXc�A�!*

loss�<�P�u       �	�1�NXc�A�!*

lossoc�;�q�x       �	���NXc�A�!*

loss���;+�k�       �	$b�NXc�A�!*

losso��<hA��       �	���NXc�A�!*

lossf
i;oA��       �	̛�NXc�A�!*

lossE�'<�G�^       �	&9�NXc�A�!*

lossA<���a       �	���NXc�A�!*

lossú�<�c       �	��NXc�A�!*

loss��a<��y�       �	-&�NXc�A�!*

loss??;��1       �	���NXc�A�!*

lossŻG<�K       �	Re�NXc�A�!*

lossZ��;����       �	P��NXc�A�!*

lossC�:���J       �	���NXc�A�!*

loss.�<��       �	�`�NXc�A�!*

loss���<�9y       �	�NXc�A�!*

loss�Ӱ<��\k       �	U��NXc�A�!*

lossfۓ:���_       �	�F�NXc�A�!*

loss�t�7\
-�       �	���NXc�A�!*

lossDA�;�&       �	{��NXc�A�!*

loss��G;u��       �	�NXc�A�!*

loss���<t;gS       �	Ե�NXc�A�!*

loss��==�O�       �	�L�NXc�A�!*

loss-�D;5�ɼ       �	1��NXc�A�!*

loss;p�9��-Z       �	~��NXc�A�!*

losse�<�ˍ       �	��NXc�A�!*

loss1�9R�h#       �	�8�NXc�A�!*

loss�1�<���Q       �	V��NXc�A�!*

losswb�</�I       �	�e�NXc�A�!*

lossh��=��S       �	#��NXc�A�!*

loss'�=5jQ@       �	q��NXc�A�!*

losss��<��       �	�(�NXc�A�!*

loss�5_=ԘW�       �	E��NXc�A�!*

lossn.:\�M       �	�P�NXc�A�!*

loss{�:;�7#\       �	J��NXc�A�!*

loss̌=ߪ�]       �	G��NXc�A�!*

loss�[�:�K��       �	�2�NXc�A�!*

lossvط<��͕       �	���NXc�A�!*

loss�p�<��h	       �	�`�NXc�A�!*

loss�;#_+       �	j��NXc�A�!*

loss`X~;���       �	���NXc�A�!*

loss��<�zzw       �	�"�NXc�A�!*

loss:ւ<�]��       �	���NXc�A�!*

loss�I<��n       �	�B�NXc�A�!*

loss |�<"�q       �	<��NXc�A�!*

lossU�;���y       �	k�NXc�A�!*

loss��8<����       �	 �NXc�A�!*

loss}�:M�)       �	���NXc�A�!*

loss}=�;]�       �	FC�NXc�A�!*

loss:�<
i�       �	���NXc�A�!*

loss��R=����       �	ݘ�NXc�A�!*

lossr�=_]G       �	v5�NXc�A�!*

loss@lV;8V/�       �	F��NXc�A�!*

loss��<���       �	�o�NXc�A�!*

loss_6�;�k��       �	��NXc�A�!*

loss���<��g-       �	֫�NXc�A�!*

lossT�"<����       �	�L�NXc�A�!*

lossm<,b�       �	)��NXc�A�!*

loss3�O<&�       �	���NXc�A�!*

loss�q@<[��e       �	u>�NXc�A�!*

loss��.;H�,'       �	�J�NXc�A�!*

lossF�;-��       �	���NXc�A�!*

lossßU<i-�       �	ǁ�NXc�A�!*

lossj�W;4��K       �	T�NXc�A�!*

lossv6 <���o       �	���NXc�A�!*

lossR��<̱U%       �	N�NXc�A�!*

lossR=1�       �	���NXc�A�!*

lossߙh<�5       �	��NXc�A�!*

lossE��;�O�L       �	0*�NXc�A�!*

loss;:��lo       �	¿�NXc�A�!*

losss�&;k��D       �	`Z�NXc�A�!*

loss@<�h       �	���NXc�A�!*

loss�(�; [Ɏ       �	a� OXc�A�!*

lossW�:�xmF       �	�<OXc�A�!*

lossՒ=I��       �	��OXc�A�!*

loss�T<�ҍT       �	�xOXc�A�!*

loss�P<5�Cf       �	�OXc�A�!*

lossS�<o
��       �	��OXc�A�!*

loss/�<�>E[       �	DOXc�A�!*

loss�W�<��       �	�OXc�A�!*

loss�S<�>m       �	��OXc�A�!*

loss�c=8���       �	*OXc�A�!*

loss ��9��Z       �	IOXc�A�!*

loss�t�:�.	       �	ΧOXc�A�!*

loss��H:Z�       �	:@OXc�A�!*

lossTb4;���j       �	s�OXc�A�!*

loss��;<       �	ap	OXc�A�!*

lossd^ =��e       �	�
OXc�A�!*

loss��\;��X�       �	��
OXc�A�!*

loss�n`<d0�v       �	�uOXc�A�!*

loss�(P=�        �	�OXc�A�!*

loss���;R�cw       �	��OXc�A�!*

lossl��:>ט       �	BOXc�A�!*

loss�*�9��z�       �	z�OXc�A�!*

lossx,�<�       �	@�OXc�A�!*

loss/$=���       �	�"OXc�A�!*

loss���<��x       �	@�OXc�A�!*

loss�Ķ;�}��       �	�hOXc�A�!*

loss���:c��       �	�OXc�A�!*

loss��Y<R���       �	��OXc�A�!*

lossK�<��       �	q<OXc�A�!*

loss,I�<wS�r       �	M�OXc�A�!*

loss\7�:�(�       �	��OXc�A�!*

loss?�=��+       �	RGOXc�A�!*

lossVP;��I       �	��OXc�A�!*

loss�`=ʽ�       �	��OXc�A�!*

loss��<{�]       �	=}OXc�A�!*

lossW�y;�2�       �	�:OXc�A�!*

loss.$�;�+��       �	��OXc�A�!*

loss�EB:#M��       �	�OXc�A�!*

loss�><n��       �	&mOXc�A�!*

loss�ǡ<R�6!       �	�lOXc�A�!*

loss�׳<��I       �	KXOXc�A�!*

loss%��<ul:C       �	R�OXc�A�!*

lossԇ�;v���       �	�OXc�A�!*

loss!�=Ӻ�o       �	��OXc�A�!*

lossm��<�&�       �	�OXc�A�!*

lossi��;��,&       �	��OXc�A�!*

lossl��;7]�       �	ӢOXc�A�!*

loss�\=��s       �	IH OXc�A�!*

loss�i�<�9�Y       �	�� OXc�A�"*

loss��:"��       �	c%"OXc�A�"*

lossz^8=@�-�       �	@�"OXc�A�"*

loss<.�=����       �	�}#OXc�A�"*

lossCa;9����       �	�R$OXc�A�"*

loss��;�{l       �	 c%OXc�A�"*

lossf�;E�%�       �	7�%OXc�A�"*

loss��f;O"Q       �	��&OXc�A�"*

lossI�<��U�       �	D4'OXc�A�"*

loss�-<���       �	�w(OXc�A�"*

loss�!�;A7G�       �	�$)OXc�A�"*

loss�l�<XdI�       �	f�)OXc�A�"*

loss~Z;��@       �	:]*OXc�A�"*

loss�-O<+ZX       �	�+OXc�A�"*

loss�eE:vM�       �	�3,OXc�A�"*

loss���;���       �	?�-OXc�A�"*

lossƽ<��k       �	�$.OXc�A�"*

lossa�:����       �	+�.OXc�A�"*

lossF��:M�1�       �	`W/OXc�A�"*

loss��0<G~*       �	�/OXc�A�"*

loss��%=�a=l       �	��0OXc�A�"*

loss�_�<�T�       �	c'1OXc�A�"*

loss��@;X��       �	��1OXc�A�"*

loss�*�;�d*g       �	]2OXc�A�"*

losss��;!��       �	��2OXc�A�"*

loss�̲<e=��       �	�3OXc�A�"*

loss��=�f�       �	�\4OXc�A�"*

loss�|�<ķ/�       �	��4OXc�A�"*

loss�';�".x       �	�5OXc�A�"*

loss(84:�.^}       �	�E6OXc�A�"*

loss��v=%��.       �	J�6OXc�A�"*

lossɃZ=A�       �	��7OXc�A�"*

loss��;d�       �	~98OXc�A�"*

loss�m6=��        �	��8OXc�A�"*

loss�;�9�˧:       �	jk9OXc�A�"*

loss�T�=�*��       �	:OXc�A�"*

loss�:X�4�       �	�:OXc�A�"*

loss��:�րB       �	�B;OXc�A�"*

loss{�K<�Az:       �	��;OXc�A�"*

loss��1=����       �	�s<OXc�A�"*

loss}݆<�h�$       �	�=OXc�A�"*

lossn�;���       �	�=OXc�A�"*

loss�6k;���       �	�H>OXc�A�"*

loss�ES<���       �	��>OXc�A�"*

loss[��9;`F�       �	Ks?OXc�A�"*

lossEP�=�Z{�       �	@OXc�A�"*

loss#��;
�mD       �	&�@OXc�A�"*

loss6�0=�EA�       �	C:AOXc�A�"*

loss���;�g��       �	A�AOXc�A�"*

loss�-W=]�r       �	�jBOXc�A�"*

lossq�;��       �	COXc�A�"*

lossOW$<z�ȥ       �	��COXc�A�"*

losst; {c�       �	ZDOXc�A�"*

loss��;�;`�       �	��DOXc�A�"*

loss���;�ec�       �	`#FOXc�A�"*

loss�2=a#޴       �	��FOXc�A�"*

loss���<��RK       �	8�GOXc�A�"*

loss�|�<���x       �		HOXc�A�"*

loss��)=��Vy       �	�*IOXc�A�"*

loss\e�;Gs       �	j�IOXc�A�"*

loss�X<{E>@       �	�NJOXc�A�"*

lossT� :*��{       �	��JOXc�A�"*

lossr`�;}�       �	��KOXc�A�"*

lossn��=(<e�       �	RCLOXc�A�"*

loss�G<����       �	#�LOXc�A�"*

loss�4>A�7       �	�pMOXc�A�"*

loss�٥<?�       �	yNOXc�A�"*

loss��<u��V       �	�NOXc�A�"*

loss}��<��B       �	@OOXc�A�"*

lossDkG;y       �	�OOXc�A�"*

loss��;�iy�       �	��POXc�A�"*

losszPt<h�ǵ       �	�mQOXc�A�"*

loss%��:�	s        �	ROXc�A�"*

loss���<����       �	��ROXc�A�"*

loss8}�<Y� �       �	C9SOXc�A�"*

loss_�T=���       �	��SOXc�A�"*

loss$W<��\       �	�eTOXc�A�"*

lossx.=#*5       �	��TOXc�A�"*

lossr�*<�L�2       �	��UOXc�A�"*

lossʈ�:� �       �	Z�VOXc�A�"*

losss�4;�{/�       �	�ZWOXc�A�"*

lossi|];ꕍ�       �	��WOXc�A�"*

loss�K=��P%       �	�XOXc�A�"*

lossR�<�9�+       �	{kYOXc�A�"*

lossv/	;E80       �	yZOXc�A�"*

loss}Ȯ<�A�       �	��ZOXc�A�"*

loss���:�X��       �	�B[OXc�A�"*

loss]�<w�%y       �	��[OXc�A�"*

loss��;8MQ       �	؁\OXc�A�"*

loss�<��c)       �	h!]OXc�A�"*

loss�/n<����       �	��]OXc�A�"*

loss��B;'�C       �	mU^OXc�A�"*

lossǮ=���       �	�^OXc�A�"*

lossq)�:�Hz       �	>y_OXc�A�"*

lossx��< �c       �	�`OXc�A�"*

lossͬI:�3�s       �	4�`OXc�A�"*

loss���<��       �	�0aOXc�A�"*

lossRc{=�`K       �	6�aOXc�A�"*

lossC��;�/�       �	!YbOXc�A�"*

loss�H<�� s       �	��bOXc�A�"*

loss�(a=�Md       �	��cOXc�A�"*

loss�?4<��I       �	�dOXc�A�"*

loss8(>:�T       �	��dOXc�A�"*

loss]M=�ܜR       �	�EeOXc�A�"*

loss v	;Z
�       �	�eOXc�A�"*

loss$SF<�       �	�lfOXc�A�"*

loss���;�4�B       �	��fOXc�A�"*

losslD�<p`�       �	>�gOXc�A�"*

lossQ�s<��4       �	^*hOXc�A�"*

lossZ�<.�3       �	@�hOXc�A�"*

loss�[$;Z��       �	�ViOXc�A�"*

lossX8<x��       �	��iOXc�A�"*

loss�Tl;��5       �	]�jOXc�A�"*

loss��;�m��       �	�kOXc�A�"*

losse�@<��o       �	��kOXc�A�"*

loss���=&�Rb       �	PlOXc�A�"*

loss|��<��?�       �	��lOXc�A�"*

loss9;��/O       �	�mOXc�A�"*

lossA�<�J�2       �	�7nOXc�A�"*

loss�?o<diC1       �	��nOXc�A�"*

losscE�;�@��       �	�boOXc�A�"*

loss${4=�#       �	��oOXc�A�"*

loss�=��5�       �	3�pOXc�A�"*

loss��<�/�       �	�"qOXc�A�"*

lossd�<r-��       �	̴qOXc�A�"*

lossک;փ�       �	�OrOXc�A�"*

lossԎQ:�ɾ=       �	��rOXc�A�"*

loss1O@=�RO�       �	!�sOXc�A�"*

loss!"�<�m]       �	5*tOXc�A�#*

loss;�<��.       �	��tOXc�A�#*

loss�(s;�5F�       �	�TuOXc�A�#*

lossA�;� �       �	��uOXc�A�#*

loss�L#=承       �	OyvOXc�A�#*

lossvh�;�[z       �	�wOXc�A�#*

loss�"=�pU�       �	�wOXc�A�#*

loss�x<���`       �	"7xOXc�A�#*

loss�`:��]       �	��xOXc�A�#*

loss�C=�z�~       �	�hyOXc�A�#*

loss�w�<���       �	7 zOXc�A�#*

lossT�:��l       �	��zOXc�A�#*

loss���<�]�,       �	�G{OXc�A�#*

loss���:�9,)       �	��{OXc�A�#*

lossτ�:e���       �	�r|OXc�A�#*

loss�p:ƺp�       �	}OXc�A�#*

loss?��<6$y�       �	l�}OXc�A�#*

loss�;^<��q�       �	�)~OXc�A�#*

lossZ�Y;r��c       �	��~OXc�A�#*

loss��:5�B�       �	�SOXc�A�#*

loss�n�;��       �	��OXc�A�#*

loss��<��_       �	肀OXc�A�#*

loss�C<<�CZ       �	L�OXc�A�#*

loss.b�9��{�       �	6��OXc�A�#*

losst,�;���       �	�>�OXc�A�#*

losso�=T���       �	@��OXc�A�#*

loss�\�=;zu�       �	FD�OXc�A�#*

loss�B�9�F        �	��OXc�A�#*

lossA�X<.l_{       �	+��OXc�A�#*

losso�M=�]       �	��OXc�A�#*

loss{�&;O��       �	5��OXc�A�#*

loss��;��j       �	�O�OXc�A�#*

loss�!�<�8x:       �	�OXc�A�#*

loss��'=����       �	��OXc�A�#*

lossM[�;�Mw�       �	/�OXc�A�#*

lossV&y;JD�       �	q��OXc�A�#*

loss�>�<���
       �	jM�OXc�A�#*

loss�<�p��       �	��OXc�A�#*

lossh��;��U       �	Ș�OXc�A�#*

loss���<���       �	R+�OXc�A�#*

loss��=r$�       �	+��OXc�A�#*

lossh��<� �       �	�^�OXc�A�#*

loss��1=i[ u       �	���OXc�A�#*

lossc1�<�`7d       �	��OXc�A�#*

loss(�4;����       �	�:�OXc�A�#*

loss��6;XG�       �	EُOXc�A�#*

loss.�2<Wl8       �	�w�OXc�A�#*

lossx��:���:       �	��OXc�A�#*

loss�Y1=B��       �	e��OXc�A�#*

loss���;ҷ��       �	C�OXc�A�#*

loss��<��       �	Z֒OXc�A�#*

loss�
�<!�q       �	�s�OXc�A�#*

loss��L;-2       �	0�OXc�A�#*

loss���<|/͕       �	ڪ�OXc�A�#*

loss��;��ZT       �	�G�OXc�A�#*

loss��x=��d�       �	j��OXc�A�#*

loss��<6O��       �	�1�OXc�A�#*

loss��=<O�]%       �	(՗OXc�A�#*

loss�ׅ<6⽗       �	���OXc�A�#*

loss�T%=�g�       �	"�OXc�A�#*

losscb;8�A       �	eřOXc�A�#*

loss�< 0��       �	�c�OXc�A�#*

lossNݺ:�?|�       �	��OXc�A�#*

loss\<�H�       �	���OXc�A�#*

loss�[=�ga�       �	.�OXc�A�#*

loss��M= ��       �	Y��OXc�A�#*

loss��;����       �	9^�OXc�A�#*

loss�;;R?�7       �	��OXc�A�#*

loss�aq=�9�}       �	t��OXc�A�#*

loss�ū<��}~       �	�.�OXc�A�#*

loss��;����       �	�ɟOXc�A�#*

loss�uj<� �       �	�[�OXc�A�#*

loss! �;D��l       �	Lm�OXc�A�#*

lossqxf=�^o'       �	{�OXc�A�#*

loss-� <��l       �	d��OXc�A�#*

loss�}�<+O�       �	�F�OXc�A�#*

loss{�;K� �       �	��OXc�A�#*

loss�r�; �       �	���OXc�A�#*

loss�r<J*�       �	��OXc�A�#*

loss�.*<�E�       �	���OXc�A�#*

loss���<��.       �	EG�OXc�A�#*

loss��p<��M       �	�ߦOXc�A�#*

lossȤ�<�V�       �	�z�OXc�A�#*

loss�!>Ջ�,       �	�OXc�A�#*

loss��&=k�       �	B��OXc�A�#*

loss&*=���       �	�?�OXc�A�#*

loss;��:�hG       �	թOXc�A�#*

lossƉ�;����       �	2s�OXc�A�#*

loss):3: ���       �	R�OXc�A�#*

loss�V<��6P       �	G��OXc�A�#*

loss���;�*/�       �	HQ�OXc�A�#*

lossx�<!���       �	F�OXc�A�#*

loss�İ<t�       �	'��OXc�A�#*

loss}Ղ<A�	*       �	��OXc�A�#*

lossS�#;��R       �	F��OXc�A�#*

loss�<T|F       �	{L�OXc�A�#*

loss�
;��sp       �	�ޯOXc�A�#*

lossh^ =L~�       �	7q�OXc�A�#*

loss�̌;�a�2       �	���OXc�A�#*

loss2Q�<�,(       �	"m�OXc�A�#*

loss?��;8���       �	��OXc�A�#*

loss�4;mv��       �	�l�OXc�A�#*

loss�-�<I�N�       �	G�OXc�A�#*

loss��>"�i~       �	��OXc�A�#*

loss��;��Ϋ       �	>�OXc�A�#*

loss/��85s�"       �	Q۶OXc�A�#*

lossW��;�Y�       �	�u�OXc�A�#*

loss��'=�s��       �	[�OXc�A�#*

lossn4�=941�       �	Z��OXc�A�#*

loss2d@;���       �	�<�OXc�A�#*

loss_�e:x�}�       �	�ӹOXc�A�#*

lossϴ�: (NZ       �	�g�OXc�A�#*

loss|�)<_�r       �	���OXc�A�#*

loss.��<��8�       �	B��OXc�A�#*

loss..�=@P>       �	�1�OXc�A�#*

loss��=�߱\       �	�μOXc�A�#*

loss���<k�ռ       �	�l�OXc�A�#*

lossH��9f'�        �	��OXc�A�#*

loss/�;�"�       �	�˾OXc�A�#*

loss���;�f1       �	�g�OXc�A�#*

loss`�;�y&i       �	p�OXc�A�#*

lossȚ�;��       �	��OXc�A�#*

loss(<XR7g       �	)]�OXc�A�#*

loss ��<0K�       �	���OXc�A�#*

loss��<KU Y       �	l��OXc�A�#*

loss�e�=D�'       �	a7�OXc�A�#*

loss�ɪ=���a       �	���OXc�A�#*

losso�m=�H�       �	a��OXc�A�#*

lossH/>���^       �	h%�OXc�A�$*

lossx�W=� �       �	Ǹ�OXc�A�$*

lossw_q=U=�       �	�M�OXc�A�$*

loss�a
==�T�       �	���OXc�A�$*

lossVu=�m�i       �	��OXc�A�$*

loss#�s:4r��       �	~:�OXc�A�$*

loss��=o��U       �	���OXc�A�$*

lossnc/=qw�b       �	s�OXc�A�$*

loss^�=�]�       �	l%�OXc�A�$*

loss]��:����       �	���OXc�A�$*

loss��D;ӥH"       �	b��OXc�A�$*

loss�=��ax       �	M0�OXc�A�$*

lossp=<Y��S       �	b��OXc�A�$*

lossJ�;�1z       �	8��OXc�A�$*

loss(�;v�c�       �	�+�OXc�A�$*

loss���;m^R       �	���OXc�A�$*

loss� =�(�V       �	��OXc�A�$*

loss��J=�œ       �	R+�OXc�A�$*

loss=ƴ<�&-�       �	$��OXc�A�$*

lossm �<:^=       �	z�OXc�A�$*

losshN="f       �	��OXc�A�$*

losst��;,q��       �	(��OXc�A�$*

lossD��<��       �	�W�OXc�A�$*

lossv��;=rV�       �	j��OXc�A�$*

loss7��<7�       �	��OXc�A�$*

loss]�;�̓�       �	#2�OXc�A�$*

loss��E=#}�       �	��OXc�A�$*

loss��=��2       �	}�OXc�A�$*

lossj�<��T       �	�I�OXc�A�$*

lossZ��<�	t�       �	���OXc�A�$*

loss$̖</���       �	��OXc�A�$*

lossV=�;�<<f       �	+2�OXc�A�$*

loss$_�<b�/       �	Y��OXc�A�$*

loss�=�<E�}�       �	՗�OXc�A�$*

loss*��<}�33       �	�7�OXc�A�$*

loss�Dt<����       �	���OXc�A�$*

loss�c;31       �	�s�OXc�A�$*

loss�;tp�#       �	B"�OXc�A�$*

loss�Ǒ;��V       �	۾�OXc�A�$*

lossF��<��d�       �	p\�OXc�A�$*

lossq��:V�{Z       �	��OXc�A�$*

loss��<��N       �	��OXc�A�$*

loss_��<���P       �	v4�OXc�A�$*

loss�J.<����       �	4��OXc�A�$*

loss�>�;��%       �	�u�OXc�A�$*

lossX;ne:�       �	�OXc�A�$*

loss�<v��       �	M��OXc�A�$*

loss{2=��qQ       �	�_�OXc�A�$*

lossAC�<u�o�       �	u�OXc�A�$*

loss�ӫ<�U[$       �	/��OXc�A�$*

loss@��<�Ɖ7       �	K�OXc�A�$*

loss��?;r6��       �	1��OXc�A�$*

loss��=�iv]       �	 ��OXc�A�$*

loss��;��Y�       �	�'�OXc�A�$*

loss�,=��m       �	���OXc�A�$*

loss�g�:?�       �	^�OXc�A�$*

lossO?�<��/�       �	��OXc�A�$*

loss.I�;�JR�       �	O��OXc�A�$*

lossH�-;�{�       �	�b�OXc�A�$*

loss��:A-��       �	�OXc�A�$*

loss���:���4       �	k��OXc�A�$*

loss��<�N�       �	S�OXc�A�$*

loss[�{;pQ��       �	���OXc�A�$*

loss�̥=�B��       �	���OXc�A�$*

loss��<�^hD       �	�/�OXc�A�$*

lossC*�<�#4       �	m��OXc�A�$*

loss =Y�k       �	�_�OXc�A�$*

loss��B<�5a       �	B
�OXc�A�$*

lossY�=��	�       �	z��OXc�A�$*

lossx$;�D�       �	�g�OXc�A�$*

loss�x�:H�k�       �	9�OXc�A�$*

lossn�^;���       �	��OXc�A�$*

loss��m=�C�       �	���OXc�A�$*

lossAa�=��Wa       �	�6�OXc�A�$*

loss��!=�V       �	���OXc�A�$*

loss�i�<��o�       �	d�OXc�A�$*

loss��Y<VS�       �	���OXc�A�$*

lossivY;:�z�       �	���OXc�A�$*

loss}��;���       �	�"�OXc�A�$*

loss)�:"�F       �	k��OXc�A�$*

lossuq=�I��       �	d��OXc�A�$*

loss�E;N��>       �	���OXc�A�$*

loss��;�F��       �	.X�OXc�A�$*

loss��=A�	V       �	8�OXc�A�$*

loss\��;��i.       �	���OXc�A�$*

loss��;��       �	}�OXc�A�$*

loss��=<Ctr�       �	��OXc�A�$*

lossL�F<ʲ/�       �	���OXc�A�$*

loss#:��V       �	�N�OXc�A�$*

loss	E<���       �	���OXc�A�$*

loss*�-=Ľ��       �	� PXc�A�$*

loss���<���       �	PXc�A�$*

lossfO;J�$�       �	ŭPXc�A�$*

loss�H�<&��m       �	�?PXc�A�$*

loss��8:O�Ϣ       �	I�PXc�A�$*

losscU�:̗r       �	�sPXc�A�$*

loss�b�:���       �	�PXc�A�$*

loss���;K�y       �	��PXc�A�$*

loss��<����       �	�CPXc�A�$*

lossד�<�       �	��PXc�A�$*

lossS�<��4�       �	yPXc�A�$*

loss�p�;IyW�       �	.PXc�A�$*

loss��=bN�       �	дPXc�A�$*

loss��9��       �	2UPXc�A�$*

losso��;e���       �	2�PXc�A�$*

lossHR�<W�CR       �	|	PXc�A�$*

loss$�<Y�        �	2
PXc�A�$*

lossh�;W��       �	f�
PXc�A�$*

lossZ��<\\       �	%ZPXc�A�$*

loss��=�h�       �	|�PXc�A�$*

lossS<;�.&�       �	�PXc�A�$*

loss��[:x�       �	(PXc�A�$*

loss*T\<h�w�       �	��PXc�A�$*

lossTB%:\��z       �	�_PXc�A�$*

loss!8@=}�H       �	%PXc�A�$*

lossq��9��ʄ       �	s�PXc�A�$*

loss{s<E��       �	&5PXc�A�$*

lossia�;Ry       �	��PXc�A�$*

lossON�<��]       �	�sPXc�A�$*

loss��+;�q8�       �	QPXc�A�$*

loss��=�x��       �	��PXc�A�$*

loss3�^=�`��       �	RGPXc�A�$*

lossO<9e�E       �	��PXc�A�$*

loss���;�.$       �	2sPXc�A�$*

lossMY<��m(       �	�PXc�A�$*

loss��:���,       �	_�PXc�A�$*

loss&Õ:X�@�       �	^,PXc�A�$*

loss��:�o�K       �	5�PXc�A�$*

loss�;W��k       �	��PXc�A�%*

loss:��<H>       �	$(PXc�A�%*

loss���;B��       �	�PXc�A�%*

loss���9 8|�       �	�`PXc�A�%*

lossȱ�;ѣi�       �	� PXc�A�%*

loss���8�乹       �	��PXc�A�%*

loss��:���       �	uxPXc�A�%*

loss���8ͩ�       �	�;PXc�A�%*

loss`��;O3�       �	e�PXc�A�%*

loss�|G<�$?       �	R�PXc�A�%*

losswǊ;�m��       �	?;PXc�A�%*

lossN˿;�Y��       �		3PXc�A�%*

lossv�D=x�=       �	�4 PXc�A�%*

loss�k*=��6q       �	�� PXc�A�%*

loss3c;�a�E       �	!�!PXc�A�%*

losso�7=� 	       �	��"PXc�A�%*

lossZ�=6ι�       �	4�#PXc�A�%*

loss��;��#       �	��$PXc�A�%*

loss2 �:9�`5       �	)&PXc�A�%*

loss�P�:�ԍ�       �	�<'PXc�A�%*

loss,�;��2n       �	��'PXc�A�%*

loss�V<��2O       �	��(PXc�A�%*

loss��:��M       �	�)PXc�A�%*

loss��<ԋw�       �	t*PXc�A�%*

loss:�;�Z�h       �	��+PXc�A�%*

loss�?*>(�       �	zp,PXc�A�%*

lossZ�
<���r       �	�-PXc�A�%*

loss{(<�+�#       �	�d.PXc�A�%*

loss�/�<VN�       �	ۤ/PXc�A�%*

loss�P�;ͩ�       �	��0PXc�A�%*

loss�t<R��       �	��1PXc�A�%*

loss�?�;("
z       �	��2PXc�A�%*

loss(��:�3jx       �	�03PXc�A�%*

lossN3�:*>K       �	��3PXc�A�%*

lossn:4:$�?q       �	�s4PXc�A�%*

loss�@6<��b       �	5PXc�A�%*

loss߾�;Y��g       �	��5PXc�A�%*

loss��<|z1�       �	�T6PXc�A�%*

loss%�B9��#       �	57PXc�A�%*

lossZm�<�e'�       �	��7PXc�A�%*

loss�c"=o���       �	�X8PXc�A�%*

loss|�<{�C�       �	u9PXc�A�%*

loss�?X<;'"g       �	Q�9PXc�A�%*

loss��?<:��       �	�X:PXc�A�%*

loss�n�;�^��       �	P�:PXc�A�%*

loss!��:�	       �	��;PXc�A�%*

loss=Z<!��       �	d=<PXc�A�%*

loss�@S=u���       �	��<PXc�A�%*

losse��; ��       �	Ox=PXc�A�%*

loss^��<�@ND       �	3>PXc�A�%*

loss��8<i-�[       �	g�>PXc�A�%*

loss��<_��X       �	}\?PXc�A�%*

loss͒f=R��       �	� @PXc�A�%*

loss(��=����       �	��@PXc�A�%*

loss�<�![^       �	P9APXc�A�%*

loss��<��/       �	(�APXc�A�%*

lossX��<���       �	�fBPXc�A�%*

loss_��;4�/"       �	GCPXc�A�%*

loss� �;��8�       �	�CPXc�A�%*

lossT�V<�)�Q       �	�9DPXc�A�%*

loss���<�lh       �	��DPXc�A�%*

loss.6;@50       �	�rEPXc�A�%*

loss�1;	��?       �	�FPXc�A�%*

loss��;d�e�       �	t�FPXc�A�%*

lossOi;�!�       �	�[GPXc�A�%*

loss~{�=��v       �	�/^PXc�A�%*

lossʬ�;k�5       �	��^PXc�A�%*

lossϒ|=�x��       �	&s_PXc�A�%*

lossɷj;�       �	m6`PXc�A�%*

lossHe8: �\�       �	��`PXc�A�%*

loss�m�;u��8       �	YkaPXc�A�%*

loss/5�;���A       �	�bPXc�A�%*

loss�=mŌ;       �	ˡbPXc�A�%*

lossl�_<���       �	�OcPXc�A�%*

loss��<�.{       �	}�dPXc�A�%*

loss10�;�a       �	�gePXc�A�%*

loss���<�n�       �	�fPXc�A�%*

loss�5�<ghA       �	��fPXc�A�%*

loss�*3<8#y*       �	3gPXc�A�%*

loss��;��l       �	2�gPXc�A�%*

lossIL�<�}��       �	�`hPXc�A�%*

lossO�;V�       �	 *iPXc�A�%*

loss�7c<+��       �	��iPXc�A�%*

loss�k�;0j�       �	�[jPXc�A�%*

lossLRv;DO}o       �	��jPXc�A�%*

loss{�<�II       �	e�kPXc�A�%*

lossC��<�rb�       �	6"lPXc�A�%*

loss�Ĝ:�+�p       �	��lPXc�A�%*

loss�m�=���       �	�WmPXc�A�%*

loss]�;���       �	8�mPXc�A�%*

lossB�<�Q�       �	��nPXc�A�%*

loss�
�;��'�       �	�SpPXc�A�%*

loss1�(:���L       �	B�pPXc�A�%*

lossz�<ǭ5_       �	ؚqPXc�A�%*

loss�	=F��       �	�;rPXc�A�%*

loss���;�V��       �	��rPXc�A�%*

loss��=��U�       �	��sPXc�A�%*

loss|+m;�q�        �	�tPXc�A�%*

loss��<8��       �	��tPXc�A�%*

lossQ�;�F�/       �	/iuPXc�A�%*

lossc6=AW��       �	��uPXc�A�%*

loss�Ӄ;=�I�       �	ӽvPXc�A�%*

loss_�+<���       �	>{wPXc�A�%*

loss�a&<4�*       �	��xPXc�A�%*

loss��X;Z�       �	LoyPXc�A�%*

loss��;�;��       �	�3zPXc�A�%*

loss|�<p�e�       �	8{PXc�A�%*

loss�9<��       �	��|PXc�A�%*

loss?�i<k9q�       �	�g}PXc�A�%*

lossh?�=7sp�       �	�LPXc�A�%*

loss���;p	�!       �	y[�PXc�A�%*

loss�/&=<�p�       �	��PXc�A�%*

loss��;n�(�       �	���PXc�A�%*

loss�0=A�A�       �	O�PXc�A�%*

loss̎�8��       �	��PXc�A�%*

loss��:
�p       �	 ��PXc�A�%*

loss�R=�7       �	�N�PXc�A�%*

losssմ:�iN       �	Q��PXc�A�%*

losswć=c�,�       �	�7�PXc�A�%*

loss�v<�Ŝ�       �	��PXc�A�%*

loss�);Bv�       �	D��PXc�A�%*

loss��5;s{7�       �	x^�PXc�A�%*

lossFTK:u>/|       �	��PXc�A�%*

loss��=�ao�       �	���PXc�A�%*

loss�p9=~��c       �	�$�PXc�A�%*

lossJ=pK�-       �	���PXc�A�%*

loss�+�<�T+�       �	�O�PXc�A�%*

lossV�]<*�Q�       �	��PXc�A�%*

loss���;Ê�       �	���PXc�A�&*

loss���;)�       �	|*�PXc�A�&*

loss��.;�oP       �	S΍PXc�A�&*

lossE^�<Vȶ�       �	�m�PXc�A�&*

loss���<�Iz       �	�PXc�A�&*

lossc�<��@�       �	���PXc�A�&*

loss̸<��h�       �	>�PXc�A�&*

lossn�;���       �	�PXc�A�&*

lossR =����       �	A�PXc�A�&*

loss�;�]�       �	^+�PXc�A�&*

loss�<�E8�       �	ҒPXc�A�&*

lossxQ�<*��       �	nn�PXc�A�&*

loss�μ;��T       �	��PXc�A�&*

loss��j:g?H�       �	���PXc�A�&*

loss���;��5Z       �	ޓ�PXc�A�&*

loss�k�:���       �	�-�PXc�A�&*

loss���<ߓI       �	���PXc�A�&*

lossV|�<��f       �	�_�PXc�A�&*

lossz��<*A�       �	��PXc�A�&*

lossf�<Ma�0       �	�q�PXc�A�&*

loss��?9�F��       �	4�PXc�A�&*

loss�<�7b       �	L��PXc�A�&*

loss�(<�s��       �	}=�PXc�A�&*

loss��;�C)       �	�ڛPXc�A�&*

loss�#k<�V�       �	-y�PXc�A�&*

loss��:�^�       �	7�PXc�A�&*

loss�#=@ص       �	�֝PXc�A�&*

loss�'�;Ez�       �	fl�PXc�A�&*

lossz&�:X�|w       �	\�PXc�A�&*

loss8;�^�M       �	vݟPXc�A�&*

loss��=��       �	�{�PXc�A�&*

loss%"i=^��|       �	��PXc�A�&*

lossHq�;eyMg       �	���PXc�A�&*

lossqs�=(HD�       �	+O�PXc�A�&*

loss���;B���       �	��PXc�A�&*

loss\D=��U�       �	��PXc�A�&*

loss�!=��/.       �	��PXc�A�&*

losse@x;'�r        �	���PXc�A�&*

lossV��;1ۆ�       �	�L�PXc�A�&*

loss��6;r�6X       �	�G�PXc�A�&*

loss)��:O�T       �	ݦPXc�A�&*

loss�J;k�γ       �	�n�PXc�A�&*

lossw��:W&�       �	`�PXc�A�&*

loss�J�;�Q�       �	���PXc�A�&*

loss�`�:a�+�       �	�(�PXc�A�&*

losst~U<� ~       �	éPXc�A�&*

loss���<D{.�       �	�l�PXc�A�&*

loss�\=^)-       �	��PXc�A�&*

loss;< +�       �	A��PXc�A�&*

loss�k=R�j�       �	f/�PXc�A�&*

loss�24;Gc       �	&ǬPXc�A�&*

loss��/<�0�t       �	B\�PXc�A�&*

loss4CW:�pU�       �	��PXc�A�&*

lossNXB<_�
       �	���PXc�A�&*

loss��<�]�       �	j/�PXc�A�&*

loss
��;k�|X       �	!ίPXc�A�&*

lossR�X=��j       �	p�PXc�A�&*

loss��v;)��       �	��PXc�A�&*

loss��<~:jj       �	��PXc�A�&*

lossz/o<vH+0       �	/1�PXc�A�&*

loss�^<:�       �	w�PXc�A�&*

lossؿ<�dR�       �	.��PXc�A�&*

loss�e6<����       �	d=�PXc�A�&*

loss�&�:�5�       �	tӴPXc�A�&*

loss{	l< ���       �	�h�PXc�A�&*

loss(6T=��i       �	e��PXc�A�&*

loss�߀;ms�%       �	���PXc�A�&*

loss,��;��c       �	�.�PXc�A�&*

loss�F�<ȯ1�       �	���PXc�A�&*

lossF��<����       �	�S�PXc�A�&*

loss#I;��\�       �	䟹PXc�A�&*

lossT�;W���       �	L4�PXc�A�&*

loss���9ĀJ       �	�ͺPXc�A�&*

loss�W	<u�j�       �	���PXc�A�&*

loss�]:z�1       �	�a�PXc�A�&*

lossߥ�<���R       �	� �PXc�A�&*

loss�YQ<���7       �	���PXc�A�&*

loss�":뜯:       �	�*�PXc�A�&*

loss@�;=�l��       �	`ȿPXc�A�&*

loss=<�b�       �	B^�PXc�A�&*

lossW,�<��       �	���PXc�A�&*

lossTޝ<P?�4       �	ӟ�PXc�A�&*

loss]�:����       �	hA�PXc�A�&*

loss�>{9�;�       �	��PXc�A�&*

loss��C<'��       �	z��PXc�A�&*

lossIQ�;҇n�       �	�@�PXc�A�&*

loss�Ҟ:<��X       �	{��PXc�A�&*

loss�l<qrB�       �	 {�PXc�A�&*

lossԄ�<Kp�c       �	z�PXc�A�&*

lossԳ<h.P�       �	��PXc�A�&*

loss]�;)�U       �	O�PXc�A�&*

loss<c�K       �	`��PXc�A�&*

loss���9Q�       �	�~�PXc�A�&*

lossk=�.�~       �	��PXc�A�&*

lossmA~<+�Tn       �	;��PXc�A�&*

loss���<���y       �	AI�PXc�A�&*

loss��<Q
ܔ       �	���PXc�A�&*

loss�k|<M�}       �	�x�PXc�A�&*

lossԂB<4�l�       �	��PXc�A�&*

loss��9�\5�       �	���PXc�A�&*

loss��80�I       �	<�PXc�A�&*

loss�x�<��b�       �	��PXc�A�&*

lossc4�;Fp{�       �	u�PXc�A�&*

loss�a1:Я       �	W�PXc�A�&*

loss��%<�s�J       �	���PXc�A�&*

loss�F�:��x       �	<�PXc�A�&*

loss
G�<bt!�       �	��PXc�A�&*

lossE�.=���K       �	vm�PXc�A�&*

loss��0<��       �	W
�PXc�A�&*

loss���:
[��       �	ס�PXc�A�&*

loss�ۂ=��       �	?:�PXc�A�&*

loss��;C>�       �	c��PXc�A�&*

loss��<#��       �	�m�PXc�A�&*

lossS0�;����       �	'�PXc�A�&*

loss�Z3<���       �	��PXc�A�&*

loss�R<w�b       �	�A�PXc�A�&*

loss�C�:�h_�       �	���PXc�A�&*

loss�s:8��       �	vl�PXc�A�&*

loss׋ ;
�z       �	�4�PXc�A�&*

loss��=@<�~       �	���PXc�A�&*

loss�)9��'�       �	h�PXc�A�&*

loss8��:=4��       �	��PXc�A�&*

loss���:mp|       �	ܛ�PXc�A�&*

loss��;�e&�       �	�i�PXc�A�&*

loss��S<�8       �	�
�PXc�A�&*

lossg =�UsS       �	��PXc�A�&*

loss;22<̷�m       �	_C�PXc�A�&*

loss�<����       �	��PXc�A�&*

loss&��;��x�       �	L4�PXc�A�'*

loss�G;�JI       �	���PXc�A�'*

loss2p(;����       �	�w�PXc�A�'*

loss���;��H       �	{�PXc�A�'*

loss!�B=�ʌ$       �	���PXc�A�'*

loss�M�;E���       �	sH�PXc�A�'*

lossU��<q�F�       �	���PXc�A�'*

loss�%�;p�"�       �	ˀ�PXc�A�'*

loss.�99�g�       �	@�PXc�A�'*

loss-َ<%�       �	)��PXc�A�'*

lossօ�<�"��       �	�O�PXc�A�'*

loss.+�;�8       �	B��PXc�A�'*

lossi�'<1���       �		��PXc�A�'*

loss�><���B       �	["�PXc�A�'*

lossN�]==$z�       �	���PXc�A�'*

loss��@<��W
       �	|_�PXc�A�'*

loss��<��u       �	��PXc�A�'*

loss :�(t       �	��PXc�A�'*

losst[�:���%       �	l?�PXc�A�'*

loss��<�M�)       �	���PXc�A�'*

loss�<�+�       �	�z�PXc�A�'*

loss�<^<����       �	��PXc�A�'*

lossX�:t�߰       �	��PXc�A�'*

lossq[�=f�s@       �	S�PXc�A�'*

loss��>���       �	�x�PXc�A�'*

loss�Um=u;��       �	
�PXc�A�'*

lossMFX=�       �	2��PXc�A�'*

loss,�=ص
l       �	4I�PXc�A�'*

loss/o�=�*{       �	���PXc�A�'*

lossџ=e�y        �	z��PXc�A�'*

loss	;}�C%       �	w.�PXc�A�'*

lossm�<#�܃       �	���PXc�A�'*

loss��=�U��       �	�t�PXc�A�'*

losser�=���p       �	k�PXc�A�'*

loss"В<�Gc�       �	���PXc�A�'*

loss���<��S       �	%Y�PXc�A�'*

lossv��=���       �	z��PXc�A�'*

loss|s�<}��b       �	��PXc�A�'*

loss�C�<���       �	�7�PXc�A�'*

loss�oD<��>Z       �	&��PXc�A�'*

loss�;��
�       �	���PXc�A�'*

loss Nx<G��       �	~��PXc�A�'*

loss�1#=�n#       �	Zc�PXc�A�'*

loss� H=�zf�       �	~��PXc�A�'*

loss�3<,��       �	Ș�PXc�A�'*

loss6��<�D�       �	&6�PXc�A�'*

loss�H�<�       �	���PXc�A�'*

lossg�=L���       �	�v�PXc�A�'*

lossr<;L
I       �	'�PXc�A�'*

loss�E�:"ח       �	��PXc�A�'*

loss*�=U�"       �	Rb�PXc�A�'*

lossH��<���:       �	 QXc�A�'*

loss@�<=^m�       �	U� QXc�A�'*

loss�Z<�|��       �	}ZQXc�A�'*

loss=��<=�P�       �	m QXc�A�'*

lossT=<�\�J       �	k�QXc�A�'*

losso�<6�2F       �	xCQXc�A�'*

loss�H�<�Mv�       �	��QXc�A�'*

lossZY�;�VT       �	��QXc�A�'*

lossz]^=��5T       �	b-QXc�A�'*

loss_C<|1�       �	��QXc�A�'*

loss}��;i6�       �	�eQXc�A�'*

loss��K<	�F       �	�QXc�A�'*

lossM��:��lk       �	|�QXc�A�'*

loss��:
�7�       �	\�QXc�A�'*

lossM��;I���       �	�7	QXc�A�'*

losso��<`V0�       �	4�	QXc�A�'*

loss�}=�p\�       �	t�
QXc�A�'*

lossV*q<%j>�       �	KQXc�A�'*

loss��;V5c�       �	U�QXc�A�'*

lossNc�;�n�       �	�QXc�A�'*

loss���;BS�       �	<QXc�A�'*

loss��<��@       �	��QXc�A�'*

lossV��;?w�       �	�QXc�A�'*

lossh� <��|       �	�rQXc�A�'*

loss���<;"�8       �	�QXc�A�'*

lossZ�;�)|�       �	��QXc�A�'*

loss� �:��!�       �	�cQXc�A�'*

loss�X%<��s       �	�2QXc�A�'*

loss��:wrt       �	��QXc�A�'*

loss��<����       �	 zQXc�A�'*

lossz?=�[��       �	�'QXc�A�'*

loss�:�<���q       �	��QXc�A�'*

lossC�~=��^�       �	l�QXc�A�'*

loss(c�<g�l�       �	�4QXc�A�'*

loss�-8=�9�       �	�QXc�A�'*

lossav�;�y�       �	��QXc�A�'*

loss�57<+��       �	#�QXc�A�'*

lossL;���       �	K�QXc�A�'*

loss(�F;}RY'       �	BwQXc�A�'*

loss%˱<K��       �	yZQXc�A�'*

loss��9���       �	�OQXc�A�'*

loss�7<�֋�       �	fQXc�A�'*

loss?�=���       �	�0QXc�A�'*

loss��d<��TC       �	#LQXc�A�'*

loss��O<� d�       �	�  QXc�A�'*

loss�Yr:7�q<       �	�� QXc�A�'*

loss���;F!       �	z�!QXc�A�'*

lossu�<!	�       �	@�"QXc�A�'*

loss]<=�@��       �	�(#QXc�A�'*

loss�|6<�.)g       �	��#QXc�A�'*

loss�b<�t	�       �	R�$QXc�A�'*

lossQY�9h�s�       �	�U%QXc�A�'*

loss��<����       �	��%QXc�A�'*

loss���<%\�       �	�&QXc�A�'*

lossm.b;Z��r       �	�)'QXc�A�'*

lossu�
=�Ӽ       �	{�'QXc�A�'*

loss��u<eY       �	�^(QXc�A�'*

loss৥<e8��       �	��(QXc�A�'*

loss�Ym;���       �	��)QXc�A�'*

loss��:��ڽ       �	�"*QXc�A�'*

loss�y|<���       �	�*QXc�A�'*

loss��<�h�H       �	�V+QXc�A�'*

lossV]�=(bO       �	�+QXc�A�'*

loss�I>��@       �	%�,QXc�A�'*

loss �y:��s�       �	�/-QXc�A�'*

losszb=�vX       �	}�-QXc�A�'*

lossw�a=:��1       �	ds.QXc�A�'*

lossj3�:�{b�       �	�/QXc�A�'*

loss�q�:�       �	ü/QXc�A�'*

loss~0�<��zL       �	�`0QXc�A�'*

loss	O<�(�g       �	��0QXc�A�'*

loss�;�:�[�       �	��1QXc�A�'*

loss::�</���       �	�+2QXc�A�'*

loss���;'j�       �	��2QXc�A�'*

loss$ى:ߛk�       �	nl3QXc�A�'*

loss3�H=�f�       �	\4QXc�A�'*

lossz�#=Ȩ�       �	��4QXc�A�'*

loss@l<��|D       �	��5QXc�A�(*

loss�F�<F�fj       �	�7QXc�A�(*

loss6��;F�`�       �	v�7QXc�A�(*

loss�.C<T�       �	>8QXc�A�(*

loss(O"9nG�
       �	
�8QXc�A�(*

loss:S!<L��       �	��9QXc�A�(*

loss`�;=\�b       �	ʋ:QXc�A�(*

loss�o=��v       �	�c;QXc�A�(*

loss@k4<89       �	X<QXc�A�(*

loss���<x9@       �	<j=QXc�A�(*

loss̶`<��nf       �	1
>QXc�A�(*

loss�~;��=�       �	�>QXc�A�(*

loss�=�<#PL�       �	"O?QXc�A�(*

loss�j0;Y"x	       �	:�?QXc�A�(*

loss�g<�5B�       �	��@QXc�A�(*

loss3ћ=#�q       �	P7AQXc�A�(*

loss�/:�p�       �	��AQXc�A�(*

lossȿ='�=�       �	'�BQXc�A�(*

loss�;�:zѺ�       �	>&CQXc�A�(*

loss��>=:�W�       �	<�CQXc�A�(*

lossw�d<�D`#       �	�xDQXc�A�(*

loss&=f���       �	EQXc�A�(*

loss���;��q       �	q�EQXc�A�(*

lossG˘;�\       �	�JFQXc�A�(*

loss��D<��	       �	��FQXc�A�(*

loss!�7=���<       �	4�GQXc�A�(*

loss�!<�I�       �	�HQXc�A�(*

loss�SY<�`�       �	��HQXc�A�(*

loss��j;Y���       �	�LIQXc�A�(*

loss���;��w�       �	��IQXc�A�(*

loss�=L��       �	�JQXc�A�(*

loss�n�<�w�       �	�(KQXc�A�(*

loss��<G��v       �	�KQXc�A�(*

loss�L�<yh'�       �	�\LQXc�A�(*

loss�`6;L�^       �	��LQXc�A�(*

loss��D< +O       �	��MQXc�A�(*

loss�
�;���       �	\NQXc�A�(*

loss*);�m��       �	��NQXc�A�(*

loss���;��%)       �	�OQXc�A�(*

losso;�x��       �	?PQXc�A�(*

loss�<�U       �	��PQXc�A�(*

loss5�;��<8       �	�wQQXc�A�(*

lossܤ�;�l_       �	$RQXc�A�(*

loss�eH<�ҧT       �	a�RQXc�A�(*

loss/(=���       �	pBSQXc�A�(*

loss=w�:4z�Y       �		�SQXc�A�(*

loss��<
�U       �	$�TQXc�A�(*

loss��='�[�       �	�UQXc�A�(*

loss��^<¨��       �	(�UQXc�A�(*

loss):8k��       �	�RVQXc�A�(*

loss�n[<D�
�       �	� WQXc�A�(*

lossi�;=.�q       �	�WQXc�A�(*

lossڙy<��7       �	n5XQXc�A�(*

loss��;K�       �	��XQXc�A�(*

loss$��<+���       �	�dYQXc�A�(*

loss�O�;�b#�       �	Q3ZQXc�A�(*

loss]P�;w�!       �	��ZQXc�A�(*

loss�@=����       �	�i[QXc�A�(*

loss��:j���       �	�\QXc�A�(*

loss8��=+���       �	��\QXc�A�(*

lossu�;�Wx8       �	3]QXc�A�(*

lossIg�:y��       �	0�]QXc�A�(*

lossF�D=<y�q       �	0^QXc�A�(*

loss4�x='i,       �	:Z_QXc�A�(*

loss�H�:��ܠ       �	G`QXc�A�(*

loss�s;r�B�       �	��`QXc�A�(*

loss�h:���       �	�\aQXc�A�(*

lossB9=�8��       �	r�aQXc�A�(*

lossI�i:7pA       �	0�bQXc�A�(*

loss~�	8۟�       �	�DcQXc�A�(*

lossW�O=��Le       �	��cQXc�A�(*

loss@FJ:i(�'       �	��dQXc�A�(*

loss;�1<ĝ�,       �	/eQXc�A�(*

loss��p:Tڶ5       �	��eQXc�A�(*

loss3�{;���       �	�yfQXc�A�(*

loss���:�`�w       �	�gQXc�A�(*

loss�P�<0J��       �	U.hQXc�A�(*

lossʁs8���L       �	~�hQXc�A�(*

loss#�<��ޢ       �	�liQXc�A�(*

lossՍ;��       �	djQXc�A�(*

lossVM<Y�\�       �	0�jQXc�A�(*

lossr��:�2�%       �	�>kQXc�A�(*

loss�yN<���<       �	��kQXc�A�(*

loss�:=�!z�       �	�mlQXc�A�(*

lossHo&:�N�       �	�mQXc�A�(*

lossc�s=d�`�       �	��mQXc�A�(*

loss�r<d�4       �	�nQXc�A�(*

lossnKA; ��g       �	�?oQXc�A�(*

loss�h;�D��       �	��oQXc�A�(*

loss��%<`���       �	�}pQXc�A�(*

lossT��<�	84       �	�qQXc�A�(*

loss[�.:]mBT       �	�qQXc�A�(*

losso�=Q��       �	�KrQXc�A�(*

loss;u;P��       �	tsQXc�A�(*

loss��K<��&       �	�sQXc�A�(*

loss
��<��
       �	�HtQXc�A�(*

lossN�<�
mG       �	�tQXc�A�(*

loss�T�9�� �       �	dyuQXc�A�(*

losss��;@/��       �	fvQXc�A�(*

loss�^<`�V        �	�HwQXc�A�(*

loss��; ڲ3       �	�wQXc�A�(*

loss{�g:Rr�       �	�|xQXc�A�(*

lossA�9=�H��       �	Q2yQXc�A�(*

loss�d�;JKK@       �	,�yQXc�A�(*

loss��h;TJ��       �	vzQXc�A�(*

lossa]J=ohӎ       �	D{QXc�A�(*

loss�:��#�       �	�{QXc�A�(*

loss�<��K       �	^|QXc�A�(*

loss:]�=�^�>       �	�}QXc�A�(*

loss6�-;Z�.V       �	��}QXc�A�(*

loss���:��]�       �	�:~QXc�A�(*

loss?W�=}cƟ       �	��~QXc�A�(*

loss��9��7       �	�jQXc�A�(*

loss���9�P��       �	 �QXc�A�(*

lossj��<��)       �	���QXc�A�(*

loss���<�Ӽ9       �	�0�QXc�A�(*

lossq@�;?�z,       �	�ȁQXc�A�(*

losslX�:���       �	h�QXc�A�(*

loss:!O;��+       �	��QXc�A�(*

loss��v<��S:       �	���QXc�A�(*

lossv�:�	2�       �	H4�QXc�A�(*

loss�;=����       �	5τQXc�A�(*

loss���<�/�M       �	Qj�QXc�A�(*

loss���=Y�^       �	S�QXc�A�(*

loss@zq<�|��       �	
��QXc�A�(*

lossi��8�û       �	:>�QXc�A�(*

loss)=4<턇       �	ۇQXc�A�(*

lossR�A;��:&       �	Jz�QXc�A�(*

loss�:�;Q�(	       �	5�QXc�A�)*

loss�5%:�70       �	��QXc�A�)*

loss��<��9       �	q9�QXc�A�)*

loss�A�;Y�'�       �	ԊQXc�A�)*

lossB6:�ZnM       �		k�QXc�A�)*

lossK�;�@��       �	&�QXc�A�)*

loss���;7��1       �	�QXc�A�)*

lossT�<���       �	A+�QXc�A�)*

loss�rI;�}\@       �	���QXc�A�)*

loss���:S�ؘ       �	�[�QXc�A�)*

loss�d.;AdT       �	A��QXc�A�)*

loss�m:��F       �	玏QXc�A�)*

loss�u#;��S�       �	�"�QXc�A�)*

loss��<a���       �	���QXc�A�)*

loss�LE;�i+       �	�G�QXc�A�)*

losse��;��y       �	;�QXc�A�)*

loss\+6>�x{       �	�t�QXc�A�)*

loss���;�#��       �	o�QXc�A�)*

loss�wU<Y�I       �	���QXc�A�)*

losszA:�:+       �	tD�QXc�A�)*

loss31L;84!�       �	UٔQXc�A�)*

loss'%�;N�[C       �	�r�QXc�A�)*

loss uo;-�       �	��QXc�A�)*

loss�P�<���^       �		ÖQXc�A�)*

loss�B�<�\;7       �	��QXc�A�)*

lossZ�;r��&       �	�2�QXc�A�)*

loss@�==๩>       �	_ԘQXc�A�)*

loss=��;���       �	fh�QXc�A�)*

lossw�m;��(       �	��QXc�A�)*

loss�9�9���g       �	��QXc�A�)*

loss��;o���       �	4f�QXc�A�)*

lossϠ�:�!)�       �	^�QXc�A�)*

loss�}�<���J       �	МQXc�A�)*

loss��<g���       �	]��QXc�A�)*

loss��:�
5�       �	�'�QXc�A�)*

loss�<�       �	���QXc�A�)*

lossē�9��       �	:Z�QXc�A�)*

lossN.%=��-�       �	��QXc�A�)*

loss�;I��S       �	���QXc�A�)*

loss	�;�b��       �	u��QXc�A�)*

loss�lf<8M�       �	�k�QXc�A�)*

lossf�(;����       �	J	�QXc�A�)*

loss�K�=~=U       �	#��QXc�A�)*

loss���;�>�       �	�;�QXc�A�)*

loss@L�:F�(       �	6�QXc�A�)*

loss�WQ;�<
�       �	���QXc�A�)*

loss��|:><I       �	0*�QXc�A�)*

loss�)�9�ґ       �	�ƧQXc�A�)*

loss�=�ʭ�       �	!��QXc�A�)*

loss�(<N���       �	@�QXc�A�)*

loss���9��!
       �	שQXc�A�)*

loss�%2=�|pr       �	1y�QXc�A�)*

loss:{�=�8�O       �	 t�QXc�A�)*

loss�5�<�
�       �	_	�QXc�A�)*

lossr�z:=��j       �	��QXc�A�)*

loss�L-;^��p       �	�@�QXc�A�)*

lossO�;�0�       �	sڮQXc�A�)*

loss��=���       �	c��QXc�A�)*

lossx:x�$       �	~�QXc�A�)*

loss��;�H�P       �	
��QXc�A�)*

loss��:�I�       �	�_�QXc�A�)*

loss��<�0B*       �	,��QXc�A�)*

loss=)Oq       �	;��QXc�A�)*

loss��=Ə<       �	Z*�QXc�A�)*

lossʌ�;~L1       �	U��QXc�A�)*

loss��=a��w       �	�Y�QXc�A�)*

loss쁚=�wv�       �	��QXc�A�)*

loss�a;v�ȼ       �	ɒ�QXc�A�)*

loss?�V=�k�       �		5�QXc�A�)*

loss��8<��wk       �	��QXc�A�)*

loss`��:*w�       �	߈�QXc�A�)*

loss��w;�l       �	���QXc�A�)*

loss��6<0��^       �	�E�QXc�A�)*

loss��&;f9(R       �	C�QXc�A�)*

loss��9<�W       �	k��QXc�A�)*

loss�e�:���B       �	�7�QXc�A�)*

lossA�#=��       �	��QXc�A�)*

loss@7�;&j       �	���QXc�A�)*

loss��<�r�t       �	�QXc�A�)*

loss���:���       �	���QXc�A�)*

lossL��:,<�       �	Ym�QXc�A�)*

losst�<�d2       �	��QXc�A�)*

loss1s�<]u�       �	N��QXc�A�)*

loss�="��       �	i�QXc�A�)*

loss:�<�9��       �	X�QXc�A�)*

loss4�K:����       �	x��QXc�A�)*

loss-�<�o�8       �	__�QXc�A�)*

loss��;jy�_       �	��QXc�A�)*

loss���<��       �	���QXc�A�)*

loss_:mG�       �	O�QXc�A�)*

lossM#�;��`       �	D��QXc�A�)*

loss���9m��O       �	k��QXc�A�)*

loss��~<*pS�       �	xG�QXc�A�)*

loss���8�<uK       �	���QXc�A�)*

loss��9���       �	���QXc�A�)*

loss�T�8ݲ;%       �	���QXc�A�)*

loss��9����       �	a2�QXc�A�)*

loss�<�4�        �	V��QXc�A�)*

lossFY�<0�'{       �	t�QXc�A�)*

loss���;1ă�       �	��QXc�A�)*

lossݱ�<V0��       �	���QXc�A�)*

loss��<�U_       �	�y�QXc�A�)*

lossOQ:r���       �	��QXc�A�)*

loss��'='���       �	9��QXc�A�)*

lossVJ�;�<�2       �	�V�QXc�A�)*

lossaA�<�>BA       �	5��QXc�A�)*

loss��9�r�       �	u��QXc�A�)*

lossS�<� 8H       �	�0�QXc�A�)*

loss�,</4cX       �	��QXc�A�)*

loss״;"���       �	�Y�QXc�A�)*

loss4�:��o�       �	U��QXc�A�)*

lossĩ<���       �	$��QXc�A�)*

loss-��;�Sp       �	G8�QXc�A�)*

loss�Jz;����       �	���QXc�A�)*

lossR֦=�ˁ       �	�l�QXc�A�)*

loss�4:n��4       �	��QXc�A�)*

loss�}Y=���Y       �	���QXc�A�)*

loss�ȿ;.���       �	�D�QXc�A�)*

loss�#�<	��       �	w��QXc�A�)*

loss]c�<o�?       �	�q�QXc�A�)*

loss���;�ݑc       �	u9�QXc�A�)*

loss��;˯]�       �	e��QXc�A�)*

lossf�:N0@�       �	ѯ�QXc�A�)*

lossx�N:}2��       �	߇�QXc�A�)*

loss��/;�!=�       �	kI�QXc�A�)*

lossX-�;��x       �	 &�QXc�A�)*

lossz��;/��!       �	 ��QXc�A�)*

loss�v�;���       �	
��QXc�A�)*

loss�8�;��       �	�;�QXc�A�**

lossJ�:il       �	��QXc�A�**

loss
.=�vc       �	���QXc�A�**

loss��F<RH�       �	�f�QXc�A�**

lossL �<Ԝ^V       �	�*�QXc�A�**

loss
K&< Rj�       �	���QXc�A�**

loss�Ԣ=���       �	*��QXc�A�**

loss���;\�       �	}!�QXc�A�**

loss#��;��,       �	>\�QXc�A�**

loss��:;����       �	��QXc�A�**

loss���:	��       �	��QXc�A�**

lossÊ�=���       �	t��QXc�A�**

loss(]<(VS       �	[��QXc�A�**

loss;m�;uc��       �	���QXc�A�**

loss�h:�=/1       �	J�QXc�A�**

loss-#Q;��W       �	���QXc�A�**

lossĨN;���x       �	ҋ�QXc�A�**

loss�l=�.[       �	,)�QXc�A�**

loss]�<��T�       �	-��QXc�A�**

lossbȎ:eC1       �	��QXc�A�**

loss�SV<��l       �	�?�QXc�A�**

lossT��9�b&+       �	���QXc�A�**

lossҲ�<�X�       �	x~�QXc�A�**

loss6b :���i       �	6�QXc�A�**

lossc��;5
�>       �	N��QXc�A�**

lossv��;�]z�       �	�RXc�A�**

loss��}=�X�       �	�GRXc�A�**

losse�<W�n       �	��RXc�A�**

loss���;���*       �	�jRXc�A�**

loss{1V<���       �	�RXc�A�**

loss�6�<���?       �	��	RXc�A�**

losssZ	<|�=;       �	p#
RXc�A�**

loss��;�]�       �	x�
RXc�A�**

loss/��<���-       �	IKRXc�A�**

lossR�q<��       �	��RXc�A�**

loss¢�:�-)�       �	�zRXc�A�**

loss�	:H��       �	�RXc�A�**

loss6I<Ԣ[�       �	4�RXc�A�**

loss��<���]       �	�2RXc�A�**

lossR�C<#NE       �	T�RXc�A�**

loss)=�<���s       �	_RXc�A�**

lossx[	9���       �	+�RXc�A�**

loss��<���O       �	��RXc�A�**

loss���;�d��       �	: RXc�A�**

lossD�;]r��       �	��RXc�A�**

loss	=��Ƌ       �	,HRXc�A�**

loss�i�<T��d       �	��RXc�A�**

loss���:Sj[Z       �	LlRXc�A�**

loss�{<u���       �	�RXc�A�**

loss�$;<g��       �	�RXc�A�**

loss/~O;H��       �	}"RXc�A�**

lossڞ�=�4��       �	��RXc�A�**

lossd�9\�?p       �	�FRXc�A�**

loss�,�:-���       �	@�RXc�A�**

loss��; _��       �	;oRXc�A�**

loss�k�<#D�       �	�RXc�A�**

loss1cH;�gX       �	�RXc�A�**

losst��;��e       �	�.RXc�A�**

loss�~�;����       �	`RXc�A�**

loss�f<�z       �	m�RXc�A�**

loss�91<n��       �	)yRXc�A�**

loss�d:iC�e       �	gRXc�A�**

loss
[�<���o       �	e�RXc�A�**

lossl�=�ci�       �	/QRXc�A�**

loss�
3=B��N       �	�hRXc�A�**

loss c�:���;       �	�RXc�A�**

loss%A�:���f       �	ѓRXc�A�**

loss#�9��Z       �	(* RXc�A�**

loss�f<p���       �	s� RXc�A�**

loss.��;0���       �	;R!RXc�A�**

lossxS#:D���       �	�!RXc�A�**

lossn��:��%!       �	��"RXc�A�**

loss��I:��d�       �	�#RXc�A�**

loss��$:K1X       �	��#RXc�A�**

lossQQ�9)q��       �	 D$RXc�A�**

loss��9;����       �	E�$RXc�A�**

loss�I=8R�       �	�o%RXc�A�**

loss:1�;��0�       �	�
&RXc�A�**

loss;b�; P�:       �	��&RXc�A�**

loss��<��£       �	�>'RXc�A�**

loss�C:k��       �	��'RXc�A�**

lossd�N:6Lg       �	�{(RXc�A�**

loss]";� ٵ       �	T)RXc�A�**

loss��[:}.��       �	�)RXc�A�**

lossN-�=pu�       �	�M*RXc�A�**

loss�Z8=�
�       �	��*RXc�A�**

loss&G9L$�       �	��+RXc�A�**

lossݸ=:Ff�I       �	�",RXc�A�**

loss�?O<��y       �	��,RXc�A�**

loss�1:�9;|       �	�U-RXc�A�**

loss�DS:Yk��       �	0�-RXc�A�**

loss3�0;��P-       �	w�.RXc�A�**

loss�T6;���       �	P/RXc�A�**

lossHR;th�b       �	)�/RXc�A�**

lossv��;S;W       �	hY0RXc�A�**

loss�1;"�U�       �	Y�0RXc�A�**

loss��<<�,8       �	:�1RXc�A�**

loss���;�'�(       �	�/2RXc�A�**

loss��;��U�       �	 �2RXc�A�**

loss���;����       �	�c3RXc�A�**

lossQ��;-�4�       �	�4RXc�A�**

lossz�<��~�       �	��4RXc�A�**

loss Gz<�>v�       �	NA5RXc�A�**

loss�\;>ry       �	��5RXc�A�**

lossb�:�ኩ       �	�}6RXc�A�**

loss�;\�B       �	7RXc�A�**

loss�)�;!qO       �	p�7RXc�A�**

loss\��<G�O�       �	fK8RXc�A�**

loss��y9�B�       �	��8RXc�A�**

loss���<����       �	(�9RXc�A�**

lossS�<y�g6       �	�:RXc�A�**

lossZl:/)2�       �	h�:RXc�A�**

loss/Q=�{       �	�H;RXc�A�**

loss�:(;��6�       �	��;RXc�A�**

loss!�;�o�       �	;�<RXc�A�**

loss^�<2��m       �	�"=RXc�A�**

lossܳ�=��       �	-�=RXc�A�**

loss%�B;�W��       �	5_>RXc�A�**

loss�e�;@w��       �	?RXc�A�**

loss ��<�+�)       �	ǡ?RXc�A�**

loss
��:J��       �	W?@RXc�A�**

loss�z�<J7��       �	@�@RXc�A�**

loss�QG;� ��       �	J}ARXc�A�**

losss�)=���       �	BRXc�A�**

loss�l�:L��       �	>�BRXc�A�**

loss�j29�&�       �	EFCRXc�A�**

loss9#�       �	��CRXc�A�**

loss���;s��<       �	5|DRXc�A�**

lossD�;l�N       �	�ERXc�A�+*

loss��;��       �	Q�ERXc�A�+*

loss}�=Q?Vo       �	�XFRXc�A�+*

loss� ;�b�       �	��FRXc�A�+*

loss�3: ��       �	��GRXc�A�+*

loss��7<Qw�       �	#HRXc�A�+*

loss�`:�+�z       �	
�HRXc�A�+*

loss���;!&�       �	MIRXc�A�+*

loss��z<���#       �	e�IRXc�A�+*

loss��={�`�       �	�tJRXc�A�+*

loss�ہ9+�        �	-
KRXc�A�+*

loss,{%=�A}�       �	ϞKRXc�A�+*

lossoD0<���       �	4LRXc�A�+*

loss12�:�ǖ6       �	�LRXc�A�+*

loss}��<a�       �	�^MRXc�A�+*

lossQ�1<.���       �	��MRXc�A�+*

losst+:�@h�       �	��NRXc�A�+*

lossM(<\�M       �	mORXc�A�+*

losssD9<;��T       �	ݱORXc�A�+*

lossӃ:�n��       �	EGPRXc�A�+*

loss`�^;�|t       �	��PRXc�A�+*

loss�/�:�cMD       �	zoQRXc�A�+*

loss�/i9ٗm�       �	RRXc�A�+*

loss
��;��hg       �	Z�RRXc�A�+*

loss/`�;���       �	�1SRXc�A�+*

loss_8�;�7�       �	m�SRXc�A�+*

loss��q;2�j       �	XTRXc�A�+*

loss��;��       �	 �TRXc�A�+*

loss�,;���k       �	�URXc�A�+*

loss]�U<@%�       �	�VRXc�A�+*

loss�7�9X:t�       �	{�VRXc�A�+*

loss��=��$       �	`WWRXc�A�+*

loss��<       �	��WRXc�A�+*

loss��v<$�A�       �	�XRXc�A�+*

loss9�9��       �	�YRXc�A�+*

loss�[.;�	�n       �	9{ZRXc�A�+*

loss(q<_7�       �	�O[RXc�A�+*

loss:2jMg       �	_	\RXc�A�+*

lossUY;��+K       �	�$]RXc�A�+*

loss�;s�       �	��]RXc�A�+*

lossx�<m��S       �	+�^RXc�A�+*

loss/<}�.�       �	�Y_RXc�A�+*

losstJ:栅/       �	�Q`RXc�A�+*

lossN��9�x��       �	��`RXc�A�+*

lossibS:-�7�       �	"�aRXc�A�+*

loss�O:��yR       �	�nbRXc�A�+*

loss��=sB��       �	�cRXc�A�+*

lossAA%:lem�       �	S=dRXc�A�+*

loss�.�:{� �       �	��dRXc�A�+*

loss�;�^O       �	��eRXc�A�+*

loss�/#=��$�       �	qZfRXc�A�+*

loss��<?虍       �	��fRXc�A�+*

loss��8_R7[       �	S�gRXc�A�+*

loss�;�¾�       �	�<hRXc�A�+*

lossO46<�	x       �	��hRXc�A�+*

losslNS;�v�e       �	qriRXc�A�+*

loss��T9�F�       �	RjRXc�A�+*

lossz��;���       �	v�jRXc�A�+*

loss��P9{�?       �	�GkRXc�A�+*

loss�Vo;���       �	z�kRXc�A�+*

loss�5G9W�0       �	�lRXc�A�+*

loss��U<sm�       �	�NmRXc�A�+*

loss��O:0�o)       �	{�mRXc�A�+*

loss&�;@K*"       �	>�nRXc�A�+*

loss3&;�3U�       �	�0oRXc�A�+*

loss�)<���       �	�oRXc�A�+*

loss]d:o�l�       �	 rpRXc�A�+*

loss�ߴ:�MJT       �	qRXc�A�+*

loss�O=ןs       �	ƧqRXc�A�+*

lossa:a>�<       �	�{rRXc�A�+*

loss��;8�|�       �	YsRXc�A�+*

loss��=����       �	��sRXc�A�+*

lossz��=��]�       �	�{tRXc�A�+*

lossqTp<���       �	#uRXc�A�+*

loss��;��#Y       �	\�uRXc�A�+*

lossq#�;�3(M       �	9CvRXc�A�+*

loss���:,�?       �	W�vRXc�A�+*

lossn��9@�ݘ       �	��wRXc�A�+*

loss��;���       �	�xRXc�A�+*

loss��%<"�X       �	��xRXc�A�+*

loss�xX<�9�       �	)>yRXc�A�+*

loss,��:}w�)       �	��yRXc�A�+*

lossl�<r��       �	�izRXc�A�+*

lossD��=���       �	J{RXc�A�+*

loss�6p9�?x       �	��{RXc�A�+*

loss�q=���       �	4|RXc�A�+*

loss�J<��F7       �	��|RXc�A�+*

loss���<_[ �       �	'f}RXc�A�+*

loss��;�{�       �	6~RXc�A�+*

loss��=S��.       �	NRXc�A�+*

lossI$�;]f�       �	��RXc�A�+*

loss���:I|iC       �	q��RXc�A�+*

loss��A;ݍ�W       �	:$�RXc�A�+*

lossY5 <�Pd�       �	
��RXc�A�+*

loss�j�;#�)_       �	IM�RXc�A�+*

loss��<U�       �	L�RXc�A�+*

loss�F�:xJ��       �	�u�RXc�A�+*

loss�i�=8Sv�       �	h�RXc�A�+*

loss.��:�n�       �	ϟ�RXc�A�+*

loss#�;���       �	j4�RXc�A�+*

loss{!m<!`��       �	�ʅRXc�A�+*

loss��9���V       �	�g�RXc�A�+*

lossot�9���g       �	���RXc�A�+*

loss���<)f�       �	��RXc�A�+*

loss�:4�O       �	�3�RXc�A�+*

loss���<p.��       �	�͈RXc�A�+*

loss��;��!�       �	B`�RXc�A�+*

lossr�\=����       �	_�RXc�A�+*

loss@M�</N�       �	���RXc�A�+*

loss��:�M�M       �	�G�RXc�A�+*

loss �;]Us       �	C�RXc�A�+*

lossF}<'�iH       �	w�RXc�A�+*

loss�F=���/       �	"�RXc�A�+*

loss�ξ<�â�       �	#��RXc�A�+*

loss��)<��B�       �	^�RXc�A�+*

loss���<�'�)       �	u�RXc�A�+*

lossA�1; �5)       �	��RXc�A�+*

loss�Ey={gz�       �	��RXc�A�+*

loss��;����       �	w�RXc�A�+*

loss}*�;�*�       �	���RXc�A�+*

loss.2<P�	�       �	;:�RXc�A�+*

loss!"&9���       �	�ҒRXc�A�+*

lossZe�<��L5       �	�e�RXc�A�+*

loss���;��ǵ       �	�RXc�A�+*

loss1�Z<AN�       �	O��RXc�A�+*

loss�1:�-H       �	F�RXc�A�+*

loss}��:�d�       �	�ٕRXc�A�+*

loss�	�<����       �	�m�RXc�A�+*

loss!/e;���I       �	O�RXc�A�,*

lossݔ�<���       �	��RXc�A�,*

loss�^�;�!��       �	f2�RXc�A�,*

lossnq�<\]��       �	�̘RXc�A�,*

lossV�<�
�|       �	nk�RXc�A�,*

loss��:�7Ha       �	��RXc�A�,*

loss�P=@E�       �	 ��RXc�A�,*

loss�h�8^�?�       �	�V�RXc�A�,*

loss�@;��s�       �	���RXc�A�,*

lossH[�<�`
       �	���RXc�A�,*

loss=(Q<��       �	_(�RXc�A�,*

loss�|�;X d�       �	;ŝRXc�A�,*

loss/�K<��4+       �	)\�RXc�A�,*

loss79m;����       �	/��RXc�A�,*

loss^;a�U       �	}��RXc�A�,*

loss��;�e�T       �	�)�RXc�A�,*

loss�h�:��2J       �	���RXc�A�,*

loss�;=���w       �	&Q�RXc�A�,*

loss�-<�}�       �	��RXc�A�,*

loss�J/:)A��       �	�{�RXc�A�,*

loss��
<��d*       �	��RXc�A�,*

loss�=��̩       �	ĳ�RXc�A�,*

loss��=<��       �	�O�RXc�A�,*

loss7y�<3B�       �	m�RXc�A�,*

loss���:|�       �	6w�RXc�A�,*

loss�s�:��       �	�RXc�A�,*

loss);|�b�       �	���RXc�A�,*

loss��9AP�       �	7�RXc�A�,*

lossQ�;�'[�       �	�ɧRXc�A�,*

loss�:ӳ�       �	t\�RXc�A�,*

loss��)<�&�       �	��RXc�A�,*

loss�� ;{D�       �	��RXc�A�,*

loss�/�: U�       �	��RXc�A�,*

loss(��;�ņ'       �	���RXc�A�,*

loss��i9u,�       �	�C�RXc�A�,*

loss�z�<x�       �	0׫RXc�A�,*

loss�<��J       �	�n�RXc�A�,*

lossP�:y��i       �	��RXc�A�,*

lossh�;�,�       �	���RXc�A�,*

loss
�:X8:       �	�<�RXc�A�,*

lossWZv<͞d�       �	�ԮRXc�A�,*

lossZ<�<���       �	�n�RXc�A�,*

lossڲ<�>       �	��RXc�A�,*

loss~�<����       �	�ܰRXc�A�,*

lossL�J<��       �	�m�RXc�A�,*

loss���9tC�I       �	�̲RXc�A�,*

lossWj�;9�4�       �	>w�RXc�A�,*

loss�=<�D�(       �	��RXc�A�,*

loss�(�:_;��       �	$��RXc�A�,*

loss2+�8��       �	N�RXc�A�,*

loss��<��	�       �	�D�RXc�A�,*

loss�1�<ߨ24       �	b۶RXc�A�,*

loss�.<C�%       �	t}�RXc�A�,*

lossѕ�;��T        �	�RXc�A�,*

loss��<a��       �	���RXc�A�,*

lossh&<*��E       �	�I�RXc�A�,*

lossT��=�^o       �	t'�RXc�A�,*

lossM�	<�hͣ       �	���RXc�A�,*

loss��;�anh       �	�W�RXc�A�,*

loss\��9�9��       �	��RXc�A�,*

loss=�:l�       �	R}�RXc�A�,*

loss���9#�U�       �	WZ�RXc�A�,*

loss���:Qɤ�       �	���RXc�A�,*

loss�D:tt�X       �	�6�RXc�A�,*

lossZ�
=6�i       �	ҿRXc�A�,*

lossfwL=�lݼ       �	�m�RXc�A�,*

lossdR�:ܨ       �	��RXc�A�,*

loss�q=w�YN       �	(��RXc�A�,*

loss�.�;.�"C       �	�?�RXc�A�,*

loss}�<�A�       �	��RXc�A�,*

loss;B"<=J��       �	;q�RXc�A�,*

losss�;=.��       �	F�RXc�A�,*

lossoP�;z�       �	;��RXc�A�,*

loss��;���}       �	=�RXc�A�,*

loss��=�'v%       �	L��RXc�A�,*

loss���8�/m*       �	�s�RXc�A�,*

lossZ��:�q�{       �	S�RXc�A�,*

loss���;^\?k       �	-��RXc�A�,*

lossMa�<
<R       �	S�RXc�A�,*

loss�E=�$�       �	���RXc�A�,*

loss{Q\<�yv�       �	S��RXc�A�,*

loss���;��       �	T�RXc�A�,*

loss�i�9�T��       �	|��RXc�A�,*

lossm�	;K       �	/��RXc�A�,*

losshL:��1�       �	V��RXc�A�,*

loss*�:!���       �	�R�RXc�A�,*

lossxh�<���       �	���RXc�A�,*

loss��X<Bm�       �	�~�RXc�A�,*

loss�3�<�o|�       �	s�RXc�A�,*

loss��E;�P��       �	~��RXc�A�,*

loss��5=����       �	�;�RXc�A�,*

loss<��;d��>       �	���RXc�A�,*

lossI�w:uf�       �	߇�RXc�A�,*

lossfs�;2�       �	�)�RXc�A�,*

loss<��<nI,�       �	H��RXc�A�,*

loss�0b<�#md       �	Wv�RXc�A�,*

loss�i�;vX'       �	_�RXc�A�,*

loss��;?Q       �	,f�RXc�A�,*

losss�0=��       �	���RXc�A�,*

loss��>:�¸5       �	���RXc�A�,*

loss�<��       �	�!�RXc�A�,*

loss�:{:�}\       �	p��RXc�A�,*

lossWsE:G�!y       �	g�RXc�A�,*

loss��*<�`޳       �	?�RXc�A�,*

loss�9��>       �	H��RXc�A�,*

loss�f�<��~�       �	�:�RXc�A�,*

loss�y;�68       �	�RXc�A�,*

loss�AY=/|�V       �	���RXc�A�,*

loss%��:�^�?       �	'j�RXc�A�,*

loss?�9 Ƥ       �	��RXc�A�,*

loss��<S�ג       �	���RXc�A�,*

loss]�e<��t       �	���RXc�A�,*

lossdC�<�|\       �	X�RXc�A�,*

loss�=��;6       �	4�RXc�A�,*

loss~s�<�|��       �	���RXc�A�,*

loss,p�:6s�u       �	+��RXc�A�,*

loss�~�<�H�       �	���RXc�A�,*

loss	��;��       �	�b�RXc�A�,*

loss<u�;]]��       �	 `�RXc�A�,*

loss�;i&��       �	��RXc�A�,*

loss4W+9l;�       �	e��RXc�A�,*

loss���=�ݲ�       �	��RXc�A�,*

loss���<״d�       �	�/�RXc�A�,*

loss�(4<����       �	sf�RXc�A�,*

lossV�:��l>       �	rm�RXc�A�,*

loss�<��       �	��RXc�A�,*

lossp��<�}��       �	��RXc�A�,*

lossL*:a:��       �	��RXc�A�,*

loss�W\<�.�       �	�RXc�A�-*

loss� =�]�^       �	ϻ�RXc�A�-*

loss��+9�
*�       �	���RXc�A�-*

losszh<�/K�       �	8��RXc�A�-*

loss;��;�+       �	��RXc�A�-*

loss�k�<�T��       �	�O�RXc�A�-*

lossͫ5:ī7�       �	z4�RXc�A�-*

loss�b�;t|7�       �	^��RXc�A�-*

loss.%;�H�       �	$�RXc�A�-*

loss�4;K@��       �	OW�RXc�A�-*

loss���<�;F�       �	sL�RXc�A�-*

loss�<�zH%       �	�P�RXc�A�-*

loss%l�;�m��       �	��RXc�A�-*

loss���<�P�       �	���RXc�A�-*

loss&8{:xA�       �	�|�RXc�A�-*

loss�� < �4�       �	�B�RXc�A�-*

loss�=8��C       �	���RXc�A�-*

loss��<��+�       �	��RXc�A�-*

loss�k�<���       �	���RXc�A�-*

loss��<b�:c       �	�� SXc�A�-*

loss`�)<mи       �	�ESXc�A�-*

loss�=���       �	y�SXc�A�-*

lossn�y<QZ�5       �	��SXc�A�-*

loss��7=���       �	t�SXc�A�-*

lossO�<<W���       �	ӟSXc�A�-*

loss��i:]�P�       �	=DSXc�A�-*

loss���=r���       �	�<SXc�A�-*

loss��X;;+�       �	��SXc�A�-*

loss-<�;��M       �	uuSXc�A�-*

loss�u�:�h�.       �	�ESXc�A�-*

lossT�E:@�       �	�K	SXc�A�-*

loss�-<�Ȇ       �	�	SXc�A�-*

loss�4;9�'i       �	9�
SXc�A�-*

loss��_9�X��       �	��SXc�A�-*

loss���<,X�       �	��SXc�A�-*

loss��F8��       �	�VSXc�A�-*

loss���:/       �	�eSXc�A�-*

lossC/t827j       �	�+SXc�A�-*

lossZ�9���       �	�SXc�A�-*

loss�G :.:��       �	|�SXc�A�-*

lossa�@==��       �	(�SXc�A�-*

lossX�9X��       �	)�SXc�A�-*

loss��<���       �	�zSXc�A�-*

loss���;���       �	SXc�A�-*

loss��9{N�1       �	��SXc�A�-*

loss��9;���       �	bKSXc�A�-*

losseq�:��tg       �	&�SXc�A�-*

loss�V;�SA       �	=�SXc�A�-*

lossrk�;�X�Z       �	#SXc�A�-*

loss��5=aFz       �	½SXc�A�-*

loss��0<U��       �	!VSXc�A�-*

loss�'l=9��       �	��SXc�A�-*

loss�^<C��       �	�SXc�A�-*

lossU�9��z       �	LOSXc�A�-*

loss�w=� ]�       �	;:SXc�A�-*

loss��=�ɞ       �	�mSXc�A�-*

lossc�/:-�j�       �	
SXc�A�-*

loss���9�G        �	��SXc�A�-*

loss�ZV:a�Y       �	��SXc�A�-*

lossp_�<h&�       �	e�SXc�A�-*

loss�2s<�RU       �	�� SXc�A�-*

loss��-:�|       �	��!SXc�A�-*

loss��;�Z݁       �	��"SXc�A�-*

loss!w�9�׷(       �	� $SXc�A�-*

loss��^=.(�       �	E%SXc�A�-*

loss�t=D u       �	�&SXc�A�-*

loss��:����       �	�/'SXc�A�-*

loss��C<�Q*�       �	�(SXc�A�-*

loss��l<@       �	5F)SXc�A�-*

loss��<=��z�       �	7�)SXc�A�-*

loss��:@�E       �	�~*SXc�A�-*

lossFL�<��y�       �	a�+SXc�A�-*

loss��M=��       �	k},SXc�A�-*

lossC��;�-�s       �	�-SXc�A�-*

loss�d]=�7J       �	��.SXc�A�-*

loss o�;=}B�       �	̳/SXc�A�-*

loss��K=\�ѽ       �	ũ0SXc�A�-*

loss���;����       �	_B1SXc�A�-*

loss�)=!�x�       �	^�1SXc�A�-*

loss��:��s4       �	p2SXc�A�-*

loss�?�<���       �	�3SXc�A�-*

lossj�'<�B�       �	Ԝ3SXc�A�-*

loss�K�<��֌       �	<4SXc�A�-*

loss���=8�       �	��4SXc�A�-*

loss�?�;�`�       �	�p5SXc�A�-*

lossFyD;���       �	P6SXc�A�-*

loss�:<��ȭ       �	d�6SXc�A�-*

loss
VJ<z���       �	XU7SXc�A�-*

loss_�;j�/�       �	��7SXc�A�-*

losse��:o<��       �	P�8SXc�A�-*

lossoV�=�A�       �	�,9SXc�A�-*

lossJ|=$�;       �	��9SXc�A�-*

loss!g�;�xf       �	�q:SXc�A�-*

losssڱ<���       �	�E;SXc�A�-*

lossX �<�3<       �	L�;SXc�A�-*

loss\<��3$       �	<SXc�A�-*

loss�=��¥       �	�=SXc�A�-*

loss�d�;��Z~       �	��=SXc�A�-*

loss�9;��       �	~Q>SXc�A�-*

lossa_�;ّ�       �	��>SXc�A�-*

loss_�<Fr�t       �	8�?SXc�A�-*

loss)e�<��@s       �	�@SXc�A�-*

loss?%X<�?aq       �	F�@SXc�A�-*

lossEi;�^��       �	0IASXc�A�-*

loss�8�=\��       �	8�ASXc�A�-*

loss���=��P       �	?tBSXc�A�-*

loss�==&x`       �	�CSXc�A�-*

lossn�;��W       �	�DSXc�A�-*

loss � :{C�>       �	�DSXc�A�-*

loss͒T9��#�       �	�8ESXc�A�-*

loss}�:j�       �	��ESXc�A�-*

loss.�==���       �	UkFSXc�A�-*

loss��<;Z��V       �	��FSXc�A�-*

lossli<�/k)       �	h�GSXc�A�-*

loss;�+=�Gq�       �	!=HSXc�A�-*

loss�!';m2�       �	�ISXc�A�-*

loss���;^$g�       �	LlJSXc�A�-*

loss?{|;p�       �	�KSXc�A�-*

loss��<@�,       �	$�KSXc�A�-*

lossN�:���P       �	fhLSXc�A�-*

lossjg=y�M�       �	UMSXc�A�-*

loss�$�<qs�       �	��MSXc�A�-*

lossE��<c�       �	��NSXc�A�-*

loss��<��V       �	R'OSXc�A�-*

lossL��9Q���       �	�OSXc�A�-*

loss�YQ;B��       �	��PSXc�A�-*

loss�m�:�&r�       �	�#QSXc�A�-*

loss��9#\�       �	�QSXc�A�-*

loss,<# ��       �	�TRSXc�A�.*

loss�`2;����       �	��RSXc�A�.*

lossq,=(���       �	��SSXc�A�.*

loss��<�37       �	�TSXc�A�.*

loss�{2;W�       �	ٲTSXc�A�.*

lossmK];���       �	IUSXc�A�.*

loss�%;���       �	h�USXc�A�.*

loss��b<�b�       �	$}VSXc�A�.*

lossq�;T�"P       �	zWSXc�A�.*

loss�a=�>3�       �	��WSXc�A�.*

loss��
:��Z�       �	;UXSXc�A�.*

lossZR�<��p�       �	d�XSXc�A�.*

loss��U;�{�       �	��YSXc�A�.*

losse�0<7G��       �	�ZSXc�A�.*

lossd��:�n��       �	d�ZSXc�A�.*

loss���;���       �	 T[SXc�A�.*

loss��;MV�n       �	�\SXc�A�.*

lossmpl;)���       �	�]SXc�A�.*

loss�3;s��       �	t�]SXc�A�.*

lossᐌ;�Ť       �	�^SXc�A�.*

lossL�(<�%MH       �	��_SXc�A�.*

lossh�=;�N �       �	��`SXc�A�.*

loss	�;G66       �	w�aSXc�A�.*

loss�:�i�       �	�bSXc�A�.*

loss��Z;�nO�       �	�	cSXc�A�.*

lossO�"=7�u�       �		NdSXc�A�.*

lossF@<��]       �	%�eSXc�A�.*

lossC�&;���       �	p|gSXc�A�.*

loss;0�<,M<       �	(hSXc�A�.*

loss�;=k{�       �	��hSXc�A�.*

lossH �:?��w       �	��iSXc�A�.*

lossF�<��h       �	yvjSXc�A�.*

lossm�9:=��+       �	4kSXc�A�.*

loss��<\�^       �	��kSXc�A�.*

loss�p�;���       �	JlSXc�A�.*

lossD

8�pR�       �	��lSXc�A�.*

lossro;� s5       �	xmSXc�A�.*

lossRy�<��M�       �	nSXc�A�.*

lossS	P<�؈�       �	;�nSXc�A�.*

lossҎ�9�(߸       �	xGoSXc�A�.*

loss�
N<A��       �	��oSXc�A�.*

loss�ˏ=Ѵ�       �	rpSXc�A�.*

loss��`<*��c       �	]�qSXc�A�.*

lossO�:�)�       �	�)rSXc�A�.*

loss2;�X�       �	7�rSXc�A�.*

lossZ�:�V��       �	�]sSXc�A�.*

loss��9��       �	v�sSXc�A�.*

loss���:��Id       �	m�tSXc�A�.*

loss��):X��%       �	m uSXc�A�.*

lossi�^<t�>�       �	�uSXc�A�.*

lossZ��<�h��       �	MMvSXc�A�.*

loss��7$!)�       �	��vSXc�A�.*

loss-�	<2)�       �	xwSXc�A�.*

loss��8;:�       �	��ySXc�A�.*

loss�%�9�KE�       �	/�zSXc�A�.*

loss�;M8��5.       �	�"{SXc�A�.*

loss�\=�G       �	��{SXc�A�.*

loss3;+�       �	�V|SXc�A�.*

loss��N:+� @       �	%�|SXc�A�.*

loss�U�:#Q�       �	=}SXc�A�.*

loss�W<Χ��       �	'3~SXc�A�.*

loss��:<̟��       �	*�~SXc�A�.*

loss�s9���       �	^�SXc�A�.*

loss���=�u       �	�:�SXc�A�.*

loss@==��       �	[�SXc�A�.*

loss���;k�e       �	&��SXc�A�.*

loss��<���r       �	�*�SXc�A�.*

loss��:0�       �	LƂSXc�A�.*

loss�|�=4�};       �	�`�SXc�A�.*

lossf�=�#��       �	+��SXc�A�.*

loss� <GlU�       �	ݔ�SXc�A�.*

loss
bC<"�6p       �	.�SXc�A�.*

loss��~:m>��       �	�̅SXc�A�.*

loss1��;���       �	�i�SXc�A�.*

loss,�a=��       �	H��SXc�A�.*

loss�(�8L��       �	震SXc�A�.*

loss���<N�1V       �	RD�SXc�A�.*

loss{H<�bq�       �	^وSXc�A�.*

lossr_7=�F�       �	�k�SXc�A�.*

loss2o< �       �	��SXc�A�.*

loss5�;��       �	/��SXc�A�.*

loss�};�3 E       �	�5�SXc�A�.*

loss��:J��       �	�ŋSXc�A�.*

loss6�<>qM�n       �	�V�SXc�A�.*

lossFR;'�Ģ       �	O�SXc�A�.*

loss��:�b�       �	hz�SXc�A�.*

loss��:<����       �	��SXc�A�.*

lossf��:O�       �	D��SXc�A�.*

loss#��;��-J       �	F�SXc�A�.*

loss)�r:;!�       �	�SXc�A�.*

loss��;�O��       �	�x�SXc�A�.*

loss���;`T�u       �	��SXc�A�.*

lossnvK:��Ѐ       �	H��SXc�A�.*

loss@�I<T�Q1       �	�@�SXc�A�.*

loss�eD;�;7�       �	<ےSXc�A�.*

loss��:�~�	       �	�t�SXc�A�.*

loss_�*<��V       �	]�SXc�A�.*

loss�:���       �	��SXc�A�.*

loss�7-=ֺ�       �	�N�SXc�A�.*

loss�ϩ<e       �	�'�SXc�A�.*

loss�I�;	0/�       �		��SXc�A�.*

loss�=��g       �	6Z�SXc�A�.*

lossϺ;:��q       �	V�SXc�A�.*

lossJ:O!��       �	��SXc�A�.*

lossV��:�­�       �	�SXc�A�.*

lossAZ<EY�       �	ᶙSXc�A�.*

loss�,�<Ǳ�       �	O�SXc�A�.*

loss:G�<>��Z       �	&��SXc�A�.*

lossH�<+1��       �	�v�SXc�A�.*

loss/�:����       �	�,�SXc�A�.*

loss�][;�]�q       �	
�SXc�A�.*

lossftF;�$�       �	{��SXc�A�.*

loss�t,:�߇�       �	�E�SXc�A�.*

lossl�F<�nJ�       �	*7�SXc�A�.*

loss�:c��       �	�;SXc�A�.*

loss�!;�i��       �	=d�SXc�A�.*

lossQ�$<�J�2       �	b��SXc�A�.*

loss?�&<B��       �	���SXc�A�.*

loss\3=�|g�       �	��SXc�A�.*

lossR	�;�_k*       �	��SXc�A�.*

loss+�<�P��       �	N��SXc�A�.*

loss�|Z=u�       �	��SXc�A�.*

loss�`�<µV       �	II�SXc�A�.*

losshJ�;�
^       �	���SXc�A�.*

loss[
I;�c�$       �	��SXc�A�.*

loss���<ZM       �	6V�SXc�A�.*

loss5�<k��       �	���SXc�A�.*

lossD�<\#��       �	��SXc�A�.*

loss�g�<'��'       �	:<�SXc�A�/*

loss���:*m�       �	���SXc�A�/*

loss�
o;2��       �	 ~�SXc�A�/*

loss_��;}j#       �	&q�SXc�A�/*

loss\��=�nN       �	��SXc�A�/*

loss��;m��       �	��SXc�A�/*

lossQ*�<%c��       �	�@�SXc�A�/*

loss8�p<�2��       �	O�SXc�A�/*

loss.�}=UZ�5       �	���SXc�A�/*

loss�_}=x
ſ       �	�1�SXc�A�/*

loss&��:�H�       �	���SXc�A�/*

loss �;:�+       �	�a�SXc�A�/*

loss)zu;u��       �	���SXc�A�/*

loss��0<�ܸ       �	|�SXc�A�/*

lossZq<��       �	d�SXc�A�/*

loss#��<m�M�       �	(��SXc�A�/*

loss���:�wH�       �	�U�SXc�A�/*

loss��;�[�       �	��SXc�A�/*

loss��;�o�       �	o��SXc�A�/*

loss��;�       �	;�SXc�A�/*

loss剽<�/S       �	y��SXc�A�/*

loss�K;D       �	�A�SXc�A�/*

loss��1;���M       �	.=�SXc�A�/*

loss�c= �?       �	���SXc�A�/*

loss<[w       �	Pr�SXc�A�/*

loss��<�&��       �	��SXc�A�/*

loss�P<͑��       �	˞�SXc�A�/*

lossR:�:��h.       �	�0�SXc�A�/*

loss���<j���       �	���SXc�A�/*

loss�Õ;�t       �	�^�SXc�A�/*

lossÄ�<R�ͳ       �	���SXc�A�/*

losso��<j��       �	f��SXc�A�/*

loss�K;C/lx       �	��SXc�A�/*

loss�i;cB
       �	)��SXc�A�/*

loss�tm;�Kg       �	�A�SXc�A�/*

loss��9�Oϰ       �	���SXc�A�/*

loss��;?9A       �	A��SXc�A�/*

loss�5�<�N�]       �	��SXc�A�/*

losspb=�:)       �	ۤ�SXc�A�/*

loss�&�:�X��       �	��SXc�A�/*

loss��Z:r>�L       �	��SXc�A�/*

loss6��;O��       �	~��SXc�A�/*

loss�v�8?�Ay       �	1C�SXc�A�/*

lossX�*:T��       �	��SXc�A�/*

lossfm=d6�~       �	�j�SXc�A�/*

loss]�=o�k�       �	��SXc�A�/*

loss�%<7�i0       �	���SXc�A�/*

lossZ��9h/       �	�$�SXc�A�/*

loss��;A�1T       �	�y�SXc�A�/*

loss�|�;��)n       �	��SXc�A�/*

loss�J�:��×       �	N��SXc�A�/*

loss>�<�bF       �	�7�SXc�A�/*

lossl��;?��)       �	���SXc�A�/*

loss?1�;��       �	f�SXc�A�/*

loss� �;!DPK       �	T��SXc�A�/*

loss�7H;bE�       �	���SXc�A�/*

lossõh<J�m       �	�)�SXc�A�/*

lossM�c<,���       �	��SXc�A�/*

loss�;:�[       �	&R�SXc�A�/*

loss�w2;����       �	F��SXc�A�/*

loss:l =�T�j       �	��SXc�A�/*

loss���<��B       �	r�SXc�A�/*

lossm�;i�<       �	.��SXc�A�/*

loss�~:\e       �	P�SXc�A�/*

loss�^<\�       �	���SXc�A�/*

loss��-<p��       �	=|�SXc�A�/*

loss1rn<��)       �	��SXc�A�/*

loss���<��h       �	Ƣ�SXc�A�/*

loss�\9mSpY       �	8�SXc�A�/*

loss���;�1u]       �	���SXc�A�/*

loss���:3��5       �	�g�SXc�A�/*

lossT2:�!��       �	� �SXc�A�/*

lossO![<ǯ��       �	���SXc�A�/*

loss� <ƳF       �	|)�SXc�A�/*

loss_=cC�e       �	�S�SXc�A�/*

loss(M=�T       �	���SXc�A�/*

lossa`�:H�"�       �	�SXc�A�/*

loss�L:��       �	c'�SXc�A�/*

loss��;���'       �	���SXc�A�/*

loss���;%k��       �	�i�SXc�A�/*

loss?�H:�#��       �	��SXc�A�/*

lossST=C�К       �	���SXc�A�/*

loss}�;�!�       �	&��SXc�A�/*

loss�]<�&��       �	���SXc�A�/*

loss��9�0�       �	zP�SXc�A�/*

loss&��8 �       �	^� TXc�A�/*

loss�T1:Dn!�       �	7�TXc�A�/*

lossg;<c%��       �	\TXc�A�/*

lossW��;=�V       �	��TXc�A�/*

loss�*p;w�<�       �	t@TXc�A�/*

loss��=���0       �	�TXc�A�/*

loss���;)��)       �	bhTXc�A�/*

loss{�?<�`        �	lTXc�A�/*

lossr��=�7�       �	��TXc�A�/*

loss�r�:	�       �	�-TXc�A�/*

lossa�6<���*       �	�TXc�A�/*

loss�W�<4f<@       �	�^TXc�A�/*

lossf�:�Nt~       �	� TXc�A�/*

lossI`�<���       �	��TXc�A�/*

loss�U�<ĭ�       �	�4	TXc�A�/*

loss��:`��       �	��	TXc�A�/*

loss\1�;M�2�       �	,f
TXc�A�/*

loss��S<7-�       �	��
TXc�A�/*

loss���;� ��       �	�TXc�A�/*

loss���;b_�       �	(*TXc�A�/*

loss�9�:�sע       �	v�TXc�A�/*

lossJ T;��i       �	�XTXc�A�/*

loss���<��>C       �	5�TXc�A�/*

loss�$�;#��       �	(�TXc�A�/*

losst��;aR�       �	+TXc�A�/*

loss�(`;ˡ�       �	y�TXc�A�/*

lossq��<1E�       �	�CTXc�A�/*

losse��;�5�       �	/�TXc�A�/*

loss���<4x��       �	��TXc�A�/*

loss�in;e�z�       �	�GTXc�A�/*

loss�6E;L�D�       �	O�TXc�A�/*

lossמ3=#��       �	V�TXc�A�/*

loss��;���p       �	TXc�A�/*

lossfN<j�ȷ       �	��TXc�A�/*

loss�og=g_Wq       �		TTXc�A�/*

loss{i�<3&5o       �	��TXc�A�/*

loss&zl:N��A       �	�~TXc�A�/*

loss��U<�       �	TXc�A�/*

loss� /;�xd�       �	�TXc�A�/*

loss$&;Eh_�       �	jNTXc�A�/*

loss	X�:�h�       �	a�TXc�A�/*

loss@��:���       �	yTXc�A�/*

loss�{)<::Zp       �	�TXc�A�/*

loss�  =���       �	��TXc�A�0*

losslC�:��e�       �	ETXc�A�0*

loss��:���o       �	5�TXc�A�0*

loss�=��       �	�+TXc�A�0*

loss�89NMƀ       �	��TXc�A�0*

loss�h�:���       �	x_TXc�A�0*

loss,�9����       �	��TXc�A�0*

lossBe:U�1�       �	n� TXc�A�0*

loss���9�&�       �	�}!TXc�A�0*

lossW5�<�5�       �	�"TXc�A�0*

loss�~<��I�       �	:�"TXc�A�0*

loss�P<���I       �	�C#TXc�A�0*

loss�c�;tVC^       �	��#TXc�A�0*

loss��2:c��       �	�$TXc�A�0*

loss�ى;?�%       �	�#%TXc�A�0*

loss�P�;x�R�       �	�%TXc�A�0*

losso��<��       �	:X&TXc�A�0*

loss�:=�B�O       �	8�&TXc�A�0*

loss�ޔ=��b       �	r�'TXc�A�0*

loss�];����       �	�.(TXc�A�0*

loss}��;R��       �	)�(TXc�A�0*

lossL	B;��       �	lw)TXc�A�0*

loss��:1       �	�*TXc�A�0*

lossm9�٥�       �	��*TXc�A�0*

loss���9
�h       �	\Y+TXc�A�0*

loss�U�=t��c       �	��+TXc�A�0*

loss�:>�7&       �	H�,TXc�A�0*

loss��</y�       �	k*-TXc�A�0*

lossݔ+:���U       �	&�-TXc�A�0*

loss��];���       �	�q.TXc�A�0*

loss�N;���       �	�/TXc�A�0*

loss��J;�3ھ       �	�/TXc�A�0*

loss{�l:��%       �	qX0TXc�A�0*

loss �5=��%       �	C1TXc�A�0*

loss��;��       �	�1TXc�A�0*

lossq��<�"@#       �	�M2TXc�A�0*

loss��F<:�r       �	K�2TXc�A�0*

loss�.�8WH��       �	u�3TXc�A�0*

lossM�7:'ag       �	v24TXc�A�0*

loss�9�h�t       �	_�4TXc�A�0*

loss��t:
T�(       �	*s5TXc�A�0*

loss�eE;@���       �	6TXc�A�0*

loss�b9�:�p       �	3�6TXc�A�0*

loss�<�^)�       �	)@7TXc�A�0*

loss=�i;�]T�       �	D�7TXc�A�0*

loss �*:OHl       �	dt8TXc�A�0*

loss���<+�	       �	p9TXc�A�0*

loss=m�;       �	�9TXc�A�0*

loss;�;����       �	m9:TXc�A�0*

loss���:
\�       �	8�:TXc�A�0*

lossyО:��K�       �	m�;TXc�A�0*

loss]R&=�S_        �	+<TXc�A�0*

loss	ƀ;��l       �	T�<TXc�A�0*

losshX�:1���       �	R`=TXc�A�0*

loss)�A=xCG=       �	�;>TXc�A�0*

loss���<�,i�       �	��>TXc�A�0*

loss�
�;��>       �	��?TXc�A�0*

losst)�9j�>X       �	�1@TXc�A�0*

loss66�;��`�       �	��@TXc�A�0*

lossL�<�@t�       �	�dATXc�A�0*

lossc��<�"��       �	�.BTXc�A�0*

loss{G;�I�G       �	L�BTXc�A�0*

loss�CR;%��,       �	�_CTXc�A�0*

loss#~�<Wљ�       �	T9DTXc�A�0*

loss�5�8��B�       �	��DTXc�A�0*

loss#�M<��_�       �	�lETXc�A�0*

loss�9M`��       �	5FTXc�A�0*

loss�c;�Mj       �	�FTXc�A�0*

loss�d�:.���       �	�IGTXc�A�0*

lossXُ90!�       �	�4HTXc�A�0*

loss}��:��       �	,�HTXc�A�0*

loss2�k=����       �	�hITXc�A�0*

loss!M�;�#)y       �	mJTXc�A�0*

loss-0=�Ur       �	��JTXc�A�0*

lossY�< ���       �	�5KTXc�A�0*

loss�Ih<�=��       �	,�KTXc�A�0*

loss��{<�pH�       �	]lLTXc�A�0*

loss.E<Uh�       �	-	MTXc�A�0*

lossτ.;����       �	b�MTXc�A�0*

lossI��:� ~�       �	�,NTXc�A�0*

loss��>JMo       �	׿NTXc�A�0*

loss��'<n�@       �	uUOTXc�A�0*

loss�-+=jղ_       �	��OTXc�A�0*

loss_Т;=޸�       �	��PTXc�A�0*

loss��;"O��       �	)&QTXc�A�0*

loss�7�;J��m       �	��QTXc�A�0*

loss2�<�%�       �	aPRTXc�A�0*

loss�τ:%�r       �	��RTXc�A�0*

loss�ݦ<vpS�       �	ɐSTXc�A�0*

lossN%�=�cJ.       �	�0TTXc�A�0*

loss��;��       �	��TTXc�A�0*

loss��<��F�       �	�bUTXc�A�0*

loss^g;D�)�       �	'�UTXc�A�0*

loss�z<j�\�       �	τVTXc�A�0*

loss#5<_�[�       �	�WTXc�A�0*

lossN�<����       �	[�WTXc�A�0*

loss�<���       �	DQXTXc�A�0*

loss��:�b�^       �	��XTXc�A�0*

loss���<���       �	OxYTXc�A�0*

lossq�H;��*0       �	�ZTXc�A�0*

lossrn�<1�!�       �	��ZTXc�A�0*

loss�^";9)��       �	�;[TXc�A�0*

loss�9cZ�       �	��[TXc�A�0*

loss���<I��       �	��\TXc�A�0*

loss%�.;�8N       �	�~]TXc�A�0*

loss��<�SXz       �	a^TXc�A�0*

loss�G;$֋%       �	 �^TXc�A�0*

loss*��;B�L�       �	�H_TXc�A�0*

loss�'=�^�       �	��_TXc�A�0*

lossO-=�gt5       �	�t`TXc�A�0*

lossŊE9����       �	�aTXc�A�0*

loss�x9�t�       �	��aTXc�A�0*

loss��;oTei       �	�RbTXc�A�0*

loss_z�:��#       �	)�bTXc�A�0*

lossR0A;�Α�       �		�cTXc�A�0*

loss��q;�pN       �	�.dTXc�A�0*

lossn�z=y��       �	��dTXc�A�0*

loss�;c�{f       �	�VeTXc�A�0*

loss��C;wӘ�       �	�eTXc�A�0*

lossfQ9���       �	�fTXc�A�0*

losszF�;uA3�       �	�gTXc�A�0*

loss���<��l�       �	÷gTXc�A�0*

loss,�a:�
K�       �	f�hTXc�A�0*

loss�Y�<0��       �	�CiTXc�A�0*

loss �<-��S       �	��iTXc�A�0*

loss��=�x}s       �	@ljTXc�A�0*

loss
�P==|��       �	�kTXc�A�0*

loss�m;<Ur�       �	̲kTXc�A�0*

loss{M;X'�       �	FlTXc�A�1*

loss��1;]~cB       �	��lTXc�A�1*

loss~b�:��*       �	�mmTXc�A�1*

loss2P<���       �	�nTXc�A�1*

lossh�)=�˿�       �	��nTXc�A�1*

loss(;��'"       �	KXoTXc�A�1*

loss�9X�j       �	��oTXc�A�1*

loss���<Ko��       �	(�pTXc�A�1*

loss�]�<I��q       �	;qTXc�A�1*

loss�1�;�[.       �	w�qTXc�A�1*

loss�9��_w       �	<�rTXc�A�1*

loss��;=9E�a       �	g�sTXc�A�1*

lossx2;_��       �	<�tTXc�A�1*

lossd��=.��       �	��uTXc�A�1*

loss���:��       �	�1vTXc�A�1*

loss��
=|$       �	�wTXc�A�1*

loss�&>:�z�       �	��wTXc�A�1*

loss@	�:���Y       �	�[xTXc�A�1*

loss)�:��Ģ       �	��xTXc�A�1*

loss�G<1�I�       �	Y2zTXc�A�1*

loss
�g;�NW�       �	w�zTXc�A�1*

loss!��:���       �	��{TXc�A�1*

loss��;����       �	l�|TXc�A�1*

loss��:�R�       �	�?}TXc�A�1*

loss��6=p�Z       �	^�}TXc�A�1*

loss.=g�       �	�~TXc�A�1*

loss���=~�y       �	�^TXc�A�1*

loss�"=nFC       �	1��TXc�A�1*

loss�M�; �*w       �	.�TXc�A�1*

loss���9�GY�       �	1�TXc�A�1*

loss���;�e�       �	�҂TXc�A�1*

lossEl'9��1v       �	㩃TXc�A�1*

loss���;/&�       �	�Z�TXc�A�1*

loss�x;Kw-       �	�TXc�A�1*

loss�.<~,�       �	 ��TXc�A�1*

loss��;ؑ��       �	a4�TXc�A�1*

loss�1�;����       �	��TXc�A�1*

lossџ�:d��l       �	^��TXc�A�1*

lossU�=p�n�       �	7�TXc�A�1*

lossL:��-�       �	ڮ�TXc�A�1*

lossϬ,<iC��       �	�A�TXc�A�1*

lossq4�;�5�K       �	։TXc�A�1*

lossN"A;q|}z       �	Á�TXc�A�1*

loss��9�n�C       �	��TXc�A�1*

loss�wB;�D�       �	���TXc�A�1*

lossWd_:_�P�       �	�>�TXc�A�1*

loss�<��3`       �	�όTXc�A�1*

loss�p="u       �	�b�TXc�A�1*

loss���:�&"�       �	���TXc�A�1*

loss��;"tv       �	*��TXc�A�1*

lossI��<sH�       �	:�TXc�A�1*

loss�i	<�_a�       �	J��TXc�A�1*

loss��;Ź�       �	jK�TXc�A�1*

loss}ӝ9G��       �	��TXc�A�1*

loss,�<��       �	���TXc�A�1*

loss�f;����       �	��TXc�A�1*

loss�&�:��C       �	��TXc�A�1*

loss���9^�E!       �	9�TXc�A�1*

loss(Њ=���;       �	�̓TXc�A�1*

lossH&:�՘�       �	R`�TXc�A�1*

loss!��<��2�       �	���TXc�A�1*

loss��W:[�?       �	Ή�TXc�A�1*

loss���8��u       �	+�TXc�A�1*

loss���;Z���       �	ü�TXc�A�1*

loss+��<U�.       �	���TXc�A�1*

loss��y;w9��       �	��TXc�A�1*

loss��
=b��}       �	� �TXc�A�1*

loss��;�KA�       �	��TXc�A�1*

loss�-/:���       �	LS�TXc�A�1*

loss�}8\w�       �	��TXc�A�1*

lossj�;.qE       �	k�TXc�A�1*

loss�l�<���:       �	>��TXc�A�1*

loss���;ó�       �	3��TXc�A�1*

loss���=�U`�       �	L�TXc�A�1*

loss!�O=���       �	BA�TXc�A�1*

lossP�;ړ��       �	�ԟTXc�A�1*

loss3�;�U��       �	j�TXc�A�1*

loss>4<)R?�       �	v�TXc�A�1*

loss��;=iC�V       �	5��TXc�A�1*

loss��h;>�&~       �	�J�TXc�A�1*

loss�"�96���       �	�ߢTXc�A�1*

loss��:�C       �	w�TXc�A�1*

lossE��:3T�;       �	��TXc�A�1*

lossn��<P�S�       �	���TXc�A�1*

loss�(�8�|h�       �	�F�TXc�A�1*

loss��{;���#       �	O�TXc�A�1*

loss�ۼ;��=�       �	D��TXc�A�1*

lossiS�:T���       �	�&�TXc�A�1*

loss��;�j��       �	|ѧTXc�A�1*

loss�mj;��{       �	-w�TXc�A�1*

loss��;:��       �	��TXc�A�1*

lossG_�8<8[�       �	Ӿ�TXc�A�1*

loss���<��	       �	l_�TXc�A�1*

loss��(=3~�v       �	�TXc�A�1*

loss��;�Gh       �	���TXc�A�1*

loss�93=�5�       �	z6�TXc�A�1*

lossK:x}{       �	C�TXc�A�1*

loss
]�:�OH       �	���TXc�A�1*

loss|�v;e.b       �	��TXc�A�1*

loss�;�;��%<       �	f��TXc�A�1*

loss��@=(       �	KY�TXc�A�1*

lossQ;��0       �	9�TXc�A�1*

loss|ܭ9���:       �	U��TXc�A�1*

loss�w<��W       �	q�TXc�A�1*

lossǷ=S�C       �	���TXc�A�1*

loss(�9>\       �	J�TXc�A�1*

loss�<K;�^       �	���TXc�A�1*

lossQ�.;�]�       �	+��TXc�A�1*

lossst�;~�´       �	B!�TXc�A�1*

loss��;�2Ө       �	|��TXc�A�1*

loss��
=r��p       �	eR�TXc�A�1*

loss=�C<t��4       �	#��TXc�A�1*

loss=�1<�N�l       �	z��TXc�A�1*

lossM��;�L��       �	l%�TXc�A�1*

loss��c=�kFW       �	>�TXc�A�1*

loss�[�<���       �	]ܸTXc�A�1*

loss	�;�s�       �	Po�TXc�A�1*

loss�l[<Ĩ�       �	:�TXc�A�1*

lossS��:�4�       �	=��TXc�A�1*

loss���;��$P       �	�5�TXc�A�1*

loss��;��6H       �	�ϻTXc�A�1*

loss�=<�o��       �	=c�TXc�A�1*

loss�U=x�p�       �	���TXc�A�1*

loss<y�9����       �	y��TXc�A�1*

lossi6�9�i>       �	Ț�TXc�A�1*

loss ��9]z��       �	�Q�TXc�A�1*

loss�$<g"9       �	'��TXc�A�1*

loss�>�;���       �	���TXc�A�1*

loss`W:n�Q�       �	�=�TXc�A�2*

loss���:x%       �	���TXc�A�2*

lossC�d<66N"       �	ߨ�TXc�A�2*

loss�ID=y��s       �	�V�TXc�A�2*

lossQ�="��       �	6�TXc�A�2*

loss'.=+��       �	^��TXc�A�2*

loss�O-=+��V       �	�A�TXc�A�2*

loss#�P<s�i%       �	
��TXc�A�2*

loss$��<�yLu       �	��TXc�A�2*

lossF�!;����       �	���TXc�A�2*

loss�@�<9��       �	PQ�TXc�A�2*

loss�%}=�       �	O��TXc�A�2*

loss���=�1۰       �	Q��TXc�A�2*

lossI'�9'�       �	���TXc�A�2*

loss��4=x��Z       �	�D�TXc�A�2*

loss��<I �2       �	���TXc�A�2*

loss�4M9�E��       �	�w�TXc�A�2*

loss�sU<�y^�       �	��TXc�A�2*

loss�+�=�3`       �	/��TXc�A�2*

loss:8<�s�F       �	�D�TXc�A�2*

lossF[<f��[       �	0��TXc�A�2*

lossT��<�w�       �	�z�TXc�A�2*

lossD-=c�        �	��TXc�A�2*

loss{�;Wf1�       �	ʦ�TXc�A�2*

loss��<�!��       �	_F�TXc�A�2*

loss�-<i�B�       �	���TXc�A�2*

loss�a;ʗ��       �	t�TXc�A�2*

loss���;���s       �	��TXc�A�2*

loss<af;���       �	j��TXc�A�2*

loss�'�;��	�       �	6�TXc�A�2*

loss( n;t+�_       �	���TXc�A�2*

lossf��<��~       �	,b�TXc�A�2*

loss�!;��y       �	}�TXc�A�2*

losstJ9=�>l~       �	8��TXc�A�2*

loss�d�;_���       �	�<�TXc�A�2*

lossJ<x%�}       �	0��TXc�A�2*

losso�<�P��       �	uu�TXc�A�2*

loss	.�:��S�       �	��TXc�A�2*

lossMG5<��f�       �	��TXc�A�2*

loss�7�;^6�       �	1`�TXc�A�2*

loss�V?;���       �	��TXc�A�2*

loss��<_���       �	^��TXc�A�2*

lossث<��9       �	u;�TXc�A�2*

loss���<e�~�       �	���TXc�A�2*

loss�;����       �	��TXc�A�2*

loss)�~;�Q*}       �	���TXc�A�2*

loss
kD;&cV�       �	;��TXc�A�2*

loss=Տ�       �	�~�TXc�A�2*

loss���<�6�       �	�%�TXc�A�2*

loss� ;<��c       �	���TXc�A�2*

loss���;��}8       �	Ll�TXc�A�2*

loss��;�e&       �	��TXc�A�2*

loss�V�;hE       �	g��TXc�A�2*

loss��;QV�       �	�3�TXc�A�2*

loss��=:c��       �	>��TXc�A�2*

loss���<e8       �	 c�TXc�A�2*

loss"H�<���       �	n��TXc�A�2*

loss\C<[���       �	J��TXc�A�2*

loss|f�<~Wis       �	�3�TXc�A�2*

loss�.=�qY�       �	���TXc�A�2*

loss�r�;���       �	s�TXc�A�2*

loss�%�:��0       �	
�TXc�A�2*

loss3ߨ9���*       �	���TXc�A�2*

loss��:��ٙ       �	�M�TXc�A�2*

loss�&> �x�       �	a��TXc�A�2*

loss_8=;�(�[       �	�x�TXc�A�2*

lossj`"=��ԋ       �	A�TXc�A�2*

loss�\�<��P       �	��TXc�A�2*

loss��
:���2       �	Y1�TXc�A�2*

loss4g;��3�       �	���TXc�A�2*

lossj�(:,j�       �	�`�TXc�A�2*

loss$��<�b�^       �	
��TXc�A�2*

loss�t�:���       �	~��TXc�A�2*

loss.�x:�'�       �	��TXc�A�2*

loss4��=�QҾ       �	5��TXc�A�2*

loss@�9�%(	       �	�L�TXc�A�2*

loss��T<3*B�       �	���TXc�A�2*

lossF_:w�[+       �	�t�TXc�A�2*

loss�<B��       �	��TXc�A�2*

loss-�k9�%%�       �	���TXc�A�2*

losshA=M;��       �	D0�TXc�A�2*

loss!d;��       �	���TXc�A�2*

loss���:�g�       �	�`�TXc�A�2*

loss�.E:��       �	v��TXc�A�2*

loss�8�͝       �	���TXc�A�2*

loss���;���C       �	E/�TXc�A�2*

loss�q_<'B�       �	��TXc�A�2*

loss@�;�&�       �	�d�TXc�A�2*

lossX��=�<>�       �	F�TXc�A�2*

lossȆ�:)�K+       �	���TXc�A�2*

loss-��;��:R       �	Ԃ�TXc�A�2*

loss��<�bB�       �	C�TXc�A�2*

lossl�I:���6       �	�Y�TXc�A�2*

losstk�8��a       �	bi�TXc�A�2*

loss��:9�       �	28 UXc�A�2*

lossH�<U;�       �	=� UXc�A�2*

loss���;��@�       �	DnUXc�A�2*

loss�n�;�gV       �	q UXc�A�2*

losss*59�K��       �	ՒUXc�A�2*

loss�eo<ub+       �	�'UXc�A�2*

lossX8+=��M       �	(�UXc�A�2*

loss��~;��2�       �	NUXc�A�2*

loss6�X9:�_O       �	G�UXc�A�2*

lossx֞<����       �	[{UXc�A�2*

loss���<C�       �	�UXc�A�2*

loss���<��`�       �	įUXc�A�2*

loss��:�	�       �	eUUXc�A�2*

loss�|�:��        �	��UXc�A�2*

loss�;�
ۮ       �	&�UXc�A�2*

loss�/ :�ږ0       �	�-	UXc�A�2*

loss�R<�&W'       �	��	UXc�A�2*

lossoE;�7       �	x`
UXc�A�2*

loss�G/;�A��       �	��
UXc�A�2*

loss�5�:��ߩ       �	S�UXc�A�2*

lossEd�;���       �	b.UXc�A�2*

lossh��=��       �	��UXc�A�2*

loss�� ;�W��       �	UlUXc�A�2*

loss���<w;       �	|UXc�A�2*

loss!��8b�       �	�UXc�A�2*

loss���<9��       �	FBUXc�A�2*

lossIT<�R�       �	�UXc�A�2*

lossEK�9��z�       �	�eUXc�A�2*

loss�*=7       �	U�UXc�A�2*

lossse�:	O�       �	l�UXc�A�2*

loss-ɹ:(?b       �	f/UXc�A�2*

loss�
q<����       �	+�UXc�A�2*

lossx�;�_�       �	�YUXc�A�2*

loss�K;}v�        �	��UXc�A�2*

loss��`;�Q�Y       �	ڐUXc�A�3*

lossW#=�[�       �	l%UXc�A�3*

lossj'<<"��       �	��UXc�A�3*

loss���9����       �	�WUXc�A�3*

loss�\0<8#��       �	��UXc�A�3*

loss^�9ڼ$E       �	j�UXc�A�3*

loss�k�6"��       �	h UXc�A�3*

loss� ;��#       �	߿UXc�A�3*

lossǂ9b��v       �	W`UXc�A�3*

lossQs�:��o�       �	�UXc�A�3*

loss%	�;�u�       �	_�UXc�A�3*

loss<�;0�x�       �	80UXc�A�3*

loss�,�<f��8       �	��UXc�A�3*

lossq�t9��M<       �	�`UXc�A�3*

lossj��8f���       �	�UXc�A�3*

loss݃�9��       �	g�UXc�A�3*

loss�g�:��	       �	��UXc�A�3*

loss.Oz<�&l�       �	p# UXc�A�3*

lossߣ�<�S��       �	]� UXc�A�3*

loss�:Z���       �	a!UXc�A�3*

loss��?;6��       �	��!UXc�A�3*

loss�	�<rFS�       �	ط"UXc�A�3*

loss̇9J;%       �	4J#UXc�A�3*

loss�\>I[�S       �	e�#UXc�A�3*

loss��=�h(       �	�y$UXc�A�3*

lossj͝;���8       �	{%UXc�A�3*

loss,w�;:hs�       �	��%UXc�A�3*

lossP|<��~       �	�:&UXc�A�3*

loss�ֿ;b���       �	��&UXc�A�3*

lossؠ�:����       �	jg'UXc�A�3*

loss ]A;�EK~       �	��'UXc�A�3*

lossM;�1hN       �	X�(UXc�A�3*

loss4�t<�w]6       �	�)UXc�A�3*

lossX�g:��2       �	��)UXc�A�3*

loss�|y=�r��       �	DP*UXc�A�3*

loss�I�<"�i�       �	��*UXc�A�3*

loss��==�~��       �	��+UXc�A�3*

lossfTK;��;!       �	�,UXc�A�3*

loss��:z���       �	��,UXc�A�3*

lossʦ�<��C�       �	pC-UXc�A�3*

loss|��<F_@C       �	��-UXc�A�3*

loss�-H:"���       �	�p.UXc�A�3*

loss��I<m��u       �	,/UXc�A�3*

lossz|�;��ZK       �	#�/UXc�A�3*

loss���<.��       �	%<0UXc�A�3*

lossƲ;_�܀       �	b�0UXc�A�3*

loss�T<n�î       �	��1UXc�A�3*

lossV<S:_M�       �	_&2UXc�A�3*

loss:=�+�       �	��2UXc�A�3*

loss@3n;�c^�       �	y]3UXc�A�3*

loss*I?;eB�       �	a�3UXc�A�3*

lossi�;3��       �	ޏ4UXc�A�3*

lossC�/;��Mq       �		6UXc�A�3*

loss�q<����       �	p�6UXc�A�3*

lossE�e:/(~�       �	i�7UXc�A�3*

loss�8�8�(�>       �	s08UXc�A�3*

loss(�;9[ud       �	��8UXc�A�3*

loss��<�C�       �	�z9UXc�A�3*

loss�:;!���       �	�:UXc�A�3*

loss��;�_�(       �	4�:UXc�A�3*

loss
%=W �       �	yY;UXc�A�3*

lossj��<x��       �	�;UXc�A�3*

loss�e\;�}��       �	��<UXc�A�3*

loss!�:��&       �	.=UXc�A�3*

lossD��;����       �	��=UXc�A�3*

loss#��;N�o>       �	Rb>UXc�A�3*

loss?�i:�W�       �	��>UXc�A�3*

lossE�<��iM       �	�?UXc�A�3*

loss��<p�a�       �	X<@UXc�A�3*

loss�r�;���       �	��@UXc�A�3*

loss��: .�e       �	RAUXc�A�3*

lossH�-:�(+       �	I*BUXc�A�3*

lossA0';|��       �	��BUXc�A�3*

loss�'N;�9��       �	�"\UXc�A�3*

lossV�;��       �	մ\UXc�A�3*

loss��,<�!X�       �	 S]UXc�A�3*

loss�V�:x�TO       �	:�]UXc�A�3*

lossŇ�<�3v{       �	{^UXc�A�3*

loss:3;��N       �	,_UXc�A�3*

lossq��<�-ه       �	��_UXc�A�3*

loss�b=^#\s       �	C=`UXc�A�3*

loss&' =��R�       �	p�`UXc�A�3*

loss�V�<w��G       �	kdaUXc�A�3*

lossq7�;L�"       �	��aUXc�A�3*

loss4�=��Y       �	}�bUXc�A�3*

loss�o@<�ߖ       �	:cUXc�A�3*

loss�0�:19�m       �	��cUXc�A�3*

loss ¦<M�d�       �	�bdUXc�A�3*

lossa<qǚ3       �	��dUXc�A�3*

loss׾W9��`       �	7�eUXc�A�3*

loss��<4W�       �	K!fUXc�A�3*

lossi�o;3܊A       �	9�fUXc�A�3*

lossXD=r�:       �	�`gUXc�A�3*

loss���;��q       �	qhUXc�A�3*

loss�ش;�p�       �	�hUXc�A�3*

loss�f;��Ž       �	�=iUXc�A�3*

loss<��=���       �	��iUXc�A�3*

loss�Q�;��ڏ       �	1_jUXc�A�3*

loss�':`���       �	��jUXc�A�3*

lossw�9���       �	"�kUXc�A�3*

lossm�;��N�       �	�lUXc�A�3*

loss�+�;HuY"       �	�lUXc�A�3*

loss�d�9�8       �	�RmUXc�A�3*

loss��;Gў�       �	��mUXc�A�3*

loss�h<6��3       �		�nUXc�A�3*

loss[�,;Q���       �	6 oUXc�A�3*

lossٰ�;��j�       �	��oUXc�A�3*

loss���::�<O       �	�QpUXc�A�3*

loss�W�;O�       �	l�pUXc�A�3*

loss�I�7�Y��       �	a�qUXc�A�3*

lossf=� �       �	�.rUXc�A�3*

loss�R
<'�r�       �	��rUXc�A�3*

loss/=�       �	1asUXc�A�3*

lossύ�;���7       �	��sUXc�A�3*

loss�=�W�u       �	סtUXc�A�3*

loss-��:4I`�       �	�8uUXc�A�3*

lossnS@;��(        �	>�uUXc�A�3*

loss�M�:���W       �	 `vUXc�A�3*

lossK��;���       �	��vUXc�A�3*

loss��;�HH       �	��wUXc�A�3*

loss|N2;�r$       �	�%xUXc�A�3*

loss=D�9��.�       �	j�xUXc�A�3*

loss�x:��]�       �	RyUXc�A�3*

lossS?;ѧ-       �	��yUXc�A�3*

loss�K�=,@E�       �	U�zUXc�A�3*

loss_�<^)d       �	.{UXc�A�3*

loss���<@N"�       �	�{UXc�A�3*

loss;�:����       �	DR|UXc�A�3*

loss��;L{֊       �	�|UXc�A�4*

lossf|:���       �	o}UXc�A�4*

lossj?�9�&=�       �	F"~UXc�A�4*

loss���=O[        �	b�~UXc�A�4*

loss�#=�'I       �	�XUXc�A�4*

loss�G=���C       �	��UXc�A�4*

lossh$;!���       �	c~�UXc�A�4*

lossJ�:����       �	��UXc�A�4*

loss+7;�ьA       �	���UXc�A�4*

lossh�:�C��       �	6Z�UXc�A�4*

loss�a�9ؤy�       �	���UXc�A�4*

loss�.�;/*�U       �	��UXc�A�4*

loss�
<����       �	�6�UXc�A�4*

loss�=,<'���       �	ՄUXc�A�4*

loss!-<��       �	Hn�UXc�A�4*

loss��j<��	       �	-�UXc�A�4*

loss�g;[�N       �	̙�UXc�A�4*

loss�O�<n7�H       �	�,�UXc�A�4*

loss�	�:ɯ�       �	�ɇUXc�A�4*

loss�;�<� ��       �	>^�UXc�A�4*

loss��=<���V       �	��UXc�A�4*

lossud:�z�       �	Ή�UXc�A�4*

loss��;j+�X       �	q�UXc�A�4*

loss)��<����       �	g��UXc�A�4*

lossl:A.u       �	,J�UXc�A�4*

loss�;�KIW       �	8ۋUXc�A�4*

lossiIW;�b       �	�z�UXc�A�4*

loss�;�U�Y       �	��UXc�A�4*

loss��6�ʸ       �	-��UXc�A�4*

loss��j;��"       �	F�UXc�A�4*

lossԵ9���       �	ZَUXc�A�4*

lossqכ;�n       �	 {�UXc�A�4*

loss86;���       �	��UXc�A�4*

loss��:��       �	흐UXc�A�4*

loss��6=P�Sq       �	1�UXc�A�4*

loss�[�;۟ƚ       �	3đUXc�A�4*

loss�/�<6��       �	�_�UXc�A�4*

lossx��;ӗWT       �	��UXc�A�4*

lossZ��;IN"/       �	8��UXc�A�4*

loss��h:��>|       �	��UXc�A�4*

loss�J�:;>�Q       �	ɫ�UXc�A�4*

loss���<F 1�       �	�@�UXc�A�4*

lossd�[; ��       �	�ڕUXc�A�4*

loss��;ɯa       �	Uk�UXc�A�4*

loss\� ;+�w       �	m�UXc�A�4*

loss �:� `�       �	���UXc�A�4*

loss1F8�֐       �	()�UXc�A�4*

loss�r�9�X�       �	j��UXc�A�4*

lossj:�;���	       �	�l�UXc�A�4*

lossvP<����       �	�`�UXc�A�4*

loss_��75ݣ�       �	9*�UXc�A�4*

loss�H�;��a�       �	ʛUXc�A�4*

loss	eF:;�z       �	�i�UXc�A�4*

loss��a=�P�       �	K�UXc�A�4*

loss� ;&��       �	�ȝUXc�A�4*

loss)�<�       �	>]�UXc�A�4*

loss�r=,�s       �	⑟UXc�A�4*

lossm�=i��       �	�0�UXc�A�4*

loss��9{�6       �	�ĠUXc�A�4*

loss}��;�=t       �	���UXc�A�4*

loss���<�<�`       �	�,�UXc�A�4*

loss?�9�y�       �	��UXc�A�4*

loss�C�8A�?�       �	���UXc�A�4*

losse��<P���       �	�"�UXc�A�4*

lossφ�=����       �	�ǤUXc�A�4*

loss���:~:2�       �	�b�UXc�A�4*

loss�
P<�+       �	t�UXc�A�4*

loss� �9>���       �	O��UXc�A�4*

loss��3<O�ax       �	RG�UXc�A�4*

loss�
;2��       �	*�UXc�A�4*

lossN�T=��       �	%z�UXc�A�4*

loss�K<vP��       �	��UXc�A�4*

loss�s�:ĢT       �	K��UXc�A�4*

loss��<BT�P       �	�?�UXc�A�4*

loss�	�9��i�       �	�ӪUXc�A�4*

loss-6�8y
f�       �	�n�UXc�A�4*

loss�oH;� �       �	�	�UXc�A�4*

loss]�:���0       �	;��UXc�A�4*

lossn_�:��z       �	�=�UXc�A�4*

lossJ�Z;���A       �	��UXc�A�4*

loss�e�;;�]�       �	���UXc�A�4*

loss�I;SopQ       �	 �UXc�A�4*

loss�0�=l��       �	���UXc�A�4*

loss��:�'R�       �	Z�UXc�A�4*

loss��:m���       �	9�UXc�A�4*

loss�H�<�'��       �	 ��UXc�A�4*

loss@%K;>�        �	�*�UXc�A�4*

loss��;�Z       �	��UXc�A�4*

loss�L�;�i�       �	rS�UXc�A�4*

loss٣=f��#       �	��UXc�A�4*

loss�AN<�� �       �	Su�UXc�A�4*

lossagD:�xdP       �	�
�UXc�A�4*

lossL�:I���       �	
��UXc�A�4*

loss��8���)       �	�7�UXc�A�4*

loss�k;�U��       �	�UXc�A�4*

loss�y:��=�       �	���UXc�A�4*

loss2Z=�uL?       �	w/�UXc�A�4*

loss�)=�� _       �	y��UXc�A�4*

lossJD�;�jR       �	�@�UXc�A�4*

loss��x:Z�24       �	�ٺUXc�A�4*

loss.��:�1o�       �	�z�UXc�A�4*

loss���;�փ�       �	��UXc�A�4*

loss;�;r4>�       �	l��UXc�A�4*

lossâ�<8Cm�       �	GU�UXc�A�4*

loss��;��       �	?�UXc�A�4*

loss�`�;��R       �	g��UXc�A�4*

loss]�d=�ٟ
       �	��UXc�A�4*

loss��<`#��       �	���UXc�A�4*

loss	�|9��       �	�U�UXc�A�4*

lossf��=Vej�       �	���UXc�A�4*

loss���9yc�       �	8��UXc�A�4*

loss�9"95!��       �	)!�UXc�A�4*

loss��=:BK2�       �	0��UXc�A�4*

loss�X�;��p       �	�b�UXc�A�4*

loss,yU:���       �	;��UXc�A�4*

lossz��=�f       �	-��UXc�A�4*

lossQJu:B�<�       �	�)�UXc�A�4*

lossa�A<S��$       �	��UXc�A�4*

loss��|;.Sb#       �	�P�UXc�A�4*

loss��>9��/�       �	q��UXc�A�4*

loss��M<w���       �	���UXc�A�4*

loss�+�;����       �	k,�UXc�A�4*

loss��;	0q�       �	M��UXc�A�4*

loss�e�:�(       �	uU�UXc�A�4*

lossc[�<皰�       �	���UXc�A�4*

lossK��<zOM�       �	P��UXc�A�4*

loss1�=/�=       �	�$�UXc�A�4*

loss#_�:l��G       �	{��UXc�A�4*

lossm�89g�       �	���UXc�A�5*

loss!��;��       �	�!�UXc�A�5*

loss_u{:��       �	�UXc�A�5*

loss��<m�[�       �	���UXc�A�5*

loss�}�9<$�s       �	&V�UXc�A�5*

lossw�;	,�H       �	���UXc�A�5*

loss!4�:��i       �	��UXc�A�5*

loss�S�=��%       �	
,�UXc�A�5*

loss���;:��       �	��UXc�A�5*

loss2!=��5       �	�S�UXc�A�5*

loss��s:�$�       �	@��UXc�A�5*

loss�c�8�tʞ       �	���UXc�A�5*

loss�!�;8�Ў       �	m6�UXc�A�5*

loss�Ox;E@�       �	���UXc�A�5*

lossOf�:��9       �	�n�UXc�A�5*

loss�ݻ;f���       �	��UXc�A�5*

loss�bJ8����       �	��UXc�A�5*

loss?�;�1Î       �	L6�UXc�A�5*

lossq�=�d��       �	��UXc�A�5*

loss���:w͢r       �	�f�UXc�A�5*

loss��I<xs��       �	���UXc�A�5*

lossf��:�u(�       �	��UXc�A�5*

lossOɑ=�S��       �	�1�UXc�A�5*

loss@W<��<�       �	i��UXc�A�5*

lossƉ:��@�       �	e�UXc�A�5*

loss���;�i_�       �	��UXc�A�5*

loss�nf:6XU       �	��UXc�A�5*

lossOsV;�r{�       �	�T�UXc�A�5*

lossX�<O/       �	@��UXc�A�5*

loss��9^"�c       �	c'�UXc�A�5*

loss!��;����       �	�c�UXc�A�5*

loss�C�:�fb       �	���UXc�A�5*

lossM��<}}�4       �	���UXc�A�5*

loss�[�<]=U$       �	*V�UXc�A�5*

loss�,==�
g:       �	R��UXc�A�5*

loss-�	<�3��       �	'�UXc�A�5*

loss$�-<�^,�       �	���UXc�A�5*

loss��Y;a�|       �	n��UXc�A�5*

loss.'@=�xI�       �	�u�UXc�A�5*

lossIQ�<ȓ��       �	�UXc�A�5*

loss��<���,       �	���UXc�A�5*

loss��':�$A       �	�G�UXc�A�5*

lossJ�;��i�       �	���UXc�A�5*

loss�2�<񍇗       �	���UXc�A�5*

loss��;�@9�       �	5c�UXc�A�5*

loss�Q7<���       �	��UXc�A�5*

lossRT�:�wd       �	Y��UXc�A�5*

loss��<��,H       �	y@�UXc�A�5*

lossDz";�TM       �	>��UXc�A�5*

loss,�@<F��#       �	���UXc�A�5*

loss��0=×g!       �	,(�UXc�A�5*

loss��;�ۭ�       �	]��UXc�A�5*

loss�^�:��z�       �	=_�UXc�A�5*

loss�L�<Э       �	���UXc�A�5*

loss8L�;]s�       �	��UXc�A�5*

loss�Q;�/       �	R+�UXc�A�5*

loss�($;6��       �	���UXc�A�5*

loss�-<	8�       �	�_�UXc�A�5*

loss�o;�R�       �	��UXc�A�5*

loss���<9uĤ       �	��UXc�A�5*

lossZI;��9       �	>$�UXc�A�5*

loss�A<+7��       �	Ǹ�UXc�A�5*

lossS�;�%�       �	�U�UXc�A�5*

loss�':���       �	�-�UXc�A�5*

loss\��<����       �	z��UXc�A�5*

loss��=Q8��       �	�Y�UXc�A�5*

loss,��;�~[       �	t��UXc�A�5*

loss��
<:H       �	*��UXc�A�5*

loss�:fy��       �	�f�UXc�A�5*

loss��s9��       �	5�UXc�A�5*

loss噠8m���       �	��UXc�A�5*

loss��	8�t!7       �	�E�UXc�A�5*

lossd�:;�Y��       �	���UXc�A�5*

lossx
6=t�p�       �	F��UXc�A�5*

loss��;�H        �	E.�UXc�A�5*

loss�&C=ۧ�       �	/��UXc�A�5*

loss��;-���       �	w VXc�A�5*

loss�;pT�       �	�VXc�A�5*

lossָ9xu�S       �	�VXc�A�5*

loss$�H;�-��       �	��VXc�A�5*

loss9�;�HJ       �	��VXc�A�5*

lossj�:��.�       �	�FVXc�A�5*

loss2�7���>       �	��VXc�A�5*

loss_��;t��u       �	��VXc�A�5*

loss��=5 �T       �	�VXc�A�5*

loss��<1��       �	+�VXc�A�5*

loss��;� |A       �	c_VXc�A�5*

loss%��:@�8       �	��VXc�A�5*

loss��
<Bqqu       �	��VXc�A�5*

lossR�5<ȵ��       �	L�	VXc�A�5*

loss͋M==)3*       �	(
VXc�A�5*

loss���;�`�q       �	�
VXc�A�5*

loss��<��D�       �	k`VXc�A�5*

loss �8=.�       �	z�VXc�A�5*

loss�;�]S�       �	-�VXc�A�5*

loss�j&:����       �	8VXc�A�5*

loss]p�;���7       �	p�VXc�A�5*

loss�Q�:�LC~       �	�VXc�A�5*

lossc�h;qz|       �	 �VXc�A�5*

loss� <���       �	��VXc�A�5*

lossO� ==�       �	a6VXc�A�5*

lossݗ:�G��       �	W�VXc�A�5*

loss�;�s@       �	pVXc�A�5*

loss뺋98�{C       �	�VXc�A�5*

lossZ/=F	Z�       �	�VXc�A�5*

loss�X:���7       �	`ZVXc�A�5*

loss�F{:v�/       �	��VXc�A�5*

lossL�7ν       �	��VXc�A�5*

lossv��9?_f|       �	5$VXc�A�5*

loss��p<x��S       �	��VXc�A�5*

loss?E�:�� \       �	KuVXc�A�5*

loss���:�ݏ9       �	�VXc�A�5*

loss�^�;%�;�       �	.�VXc�A�5*

lossq˺:��?�       �	�NVXc�A�5*

loss��;!�<       �	X�VXc�A�5*

loss�;{cҋ       �	��VXc�A�5*

loss�Ĭ;�Q�q       �	K=VXc�A�5*

loss�� <�c       �	��VXc�A�5*

loss�{=�D]�       �	0VXc�A�5*

loss��<:1ZV       �	��VXc�A�5*

loss�6<C�m�       �	A�VXc�A�5*

lossϔ�: }Q�       �	!v VXc�A�5*

loss׶=dK��       �	�!VXc�A�5*

loss/M�9L�1�       �	l�!VXc�A�5*

loss�&�<K�       �	�"VXc�A�5*

loss"�:1�       �	x'#VXc�A�5*

lossn42:Oi�       �	��#VXc�A�5*

loss�W!;��0       �	�f$VXc�A�5*

lossF�:����       �	��$VXc�A�6*

lossν�;�i��       �	�%VXc�A�6*

loss?�:��ۆ       �	�7&VXc�A�6*

lossF�;�e�       �	F�&VXc�A�6*

lossO�"=�s�       �	�g'VXc�A�6*

loss�J<�߃)       �	\W(VXc�A�6*

loss��;9���       �	��(VXc�A�6*

loss��<��c       �	z�)VXc�A�6*

lossO#!=�[�       �	l#*VXc�A�6*

loss�A�;�N�       �	��*VXc�A�6*

loss�;=	��       �	�K+VXc�A�6*

loss��<�)�[       �	�+VXc�A�6*

lossO�:|!k       �	�,VXc�A�6*

loss�C};�j�r       �	<.-VXc�A�6*

lossa�<U,�Y       �	��-VXc�A�6*

lossa�2;����       �	�Z.VXc�A�6*

loss`��;&��       �	�.VXc�A�6*

loss��<>���       �	>�/VXc�A�6*

loss�ae=U�,       �	
,0VXc�A�6*

loss ;��       �		�0VXc�A�6*

loss�È<��>       �	CT1VXc�A�6*

loss�g�;|)0       �	��1VXc�A�6*

lossT�:FB�       �	Y�2VXc�A�6*

loss)H�;�a"�       �	�3VXc�A�6*

loss#7u��       �	�3VXc�A�6*

loss��<�2       �	�N4VXc�A�6*

loss��<�Fc%       �	>�4VXc�A�6*

lossυ=͚p*       �	ǀ5VXc�A�6*

lossS� <�r�       �	6VXc�A�6*

lossC)>;� �S       �	��6VXc�A�6*

loss��18��)S       �	�G7VXc�A�6*

loss��
:��>�       �	l8VXc�A�6*

loss�d�;�n�z       �	��8VXc�A�6*

lossv��=����       �	'.9VXc�A�6*

loss!�>[�H�       �	��9VXc�A�6*

loss�Q:�se�       �	�U:VXc�A�6*

loss��g:H1�       �	;VXc�A�6*

loss��;����       �	�J<VXc�A�6*

loss���=v4�-       �	��<VXc�A�6*

lossh9 7�       �	��=VXc�A�6*

loss�o
;S4�U       �	8M>VXc�A�6*

loss��8=�$�}       �	��?VXc�A�6*

lossFd/<��n       �	W#@VXc�A�6*

lossd�F=�;L�       �	T�@VXc�A�6*

loss�q�:�@n       �	tzAVXc�A�6*

loss{o�;�N�       �	�BVXc�A�6*

loss�I<D�~       �	GCVXc�A�6*

lossܸ=��       �	d�CVXc�A�6*

lossH��=�p{�       �	B�DVXc�A�6*

loss��<֚��       �	hEVXc�A�6*

loss��;Q��       �	��EVXc�A�6*

loss�H�:�       �	�FVXc�A�6*

loss��=d�o       �	�=GVXc�A�6*

loss�6C=/��       �	/�GVXc�A�6*

lossC�x=�L7�       �	huHVXc�A�6*

loss�7�;�Ӻ       �	FDIVXc�A�6*

losse�y<0��       �	��IVXc�A�6*

loss��<�<ʵ       �	�mJVXc�A�6*

lossN�V;��Mb       �	* KVXc�A�6*

lossty<PَO       �	�KVXc�A�6*

loss ~z<��k       �	�/LVXc�A�6*

loss��=2$�       �	,DMVXc�A�6*

loss�o�9QT�]       �	@�MVXc�A�6*

loss�=3�˫       �	~�NVXc�A�6*

loss�*�;�=U�       �	!OVXc�A�6*

loss�g�;�<K.       �	^�OVXc�A�6*

loss�\�;���       �	]RPVXc�A�6*

loss��<�+       �	y�PVXc�A�6*

loss��e<�Ϧ�       �	0QVXc�A�6*

loss7Ƚ=���       �	�RVXc�A�6*

lossp��;���       �	�RVXc�A�6*

loss�I;=�Yx       �	oGSVXc�A�6*

lossJf/<�3��       �	��SVXc�A�6*

loss��:��t       �	��TVXc�A�6*

loss�mR<��t       �	%UVXc�A�6*

loss��<�˻�       �	c�UVXc�A�6*

loss/L<s�P�       �	.VVVXc�A�6*

loss1�<3��       �	�VVXc�A�6*

loss���9dF¨       �	(�WVXc�A�6*

lossL5<F��       �	YXVXc�A�6*

loss�p&;ڡ�C       �	��XVXc�A�6*

loss��T<��M       �	�GYVXc�A�6*

lossvo�;St5�       �	<�YVXc�A�6*

loss��<�HO       �	GwZVXc�A�6*

loss	�;��W-       �	�[VXc�A�6*

loss��):�Z�~       �	:�[VXc�A�6*

loss(+�;���       �	1]\VXc�A�6*

loss_�t;Ή5       �	��\VXc�A�6*

loss�=";&��3       �	��]VXc�A�6*

loss8o�:���
       �	�M^VXc�A�6*

loss�A�=h��       �	��^VXc�A�6*

losssr(:W��       �	��_VXc�A�6*

loss�8�9Y ZF       �	`!`VXc�A�6*

loss[N<���       �	�`VXc�A�6*

loss��{<_��       �	/NaVXc�A�6*

loss{��;92(       �	C�aVXc�A�6*

loss�3T<��aU       �	6wbVXc�A�6*

lossO�<�Vo�       �	JcVXc�A�6*

lossD��;�s�L       �	L�cVXc�A�6*

loss�Z>ۨ�L       �	l=dVXc�A�6*

lossL�H=@vʨ       �	��dVXc�A�6*

loss�8�:�(�K       �	�eVXc�A�6*

loss�
�;BV��       �	y fVXc�A�6*

loss��<�\	       �	��fVXc�A�6*

loss�]�<`�)�       �	�jgVXc�A�6*

loss"�='�I       �	�hVXc�A�6*

lossd��9�       �	8�hVXc�A�6*

loss�<r��       �	�6iVXc�A�6*

lossJ�b<K�C6       �	�iVXc�A�6*

losss�!:�֡[       �	�fjVXc�A�6*

loss�gJ=CA��       �	��jVXc�A�6*

loss�O8��TX       �	 �kVXc�A�6*

loss��b:};��       �	I�lVXc�A�6*

loss
w:��_       �	K�mVXc�A�6*

loss
�:u�C�       �	*nVXc�A�6*

loss�؂;a� 3       �	��nVXc�A�6*

loss
��<3���       �	I�oVXc�A�6*

losss4=_=       �	vpVXc�A�6*

loss,�|<r$|T       �	ޭpVXc�A�6*

loss��0<����       �	�BqVXc�A�6*

lossN;�'�       �	��qVXc�A�6*

lossC�9��%�       �	�rrVXc�A�6*

lossdv�9 w��       �	�sVXc�A�6*

loss�<6�       �	�sVXc�A�6*

loss��;/-.g       �	37tVXc�A�6*

lossC[�9���l       �	��uVXc�A�6*

loss60�;F$v       �	�vVXc�A�6*

loss�<�d�       �	��vVXc�A�6*

lossc�:FP�       �	&qwVXc�A�7*

loss���;����       �	�xVXc�A�7*

loss�r;,�?�       �	D�xVXc�A�7*

lossm��;0�Ih       �	ͯyVXc�A�7*

loss�+ =0�F�       �	JzVXc�A�7*

lossdJC;Np       �	i�zVXc�A�7*

loss=
�<���       �	�y{VXc�A�7*

lossfdl>����       �	9*|VXc�A�7*

loss��:E�s       �	��|VXc�A�7*

lossW��;��Q       �	�}VXc�A�7*

losso]�;��
U       �	&~VXc�A�7*

loss��<���U       �	��~VXc�A�7*

loss�m�;�X��       �	�VXc�A�7*

loss�o�9��#       �	>y�VXc�A�7*

loss�y9:�h       �	1�VXc�A�7*

loss}|�<&�T<       �	"ǁVXc�A�7*

loss%<�� �       �	be�VXc�A�7*

loss�x<�#��       �	,f�VXc�A�7*

losslwN<_��       �	Y��VXc�A�7*

loss&�6:��>^       �	犄VXc�A�7*

loss�I+;�і�       �	Q.�VXc�A�7*

loss!D*;B�&�       �	�ɅVXc�A�7*

loss4�7=��L       �	}\�VXc�A�7*

loss��?;��Oc       �	80�VXc�A�7*

lossJA8=
�C�       �	6�VXc�A�7*

loss4W<�� �       �	��VXc�A�7*

loss��;�?@�       �	�VXc�A�7*

loss��x<J$��       �	ƨ�VXc�A�7*

loss��;7�&       �	P:�VXc�A�7*

lossͥL;�|�       �	>ϊVXc�A�7*

lossH�:��       �	�a�VXc�A�7*

loss���=�^n       �	+��VXc�A�7*

lossLe�<q�       �	d��VXc�A�7*

loss��)=ߍ�       �	�+�VXc�A�7*

loss��X<G��       �	�čVXc�A�7*

loss��z<���       �	d�VXc�A�7*

loss*A[<�<��       �	��VXc�A�7*

loss֡<@֗       �	/��VXc�A�7*

loss!O�;��?�       �	=�VXc�A�7*

loss�:;{>�2       �	�ԐVXc�A�7*

loss� =𭽓       �	�l�VXc�A�7*

lossm=����       �	��VXc�A�7*

lossy��;xp]�       �	˞�VXc�A�7*

loss�A�;�O�       �	�0�VXc�A�7*

loss�{;��"       �	!̓VXc�A�7*

lossd�;]�`�       �	ձ�VXc�A�7*

lossƭ�:�36       �	9E�VXc�A�7*

loss�p�:��G�       �	#ؕVXc�A�7*

loss-<C�S       �	7m�VXc�A�7*

loss$�7;��7@       �	C��VXc�A�7*

loss��=��0       �	�ėVXc�A�7*

lossj$Q="N%�       �	.W�VXc�A�7*

loss7=3:����       �	��VXc�A�7*

loss���9�ζ4       �	�z�VXc�A�7*

loss�6:@[4�       �	�VXc�A�7*

lossZt�:���t       �	Y��VXc�A�7*

loss�(=b��       �	*:�VXc�A�7*

loss���:�k�       �	xћVXc�A�7*

lossv�\<�0<�       �	�e�VXc�A�7*

lossHz�<| ��       �	�VXc�A�7*

loss@z=����       �	��VXc�A�7*

loss#�);�Sa�       �	�+�VXc�A�7*

loss�1:�t       �	˞VXc�A�7*

loss��;���       �	B]�VXc�A�7*

loss�1�9��       �	�VXc�A�7*

loss�x8;���       �	}�VXc�A�7*

lossC��;�b�       �	���VXc�A�7*

loss��=�u_�       �	W'�VXc�A�7*

loss�Dz;�P7       �	���VXc�A�7*

loss��=�猋       �	vR�VXc�A�7*

loss�z.:�]o~       �	��VXc�A�7*

loss=;����       �	I��VXc�A�7*

losstU�9�K��       �	�!�VXc�A�7*

loss���;A�y       �	p��VXc�A�7*

loss�~z<�h��       �	��VXc�A�7*

loss�D.:^S       �	3��VXc�A�7*

lossr�;�6h�       �	F"�VXc�A�7*

loss:TP=���4       �	�ԨVXc�A�7*

lossL�C9��(�       �	�k�VXc�A�7*

loss��;�NZ�       �	� �VXc�A�7*

lossly9��)]       �	{��VXc�A�7*

loss ��<�_i�       �	2;�VXc�A�7*

lossY�9�,s�       �	9ЫVXc�A�7*

loss�;>ߵ       �	�m�VXc�A�7*

loss�o�:|�D       �	l�VXc�A�7*

loss�H�<�8�       �	g��VXc�A�7*

lossg&;�w8�       �	�=�VXc�A�7*

loss�;@H��       �	�߮VXc�A�7*

loss�u�<�\+�       �	�t�VXc�A�7*

loss�QH=���C       �	�VXc�A�7*

loss��<�&�L       �	Y��VXc�A�7*

lossV6r<���       �	dW�VXc�A�7*

loss�jp<�\�        �	W�VXc�A�7*

loss�ʛ;��~       �	{��VXc�A�7*

lossb99{s��       �	�VXc�A�7*

loss�q9~�uW       �	(��VXc�A�7*

loss�9�)�b       �	�[�VXc�A�7*

lossV�c;�&�       �	{��VXc�A�7*

loss��<�؄       �	v��VXc�A�7*

loss&��8�K6�       �	I�VXc�A�7*

lossC��:`���       �	��VXc�A�7*

loss �7�I��       �	p��VXc�A�7*

loss�6�8�R�Z       �	1�VXc�A�7*

loss:m�7��       �	CŸVXc�A�7*

lossq��;_X2`       �	GU�VXc�A�7*

lossZ>Q;/�       �	`�VXc�A�7*

loss�e�<L���       �	#��VXc�A�7*

lossl�I7$ҏ       �	�!�VXc�A�7*

lossFA9����       �	˻VXc�A�7*

loss��d=�PQ\       �	�̼VXc�A�7*

loss�_�8�o�e       �	�^�VXc�A�7*

lossv�z=R�tl       �	z��VXc�A�7*

lossUX;���       �	l��VXc�A�7*

loss�=��[:       �	.�VXc�A�7*

loss��];�[#7       �	j��VXc�A�7*

lossm}+;�mR       �	ٗ�VXc�A�7*

lossTc;z$�       �	|(�VXc�A�7*

loss�)�;���       �	���VXc�A�7*

loss�[;�lg       �	uv�VXc�A�7*

loss��s:if�T       �	�	�VXc�A�7*

lossTo;<�^�       �	̘�VXc�A�7*

loss"<���       �	�9�VXc�A�7*

lossY�=8�;N       �	��VXc�A�7*

loss��W;�u�       �	+��VXc�A�7*

lossV��; �Q       �	3��VXc�A�7*

loss6JB<����       �	Zf�VXc�A�7*

lossra< ^       �	I�VXc�A�7*

loss)�!:����       �	v��VXc�A�7*

loss�wN=�1�y       �	���VXc�A�8*

loss�<0�<�       �	U�VXc�A�8*

loss�`�;��69       �	��VXc�A�8*

loss��	<��t�       �	�L�VXc�A�8*

lossj	b;�� �       �	~��VXc�A�8*

loss:��<����       �	��VXc�A�8*

loss�}�:�܏       �		p�VXc�A�8*

loss�4:p��T       �	�VXc�A�8*

lossZ�6;s B       �	Z��VXc�A�8*

loss��^;�c��       �	�1�VXc�A�8*

loss�[<=F_~       �	q��VXc�A�8*

loss��7<O t�       �	���VXc�A�8*

loss�=,<�5��       �	i;�VXc�A�8*

lossO�:�]��       �	c��VXc�A�8*

loss�ˉ;��U�       �	wg�VXc�A�8*

loss�-�;���D       �	���VXc�A�8*

loss��;����       �	2��VXc�A�8*

loss�;���=       �	à�VXc�A�8*

loss]5<P+��       �	6;�VXc�A�8*

loss��$<y��       �	���VXc�A�8*

lossR� <��<       �	���VXc�A�8*

lossʔ�<9��)       �	 z�VXc�A�8*

loss�G <��W�       �	,�VXc�A�8*

loss�<�Q|�       �	��VXc�A�8*

loss.�:5g       �	+Q�VXc�A�8*

loss��d9S�ˈ       �	:�VXc�A�8*

lossu�<��j�       �	���VXc�A�8*

loss@�9r�R       �	/�VXc�A�8*

loss�q;��.       �	~��VXc�A�8*

loss�k/;C(�(       �	?X�VXc�A�8*

loss$�<�Q1       �	���VXc�A�8*

lossLi�:�a�E       �	��VXc�A�8*

lossZ��9��!       �	-B�VXc�A�8*

lossC4<<�St~