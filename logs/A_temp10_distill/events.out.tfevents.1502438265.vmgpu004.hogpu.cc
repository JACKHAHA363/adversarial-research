       �K"	  @�Xc�Abrain.Event:2�Q�!�     9���	�X�Xc�A"��
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
 *�x�*
_output_shapes
: *
dtype0
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
seed2���*
T0*
seed���)*
dtype0
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
:@@*
seed2ƃh
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
:���������@*
seed2���*
T0*
seed���)*
dtype0
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
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*
T0*$
_class
loc:@activation_2/Relu*J
_output_shapes8
6:���������@:���������@
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
seed2��[
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
:����������*
seed2*
T0*
seed���)*
dtype0
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
   *
_output_shapes
:*
dtype0
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
seed2���*
T0*
seed���)*
dtype0
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
 *  @?*
_output_shapes
: *
dtype0
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2���*
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
T0*
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
 *   ?*
_output_shapes
: *
dtype0
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2�W
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
 *   A*
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
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*
Tshape0*0
_output_shapes
:������������������
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
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
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
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
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
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
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
]
PlaceholderPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
L
div_2/yConst*
valueB
 *   A*
_output_shapes
: *
dtype0
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*'
_output_shapes
:���������
*
T0
c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
T0*
out_type0*
_output_shapes
:
d
"softmax_cross_entropy_loss_1/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
_output_shapes
: *
T0
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
T0*

axis *
N*
_output_shapes
:
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss_1/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
j
(softmax_cross_entropy_loss_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*
Tshape0*0
_output_shapes
:������������������*
T0
e
#softmax_cross_entropy_loss_1/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
out_type0*
_output_shapes
:*
T0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*

axis *
_output_shapes
:*
T0*
N
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*

axis *
_output_shapes
:*
T0*
N
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
_output_shapes
: *
T0
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
_output_shapes
: *
T0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
T0*
_output_shapes
: 
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
_output_shapes
: *
T0
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
T0*
_output_shapes
: 
P
Placeholder_1Placeholder*
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
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
_output_shapes
: *
T0
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: 
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: *
T0
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1RealDiv3gradients/softmax_cross_entropy_loss_1/div_grad/Neg#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
_output_shapes
: *
T0
�
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
_output_shapes
: *
T0
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
_output_shapes
: *
T0
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select*
_output_shapes
: *
T0
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: *
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
out_type0*
_output_shapes
:*
T0
�
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*#
_output_shapes
:���������*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape*#
_output_shapes
:���������
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
T0*
out_type0*
_output_shapes
:
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
gradients/div_2_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_2_grad/RealDivRealDiv;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapediv_2/y*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

o
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_2_grad/RealDiv_1RealDivgradients/div_2_grad/Negdiv_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/RealDiv_2RealDivgradients/div_2_grad/RealDiv_1div_2/y*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients/div_2_grad/tuple/group_depsNoOp^gradients/div_2_grad/Reshape^gradients/div_2_grad/Reshape_1
�
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������
*
T0
�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: *
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
_output_shapes
:
*
T0*
data_formatNHWC
�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
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
transpose_b(*)
_output_shapes
:�����������*
transpose_a( *
T0
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
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
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
Lgradients/sequential_1/dropout_1/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
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
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
T0
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
valueB�*    *
_output_shapes	
:�*
dtype0
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
validate_shape(*
use_locking(
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
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0*
use_locking( 
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes	
:�*
T0*
use_locking( 
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "#P���,     ���{	7�[�Xc�AJ��
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
seed2���
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
 *�\1=*
_output_shapes
: *
dtype0
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2ƃh*
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
valueB"      *
dtype0*
_output_shapes
:
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
seed2���
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
seed2��[
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
:����������*
seed2*
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
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2���
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
"sequential_1/dropout_1/cond/SwitchSwitchdropout_1/keras_learning_phasedropout_1/keras_learning_phase*
_output_shapes

::*
T0

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
 *  @?*
_output_shapes
: *
dtype0
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
seed2���
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
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2�W*
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
 *   A*
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
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
out_type0*
_output_shapes
:
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
valueB *
dtype0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
]
PlaceholderPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������

L
div_2/yConst*
valueB
 *   A*
_output_shapes
: *
dtype0
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*'
_output_shapes
:���������
*
T0
c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
T0*
out_type0*
_output_shapes
:
d
"softmax_cross_entropy_loss_1/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*

axis *
_output_shapes
:*
T0*
N
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss_1/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*
T0*
Tshape0*0
_output_shapes
:������������������
e
#softmax_cross_entropy_loss_1/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
out_type0*
_output_shapes
:*
T0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
T0*

axis *
N*
_output_shapes
:
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
�
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*
Tshape0*#
_output_shapes
:���������*
T0
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
_output_shapes
: *
T0
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
_output_shapes
: *
T0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
_output_shapes
: *
T0
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
T0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
_output_shapes
: *
T0
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
_output_shapes
: *
T0
P
Placeholder_1Placeholder*
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
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
�
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
T0*
_output_shapes
: 
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
T0*
_output_shapes
: 
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: *
T0
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: *
T0
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
Tshape0*
_output_shapes
: *
T0

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1RealDiv3gradients/softmax_cross_entropy_loss_1/div_grad/Neg#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
_output_shapes
: *
T0
�
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
_output_shapes
: 
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
_output_shapes
: *
T0
�
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
_output_shapes
: *
T0
�
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select*
_output_shapes
: 
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: *
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
_output_shapes
: *
dtype0
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
T0*
out_type0*
_output_shapes
:
�
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
out_type0*
_output_shapes
:
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape*#
_output_shapes
:���������*
T0
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
T0*
out_type0*
_output_shapes
:
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
gradients/div_2_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_2_grad/RealDivRealDiv;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapediv_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
o
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
~
gradients/div_2_grad/RealDiv_1RealDivgradients/div_2_grad/Negdiv_2/y*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/RealDiv_2RealDivgradients/div_2_grad/RealDiv_1div_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/div_2_grad/tuple/group_depsNoOp^gradients/div_2_grad/Reshape^gradients/div_2_grad/Reshape_1
�
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������

�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: 
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
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
transpose_b(*(
_output_shapes
:����������*
transpose_a( *
T0
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
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
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
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/NegNegsequential_1/dropout_1/cond/mul*
T0*/
_output_shapes
:���������@
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
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
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
b
zeros_9Const* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
dense_1/kernel/Adam_1
VariableV2*
	container *
dtype0*!
_class
loc:@dense_1/kernel*
shared_name *!
_output_shapes
:���*
shape:���
�
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
W
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
VariableV2*
	container *
shared_name *
dtype0*
shape:	�
*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
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
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1zeros_13*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
�
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
shared_name *
_output_shapes
:
*
shape:

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
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking( 
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0*
use_locking( 
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0�o��       ��-	WB��Xc�A*

lossj]@��x:       ��-	���Xc�A*

loss�@j��H       ��-	]���Xc�A*

loss|�@� �       ��-	�4��Xc�A*

loss{�@>(�s       ��-	�ב�Xc�A*

loss=@G�z,       ��-	t{��Xc�A*

loss��?��G@       ��-	���Xc�A*

loss��?qQ�       ��-	���Xc�A*

loss)J�?���       ��-	>���Xc�A	*

loss��?�^D�       ��-	�I��Xc�A
*

lossA�?G�N|       ��-	!��Xc�A*

loss^0�?U�}       ��-	E���Xc�A*

lossh/�?��w       ��-	���Xc�A*

loss4�?���       ��-	�З�Xc�A*

lossvɂ?����       ��-	�q��Xc�A*

loss��?w���       ��-	���Xc�A*

loss�.J?G��w       ��-	v���Xc�A*

loss��e?�\�       ��-	�E��Xc�A*

loss-�<?f 5       ��-	&��Xc�A*

loss�T?w�I�       ��-	^���Xc�A*

loss�e?���M       ��-	��Xc�A*

lossLi?��a       ��-	Թ��Xc�A*

lossM?�c�"       ��-	&S��Xc�A*

lossq��?�j�L       ��-	<���Xc�A*

loss4�X?gW�       ��-	攞�Xc�A*

loss��W?J3#�       ��-	JF��Xc�A*

loss�#;?�+�b       ��-	���Xc�A*

loss��:?Ս�p       ��-	����Xc�A*

loss�?�?l�+�       ��-	�0��Xc�A*

loss�Si?G���       ��-	WΡ�Xc�A*

loss�5U?�4       ��-	Ul��Xc�A*

loss=�;?�l�i       ��-	���Xc�A *

loss�H?=6(9       ��-	=���Xc�A!*

lossH�c?0j�       ��-	�f��Xc�A"*

loss�V"?�ռ�       ��-	��Xc�A#*

loss��Y?���!       ��-	1���Xc�A$*

loss��?y�|�       ��-	~S��Xc�A%*

losscZ?��a�       ��-	����Xc�A&*

loss�6g?HK^       ��-	��Xc�A'*

loss�Q?��:�       ��-	6��Xc�A(*

lossJw�>�P��       ��-	%Ψ�Xc�A)*

loss��>P��       ��-	2s��Xc�A**

loss�$$?	>��       ��-	���Xc�A+*

loss~	?QJ��       ��-	���Xc�A,*

loss��/?ӣ�       ��-	�d��Xc�A-*

lossIC&?mr��       ��-	;V��Xc�A.*

loss"�?|B�       ��-	����Xc�A/*

loss��>��P       ��-	����Xc�A0*

loss�O-?"8�I       ��-	�8��Xc�A1*

losstj�>e5a       ��-	ծ�Xc�A2*

loss�Y�>�$4�       ��-	�x��Xc�A3*

loss?*?3(w�       ��-	����Xc�A4*

loss���>�00�       ��-	�(��Xc�A5*

loss4s?u��       ��-	�ű�Xc�A6*

loss%��>c�Cx       ��-	S���Xc�A7*

loss��>�ڨ�       ��-	8-��Xc�A8*

loss,��>M���       ��-	3ó�Xc�A9*

loss��>Jc��       ��-	XV��Xc�A:*

lossΣ?��?       ��-	����Xc�A;*

loss��>��i�       ��-	����Xc�A<*

loss,I�>~o��       ��-	�;��Xc�A=*

lossL�>*�	�       ��-	s׶�Xc�A>*

loss��>�oN       ��-	�x��Xc�A?*

loss��?X��       ��-	`��Xc�A@*

loss}��>aS3       ��-	U���Xc�AA*

loss���>1��       ��-	�]��Xc�AB*

loss�:?MP       ��-	���Xc�AC*

loss\/	?�d?       ��-	q���Xc�AD*

lossH5??�$&�       ��-	O��Xc�AE*

loss�8�>���       ��-	���Xc�AF*

loss��?�(�       ��-	����Xc�AG*

lossz��>-̛�       ��-	�)��Xc�AH*

loss<t�>~;�
       ��-	ý�Xc�AI*

lossB
?�� �       ��-	>[��Xc�AJ*

lossָ?bw�K       ��-	���Xc�AK*

loss�?�י�       ��-	����Xc�AL*

loss�K?�nY$       ��-	D3��Xc�AM*

loss�+?��O�       ��-	����Xc�AN*

lossF�>���       ��-	9c��Xc�AO*

loss��>�jQ       ��-	P��Xc�AP*

loss�H�>0P�<       ��-	����Xc�AQ*

loss���>�7��       ��-	Cq��Xc�AR*

loss]�?���       ��-	^��Xc�AS*

loss�"�>J�ޜ       ��-	E���Xc�AT*

loss��>����       ��-	�]��Xc�AU*

lossN��>�K�,       ��-	����Xc�AV*

loss���>"���       ��-	���Xc�AW*

loss�?�>�6�       ��-	N��Xc�AX*

loss=��>}�0?       ��-	����Xc�AY*

loss��>�40c       ��-	���Xc�AZ*

loss,?���       ��-	G9��Xc�A[*

loss�>���       ��-	Z���Xc�A\*

loss@��>�b�       ��-	�w��Xc�A]*

loss25�>��~D       ��-	���Xc�A^*

lossD�?$�;       ��-	S���Xc�A_*

loss��>�4�       ��-	�T��Xc�A`*

loss�L�>�&�       ��-	����Xc�Aa*

lossz�?zC8�       ��-	����Xc�Ab*

loss۹B?~�,       ��-	�2��Xc�Ac*

lossN�>����       ��-	����Xc�Ad*

loss��V>��r�       ��-	���Xc�Ae*

loss�Jh>[�N�       ��-	���Xc�Af*

loss���>x��       ��-	����Xc�Ag*

loss���>��OB       ��-	*V��Xc�Ah*

losseR�>�J�       ��-	����Xc�Ai*

loss���>���       ��-	����Xc�Aj*

loss�m�>�UxL       ��-	n4��Xc�Ak*

lossA�>��2]       ��-	����Xc�Al*

loss��>s!5�       ��-	�|��Xc�Am*

loss�B�>g��       ��-	���Xc�An*

loss��?֞Jc       ��-	p���Xc�Ao*

loss��>�gV       ��-	�I��Xc�Ap*

loss]`>͎H�       ��-	���Xc�Aq*

loss�@W>��R�       ��-	����Xc�Ar*

losso�N>{�%�       ��-	j4��Xc�As*

loss9ր>����       ��-	�H��Xc�At*

loss,��>)s�       ��-	����Xc�Au*

loss�&X>l�dx       ��-	Cp��Xc�Av*

lossxS�>N[��       ��-	���Xc�Aw*

loss��>�
k'       ��-	6���Xc�Ax*

loss]
�>�g��       ��-	�F��Xc�Ay*

loss���>S*`       ��-	����Xc�Az*

loss���>����       ��-	�v��Xc�A{*

loss߁�>U���       ��-	���Xc�A|*

lossQ��=rFK       ��-	ͬ��Xc�A}*

loss�M�>5���       ��-	r���Xc�A~*

loss���>���       ��-	�>��Xc�A*

loss�>���       �	#/��Xc�A�*

loss���>1dd       �	����Xc�A�*

loss�k�>^lC^       �	,f��Xc�A�*

loss�}�>��U�       �	���Xc�A�*

loss�>F�$       �	����Xc�A�*

lossN�=�e9�       �	�;��Xc�A�*

loss�7>ͽ+n       �	5���Xc�A�*

loss�`>?���       �	<j��Xc�A�*

loss���>��       �	� ��Xc�A�*

loss� �>��{�       �	,���Xc�A�*

loss8>O�;�       �	y=��Xc�A�*

loss_[;>P��       �	����Xc�A�*

losszT>B��S       �	�f��Xc�A�*

loss1��=3�\       �	���Xc�A�*

loss��\>-5�       �	o���Xc�A�*

loss�@_>�R>)       �	�9��Xc�A�*

loss�(�>��,       �	(���Xc�A�*

loss�G>d陧       �	<h��Xc�A�*

loss|#>�o�       �	����Xc�A�*

loss�:>�h��       �	���Xc�A�*

loss˱>��0       �	�*��Xc�A�*

lossO�>�P��       �	���Xc�A�*

loss�K>��`�       �	�g��Xc�A�*

loss/%�=���       �	���Xc�A�*

loss=o�>��M�       �	���Xc�A�*

loss)�>��|�       �	eP��Xc�A�*

loss>��>��       �	|���Xc�A�*

lossJ��>���u       �	���Xc�A�*

loss���=�B�       �	�%��Xc�A�*

lossf�>��5�       �	3���Xc�A�*

loss�p�>�d��       �	Vd��Xc�A�*

lossR��>���6       �	 ��Xc�A�*

loss{>=��       �	����Xc�A�*

loss�b�>=��       �	�7��Xc�A�*

loss�!>�@.b       �	����Xc�A�*

lossj�;>�uݙ       �	f��Xc�A�*

loss��=��       �	����Xc�A�*

lossT�>�`�3       �	S���Xc�A�*

loss��B>�9��       �	(��Xc�A�*

lossө>6|�       �	����Xc�A�*

loss�}>Ws�^       �	�P��Xc�A�*

lossȝ>Z �W       �	7���Xc�A�*

loss0>n�]�       �	<���Xc�A�*

loss��7>';a�       �	n��Xc�A�*

loss�&�=!��       �	Q���Xc�A�*

lossH�P>����       �	����Xc�A�*

loss.k�>�P�%       �	�$��Xc�A�*

loss�w�=�Ot       �	+���Xc�A�*

loss�?�>�V�       �	�_ �Xc�A�*

loss �>���       �	�� �Xc�A�*

lossXS�>��_       �	O��Xc�A�*

loss3�>�?b       �	�1�Xc�A�*

loss�C>�d'�       �	���Xc�A�*

loss�w�=\$�       �	���Xc�A�*

lossl�t>y���       �	[��Xc�A�*

lossϙ->n�p4       �	3�Xc�A�*

loss��=y�       �	^��Xc�A�*

loss!%>�J�       �	Kt�Xc�A�*

loss�"+>�C�       �	��Xc�A�*

loss�Oa>����       �	���Xc�A�*

losst��>��y       �	׿�Xc�A�*

loss�V�>N�n�       �	�\	�Xc�A�*

loss�]>gi��       �	?�	�Xc�A�*

loss|�U>2{/�       �	�
�Xc�A�*

loss�7>3�       �	�>�Xc�A�*

lossі>���       �	{��Xc�A�*

loss�B6>�       �	�|�Xc�A�*

loss��>���a       �	��Xc�A�*

loss�z>u���       �	h��Xc�A�*

lossyP�>       �	�S�Xc�A�*

loss�:�>���H       �	���Xc�A�*

loss�B�>��%       �	���Xc�A�*

lossfK0>v�[       �	�.�Xc�A�*

loss���=UC��       �	���Xc�A�*

loss_��=���	       �	�i�Xc�A�*

loss�1> ��F       �	�Xc�A�*

loss�^2>�X��       �	 ��Xc�A�*

loss�v�=�>��       �	�b�Xc�A�*

lossjM�=^
�       �	���Xc�A�*

loss�>A`n�       �	Ü�Xc�A�*

loss�r:>U�       �	n2�Xc�A�*

lossS%>Vű       �	���Xc�A�*

loss3Sn>z1�q       �	�]�Xc�A�*

loss_��>���       �	���Xc�A�*

loss�@ >4��0       �	���Xc�A�*

loss
>�F�       �	��Xc�A�*

loss�0>��1       �	��Xc�A�*

loss���>#Hv�       �	�I�Xc�A�*

loss��>~�       �	���Xc�A�*

loss6
&>��M�       �	�x�Xc�A�*

loss�|>���       �	^�Xc�A�*

lossD^�>qߣI       �	֩�Xc�A�*

loss}��><�a]       �	�?�Xc�A�*

loss��=Y2g       �	���Xc�A�*

loss%�>;"X�       �	qs�Xc�A�*

losss�>ݹ��       �	�	�Xc�A�*

loss�)5>�"�       �	ͯ�Xc�A�*

lossxP>�!�       �	2V�Xc�A�*

lossr>��rR       �	���Xc�A�*

lossH�R>��XP       �	� �Xc�A�*

loss��}>��J       �	_'!�Xc�A�*

loss�RA>t�9       �	>�!�Xc�A�*

loss �2>��K�       �	�e"�Xc�A�*

loss��>����       �	��"�Xc�A�*

loss���=�y=       �	[�#�Xc�A�*

loss��>��d       �	.$�Xc�A�*

loss���=�l��       �	~�$�Xc�A�*

loss.�~>�k��       �	$^%�Xc�A�*

loss��>��3L       �	 �%�Xc�A�*

loss��?>�[3q       �	�&�Xc�A�*

lossH�>;�       �	�5'�Xc�A�*

loss�>6�        �	��'�Xc�A�*

loss,� >�U��       �	�s(�Xc�A�*

lossRT>ܠX       �	{)�Xc�A�*

loss7%>��       �	ũ)�Xc�A�*

loss�>��b       �	HR*�Xc�A�*

lossN�5>�\i�       �	��*�Xc�A�*

loss-�>s�       �	�+�Xc�A�*

loss�/L>a�m�       �	[C,�Xc�A�*

loss�q�= ��       �	��,�Xc�A�*

loss8}>Of��       �	�-�Xc�A�*

loss�l>���       �	�2.�Xc�A�*

loss�a7>O���       �	o�.�Xc�A�*

lossW�>�"       �	^�/�Xc�A�*

lossV�)>9���       �	�0�Xc�A�*

loss���>x��?       �	J�0�Xc�A�*

lossi�q=Q5�       �	�U1�Xc�A�*

loss��>M��       �	��1�Xc�A�*

lossW7F>�+"       �	ٗ2�Xc�A�*

lossőc>M�~]       �	83�Xc�A�*

lossE�>�.       �	��3�Xc�A�*

loss��
>�+F       �	2q4�Xc�A�*

loss.�|>�6�c       �	=
5�Xc�A�*

lossrt>��O�       �	�5�Xc�A�*

losso��=M�Sn       �	h?6�Xc�A�*

lossj�0>���"       �	��6�Xc�A�*

lossz�>�9       �	�{7�Xc�A�*

loss}ժ>틊s       �	�%8�Xc�A�*

loss�>y+�i       �	��8�Xc�A�*

loss�3>\q��       �	�v9�Xc�A�*

loss\�H>���B       �	:�Xc�A�*

loss�.�>
kh       �	��:�Xc�A�*

lossV�>S��       �	^g;�Xc�A�*

loss�E>�,hq       �	�<�Xc�A�*

lossO�f>�I~�       �	~�<�Xc�A�*

lossa�N>�7)�       �	�X=�Xc�A�*

loss��>�eKY       �	��=�Xc�A�*

loss�	a>����       �	��>�Xc�A�*

loss
�}>Y��       �	��@�Xc�A�*

loss-�#>Ox�       �	��A�Xc�A�*

loss�o�=�'m       �	`B�Xc�A�*

lossB9>����       �	��B�Xc�A�*

loss,B#>�)�5       �	x�C�Xc�A�*

loss%��=�>).       �	]7D�Xc�A�*

loss]�>����       �	4�D�Xc�A�*

loss��b>�;q       �	_}F�Xc�A�*

loss.��=3p8�       �	~G�Xc�A�*

loss�]>M��       �	9�G�Xc�A�*

loss��	>x�d       �	IJH�Xc�A�*

lossL��=�9�       �	��H�Xc�A�*

lossq-�=:bi       �	>yI�Xc�A�*

loss�0>
��]       �	uJ�Xc�A�*

lossl ]=՛��       �	=K�Xc�A�*

loss��=P�_N       �	ͫK�Xc�A�*

loss��>��       �	<LL�Xc�A�*

loss�M�=y       �	[�L�Xc�A�*

loss�{�=j �,       �	:�M�Xc�A�*

loss��=�*�L       �	w0N�Xc�A�*

loss��>z&�       �	��N�Xc�A�*

loss_PM>����       �	g�O�Xc�A�*

loss�Mj>'�>9       �	�3P�Xc�A�*

lossD�>n�U�       �	��P�Xc�A�*

loss�*#>0�K       �	jQ�Xc�A�*

loss�>���       �	�R�Xc�A�*

lossr��=��       �	��R�Xc�A�*

lossLE�=⺊�       �	AS�Xc�A�*

loss_�a>\�d       �	jT�Xc�A�*

loss���='�z       �	p�T�Xc�A�*

loss�x$>�E�       �	�JU�Xc�A�*

loss�H�=g�U�       �	?�U�Xc�A�*

loss���=�͡0       �	x|V�Xc�A�*

loss?D�=�G��       �	�W�Xc�A�*

loss!�>t�c.       �	��W�Xc�A�*

loss���=���       �	_X�Xc�A�*

lossT�/>�׷       �	��X�Xc�A�*

loss��7>���=       �	��Y�Xc�A�*

loss��G>ٖ y       �	4.Z�Xc�A�*

lossA�i=$�       �	��Z�Xc�A�*

lossJ	>��t�       �	χ[�Xc�A�*

lossR�=�yS       �	�j\�Xc�A�*

loss�t�=9[Q�       �	�]�Xc�A�*

loss�p>>|��{       �	��]�Xc�A�*

loss�>��       �	�Y^�Xc�A�*

loss��p=}e�       �	�^�Xc�A�*

lossL��>}�T$       �	x�_�Xc�A�*

loss��J>�q A       �	��`�Xc�A�*

lossa�>E��       �	�+a�Xc�A�*

loss�'_>i�}�       �	�-b�Xc�A�*

loss�!�>�Mv       �	 �b�Xc�A�*

loss��?>����       �	�id�Xc�A�*

loss���=�4       �	�e�Xc�A�*

loss��>�d�K       �	4�e�Xc�A�*

loss}t>>7��B       �	�9f�Xc�A�*

loss419>{��       �	/kg�Xc�A�*

loss��=�'�I       �	~h�Xc�A�*

loss�t>}�<I       �	(�h�Xc�A�*

loss��=����       �	�pi�Xc�A�*

losse�]>��[�       �	��j�Xc�A�*

loss���=1EY�       �	V*k�Xc�A�*

loss%��=�.       �	��k�Xc�A�*

lossD�=,c�       �	D�l�Xc�A�*

loss��
>��       �	�&m�Xc�A�*

loss�`=��L�       �	{�m�Xc�A�*

lossa�=� T       �	zVn�Xc�A�*

loss�)�=�5��       �	��n�Xc�A�*

loss3m4>=I�       �	��o�Xc�A�*

loss�=m0�i       �	Up�Xc�A�*

loss���>7��       �	��p�Xc�A�*

loss1R>im�       �	�Zq�Xc�A�*

loss@t�=��]       �	��r�Xc�A�*

lossK�>'9�J       �	�-s�Xc�A�*

lossE��=��       �	��s�Xc�A�*

loss��K>��+       �	�^t�Xc�A�*

lossS��=��AO       �	L�t�Xc�A�*

loss���=~Z�B       �	�u�Xc�A�*

loss!�=��t       �	fKv�Xc�A�*

loss�B>���       �	��v�Xc�A�*

loss��[>݆��       �	��w�Xc�A�*

loss���=��"       �	G x�Xc�A�*

loss��=b�z       �	Զx�Xc�A�*

lossj�>�>l�       �	eSy�Xc�A�*

lossx�=b'Ɵ       �	��y�Xc�A�*

loss��#>�Е       �	Ҋz�Xc�A�*

lossL>w��       �	�"{�Xc�A�*

loss
�>��       �	��{�Xc�A�*

lossU�>s�*       �	�Q|�Xc�A�*

loss��~>×��       �	�|�Xc�A�*

lossѢ_>.�$       �	0�}�Xc�A�*

lossR��=�gb       �	�~�Xc�A�*

loss��k>'�u�       �	t�~�Xc�A�*

lossh�&>��T�       �	�N�Xc�A�*

lossv�f>Rӷ       �	��Xc�A�*

loss��>���J       �	<���Xc�A�*

loss�[>z�q       �	a��Xc�A�*

loss׆9>�TB�       �	���Xc�A�*

loss��>x       �	zP��Xc�A�*

loss�u�=CsQ       �	6��Xc�A�*

lossv�={�E�       �	|��Xc�A�*

loss��]>�7�       �	 ��Xc�A�*

loss]��=\n��       �	Ȅ�Xc�A�*

loss���=y���       �	c��Xc�A�*

loss���=<��       �	r���Xc�A�*

loss�=�>���;       �	q���Xc�A�*

loss6TQ>��`       �	N*��Xc�A�*

loss8�I>a2�'       �	Ƈ�Xc�A�*

loss���=�~       �	}v��Xc�A�*

loss�p�=*F       �	歊�Xc�A�*

loss�e�=@� �       �	NG��Xc�A�*

loss U[=����       �	^���Xc�A�*

loss�E5>���       �	��Xc�A�*

loss���=3��       �	�;��Xc�A�*

loss_��={^�       �	���Xc�A�*

loss@>Ȣ��       �	l���Xc�A�*

lossI{H>&NR       �	M2��Xc�A�*

loss�}�=ьqd       �	n���Xc�A�*

loss҃Q>M'�       �	���Xc�A�*

lossܭ�=P"l       �	o���Xc�A�*

loss��T>�=O       �	�W��Xc�A�*

lossXJ�>��Id       �	�6��Xc�A�*

loss��=g:�S       �	*���Xc�A�*

loss�6X=�J�w       �	����Xc�A�*

loss,s�=n'��       �	�X��Xc�A�*

loss�>0�8^       �	�n��Xc�A�*

loss(+>�w�:       �	'��Xc�A�*

lossi�8>6�|�       �	����Xc�A�*

loss>6�.       �	UQ��Xc�A�*

loss<�z=�f2       �	���Xc�A�*

lossIj==����       �	g���Xc�A�*

lossv�$>��       �	�`��Xc�A�*

loss��v>�!�3       �	���Xc�A�*

losssdZ>jb�       �	̙��Xc�A�*

loss\��=�Kw�       �	�T��Xc�A�*

loss�Lg>;Ku�       �	����Xc�A�*

loss;�R>�̓�       �	����Xc�A�*

lossi8>E5       �	.=��Xc�A�*

loss���=&��&       �	c՟�Xc�A�*

loss8|>�wHr       �	�~��Xc�A�*

loss!��>ĸ"Z       �	�'��Xc�A�*

lossa�Z>@       �	�á�Xc�A�*

losso�=u�       �	�\��Xc�A�*

loss��=��ax       �	����Xc�A�*

loss�;g>cd
T       �	g���Xc�A�*

loss���=O�       �	�i��Xc�A�*

loss[vq=C��       �	���Xc�A�*

loss֨�=���       �	0���Xc�A�*

loss��=��ì       �	�S��Xc�A�*

lossO��>/���       �	p��Xc�A�*

loss�Z>�տ�       �	]���Xc�A�*

loss��>����       �	����Xc�A�*

loss9>�ن�       �	Ƌ��Xc�A�*

lossA�1>l�n       �	�'��Xc�A�*

loss�>����       �	���Xc�A�*

loss�FH=���r       �	\��Xc�A�*

loss3�7>G !w       �	����Xc�A�*

loss�?;>Z�{       �	�[��Xc�A�*

lossv�D=l�x�       �	�
��Xc�A�*

loss|�>�rw       �	멮�Xc�A�*

loss�N�=��}       �	�D��Xc�A�*

loss��;=Ia��       �	Qۯ�Xc�A�*

loss-�=���       �	�~��Xc�A�*

loss-Վ=�]u�       �	B!��Xc�A�*

loss��<8m0�       �	�±�Xc�A�*

loss��=nɱ�       �	>\��Xc�A�*

loss1pV><�       �	 ��Xc�A�*

lossT�0>���       �	���Xc�A�*

loss鐗=�Sp�       �	�=��Xc�A�*

lossLw�=6�w       �	K��Xc�A�*

lossN�=��'       �	���Xc�A�*

losszq}>F�-�       �	�5��Xc�A�*

loss꣊>$�%u       �	dͶ�Xc�A�*

loss�5>�Lӊ       �	jh��Xc�A�*

loss���=4���       �	\��Xc�A�*

loss�P>.a�       �	����Xc�A�*

loss->�+��       �	2W��Xc�A�*

loss���=�U$�       �	/���Xc�A�*

lossf�=��>T       �	Ȕ��Xc�A�*

loss�#>�w6�       �	�5��Xc�A�*

loss�1�>
[�       �	m8��Xc�A�*

loss�[>��T       �	�ܼ�Xc�A�*

loss��E>�R�x       �	|}��Xc�A�*

loss�w>��F       �	�$��Xc�A�*

loss�	�=(.�       �	���Xc�A�*

loss�!>|=:U       �	;T��Xc�A�*

loss$u�=,�E�       �	1'��Xc�A�*

loss<�&=�T�       �	T���Xc�A�*

loss7Q�=���       �	>]��Xc�A�*

loss1P>b��V       �	`���Xc�A�*

loss$�=o�       �	?s��Xc�A�*

loss��f=�F�b       �	z��Xc�A�*

lossfG>�o��       �	ݶ��Xc�A�*

loss���=�֦       �	����Xc�A�*

loss���=��>D       �	�8��Xc�A�*

loss;RG=�}�       �	����Xc�A�*

lossJ�>���       �	���Xc�A�*

loss�z>���       �	l%��Xc�A�*

loss�j.>�1Y�       �	���Xc�A�*

loss@�I>-�q       �	\r��Xc�A�*

loss���<#��@       �	��Xc�A�*

lossi�M=��x�       �	����Xc�A�*

loss�=��&�       �	9b��Xc�A�*

loss͞�=Z��       �	�k��Xc�A�*

loss��>�Z�E       �	 	��Xc�A�*

loss��F>�U��       �	����Xc�A�*

loss�>����       �	G;��Xc�A�*

lossA��=��H�       �	[(��Xc�A�*

loss�9�=9��       �	�Y��Xc�A�*

loss�>Vq�       �	M0��Xc�A�*

loss�I�=���       �	���Xc�A�*

lossqV=����       �	˹��Xc�A�*

loss/>��1       �	���Xc�A�*

loss�}>��T�       �	8f��Xc�A�*

lossD)�=�+��       �	���Xc�A�*

loss	�>)��(       �	����Xc�A�*

loss�R&>�"~       �	����Xc�A�*

loss��)>�<AR       �	����Xc�A�*

loss���=��<�       �	!���Xc�A�*

loss��=�A=6       �	�;��Xc�A�*

loss8�=|�       �	&���Xc�A�*

lossE%|=��d�       �	[���Xc�A�*

loss6@>�JD7       �	1��Xc�A�*

loss��O=���       �	� ��Xc�A�*

loss�8�=���       �	�(��Xc�A�*

loss�i>���       �	6��Xc�A�*

lossqs�=0�!       �	�5��Xc�A�*

loss�\}=��       �	F[��Xc�A�*

loss�x�=l<?       �	;:��Xc�A�*

loss&a=Y�ej       �	���Xc�A�*

loss��>	7m�       �	����Xc�A�*

loss�o�=O�h       �	3n��Xc�A�*

loss� >��I�       �	���Xc�A�*

loss�J=p�       �	���Xc�A�*

loss_р>Ħy%       �	���Xc�A�*

loss�=�V�t       �	Ͽ��Xc�A�*

lossMͭ=�#i       �	����Xc�A�*

loss��=�=O       �	����Xc�A�*

loss�%=*���       �	����Xc�A�*

lossvA	>~�q�       �	�p��Xc�A�*

loss�`�=��8       �	3��Xc�A�*

loss�^>�¼�       �	!���Xc�A�*

loss1o�=���       �	V��Xc�A�*

loss�a>a�       �	���Xc�A�*

loss��=��d�       �	����Xc�A�*

loss�C�=8��;       �	����Xc�A�*

lossq{�=���p       �	����Xc�A�*

loss1��=��i       �	�a��Xc�A�*

loss.��<���       �	+���Xc�A�*

loss.��=�/e       �	�g��Xc�A�*

lossc�=�L�j       �	]���Xc�A�*

lossc��=9;&n       �	_���Xc�A�*

loss�S�=��       �	�|��Xc�A�*

loss��+>�]�       �	"l��Xc�A�*

loss��)>��c�       �	o��Xc�A�*

lossi��<��.�       �	n���Xc�A�*

loss��=u�4V       �	�>��Xc�A�*

loss��2=M]
       �	����Xc�A�*

lossļ�=����       �	Zg��Xc�A�*

lossw(P<�ҵ       �	n���Xc�A�*

loss�3�<��I       �	���Xc�A�*

lossȳ[=1�T       �	�-��Xc�A�*

loss��=,;*�       �	Y���Xc�A�*

lossȄA=n��       �	�y��Xc�A�*

loss6�<��i�       �	N^��Xc�A�*

losscI>z�>-       �	����Xc�A�*

loss{k�;�ي�       �	���Xc�A�*

loss`��;W�e�       �	�'��Xc�A�*

loss�W<DHie       �	���Xc�A�*

lossȾ�=����       �		T �Xc�A�*

loss��$>�       �	�� �Xc�A�*

loss���=}�v{       �	���Xc�A�*

lossnk�;0�z       �	6�Xc�A�*

lossf�C=
��       �	e��Xc�A�*

loss5?͵       �	K�Xc�A�*

lossq$�;ǭ��       �	���Xc�A�*

loss�l=v�G!       �	���Xc�A�*

loss�g>��       �	��Xc�A�*

loss�#<>V��       �	e9�Xc�A�*

loss�x�=�{K
       �	j��Xc�A�*

lossJ�m=�[0       �	|�Xc�A�*

lossj�>�a!f       �		�Xc�A�*

loss#��=�ȼ�       �	Ŭ	�Xc�A�*

loss�7>�-�I       �	�O
�Xc�A�*

loss��+>̡�[       �	)�
�Xc�A�*

loss�=�^�;       �	�~�Xc�A�*

lossl+P>�r��       �	j�Xc�A�*

loss�6K>x�*       �	\��Xc�A�*

loss��=1�R       �	�E�Xc�A�*

loss��">X���       �	M��Xc�A�*

loss�ae>��q�       �	!w�Xc�A�*

loss�{�=2/��       �	��Xc�A�*

loss>@
>�Q�       �	��Xc�A�*

lossg_#>9["       �	�f�Xc�A�*

loss�}�=r�'{       �	K�Xc�A�*

lossD��=��˄       �	��Xc�A�*

loss`V�=2��       �	9�Xc�A�*

loss<>��=�       �	��Xc�A�*

loss%�B=       �	
d�Xc�A�*

loss�*�="��       �	I��Xc�A�*

loss��L=q ӹ       �	���Xc�A�*

lossh�=��J        �	d:�Xc�A�*

lossڄ�=*J�n       �	M��Xc�A�*

loss�G>.�$L       �	ݗ�Xc�A�*

loss���=��s'       �		3�Xc�A�*

loss�=@�1       �	.��Xc�A�*

lossܢ�=GH�	       �	\�Xc�A�*

loss)��=��y       �	s��Xc�A�*

lossn��</��       �	q��Xc�A�*

loss�#�=�>/       �	V+�Xc�A�*

loss䮉=Y�lI       �	���Xc�A�*

loss15P=
��>       �	'f�Xc�A�*

loss.��=2�ی       �	��Xc�A�*

loss8�>�	nz       �	U��Xc�A�*

loss��<> �1       �	�B�Xc�A�*

loss'�=ut       �	���Xc�A�*

losse�E=6!B+       �	Gs�Xc�A�*

loss1�{=�
:       �		�Xc�A�*

loss�;�=7�~�       �	Ӡ�Xc�A�*

loss���=7�2�       �	K= �Xc�A�*

loss�L�=�>�       �	,� �Xc�A�*

loss���=P���       �	�p!�Xc�A�*

loss�=4�L       �	�"�Xc�A�*

lossRF>i�       �	ǜ"�Xc�A�*

loss<]t=.��0       �	z4#�Xc�A�*

lossv�:=��tp       �	��#�Xc�A�*

losso[>aKgY       �	�o?�Xc�A�*

loss��=�S��       �	I@�Xc�A�*

loss.��=r9��       �	F�@�Xc�A�*

loss�z)>���s       �	TA�Xc�A�*

loss�>m�>       �	��A�Xc�A�*

loss��$=	��       �	�B�Xc�A�*

loss��>,M;�       �	�7C�Xc�A�*

loss:H�>W|v       �	jD�Xc�A�*

loss�c,>��('       �	��D�Xc�A�*

loss�$�>W�       �	+PE�Xc�A�*

loss���=�	��       �	�E�Xc�A�*

loss�H>=�!��       �	i�F�Xc�A�*

loss��U=��X�       �	�/G�Xc�A�*

loss�=nfTM       �	N�G�Xc�A�*

loss�[}=R��+       �	�kH�Xc�A�*

loss�5E>���       �	�
I�Xc�A�*

loss�'=DT�       �	 �I�Xc�A�*

loss�Ϻ=��}�       �	�GJ�Xc�A�*

loss��g=��n�       �	q�J�Xc�A�*

loss�h>��}       �	X�K�Xc�A�*

loss��=D�u       �	�&L�Xc�A�*

loss�&�=Q��'       �	��L�Xc�A�*

loss���=;z8       �	�VM�Xc�A�*

loss��<>�)��       �	N�M�Xc�A�*

loss6O�=���b       �	N�Xc�A�*

loss�fQ=s�1       �	U.O�Xc�A�*

loss�-,>;+%       �	�P�Xc�A�*

lossE&�=&邑       �	7�P�Xc�A�*

loss ��=v�q       �	>vQ�Xc�A�*

lossԍ=���       �	�R�Xc�A�*

loss}�M=0"��       �	��R�Xc�A�*

lossy=<ۑ       �	UiS�Xc�A�*

lossA�>_
~�       �	�T�Xc�A�*

lossW�J>4�%4       �	��T�Xc�A�*

loss��}=6�T       �	�jU�Xc�A�*

lossq�>�{��       �	�V�Xc�A�*

loss�i=۹�b       �	��V�Xc�A�*

lossCNa>(��       �	�^W�Xc�A�*

lossʦ�>����       �	��W�Xc�A�*

loss
_T>	��r       �	#�X�Xc�A�*

lossW`N>>"z.       �	�AY�Xc�A�*

lossN��=q6�       �	��Y�Xc�A�*

losss��=5۪�       �	�xZ�Xc�A�*

loss��>�ɾ       �	�9[�Xc�A�*

lossh�=�+UE       �	*�[�Xc�A�*

loss.O>�B|       �	�\�Xc�A�*

lossې�=��~�       �	>>]�Xc�A�*

loss�9>���b       �	��]�Xc�A�*

loss_��=�&�D       �	F�^�Xc�A�*

loss=V%�       �	�?_�Xc�A�*

loss��=bۊ�       �	�_�Xc�A�*

loss۟�=`�t       �	_�`�Xc�A�*

loss��=)F3w       �	�Ma�Xc�A�*

lossL?G�\T       �	�a�Xc�A�*

loss�_�=�A�A       �	��b�Xc�A�*

loss|�,=#���       �	bJc�Xc�A�*

loss��=����       �	i�c�Xc�A�*

lossz=�ܹ^       �	)�d�Xc�A�*

lossjB�=�q��       �	�[e�Xc�A�*

loss@��=<>�       �	V,f�Xc�A�*

loss�:�=
�J�       �	��f�Xc�A�*

losse��=��m       �	��g�Xc�A�*

loss�p=v��U       �	SYh�Xc�A�*

lossڬ=�v�       �	!i�Xc�A�*

loss��=R)P�       �	��i�Xc�A�*

lossZА=�4�G       �	%�j�Xc�A�*

loss6I�=K�!�       �	�=k�Xc�A�*

loss���=E�J]       �	S�k�Xc�A�*

lossؤF>�L�       �	=�l�Xc�A�*

loss8��=!8��       �	WZm�Xc�A�*

loss]l�=���\       �	in�Xc�A�*

loss�M�=��H       �	D�n�Xc�A�*

lossC�=���       �	 Fo�Xc�A�*

lossW͸=�b�       �	�o�Xc�A�*

loss)� >P(�       �	��p�Xc�A�*

loss�I>��y       �	A)q�Xc�A�*

loss	W[>�\;       �	�r�Xc�A�*

loss�v>�)J       �	z�r�Xc�A�*

lossL�E=Ԓ��       �	�Rs�Xc�A�*

loss���=�o|�       �	��s�Xc�A�*

loss�o�=8�	�       �	q�t�Xc�A�*

loss�N>��'A       �	�/u�Xc�A�*

loss�A>~��       �	�u�Xc�A�*

loss��=�]       �	��v�Xc�A�*

lossr5�=��       �	<w�Xc�A�*

loss޻=�٤�       �	:�w�Xc�A�*

loss��=-G�       �	��x�Xc�A�*

loss�=�E7U       �	o�y�Xc�A�*

loss�B�='@I       �	�z�Xc�A�*

loss��C=Za��       �	�{�Xc�A�*

loss��>^F�        �	J�{�Xc�A�*

loss9�>f:�K       �	d]|�Xc�A�*

loss�^�=)~��       �	:}�Xc�A�*

loss��h=����       �	�}�Xc�A�*

lossR�>�c5�       �	�8~�Xc�A�*

loss
��=�V       �	J�~�Xc�A�*

lossx�>	��       �	�p�Xc�A�*

loss�=wђ�       �	e��Xc�A�*

loss9֊>���       �	���Xc�A�*

lossP�=��aL       �	�c��Xc�A�*

loss�a=I�u�       �	����Xc�A�*

lossA�2=���5       �	0���Xc�A�*

lossvI�=i�-i       �	����Xc�A�*

lossiug=X��'       �	U0��Xc�A�*

loss�pu=E7�       �	eƄ�Xc�A�*

loss�X]>U�!�       �	uZ��Xc�A�*

loss�:C=�t-�       �	����Xc�A�*

lossc'z<��1       �	f���Xc�A�*

loss_�>�9�w       �	�-��Xc�A�*

loss&<0>�?��       �	�Ň�Xc�A�*

lossf�>T�*�       �	ϡ��Xc�A�*

loss
��=v�@9       �	;��Xc�A�*

loss�3�=m^�       �	�҉�Xc�A�*

loss9V�=��Z       �	�x��Xc�A�*

loss�E7=���       �	e��Xc�A�*

loss�|C=BM�_       �	$���Xc�A�*

losso@�=�_�       �	?V��Xc�A�*

loss��=3�x       �	#��Xc�A�*

loss��=b5��       �	ꔍ�Xc�A�*

loss$v=�n�"       �	�A��Xc�A�*

lossOܳ=�v%       �	/��Xc�A�*

loss�ؕ=|0M       �	���Xc�A�*

lossT'>=K�       �	{/��Xc�A�*

loss���=�Kg        �	���Xc�A�*

loss�%�<2ì_       �	�ґ�Xc�A�*

loss!BF=#v�6       �	#���Xc�A�*

loss���=����       �	ޏ��Xc�A�*

loss\��<
*�a       �	�4��Xc�A�*

loss�,<>ġT�       �	����Xc�A�*

loss��=;cgI       �	����Xc�A�*

loss��=b�W�       �	2r��Xc�A�*

loss��=�K��       �	�1��Xc�A�*

loss�b=,o#�       �	���Xc�A�*

loss���<iɘ�       �	h˘�Xc�A�*

loss&>R�ב       �	�r��Xc�A�*

lossE�F>.-]�       �	I��Xc�A�*

loss��=X�       �	����Xc�A�*

loss�Z=�/+�       �	�X��Xc�A�*

loss�\(=N�}       �	��Xc�A�*

loss4�!=���       �	����Xc�A�*

loss�C=���       �	�h��Xc�A�*

loss]R�=L��'       �	���Xc�A�*

loss&}�=���       �	州�Xc�A�*

lossA��=�\��       �	�g��Xc�A�*

loss�6>��       �	���Xc�A�*

loss�Yf=ÁMu       �	輠�Xc�A�*

loss�q/>�t       �	lz��Xc�A�*

loss�y�<La�       �	}$��Xc�A�*

loss�xU=P\{6       �	�͢�Xc�A�*

loss��=��)�       �	��Xc�A�*

loss��,=�~�       �	�N��Xc�A�*

loss��<>~�       �	r���Xc�A�*

loss�
>��f�       �	����Xc�A�*

lossz�+>}�%�       �	uY��Xc�A�*

lossw>��       �	c��Xc�A�*

lossov�<�wq       �	ק�Xc�A�*

loss-8�=��       �	����Xc�A�*

loss-EK>Q4�       �	�(��Xc�A�*

loss�H>�7tC       �	�ͩ�Xc�A�*

loss�$6=�       �	�r��Xc�A�*

loss��=^��       �	���Xc�A�*

loss
=���       �	����Xc�A�*

lossM��=�       �	�P��Xc�A�*

loss���<@��       �	����Xc�A�*

loss�I�=���       �	~���Xc�A�*

losss�=p��       �	�/��Xc�A�*

loss��'=8���       �	�Ȯ�Xc�A�*

loss4��=����       �	����Xc�A�*

lossjG�=�V��       �	�#��Xc�A�*

loss�ե=�؇�       �	n���Xc�A�*

loss�G�=b�e�       �	tb��Xc�A�*

losszX'=��3t       �	���Xc�A�*

lossR�>�	2y       �	)���Xc�A�*

loss|�@>K�6(       �	A.��Xc�A�*

lossc|�<�9�       �	=��Xc�A�*

loss+�=]��*       �	���Xc�A�*

lossTj>��R       �	zT��Xc�A�*

loss١>w���       �	���Xc�A�*

loss\�<];�a       �	����Xc�A�*

loss�5,=!�	i       �	z7��Xc�A�*

loss�y�<F��       �	�ͷ�Xc�A�*

loss-��=*�t       �	�e��Xc�A�*

loss�T>����       �	����Xc�A�*

loss��=���       �	ꖹ�Xc�A�*

loss܊<���       �	><��Xc�A�*

loss���=fMC       �	
غ�Xc�A�*

loss���=�Ȅ[       �	&r��Xc�A�*

loss���=-�X�       �	���Xc�A�*

lossa�\=L�1       �	-���Xc�A�*

loss��>oS�       �	�L��Xc�A�*

lossg�=�6�       �	����Xc�A�*

loss֗e=�U       �	���Xc�A�*

loss�X=!���       �	�0��Xc�A�*

loss �=�f�]       �	�Կ�Xc�A�*

loss�@<�
��       �	�j��Xc�A�*

loss�\=%���       �	s��Xc�A�*

loss�e�=X��       �	���Xc�A�*

lossq��=`�C�       �	F[��Xc�A�*

loss�( >�y�       �	���Xc�A�*

lossx�\=��J       �	����Xc�A�*

loss�^V=Ͽ�V       �	����Xc�A�*

loss�Q=5�4f       �	\u��Xc�A�*

losss��=Vt;}       �	���Xc�A�*

loss�q=��a�       �	����Xc�A�*

loss/,-=��[R       �	^i��Xc�A�*

lossQ�=��$o       �	+3��Xc�A�*

lossi	�<���u       �	���Xc�A�*

lossM��=zu�       �	����Xc�A�*

loss@�'>(�Vx       �	�G��Xc�A�*

lossɂ<��       �	L���Xc�A�*

lossO��=Z��       �	����Xc�A�*

lossD��<<ϔ*       �	g&��Xc�A�*

loss��&=����       �	����Xc�A�*

loss�=C��       �	�u��Xc�A�*

loss;��=pK�H       �	��Xc�A�*

lossC�=>_�       �	����Xc�A�*

lossa
>���       �	�\��Xc�A�*

lossĐH>狁c       �	F��Xc�A�*

loss#n>9Ql`       �	���Xc�A�*

loss�[>�Z       �	����Xc�A�*

loss���<�P       �	a7��Xc�A�*

loss��=��D       �	"���Xc�A�*

loss�"�=�Z�       �	r���Xc�A�*

loss�/'>��n       �	;7��Xc�A�*

loss#2�=�!�a       �	����Xc�A�*

loss��{=��       �	`x��Xc�A�*

loss���<,S˚       �	8��Xc�A�*

lossZ9>�qs        �	ͯ��Xc�A�*

lossZ��=$B1S       �	�S��Xc�A�*

loss�I�=V�       �	A���Xc�A�*

loss(�2=6S       �	����Xc�A�*

loss�ߒ<�-o       �	�&��Xc�A�*

loss_w>�U�       �	����Xc�A�*

loss��=�B�       �	�Y��Xc�A�*

loss][>K��       �	����Xc�A�*

loss��=9�g�       �	n���Xc�A�*

loss���=���h       �	w+��Xc�A�*

lossV6>�E�5       �	����Xc�A�*

loss��>v��       �	d��Xc�A�*

lossO^y=}PY�       �	���Xc�A�*

loss��\=�¨�       �	Ϟ��Xc�A�*

lossX��=?�~e       �	�B��Xc�A�*

lossW�=-Y�2       �	����Xc�A�*

loss��=BŻ�       �	Qh��Xc�A�*

lossa��=$E$�       �	����Xc�A�*

loss��>]���       �	l���Xc�A�*

loss^N=�=o�       �	����Xc�A�*

loss$4}=҃�`       �	�q��Xc�A�*

loss�=B���       �	7��Xc�A�*

loss ��=&���       �	���Xc�A�*

loss`�>nc�_       �		���Xc�A�*

loss{�=�Dqd       �	pB��Xc�A�*

lossht=��RX       �	����Xc�A�*

lossM9�<����       �	E���Xc�A�*

loss_/[<��Í       �	�U��Xc�A�*

loss�/=݅kj       �	����Xc�A�*

loss!�=�xiN       �	���Xc�A�*

lossj��=L���       �	+m��Xc�A�*

loss�m�<)�(�       �	���Xc�A�*

loss
�=�1�       �	���Xc�A�*

loss�e�=����       �	�]��Xc�A�*

loss�[�=j��       �	����Xc�A�*

loss��<>\Q��       �	���Xc�A�*

lossR�l=�>�k       �	�:��Xc�A�*

loss��.>Чo       �	����Xc�A�*

lossŝ|=���       �	6u��Xc�A�*

loss&?>66�       �	���Xc�A�*

loss|�>z��       �	���Xc�A�*

lossc�D>3�w       �	C9��Xc�A�*

loss���=����       �	����Xc�A�*

loss�aX=h�H{       �	����Xc�A�*

loss2�=P~s       �	k(��Xc�A�*

loss��=�o�       �	����Xc�A�*

lossR�o= �<�       �	�Z��Xc�A�*

lossR�>c<W       �	����Xc�A�*

loss�sG>F�       �	x���Xc�A�*

loss4��=\E       �	9��Xc�A�*

loss�n�=T��       �	����Xc�A�*

loss�	>k�        �	v���Xc�A�*

loss_��=��`�       �	�&��Xc�A�*

loss ?0<���       �	����Xc�A�*

loss�^�<5-�       �	�t��Xc�A�*

loss��J>6�X       �	���Xc�A�*

loss�j(=R�l       �	p���Xc�A�*

loss&�=^���       �	Y3��Xc�A�*

loss
,�=/��1       �	�C��Xc�A�*

loss��=��K       �	����Xc�A�*

lossf~o=�P�       �	~p��Xc�A�*

loss�w="�2l       �	c	��Xc�A�*

loss�(�<���m       �	X���Xc�A�*

loss\UM=� O]       �	�I �Xc�A�*

loss���=�*�       �	?� �Xc�A�*

loss��<܊��       �	F}�Xc�A�*

loss�;z<�H̱       �	M�Xc�A�*

loss<�=�4       �	��Xc�A�*

loss�"�=2H�W       �	�H�Xc�A�*

lossM��=�߻b       �	��Xc�A�*

loss�)>�G��       �	x�Xc�A�*

loss��)>���       �	��Xc�A�*

lossMTK>���B       �	0�Xc�A�*

loss�L�=���       �	.��Xc�A�*

loss ��=��\       �	�P�Xc�A�*

loss�^:=a���       �	���Xc�A�*

loss�S�=j悀       �	���Xc�A�*

loss�TK=�j�2       �	,+	�Xc�A�*

loss�@>�h)       �	��	�Xc�A�*

lossʻ<�u�       �	�h
�Xc�A�*

loss�i=��/�       �	z�
�Xc�A�*

loss⸃=L5X�       �	<��Xc�A�*

loss  >V=       �	�6�Xc�A�*

lossz5=/[mK       �	���Xc�A�*

loss�Ȇ=��y       �	�t�Xc�A�*

loss���=���       �	g
�Xc�A�*

loss��<���s       �	Q��Xc�A�*

loss�?=��       �	�?�Xc�A�*

loss�;�='���       �	���Xc�A�*

lossL��=�3f       �	��Xc�A�*

loss�?={��       �	d!�Xc�A�*

lossq�=�.�       �	��Xc�A�*

loss���=�i1       �	�b�Xc�A�*

loss1U=��o       �	 �Xc�A�*

losstr'>� �       �	���Xc�A�*

loss�
�=E��       �	O��Xc�A�*

loss��=^�Α       �	�#�Xc�A�*

lossܑ=��j       �	`��Xc�A�*

lossi�>m2ř       �	�a�Xc�A�*

lossT�=���       �	���Xc�A�*

lossE�A=04�       �	;��Xc�A�*

loss�W�=�,�       �	� �Xc�A�*

loss�m`>��d�       �	���Xc�A�*

loss<�=��J�       �	!\�Xc�A�*

lossm��=�"l       �	���Xc�A�*

lossʠ=��*�       �	��Xc�A�*

loss��
=�.٦       �	1B�Xc�A�*

loss��=d࠷       �	w��Xc�A�*

loss��w=��|       �	�~�Xc�A�*

losss��<���       �	��Xc�A�*

loss;2>;Ǩ(       �	 ��Xc�A�*

lossW��=��q�       �	�A�Xc�A�*

loss�A<���y       �	a��Xc�A�*

loss��=ϸ~:       �	�x�Xc�A�*

loss���<��?       �	� �Xc�A�*

loss!��=t-V       �	�� �Xc�A�*

loss��=5)��       �	�J!�Xc�A�*

loss�l�>�GC�       �	P�!�Xc�A�*

loss*A�=���       �	�w"�Xc�A�*

loss6<=�g�       �	�#�Xc�A�*

loss�u�=쨍�       �	@�#�Xc�A�*

loss��0=��!�       �	d=$�Xc�A�*

loss�z+=2�       �	^,%�Xc�A�*

loss�w=]U�*       �	D�%�Xc�A�*

loss��>��D       �	hY&�Xc�A�*

loss:�>=���       �	�&�Xc�A�*

loss�lG=�jf"       �	G�'�Xc�A�*

lossѽ6>�{�       �	fI(�Xc�A�*

loss�r=^�ax       �	e�(�Xc�A�*

loss�8=�Yi�       �	��)�Xc�A�*

loss�K�=9���       �	�@*�Xc�A�*

loss�{�=���       �	s�*�Xc�A�*

loss��j=@Ձ       �	�w+�Xc�A�*

loss1=-=�O�       �	�,�Xc�A�*

loss��Z<�}c       �	H�,�Xc�A�*

loss�=��@L       �	�:-�Xc�A�*

lossz{�=z�1�       �	��-�Xc�A�*

loss���=֓u�       �	e.�Xc�A�*

loss;��=��vN       �	�/�Xc�A�*

loss��=�x�       �	��/�Xc�A�*

loss�u�=��|t       �	�E0�Xc�A�*

loss`݊=O8RW       �	��0�Xc�A�*

loss-FW=�\1�       �	~n1�Xc�A�*

loss�Sd=�Hb�       �	S2�Xc�A�*

loss��>2K�y       �	(�2�Xc�A�*

loss�ؤ=��       �	�t3�Xc�A�*

lossg:=6�       �	�4�Xc�A�*

loss�ۤ=ᱯM       �	��4�Xc�A�*

loss#� >�4�       �	�I5�Xc�A�*

loss=��=]i=       �	��5�Xc�A�*

loss�3�=���|       �	�y6�Xc�A�*

loss���=*�       �	7�Xc�A�*

lossFk�=�.:�       �	Ӣ7�Xc�A�*

loss&��=�3�       �	�68�Xc�A�*

loss���=z¿f       �	�8�Xc�A�*

loss�"�=D<oj       �	�\9�Xc�A�*

lossh�=ݲ�2       �	5�9�Xc�A�*

loss�t=c��x       �	��:�Xc�A�*

loss=��<��ؠ       �	�!;�Xc�A�*

loss��=�b��       �	��;�Xc�A�*

loss?ro=�8*�       �	�v<�Xc�A�*

loss�Q�=�5ϓ       �	b=�Xc�A�*

loss��=���       �	Ȳ=�Xc�A�*

loss�_>��f       �	N>�Xc�A�*

loss�I�=%��       �	`�>�Xc�A�*

loss�'L=<�&�       �	ly?�Xc�A�*

loss�Q�=�$�[       �	/@�Xc�A�*

loss1�n=]
v       �	G�@�Xc�A�*

loss�S>�w�       �	�KA�Xc�A�*

loss@��<��       �	P�A�Xc�A�*

loss� e=� ��       �	�zB�Xc�A�*

loss��=v-r       �	nC�Xc�A�*

loss,�="�       �	�C�Xc�A�*

loss�a�<��F8       �	[|D�Xc�A�*

lossX��=��֨       �	�E�Xc�A�*

loss�M�=�.       �	8�E�Xc�A�*

lossF�'=�dA       �	`F�Xc�A�*

loss�@=?r�<       �	f�F�Xc�A�*

loss4�D=_�b�       �	�G�Xc�A�*

loss�!>��9>       �	�2H�Xc�A�*

loss��=[-7       �	��H�Xc�A�*

lossr>g=	�</       �	YkI�Xc�A�*

loss1��=ijiJ       �	3J�Xc�A�*

loss���=#x       �	��J�Xc�A�*

loss�>2W�(       �	�OK�Xc�A�*

loss���=?�       �	��K�Xc�A�*

loss��H>�*1�       �	L�L�Xc�A�*

loss
S�=����       �	�$M�Xc�A�*

loss��J>����       �	�N�Xc�A�*

loss�g�<͔�       �	�N�Xc�A�*

loss�[=�7�       �	��O�Xc�A�*

loss12>�A�       �	��P�Xc�A�*

loss���=�@~�       �	�!Q�Xc�A�*

loss�:%=ˏ,L       �	��Q�Xc�A�*

lossc+�=��^E       �	%�R�Xc�A�*

loss��0=DC�       �	_~S�Xc�A�*

loss��1>��;�       �	�T�Xc�A�*

loss��Q=�pF�       �	*�T�Xc�A�*

loss3	>�ݖ       �	�HU�Xc�A�*

loss-�=���       �	��U�Xc�A�*

loss.�>?�9       �	v�V�Xc�A�*

lossLS�=,�       �	�W�Xc�A�*

lossQ��<
G�[       �	7�W�Xc�A�*

loss��(=�9D�       �	.WX�Xc�A�*

lossx/=�e�(       �	��X�Xc�A�*

loss�M�<���       �	Z�Y�Xc�A�*

loss�V�=��d�       �	�Z�Xc�A�*

lossw�K=◚v       �	��Z�Xc�A�*

loss(a=�qi�       �	lC[�Xc�A�*

loss��<�R=�       �	��[�Xc�A�*

loss!�D<3Z�       �	�p\�Xc�A�*

loss8[0=R��&       �	�]�Xc�A�*

loss���<� �a       �	0�]�Xc�A�*

loss}_D>B���       �	3^�Xc�A�*

loss$M>Wd/�       �	�^�Xc�A�*

loss�8>K�t       �	�^_�Xc�A�*

loss	r=H�o�       �	��_�Xc�A�*

loss {<@��_       �	�`�Xc�A�*

loss2��=E���       �	2a�Xc�A�*

loss�f>WE$�       �	J�a�Xc�A�*

loss���=R�4       �	�Ab�Xc�A�*

loss	0�=9	3/       �	�b�Xc�A�*

loss��>vkڨ       �		mc�Xc�A�*

loss��= ͭ}       �	�d�Xc�A�*

lossaZ=Wb��       �	��d�Xc�A�*

loss}��=�GY       �	�e�Xc�A�*

loss�1�=8S�       �	�(f�Xc�A�*

loss�(>!0��       �	�Wg�Xc�A�*

loss��=�A�       �	��g�Xc�A�*

loss��=�CE~       �	V�h�Xc�A�*

loss�D�=ķ?$       �	rli�Xc�A�*

loss��.=gC�1       �	&mj�Xc�A�*

loss1��=,\CB       �	�k�Xc�A�*

loss�o0=l.1\       �	��k�Xc�A�*

lossS��<f�?y       �	�Sl�Xc�A�*

loss��S=x�X5       �	H�l�Xc�A�*

loss��:>J�0       �	�m�Xc�A�*

lossSp=_�D       �	%@n�Xc�A�*

lossn��<d��q       �	�Go�Xc�A�*

loss�?�=�G&       �	�p�Xc�A�*

loss���=��2�       �	.�p�Xc�A�*

loss�=�R0       �	zPq�Xc�A�*

loss�=X�:N       �	}�q�Xc�A�*

lossMS�=<��       �	�r�Xc�A�*

lossmh�=t�j?       �	s�Xc�A�*

loss�C�=\��       �	�s�Xc�A�*

loss
��=�Uo       �	�Ot�Xc�A�*

loss�C=���`       �	��t�Xc�A�*

lossD�=���       �	�u�Xc�A�*

lossъq<�4       �	�Sv�Xc�A�*

lossnv�=�'	       �	��v�Xc�A�*

lossj�=۩�N       �	֏w�Xc�A�*

loss���=�k�       �	.7x�Xc�A�*

lossz�R>	��       �	Z�x�Xc�A�*

losss� >C[��       �	�ty�Xc�A�*

loss +�=�l�z       �	�z�Xc�A�*

loss8l>�ǅ       �	��z�Xc�A�*

loss;��=)N�c       �	�b{�Xc�A�*

loss]�/=2��T       �	h|�Xc�A�*

lossd=�t+�       �	-�|�Xc�A�*

loss
>�[h�       �	9E}�Xc�A�*

loss���<��0�       �	��}�Xc�A�*

lossFg�=[R��       �	}z~�Xc�A�*

loss��=�y�       �	�#�Xc�A�*

loss��x=�p�9       �	a��Xc�A�*

loss��;=�_�       �	 `��Xc�A�*

lossJ��<;�A       �	����Xc�A�*

loss?+�<	�83       �	G���Xc�A�*

lossč'=i�)!       �	�'��Xc�A�*

loss4`=��f#       �	�Ȃ�Xc�A�*

loss2�=���       �	�Z��Xc�A�*

loss�y.=z��w       �	|���Xc�A�*

loss}Et=���       �	{���Xc�A�*

loss�jq=J�&�       �	�%��Xc�A�*

loss��M<@7�       �	L���Xc�A�*

loss�y=��       �	�Y��Xc�A�*

loss7�R=�H       �	��Xc�A�*

loss���=7��       �	����Xc�A�*

loss7�=�#6�       �	�(��Xc�A�*

loss��>W�J       �	ʈ�Xc�A�*

loss`��<�w        �	^f��Xc�A�*

losso�>0H��       �	2 ��Xc�A�*

loss�T�=����       �	����Xc�A�*

loss��=d�\       �	�B��Xc�A�*

loss�F=��X�       �	bڋ�Xc�A�*

loss@b�<�]�C       �	�q��Xc�A�*

loss�eb=���       �	-A��Xc�A�*

loss��6=m86�       �	Dݍ�Xc�A�*

loss��=/o�       �	z��Xc�A�*

loss@��=- ��       �	N��Xc�A�*

loss�Y`=%\gC       �	Q���Xc�A�*

loss에<UZy       �	�7��Xc�A�*

loss/�
=�,A       �	�ѐ�Xc�A�*

loss��3=ۧ�       �	�i��Xc�A�*

loss�p=�H3       �	���Xc�A�*

loss���<��       �	���Xc�A�*

loss~%=�+��       �	����Xc�A�*

loss�Ӕ=Ҧ*       �	S���Xc�A�*

loss�q4=ԧ��       �	�V��Xc�A�*

lossf�=5e�c       �	���Xc�A�*

loss�5>�ƯQ       �	���Xc�A�*

lossR�4>�k�z       �	�1��Xc�A�*

loss37�<�B�       �	\ɗ�Xc�A�*

loss��=�ν+       �	�^��Xc�A�*

loss/�=����       �	'���Xc�A�*

lossv�<�P �       �	����Xc�A�*

loss�<�v#C       �	�A��Xc�A�*

loss8
G<c�پ       �	u��Xc�A�*

loss��'=�;]       �	~���Xc�A�*

lossl�<؜�       �	�-��Xc�A�*

loss7�[<��       �	����Xc�A�*

loss��V<�6��       �	����Xc�A�*

loss���=�0�       �	�F��Xc�A�*

loss4�Y;�*       �	��Xc�A�*

lossqh:鳖       �	����Xc�A�*

loss���:a�'�       �	=*��Xc�A�*

loss<9=c>9�       �	ʠ�Xc�A�*

lossXo�=ܲߒ       �	Ve��Xc�A�*

loss�˃=:
�       �	 ��Xc�A�*

lossσ�<H,��       �	(���Xc�A�*

loss��=��w       �	�:��Xc�A�*

lossȞ>�X��       �	fأ�Xc�A�*

loss�2<�T       �	u��Xc�A�*

loss���<���w       �	���Xc�A�*

loss/��=V�_       �	.���Xc�A�	*

loss<a>�y       �	lC��Xc�A�	*

loss45=GyW�       �	�٦�Xc�A�	*

loss�Q�<DV�       �	�o��Xc�A�	*

loss��>%%��       �	��Xc�A�	*

loss.y�=�T!�       �	����Xc�A�	*

lossm�=˳�,       �	�@��Xc�A�	*

loss���=��j�       �	���Xc�A�	*

lossci�=��@       �	4���Xc�A�	*

loss.�	>ҘD�       �	T��Xc�A�	*

lossO�7>ɢ�       �	����Xc�A�	*

loss��=��.�       �	���Xc�A�	*

loss3��=-{�       �	�B��Xc�A�	*

loss���=/Ͱ�       �	aݭ�Xc�A�	*

lossXb=�@�       �	J~��Xc�A�	*

loss�Z=�I�       �	���Xc�A�	*

loss�g>œc       �	���Xc�A�	*

loss��=���V       �	3P��Xc�A�	*

loss�|!=�+br       �	���Xc�A�	*

lossA��=tL�]       �	�~��Xc�A�	*

loss�Y�=�|`�       �	z��Xc�A�	*

loss�=+D�q       �	X���Xc�A�	*

loss���<���"       �	�=��Xc�A�	*

loss�<��|       �	-ҳ�Xc�A�	*

loss�ˍ=D,#>       �	�l��Xc�A�	*

loss��=�Ve�       �	���Xc�A�	*

loss�6J>N��       �	����Xc�A�	*

loss��=c��o       �	�6��Xc�A�	*

loss��Q=/7�       �	yͶ�Xc�A�	*

loss7�=���       �	�f��Xc�A�	*

loss,�c=?
��       �	X���Xc�A�	*

loss��F<>E       �	����Xc�A�	*

loss:�=��       �	�E��Xc�A�	*

lossY�=FV       �	����Xc�A�	*

loss��={���       �	Ō��Xc�A�	*

lossw��=����       �	�f��Xc�A�	*

loss:��=��n       �	j���Xc�A�	*

loss=h�=f�       �	⑼�Xc�A�	*

loss�?=�@       �	�*��Xc�A�	*

lossT�=��n       �	v���Xc�A�	*

lossüH=��M;       �	GY��Xc�A�	*

lossj��=�3       �	J��Xc�A�	*

loss�=��>       �	
���Xc�A�	*

loss��=�j!�       �	3��Xc�A�	*

lossc��=Ȧ	�       �	����Xc�A�	*

lossM�<x�vd       �	�\��Xc�A�	*

loss���=�2��       �	����Xc�A�	*

loss(a'=0�       �	���Xc�A�	*

loss�v=Q��	       �	�&��Xc�A�	*

lossM-�=��0U       �	1'��Xc�A�	*

loss��=`d�n       �	8���Xc�A�	*

loss�4>�}�;       �	R��Xc�A�	*

loss�˷=��}�       �	����Xc�A�	*

loss�$�=�"�       �	9|��Xc�A�	*

loss�}=*���       �	���Xc�A�	*

loss�,=���+       �	X���Xc�A�	*

lossF��=��D       �	NF��Xc�A�	*

loss�U>��p       �	����Xc�A�	*

lossX|�=Xi��       �	�z��Xc�A�	*

loss�%.=��`       �	n��Xc�A�	*

loss��=�a       �	.���Xc�A�	*

loss��>@�	6       �	 E��Xc�A�	*

loss���=ߞ�       �	����Xc�A�	*

loss���=[t-�       �	�{��Xc�A�	*

lossoȡ=����       �	���Xc�A�	*

loss��;��F       �	K���Xc�A�	*

lossi=�oRd       �	�[��Xc�A�	*

loss��=�N��       �	����Xc�A�	*

loss��(>�g�?       �	й��Xc�A�	*

loss)��=��       �	&Q��Xc�A�	*

loss���=q��h       �	=���Xc�A�	*

loss��2=��       �	���Xc�A�	*

loss���= g�       �	�O��Xc�A�	*

lossϬ4=��z       �	����Xc�A�	*

loss̛�<���R       �	����Xc�A�	*

loss�Sj=��`!       �	�)��Xc�A�	*

loss1I=G�       �	g���Xc�A�	*

loss��c=�1p$       �		l��Xc�A�	*

loss�?�=�n�       �	���Xc�A�	*

loss��=m^��       �	$��Xc�A�	*

loss��=�=��       �	����Xc�A�	*

loss��'=��{1       �	7���Xc�A�	*

loss(�w>�ݪ�       �	�x��Xc�A�	*

loss���<&��:       �	���Xc�A�	*

loss��f=�T��       �	ʥ��Xc�A�	*

loss�"�<���       �	u:��Xc�A�	*

loss�/�=˄��       �	���Xc�A�	*

loss�_x>���       �	'g��Xc�A�	*

loss��=�:�C       �	����Xc�A�	*

loss�P=���       �	g���Xc�A�	*

lossq�=i�_�       �	�3��Xc�A�	*

loss�Ҁ<� 3       �	F���Xc�A�	*

loss���=3)в       �	�m��Xc�A�	*

losstw=���       �	���Xc�A�	*

loss���=����       �	y���Xc�A�	*

loss���<��æ       �	�K��Xc�A�	*

loss�b1=-T�O       �	c���Xc�A�	*

loss�k=��ɀ       �	����Xc�A�	*

lossܡ�<P�       �	y ��Xc�A�	*

loss�%B=�+)       �	t���Xc�A�	*

loss	q�=�΁       �	(���Xc�A�	*

lossQ=�.��       �	���Xc�A�	*

loss_�>���       �	ظ��Xc�A�	*

lossQ}&=smP�       �	R��Xc�A�	*

lossI��;��       �	B���Xc�A�	*

loss��S<7Dg�       �	���Xc�A�	*

lossJd�<?��       �	���Xc�A�	*

loss2!�=�u~�       �	����Xc�A�	*

loss�y�=��b?       �	HP �Xc�A�	*

loss>���       �	-� �Xc�A�	*

loss���=�?G�       �	���Xc�A�	*

lossہ<���       �	�K�Xc�A�	*

lossV��<yG��       �	��Xc�A�	*

loss��=ku	�       �	��Xc�A�	*

loss7Q>�ˁ       �	cG�Xc�A�	*

loss��Z=@��       �	~��Xc�A�	*

loss���=Wl��       �	x}�Xc�A�	*

lossV7>���       �	�Xc�A�	*

loss#�>o��'       �	Ū�Xc�A�	*

loss�Ń=؀=N       �	l@�Xc�A�	*

loss�W0=��d       �	���Xc�A�	*

loss?]�=�~��       �	�m�Xc�A�	*

loss�g=Md��       �	�	�Xc�A�	*

loss��l=�˙        �	؝	�Xc�A�	*

lossU(=L��O       �	�3
�Xc�A�	*

loss^��=xA�       �	��
�Xc�A�	*

loss8��=�R       �	�e�Xc�A�	*

loss��<엒�       �	q �Xc�A�	*

loss;��=�\A|       �	��Xc�A�
*

loss3G�=����       �	V��Xc�A�
*

lossjؕ=�.؝       �	X�Xc�A�
*

lossi�=����       �	ܹ�Xc�A�
*

loss�7=���       �	rP�Xc�A�
*

lossTY=y�[`       �	���Xc�A�
*

lossk�=ؠ)�       �	z��Xc�A�
*

loss�Q=��       �	x�Xc�A�
*

loss� b=�|d       �	'�Xc�A�
*

lossa=��       �	���Xc�A�
*

lossw�=��S       �	�g�Xc�A�
*

lossAT=��qY       �	��Xc�A�
*

loss��R=60{       �	h��Xc�A�
*

loss��K=y�2�       �	�K�Xc�A�
*

lossZ��=�̡�       �	���Xc�A�
*

loss�L�=G΄       �	G��Xc�A�
*

losswb$<��w�       �	�/�Xc�A�
*

loss�~=H(       �	���Xc�A�
*

lossm��</�L       �	+��Xc�A�
*

loss� >#��w       �	��Xc�A�
*

loss
ŀ=���j       �	>��Xc�A�
*

lossϼ�<����       �	�h�Xc�A�
*

losst<=`��       �	� �Xc�A�
*

loss%��=��i�       �	5��Xc�A�
*

loss���<��        �	�c�Xc�A�
*

lossfaO=*�<Q       �	��Xc�A�
*

loss��=�˾       �	{��Xc�A�
*

lossGh=AI       �	 T�Xc�A�
*

loss��/=.W\       �	���Xc�A�
*

loss��>�Dq       �	[��Xc�A�
*

loss���=k�t�       �	�7 �Xc�A�
*

loss)%�=��[       �	^� �Xc�A�
*

loss��&=�̊H       �	�z!�Xc�A�
*

loss�V�=�;%-       �	�"�Xc�A�
*

lossPD<Z^��       �	�"�Xc�A�
*

losshR=�Ր�       �	LP#�Xc�A�
*

loss�<=�"�       �	��#�Xc�A�
*

loss_5D=��&�       �	�$�Xc�A�
*

loss���=ޘ#       �	1%�Xc�A�
*

loss��"=<Z�       �	��%�Xc�A�
*

loss��O<��'�       �	�&�Xc�A�
*

loss8u=?���       �	�C'�Xc�A�
*

loss���=���       �	��'�Xc�A�
*

loss\�> !       �	�(�Xc�A�
*

losse�=�u�f       �	� )�Xc�A�
*

loss��<<}<M       �	u�)�Xc�A�
*

loss�E�<#p�)       �	�j*�Xc�A�
*

loss?�=�k�O       �	|+�Xc�A�
*

loss��=>,�)       �	 �+�Xc�A�
*

loss��=��/9       �	�G,�Xc�A�
*

loss��Q=w�yu       �	��,�Xc�A�
*

losslB>�EE       �	:�-�Xc�A�
*

loss;m=��       �	@L.�Xc�A�
*

loss�&=����       �	�.�Xc�A�
*

lossV�<�O^�       �	�/�Xc�A�
*

loss��=?���       �	�0�Xc�A�
*

loss���=ti�l       �	ȶ0�Xc�A�
*

losss
 =�XW�       �	"S1�Xc�A�
*

lossj(=��}       �	��1�Xc�A�
*

loss��<���       �	j�2�Xc�A�
*

loss3+.<_^��       �	\3�Xc�A�
*

lossq�<��'�       �	��3�Xc�A�
*

lossR��=�ch�       �	�4�Xc�A�
*

loss��1=łQ       �	2<5�Xc�A�
*

loss:�<�//       �	�5�Xc�A�
*

loss�!>���       �	c|6�Xc�A�
*

lossSQ�<��qL       �	�7�Xc�A�
*

loss��=�)       �	r�7�Xc�A�
*

loss��<h�)0       �	�W8�Xc�A�
*

loss���<f�       �	|�8�Xc�A�
*

lossE�<uWp        �	��9�Xc�A�
*

lossT�<2��       �	{/:�Xc�A�
*

loss�w=��#       �	��:�Xc�A�
*

loss���=6�       �	��;�Xc�A�
*

loss^�=�FK�       �	"R<�Xc�A�
*

losst�n=^��       �	o�<�Xc�A�
*

loss!�1<qp4�       �	k�=�Xc�A�
*

loss5�	=����       �	@>�Xc�A�
*

losst��=�'       �	q�>�Xc�A�
*

loss�2�=n�)f       �		�?�Xc�A�
*

lossZ��=)P�        �	Q0@�Xc�A�
*

loss,��=�U�       �	��@�Xc�A�
*

loss�3=r�[       �	�qA�Xc�A�
*

loss�V=�q�       �	B�Xc�A�
*

loss���<޹e�       �	��B�Xc�A�
*

losswY =G��       �	jiC�Xc�A�
*

loss��?<>�J       �	�D�Xc�A�
*

loss�/8<&�V       �	ΦD�Xc�A�
*

lossZ'�=(���       �	�BE�Xc�A�
*

loss��O=����       �	��E�Xc�A�
*

loss7%'=���	       �	KxF�Xc�A�
*

loss���=Kt       �	 G�Xc�A�
*

loss�:�<x59       �	]�G�Xc�A�
*

loss���<�X�       �	_H�Xc�A�
*

loss��=!ȓ�       �	M,I�Xc�A�
*

lossT��;)�.       �	��I�Xc�A�
*

loss�/�="���       �	�bJ�Xc�A�
*

loss�"�=���l       �	e�J�Xc�A�
*

loss��=�8��       �	_�K�Xc�A�
*

lossa=�X��       �	�KL�Xc�A�
*

loss�O<�H�H       �	=�L�Xc�A�
*

loss~<�Gf       �	7�M�Xc�A�
*

loss�:X=�:r       �	J`N�Xc�A�
*

loss���=6�F
       �	+�N�Xc�A�
*

loss���<��SH       �	ԜO�Xc�A�
*

loss2�?=���	       �	�9P�Xc�A�
*

lossb;<R(W       �	N�P�Xc�A�
*

loss�_�=�{�       �	�sQ�Xc�A�
*

loss�:�<e��`       �	�R�Xc�A�
*

loss�!�=d��       �	9�R�Xc�A�
*

loss���=w�}w       �	�zS�Xc�A�
*

loss��=�`z�       �	�+T�Xc�A�
*

lossx[�<�L�       �	@U�Xc�A�
*

loss$�<�p       �	o,V�Xc�A�
*

loss
&�<��T       �	}#W�Xc�A�
*

lossZP
<��z       �	�W�Xc�A�
*

lossg�=>Mv       �	�gX�Xc�A�
*

loss(�=�@�       �	5)Y�Xc�A�
*

loss�c=.�u       �	��Y�Xc�A�
*

lossF>��W�       �	��Z�Xc�A�
*

lossc%=j�	�       �	��[�Xc�A�
*

lossͯ=WWJ�       �	:�\�Xc�A�
*

loss�p= ���       �	��]�Xc�A�
*

loss�\J=c7�       �	Ve^�Xc�A�
*

losst=Kp�       �	�_�Xc�A�
*

loss�[�<�{��       �	��_�Xc�A�
*

loss3��<��ڀ       �	?8`�Xc�A�
*

loss�M=�+�       �	Z�`�Xc�A�
*

loss��=.�#       �	Jya�Xc�A�*

loss��N=��#       �	�!b�Xc�A�*

loss{�=��Z�       �	��b�Xc�A�*

lossz-)=���       �	�tc�Xc�A�*

loss��=sN�       �	�d�Xc�A�*

loss�l&<{�g8       �	½d�Xc�A�*

loss�b=U$��       �	�_e�Xc�A�*

loss&�=yK��       �	qf�Xc�A�*

lossM�=�P"Q       �	=�f�Xc�A�*

loss#��=��g�       �	fIg�Xc�A�*

loss���=��?�       �	��g�Xc�A�*

loss^w�=H��       �	Ƌh�Xc�A�*

loss��b>�2�       �	(i�Xc�A�*

loss�H]<��]$       �	��i�Xc�A�*

lossj-U=�IJ�       �	�dj�Xc�A�*

lossLd�=�       �	�k�Xc�A�*

lossn�=H���       �	Ϡk�Xc�A�*

loss�K�<��       �	<l�Xc�A�*

loss��<S� U       �	�l�Xc�A�*

loss��=��P�       �	?tm�Xc�A�*

loss�2�=ix       �	�n�Xc�A�*

loss�ν=]��       �	c�n�Xc�A�*

loss|J�=���       �	?Wo�Xc�A�*

loss!��<�E�5       �	�4p�Xc�A�*

loss�O�<�9�       �	Z�p�Xc�A�*

lossfͥ=��6       �	t�q�Xc�A�*

lossj�e=ټ�4       �	�9r�Xc�A�*

lossG��=�͝�       �	��r�Xc�A�*

loss���=�\SA       �	$�s�Xc�A�*

loss_2:=��P�       �	vt�Xc�A�*

lossTdb=���       �	��t�Xc�A�*

loss���=�0P�       �	�ru�Xc�A�*

losslcs<B�(       �	8v�Xc�A�*

loss�<�<�vQf       �	��v�Xc�A�*

lossN71=�td�       �	 Ww�Xc�A�*

loss7?�=���       �	��w�Xc�A�*

loss�"%=Yw��       �	Ōx�Xc�A�*

lossn��=�0Ox       �	�,y�Xc�A�*

loss�&Q=��.       �	��y�Xc�A�*

loss�]<UIM       �	��z�Xc�A�*

lossx�5=���       �	�{�Xc�A�*

loss1+�=PV>�       �	��{�Xc�A�*

lossݛS=����       �	�|�Xc�A�*

loss6�>c�       �	�-}�Xc�A�*

lossA=i=.�Qa       �	l�}�Xc�A�*

loss��=���       �	mt~�Xc�A�*

loss[��;)�       �	�!�Xc�A�*

loss���<jr�Y       �	U��Xc�A�*

lossu�=��N�       �	)_��Xc�A�*

loss�T=9`>�       �	)	��Xc�A�*

loss��=iuv       �	>���Xc�A�*

loss�Z,=_�!       �	PT��Xc�A�*

lossoڸ=�_��       �	����Xc�A�*

lossFW�=��y�       �	Q���Xc�A�*

loss}U�<@�       �	[?��Xc�A�*

loss��>��Ų       �	���Xc�A�*

loss�=���       �	���Xc�A�*

loss��'=�dW       �	{-��Xc�A�*

lossx0=ƻ2       �	-Ά�Xc�A�*

loss���=Q�Q�       �	jl��Xc�A�*

loss<0i=�~��       �	���Xc�A�*

loss)=�=]J��       �	>���Xc�A�*

loss�ڰ=�~[-       �	�[��Xc�A�*

lossO�A=��=       �	���Xc�A�*

lossg=V�0       �	`ʊ�Xc�A�*

loss4��=R}��       �	7o��Xc�A�*

loss:8!=�A-       �	���Xc�A�*

loss
v2>��       �	@���Xc�A�*

loss�%E>���       �	d��Xc�A�*

lossTr=���       �	/��Xc�A�*

loss��<MK4       �	����Xc�A�*

loss�;�=�;��       �	�[��Xc�A�*

loss��<�k[       �	�@��Xc�A�*

lossN~�<�U+       �	/ߐ�Xc�A�*

loss���<�H&       �	���Xc�A�*

loss��=+Ip�       �	:��Xc�A�*

loss)�n=�"vH       �	�ڒ�Xc�A�*

lossr�d=�&�       �	���Xc�A�*

loss�h�=aig,       �	�3��Xc�A�*

loss�`"=�7�       �	�Ҕ�Xc�A�*

loss4�O<�m��       �	w��Xc�A�*

loss}�=}=C�       �	�5��Xc�A�*

loss���<Θ:N       �	���Xc�A�*

loss��k<"~�{       �	���Xc�A�*

loss�X=�K       �	hz��Xc�A�*

lossj֨<^�AT       �	$C��Xc�A�*

loss�j�;�KKT       �	����Xc�A�*

lossx��<�+2       �	�˚�Xc�A�*

lossr��=wԌ�       �	#���Xc�A�*

loss)'�<�S4�       �	�V��Xc�A�*

loss/��=��       �	g��Xc�A�*

loss��=�Y#�       �	����Xc�A�*

loss��=G3�       �	B"��Xc�A�*

loss8�=�$
�       �	����Xc�A�*

loss�9+=��i       �	=���Xc�A�*

loss$��<�cs�       �	���Xc�A�*

loss|��=O�X�       �	���Xc�A�*

lossc�=d�GS       �	{K��Xc�A�*

loss�D>���n       �	i��Xc�A�*

loss@�?<���       �	_|��Xc�A�*

loss ��=��,�       �	JD��Xc�A�*

lossdr=��       �	G��Xc�A�*

loss��=0�d�       �	���Xc�A�*

loss�]�<Sz��       �	 ��Xc�A�*

loss��=#ݳ       �	ᶥ�Xc�A�*

loss�V=O��       �	iT��Xc�A�*

loss�1(<��ݖ       �	���Xc�A�*

loss��<S��+       �	����Xc�A�*

loss�Q=<��r       �	~;��Xc�A�*

loss�=�'�       �	�ب�Xc�A�*

loss�=����       �	Ot��Xc�A�*

loss���=v�M�       �	Z��Xc�A�*

loss��H=���       �	���Xc�A�*

loss
^=x��[       �	�O��Xc�A�*

loss�*�=��̼       �	h��Xc�A�*

lossM��=��Tp       �	臬�Xc�A�*

lossiw7=����       �	���Xc�A�*

loss�m[=�
��       �	'���Xc�A�*

loss���=�7�       �	�Y��Xc�A�*

loss�'=�޹       �	����Xc�A�*

loss�kK=q� �       �	����Xc�A�*

loss�;(=0'�       �	�G��Xc�A�*

loss8��=�}|�       �	l���Xc�A�*

loss���=�ׅ       �	����Xc�A�*

lossfR�<�p       �	Z*��Xc�A�*

loss�rF<��7�       �	.˲�Xc�A�*

loss��*=���       �	�e��Xc�A�*

loss���=���m       �	���Xc�A�*

loss��=*��k       �	ࢴ�Xc�A�*

lossH@�<��       �	_E��Xc�A�*

loss�y=^%�       �	P��Xc�A�*

loss���=�        �	h���Xc�A�*

loss��J<�6�       �	�O��Xc�A�*

lossdD<��=�       �	,��Xc�A�*

loss�ZT=�q�       �	���Xc�A�*

loss��[=��P�       �	�"��Xc�A�*

loss`��<{q0'       �	����Xc�A�*

loss�L>�Vh       �	�X��Xc�A�*

loss��>��ٓ       �	(��Xc�A�*

loss�v<<D*�       �	'���Xc�A�*

loss��=��C�       �	G ��Xc�A�*

loss#=�K1       �	���Xc�A�*

loss�2=�s6A       �	HP��Xc�A�*

loss���<�j�T       �	W��Xc�A�*

lossa�Q=w��       �	
���Xc�A�*

loss.��<Q[�9       �	���Xc�A�*

loss�j[=nķ-       �	����Xc�A�*

loss��=<5c=       �	EK��Xc�A�*

lossXփ<��>       �	C���Xc�A�*

lossw&<��$�       �	���Xc�A�*

loss�9�=���       �	���Xc�A�*

loss]�>���,       �	V���Xc�A�*

loss`��=�ٓ�       �	j���Xc�A�*

loss�=�(K9       �	,)��Xc�A�*

loss�=SK�K       �	���Xc�A�*

lossÆ?=�X�       �	����Xc�A�*

loss��=�P�$       �	zP��Xc�A�*

loss��=D��       �	F���Xc�A�*

loss1�W=
ۼ.       �	���Xc�A�*

loss���=�'l�       �	V)��Xc�A�*

loss�Y�;�1��       �	����Xc�A�*

lossÜX=;�^�       �	�^��Xc�A�*

loss�&l=��       �	2��Xc�A�*

loss�x=����       �	ۢ��Xc�A�*

loss���<W��8       �	�K��Xc�A�*

loss��=./g�       �	����Xc�A�*

loss�#�<����       �	�|��Xc�A�*

loss��=O��       �	���Xc�A�*

lossr�,>{u�       �	g���Xc�A�*

loss���=�*�~       �	fL��Xc�A�*

loss�K=����       �	���Xc�A�*

lossjF=���=       �	����Xc�A�*

loss�X�=�*"j       �	�[��Xc�A�*

lossb�=����       �	v���Xc�A�*

loss3�<��a       �	R���Xc�A�*

lossV��<��K�       �	5��Xc�A�*

loss��=(q�:       �	����Xc�A�*

lossW8e<LR^       �	�s��Xc�A�*

loss�f�<;��       �	�O��Xc�A�*

loss�o�=����       �	I���Xc�A�*

loss�1B=��}�       �	����Xc�A�*

lossߛ%=5�^       �	���Xc�A�*

loss �V=s0       �	;5��Xc�A�*

lossQ�=ɫ�{       �	D���Xc�A�*

lossm�e=+�5�       �	����Xc�A�*

loss���<�כ       �	hy��Xc�A�*

loss|;=���       �	���Xc�A�*

lossd��=�,(       �	2��Xc�A�*

loss�=��@       �	+���Xc�A�*

loss�H�;�|W�       �	r��Xc�A�*

lossq�<<-n�       �	.U��Xc�A�*

loss8e�<@��       �	�(��Xc�A�*

loss��=��/T       �	����Xc�A�*

loss6��<�a#l       �	
���Xc�A�*

loss�%�=6�       �	����Xc�A�*

loss���=� 5�       �	�Y��Xc�A�*

loss��J<�l�H       �	��Xc�A�*

loss���<�$       �	*���Xc�A�*

loss��=�͙�       �	+Q��Xc�A�*

loss�Ê=ģj�       �	N���Xc�A�*

loss��>Af�       �	����Xc�A�*

loss%�f=�W       �	�?��Xc�A�*

loss,�)>vEz       �	���Xc�A�*

loss	��=��$       �	���Xc�A�*

loss��>Jr>I       �	Z��Xc�A�*

loss5D=��%�       �	t
��Xc�A�*

lossx��=H�       �	r���Xc�A�*

loss�X�=�^�       �	C<��Xc�A�*

loss�+Q=dmX�       �	g���Xc�A�*

loss��<�e<Q       �	�l��Xc�A�*

loss��<�}�        �	���Xc�A�*

loss��<��h       �	����Xc�A�*

loss֠=C�,�       �	�3��Xc�A�*

loss6b=&;��       �	����Xc�A�*

loss��c=w�O       �	�m��Xc�A�*

loss��P=ؾ�       �	5	��Xc�A�*

loss�"I>�^O�       �	a���Xc�A�*

lossn&q=�j       �	)A��Xc�A�*

lossL(�=��w�       �	���Xc�A�*

loss&�w=��K       �	|~��Xc�A�*

loss��=|;��       �	%$��Xc�A�*

lossLE�=����       �	���Xc�A�*

loss�,k<���_       �	j��Xc�A�*

loss�=N�+�       �	G��Xc�A�*

loss�� =I��D       �	���Xc�A�*

loss�s�<'_��       �	W@��Xc�A�*

lossHAg<ƨcA       �	���Xc�A�*

loss,}�=��"�       �	}u��Xc�A�*

lossl�F=���        �	���Xc�A�*

lossʘ�<�aa$       �	]���Xc�A�*

loss��<����       �	5@��Xc�A�*

loss��<�#��       �	���Xc�A�*

loss��=��q_       �	z���Xc�A�*

loss���=��H.       �	�>��Xc�A�*

lossL�>��0       �	���Xc�A�*

loss���=p5�       �	�k��Xc�A�*

loss4��=��6       �	P��Xc�A�*

loss�(�<��+�       �	����Xc�A�*

loss�{�=#:�       �	UN��Xc�A�*

loss�Vc>�V       �	����Xc�A�*

loss�4;=����       �	���Xc�A�*

loss���= �r�       �	*��Xc�A�*

loss�a�=c��       �	����Xc�A�*

loss�_D=E���       �	�C��Xc�A�*

loss��<����       �	 ���Xc�A�*

loss�;4<��d       �	�� �Xc�A�*

loss�w�=w��       �	! �Xc�A�*

lossA�=��O       �	j��Xc�A�*

losso,=���(       �	OZ�Xc�A�*

loss�\=Cc�       �	'��Xc�A�*

loss���=�t�       �	���Xc�A�*

loss+�<�Ӧ       �	2<�Xc�A�*

lossf��=��%       �	{��Xc�A�*

lossiL=U���       �	7p�Xc�A�*

lossc��<݂s�       �	-�Xc�A�*

loss7[<�r�Z       �	/��Xc�A�*

loss�\8=���       �	u>�Xc�A�*

loss��=<��       �	f��Xc�A�*

loss�=�N4�       �	���Xc�A�*

loss��=��u       �	Tt	�Xc�A�*

lossm֦<�?��       �	<
�Xc�A�*

loss];=�c       �	oK�Xc�A�*

lossX�Q<G'�       �	y��Xc�A�*

loss���=���       �	��Xc�A�*

loss��8=��N�       �	�&�Xc�A�*

loss
�=SI�       �	L��Xc�A�*

lossE�D>M�ނ       �	S^�Xc�A�*

loss�Hw<}w:       �	� �Xc�A�*

loss��<��D{       �	ݚ�Xc�A�*

loss�	q=��K�       �	À�Xc�A�*

lossę�<��R       �	��Xc�A�*

lossH��<�N$�       �	 ��Xc�A�*

loss3�0>t�t�       �	�R�Xc�A�*

loss�C2>�D�y       �	D��Xc�A�*

losseܣ=@��       �	���Xc�A�*

loss�**=�{�       �	m<�Xc�A�*

loss��=���       �	8�Xc�A�*

loss@�_=UȺ�       �	���Xc�A�*

loss$"�<T�7!       �	l�Xc�A�*

lossɐa<�\�       �	��Xc�A�*

lossM�=�;�       �	���Xc�A�*

lossq��<�W��       �	��Xc�A�*

lossDi�=}DE'       �	|C�Xc�A�*

loss�m|=���       �	{��Xc�A�*

lossw�==�pE       �	�t�Xc�A�*

lossϦ'<[J�3       �	�
�Xc�A�*

losst҆<@�"       �	'��Xc�A�*

loss��P<���[       �	�6�Xc�A�*

loss�Mv=����       �	���Xc�A�*

loss�>j��<       �	�y�Xc�A�*

loss\,{<���F       �	��Xc�A�*

loss.��<���        �	]��Xc�A�*

loss�k�=w��       �	�_ �Xc�A�*

loss���=䋹{       �	� �Xc�A�*

loss�*}<b�[
       �	6�!�Xc�A�*

loss�p4=њoY       �	�("�Xc�A�*

loss�&�<ʗ6�       �	^�"�Xc�A�*

loss�;�=�U       �	DQ#�Xc�A�*

loss�R,=�t�T       �	!�#�Xc�A�*

lossî�=���       �	�$�Xc�A�*

loss�d=B�'�       �	�%�Xc�A�*

loss�=��s�       �	&�%�Xc�A�*

loss���=���       �	@&�Xc�A�*

loss��<���       �	4�&�Xc�A�*

loss�K)=!ޟ       �	Ul'�Xc�A�*

loss,��<4�/�       �	 (�Xc�A�*

lossc��=$7.       �	��(�Xc�A�*

lossd�;=*���       �	�g)�Xc�A�*

loss ٦=a��       �	��)�Xc�A�*

loss�r=�+pE       �	�*�Xc�A�*

lossc=���5       �	�.+�Xc�A�*

loss}�=��       �	��+�Xc�A�*

lossA<�=X��v       �	��,�Xc�A�*

lossڮ:<��d�       �	1%-�Xc�A�*

loss�p{=���       �	n�-�Xc�A�*

loss�@�<�y�_       �	\q.�Xc�A�*

lossm4=!�?�       �	/�Xc�A�*

loss� =U[�I       �	��/�Xc�A�*

loss4�H=;\�       �	�0�Xc�A�*

loss�1U<[l�Z       �	�"1�Xc�A�*

loss{1%=S���       �	l�1�Xc�A�*

loss�^�=�$)�       �	}y2�Xc�A�*

loss�JW=���       �	B$3�Xc�A�*

lossy�=�-�F       �	��3�Xc�A�*

loss��2=G��       �	Tn4�Xc�A�*

loss�%�;��OZ       �	m5�Xc�A�*

loss	�b;\n��       �	ض5�Xc�A�*

loss2j<bк       �	U6�Xc�A�*

lossT��;���K       �	��6�Xc�A�*

lossJV<�@�       �	Ã7�Xc�A�*

loss\�;� �y       �	�,8�Xc�A�*

loss���<Lr��       �	}�8�Xc�A�*

loss��=��Xz       �	Yl9�Xc�A�*

loss��1<&wf�       �	�:�Xc�A�*

loss�:t��       �	��:�Xc�A�*

loss�  :�a��       �	"q;�Xc�A�*

loss.�]<�C�       �	�<�Xc�A�*

loss
Nz=O��       �	P�<�Xc�A�*

loss=��<:r��       �		o=�Xc�A�*

loss�\';Y:+o       �	0>�Xc�A�*

lossĸ
<�A{.       �	�>�Xc�A�*

lossR�V>�@       �	a?�Xc�A�*

loss �/<�U��       �	R
@�Xc�A�*

loss�6�<�fL       �	�@�Xc�A�*

loss��t=����       �	S^A�Xc�A�*

loss7�=7���       �	ULB�Xc�A�*

loss���<�)}�       �	��B�Xc�A�*

loss�.<I��C       �	��C�Xc�A�*

losssQ�=�؜�       �	�QD�Xc�A�*

loss�o=��r�       �	�D�Xc�A�*

loss���=��N       �	�E�Xc�A�*

loss#-[=ۃF�       �	�BF�Xc�A�*

loss�l�<-��f       �	��F�Xc�A�*

loss��>2pk'       �	φG�Xc�A�*

loss�dc=1-h       �	U1H�Xc�A�*

loss��=�+Rl       �	^�H�Xc�A�*

loss�=p2��       �	��I�Xc�A�*

loss�=�[       �	,.J�Xc�A�*

loss��=��MJ       �	(�J�Xc�A�*

loss���<��Ԉ       �	�xK�Xc�A�*

loss��=Z^��       �	K�L�Xc�A�*

loss��=���9       �	ׇM�Xc�A�*

loss	q<�ȉ       �	�3N�Xc�A�*

lossdރ=��Я       �	t�N�Xc�A�*

loss]
�=���!       �	�sO�Xc�A�*

loss�}=-��       �	�P�Xc�A�*

loss`ψ<�aW�       �	��P�Xc�A�*

loss��<�M6       �	�WQ�Xc�A�*

loss�:�<��ߕ       �	�R�Xc�A�*

loss���<�>Q�       �	@�R�Xc�A�*

loss��=;�*�       �	FS�Xc�A�*

loss{8m=]��       �	��S�Xc�A�*

loss|�=A|�-       �	��T�Xc�A�*

loss��I=Lj�-       �	J|U�Xc�A�*

loss��i<�o       �	U0V�Xc�A�*

loss���<�2iq       �	-�V�Xc�A�*

loss�'�<*zq       �	GX�Xc�A�*

loss�4=Z��       �	�Y�Xc�A�*

loss�J�<o��       �	e�Y�Xc�A�*

loss�F�=�]�^       �	`rZ�Xc�A�*

loss�ĵ= ��       �	6[�Xc�A�*

loss2�<>��       �	��[�Xc�A�*

loss�rR=�{;n       �	6s\�Xc�A�*

lossա�=��J       �	�']�Xc�A�*

loss!4�<д8c       �	��]�Xc�A�*

loss��=UX��       �	[�^�Xc�A�*

lossVh=$!�       �	{�_�Xc�A�*

lossA_0=6�y       �	<1`�Xc�A�*

loss��=�:�       �	T�`�Xc�A�*

loss�=_�@       �	2�a�Xc�A�*

loss蔌=�[z       �	�1b�Xc�A�*

loss�y8<�5A�       �	��b�Xc�A�*

lossx��<H
�       �	�wc�Xc�A�*

loss�;�=EB�       �	��|�Xc�A�*

loss���=j�       �	U}�Xc�A�*

loss�>4Վ'       �	��}�Xc�A�*

lossd�=4-2P       �	�~�Xc�A�*

lossOw�=7$l       �	T8�Xc�A�*

loss*�<�ͦ�       �	>��Xc�A�*

loss��=1;q       �	=e��Xc�A�*

loss�= l       �	���Xc�A�*

loss���=I^�P       �	*���Xc�A�*

loss��=ARD       �	�T��Xc�A�*

lossH�k<���       �	����Xc�A�*

loss�vQ=J�       �	Й��Xc�A�*

loss��<��sR       �	/��Xc�A�*

loss�K�=�9H       �	�Ǆ�Xc�A�*

loss���=v�_s       �	sc��Xc�A�*

loss�g>�C�Z       �	����Xc�A�*

loss��?;	D       �	����Xc�A�*

loss�H=�;>Z       �	M,��Xc�A�*

loss�w>=d�"=       �	pЇ�Xc�A�*

loss;��=�n��       �	�u��Xc�A�*

lossT܁=��3       �	"��Xc�A�*

loss��>����       �	����Xc�A�*

loss:[M=m��@       �	�J��Xc�A�*

loss�.f=��	�       �	@ߊ�Xc�A�*

lossAK=e��?       �	䃋�Xc�A�*

lossfK=�.�       �	j��Xc�A�*

loss�ߧ=v���       �	����Xc�A�*

lossS�=��l�       �	%X��Xc�A�*

loss��=R���       �	����Xc�A�*

lossy\=z ��       �	����Xc�A�*

lossF�=-��N       �	b���Xc�A�*

loss�=��4W       �	sJ��Xc�A�*

loss���<���       �	���Xc�A�*

loss-��=��       �	����Xc�A�*

loss�]P=;�6#       �	M-��Xc�A�*

lossw�&>�:�V       �	JҒ�Xc�A�*

loss��<R1�1       �	bi��Xc�A�*

loss��=�46�       �	%��Xc�A�*

loss��R>�9Y�       �	�F��Xc�A�*

loss���=�3       �	�	��Xc�A�*

loss)�0=[vO�       �	TƖ�Xc�A�*

loss�<=K��u       �	-���Xc�A�*

loss�>k<xٽ�       �	���Xc�A�*

loss!�)=Ao��       �	9B��Xc�A�*

loss���<m9��       �	�W��Xc�A�*

loss���=}�)X       �	ʧ��Xc�A�*

loss�PX=���i       �	�v��Xc�A�*

loss���<4��4       �	j/��Xc�A�*

lossmIL=��;�       �	���Xc�A�*

loss��;R*8       �	q���Xc�A�*

loss��j<�3��       �	�T��Xc�A�*

loss�F�=��       �	`��Xc�A�*

loss�;=ta�       �	+���Xc�A�*

loss�p>�6��       �	���Xc�A�*

lossX�2=p�m       �	ࡢ�Xc�A�*

loss�%<���       �	J��Xc�A�*

loss��#;ME�       �	0��Xc�A�*

loss߃�<�#�       �	ލ��Xc�A�*

loss�'<��       �	.��Xc�A�*

loss!�D=y�ZA       �	�ǥ�Xc�A�*

lossi1�=��G       �	�v��Xc�A�*

loss�& =�$�L       �	��Xc�A�*

loss}��<���m       �	Ƨ�Xc�A�*

loss�z�<+웨       �	�b��Xc�A�*

lossט�=(�1       �	�
��Xc�A�*

loss��&=x�       �	����Xc�A�*

loss��=.�"$       �	Pm��Xc�A�*

loss3�=Ցw       �	���Xc�A�*

lossk�=�u.�       �	 ī�Xc�A�*

loss��=a��       �	�^��Xc�A�*

loss�O3=H�W�       �	���Xc�A�*

loss�{�<����       �	����Xc�A�*

loss��=���a       �	2��Xc�A�*

loss��=7,p,       �	�Ǯ�Xc�A�*

lossͧ�=�1/�       �	 ^��Xc�A�*

loss$�<�`�7       �	����Xc�A�*

loss�b=Ct��       �	F���Xc�A�*

loss��<z`�       �	�a��Xc�A�*

loss�c=)MG�       �	� ��Xc�A�*

loss�=<�I�       �	E���Xc�A�*

loss�r�=���       �	`=��Xc�A�*

loss�}�=�       �	�ݳ�Xc�A�*

lossD��=�        �	g}��Xc�A�*

loss1s<��
�       �	k-��Xc�A�*

loss���;�y:�       �	�е�Xc�A�*

loss���<a�       �	p��Xc�A�*

loss)��<W׏       �	���Xc�A�*

loss�+�<Ag       �	䟷�Xc�A�*

loss���;�Ɠ       �	����Xc�A�*

lossꇃ=� ��       �	�:��Xc�A�*

loss�T�= 熇       �	ѹ�Xc�A�*

loss9�<�ؗ�       �	Pn��Xc�A�*

loss��<��Ȋ       �	m��Xc�A�*

loss*�"=�ā       �	����Xc�A�*

loss�:�=$=%�       �	x���Xc�A�*

lossd8�<ґ��       �	;5��Xc�A�*

lossr�e=.Q=�       �	���Xc�A�*

lossc�a<���       �	M���Xc�A�*

loss�#>>��OS       �	"��Xc�A�*

loss_V�<���2       �	໿�Xc�A�*

lossh�<ft�       �	�V��Xc�A�*

loss)l�<�e-�       �	$���Xc�A�*

loss���<f��K       �	8���Xc�A�*

lossS�N=o�G�       �	L��Xc�A�*

loss_�V=��2)       �	}���Xc�A�*

lossC+=پ[�       �	L��Xc�A�*

lossd��<?�P       �	����Xc�A�*

loss\�<���a       �	4���Xc�A�*

loss�g�=��%       �	/��Xc�A�*

loss�<$<�L       �	X���Xc�A�*

loss�Q>x�Q1       �	od��Xc�A�*

loss	�T=���       �	���Xc�A�*

loss�@t<��(       �	����Xc�A�*

loss4W=]��t       �	�9��Xc�A�*

loss���<��*       �	 ��Xc�A�*

loss�\�<v�4       �	?���Xc�A�*

loss��=K��M       �	�`��Xc�A�*

loss�tu=/�       �	���Xc�A�*

loss-7?=
J�/       �	���Xc�A�*

lossk�<f�Q|       �	I*��Xc�A�*

loss)��<�?&       �	+���Xc�A�*

loss���<�_f�       �	@h��Xc�A�*

loss�qv=�Y[       �	���Xc�A�*

loss��<9�       �	<���Xc�A�*

loss�m!=n%��       �	F��Xc�A�*

loss�	Q=���       �	����Xc�A�*

loss�Er=%^*       �	����Xc�A�*

loss͢<����       �	^*��Xc�A�*

loss4v2=�.�       �	R���Xc�A�*

loss���<���       �	6t��Xc�A�*

lossnu�=�4��       �	�!��Xc�A�*

loss��=Z�T�       �	x���Xc�A�*

loss���<#|       �	Hl��Xc�A�*

lossݹ;/<!       �	�2��Xc�A�*

loss��=��(@       �	���Xc�A�*

loss{=[�       �	Χ��Xc�A�*

loss���=�,fx       �	*R��Xc�A�*

loss��<w3�       �	���Xc�A�*

loss��<4�e       �	L���Xc�A�*

loss<�<T3       �	����Xc�A�*

lossl�;L�u       �	)��Xc�A�*

loss*�H=I��       �	����Xc�A�*

loss�U�<k�%�       �	Ih��Xc�A�*

loss���=��}       �	6��Xc�A�*

loss`G=�Q�       �	=���Xc�A�*

lossdn�<����       �	3��Xc�A�*

loss�	�=.m��       �	u���Xc�A�*

loss�=<q���       �	g��Xc�A�*

loss��;"(>�       �	%��Xc�A�*

lossa΅<k;�       �	����Xc�A�*

loss�t�<��       �	hZ��Xc�A�*

loss|��=��u�       �	����Xc�A�*

lossm�.=���       �	���Xc�A�*

loss�C�<�A	       �	���Xc�A�*

loss�8�=ћ��       �	E���Xc�A�*

lossܥ�;k�B�       �	�_��Xc�A�*

loss`8=d&�]       �	�
��Xc�A�*

loss�ͧ=C
�m       �	���Xc�A�*

loss
�=�d�       �	�w��Xc�A�*

loss�,b<���       �	���Xc�A�*

loss]E== ��       �	q���Xc�A�*

losst�<'�>*       �	�G��Xc�A�*

loss�T*=�O�       �	����Xc�A�*

loss��C<<@�       �	����Xc�A�*

loss��;=2%�+       �	-#��Xc�A�*

loss��<��7r       �	����Xc�A�*

loss隄<9g+�       �	z���Xc�A�*

lossU�=��y�       �	�@��Xc�A�*

loss�0�=ꚹ       �	����Xc�A�*

lossXR�<����       �	ݱ��Xc�A�*

loss�U=`ax       �	�K��Xc�A�*

lossf�<稀       �	" ��Xc�A�*

loss��>=���x       �	����Xc�A�*

lossj� >=��p       �	�9��Xc�A�*

loss_n�<���       �	���Xc�A�*

loss��Z=��       �	r��Xc�A�*

lossc5;=�sh       �	{��Xc�A�*

loss;�7=ْ��       �	����Xc�A�*

loss
��<���       �	�P��Xc�A�*

loss���;92��       �	���Xc�A�*

loss6+D<�i       �	���Xc�A�*

loss��=�-w.       �	�!��Xc�A�*

loss�yz<��=       �	ܹ��Xc�A�*

loss�&i<�{��       �	@Q��Xc�A�*

loss�z�<��	#       �	y���Xc�A�*

lossѫ=4߉       �	����Xc�A�*

lossQx�<�K/�       �	)��Xc�A�*

loss��<͞R       �	ۿ��Xc�A�*

lossj�G=筂�       �	Rd��Xc�A�*

loss��k<�sH       �	*��Xc�A�*

loss��u=@��&       �	����Xc�A�*

loss��h<� �z       �	>��Xc�A�*

lossk0=s*�       �	Q���Xc�A�*

loss19=<�%�       �	Tt��Xc�A�*

lossWE�<��#       �	���Xc�A�*

loss��=�A��       �	2���Xc�A�*

loss(&7<�ڀ=       �	T��Xc�A�*

loss ��<�$Oc       �	����Xc�A�*

loss:�6=Q��U       �	����Xc�A�*

lossM8<����       �	�+��Xc�A�*

loss��=�D%�       �	"���Xc�A�*

loss���<z��       �	�W �Xc�A�*

loss�Б=,81       �	� �Xc�A�*

loss�)�=��       �	Ύ�Xc�A�*

lossl��<�ʚ       �	E*�Xc�A�*

loss�~=xϘe       �	���Xc�A�*

loss�Y=��       �	d�Xc�A�*

loss��=ŝz       �	U��Xc�A�*

loss��=-���       �	9��Xc�A�*

loss F=��]�       �	�>�Xc�A�*

loss�$=�Z%       �	���Xc�A�*

lossFY�=���       �	���Xc�A�*

lossH��<{e       �	�_�Xc�A�*

loss���<ó�       �	���Xc�A�*

loss2-j=��2       �	���Xc�A�*

loss�'�=k��       �	1	�Xc�A�*

loss�jf=6ɴ�       �	��	�Xc�A�*

lossRTR>Z[�7       �	�u
�Xc�A�*

loss�>=���=       �	��Xc�A�*

loss��;=�Y7/       �	̵�Xc�A�*

loss��<Y�I�       �	@P�Xc�A�*

loss��<�P�       �	���Xc�A�*

loss^=�(;d       �	��Xc�A�*

loss
!>�       �	:!�Xc�A�*

loss-�=�u}       �	��Xc�A�*

lossXFK=$Iu�       �	�e�Xc�A�*

loss�S9=����       �	X��Xc�A�*

loss��>9$�       �	���Xc�A�*

lossA,3=�@Y�       �	�?�Xc�A�*

loss�>w=k�^{       �	b��Xc�A�*

loss�ޅ<��       �	@l�Xc�A�*

loss��<��~       �	�Xc�A�*

lossVl=�XrM       �	t��Xc�A�*

lossH�:<QA�       �	�4�Xc�A�*

lossS�+=U<�a       �	S��Xc�A�*

loss���<�Y��       �	���Xc�A�*

lossأ�=�sX�       �	��Xc�A�*

loss�u=m�d       �	gb�Xc�A�*

loss�j�=�[R       �	c+�Xc�A�*

lossy�<X�H       �	;��Xc�A�*

lossQJ=�[�       �	��Xc�A�*

loss��*=�z�       �	�M�Xc�A�*

loss��.=���	       �	��Xc�A�*

loss]Q"=�4*�       �	���Xc�A�*

lossz��=��       �	���Xc�A�*

lossT�<L�q	       �	o.�Xc�A�*

loss��f<�θ,       �	.��Xc�A�*

loss�á=m��       �	�r�Xc�A�*

loss��#=R�cY       �	��Xc�A�*

lossJ�"=R�P�       �	��Xc�A�*

loss���=�~��       �	�I �Xc�A�*

loss���=ȭ�       �	�� �Xc�A�*

lossTn�<,1��       �	�!�Xc�A�*

loss;��<�Cz
       �	35"�Xc�A�*

loss��1<���Q       �	-�"�Xc�A�*

loss��k<�oiH       �	�f#�Xc�A�*

loss�u�<���       �	�$�Xc�A�*

loss�W=���       �	3�$�Xc�A�*

lossS�U<ʄ'�       �	;S%�Xc�A�*

loss�O�<��ؙ       �	|�%�Xc�A�*

loss���;����       �	e�&�Xc�A�*

lossW<��¿       �	F''�Xc�A�*

loss\��=O�o�       �	 �'�Xc�A�*

loss�p�<�\NZ       �	�(�Xc�A�*

loss?=�(�U       �	n�)�Xc�A�*

loss��l<�-?       �	_y*�Xc�A�*

lossOi�=�`1%       �	/+�Xc�A�*

loss��$=�F�K       �	�+�Xc�A�*

loss�4>f%
       �	�P,�Xc�A�*

loss���=lѐ7       �	��,�Xc�A�*

loss�v�=7�?       �	7�-�Xc�A�*

lossIN=�k�r       �	�..�Xc�A�*

loss)3=ѩ��       �	��.�Xc�A�*

loss]�=�Ö9       �	�r/�Xc�A�*

loss,n�=�	<�       �	�0�Xc�A�*

loss
��=5�       �	Ƨ0�Xc�A�*

loss9>=���8       �	4�1�Xc�A�*

loss��7=�.�       �	�2�Xc�A�*

loss��=��       �	ϻ2�Xc�A�*

loss�x�=I|�       �	O]3�Xc�A�*

loss2�<��3�       �	��3�Xc�A�*

lossB9=`�ѿ       �	ė4�Xc�A�*

loss�V<+��       �	�<5�Xc�A�*

lossZ2�<��2�       �	�5�Xc�A�*

loss�q�=���       �	*s6�Xc�A�*

lossm��=]� 3       �	�
7�Xc�A�*

loss��N=���y       �	��7�Xc�A�*

losss�<(��       �	�>8�Xc�A�*

loss*�<:9T       �	��8�Xc�A�*

loss��<�P:�       �	�y9�Xc�A�*

loss�$�;�L��       �	B
:�Xc�A�*

loss�$�=J��;       �	��:�Xc�A�*

lossO�;"�S       �	j2;�Xc�A�*

loss;w<���       �	G�;�Xc�A�*

loss�`�<h�8p       �	�o<�Xc�A�*

loss8w�=���L       �	5	=�Xc�A�*

loss{_(=��
       �	�=�Xc�A�*

loss��j=���5       �	�>>�Xc�A�*

lossT~�=A�f�       �	��>�Xc�A�*

lossa>����       �	Ov?�Xc�A�*

loss�Ko=3v|�       �	�Q@�Xc�A�*

loss7�%=��r       �	}�@�Xc�A�*

loss�#=��p       �	�A�Xc�A�*

lossaiI=;��       �	�B�Xc�A�*

loss�)2<�Ѭ       �	��B�Xc�A�*

loss3��=��n�       �	�EC�Xc�A�*

loss/><o���       �	%�C�Xc�A�*

loss=o6=�h       �	.�D�Xc�A�*

loss�+�<�C�       �	wE�Xc�A�*

loss�0�<Z�       �	>yF�Xc�A�*

loss/� =G�o�       �	G�Xc�A�*

lossK�<�)-�       �	Y�G�Xc�A�*

lossA,�=Lp�       �	CH�Xc�A�*

loss�`=��%�       �	Z�H�Xc�A�*

loss���=`       �	�qI�Xc�A�*

loss�=�=�V�       �	F
J�Xc�A�*

loss�	�<P:�q       �	��J�Xc�A�*

lossTj�<1R�       �	W>K�Xc�A�*

loss]��<w�J       �	��K�Xc�A�*

loss��<{�       �	��L�Xc�A�*

loss[=1�       �	�3M�Xc�A�*

lossVQ=��]       �	�M�Xc�A�*

loss�-�<s��       �	NcN�Xc�A�*

loss�)=>G�       �	��N�Xc�A�*

loss� =���       �	��O�Xc�A�*

loss�p=h:�       �	9(P�Xc�A�*

loss��<�M3       �	��P�Xc�A�*

lossX� =c	�       �	�Q�Xc�A�*

lossm��<��hM       �	�+R�Xc�A�*

lossR'�=$�sS       �	��R�Xc�A�*

loss�ս<O�	       �	qYS�Xc�A�*

lossQ�.=�e-�       �	_�S�Xc�A�*

lossZ��<��       �	�T�Xc�A�*

lossWTC<�$�       �	_%U�Xc�A�*

loss�3�=o�*       �	q�U�Xc�A�*

loss|/=����       �	�xV�Xc�A�*

lossl,�<_s�	       �	�4W�Xc�A�*

loss6߹<B�       �	�hX�Xc�A�*

loss21=I�`       �	�JY�Xc�A�*

lossҊ;8��_       �	+Z�Xc�A�*

loss�Ad;!ha�       �	�
[�Xc�A�*

loss�y�=����       �	��[�Xc�A�*

lossJ�=*[�       �	|\�Xc�A�*

losse��<��=�       �	��]�Xc�A�*

lossP>��N�       �	�>^�Xc�A�*

loss�u�=`,3�       �	��^�Xc�A�*

loss�	"<n�dk       �	x_�Xc�A�*

loss�A�=�^�       �	�`�Xc�A�*

loss䉘<�^�       �	ϼ`�Xc�A�*

loss�#	=$��       �	�Za�Xc�A�*

lossF]<����       �	`b�Xc�A�*

loss�&=tʹ*       �	�b�Xc�A�*

loss�#=)^R       �	rNc�Xc�A�*

loss�<�ie       �	`�c�Xc�A�*

loss�T=���T       �	��d�Xc�A�*

loss!�<}���       �	,(e�Xc�A�*

loss��;T�c�       �	�e�Xc�A�*

loss��<IT�       �	�cf�Xc�A�*

loss�z=f�
        �	�Qg�Xc�A�*

loss�Z=Ѯ��       �	}�g�Xc�A�*

loss�^�=mb       �	+�h�Xc�A�*

loss���<���"       �	B!i�Xc�A�*

loss��C=�t��       �	5�i�Xc�A�*

loss}�=V27�       �	�Kj�Xc�A�*

lossn_1=���l       �	��k�Xc�A�*

lossϴ�=���       �	Xrl�Xc�A�*

loss�f�=�`�       �	�m�Xc�A�*

loss��<=H�U       �	��m�Xc�A�*

loss��8=P�       �	�Sn�Xc�A�*

lossj��<GAg�       �	�@o�Xc�A�*

lossůD=V���       �	��o�Xc�A�*

lossz=��       �	t|p�Xc�A�*

loss���<���       �	�"q�Xc�A�*

loss�5<�l�9       �	�q�Xc�A�*

loss��==V�       �	nr�Xc�A�*

loss�;	=�b��       �	�s�Xc�A�*

loss�%=n/�       �	еs�Xc�A�*

loss�rh=	d�       �	XTt�Xc�A�*

loss,=�};�       �	��t�Xc�A�*

losss�=Э�Q       �	��u�Xc�A�*

loss|E=Y��       �	�-v�Xc�A�*

loss���<�Uέ       �	��v�Xc�A�*

loss% �<�o1�       �	:uw�Xc�A�*

loss�j<�2��       �	^x�Xc�A�*

loss��<��>       �	 �x�Xc�A�*

lossH�=-�5        �	�Ly�Xc�A�*

loss��=���       �	�z�Xc�A�*

lossZ�=t�hc       �	��z�Xc�A�*

loss��!=�r�?       �	L{�Xc�A�*

lossֶ�<���U       �	-�{�Xc�A�*

lossoa�=c"^�       �	�|�Xc�A�*

loss��L=�`�%       �	)%}�Xc�A�*

lossr��<Z���       �	��}�Xc�A�*

loss�=��nO       �	�g~�Xc�A�*

loss�%}<!o)�       �	M�Xc�A�*

loss<� >��o       �	;��Xc�A�*

loss�?;ڵ�       �	MH��Xc�A�*

lossD�;��o�       �	���Xc�A�*

loss�<V��       �	���Xc�A�*

lossZ�<�ȑ       �	� ��Xc�A�*

losstF�<����       �	lЂ�Xc�A�*

loss���<0�vR       �	?s��Xc�A�*

loss�=�$       �	fM��Xc�A�*

loss�&�<+_g�       �	����Xc�A�*

lossP�<l�A�       �	.���Xc�A�*

lossS$�<W[fS       �	�&��Xc�A�*

loss��<�{;6       �	N}��Xc�A�*

loss�w = <��       �	���Xc�A�*

loss*�	=�d��       �	��Xc�A�*

lossH�^=u��A       �	\U��Xc�A�*

loss���=z�l       �	@���Xc�A�*

loss�2^=���       �	����Xc�A�*

loss�x�;�8�t       �	�1��Xc�A�*

lossq�R=c�|�       �	�ы�Xc�A�*

lossdL7=�f�       �	2q��Xc�A�*

loss�?=�A��       �	���Xc�A�*

loss��;q��       �	����Xc�A�*

loss,1�;L1�f       �	�J��Xc�A�*

loss@$I=G��       �	���Xc�A�*

losst��=��1       �	x��Xc�A�*

loss�:5<$���       �	���Xc�A�*

lossʲ=���       �	����Xc�A�*

lossj�;��Ѹ       �	�S��Xc�A�*

loss�>����       �	���Xc�A�*

loss��%=�ѽY       �	����Xc�A�*

loss�J&=��J�       �	�0��Xc�A�*

loss,A3=�i&       �	�͓�Xc�A�*

loss8�=ι�       �	si��Xc�A�*

loss��7=>ՙ�       �	���Xc�A�*

loss[�=�?�       �	i���Xc�A�*

loss��=����       �	ӥ��Xc�A�*

loss��'=e`��       �	�P��Xc�A�*

loss̈́�<P��#       �	U��Xc�A�*

lossۢ�;�A��       �	j��Xc�A�*

loss��N=�	�o       �	���Xc�A�*

loss͂[=��lK       �	����Xc�A�*

loss@�M<��       �	�^��Xc�A�*

loss=��;0�       �	���Xc�A�*

lossr�<����       �	I���Xc�A�*

loss*^�<9wG+       �	�B��Xc�A�*

loss�\�=
^X�       �	X��Xc�A�*

loss���=�xŀ       �	���Xc�A�*

loss6:�=� ֘       �	:!��Xc�A�*

loss �L=|��       �	���Xc�A�*

loss�z/<���       �	���Xc�A�*

loss >=����       �	�"��Xc�A�*

loss��=���       �	+���Xc�A�*

loss8�<��2       �	W[��Xc�A�*

loss�0�<C��       �	���Xc�A�*

loss)6g=p�J       �	稣�Xc�A�*

loss�� =<�n       �	�U��Xc�A�*

loss}�<��F\       �	V��Xc�A�*

lossZ_<H(3>       �	a���Xc�A�*

loss�v^=<'A0       �	�/��Xc�A�*

loss�oB>��h       �	��Xc�A�*

lossR��=�_V;       �	����Xc�A�*

loss��^=`Z       �	X;��Xc�A�*

lossX$=���(       �	�ݨ�Xc�A�*

loss!M�;Z[��       �	䇩�Xc�A�*

loss{�<$!b�       �	�(��Xc�A�*

loss��S=��       �	�Ǫ�Xc�A�*

loss��<��x*       �	0h��Xc�A�*

loss
��;�M!J       �	W��Xc�A�*

loss���=���       �	]���Xc�A�*

loss���;�ymg       �	�D��Xc�A�*

loss�/�<b8�       �	*��Xc�A�*

loss�>(=���-       �	�y��Xc�A�*

loss�s<T��s       �	���Xc�A�*

loss�	K=??�H       �	����Xc�A�*

loss�C<q��       �	�X��Xc�A�*

loss (Y=�p�c       �	����Xc�A�*

lossC�<�d�       �	Н��Xc�A�*

lossn7n={�p       �	�G��Xc�A�*

lossJn>|/�       �	���Xc�A�*

loss�S�<�c�       �	����Xc�A�*

loss߯�<��h�       �	�(��Xc�A�*

loss y�;�9       �	Ӵ�Xc�A�*

loss]NJ<2D       �	��Xc�A�*

loss�sk<�P�x       �	���Xc�A�*

lossV[�=UKgd       �	+M��Xc�A�*

loss��2>����       �	g��Xc�A�*

loss�f=1K�       �	����Xc�A�*

lossO&=��       �	�!��Xc�A�*

lossH��=쒞z       �	����Xc�A�*

lossq��<;Uޅ       �	$���Xc�A�*

loss�<��|       �	�D��Xc�A�*

lossm �<{m8�       �	߻�Xc�A�*

lossjL=Ǚ�%       �	��Xc�A�*

loss�Z�<��P       �	�T��Xc�A�*

loss�?B=�       �	Q���Xc�A�*

lossuF�=����       �	-���Xc�A�*

losst�V="��       �	Y1��Xc�A�*

loss�D�;� �<       �	�ֿ�Xc�A�*

loss-n&=��k1       �	u��Xc�A�*

loss5�<!�pE       �	���Xc�A�*

lossx�'<�!       �	���Xc�A�*

loss���=Gb�       �	UQ��Xc�A�*

lossaH\<&�n�       �	����Xc�A�*

loss�m�<v}5'       �	՗��Xc�A�*

loss��=��}       �	�.��Xc�A�*

loss��=v-:�       �	����Xc�A�*

loss <��5�       �	%z��Xc�A�*

loss��y<��       �	���Xc�A�*

lossH%�<At       �	A���Xc�A�*

loss��=���F       �	�[��Xc�A�*

loss�6�<��aQ       �	d��Xc�A�*

loss�C�<����       �	����Xc�A�*

loss��U<�	��       �	�M��Xc�A�*

lossQ͒=�Ņa       �	���Xc�A�*

loss�c=b��!       �	h���Xc�A�*

loss��7<Cn@9       �	29��Xc�A�*

loss��r=�#       �	����Xc�A�*

loss
i�<~���       �	;q��Xc�A�*

loss��N=�G�       �	�$��Xc�A�*

loss���=��o       �	D���Xc�A�*

loss/�=^C�       �	�m��Xc�A�*

lossҮ�=ѲL�       �	��Xc�A�*

loss/�<&��       �	����Xc�A�*

loss���;����       �	J_��Xc�A�*

loss��=�^�       �	& ��Xc�A�*

loss�H=��       �	����Xc�A�*

loss�?y=�V�       �	@h��Xc�A�*

lossCs;_GJ�       �	)��Xc�A�*

lossH"=2���       �	���Xc�A�*

loss6�p=a[��       �	1C��Xc�A�*

lossbj�=��       �	����Xc�A�*

loss���;�[�       �	����Xc�A�*

lossTH�<w	\-       �	����Xc�A�*

loss�2�=���       �	�D��Xc�A�*

loss�:<ѭ�       �	����Xc�A�*

loss�{<�>~�       �	ڐ��Xc�A�*

loss�>=��D�       �	�8��Xc�A�*

loss��;=���       �	/���Xc�A�*

lossĒ�<�̒�       �	6v��Xc�A�*

loss.�h<hͼ       �	&��Xc�A�*

loss��<�9t�       �	����Xc�A�*

lossT�<+��]       �	MI��Xc�A�*

lossp�;5K�$       �	���Xc�A�*

loss�Q;�OkL       �	���Xc�A�*

lossӈ:={���       �	�C��Xc�A�*

loss$�;�^       �	����Xc�A�*

loss,��9�� r       �	�~��Xc�A�*

loss�i�:�P�       �	v��Xc�A�*

loss8�;��       �	����Xc�A�*

loss��<���)       �	m��Xc�A�*

lossEU=��QL       �	a��Xc�A�*

loss!;�2<�       �	���Xc�A�*

loss��=J��       �	k`��Xc�A�*

lossa�>�P�       �	D���Xc�A�*

lossh7�;5A��       �	0���Xc�A�*

loss��=�:��       �	e8��Xc�A�*

loss��U=��gk       �	*���Xc�A�*

loss8�=)A�       �	���Xc�A�*

lossT�<UB�       �	�O��Xc�A�*

loss���<�XA>       �	<���Xc�A�*

loss�s�=��r�       �	t���Xc�A�*

loss��<����       �	B>��Xc�A�*

loss��=��K�       �	 ��Xc�A�*

loss�<=��#�       �	�l��Xc�A�*

loss�3o=�       �	���Xc�A�*

lossD�]=#��       �	����Xc�A�*

lossM�=���       �	�f��Xc�A�*

loss|(�<)��P       �	{3��Xc�A�*

loss�Jd=�v       �	����Xc�A�*

loss���=�A       �	����Xc�A�*

loss6jD=J�b{       �	r���Xc�A�*

lossi��<41�f       �	�D��Xc�A�*

lossQ�=0��E       �	U��Xc�A�*

loss{w<�Xo       �	|���Xc�A�*

loss���<��æ       �	u���Xc�A�*

loss�}|<��~�       �	�-��Xc�A�*

lossiB=ۿ�       �	����Xc�A�*

loss�7G<
� �       �	����Xc�A�*

loss���<�~X       �	�>��Xc�A�*

lossZ�<���B       �	m���Xc�A�*

lossN7c<?�       �	����Xc�A�*

loss(+=OA��       �	�J��Xc�A�*

loss�:�=B�,       �	[���Xc�A�*

loss�R=�Q%       �	hY��Xc�A�*

loss;�9<R6Ê       �	���Xc�A�*

loss�c�<���       �	8���Xc�A�*

loss!c�<�Pz/       �	����Xc�A�*

loss <�xi7       �	�C��Xc�A�*

loss�c�<E^�       �	i ��Xc�A�*

loss*l�<�:2.       �	%���Xc�A�*

loss���<��2�       �	�6 �Xc�A�*

loss�=Wd'       �	c� �Xc�A�*

loss���=ܩܫ       �	���Xc�A�*

loss�5=%@1�       �	���Xc�A�*

loss��<xl�x       �	2=�Xc�A�*

lossX<͂�4       �	y�Xc�A�*

loss4R=�l��       �	ҩ�Xc�A�*

loss�g-=�j�       �	h�Xc�A�*

losso�< ¦u       �	�
�Xc�A�*

lossv��<!��       �	İ�Xc�A�*

loss�i=�nW�       �	\X�Xc�A�*

lossB��<��j�       �	���Xc�A�*

loss�X�<��E�       �	��Xc�A�*

loss;�`;Tq�h       �	OZ	�Xc�A�*

loss�
=W@�       �	J
�Xc�A�*

loss��<�I��       �	Q�"�Xc�A�*

loss-̮<�%��       �	>A#�Xc�A�*

loss��=�8       �	��#�Xc�A�*

losszP=���       �	*p$�Xc�A�*

lossW�4=hL�       �	
%�Xc�A�*

lossz�5=��4V       �	��%�Xc�A�*

lossi6=�Ug�       �	�A&�Xc�A�*

loss�[ =�1�       �	8�&�Xc�A�*

loss\7�<�`<�       �	��'�Xc�A�*

loss��=���       �	GT(�Xc�A�*

loss*�<�T��       �	��(�Xc�A�*

lossX8�<N��       �	q�)�Xc�A�*

loss*aJ=<ٱ�       �	��*�Xc�A�*

loss���<<��       �	W,�Xc�A�*

lossG�"='�u�       �	� -�Xc�A�*

loss1:<-�8       �	G�-�Xc�A�*

lossN��;�,bI       �	jg.�Xc�A�*

loss$�.=3�5       �	}/�Xc�A�*

loss��?<e��       �	'k0�Xc�A�*

loss�k�=<;�?       �	1�Xc�A�*

loss���<ˌ��       �	�2�Xc�A�*

loss�0A=#�2�       �	EH3�Xc�A�*

lossJ-�<��N       �	g�3�Xc�A�*

loss �r=ͷ��       �	��4�Xc�A�*

loss�I~<K�\       �	sI5�Xc�A�*

lossDˬ<T[�<       �	c�5�Xc�A�*

loss��M=몹�       �	�6�Xc�A�*

lossw�=���       �	�]7�Xc�A�*

lossF��<��6       �	8�Xc�A�*

loss#
�<c޶�       �	ͱ8�Xc�A�*

loss�\=�ր�       �	�Y9�Xc�A�*

lossC�<K��o       �	�:�Xc�A�*

loss�y<M
.       �	ճ:�Xc�A�*

loss*�G=2��       �	6Z;�Xc�A�*

loss�=?�iP       �	�<�Xc�A�*

lossq;�=�ӆ       �	H�<�Xc�A�*

loss�;)*�       �	�P=�Xc�A�*

loss�h=�w�X       �	�=�Xc�A�*

lossp6>0)'�       �	͏>�Xc�A�*

loss�<#=Q��       �	:?�Xc�A�*

loss�+@=i���       �	
�?�Xc�A�*

loss�?o<�o1�       �	��@�Xc�A�*

loss� 0<��t       �	l%A�Xc�A�*

losst��<�t�       �	��A�Xc�A�*

loss�/S<�J�       �	��B�Xc�A�*

loss�=���       �	EFC�Xc�A�*

lossK�=��o�       �	��C�Xc�A�*

loss�l�<|�tJ       �	gD�Xc�A�*

loss�S=��Ժ       �	�!E�Xc�A�*

loss1�6<�E       �	[�E�Xc�A�*

loss���<]�*       �	͓F�Xc�A�*

loss<^S=��       �	�7G�Xc�A�*

loss�<vr�       �	��G�Xc�A�*

loss�K�=��*�       �	��H�Xc�A�*

loss`�s<�Є�       �	7RI�Xc�A�*

loss��i;I�4�       �	��I�Xc�A�*

lossJV<Hnr       �	��J�Xc�A�*

loss�X�;2Q��       �	74K�Xc�A�*

loss���<Ʌ��       �	��K�Xc�A�*

loss�b=B�r�       �	�vL�Xc�A�*

loss�-v=��Ĳ       �	M�Xc�A�*

loss��2<�v�       �	j�M�Xc�A�*

loss?,�<j���       �	�XN�Xc�A�*

loss���<����       �	��N�Xc�A�*

loss/�<O鍅       �	g�O�Xc�A�*

loss�x�;on:9       �	cCP�Xc�A�*

loss�Z>=S�=�       �	W�P�Xc�A�*

loss2�3=f8��       �	׉Q�Xc�A�*

loss��!>�}�.       �	0/R�Xc�A�*

loss��= ok�       �	.�R�Xc�A�*

loss�ws<�א�       �	c�S�Xc�A�*

lossa6=g��B       �	UT�Xc�A�*

loss�|<2�       �	6U�Xc�A�*

loss̬�<��a       �	��U�Xc�A�*

loss<�.=4��       �	PV�Xc�A�*

loss`�H=j��       �	�	W�Xc�A�*

lossoi?=�x�       �	j�W�Xc�A�*

loss�4 =L;�+       �	�X�Xc�A�*

loss��e<����       �	ڎY�Xc�A�*

loss=*k=�C       �	�Z�Xc�A�*

loss�']=̅��       �	gH[�Xc�A�*

loss{1$=����       �	��[�Xc�A�*

loss=�'=5�2�       �	q�\�Xc�A�*

loss�B�<�HE       �	�6]�Xc�A�*

loss�Wn<����       �	��]�Xc�A�*

loss��.=�$�       �	`s^�Xc�A�*

lossA�H=	�A       �	�_�Xc�A�*

loss=��<,�G       �	.�_�Xc�A�*

lossF�<Xx)       �	�M`�Xc�A�*

loss��<M�       �	��`�Xc�A�*

loss�g=c"WA       �	d�a�Xc�A�*

loss=�T=���^       �	'0b�Xc�A�*

loss՟;o�s9       �	��b�Xc�A�*

lossC�<�K�       �	�pc�Xc�A�*

loss�
=hm��       �	�	d�Xc�A�*

loss}- <����       �	}�d�Xc�A�*

loss���=����       �	Ve�Xc�A�*

lossIe�<��O�       �	��e�Xc�A�*

loss6�K=3�;n       �	:�f�Xc�A�*

loss)h�;�2m$       �	�.g�Xc�A�*

lossA�;籥�       �	��g�Xc�A�*

loss���;j��       �	�ph�Xc�A�*

loss}9=)��o       �	Li�Xc�A�*

losso<K��7       �	��i�Xc�A�*

loss �j=��3�       �	^j�Xc�A�*

lossQ.�="T�Q       �	/�j�Xc�A�*

loss�H-<���       �	c�k�Xc�A�*

loss���<"L1       �	6l�Xc�A�*

lossCzY=ٯ�       �	(�l�Xc�A�*

loss|O<�j,       �	;qm�Xc�A�*

loss�`r=� �%       �	�n�Xc�A�*

loss)E>=���:       �	=�n�Xc�A�*

lossH$<C4\�       �	 Uo�Xc�A�*

loss/#�;� �}       �	�o�Xc�A�*

loss���<{ �,       �	��p�Xc�A�*

loss
�,<6j�o       �	�.q�Xc�A�*

loss�	�=�<��       �	?�q�Xc�A�*

loss��=�|'       �	cr�Xc�A�*

lossL�<fQ��       �	�s�Xc�A�*

loss�,�;i�ǎ       �	Ӥs�Xc�A�*

lossC�<��       �	�Bt�Xc�A�*

loss��=f��%       �	��t�Xc�A�*

loss�޺=q��\       �	v�u�Xc�A�*

loss��t<Yl9Y       �	|*v�Xc�A�*

loss�6!=���       �	��v�Xc�A�*

loss�NR=4�Q       �	Agw�Xc�A�*

loss-V%=),}       �	9x�Xc�A�*

loss3*-<44�?       �	S�x�Xc�A�*

loss��k=-.��       �	oJy�Xc�A�*

loss�><�#       �	�y�Xc�A�*

lossx��=��{       �	�zz�Xc�A�*

loss�:<��X"       �	/{�Xc�A�*

loss�<�m       �	��{�Xc�A�*

loss�M<F6��       �	�L|�Xc�A�*

loss�>��        �	��|�Xc�A�*

lossܥ>=
��       �	�}�Xc�A�*

loss�j7=O*�       �	�$~�Xc�A�*

loss�`<I�S�       �	��~�Xc�A�*

loss!B<3K       �	�w�Xc�A�*

loss���<,�
       �	 ��Xc�A�*

loss��)<��       �	n���Xc�A�*

loss1F'=�6[�       �	$_��Xc�A�*

loss�}�<Q��E       �	.���Xc�A�*

loss&�<���l       �	Q���Xc�A�*

loss̗d=98F�       �	�E��Xc�A�*

loss�z�;����       �	����Xc�A�*

loss@	�<@��       �	g���Xc�A�*

loss�p2<�7��       �	�$��Xc�A�*

lossnf=���z       �	Hą�Xc�A�*

loss�D8=8�hS       �	�\��Xc�A�*

loss�n<'�       �	����Xc�A�*

loss�~&=����       �	����Xc�A�*

loss�U=j�>       �	;��Xc�A�*

loss�P =��#       �	݈�Xc�A�*

lossl%�<'_9�       �	_y��Xc�A�*

loss�k�;�$��       �	���Xc�A�*

loss�\?<:��       �	B���Xc�A�*

loss7��<�_"�       �	�Q��Xc�A�*

loss9?=�O(       �	Q���Xc�A�*

loss��G<K�\�       �	ϣ��Xc�A�*

loss�1=SS�T       �	bM��Xc�A�*

loss��=���       �	-��Xc�A�*

loss�?<O�v       �	���Xc�A�*

loss\es;ښ�M       �	�,��Xc�A�*

lossZ~�=��       �	�ŏ�Xc�A�*

loss؟�<[��B       �	�d��Xc�A�*

loss��;��J       �	�/��Xc�A�*

loss䪅=�ӈ       �	�Ǒ�Xc�A�*

lossD/G=ۤ��       �	�a��Xc�A�*

loss��'=��!V       �	���Xc�A�*

loss]ڴ<^i0B       �	���Xc�A�*

loss�"<�Tt       �	iS��Xc�A�*

loss�V< �h�       �	*���Xc�A�*

lossQ\�<�F��       �	���Xc�A�*

lossJ˛;�v�       �	<P��Xc�A�*

loss=�=���	       �	��Xc�A�*

lossz�.=��n       �	����Xc�A�*

loss�]_=�C�k       �	�4��Xc�A�*

loss[/4=P�l       �	NИ�Xc�A�*

lossi �<Hs��       �	m���Xc�A�*

loss!;�CSO       �	�/��Xc�A�*

loss�d=sR�8       �	�˚�Xc�A�*

loss�/�<0zIi       �	�c��Xc�A�*

loss�� <��m       �	` ��Xc�A�*

lossi=�(��       �	Ú��Xc�A�*

lossD�P<���e       �	�4��Xc�A�*

loss�=�݌       �	Xɝ�Xc�A�*

loss#��<ݩ�       �	e��Xc�A�*

lossH�$=~w&<       �	)��Xc�A�*

loss	�q=o��3       �	����Xc�A�*

loss-5�<l���       �	�T��Xc�A�*

loss��I<�h�       �	r���Xc�A�*

loss��V<��       �	���Xc�A�*

loss�=l.�&       �	&S��Xc�A�*

loss��";Mj�       �	D���Xc�A�*

loss��`<ԙ�       �	���Xc�A�*

loss�k=	��j       �	�F��Xc�A�*

loss#��=(v�       �	�ץ�Xc�A�*

lossE54=��i       �	G��Xc�A�*

loss�/�;nRM       �	���Xc�A�*

loss��=�f�-       �	�V��Xc�A�*

lossRΥ<BS�       �	����Xc�A�*

loss���=ﱔ       �	��Xc�A�*

loss�,�=�c�-       �	
M��Xc�A�*

loss��Y<���       �	1&��Xc�A�*

losszF�;o<F�       �	�ǫ�Xc�A�*

loss%�==G�e�       �	���Xc�A�*

lossk� =!9�       �	>��Xc�A�*

lossId=!8�8       �	���Xc�A�*

loss�=�W4�       �	�Ů�Xc�A�*

loss���<�|��       �	�f��Xc�A�*

lossC��<9�?�       �	�@��Xc�A�*

loss���<�]��       �		ݰ�Xc�A�*

loss�PN<;A�       �	����Xc�A�*

lossc�=��Q�       �	S ��Xc�A�*

lossh�=��wY       �	s���Xc�A�*

loss�[R=�T�       �	d��Xc�A�*

lossb�>����       �	���Xc�A�*

loss�1=���       �	ú��Xc�A�*

loss��=���J       �	�_��Xc�A�*

lossC��<�uV�       �	e���Xc�A�*

loss�G�<��X�       �	Z���Xc�A�*

loss�96=~u�       �	F?��Xc�A�*

loss�e=�6��       �	�ڷ�Xc�A�*

loss�R!=)�J�       �	����Xc�A�*

loss���<$X�       �	�$��Xc�A�*

loss���=�~��       �	���Xc�A�*

losss��=B �,       �	����Xc�A�*

loss)�=��zk       �	���Xc�A�*

loss�=p�e�       �	곻�Xc�A�*

loss�
<-�XJ       �	�N��Xc�A�*

loss�[L<=P3�       �	$��Xc�A�*

lossò4=�D6�       �	��Xc�A�*

loss|T�;=Uzl       �	�d��Xc�A�*

loss�g=�~&�       �	?���Xc�A�*

loss�+A=6"d�       �	4���Xc�A�*

loss�<��       �	)A��Xc�A�*

loss�ފ=�;       �		���Xc�A�*

loss�)�=�N 7       �	�x��Xc�A�*

lossdݞ<AQ8       �	���Xc�A�*

lossLE<�ӅO       �	����Xc�A�*

loss��=�b$8       �	����Xc�A�*

loss6r�=�m-2       �	+��Xc�A�*

loss��=���       �	���Xc�A�*

lossa�=J��=       �	�c��Xc�A�*

lossD�=����       �	;���Xc�A�*

loss��=����       �	A���Xc�A�*

lossh�=_2       �	6Z��Xc�A�*

loss?�3=/mpo       �	����Xc�A�*

loss�3E=Ny�       �	ߧ��Xc�A�*

lossZh�<v{�       �	�F��Xc�A�*

loss�Q=�i�       �	����Xc�A�*

lossa�=���(       �	���Xc�A�*

lossq��<0�Ni       �	�*��Xc�A�*

loss8N+<$���       �	}���Xc�A�*

lossr�d<ƾ�f       �	�l��Xc�A�*

loss�B=�撯       �	���Xc�A�*

loss*�=�O�       �	����Xc�A�*

loss<�*=K�)       �	3Q��Xc�A�*

loss9ƕ<n�       �	���Xc�A�*

loss|S<��2�       �	+���Xc�A�*

lossxW�;���       �	�;��Xc�A�*

loss�U5=s���       �	����Xc�A�*

lossEn�<�T��       �	,���Xc�A�*

losszvp=rL־       �	�4��Xc�A�*

lossf;�<g�;�       �	}���Xc�A�*

loss\�=�q��       �	'g��Xc�A�*

loss	�<��       �	���Xc�A�*

lossO�<o��       �	.���Xc�A�*

lossږa=G88       �	�L��Xc�A�*

loss���<_�vN       �	����Xc�A�*

loss��=�z;       �	���Xc�A�*

loss�� ="C�       �	(��Xc�A�*

loss���<k��o       �	�#��Xc�A�*

loss���=��       �	v���Xc�A�*

loss�x4>yV�       �	˜��Xc�A�*

loss�;�;��j�       �	\W��Xc�A�*

lossϱ<m��F       �	u��Xc�A�*

lossS�>�D�\       �	����Xc�A�*

lossɹ"=�I>/       �	����Xc�A�*

losstf=<)��       �	w���Xc�A�*

lossh�z<��Z       �	�7��Xc�A�*

losss�<N���       �	�Z��Xc�A�*

loss-!=Xԇe       �	���Xc�A�*

lossD�o=�c��       �	����Xc�A�*

loss���=rȃ]       �	����Xc�A�*

loss�d=��?       �	D��Xc�A�*

loss�w�<�g��       �	����Xc�A�*

loss!Kb=�D�       �	����Xc�A�*

lossG�<f���       �	-%��Xc�A�*

lossl4<�       �	a���Xc�A�*

loss��<�H       �	�c��Xc�A�*

loss��;}���       �	���Xc�A�*

loss��<���       �	Q���Xc�A�*

loss�[�<���       �	ZF��Xc�A�*

loss.==�{`]       �	~���Xc�A�*

loss�a=��u�       �	����Xc�A�*

loss�ǆ=�:^       �	�Y��Xc�A�*

lossY��=I;\�       �	���Xc�A�*

loss�Cp=DW       �	V���Xc�A�*

lossr�O=Ni�#       �	�3��Xc�A�*

loss�U<T/�@       �	�-��Xc�A�*

loss��g<1B�       �	X���Xc�A�*

lossA�=w1�S       �	���Xc�A�*

loss���=Ed��       �	m��Xc�A�*

lossq��=�5       �	���Xc�A�*

loss�M];?Q�I       �	ʥ��Xc�A�*

loss'N�=��$�       �	E��Xc�A�*

loss���<=��       �	|���Xc�A�*

loss��D<���       �	:���Xc�A�*

loss��/<# r4       �	�8��Xc�A�*

loss�l=�gw�       �	U���Xc�A�*

lossm�[=`��       �	���Xc�A�*

loss�?<��-�       �	�C��Xc�A�*

loss%=�Y�       �	����Xc�A�*

loss�N�=�ܮ       �	e���Xc�A�*

lossw�<�E�0       �	u?��Xc�A�*

loss��<�C       �	���Xc�A�*

loss	��<�㫋       �	9���Xc�A�*

loss�+�<���       �	���Xc�A�*

loss]=�Ѝ       �	���Xc�A�*

lossIp�=tF��       �	�[��Xc�A�*

loss�J%=V�'�       �	I���Xc�A�*

lossϊK<B���       �	3���Xc�A�*

lossȴ�<'�u       �	�)��Xc�A�*

lossa|.=�}�       �	c���Xc�A�*

lossR�=wV��       �	�l��Xc�A�*

lossݽG=�ET&       �	�>��Xc�A�*

loss�^B=ȧ�       �	����Xc�A�*

loss���=xsP       �	�q��Xc�A�*

loss���<�]��       �	��Xc�A�*

lossg҈<�� I       �	ɪ��Xc�A�*

loss�0�;�+       �	�B �Xc�A�*

lossL5�<��>       �	f� �Xc�A�*

loss��=�~�<       �	�r�Xc�A�*

lossi��<�q��       �	��Xc�A�*

loss�d><\�G7       �	@��Xc�A�*

loss{�<��L~       �	�=�Xc�A�*

loss?�=ɛR       �	���Xc�A�*

loss_�;��C       �	r�Xc�A�*

lossP�<�,�<       �	k�Xc�A�*

loss�$�<HK1�       �	���Xc�A�*

loss��e=y)ܭ       �	�C�Xc�A�*

loss��8=Q=�       �	
��Xc�A�*

lossD�>l\@�       �	,�Xc�A�*

loss��=���       �	��Xc�A�*

loss|�!<1�/p       �	Ͱ�Xc�A�*

loss���=��       �	L	�Xc�A�*

loss��=޵N�       �	[
�Xc�A�*

loss���<�0,       �	n�
�Xc�A�*

lossכ�;tX��       �	�A�Xc�A�*

lossA��<M���       �	��Xc�A�*

loss5 �=�M�       �	u�Xc�A�*

loss���<�~[�       �	��Xc�A�*

lossH�=���       �	���Xc�A�*

lossJ�O<�cJ�       �	nL�Xc�A�*

loss���;	�*       �	g��Xc�A�*

lossL��<V~vM       �	��Xc�A�*

lossE�D<	9#�       �	�5�Xc�A�*

lossL6%=�H��       �	���Xc�A�*

loss��=��y�       �	b��Xc�A�*

loss(�*=����       �	�*�Xc�A�*

loss��<��k�       �	��Xc�A�*

loss���=)@��       �	���Xc�A�*

lossI�<�4�r       �	OZ�Xc�A�*

loss3�=���        �	���Xc�A�*

loss���=�x`       �	���Xc�A�*

loss�@�=,$��       �	�2�Xc�A�*

loss��-=��a       �	C��Xc�A�*

lossZ�:=���       �	�a�Xc�A�*

loss��4=�@�       �	�9�Xc�A�*

loss&�=U��       �	B!�Xc�A�*

loss]�=\��Y       �	b��Xc�A�*

loss�<�3��       �	S��Xc�A�*

loss��\=De!�       �	k�Xc�A�*

lossA��=����       �	l
�Xc�A�*

loss�^=}�'       �	���Xc�A�*

loss�^=ԃh       �	�U�Xc�A�*

loss�j�<���       �	5��Xc�A�*

lossn+=�yY�       �	'��Xc�A�*

loss_i�<oLgS       �	6 �Xc�A�*

loss��q=��=�       �	o��Xc�A�*

loss��<��ro       �	HP �Xc�A�*

lossS6�<YRw�       �	|� �Xc�A�*

loss���;��6       �	��!�Xc�A�*

loss��<��~        �	D0"�Xc�A�*

loss).�<��,Y       �	�"�Xc�A�*

lossI�<:�ɇ       �	��#�Xc�A�*

lossqF�<�g       �	�3$�Xc�A�*

loss��K=�\Hb       �	��$�Xc�A�*

loss���<����       �	�x%�Xc�A�*

loss�C�<����       �	�&�Xc�A�*

lossLP=�ː       �	�&�Xc�A�*

loss�F=r}77       �	.X'�Xc�A�*

loss�׽<�.3&       �	Z�'�Xc�A�*

loss�D�=�B       �	�(�Xc�A�*

loss��(<�gR        �	�8)�Xc�A�*

loss���<m�.�       �	��)�Xc�A�*

loss�t�<t��       �	�t*�Xc�A�*

loss=�P<NF�       �	�d+�Xc�A�*

lossg�;���       �	�,�Xc�A�*

lossz�Q<�VP       �	@�,�Xc�A�*

loss�)h=��[�       �	p�-�Xc�A�*

lossFv`<��}       �	�Y.�Xc�A�*

loss�<��1�       �	R�.�Xc�A�*

loss��<:�w�       �	l0�Xc�A�*

loss��=ą��       �	K�0�Xc�A�*

lossL;�=p{b�       �	~1�Xc�A�*

loss;�<��P       �	L2�Xc�A�*

lossW"\=�h�       �	��2�Xc�A�*

loss�Lc=E�F       �	w3�Xc�A�*

loss6�(=�S�       �	�'4�Xc�A�*

loss�d~=��x       �	�5�Xc�A�*

lossn��=`pk       �	"6�Xc�A�*

lossϯy<���       �	��6�Xc�A�*

loss@�=���       �	�W7�Xc�A�*

lossج�;�mZ       �	�)8�Xc�A�*

loss�+<�2�       �	Y�8�Xc�A�*

lossS�=���       �	Ϡ9�Xc�A�*

loss s=�`��       �	�B:�Xc�A�*

loss/lE<-�       �	n;�Xc�A�*

loss��=�%�A       �	��;�Xc�A�*

lossNf>;�C�       �	�e<�Xc�A�*

loss[��=ʍ��       �	�	=�Xc�A�*

loss��k=,4bX       �	��=�Xc�A�*

loss�=�5W       �	�\>�Xc�A�*

lossE�=�       �	
?�Xc�A�*

loss%b�=XLQ       �	_�?�Xc�A�*

loss,=Q��       �	�~@�Xc�A�*

lossp7�<A=       �	�A�Xc�A�*

loss�T=��       �	0�A�Xc�A�*

loss�}*<�       �	�bB�Xc�A�*

lossz��<��\�       �	C�Xc�A�*

loss�b<�zN       �		�C�Xc�A�*

lossy=�5�       �	��D�Xc�A�*

loss��*=�|��       �	�,E�Xc�A�*

loss-�;_��,       �	��E�Xc�A�*

loss}�=��       �	*sF�Xc�A�*

loss�
�;m���       �	!\G�Xc�A�*

loss~S =l�       �	�H�Xc�A�*

loss���=�R�Q       �	��H�Xc�A�*

loss���=W&]       �	enI�Xc�A�*

loss�_6=���6       �	J�Xc�A�*

loss�{<=F���       �	w�J�Xc�A�*

loss^<�1F       �	
gK�Xc�A�*

lossM�=�lP       �	 L�Xc�A�*

loss;�=R�P       �	׿L�Xc�A�*

loss���<���       �	�hM�Xc�A�*

lossM[�=���%       �	N�Xc�A�*

loss9�=^z       �	��N�Xc�A�*

lossfFS=����       �	�eO�Xc�A�*

loss�^�;���       �	P�Xc�A�*

lossQ�t<�́�       �	�P�Xc�A�*

lossdZ(=R�       �	�NQ�Xc�A�*

loss��=eVC�       �	��Q�Xc�A�*

loss��<:��       �	w�R�Xc�A�*

lossEd�=5d       �	=ES�Xc�A�*

loss���<�]Bg       �	?�S�Xc�A�*

loss�Y	=W�1       �	<�T�Xc�A�*

loss��1=.�       �	�%U�Xc�A�*

loss�2<|3�       �	z�U�Xc�A�*

loss���;I�RR       �	0eV�Xc�A�*

loss�V�;u�       �		�V�Xc�A�*

loss$��=
�        �	E�W�Xc�A�*

loss5�<�h        �	(GX�Xc�A�*

loss�]<D��       �	�Y�Xc�A�*

lossEl=��-_       �	A�Y�Xc�A�*

loss6=�̶e       �	�oZ�Xc�A�*

loss���<��       �	�`[�Xc�A�*

loss�x=[�?       �	~�[�Xc�A�*

lossߍ=GHk       �	Օ\�Xc�A�*

loss�h`=?���       �	�.]�Xc�A�*

lossa3=%�       �	��]�Xc�A�*

loss���=�e       �	�|^�Xc�A�*

lossh�<ENY       �	#_�Xc�A�*

lossdKd<���T       �	��_�Xc�A�*

loss�<^zF       �	�`�Xc�A�*

loss7�=d�*       �	?a�Xc�A�*

loss�s�<�&V�       �	0�a�Xc�A�*

loss$X�=K��s       �	�]b�Xc�A�*

loss�Y�=i���       �	&c�Xc�A�*

lossH�A=�͐�       �	�c�Xc�A�*

lossoFD=�a�       �	�Zd�Xc�A�*

loss?.:=���       �	��d�Xc�A�*

loss�a=��b�       �	t�e�Xc�A�*

lossf7�;�|r�       �	7f�Xc�A�*

lossr�;�I!}       �	o�f�Xc�A�*

loss_�=�[��       �	�yg�Xc�A�*

loss�d#=&�j�       �	&h�Xc�A�*

lossԍ=�k       �	��h�Xc�A�*

loss=a�<���       �	�}i�Xc�A�*

loss�\ =�I�g       �	ũj�Xc�A�*

lossM��;�       �	Ŭk�Xc�A�*

loss�X�<h�T�       �	{Kl�Xc�A�*

loss#�<���`       �	-�l�Xc�A�*

loss�<����       �	��m�Xc�A�*

lossI1=�l
       �	NCn�Xc�A�*

loss4�<_0pN       �	��n�Xc�A�*

loss�*�<3=       �	�vo�Xc�A�*

lossqg�=��O       �	Ep�Xc�A�*

loss�0=����       �	�p�Xc�A�*

loss�oN<P]�U       �	�xq�Xc�A�*

loss�m�;,�k       �	�r�Xc�A�*

loss���;/N       �	��r�Xc�A�*

lossǯ�<�n�       �	�s�Xc�A�*

loss�:�=7&V       �	(Et�Xc�A�*

loss2{$=�È�       �	T�t�Xc�A�*

loss��.<�JG       �	��u�Xc�A�*

loss��'=���       �	�v�Xc�A�*

lossr=�
�W       �	%�v�Xc�A�*

loss��];T5�*       �	��w�Xc�A�*

lossH�?<�*v'       �	�"x�Xc�A�*

loss�@�<Y�O       �	�x�Xc�A�*

lossl�=��       �	c^y�Xc�A�*

loss!�<e2�       �	U�y�Xc�A�*

loss>�
=�sV�       �	�z�Xc�A�*

loss�� =��1;       �	B?{�Xc�A�*

loss�x$<���9       �	~�{�Xc�A�*

loss���<�D��       �	�w|�Xc�A�*

loss6�#=i�V�       �	�}�Xc�A�*

loss��<�3�       �	i�}�Xc�A�*

loss���<p�KF       �	�;~�Xc�A�*

loss�{];�e��       �	I�~�Xc�A�*

loss<F�<A�@�       �	nn�Xc�A�*

loss��<)��       �	���Xc�A�*

loss	�;k�E       �	6���Xc�A�*

lossr��<-�@       �	-C��Xc�A�*

lossc�=��ֈ       �	���Xc�A�*

loss�N=�Z       �	5z��Xc�A�*

loss��+<�᯲       �	{��Xc�A�*

loss8��;��d�       �	遲�Xc�A�*

loss=��<��Ji       �	(I��Xc�A�*

loss��~;�qur       �	���Xc�A�*

loss�K;0��U       �	Â��Xc�A�*

loss揯;�;6�       �	��Xc�A�*

loss�TX;�@\       �	9���Xc�A�*

loss�<��       �	+P��Xc�A�*

loss��=R�z�       �	���Xc�A�*

loss�< ;���S       �	����Xc�A�*

lossq�l<��       �	)!��Xc�A�*

loss���:�c�L       �	���Xc�A�*

loss{R�9���#       �	Z��Xc�A�*

loss�/�<=5�       �	���Xc�A�*

loss̫�;�;c�       �	����Xc�A�*

loss��3<=FRI       �	L��Xc�A�*

loss_�;�"M       �	�5��Xc�A�*

loss�%
;І��       �	pύ�Xc�A�*

loss&l�;�]�:       �	h��Xc�A�*

loss�y�=,�A�       �	E��Xc�A�*

loss�`Y;~NT�       �	Χ��Xc�A�*

loss��=_S       �	�<��Xc�A�*

lossT�^=�Z��       �	�Ӑ�Xc�A�*

loss$�^=��       �	fj��Xc�A�*

loss!!�<�қ       �	����Xc�A�*

loss�3=�Ѿh       �	ђ��Xc�A�*

loss$|�<���       �	<1��Xc�A�*

loss�\=h�lg       �	���Xc�A�*

lossh�<�l+       �	Q���Xc�A�*

loss��T=�V*�       �	�6��Xc�A�*

lossW/�<|�K       �	l͕�Xc�A�*

loss���=�5<E       �	�`��Xc�A�*

loss��:=�0k       �	��Xc�A�*

lossZ��<�ʋp       �	����Xc�A�*

loss���<qH3�       �	w-��Xc�A�*

loss��=N�.�       �	���Xc�A�*

loss���<uP�       �	��Xc�A�*

lossw�A<���       �	���Xc�A�*

loss��t=ȋ��       �	Ϊ��Xc�A�*

lossQE=�V�J       �	!V��Xc�A�*

lossׂ�;��Pg       �	v���Xc�A�*

loss�T�=WJ*T       �	��Xc�A�*

loss~T<���       �	�Ξ�Xc�A�*

loss�7�;Р�       �	�s��Xc�A�*

loss�dT;�t�       �	#J��Xc�A�*

loss�δ;M���       �	%��Xc�A�*

loss��j=�z�       �	����Xc�A�*

loss:�X<Ae�       �	P��Xc�A�*

lossX�=�J�       �	���Xc�A�*

loss�d=�=<M       �	ؚ��Xc�A�*

loss�S<��j       �	����Xc�A�*

loss�l=%���       �	�6��Xc�A�*

lossw�<��       �	U���Xc�A�*

lossOn�:Ř`{       �	Y���Xc�A�*

loss�h�<�hV       �	�o��Xc�A�*

loss��;��Ô       �	B��Xc�A�*

losso�P={Z�       �	v���Xc�A�*

loss@��=���       �	��Xc�A�*

loss�U�=\�[0       �	$��Xc�A�*

lossv��<gh�       �	f3��Xc�A�*

lossQ��;Rz��       �	8N��Xc�A�*

lossh��<C��U       �	�C��Xc�A�*

loss;{=i��V       �	���Xc�A�*

lossq�=-�f       �	"į�Xc�A�*

lossM�N=w|��       �	�m��Xc�A�*

loss��q=��h�       �	�y��Xc�A�*

loss��< 4�f       �	��Xc�A�*

lossJ��<dU�T       �	V���Xc�A�*

lossx��=K8�       �	�5��Xc�A�*

loss�3�<
w       �	\˴�Xc�A�*

loss�d	<��=+       �	U���Xc�A�*

loss|�<uՀ       �	�J��Xc�A�*

loss2�B=ɤ$       �	���Xc�A�*

lossö>��[       �	���Xc�A�*

loss{}V=���       �	L���Xc�A�*

loss�C!=	���       �	$��Xc�A�*

loss�u =����       �	,���Xc�A�*

loss�V�<rP'`       �	 T��Xc�A�*

lossc��<��X       �	r��Xc�A�*

loss���=�	�d       �	����Xc�A�*

losse�]<[e�       �	�C��Xc�A�*

lossV@$<�ED       �	����Xc�A�*

loss�/==z��<       �	N��Xc�A�*

loss�[=�k�       �	�:��Xc�A�*

loss1o�=����       �	u��Xc�A�*

loss$��=|�^�       �	&���Xc�A�*

loss?��=��Rm       �	�N��Xc�A�*

loss:K;�׵       �	�[��Xc�A�*

loss��<�]�O       �	�>��Xc�A�*

loss��;%�|�       �	29��Xc�A�*

loss�=tH��       �	�-��Xc�A�*

loss#��<7�p       �	2���Xc�A�*

loss�X8=?<J�       �	�k��Xc�A�*

loss)u<����       �	g��Xc�A�*

lossF!�=�k#!       �	n���Xc�A�*

loss��'=���W       �	�<��Xc�A�*

loss#^l<S���       �	0���Xc�A�*

loss#=�V��       �	z��Xc�A�*

loss�*v<�       �	k��Xc�A�*

loss<c<Ӝ��       �	����Xc�A�*

loss�D%<�j       �	�<��Xc�A�*

loss'�<*[�B       �	����Xc�A�*

loss��<Nh"       �	s��Xc�A�*

loss�6�;�vv�       �	��Xc�A�*

loss��2=�CIR       �	����Xc�A�*

loss�n,<�Y��       �	4I��Xc�A�*

loss��=8��
       �	����Xc�A�*

loss�j�<�c��       �	���Xc�A�*

lossi��<�p�4       �	f0��Xc�A�*

loss�:�=�7'�       �	J���Xc�A�*

loss�3@=���[       �	&o��Xc�A�*

loss<�"=?��       �	�<��Xc�A�*

loss#�Y<h�w�       �	'���Xc�A�*

loss;î;f�0       �	Jy��Xc�A�*

loss���<A�j�       �	H��Xc�A�*

loss��;	�       �	����Xc�A�*

lossd�<c�?       �	�Y��Xc�A�*

loss�=��$�       �	���Xc�A�*

loss�H�<羦�       �	���Xc�A�*

loss{��;��/       �	���Xc�A�*

loss��<U�;e       �	�n��Xc�A�*

loss��<3��       �	���Xc�A�*

lossФ =b|�       �	<���Xc�A�*

loss㩍<xϴ�       �	�Z��Xc�A�*

loss�x�=�]c       �	����Xc�A�*

loss��<�� �       �	���Xc�A�*

lossqƘ<���       �	�;��Xc�A�*

loss1��;��@�       �	E���Xc�A�*

loss��:*~��       �	]���Xc�A�*

loss�ʭ<W��6       �	� ��Xc�A�*

loss���=�v)       �	����Xc�A�*

loss�7=䘐�       �	GZ��Xc�A�*

loss��w<P�\�       �	<���Xc�A�*

loss�	�;��       �	\���Xc�A�*

lossha�<54)+       �	S��Xc�A�*

loss�p(<Y ^�       �	#���Xc�A�*

lossl�;d7EY       �	����Xc�A�*

lossO��<�؟�       �	�X��Xc�A�*

lossƼ<�;h�       �	����Xc�A�*

lossc�=�خ�       �	b� �Xc�A�*

loss3du=v�8�       �	��Xc�A�*

loss"�<����       �	���Xc�A�*

lossډ!=Mɯ       �	�_�Xc�A�*

lossx.�<&��       �	�(�Xc�A�*

loss2G�=Q��       �	���Xc�A�*

loss���=���       �	�k�Xc�A�*

lossC�k<�%=       �	)�Xc�A�*

loss��Y<�93B       �	��Xc�A�*

loss�~9<+�2�       �	���Xc�A�*

loss��<�Q�       �	AH�Xc�A�*

loss�-\<�K��       �	���Xc�A�*

loss�{=�V       �	�~�Xc�A�*

loss��<:� l       �		�Xc�A�*

loss�C=:�z�       �	l�	�Xc�A�*

loss��=KZ's       �	bN
�Xc�A�*

loss%I�<ceD       �	:�
�Xc�A�*

loss1��<�o9�       �	˂�Xc�A�*

loss�,
<R0�       �	g*�Xc�A�*

loss��x<_���       �	���Xc�A�*

lossW�<���       �	H��Xc�A�*

loss/�=}Hl       �	g(�Xc�A�*

loss��I<�X=       �	��Xc�A�*

loss�V�<m%�C       �	�f�Xc�A�*

lossw$�<�P�       �	�Xc�A�*

loss�PJ=�~��       �	 ��Xc�A�*

loss�6`=qK1n       �	�3�Xc�A�*

loss���<����       �	F��Xc�A�*

loss��=r�       �	C��Xc�A�*

loss1�<�;-�       �	�0�Xc�A�*

loss �=���       �	���Xc�A�*

lossf��< ��P       �	���Xc�A�*

loss��+<e�&�       �	�7�Xc�A�*

loss��<�>%       �	F��Xc�A�*

lossH�4=Ѓ�       �	"o�Xc�A�*

lossMֽ;��D       �	�	�Xc�A�*

loss�z�<�~u        �	Ƨ�Xc�A�*

lossw <~�_       �	�N�Xc�A�*

lossL~%;K{�,       �	)��Xc�A�*

loss��<WF�       �	׉�Xc�A�*

losss�=<WW�       �	u!�Xc�A�*

losse8<U��x       �	h��Xc�A�*

loss���<!�m       �	&q�Xc�A�*

lossʞ�<5'�#       �	��Xc�A�*

loss�&=O~�       �	/��Xc�A�*

lossd��;<�N�       �	vl�Xc�A�*

loss�G}=��2c       �	-
�Xc�A�*

loss�p�<LL]       �	��Xc�A�*

loss�5�=F�%       �	qZ�Xc�A�*

loss���=8��i       �	% �Xc�A�*

loss���<ȯ\�       �	b� �Xc�A�*

loss�5�=����       �	�J!�Xc�A�*

loss�a�<y�u�       �	�!�Xc�A�*

lossT=�`�       �	�x"�Xc�A�*

loss �<����       �	�#�Xc�A�*

loss:(=c�qA       �	ZH$�Xc�A�*

loss$��;5!�R       �	��$�Xc�A�*

loss`�<7j7p       �	ς%�Xc�A�*

loss��=���       �	`&�Xc�A�*

loss�tr=�g��       �	��&�Xc�A�*

loss�p)=�A�>       �	��'�Xc�A�*

loss�=M�       �	%(�Xc�A�*

lossW�|<e�ՠ       �	"O)�Xc�A�*

loss=@={�3�       �	q*�Xc�A�*

loss)`�;�Kʡ       �	�*�Xc�A�*

loss�&�;扷�       �	Q+�Xc�A�*

loss�H�<�C��       �	k�+�Xc�A�*

loss\��<cM       �	k�,�Xc�A�*

loss!��<��<�       �	34-�Xc�A�*

loss��;����       �	o�-�Xc�A�*

lossj�W<7�       �	�v.�Xc�A�*

lossW2�<\��       �	�/�Xc�A�*

lossNO�;;� D       �	,�/�Xc�A�*

loss�0�<q�|       �	A�0�Xc�A�*

lossS�<�i|       �	\1�Xc�A�*

loss�_<=X_�       �	C�1�Xc�A�*

loss�e�<�n�(       �	F~2�Xc�A�*

loss��<�-e
       �	i3�Xc�A�*

loss�1�=���d       �	˾3�Xc�A�*

loss���;|J.       �	�V4�Xc�A�*

losshS<��.g       �	��4�Xc�A�*

lossgΐ=M�P&       �	=�5�Xc�A�*

lossD�;��       �	>6�Xc�A�*

lossz�&=�.�z       �	�6�Xc�A�*

lossA��<���       �	Fx7�Xc�A�*

loss=�2��       �	 8�Xc�A�*

loss��=>��a       �	�8�Xc�A�*

loss�1z:��s       �	�W9�Xc�A�*

lossV��<i�'�       �	{�9�Xc�A�*

loss]�=f�v-       �	��:�Xc�A�*

loss�.>���       �	{1;�Xc�A�*

lossz%�;�|�       �	��;�Xc�A�*

loss�6=���       �	�<�Xc�A�*

loss��;����       �	�=�Xc�A�*

lossNR=�:q�       �	j�=�Xc�A�*

lossEa�:?�h       �	�f>�Xc�A�*

loss#��<���       �	�?�Xc�A�*

lossW<��#�       �	�?�Xc�A�*

loss��;~��K       �	�W@�Xc�A�*

loss��<vL       �	�A�Xc�A�*

lossݜp<V@��       �	��A�Xc�A�*

loss
�g<�{e       �	�`B�Xc�A�*

loss�0�<�^�       �	P�B�Xc�A�*

lossNG<9[�G       �	�C�Xc�A�*

loss�<'$�       �	yuD�Xc�A�*

loss�=&n�       �	�VE�Xc�A�*

loss��<z��       �	� F�Xc�A�*

loss���<����       �	��F�Xc�A�*

loss��=޴Ҍ       �	QG�Xc�A�*

loss:8�==��q       �	@�G�Xc�A�*

lossc�l=[2�       �	��H�Xc�A�*

loss*=g���       �	�CI�Xc�A�*

lossO:I;"J"�       �	�J�Xc�A�*

loss�C=�H�t       �	"TK�Xc�A�*

loss��m<�4��       �	��K�Xc�A�*

loss	�<�6/�       �	T�L�Xc�A�*

lossS2j<%�p�       �	VEM�Xc�A�*

lossL�<9e�S       �	��M�Xc�A�*

loss��<n���       �	h�N�Xc�A�*

lossȔ=m�\       �	�*O�Xc�A�*

loss�H=��t�       �	�KP�Xc�A�*

lossϊ�<*���       �	��P�Xc�A�*

loss���<H0h       �	ܞQ�Xc�A�*

loss��<�_4       �	m�R�Xc�A�*

loss�Ux<�&,R       �	�S�Xc�A�*

loss���<UL\T       �	�*T�Xc�A�*

loss���:0ĵ       �	Y�T�Xc�A�*

loss"�<�l�       �	�aU�Xc�A�*

loss�q<��Pa       �	� V�Xc�A�*

loss�$�=��,       �	ĘV�Xc�A�*

losso��=}-/h       �	�1W�Xc�A�*

lossO$	<��YQ       �	��W�Xc�A�*

loss��e=y&E�       �	sX�Xc�A�*

lossN�c<��       �	=Y�Xc�A�*

loss]:�<��       �	g�Y�Xc�A�*

loss�U�;��0Q       �	ӆZ�Xc�A�*

loss&W�;�AN�       �	%[�Xc�A�*

losst�<���       �	$�[�Xc�A�*

loss�O�<�n"T       �	��\�Xc�A�*

loss<q�<j���       �	�V]�Xc�A�*

loss{�2=Zpi$       �	�]�Xc�A�*

loss
�Z;��M       �	��^�Xc�A�*

loss�M<�?�       �	L�_�Xc�A�*

lossD�<��0       �	{h`�Xc�A�*

loss*s<�/       �	�	a�Xc�A�*

loss�B=}�>�       �	i�a�Xc�A�*

lossR�M=2�D        �	�Bb�Xc�A�*

lossw�6=q��>       �	�c�Xc�A�*

loss�4�=���       �	}�c�Xc�A�*

loss?��="Q|5       �	bHd�Xc�A�*

loss��E=���       �	�$e�Xc�A�*

loss�w�<WAC       �	��e�Xc�A�*

loss��;"�I�       �	�{f�Xc�A�*

loss��2;�5'r       �	��g�Xc�A�*

loss�/�;.��       �	ǃh�Xc�A�*

loss�=z[��       �	?Vi�Xc�A�*

loss��<�5�       �	L�i�Xc�A�*

loss�{�=���       �	�j�Xc�A�*

lossuӠ<�.�       �	�1k�Xc�A�*

lossGl<�M       �	��k�Xc�A�*

loss �;�d*       �	 al�Xc�A�*

lossvb[<�ZK(       �	�=m�Xc�A�*

loss)25<�uk$       �	^�m�Xc�A�*

lossd�;Jb�9       �	�rn�Xc�A�*

loss��D<��5       �	�o�Xc�A�*

loss��[;Ʋ�       �	��o�Xc�A�*

loss�� =F�m       �	�wp�Xc�A�*

loss
ھ<8�]       �	c&q�Xc�A�*

loss�<�޺�       �	��q�Xc�A�*

loss��
=��       �	_zr�Xc�A�*

lossh�r=:L�T       �	&s�Xc�A�*

loss<\�=]
�       �	��s�Xc�A�*

loss���<ZG��       �	�Rt�Xc�A�*

loss��=Zf�       �	��t�Xc�A�*

loss�0=�m0       �	u�u�Xc�A�*

lossv\�<o1a/       �	�)v�Xc�A�*

loss�=9X��       �	;�v�Xc�A�*

loss;�x<�@��       �	`w�Xc�A�*

loss��;(���       �	r�w�Xc�A�*

loss�l<�`B�       �	�x�Xc�A�*

loss��`=�P6-       �		4y�Xc�A�*

lossQ�	=�܉�       �	_�y�Xc�A�*

lossz��<�
e�       �	mz�Xc�A�*

loss_7^=7v�
       �	�{�Xc�A�*

loss�Q.=!��>       �	հ{�Xc�A�*

lossc�e<-H��       �	�X|�Xc�A�*

loss/�;�jU       �	}}�Xc�A�*

lossq�=8� �       �	��}�Xc�A�*

lossӪ<��ņ       �	y:~�Xc�A�*

loss��K=P��e       �	�~�Xc�A�*

loss��-=�4       �	Pn�Xc�A�*

loss�#=:5��       �	���Xc�A�*

loss��<Eñ�       �	Ѳ��Xc�A�*

loss��9=,��M       �	I���Xc�A�*

loss&�B=��X*       �	9)��Xc�A�*

loss ��<#P�`       �	�̂�Xc�A�*

loss�Y�<܎       �	�l��Xc�A�*

loss̟ ;o��       �	c��Xc�A�*

loss�K�=���       �	ɰ��Xc�A�*

loss��=>�u       �	�Q��Xc�A�*

loss��=l�V�       �	A��Xc�A�*

loss��<�:�       �	���Xc�A�*

loss��;�Yی       �	!@��Xc�A�*

loss�kh<�_�a       �	@݇�Xc�A�*

loss6=
=��ݨ       �	y��Xc�A�*

loss���='V��       �	���Xc�A�*

loss.8>"Y       �	���Xc�A�*

lossʼy=ʊ:�       �	亊�Xc�A�*

lossM��<��a]       �	�\��Xc�A�*

loss���;zv       �	����Xc�A�*

loss�H=����       �	z���Xc�A�*

loss<i�<�,�D       �	DO��Xc�A�*

lossR�;jT/       �	o���Xc�A�*

loss� <<�:pN       �	���Xc�A�*

loss��<*ڋ�       �	���Xc�A�*

loss��=�m��       �	��Xc�A�*

loss4Y�<���?       �	����Xc�A�*

loss�|=�ӄ�       �		O��Xc�A�*

lossq;=���       �	�g��Xc�A�*

loss��S=RE��       �	6"��Xc�A�*

loss�[�<��       �	ԕ�Xc�A�*

lossJA�;�5=       �	ݕ��Xc�A�*

loss('?<���       �	S?��Xc�A�*

loss��<�qj�       �	���Xc�A�*

lossQ+;2��       �	�y��Xc�A�*

loss���;��g       �	���Xc�A�*

loss�Z'<1�1�       �	���Xc�A�*

loss�<=�m�8       �	L��Xc�A�*

losst��<��>�       �	�)��Xc�A�*

loss��=�       �	����Xc�A�*

loss&\�=�]�       �	F\��Xc�A�*

loss	��=h�k�       �	���Xc�A�*

loss��9="�       �	����Xc�A�*

lossZ��<�Y-�       �	�G��Xc�A�*

loss
p�<�T^       �	7��Xc�A�*

loss�K�=��R       �	����Xc�A�*

lossVJ�<ށdu       �	nL��Xc�A�*

loss���=ShU�       �	��Xc�A�*

loss ��;vT       �	C���Xc�A�*

loss;�)=-(�        �	�&��Xc�A�*

loss�`%=EX<       �	�â�Xc�A�*

loss�R=UGe�       �	�^��Xc�A�*

lossN�_<��s       �	@���Xc�A�*

loss��;n��       �	=���Xc�A�*

loss��8=�� `       �	�<��Xc�A�*

loss��<3,�b       �	_ӥ�Xc�A�*

loss��4<�^�       �	�q��Xc�A�*

loss���=��[�       �	���Xc�A�*

loss���;U?4s       �	��Xc�A�*

loss���<�&Ϻ       �	/R��Xc�A�*

loss�_�<�>�s       �	����Xc�A�*

loss��
=�,3�       �	��Xc�A�*

loss�ǉ<~Tq       �	�/��Xc�A�*

loss��G=oC�       �	�ͪ�Xc�A�*

loss�g=Y���       �	Ts��Xc�A�*

lossf�z<�c�       �	���Xc�A�*

loss��q<Ԋ0       �	����Xc�A�*

loss�l�=�	��       �	�K��Xc�A�*

loss]��<,���       �	���Xc�A�*

loss�K]<�ײ       �	<���Xc�A�*

loss���<��2�       �	.��Xc�A�*

loss^=�y�t       �	Ư�Xc�A�*

loss���<�U�`       �	|e��Xc�A�*

lossA�=���       �	���Xc�A�*

loss���;y�&�       �	���Xc�A�*

loss�b�;��       �	pa��Xc�A�*

loss�*�=��       �	���Xc�A�*

loss!z=�[��       �	����Xc�A�*

loss�vN<�ͫ�       �	�H��Xc�A�*

lossk<2�       �	�Z��Xc�A�*

lossEl]=�1z�       �	����Xc�A�*

loss���;t�=       �	O���Xc�A�*

loss�EZ;��r�       �	�6��Xc�A�*

lossI��<��^�       �	,Է�Xc�A�*

lossQ�=���       �	g~��Xc�A�*

lossmB<[���       �	a��Xc�A�*

loss�o�=�r�#       �	����Xc�A�*

lossĄ�=��_;       �	LQ��Xc�A�*

loss��[=~��B       �	����Xc�A�*

loss���<z�T       �	����Xc�A�*

loss}�F=cJӛ       �	�"��Xc�A�*

lossۘ\=W��       �	����Xc�A�*

loss��<՛�       �	&R��Xc�A�*

loss1)<�e{       �	���Xc�A�*

loss
p=Năb       �	V��Xc�A�*

loss�p6=��
�       �	��Xc�A�*

losshd�<])�       �	�V��Xc�A�*

lossIx�<�j��       �	%���Xc�A�*

lossc��<�L��       �	����Xc�A�*

lossd�<\�DC       �	�&��Xc�A�*

lossɂk=ܘa       �		���Xc�A�*

loss��<�D�       �	�g��Xc�A�*

loss�=#SL       �	~���Xc�A�*

loss�t
=&�}       �	����Xc�A�*

lossl[=�/s�       �	8J��Xc�A�*

loss�=)���       �	9���Xc�A�*

lossӲ�<���       �	����Xc�A�*

loss�)6=Џ}       �	�9��Xc�A�*

loss�L�=��E�       �	+���Xc�A�*

lossn�H<�"�       �	���Xc�A�*

loss�2=��Rk       �	�?��Xc�A�*

loss	�=��;�       �	}���Xc�A�*

loss7Ë<���       �	a���Xc�A�*

loss}<*�B       �	 )��Xc�A�*

loss�<ɽ�       �	����Xc�A�*

loss�:;��YI       �	wg��Xc�A�*

loss���<PQTI       �	� ��Xc�A�*

loss�Tc<�Т       �	4���Xc�A�*

loss�=.9E{       �	;��Xc�A�*

lossTc�=���       �	9���Xc�A�*

lossI�g=^�.       �	w��Xc�A�*

loss�o=�]U�       �	���Xc�A�*

lossָ�;���       �	����Xc�A�*

loss.=Yq8�       �	`t��Xc�A�*

loss�8�<<'{       �	\��Xc�A�*

loss�+W<uuܞ       �	W���Xc�A�*

loss�+�:� �       �	wM��Xc�A�*

lossܭ�<���T       �	���Xc�A�*

loss6�<ￊ       �	]���Xc�A�*

loss&�|<j�Av       �	 X��Xc�A�*

loss,�4<��       �	��Xc�A�*

loss�4<O��       �	����Xc�A�*

lossOm�<���       �	�l��Xc�A�*

lossA}<�Z�       �	Z��Xc�A�*

loss�֜=�� X       �	���Xc�A�*

loss:7=���k       �	�y��Xc�A�*

loss�F_<F�7�       �	A���Xc�A�*

loss%�=Jj�L       �	�x��Xc�A�*

loss��d;?y       �	<��Xc�A�*

loss�?<���       �	�~��Xc�A�*

loss}��;��Ha       �	'��Xc�A�*

loss�ѩ<�G�x       �	%��Xc�A�*

loss���<��	|       �	����Xc�A�*

lossr�`<:�       �	���Xc�A�*

loss=	=�0��       �	�b��Xc�A�*

loss��u=,OY7       �	���Xc�A�*

lossT_<k�Q�       �	����Xc�A�*

loss?O=}��G       �	�b��Xc�A�*

loss���<b��S       �	XU��Xc�A�*

loss���<�^J       �	����Xc�A�*

lossoWM<�p�       �	����Xc�A�*

lossM5`=o]��       �	*��Xc�A�*

loss@�U=��C�       �	����Xc�A�*

loss �K=�w�       �	���Xc�A�*

loss�v&<6���       �	w���Xc�A�*

lossZ@�=j���       �	!?��Xc�A�*

loss���<Hb��       �	U���Xc�A�*

loss�+=G���       �	�r��Xc�A�*

lossq��;�S       �	���Xc�A�*

loss!�;T�r       �	n���Xc�A�*

loss���<#q�t       �	�F��Xc�A�*

loss��=g�fO       �	�b��Xc�A�*

lossס3<�&5       �	���Xc�A�*

loss�m=;�;:       �	����Xc�A�*

lossܝ�<,�/�       �	�a��Xc�A�*

loss�.�=���       �	���Xc�A�*

loss�:=�D�H       �	z���Xc�A�*

lossւ<��!       �	G��Xc�A�*

loss}�9=
��	       �	9&��Xc�A�*

losstw@=��m�       �	����Xc�A�*

loss�.=�J       �	���Xc�A�*

loss�7�<��R�       �	�s��Xc�A�*

lossŐZ<�e�       �	{��Xc�A�*

lossH�Q<7o       �	1���Xc�A�*

loss�W�<�w��       �	Y��Xc�A�*

loss��b=�x;�       �	I���Xc�A�*

lossI3'=#{��       �	؝��Xc�A�*

loss��o<����       �	�8��Xc�A�*

lossl\;��       �	����Xc�A�*

lossv�<�Tg       �	��Xc�A�*

loss��<�pR       �	�*��Xc�A�*

lossNg<�`       �	���Xc�A�*

lossr==i��@       �	�X��Xc�A�*

loss�-�<d��       �	����Xc�A�*

loss&�)=�&X�       �	����Xc�A�*

loss��z=%��:       �	����Xc�A�*

loss�s�<���       �	T �Xc�A�*

lossa� =d��       �	9� �Xc�A�*

loss�V�=x��       �	3��Xc�A�*

loss�ħ<�;x�       �	�#�Xc�A�*

loss�گ<�;       �	���Xc�A�*

loss��='RXk       �	kd�Xc�A�*

loss�=���       �	��Xc�A�*

loss?�<�|}       �	Q��Xc�A�*

loss�:�;bsw       �	rO�Xc�A�*

loss��=e�       �	h��Xc�A�*

loss�=�=�%��       �	��Xc�A�*

loss���=�nA       �	.�Xc�A�*

loss�
=��       �	#��Xc�A�*

loss��<`E2~       �	/n�Xc�A�*

loss@~�;��       �	-		�Xc�A�*

loss}9<̳v�       �	��	�Xc�A�*

loss�@�<6���       �	+5
�Xc�A�*

losss�<G�n�       �	��
�Xc�A�*

loss\`<�T��       �	�h�Xc�A�*

loss��=r
ؚ       �	u�Xc�A�*

loss=��<@���       �	d��Xc�A�*

loss�y=��       �	�A�Xc�A�*

loss�LX=D�~�       �	���Xc�A�*

loss�\�< s<u       �	�p�Xc�A�*

loss&	�<��m       �	)�Xc�A�*

loss6%�<k�&�       �	w��Xc�A�*

loss�1=�X�       �	Y�Xc�A�*

loss��<� ��       �	���Xc�A�*

loss�=#��-       �	��Xc�A�*

loss!?�<R߇I       �	�&�Xc�A�*

lossF:<1�-       �	��Xc�A�*

loss�2=�N8       �	c�Xc�A�*

lossC��;�0�}       �	D��Xc�A�*

loss�;�=ϭc�       �	6��Xc�A�*

loss�"�<;�       �	^*�Xc�A�*

lossddG=E       �	��Xc�A�*

losswv!=���c       �	�c�Xc�A�*

loss���<�峹       �	G �Xc�A�*

lossa�<&��	       �	9��Xc�A�*

loss�<�<�G0       �	�4�Xc�A�*

loss�z	<�T�       �	��Xc�A�*

loss#h;�p�       �	��Xc�A�*

loss��<��W�       �	��Xc�A�*

loss��!=�pm7       �	�>�Xc�A�*

loss���<�%i       �	H��Xc�A�*

loss���< ��       �	���Xc�A�*

lossj�=�>?X       �	%#�Xc�A�*

loss��$<N]$�       �	��Xc�A�*

lossҤ�:�	�x       �	�d�Xc�A�*

lossZ�@=����       �	<K�Xc�A�*

loss`˓<z��       �	(��Xc�A�*

loss�=3��)       �	� �Xc�A�*

loss�\�<Ϣ)�       �	_!�Xc�A�*

loss�8=4��       �	G"�Xc�A�*

loss�8N='>uO       �	#�"�Xc�A�*

lossw'd=�4�=       �	�H$�Xc�A�*

loss!�$=+�)*       �	��$�Xc�A�*

loss�2�; �܈       �	�%�Xc�A�*

loss|�<�Gt�       �	 8&�Xc�A�*

lossN �;���9       �	��&�Xc�A�*

lossH;T=u'�C       �	�x'�Xc�A�*

losss'=���       �	�(�Xc�A�*

loss�Ԃ<.��       �	/�(�Xc�A�*

lossL�;<���       �	�])�Xc�A�*

lossaÃ<�>       �	e�)�Xc�A�*

loss?V=�iz�       �	Ú*�Xc�A�*

losslUO<��	       �	I/+�Xc�A�*

loss�<��6�       �	\�,�Xc�A�*

loss�7�;K��h       �	XV-�Xc�A�*

loss�=0+��       �	�9.�Xc�A�*

loss��0<����       �	e�.�Xc�A�*

lossO��<T�,Y       �	(~/�Xc�A�*

loss O=D%d       �	�v0�Xc�A�*

loss���<ۏ�       �	D1�Xc�A�*

loss\a�<�|Ns       �	42�Xc�A�*

lossR�7=��       �	��2�Xc�A�*

loss:'�<���&       �	��3�Xc�A�*

loss�y<<��       �	�L4�Xc�A�*

loss�!;�Wڌ       �	��4�Xc�A�*

loss���<I���       �	�@6�Xc�A�*

loss��<�J��       �	@27�Xc�A�*

loss�<����       �	��7�Xc�A�*

loss��><0O�       �	��8�Xc�A�*

loss�K�<��t       �	@9�Xc�A�*

loss��=g�?       �	L:�Xc�A�*

lossQ\'=�>�       �	��:�Xc�A�*

loss%�T<C#p       �	k;�Xc�A�*

loss�g�<���       �	`=<�Xc�A�*

loss�&<�RԱ       �	B�<�Xc�A�*

loss���9�hw       �	�=�Xc�A�*

loss��:J��-       �	�;>�Xc�A�*

loss�נ<'�V�       �	)�>�Xc�A�*

lossO��;$h0�       �	ő?�Xc�A�*

lossv<��       �	�3@�Xc�A�*

loss愚;�Q�%       �	��@�Xc�A�*

loss*+A=��-�       �	�hA�Xc�A�*

lossf��9t���       �	C�Xc�A�*

loss�8�9��       �	$�C�Xc�A�*

lossP��8=$�C       �	�[D�Xc�A�*

loss�<<��9+       �	��D�Xc�A�*

loss�]�<?L�       �	�E�Xc�A�*

lossxd�<��V+       �	KF�Xc�A�*

loss6�c:�}�$       �	=G�Xc�A�*

loss��=�� O       �	�G�Xc�A�*

losslW=��/�       �	؂H�Xc�A�*

lossfi;�59       �	0+I�Xc�A�*

loss���<O||�       �	0�I�Xc�A�*

lossJK=%��F       �	��J�Xc�A�*

lossS|=�Jp[       �	�XK�Xc�A�*

loss�@�<����       �	P�K�Xc�A�*

loss�@<p��       �	՗L�Xc�A�*

loss�9=x��       �	�-M�Xc�A�*

losse�H=2 ��       �	��M�Xc�A�*

loss���<8Q�`       �	A�N�Xc�A�*

losszWD=�c��       �	\:O�Xc�A�*

lossŧ;����       �	��O�Xc�A�*

loss�J=���       �	��P�Xc�A�*

lossl�U=Q�E       �	�eQ�Xc�A�*

loss&#
=P �       �	NR�Xc�A�*

loss8ޕ<��q       �		�R�Xc�A�*

loss�k�<m��       �	�S�Xc�A�*

loss�[ =Ë��       �	9T�Xc�A�*

lossK<jr�       �	�U�Xc�A�*

loss��<%R��       �	��U�Xc�A�*

losstY�;\r�       �	�MV�Xc�A�*

loss��;���@       �	c�V�Xc�A�*

loss;=a��O       �	͏W�Xc�A�*

loss���<�^9       �	A*X�Xc�A�*

loss��;%�X�       �	�X�Xc�A�*

loss�R�;*��       �	�aY�Xc�A�*

lossT��;z�0�       �	&�Z�Xc�A�*

loss�0�<3��       �	�E[�Xc�A�*

loss���;�Ël       �	��[�Xc�A�*

lossx9�;���       �	��\�Xc�A�*

loss�=N3%�       �	�e]�Xc�A�*

lossc� <��
       �	'K^�Xc�A�*

loss��E<Ϸ$q       �	�_�Xc�A�*

loss��;kF6�       �	�_�Xc�A�*

losss�Y; ��       �	;o`�Xc�A�*

loss��><(W��       �	�a�Xc�A�*

losst��;��u�       �	)�a�Xc�A�*

lossۣ�<��$       �	�Rb�Xc�A�*

losst�e<�I�       �	��b�Xc�A�*

losse(�<=�       �	�c�Xc�A�*

loss��i=��ܫ       �	�Ed�Xc�A�*

loss_3q;����       �	7�d�Xc�A�*

loss�gX=�ad:       �	��e�Xc�A�*

loss���<c��       �	?Xf�Xc�A�*

loss�K�<��l       �	��f�Xc�A�*

lossCi�<�ݠ       �	�g�Xc�A�*

loss�_�=7'       �	kIh�Xc�A�*

loss_��<+��p       �	��h�Xc�A�*

lossh5<mwl�       �	�i�Xc�A�*

loss��<s��Z       �	�'j�Xc�A�*

loss�;�cSM       �	X�j�Xc�A�*

loss�C=T8�       �	Ick�Xc�A�*

loss)�X<1�@�       �	ND��Xc�A�*

lossLb
=�u       �	���Xc�A�*

lossyx=N5t�       �	�{��Xc�A�*

loss?��<Z?       �	@��Xc�A�*

loss3�e=��޹       �	���Xc�A�*

lossa!=��t�       �	�]��Xc�A�*

lossq�M<�zx       �	��Xc�A�*

lossz=����       �	ڌ��Xc�A�*

lossO�>��^�       �	�$��Xc�A�*

loss!��<�ǵ�       �		���Xc�A�*

lossE�;8�       �	����Xc�A�*

lossi�6<�� 8       �	�.��Xc�A�*

loss��=x�       �	Ɍ�Xc�A�*

loss~�=��       �	Yl��Xc�A�*

loss}�D<���       �	b��Xc�A�*

loss[{�<}x��       �	,���Xc�A�*

loss�C�;���       �	U��Xc�A�*

lossf�;V�       �	��Xc�A�*

losst�;9y�X       �	w���Xc�A�*

loss�V=���       �	{3��Xc�A�*

loss[�<,�\       �	:ϑ�Xc�A�*

lossid;<}O��       �	̖��Xc�A�*

loss趡;���       �	<K��Xc�A�*

lossmtW=��xB       �	����Xc�A�*

lossT�	<*ɉ�       �	��Xc�A�*

loss{<��R       �	>˕�Xc�A�*

loss��<Q�7       �	�m��Xc�A�*

loss�Ǖ<�)��       �	���Xc�A�*

loss�R=TN��       �	����Xc�A�*

loss纇<c��       �	O]��Xc�A�*

loss�>=Z`~,       �	D���Xc�A�*

lossem�;]�       �	����Xc�A�*

lossؔ<ܕ5�       �	P5��Xc�A�*

loss�U�=��	       �	�Ϛ�Xc�A�*

lossZS<���       �	�r��Xc�A�*

loss�Wv=\��C       �	��Xc�A�*

loss�<G�5       �	����Xc�A�*

loss�>:=#�}       �	�Q��Xc�A�*

loss&�b=
+ �       �	�+��Xc�A�*

lossd�<��X*       �	�Ş�Xc�A�*

loss���<�h�8       �	Z���Xc�A�*

loss7�<�4��       �	�A��Xc�A�*

loss�#<��I       �	S��Xc�A�*

lossg=y       �	����Xc�A�*

loss�U<<��Wc       �	�&��Xc�A�*

loss��=kAV       �	�Ƣ�Xc�A�*

loss���<N��       �	�g��Xc�A�*

loss�A<����       �	�
��Xc�A�*

loss�;=�69�       �	매�Xc�A�*

lossƾ
;�͆       �	,F��Xc�A�*

loss1�;щI       �	���Xc�A�*

loss��/=Dh`D       �	��Xc�A�*

losss{(<��Ld       �	�"��Xc�A�*

lossڨ�=Q�'       �	8���Xc�A�*

loss���<��Z�       �	�X��Xc�A�*

loss]g�:�       �	r���Xc�A�*

loss���;Ij�F       �	����Xc�A�*

lossC�;�Y��       �	�5��Xc�A�*

lossv.�<B"�B       �	�Ҫ�Xc�A�*

loss��<�ݶ       �	+j��Xc�A�*

lossWV�<Q �8       �	���Xc�A�*

loss.R�;W��(       �	����Xc�A�*

loss�
;��	�       �	�<��Xc�A�*

loss��9<o)�       �	s٭�Xc�A�*

loss��v<&�       �	zr��Xc�A�*

loss���;Z��y       �	F	��Xc�A�*

loss���=�l�       �	f���Xc�A�*

lossh�r<�"[)       �	m���Xc�A�*

loss,bH=�w�       �	�E��Xc�A�*

loss��< �yq       �	vݱ�Xc�A�*

loss3�<!¸       �	u��Xc�A�*

lossI]=��       �	g(��Xc�A�*

loss��<���	       �	���Xc�A�*

loss�{S<}W	�       �	4���Xc�A�*

loss�!�<*��T       �	ö��Xc�A�*

loss�d<����       �	�Z��Xc�A�*

loss��<���       �	���Xc�A�*

loss��<� L       �	n��Xc�A�*

loss�*6<$�t       �	����Xc�A�*

loss�?<g       �	����Xc�A�*

loss�T(=���}       �	�2��Xc�A�*

loss;}�<U$�       �	κ�Xc�A�*

loss�5�<�d�E       �	�n��Xc�A�*

loss$��<�4�       �	���Xc�A�*

loss8�<���       �	����Xc�A�*

loss�+�;Uq�F       �	/Q��Xc�A�*

loss <�{�A       �	
���Xc�A�*

lossn��<O^�       �	����Xc�A�*

loss�<ɤ��       �	;��Xc�A�*

lossS�<eyB       �	Mٿ�Xc�A�*

loss���<����       �	����Xc�A�*

loss�z�<�X��       �	�-��Xc�A�*

losshE<�T1       �	���Xc�A�*

lossJw<���E       �	3p��Xc�A�*

loss-ў<)ݺ�       �	�E��Xc�A�*

lossE��<A�       �	����Xc�A�*

loss��=�X_t       �	�w��Xc�A�*

lossx�+<9���       �	���Xc�A�*

loss��=$�r       �	ƨ��Xc�A�*

loss��<�=\�       �	|B��Xc�A�*

losss<��f       �	����Xc�A�*

loss�?=a�/g       �	ݕ��Xc�A�*

loss�<��4S       �	�>��Xc�A�*

loss/P�<q���       �	u���Xc�A�*

loss���<�&9�       �	C���Xc�A�*

loss��=J��=       �	A��Xc�A�*

loss��;;ۏ�       �	����Xc�A�*

lossR�;bz�|       �	Ҋ��Xc�A�*

loss$�X=Y�$H       �	,)��Xc�A�*

lossώ�<�s�       �	����Xc�A�*

lossl�l=K�=�       �	�]��Xc�A�*

lossC%�<s|C�       �	
���Xc�A�*

loss,j<? D:       �	I���Xc�A�*

lossH��;_�U       �	:��Xc�A�*

loss0s�<.��h       �	C���Xc�A�*

loss�<�˘�       �	�}��Xc�A�*

loss��<c8#�       �	���Xc�A�*

loss�5=rj��       �	~���Xc�A�*

loss̣�=���       �	ZI��Xc�A�*

lossf�#=�m�       �	����Xc�A�*

loss[��<�l�3       �	���Xc�A�*

loss�E=�"*�       �	N+��Xc�A�*

loss�e�<%]       �	W���Xc�A�*

loss�[^=����       �	Nb��Xc�A�*

loss�)�<�4�       �	���Xc�A�*

lossؑf=( �       �	����Xc�A�*

lossD�n<\�~       �	4I��Xc�A�*

lossZ�`=jm�       �	����Xc�A�*

lossh�e=���"       �	�{��Xc�A�*

lossSP=�'f;       �	���Xc�A�*

loss�X�;h�/�       �	����Xc�A�*

lossh)R=2��Q       �	�B��Xc�A�*

lossO��<Y���       �	#���Xc�A�*

loss�=�]Y       �	0���Xc�A�*

lossH�=ׅ       �	�4��Xc�A�*

loss�j�<���       �	����Xc�A�*

loss �<;C�T       �	sd��Xc�A�*

lossJ��;���       �	P���Xc�A�*

loss:ɐ;�V�       �	���Xc�A�*

loss(%&=�A�       �	@O��Xc�A�*

loss{ĵ:.�NZ       �	����Xc�A�*

loss��<=E(�F       �	Q���Xc�A�*

losse(�<���       �	�d��Xc�A�*

loss�=���I       �	���Xc�A�*

loss14=&��       �	c���Xc�A�*

loss��j<8�'       �	�:��Xc�A�*

loss�B<!|"       �	���Xc�A�*

lossJ�=Ҋ�       �	����Xc�A�*

lossΒU;1T,�       �	\>��Xc�A�*

loss�#I<�g       �	����Xc�A�*

loss�-�<m‘       �	Sw��Xc�A�*

loss��<pu��       �	���Xc�A�*

loss_�=`��       �	����Xc�A�*

loss<QO=��ع       �	�?��Xc�A�*

loss�|)=�j�;       �	^���Xc�A�*

loss_d�;�`)�       �	/o��Xc�A�*

losst��;�d��       �	J��Xc�A�*

losstc=��T�       �	4���Xc�A�*

loss��H=pu       �	�0��Xc�A�*

loss_e=<�l��       �	����Xc�A�*

losso�=�s�o       �	�\��Xc�A�*

lossW3�;g���       �	x���Xc�A�*

loss�X=	��       �	(���Xc�A�*

loss蔯;$���       �	��Xc�A�*

loss��H<���       �	j���Xc�A�*

lossm�;0˦H       �	����Xc�A�*

lossy?<��a       �	�'��Xc�A�*

loss$��<�P,�       �	����Xc�A�*

loss�:�<.�]]       �	f��Xc�A�*

losss�<��'o       �	���Xc�A�*

loss�<>/'��       �	<���Xc�A�*

loss�$�<ר�[       �	�C��Xc�A�*

loss�@^=���       �	=���Xc�A�*

loss蕮=�\��       �	ɒ��Xc�A�*

loss�Q4<?)�H       �	�3��Xc�A�*

loss�.<E�4�       �	����Xc�A�*

lossD)�<
��       �	�c��Xc�A�*

loss��<�k�Q       �	���Xc�A�*

losss�=�6��       �	����Xc�A�*

loss�l�;��15       �	�q��Xc�A�*

lossU�:���l       �	���Xc�A�*

lossϽK=��9       �	-���Xc�A�*

loss2`[<�}�       �	K��Xc�A�*

loss�} <y�n       �	����Xc�A�*

lossEY�;u�^�       �	���Xc�A�*

loss��K<�x�^       �	�M��Xc�A�*

loss`�<u��y       �	d���Xc�A�*

losse!�<)�)       �	ƅ��Xc�A�*

loss�Lz=�mK*       �	���Xc�A�*

loss�sT<���       �	����Xc�A�*

loss�e?='���       �	����Xc�A�*

lossn�1;N^D       �	() �Xc�A�*

loss�{�;baR       �	q� �Xc�A�*

lossH�7<�r�       �	�d�Xc�A�*

lossG�:�ހj       �	r��Xc�A�*

loss��<�$U       �	��Xc�A�*

lossD�<s�֩       �	�/�Xc�A�*

lossx�>P*�H       �	���Xc�A�*

lossz�f=F0n�       �	"m�Xc�A�*

loss\�/=<�d�       �	��Xc�A�*

loss��<e��b       �	���Xc�A�*

loss�/�;w��       �	�1�Xc�A�*

lossY8<p���       �	��Xc�A�*

lossA�r<�	�       �	�i�Xc�A�*

lossWo�;ğ�       �	(�Xc�A�*

loss���;q�R�       �	���Xc�A�*

lossF�<����       �	=	�Xc�A�*

loss*s�=V��E       �	w�	�Xc�A�*

loss���<�s��       �	�n
�Xc�A�*

loss�3�<&�I�       �	�Xc�A�*

loss#�
<*�j�       �	d��Xc�A�*

loss���<�M�       �	S�Xc�A�*

loss��<*��       �	���Xc�A�*

loss���; `iJ       �	��Xc�A�*

loss�i=�2n       �	;:�Xc�A�*

loss��=h���       �	���Xc�A�*

loss��=]���       �	3k�Xc�A�*

loss�N=+t�       �	��Xc�A�*

loss��=��?�       �	ĳ�Xc�A�*

lossP��=4��]       �	�N�Xc�A�*

loss_ <��       �	1��Xc�A�*

loss���;��       �	d��Xc�A�*

loss�&=�@�t       �	�0�Xc�A�*

loss�#<��       �	R��Xc�A�*

loss� =G���       �	;q�Xc�A�*

loss�ʮ<B�W�       �	��Xc�A�*

loss�i=��.J       �	���Xc�A�*

loss��o<�,E�       �	9a�Xc�A�*

loss���;��i�       �	y�Xc�A�*

lossמ�;̆        �	��Xc�A�*

lossG� <�_       �	}>�Xc�A�*

loss)W};Z�"       �	��Xc�A�*

loss��b<��6       �	aq�Xc�A�*

loss�.U<>�3�       �	��Xc�A�*

loss%��<u?W       �	���Xc�A�*

lossZ+=�O       �	i;�Xc�A�*

loss��p<��gm       �	[��Xc�A�*

loss� �<NiR�       �	��Xc�A�*

lossA�&=���m       �	"6�Xc�A�*

loss��<���       �	���Xc�A�*

loss1�$=|:�Y       �	��Xc�A�*

lossaB=V�7�       �	�]�Xc�A�*

loss%�<�F�       �	G: �Xc�A�*

loss�X�<)3aL       �	>!�Xc�A�*

loss�2=k�~\       �	L�!�Xc�A�*

loss�ς<\�3       �	~#�Xc�A�*

loss��<�O       �	%$�Xc�A�*

loss4�)=p��       �	�$�Xc�A�*

lossc�5=߶%�       �	��%�Xc�A�*

loss�_�<~&��       �	�H&�Xc�A�*

loss��<����       �	��&�Xc�A�*

lossE7�<a�ߦ       �	�p'�Xc�A�*

loss�F<��A�       �	W(�Xc�A�*

lossZZ�;j|�       �	/�(�Xc�A�*

lossg�<�&�|       �	�E)�Xc�A�*

loss[T;���       �	��)�Xc�A�*

loss���<���       �	|�*�Xc�A�*

loss�UY=ٗt�       �	!W+�Xc�A�*

loss� y<RA�       �	,�Xc�A�*

loss�y
=E��       �	��,�Xc�A�*

lossO��<,T�       �	�F-�Xc�A�*

lossa�1<�|�       �	��-�Xc�A�*

lossٯ<�z�a       �	}y.�Xc�A�*

loss���<���       �	b/�Xc�A�*

loss�1Q=���       �	K�/�Xc�A�*

loss��7<���g       �	�D0�Xc�A�*

loss�]p=���&       �	;�0�Xc�A�*

loss]`�=����       �	�{1�Xc�A�*

lossEy=;�       �	�2�Xc�A�*

lossN��<jV�       �	ڪ2�Xc�A�*

loss��/<�L�       �	0b3�Xc�A�*

loss�M�;l^�       �	s�3�Xc�A�*

loss��;���K       �	!�4�Xc�A�*

lossM��<�|kV       �	&5�Xc�A�*

loss��1=.m)t       �	zR6�Xc�A�*

loss�=���       �	��6�Xc�A�*

loss'ޙ;3�N       �	w�7�Xc�A�*

lossii�;��:�       �	�8�Xc�A�*

loss���<É�       �	W�8�Xc�A�*

loss|�=�<�'       �	*�9�Xc�A�*

lossI�<���       �	�':�Xc�A�*

losss�>;M�l       �	��:�Xc�A�*

loss<��;C�>       �	�`;�Xc�A�*

lossӡ�<��;j       �	Q�;�Xc�A�*

loss�]c=���       �	/�<�Xc�A�*

loss=���9       �	�9=�Xc�A�*

loss7�=}AZ�       �	R�=�Xc�A�*

loss� <���       �	�l>�Xc�A�*

loss,TI=I�
I       �	�?�Xc�A�*

loss�^�<�5       �	��?�Xc�A�*

loss1��;����       �	y:@�Xc�A�*

loss�K@=�4�       �	��@�Xc�A�*

lossr.�;Js&^       �	%xA�Xc�A�*

loss��0;2��6       �	(B�Xc�A�*

loss`K�<0&��       �	ƢB�Xc�A�*

lossO�	=��h(       �	�qC�Xc�A�*

lossv߆<n�}�       �	�
D�Xc�A�*

loss q=��c       �	עD�Xc�A�*

loss�z=��PE       �	?E�Xc�A�*

losst�=)��H       �	�E�Xc�A�*

lossC�=q��Z       �	PnF�Xc�A�*

losso��<K       �	MG�Xc�A�*

losst��;�3��       �	��G�Xc�A�*

loss�B�=aI��       �	�<H�Xc�A�*

loss�ׄ<���       �	�H�Xc�A�*

lossR�=Ŧ��       �	hI�Xc�A�*

loss��;�|�{       �	�J�Xc�A�*

lossȔ=Wm��       �	0�J�Xc�A�*

loss[��< �O       �	)@K�Xc�A�*

losscؐ<���       �	��K�Xc�A�*

loss�ɵ<	���       �	anL�Xc�A�*

loss�`�;#�       �	�M�Xc�A�*

loss�=��_       �	S�M�Xc�A�*

lossVs�<#B)        �	=,N�Xc�A�*

loss�;��       �	��N�Xc�A�*

lossS�(=�3�       �	__O�Xc�A�*

loss$
P;�O�       �	��O�Xc�A�*

loss��,<�5*�       �	��P�Xc�A�*

loss���<8��       �	�Q�Xc�A�*

lossd�=U��<       �	̳Q�Xc�A�*

loss���<�"��       �	xGR�Xc�A�*

loss6 =��s�       �	�R�Xc�A�*

loss=,Β�       �	�oS�Xc�A�*

loss�;�;��Z&       �	�T�Xc�A�*

loss��<���       �	e�T�Xc�A�*

loss��=.I       �	+lU�Xc�A�*

loss!��<��F       �	mV�Xc�A�*

loss�+<��H\       �	k�V�Xc�A�*

loss?��<
7�       �	�?W�Xc�A�*

loss��<�ܐV       �	��W�Xc�A�*

lossQl�<3��-       �	�tX�Xc�A�*

loss�g�<Z�r�       �	�Y�Xc�A�*

loss���;��       �	�Y�Xc�A�*

losskS	<Q9U�       �	_FZ�Xc�A�*

loss�K�<HB��       �	i�Z�Xc�A�*

lossѾS<�ej       �	ß[�Xc�A�*

loss� =<
f       �	�]�Xc�A�*

loss�i�<;�g       �	Y�]�Xc�A�*

loss|��<��
       �	2;^�Xc�A�*

loss��e;z�        �	o�^�Xc�A�*

loss�`�;���       �	�p_�Xc�A�*

loss�H=�� �       �	k,`�Xc�A�*

loss�C�<D}       �	��`�Xc�A�*

loss�y�<�ߕ�       �	�\a�Xc�A�*

loss�=pV��       �	��a�Xc�A�*

lossH��=�G�]       �	��b�Xc�A�*

loss=�;��^       �	�;c�Xc�A�*

loss��=���b       �	��c�Xc�A�*

lossW�Z=�#f2       �	�fd�Xc�A�*

loss�X=X=�s       �	��d�Xc�A�*

loss�4�:�o�}       �	Ԙe�Xc�A�*

loss)��;��U8       �	2f�Xc�A�*

loss�KT=	)]�       �	-�f�Xc�A�*

lossw�1<�+�       �	�^g�Xc�A�*

lossL�<��/o       �	� h�Xc�A�*

loss��=.�c       �	��h�Xc�A�*

loss��;|fN       �	nni�Xc�A�*

loss`a<�Kg�       �	j�Xc�A�*

loss�ʫ<�4��       �	^�j�Xc�A�*

loss��1=��O�       �	2k�Xc�A�*

loss�z={�       �	�k�Xc�A�*

loss�<��I�       �	�ml�Xc�A�*

loss�і=5�,       �	�m�Xc�A�*

loss�_=\��       �	�m�Xc�A�*

lossi��=m�m       �	�8n�Xc�A�*

loss�2=�8��       �	4o�Xc�A�*

loss!�N=��n       �	'�o�Xc�A�*

lossm��<�"=�       �	�=p�Xc�A�*

loss��<�i.�       �	��p�Xc�A�*

loss�'2=J�        �	�q�Xc�A�*

loss�n=�<�       �	nr�Xc�A�*

loss�W;�!��       �	��r�Xc�A�*

lossΠ�<�O;       �	�?s�Xc�A�*

losskl<[���       �	��s�Xc�A�*

lossS5q<&0gS       �	=�t�Xc�A�*

loss���<��u�       �	u�Xc�A�*

loss���<�;i       �	d�u�Xc�A�*

lossw��<4��       �	
Kv�Xc�A�*

loss&	=Qɥ       �	��v�Xc�A�*

loss(�==l�Έ       �	Ӈw�Xc�A�*

loss�;I�+       �	5)x�Xc�A�*

loss�b<QKN-       �	0�z�Xc�A�*

loss�=�g��       �	q8{�Xc�A�*

loss<<�J��       �	��{�Xc�A�*

loss�@;��       �	��|�Xc�A�*

loss��<�|��       �	#�}�Xc�A�*

lossZ��<�Z       �	 )~�Xc�A�*

loss�Z<��D       �	��~�Xc�A�*

loss�_[;��ws       �	$�Xc�A�*

loss;�<J�b       �	D3��Xc�A�*

lossV�%<��kG       �	?��Xc�A�*

loss�EZ<��       �	�{��Xc�A�*

loss8�<'t�O       �	���Xc�A�*

loss��q<��2�       �	ȵ��Xc�A�*

lossjW?<��4       �	�Q��Xc�A�*

lossQހ=�YOI       �	��Xc�A�*

loss�7<��w       �	�Ƅ�Xc�A�*

loss͡�;�'{       �	�f��Xc�A�*

loss!�<<n�{9       �	���Xc�A�*

loss#k#<b��V       �	���Xc�A�*

loss*�<>q�+       �	4H��Xc�A�*

lossO�;�*�T       �	X��Xc�A�*

loss�j1=|9�       �	����Xc�A�*

loss2�;<���       �	�2��Xc�A�*

loss��6<3�6�       �	�̉�Xc�A�*

loss��=6�G�       �	bf��Xc�A�*

loss{��<r       �	$	��Xc�A�*

lossh��<��rM       �	nۋ�Xc�A�*

loss�>h<�       �	[y��Xc�A�*

loss1<v�       �	 ��Xc�A�*

loss��="%��       �	du��Xc�A�*

loss�=��1       �	���Xc�A�*

loss(T_=E���       �	v���Xc�A�*

loss�C=�� 
       �	�?��Xc�A�*

loss�	+=Ϫ�u       �	�ѐ�Xc�A�*

loss���<�1�(       �	+i��Xc�A�*

loss��:��~�       �	���Xc�A�*

loss!O�;K���       �	մ��Xc�A�*

loss	=��Oe       �	�X��Xc�A�*

loss�ۿ<�r|�       �	R��Xc�A�*

loss}�;W�Ew       �	���Xc�A�*

loss{q�<���7       �	�C��Xc�A�*

loss��;|��j       �	�ݕ�Xc�A�*

loss^�=�	       �	N���Xc�A�*

loss�^ =l�ud       �	�!��Xc�A�*

loss�o=��-�       �	�ʗ�Xc�A�*

loss�}�<���       �	�i��Xc�A�*

loss�{w<��J       �	$)��Xc�A�*

loss�Q<��1�       �	���Xc�A�*

loss��x<C�d@       �	hX��Xc�A�*

loss���<h�(�       �	W��Xc�A�*

loss/ɜ;Mp��       �	H���Xc�A�*

loss���;�?<�       �	>"��Xc�A�*

loss2��;RH0�       �		���Xc�A�*

loss2�<C-µ       �	�b��Xc�A�*

loss��<E�\a       �	Y��Xc�A�*

loss��<�:�       �	����Xc�A�*

lossĊ�<�B >       �	�R��Xc�A�*

loss���;��q       �	��Xc�A�*

loss���<	�       �	䄡�Xc�A�*

loss�N.=0ǒ       �	�L��Xc�A�*

loss!9�=����       �	q���Xc�A�*

loss�}{<�1�|       �	�.��Xc�A�*

loss;IM=v��"       �	���Xc�A�*

loss@R�<	�?       �	���Xc�A�*

loss�RS< �7       �	�,��Xc�A�*

loss��1=�bp       �	�զ�Xc�A�*

loss�s�<r.��       �	t��Xc�A�*

loss6<4��)       �	��Xc�A�*

lossT��<�m:�       �	���Xc�A�*

lossi��<&�;       �	6\��Xc�A�*

loss��<Č�w       �	� ��Xc�A�*

lossO =�(��       �	(���Xc�A�*

loss�I�;])m�       �	Y2��Xc�A�*

loss�=3qv�       �	Jѫ�Xc�A�*

lossO�=�3��       �	a���Xc�A�*

loss�c< ��w       �	�+��Xc�A�*

loss��<!�       �	Xŭ�Xc�A�*

loss�;�;���l       �	-^��Xc�A�*

loss@~�;���       �	/���Xc�A�*

lossVb�<��{F       �	���Xc�A�*

loss�
�;�SD-       �	N���Xc�A�*

losszn�;k�kT       �	����Xc�A�*

loss�P�=���,       �	i5��Xc�A�*

loss�9<�UL4       �	:̲�Xc�A�*

loss�b�<Z�       �	nn��Xc�A�*

loss��G<cg N       �	A��Xc�A�*

loss�N<�X��       �	+���Xc�A�*

lossX��<+�;�       �	�J��Xc�A�*

loss��!<�^       �	���Xc�A�*

loss�S=Yp��       �	ö�Xc�A�*

loss<��<����       �	�g��Xc�A�*

loss��R=%c&T       �	�
��Xc�A�*

loss K�=R=͛       �	����Xc�A�*

loss\y#<c/jM       �	�:��Xc�A�*

loss�.�; ���       �	#ݹ�Xc�A�*

loss=ٍ;���b       �	Ks��Xc�A�*

loss��4=E��!       �	���Xc�A�*

loss.�N<�kp�       �	����Xc�A�*

loss��z=9�       �	CT��Xc�A�*

loss�G�=�p�       �	����Xc�A�*

loss�B==�Ć�       �	����Xc�A�*

loss�9�<��(       �	&m��Xc�A�*

loss��$=�Ɗ�       �	���Xc�A�*

loss4 �=1��(       �	᛿�Xc�A�*

lossW=F�       �	�2��Xc�A�*

loss�w�;&p[�       �	����Xc�A�*

loss���<��]       �	�q��Xc�A�*

loss�Q<�QPE       �	O#��Xc�A�*

loss4�<�v       �	l���Xc�A�*

loss��2=D�d(       �	�h��Xc�A�*

loss@�0<Ӝ�       �	� ��Xc�A�*

loss-�;�E�0       �	d���Xc�A�*

loss�}�=���       �	�J��Xc�A�*

loss@�=<���       �	����Xc�A�*

loss��;a(j       �	-���Xc�A� *

loss�8=V}��       �	�:��Xc�A� *

losse�<,�aH       �	W���Xc�A� *

loss�*=)���       �	�c��Xc�A� *

loss,�!=�5%�       �	�
��Xc�A� *

loss���=� �       �	����Xc�A� *

lossԲ�;��h�       �	�D��Xc�A� *

loss�g<���[       �	����Xc�A� *

loss�|<��9       �	�x��Xc�A� *

loss�e�<'��       �	v��Xc�A� *

loss*][<��jo       �	���Xc�A� *

loss�v�='�	�       �	�N��Xc�A� *

loss>-=��-       �	����Xc�A� *

loss���=r�]       �	����Xc�A� *

loss�d=��<       �	a��Xc�A� *

losse>^<��V~       �	}���Xc�A� *

loss���<�"i�       �	L��Xc�A� *

lossc��< ��$       �	J���Xc�A� *

loss�c=�u
       �	����Xc�A� *

loss?��<
�<       �	���Xc�A� *

loss^�;��       �	˾��Xc�A� *

loss�=�>��       �	OZ��Xc�A� *

loss��<�8e       �	���Xc�A� *

lossz�<�.       �	����Xc�A� *

loss�U�<(AJ=       �	�4��Xc�A� *

loss�	<�v��       �	 ���Xc�A� *

loss(��;�>x       �	�m��Xc�A� *

loss�5�:�>]�       �	c	��Xc�A� *

loss���;j|K�       �	���Xc�A� *

loss��=�
W       �	�=��Xc�A� *

loss��=K��       �	k���Xc�A� *

loss慹<n�B       �	���Xc�A� *

loss�8o<��3       �	���Xc�A� *

loss� $=i^s       �	W=��Xc�A� *

lossQH	=(��       �	h%��Xc�A� *

loss�{�;B�~�       �	Q���Xc�A� *

lossn �<�UX�       �	?S��Xc�A� *

loss�1Y<Յ�/       �	 ���Xc�A� *

loss��|:y�       �	����Xc�A� *

loss�:�;��*       �	0/��Xc�A� *

loss�h�<.8��       �	����Xc�A� *

lossL-�;�N�       �	i��Xc�A� *

loss���;���       �	��Xc�A� *

loss�!<�;�       �	b���Xc�A� *

loss��<�i�       �	;8��Xc�A� *

loss���9۠f�       �	����Xc�A� *

loss��9G�       �	�e��Xc�A� *

loss*6:�c��       �	.U��Xc�A� *

loss� =@u��       �	v���Xc�A� *

loss�=����       �	!���Xc�A� *

losswb�;+��A       �	>@��Xc�A� *

loss�Ѓ:�[Y       �	����Xc�A� *

loss��;�0�o       �	�v��Xc�A� *

loss�=�c�       �	r���Xc�A� *

loss��F:�
0       �	}?��Xc�A� *

loss��<T*�       �	���Xc�A� *

lossN$r<�{`�       �	/n��Xc�A� *

loss.��;L�[�       �	G��Xc�A� *

loss�<�f��       �	����Xc�A� *

loss�K�;�ry{       �	,.��Xc�A� *

loss2�s=��8       �	i���Xc�A� *

lossM��<8	�       �	�y��Xc�A� *

lossH9�<ߛI       �	���Xc�A� *

lossjc=�h��       �	Y���Xc�A� *

loss���<�Ob�       �	4J��Xc�A� *

loss�[�=�ϸM       �	���Xc�A� *

loss	K�=���       �	[���Xc�A� *

loss��u;�[�.       �	�O��Xc�A� *

loss�"�<ݞ/�       �	O���Xc�A� *

loss3��<忰�       �	7���Xc�A� *

lossZ��<�А=       �	�-��Xc�A� *

loss��n<x�_�       �	"���Xc�A� *

lossv�=�O�       �	���Xc�A� *

loss[W�<^��       �	pz��Xc�A� *

loss/��;�<9p       �	���Xc�A� *

loss�J%<x	p       �	T��Xc�A� *

lossum<����       �	÷��Xc�A� *

loss�A�<��o       �	]O��Xc�A� *

lossıb<�{       �	}���Xc�A� *

loss�k<����       �	���Xc�A� *

loss�T<��       �	nO��Xc�A� *

loss��;!!�       �	����Xc�A� *

losst1F=�xd       �	����Xc�A� *

lossx=��?�       �	�&��Xc�A� *

loss�8�;�s�
       �	+��Xc�A� *

loss��=�?�       �	����Xc�A� *

loss�"<��B�       �	�I��Xc�A� *

loss| /;��{g       �	���Xc�A� *

loss(�<�̂       �	�� �Xc�A� *

lossߜ�;<��P       �	n4�Xc�A� *

loss2(=~��       �	)��Xc�A� *

lossg�<!��:       �	�`�Xc�A� *

loss��o=�?��       �	D��Xc�A� *

loss<��       �	���Xc�A� *

lossT�;O��       �	!;�Xc�A� *

lossC��<.�`�       �	{��Xc�A� *

loss�.=S
�H       �	dv�Xc�A� *

loss``=��A       �	�5�Xc�A� *

loss��
=%��       �	���Xc�A� *

loss4v�<r�S       �	7p�Xc�A� *

lossJ!=6�v�       �	s�Xc�A� *

losslO<pc��       �	��Xc�A� *

loss2��<�kyM       �	jL	�Xc�A� *

lossd�q;��e       �	
�Xc�A� *

lossJ׽;��QH       �	ܛ
�Xc�A� *

loss���<��       �	`#�Xc�A� *

lossi�V=�FD-       �	��#�Xc�A� *

loss�&�=e��E       �	H�$�Xc�A� *

lossb�=�_�       �	�%%�Xc�A� *

loss	Y�<P       �	4�%�Xc�A� *

lossZ��;��wm       �	�W&�Xc�A� *

loss�,�<:��       �	�&�Xc�A� *

loss��=9�       �	�'�Xc�A� *

loss6"V=�r=�       �	`(�Xc�A� *

loss.��;i��       �	c�(�Xc�A� *

loss���;1�٥       �	#J)�Xc�A� *

loss͘<q��       �	1$*�Xc�A� *

loss��!<�q�       �	x�*�Xc�A� *

lossB�=n���       �	UL+�Xc�A� *

lossEº<T��Q       �	��+�Xc�A� *

lossrZ�=���       �	1�,�Xc�A� *

lossCڃ:=i�"       �	K-�Xc�A� *

loss�R<x�3       �	�-�Xc�A� *

loss\�o;Ղb       �	q�.�Xc�A� *

loss�<i?�       �	M,/�Xc�A� *

loss2�=�sa       �	+�/�Xc�A� *

lossC�<��e       �	��0�Xc�A� *

loss��;�}H=       �	?p1�Xc�A� *

loss��b=c�+�       �	�2�Xc�A�!*

lossĶ
;JU�f       �	&�2�Xc�A�!*

loss�F]<Y#�5       �	��3�Xc�A�!*

loss���<�Z	       �	�h4�Xc�A�!*

loss�4�;b?       �	�5�Xc�A�!*

loss�<!��T       �	�26�Xc�A�!*

loss0=s��       �	'�6�Xc�A�!*

losssK�<G�Ƙ       �	57�Xc�A�!*

loss���=���       �	\8�Xc�A�!*

lossͰ�;/p�       �	��8�Xc�A�!*

loss�j�=v�y       �	��9�Xc�A�!*

loss��=�/�b       �	�?:�Xc�A�!*

loss[A�<>�,�       �	��:�Xc�A�!*

lossx<���       �	�};�Xc�A�!*

loss��n<Vs�       �	�y<�Xc�A�!*

loss@�x=z�!~       �	0)=�Xc�A�!*

loss��== __�       �	�=�Xc�A�!*

lossR�"=�l�L       �	2Y>�Xc�A�!*

loss�/�<���H       �	��>�Xc�A�!*

lossѮe<W�R       �	f�?�Xc�A�!*

lossX�6<f��       �	 A@�Xc�A�!*

lossH=h��       �	7�@�Xc�A�!*

loss*U�<9C@       �	�}A�Xc�A�!*

loss�@�<�*UK       �	vB�Xc�A�!*

loss6z<=m'T�       �	V�B�Xc�A�!*

loss8��<��"       �	�PC�Xc�A�!*

loss?<*189       �	��C�Xc�A�!*

loss�NU<��>�       �	L�D�Xc�A�!*

loss�B4=��56       �	�-E�Xc�A�!*

lossĉ�<F��       �	B�E�Xc�A�!*

losss@�=:
��       �	kF�Xc�A�!*

loss�w�<��g�       �	yG�Xc�A�!*

loss/1�<��       �	k�G�Xc�A�!*

lossn�
;��       �	v6H�Xc�A�!*

loss
 �;��q�       �	u�H�Xc�A�!*

losss
�;�B�       �	ѐI�Xc�A�!*

loss*��<�N�       �	�+J�Xc�A�!*

loss�#=~F
3       �	�J�Xc�A�!*

loss�f�<�$�       �	OxK�Xc�A�!*

loss�|^=C8\	       �	oL�Xc�A�!*

loss��<����       �	v�L�Xc�A�!*

loss�,<�`�       �	C=M�Xc�A�!*

loss;֩:����       �	kN�Xc�A�!*

loss��M=-��{       �	ߣN�Xc�A�!*

loss�X=?���       �	6;O�Xc�A�!*

lossM��<c�Zh       �	��O�Xc�A�!*

lossXL�<�W       �	geP�Xc�A�!*

loss�N<fdn�       �	��P�Xc�A�!*

lossn��<W�W�       �	ڑQ�Xc�A�!*

lossҹ�=E��n       �	�'R�Xc�A�!*

loss�oK=4�j       �	Q�R�Xc�A�!*

loss�%=��*�       �	��S�Xc�A�!*

loss!<Sy       �	�T�Xc�A�!*

lossx2<d»u       �	cEU�Xc�A�!*

loss���<�f�       �	�$V�Xc�A�!*

loss�< w��       �	��V�Xc�A�!*

loss]��<�n�       �	cW�Xc�A�!*

loss�0<��iw       �	U�W�Xc�A�!*

lossn�=76v,       �	��X�Xc�A�!*

loss���<K�I       �	��Y�Xc�A�!*

lossY
�;��       �	*Z�Xc�A�!*

loss3�=<�p��       �	��Z�Xc�A�!*

loss	M=n���       �	CU[�Xc�A�!*

loss?�><.�y       �	�[�Xc�A�!*

loss��/=��       �	��\�Xc�A�!*

loss�%�<�"n       �	�]�Xc�A�!*

loss�<[(�       �	~8^�Xc�A�!*

loss�_<��!�       �	�1_�Xc�A�!*

lossf�
=��       �	.�_�Xc�A�!*

loss�b_<YVE       �	{a�Xc�A�!*

loss��<(��       �	RDb�Xc�A�!*

loss�e,=q�       �	�c�Xc�A�!*

loss�� <�=,M       �	��c�Xc�A�!*

loss؍,=Oi       �	�`d�Xc�A�!*

loss��;nrm�       �	Se�Xc�A�!*

loss�%>=�x��       �	>�e�Xc�A�!*

lossE�h;k��W       �	+Pf�Xc�A�!*

losss�;�	��       �	��f�Xc�A�!*

loss �G;-iK�       �	��g�Xc�A�!*

loss��=��K       �	�3h�Xc�A�!*

loss���;/�h       �	��h�Xc�A�!*

loss�(<�$       �	mui�Xc�A�!*

loss��<F{��       �	%j�Xc�A�!*

loss���<���       �	��j�Xc�A�!*

lossC)<��       �	GZk�Xc�A�!*

loss�� ="�$�       �	��k�Xc�A�!*

loss3�<�VV�       �	�l�Xc�A�!*

loss�
M<M��       �	�%m�Xc�A�!*

loss�E=ϞK       �	Z�m�Xc�A�!*

loss�;O+�       �	:Zn�Xc�A�!*

loss�=�;�'��       �	��n�Xc�A�!*

loss��	=v���       �	z�o�Xc�A�!*

lossH��;�ț       �	&p�Xc�A�!*

losss�<�ʰ�       �	��p�Xc�A�!*

loss@�G=i��       �	�[q�Xc�A�!*

loss�g�<���"       �	�q�Xc�A�!*

loss�%8;E�-c       �	�r�Xc�A�!*

lossq=�<M���       �	�'s�Xc�A�!*

lossv�0=��:       �	�s�Xc�A�!*

loss��T;�r       �	�St�Xc�A�!*

lossw�;�Dh�       �	��t�Xc�A�!*

loss���;v�ԟ       �	��u�Xc�A�!*

loss9�<��5       �	��v�Xc�A�!*

loss[H�<أ�z       �	�:w�Xc�A�!*

loss���;G�4�       �	k�w�Xc�A�!*

loss8��<~kL       �	�mx�Xc�A�!*

loss���<�H��       �	�y�Xc�A�!*

loss�4�<I�z4       �	`�y�Xc�A�!*

loss��=�D u       �	,Hz�Xc�A�!*

loss��<���       �	��z�Xc�A�!*

loss���;�.��       �	D�{�Xc�A�!*

loss{;�< �R       �	T|�Xc�A�!*

loss�w�<�       �	�|�Xc�A�!*

loss�H�<����       �	wN}�Xc�A�!*

loss��	<�t9       �	��}�Xc�A�!*

loss�	<zń�       �	�~�Xc�A�!*

lossv��:�&��       �	�7�Xc�A�!*

loss:�x<�L�       �	���Xc�A�!*

loss"�<�Ԃ       �	=��Xc�A�!*

loss��:<�.�c       �	.���Xc�A�!*

losso�=����       �	.v��Xc�A�!*

loss��h=^�H�       �	_��Xc�A�!*

loss��B;3���       �	����Xc�A�!*

loss�ġ<xAVA       �	���Xc�A�!*

loss}/<n2��       �	cA��Xc�A�!*

lossxHB;+       �	���Xc�A�!*

lossp=���       �	����Xc�A�!*

lossVn�;i0�a       �	F'��Xc�A�!*

loss9;=��+       �	�ֈ�Xc�A�"*

loss��)=�$�$       �	�{��Xc�A�"*

loss�s!=�/�=       �	Y��Xc�A�"*

loss���<�b@�       �	ծ��Xc�A�"*

loss�8;�ku       �	�_��Xc�A�"*

loss ) ;[��       �	Z��Xc�A�"*

loss?�n=#/�S       �	���Xc�A�"*

loss�<@=~s��       �	Q2��Xc�A�"*

lossv�(<�6�O       �	O΍�Xc�A�"*

loss���<�k��       �	�t��Xc�A�"*

loss�"<���       �	��Xc�A�"*

loss�@<�Qi�       �	����Xc�A�"*

lossaG;�Ǡ�       �	=H��Xc�A�"*

loss�F�=��       �	w���Xc�A�"*

lossx��<� ��       �	A���Xc�A�"*

loss���:A\�W       �	Gɒ�Xc�A�"*

lossv��=�+�       �	�i��Xc�A�"*

loss;�e<�5.       �	㦔�Xc�A�"*

loss�=��+       �	�=��Xc�A�"*

loss�֫<��       �	jە�Xc�A�"*

lossXR�;�7=�       �	2s��Xc�A�"*

loss�*="f       �	��Xc�A�"*

lossn
=�ln       �	5���Xc�A�"*

loss��1;4@m       �	Na��Xc�A�"*

loss�uY<�E38       �	& ��Xc�A�"*

loss��<?��       �	���Xc�A�"*

lossz=,��#       �	�~��Xc�A�"*

loss5��<emb       �	���Xc�A�"*

loss�T�;9��       �	����Xc�A�"*

loss��:��v       �	�\��Xc�A�"*

lossSW�<\f\5       �	���Xc�A�"*

lossk;�_�       �	�-��Xc�A�"*

loss�j;���%       �	�R��Xc�A�"*

loss���<��Z;       �	���Xc�A�"*

loss�2=;G�       �	lB��Xc�A�"*

loss�ʇ<��@U       �	�X��Xc�A�"*

lossؽ`<���       �	� ��Xc�A�"*

loss �=)�5       �	�	��Xc�A�"*

loss��"=��y       �	����Xc�A�"*

loss� =�俖       �	˻��Xc�A�"*

loss��<�X�z       �	�Y��Xc�A�"*

lossd<k~��       �	A��Xc�A�"*

loss��;�~��       �	���Xc�A�"*

loss��9��ei       �	υ��Xc�A�"*

loss�
<ǩ�       �	|*��Xc�A�"*

loss4�;p��       �	���Xc�A�"*

lossl�Q='J��       �	����Xc�A�"*

loss��=�.�       �	�&��Xc�A�"*

loss<�gu       �	3��Xc�A�"*

lossX�<\ͷ&       �	Z٭�Xc�A�"*

loss7��;Npm       �	�y��Xc�A�"*

loss7N=���+       �	��Xc�A�"*

loss��)=U:j�       �	`���Xc�A�"*

loss��;+*_�       �	hY��Xc�A�"*

loss��<ĵZ       �	��Xc�A�"*

loss̈́7<޿׀       �	䣱�Xc�A�"*

loss�ک=�i̒       �	h?��Xc�A�"*

loss�.=B��       �	Rֲ�Xc�A�"*

loss�Ec<�l�m       �	�x��Xc�A�"*

loss��J<n�þ       �	��Xc�A�"*

loss��<�I��       �	o���Xc�A�"*

loss��<�>�       �	�\��Xc�A�"*

loss �;�Q       �	���Xc�A�"*

loss�z�<�>       �	����Xc�A�"*

lossƑ=����       �	;7��Xc�A�"*

lossػo=$j��       �	�߷�Xc�A�"*

loss*	c<��0       �	�~��Xc�A�"*

lossFW	=��L�       �	} ��Xc�A�"*

loss�j=n#V       �	V���Xc�A�"*

loss��;}4��       �	�Z��Xc�A�"*

loss7�;X��       �	b���Xc�A�"*

lossw��;ޥ	       �	����Xc�A�"*

loss�uq<D��p       �	@��Xc�A�"*

lossrݵ<#U��       �	���Xc�A�"*

loss[��;o']"       �	͒��Xc�A�"*

loss�E=!�       �	@��Xc�A�"*

loss�q<`Ui�       �	��Xc�A�"*

loss�<(��#       �	���Xc�A�"*

loss��<1�Z       �	e4��Xc�A�"*

loss�0�<���       �	W���Xc�A�"*

loss)M�;�1��       �	�t��Xc�A�"*

lossu��<����       �	*��Xc�A�"*

lossD��<6A�       �	����Xc�A�"*

lossN��<�ڲ       �	
h��Xc�A�"*

loss���<l?#       �	t
��Xc�A�"*

loss��f< �       �	����Xc�A�"*

loss��<�#J       �	X9��Xc�A�"*

loss�b=O�E]       �	$���Xc�A�"*

loss�u�:	��       �	�l��Xc�A�"*

loss7;<�Ώ�       �	���Xc�A�"*

lossM�<���       �	����Xc�A�"*

loss�;=�s��       �	�>��Xc�A�"*

loss�+�<�B�:       �	����Xc�A�"*

loss��6=��G       �	Jz��Xc�A�"*

lossx��<���       �	4��Xc�A�"*

loss�,<[T��       �	����Xc�A�"*

loss��f;�l       �	Y���Xc�A�"*

loss&6< �S       �	m��Xc�A�"*

loss�z�<=ua       �	����Xc�A�"*

loss�<��3�       �	�Z��Xc�A�"*

lossA��<P�7       �	A���Xc�A�"*

loss[��<6�(       �	D���Xc�A�"*

loss�� <WoOu       �	&p��Xc�A�"*

loss��,;i��       �	��Xc�A�"*

lossS�<܎�       �	���Xc�A�"*

loss���<7k��       �	E��Xc�A�"*

loss��M<	V       �	D���Xc�A�"*

loss�|8=�\�-       �	%y��Xc�A�"*

loss���<���       �	���Xc�A�"*

lossx;@<jmI       �	���Xc�A�"*

loss��D<�N�       �	�Z��Xc�A�"*

loss4"=LXx-       �	o���Xc�A�"*

loss<r�<:�       �	ѓ��Xc�A�"*

loss�=@J       �	q9��Xc�A�"*

lossD<�s�D       �	���Xc�A�"*

loss4�<9��M       �	og��Xc�A�"*

loss
��<.ɡ�       �	���Xc�A�"*

loss��<+�w�       �	"���Xc�A�"*

lossߎb=�]E;       �	�?��Xc�A�"*

lossn��;d"R       �	��Xc�A�"*

loss�X.=���       �	H���Xc�A�"*

lossXAu<��       �	2<��Xc�A�"*

loss�T�=i�       �	c���Xc�A�"*

loss�4�=�v       �	k��Xc�A�"*

loss��<$�2       �	t��Xc�A�"*

loss|��;h�[�       �	����Xc�A�"*

loss�J<�l��       �	,I��Xc�A�"*

loss1��<j��       �	H���Xc�A�"*

loss�e0=9�9       �	�|��Xc�A�#*

loss�e;����       �	&��Xc�A�#*

loss���<�^{�       �	5���Xc�A�#*

loss(��;�T>�       �	S��Xc�A�#*

loss��7<vვ       �	����Xc�A�#*

loss�g�<��N�       �	1~��Xc�A�#*

loss�=��X       �	���Xc�A�#*

loss\��<q��       �	д��Xc�A�#*

loss��8=
.z       �	L��Xc�A�#*

loss�t�<���b       �	����Xc�A�#*

loss7S�:}Ќ)       �	(��Xc�A�#*

loss�Y ;�%*       �	P��Xc�A�#*

loss
u�<���       �	o���Xc�A�#*

loss�ٲ<2(�y       �	�T��Xc�A�#*

loss�>�;F^[       �	����Xc�A�#*

loss,�<ͅ8�       �	ʨ��Xc�A�#*

loss�U�=��+       �	�x��Xc�A�#*

loss�Ԫ<�Z?       �	!��Xc�A�#*

loss���<��|       �	e���Xc�A�#*

lossO:�<X{<~       �	_���Xc�A�#*

losscL=�i3�       �	�6��Xc�A�#*

loss�j=���       �	[���Xc�A�#*

loss;N
;�$�_       �	�q��Xc�A�#*

losss�;�E��       �	���Xc�A�#*

loss���=�*2       �	����Xc�A�#*

loss���<���        �	�N��Xc�A�#*

loss�|�=^��;       �	d���Xc�A�#*

loss�@�;ڌyW       �	r���Xc�A�#*

loss�ȉ<p�j!       �	���Xc�A�#*

loss�3m=�́S       �	;T��Xc�A�#*

loss��= ��%       �	����Xc�A�#*

loss���<�8�P       �	.���Xc�A�#*

loss,9=��_R       �	�(��Xc�A�#*

lossQ�}=6j�       �	9���Xc�A�#*

loss[U;u�&       �	����Xc�A�#*

loss�<KO@J       �	�5��Xc�A�#*

lossδ�<��       �	���Xc�A�#*

loss�#=ڜ�       �	.���Xc�A�#*

lossX��<J97�       �	�)��Xc�A�#*

loss���<q��q       �	����Xc�A�#*

loss!�<([n�       �	�T��Xc�A�#*

loss_�=����       �	�&��Xc�A�#*

loss��w=.�       �	����Xc�A�#*

loss7�<{M��       �	�Z��Xc�A�#*

loss��:��S5       �	���Xc�A�#*

loss���<S]�       �	Ƈ��Xc�A�#*

loss�y.=�cA       �	���Xc�A�#*

loss�{�<���       �	)���Xc�A�#*

losshϼ<��9/       �	nL��Xc�A�#*

loss��<����       �	���Xc�A�#*

lossO@t=[2��       �	�� �Xc�A�#*

loss|��<��[C       �	��Xc�A�#*

lossx{g<U��       �	B��Xc�A�#*

loss	�;��=�       �	ZH�Xc�A�#*

loss �6;:��Y       �	���Xc�A�#*

loss3�\=���       �	���Xc�A�#*

lossȢ<)��       �	t%�Xc�A�#*

loss��<H�œ       �	���Xc�A�#*

loss"�<��N:       �	�f�Xc�A�#*

loss�w�<�g�G       �	��Xc�A�#*

loss}QT:z�<       �	���Xc�A�#*

lossa_�:+|�(       �	�?�Xc�A�#*

loss�2<}A�       �	A��Xc�A�#*

losso�=ߐ        �	ʊ�Xc�A�#*

loss�|�<}gj�       �	�#	�Xc�A�#*

loss��=����       �	��	�Xc�A�#*

loss&�=���q       �	�a
�Xc�A�#*

loss%m<4�"       �	6�Xc�A�#*

loss���<��S       �	��Xc�A�#*

loss��<��T       �	A�Xc�A�#*

loss
k�;$'�s       �	���Xc�A�#*

lossz�O:�?e�       �	Ί�Xc�A�#*

loss��<���c       �	S@�Xc�A�#*

loss�=G<o�+       �	���Xc�A�#*

lossA��<�	�       �	��Xc�A�#*

loss�,'<p�6G       �	��Xc�A�#*

loss���<�9!       �	��Xc�A�#*

loss �;��y2       �	Hl�Xc�A�#*

loss9M=:��p       �	
�Xc�A�#*

lossڮ�=-ʶ,       �	)��Xc�A�#*

losseE8=FƋ�       �	�U�Xc�A�#*

loss@�<��,y       �	���Xc�A�#*

loss�:<-+�       �	���Xc�A�#*

loss
�g<q��       �	�B�Xc�A�#*

lossE�O=��z       �	���Xc�A�#*

loss���=�B��       �	���Xc�A�#*

loss_��<�x :       �	�f�Xc�A�#*

loss ?x<x�V~       �	�Xc�A�#*

lossMP�<(�f�       �	L��Xc�A�#*

loss���<򮨃       �	�B�Xc�A�#*

lossL��<"?��       �	 ��Xc�A�#*

loss[)�<��p�       �	���Xc�A�#*

lossZ\�;���       �	���Xc�A�#*

loss���<*�(I       �	6�Xc�A�#*

loss/g'<&        �	���Xc�A�#*

loss9��<R�j%       �	�f�Xc�A�#*

loss��d=��H       �	�D�Xc�A�#*

loss��=��I       �	g��Xc�A�#*

lossH=��	       �	��Xc�A�#*

lossY
=*X�       �	J$ �Xc�A�#*

loss�m�=��]�       �	�� �Xc�A�#*

lossx!�=īT       �	��!�Xc�A�#*

lossT��<�\N�       �	S"�Xc�A�#*

loss
�(=���       �	N�"�Xc�A�#*

loss�1�<#�Yc       �	��#�Xc�A�#*

loss���:����       �	T$�Xc�A�#*

loss��T<�Wtk       �	��$�Xc�A�#*

loss�=S�       �	�Q%�Xc�A�#*

loss��<�       �	5�%�Xc�A�#*

loss�o�:���       �	L�&�Xc�A�#*

loss�hH;� 'y       �	�)'�Xc�A�#*

loss��<k~n       �	l�'�Xc�A�#*

loss�=���r       �	�d(�Xc�A�#*

lossT(�=���J       �	�)�Xc�A�#*

loss�È<~p�       �	O�)�Xc�A�#*

loss�<W<V       �	�+�Xc�A�#*

loss�$�=@XG       �	E,�Xc�A�#*

loss�:{�q       �	��,�Xc�A�#*

lossLgf<��       �	�-�Xc�A�#*

loss���;�,�       �	�*.�Xc�A�#*

loss��?<�0j�       �	~�.�Xc�A�#*

loss�v�;�1+       �	�v/�Xc�A�#*

loss��s<_wY       �	0�Xc�A�#*

loss�	 ;���       �	߿0�Xc�A�#*

losse(<<?o       �	�Z1�Xc�A�#*

lossZ�v<�A��       �	:2�Xc�A�#*

lossW�<`�R       �	=�2�Xc�A�#*

loss"~�<'w�j       �	{.3�Xc�A�#*

loss�5?<G��       �	��3�Xc�A�$*

loss8��<��Ki       �	n35�Xc�A�$*

loss�A�<ȷ��       �	��5�Xc�A�$*

lossy�=� �       �	�y6�Xc�A�$*

loss/�<��
�       �	�!7�Xc�A�$*

loss;
=�3��       �	0+8�Xc�A�$*

lossa$=���       �	�k9�Xc�A�$*

loss�:/=* ��       �	G:�Xc�A�$*

loss�=��8�       �	��:�Xc�A�$*

lossWJ�;�xe       �	��;�Xc�A�$*

loss�N=�~��       �	�;<�Xc�A�$*

loss]�s<_�RH       �	��<�Xc�A�$*

lossW�=�Ф~       �	�=�Xc�A�$*

lossS"h<X�        �	�>�Xc�A�$*

loss��<^�V�       �	�>�Xc�A�$*

loss�A;�u%�       �	�H?�Xc�A�$*

loss�b_=�N�       �	��?�Xc�A�$*

loss�c$=dA       �	-�@�Xc�A�$*

loss�b�<�ߤ�       �	1A�Xc�A�$*

loss��g=/k�       �	6�A�Xc�A�$*

losso�<(kb       �	�cB�Xc�A�$*

loss|�[<�-Ծ       �	��B�Xc�A�$*

loss��<Gp��       �	O�C�Xc�A�$*

loss�l�=��       �	�,D�Xc�A�$*

loss@ؓ;<�?�       �	��D�Xc�A�$*

lossrhD<����       �	�jE�Xc�A�$*

loss��;c�p       �	�8F�Xc�A�$*

loss���<�"D�       �	��F�Xc�A�$*

loss�a�<����       �	�vG�Xc�A�$*

loss�3�;gV}�       �	�H�Xc�A�$*

loss㉵<J��       �	ѲH�Xc�A�$*

loss���;jZ.�       �	�MI�Xc�A�$*

lossWP�<? i       �	y�I�Xc�A�$*

loss�\�<�z�G       �	A�J�Xc�A�$*

loss��m=1W�       �	�K�Xc�A�$*

loss�%�<��        �	Z�K�Xc�A�$*

lossh��<ب�3       �	�VL�Xc�A�$*

loss��3;���       �	/�L�Xc�A�$*

loss�u�<a��]       �	��M�Xc�A�$*

loss�e�<M$�       �	�<N�Xc�A�$*

loss���<�	�       �	Q�N�Xc�A�$*

loss�R<D�8T       �	��O�Xc�A�$*

loss/Q=4G        �	BP�Xc�A�$*

loss�8j<1T��       �	��P�Xc�A�$*

loss��[<3z�       �	�Q�Xc�A�$*

loss�8�;q$Q       �	xAR�Xc�A�$*

lossC�'=+e�       �	)�R�Xc�A�$*

loss��.=�p��       �	 �S�Xc�A�$*

lossvJ
=�@       �	d>T�Xc�A�$*

loss-R�<�iV�       �	�T�Xc�A�$*

lossMV�;ؑ�       �	n�U�Xc�A�$*

loss@��;����       �	|*V�Xc�A�$*

loss��Z<3��       �	��V�Xc�A�$*

lossf{;}�       �	�hW�Xc�A�$*

loss�K;I���       �	?X�Xc�A�$*

loss
\~;��O�       �	w�X�Xc�A�$*

lossڢz=���y       �	�?Y�Xc�A�$*

loss��7<��oW       �	%�Y�Xc�A�$*

loss��
=��}?       �	n�Z�Xc�A�$*

loss���<Qb��       �		2[�Xc�A�$*

loss_�a<7��       �	�[�Xc�A�$*

loss__�;MF!�       �	�x\�Xc�A�$*

loss$$�;�m�       �	�-]�Xc�A�$*

loss}ȿ<����       �	.^�Xc�A�$*

loss�M�<i�z)       �	�^�Xc�A�$*

loss�_�<x�9�       �	l`�Xc�A�$*

loss��=3��Q       �	w�`�Xc�A�$*

losss�_;!D��       �	�<a�Xc�A�$*

loss�d�:@�t       �	��a�Xc�A�$*

loss�r�:qi,�       �	tb�Xc�A�$*

loss���;���       �	8�c�Xc�A�$*

loss�/<|[�       �	�d�Xc�A�$*

lossA�;=;�       �	g�d�Xc�A�$*

loss)L=r�       �	aUe�Xc�A�$*

lossL/r=�q�{       �	(�e�Xc�A�$*

loss��G=���       �	ގf�Xc�A�$*

loss�Q%=�5�       �	h%g�Xc�A�$*

loss
u�;�m�       �	��g�Xc�A�$*

lossҊ;
u�C       �	�`h�Xc�A�$*

lossJd�:�^�       �	a�h�Xc�A�$*

loss� =����       �	��i�Xc�A�$*

lossO<]��;       �	�9j�Xc�A�$*

loss&�<1@��       �	��j�Xc�A�$*

losswJ7=�ٱ�       �	|k�Xc�A�$*

lossax�;5�p�       �	\l�Xc�A�$*

loss�Ha:7�2y       �	��l�Xc�A�$*

loss���;Hj��       �	�bm�Xc�A�$*

loss��<�v:�       �	�n�Xc�A�$*

lossd>�<Zsp�       �	ܜn�Xc�A�$*

loss�j<�مk       �	H7o�Xc�A�$*

loss�l<��pD       �	��o�Xc�A�$*

loss�`u<��Χ       �	�mp�Xc�A�$*

loss�ĵ<�>H       �	Kq�Xc�A�$*

lossZ�9=�TA       �	�q�Xc�A�$*

loss\��:ЎG       �	�8r�Xc�A�$*

loss	��<B
       �	�r�Xc�A�$*

losst�;���       �	I�s�Xc�A�$*

lossT�=���       �	�<t�Xc�A�$*

loss<�<���l       �	(�t�Xc�A�$*

loss��}<A�F/       �	mu�Xc�A�$*

loss���;h_B       �	v�Xc�A�$*

lossI-�=��N5       �	�v�Xc�A�$*

loss��f<{g�       �	�Ew�Xc�A�$*

loss�E�;�)�       �	r�w�Xc�A�$*

loss�p<����       �	�zx�Xc�A�$*

loss8�h<R7]       �	jy�Xc�A�$*

lossN��=Z�QZ       �	��y�Xc�A�$*

loss �O=RM�       �	��z�Xc�A�$*

loss�y@=�)       �	��{�Xc�A�$*

lossMog=Ỏi       �	�|�Xc�A�$*

lossp5�;��]       �	��|�Xc�A�$*

loss6J;~�;�       �	(�}�Xc�A�$*

loss���<��=       �	�4~�Xc�A�$*

loss=p;��ac       �	W�~�Xc�A�$*

loss���<���       �	�c�Xc�A�$*

loss�c<�i��       �	���Xc�A�$*

loss��h<y��[       �	����Xc�A�$*

loss��=.{��       �	#.��Xc�A�$*

loss��l<��       �	�ց�Xc�A�$*

lossn%�;��H�       �	Tr��Xc�A�$*

loss�9�<]�]�       �	(��Xc�A�$*

loss}_�<��Y�       �	+���Xc�A�$*

loss�N<<�u        �	?��Xc�A�$*

loss��C;D�ʟ       �	�ۄ�Xc�A�$*

loss�iT=D�S       �	:v��Xc�A�$*

lossI-R:i)��       �	���Xc�A�$*

loss�ѯ9�r��       �	���Xc�A�$*

loss��;��       �	cG��Xc�A�$*

loss�U;�{��       �	^���Xc�A�%*

loss�s�;\,�D       �	�2��Xc�A�%*

lossSSD<�&�<       �	�ى�Xc�A�%*

loss#�;�!��       �	�n��Xc�A�%*

loss��;P�~/       �	���Xc�A�%*

loss��:��u       �	U�Xc�A�%*

loss�u9�
/P       �	�z��Xc�A�%*

loss�V�9��f�       �	���Xc�A�%*

loss���:}7۸       �	���Xc�A�%*

lossT��<����       �	�G��Xc�A�%*

lossM��;��=       �	���Xc�A�%*

loss���9aߌ       �	����Xc�A�%*

loss�u:�; �       �	�%��Xc�A�%*

loss_��=��'m       �	����Xc�A�%*

lossX��9����       �	%[��Xc�A�%*

lossn��;5r�       �	Q���Xc�A�%*

loss��=�dv       �	����Xc�A�%*

loss�~<��e�       �	|��Xc�A�%*

loss�.,<!	u,       �	���Xc�A�%*

loss��<t1��       �	p���Xc�A�%*

loss�&=hk�       �	�M��Xc�A�%*

loss���<��S�       �	���Xc�A�%*

loss���<��        �	����Xc�A�%*

loss�=����       �	�V��Xc�A�%*

loss��;H�       �	'��Xc�A�%*

loss���<��b       �	7Ƙ�Xc�A�%*

loss�+�<�%�w       �	�d��Xc�A�%*

loss
{ <��3       �	@���Xc�A�%*

loss�w�<�6�5       �	����Xc�A�%*

loss�ֈ< �X�       �	!:��Xc�A�%*

loss̝<Mo�       �	nݛ�Xc�A�%*

lossI�;�(T�       �	&���Xc�A�%*

loss���=v���       �	�j��Xc�A�%*

loss�th;7$:�       �	��Xc�A�%*

loss)�:�9[i       �	����Xc�A�%*

lossZ��<��f�       �	���Xc�A�%*

loss�=�K�5       �	����Xc�A�%*

loss`BC=XJݟ       �	�4��Xc�A�%*

loss"�;r��       �	�ϡ�Xc�A�%*

loss���;(c�       �	bi��Xc�A�%*

loss?�=h2�       �	h��Xc�A�%*

loss6!d=�i�       �	���Xc�A�%*

loss�<D^�E       �	�>��Xc�A�%*

lossaO�=S2��       �	,դ�Xc�A�%*

loss�b�;�O�       �	�j��Xc�A�%*

loss��<�D       �	���Xc�A�%*

losssژ<��.       �	|զ�Xc�A�%*

lossG�;2���       �	�l��Xc�A�%*

loss[��;uW��       �	���Xc�A�%*

loss�G<�Mu       �	���Xc�A�%*

loss2I2<���        �	)Z��Xc�A�%*

loss��#<���3       �	����Xc�A�%*

loss1D=H펨       �	&���Xc�A�%*

loss�/�;��       �	S%��Xc�A�%*

lossj�;��x�       �	���Xc�A�%*

losso��<|Z�       �	}���Xc�A�%*

loss�>M<�D#^       �		O��Xc�A�%*

loss��;̚t       �	��Xc�A�%*

loss�j;-��       �	f���Xc�A�%*

lossL��< ���       �	� ��Xc�A�%*

loss��:=���       �	ػ��Xc�A�%*

loss/O?;Sj#9       �	-^��Xc�A�%*

lossN\=���i       �	����Xc�A�%*

loss���;�ѫ?       �	}���Xc�A�%*

lossS��<��       �	�3��Xc�A�%*

loss��O<~~m�       �	����Xc�A�%*

loss��x=��       �	Q���Xc�A�%*

loss� �=��V       �	9��Xc�A�%*

loss��j<Φ�       �	����Xc�A�%*

lossD�D=��       �	>���Xc�A�%*

loss��;�1�       �	6��Xc�A�%*

loss>�<�}��       �	k���Xc�A�%*

loss��=� �       �	�t��Xc�A�%*

lossL*�=^l�       �	�#��Xc�A�%*

loss�W�<FE<�       �	"���Xc�A�%*

loss�;��)v       �	/i��Xc�A�%*

lossѯ<�i"�       �	���Xc�A�%*

lossݽ�=Z���       �	���Xc�A�%*

loss�X:=�=��       �	�>��Xc�A�%*

loss�_<�Ɗ�       �	j���Xc�A�%*

loss�d�;�V��       �	Л��Xc�A�%*

loss�<�&-�       �	@3��Xc�A�%*

loss�Վ;��:       �	����Xc�A�%*

lossֈ�;`�0#       �	����Xc�A�%*

lossR�<���w       �	���Xc�A�%*

lossA�<�\�       �	�E��Xc�A�%*

loss3}F<��       �	����Xc�A�%*

loss��<�m.�       �	\���Xc�A�%*

loss��k=fڵ@       �	z4��Xc�A�%*

lossh�<�G;       �	v���Xc�A�%*

loss�|<��v�       �	����Xc�A�%*

lossv�B=��       �	+��Xc�A�%*

lossw�4;øA#       �	����Xc�A�%*

loss��k=+�ʰ       �	'k��Xc�A�%*

loss�J�<��3X       �	->��Xc�A�%*

loss�z<�r��       �	����Xc�A�%*

loss���< }�       �	����Xc�A�%*

loss�v�<��Y       �	Z��Xc�A�%*

loss�S�=�Y��       �	��Xc�A�%*

loss$f�<��T�       �	����Xc�A�%*

loss�==Vw       �	�Q��Xc�A�%*

loss<>L=ƞ2�       �	����Xc�A�%*

lossP=��.�       �	���Xc�A�%*

loss��=[�܁       �	o)��Xc�A�%*

loss�T=�14       �	����Xc�A�%*

loss[�f<5�Փ       �	Ie��Xc�A�%*

loss&3<!�n       �	d��Xc�A�%*

loss{�x<��ә       �	ܛ��Xc�A�%*

loss[�<����       �	d>��Xc�A�%*

loss7�=dFi       �	P���Xc�A�%*

lossy	�=2HHx       �	����Xc�A�%*

lossUJ=&3�4       �	�^��Xc�A�%*

loss?f�;�g4�       �	��Xc�A�%*

lossr<<���       �	����Xc�A�%*

loss�M]:s�<       �	���Xc�A�%*

lossi��;�p��       �	�h��Xc�A�%*

loss���<ES�       �	���Xc�A�%*

loss���;Ȕl�       �	���Xc�A�%*

lossx�=�c�%       �	DO��Xc�A�%*

loss�F�<ޓ�*       �	����Xc�A�%*

losso�;`R       �	���Xc�A�%*

loss�.: �        �	�+��Xc�A�%*

loss�O�9�s       �	����Xc�A�%*

loss}ԫ;�)�p       �	Ύ��Xc�A�%*

loss��y=�u<�       �	s0��Xc�A�%*

loss�d=z�xg       �	u���Xc�A�%*

loss V<δg5       �	�e��Xc�A�%*

loss
��:���       �	����Xc�A�%*

lossj�<���       �	n���Xc�A�&*

loss49<!�       �	>\��Xc�A�&*

lossj�7=X�       �	{���Xc�A�&*

loss���;����       �	¾��Xc�A�&*

loss�a"<7�I�       �	4g��Xc�A�&*

lossT�#>���0       �	�?��Xc�A�&*

loss�m<K�9       �	����Xc�A�&*

loss�<m';       �	_z��Xc�A�&*

lossc�<�{��       �	���Xc�A�&*

loss�$�;�@7�       �	����Xc�A�&*

loss
0^<���       �	`[��Xc�A�&*

lossʵ�<M<��       �	���Xc�A�&*

loss���=l"�       �	 ���Xc�A�&*

lossh�;5P~�       �	�L��Xc�A�&*

loss��B<���       �	����Xc�A�&*

loss�=B��       �	z���Xc�A�&*

loss�7�<����       �	4I �Xc�A�&*

loss�hj<�k�       �	�� �Xc�A�&*

lossS��<|��       �	y�Xc�A�&*

loss���<�s        �	X�Xc�A�&*

lossNFQ<�s/       �	��Xc�A�&*

loss��+<�B��       �	YP�Xc�A�&*

lossy�;�=d       �	���Xc�A�&*

loss��<���       �	G��Xc�A�&*

loss&�;���       �	�)�Xc�A�&*

loss��<4��       �	7��Xc�A�&*

loss��;Z%       �	\�Xc�A�&*

loss�R�<�kk       �	���Xc�A�&*

losst�=
���       �	K��Xc�A�&*

loss�zw;�촊       �	�*�Xc�A�&*

loss�+�;�u��       �	���Xc�A�&*

loss��=�7>�       �	�_	�Xc�A�&*

loss��<ke�,       �	��	�Xc�A�&*

loss=��<�nA�       �	s�
�Xc�A�&*

loss��<�F�       �	;�Xc�A�&*

loss�Y=�~       �	+��Xc�A�&*

losst#<�&c       �	�|�Xc�A�&*

loss��e<���       �	3�Xc�A�&*

loss[�e;7n�s       �	���Xc�A�&*

loss3�;��K       �	�b�Xc�A�&*

loss9�<���       �	K�Xc�A�&*

loss��=�@v       �	���Xc�A�&*

loss�\�<���       �	�n�Xc�A�&*

loss.��;��a�       �	��Xc�A�&*

loss 3x<��x)       �	;��Xc�A�&*

loss�DC=,�t       �	�M�Xc�A�&*

loss�? =Q*�p       �	v��Xc�A�&*

lossw��=�J�4       �	_��Xc�A�&*

loss�ԇ<��       �	Z��Xc�A�&*

loss���;�d�       �	�=�Xc�A�&*

loss��:A�Nm       �	�,�Xc�A�&*

loss�2�<�[�       �	��Xc�A�&*

loss
5n<��3�       �	���Xc�A�&*

loss
�f=����       �	�@�Xc�A�&*

loss@=Aɔ       �	���Xc�A�&*

lossm�=��       �	,�Xc�A�&*

loss��:�'V�       �	��Xc�A�&*

loss�=����       �	��Xc�A�&*

loss���<�2`f       �	���Xc�A�&*

loss�k�;�b�       �	�G�Xc�A�&*

loss-�*<9��w       �	i��Xc�A�&*

loss�!�;�R.�       �	���Xc�A�&*

loss���<���       �	q�Xc�A�&*

loss_�q<@�r       �	\�Xc�A�&*

loss-N<���o       �	�& �Xc�A�&*

loss�<��W       �	`!�Xc�A�&*

lossiBD;���       �	�6"�Xc�A�&*

loss�<<��       �	��"�Xc�A�&*

loss�-E<K��       �	�#�Xc�A�&*

loss	=ϹX       �	�#$�Xc�A�&*

loss�SS<pJ�;       �	�%�Xc�A�&*

loss�=�0N       �	~�%�Xc�A�&*

lossĳ�<у��       �	A&�Xc�A�&*

lossa��<JA]       �	��&�Xc�A�&*

loss��u<љ��       �	�'�Xc�A�&*

loss���;�&f�       �	#(�Xc�A�&*

loss>�	;h~t�       �	k�(�Xc�A�&*

loss���:	��       �	�_)�Xc�A�&*

loss�X�<&̦W       �	�)�Xc�A�&*

loss�~=u�       �	��*�Xc�A�&*

loss��<!���       �	�+�Xc�A�&*

loss��4<��       �	��+�Xc�A�&*

loss�0<�L�       �	0-�Xc�A�&*

loss���;�7xI       �	>�-�Xc�A�&*

loss��E;O��       �	�U.�Xc�A�&*

loss,\�;���T       �	I�.�Xc�A�&*

loss�R�<}���       �	��/�Xc�A�&*

loss*�;X�G	       �	ZF0�Xc�A�&*

loss��v=���       �	��0�Xc�A�&*

lossRc�<��j       �	x�1�Xc�A�&*

loss{��<�)ŏ       �	�=2�Xc�A�&*

loss�*2<���       �	��2�Xc�A�&*

loss
�);u�       �	q3�Xc�A�&*

loss��;���       �	�4�Xc�A�&*

loss! <�?       �	��4�Xc�A�&*

loss@�=�z�-       �	�O5�Xc�A�&*

loss���;���       �	_�5�Xc�A�&*

lossᦠ<J��       �	��6�Xc�A�&*

loss���;���2       �	/7�Xc�A�&*

loss��;�       �	��7�Xc�A�&*

loss� X:4:�)       �	8k8�Xc�A�&*

loss��O=)� �       �	�9�Xc�A�&*

loss���;�o��       �	2�9�Xc�A�&*

lossXy�;Mɞ0       �	�`:�Xc�A�&*

loss���<��       �	 6;�Xc�A�&*

lossr�l<M~�       �	��;�Xc�A�&*

loss��V<���h       �	i<�Xc�A�&*

loss�E-=�0O�       �	� =�Xc�A�&*

loss�N;B���       �	y�=�Xc�A�&*

loss�D<��6       �	�F>�Xc�A�&*

loss��B<��Ӊ       �	P�>�Xc�A�&*

lossO�	=I��       �	px?�Xc�A�&*

loss��n<v)-       �	�@�Xc�A�&*

loss��e=�.��       �	�@�Xc�A�&*

lossr�=6��       �	�QA�Xc�A�&*

loss�=\�|       �	��A�Xc�A�&*

lossnl�;�iR        �	؁B�Xc�A�&*

loss��:��Q       �	'C�Xc�A�&*

loss��=�)�_       �	��C�Xc�A�&*

loss6�"<�$��       �	�[D�Xc�A�&*

loss/�:���       �	�D�Xc�A�&*

loss�<��R.       �	�E�Xc�A�&*

loss/=[j��       �	sF�Xc�A�&*

lossܐ�;=��D       �	�G�Xc�A�&*

loss�<^��       �	9�G�Xc�A�&*

lossf R=��S�       �	�SH�Xc�A�&*

loss��@=�3       �	L�H�Xc�A�&*

lossA�;γ�-       �	��I�Xc�A�&*

loss@v�;@6�       �	=IJ�Xc�A�'*

loss/b;��<       �	��J�Xc�A�'*

loss�4�<mޯ=       �	ͬK�Xc�A�'*

loss
�x;�       �	l^L�Xc�A�'*

loss|.<�Su       �	 M�Xc�A�'*

loss���<��       �	еM�Xc�A�'*

loss�M=D�       �	}\N�Xc�A�'*

loss�CA=�F�z       �	�	O�Xc�A�'*

lossJbs</���       �	��O�Xc�A�'*

lossv��<�^�       �	�UP�Xc�A�'*

loss(K =�v       �	��P�Xc�A�'*

loss2�<��!�       �	�Q�Xc�A�'*

lossFW<+��       �	�:R�Xc�A�'*

loss6d�<xk"�       �	�R�Xc�A�'*

loss`C=Q�       �	��S�Xc�A�'*

loss��T<�D��       �	%"T�Xc�A�'*

lossRM1<�H�       �	q�T�Xc�A�'*

loss!�=���       �	@jU�Xc�A�'*

loss�]�;�D��       �	8JV�Xc�A�'*

loss���;>y~       �	��V�Xc�A�'*

loss�"�=� ��       �	��W�Xc�A�'*

lossF��<Ot�[       �	j1X�Xc�A�'*

loss�Č<,g9       �	p�X�Xc�A�'*

lossȕ >B���       �	��Y�Xc�A�'*

loss�	6=,�       �	J]Z�Xc�A�'*

loss�R�<?vZ       �	� [�Xc�A�'*

lossxU=��e       �	�[�Xc�A�'*

loss�
='�Z       �	<O\�Xc�A�'*

loss	�D<���       �	-�\�Xc�A�'*

loss`��;AG�|       �	\�]�Xc�A�'*

loss6�<\�g       �	�9^�Xc�A�'*

loss�K�;)i�        �	F#_�Xc�A�'*

loss.�=h�T�       �	]l`�Xc�A�'*

losse�<bԕ�       �	ja�Xc�A�'*

loss��*=� �Q       �	\�a�Xc�A�'*

lossA�i<�*�       �	gb�Xc�A�'*

loss3qr<�3��       �	�Fc�Xc�A�'*

loss�x\<���       �	)d�Xc�A�'*

loss���;dm2�       �	e�Xc�A�'*

loss��Y;����       �	�f�Xc�A�'*

lossv;��I       �	��f�Xc�A�'*

loss:�+=�^r       �	'�g�Xc�A�'*

loss"�"<Y�s�       �	��h�Xc�A�'*

loss�(<�)"�       �	bi�Xc�A�'*

loss\�{<�|{�       �	9j�Xc�A�'*

loss&�f=lO�       �	�j�Xc�A�'*

loss�;<�v�       �	�Qk�Xc�A�'*

lossq+=�"F�       �	<�k�Xc�A�'*

loss�� <���       �	4�l�Xc�A�'*

loss?�;����       �	�>m�Xc�A�'*

loss�P�<�v�       �	)�m�Xc�A�'*

loss2R:<�?A�       �	��n�Xc�A�'*

lossF�=咊�       �	�>o�Xc�A�'*

lossw�<���^       �	p�Xc�A�'*

loss��*<�-(�       �	��p�Xc�A�'*

loss���;nxW�       �	�cq�Xc�A�'*

loss{�;Q���       �	�r�Xc�A�'*

lossl@�<A-�       �	ϣr�Xc�A�'*

loss8�!<��O       �	�Es�Xc�A�'*

loss�ug<J8"       �	v�s�Xc�A�'*

loss1Z=��~H       �	p}t�Xc�A�'*

loss�g<ʼ�+       �	�#u�Xc�A�'*

loss�8=Mę:       �	��u�Xc�A�'*

lossF��:O6C       �	�`v�Xc�A�'*

loss�R�;�\�       �	�v�Xc�A�'*

loss�9=�_&       �	��w�Xc�A�'*

lossQr�<ͫT�       �	��x�Xc�A�'*

loss��<>ڊ�       �	�2y�Xc�A�'*

loss8�?=��       �	��y�Xc�A�'*

loss�<Lbb�       �	�_z�Xc�A�'*

loss,�<�1�d       �	<�z�Xc�A�'*

loss��6=Nk�       �	�(|�Xc�A�'*

lossP=Mp��       �	q�|�Xc�A�'*

loss�`�<Y�B       �	�}�Xc�A�'*

lossn�:�?��       �	�#~�Xc�A�'*

loss���<���m       �	��~�Xc�A�'*

lossa�; j2       �	�S�Xc�A�'*

lossN�Y=9       �	q��Xc�A�'*

loss�8�<�!6X       �	!���Xc�A�'*

loss�i�<�W       �	8.��Xc�A�'*

lossh�<���       �	@ہ�Xc�A�'*

loss۔6<i�O       �	uu��Xc�A�'*

loss�G,=T��       �	�!��Xc�A�'*

loss�a�=~ߓ�       �	Q؃�Xc�A�'*

loss�]�<��6�       �	�n��Xc�A�'*

loss:�<s�U       �	0��Xc�A�'*

loss��.<bZ�c       �	؅�Xc�A�'*

loss�:@<5��       �	�q��Xc�A�'*

loss���<'�B�       �	���Xc�A�'*

loss_�;���       �	����Xc�A�'*

loss�<�;577       �	�C��Xc�A�'*

lossi+
=f�H�       �	i���Xc�A�'*

loss<�=��Q       �	p���Xc�A�'*

lossH�9<���       �	�I��Xc�A�'*

loss*Њ<�	�M       �	���Xc�A�'*

loss֯n=o��       �	{��Xc�A�'*

loss���<G�t�       �	;��Xc�A�'*

loss��<Yo_H       �	�Xc�A�'*

loss@2�;����       �	�D��Xc�A�'*

loss<K<5IP^       �	i��Xc�A�'*

loss�`�<��0�       �	�Î�Xc�A�'*

loss��;��z�       �	�e��Xc�A�'*

loss4�,;]5$�       �	C��Xc�A�'*

loss#8�;��"u       �	.ɐ�Xc�A�'*

loss6�<���       �	Vc��Xc�A�'*

loss��r<�M;G       �	G��Xc�A�'*

lossCT=^�χ       �	˝��Xc�A�'*

lossv�[=ݏp�       �	?��Xc�A�'*

lossq��<-a��       �	����Xc�A�'*

loss�|y=Z?�       �	����Xc�A�'*

loss���<�%�       �	k'��Xc�A�'*

loss��%;C��       �	"���Xc�A�'*

loss	�<$i��       �	]��Xc�A�'*

loss�*�<�1�       �	����Xc�A�'*

lossߥ�=.
       �	�ї�Xc�A�'*

loss(��:��       �	h��Xc�A�'*

loss�AY=�Ip�       �	����Xc�A�'*

lossD�<'�\�       �	����Xc�A�'*

loss��:=��{       �	�0��Xc�A�'*

loss�B(<���=       �	Cɚ�Xc�A�'*

loss�,<6^��       �	~q��Xc�A�'*

loss ��<˨,e       �	���Xc�A�'*

lossR�%;�|�1       �	嘜�Xc�A�'*

lossdI�:s�D       �	'.��Xc�A�'*

lossW��<_m�       �	�ǝ�Xc�A�'*

lossT��:�M�       �	[^��Xc�A�'*

loss�¢<>�δ       �	V��Xc�A�'*

loss�W=Q��       �	ߩ��Xc�A�'*

loss;)F<�Q��       �	����Xc�A�(*

lossS�<9���       �	�V��Xc�A�(*

lossSd=S�A       �	����Xc�A�(*

loss�$<n��       �	d���Xc�A�(*

loss<.<��S@       �	e9��Xc�A�(*

loss��0=\S$;       �	Fϣ�Xc�A�(*

lossR��<�j"�       �	nj��Xc�A�(*

lossN�<%3}       �	� ��Xc�A�(*

lossdg�<i�Y       �	W���Xc�A�(*

loss�ȉ<�?�       �	�(��Xc�A�(*

loss=�=�U�R       �	����Xc�A�(*

lossd]P<��a       �	�W��Xc�A�(*

loss2�U<�R�       �	R,��Xc�A�(*

lossh��:G��        �	gҨ�Xc�A�(*

loss��c<͘�       �	�l��Xc�A�(*

lossrlP<�I�       �	N
��Xc�A�(*

lossE��;U���       �	���Xc�A�(*

loss33m<��IK       �	+P��Xc�A�(*

lossM� =笾{       �	����Xc�A�(*

loss�^�<m�qs       �	p���Xc�A�(*

lossB�:���,       �	.9��Xc�A�(*

loss��=�_o       �	oխ�Xc�A�(*

loss�#A<ru-�       �	�z��Xc�A�(*

loss��<���       �	j��Xc�A�(*

loss�*=h��       �	软�Xc�A�(*

lossvw=��r-       �	R`��Xc�A�(*

loss���=���       �	C ��Xc�A�(*

loss���<�Mi�       �	-α�Xc�A�(*

loss�0�<���       �	
e��Xc�A�(*

loss2M�<���       �	� ��Xc�A�(*

lossq-�<�l�       �	]���Xc�A�(*

loss	[_<�K�       �	�;��Xc�A�(*

loss݊<K���       �	O��Xc�A�(*

lossx�B<Gn�W       �	����Xc�A�(*

loss�7y<��l       �	�&��Xc�A�(*

loss���<�8*       �	zƶ�Xc�A�(*

loss��=w0��       �	ke��Xc�A�(*

loss�9�;�M"�       �	���Xc�A�(*

loss-V�;d��       �	���Xc�A�(*

lossS�<.v��       �	�G��Xc�A�(*

loss���<�k�       �	���Xc�A�(*

loss��=L�Ǚ       �	�~��Xc�A�(*

loss�3;yWO       �	P��Xc�A�(*

lossӝU=���       �	x���Xc�A�(*

loss��/=p�'       �	�N��Xc�A�(*

loss�S�<[\Nr       �	���Xc�A�(*

lossC`�<,L�       �	/���Xc�A�(*

loss;��=Y�       �	9D��Xc�A�(*

loss�b�<�e�l       �	�۾�Xc�A�(*

lossE�<���w       �	k���Xc�A�(*

lossz��<
O!       �	e��Xc�A�(*

loss�#�<�TuU       �	����Xc�A�(*

loss:�<�zg       �	�M��Xc�A�(*

loss&��<��W�       �	����Xc�A�(*

lossD].;u�S       �	e���Xc�A�(*

loss���<h~{       �	J%��Xc�A�(*

loss�2�<����       �	����Xc�A�(*

loss^�<( S�       �	�b��Xc�A�(*

lossFh%=���V       �	���Xc�A�(*

losst�=	�       �	����Xc�A�(*

loss �!=�o)%       �	�0��Xc�A�(*

loss�f-;�q       �	����Xc�A�(*

loss�-�<�Ό       �	 s��Xc�A�(*

loss$�<?���       �	���Xc�A�(*

losss�F<@��I       �	a���Xc�A�(*

loss�J�;Qʩ       �	�M��Xc�A�(*

loss,�`<�͗l       �	���Xc�A�(*

loss�wi=6|�       �	����Xc�A�(*

loss\6�<�V       �	�$��Xc�A�(*

lossz�|;����       �	'���Xc�A�(*

lossY(=Ɍ��       �	�h��Xc�A�(*

loss}ύ;���'       �	��Xc�A�(*

lossL<�
       �	����Xc�A�(*

loss��=){9�       �	�:��Xc�A�(*

loss�=;���       �	����Xc�A�(*

loss�w0<{�*U       �	~o��Xc�A�(*

loss���=xHg�       �	�
��Xc�A�(*

loss���;G3f�       �	���Xc�A�(*

loss�Q=.A]       �	 {��Xc�A�(*

lossh��<��I       �	���Xc�A�(*

loss[]<
]��       �	����Xc�A�(*

lossd3�<�j��       �	�]��Xc�A�(*

loss��9=mYgL       �	&���Xc�A�(*

loss��S=i�6�       �	����Xc�A�(*

lossL�=���       �	�W��Xc�A�(*

loss(!e<�J5       �	���Xc�A�(*

loss���;g��.       �	1���Xc�A�(*

loss`�Q<ge��       �	�5��Xc�A�(*

loss,�P<�̂       �	���Xc�A�(*

loss�<p�h}       �	����Xc�A�(*

lossZ�=M�ߞ       �	lA��Xc�A�(*

loss�E8=�`F       �	
���Xc�A�(*

loss���<s��        �	*t��Xc�A�(*

lossSeZ<�=H       �	���Xc�A�(*

loss��=R��=       �	a���Xc�A�(*

losso<���       �	F]��Xc�A�(*

loss�0<����       �	j���Xc�A�(*

lossQ�;gp��       �	����Xc�A�(*

loss4�<�qB�       �	;7��Xc�A�(*

loss�x><d*��       �	����Xc�A�(*

loss�<��       �	�v��Xc�A�(*

lossaO�;���       �	���Xc�A�(*

loss���<� �G       �	����Xc�A�(*

loss��k<��       �	�^��Xc�A�(*

lossX��=�qD�       �	����Xc�A�(*

loss��<J��       �	����Xc�A�(*

loss��R<���       �	'g��Xc�A�(*

lossk�=�ۃc       �	v���Xc�A�(*

loss���<��       �	E���Xc�A�(*

loss���<��@       �	H4��Xc�A�(*

lossɽ�;�#       �	���Xc�A�(*

loss�o�=�^K       �	y��Xc�A�(*

loss� <Pa       �	�!��Xc�A�(*

lossh��;���@       �	|���Xc�A�(*

loss��<&M�~       �	�e��Xc�A�(*

lossz�*=:s0�       �	����Xc�A�(*

lossd�<�9�       �	J���Xc�A�(*

loss�E<(���       �	/��Xc�A�(*

lossr9r=�u��       �	����Xc�A�(*

lossm��<�?�       �	Ui��Xc�A�(*

loss���<�T�"       �	x���Xc�A�(*

lossJ�%=�;�Q       �	�\��Xc�A�(*

loss�)=�U       �	i��Xc�A�(*

loss��5=z��       �	���Xc�A�(*

lossq�<O<�8       �	�>��Xc�A�(*

loss�Y�:�nT       �	^���Xc�A�(*

loss�؞<����       �	�|��Xc�A�(*

lossiA�=9�;       �	 ��Xc�A�(*

lossJ�<�D��       �	����Xc�A�)*

lossFB<=��4       �	�M��Xc�A�)*

loss)=��       �	����Xc�A�)*

losst�=�}k       �	0��Xc�A�)*

lossI�1<WM<E       �	�.��Xc�A�)*

loss��<��&       �	.���Xc�A�)*

loss��;���       �	q��Xc�A�)*

loss�KC=�s@9       �	���Xc�A�)*

loss��<����       �	���Xc�A�)*

lossdF�<�i       �	�Z��Xc�A�)*

loss���<�a�K       �	��Xc�A�)*

loss%;'"�       �	צ��Xc�A�)*

lossU.�<b��       �	�R��Xc�A�)*

losssKJ<��Ԃ       �	6���Xc�A�)*

loss��i<�z�       �	W���Xc�A�)*

lossW2;����       �	@O��Xc�A�)*

lossҥ�=�|yd       �	�C��Xc�A�)*

lossr��=�:a�       �	'���Xc�A�)*

losseK=M�E       �	:u��Xc�A�)*

loss�j=\��G       �	P��Xc�A�)*

loss���;ՙ>o       �	�� �Xc�A�)*

loss�y)<���       �	�6�Xc�A�)*

loss]��;^��       �	���Xc�A�)*

lossl��<t̩�       �	du�Xc�A�)*

lossh��;"sNe       �	V�Xc�A�)*

loss��i<�@�       �	:��Xc�A�)*

loss���=���       �	�V�Xc�A�)*

loss�'8;��=       �	��Xc�A�)*

loss���;�K�|       �	_��Xc�A�)*

loss��:����       �	�/�Xc�A�)*

lossi1<p��       �	Y��Xc�A�)*

loss�*�<κ0{       �	���Xc�A�)*

loss_4�<Y|�t       �	p&�Xc�A�)*

loss�]7=�!       �	���Xc�A�)*

loss��<����       �	t	�Xc�A�)*

loss8w�<��HB       �	*
�Xc�A�)*

loss�!`;��5.       �	��
�Xc�A�)*

loss�k�<&]y�       �	�^�Xc�A�)*

loss �x<��H�       �	%�Xc�A�)*

lossN�*<2�|h       �	m��Xc�A�)*

loss�Y=A��       �	�O�Xc�A�)*

loss)�3<k�       �	���Xc�A�)*

loss�T=����       �	b��Xc�A�)*

loss|(�<)�F       �	#�Xc�A�)*

loss���;��a       �	���Xc�A�)*

lossʺ:;��rc       �	�l�Xc�A�)*

loss�0�<��`       �	8�Xc�A�)*

loss�x;}���       �	^��Xc�A�)*

loss�Z;5Ky�       �	�b�Xc�A�)*

loss=��<>��Y       �	��Xc�A�)*

loss*�;�"w�       �	��Xc�A�)*

loss��(=���       �	�L�Xc�A�)*

loss� =�g.�       �	���Xc�A�)*

lossqqy<�d       �	���Xc�A�)*

losso��:M�!       �	v3�Xc�A�)*

loss�D�:����       �	���Xc�A�)*

lossq~�;��!       �	Xr�Xc�A�)*

loss�5b<�-�       �	��Xc�A�)*

loss��=<�ӭ�       �	:��Xc�A�)*

lossJ35<�<��       �	�d�Xc�A�)*

loss{�.<��^p       �	K�Xc�A�)*

loss�a=:���       �	#��Xc�A�)*

loss�T�<S���       �	�@�Xc�A�)*

loss���;P�x       �	���Xc�A�)*

loss�H�<v��       �	�u�Xc�A�)*

lossI�u<�N�       �	{2�Xc�A�)*

loss�Di=��s       �	,��Xc�A�)*

loss\�x<���       �	�G�Xc�A�)*

loss�� =��HC       �	���Xc�A�)*

loss\�&=P��t       �	b� �Xc�A�)*

loss���<\��       �	�u!�Xc�A�)*

lossR��;��1       �	�"�Xc�A�)*

lossl<�j��       �	H�"�Xc�A�)*

lossS&n<E��       �	��#�Xc�A�)*

loss�`<���       �	l$�Xc�A�)*

loss�3X;�*�       �	�%�Xc�A�)*

loss�bW;o���       �	��%�Xc�A�)*

losse�a=1��J       �	�J&�Xc�A�)*

loss�"=x���       �	G�&�Xc�A�)*

loss��;['ű       �	��'�Xc�A�)*

loss�b0<E6��       �	�2(�Xc�A�)*

loss�$<�{*       �	D�(�Xc�A�)*

loss�<T���       �	|)�Xc�A�)*

lossXx;�Xnq       �	�*�Xc�A�)*

loss-�B<��4       �	��*�Xc�A�)*

loss�O�:
��N       �	:[+�Xc�A�)*

loss}�<]4�       �	 ,�Xc�A�)*

loss�6Y<��Q>       �	{�,�Xc�A�)*

loss��:�D�       �	�@-�Xc�A�)*

lossW�;jp�e       �	/�-�Xc�A�)*

loss_�w<|-��       �	�|.�Xc�A�)*

loss�A_:�ǡY       �	?/�Xc�A�)*

loss��<�7S       �	,�/�Xc�A�)*

loss�cR:�ٝ�       �	�R0�Xc�A�)*

loss���87E       �	��0�Xc�A�)*

loss��9k���       �	�1�Xc�A�)*

loss"�:0~�^       �	�&2�Xc�A�)*

loss_*Z<���       �	3�Xc�A�)*

lossMkO<�3�W       �	*�3�Xc�A�)*

loss���:�7��       �	JE4�Xc�A�)*

loss�3<d�z-       �	�4�Xc�A�)*

loss(=�|x�       �	Y�5�Xc�A�)*

loss�';񲁭       �	-"6�Xc�A�)*

loss���;��b       �	]�6�Xc�A�)*

loss�<J�s�       �	�f7�Xc�A�)*

loss{X=���=       �	�
8�Xc�A�)*

loss}!F<��?       �	Ƨ8�Xc�A�)*

loss)�;%V�1       �	^M9�Xc�A�)*

loss�y<n��       �	k�9�Xc�A�)*

loss��=����       �	��:�Xc�A�)*

loss,@<��       �	�*;�Xc�A�)*

loss���<n>��       �	�;�Xc�A�)*

loss��;�@N�       �	(`<�Xc�A�)*

lossL��< L�I       �	�	=�Xc�A�)*

loss_�g<S-�       �	�=�Xc�A�)*

loss��O<���       �	K>�Xc�A�)*

loss�[�<�XJ�       �	��>�Xc�A�)*

lossu�<��       �	��?�Xc�A�)*

loss�h�;���       �	n1@�Xc�A�)*

loss�=<S��
       �	��@�Xc�A�)*

loss�~<�)^       �	�yA�Xc�A�)*

loss�*�:�9�       �	B�Xc�A�)*

lossh�;ǖ��       �	��B�Xc�A�)*

loss��<���       �	.SC�Xc�A�)*

lossL�<թ��       �	A�C�Xc�A�)*

losscem:ƴ�m       �	;�D�Xc�A�)*

loss��E<[�4�       �	�2E�Xc�A�)*

loss��i;ƣ��       �	�E�Xc�A�)*

lossm"�<��O^       �	tyF�Xc�A�**

lossO�;��        �	DG�Xc�A�**

loss���;ո`I       �	ٴG�Xc�A�**

loss���<٫        �	TH�Xc�A�**

loss��;�b�I       �	�H�Xc�A�**

lossӂC<?���       �	��I�Xc�A�**

lossD�:<�I!       �	#J�Xc�A�**

loss��n;��       �	��J�Xc�A�**

loss�5�<�u       �	]mK�Xc�A�**

lossZZ�;�gi       �	RL�Xc�A�**

lossN #<%,��       �	:�L�Xc�A�**

loss��x<O��       �	�QM�Xc�A�**

loss��<�;�f       �	��M�Xc�A�**

lossf��;Þv_       �	��N�Xc�A�**

loss�~4;`s�4       �	1O�Xc�A�**

loss�ZB<�%N       �	�O�Xc�A�**

loss=os<��KZ       �	t`P�Xc�A�**

loss�u<O}ʓ       �	��P�Xc�A�**

loss�h<D�/�       �	t�Q�Xc�A�**

lossHT?<���P       �	�4R�Xc�A�**

lossӓ�=�N8�       �	A�R�Xc�A�**

loss(�:RW�I       �	mS�Xc�A�**

loss���<�(}�       �	T�Xc�A�**

loss�GU;s�/       �	��T�Xc�A�**

loss��1;oo�m       �	vPU�Xc�A�**

loss��;�)7�       �	Pr�Xc�A�**

loss̎�=V�       �	#�r�Xc�A�**

loss�xt=3       �	��s�Xc�A�**

loss��|;��3�       �	�/t�Xc�A�**

lossR��<F�       �	d�t�Xc�A�**

loss��\;��6       �	+ju�Xc�A�**

loss���<��<_       �	`v�Xc�A�**

loss4��<4f       �	ܡv�Xc�A�**

loss�T,=T�>P       �	�9w�Xc�A�**

lossid==s�\       �	j�w�Xc�A�**

lossh��;�T��       �	��x�Xc�A�**

loss�u�;=�FO       �	(Cy�Xc�A�**

loss��v<%*1       �	�az�Xc�A�**

loss��<�.O       �	M�z�Xc�A�**

loss��=B�4       �	?�{�Xc�A�**

loss���;�0k�       �	�(|�Xc�A�**

loss=N?:]ć       �	��|�Xc�A�**

loss��4<�       �	��}�Xc�A�**

loss��;�H�u       �	G�~�Xc�A�**

lossWGT<�B.�       �	�-�Xc�A�**

lossͼ�<�I��       �	v��Xc�A�**

loss���<����       �	5b��Xc�A�**

lossi�S;�t       �	�
��Xc�A�**

lossDπ=-̍       �	u���Xc�A�**

lossża<V!c       �	���Xc�A�**

loss���;R��       �	�8��Xc�A�**

loss��=�qI�       �	 Є�Xc�A�**

loss{W�<����       �	hu��Xc�A�**

loss.��<����       �	��Xc�A�**

loss��3; ���       �	,���Xc�A�**

loss8�<{_�       �	�R��Xc�A�**

loss�1�;T�m       �	��Xc�A�**

loss�aW<��       �	�׈�Xc�A�**

loss-Z�<�LA�       �	�w��Xc�A�**

loss@ۺ<�*�,       �	v��Xc�A�**

loss��=,A       �	į��Xc�A�**

lossF��:�${       �	�P��Xc�A�**

lossZ=�=�4�       �	���Xc�A�**

lossO�2=���       �	����Xc�A�**

loss�=K=�
x/       �	h$��Xc�A�**

loss�qo=���       �	���Xc�A�**

loss���;��       �	�~��Xc�A�**

lossr��;�+�4       �	���Xc�A�**

loss&^�<��_I       �	Ҧ��Xc�A�**

loss�$�<�R�       �	�x��Xc�A�**

lossv��<�B       �	��Xc�A�**

loss�,%=H� �       �	��Xc�A�**

loss��8;�;T       �	�0��Xc�A�**

loss���;�L|       �	�Ӓ�Xc�A�**

loss�p:r��       �	
h��Xc�A�**

loss�d;%�       �	����Xc�A�**

loss�i�<])�       �	Ɏ��Xc�A�**

lossu�;����       �	9(��Xc�A�**

loss��E=lzZ�       �	H���Xc�A�**

loss��;X&�       �	yW��Xc�A�**

loss���:����       �	���Xc�A�**

loss���:lh��       �	���Xc�A�**

lossV<;^�&}       �	��Xc�A�**

loss�k<~�F       �	ݵ��Xc�A�**

loss ^�<�^�       �	a��Xc�A�**

loss�g%<�       �	����Xc�A�**

loss��1;�B�       �	����Xc�A�**

loss_t<���v       �	�0��Xc�A�**

loss�X�<)�(       �	�қ�Xc�A�**

loss��;ž"�       �	g��Xc�A�**

loss�V�:r�M       �	���Xc�A�**

loss���;?��       �	����Xc�A�**

loss��.<v4k       �	=C��Xc�A�**

loss�%�<-        �	T��Xc�A�**

loss���;� V6       �	Z��Xc�A�**

loss�^$<H��)       �	���Xc�A�**

loss�O�<*Y"|       �	W���Xc�A�**

loss��<I5|       �	�H��Xc�A�**

loss��;���h       �	��Xc�A�**

loss�j=�N��       �	�J��Xc�A�**

lossI�%<�9�}       �	X���Xc�A�**

loss��i;7O�       �	��Xc�A�**

loss 9t<�S��       �	f���Xc�A�**

loss��;$n�       �	����Xc�A�**

loss*�<F�       �	nL��Xc�A�**

lossI�;�q�Z       �	}��Xc�A�**

loss�`�=��(�       �	%���Xc�A�**

loss�k<�>oj       �	{1��Xc�A�**

loss��;;D��       �	���Xc�A�**

lossZ�0<C�0�       �	���Xc�A�**

loss��,=r��x       �	�(��Xc�A�**

loss��;���J       �	����Xc�A�**

lossv�A<�9��       �	h��Xc�A�**

loss/\<+���       �	���Xc�A�**

losse�;�gJ       �	ܡ��Xc�A�**

loss�e�<�ݪ       �	�9��Xc�A�**

lossZb�<��!       �	pҮ�Xc�A�**

loss�R�:�^�       �	�k��Xc�A�**

lossB=Q���       �	���Xc�A�**

loss
x'=i�\x       �	����Xc�A�**

loss1�<7y�l       �	�0��Xc�A�**

loss�#=��X       �	̱�Xc�A�**

lossr�<l���       �	fj��Xc�A�**

lossC�<ծ�B       �	����Xc�A�**

lossId�;=`ڎ       �	R���Xc�A�**

loss�m�;m�-�       �	iW��Xc�A�**

loss��;�`�       �	&��Xc�A�**

loss�(N;�'X�       �	I���Xc�A�**

loss_�<GG�8       �	/2��Xc�A�+*

lossf(R<l M       �	l̶�Xc�A�+*

lossa��<?FU.       �	c��Xc�A�+*

lossܸ;E�m�       �	X���Xc�A�+*

lossT�:/@�       �	����Xc�A�+*

lossV�=����       �	�;��Xc�A�+*

loss2��;�8p�       �	3Q��Xc�A�+*

loss�C=&�       �	� ��Xc�A�+*

loss��<�L�       �	;Ļ�Xc�A�+*

loss�0s=����       �	���Xc�A�+*

loss��p;>"(�       �	T���Xc�A�+*

loss��d=�p�       �	.<��Xc�A�+*

lossMT�;OƷ!       �	xѾ�Xc�A�+*

loss���;��/       �	�p��Xc�A�+*

loss�=*=*)�&       �	O��Xc�A�+*

loss-�=��       �	����Xc�A�+*

loss��;�¹       �	Fy��Xc�A�+*

lossI�<����       �	��Xc�A�+*

loss|04=n<�g       �	]���Xc�A�+*

loss@�n<S٧�       �	0���Xc�A�+*

loss�L	<{�       �	iU��Xc�A�+*

loss�@J<���       �	N���Xc�A�+*

loss�(=�Cw       �	Q���Xc�A�+*

loss�<�n��       �	K��Xc�A�+*

loss�gV;!-��       �	���Xc�A�+*

loss���<��K�       �	�r��Xc�A�+*

lossmQ�;�9�       �	}��Xc�A�+*

loss�_=�7B       �	s���Xc�A�+*

loss{^=��P       �	2;��Xc�A�+*

loss�_<��֙       �	|���Xc�A�+*

lossEƑ<�܆�       �	Gq��Xc�A�+*

loss��=RN��       �	!��Xc�A�+*

lossVL�<a|H       �	���Xc�A�+*

loss�X=���&       �	�9��Xc�A�+*

loss��<��       �	����Xc�A�+*

loss~(�;
�pn       �	6���Xc�A�+*

loss��1;��)�       �	O@��Xc�A�+*

lossT�:��       �	����Xc�A�+*

lossO��<��D�       �	ǃ��Xc�A�+*

loss�Oh<�]f)       �	,��Xc�A�+*

loss(~}<t �       �	����Xc�A�+*

losskR=��ُ       �	
j��Xc�A�+*

loss6�;N��       �	H��Xc�A�+*

loss��(<'�       �	���Xc�A�+*

loss:�L<x��<       �	�R��Xc�A�+*

lossG;I-�	       �	%���Xc�A�+*

loss�� =`�       �	j���Xc�A�+*

loss̻�<&���       �	O"��Xc�A�+*

loss��=a��	       �	���Xc�A�+*

lossrV�=�؜)       �	nh��Xc�A�+*

lossD��<�'�F       �	�	��Xc�A�+*

lossJ�<P!>�       �	Ψ��Xc�A�+*

lossr��9�nt       �	~��Xc�A�+*

loss��={���       �	���Xc�A�+*

lossa�S=�       �	y���Xc�A�+*

lossT�R<�(r�       �	�G��Xc�A�+*

loss�:�<��h       �	7���Xc�A�+*

loss���<!.��       �	�z��Xc�A�+*

loss���;$�~       �	'��Xc�A�+*

loss}�a<�K`�       �	s���Xc�A�+*

loss��;C+�       �	�`��Xc�A�+*

loss��<)9��       �	����Xc�A�+*

loss�q<�i]       �	L���Xc�A�+*

lossWC<� �z       �	 ��Xc�A�+*

loss���<|w�       �	����Xc�A�+*

loss��<���       �	�X��Xc�A�+*

loss8=��       �	s��Xc�A�+*

loss�JS=�Qަ       �	)��Xc�A�+*

loss[�;X_&       �	&���Xc�A�+*

lossM��<W��       �	����Xc�A�+*

loss��<���       �	�f��Xc�A�+*

loss�J�;�~��       �	X��Xc�A�+*

lossEF�;�љ       �	����Xc�A�+*

loss!�G=Q͉�       �	�i��Xc�A�+*

loss�=g,��       �	���Xc�A�+*

loss"�=�*�       �	 ���Xc�A�+*

loss�l;��s       �	�Q��Xc�A�+*

lossυ:���D       �	s���Xc�A�+*

lossxt =��)�       �	}���Xc�A�+*

loss�ͺ;�P:       �	�6��Xc�A�+*

loss�/	;���       �	#��Xc�A�+*

lossn'�<��ܴ       �	����Xc�A�+*

loss�b�;[,D       �	n���Xc�A�+*

loss<�i<mnL�       �	B?��Xc�A�+*

loss[�=�R       �	����Xc�A�+*

loss��>=괽       �	"o��Xc�A�+*

loss��<�J�t       �	��Xc�A�+*

loss ��<��       �	���Xc�A�+*

loss�?j<\A֩       �	�7��Xc�A�+*

loss��;}$�h       �	\���Xc�A�+*

lossh�<��       �	�\��Xc�A�+*

lossʗ�;�@��       �	5��Xc�A�+*

loss16�<�R�       �	`���Xc�A�+*

loss[<7�/       �	�a��Xc�A�+*

loss�z"=��6!       �	����Xc�A�+*

loss��~<���/       �	���Xc�A�+*

lossFp<ʺ�|       �	TS��Xc�A�+*

loss��-=��XS       �	����Xc�A�+*

loss�iD<G���       �	���Xc�A�+*

loss=qK<�-b       �	�M��Xc�A�+*

loss��4=�o"       �	����Xc�A�+*

lossC�!<s$�n       �	;���Xc�A�+*

loss�|z<"5�       �	O ��Xc�A�+*

loss��>=G�       �	����Xc�A�+*

loss��=�i       �	���Xc�A�+*

loss�=�       �	�,��Xc�A�+*

lossOԅ=-��       �	���Xc�A�+*

loss���;��       �	���Xc�A�+*

lossS�;{       �	�V��Xc�A�+*

lossM	�<��       �	����Xc�A�+*

loss�<���       �	~���Xc�A�+*

losswb=y��       �	T7 �Xc�A�+*

loss;�k=�~�       �	D� �Xc�A�+*

loss�<�E�       �	~�Xc�A�+*

loss�'$=���P       �	� �Xc�A�+*

loss�UB=k�       �	���Xc�A�+*

loss7��<�H��       �	(��Xc�A�+*

lossc�\<^�       �	)�Xc�A�+*

loss2�;7�Ν       �	��Xc�A�+*

loss5(<���&       �	{h�Xc�A�+*

loss�j�<cF��       �	T �Xc�A�+*

loss�&2=y��       �	���Xc�A�+*

loss��$<P0H�       �	�I�Xc�A�+*

lossr�9<��       �	���Xc�A�+*

loss��<6!�g       �	�	�Xc�A�+*

lossl��;�ǉo       �	�9
�Xc�A�+*

loss#F	<ԥ$       �	��
�Xc�A�+*

loss!]�;];M       �	cz�Xc�A�+*

loss��<?P��       �	_$�Xc�A�,*

loss���;)kww       �	���Xc�A�,*

loss`I�;5���       �	
��Xc�A�,*

loss��= ���       �	�G�Xc�A�,*

lossr�r<Q+��       �	���Xc�A�,*

loss��=�K��       �	`��Xc�A�,*

loss� =�%*       �	K9�Xc�A�,*

loss���<���       �	���Xc�A�,*

loss�{�;��T�       �	��Xc�A�,*

loss��;3�J       �	 ��Xc�A�,*

loss_��<t�LQ       �	S�Xc�A�,*

lossfW�<I4ň       �	 ��Xc�A�,*

loss�Mb<��*1       �	��Xc�A�,*

loss.��<S��C       �	�5�Xc�A�,*

loss�<W_3(       �	���Xc�A�,*

loss�{�<RO�       �	�u�Xc�A�,*

loss�Q�;֗r}       �	~7�Xc�A�,*

loss��|<L1�        �	(��Xc�A�,*

loss��< ћ�       �	�p�Xc�A�,*

loss�<�K       �	=�Xc�A�,*

loss�8!=jv       �	)��Xc�A�,*

lossj�X<#JV�       �	EI�Xc�A�,*

loss[U�;9�W�       �	���Xc�A�,*

loss���<��       �	���Xc�A�,*

loss�؄;gW       �	���Xc�A�,*

lossMH<G�&�       �	eR�Xc�A�,*

loss,�
=��
�       �	k��Xc�A�,*

loss�O�;y��       �	w��Xc�A�,*

loss��<���       �	�>�Xc�A�,*

lossZ	�<���       �	 : �Xc�A�,*

lossY=G~�Y       �	@� �Xc�A�,*

loss<��<�f�       �	Ov!�Xc�A�,*

lossLaa=p3�       �	�"�Xc�A�,*

loss1�=O��       �	b�"�Xc�A�,*

loss��;wY��       �	�[#�Xc�A�,*

loss�1�<X��^       �	P�#�Xc�A�,*

lossDCc<�v;       �	��$�Xc�A�,*

loss���<X'�L       �	<�%�Xc�A�,*

loss�eH<z�Q}       �	�O&�Xc�A�,*

loss%7<EfZ       �	�'�Xc�A�,*

loss<�͝       �	40(�Xc�A�,*

loss��<� �       �	n�(�Xc�A�,*

loss�&=���       �	S�*�Xc�A�,*

lossȵj=�E��       �	#J+�Xc�A�,*

lossץ�<��ou       �	,,�Xc�A�,*

lossr�<�Pd       �	n�,�Xc�A�,*

loss���;� ��       �	&m.�Xc�A�,*

loss!�=DO<p       �	�/�Xc�A�,*

loss�?V=�H��       �	
�/�Xc�A�,*

loss��^;���       �	it0�Xc�A�,*

loss푦;~cqp       �	��1�Xc�A�,*

loss:%�;U5�       �	�+2�Xc�A�,*

loss���<VA�       �	�3�Xc�A�,*

loss���<�;�       �	�3�Xc�A�,*

loss-,�<��8�       �	�C4�Xc�A�,*

loss�G�<f��       �	&�4�Xc�A�,*

loss���<���L       �	�5�Xc�A�,*

loss��;&���       �	}$6�Xc�A�,*

lossd�~;�fh�       �	��6�Xc�A�,*

lossoo?<8��^       �	fi7�Xc�A�,*

loss2w�<
�$�       �	�8�Xc�A�,*

loss���:���k       �	V�8�Xc�A�,*

loss*\?;3#�       �	|~9�Xc�A�,*

loss2&�;C�w2       �	+3:�Xc�A�,*

lossTx�<`n��       �	J�:�Xc�A�,*

loss�p< s       �	�;�Xc�A�,*

lossf=� �q       �	}><�Xc�A�,*

loss��5=���u       �	3�<�Xc�A�,*

loss^=O.��       �	�Y>�Xc�A�,*

loss6C�< �\U       �	� ?�Xc�A�,*

loss��#=�s+4       �	�?�Xc�A�,*

lossQ<U��h       �	@�@�Xc�A�,*

loss�~�<lA�	       �	4-A�Xc�A�,*

loss҅�<b��       �	y�A�Xc�A�,*

loss�E=a%�       �	zmB�Xc�A�,*

lossE-s;䴖       �	=C�Xc�A�,*

loss=�=�t$       �	�@D�Xc�A�,*

loss�Ӯ<�.0�       �	��D�Xc�A�,*

loss��<m���       �	E�Xc�A�,*

loss�d<"�y;       �	� F�Xc�A�,*

loss��m<����       �	
G�Xc�A�,*

loss��)=%О       �	u�G�Xc�A�,*

loss�;<����       �	EGH�Xc�A�,*

losss<,
_       �	x�H�Xc�A�,*

lossl��<��?2       �	�&J�Xc�A�,*

loss��;�/�       �	Y�J�Xc�A�,*

lossdY�<(K��       �	|aK�Xc�A�,*

loss�hH<��       �	�L�Xc�A�,*

loss#��;a�       �	ЙL�Xc�A�,*

loss��<m$       �	0/M�Xc�A�,*

lossґ$=t��       �	��M�Xc�A�,*

loss�y�<g�!       �	��N�Xc�A�,*

lossd�<n�K�       �	%O�Xc�A�,*

lossnu�;t��       �	��O�Xc�A�,*

loss��y=H��       �	R_P�Xc�A�,*

loss�nx<l�m�       �	M�P�Xc�A�,*

loss��;��       �	��Q�Xc�A�,*

loss�P=��tE       �	�&R�Xc�A�,*

loss=aX=�\�       �	?�R�Xc�A�,*

lossi�<���       �	�kS�Xc�A�,*

loss:�<u��       �	�T�Xc�A�,*

loss���9���       �	��T�Xc�A�,*

loss�P	;r��*       �	!YU�Xc�A�,*

loss흹=�t�       �	��U�Xc�A�,*

loss��$<0u�       �	��V�Xc�A�,*

loss��<cl��       �	�5W�Xc�A�,*

loss��;��H�       �	��W�Xc�A�,*

loss�S�<�K�)       �	rX�Xc�A�,*

loss��z;έ�       �	Y�Xc�A�,*

loss�.7;B�n�       �	��Y�Xc�A�,*

loss��e<rAN�       �	R`Z�Xc�A�,*

loss�=R}F}       �	�[�Xc�A�,*

loss�=c=@�       �	��[�Xc�A�,*

loss*mp=��       �	�I\�Xc�A�,*

loss�n=F@!�       �	y�\�Xc�A�,*

lossx�<���       �	b�]�Xc�A�,*

loss�S�<�#�`       �	�#^�Xc�A�,*

loss{\=�z�       �	��^�Xc�A�,*

loss�<�Z�[       �	�__�Xc�A�,*

loss��V;I��       �	�_�Xc�A�,*

loss���:�e�       �	�`�Xc�A�,*

loss(<*<��       �	 za�Xc�A�,*

lossj=^=Qt       �	�b�Xc�A�,*

loss��<�.7       �	��b�Xc�A�,*

loss1�<,W#�       �	��c�Xc�A�,*

losse�;���       �	F_d�Xc�A�,*

loss��	<��       �	we�Xc�A�,*

loss.7< �x       �	�e�Xc�A�,*

loss�5=gd�       �	U�f�Xc�A�-*

loss}�<y2+�       �	�,g�Xc�A�-*

loss�*�<j�       �	��g�Xc�A�-*

lossb��<l���       �	Meh�Xc�A�-*

loss�0<=zB)       �	� i�Xc�A�-*

loss�A<6L@B       �	}�i�Xc�A�-*

lossօ�<ރ�       �	9Bj�Xc�A�-*

loss&J�<F��       �	��j�Xc�A�-*

loss�5�<�ܯ�       �	B|k�Xc�A�-*

loss�@�<?��       �	el�Xc�A�-*

loss�ٛ<|���       �	��l�Xc�A�-*

loss�.�=���V       �	`Xm�Xc�A�-*

loss�s: �KO       �	O�n�Xc�A�-*

lossO<���n       �	�)o�Xc�A�-*

loss��;���Y       �	��o�Xc�A�-*

lossl�<A<�       �	�fp�Xc�A�-*

lossCԈ<Ht��       �	Gq�Xc�A�-*

loss�(W<sC(�       �	;�q�Xc�A�-*

loss�'=hk�{       �	�?r�Xc�A�-*

loss�qF=v�R       �	��r�Xc�A�-*

loss4h�<p%P�       �	}s�Xc�A�-*

loss)��<0��H       �	�t�Xc�A�-*

loss�ҳ<�(�3       �	c�t�Xc�A�-*

loss|&�<}���       �	�ku�Xc�A�-*

loss�W;Hl�       �	_
v�Xc�A�-*

lossB'<��       �	5�v�Xc�A�-*

loss�[<{i^�       �	��w�Xc�A�-*

loss��^<?���       �	lvx�Xc�A�-*

lossc��<�+G�       �	�y�Xc�A�-*

loss�<��;5       �	9*z�Xc�A�-*

loss�O�;�Q��       �	��z�Xc�A�-*

loss^A�;�Rf       �	�y{�Xc�A�-*

loss?j�<�>K       �	�X|�Xc�A�-*

lossır<@�Ν       �		k}�Xc�A�-*

loss���<6-X�       �	~�Xc�A�-*

loss]�#=%.i       �	��~�Xc�A�-*

loss��w=oţ       �	���Xc�A�-*

loss�;O�<V       �	`��Xc�A�-*

lossD`<@Nfe       �	����Xc�A�-*

lossܘ<���       �	�~��Xc�A�-*

lossQ��;�2       �	�P��Xc�A�-*

loss�[�;��H�       �	�3��Xc�A�-*

loss�g�;I�W:       �	d΃�Xc�A�-*

loss۟�:
���       �	Cr��Xc�A�-*

lossh�j;	��N       �	S��Xc�A�-*

lossA&< ��       �	���Xc�A�-*

loss��;̹�       �	R���Xc�A�-*

loss��O<�'@�       �	`"��Xc�A�-*

loss�ҭ=�j<�       �	����Xc�A�-*

loss�"�<H�\       �	;T��Xc�A�-*

loss<�F= �*I       �	�)��Xc�A�-*

loss���=��       �	�;��Xc�A�-*

loss�+j=6��       �	Պ�Xc�A�-*

loss�!�<�v��       �	is��Xc�A�-*

loss8�)=�V�       �	���Xc�A�-*

lossv9�<rWt       �	����Xc�A�-*

loss�sO="��E       �	V��Xc�A�-*

loss�gu<�J�       �	k��Xc�A�-*

lossXW;^牝       �	q���Xc�A�-*

lossWL�<��F       �	M1��Xc�A�-*

loss�@=q��
       �	9Տ�Xc�A�-*

loss2��<x���       �	�r��Xc�A�-*

loss�\<}zm&       �	��Xc�A�-*

loss�bt:��v�       �	����Xc�A�-*

lossS%�=���       �	.T��Xc�A�-*

lossF+�<W�x�       �	���Xc�A�-*

loss�;�=B� �       �	L���Xc�A�-*

loss%��<"�o�       �	�&��Xc�A�-*

loss1�=A���       �	+Ô�Xc�A�-*

loss,�<� W�       �	�[��Xc�A�-*

lossz��<LA�       �	H���Xc�A�-*

loss3�=&��p       �	���Xc�A�-*

loss�Z�;��Cw       �	^.��Xc�A�-*

loss��=<Y���       �	�ŗ�Xc�A�-*

losslim;�¯�       �	Tq��Xc�A�-*

loss�J�<eT�&       �	x��Xc�A�-*

loss}bv<�i?�       �	�י�Xc�A�-*

loss?�;Ҧv�       �	nm��Xc�A�-*

loss�Z;cpo�       �	�$��Xc�A�-*

loss��'<�O5�       �	����Xc�A�-*

loss�ͥ<�]��       �	�V��Xc�A�-*

lossT�'<�O�O       �	���Xc�A�-*

loss�zE=�@1�       �	狝�Xc�A�-*

loss�=�#6C       �	c&��Xc�A�-*

loss{8�<\~�J       �	�Ϟ�Xc�A�-*

lossό;9'�T       �	�e��Xc�A�-*

lossd3<GN-�       �	�?��Xc�A�-*

loss��\=U��       �	
ڠ�Xc�A�-*

lossm�L<��|�       �	�q��Xc�A�-*

loss$�_<����       �	���Xc�A�-*

loss���<*{�k       �	����Xc�A�-*

loss!%�<�Lګ       �	N��Xc�A�-*

loss�/�<�>*�       �	���Xc�A�-*

lossg� ;�       �	�Ĥ�Xc�A�-*

loss<��<���       �	ge��Xc�A�-*

loss��<����       �	t��Xc�A�-*

loss�$=|�s       �	D3��Xc�A�-*

loss��<$��       �	ӧ�Xc�A�-*

loss:�`=�:�       �	��Xc�A�-*

loss�<z��.       �	b���Xc�A�-*

loss4"�<�U.�       �	BA��Xc�A�-*

loss/<�99       �	+٪�Xc�A�-*

loss��j;߬       �	S���Xc�A�-*

loss��<R���       �	�0��Xc�A�-*

loss�t=�X�?       �	�٬�Xc�A�-*

loss�ƛ:���       �	�x��Xc�A�-*

loss�B�<`�w       �	[%��Xc�A�-*

loss��=B�vq       �	�Ǯ�Xc�A�-*

loss��<kwj�       �	h��Xc�A�-*

loss	0�;g�^�       �	��Xc�A�-*

loss��3;�2��       �	��Xc�A�-*

loss���<W�ϼ       �	���Xc�A�-*

lossKF�<�c^       �	����Xc�A�-*

loss#
=���       �	�;��Xc�A�-*

loss��]=��]�       �	�߳�Xc�A�-*

lossN=�;���       �	����Xc�A�-*

lossn�M;`�'       �	�&��Xc�A�-*

losse~x<����       �	µ�Xc�A�-*

lossMFV<�3�.       �	a��Xc�A�-*

loss�<.�0       �	����Xc�A�-*

loss�=� �       �	����Xc�A�-*

loss�Ha=��       �	�Q��Xc�A�-*

lossD.2=]��I       �	���Xc�A�-*

loss�`�<�l�&       �	꒹�Xc�A�-*

loss-T�:�f-       �	0��Xc�A�-*

loss�=
=lY       �	�ֺ�Xc�A�-*

loss�h�;y��       �	�0��Xc�A�-*

lossÁ;����       �	��Xc�A�-*

loss͉=�1�A       �	1Ͻ�Xc�A�.*

loss7�C<��T       �	~��Xc�A�.*

loss�،<�q�\       �	���Xc�A�.*

lossUD=����       �	���Xc�A�.*

loss�w�;�ί       �	����Xc�A�.*

lossM�7:&��       �	����Xc�A�.*

loss���;ڍ�9       �	O��Xc�A�.*

loss��;8�W       �	����Xc�A�.*

loss\�H; �_       �	Υ��Xc�A�.*

loss�$q<8W       �	"R��Xc�A�.*

loss��&<�пb       �	����Xc�A�.*

loss���<����       �	����Xc�A�.*

loss(Y�;���       �	�@��Xc�A�.*

loss|ߜ<���       �	1���Xc�A�.*

loss���:eye�       �	���Xc�A�.*

loss�#Q=}�K       �	=��Xc�A�.*

loss��I;��N       �	����Xc�A�.*

loss�=���L       �	T���Xc�A�.*

loss��<x�y       �	`���Xc�A�.*

losseDE<�mo       �	w.��Xc�A�.*

loss�A!<(��/       �	y���Xc�A�.*

lossp^�<C���       �	uv��Xc�A�.*

lossF�=��g       �	b��Xc�A�.*

loss�_a:i�k�       �	ޭ��Xc�A�.*

lossT�~<�T"�       �	Eb��Xc�A�.*

loss\�a<;��       �	o��Xc�A�.*

loss8L�<b��       �	���Xc�A�.*

loss��<�;3       �	/n��Xc�A�.*

loss�<�<��       �	
��Xc�A�.*

lossM.�=)��       �	Z���Xc�A�.*

lossT��<�E�T       �	�i��Xc�A�.*

loss��<�j�       �	o���Xc�A�.*

lossTkZ<˥�        �	�+��Xc�A�.*

lossu�;�q�h       �	����Xc�A�.*

loss�G�;ڇ�h       �	4���Xc�A�.*

loss[��;��1;       �	-��Xc�A�.*

lossT@<���       �	����Xc�A�.*

loss"r�<���       �	�r��Xc�A�.*

loss&զ;��o[       �	���Xc�A�.*

loss<1�;�a�       �	����Xc�A�.*

loss��
=�o��       �	�R��Xc�A�.*

loss��H={�d       �	����Xc�A�.*

loss��j;+^1       �	f���Xc�A�.*

loss5�<���R       �	 B��Xc�A�.*

loss�G�<��ֿ       �	����Xc�A�.*

loss���:�/�[       �	����Xc�A�.*

lossn\9��v;       �	;��Xc�A�.*

loss�b ;τ*�       �	���Xc�A�.*

loss�_�;�s0       �	�{��Xc�A�.*

loss;
;����       �	��Xc�A�.*

loss�"�;.�d�       �	���Xc�A�.*

loss�';h�y       �	�]��Xc�A�.*

lossD�<�`�       �	���Xc�A�.*

loss&:#��       �	P���Xc�A�.*

loss ��9��,       �	,J��Xc�A�.*

lossf9Z5�       �	���Xc�A�.*

loss�S;/"ZV       �	j���Xc�A�.*

loss%�<�7       �	9���Xc�A�.*

loss:q�<8�       �	�G��Xc�A�.*

loss�ٿ9��Ql       �	z��Xc�A�.*

loss��:�Q�n       �	@���Xc�A�.*

loss{�=��)}       �	�b��Xc�A�.*

loss(�;#[K�       �	���Xc�A�.*

lossc�;��t       �	;���Xc�A�.*

loss�!}<�s?�       �	E��Xc�A�.*

loss��Z=v���       �	z���Xc�A�.*

loss�	<�@{^       �	����Xc�A�.*

loss&��<k3G�       �	�*��Xc�A�.*

loss:k-=̲�       �	����Xc�A�.*

loss���</+��       �	�u��Xc�A�.*

lossJ�=ixU       �	b��Xc�A�.*

lossd �<�e��       �	���Xc�A�.*

lossue;
��o       �	����Xc�A�.*

loss���</}�t       �	�x��Xc�A�.*

loss��<ނ}       �	���Xc�A�.*

loss��:�HA       �	3���Xc�A�.*

loss���=��^       �	�`��Xc�A�.*

lossM�<2�       �	��Xc�A�.*

lossi =٧��       �	ϣ��Xc�A�.*

loss��	;6V6�       �	2W��Xc�A�.*

loss_%9<�5[       �	�4��Xc�A�.*

lossw�%;bb8b       �	����Xc�A�.*

loss�<e?��       �	 t��Xc�A�.*

lossl<�:�~       �	���Xc�A�.*

lossX�[<~}q       �	����Xc�A�.*

loss�4�;%u�k       �	NA��Xc�A�.*

loss�3B;�m)|       �	����Xc�A�.*

loss?��;���       �	���Xc�A�.*

loss� ?=�\       �	�L��Xc�A�.*

loss��;jTx       �	<���Xc�A�.*

loss���;�e��       �	� ��Xc�A�.*

loss4)�<"�j       �	����Xc�A�.*

losss�;Q��       �	Y��Xc�A�.*

lossIU*<��4       �	z ��Xc�A�.*

lossߝ�:f1       �	���Xc�A�.*

loss�sa<��x�       �	m<��Xc�A�.*

loss���;n��       �	d���Xc�A�.*

loss�ni<���[       �	i���Xc�A�.*

loss��a<���b       �	�h �Xc�A�.*

loss��<!��)       �	���Xc�A�.*

loss�>t<F\��       �	��Xc�A�.*

loss�<�v	?       �	�+�Xc�A�.*

loss}�9;'�w�       �	��Xc�A�.*

loss�1=ֈ�       �	;9�Xc�A�.*

loss.r�;����       �	a��Xc�A�.*

loss|��<>���       �	n��Xc�A�.*

loss��<��       �	��Xc�A�.*

losss۔<9��       �	�o�Xc�A�.*

loss
=�K߇       �	2	�Xc�A�.*

loss�֞;��        �	��	�Xc�A�.*

loss��<���       �	�h
�Xc�A�.*

loss<)(=h��       �	 ��Xc�A�.*

loss��<;Z��0       �	�)�Xc�A�.*

loss. O<A>a�       �	X #�Xc�A�.*

loss��;��Zt       �	�#�Xc�A�.*

loss�q=*�?       �	BC$�Xc�A�.*

loss�;��;\       �	f�$�Xc�A�.*

loss[��<:�F#       �	�z%�Xc�A�.*

loss�U<�2�       �	�&�Xc�A�.*

loss��<u���       �	��&�Xc�A�.*

loss�1=�4       �	�k'�Xc�A�.*

loss��X=#/�8       �	�(�Xc�A�.*

loss4<"^;�       �	��(�Xc�A�.*

loss�A�;��       �	�;)�Xc�A�.*

loss�q�;KL       �	��)�Xc�A�.*

loss3e�<����       �	_~*�Xc�A�.*

loss�=iH�3       �	=)+�Xc�A�.*

loss��=� H�       �	��+�Xc�A�.*

loss�q<P�1       �	�f,�Xc�A�/*

lossW�:�Fgq       �	��,�Xc�A�/*

lossC@<�bt�       �	�-�Xc�A�/*

loss8M�;\{��       �	�~.�Xc�A�/*

loss�b�<� �       �	r�/�Xc�A�/*

loss�c(<�l       �	*�0�Xc�A�/*

loss:��<�^�
       �	p'1�Xc�A�/*

loss�v�:2�a�       �	+�1�Xc�A�/*

loss�PF=Q�V�       �	\2�Xc�A�/*

loss*��:N�
R       �	<�2�Xc�A�/*

loss�q�<���       �	�3�Xc�A�/*

loss���<�w�W       �	k*4�Xc�A�/*

losso��;��|3       �	��4�Xc�A�/*

loss\� =݋4d       �	�\5�Xc�A�/*

loss͙�<z��\       �	��5�Xc�A�/*

loss���;��
K       �	C�6�Xc�A�/*

loss�	 =ƒMp       �	�:7�Xc�A�/*

loss֋;� !(       �	��7�Xc�A�/*

loss�k=��a       �	s�8�Xc�A�/*

loss��<1��       �	;9�Xc�A�/*

loss�D�<-m�t       �	8:�Xc�A�/*

lossH;�e��       �	��:�Xc�A�/*

loss*�=�֙B       �	U;�Xc�A�/*

loss� =κZ       �	��;�Xc�A�/*

loss�K<���       �	R�<�Xc�A�/*

losss<̓�       �	�C=�Xc�A�/*

loss��'<S�j`       �	��=�Xc�A�/*

loss��b<-�6�       �	c~>�Xc�A�/*

lossl�B<Z�ڼ       �	g�?�Xc�A�/*

loss)�/<G]��       �	s�@�Xc�A�/*

lossVit=t9�       �	><A�Xc�A�/*

loss��N<��B       �	�A�Xc�A�/*

loss�0�<8{>�       �	#�B�Xc�A�/*

lossJ^�;�f�       �	^fC�Xc�A�/*

lossϬ;��z�       �	�D�Xc�A�/*

lossJ�T;r�`�       �	�D�Xc�A�/*

loss?R=��9       �	3TE�Xc�A�/*

loss�[�;���       �	��E�Xc�A�/*

loss.V=v"sz       �	h�F�Xc�A�/*

loss��=���e       �	�0G�Xc�A�/*

loss�z�;_�1       �	��G�Xc�A�/*

loss�(�:@���       �	�gH�Xc�A�/*

loss��:��z�       �	�I�Xc�A�/*

loss���;�Xd�       �	E�I�Xc�A�/*

loss�a�=}4��       �	O=J�Xc�A�/*

lossZ�E<�a0r       �	#�J�Xc�A�/*

loss/��<>Ø       �	sK�Xc�A�/*

loss8�d<��       �	L�Xc�A�/*

loss-J<!���       �	b�L�Xc�A�/*

lossO�<��m�       �	75M�Xc�A�/*

loss�t�:*x       �	��M�Xc�A�/*

loss��z={���       �	p`N�Xc�A�/*

loss�U�<�i�.       �	A�N�Xc�A�/*

loss��<N�)       �	�O�Xc�A�/*

loss��\=�#�       �	�0P�Xc�A�/*

loss��<�FЪ       �	C�P�Xc�A�/*

loss8�?<J��#       �	�_Q�Xc�A�/*

loss4_�<p�h�       �	Q�Q�Xc�A�/*

loss�<��`�       �	L�R�Xc�A�/*

loss�h�<�2�       �	�#S�Xc�A�/*

loss��I<W���       �	��S�Xc�A�/*

loss2{+;�!ƀ       �	�HT�Xc�A�/*

loss�"�<g`t�       �	'�T�Xc�A�/*

loss�Y;��F       �	*qU�Xc�A�/*

lossj.#=U-�       �	�V�Xc�A�/*

loss �Y<�0��       �	�V�Xc�A�/*

loss�8=���       �	�vW�Xc�A�/*

loss�<�<{�o�       �	X�Xc�A�/*

loss�o;[�v       �	��X�Xc�A�/*

loss:��<�@K#       �	�CY�Xc�A�/*

losstN�<wCI�       �	��Y�Xc�A�/*

loss3�<:	�       �	�kZ�Xc�A�/*

loss!�;��M(       �	m�Z�Xc�A�/*

loss���<�=-d       �	��[�Xc�A�/*

loss��!<s�N�       �	-\�Xc�A�/*

lossܦ�<��5�       �	m�\�Xc�A�/*

loss���<v���       �	�\]�Xc�A�/*

loss���:�U��       �	��]�Xc�A�/*

loss6��<ܠl       �	@�^�Xc�A�/*

loss��I<�>+$       �	'_�Xc�A�/*

loss���;$|�       �	��_�Xc�A�/*

lossX��<njP       �	�{`�Xc�A�/*

lossD��<g1�       �	�a�Xc�A�/*

loss��=���-       �	0�a�Xc�A�/*

loss_��<��4       �	�Qb�Xc�A�/*

loss��@<3gr       �	��b�Xc�A�/*

lossʍ�<$ �       �	 c�Xc�A�/*

loss̊�;�o�c       �	 d�Xc�A�/*

loss�fh;��,K       �	��d�Xc�A�/*

loss��;�9��       �	�e�Xc�A�/*

lossMKL;@�78       �	��f�Xc�A�/*

loss|��;t���       �	�Og�Xc�A�/*

loss��q<nC�7       �	T�g�Xc�A�/*

loss���<�r0~       �	�-i�Xc�A�/*

loss&�Y<//�]       �	 �i�Xc�A�/*

loss ��;(I[�       �	�\j�Xc�A�/*

loss��<1�       �	��j�Xc�A�/*

loss� <'�(       �	3�k�Xc�A�/*

loss���:);z�       �	�@l�Xc�A�/*

loss�I/<37�x       �	~�l�Xc�A�/*

loss�u�:�;lR       �	��m�Xc�A�/*

loss��<+�H�       �	�n�Xc�A�/*

losst8=Z�       �	/�n�Xc�A�/*

loss�8	=A7�       �	�Yo�Xc�A�/*

loss#h�:�[�v       �	��o�Xc�A�/*

loss�j�;���
       �	��p�Xc�A�/*

loss�Y&=�v       �	:q�Xc�A�/*

loss)L;�W��       �	q�q�Xc�A�/*

loss���<7)`       �	_{r�Xc�A�/*

lossԯ<;<�e       �	vs�Xc�A�/*

loss�%=�}f^       �	Q�s�Xc�A�/*

lossj�<��=       �	�pt�Xc�A�/*

lossCܑ<Y���       �	�u�Xc�A�/*

loss���<�[�       �	H�u�Xc�A�/*

lossW.<e=}�       �	Rv�Xc�A�/*

lossUw�;t���       �	
�v�Xc�A�/*

loss��<��       �	��w�Xc�A�/*

loss?�P<���       �	P6x�Xc�A�/*

lossL��;;�       �	��x�Xc�A�/*

loss�~{<�f�       �	��y�Xc�A�/*

lossQ��<!���       �	�.z�Xc�A�/*

loss��C<.��&       �	��z�Xc�A�/*

loss��%<j<">       �	ms{�Xc�A�/*

loss�g�<9|,       �	�|�Xc�A�/*

loss)��:ABKG       �	1�|�Xc�A�/*

loss�h�:�ǐ�       �	f}�Xc�A�/*

loss{�J<Lm?�       �	� ~�Xc�A�/*

loss��S<�M6       �	�~�Xc�A�/*

lossa�-;}[s4       �	�T�Xc�A�0*

loss=�)=M��x       �	^��Xc�A�0*

lossJNV;��!       �	���Xc�A�0*

loss�8h=�C�>       �	:��Xc�A�0*

loss���;�O)       �	^ہ�Xc�A�0*

loss]�w;�G�       �	���Xc�A�0*

lossf=V���       �	�*��Xc�A�0*

loss웒;DwJ�       �	�σ�Xc�A�0*

loss(��<�?=e       �	�r��Xc�A�0*

loss��<��z�       �	L��Xc�A�0*

loss��\=�5�       �	ĵ��Xc�A�0*

loss��<>?�       �	TU��Xc�A�0*

loss�H�9t~I       �	 ���Xc�A�0*

loss��;/\�%       �	#���Xc�A�0*

loss��<��*�       �	II��Xc�A�0*

loss��<u�       �	���Xc�A�0*

loss.W�:�
z�       �	ʋ��Xc�A�0*

loss�it<�λ�       �	,(��Xc�A�0*

loss|�="�P�       �	�Xc�A�0*

loss;��;���       �	�]��Xc�A�0*

losso; F       �	����Xc�A�0*

lossT��=�̖�       �	~���Xc�A�0*

loss�1<ޝ�       �	{-��Xc�A�0*

loss�o(;�N�f       �	5ҍ�Xc�A�0*

loss��h=1��       �	���Xc�A�0*

loss�I;x��N       �	bH��Xc�A�0*

lossh�<\�?       �	 ��Xc�A�0*

loss��<`�?�       �	~���Xc�A�0*

loss<�c;�+W�       �	�K��Xc�A�0*

lossĘ<�)n�       �	���Xc�A�0*

lossU<���4       �	D���Xc�A�0*

loss�e�:��Ŝ       �	\��Xc�A�0*

lossyG=J�(�       �	H���Xc�A�0*

loss->�<!>       �	^��Xc�A�0*

lossD�I<l��T       �	3���Xc�A�0*

loss?�E<-2x       �	}���Xc�A�0*

loss��?;�l�w       �	�.��Xc�A�0*

loss^�;<o��       �	*Ɩ�Xc�A�0*

loss�Bh=�q��       �	Ih��Xc�A�0*

loss�u�;�H��       �		���Xc�A�0*

loss>;dVh       �	{���Xc�A�0*

loss�; <�-"
       �	�?��Xc�A�0*

loss*��:r}%       �	wڙ�Xc�A�0*

loss�>�;!� �       �	:v��Xc�A�0*

loss܆�<V<I�       �	$(��Xc�A�0*

loss{I=�&��       �	��Xc�A�0*

loss<�><�.�       �	[}��Xc�A�0*

lossf�<ן��       �	2��Xc�A�0*

loss껽<�4�5       �	ٯ��Xc�A�0*

loss<�:�/(       �	�I��Xc�A�0*

loss��;����       �	���Xc�A�0*

loss�r�:�[�:       �	����Xc�A�0*

loss;��;X��E       �	C��Xc�A�0*

loss�/�<ou�       �	���Xc�A�0*

loss��(=��>	       �	r���Xc�A�0*

loss�$=I�       �	�:��Xc�A�0*

loss�zm<ȑ       �	����Xc�A�0*

loss��Y=AI'�       �	A���Xc�A�0*

loss"��<^'��       �	WC��Xc�A�0*

loss�.<L"�:       �	@1��Xc�A�0*

loss��&<[KJ       �	0���Xc�A�0*

loss���;�$�v       �	�s��Xc�A�0*

lossj�;P6֜       �	Q3��Xc�A�0*

loss/��;NT       �	�O��Xc�A�0*

lossY��=��d�       �	d��Xc�A�0*

loss�V=��и       �	l̪�Xc�A�0*

loss���<�bOH       �	����Xc�A�0*

loss3��;�+*�       �	צ��Xc�A�0*

loss�6<��+�       �	u[��Xc�A�0*

lossTv<4A�       �	�i��Xc�A�0*

loss6��;�e��       �	���Xc�A�0*

lossz��=��KG       �	����Xc�A�0*

loss�:=�ϥ�       �	z���Xc�A�0*

loss?��=�k�       �	��Xc�A�0*

lossLy=2"        �	�C��Xc�A�0*

loss��=����       �	����Xc�A�0*

losszY�<fo�       �	B?��Xc�A�0*

loss�<��       �	���Xc�A�0*

lossA��<1d�       �	D��Xc�A�0*

lossACq;}d9       �	U���Xc�A�0*

loss�/<��       �	�	��Xc�A�0*

loss��c<wr]�       �	毸�Xc�A�0*

loss�<�$>�       �	AG��Xc�A�0*

lossYu;�U1       �	���Xc�A�0*

loss��?<���       �	�x��Xc�A�0*

loss��<>։W       �	s��Xc�A�0*

loss��G=�X       �	����Xc�A�0*

loss_�<*UKv       �	�a��Xc�A�0*

loss�1D;J&�       �	C��Xc�A�0*

lossh��;X̮       �	��Xc�A�0*

loss
+H<�Y��       �	�þ�Xc�A�0*

loss���;j� I       �	8g��Xc�A�0*

lossO��=�5�[       �	���Xc�A�0*

loss8��<���       �	j���Xc�A�0*

loss�E=Z���       �	�D��Xc�A�0*

loss3)�<�vxF       �	m���Xc�A�0*

loss�L:���       �	����Xc�A�0*

loss��;��       �	=���Xc�A�0*

lossM�t=�%�       �	EI��Xc�A�0*

loss��"<fi�        �	]���Xc�A�0*

loss@<L��<       �	����Xc�A�0*

loss�=<wd�@       �	�A��Xc�A�0*

loss ,<	q�s       �	���Xc�A�0*

loss�9�;���       �	K���Xc�A�0*

loss;g�;�R<�       �	
L��Xc�A�0*

loss�=3=oP       �	ø��Xc�A�0*

loss���<3�lU       �	����Xc�A�0*

loss��,=(��'       �	�7��Xc�A�0*

loss!^�<����       �	���Xc�A�0*

lossӀ�<F?��       �	����Xc�A�0*

loss[�;�;#       �	�5��Xc�A�0*

losso4�:�C�       �	���Xc�A�0*

lossj;.&�       �	���Xc�A�0*

loss%�<o�.       �	C��Xc�A�0*

loss��Q<���O       �	����Xc�A�0*

loss�Z3<����       �	%v��Xc�A�0*

loss�|9<U��       �	���Xc�A�0*

loss���<Ko�d       �	3��Xc�A�0*

lossO.�;����       �	����Xc�A�0*

loss�@�<'��       �	*���Xc�A�0*

loss�o_<V(��       �	�{��Xc�A�0*

lossq�A=��       �	���Xc�A�0*

loss�;�2       �	���Xc�A�0*

loss�m\=@M:�       �	;���Xc�A�0*

loss��M<�m��       �	�.��Xc�A�0*

loss���<�8l�       �	z���Xc�A�0*

losspY<o�       �	�d��Xc�A�0*

loss���;�v
�       �	����Xc�A�0*

loss�%<5}f�       �	t���Xc�A�1*

lossF��;P-�       �	a��Xc�A�1*

loss�;=>���       �	I���Xc�A�1*

lossl�=����       �	����Xc�A�1*

loss��=;���       �	0,��Xc�A�1*

loss��=��&�       �	���Xc�A�1*

loss��;E���       �	�^��Xc�A�1*

loss�x�<����       �	r���Xc�A�1*

loss�N�<Y톴       �	u���Xc�A�1*

lossG[ ;�>l�       �	�e��Xc�A�1*

loss3�<����       �	{���Xc�A�1*

loss�4=tm1[       �	G���Xc�A�1*

loss�ަ<���       �	�&��Xc�A�1*

loss�3<w��       �	^���Xc�A�1*

loss�&�<�)�       �	�M��Xc�A�1*

loss�Y�<��\       �	����Xc�A�1*

loss���<���       �	!��Xc�A�1*

loss\ b;�kŴ       �	����Xc�A�1*

loss#}�;y�.�       �	�G��Xc�A�1*

loss�7�;�ǫ�       �	]N��Xc�A�1*

loss���<evG       �	v���Xc�A�1*

loss�<�M�?       �	И��Xc�A�1*

loss�|R<���r       �	[C��Xc�A�1*

loss���;Z� �       �	t`��Xc�A�1*

loss.�=ݠ�t       �	m��Xc�A�1*

loss<?b<N��        �	qU��Xc�A�1*

loss��5=?���       �	t���Xc�A�1*

loss�W=<�{�       �	���Xc�A�1*

lossQ�<�mz�       �	4L��Xc�A�1*

loss!��<1�e�       �	j���Xc�A�1*

lossI��<��        �	�!��Xc�A�1*

loss�a3<|���       �	����Xc�A�1*

loss�<5=y���       �	����Xc�A�1*

loss�s<��       �	����Xc�A�1*

loss�N=�!*�       �	ڑ��Xc�A�1*

loss�ǡ;��4,       �	�/��Xc�A�1*

loss �:<U-��       �	R���Xc�A�1*

lossZJ=k�j�       �	�o��Xc�A�1*

loss8%;�{�K       �	���Xc�A�1*

lossR��;\�+       �	����Xc�A�1*

loss�4�;��hA       �	����Xc�A�1*

loss� =���       �	�,��Xc�A�1*

loss��&;e�7W       �	����Xc�A�1*

loss�o<�n؁       �	�w��Xc�A�1*

loss�	=���I       �	���Xc�A�1*

loss�I�:�,�       �	,���Xc�A�1*

loss<p+<�UE/       �	>Z��Xc�A�1*

loss*��<�M�G       �	f���Xc�A�1*

loss���<�B��       �	���Xc�A�1*

loss�	�<L-       �	{-��Xc�A�1*

loss�-=v~��       �	���Xc�A�1*

loss��<��H}       �	c^��Xc�A�1*

loss,�;�O�D       �	(��Xc�A�1*

loss�?<�W�       �	%���Xc�A�1*

loss���<,H�2       �	�a��Xc�A�1*

loss���<i�~       �	2� �Xc�A�1*

loss�s�<�U�       �	lz�Xc�A�1*

lossq��<j�:�       �	��Xc�A�1*

loss�=� �       �	���Xc�A�1*

lossEym<7��+       �	���Xc�A�1*

loss6�N<N�^       �	\��Xc�A�1*

loss&uM;$��       �	�*�Xc�A�1*

loss��;95�       �	���Xc�A�1*

lossyx<��#       �	Qh�Xc�A�1*

loss��g=�'9O       �	� �Xc�A�1*

loss���;;u       �	���Xc�A�1*

lossF��;�g�       �	aq�Xc�A�1*

loss9X�<�j0       �	�		�Xc�A�1*

lossHyG;C        �	�	�Xc�A�1*

lossW�;6�       �	6
�Xc�A�1*

loss���;��oL       �	��
�Xc�A�1*

lossZ3�<,$��       �	~��Xc�A�1*

loss �<uܮv       �	g'�Xc�A�1*

loss��j=$ȭ}       �	,��Xc�A�1*

loss��= v�       �	��Xc�A�1*

loss�s6<�Zi�       �	\ �Xc�A�1*

loss	�<�r:       �	��Xc�A�1*

loss��*<w��       �	�b�Xc�A�1*

lossس;�Y�       �	��Xc�A�1*

loss?N;�V��       �	à�Xc�A�1*

loss4��< ��       �	Ec�Xc�A�1*

loss�a=�$ؤ       �	�
�Xc�A�1*

lossWuc<�H�~       �	���Xc�A�1*

loss��;/�       �	1@�Xc�A�1*

loss�y�;>{i�       �	���Xc�A�1*

loss��@;���       �	�x�Xc�A�1*

lossD��;����       �	�
�Xc�A�1*

loss!!>;B��~       �	���Xc�A�1*

loss�n�<6#{�       �	�=�Xc�A�1*

lossF�=C�       �	r��Xc�A�1*

loss$��;�3T       �	�u�Xc�A�1*

lossjn�<�6os       �		�Xc�A�1*

loss�K=��by       �	R��Xc�A�1*

loss��<L��       �	%Z�Xc�A�1*

lossC.�<Ġ6       �	�C�Xc�A�1*

loss��;kй]       �	/��Xc�A�1*

loss��I<ʾ��       �	ݙ�Xc�A�1*

loss��<��*,       �	�0�Xc�A�1*

losse�=��<�       �	���Xc�A�1*

loss�[�<����       �	d�Xc�A�1*

loss�'�<�9?�       �	��Xc�A�1*

loss�1o<��       �	��Xc�A�1*

lossH1)<��ؑ       �	=G�Xc�A�1*

loss�+�<�(�       �	���Xc�A�1*

lossN<���1       �	1z �Xc�A�1*

loss݀(=��6       �	�!�Xc�A�1*

loss,�<02       �	D�!�Xc�A�1*

lossx�<Ӆ`)       �	_}"�Xc�A�1*

lossz�<�;�~       �	�$#�Xc�A�1*

loss(�;�?��       �	��#�Xc�A�1*

loss�|�<�[�       �	Zg$�Xc�A�1*

loss/�H<�x��       �	[%�Xc�A�1*

loss��	<v2       �	��%�Xc�A�1*

loss �:N�V       �	��&�Xc�A�1*

loss�4{<!� m       �	�{'�Xc�A�1*

lossv{�<G���       �	P(�Xc�A�1*

loss1�=�'W[       �	��(�Xc�A�1*

loss �<��c�       �	�U)�Xc�A�1*

lossS<n��       �	��)�Xc�A�1*

lossv:.<Ӏ>       �	u�*�Xc�A�1*

loss��<�%�       �	�4+�Xc�A�1*

loss���;��       �	��+�Xc�A�1*

loss�-�<�N�       �	�,�Xc�A�1*

loss�)�<E�       �	B!-�Xc�A�1*

losss��<��M       �	�-�Xc�A�1*

loss�iV:r�$c       �	�T.�Xc�A�1*

loss�9G;���        �	��.�Xc�A�1*

loss-�O<���       �	Ϣ/�Xc�A�1*

loss�T;��k       �	�70�Xc�A�2*

lossN�:���
       �	��0�Xc�A�2*

loss�P<GL�O       �	Gq1�Xc�A�2*

lossϬ�<hg+�       �	W2�Xc�A�2*

loss,�Y;<6��       �	��2�Xc�A�2*

loss�0<�;b�       �	�63�Xc�A�2*

losshr�<��       �	��3�Xc�A�2*

loss�R =z�'       �	fh4�Xc�A�2*

loss?E@=5ߚe       �	�5�Xc�A�2*

loss��=����       �	L�5�Xc�A�2*

lossŧI;ڲ{9       �	��6�Xc�A�2*

loss��D=����       �	�I7�Xc�A�2*

loss�,6<��-b       �	8�7�Xc�A�2*

loss ��;$r�       �	U�8�Xc�A�2*

loss���=�yI       �	�I9�Xc�A�2*

loss��<�3��       �	4�9�Xc�A�2*

losss<7��       �	ʣ:�Xc�A�2*

lossb<����       �	9F;�Xc�A�2*

loss���;���       �	��;�Xc�A�2*

loss(݃<+bi       �	��<�Xc�A�2*

loss��z<H�       �	�D=�Xc�A�2*

loss�g�;�3d       �	8�=�Xc�A�2*

loss�o<"aG       �	�>�Xc�A�2*

loss�4;*�<       �	fN?�Xc�A�2*

loss��*=k�       �	��?�Xc�A�2*

loss���<:�ڻ       �	��@�Xc�A�2*

loss��F=�_       �	�pA�Xc�A�2*

loss�;z=46X	       �	XB�Xc�A�2*

lossf�m<���0       �	�C�Xc�A�2*

loss���<.�l_       �	K�C�Xc�A�2*

lossL��;C�       �	,bD�Xc�A�2*

loss��;|��       �	�E�Xc�A�2*

lossr;P;����       �	e�E�Xc�A�2*

loss��<_�}�       �	{F�Xc�A�2*

loss/�;eTC�       �	r5G�Xc�A�2*

loss�$�<��r�       �	��G�Xc�A�2*

loss��\<.n�       �	Y�H�Xc�A�2*

loss�@8;��cs       �	t_I�Xc�A�2*

lossH��:��CP       �	\J�Xc�A�2*

loss���;%V�r       �	[�J�Xc�A�2*

loss�R�;�;p�       �	�zK�Xc�A�2*

loss��<����       �	3L�Xc�A�2*

lossԧ�<�Z8�       �	��L�Xc�A�2*

loss�pT=����       �	3�M�Xc�A�2*

loss�l=�Bh�       �	T;N�Xc�A�2*

loss-$;�D       �	2�N�Xc�A�2*

loss��[<�`s
       �	Y�O�Xc�A�2*

lossU0=yY�       �	�1P�Xc�A�2*

loss2��<q��&       �	��P�Xc�A�2*

loss�=@��R       �	R}Q�Xc�A�2*

loss.�/<�ck�       �	CUR�Xc�A�2*

lossF�=<��H�       �	�S�Xc�A�2*

loss%"�;K��       �	��S�Xc�A�2*

loss�C;�i&�       �	�TT�Xc�A�2*

loss��<�/��       �	�!U�Xc�A�2*

loss=�v
�       �	A�U�Xc�A�2*

loss��X<Nݍ�       �	6�V�Xc�A�2*

lossc�6<���       �	�-W�Xc�A�2*

loss@R><y��U       �	��W�Xc�A�2*

loss�g�;��VL       �	�rX�Xc�A�2*

loss|E%<v�M�       �	Y�Xc�A�2*

loss��&;�zO!       �	��Y�Xc�A�2*

lossv�:;u��v       �	SzZ�Xc�A�2*

lossS��;��-P       �	�[�Xc�A�2*

loss?�=ň��       �	��[�Xc�A�2*

loss&��;���       �	�h\�Xc�A�2*

loss}�<��Y       �	]�Xc�A�2*

lossEd$<�&       �	��]�Xc�A�2*

loss!¾;L       �	��^�Xc�A�2*

loss��;�Ic        �	�3_�Xc�A�2*

loss�۫:y��       �	R�_�Xc�A�2*

lossʊ=�K��       �	vp`�Xc�A�2*

loss�B=Ց4�       �	a�Xc�A�2*

loss�e<�QIv       �	��a�Xc�A�2*

loss�U�=�z�       �	*Tb�Xc�A�2*

loss�R�;����       �	#�b�Xc�A�2*

lossY�:��ů       �	�c�Xc�A�2*

loss&��9}#       �	nNd�Xc�A�2*

losszT�;��       �	�d�Xc�A�2*

loss�"�<��(       �	e�Xc�A�2*

loss�+�==�r"       �	�f�Xc�A�2*

loss�h�<�i�       �	g�Xc�A�2*

loss�{�<!6�       �	�*h�Xc�A�2*

losssv�<}SvQ       �	i�Xc�A�2*

loss[yg=�k"       �	��i�Xc�A�2*

loss[a�<���       �	��j�Xc�A�2*

loss>ߎ;�C�       �	�wk�Xc�A�2*

loss/z/=<s�       �	D�l�Xc�A�2*

loss���<"u�       �	~m�Xc�A�2*

loss��T=h�       �	�*n�Xc�A�2*

loss�h�<�;��       �	Z�n�Xc�A�2*

loss��E=/p��       �	�vo�Xc�A�2*

loss
�4;_��w       �	Âp�Xc�A�2*

loss�=�;��l�       �	�%q�Xc�A�2*

lossT�;?���       �	��q�Xc�A�2*

loss�X<.���       �	�hr�Xc�A�2*

loss���=/�;�       �	Fs�Xc�A�2*

loss�::<M>�       �	;�s�Xc�A�2*

loss�z<?�j       �	?Tt�Xc�A�2*

lossљ�;���       �	� u�Xc�A�2*

lossm��<�
��       �	��u�Xc�A�2*

loss�D<���        �	Svv�Xc�A�2*

loss�f:)/$       �	�&w�Xc�A�2*

loss�
�;��f       �	��w�Xc�A�2*

losshč<)���       �	 qx�Xc�A�2*

loss��<DH�U       �	y�Xc�A�2*

lossW�T</�!�       �	��y�Xc�A�2*

loss;pO<&w��       �	!wz�Xc�A�2*

loss�fJ<�7�       �	1�{�Xc�A�2*

losschC<_@;�       �	NB|�Xc�A�2*

loss�
�<3��       �	u�|�Xc�A�2*

loss���<��n6       �	X~�Xc�A�2*

loss�+<�[9{       �	y�Xc�A�2*

lossk�<����       �	ܡ�Xc�A�2*

lossp�!=�d��       �	����Xc�A�2*

loss�<�Q{�       �	�V��Xc�A�2*

loss��.<Ǡ�C       �	����Xc�A�2*

losssZ�<��Kk       �	2��Xc�A�2*

lossa��<Il�       �	e���Xc�A�2*

loss��<ڗzB       �	eU��Xc�A�2*

lossd��<7�@       �	���Xc�A�2*

lossZ4T;�{f=       �	`ͅ�Xc�A�2*

loss�<�       �	i��Xc�A�2*

loss�9::�[       �	���Xc�A�2*

loss���<��?�       �	
���Xc�A�2*

loss�]�<M��{       �	�g��Xc�A�2*

lossπ<�Ʋ        �		��Xc�A�2*

lossv�;=t�u�       �	���Xc�A�2*

loss(<�M�       �	-y��Xc�A�3*

lossN>==W��       �	���Xc�A�3*

lossCo�;.vS       �	ʤ��Xc�A�3*

loss���:��p;       �	�>��Xc�A�3*

loss�ϰ<�@�5       �	�Ռ�Xc�A�3*

loss���:k��       �	����Xc�A�3*

lossJ��8%P�       �	�!��Xc�A�3*

loss�V;����       �	�ˎ�Xc�A�3*

loss�C�<amvy       �	�`��Xc�A�3*

loss�:(��~       �	��Xc�A�3*

lossӧ= �,       �	d���Xc�A�3*

loss�YW;J(L2       �	�S��Xc�A�3*

loss8�;���5       �	W��Xc�A�3*

loss�j:f+��       �	D���Xc�A�3*

loss[��7��g�       �	�!��Xc�A�3*

loss��l;/�Q�       �	����Xc�A�3*

loss�|i;��go       �		N��Xc�A�3*

loss&��<;r       �	Q���Xc�A�3*

loss�>�;G��9       �	����Xc�A�3*

loss��:0�$       �	�,��Xc�A�3*

loss`��<�+?+       �	nĖ�Xc�A�3*

lossfi^=o�-^       �	:]��Xc�A�3*

loss�;�d-g       �	���Xc�A�3*

loss��;���       �	����Xc�A�3*

loss#�W=a��       �	�]��Xc�A�3*

lossJ�<NID�       �	J��Xc�A�3*

loss�sa<�uI�       �	O���Xc�A�3*

loss�|<#��       �	D��Xc�A�3*

loss�h<~��       �	�ݛ�Xc�A�3*

lossUR�<��K       �	����Xc�A�3*

loss�Q�=5��       �	s-��Xc�A�3*

lossL`=17}2       �	�ɝ�Xc�A�3*

loss��R<׮��       �	�e��Xc�A�3*

loss�C=c��       �	���Xc�A�3*

loss�ۍ<��O       �	C���Xc�A�3*

loss�[�;�Q&�       �	�X��Xc�A�3*

loss/S�<�"?       �	M���Xc�A�3*

lossZ'H=ӟH_       �	���Xc�A�3*

loss@��;'A       �	|,��Xc�A�3*

loss�v;����       �	uˢ�Xc�A�3*

loss�=�g"�       �	w��Xc�A�3*

loss|�;,� f       �	9'��Xc�A�3*

loss��<�W��       �	�ʤ�Xc�A�3*

loss��$<�v       �	Z��Xc�A�3*

losso�=�0��       �	좦�Xc�A�3*

loss$� ;,D��       �	N��Xc�A�3*

loss6_�; �8�       �	�\��Xc�A�3*

lossp# ;O��h       �	n���Xc�A�3*

lossZ�s<\�G       �	����Xc�A�3*

loss�^�;PFt}       �	DR��Xc�A�3*

loss�F�<�f��       �	%��Xc�A�3*

lossO	�<
�J       �	n���Xc�A�3*

lossZ�z<���       �	B��Xc�A�3*

lossO��<�c��       �	}��Xc�A�3*

loss��;@��{       �	����Xc�A�3*

loss��<4H       �	�)��Xc�A�3*

lossm R<���       �	�Ү�Xc�A�3*

lossd�K;Ѽ6       �	:z��Xc�A�3*

loss<�e<��h^       �	�[��Xc�A�3*

loss��A<���       �	_
��Xc�A�3*

lossư�<U
%�       �	m���Xc�A�3*

loss��<��1�       �	�C��Xc�A�3*

loss)W�;(U�       �	���Xc�A�3*

lossF��;<I��       �	����Xc�A�3*

loss��<}Id       �	,*��Xc�A�3*

loss�l<a=�1       �	;ȴ�Xc�A�3*

loss��:8��        �	�k��Xc�A�3*

loss_i=���|       �	� ��Xc�A�3*

lossV?=%��       �	Ͻ��Xc�A�3*

loss/�:�_�       �	�Z��Xc�A�3*

loss��y<�*c       �	����Xc�A�3*

lossM m:
)�       �	.���Xc�A�3*

loss�4 <�L�       �	�$��Xc�A�3*

loss-�=��?�       �	x���Xc�A�3*

lossA�=���w       �	7��Xc�A�3*

loss.ȧ<V�g       �	O���Xc�A�3*

lossen�;Oa�       �	Nb��Xc�A�3*

loss�N2=Y=�       �	]���Xc�A�3*

loss��C<���       �	c���Xc�A�3*

loss��;�B�       �	�9��Xc�A�3*

loss���<Ą��       �	����Xc�A�3*

lossZT�<�$�       �	&���Xc�A�3*

loss�B�<�[=�       �	�K��Xc�A�3*

lossZȰ;��E       �	����Xc�A�3*

lossG]�;��A�       �	����Xc�A�3*

loss�;���       �	r���Xc�A�3*

lossH��<��|       �	�!��Xc�A�3*

loss:A<o�߀       �	^���Xc�A�3*

lossk��<�U�3       �	�P��Xc�A�3*

loss�:'�S�       �	<l��Xc�A�3*

lossJD�<ᦏ�       �	K��Xc�A�3*

lossV�<���       �	J���Xc�A�3*

loss�?�;dW��       �	�.��Xc�A�3*

lossH,<��       �	����Xc�A�3*

loss�Y�<�2�p       �	�[��Xc�A�3*

loss���;_��d       �	���Xc�A�3*

loss���<\�M       �	���Xc�A�3*

loss)7�:t�0       �	�*��Xc�A�3*

loss�?3<_'�       �	����Xc�A�3*

lossa��;M~�       �	v��Xc�A�3*

loss4��;x<#�       �	�!��Xc�A�3*

loss��9<���       �	���Xc�A�3*

lossTC;pD�       �	[]��Xc�A�3*

loss�;�v�l       �	���Xc�A�3*

loss�>;��       �	����Xc�A�3*

lossI1�<{�       �	�'��Xc�A�3*

loss��<���p       �	����Xc�A�3*

lossAӸ;�v�       �	}Y��Xc�A�3*

loss��<L��       �	����Xc�A�3*

loss$��:h~�       �	ע��Xc�A�3*

loss���<j^��       �	�>��Xc�A�3*

loss;;�<�9ھ       �	����Xc�A�3*

loss,�o<9��U       �	����Xc�A�3*

loss��X<ޭe       �	���Xc�A�3*

loss|b=S�K       �	���Xc�A�3*

loss_��=y��b       �	GY��Xc�A�3*

lossr@<�Qz       �	����Xc�A�3*

loss3$< �>a       �	���Xc�A�3*

lossj��<���       �	�@��Xc�A�3*

loss�a<͟Q�       �	���Xc�A�3*

losse*+<r       �	���Xc�A�3*

loss�0�;�T<�       �	Y1��Xc�A�3*

loss�%;bhU       �	1���Xc�A�3*

lossC�;��Gp       �	�{��Xc�A�3*

loss�%�<(�l       �	0��Xc�A�3*

lossa`�;tE�u       �	K���Xc�A�3*

loss�0�=v�u       �	1B��Xc�A�3*

loss�<l��       �	M���Xc�A�3*

lossn8;��5       �	2t��Xc�A�4*

loss��9��7A       �	���Xc�A�4*

loss�4�9���	       �	]���Xc�A�4*

loss�;��#c       �	�@��Xc�A�4*

loss���<��       �	���Xc�A�4*

loss�F8=���@       �	�o��Xc�A�4*

lossô9;�S��       �	���Xc�A�4*

loss�q8;�ZE       �	&���Xc�A�4*

losss��;����       �	�=��Xc�A�4*

lossZ��<Q�P       �	����Xc�A�4*

loss/�y:f���       �	s��Xc�A�4*

loss �*<f��       �	[	��Xc�A�4*

lossmnf;�L�)       �	ԛ��Xc�A�4*

loss,�<�Q�       �	�/��Xc�A�4*

loss��<I��*       �	X���Xc�A�4*

lossb߇;�ު       �	%]��Xc�A�4*

loss�C�;��       �	w���Xc�A�4*

loss�<;�V�       �	l���Xc�A�4*

loss!Z=���2       �	�.��Xc�A�4*

loss��<t�V�       �	����Xc�A�4*

loss�<��       �	�\ �Xc�A�4*

losss:8:m�Yp       �	5� �Xc�A�4*

loss#�4<�\�       �	U��Xc�A�4*

lossؙ�<���x       �	S �Xc�A�4*

loss�LY<}�`       �	o��Xc�A�4*

lossu�<���       �	KX�Xc�A�4*

loss�q=�g�       �	���Xc�A�4*

loss{-<�
g       �	���Xc�A�4*

loss�`;'Y��       �	�>�Xc�A�4*

lossVD=>�       �	��Xc�A�4*

loss�0<�J�:       �	���Xc�A�4*

loss���;G�X2       �	�,�Xc�A�4*

loss�R�<�$�W       �	��Xc�A�4*

lossv��<�ޭ�       �	.p�Xc�A�4*

loss���<&�:c       �	�	�Xc�A�4*

loss�r<��U�       �	��	�Xc�A�4*

loss�:�<[       �	�N
�Xc�A�4*

loss4�<��f       �	_�
�Xc�A�4*

loss�<T2u�       �	3��Xc�A�4*

loss��=��       �	�$�Xc�A�4*

loss��=�#�       �	���Xc�A�4*

loss`Q�<���I       �	�a�Xc�A�4*

lossʬt<g%
�       �	*��Xc�A�4*

loss�(�<��       �	V��Xc�A�4*

loss}R�<M�p�       �	
L�Xc�A�4*

loss;
<O��J       �	X��Xc�A�4*

loss�T};C��       �	;��Xc�A�4*

loss ��;ds��       �	O��Xc�A�4*

loss�;�0��       �	�4�Xc�A�4*

loss�*l=3�g�       �	��Xc�A�4*

lossf�<[�       �	�t�Xc�A�4*

loss�Y;��n       �	��Xc�A�4*

lossa��<rY�       �	���Xc�A�4*

lossld =�Jr4       �	sJ�Xc�A�4*

lossٯ;�k��       �	���Xc�A�4*

loss@�-<;И�       �	Ւ�Xc�A�4*

lossM�
=���       �	E,�Xc�A�4*

loss��K<_��       �	H��Xc�A�4*

loss��;�v2@       �	B^�Xc�A�4*

loss�TY<i^֬       �	���Xc�A�4*

lossz` ;�Gog       �	2��Xc�A�4*

loss�Ob;g�^       �	�F�Xc�A�4*

lossf+=�^�       �	���Xc�A�4*

loss=WV       �	�t�Xc�A�4*

loss=;:T�U       �	��Xc�A�4*

loss�/<�ݬ       �	ګ�Xc�A�4*

loss�$�<�(�:       �	�A�Xc�A�4*

loss�H&<���3       �	��Xc�A�4*

loss���<����       �	���Xc�A�4*

loss�.=���       �	R(�Xc�A�4*

loss�=�b?=       �	f��Xc�A�4*

loss7�<뷯       �	�Y �Xc�A�4*

loss?�;]#�f       �	�� �Xc�A�4*

loss���<'�       �	{�!�Xc�A�4*

loss�v�;�#��       �	� "�Xc�A�4*

lossxi�;<���       �	ȷ"�Xc�A�4*

loss:h<�)       �	�O#�Xc�A�4*

loss��z<Vw       �	-�#�Xc�A�4*

lossԀ];�       �	�$�Xc�A�4*

loss���<�\c       �	�%�Xc�A�4*

loss��<ȡM       �	5�%�Xc�A�4*

losswBk<qXQ�       �	mX&�Xc�A�4*

loss��;�?��       �	��&�Xc�A�4*

loss���:�[v�       �	6�'�Xc�A�4*

loss�S;ەQ       �	�c(�Xc�A�4*

loss�>�<���p       �	Н)�Xc�A�4*

loss=R-<Q��       �	�9*�Xc�A�4*

loss���<WwXO       �	?�*�Xc�A�4*

loss�?=�J�       �	2�+�Xc�A�4*

loss8�;�0�       �	�j,�Xc�A�4*

loss�� ; 9Nv       �	�!-�Xc�A�4*

loss�p�<��       �	G�-�Xc�A�4*

loss6��<�R�V       �	V�.�Xc�A�4*

loss؊�;)6�       �	�B/�Xc�A�4*

lossʫ�<��8       �	��/�Xc�A�4*

loss-�p<���       �	2�0�Xc�A�4*

loss�t�<���       �	�}1�Xc�A�4*

lossT�<���       �	$2�Xc�A�4*

loss�E=)Tx�       �	g�2�Xc�A�4*

losss�
=jF       �	si3�Xc�A�4*

loss,�;2�v-       �	0�4�Xc�A�4*

loss��~;�@�m       �	�E5�Xc�A�4*

loss�� =�e�>       �	`[6�Xc�A�4*

lossZNF<Ė�F       �	� 7�Xc�A�4*

loss��8<��a5       �	��7�Xc�A�4*

loss��<:^*�       �	\�8�Xc�A�4*

lossi��;�;��       �	�69�Xc�A�4*

loss?/<o
*       �	�9�Xc�A�4*

loss�$<�B4       �	5;�Xc�A�4*

lossC�+=�KD�       �	r�;�Xc�A�4*

lossR=<<��S       �	��<�Xc�A�4*

loss��:����       �	�=�Xc�A�4*

lossF��<̿��       �	�U>�Xc�A�4*

lossq�9;,J�'       �	��>�Xc�A�4*

loss/G�<��       �	��?�Xc�A�4*

loss�Be<pN�8       �	��@�Xc�A�4*

loss)u*:w�       �	�sA�Xc�A�4*

loss|�%=����       �	s*B�Xc�A�4*

lossޤ<�}�W       �	>vC�Xc�A�4*

lossZ�;��       �	I,D�Xc�A�4*

loss/L<t=�	       �	(�D�Xc�A�4*

loss���<"YKQ       �	�lE�Xc�A�4*

loss��
=�R�3       �	-�F�Xc�A�4*

loss��<s�q       �	�XG�Xc�A�4*

loss��;�Tb       �	��G�Xc�A�4*

lossZm:��[u       �	e�H�Xc�A�4*

lossp=�#�.       �	�,I�Xc�A�4*

lossx� ;�|��       �	��I�Xc�A�4*

loss��y;�~       �	�iJ�Xc�A�5*

loss�)�;-�<       �	wK�Xc�A�5*

loss �Z<)%I       �	7�K�Xc�A�5*

loss=y<bZe.       �	@L�Xc�A�5*

loss@=�<���       �	��L�Xc�A�5*

loss�v�<��u`       �	��M�Xc�A�5*

lossj��<g=��       �	:�N�Xc�A�5*

loss�(J=6��       �	�1O�Xc�A�5*

loss��;o]�=       �	7�O�Xc�A�5*

loss�4�;V�cQ       �	VP�Xc�A�5*

loss*�
<%~t       �	��P�Xc�A�5*

loss��@:�҅�       �	��Q�Xc�A�5*

lossAP!<��b�       �	�)R�Xc�A�5*

loss�%=} �       �	�R�Xc�A�5*

loss���<�)�j       �	&SS�Xc�A�5*

loss�=7
6       �	X�S�Xc�A�5*

lossݦ;#*ř       �	�xT�Xc�A�5*

loss|O	=��       �	IU�Xc�A�5*

loss�;�<       �	n�U�Xc�A�5*

loss���<����       �	W=V�Xc�A�5*

lossH��=uZ�;       �	��V�Xc�A�5*

loss�$[;�G�       �	�pW�Xc�A�5*

loss>�;%��       �	�X�Xc�A�5*

losss��;��y�       �	��X�Xc�A�5*

loss�� <�:�s       �	7Y�Xc�A�5*

loss�G(=2Fٱ       �	:�Y�Xc�A�5*

loss1�<,I�!       �	`Z�Xc�A�5*

lossx��;��q       �	��Z�Xc�A�5*

lossD� <��7N       �	�[�Xc�A�5*

loss�ju;<�V�       �	E-\�Xc�A�5*

lossی2<�@�       �	 �\�Xc�A�5*

loss)Z�<����       �	��]�Xc�A�5*

loss���<s�       �	,�^�Xc�A�5*

loss��y=>F�       �	�_�Xc�A�5*

loss麕<�&��       �	�_�Xc�A�5*

loss�<$�Ҩ       �	�U`�Xc�A�5*

loss�J6<�w3       �	�a�Xc�A�5*

loss��@=�F�9       �	"�a�Xc�A�5*

loss���:Q���       �	�Ib�Xc�A�5*

loss�ģ;�U/!       �	c�Xc�A�5*

loss<�(<�K�       �	@d�Xc�A�5*

lossH}.<���       �	�d�Xc�A�5*

loss;U�<�/Jq       �	O?e�Xc�A�5*

loss��b<M��R       �	Z�e�Xc�A�5*

loss��<
��       �	�yf�Xc�A�5*

loss�/;�\*       �	�g�Xc�A�5*

loss��<4�z       �	
h�Xc�A�5*

lossi��:Ƶ��       �	��h�Xc�A�5*

lossCY�:�Jvt       �	��i�Xc�A�5*

loss���;ΐӜ       �	@�j�Xc�A�5*

lossԹv<Ip�H       �	VHk�Xc�A�5*

lossî3<�}       �	�l�Xc�A�5*

loss��><pjV       �	�l�Xc�A�5*

loss��;�^       �	odm�Xc�A�5*

loss�=����       �	�rn�Xc�A�5*

loss��5=�Ԙ�       �	�o�Xc�A�5*

loss4��:2K��       �	oIp�Xc�A�5*

loss��%<�!I�       �	q�Xc�A�5*

loss�F=6��_       �	<�q�Xc�A�5*

loss�`^<����       �	ns�Xc�A�5*

loss��<}=J�       �	��s�Xc�A�5*

loss1��<���&       �	�Ft�Xc�A�5*

loss_�6=���       �	�'u�Xc�A�5*

lossM��;�N��       �	��u�Xc�A�5*

loss���<�ۭ+       �	�`v�Xc�A�5*

loss���<ُ�d       �	m�v�Xc�A�5*

loss��<��bu       �	��w�Xc�A�5*

lossx�<4�\U       �	bHx�Xc�A�5*

loss!�<���       �	�x�Xc�A�5*

loss�T<.�       �	˄y�Xc�A�5*

loss��@<��3       �	��z�Xc�A�5*

loss��8��-�       �	Ug{�Xc�A�5*

loss��:۲x\       �	{|�Xc�A�5*

loss�N�<1X��       �	��|�Xc�A�5*

losse��<s�c�       �	�]~�Xc�A�5*

loss�U<Fv0�       �	��~�Xc�A�5*

loss�Х<�}	8       �	ލ�Xc�A�5*

loss�=�DW       �	_$��Xc�A�5*

loss݀�;��%       �	���Xc�A�5*

loss��S=|
��       �	LS��Xc�A�5*

loss=n�;�!	�       �	���Xc�A�5*

loss��=[���       �	����Xc�A�5*

lossv�:��cg       �	q8��Xc�A�5*

loss);�<n'i       �	�Ƀ�Xc�A�5*

loss��<v�o�       �	>^��Xc�A�5*

lossWP�<>�x       �	M��Xc�A�5*

loss��O=tH��       �	U���Xc�A�5*

loss�V<RZV       �	GT��Xc�A�5*

lossn��;R�       �	��Xc�A�5*

loss�E<���s       �	����Xc�A�5*

lossrP�<���       �	�0��Xc�A�5*

loss]M�<f�jV       �	(Ո�Xc�A�5*

lossT��<l��       �	7o��Xc�A�5*

lossm	�<ϊ��       �	��Xc�A�5*

loss�H�;$(�       �	a���Xc�A�5*

loss��<Z<}       �	%x��Xc�A�5*

loss=��<�	�       �	{��Xc�A�5*

loss�QZ;�\�O       �	Ͱ��Xc�A�5*

loss�r;Y�m�       �	�B��Xc�A�5*

loss��!;��n�       �	0؍�Xc�A�5*

loss-��;��       �	@k��Xc�A�5*

lossQ�G<��E]       �	F��Xc�A�5*

lossM��<^K��       �	#���Xc�A�5*

loss���<LL�       �	�6��Xc�A�5*

loss�=S�Eb       �	����Xc�A�5*

loss/�;s2�p       �	8���Xc�A�5*

loss�F�:h�wo       �	�3��Xc�A�5*

lossUf:����       �	�̒�Xc�A�5*

lossTV~<�
       �	_��Xc�A�5*

loss�\�:�|)�       �	���Xc�A�5*

losst��:.��       �	����Xc�A�5*

lossxE<I�B�       �	}%��Xc�A�5*

loss}=�9<       �	����Xc�A�5*

lossI��<d�<       �	�\��Xc�A�5*

loss�~=���       �	����Xc�A�5*

loss�2<��4�       �	���Xc�A�5*

lossK�=X�l       �	�$��Xc�A�5*

loss��<_\My       �	���Xc�A�5*

loss	d�;�2�D       �	�Z��Xc�A�5*

loss��;��       �	g��Xc�A�5*

loss��=s|�;       �	����Xc�A�5*

loss�z�;�qS�       �	�N��Xc�A�5*

loss�'=;�>       �	��Xc�A�5*

loss
+�:���Y       �	S���Xc�A�5*

loss&f�<!��t       �	�G��Xc�A�5*

lossԬ�<_���       �	���Xc�A�5*

loss1��;���       �	����Xc�A�5*

loss{��</��       �	���Xc�A�5*

loss���;����       �	���Xc�A�6*

lossr��<M���       �	�B��Xc�A�6*

loss��%;7       �	֠�Xc�A�6*

loss�,�:��)�       �	�l��Xc�A�6*

lossN�<���       �	���Xc�A�6*

loss�|N:u��       �	����Xc�A�6*

loss.j <i$�o       �	�'��Xc�A�6*

lossPA<"8{       �	�ԣ�Xc�A�6*

loss�9�;
$d       �	�r��Xc�A�6*

loss�%=e���       �	���Xc�A�6*

loss��A=�Y˷       �	U���Xc�A�6*

loss�y<@��n       �	�=��Xc�A�6*

loss��!<)�RQ       �	�ߦ�Xc�A�6*

loss�C�<�	��       �	�|��Xc�A�6*

loss�[t<1�       �	�,��Xc�A�6*

lossJ(K<('f�       �	!��Xc�A�6*

lossk�<�\�K       �	Pũ�Xc�A�6*

loss��<|��       �	GZ��Xc�A�6*

lossRF�=�ݿ�       �	���Xc�A�6*

loss���;S�A       �	,���Xc�A�6*

loss�_<3���       �	Y0��Xc�A�6*

lossߢ�:'\KY       �	�Ϭ�Xc�A�6*

loss�8�<<c�       �	�n��Xc�A�6*

loss�4=Y��3       �	���Xc�A�6*

loss3h<*;�B       �	����Xc�A�6*

loss���;�ԃ�       �	�X��Xc�A�6*

loss�kW<�&�	       �	����Xc�A�6*

lossM0�<�L1       �	����Xc�A�6*

loss%s�:����       �	#0��Xc�A�6*

loss�a�:p�i       �	H���Xc�A�6*

loss�V]<��l       �	y���Xc�A�6*

loss��=dA       �	�=��Xc�A�6*

loss��_<����       �	���Xc�A�6*

loss��=��i       �	����Xc�A�6*

loss�8�<9��       �	4+��Xc�A�6*

lossY<2�18       �	�Ե�Xc�A�6*

loss:�.<�%�       �	�Ӷ�Xc�A�6*

loss^F�<Y�a�       �	EԷ�Xc�A�6*

lossFW<�ֆ�       �	Dk��Xc�A�6*

loss�ɪ:!�1       �	���Xc�A�6*

lossM��:��r�       �	cd��Xc�A�6*

loss)}�<=5l       �	`��Xc�A�6*

loss�)�<կ��       �	����Xc�A�6*

loss�2=�O�       �	͓��Xc�A�6*

loss.s�;���       �	^,��Xc�A�6*

loss�<;����       �	�<��Xc�A�6*

loss���;EC-       �	��Xc�A�6*

lossI�;J_�F       �	���Xc�A�6*

lossnm9<�52       �	6V��Xc�A�6*

loss���==7       �	-���Xc�A�6*

loss�$;в�^       �	w���Xc�A�6*

lossî@<[9��       �	X��Xc�A�6*

lossԺ�=���       �	)���Xc�A�6*

lossc�<h���       �	����Xc�A�6*

loss[ѣ=X       �	�I��Xc�A�6*

lossH��<G�x       �	P���Xc�A�6*

loss!$<�Ni       �	���Xc�A�6*

loss���<Ҽ9       �	��Xc�A�6*

lossd��<�       �	>���Xc�A�6*

loss�$�;���       �	S��Xc�A�6*

loss��<l�RE       �	����Xc�A�6*

loss�[~<o��K       �	�~��Xc�A�6*

loss�SK;��D,       �	�,��Xc�A�6*

loss�'�<�{M�       �	���Xc�A�6*

loss�\<�y�       �	��Xc�A�6*

loss:?l=f�3g       �	 %��Xc�A�6*

loss/{V=֞_1       �	����Xc�A�6*

loss%t�<if�R       �	f���Xc�A�6*

lossjT�<�uy       �	IH��Xc�A�6*

loss��w;"�       �	Q���Xc�A�6*

lossv�7<�kk�       �	c���Xc�A�6*

lossi�<ϯ8�       �	}<��Xc�A�6*

loss#�f;�[�       �	���Xc�A�6*

loss��;X�D�       �	����Xc�A�6*

lossJ.W<����       �	$a��Xc�A�6*

lossW;�<V,gR       �	,��Xc�A�6*

loss�=F<4��       �	/���Xc�A�6*

loss�/;�3�R       �	�a��Xc�A�6*

lossl��;�i�i       �	:��Xc�A�6*

loss���:��<�       �	����Xc�A�6*

loss18<#\`�       �	|F��Xc�A�6*

loss%m'<7vFJ       �	����Xc�A�6*

lossl(?=�O��       �	~���Xc�A�6*

lossAA�;5�t       �	�5��Xc�A�6*

lossw:�<��~�       �	����Xc�A�6*

lossA�:��d�       �	5z��Xc�A�6*

lossߑ;[hi�       �	�&��Xc�A�6*

loss��]<��#       �	:���Xc�A�6*

loss���<��P{       �	�{��Xc�A�6*

lossZ�;� �+       �	��Xc�A�6*

loss�~>;�� �       �	ٴ��Xc�A�6*

loss	�<+��       �	�S��Xc�A�6*

lossL�]<,=ga       �	����Xc�A�6*

loss�Y<�j8�       �	����Xc�A�6*

loss��;~GP%       �	,��Xc�A�6*

loss~a</�I       �	�6��Xc�A�6*

loss��<�5�       �	
���Xc�A�6*

lossۯ<H��/       �	����Xc�A�6*

lossO�<�8R=       �	4J��Xc�A�6*

lossS��=���$       �	?���Xc�A�6*

lossZi2<�cW�       �	T���Xc�A�6*

lossy@<��       �	A��Xc�A�6*

loss��j=�w߅       �	���Xc�A�6*

loss^�<Tݢ0       �	����Xc�A�6*

loss�=aM�       �	d��Xc�A�6*

lossd�b<��;       �	u��Xc�A�6*

loss��];7!�Y       �	���Xc�A�6*

loss�C�<g� �       �	�S��Xc�A�6*

loss2N�<TB1       �	R���Xc�A�6*

losse��;Ay!�       �	����Xc�A�6*

loss���<&t��       �	/��Xc�A�6*

loss��9�3S�       �	k���Xc�A�6*

lossQ?�=æ�       �	�t��Xc�A�6*

loss@?�<c���       �	���Xc�A�6*

loss)xn<���       �	ܺ��Xc�A�6*

loss�7=.��       �	�^��Xc�A�6*

lossڜ�;,d*       �	���Xc�A�6*

lossP7�<h:Ƅ       �	÷��Xc�A�6*

loss�E<g=�M       �	�]��Xc�A�6*

loss*7x<{��       �	�
��Xc�A�6*

loss|��;�]�l       �	Զ��Xc�A�6*

loss�P�:ǅ3       �	�a��Xc�A�6*

loss��b<k:�       �	A��Xc�A�6*

loss���<V�U       �	����Xc�A�6*

loss���<J'i�       �	�h��Xc�A�6*

loss2�/<Ts"       �	��Xc�A�6*

loss^f <g���       �	����Xc�A�6*

loss�X*<�#�       �		���Xc�A�6*

loss�>e<�/��       �	8��Xc�A�7*

loss��<+�g�       �	����Xc�A�7*

loss@��<��P�       �	[���Xc�A�7*

loss��q</��q       �	wK��Xc�A�7*

loss1<�ra�       �	���Xc�A�7*

loss8Zm<rI�        �	Ȕ��Xc�A�7*

loss�L<;��       �	K>��Xc�A�7*

lossZ�<��F�       �	h���Xc�A�7*

loss̼
<�K��       �	����Xc�A�7*

loss.��<��y�       �	�G��Xc�A�7*

loss*��<��ݘ       �	n���Xc�A�7*

loss�'�<a��       �	����Xc�A�7*

loss�/�;�+-+       �	�U��Xc�A�7*

loss��<˔�       �	����Xc�A�7*

loss��;���       �	"���Xc�A�7*

lossW[�<2�@�       �	�I��Xc�A�7*

loss�i�<���       �	" �Xc�A�7*

lossf"�<8C�       �	|� �Xc�A�7*

lossr��;�GX       �	7o�Xc�A�7*

loss)w�;|��-       �	��Xc�A�7*

loss[N�<���       �	���Xc�A�7*

lossR';�*U\       �	qY�Xc�A�7*

loss�r9;�`       �	��Xc�A�7*

loss���;z���       �	V��Xc�A�7*

loss��<X�[�       �	��Xc�A�7*

loss�%;;S�1       �	l"�Xc�A�7*

loss^*=��E       �	9��Xc�A�7*

lossW�S<_m�       �	�j�Xc�A�7*

loss��s<$Z       �	�Xc�A�7*

lossn[<�	��       �	F��Xc�A�7*

loss�#�;J�N�       �	�i	�Xc�A�7*

loss�֝<U�e+       �	�	
�Xc�A�7*

loss�v<c��       �	@�
�Xc�A�7*

loss\�i;=�5       �	�9�Xc�A�7*

loss7��<&��       �	���Xc�A�7*

loss���<�`š       �	+l�Xc�A�7*

loss�մ;�x�-       �	F�Xc�A�7*

loss���9�3s       �	?��Xc�A�7*

loss��;��       �	�y�Xc�A�7*

loss�@<uz�       �	�Xc�A�7*

loss��<����       �	���Xc�A�7*

lossE��<ˏ"S       �	)A�Xc�A�7*

loss��<�A�       �	��Xc�A�7*

loss�s%=��Ϋ       �	��Xc�A�7*

loss��u:�H`       �	$�Xc�A�7*

lossd.o<�z?i       �	��Xc�A�7*

loss�}�;�N=B       �	���Xc�A�7*

loss��;�`�       �	2s�Xc�A�7*

loss3ſ<V�+       �	Y�Xc�A�7*

loss�4#<�}�       �	��Xc�A�7*

loss@ �<���       �	�h�Xc�A�7*

loss���<�~�       �	�Xc�A�7*

loss�&�=`e�;       �	���Xc�A�7*

loss���:_���       �	�d�Xc�A�7*

loss_<<d�
�       �	N	�Xc�A�7*

loss�	-;�$c�       �	���Xc�A�7*

loss��;I�=       �	wM�Xc�A�7*

loss#� =Q���       �	���Xc�A�7*

loss�b<u�       �	^��Xc�A�7*

loss�	=1��;       �	G�Xc�A�7*

lossM�<�@'       �	?��Xc�A�7*

lossvh<��       �	֋�Xc�A�7*

lossv�y:-��       �	{0�Xc�A�7*

loss#%<^�ů       �	���Xc�A�7*

loss,+;o\��       �	x}�Xc�A�7*

loss�<�       �	�� �Xc�A�7*

lossJ�<��&       �	��!�Xc�A�7*

lossi��;,;�       �	�3"�Xc�A�7*

loss�he;��s       �	��"�Xc�A�7*

lossWpr<D`G�       �	�#�Xc�A�7*

loss���<����       �	�#$�Xc�A�7*

loss@?=�n�       �	s�$�Xc�A�7*

loss?�;����       �	[]%�Xc�A�7*

lossō};^ry       �	��%�Xc�A�7*

loss(�<�Ѥ       �	 �&�Xc�A�7*

lossJ�j;O���       �	F]'�Xc�A�7*

loss���<���       �	�(�Xc�A�7*

loss���<!GC`       �	y)�Xc�A�7*

loss���<��`%       �	�B*�Xc�A�7*

lossweG<oS&       �	��*�Xc�A�7*

loss�d<2��h       �	��+�Xc�A�7*

lossZ� ;+��q       �	��,�Xc�A�7*

lossռ;��;4       �	_-�Xc�A�7*

loss䰋:˼B+       �	�T.�Xc�A�7*

loss`}�;yn�       �	�P/�Xc�A�7*

loss� =�)ֆ       �	B�0�Xc�A�7*

loss�%;�       �	g1�Xc�A�7*

loss�=\;�<        �	��2�Xc�A�7*

loss !�<�a��       �	�3�Xc�A�7*

loss鯰<���<       �	�4�Xc�A�7*

lossm�;2�^e       �	�F5�Xc�A�7*

lossC�;��]�       �	b�5�Xc�A�7*

lossxMI<�V4�       �	��6�Xc�A�7*

loss�H9;���       �	s+8�Xc�A�7*

loss�v�8�E�       �	��8�Xc�A�7*

loss���;����       �	F�9�Xc�A�7*

lossqd;uh       �	�:�Xc�A�7*

loss��<;��@       �	ka;�Xc�A�7*

loss��;DϬ       �	�R<�Xc�A�7*

loss�}�9�|I       �	dt=�Xc�A�7*

loss]�:<5���       �	?q>�Xc�A�7*

loss�˴9�sZ       �	�S?�Xc�A�7*

loss��8ڒ��       �	�?�Xc�A�7*

losswhU;Ղ��       �	��@�Xc�A�7*

loss|�:����       �	@�A�Xc�A�7*

lossm�K;?Z�       �	�dB�Xc�A�7*

loss�<l��;       �	�C�Xc�A�7*

lossa��9��s�       �	*�C�Xc�A�7*

loss(�"<�<
�       �	��D�Xc�A�7*

loss��E<x�Q.       �	Z�E�Xc�A�7*

loss�;�U�       �	Y0F�Xc�A�7*

lossR�m<֒v�       �	 �F�Xc�A�7*

loss`~<}d�%       �	��G�Xc�A�7*

loss�u�;��T       �	{�H�Xc�A�7*

loss�5= �C        �	�3I�Xc�A�7*

loss$��:	�A�       �	��I�Xc�A�7*

losso/�=��G�       �	�J�Xc�A�7*

loss�N<�К�       �	{2K�Xc�A�7*

lossr��;O���       �	M�K�Xc�A�7*

lossW~N<5%�       �	_zL�Xc�A�7*

loss鍨;e"�       �	3lM�Xc�A�7*

lossm��<n^�       �	#N�Xc�A�7*

lossD�m<&xώ       �	>�N�Xc�A�7*

loss��;��v       �	rQO�Xc�A�7*

loss��<4{��       �	�O�Xc�A�7*

lossʏ�=��u�       �	U�P�Xc�A�7*

loss�F0=$�W       �	�KQ�Xc�A�7*

loss�q;�J�       �	�$R�Xc�A�7*

loss�H�<#��2       �	2�R�Xc�A�8*

loss2�<P� $       �	bgS�Xc�A�8*

loss&�k:km��       �	[
T�Xc�A�8*

loss��F<����       �	�T�Xc�A�8*

loss�:�;g���       �	�DU�Xc�A�8*

loss�/f;��       �	/�U�Xc�A�8*

lossw�U<�jV       �	}xV�Xc�A�8*

loss!5l;����       �	�W�Xc�A�8*

lossVB=+t�"       �	�X�Xc�A�8*

loss�R�;��       �	��Y�Xc�A�8*

loss�p�<&�(       �	]3Z�Xc�A�8*

loss��<);       �	��[�Xc�A�8*

loss�6<Po�U       �	�U\�Xc�A�8*

loss��);k�_       �	��\�Xc�A�8*

loss�~:�e_�       �		�]�Xc�A�8*

loss:+�:���       �	�&^�Xc�A�8*

loss��q:��6s       �	e�^�Xc�A�8*

loss�e;8f"6       �	5b_�Xc�A�8*

loss&�X<*��       �		�_�Xc�A�8*

loss��y<��E       �	r�`�Xc�A�8*

lossͧk<map�       �	[?a�Xc�A�8*

loss�x�<�g�       �	��a�Xc�A�8*

loss{e�;] w�       �	�yb�Xc�A�8*

loss�<��l       �	,c�Xc�A�8*

lossTI�;_��V       �	�c�Xc�A�8*

loss�Kq<�<�       �	G=d�Xc�A�8*

loss��<E��Z       �	��d�Xc�A�8*

loss�~�;���Y       �	ke�Xc�A�8*

loss�=���       �	�f�Xc�A�8*

loss�3=,�       �	��f�Xc�A�8*

loss	�o<��)Y       �	Hg�Xc�A�8*

loss�ۛ:J�=�       �	H�g�Xc�A�8*

loss�91;��"       �	Ii�Xc�A�8*

loss!~M;���;