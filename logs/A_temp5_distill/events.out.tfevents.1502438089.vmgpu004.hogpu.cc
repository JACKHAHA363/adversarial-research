       �K"	  @�Xc�Abrain.Event:2|��"�     �])�	Q�D�Xc�A"��
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
:@@*
seed2��
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
dropout_1/keras_learning_phasePlaceholder*
_output_shapes
:*
shape: *
dtype0

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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2͑B*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
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
seed2��
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
seed2́�*
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
���������*
_output_shapes
: *
dtype0
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
 *  �?*
dtype0*
_output_shapes
: 
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
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
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
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_class
loc:@num_inst*
_output_shapes
: 
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
 *  �@*
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
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
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
]
PlaceholderPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������

L
div_2/yConst*
valueB
 *  �@*
dtype0*
_output_shapes
: 
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*
T0*'
_output_shapes
:���������

c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
out_type0*
_output_shapes
:*
T0
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
valueB:*
dtype0*
_output_shapes
:
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
Index0*
T0*
_output_shapes
:
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
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
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
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
_output_shapes
: *
T0
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
���������*
dtype0*
_output_shapes
:
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
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
T0*
_output_shapes
: 
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
value	B :*
dtype0*
_output_shapes
: 
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
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*
_output_shapes
: 
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
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
_output_shapes
: *
T0
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
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
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
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
valueB *
dtype0*
_output_shapes
: 
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
 *    *
dtype0*
_output_shapes
: 
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
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
_output_shapes
: *
T0
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&softmax_cross_entropy_loss_1/ones_likeFill,softmax_cross_entropy_loss_1/ones_like/Shape,softmax_cross_entropy_loss_1/ones_like/Const*
T0*
_output_shapes
: 
�
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
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
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
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
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
: 
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
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: 
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
valueB *
_output_shapes
: *
dtype0
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
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
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
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
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
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
valueB *
_output_shapes
: *
dtype0
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
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*
T0*#
_output_shapes
:���������
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
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
out_type0*
_output_shapes
:*
T0
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
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

Tdim0*'
_output_shapes
:���������*
T0
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
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
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
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
_output_shapes
: 
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
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
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
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2RealDiv@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1-sequential_1/dropout_1/cond/dropout/keep_prob*/
_output_shapes
:���������@*
T0
�
:gradients/sequential_1/dropout_1/cond/dropout/div_grad/mulMulOgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_2*/
_output_shapes
:���������@*
T0
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
valueB"      @   @   *
dtype0*
_output_shapes
:
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
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
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
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
j
zerosConst*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
conv2d_1/kernel/Adam
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
valueB@*    *&
_output_shapes
:@*
dtype0
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:@
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
 *w�+2*
_output_shapes
: *
dtype0
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
use_locking( 
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
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
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
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
 Bloss*
_output_shapes
: *
dtype0
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"s�#�,     nKqz	UH�Xc�AJ��
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
seed2��*
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2���*
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
:���*
seed2͑B
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
seed2���
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
seed2��
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
seed2́�*
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
div_1/yConst*
valueB
 *  �@*
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
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*

axis *
_output_shapes
:*
T0*
N
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
 *  �@*
dtype0*
_output_shapes
: 
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
value	B :*
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
T0*
_output_shapes
: 
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
���������*
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_loss_1/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
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
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
T0*
out_type0*
_output_shapes
:
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
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
T0*
_output_shapes
: 
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
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
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
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
out_type0*
_output_shapes
:*
T0
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
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
T0*
_output_shapes
: 
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*
_output_shapes
: 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
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
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
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
Dsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_likeFillJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeJsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:���������
�
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
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
#softmax_cross_entropy_loss_1/SelectSelect"softmax_cross_entropy_loss_1/Equal&softmax_cross_entropy_loss_1/ones_like(softmax_cross_entropy_loss_1/num_present*
T0*
_output_shapes
: 
�
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
y
'softmax_cross_entropy_loss_1/zeros_like	ZerosLike"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
T0*
_output_shapes
: 
P
Placeholder_1Placeholder*
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
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
�
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
T0*
_output_shapes
: 
�
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
_output_shapes
: *
T0
�
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
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
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
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
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
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
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
valueB *
_output_shapes
: *
dtype0
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
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
valueB:*
dtype0*
_output_shapes
:
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
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
valueB *
_output_shapes
: *
dtype0
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
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
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
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
�
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mulMul<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
Mgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumSumMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*
T0*#
_output_shapes
:���������
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
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
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
_
gradients/div_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_2_grad/RealDiv_1RealDivgradients/div_2_grad/Negdiv_2/y*
T0*'
_output_shapes
:���������

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
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������
*
T0
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
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
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
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Mul'sequential_1/dropout_1/cond/dropout/divKgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1*
T0*/
_output_shapes
:���������@
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
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
out_type0*
_output_shapes
:
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
valueB"      @   @   *
dtype0*
_output_shapes
:
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
valueB@*    *&
_output_shapes
:@*
dtype0
�
conv2d_1/kernel/Adam_1
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
valueB@@*    *
dtype0*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam
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
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
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
_output_shapes
:	�
*
dtype0
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
 *w�+2*
_output_shapes
: *
dtype0
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
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
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�

�
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
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
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0�g��       ��-	��s�Xc�A*

loss�\@��'�       ��-	�rt�Xc�A*

lossh�@\�3�       ��-	�u�Xc�A*

loss)�@?+(       ��-	ßv�Xc�A*

lossHM@�ɔ�       ��-	JBw�Xc�A*

loss�^�?��E�       ��-	�w�Xc�A*

loss���?��a�       ��-	/�x�Xc�A*

loss
�?����       ��-	9%y�Xc�A*

loss�ѿ?P#I�       ��-	C�y�Xc�A	*

lossH8�?�,*�       ��-	�jz�Xc�A
*

loss.�?�a�       ��-	�{�Xc�A*

lossVH]?Q��       ��-	��{�Xc�A*

loss��R?���{       ��-	mT|�Xc�A*

lossj�l?�5�       ��-	D�|�Xc�A*

loss	�A?H��       ��-	K�}�Xc�A*

lossg��?C�F       ��-	�+~�Xc�A*

loss�?��4       ��-	+�~�Xc�A*

loss�-:?ơ��       ��-	�p�Xc�A*

lossJ08?Q���       ��-	���Xc�A*

loss2�M?�P�x       ��-	P���Xc�A*

lossh]?����       ��-	xD��Xc�A*

loss,�N?j�[       ��-	�偲Xc�A*

loss��3?��u       ��-	����Xc�A*

lossA�?,N#d       ��-	K#��Xc�A*

loss��^?��?k       ��-	Ã�Xc�A*

loss߇6?��&�       ��-	9_��Xc�A*

losstK?�d?       ��-	
���Xc�A*

loss�J?��I       ��-	����Xc�A*

losss�:??Հw       ��-	����Xc�A*

loss�[??D��       ��-	FE��Xc�A*

lossx�K?���/       ��-	#���Xc�A*

loss[i!? $rr       ��-	P���Xc�A *

loss�?Н��       ��-	+��Xc�A!*

loss2]A?���       ��-	Yĉ�Xc�A"*

loss��?��3       ��-	h]��Xc�A#*

loss�8 ?|�"�       ��-	���Xc�A$*

loss�0�>ᒪ�       ��-	?���Xc�A%*

loss��/?Sm�H       ��-	+��Xc�A&*

loss2�`?S0-       ��-	����Xc�A'*

loss��#?��o       ��-	���Xc�A(*

lossZ2*?4���       ��-	�?��Xc�A)*

lossO��>�~�       ��-	�܎�Xc�A**

lossW?����       ��-	0��Xc�A+*

loss�?�L�+       ��-	����Xc�A,*

lossAH?q��5       ��-	(I��Xc�A-*

loss��>=�$       ��-	�呲Xc�A.*

loss���>�:�U       ��-	􈒲Xc�A/*

loss�/�>����       ��-	-&��Xc�A0*

loss_\?]F�c       ��-	����Xc�A1*

loss�˹>�ή�       ��-	-]��Xc�A2*

loss{1�>��       ��-	����Xc�A3*

loss.�>�맔       ��-	t���Xc�A4*

loss]�>����       ��-	�7��Xc�A5*

loss�_?㦖^       ��-	�ϖ�Xc�A6*

loss�	]>�j�5       ��-	�e��Xc�A7*

loss���>[ Ř       ��-	����Xc�A8*

loss4�>�`Ӿ       ��-	����Xc�A9*

loss1�>�'�       ��-	9)��Xc�A:*

lossy?qm�       ��-	"ř�Xc�A;*

loss���>n�*       ��-	[\��Xc�A<*

loss�4?_��       ��-	M���Xc�A=*

loss�g�>�`7       ��-	����Xc�A>*

loss��>�#x       ��-	�6��Xc�A?*

loss`�>E䶁       ��-	
f��Xc�A@*

lossqch>��i       ��-	��Xc�AA*

loss��>o���       ��-	T���Xc�AB*

lossE?k`T�       ��-	�C��Xc�AC*

loss���>�-�       ��-	�۟�Xc�AD*

loss�<?h]s�       ��-	�w��Xc�AE*

loss8�>kƬ�       ��-	Z��Xc�AF*

loss]�>�x�a       ��-	����Xc�AG*

loss}��>b�-s       ��-	�O��Xc�AH*

loss��>�C�f       ��-	q碲Xc�AI*

loss��>�p��       ��-	A���Xc�AJ*

loss5�?�m�       ��-	���Xc�AK*

loss�#�>_=�       ��-	^���Xc�AL*

loss�?����       ��-	�S��Xc�AM*

loss$�?[�4       ��-	����Xc�AN*

loss	�>m�#�       ��-	5���Xc�AO*

lossVs�>2�k�       ��-	�>��Xc�AP*

loss�i�>0ʯ       ��-	�ۧ�Xc�AQ*

loss��?�Bx       ��-	r��Xc�AR*

loss��>�       ��-	���Xc�AS*

loss�c>7&�a       ��-	����Xc�AT*

loss���>'`��       ��-	�X��Xc�AU*

loss|�y>�j       ��-	E�Xc�AV*

loss=�>��       ��-	 ���Xc�AW*

losso�o>���X       ��-	�3��Xc�AX*

lossr]�>|��       ��-	�Ӭ�Xc�AY*

loss> �>�O��       ��-	t��Xc�AZ*

loss���>�(��       ��-	�
��Xc�A[*

loss��>�_��       ��-	G���Xc�A\*

lossǔ>��FV       ��-	�Q��Xc�A]*

lossF�>0�L       ��-	��Xc�A^*

loss��?m�fW       ��-	����Xc�A_*

loss2�:>��XR       ��-	� ��Xc�A`*

loss)Nu>@x'j       ��-	f���Xc�Aa*

loss���>��{d       ��-	"Q��Xc�Ab*

loss-Q?�T��       ��-	�5��Xc�Ac*

lossCپ>��l       ��-	w���Xc�Ad*

lossw^�>��+�       ��-	����Xc�Ae*

lossT�2>{K�O       ��-	FB��Xc�Af*

loss�*�>��{       ��-	oص�Xc�Ag*

loss� T>І9�       ��-	&p��Xc�Ah*

loss�eZ>� �4       ��-	Ra��Xc�Ai*

lossv�>@E=�       ��-	���Xc�Aj*

loss�]�>��L       ��-	���Xc�Ak*

loss��>6p'`       ��-	WA��Xc�Al*

losstԨ>d��8       ��-	ݹ�Xc�Am*

loss�m�>��R�       ��-	󍺲Xc�An*

loss��>�#�w       ��-	�*��Xc�Ao*

loss-��>-D�$       ��-	mŻ�Xc�Ap*

lossiQ_>EI2       ��-	�^��Xc�Aq*

lossh>kl�       ��-	* ��Xc�Ar*

loss��=	��j       ��-	^���Xc�As*

loss�0>�Of=       ��-	�4��Xc�At*

loss�w>��Ʋ       ��-	(Ծ�Xc�Au*

loss�Vv>"�       ��-	Hn��Xc�Av*

loss�&�>d�       ��-	t��Xc�Aw*

lossqFT>���       ��-	e���Xc�Ax*

loss�S`>���       ��-	`>��Xc�Ay*

loss�R>��       ��-	����Xc�Az*

loss�k>y��4       ��-	p²Xc�A{*

lossV��>	�       ��-	�òXc�A|*

loss�z`=�ޔ�       ��-	g�òXc�A}*

lossf�|>��       ��-	z5ĲXc�A~*

loss���>�}ҭ       ��-	K�ĲXc�A*

lossމ>bP��       �	xcŲXc�A�*

loss�g�>�D�r       �	L�ŲXc�A�*

loss��z>z���       �	��ƲXc�A�*

lossi~>��ٹ       �	�0ǲXc�A�*

loss�i�>�&O       �	��ǲXc�A�*

lossl�=g.�x       �	W`ȲXc�A�*

loss��`>!E�       �	#�ȲXc�A�*

loss� G>DA�C       �	7�ɲXc�A�*

loss�%g>H��       �	NCʲXc�A�*

loss�Hs>�a�       �		�ʲXc�A�*

loss��8>���1       �	�z˲Xc�A�*

loss�(>S*��       �	�̲Xc�A�*

lossD�=n7��       �	0�̲Xc�A�*

lossQ=�=���       �	2XͲXc�A�*

loss<�>n#!�       �	��ͲXc�A�*

loss-�n>a�Ҥ       �	ıβXc�A�*

loss��B>Վ�M       �	�iϲXc�A�*

loss/,|>�2F�       �	�&вXc�A�*

loss��=W&��       �	K�ѲXc�A�*

lossO|=>���       �	$�ҲXc�A�*

loss�ɮ=��       �	�ZӲXc�A�*

lossv>��\�       �	E/ԲXc�A�*

loss��>M�;�       �	d�ԲXc�A�*

loss��>I�M       �	�nղXc�A�*

loss��>zzU       �	ֲXc�A�*

loss�P�>y��       �	��ֲXc�A�*

loss?;o>|Isg       �	paײXc�A�*

loss��I>5YC�       �	C زXc�A�*

loss��=����       �	{�زXc�A�*

loss�>O>��'       �	i�ٲXc�A�*

loss6�>vG9       �	��ڲXc�A�*

loss<�> &�`       �	~8۲Xc�A�*

loss.vh>yS��       �	�)ܲXc�A�*

loss��>�*q�       �	dݲXc�A�*

loss�%�>Q�D�       �	�޲Xc�A�*

loss��=>�TeW       �	p�޲Xc�A�*

loss.M�=͂t�       �	�^߲Xc�A�*

loss��2>a��       �	��Xc�A�*

loss�" >�۔�       �	b��Xc�A�*

loss)�?>4Dr�       �	i�Xc�A�*

loss��e>���       �	�Xc�A�*

loss6<>�\hJ       �	¾�Xc�A�*

loss��n>��u       �	�`�Xc�A�*

loss��=�g�r       �	A�Xc�A�*

loss(%�=�sZ       �	J��Xc�A�*

lossԛ>��       �	$b�Xc�A�*

loss���>D�)	       �	&��Xc�A�*

loss-�z=��]�       �	���Xc�A�*

loss�A�>�e       �	p@�Xc�A�*

loss�>G�       �	U��Xc�A�*

losssb>����       �	�|�Xc�A�*

lossv>H��
       �	��Xc�A�*

loss�[I>-��       �	V��Xc�A�*

lossj��=\[��       �	c^�Xc�A�*

loss�>�$�
       �	���Xc�A�*

loss��>����       �	c��Xc�A�*

loss~�=\φH       �	^J�Xc�A�*

loss�{?>�N�T       �	;��Xc�A�*

loss���=��E�       �	���Xc�A�*

lossjSW>HZ0       �	�Z�Xc�A�*

loss��O>fǫ�       �	���Xc�A�*

lossG`>iF�       �	���Xc�A�*

lossQ�&>�SN�       �	�k�Xc�A�*

lossCi�>��       �	��Xc�A�*

loss�9>:�k.       �	T��Xc�A�*

loss��>ً;�       �	�F�Xc�A�*

loss!>���n       �	O#�Xc�A�*

loss��=/�z;       �	���Xc�A�*

lossh�I>��y�       �	.���Xc�A�*

loss��<>�AV�       �	����Xc�A�*

loss��>>q Z       �	�`��Xc�A�*

loss���>`�$       �	����Xc�A�*

loss��>F���       �	����Xc�A�*

loss�p>D�        �	w-��Xc�A�*

loss�>��       �	����Xc�A�*

loss� >��{�       �	\t��Xc�A�*

lossf*�=���       �	P��Xc�A�*

loss�B>���       �	����Xc�A�*

loss���=c\��       �	�4��Xc�A�*

lossׂ>���       �	���Xc�A�*

lossd�>bC�       �	|{��Xc�A�*

loss?�?>"��z       �	����Xc�A�*

loss��i>Y�E2       �	�h��Xc�A�*

loss�އ>�EQd       �	� �Xc�A�*

loss�8�=�tW       �	P� �Xc�A�*

loss`.�=`x�v       �	��Xc�A�*

loss�=�1S       �	��Xc�A�*

loss���>R1@�       �	yZ�Xc�A�*

loss웂>N3��       �	��Xc�A�*

loss�6�=�[       �	|��Xc�A�*

lossc�>�[�X       �	�?�Xc�A�*

loss�:�>��t       �	���Xc�A�*

loss
4�><��       �	�|�Xc�A�*

loss_BC=�py       �	C�Xc�A�*

lossߌ�=���       �	���Xc�A�*

lossP8�>��        �	�a�Xc�A�*

loss�ۗ>�ѧi       �	v��Xc�A�*

loss�>/�I       �	��	�Xc�A�*

loss��->�5�H       �	H3
�Xc�A�*

lossq>%X}�       �	��
�Xc�A�*

lossF�>���S       �	*o�Xc�A�*

lossW+o>uAGr       �	�Xc�A�*

loss�w�=��5+       �	��Xc�A�*

losso��=�u��       �	1@�Xc�A�*

lossTJn=#S~       �	�y�Xc�A�*

loss㴰>BL��       �	�j�Xc�A�*

loss��8>��       �	�Xc�A�*

lossOj>j�       �	���Xc�A�*

loss
Y0>|�       �	MN�Xc�A�*

lossF�>*.I�       �	X��Xc�A�*

loss��>nr�3       �	��Xc�A�*

lossd�H>f|a       �	0)�Xc�A�*

loss�D
>�7��       �	/��Xc�A�*

lossiA�=��!X       �	�b�Xc�A�*

loss��>��~       �	��Xc�A�*

loss�.d>�B�       �	N��Xc�A�*

loss��>xe�       �	ס�Xc�A�*

loss�^R>o %5       �	�9�Xc�A�*

loss�6>�P�       �	��Xc�A�*

losse*>Q&��       �	�p�Xc�A�*

loss)_�=M�Y�       �	��Xc�A�*

lossZ��=���       �	��Xc�A�*

loss:L[>�8uO       �	n0�Xc�A�*

loss�->� gZ       �	���Xc�A�*

loss�>�6�       �	�Y�Xc�A�*

lossr?>e�0�       �	���Xc�A�*

loss�ɒ=�]l       �	��Xc�A�*

loss�=�n��       �	��Xc�A�*

loss�0�=�$�Y       �	B��Xc�A�*

loss�؀>u@��       �	I�Xc�A�*

loss�ߌ>���       �	���Xc�A�*

loss���=��e0       �	j��Xc�A�*

loss�c,>Y^CK       �	# �Xc�A�*

loss>���       �	^� �Xc�A�*

loss�>��       �	�V!�Xc�A�*

loss䵨>�(       �	��!�Xc�A�*

loss���=zY��       �	��"�Xc�A�*

loss��>� �0       �	�?#�Xc�A�*

lossxj5>6գ'       �	��#�Xc�A�*

loss�6�=�,�       �	k$�Xc�A�*

loss��6>����       �	�%�Xc�A�*

loss�W�>���G       �	P�%�Xc�A�*

loss��=���       �	~U&�Xc�A�*

lossӚ�=����       �	��&�Xc�A�*

loss�>��&�       �	��'�Xc�A�*

loss�FA>�(��       �	�D(�Xc�A�*

loss��(>�       �	 �(�Xc�A�*

loss���=?�k       �	˅)�Xc�A�*

lossԇy>�+��       �	�$*�Xc�A�*

loss��=�>ʣ       �	�*�Xc�A�*

lossc �=��j�       �	�`+�Xc�A�*

loss̓�=o�g�       �	�
,�Xc�A�*

lossF6>�>ُ       �	�,�Xc�A�*

loss�_�<&��       �	II-�Xc�A�*

loss���=EoE       �	u�-�Xc�A�*

loss3��>����       �	K�.�Xc�A�*

loss�=}T�>       �	�8/�Xc�A�*

lossf�->]�N�       �	;�/�Xc�A�*

lossj�!>�/�{       �	˃0�Xc�A�*

loss�>�0#�       �	[%1�Xc�A�*

loss{�=U!��       �	��1�Xc�A�*

loss\�>�d��       �	i2�Xc�A�*

lossw>��"~       �	�3�Xc�A�*

loss�YW= �/       �	��3�Xc�A�*

loss=">zMlC       �	��4�Xc�A�*

lossS��=5 Y:       �	�5�Xc�A�*

loss�~=�n@       �	��6�Xc�A�*

lossͽ�=ߑG]       �	^e7�Xc�A�*

losszU�>yUD       �	Y8�Xc�A�*

loss�P>'�"T       �	�9�Xc�A�*

loss�Vo>a�m       �	��9�Xc�A�*

loss_�,>�٢       �	-�:�Xc�A�*

loss#<u>���       �	w�;�Xc�A�*

loss6>�3	�       �	�,<�Xc�A�*

lossз>X��       �	7�<�Xc�A�*

loss�	�=t��       �	d�=�Xc�A�*

loss��L>D�u2       �	}>�Xc�A�*

loss!ʟ=��       �	[$?�Xc�A�*

loss1N�>�d��       �	��?�Xc�A�*

loss���=�^       �	0e@�Xc�A�*

loss!�'>�I�)       �	�lA�Xc�A�*

loss�@�=��O       �	�B�Xc�A�*

lossE�'>F��       �	�B�Xc�A�*

loss��>eC?H       �	R`C�Xc�A�*

loss��w>"�d       �	WD�Xc�A�*

loss�C(>x�O�       �	��D�Xc�A�*

loss�v�=
�<       �	�UE�Xc�A�*

loss[��=�D��       �	&�E�Xc�A�*

loss���>0��       �	4�F�Xc�A�*

loss��=�J%       �	�9G�Xc�A�*

lossLRr=g���       �	Z�G�Xc�A�*

loss��>P�g}       �	hxH�Xc�A�*

loss�->AY�       �	�I�Xc�A�*

loss�B�=��N/       �	3�I�Xc�A�*

loss���>�z�       �	jJ�Xc�A�*

loss��>����       �	>K�Xc�A�*

loss�+�=���v       �	{�K�Xc�A�*

loss�+a>�煃       �	fML�Xc�A�*

lossÏd>�
?�       �	��L�Xc�A�*

loss�f�=e�	E       �	�M�Xc�A�*

lossdb�=�*��       �	 �N�Xc�A�*

loss8�I>�7N�       �	�DO�Xc�A�*

loss*+�>��       �	�O�Xc�A�*

loss,i�=�qC�       �	��P�Xc�A�*

loss��=�g(�       �	>vQ�Xc�A�*

lossa��=�@�j       �	�R�Xc�A�*

loss,)�=�3�k       �	Z�R�Xc�A�*

loss��4>#�t�       �	YS�Xc�A�*

loss���=a��       �	=�S�Xc�A�*

loss��>i��,       �	�T�Xc�A�*

lossGI>�3��       �	a4U�Xc�A�*

loss�>�H�       �	j�U�Xc�A�*

lossZ�=���       �	�V�Xc�A�*

loss���=q=J       �	6W�Xc�A�*

loss{n%=���       �	��W�Xc�A�*

lossN��=B�h       �	�qX�Xc�A�*

loss�A�=w��g       �	IY�Xc�A�*

loss�o�> �rO       �	��Y�Xc�A�*

loss�@V>�!�"       �	:�Z�Xc�A�*

loss���=���       �	�^[�Xc�A�*

lossH>gQ�       �	C\�Xc�A�*

loss��0>�.�M       �	��\�Xc�A�*

loss8>��       �	5]�Xc�A�*

lossY)>7�\.       �	O�]�Xc�A�*

loss^�>:jU'       �	uu^�Xc�A�*

lossd�=h8}w       �	/_�Xc�A�*

loss��=P�Tr       �	.�_�Xc�A�*

lossN�^>��y       �	�A`�Xc�A�*

loss��E=\H	�       �	T�`�Xc�A�*

lossf��=��       �	j�a�Xc�A�*

loss��=��A�       �	c%b�Xc�A�*

lossS9�=�X��       �	��b�Xc�A�*

loss$��=�]�       �	Lmc�Xc�A�*

loss�O�=��,       �	
d�Xc�A�*

loss�h>%�h        �	�d�Xc�A�*

loss��,>o�s       �	SZe�Xc�A�*

loss�g(>��3�       �	�e�Xc�A�*

loss�.>1y�       �	Ƌf�Xc�A�*

loss��>��       �	F(g�Xc�A�*

lossH"i>e���       �	�g�Xc�A�*

loss`Z�=\+E�       �	�ph�Xc�A�*

loss
�=�ps       �	�i�Xc�A�*

loss��>AS��       �	*�i�Xc�A�*

loss.�.>��i       �	�Ej�Xc�A�*

loss�>nL�7       �	~�j�Xc�A�*

lossH��=�,�@       �	�|k�Xc�A�*

lossf�>#�{�       �	�!l�Xc�A�*

loss��>q�5�       �	
�l�Xc�A�*

loss�h\>�R�       �	�Ym�Xc�A�*

loss�R�=���       �	�m�Xc�A�*

lossE;�="�       �	-�n�Xc�A�*

lossB#>��w       �	a2o�Xc�A�*

loss�A>,��       �	��o�Xc�A�*

loss"�>*��       �	fjp�Xc�A�*

lossH[>���       �	�q�Xc�A�*

loss���=
,       �	��q�Xc�A�*

lossF��=cq        �	�Or�Xc�A�*

loss��=����       �	B�r�Xc�A�*

loss-��=�wOy       �	Ίs�Xc�A�*

loss]�=�        �	_'t�Xc�A�*

loss+�>j��       �	��t�Xc�A�*

loss���=��$�       �	Swu�Xc�A�*

loss�J=>�	��       �	�Jv�Xc�A�*

loss7q�=��}!       �	��v�Xc�A�*

loss#}">qX��       �	�$x�Xc�A�*

loss @�=�-�       �	�.y�Xc�A�*

lossī>�]Md       �	L�y�Xc�A�*

loss<�>>H&>       �	}�z�Xc�A�*

loss �m>��7       �	�n{�Xc�A�*

loss:�E=� �       �	J|�Xc�A�*

loss��=�N       �	��|�Xc�A�*

loss���=���       �	v�}�Xc�A�*

loss}�3>%��v       �	�l~�Xc�A�*

loss�� >V��p       �	^�Xc�A�*

loss�->RQ��       �	�U��Xc�A�*

loss�a>|�       �	A���Xc�A�*

loss��=O��F       �	]���Xc�A�*

loss�K=�#       �	�#��Xc�A�*

lossS >�lrE       �	Cɂ�Xc�A�*

loss�>�f       �	&o��Xc�A�*

lossR��=�V�l       �	l��Xc�A�*

loss M">�*'>       �	����Xc�A�*

loss��%>���       �	a4��Xc�A�*

loss�>�yN       �	�˅�Xc�A�*

loss�|H>����       �	^d��Xc�A�*

loss���=���       �		���Xc�A�*

loss��<>�[�       �	ٔ��Xc�A�*

loss��>��!H       �	o.��Xc�A�*

loss��Y>Q��	       �	6ʈ�Xc�A�*

loss�z=���:       �	�a��Xc�A�*

loss�w�=Y58        �	����Xc�A�*

loss�U>(��       �	є��Xc�A�*

loss��>=��w       �	�3��Xc�A�*

lossa��=���       �	�̋�Xc�A�*

loss�K�=����       �	Rd��Xc�A�*

loss���=�5f�       �	?��Xc�A�*

loss��V>]��       �	敍�Xc�A�*

loss .>�u��       �	�;��Xc�A�*

loss_>�>��V{       �	�ڎ�Xc�A�*

lossR��=��b       �	Bw��Xc�A�*

loss�>>����       �	�]��Xc�A�*

loss;��=��       �	����Xc�A�*

loss"�=U��       �	����Xc�A�*

loss��K>���4       �	�?��Xc�A�*

lossȥ>���       �	e���Xc�A�*

loss��=Fk�J       �	�x��Xc�A�*

loss�>Nf��       �	M/��Xc�A�*

losst�=m��       �	B\��Xc�A�*

loss!~c=\1�       �	f���Xc�A�*

loss}5�=eq{}       �	Ց��Xc�A�*

loss� >b�E%       �	3��Xc�A�*

loss�B=.��       �	�͗�Xc�A�*

lossf��=�;�x       �	�v��Xc�A�*

loss�;]>^2�?       �	���Xc�A�*

loss�2g>*^5�       �	T���Xc�A�*

loss��=�c       �	�K��Xc�A�*

lossڐ�=61�p       �	�⚳Xc�A�*

loss��=ZS:�       �	�w��Xc�A�*

lossL�=>s��J       �	A��Xc�A�*

lossTc�>KF�       �	P���Xc�A�*

loss\�1>��z�       �	�?��Xc�A�*

loss�N�=���"       �	Zٝ�Xc�A�*

loss�)]> A       �	ly��Xc�A�*

lossa�I>QP8       �	���Xc�A�*

lossd�{=�m?       �	沟�Xc�A�*

loss��g=��Á       �	�M��Xc�A�*

loss��:>|       �	�頳Xc�A�*

loss�y�>���T       �	����Xc�A�*

loss��)>c��_       �	�$��Xc�A�*

loss�Z>z���       �	M���Xc�A�*

lossH�=�kG        �	J^��Xc�A�*

loss.��=�k�q       �	����Xc�A�*

lossa Q>����       �	���Xc�A�*

lossT�=m��       �	A��Xc�A�*

loss�D�=�) �       �	Yޥ�Xc�A�*

loss��=�/��       �	���Xc�A�*

loss��>]��N       �	���Xc�A�*

lossi:{=�le4       �	-���Xc�A�*

loss��=T@��       �	�I��Xc�A�*

lossM'�=r5�       �	娳Xc�A�*

loss�(�=�� �       �	s���Xc�A�*

loss]f�=���       �	���Xc�A�*

loss�	�<��&       �	����Xc�A�*

loss�F>��r       �	U��Xc�A�*

lossc?	>����       �	��Xc�A�*

loss@�b>$ST       �	Ō��Xc�A�*

lossv�=��       �	�$��Xc�A�*

lossl�l=[�N�       �	����Xc�A�*

lossn�N=
S��       �	LQ��Xc�A�*

loss]�5=���       �	�鮳Xc�A�*

lossjq>�(�       �	����Xc�A�*

lossE`x=�0tN       �	�3��Xc�A�*

loss��>�[�       �	۰�Xc�A�*

loss�̩>(E	       �	8���Xc�A�*

lossn�v=���6       �	�&��Xc�A�*

loss��=2fJ�       �	�̲�Xc�A�*

losst�=���a       �	�c��Xc�A�*

lossO,O=��@       �	���Xc�A�*

loss-Zg=��       �	����Xc�A�*

loss��=l�K�       �	�H��Xc�A�*

loss�s3>�]       �	,��Xc�A�*

lossN�=>V       �	n޶�Xc�A�*

loss��0>��N�       �	:��Xc�A�*

lossL>?�$o       �	�ϸ�Xc�A�*

loss@~d>6�3�       �	si��Xc�A�*

lossCԔ=�^�       �	���Xc�A�*

loss���=�C_8       �	����Xc�A�*

loss{ �=l3!       �	�9��Xc�A�*

lossm�=���       �	ػ�Xc�A�*

loss7�*>i��Q       �	�p��Xc�A�*

loss�m�=G�:       �	���Xc�A�*

loss-�=�Ʃ�       �	=���Xc�A�*

lossw
w>A�<�       �	�P��Xc�A�*

loss���=�dX�       �	9쾳Xc�A�*

loss��`=Z`(�       �	f���Xc�A�*

loss�w�=��w       �	�&��Xc�A�*

lossf�+=R��       �	����Xc�A�*

loss-h�=�R@       �	m��Xc�A�*

loss�J�=nZ^       �	�³Xc�A�*

loss�_>�k       �	�³Xc�A�*

loss6n�=E#       �	�ZóXc�A�*

loss��>�#k�       �	��óXc�A�*

loss��#>Ve=�       �	��ĳXc�A�*

losslv>AU�       �	G9ųXc�A�*

loss��=E��       �	j�ųXc�A�*

loss�A=�v�c       �	�ƳXc�A�*

loss(h>�I&(       �	g&ǳXc�A�*

lossԢ�=�A �       �	&�ǳXc�A�*

loss��=P��k       �	�iȳXc�A�*

loss�j�=���C       �	�ɳXc�A�*

loss�x�=��b4       �	�ɳXc�A�*

loss��x=�G,       �	�KʳXc�A�*

lossX*>�'��       �	L�ʳXc�A�*

loss�X=9t\[       �	��˳Xc�A�*

lossT&>͑�       �	�+̳Xc�A�*

loss�p:=��       �	��̳Xc�A�*

loss&�?=Ðdd       �	�]ͳXc�A�*

loss}�}=���       �	7�ͳXc�A�*

loss/�v=��$�       �	1
ϳXc�A�*

loss�D�=�$        �	��ϳXc�A�*

loss�V1>�d!2       �	(|гXc�A�*

loss�z[>�kT!       �	�ѳXc�A�*

loss�=]�x`       �	�ѳXc�A�*

loss�=�,,�       �	�QҳXc�A�*

loss=�=�4��       �	��ҳXc�A�*

loss3��=�zݿ       �	��ӳXc�A�*

lossR�<,��       �	` ԳXc�A�*

losswL=7���       �	��ԳXc�A�*

loss�շ<X,/       �	rQճXc�A�*

loss5ɞ<:G�       �	��ճXc�A�*

loss$T<��e�       �	�ֳXc�A�*

loss&7=68�       �	�׳Xc�A�*

loss�)�=s���       �	��׳Xc�A�*

lossĮ>;��ժ       �	�PسXc�A�*

loss���;^��       �	��سXc�A�*

loss*�;�P�       �	��ٳXc�A�*

loss!A�=�-�{       �	c)ڳXc�A�*

loss�`>����       �	3�ڳXc�A�*

lossH`6=���       �	_]۳Xc�A�*

loss�3<#B��       �	��۳Xc�A�*

loss�4=�+G        �	ڎܳXc�A�*

loss���>ٯ�Z       �	�0ݳXc�A�*

loss�1~<���B       �	��ݳXc�A�*

loss]p�=��
�       �	�j޳Xc�A�*

lossD��=�:8�       �	�߳Xc�A�*

loss�A�="R��       �	r�߳Xc�A�*

loss�Ӆ=�Λ       �	`:�Xc�A�*

loss��=��7W       �	g��Xc�A�*

loss�>!�q       �	�x�Xc�A�*

loss�>";`       �	��Xc�A�*

loss��">��7       �	E��Xc�A�*

loss��2>N��M       �	2Y�Xc�A�*

loss��>S��       �	��Xc�A�*

loss}�!>���       �	��Xc�A�*

lossL�>Z/�	       �	)�Xc�A�*

lossX��=�r�       �	���Xc�A�*

loss]]>����       �	�]�Xc�A�*

loss�?>�u+       �	���Xc�A�*

loss|[=��       �	ʌ�Xc�A�*

loss���=,���       �	�"�Xc�A�*

lossf��=����       �	���Xc�A�*

loss���=ȣ�       �	�O�Xc�A�*

lossZ+�=����       �	��Xc�A�*

loss���=�b5P       �	W{�Xc�A�*

lossf��=^��       �	'�Xc�A�*

loss)��<j$jn       �	3��Xc�A�*

loss��K=�+�w       �	���Xc�A�*

lossQ��<k�]       �	Q��Xc�A�*

lossMs>���c       �	B���Xc�A�*

lossW<'=��h,       �	h��Xc�A�*

lossN�>���~       �	r1�Xc�A�*

loss�R#>���       �	<��Xc�A�*

loss6�=r�vY       �	���Xc�A�*

loss���=H�       �	. �Xc�A�*

loss�=��       �	���Xc�A�*

loss�4�<�5(k       �	Lp�Xc�A�*

loss�Q>����       �	;�Xc�A�*

losss�W=B�       �	���Xc�A�*

lossZ�[=���q       �	�f��Xc�A�*

loss�t�=�)b�       �	���Xc�A�*

loss�">�k��       �	f���Xc�A�*

loss�\�=����       �	vQ��Xc�A�*

loss	�f=��΢       �	~���Xc�A�*

loss>�=J(zz       �	`���Xc�A�*

loss���=�7��       �	�!��Xc�A�*

lossH��=�ZS%       �	Q���Xc�A�*

loss��>5��}       �	���Xc�A�*

loss��=�-P_       �	���Xc�A�*

loss�>,	u�       �	����Xc�A�*

loss���=DHp�       �	AE��Xc�A�*

lossi	�=#�0�       �	����Xc�A�*

loss�e�=*#o�       �	.���Xc�A�*

lossX�=r�:       �	ӈ��Xc�A�*

lossS� >|       �	� �Xc�A�*

loss0�=3~D       �	�� �Xc�A�*

loss��L>�`       �	�D!�Xc�A�*

loss�2>�n�       �	c�!�Xc�A�*

lossN>_��       �	��"�Xc�A�*

loss��$=���       �	�4#�Xc�A�*

lossm�=��pq       �	N�#�Xc�A�*

losss��=.B       �	�i$�Xc�A�*

loss�J>F�J�       �	y%�Xc�A�*

loss��>�*Ͳ       �	¤%�Xc�A�*

loss��=�k`�       �	�@&�Xc�A�*

lossL�i=�r�o       �	��&�Xc�A�*

loss���=�M��       �	E'�Xc�A�*

loss�j�=�       �	a(�Xc�A�*

loss�`�=��	�       �	�(�Xc�A�*

loss�i>�lk�       �	�h)�Xc�A�*

loss���=	b	       �	*�Xc�A�*

lossTc�=��S�       �	ѭ*�Xc�A�*

lossԠO=m~��       �	}Y+�Xc�A�*

loss_>r>��       �	C,�Xc�A�*

losss�=��#       �	�,�Xc�A�*

loss�ݾ=�v��       �	�C-�Xc�A�*

lossm}q=7eu�       �	R�-�Xc�A�*

loss��!>c��       �	�.�Xc�A�*

loss��=��q       �	�2/�Xc�A�*

loss��=Y�*H       �	,�/�Xc�A�*

lossx�=/�?       �	�l0�Xc�A�*

loss�B =n�E       �	�1�Xc�A�*

loss���=��@a       �	k�1�Xc�A�*

loss��>�l`�       �	�R2�Xc�A�*

lossK�>Qv'       �	��2�Xc�A�*

loss��=��3+       �	�Q4�Xc�A�*

loss�<�=���       �	4�4�Xc�A�*

loss�k?>v1��       �	)�5�Xc�A�*

loss�E�=G�       �	�;6�Xc�A�*

loss��&>!Y�l       �	:]7�Xc�A�*

loss�3=�zY�       �	�8�Xc�A�*

loss���=||��       �	L9�Xc�A�*

loss���>��{^       �	�9�Xc�A�*

loss��2>��7       �	5c:�Xc�A�*

loss[�=_av�       �	_;�Xc�A�*

loss�>�❘       �	K�;�Xc�A�*

loss�d\=U��       �	<M<�Xc�A�*

loss���=��u       �	��<�Xc�A�*

loss@�M>���       �	Ȕ=�Xc�A�*

loss3��=����       �	�4>�Xc�A�*

loss�M>�Mk       �	��>�Xc�A�*

lossA�=�;�x       �	�i?�Xc�A�*

lossM��=��?       �	_@�Xc�A�*

loss�=�4��       �	��@�Xc�A�*

loss���=�I/       �	h\A�Xc�A�*

lossƸ�=3#x       �	��A�Xc�A�*

loss;Q�=(�       �	��B�Xc�A�*

loss��?qx��       �	�DC�Xc�A�*

loss��=Ճ�U       �	��C�Xc�A�*

lossX�B=����       �	܃D�Xc�A�*

loss�==��Q       �	XE�Xc�A�*

loss�^�<�<�       �	EF�Xc�A�*

loss$C�=!       �	��F�Xc�A�*

lossA�=R��       �	rOG�Xc�A�*

lossn�>�|1�       �	R�G�Xc�A�*

lossJ��=��       �	ԘH�Xc�A�*

loss�b�<�֫�       �	~8I�Xc�A�*

loss�+�=`�       �	��I�Xc�A�*

lossq��=�)��       �	,}J�Xc�A�*

loss��=�q	v       �	�K�Xc�A�*

loss���=�&ͷ       �	�+L�Xc�A�*

loss[�>�?�       �	O�L�Xc�A�*

lossA>y���       �	�gM�Xc�A�*

loss�ο=I'��       �	a�M�Xc�A�*

loss���=��       �	Z�N�Xc�A�*

lossp1>Tj�       �	)>O�Xc�A�*

loss�1�=�Gf       �	'�O�Xc�A�*

loss�=��(�       �	)�P�Xc�A�*

lossw��=IR��       �	ޮQ�Xc�A�*

loss�}=rn��       �	kHR�Xc�A�*

lossߊ�=Q�4       �	!�R�Xc�A�*

loss�a*> ��w       �	�}S�Xc�A�*

loss|W�=�k�o       �	�T�Xc�A�*

loss**�=�E{�       �	h�T�Xc�A�*

losso5�=+]@�       �	�IU�Xc�A�*

losse�>u��8       �	��U�Xc�A�*

lossS�>;ᨊ       �	�wV�Xc�A�*

loss$��<wN�       �	IW�Xc�A�*

loss��=�� �       �	צW�Xc�A�*

loss.HZ=�4       �	�>X�Xc�A�*

loss@\�=��       �	��X�Xc�A�*

loss�=�i�       �	�Y�Xc�A�*

loss"�=W�w       �	zZ�Xc�A�*

loss�S�=�6\       �	��Z�Xc�A�*

loss _>��J       �	�P[�Xc�A�*

loss4�= 	چ       �	_\�Xc�A�*

lossH��=��1�       �	��\�Xc�A�*

loss�n�=��       �	�D]�Xc�A�*

loss��(>=���       �	��]�Xc�A�*

loss��]=�6�2       �	a�^�Xc�A�*

loss���=��       �	Q2`�Xc�A�*

loss_� >X��       �	��a�Xc�A�*

lossw<C>PX]P       �	fJb�Xc�A�*

loss�t>���0       �	0�b�Xc�A�*

loss�f=���       �	ݙc�Xc�A�*

losst��<�7e       �	�7d�Xc�A�*

loss��=2F�       �	�ue�Xc�A�*

lossq<�=��ݏ       �	If�Xc�A�*

lossT&�=v�Wu       �	��f�Xc�A�*

loss$�7>5X�       �	�Lg�Xc�A�*

loss��<'��)       �	��g�Xc�A�*

lossz�,=$��       �	H�h�Xc�A�*

lossܐW>�"�       �	77i�Xc�A�*

loss�%�=���q       �	n�i�Xc�A�*

lossQUN>��?q       �	5�j�Xc�A�*

lossA|=�Y�I       �	MMk�Xc�A�*

loss�)�=c��       �	�k�Xc�A�*

lossrw�<2Tp�       �	=|l�Xc�A�*

lossw�=4*�       �	<m�Xc�A�*

loss ��<d�z�       �	��m�Xc�A�*

lossS>6=V8�d       �	�^n�Xc�A�*

loss��=����       �	��n�Xc�A�*

lossO^�=�pr       �	O�o�Xc�A�*

loss�k=��fd       �	�/p�Xc�A�*

loss�>ӿ��       �	��p�Xc�A�*

loss���=�ep\       �	�fq�Xc�A�*

loss���=��z       �	-r�Xc�A�*

loss�t>;o��       �	ƣr�Xc�A�*

loss��2=;�       �	�?s�Xc�A�*

loss諏=֢��       �	7�s�Xc�A�*

lossf �=׳l9       �	�wt�Xc�A�*

lossm��<��       �	&u�Xc�A�*

loss2>?a	       �	��u�Xc�A�*

loss�7�=��TY       �	�{v�Xc�A�*

loss���=H��_       �	�w�Xc�A�*

loss�?�=�.��       �	��x�Xc�A�*

loss*��=6-+)       �	�;y�Xc�A�*

lossq��<u�       �	��y�Xc�A�*

loss�м=^��       �	c�z�Xc�A�*

loss��<6�       �	PQ{�Xc�A�*

loss��=�}�       �	�|�Xc�A�*

loss��*=a)�n       �	%}�Xc�A�*

loss��=e�&<       �	��}�Xc�A�*

lossh�,=�(��       �	ё~�Xc�A�*

loss��@<��e�       �	5�Xc�A�*

loss?E�=��x       �	b��Xc�A�*

loss1��=xX 1       �	�o��Xc�A�*

loss�C,>���Y       �	F
��Xc�A�*

loss��>��j       �	&���Xc�A�*

loss��=j��y       �	,���Xc�A�*

loss9p�=���       �	���Xc�A�*

loss�_=M?>m       �	����Xc�A�*

lossn��<#��m       �	Cu��Xc�A�*

lossD6�=���       �	���Xc�A�*

lossn�t=�f�o       �	#���Xc�A�*

loss�>~��%       �	Eh��Xc�A�*

lossj��=��2>       �	y��Xc�A�*

loss�0�=�p�b       �	����Xc�A�*

loss���=�nlB       �	�M��Xc�A�*

lossJ��<���S       �	g퉴Xc�A�*

lossa�0=k8       �	����Xc�A�*

lossة�=��L       �	_$��Xc�A�*

loss6�>ZY��       �	
���Xc�A�*

loss���=�T}�       �	����Xc�A�*

loss߂�=�4E       �	*X��Xc�A�*

lossz�m=c��       �	a���Xc�A�*

lossr�>uƂu       �	���Xc�A�*

loss{x"=��S       �	�4��Xc�A�*

loss;p�=���       �	
א�Xc�A�*

loss�/T=nTr�       �	w���Xc�A�*

loss�1N=��       �	�;��Xc�A�*

lossv
�=^v�/       �	�В�Xc�A�*

loss|Δ=�X��       �	����Xc�A�*

loss]A�=���z       �	~R��Xc�A�*

loss�>>��       �	���Xc�A�*

loss_e�<3"4j       �	����Xc�A�*

loss�L"=�~k       �	1D��Xc�A�*

loss��*>�Q�       �	1ꖴXc�A�*

lossMKC<����       �	���Xc�A�*

loss�i=؉�       �	I,��Xc�A�*

loss<�4>g�ĝ       �	�ǘ�Xc�A�*

lossl_�=�       �	5c��Xc�A�*

loss@��=�*�       �	O"��Xc�A�*

loss*=�5r�       �	Ś�Xc�A�*

loss��;[1O�       �	�m��Xc�A�*

loss,��=���       �	���Xc�A�*

lossC��=�        �	����Xc�A�*

loss�f�<�ߵ�       �	jN��Xc�A�*

loss���<���       �	A�Xc�A�*

loss���==��r       �	O���Xc�A�*

loss6R�=dԐ6       �	/��Xc�A�*

loss͏5=�<oH       �	�ȟ�Xc�A�*

loss{h=�i�       �	l���Xc�A�*

loss�C�=��       �	n2��Xc�A�*

loss)>=�|       �	ˡ�Xc�A�*

loss�4=z�       �	�a��Xc�A�*

lossʡ=�r��       �	����Xc�A�*

loss �<���Z       �	���Xc�A�*

lossd��<J%       �	�/��Xc�A�*

lossh=5�|�       �	Xˤ�Xc�A�*

lossE��=xQ��       �	{f��Xc�A�*

lossx��=�b�       �	��Xc�A�*

loss`�>3y��       �	ǜ��Xc�A�*

lossC�&=OJȇ       �	�A��Xc�A�*

lossIJ=�'b�       �	�ܧ�Xc�A�*

loss���=��       �	�|��Xc�A�*

loss�q=��`C       �	���Xc�A�*

loss��=����       �	2"��Xc�A�*

losst��=j͡=       �	[Ϋ�Xc�A�*

loss`f7=��e�       �	�}��Xc�A�*

loss�D�=!���       �	)��Xc�A�*

loss�;�=��s]       �	�έ�Xc�A�*

loss��>=�YW       �	Gv��Xc�A�*

loss!}R=�]ȿ       �	G��Xc�A�*

lossj�=G�}�       �	����Xc�A�*

loss߿�=e�       �	)\��Xc�A�*

loss�vm=���I       �	� ��Xc�A�*

loss��<{Zح       �	<���Xc�A�*

lossJ2>h�5�       �	�B��Xc�A�*

lossN��=�\s       �	체Xc�A�*

loss�I~=�z       �	l���Xc�A�*

loss�
">�       �	�<��Xc�A�*

loss!��=�d��       �	�ݴ�Xc�A�*

loss�=a>ӕ�g       �	=���Xc�A�*

loss�"="h��       �	���Xc�A�*

loss�O�=Mca<       �	����Xc�A�*

lossq�=M^�|       �	����Xc�A�*

loss}@!>�%Թ       �	;���Xc�A�*

lossc�B=���7       �	+���Xc�A�*

lossT��=99�       �	����Xc�A�*

loss\.�=+�4        �	�ݻ�Xc�A�*

lossA,>�КT       �	Ov��Xc�A�*

loss`�+=�ng       �	�t��Xc�A�*

loss��=���S       �	xG��Xc�A�*

lossN�8=��       �	�ܾ�Xc�A�*

lossz�<��4@       �	)w��Xc�A�*

loss��
>��s       �	���Xc�A�*

losslѰ=�J�e       �	ū��Xc�A�*

loss�T�=����       �	,H��Xc�A�*

loss���=�9��       �	����Xc�A�*

loss=T=}��~       �	�s´Xc�A�*

loss/�= R��       �	ôXc�A�*

lossz�>����       �	��ôXc�A�*

loss$vV=9"�       �	�=ĴXc�A�*

loss�H&=��	�       �	��ĴXc�A�*

loss�`=~"       �	�nŴXc�A�*

loss��=�d�E       �	�ƴXc�A�*

loss_�=�0T       �	'�ƴXc�A�*

lossL�E=��8�       �	�MǴXc�A�*

lossH�*>ǁ�O       �	�ȴXc�A�*

loss�H�=���       �	 �ȴXc�A�*

loss��=Z���       �	U/ɴXc�A�*

loss�^=ܴl�       �	��ɴXc�A�*

loss��=z�#H       �	�nʴXc�A�*

lossͣ�=q���       �	 
˴Xc�A�*

loss8ݐ= �r9       �	��˴Xc�A�*

lossq=��qR       �	�G̴Xc�A�*

loss��R=��3z       �	�̴Xc�A�*

loss)Y<��j       �	�wʹXc�A�*

loss���<<�|       �	�δXc�A�*

loss�<�=�N��       �	��δXc�A�*

loss::>?��       �	�nϴXc�A�*

lossA��=:Z��       �	l	дXc�A�*

loss�5S=A�\       �	��дXc�A�*

loss�~�=l&��       �	9ѴXc�A�*

loss�D�<�^K       �	��ѴXc�A�*

lossW�=��ab       �	�cҴXc�A�*

lossS�t=v^��       �	�ҴXc�A�*

loss\w�=��|       �	q�ӴXc�A�*

loss��=�u�       �	�xԴXc�A�*

lossO�>�F:�       �	�մXc�A�*

lossc��=�Y94       �	m�մXc�A�*

loss#>��~       �	mִXc�A�*

loss��>N�       �	�״Xc�A�*

loss\@=/�k       �	d�״Xc�A�*

loss��:>����       �	�DشXc�A�*

loss
�$>[���       �	��شXc�A�*

lossS,�=�;�Q       �	yٴXc�A�*

loss���=t�<       �	�ڴXc�A�*

loss��>~���       �	<�ڴXc�A�*

lossd��=��
�       �	 W۴Xc�A�*

loss2/=�I\       �	}�۴Xc�A�*

loss�)�<(��       �	��ܴXc�A�*

loss1��=�k��       �	} ݴXc�A�*

loss�<{HiL       �	��ݴXc�A�*

loss��&=�`2�       �	�R޴Xc�A�*

loss��>Er�       �	-�޴Xc�A�*

loss�-�=u� �       �	�ߴXc�A�*

lossJa�=4��       �	��Xc�A�*

lossa��=��8[       �	w��Xc�A�*

lossq�=��6       �	�g�Xc�A�*

loss��=h^�{       �	���Xc�A�*

loss6�[=�^��       �	u��Xc�A�*

loss���<z��r       �	j�Xc�A�*

loss�*<[���       �	K �Xc�A�*

loss���=6��L       �	��Xc�A�*

lossk<y��       �	�-�Xc�A�*

lossN<���)       �	v��Xc�A�*

lossط=�J��       �	�\�Xc�A�*

loss�#>���f       �	��Xc�A�*

loss(�x=D@��       �	E��Xc�A�*

loss�w>�4qQ       �	EJ�Xc�A�*

lossx1>�\9�       �	���Xc�A�*

lossT��=
�V�       �	���Xc�A�*

loss���=��       �	h�Xc�A�*

loss�^�=,���       �	L��Xc�A�*

lossM��=eB�       �	�b�Xc�A�*

loss�d�=�r�       �	��Xc�A�*

lossE4&=��Ɩ       �	^��Xc�A�*

loss�U>��       �	DR��Xc�A�*

loss84�<��T       �	���Xc�A�*

loss�)>�t�       �	s��Xc�A�*

loss].4=�%+       �	�,�Xc�A�*

loss,H>&�!�       �	*��Xc�A�*

loss���=�� �       �	Ve�Xc�A�*

loss�-�=��       �	��Xc�A�*

lossN�=Y,�       �	�Xc�A�*

loss�/d={V9�       �	���Xc�A�*

loss)A<�Y}�       �	�S�Xc�A�*

loss�O>-���       �	|��Xc�A�*

loss7%X=sg�       �	����Xc�A�*

loss<W=o0�       �	F%��Xc�A�*

loss_*�=�|��       �	ܻ��Xc�A�*

lossi�i=��R       �	�Y��Xc�A�*

loss��E=��       �	F&��Xc�A�*

loss �J>F��F       �	���Xc�A�*

loss�,�=|�6
       �	����Xc�A�*

lossHr=�kQ       �	à��Xc�A�*

loss��=ެP�       �	�^��Xc�A�*

loss߶=ݒ�       �	����Xc�A�*

loss�&�=��A       �	֎��Xc�A�*

loss��=4<�       �	�"��Xc�A�*

loss��=�.��       �	ö��Xc�A�*

loss��=�Q�       �	{I��Xc�A�*

lossb\=�ow�       �	����Xc�A�*

loss���=�W1       �	���Xc�A�*

loss Cl<> e8       �	H��Xc�A�*

loss��a=���       �	����Xc�A�*

lossz�=�_M;       �	�J �Xc�A�*

loss�8�=u�Y�       �	�� �Xc�A�*

loss��=��S       �	A��Xc�A�*

loss���=�
6       �	��Xc�A�*

loss�P�=jJN�       �	��Xc�A�*

loss28<��=       �	�U�Xc�A�*

loss�b�<Ft       �	���Xc�A�*

lossL=�d=h       �	��Xc�A�*

lossC"c=�%�G       �	�&�Xc�A�*

lossS__=�c�Z       �	���Xc�A�*

loss�; >Z� �       �	�V�Xc�A�*

loss��=t�`�       �	�1�Xc�A�*

loss�`=.#PN       �	�(�Xc�A�*

lossd�1>�W�#       �	���Xc�A�*

lossi.=f��       �	Xt	�Xc�A�*

loss���=��,G       �	M
�Xc�A�*

loss}��=8��       �	ۧ
�Xc�A�*

loss�}�=��       �	4J�Xc�A�*

lossv7=s���       �	�0�Xc�A�*

loss8q�=F�A       �	���Xc�A�*

loss.ѱ='���       �	ٙ�Xc�A�*

loss��u<[�       �	�2�Xc�A�*

loss{��<'��       �	���Xc�A�*

loss�:�=ƓP�       �	*p�Xc�A�*

loss�A�=I�d       �	]�Xc�A�*

loss���<.��       �	R��Xc�A�*

loss�d$=��$�       �	dt�Xc�A�*

lossV�u=��o       �	��Xc�A�*

loss��V=(.��       �	��Xc�A�*

loss�6>w�/       �	�W�Xc�A�*

loss��%>1���       �	���Xc�A�*

loss�H>�f�#       �	U��Xc�A�*

loss�<,=���       �	�Xc�A�*

loss��=u       �	��Xc�A�*

loss\,=��       �	XV�Xc�A�*

loss��=���        �	��Xc�A�*

loss`E=�y��       �	d��Xc�A�*

loss��_=��       �	�+�Xc�A�*

loss��=?�Z�       �	��Xc�A�*

loss��<��P       �	�n�Xc�A�*

loss���=;R�       �	�Xc�A�*

loss_��=��	'       �	ˠ�Xc�A�*

loss4��=מ �       �	9�Xc�A�*

loss��=�d�       �	6��Xc�A�*

loss3�B=�1�       �	]�Xc�A�*

lossq��=��       �	���Xc�A�*

loss���=�>lU       �	���Xc�A�*

loss�H=]D��       �	�-�Xc�A�*

loss4|[=,���       �	��Xc�A�*

loss�s=�
�       �	GX �Xc�A�*

loss&��<�E�       �	4� �Xc�A�*

loss�|
>�}ɻ       �	��!�Xc�A�*

loss� �=u��       �	�"�Xc�A�*

loss��=�v��       �	մ"�Xc�A�*

loss,��< :�3       �	�Q#�Xc�A�*

loss���=2�}       �	�$�Xc�A�*

lossϫ�=�P,       �	d�$�Xc�A�*

loss��R=�R��       �	�S%�Xc�A�*

loss�ܓ=�l\\       �	=�%�Xc�A�*

loss;�=y0�       �	��&�Xc�A�*

loss��e=�{��       �	B''�Xc�A�*

loss�>E-�       �	��'�Xc�A�*

loss�p!<�:�       �	�X(�Xc�A�*

loss���;,�Y�       �	[�(�Xc�A�*

loss��P=c�L       �	^�)�Xc�A�*

loss���=j�       �	2*�Xc�A�*

loss/��<=@1�       �	��*�Xc�A�*

loss>?=-���       �	��+�Xc�A�*

lossד=��5       �	j,�Xc�A�*

loss�%�<��<f       �	�-�Xc�A�*

loss�C=k<��       �	ʣ-�Xc�A�*

loss��_=����       �	�:.�Xc�A�*

loss��=��S�       �	 �.�Xc�A�*

loss�= ���       �	Tt/�Xc�A�*

lossa�=�p�       �	0�Xc�A�*

loss
�">����       �	֨0�Xc�A�*

loss�H^=(�ؿ       �	HR1�Xc�A�*

loss���=-C�D       �	m�1�Xc�A�*

loss���=|�_       �	,�2�Xc�A�*

loss��>O��	       �	�3�Xc�A�*

loss�V>&#�       �	#�3�Xc�A�*

loss�y�=%wTv       �	�T4�Xc�A�*

loss���<��       �	�4�Xc�A�*

loss#�	=b��       �	*�5�Xc�A�*

loss�ɷ=S��       �	-?6�Xc�A�*

loss`�=����       �	n�6�Xc�A�*

lossF�=�fJ       �	j�7�Xc�A�*

loss���="i�{       �	�=8�Xc�A�*

losss�=�Lh       �	�,9�Xc�A�*

lossq[>��z�       �	�:�Xc�A�*

loss���=x�'v       �	��:�Xc�A�*

loss�|>�x�       �	�s;�Xc�A�*

loss;�=Sc�P       �	�4<�Xc�A�*

loss��%>���       �	�=�Xc�A�*

loss�2K=0Ok       �	�=�Xc�A�*

loss�Q[=��H       �	#e>�Xc�A�*

lossA�=�G�       �	j�>�Xc�A�*

loss]~>=Ǖ�       �	��?�Xc�A�*

loss��q<�+�       �	�6@�Xc�A�*

lossT��=өK       �	��@�Xc�A�*

lossVyX=W:	�       �	MeA�Xc�A�*

loss���<��Q>       �	�B�Xc�A�*

loss�{8=׶�v       �	�B�Xc�A�*

loss�r=�s�       �	:C�Xc�A�*

loss��<	�       �	��C�Xc�A�*

loss}Y�<9��Q       �	��D�Xc�A�*

loss�n�=�so       �	�,E�Xc�A�*

loss��>IGդ       �	j�E�Xc�A�*

loss��=g�T       �	�RF�Xc�A�*

losscX>	�<       �	M�F�Xc�A�*

loss�ڬ<�ӡ       �	G�Xc�A�*

lossH'�=�7�!       �	R&H�Xc�A�*

loss�8>��<       �	��H�Xc�A�*

loss��=�ɬ(       �	�~I�Xc�A�*

lossm�V=^�       �	�J�Xc�A�*

loss�X�=�bn�       �	��J�Xc�A�*

lossl8�=A��x       �	)]K�Xc�A�*

loss�&=�c�m       �	L�Xc�A�*

loss���<$��]       �	W�L�Xc�A�*

lossn��=��m�       �	PVM�Xc�A�*

lossL�>���       �	j�M�Xc�A�*

loss��A>|Mk�       �	g�N�Xc�A�*

loss��=���z       �	�;O�Xc�A�*

loss��<�Ԃ       �	��O�Xc�A�*

loss��a=T� 9       �	�P�Xc�A�*

loss�5=�?'G       �	�xQ�Xc�A�*

lossW�P=�/5       �	�2R�Xc�A�*

loss{�<��p�       �	��R�Xc�A�*

loss:��<#��-       �	�jS�Xc�A�*

loss���=�P��       �	�%T�Xc�A�*

loss�׷<;:N       �	{�T�Xc�A�*

loss�7!=P��       �	�]U�Xc�A�*

loss�S=4��U       �	'.V�Xc�A�*

lossi1<�k�       �	%�V�Xc�A�*

loss���=�UCV       �	�aX�Xc�A�*

loss��;�B�       �	C Y�Xc�A�*

loss���=wßy       �	V�Y�Xc�A�*

loss�-:=��J       �	�GZ�Xc�A�*

loss��=+��*       �	��Z�Xc�A�*

loss���=�h       �	��[�Xc�A�*

lossMp=3c�       �	*\�Xc�A�*

lossӫ�<����       �	��\�Xc�A�*

loss�<�zd�       �	�]�Xc�A�*

loss�T=��b�       �	� ^�Xc�A�*

losss1%=/�k�       �	�^�Xc�A�*

lossȂ=>��       �	dy_�Xc�A�*

loss8��>�)       �	�`�Xc�A�*

loss}yH=Og(       �	%�`�Xc�A�*

lossFl=�YX�       �	�Ja�Xc�A�*

loss�0�=�i��       �	�a�Xc�A�*

loss9t�=r�f       �	R~b�Xc�A�*

losss=�An       �	`c�Xc�A�*

loss�Z�<%��}       �	r�c�Xc�A�*

loss$��=ʚ^�       �	Зd�Xc�A�*

loss��<���0       �	�De�Xc�A�*

loss��>��l�       �	��e�Xc�A�*

loss�Bg=w�3�       �	��f�Xc�A�*

loss ŏ=�A       �	_(g�Xc�A�*

loss��<j�EY       �	�g�Xc�A�*

loss��=�K�       �	Z�h�Xc�A�*

lossB"=ӓ�       �	�/i�Xc�A�*

loss�Zu<}�M       �	h�i�Xc�A�*

loss��=����       �	�uj�Xc�A�*

loss3;�<`�h       �	�k�Xc�A�*

loss�n=�h�=       �	��k�Xc�A�*

loss(��=��Vu       �	g�l�Xc�A�*

loss3z=LW�       �	FBm�Xc�A�*

loss69=���       �	\�m�Xc�A�*

loss ӗ=��g       �	��n�Xc�A�*

loss�=~;(V       �	�|o�Xc�A�*

loss���='�{�       �	J$p�Xc�A�*

lossM�=Z�A       �	��p�Xc�A�*

loss���="8��       �	�cq�Xc�A�*

loss�*h=���       �	 r�Xc�A�*

loss�$�=r�"       �	��r�Xc�A�*

loss��=p��V       �	S?s�Xc�A�*

losspn=>sA       �	��s�Xc�A�*

loss�|c=LW�       �		qt�Xc�A�*

loss`�<̘�       �	gu�Xc�A�*

loss_?@=�p|       �	��u�Xc�A�*

loss�a=z׿_       �	�Tv�Xc�A�*

loss3��=s9�I       �	M�v�Xc�A�*

lossv�=���       �	��w�Xc�A�*

lossEwK=b	�       �	9�y�Xc�A�*

loss)��=�[�       �	��z�Xc�A�*

loss;�o=T3��       �	��{�Xc�A�*

loss.��<�lh       �	5|�Xc�A�*

loss��<G��L       �	��|�Xc�A�*

lossqH�<.�g�       �	�q}�Xc�A�*

lossZ�-=2ߔ=       �	�~�Xc�A�*

loss��G=� Ko       �	L�~�Xc�A�*

loss��_=5�g�       �	w��Xc�A�*

loss�q=� �d       �	l&��Xc�A�*

loss��=� �l       �	TȀ�Xc�A�*

loss `�=�i�I       �	�n��Xc�A�*

loss�7�<	Q�       �	���Xc�A�*

lossn�E<��#.       �	Ḃ�Xc�A�*

loss �=��G�       �	�O��Xc�A�*

lossIS�<G�|       �	胵Xc�A�*

lossJ�<�Y�9       �	J~��Xc�A�*

loss*�<�S^�       �	���Xc�A�*

loss�<jS��       �	���Xc�A�*

loss�Q=��       �	)Z��Xc�A�*

lossa:�;b��       �	�3��Xc�A�*

loss���<�h       �	CǇ�Xc�A�*

loss��=ʆv1       �	O\��Xc�A�*

loss���:�U��       �	�Xc�A�*

loss h�<$l_       �	����Xc�A�*

loss�l;/h��       �	����Xc�A�*

lossRT9=���       �	�M��Xc�A�*

loss�D�=%ƲP       �	7⋵Xc�A�*

loss�6A=d$�       �	����Xc�A�*

loss���;�c�       �	-��Xc�A�*

loss�N�=�&�       �	�ƍ�Xc�A�*

loss��]>�Օ       �	Yl��Xc�A�*

lossÅ;��Q       �	���Xc�A�*

loss�{�=�O6�       �	ݖ��Xc�A�*

loss���=ׅ��       �	o-��Xc�A�	*

loss�?>�6�       �	+Ð�Xc�A�	*

loss��f=c�B�       �	<��Xc�A�	*

loss<<=o'�       �	$���Xc�A�	*

loss�=�V��       �	�M��Xc�A�	*

loss,M�=��*       �	 ᓵXc�A�	*

loss�ۛ=+�\�       �	hv��Xc�A�	*

loss��!>�`l�       �	���Xc�A�	*

lossoX6=��       �	����Xc�A�	*

lossZ4�=�o:�       �	.Y��Xc�A�	*

loss}��=O���       �	����Xc�A�	*

lossT�=&e�       �	ޒ��Xc�A�	*

loss70�=cq�,       �	�,��Xc�A�	*

loss1Z�=�b�       �	Ø�Xc�A�	*

loss�p=�J`r       �	����Xc�A�	*

loss���<��wg       �	2��Xc�A�	*

loss�V�=n��       �	�Ě�Xc�A�	*

loss:`�=B	#       �	�W��Xc�A�	*

lossԫ=���!       �	K꛵Xc�A�	*

loss�M_=/��       �	���Xc�A�	*

losss�=u�>       �	I��Xc�A�	*

loss�B�;�]��       �	�ǝ�Xc�A�	*

loss�z=��T(       �	B^��Xc�A�	*

loss ��<ߤ-c       �	����Xc�A�	*

loss��=+��       �	?���Xc�A�	*

loss�=�?qr       �	�#��Xc�A�	*

loss�:A>�aZ�       �	����Xc�A�	*

loss�'�=޴*�       �	HR��Xc�A�	*

lossE��<�}�5       �	t롵Xc�A�	*

loss���="ÿ�       �	����Xc�A�	*

lossf��=� {       �	J`��Xc�A�	*

loss`<y��       �	����Xc�A�	*

loss�S�=)�Az       �	����Xc�A�	*

loss1��<�ߟ       �	�3��Xc�A�	*

loss��^=�2�       �	֥�Xc�A�	*

lossA�=�O�       �	Ow��Xc�A�	*

lossFT=d��       �	��Xc�A�	*

loss�1�=sP��       �	}���Xc�A�	*

loss��%<zFQ[       �	�T��Xc�A�	*

loss�>=5Lh       �	Z�Xc�A�	*

loss�a=�6�       �	����Xc�A�	*

lossd�+=<��       �	�1��Xc�A�	*

loss���<zIx�       �	���Xc�A�	*

lossX�=_�S       �	����Xc�A�	*

loss�Bu=�F��       �	�Q��Xc�A�	*

lossrIc<��6�       �	쬵Xc�A�	*

lossIރ=V}`_       �	F���Xc�A�	*

loss�N�<z�+       �	�?��Xc�A�	*

loss�3�<�me�       �	'���Xc�A�	*

loss���=��{E       �	)�ƵXc�A�	*

loss=��=��Cz       �	�DǵXc�A�	*

loss��=-g�       �	�ǵXc�A�	*

loss�s<�&,       �	\tȵXc�A�	*

loss���=8q,!       �	4ɵXc�A�	*

loss�t=&�]       �	��ɵXc�A�	*

loss:�A=��        �	F�ʵXc�A�	*

loss��=5nR�       �	��˵Xc�A�	*

lossR4>����       �	e4̵Xc�A�	*

loss��=��)�       �	%�̵Xc�A�	*

loss*�=Y���       �	~͵Xc�A�	*

lossl�?=tW5       �	E�εXc�A�	*

loss�>����       �	�>ϵXc�A�	*

loss�W�=]���       �	9�ϵXc�A�	*

lossO�p=+t��       �	I�еXc�A�	*

loss���<65[�       �	�OѵXc�A�	*

loss�<��x�       �	
�ѵXc�A�	*

lossn��==P�q       �	��ҵXc�A�	*

lossx~b=�;�       �	=�ӵXc�A�	*

lossC(
>��G�       �	�xԵXc�A�	*

loss�Z�=0�E`       �	�&յXc�A�	*

loss�>��V       �	�յXc�A�	*

lossW��< �p       �	J�ֵXc�A�	*

loss�k=,��P       �	X�׵Xc�A�	*

lossf�<tJ�       �	�LصXc�A�	*

loss��.<��o       �	D�صXc�A�	*

loss=Qs=�g$l       �	��ٵXc�A�	*

lossf�Q=���       �	�9ڵXc�A�	*

loss(%>�*��       �	3�ڵXc�A�	*

loss�[M=ה�       �	a�۵Xc�A�	*

loss�՝=���E       �	��ܵXc�A�	*

loss�wO<��=�       �	<gݵXc�A�	*

loss|��<��[�       �	(޵Xc�A�	*

loss�2�=��_       �	��޵Xc�A�	*

loss�8r=��       �	rߵXc�A�	*

loss�v�=W��       �	a�Xc�A�	*

loss��;���3       �	���Xc�A�	*

lossrZ�=�e��       �	j�Xc�A�	*

loss�L>+�y       �	$	�Xc�A�	*

loss8�T=�c8�       �	��Xc�A�	*

lossz��=�qߍ       �	ZJ�Xc�A�	*

loss?Y]=� ��       �	`��Xc�A�	*

losstZ�<L��       �	��Xc�A�	*

loss�M�=
�q4       �	^M�Xc�A�	*

loss1H�<V.k2       �	��Xc�A�	*

loss`��=sD�       �	F��Xc�A�	*

loss�'3=3\��       �	�6�Xc�A�	*

loss9=�l       �	c��Xc�A�	*

lossߏ=PƳ�       �	C��Xc�A�	*

lossA�;:��       �	-"�Xc�A�	*

loss{��<@��       �	���Xc�A�	*

loss*��=a!��       �	�s�Xc�A�	*

lossv�<�H       �	h!�Xc�A�	*

loss��N>
�[�       �	���Xc�A�	*

lossm�f=����       �	�x��Xc�A�	*

loss�$�<���       �	� �Xc�A�	*

loss�SI<�C�<       �	��Xc�A�	*

loss�ɓ;Z,vb       �	2q�Xc�A�	*

loss;��<��       �	��Xc�A�	*

loss��=̨�d       �	���Xc�A�	*

loss:U�=��;       �	*X�Xc�A�	*

lossq�B>�ٝ�       �	��Xc�A�	*

loss���<Sp�q       �	@��Xc�A�	*

loss�E�<c��       �	�N�Xc�A�	*

loss�H�=~��       �	���Xc�A�	*

lossr=L�5�       �	����Xc�A�	*

loss��=�`S.       �	�U��Xc�A�	*

loss���<���Q       �	,��Xc�A�	*

loss���=�})t       �	
���Xc�A�	*

loss�:Z=.3�l       �	���Xc�A�	*

lossM��=��+'       �	�\��Xc�A�	*

loss){t=>J��       �	�)��Xc�A�	*

loss"R=c`CH       �	���Xc�A�	*

loss�[=| �       �	�M��Xc�A�	*

loss[o�=���       �	l���Xc�A�	*

lossl��<LM��       �	���Xc�A�	*

loss��N=�g,�       �	I��Xc�A�	*

loss�0=���       �	`���Xc�A�	*

lossD%=�=�~       �	����Xc�A�	*

lossi_j=;0�       �	�!��Xc�A�
*

loss�U_=�        �	���Xc�A�
*

loss��7=���5       �	�Z �Xc�A�
*

loss��=	z!�       �	�� �Xc�A�
*

loss�3=���       �	ߣ�Xc�A�
*

lossd%.=`�u       �	�A�Xc�A�
*

loss��,=��|       �	���Xc�A�
*

loss�=�s�       �	���Xc�A�
*

loss#�<d��2       �	oE�Xc�A�
*

loss��<()l�       �	��Xc�A�
*

loss�l=%�~       �	�{�Xc�A�
*

lossle�=R3�       �	��Xc�A�
*

lossLW�=���       �	��Xc�A�
*

loss�� >`5��       �	�F�Xc�A�
*

loss��=�K�z       �	z��Xc�A�
*

loss?�U=���       �	q��Xc�A�
*

loss�{�<yǟ       �	+	�Xc�A�
*

lossx� >����       �	C�	�Xc�A�
*

lossW��<���       �	Qg
�Xc�A�
*

loss�'�>'S�L       �	,�Xc�A�
*

loss�t�<�]{�       �	v��Xc�A�
*

loss#Bp=!�_       �	�_�Xc�A�
*

loss�/�<=x       �	��Xc�A�
*

loss=�r=T:��       �	)��Xc�A�
*

loss *=/�|�       �	{j�Xc�A�
*

lossw�|=ԢKi       �	�C�Xc�A�
*

loss���=��       �	���Xc�A�
*

loss�C�<l䇘       �	Cu�Xc�A�
*

loss:=|��"       �	��Xc�A�
*

loss:0>鈙�       �	��Xc�A�
*

loss:/=����       �	uX�Xc�A�
*

loss��=�D�       �	��Xc�A�
*

loss#�i=?��       �	���Xc�A�
*

loss��=t3�       �	�I�Xc�A�
*

loss���<���!       �	���Xc�A�
*

lossA��<EN�f       �	L�Xc�A�
*

loss��(=O?r       �	1��Xc�A�
*

loss�C=�5Ո       �	]��Xc�A�
*

loss��=.N�       �	{��Xc�A�
*

lossZV>=���f       �	^+�Xc�A�
*

loss|9�=s�{Y       �	3�Xc�A�
*

loss!��<��       �	�Xc�A�
*

loss�zR=���       �	��Xc�A�
*

loss�>=w�f�       �	�l�Xc�A�
*

loss��=q��       �	��Xc�A�
*

loss-t'<=f�g       �	!��Xc�A�
*

losss:�=��΅       �	�\�Xc�A�
*

loss���=y"��       �	��Xc�A�
*

loss嘳<�h.       �	f- �Xc�A�
*

loss���=է��       �	� �Xc�A�
*

loss�U>>J��:       �	�@"�Xc�A�
*

loss(��=��       �	��"�Xc�A�
*

loss��r=H9-I       �	K�#�Xc�A�
*

loss-�<�Nf       �	O>$�Xc�A�
*

loss?�<�ri+       �	2�$�Xc�A�
*

loss.��=V��H       �	j�%�Xc�A�
*

loss=��<Ɠ��       �	�/&�Xc�A�
*

loss4==M�)       �	��&�Xc�A�
*

loss��<c�b       �	8�'�Xc�A�
*

loss�L<���       �	(�Xc�A�
*

loss!w�<N       �	�(�Xc�A�
*

loss6Tm<C^ܮ       �	rQ)�Xc�A�
*

loss$
�=\�       �	��)�Xc�A�
*

lossHV=Hd       �	�*�Xc�A�
*

losstG�=�\�       �	�M+�Xc�A�
*

loss.&=�2�       �	�],�Xc�A�
*

loss���<�r�       �	7�,�Xc�A�
*

loss8�B=�UC�       �	��-�Xc�A�
*

loss?D<���       �	C.�Xc�A�
*

loss��}<�6       �	��.�Xc�A�
*

loss��D=͕�F       �	:�/�Xc�A�
*

lossfm�<��       �	{30�Xc�A�
*

lossㆎ=�
^       �	c�0�Xc�A�
*

loss�b4=��L       �	�1�Xc�A�
*

loss�X=���{       �	k,2�Xc�A�
*

loss���='*�M       �	�2�Xc�A�
*

loss���;���R       �	@�3�Xc�A�
*

loss���<!x�       �	�>4�Xc�A�
*

loss>S�:�       �	&�4�Xc�A�
*

loss�=��p       �	��5�Xc�A�
*

loss�>L<~q�p       �	Q�6�Xc�A�
*

loss\�=B&��       �	�N7�Xc�A�
*

loss|W�<�Zn%       �	��7�Xc�A�
*

loss1=�<!y&!       �	�8�Xc�A�
*

lossK�<'���       �	ϻ9�Xc�A�
*

loss08=0���       �	�j:�Xc�A�
*

loss/x<���       �	9
;�Xc�A�
*

loss���<�1y       �	z�;�Xc�A�
*

lossv�$=�g�T       �	�D<�Xc�A�
*

lossv�|<����       �	��<�Xc�A�
*

loss���<�y�       �	�w=�Xc�A�
*

lossjhf=U�kK       �	�>�Xc�A�
*

loss?ؓ<���T       �	Ѳ>�Xc�A�
*

loss��F=�1C�       �	]O?�Xc�A�
*

loss�\=vc��       �	��?�Xc�A�
*

loss��;rr��       �	Ԃ@�Xc�A�
*

loss�H7=ni�       �	;A�Xc�A�
*

loss ��=c7�       �	F�A�Xc�A�
*

loss�,f=��a       �	3PB�Xc�A�
*

loss[�c=���M       �	��B�Xc�A�
*

losszX=��88       �	M�C�Xc�A�
*

loss��M<S���       �	:;D�Xc�A�
*

lossA�=���       �	��D�Xc�A�
*

loss��>ʟ��       �	=}E�Xc�A�
*

lossH��<�Qk.       �	�F�Xc�A�
*

loss��Z<]h�F       �	q�F�Xc�A�
*

loss��=�k�       �	�VG�Xc�A�
*

loss�Zw=d`�       �	��H�Xc�A�
*

lossw{4={�       �	|�I�Xc�A�
*

loss 0�=i���       �	�]J�Xc�A�
*

lossH�"=�A�       �	CK�Xc�A�
*

loss��=s!�       �	1�K�Xc�A�
*

loss
�8=ђ�       �	=`L�Xc�A�
*

loss�Sg<_��z       �	h<M�Xc�A�
*

lossq�<�]TL       �	��M�Xc�A�
*

lossVl�<��B       �	9�N�Xc�A�
*

loss��<ќ'�       �	p|O�Xc�A�
*

loss�2.=Wi��       �	�&P�Xc�A�
*

loss��=Љ�B       �	�Q�Xc�A�
*

losszI>��       �	�Q�Xc�A�
*

loss*f<47M?       �	�rR�Xc�A�
*

loss�Pn=}��A       �	�S�Xc�A�
*

loss6�T=$�6M       �	��S�Xc�A�
*

loss �`=�Ձg       �	��T�Xc�A�
*

loss��=�Y\<       �	,EU�Xc�A�
*

lossէ<|ɠ       �	9
V�Xc�A�
*

loss#�<�T�       �	��V�Xc�A�
*

lossh=Հ#p       �	o~W�Xc�A�
*

loss:�>�/�}       �	�X�Xc�A�*

loss��w=H8��       �	F�X�Xc�A�*

loss<dx=dT �       �	�RY�Xc�A�*

loss�-=��:�       �	��Y�Xc�A�*

loss���=�<G�       �	��Z�Xc�A�*

loss�`'<�-,�       �	 [�Xc�A�*

loss�q=�H       �	մ[�Xc�A�*

loss��=[!��       �	Z\�Xc�A�*

lossr�z=x�       �	��\�Xc�A�*

loss�@�=N�}�       �	
�]�Xc�A�*

loss��=V�F�       �	C^�Xc�A�*

loss���=�4�       �	*�^�Xc�A�*

loss�>�ζ       �	�w_�Xc�A�*

loss�ж<��`�       �	�`�Xc�A�*

loss��<��GL       �	1�`�Xc�A�*

loss5�=���       �	1aa�Xc�A�*

loss�VE=��j5       �	�b�Xc�A�*

loss�2�=T���       �	�b�Xc�A�*

lossv)=UZe       �	�Wc�Xc�A�*

lossޖ�=lr+,       �	U�c�Xc�A�*

lossv[a=��[       �	��d�Xc�A�*

loss�`=��f       �	L8e�Xc�A�*

loss�$�==���       �	�e�Xc�A�*

loss�(�<��b\       �	��f�Xc�A�*

loss��_=��fG       �	��g�Xc�A�*

loss=��<bÑ       �	�;h�Xc�A�*

loss��#=�Q�       �	��h�Xc�A�*

loss�"=�#ɤ       �	��i�Xc�A�*

loss��O=V��       �	_bj�Xc�A�*

lossq�X=���       �	�k�Xc�A�*

loss�P�=��       �	z�k�Xc�A�*

loss`��=+:       �	W_l�Xc�A�*

loss�&R< ػ]       �	�m�Xc�A�*

loss])N=wY�'       �	ؼm�Xc�A�*

loss�1|=�hG       �	�on�Xc�A�*

loss��=&l�8       �	no�Xc�A�*

loss�F=SJ�       �	�o�Xc�A�*

lossqX�={�BT       �	^hp�Xc�A�*

loss>s=�F       �	Yq�Xc�A�*

losss�~<7�yO       �	z�q�Xc�A�*

lossw�~=����       �	Djr�Xc�A�*

lossoM�=b�{�       �	�s�Xc�A�*

loss!*=Y�[o       �	|�s�Xc�A�*

loss�Ջ=���_       �	=�t�Xc�A�*

loss��<aK0[       �	+4u�Xc�A�*

lossT�S=�ӑ�       �	*�u�Xc�A�*

lossX��<�-E�       �	Őv�Xc�A�*

loss� <*�       �	7w�Xc�A�*

loss
�W<��s�       �	��w�Xc�A�*

loss��=B�X�       �	�x�Xc�A�*

loss�h{<d�t�       �	bey�Xc�A�*

loss#�d<Q�?{       �	�z�Xc�A�*

loss�>r=�M�q       �	#�z�Xc�A�*

lossl�<=KӁ�       �	�a{�Xc�A�*

lossR�l=oc�       �	x|�Xc�A�*

loss��`=��h�       �	��|�Xc�A�*

loss��=���|       �	�Y}�Xc�A�*

loss�z=ƴ��       �	��}�Xc�A�*

loss\��<#ޯ�       �	c�~�Xc�A�*

lossڠ=@�z�       �	9�Xc�A�*

loss� �<ˇ!�       �	���Xc�A�*

loss�Z=�M�7       �	t��Xc�A�*

loss��=�[�Q       �	���Xc�A�*

loss��<:B��       �	����Xc�A�*

loss}�>=r��       �	XU��Xc�A�*

loss���<K��       �	g���Xc�A�*

loss�'>�3�       �	����Xc�A�*

loss���=Ob�       �	3��Xc�A�*

loss-ל=G�ٿ       �	�ф�Xc�A�*

lossnB(=��f�       �	�j��Xc�A�*

lossGG=W�z       �	� ��Xc�A�*

loss1��=�:J       �	ϡ��Xc�A�*

loss;xJ=���       �	>=��Xc�A�*

lossv4�<R���       �	sՇ�Xc�A�*

loss���=��       �	�l��Xc�A�*

loss���<9O�>       �	���Xc�A�*

loss�]�<[q->       �	褉�Xc�A�*

lossrT=s���       �	?��Xc�A�*

loss���=ZL	       �	f܊�Xc�A�*

loss�>�=n��?       �	r��Xc�A�*

loss{��<����       �	�
��Xc�A�*

loss}��<3��       �	ݳ��Xc�A�*

loss��y<����       �	)Y��Xc�A�*

loss:�z;_9��       �	;���Xc�A�*

loss,�=
s       �	����Xc�A�*

loss,��;óF�       �	�C��Xc�A�*

loss�wX<Q�n�       �	돶Xc�A�*

lossO=z��       �	f���Xc�A�*

loss�n=FɊ�       �	���Xc�A�*

lossRi�<=��       �	wڑ�Xc�A�*

loss`��=N�x�       �	o���Xc�A�*

loss���=����       �	�&��Xc�A�*

lossԚ�=��j       �	�ғ�Xc�A�*

loss�J=��k�       �	��Xc�A�*

loss T1="4��       �	졕�Xc�A�*

lossX��<����       �	�P��Xc�A�*

loss\�>�       �	E���Xc�A�*

lossW �<���B       �	����Xc�A�*

loss_>5���       �	5��Xc�A�*

loss&��;Xc�H       �	CᘶXc�A�*

loss·�=+zd       �	Ӆ��Xc�A�*

loss	�-<�wHX       �	jO��Xc�A�*

loss
@�=s�w       �	����Xc�A�*

loss8s<=}eo       �	I���Xc�A�*

loss�{P=Ȩ��       �	8��Xc�A�*

loss�=����       �	?᜶Xc�A�*

losss�;�I��       �	�v��Xc�A�*

loss-�=�Pn�       �	~��Xc�A�*

losst��=  2�       �	j���Xc�A�*

lossc��<�|�       �	�g��Xc�A�*

loss�80=�Q�.       �	��Xc�A�*

loss��=�,�       �	ȳ��Xc�A�*

loss~�=���       �	�J��Xc�A�*

loss�+=��<       �	�ࡶXc�A�*

loss��=�wq�       �	_y��Xc�A�*

lossϜ=�`�       �	a��Xc�A�*

loss�L�<���       �	ū��Xc�A�*

loss1�v=M�7       �	kF��Xc�A�*

loss:�>i��       �	�㤶Xc�A�*

loss���<����       �	����Xc�A�*

loss��=��       �	���Xc�A�*

loss�j�=K�       �	����Xc�A�*

loss�1*=���       �	댧�Xc�A�*

loss�=8U
$       �	c%��Xc�A�*

loss�|�<�$�       �	�ɨ�Xc�A�*

loss���<{?�c       �	�k��Xc�A�*

lossnk�=o
t�       �	�	��Xc�A�*

loss���=�y�       �	�O��Xc�A�*

loss�=U6�       �	j���Xc�A�*

loss��h=�L��       �	O���Xc�A�*

loss�#=RX]�       �	1��Xc�A�*

loss��=B�|       �	hͭ�Xc�A�*

loss���;�͌)       �	���Xc�A�*

lossO|g=���N       �	2���Xc�A�*

loss
'�<6�       �	nQ��Xc�A�*

loss��=�6�       �	���Xc�A�*

loss)@y=;�@�       �	S���Xc�A�*

loss�R >�Nc       �	t_��Xc�A�*

loss��=���       �	���Xc�A�*

lossU</3�       �	䞳�Xc�A�*

loss�4=���       �	?��Xc�A�*

loss�e^=���5       �	ശXc�A�*

loss���<{���       �	P���Xc�A�*

loss;�Q<��~       �	[(��Xc�A�*

lossn��<�AY       �	�Ͷ�Xc�A�*

loss��i=�&��       �	�g��Xc�A�*

lossG�<�**       �	���Xc�A�*

loss���=��k;       �	����Xc�A�*

loss���<��8U       �	.=��Xc�A�*

loss�n<Cs       �	f۹�Xc�A�*

lossW��<����       �	T���Xc�A�*

loss2J=16^       �	;6��Xc�A�*

loss�==���       �	�ٻ�Xc�A�*

lossqZ�=����       �	�~��Xc�A�*

lossýl<�E��       �	�$��Xc�A�*

loss��=�'i"       �	�½�Xc�A�*

lossO�y=�^7       �	>[��Xc�A�*

loss1>߉9       �	���Xc�A�*

lossLR�= �O        �	s���Xc�A�*

loss�݄=�س�       �	�7��Xc�A�*

loss�9�<���       �	���Xc�A�*

loss �"=���H       �	f���Xc�A�*

loss��<�6�       �	?¶Xc�A�*

loss���<�A       �	�¶Xc�A�*

lossh��<0        �	WöXc�A�*

loss��6=���       �	�ĶXc�A�*

loss>i	=	,��       �	��ĶXc�A�*

lossR=�"       �	-@ŶXc�A�*

loss��=����       �	��ŶXc�A�*

loss��.=�H�       �	�rƶXc�A�*

losstO�<^F��       �	�ǶXc�A�*

loss���=��ox       �	��ǶXc�A�*

loss��
>�F<       �	NȶXc�A�*

loss��C=�Q*       �	��ɶXc�A�*

loss�-�<�k�       �	��ʶXc�A�*

lossT�<���       �	9�˶Xc�A�*

loss��<<2�V       �	NF̶Xc�A�*

lossM'=���       �	K�̶Xc�A�*

loss`m�=L��       �	P�ͶXc�A�*

loss��G=>�G~       �	�#ζXc�A�*

lossCܼ=����       �	�ζXc�A�*

loss�z�;��ب       �	��϶Xc�A�*

loss�s�<eqA       �	�fжXc�A�*

lossqf;=6~!�       �	�ѶXc�A�*

loss�=����       �	�ѶXc�A�*

loss�L<#k�@       �		OҶXc�A�*

loss�oj=@�,#       �	D�ҶXc�A�*

loss�6�<N*��       �	&�ӶXc�A�*

loss�}�=�T�       �	[%ԶXc�A�*

lossA�s<E[��       �	 �ԶXc�A�*

loss���;�_��       �	�hնXc�A�*

loss�eK<6�Lh       �	FֶXc�A�*

lossg�=�t�j       �	�ֶXc�A�*

loss��D<�I
@       �	R`׶Xc�A�*

lossۂZ=��I�       �	|ضXc�A�*

loss���<G���       �	�ضXc�A�*

loss��F<90�       �	�}ٶXc�A�*

losso1=���       �	ڶXc�A�*

lossFC:<�;s       �	�ڶXc�A�*

loss/a�=P��       �	)Y۶Xc�A�*

lossț^=���       �	�`ܶXc�A�*

lossL�=#���       �	{�ܶXc�A�*

loss`V"=���"       �	I�ݶXc�A�*

loss���<����       �	�@޶Xc�A�*

loss��=*U��       �	L�޶Xc�A�*

loss6�P=�8+�       �	��߶Xc�A�*

loss��=��2�       �	�*�Xc�A�*

loss:�=�h�T       �	���Xc�A�*

lossZ�=�/��       �	�f�Xc�A�*

loss�@<A���       �	�
�Xc�A�*

loss���;G���       �	��Xc�A�*

loss6�=ݎ>�       �	)>�Xc�A�*

loss�=|sC�       �	#��Xc�A�*

loss�"0=2�5�       �	�m�Xc�A�*

loss͐�<��       �	��Xc�A�*

loss\	=$߉_       �	��Xc�A�*

loss:��=IW�N       �	T5�Xc�A�*

lossT�4=���N       �	@��Xc�A�*

loss���={��V       �	C��Xc�A�*

loss�Z=n_{�       �	:?�Xc�A�*

loss{�=�I       �	>��Xc�A�*

loss��=L���       �	,��Xc�A�*

loss�DV<D��f       �	�F�Xc�A�*

loss���=��A�       �	���Xc�A�*

loss�?�<��7:       �	֋�Xc�A�*

loss���<$�k�       �	�1�Xc�A�*

lossx�=��[�       �	���Xc�A�*

lossnJ�<.��$       �	o���Xc�A�*

loss��=?��       �	�1�Xc�A�*

lossQ�w=U��       �	A��Xc�A�*

loss�<���       �	�z�Xc�A�*

loss�us<O���       �	�"�Xc�A�*

loss�O�<�p�       �	���Xc�A�*

lossN�\=��       �	 }�Xc�A�*

loss}x%>��p�       �	V)�Xc�A�*

loss�߅=�'E�       �	���Xc�A�*

loss�x?=W��       �	�i�Xc�A�*

loss�sB=3�%�       �	���Xc�A�*

loss��I=]R��       �	����Xc�A�*

loss��=�w/�       �	II��Xc�A�*

lossԍ�<����       �	-���Xc�A�*

losszA�<d�p�       �	����Xc�A�*

losss=I��       �	�#��Xc�A�*

loss F�<���       �	b���Xc�A�*

loss��<���       �	�\��Xc�A�*

lossL(0<MA�'       �	���Xc�A�*

loss���=���       �	h���Xc�A�*

loss8y�=b�m>       �	
���Xc�A�*

lossf��=��?       �	�w��Xc�A�*

loss�A�=�m��       �	A��Xc�A�*

loss\��<#�n       �	���Xc�A�*

loss�Z=���       �	p���Xc�A�*

lossA�c=ا$       �	���Xc�A�*

loss�
�<�3�       �	?���Xc�A�*

loss��<� $4       �	^ �Xc�A�*

lossA��;��v4       �	� �Xc�A�*

loss�7=G��       �	q��Xc�A�*

loss�ޏ<�i��       �	�'�Xc�A�*

loss��F=��       �	�;�Xc�A�*

loss��s=0]t�       �	���Xc�A�*

loss(�<��.       �	C��Xc�A�*

loss�;_<���       �	��Xc�A�*

loss&�<�\6�       �	�;�Xc�A�*

lossvu�=��bm       �	^K�Xc�A�*

loss\ք<��)�       �	�=	�Xc�A�*

lossh��<KPpn       �	;S
�Xc�A�*

lossR��=�^;�       �	��
�Xc�A�*

loss�$�= ��       �	n��Xc�A�*

loss*=M��:       �	O��Xc�A�*

lossӢ<R2�       �	j��Xc�A�*

loss��<x��       �	V��Xc�A�*

loss��<$��{       �	*R�Xc�A�*

loss�HH>v       �	�^�Xc�A�*

loss$�=� x{       �	��Xc�A�*

lossp#=u��(       �	��Xc�A�*

loss[��<g�v        �	>�Xc�A�*

loss�rP=�L�       �		��Xc�A�*

lossh.`=�xsy       �	�z�Xc�A�*

loss�R�<4�6       �	4�Xc�A�*

loss�;K�X�       �	��Xc�A�*

loss}ey=H
�       �	İ�Xc�A�*

loss��<R��f       �	�L�Xc�A�*

lossn�=n��       �	�)�Xc�A�*

loss�=-6�       �	���Xc�A�*

lossd�	=���`       �	iq�Xc�A�*

lossʶ{<�s�       �	��Xc�A�*

lossܽN=�n�       �	���Xc�A�*

loss�%�<H�h�       �	"P�Xc�A�*

lossq�<o1x       �	W��Xc�A�*

loss �=����       �	@��Xc�A�*

losseDr<�*��       �	�#�Xc�A�*

lossʬ'=�o       �	˺�Xc�A�*

lossF��=�%�       �		��Xc�A�*

lossr6 ==��       �	h�Xc�A�*

loss|��<׏�        �	��Xc�A�*

loss8�=�]��       �	�Q�Xc�A�*

loss�'=����       �	}��Xc�A�*

lossW�=�(3�       �	h� �Xc�A�*

loss8�<�wa@       �	�:!�Xc�A�*

loss�ہ=8<�       �	F�!�Xc�A�*

loss5`=��WQ       �	�w"�Xc�A�*

loss3]>W솲       �	�#�Xc�A�*

loss�
N=��       �	�#�Xc�A�*

loss�<�єa       �	�W$�Xc�A�*

loss��<Lɹ       �	R�$�Xc�A�*

loss�;=&k��       �	.�%�Xc�A�*

lossU�=hG�W       �	�%&�Xc�A�*

loss���<�6	K       �	�&�Xc�A�*

loss/�k=.)C       �	:W'�Xc�A�*

loss��d=L��C       �	��'�Xc�A�*

loss�X=2�       �	͔(�Xc�A�*

loss�v<;��       �	|*)�Xc�A�*

loss���=D
u�       �	�)�Xc�A�*

loss�2�<���       �	�r*�Xc�A�*

loss�q!=��i       �	=+�Xc�A�*

loss�H�<�#K�       �	u�+�Xc�A�*

loss�=�=]�       �	�P,�Xc�A�*

loss��<�6��       �	��,�Xc�A�*

lossVek=��_�       �	��-�Xc�A�*

losse�K=�O��       �	i8.�Xc�A�*

loss-4^=*T��       �	��.�Xc�A�*

loss(��=���       �	�{/�Xc�A�*

loss��=a��       �	�0�Xc�A�*

loss��\<	KK       �	��0�Xc�A�*

loss���=Y�G�       �	�l1�Xc�A�*

loss*�W<�Q��       �	2�Xc�A�*

lossj�!<O\�       �	c�2�Xc�A�*

loss$;�;+12�       �	 ]3�Xc�A�*

loss�1G<�@�'       �	�4�Xc�A�*

loss�I�<�U�       �	��4�Xc�A�*

loss�;<�R�`       �	�m5�Xc�A�*

loss$X�<�bc�       �	46�Xc�A�*

loss(��<��]S       �	D�6�Xc�A�*

loss���:���'       �	_^7�Xc�A�*

lossQ!;m�9n       �	�8�Xc�A�*

loss�9^:����       �	��8�Xc�A�*

loss\��;�'�       �	�D9�Xc�A�*

loss͇w=T�U�       �	��9�Xc�A�*

loss�0=<3 =       �	�:�Xc�A�*

loss���:
���       �	]�;�Xc�A�*

loss�N<*��<       �	�B<�Xc�A�*

loss�Ao>'&       �	��<�Xc�A�*

loss�/�;��ԡ       �	Ƥ=�Xc�A�*

loss�F=poo�       �	�T>�Xc�A�*

loss��F<��/3       �	f�>�Xc�A�*

loss�|=veۺ       �	9�?�Xc�A�*

loss�V�=�-H       �	�K@�Xc�A�*

lossE5�<��F       �	h�@�Xc�A�*

loss���=~�@�       �	��A�Xc�A�*

loss1V�=�
N       �	P9B�Xc�A�*

lossTK.=��?$       �	ѐC�Xc�A�*

loss!��=*��       �	�9D�Xc�A�*

loss��f=�)�       �	H�D�Xc�A�*

loss�y=³��       �	�E�Xc�A�*

loss�5'=�p��       �	�0F�Xc�A�*

loss��8=3��       �	$�F�Xc�A�*

loss�=��q�       �	�jG�Xc�A�*

lossn5=���X       �	�H�Xc�A�*

loss=�� �       �	2�H�Xc�A�*

loss�O�<R�8       �	\I�Xc�A�*

lossqܜ=m�XP       �	�J�Xc�A�*

lossy��<Bư�       �	�J�Xc�A�*

lossB<K6��       �	�jK�Xc�A�*

loss�Z�=�ݧ\       �	�L�Xc�A�*

loss�X5=��R'       �	a�L�Xc�A�*

loss��=:���       �	;UM�Xc�A�*

lossE��;`�       �	>N�Xc�A�*

losszfS<��&       �	�N�Xc�A�*

loss��=� v�       �	�3O�Xc�A�*

lossM7�<I�J       �	x�O�Xc�A�*

loss� �="L�       �	��P�Xc�A�*

losse�i=m�9       �	�FQ�Xc�A�*

lossZ9Q=�`]�       �	�Q�Xc�A�*

loss��J=c%��       �	��R�Xc�A�*

loss�e<_M�       �	�1S�Xc�A�*

loss&��;��G�       �	g�S�Xc�A�*

loss�Q�<>��       �	�}T�Xc�A�*

lossOt/<2,`       �	&U�Xc�A�*

loss`�<�ǥ       �	Q�U�Xc�A�*

loss(��<_&5n       �	�_V�Xc�A�*

lossNr�<���       �	1W�Xc�A�*

loss��;=��&�       �	��W�Xc�A�*

loss- �;&���       �	9FX�Xc�A�*

loss�^�<��6�       �	��X�Xc�A�*

losss��<�k[       �	�Y�Xc�A�*

lossS=��l       �	� Z�Xc�A�*

loss��<��g       �	Q�Z�Xc�A�*

lossE��=�eK$       �	i�[�Xc�A�*

lossm�=jN=�       �	�D\�Xc�A�*

loss��<�,�       �	~�\�Xc�A�*

loss�;#="�K       �	5z]�Xc�A�*

lossϾ�;m�       �	�^�Xc�A�*

loss6_<���       �	��^�Xc�A�*

lossoh
=As       �	Q-u�Xc�A�*

loss�=�=d'6�       �	s�u�Xc�A�*

loss%��="pNH       �	�v�Xc�A�*

loss%�>	�i       �	�Aw�Xc�A�*

loss�<��"       �	�w�Xc�A�*

loss��c<�2��       �	f�x�Xc�A�*

loss\�<�i��       �	�Uy�Xc�A�*

loss�J�<&D��       �	H�y�Xc�A�*

loss�E>Rn٤       �	�z�Xc�A�*

loss�r�<	��L       �	��{�Xc�A�*

loss�<d.       �	�}�Xc�A�*

loss��t=���       �	�<~�Xc�A�*

loss͂g<��       �	R�~�Xc�A�*

loss�w�=`�ү       �		��Xc�A�*

loss��?=��<       �	�E��Xc�A�*

loss�|p=!ҡ�       �	(�Xc�A�*

loss�;\;w�&%       �	����Xc�A�*

loss�u�<)�/       �	j���Xc�A�*

loss�w=��c       �	�3��Xc�A�*

lossj��=��f�       �	9`��Xc�A�*

loss.�2=�[NF       �	���Xc�A�*

lossŊ�=��;c       �	h���Xc�A�*

loss���<�~c
       �	!Z��Xc�A�*

loss��4=p��       �	5��Xc�A�*

loss�wE=�{.�       �	󮈷Xc�A�*

loss��<�m8       �	2W��Xc�A�*

loss���=�5�1       �	j���Xc�A�*

lossw�	<9�)       �	����Xc�A�*

lossa��<��"�       �	�K��Xc�A�*

loss]�=#��       �	�鋷Xc�A�*

loss�R/=6TO�       �	����Xc�A�*

loss[�<��2B       �	s0��Xc�A�*

loss� =b�C�       �	 ɍ�Xc�A�*

loss�6=�fB<       �	�h��Xc�A�*

lossۣ�<�!_�       �	���Xc�A�*

loss�P�=�J^�       �	`��Xc�A�*

loss2��<9���       �	��Xc�A�*

loss�A=�i�K       �	����Xc�A�*

lossD�=>%v^       �	�O��Xc�A�*

lossJ>K=4�"a       �	�Xc�A�*

loss�d�=,8»       �	����Xc�A�*

lossS�X=x��m       �	�A��Xc�A�*

loss���<+�/�       �	D���Xc�A�*

lossZ�=���\       �	����Xc�A�*

loss�Q=�B�v       �	zn��Xc�A�*

loss��d=ј�       �	5%��Xc�A�*

loss�B*=أ\�       �	�ԗ�Xc�A�*

loss���<� ɜ       �	&p��Xc�A�*

lossE�6<V�       �	���Xc�A�*

loss@�7;H�l       �	H���Xc�A�*

loss QF<�e1�       �	/���Xc�A�*

loss�+0=n�/       �	@Q��Xc�A�*

loss
�&<�(&�       �	T��Xc�A�*

lossA�M>lj�N       �	����Xc�A�*

loss���=��q        �	iU��Xc�A�*

loss��0<e@\       �	�`��Xc�A�*

lossR�[</���       �	Ih��Xc�A�*

losszk�;,���       �	���Xc�A�*

loss��;4Q�       �	I���Xc�A�*

lossx�P=\�       �	�c��Xc�A�*

loss;��=mU��       �	�	��Xc�A�*

loss�'�<��N�       �	����Xc�A�*

loss��<����       �	F��Xc�A�*

loss�D=k��z       �	�㣷Xc�A�*

loss�k�=`�v�       �	؁��Xc�A�*

lossƼK<A �       �	�)��Xc�A�*

lossX��<��       �	�ƥ�Xc�A�*

loss��=�:��       �	b��Xc�A�*

loss���=-e�       �	
��Xc�A�*

losso2"=��J�       �	����Xc�A�*

loss�h�<^kq}       �	�J��Xc�A�*

lossJ�S=�       �	���Xc�A�*

loss�,�<~���       �	j���Xc�A�*

loss���<��       �	#-��Xc�A�*

loss��<�Zu;       �	2ʪ�Xc�A�*

lossh��<nF�       �	�{��Xc�A�*

loss(`�<u�u       �	���Xc�A�*

loss��="�Q�       �	转�Xc�A�*

loss{`�<%�ۈ       �	�\��Xc�A�*

lossI��=RH��       �	���Xc�A�*

loss#<�<1h�       �	&���Xc�A�*

loss�Ac=�*h�       �	S��Xc�A�*

lossl�F=�m^�       �	O��Xc�A�*

loss���<BL�
       �	���Xc�A�*

loss�=�@�=       �	3R��Xc�A�*

loss3��<(��i       �	����Xc�A�*

loss(<V�uo       �	[���Xc�A�*

loss|� =��yR       �	Q��Xc�A�*

loss��}=X��q       �	��Xc�A�*

loss*g=4�c�       �	Ք��Xc�A�*

loss� �=�;�S       �	�:��Xc�A�*

loss��&=0[�       �	���Xc�A�*

loss�ȇ= l�	       �	���Xc�A�*

loss8�b=�E&       �	-Z��Xc�A�*

loss��C=����       �	���Xc�A�*

lossV�;j\�       �	;���Xc�A�*

loss�\q<L<	�       �	S��Xc�A�*

loss��<�SL�       �	6��Xc�A�*

loss��=�Y��       �	���Xc�A�*

lossh��<��f�       �	����Xc�A�*

lossP�<�       �	^i��Xc�A�*

lossC<�;)e�       �	$��Xc�A�*

loss8=��       �	����Xc�A�*

loss�<$l�       �	�T��Xc�A�*

loss��)=����       �	T���Xc�A�*

loss�߃=�$#       �	���Xc�A�*

loss�;�_Y�       �	mS��Xc�A�*

lossH%�;�3�       �	� ��Xc�A�*

loss��=x�4       �	����Xc�A�*

loss4��<8��       �	)^·Xc�A�*

loss$�=>xs�       �	�÷Xc�A�*

loss1�;=��       �	�÷Xc�A�*

lossVy/=���       �	�UķXc�A�*

lossN�<��&       �	(ŷXc�A�*

lossB7=j�U       �	��ŷXc�A�*

loss���;B>s       �	eƷXc�A�*

loss��z="L��       �	�ǷXc�A�*

loss�" =�K_       �	o�ǷXc�A�*

loss��U=[5       �	V�ȷXc�A�*

loss-�r=��W�       �	�JɷXc�A�*

loss���<�R	       �	��ɷXc�A�*

loss�e�<�=�       �	��˷Xc�A�*

loss8�<��        �	m�̷Xc�A�*

loss��<=���       �	 6ͷXc�A�*

loss)d`<�,�       �	��ͷXc�A�*

loss�#=��       �	��ηXc�A�*

loss���=,�<K       �	�lϷXc�A�*

loss|8
<+��       �	5зXc�A�*

loss��=[�<       �	��зXc�A�*

loss�=g�^       �	BxѷXc�A�*

lossh=e�W7       �	� ҷXc�A�*

loss>
�=!ۼ�       �	��ҷXc�A�*

lossVeH=��q       �	�}ӷXc�A�*

loss�E�<#�:�       �	�'ԷXc�A�*

lossdh=D�2       �	x�ԷXc�A�*

loss���<g�p�       �	�
ַXc�A�*

lossH*==��       �	~�ַXc�A�*

loss:w<�ɉ       �	�W׷Xc�A�*

loss���<w�Z�       �	CطXc�A�*

loss��<���       �	��طXc�A�*

loss��<{<�       �	;ٷXc�A�*

loss7k=��t�       �	?�ڷXc�A�*

loss�M=� w       �	�7۷Xc�A�*

lossW��=�l�c       �	��۷Xc�A�*

loss�Q$=&��       �	�oݷXc�A�*

loss}U�<(��D       �	?;޷Xc�A�*

loss���=���       �	߷Xc�A�*

lossCM)<|�&e       �	ͱ߷Xc�A�*

loss�	�<����       �	og�Xc�A�*

lossȮ�<ƞ��       �	C��Xc�A�*

loss��<�.l�       �	��Xc�A�*

lossL�=z��/       �	�T�Xc�A�*

loss��Y=N7        �	n��Xc�A�*

loss��|=h̍�       �	ߧ�Xc�A�*

loss�J=����       �	�D�Xc�A�*

loss�9�;�;Ei       �	v��Xc�A�*

loss�"<�B:q       �	���Xc�A�*

lossA=�1�       �	H3�Xc�A�*

lossࡸ=�ﾅ       �	r��Xc�A�*

loss$T�<Σ\�       �	 z�Xc�A�*

loss)w5=!��       �	p"�Xc�A�*

loss�p=߲��       �	���Xc�A�*

loss)6�<'I΃       �	���Xc�A�*

lossTQ(;^|��       �	���Xc�A�*

lossZ�<�t�       �	E.�Xc�A�*

lossw��<	-<�       �	� �Xc�A�*

lossI��<a�       �	���Xc�A�*

loss./=�oH       �	;R��Xc�A�*

loss(XD= ���       �	 �Xc�A�*

lossn��<�qW       �	���Xc�A�*

lossE:=��
?       �	���Xc�A�*

loss6K�;}�D�       �	�(�Xc�A�*

loss�=1���       �	���Xc�A�*

lossm�<�(2a       �	�}�Xc�A�*

loss,�q;�Y9�       �	G�Xc�A�*

lossqLA=_       �	Ժ�Xc�A�*

loss��T=�M\0       �	��Xc�A�*

loss�� =y<[�       �	�-��Xc�A�*

loss�e'=� C�       �	����Xc�A�*

lossv=���t       �	\��Xc�A�*

loss؈�:1A��       �	����Xc�A�*

loss;U_=��       �	"���Xc�A�*

loss�d<3
q�       �	�,��Xc�A�*

loss`�<��N       �	�F��Xc�A�*

loss���;`(�       �	!���Xc�A�*

lossX�;I��$       �	c{��Xc�A�*

loss�=i�E       �		��Xc�A�*

losse�<�U�       �	����Xc�A�*

loss|�=D��        �	mW��Xc�A�*

loss-�=�L�       �	�I��Xc�A�*

loss�_�=�6�       �	����Xc�A�*

loss=�<�gE�       �	A���Xc�A�*

loss��"<�9��       �	 ��Xc�A�*

loss���<��W�       �	����Xc�A�*

loss�t�;��       �	hZ��Xc�A�*

loss�C =ږ��       �	; �Xc�A�*

loss�tw=$Z��       �	�� �Xc�A�*

loss��=M�P       �	�z�Xc�A�*

loss�n�=$��       �	��Xc�A�*

loss���<L��m       �	���Xc�A�*

loss���<p��       �	�\�Xc�A�*

loss��o<ӂ»       �	���Xc�A�*

loss�<c<��8D       �	 ��Xc�A�*

loss�<�(us       �	J'�Xc�A�*

lossy��<H�:�       �	���Xc�A�*

loss�=m�gc       �	Φ�Xc�A�*

lossد�=j"�       �	�>�Xc�A�*

loss�=�J�       �	>"�Xc�A�*

loss63[<��~       �	@��Xc�A�*

loss,��=(ha�       �	dW	�Xc�A�*

loss�<[i�       �	+�	�Xc�A�*

loss*�e=�B�@       �	ڐ
�Xc�A�*

lossJ��<��       �	g(�Xc�A�*

loss�3<��       �	h��Xc�A�*

loss��=���       �	wh�Xc�A�*

loss���=tL��       �	��Xc�A�*

loss\4w=#���       �	S��Xc�A�*

loss!`>����       �	6X�Xc�A�*

loss�w=�p       �	���Xc�A�*

loss>��<�
�       �	��Xc�A�*

loss|�<���       �	&5�Xc�A�*

lossj�0=<�"�       �	9��Xc�A�*

loss�"�<�k�       �	]p�Xc�A�*

loss�=�=ȯ��       �	A�Xc�A�*

lossJ=#^sP       �	��Xc�A�*

loss��=��,       �	0I�Xc�A�*

loss�C=�1       �	��Xc�A�*

lossc��<�C��       �	�|�Xc�A�*

lossF�#=K��       �	� �Xc�A�*

loss)�<�4�       �	��Xc�A�*

lossr�<�[       �	(��Xc�A�*

loss�sg=5��       �	��Xc�A�*

loss��K=�e�:       �	�!�Xc�A�*

lossϋX<�ֆ       �	���Xc�A�*

loss�-=#��       �	[�Xc�A�*

lossؘ�<�Uz�       �	�
�Xc�A�*

loss�k=[~h&       �	���Xc�A�*

loss�j=@e�       �	�y�Xc�A�*

loss%�e=���       �	��Xc�A�*

loss�܌<�w�_       �	]��Xc�A�*

loss�,<�V�@       �	NC�Xc�A�*

loss��<3�
�       �	���Xc�A�*

losst��=���       �	}v�Xc�A�*

loss�6=s�       �	��Xc�A�*

lossQy�=����       �	��Xc�A�*

loss��=b�aP       �	�@ �Xc�A�*

loss�ן<{z^       �	w� �Xc�A�*

loss"�<$[�       �	��!�Xc�A�*

losss�F=��       �	~"�Xc�A�*

loss���<F��6       �	��"�Xc�A�*

lossI�$=��       �	R#�Xc�A�*

loss��O=��
�       �	�#�Xc�A�*

loss3E@=�s1n       �	�$�Xc�A�*

loss�!<��_       �	�%�Xc�A�*

loss,_�<M�x~       �	��%�Xc�A�*

loss�`�<f,�@       �	O&�Xc�A�*

lossqm�<�d�t       �	{�&�Xc�A�*

lossf�r<߸       �	O�'�Xc�A�*

loss��<�  [       �	4(�Xc�A�*

lossev=��L�       �	��(�Xc�A�*

loss��<�l�r       �	f�)�Xc�A�*

loss���<��         �	#,*�Xc�A�*

lossu��=9u�c       �	�*�Xc�A�*

loss�\�=��"U       �	 u+�Xc�A�*

loss@uG= �B       �	�,�Xc�A�*

loss.�F=�Y�       �	Q�,�Xc�A�*

lossV֭=�`.       �	�U-�Xc�A�*

lossO��<��       �	��-�Xc�A�*

loss]�o=e@J�       �	7�.�Xc�A�*

loss���=F $       �	\!/�Xc�A�*

loss�*0<7�V�       �	��/�Xc�A�*

loss~�<��       �	�T0�Xc�A�*

loss{\�<T���       �	�1�Xc�A�*

loss6��<����       �	��1�Xc�A�*

loss���=-L"C       �	�Q2�Xc�A�*

lossi��={Q�       �	�2�Xc�A�*

loss/z<��/�       �	~�3�Xc�A�*

loss�e;�ߦ�       �	*4�Xc�A�*

loss́=�n�       �	N�4�Xc�A�*

lossFo6=+���       �	��5�Xc�A�*

loss.�=7�(       �	U36�Xc�A�*

loss��=�ֳ�       �	��6�Xc�A�*

lossn]q<_l��       �	^f7�Xc�A�*

loss�W=���h       �	� 8�Xc�A�*

loss?��<��F�       �	9�8�Xc�A�*

loss-k�=xWt+       �	U29�Xc�A�*

loss�#4=�qp       �	G�9�Xc�A�*

loss��9=�v�       �	(b:�Xc�A�*

lossk�=P���       �	��:�Xc�A�*

loss��;�K�/       �	K�;�Xc�A�*

loss���; $�       �	o+<�Xc�A�*

loss"=�=\J�       �	�<�Xc�A�*

lossW�;;�X�       �	�^=�Xc�A�*

lossdu�<��3r       �	]�=�Xc�A�*

loss��<��c�       �	��>�Xc�A�*

loss��q=)���       �	�&?�Xc�A�*

lossgM=���       �	��?�Xc�A�*

loss�d�=)��       �	B]@�Xc�A�*

loss�=ȩ�~       �	�	A�Xc�A�*

loss�ƫ=d��       �		�A�Xc�A�*

loss���<��G�       �	hAB�Xc�A�*

loss`3�=�W>       �	��B�Xc�A�*

lossC�+=P,V�       �	ڎC�Xc�A�*

lossa|�=��;_       �	9(D�Xc�A�*

loss[��<�9�x       �	��D�Xc�A�*

lossv�&>|���       �	��E�Xc�A�*

loss���;��Y       �	~F�Xc�A�*

lossn�4=y��'       �	дF�Xc�A�*

loss$�"=����       �	NG�Xc�A�*

loss�g�<���3       �	H�Xc�A�*

loss���<��;       �	X�H�Xc�A�*

loss��s=�� �       �	�PI�Xc�A�*

loss[nO=?p)       �	{�I�Xc�A�*

loss��<��       �	�J�Xc�A�*

loss3�<�K       �	 `K�Xc�A�*

loss	o�=_NA<       �	�L�Xc�A�*

loss��a<�Q��       �	�L�Xc�A�*

lossl�%=1�PL       �	�FM�Xc�A�*

loss���<�,'$       �	z�M�Xc�A�*

loss�<H��       �	�{N�Xc�A�*

loss�i<ך9�       �	�O�Xc�A�*

loss�?�=�|b       �	�O�Xc�A�*

loss�3=ȱ��       �	�nP�Xc�A�*

loss�	-={�p       �	�Q�Xc�A�*

loss�v�<L�]       �	��Q�Xc�A�*

loss�lq=���       �	
JR�Xc�A�*

loss~==f�4|       �	��R�Xc�A�*

loss��A=xW�6       �	ׇS�Xc�A�*

loss�3'=���}       �	q!T�Xc�A�*

loss���=89�       �	a�T�Xc�A�*

lossQ��<J���       �	�aU�Xc�A�*

loss �=��Pr       �	Q�U�Xc�A�*

loss8=Dt�Z       �	�V�Xc�A�*

loss�@<�&�       �	�AW�Xc�A�*

loss��=Q�bS       �	��W�Xc�A�*

losse�S=���       �	M�X�Xc�A�*

lossT=(
=L       �	�"Y�Xc�A�*

loss�99=��@�       �	�Y�Xc�A�*

loss�5=���       �	�`Z�Xc�A�*

loss��<rȠ/       �	�
[�Xc�A�*

loss�"<���       �	e�[�Xc�A�*

lossn��<IA��       �	�=\�Xc�A�*

loss��=��(�       �	��\�Xc�A�*

losseo�<#T       �	�m]�Xc�A�*

lossN�=Y���       �	_^�Xc�A�*

loss�J�=��       �	#�^�Xc�A�*

lossAgi<��       �	h<_�Xc�A�*

loss�?�=`
ߔ       �	��_�Xc�A�*

loss��<|'BC       �	�g`�Xc�A�*

loss��<&�r       �	�a�Xc�A�*

lossVX<�'�c       �	
�a�Xc�A�*

loss,�=r�`       �	�Ab�Xc�A�*

loss�J�=]P,�       �	�b�Xc�A�*

loss�^�;��[       �	��c�Xc�A�*

loss�\�=c�       �	o+d�Xc�A�*

lossli�<ؚ8�       �	!�d�Xc�A�*

loss�=4	Y�       �	�|e�Xc�A�*

loss�^�<�R6
       �	^Hf�Xc�A�*

loss�1=̈́'�       �	��f�Xc�A�*

loss��=bꕷ       �	�g�Xc�A�*

loss��=����       �	�"h�Xc�A�*

loss��=�e��       �	8�h�Xc�A�*

loss�g�=?���       �	�Yi�Xc�A�*

loss���=��0&       �	��i�Xc�A�*

loss�U0=�*��       �	�j�Xc�A�*

lossF<�=Q0Z       �	�)k�Xc�A�*

lossMY=_B�(       �	��k�Xc�A�*

lossZ�<�Q�       �	mYl�Xc�A�*

loss��F=�Az       �	m�Xc�A�*

lossȮ�<	       �	��m�Xc�A�*

lossV�=��       �	�Rn�Xc�A�*

lossm�B=J��       �	�n�Xc�A�*

lossZ�%=h ��       �	w�o�Xc�A�*

lossZP�=�YT       �	�1p�Xc�A�*

lossy2�<�/��       �	9dq�Xc�A�*

lossM�==�F       �	i�q�Xc�A�*

loss*gB=�$�       �	��r�Xc�A�*

lossR�@<�S�       �	V+s�Xc�A�*

loss7:�<�{       �	-�s�Xc�A�*

loss�Y=@���       �	;pt�Xc�A�*

loss�V< Z:       �	xu�Xc�A�*

losse�<CE��       �	�u�Xc�A�*

lossh�<�m��       �	(Iv�Xc�A�*

lossi��;A�t�       �	\�v�Xc�A�*

loss���;��_R       �	��w�Xc�A�*

loss:��=q��       �	vx�Xc�A�*

lossc\ =>�        �	Ͱx�Xc�A�*

loss=�=���)       �	0Hy�Xc�A�*

loss�#<�N�       �	u�y�Xc�A�*

loss��<�a       �	$~z�Xc�A�*

lossȭ�<�֥       �	'{�Xc�A�*

loss+��<|V�       �	'�{�Xc�A�*

loss�F�<ڥ
�       �	�|�Xc�A�*

loss��=��       �	H1}�Xc�A�*

loss�j2=� .       �	L�}�Xc�A�*

loss�g>�\"       �	q�Xc�A�*

lossjI=�l�       �	8��Xc�A�*

lossMs�<�ք       �	�c��Xc�A�*

loss��<>rO�       �	����Xc�A�*

loss�V<����       �	���Xc�A�*

loss�T�<*�O�       �	�H��Xc�A�*

loss�d<[�f       �	|���Xc�A�*

loss�v=<��       �	2���Xc�A�*

loss.�<WN       �	�:��Xc�A�*

loss�Ф=��wK       �	�Մ�Xc�A�*

loss���<��3�       �	����Xc�A�*

loss��<w�k�       �	�d��Xc�A�*

loss}<=Oq&       �		���Xc�A�*

loss�ł<k�       �	����Xc�A�*

loss�=v�7�       �	�.��Xc�A�*

lossh�U=��       �	�ʈ�Xc�A�*

loss\�=aS�l       �	�c��Xc�A�*

loss��h<�&0�       �	 ��Xc�A�*

loss��$=�E�Y       �	ᛊ�Xc�A�*

loss���=N�`       �	a4��Xc�A�*

lossw��=����       �	�̋�Xc�A�*

loss n=|QFK       �	_b��Xc�A�*

lossY<�j�       �	@���Xc�A�*

loss�|=(DBK       �	����Xc�A�*

lossnj_=����       �	e8��Xc�A�*

loss�j�<a�x       �	�ώ�Xc�A�*

loss���=`�?       �	�f��Xc�A�*

loss��f<�kyZ       �	����Xc�A�*

loss��">�R�       �	����Xc�A�*

loss��4=����       �	x'��Xc�A�*

loss�MS=�e �       �	����Xc�A�*

loss�<(�W       �	F{��Xc�A�*

loss�6T=B��       �	���Xc�A�*

loss�=5y�>       �	���Xc�A�*

loss�؄=u�=       �	{M��Xc�A�*

lossxX6=�Mr       �	:锸Xc�A�*

lossO��<y�,�       �	x��Xc�A�*

loss��'<3d��       �	X��Xc�A�*

lossf{5=�q�3       �	[���Xc�A�*

loss�(=I��[       �	PP��Xc�A�*

loss��8<:�Z       �	�闸Xc�A�*

loss$O�<z��       �	����Xc�A�*

loss�ؗ<uo}N       �	�#��Xc�A�*

loss �L;��nT       �	����Xc�A�*

loss�=�<ϊ��       �	�X��Xc�A�*

loss��={�W�       �	�Xc�A�*

loss�+3=\R[�       �	�=��Xc�A�*

loss[8s<�[!�       �	L���Xc�A�*

loss11�=�>(       �	�.��Xc�A�*

loss%Q�<��U�       �	�̞�Xc�A�*

lossLĞ<�I       �	V���Xc�A�*

loss�l>Y�        �	�:��Xc�A�*

loss&�D<!�$�       �	@ߠ�Xc�A�*

lossh_\=�H2       �	����Xc�A�*

loss��=^@       �	�*��Xc�A�*

lossc#>���       �	zƢ�Xc�A�*

loss�@�<V�/       �	�a��Xc�A�*

loss�ǉ<O%�       �	u��Xc�A�*

lossl/g=�`X       �	���Xc�A�*

loss���=�'>�       �	�N��Xc�A�*

lossx�=T��E       �	�饸Xc�A�*

loss�%=�=�       �	Q���Xc�A�*

lossv==:��n       �	/��Xc�A�*

loss&��<\�Y       �	�G��Xc�A�*

loss�.
=����       �	�訸Xc�A�*

lossnM�<�9M       �	����Xc�A�*

lossJ+�;k�1]       �	,��Xc�A�*

loss���;�U5       �	Pê�Xc�A�*

loss��<��3       �	�`��Xc�A�*

lossE6~<�CO<       �	�$��Xc�A�*

loss׆<�y�       �	�ˬ�Xc�A�*

lossE�=��       �	Yi��Xc�A�*

lossw*T<�Jc       �	g��Xc�A�*

loss��6=�ئ�       �	?���Xc�A�*

lossB<.=�       �	aQ��Xc�A�*

lossՅ�={[�h       �	m ��Xc�A�*

loss�[=��R=       �	ʥ��Xc�A�*

loss�ڃ=/��       �	�:��Xc�A�*

loss�z�=�       �	ֱ�Xc�A�*

loss�m|<H�v       �	x��Xc�A�*

loss(;�<ƞ)�       �	��Xc�A�*

loss���<���?       �	����Xc�A�*

loss�==]�T�       �	�S��Xc�A�*

loss��<+oT       �	r��Xc�A�*

lossM��=}2}       �	����Xc�A�*

lossTѳ=0D       �	�h��Xc�A�*

loss��	=4:�7       �	? ��Xc�A�*

loss(�<�2��       �	]���Xc�A�*

loss	�=ş��       �	IH��Xc�A�*

loss��7=�A��       �	��Xc�A�*

loss�<�/�2       �	+���Xc�A�*

loss)�;�Q�       �	RH��Xc�A�*

loss���=���       �	����Xc�A�*

lossB)=��p       �	���Xc�A�*

loss�p�=IV2&       �	����Xc�A�*

loss��<{��       �	�U��Xc�A�*

loss$�<g@~0       �	����Xc�A�*

loss�<��@�       �	����Xc�A�*

lossq��<��B7       �	\:��Xc�A�*

loss�	?=٧��       �	+߿�Xc�A�*

loss]�=�B�       �	����Xc�A�*

loss]r=;3k�       �	,��Xc�A�*

loss�p�<M�V       �	X���Xc�A�*

loss�s2=@��M       �	�d¸Xc�A�*

loss���=L�!�       �	-øXc�A�*

lossx�<.�@#       �	�øXc�A�*

loss_��<��       �	cFĸXc�A�*

loss��j=��N�       �	��ĸXc�A�*

loss���<=�d       �	k�ŸXc�A�*

loss��z=�PqM       �	h$ƸXc�A�*

lossN�2=�
�       �	��ƸXc�A�*

lossz�=�F��       �	&nǸXc�A�*

losso�+<��       �	�ȸXc�A�*

lossԎ=b�j�       �	�ȸXc�A�*

loss���<ȇ�-       �	HRɸXc�A�*

loss�#�;�/�       �	��ɸXc�A�*

lossM��=�y1s       �	�ʸXc�A�*

loss��<#��       �	]3˸Xc�A�*

loss���<��.�       �	)�˸Xc�A�*

loss�<==|s�       �	j̸Xc�A�*

loss��k=	4�       �	F͸Xc�A�*

loss�o=*.n       �	M�͸Xc�A�*

loss�xa=�4j       �	�IθXc�A�*

lossLն<``P@       �	F�θXc�A�*

lossAT	=�a\�       �	�ϸXc�A�*

loss,ۇ<M"�p       �	�иXc�A�*

loss�ְ;˩�6       �	0�иXc�A�*

loss�;�
�       �	�ZѸXc�A�*

lossg� =Y��c       �	A�ҸXc�A�*

lossN�o<�b�       �	�rӸXc�A�*

loss��N<�-D       �	ԸXc�A�*

lossaѳ;��g       �	H�ԸXc�A�*

loss���<���       �	�aոXc�A�*

lossi�B=�
�A       �	5
ָXc�A�*

loss$�<f�_       �	W�ָXc�A�*

lossl.=��V�       �	�d׸Xc�A�*

lossc�L=<�T       �	� ظXc�A�*

loss��R<}\�a       �	��ظXc�A�*

lossdw;Q�:�       �	 CٸXc�A�*

loss�"w:�QXU       �	��ٸXc�A�*

lossf��;g�%1       �	4�ڸXc�A�*

loss�/U<����       �	�۸Xc�A�*

loss��;�R��       �	)BܸXc�A�*

loss�Gf;}rT       �	n�ܸXc�A�*

loss*= �pi       �	BzݸXc�A�*

losst�';���       �	t)޸Xc�A�*

lossG:�i��       �	��޸Xc�A�*

loss�:r��v       �	s߸Xc�A�*

loss��:�$       �	�
�Xc�A�*

loss$�|<v��       �	Ih�Xc�A�*

loss&�@=�XN�       �	��Xc�A�*

loss7��:e�?        �	 ��Xc�A�*

loss���<����       �	���Xc�A�*

loss��=,D�       �	��Xc�A�*

loss �<���       �	��Xc�A�*

loss��f=��*       �	�`�Xc�A�*

loss͊Y=��r�       �	���Xc�A�*

lossaă=ȏ~J       �	��Xc�A�*

loss�j<�6�\       �	�V�Xc�A�*

loss�6O<��pk       �	C�Xc�A�*

losseJ�<x!Aq       �	*��Xc�A�*

lossx[=(�)�       �	�O�Xc�A�*

loss8oZ=�E��       �	���Xc�A�*

loss���<*e%�       �	��Xc�A�*

lossD�G=Zɔ�       �	�;�Xc�A�*

lossG�=mS�       �	���Xc�A�*

loss���=q ��       �	A��Xc�A�*

loss��<��       �	���Xc�A�*

loss�Č=�ꚾ       �	����Xc�A�*

loss�O�<��V�       �	�m�Xc�A�*

loss�=���       �	��Xc�A�*

lossD`<�a�       �	���Xc�A�*

loss΂c=WfG�       �	[�Xc�A�*

loss�=�x�       �	K �Xc�A�*

loss\�s<X�E�       �	��Xc�A�*

loss���=�٧�       �	/Q�Xc�A�*

loss��=���       �	���Xc�A�*

loss,�%<Wge       �	���Xc�A�*

lossQ9�;TI|J       �	����Xc�A�*

loss��#<�`;       �	#��Xc�A�*

loss�2�<9��I       �	���Xc�A�*

loss�&�<�HY       �	�_��Xc�A�*

lossfi�=��6�       �	���Xc�A�*

loss��<9A�z       �	>���Xc�A�*

lossR��<�I{E       �	=~��Xc�A�*

loss#/e=��)�       �	"R��Xc�A�*

lossԠ�<|��q       �	����Xc�A�*

loss|�q;J6�       �	����Xc�A�*

lossQbr=)�^       �	�*��Xc�A�*

loss;C=���       �	����Xc�A�*

lossj��<m�7I       �	bf��Xc�A�*

loss$�j=���       �	�J��Xc�A�*

loss��\=���       �	�?��Xc�A�*

lossDS.=S9��       �	�W��Xc�A�*

loss-�<���       �	n���Xc�A�*

loss�Gv<~�l�       �	� �Xc�A�*

lossW,6=�z        �	X�Xc�A�*

loss�י<Lg��       �	#��Xc�A�*

loss�0=�­�       �	���Xc�A�*

loss�Ԥ<p�?f       �	�;�Xc�A�*

lossm$=#��       �	���Xc�A�*

losst�<z��       �	�Xc�A�*

loss=<�<���       �	�*�Xc�A�*

lossN%<��	?       �	���Xc�A�*

lossq�j<:�       �	�h�Xc�A�*

loss}B�<2&M]       �	���Xc�A�*

loss�-.= �ڿ       �	rR�Xc�A�*

lossl�0>���h       �	���Xc�A�*

loss�C<ts/k       �	0� �Xc�A�*

loss�q=(�       �	�4!�Xc�A�*

lossz��<��w�       �	��!�Xc�A�*

lossa�S=�T%�       �	du"�Xc�A�*

loss��
=6�+7       �	�#�Xc�A�*

loss�<H=��N	       �	3�#�Xc�A�*

lossjw�=>Po       �	5A$�Xc�A�*

loss��/<�@B       �	��$�Xc�A�*

loss�1�<�S$�       �	Bv%�Xc�A�*

loss�L^=k�*�       �	�&�Xc�A�*

lossI�=�}�       �	��&�Xc�A�*

loss�Pi=on/�       �	�A'�Xc�A�*

loss*a&=ء��       �	�'�Xc�A�*

loss_o�<5S��       �	�~(�Xc�A�*

loss��<�<.y       �	�)�Xc�A�*

lossa^�<�H5W       �	�)�Xc�A�*

loss&0 =�Q�E       �	,G*�Xc�A�*

loss���<��Z.       �	�
+�Xc�A�*

loss�6n=�F��       �	y ,�Xc�A�*

loss?��;=���       �	��,�Xc�A�*

loss��=K�       �	�N-�Xc�A�*

lossŹ<So٫       �	�-�Xc�A�*

loss��a<$	�s       �	�}.�Xc�A�*

loss�3<�R٣       �	n/�Xc�A�*

loss��<_uf�       �	��/�Xc�A�*

loss6C=:��       �	�E0�Xc�A�*

loss��H<dZ�       �	��0�Xc�A�*

loss?Q4=)� 8       �	�~1�Xc�A�*

lossL��<e�Z       �	2�Xc�A�*

loss\��<��       �	}�2�Xc�A�*

lossҟA=T �y       �	mU3�Xc�A�*

lossxn�<x��       �	��3�Xc�A�*

loss���<)�R       �	�4�Xc�A�*

lossA[�<]]�       �	�E5�Xc�A�*

lossF�<f;6}       �	��5�Xc�A�*

loss�*�=����       �	��6�Xc�A�*

loss];�=ѯÌ       �	�F7�Xc�A�*

loss�4�<�b�       �	�/8�Xc�A�*

loss���;"+7       �	�8�Xc�A�*

loss  �;�,�       �	K�9�Xc�A�*

loss��<xt��       �	Af:�Xc�A�*

loss/�=� �a       �	�r;�Xc�A�*

loss�'�=;]�       �	�4<�Xc�A�*

loss��r=c���       �	� =�Xc�A�*

loss�=A= ��       �	1�=�Xc�A�*

lossi� <��"�       �	��>�Xc�A�*

loss���:bW       �	IH?�Xc�A�*

loss�<���       �	\8@�Xc�A�*

loss�t�<q�-       �	��@�Xc�A�*

loss�a<�?�:       �	T�A�Xc�A�*

loss1'>D&       �	S�B�Xc�A�*

loss��</��N       �	%D�Xc�A�*

lossL�;z��_       �	I�D�Xc�A�*

loss�*<�<�V       �	[BE�Xc�A�*

loss�C;gC��       �	2�E�Xc�A�*

loss��D<x4M�       �	(G�Xc�A�*

losstu�< �0�       �	M�G�Xc�A�*

loss(�v=x@Me       �	�I�Xc�A�*

lossc1�<�T%d       �	��I�Xc�A�*

loss�k�;���n       �	F}J�Xc�A�*

loss��=���|       �	�qK�Xc�A�*

loss&B�;-o       �	L�Xc�A�*

lossɈ�;��4�       �	(�L�Xc�A�*

lossS�r=6��       �		PM�Xc�A�*

loss~f=#M=�       �	��M�Xc�A�*

loss�Ia<Zq�X       �	��N�Xc�A�*

lossݧ<bt�#       �	�O�Xc�A�*

lossΘ�< a�       �	;P�Xc�A�*

lossrh�=m�&�       �	��P�Xc�A�*

loss]=P~�       �	:yQ�Xc�A�*

loss8��<��=b       �	MR�Xc�A�*

loss��<�\��       �	O�R�Xc�A�*

lossx),<���       �	�HS�Xc�A�*

lossV�<�<�       �	�T�Xc�A�*

loss��=�x��       �	.�T�Xc�A�*

lossL�<���,       �	�;U�Xc�A�*

lossx =�o*       �	u�U�Xc�A�*

loss7ݞ;�+L       �	 �V�Xc�A�*

loss���<���       �	E/W�Xc�A�*

loss䞂=J���       �	&�W�Xc�A�*

loss=%�<.�P       �	��X�Xc�A�*

loss��=�� �       �	t$Y�Xc�A�*

loss�P+=޼��       �	Z�Y�Xc�A�*

loss,��<��~       �	�lZ�Xc�A�*

loss��<���       �	�[�Xc�A�*

loss���<       �	4�[�Xc�A�*

loss��=��.�       �	0e\�Xc�A�*

loss&��<t��       �	��\�Xc�A�*

lossu=O.a4       �	��]�Xc�A�*

loss�.2<Zh:       �	�#^�Xc�A�*

loss���<
Α       �	��^�Xc�A�*

loss�V9=��X�       �	��_�Xc�A�*

loss��<�M��       �	�(`�Xc�A�*

lossM�V<�h�L       �	�`�Xc�A�*

lossd�<s{��       �	�ua�Xc�A�*

loss��=K�:       �	�
b�Xc�A�*

loss`"n<�ױ�       �	
�b�Xc�A�*

loss<��;b	�       �	a5c�Xc�A�*

loss:�#<�4��       �	��c�Xc�A�*

lossc�<G[[       �	�jd�Xc�A�*

loss��;u�*4       �	)e�Xc�A�*

loss�x�<����       �	��e�Xc�A�*

lossj�|=x '       �	�pf�Xc�A�*

loss� <a��:       �	�g�Xc�A�*

loss�^<��;�       �	�g�Xc�A�*

loss�k�<���
       �	q=h�Xc�A�*

lossS,�<��+       �	E�h�Xc�A�*

loss�n�=���       �	3ji�Xc�A�*

losszK/<��g`       �	*�i�Xc�A�*

lossl�=ȴ��       �	D�j�Xc�A�*

loss�F<�$q       �	[xk�Xc�A�*

lossL]�<y��       �	�l�Xc�A�*

loss�x;Έ��       �	J�l�Xc�A�*

loss��=���       �	��m�Xc�A�*

lossꄰ=�BN�       �	In�Xc�A�*

loss�o	=:o       �	��n�Xc�A�*

lossC]#=�G��       �	zo�Xc�A�*

loss��E=@���       �	�%p�Xc�A�*

loss�e@=�N�       �	˼p�Xc�A�*

loss���<�ˑn       �	!�q�Xc�A�*

loss<V�<;+       �	{Jr�Xc�A�*

lossA�);�ԌS       �	&�r�Xc�A�*

loss��u=��yP       �	lws�Xc�A�*

loss2�3=[/0h       �	�t�Xc�A�*

lossd�<���       �	ܼt�Xc�A�*

loss�r�<c��       �	�gu�Xc�A�*

loss�N&=-�       �	� v�Xc�A�*

loss�
$< �0�       �	x�v�Xc�A�*

loss�k<��H}       �	*7w�Xc�A�*

loss��<�O}       �	��w�Xc�A�*

loss��<�I�n       �		mx�Xc�A�*

loss�^>��ӽ       �	�	y�Xc�A�*

lossn�<�5,�       �	n�y�Xc�A�*

loss@s<�6J>       �	@z�Xc�A�*

loss���<��       �	��z�Xc�A�*

loss=w?<��X�       �	�g{�Xc�A�*

lossJ��<À�       �	d|�Xc�A�*

loss��<�h'�       �	ϟ|�Xc�A�*

loss��K<Kjr,       �	/P}�Xc�A�*

loss}6E=E�ڔ       �	 �}�Xc�A�*

loss<:U<؛�       �	�~�Xc�A�*

loss(�<�i}j       �	��Xc�A�*

loss��<"��u       �	��Xc�A�*

loss�o=�ˈ.       �	W��Xc�A�*

lossq�V;%'l       �	����Xc�A�*

loss}i|<�I�       �	����Xc�A�*

lossiѯ<f#       �	F%��Xc�A�*

lossDT=��!�       �	_���Xc�A�*

loss�v=�O-       �	�L��Xc�A�*

loss��$= I=�       �	�߃�Xc�A�*

loss�d�=ǂ�	       �	^���Xc�A�*

loss�Л<ۚ�6       �		��Xc�A�*

loss���;�x�       �	⮅�Xc�A�*

loss��;R���       �	tC��Xc�A�*

loss4�P=?�51       �	M׆�Xc�A�*

loss&
�=(�       �	Bv��Xc�A�*

loss��<Fu�{       �	��Xc�A�*

loss�(�<1��       �	���Xc�A�*

losss�a<�9u6       �	i9��Xc�A�*

lossd��;E.�       �	�҉�Xc�A�*

loss�1�:��       �	�l��Xc�A�*

loss�AE=$��       �	��Xc�A�*

losso�X<����       �	����Xc�A�*

loss�. <<zw       �	EG��Xc�A�*

loss��C=��XC       �	���Xc�A�*

loss3F�<(#�.       �	���Xc�A�*

loss� =(��       �	AJ��Xc�A�*

loss�q�<1$��       �	�⎹Xc�A�*

loss �;�S�       �	�v��Xc�A�*

loss�&<ȼ�_       �	P��Xc�A�*

lossN��=�=cH       �	񺐹Xc�A�*

loss���;8$��       �	>^��Xc�A�*

lossz��=�M��       �	����Xc�A�*

lossX�=��sV       �	�В�Xc�A�*

loss���<r:�       �	"r��Xc�A�*

loss|i�=5o�.       �	Z��Xc�A�*

loss�с<���       �	����Xc�A�*

lossv�V<N}�       �	�N��Xc�A�*

loss��<����       �	�핹Xc�A�*

loss��M<���       �	�Ζ�Xc�A�*

lossd�1<,��       �	�d��Xc�A�*

loss^��;W��4       �	x
��Xc�A�*

loss��X<ʺ4�       �	滛�Xc�A�*

loss�{<�ϓS       �	H��Xc�A�*

loss�m=5ܑd       �	u噹Xc�A�*

loss�(p={�3       �	�|��Xc�A�*

loss;��<Q�A       �	L��Xc�A�*

lossw?<��>       �	����Xc�A�*

losss��<�F��       �	�J��Xc�A�*

lossl4�<��$       �	 휹Xc�A�*

loss�%�;�ߡ       �	׊��Xc�A�*

lossGi�:9��8       �	Eb��Xc�A�*

loss�4=����       �	��Xc�A�*

lossӨ�;��Q�       �	����Xc�A�*

loss&�=���       �	�A��Xc�A�*

loss��C==:�       �	Dܠ�Xc�A�*

loss�><aFBE       �	mr��Xc�A�*

loss��=�:D�       �	���Xc�A�*

loss���;|�3       �	����Xc�A�*

loss�X<�j       �	�L��Xc�A�*

loss�ֶ<�u�D       �	죹Xc�A�*

loss�t�<c٘       �	͓��Xc�A�*

lossm8�<L���       �	�0��Xc�A�*

loss?�=+�]       �	�٥�Xc�A�*

loss��<�m�       �	w���Xc�A�*

loss���<����       �	���Xc�A�*

loss��<�]��       �	g���Xc�A�*

loss%�<���T       �	�M��Xc�A�*

loss�f�;�ssM       �	X橹Xc�A�*

loss���;ԟ�       �	�|��Xc�A�*

losso&I;�mMH       �	�$��Xc�A�*

lossm�3=�q�       �	���Xc�A�*

loss�uA=O�g:       �	eS��Xc�A�*

lossHd<=����       �	W鬹Xc�A�*

loss�f=2O�       �	��Xc�A�*

loss���<g k       �	�8��Xc�A�*

loss��r=�1�]       �	�᮹Xc�A�*

loss�|b=�K/_       �	�~��Xc�A�*

loss�~<T��       �	���Xc�A�*

loss��<lU�&       �	9���Xc�A�*

lossS�=G�Â       �	�`��Xc�A�*

loss��<���       �	����Xc�A�*

lossM�=p��x       �	a���Xc�A�*

lossju�<�^�       �	�_��Xc�A�*

loss���;8g�4       �	����Xc�A�*

lossg)�;�2U�       �	ѐ��Xc�A�*

lossW��<�I�4       �	kF��Xc�A�*

loss��t<���       �	�Xc�A�*

loss@<<�ʙ�       �	����Xc�A�*

loss�<k��5       �	e8��Xc�A�*

loss��<y+��       �	tз�Xc�A�*

loss�3�=�V       �	�o��Xc�A�*

loss_�=��[�       �	���Xc�A�*

lossq5e=�q�       �	���Xc�A�*

loss�A=i�jf       �	����Xc�A�*

loss�}�=:%��       �	"3��Xc�A�*

lossz/%<����       �	�л�Xc�A�*

loss���<sW5       �	1���Xc�A�*

loss�۞<]�C�       �	�p��Xc�A�*

loss䝆<��׽       �	�5��Xc�A�*

loss�:e<'�h       �	�о�Xc�A�*

loss}<���"       �	�x��Xc�A�*

loss�<
c��       �	���Xc�A�*

loss���<�)�       �	\���Xc�A�*

loss�J�;��60       �	�j��Xc�A�*

loss8=L}       �	�¹Xc�A�*

loss��<Ǟ��       �	H�¹Xc�A�*

loss�3<���B       �	�KùXc�A�*

lossA]�=�I4       �	��ùXc�A�*

loss3�<w9�w       �	"�ĹXc�A�*

loss4Ʈ:���       �	�8ŹXc�A�*

loss�HE<TDt&       �	j�ŹXc�A�*

lossq(�<��:�       �	tƹXc�A�*

loss��1<b�9�       �	ǹXc�A�*

loss&�;<���       �	�ǹXc�A�*

loss B�;��O"       �	JȹXc�A�*

lossv'�<<#       �	��ȹXc�A�*

loss*=�w�       �	�ɹXc�A�*

loss��x<n4�       �	�YʹXc�A�*

loss�{=h��l       �	7�ʹXc�A�*

loss��(<�Of�       �	74̹Xc�A�*

loss֤v=;|wb       �	G�̹Xc�A�*

loss	6%<_��u       �	+l͹Xc�A�*

loss���=ݙ��       �	�ιXc�A�*

lossP�
=�W�Z       �	$�ιXc�A�*

lossq��<٥�       �	�OϹXc�A�*

loss:�<�tb�       �	�ϹXc�A�*

loss��2<�rS�       �	��йXc�A�*

loss�9�<����       �	W&ѹXc�A�*

lossF�Z=�d׎       �	��ѹXc�A�*

loss�Ƙ<.I�       �	��ҹXc�A�*

loss*�>ƾ�d       �	�@ӹXc�A�*

loss���=���       �	��ӹXc�A�*

loss��<�ۃ       �	�ԹXc�A�*

lossU=^M�       �	E.չXc�A�*

loss�B=�|�K       �	`�չXc�A�*

lossd2=J��9       �	�qֹXc�A�*

loss+#�:Q;)       �	�*׹Xc�A�*

loss���;�ny       �	q�׹Xc�A�*

loss�=�)y[       �	�lعXc�A�*

loss��b=s��       �	1	ٹXc�A�*

lossWq�<���3       �	�ٹXc�A�*

lossD1�=c�/       �	�TڹXc�A�*

loss���<��jP       �	;�ڹXc�A�*

lossde7;�H�       �	'�۹Xc�A�*

loss׏=x�C       �	�>ܹXc�A�*

loss2� <}��        �	��ܹXc�A�*

lossq�z;�%,D       �	��ݹXc�A�*

loss��g=?���       �	��޹Xc�A�*

lossIWn:v��       �	c|߹Xc�A�*

losslu�:�Q[c       �	L�Xc�A�*

loss��<[�,E       �	���Xc�A�*

lossr�.<e^�U       �	t]�Xc�A�*

loss��&<�{m�       �	��Xc�A�*

loss��=w��       �	��Xc�A�*

loss3�=�i�Z       �	w�Xc�A�*

loss��=�ܑ�       �	��Xc�A�*

loss��P=�_S�       �	F��Xc�A�*

lossT)b<P���       �	�i�Xc�A�*

loss�*=�Q�       �	�Xc�A�*

losss&�=8��       �	8��Xc�A�*

losso{�<UgF�       �	���Xc�A�*

losse�=��z       �	Ll�Xc�A�*

loss3z<�/�       �	�Xc�A�*

loss���=@8�:       �	a��Xc�A�*

loss��==qAQ�       �	;�Xc�A�*

loss�.�;��4�       �	w��Xc�A�*

loss^D<c�l       �	aq�Xc�A�*

loss���<$�os       �	��Xc�A�*

loss�
2=�w�_       �	���Xc�A�*

lossRR�<���       �	�z��Xc�A�*

loss���;�C3       �	��Xc�A�*

loss�=���       �	~��Xc�A�*

loss)I=.VL       �	A�Xc�A�*

loss��X<��u�       �	 '�Xc�A�*

lossX	=��41       �	���Xc�A�*

loss\6�; !�/       �	zR�Xc�A�*

loss��<W`��       �	G�Xc�A�*

loss���=�F�D       �	c��Xc�A�*

loss�|=t��s       �	Y�Xc�A�*

lossQ+G<0*�       �	�3��Xc�A�*

lossf[=�xP       �	����Xc�A�*

loss���<L��       �	�g��Xc�A�*

loss��=��d       �	���Xc�A�*

loss�6�<0��       �	o���Xc�A�*

lossj%=�UFt       �	8��Xc�A�*

lossc=$]՞       �	����Xc�A�*

loss*=yr6u       �	�p��Xc�A�*

losslS<�g       �	5	��Xc�A�*

lossj6.<���       �	���Xc�A�*

lossu;<gqi�       �	�@��Xc�A�*

loss�1�<2�}.       �	Y���Xc�A�*

loss��=��!�       �	�z��Xc�A�*

loss���<�@��       �	���Xc�A�*

loss��{=�L/       �	����Xc�A�*

lossv�B=@:6�       �	�D��Xc�A�*

lossɼ;�Z/       �	9��Xc�A�*

loss�{U;��2�       �	ƣ��Xc�A�*

loss�&�<4֋       �	����Xc�A�*

loss���=�U��       �	�� �Xc�A�*

loss�K�;s��       �	q�Xc�A�*

loss��B=cð
       �	��Xc�A�*

loss_�M=V��       �	���Xc�A�*

loss�:�<�Ǆ�       �	!�Xc�A�*

losss>	=�w�h       �	���Xc�A�*

loss��=JR�l       �	���Xc�A�*

loss)U�<d��       �	�m�Xc�A�*

loss��;����       �	�v�Xc�A�*

loss��c<�/�       �	rm�Xc�A�*

loss�=0_ �       �	=	�Xc�A�*

loss�z�<��{q       �	�
�Xc�A�*

loss?)�<%�e       �	R�
�Xc�A�*

loss��=rw�U       �	�R�Xc�A�*

loss]��;�       �	p��Xc�A�*

loss��\<0#�       �	�Xc�A�*

loss	m�<�,�       �	��Xc�A�*

loss��R=��       �	���Xc�A�*

loss��=�o�       �	�g�Xc�A�*

loss}I;�>       �	i��Xc�A�*

loss��<N\`B       �	��Xc�A�*

losstB�=lW�
       �	;�Xc�A�*

loss$q�<m���       �	���Xc�A�*

lossX�?=��       �	2q�Xc�A�*

lossF�= ^�Q       �	W�Xc�A�*

loss���;����       �	���Xc�A�*

lossh�&=����       �	�H�Xc�A�*

loss`:T<7ؒ       �	���Xc�A�*

lossX�q<\P�E       �	�t�Xc�A�*

lossT�>=�s��       �	�	�Xc�A�*

loss�"=}rU	       �	��Xc�A�*

loss�=���       �	�F�Xc�A�*

loss�b9=2'
       �	<�Xc�A�*

loss!��<c���       �	���Xc�A�*

loss\}>=Y�XV       �	k�Xc�A�*

loss�M=~�[�       �	�e�Xc�A�*

loss�'i=+k=       �	��Xc�A�*

loss��I<��       �	���Xc�A�*

loss��j==⌓       �	]��Xc�A�*

lossr9�<5�:       �	t�Xc�A�*

loss7�S<O��w       �	��Xc�A�*

loss1�<F�˷       �	���Xc�A�*

loss��<&�       �	�4!�Xc�A�*

loss/i�<���R       �	�"�Xc�A�*

loss��=B��"       �	=�#�Xc�A�*

lossy9�=����       �	j�$�Xc�A�*

loss���< e��       �	�2%�Xc�A�*

loss��<��       �	��%�Xc�A�*

lossi�+<��#�       �	�&�Xc�A�*

loss�h�=24��       �	1]'�Xc�A�*

loss��k=d�2       �	(�Xc�A�*

loss��=���9       �	�(�Xc�A�*

loss$��;�+}       �	h@)�Xc�A�*

lossF��=��^       �	<�)�Xc�A�*

loss�G�:hO��       �	�q*�Xc�A�*

loss���;���[       �	=+�Xc�A�*

loss�GY<���       �	q�+�Xc�A�*

loss7��<(�h�       �	H,�Xc�A�*

loss��<�H7       �	�&-�Xc�A�*

loss(�;<�ڒ�       �	;�-�Xc�A�*

losst��=�+%4       �	�[.�Xc�A�*

loss=!<<�4�       �	�$/�Xc�A�*

loss^�=����       �	n�/�Xc�A�*

loss�+O<c�$�       �	(e0�Xc�A�*

loss�-�=��       �	_	1�Xc�A�*

loss.�a<���       �	��1�Xc�A�*

loss6-<�ؔ{       �	<K2�Xc�A�*

loss���<P���       �		�2�Xc�A�*

lossC=M�T<       �	��3�Xc�A�*

loss�wy=�T��       �	V,4�Xc�A�*

loss���< 潹       �	��4�Xc�A�*

loss�h�=D�<       �	��5�Xc�A�*

loss��u=���N       �	�o6�Xc�A�*

loss�p�=�?,        �	�7�Xc�A�*

loss��};5tb@       �	��7�Xc�A�*

loss��<��       �	�K8�Xc�A�*

lossԣy=�       �	.�8�Xc�A�*

loss$��<?�]       �	�9�Xc�A�*

loss�<35�A       �	*:�Xc�A�*

lossh��=d~�       �	��:�Xc�A�*

loss-��;��U       �	�s;�Xc�A�*

loss��=îQN       �	L<�Xc�A�*

loss��<U��       �	д<�Xc�A�*

lossŏG=�=h�       �	TR=�Xc�A�*

loss��o<K�0�       �	�=�Xc�A�*

loss�6=m-       �	�>�Xc�A�*

lossd�S<T���       �	�K?�Xc�A�*

loss�Q=��q       �	��@�Xc�A�*

loss��=}�`       �	�wA�Xc�A�*

lossD=3�X�       �	� B�Xc�A�*

loss==Vd��       �	�B�Xc�A�*

lossW�C=��o       �	�/D�Xc�A�*

loss�Ȥ<Ͽ�       �	��D�Xc�A�*

loss!d8<s<��       �	�iE�Xc�A�*

losso�<�j@       �	�F�Xc�A�*

loss��<3�s�       �	^HG�Xc�A�*

loss��;�\`�       �	��G�Xc�A�*

losshQ�<�ǉ�       �		�H�Xc�A�*

lossXB�=��Y       �		PI�Xc�A�*

loss�wR=�u��       �	��J�Xc�A�*

loss��3=����       �	VGK�Xc�A�*

loss���<�s       �	�L�Xc�A�*

loss
��;��]m       �	�fM�Xc�A�*

loss��<���`       �	VEN�Xc�A�*

loss\-�=@$,_       �	J�N�Xc�A�*

lossD�<���       �	��O�Xc�A�*

loss��<J�OD       �	�^P�Xc�A�*

loss���<a:8       �	MfQ�Xc�A�*

lossv�<���#       �	e�Q�Xc�A�*

lossn�=��~�       �	�S�Xc�A�*

loss�%<�+��       �	��S�Xc�A�*

loss��<L�z       �	��T�Xc�A�*

loss%q=D��>       �	�UU�Xc�A�*

loss��<��Y(       �	��U�Xc�A�*

loss��<��       �	�V�Xc�A�*

losss�6<N#�       �	�3W�Xc�A�*

loss���;��!s       �	�X�Xc�A�*

loss��
=�5r�       �	��X�Xc�A�*

loss�/K<���O       �	�lY�Xc�A�*

lossH�<M�d~       �	FZ�Xc�A�*

lossL�<w͸�       �	ƤZ�Xc�A�*

loss|C=���       �	 :[�Xc�A�*

loss
�<OOd       �	J\�Xc�A�*

loss��?<`��       �	�]�Xc�A�*

loss��\=5C�h       �	r�]�Xc�A�*

loss�
<5�2       �	�;^�Xc�A�*

loss��1<�t       �	�^�Xc�A�*

loss;;Sڣ�       �	iq_�Xc�A�*

lossM=�3F�       �	�`�Xc�A�*

loss��4=�S��       �		�`�Xc�A�*

loss�6
='th       �	-Ba�Xc�A�*

loss�
�<L�Q�       �	�a�Xc�A�*

loss��;�{�       �	�ub�Xc�A�*

loss�;���       �	�c�Xc�A�*

loss�Z�;\�Q�       �	d�c�Xc�A�*

loss��<n�j       �	�Od�Xc�A�*

loss��%<@׹       �	��d�Xc�A�*

loss�Y=>�PT       �	��e�Xc�A�*

loss��%=�FS       �	q<f�Xc�A�*

lossX�,=v��)       �	?�f�Xc�A�*

loss�H<�(6v       �	�{g�Xc�A�*

loss2K�=�Ɨ�       �	�h�Xc�A�*

lossAp�<J���       �	��h�Xc�A�*

loss�S�<��       �	�qi�Xc�A�*

loss��<�Lי       �	�j�Xc�A�*

loss��<Y1iG       �	D�j�Xc�A�*

loss���;Ы�X       �	.=k�Xc�A�*

loss��<B�^(       �	�k�Xc�A�*

loss
�<���       �	F{l�Xc�A�*

loss�pU=�	�|       �	�m�Xc�A�*

loss|�0;`N�J       �	e�m�Xc�A�*

loss:b<�xη       �	Jn�Xc�A�*

loss��D<_\��       �	��n�Xc�A�*

loss
>D�O�       �	�o�Xc�A�*

loss@^t=o9       �	\p�Xc�A�*

lossR�o<��*`       �	Z�p�Xc�A�*

loss���<�!R{       �	�Nq�Xc�A�*

loss�3<=���!       �	��q�Xc�A�*

loss���<$��       �	��r�Xc�A�*

loss��;N��       �	�%s�Xc�A�*

lossh�U<(���       �	��s�Xc�A�*

lossX��<:�
       �	;mt�Xc�A�*

loss-�"=�       �	|
u�Xc�A�*

lossV�=�ɺ�       �	f�u�Xc�A�*

loss��<�i��       �	29v�Xc�A�*

loss�A�=p���       �	p�v�Xc�A�*

loss.�=ė�6       �	�jw�Xc�A�*

loss�=p��       �	�x�Xc�A�*

loss8~�<����       �	��x�Xc�A�*

loss\|+=x��       �	qZy�Xc�A�*

loss��5<k�	�       �	��y�Xc�A�*

lossX-m=�g�       �	��z�Xc�A�*

loss�5=9�!       �	�.{�Xc�A�*

loss��F=���       �	��{�Xc�A�*

loss���<�
�       �	|�|�Xc�A�*

lossM�=�P�Y       �	�#}�Xc�A�*

lossa߂<BE��       �	��}�Xc�A�*

lossl�.=W�,/       �	̷~�Xc�A�*

loss��"<�9L}       �	}��Xc�A�*

loss�VE=��Ϣ       �	�耺Xc�A�*

loss$\�:�2q       �	i���Xc�A�*

loss���;���       �	�:��Xc�A�*

loss�K#=p�k�       �	�悺Xc�A�*

lossy~<T11�       �	Ƈ��Xc�A�*

loss��<-皫       �	�+��Xc�A�*

loss�k�<�p�       �	oԄ�Xc�A�*

loss��="��<       �	f���Xc�A�*

lossݡ;�� |       �	q ��Xc�A�*

loss}�^<��f�       �	辆�Xc�A�*

loss�4=�C��       �	h��Xc�A�*

loss�jT;fs��       �	���Xc�A�*

lossx&:�F4       �	����Xc�A�*

lossU� =u�i�       �	Y0��Xc�A�*

lossE<d���       �	�Ή�Xc�A�*

losscz�;Ο�       �	7l��Xc�A�*

loss�{;9%�#       �	���Xc�A�*

losszP;w�L       �	���Xc�A�*

loss_8�;��4       �	�O��Xc�A�*

loss�:���       �	k���Xc�A�*

loss�^:��ޏ       �	�S��Xc�A�*

lossz��9�u�       �	�Xc�A�*

lossv�<<#�b       �	����Xc�A�*

loss.��<3$D�       �	�@��Xc�A�*

loss.u<=R̨       �	[쐺Xc�A�*

loss}g�:7)�\       �	����Xc�A�*

loss*h�<�9L       �	�A��Xc�A�*

loss$�=���       �	����Xc�A�*

loss�q�;��F�       �	~R��Xc�A�*

loss=^8=�z�       �	�蔺Xc�A�*

loss(c�=����       �	˝��Xc�A�*

loss�3/=Ew       �	�1��Xc�A�*

loss�'
=��]�       �	�Ȗ�Xc�A�*

lossfo�; ۽�       �	a��Xc�A�*

loss[Ƅ=�W��       �	B
��Xc�A�*

loss_>�=�gR�       �	f���Xc�A�*

lossR�3=��12       �	�4��Xc�A�*

loss<�<�       �	lΙ�Xc�A�*

loss���<��<       �	<j��Xc�A�*

lossc�<�2�       �	G��Xc�A�*

lossx�=*(�       �	﫛�Xc�A�*

loss8o�;H
�       �	�G��Xc�A�*

loss��<4�Ht       �	�ݜ�Xc�A�*

loss��<L���       �	@���Xc�A�*

lossA��<͠       �	�N��Xc�A�*

loss&!c<})�       �	�Xc�A�*

loss�,t=�v+�       �	]���Xc�A�*

loss`yk;�4�       �	1%��Xc�A�*

loss�D<�g�       �	U���Xc�A�*

loss�4�=��Up       �	X��Xc�A�*

loss s=}��       �	����Xc�A�*

lossh�<;�%�       �	����Xc�A�*

loss��;x!?       �	�8��Xc�A�*

loss�/�;����       �	]⣺Xc�A�*

loss?�7=�1�       �	����Xc�A�*

lossj1E<�+5       �	�+��Xc�A�*

loss��#=�`	�       �	�ӥ�Xc�A�*

loss6|�=�"�_       �	�r��Xc�A�*

loss�<q��       �	���Xc�A�*

loss(g�<wX��       �	z���Xc�A�*

lossZ?�;�_yi       �	�A��Xc�A�*

loss�<;�C       �	ب�Xc�A�*

loss��f<i���       �	"r��Xc�A�*

loss��:<ݱ�       �	���Xc�A�*

loss�̨<��       �	L���Xc�A�*

loss��<��       �	�|��Xc�A�*

loss��j=Cy��       �	���Xc�A�*

lossO� <���       �	����Xc�A�*

loss�y�;��       �	Me��Xc�A�*

loss���;[��E       �	���Xc�A�*

loss��A=L���       �	D���Xc�A�*

lossz��=�Yk        �	Va��Xc�A�*

loss�@�;f֑�       �	����Xc�A�*

lossӎ=��5       �	ũ��Xc�A�*

lossw�?=�,�       �	LO��Xc�A�*

lossJ��;�e��       �	��Xc�A�*

lossd�=<�Zx       �	����Xc�A�*

loss��<��8�       �	�9��Xc�A�*

lossJ@�<L��       �	�״�Xc�A�*

loss	Ĺ<Wٺ       �	�yκXc�A�*

loss�G=��V       �	CϺXc�A�*

loss�Y+<Z\�       �	кXc�A�*

lossF =úW       �	��кXc�A�*

loss���<�ic	       �	�KѺXc�A�*

loss<��a�       �	�ѺXc�A�*

loss���;�K�       �	ѲҺXc�A�*

loss�c<^�       �	HRӺXc�A�*

loss^g=���       �	�ԺXc�A�*

loss��;=�
3+       �	ȷԺXc�A�*

lossi��;)�]       �	�aպXc�A�*

loss\`�=�̖       �	�ֺXc�A�*

lossq�c=H��}       �	*�ֺXc�A�*

lossx'=���       �	nR׺Xc�A�*

losso �<�A]�       �	
�׺Xc�A�*

lossS>=�|�       �	,�غXc�A�*

loss��;����       �	�BٺXc�A�*

loss��<�.o       �	��ٺXc�A�*

loss�I<ܝ�U       �	ՔںXc�A�*

losso=[+�8       �	�/ۺXc�A�*

loss��P<N��       �	|�ۺXc�A�*

lossȤ�:ơA�       �	>=ݺXc�A�*

loss*W<`�/b       �	��ݺXc�A�*

losshD�=�U"%       �	�~޺Xc�A�*

loss�%�<���       �	�ߺXc�A�*

loss<����       �	��ߺXc�A�*

loss��<z6�z       �	 X�Xc�A�*

loss��<��x       �	�Xc�A�*

loss��
=�8Q�       �	m��Xc�A�*

loss�Y�<��       �	F\�Xc�A�*

loss&A<-�       �	���Xc�A�*

lossc�=�j|�       �	N��Xc�A�*

lossS<k���       �	�1�Xc�A�*

lossV_�=�t�       �	���Xc�A�*

lossϨ"=t�k�       �	#i�Xc�A�*

lossv��<m�X�       �	��Xc�A�*

loss��G<��~       �	?��Xc�A�*

loss d�=jh32       �	�R�Xc�A�*

loss�N�=�d+�       �	���Xc�A�*

loss�7=>�,       �	���Xc�A�*

loss�-<�4�       �	,-�Xc�A�*

loss$h<GNv1       �	_��Xc�A�*

lossM��;C�Jh       �	B|�Xc�A�*

lossiP�<�z�       �	��Xc�A�*

loss�[<,��q       �	'��Xc�A�*

loss9�<tkY�       �	>x�Xc�A�*

loss�4=G��       �	&��Xc�A�*

loss=�;9�6�       �	���Xc�A�*

loss��;2f��       �	3S�Xc�A�*

lossr �9W;�x       �	S��Xc�A�*

loss�<��װ       �	���Xc�A�*

loss��==$�       �	7�Xc�A�*

loss��<�f��       �	��Xc�A�*

lossډ�=�q��       �	e�Xc�A�*

loss��<���b       �	�!�Xc�A�*

lossH�;���        �	��Xc�A�*

loss�<��e       �	se�Xc�A�*

lossd�:��Y       �	z��Xc�A�*

loss��<��5&       �	9���Xc�A�*

lossΙ�=B���       �	�<��Xc�A�*

loss?ށ=�,+       �	l���Xc�A�*

loss�w�<�ĵ�       �	����Xc�A�*

loss��;���       �	� ��Xc�A�*

loss�T=���L       �	����Xc�A�*

loss:�e;U[�       �	^��Xc�A�*

lossv�5;�&�#       �	w���Xc�A�*

loss�.=#�&n       �	���Xc�A�*

loss��F=�\J       �	�2��Xc�A�*

loss���=�+E�       �	1���Xc�A�*

loss�e�<�h��       �	���Xc�A�*

lossF��< �o       �	���Xc�A�*

loss�M!=}�bm       �	����Xc�A�*

loss��=���s       �	?t��Xc�A�*

loss
�;�"       �	���Xc�A�*

loss�sK=�1��       �	G���Xc�A�*

loss��<�L       �	)���Xc�A�*

loss$�<^�'�       �	�b �Xc�A�*

loss=�=}��{       �	�� �Xc�A�*

lossM�;�:	u       �	���Xc�A�*

loss:�<o��%       �	�'�Xc�A�*

loss��X=�G�       �	��Xc�A�*

loss� =���       �	�_�Xc�A�*

loss��=@�       �	H��Xc�A�*

loss��<���       �	ٓ�Xc�A�*

loss�ch<��^y       �	Q0�Xc�A�*

loss�}�<�.%�       �	*��Xc�A�*

lossd�z=ƻ��       �	�_�Xc�A�*

loss�[=ȶ;m       �	^��Xc�A�*

losso��<_,K�       �	"��Xc�A�*

loss�5X=i-�        �	 �Xc�A�*

lossz�+<���
       �	p��Xc�A�*

loss8 =&��       �	J	�Xc�A�*

losss��;{��       �	�	�Xc�A�*

lossjA�<��v       �	%z
�Xc�A�*

loss��<v���       �	��Xc�A�*

lossI�?<l�'       �	��Xc�A�*

loss�X9<5-�B       �	C<�Xc�A�*

lossV�=���       �	���Xc�A�*

loss���=ڇ�2       �	�r�Xc�A�*

loss6n�;�~��       �	��Xc�A�*

loss	C�;1��       �	���Xc�A�*

loss�<��z       �	lB�Xc�A�*

loss��<�Ru�       �	(��Xc�A�*

loss��"<"B�       �	n��Xc�A�*

loss�2Z<�d�       �	r7�Xc�A�*

loss���=�ߪ       �	���Xc�A�*

loss�,�;� A       �	��Xc�A�*

loss:n!=�E��       �	��Xc�A�*

lossϼ�=�!��       �	<��Xc�A�*

lossq�y;gÕ~       �	`�Xc�A�*

loss�P=z=�^       �	3�Xc�A�*

loss��;yh��       �	f��Xc�A�*

loss�Sk=�#�Q       �	d�Xc�A�*

loss �;6�CP       �	*�Xc�A�*

loss�n<�+�       �	/��Xc�A�*

loss�0<���        �	�f�Xc�A�*

loss(@�=�=*�       �	���Xc�A�*

loss �P=�mE,       �	%��Xc�A�*

loss��=��%       �	�/�Xc�A�*

loss�Ж<H���       �	G��Xc�A�*

losszx
=C<#�       �	�a�Xc�A�*

loss��<51�       �	Y��Xc�A�*

loss���;.}A       �	���Xc�A�*

loss�[T<x1�S       �	1�Xc�A�*

loss!��;�6�C       �	�=�Xc�A�*

loss�N�<{�>       �	� �Xc�A�*

loss~u�=T0�7       �	ڪ �Xc�A�*

loss�S�;b�jt       �	�R!�Xc�A�*

loss�E=;r&       �	��!�Xc�A�*

loss��=���       �	��"�Xc�A�*

lossq��<�#Y~       �	�A#�Xc�A�*

loss} �<N�B       �	��$�Xc�A�*

loss�<vV       �	<P%�Xc�A�*

loss	oW<ݼ�W       �	W�%�Xc�A�*

loss�EG=e"Q       �	Q�&�Xc�A�*

loss-��<�Gm       �	s*'�Xc�A�*

loss�"�;mVA�       �	�'�Xc�A�*

loss�)J<���)       �	��(�Xc�A�*

lossm�;���       �	�)�Xc�A�*

loss���<�u��       �	
�)�Xc�A�*

loss�!�;PPr�       �	�U*�Xc�A�*

loss�Ƞ<v
(s       �	��*�Xc�A�*

loss�+&=u� '       �	��+�Xc�A�*

loss/-X=^��#       �	�(,�Xc�A�*

loss��<3Qp�       �	��,�Xc�A�*

loss�1,='�d       �	n-�Xc�A�*

lossz�$<��       �	1	.�Xc�A�*

lossqT<+�)       �	�.�Xc�A�*

lossD��<7��|       �	zn/�Xc�A�*

loss��=�:��       �	v0�Xc�A�*

lossl�=�<       �	��0�Xc�A�*

lossc�<5���       �	�T1�Xc�A�*

loss��<�Lc       �	��1�Xc�A�*

lossty=9"T�       �	��2�Xc�A�*

losssh�<�±
       �	
,3�Xc�A�*

loss��;[��       �	R�3�Xc�A�*

loss��n;�s3       �	cd4�Xc�A�*

loss6�=��qI       �	�5�Xc�A�*

loss��=���       �	��5�Xc�A�*

lossC�;�STm       �	�06�Xc�A�*

lossrw<D�9       �	��6�Xc�A�*

loss��;�p�`       �	6s7�Xc�A�*

loss�=�Ah       �	�8�Xc�A�*

loss|��:�L0�       �	�8�Xc�A�*

lossz��<�$7       �	�49�Xc�A�*

loss�Y�;!gK       �	�9�Xc�A�*

loss.�X<���       �	�u:�Xc�A�*

loss�?<�T��       �	;�Xc�A�*

loss�_'<���       �	ٱ;�Xc�A�*

loss�)h<�R;       �	�N<�Xc�A�*

lossd�><�_~�       �	�<�Xc�A�*

lossV�<}�@�       �	e�=�Xc�A�*

loss��$<��ִ       �	�A>�Xc�A�*

loss �#=�kx�       �	��>�Xc�A�*

loss��;p�s�       �	��?�Xc�A�*

lossV+Y=��)       �	�4@�Xc�A�*

loss<��<+q       �	7�@�Xc�A�*

loss�?�< ��@       �	s�A�Xc�A�*

lossx=�K)        �	ՕB�Xc�A�*

loss4B�<?�(g       �	�7C�Xc�A�*

loss�� :���       �	zmD�Xc�A�*

loss�V<��#�       �	��E�Xc�A�*

loss��<�É�       �	<�F�Xc�A�*

loss��$;(���       �	D�G�Xc�A�*

loss�xd<���       �	�4H�Xc�A�*

loss� �<���[       �	o�H�Xc�A�*

loss��=�,�       �	��I�Xc�A�*

loss�Ԡ<Ւ��       �	@0J�Xc�A�*

loss`�<YR��       �	�K�Xc�A�*

loss��=i��~       �	ЛK�Xc�A�*

loss\�>�L�       �	��L�Xc�A�*

loss��=���o       �	[BM�Xc�A�*

lossc�P<���m       �	[�M�Xc�A�*

loss݁�;�:GH       �	��N�Xc�A�*

lossFT�;����       �	�:O�Xc�A�*

loss ��<qn�       �	�6P�Xc�A�*

loss�@�=���       �	�qQ�Xc�A�*

loss.��=*˽%       �	6�R�Xc�A�*

lossV�|<�"ef       �	�RS�Xc�A�*

loss�*<�rp�       �	�
T�Xc�A�*

lossԽ�<�{�       �	j�T�Xc�A�*

lossC_�;{��       �	E�U�Xc�A�*

loss��=#sY       �	`:V�Xc�A�*

loss��<@4=       �	��V�Xc�A�*

lossBU�=���,       �	�uW�Xc�A�*

loss;$�<�        �	;X�Xc�A�*

loss�<5="�       �	�X�Xc�A�*

loss�֚<?���       �	�UY�Xc�A�*

loss���=�,�       �	��Y�Xc�A�*

loss@��<����       �	 �Z�Xc�A�*

lossOQ�<��N�       �	B$[�Xc�A�*

loss�I�<[�4       �	n�[�Xc�A�*

loss撆;�7��       �	g\�Xc�A�*

lossoH�;9{[       �	U]�Xc�A�*

loss�c�<�'9       �	߿]�Xc�A�*

loss��=��/Y       �	��^�Xc�A�*

loss$�=�? �       �	�C_�Xc�A�*

loss J�<QJ       �	'`�Xc�A�*

loss��S=˄�5       �	�-a�Xc�A�*

lossu��=����       �	�a�Xc�A�*

loss ئ;~)��       �	vqb�Xc�A�*

lossTR<O_��       �	@c�Xc�A�*

loss�v<����       �	��c�Xc�A�*

loss���<����       �	�Sd�Xc�A�*

loss�&P<�m��       �	��d�Xc�A�*

loss
y�<���       �	P�e�Xc�A�*

loss5�=+�:#       �	#/f�Xc�A�*

lossà
=�k��       �	G�f�Xc�A�*

loss���<�y��       �	�g�Xc�A�*

lossqu�<K���       �	� h�Xc�A�*

loss��r<�S�       �	��h�Xc�A�*

loss�h�<!��       �	�^i�Xc�A�*

loss|�;<� c       �	j3j�Xc�A�*

losss��<sb9�       �	��j�Xc�A�*

loss(f�<���       �	@mk�Xc�A�*

loss�� =�(:�       �	l�Xc�A�*

loss��=N>P       �	��l�Xc�A�*

lossĀ<Ǧ�       �		�m�Xc�A�*

loss�^�<y�9�       �	B%n�Xc�A�*

loss��;��2b       �	H�n�Xc�A�*

loss]�%<l.�3       �	�[o�Xc�A�*

loss�pe<1j�       �	��o�Xc�A�*

loss�9$<ƍS�       �	ףp�Xc�A�*

lossO.=��N2       �	K>q�Xc�A�*

lossT�I=r=Y�       �	k�q�Xc�A�*

lossA@=�o��       �	�or�Xc�A�*

loss�h�<��j       �	�
s�Xc�A�*

lossS�=/��t       �	��s�Xc�A�*

loss�u=IҜ       �	�@t�Xc�A�*

loss=:n<��)�       �	9�t�Xc�A�*

lossJw=��W�       �	8ku�Xc�A�*

loss��w<�W�       �	uv�Xc�A�*

loss�gj<�̞b       �	؝v�Xc�A�*

lossS�#;����       �	�<w�Xc�A�*

loss5�<"�{H       �	 �w�Xc�A�*

lossǸ<b���       �	8jx�Xc�A�*

lossr��<�;8q       �	�y�Xc�A�*

loss�}T=��C�       �	�y�Xc�A�*

loss�fD=�zb       �	AGz�Xc�A�*

lossC�1=��E�       �	a�z�Xc�A�*

loss$wM='ش-       �	k}{�Xc�A�*

loss=��<B�3�       �	�*|�Xc�A�*

loss�E�<=Og"       �	��|�Xc�A�*

loss��;�Z%       �	�k}�Xc�A�*

loss�R@=j��       �	�~�Xc�A�*

loss��<61��       �	��~�Xc�A�*

loss�F=�B#       �	'J�Xc�A�*

loss���;��       �	���Xc�A�*

loss�r�<�Ǘ       �	⒀�Xc�A�*

loss:&	=)���       �	>y��Xc�A�*

loss��<l�tD       �	A��Xc�A�*

loss��L<����       �	a���Xc�A�*

loss!�X=7��       �	d<��Xc�A�*

lossFo�;$,ґ       �	�؃�Xc�A�*

lossa:�=�Q8�       �	�o��Xc�A�*

lossu�=��@{       �	a��Xc�A�*

lossvi+=֔�       �	%Ʌ�Xc�A�*

lossl�=�te�       �	 b��Xc�A�*

loss�-><jJ��       �	z���Xc�A�*

loss_��=��]�       �	"���Xc�A�*

loss�;+�       �	�A��Xc�A�*

lossl,
<�S*       �	�׈�Xc�A�*

loss�;8���       �	�t��Xc�A�*

loss��==v�h       �	s��Xc�A�*

loss1I�<ѮN�       �	]���Xc�A�*

lossϦ�=�ib       �	�=��Xc�A�*

losso�+=���       �	�Ջ�Xc�A�*

loss�k<:��       �	vp��Xc�A�*

loss�aZ=�4W�       �	���Xc�A�*

loss3��<��[       �	����Xc�A�*

loss�_8;�"�       �	4J��Xc�A�*

loss��7=,�W�       �	C㎻Xc�A�*

loss�N�<�H�       �	_}��Xc�A�*

lossIJ�;�>¨       �	z��Xc�A�*

loss:VQ<���F       �	1���Xc�A�*

loss��=��8	       �	�[��Xc�A�*

loss�0�<K�       �	��Xc�A�*

loss�2=���       �	����Xc�A�*

loss�ۆ=)�&       �	"��Xc�A�*

lossS�< ���       �	곓�Xc�A�*

lossہ=)�C       �	�J��Xc�A�*

loss�E�;&�b       �	�┻Xc�A�*

loss��<i��       �	�}��Xc�A�*

loss��O=D��b       �	&��Xc�A�*

loss�0=x�ܑ       �	���Xc�A�*

loss�|�=VЁ^       �	{M��Xc�A�*

loss��u;9�       �	}痻Xc�A�*

loss�s'=���Y       �	�{��Xc�A�*

lossPl<�q�H       �	E��Xc�A�*

loss#o�<�<0X       �	�虻Xc�A�*

loss$[<*       �	i���Xc�A�*

loss��A=[k�x       �	/��Xc�A�*

loss2��<5Q�       �	�ț�Xc�A�*

loss�L�<x��m       �	l^��Xc�A�*

loss�_=�
��       �	�#��Xc�A�*

loss)�=�5t       �	�䝻Xc�A�*

loss�k�:h��       �	X���Xc�A�*

loss�/< l�       �	�"��Xc�A�*

lossf	=��;       �	˻��Xc�A�*

loss���;�zF�       �	[{��Xc�A�*

loss;,�<��       �	�]��Xc�A�*

loss���=U�       �	����Xc�A�*

loss��=����       �	�*��Xc�A�*

loss.�=z��       �	�ģ�Xc�A�*

lossst
=\��       �	`��Xc�A�*

lossŉ=lS�       �	�~��Xc�A�*

loss�5�<	a�       �	�;��Xc�A�*

lossY&�=V!>W       �	�ᦻXc�A�*

loss��l<�       �	�v��Xc�A�*

losse�=���i       �	��Xc�A�*

lossM�;D��       �	���Xc�A�*

loss�ۡ;�\�P       �	q���Xc�A�*

loss�n�;�_m�       �	ZH��Xc�A�*

loss�1�<��       �	�۪�Xc�A�*

lossx��<�0g�       �	�t��Xc�A�*

lossΏ=��k9       �	���Xc�A�*

loss���<��       �	U���Xc�A�*

loss�^�<.%$K       �	uɭ�Xc�A�*

lossʓ!=r���       �	�_��Xc�A�*

lossq'�;r�<�       �	'���Xc�A�*

loss�c�;􀴖       �	ԛ��Xc�A�*

lossDA=��#�       �	0��Xc�A�*

loss�&=�9"       �	���Xc�A�*

lossZ��;T�       �	J���Xc�A�*

loss�^�=Y��q       �	\Y��Xc�A�*

loss�=< �2       �	ﲻXc�A�*

loss��;ΙV�       �	g���Xc�A�*

loss�^k=�1h       �	�3��Xc�A�*

loss�~=�;��       �	mȴ�Xc�A�*

lossL�;$d]�       �	sd��Xc�A�*

loss��;���       �	����Xc�A�*

loss�&�;�oDA       �	���Xc�A�*

loss��=0�y�       �	�Q��Xc�A�*

loss��A;F�{�       �	�䷻Xc�A�*

loss�^�=�6X       �	Z���Xc�A�*

loss�;\��       �	<N��Xc�A�*

loss7%#<�0M�       �	�幻Xc�A�*

loss��
=r~�A       �	ض��Xc�A�*

loss�]�<�[�i       �	DN��Xc�A�*

loss=�<p[&       �	|�Xc�A�*

loss�sN<�.�       �	]���Xc�A�*

lossd��=���       �	���Xc�A�*

loss �R<���O       �	Ͼ��Xc�A�*

lossN�=�T*�       �	�]��Xc�A�*

loss���<�k/�       �	��Xc�A�*

lossG�
>��³       �	���Xc�A�*

loss�i�<u�-       �	mS��Xc�A�*

lossF&<�m�>       �	�.��Xc�A�*

loss���<W�R       �	���Xc�A�*

loss9v<7W��       �	��»Xc�A�*

loss���<��       �	wMûXc�A�*

loss�E�;��Q�       �	��ûXc�A�*

loss�B�<��       �	��ĻXc�A�*

loss�J<+��       �	�nŻXc�A�*

losswk�<E�       �	�
ƻXc�A�*

lossؐ�<�2Q       �	x�ƻXc�A�*

loss�-=�t�       �	VǻXc�A�*

loss6��<��       �	��ǻXc�A�*

loss�o�<t���       �	w�ȻXc�A�*

lossRD'<w~�       �	a6ɻXc�A�*

loss��<��-/       �	��ɻXc�A�*

loss�ɗ<n�A�       �	�yʻXc�A�*

lossJ=]<��       �	�˻Xc�A�*

loss E�;<���       �	z�˻Xc�A�*

lossRF�:|qUf       �	�}̻Xc�A�*

loss��)=|�i�       �	�ͻXc�A�*

lossI1=c��       �	}�ͻXc�A�*

loss��=��F       �	IλXc�A�*

lossSC;'�       �	��λXc�A�*

loss�<�0G       �	��ϻXc�A�*

lossE�V;UV�=       �	�VлXc�A�*

loss�S=ؔB�       �	�лXc�A�*

loss�M<���       �	C�ѻXc�A�*

lossX�A=�F�4       �	g&һXc�A�*

loss�O�<2M�?       �	H�һXc�A�*

loss���<My��       �	NdӻXc�A�*

loss&Y�:~c�       �	[(ԻXc�A�*

lossn�<���2       �	�ԻXc�A�*

loss�v<N�q       �	�ZջXc�A�*

loss��=�L       �	ֻXc�A�*

lossu�;O!˜       �	�ֻXc�A�*

loss�Nr<�m'=       �	+�׻Xc�A�*

loss�U�<AK�       �	-#ػXc�A�*

lossaf=�ш       �	Q�ػXc�A�*

lossS��<�8       �	�TٻXc�A�*

loss4SU=����       �	c�ٻXc�A�*

loss�@=�0k       �	�ڻXc�A�*

loss�
�<4���       �	�!ۻXc�A�*

loss��<�-       �	˹ۻXc�A�*

loss��p=�(GP       �	V�ܻXc�A�*

loss�t=u���       �	$ݻXc�A�*

loss.��<fx��       �	�4޻Xc�A�*

loss�"=��t       �	��޻Xc�A�*

lossI�==�y�^       �	zp߻Xc�A�*

lossy��=���k       �	t
�Xc�A�*

lossI�*=vƯN       �	���Xc�A�*

loss��!<⩠       �	o�Xc�A�*

lossT]x<���       �	��Xc�A�*

loss`b�<��?       �	9��Xc�A�*

loss?��=�v�       �	��Xc�A�*

lossA��<�	�       �	��Xc�A�*

loss���=�B�l       �	o��Xc�A�*

loss�a<5%��       �	{��Xc�A�*

lossX=�=)7]0       �	l"�Xc�A�*

loss2�<q:�       �	��Xc�A�*

loss��<=�]�       �	�W�Xc�A�*

lossv
&<��!       �	���Xc�A�*

loss�q�=,��9       �	���Xc�A�*

lossZ��<�D�G       �	�'�Xc�A�*

loss���:�FT       �	���Xc�A�*

lossc=/��h       �	�k�Xc�A�*

loss�º<���       �	��Xc�A�*

loss|�i<�v�Q       �	���Xc�A�*

loss�@W=�%|�       �	�o��Xc�A�*

loss=<F�{�       �	R�Xc�A�*

loss��<��ǧ       �	c��Xc�A�*

loss�<�َ       �	�R�Xc�A�*

lossQI<<�:�.       �	���Xc�A�*

loss��]<���       �	c��Xc�A�*

loss�<���       �	�<�Xc�A�*

loss:j�<<�k�       �	���Xc�A�*

loss%�'=�K!�       �	�y�Xc�A�*

loss���=�)�       �	f�Xc�A�*

lossy�!=k1�       �	��Xc�A�*

loss8�;��       �	SZ��Xc�A�*

loss�в<Ipa�       �	����Xc�A�*

loss=%�=zk��       �	e���Xc�A�*

loss�=<���P       �	�'��Xc�A�*

lossn^�<�ɑo       �	����Xc�A�*

lossO��<C�       �	�Y��Xc�A�*

loss��=ׯ�N       �	3���Xc�A�*

lossP��<K�Y       �	
���Xc�A�*

loss&L�<�-��       �	�M��Xc�A�*

loss��+=j�sD       �	����Xc�A�*

loss�]	=�       �	���Xc�A�*

lossj�?=�A�	       �	�;��Xc�A�*

loss)��<���i       �	���Xc�A�*

loss(Ŕ<j�[�       �	Qh��Xc�A�*

lossC��<�e'       �	�-��Xc�A�*

loss�	�=�c	       �	U���Xc�A�*

lossVP=����       �	XY��Xc�A�*

loss�A�<�+*j       �	> �Xc�A�*

loss���;�,Z�       �	�� �Xc�A�*

loss=�I=�C�B       �	Gr�Xc�A�*

loss��;�0��       �	��Xc�A�*

loss�~y<n���       �	T��Xc�A�*

loss[1�<���!       �	�C�Xc�A�*

lossn�G<���       �	���Xc�A�*

loss�܎<�f�$       �	?q�Xc�A�*

loss��;��       �	�Xc�A�*

lossm��=��C#       �	]��Xc�A�*

loss�X<�et       �	�7�Xc�A�*

loss���<!�a       �	6��Xc�A�*

lossρ�= "       �	D��Xc�A�*

loss�2_<��"       �	r�Xc�A�*

loss��T<s�,       �	WC	�Xc�A�*

loss�a(={^+�       �	v�	�Xc�A�*

loss	�=�)��       �	~t
�Xc�A�*

loss���<DG�       �	d�Xc�A�*

loss�B�=~턒       �	���Xc�A�*

loss�=���       �	j3�Xc�A�*

loss�jB<#�$       �	���Xc�A�*

lossG4<7���       �	�v�Xc�A�*

lossU��;��       �	��Xc�A�*

loss(6<�r�       �	���Xc�A�*

loss�n=��D       �	`�Xc�A�*

loss��;Q��       �	?��Xc�A�*

lossI�<N��       �	t��Xc�A�*

loss���<���       �	�:�Xc�A�*

loss@��<	PX       �	E��Xc�A�*

loss�
�<O��E       �	r�Xc�A�*

loss�A/<�&��       �	��Xc�A�*

loss\S�;,4.o       �	��Xc�A�*

loss�,�<K�'8       �	t^�Xc�A�*

loss�W�;*��       �	~�Xc�A�*

loss_��<]��\       �	���Xc�A�*

loss2jG=�׎k       �	L3�Xc�A�*

loss�n<��Y�       �	���Xc�A�*

loss���;���       �	���Xc�A�*

losst�<��}       �	L�Xc�A�*

loss�l�<��A	       �	��Xc�A�*

loss���;%�3p       �	�V�Xc�A�*

loss���<���L       �	���Xc�A�*

loss\�<=�\k�       �		��Xc�A�*

lossg<���       �	�,�Xc�A�*

lossW5�;�|       �	���Xc�A�*

loss�Ň<�{��       �	j�Xc�A�*

loss]H�;ևL�       �	T�Xc�A�*

loss��=;�I6       �	���Xc�A�*

loss_Ʌ<`��       �	%<�Xc�A�*

loss���;E���       �	���Xc�A�*

lossn��<A�       �	��Xc�A�*

loss*��<�>��       �	1$ �Xc�A�*

loss�/6=�y�       �	� �Xc�A�*

loss�L= L��       �	If!�Xc�A�*

loss�pQ<���       �	� "�Xc�A�*

loss}N[=f���       �	_�"�Xc�A�*

loss�O,=��V       �	�C#�Xc�A�*

loss�ğ;k���       �	4$�Xc�A�*

loss�]U<�#�#       �	�$�Xc�A�*

loss���<Q��Y       �	�W%�Xc�A�*

lossC�<����       �	� &�Xc�A�*

loss��j:�l��       �	�&�Xc�A�*

loss�#�;Y���       �	�@'�Xc�A�*

loss�=�S*       �	l�'�Xc�A�*

loss��;��OQ       �	H�(�Xc�A�*

loss���;�	hx       �	�6)�Xc�A�*

loss��9=�(
       �	s�)�Xc�A�*

loss�y?=��-�       �	�*�Xc�A�*

loss�,<��       �	�L+�Xc�A�*

loss��&;�_�/       �	O�+�Xc�A�*

loss��y<���K       �	A�,�Xc�A�*

loss��_;�B�M       �	�"-�Xc�A�*

loss�/':�令       �	,�-�Xc�A�*

loss���<�ɞH       �	�`.�Xc�A�*

loss%�T;�y       �	�.�Xc�A�*

loss;@M;���       �	ߤ/�Xc�A�*

loss��;ܻ#	       �	:0�Xc�A�*

lossڋ�;�v�       �	��0�Xc�A�*

loss��7;�JFa       �	�f1�Xc�A�*

lossF;I��       �	�2�Xc�A�*

loss�k�:�r6�       �	f-3�Xc�A�*

lossI��8��+�       �	��3�Xc�A�*

loss��[=2Pք       �	sh4�Xc�A�*

loss�e8<��^�       �	�5�Xc�A�*

loss�;q�1       �	�5�Xc�A�*

loss��:����       �	T86�Xc�A�*

loss�Q<�މ�       �	�6�Xc�A�*

loss]�>V��       �	�f7�Xc�A�*

lossJ�;-��       �	u 8�Xc�A�*

loss]�!=�`�d       �	��8�Xc�A�*

loss��<�-�       �	�t9�Xc�A�*

loss,�f;���>       �	�:�Xc�A�*

loss��E;��       �	 �:�Xc�A�*

loss��(<>�ӂ       �	�W;�Xc�A�*

loss�&�<�_��       �	��;�Xc�A�*

loss�Q�<v�>+       �	N�<�Xc�A�*

loss1��;5l�       �	M2=�Xc�A�*

loss���<��ts       �	��=�Xc�A�*

loss��<�,�       �	Ef>�Xc�A�*

loss�=��       �	�?�Xc�A�*

loss`�= &��       �	T�?�Xc�A�*

loss
8�<��       �	Po@�Xc�A�*

loss�U=���       �	��A�Xc�A�*

loss�8J<FK׋       �	mB�Xc�A�*

loss���<��       �	�-C�Xc�A�*

loss�E�;}�Z]       �	��C�Xc�A�*

lossә=v�:       �	�zD�Xc�A�*

lossC_�;Ha�       �	�E�Xc�A�*

loss�<�nA�       �	R�E�Xc�A�*

loss}��<7���       �	VF�Xc�A�*

loss|%=��m       �	��F�Xc�A�*

lossOU�;D9N�       �	��G�Xc�A�*

loss]I7=joz       �	c�H�Xc�A�*

loss��:��T�       �	�DI�Xc�A�*

loss���<Z�	�       �	��I�Xc�A�*

lossT�7=�.:�       �	dxJ�Xc�A�*

loss�A�<ujX�       �	K�Xc�A�*

loss�wc=6X       �	�K�Xc�A�*

loss܆s;�˫       �	'KL�Xc�A�*

loss���<@�W�       �	��L�Xc�A�*

loss�cp=�N�X       �	�yM�Xc�A�*

lossZ�;�}VD       �	�N�Xc�A�*

loss�D<��?�       �	մN�Xc�A�*

loss�"<��       �	@LO�Xc�A�*

loss�d=�n�n       �	m�O�Xc�A�*

loss�g<���v       �	${P�Xc�A�*

loss�<���       �	�Q�Xc�A�*

loss���;U��       �	^�Q�Xc�A�*

lossQn�;n��       �	3TR�Xc�A�*

loss�:H=l"�/       �	��R�Xc�A�*

loss�	�<�
Q;       �	зS�Xc�A�*

loss�
�<(��       �	OT�Xc�A�*

loss�z;R�{       �	�T�Xc�A�*

loss�`=��lj       �	�U�Xc�A�*

lossh�<���N       �	�-V�Xc�A�*

loss$�p;�<Z       �	��V�Xc�A�*

lossx�=ޢ��       �	�iW�Xc�A�*

lossE;�;�A'9       �	X�Xc�A�*

lossLA=3y\�       �	��X�Xc�A�*

loss��=X�P�       �	��q�Xc�A�*

loss�=/���       �	Mr�Xc�A�*

loss��h=N�xT       �	��r�Xc�A�*

lossm�Y<����       �	��s�Xc�A�*

loss��<�f��       �	�&t�Xc�A�*

loss@@4<�[��       �	��t�Xc�A�*

loss<�ɣ�       �	"mu�Xc�A�*

loss�~�;�T?�       �	�v�Xc�A�*

loss횡=\�!�       �	��v�Xc�A�*

loss裣<�Dv(       �	�hw�Xc�A�*

loss*�x;Fmx�       �	x�Xc�A�*

loss<7�<�웁       �	��x�Xc�A�*

loss*��=�&�       �	��y�Xc�A�*

losse�==]�       �	a2z�Xc�A�*

lossE�k<��yD       �	��z�Xc�A�*

lossT��<"u��       �	�q{�Xc�A�*

loss�7�:$mz�       �	�|�Xc�A�*

lossd_�;�jķ       �	��|�Xc�A�*

loss�=jJ�l       �	��}�Xc�A�*

loss��=9pґ       �	�(~�Xc�A�*

loss�X�<E��       �	��~�Xc�A�*

loss���;�M/       �	j��Xc�A�*

lossQz�;��c       �	^*��Xc�A�*

loss��h=�$��       �	�ɀ�Xc�A�*

loss�l"<��o�       �	v���Xc�A�*

loss�֍<q���       �	����Xc�A�*

loss�=��T       �	����Xc�A�*

lossO��=��       �	oK��Xc�A�*

loss��<���I       �	���Xc�A�*

loss2=��!�       �	���Xc�A�*

loss�-�<���       �	�J��Xc�A�*

loss�#<�12O       �	�솼Xc�A�*

loss�{b<��pt       �	����Xc�A�*

loss�)=��       �	�K��Xc�A�*

loss\*c<�h        �	����Xc�A�*

loss��Y=���3       �	���Xc�A�*

loss�y;4���       �	�L��Xc�A�*

lossH�<�"!       �	�ꊼXc�A�*

loss��W=z~0�       �	����Xc�A�*

loss�V�=ҞX�       �	u?��Xc�A�*

loss6�<��85       �	�挼Xc�A�*

loss��=��m       �	]���Xc�A�*

loss���;k_o       �	I+��Xc�A�*

loss�OU<C�]e       �	>̎�Xc�A�*

loss��\;����       �	si��Xc�A�*

lossl8�<IN�       �	[	��Xc�A�*

losst�;��l       �	z���Xc�A�*

lossfK�;��f       �	,J��Xc�A�*

loss_�_;��;|       �	C摼Xc�A�*

loss���:�X       �	����Xc�A�*

loss�-�<��;       �	w+��Xc�A�*

loss��9=[%��       �	E�Xc�A�*

lossx��;)l�,       �	󐔼Xc�A�*

loss�`>=D^V       �	�4��Xc�A�*

losssG<ovV�       �	�ؕ�Xc�A�*

loss!�u<��<�       �	�|��Xc�A�*

loss�%l;�!+       �	L��Xc�A�*

loss��;l�S�       �	���Xc�A�*

loss��;3��       �	}^��Xc�A�*

loss�:�=^/(N       �	�H��Xc�A�*

loss\*G=R	Eh       �	�Xc�A�*

lossAP'<X��z       �	g���Xc�A�*

loss���:j,�       �	�?��Xc�A�*

loss�<<���#       �	蛼Xc�A�*

loss�!<���       �	}���Xc�A�*

loss��)<Y���       �	�5��Xc�A�*

loss��<W0a       �	�Xc�A�*

loss���<�^       �	'���Xc�A�*

loss\��<��P?       �	�A��Xc�A�*

lossoڡ<�S,�       �	�ߟ�Xc�A�*

loss�)n<M���       �	 |��Xc�A�*

loss_�=t�X|       �	���Xc�A�*

loss`��<,Fh^       �	á�Xc�A�*

loss�f�<�e�r       �	�]��Xc�A�*

loss gL<�erY       �	���Xc�A�*

loss�7�<|Jp�       �	z���Xc�A�*

loss|�<<��J       �	EG��Xc�A�*

loss�w;�'0       �	 ⤼Xc�A�*

lossDP;6��H       �	w���Xc�A�*

loss�c=�(�       �	!��Xc�A�*

losse`t<�x�3       �	����Xc�A�*

loss��C<�av       �	�d��Xc�A�*

lossWGO<���       �	|)��Xc�A�*

loss���;��Օ       �	�Ө�Xc�A�*

loss�{9</�       �	|��Xc�A�*

lossR�-=^�       �	�"��Xc�A�*

loss�d;����       �	XǪ�Xc�A�*

loss�!=J�9       �	�h��Xc�A�*

loss��e<�8��       �	���Xc�A�*

loss�+ =��~       �	K���Xc�A�*

loss�K=���       �	�Q��Xc�A�*

loss<�=d�       �	�쭼Xc�A�*

lossq�>=��       �	獮�Xc�A�*

loss�� =��       �	4��Xc�A�*

loss3F�=Z.��       �	$ѯ�Xc�A�*

lossX�;�3K�       �	;n��Xc�A�*

loss;ߖ<	       �	�h��Xc�A�*

loss�3==v�?       �	_	��Xc�A�*

loss1^�<�&6�       �	���Xc�A�*

loss�0[=R7��       �	�r��Xc�A�*

loss�<��       �	���Xc�A�*

lossd ;�lN       �	����Xc�A�*

loss���<<�       �	�=��Xc�A�*

loss�V�;���d       �	h浼Xc�A�*

lossd�=XC       �	n���Xc�A�*

losshM=m���       �	z��Xc�A�*

loss��;e
�       �	.��Xc�A�*

losssC�;
S��       �	�ø�Xc�A�*

lossa��<4 k       �	bj��Xc�A�*

loss���<T�       �	o��Xc�A�*

loss���=$3��       �	⮺�Xc�A�*

loss���;�(V=       �	�R��Xc�A�*

loss��<�/�j       �	����Xc�A�*

lossa0g;��~�       �	+���Xc�A�*

lossc�;��T�       �	�I��Xc�A�*

loss�B=�JE       �	JｼXc�A�*

loss�.<<)�x�       �	"ƾ�Xc�A�*

loss�5%=)�k       �	�q��Xc�A�*

loss@��<+5i       �	{��Xc�A�*

losste�=zڱo       �	����Xc�A�*

loss�t�<�*�       �	ø��Xc�A�*

loss��<���       �	��¼Xc�A�*

loss��<i��\       �	1ļXc�A�*

loss�X�<�* "       �	f.żXc�A�*

loss��;+��S       �	,�żXc�A�*

lossI�Q=sT\       �	-yƼXc�A�*

loss8�!=�M��       �	/ǼXc�A�*

loss~<�v�       �	��ǼXc�A�*

lossjé=��       �	�ZȼXc�A�*

loss7w�<�8\       �	kɼXc�A�*

loss�0�;�-�       �	$�ɼXc�A�*

loss�*S=�
��       �	9aʼXc�A�*

loss���<P�W       �	7�ʼXc�A�*

loss��a<�F�       �	U�˼Xc�A�*

loss�?=�yb       �	�:̼Xc�A�*

loss셛<~F��       �	��̼Xc�A�*

lossٹ;��       �	\rͼXc�A�*

loss�s�<7�3       �	�μXc�A�*

lossF��;P�O�       �	׽μXc�A�*

lossgǕ<�U��       �	.TϼXc�A�*

loss���<�XT       �	E�ϼXc�A�*

lossD6=���       �	?�мXc�A�*

loss�d�;�0�       �	|&ѼXc�A�*

loss���<�
�-       �	y�ѼXc�A�*

loss=���h       �	&�ҼXc�A�*

loss�<<���       �	`?ӼXc�A�*

lossdy�<��=�       �	��ӼXc�A�*

loss#x<R��[       �	]�ԼXc�A�*

loss=3�;�du�       �	�)ռXc�A�*

loss���=�	�Y       �	��ռXc�A�*

loss1 �<V��       �	�pּXc�A�*

lossS��<RQ�       �	׼Xc�A�*

loss���<���>       �	t�׼Xc�A�*

loss��y<Җ�       �	wMؼXc�A�*

lossLDA='�       �	-�ؼXc�A�*

loss��<u+�       �	��ټXc�A�*

loss�Z�<���       �	)ڼXc�A�*

loss�K0=�̡       �	Y�ڼXc�A�*

loss�0<�ʳ       �	�VۼXc�A�*

loss�!<�:�i       �	A�ۼXc�A�*

loss��q<ͧG$       �	ƇܼXc�A�*

loss��;���r       �	�@ݼXc�A�*

lossaLQ<[`��       �	�;޼Xc�A�*

loss��:�>�\       �	��޼Xc�A�*

loss���=x��	       �	�v߼Xc�A�*

loss���<�Ե       �	��Xc�A�*

lossT S<1V�       �	��Xc�A�*

loss�i�<�aR       �	�t�Xc�A�*

loss�m/<�d|�       �	.�Xc�A�*

loss��J;�(��       �	���Xc�A�*

loss�<G(^       �	�]�Xc�A�*

lossx�{<� l�       �	��Xc�A�*

losssb�<c1'�       �	F��Xc�A�*

loss�=�	�4       �	�:�Xc�A�*

loss�y ="[�       �	[B�Xc�A�*

loss�2�<3@hB       �	���Xc�A�*

loss�i�<��       �	�y�Xc�A�*

lossn�a=]U��       �	b�Xc�A�*

loss|vI=�ʴ       �	ѭ�Xc�A�*

loss��<���       �	�C�Xc�A�*

loss�H�:����       �	��Xc�A�*

losst��;�t�;       �	��Xc�A�*

loss]-�<�۩�       �	i�Xc�A�*

loss�-B<���       �	$B�Xc�A�*

loss�xv:��+5       �	���Xc�A�*

loss��<�/��       �	M���Xc�A�*

loss��=��       �	�F�Xc�A�*

loss
��<3^MP       �	��Xc�A�*

loss��A=���"       �	Wz�Xc�A�*

loss-��<��       �	��Xc�A�*

loss�E=iU#,       �	)��Xc�A�*

loss��=�#       �	h�Xc�A�*

loss�F�;�i.       �	��Xc�A�*

loss�0\<��z�       �	|��Xc�A�*

loss�k<~{�}       �	)A�Xc�A�*

loss)��<�f�L       �	���Xc�A�*

lossE�w<W��       �	����Xc�A�*

lossO��<;P�"       �	U2��Xc�A�*

loss�;[<����       �	����Xc�A�*

loss��*<v���       �	�t��Xc�A�*

loss��@<~hN�       �	���Xc�A�*

loss!��;8�I�       �	̶��Xc�A�*

loss =���       �	�T��Xc�A�*

lossS��;���       �	���Xc�A�*

loss���;��L       �	���Xc�A�*

loss��E<k4)�       �	�[��Xc�A�*

loss��=�_
       �	����Xc�A�*

loss��=(h�       �	���Xc�A�*

loss��
</���       �	�9��Xc�A�*

loss���<�o       �	����Xc�A�*

loss�7<s a�       �	G���Xc�A�*

loss,9�;��H�       �	�(��Xc�A�*

lossa�#<x�H�       �	D���Xc�A�*

lossd2�<��9I       �	����Xc�A�*

lossz�=:���       �	s- �Xc�A�*

loss��=	 �a       �	�� �Xc�A�*

loss�0=W�:^       �	�w�Xc�A�*

loss��=���o       �	X�Xc�A�*

loss��=+!�,       �	���Xc�A�*

loss���<�@�       �	J��Xc�A�*

loss<=(<�7w       �	 6�Xc�A�*

loss <&;^\��       �	,��Xc�A�*

lossZ��;S�*q       �	k~�Xc�A�*

lossTU�= A��       �		�Xc�A�*

loss��q<�($�       �	s��Xc�A�*

lossj�j<0�ze       �	Ze�Xc�A�*

loss�*>=3��       �	[�Xc�A�*

loss!ȳ<֏�g       �	>��Xc�A�*

loss7o�<�;~�       �	��	�Xc�A�*

lossaW�<���]       �	�^
�Xc�A�*

loss�:<���       �	��Xc�A�*

loss�,�<l��       �	���Xc�A�*

loss=ޟ=A;�%       �	�I�Xc�A�*

loss�ϩ<�?       �	u��Xc�A�*

loss��<���       �	Ǆ�Xc�A�*

lossL,=���       �	l&�Xc�A�*

loss�<��L       �	��Xc�A�*

loss�0=�ǐ�       �	�_�Xc�A�*

losso�"=�PN�       �	��Xc�A�*

losst��;Yۛ�       �	��Xc�A�*

loss��v<�6�>       �	\=�Xc�A�*

lossi�=��*       �	���Xc�A�*

loss;�>;�C T       �	&��Xc�A�*

loss_ј<�;�       �	S��Xc�A�*

loss��=<�j�       �	�L�Xc�A�*

loss�V�<�DN�       �	�9�Xc�A�*

loss�V�;�z�       �	���Xc�A�*

loss0=p�#s       �	4��Xc�A�*

loss �;+rsC       �	�!�Xc�A�*

lossP��<��5�       �	��Xc�A�*

loss��4=�81       �	X�Xc�A�*

loss��=��;�       �	C�Xc�A�*

loss� <o`~       �	ٴ�Xc�A�*

loss�+/<�7��       �	P�Xc�A�*

loss���;Uh�u       �	��Xc�A�*

lossߊ*<0� �       �	���Xc�A�*

lossx�p=�^�x       �	5`�Xc�A�*

lossd<Zj	�       �	���Xc�A�*

loss�d�<�)��       �	y��Xc�A�*

lossa
�<Zk&�       �	�)�Xc�A�*

loss(�F<����       �	&��Xc�A�*

loss�M<�@B       �	_a�Xc�A�*

loss�'n<�P��       �	eU �Xc�A�*

loss֕P=�B$p       �	�(!�Xc�A�*

loss��=a��e       �	U�!�Xc�A�*

loss|k<|��       �	�c"�Xc�A�*

lossۏ�<-"       �	��"�Xc�A�*

loss�6�<xhio       �	l�#�Xc�A�*

loss�%=0C�T       �	Z+$�Xc�A�*

loss��<ѭSk       �	~�$�Xc�A�*

lossTXF;�
��       �	N~%�Xc�A�*

loss�٪<�St       �	! &�Xc�A�*

lossGs<���       �	�&�Xc�A�*

loss�e=X[�6       �	>Y'�Xc�A�*

loss��8=�ݛ�       �	�'�Xc�A�*

loss*��<�O�       �	�(�Xc�A�*

loss}B�;�ٴ�       �	�3)�Xc�A�*

loss�?;t�/j       �	O�)�Xc�A�*

lossV�;f�ƥ       �	al*�Xc�A�*

loss�D=��`       �	�+�Xc�A�*

loss<�:�&`�       �	<�+�Xc�A�*

loss�k�;�k�
       �	�=,�Xc�A�*

lossJv<�>Y�       �	Q�,�Xc�A�*

loss{�:<��g       �	/�-�Xc�A�*

loss1�w=�*��       �	�F.�Xc�A�*

loss�w�<zx�       �	;�.�Xc�A�*

loss!��<��r�       �	��/�Xc�A�*

loss4Y<��       �	��0�Xc�A�*

loss{�<҇       �	�P1�Xc�A�*

loss�2;UVY�       �	E�1�Xc�A�*

loss�%;��I�       �	\�2�Xc�A�*

loss�<@>�       �	C 3�Xc�A�*

loss���;�Jښ       �	�3�Xc�A�*

loss�OT;�!       �	�d4�Xc�A�*

loss%��; m�F       �	5�Xc�A�*

loss�x<�
�P       �	Y�5�Xc�A�*

loss� >=#0U'       �	EK6�Xc�A�*

lossd_y=Vp       �	�6�Xc�A�*

loss}�=�շ�       �	K�7�Xc�A�*

loss{A3=��f       �	�J8�Xc�A�*

loss{y�<�R�a       �	}!9�Xc�A�*

loss�j�:�o+       �	e�9�Xc�A�*

lossH!�<�oU�       �	s:�Xc�A�*

loss�@�<��>       �	�;�Xc�A�*

loss�8�<�}��       �	6�;�Xc�A�*

lossq$M=�f;       �	�V<�Xc�A�*

loss7�;3�0A       �	��<�Xc�A�*

loss��=ê)       �	��=�Xc�A�*

loss�3�;�]��       �	76>�Xc�A�*

loss� <�μ�       �	�:?�Xc�A�*

loss���;��6i       �	��?�Xc�A�*

loss�G�<Q&ֳ       �	�h@�Xc�A�*

lossa��<���       �	�A�Xc�A�*

loss!�<�Vp�       �	�A�Xc�A�*

loss���;�]:�       �	�{B�Xc�A�*

loss�A�<�h�       �	�!C�Xc�A�*

loss�%<��;z       �	��C�Xc�A�*

lossj�<�U�       �	�lD�Xc�A�*

lossg��<�6       �	=
E�Xc�A�*

loss�/�;�i�|       �	$�E�Xc�A�*

loss�v�<�F�       �	)YF�Xc�A�*

loss���<J΋       �	�G�Xc�A�*

loss�@u<��E       �	!�G�Xc�A�*

loss�v�<Mlm       �	�PH�Xc�A�*

lossX.=ͪ:�       �	�OI�Xc�A�*

loss &�=��o       �	��I�Xc�A�*

lossn!=��       �	ӆJ�Xc�A�*

loss�1�<�uQ       �	� K�Xc�A�*

loss:�o<�֐       �	E�K�Xc�A�*

loss��=�$       �	ZdL�Xc�A�*

loss.�;�q�       �	�L�Xc�A�*

loss4�h<��        �	f�M�Xc�A�*

loss�p;U�D�       �	6ZN�Xc�A�*

loss�i�=R�g�       �	��N�Xc�A�*

loss��<���       �	�O�Xc�A�*

loss/�<���       �	�>P�Xc�A�*

loss��<�?�       �	x�P�Xc�A�*

loss�@�<�6       �	T�Q�Xc�A�*

loss���<\�\:       �	�'R�Xc�A�*

lossK;!^       �	��R�Xc�A�*

loss���<L4�       �	�^S�Xc�A�*

loss�9�;�7?�       �	��S�Xc�A�*

lossప;I�       �	!�T�Xc�A�*

loss��;�m�*       �	 V�Xc�A�*

lossW�=�O-
       �	��V�Xc�A�*

loss�@=\b@�       �	'jW�Xc�A�*

lossC,�;�|)       �	�1X�Xc�A�*

lossC��<��       �	��X�Xc�A�*

loss�)8=�\J       �	v�Y�Xc�A�*

loss�3=#�v�       �	:"Z�Xc�A�*

loss �1<P��       �	q�Z�Xc�A�*

loss���;��       �	ǁ[�Xc�A�*

loss*?=$��       �	p$\�Xc�A�*

loss�0F<��       �	G�\�Xc�A�*

loss�?%=.@m       �	�]]�Xc�A�*

loss�MP<Wx�       �	��]�Xc�A�*

loss�<���       �	��^�Xc�A�*

loss;�<�0=       �	�k_�Xc�A�*

lossev2=e"�       �		`�Xc�A�*

loss;�6;�Cp6       �	B�`�Xc�A�*

loss��8<�+�'       �	�ta�Xc�A�*

lossCc=�lL�       �	Ub�Xc�A�*

lossT��;0iAz       �	��b�Xc�A�*

loss���=*z_       �	�Lc�Xc�A�*

loss�C�=�¶�       �	��c�Xc�A�*

lossN�l="�9       �	�~d�Xc�A�*

loss
��<F�       �	�e�Xc�A�*

loss���;��I       �	 �e�Xc�A�*

lossr�<p�P       �	�f�Xc�A�*

loss�%k<H5#       �	Eg�Xc�A�*

loss>�<.}v�       �		Rh�Xc�A�*

loss�e�;��~�       �	��h�Xc�A�*

loss2�<L2%�       �	ӡi�Xc�A�*

lossL<]	a�       �	%<j�Xc�A�*

loss�۸;�K!�       �	h�j�Xc�A�*

loss��=���       �	�k�Xc�A�*

loss�]�<q���       �	�-l�Xc�A�*

loss��<�[��       �	��l�Xc�A�*

loss�= �Is       �	�fm�Xc�A�*

loss�B�=�HdQ       �	��m�Xc�A�*

lossI�<��"�       �	��n�Xc�A�*

loss���<;��       �	�)o�Xc�A�*

lossZ)<;nO�       �	��o�Xc�A�*

loss�<h}a       �	Qgp�Xc�A�*

loss�&�<��_       �	Vq�Xc�A�*

loss��Y=�٨       �	5�q�Xc�A�*

lossa�M<`��{       �	�[r�Xc�A�*

lossc,�<'
;�       �	��r�Xc�A�*

loss���<��@       �	іs�Xc�A�*

loss��
< ��       �	�8t�Xc�A�*

loss-�=[��       �	��t�Xc�A�*

loss�B�;5$��       �	�vu�Xc�A�*

loss0�:��v:       �	"v�Xc�A�*

loss�^=D�%       �	��v�Xc�A�*

loss��A=�ɼ�       �	�Ow�Xc�A�*

lossҩ�=0�       �	��w�Xc�A�*

loss}��:S�k       �	<�x�Xc�A�*

loss\�%;�Jǉ       �	<-y�Xc�A�*

loss�<���       �	�y�Xc�A�*

lossa�;�ay       �	�mz�Xc�A�*

loss���;*1��       �	{�Xc�A�*

loss暐;Ί�       �	*�{�Xc�A�*

loss�`<�ȡ       �	�~|�Xc�A�*

loss͉�; wIe       �	�}�Xc�A�*

loss3{$=p��Z       �	e~�Xc�A�*

loss���<��w       �	��~�Xc�A�*

loss2��<
9��       �	_�Xc�A�*

loss[��;P�Q�       �	���Xc�A�*

loss@�;6��       �	�逽Xc�A�*

loss-ڳ;M1��       �	���Xc�A�*

loss M�=��lV       �	-��Xc�A�*

loss��=�/g       �	|��Xc�A�*

lossd�;�E�       �	zT��Xc�A�*

loss�X9=�3P�       �	ԝ��Xc�A�*

loss���<��f�       �	gC��Xc�A�*

loss��+=4B�       �	���Xc�A�*

loss��i;O+g       �	�Xc�A�*

loss=�F<���]       �	�,��Xc�A�*

lossN=����       �	/���Xc�A�*

loss���<#�
       �	d���Xc�A�*

lossIh8=��g�       �	�H��Xc�A�*

loss��=��Gn       �	�Xc�A�*

loss�{�<߻��       �	X���Xc�A�*

lossL7=��C+       �	
.��Xc�A�*

loss��=�       �	aލ�Xc�A�*

loss%��<�$�       �	>z��Xc�A�*

lossS��;��       �	���Xc�A�*

loss@�c<N��       �	)���Xc�A�*

lossa��<Q�%�       �	 T��Xc�A�*

loss�5;�7�5       �	8�Xc�A�*

loss�U*<S���       �	B���Xc�A�*

lossa��<��H       �	�7��Xc�A�*

loss��</���       �	h撽Xc�A�*

loss�\=�i�       �	b���Xc�A�*

loss-;<�\(�       �	�#��Xc�A�*

loss$��<w��       �	]Ô�Xc�A�*

loss�}*<�EJ'       �	<f��Xc�A�*

lossnv<N(�b       �	���Xc�A�*

lossQ�;�*�       �	����Xc�A�*

loss�v�<�\       �	>>��Xc�A�*

loss���<K7�       �	�ٗ�Xc�A�*

loss\#�=o���       �	y��Xc�A�*

loss�*>�Dc       �	�<��Xc�A�*

loss��O=��Z       �	�v��Xc�A�*

loss�d�; �ɵ       �	���Xc�A�*

loss�œ<��z�       �	����Xc�A�*

loss
v�<:`��       �	Dn��Xc�A�*

loss�sa=��;       �	���Xc�A�*

loss�s�<����       �	Q���Xc�A�*

lossA�2=��<       �	%���Xc�A�*

loss�=��}       �	i���Xc�A�*

loss��<Gw�       �	�%��Xc�A�*

loss�e;��P       �	�Ƞ�Xc�A�*

lossX�<|b:�       �	�j��Xc�A�*

lossu	>a��       �	���Xc�A�*

lossd��<�t�S       �	���Xc�A�*

lossݲ�<��       �	�^��Xc�A�*

loss8��<��]�       �	F��Xc�A�*

loss�;v��        �	�̤�Xc�A�*

lossU�<�>i       �	�l��Xc�A�*

loss���<�G^`       �	���Xc�A�*

loss�A<#�j�       �	�Ħ�Xc�A�*

loss�
V;Cv��       �	�r��Xc�A�*

loss��<}_�       �	� ��Xc�A�*

loss�<U=�1ҵ       �	�ʨ�Xc�A�*

lossC�C<r=       �	�v��Xc�A�*

loss�-<�T%       �	���Xc�A�*

loss��0;�}�       �	Y���Xc�A�*

loss2ٟ<5_S       �	�g��Xc�A�*

loss@�:Ͷ�(       �	���Xc�A�*

loss�D=�L�       �	����Xc�A�*

loss�7<���       �	iR��Xc�A�*

loss�q<oyv�       �	����Xc�A�*

loss�mA=g@B       �	j���Xc�A�*

lossM�<¥ܞ       �	�F��Xc�A�*

loss:��;�u�       �	y寽Xc�A�*

loss�E�;~�R       �	8���Xc�A�*

loss�!<LR
       �	'��Xc�A�*

loss��G<d	��       �	�ű�Xc�A�*

loss�!=M5L       �	$b��Xc�A�*

loss��1=`y-�       �	���Xc�A�*

loss�`=�W~�       �	���Xc�A�*

lossg=$R7D       �	v4��Xc�A�*

lossho;5d:�       �	sش�Xc�A�*

loss��t=�[�8       �	l|��Xc�A�*

lossc�<#7�3       �	���Xc�A�*

loss�:�~2/       �	���Xc�A�*

lossx��<m�
�       �	�P��Xc�A�*

loss�!$<�IƄ       �	�췽Xc�A�*

loss\i�=Q�y       �	����Xc�A�*

loss*l�;X��       �	%��Xc�A�*

lossLċ;\�$       �	�ù�Xc�A�*

loss��;c��       �	J]��Xc�A�*

loss��m=c,ɽ       �	�>��Xc�A�*

losso&5=;�`�       �	4ػ�Xc�A�*

loss��<+[       �	}��Xc�A� *

loss�[�<�`�       �	*��Xc�A� *

loss���<�5�(       �	汽�Xc�A� *

loss
=a�3       �	�Q��Xc�A� *

loss��<<}�(       �	�龽Xc�A� *

loss�0�<X�       �	����Xc�A� *

loss�Cg;5�(       �	ca��Xc�A� *

loss�{�<�&��       �	���Xc�A� *

lossΕ=;R�|       �	_���Xc�A� *

loss4d�<�?0j       �	�V½Xc�A� *

loss=jO<����       �	AGýXc�A� *

loss��%<���f       �	��ĽXc�A� *

loss���:a���       �	��ŽXc�A� *

loss_�;!��       �	��ƽXc�A� *

loss�=yޖ�       �	F�ǽXc�A� *

loss\;A�*        �	�fȽXc�A� *

loss,H
<G~ԩ       �	�0ɽXc�A� *

loss|�
<��k�       �	'NʽXc�A� *

lossT��<��s2       �	��ʽXc�A� *

loss֛�=x6q}       �	�̽Xc�A� *

lossDG�<�Ht       �	cͽXc�A� *

loss��@=~�p       �	��ͽXc�A� *

lossE�;�c˺       �	W�νXc�A� *

lossJ�<<F+1       �	��ϽXc�A� *

lossB�=b��e       �	�CнXc�A� *

loss��T<f�B)       �	-ѽXc�A� *

loss���;��DM       �	�RҽXc�A� *

loss��:�̣8       �	OyӽXc�A� *

loss�5@=��       �	�ԽXc�A� *

lossҍf=���Z       �	�ԽXc�A� *

lossj <�k��       �	5yսXc�A� *

loss��=57+�       �	ֽXc�A� *

loss��=E3-       �	��ֽXc�A� *

loss�=j�Z�       �	�V׽Xc�A� *

loss;2�;`�A�       �	��׽Xc�A� *

loss-S.=�9�e       �	��ؽXc�A� *

lossш�<b��       �	LUٽXc�A� *

loss8�:2Ł2       �	��ٽXc�A� *

loss�ވ9k.�       �	Y�ڽXc�A� *

loss��;�Z�       �	�/۽Xc�A� *

loss��=M5ŝ       �	�۽Xc�A� *

loss�8<5	�       �	I�ܽXc�A� *

lossU�	=mF`K       �	�%ݽXc�A� *

lossF��9��P       �	�ݽXc�A� *

loss��F= ��       �	`޽Xc�A� *

loss�T�:��K�       �	�߽Xc�A� *

loss�9�}6�       �	�7�Xc�A� *

loss���9j�       �	��Xc�A� *

lossO^<��>�       �	�s�Xc�A� *

loss��G=��Q;       �	��Xc�A� *

loss��e;�'��       �	B��Xc�A� *

loss�@;�O�       �	�J�Xc�A� *

loss�>�;R�U       �	���Xc�A� *

lossC�c=q�v�       �	*��Xc�A� *

loss2�Y;���s       �	�!�Xc�A� *

loss]R
=d�S�       �	A��Xc�A� *

lossωl<购8       �	X�Xc�A� *

lossmĊ<e�n�       �	��Xc�A� *

lossw=w��       �	���Xc�A� *

loss�;"P��       �	$�Xc�A� *

loss��w=%>�       �	Q��Xc�A� *

loss���<�iߏ       �	�L�Xc�A� *

loss
;}<��=F       �	���Xc�A� *

loss�˄<�gM�       �	{�Xc�A� *

loss�G�<���       �	�E�Xc�A� *

loss�_�=c�8�       �	���Xc�A� *

loss�a%=�Q       �	w��Xc�A� *

loss���;�GE       �	���Xc�A� *

lossS�<	"3       �	�K�Xc�A� *

loss�h=�,��       �	���Xc�A� *

losss=<l7       �	�&�Xc�A� *

loss�ѣ<�S8�       �	���Xc�A� *

loss���<+B�       �	�v�Xc�A� *

loss��<�c4       �	��Xc�A� *

loss��;�)�       �	����Xc�A� *

loss�o�<���a       �	P6��Xc�A� *

loss!��:��v       �	h���Xc�A� *

loss���;h��       �	us��Xc�A� *

loss�$�:��s       �	���Xc�A� *

loss��;[n       �	����Xc�A� *

lossr�?=o�J$       �	�M��Xc�A� *

loss��<���       �	���Xc�A� *

loss_��<��ru       �	C���Xc�A� *

loss��f=I�$       �	)��Xc�A� *

lossnb�;QV��       �	����Xc�A� *

lossn%8<Ǎ�2       �	�a��Xc�A� *

lossI��;O��       �	"���Xc�A� *

lossHm;y�       �	����Xc�A� *

loss
�{<�n��       �	�9��Xc�A� *

lossahL;NUg)       �	#���Xc�A� *

lossS�F<L��6       �	v��Xc�A� *

loss)�'<���X       �	��Xc�A� *

lossʍ2=�ޕk       �	���Xc�A� *

lossY��<c�l�       �	�E �Xc�A� *

loss X�<}��       �	�-�Xc�A� *

loss}��;'�_�       �	��Xc�A� *

loss%�<0���       �	�|�Xc�A� *

lossq(p<T7"�       �	�,�Xc�A� *

loss��&<��Z       �	t��Xc�A� *

lossk�<��]9       �	7m�Xc�A� *

lossO��<~W�       �	d�Xc�A� *

loss�{�<C�       �	���Xc�A� *

loss���=�ȴ       �	b��Xc�A� *

loss���;��%N       �	�$�Xc�A� *

lossm��;�RlU       �	���Xc�A� *

lossZb)<A�"�       �	AG �Xc�A� *

loss8�<@C��       �	�!�Xc�A� *

loss\.�=�gV       �	��!�Xc�A� *

loss��<&���       �	#f"�Xc�A� *

lossl�=��+�       �	`#�Xc�A� *

loss�ߠ<`w/�       �	u=$�Xc�A� *

loss��;@��"       �	��$�Xc�A� *

loss��<����       �	�%�Xc�A� *

loss�=��       �	
.&�Xc�A� *

loss�p�<4SM�       �	��&�Xc�A� *

lossd��;ϑ`,       �	Cq'�Xc�A� *

lossqw�<�_�       �	�(�Xc�A� *

lossٸ�;8��        �	�(�Xc�A� *

loss$9�;����       �	�j)�Xc�A� *

loss:0�<�;�       �	%*�Xc�A� *

loss-9<AF?�       �	ʩ*�Xc�A� *

loss$��9��=v       �	��+�Xc�A� *

loss�;�v��       �	�b,�Xc�A� *

loss��<��]�       �	�-�Xc�A� *

loss��<(�A�       �	H�-�Xc�A� *

loss*; <1v��       �	�g.�Xc�A� *

loss��A<���       �	�/�Xc�A� *

loss R<��       �	��/�Xc�A� *

loss��S=�A       �	�r0�Xc�A�!*

loss��;�g}�       �	�1�Xc�A�!*

loss7qC;Z���       �	W�1�Xc�A�!*

loss:|�<�O�       �	�R2�Xc�A�!*

loss�<[M�       �	��2�Xc�A�!*

loss<�P       �	�3�Xc�A�!*

loss�x�<^d4u       �	dW4�Xc�A�!*

loss	��;fP�       �	v�4�Xc�A�!*

loss���:-��       �	E�5�Xc�A�!*

loss��l;{�M�       �	8k6�Xc�A�!*

lossC��<i�`       �	E)7�Xc�A�!*

loss3-�;���:       �	��7�Xc�A�!*

lossZ��<���       �	�j8�Xc�A�!*

lossN6><?�*       �	q9�Xc�A�!*

loss���<Y��       �	<�9�Xc�A�!*

lossA��=^��       �	T:�Xc�A�!*

loss��=]R��       �	<;�Xc�A�!*

loss��<���       �	�;�Xc�A�!*

loss�;+=d'�N       �	�<�Xc�A�!*

loss�̔=�;��       �	�=�Xc�A�!*

lossD��<�8�l       �	JC>�Xc�A�!*

loss�YW<�M��       �	t�>�Xc�A�!*

loss7u�=SX��       �	��?�Xc�A�!*

lossl�<ָmW       �	�w@�Xc�A�!*

lossD-B=-�K�       �	�A�Xc�A�!*

loss�E�;��p�       �	ƾA�Xc�A�!*

loss̯�;'�       �	hB�Xc�A�!*

loss{߂<��2       �	�C�Xc�A�!*

loss1�@<�U-       �	�C�Xc�A�!*

loss�TL:	��)       �	��D�Xc�A�!*

lossn<�=s�A=       �	0�E�Xc�A�!*

losso�><�9.Q       �	��F�Xc�A�!*

loss�,�</��D       �	�HG�Xc�A�!*

lossEG�:���       �	�H�Xc�A�!*

lossO�i;\z       �	��H�Xc�A�!*

loss��;���       �	�rI�Xc�A�!*

lossϖ<��       �	�#J�Xc�A�!*

loss[�w=���       �	.K�Xc�A�!*

loss�q�<�T�       �	��K�Xc�A�!*

loss��;�EO       �	T�L�Xc�A�!*

loss�V=?�"R       �	�3M�Xc�A�!*

lossC��;L��       �	b�M�Xc�A�!*

lossqT=<ߩ��       �	��N�Xc�A�!*

loss��<���       �	�QO�Xc�A�!*

lossdm�<�,�       �	V*P�Xc�A�!*

loss��=�[ҽ       �	[�P�Xc�A�!*

loss��<l>       �	X�Q�Xc�A�!*

lossFޅ</�Ӭ       �	 �R�Xc�A�!*

loss�W�<:5       �	ۉS�Xc�A�!*

loss���=���       �	�mT�Xc�A�!*

loss��&=9T@       �	��U�Xc�A�!*

loss�x<�:��       �	O;V�Xc�A�!*

loss��<�r�       �	��V�Xc�A�!*

loss��?<�U�       �	�xW�Xc�A�!*

lossS<D��       �	�,X�Xc�A�!*

loss���<�]�b       �	T�X�Xc�A�!*

loss���<��-�       �	�|Y�Xc�A�!*

loss�,�<���       �	�Z�Xc�A�!*

lossd�F<=P�A       �	��Z�Xc�A�!*

loss��e<�q�       �	�S[�Xc�A�!*

loss�چ:ȃ��       �	N\�Xc�A�!*

losstX�;a&j�       �	��\�Xc�A�!*

loss�s�<�E}       �	E]�Xc�A�!*

loss���;��i,       �	��]�Xc�A�!*

loss�J�<���(       �	�y^�Xc�A�!*

loss��<���X       �	�_�Xc�A�!*

loss.��;m �       �	.�_�Xc�A�!*

loss�=x��       �	�B`�Xc�A�!*

loss߆�<H�       �	va�Xc�A�!*

loss2V9=�9n       �	Q�a�Xc�A�!*

loss�w<@m��       �	�bb�Xc�A�!*

lossva�<p}�       �	c�Xc�A�!*

loss-��;�       �	D�c�Xc�A�!*

loss�T<<a�       �	Bd�Xc�A�!*

loss���<��y�       �	��d�Xc�A�!*

loss��<�pvl       �	b�e�Xc�A�!*

lossxz�;�Ɠ       �	*�f�Xc�A�!*

loss
�`;�UU       �	W_g�Xc�A�!*

lossw�;:�!�       �	��g�Xc�A�!*

loss�8�<�^R�       �	ԙh�Xc�A�!*

loss�P�;�       �	�5i�Xc�A�!*

loss�1<ݭ��       �	,�i�Xc�A�!*

loss�j�;��A,       �	��j�Xc�A�!*

loss��:On#       �	9k�Xc�A�!*

loss��&<�ى�       �	��k�Xc�A�!*

lossk��=(�F�       �	�wl�Xc�A�!*

loss�ϳ<��5       �	�m�Xc�A�!*

loss
�=�7       �	�n�Xc�A�!*

lossQ��<j�w       �	ˠn�Xc�A�!*

loss꾃<ס��       �	�8o�Xc�A�!*

loss��I;�|��       �	
�o�Xc�A�!*

lossa�<[��       �	�op�Xc�A�!*

loss>�;"jl       �	q�Xc�A�!*

loss�ݨ;���       �	��q�Xc�A�!*

lossT1�=���       �	�Lr�Xc�A�!*

lossl\x<���       �	��r�Xc�A�!*

loss�x<,�c       �	a�s�Xc�A�!*

loss���;o��       �	'jt�Xc�A�!*

loss䜎<ٗ�       �	u�Xc�A�!*

loss��;�\@       �	��u�Xc�A�!*

loss��t=��v{       �	`:v�Xc�A�!*

loss8�7<��dM       �	�v�Xc�A�!*

loss��j=HM��       �	�jw�Xc�A�!*

loss��<�
 �       �	m;x�Xc�A�!*

loss���;�� �       �	�x�Xc�A�!*

loss��<k�2       �	{hy�Xc�A�!*

loss��=�Erd       �	{z�Xc�A�!*

lossȀ;Y7m       �	��z�Xc�A�!*

loss2c�<���       �	�N{�Xc�A�!*

lossj�=�A       �	V|�Xc�A�!*

loss,�<Y�[       �	��|�Xc�A�!*

loss���<���       �	�B}�Xc�A�!*

loss�.�<�)w       �	��}�Xc�A�!*

lossO4<�HS�       �	@�~�Xc�A�!*

loss��A;-Ƽ       �	�*�Xc�A�!*

loss%T<�2Q�       �	���Xc�A�!*

loss��=�&�       �	7n��Xc�A�!*

lossz�D;A�U       �	6>��Xc�A�!*

loss3�;�D       �	�聾Xc�A�!*

loss���;ޛ6       �	����Xc�A�!*

loss��=�g0       �	�3��Xc�A�!*

lossM��<�U)�       �	�Ӄ�Xc�A�!*

loss1X�;�$!s       �	�}��Xc�A�!*

loss_A<h[�       �	d>��Xc�A�!*

loss�N�<�:�       �	�ⅾXc�A�!*

loss_��:�1�&       �	}ʆ�Xc�A�!*

loss�C�<SX�       �	�a��Xc�A�!*

loss���;r�       �	)��Xc�A�!*

lossF?f<oy@t       �	0��Xc�A�"*

lossF��<I��b       �	艾Xc�A�"*

loss�6�<3�B�       �	����Xc�A�"*

loss�x�<�,�       �	İ��Xc�A�"*

loss}�;��e       �	E���Xc�A�"*

losscu�;�1�V       �	a4��Xc�A�"*

loss�k =��       �	���Xc�A�"*

loss��<�U|       �	�厾Xc�A�"*

lossi]�<^�u�       �	�F��Xc�A�"*

loss�{�<�w�       �	�]��Xc�A�"*

loss�6$:�u�       �	X��Xc�A�"*

loss�_r<!R�       �	7���Xc�A�"*

lossqӌ;�!       �	:Y��Xc�A�"*

loss��=ou�o       �	
L��Xc�A�"*

lossWG�<��n       �	�0��Xc�A�"*

loss���9P`�       �	���Xc�A�"*

loss��<���*       �	\ɗ�Xc�A�"*

loss!ݟ<�iV       �	ё��Xc�A�"*

lossJ08<��       �	����Xc�A�"*

loss��3=�HV       �	���Xc�A�"*

lossz�5;~�a=       �	ߧ��Xc�A�"*

loss6�;(=D�       �	�O��Xc�A�"*

loss@�<��ӏ       �	?���Xc�A�"*

lossc�<�ޟ�       �	����Xc�A�"*

loss8?=?%�t       �	X<��Xc�A�"*

loss.�T=R�%       �	�ើXc�A�"*

loss]5=� P       �	!˟�Xc�A�"*

loss�Ō==��+       �	�y��Xc�A�"*

loss��^<�,�       �	���Xc�A�"*

loss3�;:%[ۥ       �	Aԡ�Xc�A�"*

loss���;&�`7       �	�s��Xc�A�"*

loss�:�;�m��       �	'��Xc�A�"*

loss�k�;T���       �	zģ�Xc�A�"*

losso�l<	�       �	sc��Xc�A�"*

loss���;�)�       �	���Xc�A�"*

loss�T�<�&�       �	����Xc�A�"*

loss��"<4�5�       �	�A��Xc�A�"*

lossh2=r_��       �	*㦾Xc�A�"*

lossE�=��hR       �	#���Xc�A�"*

loss��=�k�       �	x'��Xc�A�"*

loss8m =�3�H       �	��Xc�A�"*

loss�;_��h       �	����Xc�A�"*

loss�
<��       �	�1��Xc�A�"*

lossHe�:\J�       �	�Ϫ�Xc�A�"*

lossn��<7l       �	l��Xc�A�"*

lossċ@=	�0       �	�A��Xc�A�"*

loss�-<���       �	�ᬾXc�A�"*

loss���<��|P       �	���Xc�A�"*

loss6�W<�.K�       �	� ��Xc�A�"*

loss�!�<Ы$�       �	÷��Xc�A�"*

lossn�]=�4-       �	Gu��Xc�A�"*

loss�	<{B~       �	���Xc�A�"*

lossD�=qks!       �	����Xc�A�"*

loss��,<��i�       �	`V��Xc�A�"*

loss$ă;g�G�       �	A�Xc�A�"*

loss���=����       �	摲�Xc�A�"*

loss��#<I���       �	G;��Xc�A�"*

loss��;��J       �	Eٳ�Xc�A�"*

loss� �<�U�       �	�v��Xc�A�"*

loss�U�<�dA       �	���Xc�A�"*

lossג&;*w       �	���Xc�A�"*

loss|<�;� W�       �	Zb��Xc�A�"*

loss�=�z`�       �	���Xc�A�"*

lossJ=f���       �	ӽ��Xc�A�"*

lossw�=�|�       �	�]��Xc�A�"*

loss�y=����       �	����Xc�A�"*

loss73=����       �	[���Xc�A�"*

loss$��="�x�       �	4��Xc�A�"*

loss��b=���Y       �	9Ժ�Xc�A�"*

lossi��<�o��       �	|��Xc�A�"*

loss�F;���       �	���Xc�A�"*

loss��=ݙn�       �	(���Xc�A�"*

loss���<�5K       �	u��Xc�A�"*

loss $=�KZ#       �	��Xc�A�"*

loss�~[<�o@       �	l���Xc�A�"*

lossT�<���       �	�M��Xc�A�"*

loss�k�;S�}       �	���Xc�A�"*

lossݞ�:��K?       �	Ȕ��Xc�A�"*

loss� 	;K       �	`;��Xc�A�"*

lossf�;�L��       �	����Xc�A�"*

loss6��<X��       �	�}¾Xc�A�"*

lossz��<�0&�       �	PþXc�A�"*

loss�S�;J���       �	^�þXc�A�"*

lossM��<���O       �	aľXc�A�"*

loss�߹<��       �	7žXc�A�"*

loss��d=��W       �	��žXc�A�"*

lossSJ=o`ߟ       �	܃ƾXc�A�"*

lossaR�<%���       �	�'ǾXc�A�"*

loss�j2<���       �	��ǾXc�A�"*

loss�E<�ǵ�       �	FxȾXc�A�"*

lossX=R�^�       �	<ɾXc�A�"*

lossLt.=7�E       �	��ɾXc�A�"*

loss$��<�)��       �	eʾXc�A�"*

lossr��;���,       �	>˾Xc�A�"*

loss#7=Wu0[       �	��˾Xc�A�"*

loss�	q<�o�'       �	�S̾Xc�A�"*

loss3i<w<�0       �	��̾Xc�A�"*

loss��5<�9       �	��;Xc�A�"*

loss�Q�<��]       �	�4ξXc�A�"*

losswm�<Q�       �	g�ξXc�A�"*

loss=2�<<���       �	}ϾXc�A�"*

lossi{u<2Z?V       �	l$оXc�A�"*

loss�2#<��,�       �	��оXc�A�"*

loss8�n<i?e       �	-^ѾXc�A�"*

loss���<#l�{       �	�ҾXc�A�"*

loss�,�<�z��       �	��ҾXc�A�"*

lossz�Q<��֗       �	;sӾXc�A�"*

lossX}�<Z���       �	ԾXc�A�"*

loss�ϱ<�HT       �	��ԾXc�A�"*

loss��>;�|�<       �	mվXc�A�"*

loss,��;�!G&       �	{־Xc�A�"*

loss*�<�p<       �	�־Xc�A�"*

lossb�=ҭ�       �	V׾Xc�A�"*

loss�<W{��       �	 ؾXc�A�"*

loss3i�<x�4�       �	r�ؾXc�A�"*

loss��d=%���       �	?SپXc�A�"*

loss��<v�9*       �	��پXc�A�"*

loss��<?O       �	t�ھXc�A�"*

loss�o<p徖       �	�H۾Xc�A�"*

lossԀ�;��d�       �	��۾Xc�A�"*

loss{7�;+X.a       �	֏ܾXc�A�"*

losst�<���       �	�0ݾXc�A�"*

loss`�<�,��       �	��ݾXc�A�"*

loss�-8=?�`        �	lz޾Xc�A�"*

loss���<����       �	�R߾Xc�A�"*

loss�^�:�#)�       �	W]�Xc�A�"*

loss��#<��       �	���Xc�A�"*

loss��<����       �	���Xc�A�"*

loss�*=�\��       �	�@�Xc�A�#*

lossP�9a���       �	���Xc�A�#*

loss4�;q� ,       �	g~�Xc�A�#*

loss��?<��Q�       �	C�Xc�A�#*

loss�"Q<�Ml       �	��Xc�A�#*

lossƂp<��0�       �	h\�Xc�A�#*

lossv�<�f�       �	f��Xc�A�#*

lossV}N=�Rr       �	¡�Xc�A�#*

lossC)D=+`��       �	�G�Xc�A�#*

loss!�<�w�       �	���Xc�A�#*

loss�:`�7       �	 ��Xc�A�#*

loss��:e^��       �	�2�Xc�A�#*

loss�1=H>��       �	9��Xc�A�#*

loss�Ă:EE�z       �	�j�Xc�A�#*

lossB?;H�BK       �	�Xc�A�#*

losse�b<��u       �	���Xc�A�#*

lossl��=��!       �	�F�Xc�A�#*

lossTF�;��	N       �	v��Xc�A�#*

loss��<���       �	p}��Xc�A�#*

loss��=*�3       �	G �Xc�A�#*

loss�=^<iP��       �	��Xc�A�#*

lossh�V<2q��       �	�Z�Xc�A�#*

lossj�H;�бT       �	E��Xc�A�#*

lossYh<8I'       �	���Xc�A�#*

loss)M=�x�       �	�3�Xc�A�#*

loss��1<�,Rx       �	p
�Xc�A�#*

loss8LL=�W       �	z��Xc�A�#*

loss_�y;���       �	�Y�Xc�A�#*

loss�w;�z8�       �	���Xc�A�#*

lossA��;�۹       �	F���Xc�A�#*

loss�e�:�A��       �	&Q��Xc�A�#*

lossݘ�;MF       �	����Xc�A�#*

loss_�J<4�E       �	���Xc�A�#*

loss��h<'�       �	 D��Xc�A�#*

loss��$=0�       �	����Xc�A�#*

loss?��;���#       �	����Xc�A�#*

loss�c)=����       �	�*��Xc�A�#*

loss�b�<��I�       �	����Xc�A�#*

loss�4b<�ђ�       �	�l��Xc�A�#*

loss�Y=��Ɉ       �	$
��Xc�A�#*

loss}� <��N�       �	����Xc�A�#*

loss���<�$g       �	�M��Xc�A�#*

loss� �<:��%       �	V���Xc�A�#*

loss��e=���       �	ٔ��Xc�A�#*

lossn��<�16�       �	{2��Xc�A�#*

lossD�G<�c�       �	����Xc�A�#*

loss�v<
��       �	�q��Xc�A�#*

loss���<�/�5       �	� �Xc�A�#*

loss)��<��@       �	!� �Xc�A�#*

lossN�=g�       �	G�Xc�A�#*

loss&�<A`&       �	���Xc�A�#*

loss���<��L       �	ǂ�Xc�A�#*

lossfvC<D��B       �	��Xc�A�#*

loss�dg:�}_P       �	���Xc�A�#*

loss��f<�{_�       �	�k�Xc�A�#*

lossq��=]:�       �	��Xc�A�#*

loss�i;�*�e       �	��Xc�A�#*

loss��a<5�@e       �	�Y�Xc�A�#*

loss�)<B�ո       �	��Xc�A�#*

lossw�<K�C�       �	v��Xc�A�#*

loss��?<��R�       �	�&�Xc�A�#*

loss+P<
��       �	ƿ�Xc�A�#*

lossC�<�C��       �	-\	�Xc�A�#*

loss��<�q�f       �	��	�Xc�A�#*

loss�>�;���       �	x�
�Xc�A�#*

loss�v=�`G       �	d<�Xc�A�#*

lossrE�<K�u;       �	���Xc�A�#*

loss!ڕ<b���       �	V�Xc�A�#*

loss#�G<���       �	%�Xc�A�#*

loss���<�Vb&       �	B��Xc�A�#*

loss<<�<'��       �	:��Xc�A�#*

lossh�;�z.       �	4�Xc�A�#*

losst =��       �	�>�Xc�A�#*

loss��;涝�       �	��Xc�A�#*

loss��<.�l�       �	���Xc�A�#*

loss�<m�˻       �	!Z�Xc�A�#*

loss��+;�׊�       �	"��Xc�A�#*

loss�מ;oZ�e       �	/��Xc�A�#*

loss8D<a�{�       �	�L�Xc�A�#*

loss��;O+*;       �	� �Xc�A�#*

lossYR=��Z�       �	:�Xc�A�#*

loss�P�<�N�       �	O��Xc�A�#*

loss��;���]       �	�N�Xc�A�#*

loss�a�<���       �	���Xc�A�#*

loss��=a��       �	B��Xc�A�#*

loss�uY=Ux_�       �	�:�Xc�A�#*

loss�w=*���       �	]�Xc�A�#*

loss�w=௃�       �	���Xc�A�#*

losss��;�iL�       �	O�Xc�A�#*

loss���<�͋y       �	��Xc�A�#*

loss�&<���       �	
��Xc�A�#*

loss	��<����       �	��Xc�A�#*

loss�~�9qZ,3       �	���Xc�A�#*

loss�@�<�H��       �	�Q�Xc�A�#*

loss�>�;����       �	�*�Xc�A�#*

loss�C�<¦$n       �	��Xc�A�#*

lossN�<�h��       �	�_ �Xc�A�#*

loss�K<N1�       �	n� �Xc�A�#*

loss|2=I�
�       �	 �!�Xc�A�#*

loss�=�)       �	RG"�Xc�A�#*

losss��=�A       �	b�"�Xc�A�#*

lossv۶:����       �	��#�Xc�A�#*

lossX�<��n�       �	i5$�Xc�A�#*

loss��a=��t       �	��$�Xc�A�#*

loss)�;ҫ��       �	�j%�Xc�A�#*

loss��:��('       �	F�&�Xc�A�#*

lossCy�<ӿnC       �	�(�Xc�A�#*

loss>= '!       �	��(�Xc�A�#*

loss���<��U�       �	�;)�Xc�A�#*

loss��:'7�       �	&�)�Xc�A�#*

loss/i<�
�       �	�z*�Xc�A�#*

loss?��: ��Q       �	�*+�Xc�A�#*

loss�EW<��9       �	�+�Xc�A�#*

loss���<�wͦ       �	�\,�Xc�A�#*

loss�$`<]9
�       �	�-�Xc�A�#*

loss��<���a       �	b�-�Xc�A�#*

lossC�,<@��       �	$a.�Xc�A�#*

loss��`;Mc�       �	�/�Xc�A�#*

loss��;AW�       �	�/�Xc�A�#*

loss%�< 8��       �	L0�Xc�A�#*

lossq��<�t��       �	�0�Xc�A�#*

lossJ��<���F       �	(�1�Xc�A�#*

loss�I9<����       �	6#2�Xc�A�#*

loss�Z�<�X��       �	��2�Xc�A�#*

lossWH*;d,��       �	�{3�Xc�A�#*

loss^3=s|�       �	�4�Xc�A�#*

lossO�e<����       �	İ4�Xc�A�#*

loss7��<�tCe       �	X5�Xc�A�#*

loss��0<�65r       �	��5�Xc�A�$*

loss-��<;c��       �	q�6�Xc�A�$*

lossJ��<��]       �	�&7�Xc�A�$*

loss��a=��T       �	�7�Xc�A�$*

loss�8J=V���       �	[8�Xc�A�$*

loss쩚=Y��       �	��8�Xc�A�$*

loss��;[uy�       �	��9�Xc�A�$*

loss�
�=��fV       �	�+:�Xc�A�$*

loss/k=�E       �	��:�Xc�A�$*

loss3R<���X       �	k;�Xc�A�$*

lossY�<$�)�       �	A<�Xc�A�$*

lossz.�<�㠿       �	�<�Xc�A�$*

lossT#[<�Q�       �	dZ=�Xc�A�$*

loss��c<�{d�       �	E�=�Xc�A�$*

loss/
�<|ʉ        �	d�>�Xc�A�$*

loss�<����       �	�6?�Xc�A�$*

lossm֡=M�#w       �	��?�Xc�A�$*

loss���<��-       �	�i@�Xc�A�$*

lossU�=�X�       �	� A�Xc�A�$*

loss�*�<����       �	٘A�Xc�A�$*

lossĊ=p��       �	$�B�Xc�A�$*

lossH��;<�W       �	�C�Xc�A�$*

loss�[�:�	�s       �	�%D�Xc�A�$*

loss��Q=���       �	v�D�Xc�A�$*

loss� �<�]z       �	NdE�Xc�A�$*

loss�_�;F��       �	�F�Xc�A�$*

lossݎ�<�׻       �	��F�Xc�A�$*

loss��o<��0       �	�tG�Xc�A�$*

loss���;�nߝ       �	H�Xc�A�$*

loss�^�<��B)       �	��H�Xc�A�$*

loss�@�<��        �	YRI�Xc�A�$*

loss�"<��"       �	��I�Xc�A�$*

loss��]<A�a       �	ԛJ�Xc�A�$*

loss�ص=.a�       �	�ZK�Xc�A�$*

loss*�=�^       �	w�K�Xc�A�$*

loss�Ԡ<����       �	 �L�Xc�A�$*

lossN~�<�B�       �	�3M�Xc�A�$*

loss��;��ܿ       �	��M�Xc�A�$*

loss_��<B#��       �	�lN�Xc�A�$*

loss�r4=�0#�       �	�O�Xc�A�$*

loss��G<ij1�       �	�O�Xc�A�$*

loss�g<�c .       �	�DP�Xc�A�$*

loss��u<z�s       �	�P�Xc�A�$*

loss漀=ɛ+u       �	��Q�Xc�A�$*

loss�%�<�o2'       �	 )R�Xc�A�$*

loss��=,��       �	#�R�Xc�A�$*

loss�<���I       �	�fS�Xc�A�$*

loss��<?p.8       �	��S�Xc�A�$*

loss�J~<�[C�       �	l�T�Xc�A�$*

loss��T<�m��       �	�/U�Xc�A�$*

lossTӮ<F��A       �	 &V�Xc�A�$*

loss[�%;4��p       �	��V�Xc�A�$*

loss��&=��Y       �	obW�Xc�A�$*

losswU�;(�(       �	/�W�Xc�A�$*

lossb��;�1X       �	�X�Xc�A�$*

loss�q9;=��       �	�3Y�Xc�A�$*

loss�b�<j��&       �	��Y�Xc�A�$*

loss!+�;޸}z       �	�mZ�Xc�A�$*

loss{H�<���       �	�[�Xc�A�$*

loss��Y=O]�       �	d�[�Xc�A�$*

lossÊ;=Y��       �	�{\�Xc�A�$*

loss��[=xc�       �	�J]�Xc�A�$*

loss�	�:RY�[       �	n�]�Xc�A�$*

loss�w�=h}*a       �	P�^�Xc�A�$*

loss�O(<���       �	R_�Xc�A�$*

loss��<n
X�       �	��_�Xc�A�$*

loss�t<�&�       �	�a�Xc�A�$*

loss�k[;w�$       �	��a�Xc�A�$*

loss`u3:	�P*       �	�gb�Xc�A�$*

loss���:�9.       �	�c�Xc�A�$*

lossT�:;%�	�       �	e�c�Xc�A�$*

loss)�i<'y1�       �	$Fd�Xc�A�$*

lossw�$=��       �	2�d�Xc�A�$*

lossj&=��       �	]�e�Xc�A�$*

loss�<���       �	=)f�Xc�A�$*

lossH�;�o�       �	��f�Xc�A�$*

loss�*�==�xC       �	fg�Xc�A�$*

lossi�=�LU       �	`h�Xc�A�$*

loss�h�;]�K       �	��h�Xc�A�$*

loss	��<�)Sq       �	�9i�Xc�A�$*

loss��<Ȥ<       �	��i�Xc�A�$*

loss��-=3�B       �	�|j�Xc�A�$*

lossV�_<�^��       �	�k�Xc�A�$*

loss�v-<��B�       �	�l�Xc�A�$*

loss��v<ce�
       �	(�l�Xc�A�$*

loss�*L;~�       �	�Nm�Xc�A�$*

loss@�&<	���       �	��m�Xc�A�$*

loss�6;݂~9       �	�n�Xc�A�$*

lossF|<��z�       �	a4o�Xc�A�$*

lossZ	<�jT       �	�o�Xc�A�$*

lossݴh<^Nf       �	I�p�Xc�A�$*

loss�GZ=f�       �	�q�Xc�A�$*

lossD��< �
�       �	�q�Xc�A�$*

loss.v=���       �	h^r�Xc�A�$*

loss�N
<L'       �	��r�Xc�A�$*

loss*�<<d*��       �	�s�Xc�A�$*

loss�U�;�6�       �	h?t�Xc�A�$*

loss<� <h�	�       �	��t�Xc�A�$*

loss; )<e\X       �	w�u�Xc�A�$*

loss*ȝ;�xH�       �	ov�Xc�A�$*

loss�,(; ��q       �	�w�Xc�A�$*

loss0R<�}��       �	r�w�Xc�A�$*

loss��?<
�\       �	�ex�Xc�A�$*

loss��<OR�       �	��x�Xc�A�$*

loss��;�#_o       �	�y�Xc�A�$*

loss�n=հԴ       �	C9z�Xc�A�$*

loss��<Ŏ�v       �	��z�Xc�A�$*

lossoN�<\�*�       �	�l{�Xc�A�$*

lossV�&=�^x�       �	r|�Xc�A�$*

loss�yP=₱0       �	*�|�Xc�A�$*

lossN�D<r�       �	oe}�Xc�A�$*

loss@g9<3J#�       �	� ~�Xc�A�$*

loss�b�;��F�       �	(�~�Xc�A�$*

loss�	�;wmy       �	����Xc�A�$*

loss��;{�i�       �	�)��Xc�A�$*

loss���:�A�q       �	�Ɓ�Xc�A�$*

loss�D=|��|       �	�o��Xc�A�$*

loss���;��V�       �	���Xc�A�$*

lossےJ=@F�       �	p���Xc�A�$*

loss�dT<K���       �	W\��Xc�A�$*

loss�U<ɴ�5       �	�	��Xc�A�$*

loss���<��a�       �	�Ņ�Xc�A�$*

lossZٛ:!�m�       �	�g��Xc�A�$*

loss���;rj��       �	k��Xc�A�$*

lossL@�<!���       �	���Xc�A�$*

lossp�<mlr�       �	W[��Xc�A�$*

lossPc :��e[       �	R
��Xc�A�$*

lossק<K�}       �	3���Xc�A�$*

loss��9%`�s       �	�>��Xc�A�%*

lossd�;��N       �	�؊�Xc�A�%*

loss�@�;��L�       �	���Xc�A�%*

loss��B9.��       �	,G��Xc�A�%*

loss0ׁ;�K�5       �	�쌿Xc�A�%*

loss:�:"~-g       �	M���Xc�A�%*

loss�m�8�$��       �	���Xc�A�%*

loss���9xϭZ       �	����Xc�A�%*

lossb�;= �       �	�f��Xc�A�%*

loss̴�<��;       �	P��Xc�A�%*

loss�<���>       �	 ���Xc�A�%*

lossü:	�1�       �	;S��Xc�A�%*

loss���<T�IR       �	����Xc�A�%*

lossF�%=�c}       �	Q���Xc�A�%*

lossoC;���       �	.=��Xc�A�%*

lossڥh=Lr��       �	�ݓ�Xc�A�%*

loss��<9�g�       �	�v��Xc�A�%*

loss�[,=�;It       �	.��Xc�A�%*

loss���:B��       �	Ǖ�Xc�A�%*

lossx{�; Rw�       �	�a��Xc�A�%*

lossp�<�d��       �	����Xc�A�%*

loss�4�<-�D�       �	����Xc�A�%*

loss&�=�%�       �	0/��Xc�A�%*

loss�11<��ϭ       �	�ǘ�Xc�A�%*

loss��<��[       �	�`��Xc�A�%*

loss1UL=HӬ/       �	����Xc�A�%*

loss�k
<o'��       �	����Xc�A�%*

lossУ;;/��       �	6��Xc�A�%*

lossz̀<�Z�       �	̛�Xc�A�%*

loss�n=Y	�       �	s���Xc�A�%*

loss]><,J       �	���Xc�A�%*

lossv <�       �	����Xc�A�%*

loss�Or<���       �	"Q��Xc�A�%*

lossT6W;>E�       �	�瞿Xc�A�%*

loss�s\;�q�I       �	�ß�Xc�A�%*

loss�ՠ<��       �	;r��Xc�A�%*

loss�^�;�]��       �	j��Xc�A�%*

lossA!*<G�W       �	;ǡ�Xc�A�%*

loss>;��f       �	�`��Xc�A�%*

loss��w:v       �	
��Xc�A�%*

loss�,=	��       �	@���Xc�A�%*

lossH�<��K�       �	5D��Xc�A�%*

loss�Bj=(V�       �	�ۤ�Xc�A�%*

loss�D<�!x       �	�s��Xc�A�%*

lossYR<���       �	���Xc�A�%*

loss�=��w�       �	R��Xc�A�%*

lossҊ�<h�       �	���Xc�A�%*

lossL��<6U�(       �	����Xc�A�%*

loss��<��%4       �	y<��Xc�A�%*

loss?�J;�;�       �	ש�Xc�A�%*

lossZ<Zn4       �	{.��Xc�A�%*

loss8�==&���       �	�ͫ�Xc�A�%*

lossm��<*^L       �	?p��Xc�A�%*

loss��&;�d       �	w��Xc�A�%*

loss�;�k'       �	9���Xc�A�%*

loss�
�<�؏H       �	�Y��Xc�A�%*

loss���;��8       �	e���Xc�A�%*

loss1��<ź��       �	����Xc�A�%*

loss �2=��NM       �	!?��Xc�A�%*

loss^�=}E��       �	�ڰ�Xc�A�%*

loss���<�ت�       �	�v��Xc�A�%*

loss��<YJ8       �	9��Xc�A�%*

loss�Ft<�6��       �	���Xc�A�%*

loss��i<FK	(       �	O��Xc�A�%*

loss2>;���?       �	賿Xc�A�%*

loss|Ms<����       �	�οXc�A�%*

loss��=q�&       �	�pϿXc�A�%*

loss�Y�<�χx       �	]пXc�A�%*

loss�͛<6�44       �	:�пXc�A�%*

lossSkl<�SvU       �	,�ѿXc�A�%*

loss�&�;�j�       �	�ҿXc�A�%*

loss��=�N	v       �	F�ӿXc�A�%*

loss�L�=@�X�       �	�`ԿXc�A�%*

loss@W�<@VZ�       �	�&տXc�A�%*

lossH=;̦�        �	�:ֿXc�A�%*

lossQ�;C���       �	��ֿXc�A�%*

lossI<E0{0       �	�׿Xc�A�%*

loss�v<o��       �	�CؿXc�A�%*

lossqv<g�u       �	�ٿXc�A�%*

loss��<�=~'       �	1$ڿXc�A�%*

loss- =�I��       �	@�ڿXc�A�%*

loss���9{�+       �	��ۿXc�A�%*

lossVZ<��	�       �	�oܿXc�A�%*

loss'�=n���       �	k�ݿXc�A�%*

loss�6�<)	)�       �	rS޿Xc�A�%*

loss��<�cp       �	s�޿Xc�A�%*

lossv�<�88       �	r�߿Xc�A�%*

loss�N%<�1ު       �	�p�Xc�A�%*

loss]�=	�H       �	΍�Xc�A�%*

lossg@<J3<�       �	�D�Xc�A�%*

loss�v;G�=n       �	��Xc�A�%*

lossIP<U���       �	�
�Xc�A�%*

loss$�;����       �	��Xc�A�%*

loss�[�<%�       �	(e�Xc�A�%*

loss<��;q���       �	���Xc�A�%*

loss*@<�c��       �	e��Xc�A�%*

loss�O�;\w�       �	�d�Xc�A�%*

loss�ƍ<���       �	�T�Xc�A�%*

lossڿo=���4       �	�-�Xc�A�%*

loss<�;K��       �	y=�Xc�A�%*

loss�@�=u��        �	*�Xc�A�%*

lossڸ�;�^�       �	���Xc�A�%*

loss��3=:�       �	���Xc�A�%*

lossJ �=
�P�       �	����Xc�A�%*

loss�,=(ݩY       �	V��Xc�A�%*

lossS� <�ߖ�       �	�w�Xc�A�%*

lossI�<��       �	��Xc�A�%*

loss,�:$�9       �	6��Xc�A�%*

lossu�=韔�       �	�Q�Xc�A�%*

loss�_<˄>�       �	���Xc�A�%*

lossUԀ=��       �	e�Xc�A�%*

loss���;���       �	s��Xc�A�%*

loss�~H<Lb��       �	�`��Xc�A�%*

loss���<'|3       �	����Xc�A�%*

loss|Ԭ:�bEx       �	����Xc�A�%*

lossym�<���{       �	Hm��Xc�A�%*

loss���=;!+�       �	���Xc�A�%*

lossq1(;����       �	���Xc�A�%*

lossĤ-=cz       �	\r��Xc�A�%*

loss�2�<]�a�       �	���Xc�A�%*

loss��;L�{       �	����Xc�A�%*

loss(X�:Xs�       �	pB��Xc�A�%*

lossb�9b��       �	D���Xc�A�%*

lossm�=���       �	8���Xc�A�%*

loss![~=><�       �	�,��Xc�A�%*

loss}�i<zy\�       �	���Xc�A�%*

loss��'<9)oL       �	(b��Xc�A�%*

losslP=C}�       �	`��Xc�A�%*

loss%��<���       �	v���Xc�A�&*

loss�^;�$$�       �	$D��Xc�A�&*

lossW��:KS/�       �	����Xc�A�&*

loss:#�;���x       �	؁ �Xc�A�&*

loss��=�-�E       �	��Xc�A�&*

lossj\Q=%��       �	��Xc�A�&*

loss㋓<���       �	zR�Xc�A�&*

loss�w:<5n3�       �	���Xc�A�&*

lossy�=�v�       �	ҋ�Xc�A�&*

loss��=��D        �	t$�Xc�A�&*

lossA�<�q��       �	{��Xc�A�&*

loss� �<���k       �	�[�Xc�A�&*

loss��J<�"��       �	���Xc�A�&*

loss��B<�պ�       �	��Xc�A�&*

loss/q<N%�I       �	�q�Xc�A�&*

loss��Y<���       �	x	�Xc�A�&*

loss��n<��9       �	���Xc�A�&*

lossH%=��B�       �	�F	�Xc�A�&*

loss��8<��       �	��	�Xc�A�&*

lossJ<�"�       �	Y�
�Xc�A�&*

loss!��;)�z^       �	RH�Xc�A�&*

lossș<���       �	��Xc�A�&*

loss���<��	       �	F��Xc�A�&*

losst��<֎B�       �	�9�Xc�A�&*

lossw?�<�u       �	���Xc�A�&*

loss�7<w�W�       �	_��Xc�A�&*

lossS(�<��d0       �	X�Xc�A�&*

loss���<8o�       �	��Xc�A�&*

loss���<&U��       �	���Xc�A�&*

lossf�;���(       �	�0�Xc�A�&*

loss!�j<yn��       �	l��Xc�A�&*

loss�Z�<p�       �	"p�Xc�A�&*

loss2�;�g{r       �		�Xc�A�&*

lossR�<~҉r       �	J��Xc�A�&*

lossh�<����       �	DM�Xc�A�&*

loss��=7=       �	���Xc�A�&*

loss��<d��       �	��Xc�A�&*

loss��;�$�       �	�B�Xc�A�&*

loss�]�;|�`�       �	 ��Xc�A�&*

losstv�<"�F%       �	���Xc�A�&*

loss�U�;LL޺       �	'/�Xc�A�&*

lossS�<z���       �	���Xc�A�&*

lossMq�=��s�       �	�^�Xc�A�&*

loss -;=^���       �	v��Xc�A�&*

loss �<xE�       �	Ǡ�Xc�A�&*

loss�<0<����       �	RF�Xc�A�&*

loss�N<��S       �	G��Xc�A�&*

loss��<��3       �	v��Xc�A�&*

loss��}<K��       �	�&�Xc�A�&*

loss���<R�"�       �	<��Xc�A�&*

loss�I/;2���       �	���Xc�A�&*

loss�y�:�>�S       �	�:�Xc�A�&*

loss#�=	�o?       �	���Xc�A�&*

loss�'<��P�       �	�x �Xc�A�&*

loss*=L8(�       �	�C!�Xc�A�&*

loss���<��4       �	e�!�Xc�A�&*

lossvl7=�n<<       �	mt"�Xc�A�&*

loss�=Q< �U�       �	#�Xc�A�&*

loss��]<��1       �	ɫ#�Xc�A�&*

loss1��;�-_`       �	(E$�Xc�A�&*

loss��A<�6�q       �	��$�Xc�A�&*

lossI�Y;����       �	=�%�Xc�A�&*

loss���<-_��       �	
0&�Xc�A�&*

loss�4�<E���       �	��&�Xc�A�&*

loss$�M;��
       �	�'�Xc�A�&*

lossje]<�c4       �	�(�Xc�A�&*

loss��
=�
�       �	̷(�Xc�A�&*

loss�M�:�w^-       �	I�)�Xc�A�&*

loss�eJ<ĆƄ       �	r7*�Xc�A�&*

lossؒR<��       �	^�*�Xc�A�&*

loss$`�<^�Џ       �	�p+�Xc�A�&*

loss���=�/	�       �	%,�Xc�A�&*

loss�9w;��=       �	��,�Xc�A�&*

loss,�<�9iD       �	~;-�Xc�A�&*

lossD�_<�l�       �	g�-�Xc�A�&*

loss�_<%�       �	Pn.�Xc�A�&*

lossƺ	=w�       �	G/�Xc�A�&*

loss�L�;X�l       �	Ԝ/�Xc�A�&*

loss��<�`Y       �	"90�Xc�A�&*

loss!9e=��       �	�0�Xc�A�&*

loss]O1=�~       �	�c1�Xc�A�&*

loss��=_���       �	��1�Xc�A�&*

loss1d�;�1��       �	a�2�Xc�A�&*

loss�S4=20r        �	A3�Xc�A�&*

lossȑD;��#=       �	G�3�Xc�A�&*

loss�<��Ѻ       �	�4�Xc�A�&*

loss�)�=�p�       �	�85�Xc�A�&*

loss��N=8�U       �	�5�Xc�A�&*

loss��*=�;�8       �	�o6�Xc�A�&*

loss���=�z�-       �	�7�Xc�A�&*

loss(׊=&�{       �	d�7�Xc�A�&*

loss�T�<o2�"       �	�O8�Xc�A�&*

loss��<���`       �	j9�Xc�A�&*

loss���;pRE�       �	�9�Xc�A�&*

loss�G�<��       �	�B:�Xc�A�&*

loss�O?=~�s$       �	��:�Xc�A�&*

lossϣN<)��h       �	�w;�Xc�A�&*

loss[�p=w��       �	�<�Xc�A�&*

loss���;.��f       �	z�<�Xc�A�&*

loss�B�<�9l3       �	+�=�Xc�A�&*

loss��:�h��       �	A+>�Xc�A�&*

loss��<��       �	6�>�Xc�A�&*

loss��+=����       �	�q?�Xc�A�&*

lossy<���       �	@�Xc�A�&*

loss�+�<���o       �	^�@�Xc�A�&*

loss�z�;��`       �	�SA�Xc�A�&*

loss�(=��S�       �	f�A�Xc�A�&*

lossL��<�`��       �	S�B�Xc�A�&*

loss?��;�G�       �	�7C�Xc�A�&*

loss?�<%0�       �	I�C�Xc�A�&*

loss-|<܏ng       �	�D�Xc�A�&*

loss~-<�@�       �	iRE�Xc�A�&*

loss�[�<dO�Z       �	��E�Xc�A�&*

losss��<ؿ�       �	ǃF�Xc�A�&*

loss)��<9��C       �	�G�Xc�A�&*

lossV�<��H       �	��G�Xc�A�&*

lossDMa<<o��       �	�lH�Xc�A�&*

loss�=�:b�5�       �	�I�Xc�A�&*

loss{�e=���       �	еI�Xc�A�&*

loss8=�;�q�       �	�RJ�Xc�A�&*

loss(NK;�O�       �	|�J�Xc�A�&*

lossQ�W;��X�       �	��K�Xc�A�&*

loss��;3�F�       �	֐L�Xc�A�&*

loss1�<֓�       �	�tN�Xc�A�&*

loss -M=q���       �	]O�Xc�A�&*

loss��<�u�       �	^�O�Xc�A�&*

loss?�<�Q^�       �	2YP�Xc�A�&*

loss���=W��       �	��P�Xc�A�&*

lossn��<"��7       �	�Q�Xc�A�'*

loss���<c���       �	U3R�Xc�A�'*

loss��;q&��       �	|�R�Xc�A�'*

loss��:�#	�       �	�pS�Xc�A�'*

lossqO�<�#/b       �	�T�Xc�A�'*

loss �=ٗ��       �	ݱT�Xc�A�'*

loss��w<P	*�       �	MMU�Xc�A�'*

lossm�<�O�C       �	�U�Xc�A�'*

loss*��;+�i       �	j�V�Xc�A�'*

lossE8�<��       �	yZW�Xc�A�'*

loss�vM;�伧       �	��W�Xc�A�'*

loss�lB<.w��       �	O�X�Xc�A�'*

loss���;�܃       �	�2Y�Xc�A�'*

loss�@�;*吙       �	�Y�Xc�A�'*

loss.g�;9i�       �	eZ�Xc�A�'*

lossƧ�<zu��       �	 [�Xc�A�'*

loss�^<f�D`       �	/�[�Xc�A�'*

loss��<�N0A       �	�[\�Xc�A�'*

loss���<���4       �	��\�Xc�A�'*

lossVSH<��}�       �	p�]�Xc�A�'*

loss���;!��       �	4/^�Xc�A�'*

loss�l�<��3�       �	R�^�Xc�A�'*

loss��|;�-       �	�s_�Xc�A�'*

loss�Y2=���_       �	(`�Xc�A�'*

loss�X=fm�       �	�`�Xc�A�'*

loss&%�<]�s�       �	5Aa�Xc�A�'*

loss(�=�d��       �	`�a�Xc�A�'*

lossE�-=Y���       �	�b�Xc�A�'*

lossJ�=J���       �	�\c�Xc�A�'*

loss쭪<��       �	��c�Xc�A�'*

loss(8�;�C;�       �	L�d�Xc�A�'*

loss��<÷7       �	^/e�Xc�A�'*

lossJ�L=�܂�       �	u�e�Xc�A�'*

loss*X@=|�>       �	+�f�Xc�A�'*

loss��<��[       �	O;g�Xc�A�'*

loss�֍<ۢ�:       �	H�g�Xc�A�'*

loss	��<>>v       �	t�h�Xc�A�'*

loss+� <$\e�       �	�ki�Xc�A�'*

loss�� <��a       �	�j�Xc�A�'*

loss�&S<n8�       �	ٴj�Xc�A�'*

lossr_<�q��       �	�Rk�Xc�A�'*

loss-�=" `       �	t)l�Xc�A�'*

loss܌#<j�Ѣ       �	��l�Xc�A�'*

loss��'=���       �	�em�Xc�A�'*

loss��<�E�       �	��m�Xc�A�'*

loss̀�<x:��       �	�n�Xc�A�'*

loss�d;=5�V�       �	�Eo�Xc�A�'*

loss&�7=Nx�Q       �	��o�Xc�A�'*

loss�%�;���       �	��p�Xc�A�'*

loss��0;w��       �	�0q�Xc�A�'*

loss[P=����       �	�q�Xc�A�'*

lossL�<�0�       �	S{r�Xc�A�'*

losso�=dR��       �	�$s�Xc�A�'*

loss�}<��	�       �	y�s�Xc�A�'*

loss�*�<<.:�       �	��t�Xc�A�'*

lossъ�<��       �	��u�Xc�A�'*

loss��#<�       �	C<v�Xc�A�'*

loss��==bf�       �	s�v�Xc�A�'*

loss��Y<��u       �	E�w�Xc�A�'*

loss��O;ސz�       �	`"x�Xc�A�'*

lossx��<��X       �	��x�Xc�A�'*

loss�-Z=+o��       �	<�y�Xc�A�'*

loss�;�O<       �	<z�Xc�A�'*

loss�c�;�[{�       �	��z�Xc�A�'*

loss�,h<$�a       �	��{�Xc�A�'*

loss�-b<gh�       �	�|�Xc�A�'*

lossl6<���       �	�|�Xc�A�'*

loss�I�=�U�       �	l}�Xc�A�'*

loss���<8���       �	�~�Xc�A�'*

lossc.C<=��       �	��~�Xc�A�'*

lossSoC<��f       �	�d�Xc�A�'*

loss�%�;���       �	�w��Xc�A�'*

loss�w<
W;�       �	���Xc�A�'*

loss�3�<��~�       �	$���Xc�A�'*

loss��W<l-��       �	�L��Xc�A�'*

loss/=��_D       �	u��Xc�A�'*

loss�A<:5z       �	����Xc�A�'*

loss�b�<]��&       �	"��Xc�A�'*

loss/=y&��       �	���Xc�A�'*

loss��M;
1l       �	U��Xc�A�'*

loss��=W�IO       �	���Xc�A�'*

loss��<���       �	����Xc�A�'*

loss,��;�
�       �	[���Xc�A�'*

loss��5=v�:       �	�Q��Xc�A�'*

loss�y�=�(��       �	����Xc�A�'*

loss�ߜ;�m|,       �	O���Xc�A�'*

loss��N;��p       �	rj��Xc�A�'*

lossSJ=�BI       �	 ��Xc�A�'*

loss:�r=L��f       �	1?��Xc�A�'*

lossx֜;��d�       �	Z׌�Xc�A�'*

loss���;�*�a       �	�)��Xc�A�'*

loss��=RH�       �	���Xc�A�'*

lossD��<r �       �	����Xc�A�'*

lossX=���<       �	t|��Xc�A�'*

loss�9�<�8V�       �	;��Xc�A�'*

loss�փ<��?-       �	���Xc�A�'*

loss�<����       �	�j��Xc�A�'*

lossH�;P n       �	`��Xc�A�'*

loss���:�NT�       �	���Xc�A�'*

loss��;�a�       �	�H��Xc�A�'*

loss��=�s�       �	���Xc�A�'*

lossD:W;�9��       �	Â��Xc�A�'*

loss��<���       �	�!��Xc�A�'*

loss��=�d�"       �	d��Xc�A�'*

loss�0X<�r�T       �	e×�Xc�A�'*

lossP��<�W�0       �	F[��Xc�A�'*

loss��=�㔻       �	� ��Xc�A�'*

loss�A=���Y       �	����Xc�A�'*

loss�4�<�yG       �	�G��Xc�A�'*

loss$z�<#��       �	@ܚ�Xc�A�'*

loss73<����       �	vp��Xc�A�'*

loss��I<[�e,       �	4��Xc�A�'*

loss��=%��       �	nݜ�Xc�A�'*

lossZ<�<iW�=       �	lz��Xc�A�'*

lossa"-=��H�       �	�"��Xc�A�'*

loss	��;O��       �	r��Xc�A�'*

loss��=d�kS       �	f��Xc�A�'*

loss�8�<����       �	ᵠ�Xc�A�'*

loss���<Fa��       �	�N��Xc�A�'*

loss!��<�6�m       �	����Xc�A�'*

loss�h�<;_��       �	p���Xc�A�'*

lossτ�=p矧       �	�*��Xc�A�'*

loss��9<B�M�       �	 ʣ�Xc�A�'*

loss8Ȭ;�S7       �	9b��Xc�A�'*

loss�Q=�lFE       �		���Xc�A�'*

loss$��;y"�e       �	ԥ�Xc�A�'*

loss�&><PG~       �	�j��Xc�A�'*

loss��o<�f7       �	S��Xc�A�'*

loss��b=l�c        �	Yħ�Xc�A�(*

loss�a=��a        �	�e��Xc�A�(*

loss*4R=?��K       �	����Xc�A�(*

loss��B<���       �	ꔩ�Xc�A�(*

loss�H<��?]       �	�:��Xc�A�(*

loss���<����       �	0֪�Xc�A�(*

loss��<PW�m       �	>v��Xc�A�(*

loss��;T:��       �	��Xc�A�(*

lossa�;G�t       �	禬�Xc�A�(*

lossnx�<�&�H       �	�T��Xc�A�(*

lossn�I=b���       �	���Xc�A�(*

lossj��<t��       �	)���Xc�A�(*

loss��;��0�       �	�,��Xc�A�(*

loss���:.R�       �	:ί�Xc�A�(*

loss�.�:�	       �	�l��Xc�A�(*

lossf�,=���       �	^��Xc�A�(*

lossD��;B�R�       �	O���Xc�A�(*

loss\f=�Y�       �	yW��Xc�A�(*

loss�s<ų�       �	���Xc�A�(*

lossn�f<:��       �	����Xc�A�(*

lossb�:���       �	H��Xc�A�(*

lossoQ;I4S       �	X ��Xc�A�(*

loss?<���@       �	ܞ��Xc�A�(*

loss!�<�(��       �	�J��Xc�A�(*

loss��;�3=�       �	4��Xc�A�(*

loss���<����       �	x���Xc�A�(*

lossl��<�ʍ       �	J��Xc�A�(*

lossI��;'&��       �	����Xc�A�(*

loss���<W�b�       �	���Xc�A�(*

loss���<ů��       �	�.��Xc�A�(*

loss��=�'��       �	_κ�Xc�A�(*

loss@�h;�Q��       �	�k��Xc�A�(*

loss� <ԉ��       �	!��Xc�A�(*

loss�t�<�=�       �	����Xc�A�(*

loss z�;B���       �	<��Xc�A�(*

loss�<�b=       �	T���Xc�A�(*

loss��	<<'i.       �	흾�Xc�A�(*

loss��5;8��W       �	�?��Xc�A�(*

loss���;�D       �	�ڿ�Xc�A�(*

loss\U&<���       �	,���Xc�A�(*

loss�E)=���       �	���Xc�A�(*

loss���<�V       �	>���Xc�A�(*

loss�&�<_3�w       �	-[��Xc�A�(*

loss ¯;Ԭ�       �	�e��Xc�A�(*

loss[� >'��       �	<��Xc�A�(*

lossϦX<�(�?       �	���Xc�A�(*

loss :�<�]�M       �	�H��Xc�A�(*

loss���<�u�8       �	����Xc�A�(*

loss�|�;`�       �	����Xc�A�(*

loss@�"=��3       �	�&��Xc�A�(*

loss=��;tȰ       �	���Xc�A�(*

loss��<cV�       �	%���Xc�A�(*

lossa�I=,Q       �	���Xc�A�(*

loss��;l�
       �	�2��Xc�A�(*

loss�CH<�h^�       �	�f��Xc�A�(*

loss��-=5ӛ�       �	���Xc�A�(*

loss�	�<���       �	���Xc�A�(*

loss�9�<����       �	����Xc�A�(*

loss*3X<���       �	�R��Xc�A�(*

loss8��</�Kx       �	�t��Xc�A�(*

loss�|</�T       �	���Xc�A�(*

loss@y<�9�       �	����Xc�A�(*

loss��=��i       �	"n��Xc�A�(*

lossD	<�	�       �	�w��Xc�A�(*

loss���;?���       �	���Xc�A�(*

loss�ܐ;S�?)       �	M���Xc�A�(*

losso��<F��       �	*���Xc�A�(*

lossW��<���,       �	�5��Xc�A�(*

loss��0=�(a       �	5���Xc�A�(*

loss��p<��9�       �	Xp��Xc�A�(*

loss���;sf[�       �	!��Xc�A�(*

lossXۀ;m���       �	9���Xc�A�(*

loss\��;�@�       �	wK��Xc�A�(*

loss���;�~�=       �	4���Xc�A�(*

loss3G<�a��       �	����Xc�A�(*

loss�Ԉ<�<�       �	�_��Xc�A�(*

loss��2=�+�       �	����Xc�A�(*

losso
:�Yx{       �	P���Xc�A�(*

loss�N;1���       �	_@��Xc�A�(*

loss��c;VA}�       �	����Xc�A�(*

loss���<�^       �	ׅ��Xc�A�(*

loss_��:�K[       �	B]��Xc�A�(*

loss��L<���$       �	���Xc�A�(*

loss��}<�q�       �	���Xc�A�(*

lossT�><���       �	�q��Xc�A�(*

losswn;=�B+�       �	P��Xc�A�(*

loss�b|<ᦢ�       �	G���Xc�A�(*

loss��<��x�       �	�w��Xc�A�(*

loss��q=q;��       �	$
��Xc�A�(*

loss��<�3`�       �	����Xc�A�(*

loss��=!���       �	>��Xc�A�(*

loss��<=rjD       �	���Xc�A�(*

loss*�}=М�       �	����Xc�A�(*

loss}8�<8}\       �	����Xc�A�(*

loss�8�=�NX       �	�D��Xc�A�(*

loss��<=<�       �	w���Xc�A�(*

losseX�<�j~�       �	�~��Xc�A�(*

lossݠ�:�J       �	 ��Xc�A�(*

loss��};?1}�       �	���Xc�A�(*

loss]33<�}��       �	?R��Xc�A�(*

loss���<!�       �	����Xc�A�(*

loss��
<�:]:       �	0���Xc�A�(*

loss8��=��       �	4H��Xc�A�(*

loss��K<�M       �	����Xc�A�(*

loss�m_=1'�       �	܂��Xc�A�(*

loss�Y�<�qê       �	'2��Xc�A�(*

lossJi�<�^       �	����Xc�A�(*

loss���:�!�       �	mu��Xc�A�(*

loss���< �       �	���Xc�A�(*

loss��&<�@�C       �	@���Xc�A�(*

loss�s�;K��'       �	RE��Xc�A�(*

lossׂ<]�b       �	����Xc�A�(*

loss�/*<����       �	Z���Xc�A�(*

loss1;<�	�       �	 ��Xc�A�(*

loss'�<W�p�       �	���Xc�A�(*

loss���<y�x�       �	����Xc�A�(*

loss\�;���       �	�)��Xc�A�(*

loss���;4�.       �	B���Xc�A�(*

lossrŁ=Txk       �	�s��Xc�A�(*

lossJ�<i�       �	���Xc�A�(*

lossɛ<I���       �	����Xc�A�(*

loss�	�<':�       �	Q��Xc�A�(*

loss�<%��1       �	����Xc�A�(*

loss��<s#g�       �	g���Xc�A�(*

loss���<�, �       �	l>��Xc�A�(*

lossȉ.=�       �	\���Xc�A�(*

loss���<w)��       �	�~��Xc�A�(*

losswG�= �?�       �	F"��Xc�A�(*

lossZC<O		3       �	���Xc�A�)*

loss_:h<�pcs       �	�e��Xc�A�)*

loss�֦<�ח       �	_��Xc�A�)*

loss�g={!.�       �	����Xc�A�)*

loss��<Ċ~�       �	�G �Xc�A�)*

loss}��:�� �       �	�� �Xc�A�)*

loss��8<W(�       �	�Xc�A�)*

lossX�=����       �	~�Xc�A�)*

loss|�M<q� �       �	ȴ�Xc�A�)*

lossW�;�֣s       �	�N�Xc�A�)*

loss�H�<qx       �	���Xc�A�)*

losss0;���k       �	||�Xc�A�)*

lossO�J<k��z       �	n�Xc�A�)*

lossX�;;R�{�       �	u��Xc�A�)*

loss���;q[�       �	�R�Xc�A�)*

loss�F�<f���       �	� �Xc�A�)*

loss��<i�2�       �	ʨ�Xc�A�)*

loss{��;���<       �	Wx�Xc�A�)*

lossi~=�o�E       �	�	�Xc�A�)*

lossA��<�U��       �	z�	�Xc�A�)*

loss��s;�5�       �	��
�Xc�A�)*

loss�J�<�!{"       �	�A�Xc�A�)*

loss��;x�jK       �	Y��Xc�A�)*

loss$I=5��e       �	>��Xc�A�)*

loss���;��0       �	ZE�Xc�A�)*

loss4�<�(��       �	��Xc�A�)*

lossx�<�%       �	dy�Xc�A�)*

loss�X9;$+��       �	U�Xc�A�)*

lossϳ�:DPs       �	���Xc�A�)*

loss�^_;����       �	UK�Xc�A�)*

lossLo�<�W       �	!��Xc�A�)*

loss��	<�`P+       �	�~�Xc�A�)*

loss��}<x�;Z       �	X�Xc�A�)*

lossv�G=���O       �	���Xc�A�)*

loss�`<�vN       �	F]�Xc�A�)*

lossI��<�OA       �	!�Xc�A�)*

loss���:���       �	���Xc�A�)*

lossqG=؉�^       �	�G�Xc�A�)*

loss�m;g�;'       �	e��Xc�A�)*

loss6@�;�[	       �	[��Xc�A�)*

losse�<�]x       �	�3�Xc�A�)*

loss@�]<��ç       �	p��Xc�A�)*

loss�se<�I��       �	^f�Xc�A�)*

lossx�;�� �       �	���Xc�A�)*

loss Jf=�ۦ;       �	)��Xc�A�)*

loss+�:c-��       �	�,�Xc�A�)*

lossf�<�)��       �	P��Xc�A�)*

loss?߇;�m|�       �	�]�Xc�A�)*

loss�=��       �	���Xc�A�)*

loss�_g<�V��       �	���Xc�A�)*

loss�{><*'�L       �	*�Xc�A�)*

loss��;rFZ       �	3��Xc�A�)*

loss�0I=�H�       �	�_�Xc�A�)*

loss��K=.ī       �	R_�Xc�A�)*

lossdO�=N�S�       �	� �Xc�A�)*

loss��;���       �	� �Xc�A�)*

loss�~	<�3�c       �	`V!�Xc�A�)*

lossi=�2��       �	K"�Xc�A�)*

loss �4<_�	       �	��"�Xc�A�)*

loss\|8<�LbF       �	`;#�Xc�A�)*

loss�/�;c��       �	</$�Xc�A�)*

loss��;�VH-       �	��$�Xc�A�)*

loss��<Cp�:       �	��%�Xc�A�)*

loss͜�;Sp*�       �	'&�Xc�A�)*

loss���<�-       �	 �&�Xc�A�)*

loss1�e;L�Ɏ       �	�a'�Xc�A�)*

loss�5�;K�p�       �	]�'�Xc�A�)*

loss�f�<K	�       �	ϟ(�Xc�A�)*

loss�K[<J	�       �	?7)�Xc�A�)*

lossC�=����       �	��)�Xc�A�)*

loss| �;�E�s       �	��*�Xc�A�)*

loss���<�+��       �	�$+�Xc�A�)*

lossEiy=cX�       �	��+�Xc�A�)*

loss$Ҵ;z#�       �	�`,�Xc�A�)*

loss�;�Z��       �	-�Xc�A�)*

loss���;�'c       �	��-�Xc�A�)*

loss;z�;aU       �	g.�Xc�A�)*

loss�&b<����       �	/�Xc�A�)*

lossj;��W�       �	g�/�Xc�A�)*

loss���;o;��       �	"Q0�Xc�A�)*

loss�S<6��       �	��0�Xc�A�)*

loss�!U=j�2�       �	 �1�Xc�A�)*

loss<%�<�v��       �	�.2�Xc�A�)*

loss�g�;���       �	.�2�Xc�A�)*

loss��9<��d       �	�b3�Xc�A�)*

loss��v;�3       �	&4�Xc�A�)*

lossH�;�H��       �	��4�Xc�A�)*

loss��
<SU�       �	+j5�Xc�A�)*

loss��6;� c       �	m6�Xc�A�)*

loss9�:���d       �	��6�Xc�A�)*

loss!��<9�y�       �	�Q7�Xc�A�)*

lossV$�;�v�       �	F�7�Xc�A�)*

lossӍ�<Y_��       �	k�8�Xc�A�)*

loss߷�:���       �	�C9�Xc�A�)*

loss}9�<N�       �	��9�Xc�A�)*

loss&.�8o�:X       �		q:�Xc�A�)*

loss��:'V�       �	�;�Xc�A�)*

loss K?=���       �	ɫ;�Xc�A�)*

loss���;��2       �	G<�Xc�A�)*

loss��:�=��       �	]=�Xc�A�)*

loss��D<�e�       �	ͮ=�Xc�A�)*

loss�$=3bh\       �	|G>�Xc�A�)*

loss���:ʩ��       �	M�>�Xc�A�)*

lossdi)=���c       �	){?�Xc�A�)*

loss��<9��h       �	v@�Xc�A�)*

lossƒ�;!P5       �	 �@�Xc�A�)*

lossv�;�r�       �	�gA�Xc�A�)*

loss&V6=�ଲ       �	.B�Xc�A�)*

loss�J<�A1�       �	��B�Xc�A�)*

loss	�z<���       �	T8C�Xc�A�)*

lossz1�="�        �	J�C�Xc�A�)*

lossۊ6<c�$       �	mpD�Xc�A�)*

lossA�<f��e       �	�E�Xc�A�)*

loss��<��k       �	��E�Xc�A�)*

loss\��<6�$       �	3PF�Xc�A�)*

loss4�<[ʥ       �	p�F�Xc�A�)*

loss��;��IJ       �	��G�Xc�A�)*

lossׂ�<��       �	JDH�Xc�A�)*

loss�  <�v�       �	��H�Xc�A�)*

loss�i<WE��       �	ծI�Xc�A�)*

loss؇=k1ݑ       �	�J�Xc�A�)*

loss�G�;'��1       �	��K�Xc�A�)*

lossd�Y;6�6w       �	�6L�Xc�A�)*

loss��=b�N.       �	�>M�Xc�A�)*

loss�=:M3�       �	�3N�Xc�A�)*

loss���;��Q       �	��N�Xc�A�)*

loss6S;Q�0�       �	&�O�Xc�A�)*

loss? �;B4��       �	MMP�Xc�A�)*

loss	�
=��h�       �	Q�Xc�A�**

loss��g=<���       �	��Q�Xc�A�**

loss��o<�Ad       �	@�R�Xc�A�**

loss�jJ=:E�       �	,�S�Xc�A�**

loss�N�;_b�       �	4T�Xc�A�**

loss��=���=       �	��T�Xc�A�**

loss�nP;���       �	�uU�Xc�A�**

loss[�;�V�       �	�$V�Xc�A�**

losso�=���       �	[�V�Xc�A�**

loss8<�:)RN�       �	��W�Xc�A�**

lossY?<ӣ��       �	*X�Xc�A�**

loss��<\N       �	��X�Xc�A�**

loss��;��W�       �	�nY�Xc�A�**

loss߭�<<P��       �	�Z�Xc�A�**

loss�h;���       �	��Z�Xc�A�**

loss���;ip�q       �	t~[�Xc�A�**

losslWa<�B5       �	�Y\�Xc�A�**

loss@<Ip       �	E+]�Xc�A�**

loss`g�;��K       �	2�]�Xc�A�**

loss���<`"<o       �	px^�Xc�A�**

loss8��<�P��       �	,*_�Xc�A�**

loss��;c4��       �	�_�Xc�A�**

loss�b;k��       �	]j`�Xc�A�**

loss-�;�I1       �	�a�Xc�A�**

lossP�<��K�       �	b�Xc�A�**

lossq��;q��Q       �	P~�Xc�A�**

loss	�\=L���       �	�~�Xc�A�**

loss�;)=�.~       �	�N�Xc�A�**

loss��<ƭ�?       �	���Xc�A�**

loss��<��
�       �	x���Xc�A�**

lossN��=I춓       �	�8��Xc�A�**

loss�	�;�N'       �	�݁�Xc�A�**

lossm��;�
��       �	4���Xc�A�**

lossI�=d#^        �	�6��Xc�A�**

loss�a�<�1�       �	Ճ�Xc�A�**

loss���:�F�       �	^���Xc�A�**

loss��K<��9       �	���Xc�A�**

loss�;<���       �	A���Xc�A�**

loss��<�	��       �	^��Xc�A�**

loss�}f<9�u�       �	�	��Xc�A�**

lossh��;5��       �	n���Xc�A�**

loss
%:5w�'       �	D��Xc�A�**

lossJW;]�Mf       �	8��Xc�A�**

loss�o�;�e��       �	�։�Xc�A�**

loss�W�<�́_       �	�n��Xc�A�**

loss�W7<�	Uc       �	�G��Xc�A�**

loss��:s+       �	t`��Xc�A�**

loss��	<J�ӫ       �	����Xc�A�**

loss��=+X�       �	�ҍ�Xc�A�**

loss�m�<�7j_       �	�m��Xc�A�**

loss�&�;��&�       �	�%��Xc�A�**

loss�O�<�N��       �	�ڏ�Xc�A�**

lossn�_;�Ιv       �	����Xc�A�**

loss���<'M�       �	NC��Xc�A�**

loss ��;�cl�       �	4���Xc�A�**

loss��;���       �	���Xc�A�**

loss��;B ��       �	H��Xc�A�**

loss��;Ε4q       �	���Xc�A�**

loss���<7��       �	u���Xc�A�**

lossH�;c�       �	6��Xc�A�**

loss��<8�I       �	�֕�Xc�A�**

loss�-*;Fq~�       �	�u��Xc�A�**

loss>\<��nC       �	��Xc�A�**

loss<��<\!�       �	Lė�Xc�A�**

lossʹ�<���       �	Ct��Xc�A�**

loss�. <��       �	���Xc�A�**

losst��;FG       �	<���Xc�A�**

loss��<�9��       �	�`��Xc�A�**

loss�`�<ۃ;\       �	/���Xc�A�**

loss.�R</�zE       �	 ���Xc�A�**

loss1��<�2r       �	�C��Xc�A�**

lossƘ�;W_fM       �	��Xc�A�**

loss�[�<3	�       �	hy��Xc�A�**

loss ��;"�       �	���Xc�A�**

loss�9���)       �	걞�Xc�A�**

loss�b�<�!i�       �	�Z��Xc�A�**

loss�$8<�       �	����Xc�A�**

loss��:a��       �	����Xc�A�**

loss�ʤ=�K       �	Y2��Xc�A�**

loss��;|�'h       �	-͡�Xc�A�**

lossn1�;��$       �	;o��Xc�A�**

loss�!j:��Ou       �	��Xc�A�**

loss�е97��       �	����Xc�A�**

loss��<Nղ�       �	k��Xc�A�**

loss��<���       �	
��Xc�A�**

loss� �<�93       �	���Xc�A�**

loss!�;=��       �	�D��Xc�A�**

lossj�s:8iU�       �	���Xc�A�**

loss�=�	�0       �	~��Xc�A�**

loss���;�ᥒ       �	)ͨ�Xc�A�**

lossΘ�<!��       �	�c��Xc�A�**

loss]ɑ;��l       �	i���Xc�A�**

loss�f�=Ck�b       �	ʣ��Xc�A�**

loss�V+<(!v�       �	>��Xc�A�**

loss��=�Zg       �	P��Xc�A�**

losslap<�\�       �	�~��Xc�A�**

loss��<�Fl~       �	��Xc�A�**

lossʰ<��8O       �	���Xc�A�**

loss�<{��       �	�]��Xc�A�**

loss���<�!��       �	H���Xc�A�**

loss��<��Z        �	A���Xc�A�**

lossFX8=[�\�       �	y;��Xc�A�**

loss�c�<��t       �	�ٰ�Xc�A�**

loss�S�:uЃm       �	���Xc�A�**

lossȗ|<$��       �	�?��Xc�A�**

losssٿ<��u�       �	�ڲ�Xc�A�**

loss���<�++(       �	�v��Xc�A�**

loss��<zh��       �	���Xc�A�**

loss�B;���       �	��Xc�A�**

loss�=TŻ�       �	N���Xc�A�**

loss��N=��       �	�7��Xc�A�**

lossh0�;cÈD       �	�Ͷ�Xc�A�**

loss8�=��/�       �	�h��Xc�A�**

loss�O<q/�       �	���Xc�A�**

lossn��<Ǔ�h       �	��Xc�A�**

loss�Jj<F�5T       �	�<��Xc�A�**

loss3��<F���       �	�ҹ�Xc�A�**

loss�'�;"Q�       �	�j��Xc�A�**

loss,E�<�T       �	��Xc�A�**

lossi�<+���       �	g���Xc�A�**

loss�`<����       �	�1��Xc�A�**

loss>(=n+��       �	ͼ�Xc�A�**

lossV��<?�8Q       �	wf��Xc�A�**

loss#1=(��       �	� ��Xc�A�**

lossi}�<�D��       �	E���Xc�A�**

loss��g;�{��       �	v3��Xc�A�**

lossċ;�D�       �	ɿ�Xc�A�**

loss�r�<d%j       �	�l��Xc�A�**

loss�W�;bˀ�       �	���Xc�A�+*

loss�m2<	�M=       �	г��Xc�A�+*

lossp;�CwA       �	>Y��Xc�A�+*

loss2��<����       �	����Xc�A�+*

loss1�>;���i       �	w���Xc�A�+*

lossқ�<���       �	F^��Xc�A�+*

lossa43;Y��       �	���Xc�A�+*

loss��|=Fy��       �		���Xc�A�+*

loss)�<���       �	kH��Xc�A�+*

lossjS=���       �	����Xc�A�+*

loss,�%=�D�       �	A}��Xc�A�+*

lossmj;;��H       �	���Xc�A�+*

lossQc�:�c"N       �	6���Xc�A�+*

loss7^�;�<�       �	����Xc�A�+*

lossA�v=a(�#       �	����Xc�A�+*

loss��/=�qӏ       �	�g��Xc�A�+*

loss�V�;]՟       �	:���Xc�A�+*

loss��s<=�ƹ       �	���Xc�A�+*

lossTѦ=T�9       �	d$��Xc�A�+*

lossL��;�5f
       �	/���Xc�A�+*

loss�#�<�$R�       �	�b��Xc�A�+*

loss��<�j�       �	����Xc�A�+*

loss��R<�q�-       �	)���Xc�A�+*

loss��<4Ƭ       �	+1��Xc�A�+*

lossN=�<{kD�       �	���Xc�A�+*

loss\"h<���       �	Ie��Xc�A�+*

loss���<���>       �	>��Xc�A�+*

loss{8�<����       �	r���Xc�A�+*

loss.)<��oG       �	YN��Xc�A�+*

loss-��<��2�       �	����Xc�A�+*

lossQc�<2(�w       �	i���Xc�A�+*

loss�<�FU�       �	*��Xc�A�+*

loss(�2<����       �	:���Xc�A�+*

loss�G<͚֫       �	�m��Xc�A�+*

loss�&�;���       �	b��Xc�A�+*

loss ՜<J�Pv       �	���Xc�A�+*

loss֠�<�S��       �	sL��Xc�A�+*

lossV�=;��RH       �	����Xc�A�+*

loss w�<�B�4       �	N���Xc�A�+*

loss�J<��       �	^��Xc�A�+*

loss��]<�U�\       �	}��Xc�A�+*

loss8W�=�F�       �	����Xc�A�+*

loss=��<�ŏ       �	FB��Xc�A�+*

loss�F=�H��       �	a���Xc�A�+*

loss]�1<�$�       �	Q���Xc�A�+*

lossZ�;�"�       �	�"��Xc�A�+*

loss�><���|       �	����Xc�A�+*

loss,:i<�jk       �	\��Xc�A�+*

loss24<�J��       �	 ��Xc�A�+*

lossvgx=���8       �	���Xc�A�+*

loss!jf=��'�       �	2=��Xc�A�+*

loss�2@<�z�       �	,���Xc�A�+*

lossƧ<���6       �	���Xc�A�+*

lossVc�:��c7       �	� ��Xc�A�+*

loss�<�K�       �	����Xc�A�+*

loss��<rej       �	�`��Xc�A�+*

lossF��;d��]       �	����Xc�A�+*

loss1�'=�7�       �	����Xc�A�+*

loss�Sm;��V�       �	�)��Xc�A�+*

loss��<�       �	˿��Xc�A�+*

lossU�9��q       �	����Xc�A�+*

lossfS�<�NS�       �	kF��Xc�A�+*

loss�<�;h�
�       �	`��Xc�A�+*

losstk;vG�0       �	����Xc�A�+*

loss��=Z���       �	�r��Xc�A�+*

loss�Һ<Ϟ}�       �	c��Xc�A�+*

lossN��;I�P�       �	6���Xc�A�+*

loss��6<A�]       �	�T��Xc�A�+*

loss*˔<*��j       �	����Xc�A�+*

loss?9p;`@~       �	����Xc�A�+*

loss7��<� ��       �	�>��Xc�A�+*

losst;�[Q       �	����Xc�A�+*

lossq{f<�I       �	`s��Xc�A�+*

loss5��<�y�       �	���Xc�A�+*

loss���;�
       �	����Xc�A�+*

loss./t<Ҭ3       �	�v��Xc�A�+*

loss_��<�4�       �	���Xc�A�+*

loss�o*9��`       �	_���Xc�A�+*

loss���<�\��       �	�_��Xc�A�+*

lossT��:@�/-       �	���Xc�A�+*

loss���:���       �	����Xc�A�+*

loss�D<��1       �	<P��Xc�A�+*

lossԻ�:Q��       �	@���Xc�A�+*

lossI/r<�[�v       �	����Xc�A�+*

loss;@h=$�       �	G<��Xc�A�+*

loss�P�;Ńe       �	M���Xc�A�+*

lossn�<oQ��       �	){��Xc�A�+*

loss5`<Yj�1       �	��Xc�A�+*

loss�W�<(�Ш       �	+���Xc�A�+*

loss��;o=|l       �	XW��Xc�A�+*

lossź�;�&�<       �	����Xc�A�+*

lossi�<�i�       �	ۊ��Xc�A�+*

lossj�<<��       �	�4��Xc�A�+*

loss� <���       �	q���Xc�A�+*

loss:`�<��0�       �	Gs��Xc�A�+*

lossH=�'�       �	���Xc�A�+*

loss(x;��i�       �	.���Xc�A�+*

loss�<`y��       �	"p �Xc�A�+*

loss��j;m��       �	��Xc�A�+*

loss�e�<T�ʚ       �	3R�Xc�A�+*

lossf�;���       �	���Xc�A�+*

lossxs<���$       �	Ę�Xc�A�+*

loss�w~<1(��       �	U/�Xc�A�+*

lossc��<�<u       �	N��Xc�A�+*

loss�K`=�v�y       �		n�Xc�A�+*

loss���<���       �	A,�Xc�A�+*

loss�y�<���       �	���Xc�A�+*

losss;R<��4.       �	U��Xc�A�+*

loss�D�;#��       �	|G�Xc�A�+*

loss�ur<!R�       �	�
	�Xc�A�+*

loss���;D�&       �	)�	�Xc�A�+*

lossT�<�p�       �	ML
�Xc�A�+*

lossz|=��
�       �	��
�Xc�A�+*

loss��'=�pZ�       �	 ~�Xc�A�+*

lossXЀ=��       �	�&�Xc�A�+*

loss ��<g�9�       �	��Xc�A�+*

lossƪ(=����       �	�Z�Xc�A�+*

loss�7d;�RBg       �	g��Xc�A�+*

lossԺ~;E6g!       �	���Xc�A�+*

lossF��;����       �	�%�Xc�A�+*

loss� �<`�^�       �	
��Xc�A�+*

lossF��<��bw       �	���Xc�A�+*

losst�s<�y�       �	7�Xc�A�+*

loss��<$.;       �	O��Xc�A�+*

lossnm;�~��       �	
h�Xc�A�+*

loss��;ͦ8       �	C�Xc�A�+*

loss�C�;����       �	{��Xc�A�+*

loss��<��6�       �	�>�Xc�A�+*

loss��<6<Ĥ       �	��Xc�A�,*

loss=L[<�.�       �	+��Xc�A�,*

loss�7�;����       �	�"�Xc�A�,*

loss��R<��       �	.�Xc�A�,*

lossO~�<۫��       �	��Xc�A�,*

loss�{<VT��       �	H5�Xc�A�,*

lossc<�Д       �	l��Xc�A�,*

loss�x�<�0�q       �	f�Xc�A�,*

losst��:;�       �	��Xc�A�,*

loss�K<�1[       �	���Xc�A�,*

loss�Ө<n��'       �	f.�Xc�A�,*

loss���;Igz�       �	!��Xc�A�,*

lossE0�=�;�       �	�]�Xc�A�,*

loss�*�<��       �	k��Xc�A�,*

loss�Y�;�P�Z       �	���Xc�A�,*

loss��z:��l       �	� �Xc�A�,*

loss�B�<����       �	Թ�Xc�A�,*

lossR[E;��|2       �	3N�Xc�A�,*

loss� <I�uI       �	m��Xc�A�,*

loss��;m��I       �	�x �Xc�A�,*

loss6fA<���U       �	�H!�Xc�A�,*

lossn:=�4��       �	��!�Xc�A�,*

lossfUI:J;B       �	�"�Xc�A�,*

loss��<�u�       �	�#�Xc�A�,*

loss�[
<�Is;       �	HN$�Xc�A�,*

loss.o�<�*Q�       �	��$�Xc�A�,*

loss��k<i�'       �	x�%�Xc�A�,*

loss�'!<�S'o       �	;&�Xc�A�,*

loss3�s<B"bm       �	��&�Xc�A�,*

loss�=�
ͨ       �	�}'�Xc�A�,*

lossD�7;l��       �	(�Xc�A�,*

loss�1=�-       �	�(�Xc�A�,*

loss��c;F�)       �	��)�Xc�A�,*

loss��< Pj�       �	Q�*�Xc�A�,*

loss��<�g��       �	P�+�Xc�A�,*

loss@Q�<��       �	�B,�Xc�A�,*

lossq�i<q�]�       �	G-�Xc�A�,*

lossw�=�Wfj       �	�-�Xc�A�,*

loss��U<=��e       �	?�.�Xc�A�,*

loss�̇;Vɷ       �	�K/�Xc�A�,*

loss���;_�C       �	��/�Xc�A�,*

lossf0=�P��       �	D�0�Xc�A�,*

lossF��;	�(�       �	�81�Xc�A�,*

loss2|<(��k       �	��1�Xc�A�,*

lossߜt=�Ԙ�       �	�h2�Xc�A�,*

lossS�;��X�       �	d:3�Xc�A�,*

loss�j�<l'I�       �	b�3�Xc�A�,*

loss��t<�._�       �	 p4�Xc�A�,*

loss�`=����       �	$	5�Xc�A�,*

loss ��<���Q       �	��5�Xc�A�,*

loss���;��k       �	B6�Xc�A�,*

lossΤ�<?x�w       �	��6�Xc�A�,*

loss�'<�fU�       �	�7�Xc�A�,*

losshi�<gx�       �	�'8�Xc�A�,*

loss�~2=�6��       �	��8�Xc�A�,*

lossD�4<p�+�       �	,e9�Xc�A�,*

loss�n<ha�       �	��9�Xc�A�,*

lossJ�i;�l�       �	d�:�Xc�A�,*

lossM%a;�g�       �	�?;�Xc�A�,*

loss<�;A苨       �	k�;�Xc�A�,*

lossds�<�Z�x       �	{�<�Xc�A�,*

loss��<�@?�       �	]6=�Xc�A�,*

loss�A: 	I�       �	��=�Xc�A�,*

loss�<L@"       �	{f>�Xc�A�,*

loss��:<P�f       �	��>�Xc�A�,*

loss�`<߽�_       �	��?�Xc�A�,*

lossZs�;TÌ       �	�)@�Xc�A�,*

loss(=�|       �	��@�Xc�A�,*

loss���;�^�I       �	�UA�Xc�A�,*

loss �<c��       �	��A�Xc�A�,*

loss�o�:5���       �	A�B�Xc�A�,*

losst��<�K�e       �	�.C�Xc�A�,*

loss:�<���/       �	*�C�Xc�A�,*

loss$ܵ<�^�       �	�bD�Xc�A�,*

loss擰=/��W       �	��D�Xc�A�,*

loss���;�g`�       �	h�E�Xc�A�,*

lossO*�;���q       �	qsF�Xc�A�,*

lossZ �;��aP       �	�H�Xc�A�,*

loss�%
=��v�       �	_�H�Xc�A�,*

loss1=@;��       �	[_I�Xc�A�,*

loss�S=5�i�       �	�J�Xc�A�,*

loss��f=       �	��J�Xc�A�,*

lossV��<�@�`       �	XqK�Xc�A�,*

loss�O�;$���       �	�L�Xc�A�,*

loss�E<[멾       �	��L�Xc�A�,*

loss�4�<�ۓ^       �	+PM�Xc�A�,*

lossx7<
�H�       �	3�M�Xc�A�,*

lossF�n<��1       �	��N�Xc�A�,*

loss*�;W�\�       �	v6O�Xc�A�,*

lossJ�X<�?k       �	��O�Xc�A�,*

loss�y=b�Q�       �	�P�Xc�A�,*

loss�sI<�;�#       �	cQ�Xc�A�,*

loss��n<!�ع       �	T R�Xc�A�,*

loss���;!�r>       �	��R�Xc�A�,*

loss�,<��s       �	Y6S�Xc�A�,*

loss��t<�k�v       �	y�S�Xc�A�,*

loss	u<�*�6       �	�`T�Xc�A�,*

loss#� <�Y�       �	��T�Xc�A�,*

loss1:�=�w�       �	o�U�Xc�A�,*

lossZ�~<�Ű       �	V�Xc�A�,*

loss㕍<�/!       �	��V�Xc�A�,*

loss؆;���       �	^W�Xc�A�,*

lossM�x;�O�       �	+�W�Xc�A�,*

losssL�<�H-6       �	\�X�Xc�A�,*

loss㖏;�Q�       �	�&Y�Xc�A�,*

loss�W�<\�;�       �	��Y�Xc�A�,*

loss�d�<�'�       �	[Z�Xc�A�,*

loss�<=U�4�       �	_	[�Xc�A�,*

loss�U�:�FK       �	��[�Xc�A�,*

loss�P�:�Bc5       �	�E\�Xc�A�,*

loss<��<���/       �	6�\�Xc�A�,*

loss�|y<�J\       �	p}]�Xc�A�,*

loss���;u� $       �	�^�Xc�A�,*

loss��=9�)       �	��^�Xc�A�,*

loss�=��!�       �	�<_�Xc�A�,*

loss�%�;��)s       �	$�_�Xc�A�,*

lossׅ<	���       �	�s`�Xc�A�,*

loss�R<m�S       �	�a�Xc�A�,*

loss���;7��       �	?�a�Xc�A�,*

lossԿ�:~�q�       �	IHb�Xc�A�,*

loss�\<,�       �	�c�Xc�A�,*

lossQ�]=�7u�       �	�c�Xc�A�,*

loss�_<�џM       �	�Dd�Xc�A�,*

loss�[<k�       �	��d�Xc�A�,*

loss�h;
�H�       �	�e�Xc�A�,*

loss�w];�B�       �	i6f�Xc�A�,*

loss�+;���       �	�f�Xc�A�,*

lossx��;�b�       �	�g�Xc�A�,*

loss_�<`fGB       �	�Dh�Xc�A�-*

loss�,<��V       �	��h�Xc�A�-*

loss�>8<�&P       �	˂i�Xc�A�-*

loss��<�(R       �	�'j�Xc�A�-*

loss��X=��       �	��j�Xc�A�-*

loss��A<��-�       �	��k�Xc�A�-*

loss�5=+��M       �	Gm�Xc�A�-*

loss�S*<(�       �	4�m�Xc�A�-*

loss��;�#�       �	Àn�Xc�A�-*

lossC�a<2��p       �	�*o�Xc�A�-*

loss�j<1���       �	~�o�Xc�A�-*

loss7o:<צ͊       �	�fp�Xc�A�-*

loss��<&7��       �	�q�Xc�A�-*

loss�P\;s�Y�       �	�q�Xc�A�-*

lossFu <	��       �	*Wr�Xc�A�-*

loss4~<%���       �	�s�Xc�A�-*

lossT�;��7d       �	�s�Xc�A�-*

loss��<�8-�       �	�Ft�Xc�A�-*

loss-�<'1-�       �	��t�Xc�A�-*

loss���<��ħ       �	c�u�Xc�A�-*

loss}�S<        �	V�v�Xc�A�-*

loss��<�       �	�Zw�Xc�A�-*

loss�y�<)m�       �	�x�Xc�A�-*

lossھ-<>�2       �	��x�Xc�A�-*

loss��:�I�       �	YNy�Xc�A�-*

lossM�[:�+T       �	u�y�Xc�A�-*

loss�r�<i�ǃ       �	s�z�Xc�A�-*

loss��;+��        �	[^{�Xc�A�-*

loss�p<�9��       �	|�Xc�A�-*

loss
��:��]       �	 �|�Xc�A�-*

loss�э<H5�       �	�d}�Xc�A�-*

loss9}<Z 	       �	�
~�Xc�A�-*

loss�{�;��,       �	S�~�Xc�A�-*

lossOI�<�^       �	�R�Xc�A�-*

loss�6=.OU       �	3��Xc�A�-*

lossO:0<�
:�       �	gր�Xc�A�-*

loss?}�<�Yc       �	�u��Xc�A�-*

loss7hU:3��       �	���Xc�A�-*

lossע�;�8A�       �	���Xc�A�-*

loss[	�;�w|X       �	�P��Xc�A�-*

loss�i�;�!��       �	���Xc�A�-*

loss?w�:����       �	}��Xc�A�-*

loss�hR;��8�       �	�Q��Xc�A�-*

lossD�<�#�       �	v��Xc�A�-*

loss��<��m�       �	�~��Xc�A�-*

lossI�=�Ae       �	�!��Xc�A�-*

loss���;�g0�       �	߇�Xc�A�-*

loss�Aa=-x       �	dx��Xc�A�-*

loss��z<6Q�'       �	���Xc�A�-*

lossH;�tk       �	����Xc�A�-*

loss\�1<��iN       �	ץ��Xc�A�-*

loss��S<�       �	^���Xc�A�-*

lossN�=��~i       �		���Xc�A�-*

loss�#;z8�
       �	�u��Xc�A�-*

lossP��<���d       �	1&��Xc�A�-*

loss�=��G       �	���Xc�A�-*

loss��<�	=�       �	�Ï�Xc�A�-*

loss�/;       �	t��Xc�A�-*

loss�֫;i�?       �	?���Xc�A�-*

loss�d:<���=       �	�P��Xc�A�-*

loss�<|�       �	���Xc�A�-*

loss��<��o�       �	d���Xc�A�-*

loss��<�0H       �	<-��Xc�A�-*

loss�;'��       �	�˔�Xc�A�-*

loss�?�<�}       �	�f��Xc�A�-*

loss��<�\�`       �	y��Xc�A�-*

loss3A�=���       �	*���Xc�A�-*

loss�q<�        �	xB��Xc�A�-*

lossX(<8|��       �	�ۗ�Xc�A�-*

loss��h<~��d       �	Ց��Xc�A�-*

loss���<�5�       �	/5��Xc�A�-*

loss�V<���       �	�Ι�Xc�A�-*

loss8��<�Q��       �	�p��Xc�A�-*

loss�:�0#�       �	� ��Xc�A�-*

loss�%=�/�b       �	=��Xc�A�-*

loss6B�;�y�0       �	����Xc�A�-*

loss���<:�.N       �	:#��Xc�A�-*

loss���;�_-[       �	s���Xc�A�-*

loss|�;�E�       �	�S��Xc�A�-*

loss�j;p���       �	����Xc�A�-*

lossE[|<ךV�       �	X���Xc�A�-*

lossw�=��       �	-��Xc�A�-*

loss�z�=��~%       �	�ɠ�Xc�A�-*

loss�B�<j&]�       �	,c��Xc�A�-*

loss��<s��       �	���Xc�A�-*

loss��P;���       �	>���Xc�A�-*

loss���<[7G�       �	^.��Xc�A�-*

loss�<�?1�       �	�ǣ�Xc�A�-*

lossB<��G       �	=c��Xc�A�-*

loss)��<fB�:       �	e���Xc�A�-*

lossH�)<s���       �	���Xc�A�-*

loss?{�<�r;       �	2��Xc�A�-*

loss�X<�.�       �	\ʦ�Xc�A�-*

lossI$7<��3J       �	0c��Xc�A�-*

loss}ۥ;�l       �	]���Xc�A�-*

loss�j<�t5z       �	ȗ��Xc�A�-*

lossK�<V�r�       �	'1��Xc�A�-*

loss�9�;3��'       �	�ǩ�Xc�A�-*

loss-��;2�+i       �	`��Xc�A�-*

lossb<^J�
       �	����Xc�A�-*

loss[� <�T9�       �	8���Xc�A�-*

loss;P       �	�7��Xc�A�-*

lossO{n;C�0�       �	X ��Xc�A�-*

loss�y�:�$       �	^���Xc�A�-*

lossn5 =0qa�       �	�4��Xc�A�-*

loss !�;�ť       �	�̮�Xc�A�-*

loss�-�<b�G	       �	�e��Xc�A�-*

loss���<�Y}l       �	h��Xc�A�-*

loss�8�;�Bl6       �	��Xc�A�-*

loss]��:εH       �	C;��Xc�A�-*

loss��.:R��       �	�ӱ�Xc�A�-*

loss��=��J�       �	u��Xc�A�-*

lossR�N;;�
�       �	���Xc�A�-*

loss��;ΰi�       �	뭳�Xc�A�-*

loss<-<Z��       �	�J��Xc�A�-*

loss�e�<�˦4       �	���Xc�A�-*

loss���:���       �	I���Xc�A�-*

lossE�":QSq       �	���Xc�A�-*

loss�EA<CO�       �	:���Xc�A�-*

loss�1<�ļF       �	�_��Xc�A�-*

lossss*>@�$       �	����Xc�A�-*

loss��=��j       �	����Xc�A�-*

loss��<���+       �	�,��Xc�A�-*

loss���;�[C�       �	Ĺ�Xc�A�-*

loss,�;#Q�%       �	p\��Xc�A�-*

loss}7�;q��"       �	���Xc�A�-*

loss�!�<a���       �	����Xc�A�-*

lossLX�:�:
       �	�@��Xc�A�-*

loss�fm<�       �	vݼ�Xc�A�.*

loss�/<��wi       �	z��Xc�A�.*

loss��C=�n��       �	*��Xc�A�.*

loss�a=� �       �	$���Xc�A�.*

lossZD;�e��       �	\W��Xc�A�.*

loss��'=�O�       �	R��Xc�A�.*

losscEr;�֢�       �	���Xc�A�.*

lossf�,=*F       �	�/��Xc�A�.*

loss��#;FI�       �	-���Xc�A�.*

loss���; �i       �	�q��Xc�A�.*

loss�:}<��?5       �	U��Xc�A�.*

loss��<�V�F       �	S���Xc�A�.*

lossV;n;Q#�       �	,H��Xc�A�.*

lossj�<�ɚ�       �	����Xc�A�.*

loss�S;u��7       �	b���Xc�A�.*

lossw��<�Le�       �	���Xc�A�.*

loss��C<���       �	����Xc�A�.*

loss��<)Id       �	 W��Xc�A�.*

lossSN�<�s�       �	����Xc�A�.*

lossP�;}��       �	Ό��Xc�A�.*

loss��:B=��       �	�:��Xc�A�.*

loss��;����       �	����Xc�A�.*

loss�;�<��~       �	܃��Xc�A�.*

loss#F�;��,�       �	�!��Xc�A�.*

loss��<g�N       �	���Xc�A�.*

loss��;�aA       �	-\��Xc�A�.*

lossN�w<�r�       �	A��Xc�A�.*

loss!�5<��޷       �	����Xc�A�.*

loss��?<V�       �	�Z��Xc�A�.*

loss��4<kq�       �	����Xc�A�.*

loss��<ػ��       �	���Xc�A�.*

loss�z�;�7Ky       �	1��Xc�A�.*

loss4Z�;���       �	����Xc�A�.*

loss}��;өi�       �	�n��Xc�A�.*

lossL[�<�`�       �	�"��Xc�A�.*

loss�E!:(�L�       �	����Xc�A�.*

lossg)<
�0}       �	7m��Xc�A�.*

loss���<3�2�       �	���Xc�A�.*

loss�{<��"L       �	���Xc�A�.*

loss���;um��       �	*R��Xc�A�.*

loss�)<�:��       �	����Xc�A�.*

loss�M;=(r:Y       �	���Xc�A�.*

loss��-:Kn�2       �	�G��Xc�A�.*

loss�$`<U˓       �	����Xc�A�.*

loss�v<�ْ       �	����Xc�A�.*

loss��:�S+       �	rk��Xc�A�.*

loss��	83g�)       �	���Xc�A�.*

loss�(x;�mƹ       �	7���Xc�A�.*

loss�s:T��       �	kE��Xc�A�.*

loss�	�<ή�       �	���Xc�A�.*

loss��{;�#	       �	g}��Xc�A�.*

lossTEO;� %       �	r��Xc�A�.*

loss[��<��)�       �	ծ��Xc�A�.*

loss��=0u"       �	oI��Xc�A�.*

loss��n8 ��       �	!���Xc�A�.*

losse�G7���F       �	���Xc�A�.*

lossZ�91�a�       �	�7��Xc�A�.*

loss�R<ҭ�^       �	���Xc�A�.*

lossߢ�;j��       �	<���Xc�A�.*

loss�i�:��O(       �	� ��Xc�A�.*

lossKǏ;q2/�       �	����Xc�A�.*

loss�Kb=�C��       �	�e��Xc�A�.*

loss�?=;F��       �	���Xc�A�.*

loss�;=��F�       �	����Xc�A�.*

lossO�O;�l�g       �	aR��Xc�A�.*

loss�(=Vvc�       �	S���Xc�A�.*

loss�/;���       �	����Xc�A�.*

loss�� <���       �	og��Xc�A�.*

lossW��=����       �	<3��Xc�A�.*

loss��;F�d�       �	Z���Xc�A�.*

lossژ�;X�y       �	���Xc�A�.*

loss�E�:fξ�       �	����Xc�A�.*

losss<h� �       �	����Xc�A�.*

loss�(c=r?h       �	(E��Xc�A�.*

lossH�<��       �	����Xc�A�.*

loss�V<�F�       �	���Xc�A�.*

loss	� <��V       �	����Xc�A�.*

loss��<�~5�       �	W&��Xc�A�.*

loss_�<�#�       �	+���Xc�A�.*

loss�Ą;�%       �	CX��Xc�A�.*

loss�"=2}       �		��Xc�A�.*

loss�6b;��cr       �	���Xc�A�.*

loss�q;�X�t       �	Zg��Xc�A�.*

loss�<\k�       �	���Xc�A�.*

loss���:'#&       �	���Xc�A�.*

loss�U�;����       �	�N��Xc�A�.*

loss��s<��       �	eV��Xc�A�.*

losss6�;�?�K       �	N���Xc�A�.*

loss�J*=�_�       �	ʋ��Xc�A�.*

loss^�;�-7       �	�/��Xc�A�.*

loss���<c|�       �	K���Xc�A�.*

loss<=Z}       �	g`��Xc�A�.*

loss/�Z<
L�       �	w���Xc�A�.*

loss�{<�	       �	����Xc�A�.*

lossVR[<�mS       �	t(��Xc�A�.*

loss���<_�b       �	@���Xc�A�.*

loss��<���       �	�b��Xc�A�.*

lossb�;mHd�       �	 ��Xc�A�.*

loss�ދ<���       �	V���Xc�A�.*

lossk<��q�       �	X7��Xc�A�.*

loss�F"=CC�       �	E���Xc�A�.*

loss�<��c       �	q��Xc�A�.*

lossT�4;^k]�       �	@��Xc�A�.*

loss��;;C�Q       �	ѭ��Xc�A�.*

loss��<�B3D       �	�E �Xc�A�.*

loss#��<�O�       �	� �Xc�A�.*

loss#<�f��       �	b��Xc�A�.*

loss�~�;�3�/       �	��Xc�A�.*

loss.�<=��       �	���Xc�A�.*

loss�PP<��{       �	���Xc�A�.*

loss�G�<�f�G       �	��Xc�A�.*

loss��:,[       �	5��Xc�A�.*

loss���:>��
       �	�^�Xc�A�.*

loss}�{<2mi�       �	���Xc�A�.*

loss�Y=��^       �	�q�Xc�A�.*

loss@�<�Վ       �	�	�Xc�A�.*

losss�<r���       �	P��Xc�A�.*

loss�=�S�y       �	HQ �Xc�A�.*

lossN�;<Q'|       �	�� �Xc�A�.*

loss2G3=`֚�       �	��!�Xc�A�.*

lossH��<�k�       �	��"�Xc�A�.*

loss��<˵^       �	�9#�Xc�A�.*

loss�T<ek�e       �	j�#�Xc�A�.*

loss�^;$eH[       �	�z$�Xc�A�.*

loss�*]<�h�       �	Z%�Xc�A�.*

loss��;;�       �	�O&�Xc�A�.*

lossH%<��       �	��&�Xc�A�.*

loss{��<��q       �	�'�Xc�A�.*

loss)S�;n�q       �	ۣ(�Xc�A�/*

loss���9�i��       �	�H)�Xc�A�/*

loss� �<⚔�       �	�"*�Xc�A�/*

lossǬ<1�\       �	0�*�Xc�A�/*

lossN�)<\�S       �	Fa+�Xc�A�/*

loss�<q��       �	�,�Xc�A�/*

loss8Vb;
��       �	w�,�Xc�A�/*

loss�S;�_A�       �	!\-�Xc�A�/*

loss��h="2       �	 .�Xc�A�/*

loss�Ew:8�,       �	Z�.�Xc�A�/*

lossrY�:h�1       �	�C/�Xc�A�/*

loss��<U}S�       �	��/�Xc�A�/*

loss�w<�b�       �	��0�Xc�A�/*

loss�9�<��V�       �	�/1�Xc�A�/*

loss���;�y�w       �	��1�Xc�A�/*

loss�lf<�ؐI       �	)u2�Xc�A�/*

loss�m;��gy       �	R3�Xc�A�/*

lossi�B;0��M       �	[�3�Xc�A�/*

loss
 
=HןL       �	�O4�Xc�A�/*

loss�f�;">3        �	��4�Xc�A�/*

loss�==��7       �	�5�Xc�A�/*

loss��:�R�       �	N6�Xc�A�/*

loss� -<(�K�       �	��6�Xc�A�/*

lossXB*=�̞�       �	$�7�Xc�A�/*

loss>m<�D��       �	@8�Xc�A�/*

loss`.0<�� 5       �	�8�Xc�A�/*

loss�7 <M���       �	�H9�Xc�A�/*

loss�.<���|       �	]�9�Xc�A�/*

lossW�U<9��E       �	||:�Xc�A�/*

loss���=ʕ�O       �	U;�Xc�A�/*

loss���=gh=�       �	r�;�Xc�A�/*

lossV�;���       �	p><�Xc�A�/*

loss�m;��C)       �	�=�Xc�A�/*

loss�-7;�1a�       �	��=�Xc�A�/*

loss��89c�       �	��>�Xc�A�/*

lossM�<�FK�       �	D2?�Xc�A�/*

loss�-�<<�EW       �	%�?�Xc�A�/*

loss|�: ���       �	�a@�Xc�A�/*

loss���=��V       �	a�@�Xc�A�/*

loss���;ٷ�k       �	�A�Xc�A�/*

loss�Ą;f$��       �	�:B�Xc�A�/*

loss�h;@mG       �	&�B�Xc�A�/*

loss_˗:�E��       �	ܹC�Xc�A�/*

loss!��;OҘ�       �	�dD�Xc�A�/*

loss��>=e5`       �	:E�Xc�A�/*

loss�<�<f�       �	r�E�Xc�A�/*

loss]�<�Kg|       �	$GF�Xc�A�/*

loss��6<���        �	��F�Xc�A�/*

losss�<���P       �	��G�Xc�A�/*

loss=�;ziM�       �	+H�Xc�A�/*

loss���<�y#       �	��H�Xc�A�/*

loss�q!=QDF�       �	B]I�Xc�A�/*

loss�0�<�Q��       �	��I�Xc�A�/*

lossE�=��@�       �	��J�Xc�A�/*

loss��<�Vǂ       �	<2K�Xc�A�/*

loss<=u�U       �	p	L�Xc�A�/*

loss�=<��l       �	�L�Xc�A�/*

loss���<�v9h       �	�BM�Xc�A�/*

loss$�;��9�       �	�\N�Xc�A�/*

loss%N�<�j�U       �	��N�Xc�A�/*

loss�;r;j���       �	`�O�Xc�A�/*

lossC=�=1�       �	sHP�Xc�A�/*

loss�e�;���       �	�	Q�Xc�A�/*

loss���;,��.       �	M�Q�Xc�A�/*

loss��<%�
       �	�uR�Xc�A�/*

loss1�<wmS�       �	�S�Xc�A�/*

loss���<H�L�       �	$�S�Xc�A�/*

loss���<��̒       �	alT�Xc�A�/*

loss���<�x(5       �	+U�Xc�A�/*

loss.
�;��9       �	��U�Xc�A�/*

loss"�<R�VI       �	t]V�Xc�A�/*

loss@#;�Cϭ       �	��V�Xc�A�/*

lossl� =D��       �	�W�Xc�A�/*

loss��<ϝ�       �	�=X�Xc�A�/*

loss��< ��       �	�X�Xc�A�/*

loss-��=~:��       �	�}Y�Xc�A�/*

loss��<����       �	P7Z�Xc�A�/*

loss7ˡ;9�E<       �	��Z�Xc�A�/*

loss�_�<��b�       �	�g[�Xc�A�/*

losso� =�tx       �	��[�Xc�A�/*

loss��;�4C�       �	 �\�Xc�A�/*

lossL!<��q�       �	(-]�Xc�A�/*

loss�L<�{�O       �	��]�Xc�A�/*

loss鶌=1�+�       �	}^�Xc�A�/*

loss�g<˒�(       �	� _�Xc�A�/*

loss��"=��YV       �	_�_�Xc�A�/*

loss �p;�       �	�T`�Xc�A�/*

loss�$�;���       �	��`�Xc�A�/*

lossD��:f|my       �	�a�Xc�A�/*

loss�)$=$y�\       �	�#b�Xc�A�/*

loss�<���       �	�b�Xc�A�/*

lossء�:BEN       �	]Qc�Xc�A�/*

loss��<z��       �	h�c�Xc�A�/*

loss̳4<��       �	 �d�Xc�A�/*

lossa�;V��       �	�0e�Xc�A�/*

loss,	=1�P       �	6�e�Xc�A�/*

loss�m�<��m       �	nf�Xc�A�/*

loss�A�<��       �	�g�Xc�A�/*

lossvUg; =ʬ       �	>�g�Xc�A�/*

loss8��<,$�l       �	�Oh�Xc�A�/*

loss���:��       �	F�h�Xc�A�/*

loss*��;�կ/       �	��i�Xc�A�/*

losss=8�       �	�j�Xc�A�/*

loss�ɭ<t�       �	��j�Xc�A�/*

loss�x<K�u       �	�Vk�Xc�A�/*

loss�g<��ma       �	�k�Xc�A�/*

lossD�=�%�(       �	l�l�Xc�A�/*

loss��<R�       �	�Fm�Xc�A�/*

lossW<	4U�       �	q�m�Xc�A�/*

loss%y@<h`Z       �	��n�Xc�A�/*

loss��<V~��       �	��o�Xc�A�/*

loss,��<�!7       �	�]p�Xc�A�/*

loss���:w��>       �	��p�Xc�A�/*

lossOV�;�`�       �	��q�Xc�A�/*

lossM��<:�
       �	&Vr�Xc�A�/*

loss�B<�9��       �	-�r�Xc�A�/*

loss
�;��"�       �	��s�Xc�A�/*

lossOC�<�T'�       �	�?t�Xc�A�/*

loss��<\ջ�       �	��t�Xc�A�/*

loss�)�<�؍       �	�u�Xc�A�/*

loss���;~�A�       �	�0v�Xc�A�/*

loss���=�;��       �	��v�Xc�A�/*

loss P;�ڥ{       �	j�w�Xc�A�/*

loss!�#;����       �	�5x�Xc�A�/*

lossl<�< �r�       �	<�x�Xc�A�/*

loss���;-~~       �	5�y�Xc�A�/*

loss���=z�       �	�Sz�Xc�A�/*

lossJY�<ѱ�Y       �	�z�Xc�A�/*

loss��<���       �	0�{�Xc�A�0*

lossO}�<h��7       �	v5|�Xc�A�0*

loss���<���4       �	_�|�Xc�A�0*

loss\�E<�:_       �	�t}�Xc�A�0*

lossY�;�W>D       �	�~�Xc�A�0*

loss��(;����       �	��~�Xc�A�0*

loss�?2<�f�       �	�G�Xc�A�0*

loss�ZC<wP�i       �	'��Xc�A�0*

loss�U9=/�       �	�u��Xc�A�0*

lossV-8=��G�       �	���Xc�A�0*

losss�=>��q       �	R���Xc�A�0*

lossES�<��+       �	�O��Xc�A�0*

loss��;%��       �	>��Xc�A�0*

loss<�d,       �	0���Xc�A�0*

loss���<�<�       �	���Xc�A�0*

loss�I�<T�       �	|���Xc�A�0*

loss��9<> 2       �	R��Xc�A�0*

loss\�{<��J       �	x��Xc�A�0*

loss2I�:C� �       �	����Xc�A�0*

loss��;�Z       �	^*��Xc�A�0*

loss�.;�]O�       �	Ç�Xc�A�0*

loss{׃<n�>R       �	�Z��Xc�A�0*

loss�=1<)
*�       �	f���Xc�A�0*

loss��: ��n       �	���Xc�A�0*

loss���;��]�       �	�1��Xc�A�0*

loss.��;���       �	)͊�Xc�A�0*

loss@�
;���       �	����Xc�A�0*

loss�)z<oT�       �	�E��Xc�A�0*

loss�<[X	�       �	U���Xc�A�0*

loss�D�:&S�       �	���Xc�A�0*

loss���<����       �	鷎�Xc�A�0*

lossH��;�F��       �	Ox��Xc�A�0*

lossz�=�{e       �	�(��Xc�A�0*

lossv2<�|�*       �	`"��Xc�A�0*

losswO=x�6       �	���Xc�A�0*

losswmI<�RU�       �	���Xc�A�0*

loss��<��       �	�^��Xc�A�0*

lossI]�9���       �	�B��Xc�A�0*

loss���;�\       �	���Xc�A�0*

loss/�;Z	1�       �	����Xc�A�0*

lossHCs:���X       �	�Q��Xc�A�0*

lossl�L<"4�       �	���Xc�A�0*

loss��2:�֪�       �	�Xc�A�0*

lossۏ�<����       �	�+��Xc�A�0*

loss�8�<���!       �	�ט�Xc�A�0*

lossv�<�(p       �	Ks��Xc�A�0*

loss���<��jN       �	���Xc�A�0*

loss�ߋ=7���       �	ޭ��Xc�A�0*

loss���<��&�       �	�\��Xc�A�0*

loss�<�N��       �	����Xc�A�0*

loss>�=S~J�       �	����Xc�A�0*

loss�OX:d4�j       �	�9��Xc�A�0*

lossD�=P�h|       �	/���Xc�A�0*

loss�d�=��       �	x���Xc�A�0*

loss>=�X �       �	Y4��Xc�A�0*

loss7z�<�~7�       �	�̟�Xc�A�0*

loss�]�;�U��       �	f��Xc�A�0*

loss6�=;���       �	 ���Xc�A�0*

lossv��;f�       �	����Xc�A�0*

loss��;"���       �	76��Xc�A�0*

loss��<{�vb       �	�͢�Xc�A�0*

lossVd
=���       �	�m��Xc�A�0*

loss�T<��՞       �	
��Xc�A�0*

loss���<U�ʞ       �	����Xc�A�0*

lossT��<$��b       �	MH��Xc�A�0*

loss�+J<;���       �	���Xc�A�0*

lossF'�<���       �	r���Xc�A�0*

loss_��<N�MH       �	,)��Xc�A�0*

lossNy�<e#.�       �	�ħ�Xc�A�0*

lossd�;��       �	�_��Xc�A�0*

lossS�;��ȼ       �	����Xc�A�0*

loss�)3=d�I       �	����Xc�A�0*

loss�ߚ=��       �	�*��Xc�A�0*

loss��='Е       �	
���Xc�A�0*

losse/=�et�       �	���Xc�A�0*

loss&=l+�       �	2��Xc�A�0*

lossF�y<J}JK       �	�̬�Xc�A�0*

loss6ړ;��@       �	�c��Xc�A�0*

lossoA�;b8�       �	?���Xc�A�0*

loss̜�;��>       �	̮�Xc�A�0*

loss�T�<��OZ       �	�j��Xc�A�0*

loss	]=n��       �	���Xc�A�0*

loss��<L(��       �	����Xc�A�0*

loss��<�>�       �	�2��Xc�A�0*

loss�[C<�S�\       �	�>��Xc�A�0*

loss( ;��       �	ز�Xc�A�0*

loss!<�^��       �	�t��Xc�A�0*

loss�l�:\�i       �	%@��Xc�A�0*

lossT�1<���       �	���Xc�A�0*

lossh�;��_       �	ƅ��Xc�A�0*

loss!��;J���       �	R+��Xc�A�0*

loss,�	=���s       �	`ʶ�Xc�A�0*

loss�;�<|nq�       �	Kx��Xc�A�0*

loss�	�<%^A       �	���Xc�A�0*

loss��C;��       �	1���Xc�A�0*

loss�E=_���       �	||��Xc�A�0*

loss�Z;�I�       �	D��Xc�A�0*

lossW�b<`�X+       �	մ��Xc�A�0*

loss�,=D��1       �	�M��Xc�A�0*

loss���<�t6�       �	y��Xc�A�0*

loss�S$<Ht:       �	���Xc�A�0*

loss j�;��%       �	v��Xc�A�0*

loss�0S=$��       �	����Xc�A�0*

loss�;��       �	MN��Xc�A�0*

lossVf3=4���       �	2��Xc�A�0*

lossn<�eO       �	�~��Xc�A�0*

lossJ=�<؀Tu       �	���Xc�A�0*

loss�T<����       �	����Xc�A�0*

lossd�=��s�       �	M��Xc�A�0*

loss��{;;�D       �	r���Xc�A�0*

lossF�<�k$R       �	R���Xc�A�0*

loss��B:�_�       �	�4��Xc�A�0*

loss��&<��
p       �	����Xc�A�0*

loss�b�<�f�       �	zp��Xc�A�0*

loss(��<5wO�       �	H��Xc�A�0*

loss
~�;�TF=       �	#���Xc�A�0*

lossVuy<j�ɍ       �	�]��Xc�A�0*

loss���;�8K�       �	"���Xc�A�0*

loss���<A��       �	����Xc�A�0*

loss:_�;��!	       �	�@��Xc�A�0*

loss��`;W� q       �	q���Xc�A�0*

loss��<��/�       �	{���Xc�A�0*

loss��!<���       �	� ��Xc�A�0*

loss��<�-Y�       �	n���Xc�A�0*

lossQ8k<��|        �	%���Xc�A�0*

lossWb�<�q       �	����Xc�A�0*

loss��B<���0       �	'��Xc�A�0*

lossfA';�3MV       �	E���Xc�A�0*

lossX�;�T�       �	dZ��Xc�A�1*

loss�ܤ;��o�       �	M���Xc�A�1*

loss���;OeI�       �	~���Xc�A�1*

lossw=4='C       �	�'��Xc�A�1*

loss��k<Q��*       �	+���Xc�A�1*

loss��9<!�       �	�a��Xc�A�1*

loss��;��/       �	���Xc�A�1*

loss��<;l�       �	x���Xc�A�1*

loss��<z�       �	�/��Xc�A�1*

loss��h<����       �	
���Xc�A�1*

loss�;����       �	P���Xc�A�1*

lossJ��<|�;�       �	�)��Xc�A�1*

loss���;d$_�       �	r���Xc�A�1*

loss���;�j7       �	hX��Xc�A�1*

loss��J<׷#�       �	#���Xc�A�1*

loss
�<d���       �	'���Xc�A�1*

loss��=]�>�       �	� ��Xc�A�1*

lossi�*;��[       �	R���Xc�A�1*

lossh�=�5
        �	&S��Xc�A�1*

losssS�:�x	       �	���Xc�A�1*

loss��J<T�<�       �	����Xc�A�1*

loss���9����       �	G9��Xc�A�1*

loss�IE;�N I       �	!���Xc�A�1*

loss\��; �\       �	�g��Xc�A�1*

loss�5�<�Lx       �	���Xc�A�1*

loss�M�;�F��       �	o���Xc�A�1*

loss���<���5       �	�3��Xc�A�1*

loss��*=���       �	%���Xc�A�1*

losstW�;�W       �	B`��Xc�A�1*

loss�:>=�]\)       �	f���Xc�A�1*

loss���;��c       �	 ���Xc�A�1*

loss��<ۆ#       �	�/��Xc�A�1*

loss΄�=�p,�       �	����Xc�A�1*

loss�<��r�       �	o���Xc�A�1*

loss6��=���J       �	�5��Xc�A�1*

loss�q; �       �	�v��Xc�A�1*

loss�U�;Bԋ�       �	���Xc�A�1*

lossx,�<�xUk       �	����Xc�A�1*

loss�=�MWf       �	K���Xc�A�1*

loss��A;p_p�       �	�0��Xc�A�1*

lossq�<)q��       �	����Xc�A�1*

loss�O<�k�D       �	�u��Xc�A�1*

loss��w<����       �	����Xc�A�1*

lossx�<��       �	�J��Xc�A�1*

lossh2m<�K%       �	����Xc�A�1*

lossfG<���       �	;���Xc�A�1*

loss�g;���       �	�&��Xc�A�1*

loss�=��4�       �	w���Xc�A�1*

loss��n<!c�       �	���Xc�A�1*

lossl�k=c��       �	4��Xc�A�1*

loss���<��{X       �	���Xc�A�1*

loss�P=i��>       �	H���Xc�A�1*

loss�7*<7�~       �	�c��Xc�A�1*

loss��<]� +       �	D5��Xc�A�1*

loss�^H=v�ʊ       �	n���Xc�A�1*

loss톀<﵄�       �	�~��Xc�A�1*

lossш<��%�       �	9%��Xc�A�1*

losst�~<:��~       �	9���Xc�A�1*

loss�8d<��^       �	y��Xc�A�1*

loss[�#<�2�       �	���Xc�A�1*

loss��;�]�       �	����Xc�A�1*

loss:k-;#m�l       �	G���Xc�A�1*

lossp�:�{��       �	����Xc�A�1*

lossR�1=|.��       �	?;��Xc�A�1*

loss�w.<��M       �	����Xc�A�1*

loss��4<�U�(       �	"���Xc�A�1*

loss{�<R�*       �	l?��Xc�A�1*

loss=�<.��       �	����Xc�A�1*

loss��n:�2��       �	���Xc�A�1*

lossa;5R!�       �	 ��Xc�A�1*

loss]@<�d F       �	����Xc�A�1*

loss)=^M       �	@���Xc�A�1*

loss��<��       �	�G��Xc�A�1*

lossT�	=&5{       �	����Xc�A�1*

loss#��<�>A       �	����Xc�A�1*

loss�4�;���*       �	�. �Xc�A�1*

loss1�(<ݱ�       �	v� �Xc�A�1*

loss�;7,�K       �	}�Xc�A�1*

lossC�6;/���       �	 �Xc�A�1*

lossV�
<�ː�       �	Ժ�Xc�A�1*

loss.N<1Og       �	�W�Xc�A�1*

loss�X =2��       �	���Xc�A�1*

loss��:�Q�       �	;��Xc�A�1*

loss��;���"       �	9&�Xc�A�1*

lossT�;��kA       �	:�Xc�A�1*

loss�BA=�6?�       �	|�Xc�A�1*

loss(�)<�B,�       �	��Xc�A�1*

loss�V<E�l�       �	;�Xc�A�1*

loss"ч;��z       �	���Xc�A�1*

losssx�;�%�       �	�i	�Xc�A�1*

loss���;�5��       �	z�	�Xc�A�1*

loss*��<�a{`       �	R�
�Xc�A�1*

lossI�=ޫF�       �	b0�Xc�A�1*

lossKŃ;:�       �	Q�Xc�A�1*

losse!K=���       �	G��Xc�A�1*

loss=�<�Q�       �	=D�Xc�A�1*

loss��;�r       �	 ��Xc�A�1*

loss3�<�~��       �	`x�Xc�A�1*

lossoN�;�(Q       �	0K�Xc�A�1*

loss%.=<H]�-       �	}��Xc�A�1*

loss �<���       �	���Xc�A�1*

loss�T�;v�J       �	��Xc�A�1*

lossm]�<�*�       �	���Xc�A�1*

loss_N�<Q��Q       �	W]�Xc�A�1*

loss��/=k�'�       �	���Xc�A�1*

lossHO=�<6�       �	З�Xc�A�1*

lossD=Yi�5       �	�8�Xc�A�1*

loss
�T=��ɰ       �	���Xc�A�1*

lossFV <Z�ѩ       �	ܸ�Xc�A�1*

loss�s<�	�       �	�U�Xc�A�1*

loss�:<�NN       �	-��Xc�A�1*

loss�m�<>8~�       �	���Xc�A�1*

loss�:d;�zʊ       �	��Xc�A�1*

loss��V:$���       �	ٰ�Xc�A�1*

losso��<��=E       �	�E�Xc�A�1*

lossN�<�N�       �	���Xc�A�1*

lossw��<�M�       �	9��Xc�A�1*

loss:�7.|       �	�8�Xc�A�1*

loss�V�;W���       �	��Xc�A�1*

lossh�<'��]       �	Gv�Xc�A�1*

loss?o�<e�L       �	�Xc�A�1*

losso�;�L��       �	���Xc�A�1*

loss`�=�=S       �	�J�Xc�A�1*

loss��C;o8��       �	m��Xc�A�1*

loss�l<�7�       �	؂�Xc�A�1*

lossc�:@*Om       �	�V �Xc�A�1*

loss.�;33?�       �	�!�Xc�A�1*

lossԇ<Tj(       �	@�!�Xc�A�1*

loss`�+;�ny�       �	�E"�Xc�A�2*

loss]�R;a�?x       �	e�"�Xc�A�2*

lossd��<�!�       �	$�#�Xc�A�2*

loss&�&=���9       �	�$�Xc�A�2*

loss]^=�s�       �	@�$�Xc�A�2*

loss��)<n8       �	�c%�Xc�A�2*

loss��w;`K�       �	T &�Xc�A�2*

loss���<�2�       �	�&�Xc�A�2*

loss���;@%�R       �	P'�Xc�A�2*

lossL?�<4/�       �	��'�Xc�A�2*

loss���<1�@       �	ڌ(�Xc�A�2*

loss��<ͷx�       �	�5)�Xc�A�2*

lossY=BO|:       �	��)�Xc�A�2*

lossow;�')       �	��*�Xc�A�2*

lossQ��<d��       �	t%+�Xc�A�2*

loss�d�<ȳՕ       �	��,�Xc�A�2*

lossN1�< �ǫ       �	�e-�Xc�A�2*

loss�%j<@a�       �	G.�Xc�A�2*

loss&�<E[c       �	��.�Xc�A�2*

loss0�;2�so       �	>/�Xc�A�2*

loss��<<�)�       �	  0�Xc�A�2*

loss�#�;�S��       �	М0�Xc�A�2*

lossw��=��
       �	u1�Xc�A�2*

loss�.:�W+       �	|2�Xc�A�2*

loss�*=�]�N       �	4�2�Xc�A�2*

loss<x#=r       �	z43�Xc�A�2*

loss�k <%</h       �	��3�Xc�A�2*

loss�<�x��       �	Xp4�Xc�A�2*

loss3�=��y�       �	W5�Xc�A�2*

loss6<�V��       �	5�5�Xc�A�2*

loss��J;�Ф�       �	��6�Xc�A�2*

loss)�<��=X       �	�7�Xc�A�2*

loss�<��       �	��7�Xc�A�2*

loss��<΁$       �	��8�Xc�A�2*

loss�$=�Ϯ       �	�^9�Xc�A�2*

loss�s�<��db       �	��9�Xc�A�2*

loss��<k�\P       �	��:�Xc�A�2*

losszg�<���       �	�j;�Xc�A�2*

loss�G�<�]��       �	�<�Xc�A�2*

loss�S�<���r       �	f�<�Xc�A�2*

loss���;��c�       �	>�Xc�A�2*

loss��<=�U�       �	;�>�Xc�A�2*

loss)+v<z7@�       �	�D?�Xc�A�2*

loss�'�<�H9�       �		�?�Xc�A�2*

lossD�"=ƞ�       �	o�@�Xc�A�2*

loss19L;o�v2       �	J$A�Xc�A�2*

lossO�1<�؄�       �	�A�Xc�A�2*

loss�
�=B!'�       �	F[B�Xc�A�2*

lossЎ<����       �	��B�Xc�A�2*

lossχ�<)�h�       �	�C�Xc�A�2*

loss��<+���       �	/D�Xc�A�2*

loss`�@<j�       �	��D�Xc�A�2*

loss���<fN       �	��E�Xc�A�2*

loss��P;�Ɓ       �	*F�Xc�A�2*

lossnX;Z��       �	Y�F�Xc�A�2*

loss읪<�2�K       �	�VG�Xc�A�2*

loss܍2<n<l       �	�=H�Xc�A�2*

loss�<�9��       �	��H�Xc�A�2*

loss\�;�״�       �	עI�Xc�A�2*

loss%<X5       �	=CJ�Xc�A�2*

loss!�<���8       �	��J�Xc�A�2*

loss�w%<Ͽ�       �	�yK�Xc�A�2*

loss�A�;�w�T       �	wiL�Xc�A�2*

lossı;�wP|       �	R,M�Xc�A�2*

loss�q<���P       �	�
N�Xc�A�2*

loss�$ <�8�r       �	.9O�Xc�A�2*

loss=�j< Z�@       �	�	P�Xc�A�2*

loss�^<T�*�       �	EQ�Xc�A�2*

loss��1;q�       �	��Q�Xc�A�2*

loss�{:;�2F�       �	�9S�Xc�A�2*

loss��;>���       �	�S�Xc�A�2*

loss׵+=j ,       �	5{T�Xc�A�2*

loss(l<��[�       �	�&U�Xc�A�2*

loss�{�<��       �	r�U�Xc�A�2*

loss\��;�aF[       �	�V�Xc�A�2*

loss�Հ;؎��       �	) W�Xc�A�2*

lossI�;�K��       �	��W�Xc�A�2*

loss��;�3��       �	_X�Xc�A�2*

lossq~�;���       �	Q�X�Xc�A�2*

loss&: <{���       �	��Y�Xc�A�2*

loss���<�n�T       �	ρZ�Xc�A�2*

loss�hp<ub9       �	�[�Xc�A�2*

lossS<2ܲ�       �	��[�Xc�A�2*

loss��;���       �	4�\�Xc�A�2*

loss!�'<(�o       �	��]�Xc�A�2*

loss�.�<g���       �	�5^�Xc�A�2*

loss=	V=�       �	�^�Xc�A�2*

lossf^@:y�J       �	R}_�Xc�A�2*

loss*i�<�4O�       �	�`�Xc�A�2*

loss���<wG@-       �	D�`�Xc�A�2*

loss�9+<`��       �	Ͱa�Xc�A�2*

loss��=jFg�       �	�Sb�Xc�A�2*

loss3��;��@       �	q�b�Xc�A�2*

lossOvJ<��S�       �	��c�Xc�A�2*

loss\��<�p��       �	іd�Xc�A�2*

loss�`&; ���       �	6e�Xc�A�2*

loss�#;�$��       �		�e�Xc�A�2*

loss��</y3�       �	҉f�Xc�A�2*

loss��=���       �	�7g�Xc�A�2*

loss=~�;ÄM       �	J`h�Xc�A�2*

loss�j�;N�y>       �	��h�Xc�A�2*

loss��;W��       �	�i�Xc�A�2*

loss��:��b       �	 7j�Xc�A�2*

loss-�'<7h/X       �	�j�Xc�A�2*

lossHI;���*       �	�sk�Xc�A�2*

lossNc<���o       �	�l�Xc�A�2*

loss��<7#F       �	��l�Xc�A�2*

loss�<�)u       �	Xm�Xc�A�2*

loss��7:*�bL       �	An�Xc�A�2*

loss��D<�L�       �	6�n�Xc�A�2*

loss@��<�`@�       �	-	p�Xc�A�2*

lossA@\<=�5�       �	��p�Xc�A�2*

loss��3;�Z�"       �	UMq�Xc�A�2*

loss;kR�       �	�r�Xc�A�2*

loss)��;���       �	�r�Xc�A�2*

loss���<�S>"       �	�Js�Xc�A�2*

loss�0m=1ʝb       �	At�Xc�A�2*

loss��< ��       �	�0u�Xc�A�2*

loss�I;����       �	��u�Xc�A�2*

loss}� <�f�       �	��v�Xc�A�2*

loss_ib<^F�       �	�-w�Xc�A�2*

loss���<�0\w       �	��w�Xc�A�2*

loss�)n<.tD	       �	gex�Xc�A�2*

lossS�:ޠ       �	�y�Xc�A�2*

loss�1;����       �	v�y�Xc�A�2*

lossLR<j��s       �	"Sz�Xc�A�2*

lossM�0;��5       �	
�z�Xc�A�2*

loss��;�+F�       �	ܜ{�Xc�A�2*

loss�9�;^K��       �	�e|�Xc�A�3*

loss�m|<�У       �	�%}�Xc�A�3*

loss:�<U:�1       �	n�}�Xc�A�3*

loss/�;��/�       �	]~�Xc�A�3*

loss]@=l       �	��~�Xc�A�3*

loss%S�:���       �	Փ�Xc�A�3*

loss��8 �d       �	�:��Xc�A�3*

loss�@�9$�6�       �	�Ԁ�Xc�A�3*

lossN�O=.{�d       �	l��Xc�A�3*

lossV�p;���       �	o��Xc�A�3*

loss�:R;_��       �	����Xc�A�3*

lossLk�9�$^       �	�Q��Xc�A�3*

loss�tH<N�s�       �	����Xc�A�3*

loss��(:�s�       �	a���Xc�A�3*

loss_G9�o�,       �	(D��Xc�A�3*

loss���8Ҳ]�       �	�	��Xc�A�3*

loss���:���       �	r���Xc�A�3*

lossc؎;�xK�       �	5A��Xc�A�3*

lossp<��W�       �	AՇ�Xc�A�3*

loss$�:�)N       �	���Xc�A�3*

lossi6�;0w�       �	�:��Xc�A�3*

loss $>CE��       �	���Xc�A�3*

loss��;�H��       �	�|��Xc�A�3*

loss�C-=��       �	��Xc�A�3*

loss�<���       �	P���Xc�A�3*

loss�e�<��|       �	C��Xc�A�3*

loss�n.;�G�       �	�܌�Xc�A�3*

loss3S�;O�J       �	Cp��Xc�A�3*

loss�4�;�XD       �	@��Xc�A�3*

loss�K�<�E��       �	龜�Xc�A�3*

lossT�;z�c       �	Uj��Xc�A�3*

loss
g�<R+�       �	 ��Xc�A�3*

loss]a<���       �	�Xc�A�3*

lossJ�<���D       �	�<��Xc�A�3*

loss	 �;�=V�       �	�ԑ�Xc�A�3*

loss��;bo��       �	<l��Xc�A�3*

lossC��;h�)R       �	���Xc�A�3*

loss��_=�Z9�       �	cГ�Xc�A�3*

loss�'�;+%{�       �	&o��Xc�A�3*

lossJ��=�ֹh       �	���Xc�A�3*

loss&�<���+       �	f���Xc�A�3*

lossY<��B�       �	D���Xc�A�3*

loss(\_;\?��       �	���Xc�A�3*

lossX@�<��R$       �	�ɗ�Xc�A�3*

loss��;���~       �	�^��Xc�A�3*

loss�4�<�[��       �	a���Xc�A�3*

loss�H�:��4       �	ޏ��Xc�A�3*

loss���:�4L       �	c)��Xc�A�3*

loss�y�<��9�       �	�ǚ�Xc�A�3*

loss�)=���[       �	9d��Xc�A�3*

loss}L;=Χ��       �	.���Xc�A�3*

loss�0"="�       �	S���Xc�A�3*

loss���;��a�       �	�+��Xc�A�3*

loss�9<���n       �	7�Xc�A�3*

loss��;?!��       �	����Xc�A�3*

loss�M&;�kO^       �	�$��Xc�A�3*

loss̑=�Tť       �	(���Xc�A�3*

loss���;��b       �	�N��Xc�A�3*

loss8d<\s]       �	���Xc�A�3*

loss�!=��|�       �	t��Xc�A�3*

loss&�_<��1       �	j���Xc�A�3*

lossj�;~���       �	rN��Xc�A�3*

loss
ؙ:kFFU       �	���Xc�A�3*

lossMx�:��I�       �	���Xc�A�3*

losss�=�Lc       �	~��Xc�A�3*

loss=�<|�xd       �	���Xc�A�3*

loss��<^R�       �	.Ŧ�Xc�A�3*

lossiֺ<Q��       �	�k��Xc�A�3*

lossV�o<@��       �	6��Xc�A�3*

loss��d;�BG^       �	ǂ��Xc�A�3*

loss#<&sy�       �	���Xc�A�3*

loss��;��|�       �	ճ��Xc�A�3*

loss��';��vj       �	mU��Xc�A�3*

loss{�;H N5       �	�/��Xc�A�3*

loss���=�B��       �	���Xc�A�3*

loss�o<�~�Q       �	�e��Xc�A�3*

loss���;a�%H       �	���Xc�A�3*

loss߿@<MG6       �	R���Xc�A�3*

lossp�;z*�       �	|b��Xc�A�3*

lossi�`<����       �	��Xc�A�3*

loss;��<�       �	*���Xc�A�3*

loss�.=�g@�       �	N��Xc�A�3*

loss�Q�<y� u       �	h���Xc�A�3*

loss�1Z;�~�       �	7���Xc�A�3*

lossȪ=�z�       �	�,��Xc�A�3*

loss64�<����       �	����Xc�A�3*

loss�)<n�#�       �	�p��Xc�A�3*

loss��=�-C       �	�	��Xc�A�3*

lossz�<! �o       �	���Xc�A�3*

loss�!2:���       �	C���Xc�A�3*

loss 1�;���/       �	�K��Xc�A�3*

loss݋<�<�}       �	p���Xc�A�3*

loss��<��j       �	�,��Xc�A�3*

loss�J<q{       �	&���Xc�A�3*

loss���;�j�t       �	g`��Xc�A�3*

loss�;���       �	���Xc�A�3*

lossY�<�̍�       �	����Xc�A�3*

losst��; �g�       �	>{��Xc�A�3*

loss�?v;�l}       �	!��Xc�A�3*

loss���<�<�       �	k���Xc�A�3*

loss��H:��W       �	����Xc�A�3*

loss�7�<�d�       �	����Xc�A�3*

lossm܂;܉�I       �	�0��Xc�A�3*

lossg<��iA       �	����Xc�A�3*

loss��K;��[       �	���Xc�A�3*

loss\PH<��f�       �	�+��Xc�A�3*

loss��J=���       �	����Xc�A�3*

loss�w�;H��       �	�x��Xc�A�3*

loss_�,=���@       �	���Xc�A�3*

lossD��9� ��       �	����Xc�A�3*

loss�k<�w�U       �	h��Xc�A�3*

loss�F=6"wH       �	t��Xc�A�3*

lossT�7<Q?�       �	���Xc�A�3*

loss�p�;�+|�       �	n5��Xc�A�3*

loss�h�;�Y�       �	����Xc�A�3*

loss��"<�?S       �	A}��Xc�A�3*

loss=�<���       �	�+��Xc�A�3*

loss��S;�ƭ       �	���Xc�A�3*

loss<�i<Y��W       �	�t��Xc�A�3*

loss�[�<�{	�       �	@��Xc�A�3*

loss�vp;H��       �	���Xc�A�3*

loss�V�:��x       �	'O��Xc�A�3*

loss�,�9{@6�       �	����Xc�A�3*

loss�[<݅)X       �	ڐ��Xc�A�3*

loss%�p<��?f       �	S?��Xc�A�3*

lossMr&<A1O�       �	j���Xc�A�3*

loss���=&��       �	_~��Xc�A�3*

loss��S=���       �	K"��Xc�A�3*

loss��)<L_AJ       �	����Xc�A�4*

loss��	;�
�       �	_��Xc�A�4*

loss�V�:�|��       �	��Xc�A�4*

loss{�=a��.       �	����Xc�A�4*

loss�,�<��       �	_z��Xc�A�4*

loss_F�=7�y�       �	�'��Xc�A�4*

lossC*�;���       �	$���Xc�A�4*

loss-3�;��8       �	;o��Xc�A�4*

loss�S=��P       �	v��Xc�A�4*

lossf�;���h       �	ܻ��Xc�A�4*

loss�u&<q�       �	ܸ��Xc�A�4*

loss�%�;�"�       �	Vf��Xc�A�4*

loss_�<��D�       �	�B��Xc�A�4*

loss���=d��       �	����Xc�A�4*

loss$Km=Π�.       �	����Xc�A�4*

loss3��;}��       �	�2��Xc�A�4*

loss7X�<׼,T       �	%���Xc�A�4*

loss�� <��X�       �	����Xc�A�4*

loss*�;\,e�       �	�/��Xc�A�4*

loss�wH< 71W       �	s���Xc�A�4*

lossC]�;��fk       �	����Xc�A�4*

loss)@e=�=�       �	o.��Xc�A�4*

lossjRv;L6��       �	����Xc�A�4*

loss�k;ށD�       �	��Xc�A�4*

loss�=C$��       �	&��Xc�A�4*

lossf�N<�1�       �	����Xc�A�4*

lossάj;7�u�       �	�o��Xc�A�4*

loss�5�<�P�h       �	��Xc�A�4*

loss�R;H���       �	����Xc�A�4*

loss+2<�X       �	b��Xc�A�4*

loss��
=EC��       �	�1��Xc�A�4*

loss��;��j>       �	F���Xc�A�4*

loss{�c<�=�       �	�r��Xc�A�4*

lossq<���       �	���Xc�A�4*

loss��Z=ɗYP       �	/���Xc�A�4*

loss�k�;���       �	}[��Xc�A�4*

loss���<���       �	r���Xc�A�4*

loss��_<M�>�       �	�� �Xc�A�4*

loss�'o<R<�       �	�B�Xc�A�4*

loss��=�6p       �	+��Xc�A�4*

loss�=�8�X       �	�{�Xc�A�4*

loss\Ki=ċ��       �	� �Xc�A�4*

loss��<����       �	o��Xc�A�4*

lossSU�<��D       �	�Z�Xc�A�4*

loss��G<�C��       �	���Xc�A�4*

loss2o�;�)�       �	��Xc�A�4*

loss�;�5�N       �	�(�Xc�A�4*

lossQ��;o��i       �	���Xc�A�4*

loss,��<�ݓ1       �	,a�Xc�A�4*

loss�C=�@��       �	���Xc�A�4*

loss��$=���h       �	���Xc�A�4*

loss��<�
       �	g	�Xc�A�4*

loss��D<�4�o       �	�
�Xc�A�4*

loss�ϧ<��a�       �	�
�Xc�A�4*

loss��;|��C       �	�G�Xc�A�4*

loss@	>=��z       �	���Xc�A�4*

loss��<�Ցt       �	��Xc�A�4*

lossZO<�=��       �	�I�Xc�A�4*

loss��;��$T       �	>��Xc�A�4*

loss� �:���       �	���Xc�A�4*

loss���:�Ry�       �	9��Xc�A�4*

loss�E�;���0       �	Sz�Xc�A�4*

loss�q�<�5�A       �	�>�Xc�A�4*

loss��<�T       �	���Xc�A�4*

loss�Ҩ<��@       �	}��Xc�A�4*

loss��
<����       �	1��Xc�A�4*

loss��t<|�Bn       �	�X�Xc�A�4*

loss���;:ø�       �	G�Xc�A�4*

lossc_<��-�       �	���Xc�A�4*

loss�<*��       �	���Xc�A�4*

loss@�:<��       �	�A�Xc�A�4*

lossz�{<W��       �	���Xc�A�4*

loss�_E;�4}M       �	2��Xc�A�4*

loss�!|<��D�       �	�.�Xc�A�4*

loss�*�<%�       �	@��Xc�A�4*

losshN�;h�3       �	ʥ�Xc�A�4*

loss�:�<)%6F       �	�M�Xc�A�4*

loss���<H��	       �	���Xc�A�4*

loss��M=�幪       �	��Xc�A�4*

loss��=�p��       �	�A�Xc�A�4*

loss���;u$S�       �	���Xc�A�4*

loss!�|<V�n       �	���Xc�A�4*

loss���;���       �	Q1�Xc�A�4*

loss��;MU�       �	���Xc�A�4*

loss&�<��       �	z� �Xc�A�4*

loss쑿;P��       �	t]!�Xc�A�4*

loss@
�<B'�       �	�"�Xc�A�4*

loss�**<�59       �	[�"�Xc�A�4*

loss�M<)���       �	�O#�Xc�A�4*

loss6�)=���       �	��#�Xc�A�4*

lossQ�<�Y6       �	�$�Xc�A�4*

loss��;�D�       �	q8%�Xc�A�4*

lossDg<�
       �	��%�Xc�A�4*

lossWr9;A/��       �	�i&�Xc�A�4*

lossN��;��r       �	�'�Xc�A�4*

lossQF=�D!_       �	%�(�Xc�A�4*

loss�z�<��)       �	&s)�Xc�A�4*

loss@�<Y�g       �	�*�Xc�A�4*

loss�E=�P��       �	�*�Xc�A�4*

loss;��<��r�       �	&�+�Xc�A�4*

loss��;�l"       �	^.,�Xc�A�4*

loss\�k:{       �	��,�Xc�A�4*

loss1�><0��       �	�l-�Xc�A�4*

loss ֹ<K��       �	�4/�Xc�A�4*

loss���;�z��       �	#�/�Xc�A�4*

loss.�r<z��       �	�0�Xc�A�4*

loss��
<ad��       �	�(1�Xc�A�4*

loss?�<�5��       �	��1�Xc�A�4*

lossOi�9���       �	+n2�Xc�A�4*

loss)�;<�6P       �	�	3�Xc�A�4*

loss�j;s3v�       �	��3�Xc�A�4*

loss��i:*���       �	iW4�Xc�A�4*

loss��*=�?E       �	P 5�Xc�A�4*

loss�L;�Ζ�       �	��5�Xc�A�4*

loss��f;��       �	�76�Xc�A�4*

loss��<�]�       �	<�6�Xc�A�4*

loss��B;�*?        �	*�7�Xc�A�4*

loss�H1;b]^=       �	?8�Xc�A�4*

loss#a<gg�       �	f�8�Xc�A�4*

lossVg.<e=ў       �	z9�Xc�A�4*

loss,/<Vz�       �	L:�Xc�A�4*

loss�;�ݫ�       �	��:�Xc�A�4*

loss�%�;#��[       �	�M;�Xc�A�4*

loss�7C<o�{4       �	�-<�Xc�A�4*

loss}*:<d>�       �	q�<�Xc�A�4*

loss�~�9k3�}       �	J_=�Xc�A�4*

lossnQ�=R��0       �	7�=�Xc�A�4*

loss�Қ< ��       �	Þ>�Xc�A�4*

loss:{�:)��       �	u<?�Xc�A�5*

lossȱ;���       �		�?�Xc�A�5*

loss��:�/%       �	�@�Xc�A�5*

loss���<����       �	()A�Xc�A�5*

loss�Z:<��P�       �	��A�Xc�A�5*

lossľ<��s�       �	�gB�Xc�A�5*

lossʘ@<Mϖ�       �	�C�Xc�A�5*

loss���< x�f       �	��C�Xc�A�5*

loss�Vd<X���       �	7D�Xc�A�5*

lossݿ;Kv�r       �	�D�Xc�A�5*

lossH��;�A�       �	nE�Xc�A�5*

loss~<�OX�       �	�F�Xc�A�5*

losse<KN�       �	�F�Xc�A�5*

loss;�;J��       �	�VG�Xc�A�5*

loss�W�<q�JR       �	J�G�Xc�A�5*

loss��<S���       �	�H�Xc�A�5*

loss��k;N#�.       �	�#I�Xc�A�5*

loss� �<��r       �	4�I�Xc�A�5*

losst�X:���e       �	�\J�Xc�A�5*

loss�7�<5 H�       �	�J�Xc�A�5*

losst=�)N       �	�K�Xc�A�5*

loss߯b;���[       �	�3L�Xc�A�5*

loss�?B<O
�       �	��L�Xc�A�5*

loss��I<��q       �	=aM�Xc�A�5*

loss�s�<	�       �	��M�Xc�A�5*

lossF(�;� �       �	��N�Xc�A�5*

loss�J�<�תL       �	d#O�Xc�A�5*

loss��;Պ*Z       �	�P�Xc�A�5*

lossI�I<��6�       �	ףP�Xc�A�5*

lossT�8=���       �	�EQ�Xc�A�5*

loss�{i<8Ɵ       �	T�Q�Xc�A�5*

loss?�U=h��b       �	vR�Xc�A�5*

loss�2�=�@0	       �	aS�Xc�A�5*

loss���<VzО       �	��S�Xc�A�5*

lossr��=��V�       �	�oT�Xc�A�5*

loss2��<̨V       �	jU�Xc�A�5*

loss��N<m��0       �	�U�Xc�A�5*

loss�\�;Wʼ�       �	vSV�Xc�A�5*

loss�`=�Y��       �	Z�V�Xc�A�5*

loss8+�<��4       �	a�W�Xc�A�5*

lossvm�<ڌ��       �	
-X�Xc�A�5*

loss��=*9қ       �	��X�Xc�A�5*

lossX8=�7B       �	�jY�Xc�A�5*

losse��;C{�       �	 
Z�Xc�A�5*

loss`F�<���       �	�[�Xc�A�5*

loss8�;p*9       �	��[�Xc�A�5*

loss�bu:��_=       �	lA\�Xc�A�5*

lossԀ<�.
�       �	��\�Xc�A�5*

lossLH=B��X       �	ty]�Xc�A�5*

loss��;�z�       �	�^�Xc�A�5*

loss���;��8g       �	�^�Xc�A�5*

loss�n4<�y�       �	tD_�Xc�A�5*

loss�_�<�!t       �	�_�Xc�A�5*

lossO�=���       �	*�`�Xc�A�5*

lossי�<�)D       �	$(a�Xc�A�5*

lossZ�<��P       �	��a�Xc�A�5*

lossO��:}x�P       �	=|b�Xc�A�5*

loss�E�;L[e       �	ac�Xc�A�5*

loss,b=f���       �	j�c�Xc�A�5*

losss�8=��¥       �	�fd�Xc�A�5*

loss�jV<\e"�       �	�e�Xc�A�5*

loss/m�;�s��       �	��e�Xc�A�5*

loss��7< �<1       �	JAf�Xc�A�5*

loss� �;�#�b       �	��f�Xc�A�5*

loss�9�<��E�       �	hh�Xc�A�5*

loss�m�=A^       �	��h�Xc�A�5*

loss<�<2�!�       �	Ni�Xc�A�5*

loss�'R<�m       �	J�i�Xc�A�5*

loss�&�<����       �	g�j�Xc�A�5*

lossF��;$oL~       �	oJk�Xc�A�5*

losstC";�n�       �	��k�Xc�A�5*

lossψ�:G+$�       �	�l�Xc�A�5*

lossO��<3��       �	�Nm�Xc�A�5*

loss�=�<c�[�       �	��m�Xc�A�5*

lossO��<�5��       �	�n�Xc�A�5*

loss�&;�O�z       �	�To�Xc�A�5*

lossUa�<p�       �	�p�Xc�A�5*

loss1�T;h&��       �	ؼp�Xc�A�5*

lossJ�=}#N       �	�aq�Xc�A�5*

loss�; mT�       �	�r�Xc�A�5*

loss	�;����       �	��r�Xc�A�5*

loss\��<j��g       �	Vs�Xc�A�5*

loss`o�<�~$c       �	��s�Xc�A�5*

loss�"{=���       �	Χt�Xc�A�5*

loss@�k;��_#       �	iWu�Xc�A�5*

lossW5�;�[�       �	��u�Xc�A�5*

loss��>=�2��       �	��v�Xc�A�5*

loss�I;�)!v       �	:w�Xc�A�5*

loss̄�;���       �	��w�Xc�A�5*

loss�^<��6�       �	�x�Xc�A�5*

loss[�;'�       �	�y�Xc�A�5*

lossU�<⦳G       �	 �y�Xc�A�5*

loss���<T��T       �	�kz�Xc�A�5*

loss���;��<       �	�{�Xc�A�5*

losss�;ϐȑ       �	ܡ{�Xc�A�5*

loss�a<_�	       �	�>|�Xc�A�5*

loss}�=H�       �	��|�Xc�A�5*

loss��T;���       �	��}�Xc�A�5*

loss/Ͱ;b9�0       �	�*~�Xc�A�5*

loss�[<��       �	�~�Xc�A�5*

lossD��;'i�       �	�\�Xc�A�5*

loss.@<�;�)       �	���Xc�A�5*

loss�K<���0       �	����Xc�A�5*

loss�dv<ϓ       �	�&��Xc�A�5*

loss�]�;�P}�       �	����Xc�A�5*

lossn�;��       �	[��Xc�A�5*

loss�Ҫ:��	�       �	���Xc�A�5*

loss6�|:or��       �	���Xc�A�5*

lossJ��<�m�       �	3��Xc�A�5*

loss�0*<(�$�       �	݄�Xc�A�5*

loss�r�:��5       �	~��Xc�A�5*

loss]m<���       �	�(��Xc�A�5*

lossܧ�<�jw       �	����Xc�A�5*

lossݒ;)�j�       �	���Xc�A�5*

loss}��< �g�       �	�+��Xc�A�5*

loss,ф=N��       �	p��Xc�A�5*

loss��Y<��       �	0���Xc�A�5*

lossvp�<)��$       �	���Xc�A�5*

loss�J�<�D��       �	¾��Xc�A�5*

lossHh�<F�       �	XW��Xc�A�5*

lossIG�=E-L       �	��Xc�A�5*

loss�]A< N       �	p���Xc�A�5*

lossam=5�lF       �	�W��Xc�A�5*

loss��#;�6        �	
���Xc�A�5*

loss'<�F`       �	gH��Xc�A�5*

loss:S�;Z��       �	|(��Xc�A�5*

loss�@;�0��       �	���Xc�A�5*

loss]x�:�r҆       �	��Xc�A�5*

loss�<��ǐ       �	�2��Xc�A�6*

loss�|-<�K~�       �	ڒ�Xc�A�6*

loss�k�;r�6       �	pw��Xc�A�6*

loss��;��k       �	�)��Xc�A�6*

losssz1<�q�       �	�ǔ�Xc�A�6*

loss.@<L7(�       �	�j��Xc�A�6*

loss�>;y��f       �	���Xc�A�6*

loss)ۡ<¦V
       �	G���Xc�A�6*

loss:�;���2       �	�M��Xc�A�6*

loss	��<�}�]       �	���Xc�A�6*

lossN�<ũN       �	���Xc�A�6*

loss�L�<:hc0       �	-��Xc�A�6*

loss$�<�       �	+4��Xc�A�6*

loss���<��M�       �	�Κ�Xc�A�6*

lossڱF=z,��       �	�i��Xc�A�6*

loss�	�;�˦$       �	X��Xc�A�6*

loss�v�<Tv;�       �	#���Xc�A�6*

loss��,<�o       �	�4��Xc�A�6*

loss�I9<Ab�[       �	�Ν�Xc�A�6*

loss�"<���       �	�e��Xc�A�6*

loss���<+��       �	�B��Xc�A�6*

loss��:< ���       �	�ޟ�Xc�A�6*

loss�	.<���       �	���Xc�A�6*

loss�'"=���       �	2��Xc�A�6*

lossR�=;늲�       �	�ϡ�Xc�A�6*

loss�9<���;       �	"l��Xc�A�6*

lossc�<��       �	���Xc�A�6*

loss��9<�y~       �	,ԣ�Xc�A�6*

loss�3:���V       �	���Xc�A�6*

loss���;̜��       �	m��Xc�A�6*

loss�t<lR�       �	�ɥ�Xc�A�6*

loss��;�7       �	�q��Xc�A�6*

loss<I�;P:,       �	i��Xc�A�6*

loss���<!	�       �	ħ�Xc�A�6*

loss�X4=��:       �	�a��Xc�A�6*

loss�~�;�F�       �	\��Xc�A�6*

loss���;Џ       �	=���Xc�A�6*

loss�m8<���       �	?7��Xc�A�6*

lossۯ5<ܱw       �	�ժ�Xc�A�6*

loss���:���       �	����Xc�A�6*

loss�Ҁ<w��       �	�+��Xc�A�6*

loss�=�!'C       �	f��Xc�A�6*

loss�	<�p��       �	���Xc�A�6*

lossSk�=����       �	���Xc�A�6*

loss��B<��<�       �	1^��Xc�A�6*

loss���:�?�?       �	m��Xc�A�6*

loss[S�;<61�       �	���Xc�A�6*

loss�=���       �	=���Xc�A�6*

loss�� ;��       �	�_��Xc�A�6*

loss���<D��       �	���Xc�A�6*

loss2�:�5�       �	h���Xc�A�6*

loss�.�;.�A       �	���Xc�A�6*

lossTRj=�s�       �	�K��Xc�A�6*

lossr�<a�|o       �	X��Xc�A�6*

loss��=��F�       �	W���Xc�A�6*

lossD~=�V��       �	`��Xc�A�6*

lossIԩ;r���       �	���Xc�A�6*

loss��=m`�       �	{���Xc�A�6*

loss���:0�.       �	mt��Xc�A�6*

loss��<?��       �	�&��Xc�A�6*

lossM�<w
JR       �	�ߺ�Xc�A�6*

lossa�;|17�       �	&���Xc�A�6*

loss���;���%       �	73��Xc�A�6*

lossTW�;����       �	�Ѽ�Xc�A�6*

lossJ�;+U\       �	Z���Xc�A�6*

loss���<~
%       �	_$��Xc�A�6*

loss�<ȩHH       �	_Ծ�Xc�A�6*

lossM��<�j       �	�z��Xc�A�6*

loss�7�;���       �	�*��Xc�A�6*

loss� ;��*       �	����Xc�A�6*

lossR'(;�^X       �	w��Xc�A�6*

loss��<�-M�       �	��Xc�A�6*

lossd�<)a
�       �	B���Xc�A�6*

loss��c:_�3       �	nR��Xc�A�6*

loss �=�Y��       �	����Xc�A�6*

loss��:</��       �	����Xc�A�6*

losshπ<,݌K       �	^+��Xc�A�6*

lossN�6:7�T       �	����Xc�A�6*

loss��<;����       �	h��Xc�A�6*

loss ��<��^:       �	���Xc�A�6*

lossS�;߉�)       �	���Xc�A�6*

lossE��:V�$       �	6?��Xc�A�6*

loss/w;k���       �	j��Xc�A�6*

loss��=BI�       �	����Xc�A�6*

loss�=RH       �	fL��Xc�A�6*

losshT�:��e       �	A���Xc�A�6*

loss��:���P       �	-���Xc�A�6*

loss��<�G��       �	�?��Xc�A�6*

loss�ۖ;%�W�       �	����Xc�A�6*

loss��&;*f�.       �	����Xc�A�6*

lossE{<�F       �	���Xc�A�6*

loss὎<�F��       �	us��Xc�A�6*

loss���;6z�       �	�}��Xc�A�6*

lossB<���       �	�D��Xc�A�6*

loss�*�<��6       �	����Xc�A�6*

loss���<�M��       �	0���Xc�A�6*

loss�^<����       �	�=��Xc�A�6*

loss{�K<2�Y�       �	@���Xc�A�6*

loss��:�Gpv       �	�s��Xc�A�6*

loss	-�<5�N(       �	��Xc�A�6*

loss�-=2_"       �	����Xc�A�6*

loss,�;e��
       �	�P��Xc�A�6*

loss}�W<���?       �	����Xc�A�6*

loss�j�;��       �	����Xc�A�6*

loss�=�~N       �	G ��Xc�A�6*

lossf�:�Efs       �	˻��Xc�A�6*

loss��E<��{       �	}^��Xc�A�6*

loss��;|���       �	����Xc�A�6*

loss:�<B���       �	P���Xc�A�6*

loss�YO<��
       �	J'��Xc�A�6*

lossW�k<���       �	�,��Xc�A�6*

loss�"�:���       �	9F��Xc�A�6*

loss��	=;�s�       �	����Xc�A�6*

loss�a�<'
a       �	2w��Xc�A�6*

lossP"</:�       �	��Xc�A�6*

lossmBI;�#-       �	ݶ��Xc�A�6*

loss[=�<z�:L       �	�N��Xc�A�6*

lossn�X;x�f�       �	����Xc�A�6*

loss/��<KAÓ       �	N~��Xc�A�6*

loss�R<��}b       �	G��Xc�A�6*

loss��<O��e       �	A���Xc�A�6*

loss~q;U��       �	QN��Xc�A�6*

losss=��%�       �	i���Xc�A�6*

loss��<��9�       �	E���Xc�A�6*

loss���<����       �	���Xc�A�6*

loss	�<i5~       �	k���Xc�A�6*

lossf)�<�aN�       �	tb��Xc�A�6*

loss�;D;���       �	a���Xc�A�6*

loss�*T;�3e�       �	����Xc�A�7*

lossihL=_��       �	,a��Xc�A�7*

loss�=�)�       �	���Xc�A�7*

losst=cPDS       �	d#��Xc�A�7*

loss,xm<� �       �	���Xc�A�7*

loss
P�;�$��       �	u��Xc�A�7*

loss��<K/�       �	���Xc�A�7*

lossDu=����       �	���Xc�A�7*

lossr�(<h��|       �	aQ��Xc�A�7*

loss��<W�       �	� ��Xc�A�7*

loss��<�0�	       �	Й��Xc�A�7*

loss�)J=>��       �	bH��Xc�A�7*

loss��<�$��       �	����Xc�A�7*

loss�C�:�[q       �	P���Xc�A�7*

loss���<��1�       �	Y2��Xc�A�7*

loss�=G4_�       �	���Xc�A�7*

loss��0=���       �	k���Xc�A�7*

loss���=S�ґ       �	�l��Xc�A�7*

lossY;{k       �	���Xc�A�7*

loss�M�;"��<       �	����Xc�A�7*

losshyE<�oH       �	]R��Xc�A�7*

loss�u�;�j�       �	I���Xc�A�7*

loss��3<ߡ�       �	N���Xc�A�7*

loss���;W���       �	����Xc�A�7*

losszv<���       �	:��Xc�A�7*

lossf�><t`�       �	A���Xc�A�7*

lossdN~<g��       �	���Xc�A�7*

loss_�a<��       �	���Xc�A�7*

loss�t�;x�4�       �	����Xc�A�7*

lossw��:��^�       �	�a��Xc�A�7*

loss[j;�9e�       �	���Xc�A�7*

loss��<�̶       �	Ւ��Xc�A�7*

loss�;=�3�q       �	w+��Xc�A�7*

loss�h�<�/��       �	����Xc�A�7*

loss��z<6e��       �	.U��Xc�A�7*

losswj];t*��       �	����Xc�A�7*

loss�h;%I˶       �	;���Xc�A�7*

loss|��;����       �	�D �Xc�A�7*

loss%S�;�ǜ@       �	�� �Xc�A�7*

lossT-<����       �	j��Xc�A�7*

loss��=a63       �	!�Xc�A�7*

loss�G�<��       �	,��Xc�A�7*

loss���<_K�       �	�Q�Xc�A�7*

lossʺ;<gm�       �	_�Xc�A�7*

loss���:�m�X       �	��Xc�A�7*

loss�Z<{6'�       �	d;�Xc�A�7*

loss���;��a       �	���Xc�A�7*

loss�;��       �	o�Xc�A�7*

loss��<���       �	��Xc�A�7*

loss���;M 8       �	x��Xc�A�7*

loss�U�<�Ѝ       �	�5�Xc�A�7*

lossW��<��Q+       �	:��Xc�A�7*

loss?�E;4���       �	�h	�Xc�A�7*

lossX�;s��       �	d
�Xc�A�7*

loss��;��q{       �	xb�Xc�A�7*

loss��<��܉       �	c�Xc�A�7*

lossI�;bM��       �	��Xc�A�7*

loss��M<�       �	�G�Xc�A�7*

loss3�Q<��r       �	���Xc�A�7*

loss �;�+Q�       �	���Xc�A�7*

loss��<�6(       �	�V�Xc�A�7*

loss��r<5�"       �	~��Xc�A�7*

loss8V:<0       �	C��Xc�A�7*

lossN<M^t       �	�c�Xc�A�7*

lossz;�� Q       �	���Xc�A�7*

loss�͒<r��A       �	ˡ�Xc�A�7*

lossݼH<�N�       �	�>�Xc�A�7*

loss���<cs�       �	���Xc�A�7*

loss=k�:A/�g       �	>y�Xc�A�7*

loss�֗;�/;�       �	��Xc�A�7*

loss�<��+       �	ɭ�Xc�A�7*

lossJr�;_�!O       �	�G�Xc�A�7*

loss��;j�k       �	���Xc�A�7*

lossmM�<�ѽ�       �	�|�Xc�A�7*

loss�G�;�        �	�$�Xc�A�7*

lossr�<��8       �	_��Xc�A�7*

loss=�<�h�H       �	Ps�Xc�A�7*

loss傛<�^       �	�(�Xc�A�7*

lossM��:��       �	?��Xc�A�7*

loss��m;��bB       �	�m�Xc�A�7*

loss�>;PdM�       �	��Xc�A�7*

lossx�;� �       �	:��Xc�A�7*

loss���;:D�1       �	YR�Xc�A�7*

loss�"�9�'
       �	��Xc�A�7*

loss=Y�;N?�       �	J��Xc�A�7*

loss%��<d��       �	�> �Xc�A�7*

loss���:�oD       �	=e!�Xc�A�7*

loss��;*��       �	�"�Xc�A�7*

loss��;`��&       �	��"�Xc�A�7*

loss&�=�B��       �	�I#�Xc�A�7*

loss/��<j��\       �	��#�Xc�A�7*

lossD�<�@~0       �	#�$�Xc�A�7*

loss�ϴ<��"�       �	� %�Xc�A�7*

loss͎(;��A       �	�%�Xc�A�7*

loss��8 K;       �	ޏ&�Xc�A�7*

loss�W�:!�y�       �	a3'�Xc�A�7*

loss?�D:�[�       �	��'�Xc�A�7*

lossb;�V)       �	c(�Xc�A�7*

loss��6;Ǥ�
       �	8,)�Xc�A�7*

loss�8${��       �	d�)�Xc�A�7*

lossX/�;���       �	�m*�Xc�A�7*

lossi)�:��xO       �	�+�Xc�A�7*

loss�*:��
-       �	�+�Xc�A�7*

loss��8�
�       �	mY,�Xc�A�7*

loss1�N:���       �	n�,�Xc�A�7*

loss���;yv5+       �	�-�Xc�A�7*

lossf�; ���       �	y:.�Xc�A�7*

loss&-;��^       �	��.�Xc�A�7*

loss5��=laa       �	Z�/�Xc�A�7*

loss�X=9��\       �	#0�Xc�A�7*

lossq�c;��       �	�0�Xc�A�7*

loss;��<A�b       �	�a1�Xc�A�7*

lossZW�;`��       �	J2�Xc�A�7*

loss���;`Ŏ2       �	��2�Xc�A�7*

loss�
c<��\�       �	QM3�Xc�A�7*

loss\5�;m���       �	��3�Xc�A�7*

loss�!<݈��       �	H�4�Xc�A�7*

loss�N[<��v       �	�C5�Xc�A�7*

loss�0�<�r��       �	!�5�Xc�A�7*

loss�U�;��W�       �	H�6�Xc�A�7*

loss��<��c       �	�17�Xc�A�7*

loss16_<��2       �	^�7�Xc�A�7*

lossh"5=�q��       �	��8�Xc�A�7*

lossd�i<���[       �	[:�Xc�A�7*

lossL1�;��D       �	��:�Xc�A�7*

loss�8�=�(C�       �	Ӈ;�Xc�A�7*

loss�;K�8       �	L<�Xc�A�7*

lossJw�;l��D       �	��<�Xc�A�7*

loss�+o<�uk
       �	�I=�Xc�A�8*

loss��I<�H�L       �	�=�Xc�A�8*

lossd�O;��!C       �	�t>�Xc�A�8*

loss��<}i��       �		?�Xc�A�8*

lossI�s:eD�       �	#�?�Xc�A�8*

loss�Q<ڣ�S       �	�8@�Xc�A�8*

loss��;"���       �	8�@�Xc�A�8*

loss�:���       �	�nA�Xc�A�8*

lossQW<��(       �	NB�Xc�A�8*

loss5"�<��s'       �	/�B�Xc�A�8*

loss��<;`��       �	�<C�Xc�A�8*

lossKX<[8       �	��C�Xc�A�8*

lossdj�<Rܕ�       �	��D�Xc�A�8*

loss��<Zs�-       �	~E�Xc�A�8*

loss���;���y       �	��E�Xc�A�8*

lossh&:'T�       �	��F�Xc�A�8*

loss�/m<{��       �	I*G�Xc�A�8*

loss;�P;��ln       �	��G�Xc�A�8*

lossc�<=�h�       �	�[H�Xc�A�8*

loss6�;�Kz�       �	'�H�Xc�A�8*

loss8/�<����       �	��I�Xc�A�8*

lossEV;9��
       �	BJ�Xc�A�8*

lossi1;��G�       �	"�J�Xc�A�8*

lossL-;���       �	˅K�Xc�A�8*

loss��!=XW��       �	�!L�Xc�A�8*

loss�[�;�-/�       �	/�L�Xc�A�8*

loss�G�<�� F       �	�XM�Xc�A�8*

lossQ�8=z.�2       �	�M�Xc�A�8*

lossD�e<�F.M       �	e�N�Xc�A�8*

loss���;o4       �	�jO�Xc�A�8*

loss�9`<�C�q       �	�P�Xc�A�8*

lossWU�9�U�       �	a�P�Xc�A�8*

lossd�;r���       �	vOQ�Xc�A�8*

loss乭;z�