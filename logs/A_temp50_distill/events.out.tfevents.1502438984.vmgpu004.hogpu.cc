       �K"	   �Yc�Abrain.Event:2%}�� �     �@��	�%�Yc�A"��
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
:@*
seed2���
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
:@@*
seed2ō,
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
seed2��]
�
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
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
:*
ellipsis_mask *

begin_mask 
Y
flatten_1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2��*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2
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
*
seed2��(
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2쾐*
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
:����������*
seed2߼h
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
 *  HB*
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
 *  �?*
_output_shapes
: *
dtype0
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
PlaceholderPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
L
div_2/yConst*
valueB
 *  HB*
_output_shapes
: *
dtype0
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*
T0*'
_output_shapes
:���������

c
!softmax_cross_entropy_loss_1/RankConst*
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
out_type0*
_output_shapes
:*
T0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
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
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
Index0*
T0*
_output_shapes
:
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
value	B :*
dtype0*
_output_shapes
: 
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
valueB:*
dtype0*
_output_shapes
:
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
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
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
value	B :*
dtype0*
_output_shapes
: 
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
.softmax_cross_entropy_loss_1/num_present/EqualEqual(softmax_cross_entropy_loss_1/ToFloat_1/x0softmax_cross_entropy_loss_1/num_present/Equal/y*
T0*
_output_shapes
: 
�
3softmax_cross_entropy_loss_1/num_present/zeros_like	ZerosLike(softmax_cross_entropy_loss_1/ToFloat_1/x*
_output_shapes
: *
T0
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
_output_shapes
: *
dtype0
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
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
T0*
_output_shapes
: 
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
T0*
_output_shapes
: 
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
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: *
T0
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: 
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
T0*
_output_shapes
: 
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
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
T0*
_output_shapes
: 
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
valueB *
_output_shapes
: *
dtype0
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
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
out_type0*
_output_shapes
:*
T0
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulMul4gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
3gradients/softmax_cross_entropy_loss_1/Mul_grad/SumSum3gradients/softmax_cross_entropy_loss_1/Mul_grad/mulEgradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*
T0*#
_output_shapes
:���������
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
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: *
T0
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
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
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
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
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
*gradients/div_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_2_grad/Shapegradients/div_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_1Sum4gradients/sequential_1/dropout_2/cond/mul_grad/mul_1Fgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
 *    *
_output_shapes
: *
dtype0
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
valueB@@*    *&
_output_shapes
:@@*
dtype0
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
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
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
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"f:5��,     ?/��	�d(�Yc�AJ��
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
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
v
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2ō,*
dtype0*
T0*
seed���)
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
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
[
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
�
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
s
conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
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
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
T0
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
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2��]
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
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
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
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
:*
shrink_axis_mask 
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
dense_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2��*
dtype0*
T0*
seed���)
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
	container *
dtype0*
shared_name *
shape:���
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
VariableV2*
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
	container 
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
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
:����������*
data_formatNHWC*
T0
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
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
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
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
seed2
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
dropout_2/cond/dropout/mulMuldropout_2/cond/dropout/divdropout_2/cond/dropout/Floor*(
_output_shapes
:����������*
T0
�
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   
   
_
dense_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U�
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U>
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2��(*
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
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�

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
VariableV2*
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
_output_shapes
:
*
_class
loc:@dense_2/bias*
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*
strides
*
data_formatNHWC*/
_output_shapes
:���������@*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
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
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:���������@
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
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
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
_output_shapes
:*
T0

�
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
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
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2쾐
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
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu*
T0
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
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
,sequential_1/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
v
,sequential_1/flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
T0*
Index0
f
sequential_1/flatten_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
sequential_1/flatten_1/ProdProd$sequential_1/flatten_1/strided_slicesequential_1/flatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
sequential_1/flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
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
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
data_formatNHWC*
T0
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
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
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
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
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
seed2߼h
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
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
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
num_inst/initial_valueConst*
dtype0*
_output_shapes
: *
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
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_inst
a
num_inst/readIdentitynum_inst*
T0*
_output_shapes
: *
_class
loc:@num_inst
^
num_correct/initial_valueConst*
dtype0*
_output_shapes
: *
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
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
use_locking(
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
AssignAdd_1	AssignAddnum_correctSum*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@num_correct
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
�
AssignAssignnum_instConst_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_inst
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
add/yConst*
dtype0*
_output_shapes
: *
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
 *  HB
i
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*'
_output_shapes
:���������
*
T0
a
softmax_cross_entropy_loss/RankConst*
_output_shapes
: *
dtype0*
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
T0*
_output_shapes
:*
out_type0
c
!softmax_cross_entropy_loss/Rank_1Const*
dtype0*
_output_shapes
: *
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
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
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
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
c
!softmax_cross_entropy_loss/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
_output_shapes
: *
dtype0*
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
'softmax_cross_entropy_loss/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
_output_shapes
:*
dtype0*
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
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
�
 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
T0*
_output_shapes
:*

axis 
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
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
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
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
_output_shapes
:*
out_type0*
T0
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
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
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
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
PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

L
div_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
i
div_2RealDivsequential_1/dense_2/BiasAdddiv_2/y*
T0*'
_output_shapes
:���������

c
!softmax_cross_entropy_loss_1/RankConst*
dtype0*
_output_shapes
: *
value	B :
g
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
T0*
_output_shapes
:*
out_type0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
_output_shapes
:*
out_type0*
T0
d
"softmax_cross_entropy_loss_1/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
"softmax_cross_entropy_loss_1/SliceSlice$softmax_cross_entropy_loss_1/Shape_1(softmax_cross_entropy_loss_1/Slice/begin'softmax_cross_entropy_loss_1/Slice/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss_1/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
j
(softmax_cross_entropy_loss_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*0
_output_shapes
:������������������*
Tshape0*
T0
e
#softmax_cross_entropy_loss_1/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
o
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
_output_shapes
:*
out_type0*
T0
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
"softmax_cross_entropy_loss_1/Sub_1Sub#softmax_cross_entropy_loss_1/Rank_2$softmax_cross_entropy_loss_1/Sub_1/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
_output_shapes
:*
N*

axis *
T0
s
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
_output_shapes
:*
N*

axis *
T0
�
$softmax_cross_entropy_loss_1/Slice_2Slice"softmax_cross_entropy_loss_1/Shape*softmax_cross_entropy_loss_1/Slice_2/begin)softmax_cross_entropy_loss_1/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
&softmax_cross_entropy_loss_1/Reshape_2Reshape%softmax_cross_entropy_loss_1/xentropy$softmax_cross_entropy_loss_1/Slice_2*#
_output_shapes
:���������*
Tshape0*
T0
~
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B : 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
_output_shapes
:*
out_type0*
T0

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
2softmax_cross_entropy_loss_1/num_present/ones_likeFill8softmax_cross_entropy_loss_1/num_present/ones_like/Shape8softmax_cross_entropy_loss_1/num_present/ones_like/Const*
_output_shapes
: *
T0
�
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss_1/Const_1ConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
"softmax_cross_entropy_loss_1/Sum_1Sum softmax_cross_entropy_loss_1/Sum$softmax_cross_entropy_loss_1/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
"softmax_cross_entropy_loss_1/EqualEqual(softmax_cross_entropy_loss_1/num_present$softmax_cross_entropy_loss_1/Equal/y*
_output_shapes
: *
T0
�
,softmax_cross_entropy_loss_1/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
z
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
Egradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/div_grad/Shape7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivRealDivJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency#softmax_cross_entropy_loss_1/Select*
T0*
_output_shapes
: 
�
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
_output_shapes
: *
Tshape0

3gradients/softmax_cross_entropy_loss_1/div_grad/NegNeg"softmax_cross_entropy_loss_1/Sum_1*
_output_shapes
: *
T0
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
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss_1/div_grad/mulGgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
T0
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
T0
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
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select*
T0
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
_output_shapes
: *N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
_output_shapes
: *
dtype0*
valueB 
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
_output_shapes
:*
out_type0*
T0
�
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
T0
�
<gradients/softmax_cross_entropy_loss_1/num_present_grad/TileTile?gradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape=gradients/softmax_cross_entropy_loss_1/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
_output_shapes
:*
out_type0*
T0
z
7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Mul&softmax_cross_entropy_loss_1/Reshape_24gradients/softmax_cross_entropy_loss_1/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_1Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/mul_1Ggradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
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
T0*
_output_shapes
: *
Tshape0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1Mul/softmax_cross_entropy_loss_1/num_present/Select<gradients/softmax_cross_entropy_loss_1/num_present_grad/Tile*#
_output_shapes
:���������*
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
_output_shapes
: *d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
T0
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*#
_output_shapes
:���������*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*
T0
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
_output_shapes
:*
out_type0*
T0
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
�
gradients/zeros_like	ZerosLike'softmax_cross_entropy_loss_1/xentropy:1*0
_output_shapes
:������������������*
T0
�
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*'
_output_shapes
:���������*
T0*

Tdim0
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
_output_shapes
:*
out_type0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
_
gradients/div_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
o
gradients/div_2_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*
T0*'
_output_shapes
:���������

�
gradients/div_2_grad/Sum_1Sumgradients/div_2_grad/mul,gradients/div_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
m
%gradients/div_2_grad/tuple/group_depsNoOp^gradients/div_2_grad/Reshape^gradients/div_2_grad/Reshape_1
�
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape*
T0
�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*
_output_shapes
: *1
_class'
%#loc:@gradients/div_2_grad/Reshape_1*
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_2_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape*
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
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
gradients/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
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
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*(
_output_shapes
:����������*
T0
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros*
N*
T0**
_output_shapes
:����������: 
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
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
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
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
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
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*
T0
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
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1*
N*
T0**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*(
_output_shapes
:����������*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
T0
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*
T0*(
_output_shapes
:����������
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:�
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
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
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
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
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
:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/SumSum:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mulLgradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
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
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
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
2gradients/sequential_1/dropout_1/cond/mul_grad/SumSum2gradients/sequential_1/dropout_1/cond/mul_grad/mulDgradients/sequential_1/dropout_1/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_1/cond/mul_grad/Sum_16gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
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
gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*/
_output_shapes
:���������@
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
N*
T0
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
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
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
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
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
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
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
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
zerosConst*
dtype0*&
_output_shapes
:@*%
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
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
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
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
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
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
	container *
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
zeros_6Const*
dtype0*
_output_shapes
:@*
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
b
zeros_8Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
�
dense_1/kernel/Adam
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
�
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
W
zeros_10Const*
_output_shapes	
:�*
dtype0*
valueB�*    
�
dense_1/bias/Adam
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
_output_shapes	
:�*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
use_locking(
|
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
W
zeros_11Const*
dtype0*
_output_shapes	
:�*
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
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
_output_shapes
:	�
*
validate_shape(*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
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
dense_2/bias/Adam/AssignAssigndense_2/bias/Adamzeros_14*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
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
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1zeros_15*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
use_locking(
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
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
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0*
use_locking( 
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
	loss/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0"�
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0sx�m       ��-	��T�Yc�A*

lossC^@�1�       ��-	�VU�Yc�A*

lossZ@؉G       ��-	ۉV�Yc�A*

loss�8@��H       ��-	^�W�Yc�A*

lossg@�a��       ��-	l\X�Yc�A*

loss��@�jۘ       ��-	�Y�Yc�A*

loss��
@�N�       ��-	�Y�Yc�A*

loss�R@t��       ��-	�MZ�Yc�A*

lossF^@:0[8       ��-	-�Z�Yc�A	*

loss$Y�?�d�s       ��-	D�[�Yc�A
*

loss���?�!��       ��-	�<\�Yc�A*

lossMY�?�	�       ��-	7�\�Yc�A*

loss���?�;�B       ��-	p{]�Yc�A*

loss���?S���       ��-	�^�Yc�A*

lossf��?���       ��-	��^�Yc�A*

lossn�?��       ��-	�Q_�Yc�A*

lossm�?�u��       ��-	)�_�Yc�A*

loss�.�?��Ƭ       ��-	��`�Yc�A*

loss��?Q��       ��-	�a�Yc�A*

lossq��?{T��       ��-	Ҭa�Yc�A*

loss��v?�B       ��-	[Ab�Yc�A*

loss��?7��       ��-	%\c�Yc�A*

loss��k?���       ��-	@jd�Yc�A*

loss� �?ۑ�       ��-	�e�Yc�A*

loss�|?e��9       ��-	��e�Yc�A*

loss��?
/�       ��-	�?f�Yc�A*

loss�r?=���       ��-	��f�Yc�A*

loss��n?sdp       ��-	Ng�Yc�A*

loss
5�?�sT       ��-	Ch�Yc�A*

loss�in?"p;�       ��-	�h�Yc�A*

lossz�f?���       ��-	F\i�Yc�A*

loss�W?g�d       ��-	j�i�Yc�A *

loss�n?l���       ��-	��j�Yc�A!*

loss$�o?�z��       ��-	�*k�Yc�A"*

loss��R?m�؉       ��-	e�k�Yc�A#*

loss{1[?&�       ��-	�`l�Yc�A$*

loss�3 ?�Kd_       ��-	Vm�Yc�A%*

loss6�k?�,��       ��-	�n�Yc�A&*

loss��o?H2�       ��-	��n�Yc�A'*

losstc?m��       ��-	�o�Yc�A(*

loss��'?�5l�       ��-	*p�Yc�A)*

lossi�?|4U       ��-	+�p�Yc�A**

loss.9?�A8�       ��-	�Yq�Yc�A+*

lossR0=?}6,T       ��-	=�q�Yc�A,*

lossw�8?�ɰ�       ��-	v�r�Yc�A-*

lossJ�?Sؠ-       ��-	h@s�Yc�A.*

loss?�f       ��-	<�s�Yc�A/*

lossf\?S(�J       ��-	tt�Yc�A0*

lossO?�`�+       ��-	�u�Yc�A1*

loss��><�       ��-	��u�Yc�A2*

loss 0�>��<~       ��-	AHv�Yc�A3*

loss��?�ֱ       ��-	�v�Yc�A4*

loss9?Q�!       ��-	�pw�Yc�A5*

loss��V?n��       ��-	�x�Yc�A6*

loss/��>5o       ��-	�x�Yc�A7*

loss���>C�x�       ��-	eVy�Yc�A8*

loss�Ӹ>+2��       ��-	�z�Yc�A9*

loss���>�
!F       ��-	b�z�Yc�A:*

loss�E?��Z:       ��-	�8{�Yc�A;*

lossM��>j
�       ��-	��{�Yc�A<*

losst�?&�       ��-	rk|�Yc�A=*

lossd��>��       ��-	%}�Yc�A>*

loss�%�>��v^       ��-	[�}�Yc�A?*

loss�M?=1�       ��-	,~�Yc�A@*

loss3� ?�G�V       ��-	��~�Yc�AA*

loss�%�>C��       ��-	\�Yc�AB*

lossjV(?�J�       ��-	���Yc�AC*

loss��?/6�       ��-	����Yc�AD*

loss}�5?V=2�       ��-	:!��Yc�AE*

loss��?.��       ��-	����Yc�AF*

loss_�?ނ�       ��-	�U��Yc�AG*

loss%X?m!r       ��-	��Yc�AH*

loss���>���       ��-	���Yc�AI*

loss�?
�       ��-	t&��Yc�AJ*

loss�'?�΍       ��-	����Yc�AK*

loss��?Ԁ&       ��-	uY��Yc�AL*

loss�Yr?,��       ��-	����Yc�AM*

loss�&I? ��       ��-	���Yc�AN*

loss4�?X�P
       ��-	\��Yc�AO*

loss�ػ>�W�<       ��-	����Yc�AP*

loss�!?X�].       ��-	�R��Yc�AQ*

loss8K?M���       ��-	�ꈒYc�AR*

loss#
?���       ��-	(���Yc�AS*

loss2�>��R       ��-	~��Yc�AT*

loss��	?1��       ��-	^���Yc�AU*

loss[J�>���       ��-	6Y��Yc�AV*

loss���>���       ��-	���Yc�AW*

loss=m?�D�       ��-	���Yc�AX*

loss/'�>U@��       ��-	�M��Yc�AY*

loss�ؿ>�G�       ��-	pꍒYc�AZ*

loss��?�י�       ��-	���Yc�A[*

lossI�?���       ��-	�7��Yc�A\*

loss�I?~���       ��-	�؏�Yc�A]*

loss�h?}��       ��-	�t��Yc�A^*

loss�O+?���?       ��-	j��Yc�A_*

loss�<�>�Ô       ��-	_���Yc�A`*

lossJ{?��j�       ��-	
J��Yc�Aa*

lossX�C?�|�F       ��-	�⒒Yc�Ab*

loss,�o?�#.�       ��-	_z��Yc�Ac*

lossq��>���       ��-	���Yc�Ad*

loss�D�>���N       ��-	⫔�Yc�Ae*

loss|u�>��       ��-	�A��Yc�Af*

loss��?���b       ��-	�ڕ�Yc�Ag*

loss���>��F       ��-	�x��Yc�Ah*

loss	��>:x�~       ��-	���Yc�Ai*

loss�Z�>��9�       ��-	e���Yc�Aj*

lossJ2�>G�P       ��-	hA��Yc�Ak*

loss�ʫ>7Ț�       ��-	�ژ�Yc�Al*

loss1�?ŵ[d       ��-	ip��Yc�Am*

lossW�?���       ��-	���Yc�An*

lossRO?%� @       ��-	^���Yc�Ao*

loss��?B-TO       ��-	I��Yc�Ap*

loss$Z�>�k�
       ��-	�ߛ�Yc�Aq*

loss�i�>����       ��-	����Yc�Ar*

lossñ�>�p       ��-	���Yc�As*

lossdų>,N�h       ��-	����Yc�At*

loss���>.F�       ��-	CS��Yc�Au*

lossqC�>�JR�       ��-	l螒Yc�Av*

loss���>N,��       ��-	����Yc�Aw*

loss6Z�>�[�       ��-	u��Yc�Ax*

loss�̴>�Af�       ��-	#���Yc�Ay*

loss���>�$       ��-	�W��Yc�Az*

loss:�>��&       ��-	J롒Yc�A{*

losso~�>���K       ��-	s���Yc�A|*

loss��D>�O       ��-	q��Yc�A}*

loss���>R���       ��-	����Yc�A~*

loss/�>4��;       ��-	�M��Yc�A*

lossr+�>�{��       �	�餒Yc�A�*

loss��>���>       �	{���Yc�A�*

lossl��>�+�
       �	���Yc�A�*

lossMH�>v�(�       �	5���Yc�A�*

loss}��>֊�       �	F[��Yc�A�*

loss��{>��       �	����Yc�A�*

loss��Q>\�e       �	\���Yc�A�*

loss��>�j�
       �	�%��Yc�A�*

loss��>���       �	����Yc�A�*

loss�ZW>�G �       �	Y��Yc�A�*

loss
�|>��f	       �	��Yc�A�*

loss�{s>��3d       �	S���Yc�A�*

loss�9a>2��       �	Z,��Yc�A�*

lossI�9>��Eh       �	mʬ�Yc�A�*

loss_��>-�       �	�e��Yc�A�*

loss4��>��Ί       �	���Yc�A�*

loss��>�(/       �	t���Yc�A�*

lossOj�>f�/       �	�8��Yc�A�*

loss�gz>�?/�       �	�ܯ�Yc�A�*

lossW�>���       �	Ox��Yc�A�*

loss|�B>��n�       �	��Yc�A�*

loss��J>╹�       �	����Yc�A�*

lossZ�x>[g�       �	b��Yc�A�*

loss��>��       �	C��Yc�A�*

loss���>���|       �	����Yc�A�*

loss���>��*�       �	�>��Yc�A�*

lossE��>�3�G       �	�ݴ�Yc�A�*

loss�IP>VoI�       �	�t��Yc�A�*

loss�4>TkH�       �	���Yc�A�*

lossKݟ>y^x       �	m���Yc�A�*

lossD�}>/�       �	�N��Yc�A�*

loss�P�>��Q       �	}跒Yc�A�*

loss���>l�)       �	v���Yc�A�*

lossXn�>hqB       �	-%��Yc�A�*

loss8��>��       �	
���Yc�A�*

loss?�>=�_'       �		R��Yc�A�*

lossR�d>0Mq*       �	����Yc�A�*

loss�>U�       �	4���Yc�A�*

loss��>4�6�       �	�>��Yc�A�*

loss�s>A��       �	�ἒYc�A�*

lossfu>���       �	f���Yc�A�*

loss�dH>�&��       �	�)��Yc�A�*

lossM�!>���       �	�Ǿ�Yc�A�*

loss�7T>^��       �	g��Yc�A�*

lossn��=(Y�g       �	� ��Yc�A�*

loss�?�>�o�8       �	���Yc�A�*

lossC.�>�ܨ�       �	�;��Yc�A�*

loss�#&>���       �	-���Yc�A�*

loss�/�>&�"�       �	mYc�A�*

loss@��>q�B+       �	~ÒYc�A�*

loss���>O���       �	Z�ÒYc�A�*

loss�^�>�Gh       �	;7ĒYc�A�*

loss��}>��v0       �	�ĒYc�A�*

loss�c�=��}       �	�lŒYc�A�*

loss!Ѽ>��l       �	ƒYc�A�*

lossȤ>��TS       �	ʩƒYc�A�*

lossn�>E�z       �	�JǒYc�A�*

loss�s�=Epx�       �	��ǒYc�A�*

loss}O�>��n�       �	ݖȒYc�A�*

loss)#�>�@�-       �	7ɒYc�A�*

loss�P�>�e'b       �	��ɒYc�A�*

loss*�]>���       �	yʒYc�A�*

lossJ_�>$J�       �	I˒Yc�A�*

loss�݋>l���       �	ު˒Yc�A�*

loss�c;>!��       �	�Q̒Yc�A�*

loss%ʍ>N��6       �	l�̒Yc�A�*

loss�>�Z*       �	��͒Yc�A�*

lossq�?>�/ht       �	)ΒYc�A�*

loss��>wX       �	2�ΒYc�A�*

loss�Ր>=�7�       �	W^ϒYc�A�*

loss15�>h^Z       �	��ϒYc�A�*

lossq��>�Q~       �	��ВYc�A�*

loss_1u>lN�       �	EђYc�A�*

lossn�V>��       �	)"ҒYc�A�*

loss��>��:       �	�ZӒYc�A�*

lossm?>�6 �       �	��ӒYc�A�*

lossnƄ>ΨF       �	��ԒYc�A�*

losso�O>Vu{       �	�OՒYc�A�*

loss)�>�bQ       �	�6֒Yc�A�*

loss g>�J�       �	9גYc�A�*

lossq��=^���       �	�<ؒYc�A�*

lossL>�|�'       �	��ؒYc�A�*

lossBZ�>}8�]       �	hْYc�A�*

lossjV>�H�       �	��ْYc�A�*

loss�fA>9���       �	��ڒYc�A�*

loss0�>�T�[       �	�:ےYc�A�*

loss�.]>1��       �	��ےYc�A�*

lossWͥ>�J�       �	�cܒYc�A�*

lossF?�>���k       �	
�ܒYc�A�*

loss0�>���       �	ΌݒYc�A�*

loss�B�>'v��       �	Z)ޒYc�A�*

loss4�>�L��       �	��ޒYc�A�*

loss1�>�&1�       �	�VߒYc�A�*

loss���=̍"       �	�ߒYc�A�*

loss�w1>^�M       �	#���Yc�A�*

loss�r�>S�n       �	��Yc�A�*

loss���>���       �	^��Yc�A�*

loss�@>ɻI�       �	�T�Yc�A�*

loss�R>��       �	��Yc�A�*

loss4�>p�KV       �	���Yc�A�*

lossHo�>��q       �	�$�Yc�A�*

lossW\>�>�<       �	���Yc�A�*

loss��7>7"Z       �	�}�Yc�A�*

loss�A6>R?v'       �	/�Yc�A�*

loss1�=}ߘu       �	��Yc�A�*

lossFS�>��e       �	�A�Yc�A�*

lossv#>jMĖ       �	#��Yc�A�*

loss�?�>j��       �	��Yc�A�*

loss��\>�*�       �	)�Yc�A�*

loss�B>4�ك       �	a��Yc�A�*

loss\^�>�I       �	^�Yc�A�*

loss��_>ɤ��       �	���Yc�A�*

loss��1>v\=       �	9��Yc�A�*

loss}�[>��s       �	�2�Yc�A�*

loss�_>�y+�       �	S��Yc�A�*

loss_��>#E       �	d�Yc�A�*

lossĴ@>�]�       �	D��Yc�A�*

lossaO�>\��       �	K��Yc�A�*

loss x>7���       �	,*�Yc�A�*

lossJ�%>A}Q       �	���Yc�A�*

loss8g6>�i^       �	O�Yc�A�*

loss���>��?       �	���Yc�A�*

lossݰ>�Yt)       �	��Yc�A�*

lossnՇ>       �	k(�Yc�A�*

loss��k>�:y�       �	S��Yc�A�*

loss`ϧ>+`͒       �	ka��Yc�A�*

loss6 >0v�       �	{���Yc�A�*

lossm{>�.�^       �	���Yc�A�*

loss(�X>��$
       �	� ��Yc�A�*

lossgM�>N��3       �	ٴ��Yc�A�*

loss�F�>�A`       �	K��Yc�A�*

loss�[>m���       �	&���Yc�A�*

loss�ޥ>����       �	x}��Yc�A�*

loss��>�>�       �	��Yc�A�*

loss$�=Q�|       �	���Yc�A�*

loss)�>��       �	<��Yc�A�*

lossQ�>�4i�       �	����Yc�A�*

loss�g�>Q|�       �	
i��Yc�A�*

loss
nl>`W�e       �	���Yc�A�*

loss��l>B��       �	Y���Yc�A�*

loss
�x>Q��s       �	K��Yc�A�*

loss��>��r       �	$���Yc�A�*

loss;�R>��aE       �	�� �Yc�A�*

lossI_>�ǲ�       �	3�Yc�A�*

lossaB�>{,       �	Y��Yc�A�*

loss1�v>�y��       �	��Yc�A�*

loss�p@>N�ҷ       �	�4�Yc�A�*

loss� �>�z��       �	���Yc�A�*

loss<�> �	       �	�t�Yc�A�*

loss��>��Q�       �	C �Yc�A�*

loss~�>�v��       �	���Yc�A�*

loss�K>���L       �	pa�Yc�A�*

loss�׊>�       �	��Yc�A�*

loss���=��q       �	���Yc�A�*

loss�]>��ڒ       �	��Yc�A�*

loss�X>�k{       �	]1	�Yc�A�*

lossH?>�02�       �	^�	�Yc�A�*

loss6,m>�x��       �	ʈ
�Yc�A�*

loss�>�X�       �	�7�Yc�A�*

loss!>ѻ�8       �	��Yc�A�*

loss,��=�`�       �	���Yc�A�*

loss�bE>���       �	�:�Yc�A�*

lossiE:=�|]7       �	}��Yc�A�*

lossÂ�=���       �	��Yc�A�*

lossL->�1N�       �	AE�Yc�A�*

loss�P�=e�F       �	y��Yc�A�*

loss�p>�XUa       �	%��Yc�A�*

loss��>>j��       �	�5�Yc�A�*

lossoB�>(A��       �	���Yc�A�*

loss�>�>b�iY       �	��Yc�A�*

lossvC6>�{�       �	�+�Yc�A�*

loss �&>���0       �	~��Yc�A�*

loss��>�@�       �	ö�Yc�A�*

loss��~>��s       �	x�Yc�A�*

loss�6�=��:       �	�t�Yc�A�*

loss�y>qF       �	$`�Yc�A�*

loss�!�>.u       �	�'�Yc�A�*

loss�;>8��       �	���Yc�A�*

lossc�p>�6�       �	B_�Yc�A�*

loss4�=�0�S       �	e�Yc�A�*

loss�>��U       �	���Yc�A�*

loss�a�=g;S�       �	a�Yc�A�*

loss�5j>N�       �	���Yc�A�*

loss��=:7��       �	J��Yc�A�*

loss�
s>�@�       �	6�Yc�A�*

loss
G>:���       �	���Yc�A�*

loss�Z�=��       �	�i�Yc�A�*

loss�h>��1       �	?�Yc�A�*

lossm+�>��D       �	B��Yc�A�*

loss��>�~       �	+1 �Yc�A�*

loss��=p�h.       �	�� �Yc�A�*

losso�Y>�7X�       �	G:"�Yc�A�*

loss���=�~       �	��"�Yc�A�*

loss_�	>W�       �	�h#�Yc�A�*

loss��>���       �	\$�Yc�A�*

loss<ki>���       �	��%�Yc�A�*

loss��{>��w       �	!x&�Yc�A�*

loss�C�>V�       �	�'�Yc�A�*

lossCD�>��       �	k�'�Yc�A�*

loss�.>q2<�       �	�U(�Yc�A�*

loss.I>�ҹ       �	I�(�Yc�A�*

lossĆc>����       �	6�)�Yc�A�*

loss�)>�^�       �	0.*�Yc�A�*

loss��9>��r&       �	N�*�Yc�A�*

loss\w>���       �	�p+�Yc�A�*

loss��
>z^֪       �	b,�Yc�A�*

loss	)�=ҵ;       �	�,�Yc�A�*

lossj>��       �	rS-�Yc�A�*

loss���=��       �	V.�Yc�A�*

loss���>����       �	�/�Yc�A�*

loss�9x>��6�       �	�p0�Yc�A�*

lossʣ�>ʚv�       �	1�Yc�A�*

loss3A4>�K�       �	�1�Yc�A�*

loss�qi>�KI       �	�2�Yc�A�*

loss�N>AP�z       �	��3�Yc�A�*

losss�r>�4       �	ٓ4�Yc�A�*

lossO:>�y�       �	U/5�Yc�A�*

loss��?zr�N       �	�5�Yc�A�*

loss��k>))L+       �	�c6�Yc�A�*

loss�A
>�t��       �	�7�Yc�A�*

loss�:;>K�       �	��7�Yc�A�*

loss��O>��a�       �	�E8�Yc�A�*

loss��W>���       �	�8�Yc�A�*

loss,,>&�u�       �	�v9�Yc�A�*

loss#�'>�`�       �	�:�Yc�A�*

loss{۱=9��.       �	o�:�Yc�A�*

loss��>�Л       �	mY;�Yc�A�*

loss��*>��W�       �	��;�Yc�A�*

loss1X>*��m       �	E�<�Yc�A�*

loss�|�=2>��       �	�A=�Yc�A�*

loss�C>�^g       �	C�=�Yc�A�*

loss���=l���       �	�{>�Yc�A�*

loss�>@}co       �	�?�Yc�A�*

lossJY>��Z�       �	��?�Yc�A�*

loss�">��Z       �	�h@�Yc�A�*

loss���=�֞       �	�A�Yc�A�*

loss/ �>l���       �	פA�Yc�A�*

loss���>��4       �	@B�Yc�A�*

lossExN>��]       �	K�B�Yc�A�*

lossb?�>��       �	ǄC�Yc�A�*

loss�>���       �	|&D�Yc�A�*

loss�B>L���       �	�HE�Yc�A�*

loss�5�=�6�:       �	�E�Yc�A�*

loss�+K>�$?`       �	{�F�Yc�A�*

loss�P>`�ѹ       �	G�Yc�A�*

losslv>P�f�       �	�G�Yc�A�*

loss*>�ڋ       �	*TH�Yc�A�*

loss�@<>s) �       �	��H�Yc�A�*

loss��>�R       �	��I�Yc�A�*

loss��>Dг       �	N)J�Yc�A�*

loss
\Z>�}#�       �	!�J�Yc�A�*

loss��8>��ױ       �	�K�Yc�A�*

loss�>9���       �	��L�Yc�A�*

loss���>�e       �	XM�Yc�A�*

loss<R>ռ�`       �	]�M�Yc�A�*

loss�EJ>@���       �	-[N�Yc�A�*

loss.-0>դe�       �	��N�Yc�A�*

loss&�=�?z       �	ڒO�Yc�A�*

loss��=Y0�p       �	E*P�Yc�A�*

lossʪ2>~n       �	A*Q�Yc�A�*

loss31>߅8       �	��Q�Yc�A�*

lossS�=�⫒       �	�R�Yc�A�*

lossdqK>�SS       �	FS�Yc�A�*

lossZ>��n       �	��S�Yc�A�*

loss��>T��o       �	�|T�Yc�A�*

lossh'l>���X       �	��U�Yc�A�*

loss�dA>���       �	��V�Yc�A�*

loss��|>�^��       �	�EW�Yc�A�*

loss:O>s ��       �	�X�Yc�A�*

lossv��=���       �	�Y�Yc�A�*

loss(��=Ͼ*       �	�Y�Yc�A�*

loss΄�=:��       �	�UZ�Yc�A�*

loss�j/>=�J       �	g�Z�Yc�A�*

loss�>ǋ��       �	*�[�Yc�A�*

loss*7T>F��       �	d$\�Yc�A�*

loss��)>��       �	��\�Yc�A�*

loss��=a5�       �	�W]�Yc�A�*

loss�l=w�v       �	#�]�Yc�A�*

loss�T>��)�       �	��^�Yc�A�*

loss�b}> ��       �	�#_�Yc�A�*

lossʭ`>r��y       �	N�_�Yc�A�*

loss���=W�[       �	��`�Yc�A�*

loss�X>��e       �	�5a�Yc�A�*

loss��>���       �	��a�Yc�A�*

loss�j~>x���       �	kb�Yc�A�*

loss�z>��       �	c�Yc�A�*

lossM�>�G�6       �	md�Yc�A�*

lossD)9>�)S       �	e�Yc�A�*

lossS��>[	�       �	��e�Yc�A�*

loss��t=�z(       �	�Af�Yc�A�*

lossS��=cmG�       �	��f�Yc�A�*

lossS�p>2n��       �	؂g�Yc�A�*

loss{!>,H_       �	�h�Yc�A�*

loss�>��6�       �	ȷh�Yc�A�*

loss܁	>$���       �	�Ti�Yc�A�*

lossCr.>���       �	�i�Yc�A�*

loss��>�> �       �	��j�Yc�A�*

lossMBl>s���       �	�k�Yc�A�*

loss��>���       �	��k�Yc�A�*

loss�"">�'k       �	�Sl�Yc�A�*

loss�^d> `4:       �	9�l�Yc�A�*

lossr�>�$       �	!�m�Yc�A�*

loss�>��n�       �	�n�Yc�A�*

loss��Z>kY��       �	�to�Yc�A�*

lossvM�=y�%       �	Wp�Yc�A�*

loss�U�=���X       �	�p�Yc�A�*

loss"�=X�]       �	6�q�Yc�A�*

lossn�3>�+�       �	��r�Yc�A�*

loss�N�=��q�       �	�ls�Yc�A�*

loss}��=%r&�       �	�t�Yc�A�*

loss��>����       �	��t�Yc�A�*

lossOD�=��LB       �	`�u�Yc�A�*

loss��=>o��       �	M/v�Yc�A�*

loss�m>e�	�       �	:�v�Yc�A�*

lossΏ�>�I �       �	N{w�Yc�A�*

loss-O	>K�'�       �	x�Yc�A�*

loss�N:>#K~       �	�x�Yc�A�*

loss�a�=��`       �	NGy�Yc�A�*

loss2�n>�-��       �	��y�Yc�A�*

loss���>�� 0       �	qsz�Yc�A�*

loss@3q>�9�       �		{�Yc�A�*

loss��=��       �	˿{�Yc�A�*

lossM�>o�'�       �	"U|�Yc�A�*

lossD>�k��       �	u�|�Yc�A�*

loss�R�=�N�       �	}�Yc�A�*

lossTX�=��4�       �	4~�Yc�A�*

loss��y>ͥ�i       �	��~�Yc�A�*

loss�5�>�tL;       �	�G�Yc�A�*

lossۆ>���'       �	H��Yc�A�*

lossJP>j�B       �	�v��Yc�A�*

lossK�#>g��&       �	���Yc�A�*

loss��P>��P       �	⬁�Yc�A�*

loss�W>���G       �	 C��Yc�A�*

loss�đ= ���       �	����Yc�A�*

loss���=�l��       �	����Yc�A�*

loss��>�0e�       �	�:��Yc�A�*

loss@>>��y       �	-Є�Yc�A�*

loss���=q��[       �	!u��Yc�A�*

lossNof=$_Nl       �	���Yc�A�*

loss&>\��y       �	���Yc�A�*

lossO}�=ŋ9�       �	�N��Yc�A�*

lossʩ:>~_ٯ       �	�燓Yc�A�*

loss&�j=�K �       �	2���Yc�A�*

loss��V>�A       �	�)��Yc�A�*

loss�d(>���       �	։�Yc�A�*

lossv(P>c��Q       �	�r��Yc�A�*

loss,v)>�i�       �		��Yc�A�*

lossOp>T�U       �	����Yc�A�*

loss1Ճ=��       �	|F��Yc�A�*

loss?v=�
2       �	����Yc�A�*

loss;�=�]V�       �	{���Yc�A�*

loss.L>\�l       �	+��Yc�A�*

loss��I>�m~�       �	�ǎ�Yc�A�*

loss J�>��ɪ       �	�i��Yc�A�*

lossh�>�C�       �	���Yc�A�*

loss�>}D�       �	|��Yc�A�*

loss��=��j�       �	����Yc�A�*

loss?� >�-       �	�O��Yc�A�*

loss��=]��b       �	q撓Yc�A�*

lossM�>&)v=       �	ҍ��Yc�A�*

loss�&>��"�       �	�B��Yc�A�*

loss���=J���       �	����Yc�A�*

lossMqq>w�D       �	����Yc�A�*

lossd[&>:�Y�       �	�<��Yc�A�*

loss��)>��g       �	R�Yc�A�*

loss+�>:\       �	,���Yc�A�*

loss�
a=>#?�       �	<��Yc�A�*

loss0">{�G       �	�Ԙ�Yc�A�*

lossLi=ac�\       �	Ln��Yc�A�*

loss��->�+�       �	:;��Yc�A�*

loss �=�{       �	8ښ�Yc�A�*

loss��=��       �	(���Yc�A�*

loss���=���U       �	~6��Yc�A�*

loss��'>�z-�       �	aߜ�Yc�A�*

loss�6=�YG�       �	�z��Yc�A�*

loss$�="       �	E��Yc�A�*

loss�_>On)�       �	:���Yc�A�*

lossc$>P���       �	XS��Yc�A�*

loss���=�J       �	����Yc�A�*

lossK�	>�wy�       �	Ĕ��Yc�A�*

loss�n�=$���       �	A(��Yc�A�*

loss_fH>a,?s       �	
���Yc�A�*

loss`�=��5b       �	,�Yc�A�*

loss@��=�       �	k���Yc�A�*

loss���=p]��       �	�@��Yc�A�*

loss{��=��s�       �	W褓Yc�A�*

lossR�>��o�       �	�Yc�A�*

lossC��=:%��       �	�1��Yc�A�*

loss}�d>/.�       �	�զ�Yc�A�*

loss(�=��(p       �	�y��Yc�A�*

lossI�=���(       �	W#��Yc�A�*

loss��*>qL�       �	�ɨ�Yc�A�*

loss3t>�t�       �	�g��Yc�A�*

loss�>=甔5       �	��Yc�A�*

loss��!>+���       �	"���Yc�A�*

loss��h=!���       �	�N��Yc�A�*

lossa�=��L       �	U���Yc�A�*

lossX��=��v\       �	T���Yc�A�*

loss/�=�4K�       �	�J��Yc�A�*

loss(	>�j�       �	k���Yc�A�*

lossx�#> �V       �	ѕ��Yc�A�*

lossO_z>�S�       �	E/��Yc�A�*

loss�Pw=A92N       �	��Yc�A�*

loss��=�r�y       �	V���Yc�A�*

loss%>_�}�       �	�9��Yc�A�*

loss*��=9Ui       �	ٱ�Yc�A�*

loss2�-=$:{�       �	�{��Yc�A�*

loss��=��	�       �	+N��Yc�A�*

loss٢=�h�       �	B볓Yc�A�*

loss�<����       �	D���Yc�A�*

lossr�p=�w'�       �	�#��Yc�A�*

lossJ�=��,�       �	0��Yc�A�*

lossi�+>ɻx       �	-���Yc�A�*

loss��<��T`       �	�J��Yc�A�*

loss_�\<%M�       �	!䷓Yc�A�*

loss��f<n�\8       �	����Yc�A�*

loss�"A>2��       �	L��Yc�A�*

loss��V>f�B*       �	�⹓Yc�A�*

loss�=2D�/       �	F~��Yc�A�*

lossxh�<��f       �	.��Yc�A�*

loss&�=�,       �	���Yc�A�*

loss��?�,       �	���Yc�A�*

lossG0=���       �	�5��Yc�A�*

loss���=��9}       �	�ͽ�Yc�A�*

loss�{>d��       �	s��Yc�A�*

loss�;�>q�u       �	���Yc�A�*

loss���=�T�0       �	S���Yc�A�*

loss#4�=�&       �	LQ��Yc�A�*

loss��=f��        �	����Yc�A�*

loss�t>�k       �	����Yc�A�*

loss�, >����       �	&8Yc�A�*

loss2�9>����       �	c�Yc�A�*

lossM>�hDh       �	^iÓYc�A�*

loss�^4>�I       �	_	ēYc�A�*

lossm�>z�		       �	v�ēYc�A�*

loss��>Hk��       �	�GœYc�A�*

loss�1�>��^�       �	~�œYc�A�*

lossW�y>(�a       �	V~ƓYc�A�*

loss.�=J�c       �	"ǓYc�A�*

loss^&>l.�d       �	��ȓYc�A�*

lossYq�=P�o�       �	 {ɓYc�A�*

loss�=�>��       �	UʓYc�A�*

loss�b�=�ٺZ       �	�ʓYc�A�*

loss�!>"ǭ�       �	�n˓Yc�A�*

loss�{M>�W�b       �	�̓Yc�A�*

loss�Y}=D��       �	�ΓYc�A�*

loss�S�=���       �	��ΓYc�A�*

loss!�=��%       �	<MϓYc�A�*

loss$Ԛ=>H�       �	�ϓYc�A�*

loss)Zq=(��       �	^�ГYc�A�*

loss�w�='�        �	AѓYc�A�*

loss8�h>�F��       �	��ѓYc�A�*

loss��=B	��       �	�tғYc�A�*

loss�I�=�Y��       �	�ӓYc�A�*

loss��>��
       �	��ӓYc�A�*

loss�9�<3s�R       �	�GԓYc�A�*

loss�f�=a?"�       �	�-ՓYc�A�*

loss���=�Ϝ]       �	K�ՓYc�A�*

losscNZ=��n�       �	Ze֓Yc�A�*

loss�e�=R�       �	�4דYc�A�*

lossT{>�:hN       �	BvؓYc�A�*

lossZ�>��       �	KٓYc�A�*

loss��=}��1       �	%�ٓYc�A�*

loss�օ=��O       �	��ړYc�A�*

loss��=C�"       �	�ۓYc�A�*

loss��0>צ^�       �	�ۓYc�A�*

loss@�=�T�_       �	RܓYc�A�*

lossA�=�L3�       �	��ܓYc�A�*

lossr�=��       �	ǃݓYc�A�*

loss�>�I       �	aޓYc�A�*

loss��A>c�	�       �	ϾޓYc�A�*

loss�` >�w4X       �	�VߓYc�A�*

lossi��=��       �		�ߓYc�A�*

loss$>!��       �	����Yc�A�*

loss�f�=�J�       �	�X��Yc�A�*

loss�SL>/���       �	A���Yc�A�*

loss{�>q�$       �	h���Yc�A�*

loss w0>WV�       �	�4��Yc�A�*

loss���=�~�       �	����Yc�A�*

loss��#>Ō��       �	�n �Yc�A�*

loss��6>W�q�       �	��Yc�A�*

lossft>�       �	���Yc�A�*

loss�[>k[5m       �	?�Yc�A�*

loss�
=r��;       �	��Yc�A�*

lossM{�=��m)       �	�o�Yc�A�*

lossF��=_6HF       �	��Yc�A�*

loss�o,>&��       �	ѯ�Yc�A�*

loss�# >/?       �	�F�Yc�A�*

lossu>�jh       �	���Yc�A�*

loss�T=�y�s       �	N�Yc�A�*

loss�>S��       �	��Yc�A�*

loss\|i=�j��       �	��Yc�A�*

loss6�>�h�       �	pD�Yc�A�*

loss�7�=]˲a       �	���Yc�A�*

lossA�d>ol��       �	�z	�Yc�A�*

loss�6>t�7�       �	@
�Yc�A�*

loss�o>��2       �	ޫ
�Yc�A�*

loss���=��fz       �	�G�Yc�A�*

loss��=�㶑       �	���Yc�A�*

loss`>��~i       �	��Yc�A�*

loss�>�M�k       �	��Yc�A�*

lossWO>tE�\       �	���Yc�A�*

loss�E2>�E       �	pD�Yc�A�*

lossZ�=��B       �	n��Yc�A�*

loss>�=�HU       �	|�Yc�A�*

loss�4>�`�l       �	��Yc�A�*

loss�/�>�|�       �	��Yc�A�*

lossj7R=]U�$       �	�D�Yc�A�*

loss��C>��"	       �	O��Yc�A�*

loss��K=�3�n       �	���Yc�A�*

losse�K>c�c�       �	�S�Yc�A�*

lossݞ>� ;�       �	2�Yc�A�*

loss�%�>�I3       �	��Yc�A�*

lossx�!>��T�       �	~p�Yc�A�*

loss$Q�=���       �	E�Yc�A�*

loss���=���       �	W��Yc�A�*

loss�pZ>x��       �	�{�Yc�A�*

loss��!>̟�       �	�%�Yc�A�*

loss�� >{�\       �	���Yc�A�*

loss[��=$}�       �	�Y�Yc�A�*

lossln�=��V�       �	���Yc�A�*

losstOJ>��p0       �	<��Yc�A�*

loss$�7=�N�       �	� �Yc�A�*

loss� �=�_�O       �	���Yc�A�*

loss�y�=�joR       �	�V�Yc�A�*

lossA�>W]       �	���Yc�A�*

loss=#?g�V       �	��Yc�A�*

loss,��=R0�       �	 �Yc�A�*

loss�7
=�|�       �	���Yc�A�*

loss�S=U>�j       �	Y��Yc�A�*

loss�6=�7�^       �	�" �Yc�A�*

lossι>�ˤ?       �	k� �Yc�A�*

loss+ >8�W�       �	�Q!�Yc�A�*

loss�>�C��       �	�!�Yc�A�*

lossƼ=���u       �	�"�Yc�A�*

loss���=�s       �	:#�Yc�A�*

lossMb�=�r"F       �	��#�Yc�A�*

loss�>E�E       �	Q$�Yc�A�*

loss?�=d�:M       �	��$�Yc�A�*

lossvd&> �p       �	��%�Yc�A�*

loss��8>��I       �	q&�Yc�A�*

loss`��>�k       �	��&�Yc�A�*

loss}�R>~ߣ       �	��'�Yc�A�*

loss���=�͜�       �	�a(�Yc�A�*

loss���=3J��       �	�(�Yc�A�*

lossp}>�p��       �	�)�Yc�A�*

loss�x!>�1       �	�C*�Yc�A�*

loss�?P>��@�       �	��*�Yc�A�*

loss!��= �#�       �	��+�Yc�A�*

loss�d>C�p�       �	�w,�Yc�A�*

loss��z>~+Br       �	�-�Yc�A�*

loss���=��F�       �	|�-�Yc�A�*

lossxf�=Ћ��       �	6W.�Yc�A�*

lossc\�=�~�@       �	��.�Yc�A�*

loss)>>�o.       �	��/�Yc�A�*

loss��=%��{       �	�M0�Yc�A�*

loss[��=��o�       �	6�0�Yc�A�*

loss���=�'{       �	C�1�Yc�A�*

lossZ>AF��       �	�-2�Yc�A�*

loss�$�=�H#F       �	��2�Yc�A�*

loss �=���       �	�d3�Yc�A�*

loss��=R�/�       �	�+4�Yc�A�*

loss@e=�~��       �	��5�Yc�A�*

loss�_>��<-       �	/56�Yc�A�*

loss�>���       �	y�6�Yc�A�*

loss�>�c�K       �	�k7�Yc�A�*

lossr�f>��ݦ       �	6�8�Yc�A�*

loss0n�>>d!�       �	"49�Yc�A�*

loss�=]�       �	��9�Yc�A�*

loss	S>�q8       �	�f:�Yc�A�*

loss6��=e�U�       �	� ;�Yc�A�*

loss�>�lS       �	נ;�Yc�A�*

loss2��=n�S�       �	I<�Yc�A�*

loss�ζ=��M       �	��<�Yc�A�*

loss��=�$       �	=�Yc�A�*

loss8>�)B�       �	�!>�Yc�A�*

loss;�_=�S �       �		�>�Yc�A�*

loss�=>w��b       �	��?�Yc�A�*

loss=�>�ʜ�       �	�9@�Yc�A�*

loss�bK=-�"{       �	�@�Yc�A�*

loss�9=H�:       �	(~A�Yc�A�*

loss�E>����       �	�B�Yc�A�*

loss8K>Ǥ4�       �	�B�Yc�A�*

loss2�+>��zf       �	�VC�Yc�A�*

loss��=�uj       �	��C�Yc�A�*

lossI�*>N��       �	t�D�Yc�A�*

loss�2J=��{�       �	3E�Yc�A�*

loss���=�P%�       �	�E�Yc�A�*

loss�=T�       �	CtF�Yc�A�*

loss$�=-�       �	�JG�Yc�A�*

lossX�>�X4       �	?�G�Yc�A�*

lossRg�=ի��       �	�zH�Yc�A�*

loss�^�=��<�       �	�&I�Yc�A�*

loss=&>K7]�       �	��I�Yc�A�*

loss�� >�$��       �	�YJ�Yc�A�*

loss�u	>�ۊz       �	��J�Yc�A�*

loss0 >Oڨ�       �	k�K�Yc�A�*

loss<2=��       �	�$L�Yc�A�*

lossD8�=t�f�       �	�L�Yc�A�*

loss�>{>�X9       �	�TM�Yc�A�*

loss��=���       �	��M�Yc�A�*

loss�>C�       �	D�N�Yc�A�*

loss4��=��       �	F#O�Yc�A�*

loss�;�=��`a       �	(�O�Yc�A�*

lossx��=w?'c       �	�NP�Yc�A�*

loss=T�=���!       �	�P�Yc�A�*

loss��.={��       �	֍Q�Yc�A�*

loss��=^R"       �	�&R�Yc�A�*

loss�|>��!l       �	��R�Yc�A�*

loss��>�
Tb       �	bS�Yc�A�*

loss? >pϱ;       �	�S�Yc�A�*

loss{��=J8t�       �	��T�Yc�A�*

loss}��=�k�       �	�2U�Yc�A�*

loss)2�<�r\�       �	��U�Yc�A�*

loss��
>7,�A       �	2vV�Yc�A�*

lossFW�=�w�       �	LW�Yc�A�*

loss�q>��*       �	��W�Yc�A�*

lossaF1>�i�       �	WX�Yc�A�*

loss��=��s        �	��X�Yc�A�*

lossE��=�uY       �	��Y�Yc�A�*

loss�b=P�R       �	kDZ�Yc�A�*

loss���=���l       �	�Z�Yc�A�*

loss�Ͽ=��|�       �	��[�Yc�A�*

loss���=k�L�       �	�!\�Yc�A�*

loss-�&>]��       �	��\�Yc�A�*

lossIN>B<       �	U]�Yc�A�*

lossC�>C_       �	<�]�Yc�A�*

loss,�>���       �	7�^�Yc�A�*

loss��<C�b(       �	 _�Yc�A�*

loss��	>��h�       �	��_�Yc�A�*

loss8�n>�w�s       �	/O`�Yc�A�*

lossl�>z��       �	��`�Yc�A�*

loss�z�=��3j       �	b�a�Yc�A�*

loss���=���       �	�'b�Yc�A�*

loss�H�=�1       �	a�b�Yc�A�*

loss���=C��_       �	�Xc�Yc�A�*

loss%1=fΐ       �	��c�Yc�A�*

loss���=��k�       �	�d�Yc�A�*

loss���=�Ll�       �	�8e�Yc�A�*

loss	X�=\w�       �	4f�Yc�A�*

loss�b(>�r$H       �	T�f�Yc�A�*

loss�� >�8�       �	N�g�Yc�A�*

loss�"*=��v:       �	�Jh�Yc�A�*

loss�>�=X3t�       �	D�h�Yc�A�*

loss��=Ż�       �	��i�Yc�A�*

losssn >B�%R       �	Ɔl�Yc�A�*

lossF�f>�3�w       �	�2m�Yc�A�*

lossf�=^.��       �	�m�Yc�A�*

loss�>>�W�       �	�yn�Yc�A�*

loss�CP>�Ϗ�       �	R,o�Yc�A�*

lossB>�Vi       �	7�o�Yc�A�*

loss�>�=����       �	ҫp�Yc�A�*

loss�a�=���c       �	�Tq�Yc�A�*

loss�`
=�&C�       �	��q�Yc�A�*

lossw�A>tO       �	M�r�Yc�A�*

loss3��=�Cu       �	/�s�Yc�A�*

loss3U8=�J�       �	}t�Yc�A�*

loss���=�c       �	�0u�Yc�A�*

loss���=l^�S       �	��u�Yc�A�*

lossW��=Zb�       �	��v�Yc�A�*

lossZ�=����       �	�(w�Yc�A�*

losss��=]b�       �	y�w�Yc�A�*

loss��>��%�       �	�px�Yc�A�*

lossfj>����       �	�y�Yc�A�*

loss�n�=Ǝ�(       �	�y�Yc�A�*

loss��>�ſe       �	^ez�Yc�A�*

loss�֊=W��       �	P{�Yc�A�*

loss(�<=tp�       �	 �{�Yc�A�*

loss��f=k&��       �	J|�Yc�A�*

loss��=��Z�       �	��|�Yc�A�*

lossT�>H��       �	��}�Yc�A�*

loss*�"> ��       �	�~�Yc�A�*

losssp�=+v�!       �	�~�Yc�A�*

loss��=!&�       �	�Y�Yc�A�*

lossX�d=.�Z3       �	Z��Yc�A�*

loss�'�=�2|       �	n���Yc�A�*

loss_�=�g�       �	4��Yc�A�*

loss��=�Eج       �	�ȁ�Yc�A�*

loss*��=�       �	#k��Yc�A�*

loss
�=js"s       �	�,��Yc�A�*

lossc��=z �L       �	ȃ�Yc�A�*

loss� �=ČO+       �	N`��Yc�A�*

lossvG�=��uH       �	����Yc�A�*

loss3:�=Y94n       �	͕��Yc�A�*

loss#.>sW       �	Y1��Yc�A�*

loss�͍=h�m�       �	dˆ�Yc�A�*

loss��=�h�       �	kd��Yc�A�*

loss��=��_�       �	���Yc�A�*

loss�_�=*��       �	o���Yc�A�*

lossh� >��&�       �	]Q��Yc�A�*

loss��N>��H1       �	ꉔYc�A�*

loss��=6H��       �	����Yc�A�*

lossp�>5C�U       �	�6��Yc�A�*

loss�R�<�h��       �	�Ћ�Yc�A�*

lossD�=�g��       �	g��Yc�A�*

loss�">�Q�g       �	���Yc�A�*

lossh�>o�i�       �	����Yc�A�*

loss

>?��P       �	<��Yc�A�*

lossT9|=i�0       �	����Yc�A�*

loss�%
>tl�       �	�w��Yc�A�*

loss��>:@�       �	���Yc�A�*

loss��=�Fz[       �	o���Yc�A�*

loss��>�d_�       �	DQ��Yc�A�*

loss�ߒ=��?�       �	d둔Yc�A�*

loss��=���       �	񂒔Yc�A�*

loss�`&>.�7       �	7��Yc�A�*

lossfe�=�̼�       �	y���Yc�A�*

lossO�E>�Ds�       �	�G��Yc�A�*

loss�^�=HN       �	�ߔ�Yc�A�*

loss_?o=�8�       �	�x��Yc�A�*

loss:��=6�}�       �	n��Yc�A�*

loss�
>_c{       �	�ח�Yc�A�*

loss�R�=�cw�       �	����Yc�A�*

lossqÆ=���       �	љ�Yc�A�*

loss��=�_��       �	
��Yc�A�*

loss��>Z�       �	ף��Yc�A�*

loss��=O�^�       �	0G��Yc�A�*

lossn�=��L'       �	N�Yc�A�*

lossH]m>�O�       �	f���Yc�A�*

loss8^=�p	�       �	�&��Yc�A�*

loss�D�=g���       �	�Yc�A�*

loss|�>�o�       �	П�Yc�A�*

loss��7>D�OD       �	�g��Yc�A�*

lossCJ>g��       �	i���Yc�A�*

loss)W�=�w)�       �	����Yc�A�*

loss��=D�        �	j0��Yc�A�*

loss���==���       �	(Ң�Yc�A�*

lossfA,=��.       �	���Yc�A�*

loss�@]=���       �	0F��Yc�A�*

lossF��=�E-�       �	U���Yc�A�*

loss]s>�Z�x       �	ҍ��Yc�A�*

loss���=ȻY�       �	�*��Yc�A�*

loss���=���J       �	r���Yc�A�*

loss)>�=�T�       �	e��Yc�A�*

lossh֍=U6��       �	g��Yc�A�*

lossi(�>�p�       �	z���Yc�A�*

loss��=�a��       �	�I��Yc�A�*

lossn�[>r
ۦ       �	�쪔Yc�A�*

loss�=�=˱�J       �	m���Yc�A�*

loss��>��^�       �	o-��Yc�A�*

loss-�>)��       �	6ʬ�Yc�A�*

lossmQ>�Q�L       �	�f��Yc�A�*

loss3�!>����       �	���Yc�A�*

loss�W�=��E       �	S���Yc�A�*

loss�>�wq       �	�S��Yc�A�*

loss+.>�D       �	J'��Yc�A�*

lossX�=]�l�       �	
���Yc�A�*

lossZ��=�ad       �	jl��Yc�A�*

loss=W0>1���       �	8���Yc�A�*

loss�OT=ʞ       �	���Yc�A�*

loss��=��1       �	����Yc�A�*

losst�>��E       �	x���Yc�A�*

loss&�%>�Kz       �	�]��Yc�A�*

loss�`9=�\��       �	����Yc�A�*

loss���=0���       �	Н��Yc�A�*

loss-�>����       �	<��Yc�A�*

loss�d9=����       �	��Yc�A�*

lossMa�=}Ɖ�       �	~���Yc�A�*

loss��>;��       �	o+��Yc�A�*

loss�[�=3�	(       �	�й�Yc�A�*

loss1I_=eGn       �	!w��Yc�A�*

loss�`>�L�q       �	&��Yc�A�*

loss�B=�O.�       �	n���Yc�A�*

loss���=��:;       �	���Yc�A�*

loss��=u(;�       �	�I��Yc�A�*

loss\2)=�fZ�       �	d齔Yc�A�*

loss\J�=iC�/       �	'���Yc�A�*

loss-R�=����       �	m��Yc�A�*

lossS|�=Y��F       �	n���Yc�A�*

loss(�e=/6��       �	.Y��Yc�A�*

loss�>TG�Y       �	U���Yc�A�*

loss��>�i�       �	e���Yc�A�*

lossA�>de�       �	�XYc�A�*

losscI�=���       �	��Yc�A�*

loss�.�=4>��       �	�ÔYc�A�*

lossTN�=6�ò       �	�6ĔYc�A�*

loss$�=w���       �	�ĔYc�A�*

loss�B>=#��_       �	qŔYc�A�*

loss��>�(b       �	�ƔYc�A�*

loss�7=�C�       �	O�ƔYc�A�*

loss�>���       �	rǔYc�A�*

loss.�{=\w�       �	IȔYc�A�*

lossoE�=��,5       �	~�ȔYc�A�*

loss���=�       �	�DɔYc�A�*

loss{�>�m�+       �	��ɔYc�A�*

loss���=��w�       �	��ʔYc�A�*

loss�T�=���H       �	N'˔Yc�A�*

loss�I=�i��       �	��˔Yc�A�*

loss�>�y[       �	�w̔Yc�A�*

loss��>9{#u       �	q͔Yc�A�*

lossa�&=�F*       �	��͔Yc�A�*

loss��>vl��       �	RaΔYc�A�*

lossr��=˽�c       �	��ΔYc�A�*

loss�"=o$|�       �	g�ϔYc�A�*

loss��.>�V�A       �	�LДYc�A�*

loss�}�=B߲S       �	[�ДYc�A�*

loss��g=�@��       �	�єYc�A�*

lossX��=a0g�       �	@1ҔYc�A�*

loss;I>�zC       �	D�ҔYc�A�*

loss��=���       �	_|ӔYc�A�*

loss6�>bs��       �	!ԔYc�A�*

loss��>���V       �	r�ԔYc�A�*

loss���=h���       �	�eՔYc�A�*

lossJz=<t1�       �	2֔Yc�A�*

loss��=��R�       �	��֔Yc�A�*

loss�5g=�c�       �	fIהYc�A�*

lossd�'=���       �	*�הYc�A�*

loss���=��       �	/�ؔYc�A�*

loss��=����       �	�cٔYc�A�*

loss_�=��E       �	6ڔYc�A�*

loss�?>�|>       �	��ڔYc�A�*

loss-��=�_��       �	�Z۔Yc�A�*

loss���<�T�4       �	�۔Yc�A�*

lossxt�=,8^       �	,�ܔYc�A�*

loss,[=`�wu       �	6�ݔYc�A�*

loss1` >";�       �	�tޔYc�A�*

loss���=ٵ�       �	\ߔYc�A�*

loss�Ș>²�N       �	3�ߔYc�A�*

loss��
>�"��       �	����Yc�A�*

loss_8�=����       �	�w�Yc�A�*

loss-O">���       �	-'�Yc�A�*

loss�Z�=0);       �	��Yc�A�*

loss$h�=���       �	X�Yc�A�*

loss)�s=k�f        �	���Yc�A�*

loss�">RrE�       �	E��Yc�A�*

lossS<A=(�g4       �	RE�Yc�A�*

loss�Q>���}       �	/��Yc�A�*

loss��W>�a�}       �	,��Yc�A�*

lossQw�=A�"       �	�E�Yc�A�*

loss%��=0z��       �	���Yc�A�*

loss�r�=[�
�       �	D��Yc�A�*

loss_��=R=Za       �	[a�Yc�A�*

loss�e>u��       �	�3�Yc�A�*

losstC�= t�       �	���Yc�A�*

lossJ�=�|2       �	�j�Yc�A�*

lossmj+=2�!=       �	�?�Yc�A�*

loss��V>5��       �	��Yc�A�*

loss�3�=��:       �	�o�Yc�A�*

loss	�+>�g�}       �	|�Yc�A�*

loss��>,qj       �	*��Yc�A�*

loss�z�=W	"�       �	\t�Yc�A�*

loss��=�Mt       �	`:�Yc�A�*

lossn~f=��u       �	��Yc�A�*

loss��=ٓ�
       �	���Yc�A�*

loss���=���       �	��Yc�A�*

loss�=�
��       �	��Yc�A�*

loss���=&�.       �	rM�Yc�A�*

loss1I >j�z�       �	S��Yc�A�*

loss;�>�1��       �	����Yc�A�*

loss�OH>��"       �	hZ��Yc�A�*

loss���=��eJ       �	j���Yc�A�*

lossf�k=Yǩu       �	!���Yc�A�*

loss�@�=�u�       �	*��Yc�A�*

loss��=��s�       �	u���Yc�A�*

loss��=ቪ-       �	xa��Yc�A�*

lossԪ�=1��'       �	���Yc�A�*

loss��=�v�       �	����Yc�A�*

loss�3=���       �	�*��Yc�A�*

loss�\=Y1��       �	����Yc�A�*

lossn�E>e       �	�Z��Yc�A�*

loss��=o���       �	4��Yc�A�*

loss?� =`VNy       �	����Yc�A�*

loss&��=�7(�       �	>��Yc�A�*

loss�v>b�c)       �	���Yc�A�*

loss��=Z�J�       �	�n��Yc�A�*

loss|�>R�       �	���Yc�A�*

loss���=���N       �	����Yc�A�*

loss��=�`�&       �	�P �Yc�A�*

loss� 7>8�X�       �	�� �Yc�A�*

lossx��<��o~       �	/��Yc�A�*

loss���=]r��       �	80�Yc�A�*

loss:ɐ=�A��       �	��Yc�A�*

loss� >֍�@       �	�s�Yc�A�*

loss���=�ɓ�       �	9�Yc�A�*

loss(��=����       �	���Yc�A�*

loss�	>,�}�       �	�O�Yc�A�*

loss�<�<�z{�       �	9��Yc�A�*

loss/O=*@S�       �	K��Yc�A�*

loss�x>����       �	)"�Yc�A�*

loss=N�>�i��       �	���Yc�A�*

loss
��=�,�        �	eU�Yc�A�*

losshf�=9��       �	0��Yc�A�*

lossA>���r       �	ڐ	�Yc�A�*

lossM��=*Md       �	E,
�Yc�A�*

loss�� >�d��       �	6�
�Yc�A�*

loss���=O\=       �	�a�Yc�A�*

loss\y>8��       �	8��Yc�A�*

loss��*>��       �	��Yc�A�*

loss���=��q�       �	R(�Yc�A�*

loss���<Yt�       �	%��Yc�A�*

loss��=b�&       �	k�Yc�A�*

lossh#>A�       �	�=�Yc�A�*

loss�I	>*Wb       �	I��Yc�A�*

loss�K�=`3^F       �	Oy�Yc�A�*

lossȕ�=���       �	��Yc�A�*

loss��<R�[�       �	g��Yc�A�*

loss��C>���       �	!X�Yc�A�*

loss>���m       �	��Yc�A�*

loss�>�\�x       �	���Yc�A�*

loss��=�{��       �	~:�Yc�A�*

loss�}'>��)       �	��Yc�A�*

loss���=xg��       �		m�Yc�A�*

loss?��=Kի'       �	�Yc�A�*

lossD�>�oè       �	���Yc�A�*

loss+B>���       �	 7�Yc�A�*

lossy=
�
        �	���Yc�A�*

loss\ٺ=IF��       �	���Yc�A�*

loss!��=� ��       �	w��Yc�A�*

loss!W8=DtY�       �	"T�Yc�A�*

loss!�p=��	%       �	 ��Yc�A�*

loss�!+=��KS       �	,��Yc�A�*

loss--(=���       �	ϡ�Yc�A�*

loss즘=[ $       �	�U�Yc�A�*

loss#�D>���t       �	���Yc�A�*

loss�7�=l�d       �	<��Yc�A�*

loss���=�^�=       �	��Yc�A�*

loss�a�=~��E       �	�$ �Yc�A�*

loss��>=��e�       �	�� �Yc�A�*

lossK�=g���       �	�~!�Yc�A�*

loss�~>�� �       �	L"�Yc�A�*

loss�<�=�0JN       �	9�"�Yc�A�*

loss�=�?r�       �	ڍ#�Yc�A�*

loss_��=���       �	�0$�Yc�A�*

loss3/�=���       �	1%�Yc�A�*

loss�j�=�*&       �	��%�Yc�A�*

loss��T=����       �	�:&�Yc�A�*

losss�J>{1&       �	��&�Yc�A�*

lossN2>�~�       �	�m'�Yc�A�*

loss�\>p�O�       �	�
(�Yc�A�*

loss�!>��b�       �	��(�Yc�A�*

loss�+�=�։�       �	�x)�Yc�A�*

lossn}=ul7�       �	�"*�Yc�A�*

loss���=�       �	��*�Yc�A�*

loss�Љ=�S��       �	�Z+�Yc�A�*

lossα=[�w�       �	8�+�Yc�A�*

lossٰ>��,�       �	9�,�Yc�A�*

loss��R>\�t       �	�?-�Yc�A�*

losszq=����       �	�.�Yc�A�*

loss��L=�
��       �	��.�Yc�A�*

loss��=T��X       �	{N/�Yc�A�*

loss�t�=+� �       �	��/�Yc�A�*

loss
�>W9R�       �	)�0�Yc�A�*

loss���<��Ww       �	�11�Yc�A�*

lossJ>��       �	V�1�Yc�A�*

lossPǋ=��lH       �	)y2�Yc�A�*

lossm��=ͻ �       �	3�Yc�A�*

loss�'>6{��       �	ŭ3�Yc�A�*

lossƯk=�Ru       �	_4�Yc�A�*

lossK(=aլz       �	v5�Yc�A�*

lossɏ<���       �	�5�Yc�A�*

loss�Op=��       �	�P6�Yc�A�*

loss��]=s��A       �	|�6�Yc�A�*

loss��>�       �	��7�Yc�A�*

loss��:>&��l       �	�#8�Yc�A�*

loss�1=�Q��       �	��8�Yc�A�*

loss�F�=�j��       �	�9�Yc�A�*

loss�4->w+�       �	<2:�Yc�A�*

losszÁ=�h��       �	��:�Yc�A�*

lossf��=.��       �	�d;�Yc�A�*

loss�>��       �	��;�Yc�A�*

lossE"�=@7�       �	��<�Yc�A�*

loss#n�=��o       �	of=�Yc�A�*

loss��!>�.ga       �	>�Yc�A�*

loss��=��:�       �	$�>�Yc�A�*

lossd��=cm%       �	�0?�Yc�A�*

loss�\=��9       �	R�?�Yc�A�*

loss�<��LT       �	��@�Yc�A�*

loss�i=�<0       �	�$A�Yc�A�*

loss��u=[���       �	Q�A�Yc�A�*

loss=�>�S%�       �	�XB�Yc�A�*

losso�
=�|��       �	��B�Yc�A�*

loss�>�=�3R�       �	��C�Yc�A�*

loss��=��|�       �	u;D�Yc�A�*

loss��=%��-       �	o�D�Yc�A�*

loss��@=�ܣ�       �	XoE�Yc�A�*

loss��=R�a       �	�F�Yc�A�*

lossX7Q=n��h       �	:�F�Yc�A�*

lossƹ>o�2T       �	zG�Yc�A�*

loss��"=V��m       �	H�Yc�A�*

loss�_D>nt@       �	��H�Yc�A�*

loss�%=[�=�       �	|dI�Yc�A�*

loss��>���B       �	,J�Yc�A�*

lossL}�=s@<       �	S�J�Yc�A�*

loss=�h=��B       �	SXK�Yc�A�*

loss�>�=h��       �	$	L�Yc�A�*

loss�cN=W{Y�       �	֩L�Yc�A�*

loss�S>��Y8       �	�NM�Yc�A�*

loss��=D�Z       �	��M�Yc�A�*

loss�r�=Bm�       �	٘N�Yc�A�*

loss�wN=���a       �	^KO�Yc�A�*

lossR��=�tu       �	�O�Yc�A�*

loss�.}=��݈       �	��P�Yc�A�*

loss���=�~`�       �	�JQ�Yc�A�*

loss�i=:g�       �	 �Q�Yc�A�*

loss`">�#��       �	�R�Yc�A�*

loss��<]��       �	͔S�Yc�A�*

loss��=h�P       �	�*T�Yc�A�*

loss���=/�G9       �	��T�Yc�A�*

lossW(\=�I��       �	�U�Yc�A�*

lossO�=���y       �	�EV�Yc�A�*

loss�C	>J��/       �	T�V�Yc�A�*

loss�A	>�v       �	�yW�Yc�A�*

loss���<2l22       �	� X�Yc�A�*

lossY\�<\kR�       �	��X�Yc�A�*

loss�4�=\�        �	�Y�Yc�A�*

loss�B�<��W�       �	�+Z�Yc�A�*

lossɬ�;��É       �	7�Z�Yc�A�*

loss��w=lA�       �	�^[�Yc�A�*

lossF0=~��       �	+5\�Yc�A�*

loss�4=?�W       �	��\�Yc�A�*

loss2y�<0�6       �	w]�Yc�A�*

loss�,=߂��       �	�^�Yc�A�*

lossc`�=�E�       �	��^�Yc�A�*

loss��=��m�       �	�J_�Yc�A�*

loss��<C~&�       �	��_�Yc�A�*

loss��i;5�       �	��`�Yc�A�*

loss/�(=K��       �	�'a�Yc�A�*

lossӓ�=���       �	�a�Yc�A�*

loss�A=Vь0       �	rb�Yc�A�*

loss��	<�X       �	sc�Yc�A�*

loss� >F���       �	"�c�Yc�A�*

loss�M�>L��+       �	yd�Yc�A�*

lossw�V=.��Z       �	e�Yc�A�*

loss���<z�&       �	��e�Yc�A�*

loss�r=��z|       �	��f�Yc�A�	*

loss�s=]�V       �	�g�Yc�A�	*

loss3��=m�e�       �	�g�Yc�A�	*

lossR0�<�S       �	�eh�Yc�A�	*

loss�K>,Sj       �	��h�Yc�A�	*

loss���= �       �	J�i�Yc�A�	*

lossH@�=��Q�       �	��j�Yc�A�	*

lossV3>�"g       �	z4k�Yc�A�	*

loss���=�y��       �	x�k�Yc�A�	*

lossyE>��Q�       �	�l�Yc�A�	*

lossHG=>�2�       �	��m�Yc�A�	*

lossq��=�ݮ       �	��n�Yc�A�	*

loss^�	>P&��       �	G9o�Yc�A�	*

lossJj2>���       �	<�o�Yc�A�	*

loss2�0=a�       �	߉p�Yc�A�	*

loss7*�=����       �	-q�Yc�A�	*

loss3]�=�Lt       �	��q�Yc�A�	*

loss��4=��       �	�s�Yc�A�	*

loss��e=ݎ        �	n�s�Yc�A�	*

loss�e=V�?       �	�t�Yc�A�	*

loss:��=*}ȅ       �	�Fu�Yc�A�	*

loss���<\�[       �	#Jv�Yc�A�	*

lossT/=総<       �	e w�Yc�A�	*

loss��E=��eg       �	��w�Yc�A�	*

loss���=�iAr       �	VJx�Yc�A�	*

loss��d=��J'       �	H�x�Yc�A�	*

loss��
>
       �	c{y�Yc�A�	*

loss�'�=���?       �	�z�Yc�A�	*

loss?�z=���       �	��z�Yc�A�	*

loss}3>+�Zn       �	KX{�Yc�A�	*

loss�J�=X��       �	,�{�Yc�A�	*

loss���<X#dV       �	Y�|�Yc�A�	*

loss��V=XJ�       �	W=}�Yc�A�	*

loss{,=�̕K       �	}~�Yc�A�	*

loss�g�=�q}       �	r�~�Yc�A�	*

lossq��=WV        �	<�Yc�A�	*

loss��=߲	       �	���Yc�A�	*

lossE��=�I((       �	Di��Yc�A�	*

loss>g=�@v       �	���Yc�A�	*

loss�a�=��       �	����Yc�A�	*

loss�=�i�G       �	�u��Yc�A�	*

lossC>Κ1�       �	���Yc�A�	*

loss� >���       �	&���Yc�A�	*

lossG<�=W}       �	�U��Yc�A�	*

lossE�=��&�       �	\��Yc�A�	*

loss�#�<��>�       �	D���Yc�A�	*

loss�e�=-�.       �	IJ��Yc�A�	*

loss];u=���l       �	T↕Yc�A�	*

loss�>!=zN��       �	�|��Yc�A�	*

loss�ش=�u       �	�Q��Yc�A�	*

loss�J >M)�j       �	ꞕYc�A�	*

loss �>RS�       �	8���Yc�A�	*

lossM��=�;_�       �	���Yc�A�	*

loss��=�6b�       �	H���Yc�A�	*

loss�I�=,���       �	�d��Yc�A�	*

lossȄk=��0       �	����Yc�A�	*

loss���=�Ͳ�       �	����Yc�A�	*

lossQ�!>?���       �	)��Yc�A�	*

loss/��=�k�r       �	����Yc�A�	*

loss���<KTlJ       �	OZ��Yc�A�	*

loss3<�=��zb       �	M���Yc�A�	*

loss���=l       �	򕥕Yc�A�	*

lossV��=(���       �	I-��Yc�A�	*

loss��=V"�       �	�Ʀ�Yc�A�	*

loss�z>"���       �	�h��Yc�A�	*

loss��<Gh��       �	C��Yc�A�	*

loss:��=�Cj�       �	N���Yc�A�	*

loss
�=����       �	-��Yc�A�	*

loss��W>��+�       �	ié�Yc�A�	*

lossE�=/[6�       �	![��Yc�A�	*

loss�y>��1�       �	�T��Yc�A�	*

loss\�B=bS�       �	�Yc�A�	*

loss�~>��F�       �	e���Yc�A�	*

loss�T�=�/�       �	1'��Yc�A�	*

loss��I=
Ęf       �	����Yc�A�	*

losst�=��z        �	̘��Yc�A�	*

loss��W=m��       �	}A��Yc�A�	*

loss=/�=�˜�       �	�篕Yc�A�	*

loss[x�=^��H       �	%���Yc�A�	*

lossM�=)��       �	�)��Yc�A�	*

loss}�=4�`�       �	��Yc�A�	*

loss�J�=׽_`       �	���Yc�A�	*

loss�EI>�zUi       �	G��Yc�A�	*

loss�<=���       �	à��Yc�A�	*

lossf�=���       �	���Yc�A�	*

loss�#=P��D       �	t\��Yc�A�	*

lossJØ=�3�       �	����Yc�A�	*

loss/x>�G	       �	�Yc�A�	*

loss�[>3z@       �	����Yc�A�	*

loss��>��q�       �	z��Yc�A�	*

lossQz�=�~��       �	���Yc�A�	*

loss�KI=�p,       �	+���Yc�A�	*

loss-E�=��V?       �	X���Yc�A�	*

loss�>__�       �	�J��Yc�A�	*

loss6��=sC.       �	�漕Yc�A�	*

lossOA�=.�}�       �	���Yc�A�	*

loss}>%��       �	l&��Yc�A�	*

loss�|�=��2       �	���Yc�A�	*

loss�=�~�i       �	Y��Yc�A�	*

loss�a�<\7a       �	-&��Yc�A�	*

loss]��=��A       �	j���Yc�A�	*

lossZ��=����       �	�\��Yc�A�	*

loss�D�>��R�       �	n���Yc�A�	*

lossb3="ݮ^       �	�Yc�A�	*

lossQ�<���       �	�NÕYc�A�	*

loss;�X<`��       �	��ÕYc�A�	*

loss�U�<)���       �	ēĕYc�A�	*

loss�>�x��       �	G:ŕYc�A�	*

loss��=B�L�       �	 �ŕYc�A�	*

loss$�=�43       �	aoƕYc�A�	*

loss�p�=��.       �	+ǕYc�A�	*

loss���={I{       �	��ǕYc�A�	*

loss�x=7��       �	zȕYc�A�	*

loss��=f\�       �	�!ɕYc�A�	*

loss�lD=�)�       �	v�ɕYc�A�	*

loss�FN=���w       �	`ʕYc�A�	*

lossE�=���E       �	M�ʕYc�A�	*

loss>`>
�=�       �	��˕Yc�A�	*

loss8�=���       �	�*̕Yc�A�	*

loss���=x,C       �	��̕Yc�A�	*

loss���=VG)�       �	�h͕Yc�A�	*

loss�ڏ=���R       �	�ΕYc�A�	*

loss�A�==k�       �	��ΕYc�A�	*

loss6L�=���       �	��ϕYc�A�	*

loss�K=��       �	�ЕYc�A�	*

loss���=�M       �	��ЕYc�A�	*

loss�$>-F(/       �	?VѕYc�A�	*

loss�=�S�       �	��ѕYc�A�	*

loss.~�=4EH       �	1�ҕYc�A�
*

loss���=���N       �	
0ӕYc�A�
*

loss#�>�*�*       �	��ӕYc�A�
*

loss_�=����       �	9aԕYc�A�
*

loss�o�=�}d\       �	9+ՕYc�A�
*

loss�=��       �	��ՕYc�A�
*

loss���=� �_       �	iV֕Yc�A�
*

loss>=��{       �	E�֕Yc�A�
*

loss�W}=��t�       �	�וYc�A�
*

loss�'n=�Җ       �	0ؕYc�A�
*

loss�a=�n�       �	��ؕYc�A�
*

loss���=X;�       �	~rٕYc�A�
*

loss��>�⒊       �	�ڕYc�A�
*

loss��P= <�       �	��ڕYc�A�
*

loss��>+���       �	��ەYc�A�
*

loss�I2>��-(       �	��ܕYc�A�
*

lossz,�<�MYN       �	Y5ݕYc�A�
*

lossإ�=��(�       �	��ݕYc�A�
*

loss/��=$�g�       �	��ޕYc�A�
*

loss�|Q>Z�O       �	�nߕYc�A�
*

loss���=K��
       �	T:��Yc�A�
*

loss*=��l�       �	����Yc�A�
*

loss���<�?�:       �	ȕ�Yc�A�
*

lossT��=�u�v       �	sJ�Yc�A�
*

loss��=^ȍ8       �	V(�Yc�A�
*

loss6Q�=���       �	�_�Yc�A�
*

loss=j�=�MD       �	��Yc�A�
*

loss� �<�3g�       �	:�Yc�A�
*

loss��=#y�P       �	?��Yc�A�
*

loss�o�=6e
:       �	��Yc�A�
*

lossr��=��G,       �	��Yc�A�
*

loss�>� �       �	��Yc�A�
*

lossS��=/�H       �	XV�Yc�A�
*

loss,Yi=0��       �	���Yc�A�
*

loss�sU=^[T       �	���Yc�A�
*

loss#[2=�W|�       �	��Yc�A�
*

loss�3/=@3c       �	���Yc�A�
*

loss9F=���       �	�S�Yc�A�
*

loss���=}��E       �	E��Yc�A�
*

loss�H>)��}       �	��Yc�A�
*

lossC�=��       �	K�Yc�A�
*

loss��=�F�       �	���Yc�A�
*

loss���=5$�       �	��Yc�A�
*

losss�>	�       �	 �Yc�A�
*

loss�5�=鈖       �	W��Yc�A�
*

loss�ś<}�H�       �	�J�Yc�A�
*

loss$A?=fZ��       �	N��Yc�A�
*

loss	SM=�KfL       �	�'�Yc�A�
*

loss�o=q�a�       �	���Yc�A�
*

loss��>F�       �	ݗ��Yc�A�
*

lossѬN>�^tV       �	r4��Yc�A�
*

losss=���6       �	���Yc�A�
*

loss=�=yt�       �	�u��Yc�A�
*

loss
��=�k|)       �	�G��Yc�A�
*

loss,,=�v�I       �	����Yc�A�
*

loss8�>M���       �	���Yc�A�
*

losse'r=�b�s       �	�L��Yc�A�
*

loss���<��a�       �	$��Yc�A�
*

loss;y�<�bA�       �	j���Yc�A�
*

loss�F�<�`�b       �	�Z��Yc�A�
*

lossR2=�O��       �	o���Yc�A�
*

loss��<A��K       �	Œ��Yc�A�
*

loss�{=�I�       �	,��Yc�A�
*

loss��==K�1       �	7���Yc�A�
*

loss�B�=��I       �	d��Yc�A�
*

lossh��=�4       �	� ��Yc�A�
*

loss8Y�=;Nq�       �	ӡ��Yc�A�
*

loss�O=�  .       �	�C �Yc�A�
*

loss?8�<O��       �	3� �Yc�A�
*

loss��=�w��       �	�*�Yc�A�
*

loss�J>y�#l       �	���Yc�A�
*

loss�Q�=
�z`       �	l��Yc�A�
*

loss��=�^�       �	�;�Yc�A�
*

loss=��=�}�       �	#��Yc�A�
*

loss�W�=���       �	���Yc�A�
*

loss&y�=,(�(       �	�)�Yc�A�
*

loss�{�=�؇       �	���Yc�A�
*

loss71F=m-(M       �	�n�Yc�A�
*

lossj�=iF��       �	��Yc�A�
*

loss��=���       �	���Yc�A�
*

loss�H=�1�       �	�f	�Yc�A�
*

loss
��=���s       �	�
�Yc�A�
*

loss�m/=�T�       �	��
�Yc�A�
*

loss�8!=	{h�       �	]Q�Yc�A�
*

loss���<�Z�       �	���Yc�A�
*

lossdZ=h|�       �	��Yc�A�
*

loss��"=N��%       �	a5�Yc�A�
*

lossz�9=��b6       �	@��Yc�A�
*

loss.Dv=]wI       �	��Yc�A�
*

lossI�J=���       �	�B�Yc�A�
*

loss�4=��wl       �	���Yc�A�
*

loss*�!=�F�_       �	L��Yc�A�
*

loss`[=,��       �	�J�Yc�A�
*

loss��=���       �	Y��Yc�A�
*

loss��= �&�       �	k��Yc�A�
*

lossӾ�<6���       �	�=�Yc�A�
*

loss&��=��t�       �	0��Yc�A�
*

loss:e>���       �	̴�Yc�A�
*

loss���=����       �	h\�Yc�A�
*

loss�4k=���       �	� �Yc�A�
*

loss�U�<��u       �	J��Yc�A�
*

lossM�D<���       �	�a�Yc�A�
*

loss���=؈��       �	B�Yc�A�
*

loss,5�=G�       �	S��Yc�A�
*

loss\��<D�9W       �	N�Yc�A�
*

loss��<�.D�       �	�L�Yc�A�
*

lossZ�<ᐆc       �	Nd�Yc�A�
*

loss���=Ys��       �	��Yc�A�
*

loss ��=��       �	P��Yc�A�
*

lossx�9=��҈       �	�y�Yc�A�
*

lossJ�=�#��       �	o�Yc�A�
*

loss
��=G~�8       �	�	�Yc�A�
*

loss%7=�-Zl       �	F �Yc�A�
*

losso5�=9�K�       �	S� �Yc�A�
*

loss_�<=       �	I�!�Yc�A�
*

lossCł;\2�       �	>@"�Yc�A�
*

loss��f=��%�       �	��"�Yc�A�
*

loss�c�=�5��       �	_�#�Yc�A�
*

loss/F�=�s6       �	�1$�Yc�A�
*

loss�7�=]��`       �	��$�Yc�A�
*

loss��m=�$�        �	�e%�Yc�A�
*

lossӲ�=����       �	��%�Yc�A�
*

loss:��<�AA       �	��&�Yc�A�
*

loss��u=l���       �	;'�Yc�A�
*

loss�=ԡ?�       �	��'�Yc�A�
*

lossdF�=�Xg       �	�|(�Yc�A�
*

loss���=/��       �	�)�Yc�A�
*

loss��=�Y�       �	�)�Yc�A�
*

lossU�=YJ�       �	�e*�Yc�A�*

loss�~�=3��       �	+�Yc�A�*

loss�(=	��       �	�+�Yc�A�*

lossi��=T�E       �	rS,�Yc�A�*

lossO�==;�       �	�-�Yc�A�*

loss�^H=I��
       �	�-�Yc�A�*

lossT��<S�       �	�Z.�Yc�A�*

loss��=��2�       �	�/�Yc�A�*

lossq��=�/�{       �	��/�Yc�A�*

loss$�=>q       �	��0�Yc�A�*

loss%z�=�e!�       �	Xp1�Yc�A�*

loss`�=�}
�       �	�2�Yc�A�*

loss�A�=��W�       �	*�2�Yc�A�*

loss�.�<41        �	8i3�Yc�A�*

loss��=z/��       �	4�Yc�A�*

loss�jO=��       �	��4�Yc�A�*

lossN�>�ml       �	=e5�Yc�A�*

lossPE=��8q       �	6�Yc�A�*

lossC<�=�       �	��6�Yc�A�*

loss���=��9�       �	l7�Yc�A�*

loss�>�b�7       �	�8�Yc�A�*

lossM�=Y�,       �	]�8�Yc�A�*

loss9�=í��       �	�d9�Yc�A�*

loss_��<DL8�       �	; :�Yc�A�*

loss:�<_X       �	8�:�Yc�A�*

loss�]�=Υ��       �	5A;�Yc�A�*

loss�Թ=4��       �	&�;�Yc�A�*

loss���=#�~<       �	M�<�Yc�A�*

loss>�+/       �	t#=�Yc�A�*

loss]�=\̫[       �	��=�Yc�A�*

loss$5�=uW,       �	�d>�Yc�A�*

lossZ��=�>�Q       �	?�Yc�A�*

lossN��<);͸       �	S�?�Yc�A�*

loss�@(=	.�-       �	[@�Yc�A�*

loss��=����       �	[A�Yc�A�*

loss&Ξ=X�:       �	�A�Yc�A�*

loss��=tn�       �	PTB�Yc�A�*

loss�F�=�$��       �	V�B�Yc�A�*

loss�z>#]�Q       �	}�C�Yc�A�*

loss��=�Ui�       �	m7D�Yc�A�*

lossax�=9��       �	c�D�Yc�A�*

lossၦ=��o�       �	�iE�Yc�A�*

lossMr=h�       �	h	F�Yc�A�*

lossX�=��       �	йF�Yc�A�*

loss�P�=>�4J       �	d[G�Yc�A�*

lossir�=���       �	4�G�Yc�A�*

loss��<�|p�       �	��H�Yc�A�*

loss%��<�_��       �	�,I�Yc�A�*

loss��<����       �	��I�Yc�A�*

lossi��=�,rQ       �	~sJ�Yc�A�*

lossL�=e�/i       �	K�Yc�A�*

loss��=���x       �	e�K�Yc�A�*

loss.i�=P�|�       �	�L�Yc�A�*

loss18=V�b�       �	�3M�Yc�A�*

lossi��=��       �	��M�Yc�A�*

loss2��=;��       �	�N�Yc�A�*

loss���={7d�       �	�PO�Yc�A�*

lossל�=�0�&       �	}�O�Yc�A�*

loss�(=(��       �	�P�Yc�A�*

loss�6�=�	T       �	�Q�Yc�A�*

loss�ӽ=r�.�       �	>�Q�Yc�A�*

lossNe�=���&       �	�LR�Yc�A�*

loss)=WT��       �	l�R�Yc�A�*

loss;�T=�\�J       �	f�S�Yc�A�*

loss�zy=��        �	�T�Yc�A�*

loss���=?�q       �	�T�Yc�A�*

loss��=�^|       �	`wU�Yc�A�*

lossz�>D�M       �	�V�Yc�A�*

loss�h�=����       �	6�V�Yc�A�*

loss�cJ=����       �	�IW�Yc�A�*

lossX�:=�@��       �	q�W�Yc�A�*

loss��=z��       �	6�X�Yc�A�*

lossz�=��J       �	
KY�Yc�A�*

loss�`<�8�1       �	��Y�Yc�A�*

loss��G='��       �	�zZ�Yc�A�*

loss�PR=���       �	=�[�Yc�A�*

lossτ�=� 7       �		O\�Yc�A�*

loss��	>�)=       �	��]�Yc�A�*

loss���=Ao�[       �	؃^�Yc�A�*

lossX5=�`2H       �	�#_�Yc�A�*

lossSo�=vy%X       �	z `�Yc�A�*

loss4�f=3���       �	�`�Yc�A�*

loss�v_<��m�       �	�Na�Yc�A�*

loss#P=�@�       �	��a�Yc�A�*

loss!��=I�X@       �	.�b�Yc�A�*

loss�d<= b�       �	�'c�Yc�A�*

loss��<���       �	h<d�Yc�A�*

loss*�c=�"c\       �	��d�Yc�A�*

lossV��=QƟ�       �	5}e�Yc�A�*

loss�`>=���       �	�lf�Yc�A�*

loss!�=�>�?       �	F
g�Yc�A�*

loss��l=N�q       �	��g�Yc�A�*

loss^��=4�	�       �	%Zh�Yc�A�*

losso�9=M�6�       �	L�h�Yc�A�*

loss&P�=kAB       �	��i�Yc�A�*

lossj�A=Z��       �	q;j�Yc�A�*

lossL�=FIZ�       �	K�j�Yc�A�*

loss �R=��$�       �	�k�Yc�A�*

loss��>?��w       �	�+l�Yc�A�*

loss�=Y�9       �	[�l�Yc�A�*

lossJ��=�3An       �	�um�Yc�A�*

losst�=�8v9       �	(n�Yc�A�*

lossv4�=��$       �	&�n�Yc�A�*

loss�n=׉�       �	�Go�Yc�A�*

loss[��=�]�O       �	2�o�Yc�A�*

loss�˗=�V�B       �	�{p�Yc�A�*

loss��=��Rg       �	�q�Yc�A�*

loss�o�=wig       �	ͫq�Yc�A�*

lossLgl=gPʊ       �	8Hr�Yc�A�*

loss}C+=")8       �	��r�Yc�A�*

loss��J=y��'       �	�s�Yc�A�*

losst�=X��(       �	_'t�Yc�A�*

loss� �=�4U       �	n�t�Yc�A�*

loss��<=�"L�       �	}Xu�Yc�A�*

lossm��=8��       �	� v�Yc�A�*

lossN3�=��B       �	<�v�Yc�A�*

lossx�<=����       �	dxw�Yc�A�*

loss�?=8/х       �	x�Yc�A�*

loss��!>��       �	>�x�Yc�A�*

lossaj)=l�R�       �	�Ky�Yc�A�*

loss�0�=ר�       �	d�y�Yc�A�*

loss��>��Q�       �	��z�Yc�A�*

loss�>O��       �	��{�Yc�A�*

lossI�H=�I��       �	�@|�Yc�A�*

loss�$�=n�o       �	�}�Yc�A�*

lossh��<	 -�       �	0�}�Yc�A�*

loss4IH=Z1C�       �	�Q~�Yc�A�*

loss�6>�V~       �	��~�Yc�A�*

loss`�=.%�       �	��Yc�A�*

loss�<B=���       �	�*��Yc�A�*

loss�(�=����       �	�À�Yc�A�*

loss)��=�;l�       �	�Z��Yc�A�*

loss�9g<f�n�       �	8���Yc�A�*

loss��j=��c       �	���Yc�A�*

loss|c�<���       �	+��Yc�A�*

loss/��=|�m       �	�Ń�Yc�A�*

loss�f&=PT��       �	�c��Yc�A�*

loss4� >rh�       �	����Yc�A�*

lossF�>�`Q       �	S���Yc�A�*

lossĜ<�:?       �	�4��Yc�A�*

loss�6�=�݅S       �	OɆ�Yc�A�*

loss$�=���U       �	 `��Yc�A�*

lossL-�=��8       �	n���Yc�A�*

loss-L)=j��       �	"���Yc�A�*

loss�*=!�p�       �	�'��Yc�A�*

loss�E[=��՛       �	'�Yc�A�*

lossvQ�<l��       �	2W��Yc�A�*

loss��=�Kf�       �	����Yc�A�*

loss�l�=�L�&       �	����Yc�A�*

lossHn3=�Y��       �	�*��Yc�A�*

lossF�;=���       �	�ƌ�Yc�A�*

loss�	"=��}       �	jl��Yc�A�*

loss�]=;��       �	���Yc�A�*

losse<�=DW�       �	L���Yc�A�*

losss4�=����       �	�A��Yc�A�*

loss
�p=\�-d       �	7ߏ�Yc�A�*

loss�8">����       �	���Yc�A�*

lossxYw>XZ�       �	1#��Yc�A�*

loss$�_=�i�       �	U���Yc�A�*

loss$hT=����       �	�\��Yc�A�*

lossf��=L��       �	���Yc�A�*

lossf�=a���       �	���Yc�A�*

loss���=Hex       �	e6��Yc�A�*

loss:/<=ɫ]
       �	>ϔ�Yc�A�*

loss#�=֮g�       �	Dl��Yc�A�*

loss�d4=���       �	r��Yc�A�*

lossS�k=��;�       �	毖�Yc�A�*

loss�Q�=Ul       �	���Yc�A�*

loss�N!=4�:       �	�8��Yc�A�*

loss7�->���       �	蘖Yc�A�*

lossDsh=��j�       �	����Yc�A�*

loss�=�$a       �	!��Yc�A�*

loss_>����       �	
���Yc�A�*

lossf=!�hr       �	�Z��Yc�A�*

loss:�>'���       �	?��Yc�A�*

loss��=Y��       �	���Yc�A�*

loss�q�<�sm�       �	y��Yc�A�*

lossT8=��ʗ       �	���Yc�A�*

lossmو=���       �	����Yc�A�*

loss���=�B�       �	C���Yc�A�*

loss_�=�k��       �	:Π�Yc�A�*

loss($=��       �	�p��Yc�A�*

lossCR~=[L*�       �	�[��Yc�A�*

loss�z�=�0��       �	|��Yc�A�*

loss(��=�JY       �	� ��Yc�A�*

loss	�=�~��       �	����Yc�A�*

loss���=���       �	2v��Yc�A�*

lossV�X=�^��       �	���Yc�A�*

loss��=~�<       �	0���Yc�A�*

loss2/�=��b       �	s��Yc�A�*

loss�|=�{��       �	;���Yc�A�*

loss'��=��i       �	N��Yc�A�*

lossvzp=�m
�       �	9|��Yc�A�*

loss-��<�U'       �	υ��Yc�A�*

lossC��=���;       �	�#��Yc�A�*

loss�!�=�f�       �	ۿ��Yc�A�*

loss�i�<	"       �	����Yc�A�*

loss*&=3       �	�?��Yc�A�*

lossp�=����       �	�.��Yc�A�*

loss6E�=���       �	�ԯ�Yc�A�*

loss3E	>uaP       �	�q��Yc�A�*

loss�R�=߄@       �	�	��Yc�A�*

lossX��=����       �	M���Yc�A�*

loss�J�=9�X=       �	�>��Yc�A�*

loss�(�=�r�       �	-Yc�A�*

lossS�-=�6        �	����Yc�A�*

loss�"B>"/�       �	|&��Yc�A�*

lossA�	>q��       �	�ƴ�Yc�A�*

loss���=���l       �	i��Yc�A�*

loss���<r`-       �	�l��Yc�A�*

loss�c�<$�7�       �	���Yc�A�*

loss�޼=�M
�       �	񡷖Yc�A�*

loss�=ޓa       �	=��Yc�A�*

losso!=Z|       �	C縖Yc�A�*

lossrg>wv�       �	�~��Yc�A�*

loss[��=A�M        �	���Yc�A�*

loss8>U��       �	>���Yc�A�*

loss�2�=�+�       �	�G��Yc�A�*

loss�n)>6]       �	�ܻ�Yc�A�*

loss!t~=4��S       �	�Ƽ�Yc�A�*

losshJ>��Y       �	�v��Yc�A�*

lossei�=��4�       �	
��Yc�A�*

loss�K?=��;       �	����Yc�A�*

loss�0>;� �       �	(D��Yc�A�*

loss�Vc=�+��       �	L޿�Yc�A�*

lossW�w<�:��       �	:v��Yc�A�*

loss�<l=8�*       �	���Yc�A�*

lossSM�=^s�       �	����Yc�A�*

loss�Q�<ڙ�:       �	�aYc�A�*

loss�,=�'�       �	�ÖYc�A�*

loss.��<_vV�       �	%�ÖYc�A�*

lossR�<{-�       �	7OĖYc�A�*

loss3�3=�b]3       �	��ĖYc�A�*

loss�9�=jQU�       �	��ŖYc�A�*

loss��=
�D�       �	�ƖYc�A�*

loss��=���       �	ԵƖYc�A�*

loss1=�=��       �	 `ǖYc�A�*

loss��/<�FO�       �	��ǖYc�A�*

loss�8�=Q�0�       �	�ȖYc�A�*

loss�M>���       �	Y0ɖYc�A�*

loss���=I�       �	`�ɖYc�A�*

loss���=\~�       �	�eʖYc�A�*

lossӍ">��n       �	�˖Yc�A�*

loss�>����       �	��˖Yc�A�*

loss�F1=^��       �	]̖Yc�A�*

loss���<E#?�       �	�̖Yc�A�*

loss���=��       �	ڑ͖Yc�A�*

lossdl>����       �	M,ΖYc�A�*

loss/�=�(6�       �	��ΖYc�A�*

lossV�O=��v]       �	![ϖYc�A�*

loss�Q6=��۔       �	��ϖYc�A�*

lossa�n=�S�       �	��ЖYc�A�*

loss�Z�=��P       �	Z*іYc�A�*

loss�8�<��       �	��іYc�A�*

loss)
}<��f�       �	-\ҖYc�A�*

loss�v(=,F�       �	��ҖYc�A�*

lossf��=��       �	�ӖYc�A�*

losswQP=6B�       �	�'ԖYc�A�*

lossQ�=���       �	��ԖYc�A�*

loss��S=�'�       �	bhՖYc�A�*

loss��=0��       �	�֖Yc�A�*

loss�$�=z�<       �	=�֖Yc�A�*

lossݕ<�?��       �	�^זYc�A�*

loss6Z�=<Q�g       �	��זYc�A�*

loss��=˲�       �	�ؖYc�A�*

lossd}X=x�!       �	@ٖYc�A�*

lossD}>_b�       �	��ٖYc�A�*

loss}F3=_��       �	��ږYc�A�*

loss���<&ac       �	u:ۖYc�A�*

loss�LI<$P�       �	�ܖYc�A�*

loss�G�=W��       �	��ܖYc�A�*

loss�3O=6��H       �	@�ݖYc�A�*

loss-��=��#�       �	��ޖYc�A�*

loss��6>�N��       �	�mߖYc�A�*

loss�=�=H�       �	I.��Yc�A�*

lossG)=䡋�       �	i �Yc�A�*

loss+�=��!       �	n��Yc�A�*

loss@~a=Vi,K       �	d��Yc�A�*

lossh,=>���       �	,b�Yc�A�*

loss��<��       �	0*�Yc�A�*

loss��=�}�       �	T��Yc�A�*

loss�Q�<���       �	�[�Yc�A�*

loss`�>�T��       �	���Yc�A�*

loss�͍=�!       �	���Yc�A�*

lossA��=�J/�       �	�C�Yc�A�*

loss,�"=Z��       �	��Yc�A�*

loss��<E�g�       �	Cs�Yc�A�*

loss*�F=<9l�       �	3�Yc�A�*

loss��<�7�       �	h��Yc�A�*

loss%�=�V       �	0G�Yc�A�*

loss�~< ��w       �	z��Yc�A�*

loss�h%=����       �	n��Yc�A�*

loss\��=��       �	�?�Yc�A�*

loss��#=-��       �	���Yc�A�*

loss#��<~��r       �	�t�Yc�A�*

loss���<\�e�       �	��Yc�A�*

loss̐
=:��       �	}��Yc�A�*

loss��}=�
�       �	I�Yc�A�*

lossͽ=rL�       �	?��Yc�A�*

loss;F�=�N+       �	~�Yc�A�*

lossV��=f�(�       �	3�Yc�A�*

lossخ�={3�       �	��Yc�A�*

loss�Z�=N�v       �	kH�Yc�A�*

loss�=!AP]       �	L��Yc�A�*

loss��=��6�       �	O=��Yc�A�*

loss�[�<���       �	J���Yc�A�*

loss���=CV#6       �	y��Yc�A�*

loss*��<a���       �	���Yc�A�*

lossw��=�%��       �	<���Yc�A�*

lossiU=۔E.       �	'���Yc�A�*

loss�`�=�҈�       �	���Yc�A�*

loss���<@�       �	e���Yc�A�*

loss��=���       �	�v��Yc�A�*

lossF_M=מ�       �	���Yc�A�*

loss�]N=m���       �	���Yc�A�*

lossD�<��N�       �	����Yc�A�*

loss`
B=-X�       �	����Yc�A�*

lossN�V=�a|�       �	�E��Yc�A�*

loss�! =Vh       �	H���Yc�A�*

loss�K�=�C��       �	����Yc�A�*

loss��>�r       �	�<��Yc�A�*

lossF�=�<t       �	 ���Yc�A�*

loss@4<R%a       �	�f �Yc�A�*

lossfԇ=L$?       �	��Yc�A�*

loss߽>�&"       �	���Yc�A�*

lossr6<{��       �	?V�Yc�A�*

loss�~V;&�s       �	���Yc�A�*

loss�7�<��L�       �	��Yc�A�*

loss�1r<f�b�       �	�;�Yc�A�*

lossh�<r���       �	t��Yc�A�*

loss��N<�a�W       �	�w�Yc�A�*

loss��R<-��       �	��Yc�A�*

loss���=�R�       �	���Yc�A�*

lossm<�;�       �	�>�Yc�A�*

lossE��;f��v       �	��Yc�A�*

loss�h2;�(B       �	rk	�Yc�A�*

lossf��<��/C       �	� 
�Yc�A�*

loss}�=�2X       �	k�
�Yc�A�*

loss� =[���       �	.r�Yc�A�*

lossc��; ���       �	-�Yc�A�*

loss�K>=�$U       �	)��Yc�A�*

loss���>�1�       �	�\�Yc�A�*

loss:�<����       �	���Yc�A�*

lossN/=�v�N       �	Q��Yc�A�*

loss�m=�& �       �	RH�Yc�A�*

loss���=]�P       �	y��Yc�A�*

loss��U=��8       �	���Yc�A�*

loss b<���       �	T;�Yc�A�*

loss�g�=|s0�       �	���Yc�A�*

loss;7=="ڈ�       �	{��Yc�A�*

loss���=�       �	�%�Yc�A�*

loss[��=Yr�P       �	D��Yc�A�*

loss�9�=4�H�       �	X�Yc�A�*

loss� >��.r       �	%��Yc�A�*

loss���=��       �	���Yc�A�*

lossD�7=��|       �	J&�Yc�A�*

lossf��=x8h       �	<��Yc�A�*

loss���=M��       �	e�Yc�A�*

loss3vd=�qN�       �	s��Yc�A�*

loss�C=Q�@       �	&��Yc�A�*

loss�*�=o��       �	(�Yc�A�*

loss1�=ސ�       �	���Yc�A�*

loss_W�<|ĺ�       �	*T�Yc�A�*

loss�v�=�ژ>       �	���Yc�A�*

loss���<�G��       �	�y�Yc�A�*

loss�+=���W       �	� �Yc�A�*

loss�>�<����       �	���Yc�A�*

loss�Z</�f~       �	��Yc�A�*

loss/��=�|*t       �	�I�Yc�A�*

loss$ʷ<�U'I       �	$F�Yc�A�*

loss3	V=sO��       �	�2 �Yc�A�*

loss���=;q��       �	{-!�Yc�A�*

loss :�=HF�F       �	��!�Yc�A�*

loss`k�=Է��       �	˼"�Yc�A�*

lossʳ==�ve       �	�X#�Yc�A�*

lossM�;���q       �	R�#�Yc�A�*

loss�@=�W�9       �	�$�Yc�A�*

loss]�<��T       �	��%�Yc�A�*

loss&�=��YE       �	�[&�Yc�A�*

loss��=����       �	��&�Yc�A�*

loss���=w��       �	��'�Yc�A�*

loss�[�=ݳ�       �	�#(�Yc�A�*

loss]gv=p0�t       �	0)�Yc�A�*

loss�~s<�P�       �	��)�Yc�A�*

loss��C=N��       �	�J*�Yc�A�*

lossd�=:��        �	��*�Yc�A�*

losswM:=�+       �	r+�Yc�A�*

loss;G�=D��       �	�
,�Yc�A�*

loss=g�T�       �	0�,�Yc�A�*

loss
��<w�R�       �	�4-�Yc�A�*

loss*�G=f��       �	m�-�Yc�A�*

loss}=�!�       �	&W.�Yc�A�*

loss��C<���       �	��.�Yc�A�*

loss�e�==�6       �	DoH�Yc�A�*

loss-�>s�Q       �	�I�Yc�A�*

lossx
>�xe       �	�I�Yc�A�*

loss���=)�b       �	HJ�Yc�A�*

lossBS�=R*�       �	K�Yc�A�*

loss��=x��Q       �	٘K�Yc�A�*

loss�9�=����       �	�]L�Yc�A�*

lossd!K=�l&�       �	'�L�Yc�A�*

loss�$�=pڄ�       �	W�M�Yc�A�*

lossô�=EJʚ       �	U4N�Yc�A�*

loss��<Z?�       �	x�N�Yc�A�*

loss�C�=���G       �	yO�Yc�A�*

loss}֯=/<Z�       �	fP�Yc�A�*

loss��=@���       �	d�P�Yc�A�*

loss�3=.Ӯi       �	�YQ�Yc�A�*

loss��7>�
�U       �	��Q�Yc�A�*

lossJ��<�R       �	O�R�Yc�A�*

lossx��=��&       �	�5S�Yc�A�*

lossK�<�}�       �	�S�Yc�A�*

loss��>ON+�       �	�nT�Yc�A�*

loss�=� 1�       �	�U�Yc�A�*

loss�|�= ��;       �	&�U�Yc�A�*

loss���=�)P       �	{LV�Yc�A�*

loss��=Pa��       �	��V�Yc�A�*

loss��{= ��       �	?�W�Yc�A�*

loss�2=zE��       �	�3X�Yc�A�*

lossD�p=���[       �	_Y�Yc�A�*

loss�&?=SXI       �	>�Y�Yc�A�*

lossL�)=�vj        �	�XZ�Yc�A�*

loss)�=�.Vo       �	��Z�Yc�A�*

loss��<O\�	       �	�[�Yc�A�*

lossZkM=����       �	Z�\�Yc�A�*

loss��=�C	       �	@]�Yc�A�*

lossd�=�� K       �	��]�Yc�A�*

lossr=s�       �	��^�Yc�A�*

loss�Fj=xn&�       �	�8_�Yc�A�*

loss=e�<��Y�       �	��_�Yc�A�*

loss3�>=G��s       �	�t`�Yc�A�*

loss링>�{E�       �	'a�Yc�A�*

loss审=���       �	��a�Yc�A�*

loss�\|=;֮j       �	�qb�Yc�A�*

loss�W
=��       �	c�Yc�A�*

lossh�=_Y�       �	̴c�Yc�A�*

loss�m�=zX�       �	SYd�Yc�A�*

loss��=�+�Z       �	��d�Yc�A�*

losst�=0u��       �	p�e�Yc�A�*

loss�@�=��4       �	D4f�Yc�A�*

loss� ==��l�       �	H�f�Yc�A�*

loss�o$=4��.       �	��g�Yc�A�*

loss!E;M|�u       �	>!h�Yc�A�*

loss�"=��z�       �	��h�Yc�A�*

loss���=����       �	Zbi�Yc�A�*

loss!e=[a;-       �	��i�Yc�A�*

loss��>L���       �	�j�Yc�A�*

loss-]�=l�1�       �	pBk�Yc�A�*

loss{P�;���       �	��k�Yc�A�*

loss��F<<�4       �	m<m�Yc�A�*

loss]. <�ik�       �	��m�Yc�A�*

loss��3= F=�       �	��n�Yc�A�*

loss���=M�2       �	�8o�Yc�A�*

loss��@=V��       �	��o�Yc�A�*

loss��=b|��       �	3�p�Yc�A�*

lossl9=p�5�       �	�-q�Yc�A�*

loss8�.=���]       �	}�q�Yc�A�*

lossZ%=!8��       �	�kr�Yc�A�*

loss8�<B�a2       �	x	s�Yc�A�*

loss.�=�Z�       �	E�s�Yc�A�*

loss��B=�PZ       �	>=t�Yc�A�*

lossO��=��>@       �	��t�Yc�A�*

loss&��=&���       �	�tu�Yc�A�*

loss���={{z�       �		v�Yc�A�*

loss ��=��{�       �	��v�Yc�A�*

loss��Z=�'       �	�`w�Yc�A�*

loss��I=�0E       �		�w�Yc�A�*

lossC��=�T�       �	�x�Yc�A�*

loss�C0=~�Ƙ       �	�,y�Yc�A�*

lossqvA=���       �	C�y�Yc�A�*

loss�23=>�e       �	taz�Yc�A�*

lossC�Z<�N��       �	<�z�Yc�A�*

lossMr=���&       �	��{�Yc�A�*

loss�lb=���J       �	�L|�Yc�A�*

lossE��=����       �	��|�Yc�A�*

lossۘN=񃻮       �	 {}�Yc�A�*

loss
;L<����       �	�~�Yc�A�*

loss��<��[�       �	�~�Yc�A�*

loss|�<�;��       �	܄�Yc�A�*

loss��=�Є       �	���Yc�A�*

loss7�C=}�l       �	g���Yc�A�*

loss!�}=�<��       �	F_��Yc�A�*

loss�o�={�4�       �	����Yc�A�*

loss[��="
��       �	w���Yc�A�*

loss�mR=p¹�       �	�;��Yc�A�*

lossi=�<�       �	?䃗Yc�A�*

loss�H�=�0�7       �	?���Yc�A�*

loss0�=8�e�       �	�Q��Yc�A�*

loss�>=~uY�       �	J셗Yc�A�*

lossl�?=����       �	ߋ��Yc�A�*

lossj?�=� ��       �	t'��Yc�A�*

loss�"�=�8ٗ       �	�Ň�Yc�A�*

lossq� =�z�K       �	Su��Yc�A�*

loss|��;�wS2       �	j��Yc�A�*

loss椹=�N{       �	�։�Yc�A�*

losse]�=}�8t       �	�m��Yc�A�*

loss�w�<�}�       �	���Yc�A�*

loss=?��$       �	ࠋ�Yc�A�*

loss�j]=�⨧       �	z9��Yc�A�*

loss�au=�҄       �	�Ԍ�Yc�A�*

loss6X<���       �	���Yc�A�*

loss��=�m9�       �	�D��Yc�A�*

loss��#=Zap       �	Dێ�Yc�A�*

lossJ,�=��RF       �	�u��Yc�A�*

lossIm=�!a$       �	d��Yc�A�*

loss�=f�CQ       �	����Yc�A�*

loss�)X=����       �	����Yc�A�*

loss6ċ=�S��       �	� ��Yc�A�*

lossf�<U��       �	c���Yc�A�*

loss =<y��       �	Z��Yc�A�*

lossR��=v#�~       �	��Yc�A�*

lossW*=z<o�       �	􋔗Yc�A�*

loss��<�t��       �	�/��Yc�A�*

lossS��=ϒw�       �	ȕ�Yc�A�*

loss�#�=$
D�       �	�_��Yc�A�*

loss��7=7���       �	���Yc�A�*

loss���=M�"�       �	cYc�A�*

loss��9=��S�       �	����Yc�A�*

loss,Y�=�~       �	>=��Yc�A�*

loss�^�=��Qq       �	[홗Yc�A�*

loss�"?=��       �	���Yc�A�*

loss��>9�F9       �	�8��Yc�A�*

loss2��=���       �	
��Yc�A�*

lossuC=G�r       �	�ל�Yc�A�*

loss��Q=�0       �	���Yc�A�*

loss$�O=~�       �	�ភYc�A�*

loss&&�<�ݵ       �	G柗Yc�A�*

loss�:>�X@       �	���Yc�A�*

lossX�=�V��       �	��Yc�A�*

loss��<����       �	�á�Yc�A�*

loss�-k=��zw       �	]��Yc�A�*

loss9�<��η       �	����Yc�A�*

loss�d�<�Ҝ�       �	p���Yc�A�*

lossV�R<���       �	�H��Yc�A�*

loss*~=� ^G       �	��Yc�A�*

lossEei=�*�       �	ӡ��Yc�A�*

losss>��h       �	�T��Yc�A�*

lossQ�>���       �	����Yc�A�*

loss��<��n�       �	q���Yc�A�*

lossC�=gx��       �	�L��Yc�A�*

loss}.F<p�Զ       �	�娗Yc�A�*

loss�$,=�֚        �	���Yc�A�*

lossFޞ='F�}       �	��Yc�A�*

loss۞j=� �       �	¾��Yc�A�*

loss��=+3       �	@���Yc�A�*

loss�s_=���<       �	J%��Yc�A�*

lossh=`=Ҿ��       �	&Ƭ�Yc�A�*

loss<�=7���       �	l_��Yc�A�*

loss�Ȣ<�D�       �		��Yc�A�*

loss���=xCV�       �	P���Yc�A�*

loss\$�=��       �	�J��Yc�A�*

loss��=v�h       �	�鯗Yc�A�*

loss_}=H	��       �	����Yc�A�*

lossz�_=jyV&       �	K9��Yc�A�*

loss�V3=j]�       �	oر�Yc�A�*

loss8r]=�_��       �	u��Yc�A�*

loss�0!<�H�b       �	Z��Yc�A�*

lossנm=�|$�       �	I���Yc�A�*

loss�{=���       �	�X��Yc�A�*

lossm$�<E��c       �	����Yc�A�*

loss!cM=[�Ƌ       �	R���Yc�A�*

loss��y=" ��       �	7��Yc�A�*

loss���<�u�       �	Ѷ�Yc�A�*

lossR�=�,v�       �	�m��Yc�A�*

loss&((<�k�       �	���Yc�A�*

loss��=��E       �	4���Yc�A�*

loss-=��       �	���Yc�A�*

loss�Q<����       �	�4��Yc�A�*

loss6�<����       �	غ�Yc�A�*

loss��=�@�C       �	�w��Yc�A�*

loss�o=���       �	�"��Yc�A�*

loss���=	       �	!̼�Yc�A�*

lossE�=\k�       �	�i��Yc�A�*

loss��/= :�       �	���Yc�A�*

lossQ�7=��S�       �	���Yc�A�*

lossZ��<
��       �	bL��Yc�A�*

loss�	�<T�[       �	���Yc�A�*

loss���<(�=�       �	z���Yc�A�*

loss���<у��       �	c*��Yc�A�*

loss�4=(*!'       �	����Yc�A�*

lossX�H=�B       �	�mYc�A�*

loss8�V=��rq       �	�×Yc�A�*

lossL�=�d
       �	��×Yc�A�*

lossm��==�       �	�MėYc�A�*

loss�s=�ng       �	qWŗYc�A�*

loss�=g��~       �	mƗYc�A�*

loss�%�<�a~\       �	b�ƗYc�A�*

lossd��<j,
       �	u:ǗYc�A�*

loss�=�
Q�       �	�ǗYc�A�*

lossb�=.��:       �	�ȗYc�A�*

loss�Q�=��f       �	u!ɗYc�A�*

loss	Ħ=�u	�       �	��ɗYc�A�*

loss�
�=�(�=       �	>]ʗYc�A�*

loss��=���<       �	P�ʗYc�A�*

lossY�='Ǥ�       �	і˗Yc�A�*

lossv=6��       �	��̗Yc�A�*

loss���<��G       �	�,͗Yc�A�*

loss�o=��       �	��͗Yc�A�*

loss��7=��Wm       �	,fΗYc�A�*

loss���=��^�       �	UiϗYc�A�*

loss3-�<�w��       �	{�ЗYc�A�*

lossw�3=��Yr       �	�$їYc�A�*

loss,�D=䌈�       �	r�їYc�A�*

loss�l�=�]�L       �	�hҗYc�A�*

loss�V&=�s9z       �	5
ӗYc�A�*

loss�;=
�X       �	��ӗYc�A�*

loss��Y=��%7       �	VԗYc�A�*

lossJ��=v�̶       �	�ԗYc�A�*

loss&��=���       �	�"֗Yc�A�*

loss�в=X���       �	��֗Yc�A�*

loss+�>n��       �	�pחYc�A�*

loss�S�=*<       �	bؗYc�A�*

loss?4�=���       �	ʧؗYc�A�*

loss�<t-��       �	�CٗYc�A�*

loss��=��ŧ       �	8�ٗYc�A�*

loss��[=�i��       �	�oڗYc�A�*

loss�H�=H�(       �	�(ۗYc�A�*

lossl�g=t��       �	��ۗYc�A�*

loss[=kr�       �	\ܗYc�A�*

loss�~�=#�LB       �	�ݗYc�A�*

loss�� =[	�       �	E�ݗYc�A�*

loss!Z�<#���       �	>uޗYc�A�*

loss�o�=��j:       �	�ߗYc�A�*

loss���<şN�       �	��ߗYc�A�*

loss��H<��
�       �	�<��Yc�A�*

loss���=�y       �	����Yc�A�*

loss,�<�Sp�       �	�u�Yc�A�*

loss�Q=S�R�       �	�I�Yc�A�*

lossmY�=P��       �	h�Yc�A�*

loss��<ٟ�       �	b��Yc�A�*

loss�^p=�'"       �	_@�Yc�A�*

loss���=�	       �	!��Yc�A�*

loss9�<��uL       �	w��Yc�A�*

loss�Ƽ<
��u       �	� �Yc�A�*

loss�=m��       �	ü�Yc�A�*

lossa��<���       �	�Y�Yc�A�*

loss$ǜ=��D�       �	H��Yc�A�*

losso��=@- f       �	Q��Yc�A�*

loss!D�=(Z�       �	[�Yc�A�*

loss�`=���V       �	��Yc�A�*

loss�ʊ<;���       �	q��Yc�A�*

loss��w=I\�       �	/�Yc�A�*

lossi�=���       �	���Yc�A�*

loss=up�       �	���Yc�A�*

lossl=:��5       �	�$�Yc�A�*

lossJ��=$��       �	���Yc�A�*

lossV�<���       �	SY�Yc�A�*

loss�|<E��       �	���Yc�A�*

loss���<cR�u       �	��Yc�A�*

lossS��=6竏       �	^e�Yc�A�*

loss��1=-���       �	ߤ�Yc�A�*

loss��e=B��u       �	JB�Yc�A�*

loss�);=Y�^�       �	3��Yc�A�*

loss�k�=��n�       �	|�Yc�A�*

lossqK;=�\�       �	/��Yc�A�*

loss�5�=թ�n       �	9���Yc�A�*

loss(�P=p#�       �	*X��Yc�A�*

loss�@�=��       �	���Yc�A�*

loss]�<[�~i       �	���Yc�A�*

loss
��=�a�       �	�m��Yc�A�*

loss�v =m�^�       �	G��Yc�A�*

loss�~�=ؓ       �	A���Yc�A�*

loss�'K=��       �	�:��Yc�A�*

loss�Dy=s�t�       �	�`��Yc�A�*

loss==O=S��       �	@���Yc�A�*

loss�?�=^r�       �	����Yc�A�*

lossj3=㳣a       �	�-��Yc�A�*

loss��a=�O�l       �	���Yc�A�*

loss8��=E�_       �	�[��Yc�A�*

loss3��<1�h�       �	���Yc�A�*

lossÐ�=G;-�       �	֬��Yc�A�*

loss+ �=J��       �	(F��Yc�A�*

losss��=�9�       �	z���Yc�A�*

loss�a<��       �	%w �Yc�A�*

loss��<���       �	s�Yc�A�*

loss��f=tQ��       �	��Yc�A�*

loss<i
$       �	���Yc�A�*

loss?�=規h       �	�d�Yc�A�*

loss��=��        �	F	�Yc�A�*

loss$+=Z�,       �	<��Yc�A�*

lossM�=���       �	�6�Yc�A�*

lossc5=E��=       �	��Yc�A�*

lossQg<&�4       �	�p�Yc�A�*

loss���<rw�C       �	��Yc�A�*

loss�
@= l�       �	I��Yc�A�*

loss���<*���       �	�?�Yc�A�*

lossjL-=^��       �	���Yc�A�*

loss\�=��W       �	q	�Yc�A�*

loss1a�=��:�       �	W
�Yc�A�*

loss��<�]�       �	�
�Yc�A�*

loss�˼=��       �	�I�Yc�A�*

loss��=��1�       �	���Yc�A�*

loss�+�=���       �	U��Yc�A�*

lossK�=ޭZ       �	0,�Yc�A�*

loss�1=���       �	���Yc�A�*

loss? �=r�UA       �	=�Yc�A�*

loss;�>r�G$       �	�!�Yc�A�*

loss2��=!�ȅ       �	E��Yc�A�*

loss�߷=����       �	�[�Yc�A�*

loss@��<rAo       �	�Yc�A�*

lossLGp=�D�       �	��Yc�A�*

loss��=����       �	1A�Yc�A�*

loss�k=���h       �	��Yc�A�*

lossE�a=R��-       �	?t�Yc�A�*

loss��;=�Xl       �	(�Yc�A�*

loss�&h=Q4�>       �	&��Yc�A�*

loss�`�<,��       �	"S�Yc�A�*

loss��R<����       �	���Yc�A�*

lossm�o=�S<       �	��Yc�A�*

loss%L=ˮ�       �	b0�Yc�A�*

loss#z�<���       �	��Yc�A�*

loss�U�=���       �	B��Yc�A�*

loss�=E�       �	�K�Yc�A�*

loss\V�=H#L_       �	R��Yc�A�*

loss�;�=��E�       �	��Yc�A�*

loss+��=#�H�       �	,,�Yc�A�*

loss��t<���4       �	���Yc�A�*

loss�$�=F_�|       �	�`�Yc�A�*

loss�q�=�h��       �	���Yc�A�*

loss�>�<��w�       �	��Yc�A�*

loss�A�<7�q:       �	�K�Yc�A�*

loss��=��$�       �	���Yc�A�*

loss��=QFVD       �	���Yc�A�*

lossM�B=��z       �	�+ �Yc�A�*

lossE��<CӃ       �	�� �Yc�A�*

lossjG�<ټ��       �	�s!�Yc�A�*

loss�g<6�q�       �	!"�Yc�A�*

loss���=9"��       �	6�"�Yc�A�*

loss�*=[8b�       �	�t#�Yc�A�*

loss���=c�Դ       �	�$�Yc�A�*

loss}�&=dm�F       �	��$�Yc�A�*

loss�ZU=bl       �	�M%�Yc�A�*

loss=Ή<��        �	��%�Yc�A�*

losss}�<���       �	��&�Yc�A�*

lossTU�<v�ݾ       �	�*'�Yc�A�*

loss1w>=g,g�       �	��'�Yc�A�*

loss���=�X�       �	
d(�Yc�A�*

lossT#>�F�       �	1)�Yc�A�*

lossd-�=���       �	>�)�Yc�A�*

loss'��<���       �	 *�Yc�A�*

loss��^=+(�a       �	!+�Yc�A�*

loss�Z=����       �	ض+�Yc�A�*

loss��;=�C�       �	xa,�Yc�A�*

loss;G=� 	�       �	 -�Yc�A�*

lossi�9=�$v       �	��-�Yc�A�*

loss��R<.Q-�       �	�<.�Yc�A�*

lossH�>|ZA�       �	��.�Yc�A�*

loss��`=ق�       �	�q/�Yc�A�*

loss���<�P3       �	80�Yc�A�*

loss��=
A�y       �	?�0�Yc�A�*

loss}o<$s�       �	�E1�Yc�A�*

lossƋP=y8=       �	�2�Yc�A�*

loss�@�=c_ڿ       �	Ӣ2�Yc�A�*

lossȞ�=��}       �	�=3�Yc�A�*

lossd=*��       �	��3�Yc�A�*

loss�`�<gFDx       �	�n4�Yc�A�*

lossQ_'>:�]'       �	[5�Yc�A�*

loss'�>�ф       �	�5�Yc�A�*

loss���=A�b�       �	�6�Yc�A�*

loss�D\=.���       �	�7�Yc�A�*

loss��<	>̃       �	��8�Yc�A�*

lossN�=0JG3       �	�%9�Yc�A�*

loss��=���       �	8�9�Yc�A�*

loss_� >tϳ       �	�l:�Yc�A�*

loss_�]=f��       �	�;�Yc�A�*

loss{#=K�m�       �	c�;�Yc�A�*

loss�X�=�9�       �	=�Yc�A�*

loss�u=c��[       �	>�Yc�A�*

lossw��=>�       �	\�>�Yc�A�*

lossL�=��       �	��?�Yc�A�*

lossEĄ=��r[       �	$+@�Yc�A�*

lossv^F=�ϯ�       �	g�@�Yc�A�*

loss6g�=��R'       �	�MB�Yc�A�*

loss��=���       �	O�B�Yc�A�*

loss�f�=&��       �	s�C�Yc�A�*

loss��=�p;�       �	�D�Yc�A�*

loss�ʤ<��G�       �	f�D�Yc�A�*

loss�V�<���       �	\�E�Yc�A�*

loss;��<UMd�       �	q8F�Yc�A�*

lossM"�=�G8�       �	��F�Yc�A�*

loss��=`��=       �	hG�Yc�A�*

loss_��<S���       �	�H�Yc�A�*

loss�J=���       �	ŪH�Yc�A�*

lossq�=�s       �	�DI�Yc�A�*

lossI�?=\�&       �	Q�I�Yc�A�*

loss R�<׌D�       �	.�J�Yc�A�*

loss�_=v��       �	O$K�Yc�A�*

loss�C�<~�       �	��K�Yc�A�*

loss);5>��Y�       �	hL�Yc�A�*

loss�@#=ҁ       �	��L�Yc�A�*

lossn1J=��eO       �	/�M�Yc�A�*

loss��<��l�       �	�AN�Yc�A�*

lossh�=�       �	`!O�Yc�A�*

loss��=D�3\       �	t�O�Yc�A�*

loss�=���       �	�MP�Yc�A�*

lossS��==��       �	�TQ�Yc�A�*

lossR n<��'�       �	M�Q�Yc�A�*

loss���=�a�       �	ǜR�Yc�A�*

loss�!L=\Y��       �	�:S�Yc�A�*

loss���=�5�       �	M�S�Yc�A�*

loss]`=+�"       �	pT�Yc�A�*

loss���=g��       �	S	U�Yc�A�*

loss�Ȏ=��       �	ƤU�Yc�A�*

loss40�<6��u       �	m<V�Yc�A�*

loss�C�=��4Z       �	e�V�Yc�A�*

loss��=8��       �	V�W�Yc�A�*

loss��=�h{�       �	�X�Yc�A�*

loss�տ=_%       �	��X�Yc�A�*

loss��>���6       �	�KY�Yc�A�*

loss1�<�Wc�       �	��Y�Yc�A�*

loss�JQ<���       �	��Z�Yc�A�*

lossO�=�E�)       �	Jb[�Yc�A�*

loss��=�j��       �	*9\�Yc�A�*

lossf�A=�2       �	��\�Yc�A�*

loss6��=~ث�       �	@^�Yc�A�*

loss��=*u�       �	�_�Yc�A�*

loss��=\�1�       �	��_�Yc�A�*

loss["=����       �	h^`�Yc�A�*

loss��=�z��       �	�La�Yc�A�*

loss�=�&¨       �	�Zb�Yc�A�*

loss�>V&�       �	'c�Yc�A�*

loss)�2=R���       �	|�c�Yc�A�*

lossL�k=6�l       �	K�d�Yc�A�*

loss���=�r�       �	{�e�Yc�A�*

loss.��<XĖ       �	��f�Yc�A�*

loss��<�G5       �	OZg�Yc�A�*

loss}�c=E6&^       �	V�g�Yc�A�*

loss�� =:YOV       �	�h�Yc�A�*

lossm�=$�.�       �	vOi�Yc�A�*

lossv�x=�       �	0j�Yc�A�*

loss��<c��$       �	�j�Yc�A�*

loss {�<��-       �	uxk�Yc�A�*

loss��=8%�c       �	<l�Yc�A�*

loss�8�=FN]       �	(�l�Yc�A�*

lossX�>i��:       �	XXm�Yc�A�*

loss]�=;}�       �	��m�Yc�A�*

lossA�=9gx       �	Ún�Yc�A�*

loss�k�<���       �	0o�Yc�A�*

loss�̛=k'��       �	u�o�Yc�A�*

loss�u�=[�        �	F_p�Yc�A�*

loss=���;       �	��p�Yc�A�*

loss �^=)m�8       �	��q�Yc�A�*

lossD$$=�S�7       �	2r�Yc�A�*

loss��=d��       �	��r�Yc�A�*

loss��o<]��"       �	�rs�Yc�A�*

loss��	<8`4       �	�Jt�Yc�A�*

lossX��=?�V       �	��t�Yc�A�*

loss��3>g���       �	�u�Yc�A�*

loss)�b=���       �	!v�Yc�A�*

lossd\�=��4I       �	�v�Yc�A�*

lossW�B=l��       �	*Rw�Yc�A�*

loss)�=Ɨo2       �	�w�Yc�A�*

loss��#=M�d�       �	C�x�Yc�A�*

loss�:=U0�h       �	t'y�Yc�A�*

lossc�=s��       �	w�y�Yc�A�*

lossc�]<|2`J       �	�cz�Yc�A�*

loss��-=f��       �	��{�Yc�A�*

loss��<-�7�       �	�3|�Yc�A�*

loss,@�;_�=       �	��|�Yc�A�*

loss��A=��SX       �	dw}�Yc�A�*

loss��m<҂�       �	�~�Yc�A�*

loss��=Z}       �	��~�Yc�A�*

loss��<F�I       �	If�Yc�A�*

loss�=�)?       �	���Yc�A�*

lossƷ�=���       �	^���Yc�A�*

loss:X�=j?X�       �	hA��Yc�A�*

loss�C=ʐ@M       �	�ف�Yc�A�*

loss�R�<��u�       �	 }��Yc�A�*

loss�,<�L�e       �	���Yc�A�*

loss�|�<W_       �	����Yc�A�*

loss�B�<�;�       �	�I��Yc�A�*

loss�'�<cK��       �	ᄘYc�A�*

loss��6>Ã֧       �	�~��Yc�A�*

loss�(�=9�;�       �	H��Yc�A�*

loss-��<���`       �	ެ��Yc�A�*

lossc�=�~OW       �	gF��Yc�A�*

loss��p=G��       �	�އ�Yc�A�*

loss!�=��{I       �	Kr��Yc�A�*

loss�nd=u1�(       �	���Yc�A�*

loss�<4=����       �	����Yc�A�*

loss%��=�N&       �	�>��Yc�A�*

loss�o<��=       �	�Ԋ�Yc�A�*

loss���=8�s�       �	~r��Yc�A�*

loss-�"=����       �	A��Yc�A�*

loss͕>v1֛       �	����Yc�A�*

loss�3�<b[e�       �	�:��Yc�A�*

loss;.�=���       �	�э�Yc�A�*

loss�I=J`f       �	n��Yc�A�*

lossD�<�Z��       �	r��Yc�A�*

lossw�>h���       �	y���Yc�A�*

loss��j<���       �		Q��Yc�A�*

loss���<3ܭ       �	�퐘Yc�A�*

loss��>���-       �	����Yc�A�*

loss[�x=%h0       �	-��Yc�A�*

loss���;�� !       �	{���Yc�A�*

lossl�l<�Tl�       �	�Z��Yc�A�*

loss�$=Q�`       �	P���Yc�A�*

loss1�Q=|�       �	>���Yc�A�*

loss)8<f]E       �	E.��Yc�A�*

loss)�=o��       �	�ϕ�Yc�A�*

loss�IP=�DC       �	yu��Yc�A�*

lossV�=3�/�       �	���Yc�A�*

lossh��<x�>       �	*���Yc�A�*

loss	�C<���       �	oH��Yc�A�*

loss�4�=�e��       �	�䘘Yc�A�*

lossF��<��ܰ       �	����Yc�A�*

loss{===3Ef�       �	m:��Yc�A�*

losstY=7s��       �	fؚ�Yc�A�*

loss.��=��q       �	^���Yc�A�*

loss���=ɍ�       �	�/��Yc�A�*

loss}�=���       �	tԜ�Yc�A�*

loss�Ȃ<ƨ��       �	���Yc�A�*

loss;=W=�퇈       �	�+��Yc�A�*

loss�=�C`       �	=՟�Yc�A�*

losst�=+[^(       �	�r��Yc�A�*

loss�|S;���       �	���Yc�A�*

loss<E�<�\}       �	y���Yc�A�*

loss:U=�       �	�Ţ�Yc�A�*

lossV=��$       �	�c��Yc�A�*

loss3>A=�U�        �	���Yc�A�*

loss�c�=y��c       �	����Yc�A�*

loss���=&��6       �	�@��Yc�A�*

loss}w�=�_�       �	�ߥ�Yc�A�*

loss]P#<��z       �	�z��Yc�A�*

lossHW�=񷜪       �	N(��Yc�A�*

loss�=<8�i        �	ɧ�Yc�A�*

loss�Ĕ=\��       �	o��Yc�A�*

loss۫5<��L�       �	���Yc�A�*

loss���<f=~�       �	(���Yc�A�*

loss�f<(�(�       �	S��Yc�A�*

loss#�x<4��       �	��Yc�A�*

lossHQ�;Aqub       �	���Yc�A�*

loss�Ek=�
{�       �	�&��Yc�A�*

loss�?<�P�       �	����Yc�A�*

lossTq�:���       �	K[��Yc�A�*

loss�
�;C�C;       �	��Yc�A�*

loss�q-=�ڏ�       �	H���Yc�A�*

loss)e=_�Σ       �	�S��Yc�A�*

loss�%=@R!�       �	ﯘYc�A�*

lossO�;���       �	����Yc�A�*

lossc-E=&��       �	�%��Yc�A�*

loss�گ>g��       �	���Yc�A�*

loss��<4~=�       �	�`��Yc�A�*

loss#(�=�       �	�%��Yc�A�*

loss\#Z=`	0        �	 ѳ�Yc�A�*

loss
�=��U�       �	�p��Yc�A�*

loss���<qH��       �	|��Yc�A�*

loss�Q=���       �	⮵�Yc�A�*

lossQ�>�) 2       �	RI��Yc�A�*

loss�!�=��       �	&㶘Yc�A�*

loss���<��n       �	�|��Yc�A�*

loss�ٖ=N�P       �	 ��Yc�A�*

lossf�K<
�C       �	4���Yc�A�*

lossf��=�)��       �	�_��Yc�A�*

loss���=l�T       �	m ��Yc�A�*

loss�ѝ=S�8�       �	����Yc�A�*

loss���=��f       �	�@��Yc�A�*

loss�@�=�ux       �	8ܻ�Yc�A�*

lossf,=�:e�       �	Z~��Yc�A�*

loss��)=�}B       �	MH��Yc�A�*

loss{�=����       �	Sʾ�Yc�A�*

loss}߬<X�޵       �	���Yc�A�*

lossJ�P=mG{       �	�=��Yc�A�*

loss��	=J�t�       �	<���Yc�A�*

lossJ9=R`��       �	���Yc�A�*

loss ��<��l       �	�Yc�A�*

loss��C<��$       �	9�Yc�A�*

loss]�,<�\=�       �	�UØYc�A�*

loss���=`��^       �	��ØYc�A�*

loss{B=A�g       �	��ĘYc�A�*

loss���=����       �	�2ŘYc�A�*

loss�"�=���       �	��ŘYc�A�*

loss�H'=1��       �	YlƘYc�A�*

lossK>	=hF�       �	MǘYc�A�*

loss\�=�W�       �	!�ǘYc�A�*

loss���;q^Z�       �	iVȘYc�A�*

loss�<�s�	       �	��ȘYc�A�*

losse��<䊬�       �	��ɘYc�A�*

loss*�<�,�       �	76ʘYc�A�*

loss�]=rN�       �	[�ʘYc�A�*

lossV�=k߾_       �	�n˘Yc�A�*

loss�n=��       �	�̘Yc�A�*

lossd�k<_ϐ�       �	`�̘Yc�A�*

loss!=8�ê       �	�M͘Yc�A�*

loss�!=����       �	jΘYc�A�*

lossVǽ<;�.�       �	��ΘYc�A�*

loss�U=t&^�       �	�[ϘYc�A�*

loss�G�=N�$�       �	b�ϘYc�A�*

loss2��=#/�       �	��ИYc�A�*

lossŸA<��        �	�.јYc�A�*

loss��q=}p�A       �	��јYc�A�*

loss���;Z�&       �	��ҘYc�A�*

lossH7l<����       �	 ӘYc�A�*

loss�)/=���       �	�2�Yc�A�*

loss��U=8#w�       �	���Yc�A�*

loss�>��y       �	���Yc�A�*

loss&v=�w��       �	>�Yc�A�*

loss�`m=�Z�d       �	5��Yc�A�*

loss�'=a6��       �	�j�Yc�A�*

loss�I�=��M        �	�Yc�A�*

lossl��=��       �	9��Yc�A�*

loss��=�S��       �	{��Yc�A�*

lossW��<�l�       �	�6��Yc�A�*

loss�2�<��[       �	���Yc�A�*

loss��,=�`�       �	;���Yc�A�*

lossjW�=�u�       �	�)��Yc�A�*

loss��{=}��_       �	a���Yc�A�*

loss,A�<����       �	�`��Yc�A�*

lossj�=���       �	Tn��Yc�A�*

loss�d�<�^�       �	���Yc�A�*

lossX2=���}       �	˜��Yc�A�*

lossĹ=�+�@       �	r1��Yc�A�*

loss���=�rq       �	���Yc�A�*

loss��<�m��       �	ro��Yc�A�*

lossTp=��,*       �	<��Yc�A�*

lossl/�<O�k?       �	7���Yc�A�*

loss��=���       �	8h��Yc�A�*

loss��C= ��%       �	W��Yc�A�*

loss@�P<���D       �	���Yc�A�*

loss��<�˰       �	)@��Yc�A�*

lossl?=u�t       �	����Yc�A�*

loss��=I1�       �	Y� �Yc�A�*

loss �=�I��       �	l]�Yc�A�*

loss|LE<z�U�       �	I��Yc�A�*

loss��<��f       �	a��Yc�A�*

loss�:�<YY��       �	.�Yc�A�*

loss4X�=�       �	L��Yc�A�*

lossn�y=8�%       �	-^�Yc�A�*

loss���=g�       �	���Yc�A�*

loss�#�<G%��       �	)��Yc�A�*

loss�2A=����       �	�_�Yc�A�*

loss=>�.�       �	r��Yc�A�*

loss,��=��y       �	��Yc�A�*

lossw��<��|@       �	7�Yc�A�*

loss�n�<hN4<       �	���Yc�A�*

loss���<F�t       �	�m	�Yc�A�*

lossM�e=��$�       �	�	
�Yc�A�*

loss�Ś=�_�       �	��
�Yc�A�*

loss�l>���       �	�:�Yc�A�*

lossm�<�}�G       �	��Yc�A�*

lossfa�<����       �	�n�Yc�A�*

losst.G<��       �	p�Yc�A�*

loss���;x ɿ       �	4��Yc�A�*

loss��=淪d       �	�;�Yc�A�*

loss3;�=�la�       �	���Yc�A�*

lossQ>k=���B       �	]o�Yc�A�*

loss���=憭       �	�'�Yc�A�*

loss,��<���       �	H��Yc�A�*

loss/!�;X��       �	f�Yc�A�*

loss�#�;�yH�       �	��Yc�A�*

loss�-><1���       �	��Yc�A�*

loss�Z"=���       �	8�Yc�A�*

loss��<�ܿ       �	[��Yc�A�*

losslK�<!�"�       �	�p�Yc�A�*

loss���<��       �	�Yc�A�*

loss�*<</F-�       �	���Yc�A�*

loss�]T<B�oF       �	�9�Yc�A�*

lossO�B<PO�{       �	$��Yc�A�*

lossQ�2=��^�       �	g|�Yc�A�*

lossF�=�H��       �	6?�Yc�A�*

loss�Ke=_)V�       �	���Yc�A�*

loss�N�=�T�       �	�s�Yc�A�*

lossW�=���       �	���Yc�A�*

loss �l=�Y{       �	d�Yc�A�*

loss�{=N+�       �	8��Yc�A�*

loss`�"=e��n       �	mY�Yc�A�*

loss��6=`�<       �	��Yc�A�*

lossv-=#�te       �	���Yc�A�*

lossl:5=���l       �	v8�Yc�A�*

loss(u�<q�X�       �	���Yc�A�*

loss]<�=��8-       �	�{�Yc�A�*

loss6 <���k       �	 �Yc�A�*

lossxO�<W�ں       �	}� �Yc�A�*

loss[ڣ=���       �	�M!�Yc�A�*

loss�	K=���$       �	�!�Yc�A�*

loss��=�JF       �	N�"�Yc�A�*

loss*�;�LT       �	fi#�Yc�A�*

loss.==*��       �	W$�Yc�A�*

lossE�<���       �	�$�Yc�A�*

loss��=��h       �	;%�Yc�A�*

loss��\=;���       �	+�%�Yc�A�*

loss��H=?��Z       �	�p&�Yc�A�*

loss�i=X�/�       �	w'�Yc�A�*

losslɼ=cN�m       �	j�'�Yc�A�*

loss��<Ζ��       �	 F(�Yc�A�*

loss:�<�;~       �	S�(�Yc�A�*

loss
=�E	�       �	߈)�Yc�A�*

loss�tu=:!�       �	5*�Yc�A�*

lossF�m<���       �	��*�Yc�A�*

lossȘ.=�){�       �	��+�Yc�A�*

loss��=��TA       �	#.,�Yc�A�*

loss[+�=���,       �	V�,�Yc�A�*

losse��<��H�       �	�o-�Yc�A�*

loss,�<���Z       �	#.�Yc�A�*

lossc<A��'       �	��.�Yc�A�*

lossZh�=,�1       �	�A/�Yc�A�*

loss���<���       �	^�/�Yc�A�*

loss��;=o       �	�x0�Yc�A�*

loss�_�=w�}�       �	71�Yc�A�*

lossQ� <�c��       �	��1�Yc�A�*

loss��<�5�       �	�Q2�Yc�A�*

loss	�=��;�       �	��2�Yc�A�*

loss��<�7�       �	�3�Yc�A�*

loss��
>!�DZ       �	i4�Yc�A�*

loss �G=hw��       �	;�4�Yc�A�*

loss;��<�?       �	l�5�Yc�A�*

lossѺ<�6�       �	n26�Yc�A�*

loss��#=�6       �	��6�Yc�A�*

loss�=�&r       �	ף7�Yc�A�*

loss3�J=^&}�       �	^�8�Yc�A�*

loss�}z=#a�       �	g�9�Yc�A�*

loss�P=�~O�       �	!:�Yc�A�*

loss���<�ɹ       �	��:�Yc�A�*

lossʡ#=�(>[       �	2t;�Yc�A�*

loss3+�=X�       �	<�Yc�A�*

lossSP�=":�L       �	/�<�Yc�A�*

loss��=_Ow�       �	=c=�Yc�A�*

lossN�~=�       �	�
>�Yc�A�*

lossF�x=�w*�       �	��>�Yc�A�*

loss� =�> �       �	�^?�Yc�A�*

loss��=�c'f       �	.@�Yc�A�*

loss��=�V�-       �	�@�Yc�A�*

loss2�Z=��       �	FA�Yc�A�*

lossa�=�.��       �	��A�Yc�A�*

lossj�|=��֢       �	`�B�Yc�A�*

loss@��<Y��       �	BxC�Yc�A�*

loss��<��[�       �	ED�Yc�A�*

loss)c�=��@       �	[�D�Yc�A�*

loss�r(=��E�       �	�E�Yc�A�*

loss�k=�A��       �	�;F�Yc�A�*

loss=}��v       �	��F�Yc�A�*

lossaɈ<�U4Q       �	;oG�Yc�A�*

lossl�<�_��       �	)H�Yc�A�*

lossӶ�;�/f#       �	�H�Yc�A�*

loss��=Gn��       �	pDI�Yc�A�*

lossϵ�=f��	       �	&�I�Yc�A�*

loss�\=��A�       �	�yJ�Yc�A�*

loss_�=����       �	9K�Yc�A�*

loss�<S��       �	X�K�Yc�A�*

loss�0�<:.�       �	bHL�Yc�A�*

loss�mY<yNI@       �	/�L�Yc�A�*

loss:�=�K,       �	x�M�Yc�A�*

loss�<Q=u0��       �	�!N�Yc�A�*

loss��#<M�|�       �	w�N�Yc�A�*

lossڮy=UUO       �	aPO�Yc�A�*

loss�o=�f�       �	��O�Yc�A�*

loss.�=vä`       �	M�P�Yc�A�*

loss%�k=��h       �	Q�Yc�A�*

loss�J�;����       �	�Q�Yc�A�*

loss�P�<ca�       �	�PR�Yc�A�*

lossR̍=�t       �	�R�Yc�A�*

loss�u=@i�       �	#�S�Yc�A�*

loss)��<�7�       �	�T�Yc�A�*

lossq��= p�       �	��T�Yc�A�*

lossO��<���       �	�OU�Yc�A�*

loss�t=����       �	��U�Yc�A�*

loss?��;-�f        �	�|V�Yc�A�*

loss�=�34       �	W�Yc�A�*

loss��=K�       �	ͰW�Yc�A�*

lossfA�;���       �	5FX�Yc�A�*

loss���=��       �	��X�Yc�A�*

loss�1={�md       �	r�Y�Yc�A�*

loss*�<��M       �	�#Z�Yc�A�*

loss7=��       �	˹Z�Yc�A�*

lossrL�;����       �	3S[�Yc�A�*

loss�`Z<	7�       �	��[�Yc�A�*

lossLF�=:�;       �	��\�Yc�A�*

lossnM]<Jm��       �	((]�Yc�A�*

loss5�=����       �	��]�Yc�A�*

lossV>����       �	 z^�Yc�A�*

loss���=Oǻ�       �	�A_�Yc�A�*

lossV�=����       �	��_�Yc�A�*

loss��=.��       �	�`�Yc�A�*

loss���<�M�L       �	wLa�Yc�A�*

lossl>c���       �	��a�Yc�A�*

lossax5=�X�       �	z�b�Yc�A�*

loss�"�;�
�/       �	R,c�Yc�A�*

loss�I�<��       �	��c�Yc�A�*

loss��=t��k       �	�d�Yc�A�*

lossX�,=�.�       �	ne�Yc�A�*

loss_Gn<�u        �	��e�Yc�A�*

loss�>�=h���       �	#Kf�Yc�A�*

loss3�\=��       �	q�f�Yc�A�*

loss\J�=B��       �	�g�Yc�A�*

loss?�=3.�       �	�h�Yc�A�*

loss�*{<N��       �	�*i�Yc�A�*

lossm{=��ֱ       �	��i�Yc�A�*

loss�c�;��       �	[]j�Yc�A�*

loss,�M=>s/?       �	��j�Yc�A�*

loss=jS�       �	O�k�Yc�A�*

losss�9=��*       �	�7l�Yc�A�*

loss�r�=�	��       �	��l�Yc�A�*

lossTt�<A\�U       �	��m�Yc�A�*

loss!61=~)n"       �	*n�Yc�A�*

loss\f<�Ѽ.       �	��n�Yc�A�*

loss6�a=��l�       �	^ho�Yc�A�*

lossg!=#w��       �	p�Yc�A�*

lossP�=�bV       �	��p�Yc�A�*

lossQkL=<�v�       �	�Eq�Yc�A�*

loss���=��       �	��q�Yc�A�*

loss���<�'{       �	��r�Yc�A�*

loss�A�=#�5       �	�0s�Yc�A�*

loss���<��[�       �	��s�Yc�A�*

loss��<�ű       �	�it�Yc�A�*

loss\�>���.       �	�%u�Yc�A�*

loss6�x<���       �	}�u�Yc�A�*

loss
'=ٔ�>       �	�jv�Yc�A�*

losswRA=+d�       �	�w�Yc�A�*

loss�M�=l��z       �	��w�Yc�A�*

lossrqQ=QSN�       �	�x�Yc�A�*

lossfB�=��Q�       �	�y�Yc�A�*

loss�>.z	       �	J�z�Yc�A�*

loss��=[ߙ"       �	{k{�Yc�A�*

loss��=;'�       �	H|�Yc�A�*

loss_Cf<	?��       �	��|�Yc�A�*

loss w�<�W:�       �	t}�Yc�A�*

lossz2�=)Ng       �	~�Yc�A�*

loss�0�<��"       �	��~�Yc�A�*

loss6��<�njz       �	=e�Yc�A�*

loss<ou=B�@�       �	�g��Yc�A�*

lossR)x=K�M       �	��Yc�A�*

loss[T�<1`	@       �	����Yc�A�*

loss�>	=���       �	����Yc�A�*

loss�yX<u� u       �	e7��Yc�A�*

loss�<D       �	�ك�Yc�A�*

loss��`=wm�       �	�o��Yc�A�*

lossh�<v���       �	vn��Yc�A�*

loss���=�c       �	 	��Yc�A�*

loss��
>�,�       �	����Yc�A�*

loss3��<�t�       �	A��Yc�A�*

loss��=B<�H       �	؇�Yc�A�*

loss$g�=���7       �	r��Yc�A�*

lossm�=�b�<       �	|��Yc�A�*

loss�Z1=ڈ       �	秉�Yc�A�*

loss��=a,��       �	���Yc�A�*

loss/�*='%�       �	%��Yc�A�*

loss�s!=�y�       �	l̋�Yc�A�*

loss�&�=�Y       �	�r��Yc�A�*

loss��R=�]s	       �	a��Yc�A�*

loss�h�<{��/       �	��Yc�A�*

loss�E�=?�       �	5]��Yc�A�*

loss��o=��K�       �	���Yc�A�*

lossɕ�<���y       �	ݏ�Yc�A�*

loss���<4(��       �	�|��Yc�A�*

lossV��=�g�       �	\��Yc�A�*

losss~_=H$        �	鸑�Yc�A�*

loss��=�P�       �	�\��Yc�A�*

loss�5�;�t��       �	����Yc�A�*

loss�*�<��	       �	?���Yc�A�*

lossh�=�M�~       �	�'��Yc�A�*

lossSD�<	s��       �	7���Yc�A�*

loss��<F��       �	8f��Yc�A�*

loss	�=�5�       �	Z.��Yc�A�*

lossD.�=Uk��       �	�Ж�Yc�A�*

loss��!=z�]�       �	~n��Yc�A�*

loss��=�΄       �	���Yc�A�*

loss���<s��Z       �	
���Yc�A�*

loss���=5	|r       �	�7��Yc�A�*

lossw�<��       �	�Й�Yc�A�*

loss�$�=����       �	Ql��Yc�A�*

lossm!=��c�       �	���Yc�A�*

loss�2�=m�!�       �	����Yc�A�*

loss,f�=�7�f       �	Φ��Yc�A�*

loss��<��`F       �	D��Yc�A�*

loss���<X;�N       �	 䝙Yc�A�*

lossڠJ=��SA       �	㍞�Yc�A�*

loss�=����       �	�o��Yc�A�*

loss�s�==O�       �	���Yc�A�*

loss�;�=���4       �	
M��Yc�A�*

lossz)=����       �	T㡙Yc�A�*

loss�.=�#�       �	dɢ�Yc�A�*

loss��d=��&�       �	ݣ�Yc�A�*

loss�=ܭ�       �	�|��Yc�A�*

lossW�'=R@,E       �	�0��Yc�A�*

loss�	�<2:9/       �	�ץ�Yc�A�*

loss��=�`�G       �	|{��Yc�A�*

lossJ)<w�+[       �	�?��Yc�A�*

loss��=�I/S       �	'ا�Yc�A�*

losslF�=�'�_       �	����Yc�A�*

loss�X+=��S�       �	���Yc�A�*

loss��=k���       �	@��Yc�A�*

loss�@==g��       �	�	��Yc�A�*

lossv�><5�ӄ       �	���Yc�A�*

loss��y<LK]B       �	ZI��Yc�A�*

loss�Y=��       �	&��Yc�A�*

loss���;���       �	����Yc�A�*

loss��#<���       �	Y���Yc�A�*

loss)Α=>(;       �	cA��Yc�A�*

lossj��=�B�       �	�گ�Yc�A�*

loss_�=g2To       �	�y��Yc�A�*

lossT1�=gG;�       �	�q��Yc�A�*

loss-�=���       �	F��Yc�A�*

loss_��=\�t�       �	����Yc�A�*

lossHb=0��*       �	xE��Yc�A�*

loss8��<���J       �	�޳�Yc�A�*

loss�#�<!A�0       �	�z��Yc�A�*

loss&Μ=��M       �	8��Yc�A�*

lossKc=+�m       �	����Yc�A�*

loss�n�=�i'U       �	%X��Yc�A�*

lossR�<?�q       �	����Yc�A�*

loss���=�g�       �	�ܷ�Yc�A�*

loss_g�<	��       �	�y��Yc�A�*

loss�2=x�m�       �	2��Yc�A�*

loss3e=x�5�       �	s���Yc�A�*

loss��,=O�a       �	����Yc�A�*

loss��=0?'%       �	����Yc�A�*

loss��<i��       �	����Yc�A�*

loss�+<�ס�       �	-_��Yc�A�*

loss!l3=�[��       �	���Yc�A�*

loss�b�<�ea�       �	ũ��Yc�A�*

loss
��<�#0       �	,D��Yc�A�*

loss%=�Y�=       �	7߿�Yc�A�*

loss�*E=���       �	���Yc�A�*

loss���<`q9       �	}��Yc�A�*

loss��>y�/        �	����Yc�A�*

lossi�=��Y       �	oeYc�A�*

lossx<[�Z       �	R,ÙYc�A�*

loss�D=[��/       �	*�ÙYc�A�*

loss�/�=�G��       �	{gęYc�A�*

loss ğ<T��s       �	%řYc�A�*

loss�:n<hmCO       �	1�řYc�A�*

lossC�<q}��       �	�1ƙYc�A�*

loss���="��       �	O�ƙYc�A�*

loss<��<3xO�       �	nhǙYc�A�*

loss�9=]�Ƀ       �	�șYc�A�*

loss�Ԟ<E��       �	��șYc�A�*

loss���<n ��       �	_BəYc�A�*

loss�@�=��۫       �	n�əYc�A�*

loss�"=K���       �	�xʙYc�A�*

loss'"�<	���       �	�˙Yc�A�*

loss���=���       �	#�˙Yc�A�*

loss&W�=0���       �	�y̙Yc�A�*

loss3I�;\��       �	�͙Yc�A�*

loss܇�<�	       �		�͙Yc�A�*

loss�]9=�ܘ�       �	obΙYc�A�*

lossG�=����       �	ϙYc�A�*

loss��=�P       �	ҫϙYc�A�*

loss�/>P��3       �	�LЙYc�A�*

loss�L�=�1��       �	��ЙYc�A�*

lossq�=9��N       �	�љYc�A�*

loss	�?=�ɒ       �	aҙYc�A�*

loss!��<�u�E       �	)�ҙYc�A�*

loss͐�=s3e�       �	a�әYc�A�*

lossU
=�gz       �	W?ԙYc�A�*

loss:�n=�P��       �	��ԙYc�A�*

loss8G�<1F9       �	��ՙYc�A�*

lossվ�<!�&C       �	Y֙Yc�A�*

lossr��=�/g       �	�֙Yc�A�*

loss(8�<3�:_       �	xEיYc�A�*

loss@�<�?�k       �	W�יYc�A�*

loss��<ģ�       �	-�ؙYc�A�*

loss]!=@D�       �	�YٙYc�A�*

loss?�=f��       �	��ٙYc�A�*

loss��=�7�       �		�ڙYc�A�*

loss��Y=:��G       �	!!ۙYc�A�*

lossZ:�<=��       �	��ۙYc�A�*

loss;�=P�T�       �	0KܙYc�A�*

loss�ժ=�E�       �	v�ܙYc�A�*

loss�W�=�;�B       �	�vݙYc�A�*

lossb��=z�~       �	TޙYc�A�*

lossň(=�       �	ݴޙYc�A�*

loss��<���E       �	-|��Yc�A�*

loss�`=_�).       �	q�Yc�A�*

loss�b=;�tr       �	w��Yc�A�*

loss�Y<��       �	\�Yc�A�*

lossсF=v�Y�       �	�Yc�A�*

loss�=��5       �	���Yc�A�*

loss�<��+�       �	=�Yc�A�*

loss��<��       �	�'�Yc�A�*

lossi��=9�QB       �	���Yc�A�*

loss��c=�oΰ       �	�g�Yc�A�*

loss���<�ɤa       �	J	�Yc�A�*

losss�=xz��       �	���Yc�A�*

loss/͒=}j�       �	�G�Yc�A�*

loss,��<����       �	���Yc�A�*

lossS��=�ϫ�       �	���Yc�A�*

loss�7<��R�       �	2�Yc�A�*

lossz��<Z�       �	Ǹ�Yc�A�*

loss�L�<��Ra       �	"Q�Yc�A�*

loss�`�=���       �	W��Yc�A�*

lossEǷ=0`ɔ       �	=��Yc�A�*

loss�^�=�51$       �	��Yc�A�*

loss��<\rw       �	��Yc�A�*

loss�z�< ���       �	�T�Yc�A�*

loss_��<s�k�       �	���Yc�A�*

loss�b�<~I�T       �	���Yc�A�*

losst�l=�@�       �	�.�Yc�A�*

loss�_=K�#       �	��Yc�A�*

loss8�*=�.t�       �	�k�Yc�A�*

loss;A�<5=E�       �	��Yc�A�*

loss��n=k�+       �	���Yc�A�*

loss�;�<�c-w       �	GZ�Yc�A�*

loss�4�<��M       �	_��Yc�A�*

loss���;a��U       �	0���Yc�A�*

lossR<�=:       �	(��Yc�A�*

loss=��=!O�|       �	���Yc�A�*

loss� <KoJ0       �	`X��Yc�A�*

loss�z[<ڒ$�       �	����Yc�A�*

loss7��<@���       �	����Yc�A�*

loss��=U8�~       �	����Yc�A�*

loss�`=q"�v       �	/M��Yc�A�*

loss���<�h��       �	���Yc�A�*

loss�T�=1H�E       �	2���Yc�A�*

loss��U=!V��       �	�A��Yc�A�*

loss�jY=����       �	����Yc�A�*

loss ��<��       �	ő��Yc�A�*

loss���=�O��       �	!���Yc�A�*

loss�b=Z�E       �	�6��Yc�A�*

loss�=?N[�       �	Y���Yc�A�*

loss�zz<>���       �	&� �Yc�A�*

loss�m2=[AB       �	bN�Yc�A�*

loss�5�=X��V       �	F��Yc�A�*

loss�F>���;       �	e��Yc�A�*

lossE��<<Ǧ�       �	ٖ�Yc�A�*

loss��h=��G�       �	@1�Yc�A�*

loss*�!=D2�       �	y��Yc�A�*

loss���=�c��       �	�i�Yc�A�*

losswS�=	���       �	u!�Yc�A�*

loss��>����       �	��Yc�A�*

loss�D =��       �	�k�Yc�A�*

loss���=J���       �	�Yc�A�*

lossl�=��}       �	D��Yc�A�*

loss_3=xO       �	Y	�Yc�A�*

loss)sI=�ή�       �	�	�Yc�A�*

loss&�H=��       �	W�
�Yc�A�*

loss�}�<��|x       �	�+�Yc�A�*

loss��:=��a�       �	���Yc�A�*

loss���<���       �	�[�Yc�A�*

loss�Q�< �L�       �	�2�Yc�A�*

lossz9�<��C�       �	���Yc�A�*

losss�=Q�a       �	ni�Yc�A�*

lossqi�<t�       �	.�Yc�A�*

losso��<�&z       �	���Yc�A�*

loss=am=���~       �	J�Yc�A�*

loss�+�=˜�       �	�	�Yc�A�*

lossx��<a�c       �	���Yc�A�*

loss�=zcG@       �	�L�Yc�A�*

loss�8�<S?Q�       �	g��Yc�A�*

loss�;1=��7       �	���Yc�A�*

lossa��=x�ZK       �	��Yc�A�*

loss;ӭ<�:�       �	���Yc�A�*

loss�A�<W,�       �	�V�Yc�A�*

loss�=�^'�       �	#��Yc�A�*

loss�@=g��8       �	 ��Yc�A�*

lossò�<�7ye       �	�9�Yc�A�*

lossF7�;����       �	���Yc�A�*

loss&��=<`
H       �	5}�Yc�A�*

lossoX>��	�       �	.�Yc�A�*

loss��U=
ސ"       �	��Yc�A�*

lossS̘=�ZBX       �	^K�Yc�A�*

lossx��<��^8       �	���Yc�A�*

loss��<�Az       �	�x�Yc�A�*

loss���=ɷ~       �	��Yc�A�*

lossH =K蔅       �	j��Yc�A�*

loss�O�<ޤ        �	oF�Yc�A�*

loss��a<
I J       �	���Yc�A�*

losse	|=����       �	��Yc�A�*

loss���<����       �	G�Yc�A�*

lossȲ2<� ^X       �	���Yc�A�*

lossi�;<+�Ս       �	� �Yc�A�*

loss�=���6       �	w�!�Yc�A�*

loss,�m<���       �	(a"�Yc�A�*

loss�<2�       �	�T#�Yc�A�*

loss,,�=�NY       �	�$�Yc�A�*

loss��<Q7�       �	C�$�Yc�A�*

loss!�<�P�A       �	��%�Yc�A�*

loss��='�       �	\�&�Yc�A�*

loss��Y<�a��       �	�2'�Yc�A�*

loss�٫<��f       �	�'�Yc�A�*

lossj��<H��       �	o�(�Yc�A�*

loss��<k��_       �	al)�Yc�A�*

loss6�3<0�n,       �	*�Yc�A�*

lossnf#>`�`�       �	@�*�Yc�A�*

lossJ��=Kz��       �	�;+�Yc�A�*

loss�M=Yy�       �	��+�Yc�A�*

loss!�<�D�       �	�p,�Yc�A�*

lossQ>v�Ŝ       �	�%-�Yc�A�*

loss�'=�Rp       �	��-�Yc�A�*

lossdO�<ّ��       �	nm.�Yc�A�*

loss=(�<��XX       �	�/�Yc�A�*

loss�Bo=Z_       �	d�/�Yc�A�*

loss�P<��X�       �	QJ0�Yc�A�*

lossğ=��5       �	/1�Yc�A�*

lossk
=����       �	��1�Yc�A�*

loss� }=ph�       �	?S2�Yc�A�*

loss�H#<}P�       �	��2�Yc�A�*

loss�)<�1�Y       �	�3�Yc�A�*

lossg�<�u�I       �	�4�Yc�A�*

loss·�<f�V       �	>�4�Yc�A�*

lossI�=�L�(       �	?�5�Yc�A�*

lossmì<k��       �	�16�Yc�A�*

loss��{<AG��       �	��6�Yc�A�*

loss��P=h�2       �	�c7�Yc�A�*

loss|�=S�Q�       �	��7�Yc�A�*

loss�+/<�ǐ       �	�8�Yc�A�*

lossF�g<�
�       �	�=9�Yc�A�*

loss�_�<m�i�       �	::�Yc�A�*

lossX�=A�T�       �	��:�Yc�A�*

lossZ<O+��       �	
M;�Yc�A�*

lossڪ>ٞ7j       �	q�;�Yc�A�*

lossx�w<�OL       �	D�<�Yc�A�*

loss�E�=��1       �	6!=�Yc�A�*

loss��= R�       �	��=�Yc�A�*

loss��=<h��       �	yW>�Yc�A�*

loss
`=�TK�       �	r�>�Yc�A�*

loss��]<+o]A       �	��?�Yc�A�*

lossvOC=�4 ~       �	�.@�Yc�A�*

loss�LF=��^       �	��@�Yc�A�*

lossS7= �       �	�\A�Yc�A�*

loss{�G<�5��       �	��A�Yc�A�*

loss&�9=���       �	�B�Yc�A�*

loss:�<��'�       �	�+C�Yc�A�*

loss�]�=��Ĭ       �	��C�Yc�A�*

lossIt�<]L�       �	�pD�Yc�A�*

lossWi�=��N       �	�E�Yc�A�*

loss�;�;"�       �	O�E�Yc�A�*

loss+L=0��l       �	�JF�Yc�A�*

loss8&�=M&�       �	��F�Yc�A�*

loss�}J='=��       �	�G�Yc�A�*

lossZj�<1V=�       �	�H�Yc�A�*

loss�@=k�.0       �	��H�Yc�A�*

loss��=6���       �	�RI�Yc�A�*

lossܾ;QF�       �	(�I�Yc�A�*

loss: R<��|`       �	�J�Yc�A�*

loss��=CI�       �	6K�Yc�A�*

lossk�<.       �	��K�Yc�A�*

lossq��;�uW�       �	�oL�Yc�A�*

lossm6+<U k�       �	;M�Yc�A�*

loss僨;���h       �	��M�Yc�A�*

loss�<Y��       �	?�N�Yc�A�*

loss��<�*]7       �	/4O�Yc�A�*

loss���;�?�       �	�O�Yc�A�*

loss� <j���       �	�xP�Yc�A�*

loss6P<~�M=       �	�Q�Yc�A�*

loss5�;�vn       �	
�Q�Yc�A�*

loss��:m��       �	mWR�Yc�A�*

loss2��<��       �	E�R�Yc�A�*

loss�Np=�QP       �	h�S�Yc�A�*

lossl>=Y/Q�       �	�4T�Yc�A�*

loss�|�:����       �	��T�Yc�A�*

loss1��<�9�       �	"RV�Yc�A�*

lossF#>Ќ?       �	��V�Yc�A�*

loss�&<�6��       �	�W�Yc�A�*

loss���;��;�       �	�!X�Yc�A�*

loss��.=
؇"       �	s*Y�Yc�A�*

lossȸ=��       �	��Y�Yc�A�*

lossv�<Kty�       �	�vZ�Yc�A�*

loss}7
<��t       �	�[�Yc�A�*

loss,�P=ן�       �	v�[�Yc�A�*

lossxj=��K�       �	_B\�Yc�A�*

loss4]�=��V�       �	d�\�Yc�A�*

losst�z=�Ẓ       �	�]�Yc�A�*

loss�6-=}�k�       �	�9^�Yc�A�*

loss{��<v�       �	��^�Yc�A�*

loss��=}bU�       �	/k_�Yc�A�*

lossqL=�
�=       �	�`�Yc�A�*

loss��=��       �	6�`�Yc�A�*

loss=�^��       �	�a�Yc�A�*

losss�J=�b9       �	�b�Yc�A�*

loss�:�<;a_�       �	�Fc�Yc�A�*

loss� =�Hw�       �	ld�Yc�A�*

loss�{K<)��       �	�d�Yc�A�*

loss���;d�&�       �	3f�Yc�A�*

loss׋=F��       �	��f�Yc�A�*

lossɂ�<7���       �	��g�Yc�A�*

loss��;�.z       �	��h�Yc�A�*

loss��<���x       �	�pi�Yc�A�*

lossݞ�<����       �	'j�Yc�A�*

lossi�<��s~       �	�j�Yc�A�*

loss��<[��       �	��k�Yc�A�*

lossH=%��       �	��l�Yc�A�*

loss�Ŀ=����       �	�|m�Yc�A�*

loss6i�<�~�       �	T�n�Yc�A�*

loss�M<��9�       �	A(o�Yc�A�*

loss�S[<�5       �	G�o�Yc�A�*

lossN];.R��       �	<�p�Yc�A�*

loss��<��
       �	�r�Yc�A�*

loss��<���
       �	,Es�Yc�A�*

lossO��<�~��       �	
�s�Yc�A�*

loss�,=�TI       �	��t�Yc�A�*

loss�F=�2�       �	�zu�Yc�A�*

lossOr�<��       �	&v�Yc�A�*

lossE�E<�N�|       �	�@w�Yc�A�*

loss�r<
en�       �	��w�Yc�A�*

loss. 0=6�t<       �	�x�Yc�A�*

losss�!<Gx&       �	�)y�Yc�A�*

loss�=k��       �	��y�Yc�A�*

loss_�==���       �	�az�Yc�A�*

loss�W�<>�@       �	�{�Yc�A�*

lossrº<���       �	q�{�Yc�A�*

loss�=�"�       �	�F|�Yc�A�*

loss�u�;���       �	c�|�Yc�A�*

loss��!=p�K�       �	Y�}�Yc�A�*

loss$�W=E�"�       �	����Yc�A�*

loss���=a2�       �	�X��Yc�A�*

loss��>(��       �	=�Yc�A�*

lossn�I=�ʈc       �	/���Yc�A�*

loss��=t]a       �	�"��Yc�A�*

lossD&n<��a       �	=���Yc�A�*

loss#hv=o֞       �	�P��Yc�A�*

losss�=ƶ5$       �	!闚Yc�A�*

loss��=ZC��       �	0��Yc�A�*

loss��=��p�       �	��Yc�A�*

loss_��<�]��       �	9���Yc�A�*

loss;[J=Kf��       �	�K��Yc�A�*

lossi�=���Z       �	�⚚Yc�A�*

loss���<'a�^       �	+���Yc�A�*

loss�L=�d�       �	J&��Yc�A�*

loss��<�qMO       �	��Yc�A�*

loss���;ٖ�       �	�d��Yc�A�*

lossHQ�<��F�       �	]��Yc�A�*

loss��%<;��Y       �	ޭ��Yc�A�*

lossx!�=���?       �	_D��Yc�A�*

loss��=u�       �	Hޟ�Yc�A�*

loss�=6θ       �	����Yc�A�*

loss)�7="�V�       �	T��Yc�A�*

loss��B=��;�       �	G顚Yc�A�*

loss�1�<�
S       �	�~��Yc�A�*

loss�y=H���       �	b/��Yc�A�*

losso�L=	�       �	���Yc�A�*

lossx"�<ZF       �	1���Yc�A�*

lossh��<= �I       �	�>��Yc�A�*

loss�!�=�թF       �	�ե�Yc�A�*

loss)�N=(�U�       �	n��Yc�A�*

lossj}==�r�M       �	6��Yc�A�*

lossXM�<���=       �	흧�Yc�A�*

lossP�=+��       �	�7��Yc�A�*

loss���<����       �	�ͨ�Yc�A�*

loss͐�==�km       �	�r��Yc�A�*

loss���<|i��       �	���Yc�A�*

lossA��=Ú��       �	1���Yc�A�*

loss�=�Q�-       �	N��Yc�A�*

lossZs�=�Z`       �	����Yc�A�*

loss|�=/���       �	��Yc�A�*

loss��0<���        �	q���Yc�A�*

loss���<�M"z       �	(��Yc�A�*

losse=ٴK       �	���Yc�A�*

loss�D)=�y�       �	İ��Yc�A�*

loss�ך=�	ur       �	
J��Yc�A�*

loss;߿<@rT       �	�ް�Yc�A�*

lossr�G=m��"       �	2��Yc�A�*

loss��c=x��       �	�˲�Yc�A�*

loss�C;��       �	�b��Yc�A�*

loss'<���        �	� ��Yc�A�*

loss :@=k��+       �	����Yc�A�*

lossD�3<.�߅       �	�B��Yc�A�*

loss��&>����       �	�൚Yc�A�*

loss)	=z`A+       �	����Yc�A�*

loss�+'<�       �	:[��Yc�A�*

loss�)�;"ܖ�       �	X���Yc�A�*

loss�?�;Ȓ��       �	����Yc�A�*

loss�U�<��       �	�D��Yc�A�*

loss�_=8�8�       �	� ��Yc�A�*

loss�p=���       �	�꺚Yc�A�*

loss��;���       �	����Yc�A�*

loss�=<��@        �	�-��Yc�A�*

loss��=sW       �	�弚Yc�A�*

losss��<R� R       �	����Yc�A�*

lossq��<�\p       �	,.��Yc�A�*

loss�B=&~r       �	�Ѿ�Yc�A�*

loss��!=�xp�       �	����Yc�A�*

loss��8=_M�s       �	D4��Yc�A�*

loss18D=�c��       �	���Yc�A�*

loss�=��b       �	�g��Yc�A�*

loss�
=۔˒       �	Yc�A�*

loss���=��pV       �	��Yc�A�*

loss$n�<���:       �	�SÚYc�A�*

loss��<U��       �	�ÚYc�A�*

loss@�N<�O��       �	��ĚYc�A�*

loss�<=F��6       �	>ŚYc�A�*

loss!C�<����       �	��ŚYc�A�*

loss�޼<�GaU       �	��ƚYc�A�*

lossO)"=y7��       �	F"ǚYc�A�*

loss_�)=��3�       �	{�ǚYc�A�*

lossh��=�"�       �	tyȚYc�A�*

loss��:=nz\       �	�!ɚYc�A�*

loss8�=�U       �	�ɚYc�A�*

losso�=+�|Q       �	W[ʚYc�A�*

loss��)<�3'�       �	��ʚYc�A�*

lossM��<�E�       �	2�˚Yc�A�*

loss�B=���d       �	�7͚Yc�A�*

loss��=UM��       �	��͚Yc�A�*

loss8�<��       �	��ΚYc�A�*

loss��=a��       �	�qϚYc�A�*

loss%�K=�T       �	LКYc�A�*

loss�&6<�0\N       �	��КYc�A�*

loss:�<�]�       �	�JњYc�A�*

loss6>�=C��       �	?�њYc�A�*

losscR8=�xXU       �	��ҚYc�A�*

lossh�V=�s(       �	iӚYc�A�*

loss�We<���       �	�ԚYc�A�*

loss�p�=4���       �	�,՚Yc�A�*

loss��<�        �	��՚Yc�A�*

lossڄ�< �`K       �	�}֚Yc�A�*

lossF	�<eȰ0       �	�!ךYc�A�*

loss�&<�GS,       �	�ךYc�A�*

loss�چ<��O       �	IfؚYc�A�*

lossI3n<���k       �	�ٚYc�A�*

loss�a>�"}�       �	;�ٚYc�A�*

lossm��<Mu�       �	�JښYc�A�*

loss�<<��
       �	��ښYc�A�*

loss%=Uꇙ       �	��ۚYc�A�*

loss�%r=ʛ҆       �	S%ܚYc�A�*

loss���=�.i�       �	��ܚYc�A�*

loss�P8=ݨ��       �	�aݚYc�A�*

loss��<v���       �	ޚYc�A�*

lossx�<!�.7       �	�LߚYc�A�*

loss{�e=�ٱ       �	[��Yc�A�*

loss�U<<��       �	����Yc�A�*

loss��=M�1       �	MK�Yc�A�*

loss���=��av       �	��Yc�A�*

loss��<̤&�       �	��Yc�A�*

loss��<��@R       �	�,�Yc�A�*

loss}Zg<n��       �	��Yc�A�*

loss�mI=o^��       �	��Yc�A�*

loss���<�       �	�E�Yc�A�*

loss���;I�3�       �	���Yc�A�*

lossX]8<����       �	���Yc�A�*

loss�,�=3֩       �	1'�Yc�A�*

loss��=���j       �	6��Yc�A�*

loss.r<1ό�       �	8e�Yc�A�*

lossX"�=�_�^       �	���Yc�A�*

loss�*�=��N       �	��Yc�A�*

loss@� =���x       �	~9�Yc�A�*

loss��=�j��       �	s��Yc�A�*

loss�[
=�-g       �	Xq�Yc�A�*

loss��<	o��       �	{�Yc�A�*

loss���=%e9�       �	���Yc�A�*

loss���<hS^       �	
J�Yc�A�*

loss�D=l(�       �	u��Yc�A�*

loss�c<�|�       �	D��Yc�A�*

loss��<�H�       �	2�Yc�A�*

loss�<����       �	�/�Yc�A�*

loss��{;�-�       �	���Yc�A�*

lossh6E<`q�6       �	Ov�Yc�A�*

lossJ�=~��       �	"�Yc�A�*

losstu�<?i39       �	0��Yc�A�*

loss8�C=���w       �	\V�Yc�A�*

lossc��<�
d�       �	���Yc�A�*

loss�V#=6�	�       �	����Yc�A�*

lossŒj<�T��       �	5��Yc�A�*

loss�˞;i�z       �	9���Yc�A�*

loss��<ᓞ�       �	x��Yc�A�*

loss`-<�;�       �	���Yc�A�*

loss���=O�+�       �	f���Yc�A�*

lossEx�=�\�Q       �	�X��Yc�A�*

loss�;V=>       �	B��Yc�A�*

loss��~=+�	       �	Q���Yc�A�*

loss
�;�Ѵ       �	����Yc�A�*

loss���=]�       �	J}��Yc�A�*

loss<2�=���       �	7��Yc�A�*

lossͥ�=�F       �	����Yc�A�*

loss��^<�5��       �	Z���Yc�A�*

loss[vI=�~}�       �	�b��Yc�A�*

lossr��<f+��       �	S��Yc�A�*

loss�|�<��_U       �	����Yc�A�*

losse�;�z��       �	�� �Yc�A�*

loss��=���e       �	z5�Yc�A�*

loss��=9&��       �	��Yc�A�*

lossC-<LO�       �	���Yc�A�*

loss�Z�=Gb�        �	$&�Yc�A�*

loss͎=�UW       �	���Yc�A�*

loss��%=����       �	H��Yc�A�*

loss.��<E�/       �	p��Yc�A�*

lossנ;��:�       �	�-�Yc�A�*

loss_&g=A��       �	_��Yc�A�*

loss*[=�*q�       �	3��Yc�A�*

loss͒(=��       �	�	�Yc�A�*

loss,�K=ձ4�       �	(�	�Yc�A�*

loss��=�L��       �	X
�Yc�A�*

loss[�=���x       �	��
�Yc�A�*

loss�s�=�\[^       �	���Yc�A�*

lossV�q=# �       �	�.�Yc�A�*

loss�QG<%�i�       �	��Yc�A�*

lossW%�<�bqO       �	�h�Yc�A�*

loss�<���       �	��Yc�A�*

loss�</<qн       �	��Yc�A�*

losslMc<Z�z       �	i��Yc�A�*

loss�[�;Y�       �	Na�Yc�A�*

loss�U>=����       �	���Yc�A�*

loss���<Q���       �	���Yc�A�*

loss�=��?�       �	�4�Yc�A�*

lossx�=PQ}       �	��Yc�A�*

loss�a�=Ώ�r       �	r��Yc�A�*

loss���<�`       �	�'�Yc�A�*

losshV]<��t       �	 ��Yc�A�*

loss1Og<��       �	-\�Yc�A�*

loss<�;�{��       �	���Yc�A�*

loss���=�?�i       �	���Yc�A�*

loss�=�(�)       �	j.�Yc�A�*

loss�t=��l3       �	*��Yc�A�*

loss��=jc��       �	:X�Yc�A�*

lossM��<��
       �	('�Yc�A�*

lossZ0Y=�;b       �	���Yc�A�*

loss�x�<a��       �	�N�Yc�A�*

loss�s<a�(�       �	Kt�Yc�A�*

lossV��=����       �	B	�Yc�A�*

losseW�<�|�       �	���Yc�A�*

loss$)Z<�Q�       �	�L�Yc�A�*

loss��=N��F       �	�q�Yc�A�*

loss��;AG��       �	_�Yc�A�*

loss�==+�W*       �	���Yc�A�*

loss�[�=4�n�       �	�� �Yc�A�*

loss���<Zk�S       �	,!�Yc�A�*

loss�k�<`!�a       �	��!�Yc�A�*

loss4��<��w       �	*�"�Yc�A�*

loss�^<����       �	�b#�Yc�A�*

loss�1=#{�"       �	��#�Yc�A�*

loss���=�r�       �	��$�Yc�A�*

loss�0{=z�ǥ       �	�&%�Yc�A�*

loss�=Д$�       �	,�%�Yc�A�*

loss 9�=AE%�       �	�&�Yc�A�*

loss�J=�υ       �	�5'�Yc�A�*

loss@�o<��V'       �	��'�Yc�A�*

loss*��<�H6H       �	�k(�Yc�A�*

loss�*�<#�n�       �	�	)�Yc�A�*

loss��=�7t5       �	@�)�Yc�A�*

loss%��=�x�       �	�4*�Yc�A�*

lossl.C=�q�Y       �	��*�Yc�A�*

lossZ��=V��       �	W`+�Yc�A�*

loss(��<��{�       �	c
,�Yc�A�*

loss&1T<waװ       �	֩,�Yc�A�*

loss�>$=e�(       �	�D-�Yc�A�*

loss�K<4�$       �	"�-�Yc�A�*

loss��;mtk}       �	a�.�Yc�A�*

lossd�=��R       �	�"/�Yc�A�*

loss�F=sB|       �	��/�Yc�A�*

lossV�-=��%       �	�^0�Yc�A�*

lossQ�=�)�       �	�1�Yc�A�*

lossܓ�<�9       �	]�1�Yc�A�*

loss�w=�X��       �	A2�Yc�A�*

loss Ow=Cv�r       �	y�2�Yc�A�*

loss|h�;���       �	�3�Yc�A�*

loss�� =�c�       �	�4�Yc�A�*

loss{��=��o�       �	�4�Yc�A�*

lossnHG=z-��       �	mY5�Yc�A�*

loss,�C=��{�       �	��5�Yc�A�*

losslZ�=���B       �	�6�Yc�A�*

losst��<0�t�       �	�57�Yc�A�*

loss�[�<�V	c       �	��7�Yc�A�*

loss���;��B       �	Lo8�Yc�A�*

loss�J�=3c�       �	�9�Yc�A�*

lossd�<����       �	��9�Yc�A�*

lossI�=9�b       �	�I:�Yc�A�*

loss��C=m�       �	l�:�Yc�A�*

lossT38=���       �	g�;�Yc�A�*

lossA`V<$Ri       �	��<�Yc�A�*

loss_ <����       �	Ɖ=�Yc�A�*

loss��<���y       �	h#>�Yc�A�*

loss��=KuBn       �	��>�Yc�A�*

loss?�=�M       �	�P?�Yc�A�*

loss6�R=2di�       �	e�?�Yc�A�*

lossO�<!ޢ�       �	�@�Yc�A�*

loss,�=b�	z       �	A�Yc�A�*

lossFo9=biA       �	��A�Yc�A�*

loss-�x=��]       �	|`B�Yc�A�*

lossa!C=�c�       �	C�Yc�A�*

loss��=�N�       �	^�C�Yc�A�*

loss�V�<IB�       �	�HD�Yc�A�*

lossA��=�%:�       �	��D�Yc�A�*

loss�c�<n���       �	�F�Yc�A�*

loss��=�\�       �	��F�Yc�A�*

lossI\�<��       �	�cG�Yc�A�*

loss���;��8       �	i H�Yc�A�*

lossZ\<�%��       �	�H�Yc�A�*

loss@=:"ty       �	YPI�Yc�A�*

loss�Cc=W)7       �	J�Yc�A�*

loss�/�=<f?       �	r�J�Yc�A�*

loss�Dx=��}       �	(dK�Yc�A�*

lossd�=A���       �	)	L�Yc�A�*

loss��<A��       �	 �L�Yc�A�*

loss���=�K�y       �	F�M�Yc�A�*

loss2�)=EW8�       �	�RN�Yc�A�*

loss��<���       �	��N�Yc�A�*

loss��<��j       �	�O�Yc�A�*

loss{f4=�Y�       �	�6P�Yc�A�*

loss�O[<��O�       �	�P�Yc�A�*

loss�	�=�_8       �	HkQ�Yc�A�*

loss��	=�B�F       �	�R�Yc�A�*

loss�	�<,�yC       �	��R�Yc�A�*

lossݗ�<��c�       �	�5S�Yc�A�*

loss�=Лo       �	��S�Yc�A�*

loss�( <�q
�       �	�gT�Yc�A�*

loss�L<�su       �	U�Yc�A�*

loss.|}=�2��       �	_�U�Yc�A�*

loss
�#<7O��       �	4V�Yc�A�*

loss�PC=�� Q       �	i�V�Yc�A�*

loss��-=��PN       �	��W�Yc�A�*

loss7o=����       �	+/X�Yc�A�*

loss��h<1e�j       �	��X�Yc�A�*

loss��<U��       �	�uY�Yc�A�*

loss�E�=�J<"       �	!Z�Yc�A�*

lossZ�p=�)       �	4�Z�Yc�A�*

loss\	=��b!       �	Ԟ[�Yc�A�*

loss���=����       �	gF\�Yc�A�*

loss%�M<Xß�       �	�\�Yc�A�*

lossi��=~H0�       �	i�]�Yc�A�*

lossN�p=xn�i       �	�B^�Yc�A�*

loss�4�=�Y       �	{�^�Yc�A�*

loss���<IrT#       �	؁_�Yc�A�*

loss�n\=��Q)       �	�!`�Yc�A�*

loss,x�<�w��       �	&�`�Yc�A�*

loss�s=m�       �	�la�Yc�A�*

loss�3=�wR       �	�	b�Yc�A�*

lossQV`=����       �	ʥb�Yc�A�*

lossW�h= ��       �	�Mc�Yc�A�*

loss�MV<u/�       �	�	d�Yc�A�*

loss�9=��ڰ       �	�@e�Yc�A�*

loss��=A�t�       �	�g�Yc�A�*

loss�<�՝<       �	��g�Yc�A�*

loss4~�<��]       �	�sh�Yc�A�*

loss#.�= dX/       �	/i�Yc�A�*

losss��<[3Y�       �	G�i�Yc�A�*

lossh�=�OEQ       �	amj�Yc�A�*

loss|�U=��DK       �	k�Yc�A�*

loss��[=���U       �	�k�Yc�A�*

lossC��<�O	�       �	h^l�Yc�A�*

loss;]�<O+h=       �	��l�Yc�A�*

loss7��=6��L       �	j�m�Yc�A�*

loss��Q=�>UT       �	Fn�Yc�A�*

loss��R<4e��       �	&�n�Yc�A�*

losst�:=*\*�       �	Xto�Yc�A�*

loss�t=�xKx       �	p�Yc�A�*

loss��_=`���       �	w�p�Yc�A�*

lossR��<��\       �	�Eq�Yc�A�*

loss�dn<���       �	��q�Yc�A�*

lossZ�<6Xԩ       �	[{r�Yc�A�*

loss��e=x�$       �	ns�Yc�A�*

loss���<�q0X       �	`�s�Yc�A�*

loss�hV<�IK�       �	'Jt�Yc�A�*

loss�V�<��       �	�Ju�Yc�A�*

loss.�<t��Q       �	 �u�Yc�A�*

loss�eC;���       �	L�v�Yc�A�*

loss��D<�A�       �	Z)w�Yc�A�*

loss-�<z�o       �	�x�Yc�A�*

lossJ�d<n�
       �	7�x�Yc�A�*

loss��,=��@�       �	WCy�Yc�A�*

loss�>�l       �	��y�Yc�A�*

loss왑=��|       �	{�Yc�A�*

loss��;R�}       �	��{�Yc�A�*

loss:r0=Ƽ�=       �	-�|�Yc�A�*

lossԷ�<���s       �	2X}�Yc�A�*

loss>�=��       �	8�}�Yc�A�*

loss�|/<��,       �	ٓ~�Yc�A�*

lossEe�<Z��       �	/�Yc�A�*

loss�=?@C�       �	���Yc�A�*

loss��=ôP       �	cb��Yc�A�*

loss`�w=��?       �	����Yc�A�*

loss慘<k�i�       �	����Yc�A�*

loss���<�c��       �	Y0��Yc�A�*

loss.k�<�)@m       �	�ڃ�Yc�A�*

loss͏�<*A�       �	�؄�Yc�A�*

loss��=���e       �	{��Yc�A�*

loss
K"=@eF�       �	;��Yc�A�*

loss �<����       �	崆�Yc�A�*

loss��l<Q*��       �	PV��Yc�A�*

loss���=��_�       �	Yc�A�*

loss\x�=�^δ       �	����Yc�A�*

lossNf�=����       �	�,��Yc�A�*

loss�=��       �	�ĉ�Yc�A�*

loss1=��b       �	Ef��Yc�A�*

loss��J<�IwI       �	���Yc�A�*

lossiֺ<��       �	����Yc�A�*

loss�@
=UB>       �	 8��Yc�A�*

loss#�<܆a�       �	 ь�Yc�A�*

loss�>�=�l��       �	Lk��Yc�A�*

loss���=w��       �	���Yc�A�*

loss��F=�f��       �	����Yc�A�*

loss��=@e��       �	%>��Yc�A�*

lossdވ=�{��       �	^Տ�Yc�A�*

loss3~D=O˘+       �	�j��Yc�A�*

loss�	=[��       �	���Yc�A�*

loss��>��^1       �	m���Yc�A�*

loss��g=�0�3       �	D��Yc�A�*

loss�_=�O4       �	C璛Yc�A�*

loss��L=�E?�       �	ۉ��Yc�A�*

lossE2a<��*�       �	R'��Yc�A�*

loss��k=
��S       �	�Yc�A�*

loss�N�<P�Vi       �	�W��Yc�A�*

loss�{&=0��s       �	�Yc�A�*

loss���=p�*�       �	���Yc�A�*

loss��<�R�;       �	*:��Yc�A�*

loss~Ρ<�*��       �	X��Yc�A�*

lossX4�<���       �	>���Yc�A�*

loss�<��       �	�I��Yc�A�*

loss,҂=
8W�       �	3���Yc�A�*

loss�=Dg�|       �	ĕ��Yc�A�*

loss2��<�q�       �	^I��Yc�A�*

loss�Q�=�0e�       �	ݛ�Yc�A�*

loss� <���       �	`u��Yc�A�*

loss1�><:{�       �	���Yc�A�*

loss���<N,�:       �	�Yc�A�*

loss�u�<Y�Ø       �	�j��Yc�A�*

loss.�<U.=0       �	R
��Yc�A�*

loss�ҷ<��       �	���Yc�A�*

loss9>=�$N�       �	�>��Yc�A�*

losszg�<��q       �	<ڠ�Yc�A�*

loss_u=
�޵       �	lw��Yc�A�*

lossۗ�<n�       �	���Yc�A�*

loss
3�=� ~q       �	\Y��Yc�A�*

loss��@<� �}       �	gd��Yc�A�*

lossA�9<)K�       �	3��Yc�A�*

loss��<����       �	0ԥ�Yc�A�*

lossO�=>pb       �	���Yc�A�*

loss�S�=���       �	�@��Yc�A�*

loss/�<i��       �	�৛Yc�A�*

losst��= rLC       �	���Yc�A�*

losse��=�W       �	G<��Yc�A�*

loss{�=���       �	�婛Yc�A�*

lossO=��S       �	����Yc�A�*

loss��<B�q0       �	�:��Yc�A�*

losseyD={��       �	�諛Yc�A�*

loss���=ii�U       �	O���Yc�A�*

loss�3�<%��[       �	�7��Yc�A�*

loss��5=�]b1       �	
׭�Yc�A�*

loss*�S<
��       �	ׅ��Yc�A�*

loss�<=_T�       �	�_��Yc�A�*

loss�V>=Ӭ�5       �	%A��Yc�A�*

loss��9=kp�       �	����Yc�A�*

loss�	=�7��       �	=���Yc�A�*

lossԄ<=W�S�       �	!:��Yc�A�*

loss�a=�f�       �	kز�Yc�A�*

loss�V=6���       �	:x��Yc�A�*

loss�Q=�K�       �	I��Yc�A�*

lossl��<\��       �	�봛Yc�A�*

loss��N<!	��       �	&���Yc�A�*

loss�;<���       �	�)��Yc�A�*

loss�38=�Ud       �	FͶ�Yc�A�*

loss��-=�b-       �	�q��Yc�A�*

lossOZ�<�h       �	���Yc�A�*

loss��7=:1?       �	����Yc�A�*

loss���<&��       �	a��Yc�A�*

lossX��;��E       �	����Yc�A�*

loss0�=�F/z       �	h���Yc�A�*

loss!#�=3b\       �	\9��Yc�A�*

loss��~=���       �	�ۻ�Yc�A�*

lossNX=#.5�       �	����Yc�A�*

loss��G<���`       �	j0��Yc�A�*

loss�X�=)�       �	�ɽ�Yc�A�*

loss��>����       �	4���Yc�A�*

loss:�<j��8       �	C��Yc�A�*

loss�G=�)R       �	R���Yc�A�*

loss�]�=�Bko       �	�Q��Yc�A�*

loss�a�=aW?F       �	l���Yc�A�*

loss6�;<W�       �	����Yc�A�*

loss���<��պ       �	�+Yc�A�*

loss���=OA       �	��Yc�A�*

loss��=j��       �	uxÛYc�A�*

lossʑ=Q�       �	{ěYc�A�*

lossz��<MK��       �	�ěYc�A�*

loss�!�<���>       �	TśYc�A�*

losssc�<Z��)       �	l�śYc�A�*

loss�-=٪��       �	�ƛYc�A�*

loss+{=�,�]       �	�ǛYc�A�*

loss��<�Ԙv       �	��ǛYc�A�*

loss\'�;jO
       �	�RțYc�A�*

loss�=G5       �	Q�țYc�A�*

loss�V =q��       �	
�ɛYc�A�*

loss���;��       �	�EʛYc�A�*

lossm<'<Cl��       �	��ʛYc�A�*

loss+؏<�DG       �	�˛Yc�A�*

loss�W>aD��       �	�-̛Yc�A�*

loss��<�QO�       �	O�̛Yc�A�*

loss���=M���       �	�f͛Yc�A�*

loss��;�eN�       �	��͛Yc�A�*

lossmЋ=�_+�       �	̚ΛYc�A�*

loss@�D=�ۺ       �	P5ϛYc�A�*

loss���<�.�       �	B�ϛYc�A�*

loss�͋<0�w       �	shЛYc�A�*

loss��;��       �	6ћYc�A�*

loss|�<q���       �	1�ћYc�A�*

loss���;�*��       �	�pқYc�A�*

lossLE�=�/p�       �	�ӛYc�A�*

loss�'Z=u�ez       �	r�ӛYc�A�*

lossc�=��Z       �	`;ԛYc�A�*

loss8�<��T       �	��ԛYc�A�*

loss���<��W       �	Do՛Yc�A�*

lossf=Y�>       �	�֛Yc�A�*

loss���<u�߉       �	�֛Yc�A�*

loss==����       �	D4כYc�A�*

lossO�.=(�x�       �	��כYc�A�*

loss�D<�8�8       �	�i؛Yc�A�*

losslmH<}�M       �	�ٛYc�A�*

lossa�=        �	��ٛYc�A�*

loss��<���U       �	Q/ڛYc�A�*

lossElz=z$*       �	��ڛYc�A�*

loss֚�=��<       �	|_ۛYc�A�*

loss��<�p��       �	��ۛYc�A�*

lossŲd<4m�       �	 �ܛYc�A�*

loss�#Y=���A       �	�)ݛYc�A�*

loss�~�<��@       �	��ݛYc�A�*

loss��D=���       �	+lޛYc�A�*

lossIhm<���       �	cߛYc�A�*

loss�3�=�꫰       �	�ߛYc�A�*

loss��;l�Dm       �	�I��Yc�A�*

lossL1<���       �	s���Yc�A�*

loss�օ<�x�       �	��Yc�A�*

loss�`=Q���       �	�l�Yc�A�*

loss��u<�k/S       �	���Yc�A�*

loss3��<>ܽ;       �	�R�Yc�A�*

lossox<�\��       �	
��Yc�A�*

lossΖ4=M�       �	���Yc�A�*

loss��:=8��       �	���Yc�A�*

loss8F6<��-*       �	�Y�Yc�A�*

lossQ�\=X�7N       �	>�Yc�A�*

loss.Y�;�G7       �	;��Yc�A�*

loss܅>=��u       �	�L�Yc�A�*

loss��s=X��       �	���Yc�A�*

loss��==�q�       �	��Yc�A�*

loss/�7<�*i�       �	U3�Yc�A�*

loss��Q<˰3�       �	r5�Yc�A�*

loss��'<��U7       �	���Yc�A�*

loss_�=�<i       �	Bv�Yc�A�*

loss���<�E[        �	��Yc�A�*

loss��<fSZ       �	�2�Yc�A�*

loss���;���4       �	���Yc�A�*

loss��<G�/�       �	��Yc�A�*

loss��'=�HO�       �	� �Yc�A�*

loss�w#=��<�       �	R��Yc�A�*

loss	'�<��"�       �	d��Yc�A�*

loss��=0á�       �	=*��Yc�A�*

loss��D=a�
`       �	����Yc�A�*

loss���<�Y�       �	�X��Yc�A�*

loss��!<x�'(       �	o���Yc�A�*

loss��9=B��       �	����Yc�A�*

loss���<Ց0�       �	t'��Yc�A�*

lossZn�::���       �	@���Yc�A�*

loss��c;Gw�       �	����Yc�A�*

loss�x<T�       �	(��Yc�A�*

loss���;����       �	b���Yc�A�*

lossV��<UK��       �	:W��Yc�A�*

lossq�<�H��       �	,���Yc�A�*

loss�k�<��`�       �	����Yc�A�*

loss��	=�T}       �	�6��Yc�A�*

loss��;��?Q       �	����Yc�A�*

loss�2�;'���       �	Xs��Yc�A�*

loss��=Q�,�       �	r��Yc�A�*

lossֲ=�U�0       �	�p��Yc�A�*

loss�*�<��)       �	 
 �Yc�A�*

lossv�9�]�       �	�� �Yc�A�*

loss�cK<v%�\       �	0K�Yc�A�*

loss��V>���#       �	��Yc�A�*

losss��:�!j.       �	_|�Yc�A�*

loss�w<]TmZ       �	�&�Yc�A�*

losst��<�v�       �	���Yc�A�*

loss��Q=��З       �	rm�Yc�A�*

loss?��<��D"       �	T�Yc�A�*

loss��c<�
&�       �	R��Yc�A�*

loss\Yo=xdd        �	�Q�Yc�A�*

lossw��<)��=       �	���Yc�A�*

loss�&Y=���c       �	��Yc�A�*

loss�u�=�'<       �	��Yc�A�*

loss*38<�ׇ
       �	0��Yc�A�*

lossD��=4�-x       �	�P	�Yc�A�*

loss�>^�`       �	��	�Yc�A�*

loss��<���       �	�|
�Yc�A�*

loss
��<\)D       �	�Yc�A�*

loss�U=ˀ@�       �	���Yc�A�*

loss�/=�@G�       �	DM�Yc�A�*

loss���;m,��       �	���Yc�A�*

lossL�=[���       �	���Yc�A�*

lossy<�e3�       �	�n�Yc�A�*

loss8x�;��լ       �	��Yc�A�*

loss�]d=��e�       �	���Yc�A�*

loss�	�;ϕ��       �	bK�Yc�A�*

lossec�:D?�       �	���Yc�A�*

loss�, <�u�X       �	[x�Yc�A�*

loss;<��F       �	�A�Yc�A�*

loss�KU=���        �	���Yc�A�*

loss8`+<J*�       �	ds�Yc�A�*

loss;Mt<���       �	��Yc�A�*

lossW�T=t��       �	���Yc�A�*

loss(��<�>       �	q=�Yc�A�*

loss��;�Ps       �	���Yc�A�*

loss!�&<#X�!       �	 z�Yc�A�*

loss�;�:�       �	�!�Yc�A�*

lossH(=fe�4       �	���Yc�A�*

loss�4=���       �	h]�Yc�A�*

loss́�<<q       �	���Yc�A�*

lossH4�<P]L:       �	B��Yc�A�*

loss!�C="�       �	I0�Yc�A�*

loss)��<�84�       �	i��Yc�A�*

loss��;��       �	�|�Yc�A�*

loss�C(<�I^-       �	�%�Yc�A�*

loss���<4�т       �	���Yc�A�*

lossD
�<���       �	(��Yc�A�*

losss��<@��       �	t%�Yc�A�*

loss7�=�Vba       �	Y��Yc�A�*

loss2�]=@-�       �	�b�Yc�A�*

loss�i�;uu�r       �	 �Yc�A�*

loss��<3�a       �	'� �Yc�A�*

loss�=x<�ЌJ       �	�B!�Yc�A�*

lossT";��l�       �	w�"�Yc�A�*

loss3�<���H       �	� :�Yc�A�*

loss��V=�.T1       �	��:�Yc�A�*

loss[i�=��.�       �	�U;�Yc�A�*

loss�4=N1�       �	��;�Yc�A�*

loss��<��i       �	�<�Yc�A�*

loss�KN<���J       �	�&=�Yc�A�*

lossm>�<��w�       �	d;?�Yc�A�*

loss�Gn=ё_;       �	��?�Yc�A�*

loss�=��       �	�@�Yc�A�*

loss� =}���       �	��A�Yc�A�*

loss��A=��x�       �	1�B�Yc�A�*

lossHy=@v	�       �	�?C�Yc�A�*

loss;:=6FQ       �	3�C�Yc�A�*

loss�o�=�M�f       �	||D�Yc�A�*

lossTNB=
W�       �	z9E�Yc�A�*

loss1�<� C       �	��E�Yc�A�*

loss,�;M�K�       �	r�F�Yc�A�*

loss; =�D�       �	1%G�Yc�A�*

loss�]�<%�sx       �	��G�Yc�A�*

lossh�=��}       �	�jH�Yc�A�*

loss4�<2���       �	cI�Yc�A�*

loss���<��       �	�I�Yc�A�*

loss�<�=�	       �	:]J�Yc�A�*

lossD�:=x{�       �	0�J�Yc�A�*

loss}å<��       �	��K�Yc�A�*

loss��<4       �	b-L�Yc�A�*

loss�W=#w       �	_�L�Yc�A�*

loss��<����       �	�zM�Yc�A�*

loss*�=Q�C       �	
,N�Yc�A�*

loss�e=Ccq\       �	��N�Yc�A�*

loss!)�<��       �	�fO�Yc�A�*

lossZv$<X�       �	��O�Yc�A�*

loss4��<�.=       �	�P�Yc�A�*

loss�=�߳�       �	�2Q�Yc�A�*

lossd�:=�0E#       �	��Q�Yc�A�*

loss�!=��R�       �	LmR�Yc�A�*

loss��;+��       �	Z)S�Yc�A�*

loss7g�<�\1       �	��S�Yc�A�*

loss�R>x<XC       �	cT�Yc�A�*

loss�i�=�-�e       �	n�T�Yc�A�*

lossk�<�v?'       �	�U�Yc�A�*

loss��=��n}       �	E*V�Yc�A�*

lossiEf<c�\       �	r�V�Yc�A�*

lossɳN=�;/\       �	�_W�Yc�A�*

loss���=���       �	��W�Yc�A�*

loss���=C�M�       �	��X�Yc�A�*

lossLB<�H1}       �	9)Y�Yc�A�*

loss
��<p���       �	��Y�Yc�A�*

loss��<��       �	�gZ�Yc�A�*

lossE�6;��       �	[�Yc�A�*

loss�c4<@7(       �	��[�Yc�A�*

loss�m7=d�8       �	�2\�Yc�A�*

lossW�3<��~       �	�\�Yc�A�*

loss���=]�iM       �	�f]�Yc�A�*

lossaS�<���       �	^�Yc�A�*

loss$�:!��L       �	ӥ^�Yc�A�*

loss�S;vQj{       �	m<_�Yc�A�*

loss�,�;�m�       �	��_�Yc�A�*

loss�w�=Gn�E       �	�u`�Yc�A�*

loss�EV=����       �	�
a�Yc�A�*

loss�^=����       �	��a�Yc�A�*

loss�O�<���       �	$cb�Yc�A�*

loss�^<���       �	ic�Yc�A�*

loss�	�<�p�.       �	��c�Yc�A�*

loss�<mKH       �	܁d�Yc�A�*

loss-�Q=�T�@       �	�Ie�Yc�A�*

loss�N�<#�E       �	��e�Yc�A�*

loss�U=.�u1       �	4�f�Yc�A�*

loss��_= ZLp       �	]7g�Yc�A�*

losso�=��       �	��g�Yc�A�*

loss�ݡ<��d�       �	{h�Yc�A�*

loss��=O.       �	i�Yc�A�*

lossQ2�<��%       �	p�i�Yc�A�*

loss:�=c*��       �	Ncj�Yc�A�*

loss�=g�I       �	�k�Yc�A�*

lossq�l<i�?       �	�k�Yc�A�*

loss��<�K�       �	�Ol�Yc�A�*

loss��<t�       �	��l�Yc�A�*

loss��!<�U~       �	=�m�Yc�A�*

loss�؃<`wљ       �	�Dn�Yc�A�*

loss%S�=N���       �	v�n�Yc�A�*

loss�sW=h&T       �	|o�Yc�A�*

loss��=#rC�       �	�p�Yc�A�*

lossLD�;[T�       �	��p�Yc�A�*

loss�b =u�ea       �	�Rq�Yc�A�*

lossO#=�A�       �	L�q�Yc�A�*

loss&\�<b�l       �	��r�Yc�A�*

loss�y<q$�O       �	?�s�Yc�A�*

loss��<�%��       �	�Et�Yc�A�*

loss��<��W�       �	 �t�Yc�A�*

loss��=�[G�       �	��u�Yc�A�*

lossr�<0��       �	@Ov�Yc�A�*

loss�L�<�'�l       �	qw�Yc�A�*

loss@�=9?��       �	�w�Yc�A�*

loss��<�MA]       �	�ux�Yc�A�*

loss��;ԁf�       �	�$y�Yc�A�*

loss�@.=|���       �	p�y�Yc�A�*

loss�[4=*�s�       �	ĵz�Yc�A�*

lossϓ�<׈�x       �	�n{�Yc�A�*

lossM��;�2||       �	PT|�Yc�A�*

lossmd;o���       �	�k}�Yc�A�*

lossB<�X�       �	@~�Yc�A�*

loss��</"�       �		�~�Yc�A�*

loss쥝<���       �	Mg�Yc�A�*

loss,u=�l��       �	7��Yc�A�*

loss&a�=�OVm       �	����Yc�A�*

loss�r =�]�       �	�ف�Yc�A�*

lossX��< ��       �	����Yc�A�*

loss �C=�)4g       �	�]��Yc�A�*

loss���<AY�j       �	
��Yc�A�*

lossb�=P35C       �	{J��Yc�A�*

loss�N<���d       �	�R��Yc�A�*

loss6g�<#֒       �	r���Yc�A�*

lossfV�;Bd�>       �	����Yc�A�*

loss{	
=��oS       �	���Yc�A�*

lossz;\<�"�       �	���Yc�A�*

lossw;=�*       �	�7��Yc�A�*

loss�=)>7�       �	"ފ�Yc�A�*

loss�mc=�Q:q       �	����Yc�A�*

loss�g�<���u       �	�=��Yc�A�*

loss� �=M^�       �	�㌜Yc�A�*

loss��=KR       �	a���Yc�A�*

loss��z=t���       �	V,��Yc�A�*

loss�a�;(�X       �	�֎�Yc�A�*

loss�ݢ<��9       �	6u��Yc�A�*

loss�3U=��CR       �	`!��Yc�A�*

lossvu1=֮�       �	�Ő�Yc�A�*

loss���<߭��       �	�f��Yc�A�*

lossϕ�=�sλ       �	��Yc�A�*

loss��	=��       �	����Yc�A�*

loss
K1<�E�O       �	�I��Yc�A�*

lossF[
=��˳       �	]ⓜYc�A�*

lossJ�=cj�z       �	����Yc�A�*

lossP�<k#1�       �	�O��Yc�A�*

lossx4=D9G0       �	RYc�A�*

lossM�)<8c�       �	M���Yc�A�*

loss���<)��       �	%��Yc�A�*

loss�Kd<3n�       �	໗�Yc�A�*

loss�X<]�l�       �	+���Yc�A�*

loss��=��       �	-!��Yc�A�*

loss���;���       �	�͙�Yc�A�*

loss�;�;�Jk       �	us��Yc�A�*

loss�v=Ș�       �	U��Yc�A�*

loss!�<� �       �	B���Yc�A�*

loss��=�R��       �	_C��Yc�A�*

lossR�<X-K       �	䜜Yc�A�*

loss,s=�Kx       �	p}��Yc�A�*

lossA�?< ��>       �	^��Yc�A�*

loss(��;M�b       �	g힜Yc�A�*

loss��=�yŇ       �	����Yc�A�*

loss4+<� x|       �	\8��Yc�A�*

lossŠ�<���       �	�ݠ�Yc�A�*

loss�y=�ٺ�       �	T���Yc�A�*

loss�B=�*�=       �	�Q��Yc�A�*

loss�6=�j�&       �	xb��Yc�A�*

loss^�= y�       �	^��Yc�A�*

loss�[�;��6�       �	\��Yc�A�*

loss���<���n       �	&V��Yc�A�*

loss۽�=O.eV       �	m ��Yc�A�*

loss���<��        �	D0��Yc�A�*

loss��/=>HE�       �	1���Yc�A�*

lossD��<;�X�       �	ȩ�Yc�A�*

loss�T<s��H       �	�q��Yc�A�*

loss��`<{b��       �	����Yc�A�*

loss
=�w�0       �	Ժ��Yc�A�*

loss�ݥ<U�l�       �	~���Yc�A�*

lossj7Y<�t       �	$���Yc�A�*

loss��<��gy       �	|��Yc�A�*

loss<��H�       �	77��Yc�A�*

loss�z�=�9�       �	[��Yc�A�*

loss��5=V%�       �	>���Yc�A�*

loss���;�.1�       �	�Q��Yc�A�*

loss�	V<u�       �	�볜Yc�A�*

loss��5=Kٯ�       �	f���Yc�A�*

loss���<����       �	I-��Yc�A�*

loss���<8q�F       �	Bϵ�Yc�A�*

loss��=m��5       �	Yi��Yc�A�*

loss3�<���       �	F��Yc�A�*

loss��=�U�       �	�޷�Yc�A�*

loss�T=N.��       �	_y��Yc�A�*

loss���;lF I       �	���Yc�A�*

loss��C=�\p�       �	X���Yc�A�*

loss�U=L[.       �	�E��Yc�A�*

loss3_�<�	�3       �	�}��Yc�A�*

loss��A<��d       �	u"��Yc�A�*

loss&�</R�       �	���Yc�A�*

loss�9�<���       �	Oy��Yc�A�*

loss��<�]2�       �	/��Yc�A�*

loss!�'=r�       �	Ӽ��Yc�A�*

lossSG=�hXA       �	�_��Yc�A�*

loss1f�=��H       �	���Yc�A�*

loss��<0Y��       �	 ���Yc�A�*

lossڔ�<��K>       �	�[��Yc�A�*

loss�.<V�?       �	,�Yc�A�*

loss��B;~�u       �	�dÜYc�A�*

loss�<���d       �	RĜYc�A�*

lossT�g<�       �	-�ĜYc�A�*

lossd�=󲨕       �	,JŜYc�A�*

loss(=�_       �	��ŜYc�A�*

loss&��<�1j�       �	��ƜYc�A�*

loss(v�<f#\       �	�"ǜYc�A�*

loss0�=Cc��       �	��ǜYc�A�*

loss_�G=(       �	�dȜYc�A�*

loss��<�2�)       �	0ɜYc�A�*

lossWz�<���       �	K�ɜYc�A�*

loss���<�*�X       �	<kʜYc�A�*

loss�Ơ<E���       �	�˜Yc�A�*

loss��h<rT�       �	̸˜Yc�A�*

loss�=���       �	�P̜Yc�A�*

loss�*#=#�m�       �	��̜Yc�A�*

lossD�%=��       �	Ü͜Yc�A�*

loss*��<)Y'       �	AFΜYc�A�*

loss ��;��       �	��ΜYc�A�*

loss��<G��       �	ΉϜYc�A�*

loss�,:=끃�       �	3МYc�A�*

loss�|�=kǱ%       �	��МYc�A�*

loss#��=�j�
       �	�uќYc�A�*

loss4�=MŴ       �	sҜYc�A�*

loss�d�=���D       �	�ҜYc�A�*

lossݖ�<�'/       �	�gӜYc�A�*

loss_��;*y�z       �	 ԜYc�A�*

lossAP�;'*Q�       �	ܹԜYc�A�*

loss�<���       �	�a՜Yc�A�*

loss��=C�n       �	9
֜Yc�A�*

loss�U="�U|       �	a�֜Yc�A�*

loss��`<}9>       �	�AלYc�A�*

loss��<xU��       �	j�לYc�A�*

lossw��<�7�U       �	s؜Yc�A�*

lossw c<=�V�       �	EٜYc�A�*

loss/^�<Զ��       �	�ٜYc�A�*

loss�|<��-�       �	�EڜYc�A�*

loss/}C;��       �	��ڜYc�A�*

loss�[<����       �	ϤۜYc�A�*

loss]x8<v	�&       �	DLܜYc�A�*

loss,�h={�6       �	��ܜYc�A�*

loss���=Y$N)       �	��ݜYc�A�*

loss���<�M_�       �	|(ޜYc�A�*

loss;=X=w)��       �	>�ޜYc�A�*

loss?��=l�4�       �	�ߜYc�A�*

lossz�<��       �	����Yc�A�*

loss�<��2       �	W@�Yc�A�*

loss,q�<�\��       �	%��Yc�A�*

lossQ��<, �T       �	@��Yc�A�*

lossnWb<(��b       �	w��Yc�A�*

loss(�=�       �	9B�Yc�A�*

loss}r>=Cn;1       �	-�Yc�A�*

loss��X<��*       �	xc�Yc�A�*

lossf��<Y��$       �	ir�Yc�A�*

loss�t=�(       �	�B�Yc�A�*

lossr��<�       �	g
�Yc�A�*

loss�=0S       �	���Yc�A�*

lossy��<�5t       �	߉�Yc�A�*

loss���<�DKX       �	VI�Yc�A�*

lossGY<G�<�       �	���Yc�A�*

loss�O%;�A�q       �	T��Yc�A�*

loss,�x<q�l�       �	f0�Yc�A�*

loss��C<\�C|       �	F��Yc�A�*

loss��=�w�       �	�Z�Yc�A�*

loss*�9=�#�r       �	���Yc�A�*

loss�+�<�f��       �	���Yc�A�*

loss�y=��g       �	M1�Yc�A�*

loss��<��_^       �	���Yc�A�*

loss�F�=�p��       �	�h�Yc�A�*

loss�F=�7�       �	��Yc�A�*

loss{}M=9hڂ       �	F��Yc�A�*

loss��;p���       �	2��Yc�A�*

lossä>>,��       �	G���Yc�A�*

loss�a=���       �	^d��Yc�A�*

loss�1=��(       �	&���Yc�A�*

loss�î<P�/F       �	t���Yc�A�*

loss���<��A�       �	�V��Yc�A�*

loss�ܷ=jsd�       �	����Yc�A�*

lossXX�=+4�       �	���Yc�A�*

loss P=O��       �	�1��Yc�A�*

loss�u+=SHF�       �	����Yc�A�*

loss���=Z��i       �	�^��Yc�A�*

loss�/�<���       �	A���Yc�A�*

loss�� =�i�       �	���Yc�A�*

losse�d<|F��       �	���Yc�A�*

loss��<O��       �	J���Yc�A�*

loss��x<��l       �	T��Yc�A�*

loss�{<jp��       �	����Yc�A�*

loss�7�<�	]e       �	6���Yc�A�*

loss6�T<����       �	5%��Yc�A�*

losst�=�p�       �	����Yc�A�*

loss%4'=`9o       �	�h �Yc�A�*

loss���<C)5n       �	�� �Yc�A�*

loss�y<X���       �	>��Yc�A�*

loss��P=��M�       �	@�Yc�A�*

loss���;��       �	w��Yc�A�*

loss��;��       �	��Yc�A�*

loss141<��N
       �	�Yc�A�*

loss�-;C�H�       �	���Yc�A�*

lossJ>�;��6       �	{h�Yc�A�*

loss�ݦ</-g2       �	` �Yc�A�*

loss��<���       �	��Yc�A�*

lossL��<C�N(       �	9E�Yc�A�*

loss�G�<��z�       �	W��Yc�A�*

loss��=7�       �	��Yc�A�*

lossz�.=+?��       �	�	�Yc�A�*

lossͤ=_,ϰ       �	J�	�Yc�A�*

loss���<��Ie       �	�R
�Yc�A�*

loss�K=/�#�       �	-�
�Yc�A�*

lossA=�u       �	ǁ�Yc�A�*

loss3�w=���G       �	��Yc�A�*

loss�	>�]M�       �	���Yc�A�*

loss��x<hFz       �	]o�Yc�A�*

loss3��=Nf�i       �	O�Yc�A�*

loss�78<�)&       �	 ��Yc�A�*

loss<�<k*�       �	kJ�Yc�A�*

loss��(=::�       �	���Yc�A�*

lossϛ�<q?�       �	���Yc�A�*

loss���<ml�J       �	�#�Yc�A�*

loss�,�;�n�t       �	r��Yc�A�*

loss��~=E�
�       �	�`�Yc�A�*

loss�=1=�fF�       �	[�Yc�A�*

loss|�<�*       �	2��Yc�A�*

loss��}=^�2�       �	�V�Yc�A�*

lossV�q=�FC�       �	&��Yc�A�*

loss���<��Y       �	ܠ�Yc�A�*

lossC��<�Ol       �	�F�Yc�A�*

loss;�=���       �	2�Yc�A�*

loss*=p�"�       �	��Yc�A�*

loss-�t<��$       �	Fa�Yc�A�*

lossQ�<fb[�       �	"��Yc�A�*

loss%�<j�p       �	��Yc�A�*

loss�(�<D�       �	�9�Yc�A�*

lossh��<?�       �	B��Yc�A�*

loss?�0=V�       �	j��Yc�A�*

lossER=�g�       �	�7�Yc�A�*

loss�ɋ<Z�ʁ       �	m��Yc�A�*

loss��<',��       �	f��Yc�A�*

loss�&<@i�       �	�(�Yc�A�*

loss��<B֕
       �	��Yc�A�*

loss�rW=��Z       �	zl�Yc�A�*

loss�t=�7.(       �	.  �Yc�A�*

loss�K�<E��       �	̚ �Yc�A�*

loss͝�<K,-�       �	�@!�Yc�A�*

loss�y�=w�}T       �	I�!�Yc�A�*

loss�s};���
       �	��"�Yc�A�*

lossaS�<hl�'       �	�#�Yc�A�*

loss���;( )�       �	�?$�Yc�A�*

loss�U<��       �	��$�Yc�A�*

loss��;�n�       �	��%�Yc�A�*

lossJ�e=@Uk       �	5*&�Yc�A�*

loss�=C�>       �	��&�Yc�A�*

loss�<��F�       �	/�'�Yc�A�*

loss6�r=X9J       �	�@(�Yc�A�*

lossJ6"=&�j-       �	r�(�Yc�A�*

loss;^�=�n�"       �	�v)�Yc�A�*

loss|��<��e!       �	8*�Yc�A�*

loss��<jӴi       �	�*�Yc�A�*

loss���<�>       �	�@+�Yc�A�*

lossA*=U��       �	o�+�Yc�A�*

loss�h<=y��
       �	�,�Yc�A�*

loss�=�w �       �	�-�Yc�A�*

loss`�<6+��       �	��-�Yc�A�*

loss�F�<�`T6       �	�d.�Yc�A�*

lossr��=�W��       �	"�.�Yc�A�*

loss.��=g^#       �	�/�Yc�A�*

loss�t�<?pb       �	��0�Yc�A�*

losse��<�B�-       �	�!1�Yc�A�*

lossLl�=�"�'       �	��1�Yc�A�*

lossi��=S&p�       �	�Z2�Yc�A�*

loss�� >��W       �	��2�Yc�A�*

losswC�<��9�       �	i�3�Yc�A�*

loss3�=~���       �	�'4�Yc�A�*

loss�_�</$7�       �	ۿ4�Yc�A�*

loss�c�<q�$       �	d\5�Yc�A�*

loss�9=��       �	��5�Yc�A�*

loss)c[=x�x�       �	c�6�Yc�A�*

loss��<b?�       �	�27�Yc�A�*

loss�p={���       �	G�7�Yc�A�*

lossqg=�?p
       �	ur8�Yc�A�*

loss��&=�h{       �	�9�Yc�A�*

loss�%�=�R       �	V�9�Yc�A�*

lossԟ�<�Fi       �	�3:�Yc�A�*

loss��=��Ƥ       �	�:�Yc�A�*

loss��(=,���       �	�{;�Yc�A�*

loss��'=W��)       �	IH<�Yc�A�*

loss���<G�r       �	%�<�Yc�A�*

loss�� =       �	�=�Yc�A�*

loss�B=-
�j       �	��>�Yc�A�*

loss��<���       �	��?�Yc�A�*

loss�i�<��T'       �	�a@�Yc�A�*

lossl��=L�f�       �	�\A�Yc�A�*

loss��=�|i       �	�fB�Yc�A�*

lossD#y<����       �	� C�Yc�A�*

loss�z�<�o�~       �	��C�Yc�A�*

loss��l<�a5�       �	�?D�Yc�A�*

loss�=�E{       �	��D�Yc�A�*

loss��<��e       �	,E�Yc�A�*

loss���<��h       �	��F�Yc�A�*

loss4�U=�W�       �	d!G�Yc�A�*

losslA�<*"�       �	G�G�Yc�A�*

losssC9=�ôx       �	AgH�Yc�A�*

loss��;� �       �	I�Yc�A�*

loss_��<���       �	]�I�Yc�A�*

loss��x<ޥ�       �	�@J�Yc�A�*

loss 2d<n�K0       �	�J�Yc�A�*

loss� �;@��       �	f�K�Yc�A�*

loss��{<�       �	p%L�Yc�A�*

lossaY�< ��       �	��L�Yc�A�*

loss�=�;�~��       �	weM�Yc�A�*

loss(ހ=d�r       �	�M�Yc�A�*

loss�=�CC�       �	�N�Yc�A�*

loss�:�=n��E       �	�'O�Yc�A�*

loss!�^<B��       �	�AP�Yc�A�*

loss��#=�woN       �	6�P�Yc�A�*

lossG<���       �	^�Q�Yc�A�*

loss���=�H�       �	y#R�Yc�A�*

loss��k=}*��       �	��R�Yc�A�*

loss���<���       �	odS�Yc�A�*

loss�a�=��<       �	�T�Yc�A�*

loss�:=����       �	�T�Yc�A�*

loss5�=��0       �	�BU�Yc�A�*

loss_G<�U��       �	�U�Yc�A�*

loss��[<�f	s       �	��V�Yc�A�*

loss Y=�{ֺ       �	L4W�Yc�A�*

loss�)=�װ       �	&�W�Yc�A�*

loss7�e<��!�       �	�X�Yc�A�*

loss�Z:=d<��       �	�!Y�Yc�A�*

loss�`<�徭       �	m�Y�Yc�A�*

lossz-�= s}�       �	�cZ�Yc�A�*

loss3P�=��+       �	>[�Yc�A�*

loss�B�=,4��       �	Χ[�Yc�A�*

loss�#=�2�Q       �	�c\�Yc�A�*

loss7�
>ژZ       �	��\�Yc�A�*

loss*=�ʓ       �	@�]�Yc�A�*

loss���<i�       �	�9^�Yc�A�*

loss#�=����       �	��^�Yc�A�*

loss,2=?���       �	�y_�Yc�A�*

lossID�;8��_       �	%`�Yc�A�*

lossM;�<P�0       �	��`�Yc�A�*

lossK=2-#�       �	&na�Yc�A�*

loss�O�<����       �	b�Yc�A�*

loss��<ͽS�       �	>�b�Yc�A�*

lossQ)=[`�;       �	.�c�Yc�A�*

lossqّ<}��x       �	i6d�Yc�A�*

loss�t3<�L��       �	V�d�Yc�A�*

loss��,=W!+a       �	"qe�Yc�A�*

loss��T=2��N       �	f�Yc�A�*

loss�0�=�n|       �	��f�Yc�A�*

loss\ �=x��2       �	�Cg�Yc�A�*

loss�K<�G       �	-h�Yc�A�*

loss%=�ަ       �	��h�Yc�A�*

loss�P�=U뜁       �	�_i�Yc�A�*

lossES�<�,�       �	{�i�Yc�A�*

loss�1�=J���       �	��j�Yc�A�*

lossN>�4Ms       �	EGk�Yc�A�*

loss���<��       �	��k�Yc�A�*

loss?~<��w       �	�yl�Yc�A�*

loss!�-<;�8       �	y>m�Yc�A�*

loss��=��((       �	��m�Yc�A�*

lossI��=ϢPR       �	�zn�Yc�A�*

loss5V=>Jɪ       �	no�Yc�A�*

lossQ'v<\w       �	@�o�Yc�A�*

loss���<rhpe       �	�^p�Yc�A�*

lossd\u<H��%       �	�q�Yc�A�*

lossd�<�9O?       �	�q�Yc�A�*

lossc3�<K��       �	�Ir�Yc�A�*

loss*v�<>�3       �	��r�Yc�A�*

loss�3�<�ӕ       �	ڍs�Yc�A�*

loss:=2CW       �	�/t�Yc�A�*

lossҹ�= %       �	\�t�Yc�A�*

lossŪ�<�l�       �	tbu�Yc�A�*

loss��=��m�       �	X�u�Yc�A�*

lossȏ�<�r��       �	��v�Yc�A�*

loss!}u<D�p^       �	�Ow�Yc�A�*

loss1��;�H;Z       �	h�w�Yc�A�*

lossE��<N       �	˽x�Yc�A�*

lossRR<��       �	�\y�Yc�A�*

loss�=#=�<       �	�y�Yc�A�*

loss׭�=K��	       �	h�z�Yc�A�*

loss�I�<��Z       �	ip{�Yc�A�*

lossD
@;�I�z       �	�	|�Yc�A�*

losszu<��^�       �	�|�Yc�A�*

loss3��<Qc�d       �	tC}�Yc�A�*

loss�_<'8�w       �	7�}�Yc�A�*

loss��=�+       �	��~�Yc�A�*

loss�ږ=J
��       �	�k�Yc�A�*

lossNY�<�/R       �	�K��Yc�A�*

loss��J=@�"       �	 ဝYc�A�*

loss��=k��q       �	����Yc�A�*

loss�G�<2x��       �	�b��Yc�A�*

loss��<PE&       �	���Yc�A�*

loss�Ĉ<�Q[       �	ک��Yc�A�*

loss��=]2�e       �	:���Yc�A�*

loss�e<;si�       �	�-��Yc�A�*

loss�=;�ݾ       �	s؅�Yc�A�*

loss� =֛+�       �	σ��Yc�A�*

loss�ˮ<�#K       �	k(��Yc�A�*

loss!b;M*b�       �	�Ç�Yc�A�*

loss!� <�#�d       �	�b��Yc�A�*

loss�9=��$�       �	���Yc�A�*

loss4G�<�nv       �	้�Yc�A� *

loss�oG=�惤       �	�R��Yc�A� *

loss��<�xf       �	��Yc�A� *

losse"(=Rƭ       �	󭋝Yc�A� *

loss#�<F��e       �	�H��Yc�A� *

loss�]�=���.       �	�㌝Yc�A� *

lossH@�<�+S�       �	||��Yc�A� *

loss�n�<*:2       �	���Yc�A� *

loss�!q<ZA={       �	����Yc�A� *

loss3�,=ۍ�Y       �	O=��Yc�A� *

loss�Ǎ<�/       �	�ݏ�Yc�A� *

losssϲ<A�*�       �	-w��Yc�A� *

loss �W=M��!       �	 ��Yc�A� *

loss�1�<m࣋       �	ӡ��Yc�A� *

loss���<wl��       �	�=��Yc�A� *

loss���;E��N       �	�㒝Yc�A� *

lossv14=�
T       �	l{��Yc�A� *

losscY�<���       �	�a��Yc�A� *

loss��0=M�<}       �	���Yc�A� *

loss���<����       �	����Yc�A� *

lossJ�=�E��       �	u>��Yc�A� *

loss�U=We(C       �	ޖ�Yc�A� *

loss8�<?���       �	�z��Yc�A� *

loss�֊<��n�       �	���Yc�A� *

lossא}=�:=�       �	�Ę�Yc�A� *

lossq=z�"       �	l��Yc�A� *

loss=��<��       �	�=��Yc�A� *

loss�(];�;�'       �	���Yc�A� *

loss�/�<��=       �	3���Yc�A� *

lossc�=ɜ�       �	�:��Yc�A� *

loss�&E=	#�_       �	5ќ�Yc�A� *

loss�P<����       �	6u��Yc�A� *

loss���=��z�       �	���Yc�A� *

loss�2�=�O�:       �	p���Yc�A� *

lossF��<@�4�       �	@P��Yc�A� *

loss=��;����       �	�꟝Yc�A� *

loss3C�<
�C�       �	䇠�Yc�A� *

lossȕ=uW��       �	��Yc�A� *

loss�:��	       �	����Yc�A� *

loss%��;U�F�       �	�F��Yc�A� *

losszo�<5>�       �	���Yc�A� *

loss���;.��       �	󯣝Yc�A� *

lossSnU<��&�       �	����Yc�A� *

loss�N�;�Q�       �	�9��Yc�A� *

loss���<2�.6       �	�ѥ�Yc�A� *

lossd�<E��       �	]p��Yc�A� *

loss�U�9��b       �	���Yc�A� *

loss�s�;��f4       �	����Yc�A� *

loss;e�<L�
       �	RC��Yc�A� *

losseH =0��a       �	�ب�Yc�A� *

loss��<�7�       �	���Yc�A� *

loss���9^�o�       �	���Yc�A� *

loss�#r<Phre       �	E���Yc�A� *

loss �W=g�	�       �	�O��Yc�A� *

loss_�;m�c       �	���Yc�A� *

loss�,<�8��       �	�%��Yc�A� *

lossfZc<�٘�       �	n���Yc�A� *

loss���<��_       �	n��Yc�A� *

loss�h=�c��       �	���Yc�A� *

loss�n<�ko^       �	𦯝Yc�A� *

loss��>XS��       �	�;��Yc�A� *

lossW��=�s�       �	�ް�Yc�A� *

lossɷM=F�n       �	M���Yc�A� *

loss�d+=
#�       �	q ��Yc�A� *

loss���<|       �	0���Yc�A� *

loss��=S�[�       �	�S��Yc�A� *

lossCI�<[�J       �	��Yc�A� *

loss,��<�pN       �	����Yc�A� *

loss�P=�rX�       �	 *��Yc�A� *

loss_�-=\��(       �	�͵�Yc�A� *

loss�g�<t�       �	�j��Yc�A� *

lossR��<#�Fe       �	q��Yc�A� *

lossR�K<H 8�       �	����Yc�A� *

loss 5o<ʤ��       �	}<��Yc�A� *

loss�ͻ;���       �	_Ը�Yc�A� *

loss]�R=i�a@       �	�j��Yc�A� *

lossC^=XWWr       �	���Yc�A� *

loss7�;�ww�       �	9���Yc�A� *

loss\K�;���       �	u��Yc�A� *

loss�z;r        �	���Yc�A� *

loss�l�<�R~       �	֨��Yc�A� *

loss=`�<��O       �	�A��Yc�A� *

loss��=�2�       �	Uٽ�Yc�A� *

loss���=��<�       �	en��Yc�A� *

loss�d$=��?i       �	���Yc�A� *

loss�+�<��~       �	|���Yc�A� *

loss�T?;����       �	�4��Yc�A� *

loss��:���       �	(���Yc�A� *

loss�bD<2�yy       �	Ym��Yc�A� *

loss7[4;{�~�       �	�Yc�A� *

loss�ȿ<p%�E       �	�Yc�A� *

loss�~
=���	       �	�pÝYc�A� *

lossw��<�sh2       �	JĝYc�A� *

lossN��=, W       �	��ĝYc�A� *

loss�W�<���       �	�AŝYc�A� *

lossͣd=t��i       �	��ŝYc�A� *

loss�<*�_[       �	�xƝYc�A� *

loss���<~4��       �	@ǝYc�A� *

loss�Ղ< ���       �	ͯǝYc�A� *

loss�,�<�w�l       �	�iȝYc�A� *

loss�G=5F��       �	hɝYc�A� *

loss_�
<	8�       �	�ɝYc�A� *

lossI11=D���       �	�=ʝYc�A� *

loss�S�:#���       �	��ʝYc�A� *

loss<�=tU�       �	E�˝Yc�A� *

loss�1\<93�       �	���Yc�A� *

loss̓Y=ux[E       �	#i�Yc�A� *

lossjO=W`8�       �	  �Yc�A� *

losse<=�:�       �	��Yc�A� *

loss.F<=@�H�       �	���Yc�A� *

lossϤQ<��#       �	���Yc�A� *

lossGQ�=ڠ�<       �	�#�Yc�A� *

loss��@=�;x�       �	`��Yc�A� *

loss.h�<6���       �	�r�Yc�A� *

loss�=��oD       �	�2�Yc�A� *

loss['z<yK��       �	�k�Yc�A� *

lossi�3=��A       �	A�Yc�A� *

loss�d=���       �	���Yc�A� *

loss��-=ׅ�       �	>��Yc�A� *

loss=��m       �	Q0�Yc�A� *

loss�3=i�f�       �	���Yc�A� *

loss���:i�T�       �	��Yc�A� *

loss� =[�J:       �	b-�Yc�A� *

loss&�0<Z�       �	���Yc�A� *

loss��=�S��       �	�t�Yc�A� *

lossD<cT-�       �	x'�Yc�A� *

loss��D=�¶       �	���Yc�A� *

lossW�s<�6c�       �	Cs�Yc�A� *

lossd�<�;u       �	���Yc�A�!*

lossa�<����       �	���Yc�A�!*

lossR��<8%Y�       �	����Yc�A�!*

loss@>=�-Gg       �	�V��Yc�A�!*

loss~�<��       �	E���Yc�A�!*

loss��M=�$�a       �	����Yc�A�!*

loss��A<TpXQ       �	~r��Yc�A�!*

loss�&=.�       �	���Yc�A�!*

loss8�<aJ4w       �	����Yc�A�!*

loss�]+<4P�       �	;o��Yc�A�!*

lossMȳ=�
�|       �	�y��Yc�A�!*

loss��=��mn       �	�/��Yc�A�!*

loss��<=fqz>       �	����Yc�A�!*

loss���;Br �       �	���Yc�A�!*

loss$@>=����       �	�1��Yc�A�!*

loss�>\5�       �	k���Yc�A�!*

loss�&=�!�       �	�p �Yc�A�!*

lossOw�;= 0�       �	 �Yc�A�!*

loss依<�b�+       �	���Yc�A�!*

loss��<dG       �	���Yc�A�!*

loss���<@���       �	@0�Yc�A�!*

loss�`_<���        �	h��Yc�A�!*

loss�O=G�n�       �	�{�Yc�A�!*

loss��<�I�|       �	3�Yc�A�!*

loss|Q�<��}       �	��Yc�A�!*

loss)�=�x��       �	���Yc�A�!*

loss��:;�$Q       �	Ow�Yc�A�!*

loss�3�;)��       �	�/	�Yc�A�!*

loss\ =�|��       �	��	�Yc�A�!*

loss�V�<rj@       �	`w
�Yc�A�!*

loss�
'>�r[�       �	�&�Yc�A�!*

loss퍏=mĔ�       �	2��Yc�A�!*

loss���;�1�       �	;o�Yc�A�!*

lossd\.<�Yr�       �	�Yc�A�!*

loss{ǎ;tS�-       �	���Yc�A�!*

loss_~<EH8o       �	t`�Yc�A�!*

loss��=��       �	�Z�Yc�A�!*

loss�r!=DI�       �	K�Yc�A�!*

loss��<�}f       �	��Yc�A�!*

loss��<;�2�       �	�d�Yc�A�!*

loss���<t�4B       �	��Yc�A�!*

loss;��<�m�       �	��Yc�A�!*

lossrO�<����       �	�S�Yc�A�!*

loss��'<�HE>       �	���Yc�A�!*

lossQ��<Q]��       �	9��Yc�A�!*

loss7�=���3       �	;9�Yc�A�!*

loss�ٴ<o՗       �	���Yc�A�!*

loss1�]=��ղ       �	��Yc�A�!*

loss���<�k
       �	(-�Yc�A�!*

losss�=~3Wa       �	���Yc�A�!*

loss:��<\��       �	W^�Yc�A�!*

lossf�=:$%       �	���Yc�A�!*

loss柦<~#�       �	���Yc�A�!*

lossb�<h:w       �	�1�Yc�A�!*

loss��<Jk�G       �	_��Yc�A�!*

loss�78<~A�       �	���Yc�A�!*

lossZ�<���P       �	O?�Yc�A�!*

loss�V�=g��       �	���Yc�A�!*

loss�=�<r�٫       �	�q�Yc�A�!*

lossK�<����       �	�Yc�A�!*

loss3��;��       �	���Yc�A�!*

loss\�(=�@�Y       �	 c�Yc�A�!*

loss��a<^���       �	� �Yc�A�!*

lossE�q<65[=       �	+� �Yc�A�!*

loss�2�=�W�s       �	D!�Yc�A�!*

loss��<��-�       �	��!�Yc�A�!*

loss��<�H�e       �	�w"�Yc�A�!*

losszz|=X8�       �	�#�Yc�A�!*

loss��<@�       �	w�#�Yc�A�!*

loss!=��n5       �	�a$�Yc�A�!*

loss6ڂ=�� �       �	m�$�Yc�A�!*

loss�6=� �0       �	�%�Yc�A�!*

loss�rr;?�s       �	N�&�Yc�A�!*

loss��W=�KF?       �	+P'�Yc�A�!*

loss��=��u�       �	>(�Yc�A�!*

loss�3f=L�j�       �	ګ(�Yc�A�!*

loss�8�<aQ�u       �	�O)�Yc�A�!*

loss;<����       �	(�)�Yc�A�!*

loss��~<I�2g       �	ԙ*�Yc�A�!*

loss-�:<?z�       �	�E+�Yc�A�!*

lossjL�;]7�       �	p�+�Yc�A�!*

losso�a=g�M       �	��,�Yc�A�!*

lossq�=���t       �	?;-�Yc�A�!*

loss�G`<}�|�       �	i�-�Yc�A�!*

loss���<^��       �	7�.�Yc�A�!*

loss�]�<�lY�       �	N'/�Yc�A�!*

lossы�<)��5       �	��/�Yc�A�!*

loss?ͳ<����       �	�0�Yc�A�!*

loss���<���       �	t1�Yc�A�!*

loss`<?1��       �	02�Yc�A�!*

loss4h�;�x�       �	��2�Yc�A�!*

loss�5�;�b��       �	�O3�Yc�A�!*

loss6w�;C'C�       �	K4�Yc�A�!*

loss�r�<MU�^       �	0�4�Yc�A�!*

loss��^=�Ej�       �	95�Yc�A�!*

loss�5C==��       �	f�5�Yc�A�!*

loss�X�;<�oo       �	�z6�Yc�A�!*

loss(�<F�2�       �	�7�Yc�A�!*

loss�Ȑ<�b       �	-�7�Yc�A�!*

loss�*�;9�:C       �	<K8�Yc�A�!*

loss�d=b��g       �	��8�Yc�A�!*

loss8L�<#���       �	�9�Yc�A�!*

loss��'=�*7       �	A(:�Yc�A�!*

lossJ�<ѥ��       �	��:�Yc�A�!*

loss��C<��b�       �	mY;�Yc�A�!*

loss��G=�k��       �	��;�Yc�A�!*

lossUn<e�Ə       �	��<�Yc�A�!*

loss��_=� �E       �	�/=�Yc�A�!*

loss3J)=��\b       �	6�=�Yc�A�!*

loss�=���       �	�w>�Yc�A�!*

losse0�;�Y ;       �	�?�Yc�A�!*

lossdp
=��.�       �	Ѱ?�Yc�A�!*

loss�Dt<��,       �	�Q@�Yc�A�!*

loss�
=+3Yb       �	��@�Yc�A�!*

loss{o<��       �	ɑA�Yc�A�!*

loss=4#=eV#       �	:B�Yc�A�!*

loss\��<�T5I       �	��B�Yc�A�!*

loss슞;�)�?       �	n�C�Yc�A�!*

loss�j<�O�       �	kD�Yc�A�!*

loss�q�=6+�       �	�E�Yc�A�!*

loss�=��4�       �	 �E�Yc�A�!*

loss��=
��W       �	�HF�Yc�A�!*

losst�8<P$bc       �	�G�Yc�A�!*

loss�=���       �	��G�Yc�A�!*

loss<:Vr�       �	�MH�Yc�A�!*

lossO��<v���       �	%�H�Yc�A�!*

loss�d�<�c�A       �	��I�Yc�A�!*

loss��v;��       �	s-J�Yc�A�!*

loss�BU=�ݎ�       �	�J�Yc�A�"*

lossŏ�=�jB       �	�K�Yc�A�"*

loss@$�<$o;d       �	�"L�Yc�A�"*

lossn!�=B       �	�L�Yc�A�"*

lossR�:G0�       �	�kM�Yc�A�"*

loss�`�;�:�#       �	 N�Yc�A�"*

loss�^n=� #}       �	�N�Yc�A�"*

loss#��=�.b~       �	�O�Yc�A�"*

lossl 3<�O��       �	33P�Yc�A�"*

lossW0=">^n       �	
�P�Yc�A�"*

loss
&f<���       �	zQ�Yc�A�"*

loss�c�<Zp�       �	�R�Yc�A�"*

loss��P;u��       �	<�R�Yc�A�"*

loss���=�*t       �	<iS�Yc�A�"*

loss�/�<����       �	ST�Yc�A�"*

loss ĥ;���,       �	n�T�Yc�A�"*

loss��<���       �	<V�Yc�A�"*

loss��f<p���       �	E�V�Yc�A�"*

loss�'�<�Yu�       �	>yW�Yc�A�"*

loss
6�<L"       �	X�Yc�A�"*

loss�_<8��       �	AcY�Yc�A�"*

loss�?�<O1�       �	�Z�Yc�A�"*

loss��=*�Ż       �	��Z�Yc�A�"*

loss��<��{S       �	܁\�Yc�A�"*

lossD��=��/       �	�8]�Yc�A�"*

loss��a=���       �	V^�Yc�A�"*

loss�Fm=s�k�       �	�^�Yc�A�"*

loss��7=r!       �	�_�Yc�A�"*

loss��=�Ȣ       �	��`�Yc�A�"*

loss�J<��F"       �	,Ea�Yc�A�"*

loss���<Q��       �	��a�Yc�A�"*

loss��2<�_ؼ       �	��b�Yc�A�"*

loss���;��P       �	�2c�Yc�A�"*

lossN��;M>�B       �	��c�Yc�A�"*

loss���<�M[       �	��d�Yc�A�"*

lossor�=��G!       �	'�e�Yc�A�"*

loss�X=���:       �	6;f�Yc�A�"*

lossG�<
fh�       �	-	g�Yc�A�"*

lossq�A=�I!�       �	��g�Yc�A�"*

loss�c�=��i�       �	�Ph�Yc�A�"*

lossD�Y=6�<       �	p�h�Yc�A�"*

loss<z	=E��^       �	��i�Yc�A�"*

loss���;���       �	�%j�Yc�A�"*

lossxk�:�0ƙ       �	��j�Yc�A�"*

loss�<B�Q       �	bk�Yc�A�"*

loss,��<�x�@       �	��k�Yc�A�"*

lossHqe=\gb       �	��l�Yc�A�"*

loss�K=Oӊ       �	�@m�Yc�A�"*

loss�k�<��x       �	Y�m�Yc�A�"*

loss��=�f       �	J~n�Yc�A�"*

loss�+�<�ue^       �	�o�Yc�A�"*

lossC�H<4��       �	��o�Yc�A�"*

loss>��N�       �	7Sp�Yc�A�"*

loss�C�<���       �	��p�Yc�A�"*

loss�=��j       �	�q�Yc�A�"*

loss���<�p��       �	�'r�Yc�A�"*

losssw�<M��       �	e�r�Yc�A�"*

losse�&=B?tC       �	�\s�Yc�A�"*

loss4��<d�(       �	�s�Yc�A�"*

lossw`<>��G       �	��t�Yc�A�"*

loss/w�<�4�       �	�0u�Yc�A�"*

loss�+<��p       �	C�u�Yc�A�"*

lossC<PE�       �	9cv�Yc�A�"*

loss9=.���       �	��v�Yc�A�"*

loss�7=9�K1       �	��w�Yc�A�"*

lossH�#=<��a       �	�@x�Yc�A�"*

loss���=l��I       �	��x�Yc�A�"*

loss���=0��       �	Lny�Yc�A�"*

loss�i�<��*�       �	=z�Yc�A�"*

loss�;�;� ��       �	r�z�Yc�A�"*

loss��K<ܫM`       �	�={�Yc�A�"*

lossf�;=�&
       �	��{�Yc�A�"*

loss��=����       �	�q|�Yc�A�"*

loss��<ţ�       �	[}�Yc�A�"*

lossB7=�mm       �	��}�Yc�A�"*

loss:	B=b��       �	
L~�Yc�A�"*

loss}
4<���       �	TS�Yc�A�"*

loss��7=���}       �	���Yc�A�"*

losso`E=̵�       �	섀�Yc�A�"*

loss�_<e_��       �	�)��Yc�A�"*

loss�jz<Z�ţ       �	�΁�Yc�A�"*

loss_w*<gf�       �	]m��Yc�A�"*

lossX�H<1\&       �	 ��Yc�A�"*

lossh��<4 �       �	����Yc�A�"*

loss�ֺ=U�$�       �	�D��Yc�A�"*

loss�f�<0}�b       �	�݄�Yc�A�"*

loss:�=�O�       �	s��Yc�A�"*

loss �=�2,q       �	���Yc�A�"*

loss��p<��nl       �	���Yc�A�"*

loss�,�<=���       �	�Ç�Yc�A�"*

loss#�=K&       �	�i��Yc�A�"*

lossN7�<��       �	���Yc�A�"*

loss�.=E`       �	C���Yc�A�"*

loss��=�V�       �	�L��Yc�A�"*

loss���<��=       �	)슞Yc�A�"*

loss\�<��       �	�ڋ�Yc�A�"*

lossd4<.��=       �	5{��Yc�A�"*

loss���<���       �	"��Yc�A�"*

loss�i^<II�k       �	����Yc�A�"*

lossLrA=P�m       �	fO��Yc�A�"*

lossԮn=��h�       �	�펞Yc�A�"*

loss�&=���N       �	H���Yc�A�"*

lossU�<�]�7       �	>?��Yc�A�"*

loss�:�;y[k       �	<ܐ�Yc�A�"*

lossm�8<!��)       �	�x��Yc�A�"*

loss=��<��,       �	��Yc�A�"*

loss�C�<�V�       �	{���Yc�A�"*

loss��A=k�P       �	i9��Yc�A�"*

loss�.�<p�X       �	�擞Yc�A�"*

loss
�=i1�r       �	܁��Yc�A�"*

loss���<���       �	�N��Yc�A�"*

loss��=6ƨB       �	�啞Yc�A�"*

loss���<1=��       �	N|��Yc�A�"*

loss�ns=QN�       �	���Yc�A�"*

loss�P�<�&i�       �	1���Yc�A�"*

loss��=j���       �	<K��Yc�A�"*

loss>�<N~�)       �	����Yc�A�"*

loss�M=T��C       �	ߊ��Yc�A�"*

loss͐=�k�       �	�$��Yc�A�"*

loss�p	=3M�       �	j���Yc�A�"*

loss�<���~       �	iU��Yc�A�"*

lossQTb=�h�|       �	�웞Yc�A�"*

loss��{<�)F       �	"���Yc�A�"*

loss���<�C6O       �	m���Yc�A�"*

loss\�=�^��       �	R(��Yc�A�"*

lossn|t<�>z�       �	����Yc�A�"*

loss�< ���       �	tb��Yc�A�"*

loss89�=2I�       �	���Yc�A�"*

loss�==��:�       �	3���Yc�A�#*

loss�i�;��g�       �	V,��Yc�A�#*

loss��1<_�Z       �	���Yc�A�#*

lossc< KBb       �	J���Yc�A�#*

loss %U<���       �	�I��Yc�A�#*

loss���<���       �	iᤞYc�A�#*

lossBȓ<џ��       �	�w��Yc�A�#*

loss��=5#��       �	hZ��Yc�A�#*

loss�c2<�m       �	;:��Yc�A�#*

loss	�~<�\�       �	
֧�Yc�A�#*

loss�^[;v�>P       �	`���Yc�A�#*

lossd�;�uS       �	a3��Yc�A�#*

loss���<X��m       �	�˩�Yc�A�#*

loss&;��-       �	h��Yc�A�#*

loss��;����       �	?��Yc�A�#*

loss�QF<�j�       �	
���Yc�A�#*

losst^0=�E�l       �	ji��Yc�A�#*

loss�Hf<M�        �	v���Yc�A�#*

lossŇR=#�Ki       �	���Yc�A�#*

loss��!=;<�u       �	�4��Yc�A�#*

loss}�-=�'޴       �	ʮ�Yc�A�#*

lossa�m=㡤�       �	�c��Yc�A�#*

loss�W�<ki       �	B	��Yc�A�#*

loss��H<r3��       �	+���Yc�A�#*

lossz��<Wߑf       �	�H��Yc�A�#*

loss4�=P|       �	�뱞Yc�A�#*

loss�F�=���       �	9���Yc�A�#*

loss��X<C�z       �	�6��Yc�A�#*

lossI=F��W       �	�߳�Yc�A�#*

loss�[=�(�{       �	dy��Yc�A�#*

loss��P=��&       �	y��Yc�A�#*

loss��<4ʎ�       �	˺��Yc�A�#*

loss��'=��S       �	l[��Yc�A�#*

lossLF-=h��       �	���Yc�A�#*

lossl\�;d�p       �	����Yc�A�#*

loss��<8"*       �	�Q��Yc�A�#*

loss�=	X�4       �	:��Yc�A�#*

loss���<�k��       �	����Yc�A�#*

loss�=ҕ��       �	T��Yc�A�#*

loss��*= 8�       �	���Yc�A�#*

loss�=%�1�       �	����Yc�A�#*

loss1��<�f��       �	�s��Yc�A�#*

loss�e�=*E��       �	�0��Yc�A�#*

loss�w"=%9�H       �	c～Yc�A�#*

loss�*o<$�x�       �	����Yc�A�#*

loss��K=F�l       �	�l��Yc�A�#*

loss,g=E��6       �	�P��Yc�A�#*

loss��<�}Ur       �	b���Yc�A�#*

lossDY@=s       �	C���Yc�A�#*

loss���<=�n:       �	{3ÞYc�A�#*

loss���<7i�       �	��ÞYc�A�#*

lossݐ�<���,       �	�ĞYc�A�#*

loss���<$4'f       �	�ŞYc�A�#*

loss��7;����       �	�KƞYc�A�#*

loss�n=��"       �	��ǞYc�A�#*

loss�Ƹ=�n�       �	DkȞYc�A�#*

loss���<�ґ       �	�ɞYc�A�#*

loss�3=�E}�       �	��ɞYc�A�#*

lossE�=$.P�       �	ȖʞYc�A�#*

lossಮ<0!*�       �	U.˞Yc�A�#*

loss���;�2-       �	X�˞Yc�A�#*

lossmV6;�`��       �	@�̞Yc�A�#*

loss��Q<���@       �	&͞Yc�A�#*

losswk�=�ۣ$       �	�͞Yc�A�#*

loss�j
=�N�       �	�lΞYc�A�#*

loss�]=:O       �		ϞYc�A�#*

loss;�|=DA�       �	(�ϞYc�A�#*

lossN�<^��       �	�ОYc�A�#*

loss��=s��       �	��ўYc�A�#*

loss=�D=��J�       �	G<ҞYc�A�#*

loss�+W=w��a       �	1	ӞYc�A�#*

loss�y9<E��|       �	��ӞYc�A�#*

loss��=��/       �	C=ԞYc�A�#*

lossAL*=�I�       �	M�ԞYc�A�#*

loss%�=��<}       �	��՞Yc�A�#*

lossh�<���       �	�֞Yc�A�#*

loss�d=�xb"       �	��֞Yc�A�#*

losso�<;7�v       �	*TמYc�A�#*

lossO��;���       �	_�מYc�A�#*

loss���<Z�h       �	�؞Yc�A�#*

loss�~b=��ih       �	�PٞYc�A�#*

loss{�^=�9��       �	�ٞYc�A�#*

loss<�xq       �	��ڞYc�A�#*

loss`�=���       �	\۞Yc�A�#*

lossݨ\=s�x~       �	(�۞Yc�A�#*

loss�\<�0��       �	iQܞYc�A�#*

loss�y=�-H       �	��ܞYc�A�#*

lossf��=��Gr       �	�ݞYc�A�#*

loss�7=OY��       �	:ޞYc�A�#*

lossa�p<�=�       �	>�ޞYc�A�#*

loss��"=���       �	LߞYc�A�#*

loss��o=�T��       �	!�ߞYc�A�#*

loss6�u=�`�h       �	�|��Yc�A�#*

loss�h�<�Ne       �	h"�Yc�A�#*

lossإW<� �z       �	#��Yc�A�#*

loss�61=��       �	}X�Yc�A�#*

lossZG\=3�ٛ       �	��Yc�A�#*

loss:'L= C8       �	��Yc�A�#*

loss��=�rs�       �	K�Yc�A�#*

loss�V=V��       �	:��Yc�A�#*

lossc=3�3�       �	{��Yc�A�#*

loss�g�<xb6       �	V)�Yc�A�#*

loss2�<�U       �	���Yc�A�#*

lossj�f=^S,       �	od�Yc�A�#*

loss1N�;~�QK       �	��Yc�A�#*

loss� u<��       �	+��Yc�A�#*

loss.�=׼�        �	�;�Yc�A�#*

loss@��<_g�       �	���Yc�A�#*

lossr��<�.�:       �	Po�Yc�A�#*

loss��<�@�       �	��Yc�A�#*

lossL�
<���2       �	&��Yc�A�#*

loss��X<ˀ��       �	���Yc�A�#*

loss�<�W{�       �	�|�Yc�A�#*

loss��<()�       �	Y�Yc�A�#*

loss)�<|�I       �	�*��Yc�A�#*

loss��O=�=       �	O��Yc�A�#*

loss��= y       �	���Yc�A�#*

loss��'<]*T       �	^f��Yc�A�#*

lossE<�< ��`       �	P��Yc�A�#*

loss���; B�       �	R���Yc�A�#*

lossfu<�p       �	�k��Yc�A�#*

lossf'�;o�P       �	K��Yc�A�#*

loss.=qI��       �	���Yc�A�#*

loss��<Xh7       �	S��Yc�A�#*

loss�n4<�|�       �	Q���Yc�A�#*

lossZ�=�$�       �	0���Yc�A�#*

loss$&<gf�       �	G��Yc�A�#*

lossnv=Eɥ       �	����Yc�A�#*

loss��=J���       �	z���Yc�A�$*

lossO�=����       �	�4��Yc�A�$*

loss���<�       �	���Yc�A�$*

lossS�=��&�       �	�w �Yc�A�$*

loss�̇=%=pq       �	��Yc�A�$*

loss6�*=.a�h       �	v��Yc�A�$*

lossFƀ=�x�~       �	�?�Yc�A�$*

loss��y=@VK�       �	H��Yc�A�$*

loss$�M=�F�       �	���Yc�A�$*

loss�"�;��z       �	.T�Yc�A�$*

loss�ּ< ��       �	���Yc�A�$*

loss�?�<�7R^       �	=��Yc�A�$*

lossF��=6w<�       �	c��Yc�A�$*

loss���<D��       �	�T�Yc�A�$*

loss�i�=�)6*       �	k��Yc�A�$*

lossn;�<w��       �	Y�	�Yc�A�$*

lossiW�=�@��       �	5)
�Yc�A�$*

lossMRe=�:       �	�/�Yc�A�$*

loss\��<H7)       �	:��Yc�A�$*

loss#��<�JU�       �	��Yc�A�$*

loss�^i=?��1       �	|C�Yc�A�$*

loss��'=n.��       �	��Yc�A�$*

lossW'=xOs�       �	�w�Yc�A�$*

loss]x5=�@��       �	s�Yc�A�$*

loss��K=ב��       �	��Yc�A�$*

loss�#<��|       �	���Yc�A�$*

loss�G�<�ʱ'       �	H�Yc�A�$*

loss�:�<Gw       �	���Yc�A�$*

lossr\A=��       �	J��Yc�A�$*

loss���<�,�       �	�G�Yc�A�$*

loss�(<\�	       �	�C�Yc�A�$*

lossQ��<'Fŉ       �	��Yc�A�$*

loss���<w�Q�       �	���Yc�A�$*

loss�8�=nk�       �	�P�Yc�A�$*

loss]��<!�H�       �	���Yc�A�$*

lossl��=�.�       �	���Yc�A�$*

loss��=�84       �	(-�Yc�A�$*

loss��"<��       �	��Yc�A�$*

loss|U|<��D�       �	�`�Yc�A�$*

lossR{�=����       �	 ��Yc�A�$*

loss��u<�p�       �	9��Yc�A�$*

loss_��<�\�       �	�1�Yc�A�$*

loss@�<@g�(       �	 ��Yc�A�$*

loss@*�=q��Q       �	Re�Yc�A�$*

loss�{q<�u�i       �	�<�Yc�A�$*

loss�^�;E�kQ       �	���Yc�A�$*

loss_*=Jl�       �	�y�Yc�A�$*

lossi�+=���       �	�Yc�A�$*

loss���=��=       �	��Yc�A�$*

lossMs=P�*F       �	O� �Yc�A�$*

lossJr=�T<C       �	�8!�Yc�A�$*

loss&'8<̒�       �	�!�Yc�A�$*

loss�'=2#>�       �	�o"�Yc�A�$*

lossd��;�#,       �	#�Yc�A�$*

loss�]z<���0       �	�#�Yc�A�$*

loss,�;��ͅ       �	`X$�Yc�A�$*

loss�K<=��	       �	z�$�Yc�A�$*

lossJպ<�ވ,       �	ݕ%�Yc�A�$*

lossU(<�3��       �	=D&�Yc�A�$*

loss?7g<���
       �	'�Yc�A�$*

loss8�;?H.       �	�'�Yc�A�$*

lossQAI=�k(       �	~r(�Yc�A�$*

loss���<�g       �	��)�Yc�A�$*

loss�'=��qV       �	̛*�Yc�A�$*

lossB�<�h�       �	3�+�Yc�A�$*

loss\�.=�>�       �	;�,�Yc�A�$*

loss(�<��       �	 �-�Yc�A�$*

loss7��;��e       �	2�.�Yc�A�$*

lossZي<�1�       �	�0�Yc�A�$*

lossr��;�zj�       �	(
1�Yc�A�$*

lossqJ�<��       �	�2�Yc�A�$*

loss�Xe<����       �	A�2�Yc�A�$*

loss�j�=W��v       �	ץ3�Yc�A�$*

loss
Ȟ=�(;�       �	��4�Yc�A�$*

losst2�;��       �	V-6�Yc�A�$*

loss@��<��ۢ       �	�7�Yc�A�$*

loss�J="G+�       �	.�7�Yc�A�$*

loss�=L.@�       �	/k8�Yc�A�$*

lossM=�R��       �	�99�Yc�A�$*

loss��;����       �	�
:�Yc�A�$*

loss�E�<p�C�       �	�I;�Yc�A�$*

loss`�;����       �	��<�Yc�A�$*

loss�H�<-n��       �	�y=�Yc�A�$*

loss�I�<����       �	P>�Yc�A�$*

loss�v�<~a-�       �	��>�Yc�A�$*

loss�;�yn       �	NG@�Yc�A�$*

loss�$�<Ӛ"       �	hzA�Yc�A�$*

lossئ=�SF[       �	B�Yc�A�$*

lossѕN<y��k       �	�XC�Yc�A�$*

loss	6�<X�Xu       �	�FD�Yc�A�$*

lossI�<�.��       �	�3E�Yc�A�$*

loss@�,<s#@       �	p�E�Yc�A�$*

loss=m�<r��       �	wiF�Yc�A�$*

loss2+&=��"       �	�G�Yc�A�$*

lossZS<�٨       �	��G�Yc�A�$*

lossTU<�N�       �	;7H�Yc�A�$*

loss�}<oa�       �	��H�Yc�A�$*

lossR��<�RQ       �	{I�Yc�A�$*

loss���<�Z:       �	V,J�Yc�A�$*

loss��<���N       �	@�J�Yc�A�$*

loss�`�<0��       �	�K�Yc�A�$*

loss!P�=� M�       �	�QL�Yc�A�$*

loss��`<2��&       �	
�L�Yc�A�$*

loss��E<���t       �	ڏM�Yc�A�$*

loss'=����       �	�#N�Yc�A�$*

loss�;0W�X       �	��N�Yc�A�$*

loss?{�=E��n       �	�aO�Yc�A�$*

lossc95<t@�       �	3�O�Yc�A�$*

lossÝN=�x=       �	�P�Yc�A�$*

loss(�<Q�N�       �	 *Q�Yc�A�$*

loss��J<θP�       �		�Q�Yc�A�$*

loss@��<����       �	[]R�Yc�A�$*

loss��<p�S       �	�
S�Yc�A�$*

loss���;����       �	��S�Yc�A�$*

loss]$<��e�       �	�aT�Yc�A�$*

loss�l�;��aM       �	��T�Yc�A�$*

lossS_"=��F       �	1�U�Yc�A�$*

loss�B.=��(�       �	]2V�Yc�A�$*

loss?�<d�%K       �	��V�Yc�A�$*

loss��;�9�       �	(~W�Yc�A�$*

loss��!=�       �	�X�Yc�A�$*

lossip�<��1       �	��X�Yc�A�$*

lossC��<F��&       �	�]Y�Yc�A�$*

loss29 =���k       �	;�Y�Yc�A�$*

loss:�=c���       �	��Z�Yc�A�$*

loss.)<���e       �	)_[�Yc�A�$*

loss?Υ9���        �	n�[�Yc�A�$*

loss�M�;��       �	%�\�Yc�A�$*

loss���;-@�1       �	B&]�Yc�A�%*

loss���;����       �	��]�Yc�A�%*

loss���;-9��       �	XU^�Yc�A�%*

loss&;�<�Ba\       �	��^�Yc�A�%*

loss��=���       �	{_�Yc�A�%*

lossՃ<�t��       �	`�Yc�A�%*

loss�ƨ9�h�z       �	��`�Yc�A�%*

loss퀾:���       �	,Ia�Yc�A�%*

loss�<Ie��       �	��a�Yc�A�%*

loss �p<1�k�       �	`wb�Yc�A�%*

loss4�<���       �	Vc�Yc�A�%*

loss��W;����       �	�c�Yc�A�%*

loss=��<��4�       �	$Dd�Yc�A�%*

loss2��=̓��       �	��d�Yc�A�%*

loss�XG;�e��       �	tze�Yc�A�%*

loss4�;c��       �	�(f�Yc�A�%*

loss�1�<A�       �	9�f�Yc�A�%*

loss���<A�       �	k'h�Yc�A�%*

loss��d<k�2�       �	+�h�Yc�A�%*

loss�b�;̥��       �	$�i�Yc�A�%*

loss}�<�T�       �	��j�Yc�A�%*

loss�i�=��&       �	2�k�Yc�A�%*

loss�==��7       �	�ml�Yc�A�%*

lossd7<�z�       �	om�Yc�A�%*

loss��;�9�{       �	��m�Yc�A�%*

lossV[�=�M��       �	�o�Yc�A�%*

loss Qu=6���       �	xp�Yc�A�%*

lossS�1=G4�g       �	��p�Yc�A�%*

lossQP=W��u       �	rNq�Yc�A�%*

loss�vI=�._A       �	_�q�Yc�A�%*

loss$= ��       �	i�r�Yc�A�%*

loss,9S=���n       �	R,s�Yc�A�%*

loss��<�S�V       �	��s�Yc�A�%*

loss�=�OA�       �	sit�Yc�A�%*

loss�'\;�Pٍ       �	u�Yc�A�%*

loss�f�<����       �	��u�Yc�A�%*

lossT�;��$       �	9Dv�Yc�A�%*

loss��;'���       �	<�v�Yc�A�%*

loss�;�`�       �	�ww�Yc�A�%*

loss��;p�2%       �	�x�Yc�A�%*

loss=)^�       �	>�x�Yc�A�%*

loss
<�2Q.       �	j�y�Yc�A�%*

lossF=�hX       �	Cz�Yc�A�%*

lossnC1=����       �	�z�Yc�A�%*

loss��<�N��       �	�N{�Yc�A�%*

lossI,�<h#ػ       �	��{�Yc�A�%*

loss;d�<��ݞ       �	�{|�Yc�A�%*

loss���:���       �	}�Yc�A�%*

lossi�<�$�       �	�}�Yc�A�%*

loss���<����       �	�D~�Yc�A�%*

loss(n<v��'       �	r�~�Yc�A�%*

loss�:/=>%�-       �	�u�Yc�A�%*

loss�1�=`S��       �	5��Yc�A�%*

loss��<y-�       �	����Yc�A�%*

loss�]�;Y�(;       �	aR��Yc�A�%*

loss*�;���       �	$���Yc�A�%*

lossw'<&��       �	G���Yc�A�%*

lossA(q=?ߜ�       �	Q���Yc�A�%*

loss;5<�S�j       �	k`��Yc�A�%*

lossE�<u���       �	U3��Yc�A�%*

lossaBC=͍V�       �	`Ʌ�Yc�A�%*

loss��V<V��h       �	@m��Yc�A�%*

lossԬB=�"�       �	L��Yc�A�%*

loss��<ia�       �	�營Yc�A�%*

loss
ɩ<37J       �	���Yc�A�%*

loss% <���       �	7���Yc�A�%*

loss��[=���1       �	�`��Yc�A�%*

loss�N�=I!�?       �	P���Yc�A�%*

loss]FN<�ugu       �	W���Yc�A�%*

loss�\R<�(       �	�,��Yc�A�%*

losswWh<��w�       �	�â�Yc�A�%*

lossۀ<#�I       �	�S��Yc�A�%*

lossd�=/[?�       �	����Yc�A�%*

loss�;<p:       �	E���Yc�A�%*

loss��=����       �	@��Yc�A�%*

lossZ��;��       �	Y���Yc�A�%*

loss��=Ԏ�	       �	�>��Yc�A�%*

loss���<�[�@       �	i��Yc�A�%*

loss�a�=+��       �	U���Yc�A�%*

loss7ZR=��sW       �	*ਟYc�A�%*

loss��"<~=�{       �	�z��Yc�A�%*

loss�2';oHv�       �	U0��Yc�A�%*

loss�D�;��K       �	G:��Yc�A�%*

loss�m1=ւL�       �	Qګ�Yc�A�%*

loss��-=�i~       �	܀��Yc�A�%*

lossf�><t�       �	���Yc�A�%*

loss��<Dѕ       �	V���Yc�A�%*

loss�ǜ;%�q       �	vO��Yc�A�%*

loss���=^�ׇ       �	�讟Yc�A�%*

lossG=�       �	���Yc�A�%*

lossR�;}HB       �	Q��Yc�A�%*

loss��G=��V�       �	:���Yc�A�%*

loss�"�;�+�p       �	F��Yc�A�%*

loss�b�<W霛       �	�걟Yc�A�%*

loss���<(���       �	i���Yc�A�%*

loss�T�<F�
       �	9)��Yc�A�%*

lossѠ�<�-�       �	�γ�Yc�A�%*

lossd�<���       �	we��Yc�A�%*

loss��B=~�(>       �	J��Yc�A�%*

loss@�d<W�.       �	j���Yc�A�%*

lossiU�<�       �	6>��Yc�A�%*

lossn1�;=�x       �	�׶�Yc�A�%*

lossj�<6�sB       �	~��Yc�A�%*

loss}��=8UV�       �	D��Yc�A�%*

loss�0=��)�       �	B���Yc�A�%*

loss���<E�P       �	����Yc�A�%*

lossx#�<���       �	���Yc�A�%*

loss�t<� m�       �	vº�Yc�A�%*

loss@L�<�؟�       �	j��Yc�A�%*

loss��q=����       �	.���Yc�A�%*

loss,�U=9h�Z       �	F���Yc�A�%*

loss]&�;�%n       �	h<��Yc�A�%*

loss�M<�_Y<       �	xҽ�Yc�A�%*

loss�D�<H���       �	�|��Yc�A�%*

loss<��;�N��       �	%"��Yc�A�%*

loss�=����       �	���Yc�A�%*

loss}[=��*�       �	J���Yc�A�%*

loss=�;��ߖ       �	���Yc�A�%*

loss8n�=�s�       �	Z.Yc�A�%*

loss㠋<)��-       �	��Yc�A�%*

loss���;����       �	�xßYc�A�%*

loss��;����       �	�ğYc�A�%*

loss\�<��G�       �	�ğYc�A�%*

loss��<p��       �	,HşYc�A�%*

loss�=��g�       �	��şYc�A�%*

loss��<�S�       �	D�ƟYc�A�%*

lossR9s=߹o�       �	�NǟYc�A�%*

lossT�#;��<�       �	1�ǟYc�A�%*

loss�h<Et�       �	��ȟYc�A�&*

loss$}%<���e       �	�*ɟYc�A�&*

loss���;��-       �	��ɟYc�A�&*

loss�cO<��       �	dʟYc�A�&*

loss�G�<���       �	�ʟYc�A�&*

loss�=M�,�       �	)�˟Yc�A�&*

loss�=��Qs       �	v7̟Yc�A�&*

loss
Y =r��0       �	��̟Yc�A�&*

loss}wZ=��Ԡ       �	j͟Yc�A�&*

lossl�(=G�kb       �	�ΟYc�A�&*

loss���<o)~>       �	��ΟYc�A�&*

loss�Nk=/�       �	+5ϟYc�A�&*

lossr �<21�       �	��ϟYc�A�&*

loss[�<��ޙ       �	ͮПYc�A�&*

loss�׆;^�5       �	�FџYc�A�&*

loss�D<Vj       �	l�џYc�A�&*

loss��k<�,�       �	�ҟYc�A�&*

loss�l=��       �	GӟYc�A�&*

losst�<{��3       �	5�ӟYc�A�&*

losse$�<V~       �	CWԟYc�A�&*

loss�j�:�#_)       �	(�ԟYc�A�&*

loss�08=r�`       �	��՟Yc�A�&*

loss�{<o�oP       �	�g֟Yc�A�&*

loss�+=�1Զ       �	��֟Yc�A�&*

loss���<�?       �	_�ןYc�A�&*

loss��<��s       �	�3؟Yc�A�&*

loss���<�-�c       �	��؟Yc�A�&*

loss&�B=_.       �	iٟYc�A�&*

lossC_�<�ǣ�       �	�ڟYc�A�&*

lossE-_<2���       �	1�ڟYc�A�&*

lossj�<<~ H�       �	�O۟Yc�A�&*

lossI�?=��R�       �	��۟Yc�A�&*

lossMHO;b���       �	}�ܟYc�A�&*

loss�==R�       �	�8ݟYc�A�&*

loss�;<���       �	��ݟYc�A�&*

loss|p=waa=       �	]kޟYc�A�&*

loss�p;<�aÆ       �	EߟYc�A�&*

loss��<�u�       �	�ߟYc�A�&*

loss��(=;s�       �	I��Yc�A�&*

loss�(=߃��       �	����Yc�A�&*

losssj�;��'P       �	�P�Yc�A�&*

loss�gO<�_�       �	N��Yc�A�&*

loss=�<=
҄       �	2��Yc�A�&*

loss,�;̟��       �	X8�Yc�A�&*

loss7mP<,2e       �	���Yc�A�&*

lossT�:=�R,J       �	�r�Yc�A�&*

loss�6:;��B�       �	��Yc�A�&*

loss`0=c�&�       �	��Yc�A�&*

loss�A}<G?7I       �	|e�Yc�A�&*

lossi"<�Q��       �	��Yc�A�&*

loss7�I;��cQ       �	[��Yc�A�&*

lossB��<KE       �	%Z�Yc�A�&*

lossn��<���       �	���Yc�A�&*

loss�}�<��@�       �	>��Yc�A�&*

loss��n=e#��       �	V+�Yc�A�&*

lossCI=�?�[       �	���Yc�A�&*

loss|�<sFa�       �	�_�Yc�A�&*

loss&�p=$�       �	  �Yc�A�&*

loss�&�=P�       �	��Yc�A�&*

loss ��<��h       �	p#�Yc�A�&*

lossT�=�|�l       �	#��Yc�A�&*

loss�u�;����       �	SY�Yc�A�&*

loss��X=�B��       �	L��Yc�A�&*

loss�,�<u)       �	Н�Yc�A�&*

loss/�<,+�r       �	�?�Yc�A�&*

loss��9=QS�o       �	���Yc�A�&*

loss�|M<0���       �	�B��Yc�A�&*

lossh�<�ۍi       �	����Yc�A�&*

loss��<a���       �	����Yc�A�&*

lossv�z<՘��       �	�*��Yc�A�&*

loss�<�;�       �	�$��Yc�A�&*

loss�=�       �	����Yc�A�&*

loss�9�<�G��       �	s���Yc�A�&*

lossg�<���i       �	�>��Yc�A�&*

loss-r<��I�       �	f���Yc�A�&*

loss��<�"�       �	 y��Yc�A�&*

loss���<t+t       �	z��Yc�A�&*

loss��;�h       �	)���Yc�A�&*

lossO5)<��'       �	nO��Yc�A�&*

loss
Z�=�zie       �	����Yc�A�&*

loss��V=/��w       �	����Yc�A�&*

lossFY�<}�-n       �	q<��Yc�A�&*

loss�l�;�C,�       �	����Yc�A�&*

loss���<���I       �	mp��Yc�A�&*

lossa�w;QL~�       �	Q �Yc�A�&*

loss�7;��2       �	ø �Yc�A�&*

losso�%=�!}       �	<f�Yc�A�&*

loss88=��Q       �	ut�Yc�A�&*

loss ��<F3M       �	�	�Yc�A�&*

lossD\�=�f��       �	��Yc�A�&*

loss�	C=U���       �	�?�Yc�A�&*

loss��c=�n��       �	��Yc�A�&*

lossl�w<ip�       �	s�Yc�A�&*

loss�B<+_sd       �	�)�Yc�A�&*

lossRM�=S��       �	���Yc�A�&*

losssq6=�J�       �	�o�Yc�A�&*

lossT��=���       �	I�Yc�A�&*

loss�
=��       �	�
	�Yc�A�&*

loss�<� �       �	1�	�Yc�A�&*

loss�R.=.�b�       �	�^
�Yc�A�&*

loss�G�;n�t       �	�Yc�A�&*

loss6 N=G�X[       �	
��Yc�A�&*

lossE�<�c�       �	gE�Yc�A�&*

losst�<��       �	���Yc�A�&*

loss�*�<)d��       �	���Yc�A�&*

loss�F�<��\P       �	@0�Yc�A�&*

loss�FT=��       �	EI�Yc�A�&*

lossa�<�z��       �	G��Yc�A�&*

lossw��;o-�a       �	{��Yc�A�&*

lossk�<}0��       �	T�Yc�A�&*

loss��s=	�       �	���Yc�A�&*

lossz�<8��       �	�R�Yc�A�&*

loss���<�W�]       �	���Yc�A�&*

loss�P�<y@�       �	��Yc�A�&*

loss��?=�ָ       �	)<�Yc�A�&*

lossxɓ<H�V       �	'��Yc�A�&*

loss�b~<Lu1�       �	|{�Yc�A�&*

lossE�0<M#Ɲ       �	�/�Yc�A�&*

lossMOS=���       �	(��Yc�A�&*

lossL\�;���       �	�{�Yc�A�&*

loss��<4�       �	��Yc�A�&*

lossa�2<SA       �	 ��Yc�A�&*

losss:�;+u'�       �	�N�Yc�A�&*

loss�E=h��       �	���Yc�A�&*

loss��"=�"�       �	��Yc�A�&*

loss!E�<���R       �	I/�Yc�A�&*

loss�%H=I8��       �	���Yc�A�&*

loss)se=ƨ�v       �	Jy�Yc�A�&*

loss}Z�<��       �	� �Yc�A�'*

loss
�=$]�       �	���Yc�A�'*

lossN"�;�i��       �	d�Yc�A�'*

lossa�g;�x�       �	T�Yc�A�'*

loss�«<Љ��       �	��Yc�A�'*

lossc�=��       �	IJ �Yc�A�'*

lossr�+=n���       �	�� �Yc�A�'*

loss��H=]B       �	.�!�Yc�A�'*

loss/�=m��       �	�,"�Yc�A�'*

loss_%�<IN�       �	��"�Yc�A�'*

loss�2<��'       �	�e#�Yc�A�'*

loss���<	�VI       �	P $�Yc�A�'*

loss/<5��E       �	<�$�Yc�A�'*

loss��;��˧       �	\=%�Yc�A�'*

loss*��<''+�       �	�%�Yc�A�'*

lossV!=%�F       �	�o&�Yc�A�'*

lossf�;���       �	#'�Yc�A�'*

lossu=L�I       �	��'�Yc�A�'*

loss�:�<���       �	��(�Yc�A�'*

loss`�M=nv*s       �	��)�Yc�A�'*

loss���<"z�D       �	�g*�Yc�A�'*

loss��<��r       �	�+�Yc�A�'*

loss�:<'�R�       �	�,�Yc�A�'*

loss��=wm�       �	�-�Yc�A�'*

loss�*�=����       �	2<.�Yc�A�'*

loss�U=�R�~       �	*T/�Yc�A�'*

loss��'=^]K�       �	�I0�Yc�A�'*

loss�kz=��[       �	$1�Yc�A�'*

loss��B=�k       �	�K2�Yc�A�'*

loss4�<�,{       �	�"3�Yc�A�'*

loss�z<yO<�       �	p�3�Yc�A�'*

loss���<���X       �	F�4�Yc�A�'*

lossO={U�       �	��5�Yc�A�'*

loss���<B��       �	y�6�Yc�A�'*

lossϾ=����       �	֐7�Yc�A�'*

loss��<}5>4       �	�Q8�Yc�A�'*

loss)�<>G{N       �	�H9�Yc�A�'*

loss@��<�H6�       �	* :�Yc�A�'*

lossQ��<<�x�       �	D�:�Yc�A�'*

loss160=�t�o       �	>�;�Yc�A�'*

loss�W�;����       �	%Y<�Yc�A�'*

losssc=E�       �	�^=�Yc�A�'*

loss�4<C?P�       �	�>�Yc�A�'*

lossW�0=;>�       �	
�>�Yc�A�'*

loss��l=c���       �	�>?�Yc�A�'*

loss(��<�ݑ       �	��?�Yc�A�'*

loss�p�<|.�M       �	�r@�Yc�A�'*

loss��e=��{a       �	�A�Yc�A�'*

loss�(}<�֘H       �	ޮA�Yc�A�'*

loss���;���w       �	CUB�Yc�A�'*

lossV��<=��'       �	�B�Yc�A�'*

loss�
O<~h�g       �	��C�Yc�A�'*

lossT�=
�       �	�"D�Yc�A�'*

loss�ˈ=t�g       �	غD�Yc�A�'*

lossF�=���       �	j�E�Yc�A�'*

lossf�a<'6�       �	%�F�Yc�A�'*

lossN��<B<pt       �	�8G�Yc�A�'*

lossh��<���       �	��G�Yc�A�'*

loss��=�ݍu       �	jlH�Yc�A�'*

loss��*=���       �	5%I�Yc�A�'*

loss��&<Ii��       �	�I�Yc�A�'*

loss��=��       �	VJ�Yc�A�'*

loss���;0-9       �	U�J�Yc�A�'*

loss�+�:����       �	�K�Yc�A�'*

loss!:<nQ�R       �	�;L�Yc�A�'*

loss�H�<�t�C       �	�L�Yc�A�'*

loss ^><��+�       �	��M�Yc�A�'*

lossW��<xY]�       �	�QN�Yc�A�'*

loss���<ԣ	�       �	��N�Yc�A�'*

loss.ʪ<��        �	4�O�Yc�A�'*

loss-�l<yQ�e       �	)%P�Yc�A�'*

lossM�<�       �	j�P�Yc�A�'*

loss��}=H�       �	�[Q�Yc�A�'*

loss*��=���       �	�+R�Yc�A�'*

lossu(;��v       �	z�R�Yc�A�'*

loss�H=��qn       �	�gS�Yc�A�'*

lossױ�<��A�       �	��S�Yc�A�'*

loss��=z�Cz       �	��T�Yc�A�'*

lossv�=u$��       �	�EU�Yc�A�'*

lossfro<�j,       �	P�U�Yc�A�'*

loss��<����       �	�V�Yc�A�'*

loss��=?\ F       �	&7W�Yc�A�'*

loss�q�<���       �	��W�Yc�A�'*

loss��=Q*�?       �	v�X�Yc�A�'*

loss���=�s       �	,Y�Yc�A�'*

loss�S<m�&�       �	��Y�Yc�A�'*

loss�wH<{�eo       �	�iZ�Yc�A�'*

lossÊ�<�@�       �	�[�Yc�A�'*

loss�3�<S?^P       �	tE\�Yc�A�'*

losscD�;Z\#x       �	��\�Yc�A�'*

loss �<<�k�_       �	;�]�Yc�A�'*

lossk�<�{��       �	�|^�Yc�A�'*

loss��<F
�       �	�_�Yc�A�'*

loss��*=�#�       �	��_�Yc�A�'*

loss7��<PW��       �	�_`�Yc�A�'*

loss��<���       �	��`�Yc�A�'*

loss�'<a��       �	̗a�Yc�A�'*

loss�=�<.��       �	j4b�Yc�A�'*

lossO��;4F�h       �	��b�Yc�A�'*

loss� 	;���v       �	��c�Yc�A�'*

loss���<�ݺ�       �	-d�Yc�A�'*

lossf�:h6v       �	��d�Yc�A�'*

loss�M]<�{ �       �	Nae�Yc�A�'*

loss��a<(zC#       �	�e�Yc�A�'*

loss��={���       �	�f�Yc�A�'*

lossr��<�j�       �	+�g�Yc�A�'*

loss`��<KS��       �	�9h�Yc�A�'*

loss�a<<7�h�       �	��h�Yc�A�'*

loss���<����       �	,}i�Yc�A�'*

loss��<��D       �	lBj�Yc�A�'*

loss.�;m_Y2       �	��j�Yc�A�'*

loss�9a<�4d       �	�k�Yc�A�'*

loss� =�	��       �	�l�Yc�A�'*

loss��3=?��{       �	�Wm�Yc�A�'*

loss3��=���       �	7�m�Yc�A�'*

loss,s�<��Ϧ       �	t�n�Yc�A�'*

loss��<(\       �	�<o�Yc�A�'*

lossx�5<��O       �	�o�Yc�A�'*

loss�,=���v       �	�wp�Yc�A�'*

losss�;�       �	�q�Yc�A�'*

loss�T=J�_�       �	��q�Yc�A�'*

lossd�U=6�3�       �	`Wr�Yc�A�'*

lossl6<P��       �	s�Yc�A�'*

loss<�;���[       �	��s�Yc�A�'*

losso�<�,�       �	B^t�Yc�A�'*

loss�0�<��(d       �	'�t�Yc�A�'*

loss��<#�x�       �	o�u�Yc�A�'*

lossi<4=��.�       �	m8v�Yc�A�'*

loss��<�,��       �	:�v�Yc�A�(*

loss�Yp<-mP�       �	uw�Yc�A�(*

lossz=��,       �	�x�Yc�A�(*

loss��.=�<�       �	g�x�Yc�A�(*

loss��=��|�       �	Wy�Yc�A�(*

loss�O=If       �	��y�Yc�A�(*

loss���=G�{       �	��z�Yc�A�(*

losslF�<�+�x       �	v2{�Yc�A�(*

loss��$<�m       �	F�{�Yc�A�(*

lossxL�<��p       �	z|�Yc�A�(*

loss�?+=5��j       �	x%}�Yc�A�(*

loss�g�<�p��       �	�}�Yc�A�(*

loss|��<��       �	�j~�Yc�A�(*

lossd�o<c5@�       �	� �Yc�A�(*

lossF�<AV,[       �	���Yc�A�(*

loss�y�=�^M�       �	�<��Yc�A�(*

loss?=}t�       �	�׀�Yc�A�(*

loss}��<`p�       �	�t��Yc�A�(*

loss��<��f       �	kE��Yc�A�(*

loss�&k=]�       �	����Yc�A�(*

loss&ֽ;o�       �	Ҋ��Yc�A�(*

loss&�5;��	�       �	�^��Yc�A�(*

loss�P<u6·       �	����Yc�A�(*

loss��a<��AJ       �	����Yc�A�(*

loss7�{<��       �	VD��Yc�A�(*

losso�#>Y;ο       �	%놠Yc�A�(*

loss�:=f��       �	ʊ��Yc�A�(*

lossz&<�@       �	�)��Yc�A�(*

loss%0B=��}�       �	ƈ�Yc�A�(*

loss}��<}͉       �	Y���Yc�A�(*

loss<;�<u"�       �	�B��Yc�A�(*

loss���;gvu       �	J��Yc�A�(*

loss@:8<��%       �	G���Yc�A�(*

lossS��<?�C�       �	�I��Yc�A�(*

loss�A=�7(�       �	����Yc�A�(*

loss�<y(�m       �	\���Yc�A�(*

lossxE�<_z--       �	�7��Yc�A�(*

lossF��;T؍B       �	�Ύ�Yc�A�(*

loss�cQ<��       �	���Yc�A�(*

loss))/=v�u�       �	Q3��Yc�A�(*

loss���<@�+H       �	�ː�Yc�A�(*

lossr�<9��#       �	�l��Yc�A�(*

lossm��<V�8�       �	��Yc�A�(*

loss&j<8d       �	���Yc�A�(*

loss�#>��[       �	SY��Yc�A�(*

loss�G=]9a�       �	����Yc�A�(*

loss?e=�.�r       �	���Yc�A�(*

loss�*^=�D4       �	�6��Yc�A�(*

lossx�<�n�       �	�애Yc�A�(*

loss��<�m~F       �	����Yc�A�(*

losso�<W���       �	�M��Yc�A�(*

loss��0=�n�       �	�Yc�A�(*

loss��=��]r       �	/���Yc�A�(*

loss�dL=���}       �	�+��Yc�A�(*

lossuY<����       �	!˙�Yc�A�(*

loss�;�<ι)       �	�i��Yc�A�(*

lossGK=�4       �	���Yc�A�(*

loss��=��N       �	瀞�Yc�A�(*

loss���=�p       �	P��Yc�A�(*

lossI�E=���       �	�蜠Yc�A�(*

loss���<[o��       �	
ם�Yc�A�(*

loss�8�<0��       �	�r��Yc�A�(*

loss�Oq=�6w�       �	���Yc�A�(*

lossw�U=��       �	���Yc�A�(*

loss�z�;��0       �	�8��Yc�A�(*

loss��<Upۊ       �	�F��Yc�A�(*

loss�-�<�a       �	�ݡ�Yc�A�(*

lossHY=�`��       �	�u��Yc�A�(*

loss��=�O�       �	���Yc�A�(*

loss2^<�z�E       �	O���Yc�A�(*

lossT�P<����       �	�P��Yc�A�(*

loss�DH==�kj       �	��Yc�A�(*

lossn��<kz��       �	ސ��Yc�A�(*

loss,�<���(       �	1��Yc�A�(*

loss�S)=.z�C       �	)Φ�Yc�A�(*

loss��<��       �	���Yc�A�(*

loss� =�g&N       �	���Yc�A�(*

loss1�#;���       �	u!��Yc�A�(*

loss$�P<���       �	�ĩ�Yc�A�(*

losswn;<��t�       �	�l��Yc�A�(*

loss�%=�Ѱ       �	M��Yc�A�(*

lossEG6<��x       �	ܻ��Yc�A�(*

loss�$�<u�M       �	�Z��Yc�A�(*

loss�Ŵ<�ၱ       �	� ��Yc�A�(*

loss��Q;��       �	����Yc�A�(*

loss�ZC=�և�       �	�I��Yc�A�(*

loss��?=��g�       �	O箠Yc�A�(*

loss%%=�
+�       �	斯�Yc�A�(*

loss��<���0       �	�;��Yc�A�(*

loss��;u�       �	$ְ�Yc�A�(*

loss��=���       �	�o��Yc�A�(*

loss�zz=�8�3       �	���Yc�A�(*

lossa$�<0��       �	����Yc�A�(*

lossEj<��P1       �	�o��Yc�A�(*

loss*�N=�g�       �	 ��Yc�A�(*

loss�==����       �	�Ĵ�Yc�A�(*

loss�F�=�t�s       �	Pn��Yc�A�(*

loss�*�;İ�       �	` ��Yc�A�(*

loss�X<BӞZ       �	M���Yc�A�(*

lossEΞ<���       �	�b��Yc�A�(*

loss�uZ=���       �	-��Yc�A�(*

loss��<CA�       �	����Yc�A�(*

loss1�=,<�       �	�Z��Yc�A�(*

lossNrO;wU�       �	����Yc�A�(*

loss�kh=�ŭ5       �	:���Yc�A�(*

loss��
=���       �	�W��Yc�A�(*

loss��X=��       �	����Yc�A�(*

loss�6<o>�x       �	r���Yc�A�(*

losst<I�\�       �	�Q��Yc�A�(*

loss-�><��ns       �	Yc�A�(*

lossj N=
�       �	y���Yc�A�(*

lossԹh=�B�}       �	9��Yc�A�(*

lossJ=��T       �	�ڿ�Yc�A�(*

loss���;���       �	{��Yc�A�(*

loss :<[�       �	�9��Yc�A�(*

loss[<�<Z�=       �	����Yc�A�(*

loss�CJ=�Uc       �	�� Yc�A�(*

loss��<��P]       �	vàYc�A�(*

loss�'�;_��       �		ĠYc�A�(*

loss�<���       �	��ĠYc�A�(*

lossz2�;�+��       �		�ŠYc�A�(*

loss�R=�K��       �	�`ƠYc�A�(*

loss��=�s��       �	OǠYc�A�(*

loss��=' N       �	H�ǠYc�A�(*

loss��s=�n��       �	[�ȠYc�A�(*

lossru0<��       �	eɠYc�A�(*

loss��=�7�       �	gʠYc�A�(*

loss�=�Q�       �	GYˠYc�A�(*

lossČ=%�       �	̠Yc�A�)*

loss��4=��F       �	�"͠Yc�A�)*

loss�oE=�U5=       �	|eΠYc�A�)*

loss٭<�K��       �	�ϠYc�A�)*

loss\�x;kY�       �	��ϠYc�A�)*

loss�}�;��       �	ZbРYc�A�)*

loss�*�<A�y�       �	v�РYc�A�)*

loss!� =��u�       �	ǛѠYc�A�)*

loss�EG=���       �	�4ҠYc�A�)*

loss��%=;�/C       �	6�ҠYc�A�)*

loss�ΰ<<��       �	��ӠYc�A�)*

loss�<���&       �	&ԠYc�A�)*

lossi=�1�       �	��ԠYc�A�)*

loss1h�;��P�       �	�fՠYc�A�)*

loss֤j<���O       �	�֠Yc�A�)*

loss<"�l        �	0�֠Yc�A�)*

loss�! =�j��       �	�tנYc�A�)*

lossAC�<��<       �	�ؠYc�A�)*

loss���<`S�       �	w�ؠYc�A�)*

lossZ��;�x�       �	~:٠Yc�A�)*

loss%��;�
6�       �	9�٠Yc�A�)*

loss,�o<���%       �	�kڠYc�A�)*

loss�G<Ap�       �	�۠Yc�A�)*

loss�=��'�       �	 �۠Yc�A�)*

lossl�g<�H��       �	@OܠYc�A�)*

loss8m�<'�#
       �	��ܠYc�A�)*

lossj!1<��4u       �	��ݠYc�A�)*

lossҨ�<ڲ�       �	*ޠYc�A�)*

loss�|�;���       �	��ޠYc�A�)*

loss��%:ƭȼ       �	F[ߠYc�A�)*

loss��'<��       �	�ߠYc�A�)*

loss�Y�;�d�U       �	��Yc�A�)*

loss$�l>�2��       �	=�Yc�A�)*

loss1d3=�P�       �	+��Yc�A�)*

loss�۵<R(�       �	�v�Yc�A�)*

lossN��<��ߞ       �	��Yc�A�)*

loss�Z�;9��       �	д�Yc�A�)*

loss�6=�G�q       �	�S�Yc�A�)*

loss}@=���c       �	���Yc�A�)*

lossm�<F�       �	���Yc�A�)*

loss&=Ҙ�-       �	�"�Yc�A�)*

loss���;B�       �	{��Yc�A�)*

loss��<U<�       �	lZ�Yc�A�)*

loss�*C<h���       �	���Yc�A�)*

lossMO?=1��       �	;�Yc�A�)*

loss��e;��R�       �	L��Yc�A�)*

loss�u
=�4�       �	M��Yc�A�)*

loss��=�W�       �	SA�Yc�A�)*

loss�,=��t       �	Ѱ�Yc�A�)*

lossv�^=41f�       �	JF��Yc�A�)*

loss��<X�Z=       �	�u�Yc�A�)*

loss�l�;�k�+       �	b�Yc�A�)*

loss
p�<�n�x       �	��Yc�A�)*

loss% �<P�˻       �	�J�Yc�A�)*

loss���;؅�;       �	���Yc�A�)*

loss�(<_-�j       �	���Yc�A�)*

loss�a�<��(l       �	B#�Yc�A�)*

loss�.�<3��k       �	Y��Yc�A�)*

loss@�<�ߐ       �	�a�Yc�A�)*

loss�g�<)t�W       �	P ��Yc�A�)*

loss�l<�׀�       �	,���Yc�A�)*

loss�0=]H�       �	�@��Yc�A�)*

loss���<H�Ӗ       �	@���Yc�A�)*

loss���;l IF       �	܀��Yc�A�)*

loss�Y<F-q       �	�$��Yc�A�)*

loss�p�<Cߵ�       �	����Yc�A�)*

loss��_=��       �	�m��Yc�A�)*

loss?:=�2�       �	���Yc�A�)*

loss��=�?5�       �	q���Yc�A�)*

loss@�<f���       �	�Q��Yc�A�)*

loss�i8<��bI       �	����Yc�A�)*

loss1�1<��C       �	����Yc�A�)*

loss8:�<b��       �	�&��Yc�A�)*

loss�Տ<�{x�       �	����Yc�A�)*

lossNE�<����       �	�v��Yc�A�)*

lossf�>;�X�U       �	�(��Yc�A�)*

lossNo<�ŚT       �	����Yc�A�)*

loss�x�<�^�       �	o��Yc�A�)*

loss�3<ʹ~       �	 �Yc�A�)*

loss�� <�8       �	u� �Yc�A�)*

loss�w�<�O�{       �	�L�Yc�A�)*

loss2NU=Sk       �	���Yc�A�)*

loss�M8<#�?�       �	0��Yc�A�)*

loss�>Y;�YØ       �	_`�Yc�A�)*

lossDG�=��?       �	���Yc�A�)*

loss��m;��+       �	ߩ�Yc�A�)*

loss|�:Lf�       �	�J�Yc�A�)*

lossHAB;�S��       �	���Yc�A�)*

loss��<�{-�       �	��Yc�A�)*

lossz�9;��*8       �	�!�Yc�A�)*

losswx8<X�       �	���Yc�A�)*

loss���;
ۏ       �	��Yc�A�)*

losso=|o&�       �	x(	�Yc�A�)*

loss~l<Bba�       �	��	�Yc�A�)*

loss��:!Jg�       �	�d
�Yc�A�)*

loss���;��W       �	u�Yc�A�)*

loss�4�<[}V�       �	4��Yc�A�)*

loss��<�b��       �	�?�Yc�A�)*

loss�|<"��       �	j��Yc�A�)*

loss�wJ:x��       �	�t�Yc�A�)*

loss여<5�w)       �	�Yc�A�)*

loss)E>���       �	:��Yc�A�)*

lossV�;�ai�       �	�H�Yc�A�)*

loss�r�;x�v�       �	i��Yc�A�)*

lossi�< Yw�       �	��Yc�A�)*

loss�	= �8�       �	�(�Yc�A�)*

loss�x�<�Ψ�       �	���Yc�A�)*

loss��;��)�       �	e�Yc�A�)*

loss���<���       �	���Yc�A�)*

lossX˜<��d�       �	J��Yc�A�)*

loss���<�Ŕ6       �	�K�Yc�A�)*

loss�<�Eu�       �	���Yc�A�)*

loss ��;�G�'       �	��Yc�A�)*

loss�6�= Է�       �	�3�Yc�A�)*

loss���=Xp       �	'��Yc�A�)*

loss�?�<u|�       �	���Yc�A�)*

loss�8�<8��       �	%�Yc�A�)*

loss<O�<���       �	���Yc�A�)*

lossaG�<��ӫ       �	�i�Yc�A�)*

loss��;XY��       �	s�Yc�A�)*

loss��B=��T�       �	u��Yc�A�)*

loss���;��zt       �	�W�Yc�A�)*

loss%�y<@��       �	j��Yc�A�)*

loss�`=X�@�       �	ș�Yc�A�)*

loss�7B<dZ�+       �	;�Yc�A�)*

loss�Yi<G�
<       �	 ��Yc�A�)*

loss���;�h�       �	�{�Yc�A�)*

loss��;�O�       �	�%�Yc�A�)*

loss�*�<D��       �	U��Yc�A�**

loss�y�<�.p�       �	By �Yc�A�**

loss)��<�Fb�       �	$!�Yc�A�**

loss�)=C��       �	0�!�Yc�A�**

loss7/q<!y��       �	)Y"�Yc�A�**

losshڶ<?�v       �		�"�Yc�A�**

loss��;}�&A       �	J�#�Yc�A�**

loss}�.;��       �	cG$�Yc�A�**

loss�%W<m���       �	C�$�Yc�A�**

loss�à<����       �	��%�Yc�A�**

lossI��<���       �	y"&�Yc�A�**

loss<>7<�{�       �	��&�Yc�A�**

loss�r�=�@��       �	�a'�Yc�A�**

loss��'<�0B�       �	;�'�Yc�A�**

loss8�;�.�       �	A�(�Yc�A�**

lossa��;���       �	�)�Yc�A�**

lossd��<�N�       �	�:*�Yc�A�**

loss,�<��       �	7+�Yc�A�**

loss�ކ<����       �	ض+�Yc�A�**

loss`��<�9��       �	�O,�Yc�A�**

loss��W<��&�       �	r�,�Yc�A�**

loss���;�O��       �	¤-�Yc�A�**

lossrǚ<�w��       �	RI.�Yc�A�**

loss�N ;��R        �	%/�Yc�A�**

lossq� =}���       �	�/�Yc�A�**

lossFȬ<%�X!       �	ݳJ�Yc�A�**

losss^=����       �	�NK�Yc�A�**

loss�u=���       �	��K�Yc�A�**

loss��<��\       �		TM�Yc�A�**

loss��<"�Y�       �	�M�Yc�A�**

loss�\%<#��!       �	~�N�Yc�A�**

loss�
�<ul6O       �	�%O�Yc�A�**

loss�S@=`=��       �	�O�Yc�A�**

loss�-�<�g�!       �	�VP�Yc�A�**

loss�=�NK       �	f�P�Yc�A�**

loss�j<|�z/       �	~�Q�Yc�A�**

loss�'�<����       �	*9R�Yc�A�**

lossrޜ<"n�       �	j�S�Yc�A�**

loss:ʓ<DܒV       �	�YT�Yc�A�**

losss�<u�;z       �	��T�Yc�A�**

loss���=ʩ`       �	r�U�Yc�A�**

lossU�; )4       �	5^V�Yc�A�**

loss,� <�8�       �	�V�Yc�A�**

loss�Ԁ< }��       �	��W�Yc�A�**

lossZ*Y=���       �	�(X�Yc�A�**

losss�;1��       �	��X�Yc�A�**

loss��<=yBjw       �	z�Y�Yc�A�**

loss�2<����       �	[$Z�Yc�A�**

loss�p=>]�        �	��Z�Yc�A�**

loss]*<���       �	}^[�Yc�A�**

loss4F�;s�ѻ       �	+�[�Yc�A�**

lossq��<rk       �	��\�Yc�A�**

lossA�<�[)�       �	d]]�Yc�A�**

loss���<��"       �	W_^�Yc�A�**

loss���<���       �	�_�Yc�A�**

loss�^=���z       �	��_�Yc�A�**

lossC�<DDES       �	8J`�Yc�A�**

loss�(�;X�0B       �	��`�Yc�A�**

lossA�N=�*�%       �	W{a�Yc�A�**

loss΃�<f1�3       �	�b�Yc�A�**

loss{_M=OHg       �	�b�Yc�A�**

lossê�;�2p       �	�Kc�Yc�A�**

loss��<��WJ       �	=�c�Yc�A�**

loss�eN=©e�       �	y�d�Yc�A�**

loss�r=u��6       �	Ae�Yc�A�**

loss��<u�V       �	.�e�Yc�A�**

loss.J�;���       �	�yf�Yc�A�**

loss�"{<�a�V       �	�g�Yc�A�**

loss��<��(�       �	ܺg�Yc�A�**

loss��<� #�       �	^h�Yc�A�**

loss�&=�/�       �	�i�Yc�A�**

loss���<��C�       �	��i�Yc�A�**

loss�p=��       �	�tj�Yc�A�**

loss__�<���       �	73k�Yc�A�**

loss�Œ:�0�       �	��k�Yc�A�**

loss�O�;l��P       �	W&m�Yc�A�**

lossq=�Py�       �	P�m�Yc�A�**

loss��;���"       �	�qn�Yc�A�**

loss�֯=;PhL       �	ko�Yc�A�**

lossN��<#��       �	�o�Yc�A�**

loss�a :�       �	�Jp�Yc�A�**

loss��:�:�!       �	�p�Yc�A�**

loss���;�ANN       �	ݚq�Yc�A�**

loss���<����       �	*9r�Yc�A�**

loss�-h=U�'       �	��s�Yc�A�**

loss�[�=��`       �	rnt�Yc�A�**

loss��;C�       �	,u�Yc�A�**

losscB(<:���       �	�u�Yc�A�**

lossq��<��       �	'Kv�Yc�A�**

loss,$\<��X       �	^�v�Yc�A�**

loss��;�d��       �	��w�Yc�A�**

loss4U�;���O       �	]5x�Yc�A�**

lossC�R<mԙ�       �	B�x�Yc�A�**

loss���<0�,}       �	�fy�Yc�A�**

lossv�=��\       �	!:z�Yc�A�**

loss@��<�ƇC       �	��z�Yc�A�**

loss��<*ްF       �	�{�Yc�A�**

loss�B'=�$L       �	|�Yc�A�**

loss�7�;� ��       �	��|�Yc�A�**

loss�=�Mt8       �	�M}�Yc�A�**

loss!�;��^6       �	�~�Yc�A�**

lossx��;n��       �	�~�Yc�A�**

loss��<;���       �	�A�Yc�A�**

loss��<�nx       �	M��Yc�A�**

loss{{<�}�_       �	���Yc�A�**

lossfWA=$�       �	�K��Yc�A�**

loss��I=�H/�       �	2恡Yc�A�**

loss�q=b_<�       �	Z���Yc�A�**

loss3{R<cs�V       �	p(��Yc�A�**

loss�#�<k�p�       �	����Yc�A�**

loss�(<�p̎       �	�j��Yc�A�**

loss�D�<�Hk�       �	�	��Yc�A�**

loss���<8��s       �	L���Yc�A�**

loss��<.(q       �	�@��Yc�A�**

loss��<U���       �	׆�Yc�A�**

loss�=����       �	�m��Yc�A�**

loss ̓<8�YJ       �	���Yc�A�**

loss	#f; �Y�       �	����Yc�A�**

loss=D#       �	u���Yc�A�**

loss�&)=]�N       �	N)��Yc�A�**

loss$?�;
�Z       �	3���Yc�A�**

lossH8�<�6�9       �	�Y��Yc�A�**

loss�]�;/�:       �	��Yc�A�**

loss\�<əؔ       �	���Yc�A�**

loss8H]<��w�       �	m��Yc�A�**

loss�;Ϟ[�       �	�ڎ�Yc�A�**

loss���<�o6�       �	�q��Yc�A�**

loss�<�,�       �	,��Yc�A�**

loss�<���/       �	�А�Yc�A�+*

lossg5={�>�       �	�k��Yc�A�+*

lossJ�=�ϋ       �	���Yc�A�+*

loss�!�;�̈)       �	Xƒ�Yc�A�+*

lossgQ;�h�       �	�a��Yc�A�+*

loss��<��4       �	R��Yc�A�+*

losshQV;��y%       �	a���Yc�A�+*

loss�=��F       �	�L��Yc�A�+*

loss4�=:       �	I���Yc�A�+*

loss��<d�=@       �	����Yc�A�+*

lossQ�6;M��       �	D��Yc�A�+*

loss�m=?z��       �	i嗡Yc�A�+*

loss�'<����       �	���Yc�A�+*

loss��=I�N�       �	J$��Yc�A�+*

lossh�E=���r       �	n�Yc�A�+*

loss��h<��P'       �	�x��Yc�A�+*

loss��;f �       �	L��Yc�A�+*

losst�}=T۸�       �	A���Yc�A�+*

loss�.�<�HQO       �	m��Yc�A�+*

lossx|<6���       �	�*��Yc�A�+*

loss���<ͧ�I       �	,՝�Yc�A�+*

loss���;�C�       �	���Yc�A�+*

loss�6�=�n�       �	�,��Yc�A�+*

loss.˔<6�o       �	�ӟ�Yc�A�+*

loss�H<Z-g_       �	*r��Yc�A�+*

lossı!=��i       �	���Yc�A�+*

lossU�<���$       �	�¡�Yc�A�+*

loss�w+=[_�       �	�j��Yc�A�+*

loss�4K=�=aO       �	���Yc�A�+*

lossfsu<�=-       �	����Yc�A�+*

loss�X3<�       �	�R��Yc�A�+*

loss2�=Y���       �	����Yc�A�+*

loss�ی<�}       �	DQ��Yc�A�+*

loss�S�<���       �	x���Yc�A�+*

lossܬ4<���       �	r���Yc�A�+*

loss���;[��       �	g&��Yc�A�+*

lossGV�<�f��       �	�v��Yc�A�+*

loss��{;�(d�       �	y��Yc�A�+*

lossf\<���       �	?Ǫ�Yc�A�+*

loss�<R�*�       �	4d��Yc�A�+*

loss�r�<��Ih       �	���Yc�A�+*

lossV8�=z�(       �	����Yc�A�+*

loss�b�<��C       �	0G��Yc�A�+*

lossot�<E���       �	�!��Yc�A�+*

lossi4e;��@u       �	
ڮ�Yc�A�+*

lossnΙ;��u       �	󒯡Yc�A�+*

loss�6=#=��       �	�:��Yc�A�+*

loss�9V<2��s       �	�ڰ�Yc�A�+*

lossi�<���i       �	ˁ��Yc�A�+*

lossT�:=6��       �	�'��Yc�A�+*

loss F�<����       �	����Yc�A�+*

lossl��<���q       �	����Yc�A�+*

loss�r;B-�       �	D0��Yc�A�+*

loss�`';wNE�       �	�ʵ�Yc�A�+*

loss\&!=��e�       �	
h��Yc�A�+*

lossH�C=��p�       �	:��Yc�A�+*

loss��j<�e�       �	�۷�Yc�A�+*

loss�=��ː       �	9{��Yc�A�+*

loss3=�<n�@�       �	���Yc�A�+*

loss�67<�w�h       �	|���Yc�A�+*

loss؋;NH�       �	�\��Yc�A�+*

loss[�=΁��       �	C ��Yc�A�+*

loss���<�Hc=       �	<���Yc�A�+*

loss�j�;L]�A       �	 D��Yc�A�+*

loss�L=Q�        �	����Yc�A�+*

loss�]�;�!�b       �	����Yc�A�+*

loss�`N<GDS       �	31��Yc�A�+*

lossZp1=�jT       �	�ʾ�Yc�A�+*

loss��:l�`�       �	�a��Yc�A�+*

loss��=�P��       �	]���Yc�A�+*

loss�<�,&       �	f���Yc�A�+*

loss�=b�;�       �	�>��Yc�A�+*

loss}�z=�&�'       �	e���Yc�A�+*

lossoM =L�q       �	|~¡Yc�A�+*

loss�p=7�=�       �	UáYc�A�+*

loss8g=�Ȉ       �	�ġYc�A�+*

loss��<���h       �	��ġYc�A�+*

loss���;s?�/       �	5BšYc�A�+*

loss�X4<y�       �	��šYc�A�+*

loss��[<�p*�       �	b�ơYc�A�+*

loss�+�<5��T       �	�(ǡYc�A�+*

loss�*�<�u��       �	��ǡYc�A�+*

loss�h�<�&       �	�^ȡYc�A�+*

lossG�=�K�J       �	��ȡYc�A�+*

lossHSM<���       �	1�ɡYc�A�+*

loss��i=d�.�       �	W^ʡYc�A�+*

losssHg=��       �	e�ʡYc�A�+*

loss�0�=2�N       �	�ˡYc�A�+*

lossW��<?# 
       �	
0̡Yc�A�+*

losssP_;�F�       �	*�̡Yc�A�+*

loss��g<|��       �	�m͡Yc�A�+*

lossfI'<��Ɲ       �	KΡYc�A�+*

loss:'�<�NL�       �	ϟΡYc�A�+*

loss��.=�4:�       �	;ϡYc�A�+*

loss�]@=؋&�       �	?�ϡYc�A�+*

loss��=Ed,:       �	��СYc�A�+*

loss��[<8��       �	�DѡYc�A�+*

loss��h=�Y��       �	��ѡYc�A�+*

loss��;o�.�       �	�yҡYc�A�+*

loss���<<���       �	�ӡYc�A�+*

loss��==�zl       �	��ӡYc�A�+*

loss(�"=p��'       �	PԡYc�A�+*

loss�
�<���D       �	�7֡Yc�A�+*

loss�٢<�_<       �	��֡Yc�A�+*

loss3W�<��\�       �	N}סYc�A�+*

loss��<�tI�       �	HءYc�A�+*

loss��*=�0p�       �	��ءYc�A�+*

lossoW<G���       �	\X١Yc�A�+*

lossS=2Q�       �	��١Yc�A�+*

loss�a�<s���       �	�ڡYc�A�+*

loss���;�a       �	�(ۡYc�A�+*

loss���<�ҠV       �	3�ۡYc�A�+*

loss�$g=LCah       �	�]ܡYc�A�+*

loss.��<�j��       �	a�ܡYc�A�+*

loss�vd=� ��       �	��ݡYc�A�+*

lossJ,=	�RP       �	4ޡYc�A�+*

loss�h1=���H       �	�ޡYc�A�+*

loss\x�<�?I       �	�rߡYc�A�+*

loss^d�=Ng�=       �	�Yc�A�+*

loss�=#��S       �	3��Yc�A�+*

loss��=�Nʅ       �	5B�Yc�A�+*

loss4`�<�Rɝ       �	@�Yc�A�+*

loss��<m�aZ       �	��Yc�A�+*

lossI6z=��Q       �	K�Yc�A�+*

loss��<׋��       �	>��Yc�A�+*

loss�U<w��^       �	�z�Yc�A�+*

loss���<��?       �	��Yc�A�+*

loss��<�!�       �	���Yc�A�+*

loss] �;��c       �	�r�Yc�A�,*

loss0�=T`��       �	k�Yc�A�,*

lossm�=���       �	?��Yc�A�,*

lossJ��<
��j       �	6s�Yc�A�,*

lossG�=Y	Y�       �	��Yc�A�,*

loss��<Y���       �	p"�Yc�A�,*

loss�~<4���       �	��Yc�A�,*

loss�<=�0��       �	\��Yc�A�,*

loss�8�;b9�;       �	L��Yc�A�,*

lossA�;�m�       �	���Yc�A�,*

loss�U=se��       �	�M�Yc�A�,*

loss$&f=��L�       �	���Yc�A�,*

loss�<+1��       �	'��Yc�A�,*

loss:�=��       �	:�Yc�A�,*

loss�=��1�       �	���Yc�A�,*

loss���<����       �	�^�Yc�A�,*

loss
a<�ZpJ       �	\�Yc�A�,*

loss��<���       �	���Yc�A�,*

loss��=����       �	=�Yc�A�,*

loss�=n\�y       �	���Yc�A�,*

loss��4<�^�       �	����Yc�A�,*

loss�2<�       �	=+��Yc�A�,*

lossRDA<}v�f       �	 ���Yc�A�,*

losse�:T� �       �	�y��Yc�A�,*

loss�(�;���       �	H��Yc�A�,*

loss�`<A;8       �	I���Yc�A�,*

loss�=]e�       �	�S��Yc�A�,*

loss�`o<��       �	B$��Yc�A�,*

loss��<F+       �	I���Yc�A�,*

lossS
=[�XH       �	&W��Yc�A�,*

loss��3<���f       �	4���Yc�A�,*

loss�@=�h"�       �	b���Yc�A�,*

loss��<�%G       �	<O��Yc�A�,*

loss��Y=�,>z       �	����Yc�A�,*

loss�S�;߁N�       �	���Yc�A�,*

loss6�=.�u�       �	�k��Yc�A�,*

lossH��<��h�       �	h��Yc�A�,*

loss��:=����       �	���Yc�A�,*

loss$3(=��u       �	�V �Yc�A�,*

loss9
<ފ)0       �	� �Yc�A�,*

loss���<�M ;       �	ŏ�Yc�A�,*

loss
=X�G�       �	�,�Yc�A�,*

loss:�<y���       �	���Yc�A�,*

lossl��=ʑ�       �	�h�Yc�A�,*

loss�k�=/kw�       �	F�Yc�A�,*

loss�S�<�/�       �	Q��Yc�A�,*

loss�ӌ<��w1       �	�@�Yc�A�,*

lossDH1=2�N�       �	s��Yc�A�,*

loss��'=���       �	[{�Yc�A�,*

loss�}B<K5Q       �	 �Yc�A�,*

loss���<�Qu,       �	���Yc�A�,*

loss�	L<�;M_       �	���Yc�A�,*

lossqg8<���-       �	#N	�Yc�A�,*

lossE��<Z���       �	��	�Yc�A�,*

loss[z�<
J;]       �	6�
�Yc�A�,*

lossC�=9ox4       �	,��Yc�A�,*

loss��l<ꎣ�       �	���Yc�A�,*

loss�x=W��d       �	:A�Yc�A�,*

loss��<��B       �	W��Yc�A�,*

loss�!�;<���       �	<��Yc�A�,*

loss(�<V6{       �	��Yc�A�,*

lossn��:��{       �	t��Yc�A�,*

loss*Z�;Y�a       �	��Yc�A�,*

loss��<��9:       �	�5�Yc�A�,*

loss�zi=ʆ��       �	���Yc�A�,*

loss/}p<�� �       �	���Yc�A�,*

loss��=\@��       �	�N�Yc�A�,*

loss�ز=���       �	���Yc�A�,*

lossl/=CF��       �	��Yc�A�,*

loss�=/1ѳ       �	;�Yc�A�,*

loss[�u<�|�)       �	��Yc�A�,*

loss$MC<��_�       �	C�Yc�A�,*

loss���<.�J�       �	���Yc�A�,*

loss(��<��λ       �	���Yc�A�,*

loss��=�ѵM       �	 S�Yc�A�,*

loss�dW<tVn|       �	k��Yc�A�,*

loss�<��b�       �	���Yc�A�,*

lossXR�<[݇�       �	�7�Yc�A�,*

loss��<��
*       �	���Yc�A�,*

loss�� =䍴�       �	�Yc�A�,*

lossr'�<��_       �	��Yc�A�,*

loss�{�<�Hˠ       �	���Yc�A�,*

loss���;>�ym       �	�S�Yc�A�,*

loss�!<�*W{       �	���Yc�A�,*

loss�M=�$`�       �	���Yc�A�,*

loss={�<���       �	�- �Yc�A�,*

loss_�r<�4�Q       �	a� �Yc�A�,*

loss2��<��c       �	�r!�Yc�A�,*

lossr,=����       �	�+"�Yc�A�,*

loss4{�<�mw-       �	�"�Yc�A�,*

loss��(=�zzm       �	6[#�Yc�A�,*

lossΝ�<"�.�       �	u[$�Yc�A�,*

loss�Yt<�(�|       �	i %�Yc�A�,*

loss�z<{R�'       �	O�%�Yc�A�,*

loss�a%=S���       �	3n&�Yc�A�,*

loss=+�;�u��       �	'�Yc�A�,*

loss�#<���       �	u�'�Yc�A�,*

lossw�=��2K       �	r�(�Yc�A�,*

loss��=_��       �	)\)�Yc�A�,*

loss(�g<,�%i       �	��)�Yc�A�,*

lossiU=��X�       �	��*�Yc�A�,*

lossr��:���m       �	�x+�Yc�A�,*

loss�d:<j��0       �	�0,�Yc�A�,*

loss���<Q��       �	�r-�Yc�A�,*

loss@$=Zfܱ       �	�T.�Yc�A�,*

loss���<�z�       �	'�/�Yc�A�,*

loss<=�]�       �	��0�Yc�A�,*

loss|[�<p�Y�       �	V�1�Yc�A�,*

loss�ƌ:k���       �	�e2�Yc�A�,*

loss��!=a��       �	 �3�Yc�A�,*

loss&�<g �C       �	�f4�Yc�A�,*

loss�n�<L�j�       �	�5�Yc�A�,*

loss4U�;5�       �	$b6�Yc�A�,*

loss�>V�0       �	v�7�Yc�A�,*

loss8;C=�ٷ�       �	�i8�Yc�A�,*

lossET�;gx��       �	@Q9�Yc�A�,*

loss�9�<rwwN       �	9B:�Yc�A�,*

loss���<��6       �	';�Yc�A�,*

loss���<�C�       �	 (<�Yc�A�,*

lossO�;�X��       �	.�<�Yc�A�,*

lossH4�<oU��       �	�=�Yc�A�,*

loss��e<�/��       �	�4>�Yc�A�,*

loss�
E<u�}N       �	��>�Yc�A�,*

lossq_�=�9�Q       �	�u@�Yc�A�,*

loss��<\��       �	OA�Yc�A�,*

loss3�H<O�,       �	/�A�Yc�A�,*

loss�e;��       �	c^B�Yc�A�,*

loss|�<�lV       �	�C�Yc�A�,*

loss��
=����       �	��C�Yc�A�-*

loss7��<��φ       �	VD�Yc�A�-*

lossn'�<�3�       �	��D�Yc�A�-*

lossv�`<8�'�       �	��E�Yc�A�-*

lossv�P=J�r       �	�/F�Yc�A�-*

loss�U�<R5�       �	��F�Yc�A�-*

loss�L,<1��       �	�pG�Yc�A�-*

losso�B=�1nE       �	�H�Yc�A�-*

lossנ6<�yO       �	[�H�Yc�A�-*

loss�
;<?ځ�       �	eI�Yc�A�-*

loss�_%=��q�       �	� J�Yc�A�-*

lossco*<��j0       �	��J�Yc�A�-*

loss@�:=7&��       �	�BK�Yc�A�-*

loss��3=ch&�       �	��K�Yc�A�-*

loss��;���       �	�N�Yc�A�-*

loss��<���~       �	H�N�Yc�A�-*

loss}�<���]       �	�iO�Yc�A�-*

loss���<�h��       �	�P�Yc�A�-*

loss��<�ۥ7       �	|�P�Yc�A�-*

loss���<*K_       �	�3Q�Yc�A�-*

lossl��<=K ~       �	��Q�Yc�A�-*

lossxV�<�˫       �	�jR�Yc�A�-*

lossW��<�0c0       �	�S�Yc�A�-*

losse�x=�i��       �	/�S�Yc�A�-*

loss��<b��       �	:YT�Yc�A�-*

loss9H<ՎM�       �	��T�Yc�A�-*

loss���<���X       �	��U�Yc�A�-*

loss��=5�͟       �	j2V�Yc�A�-*

lossv�<;'�       �	��V�Yc�A�-*

loss,�\=f�>d       �	�hW�Yc�A�-*

loss!w*<,lo       �	�X�Yc�A�-*

loss�/<\�`       �	1�X�Yc�A�-*

loss�\�<Η��       �	[Y�Yc�A�-*

loss�m<��c       �	eZ�Yc�A�-*

loss�r=m�y�       �	ԝZ�Yc�A�-*

loss!��<���       �	�7[�Yc�A�-*

loss��=I��N       �	��[�Yc�A�-*

loss=E�;���       �	֏\�Yc�A�-*

lossEv2;��       �	a7]�Yc�A�-*

lossii�<&-�V       �	Q�]�Yc�A�-*

losscޟ=4���       �	�~^�Yc�A�-*

loss��<�c��       �	�_�Yc�A�-*

lossse-=�@x_       �	մ_�Yc�A�-*

loss��
=i!�       �	�L`�Yc�A�-*

loss�<����       �	��`�Yc�A�-*

loss!j�<�y       �	עa�Yc�A�-*

lossiE�<D�;_       �	>b�Yc�A�-*

loss��<rr�       �	M�b�Yc�A�-*

loss��-=�vȵ       �	��c�Yc�A�-*

lossX��;��HJ       �	��d�Yc�A�-*

loss���<]�T-       �	�Ue�Yc�A�-*

loss�`=��p       �	��e�Yc�A�-*

loss�� =i�~       �	��f�Yc�A�-*

loss��<�fv       �	9)g�Yc�A�-*

lossN�>��X�       �	��g�Yc�A�-*

losse9=G�H�       �	}wh�Yc�A�-*

loss�;�=���       �	�i�Yc�A�-*

loss��;[i�       �	��i�Yc�A�-*

loss�<�M       �	h]j�Yc�A�-*

lossZ�I=Z���       �	�k�Yc�A�-*

loss��*=}�G       �	ӣk�Yc�A�-*

loss2|:=˘>       �	�Nl�Yc�A�-*

loss�rS=����       �	�l�Yc�A�-*

loss7�9<:6��       �	3�m�Yc�A�-*

loss�X�=]��       �	�an�Yc�A�-*

loss��,=f��       �	#�n�Yc�A�-*

losse��<�/��       �	ōo�Yc�A�-*

losse�|<	<��       �	}#p�Yc�A�-*

loss�2^=�H{�       �	��p�Yc�A�-*

loss�H�<��@       �	4fq�Yc�A�-*

loss
I�<���7       �	 �q�Yc�A�-*

loss�ڤ<��A�       �	#�r�Yc�A�-*

loss�Xw<"ﮀ       �	�2s�Yc�A�-*

loss��D<�Z       �	��s�Yc�A�-*

loss!��<k�8�       �	��t�Yc�A�-*

loss]G=�w��       �	�>u�Yc�A�-*

loss��<�X       �	^�u�Yc�A�-*

loss�.e<H�       �	�uv�Yc�A�-*

loss<��<�U�&       �	�w�Yc�A�-*

loss_<f�z       �	?�w�Yc�A�-*

loss ��<%�+#       �	]lx�Yc�A�-*

loss�ҹ=�'�       �	�y�Yc�A�-*

loss���<1�       �	��y�Yc�A�-*

loss��=�MF�       �	�Dz�Yc�A�-*

loss���=���/       �	�z�Yc�A�-*

loss��)<����       �	a�{�Yc�A�-*

loss��<g��       �	4+|�Yc�A�-*

loss�|�=��p       �	y�|�Yc�A�-*

lossݩ2=��_       �	7l}�Yc�A�-*

loss
L�<�q��       �	~�Yc�A�-*

loss
�=?���       �	{�~�Yc�A�-*

loss��=z
�       �	<N�Yc�A�-*

loss��<U�       �	$��Yc�A�-*

loss�M.;Y�       �	׉��Yc�A�-*

loss\j|=���       �	c'��Yc�A�-*

loss�T=����       �	���Yc�A�-*

loss;$=~_0       �	�Z��Yc�A�-*

loss�d�<LX*'       �	t���Yc�A�-*

loss�f�<g��Z       �	^���Yc�A�-*

lossM0<�gV�       �	^��Yc�A�-*

loss7�3=I��       �	b���Yc�A�-*

loss	L�;��*       �	���Yc�A�-*

loss�I;yE
       �	�R��Yc�A�-*

lossp�;��       �	�醢Yc�A�-*

loss��<��hr       �	g��Yc�A�-*

loss�2.<�gL�       �	@��Yc�A�-*

loss��G<n��       �	G���Yc�A�-*

loss��=�Cq�       �	O��Yc�A�-*

loss�<j>�       �	!��Yc�A�-*

loss�+<�r       �	Ę��Yc�A�-*

loss���;�WZv       �	�/��Yc�A�-*

loss�6�<܎�       �	Jы�Yc�A�-*

loss��<��q�       �	&s��Yc�A�-*

loss���<���       �	�
��Yc�A�-*

loss��@=��       �	����Yc�A�-*

lossqt<e���       �	�q��Yc�A�-*

loss$��<�x{:       �	o��Yc�A�-*

lossSؓ;.��g       �	e���Yc�A�-*

losspy= w�       �	1?��Yc�A�-*

loss �;��       �	�ܐ�Yc�A�-*

loss]!�=���&       �	Ɔ��Yc�A�-*

loss���=��U       �	� ��Yc�A�-*

lossv	=�Ã�       �	eƒ�Yc�A�-*

loss���<υ�       �	�c��Yc�A�-*

loss���;b�       �	V��Yc�A�-*

loss\�<�iT       �	p���Yc�A�-*

loss4uU<w�ԇ       �	�U��Yc�A�-*

loss���;8܂c       �	���Yc�A�-*

loss\�=^��       �	�ǖ�Yc�A�.*

loss���<#C�       �	Ic��Yc�A�.*

loss~� <�F��       �		��Yc�A�.*

lossS� =_7�Z       �	����Yc�A�.*

lossՋ#=D��       �	�J��Yc�A�.*

loss�G;����       �	�㙢Yc�A�.*

loss
-�<f�E
       �	�~��Yc�A�.*

loss�L�<|��4       �	a��Yc�A�.*

loss 76<�̾�       �	����Yc�A�.*

loss=6D<��       �	�Q��Yc�A�.*

loss8{�<,�       �	�眢Yc�A�.*

loss�A<�
%�       �	���Yc�A�.*

loss�-�<:�b       �	���Yc�A�.*

loss���=p�       �	�'��Yc�A�.*

loss|k�;b �       �	����Yc�A�.*

loss�K�<����       �	�\��Yc�A�.*

lossƭ�<�.u�       �	����Yc�A�.*

lossM�I=��
�       �	i���Yc�A�.*

lossj�<�[Sx       �	�-��Yc�A�.*

loss��=�T�D       �	�Ţ�Yc�A�.*

loss��c<��_       �	�j��Yc�A�.*

loss�==;x	�       �	K��Yc�A�.*

loss�<ES       �	���Yc�A�.*

loss�<rM��       �	�B��Yc�A�.*

loss[Go=q��       �	8ۥ�Yc�A�.*

loss�\�<��`�       �	Wv��Yc�A�.*

loss�M=;z�       �	=��Yc�A�.*

loss�&�<i���       �	\���Yc�A�.*

lossgy�<(��K       �	�E��Yc�A�.*

lossSs<�BXM       �	K䨢Yc�A�.*

lossJ=��
�       �	,}��Yc�A�.*

loss�
<!�B�       �	a��Yc�A�.*

loss��<�N�       �	�ت�Yc�A�.*

loss�Pl;��;D       �	1ϫ�Yc�A�.*

loss��Y<����       �	����Yc�A�.*

loss�r;�9�e       �	]P��Yc�A�.*

lossԈd<Q�       �	+���Yc�A�.*

loss�F�<Y�X�       �	1���Yc�A�.*

loss(�;�       �	�/��Yc�A�.*

loss�e�;ȷ�(       �	����Yc�A�.*

lossh��<&!�       �	 B��Yc�A�.*

lossE�^=���       �	'ر�Yc�A�.*

loss�1�;t���       �	F|��Yc�A�.*

loss�F=�r�z       �	y ��Yc�A�.*

loss`��<�h~�       �	�³�Yc�A�.*

loss��;�ߵb       �	�`��Yc�A�.*

lossZ��9���+       �	���Yc�A�.*

loss�ے;��]       �	ծ��Yc�A�.*

loss.+/<`�L�       �	�R��Yc�A�.*

loss��;gK�r       �	k�Yc�A�.*

loss���<Q0�       �	����Yc�A�.*

loss���;�UYN       �	���Yc�A�.*

loss6Cu;O�       �	����Yc�A�.*

loss�{<.��       �	X��Yc�A�.*

loss���9C�G       �	xYc�A�.*

loss�[�9W~M�       �	B���Yc�A�.*

loss�#h<Өz       �	�E��Yc�A�.*

loss��<��k       �	�ۻ�Yc�A�.*

loss|�o<���Z       �	y��Yc�A�.*

loss�4�;��D�       �	���Yc�A�.*

loss���<=\K       �	����Yc�A�.*

loss!=�.�+       �	w���Yc�A�.*

loss�׷;`���       �	 F��Yc�A�.*

lossqE�;�`�n       �	���Yc�A�.*

lossL��<6�މ       �	���Yc�A�.*

lossiH�<� a�       �	�I¢Yc�A�.*

loss�Z�<�S]�       �	��¢Yc�A�.*

loss*��<�`|       �	�âYc�A�.*

loss���<�6�:       �	��ĢYc�A�.*

losss$<���       �	�ŢYc�A�.*

loss��o<��       �	`tƢYc�A�.*

lossԑ=�]��       �	�ǢYc�A�.*

lossw/,<��$       �	մǢYc�A�.*

loss�TM=U���       �	NȢYc�A�.*

losss�<�Aw�       �	i�ȢYc�A�.*

loss)@�<1�       �	�yɢYc�A�.*

loss*��<��5�       �	DʢYc�A�.*

loss��x=�g'       �	�ʢYc�A�.*

lossd��<���       �	�EˢYc�A�.*

loss��<�8O       �	r�ˢYc�A�.*

lossז-=P[��       �	��̢Yc�A�.*

loss���;��
�       �	O=͢Yc�A�.*

lossV�C<���C       �	��͢Yc�A�.*

loss�	<H��       �	nn΢Yc�A�.*

loss�L�;`=B2       �	}ϢYc�A�.*

lossր(;���       �	'�ϢYc�A�.*

loss	#<:���       �	{�ТYc�A�.*

loss]�N;H��v       �	TѢYc�A�.*

loss�h�<1aq       �	9�ѢYc�A�.*

loss:�<=E��       �	�QҢYc�A�.*

loss���<���       �	��ҢYc�A�.*

loss�N=�ٟB       �	b�ӢYc�A�.*

losswܚ<�w_       �	�ZԢYc�A�.*

lossr�;�,'       �	��ԢYc�A�.*

loss :�:��       �	�բYc�A�.*

loss���:�k�N       �	�7֢Yc�A�.*

loss�ہ<諩�       �	��֢Yc�A�.*

loss=�< ���       �	\rעYc�A�.*

loss�=���       �	�آYc�A�.*

loss�v�<�m�]       �	j�آYc�A�.*

loss���<�       �	Έ٢Yc�A�.*

loss�=_r�v       �	^,ڢYc�A�.*

lossgd�<��       �	��ڢYc�A�.*

loss��<?[;l       �	��ۢYc�A�.*

loss��<l��.       �	YLܢYc�A�.*

loss���<|I�       �	��ܢYc�A�.*

loss,f�<���       �	3�ݢYc�A�.*

lossq p<Hpt       �	n3ޢYc�A�.*

lossxJ�<���       �	�ޢYc�A�.*

loss+�=u�[       �	R�ߢYc�A�.*

loss�E�<b[       �	\�Yc�A�.*

loss���:��       �	<��Yc�A�.*

loss�g-;Ӗc�       �	���Yc�A�.*

loss��;I�E�       �	����Yc�A�.*

lossVTS=k�@       �	7���Yc�A�.*

lossͧ�=��       �	�^��Yc�A�.*

loss�g
=*N��       �	3���Yc�A�.*

loss�u=�C�       �	S� �Yc�A�.*

loss���<����       �	�(�Yc�A�.*

loss��l<�K�       �	��Yc�A�.*

loss��<ox�x       �	�v�Yc�A�.*

loss��<B5�       �	��Yc�A�.*

loss�N�<�\0       �	��Yc�A�.*

loss�<���       �	C��Yc�A�.*

loss?��=H��       �	n2�Yc�A�.*

loss��< ¥�       �	0��Yc�A�.*

loss��*=b	�,       �	5~�Yc�A�.*

loss��=[;�{       �	��Yc�A�.*

loss��<���       �	��Yc�A�/*

losst�:�x��       �	wh	�Yc�A�/*

lossv�M<&�       �	e
�Yc�A�/*

loss&J(<��l       �	�
�Yc�A�/*

lossŮ<QO|       �	�=�Yc�A�/*

loss��<�JFX       �	#��Yc�A�/*

lossi�U=Xu�       �	ݵ�Yc�A�/*

loss1<<�M?d       �	�V�Yc�A�/*

loss�0�=�C�       �	�|�Yc�A�/*

loss�V�;�(l       �	d�Yc�A�/*

loss;a�<*&�c       �	M��Yc�A�/*

loss�1=�Xh       �	^�Yc�A�/*

loss�hu<��:.       �	���Yc�A�/*

lossm�<�8~       �	��Yc�A�/*

loss���;��W�       �	:�Yc�A�/*

loss;e<���       �	T�Yc�A�/*

loss�K�;)���       �	���Yc�A�/*

loss�.<6���       �	Z�Yc�A�/*

loss�Ջ=D��       �	��Yc�A�/*

loss�@�;�rV�       �	%��Yc�A�/*

loss�K�<�j�i       �	*R�Yc�A�/*

loss�¾;I�ѐ       �	��Yc�A�/*

loss�B�<���       �	���Yc�A�/*

lossa��=��       �	d �Yc�A�/*

loss�v=�z%�       �	g��Yc�A�/*

loss:�+<���       �	�r�Yc�A�/*

loss��F<B�BU       �	��Yc�A�/*

loss%DK<���       �	��Yc�A�/*

lossֻ�<N��a       �	�B�Yc�A�/*

loss�9�<d�p�       �	��Yc�A�/*

loss�T=�?�       �	�s�Yc�A�/*

lossw<a[       �	y �Yc�A�/*

loss�ʴ<�F_�       �	k��Yc�A�/*

loss�<z�       �	O�Yc�A�/*

loss`Ũ;^�C       �	���Yc�A�/*

loss���;�f       �	�Yc�A�/*

loss�K=�X-       �	� �Yc�A�/*

loss4(�;���       �	ު �Yc�A�/*

loss���=k��       �	�G!�Yc�A�/*

loss���<��Z       �	T�!�Yc�A�/*

loss}�[;�1�       �	�{"�Yc�A�/*

loss@];"�!�       �	�#�Yc�A�/*

loss��d:?�=       �	��#�Yc�A�/*

lossRD�;{�;�       �	�L$�Yc�A�/*

loss�g�<<��       �	��$�Yc�A�/*

losst�<�v#�       �	0�%�Yc�A�/*

loss�<;��       �	�&�Yc�A�/*

loss��;Q�       �	�&�Yc�A�/*

loss�v<���       �	)]'�Yc�A�/*

lossN�<��v       �	/�'�Yc�A�/*

loss�<��P�       �	%�(�Yc�A�/*

loss֘�<�TV       �	,+)�Yc�A�/*

loss��w<+H��       �	H�)�Yc�A�/*

loss��?=�Ko       �	�^*�Yc�A�/*

loss2{L=���}       �	��*�Yc�A�/*

loss���<��k�       �	�+�Yc�A�/*

losse�3=K�       �	�P,�Yc�A�/*

lossvP�<j	f       �	v-�Yc�A�/*

loss?�=�-r�       �	}y.�Yc�A�/*

loss!�=���k       �	�/�Yc�A�/*

loss.C�<��c       �	.0�Yc�A�/*

lossV��;�-	�       �	->1�Yc�A�/*

loss��;�yT       �	��1�Yc�A�/*

loss���;���G       �	Kt2�Yc�A�/*

lossQ�<        �	�$3�Yc�A�/*

lossF5�<'y��       �	F�3�Yc�A�/*

lossѿ�<��c       �	\u4�Yc�A�/*

loss%ğ<DTȏ       �	�5�Yc�A�/*

loss�T�<(�       �	�5�Yc�A�/*

lossh�!<u��G       �	�l6�Yc�A�/*

loss��$;��g       �	�7�Yc�A�/*

lossi"y<}.f       �	��7�Yc�A�/*

lossq<=!
       �	[a8�Yc�A�/*

loss�8C<�PT       �	�29�Yc�A�/*

loss�l�<��W       �	��9�Yc�A�/*

lossT�<l���       �	��:�Yc�A�/*

loss�"=o��       �	�v;�Yc�A�/*

lossѠ�;H�       �	*<�Yc�A�/*

loss�M�<U��       �	��<�Yc�A�/*

lossZ3�<kI\�       �	M�=�Yc�A�/*

loss��k;A"�       �	�q>�Yc�A�/*

loss`��=�fR�       �	�b?�Yc�A�/*

loss��;�<       �	�@�Yc�A�/*

loss�v�<���       �	��@�Yc�A�/*

loss �@<�M�       �	�FA�Yc�A�/*

loss1;�'�       �	��A�Yc�A�/*

losst��;
�i�       �	^�B�Yc�A�/*

loss��r<��a�       �	C�Yc�A�/*

lossS�<�u       �	��C�Yc�A�/*

loss�~�<h->       �	rPD�Yc�A�/*

loss2:s=o3-k       �	A�D�Yc�A�/*

lossnQ�;�A<u       �	�E�Yc�A�/*

loss��;�3j       �	�:F�Yc�A�/*

loss�=�.}�       �	��F�Yc�A�/*

loss�F<(�f�       �	oG�Yc�A�/*

loss�1�<?�S�       �	�H�Yc�A�/*

lossTC�=�o�       �	�H�Yc�A�/*

loss2�Y=B]�       �	_BI�Yc�A�/*

loss,	�;�pG       �	a�I�Yc�A�/*

loss���;�{H       �	N|J�Yc�A�/*

loss|	Y<�Q       �	�K�Yc�A�/*

loss�T"=��*$       �	6�K�Yc�A�/*

loss�m�=�	l       �	(bL�Yc�A�/*

loss"6�<I��       �	=�M�Yc�A�/*

lossh�5=6;%�       �	�XN�Yc�A�/*

loss��<�~�%       �	��N�Yc�A�/*

loss$aA= K�       �	'�O�Yc�A�/*

lossK<��
       �	FP�Yc�A�/*

loss��K<��x       �	e�P�Yc�A�/*

lossz��;�ة       �	>wQ�Yc�A�/*

loss?	y=w>r�       �	�R�Yc�A�/*

loss;"=�g&�       �	]�R�Yc�A�/*

loss���;�eI\       �	$`S�Yc�A�/*

loss��<K��y       �	T�Yc�A�/*

loss�Fe<��       �	j�T�Yc�A�/*

lossLܻ<���|       �	NU�Yc�A�/*

loss��<��4       �	
�U�Yc�A�/*

loss-nS=�x�E       �	o�V�Yc�A�/*

loss�T<#�Xb       �	=IW�Yc�A�/*

loss|�Y=�h       �	X�W�Yc�A�/*

loss�*7<���j       �	@�X�Yc�A�/*

loss��=���j       �	K:Y�Yc�A�/*

loss�V�<���       �	��Y�Yc�A�/*

lossM��;�~f�       �	zrZ�Yc�A�/*

loss߷!<�<       �	�[�Yc�A�/*

lossR0�;R��       �	T�[�Yc�A�/*

lossڛ.<z�       �	�?\�Yc�A�/*

lossֶl<]�       �	�\�Yc�A�/*

losst��<�tU       �	�v]�Yc�A�0*

loss4fi=�`A�       �	0^�Yc�A�0*

lossq?�<�J��       �	��^�Yc�A�0*

loss�<<R�!�       �	C_�Yc�A�0*

loss@��;�<�       �	7�_�Yc�A�0*

loss��5<t�,�       �	x`�Yc�A�0*

loss�)�<�5[V       �	�a�Yc�A�0*

lossw�<d��W       �	��a�Yc�A�0*

loss�:�=%���       �	�Ab�Yc�A�0*

loss"=��t       �	��b�Yc�A�0*

loss��=�\��       �	�vc�Yc�A�0*

loss�y�<����       �	�d�Yc�A�0*

lossR�:�L�       �	
�d�Yc�A�0*

loss,ڛ;k��       �	�be�Yc�A�0*

loss���<;,       �	"�e�Yc�A�0*

loss��8=<�u       �	K�f�Yc�A�0*

lossa79=r#       �	Z.g�Yc�A�0*

loss�S$=&���       �	C�g�Yc�A�0*

lossi�B<�s�       �	�_h�Yc�A�0*

losso@�<��f�       �	D�h�Yc�A�0*

loss}��;�(A       �	)�i�Yc�A�0*

loss�B�<�|��       �	.j�Yc�A�0*

loss�<j�i�       �	��j�Yc�A�0*

loss��`;�4/       �	,gk�Yc�A�0*

loss���=W]�       �	��k�Yc�A�0*

loss�x�<���       �	��l�Yc�A�0*

lossӶ-={^��       �	�3m�Yc�A�0*

loss1'G=����       �	��m�Yc�A�0*

loss���;G+�       �	 o�Yc�A�0*

lossi��;���       �	`p�Yc�A�0*

loss/"<��,,       �	��p�Yc�A�0*

lossR�u<���\       �	��q�Yc�A�0*

loss�|r<	k�       �	=�r�Yc�A�0*

loss-=�o�!       �	4Is�Yc�A�0*

loss�<,��       �	��s�Yc�A�0*

loss�@�<%:�       �	U�t�Yc�A�0*

loss���<�       �	��u�Yc�A�0*

loss���;y�=       �	�v�Yc�A�0*

loss)'8<��O       �	��v�Yc�A�0*

loss`��<=l"s       �	@Lw�Yc�A�0*

loss���;#U=       �	~�w�Yc�A�0*

loss��;�� �       �	�x�Yc�A�0*

loss,��;�8       �	*y�Yc�A�0*

loss�� =�M�B       �	&�y�Yc�A�0*

loss���<�,Ѽ       �	|�z�Yc�A�0*

loss}<�<'q       �	�{�Yc�A�0*

lossiT�<�-\       �	й{�Yc�A�0*

loss�vP=^��       �	\|�Yc�A�0*

loss��<�3�k       �	�|�Yc�A�0*

loss�1<�S�3       �	x�}�Yc�A�0*

loss�;+<��*       �	�5~�Yc�A�0*

lossʳ1<��:�       �	��~�Yc�A�0*

lossS�<���       �	�n�Yc�A�0*

loss̚�<�&y       �	���Yc�A�0*

loss�;<=�!��       �	m���Yc�A�0*

loss��<�NwT       �	�J��Yc�A�0*

loss��;��4       �	�䁣Yc�A�0*

loss�)�<��n�       �	���Yc�A�0*

loss앉;����       �	&��Yc�A�0*

loss�<s���       �	E���Yc�A�0*

loss�w=����       �	TR��Yc�A�0*

loss�v^<�#{�       �	�鄣Yc�A�0*

lossܦ�<u�       �	����Yc�A�0*

loss��<0��
       �	���Yc�A�0*

lossYk�;j��D       �	g���Yc�A�0*

loss)n=�'e       �	N��Yc�A�0*

loss�K�<��86       �	釣Yc�A�0*

loss�Q�<��       �	����Yc�A�0*

loss�t�<�D��       �	�/��Yc�A�0*

loss�l(<�=#       �	�Љ�Yc�A�0*

loss#�L;0�m�       �	 t��Yc�A�0*

loss3*=�1Un       �	���Yc�A�0*

loss��D=yd�       �	�\��Yc�A�0*

loss�;=��3       �	��Yc�A�0*

loss��=�$h�       �	񠍣Yc�A�0*

lossZ3B=��e�       �	���Yc�A�0*

lossZ=�4Ĥ       �	�?��Yc�A�0*

loss�==�&�       �	M܏�Yc�A�0*

loss��l<�W#�       �	b���Yc�A�0*

lossx��<E�R�       �	P��Yc�A�0*

loss(,9=��o�       �	����Yc�A�0*

loss���<Da��       �	����Yc�A�0*

loss$w�<��       �	�I��Yc�A�0*

lossj�t<����       �	�蓣Yc�A�0*

loss�;w>��       �	!���Yc�A�0*

loss��<ܬ�       �	{��Yc�A�0*

lossN�?=fb;+       �	�#��Yc�A�0*

lossh��;s��[       �	��Yc�A�0*

loss��<�'       �	
e��Yc�A�0*

loss��<n�       �	%��Yc�A�0*

lossm�<-C��       �	^���Yc�A�0*

loss�Ơ<p7�       �	G��Yc�A�0*

losss��<�#x�       �	&♣Yc�A�0*

loss߅�<�       �	ԝ��Yc�A�0*

loss־�=.2�       �	Z��Yc�A�0*

loss=U=b2��       �	����Yc�A�0*

loss�N<��T       �	2���Yc�A�0*

lossH2�;)~)3       �	GW��Yc�A�0*

loss1x�<���       �	����Yc�A�0*

lossfL7<��)       �	�랣Yc�A�0*

loss
�	=��       �	���Yc�A�0*

loss_�<��4V       �	�<��Yc�A�0*

loss�y=��       �	�ޠ�Yc�A�0*

loss�}=)t��       �	���Yc�A�0*

lossT�<�I�       �	���Yc�A�0*

lossd��<2��S       �	ع��Yc�A�0*

loss�cw<�P�       �	\��Yc�A�0*

loss��<z>��       �	���Yc�A�0*

losshV=�9I       �	鞤�Yc�A�0*

loss�=9>�q       �	�<��Yc�A�0*

loss�;k��       �	cե�Yc�A�0*

loss,�<��|       �	�q��Yc�A�0*

loss���<>���       �	���Yc�A�0*

loss�W<��@       �	����Yc�A�0*

lossf�D<�r�       �	kH��Yc�A�0*

loss;��<�9j,       �	u䨣Yc�A�0*

loss�<�        �	�}��Yc�A�0*

loss��<f�       �	*��Yc�A�0*

loss�<�Zy0       �	괪�Yc�A�0*

loss��&='�U       �	K��Yc�A�0*

loss{K�<|)H4       �	q櫣Yc�A�0*

loss��=���       �	${��Yc�A�0*

loss���;�h��       �	(��Yc�A�0*

loss���=�!/�       �	G��Yc�A�0*

lossь�<}t��       �	3ᮣYc�A�0*

loss}�<:cW�       �	逯�Yc�A�0*

lossW�=:�!m       �	�\��Yc�A�0*

loss� <H�n�       �	Ac��Yc�A�0*

loss�W�;��Ю       �	"���Yc�A�1*

loss�Ib=�<)m       �	,F��Yc�A�1*

loss�y�<פ�       �	���Yc�A�1*

loss�؄=s�+=       �	�ᴣYc�A�1*

loss��=[yh�       �	פ��Yc�A�1*

loss_��<*��X       �	g_��Yc�A�1*

lossi�e<����       �	h��Yc�A�1*

loss$��<��,@       �	���Yc�A�1*

loss�=���E       �	y$��Yc�A�1*

loss��;���w       �	���Yc�A�1*

loss a<X��X       �	�ٺ�Yc�A�1*

loss�/�<���       �	w���Yc�A�1*

loss��<gD[�       �	(��Yc�A�1*

loss�gL=3	6,       �	<0��Yc�A�1*

loss���<��       �	~���Yc�A�1*

loss�/=�Q�~       �	����Yc�A�1*

lossm��<��U       �	�`��Yc�A�1*

loss���<NRs       �	v4��Yc�A�1*

lossXC;��Wi       �	^*£Yc�A�1*

loss��<�ԯ�       �	��£Yc�A�1*

loss}�z<-�       �	w�ãYc�A�1*

loss�I�:} �       �	�+ģYc�A�1*

loss�bB<�}u       �	�ģYc�A�1*

loss�l-<���       �	f�ţYc�A�1*

lossS=��Y%       �	�3ƣYc�A�1*

loss��V<\Ǜ<       �	��ƣYc�A�1*

loss�=�2�       �	��ǣYc�A�1*

loss{˷<���       �	IɣYc�A�1*

lossM�M=m^       �	j�ɣYc�A�1*

loss*֫<=-`�       �	�ʣYc�A�1*

loss]L�<�*u�       �	ZEˣYc�A�1*

loss1�*<�l��       �	��ˣYc�A�1*

loss�x�<�Ț�       �	��̣Yc�A�1*

lossO:�<ڙ*       �	�AͣYc�A�1*

lossnς=	]:�       �	T�ͣYc�A�1*

loss�޿<���       �	7�ΣYc�A�1*

loss�(:=3���       �	�.ϣYc�A�1*

loss�T�<�o��       �	��ϣYc�A�1*

loss���<)�Bu       �	qУYc�A�1*

loss���<�?t�       �	�ѣYc�A�1*

loss��<�'?�       �	C�ѣYc�A�1*

lossd�#=0���       �	sLңYc�A�1*

loss�&L;?��       �	F�ңYc�A�1*

loss��{;���       �	�ӣYc�A�1*

loss��<���a       �	<-ԣYc�A�1*

lossKH=J%2�       �	��ԣYc�A�1*

loss �<�d��       �	jգYc�A�1*

loss�|�<(�N       �	e֣Yc�A�1*

loss W�<Z�ǚ       �	��֣Yc�A�1*

loss�Q�<��b       �	XףYc�A�1*

loss̈-=�P0�       �	4�ףYc�A�1*

loss@��<b4       �	ęأYc�A�1*

loss�8�<Ģ0�       �	�3٣Yc�A�1*

loss�A*=�<�       �	B�٣Yc�A�1*

loss�<�l�'       �	eڣYc�A�1*

loss&��<?W?|       �	h$ۣYc�A�1*

loss�?�<�`�f       �	��ۣYc�A�1*

loss�9�<�4��       �	>\ܣYc�A�1*

loss���<�c�k       �	8�ܣYc�A�1*

loss��(<>0�       �	�ݣYc�A�1*

loss��<����       �	^*ޣYc�A�1*

loss��';���
       �	dߣYc�A�1*

loss��<���       �	�ߣYc�A�1*

loss�ǿ<���M       �	�v�Yc�A�1*

lossW=_α�       �	��Yc�A�1*

loss�w�<`g�       �	o��Yc�A�1*

lossX�<:͋�       �	S�Yc�A�1*

loss��<�o�&       �	c��Yc�A�1*

loss��:��K�       �	���Yc�A�1*

loss5<5��       �	(�Yc�A�1*

loss,�<f���       �	r��Yc�A�1*

loss�=�EZ       �	5c�Yc�A�1*

loss�4u<�Z��       �	���Yc�A�1*

loss�7W=P���       �	X8�Yc�A�1*

loss6�.=�#       �	"��Yc�A�1*

loss�$�<{"/�       �	$�Yc�A�1*

loss3�<���       �	(��Yc�A�1*

loss�4=� �       �	��Yc�A�1*

loss��<�ӹ       �	:A�Yc�A�1*

lossO�<���       �	��Yc�A�1*

lossvl=�@��       �	���Yc�A�1*

loss���<qjN�       �	H���Yc�A�1*

loss�f�<#��       �	E��Yc�A�1*

lossQL<�MȎ       �	H7�Yc�A�1*

loss���<��~       �	���Yc�A�1*

loss���<Z{el       �	�|�Yc�A�1*

loss��<���       �	�(�Yc�A�1*

lossy�< ���       �	�$�Yc�A�1*

loss�Q�=�1ҵ       �	Q��Yc�A�1*

loss���<��=�       �	�~�Yc�A�1*

lossW`w<~0:�       �	���Yc�A�1*

loss��=W2\�       �	����Yc�A�1*

lossF��<Þ��       �	�V��Yc�A�1*

loss��<�R��       �	��Yc�A�1*

loss֭<���       �	����Yc�A�1*

loss&ux<��?       �	6?��Yc�A�1*

loss=�=B̎       �	����Yc�A�1*

lossȥ'<y=s       �	���Yc�A�1*

lossq�b<��*�       �	<N��Yc�A�1*

loss�?<#�s�       �	����Yc�A�1*

loss�Wr<l�>       �	w���Yc�A�1*

loss��<�P+       �	a4��Yc�A�1*

loss��+<�]�       �	����Yc�A�1*

loss�A=�~O       �	�j��Yc�A�1*

loss�Q=�_�_       �	���Yc�A�1*

loss��C=bA��       �	����Yc�A�1*

loss⽝<�ՉZ       �	e9��Yc�A�1*

loss�v=��]       �	v���Yc�A�1*

lossq�5=����       �	����Yc�A�1*

lossJ$P<��R       �	�? �Yc�A�1*

loss���<����       �	�� �Yc�A�1*

loss�g0=z�	       �	��Yc�A�1*

lossy <�f       �	o,�Yc�A�1*

loss�<�=�i       �	�V�Yc�A�1*

lossJ��<��~       �	���Yc�A�1*

loss!X�<����       �	���Yc�A�1*

loss���;[�|�       �	�4�Yc�A�1*

lossi��<�ES�       �	���Yc�A�1*

loss���;����       �	o~�Yc�A�1*

loss�:<�K�{       �	<.�Yc�A�1*

loss��<b~�x       �	���Yc�A�1*

loss�3<aGI�       �	Qg�Yc�A�1*

loss��K=�       �	�	�Yc�A�1*

loss�[<eq3       �	��	�Yc�A�1*

loss<ߓ<��/�       �	�@
�Yc�A�1*

loss��;���       �	�Yc�A�1*

loss�:�;�kd       �	��Yc�A�1*

loss�;*1T�       �	)=�Yc�A�1*

lossl�<�G7�       �	���Yc�A�2*

loss�9<� ��       �	%y�Yc�A�2*

lossq�<L�T�       �	��Yc�A�2*

loss�l�<��3�       �	���Yc�A�2*

loss�w><��71       �	��Yc�A�2*

loss��"=���       �	�=�Yc�A�2*

lossߤ;=qj��       �	a��Yc�A�2*

loss���<�I��       �	=��Yc�A�2*

loss��<��)�       �	:=�Yc�A�2*

loss�!=<i7�       �	���Yc�A�2*

lossZ�j;�})�       �	f��Yc�A�2*

loss�h7=R���       �	f-�Yc�A�2*

loss�=Q.��       �	���Yc�A�2*

lossa��;�?%'       �	rn�Yc�A�2*

lossϿ;=,��       �	��Yc�A�2*

lossi8�<���m       �	ū�Yc�A�2*

lossMI
=�0�       �	C�Yc�A�2*

lossq�8<�N��       �	���Yc�A�2*

lossq�&<�˴       �	Sx�Yc�A�2*

lossh�<�7>       �	��Yc�A�2*

lossDt�<��G       �	��Yc�A�2*

loss���<���7       �	���Yc�A�2*

loss�Wr=�>�       �	�/�Yc�A�2*

loss �6;C���       �	���Yc�A�2*

lossRq=����       �	Cq�Yc�A�2*

loss%�(=���       �	.�Yc�A�2*

loss��<4-�       �	Ӽ�Yc�A�2*

lossE��;\a��       �	���Yc�A�2*

lossRS�<��l       �	�o�Yc�A�2*

lossAN�<d"�       �	Q �Yc�A�2*

lossoR�=gr��       �	x� �Yc�A�2*

lossS�k<9�aY       �	_\!�Yc�A�2*

loss4|=��oH       �	5"�Yc�A�2*

lossf<���+       �	��"�Yc�A�2*

loss�e�<e��k       �	W#�Yc�A�2*

lossÊH<.;�)       �	�#�Yc�A�2*

loss�)=�.       �	��$�Yc�A�2*

lossh�<����       �	�D%�Yc�A�2*

lossP)=v:`       �	��%�Yc�A�2*

loss��<P�       �	_�&�Yc�A�2*

lossDL�;��C       �	#J'�Yc�A�2*

loss7=��       �	��'�Yc�A�2*

lossq�=���%       �	�(�Yc�A�2*

lossd2%=���       �	O#)�Yc�A�2*

loss���<j��P       �	��)�Yc�A�2*

loss��;	       �	�\*�Yc�A�2*

lossv��<���       �	>+�Yc�A�2*

lossSI>b{�       �	4�+�Yc�A�2*

loss��<++A       �	�|,�Yc�A�2*

loss�f<�_al       �	/-�Yc�A�2*

loss�<���j       �	�
.�Yc�A�2*

loss���<��p       �	��.�Yc�A�2*

loss|��;��x�       �	+O/�Yc�A�2*

lossCM<�c�L       �	y�/�Yc�A�2*

loss�_=�Nɑ       �	8�0�Yc�A�2*

loss���=}݋       �	�1�Yc�A�2*

loss�2	=����       �	��1�Yc�A�2*

lossG9<2*3�       �	�Y2�Yc�A�2*

loss���<�8!       �	� 3�Yc�A�2*

loss�6�;L>��       �	Ƣ3�Yc�A�2*

lossq�	=IXQ�       �	u<4�Yc�A�2*

loss]*�<9�X       �	��4�Yc�A�2*

loss��<�Z       �	�x5�Yc�A�2*

loss,Ҍ;cQ\K       �	%6�Yc�A�2*

loss��g=7���       �	M�6�Yc�A�2*

loss��h<ޙL       �	1[7�Yc�A�2*

lossڧ�;��d:       �	��7�Yc�A�2*

loss�r�<�AaY       �	��8�Yc�A�2*

lossD��<d*       �	�;9�Yc�A�2*

loss�~=Z<�       �	�9�Yc�A�2*

loss�9�;w���       �	Dj:�Yc�A�2*

loss�6�<dk�       �	�;�Yc�A�2*

lossfn<O��       �	�;�Yc�A�2*

loss?J=�)��       �	UL<�Yc�A�2*

lossw�<�!Cs       �	q�<�Yc�A�2*

lossq�F<�b��       �	�|=�Yc�A�2*

lossr =�k�       �	�>�Yc�A�2*

loss� �<���       �	�>�Yc�A�2*

loss|��;�P�&       �	�P?�Yc�A�2*

loss�vs;�B�       �	�?�Yc�A�2*

loss��=l��       �	\�@�Yc�A�2*

loss��<"�       �	�0A�Yc�A�2*

loss��9=(/)       �	W�A�Yc�A�2*

lossx�<�ul�       �	?nB�Yc�A�2*

loss���;,��       �	C�Yc�A�2*

loss,��<�X��       �	��C�Yc�A�2*

loss��<w�}F       �	�MD�Yc�A�2*

loss�b=>ʁ�       �	�D�Yc�A�2*

loss��a=%5�       �	E�Yc�A�2*

loss�:�;d��+       �	?�F�Yc�A�2*

lossRa�</�s]       �	)G�Yc�A�2*

lossZ�C< �ր       �	J�G�Yc�A�2*

loss?I<�T��       �	&mH�Yc�A�2*

lossc�;@�F�       �	5I�Yc�A�2*

loss[<�w��       �	��I�Yc�A�2*

loss���<i��v       �	�QJ�Yc�A�2*

lossF�=���       �	�J�Yc�A�2*

lossy�<���       �	܂K�Yc�A�2*

loss&�<7��&       �	6L�Yc�A�2*

lossr�<��1       �	1�L�Yc�A�2*

lossAC=�ݺ       �	6�M�Yc�A�2*

loss��r=�4F�       �	b/N�Yc�A�2*

loss�<�pQ       �	}�N�Yc�A�2*

loss��.<|�|�       �	��O�Yc�A�2*

loss�<4^��       �	=(P�Yc�A�2*

loss��7=���b       �	�P�Yc�A�2*

loss-6`=�4       �	�pQ�Yc�A�2*

loss.�<gf�w       �	�eR�Yc�A�2*

loss�<�:��       �	WS�Yc�A�2*

loss|�=�X��       �	U�S�Yc�A�2*

loss�L�;�g�       �	ZIT�Yc�A�2*

lossᎅ<�^��       �	�*U�Yc�A�2*

lossf2�<��r       �	�)V�Yc�A�2*

loss,��<)��       �	.�V�Yc�A�2*

lossnq=<��       �	�_W�Yc�A�2*

loss��=$
       �	`X�Yc�A�2*

loss�z8=Ju��       �	ӡX�Yc�A�2*

loss�B�<I�t       �	�9Y�Yc�A�2*

lossE�8<�n       �	��Y�Yc�A�2*

lossWx�;�xm?       �	�gZ�Yc�A�2*

loss�t�<����       �	�[�Yc�A�2*

loss� ;���       �	y�[�Yc�A�2*

loss�<g���       �	>]\�Yc�A�2*

loss(0;BP�       �	��\�Yc�A�2*

loss/Ç< �J       �	��]�Yc�A�2*

loss_m!=d�b       �	Pn^�Yc�A�2*

loss
��<���       �	�_�Yc�A�2*

loss%v;�n:�       �	N�_�Yc�A�2*

loss<��<Nnl       �	�U`�Yc�A�3*

loss<@�<߬`       �	{�`�Yc�A�3*

loss3;�;/vF       �	��a�Yc�A�3*

loss%;��V\       �	A*b�Yc�A�3*

loss���<�jFc       �	v�b�Yc�A�3*

loss��;;�R��       �	^c�Yc�A�3*

lossMB/;w���       �	�d�Yc�A�3*

lossq��;��{       �	|�d�Yc�A�3*

loss�H<��(�       �	"3e�Yc�A�3*

lossS�;�OB�       �	X�e�Yc�A�3*

loss|�;u�r       �	Xof�Yc�A�3*

loss�@<��/       �	g�Yc�A�3*

lossNB�;O �       �	8�g�Yc�A�3*

lossJYM<�С*       �	ADh�Yc�A�3*

loss;����       �	��h�Yc�A�3*

loss�?�;�Pڊ       �	��i�Yc�A�3*

loss._�<�k��       �	Xj�Yc�A�3*

loss$�<�.d       �	��j�Yc�A�3*

losstl<�v       �	GYk�Yc�A�3*

loss��9Ws�j       �	��k�Yc�A�3*

loss!�<��       �	�l�Yc�A�3*

losszWl=��)�       �	#0m�Yc�A�3*

loss{g�;�~��       �	z�m�Yc�A�3*

loss��:M>�.       �	�ln�Yc�A�3*

loss]�,<0W�7       �	oo�Yc�A�3*

loss.�<�H��       �	qp�Yc�A�3*

loss�V<���D       �	��p�Yc�A�3*

loss�0�;ɖ�       �	�q�Yc�A�3*

loss��U<���       �	�Br�Yc�A�3*

loss�Gl=�g�       �	�r�Yc�A�3*

loss&�=�/       �	�s�Yc�A�3*

loss��<�ВS       �	�Pt�Yc�A�3*

lossf�<��j�       �	�)u�Yc�A�3*

lossi�H=�t�-       �	B>v�Yc�A�3*

loss��=j�;�       �	��v�Yc�A�3*

loss(R�=���       �	w�Yc�A�3*

loss%�q=@��'       �	@x�Yc�A�3*

loss�^�=#�x�       �	o�x�Yc�A�3*

loss��=�=��       �	�Qy�Yc�A�3*

loss:l�;=X�;       �	�z�Yc�A�3*

losss\=$�Tp       �	B�z�Yc�A�3*

lossCv�;uh��       �	�y{�Yc�A�3*

lossC�";�_��       �	�j|�Yc�A�3*

lossEG<��       �	~ }�Yc�A�3*

lossVqj=�}�B       �	��}�Yc�A�3*

loss��:�K�4       �	/��Yc�A�3*

loss�)j<�"       �	�ဤYc�A�3*

lossj��<A�d       �	���Yc�A�3*

lossZՕ<g�-       �	#���Yc�A�3*

loss�b1<�t�       �	���Yc�A�3*

loss��<q��       �	�8��Yc�A�3*

loss~�=��F       �	�݄�Yc�A�3*

loss�,�<�c_�       �	����Yc�A�3*

loss\̺<���B       �	4ۆ�Yc�A�3*

loss#��;��X�       �	-z��Yc�A�3*

lossi%<[h�%       �	���Yc�A�3*

lossR�;`��       �	7�Yc�A�3*

loss'�<{ٗ�       �	�e��Yc�A�3*

lossow<+�oC       �	0��Yc�A�3*

loss$��<~��       �	ձ��Yc�A�3*

lossi�#=+LB       �	 ^��Yc�A�3*

loss�V0=���       �	~���Yc�A�3*

lossMȶ;��1       �	����Yc�A�3*

lossc�=ӎc       �	�1��Yc�A�3*

loss�/5<��Ԁ       �	�ҍ�Yc�A�3*

loss_�a<>2�       �	���Yc�A�3*

lossA��<Q�       �	_���Yc�A�3*

loss��j<ǻP       �	�4��Yc�A�3*

loss�c<��
R       �	d̐�Yc�A�3*

loss��;#*ǥ       �	%x��Yc�A�3*

loss?��<��h       �	�j��Yc�A�3*

lossa��;W��       �	y��Yc�A�3*

lossq�m;��\�       �	�ϓ�Yc�A�3*

loss���;-#�       �	m6��Yc�A�3*

loss�Y=H�mz       �	|ѫ�Yc�A�3*

loss�`Z=���       �	�m��Yc�A�3*

lossQd�;� 6       �	G��Yc�A�3*

lossd�|<ŵ��       �	'���Yc�A�3*

loss�b=�$�       �	����Yc�A�3*

loss��l<qՖ;       �	����Yc�A�3*

loss�=.��       �	Ό��Yc�A�3*

loss�e=zk�       �	�:��Yc�A�3*

loss���=���       �	� ��Yc�A�3*

loss�w�<��P       �	�沤Yc�A�3*

loss�*�<�A�       �	`���Yc�A�3*

loss��V=H��       �	�r��Yc�A�3*

loss���<��.       �	�)��Yc�A�3*

lossu��<h�Í       �	N��Yc�A�3*

loss!�"<FK��       �	y��Yc�A�3*

loss���:!C�\       �	���Yc�A�3*

loss=�#<��       �	Eg��Yc�A�3*

loss<n%<ͩ��       �	i��Yc�A�3*

loss��<��[       �	8J��Yc�A�3*

loss�q
<�\9�       �	�㺤Yc�A�3*

lossH��<��       �	����Yc�A�3*

lossx�'<�� �       �	TƼ�Yc�A�3*

lossO8=fQ�        �	�a��Yc�A�3*

loss�YE<F0$U       �	���Yc�A�3*

loss��P;��%       �	g���Yc�A�3*

loss�i�=w���       �	�H��Yc�A�3*

loss*Py;L��F       �	�㿤Yc�A�3*

lossEp�<�ԗ�       �	����Yc�A�3*

loss�"B<�[�       �	�3��Yc�A�3*

loss��<�pTE       �	����Yc�A�3*

loss %�<Pf��       �	fk¤Yc�A�3*

loss�<��d�       �	�äYc�A�3*

lossNQ=�r�       �	?tĤYc�A�3*

loss���<]�A       �	�'ŤYc�A�3*

loss1��<3B�       �	�ŤYc�A�3*

loss�w�:N9Z       �	N{ƤYc�A�3*

loss�?)=:/�`       �	'ǤYc�A�3*

loss?�B=��~       �	��ǤYc�A�3*

loss�=�i�       �	�iȤYc�A�3*

loss�T�;x�o�       �	RɤYc�A�3*

loss�;�Vv7       �	ŭɤYc�A�3*

loss�}=�g�       �	IGʤYc�A�3*

loss��[<�m(l       �	��ʤYc�A�3*

loss� G=�x]       �	��ˤYc�A�3*

losssa�<��Y       �	�=̤Yc�A�3*

lossO�<B[�B       �	"�̤Yc�A�3*

loss��(=v��       �	w�ͤYc�A�3*

loss\b<}K(�       �	*ΤYc�A�3*

loss�l:4��       �	��ΤYc�A�3*

loss���;�Z#l       �	zqϤYc�A�3*

loss��=�<{�       �	�ФYc�A�3*

loss�ǭ<)���       �	j�ФYc�A�3*

loss�=����       �	�^ѤYc�A�3*

loss)ː<�1��       �	�ҤYc�A�3*

lossF �;P�H{       �	��ҤYc�A�4*

lossd��<�$:       �	�%ԤYc�A�4*

lossN�<��:       �	��ԤYc�A�4*

loss�j�;|Bk�       �	��դYc�A�4*

lossL\�<����       �	|�֤Yc�A�4*

loss�6=�\��       �	�9פYc�A�4*

loss=�B;�(�       �	p�פYc�A�4*

loss:և=G���       �	jؤYc�A�4*

loss��d<Q���       �	�٤Yc�A�4*

loss�[<F,�s       �	��٤Yc�A�4*

lossf��;	[�       �	r7ڤYc�A�4*

loss�M<H�       �	_�ڤYc�A�4*

loss<zC<��&l       �	�iۤYc�A�4*

loss�V =ǷHh       �	6ܤYc�A�4*

lossz=Dv�       �	��ܤYc�A�4*

loss-]<����       �	%\ݤYc�A�4*

lossj+�=��f       �	/�ݤYc�A�4*

lossU��=B�C�       �	��ޤYc�A�4*

loss*�<�5�o       �	TVߤYc�A�4*

loss��B=w       �	��ߤYc�A�4*

loss��<1GD       �	��Yc�A�4*

lossq�;L@^�       �	G�Yc�A�4*

loss(�(=�9�       �	��Yc�A�4*

lossV`�;}@��       �	�S�Yc�A�4*

lossQ�V<�6�-       �	�Yc�A�4*

loss�5�='ٺ�       �	��Yc�A�4*

lossX��<��       �		p�Yc�A�4*

lossh�b<��|<       �	N	�Yc�A�4*

lossM��;O�       �	#��Yc�A�4*

lossQTK<��Ӱ       �	`=�Yc�A�4*

loss1eY;
)6       �	���Yc�A�4*

loss�R=����       �	���Yc�A�4*

lossr�=��       �	�*�Yc�A�4*

loss��<3��       �	��Yc�A�4*

loss���<�<�       �	n��Yc�A�4*

loss�VI<_<Đ       �	FC�Yc�A�4*

loss�v=���5       �	z��Yc�A�4*

loss�a�;�x�       �	9��Yc�A�4*

loss�T'<�D�?       �	�Yc�A�4*

loss�at=�xG�       �	*��Yc�A�4*

loss��<՚�/       �	^��Yc�A�4*

loss��=��       �	����Yc�A�4*

loss�I�<%j�f       �	�1�Yc�A�4*

loss��=�K�'       �	hy�Yc�A�4*

loss�ݬ<_��       �	`�Yc�A�4*

loss���;�H��       �	1�Yc�A�4*

loss�<FSYu       �	�J�Yc�A�4*

loss-T�=S�A�       �	��Yc�A�4*

loss���;~��       �	�b��Yc�A�4*

loss��=��]�       �	�g��Yc�A�4*

loss�h<��I       �	�8��Yc�A�4*

lossz=l��'       �	����Yc�A�4*

loss�0�;�6�       �	���Yc�A�4*

loss�={Ň       �	U2��Yc�A�4*

loss�N�<��'�       �	����Yc�A�4*

loss*�<�PO�       �	Pq��Yc�A�4*

lossʄ<_O��       �	ʇ��Yc�A�4*

loss{�<Z��       �	�#��Yc�A�4*

loss��;>,��       �	C���Yc�A�4*

loss�#�<ob�       �	n��Yc�A�4*

loss�)�;8��k       �	
��Yc�A�4*

loss�X<��2       �	� ��Yc�A�4*

loss*G==���C       �	J���Yc�A�4*

lossi��<F��       �	�3 �Yc�A�4*

lossv ;��       �	�� �Yc�A�4*

loss)9m<v &9       �	�k�Yc�A�4*

loss�?=@�       �	��Yc�A�4*

loss�T<����       �	���Yc�A�4*

loss�I=~\       �	�9�Yc�A�4*

loss��X;m�uY       �	���Yc�A�4*

loss��<�,       �	hx�Yc�A�4*

loss��=�       �	f�Yc�A�4*

loss\]u<N��       �	���Yc�A�4*

loss���<�T�m       �	�G�Yc�A�4*

loss��;[�	       �	���Yc�A�4*

loss9=tb"       �	�t�Yc�A�4*

loss,��<8�+p       �	K�Yc�A�4*

loss3 �<Wd       �	S	�Yc�A�4*

loss,ej<�jF       �	��	�Yc�A�4*

losss�=��*       �	��
�Yc�A�4*

loss�u<��       �	�c�Yc�A�4*

lossv��<),��       �	�Yc�A�4*

lossػW<���       �	���Yc�A�4*

loss��K<�Ț-       �	G�Yc�A�4*

loss��9<�s��       �	���Yc�A�4*

lossO<�"��       �	W	�Yc�A�4*

loss�(P=+[	5       �	��Yc�A�4*

loss֫<V�`�       �	�Y�Yc�A�4*

loss��<zx�       �	���Yc�A�4*

loss=3=��o�       �	)��Yc�A�4*

loss��<��C�       �	�8�Yc�A�4*

loss��<J�       �	���Yc�A�4*

loss��1;H|�`       �	Q��Yc�A�4*

loss���;�h6p       �	T8�Yc�A�4*

loss{��<D`g�       �	���Yc�A�4*

loss���;�γ�       �	���Yc�A�4*

losss��<^��       �	ǁ�Yc�A�4*

lossH�O=�]       �	3�Yc�A�4*

lossߡ�<��$j       �	���Yc�A�4*

loss���<R2k4       �	eV�Yc�A�4*

loss
_N;7�B1       �	b��Yc�A�4*

loss���;���       �	n��Yc�A�4*

loss�[�=H��s       �	�!�Yc�A�4*

loss��=��"       �	3��Yc�A�4*

loss�<}}       �	}]�Yc�A�4*

loss���<�7�       �	���Yc�A�4*

loss��"=[P,�       �	���Yc�A�4*

loss�M<���       �	�U�Yc�A�4*

loss��N;~:��       �	y�Yc�A�4*

loss�=���S       �	~��Yc�A�4*

loss:5�<_Jh�       �	hZ�Yc�A�4*

loss�{<,��       �		��Yc�A�4*

loss�D�=�ꩣ       �	*� �Yc�A�4*

loss�~c<K7       �	�K!�Yc�A�4*

loss��<�r�       �	��!�Yc�A�4*

loss�%=�~;       �	w�"�Yc�A�4*

loss$��;6�       �	sJ#�Yc�A�4*

loss��;%�,9       �	#�#�Yc�A�4*

loss�C=<��5�       �	5�$�Yc�A�4*

loss��%<-�U       �	@%�Yc�A�4*

loss�;<��.       �	y�%�Yc�A�4*

loss�v�<0j0�       �	*�&�Yc�A�4*

loss}��<�pR       �	�7'�Yc�A�4*

loss:��<̾�       �	�'�Yc�A�4*

loss혃<���|       �	�x(�Yc�A�4*

loss��};��_�       �	*)�Yc�A�4*

loss�`==�]�C       �	�)�Yc�A�4*

lossf��;�x��       �	uX*�Yc�A�4*

loss,<��ǲ       �	3�*�Yc�A�5*

loss���<ᘩ�       �	͓+�Yc�A�5*

loss��<�H��       �	A+,�Yc�A�5*

loss �<��TQ       �	��,�Yc�A�5*

loss���<=D�       �	5|-�Yc�A�5*

loss�*�<�wO�       �	�.�Yc�A�5*

lossf@�<���%       �	�/�Yc�A�5*

loss<]�=E�G{       �	��/�Yc�A�5*

loss��<�j�       �	��0�Yc�A�5*

lossNa�;T�u       �	��1�Yc�A�5*

loss7�;<{l�       �	�H3�Yc�A�5*

loss��;=�_�       �	��3�Yc�A�5*

lossfя<�VI�       �	#�4�Yc�A�5*

loss�K<k���       �	X 5�Yc�A�5*

loss;#={-�       �	V�5�Yc�A�5*

loss�%=
�]\       �	�Z6�Yc�A�5*

loss���;\Bŭ       �	��6�Yc�A�5*

lossҲz<A�e�       �	b�7�Yc�A�5*

loss�3�<T��       �	�B8�Yc�A�5*

loss��l<�j�O       �	e�8�Yc�A�5*

loss�Q<�YQs       �	�9�Yc�A�5*

loss��@<�E4       �	pC:�Yc�A�5*

lossE�<a��       �	q�:�Yc�A�5*

losse(U==�.�       �	I�;�Yc�A�5*

loss|��<���       �	�<�Yc�A�5*

lossm<c��       �	�<�Yc�A�5*

loss&,�<�'�       �	�O=�Yc�A�5*

loss��<<���       �	�>�Yc�A�5*

lossE�,=��       �	��>�Yc�A�5*

loss�y�;�?ԫ       �	M�?�Yc�A�5*

loss��w;�d��       �	�:@�Yc�A�5*

loss7d=��p       �	��@�Yc�A�5*

loss_=t=3���       �	�A�Yc�A�5*

loss��<���N       �	�BB�Yc�A�5*

loss��<y��       �	��B�Yc�A�5*

loss�	;=�E�E       �	>yC�Yc�A�5*

loss���<�m �       �	�D�Yc�A�5*

lossWz�;�n�I       �	��D�Yc�A�5*

lossë <Ы�       �	�IE�Yc�A�5*

loss�pu<t�+       �	q�E�Yc�A�5*

loss�z�=�f�q       �	{F�Yc�A�5*

loss4��<���       �	�G�Yc�A�5*

loss���<���       �	j�G�Yc�A�5*

lossm_"<�DZ       �	4�H�Yc�A�5*

loss���<�k��       �	?�I�Yc�A�5*

loss,�;!wd�       �	�]J�Yc�A�5*

loss!��<�\�       �	��J�Yc�A�5*

loss�u�;���       �	��K�Yc�A�5*

loss.�;m��       �	IHL�Yc�A�5*

loss[�=���       �	B�L�Yc�A�5*

lossn�a<iV�       �	��M�Yc�A�5*

loss�l�<WRP�       �	7N�Yc�A�5*

loss�=��j�       �	0�N�Yc�A�5*

loss�)
=BN-�       �	/RO�Yc�A�5*

loss��<�UR�       �	�P�Yc�A�5*

loss�U=~HX�       �	A�P�Yc�A�5*

loss|J<}M>X       �	�[Q�Yc�A�5*

loss�-�;���       �	#�Q�Yc�A�5*

loss��<�⻾       �	��R�Yc�A�5*

loss�>=����       �	E*S�Yc�A�5*

lossa:=s>L       �	��S�Yc�A�5*

lossZ�0=o�f�       �	1_T�Yc�A�5*

lossι�<��Z�       �	Z.U�Yc�A�5*

loss*=�?)       �	��U�Yc�A�5*

loss���<[�c�       �	LoV�Yc�A�5*

lossSn�=&8@�       �	AW�Yc�A�5*

lossx�<*.ڀ       �	.�W�Yc�A�5*

lossV��<n��       �	�CX�Yc�A�5*

lossQK<��8       �	�X�Yc�A�5*

loss�p�=T�"       �	�qY�Yc�A�5*

loss;4�;�%2       �	> Z�Yc�A�5*

lossV��:�}X�       �	]�Z�Yc�A�5*

loss%��;hx-       �	�d[�Yc�A�5*

lossy�<A�x       �	�\�Yc�A�5*

loss�i<�<+�       �	�\�Yc�A�5*

loss�=��       �	�H]�Yc�A�5*

loss�!%=ЍpU       �	 �]�Yc�A�5*

loss��<-֧�       �	��^�Yc�A�5*

loss$=��       �	�:_�Yc�A�5*

loss]�=���       �	�_�Yc�A�5*

loss��_< Ն       �	ۆ`�Yc�A�5*

lossVR<=NJ(�       �	!"a�Yc�A�5*

lossO�,<�G+�       �	z�a�Yc�A�5*

loss	�=4k       �	B�b�Yc�A�5*

loss�Ι=��'�       �	r1c�Yc�A�5*

loss(�<7��       �	)�c�Yc�A�5*

lossr".=+��B       �	jd�Yc�A�5*

loss)�<���       �	Xe�Yc�A�5*

loss��<�]��       �	;�e�Yc�A�5*

loss_	=���	       �	�Af�Yc�A�5*

loss��C=�'l�       �	f�f�Yc�A�5*

loss���=�v%c       �	��g�Yc�A�5*

loss��=pl^�       �	�3h�Yc�A�5*

loss���<`��       �	��h�Yc�A�5*

loss�u=�٨       �	�di�Yc�A�5*

loss�$�=���5       �	��i�Yc�A�5*

loss���<;<%       �	B�j�Yc�A�5*

loss��A;���       �	�4k�Yc�A�5*

loss�\<���       �	��k�Yc�A�5*

loss&A�;@f�4       �	8il�Yc�A�5*

loss}2]<��       �	�m�Yc�A�5*

loss�a<�~       �	+�m�Yc�A�5*

lossM�|=��       �	]�n�Yc�A�5*

lossHo�<�D       �	�Yo�Yc�A�5*

loss�)<<��sX       �	��p�Yc�A�5*

loss,��<�A�?       �	4�q�Yc�A�5*

loss��<�oA       �	@kr�Yc�A�5*

lossJH�:�
r�       �	s�Yc�A�5*

loss���</G�       �	��s�Yc�A�5*

lossg�<�e[�       �	]�t�Yc�A�5*

loss�e;�8�       �	�u�Yc�A�5*

loss|�<��N�       �	�fv�Yc�A�5*

loss��<Cs�       �	�Rw�Yc�A�5*

loss�|X<�j5�       �	�#x�Yc�A�5*

loss$�|<��>       �	8�x�Yc�A�5*

lossX�%=i��\       �	��y�Yc�A�5*

loss�j�<�}��       �	�.z�Yc�A�5*

lossw�4=sm:�       �	��z�Yc�A�5*

loss�|�<���       �	�_{�Yc�A�5*

loss��;�0��       �	�{�Yc�A�5*

loss!�X=���@       �	��|�Yc�A�5*

loss6B=h�9       �	4}�Yc�A�5*

loss.3�=K":       �	"�}�Yc�A�5*

lossL3�< *��       �	�[~�Yc�A�5*

loss޼�<��P       �	��~�Yc�A�5*

loss`3�<�S�       �	���Yc�A�5*

loss�:�;,��J       �	hx��Yc�A�5*

loss�<ZG��       �	���Yc�A�5*

loss�1=��@�       �	����Yc�A�6*

loss�_�<W��       �	�S��Yc�A�6*

loss�!�<���       �	���Yc�A�6*

loss��;�q       �	o���Yc�A�6*

loss��;=�D��       �	I�Yc�A�6*

loss��<�E       �	����Yc�A�6*

loss6�<<4�       �	�F��Yc�A�6*

loss���<���       �	����Yc�A�6*

loss;�<���F       �	f���Yc�A�6*

loss���<���,       �	4M��Yc�A�6*

loss�;=c�?�       �	u��Yc�A�6*

loss���<�^9       �		ĉ�Yc�A�6*

lossO��<;Tӎ       �	�]��Yc�A�6*

loss�>�<�#0�       �	����Yc�A�6*

loss6�x=�퍜       �	g���Yc�A�6*

loss;��<��w�       �	�G��Yc�A�6*

loss���;�}�G       �	`䌥Yc�A�6*

loss& <�w��       �	��Yc�A�6*

lossf�=�6�       �	Tp��Yc�A�6*

lossNCQ=���       �	���Yc�A�6*

lossY)=�R��       �	���Yc�A�6*

loss_�:�޼c       �	�$��Yc�A�6*

loss|:<ޱ�^       �	,���Yc�A�6*

loss��<Ss�*       �	�Q��Yc�A�6*

loss�==�$��       �	�撥Yc�A�6*

loss�4<�S!�       �	_~��Yc�A�6*

loss;�<�@?�       �	{��Yc�A�6*

loss�.�<~B�       �	 ���Yc�A�6*

loss��h;V矍       �	�H��Yc�A�6*

loss�'�;`ZL.       �	�敥Yc�A�6*

loss�{<Xu��       �	����Yc�A�6*

loss��<�2G�       �	=��Yc�A�6*

lossi!=���       �	H◥Yc�A�6*

loss�V@=g��       �	x{��Yc�A�6*

loss�=�L�       �	���Yc�A�6*

loss-}5;�ɔh       �	ɭ��Yc�A�6*

loss��i<@���       �	be��Yc�A�6*

loss��o<�u�r       �	.��Yc�A�6*

lossh��<!��       �	斛�Yc�A�6*

lossI&<�A��       �	�-��Yc�A�6*

loss�A=��ŷ       �	�Ŝ�Yc�A�6*

lossĎF<�       �	�]��Yc�A�6*

lossXf<F��       �	���Yc�A�6*

loss�q�;��qL       �	���Yc�A�6*

lossҲ<�c�       �	�H��Yc�A�6*

loss2�<���M       �	&៥Yc�A�6*

loss�;�ut�       �	�{��Yc�A�6*

lossl�&<ہ�P       �	���Yc�A�6*

loss@ɰ<����       �	K���Yc�A�6*

loss�X�<8W�       �	cA��Yc�A�6*

loss&b�;f%�B       �	yꢥYc�A�6*

loss�7e<�fa       �	o֣�Yc�A�6*

loss �r=�F?�       �	�j��Yc�A�6*

loss8�=SnW       �	��Yc�A�6*

loss*�2=��5       �	�ӥ�Yc�A�6*

loss��<
>�       �	�q��Yc�A�6*

loss��2<�G��       �	�G��Yc�A�6*

loss�;�<� �       �	o�Yc�A�6*

loss��*<��{"       �	����Yc�A�6*

loss�*�<����       �	u9��Yc�A�6*

loss��_<b�r       �	T䩥Yc�A�6*

lossa��<�'       �	���Yc�A�6*

loss�!f<שm,       �	�&��Yc�A�6*

lossۋ=!O�       �	\ɫ�Yc�A�6*

loss'\�=��       �	c~��Yc�A�6*

loss�=�H�F       �	���Yc�A�6*

lossh�=p���       �	*ĭ�Yc�A�6*

lossj3@=��_�       �	wd��Yc�A�6*

loss�=�J�       �	���Yc�A�6*

lossq�F<�^�       �	 ���Yc�A�6*

loss��<O�6|       �	xF��Yc�A�6*

loss��	=�%       �	�>��Yc�A�6*

loss��u;�w��       �	[鱥Yc�A�6*

loss3�R<Ş,       �	����Yc�A�6*

lossF��<^'z1       �	((��Yc�A�6*

loss��<�4JZ       �	˳�Yc�A�6*

losst=w�       �	���Yc�A�6*

loss3�D<y���       �	�-��Yc�A�6*

loss(�;�c��       �	;���Yc�A�6*

loss���;ł       �	����Yc�A�6*

loss���<)u��       �	阷�Yc�A�6*

loss
 <P%       �	�=��Yc�A�6*

loss/Ă=�?y       �	ܸ�Yc�A�6*

lossXz#<�{�y       �	�r��Yc�A�6*

loss-�=U���       �	���Yc�A�6*

loss�#~:/ʓ{       �	����Yc�A�6*

lossT��;��       �	\8��Yc�A�6*

loss�Q�;�%��       �	�h��Yc�A�6*

loss�R<�,��       �	��Yc�A�6*

loss_
�< ��       �	���Yc�A�6*

loss��<���D       �	����Yc�A�6*

loss]?=�p��       �	K\��Yc�A�6*

lossTa;��D       �	���Yc�A�6*

lossm%x<�)�8       �	m���Yc�A�6*

loss�PH<w�"�       �	�1��Yc�A�6*

losscb
=����       �	����Yc�A�6*

loss�L;=�i�       �	?r¥Yc�A�6*

loss�k�;HJ�       �	� åYc�A�6*

loss_�C;��]       �	��åYc�A�6*

lossׇ%=^Cr4       �	kĥYc�A�6*

loss_�Y<�+�       �	[AťYc�A�6*

loss�ͺ;�t�       �	�ťYc�A�6*

loss?`�=Rv�S       �	ǥYc�A�6*

loss �=���-       �	�ǥYc�A�6*

loss�5G=�x�U       �	W�ȥYc�A�6*

loss��Z;Ő1&       �	�1ɥYc�A�6*

loss�<�/�H       �	��ɥYc�A�6*

loss�<n<\�       �	�ʥYc�A�6*

loss��3="��       �	�)˥Yc�A�6*

loss�T<�\(       �	e�˥Yc�A�6*

loss6��<G���       �	�̥Yc�A�6*

loss҉<]>�       �	�ͥYc�A�6*

loss�=/u�:       �	��ͥYc�A�6*

lossZD�<D�47       �	�_ΥYc�A�6*

lossr�<)ʵ�       �	��ΥYc�A�6*

loss��[<��       �	`�ϥYc�A�6*

loss�s;<h��       �	_)ХYc�A�6*

lossG=�wk       �	��ХYc�A�6*

loss��<T1r/       �	��ѥYc�A�6*

loss�I�<G(I       �	V*ҥYc�A�6*

loss_<���       �	��ҥYc�A�6*

loss��?<��3�       �	G�ӥYc�A�6*

loss��:<�(+�       �	X�ԥYc�A�6*

loss�.=z�z       �	lCեYc�A�6*

loss]4�<���       �	��եYc�A�6*

loss2B=�,�       �	Kt֥Yc�A�6*

loss!]�<ujj�       �	�:ץYc�A�6*

loss1�< �k       �	��ץYc�A�6*

lossdM?<R�;+       �	e�إYc�A�7*

loss]OE=c]�w       �	��٥Yc�A�7*

loss��=w��z       �	yvڥYc�A�7*

loss�n=[2@�       �	�ۥYc�A�7*

loss��H=
-�       �	o�ۥYc�A�7*

loss�M�;�b�'       �	`ܥYc�A�7*

lossƮ�<*�ͮ       �	�ݥYc�A�7*

loss�=^�e       �	N�ݥYc�A�7*

lossa�<��a�       �	fNޥYc�A�7*

loss��<C���       �	m�ޥYc�A�7*

loss���;3C̵       �	3�ߥYc�A�7*

loss([<�.�       �	6?�Yc�A�7*

loss� B<���r       �	���Yc�A�7*

loss��:��3       �	!t�Yc�A�7*

loss�H=�}]�       �	�:�Yc�A�7*

loss_F�=\�
       �	
��Yc�A�7*

losst %=�Ql�       �	yv�Yc�A�7*

loss��r<_F�       �	��Yc�A�7*

lossDۃ<�C�       �	@��Yc�A�7*

loss(�I<���z       �	5_�Yc�A�7*

loss.�q<�ⱕ       �	��Yc�A�7*

loss���<�e�       �	i��Yc�A�7*

lossak�<<EM�       �	�Y�Yc�A�7*

loss&Dz;:��       �	���Yc�A�7*

lossQ��<ۿЬ       �	l��Yc�A�7*

lossE�V<�O�       �	o.�Yc�A�7*

loss�k=�p�8       �	q��Yc�A�7*

lossA��<��4       �	=a�Yc�A�7*

lossn��;�aK       �	G��Yc�A�7*

loss�<��       �	���Yc�A�7*

loss;�;�W       �	�/�Yc�A�7*

loss�<�ZS�       �	���Yc�A�7*

loss!�<��cg       �	�k��Yc�A�7*

loss��=iK�       �	��Yc�A�7*

loss{|�<��\�       �	?��Yc�A�7*

loss�
�;� -�       �	�H�Yc�A�7*

loss�k<��jl       �	���Yc�A�7*

loss��E;�3rh       �	���Yc�A�7*

lossC <���       �	�;�Yc�A�7*

loss\M*<�I�'       �	���Yc�A�7*

loss-��=os^�       �	�w�Yc�A�7*

loss�:]=�       �	��Yc�A�7*

lossb-=%՜�       �	%��Yc�A�7*

loss�θ<�LM       �	����Yc�A�7*

loss��D;�V       �	e��Yc�A�7*

losss�<�جg       �	�;��Yc�A�7*

loss�f�<ر%�       �	����Yc�A�7*

lossMY;KV��       �	s��Yc�A�7*

loss���<SC;       �	 D��Yc�A�7*

loss@�;w���       �	����Yc�A�7*

loss�4p=��l�       �	V���Yc�A�7*

loss?[=Z7~       �	���Yc�A�7*

loss*L8=���       �	����Yc�A�7*

lossZ��:m��       �	Q��Yc�A�7*

loss��P<<��f       �	����Yc�A�7*

loss+�<Fђ       �	����Yc�A�7*

lossh��<O�       �	�?��Yc�A�7*

loss]Ӎ<Fy��       �	\>��Yc�A�7*

loss<�8A       �	D���Yc�A�7*

loss���<�~��       �	����Yc�A�7*

loss�)p<:�P       �	aT �Yc�A�7*

loss��!=홹4       �	�� �Yc�A�7*

lossQH><�{S7       �	���Yc�A�7*

loss��;����       �	�*�Yc�A�7*

loss�C<��F�       �	���Yc�A�7*

lossD��<�$��       �	jj�Yc�A�7*

losss�2<W1�       �	��Yc�A�7*

loss� =��       �	��Yc�A�7*

loss\r<�[       �	S]�Yc�A�7*

loss�>�<���       �	��Yc�A�7*

loss��+<��l*       �	��Yc�A�7*

loss:��;Q�Xx       �	�k�Yc�A�7*

loss��<IM�       �	(
�Yc�A�7*

lossA4 <R;�       �	~��Yc�A�7*

loss���<<       �	�C	�Yc�A�7*

lossc�/<�F�Z       �	��	�Yc�A�7*

lossV4=���~       �	Z~
�Yc�A�7*

lossJh7<�{�f       �	��Yc�A�7*

loss�&;����       �	R��Yc�A�7*

loss�E�;�_�U       �	 a�Yc�A�7*

loss4\�<.ki�       �	a��Yc�A�7*

loss�P<u���       �	ƣ�Yc�A�7*

loss��j<����       �	;�Yc�A�7*

loss1,�;� ��       �	s�Yc�A�7*

loss <�b�`       �	&9�Yc�A�7*

loss���<�ӣ       �	���Yc�A�7*

loss��;�S�       �	�p�Yc�A�7*

loss���<��CY       �	��Yc�A�7*

losso�=nI��       �	��Yc�A�7*

loss䉧=�� G       �	�<�Yc�A�7*

loss��;�J?       �	���Yc�A�7*

loss#m;�mZ�       �	�o�Yc�A�7*

loss!_�<�Ld       �	�
�Yc�A�7*

loss@3;)�y�       �	���Yc�A�7*

loss?�:�D�       �	nL�Yc�A�7*

lossȣ;�:I       �	���Yc�A�7*

lossӼ;���       �	{��Yc�A�7*

loss@�/<���       �	��Yc�A�7*

lossF�<ӫ��       �	S��Yc�A�7*

loss��;�N�       �	&Q�Yc�A�7*

loss/Yb;��       �	2��Yc�A�7*

loss�i�<��C�       �	���Yc�A�7*

loss?��9k��       �	y"�Yc�A�7*

lossa�;@=�       �	���Yc�A�7*

loss�y<o-5z       �	Y�Yc�A�7*

loss��<ٲ��       �	m�Yc�A�7*

loss�(�;R�
6       �	���Yc�A�7*

loss*jD:�I)       �	�A�Yc�A�7*

loss֩�;н�       �	�( �Yc�A�7*

loss1}=*�O       �	+� �Yc�A�7*

lossN=�:�A�       �	�f!�Yc�A�7*

lossD�:Õ�p       �	�/"�Yc�A�7*

loss��<��*       �	��"�Yc�A�7*

loss�k�<��K�       �	�s#�Yc�A�7*

loss��(<�-"       �	�$�Yc�A�7*

loss��<%       �	��$�Yc�A�7*

lossL�	=�Z�4       �	E�%�Yc�A�7*

lossd�=3��       �	T&�Yc�A�7*

lossR�y=W���       �	^�&�Yc�A�7*

loss&C�< 澣       �	:Z'�Yc�A�7*

loss���<�Gb�       �	0�'�Yc�A�7*

loss��<f��       �	-�(�Yc�A�7*

loss<>=�%u       �	�Q)�Yc�A�7*

losss\�<s��       �	W�)�Yc�A�7*

loss�S�<�8�       �	��*�Yc�A�7*

loss�Ǖ<\�ӹ       �	 7+�Yc�A�7*

loss<0�<~�@=       �	�+�Yc�A�7*

lossw�)<j�       �	ܟ,�Yc�A�7*

loss!:�<�F��       �	�8-�Yc�A�8*

loss`#<A.h
       �	�-�Yc�A�8*

loss�=;;�r       �	�o.�Yc�A�8*

loss��g<Q��       �	(/�Yc�A�8*

loss�p<<��       �	K:0�Yc�A�8*

loss�d�;�ϐ�       �	mu1�Yc�A�8*

lossŔt;ܺ_�       �	��2�Yc�A�8*

loss��;�&�       �	Ll3�Yc�A�8*

loss-�~<��E�       �	$
4�Yc�A�8*

loss <}��       �	��4�Yc�A�8*

loss��<<�4�       �	d5�Yc�A�8*

loss��&=u~i�       �	�6�Yc�A�8*

lossMh&<�'        �	se7�Yc�A�8*

loss��<2�t       �	�L8�Yc�A�8*

lossX�;	�V       �	q9�Yc�A�8*

loss�C	;��       �	a�9�Yc�A�8*

lossQ�;U�{       �	sg:�Yc�A�8*

lossN*�;���K       �	6;�Yc�A�8*

loss
=��߯       �	J�;�Yc�A�8*

loss�6�<R��a       �	 U<�Yc�A�8*

loss���<���       �	��<�Yc�A�8*

loss!߅<�{��       �	.�=�Yc�A�8*

lossc��;����       �	4>�Yc�A�8*

lossS�=%�8�       �	t�>�Yc�A�8*

loss*eG<���       �	�F@�Yc�A�8*

losss#)<����       �	#�@�Yc�A�8*

loss�[<�Ȍ       �	*rA�Yc�A�8*

loss�Ld=ЯG�       �	�HB�Yc�A�8*

loss]W=�C�R       �	��B�Yc�A�8*

loss��;-j/e       �	ݙC�Yc�A�8*

loss���<�(�*       �	
,D�Yc�A�8*

loss�^<�C       �	�D�Yc�A�8*

loss&n�;��       �	SyE�Yc�A�8*

loss��_<p��