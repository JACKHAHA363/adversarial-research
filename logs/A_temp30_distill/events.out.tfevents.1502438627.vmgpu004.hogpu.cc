       �K"	  �8Yc�Abrain.Event:2}���#�     h�k�	F�8Yc�A"��
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
seed2���*
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
:@@*
seed2���*
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
:���������@*
seed2��%
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
Index0*
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
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2�ȯ*
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
seed2���*
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
seed2���
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2ħ�*
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
:����������*
seed2���
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
PlaceholderPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������

L
div_2/yConst*
valueB
 *  �A*
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
valueB:*
_output_shapes
:*
dtype0
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
value	B : *
dtype0*
_output_shapes
: 
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
out_type0*
_output_shapes
:*
T0
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
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0
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
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
_output_shapes
: *
T0
�
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
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
<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like	ZerosLike softmax_cross_entropy_loss_1/div*
_output_shapes
: *
T0
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
valueB *
dtype0*
_output_shapes
: 
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
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
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
_gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 
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
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
out_type0*
_output_shapes
:*
T0
�
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
gradients/div_2_grad/RealDiv_2RealDivgradients/div_2_grad/RealDiv_1div_2/y*'
_output_shapes
:���������
*
T0
�
gradients/div_2_grad/mulMul;gradients/softmax_cross_entropy_loss_1/Reshape_grad/Reshapegradients/div_2_grad/RealDiv_2*
T0*'
_output_shapes
:���������

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
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
T0*
out_type0*
_output_shapes
:
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
valueB@*    *
_output_shapes
:@*
dtype0
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
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "Ji��,     ��#�	��8Yc�AJ��
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
conv2d_1_inputPlaceholder*/
_output_shapes
:���������*
shape: *
dtype0
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
seed2���
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
:@@*
seed2���
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
:���������@*
seed2��%
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
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
seed2�ȯ*
T0*
seed���)*
dtype0
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
seed2���*
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
seed2ħ�
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
 *    *
_output_shapes
: *
dtype0
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
]
PlaceholderPlaceholder*
shape: *
dtype0*'
_output_shapes
:���������

L
div_2/yConst*
valueB
 *  �A*
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
value	B :*
_output_shapes
: *
dtype0
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
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
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
 *  �?*
dtype0*
_output_shapes
: 
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
valueB: *
dtype0*
_output_shapes
:
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
 *    *
_output_shapes
: *
dtype0
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
valueB *
_output_shapes
: *
dtype0
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
valueB *
dtype0*
_output_shapes
: 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
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
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
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
 *    *
_output_shapes
: *
dtype0
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
=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like	ZerosLike&softmax_cross_entropy_loss_1/ones_like*
T0*
_output_shapes
: 
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
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
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
5gradients/softmax_cross_entropy_loss_1/Mul_grad/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2*
out_type0*
_output_shapes
:*
T0
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
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 
�
dgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentitySgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������*
T0
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
���������*
dtype0*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
gradients/div_2_grad/SumSumgradients/div_2_grad/RealDiv*gradients/div_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
T0*
N
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
valueB"         @   *
_output_shapes
:*
dtype0
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
valueB@*    *
dtype0*
_output_shapes
:@
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0��ї       ��-	��9Yc�A*

lossBb@�>�       ��-	B@9Yc�A*

lossq�@龝�       ��-	
�9Yc�A*

loss==@�J"�       ��-	��9Yc�A*

loss��@f��       ��-	�E9Yc�A*

loss$�
@N���       ��-	��9Yc�A*

lossP@FU�k       ��-	�9Yc�A*

lossra�?[�0.       ��-	�-9Yc�A*

loss[��?�2��       ��-	��9Yc�A	*

loss���?|~       ��-	�f9Yc�A
*

loss\��?�p|       ��-	>9Yc�A*

loss���?�i"%       ��-	��9Yc�A*

loss!:�?|��       ��-	YM9Yc�A*

lossm׮?!���       ��-	{�9Yc�A*

lossE�?���%       ��-	%�9Yc�A*

lossJՔ?��qy       ��-	s*9Yc�A*

loss��S?e��       ��-	��9Yc�A*

lossK��?0��       ��-	�f9Yc�A*

loss#	�?j��       ��-	�9Yc�A*

lossӓ�?��P       ��-	b�9Yc�A*

loss�6p?����       ��-	�>9Yc�A*

loss�;V?e��       ��-	
�9Yc�A*

lossD�`?��"+       ��-	x{9Yc�A*

lossJ��?��i       ��-	@9Yc�A*

loss�\s?PJ�       ��-	N�9Yc�A*

losssC7?�J9�       ��-	vR9Yc�A*

loss�[?c��
       ��-	��9Yc�A*

loss���?�L�s       ��-	?� 9Yc�A*

loss�l?��1       ��-	,!9Yc�A*

loss��f?���       ��-	��!9Yc�A*

lossZtZ?�M�       ��-	�_"9Yc�A*

loss� -?§�       ��-	��"9Yc�A *

loss�A@?*}9�       ��-	��#9Yc�A!*

loss	?�?�d       ��-	y"$9Yc�A"*

lossn?U��;       ��-	��$9Yc�A#*

lossMmQ?�5�       ��-	�n%9Yc�A$*

losscl�>V{)�       ��-	4&9Yc�A%*

lossyT?����       ��-	y�&9Yc�A&*

loss�ZR?�6t       ��-	4K'9Yc�A'*

loss@�N?sAG       ��-	�z(9Yc�A(*

lossڻ?e�k�       ��-	)9Yc�A)*

losst��>���       ��-	��)9Yc�A**

lossc?�b��       ��-	L�*9Yc�A+*

loss��?����       ��-	��+9Yc�A,*

lossҡB?(*�       ��-	=),9Yc�A-*

loss��?w��@       ��-	��,9Yc�A.*

loss�0?�"�       ��-	0e-9Yc�A/*

loss�n?�w�       ��-	�(.9Yc�A0*

loss�a�>��       ��-	��.9Yc�A1*

loss&$�>a��       ��-	�r/9Yc�A2*

loss�b�>/�       ��-	�)09Yc�A3*

lossx�>2���       ��-	�09Yc�A4*

loss}��>ҫo�       ��-	�x19Yc�A5*

lossrQ]?n�"�       ��-	��29Yc�A6*

loss��>dB��       ��-	�j39Yc�A7*

loss;��>���$       ��-	�49Yc�A8*

loss��>g٘�       ��-	D�49Yc�A9*

loss��>��I&       ��-	}>59Yc�A:*

lossJY?���       ��-	��59Yc�A;*

loss__�>@J'�       ��-	�69Yc�A<*

lossi;�>���       ��-	�579Yc�A=*

loss�B�>y�r�       ��-	��79Yc�A>*

lossq1�>�O�|       ��-	�m89Yc�A?*

loss.q�>�4`?       ��-	\99Yc�A@*

loss?�>%df       ��-	��99Yc�AA*

loss���>��       ��-	�5:9Yc�AB*

lossRL%?�N��       ��-	��:9Yc�AC*

losso�?��J       ��-	��;9Yc�AD*

lossmD,?(j9�       ��-	�S<9Yc�AE*

loss
�>��uW       ��-	��<9Yc�AF*

loss���>�ЋN       ��-	L�=9Yc�AG*

loss7�>s��       ��-	�S>9Yc�AH*

loss���>�/�^       ��-	S�>9Yc�AI*

lossx��>j*^�       ��-	$?9Yc�AJ*

loss�
?�z�       ��-	�@9Yc�AK*

lossVa?���N       ��-	��@9Yc�AL*

lossZ�+?Wg,�       ��-	GYA9Yc�AM*

loss��@?
�1       ��-	�B9Yc�AN*

loss���>$(�       ��-	��B9Yc�AO*

loss�ӷ>���       ��-	O?C9Yc�AP*

loss�s?\�0       ��-	Z�C9Yc�AQ*

loss�2?�_        ��-	�sD9Yc�AR*

loss�?��p       ��-	E9Yc�AS*

loss���>�ڢ�       ��-	C�E9Yc�AT*

loss���>�uW�       ��-	�BF9Yc�AU*

loss��>�O�       ��-	��F9Yc�AV*

loss��>9}"�       ��-	�yG9Yc�AW*

loss��>v͘       ��-	�H9Yc�AX*

lossT�!?��{L       ��-	��H9Yc�AY*

loss1O�>O�       ��-	K\I9Yc�AZ*

loss���>���       ��-	;J9Yc�A[*

loss1x�>ُT�       ��-	9�J9Yc�A\*

loss�/�>���       ��-	�=K9Yc�A]*

loss��>a       ��-	 �K9Yc�A^*

loss?����       ��-	CsL9Yc�A_*

losstu�>��       ��-	NM9Yc�A`*

loss���>���v       ��-	��M9Yc�Aa*

loss�B�>��]       ��-	�BN9Yc�Ab*

lossWvP?�6�       ��-	E�N9Yc�Ac*

loss$�>�n�       ��-	�O9Yc�Ad*

lossnҨ>�@��       ��-	ZHP9Yc�Ae*

loss]�>D�'�       ��-		Q9Yc�Af*

loss�^�>�U�       ��-	%�Q9Yc�Ag*

lossV2�>�#_�       ��-	IHR9Yc�Ah*

loss�|>H�Q�       ��-	��R9Yc�Ai*

loss$��>]0��       ��-	�{S9Yc�Aj*

lossM9�>���.       ��-	T9Yc�Ak*

loss���>�n	       ��-	�T9Yc�Al*

loss���>Dј       ��-	MU9Yc�Am*

loss,?�>�$v?       ��-	��U9Yc�An*

loss��?����       ��-	U�V9Yc�Ao*

loss�r�>੏&       ��-	�W9Yc�Ap*

loss�>�K0�       ��-	��W9Yc�Aq*

loss���>t�A       ��-	�ZX9Yc�Ar*

loss�̇>{��!       ��-	��X9Yc�As*

loss�İ>���       ��-	��Y9Yc�At*

loss,�>\&u       ��-	�3Z9Yc�Au*

loss��>�K�       ��-	��Z9Yc�Av*

loss���>�﷬       ��-	�{[9Yc�Aw*

loss�>�q=       ��-	\9Yc�Ax*

loss�I�>ج��       ��-	�\9Yc�Ay*

lossem�>+��K       ��-	+N]9Yc�Az*

lossx�><��Q       ��-	��]9Yc�A{*

loss=�>ieIY       ��-	��^9Yc�A|*

lossWk>�n��       ��-	�_9Yc�A}*

loss��>1�O�       ��-	$F`9Yc�A~*

loss���>DAw       ��-	��`9Yc�A*

loss��>��z�       �	�a9Yc�A�*

lossS|�>���       �	SYb9Yc�A�*

loss��>�4k       �	��b9Yc�A�*

loss��>hA��       �	ץc9Yc�A�*

lossn�>���       �	|Dd9Yc�A�*

loss��J>%��w       �	#�d9Yc�A�*

loss��>��        �	"�e9Yc�A�*

loss�Œ>��i       �	J%f9Yc�A�*

loss��c>�A��       �	��g9Yc�A�*

loss�/�>�d3       �	�Ji9Yc�A�*

loss��><g�X       �	�zj9Yc�A�*

loss8P>m�y�       �	�%k9Yc�A�*

loss8�P>m�K       �	b0l9Yc�A�*

loss;[!>Ԃ�       �	m9Yc�A�*

loss/5�>A��       �	��m9Yc�A�*

loss@�V>D��       �	=Hn9Yc�A�*

loss��>Ȗ�4       �	�ho9Yc�A�*

loss1�>m���       �	p9Yc�A�*

loss*�:>���d       �	ߧp9Yc�A�*

loss���>���e       �	Lq9Yc�A�*

lossc�>���       �	��q9Yc�A�*

loss��`>�3       �	V~r9Yc�A�*

loss�l>F���       �	{s9Yc�A�*

loss��c>gu�W       �	`�s9Yc�A�*

lossF�>��m�       �	_Bt9Yc�A�*

loss�c�>�ԏ       �	G�t9Yc�A�*

loss.��>*��*       �	)zu9Yc�A�*

lossL�C>��3E       �	�v9Yc�A�*

loss�2>�t�       �	̷v9Yc�A�*

loss^ړ>��{F       �	�w9Yc�A�*

loss%��>��C�       �	�Bx9Yc�A�*

lossVh�>�AJ       �	��x9Yc�A�*

loss>{>'��       �	�y9Yc�A�*

loss�*�>m�S=       �	l$z9Yc�A�*

loss@l�>�'~s       �	�z9Yc�A�*

loss�(�>���       �	X{9Yc�A�*

loss�T(>�l       �	��{9Yc�A�*

loss�:�>^p"       �	��|9Yc�A�*

lossW~�>�\��       �	L3}9Yc�A�*

loss�Æ>�VnF       �	��}9Yc�A�*

loss�J�>�k�       �	�k~9Yc�A�*

loss�&f>	�>�       �	�9Yc�A�*

loss&�K>qH%�       �	��9Yc�A�*

loss)�}>��$�       �	�F�9Yc�A�*

loss�#>1���       �	��9Yc�A�*

loss6�>�WE       �	)z�9Yc�A�*

loss�Y�>q70       �	<�9Yc�A�*

lossJW>���       �	���9Yc�A�*

loss��> 2�y       �	�F�9Yc�A�*

loss(��>��_p       �	?�9Yc�A�*

loss�.�>�8��       �	�}�9Yc�A�*

loss��Y>TȠ4       �	�9Yc�A�*

loss��7>Lr�'       �	ݶ�9Yc�A�*

loss��>���       �	"T�9Yc�A�*

loss��>E��        �	��9Yc�A�*

loss�h>Y�       �	���9Yc�A�*

loss��=�*/       �	O!�9Yc�A�*

loss��3>����       �	k��9Yc�A�*

loss(X>�@       �	.V�9Yc�A�*

loss��1>�j*Q       �	��9Yc�A�*

loss�q�>���       �	)��9Yc�A�*

loss���>,���       �	R'�9Yc�A�*

lossӺ�>����       �	���9Yc�A�*

lossAj>{�=�       �	�\�9Yc�A�*

loss&O>�܏�       �	��9Yc�A�*

loss�np>���       �	���9Yc�A�*

loss�l�>Y���       �	쿎9Yc�A�*

loss�91>k`��       �	uV�9Yc�A�*

lossT��>ݟ��       �	s�9Yc�A�*

lossz��>�!Pe       �	NӐ9Yc�A�*

loss�s]>�P       �	��9Yc�A�*

loss6�>��&�       �	���9Yc�A�*

loss���>��ٌ       �	���9Yc�A�*

lossR��>�/��       �	���9Yc�A�*

loss�Q>��%       �	/��9Yc�A�*

loss�ڀ>~}��       �	�.�9Yc�A�*

loss�B>n.O*       �	s��9Yc�A�*

lossӫ#>�� �       �	���9Yc�A�*

loss�>�*�       �	�$�9Yc�A�*

loss�Z>�I�       �	K�9Yc�A�*

loss�:>�U.�       �	H�9Yc�A�*

loss�<>�~       �	�{�9Yc�A�*

loss�Ń>�/�       �	��9Yc�A�*

loss1@T>5HKv       �	�М9Yc�A�*

lossQ*>���c       �	�k�9Yc�A�*

loss\	6>La��       �	�9Yc�A�*

lossZ�>��Cz       �	���9Yc�A�*

loss&��>�&�       �	Af�9Yc�A�*

loss�>6 �E       �	��9Yc�A�*

loss9H>���       �	|��9Yc�A�*

loss�_�>-5�       �	W�9Yc�A�*

loss���>P��C       �	���9Yc�A�*

lossڒ�>�/�\       �	)��9Yc�A�*

loss<��==nb�       �	K:�9Yc�A�*

loss" >��'       �	wڣ9Yc�A�*

lossz/�>�.%�       �	[z�9Yc�A�*

loss��_>��=       �	��9Yc�A�*

loss�b>�{2.       �	|��9Yc�A�*

loss�QW>�ނ6       �	<��9Yc�A�*

loss��[>+~-�       �	l��9Yc�A�*

loss6�Y>��%�       �	g*�9Yc�A�*

loss��5>04��       �	P¨9Yc�A�*

loss��?>��Uk       �	1]�9Yc�A�*

loss�<#>����       �	{��9Yc�A�*

loss�>�AV       �	���9Yc�A�*

loss\i�>�]�u       �	�&�9Yc�A�*

losss�1>���.       �	Pū9Yc�A�*

lossV%�>�B6�       �	�\�9Yc�A�*

loss� >U��       �	���9Yc�A�*

loss�i{>5���       �	���9Yc�A�*

lossD��>�       �	�S�9Yc�A�*

lossE]�>�bx       �	}�9Yc�A�*

lossZ�H>��;|       �	��9Yc�A�*

loss�&�=��=       �	�,�9Yc�A�*

lossׅ<>gXί       �	�Ű9Yc�A�*

loss@v�>��%�       �	1a�9Yc�A�*

lossD�Z>�5"�       �	��9Yc�A�*

loss8m�>��       �	��9Yc�A�*

lossl�E>#�j       �	�X�9Yc�A�*

loss�/>*]�       �	���9Yc�A�*

loss�R.>F6��       �	k��9Yc�A�*

loss�݈>����       �	댵9Yc�A�*

loss��>�@/       �	sK�9Yc�A�*

lossƪ�>y뵣       �	p�9Yc�A�*

loss�Zr>�%��       �	��9Yc�A�*

loss[d>˚h       �	�'�9Yc�A�*

loss���=i��%       �	�ø9Yc�A�*

loss�->ù�J       �	�`�9Yc�A�*

lossW�,>�MY0       �	��9Yc�A�*

lossL�P>B�       �	H��9Yc�A�*

lossG�>fz�       �	TU�9Yc�A�*

loss�[>���       �	��9Yc�A�*

lossR�n>p �&       �	3��9Yc�A�*

loss�F>��z       �	�o�9Yc�A�*

loss �>�J�>       �	�9Yc�A�*

loss� ?~��       �	1;9Yc�A�*

loss��`>�F�       �	vo�9Yc�A�*

lossݥ�>m���       �	L�9Yc�A�*

loss�l=>Q�       �	��9Yc�A�*

loss`�q>�G��       �	o�9Yc�A�*

loss}��>Z�]�       �	��9Yc�A�*

loss�o�>_�       �	"7�9Yc�A�*

loss�>mad       �	���9Yc�A�*

loss��>D>V       �	�n�9Yc�A�*

loss�&>��       �	1�9Yc�A�*

lossi�:>�7�*       �	,��9Yc�A�*

loss��:>q_�v       �	�B�9Yc�A�*

loss�{>�ĕ�       �	:��9Yc�A�*

loss�۪>A�(       �	��9Yc�A�*

loss%�>�\E       �	��9Yc�A�*

loss��>�U       �	]��9Yc�A�*

losst\>�1�       �	 b�9Yc�A�*

loss�M`>�-�e       �	���9Yc�A�*

lossE��=7��E       �	��9Yc�A�*

lossfJ>�6�       �	�4�9Yc�A�*

loss�J>�V�$       �	���9Yc�A�*

loss�h >��n[       �	i�9Yc�A�*

loss��>��l       �	�9Yc�A�*

lossw�>�u       �	&��9Yc�A�*

lossi*>���       �	ZK�9Yc�A�*

loss_��=��       �	?��9Yc�A�*

loss"]>�'�       �	]��9Yc�A�*

loss!	r=�ժ�       �	�e�9Yc�A�*

loss,��=hJ�       �	���9Yc�A�*

lossi�!>�}�4       �	\��9Yc�A�*

loss���={���       �	�R�9Yc�A�*

loss���=�/%       �	���9Yc�A�*

loss�A�=t\#       �	Ku�9Yc�A�*

loss�A>�|�       �	_�9Yc�A�*

loss嚃>�:��       �	���9Yc�A�*

lossZ�K>���       �	�Q�9Yc�A�*

loss=y>��d*       �	���9Yc�A�*

losssN�>��M)       �	��9Yc�A�*

loss��>%a�       �	��9Yc�A�*

loss�#B>{\��       �	�W�9Yc�A�*

loss�"=>!�.�       �	���9Yc�A�*

loss��>�`,F       �	��9Yc�A�*

loss"�=J��       �	�#�9Yc�A�*

loss�hX>V
u�       �	ƾ�9Yc�A�*

loss��=��4       �	~T�9Yc�A�*

loss@g>�FK       �	J��9Yc�A�*

lossO�=T��I       �	υ�9Yc�A�*

loss��#>*((x       �	�)�9Yc�A�*

loss!�>�e�f       �	���9Yc�A�*

loss�t�>��3�       �	�`�9Yc�A�*

lossVS
>!��]       �	���9Yc�A�*

lossF0D>y       �	���9Yc�A�*

loss�*�=fE�       �	�W�9Yc�A�*

loss�U�>v���       �	���9Yc�A�*

loss.>)�B       �	9��9Yc�A�*

loss���=|�-�       �	�9�9Yc�A�*

lossȑ�>Ә       �		��9Yc�A�*

loss_Pe>1�Vv       �	��9Yc�A�*

loss��>�8�       �	���9Yc�A�*

loss���>�+�       �	���9Yc�A�*

lossM%[>ӑa       �	�9Yc�A�*

loss��>��       �	���9Yc�A�*

loss�I�>��(       �	[[�9Yc�A�*

loss�(�>�.       �	���9Yc�A�*

lossuT>!
g�       �	�	�9Yc�A�*

loss?i>��{M       �	h��9Yc�A�*

loss\Nt>����       �	^��9Yc�A�*

loss�%`>�⿰       �	j1�9Yc�A�*

lossƕ >
P�       �	8��9Yc�A�*

loss��=�^QR       �	f��9Yc�A�*

loss%P>c�p       �	�"�9Yc�A�*

lossF/�=bD�       �	{��9Yc�A�*

loss&�>U'3       �	Na�9Yc�A�*

lossrq�=�[�"       �	[�9Yc�A�*

loss�>����       �	��9Yc�A�*

lossRJW>h��       �	rQ�9Yc�A�*

loss�?M>�
�       �	��9Yc�A�*

losso��=P�z       �	���9Yc�A�*

loss�4#>���       �	U0�9Yc�A�*

lossȧ�=��KN       �	)��9Yc�A�*

loss� =>M�x=       �	�0�9Yc�A�*

loss���=Eq��       �	G��9Yc�A�*

loss̂�>n�4�       �	O��9Yc�A�*

loss�k>��%�       �	]�9Yc�A�*

loss>���8       �	���9Yc�A�*

loss��>�Y�)       �	���9Yc�A�*

loss\�(>��|�       �	�X�9Yc�A�*

loss�(>a��!       �	z��9Yc�A�*

loss�=0Y�(       �	���9Yc�A�*

loss��">�%K       �	Y��9Yc�A�*

loss��=ApU       �	�-�9Yc�A�*

loss��>{���       �	��9Yc�A�*

loss��>qƍ)       �	ke :Yc�A�*

lossg�>�\g       �	�� :Yc�A�*

loss�3�=g}h�       �	��:Yc�A�*

loss;<=>ɹ21       �	"9:Yc�A�*

loss\>���       �	��:Yc�A�*

loss�=y�[       �	v:Yc�A�*

lossQ�=?l��       �	�:Yc�A�*

loss�qL>|ށr       �	Ҩ:Yc�A�*

loss.�>�6(�       �	�C:Yc�A�*

lossT8�>3�	       �	�:Yc�A�*

loss*Z>,Z��       �	�z:Yc�A�*

lossC�p>g�1       �	�:Yc�A�*

loss<c>�fQ�       �	��:Yc�A�*

loss>n">�y       �	�X:Yc�A�*

loss�	1>���       �	A�:Yc�A�*

loss?��=���       �	�	:Yc�A�*

loss��:>eq�G       �	g,
:Yc�A�*

loss�"�=>G�       �	��
:Yc�A�*

lossR�+>��4       �	�g:Yc�A�*

loss�^>`�[�       �	`:Yc�A�*

loss4x>~""6       �	U�:Yc�A�*

loss]ƞ>��]       �	�<:Yc�A�*

losss@>g�}�       �	A�:Yc�A�*

loss}2>bFB       �	zp:Yc�A�*

loss)�a>fǱ�       �	[:Yc�A�*

loss �r>�y-       �	��:Yc�A�*

loss�L>?9`K       �	�N:Yc�A�*

loss��N>M��       �	K�:Yc�A�*

loss��=1�I       �	��:Yc�A�*

loss��=g�)�       �	�u:Yc�A�*

loss_��=�4��       �	H�:Yc�A�*

lossZ��=ifW)       �	�Y:Yc�A�*

lossZ�C>F�@       �	��:Yc�A�*

loss>n�@$       �	��:Yc�A�*

loss��
>
��       �	�E:Yc�A�*

loss{0>t�       �	��:Yc�A�*

lossX�t>co�       �	er:Yc�A�*

loss�>�C�)       �	�-:Yc�A�*

loss,yP>�L��       �	��:Yc�A�*

loss5g>�*·       �	�b:Yc�A�*

loss�D:>��:       �	(
:Yc�A�*

lossD��>�V�1       �	�:Yc�A�*

loss�)�=XͷM       �	9:Yc�A�*

loss��=wυ       �	��:Yc�A�*

loss�0>.c�       �	�d:Yc�A�*

loss�>���       �	��:Yc�A�*

loss���=�<u�       �	��:Yc�A�*

lossƁD>
Ť�       �	1A:Yc�A�*

loss��>һ֒       �	
�:Yc�A�*

loss�1�=�'�       �	�j:Yc�A�*

loss(q=+&D       �	� :Yc�A�*

loss�WL>���       �	�� :Yc�A�*

lossʀ�>�ђ�       �	�3!:Yc�A�*

lossɇ!>�}�v       �	y�!:Yc�A�*

lossZ^	>qjE�       �	(d":Yc�A�*

loss�Id>=�8�       �	��":Yc�A�*

losss�@>��&�       �	�#:Yc�A�*

loss�ɭ>%/�       �	�7$:Yc�A�*

loss)��=���       �	U�$:Yc�A�*

loss}�>�{p       �	�w%:Yc�A�*

loss%�I>ڲgW       �	j�&:Yc�A�*

loss2vj>��:*       �	��':Yc�A�*

loss�)/=�YB�       �	w.(:Yc�A�*

lossV �=[��       �	��(:Yc�A�*

loss��>�!"       �	oc):Yc�A�*

loss��>��]       �	r�):Yc�A�*

loss�)>��,�       �	q�*:Yc�A�*

lossݷ>��U       �	,+:Yc�A�*

loss��&>k�&�       �	k�+:Yc�A�*

loss|�>���G       �	�,:Yc�A�*

loss!�F>-��+       �	[%-:Yc�A�*

lossa�f>�S�       �	^�-:Yc�A�*

loss��>t:)�       �	�S.:Yc�A�*

loss�->��`       �	G�/:Yc�A�*

loss��=y��B       �	>$0:Yc�A�*

lossO��=Uy�       �	ϼ0:Yc�A�*

loss��O>�� �       �	�`1:Yc�A�*

loss��>P��H       �	��1:Yc�A�*

loss�X�=��W       �	1�2:Yc�A�*

lossXA�=��:       �	33:Yc�A�*

loss`_>�ln       �	��3:Yc�A�*

lossC�z=S)8�       �	>^4:Yc�A�*

lossM>�B       �	%@5:Yc�A�*

loss�g�=��O$       �	��5:Yc�A�*

loss�R=��hu       �	�6:Yc�A�*

loss7:>��*o       �	� 7:Yc�A�*

lossO��>�D       �	0�7:Yc�A�*

lossD�k>"$�       �	�O8:Yc�A�*

loss�+>����       �	�8:Yc�A�*

lossd>A-�       �	=�9:Yc�A�*

loss}F1=���       �	�%::Yc�A�*

loss,�>U���       �	�::Yc�A�*

loss��>�Q�r       �	�U;:Yc�A�*

loss�#>��Z�       �	��;:Yc�A�*

loss�+�=Kg�<       �	��<:Yc�A�*

loss-��>���'       �	�:=:Yc�A�*

loss�G)>��n$       �	_�=:Yc�A�*

loss�ɼ=��3?       �	hy>:Yc�A�*

loss*�a=��        �	G?:Yc�A�*

lossV	w>��       �	V�?:Yc�A�*

loss��`>Xr��       �	uZ@:Yc�A�*

lossd��>֣d/       �	��@:Yc�A�*

loss���>;�j,       �	4�A:Yc�A�*

lossF�N>D��       �	T:B:Yc�A�*

loss��9>fк�       �	��B:Yc�A�*

loss̩5>�j��       �	�{C:Yc�A�*

loss�w�=F�tB       �	%!D:Yc�A�*

loss�&O=�m       �	ԻD:Yc�A�*

loss~�=��W       �	B\E:Yc�A�*

loss
Ɲ>
�6z       �	[F:Yc�A�*

lossW��=�K�       �	�F:Yc�A�*

loss뛖=ȣ�       �	IG:Yc�A�*

loss���=!�)3       �	k�G:Yc�A�*

lossf8�=�Ih*       �	2�H:Yc�A�*

loss�[>���       �	�;I:Yc�A�*

loss��=��V       �	T J:Yc�A�*

loss%cE>qBw       �	סJ:Yc�A�*

loss�*.>"��       �	�CK:Yc�A�*

loss}D>o��^       �	�K:Yc�A�*

loss�vd>��+R       �	�L:Yc�A�*

loss�Z�=����       �	b-M:Yc�A�*

lossݫ=ƐW�       �	��M:Yc�A�*

loss,�I=_�G       �	�kN:Yc�A�*

loss�U!>4�X*       �	�O:Yc�A�*

lossw�(>�q�       �	�O:Yc�A�*

loss�[>��#�       �	�9P:Yc�A�*

loss�0�>�Q_�       �	�P:Yc�A�*

lossH%�=���       �	kQ:Yc�A�*

loss
�>�Wg�       �	�|R:Yc�A�*

loss �>H=�       �	�S:Yc�A�*

loss~|�=��       �	X�S:Yc�A�*

lossf\�=#UF�       �	\T:Yc�A�*

loss�G8>���       �	��T:Yc�A�*

loss��=��z�       �	�U:Yc�A�*

loss$�=;��       �	�BV:Yc�A�*

loss*#>/ƅ�       �	l�V:Yc�A�*

lossClC>�n�       �	��W:Yc�A�*

lossoZ>����       �	�BX:Yc�A�*

loss��=�^]       �	��X:Yc�A�*

lossH�=��3�       �	l�Y:Yc�A�*

lossҐ
>.č~       �	�DZ:Yc�A�*

lossa;n=����       �	b�Z:Yc�A�*

loss�	>}Q�[       �	t�[:Yc�A�*

loss�d=-�,       �	O\:Yc�A�*

loss6��=�s�&       �	� ]:Yc�A�*

loss�=p       �	��]:Yc�A�*

lossH3>��\O       �	�N^:Yc�A�*

lossv��=ʲ�o       �	��^:Yc�A�*

lossM%�=�O��       �	I�_:Yc�A�*

lossSf�=I�       �	c~`:Yc�A�*

loss��>'5��       �	�a:Yc�A�*

loss���=�_�8       �	x�a:Yc�A�*

loss$��>�?�{       �	�Wb:Yc�A�*

lossz�=V�=       �	��b:Yc�A�*

loss
+>��;K       �	��c:Yc�A�*

loss{s>8��,       �	V)d:Yc�A�*

lossz;�= 0�[       �	��d:Yc�A�*

loss�!�=���j       �	Qhe:Yc�A�*

loss�G=�wo�       �	�f:Yc�A�*

loss��>��g>       �	�f:Yc�A�*

lossՔ=��M       �	�tg:Yc�A�*

loss{�N>Kg��       �	<h:Yc�A�*

lossM҅=�y��       �	��i:Yc�A�*

loss��=�� �       �	�Hk:Yc�A�*

loss��=� �       �	��k:Yc�A�*

lossc�>�        �	ʣl:Yc�A�*

lossFڵ<��	       �	�Fm:Yc�A�*

loss�.>5n��       �	��m:Yc�A�*

loss�7q=l�'       �	Υn:Yc�A�*

loss�		>#^       �	Yo:Yc�A�*

loss?/�=;�]�       �	�p:Yc�A�*

loss�=�Aӄ       �	��p:Yc�A�*

loss�k�=���       �	�Oq:Yc�A�*

loss1;>���       �	D�q:Yc�A�*

loss�e]>��       �	v�r:Yc�A�*

lossh5=
��       �	'Os:Yc�A�*

loss.J4=gt��       �	��s:Yc�A�*

loss���=�;y       �	��t:Yc�A�*

lossQ�E=$�       �	�_u:Yc�A�*

lossv �<����       �	�v:Yc�A�*

loss$!�=q#��       �	��v:Yc�A�*

loss��=}q�]       �	�[w:Yc�A�*

lossW��<��ʃ       �	(x:Yc�A�*

loss�@=��=h       �	öx:Yc�A�*

loss�=�p��       �	�cy:Yc�A�*

loss�v\>��%       �	� z:Yc�A�*

loss�pH<{��4       �	0�z:Yc�A�*

losso6<�e`�       �	�B{:Yc�A�*

loss4�;�4�w       �	�{:Yc�A�*

loss�S>�G       �	"�|:Yc�A�*

loss$��>�ߊ�       �	�$}:Yc�A�*

loss($�=� @       �	�}:Yc�A�*

loss�=�bj�       �	GY~:Yc�A�*

loss���=�ŝv       �	��~:Yc�A�*

loss�H�>ivx�       �	Ӆ:Yc�A�*

lossdE�<j�|�       �	K"�:Yc�A�*

loss�9�=,���       �	�ƀ:Yc�A�*

loss,U>+,��       �	�b�:Yc�A�*

lossJ,[>�PQ�       �	���:Yc�A�*

lossQ��=�-��       �	͔�:Yc�A�*

lossrC=)�o       �	1�:Yc�A�*

loss(j>u�3       �	�ʃ:Yc�A�*

loss)5�=��8        �	Rc�:Yc�A�*

lossa`a>x�~       �	���:Yc�A�*

loss:�.>D��       �	Ė�:Yc�A�*

loss��>�9�       �	r1�:Yc�A�*

lossizt>�&F�       �	�Ȇ:Yc�A�*

loss�w>��}       �	�b�:Yc�A�*

lossɦ8>n��p       �	���:Yc�A�*

lossH�~>����       �	���:Yc�A�*

loss��N>ҟy       �	?5�:Yc�A�*

lossi�=���}       �	Vԉ:Yc�A�*

loss�j�=門]       �	Hm�:Yc�A�*

loss�M3>�g       �	N�:Yc�A�*

loss���=�x�       �	��:Yc�A�*

loss���=�rC@       �	�O�:Yc�A�*

lossA(>hBt9       �	��:Yc�A�*

loss|�4>PR��       �	���:Yc�A�*

lossc�<�c       �	��:Yc�A�*

loss)�=��'       �	aĎ:Yc�A�*

loss��:=�:        �	;n�:Yc�A�*

loss�>��       �	F�:Yc�A�*

loss�hI=�d�M       �	U��:Yc�A�*

lossiZ�=u���       �	�:�:Yc�A�*

loss_[5>���       �	Y��:Yc�A�*

loss]��=Ư��       �	М�:Yc�A�*

losse"�=�L��       �	�ړ:Yc�A�*

loss�N">i-��       �	U.�:Yc�A�*

loss�R=�~~       �	0ԕ:Yc�A�*

loss�>�R��       �	��:Yc�A�*

loss��>O�       �	wM�:Yc�A�*

loss\]l=�y*       �	��:Yc�A�*

loss_�>|e��       �	^ڙ:Yc�A�*

lossB�=ȝ��       �	�ߚ:Yc�A�*

loss�{�=�h��       �	&Û:Yc�A�*

loss�E�=��s�       �	0��:Yc�A�*

loss��=(?3       �	x)�:Yc�A�*

loss���=����       �	FΝ:Yc�A�*

lossZX>"u#       �	\u�:Yc�A�*

lossTT#>��       �	��:Yc�A�*

loss�e�=����       �	��:Yc�A�*

loss$�>����       �	�a�:Yc�A�*

loss�/3=6�(�       �	�
�:Yc�A�*

loss^B>ZŢ       �	˾�:Yc�A�*

loss�>�1��       �	�`�:Yc�A�*

lossjO�=��r       �	P�:Yc�A�*

loss� >���       �	�Խ:Yc�A�*

loss���=>��,       �	Ct�:Yc�A�*

lossXx>���       �	��:Yc�A�*

loss�O>D(=�       �	|��:Yc�A�*

lossR�
>"��       �	�U�:Yc�A�*

loss��=6 �p       �	f��:Yc�A�*

loss�<>�cS�       �	`;�:Yc�A�*

loss`B>�O       �	���:Yc�A�*

loss6R�=��EE       �	N{�:Yc�A�*

lossHA7>���	       �	�!�:Yc�A�*

loss��7=��       �	f��:Yc�A�*

loss�E=�h�       �	�X�:Yc�A�*

lossH�>'5{       �	��:Yc�A�*

loss��>�Z7�       �	��:Yc�A�*

loss���=FnI?       �	JC�:Yc�A�*

loss�I0>=�E�       �	���:Yc�A�*

loss�]=�!�t       �	���:Yc�A�*

loss��=�{3%       �	K �:Yc�A�*

loss��=���       �	���:Yc�A�*

loss��>����       �	�]�:Yc�A�*

lossc�=�$:�       �	��:Yc�A�*

lossi�>|���       �	6��:Yc�A�*

loss��(>?��C       �	�+�:Yc�A�*

loss<�>�h/<       �	���:Yc�A�*

loss���=� n�       �	\�:Yc�A�*

loss@ڽ=o.g�       �	;��:Yc�A�*

loss�?>�>��       �	ݗ�:Yc�A�*

loss��=��        �	&5�:Yc�A�*

loss��='�]�       �	���:Yc�A�*

loss�,N>q�e       �	�h�:Yc�A�*

loss8P�=i>�       �	)�:Yc�A�*

loss��>�e�       �	��:Yc�A�*

loss�=����       �	���:Yc�A�*

loss���>zj�       �	'��:Yc�A�*

loss�=���<       �	̚�:Yc�A�*

lossx8>2�r�       �	Y4�:Yc�A�*

lossű=Hv�       �	���:Yc�A�*

loss��>���       �	��:Yc�A�*

loss&-�>k���       �	R��:Yc�A�*

loss_>�>&�       �	�|�:Yc�A�*

loss_#>���        �	o��:Yc�A�*

loss$��= �}�       �	t_�:Yc�A�*

loss���=��       �	���:Yc�A�*

loss��>�&�D       �	A��:Yc�A�*

loss�>>%%>;       �	!��:Yc�A�*

lossڕa>a>��       �	f��:Yc�A�*

loss8=�=�cH#       �	��:Yc�A�*

loss���=|���       �	o�:Yc�A�*

lossj�=��)Z       �	i �:Yc�A�*

loss�T=a/Ǝ       �	-��:Yc�A�*

loss��c=�A�j       �	�f�:Yc�A�*

loss&��=}���       �	;�:Yc�A�*

loss݀>a3n,       �	���:Yc�A�*

loss�?ױ'�       �	6v�:Yc�A�*

lossk��=��b�       �	�'�:Yc�A�*

loss�7�=���       �	A��:Yc�A�*

loss�e
=�P��       �	���:Yc�A�*

loss��=���       �	E.�:Yc�A�*

loss\:>%(l�       �	���:Yc�A�*

lossD��=)VK�       �	N��:Yc�A�*

loss��=���       �	�h�:Yc�A�*

loss� �=���       �	��:Yc�A�*

loss�=�B>�       �	���:Yc�A�*

loss/��=A�_�       �	҉�:Yc�A�*

loss$jr=<P	V       �	D�:Yc�A�*

lossσ=��"       �	���:Yc�A�*

loss9
>v��
       �	��:Yc�A�*

loss7]L>��w       �	�Z�:Yc�A�*

loss>���       �	�:Yc�A�*

lossww�=����       �	j��:Yc�A�*

loss_Ũ=�Du       �	�o�:Yc�A�*

loss��=��'       �	�"�:Yc�A�*

loss�{�=yi�R       �	$��:Yc�A�*

loss@�I>��t�       �	��:Yc�A�*

loss���=���Y       �	M2�:Yc�A�*

loss7>?KA�       �	���:Yc�A�*

loss!�^>L��       �	ޑ�:Yc�A�*

loss�eU>�IZ�       �	�7�:Yc�A�*

loss��[=,_^       �	��:Yc�A�*

loss��|=��0       �	�~�:Yc�A�*

lossR(�=F��       �	2�:Yc�A�*

loss܏>�m��       �	��:Yc�A�*

loss�Ml>8Ȕ�       �	5^�:Yc�A�*

lossNk�=��1       �	��:Yc�A�*

losso%>��%�       �	���:Yc�A�*

loss}>��x       �	+�:Yc�A�*

lossZ+�=��x       �	���:Yc�A�*

loss�8�=�;��       �	�l�:Yc�A�*

loss�,�=#)�       �	��:Yc�A�*

lossa��=�^�G       �	ș�:Yc�A�*

loss:�>1��       �	�4�:Yc�A�*

lossߒ>���'       �	I��:Yc�A�*

lossA�=~Y.       �	�t�:Yc�A�*

loss�/>J�.%       �	/ ;Yc�A�*

loss��>�1�u       �	�� ;Yc�A�*

loss*��=��N       �	|E;Yc�A�*

loss���=��N�       �	/�;Yc�A�*

lossr�=�&�a       �	1};Yc�A�*

loss�|>��g?       �	5);Yc�A�*

lossX>��       �	�;Yc�A�*

loss�v=
�C�       �	uX;Yc�A�*

loss��=\l�!       �	��;Yc�A�*

lossx;>W�P�       �	>�;Yc�A�*

loss���=��7,       �	�2;Yc�A�*

loss�/!>�hf       �	��;Yc�A�*

loss�'>��;       �	�c;Yc�A�*

loss!�=�.�H       �	h;Yc�A�*

loss���<8ۏ�       �	Ԙ;Yc�A�*

loss��@>/u��       �	�3	;Yc�A�*

lossd��=\���       �	��	;Yc�A�*

loss�Mi>䳽)       �	e
;Yc�A�*

loss�%�=6 �       �	&�
;Yc�A�*

loss�&=����       �	 �;Yc�A�*

loss��.=ɵ1       �	y;;Yc�A�*

loss@�=5��       �	��;Yc�A�*

loss4�=�>D�       �	Ww;Yc�A�*

loss�d=K�8�       �	�;Yc�A�*

loss�=���k       �	 �;Yc�A�*

loss"�>��       �	6�;Yc�A�*

loss�au=�$��       �	V-;Yc�A�*

loss&v�=+�;	       �	.�;Yc�A�*

lossW��=��       �	K ;Yc�A�*

loss]	*>��=Z       �	Y�;Yc�A�*

loss͟�=Qp       �	�C;Yc�A�*

loss�=��~       �	,b;Yc�A�*

lossT��=j.       �	�;Yc�A�*

loss^8>���       �	��;Yc�A�*

loss!d=�F��       �	gd;Yc�A�*

lossj5%>��D       �	�;Yc�A�*

loss��N>��p       �	��;Yc�A�*

loss��=#�B>       �	�H;Yc�A�*

loss�-�=����       �	�;Yc�A�*

loss���=�<��       �	q�;Yc�A�*

loss��m=Vif�       �	Bv;Yc�A�*

loss��>t&       �	H;Yc�A�*

loss�!�=8��x       �	��;Yc�A�*

loss�=\�       �	Ed;Yc�A�*

loss�"=��>u       �	Q;Yc�A�*

loss�=���_       �	7�;Yc�A�*

loss�f=��Or       �	3k;Yc�A�*

loss�&�<��       �	w;Yc�A�*

lossZ� >]k�       �	U�;Yc�A�*

loss{(>?r�       �	�X ;Yc�A�*

loss��=����       �	� ;Yc�A�*

loss�k>���       �	¡!;Yc�A�*

loss�1�=r���       �	V";Yc�A�*

losst9>Mr�       �	�#;Yc�A�*

loss#��<��?       �	I�#;Yc�A�*

loss�7=2��h       �	&p$;Yc�A�*

loss=��;�       �	�%;Yc�A�*

loss`'=[��       �	�&;Yc�A�*

loss�N3>�Hx�       �	�';Yc�A�*

loss�K�=�i�\       �	�';Yc�A�*

lossz1�=͟a0       �	�Z(;Yc�A�*

loss�>��+E       �	�,);Yc�A�*

loss$˨<7�       �	�G*;Yc�A�*

lossW��<0�       �	'�*;Yc�A�*

loss(V>*���       �	�+;Yc�A�*

loss M>>�(�u       �	 X,;Yc�A�*

loss;�<=�τ       �	-;Yc�A�*

loss)�	>��R�       �	X�-;Yc�A�*

loss�| >F�̡       �	iQ.;Yc�A�*

loss�i�=Ձ8+       �	3�.;Yc�A�*

loss_G�=� (       �	��/;Yc�A�*

loss�� >+��{       �	aT0;Yc�A�*

loss�^y=��L�       �	4�0;Yc�A�*

loss�O�= �O�       �	�1;Yc�A�*

loss�J�=D?h�       �	�W2;Yc�A�*

lossԲ�=�1P<       �	y3;Yc�A�*

lossi��=��o�       �	��3;Yc�A�*

loss	j=���v       �	9c4;Yc�A�*

loss,`@=�q)�       �	�5;Yc�A�*

loss��=���       �	u�5;Yc�A�*

loss�>� �       �	�Y6;Yc�A�*

loss��E=&�^{       �	��6;Yc�A�*

loss��=-��`       �	��7;Yc�A�*

loss��m>��>�       �	�G8;Yc�A�*

loss"�=}wf�       �	�8;Yc�A�*

loss(EH=E�[       �	G:;Yc�A�*

lossQ��=HX��       �	�:;Yc�A�*

lossR��<#�J�       �	�J;;Yc�A�*

loss��>��$       �	; <;Yc�A�*

loss{/�=~�A       �	 =;Yc�A�*

lossvD�<	u#H       �	'�=;Yc�A�*

loss:�E=���       �	d>;Yc�A�*

loss� >�n�(       �	%?;Yc�A�*

loss�J�=m,�       �	ף?;Yc�A�*

losswv�=�wk       �	�N@;Yc�A�*

loss�k�=g�       �	�<A;Yc�A�*

loss@��=�$       �	=�A;Yc�A�*

lossn�u=X��c       �	i�B;Yc�A�*

lossc��=��       �	d;C;Yc�A�*

loss��O=Da��       �	�C;Yc�A�*

loss���=�g=       �	�D;Yc�A�*

loss���<_zI       �	�$E;Yc�A�*

lossϋ�=�h�       �	S�E;Yc�A�*

losseZ�={D7�       �	@lF;Yc�A�*

loss�Բ=��0       �	�IG;Yc�A�*

loss47D>�r��       �	��G;Yc�A�*

loss*�=�ԡ.       �	@�H;Yc�A�*

losss��="}�       �	�#I;Yc�A�*

loss��h=�Xf�       �	��I;Yc�A�*

loss��b=P��c       �	�[J;Yc�A�*

loss��#>'D�       �	A�J;Yc�A�*

loss�.i=E�k:       �	Y�K;Yc�A�*

loss�u~=�.�       �	 BL;Yc�A�*

loss/��=��       �	j�L;Yc�A�*

loss]}�=��]u       �	,~M;Yc�A�*

loss�� >�^f       �	�N;Yc�A�*

loss�#N>�7�       �	��N;Yc�A�*

loss�u�=��O�       �	OO;Yc�A�*

loss&y=r>��       �	i�O;Yc�A�*

lossZL�<F�$�       �	I�P;Yc�A�*

loss��=�3�       �	�oQ;Yc�A�*

loss<�>g�!       �	�NR;Yc�A�*

loss�9�=��f/       �	S;Yc�A�*

lossE�=�
�W       �	i�S;Yc�A�*

loss34b>��       �	Y�T;Yc�A�*

loss�5�>~8~       �	�{U;Yc�A�*

loss��">l�	       �	�V;Yc�A�*

losss=�ci`       �	j�V;Yc�A�*

loss��=�       �	tzW;Yc�A�*

loss�WC>�D�O       �	�X;Yc�A�*

loss�RF>�b<�       �	�Y;Yc�A�*

loss�q=��U"       �	$
Z;Yc�A�*

loss2*�=���4       �	ϢZ;Yc�A�*

lossr�=:v�       �	�<[;Yc�A�*

loss�N>�Ӟz       �	��[;Yc�A�*

loss8c�=9�@       �	ap\;Yc�A�*

loss��>��       �	5];Yc�A�*

loss�_�=l;�       �	��];Yc�A�*

lossfk�<�|�       �	�>^;Yc�A�*

loss�x(>ϟu�       �	��^;Yc�A�*

loss=�>9���       �	8�_;Yc�A�*

lossh"><�B�       �	�5`;Yc�A�*

loss1f�=(�       �	��`;Yc�A�*

loss�d�=M�       �	�qa;Yc�A�*

loss���=����       �	�b;Yc�A�*

loss�S�=�X��       �	:�b;Yc�A�*

loss��?=�Q��       �	�Oc;Yc�A�*

lossH�i=�/2       �	��c;Yc�A�*

loss��=����       �	��d;Yc�A�*

loss��=&�!X       �	zoe;Yc�A�*

loss�>���       �	X f;Yc�A�*

lossc��=\%�_       �	>�f;Yc�A�*

lossr&�=�9�       �	Ƨg;Yc�A�*

loss��=�d�8       �	%Yh;Yc�A�*

loss_"N=Ӟ�u       �	��h;Yc�A�*

loss�Q�=�`��       �	�i;Yc�A�*

loss��=·�       �	�j;Yc�A�*

loss���=���       �	ӽj;Yc�A�*

loss@͞=��       �	��k;Yc�A�*

loss\��=Ti��       �	!@l;Yc�A�*

loss��=ky�       �	L�l;Yc�A�*

loss�E=��[       �	׊m;Yc�A�*

loss�`k=���        �	r4n;Yc�A�*

loss'��=�m&h       �	��n;Yc�A�*

loss;�@>@Ȇ�       �	<fo;Yc�A�*

loss<,\=�l�v       �	�p;Yc�A�*

loss���=�`6�       �	�p;Yc�A�*

lossq�=��p�       �	�Pq;Yc�A�*

loss�8�=����       �	��q;Yc�A�*

loss��>�4�       �	�~r;Yc�A�*

loss㻰=�)�c       �	Ys;Yc�A�*

loss��X>z�       �	��s;Yc�A�*

loss��,=*�i�       �	�?t;Yc�A�*

losslC�=f�       �	��t;Yc�A�*

lossH�=��T       �	mu;Yc�A�*

lossz�>b�J�       �	m�u;Yc�A�*

loss!�=E�       �	R�v;Yc�A�*

loss�R�=Zڣ�       �	�3w;Yc�A�*

loss|�=˩��       �	2 x;Yc�A�*

loss.9n=l�,O       �	S�x;Yc�A�*

loss���=><,�       �	�[y;Yc�A�*

lossf�=\�       �	o�y;Yc�A�*

lossF�?>�+��       �	��z;Yc�A�*

loss�7G=yqR       �	�{;Yc�A�*

loss��=�R�k       �	 �{;Yc�A�*

loss��=��D       �	V|;Yc�A�*

loss�X�=r﨡       �	1�|;Yc�A�*

loss�V�<�J-�       �	
�};Yc�A�*

loss���=^b�Y       �	W$~;Yc�A�*

loss�O�=���m       �	f�~;Yc�A�*

loss���<j�U       �	�X;Yc�A�*

lossF�>"�W�       �	_�;Yc�A�*

loss���='$�R       �	��;Yc�A�*

loss�4�=���       �	X;�;Yc�A�*

loss�U�=���       �	Ձ;Yc�A�*

loss��=�ĳ/       �		l�;Yc�A�*

lossd�<�0$�       �	��;Yc�A�*

loss�8�<gGe       �	|��;Yc�A�*

loss'1>a�       �	&6�;Yc�A�*

loss���<���n       �	�΄;Yc�A�*

lossW{�=4Q{%       �	(d�;Yc�A�*

loss֤z=��?v       �	���;Yc�A�*

lossL:>+�U       �	;Yc�A�*

loss
>?�Hq       �	 R�;Yc�A�*

loss#�k>-�       �	��;Yc�A�*

losstؾ=��ݰ       �	���;Yc�A�*

loss��	>�r�$       �	��;Yc�A�*

loss���=R�>�       �	���;Yc�A�*

loss�$�=�k��       �	iU�;Yc�A�*

loss�,q=�1Pg       �	��;Yc�A�*

lossЃ�=W8H�       �	���;Yc�A�*

loss`�,=�Φ       �	�'�;Yc�A�*

loss&9�=L�$g       �	��;Yc�A�*

loss�V�<W��)       �	�V�;Yc�A�*

loss
��=Mc��       �	��;Yc�A�*

lossM�<�=��       �	F|�;Yc�A�*

loss�7�=2@W       �	��;Yc�A�*

loss=!���       �	>��;Yc�A�*

losss��=�k�u       �	�G�;Yc�A�*

loss���=�7�       �	ސ;Yc�A�*

loss�x3=.�%�       �	�;Yc�A�*

loss/�f=�¢       �	r�;Yc�A�*

loss�P�=��=�       �	���;Yc�A�*

loss���=�r��       �	�Y�;Yc�A�*

loss3=�p�       �	� �;Yc�A�*

loss3�=㩽�       �	��;Yc�A�*

loss.>gb>.       �	.t�;Yc�A�*

loss�+=��t�       �	��;Yc�A�*

loss��>�>]       �	k��;Yc�A�*

loss��=���}       �	�S�;Yc�A�*

loss�8=%�<       �	:�;Yc�A�*

lossI�>ŝ��       �	؂�;Yc�A�*

loss]I>P��       �	%�;Yc�A�*

loss�y�=��       �	���;Yc�A�*

loss\V�=Op�       �	�]�;Yc�A�*

loss�B9>T梠       �	*R�;Yc�A�*

loss�[�=	%F1       �	�;Yc�A�*

loss|q�=A�R       �	~�;Yc�A�*

lossZo=<S54       �	�6�;Yc�A�*

loss��=��       �	�̝;Yc�A�*

loss��O=(�.�       �	�b�;Yc�A�*

loss��:>L�GR       �	���;Yc�A�*

loss���=$<��       �	A��;Yc�A�*

loss�$s=k��       �	�L�;Yc�A�*

loss�/�=��8       �	��;Yc�A�*

loss��=�V��       �	���;Yc�A�*

loss��7=�}��       �	K�;Yc�A�*

loss�Y	=�]��       �	�ߢ;Yc�A�*

loss��=�;SU       �	�w�;Yc�A�*

loss�n@>g�       �	r�;Yc�A�*

loss֢�=�	�       �	���;Yc�A�*

loss��H>��ʿ       �	L�;Yc�A�*

loss�g>��^�       �	g�;Yc�A�*

loss#��=5��?       �	ʋ�;Yc�A�*

lossn�>k%       �	�c�;Yc�A�*

loss84�=��        �	;��;Yc�A�*

loss<�=9�:;       �	;Yc�A�*

loss��=�_06       �	�2�;Yc�A�*

lossT�=Td�r       �	�Ω;Yc�A�*

loss�W= d��       �	���;Yc�A�*

loss��>6M/       �	�7�;Yc�A�*

loss\>s�:       �	:Ϋ;Yc�A�*

loss�	=�5��       �	�g�;Yc�A�*

loss*3z=���       �	��;Yc�A�*

loss�u&=�R�_       �	-\�;Yc�A�*

loss�w�=�:Y�       �	A�;Yc�A�*

loss��>�H�k       �	7��;Yc�A�*

loss��=i��       �	G9�;Yc�A�*

loss��=S4       �	oհ;Yc�A�*

lossT{�=I���       �	]��;Yc�A�*

loss��>��|�       �	 C�;Yc�A�*

lossÔ�=ƛ"�       �	D�;Yc�A�*

lossةY>�S��       �	�޳;Yc�A�*

loss���=ɔEo       �	Sv�;Yc�A�*

lossW=��Pw       �	��;Yc�A�*

loss1�=�Ѝ�       �	���;Yc�A�*

loss��D=P�y�       �	hB�;Yc�A�*

loss+��=]F*b       �	��;Yc�A�*

losst�>��U       �	�|�;Yc�A�*

loss�(>U�W�       �	��;Yc�A�*

loss���<;��'       �	u��;Yc�A�*

loss�̇=sj       �	Nd�;Yc�A�*

loss���=Ӈ��       �	:�;Yc�A�*

lossiǠ=/!^       �	$��;Yc�A�*

loss��=��A       �	82�;Yc�A�*

loss�<�=���       �	�Ȼ;Yc�A�*

loss��>D��+       �	�g�;Yc�A�*

loss6>�v�
       �	d:�;Yc�A�*

loss�>|��1       �	bֽ;Yc�A�*

loss�) =�Q��       �	�l�;Yc�A�*

lossn|7=̆�       �		�;Yc�A�*

loss�N�<Ȉc$       �	<��;Yc�A�*

loss��c=�ZH       �	h<�;Yc�A�*

loss�Y�=���Z       �	��;Yc�A�*

loss4��=q���       �	��;Yc�A�*

loss���=�ī�       �	�A�;Yc�A�*

lossn�1=K8_
       �	���;Yc�A�*

lossE��=���       �	�k�;Yc�A�*

loss٨	>��       �	S�;Yc�A�*

loss��=ȳ϶       �	ٲ�;Yc�A�*

lossVK>�<�       �	[�;Yc�A�*

loss��=vU��       �	��;Yc�A�*

lossګ)>�'P       �	9��;Yc�A�*

loss��J=��b�       �	�q�;Yc�A�*

loss
L2=�il�       �	�;Yc�A�*

loss��B=�M�0       �	d��;Yc�A�*

lossv}�=Ȭ=�       �	
I�;Yc�A�*

loss=�h=ɌS�       �	���;Yc�A�*

loss�Fn=�̖       �	���;Yc�A�*

loss��=� �       �	=,�;Yc�A�*

lossM,�<@9~�       �	���;Yc�A�*

loss�b�=m��       �	�o�;Yc�A�*

loss�=�6       �	��;Yc�A�*

lossL��=rk�u       �	9��;Yc�A�*

loss`��=���^       �	�W�;Yc�A�*

loss-M�=_m{6       �	x'�;Yc�A�*

loss��=w�\�       �	/��;Yc�A�*

lossO{�=���Y       �	�Y�;Yc�A�*

lossm�>~Lh�       �	���;Yc�A�*

lossе�=@��<       �	��;Yc�A�*

lossz�>�2W�       �	kJ�;Yc�A�*

loss��6>��       �	\��;Yc�A�*

loss`=>"�l�       �	���;Yc�A�*

loss�H="�Y�       �	�/�;Yc�A�*

loss�+7=ИK+       �	�6�;Yc�A�*

loss�3�=���       �	���;Yc�A�*

loss<��=���       �	�/�;Yc�A�*

loss�R=� �       �	��;Yc�A�*

loss���=���       �	���;Yc�A�*

loss�S=sq       �	��;Yc�A�*

loss��M>.       �	�x�;Yc�A�*

loss@�=�m��       �	��;Yc�A�*

loss�j$>�}VQ       �	9D�;Yc�A�*

lossZ��=N��       �	���;Yc�A�*

loss@��=,�Q�       �	���;Yc�A�*

loss#��=|�ʘ       �	I��;Yc�A�*

loss�J�=��'       �	���;Yc�A�*

loss4"�=���       �	�'�;Yc�A�*

loss���=c�<       �	1�;Yc�A�*

loss��T=ߝ�       �	�;Yc�A�*

loss��=��|       �	}��;Yc�A�*

loss���=�c�       �	���;Yc�A�*

lossh�<R�ī       �	�:�;Yc�A�*

loss}�v=S�Jf       �	��;Yc�A�*

loss��&=����       �	
��;Yc�A�*

loss*�/=�6       �	n1�;Yc�A�*

loss�R�=��       �	)"�;Yc�A�*

lossm�F>qώ#       �	���;Yc�A�*

losso�=�6��       �	*��;Yc�A�*

loss�Ǟ=����       �	g�;Yc�A�*

loss�V�=ǐ�q       �	��;Yc�A�*

loss�=yz��       �	���;Yc�A�*

loss�<�=���,       �	�;�;Yc�A�*

loss!�|>�g�L       �	���;Yc�A�*

loss���=	�a       �	�l�;Yc�A�*

lossl6�=��       �	2�;Yc�A�*

loss��>��2i       �	:��;Yc�A�*

lossso�=J��       �	�h�;Yc�A�*

loss���<ݒ�       �	���;Yc�A�*

loss�=��|       �	��;Yc�A�*

loss���=nQm�       �	�@�;Yc�A�*

loss�h>u�E�       �	���;Yc�A�*

loss3>��5       �	r��;Yc�A�*

loss�>��+       �	�?�;Yc�A�*

loss��`=�D�
       �	���;Yc�A�*

lossM\�=B��       �	��;Yc�A�*

loss���=���       �	�0�;Yc�A�*

lossT�O=�tDC       �	���;Yc�A�*

loss�C=�4�       �	?t�;Yc�A�*

loss%��<�N�       �	N�;Yc�A�*

loss�W�=~��       �	i��;Yc�A�*

loss�ǝ=d �       �	�F�;Yc�A�*

loss�s=D�v       �	C��;Yc�A�*

loss7.&>y�N       �	ٔ�;Yc�A�*

lossT�_=t�       �	3�;Yc�A�*

loss�n�=�KL�       �	���;Yc�A�*

loss�Yp<�=6n       �		n�;Yc�A�*

loss��+>D.2�       �	��;Yc�A�*

loss�?�=,8�       �	���;Yc�A�*

loss��=\�       �	�p�;Yc�A�*

loss��U>����       �	g
�;Yc�A�*

loss��<��a�       �	8��;Yc�A�*

loss�N<=��C�       �	�>�;Yc�A�*

loss��c=���       �	@��;Yc�A�*

lossc~p=���       �	�y <Yc�A�*

loss��e=D�@       �	�D<Yc�A�*

loss��=p�?h       �	��<Yc�A�*

loss�>�S�       �	�<Yc�A�*

loss��=�E%�       �	�<Yc�A�*

lossl�O=^RD       �	�<Yc�A�*

loss	��=a�*}       �	�i<Yc�A�*

lossq��=^���       �	�<Yc�A�*

loss��e=y�7�       �	,�<Yc�A�*

loss�F==+��       �	LU<Yc�A�*

loss���=���       �	0<Yc�A�*

loss-=+��       �	��<Yc�A�*

loss{[�=���       �	�	<Yc�A�*

loss�ʈ=]��Y       �	�2
<Yc�A�*

loss��0>\Uc       �	�
<Yc�A�*

loss�E�<HJT       �	Â<Yc�A�*

loss8��<�qhw       �	g&<Yc�A�*

lossv��=eX�       �	|�<Yc�A�*

loss��<Z���       �	Q�<Yc�A�*

loss�)�=6X�o       �	g'<Yc�A�*

lossC�=Fs�_       �	��<Yc�A�*

lossZ�A=iM�y       �	k<Yc�A�*

lossݨ�=���r       �	�<Yc�A�*

loss��=䝄b       �	�<Yc�A�*

loss%Y =R�       �	��<Yc�A�*

lossw�=Ŕ�~       �	�B<Yc�A�*

loss��_=oh3�       �	��<Yc�A�*

lossd��=����       �	��<Yc�A�*

loss�#i=C/7�       �	xA<Yc�A�*

loss��>��2�       �	DQ<Yc�A�*

loss=��=�>��       �	�.<Yc�A�*

loss�&><�       �	*�<Yc�A�*

lossɬ�=@�k�       �	��<Yc�A�*

loss�(L=���       �	>�<Yc�A�*

loss�T�=mY�       �	4e<Yc�A�*

lossw��=n�5L       �	�<Yc�A�*

loss�@=�-��       �	��<Yc�A�*

loss���=���       �	�R<Yc�A�*

loss���=��       �	��<Yc�A�*

loss��9=U�*�       �	'�<Yc�A�*

loss%1�="���       �	v7<Yc�A�*

loss�Չ=k�\�       �	s�<Yc�A�*

loss໱=ߞȀ       �	��<Yc�A�*

loss
�<g�S       �	\<Yc�A�*

loss���=ow�       �	��<Yc�A�*

loss�� <v�3       �	,a <Yc�A�*

loss�a�=~}5�       �	P� <Yc�A�*

loss�5=)���       �	N�!<Yc�A�*

losso�o=ahξ       �	�4"<Yc�A�*

loss3ֱ=X��       �	$�"<Yc�A�*

loss���=2�u�       �	bh#<Yc�A�*

loss/>[�uJ       �	X$<Yc�A�*

loss�K�<�;�       �	��$<Yc�A�*

loss�o�=?��       �	�=%<Yc�A�*

loss�A=nH4       �	��%<Yc�A�*

loss�Q]=zo       �	�&<Yc�A�*

loss���;6��#       �	
+'<Yc�A�*

lossz%l<�-�'       �	��'<Yc�A�*

loss,^�<�r�7       �	�p(<Yc�A�*

loss�	�<��       �	�)<Yc�A�*

lossu_=�9        �	�*<Yc�A�*

loss��;<�(�       �	p�*<Yc�A�*

loss�D=��I       �	Zf+<Yc�A�*

lossx<]��       �	h,<Yc�A�*

loss���;��B�       �	�,<Yc�A�*

loss��;u7Q�       �	S?-<Yc�A�*

lossM$=f��       �	e.<Yc�A�*

loss���=�ܥ5       �	��.<Yc�A�*

lossx��=:N�       �	60<Yc�A�*

loss=��;�S�       �	��0<Yc�A�*

losss:�=B's�       �	�n1<Yc�A�*

lossX��>�=U"       �	�2<Yc�A�*

loss���<� <       �	0�2<Yc�A�*

loss�)W=1X�s       �	�:3<Yc�A�*

lossS�=�H,       �	��3<Yc�A�	*

loss�� >3M�       �	Ym4<Yc�A�	*

losswB�=��+\       �	�5<Yc�A�	*

losscb=��d       �	g�5<Yc�A�	*

loss�>,|��       �	v�7<Yc�A�	*

loss��>��Z�       �	��8<Yc�A�	*

loss#��=����       �	J{9<Yc�A�	*

loss-�=�u��       �	�:<Yc�A�	*

loss燌=����       �	ĳ:<Yc�A�	*

loss��Y=l
�U       �	�G;<Yc�A�	*

loss�K>���)       �	��;<Yc�A�	*

loss�=˭3)       �	��<<Yc�A�	*

lossA?>�kM\       �	 =<Yc�A�	*

loss}{f>AgA�       �	��=<Yc�A�	*

loss�n=�t�       �	�T><Yc�A�	*

lossf�`=�TE�       �	��><Yc�A�	*

loss���=?��       �	�?<Yc�A�	*

loss�qt=���       �	-"@<Yc�A�	*

lossR?�<b��        �	A�@<Yc�A�	*

loss���=��p       �	.TA<Yc�A�	*

loss��=�~c�       �	��A<Yc�A�	*

loss�B�<;V�       �	��B<Yc�A�	*

lossMi�=~���       �	OC<Yc�A�	*

loss�T�<�Z�       �	��C<Yc�A�	*

loss���=�t��       �	ԀD<Yc�A�	*

loss@j=�D�       �	�#E<Yc�A�	*

loss{�H>�##       �	4�E<Yc�A�	*

lossB�>���       �	zRF<Yc�A�	*

loss�aL=n�Ky       �	��F<Yc�A�	*

loss�I�=�l�       �	�zG<Yc�A�	*

lossQ�=��jX       �	LH<Yc�A�	*

loss4=`G]�       �	��H<Yc�A�	*

lossܞZ=����       �	7TI<Yc�A�	*

loss�1=�j+.       �	��I<Yc�A�	*

loss� L=��(       �	ИJ<Yc�A�	*

loss�j=��       �	�0K<Yc�A�	*

loss}D�=��ͻ       �	*�K<Yc�A�	*

loss�?�=����       �	�qL<Yc�A�	*

loss(A�<hd"       �	�M<Yc�A�	*

lossY	=�M�/       �	�M<Yc�A�	*

loss�=�Q�       �	fN<Yc�A�	*

loss W�=	       �	�!O<Yc�A�	*

loss4Dm=@���       �	'�O<Yc�A�	*

lossdD�=�I��       �	,eP<Yc�A�	*

loss
�=aL1�       �	;�P<Yc�A�	*

loss�6=�f.
       �	��Q<Yc�A�	*

loss���=��ܑ       �	��R<Yc�A�	*

loss7{=1/�       �	R,S<Yc�A�	*

loss�ń=���       �	��S<Yc�A�	*

losss��=*�$       �	k�n<Yc�A�	*

loss��=�G�       �	P�o<Yc�A�	*

loss�2�=l�c_       �	� p<Yc�A�	*

loss��>�m�(       �	=�p<Yc�A�	*

loss���=�D}       �	jMq<Yc�A�	*

loss��=2��t       �	��q<Yc�A�	*

loss�>�V��       �	�vr<Yc�A�	*

loss
��=3�-�       �	�Bs<Yc�A�	*

loss��=����       �	a�s<Yc�A�	*

loss�Ŗ=�e��       �	yt<Yc�A�	*

lossf�<W�       �	�u<Yc�A�	*

loss�<��       �	u�u<Yc�A�	*

lossi"�=�Q ,       �	�w<Yc�A�	*

loss7�T=�S�l       �	{�w<Yc�A�	*

loss��<�*�       �	x`x<Yc�A�	*

lossq >�,��       �	�y<Yc�A�	*

loss#��<�_�C       �	t�y<Yc�A�	*

lossm�P=�㝳       �	Bz<Yc�A�	*

loss�y?=M_ZA       �	2�z<Yc�A�	*

loss��$>
:�       �	R{<Yc�A�	*

lossl��=�`�!       �	�D|<Yc�A�	*

loss���=d
�u       �	S�|<Yc�A�	*

loss�<�<̭�       �	'�}<Yc�A�	*

loss��P>%RrB       �	�~<Yc�A�	*

loss�ܘ=�/R�       �	��~<Yc�A�	*

loss$�#=���       �	
h<Yc�A�	*

loss��N=�W�       �	��<Yc�A�	*

loss���=|�       �	5��<Yc�A�	*

lossM�[=(a69       �	Ab�<Yc�A�	*

loss�r�=km��       �	��<Yc�A�	*

lossR=�:?C       �	K��<Yc�A�	*

loss�ɴ<%���       �	�B�<Yc�A�	*

loss:�=����       �	O�<Yc�A�	*

loss�(H> ��J       �	ߤ�<Yc�A�	*

loss4�7=	�n�       �	�@�<Yc�A�	*

loss��=w0��       �	z�<Yc�A�	*

loss��\<�	M       �	�y�<Yc�A�	*

loss6�=h+�       �	��<Yc�A�	*

loss6V>�T�       �	���<Yc�A�	*

loss1�$>���       �	N�<Yc�A�	*

lossw&�="�-=       �	��<Yc�A�	*

loss{=�أ1       �	(��<Yc�A�	*

loss��<���^       �	�`�<Yc�A�	*

loss�!�=�?W5       �	���<Yc�A�	*

loss	��=���r       �	���<Yc�A�	*

loss��9>���       �	bI�<Yc�A�	*

loss��=�5�       �	$�<Yc�A�	*

loss��W=�� �       �	㌍<Yc�A�	*

loss^�=�?8       �		7�<Yc�A�	*

loss}��;���       �	��<Yc�A�	*

loss��<АC�       �	���<Yc�A�	*

lossj�=W}�(       �	l!�<Yc�A�	*

loss��s=�͒�       �	���<Yc�A�	*

loss�fz>�
�k       �	�i�<Yc�A�	*

loss��=#�2�       �	1	�<Yc�A�	*

loss�U�<����       �	ګ�<Yc�A�	*

loss��<��a       �	@N�<Yc�A�	*

loss�=>���       �	��<Yc�A�	*

loss2��=�x��       �	���<Yc�A�	*

loss��}=!�`�       �	��<Yc�A�	*

lossh[=�       �	�<Yc�A�	*

lossS7�=��|       �	{��<Yc�A�	*

loss	N!=�       �	h#�<Yc�A�	*

loss/'�=1���       �	s��<Yc�A�	*

loss�=9�c       �	V�<Yc�A�	*

loss�=���n       �	�<Yc�A�	*

loss-��=�A�       �	㉙<Yc�A�	*

loss�y�=)��       �	[#�<Yc�A�	*

loss�[>$�}�       �	��<Yc�A�	*

loss6��=�mS       �	�U�<Yc�A�	*

loss*�5= ȝ�       �	��<Yc�A�	*

loss*�=��       �	���<Yc�A�	*

lossx&>�PXT       �	��<Yc�A�	*

lossxQ�=�)ى       �	ٰ�<Yc�A�	*

lossf�"=C?�       �	�G�<Yc�A�	*

loss=�d=_]R�       �	��<Yc�A�	*

loss��>��       �	��<Yc�A�	*

lossV��=��dr       �	��<Yc�A�	*

loss��,=!�f       �	���<Yc�A�	*

loss#T >3᧒       �	�C�<Yc�A�
*

loss��/=p@�a       �	]ܡ<Yc�A�
*

lossdl�=��Q�       �	��<Yc�A�
*

lossDB�=�M�^       �	� �<Yc�A�
*

loss&c�<��       �	ͣ<Yc�A�
*

loss]�@=��;       �	�f�<Yc�A�
*

loss$I=)�k�       �	�J�<Yc�A�
*

losslL�=� >�       �	��<Yc�A�
*

losst�=>a�^       �	���<Yc�A�
*

loss-��=_i�3       �	�$�<Yc�A�
*

lossA�=�'7       �	E��<Yc�A�
*

losss��=^��5       �	���<Yc�A�
*

loss=M�=�zm�       �	%A�<Yc�A�
*

lossA
-=R��       �	ݩ<Yc�A�
*

loss� �=���#       �	r�<Yc�A�
*

loss>�L��       �	+�<Yc�A�
*

loss��<��       �	ǻ�<Yc�A�
*

lossD�=���f       �	T�<Yc�A�
*

loss�`0=OY'�       �	��<Yc�A�
*

loss��>�0��       �	1��<Yc�A�
*

loss&	0=jG�       �	X8�<Yc�A�
*

lossɸ<`w�@       �	خ<Yc�A�
*

loss��<��\$       �	mq�<Yc�A�
*

lossc��=���@       �	S�<Yc�A�
*

lossl�J=�W�       �	��<Yc�A�
*

loss�:�=�e)i       �	?5�<Yc�A�
*

loss��">�3�       �	@۱<Yc�A�
*

lossw!<EE       �	~Ĳ<Yc�A�
*

loss�t=�#�       �	i�<Yc�A�
*

loss�g�=�wˈ       �	���<Yc�A�
*

loss���=�]s;       �	��<Yc�A�
*

loss��_=��&       �	�6�<Yc�A�
*

lossXa}=X�/�       �	�ڵ<Yc�A�
*

loss:�=�<%>       �	n��<Yc�A�
*

loss$�=`�l       �	~�<Yc�A�
*

loss=�;=���        �	B��<Yc�A�
*

lossb3=EPcH       �	z�<Yc�A�
*

loss}�<�[˚       �	Y�<Yc�A�
*

lossS�=�Z�       �	(��<Yc�A�
*

loss���=�{�       �	�Z�<Yc�A�
*

loss$�<�͕       �	��<Yc�A�
*

lossW)�=���z       �	'��<Yc�A�
*

losst��=�5<       �	�@�<Yc�A�
*

loss ��=k���       �	��<Yc�A�
*

loss� y=�qX#       �	��<Yc�A�
*

loss��<;�       �	�(�<Yc�A�
*

loss%"=5��       �	2˾<Yc�A�
*

loss��w=~8�"       �	�a�<Yc�A�
*

lossá=�c       �	���<Yc�A�
*

loss�?;>瓈       �	y��<Yc�A�
*

loss�4�=��RU       �	�o�<Yc�A�
*

lossZ�=g�       �	*�<Yc�A�
*

loss��[=��%       �	z��<Yc�A�
*

loss{S=*���       �	�c�<Yc�A�
*

loss
55=K���       �	��<Yc�A�
*

loss.�=���       �	���<Yc�A�
*

lossL{P=��nR       �	�R�<Yc�A�
*

loss�j�=i�g)       �	_��<Yc�A�
*

lossƓC<ٜ�       �	ς�<Yc�A�
*

loss=�}=0��       �	��<Yc�A�
*

loss��=�+Ei       �	���<Yc�A�
*

loss_X�<����       �	;n�<Yc�A�
*

lossPE�=�=T       �	w�<Yc�A�
*

loss*٠=]y�       �	6��<Yc�A�
*

lossOK�=)�u�       �	�S�<Yc�A�
*

lossv�=7d��       �	���<Yc�A�
*

lossܠ�<|(�w       �	Ք�<Yc�A�
*

lossӢ�=F��2       �	�+�<Yc�A�
*

lossfv9<%���       �	$�<Yc�A�
*

loss���<*
!^       �	���<Yc�A�
*

losst�)="�C       �	be�<Yc�A�
*

loss���<�v��       �	���<Yc�A�
*

loss�3u=?�|       �	%��<Yc�A�
*

loss�ީ=xN�       �	_(�<Yc�A�
*

loss�҄=3�f       �	2��<Yc�A�
*

loss&�|=ṱ�       �	m��<Yc�A�
*

loss���<��Y       �	jh�<Yc�A�
*

loss��=�_/%       �	7�<Yc�A�
*

loss�>x� �       �	���<Yc�A�
*

loss8L>�2�p       �	hv�<Yc�A�
*

loss6��=���       �	��<Yc�A�
*

loss.�=ы�m       �	���<Yc�A�
*

loss�p=A�=�       �	�T�<Yc�A�
*

loss� �=%Kv       �	���<Yc�A�
*

loss�"=h���       �	~��<Yc�A�
*

loss�*�=��v       �	0-�<Yc�A�
*

loss�A�=�Y2�       �	���<Yc�A�
*

loss�T=4Y�       �	 b�<Yc�A�
*

loss�Ŵ=\S�1       �	m�<Yc�A�
*

loss�ҧ=Ģ��       �	(��<Yc�A�
*

loss1s�=HL��       �	XX�<Yc�A�
*

loss�+�=���       �	�<�<Yc�A�
*

loss*��<�[�=       �	���<Yc�A�
*

lossA%7=�bY�       �	�n�<Yc�A�
*

lossSIr>��       �	V�<Yc�A�
*

lossHq�<�œ�       �	2��<Yc�A�
*

lossLn=5��0       �	�G�<Yc�A�
*

loss1K>�={       �	a��<Yc�A�
*

lossi>\E*�       �	>z�<Yc�A�
*

loss qS=>��       �	��<Yc�A�
*

loss@J,=,ke.       �	X��<Yc�A�
*

lossM��<NI�       �	gE�<Yc�A�
*

lossn�=�ٙ�       �	�
�<Yc�A�
*

loss�=�Rz�       �	��<Yc�A�
*

loss�<n<���       �	½�<Yc�A�
*

loss�m�<��@       �	�c�<Yc�A�
*

loss�%=�E�       �	��<Yc�A�
*

loss��='{�       �	֪�<Yc�A�
*

lossZ�=���-       �	�K�<Yc�A�
*

loss��=	��       �	���<Yc�A�
*

loss�Mw=�(��       �	���<Yc�A�
*

loss,s�=�\�O       �	�q�<Yc�A�
*

loss[#�=�P)-       �	�=�<Yc�A�
*

loss�4=���k       �	���<Yc�A�
*

lossW�<�5       �	A��<Yc�A�
*

loss��j<��       �	�$�<Yc�A�
*

loss��=�       �	]��<Yc�A�
*

lossEF�=��_       �	H��<Yc�A�
*

lossK�=���       �	�?�<Yc�A�
*

loss��=@�E,       �	G��<Yc�A�
*

loss_�=sx��       �	�x�<Yc�A�
*

loss_F�=h��~       �	�-�<Yc�A�
*

lossI�=%S6       �	���<Yc�A�
*

lossD�y=x��       �	Xs�<Yc�A�
*

loss�2�=�Ǎ�       �	��<Yc�A�
*

loss�&�<�"E       �	��<Yc�A�
*

loss!UV= �\       �	g`�<Yc�A�
*

loss��=W��v       �	��<Yc�A�
*

loss�b�=�A�       �	���<Yc�A�*

loss
՘=��v�       �	�K�<Yc�A�*

loss�$=hq�l       �	���<Yc�A�*

lossf5�=*�t       �	؛�<Yc�A�*

loss/ߣ=�}h�       �	�3�<Yc�A�*

lossm=Қ�u       �	���<Yc�A�*

loss�@�<1���       �	�t�<Yc�A�*

lossI��=�H�O       �	��<Yc�A�*

loss\͋=&z3�       �	���<Yc�A�*

loss3g=���`       �	9D�<Yc�A�*

loss���=%�^�       �	���<Yc�A�*

loss�p>(rW       �	���<Yc�A�*

losssa�=V�r�       �	(�<Yc�A�*

lossf7�<βw�       �	���<Yc�A�*

loss8��=ڸ@       �	)]�<Yc�A�*

loss��>n�M       �	���<Yc�A�*

loss?�>5�       �	��<Yc�A�*

loss���<�ѽ       �	j0 =Yc�A�*

lossIjc=�k!       �	�� =Yc�A�*

loss�9�<hV�2       �	W`=Yc�A�*

loss�hR=�"�       �	s�=Yc�A�*

loss]=vO�z       �	�=Yc�A�*

loss���=HH       �	�4=Yc�A�*

lossEJ=��v�       �	y�=Yc�A�*

loss�'�<�>�C       �	{g=Yc�A�*

lossR��=@ћ�       �	Y�=Yc�A�*

loss�\�=��       �	��=Yc�A�*

lossr]�=b�p       �	/4=Yc�A�*

lossz�=��O       �	K�=Yc�A�*

loss��={Y!       �	�p=Yc�A�*

loss���=��r�       �	(=Yc�A�*

loss��)=n�L�       �	��=Yc�A�*

loss���<���       �	S	=Yc�A�*

lossT�b=$�       �	��	=Yc�A�*

loss�߽=h�S       �	�
=Yc�A�*

loss-�q=|�,0       �	�=Yc�A�*

loss�q�=�-�l       �	��=Yc�A�*

loss}��=���       �	�R=Yc�A�*

losst�=m2       �	��=Yc�A�*

lossp�=}��       �	ʍ=Yc�A�*

lossF��<�U��       �	%Z=Yc�A�*

loss��=��x       �	4�=Yc�A�*

loss�pJ=k䩔       �	��=Yc�A�*

loss3��=��e�       �	5=Yc�A�*

loss��+=p�       �	��=Yc�A�*

loss�t=!�_       �	�v=Yc�A�*

loss܀%<Y��?       �	Q=Yc�A�*

loss�@B=D7�       �	��=Yc�A�*

loss���<eh�o       �	bN=Yc�A�*

lossl�=_C%#       �	)�=Yc�A�*

loss��X=mH��       �	��=Yc�A�*

loss���<8��       �	T:=Yc�A�*

loss;��=�4t�       �	S<=Yc�A�*

loss��<"�        �	��=Yc�A�*

losse�/=W��P       �	6�=Yc�A�*

lossmr>z�Z       �	��=Yc�A�*

loss��<����       �	�=Yc�A�*

loss��{=�;��       �	D�=Yc�A�*

loss��H=Yp�       �	��=Yc�A�*

loss�!�=��Q       �		�=Yc�A�*

loss!WT=Ů��       �	�y=Yc�A�*

loss
�>�       �	�h=Yc�A�*

lossL;�=/kv       �	�P=Yc�A�*

loss ��<||�        �	� =Yc�A�*

loss&�=t�=       �	�L!=Yc�A�*

lossxp=���       �	�"=Yc�A�*

loss��=@Zc       �	�0#=Yc�A�*

lossTk�=�<T�       �	i�#=Yc�A�*

loss�UX>���       �	,�$=Yc�A�*

loss���<�a�       �	Y%=Yc�A�*

loss��N=�Qi       �	0d&=Yc�A�*

lossA�^=T�       �	��&=Yc�A�*

loss2�^=?'(
       �	��'=Yc�A�*

loss <�q�       �	I�(=Yc�A�*

loss��<�7�?       �	��)=Yc�A�*

loss��r=�K�       �	oc*=Yc�A�*

loss���<(eU�       �	j�*=Yc�A�*

lossu0>e���       �	�+=Yc�A�*

loss�"�=�l)2       �	�8,=Yc�A�*

loss߱�=��?�       �	��,=Yc�A�*

loss�d1=�d	�       �	��-=Yc�A�*

loss�xo=���       �	().=Yc�A�*

loss��0<��@       �	۾.=Yc�A�*

loss�w�<��S       �	�]/=Yc�A�*

losso�=�N�       �	��/=Yc�A�*

loss }<���       �	��0=Yc�A�*

loss�<��qT       �	�#1=Yc�A�*

loss�M=�-�       �	��1=Yc�A�*

loss	%>g.       �	ms2=Yc�A�*

loss�=����       �	�3=Yc�A�*

loss;�J>KR{       �	ߨ3=Yc�A�*

loss�/�=ܼۥ       �	IG4=Yc�A�*

loss�A�=�rU       �	��4=Yc�A�*

loss\eS=-D�p       �	�y5=Yc�A�*

lossO�=ϰ�       �	Z6=Yc�A�*

loss�;�<��u       �	�6=Yc�A�*

loss�F�=����       �	�B7=Yc�A�*

loss|�|=%�\&       �	��7=Yc�A�*

lossI(A=s��       �	)v8=Yc�A�*

loss���<��Y       �	\ 9=Yc�A�*

loss4'�=��       �	&�9=Yc�A�*

loss�3C=y��[       �	��:=Yc�A�*

lossq >��E"       �	� ;=Yc�A�*

loss��=��       �	(�;=Yc�A�*

loss �D=����       �	�S<=Yc�A�*

losszIx=�T �       �	@�<=Yc�A�*

lossA8v=�m��       �	��==Yc�A�*

loss�<=Y�       �	�`>=Yc�A�*

loss}ŭ=��       �	�?=Yc�A�*

lossE+-=B�rL       �	�?=Yc�A�*

losseNi=|��       �	;5@=Yc�A�*

loss�}�=��9�       �	J�@=Yc�A�*

loss�KL=���       �	QlA=Yc�A�*

loss8h;=����       �	�B=Yc�A�*

loss��=���       �	u�B=Yc�A�*

loss.�P=�z�_       �	�HC=Yc�A�*

loss�؂=���       �	"D=Yc�A�*

loss&�=�?        �	M�D=Yc�A�*

loss{P.>-���       �	~RE=Yc�A�*

loss!M8=+~{?       �	J�E=Yc�A�*

loss���<j���       �	܄F=Yc�A�*

loss[�Q='m��       �	XG=Yc�A�*

losso�>+��       �	��G=Yc�A�*

loss� =*��       �	�LH=Yc�A�*

loss$��<��       �	d�H=Yc�A�*

losse��<��g�       �	y�I=Yc�A�*

loss�>=���       �	�)J=Yc�A�*

loss8��=H�L       �	��J=Yc�A�*

loss(Z�=�ؗ7       �	�uK=Yc�A�*

loss�S�=���       �	CL=Yc�A�*

loss�G=�t�       �	��L=Yc�A�*

loss<��=���       �	?�M=Yc�A�*

loss_��;gbq�       �	2N=Yc�A�*

loss�=*�B�       �	R�N=Yc�A�*

loss�)	=��8       �	�tO=Yc�A�*

lossjR�=��\       �	�P=Yc�A�*

lossc��=�_#       �	4�P=Yc�A�*

loss���>_g�-       �	�Q=Yc�A�*

loss`��=�� �       �	5R=Yc�A�*

loss�ʇ< �&       �	��R=Yc�A�*

lossZ"�=�?�C       �	�mS=Yc�A�*

loss��W=��S       �	fT=Yc�A�*

lossxBs=���       �	��T=Yc�A�*

loss�/u=��       �	��U=Yc�A�*

loss���<�E��       �	{V=Yc�A�*

loss u�<��Q       �	�W=Yc�A�*

loss�5=�1f       �	�W=Yc�A�*

lossw�=b�@j       �	Y=Yc�A�*

loss�P�<qE�       �	ƣY=Yc�A�*

loss8�N=�~Y       �	[?Z=Yc�A�*

lossX =�DS       �	�Z=Yc�A�*

loss#Z=�2�       �	{I\=Yc�A�*

loss|_*=�*1�       �	�\=Yc�A�*

loss�q�=l�       �	�]=Yc�A�*

loss�C�<�S�\       �	�'^=Yc�A�*

loss�WX=MC�N       �	*�^=Yc�A�*

loss�%B>�X��       �	�__=Yc�A�*

lossi�=��4�       �	f�_=Yc�A�*

loss��=�˥       �	�`=Yc�A�*

loss�=�8'�       �	�a=Yc�A�*

loss~�=L�B       �	��a=Yc�A�*

lossH�r<�!'�       �	�eb=Yc�A�*

loss%��<|���       �	��c=Yc�A�*

loss�d=��|       �	od=Yc�A�*

loss��{=:��z       �	
+e=Yc�A�*

loss�=����       �	r�e=Yc�A�*

lossN=HG�9       �	�Zf=Yc�A�*

loss68D=͡:�       �	n�f=Yc�A�*

lossa�%>俑       �	��g=Yc�A�*

loss�ʱ=�+4       �	#.h=Yc�A�*

lossA�=U��       �	�h=Yc�A�*

loss��6=�N�       �	xi=Yc�A�*

loss(�>k��       �	�j=Yc�A�*

loss֬�=}��       �	
�j=Yc�A�*

loss�_�=����       �	B[k=Yc�A�*

loss7=��2        �	:l=Yc�A�*

loss���<� �       �	V�l=Yc�A�*

loss18<�'       �	��m=Yc�A�*

loss��^=f>��       �	ōn=Yc�A�*

loss���=�\T       �	�o=Yc�A�*

loss�i�=!NGI       �	�p=Yc�A�*

loss��b=;9&�       �	��p=Yc�A�*

loss�,�<u��N       �	�Nq=Yc�A�*

loss��={�       �	[�q=Yc�A�*

loss�#�=f�6       �	3�r=Yc�A�*

loss 6'=��=       �	#s=Yc�A�*

lossnh�=2���       �	a�s=Yc�A�*

loss���<���       �	�gt=Yc�A�*

loss��F>��(       �	�u=Yc�A�*

lossb0	=����       �	��u=Yc�A�*

loss��_==]S!       �	QJv=Yc�A�*

lossS��<��S       �	��v=Yc�A�*

lossEN>=r��       �	��w=Yc�A�*

loss���<�vO       �	�(x=Yc�A�*

lossH�3=Lř       �	��x=Yc�A�*

lossx�>�( �       �	�ky=Yc�A�*

loss�=�<)��       �	�Vz=Yc�A�*

loss�G�<��w�       �	
�z=Yc�A�*

loss�x=�n�       �	U�{=Yc�A�*

loss�:�=�W�^       �	�^|=Yc�A�*

loss�4�=���       �	��|=Yc�A�*

loss)'�=+�uj       �	-�}=Yc�A�*

lossOrB=�_ �       �	.7~=Yc�A�*

loss�=�.=       �	��~=Yc�A�*

loss��=��"       �	�z=Yc�A�*

loss;-�=�^�%       �		�=Yc�A�*

lossý>c���       �	8��=Yc�A�*

loss}�=��s       �	�n�=Yc�A�*

loss��>�Z�       �	*�=Yc�A�*

lossͱ�<�x�       �	t΂=Yc�A�*

loss8%=?�I       �	6t�=Yc�A�*

loss�'�=����       �	t%�=Yc�A�*

lossOm�=���       �	n��=Yc�A�*

loss���<��_       �	�n�=Yc�A�*

loss��=
g\R       �	�=Yc�A�*

loss�.�<6O��       �	D��=Yc�A�*

loss���=%�0�       �	Zb�=Yc�A�*

loss�؁=γ       �	�=Yc�A�*

loss��>Bz       �	g��=Yc�A�*

lossf�y=M�P       �	FE�=Yc�A�*

losst��=�(Ʋ       �	��=Yc�A�*

loss�eQ=l�a�       �	=Yc�A�*

lossEī<��=        �	5)�=Yc�A�*

loss���=���L       �	oӋ=Yc�A�*

loss{�(=�RV�       �	�q�=Yc�A�*

lossf̡<�K�\       �	� �=Yc�A�*

lossS�2=>7       �	Ի�=Yc�A�*

losssE�=ȐE�       �	bh�=Yc�A�*

loss�0a=�.�       �	��=Yc�A�*

loss4��<)��p       �	���=Yc�A�*

loss4��<�ǜ�       �	�C�=Yc�A�*

loss�X=�+�       �	�=Yc�A�*

loss1xK=�G�#       �	���=Yc�A�*

loss���=ٗ	�       �	[&�=Yc�A�*

loss64?>;�*�       �	�Ò=Yc�A�*

loss#!F=BF��       �	�`�=Yc�A�*

loss/�=�T��       �	'��=Yc�A�*

loss��@<9�3       �	��=Yc�A�*

loss=�C=�
�       �	�F�=Yc�A�*

loss���>V͉�       �	��=Yc�A�*

loss�6>=�1*�       �	l{�=Yc�A�*

loss�/�=�t��       �	�)�=Yc�A�*

loss��h=V��       �	ʗ=Yc�A�*

loss�(�=��1�       �	Pp�=Yc�A�*

lossI�=�'       �	j�=Yc�A�*

lossL
E=�J�       �	���=Yc�A�*

loss��>�_�       �	�r�=Yc�A�*

loss�-H>�B�       �	�;�=Yc�A�*

loss�׫=탛�       �	E؛=Yc�A�*

lossQ+�=D���       �	2r�=Yc�A�*

loss��`=_��       �	�=�=Yc�A�*

loss�&�=oR̗       �	��=Yc�A�*

lossV��=��џ       �	���=Yc�A�*

loss:<0=�3       �	JC�=Yc�A�*

loss$��<��6       �	/۟=Yc�A�*

lossiJ:=.t��       �	v�=Yc�A�*

loss��=9շi       �	��=Yc�A�*

loss�ӥ=�11       �	���=Yc�A�*

loss/uH=4���       �	�T�=Yc�A�*

loss2��=ݨ       �	���=Yc�A�*

loss<{�<=Z!p       �	D��=Yc�A�*

lossi�e=C�ę       �	}%�=Yc�A�*

loss#}<��9       �	z¤=Yc�A�*

loss�i�=սI>       �	�n�=Yc�A�*

loss(�^=��p�       �	��=Yc�A�*

loss�W�=9,6�       �	5��=Yc�A�*

loss�J�=*�)�       �	�N�=Yc�A�*

loss��<i��5       �	f��=Yc�A�*

loss6�<�c�       �	��=Yc�A�*

loss�<���[       �	s�=Yc�A�*

lossOSF=��Rk       �	E�=Yc�A�*

loss���<��       �	���=Yc�A�*

loss�au=p��       �	)Z�=Yc�A�*

loss�g>@�a�       �	���=Yc�A�*

lossJ��=j�+&       �	ҧ�=Yc�A�*

lossc$=p��       �	g��=Yc�A�*

loss\
>@k�       �	z�=Yc�A�*

loss*=w��       �	˾�=Yc�A�*

loss�z�<��eh       �	B��=Yc�A�*

loss�+	=�WS       �	mU�=Yc�A�*

loss�-�=^�K)       �	X:�=Yc�A�*

loss�==����       �	<2�=Yc�A�*

loss,��=]�       �	{�=Yc�A�*

loss�{=���       �	���=Yc�A�*

loss�؞=���       �	�N�=Yc�A�*

loss�O=ߙs�       �	G�=Yc�A�*

lossd�D=M�a       �	4��=Yc�A�*

lossa �<��R�       �	��=Yc�A�*

lossp�<܎b       �	V��=Yc�A�*

loss}ۏ=����       �	ۉ�=Yc�A�*

loss��=�v       �	�!�=Yc�A�*

loss8�=�r       �	Qٸ=Yc�A�*

loss�nh=�       �	�x�=Yc�A�*

lossO��=Hu��       �	a�=Yc�A�*

loss��<ƕ^&       �	���=Yc�A�*

loss��X=�چ       �	�c�=Yc�A�*

loss�	=��]B       �	��=Yc�A�*

loss�Xd= K'�       �	˝�=Yc�A�*

loss�=6�{       �	�4�=Yc�A�*

lossQ��= !��       �	Iֽ=Yc�A�*

lossؠ�<A��       �	�q�=Yc�A�*

loss�U�=[r       �	q9�=Yc�A�*

loss�E�=�>��       �	�ۿ=Yc�A�*

loss�!=A�H�       �	v�=Yc�A�*

loss��='J       �	�=Yc�A�*

loss&�<�	��       �	���=Yc�A�*

loss��g=�a;       �	A�=Yc�A�*

loss�l=��Q       �	���=Yc�A�*

lossh��=1�E       �	l|�=Yc�A�*

loss��Q=/E{       �	+�=Yc�A�*

loss�f=[x       �	���=Yc�A�*

loss���=V�Dj       �	�q�=Yc�A�*

lossiid=�nj       �	A�=Yc�A�*

loss�O=j��       �	{��=Yc�A�*

lossO�]=�2�       �	�^�=Yc�A�*

loss�T�;0v��       �	�:�=Yc�A�*

loss�!=�V��       �	x��=Yc�A�*

loss��<�y.t       �	ӆ�=Yc�A�*

loss�t�=	|�       �	2�=Yc�A�*

loss�#*=E���       �	f��=Yc�A�*

loss��=��z       �	R�=Yc�A�*

lossZ��=�w�       �	���=Yc�A�*

lossܦq=S���       �	��=Yc�A�*

loss�ғ=骞�       �	� �=Yc�A�*

loss!K�=t6ts       �	ܺ�=Yc�A�*

lossϯ�<��o�       �	Mh�=Yc�A�*

lossR�f;�EȻ       �	��=Yc�A�*

lossaV =Ĭ|       �	G��=Yc�A�*

loss�b<@]�       �	�H�=Yc�A�*

loss�=<dq�       �	���=Yc�A�*

loss!��<��}       �	�~�=Yc�A�*

loss�.)=X�:G       �	!�=Yc�A�*

lossl�Y=Byp       �	���=Yc�A�*

loss`�g;@�H�       �	���=Yc�A�*

loss��B:��       �	���=Yc�A�*

lossӈ;���-       �	���=Yc�A�*

lossEB=��H       �	�E�=Yc�A�*

loss7�=�m�g       �	���=Yc�A�*

loss���=���1       �	���=Yc�A�*

loss�S�;�e�       �	���=Yc�A�*

loss��<Eol       �	�\�=Yc�A�*

loss��Z>�wj       �	G �=Yc�A�*

loss���<n�       �	H��=Yc�A�*

lossԗ=�TA�       �	9��=Yc�A�*

loss1]�=?���       �	���=Yc�A�*

lossO߁=T�K       �	�0�=Yc�A�*

losskG�=CgNa       �	���=Yc�A�*

loss��N=�) �       �	mt�=Yc�A�*

loss�i�=x�q�       �	��=Yc�A�*

loss���=W���       �	-��=Yc�A�*

lossqE=�&8D       �	Y��=Yc�A�*

losss��=mb�       �	$�=Yc�A�*

lossxT=C\��       �	3��=Yc�A�*

loss���=���"       �	�^�=Yc�A�*

loss�f�=���       �	i �=Yc�A�*

loss�E�=��K�       �	O��=Yc�A�*

loss��=�PoA       �	մ�=Yc�A�*

loss6}�=�1ͥ       �	�Q�=Yc�A�*

lossd�H=�Z�j       �	���=Yc�A�*

losse�a=��>�       �	��=Yc�A�*

loss�^E=a��|       �	���=Yc�A�*

lossf�'=�֑       �	wL�=Yc�A�*

lossx4=s���       �	��=Yc�A�*

lossEi~=�,B       �	��=Yc�A�*

loss<�4=���       �	6�=Yc�A�*

loss�E=˟a\       �	���=Yc�A�*

lossC�<x� �       �	W{�=Yc�A�*

loss���<��       �	��=Yc�A�*

loss�r=T��       �	�h�=Yc�A�*

lossi+�<�2yP       �	`9�=Yc�A�*

loss��>I}       �	���=Yc�A�*

loss��==5")       �	�k�=Yc�A�*

loss��<n���       �	i�=Yc�A�*

loss�#�=��       �	���=Yc�A�*

loss@,=�mƝ       �	�o�=Yc�A�*

loss�;<	W��       �	��=Yc�A�*

lossW��<���N       �	��=Yc�A�*

loss4��=.���       �	�B�=Yc�A�*

loss���<��}-       �	%�=Yc�A�*

loss�>�=n���       �	��=Yc�A�*

lossV�>0D�       �	�6�=Yc�A�*

loss��n=Ln�       �	���=Yc�A�*

loss�C<�;�       �	{k�=Yc�A�*

loss���<�Zw�       �	��=Yc�A�*

lossC�<El
�       �	ͮ�=Yc�A�*

loss<Y=V�!�       �	�H�=Yc�A�*

loss���<����       �	��=Yc�A�*

lossJͺ=�Bb�       �	���=Yc�A�*

loss��=��d       �	�/�=Yc�A�*

lossE��<��O       �	u��=Yc�A�*

loss�l�=R�x       �	�n�=Yc�A�*

loss�ek<pnS       �	u�=Yc�A�*

loss��<�>       �	���=Yc�A�*

lossi1>�z�       �	�R>Yc�A�*

lossDOe=���       �	Z�>Yc�A�*

lossM�>��B�       �	Ό>Yc�A�*

loss�p�<Q       �	�">Yc�A�*

loss���=baL       �	�>Yc�A�*

loss���<g�+�       �	h#>Yc�A�*

lossdː=)       �	��>Yc�A�*

lossa� =��gM       �	�>Yc�A�*

lossK>�Zo       �	�@>Yc�A�*

lossd%�= �       �	u!>Yc�A�*

loss���<���|       �	J�>Yc�A�*

loss�m�<!��       �	�m>Yc�A�*

lossF��=��][       �	C>Yc�A�*

loss��=9�i�       �	Ý>Yc�A�*

loss�5	=,���       �	PR>Yc�A�*

lossjԍ=���~       �	��>Yc�A�*

loss@R�;6I۫       �	� >Yc�A�*

loss%�<�R,       �	/�!>Yc�A�*

loss��%=��R�       �	��">Yc�A�*

loss���=⻧�       �	��#>Yc�A�*

loss
��=��Q�       �	��$>Yc�A�*

loss�ؘ={�0       �	؀%>Yc�A�*

loss�{=<z#       �	]&>Yc�A�*

lossN �=n½w       �	/�&>Yc�A�*

lossr��=��       �	(>Yc�A�*

loss�Xo=��       �	��(>Yc�A�*

lossn=�Z.n       �	n�)>Yc�A�*

loss��/<�x       �	�K*>Yc�A�*

lossa��=�
t�       �	gc+>Yc�A�*

lossAp�=p��       �		�+>Yc�A�*

loss%=�x��       �	��,>Yc�A�*

loss}�
=�O       �	c->Yc�A�*

loss4fX=�a�C       �	� .>Yc�A�*

loss^%>A���       �	��.>Yc�A�*

loss�x=7O��       �	�y/>Yc�A�*

loss��y= ���       �	{0>Yc�A�*

loss���<�REr       �	R�0>Yc�A�*

loss�`"==씍       �	�K1>Yc�A�*

loss�`>�u�       �	K�1>Yc�A�*

loss!��=�>�       �	��2>Yc�A�*

loss�S�=�&�       �	�Q3>Yc�A�*

lossO��=)�x�       �	�3>Yc�A�*

loss�>�< ��       �	�~4>Yc�A�*

lossnH>��W       �	�85>Yc�A�*

lossN��<�>�       �	��5>Yc�A�*

loss���=��t>       �	j6>Yc�A�*

lossX�7=��#a       �	�7>Yc�A�*

loss	d�=p���       �	p�7>Yc�A�*

loss�D�<P��       �	H58>Yc�A�*

loss�=���       �	��8>Yc�A�*

loss�d�<5���       �	l9>Yc�A�*

loss|5�=��i�       �	�:>Yc�A�*

lossk�=��       �	w�:>Yc�A�*

loss��>�H@       �	�;>Yc�A�*

loss�Z�<y�       �	z<>Yc�A�*

loss7�@<��B�       �	��<>Yc�A�*

loss �<��S       �	6w=>Yc�A�*

loss/�G=ڨ �       �	�>>Yc�A�*

loss+K=��Z       �	��>>Yc�A�*

loss��L=�F�       �	�A?>Yc�A�*

lossQ�=߫��       �	h�?>Yc�A�*

lossd��=�%�       �	f�@>Yc�A�*

loss��x=[�M�       �	�A>Yc�A�*

loss,�=��       �	U�A>Yc�A�*

lossV�W=䳯�       �	2WB>Yc�A�*

loss|l�<��i       �	�C>Yc�A�*

losss5�=���z       �		�C>Yc�A�*

loss��=ػ       �	�=D>Yc�A�*

loss���=�ׂ       �	��D>Yc�A�*

lossW��=�~A�       �	]nE>Yc�A�*

loss%E=�[       �	-F>Yc�A�*

losso�=��9       �	�F>Yc�A�*

loss�'W=z�4       �	i9G>Yc�A�*

lossM�
=��       �	9�G>Yc�A�*

loss�= �@M       �	pH>Yc�A�*

loss��=��1       �	�I>Yc�A�*

lossn��=$�U       �	��I>Yc�A�*

lossO�=+=�D       �	PVJ>Yc�A�*

loss�r�<s�X>       �	"�J>Yc�A�*

lossO�I=n@�<       �	�K>Yc�A�*

loss�׍=��==       �	�:L>Yc�A�*

lossr�/=����       �	��L>Yc�A�*

loss���=���?       �	�aM>Yc�A�*

loss��<��       �	��M>Yc�A�*

loss&��<�i|       �	%�N>Yc�A�*

loss��<R��       �	�oO>Yc�A�*

loss�=A'�       �	P>Yc�A�*

lossr�Y=�5�?       �	��P>Yc�A�*

loss�<�p*       �	a3Q>Yc�A�*

loss|E�<��       �	��Q>Yc�A�*

loss_Ŧ=��       �	��R>Yc�A�*

loss|ފ=p���       �	>$S>Yc�A�*

loss��=q,�       �	��S>Yc�A�*

loss��0=�h��       �	~qT>Yc�A�*

loss0="'�       �	�UU>Yc�A�*

lossq�;}ކ�       �	Z�U>Yc�A�*

losss��=��<�       �	�V>Yc�A�*

loss�w?=��=       �	]7W>Yc�A�*

loss�K>�w/�       �	��W>Yc�A�*

loss|�=$Ne�       �	�qX>Yc�A�*

loss�m{<��ab       �	�Y>Yc�A�*

loss��T=�n       �	j�Y>Yc�A�*

loss��h=,9H       �	h^Z>Yc�A�*

loss�~<�Y�c       �	M�Z>Yc�A�*

loss�=��MK       �	� \>Yc�A�*

loss�>s�$�       �	��\>Yc�A�*

loss�<+�s       �	�T]>Yc�A�*

lossə�<ٖ�6       �	��]>Yc�A�*

lossF|�=�/�.       �	w�^>Yc�A�*

lossmJ�=���f       �	we_>Yc�A�*

loss̈́�=�
�g       �	�`>Yc�A�*

loss)h7=��       �	��`>Yc�A�*

loss�'�=X+�       �	ua>Yc�A�*

lossAm<����       �	�b>Yc�A�*

loss\E�<��%B       �	��b>Yc�A�*

loss.c�<VFg       �	�ac>Yc�A�*

loss)�i<^�.       �	|&d>Yc�A�*

loss���=�p��       �	��d>Yc�A�*

loss��=#F;�       �	Gwe>Yc�A�*

lossrG�<��sD       �	?f>Yc�A�*

lossxb�<�`�       �	��f>Yc�A�*

loss�=8��       �	gag>Yc�A�*

loss�|=�d9       �	� h>Yc�A�*

loss�n=��8       �	ˡh>Yc�A�*

loss�<~�       �	�@i>Yc�A�*

loss]�=��\       �	��i>Yc�A�*

loss;�=SC�\       �	p#k>Yc�A�*

loss���<=�&       �	��k>Yc�A�*

lossfH�=z?��       �	&ol>Yc�A�*

lossE��=�3�)       �	zm>Yc�A�*

loss{=;��       �	�An>Yc�A�*

loss�=@�       �	��n>Yc�A�*

loss�>�<-�_;       �	��o>Yc�A�*

lossϼa=ٝ"d       �	�Dp>Yc�A�*

loss`�=<��F       �	��p>Yc�A�*

lossQ�\<�V�~       �	��q>Yc�A�*

loss�B<�U�A       �	�Kr>Yc�A�*

loss6��<1~�(       �	�r>Yc�A�*

lossGr�<�R��       �	<�s>Yc�A�*

loss
�e<�H��       �	�t>Yc�A�*

loss.�6<%�~       �	uu>Yc�A�*

loss	�=>.�       �	�!v>Yc�A�*

lossz�"=o��!       �	��v>Yc�A�*

loss���=���       �	�kw>Yc�A�*

lossO(�=3	�>       �	 	x>Yc�A�*

loss�&�<��΍       �	H�x>Yc�A�*

loss#[=w��       �	��y>Yc�A�*

loss`1i<y)�s       �	�vz>Yc�A�*

loss�z<w�<       �	r{>Yc�A�*

loss =n���       �	ȷ{>Yc�A�*

lossT� =�M�       �	}^|>Yc�A�*

lossݠs=�+;�       �	� }>Yc�A�*

lossݍ�=��0       �	��}>Yc�A�*

lossF�=*ߵ�       �	�<~>Yc�A�*

loss�H=�\       �	��~>Yc�A�*

loss��"<ք�s       �	��>Yc�A�*

loss�=����       �	%"�>Yc�A�*

lossii=��n�       �	�ր>Yc�A�*

loss��=z       �	��>Yc�A�*

loss�<�2�       �	�u�>Yc�A�*

loss�=(�Z       �	&9�>Yc�A�*

loss�8�<��A�       �	Qރ>Yc�A�*

loss��7=@?ˆ       �	���>Yc�A�*

loss&T�<��	\       �	KW�>Yc�A�*

loss�jw=�A1�       �	��>Yc�A�*

loss���<}M0|       �	x��>Yc�A�*

lossZi<(VR�       �	�@�>Yc�A�*

loss��
>��       �	9�>Yc�A�*

loss� =��       �	���>Yc�A�*

loss�}�<~��
       �	JD�>Yc�A�*

loss��=Vx[       �	��>Yc�A�*

loss���<8���       �	���>Yc�A�*

lossL�&<o?�       �	O;�>Yc�A�*

lossq��=�-�       �	g֋>Yc�A�*

loss[��;`�L�       �	�|�>Yc�A�*

loss�ږ=Xt��       �	P�>Yc�A�*

loss@�,=�O
       �	]��>Yc�A�*

loss�=(u:       �	Xp�>Yc�A�*

loss���=���       �	H�>Yc�A�*

loss�'=`N��       �	I��>Yc�A�*

loss(9<7���       �	Uj�>Yc�A�*

loss4�T=�^��       �	�>Yc�A�*

loss6�Z=U��       �	��>Yc�A�*

loss��<��r�       �	b�>Yc�A�*

lossx�<h~|C       �	�
�>Yc�A�*

loss�J=TB��       �	��>Yc�A�*

loss�/�=��A       �	�B�>Yc�A�*

lossV� =[ִ�       �	=�>Yc�A�*

loss�D=�j !       �	~��>Yc�A�*

lossq��=�"��       �	�Q�>Yc�A�*

loss��=�iy       �	��>Yc�A�*

loss���<�>�*       �	���>Yc�A�*

lossb�<K��       �	E�>Yc�A�*

losse{<QcR�       �	�2�>Yc�A�*

loss�P�; *�`       �	�ޚ>Yc�A�*

loss�}=�U�y       �	���>Yc�A�*

lossΙA=��W       �	?�>Yc�A�*

lossa�c=tX�K       �	�ߜ>Yc�A�*

loss���=l�T9       �	_}�>Yc�A�*

lossM�/<ø       �	h�>Yc�A�*

loss�"R= W       �	���>Yc�A�*

loss��<D�{�       �	鵟>Yc�A�*

loss�0=د��       �	�P�>Yc�A�*

lossf!<=VR�	       �	l�>Yc�A�*

lossCT=i�W�       �	Y��>Yc�A�*

lossTi�=B/�A       �	z��>Yc�A�*

loss1~�<��6Z       �	�0�>Yc�A�*

lossʵ.=���       �	|ѣ>Yc�A�*

lossr �<Tױ       �	�s�>Yc�A�*

loss�=s0��       �	��>Yc�A�*

loss�Β=�̟       �	���>Yc�A�*

loss��\=JBc�       �	WC�>Yc�A�*

loss�x�;�       �	���>Yc�A�*

loss
�,=]�|�       �	���>Yc�A�*

loss���=� �?       �	2X�>Yc�A�*

loss���=���o       �	7��>Yc�A�*

loss\5�=,'Y       �	�۩>Yc�A�*

lossj�C=ϸ>       �	�|�>Yc�A�*

loss�Ѭ=�I\v       �	C�>Yc�A�*

loss�E�=�E�       �	\̫>Yc�A�*

losse�8=�X�       �	�n�>Yc�A�*

lossQB=�.	       �	��>Yc�A�*

loss_�= w�       �	-��>Yc�A�*

loss
d>O?��       �	N�>Yc�A�*

loss���<'��\       �	��>Yc�A�*

lossd�=��D�       �	eů>Yc�A�*

loss�-�<�*��       �	�_�>Yc�A�*

lossj�=���e       �	eP�>Yc�A�*

lossc��<8���       �	�?�>Yc�A�*

loss%l�=�&�       �	�ܲ>Yc�A�*

loss���<m��B       �	x�>Yc�A�*

loss rI<�Q�       �	w�>Yc�A�*

loss�3'=�-C}       �	��>Yc�A�*

loss��=���       �	�ص>Yc�A�*

lossL�M=i�,�       �	"o�>Yc�A�*

loss��</��e       �	9
�>Yc�A�*

lossH��=�.�       �	Q��>Yc�A�*

loss}m�=h�*Q       �	�:�>Yc�A�*

lossSm�=**A�       �	��>Yc�A�*

lossv�+<k�       �	Z��>Yc�A�*

loss	%=	�T3       �	�)�>Yc�A�*

lossq�=�h=�       �	6ʺ>Yc�A�*

lossf�\= m,�       �	^i�>Yc�A�*

lossl�	=@<E1       �	�>Yc�A�*

loss�X=!��'       �	Ϡ�>Yc�A�*

loss�؎=B�	�       �	�<�>Yc�A�*

lossN5=U�=       �	��>Yc�A�*

loss�Ք<�O��       �	�|�>Yc�A�*

loss���=��p�       �	��>Yc�A�*

loss�WA=@�q�       �	˹�>Yc�A�*

loss��_=~��       �	�W�>Yc�A�*

loss\�<��h}       �	f��>Yc�A�*

loss��=�r`       �	3��>Yc�A�*

lossa�<4u�       �	�O�>Yc�A�*

loss�2�<k�>�       �	���>Yc�A�*

loss lm<)0��       �	'��>Yc�A�*

loss�Em=Cl�*       �	c*�>Yc�A�*

loss�.{=V4H       �	���>Yc�A�*

lossV�s=��!       �	�b�>Yc�A�*

loss�B=�yO       �	���>Yc�A�*

loss�4�<"[N       �	y��>Yc�A�*

loss`��=6�       �	D��>Yc�A�*

loss�D�=��>�       �	Y6�>Yc�A�*

loss�<=���       �	u��>Yc�A�*

loss��M=;.��       �	��>Yc�A�*

loss	w�<0!��       �	_^�>Yc�A�*

loss
<�=�$ݮ       �	;S�>Yc�A�*

loss���=1�R       �	l��>Yc�A�*

loss��=KG�       �	��>Yc�A�*

loss7�P=��@�       �	�@�>Yc�A�*

loss4
=�A�       �	��>Yc�A�*

loss}Ռ<Yǟ�       �	Yi�>Yc�A�*

lossmR�=��T       �	��>Yc�A�*

loss�)B=F���       �	#��>Yc�A�*

loss �=i[��       �	{3�>Yc�A�*

loss�"s=]��I       �	���>Yc�A�*

loss�=��q,       �	�c�>Yc�A�*

lossؗ�<�?C       �	��>Yc�A�*

loss��=11       �	?��>Yc�A�*

loss&��=Ppm       �	�W�>Yc�A�*

lossx?�<c�	�       �	���>Yc�A�*

loss��=�;        �	נ�>Yc�A�*

loss�=H���       �	8��>Yc�A�*

loss
�Q<�m�V       �	Jb�>Yc�A�*

lossV%p=X%��       �	��>Yc�A�*

loss�@x='ԙ       �	ۤ�>Yc�A�*

loss�C�=�tr�       �	�T�>Yc�A�*

loss��=��C~       �	���>Yc�A�*

loss��=�s!       �	ʧ�>Yc�A�*

loss�m�<P}Tu       �	�I�>Yc�A�*

loss��n<��f=       �	���>Yc�A�*

loss�v�=�i�"       �	ߦ�>Yc�A�*

loss�Y=T/�       �	�}�>Yc�A�*

loss��0<뙮�       �	&�>Yc�A�*

loss�Q�<c�mF       �	6��>Yc�A�*

loss���=���>       �		m�>Yc�A�*

loss�t<ӻo�       �	��>Yc�A�*

loss;->Kʿ�       �	K��>Yc�A�*

loss��{=�L�       �	5}�>Yc�A�*

lossvI=��2�       �	z�>Yc�A�*

loss=%0=�h�=       �	��>Yc�A�*

loss̠�=��a       �	�`�>Yc�A�*

loss�d�<�'��       �	���>Yc�A�*

lossdǅ=y+��       �	���>Yc�A�*

lossI>L=���Y       �	�N�>Yc�A�*

loss�_m=�9��       �	��>Yc�A�*

loss�V<�d�       �	ߣ�>Yc�A�*

loss�0r=9Rg�       �	-C�>Yc�A�*

loss�-<q�       �	8��>Yc�A�*

lossӤZ=�b��       �	1|�>Yc�A�*

loss�=�v       �	|+�>Yc�A�*

loss�*=�^��       �	���>Yc�A�*

lossN�#=�ɀ.       �	�o�>Yc�A�*

lossJ=G�8       �	�>Yc�A�*

loss���<c�)Y       �	W��>Yc�A�*

loss���=�d       �	UO�>Yc�A�*

lossu�=3:5       �	���>Yc�A�*

loss�4<��i       �	]1�>Yc�A�*

loss��j=�t44       �	q��>Yc�A�*

lossG�=�R��       �	��>Yc�A�*

lossŨ�<�l�-       �	��>Yc�A�*

lossŶ>��_d       �	��>Yc�A�*

lossqSB=��       �	�Z�>Yc�A�*

loss}Q�<]08�       �	� �>Yc�A�*

loss�5,=J�j       �	��>Yc�A�*

loss4o>���G       �	G;�>Yc�A�*

loss�{�=�"��       �	�)�>Yc�A�*

loss�?*= 1�       �	1	�>Yc�A�*

loss��=�㱨       �	m��>Yc�A�*

lossi$�=�� �       �	I�>Yc�A�*

loss�W�<Y7y       �	���>Yc�A�*

loss�=.�O�       �	1��>Yc�A�*

loss�$�<���       �	1�>Yc�A�*

loss�ڕ<�;�       �	(��>Yc�A�*

loss>Ջ=L��{       �	}��>Yc�A�*

loss\�j=�p��       �	�/�>Yc�A�*

loss��,=�vB       �	c��>Yc�A�*

loss��<�~c�       �	�t�>Yc�A�*

loss8#=4Qq�       �	A�>Yc�A�*

loss�yn;,]V       �	��>Yc�A�*

loss���<r���       �	�@�>Yc�A�*

loss���<�Vp�       �	O��>Yc�A�*

lossZ��=r�a�       �	��>Yc�A�*

loss_��<�v�       �	�B ?Yc�A�*

loss�C	>��f�       �	�� ?Yc�A�*

loss(?K=�长       �	��?Yc�A�*

loss̠�<#A�       �	��?Yc�A�*

loss|�=u$�       �	�:?Yc�A�*

loss�Ԅ='��       �	M�?Yc�A�*

loss�H�=���       �	�?Yc�A�*

lossr��<+#       �	g(?Yc�A�*

loss��<�S�       �	��?Yc�A�*

loss�qG=�q��       �	�]?Yc�A�*

loss.cD=p8}�       �	+�?Yc�A�*

loss ��=:�i�       �	r�?Yc�A�*

loss�.\<�p�       �	5E?Yc�A�*

lossH;=3��       �	7	?Yc�A�*

loss�H=���7       �	*�	?Yc�A�*

loss��=*�6�       �	5]
?Yc�A�*

loss��#=�+�        �	W	?Yc�A�*

loss]�=��       �	.�?Yc�A�*

lossz=�8}�       �	IH?Yc�A�*

loss�l�<�#s8       �	"?Yc�A�*

loss�	>Lǯ       �	��?Yc�A�*

loss�>�`4       �	�f?Yc�A�*

losswJs=�l7�       �	�?Yc�A�*

loss֑#>U�O       �	;�?Yc�A�*

loss�s�<���       �	�@?Yc�A�*

loss,V<��       �	��?Yc�A�*

loss��Z=����       �	�|?Yc�A�*

loss%�2=.���       �	�?Yc�A�*

loss�=�gȅ       �	U�?Yc�A�*

lossF�=�nT       �	&p?Yc�A�*

loss/��<vf߶       �	?Yc�A�*

loss�g=
N�       �	̵?Yc�A�*

loss l=�$�       �	�U?Yc�A�*

loss���=<G��       �	��?Yc�A�*

loss$WL=:"��       �	%�?Yc�A�*

loss���=����       �	J�?Yc�A�*

lossl��=ǃ�       �	�X?Yc�A�*

loss=G=8�       �	��?Yc�A�*

loss��O=�c�       �	�?Yc�A�*

loss���<�qM       �	�a?Yc�A�*

loss-�=TR�       �	9
?Yc�A�*

loss��<T�0�       �	y�?Yc�A�*

lossc�=�g��       �	�^?Yc�A�*

lossEt�=���       �	�?Yc�A�*

loss,�=�Ǫ4       �	%�?Yc�A�*

loss4�&=t2�       �	Fz?Yc�A�*

loss�E�<!P��       �	V)?Yc�A�*

loss�&T=���,       �	��?Yc�A�*

loss��f=\��       �	i ?Yc�A�*

loss]Y%=�\f       �	�M!?Yc�A�*

loss ��<K1�       �	'"?Yc�A�*

loss3�<",׳       �	��"?Yc�A�*

loss��>^�D       �	�#?Yc�A�*

loss�}�;L�g�       �	$$?Yc�A�*

lossP��<��g       �	�$?Yc�A�*

loss֐�<�Y�V       �	$�%?Yc�A�*

lossoZ�=h�F�       �	N&?Yc�A�*

loss�Ai<_� �       �	�&?Yc�A�*

loss���<i4�       �	c}'?Yc�A�*

lossZ�=�Ş�       �	�(?Yc�A�*

lossQ�<���       �	ڪ(?Yc�A�*

lossW�2=��H       �	J)?Yc�A�*

lossF��<��9�       �	B�)?Yc�A�*

loss4��=����       �	��*?Yc�A�*

loss��=�ӫ�       �	<+?Yc�A�*

loss���=Yޚ�       �	��+?Yc�A�*

loss�[=���       �	�~,?Yc�A�*

loss}�w=W%G       �	T-?Yc�A�*

loss�-�=�`<�       �	�-?Yc�A�*

lossa��=��-�       �	T.?Yc�A�*

loss��m=\{��       �	G/?Yc�A�*

loss�a�=�V˩       �	��/?Yc�A�*

loss��>^��       �	�]0?Yc�A�*

loss���<��	�       �	{�0?Yc�A�*

loss"�<��a       �	֍1?Yc�A�*

loss���=����       �	t$2?Yc�A�*

lossT8=����       �	r�2?Yc�A�*

lossE��<e�k�       �	 T3?Yc�A�*

loss�0=�`c9       �	�3?Yc�A�*

loss-�=�Do       �	�4?Yc�A�*

loss���=�;75       �	�5?Yc�A�*

loss)�H=�T,�       �	�5?Yc�A�*

loss�=��pt       �	�O6?Yc�A�*

loss��V=����       �	��6?Yc�A�*

loss��=�       �	}7?Yc�A�*

lossH�=��!       �	8?Yc�A�*

loss��(<��X       �	!�8?Yc�A�*

loss��=�       �	�B9?Yc�A�*

lossQY~=l���       �	��9?Yc�A�*

lossC#�<'?��       �	�|:?Yc�A�*

loss��=x��S       �	Q;?Yc�A�*

loss��=x�Q�       �	I�;?Yc�A�*

loss�ע<�
�}       �	�[<?Yc�A�*

loss���<��g�       �	a2=?Yc�A�*

lossSG=���0       �	�=?Yc�A�*

losssc�<D��       �	a�>?Yc�A�*

lossz�Z<X[��       �	�<??Yc�A�*

lossE��=,E "       �	��??Yc�A�*

loss�s�=�4O       �	@?Yc�A�*

loss�4=Ӊ�       �	�A?Yc�A�*

loss�|>}n��       �	(�A?Yc�A�*

loss�<Ed��       �	�fB?Yc�A�*

loss��<��`M       �	�C?Yc�A�*

loss���=����       �	0�C?Yc�A�*

loss&�=�&��       �	C;D?Yc�A�*

loss�#=U�m�       �	��D?Yc�A�*

loss%��=b�       �	��E?Yc�A�*

loss$҈=�F       �	Y6F?Yc�A�*

lossY�<�E��       �	��F?Yc�A�*

lossvW�<��       �	�nG?Yc�A�*

lossS��=�=       �	(H?Yc�A�*

loss�<�==�       �	ūH?Yc�A�*

loss�[=���4       �	|DI?Yc�A�*

lossۘM=�~�       �	��I?Yc�A�*

loss��=8
/p       �	J�J?Yc�A�*

loss�5�=��Ś       �	8K?Yc�A�*

loss?�e=(@{t       �	D�K?Yc�A�*

loss��\<�0Q       �	`sL?Yc�A�*

loss?��<��}�       �	�
M?Yc�A�*

lossֱQ<�8�       �	�M?Yc�A�*

loss��=��oj       �	�cN?Yc�A�*

loss8"=}�np       �	6O?Yc�A�*

loss�|<�d       �	J�O?Yc�A�*

loss�wv=yQ��       �	VP?Yc�A�*

loss��G=%?0       �	_�P?Yc�A�*

loss��=L�i�       �	؀Q?Yc�A�*

lossq��;w       �	R?Yc�A�*

loss��>&��,       �	��R?Yc�A�*

loss�g[=�L�       �	[T?Yc�A�*

loss$�8=i���       �	�T?Yc�A�*

loss#�>��       �	��U?Yc�A�*

lossZ.<�	:       �	�)V?Yc�A�*

loss&y�<�s       �	�V?Yc�A�*

loss�<��B^       �	�[W?Yc�A�*

lossEfp=�dH|       �	()X?Yc�A�*

lossl.�<v�R�       �	�Y?Yc�A�*

loss�r[=(Ѻ       �	��Y?Yc�A�*

loss���=L.x�       �	p�Z?Yc�A�*

loss͆+=�Џ       �	�:[?Yc�A�*

losss�c=~�,�       �	e�[?Yc�A�*

loss��r=g'       �	��\?Yc�A�*

loss�n�=��r�       �	�]?Yc�A�*

loss�!�<D#pf       �	�T^?Yc�A�*

loss�Ѕ=�
��       �	� _?Yc�A�*

loss��H=# �       �	�_?Yc�A�*

loss��<J9|       �	Ll`?Yc�A�*

loss��<�PM�       �	�Ea?Yc�A�*

loss3a�<�       �	b?Yc�A�*

loss*^�="�       �	~�b?Yc�A�*

loss�э<��V       �	�c?Yc�A�*

loss3*?<Y3E       �	�od?Yc�A�*

lossf�=VA�#       �	�1e?Yc�A�*

loss,(=R��       �	%f?Yc�A�*

loss��=�l��       �	<�f?Yc�A�*

loss�=�N��       �	`�g?Yc�A�*

loss��.=�:{+       �	.Uh?Yc�A�*

losszI�=賊�       �	�,i?Yc�A�*

loss:�=%�MB       �	4�i?Yc�A�*

lossM�;=��(       �	��j?Yc�A�*

loss�6=b�       �	'lk?Yc�A�*

lossŨ=�Z�-       �	wl?Yc�A�*

loss�jq=X���       �	Bm?Yc�A�*

loss/�u<�ɵ/       �	��m?Yc�A�*

losse�M=-Ϥ�       �	��n?Yc�A�*

lossr=�]�       �	po?Yc�A�*

loss<t=�H       �	�'p?Yc�A�*

loss�
�=B���       �	�p?Yc�A�*

loss�^R<[~_       �	E�q?Yc�A�*

lossԣ<�kɼ       �	�)r?Yc�A�*

loss��w<��<�       �	��r?Yc�A�*

loss<u�=�Ԟ�       �	��s?Yc�A�*

loss�+�<t�e}       �	�8t?Yc�A�*

losse�=:��       �	��t?Yc�A�*

loss��k=d��       �	�u?Yc�A�*

loss� =V<QA       �	�2v?Yc�A�*

loss�~<�(��       �	�v?Yc�A�*

lossC�#=j�f       �	�w?Yc�A�*

losså=�od5       �	�Qx?Yc�A�*

loss�n=<�/�       �	��x?Yc�A�*

loss�*�;g�       �	؃y?Yc�A�*

lossO��<� �8       �	qz?Yc�A�*

loss�^]=��0�       �	��z?Yc�A�*

loss{��<]��       �	�V{?Yc�A�*

loss=��<�1Ӣ       �	<|?Yc�A�*

lossC4 =Ds1H       �	w�|?Yc�A�*

loss-A�=)�9h       �	Oy}?Yc�A�*

lossHb[<{c��       �	Y~?Yc�A�*

loss���<N��       �	��~?Yc�A�*

loss��=�l,�       �	�G?Yc�A�*

loss��y<H3�h       �	*�?Yc�A�*

loss_�:��       �	{�?Yc�A�*

lossjW<�	y~       �	L�?Yc�A�*

loss�r$=�i       �	���?Yc�A�*

loss��<���       �	�e�?Yc�A�*

lossi��<0�       �	~��?Yc�A�*

loss�X~;.��W       �	S��?Yc�A�*

lossi-=�       �	g,�?Yc�A�*

loss)tP;��ā       �	���?Yc�A�*

loss�B:���       �	���?Yc�A�*

lossI �:�x�       �	)�?Yc�A�*

loss]��<1��       �	��?Yc�A�*

loss_4=��uu       �	+n�?Yc�A�*

loss}a=�uW)       �	�?Yc�A�*

loss�6;��L       �	���?Yc�A�*

lossM3<	T��       �	�S�?Yc�A�*

loss�ހ>�3P�       �	t�?Yc�A�*

loss8"�;��K       �	"��?Yc�A�*

loss�<�"4�       �	�G�?Yc�A�*

loss�zV=��|�       �	��?Yc�A�*

loss��=�;=�       �	��?Yc�A�*

lossoqk=�][�       �	�-�?Yc�A�*

lossm��<r>�O       �	[ҍ?Yc�A�*

loss?�z=��I�       �	nk�?Yc�A�*

loss���=�C�I       �	=E�?Yc�A�*

loss���=� �d       �	�
�?Yc�A�*

loss�\�=��C�       �	$�?Yc�A�*

loss��=��{       �	�ݑ?Yc�A�*

loss��w=�<=�       �	ۤ�?Yc�A�*

loss;H=���n       �	��?Yc�A�*

loss�=4��       �	峔?Yc�A�*

lossyw=O�       �	P��?Yc�A�*

loss�[.=�J|�       �	N�?Yc�A�*

loss�=~[*S       �	��?Yc�A�*

lossQ2=��98       �	�ė?Yc�A�*

lossޮ=}�W       �	c��?Yc�A�*

loss��=0���       �	�~�?Yc�A�*

loss�r#<Pty�       �	7O�?Yc�A�*

loss!��<Ë�h       �	.�?Yc�A�*

lossIT�=ٶ�       �	,כ?Yc�A�*

lossr�X<@�v�       �	��?Yc�A�*

loss\��;����       �	c��?Yc�A�*

lossج;l ��       �	�\�?Yc�A�*

loss�=/K�)       �	��?Yc�A�*

lossJ'm<���       �	���?Yc�A�*

loss��0=�p�       �	�E�?Yc�A�*

lossݽ�=�9>L       �	�ݠ?Yc�A�*

loss��<�ل�       �	'��?Yc�A�*

loss� �=D��       �	!�?Yc�A�*

losscs�=�4�       �	�Ţ?Yc�A�*

loss/<W<ToD�       �	.p�?Yc�A�*

loss.p&=3��       �	H�?Yc�A�*

lossʡ=i�       �	���?Yc�A�*

loss$dH<��`�       �	�Y�?Yc�A�*

lossr�=�ҙ!       �	��?Yc�A�*

loss�k=ґ��       �	���?Yc�A�*

loss��=M"г       �	=D�?Yc�A�*

loss�;�&1       �	��?Yc�A�*

lossEw=��59       �	��?Yc�A�*

loss�E<�w;       �	�*�?Yc�A�*

loss�,g=�Q�7       �	��?Yc�A�*

loss*��<�HfH       �	�~�?Yc�A�*

lossD)�<�߬       �	Y�?Yc�A�*

loss�T�=���       �	���?Yc�A�*

loss�:<�       �	�>�?Yc�A�*

lossZ�<6z{q       �	$Ҭ?Yc�A�*

loss�w�;�7�       �	.p�?Yc�A�*

loss2�G=��n�       �	%�?Yc�A�*

loss8=���k       �	��?Yc�A�*

lossO�=�+]�       �	*T�?Yc�A�*

loss2�=���       �	0��?Yc�A�*

loss�Ŭ=U`E�       �	��?Yc�A�*

loss�a�=1��
       �	�F�?Yc�A�*

lossݡ�<O���       �	���?Yc�A�*

lossLe<�&�       �	���?Yc�A�*

loss��p=����       �	,H�?Yc�A�*

loss���=�Y0�       �	���?Yc�A�*

loss�r=B�Z&       �	���?Yc�A�*

loss�@v<��1x       �	�5�?Yc�A�*

lossSW�=X+e4       �	Q��?Yc�A�*

loss��=�ڗR       �	u��?Yc�A�*

loss_~0=�z4       �	9�?Yc�A�*

loss�P7=o2��       �	���?Yc�A�*

loss��<�u       �	*��?Yc�A�*

loss���<� �       �	�3�?Yc�A�*

loss�?�<~��|       �	���?Yc�A�*

loss�y =�%�       �	���?Yc�A�*

lossQ��=�T�       �	:��?Yc�A�*

loss�p>=ʓ��       �	=_�?Yc�A�*

loss�~/==a=       �	��?Yc�A�*

loss\+<��"v       �	L��?Yc�A�*

loss{)=U���       �	�T�?Yc�A�*

loss��=�W�       �	w��?Yc�A�*

loss�<R<l>R       �	;��?Yc�A�*

loss�{�<oX/       �	q��?Yc�A�*

loss{	�<{�8�       �	�P�?Yc�A�*

loss"��=[��       �	��?Yc�A�*

loss�)=�a��       �	� �?Yc�A�*

loss�<3�S       �	u��?Yc�A�*

loss�#�<-z�       �	�{�?Yc�A�*

loss(�<��f)       �	�N�?Yc�A�*

loss��=Z�&�       �	�4�?Yc�A�*

lossd�<s`�       �	HP�?Yc�A�*

lossL��=����       �	�&�?Yc�A�*

loss��<K:�       �	��?Yc�A�*

loss��s=O�q\       �	���?Yc�A�*

loss�.:>Ki��       �	v��?Yc�A�*

lossxc�=!z�;       �	�S�?Yc�A�*

lossL!�<C�       �	�Z�?Yc�A�*

loss�K=с�?       �	���?Yc�A�*

loss��<<j�A-       �	���?Yc�A�*

lossa��=m�?�       �	��?Yc�A�*

lossW`�=�M�       �	{0�?Yc�A�*

loss2�F=�F��       �	���?Yc�A�*

loss�=_��@       �	���?Yc�A�*

loss(d =)�A       �	�4�?Yc�A�*

loss��<�+I�       �	���?Yc�A�*

loss&N�;LeG�       �	Ԛ�?Yc�A�*

loss/��<ǥ}�       �	9F�?Yc�A�*

loss��=˂0@       �	���?Yc�A�*

loss�ύ<U@�       �	���?Yc�A�*

lossÙZ>��t\       �	�-�?Yc�A�*

lossdO�<	��       �	`��?Yc�A�*

loss��p<�d��       �	�n�?Yc�A�*

loss�X;D�I       �	��?Yc�A�*

loss���;U^�       �	���?Yc�A�*

lossI��<�p�       �	 q�?Yc�A�*

loss�F8=܍,�       �	�G�?Yc�A�*

loss���=��m       �	���?Yc�A�*

lossݠ�=B���       �	ڐ�?Yc�A�*

loss���<�Y^H       �	�o�?Yc�A�*

loss,�=X?u�       �	R
�?Yc�A�*

lossA�C=�p��       �	`��?Yc�A�*

loss�R�=��B�       �	$F�?Yc�A�*

loss}�=D�ۮ       �	��?Yc�A�*

lossJ�=��       �	��?Yc�A�*

loss_.�=�?�&       �	�0�?Yc�A�*

lossI�V=�Ӭ�       �	��?Yc�A�*

loss��Z={\�V       �	�|�?Yc�A�*

lossd<�<���C       �	`!�?Yc�A�*

loss|�=1��       �	��?Yc�A�*

lossR�`=��ٝ       �	�x�?Yc�A�*

loss=��       �	;�?Yc�A�*

loss���<
ʊO       �	��?Yc�A�*

loss1(>=�K��       �	�R�?Yc�A�*

lossM��<���!       �	���?Yc�A�*

loss��u<��1       �	9��?Yc�A�*

loss�]A=�أ�       �	�6�?Yc�A�*

lossj��=p�+�       �	���?Yc�A�*

loss��j=}���       �	hy�?Yc�A�*

lossh�5=(O�       �	��?Yc�A�*

loss�;Y<�t�V       �	��?Yc�A�*

loss��<>�]�       �	mU @Yc�A�*

loss2��<WV"       �	�� @Yc�A�*

loss*2�=�`�[       �	��@Yc�A�*

loss���<큈�       �	�%@Yc�A�*

loss�<b< ��w       �	��@Yc�A�*

loss��m<5KŎ       �	w@Yc�A�*

loss3�^=r� Q       �	�@Yc�A�*

loss\�w=;L��       �	��@Yc�A�*

loss��T=7�>;       �	Dm@Yc�A�*

lossת?=����       �	�@Yc�A�*

loss�1=�DJ�       �	J�@Yc�A�*

loss�z;<�1�       �	|_@Yc�A�*

loss���=���       �	?�@Yc�A�*

loss��)=���       �	��@Yc�A�*

loss���=�%g       �	�6	@Yc�A�*

loss�
O=����       �	��	@Yc�A�*

loss�ߘ<\��       �	�o
@Yc�A�*

loss�U<5��       �	�q@Yc�A�*

loss���<2�       �	8@Yc�A�*

lossI6�<��"       �	��@Yc�A�*

lossZ�=]�Ì       �	eP@Yc�A�*

loss���=,�<�       �	$�@Yc�A�*

loss��<%��_       �	*�@Yc�A�*

losse�S<�~,�       �	�6@Yc�A�*

lossż�=3`X       �	��@Yc�A�*

lossD	�<2���       �	��@Yc�A�*

loss�0�=�X�       �	l!@Yc�A�*

lossҚ0=��B�       �	�	@Yc�A�*

loss�L�<�1w�       �	�@Yc�A�*

loss��<-1l7       �	<J@Yc�A�*

lossrR�<`�J@       �	��@Yc�A�*

loss�ʮ<�@�'       �	��@Yc�A�*

loss�^=+ �J       �	�;@Yc�A�*

lossw׬=�P`       �	O�@Yc�A�*

loss�G�<ĈV�       �	؂@Yc�A�*

loss���<�ڍ       �	X@Yc�A�*

lossM��<p���       �	��@Yc�A�*

lossm�=���4       �	�x@Yc�A�*

losshF�=rYM�       �	r@Yc�A�*

loss�pI=G[��       �	+@Yc�A�*

loss��P<^��       �	��@Yc�A�*

loss�z-=1�       �	��@Yc�A�*

loss���=JP       �	N�@Yc�A�*

loss���=R���       �	tD@Yc�A�*

loss,ʰ=��[       �	K�@Yc�A�*

lossh�=��U�       �	�@Yc�A�*

losst�8= 6�       �	�+@Yc�A�*

loss⇙=�__       �	��@Yc�A�*

loss�k�<j&       �	Ή @Yc�A�*

loss Kn=�o�       �	^+!@Yc�A�*

loss�{�=>��       �	��!@Yc�A�*

lossҔ<b��z       �	�"@Yc�A�*

loss�}�=��m       �	�$#@Yc�A�*

lossF|P=��x       �	1�#@Yc�A�*

loss\�=K���       �	�m$@Yc�A�*

lossl�<��-       �	q;%@Yc�A�*

loss�\�;>R�l       �	��%@Yc�A�*

loss�Mu=<#ݩ       �	�w&@Yc�A�*

loss�d=cO�r       �	�'@Yc�A�*

loss��=�)0�       �	֪'@Yc�A�*

lossq̉=�       �	�F(@Yc�A�*

lossF�<[e1       �	f�(@Yc�A�*

lossCɊ=��!       �	V�)@Yc�A�*

lossnK8<�$�{       �	=*@Yc�A�*

losscF<S�@�       �	��*@Yc�A�*

lossl��<���h       �	x{+@Yc�A�*

loss�K =��,0       �	�,@Yc�A�*

loss�TL=��Xr       �	C�,@Yc�A�*

loss,��=G.�e       �	�~-@Yc�A�*

loss�ޔ=k��       �	.@Yc�A�*

lossHgi=a>��       �	�.@Yc�A�*

loss��4<P��       �	�L/@Yc�A�*

loss�d=�@jZ       �	��/@Yc�A�*

loss�=�c�       �	�0@Yc�A�*

loss�ƽ=��{       �	�,1@Yc�A�*

loss]!l=Y��       �	��1@Yc�A�*

lossE�o=�`�       �	�n2@Yc�A�*

loss� @<�Y��       �	�I3@Yc�A�*

losszӊ<y�N�       �	 �3@Yc�A�*

lossxz<�^p�       �	��4@Yc�A�*

lossw��=�Y       �	�5@Yc�A�*

loss|��<J���       �	_�5@Yc�A�*

loss�y�<]�       �	z�6@Yc�A�*

loss��F=/%       �	�#7@Yc�A�*

lossQ�=Lw�(       �	�7@Yc�A�*

loss�=.�       �	�W8@Yc�A�*

lossDTf=(՛       �	��8@Yc�A�*

loss���<��^!       �	4�9@Yc�A�*

loss=��<)'\>       �	� :@Yc�A�*

lossw$�<�q9�       �	�:@Yc�A�*

lossC.<x�)�       �	��;@Yc�A�*

loss�-b=tU%       �	)Y<@Yc�A�*

loss{�<4k�       �	��<@Yc�A�*

loss�k=\��       �	H�=@Yc�A�*

lossl�=Iִ�       �	�&>@Yc�A�*

loss��<����       �	��>@Yc�A�*

loss_ <�5d#       �	�c?@Yc�A�*

loss�"N=
�Μ       �	W@@Yc�A�*

loss�к<����       �	�@@Yc�A�*

loss�}�;݊U�       �	�>A@Yc�A�*

loss�E�<���N       �	V�A@Yc�A�*

loss��<��KV       �	��B@Yc�A�*

loss/=�C\�       �	VIC@Yc�A�*

loss��{<��o|       �	�D@Yc�A�*

loss��<	W�       �	1DE@Yc�A�*

lossl͝<q#��       �	d�E@Yc�A�*

loss�A=X��h       �	G@Yc�A�*

loss�H=�8Л       �	1�G@Yc�A�*

loss��$=oߨ=       �	�LH@Yc�A�*

loss�=<���J       �	s-I@Yc�A�*

loss�2�;ӰrA       �	��I@Yc�A�*

loss���<U��       �	�fJ@Yc�A�*

loss/�<,�3�       �	i�J@Yc�A�*

lossH��=<g�E       �	N�K@Yc�A�*

loss$��=�H��       �	�FL@Yc�A�*

loss*=�n`�       �	:�L@Yc�A�*

lossN�9=��9       �	��M@Yc�A�*

loss��{<
^�y       �	�,N@Yc�A�*

loss��<ɟX�       �	�JO@Yc�A�*

loss�PR=3i��       �	!�O@Yc�A�*

loss��Y<��*�       �	1�P@Yc�A�*

loss=̉<���       �	�QQ@Yc�A�*

loss!��=�)>2       �	,�Q@Yc�A�*

loss�<�<�`��       �	��R@Yc�A�*

loss�>�<���       �	z7S@Yc�A�*

loss7�=��4       �	��S@Yc�A�*

loss��(=D��x       �	�T@Yc�A�*

loss���<T
��       �	:U@Yc�A�*

lossFX.=����       �	v�U@Yc�A�*

loss��<��X�       �	Z�V@Yc�A�*

loss��=�R�       �	�:W@Yc�A�*

loss<?�=���       �	w�W@Yc�A�*

loss�O=�P�       �	�yX@Yc�A�*

loss#�=��       �	�Y@Yc�A�*

loss>׿��       �	��Y@Yc�A�*

loss�B>l�       �	�gZ@Yc�A�*

loss���<X0��       �	�[@Yc�A�*

loss��|=��W       �	��[@Yc�A�*

loss��?=���       �	�R\@Yc�A�*

lossf�<��6       �	��\@Yc�A�*

loss�ۿ;>O�       �	~�]@Yc�A�*

lossL)�<6��a       �	�+^@Yc�A�*

loss���<���_       �	a�^@Yc�A�*

loss�ܠ=����       �	Bz_@Yc�A�*

loss�v=rӈ&       �	�`@Yc�A�*

lossQ��=7�Η       �	��`@Yc�A�*

loss�<C�(       �	�a@Yc�A�*

losskN=����       �	�b@Yc�A�*

loss���=�On       �	��b@Yc�A�*

loss6y+=l��       �	aSc@Yc�A�*

lossdO_=���       �	��c@Yc�A�*

loss��=���       �	D�d@Yc�A�*

loss�(�=�y       �	�"e@Yc�A�*

loss��S=�{�       �	��e@Yc�A�*

loss���=�)��       �	<�g@Yc�A�*

loss*�L<ڳ�:       �	.h@Yc�A�*

loss "�;�=��       �	i�h@Yc�A�*

loss��x=����       �	�ii@Yc�A�*

lossD��<���S       �	!j@Yc�A�*

loss��=�]�A       �	�j@Yc�A�*

lossD�9=�q�       �	�k@Yc�A�*

loss:�<����       �	�Il@Yc�A�*

loss�R�=�8I       �	��l@Yc�A�*

loss\a=�:�4       �	�xm@Yc�A�*

loss�9=H03       �	J)n@Yc�A�*

lossAjt<>��@       �	�n@Yc�A�*

loss���<mޥ       �	B�o@Yc�A�*

loss�� =2��       �	j4p@Yc�A�*

lossJ=3� W       �	�q@Yc�A�*

loss��<=�v       �	N�q@Yc�A�*

loss*��;Ê<{       �	�Pr@Yc�A�*

loss#V�<vH��       �	:�r@Yc�A�*

loss L�<����       �	4�s@Yc�A�*

lossO=£Wt       �	�t@Yc�A�*

loss�I=z>Y�       �	�t@Yc�A�*

loss��6=3��       �	�Tu@Yc�A�*

loss�+x=���p       �	 v@Yc�A�*

loss���<�PZ�       �	��v@Yc�A�*

loss��D=U��A       �	|Hw@Yc�A�*

loss�lL=��X�       �	��w@Yc�A�*

loss:p=&�h�       �	e�x@Yc�A�*

loss��=c���       �	"y@Yc�A�*

loss� c=}�U=       �	o�y@Yc�A�*

loss�� =6X       �	�Uz@Yc�A�*

loss��=�ma       �	�"{@Yc�A�*

loss�{�=T��       �	��{@Yc�A�*

loss�=�ȳ�       �	L�}@Yc�A�*

loss;��<f�^Z       �	�)~@Yc�A�*

loss(�m=�~*       �	n�~@Yc�A�*

loss	m =j`��       �	��@Yc�A�*

loss�w�=�{��       �	�)�@Yc�A�*

loss\2=ªXW       �	��@Yc�A�*

lossJ=<3�       �	O��@Yc�A�*

loss�P<� �,       �	�/�@Yc�A�*

loss$s)=�ů�       �	�ǂ@Yc�A�*

loss#"m=�Z)       �	Zf�@Yc�A�*

loss��<���       �	��@Yc�A�*

loss��<o4       �	���@Yc�A�*

loss�6H=��	�       �	>]�@Yc�A�*

loss/�<]��       �	���@Yc�A�*

loss��=�#��       �	���@Yc�A�*

lossj��=�[��       �	�$�@Yc�A�*

loss{�p=�� �       �	�@Yc�A�*

loss�V�<*X��       �	b��@Yc�A�*

lossJ}�<���q       �	�7�@Yc�A�*

lossw\�<���p       �	�܉@Yc�A�*

loss���;����       �	,~�@Yc�A�*

loss�^�=+I�y       �	��@Yc�A�*

loss�ō<�x�}       �	���@Yc�A�*

loss�d�;+�	�       �	@M�@Yc�A�*

loss��>=Z��       �	K�@Yc�A�*

losss�|=I�eg       �	���@Yc�A�*

loss��=�4�       �	Y�@Yc�A�*

loss���=*�       �	��@Yc�A�*

loss/.�=-��z       �	���@Yc�A�*

loss�=l��>       �	I�@Yc�A�*

loss-��<�1�9       �	C�@Yc�A�*

loss	v=���       �	R~�@Yc�A�*

loss��y;;r�       �	H�@Yc�A�*

loss7p�=�)��       �	 ��@Yc�A�*

loss|F;=��        �	nL�@Yc�A�*

loss`8�=�7��       �	�@Yc�A�*

loss���<Ai�       �	ٓ�@Yc�A�*

loss�w=2~�       �	{-�@Yc�A�*

loss��<��=�       �	�@Yc�A�*

loss臫=��N�       �	���@Yc�A�*

lossŹj=����       �	�9�@Yc�A�*

loss�=��       �	sח@Yc�A�*

loss\�<�ƚ�       �	yv�@Yc�A�*

loss�E�;�b��       �	�j�@Yc�A�*

lossVC	=?�/�       �	�@Yc�A�*

loss�ԭ=(N��       �	���@Yc�A�*

lossF#<�z�       �	�K�@Yc�A�*

lossA#<���g       �	��@Yc�A�*

loss|f=�c�       �	$�@Yc�A�*

loss�9m=����       �	���@Yc�A�*

lossK�<@�       �	�B�@Yc�A�*

loss�x=v��_       �	�ݞ@Yc�A�*

loss�I�=6)��       �	7��@Yc�A�*

loss�2<F!p       �	�,�@Yc�A�*

loss��T=>�b�       �	���@Yc�A�*

loss@ܥ=�s       �	ɏ�@Yc�A�*

lossj�<�i�       �	S%�@Yc�A�*

lossv}o<SF�       �	�Ѣ@Yc�A�*

lossT�[=�ḏ       �	~t�@Yc�A�*

loss��>!��u       �	^�@Yc�A�*

loss#�:=� i       �	�¤@Yc�A�*

lossυ�<��Qz       �	�^�@Yc�A�*

loss}��;�aZ�       �	?�@Yc�A�*

loss��<�ȏ       �	���@Yc�A�*

loss1L�=PC�x       �	E�@Yc�A�*

loss_ʁ=��z2       �	��@Yc�A�*

loss$�F=\T
�       �	J|�@Yc�A�*

loss�U=���       �	�%�@Yc�A�*

loss�s	=�v��       �	"��@Yc�A�*

lossX��;Z��+       �	�]�@Yc�A�*

loss�ɞ<o+�        �	���@Yc�A�*

loss�g==���b       �	��@Yc�A�*

loss�̮<�>"K       �	���@Yc�A�*

loss!�;=�	#       �	�@Yc�A�*

loss�k�=0"��       �	k��@Yc�A�*

loss*�=��Ӊ       �	�O�@Yc�A�*

lossr\�;��p�       �	 �@Yc�A�*

lossC=�=G"9�       �	5��@Yc�A�*

lossf�1=�N�Y       �	�İ@Yc�A�*

loss{�S=Y��7       �	�`�@Yc�A�*

loss,��;��e       �	�@Yc�A�*

loss%�u<w��J       �	맲@Yc�A�*

loss�<�DB
       �	�P�@Yc�A�*

loss�S=�و�       �	��@Yc�A�*

loss�N�<3��>       �	xҴ@Yc�A�*

lossk�<��qo       �	}�@Yc�A�*

loss%j|<1�QM       �	�^�@Yc�A�*

loss �=�z��       �	��@Yc�A�*

lossSpm=V�.i       �	%��@Yc�A�*

losseL=���c       �	��@Yc�A�*

loss�|=�~e       �	�K�@Yc�A�*

loss�7="�X�       �	�@Yc�A�*

loss�{�<�M_       �	X��@Yc�A�*

lossFD�=�8ھ       �	P4�@Yc�A�*

loss<L=~}�       �	Aӻ@Yc�A�*

lossS=��}�       �	ҋ�@Yc�A�*

loss�L�=��f       �	�,�@Yc�A�*

loss�2=��c�       �	6Ƚ@Yc�A�*

loss�G�<R��)       �	an�@Yc�A�*

lossč=);��       �	��@Yc�A�*

lossXh�<��X	       �	���@Yc�A�*

lossi�<1�ɵ       �	�T�@Yc�A�*

loss�=m~��       �	v��@Yc�A�*

loss�@N<:{       �	���@Yc�A�*

loss��=R���       �	]2�@Yc�A�*

lossp��=ޚ�       �	���@Yc�A�*

lossR��=�gyR       �	$d�@Yc�A�*

loss�\=*
��       �	�.�@Yc�A�*

loss F =�2]�       �	���@Yc�A�*

loss���=�4�]       �	[^�@Yc�A�*

loss̻�=EZ��       �	���@Yc�A�*

loss��=�˦4       �	w��@Yc�A�*

loss��<!��:       �	~:�@Yc�A�*

loss�uz<]�BH       �	��@Yc�A�*

loss鄕;U)^b       �	���@Yc�A�*

loss�Fn=ؖ��       �	�;�@Yc�A�*

loss�hz=f��H       �	p��@Yc�A�*

lossf�w=~!z       �	ۈ�@Yc�A�*

loss�ۅ</��&       �	#�@Yc�A�*

loss]�<=�Ju
       �	���@Yc�A�*

lossԒ�<�=       �	�U�@Yc�A�*

lossjۈ<����       �	#�@Yc�A�*

loss&�4=q��       �	���@Yc�A�*

lossb�=��Yl       �	zq�@Yc�A�*

loss�y#=M��       �	��@Yc�A�*

loss2�=�Z{n       �	A��@Yc�A�*

lossL<>r�       �	�P�@Yc�A�*

loss8[�<?f�       �	:��@Yc�A�*

loss�g�<�Q�Y       �	x~�@Yc�A�*

loss㐮<��Xp       �	��@Yc�A�*

loss�4=?s�       �	���@Yc�A�*

losse�<"h       �	`�@Yc�A�*

loss��=�/:(       �	w��@Yc�A�*

loss��;s_=�       �	h��@Yc�A�*

loss��q=g�0�       �	a7�@Yc�A�*

loss%`<7�L       �	���@Yc�A�*

loss�L�=va��       �	�z�@Yc�A�*

loss���=�q�e       �	Vd�@Yc�A�*

loss�:=H���       �	G8�@Yc�A�*

loss|�!=�I       �	M��@Yc�A�*

lossC�i=�k%�       �	`w�@Yc�A�*

lossd��=~�Oc       �	��@Yc�A�*

loss3��<U2�-       �	>��@Yc�A�*

loss[0z=��O�       �	ML�@Yc�A�*

loss0A=dcb       �	��@Yc�A�*

loss,�=��P       �	���@Yc�A�*

lossf��<�6�>       �	T�@Yc�A�*

loss*&<Գ�	       �	���@Yc�A�*

loss�ҥ="       �	P�@Yc�A�*

loss�o�=O}_(       �	��@Yc�A�*

loss��<�e�       �	���@Yc�A�*

lossd�=��>       �	�Q�@Yc�A�*

loss���;��}G       �	N%�@Yc�A�*

loss�=�oFR       �	���@Yc�A�*

lossܮ/=�r��       �	*T�@Yc�A�*

loss��s=O[�       �	��@Yc�A�*

lossq�<lŢ�       �	���@Yc�A�*

loss�<=���       �	�'�@Yc�A�*

lossV�<OX*       �	���@Yc�A�*

loss��<��i�       �	�X�@Yc�A�*

loss�!=�%ݸ       �	I��@Yc�A�*

loss�=(�_�       �	[��@Yc�A�*

loss:�&<���M       �	 ;�@Yc�A�*

loss�]�<n���       �	@��@Yc�A�*

loss=�=({        �	�x�@Yc�A�*

lossOU�<�Ϟi       �	��@Yc�A�*

loss��<�A|m       �	Ѯ�@Yc�A�*

loss�<�F��       �	�U�@Yc�A�*

lossT�=�g'       �	���@Yc�A�*

loss|Ԩ<�J�       �	���@Yc�A�*

loss�=Pq�       �	5$�@Yc�A�*

loss#�=@���       �	 ��@Yc�A�*

loss�\�=x>�\       �	�}�@Yc�A�*

loss��Z=��97       �	� �@Yc�A�*

lossʌ/<,�       �	���@Yc�A�*

loss!�<&�B       �	���@Yc�A�*

loss�i>d�d       �	��@Yc�A�*

loss|ƾ<�>��       �	Ĵ�@Yc�A�*

loss���<��b       �	L�@Yc�A�*

loss��=�Ӈ#       �	;��@Yc�A�*

lossZ�=^�X�       �	#��@Yc�A�*

loss_��<H��       �	rO�@Yc�A�*

lossCd(<��w       �	���@Yc�A�*

lossa�e=�%��       �	<��@Yc�A�*

loss�R�=�Ռ       �	��@Yc�A�*

loss�ΰ="��%       �	���@Yc�A�*

loss�2�<I��R       �	Xo�@Yc�A�*

loss%eT=�7�-       �	!�@Yc�A�*

lossp�<2��       �	0��@Yc�A�*

lossXx<*��w       �	�H�@Yc�A�*

lossF
h<�Z�|       �	���@Yc�A�*

loss���;7��       �	�x�@Yc�A�*

lossA5�;}��       �	nQ�@Yc�A�*

loss$S�=]c�c       �	u��@Yc�A�*

loss�$�=40V       �	]��@Yc�A�*

lossT�5<M��.       �	��@Yc�A�*

lossCѐ=q�*�       �	5��@Yc�A�*

loss}��<p�I       �	�M�@Yc�A�*

loss-�=#��       �	q�@Yc�A�*

losssS<�<�       �	 ��@Yc�A�*

lossS �=�Mm�       �	7O AYc�A�*

lossV\�<Ⱙ�       �	�� AYc�A�*

loss���<�׽C       �	�AYc�A�*

loss#�g= Z]�       �	�AYc�A�*

loss,��<�*-�       �	��AYc�A�*

lossa��=�@��       �	�YAYc�A�*

loss��K<L�Y�       �	��AYc�A�*

loss�$A=!L;p       �	O�AYc�A�*

loss�c=�yH�       �	$+AYc�A�*

loss1�=N��       �	D�AYc�A�*

loss�z�=�PΞ       �	�dAYc�A�*

loss_"=�@C�       �	"�AYc�A�*

loss�G==�}       �	�AYc�A�*

loss�L�=�D�       �	�.AYc�A�*

loss��B==��       �	7�AYc�A�*

lossM:Q<�&y�       �	�a	AYc�A�*

lossV�b<��g       �	f�	AYc�A�*

loss��=�U��       �	��
AYc�A�*

lossgO�<�`       �	0+AYc�A�*

loss��e=S��@       �	V�AYc�A�*

loss�vj=V�b       �	��AYc�A�*

lossU6= ho       �	w/AYc�A�*

loss�ar;|I�       �	�AYc�A�*

lossf�_<����       �	s�AYc�A�*

lossN]=��B�       �	�9AYc�A�*

loss�	�<Z�.       �	��AYc�A�*

losssJ�=�ݳ�       �	kAYc�A�*

lossV�<�$�       �	AYc�A�*

loss.a�<�{G�       �	��AYc�A�*

loss��<7��       �	�=AYc�A�*

loss��<Щ�       �	��AYc�A�*

loss���<_%?       �	��AYc�A�*

loss���<�I	       �	mAYc�A�*

loss�ͬ=����       �	��AYc�A�*

lossud�=!�u�       �	l]AYc�A�*

loss1�t<ط'O       �	�AYc�A�*

loss��<�/?�       �	[�AYc�A�*

loss�G<��L       �	D2AYc�A�*

loss{=l�g}       �	�AYc�A�*

lossv;=�r�       �	�cAYc�A�*

loss��<j?��       �	"�AYc�A�*

loss(iE= �)�       �	�AYc�A�*

loss�n<;�       �	=�AYc�A�*

loss܇�=d���       �	c�AYc�A�*

loss�W=��[y       �	'OAYc�A�*

loss�?K=2��       �	{�AYc�A�*

lossz��<�n��       �	�CAYc�A�*

loss��<ݚA       �	~�AYc�A�*

loss>6�<O<y       �	�|AYc�A�*

losss>=x��A       �	� AYc�A�*

loss�'�<�%v�       �	Է AYc�A�*

loss� �<⁲�       �	5�!AYc�A�*

loss��;�@�>       �	#1"AYc�A�*

loss�Ե<�ke       �	��"AYc�A�*

lossQ�3=� 6       �	Ag#AYc�A�*

loss��=�؏       �	?�#AYc�A�*

lossɦ�<}I+       �	�$AYc�A�*

loss�b4=%���       �	ND%AYc�A�*

loss�£=P��f       �	"�%AYc�A�*

loss!w�<c�rU       �	�&AYc�A�*

loss4|=��1�       �	\ 'AYc�A�*

loss�a%=U��P       �		�'AYc�A�*

lossX}�;��i�       �	KY(AYc�A�*

lossM'=<tv��       �	��(AYc�A�*

lossmy+<>�q       �	!�)AYc�A�*

loss�q<����       �	**AYc�A�*

loss��<�5E�       �	Ͽ*AYc�A�*

loss�o<bnW       �	�Y+AYc�A�*

loss\�<e˦�       �	��+AYc�A�*

lossvLm<Z�       �	͐,AYc�A�*

loss�r;Y>�       �	C-AYc�A�*

loss;�:c�       �		�-AYc�A�*

loss(ɏ:�Bu/       �	�u.AYc�A�*

lossݯ�;n��       �	�/AYc�A�*

loss��<�l	�       �	r�/AYc�A�*

loss
U�<�HD       �	@0AYc�A�*

loss�*>;�G�       �	��0AYc�A�*

lossĪ�<��|�       �	qu1AYc�A�*

loss>�N:b       �	�02AYc�A�*

loss��L;L�       �	u�2AYc�A�*

loss�.=�,��       �	��3AYc�A�*

loss�j�=B�Q7       �	��4AYc�A�*

loss�7�<��}       �	~�5AYc�A�*

loss.�<��Dp       �	�6AYc�A�*

loss�e=�oX       �	�R7AYc�A�*

lossW��=�a�       �	��8AYc�A�*

lossd�=��*       �	,I9AYc�A�*

loss���<�(�       �	 �9AYc�A�*

lossx�z=2qX�       �	$�:AYc�A�*

loss��q<�A"�       �	�,;AYc�A�*

loss�5z=�2�       �	�;AYc�A�*

loss!��= D�r       �	q>AYc�A�*

loss���<3�+       �	��>AYc�A�*

loss���=e��       �	-Z?AYc�A�*

loss�=�Z�A       �	 @AYc�A�*

lossWP/=��#�       �	"�@AYc�A�*

loss�==��a�       �	`�AAYc�A�*

lossw��<���       �	vSBAYc�A�*

loss7��=���       �	�CAYc�A�*

loss�\;<H�f       �	�CAYc�A�*

loss.�<���p       �	�rDAYc�A�*

loss�==#o�       �	FEAYc�A�*

loss16X<�usy       �	�EAYc�A�*

lossְ<����       �	�MFAYc�A�*

loss���;���       �	q�FAYc�A�*

loss�c�<���*       �	9~GAYc�A�*

lossF�<�u��       �	�1HAYc�A�*

loss-g%=�Ⓔ       �	z�HAYc�A�*

loss#U}=�{��       �	�fIAYc�A�*

loss�H�<��b       �	�JAYc�A�*

loss��=�P       �	ƢJAYc�A�*

loss><=��       �	��KAYc�A�*

loss��;��E9       �	\;LAYc�A�*

loss��<iW��       �	�LAYc�A�*

loss���;�FE       �	�rMAYc�A�*

loss�p=<9��z       �	FNAYc�A�*

lossӐP<qU��       �	ٱNAYc�A�*

loss�Nm=�5B�       �	�QOAYc�A�*

loss̝<B;c�       �	��OAYc�A�*

loss7�;!~       �	��PAYc�A�*

loss���<;�       �	�4QAYc�A�*

loss�:<F�I       �	��QAYc�A�*

loss�=.v       �	�tRAYc�A�*

loss7G�<uݼ       �	�SAYc�A�*

loss8�Q=�y/[       �	k�SAYc�A�*

loss��O=�m�       �	oTAYc�A�*

loss�j;�y�       �	�
UAYc�A�*

loss�@<={��       �	M�UAYc�A�*

lossF��<���x       �	�;VAYc�A�*

loss)
=��/       �	�VAYc�A�*

lossQB=쉖�       �	$�qAYc�A�*

loss�U=�R��       �	��rAYc�A�*

loss
��=N��L       �	?sAYc�A�*

loss�$=�!�&       �	B�sAYc�A�*

loss5=�0�       �	ZKtAYc�A�*

loss�bP<��p!       �	 uAYc�A�*

lossqc=m%s%       �	�_vAYc�A�*

lossXKo=�;�=       �	B&wAYc�A�*

loss��g=��1�       �	��wAYc�A�*

loss��=�^+l       �	�xAYc�A�*

losso�<�Bz
       �	�)yAYc�A�*

loss��<�g��       �	�zAYc�A�*

losshՙ<���       �	��zAYc�A�*

loss;��<��       �	�x{AYc�A�*

losso�&=���       �	/|AYc�A�*

loss�.�=��       �	��|AYc�A�*

loss�% ;ޙ~G       �	��}AYc�A�*

loss�o�<��_�       �	�$~AYc�A�*

loss��Q<����       �	��~AYc�A�*

loss�r=�Y       �	�bAYc�A�*

loss6��=dz       �	��AYc�A�*

loss��=� 2)       �	���AYc�A�*

loss�9�<Q�        �	�H�AYc�A�*

lossMkR=gU\V       �	��AYc�A�*

loss���<�=8�       �	L��AYc�A�*

loss(��;�0p       �	-'�AYc�A�*

loss�tT<-���       �	ÃAYc�A�*

loss��<&�"       �	�t�AYc�A�*

lossVC�<�,�       �	x�AYc�A�*

loss�;�<��c       �	CɅAYc�A�*

loss0i=���       �	wd�AYc�A�*

loss��p<I�       �	:�AYc�A�*

lossM�<?J        �	@��AYc�A�*

loss��>�G�       �	[?�AYc�A�*

loss(�f=ڝ��       �	܈AYc�A�*

lossoh�=�Fb�       �	J{�AYc�A�*

loss�=;�2�(       �	��AYc�A�*

loss��=`_I       �	vފAYc�A�*

lossAι=G�Q�       �	�y�AYc�A�*

loss���=<��       �	�L�AYc�A�*

loss%q<Zh'       �	�{�AYc�A�*

loss_& =b(�       �	u�AYc�A�*

loss8N<���F       �	��AYc�A�*

loss��=q��K       �	Hk�AYc�A�*

loss���<q���       �	�AYc�A�*

loss&�g=#��       �	T��AYc�A�*

lossX�=
i��       �	9F�AYc�A�*

loss�W<='P��       �	�AYc�A�*

lossDh�=�5O       �	���AYc�A�*

loss��;I��       �	>�AYc�A�*

losst��<��       �	`�AYc�A�*

loss	�7=D]�       �	I��AYc�A�*

lossG�=6W�       �	�'�AYc�A�*

lossw">�)T        �	I��AYc�A�*

loss�D�<��e�       �	S�AYc�A�*

loss�(V<F`'�       �	h�AYc�A�*

loss-<;��       �	��AYc�A�*

lossO\�:�}�       �	Q3�AYc�A�*

lossݤ�<"�{5       �	�˘AYc�A�*

loss��$=�<ӌ       �	�d�AYc�A�*

loss��<�Y       �	'�AYc�A�*

lossd�=,ͧ       �	0��AYc�A�*

lossX~�;�>�M       �	�\�AYc�A�*

lossN=��VF       �	�N�AYc�A�*

loss�e�<-��       �	I��AYc�A�*

loss\4<��J       �	c��AYc�A�*

loss\a�<l��Q       �	wM�AYc�A�*

loss��<[��U       �	 �AYc�A�*

loss�n=�c�       �	*��AYc�A�*

lossf�s=Lm#*       �	�?�AYc�A�*

loss�hj=jR��       �	B`�AYc�A�*

loss��%=�X��       �	�	�AYc�A�*

loss���<n�
       �	%��AYc�A�*

loss�Z =�Uc�       �	mS�AYc�A�*

loss}1�<j6       �	��AYc�A�*

loss?�=
���       �	���AYc�A�*

loss�Wc<�C��       �	![�AYc�A�*

lossi4=
aQ�       �	��AYc�A�*

loss�1O<`�̷       �	ѱ�AYc�A�*

loss��<tY�0       �	�[�AYc�A�*

loss3΅=��m]       �	���AYc�A�*

loss�w=��       �	���AYc�A�*

loss�N�=l�/*       �	<J�AYc�A�*

lossV��;rB��       �	��AYc�A�*

loss�T	=Z��       �	���AYc�A�*

loss;~�<b=ӽ       �	6>�AYc�A�*

loss��&=�{$A       �	�߫AYc�A�*

loss���<g�d_       �	���AYc�A�*

loss��<RJ!]       �	_%�AYc�A�*

loss�6�<���=       �	�έAYc�A�*

loss�{,=e�w       �	hy�AYc�A�*

lossss�<iC�       �	!�AYc�A�*

loss� �<n�O�       �	�̯AYc�A�*

lossE��<04J�       �	�|�AYc�A�*

loss�0�=^��_       �	��AYc�A�*

loss,;d<᫄�       �	>�AYc�A�*

loss�h=D:�       �	:�AYc�A�*

loss��=�8\�       �	���AYc�A�*

lossܖV=�Ef       �	4�AYc�A�*

lossW��<�ᾈ       �	z�AYc�A�*

loss3�P<��~q       �	҉�AYc�A�*

loss%#<Bx��       �	+3�AYc�A�*

lossצ�< '�'       �	�ܶAYc�A�*

loss��n<�       �	��AYc�A�*

loss�=�H/       �	�(�AYc�A�*

loss�=��        �	�׸AYc�A�*

loss�;�&�k       �	4��AYc�A�*

lossܽ <	���       �	�D�AYc�A�*

loss�o�=�0�       �	�AYc�A�*

loss��C=^7�       �	��AYc�A�*

loss��}=��F�       �	�;�AYc�A�*

losso�=9UѬ       �	v�AYc�A�*

loss��3={b       �	���AYc�A�*

loss�};��       �	1�AYc�A�*

loss%<@�<       �	�־AYc�A�*

loss�m0<�       �	댿AYc�A�*

loss�$�<p��       �	l=�AYc�A�*

loss�J\=�)�K       �	`��AYc�A�*

lossQ=2(�d       �	���AYc�A�*

loss��;oc��       �	�/�AYc�A�*

loss��	=����       �	<��AYc�A�*

loss�=dk       �	���AYc�A�*

lossJ�<���       �	7�AYc�A�*

lossmt�<�� *       �	��AYc�A�*

loss�e�;�V{�       �	u��AYc�A�*

lossm>(=��|       �	�C�AYc�A�*

loss�$V<W(�|       �	R��AYc�A�*

loss��<�5��       �	��AYc�A�*

lossHH=�?�       �	�P�AYc�A�*

lossC�j=lh�       �	��AYc�A�*

loss-�r<�[C�       �	���AYc�A�*

loss��=����       �	'K�AYc�A�*

lossĈ�=�xO�       �	u �AYc�A�*

loss���<�w{       �	׾�AYc�A�*

lossW��<�P�N       �	N�AYc�A�*

lossݚ�<M�d       �	|,�AYc�A�*

loss�!�<��w       �	��AYc�A�*

lossͰ�;s#s]       �	F��AYc�A�*

lossq�<!/��       �	�8�AYc�A�*

loss��C<XZ�       �	���AYc�A�*

lossԥ0;;S(�       �	1~�AYc�A�*

lossh*=�w׃       �	?X�AYc�A�*

lossT��<C�       �	�	�AYc�A�*

loss.B�=�/�C       �	N��AYc�A�*

lossx=7� �       �	^�AYc�A�*

lossP=28�       �	� �AYc�A�*

loss�i�<{���       �	��AYc�A�*

loss��;~k}�       �	�g�AYc�A�*

loss�<�::y       �	��AYc�A�*

loss�=*��       �	���AYc�A�*

loss�  =Do��       �	a��AYc�A�*

loss�UP=�h�a       �	�>�AYc�A�*

loss��%=��X�       �	���AYc�A�*

loss�bx=n�J�       �	���AYc�A�*

lossڗI=�C��       �	���AYc�A�*

loss<��<$�       �	�@�AYc�A�*

loss��;��       �	w��AYc�A�*

loss���<lNگ       �	z��AYc�A�*

loss��Q=x���       �	.W�AYc�A�*

loss�b2=����       �	��AYc�A�*

lossʌC=��$�       �	���AYc�A�*

loss��;,��       �	*W�AYc�A�*

loss	�&=��r�       �	���AYc�A�*

loss��;�(q�       �	ܠ�AYc�A�*

loss!�=���       �	�B�AYc�A�*

loss6~�=H�{e       �	Z/�AYc�A�*

loss�9�<�JtC       �	Z��AYc�A�*

loss��=��v       �	�~�AYc�A�*

loss#&<B��g       �	� �AYc�A�*

loss M�=n�>�       �	e��AYc�A�*

loss�q�<S��       �	�f�AYc�A�*

loss���<ۺ��       �	�AYc�A�*

loss&�<<�G�       �	ߧ�AYc�A�*

loss��`=�W�<       �	^G�AYc�A�*

lossߣ<+[c       �	���AYc�A�*

lossh=���N       �	���AYc�A�*

loss�5=I��W       �	 :�AYc�A�*

loss�r�=�V�0       �	E��AYc�A�*

loss8�<a�O       �	�s�AYc�A�*

loss��<S|��       �	��AYc�A�*

loss<�;��U       �	���AYc�A�*

loss�=��        �	jO�AYc�A�*

loss��<�ߙ>       �	���AYc�A�*

loss�	X=k8q       �	��AYc�A�*

loss�$<qQ�L       �	|G�AYc�A�*

lossh�<��       �	���AYc�A�*

lossM^A=��6{       �	Oy�AYc�A�*

loss��<��       �	�$�AYc�A�*

lossj6�<��%       �	?��AYc�A�*

loss�Z�<��
�       �	�m�AYc�A�*

lossAH=.�P�       �	��AYc�A�*

loss?\=��V       �	
��AYc�A�*

lossI��<�Մ�       �	�[�AYc�A�*

loss��<=���       �	���AYc�A�*

loss"�;ʴ�        �	��AYc�A�*

loss��<��v       �	���AYc�A�*

loss\"4<�;&�       �	P��AYc�A�*

lossDȕ=E��n       �	H�AYc�A�*

loss�DD=h���       �	���AYc�A�*

loss�o3<�i.       �	N��AYc�A�*

loss\�=�0E       �	oI�AYc�A�*

loss�#<F�Y�       �	N��AYc�A�*

loss�3�<��7�       �	s��AYc�A�*

loss�>=$1��       �	L�AYc�A�*

loss�
�<OY {       �	���AYc�A�*

lossU�;�Puq       �	l��AYc�A�*

loss ��=���m       �	�0�AYc�A�*

loss]�=�h6�       �	��AYc�A�*

loss��<���X       �	ǁ�AYc�A�*

loss�̅=�'�       �	�1�AYc�A�*

loss�=�L>�       �	!��AYc�A�*

lossګ=����       �	� BYc�A�*

loss���;�H�       �	�1BYc�A�*

loss�p<�I�e       �	R�BYc�A�*

loss� u=���?       �	�zBYc�A�*

lossZ5n=U��       �	sJBYc�A�*

loss.�W=X�       �	��BYc�A�*

lossVÌ=|��       �	ђBYc�A�*

loss$��=	/�       �	f1BYc�A�*

loss�=<�&�       �	��BYc�A�*

loss1��<�g�d       �	�vBYc�A�*

lossa�=��       �	�BYc�A�*

loss�s =I��       �	÷BYc�A�*

lossr=�3�       �	�YBYc�A�*

lossp�
<2$�?       �	u 	BYc�A�*

loss�v�=���Q       �	@�	BYc�A�*

loss1�{<�S��       �	J
BYc�A�*

loss�y�<G<j�       �	��
BYc�A�*

loss��z=?�P       �	?�BYc�A�*

loss��<�ӆ<       �	�0BYc�A�*

loss�;�<B벩       �	�BYc�A�*

loss��:=�k�;       �	�~BYc�A�*

loss$�R=��       �	4,BYc�A�*

loss ��=���       �	��BYc�A�*

loss\/�=�xO�       �	�{BYc�A�*

loss�y�<�!jc       �	5*BYc�A�*

loss��<2�P�       �	��BYc�A�*

loss��=���*       �	^�BYc�A�*

loss��*>}.�       �	@1BYc�A�*

loss��;W˃R       �	��BYc�A�*

loss�sO=k�Az       �	�BYc�A�*

loss[-s=%�       �	U.BYc�A�*

loss�O=��2       �	V�BYc�A�*

lossΤG=;���       �	BvBYc�A�*

loss�I=�u�       �	8BYc�A�*

loss��=32�       �	y�BYc�A�*

loss!�I<��#�       �	�vBYc�A�*

loss�-8<�6��       �	�BYc�A�*

loss��[=q��O       �	x�BYc�A�*

lossZ.�<S�6       �	SBYc�A�*

loss�#�<i       �	Y�BYc�A�*

loss;a=�F�K       �	i�BYc�A�*

loss���<����       �	I�BYc�A�*

loss�K�<xN�       �	��BYc�A�*

loss4�><�?��       �	{0BYc�A�*

loss-1<��       �	}�BYc�A�*

lossC�=:�Q       �	�fBYc�A�*

loss}"�<��        �	  BYc�A�*

loss%�<���Y       �	�V BYc�A�*

loss�<���S       �	�� BYc�A�*

loss��*<��Z^       �	��!BYc�A�*

loss;�	<E�l�       �	�U"BYc�A�*

loss{��<��o�       �	��"BYc�A�*

loss$�<+�P       �	��#BYc�A�*

lossQ�=̫�       �	�=$BYc�A�*

lossL1<^��v       �	��$BYc�A�*

loss��]= b�F       �	��%BYc�A�*

loss�=��`       �	�K&BYc�A�*

loss��=��V�       �	��&BYc�A�*

lossױ�=0���       �	�'BYc�A�*

loss	�=B��       �	�(BYc�A�*

loss��1=���)       �	��(BYc�A�*

loss�8D<�	�       �	�\)BYc�A�*

lossxV=�ct�       �	�*BYc�A�*

lossŖ`=�?�        �	r�*BYc�A�*

loss`�>=T՛�       �	�J+BYc�A�*

loss|�<��:,       �	H�+BYc�A�*

loss�%j<����       �	ޒ,BYc�A�*

loss�<;Si       �	4,-BYc�A�*

lossF��=�G|]       �	T�-BYc�A�*

loss|�=���       �	�\.BYc�A�*

loss_��;���       �	�/BYc�A�*

lossw[6=��m       �	��/BYc�A�*

loss]�<����       �	�A0BYc�A�*

lossԞ�<$Q�.       �	�0BYc�A�*

loss�K�<���       �	]n1BYc�A�*

lossYV=(��       �	0*2BYc�A�*

loss�s=0�F�       �	��2BYc�A�*

loss5�<赍3       �	�]3BYc�A�*

lossǙ�;���       �		�3BYc�A�*

lossw�;��i       �	�4BYc�A�*

loss;m=�F�       �	K<5BYc�A�*

lossa�;6<��       �	��5BYc�A�*

loss�i�;�VS       �	�6BYc�A�*

lossE��<ḰP       �	�67BYc�A�*

loss���<�y�       �	��7BYc�A�*

loss��<��4U       �	�8BYc�A�*

loss�.�=h��       �	m:9BYc�A�*

loss�q2=��N'       �	=:BYc�A�*

loss �3=kj�+       �	b�:BYc�A�*

loss�= =R��       �	%u;BYc�A�*

loss!Q=�I��       �	�<BYc�A�*

loss{¸;�P�P       �	�<BYc�A�*

loss=�=ڋq�       �	7S=BYc�A�*

lossB=��       �	��=BYc�A�*

loss�=bL�       �	��>BYc�A�*

lossQ��<e�       �	�;?BYc�A�*

lossn�=��       �	m�?BYc�A�*

lossםr</"       �	�@BYc�A�*

lossw?=w�       �	�*ABYc�A�*

lossc �<
4�       �	x�ABYc�A�*

loss��<5�W�       �	]kBBYc�A�*

loss�r�<)�,e       �	CBYc�A�*

loss��8<�S�^       �	�CBYc�A�*

lossI]<�o��       �	�TDBYc�A�*

loss�=��j       �	��DBYc�A�*

loss[V<M�~       �	I�EBYc�A�*

loss���<��z       �	�DFBYc�A�*

loss+�=��jX       �	��FBYc�A�*

loss�!'=hsk�       �	GBYc�A�*

loss���<'�c�       �	y!HBYc�A�*

loss�y�=�       �	b�HBYc�A�*

lossI�<lQ��       �	�zIBYc�A�*

loss�rH=l��m       �	�JBYc�A�*

loss=�$<���/       �	��JBYc�A�*

lossH�}=3X{       �	kaKBYc�A�*

lossq0=Qw�-       �	 *LBYc�A�*

lossg<L��<       �	��LBYc�A�*

loss�ֺ<uD��       �	!vMBYc�A�*

loss��=c|�       �	NBYc�A�*

loss��/<K��       �	��NBYc�A�*

loss��<JE��       �	��OBYc�A�*

loss;�(<U~>�       �	)%PBYc�A�*

loss���=��#�       �	/�PBYc�A�*

loss��=�Ⱥ�       �	J`QBYc�A�*

loss�j=;,�       �	i�QBYc�A�*

lossj/7=��^/       �	�RBYc�A�*

loss�`�<�R�       �	CSBYc�A�*

loss.�<լ�       �	TBYc�A�*

loss��<����       �	֪TBYc�A�*

loss���;f��%       �	'LUBYc�A�*

lossS��<�}Wj       �	��UBYc�A�*

loss��(=�T$u       �	1�VBYc�A�*

loss���<F\d       �	<2WBYc�A�*

loss�!�=�BD,       �	G�WBYc�A�*

loss�Ƙ=�1��       �	'gXBYc�A�*

loss���<�:5       �	��XBYc�A�*

loss)Z�<a�       �	��YBYc�A�*

loss��=��       �	�rZBYc�A�*

loss�<,-��       �	s[BYc�A�*

lossn�\<���L       �	��[BYc�A�*

losst�F<�w�7       �	�\BYc�A�*

lossΧ�<��C�       �	H]BYc�A�*

lossW�Q=��ʿ       �	^BYc�A�*

loss�x�=�@�       �	*�^BYc�A�*

loss���;-��       �	�K_BYc�A�*

loss@��<� �       �	R�_BYc�A�*

loss���<�'�l       �	 �`BYc�A�*

loss��9=%�%       �	�7aBYc�A�*

loss��=�uH�       �	��aBYc�A�*

loss�o�<���'       �	��bBYc�A�*

loss��n<���Q       �	�hcBYc�A�*

lossh�=���I       �	fdBYc�A�*

loss߫v=%^�       �	��dBYc�A�*

lossߓ�=[�.Q       �	feBYc�A�*

loss�'z=@l       �	�fBYc�A�*

loss]1{=�Fl       �	j�fBYc�A�*

lossj��<2�D�       �	PrgBYc�A�*

loss�s<�c       �	
hBYc�A�*

loss�=ȇ~�       �	=�hBYc�A�*

lossrƂ=�ߗ�       �	�aiBYc�A�*

lossi��<ԅ�W       �	�jBYc�A�*

loss�Y:=m�	�       �	F�jBYc�A�*

loss��<�̥�       �	p`kBYc�A�*

loss�	1=kR%e       �	�lBYc�A�*

loss7�=EN�       �	�lBYc�A�*

lossQ�I=h�$       �	�NmBYc�A�*

loss�#.=H�r       �	��mBYc�A�*

loss�>=)X(�       �	N�nBYc�A�*

lossߟ�=̖��       �	�:oBYc�A�*

loss�:�=⁊       �	Q�oBYc�A�*

lossߣa="�r       �	�xpBYc�A�*

loss%B'<S+,       �	�qBYc�A�*

loss&�<��`�       �	�qBYc�A�*

loss��;�)S2       �	�ZrBYc�A�*

loss{�Y=/QY       �	MLsBYc�A�*

loss~(=g�[_       �	�tBYc�A�*

loss��+=�Ɔ?       �	�tBYc�A�*

loss��T<�EY"       �	{�uBYc�A�*

lossf�[=W7�g       �	eSvBYc�A�*

loss��<;��t       �	��wBYc�A�*

loss�=Z�       �	��xBYc�A�*

loss���<���       �	cCyBYc�A�*

losssv�<�g�       �	��yBYc�A�*

lossi��<�T,       �	�zBYc�A�*

loss� �=F�       �	lw{BYc�A�*

loss��(<`��       �	o,|BYc�A�*

loss_�=�Kx       �	z�|BYc�A�*

lossش�<$c�J       �	~BYc�A�*

loss�~�<(?�+       �	6�~BYc�A�*

lossQ<�v�       �	�BYc�A�*

loss=�*<�/��       �	,-�BYc�A�*

loss�a�=���       �	F�BYc�A�*

loss��<����       �	�	�BYc�A�*

lossCN=��U�       �	1��BYc�A�*

loss��w=~uQ       �	�i�BYc�A�*

loss���<D�P       �	$�BYc�A�*

loss��$=2d       �	ܡ�BYc�A�*

loss�K?=H��L       �	�T�BYc�A�*

lossQ� =ȳx�       �	��BYc�A�*

loss�y=k�f`       �	v��BYc�A�*

loss�]�=Y~�       �	e�BYc�A�*

loss�k�<K�       �	 ��BYc�A�*

loss��a=u�I?       �	�b�BYc�A�*

losshP}=�~^=       �	��BYc�A�*

losso�F=��       �	��BYc�A�*

loss$~�;b�GR       �	LR�BYc�A�*

lossſ<A�       �	F	�BYc�A�*

loss��=�CT       �	[��BYc�A�*

lossF��=���b       �	-_�BYc�A�*

loss�9�<���       �	��BYc�A�*

loss{�3=�9��       �	���BYc�A�*

lossva=���a       �	dX�BYc�A�*

loss??�=�IJ       �	~�BYc�A�*

loss���<�9fW       �	G��BYc�A�*

loss��B=�:@�       �	fL�BYc�A�*

loss�c�=tZ��       �	��BYc�A�*

loss́�= �J       �	��BYc�A�*

loss]�6=f�=&       �	V��BYc�A�*

loss_��;�3n�       �	�]�BYc�A�*

loss@I=�       �	@��BYc�A�*

lossA9�<�i       �	ٖ�BYc�A�*

loss.�f<`}"       �	�=�BYc�A�*

lossh��<�Y%	       �	��BYc�A�*

loss�t�<�I�~       �	���BYc�A�*

lossa[=�#�]       �	C9�BYc�A�*

lossn1�<�7       �	 �BYc�A�*

loss��W<����       �	���BYc�A�*

lossc�v=в�\       �	�/�BYc�A�*

loss�G�<@�f       �	�ܛBYc�A�*

loss\6=�B�       �	�BYc�A�*

lossqA�=��CM       �	)>�BYc�A�*

loss��A<`�Fq       �	��BYc�A�*

loss��=`l�f       �	8��BYc�A�*

lossW��<��Q�       �	��BYc�A�*

lossG=}OH@       �	�7�BYc�A�*

losst_�=��8       �	H�BYc�A�*

loss�U=eB=�       �	L��BYc�A�*

lossj|�<[f�       �	ĵ�BYc�A�*

loss�S�<���2       �	|��BYc�A�*

loss���=�]/�       �	}"�BYc�A�*

loss
��;l       �	�äBYc�A�*

loss;�(<��fT       �	�c�BYc�A�*

loss假='Z��       �	��BYc�A�*

loss��2=�Y��       �	��BYc�A�*

loss_"=�s�{       �		Q�BYc�A�*

loss]��<�,�G       �	��BYc�A�*

loss���<�3�V       �	[��BYc�A�*

loss=ن<��E�       �	�<�BYc�A�*

loss,Yk=��~�       �	کBYc�A�*

loss�F=���       �	�t�BYc�A�*

loss]q=����       �	D�BYc�A�*

loss���;�З       �	2�BYc�A�*

loss�c=b��       �	��BYc�A�*

loss=�=�r\�       �	��BYc�A�*

lossz��;>�j       �	���BYc�A�*

lossʖV=��a�       �	���BYc�A�*

losse�J<I��       �	^�BYc�A�*

loss[��;R�X�       �	���BYc�A�*

loss`��;�
�       �	_��BYc�A�*

loss�F�=޸�V       �	�]�BYc�A�*

loss�r�<��^�       �	��BYc�A�*

loss=߇=��       �	؞�BYc�A�*

loss��=R��d       �	O;�BYc�A�*

loss7v=���       �	4سBYc�A�*

lossh��<���r       �	τ�BYc�A�*

lossw��;nv~c       �	� �BYc�A�*

loss&�(=�X�       �	׾�BYc�A�*

lossҞ<$!�       �	t�BYc�A�*

loss���=�%��       �	���BYc�A�*

lossE��=��=S       �	���BYc�A�*

loss��1=���       �	P��BYc�A�*

loss�lk<�i       �	eR�BYc�A�*

loss�N5=��       �	U�BYc�A�*

lossݖ.=���       �	���BYc�A�*

loss�َ<B���       �	�`�BYc�A�*

loss��<�n�       �	��BYc�A�*

loss�rd=
�8>       �	2��BYc�A�*

loss���;(��`       �	���BYc�A�*

loss���<ٛ��       �	�n�BYc�A�*

loss���=�Zf}       �	�BYc�A�*

loss��'=@�%�       �	ղ�BYc�A�*

loss<oM<7�~)       �	�T�BYc�A�*

loss�`�=m�œ       �	@��BYc�A�*

lossdy�<G�r       �	���BYc�A�*

loss��Y<�<|�       �	>A�BYc�A�*

loss���<�W�'       �	*��BYc�A�*

loss���<���R       �	��BYc�A�*

loss�,�<q�<       �	T�BYc�A�*

loss�	�<��%�       �	���BYc�A�*

loss���<���       �	�_�BYc�A�*

loss��A;�K�$       �	���BYc�A�*

lossE*8<Q�&       �	V��BYc�A�*

loss"�<"s��       �	�7�BYc�A�*

loss߆�<����       �	Q��BYc�A�*

loss�z<J��       �	Z��BYc�A�*

loss.��<>#[�       �	�;�BYc�A�*

lossuW=X��       �	���BYc�A�*

loss�E3=���E       �	;o�BYc�A�*

lossq�=�-o       �	@�BYc�A�*

loss�T~<�H�       �	���BYc�A�*

loss�<�>��       �	�J�BYc�A�*

loss6��;����       �	(��BYc�A�*

loss�j�=�t�       �	���BYc�A�*

loss�=�<��x       �	zr�BYc�A�*

loss6��=4�x       �	�	�BYc�A�*

loss�v<{O,W       �	��BYc�A�*

loss�A=N#�       �	 F�BYc�A�*

loss��A=b� z       �	���BYc�A�*

lossG	=�}�W       �	���BYc�A�*

loss<�U�       �	�(�BYc�A�*

loss���<�̽�       �	���BYc�A�*

loss�:S���       �	�b�BYc�A�*

loss7��<QJ       �	���BYc�A�*

lossfs<�gv       �	��BYc�A�*

loss�,u<ji�       �	�0�BYc�A�*

loss��<��Q�       �	���BYc�A�*

lossR�<�:w(       �	vl�BYc�A�*

loss���=�Wǒ       �	�BYc�A�*

loss̧l<�Й�       �	��BYc�A�*

loss�š<^�xQ       �	�U�BYc�A�*

loss�3�=�L|�       �	c��BYc�A�*

loss�.<?r�       �	a��BYc�A�*

loss�Н:�sY       �	_'�BYc�A�*

loss`	;<m6M�       �	���BYc�A�*

loss�e;��:z       �	�BYc�A�*

loss�p�;���\       �	ND�BYc�A�*

loss�9<v�e�       �	�@�BYc�A�*

loss���;W���       �	���BYc�A�*

lossHr=5�       �	���BYc�A�*

loss�5+<��L       �	NC�BYc�A�*

loss��u::�O"       �	���BYc�A�*

loss�4:��       �	И�BYc�A�*

loss�u+<�jY�       �	L8�BYc�A�*

loss-j.=�A��       �	���BYc�A�*

loss�Z�<l-       �	-w�BYc�A�*

loss�V�:�_j�       �	��BYc�A�*

loss�_�;[W5�       �	���BYc�A�*

loss6�L=n��       �	͑�BYc�A�*

lossJ5�:��x8       �	�;�BYc�A�*

loss��<���       �	��BYc�A�*

loss{��<��f�       �	To�BYc�A�*

lossn�=&`�3       �	��BYc�A�*

loss�4�<��@       �	���BYc�A�*

loss(=<g@R'       �	aU�BYc�A�*

loss7�<\���       �		�BYc�A�*

loss(��<�       �	��BYc�A�*

loss�9 =����       �	�C�BYc�A�*

lossZ-�<͓Gy       �	���BYc�A�*

lossֱ�=�t2�       �	��BYc�A�*

loss���=�%��       �	B%�BYc�A�*

loss]��==Y�       �	���BYc�A�*

loss{�<QT7�       �	]�BYc�A�*

loss��<G΋�       �	0*�BYc�A�*

loss!��<�)��       �	)��BYc�A�*

loss��@=��?       �	��BYc�A�*

loss�/<䆨z       �	DR�BYc�A�*

loss�f�=U�=�       �	���BYc�A�*

loss�<��AP       �	��BYc�A�*

lossҎ�<cZ��       �	rN�BYc�A�*

loss��<�dP       �	���BYc�A�*

loss���<�oM�       �	,��BYc�A�*

lossax<��       �	�8�BYc�A�*

loss%�<r?TA       �	0��BYc�A�*

loss]x4;�       �	q�BYc�A�*

loss��:=x��'       �	�L�BYc�A�*

losse��<@�7�       �	��BYc�A�*

loss@��=e���       �	��BYc�A�*

loss��m=Kk?       �	.:�BYc�A�*

loss�vH<��       �	���BYc�A�*

loss�$�<�V�       �	�|�BYc�A�*

lossL	<u�W       �	��BYc�A�*

lossv�;��Y�       �	Y��BYc�A�*

loss\�,<�MY�       �	~n�BYc�A�*

loss	X�;'�w�       �	��BYc�A�*

loss$l<�r	       �	���BYc�A�*

loss���<�f�       �	}[�BYc�A�*

loss��=E~��       �	i��BYc�A�*

lossJ��<�ST�       �	�� CYc�A�*

lossC1<xHY       �	C8CYc�A�*

loss�S&<-��       �	��CYc�A�*

loss��<~��       �	�qCYc�A�*

loss��<h���       �	�	CYc�A�*

loss�]<g�Q�       �	v�CYc�A�*

loss�n�<�"�       �	�ACYc�A�*

loss�&U=�$!	       �	n�CYc�A�*

lossVt<�ʬ       �	4�CYc�A�*

losst��<���       �	�(CYc�A�*

loss�J�<H6B�       �	��CYc�A�*

lossʳ�=�(L/       �	iCYc�A�*

loss8��<2o1       �	��CYc�A�*

loss
�2=�u}B       �	1BCYc�A�*

loss�`=8D�       �	��CYc�A�*

loss�մ<��F       �	��CYc�A�*

lossί<�       �	1# CYc�A�*

loss���<�'��       �	� CYc�A�*

loss��<��[D       �	�P!CYc�A�*

loss�r�<?�'a       �	��!CYc�A�*

loss�y=5���       �	�{"CYc�A�*

lossl]�<bd1�       �	
#CYc�A�*

loss���;�}��       �	��#CYc�A�*

loss��=��        �	�N$CYc�A�*

lossP<�(�       �	A�$CYc�A�*

loss}}j=%s�       �	i�%CYc�A�*

lossw\=��:�       �	q:&CYc�A�*

lossVT�=��       �	U�&CYc�A�*

lossœ6:�M;       �	�|'CYc�A�*

loss�5=+eDc       �	�'(CYc�A�*

loss�v<��?       �	?�(CYc�A�*

lossJ�D=����       �	c)CYc�A�*

lossw�?=5 t       �	��)CYc�A�*

lossv �=x�E       �	�*CYc�A�*

loss��;eX�       �	<+CYc�A�*

loss���=��       �	��+CYc�A�*

lossE�/<�T�R       �	��,CYc�A�*

loss��<L��0       �	�-CYc�A�*

lossu�=��<�       �	w�-CYc�A�*

loss�s<�:̵       �	�Q.CYc�A�*

loss�!�<��w       �	�.CYc�A�*

loss���<�8�       �	��/CYc�A�*

loss���<���       �	790CYc�A�*

loss�8=���       �	��0CYc�A�*

lossq�<�}�>       �	uw1CYc�A�*

lossҳ�=Hٴ�       �	�!2CYc�A�*

loss��S<���?       �	��2CYc�A�*

lossu�=�S|       �	�i3CYc�A�*

loss���<��       �	h%4CYc�A�*

lossLr�<�W~       �	D�4CYc�A�*

loss��>G���       �	�r5CYc�A�*

loss�*�=�F��       �	76CYc�A�*

loss:��;VLA       �	��6CYc�A�*

loss�=�       �	�7CYc�A�*

loss3�<�l^       �	~T8CYc�A�*

lossڟ�<�y�B       �	��8CYc�A�*

lossP<�^!l       �	L�9CYc�A�*

lossf�I=C|E       �	�`:CYc�A�*

loss��=�G�|       �	�;CYc�A�*

loss��=Ռ�       �	}�;CYc�A�*

loss�f2=��S�       �	�D<CYc�A�*

loss��L;��S�       �	��<CYc�A�*

loss�R�<�M�       �	n�=CYc�A�*

loss��=��5       �	)[>CYc�A�*

lossi/�<�E�E       �	* ?CYc�A�*

loss��=@p�       �	Ŭ?CYc�A�*

loss�ѩ<��       �	@Q@CYc�A�*

loss�Y;��^       �	f�@CYc�A�*

loss`�;�'�Z       �	��ACYc�A�*

lossv(�:b7�!       �	�cBCYc�A�*

loss�t�<��       �	>CCYc�A�*

loss*��<U�5o       �	��CCYc�A�*

loss���<C�[�       �	.9DCYc�A�*

loss�\�<6��f       �	O�DCYc�A�*

lossn \<���D       �	�}ECYc�A�*

lossv�=�d�Z       �	b-FCYc�A�*

loss8�<'��A       �	��FCYc�A�*

loss�9�<���       �	x�GCYc�A�*

lossSQ=ٔ'�       �	�WHCYc�A�*

loss(��<]3�4       �	3�HCYc�A�*

loss(x8==�RW       �	ƣICYc�A�*

loss�B�=E6گ       �	�AJCYc�A�*

loss27�<h���       �	��JCYc�A�*

losss-�=�( M       �	A�KCYc�A�*

loss/�=I��       �	�LCYc�A�*

loss*�S=���X       �	��LCYc�A�*

loss�ϳ<�)�       �	�dMCYc�A�*

lossOt$<�F�a       �	NCYc�A�*

loss{�}<3���       �	>�NCYc�A�*

loss%�0<�m�       �	#JOCYc�A�*

loss��k<D��       �	g�OCYc�A�*

loss1�=���       �	ʋPCYc�A�*

loss�>�<,�j�       �	�!QCYc�A�*

lossI�2='���       �	�QCYc�A�*

lossW�u<J�0u       �	@�RCYc�A�*

lossc�<�z�f       �	T;SCYc�A�*

loss��<��[�       �	��SCYc�A�*

loss�,k<_ڊ�       �	�sTCYc�A�*

lossRI<���       �	{UCYc�A�*

loss���=w�L�       �	�UCYc�A�*

loss���<�렶       �	LVCYc�A�*

loss@��<x�       �	��VCYc�A�*

losss<���       �	��WCYc�A�*

lossE�u=4��       �	�!XCYc�A�*

loss7�<uk�}       �	�XCYc�A�*

loss��<~3+�       �	�UYCYc�A�*

loss:�=���       �	-�YCYc�A�*

loss� <|g?       �	]�ZCYc�A�*

loss�=���       �	O[CYc�A�*

lossl�=���       �	R�[CYc�A�*

loss�y�=ѠN       �	�^\CYc�A�*

loss��<,�       �	�]CYc�A�*

loss@
�;n��       �	0b^CYc�A�*

loss�Ȱ<A;�$       �	�^CYc�A�*

loss��=?j       �	W�_CYc�A�*

lossz�;��Ar       �	�/`CYc�A�*

lossj�<x��]       �	��`CYc�A�*

loss�=ò��       �	�faCYc�A�*

lossm'r<�~m<       �	� bCYc�A�*

loss%�9=Pz6!       �	J�bCYc�A�*

loss���=u�,       �	�2cCYc�A�*

loss�"�<��4�       �	�cCYc�A�*

loss�:�=�z��       �	jeCYc�A�*

lossj��<�fJ�       �	C�eCYc�A�*

lossv"1=��       �	.WfCYc�A�*

loss�7�;t���       �	F�fCYc�A�*

loss�P�<-��       �	E�gCYc�A�*

loss���;*T�       �	$hCYc�A�*

loss=�i<]���       �	V�hCYc�A�*

loss�i="��       �	�SiCYc�A�*

loss�r'=ʹÚ       �	+�iCYc�A�*

loss�ҡ<�
L�       �	p�jCYc�A�*

lossHq�<6=�       �	�<kCYc�A�*

loss4n=�ml       �	��kCYc�A�*

loss=�Md�       �	�llCYc�A�*

loss��l<aR�       �	�mCYc�A�*

loss���;y� �       �	 �mCYc�A�*

loss`�6=?C�       �	$EnCYc�A�*

lossL96=��T�       �	�oCYc�A�*

lossC\�<ǐ%�       �	_�oCYc�A�*

loss��G=��	)       �	�^pCYc�A�*

lossR�Q=��.)       �	
�pCYc�A�*

loss�<Nlo�       �	�qCYc�A�*

loss4�=�ͩ�       �	R)rCYc�A�*

lossm�j=J�=       �	��rCYc�A�*

loss���<�/       �	�ksCYc�A�*

loss/n�=~Wp}       �	6tCYc�A�*

loss<<�$"a       �	��tCYc�A�*

loss�n�;(��       �	�5uCYc�A�*

lossO��<�q�       �	K�uCYc�A�*

loss/�=QW=G       �	�evCYc�A�*

loss$�<g3@�       �	�wCYc�A�*

loss�N;U0q       �	C�wCYc�A�*

loss�@�<;��       �	
IxCYc�A�*

loss�m�<f�n�       �	�xCYc�A�*

lossR%l<-bF       �	�vyCYc�A�*

loss��=�9       �	@zCYc�A�*

loss��<Qz��       �	��zCYc�A�*

loss�C�<Ѩ\       �	<i{CYc�A�*

loss��;nZ(i       �	;|CYc�A�*

losshf;Y�7D       �	��|CYc�A�*

lossvչ<o��       �	�5}CYc�A�*

loss���<��u       �	��}CYc�A�*

loss&b�=(_       �	�}~CYc�A�*

loss��<���       �	�CYc�A�*

lossFE=)(��       �	��CYc�A�*

lossz@/=� ��       �	>��CYc�A�*

loss۞T<�[�       �	�O�CYc�A�*

loss�;�;���       �	t��CYc�A�*

loss�f=����       �	���CYc�A�*

loss��p=3"�4       �	&9�CYc�A�*

loss:i�<.1��       �	7݃CYc�A�*

loss�ib=�o�       �	��CYc�A�*

loss��<t���       �	��CYc�A�*

loss��
={��       �	���CYc�A�*

loss�s<}Q��       �	�o�CYc�A�*

lossn*�=�F�       �	�CYc�A�*

loss� "<��
        �	��CYc�A�*

loss���;��>�       �	�P�CYc�A�*

lossT�=�_��       �	W�CYc�A�*

loss�p�<���m       �	P��CYc�A�*

loss�I<]խ�       �	s.�CYc�A�*

loss<��;�O       �	\ɊCYc�A�*

lossZF<ɵ�       �	�f�CYc�A�*

lossu�;��m�       �	���CYc�A�*

lossM�=»O�       �	��CYc�A�*

lossmrq<ܸ�T       �	�7�CYc�A�*

lossd�=�˻�       �	'��CYc�A�*

loss`��=���       �	���CYc�A�*

loss�o=�{�       �	/�CYc�A�*

loss�l==�j��       �	ʏCYc�A�*

lossh��<Cw       �	���CYc�A�*

lossτ�<�J�8       �	i:�CYc�A�*

lossj/�<ŧ�       �	��CYc�A�*

loss���=(�1�       �	
f�CYc�A�*

lossڃe<��I�       �	��CYc�A�*

loss�a=6��=       �	���CYc�A�*

loss��<	���       �	�>�CYc�A�*

loss7=�z8�       �	�CYc�A�*

loss0=G�_       �	���CYc�A�*

lossd��<}�`       �	}]�CYc�A�*

loss��=O�w�       �	+Q�CYc�A�*

loss��<�Ϸ       �	8��CYc�A�*

loss�A=2U       �	i��CYc�A�*

loss< =���       �	�K�CYc�A�*

loss�2�;F^��       �	9�CYc�A�*

lossq�y;c���       �	ѕ�CYc�A�*

loss$
�<���       �	J�CYc�A�*

loss��<����       �	i�CYc�A�*

lossc=��4�       �	�ϝCYc�A�*

loss��H=�o�       �	�~�CYc�A�*

loss��<�t�/       �	�)�CYc�A�*

loss�q
=���       �	*ʟCYc�A�*

loss�<�_�       �	�y�CYc�A�*

loss[�;<ʎ�       �	��CYc�A�*

loss���<g=|�       �	záCYc�A�*

loss��<H�       �	n�CYc�A�*

loss% �;`��       �	��CYc�A�*

loss-ַ<�|��       �	���CYc�A�*

loss#��<V��2       �	bK�CYc�A�*

loss��<��#       �	h�CYc�A�*

loss�͢=�lp\       �	'��CYc�A�*

lossCM�<$)4�       �	�`�CYc�A�*

loss���;��       �	��CYc�A�*

loss$^d<�Z��       �	i��CYc�A�*

loss$<�cPr       �	�I�CYc�A�*

lossl�2=���?       �	��CYc�A�*

lossCѳ=fXz�       �	ё�CYc�A�*

loss��=O�3       �	�:�CYc�A�*

loss��=�B.�       �	�ߪCYc�A�*

loss0
�=U@k       �	���CYc�A�*

loss$x�<!#�       �	�1�CYc�A�*

lossWB�<T��       �	�߬CYc�A�*

loss��<?��0       �	+��CYc�A�*

loss] �<��Fm       �	�+�CYc�A�*

loss�=�?�       �	jޮCYc�A�*

lossI�k<!'�       �	&��CYc�A�*

lossD5=����       �	�-�CYc�A�*

lossћ�<�4N7       �	װCYc�A�*

loss��<Q���       �	�±CYc�A�*

loss�=Ǟ�4       �	Mj�CYc�A�*

loss�U4< �g       �	��CYc�A�*

loss�j<�bO�       �	���CYc�A�*

loss���:��&       �	6V�CYc�A�*

loss6V=��.       �	��CYc�A�*

loss� =�T�       �	-��CYc�A�*

lossv��<59$       �	���CYc�A�*

loss#��<�c�       �	#�CYc�A�*

lossM�=]M�)       �	��CYc�A�*

loss�l<=��w�       �	~��CYc�A�*

loss�6N=�<+       �	)�CYc�A�*

loss�4�<�Ŧ�       �	�¹CYc�A�*

loss��=�}       �	l[�CYc�A�*

loss
�f<z���       �	���CYc�A�*

loss}Qb=�0l�       �	
J�CYc�A�*

lossћ�=H0�)       �	�CYc�A�*

loss��<��2       �	l{�CYc�A�*

loss!��<0�U       �	<�CYc�A�*

loss���<Rvl~       �	`��CYc�A�*

lossTkx=�@0�       �	_B�CYc�A�*

lossܾ=8�S       �	��CYc�A�*

loss�0<[��       �	��CYc�A�*

loss_-=��       �	I0�CYc�A�*

lossM�-=}�:       �	���CYc�A�*

loss?��<���d       �	Xr�CYc�A�*

loss�<��       �	�@�CYc�A�*

loss��;��m       �	���CYc�A�*

loss�*�;�>       �	̘�CYc�A�*

loss�C�<-C)�       �	�B�CYc�A�*

loss���<��c�       �	���CYc�A�*

loss�y�<�9�       �	9~�CYc�A�*

loss��0=�F       �	��CYc�A�*

loss��<�=q       �	
��CYc�A�*

loss�K�<�t       �	Rc�CYc�A�*

lossH�u=���p       �	���CYc�A�*

loss�q<֮0�       �	���CYc�A�*

lossHw�<+*       �	D�CYc�A�*

lossL�G=ݳEi       �	&��CYc�A�*

loss���<7jD�       �	+��CYc�A�*

lossH�Q=�sQ�       �	�E�CYc�A�*

loss��=p���       �	���CYc�A�*

loss��=�}�Y       �	�}�CYc�A�*

loss�L�<u�e�       �	�&�CYc�A�*

loss{0�<��@�       �	3��CYc�A�*

loss��A=�R}       �	_�CYc�A�*

lossi=�)�*       �	L��CYc�A�*

lossū�=�$��       �	h��CYc�A�*

lossML=W�B�       �	^K�CYc�A�*

lossHz�<%�&       �	���CYc�A�*

lossTO=��       �	&��CYc�A�*

lossDT*=��x       �	�0�CYc�A�*

loss�t�<���G       �	���CYc�A�*

loss���:���       �	�m�CYc�A�*

lossi�O<k�I       �	��CYc�A�*

loss�6<����       �	Ǟ�CYc�A�*

loss�;la_C       �	�=�CYc�A�*

loss�S�<A��       �	�CYc�A�*

lossju�=�Co       �	��CYc�A�*

loss�?=�%       �	I�CYc�A�*

loss�C"=��a�       �	"��CYc�A�*

loss�p%<�+��       �	H��CYc�A�*

lossa�O<�S`       �	A(�CYc�A�*

lossf�"<����       �	���CYc�A�*

lossjpd=�b'W       �	x_�CYc�A�*

loss2@�;,y�#       �	�<�CYc�A�*

loss��<Um�f       �	^��CYc�A�*

loss��<ڨf�       �	���CYc�A�*

loss��L=���P       �	v��CYc�A�*

losswR�<���       �	�B�CYc�A�*

lossl�4=I�7i       �	��CYc�A�*

loss��=�Y�       �	�w�CYc�A�*

loss��}=G�L       �	T�CYc�A�*

loss�z=�I�)       �	0��CYc�A�*

lossܳh=7�w       �	ke�CYc�A�*

loss��6=��|       �	&��CYc�A�*

loss2��=5���       �	���CYc�A�*

loss�d=����       �	(I�CYc�A�*

lossQ= H�       �	���CYc�A�*

loss��<r�       �	V��CYc�A�*

loss:�=yvט       �	%�CYc�A�*

loss�=���$       �	���CYc�A�*

losso�f<갭O       �	%\�CYc�A�*

loss7׸<���       �	��CYc�A�*

loss�/�<��p       �	 ��CYc�A�*

loss4g~<�j       �	�&�CYc�A�*

loss	��<&@�8       �	f��CYc�A�*

loss��<�칕       �	`�CYc�A�*

loss���=٘�U       �	j��CYc�A�*

loss���;7Jd$       �	��CYc�A�*

lossF�=g���       �	�E�CYc�A�*

loss�"�<�*�a       �	e��CYc�A�*

loss}��<����       �	�x�CYc�A�*

loss���<��E�       �	@�CYc�A�*

loss��=��!�       �	2��CYc�A�*

lossZ��</��       �	�~�CYc�A�*

loss�<=<�3�       �	A.�CYc�A�*

lossV��<�`E       �	i��CYc�A�*

loss���=����       �	�`�CYc�A�*

loss�Q=����       �	�T�CYc�A�*

loss�͗<:��`       �	<��CYc�A�*

loss�B�=�7�       �	͓�CYc�A�*

lossÙx=�i1�       �	E��CYc�A�*

loss =�4       �	�CYc�A�*

lossz��<�.�\       �	й�CYc�A�*

loss��=;���       �	�S�CYc�A�*

loss]��<gVQ�       �	���CYc�A�*

lossT�R=l�=p       �	���CYc�A�*

losslk@=_�S�       �	���CYc�A�*

loss
�<�;�       �	��CYc�A�*

loss@�3<s�2U       �	���CYc�A�*

loss۽P<4��       �	�P�CYc�A�*

lossK#;��       �	���CYc�A�*

loss��;s�_g       �	���CYc�A�*

loss��=�N       �	�'�CYc�A�*

loss�#v=s�@       �	r��CYc�A�*

loss�|-<���       �	�V�CYc�A�*

lossh�='u��       �	_��CYc�A�*

loss��c=T֢	       �	8��CYc�A�*

loss��l<��K       �	� DYc�A�*

losso�N=cK       �	̳ DYc�A�*

loss��<��(�       �	ZIDYc�A�*

loss��a=7}�       �	��DYc�A�*

loss�%�<��lg       �	6�DYc�A�*

loss�{�<��!       �	�DYc�A�*

loss*(=_k�.       �	�+DYc�A�*

lossծ�<���^       �	��DYc�A�*

lossZh@=��1Y       �	N_DYc�A�*

lossJ�t<�o#Q       �	��DYc�A�*

loss��<$�{�       �	 �DYc�A�*

loss�w@=K[�j       �	�+DYc�A�*

loss�i�<8 ;�       �	n�DYc�A�*

loss=�)<"Vy       �	Z�DYc�A�*

loss���<	�LJ       �	R�	DYc�A�*

loss>�<��.       �	J�
DYc�A�*

loss�{�<�駢       �	�/DYc�A�*

lossZr�==�AJ       �	m�DYc�A�*

loss*��=(�<       �	�dDYc�A�*

loss=�=}�h       �	X DYc�A�*

loss�
=�6��       �	��DYc�A�*

lossc܎</�       �	�cDYc�A�*

loss�!�<�D�       �	�DYc�A�*

loss�²<��4_       �	��DYc�A�*

lossZ�<ԟ��       �	�HDYc�A�*

loss+=����       �	��DYc�A�*

loss
<�l�k       �	��DYc�A�*

lossd�m=����       �	�5DYc�A�*

loss$��<��_�       �	o�DYc�A�*

loss�ʓ=%�O       �	`tDYc�A�*

loss�Vq=���Z       �	�DYc�A�*

loss��<�!)�       �	��DYc�A�*

loss2�4=;V�       �	/MDYc�A�*

loss�/b=�\       �	�qDYc�A�*

loss��=���       �	9	DYc�A�*

lossi!~=��       �	�DYc�A�*

loss��m<T�F�       �	s�DYc�A�*

loss��^<)�B�       �	 DYc�A�*

loss��;��v�       �	_�DYc�A�*

loss`=V�       �	fMDYc�A�*

loss�D7=���       �	��DYc�A�*

loss�j=E�m�       �	dyDYc�A�*

lossx8�<��Y       �	� DYc�A�*

loss-��<��;       �	��DYc�A�*

loss���<b�       �	�cDYc�A�*

loss{5C<?�4�       �	s�DYc�A�*

loss%�<���&       �	}�DYc�A�*

loss��=�n       �	��DYc�A�*

loss�!<�<|�       �	|c DYc�A�*

loss���=Ց�d       �	v� DYc�A�*

lossm�;�^�x       �		�!DYc�A�*

loss�vP<���=       �	�"DYc�A�*

loss�<�@�       �	l�#DYc�A�*

loss��=.3E}       �	~�$DYc�A�*

loss��"=W%�       �	�E%DYc�A�*

loss(�;dX�       �	��%DYc�A�*

loss&
=��#P       �	ˁ&DYc�A�*

loss�m0=�#e�       �	9)'DYc�A�*

loss{�=Zm4       �	��'DYc�A�*

loss�DT<��LT       �	�(DYc�A�*

loss/�=3���       �	)�)DYc�A�*

lossq�<��ӛ       �	�g*DYc�A�*

loss!4G=)�       �	e +DYc�A�*

loss%n*<�:v       �	�+DYc�A�*

loss�=�}��       �	IL,DYc�A�*

loss(,�=�-       �	�,DYc�A�*

loss?�=l�       �	j�-DYc�A�*

loss�x=� r�       �	�P.DYc�A�*

loss%6=���       �	y�.DYc�A�*

loss��!=�[       �	6�/DYc�A�*

losst��;�̲�       �	�60DYc�A�*

loss߳�<���       �	��0DYc�A�*

loss�n="Xvc       �	is1DYc�A�*

loss]*x=:�4       �	�*2DYc�A�*

lossoT�<��@       �	��2DYc�A�*

loss��m=�J�       �	�{3DYc�A�*

lossqx;���       �	�x4DYc�A�*

loss,h�=$��       �	�5DYc�A�*

loss�TH=���       �	��5DYc�A�*

loss���=�kO"       �	�G6DYc�A�*

loss<�=X���       �	�6DYc�A�*

lossH�=���       �	�v7DYc�A�*

loss	՜<I�n�       �	�8DYc�A�*

loss�_V=�TN       �	��8DYc�A�*

loss#�`=o���       �	��9DYc�A�*

lossl��<���       �	�i:DYc�A�*

lossN.�<]�Yt       �	��:DYc�A�*

loss�:r<�(       �	$�;DYc�A�*

loss��U=`�+       �	H5<DYc�A�*

loss4�M=ɮ��       �	��<DYc�A�*

loss4��<��       �	mt=DYc�A�*

loss�:V<-�h4       �	�>DYc�A�*

loss!^=U���       �	��>DYc�A�*

loss�<HN`�       �	�2?DYc�A�*

loss*4B=�-�       �	G�?DYc�A�*

loss��=��$       �	 b@DYc�A�*

lossL��=��D.       �	]�@DYc�A�*

loss��=��~       �	W�ADYc�A�*

loss1��;F[�h       �	p(BDYc�A�*

loss,��<*���       �	�BDYc�A�*

loss���=�[Y       �	�oCDYc�A�*

loss1�<��%       �	DDYc�A�*

loss`s�<[_�%       �	�DDYc�A�*

loss��]<�z��       �	1EEDYc�A�*

loss�S5=t�[l       �	~�EDYc�A�*

loss�lL<��}�       �	2�FDYc�A�*

loss�V�<�-       �	u;GDYc�A�*

loss�%�=<8Ï       �	�GDYc�A�*

loss�=-H��       �	�iHDYc�A�*

lossڴ�<�v�       �	VIDYc�A�*

loss_|�<����       �	E�IDYc�A�*

loss��=Ƨn�       �	RJDYc�A�*

lossT�<����       �	KDYc�A�*

loss��<�o       �	(�KDYc�A�*

lossZ=K���       �	kDLDYc�A�*

loss�K�;q�bL       �	:�LDYc�A�*

lossf|<X=ę       �	�MDYc�A�*

loss�	=��I�       �	_\NDYc�A�*

loss�$T=\��       �	L�NDYc�A�*

loss���<�W<	       �	P�ODYc�A�*

losssq=@�H.       �	�KPDYc�A�*

loss�-<�+�j       �	��PDYc�A�*

lossI
�<M�[�       �	cQDYc�A�*

loss���<tf�7       �	%RDYc�A�*

loss?
z=ְ.�       �	6�RDYc�A�*

loss@G�<�Ⱥ       �	�cSDYc�A�*

loss�+I<�y��       �	"�SDYc�A�*

loss�Z�=��6�       �	&�TDYc�A�*

loss��;S�T       �	�nUDYc�A�*

loss��<���       �	'VDYc�A�*

loss�2�<��J�       �	��VDYc�A�*

lossu�=���       �	TWDYc�A�*

losshX�<�h�       �	�WDYc�A�*

loss1W=�5       �	X�XDYc�A�*

loss��=�oci       �	(YDYc�A�*

loss��<t�*       �	�YDYc�A�*

loss��<�$1       �	�dZDYc�A�*

loss�.7<���o       �	6[DYc�A�*

loss�c�<�+�       �	�[DYc�A�*

loss��c<5���       �	wj\DYc�A�*

loss�ʃ<z2E�       �	�]DYc�A�*

loss3)�<��D       �	ު]DYc�A�*

lossZ�;V-j�       �	Nc^DYc�A�*

loss���=�:)       �	�^DYc�A�*

lossL�G=%�/�       �	-�_DYc�A�*

loss��=���}       �	�6`DYc�A�*

lossz��<&�D�       �	��`DYc�A�*

loss�1<40ۡ       �	1}aDYc�A�*

lossa��<�'}�       �	�bDYc�A�*

lossQ��<�'{       �	��bDYc�A� *

loss�a�=iAA�       �	AcDYc�A� *

lossM<h<�c�       �	��cDYc�A� *

loss�ʁ<"g�       �	tydDYc�A� *

loss��m=����       �	�eDYc�A� *

loss���<�{
�       �	�eDYc�A� *

loss)��;�Կ�       �	�LfDYc�A� *

loss�פ<*�D�       �	I�fDYc�A� *

loss$�<"%��       �	6�gDYc�A� *

loss��=����       �	.8hDYc�A� *

loss���<�\t�       �	��hDYc�A� *

lossOE(=Fkc�       �	�viDYc�A� *

lossZO<��b�       �	"jDYc�A� *

loss\^M=�Hȍ       �	t�jDYc�A� *

loss��,=�w5       �	iQkDYc�A� *

loss���<���       �	��kDYc�A� *

lossd�<���d       �	@�lDYc�A� *

lossU�<����       �	�'mDYc�A� *

loss���=��1       �	��mDYc�A� *

loss���<"R��       �	VanDYc�A� *

loss/�<�$V9       �	��nDYc�A� *

lossa��<W���       �	_�oDYc�A� *

loss�<Q$O�       �	/pDYc�A� *

loss�c�<��|       �	q�pDYc�A� *

loss��<GTl\       �	4fqDYc�A� *

loss�;���[       �	�rDYc�A� *

lossˆ�=H�.�       �	�rDYc�A� *

loss���;�E�       �	�~sDYc�A� *

loss��<�T�1       �	DtDYc�A� *

loss;��<�nd�       �	��tDYc�A� *

loss=	J<4        �	�DuDYc�A� *

loss���<Z���       �	�uDYc�A� *

loss��3=]}"�       �	]�vDYc�A� *

loss`��=�s=�       �	JbwDYc�A� *

loss\�<���       �	��wDYc�A� *

loss��;�2H       �	i�xDYc�A� *

loss�#�=�r!       �	�_yDYc�A� *

loss�,<
��       �	��yDYc�A� *

loss\iG:v?w       �	6�zDYc�A� *

lossa]�<
�o�       �	�t{DYc�A� *

loss�u�;@��       �	�|DYc�A� *

lossc��<���       �	I}DYc�A� *

lossO�<v�U       �	0�}DYc�A� *

loss��I;fu��       �	�n~DYc�A� *

loss_c�<�J	       �	<DYc�A� *

loss{u<QO��       �	0�DYc�A� *

lossa�:��Y       �	�Y�DYc�A� *

loss9`�:�n�       �	3��DYc�A� *

lossE�<d(.       �	�DYc�A� *

loss�y�=2�6�       �	�=�DYc�A� *

lossc@�<��޸       �	�߂DYc�A� *

loss4��:�B�       �	���DYc�A� *

loss <�tW       �	�(�DYc�A� *

lossf��=��ê       �	�ĄDYc�A� *

loss��<����       �	{��DYc�A� *

loss��h<ü@$       �	k+�DYc�A� *

loss��=,[Oo       �	�ˆDYc�A� *

loss��=�S�       �	og�DYc�A� *

loss�^�<�F       �	:�DYc�A� *

loss��2<Vx�       �	8��DYc�A� *

loss���<��m�       �	Q�DYc�A� *

lossim[=�,3L       �	��DYc�A� *

loss���<_B#�       �	#��DYc�A� *

loss��O<	��       �	��DYc�A� *

loss.Ȃ<I�Z       �	�DYc�A� *

lossA�A= ��[       �	�^�DYc�A� *

loss�3= 	�T       �	F#�DYc�A� *

lossT��<��       �	��DYc�A� *

loss��>=�7�~       �	�W�DYc�A� *

loss��=��       �	���DYc�A� *

loss��=�@qv       �	҉�DYc�A� *

loss$�=��E�       �	�#�DYc�A� *

losslC�=U�+~       �	HÐDYc�A� *

loss��c<h�Q       �	al�DYc�A� *

loss�d<>i��       �	��DYc�A� *

loss���<u!�       �	���DYc�A� *

loss�7�;�Us       �	�E�DYc�A� *

loss�;Q��       �	���DYc�A� *

loss�fC<l�k�       �	��DYc�A� *

lossOo<\;.       �	ڎ�DYc�A� *

loss��=<%��	       �	�)�DYc�A� *

loss(6i=�r3       �	�ÖDYc�A� *

loss|��=#Z��       �	�Z�DYc�A� *

loss�Ԓ=r���       �	���DYc�A� *

loss�8&=
-i�       �	���DYc�A� *

loss�M	<�k��       �	�`�DYc�A� *

loss�v�;G2�       �	O�DYc�A� *

loss`��<B��       �	I��DYc�A� *

lossV�=���       �	�8�DYc�A� *

loss�<Nz�       �	�3�DYc�A� *

lossE^<�:�
       �	�՜DYc�A� *

lossC�C<M��       �	`v�DYc�A� *

loss���<Ƞ�5       �	��DYc�A� *

loss�Z<} ��       �	�!�DYc�A� *

loss�^P;���#       �	�^�DYc�A� *

loss7��<'�       �	s�DYc�A� *

loss�.�<�8�:       �	˾�DYc�A� *

loss�Z�<c�ۼ       �	�\�DYc�A� *

loss�CA=1       �	��DYc�A� *

loss�,=!�xd       �	���DYc�A� *

loss��7=�gw�       �	�L�DYc�A� *

loss8�<��q       �	��DYc�A� *

lossI!�<x�q�       �	ٵ�DYc�A� *

loss*bV<��       �	[�DYc�A� *

lossók;d_�       �	��DYc�A� *

loss?S+<����       �	hZ�DYc�A� *

lossB=�~0       �	�DYc�A� *

loss�6=@ڥ?       �	0��DYc�A� *

loss���<wL��       �	n4�DYc�A� *

losse��<b�H�       �	���DYc�A� *

loss#Ł<dN)       �	�b�DYc�A� *

loss��=J       �	r��DYc�A� *

lossZ�6=m�f�       �	��DYc�A� *

lossiIB=���       �	�!�DYc�A� *

loss��r=�[M3       �	���DYc�A� *

loss�\;�jz%       �	�a�DYc�A� *

lossC�<y��       �	��DYc�A� *

loss��?=~#       �	��DYc�A� *

loss*�<%�*�       �	l$�DYc�A� *

loss��<��#       �	T��DYc�A� *

loss��<!:�~       �	Do�DYc�A� *

loss��=;q��       �	��DYc�A� *

loss��<�`�       �	��DYc�A� *

loss�R<�k3n       �	�T�DYc�A� *

loss��M=j��       �	��DYc�A� *

loss��<x��       �	K��DYc�A� *

loss�|�<��       �	N`�DYc�A� *

lossLY<(��J       �	j��DYc�A� *

loss�]=���       �	���DYc�A�!*

loss	=j:�>       �	7Q�DYc�A�!*

loss?Tp;r�l       �	
��DYc�A�!*

loss}/<L��       �	)��DYc�A�!*

loss��<����       �	9B�DYc�A�!*

loss+�=[I*.       �	<��DYc�A�!*

lossf�<�TP       �	Bv�DYc�A�!*

loss�0�<E_׆       �	+/�DYc�A�!*

loss���;Zv�       �	\��DYc�A�!*

loss���<�(�c       �	gc�DYc�A�!*

loss㍗=�n��       �	 ��DYc�A�!*

loss�%�;��        �	w��DYc�A�!*

loss(X�<�#�       �	�8�DYc�A�!*

loss��1=�2D       �	H��DYc�A�!*

loss�z�<Mx�       �	2v�DYc�A�!*

loss���=#JHI       �	6 �DYc�A�!*

lossf�=1�MP       �	b��DYc�A�!*

loss���<�}��       �	�`�DYc�A�!*

loss�H�;�gs�       �	~�DYc�A�!*

loss�Q�;U%?`       �	$��DYc�A�!*

loss=�\.`       �	�1�DYc�A�!*

lossO4=��4       �	d��DYc�A�!*

loss��<��       �	<��DYc�A�!*

lossu�<��8       �	��DYc�A�!*

loss8k$=!vI�       �	���DYc�A�!*

loss���<r�|�       �	�Y�DYc�A�!*

lossJ��:��F        �	���DYc�A�!*

loss��T<G�o�       �	U��DYc�A�!*

loss�_"=��K       �	W"�DYc�A�!*

loss���;��       �	s��DYc�A�!*

loss&i@>
��       �	�W�DYc�A�!*

lossJ�<�<�       �	���DYc�A�!*

loss%�{;f�Bp       �	��DYc�A�!*

loss�Ʉ;~�d}       �	��DYc�A�!*

loss���<�Y4�       �	ù�DYc�A�!*

loss�k;��(       �	�P�DYc�A�!*

lossG$�=�`Z�       �	���DYc�A�!*

loss/v�<���       �	���DYc�A�!*

loss�;=��[       �	�G�DYc�A�!*

lossM�}<s�)       �	���DYc�A�!*

loss�&= |W{       �	I��DYc�A�!*

loss�͢<�e}�       �	7�DYc�A�!*

loss�wp=�a��       �	��DYc�A�!*

loss廈<�"�       �	�u�DYc�A�!*

loss��<�S&       �	AI�DYc�A�!*

loss��*=L0��       �	r��DYc�A�!*

loss�Km=G�_D       �	�}�DYc�A�!*

loss���<�P�       �	��DYc�A�!*

loss�9�<8g7       �	���DYc�A�!*

lossHΪ<��       �	xC�DYc�A�!*

loss�@�=`�IU       �	���DYc�A�!*

loss�o�<$�        �	X��DYc�A�!*

loss���;Ϫܲ       �	0�DYc�A�!*

losso<��"�       �	���DYc�A�!*

lossR�;<�s       �	%u�DYc�A�!*

loss��%<'��w       �	��DYc�A�!*

loss T�<N/�`       �	P��DYc�A�!*

loss}l�<�K�^       �	Mh�DYc�A�!*

loss�R�<@	�       �	�	�DYc�A�!*

loss��<,       �	w��DYc�A�!*

loss��<��%�       �	IK�DYc�A�!*

loss�<�ή�       �	���DYc�A�!*

loss�N</)3�       �	L��DYc�A�!*

lossx�"=H�%       �	��DYc�A�!*

losss31<)x�       �	���DYc�A�!*

loss\��<9,       �	~V�DYc�A�!*

loss�P�<��7       �	���DYc�A�!*

loss_\=O`a       �	���DYc�A�!*

loss#�!=Q��       �	��DYc�A�!*

loss��<���       �	[$�DYc�A�!*

loss$jg<�y�F       �	���DYc�A�!*

loss�G=e��7       �	�y�DYc�A�!*

loss���<M���       �	D EYc�A�!*

lossZ��=-Ǣ�       �	�� EYc�A�!*

loss��<���
       �	�EYc�A�!*

lossN"�=����       �	�;EYc�A�!*

loss��,<��o�       �	��EYc�A�!*

lossW@�;b%�       �	�EYc�A�!*

loss�k�;<f�9       �	��EYc�A�!*

loss 5%<k]�       �	2EYc�A�!*

loss�F<W8=�       �	6�EYc�A�!*

loss��<�+�       �	mpEYc�A�!*

loss�#�<�k�'       �	�EYc�A�!*

loss62�;��       �	y�EYc�A�!*

losscFu<���       �	^MEYc�A�!*

loss�E=�̡       �	��EYc�A�!*

loss�?><L-�       �	�}	EYc�A�!*

loss1-=��0�       �	"
EYc�A�!*

loss�&�<7�x       �	s�
EYc�A�!*

lossf�<���       �	1\EYc�A�!*

loss-�;!��u       �	�EYc�A�!*

loss�=�j=�       �	�EYc�A�!*

losss/
=u'��       �	a3EYc�A�!*

loss���;"�b�       �	\�EYc�A�!*

loss��
=���{       �	sEYc�A�!*

loss���</��       �	{EYc�A�!*

lossSU�;�Z��       �	��EYc�A�!*

loss�5�<�ˮ       �	TEYc�A�!*

loss�M�<	~��       �	�EYc�A�!*

loss��	<��b\       �	y�EYc�A�!*

lossr$=H���       �	�0EYc�A�!*

loss3�;Q,�n       �	�EYc�A�!*

loss�I<G(��       �	�bEYc�A�!*

loss�<.�#�       �	e�EYc�A�!*

lossu7<��j�       �	h�EYc�A�!*

loss=Z=�c��       �	T5EYc�A�!*

loss1�a<9~       �	�EYc�A�!*

loss�[�<��i       �	&nEYc�A�!*

loss�f=��       �	�EYc�A�!*

loss�ط<UY�       �	�EYc�A�!*

loss��<��Ma       �	�DEYc�A�!*

lossC�<=�PL�       �	�EYc�A�!*

loss���<�>2       �	`�EYc�A�!*

loss%9�<f%��       �	aREYc�A�!*

lossX��;�Q��       �	z EYc�A�!*

lossx�h=��k�       �	��EYc�A�!*

loss�k<,�?�       �	1EYc�A�!*

loss�o;�
ld       �	1�EYc�A�!*

loss�U3<�D�#       �	BzEYc�A�!*

loss
j�<�1#       �	EYc�A�!*

lossF��<�P       �	��EYc�A�!*

loss�=�U��       �	{OEYc�A�!*

loss�+<D�x�       �	QL EYc�A�!*

loss��=��ٟ       �	�!EYc�A�!*

loss:% =y�,       �	�A"EYc�A�!*

loss���;ͷ�\       �	)Y#EYc�A�!*

loss�e='�<       �	�S$EYc�A�!*

loss�/�;oF�@       �	GZ%EYc�A�!*

loss���<�3�L       �	��%EYc�A�"*

loss�=�=��'`       �	��&EYc�A�"*

loss1�=/,g�       �	|H'EYc�A�"*

lossD� =xĸ       �	V�'EYc�A�"*

lossH��;��}p       �	��(EYc�A�"*

loss=�C<D�mJ       �	�D)EYc�A�"*

loss�N(=�I��       �	��)EYc�A�"*

losswHd=ak�E       �	�w*EYc�A�"*

loss�<�a ?       �	�+EYc�A�"*

lossXF=���F       �	B�+EYc�A�"*

loss|H�;���       �	rN,EYc�A�"*

lossϴ�<�ī�       �	-�,EYc�A�"*

loss��z;�Ώ�       �	��-EYc�A�"*

loss��<�"��       �	�o.EYc�A�"*

loss7c�<�ij        �	�^/EYc�A�"*

loss�zh< -�;       �	H�/EYc�A�"*

loss�f]=õ�       �	3�0EYc�A�"*

lossQ�<����       �	)1EYc�A�"*

lossc0<D��       �	z�1EYc�A�"*

loss�^�=rTɝ       �	J`2EYc�A�"*

lossmۙ<N�{�       �	W3EYc�A�"*

loss���<}Ґ�       �	��3EYc�A�"*

lossϖ�<�.�k       �	�B4EYc�A�"*

lossd�;�6X)       �	z�4EYc�A�"*

losss_}<b]�       �	N�5EYc�A�"*

lossAZ�<O�w       �	�56EYc�A�"*

loss-�=��2.       �	�7EYc�A�"*

loss�,�<�4�       �	��8EYc�A�"*

loss��=]k �       �	5*9EYc�A�"*

loss�k;��`�       �	<�9EYc�A�"*

lossR�=��LB       �	�R:EYc�A�"*

loss�~�;K�1       �	f�:EYc�A�"*

loss�� =4�}       �	i9<EYc�A�"*

loss���;�߃�       �	h#=EYc�A�"*

loss��<#��       �	7 >EYc�A�"*

loss�=f&5       �	c�>EYc�A�"*

loss�Q<�-��       �	�?EYc�A�"*

loss2�9=���       �	� @EYc�A�"*

loss���;����       �	�@EYc�A�"*

loss��=qL�p       �	�TAEYc�A�"*

loss?�<��&       �	. BEYc�A�"*

loss��<@j�       �	|�BEYc�A�"*

loss��q<�ȻO       �	�6CEYc�A�"*

loss8T+;���       �	��CEYc�A�"*

loss��=�^c�       �	tDEYc�A�"*

loss��;��2�       �	�EEYc�A�"*

lossi�=ä�       �	�EEYc�A�"*

loss[;�<GmG�       �	�>FEYc�A�"*

loss�<���       �	Y�FEYc�A�"*

loss��S<:U�p       �	 �GEYc�A�"*

lossm�b<%�}�       �	�6HEYc�A�"*

lossϞ<�H-       �	��HEYc�A�"*

loss��)=�e��       �	zIEYc�A�"*

loss��R<*:O       �	h JEYc�A�"*

lossaH0=�G�h       �	�JEYc�A�"*

loss	=|�Ť       �	�eKEYc�A�"*

loss�	�=޴ѓ       �	-LEYc�A�"*

loss��=��^�       �	^�LEYc�A�"*

loss���<&�Ԗ       �	caMEYc�A�"*

loss�V=���R       �	��MEYc�A�"*

loss��Z<"��       �	��NEYc�A�"*

loss�E�<_o�       �	f�OEYc�A�"*

lossC�P<6�˼       �	iPEYc�A�"*

losso�$=0�       �	��PEYc�A�"*

loss�ZP=w_��       �	D�QEYc�A�"*

lossJ��<�X�       �	t^REYc�A�"*

loss�s=?Zr1       �	�REYc�A�"*

lossa8�=:�u�       �	��SEYc�A�"*

loss�K�<[�kw       �	wNTEYc�A�"*

loss��<��       �	�TEYc�A�"*

loss�=**e       �	��UEYc�A�"*

loss���<���       �	tAVEYc�A�"*

loss��=�A��       �	��VEYc�A�"*

loss'�<�m��       �	�WEYc�A�"*

loss��O=t�;�       �	�UXEYc�A�"*

lossSx�<�~\�       �	Z�XEYc�A�"*

losso�	=&6�       �	�YEYc�A�"*

loss��=7	�L       �	w�ZEYc�A�"*

loss �%=D�ʵ       �		[EYc�A�"*

loss�_<�Z�       �	H�[EYc�A�"*

loss��!;��t       �	Z\EYc�A�"*

lossЋ<���       �	��\EYc�A�"*

lossF�=�V�Y       �	�]EYc�A�"*

loss;��<B�ڎ       �	�1^EYc�A�"*

loss=��'�       �	��^EYc�A�"*

loss��=�fl�       �	vl_EYc�A�"*

loss)a�<��`%       �	Hk`EYc�A�"*

lossV�<=��kH       �	�aEYc�A�"*

lossd&�;��b       �	֭aEYc�A�"*

loss��:<�E��       �	QLbEYc�A�"*

loss���<ܕ�)       �	A(cEYc�A�"*

loss�>,=�c       �	��cEYc�A�"*

lossݲ<:2s�       �	4idEYc�A�"*

loss$�=M�;       �	MeEYc�A�"*

lossMrZ<@�       �	6�eEYc�A�"*

loss�=����       �	�RfEYc�A�"*

loss 9�;���p       �	
�fEYc�A�"*

loss:U=�(0�       �	�gEYc�A�"*

lossj2<���       �	�0hEYc�A�"*

loss���<Qڵ�       �	��hEYc�A�"*

lossm.�<r��       �	V�iEYc�A�"*

losse�?<:,"�       �	AjEYc�A�"*

lossT��;�r�X       �	��jEYc�A�"*

loss���;X��       �	�kEYc�A�"*

loss�r�;+��       �	�%lEYc�A�"*

loss�/�<o��       �	¾lEYc�A�"*

losse�<���       �	�\mEYc�A�"*

loss���;k��       �	��mEYc�A�"*

loss�MT=�*�       �	S�nEYc�A�"*

loss1^F=�(�       �	U1oEYc�A�"*

loss%�r=�       �	(pEYc�A�"*

loss�8=P�9T       �	�pEYc�A�"*

loss��<r�,       �	JbqEYc�A�"*

loss�Bz=1�҆       �	}�rEYc�A�"*

lossF�;�IX�       �	�/sEYc�A�"*

loss��<��1�       �	�sEYc�A�"*

loss�(=�H~       �	=|tEYc�A�"*

loss�-#=TQ��       �	K"uEYc�A�"*

loss�=!A�[       �	>�uEYc�A�"*

loss%�=�a��       �	YmvEYc�A�"*

loss�"=}�8�       �	�
wEYc�A�"*

lossE�<�ޣ>       �	��wEYc�A�"*

loss�{�<8���       �	�PxEYc�A�"*

loss��W=hfI       �	�yEYc�A�"*

loss��=lK       �	B�yEYc�A�"*

lossE��;���       �	��zEYc�A�"*

lossz.�<��(�       �	B{EYc�A�"*

lossݍ�=B��h       �	n�{EYc�A�"*

loss��=C�A�       �	5z|EYc�A�#*

lossô�;u�|z       �	}EYc�A�#*

loss�S�;fk�       �	��}EYc�A�#*

lossՐ�<K�       �	|D~EYc�A�#*

lossTI=e���       �	7�~EYc�A�#*

loss$m�<X��       �	c�EYc�A�#*

loss�=��&�       �	U4�EYc�A�#*

lossģB=. �g       �	-̀EYc�A�#*

loss�<��z&       �	$��EYc�A�#*

lossy�<���       �	��EYc�A�#*

loss!��;�D�       �	ܸ�EYc�A�#*

loss���:�I�/       �	�a�EYc�A�#*

loss�<���!       �	X �EYc�A�#*

lossT5�;A�8�       �	V��EYc�A�#*

lossA�D;S5�S       �	�7�EYc�A�#*

loss��B<'@!�       �	_ӅEYc�A�#*

loss�E=�#�[       �	�m�EYc�A�#*

loss\��<S��       �	��EYc�A�#*

loss�8=��=\       �	^��EYc�A�#*

loss�==��       �	Y�EYc�A�#*

loss�>�<i��s       �	8�EYc�A�#*

loss�c�<����       �	ꮉEYc�A�#*

lossI=^�j       �	�L�EYc�A�#*

loss�M<��       �	T�EYc�A�#*

lossE�=�_�R       �	��EYc�A�#*

loss��@= ��       �	�6�EYc�A�#*

loss�w�<kM�       �	�ьEYc�A�#*

lossE��;*y�       �		��EYc�A�#*

loss58=}���       �	p'�EYc�A�#*

lossT =���=       �	#��EYc�A�#*

loss�Ja=� �N       �	}[�EYc�A�#*

lossW�e<~�H�       �	���EYc�A�#*

loss��<J�+       �	���EYc�A�#*

loss�A�<Yy��       �	�l�EYc�A�#*

lossFA<j�J!       �	��EYc�A�#*

loss��;��/       �	��EYc�A�#*

loss`B�=��*X       �	�<�EYc�A�#*

loss�G�;ďh�       �	ZՓEYc�A�#*

loss��<i�8       �	o�EYc�A�#*

loss��o=o��`       �	�=�EYc�A�#*

loss~� ==�U�       �	^ڕEYc�A�#*

loss�%=��y*       �	�t�EYc�A�#*

loss�8==;�@�       �	EYc�A�#*

lossa�=Ȃ-�       �	#/�EYc�A�#*

loss��<�b�       �	q˘EYc�A�#*

loss�sK<e^w       �	Ve�EYc�A�#*

loss�=q���       �	�,�EYc�A�#*

loss��h<u�X       �	�ʚEYc�A�#*

lossq.~=�!�       �	~p�EYc�A�#*

lossK�=Uaj�       �	W��EYc�A�#*

loss}�C=���       �	�G�EYc�A�#*

loss�u<�+��       �	��EYc�A�#*

loss�E�<J�	R       �	���EYc�A�#*

loss <���       �	y!�EYc�A�#*

loss�D<����       �	�EYc�A�#*

loss�Ǩ=�g��       �	�b�EYc�A�#*

loss�I�<��c       �	��EYc�A�#*

loss,i=�n.X       �	��EYc�A�#*

lossT��<�Ů;       �	���EYc�A�#*

loss�z�=s_M�       �	���EYc�A�#*

loss��;�;N+       �	�^�EYc�A�#*

loss&��;����       �	#L�EYc�A�#*

loss�h�<?!       �	�l�EYc�A�#*

loss{C�=�h       �	)Y�EYc�A�#*

lossDV!<�S�       �	HQ�EYc�A�#*

loss�h�=��x�       �	X�EYc�A�#*

lossI>K=l��a       �	M��EYc�A�#*

lossX<@o8       �	�<�EYc�A�#*

lossì4=�P=       �	��EYc�A�#*

lossؑ1=���       �	�EYc�A�#*

loss��< �       �	�.�EYc�A�#*

lossT�E<���       �	]��EYc�A�#*

loss��_<�"c       �	���EYc�A�#*

loss�y&<��P       �	6W�EYc�A�#*

loss2j�<����       �	���EYc�A�#*

loss�2=m�D       �	���EYc�A�#*

loss�e<bs,       �	0f�EYc�A�#*

loss���<1fm       �	r��EYc�A�#*

loss��;y'"       �	i��EYc�A�#*

loss��z<	�       �	�@�EYc�A�#*

lossXck<i�Od       �	�سEYc�A�#*

loss��=%�g       �	�m�EYc�A�#*

loss=��<�4�"       �	�>�EYc�A�#*

lossf�<<���       �	JԵEYc�A�#*

loss!��=w�"}       �	�m�EYc�A�#*

loss:R =k�N       �	��EYc�A�#*

loss�{�<0@+�       �	ͷEYc�A�#*

loss��&=n�       �	�w�EYc�A�#*

loss-
2<�c       �	 o�EYc�A�#*

loss?��;�|�b       �	�EYc�A�#*

lossDa�;mBI#       �	ʺEYc�A�#*

lossz��<�#��       �	-��EYc�A�#*

loss��F<3�p�       �	,,�EYc�A�#*

lossj�<�N       �	�EYc�A�#*

loss��<U"��       �	۾�EYc�A�#*

loss�,�<�G�R       �	Qk�EYc�A�#*

loss�F�<�=�S       �	ǜ�EYc�A�#*

loss�fp=A_��       �	z5�EYc�A�#*

loss-7�<纡       �	���EYc�A�#*

loss��=��X       �	mr�EYc�A�#*

loss��w<�C��       �	��EYc�A�#*

loss�S�<𞚙       �	���EYc�A�#*

lossP�<��#z       �	�I�EYc�A�#*

lossҝc<�MA�       �	���EYc�A�#*

lossh��<�P?       �	���EYc�A�#*

loss���;�]8|       �	�!�EYc�A�#*

loss�@N<G	��       �	k��EYc�A�#*

loss��;=���       �	Ϡ�EYc�A�#*

lossn��<����       �	�>�EYc�A�#*

lossC�;Ļp�       �	���EYc�A�#*

loss��<n)��       �	�}�EYc�A�#*

lossR�c<ȹJ8       �	&�EYc�A�#*

lossؘ=-��       �	s��EYc�A�#*

loss|�<��n       �	rS�EYc�A�#*

loss$�<��;       �	���EYc�A�#*

loss�r=���        �	I��EYc�A�#*

loss��=��	,       �	W#�EYc�A�#*

loss�$;G@       �	���EYc�A�#*

lossV݅<נ�n       �	��EYc�A�#*

loss�0<Ȇ�       �	
+�EYc�A�#*

loss���=$̛U       �	���EYc�A�#*

loss��=�еA       �	g�EYc�A�#*

loss>�<ɯS�       �	-�EYc�A�#*

loss��<X�N       �	���EYc�A�#*

loss�p�;}`�       �	$G�EYc�A�#*

loss���<og�       �	��EYc�A�#*

loss��<�Vg�       �	�}�EYc�A�#*

lossȂ=׉�o       �	*�EYc�A�#*

loss�4�=޽�c       �	���EYc�A�$*

loss�o�<�:       �	wN�EYc�A�$*

lossh{`=�I(g       �	���EYc�A�$*

loss&�T=L��       �	3��EYc�A�$*

loss e+=@�ª       �	� �EYc�A�$*

loss���<U�1       �	I��EYc�A�$*

loss�w�= �       �	eU�EYc�A�$*

lossa�<��d�       �	���EYc�A�$*

loss$ǝ<PaԎ       �	Ɗ�EYc�A�$*

loss=<����       �	�%�EYc�A�$*

lossQq�;�0ɭ       �	s��EYc�A�$*

loss�c<�        �	�g�EYc�A�$*

loss�YY=���d       �	�EYc�A�$*

lossR��<R�Sb       �	���EYc�A�$*

loss~==�UR       �	S>�EYc�A�$*

lossH��;�o]�       �	{��EYc�A�$*

loss[V�=�y��       �	Sw�EYc�A�$*

loss���<����       �	��EYc�A�$*

loss�.�=;�ň       �	ɯ�EYc�A�$*

loss]��<��b       �	�L�EYc�A�$*

loss4 D=�l�       �	y��EYc�A�$*

loss�r�<}v;       �	���EYc�A�$*

lossI�5=�W       �	�q�EYc�A�$*

loss%=ׄ�       �	?�EYc�A�$*

loss�Vu=���8       �	��EYc�A�$*

loss��<|d�       �	�d�EYc�A�$*

lossכM<\�l�       �	�EYc�A�$*

loss��I=���       �	���EYc�A�$*

lossa	�<�m؛       �	w�EYc�A�$*

lossA<�A�f       �	?�EYc�A�$*

loss�KW<m�	z       �	���EYc�A�$*

loss��<�KzX       �	�\�EYc�A�$*

loss!�s<iPg�       �	E��EYc�A�$*

loss���<B���       �	��EYc�A�$*

lossz�x=:/��       �	2�EYc�A�$*

loss�N	=6��       �	���EYc�A�$*

lossR��<x)Ɨ       �	�k�EYc�A�$*

loss�=�z�       �	\�EYc�A�$*

lossX�<4r[<       �	���EYc�A�$*

loss�=l��?       �	�7�EYc�A�$*

lossZ@^<^Cd       �	���EYc�A�$*

lossc��<�
�       �	�y�EYc�A�$*

loss�|;<��W       �	��EYc�A�$*

loss��4={��       �	D��EYc�A�$*

loss��<�`�q       �	��EYc�A�$*

loss�v�;]��       �	.�EYc�A�$*

lossa#=�*�=       �	���EYc�A�$*

loss P=4�A�       �	�b�EYc�A�$*

loss�%\=H�+X       �	���EYc�A�$*

lossS.F<�MfS       �	���EYc�A�$*

loss�F�<�ה       �	2Y�EYc�A�$*

loss -�<z2]D       �	���EYc�A�$*

lossx��<³2�       �	���EYc�A�$*

loss1�L<���       �	e6�EYc�A�$*

loss<cm<��q�       �	���EYc�A�$*

loss�K�;�OK�       �	?s�EYc�A�$*

loss{O�<D��S       �	��EYc�A�$*

loss���<q$T       �	B��EYc�A�$*

lossPp�<]O�       �	�[�EYc�A�$*

lossv�m<�#��       �	��EYc�A�$*

loss���;� �       �	R��EYc�A�$*

loss
�|<�׆�       �	��EYc�A�$*

loss��t;�si�       �	���EYc�A�$*

loss(�= �'N       �	��EYc�A�$*

loss��s=�e       �	�p�EYc�A�$*

loss�o�<S��       �	��EYc�A�$*

loss�!="!�        �	5��EYc�A�$*

loss�>F<��n       �	eQ�EYc�A�$*

loss
N�;V]Z       �	�- FYc�A�$*

loss�c;����       �	�� FYc�A�$*

loss���<�7�       �	(aFYc�A�$*

loss;=�>�       �	��FYc�A�$*

loss�:�=y��       �	�FYc�A�$*

loss��=blL�       �	�?FYc�A�$*

loss�=��<�       �	��FYc�A�$*

loss���;sʱ       �	��FYc�A�$*

loss{��<i,t�       �		FYc�A�$*

loss�^7=q���       �	�FYc�A�$*

loss��<K7M�       �	*TFYc�A�$*

loss�'�<�Rq$       �	DFYc�A�$*

lossQq	=�!��       �	j�FYc�A�$*

loss�,�;�z�$       �	hXFYc�A�$*

loss��>=���       �	��FYc�A�$*

loss��'=�~*�       �	��	FYc�A�$*

loss�RY<�       �	�
FYc�A�$*

loss̭�;��P       �	��
FYc�A�$*

loss�'=(�vR       �	^iFYc�A�$*

loss\�g=h�d       �	FYc�A�$*

lossV�<��K       �	v�FYc�A�$*

loss_7�<{��       �	tBFYc�A�$*

loss��/=~�E�       �	��FYc�A�$*

lossI�Y<~�r�       �	CqFYc�A�$*

loss�tl<>i       �	�
FYc�A�$*

loss?�<���       �	�FYc�A�$*

loss�
�<�� n       �	�?FYc�A�$*

lossV��<��%}       �	��FYc�A�$*

loss=��#       �	�rFYc�A�$*

loss�{=��	�       �	�FYc�A�$*

loss:T�;�a��       �	ҧFYc�A�$*

lossq
<��|&       �	�<FYc�A�$*

lossC��;E�{�       �	�FYc�A�$*

loss�	8<���I       �	pwFYc�A�$*

loss��=���       �	�NFYc�A�$*

loss��<A���       �	��FYc�A�$*

loss�2�<؟4       �	��FYc�A�$*

loss4�<<d��N       �	W!FYc�A�$*

loss-�~=�K�d       �	��FYc�A�$*

loss�r=�vIF       �	R`FYc�A�$*

loss
�<L�B       �	��FYc�A�$*

loss�*�<278       �	��FYc�A�$*

loss�ݕ<p���       �	�>FYc�A�$*

loss�<
�8�       �	��FYc�A�$*

lossAW^<��x       �	Q�FYc�A�$*

lossM��;�[6       �	fFYc�A�$*

lossZN�<Q��n       �	0�FYc�A�$*

loss�*B;�|a-       �	�ZFYc�A�$*

loss.zQ<~dK       �	]�FYc�A�$*

loss�]=�He       �	��FYc�A�$*

loss��<A�       �	N� FYc�A�$*

loss��<F��       �	�h!FYc�A�$*

lossRN<Iv�x       �	"FYc�A�$*

loss��=�[       �	��"FYc�A�$*

loss���;v��       �	��#FYc�A�$*

loss͆�;��.�       �	I�$FYc�A�$*

loss�3�<��o'       �	��%FYc�A�$*

loss�)<<P��       �	_\&FYc�A�$*

loss�mN;��       �	vR'FYc�A�$*

loss]�;?�Y       �	#(FYc�A�$*

loss��<V?$       �	��(FYc�A�%*

lossi]�;�)�       �	-^)FYc�A�%*

lossl�<��0       �	b�*FYc�A�%*

loss��~;�1�9       �	�o+FYc�A�%*

lossS�;�0��       �	�,FYc�A�%*

loss YO;����       �	�W-FYc�A�%*

lossLS�94ŕ	       �	�>.FYc�A�%*

loss��9�P��       �	��.FYc�A�%*

loss�|�:�C.�       �	��/FYc�A�%*

loss��&<Dx��       �	eQ0FYc�A�%*

loss�DR<�Ir�       �	��0FYc�A�%*

loss��:���       �	��1FYc�A�%*

lossv�4;3��       �	�B2FYc�A�%*

loss�k�=���       �	�*3FYc�A�%*

loss ��:��       �	�3FYc�A�%*

lossi<�<S�%�       �	��4FYc�A�%*

loss!V�<n)��       �	C=5FYc�A�%*

lossH28=��       �	��5FYc�A�%*

losstW	=�n�l       �	�y6FYc�A�%*

loss3�m<�]O�       �	�7FYc�A�%*

lossR��<y�eY       �	��7FYc�A�%*

loss#==�?Y�       �	�8FYc�A�%*

loss�H�;�&B       �	379FYc�A�%*

lossz�<T�Њ       �	��9FYc�A�%*

loss�g�;�3�W       �	�q:FYc�A�%*

loss���=� b�       �	�;FYc�A�%*

lossS3I=5��       �	��;FYc�A�%*

loss��<l�^�       �	!W<FYc�A�%*

loss�(`<D�&       �	M�<FYc�A�%*

loss��g=z�s       �	�=FYc�A�%*

losse�<G<�S       �	Z->FYc�A�%*

lossH�;���       �	��>FYc�A�%*

loss�R<':%       �	�?FYc�A�%*

loss���;��       �	%Y@FYc�A�%*

loss��R;�yCI       �	JAFYc�A�%*

loss�	e<�>�       �	��AFYc�A�%*

loss�
M<�D       �	�XBFYc�A�%*

loss�3	<�'�       �	~CFYc�A�%*

lossTI7;v7KH       �	G�CFYc�A�%*

lossc�<c��p       �	�QDFYc�A�%*

lossv��<G���       �	G!EFYc�A�%*

loss�!<�>�       �	 �EFYc�A�%*

loss=f�<���       �	�fFFYc�A�%*

loss�Uq=�:�       �	�GFYc�A�%*

lossӵ�<�s$�       �	ĲGFYc�A�%*

loss-"=4I       �	OXHFYc�A�%*

loss|T�;���Z       �	.IFYc�A�%*

loss���;�]q       �	�IFYc�A�%*

loss�w�<�_l�       �	PJFYc�A�%*

loss{��<�k��       �	��JFYc�A�%*

loss�;��       �	P�KFYc�A�%*

lossC<�;��|�       �	��LFYc�A�%*

loss�{�<l�^�       �	I-MFYc�A�%*

loss��<��o       �	��MFYc�A�%*

loss�<a{.[       �	�sNFYc�A�%*

lossh�H<��1m       �	^OFYc�A�%*

loss=�<�Z�Z       �	��OFYc�A�%*

lossq�B=PS�S       �	�PFYc�A�%*

loss��;ISN       �	TQFYc�A�%*

loss� �<Y�8&       �	�QFYc�A�%*

loss�7�<l���       �	��RFYc�A�%*

loss!u�<'Z       �	{/SFYc�A�%*

lossݩ�<ُ[�       �	��SFYc�A�%*

loss���;P�%       �	7oTFYc�A�%*

loss�B�;��Bn       �	(
UFYc�A�%*

loss���=C7�f       �	LmFYc�A�%*

loss��=\|       �	�mFYc�A�%*

loss��= ;�        �	�OnFYc�A�%*

loss7W�<˥�       �	��nFYc�A�%*

loss�u�<�<-D       �	�oFYc�A�%*

loss$8=[��9       �	xcpFYc�A�%*

loss�8�<�.;�       �	. qFYc�A�%*

lossr�=���       �	W�qFYc�A�%*

loss�=�r�       �	5rFYc�A�%*

loss�,�=@��       �	e�rFYc�A�%*

loss��;�*f       �	�ysFYc�A�%*

lossgW<{�8�       �	�tFYc�A�%*

loss���=�ߕ�       �	_�tFYc�A�%*

lossFe=<o�K       �	�buFYc�A�%*

lossl�<_Zi}       �	RvFYc�A�%*

loss�˞<�A�       �	3�vFYc�A�%*

loss{7%;�Ė       �	ZcwFYc�A�%*

loss��/<�א       �	L�wFYc�A�%*

loss<7�<��Xr       �	�xFYc�A�%*

loss���=��       �	�=yFYc�A�%*

loss�Uj<�s�       �	��yFYc�A�%*

loss���<[�g       �	zFYc�A�%*

lossN�;�pt       �	�({FYc�A�%*

loss=~�=<9�8       �	;|FYc�A�%*

lossm��;P�yj       �	^}FYc�A�%*

loss�̒<d�7-       �	��}FYc�A�%*

loss�=�d�3       �	�U~FYc�A�%*

lossj�*=vW�       �	MFYc�A�%*

loss�=��s�       �	F�FYc�A�%*

lossd8�<B��       �	��FYc�A�%*

loss_j�<?�       �	���FYc�A�%*

lossa�<��^L       �	5�FYc�A�%*

loss���;!�$�       �	�܂FYc�A�%*

loss���=	�       �	���FYc�A�%*

loss��<��iY       �	��FYc�A�%*

loss��=�FS�       �	�[�FYc�A�%*

loss���;��       �	��FYc�A�%*

loss2P="?@�       �	1φFYc�A�%*

loss=��[�       �	|�FYc�A�%*

lossӀ=����       �	W>�FYc�A�%*

loss��|<.-��       �	'ވFYc�A�%*

loss�v�<V��       �	�~�FYc�A�%*

loss��;<�/E�       �	;�FYc�A�%*

lossLb=��n�       �	k׊FYc�A�%*

loss�V='؎�       �	�s�FYc�A�%*

loss��<��-       �	n�FYc�A�%*

loss� �<:ܰ       �	=֌FYc�A�%*

loss��=�co       �	�u�FYc�A�%*

loss_W�<-�       �	��FYc�A�%*

lossmF:�R�t       �	���FYc�A�%*

loss̄<<���N       �	�G�FYc�A�%*

loss�s�<�ȼ       �	v�FYc�A�%*

lossdH<2�f�       �	V��FYc�A�%*

loss���=Ø��       �	�FYc�A�%*

loss!�<�n5       �	�FYc�A�%*

loss��E;I�8U       �	V�FYc�A�%*

lossVEw;�l�       �	M��FYc�A�%*

lossHA�:���G       �	R��FYc�A�%*

loss�#<S=Y=       �	9c�FYc�A�%*

loss�E�<P�j       �	��FYc�A�%*

lossS{�<Yaw       �	ß�FYc�A�%*

loss�<gG��       �	8I�FYc�A�%*

loss��|;�s8�       �	��FYc�A�%*

loss��<_?�       �	���FYc�A�&*

lossl��<	c �       �	�2�FYc�A�&*

loss_�b<i��       �	-јFYc�A�&*

lossd3<��s       �	pz�FYc�A�&*

loss۔�<��j       �	��FYc�A�&*

loss�=���       �	��FYc�A�&*

loss��=т��       �	�O�FYc�A�&*

loss���<Dٛ       �	�FYc�A�&*

loss��<���       �	���FYc�A�&*

loss��<A���       �	�"�FYc�A�&*

lossZ&6=�S4�       �	���FYc�A�&*

loss8�=Yqpy       �	Փ�FYc�A�&*

loss;H�;1l       �	Y0�FYc�A�&*

loss��L=$�}�       �	{ןFYc�A�&*

loss���<�8�       �	s�FYc�A�&*

loss)Xq<��       �	��FYc�A�&*

loss&X�<�Ӟ�       �	�FYc�A�&*

lossݿ=�h�~       �	��FYc�A�&*

loss	�<:���       �	;��FYc�A�&*

loss�D<Op�s       �	wj�FYc�A�&*

loss�?|<惏h       �	�h�FYc�A�&*

loss{�w<h���       �	�1�FYc�A�&*

loss�ap<�Fم       �	&:�FYc�A�&*

loss�<��R�       �	�J�FYc�A�&*

loss���<�l       �	��FYc�A�&*

loss��7=�}�       �	��FYc�A�&*

loss.:�;~�<�       �	䃪FYc�A�&*

loss���=�X2�       �	
/�FYc�A�&*

lossIH�<�l��       �	d�FYc�A�&*

loss�<�K�       �	F(�FYc�A�&*

loss
�{<�,z       �	�ڭFYc�A�&*

loss��1=F�'5       �	Ǹ�FYc�A�&*

loss,<�й�       �	$d�FYc�A�&*

lossjq�<�}��       �	OX�FYc�A�&*

loss�X(=]!��       �	��FYc�A�&*

loss��K=�oF#       �	竱FYc�A�&*

loss�$=<����       �	7O�FYc�A�&*

loss���;P�|       �	n��FYc�A�&*

losshv;_3�       �	���FYc�A�&*

loss��2=�k�       �	�x�FYc�A�&*

lossJ�;g�7       �	�.�FYc�A�&*

loss_�><��       �	�͵FYc�A�&*

lossJ��<G�i       �	�g�FYc�A�&*

loss��j;�       �	h�FYc�A�&*

lossO�<��i+       �	���FYc�A�&*

loss�4=�'�/       �	N�FYc�A�&*

loss��Q<w��?       �	���FYc�A�&*

loss`�H=�s}       �	ѐ�FYc�A�&*

loss���<�-Ɗ       �	���FYc�A�&*

loss��U=Ґ[?       �	w1�FYc�A�&*

loss]W<Ɂ��       �	!˻FYc�A�&*

lossQ�<��>       �	�d�FYc�A�&*

lossm�;�0��       �	#�FYc�A�&*

loss���;)�       �	྽FYc�A�&*

loss�.=� ��       �	�d�FYc�A�&*

loss��<K�\�       �	��FYc�A�&*

loss���<�5�y       �	��FYc�A�&*

loss��Q<!%�       �	h@�FYc�A�&*

loss��6=���       �	n��FYc�A�&*

lossD
�;���       �	���FYc�A�&*

lossD�=��+�       �	"�FYc�A�&*

loss�1�<W[�T       �	���FYc�A�&*

loss��=�+�       �	nh�FYc�A�&*

loss�6y=��g^       �	��FYc�A�&*

loss�	�;T7       �	P��FYc�A�&*

loss�>=Γ�|       �	��FYc�A�&*

loss�l�<DNh       �	�8�FYc�A�&*

lossu�;`B��       �	��FYc�A�&*

lossgw=�#��       �	Pq�FYc�A�&*

loss���<���       �	9�FYc�A�&*

loss�N�<��,_       �	���FYc�A�&*

lossa.�<We�>       �	�D�FYc�A�&*

loss�A�<�Od       �	���FYc�A�&*

loss!;?��o       �	Q��FYc�A�&*

lossX=F|�       �	�E�FYc�A�&*

loss�;<�"p       �	�S�FYc�A�&*

loss��<���       �	���FYc�A�&*

loss���;�yJ�       �	e��FYc�A�&*

losseE<W]��       �	�(�FYc�A�&*

loss)��<CO�       �	G��FYc�A�&*

loss��m<&$�       �	�i�FYc�A�&*

lossz�6=��C!       �	m�FYc�A�&*

loss���<����       �	�
�FYc�A�&*

loss��?<^���       �	���FYc�A�&*

loss{�;��m�       �	y?�FYc�A�&*

loss�9�;��j       �	���FYc�A�&*

loss�
�<4KS�       �	��FYc�A�&*

loss܄�<�=4       �	���FYc�A�&*

loss7 �<M�iJ       �	Id�FYc�A�&*

loss$�<D�r�       �	�
�FYc�A�&*

loss<�F=���       �	H��FYc�A�&*

loss2��<���       �	�a�FYc�A�&*

lossۄ�;��2       �	P �FYc�A�&*

loss���;*h�       �	���FYc�A�&*

lossze<���       �	�;�FYc�A�&*

loss���<U��       �	���FYc�A�&*

loss�ס<��       �	C��FYc�A�&*

loss�l=�H��       �	�(�FYc�A�&*

loss��n;��_U       �	{��FYc�A�&*

loss|<UZ�7       �	�r�FYc�A�&*

loss�ǉ;���       �	Z�FYc�A�&*

loss��=�w)�       �	���FYc�A�&*

loss$�;8wK       �	��FYc�A�&*

lossV�;�rd       �	34�FYc�A�&*

loss�9=�j�x       �	I��FYc�A�&*

loss"�;��q       �	�u�FYc�A�&*

loss��O<mg       �	��FYc�A�&*

loss��E<���(       �	#��FYc�A�&*

loss�j3</%wV       �	��FYc�A�&*

loss�:�;�,NT       �	(b�FYc�A�&*

loss�:U=�]�e       �	���FYc�A�&*

loss=�8<CA       �	؜�FYc�A�&*

loss � =WU�       �	u?�FYc�A�&*

lossD�Y<���       �	0��FYc�A�&*

loss/|0=%�       �	�t�FYc�A�&*

lossy� =¤��       �	��FYc�A�&*

loss@}<g�t       �	���FYc�A�&*

loss��;�D&m       �	�C�FYc�A�&*

loss�F=^�!�       �	���FYc�A�&*

losss�s<��       �	�t�FYc�A�&*

losst��;v#       �	��FYc�A�&*

loss�}�;]���       �	���FYc�A�&*

loss��#=��X�       �	Zf�FYc�A�&*

losst��< ���       �	A�FYc�A�&*

loss�g<"B       �	���FYc�A�&*

loss� =�:��       �	w��FYc�A�&*

loss�2=       �	�!�FYc�A�&*

lossiH=��|       �	8��FYc�A�&*

loss��=�|�       �	�Z�FYc�A�'*

loss�;�F!�       �	��FYc�A�'*

loss26�;��       �	��FYc�A�'*

loss��;��<       �	�>�FYc�A�'*

loss��=�b��       �	���FYc�A�'*

lossM�;N�f�       �	}x�FYc�A�'*

lossNx8=���       �	`!�FYc�A�'*

loss�=�{�s       �	���FYc�A�'*

loss@�;RX�>       �	�X�FYc�A�'*

lossG�<>\       �	���FYc�A�'*

loss�{<��4�       �	ē�FYc�A�'*

loss��<B�       �	�,�FYc�A�'*

loss��<SO�       �	���FYc�A�'*

loss�ֆ<��i�       �	���FYc�A�'*

lossE��<1�G�       �	�\�FYc�A�'*

loss�zO<B�	       �	E��FYc�A�'*

loss�\=����       �	���FYc�A�'*

loss��<��Q       �	(�FYc�A�'*

lossx�y=O��       �	���FYc�A�'*

loss�y�<Y�<�       �	a��FYc�A�'*

loss&m�<=��       �	B%�FYc�A�'*

loss��f;\G�U       �	8��FYc�A�'*

loss��;�Rp<       �	�W�FYc�A�'*

loss�d=����       �	 ��FYc�A�'*

loss�F=�h�%       �	���FYc�A�'*

loss��C=��*       �	;5�FYc�A�'*

loss��1=�F5�       �	-��FYc�A�'*

loss4vR=��       �	�f GYc�A�'*

loss�+�<����       �	z� GYc�A�'*

loss��<�h��       �	̚GYc�A�'*

loss,��<�m,       �	�zGYc�A�'*

loss= ?=n�H�       �	�GYc�A�'*

lossH�B=\S�       �	�GYc�A�'*

lossfV�;�B�y       �	JFGYc�A�'*

loss,��<�J�       �	v�GYc�A�'*

loss慺<Pٛ�       �	�vGYc�A�'*

loss|*�<�'D�       �	�GYc�A�'*

loss���=v���       �	�GYc�A�'*

loss`��<I��       �	<GYc�A�'*

loss��,<A�>�       �	0�GYc�A�'*

loss�ƨ;c@�H       �	~�GYc�A�'*

loss_��=�J��       �	�{	GYc�A�'*

loss���<-��y       �	�
GYc�A�'*

lossJ��<Amn       �	=�
GYc�A�'*

loss6~X=6�W\       �	��GYc�A�'*

loss�e�=~	e       �	y?GYc�A�'*

loss��h=�i�       �	��GYc�A�'*

loss�~=�v�)       �	�GYc�A�'*

loss�M�;�$}O       �	�GYc�A�'*

loss�n�;Fv4X       �	��GYc�A�'*

lossr/�<�Op�       �	XsGYc�A�'*

loss�_?=8�2A       �	QGYc�A�'*

loss>�<�*��       �	[�GYc�A�'*

loss�mc='��       �	%[GYc�A�'*

loss��;ka�       �	a�GYc�A�'*

loss��<JZ��       �	�GYc�A�'*

lossA�;��JI       �	$GGYc�A�'*

loss,�|=Q�,�       �	��GYc�A�'*

loss���<m��       �	W�GYc�A�'*

loss4b<5�
�       �	�4GYc�A�'*

losseN=h@�x       �	��GYc�A�'*

loss7�m<?�Z3       �	7qGYc�A�'*

loss�N�;\ݎ%       �	fGYc�A�'*

loss�!;!���       �	��GYc�A�'*

loss���<���       �	�NGYc�A�'*

loss��=<��P�       �	{�GYc�A�'*

loss�e�<ހ       �	m�GYc�A�'*

loss�3*<S��P       �	8LGYc�A�'*

loss�چ<Q���       �	[�GYc�A�'*

losslH�;ci��       �	[�GYc�A�'*

loss+�<����       �	a7GYc�A�'*

lossa<����       �	��GYc�A�'*

loss�M,<�C;#       �	sGYc�A�'*

loss�9=��Y2       �	�6GYc�A�'*

loss���:��ԙ       �	8�GYc�A�'*

losssw�<�w�|       �	�~GYc�A�'*

loss���<� �/       �	! GYc�A�'*

loss:��<����       �	L� GYc�A�'*

loss�Ql=� -L       �	�!GYc�A�'*

lossr"*<�x�       �	�`"GYc�A�'*

loss���<�y�       �	�#GYc�A�'*

loss4�=��r�       �	�#GYc�A�'*

loss<<N&o�       �	��$GYc�A�'*

loss�g>=-�G%       �	�L%GYc�A�'*

loss?9=��1       �	h�%GYc�A�'*

lossiQ;,'h       �	��&GYc�A�'*

loss�я<M�@P       �	�9'GYc�A�'*

loss�F�<ɱ�        �	��'GYc�A�'*

losssO�<��V       �	Nz(GYc�A�'*

lossF�;���.       �	H)GYc�A�'*

loss���;RDgf       �	b�)GYc�A�'*

loss��$<)       �	�Z*GYc�A�'*

loss��;j�^�       �	��*GYc�A�'*

loss2�=���3       �	��+GYc�A�'*

losseU�='.�       �	�2,GYc�A�'*

loss�x=�7�       �	^�,GYc�A�'*

loss� �<�w��       �	-y-GYc�A�'*

loss�K<��L�       �	8.GYc�A�'*

loss�B/<}l��       �	��.GYc�A�'*

loss8�i;W��       �	��/GYc�A�'*

loss~�<_b�,       �	}$0GYc�A�'*

loss-�;�0�       �	��0GYc�A�'*

loss\�;���        �		o1GYc�A�'*

lossq�o<�P3�       �	L2GYc�A�'*

loss���<H�       �	Y�2GYc�A�'*

loss���<��(�       �	Lm3GYc�A�'*

loss���=�$L
       �	v4GYc�A�'*

loss@��<WX��       �	;�4GYc�A�'*

loss�I�<�]\       �	o5GYc�A�'*

loss�^6<
�
�       �	6GYc�A�'*

loss��<��#^       �	e�6GYc�A�'*

loss�o�;����       �	�m7GYc�A�'*

lossA�=�>�       �	�8GYc�A�'*

loss%B<���4       �	��8GYc�A�'*

lossL�@=����       �	-]9GYc�A�'*

loss�e<+��#       �	��9GYc�A�'*

lossf =����       �	@�:GYc�A�'*

loss��;o��       �	1A;GYc�A�'*

loss��><��v       �	%�;GYc�A�'*

loss]��<?��       �	6�<GYc�A�'*

loss��/=Ǿ|�       �	QN=GYc�A�'*

loss��=���C       �	)%>GYc�A�'*

loss�<�c&       �	��>GYc�A�'*

lossN�<�R��       �	*T?GYc�A�'*

lossx+=\�       �	@GYc�A�'*

loss3B1<*6"       �	ߩ@GYc�A�'*

loss) P<L))       �	-yAGYc�A�'*

loss|$�<�]U�       �	�BGYc�A�'*

loss#;M=ޜ��       �	�BGYc�A�(*

loss�o?<���)       �	�cCGYc�A�(*

lossa�H=
y�       �	�DGYc�A�(*

loss[�=u˻�       �	S�DGYc�A�(*

loss�/;X�)6       �	�JEGYc�A�(*

lossI��<6�o\       �	�MFGYc�A�(*

lossL��=|��$       �	j�FGYc�A�(*

loss��<��N_       �	�GGYc�A�(*

loss:�<�.�%       �	1DHGYc�A�(*

losse =�m��       �	��HGYc�A�(*

loss��k=f
q�       �	��IGYc�A�(*

loss��4<:D�R       �	+JGYc�A�(*

loss&�<�E5.       �	W�JGYc�A�(*

loss�;��mt       �	�zKGYc�A�(*

loss�j�<���2       �	�LGYc�A�(*

loss�=>	�       �	B�LGYc�A�(*

loss�a�<�U       �	��MGYc�A�(*

loss:�=zLQ�       �	y?NGYc�A�(*

loss���;��.       �	4�NGYc�A�(*

lossd�=�B�       �	��OGYc�A�(*

loss��;M�@       �	CPGYc�A�(*

loss���;9N*�       �	 �PGYc�A�(*

loss	+�<C���       �	4�QGYc�A�(*

loss.�<���+       �	�"RGYc�A�(*

loss��<-X       �	��RGYc�A�(*

loss��=d��       �	\wSGYc�A�(*

loss6i=��S       �	�TGYc�A�(*

loss��<�}�       �	j�TGYc�A�(*

loss͐�<E0GQ       �	fUGYc�A�(*

lossO%i=J>�       �	�VGYc�A�(*

loss��4=gE4       �	}�VGYc�A�(*

loss�X�;gEH       �	�RWGYc�A�(*

loss�+<�u��       �	��WGYc�A�(*

loss�>�< ��>       �	��XGYc�A�(*

loss��<K$}�       �	.=YGYc�A�(*

loss���<S�l       �	
�YGYc�A�(*

loss=K�<�س/       �	�uZGYc�A�(*

lossޛ<,ذ       �	�[GYc�A�(*

lossù,<票�       �	F�[GYc�A�(*

loss�zD<�&�N       �	�P\GYc�A�(*

loss���<��	       �	A�\GYc�A�(*

losst͓<�Pxs       �	��]GYc�A�(*

loss��E<ry�       �	.^GYc�A�(*

loss��<_�m�       �	��^GYc�A�(*

loss�+�=c��       �	bg_GYc�A�(*

lossPE�=Dʑ�       �	��_GYc�A�(*

lossh��=��       �	��`GYc�A�(*

lossa�=H���       �	 9aGYc�A�(*

loss��<���P       �	�aGYc�A�(*

loss�)	<����       �	�bGYc�A�(*

losse�A<��Y       �	�+cGYc�A�(*

lossi�S=�y}�       �	XpdGYc�A�(*

lossr�<M�$1       �	geGYc�A�(*

lossIP�;\Ը�       �	:�eGYc�A�(*

losslJ
=�/�i       �	l_fGYc�A�(*

loss =R�$�       �	��gGYc�A�(*

loss�/�<L�e�       �	x*hGYc�A�(*

loss��<���i       �	��hGYc�A�(*

lossܼ<Ӂ1�       �	-�iGYc�A�(*

lossþ5=����       �	�jGYc�A�(*

lossXR�<��Y�       �	�kGYc�A�(*

losst D=aRG       �	C�lGYc�A�(*

lossȍ�<���       �	��mGYc�A�(*

loss��;*���       �	}nGYc�A�(*

loss�1O<���       �	AoGYc�A�(*

loss�=�@w�       �	��oGYc�A�(*

loss��<㬥*       �	��pGYc�A�(*

loss��</ԕ       �	�|qGYc�A�(*

loss�CT<H�9       �	�_rGYc�A�(*

lossۗ�;�E�}       �	��rGYc�A�(*

loss�U<�_�       �	��sGYc�A�(*

lossZ�=��       �	FDtGYc�A�(*

loss��<x	�       �	R�tGYc�A�(*

loss�-�<�H��       �	G�uGYc�A�(*

loss8/<R��       �	4vGYc�A�(*

loss�T3<�:�       �	��vGYc�A�(*

loss�~,=��M       �	6�wGYc�A�(*

loss �<�0;	       �	D�xGYc�A�(*

loss�
<۳��       �	$&yGYc�A�(*

loss�3�;�!�       �	U�yGYc�A�(*

loss�"=���       �	:[zGYc�A�(*

losst�;u7       �	Z�zGYc�A�(*

loss�<��<�       �	ߊ{GYc�A�(*

loss��=Y[��       �	�L|GYc�A�(*

loss~� <�AZ,       �	��|GYc�A�(*

loss�9=�`yY       �	 �}GYc�A�(*

loss4?=�~�       �	'~GYc�A�(*

loss�o�=�>�       �	Ӽ~GYc�A�(*

loss�	>=����       �	 RGYc�A�(*

loss�,	=���'       �	��GYc�A�(*

loss�4�<�Q�       �	Ӆ�GYc�A�(*

loss��V=-v?x       �	K#�GYc�A�(*

loss�ф=�.�       �	���GYc�A�(*

loss�=O��>       �	�|�GYc�A�(*

loss+o=< �J       �	�GYc�A�(*

loss]�=���B       �	|�GYc�A�(*

loss=T�       �	"��GYc�A�(*

lossf�<w؃       �	�#�GYc�A�(*

loss���;�J)       �	U�GYc�A�(*

loss-1= 
�@       �	Ů�GYc�A�(*

lossAW�<��{�       �		P�GYc�A�(*

loss?�\<$?�       �	\�GYc�A�(*

loss�tX=ܼ�       �	8��GYc�A�(*

loss�'9;<6�z       �	W#�GYc�A�(*

loss��=��8�       �	^��GYc�A�(*

loss
�<Ƞ�       �	�P�GYc�A�(*

loss��=���F       �	��GYc�A�(*

loss���<V6�       �	M��GYc�A�(*

loss���<a��       �	��GYc�A�(*

loss��<̽8       �	ƌGYc�A�(*

loss�V�<Ո�M       �	�a�GYc�A�(*

lossc�<�R�       �	_�GYc�A�(*

loss`��<��Z�       �	뫎GYc�A�(*

loss{�<[�S7       �	UN�GYc�A�(*

loss���<#|�W       �	>�GYc�A�(*

loss� =s       �	�GYc�A�(*

loss�co<��D�       �	'1�GYc�A�(*

loss��1<K`h�       �	ϑGYc�A�(*

loss���;��2�       �	Af�GYc�A�(*

loss��<:�=i       �	r��GYc�A�(*

loss&�;�Q�       �	���GYc�A�(*

loss�.�<�7�p       �	�,�GYc�A�(*

loss��=u
�       �	 ƔGYc�A�(*

loss��<�˕@       �	�e�GYc�A�(*

loss4�<�@�       �	���GYc�A�(*

loss��;a�       �	嘖GYc�A�(*

loss���<|��h       �	�2�GYc�A�(*

loss�`�=�[��       �	�ŗGYc�A�(*

lossӗw<�)�m       �	�X�GYc�A�)*

loss�K=����       �	)�GYc�A�)*

loss>T<���2       �	�~�GYc�A�)*

loss�&=^�e5       �	n�GYc�A�)*

lossVƴ<Ƀ[�       �	���GYc�A�)*

loss�r�<�8�       �	�;�GYc�A�)*

loss]�"=݋��       �	<ݛGYc�A�)*

loss1�;=�r$       �	?ŜGYc�A�)*

lossΠ6=x(�-       �	g`�GYc�A�)*

loss=oB<���+       �	u�GYc�A�)*

loss
T]<����       �	�GYc�A�)*

loss�,r<+�A�       �	�6�GYc�A�)*

loss�=x�       �	9ҠGYc�A�)*

loss���<��D�       �	�d�GYc�A�)*

lossED�;f��       �	���GYc�A�)*

loss���;X�²       �	�ϢGYc�A�)*

lossva=�ШB       �	�h�GYc�A�)*

lossB='���       �	�GYc�A�)*

lossX��<�3E       �	��GYc�A�)*

losszA�<+�R+       �	�I�GYc�A�)*

losse�;�/Ѱ       �	��GYc�A�)*

loss=�<�o��       �	=��GYc�A�)*

lossZs<ݶh�       �	��GYc�A�)*

loss2b|=�"�       �	�GYc�A�)*

loss��<�X�       �	�O�GYc�A�)*

loss+<,g#�       �	:$�GYc�A�)*

lossLW�<�q��       �	���GYc�A�)*

loss2�;���       �	T�GYc�A�)*

loss�q�;}��       �	��GYc�A�)*

loss�:�;G��       �	i��GYc�A�)*

loss,��< Y�       �	�'�GYc�A�)*

loss��<�ҩq       �	�ìGYc�A�)*

loss��=��       �	�]�GYc�A�)*

lossx�=7��!       �	M�GYc�A�)*

lossfS&<��G�       �	���GYc�A�)*

loss��==��1       �	_)�GYc�A�)*

loss-�u<$ƬU       �	$ЯGYc�A�)*

loss�E$=�T�       �	�g�GYc�A�)*

loss���;1��       �	�GYc�A�)*

loss�=+=z R       �	Ů�GYc�A�)*

loss?�=d��       �	.U�GYc�A�)*

loss�<{��       �	D��GYc�A�)*

loss��=p��?       �	��GYc�A�)*

loss�%=����       �	f-�GYc�A�)*

loss��d<���       �	mŴGYc�A�)*

lossο�:�LD�       �	�j�GYc�A�)*

loss��<\_Jk       �	J
�GYc�A�)*

lossC�=��y�       �	9��GYc�A�)*

loss�\~<W��I       �	TR�GYc�A�)*

lossoY�<12�       �	�N�GYc�A�)*

loss�=Q<���       �	��GYc�A�)*

loss��<��H�       �	���GYc�A�)*

loss�|<ޯ��       �	�)�GYc�A�)*

lossJ�;=%��       �	ȺGYc�A�)*

lossɴ:�.       �	8��GYc�A�)*

loss�HH=%G2�       �	�j�GYc�A�)*

lossx�Z=�7�!       �	��GYc�A�)*

loss��T<����       �	�G�GYc�A�)*

loss�l�<��.$       �	��GYc�A�)*

loss:`=6}�       �	P��GYc�A�)*

lossb��;�b,�       �	�-�GYc�A�)*

loss�C<{VP�       �	"��GYc�A�)*

loss�o�<����       �	�c�GYc�A�)*

loss�l�;�4�j       �	W�GYc�A�)*

loss6�<��<c       �	��GYc�A�)*

loss�{>;g�M�       �	�@�GYc�A�)*

loss��=�f�       �	���GYc�A�)*

loss8��<ᐱ       �	p��GYc�A�)*

lossI^y<䤃       �	3Q�GYc�A�)*

loss�5�<���|       �	��GYc�A�)*

loss� �<���h       �	a��GYc�A�)*

lossl�4<���       �	Y6�GYc�A�)*

loss�``=u��       �	>��GYc�A�)*

loss̸<ZRg       �	8g�GYc�A�)*

lossVt<�
�       �	*:�GYc�A�)*

loss�g�:'�r^       �	���GYc�A�)*

lossXB*;WYX8       �	�z�GYc�A�)*

loss�(`=?��       �	�GYc�A�)*

loss���<P̳n       �	���GYc�A�)*

lossZR-<brBR       �	�u�GYc�A�)*

lossZ_3=�6��       �	��GYc�A�)*

loss��"=��=�       �	��GYc�A�)*

loss\(e<՜��       �	f�GYc�A�)*

lossh�<���       �	=�GYc�A�)*

loss�P3<�$�       �	��GYc�A�)*

lossX��;�P�z       �	O�GYc�A�)*

loss�U;��f       �	��GYc�A�)*

loss�6�;!�v�       �	�:�GYc�A�)*

lossC�;���       �	���GYc�A�)*

loss�?�<9�x�       �	`��GYc�A�)*

loss�J�<�!K�       �	5&�GYc�A�)*

loss�A<�]�       �	��GYc�A�)*

loss(��<)+�       �	�N�GYc�A�)*

loss�`�;�d��       �	B��GYc�A�)*

loss`L�9��       �	,��GYc�A�)*

loss$u�9?CL       �	��GYc�A�)*

loss�cb;D7�F       �	ͭ�GYc�A�)*

lossW�<E� �       �	9D�GYc�A�)*

loss��<*O�V       �	{k�GYc�A�)*

loss�U:7�.D       �	���GYc�A�)*

loss��<��_q       �	l��GYc�A�)*

loss;��=�mP�       �	���GYc�A�)*

lossC��:ńW�       �	4e�GYc�A�)*

lossB��<��       �	3��GYc�A�)*

loss���<נ�       �	.��GYc�A�)*

lossa��<k3B+       �	y!�GYc�A�)*

lossh��<��Ǆ       �	���GYc�A�)*

lossf|#<�U5�       �	�X�GYc�A�)*

lossJ�`<���s       �	�
�GYc�A�)*

loss���<)��       �	m��GYc�A�)*

loss1C<�H       �	�C�GYc�A�)*

loss��d<�cV�       �	���GYc�A�)*

loss�	�<�B:|       �	���GYc�A�)*

lossoBT=�xd$       �	'��GYc�A�)*

loss��=��\�       �	s0�GYc�A�)*

lossn�<�Q1       �	~��GYc�A�)*

loss�RT<[F.�       �	���GYc�A�)*

loss�=g��       �	Nb�GYc�A�)*

loss�?8=z�+       �	���GYc�A�)*

loss���;�3��       �	|��GYc�A�)*

loss_0H<����       �	�/�GYc�A�)*

lossC�#<<r��       �	��GYc�A�)*

lossak;R       �	���GYc�A�)*

loss�W�<v�       �	�(�GYc�A�)*

losss)<fOq9       �	���GYc�A�)*

loss��:IP\       �	�V�GYc�A�)*

lossF%X;Q-Nm       �	��GYc�A�)*

loss���<s�r       �	$��GYc�A�)*

loss��<-�Ո       �	"�GYc�A�**

loss?��<�4P1       �	��GYc�A�**

lossT�}<ڗw       �	C�GYc�A�**

loss�V=�"�       �	�$�GYc�A�**

lossc�;|ps       �	*��GYc�A�**

lossK 
<$腊       �	�`�GYc�A�**

lossXT<�^E       �	���GYc�A�**

lossm<h��       �	ҍ�GYc�A�**

losshl<��:       �	�"�GYc�A�**

loss�V8<&�       �	���GYc�A�**

lossH��<P�߻       �	M�GYc�A�**

loss̎�;?B�:       �	L��GYc�A�**

loss{ʋ=z�2s       �	�v�GYc�A�**

loss=8�<a�        �	
�GYc�A�**

lossNB=�'�       �	x��GYc�A�**

loss�N<b�3c       �	�4�GYc�A�**

lossx��<X�)       �	���GYc�A�**

loss�3�=���       �	�l�GYc�A�**

loss=��;���0       �	��GYc�A�**

loss�E.=��:       �	���GYc�A�**

loss�7 =o��       �	!��GYc�A�**

loss�k;L�tT       �	�6�GYc�A�**

loss)DM==�       �	`��GYc�A�**

loss�+�<�~2�       �	�`�GYc�A�**

loss��M<��;n       �	���GYc�A�**

loss�OV<��       �	l$HYc�A�**

loss�i<h��       �	��HYc�A�**

lossF��<l5�C       �	�OHYc�A�**

loss�O�<�A       �	T�HYc�A�**

loss��=��nr       �	�vHYc�A�**

lossqP<}�       �	}	HYc�A�**

lossH$�<xͿ�       �	�4HYc�A�**

loss�8/=#�^�       �		�HYc�A�**

loss���=��4       �	�HYc�A�**

lossH��<)� l       �	�#HYc�A�**

loss��;���       �	��HYc�A�**

loss4�=8�D�       �	�THYc�A�**

loss�[0=�4��       �	v2HYc�A�**

loss�/�<R���       �	��HYc�A�**

loss(��<���       �	,�HYc�A�**

loss�V�<1�x       �	5'HYc�A�**

loss.�~:X��o       �	q�HYc�A�**

loss6�r<7C#       �	�gHYc�A�**

loss��J<�h�       �	�HYc�A�**

loss6�=u��?       �	'� HYc�A�**

loss��s<3��h       �		P!HYc�A�**

loss$ON=Io�\       �	�!HYc�A�**

lossy� ;��}�       �	�"HYc�A�**

loss��=fC�       �	�L#HYc�A�**

loss�߉<�J�       �	��#HYc�A�**

loss�B�<�g`       �	��$HYc�A�**

lossV�<v���       �	'.%HYc�A�**

loss=��;|x��       �		�%HYc�A�**

loss��=8H �       �	��&HYc�A�**

loss-�<�A��       �	e6'HYc�A�**

lossf��< �v$       �	n�'HYc�A�**

lossђ=-��       �	�(HYc�A�**

loss@#<���       �	�')HYc�A�**

loss�=�&/       �	
�)HYc�A�**

loss*��<�!��       �	�e*HYc�A�**

loss��<�W_       �	��*HYc�A�**

loss��<y<_�       �	n�+HYc�A�**

loss�/8=��q�       �	�=,HYc�A�**

lossT7�=%66S       �	�,HYc�A�**

loss#qu=x�Y�       �	�s-HYc�A�**

loss�)C<d��       �	)!.HYc�A�**

loss6A�;9 Ƌ       �		�.HYc�A�**

loss�K�;�K��       �	�a/HYc�A�**

loss�/=�k�       �	�0HYc�A�**

lossyJ�<�H�       �	i�0HYc�A�**

loss�.c=�赹       �	�R1HYc�A�**

loss�uL<0l3=       �	��1HYc�A�**

lossR'=��v>       �	g�2HYc�A�**

loss�Ċ<1H|!       �	i;3HYc�A�**

loss4c�:��#       �	b�3HYc�A�**

lossX?<�n;       �	�t4HYc�A�**

loss��=meu�       �	5HYc�A�**

loss��;q.�        �	^�5HYc�A�**

loss�.u=�F�       �	PQ6HYc�A�**

loss�?|<�ub%       �	l�6HYc�A�**

loss�n�;xU%       �	�7HYc�A�**

loss�4�:���       �	^,8HYc�A�**

lossíF:�9�-       �	h�8HYc�A�**

loss( �<aG�       �	�j9HYc�A�**

loss1W�<cm�       �	�:HYc�A�**

loss�f�<£ʚ       �	B�:HYc�A�**

lossqי<��sg       �	�P;HYc�A�**

loss2��;�%�       �	:�;HYc�A�**

loss���<��/P       �	��<HYc�A�**

loss�a�<ry4'       �	�5=HYc�A�**

loss@�<����       �	��=HYc�A�**

loss�01=�,�       �	Yk>HYc�A�**

lossr�|<�r%�       �	�?HYc�A�**

loss�OF=�X�       �	p�?HYc�A�**

lossj��<�),       �	�o@HYc�A�**

lossClM=x��r       �	�AHYc�A�**

lossZ� =��E       �	�BHYc�A�**

lossq�(=�eK       �	�1CHYc�A�**

lossj��<y;X       �	��CHYc�A�**

lossl�	<����       �	��EHYc�A�**

loss��< ��       �	��FHYc�A�**

loss�!�;�/P�       �	��GHYc�A�**

loss*Y^<��f
       �	�=HHYc�A�**

lossy�;S�]�       �	��HHYc�A�**

lossѝ<�7�/       �	IHYc�A�**

loss�u=04R�       �	_(JHYc�A�**

loss��=�'��       �	��JHYc�A�**

loss���<���c       �	mKHYc�A�**

loss�C<��[X       �	!ZLHYc�A�**

loss�F<����       �	:MHYc�A�**

lossUD�<�S8-       �	k�MHYc�A�**

loss�֑<ӗ��       �	
KNHYc�A�**

lossE�<�K]       �	=OHYc�A�**

loss<�<}�0�       �	��OHYc�A�**

loss&�$<� [,       �	�cPHYc�A�**

lossXr=>�=[       �	��PHYc�A�**

loss��<%o��       �	��QHYc�A�**

loss�Q�<L^��       �	77RHYc�A�**

loss�v�<�o��       �	g�RHYc�A�**

loss,�*=.��	       �	��SHYc�A�**

loss$��;�`^�       �	XVTHYc�A�**

loss�(�<Ÿ�       �	j�THYc�A�**

loss�\�<9v�       �	%�UHYc�A�**

loss�xO=0pf!       �	74VHYc�A�**

loss��<��P1       �	{�VHYc�A�**

loss+.<�KXp       �	k�WHYc�A�**

loss�{[;�1)�       �	�XHYc�A�**

loss���<���       �	�XHYc�A�**

lossq�;�$@       �	�ZYHYc�A�+*

loss��<U;	&       �	eZHYc�A�+*

loss���;+��       �	"[HYc�A�+*

loss�f�;� �       �	W�[HYc�A�+*

loss��/<�[q�       �	{M\HYc�A�+*

loss���<�L;�       �	2�\HYc�A�+*

loss�$<��       �	�]HYc�A�+*

loss��L<%��       �	�*^HYc�A�+*

loss2<�!��       �	}�^HYc�A�+*

loss��=���       �	��`HYc�A�+*

loss(>�;�n�       �	�aHYc�A�+*

loss�7H<�'�r       �	��aHYc�A�+*

loss!�;�%XV       �	�ObHYc�A�+*

losss�5=�`�       �	cHYc�A�+*

loss��z=���       �	2eHYc�A�+*

loss�Ȼ<��R�       �	;�eHYc�A�+*

lossr�;gg�3       �	�QfHYc�A�+*

lossL�3<1���       �	gHYc�A�+*

lossT�E=ڝV�       �	��gHYc�A�+*

lossj��<>�Nb       �	�nhHYc�A�+*

lossMW<+&jS       �	kiHYc�A�+*

loss3�;��H}       �	��iHYc�A�+*

loss���=!��       �	�YjHYc�A�+*

loss�7�;�0�~       �	��jHYc�A�+*

loss`��;S�ʬ       �	o�kHYc�A�+*

loss�;=���[       �	W@lHYc�A�+*

lossO̱<��D       �	�lHYc�A�+*

lossT&R= I�       �	��mHYc�A�+*

loss)i�<���x       �	}�nHYc�A�+*

loss�{B<7/J       �	�
pHYc�A�+*

loss!5�<f���       �	U�pHYc�A�+*

lossx@=&;j�       �	�RqHYc�A�+*

loss�2�;��       �	��qHYc�A�+*

loss�3;�;�D       �	�rHYc�A�+*

loss'+<����       �	�@sHYc�A�+*

loss���<C��B       �	�sHYc�A�+*

loss<��x�       �	=}tHYc�A�+*

lossl�F<��:�       �	�uHYc�A�+*

loss��S<?��j       �	��uHYc�A�+*

loss�5-=�       �	fvHYc�A�+*

lossHEI<�,ك       �	wHYc�A�+*

lossJ[=�$��       �	�wHYc�A�+*

loss��<��98       �	6WxHYc�A�+*

loss��<���&       �	��xHYc�A�+*

loss�g�;�Ę       �	H�yHYc�A�+*

loss�#�<)��       �	�[zHYc�A�+*

lossi��<6vJQ       �	��zHYc�A�+*

loss���;�7�I       �	�{HYc�A�+*

loss��U<�x��       �	�4|HYc�A�+*

loss��<TK�       �	j�|HYc�A�+*

lossJG=X��       �	�s}HYc�A�+*

loss�96<R&w       �	 ~HYc�A�+*

loss���:�YZ       �	ծ~HYc�A�+*

loss��<�@~�       �	<OHYc�A�+*

loss��=�E{m       �	�HYc�A�+*

loss�{&=��       �	搀HYc�A�+*

loss�J�<2;c       �	�.�HYc�A�+*

loss�IZ<�YY       �	���HYc�A�+*

lossd��;���}       �	B��HYc�A�+*

loss���;��
       �	~8�HYc�A�+*

loss��	;���       �	>ЃHYc�A�+*

loss���=��S�       �	V��HYc�A�+*

loss�-<�x�       �	Zh�HYc�A�+*

loss�v�;��}u       �	��HYc�A�+*

loss�{ =l�ϻ       �	֩�HYc�A�+*

loss��Y<߇�        �	8I�HYc�A�+*

lossB�<�|R�       �	P�HYc�A�+*

loss$d�<p��       �	ԁ�HYc�A�+*

loss�;/�T       �	:"�HYc�A�+*

loss�<KSj+       �	"HYc�A�+*

loss�*�<X���       �	�\�HYc�A�+*

loss���;,�z       �	��HYc�A�+*

loss\��=㧾>       �	נ�HYc�A�+*

loss��<��T       �	:�HYc�A�+*

loss\zN=���       �	g֍HYc�A�+*

loss�p=���{       �	us�HYc�A�+*

loss��2=�o��       �	�HYc�A�+*

lossvSF;;�~�       �	���HYc�A�+*

loss\h�<0	-       �	�O�HYc�A�+*

loss�?�<���       �	���HYc�A�+*

loss��G<K�ŭ       �	 ��HYc�A�+*

loss�O<1�K       �	%\�HYc�A�+*

loss
��:�z�       �	��HYc�A�+*

lossTU}=�r�       �	���HYc�A�+*

loss�$~<���       �	�-�HYc�A�+*

loss���<��g�       �	�ٔHYc�A�+*

loss�<�<�       �	�o�HYc�A�+*

loss��c=�1�       �	��HYc�A�+*

loss�[�<��1�       �	^��HYc�A�+*

lossZ��;C$
�       �	�T�HYc�A�+*

loss��@<����       �	�HYc�A�+*

lossI�x;WD�       �	H��HYc�A�+*

loss=x�<��`	       �	-B�HYc�A�+*

lossl�<�gn�       �	��HYc�A�+*

loss�Վ=RcfR       �	dy�HYc�A�+*

loss�o�;$���       �	� �HYc�A�+*

loss�1C<�s�O       �	]śHYc�A�+*

loss_�
=2��       �	
h�HYc�A�+*

loss�`�<�Ba�       �	�	�HYc�A�+*

loss��W<JV       �	m��HYc�A�+*

loss1��<�<�       �	�G�HYc�A�+*

loss��<.�\�       �	.�HYc�A�+*

loss6��;9Cqe       �	F��HYc�A�+*

loss3��<���       �	�0�HYc�A�+*

loss�%=����       �	�נHYc�A�+*

lossȉz<�j��       �	��HYc�A�+*

loss�Ӹ;��g       �	� �HYc�A�+*

loss�Xr<\��Q       �	uǢHYc�A�+*

lossI��<���       �	&�HYc�A�+*

loss���;6<�       �	[%�HYc�A�+*

loss��<�j�       �	*ĥHYc�A�+*

loss��<q��       �	�M�HYc�A�+*

loss�@=a�y       �	��HYc�A�+*

loss�<�=���       �	ę�HYc�A�+*

loss@�<8uH�       �	�4�HYc�A�+*

loss8Ǣ=H��       �	-ҩHYc�A�+*

loss&m�<��*�       �	Xu�HYc�A�+*

loss�{�<�\�       �	��HYc�A�+*

lossz��<ʏ�       �	��HYc�A�+*

loss;��<Ԇ~       �	�]�HYc�A�+*

loss1�=�R�d       �	x�HYc�A�+*

loss蜚=�Q       �	���HYc�A�+*

loss��8=���q       �	TV�HYc�A�+*

loss۹�<l���       �	���HYc�A�+*

loss�4�<�WH       �	٘�HYc�A�+*

loss���<i%P       �	v6�HYc�A�+*

loss��<�U_=       �	VְHYc�A�+*

loss��<�0�2       �	>v�HYc�A�+*

lossM�f<Xf�       �	�HYc�A�,*

loss,��;a��       �	���HYc�A�,*

lossd'=a;�       �	�?�HYc�A�,*

loss���<��
!       �	���HYc�A�,*

lossQ�e<��he       �	ڑ�HYc�A�,*

loss��*=��}�       �	,�HYc�A�,*

lossjA$=��       �	rõHYc�A�,*

loss��=�df�       �	�V�HYc�A�,*

lossE�k=���       �	��HYc�A�,*

loss��;��(�       �	yηHYc�A�,*

lossh�<��f       �	�d�HYc�A�,*

loss�@|<Ks�       �	�`�HYc�A�,*

lossۋ;=_k�:       �	���HYc�A�,*

losst�:=мA�       �	���HYc�A�,*

loss�m<��3�       �	<�HYc�A�,*

loss�Mo<�`d       �	�ٻHYc�A�,*

lossCq<x���       �	tϼHYc�A�,*

lossC�<�;��       �	Lk�HYc�A�,*

loss�ZA<S� �       �	��HYc�A�,*

loss.
w<U�Ac       �	-��HYc�A�,*

loss�v�<����       �	�ĿHYc�A�,*

loss�%�<����       �	�^�HYc�A�,*

lossge;���f       �	�HYc�A�,*

loss��<w���       �	��HYc�A�,*

loss*�;�6�]       �	#j�HYc�A�,*

loss9�<�w7       �	y�HYc�A�,*

loss��<�"ȏ       �	��HYc�A�,*

loss�=V�L�       �	=�HYc�A�,*

loss\�|=��-       �	���HYc�A�,*

loss��<���       �	�B�HYc�A�,*

loss���<��.J       �	AE�HYc�A�,*

lossD|=gu�	       �	'��HYc�A�,*

lossaq�;�+�       �	�r�HYc�A�,*

loss�<��H�       �	R�HYc�A�,*

loss�!�<�?I�       �	3��HYc�A�,*

lossȑ�=v�o�       �	WA�HYc�A�,*

loss7h�<ɋ�       �	�HYc�A�,*

lossCT=P��k       �	���HYc�A�,*

loss7�H=�qF       �	���HYc�A�,*

loss{�~<�%.�       �	E+�HYc�A�,*

loss��M<�ݺ�       �	���HYc�A�,*

lossDq�<ƪ~w       �	i�HYc�A�,*

lossf:_=���       �	��HYc�A�,*

loss@<W=�֓z       �	��HYc�A�,*

lossӃ=��C       �	G9�HYc�A�,*

loss�I><�H�       �	���HYc�A�,*

lossN!<���       �	�v�HYc�A�,*

loss��6<���       �	��HYc�A�,*

loss{��</fI�       �	\��HYc�A�,*

loss�V:̻u�       �	�G�HYc�A�,*

loss��)<gb�       �	G��HYc�A�,*

loss&^<�"/%       �	(��HYc�A�,*

loss6�n<h�-s       �	��HYc�A�,*

loss��<,H[(       �	���HYc�A�,*

loss��6=*��5       �	�b�HYc�A�,*

loss�V#=3{f�       �	Y��HYc�A�,*

loss��<�Ec�       �	��HYc�A�,*

loss�MY=˻ؚ       �	5��HYc�A�,*

lossl�<uS��       �	�.�HYc�A�,*

loss�W�;@�e�       �	���HYc�A�,*

loss<�u<��ԑ       �	�o�HYc�A�,*

loss��<�:��       �	E�HYc�A�,*

loss���;���       �	&��HYc�A�,*

loss���<�F�S       �	h�HYc�A�,*

loss*�e=�1�T       �	=�HYc�A�,*

loss�0=�(�       �	��HYc�A�,*

loss-ך=y: I       �	&S�HYc�A�,*

lossאf<�U�       �	��HYc�A�,*

loss{?�<0��       �	��HYc�A�,*

lossc��<i���       �	4+�HYc�A�,*

loss�;~��       �	���HYc�A�,*

loss(=;;gs�       �	�p�HYc�A�,*

loss�=���D       �	E�HYc�A�,*

loss*vJ<���}       �	H��HYc�A�,*

lossfEO=���P       �	�E�HYc�A�,*

loss��<q1m\       �	'��HYc�A�,*

loss�a�<l�<       �	9)�HYc�A�,*

loss�b�;ax	�       �	=��HYc�A�,*

loss7G)<%�       �	�{�HYc�A�,*

lossΜ�<�r�       �	}%�HYc�A�,*

lossU�<yΟ       �	���HYc�A�,*

lossc�<x[�       �	:t�HYc�A�,*

loss�ۗ;Ja�       �	;�HYc�A�,*

loss�o-<����       �	I��HYc�A�,*

loss���<��"       �	�`�HYc�A�,*

loss
�z;�n�       �	���HYc�A�,*

loss�\U<&�E       �	���HYc�A�,*

loss`U�;ս��       �	�;�HYc�A�,*

lossL�=���       �	���HYc�A�,*

lossi�]<I�C       �	dv�HYc�A�,*

loss�w�<��       �	��HYc�A�,*

loss�d =�6��       �	i��HYc�A�,*

loss��;���       �	�E�HYc�A�,*

loss�h]<�y8       �	"��HYc�A�,*

loss���<%ǎ       �	˅�HYc�A�,*

loss?�U<����       �	Z/�HYc�A�,*

loss.��;�tv�       �	y��HYc�A�,*

loss�K=|4��       �	��HYc�A�,*

loss�!b=�YO       �	�8�HYc�A�,*

loss�mO<	�3�       �	���HYc�A�,*

lossSK�<	_K       �	�s�HYc�A�,*

loss�?;.�0T       �	��HYc�A�,*

lossd_m;z{       �	���HYc�A�,*

lossd=P_�{       �	W`�HYc�A�,*

loss���<��`�       �	��HYc�A�,*

loss)({<I|       �	���HYc�A�,*

losshe<7�p       �	mW�HYc�A�,*

loss���< aQ�       �	/��HYc�A�,*

loss�<D;z�       �	ɰ�HYc�A�,*

loss#p<�ϻ�       �	W�HYc�A�,*

loss�;<�0�        �	m �HYc�A�,*

loss.uo=w�k�       �		��HYc�A�,*

lossrq;+P�       �	L�HYc�A�,*

loss���=�n��       �	���HYc�A�,*

loss,�e=�{�Q       �	M��HYc�A�,*

lossO��;��0c       �	\ �HYc�A�,*

lossh&
=��B�       �	���HYc�A�,*

loss+�<h1��       �	�� IYc�A�,*

loss�,�<w�&       �	%uIYc�A�,*

lossJ��:Y���       �	� IYc�A�,*

loss� <__��       �	��IYc�A�,*

lossܜg< �4       �	hIYc�A�,*

loss�M�=�sO       �	�IYc�A�,*

loss���<q�å       �	 �IYc�A�,*

loss��<�A7�       �	�OIYc�A�,*

loss��<���       �	�,IYc�A�,*

loss��<��y       �	��IYc�A�,*

loss \�<����       �	�tIYc�A�,*

loss��<tn�H       �	$IYc�A�-*

loss��<��!       �	+0	IYc�A�-*

loss)��;h�DJ       �	��	IYc�A�-*

lossB#�<*/g}       �	�
IYc�A�-*

lossd@E=�q�       �	�WIYc�A�-*

loss�=o^֮       �	��IYc�A�-*

loss���=�I��       �	��IYc�A�-*

loss��=��       �	�(IYc�A�-*

lossO�1<U��       �	T�IYc�A�-*

loss�Gm<-��
       �	cIYc�A�-*

loss{��<}Y�       �	�IYc�A�-*

losshɢ<"�O�       �	"�IYc�A�-*

loss��<[:�       �	�PIYc�A�-*

loss(h<.�	       �	��IYc�A�-*

loss��J<�+�-       �	z�IYc�A�-*

lossh��<��
W       �	�/IYc�A�-*

loss�M=��u�       �	��IYc�A�-*

loss# �<��Y       �	�aIYc�A�-*

loss�{+=Kb��       �	j3IYc�A�-*

loss҉-=���
       �	�-IYc�A�-*

loss-��<L�       �	��IYc�A�-*

loss���<	p*       �	mIYc�A�-*

lossM5�<@��3       �	`IYc�A�-*

loss�j<T��_       �	�IYc�A�-*

loss�S�<od.       �	�AIYc�A�-*

loss��	<m�qQ       �	j�IYc�A�-*

loss(X�="�G       �	dyIYc�A�-*

loss�G@=Ŀ��       �	�IYc�A�-*

loss�kn<.N�_       �	ȵIYc�A�-*

loss�q!<DC��       �	J]IYc�A�-*

loss�>=<�I�       �	�IYc�A�-*

loss�H =��[       �	��IYc�A�-*

loss�M3<2�ܵ       �	�VIYc�A�-*

loss�� <���       �	��IYc�A�-*

loss:<+2�       �	�IYc�A�-*

lossN�;0�h       �	)$ IYc�A�-*

loss��><
,�       �	y� IYc�A�-*

loss X1<,���       �	��!IYc�A�-*

loss�@g=L��       �	��"IYc�A�-*

loss_!<�3�.       �	�C#IYc�A�-*

lossAH<�z��       �	��#IYc�A�-*

losslv�;0�1�       �	�y$IYc�A�-*

lossX4<��QE       �	�%IYc�A�-*

loss�-�<�Gd�       �	�&IYc�A�-*

loss���<,F�l       �	"8'IYc�A�-*

loss���<:ŝ       �	H�'IYc�A�-*

lossP�<ٞQ�       �	��(IYc�A�-*

loss�Q=j�D�       �	�-)IYc�A�-*

loss�k�<s;-       �	V�)IYc�A�-*

lossT~?=,ؓ       �	fj*IYc�A�-*

loss�B<>�y�       �	�1+IYc�A�-*

loss�[7=xFmK       �	�+IYc�A�-*

loss	�J=[�7�       �	�~,IYc�A�-*

loss�C�<�M��       �	B�-IYc�A�-*

loss�=%�1�       �	%�.IYc�A�-*

loss�1�=�v^�       �	i5/IYc�A�-*

loss��z=1�9q       �	��/IYc�A�-*

lossj�;���       �	٘0IYc�A�-*

loss�/C<(��8       �	�H1IYc�A�-*

loss��<��b�       �	��1IYc�A�-*

lossr�=���       �	�2IYc�A�-*

loss�=Pΐ       �	�G3IYc�A�-*

loss?��=Q䋛       �	�4IYc�A�-*

loss�bh;��       �	�~5IYc�A�-*

lossq�z=���*       �	B{6IYc�A�-*

lossA�=��e       �	�)7IYc�A�-*

loss�g�<�.;"       �	��7IYc�A�-*

losse#�<k]       �	��8IYc�A�-*

lossI�<ʄ�i       �	�(9IYc�A�-*

lossqe�<F�       �	k�9IYc�A�-*

lossϢ�;J��       �	R~:IYc�A�-*

loss��]<�G�v       �	{/;IYc�A�-*

loss#��<}��       �	��;IYc�A�-*

loss��B<N��1       �	5|<IYc�A�-*

loss�K<��?�       �	�+=IYc�A�-*

loss�g)=ms�       �	M�=IYc�A�-*

loss�x<�o%       �	p|>IYc�A�-*

loss |<�
�s       �	�)?IYc�A�-*

loss�DQ=�C%B       �	��?IYc�A�-*

loss*n`< �D       �	�y@IYc�A�-*

loss��;�e��       �	mAIYc�A�-*

loss���<�mG�       �	�AIYc�A�-*

loss�>L=؊�       �	�\BIYc�A�-*

loss��=�>o       �	��BIYc�A�-*

loss�ʻ<�d3m       �	ԜCIYc�A�-*

loss��=<�̦	       �	0gDIYc�A�-*

loss�6-=N3       �	F
EIYc�A�-*

lossZW=���5       �	�EIYc�A�-*

loss	:�<͂�       �	�TFIYc�A�-*

loss)�<��l9       �	�2GIYc�A�-*

losst �=��       �	�GIYc�A�-*

loss�t*=��       �	|HIYc�A�-*

lossa�;�iA       �	�.IIYc�A�-*

loss��z<6��G       �	��IIYc�A�-*

loss�=�R	J       �	��JIYc�A�-*

loss�~�<Q���       �	OKIYc�A�-*

loss�>=���X       �	7�KIYc�A�-*

loss7<<� �       �	�LIYc�A�-*

lossI.T=��       �	�SMIYc�A�-*

loss`n<?)aR       �	�NIYc�A�-*

loss�� <�DO       �	�NIYc�A�-*

lossN�{<["�n       �	�dOIYc�A�-*

loss#+�;!q<       �	QPIYc�A�-*

loss�n[<�"�g       �	R�PIYc�A�-*

loss�<���       �	��QIYc�A�-*

loss�<�f G       �	B?RIYc�A�-*

lossҋ <�f{�       �	�RIYc�A�-*

loss�\�<V�	       �	�SIYc�A�-*

loss/w�;��       �	�/TIYc�A�-*

loss$xG;kz��       �	��TIYc�A�-*

loss8qC;���       �	��UIYc�A�-*

loss�2=IXK�       �	,,VIYc�A�-*

loss�.=-<?       �	h�VIYc�A�-*

loss1c�;VZw       �	�|WIYc�A�-*

loss��N=��H       �	�)XIYc�A�-*

loss�^L<��E       �	Y�XIYc�A�-*

loss7[<��"�       �	�YIYc�A�-*

lossa��;h�Q       �	�)ZIYc�A�-*

loss�ʅ<��7       �	��ZIYc�A�-*

lossī�<JDD       �	�[IYc�A�-*

loss�>�Wjf       �	E/\IYc�A�-*

loss�k�=���o       �	\]IYc�A�-*

loss���=�e��       �	&�]IYc�A�-*

loss*+<��@�       �	�q^IYc�A�-*

lossj{f<J�%9       �	�_IYc�A�-*

loss۝,=x�       �	��_IYc�A�-*

loss\�<���u       �	o`IYc�A�-*

loss�f<KT��       �	#aIYc�A�-*

lossa�>=�D��       �	��aIYc�A�.*

loss���;��w       �	ZcbIYc�A�.*

loss�|�<�	�
       �	�cIYc�A�.*

loss�(V=��       �	ȳcIYc�A�.*

lossh%�<j�~�       �	�PdIYc�A�.*

loss���;�3��       �	��dIYc�A�.*

loss�[b<���B       �	�eIYc�A�.*

loss�z�<~ӫL       �	��fIYc�A�.*

loss���;���       �	^gIYc�A�.*

lossˠ
=j3�$       �	thIYc�A�.*

loss�s�<j       �	�hIYc�A�.*

lossdu<SȐ#       �	�oiIYc�A�.*

loss`=/=j�v�       �	njIYc�A�.*

loss�,/<���U       �	�jIYc�A�.*

loss�Ց;�oo       �	��kIYc�A�.*

lossiu�<dp�       �	�PlIYc�A�.*

loss���<}��       �	Y�lIYc�A�.*

loss!�"=#*�       �	��mIYc�A�.*

loss��<O�a       �	F?nIYc�A�.*

lossTF<��/       �	z�nIYc�A�.*

loss�4�;bޕ�       �	�yoIYc�A�.*

loss�U<Ոn�       �	4pIYc�A�.*

loss��]<�*�E       �	dtqIYc�A�.*

loss���;���5       �	/rIYc�A�.*

lossd�R<��ƕ       �	s�rIYc�A�.*

loss
ק;�n$Q       �	nQsIYc�A�.*

loss�%�<�6q       �	�sIYc�A�.*

loss�<��       �	�tIYc�A�.*

loss
Y�<xn�}       �	b0uIYc�A�.*

loss�O<lxY�       �	>�uIYc�A�.*

loss4ݴ<����       �	�jvIYc�A�.*

loss�q<85|�       �	��wIYc�A�.*

loss�<K�B�       �	�gxIYc�A�.*

loss҉�<��k>       �	�yIYc�A�.*

loss�G�;4}��       �	N�yIYc�A�.*

loss��i:ӊ��       �	�3zIYc�A�.*

loss@A�;��9�       �	$�zIYc�A�.*

loss�<�s       �	%x{IYc�A�.*

loss��;1���       �	�|IYc�A�.*

loss��J<	��U       �	W|}IYc�A�.*

loss-z>=�Z��       �	M~IYc�A�.*

loss+I=~���       �	�~IYc�A�.*

lossW��;��r�       �	~SIYc�A�.*

loss���<JI�&       �	��IYc�A�.*

loss���<���       �	�ـIYc�A�.*

loss-j�<E*�       �	I��IYc�A�.*

loss�|;���o       �	o+�IYc�A�.*

lossO��<�^�#       �	�˂IYc�A�.*

loss��<N�E       �	V��IYc�A�.*

lossrV�;�8       �	!\�IYc�A�.*

lossx�<~r       �	���IYc�A�.*

lossad�;r[x       �	O��IYc�A�.*

loss
1;<G�I�       �	�1�IYc�A�.*

loss��;�aM       �	BІIYc�A�.*

loss�H�9����       �	�x�IYc�A�.*

loss�܀;�H2       �	b�IYc�A�.*

loss��i;��i       �	���IYc�A�.*

lossT֭<�),o       �	�O�IYc�A�.*

loss0S<�D�       �	��IYc�A�.*

loss�kr:v�\�       �	��IYc�A�.*

loss���;*��i       �	��IYc�A�.*

lossʻ�=C�       �	<��IYc�A�.*

lossÅ�:��q       �	���IYc�A�.*

loss���<���6       �	j4�IYc�A�.*

loss-�=.;Q#       �	mȍIYc�A�.*

loss�;�<|Sʈ       �	�c�IYc�A�.*

loss�Fh<#z       �	F	�IYc�A�.*

lossf�9<�%J�       �	���IYc�A�.*

loss���=+�!B       �	�x�IYc�A�.*

loss��<���       �	��IYc�A�.*

loss=�=�J       �	���IYc�A�.*

loss��<W�x[       �	>?�IYc�A�.*

loss�N=�ͥ�       �	ےIYc�A�.*

lossq,n=z��C       �	Kr�IYc�A�.*

loss�<��g       �	��IYc�A�.*

loss̑�<J23�       �	}��IYc�A�.*

loss�V =w�c       �		T�IYc�A�.*

loss�Z�=��o       �	��IYc�A�.*

loss���<��xw       �	���IYc�A�.*

loss��<J��x       �	5&�IYc�A�.*

loss��<����       �	��IYc�A�.*

loss#7<�]Ȟ       �	�V�IYc�A�.*

loss�t�;q΄       �	j��IYc�A�.*

lossfi�<4NC.       �	q��IYc�A�.*

lossm�	<�6�       �	�%�IYc�A�.*

loss�<<޾�       �	*ƚIYc�A�.*

loss� <7�I�       �	Vf�IYc�A�.*

loss��*;�L~�       �	�
�IYc�A�.*

loss�Q=!�iU       �	���IYc�A�.*

loss[�H<Z�C       �	�L�IYc�A�.*

loss���<��       �	��IYc�A�.*

lossh'�<hy��       �	���IYc�A�.*

lossxo<�V
       �	+�IYc�A�.*

loss��<�!\p       �	�ǟIYc�A�.*

lossx��<T�6k       �	4d�IYc�A�.*

loss87�;r��       �	��IYc�A�.*

lossWC�<X       �	ꖡIYc�A�.*

loss�g�;]e�       �	�B�IYc�A�.*

loss���;�n��       �	nܢIYc�A�.*

loss1n�<�(wv       �	��IYc�A�.*

loss8��<H��t       �	�%�IYc�A�.*

loss�>�<�;n�       �	mʤIYc�A�.*

loss��<;��       �	W��IYc�A�.*

lossִg<��       �	-?�IYc�A�.*

loss��e<�U�%       �	]Q�IYc�A�.*

lossjػ<We**       �	��IYc�A�.*

loss�(c<��>       �	���IYc�A�.*

loss�!�<K`�       �	�?�IYc�A�.*

loss|ˊ<2.Y       �	}�IYc�A�.*

lossI�%;�4x       �	��IYc�A�.*

loss ̮<E�4�       �	�/�IYc�A�.*

loss�F�;AI��       �	6ȫIYc�A�.*

loss�1)<߈V�       �	��IYc�A�.*

lossz��;}�t       �	�IYc�A�.*

loss�o�<V�-       �	���IYc�A�.*

loss�t=��K&       �	jL�IYc�A�.*

loss� /<*ɇ�       �	K��IYc�A�.*

lossM��<���       �	���IYc�A�.*

loss���<�E��       �	�.�IYc�A�.*

loss���<��       �	���IYc�A�.*

loss��<���       �	�}�IYc�A�.*

loss�k�<��<�       �	~�IYc�A�.*

loss[\�<k���       �	:��IYc�A�.*

loss|d;?���       �	{g�IYc�A�.*

loss�v�<eg��       �	y�IYc�A�.*

lossŹ�<�~{       �	£�IYc�A�.*

loss���<A��       �	6=�IYc�A�.*

loss��<�,�       �	��IYc�A�.*

loss��?<Kwl�       �	>��IYc�A�/*

loss�&;KA�       �	�2�IYc�A�/*

lossT�;����       �	���IYc�A�/*

loss�x�;���"       �	���IYc�A�/*

lossFdl=<���       �	y$�IYc�A�/*

loss�!�<|���       �	��IYc�A�/*

lossW
�<�ʢ<       �	CT�IYc�A�/*

loss��<�e+       �	���IYc�A�/*

loss���<�r׌       �	^��IYc�A�/*

loss��;<�Ȅ�       �	��IYc�A�/*

lossD'<��ˬ       �	y��IYc�A�/*

loss��b<��G       �	G�IYc�A�/*

losszs<��k�       �	���IYc�A�/*

loss�M=?��       �	�t�IYc�A�/*

loss)n�;�^s�       �	��IYc�A�/*

loss��n<6�2�       �	��IYc�A�/*

lossh$k;�       �	�1�IYc�A�/*

loss��<�ep�       �	���IYc�A�/*

lossnc�<(��       �	c�IYc�A�/*

loss�=8A g       �	s��IYc�A�/*

loss���<��+       �	ڒ�IYc�A�/*

loss��;�       �	�+�IYc�A�/*

loss@ې<}$l�       �	t��IYc�A�/*

lossv��=$ɶ�       �	+j�IYc�A�/*

loss�;=þ�       �	��IYc�A�/*

loss�<���       �	5��IYc�A�/*

loss��{<��p       �	�.�IYc�A�/*

lossrU;;�sރ       �	���IYc�A�/*

lossM8=&T�       �	�X�IYc�A�/*

lossŉ�<G6.�       �	���IYc�A�/*

loss`�<�lM�       �	���IYc�A�/*

loss|�H<І�       �	� �IYc�A�/*

lossi�a=P��       �	���IYc�A�/*

losse0|<H�+       �	P��IYc�A�/*

loss2�K:K��       �	h$�IYc�A�/*

loss��K<��o�       �	R��IYc�A�/*

lossg+=���       �	�R�IYc�A�/*

losss<ڹ��       �	���IYc�A�/*

lossr�=�C       �	5@�IYc�A�/*

loss�b�<�γ       �	���IYc�A�/*

loss�O<�}f       �	���IYc�A�/*

lossJ�;];7�       �	|}�IYc�A�/*

loss.j;@�       �	_D�IYc�A�/*

losstܗ;M�dO       �	�C�IYc�A�/*

loss��=�-��       �	M��IYc�A�/*

loss��<�C��       �	���IYc�A�/*

loss���<�݂�       �	�y�IYc�A�/*

loss-��<2�r        �	�b�IYc�A�/*

loss�P�<婸       �	�IYc�A�/*

lossp͉<eL�       �	s��IYc�A�/*

loss��m<����       �	C7�IYc�A�/*

loss=��<4��U       �	F��IYc�A�/*

lossE�<u�O�       �	8h�IYc�A�/*

loss�=I�,       �	K�IYc�A�/*

lossC�l=�5�       �	#��IYc�A�/*

loss�w<F{�       �	6:�IYc�A�/*

loss���< )*B       �	���IYc�A�/*

loss&ۤ<����       �	�p�IYc�A�/*

loss��|<�w       �	��IYc�A�/*

loss�f�<��'�       �	���IYc�A�/*

lossݫ�<�ΞT       �	�G�IYc�A�/*

lossC�;#��       �	���IYc�A�/*

loss�<.�.�       �	%��IYc�A�/*

loss4��<qH�P       �	+�IYc�A�/*

loss�y=-��'       �	���IYc�A�/*

loss�-�<�~�{       �	�l�IYc�A�/*

lossE7?<y�/Q       �	�IYc�A�/*

lossS�>=Jw�8       �	^��IYc�A�/*

loss
�;B�`N       �	!s�IYc�A�/*

lossӓ;���       �	�IYc�A�/*

loss�{�;
3��       �	���IYc�A�/*

loss0�<]��       �	4G�IYc�A�/*

loss�=��F�       �	���IYc�A�/*

loss�vn<V���       �	�IYc�A�/*

lossҘ<C
��       �	��IYc�A�/*

loss��i<�28}       �	��IYc�A�/*

loss>��<
pb�       �	�� JYc�A�/*

loss�{t<v���       �	�VJYc�A�/*

loss�=�	�       �	�JYc�A�/*

lossm�<���       �	� JYc�A�/*

lossܔ<��       �	��JYc�A�/*

loss���<ծ       �	SZJYc�A�/*

lossV��<���O       �	�JYc�A�/*

loss���=���       �	F�JYc�A�/*

losst��<B���       �	�MJYc�A�/*

lossF�<4��W       �	i�JYc�A�/*

loss�%�;�_�       �	�xJYc�A�/*

loss�-�;�i,l       �	�1JYc�A�/*

loss�G;!I�       �	�JYc�A�/*

loss3�=S�       �	�i
JYc�A�/*

loss)�<��       �	ZJYc�A�/*

lossw��;s��:       �	��JYc�A�/*

loss�,=����       �	"OJYc�A�/*

loss�E�<�ܶ       �	%�JYc�A�/*

loss3��;�l�1       �	I�JYc�A�/*

loss�R=l���       �	�JYc�A�/*

loss�M@<#�ʁ       �	)�JYc�A�/*

loss���<�$       �	�PJYc�A�/*

loss�Ї;X���       �	��JYc�A�/*

loss��<�ױ�       �	�JYc�A�/*

loss�&�<fK�       �	>&JYc�A�/*

loss�< 5ϲ       �	P�JYc�A�/*

lossf��=V=�j       �	�yJYc�A�/*

lossC�~<�<Z       �	�JYc�A�/*

loss���<�)�       �	JYc�A�/*

loss�g[=P+O�       �	d#JYc�A�/*

loss��Q=��YJ       �	��JYc�A�/*

loss\��<L�M�       �	�XJYc�A�/*

lossѩ0=���Y       �	�JYc�A�/*

lossX'!=Fh��       �	��JYc�A�/*

lossR<�ԡ       �	�(JYc�A�/*

loss���<����       �	+�JYc�A�/*

loss��=ќ�O       �	�tJYc�A�/*

loss�f=p=�       �	vJYc�A�/*

loss���<�*�       �	��JYc�A�/*

loss�� =�UCx       �	S^JYc�A�/*

loss�&=���       �	��JYc�A�/*

loss���<f�X#       �	�JYc�A�/*

loss�=�8       �	p>JYc�A�/*

lossOǛ<
C �       �	f�JYc�A�/*

loss\/K<A--       �	p�JYc�A�/*

loss��t<nB�v       �	�HJYc�A�/*

loss��;�W�%       �	��JYc�A�/*

loss�':<�Df-       �	�| JYc�A�/*

loss׳k<�l�       �	�7!JYc�A�/*

lossĶ;�$>b       �	��!JYc�A�/*

loss)X�<�]��       �	�g"JYc�A�/*

loss 3�<�Z{       �	�#JYc�A�/*

loss��	=�cQ�       �	'�#JYc�A�0*

loss��=�.�       �	v$JYc�A�0*

lossJ�<�br       �	q%JYc�A�0*

losst~�;e�,       �	+�%JYc�A�0*

loss`�;|��       �	mW&JYc�A�0*

lossD�
;���+       �	Z�&JYc�A�0*

loss�(F=k�MX       �	4�'JYc�A�0*

loss���;F�h�       �	+N(JYc�A�0*

loss�^<���       �	�)JYc�A�0*

loss<)=(�       �	�)JYc�A�0*

lossxt�<`h�-       �	Ę*JYc�A�0*

loss_o9<3�]       �	�C+JYc�A�0*

loss�:;�F-�       �	��+JYc�A�0*

loss���<���       �	�,JYc�A�0*

lossx5.<��
       �	!�-JYc�A�0*

lossA�==?Xe       �	�.JYc�A�0*

loss�`�<��&�       �	��/JYc�A�0*

loss
�6<I�K       �	�0JYc�A�0*

lossEpK;Q���       �	�@1JYc�A�0*

loss�޷;q��q       �	��1JYc�A�0*

loss{* ;���/       �		�3JYc�A�0*

loss<$�</��       �	K4JYc�A�0*

loss�׸;��@       �	��4JYc�A�0*

loss���;��       �	S�5JYc�A�0*

loss_�?=T���       �	S>6JYc�A�0*

losss<�       �	(*7JYc�A�0*

lossɽ<t�       �	t�7JYc�A�0*

lossE�;6�E       �	�o8JYc�A�0*

loss[6�;f��       �	$9JYc�A�0*

lossW�;�Gm�       �	6�9JYc�A�0*

loss��J<�c��       �	�C:JYc�A�0*

lossS�<|�       �	��:JYc�A�0*

loss/я< ���       �	�;JYc�A�0*

loss	�b=��       �	"o<JYc�A�0*

loss��<%lJ       �	�
=JYc�A�0*

loss���<t�,^       �	��=JYc�A�0*

loss;V,=+9�       �	�?JYc�A�0*

loss�7j;4��       �	G�?JYc�A�0*

loss.�=�P��       �	Cq@JYc�A�0*

loss|3'<��Ċ       �	AJYc�A�0*

loss��;�A5�       �	�BJYc�A�0*

lossF@2<'��       �	�BJYc�A�0*

lossӚ�;jRK       �	;QCJYc�A�0*

lossW��<�{��       �	��CJYc�A�0*

loss{�D<(��        �	/�DJYc�A�0*

loss��<ߺ�       �	�!EJYc�A�0*

loss{f�;
�K@       �	�8FJYc�A�0*

loss��<:��_       �	��FJYc�A�0*

loss�t<�&�       �	@mGJYc�A�0*

lossA�;���       �	�HJYc�A�0*

loss�|<����       �	��HJYc�A�0*

loss쎁;��o       �	�4IJYc�A�0*

loss� u<6�`       �	t�IJYc�A�0*

loss1� =��       �	�uJJYc�A�0*

loss��P=���       �	eKJYc�A�0*

loss�rq=�ٺ       �	:�KJYc�A�0*

loss{�;w�ы       �	�GLJYc�A�0*

loss�^}=0^�,       �	D�LJYc�A�0*

loss,�<N�6       �	~MJYc�A�0*

lossv4I<�{�       �	aNJYc�A�0*

loss9�<��       �	
�NJYc�A�0*

loss��;�        �	+iOJYc�A�0*

loss�3Y=�8�L       �	�PJYc�A�0*

loss\cA<0�       �	8�PJYc�A�0*

loss�	=���       �	5EQJYc�A�0*

loss���<�)	�       �	-�QJYc�A�0*

loss=�<��       �	 �RJYc�A�0*

lossE�s<� <.       �	N%SJYc�A�0*

loss��<���>       �	��SJYc�A�0*

loss���;d�ā       �	��TJYc�A�0*

lossX;�%��       �	�)UJYc�A�0*

lossa�<bmR       �	�UJYc�A�0*

loss�5=M���       �	�_VJYc�A�0*

loss���<\���       �	�WJYc�A�0*

loss��<��       �	�WJYc�A�0*

loss�I=u�_       �	�SXJYc�A�0*

loss���<����       �	��XJYc�A�0*

loss�3<ai�?       �	��YJYc�A�0*

lossw��<;OB�       �	G ZJYc�A�0*

loss�$�<��*�       �	��ZJYc�A�0*

loss~�<z���       �	�[[JYc�A�0*

loss�fe;?H�j       �	��[JYc�A�0*

lossI��<�Sr�       �	��\JYc�A�0*

lossZu�<f���       �	s-]JYc�A�0*

loss�յ<Z�	       �	��]JYc�A�0*

loss�}�<\�Z       �	 �^JYc�A�0*

loss���<��       �	O=_JYc�A�0*

loss�a<���       �	��_JYc�A�0*

loss�Z;�z�       �	�`JYc�A�0*

loss�M�<U�Wc       �	�&aJYc�A�0*

loss)O=�
$@       �	Z�aJYc�A�0*

loss��<�f��       �	eUbJYc�A�0*

loss�E�<x�a�       �	��bJYc�A�0*

loss���<�.r       �	ܝcJYc�A�0*

loss�*�;$Gy�       �	�EdJYc�A�0*

loss��%=Dq8       �	��dJYc�A�0*

lossl_�;|���       �	O;fJYc�A�0*

lossn�<%4��       �	��fJYc�A�0*

loss��;|��c       �	SwgJYc�A�0*

loss�)�<��       �	�zhJYc�A�0*

loss �<�
��       �	�KiJYc�A�0*

loss�}W=��V�       �	�$jJYc�A�0*

loss�4�<[0e       �	��jJYc�A�0*

loss���<R��       �	��kJYc�A�0*

lossP��;Z�?U       �	BylJYc�A�0*

loss�=<=����       �	(�mJYc�A�0*

lossc1R<#���       �	�OnJYc�A�0*

loss��J=TD9^       �	^�nJYc�A�0*

loss�e�<��N�       �	��oJYc�A�0*

loss�p=�z       �	�9pJYc�A�0*

loss\7#<�\�       �	��pJYc�A�0*

lossu/;{�8       �	�qJYc�A�0*

loss�/2<sQ�!       �	tArJYc�A�0*

loss���<�       �	�MsJYc�A�0*

loss(�<<�qZ�       �	�#tJYc�A�0*

loss=յ<��ϊ       �	uJYc�A�0*

loss
#^<.�/�       �	�uJYc�A�0*

loss$d<����       �	7mvJYc�A�0*

lossEim=�Y��       �	M�wJYc�A�0*

loss���<����       �	
.xJYc�A�0*

losscu<IK�u       �	�qyJYc�A�0*

lossl�<�zN       �	zJYc�A�0*

loss�%<0{]�       �	��zJYc�A�0*

loss�=ѡ       �	2�{JYc�A�0*

lossߘ=��       �	�_|JYc�A�0*

lossҘ�<�˜�       �	}JYc�A�0*

loss���<��#       �	|+~JYc�A�0*

lossA =�Z4       �	��~JYc�A�0*

loss�̌<	�w�       �	��JYc�A�1*

loss��~=�r+�       �	�,�JYc�A�1*

lossw��<|kjn       �	��JYc�A�1*

loss$�+=��W^       �	̳�JYc�A�1*

loss]=ѓ^       �	�U�JYc�A�1*

loss�y3;O�Y�       �	y��JYc�A�1*

lossܢI<+`:       �	{L�JYc�A�1*

loss�<�.)�       �	ׅ�JYc�A�1*

lossr��<���       �	U��JYc�A�1*

loss�ۆ:�9e�       �	+k�JYc�A�1*

loss��<{�O�       �	�JYc�A�1*

losso}�<g�I�       �	F��JYc�A�1*

lossjT�<^$��       �	P��JYc�A�1*

loss�)=fa�@       �	?��JYc�A�1*

loss�S�<��#       �	PR�JYc�A�1*

loss_� =����       �	��JYc�A�1*

loss;^�<�W       �	%��JYc�A�1*

loss!�(=��pA       �	3T�JYc�A�1*

losslO�;t�H�       �	��JYc�A�1*

loss�6%<qh��       �	-��JYc�A�1*

loss��<)(�       �	O�JYc�A�1*

loss��u;k�AJ       �	��JYc�A�1*

loss
�:�a�       �	T��JYc�A�1*

loss8|Y=�3��       �	4,�JYc�A�1*

loss�Lt<54^:       �	�ڒJYc�A�1*

loss6N<��|       �	��JYc�A�1*

loss&=�bk&       �	?�JYc�A�1*

loss/�=+���       �	���JYc�A�1*

loss�W�<RZ�h       �	�_�JYc�A�1*

loss�B<��V�       �	���JYc�A�1*

loss���<l ��       �	��JYc�A�1*

loss��;{�z|       �	�I�JYc�A�1*

loss/o=�w��       �	��JYc�A�1*

loss/��<���       �	(��JYc�A�1*

lossn>�<Q�3�       �	"�JYc�A�1*

lossn��<~XqW       �	���JYc�A�1*

lossk�=�a��       �	�V�JYc�A�1*

loss��;�Ջ6       �	�JYc�A�1*

loss�~�;��       �	��JYc�A�1*

lossh�h<0k6�       �	"P�JYc�A�1*

loss�N�<�w�5       �	�JYc�A�1*

lossǦ <��Z       �	��JYc�A�1*

loss��;�H�       �	?9�JYc�A�1*

loss[��<u߰,       �	�ٞJYc�A�1*

loss�G�<��o�       �	�z�JYc�A�1*

loss��;U�Bq       �	n�JYc�A�1*

loss���<��Zq       �	��JYc�A�1*

loss�+�<	�!,       �	\V�JYc�A�1*

loss$��<�_a�       �	��JYc�A�1*

loss��==�r�f       �	9��JYc�A�1*

lossz�=�ܞ	       �	�W�JYc�A�1*

loss0(�<�9�=       �	��JYc�A�1*

loss�{�;0�)t       �	�JYc�A�1*

loss�:�;J� w       �	[&�JYc�A�1*

lossa��=-��
       �	dʥJYc�A�1*

loss�*�<�Jj�       �	�j�JYc�A�1*

loss�*�<Ϗ'F       �	��JYc�A�1*

loss?i=�[�       �	�JYc�A�1*

loss�;=pa��       �	t�JYc�A�1*

loss�<�<���       �	�0�JYc�A�1*

lossݩ+<��0[       �	`ͩJYc�A�1*

loss�z<�}lE       �	h�JYc�A�1*

loss,�;L�R�       �	��JYc�A�1*

lossH�P=�I��       �	���JYc�A�1*

lossJ��<���       �	D�JYc�A�1*

loss��=?�n       �	G�JYc�A�1*

loss��G<���       �	܃�JYc�A�1*

loss\܄<v�E�       �	� �JYc�A�1*

loss_�;E-       �	#��JYc�A�1*

loss@��<���       �	-`�JYc�A�1*

loss�ʆ<���       �	���JYc�A�1*

lossc�<�VAu       �	���JYc�A�1*

lossW`J<�(��       �	py�JYc�A�1*

lossm�=ª�d       �	�JYc�A�1*

loss*]b<O�\       �	���JYc�A�1*

loss��<�N\       �	�ĳJYc�A�1*

loss�W=����       �	^�JYc�A�1*

loss�W;=J�.       �	 �JYc�A�1*

loss��k<2�V       �	���JYc�A�1*

lossC��<��)	       �	�9�JYc�A�1*

lossŪ�;�k/       �	q�JYc�A�1*

loss�'1=K??       �	�{�JYc�A�1*

loss��o<gք�       �	/�JYc�A�1*

lossd�G=��>       �	���JYc�A�1*

loss�;.א       �	�J�JYc�A�1*

loss���<Aq['       �	`#�JYc�A�1*

loss�m=�z       �	���JYc�A�1*

loss)�M<[0��       �	�Y�JYc�A�1*

loss7W=���q       �	#��JYc�A�1*

lossl� <���W       �	2��JYc�A�1*

lossl��;�n       �	�-�JYc�A�1*

loss6�Q<y��w       �	ŽJYc�A�1*

loss��%=�9_�       �	_^�JYc�A�1*

lossr�;=�ʼ       �	���JYc�A�1*

lossi$g=��V,       �	3��JYc�A�1*

losso\=��{       �	���JYc�A�1*

loss�.�<ÇL�       �	`"�JYc�A�1*

loss�kE<̆�       �	 S�JYc�A�1*

loss|!�<|�B       �	�e�JYc�A�1*

loss�<�I{       �	p�JYc�A�1*

loss8��; ��       �	N��JYc�A�1*

loss$H9=f�9       �	���JYc�A�1*

loss�#b<����       �	Xu�JYc�A�1*

loss,�w<��=�       �	��JYc�A�1*

lossj��<���       �	���JYc�A�1*

loss��H=C�	"       �	k��JYc�A�1*

lossO��<�̍       �	ڐ�JYc�A�1*

loss	B3=; '       �	�,�JYc�A�1*

loss1`�<؎q       �	�#�JYc�A�1*

loss7�<��\�       �	"�JYc�A�1*

loss�y�<٦Sg       �	���JYc�A�1*

loss ��<7��       �	sg�JYc�A�1*

loss9F<&*Z       �	�JYc�A�1*

loss���;��       �	t��JYc�A�1*

loss}��<��"�       �	i9�JYc�A�1*

loss$�<�>       �	���JYc�A�1*

loss�Z<H�#       �	�t�JYc�A�1*

loss�s`;���       �	��JYc�A�1*

loss*S)<���I       �	���JYc�A�1*

loss�n�;>S%�       �	N�JYc�A�1*

loss\.:<H^[�       �	���JYc�A�1*

loss�Ɲ<٣��       �	Ӈ�JYc�A�1*

loss�2=]�p       �	i��JYc�A�1*

lossH�;���       �	�%�JYc�A�1*

loss_n�<��       �	��JYc�A�1*

loss��
<��RC       �	�U�JYc�A�1*

loss`u�<��t       �	9��JYc�A�1*

loss��;<�u�        �	׊�JYc�A�1*

lossf0T;��;       �	�$�JYc�A�2*

loss�|l<2C�       �	ӽ�JYc�A�2*

lossa�F<���2       �	`Z�JYc�A�2*

loss�xT=���C       �	���JYc�A�2*

loss��$;���       �	X��JYc�A�2*

loss4��<���j       �	�(�JYc�A�2*

loss���;6�       �	���JYc�A�2*

loss���<�|       �	]�JYc�A�2*

loss�u<���       �	���JYc�A�2*

lossl3�=4װd       �	^��JYc�A�2*

loss�8Y<���       �	:w�JYc�A�2*

loss��=���       �	�"�JYc�A�2*

lossx�r=��       �	Ӿ�JYc�A�2*

lossT�c<�~�l       �	t_�JYc�A�2*

loss�}�<����       �	���JYc�A�2*

loss�j�<W]fe       �	,��JYc�A�2*

loss�Q�<����       �	�?�JYc�A�2*

lossO�(<���       �	��JYc�A�2*

loss��;��v!       �	��JYc�A�2*

loss��=qB�       �	�%�JYc�A�2*

loss��=���       �	���JYc�A�2*

loss��=�^^u       �	�m�JYc�A�2*

loss�?=��       �	��JYc�A�2*

loss߸�;!��<       �	���JYc�A�2*

loss��=A��       �	vP�JYc�A�2*

lossQW<.B�       �	��JYc�A�2*

loss��=r6       �	�	�JYc�A�2*

loss��=e/�       �	-��JYc�A�2*

loss?	�<<Y�       �	�Z�JYc�A�2*

loss�U<�vp       �	�d�JYc�A�2*

loss��d<_!�Y       �	�D�JYc�A�2*

loss�==L���       �	��JYc�A�2*

loss W<��J-       �	���JYc�A�2*

loss��4<���       �	��JYc�A�2*

loss�E=H3�       �	�y�JYc�A�2*

lossn�<kj�K       �	�B�JYc�A�2*

lossZr�<iQ\�       �	p��JYc�A�2*

loss��Y<}�
       �	J��JYc�A�2*

loss���;��k�       �	G��JYc�A�2*

loss��i<"�Ve       �	#��JYc�A�2*

lossO��;�B�       �	�?�JYc�A�2*

lossĬr=.�Z]       �	��JYc�A�2*

lossv=�Rr�       �	���JYc�A�2*

loss���<�&y       �	���JYc�A�2*

loss���<�C�W       �	�v�JYc�A�2*

lossZn�;+���       �	��JYc�A�2*

loss:��<Vҳ�       �	��JYc�A�2*

loss$��<<x��       �	��JYc�A�2*

loss���<ֿ-$       �	ܷ�JYc�A�2*

loss`�<,��        �	�[�JYc�A�2*

loss��<�oR       �	RG�JYc�A�2*

lossn�=e�i?       �	�JYc�A�2*

lossvj�;�8p�       �	���JYc�A�2*

loss��<'j?       �	���JYc�A�2*

loss2�<�i>       �	>Y�JYc�A�2*

lossW��<��%[       �	� KYc�A�2*

loss��<�^��       �	�KYc�A�2*

loss�a�<�3�       �	�KYc�A�2*

loss��6=�W%       �	�KYc�A�2*

loss�,<�f�|       �	,KYc�A�2*

loss�=v�V�       �	ϟKYc�A�2*

loss�p�;��       �	�QKYc�A�2*

lossY��<�rV       �	��KYc�A�2*

loss� }<Q��       �	B�KYc�A�2*

loss!*=LY"H       �	~;KYc�A�2*

loss���<��TJ       �	%�KYc�A�2*

loss,@<���E       �	I�KYc�A�2*

loss�|\<ԫ�       �	�B	KYc�A�2*

loss�3'<�4R<       �	�	KYc�A�2*

loss�Cv;J�       �	�
KYc�A�2*

loss#�;�go%       �	9)KYc�A�2*

loss�5=��-�       �	�KYc�A�2*

lossw��<���2       �	2vKYc�A�2*

loss��=�9        �	�KYc�A�2*

loss�]�<s�       �	֬KYc�A�2*

lossa�<�b       �	CKYc�A�2*

loss�?;�S��       �	��KYc�A�2*

lossT�:�e�`       �	G�KYc�A�2*

loss0Ս<���c       �	�/KYc�A�2*

loss���<�       �	t�KYc�A�2*

loss��W<ˬ5       �	wiKYc�A�2*

loss��b=>i       �	�^KYc�A�2*

loss�r�<� �       �	-	KYc�A�2*

loss�D�<�qC�       �	��KYc�A�2*

loss8[�<�n       �	�=KYc�A�2*

loss�ZC<ҩc�       �	\�KYc�A�2*

lossSY�;�o�w       �	�KYc�A�2*

loss	g�;bK�j       �	�+KYc�A�2*

loss_w=�A�       �	��KYc�A�2*

loss��<����       �	��KYc�A�2*

loss�<����       �	�+KYc�A�2*

loss��<��8�       �	/�KYc�A�2*

loss`�<�%�       �	�cKYc�A�2*

lossh��:���Z       �	�KYc�A�2*

lossȴ�;�*�       �	��KYc�A�2*

loss��=�r~       �	�=KYc�A�2*

lossL<<��v�       �	��KYc�A�2*

loss��<-�P�       �	��KYc�A�2*

loss���<���9       �	�!KYc�A�2*

lossl��<�˻�       �	i�KYc�A�2*

loss�:�;�*�f       �	�]KYc�A�2*

loss�I�<��k       �	��KYc�A�2*

loss(�_;S�N       �	u�KYc�A�2*

loss;q=����       �	�3 KYc�A�2*

loss��@<�R3�       �	�� KYc�A�2*

lossJ�<3��F       �	yx!KYc�A�2*

loss:�=ܞ�=       �	�!"KYc�A�2*

lossM��;�A�       �	4�"KYc�A�2*

loss'�<	<��       �	!X#KYc�A�2*

lossn��<)A�       �	6$KYc�A�2*

lossd�=��/       �	=�$KYc�A�2*

loss���;U��i       �	�D%KYc�A�2*

loss�t=�]�        �	�%KYc�A�2*

loss8�;�޶N       �	�w&KYc�A�2*

loss P@=ZFa       �	�'KYc�A�2*

loss20Y<+��       �	W�'KYc�A�2*

loss�B�<)r��       �	�X(KYc�A�2*

loss�P<c���       �	�")KYc�A�2*

loss�0�;�!�k       �	��)KYc�A�2*

lossZ�<3�V�       �	�*KYc�A�2*

lossR�<@T4�       �	�Z+KYc�A�2*

lossօ�<B��       �	9	,KYc�A�2*

lossɦ�<����       �	��,KYc�A�2*

loss1C�:�TC�       �	P-KYc�A�2*

lossu<'��       �	��-KYc�A�2*

lossʰ=�,I       �	5�.KYc�A�2*

loss��<xх�       �	@/KYc�A�2*

loss�O/<���d       �	7�/KYc�A�2*

loss��<:�       �	�0KYc�A�3*

loss�<�=�&�       �	�[1KYc�A�3*

loss��=<��b�       �	�2KYc�A�3*

lossS^�;!��-       �	�2KYc�A�3*

loss�3<���       �	��3KYc�A�3*

loss���;-5�       �	fh4KYc�A�3*

loss�p�;4�F�       �	]5KYc�A�3*

loss�<��Yd       �	 �5KYc�A�3*

loss��;�pք       �	�o6KYc�A�3*

loss�a�;f�<�       �	�7KYc�A�3*

lossq�w<��M       �	��7KYc�A�3*

loss
>�;R�       �	y�8KYc�A�3*

lossFF&<�7�       �	�79KYc�A�3*

loss<Z��       �	��9KYc�A�3*

loss�-v:V�a       �	��:KYc�A�3*

loss�7�8=uV       �	�&;KYc�A�3*

loss��>;�S6�       �	�;KYc�A�3*

loss�_�<C�       �	?p<KYc�A�3*

lossC��<'A;�       �	r=KYc�A�3*

loss��<p��       �	�>KYc�A�3*

loss��;F��       �	��>KYc�A�3*

loss���<��{$       �	Uh?KYc�A�3*

lossJ�:΁l�       �	@KYc�A�3*

losso��<{��       �	��@KYc�A�3*

lossQ�==�h�       �	�EAKYc�A�3*

lossQ�=�;6       �	u�AKYc�A�3*

lossv^�<3̎�       �	^�BKYc�A�3*

loss�X<b��       �	�!CKYc�A�3*

loss���<Z�%       �	��CKYc�A�3*

loss®=2��       �	z�DKYc�A�3*

loss�
<D�H�       �	1�EKYc�A�3*

loss�@�<�'_2       �	�PFKYc�A�3*

loss���;S�}�       �	� GKYc�A�3*

loss�.<y��       �	�GKYc�A�3*

lossڔk<�ʶ       �	.HKYc�A�3*

loss��A<���       �	OIKYc�A�3*

loss��<]��       �	��IKYc�A�3*

lossJG%=�"MZ       �	��JKYc�A�3*

loss���<9'*       �	!WKKYc�A�3*

loss͎�<:�$
       �	H�KKYc�A�3*

loss?8�<5h�       �	�LKYc�A�3*

lossO=�<� �!       �	�>MKYc�A�3*

loss��;ｸ�       �	��MKYc�A�3*

losso��<�D�V       �	�uNKYc�A�3*

loss���<WT       �	
OKYc�A�3*

loss��/;aYF        �	��OKYc�A�3*

loss�<�~z       �	`PKYc�A�3*

loss��<��T       �	b�PKYc�A�3*

loss�z&=�/+       �	�QKYc�A�3*

loss
mz<�1��       �	t)RKYc�A�3*

loss\k=���Z       �	��RKYc�A�3*

loss�|6=���       �	5�SKYc�A�3*

loss�H�<�a�5       �	a5TKYc�A�3*

loss��<Գ�;       �	O�TKYc�A�3*

lossF��;�-��       �	�|UKYc�A�3*

loss,�;<��c       �	�VKYc�A�3*

loss=.<&��       �	��VKYc�A�3*

loss�'<��
       �	�VWKYc�A�3*

loss}��;� O       �	��WKYc�A�3*

loss �{<��f�       �	͏XKYc�A�3*

loss>��=
!�Z       �	|+YKYc�A�3*

loss�Q<PX       �	p
ZKYc�A�3*

loss���:�e        �	��ZKYc�A�3*

loss&¸<�I��       �	;S[KYc�A�3*

loss
b<��       �	�\KYc�A�3*

loss��9<lj�`       �	�\KYc�A�3*

loss�(�;nNF       �	;:]KYc�A�3*

loss@G<q}K�       �	��]KYc�A�3*

loss=}9=��       �	Yn^KYc�A�3*

loss��\;cFB�       �	�,_KYc�A�3*

loss��<�=w�       �	��_KYc�A�3*

loss���:>:8�       �	�w`KYc�A�3*

loss�<����       �	�aKYc�A�3*

loss�̺;ت~       �	 ��KYc�A�3*

loss��<j��G       �	#J�KYc�A�3*

loss���<���       �	��KYc�A�3*

lossq��<�M       �	${�KYc�A�3*

loss��<� a       �	�KYc�A�3*

lossM��<���       �	.��KYc�A�3*

loss�A=>�C       �	HO�KYc�A�3*

loss*��<��       �	K�KYc�A�3*

loss�(=c�_       �	΋�KYc�A�3*

loss�\B=M�V>       �	�#�KYc�A�3*

loss���<�'T_       �	��KYc�A�3*

loss��<����       �	�Z�KYc�A�3*

loss�(=�m�       �	���KYc�A�3*

lossA�<*5��       �	h��KYc�A�3*

losss)<9
�T       �	䆉KYc�A�3*

lossh -;�=t�       �	> �KYc�A�3*

lossx(C:V_+       �	��KYc�A�3*

lossЬ�<;���       �	�U�KYc�A�3*

loss�<V?}t       �	���KYc�A�3*

loss�w=�zz�       �	ьKYc�A�3*

loss'}<0Y�       �	wf�KYc�A�3*

loss	'<{��       �	���KYc�A�3*

loss�;ޭ��       �	���KYc�A�3*

loss$��<��u       �	�3�KYc�A�3*

loss��;�Ϩ�       �	͏KYc�A�3*

lossK�;���       �	sg�KYc�A�3*

loss�	<�b�$       �	%"�KYc�A�3*

loss�.�; ��       �	��KYc�A�3*

lossF�-=�Mgk       �	�q�KYc�A�3*

loss���;ě4�       �	�	�KYc�A�3*

loss)�<p��8       �	���KYc�A�3*

lossx�O<�j�       �	F�KYc�A�3*

lossv��;���       �	�ߔKYc�A�3*

lossL=e?�$       �	�y�KYc�A�3*

loss�z�<\Xs�       �	�KYc�A�3*

loss�=�30       �	���KYc�A�3*

loss�Y�:�3ğ       �	�T�KYc�A�3*

loss3e�<i�i�       �	)�KYc�A�3*

loss�e�=t�AF       �	���KYc�A�3*

lossί=4�t.       �	x��KYc�A�3*

lossE]m=}�R       �	/4�KYc�A�3*

loss�<T��       �	�ΚKYc�A�3*

loss���;T��=       �	gd�KYc�A�3*

loss�=��}�       �	�KYc�A�3*

loss�=i<�n�z       �	���KYc�A�3*

lossW�U=���       �	�>�KYc�A�3*

loss4�<|��S       �	ܝKYc�A�3*

loss�=�P|�       �	๞KYc�A�3*

loss^f�<�s��       �	rS�KYc�A�3*

loss�+g:Xg��       �	��KYc�A�3*

loss�U�<�a�       �	���KYc�A�3*

loss��<�/�       �	�9�KYc�A�3*

lossTw<5��       �	ҡKYc�A�3*

loss�v<=;��       �	�r�KYc�A�3*

lossJ�2<�1B�       �	��KYc�A�3*

loss�)<,>�       �	m��KYc�A�4*

lossL��:�>�;       �	�\�KYc�A�4*

loss�T�;b�       �	D��KYc�A�4*

lossql�;E���       �	���KYc�A�4*

loss�=�Ӿ/       �	�0�KYc�A�4*

loss�a=���       �	�KYc�A�4*

loss_Y�<0��       �	d��KYc�A�4*

lossl�;|;-l       �	R��KYc�A�4*

lossɸe<���       �	���KYc�A�4*

loss�x<��U�       �	U�KYc�A�4*

lossyz<�ZZ       �	U��KYc�A�4*

lossyM<w��}       �	aƫKYc�A�4*

loss��=���2       �	@l�KYc�A�4*

lossM��<�	�       �	��KYc�A�4*

loss�� =VNQy       �	bڭKYc�A�4*

loss�u<2��       �	/��KYc�A�4*

loss,W�<�HI=       �	7��KYc�A�4*

lossrI�<�31n       �	�a�KYc�A�4*

loss`�<��*       �	�KYc�A�4*

loss3FT<��a.       �	Z��KYc�A�4*

loss��<�9�n       �	$D�KYc�A�4*

loss�}�;���       �	K�KYc�A�4*

loss2��<�x��       �	Ƈ�KYc�A�4*

loss��;-��       �	�KYc�A�4*

lossrQ7<����       �	ôKYc�A�4*

loss}��<J��*       �	Me�KYc�A�4*

loss�ݮ<S��b       �	2�KYc�A�4*

loss��*<�,�g       �	4��KYc�A�4*

loss�b�;�Sg       �	�3�KYc�A�4*

lossf5�<[��R       �	�ѷKYc�A�4*

loss�<M���       �	�k�KYc�A�4*

loss�J1<��+�       �	u�KYc�A�4*

loss���;�[��       �	�KYc�A�4*

loss��<vT�)       �	�:�KYc�A�4*

loss�Η<�	K�       �	bںKYc�A�4*

loss�y<���%       �	1z�KYc�A�4*

loss��<Y�[       �	��KYc�A�4*

lossli<c��D       �	���KYc�A�4*

loss��+=Ň        �	�F�KYc�A�4*

loss���<�h       �	S�KYc�A�4*

lossn��;��u�       �	��KYc�A�4*

loss��<c��       �	�!�KYc�A�4*

loss{<"X�       �	ؼ�KYc�A�4*

loss.�s<i�U@       �	P��KYc�A�4*

loss#*<C�pc       �	l$�KYc�A�4*

loss�HW<Pr�       �	Q��KYc�A�4*

lossʿ =�Q3       �	uU�KYc�A�4*

lossę�<����       �	x��KYc�A�4*

loss�0;���F       �	���KYc�A�4*

loss-P-<lDo       �	��KYc�A�4*

loss���<4��       �	�2�KYc�A�4*

loss:�<��.       �	u��KYc�A�4*

loss��h<"��T       �	�l�KYc�A�4*

loss �w<�T�y       �	w1�KYc�A�4*

loss6��;]�       �	���KYc�A�4*

loss�ci=�j�       �	�^�KYc�A�4*

loss�1 <��+       �	���KYc�A�4*

loss�Vn<���       �	i��KYc�A�4*

loss���<�J>#       �	x(�KYc�A�4*

loss�qv<1
X       �	��KYc�A�4*

loss-6�:�ڀ�       �	�V�KYc�A�4*

loss;<l�G       �	��KYc�A�4*

loss�/=�Nk       �	��KYc�A�4*

loss=�=��PG       �	D�KYc�A�4*

loss,�<��*4       �	Y��KYc�A�4*

loss�*�<Ic
�       �	px�KYc�A�4*

loss�*=J%N�       �	��KYc�A�4*

loss�x;���<       �	I��KYc�A�4*

loss��<.4�       �	�T�KYc�A�4*

loss#�;rL`       �	���KYc�A�4*

lossJrh<oڢH       �	7��KYc�A�4*

loss���;�Á�       �	$�KYc�A�4*

loss��;|D�       �	w��KYc�A�4*

lossq�=�n��       �	eR�KYc�A�4*

loss�CR<���l       �	��KYc�A�4*

loss�n�;�uT       �	#��KYc�A�4*

loss6&�<B       �	��KYc�A�4*

loss��W<h'��       �	���KYc�A�4*

loss{Q�<[�!       �	�y�KYc�A�4*

loss��=:��       �	w�KYc�A�4*

loss:ޅ<�`�       �	��KYc�A�4*

loss��m;47Fr       �	�E�KYc�A�4*

losscRD;����       �	m��KYc�A�4*

lossxr<p,>7       �	χ�KYc�A�4*

loss]b2<�oY       �	c&�KYc�A�4*

loss��;�Q�;       �	��KYc�A�4*

loss�,+<m,�I       �	���KYc�A�4*

loss.}=�_��       �	d��KYc�A�4*

loss�I�<6I�       �	�M�KYc�A�4*

loss"O=@<�       �	���KYc�A�4*

loss/�<F�]2       �	_'�KYc�A�4*

loss��;'�       �	���KYc�A�4*

loss�p;��a�       �	�g�KYc�A�4*

loss�r<����       �	h�KYc�A�4*

loss��#<�y
�       �	կ�KYc�A�4*

loss���;��\�       �	�J�KYc�A�4*

loss��<۳�Q       �	s��KYc�A�4*

loss��<xEe       �	?��KYc�A�4*

loss?B�=��Fc       �	�6�KYc�A�4*

loss&�&<m��#       �	���KYc�A�4*

loss�
;�`�       �	�r�KYc�A�4*

lossv�Q<�n��       �	T�KYc�A�4*

loss�#�<�d�       �	v��KYc�A�4*

loss֞�<ʦ
W       �	 ]�KYc�A�4*

loss�<ૃ�       �	���KYc�A�4*

loss�M�<C���       �	A��KYc�A�4*

loss1�<n���       �	_�KYc�A�4*

loss��<�       �	���KYc�A�4*

loss!�; ��       �	A��KYc�A�4*

loss��N=��1?       �	�n�KYc�A�4*

loss:,�;��Z       �	��KYc�A�4*

loss�.p;08<�       �	���KYc�A�4*

loss���<N{*       �	6v�KYc�A�4*

loss3M?=ɔia       �	V�KYc�A�4*

loss��	<���       �	���KYc�A�4*

loss�dv<�%�t       �	��KYc�A�4*

lossí�;�Bb�       �	�q�KYc�A�4*

loss�3<?�I�       �	��KYc�A�4*

loss 9N=�[�:       �	X�KYc�A�4*

loss̟�;�@�       �	&��KYc�A�4*

loss&��<<K�3       �	���KYc�A�4*

loss��c<�\#1       �	��KYc�A�4*

lossV4=���       �	{��KYc�A�4*

loss?�=�ec�       �	W[�KYc�A�4*

lossN�<��T       �	���KYc�A�4*

loss��;
���       �	L��KYc�A�4*

loss� �<#�a       �	�"�KYc�A�4*

loss1�=<E��       �	���KYc�A�4*

loss)>R<�$S       �	ޏ�KYc�A�5*

loss��:<
�c7       �	�(�KYc�A�5*

loss��;iT�       �	��KYc�A�5*

lossTa�<�X�       �	y]�KYc�A�5*

loss��<�c��       �	���KYc�A�5*

loss�V�<��1�       �	-��KYc�A�5*

loss��<�}�	       �	2�KYc�A�5*

loss��/=��u�       �	b��KYc�A�5*

loss�<ĩB       �	|�KYc�A�5*

lossW�;���M       �	T�KYc�A�5*

loss��;s�2       �	k��KYc�A�5*

loss�;ʪ��       �	�p LYc�A�5*

lossRK2<���       �	B	LYc�A�5*

lossc��;C��       �	��LYc�A�5*

loss�]2==���       �	�DLYc�A�5*

loss4%<=�ͺ;       �	��LYc�A�5*

loss��m;�ՠ�       �	){LYc�A�5*

loss�o<��       �	<LYc�A�5*

losstL'<t��I       �	LULYc�A�5*

lossB�<�'�       �	ҋLYc�A�5*

lossc[�=$�F�       �	/�LYc�A�5*

loss�T<;���       �	\=LYc�A�5*

loss��B<��|x       �	�;	LYc�A�5*

loss��=���	       �	��	LYc�A�5*

lossL�<1H�       �	��
LYc�A�5*

loss��<�Zȵ       �	�9LYc�A�5*

loss&$'<D��       �	]�LYc�A�5*

loss��<�&0	       �	tLYc�A�5*

loss�6<�oz       �	rLYc�A�5*

lossD��;dr�'       �	�LYc�A�5*

loss̝I<mN��       �	��LYc�A�5*

loss���<ɯO1       �	iQLYc�A�5*

lossER =ò�       �	/�LYc�A�5*

lossz�C=�n��       �	4�LYc�A�5*

loss��<���       �	�ELYc�A�5*

loss%��=�i�5       �		�LYc�A�5*

lossd�=�,�       �	o�LYc�A�5*

loss�&�<��-�       �	LYc�A�5*

lossA8�<��ui       �	N�LYc�A�5*

loss_��<-���       �	�oLYc�A�5*

loss�p5=�?׹       �	�LYc�A�5*

loss��;���       �	2�LYc�A�5*

lossS=�0"�       �	�DLYc�A�5*

loss��Z<��3�       �	��LYc�A�5*

loss$j<Q�&�       �	��LYc�A�5*

loss�"b=
�n_       �	�LYc�A�5*

lossﲘ<�	<�       �	�LYc�A�5*

loss�2I=&
       �	��LYc�A�5*

loss�GR;:�       �	��LYc�A�5*

loss���<���.       �	�7LYc�A�5*

lossS�<�I�9       �	�LYc�A�5*

loss��<e�k       �	J�LYc�A�5*

lossL�M<�֞�       �	CLYc�A�5*

lossӷ�<��͇       �	��LYc�A�5*

loss��n=���H       �	��LYc�A�5*

lossHgQ={���       �	J�LYc�A�5*

loss�z=�KJ       �	�2 LYc�A�5*

loss�K�;9}C       �	� LYc�A�5*

lossn"-=�7p       �	!LYc�A�5*

lossq
=|�OP       �	"LYc�A�5*

loss{�l=[�+       �	�"LYc�A�5*

loss@Ħ<f��       �	�`#LYc�A�5*

loss��<E���       �	$LYc�A�5*

loss\�?<��       �	ܝ$LYc�A�5*

loss�;9=)�+�       �	 8%LYc�A�5*

loss�ժ<^j��       �	I�%LYc�A�5*

losss�D<�@       �	��&LYc�A�5*

loss@ 6<1|T�       �	3'LYc�A�5*

loss��=��L       �	x�'LYc�A�5*

lossV�<a�a}       �	�z(LYc�A�5*

loss���<�#G       �	�a)LYc�A�5*

lossv�*;Z�_       �	�;*LYc�A�5*

loss<�<���d       �	6+LYc�A�5*

loss��m<��       �	��+LYc�A�5*

loss�?s<L�       �	��,LYc�A�5*

loss�Z<c�       �	k�-LYc�A�5*

lossć=j$       �	�0/LYc�A�5*

loss)D�;��J�       �	*0LYc�A�5*

loss��;m       �	��0LYc�A�5*

loss��<A?iW       �	�l1LYc�A�5*

lossHf�;捑h       �	;2LYc�A�5*

lossy��<B]       �	��2LYc�A�5*

lossv\ ;��*       �	�{3LYc�A�5*

loss&�=F�N"       �	�04LYc�A�5*

loss�<�d�       �	m�4LYc�A�5*

loss�(=�#'       �	��5LYc�A�5*

losssk=T��8       �	j6LYc�A�5*

lossf�=�8��       �	
7LYc�A�5*

loss
=�<4��        �	��7LYc�A�5*

loss܅b<N�R�       �	c�8LYc�A�5*

loss�y<kQo�       �	��9LYc�A�5*

loss̈́=�̀       �	��:LYc�A�5*

lossҊ
= ���       �	�1;LYc�A�5*

losseJ�;�AeL       �	�N<LYc�A�5*

loss1�<|���       �	x�<LYc�A�5*

loss�:<��"       �	��=LYc�A�5*

lossVyO=�       �	k,>LYc�A�5*

loss�ׅ:@�]�       �	*�>LYc�A�5*

loss��/=91u�       �	{?LYc�A�5*

loss�@�<�g&�       �	�@LYc�A�5*

loss$�<W:f       �	��@LYc�A�5*

loss��<�''       �	hALYc�A�5*

losst]�<P��       �	4BLYc�A�5*

loss�D2=�bĬ       �		�BLYc�A�5*

loss��<�j>�       �	%ACLYc�A�5*

loss�T)<��-       �	��CLYc�A�5*

loss!G�;!q�       �	ˠDLYc�A�5*

loss�|�:Za�*       �	�>ELYc�A�5*

lossβ2<%�jE       �	��ELYc�A�5*

loss�h�;��;+       �	;�FLYc�A�5*

lossl�5;���       �	5GLYc�A�5*

loss.�<�z@�       �	[�GLYc�A�5*

loss$P�<]ȫ       �	 {HLYc�A�5*

lossvQ�;�de6       �	'ILYc�A�5*

lossc�H=Ւ��       �	/�ILYc�A�5*

loss�X�<#x�       �	�ZJLYc�A�5*

loss=${<}Qe       �	]�JLYc�A�5*

loss�<�<*i&       �	d�KLYc�A�5*

loss�=��L�       �	u:LLYc�A�5*

loss�5�;t#{�       �	D�LLYc�A�5*

loss��=U�\       �	��MLYc�A�5*

loss ��<G��       �	�NLYc�A�5*

lossZ)=w�֖       �	��NLYc�A�5*

lossƊ�<��F       �	�\OLYc�A�5*

lossEWK<~��       �	nPLYc�A�5*

loss��<@���       �	%�PLYc�A�5*

loss��v<�D       �	�GQLYc�A�5*

loss@�a<!��&       �	�QLYc�A�5*

loss_�L<�,_�       �	�RLYc�A�6*

loss���<׀tU       �	WASLYc�A�6*

loss�<��[       �	��SLYc�A�6*

lossŦ�<I���       �	�TLYc�A�6*

loss���<2��       �	�CULYc�A�6*

loss&��;�0�       �	!�ULYc�A�6*

loss8K<dO�t       �	��VLYc�A�6*

loss6�=��       �	/5WLYc�A�6*

loss���<���/       �	+�WLYc�A�6*

loss�f�<��.�       �	OuXLYc�A�6*

loss��=t�c�       �	YLYc�A�6*

loss�n==no��       �	g�YLYc�A�6*

lossL��;D���       �	-ZZLYc�A�6*

lossև�=[:�y       �	7�ZLYc�A�6*

loss�n=Q�`�       �	/�[LYc�A�6*

loss��=�R�       �	�K\LYc�A�6*

loss��)=�݉7       �	.�\LYc�A�6*

lossL��<���u       �	_^LYc�A�6*

loss�%k=�H+       �	��^LYc�A�6*

loss@��;o �c       �	R_LYc�A�6*

loss?�=��e�       �	��_LYc�A�6*

loss���;����       �	N�`LYc�A�6*

loss�c&<��~�       �	�TaLYc�A�6*

loss�B=�       �	,�aLYc�A�6*

lossO~�<qz��       �	2�bLYc�A�6*

lossڨ�<�;J       �	�&cLYc�A�6*

loss*C<�B/       �	��cLYc�A�6*

lossA�<
� �       �	�VdLYc�A�6*

loss��:���       �	��dLYc�A�6*

loss��><z��       �	5�eLYc�A�6*

loss���<i��       �	�MfLYc�A�6*

lossHQ/<Nx�!       �	C�fLYc�A�6*

loss���;I�Uw       �	c|gLYc�A�6*

loss�}�=�d��       �	�hLYc�A�6*

loss�e�<��u       �	X�hLYc�A�6*

loss4�;`t+�       �	IiLYc�A�6*

loss�=��Yf       �	�jLYc�A�6*

loss�U�<ê��       �	[�jLYc�A�6*

loss�<+�)�       �	�IkLYc�A�6*

loss�=�<��kL       �	�
lLYc�A�6*

losssy<��8y       �	z�lLYc�A�6*

lossaOF<N�ȹ       �	DMmLYc�A�6*

lossɬ�<~�L�       �	K�mLYc�A�6*

loss\��<���       �	�|nLYc�A�6*

loss��Q<���       �	LoLYc�A�6*

loss�^�<�`O�       �	��oLYc�A�6*

loss�r�<�&       �	�OpLYc�A�6*

loss�	}<zΖ�       �	��pLYc�A�6*

loss:��;��O       �	��qLYc�A�6*

loss��7< �       �	`rLYc�A�6*

loss�B�;��l5       �	 �rLYc�A�6*

loss+<�<����       �	LsLYc�A�6*

loss�z=z4D&       �	�sLYc�A�6*

lossMv�=�	5       �	�tLYc�A�6*

lossګ�<����       �	�HuLYc�A�6*

loss��!=ے��       �	q�uLYc�A�6*

loss��<����       �	�vLYc�A�6*

loss[�l;���       �	�wLYc�A�6*

loss,�p<��       �	p�wLYc�A�6*

loss���<2�l�       �	�PxLYc�A�6*

lossMW <[       �	��xLYc�A�6*

loss�$<���       �	I�yLYc�A�6*

lossm˰<K�       �	� zLYc�A�6*

loss�V�<KB.(       �	�zLYc�A�6*

loss�է<Ĺ��       �	~R{LYc�A�6*

lossŵ`<�Y       �	|�{LYc�A�6*

loss#&�<��nl       �	K�|LYc�A�6*

loss�6=:f/�       �	�'}LYc�A�6*

lossᣊ<�M��       �	��}LYc�A�6*

loss��B<T{"c       �	�b~LYc�A�6*

loss��v<��k       �	p
LYc�A�6*

lossC�	=Plu�       �	�LYc�A�6*

loss퇪;ႛ�       �	=�LYc�A�6*

loss�|;�c%�       �	9ԀLYc�A�6*

loss�.�<1�~       �	�l�LYc�A�6*

lossm_�<.hKv       �	��LYc�A�6*

lossJ�@<�q       �	ܠ�LYc�A�6*

loss�ͻ;��u{       �	�8�LYc�A�6*

loss�`�<�$�;       �	wۃLYc�A�6*

loss��E<�},z       �	��LYc�A�6*

loss4a�<���)       �	�O�LYc�A�6*

loss(/=v:�m       �	��LYc�A�6*

loss�7�<T��       �	���LYc�A�6*

losso�:<��t       �	��LYc�A�6*

loss?%�<t�,r       �	���LYc�A�6*

loss7�:����       �	�O�LYc�A�6*

loss�M�;�C�       �	��LYc�A�6*

loss���<ܧ�h       �	ࢊLYc�A�6*

loss��<A[ț       �	i6�LYc�A�6*

loss�a�:�S       �	�\�LYc�A�6*

loss��1< e)�       �	9�LYc�A�6*

lossz�<��       �	{��LYc�A�6*

lossD��<�
       �	�"�LYc�A�6*

loss�G=�QFa       �	���LYc�A�6*

loss���;ʖ�       �	[]�LYc�A�6*

lossk�=D1��       �	�4�LYc�A�6*

loss�=��YJ       �	ԐLYc�A�6*

loss���<�/NW       �	'l�LYc�A�6*

loss8B%<8�       �	q�LYc�A�6*

loss�}�<A�       �	=��LYc�A�6*

loss��*=}��       �	�-�LYc�A�6*

loss�o=ۄ�       �	�ÓLYc�A�6*

loss��==�k�       �	`Y�LYc�A�6*

lossl�{=��       �	�LYc�A�6*

loss8'F=á��       �	3��LYc�A�6*

loss�02<���       �	$%�LYc�A�6*

loss���;#���       �	ӿ�LYc�A�6*

losst�< ��v       �	Z�LYc�A�6*

loss�=�_�q       �	
K�LYc�A�6*

loss���<�iFw       �	� �LYc�A�6*

loss�� =��)       �	���LYc�A�6*

loss�Xh;�&0�       �	�L�LYc�A�6*

losscT�<���`       �	l�LYc�A�6*

lossL�6=�%��       �	]ÛLYc�A�6*

loss	@=)��Y       �	�[�LYc�A�6*

loss%��<�J��       �	���LYc�A�6*

loss�~�=�n�j       �	��LYc�A�6*

loss�?P<C��       �	O�LYc�A�6*

loss$"A<�;Vj       �	��LYc�A�6*

loss�H�<�/�I       �	沟LYc�A�6*

lossm��<��Z       �	�P�LYc�A�6*

loss��'<�%>�       �	��LYc�A�6*

loss[�2<]��       �	���LYc�A�6*

loss=�V�       �	�:�LYc�A�6*

loss��&=��u@       �	��LYc�A�6*

loss��<g��N       �	s��LYc�A�6*

lossC6F=1�f�       �	g&�LYc�A�6*

lossSE�<�NC�       �	�ĤLYc�A�6*

loss@PC<��b       �	se�LYc�A�7*

lossɕ�<����       �	���LYc�A�7*

loss�,
=Ԓz       �	d��LYc�A�7*

loss|@�<��j,       �	vQ�LYc�A�7*

loss:�7=l��t       �	F�LYc�A�7*

loss}��;�$��       �	��LYc�A�7*

loss���<2-       �	��LYc�A�7*

lossC<�=�σ       �	5��LYc�A�7*

loss���<���       �	M�LYc�A�7*

loss�7�<=JZ"       �	R�LYc�A�7*

loss�l<z^       �	6�LYc�A�7*

lossA�<��s       �	%�LYc�A�7*

loss���;�mM�       �	���LYc�A�7*

loss��<��c�       �	KV�LYc�A�7*

losst'*<~+��       �	���LYc�A�7*

loss���<?2J[       �	Q��LYc�A�7*

lossN��<��'       �	kJ�LYc�A�7*

loss�<��j       �	.�LYc�A�7*

lossK�#=�8Ǥ       �	LYc�A�7*

lossq�'<�D�O       �	0�LYc�A�7*

lossA32=�M�:       �	�̲LYc�A�7*

loss	��<���       �	fg�LYc�A�7*

loss��<���       �	`�LYc�A�7*

loss�M�;�#u1       �	�޴LYc�A�7*

loss��,=��|�       �	�w�LYc�A�7*

lossFܾ<Q���       �	��LYc�A�7*

loss�\;Ē��       �	���LYc�A�7*

loss��2<��eo       �	�h�LYc�A�7*

loss��<`�I�       �	�LYc�A�7*

lossq<G9!�       �	���LYc�A�7*

lossZ�T;~o\       �	;V�LYc�A�7*

loss�%=���}       �	s�LYc�A�7*

loss��<@4�       �	牺LYc�A�7*

loss���<�9       �	�&�LYc�A�7*

lossrT�<���       �	�ûLYc�A�7*

loss�R#=#�N�       �	l]�LYc�A�7*

lossV�h;���       �	���LYc�A�7*

loss�*�:�G��       �	R��LYc�A�7*

loss|<yd��       �	�8�LYc�A�7*

loss,�!<x~B       �	�ؾLYc�A�7*

lossv�<Ib�       �	in�LYc�A�7*

loss;�*=5y       �	%�LYc�A�7*

loss��w<��       �	M��LYc�A�7*

loss�-�<�jL�       �	!;�LYc�A�7*

lossWGE;!8FC       �	0��LYc�A�7*

lossw��<4���       �	�y�LYc�A�7*

loss�=�;�N��       �	��LYc�A�7*

loss]' =L�R       �	��LYc�A�7*

loss�$�<�W��       �	uY�LYc�A�7*

loss�|c;�mj
       �	��LYc�A�7*

loss�(<�
fc       �	)��LYc�A�7*

loss9�=	;�$       �	�5�LYc�A�7*

loss��_<�Dk       �	���LYc�A�7*

loss/�<��        �	�n�LYc�A�7*

loss/ Z=��F�       �	h�LYc�A�7*

loss���<.���       �	Ԟ�LYc�A�7*

lossJ�Q<5�
       �	5�LYc�A�7*

loss���<Za�       �	:��LYc�A�7*

loss�|�<�=T       �	9d�LYc�A�7*

loss��p<�'�       �	4��LYc�A�7*

loss�t=IQ       �	���LYc�A�7*

lossA�	=����       �	�;�LYc�A�7*

loss���;w\\�       �	���LYc�A�7*

lossc�.=X,<�       �	�n�LYc�A�7*

loss��<��{�       �	��LYc�A�7*

loss�C�<l�       �	���LYc�A�7*

loss��;�X��       �	G�LYc�A�7*

loss� �<_��x       �	K��LYc�A�7*

lossV<B<}ph�       �	7��LYc�A�7*

loss�`&=���       �	��LYc�A�7*

loss��$<0=y       �	��LYc�A�7*

loss�/<D��       �	f��LYc�A�7*

loss�<(1n�       �	j�LYc�A�7*

loss�S�;��f�       �	B�LYc�A�7*

loss�x�<�Y�       �	���LYc�A�7*

lossw�k<hF�       �	�O�LYc�A�7*

loss��7<�X��       �	t��LYc�A�7*

loss�g`<H���       �	|��LYc�A�7*

loss�+�;�|h       �	L�LYc�A�7*

lossZ�<}��       �	���LYc�A�7*

loss1|<<��       �	��LYc�A�7*

loss|�d;5.|j       �	��LYc�A�7*

loss���;,�
�       �	�J�LYc�A�7*

loss��;l`��       �	 ��LYc�A�7*

lossibn;S�5       �	 z�LYc�A�7*

loss�N,=����       �	��LYc�A�7*

lossOW<E^��       �	���LYc�A�7*

loss�"�≮��       �	�D�LYc�A�7*

loss_�n=�a��       �	I��LYc�A�7*

loss�P�=^`�u       �	�s�LYc�A�7*

lossĊ�<1��\       �	�LYc�A�7*

loss���<���       �	ګ�LYc�A�7*

loss�y�<���       �	�@�LYc�A�7*

loss�-<����       �	i�LYc�A�7*

loss��V;�K�       �	���LYc�A�7*

lossR;<i��       �	<K�LYc�A�7*

lossA��:��w�       �	��LYc�A�7*

loss���;Zܲ       �	*��LYc�A�7*

loss�['<-Kf�       �	�/�LYc�A�7*

loss�^;��6-       �	���LYc�A�7*

loss1m<2�ڡ       �	�h�LYc�A�7*

loss��P;ʃ�       �	��LYc�A�7*

loss�A�9*���       �	���LYc�A�7*

lossA;�8\�=k       �	�@�LYc�A�7*

loss,�n:K���       �	w��LYc�A�7*

lossQL�<@|�       �	Sw�LYc�A�7*

loss���<J�       �	��LYc�A�7*

loss���:�,�`       �	��LYc�A�7*

loss��$;���4       �	b��LYc�A�7*

lossT��<�Vh�       �	~��LYc�A�7*

loss|��:��DR       �	-@�LYc�A�7*

lossE��<��=T       �	(�LYc�A�7*

loss�S�=q�Nk       �	��LYc�A�7*

loss��8<U�0       �	=��LYc�A�7*

loss�J�<�4Y       �	��LYc�A�7*

lossN�<�"�       �	v�LYc�A�7*

loss�n�<�4�j       �	B@�LYc�A�7*

loss1/<U�       �	f��LYc�A�7*

losss=I3��       �	���LYc�A�7*

loss��<�       �	���LYc�A�7*

loss��<�Tt�       �	��LYc�A�7*

loss��<��j       �	Ψ�LYc�A�7*

loss�a<O{�       �	N�LYc�A�7*

loss��<�,h       �	���LYc�A�7*

lossqi,<e@       �	�`�LYc�A�7*

loss�'�<��       �	"��LYc�A�7*

loss�	�<$�c       �	���LYc�A�7*

loss|�<��$       �	>z�LYc�A�7*

loss�V<[\�       �	�LYc�A�8*

loss&�;�(�       �	��LYc�A�8*

lossJ�I;Ǿ�       �	���LYc�A�8*

loss�q<bjK       �	'M�LYc�A�8*

loss��(;�U��       �	I��LYc�A�8*

lossw��;A��       �	��LYc�A�8*

loss��;����       �	I0�LYc�A�8*

loss�D;{5�A       �	��LYc�A�8*

loss��*=[}h       �	�r MYc�A�8*

loss�'<Y�h�       �	<MYc�A�8*

loss��=z5�       �	֩MYc�A�8*

loss�/=V~P       �	lCMYc�A�8*

loss��<
�#       �	o�MYc�A�8*

lossa�;�)�       �	�tMYc�A�8*

loss��=�GD�       �	IMYc�A�8*

loss�#l<���       �	6�MYc�A�8*

losseO<FO�,       �	�fMYc�A�8*

loss��;��z%       �	�MYc�A�8*

loss�t�;�nJe       �	'�MYc�A�8*

loss�i<̊�       �	�<MYc�A�8*

loss���<�SC       �	T�MYc�A�8*

loss�K�<M$X�       �	{MYc�A�8*

loss,խ;�l�4       �	�	MYc�A�8*

lossFu
=Ǔ       �	��	MYc�A�8*

loss%͌<W>��       �	5C
MYc�A�8*

lossJ�<S��       �	�
MYc�A�8*

loss�s�<�rmM       �	TrMYc�A�8*

lossD�<�?g�       �	t�MYc�A�8*

loss�;3=NָE       �	�NMYc�A�8*

loss<1���       �	q�MYc�A�8*

loss�(0<w�5�       �	v�MYc�A�8*

loss)��:d�Ѩ       �	�MYc�A�8*

losssT];k-�       �	�4MYc�A�8*

loss�L<�+�