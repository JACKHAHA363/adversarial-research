       �K"	  �Yc�Abrain.Event:2��M �     �@��	|�Yc�A"��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2��*
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
:���������@*
seed2�
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
seed2ţ
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
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
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
seed2���
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
seed2���*
T0*
seed���)*
dtype0
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2ݙ_
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
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��*
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
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
/softmax_cross_entropy_loss_1/num_present/SelectSelect.softmax_cross_entropy_loss_1/num_present/Equal3softmax_cross_entropy_loss_1/num_present/zeros_like2softmax_cross_entropy_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
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
 softmax_cross_entropy_loss_1/divRealDiv"softmax_cross_entropy_loss_1/Sum_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
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
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select*
_output_shapes
: 
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
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
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
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
out_type0*
_output_shapes
:*
T0
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
Egradients/softmax_cross_entropy_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape7gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
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
_output_shapes
:	�
*
dtype0
�
dense_2/kernel/Adam
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
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "`Ьh�,     ?/��	��Yc�AJ��
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
seed2��
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
valueB"      *
dtype0*
_output_shapes
:
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
seed2�*
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
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2ţ*
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
seed2���
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
:���������@*
seed2ݙ_*
T0*
seed���)*
dtype0
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
seed2��
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
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
valueB *
dtype0*
_output_shapes
: 
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
8gradients/softmax_cross_entropy_loss_1/value_grad/SelectSelect$softmax_cross_entropy_loss_1/Greatergradients/Fill<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_like*
_output_shapes
: *
T0
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
T0*
out_type0*
_output_shapes
:
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
gradients/div_2_grad/Reshape_1Reshapegradients/div_2_grad/Sum_1gradients/div_2_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
out_type0*
_output_shapes
:
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
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
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
valueB@@*    *
dtype0*&
_output_shapes
:@@
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
_output_shapes
:	�
*
dtype0
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
 Bloss*
dtype0*
_output_shapes
: 
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0����       ��-	_��Yc�A*

lossj@=k�       ��-	�5�Yc�A*

loss
A@"0�j       ��-	r��Yc�A*

loss�j@�x��       ��-	�x�Yc�A*

loss�M@_"i       ��-	��Yc�A*

loss�c@r0�c       ��-	���Yc�A*

loss��@���       ��-	tF�Yc�A*

loss��?���       ��-	���Yc�A*

loss)��?��M�       ��-	��Yc�A	*

loss%o�?8�l       ��-	�C�Yc�A
*

loss}��?z��       ��-	2 �Yc�A*

loss��?n]�       ��-	���Yc�A*

loss���?	Y.       ��-	;6�Yc�A*

loss-�?'o��       ��-	L��Yc�A*

loss� ?b�6�       ��-	Su�Yc�A*

loss��?��L       ��-	��Yc�A*

losssl?�_A�       ��-	ظ�Yc�A*

loss�?p�;c       ��-	Q�Yc�A*

loss7vs?�@�&       ��-	���Yc�A*

loss��?sT�B       ��-	3��Yc�A*

loss�kq?+d��       ��-	�*�Yc�A*

loss��?$�4       ��-	~��Yc�A*

loss[�R?Q�       ��-	Zb�Yc�A*

losst6�?��\T       ��-	���Yc�A*

loss�?[?��c!       ��-	���Yc�A*

loss�dZ?x׬
       ��-	�:�Yc�A*

loss�P?�zp�       ��-	���Yc�A*

loss{�|?���       ��-	�n�Yc�A*

loss��?V�       ��-	��Yc�A*

loss��g?Ҩ�F       ��-	-��Yc�A*

lossE-M?���       ��-	�H�Yc�A*

loss��"?R@�       ��-	���Yc�A *

loss;s.?���{       ��-	�k�Yc�A!*

loss�9w?�!�=       ��-	*�Yc�A"*

loss��5?Db��       ��-	��Yc�A#*

loss��_?�1%7       ��-	�1�Yc�A$*

lossl;?'�g3       ��-	���Yc�A%*

loss{]l?)�p�       ��-	�*�Yc�A&*

loss,��?)�n�       ��-	a��Yc�A'*

loss�j?Q��       ��-	p`�Yc�A(*

loss:!?��=�       ��-	�8�Yc�A)*

loss 	?�/�       ��-	���Yc�A**

lossM�1?���       ��-	M��Yc�A+*

loss`�+?��U1       ��-	�|�Yc�A,*

loss�b?�fC       ��-	��Yc�A-*

loss�V4?��|�       ��-	��Yc�A.*

loss��?b��$       ��-	9E Yc�A/*

lossL	?�njP       ��-	�� Yc�A0*

loss��?)H�v       ��-	tYc�A1*

loss��>��fW       ��-	t	Yc�A2*

loss��>[7       ��-	|�Yc�A3*

loss�?��5_       ��-	�3Yc�A4*

lossE�?�H�       ��-	��Yc�A5*

lossv�O?�r��       ��-	�_Yc�A6*

loss�U�>*a!�       ��-	8�Yc�A7*

loss&��>��H       ��-	��Yc�A8*

loss���>0�c       ��-	-$Yc�A9*

loss���>�       ��-	��Yc�A:*

loss1%E?L�1n       ��-	�[Yc�A;*

loss�6�>��       ��-	��Yc�A<*

loss��>��m       ��-	��Yc�A=*

loss=E�>�p�f       ��-	�"	Yc�A>*

loss�c�>:���       ��-	��	Yc�A?*

loss�� ?߲��       ��-	�^
Yc�A@*

loss3]�>���       ��-	��
Yc�AA*

loss���>�g;       ��-	�Yc�AB*

loss
8?Q'R�       ��-	OYc�AC*

loss|�?
@Sw       ��-	x�Yc�AD*

loss}�,?`���       ��-	�RYc�AE*

loss��?8"�       ��-	V�Yc�AF*

loss�5�>�R       ��-	z�Yc�AG*

loss��>��       ��-	�"Yc�AH*

loss���>o��;       ��-	M�Yc�AI*

loss�>J\9        ��-	]�Yc�AJ*

loss��??v�       ��-	�*Yc�AK*

loss�?�r��       ��-	<�Yc�AL*

loss��-?��       ��-	O\Yc�AM*

lossT�,?����       ��-	 Yc�AN*

loss?�?�>ٰ       ��-	A�Yc�AO*

loss@��>�
��       ��-	Y2Yc�AP*

losshi?7�^'       ��-	$Yc�AQ*

loss;�??���       ��-	q�Yc�AR*

losshF#?1���       ��-	G=Yc�AS*

lossq_�> n_�       ��-	�(Yc�AT*

loss�> H��       ��-	��Yc�AU*

loss��>�a$�       ��-	�TYc�AV*

loss�ɿ>       ��-	\�Yc�AW*

loss�?o��       ��-	��Yc�AX*

loss�?�3�?       ��-	Yc�AY*

lossO��>r�5f       ��-	�Yc�AZ*

loss��?�@m       ��-	�OYc�A[*

loss��? [�       ��-	d�Yc�A\*

loss�w?w���       ��-	|Yc�A]*

loss�w?IZ��       ��-	�Yc�A^*

lossaj!?�-�6       ��-	7�Yc�A_*

loss\��>�զ       ��-	�>Yc�A`*

lossr��>&6[�       ��-	��Yc�Aa*

loss���>��e�       ��-	%!Yc�Ab*

lossC"O?��d�       ��-	5�!Yc�Ac*

loss�(�>�#��       ��-	�"Yc�Ad*

loss1[�>3`s7       ��-	y#Yc�Ae*

loss7��>����       ��-	�7$Yc�Af*

lossn8?{���       ��-	��$Yc�Ag*

loss
��>�.�       ��-	�v%Yc�Ah*

loss�V�>x\��       ��-	��&Yc�Ai*

loss.��>=�y?       ��-	Ɖ'Yc�Aj*

loss|*�>x'�B       ��-	;5(Yc�Ak*

loss<��>�R�D       ��-	jL)Yc�Al*

loss�I?V��}       ��-	�)Yc�Am*

lossү�>�f=h       ��-	@�*Yc�An*

loss7�?2�D�       ��-	�`+Yc�Ao*

loss!��>�|�&       ��-	M,Yc�Ap*

lossN�>p��       ��-	��,Yc�Aq*

lossAl>��+       ��-	�.Yc�Ar*

loss�>I       ��-	�.Yc�As*

loss���>���       ��-	�/Yc�At*

loss���>"��       ��-	��0Yc�Au*

lossOu�>���q       ��-	
�1Yc�Av*

loss���>��Y       ��-	�I2Yc�Aw*

loss��>�.�N       ��-	��2Yc�Ax*

loss8��>�ac�       ��-	�3Yc�Ay*

loss�U?)�       ��-	�%4Yc�Az*

lossV�>E�W       ��-	:�4Yc�A{*

loss�ܘ>�P�       ��-	�5Yc�A|*

loss�$/>ᱯ       ��-	�6Yc�A}*

loss-n�>P�M       ��-	�G7Yc�A~*

loss	?_r�       ��-	��7Yc�A*

loss.<�>�{o�       �	i�8Yc�A�*

lossju�>���0       �	X99Yc�A�*

lossQN�>+J�       �	U�9Yc�A�*

loss��z>K���       �	�|:Yc�A�*

losse\�>���       �	�;Yc�A�*

lossS�=>G��       �	�V=Yc�A�*

lossN=>}��       �	_>Yc�A�*

lossf�>�n:       �	Ҭ>Yc�A�*

loss���>�̮�       �	�AAYc�A�*

lossr�>Nl˴       �	��AYc�A�*

loss��R>�4/       �	z�BYc�A�*

lossf�n>x��       �	OWCYc�A�*

lossLC>=Ĝ�       �	.DYc�A�*

loss|�>&��       �	)�DYc�A�*

loss�Z�>��
+       �	2ZEYc�A�*

loss��>����       �	)FYc�A�*

lossW��>��<�       �	��FYc�A�*

loss���>�}xd       �	.TGYc�A�*

loss�� >3eA3       �	�GYc�A�*

losshތ>���       �	��HYc�A�*

loss�~>��S3       �	�LIYc�A�*

loss=�*>m[�       �	�<JYc�A�*

loss��i>�E�3       �	��JYc�A�*

lossS��>�4       �	��KYc�A�*

loss܏�>[�;�       �	�6LYc�A�*

loss��>�L�       �	��LYc�A�*

loss�[>���       �	�MYc�A�*

lossd�6>����       �	�NYc�A�*

loss���=�d8�       �	>�NYc�A�*

lossa�>gA|W       �	�MOYc�A�*

lossSė>*�dU       �	x�OYc�A�*

losss��>~Q��       �	K�PYc�A�*

lossoՐ>\U��       �	k(QYc�A�*

loss8��>ߘ�d       �	��QYc�A�*

loss�z�>��^k       �	F{RYc�A�*

loss�*�>=���       �	�SYc�A�*

loss}T>�k�       �	F�SYc�A�*

lossm!�>���       �	�!UYc�A�*

lossq�=>��W+       �	��UYc�A�*

loss�x>�)�       �	�oVYc�A�*

loss{>��R�       �	�WYc�A�*

loss|��>7��`       �	��WYc�A�*

loss�[�>��       �	�\XYc�A�*

loss/�{>\��*       �	�XYc�A�*

lossś,>�_�       �	�YYc�A�*

lossȧ>L�m�       �	NCZYc�A�*

lossEo�>pqtA       �	��ZYc�A�*

loss
l">�e�N       �	-z[Yc�A�*

loss!�>�l��       �	\Yc�A�*

lossA?��H       �	��\Yc�A�*

loss��>t!��       �	lC]Yc�A�*

loss�>ؤø       �	��]Yc�A�*

loss�B�=�2e       �	�n^Yc�A�*

loss���=6�v_       �	K_Yc�A�*

lossCȧ>�7b�       �	?�_Yc�A�*

loss��>����       �	�F`Yc�A�*

loss�O>,N��       �	�`Yc�A�*

lossi�>i6|u       �	��aYc�A�*

lossv�f>S��       �	B�bYc�A�*

loss�|>�I&f       �	�ZcYc�A�*

loss\��>3�Rw       �	�dYc�A�*

loss�jw>B�       �	�dYc�A�*

loss�#R>�1sK       �	1�eYc�A�*

lossQX�>³�T       �	�7fYc�A�*

loss�l><���       �	{�fYc�A�*

loss�@g>����       �	��gYc�A�*

loss��>�]'�       �	�vhYc�A�*

loss#j|>��R       �	�iYc�A�*

loss�>���b       �	Z�iYc�A�*

lossl��>��Ġ       �	�fjYc�A�*

loss�r�>��׳       �	�kYc�A�*

loss$��>��x       �	��kYc�A�*

lossC�@>��       �	�DlYc�A�*

lossDp�>�k�       �	�lYc�A�*

loss�%>��       �	I�mYc�A�*

lossR>�WH�       �	�nYc�A�*

loss7̊>J�\t       �	ظnYc�A�*

lossh52>��N7       �	>�oYc�A�*

loss�!>�ؑ�       �	�KpYc�A�*

loss��6>��sr       �	��qYc�A�*

lossni>F"�       �	�urYc�A�*

loss|�A>/M�       �	�#sYc�A�*

loss2ƃ>�TZ       �	��sYc�A�*

lossqJ\>�G��       �	�ftYc�A�*

loss�{�>mF��       �	� uYc�A�*

loss�"(>�ޛb       �	L�uYc�A�*

loss���=���       �	2<vYc�A�*

lossa5�>4��       �	5�vYc�A�*

loss���>��,       �	�jwYc�A�*

loss�@>~��       �	*xYc�A�*

lossA��>O�zE       �	��xYc�A�*

lossߑ�>$�       �	�ZyYc�A�*

loss��>J�E       �	MzYc�A�*

loss	��=��K       �	y�zYc�A�*

loss�,>�k�v       �	�{Yc�A�*

lossO��>���o       �	,d|Yc�A�*

loss C�>驘f       �	�}Yc�A�*

loss`[.>D !>       �	�~Yc�A�*

loss���=��k�       �	�~Yc�A�*

lossJht>��h'       �	�jYc�A�*

loss{lh>�(       �	��Yc�A�*

lossrc>θ$�       �	��Yc�A�*

loss��\>�i�       �	3��Yc�A�*

loss�)>���E       �	�P�Yc�A�*

loss>� �       �	#�Yc�A�*

loss��>�8�A       �	���Yc�A�*

loss�A>�s��       �	�4�Yc�A�*

loss8n�>nC��       �	�مYc�A�*

loss��+>��~�       �	���Yc�A�*

loss�`/>l�%.       �	?�Yc�A�*

loss�-�>f�\6       �	�ۇYc�A�*

loss&?�>˸q�       �	^��Yc�A�*

loss&i�=x��       �	a7�Yc�A�*

lossJ*->F��       �	�ىYc�A�*

loss��f>�b8�       �	Z�Yc�A�*

loss=�t>��|       �	S$�Yc�A�*

loss��d>O�5-       �	���Yc�A�*

loss�\>s��W       �	�_�Yc�A�*

loss�z]>RF�       �		��Yc�A�*

loss}�">����       �	���Yc�A�*

loss�9�=_�+�       �		7�Yc�A�*

loss��[>��C       �	�֎Yc�A�*

loss��>h?0h       �	?q�Yc�A�*

loss�>���       �	��Yc�A�*

loss�O�>N$A�       �	@��Yc�A�*

loss�h>8K��       �	H�Yc�A�*

loss�R�=8��       �	��Yc�A�*

lossMm)>xO�4       �	Y��Yc�A�*

lossi�d>8�5       �	�B�Yc�A�*

loss��,>(�       �	c�Yc�A�*

lossO"�>���-       �	p��Yc�A�*

loss�I1>���@       �	C:�Yc�A�*

loss�@> K�       �	��Yc�A�*

loss%�o>�a�       �	r��Yc�A�*

loss�E>Z�=       �	U1�Yc�A�*

lossҽ�>ӑA       �	o֗Yc�A�*

loss�q>�=��       �	�|�Yc�A�*

loss$��>��?�       �	O#�Yc�A�*

loss}r>�"�       �	��Yc�A�*

loss��5>�(�e       �	�d�Yc�A�*

loss;�z>ڇש       �	��Yc�A�*

loss��>��̺       �	+��Yc�A�*

lossJK0>ř3r       �	=G�Yc�A�*

lossO�>ȑ�       �	T�Yc�A�*

loss���>r���       �	�~�Yc�A�*

lossR��>ަR1       �	L�Yc�A�*

loss��>�n��       �	c��Yc�A�*

loss��l>���n       �	�Z�Yc�A�*

lossf�>vr�>       �	4�Yc�A�*

loss�)G>$�;�       �	&��Yc�A�*

loss��>aU�v       �	ҡYc�A�*

loss��v>��o3       �	��Yc�A�*

loss��:>l> 3       �	4��Yc�A�*

lossE��=9..t       �	�a�Yc�A�*

loss�$/>���       �	H��Yc�A�*

losst�>wUǝ       �	���Yc�A�*

lossn>N��[       �	J^�Yc�A�*

loss!�T>�BC�       �	]��Yc�A�*

loss%>^��       �	ᛧYc�A�*

lossj?>@��       �	�P�Yc�A�*

loss	��=�6:�       �	�X�Yc�A�*

loss���>�R.Z       �	�
�Yc�A�*

loss�׆=�K       �	u��Yc�A�*

loss���=kIZ       �	K�Yc�A�*

loss�.�>�yk�       �	��Yc�A�*

loss�:�=�[�       �	��Yc�A�*

loss�m�=�9�L       �	�S�Yc�A�*

loss��h>�i�       �	�	�Yc�A�*

loss>���       �	���Yc�A�*

loss�&[>Q��       �	}@�Yc�A�*

loss86d>�Rq       �	��Yc�A�*

loss=ގ>O�       �	V��Yc�A�*

loss��>>�_�       �	�#�Yc�A�*

lossߥ=>�1��       �	���Yc�A�*

lossA6>�_`�       �	�c�Yc�A�*

loss�;/>T?�       �	�	�Yc�A�*

loss��v>삨�       �	��Yc�A�*

loss�>�I       �	�R�Yc�A�*

loss*1>&��       �	2u�Yc�A�*

loss�hE=�s��       �	n�Yc�A�*

loss$ҕ>{�2       �	'��Yc�A�*

loss��>���       �	Y�Yc�A�*

loss�� >��7       �	� �Yc�A�*

loss��-> `��       �	���Yc�A�*

loss4�Z>`?2        �	77�Yc�A�*

loss8�>���       �	��Yc�A�*

loss�y;>{F	�       �	{��Yc�A�*

loss��=g��       �	�[�Yc�A�*

loss��i>�fp�       �	Y��Yc�A�*

lossA>���       �	���Yc�A�*

loss6A�=�9M�       �	�6�Yc�A�*

loss�Is>�)�       �	MؾYc�A�*

lossj�*>
�i�       �	v��Yc�A�*

lossΪ�=ry�       �	�.�Yc�A�*

loss���>e�       �	���Yc�A�*

loss���>��EJ       �	���Yc�A�*

loss�z8>��       �	)�Yc�A�*

lossA(�>���S       �	��Yc�A�*

loss�a�>�q�       �	ׅ�Yc�A�*

loss�	�>F�F       �	g+�Yc�A�*

loss�>B�2C       �	���Yc�A�*

lossq�>��p       �	z�Yc�A�*

loss��r>7�       �	��Yc�A�*

loss�1E>�D�2       �	��Yc�A�*

loss���=���       �	fg�Yc�A�*

loss���=��\       �	��Yc�A�*

loss�P>�֒\       �	���Yc�A�*

loss(��>�]�D       �	f�Yc�A�*

lossQ�>�#       �	��Yc�A�*

lossj܌>���       �	���Yc�A�*

loss7zD>�?�       �	�[�Yc�A�*

loss6C5>G��       �	��Yc�A�*

loss ��=t�[       �	��Yc�A�*

loss��>�vyk       �	eP�Yc�A�*

loss,�=����       �	���Yc�A�*

loss�>R��       �	��Yc�A�*

loss��L>��A       �		8�Yc�A�*

loss��>ɯ�       �	���Yc�A�*

lossD�J>��       �	;s�Yc�A�*

losso6�=yI�       �	R�Yc�A�*

loss�Xz>A�       �	e��Yc�A�*

loss}�f>`�       �	�?�Yc�A�*

loss�:>	�~j       �	���Yc�A�*

loss�K�=J�       �	r�Yc�A�*

loss�|�=^�fz       �	��Yc�A�*

loss���=�߮>       �	}y�Yc�A�*

lossq�,>��T>       �	}�Yc�A�*

loss��[>E0,       �	���Yc�A�*

lossT>�dX�       �	�~�Yc�A�*

loss?�>��|�       �	X�Yc�A�*

loss�5>s�|       �	t��Yc�A�*

lossŦ2>�Y�       �	�O�Yc�A�*

loss���=�$=	       �	K��Yc�A�*

loss[j>����       �	���Yc�A�*

loss��>;DV;       �	'.�Yc�A�*

loss�P>8q*m       �	p��Yc�A�*

loss�^�>�{_�       �	�l�Yc�A�*

loss��z>�@�       �	��Yc�A�*

loss��$>��h       �	��Yc�A�*

loss��t>I�       �	�:�Yc�A�*

loss��2>w)z�       �	f��Yc�A�*

loss�l>���/       �	�w�Yc�A�*

lossq�=1�E       �	��Yc�A�*

loss��>�dG       �	���Yc�A�*

loss�z$>��/g       �	Vd�Yc�A�*

loss�:>wԔ       �	˝�Yc�A�*

loss
!P>�ܒ�       �	�U�Yc�A�*

loss��A>���9       �	��Yc�A�*

lossc�>͡�       �	Eh�Yc�A�*

losse}>�s/       �	��Yc�A�*

lossж">���@       �	D��Yc�A�*

loss!>�Mg�       �	�_�Yc�A�*

loss;�J>���       �	��Yc�A�*

lossqu>4.       �	H��Yc�A�*

loss�<�>o,��       �	�K�Yc�A�*

loss$>����       �	L��Yc�A�*

loss��!>���o       �	�}�Yc�A�*

loss�^�=c��       �	6�Yc�A�*

loss!��=�Z�       �	<��Yc�A�*

lossn�S>�l�       �	���Yc�A�*

lossV�>���O       �	v8�Yc�A�*

lossz}>�uY       �	���Yc�A�*

lossL>�&       �	���Yc�A�*

loss��x>S�wF       �	�9�Yc�A�*

loss��=<��       �	���Yc�A�*

loss��D>J�+       �	�r�Yc�A�*

lossf�8>�RrF       �	{�Yc�A�*

loss��X>i�       �	��Yc�A�*

loss���>]�W       �	��Yc�A�*

loss�f�=l	�       �	��Yc�A�*

lossW�=W��       �	�2�Yc�A�*

lossl�>�{��       �	���Yc�A�*

loss=Zx>WY�       �	oe�Yc�A�*

lossH
�=�F��       �	/�Yc�A�*

loss֕ >���       �	���Yc�A�*

loss
&>��       �	��Yc�A�*

loss�!�= ��&       �	���Yc�A�*

loss�~0=��>�       �	$��Yc�A�*

loss�N#>��A*       �	)��Yc�A�*

loss��K>Ƀ��       �	!:�Yc�A�*

loss�>��       �	���Yc�A�*

loss��(>	n�       �	�u�Yc�A�*

loss��u>�5�       �	j Yc�A�*

loss�W>���       �	>� Yc�A�*

loss
�k>�qq       �	LYc�A�*

loss?C�=a"�       �	��Yc�A�*

lossѽ>���       �	��Yc�A�*

loss!�t>�^N       �	s*Yc�A�*

loss�Fu>��
[       �	�Yc�A�*

loss�s�=%�F�       �	�eYc�A�*

lossiC>��ɓ       �	TYc�A�*

loss��<>��e       �	�Yc�A�*

lossxh/>u��       �	,JYc�A�*

lossϊ�=�B;�       �	��Yc�A�*

loss>ŭ Y       �	Y�Yc�A�*

lossto>�Xͳ       �	x&Yc�A�*

lossT(�>8�5       �	��Yc�A�*

loss�zB>��Ț       �	�c	Yc�A�*

loss�Ğ>���       �	�
Yc�A�*

loss0>pTjz       �	��
Yc�A�*

loss=�A>6�:�       �	DYc�A�*

loss8
�=�6��       �	G�Yc�A�*

loss��>w�k       �	��Yc�A�*

loss־�>~�#0       �	�,Yc�A�*

loss	�%>��       �	-�Yc�A�*

loss���=�q       �	~rYc�A�*

loss7��=z�9	       �	�Yc�A�*

loss��K>�1       �	��Yc�A�*

lossq��=�Iss       �	�WYc�A�*

loss�S�=�,¿       �	3�Yc�A�*

loss=��=@`}        �	��Yc�A�*

lossn�=d?�
       �	�7Yc�A�*

loss4d�=f%�       �	E�Yc�A�*

loss�͉>_N};       �	PqYc�A�*

lossx�V>O��       �	Yc�A�*

lossG�>�B��       �	��Yc�A�*

loss߀ >����       �	eUYc�A�*

loss�z=j-��       �	��Yc�A�*

loss�j|>rUٙ       �	�Yc�A�*

lossee�>�Z|�       �	78Yc�A�*

loss��=����       �	Y�Yc�A�*

loss��>�;
       �	){Yc�A�*

lossPs�>���       �	DYc�A�*

loss�J*>X��       �	��Yc�A�*

loss�d=�:�X       �	3QYc�A�*

loss��@=���x       �	A�Yc�A�*

loss��O>�V��       �	�Yc�A�*

loss��>%E�       �	(Yc�A�*

lossN x>�ϭ       �	u�Yc�A�*

loss��> ��a       �	�bYc�A�*

lossT�+>�18       �	Yc�A�*

loss��7>JC_3       �	�Yc�A�*

loss�>��o3       �	j3Yc�A�*

lossT:�=~!�u       �	 �Yc�A�*

loss���=Z�T       �	Pn Yc�A�*

loss,�'> ���       �	Z!Yc�A�*

loss8/�=Xê       �	ɯ!Yc�A�*

lossﺶ=k!�       �	&Q"Yc�A�*

lossF��=d e       �	(�"Yc�A�*

loss*�>��c       �	�#Yc�A�*

losssE�=Kԡ�       �	!�$Yc�A�*

lossHc=>!�.�       �	�|%Yc�A�*

loss�}�=���       �	�,&Yc�A�*

loss7�\>t9�t       �	��&Yc�A�*

loss\��=f �       �	}z'Yc�A�*

loss3��>-�v�       �	�U(Yc�A�*

lossx�q>S��       �	q�(Yc�A�*

loss�I>�lZh       �	v�)Yc�A�*

loss�\�=�|�       �	�?*Yc�A�*

loss�=,�       �	��*Yc�A�*

lossa�=>U�       �		�+Yc�A�*

loss&>=>%       �	Q0,Yc�A�*

loss%y8>�k�       �	Y�,Yc�A�*

loss��>��z*       �	n�-Yc�A�*

loss ��=_�B�       �	H.Yc�A�*

loss��=F�L�       �	��.Yc�A�*

loss���=����       �	9~/Yc�A�*

loss���=�MU       �	�0Yc�A�*

lossR�>�/       �	"�0Yc�A�*

loss�e�>����       �	�h1Yc�A�*

loss�_>��,�       �	t
2Yc�A�*

loss��>�8�=       �	J�2Yc�A�*

loss�xL>��       �	�W3Yc�A�*

loss2�L>�y�>       �	�3Yc�A�*

loss, >�,�       �	Y�4Yc�A�*

loss���=�+1m       �	�M5Yc�A�*

lossᮥ={�'�       �	R�5Yc�A�*

loss���=��t�       �	��6Yc�A�*

loss!*K=*�ǡ       �	�(7Yc�A�*

lossM*>����       �	��7Yc�A�*

loss��=��	�       �	�d8Yc�A�*

loss���=�p��       �	l	9Yc�A�*

loss��K>���P       �	ͱ9Yc�A�*

loss�>O�t>       �	�U:Yc�A�*

loss�s�=i��       �	��:Yc�A�*

loss��H=�1�j       �	J�;Yc�A�*

lossI�=vi��       �	^.<Yc�A�*

loss�OG>oh��       �	4�<Yc�A�*

loss��=%�       �	0�=Yc�A�*

loss�|�=��.�       �	SZ>Yc�A�*

lossE	�={�[�       �	�?Yc�A�*

loss�V_>:y��       �	��?Yc�A�*

loss�>@O��       �	-|@Yc�A�*

loss��=�r�       �	�%AYc�A�*

lossA~�=y�6G       �	[�AYc�A�*

lossLc�=���A       �	:vBYc�A�*

loss	�v>T�|       �	�!CYc�A�*

losst
>XcH�       �	~�CYc�A�*

loss��l>]���       �	VDYc�A�*

loss��}=�'(|       �	j.EYc�A�*

loss���=���       �	V�EYc�A�*

lossW��=q+�B       �	�sFYc�A�*

loss
=�=ȍ�A       �	LGYc�A�*

lossd�=���{       �	�GYc�A�*

loss��>��&�       �	aHYc�A�*

lossvAH=�N�       �	"�HYc�A�*

loss���=K�N4       �	�IYc�A�*

loss���=,�ּ       �	RDJYc�A�*

loss+�=%c��       �	��JYc�A�*

loss�b�=S�       �	n�KYc�A�*

loss/�*>nʗ�       �	>%LYc�A�*

loss�.>���       �	��LYc�A�*

loss�}�=�LM�       �	p\MYc�A�*

loss��=�pU�       �	2uNYc�A�*

loss��m>Q�#�       �	� OYc�A�*

lossܾ=�$g�       �	��OYc�A�*

loss�\=Æ�       �	��PYc�A�*

loss��<�H��       �	+QYc�A�*

loss�1�=M@�       �	��QYc�A�*

loss�|_=��       �	U�RYc�A�*

loss�F=��ߣ       �	TYc�A�*

loss?�<��"�       �	�UYc�A�*

losscE>z�g       �	�UYc�A�*

loss��/= ���       �	�UVYc�A�*

loss.T<ᴲ�       �	��VYc�A�*

loss�<�*0       �	؛WYc�A�*

loss,�>��i       �	JXYc�A�*

lossJ�M>��C       �	��XYc�A�*

loss���=����       �	�YYc�A�*

loss�2�<x�w2       �	�TZYc�A�*

loss��= �~\       �	�ZYc�A�*

loss���>9�"       �	`�[Yc�A�*

loss��%=�͂J       �	�5\Yc�A�*

loss�u{=��C       �	��\Yc�A�*

loss�A>�M5       �	�y]Yc�A�*

lossӸG>�>       �	"^Yc�A�*

loss��>�	a$       �	��^Yc�A�*

lossO~n=�:];       �	�t_Yc�A�*

loss�,�=�M��       �	�`Yc�A�*

loss{J�=��T       �	�`Yc�A�*

loss>ң�/       �	9`aYc�A�*

lossŪ7>((       �	U�aYc�A�*

loss-Hs>��p       �	��bYc�A�*

loss���>�       �	�/cYc�A�*

losss76>{�FE       �	��cYc�A�*

lossR��=]*?�       �	4fdYc�A�*

loss;��>����       �	��dYc�A�*

loss�)�>K��_       �	w�eYc�A�*

lossQ�|=ĨA       �	�>fYc�A�*

lossH�>��	�       �	��fYc�A�*

lossd�>6r�@       �	�zgYc�A�*

loss�	�=3���       �	shYc�A�*

loss�u�=E'p       �	�hYc�A�*

loss���=�E?5       �	�@iYc�A�*

loss��9>L{k}       �	�iYc�A�*

loss���<y���       �	�tjYc�A�*

loss���=j��       �	�kYc�A�*

loss�,=�r�M       �	ƣkYc�A�*

loss_��=_~ۮ       �	�;lYc�A�*

loss�k�=��+       �	�lYc�A�*

loss�U>}�>�       �	semYc�A�*

lossA��=,r��       �	" nYc�A�*

loss��~=&��       �	��nYc�A�*

loss�8�=���N       �	�FoYc�A�*

loss�>��%       �	<�oYc�A�*

loss�2=$xS�       �	vpYc�A�*

losss>�'z       �	�qYc�A�*

loss�:�=����       �	��qYc�A�*

lossC�J=���       �	�CrYc�A�*

loss!��=���       �	U�rYc�A�*

loss�d>=�-�       �	hvsYc�A�*

loss�^O>�;F6       �	�tYc�A�*

loss�7�=���       �	�tYc�A�*

loss��=_��*       �	�<uYc�A�*

loss}}�=�hU        �	��uYc�A�*

loss�> >���       �	'lvYc�A�*

loss*�=�_R       �	�wYc�A�*

loss���=� ��       �	�wYc�A�*

loss���=�P��       �	��xYc�A�*

loss*W3=� V�       �	�myYc�A�*

loss��>.H��       �	2|Yc�A�*

loss���=���K       �	��|Yc�A�*

loss*�=����       �	�d}Yc�A�*

loss��7>6/��       �	�ԗYc�A�*

loss)�'>}���       �	u�Yc�A�*

lossr�(>�;C�       �	t�Yc�A�*

loss��D>LX�       �	P��Yc�A�*

loss� >���        �	�A�Yc�A�*

loss�A=E8       �	�ۚYc�A�*

lossmt>��7�       �	,�Yc�A�*

loss\��=- �       �	a�Yc�A�*

loss%!>�SK�       �	���Yc�A�*

loss��0>��       �	�I�Yc�A�*

lossV�= �       �	z�Yc�A�*

lossMe�='�z       �	��Yc�A�*

loss��+>Z�
       �	���Yc�A�*

loss�h>0��       �	UQ�Yc�A�*

loss8>f=}�       �	��Yc�A�*

loss�5>eY�       �	���Yc�A�*

loss,Ð=��       �	�}�Yc�A�*

lossVP=w4�'       �	oE�Yc�A�*

loss�=���A       �	�ܣYc�A�*

loss�y�>,K�       �	䂤Yc�A�*

loss�L<>�!��       �	 �Yc�A�*

loss�>��V8       �	�ťYc�A�*

loss���=m���       �	�q�Yc�A�*

loss�g>��*       �	��Yc�A�*

lossT@%>�A�       �	˽�Yc�A�*

lossWۄ=�Ӌ�       �	9a�Yc�A�*

lossaas=�Ҍ�       �	X�Yc�A�*

loss��=~��       �	���Yc�A�*

loss �>���       �	�H�Yc�A�*

lossc;!>ZIg�       �	G�Yc�A�*

loss7��=��B�       �	ѕ�Yc�A�*

loss�=G�\       �	7�Yc�A�*

lossf��=/*�       �	/ݬYc�A�*

loss3�>�%       �	�u�Yc�A�*

loss�l=��i�       �	��Yc�A�*

loss�x6>i<z�       �	�Yc�A�*

loss�d�<QI#�       �	�V�Yc�A�*

loss$& >�M�O       �	b��Yc�A�*

loss�c�>���       �	?��Yc�A�*

loss�J>(���       �	�:�Yc�A�*

loss�(>g9�       �	ױYc�A�*

loss�W�=�w�       �	~n�Yc�A�*

lossz��=�R\�       �	�Yc�A�*

loss|�J>1�[�       �	��Yc�A�*

loss��>��       �	�I�Yc�A�*

loss�_1>���+       �	��Yc�A�*

loss�ٺ=a�=       �	�|�Yc�A�*

loss�->�C�       �	s�Yc�A�*

loss���=��!       �	���Yc�A�*

loss��0=U%�=       �	�>�Yc�A�*

loss�@=Ba�n       �	AطYc�A�*

lossc(�=3�I       �	���Yc�A�*

loss��=�2�       �	��Yc�A�*

loss���>.^�       �	c��Yc�A�*

lossNY�=�o�       �	�Y�Yc�A�*

loss"�=� /�       �	��Yc�A�*

loss$�<�f��       �	�%�Yc�A�*

loss�=;�2       �	���Yc�A�*

loss���=�       �	�V�Yc�A�*

lossς=�)�       �	�z�Yc�A�*

loss8�>��[Q       �	�P�Yc�A�*

loss�,>���       �	�Yc�A�*

loss���=�9d       �	���Yc�A�*

loss|�=��e�       �	L�Yc�A�*

loss�`�=ĳ�       �	���Yc�A�*

loss�>;�XY       �	I�Yc�A�*

lossh>3w&8       �	;��Yc�A�*

loss �:>5,�N       �	yt�Yc�A�*

loss�N>E)       �	N�Yc�A�*

loss!n�=�J�        �	)��Yc�A�*

loss���=� �       �	�R�Yc�A�*

loss�8�=<wbd       �	 ��Yc�A�*

lossQT�=A��       �	���Yc�A�*

loss=b>81M       �	�'�Yc�A�*

loss��>�� 7       �	���Yc�A�*

lossR�r=�%_       �	�i�Yc�A�*

loss�6>�Iw�       �	��Yc�A�*

loss�۪>��V3       �	O��Yc�A�*

loss�ո=b�.�       �	�W�Yc�A�*

lossB�=��       �	��Yc�A�*

lossS�->���       �	��Yc�A�*

loss=�>�2=�       �	4�Yc�A�*

loss��>�@}       �	x��Yc�A�*

loss�[�=�۲       �	�j�Yc�A�*

lossD�>�T�       �	��Yc�A�*

loss��>���       �	h��Yc�A�*

loss(g>��       �	�L�Yc�A�*

loss���=3��       �	]��Yc�A�*

loss��=@ql>       �	�v�Yc�A�*

loss���=Z��z       �	k�Yc�A�*

loss�aO>�4       �	e��Yc�A�*

loss]e>��r       �	�H�Yc�A�*

loss E>�JA�       �	H��Yc�A�*

loss29!>�m       �	Gw�Yc�A�*

loss�x>��       �	��Yc�A�*

loss�T�=�//�       �	v��Yc�A�*

loss�5D>"Z�#       �	�;�Yc�A�*

lossJ��=X��J       �	���Yc�A�*

loss�M>���       �	fh�Yc�A�*

lossj>t�ȕ       �	���Yc�A�*

loss�Ý=J.��       �	���Yc�A�*

lossmǀ=�1c�       �	->�Yc�A�*

lossi�>q���       �	N�Yc�A�*

loss���=��(       �	>��Yc�A�*

loss��>fe�       �	L��Yc�A�*

loss1�=���       �	%�Yc�A�*

loss�=��?       �	���Yc�A�*

loss��i=�ğ       �	�[�Yc�A�*

loss�e>�Td�       �	���Yc�A�*

loss�$>AD�       �	� �Yc�A�*

losso�=��p�       �	j��Yc�A�*

loss�)�=Z��       �	@�Yc�A�*

lossx�=��I�       �	3��Yc�A�*

loss�<�=�CI�       �	���Yc�A�*

loss}X^=:n�       �	"�Yc�A�*

loss��=�J�9       �	���Yc�A�*

lossE1�=ƪ�       �	�a�Yc�A�*

loss���=URR�       �	�g�Yc�A�*

loss\�2>�w3       �	��Yc�A�*

lossl1�=J��       �	o~�Yc�A�*

lossR��=���F       �	��Yc�A�*

loss�!�=u��       �	��Yc�A�*

loss�_�=Z(2       �	e�Yc�A�*

loss�w�=��p       �	��Yc�A�*

loss5Ԑ=����       �	���Yc�A�*

loss�>`'�       �	�I�Yc�A�*

lossc�>Z*��       �	k�Yc�A�*

loss�m�=3z��       �	,I�Yc�A�*

loss=�9>�f<v       �	���Yc�A�*

loss� >���       �	E��Yc�A�*

loss �=�|b)       �	���Yc�A�*

loss���=�c��       �	_�Yc�A�*

loss�(�=0�
       �	��Yc�A�*

loss~�<?f�       �	��Yc�A�*

loss$c>�l�       �	���Yc�A�*

loss��=x#lw       �	�d�Yc�A�*

lossݣ�=�x��       �	c
�Yc�A�*

loss�\==/��       �	���Yc�A�*

loss0�= ���       �	TU�Yc�A�*

loss�[ =U�o       �	���Yc�A�*

lossw�}=�mu�       �	Ԙ�Yc�A�*

lossr;�=e4��       �	"��Yc�A�*

loss3p
>�       �	ZJ�Yc�A�*

loss�a@>��,?       �	���Yc�A�*

loss`�i>�suw       �	Q��Yc�A�*

loss���=�`�g       �	}y�Yc�A�*

loss���=@�r�       �	��Yc�A�*

loss�d=)`��       �	d��Yc�A�*

loss,��<��9       �	@L�Yc�A�*

loss�E�=߽�J       �	��Yc�A�*

lossZ0=�ɇ�       �	��Yc�A�*

lossj,>f�-       �	�&�Yc�A�*

lossr:T>2�/h       �	'��Yc�A�*

loss��(>�R�       �	�X Yc�A�*

loss��=W}I�       �	�� Yc�A�*

loss��U=��u%       �	$�Yc�A�*

lossڛ�=�iS       �	�KYc�A�*

lossd#>E�.�       �	��Yc�A�*

loss}�>���+       �	N�Yc�A�*

loss�x=�       �	�Yc�A�*

loss�4>ä@�       �	�@Yc�A�*

loss�==^$       �	"�Yc�A�*

loss�T�=���<       �	0�Yc�A�*

loss4=Sl�       �	�Yc�A�*

lossN8�=r~a�       �	d�Yc�A�*

loss}��= f��       �	�GYc�A�*

lossZd=<S�       �	~�Yc�A�*

loss֛>��(�       �	�|	Yc�A�*

loss�`�=�{@�       �	�"
Yc�A�*

loss���=�~XH       �	�
Yc�A�*

loss�Â=dQ�       �	\Yc�A�*

loss�~$=�{l       �	j�Yc�A�*

loss�E>X��X       �	%�Yc�A�*

loss*/e>���       �	v8Yc�A�*

loss�~F=!�o-       �	��Yc�A�*

loss���=�c0�       �	�mYc�A�*

loss�9�=)Q�m       �	)Yc�A�*

lossD�>�u�       �	��Yc�A�*

loss�Zr=')�V       �	�7Yc�A�*

loss��N=Vm�,       �	��Yc�A�*

loss؟�<�z��       �	�kYc�A�*

loss8�>_�       �	�Yc�A�*

loss�t�=����       �	�Yc�A�*

loss�6=�=�"       �	?Yc�A�*

loss�i="       �	��Yc�A�*

lossle�=K)�       �	rmYc�A�*

lossE>	�[�       �	KYc�A�*

lossֻk=���#       �	��Yc�A�*

loss\>��y       �	]4Yc�A�*

loss4�=�p:,       �	!�Yc�A�*

loss��>q        �	�gYc�A�*

loss
��=�6�       �	mYc�A�*

loss6�=���       �	9�Yc�A�*

loss��Y=Y��       �	82Yc�A�*

loss��3=�!       �	��Yc�A�*

lossn�	>�j�       �	,gYc�A�*

loss�,�=�4/        �	�Yc�A�*

loss��=���       �	ϡYc�A�*

lossc�>>o ��       �	nOYc�A�*

loss�>��_+       �	J�Yc�A�*

losss�=���       �	��Yc�A�*

loss��=hA�       �	�3Yc�A�*

loss���=�'DT       �	��Yc�A�*

loss���=�4��       �	,}Yc�A�*

loss�4j=p�%       �	 Yc�A�*

loss��5=��3�       �	�� Yc�A�*

loss��=�tKr       �	cc!Yc�A�*

loss���=�3~       �	["Yc�A�*

loss=�=B       �	��"Yc�A�*

loss���=:20�       �	_y#Yc�A�*

loss�л=�'��       �	�n$Yc�A�*

loss�f >��$       �	p$%Yc�A�*

loss=��\       �	�&Yc�A�*

lossNG=���       �	�&Yc�A�*

loss�?>bn       �	T:'Yc�A�*

loss��I>a���       �	8�'Yc�A�*

loss�X>�ۛ       �	Lo(Yc�A�*

loss���>A�s       �	=)Yc�A�*

lossF��=��a       �	��)Yc�A�*

loss`cB>��ק       �	�=*Yc�A�*

loss�=��h�       �	?�*Yc�A�*

loss,�z=W�m�       �	Jy+Yc�A�*

lossB>fA��       �	V,Yc�A�*

lossE�8> �Ev       �	m�,Yc�A�*

loss[q==��f       �	VF-Yc�A�*

loss6��=��1       �	��-Yc�A�*

loss��>-�:       �	��.Yc�A�*

loss���=/�\       �	�/Yc�A�*

loss�]=�qS'       �	��/Yc�A�*

loss���=K��L       �	�p0Yc�A�*

loss�:�=4)�       �	(1Yc�A�*

loss'$�=,Fr�       �	��1Yc�A�*

loss2>	x�       �	�p2Yc�A�*

loss,ǥ=
l��       �	�3Yc�A�*

loss���=J��       �	ŭ3Yc�A�*

lossÄO=�{�       �	R4Yc�A�*

loss���=J�       �	��4Yc�A�*

loss_�>.Kֆ       �	!�5Yc�A�*

loss�T>U�+�       �	�)6Yc�A�*

loss���=�Q�       �	��6Yc�A�*

loss ]3=@ed(       �	�V8Yc�A�*

loss�[�=���       �	�8Yc�A�*

loss�3�=�|�       �	w�9Yc�A�*

loss|:�= �@�       �	�*:Yc�A�*

loss�>�̂m       �	�:Yc�A�*

loss$
>L�
s       �	cd;Yc�A�*

loss�6�=�kM�       �	�<Yc�A�*

lossq=��z       �	��<Yc�A�*

loss��=�sc�       �	3=Yc�A�*

loss��=���       �	��=Yc�A�*

loss0o>��Ƈ       �	zr>Yc�A�*

loss8q�=&��       �	A?Yc�A�*

lossl�>����       �	Ū?Yc�A�*

loss�=2���       �	�F@Yc�A�*

loss��=���       �	h�@Yc�A�*

loss�ݵ=�N�       �	M�AYc�A�*

loss�=�Z�       �	�EBYc�A�*

loss��>�=��       �	o�BYc�A�*

loss��+=!+�       �	�qCYc�A�*

lossn;�=6+�       �	DYc�A�*

loss)��=�\A       �	=�DYc�A�*

loss�H�<ۤ�(       �	r1EYc�A�*

loss!�j>��2       �	`�EYc�A�*

lossȢ=�Ғ�       �	AcFYc�A�*

loss^=>�]       �	��FYc�A�*

loss���=j�F�       �	w�GYc�A�*

loss#�>)���       �	u;HYc�A�*

lossW	�=���|       �	x�HYc�A�*

lossD�o>��e       �	�wIYc�A�*

loss�x�=����       �	�JYc�A�*

loss�&�=c,*       �	�JYc�A�*

lossӔ�=���       �	�NKYc�A�*

lossk�=�H@�       �	��KYc�A�*

loss���=e�m�       �	BxLYc�A�*

loss�6�=MKx�       �	�MYc�A�*

loss�S>z��       �	
�MYc�A�*

loss���=���[       �	�=NYc�A�*

loss��=��7�       �	[�NYc�A�*

lossև�=�(+�       �	{hOYc�A�*

lossxI�=A_��       �	FPYc�A�*

lossE�=��_t       �	?�PYc�A�*

loss�D�=��E:       �	pCQYc�A�*

loss:i>���       �	*�QYc�A�*

loss�=�Rr       �	��RYc�A�*

loss��>�o*       �	SYc�A�*

lossA��=Y��N       �	��SYc�A�*

loss&��=+��a       �	8ITYc�A�*

loss���=��       �	�TYc�A�*

lossE��=���$       �	�{UYc�A�*

lossL#�=!��}       �	� VYc�A�*

loss�B=n}�       �	5�VYc�A�*

losslG�=�#\       �	�QWYc�A�*

loss�A�<���8       �	}�WYc�A�*

lossN�E=�``a       �	�XYc�A�*

loss/��<�4Mj       �	g*YYc�A�*

loss�#>�q�       �	7�YYc�A�*

loss�I�=./:_       �	�kZYc�A�*

lossn�>���       �	�[Yc�A�*

loss�>�1��       �	/�[Yc�A�*

loss��=�s       �	`=\Yc�A�*

loss҆�=�~��       �	E�\Yc�A�*

loss��=����       �	6v]Yc�A�*

loss= n=�]       �	9^Yc�A�*

loss�ѽ=�)�       �	��^Yc�A�*

lossޫ#>�<�       �	!>_Yc�A�*

loss��(>���       �	#�_Yc�A�*

loss8��<�d�       �	�n`Yc�A�*

loss�e�=�-�       �	CaYc�A�*

loss�h#=뽝#       �	ИaYc�A�*

loss�x>�ݧ       �	'3bYc�A�*

lossi�=�K��       �	�bYc�A�*

loss�=s��       �	�bcYc�A�*

loss
��=�_ۂ       �	�cYc�A�*

loss�dr=�$�       �	O�dYc�A�*

lossh׀=6�-       �	K#eYc�A�*

lossj��=	M"�       �	�eYc�A�*

loss[�f=XB�       �	�lfYc�A�*

lossl�Y=8G��       �	�gYc�A�*

loss�{3=��L�       �	\�gYc�A�*

loss���=p���       �	5]hYc�A�*

loss�p=��;       �		iYc�A�*

lossFcU> �(,       �	ݵiYc�A�*

lossD��=�1�`       �	�ZjYc�A�*

loss6��=k�d       �	��jYc�A�*

loss���=W�I       �	��kYc�A�*

loss4�>d�       �	�(lYc�A�*

loss��>�"Ԙ       �	��lYc�A�*

loss3��=�́�       �	yZmYc�A�*

loss_��=|�       �	0�mYc�A�*

loss�
>��d�       �	D�nYc�A�*

loss�6B=�E       �	�oYc�A�*

loss��=�B�       �	��oYc�A�*

loss�.=e[       �	MNpYc�A�*

loss���<��        �	��pYc�A�*

lossmBL>�(�w       �	ˀqYc�A�*

loss�x=��5       �	y"rYc�A�*

loss�D�=;2.       �	s�rYc�A�*

loss���=+g�X       �	bNsYc�A�*

loss��=,�8C       �	��sYc�A�*

loss�&=%� S       �	��tYc�A�*

loss�*�=�=y       �	�uYc�A�*

loss<+�=�L5C       �	��uYc�A�*

loss��]=;RF       �	�{vYc�A�*

loss�T�=X���       �	^wYc�A�*

loss��>Pg�       �	}AxYc�A�*

loss��8>\:��       �	s�xYc�A�*

loss1��=�g�       �	r2zYc�A�*

lossV�>�        �	8H{Yc�A�*

lossCA�=���       �	��{Yc�A�*

loss�S�=����       �	M�|Yc�A�*

loss=�K�       �	u?}Yc�A�*

loss��=�V9(       �	��}Yc�A�*

loss�r=X�e       �	�g~Yc�A�*

loss/�=�_L�       �	��~Yc�A�*

lossA>e#�       �	��Yc�A�*

loss3*�=���       �	Y3�Yc�A�*

loss��m=���       �	6̀Yc�A�*

lossd��=6j�       �	�f�Yc�A�*

loss{�=<�Y       �	.��Yc�A�*

loss39"=]Uiy       �	��Yc�A�*

loss2��=3:�       �	j.�Yc�A�*

lossQ�=7�(�       �	"ÃYc�A�*

loss��=@�S       �	�X�Yc�A�*

loss�Z!>��ͣ       �	>#�Yc�A�*

lossH�'>�=       �	ܸ�Yc�A�*

loss�V�=^��'       �	�M�Yc�A�*

loss ��=�pU       �	c�Yc�A�*

loss]��<��)       �	���Yc�A�*

loss(^=��F       �	K�Yc�A�*

loss8��=���       �	V��Yc�A�*

loss��=$[�f       �	rO�Yc�A�*

lossj4�=l'��       �	��Yc�A�*

loss��6>���=       �	E��Yc�A�*

lossV�=6�+>       �	r�Yc�A�*

loss=6>jƿW       �	���Yc�A�*

lossM4>����       �	uX�Yc�A�*

loss�t�=;}k`       �	E�Yc�A�*

losss3=b��       �	���Yc�A�*

loss.�=��\       �	���Yc�A�*

loss�.>|hl�       �	U1�Yc�A�*

loss���=	̻�       �	�̏Yc�A�*

lossU8>���       �	/k�Yc�A�*

lossE��<�6�       �	=�Yc�A�*

loss:%�=LW>)       �	ɭ�Yc�A�*

loss$o�<�&�       �	wH�Yc�A�*

loss�=_H��       �	��Yc�A�*

loss.->��ּ       �	ρ�Yc�A�*

loss��=;�2�       �	�!�Yc�A�*

lossڪ=��0       �	j��Yc�A�*

loss���=wCi�       �	b�Yc�A�*

loss�2�=��EF       �	��Yc�A�*

loss��=�F�       �	��Yc�A�*

loss���=:��L       �	�L�Yc�A�*

lossL��=OH·       �	��Yc�A�*

loss��=����       �	-z�Yc�A�*

loss�}>��P       �	��Yc�A�*

lossi�X=o��       �	 ��Yc�A�*

loss��@=���)       �	�B�Yc�A�*

loss��A=��       �	EؚYc�A�*

loss�<�=Oo�       �	�q�Yc�A�*

loss��=\���       �	��Yc�A�*

loss��=3o�m       �	a��Yc�A�*

loss��>\�$�       �	�>�Yc�A�*

loss=9s=�f       �	<ٝYc�A�*

loss�Q�=��qk       �	�r�Yc�A�*

loss�γ=*�
1       �	돟Yc�A�*

loss���=
��z       �	�&�Yc�A�*

loss���=뇏�       �	���Yc�A�*

lossx��=�c#�       �	�P�Yc�A�*

loss��=�{��       �	��Yc�A�*

loss�ߊ=p�#�       �	;��Yc�A�*

loss׊�=`.jN       �	E֣Yc�A�*

loss�ē=��Պ       �	�Yc�A�*

lossn�h>)��/       �	p��Yc�A�*

loss���=y�"       �	; �Yc�A�*

loss- ^>�c��       �	�!�Yc�A�*

loss��=��a�       �	˅�Yc�A�*

loss�M=��       �	�m�Yc�A�*

loss�-�=�F\�       �	.V�Yc�A�*

loss-�4>s��       �	s��Yc�A�*

loss��=�n
R       �	�9�Yc�A�*

loss�?>���       �	�|�Yc�A�*

lossC =O{��       �	�s�Yc�A�*

loss`1(>H�X#       �	N'�Yc�A�*

loss|�	>�xT�       �	�ʰYc�A�*

loss=u>z��       �	k�Yc�A�*

loss�c�=r���       �	�A�Yc�A�*

loss�@]>1X�A       �	j��Yc�A�*

losst�Y=���A       �	7��Yc�A�*

loss��g=5>�       �	�E�Yc�A�*

loss�̴=$N       �	ݴYc�A�*

loss�Z�=>��       �	-w�Yc�A�*

loss��==a���       �	k�Yc�A�*

lossJ�=ST��       �	Ҭ�Yc�A�*

loss�1�=��       �	�I�Yc�A�*

loss�D=`�       �	a��Yc�A�*

loss��=\�2*       �	ߣ�Yc�A�*

loss��1=p$�       �	z8�Yc�A�*

loss�i�<�t��       �	�ϹYc�A�*

lossꨛ=�7�       �	�b�Yc�A�*

loss:�>��r8       �	\�Yc�A�*

lossZw>��9�       �	��Yc�A�*

lossF{�=�Sb       �	���Yc�A�*

loss8�>�u�n       �	�0�Yc�A�*

loss�.=�K�
       �	�սYc�A�*

loss�[�=��MR       �	�x�Yc�A�*

loss�f>��j�       �	i�Yc�A�*

loss�C�=i���       �	 ��Yc�A�*

loss��=��R.       �	�N�Yc�A�*

lossI��=��5       �	���Yc�A�*

loss�.�=�8'�       �	���Yc�A�*

loss�=iv�8       �	N'�Yc�A�*

loss�W0=����       �	��Yc�A�*

loss�Cq>Z���       �	�a�Yc�A�*

lossq��=)��       �	���Yc�A�*

loss�$:>ӚW�       �	̚�Yc�A�*

lossHZ0>���       �	*7�Yc�A�*

loss==�= �Z       �	���Yc�A�*

loss���=��ѓ       �	�a�Yc�A�*

loss#G1>2�k       �	]��Yc�A�*

loss�C�=+���       �	֍�Yc�A�*

loss&�=;�h�       �	'�Yc�A�*

loss�o�=��3       �	v��Yc�A�*

lossx�=�ғ:       �	�Q�Yc�A�*

loss!�`=��&       �	H��Yc�A�*

lossx��=���O       �	p��Yc�A�*

loss8�4=�2�.       �	�.�Yc�A�*

loss��
=�8�u       �	���Yc�A�*

loss�,�=m7X"       �	�m�Yc�A�*

lossT��<.z�       �	���Yc�A�*

loss�W>�[J       �	��Yc�A�*

lossV'�=�Z*�       �	��Yc�A�*

loss��=�8��       �	~�Yc�A�*

loss��>�a��       �	Z��Yc�A�*

lossF=.=X7�<       �	Fa�Yc�A�*

lossh�<=��x       �	o�Yc�A�*

loss��<�Ӑ�       �	���Yc�A�*

loss�s�=�^h�       �	F�Yc�A�*

loss��=:o��       �	f��Yc�A�*

loss!I>��-       �	2u�Yc�A�*

loss�~[><�9       �	��Yc�A�*

loss�l�=D�.�       �	y��Yc�A�*

loss�L�=�T��       �	�>�Yc�A�*

lossO��=`�0       �	���Yc�A�*

lossv��=�z�       �	r�Yc�A�*

lossyM�=�D��       �	��Yc�A�*

loss�8�=�k�t       �	��Yc�A�*

loss1��=۲5       �	IJ�Yc�A�*

loss�1�=�X�       �	d��Yc�A�*

lossa�%>P�}Y       �	���Yc�A�*

loss���=�͵p       �		5�Yc�A�*

loss���=dC��       �	z�Yc�A�*

loss��0=k@��       �	û�Yc�A�*

loss��D=8���       �	`�Yc�A�*

loss��=;4u       �	���Yc�A�*

loss��O=P���       �	=��Yc�A�*

loss���=!2�       �	D3�Yc�A�*

loss�=B=ʇ(       �	���Yc�A�*

lossJ��=�Y�9       �	�o�Yc�A�*

loss&5�=H�&/       �	x�Yc�A�*

loss�>[�Q�       �	��Yc�A�*

loss��!=��        �	zR�Yc�A�*

loss-�g=!/��       �	���Yc�A�*

loss�M=�i��       �	͔�Yc�A�*

loss&*�=s��       �	?8�Yc�A�*

loss�={ٗ�       �	t	�Yc�A�*

loss�0>=�:�       �	v��Yc�A�*

loss�ra=�^       �	�_�Yc�A�*

lossE�,>|�2       �	�m�Yc�A�*

lossRk
>�E�       �	�>�Yc�A�*

lossMV�=&��       �	���Yc�A�*

loss1n�=����       �	Ή�Yc�A�*

loss�ߐ<e�7�       �	)%�Yc�A�*

loss��o=>��K       �	���Yc�A�*

loss��*=*��       �	�s�Yc�A�*

loss/1>_��       �	��Yc�A�*

loss-FC=�X�       �	1��Yc�A�*

loss*@�=Ǉ�y       �	{g�Yc�A�*

loss�N�=�fz�       �	-�Yc�A�*

loss*J�=66��       �	��Yc�A�*

loss$��<|���       �	O�Yc�A�*

loss�G�=T�}#       �	)��Yc�A�*

loss`u<߯�       �	��Yc�A�*

loss?�z=z[�       �	J&�Yc�A�*

lossix.=�U{{       �	/��Yc�A�*

loss ��=���       �	�c�Yc�A�*

loss�A=<C��       �	� �Yc�A�*

lossm�>��P�       �	��Yc�A�*

loss�C>�Y*       �	B?�Yc�A�*

loss�l
=���i       �	+��Yc�A�*

loss/�=l�o@       �	x{�Yc�A�*

loss�A�=I�b       �	�)�Yc�A�*

loss���<)�k�       �	���Yc�A�*

loss���<�N�       �	Vb�Yc�A�*

loss�jx<a�m       �	Y��Yc�A�*

loss��V<8e��       �	t��Yc�A�*

loss�E<�*	       �	�7�Yc�A�*

loss�P�=%jb�       �	��Yc�A�*

lossq�<��       �	��Yc�A�*

loss!`�=����       �	,)�Yc�A�*

loss�P�;���       �	T��Yc�A�*

lossM��:k���       �	X��Yc�A�*

loss���:H���       �	�D�Yc�A�*

lossm0�=]�P       �	���Yc�A�*

lossgg>\��Y       �	���Yc�A�*

loss�rD=%nq�       �	ap�Yc�A�*

lossU��<}��       �	�Z Yc�A�*

loss�Fm=�K�Q       �	8� Yc�A�*

lossV<j>�ܝ.       �	��Yc�A�*

loss�\3<&w1�       �	�3Yc�A�*

loss�TE=<��       �	9�Yc�A�*

loss��O=����       �	�oYc�A�	*

lossQ��>V:��       �	,Yc�A�	*

lossR�=_Hj       �	�Yc�A�	*

lossT�&=#�T       �	�EYc�A�	*

lossME6>o��       �	��Yc�A�	*

loss���=Bi�_       �	ԀYc�A�	*

lossZ�>��5       �	�Yc�A�	*

loss��>\�       �	1�Yc�A�	*

loss�C�=�?sO       �	LYc�A�	*

lossG >Ϛ       �	��Yc�A�	*

loss��=�e)       �	F~	Yc�A�	*

loss��=�k~�       �	�
Yc�A�	*

losslA>�P�       �	>�
Yc�A�	*

lossD�>��@�       �	�KYc�A�	*

loss\[_=Qؕ�       �	jYc�A�	*

lossM��=��J       �	�Yc�A�	*

loss�I�=}��       �	��Yc�A�	*

loss�!�=�(       �	.=Yc�A�	*

loss1��<૪E       �	�0Yc�A�	*

loss�;c=�pV       �	��Yc�A�	*

loss�B>nA��       �	�mYc�A�	*

loss �L=��       �	�Yc�A�	*

loss��H=S��       �	��Yc�A�	*

loss���<am`       �	oHYc�A�	*

lossoH�=��\       �	�)Yc�A�	*

loss�<t=p�h�       �	��Yc�A�	*

loss��>�K��       �	�iYc�A�	*

loss�n,>��       �	C:Yc�A�	*

loss��'=7�)M       �	��Yc�A�	*

lossNg�=�qx�       �	�{Yc�A�	*

loss��=_���       �	YYc�A�	*

loss��<&*=       �	|�Yc�A�	*

loss��/=C�G�       �	�OYc�A�	*

loss�f�=ò8�       �	�Yc�A�	*

losst�I=��       �	u�Yc�A�	*

lossvVD=_��       �	YLYc�A�	*

lossv[>�B-{       �	�Yc�A�	*

loss)O�=+敕       �	˺Yc�A�	*

lossÕA=�        �	^hYc�A�	*

loss�Rb=�*�A       �	+ Yc�A�	*

loss-4=y�c       �	�� Yc�A�	*

loss�J�=L��       �	h�!Yc�A�	*

lossA.�=��&�       �	n0"Yc�A�	*

loss���=|龊       �	��"Yc�A�	*

loss�6>�waS       �	s#Yc�A�	*

loss�8=�!6       �	1$Yc�A�	*

loss���=9/�       �	T�$Yc�A�	*

loss�h�=���       �	�E%Yc�A�	*

lossT�+=����       �	�#&Yc�A�	*

loss�8>K}ؠ       �	ke@Yc�A�	*

loss��=2���       �	��@Yc�A�	*

loss�i�=�D�       �	�AYc�A�	*

lossd��=��UM       �	�>BYc�A�	*

loss�\�=�P.�       �	|{CYc�A�	*

loss.�=���       �	{-DYc�A�	*

loss���=�CC(       �	��DYc�A�	*

loss�M�=E�45       �	K[EYc�A�	*

loss2c�=K��       �	��EYc�A�	*

loss�>ڀdD       �	)�FYc�A�	*

loss��<u;��       �	�bGYc�A�	*

loss�ڰ=�%�       �	��HYc�A�	*

loss}z�=q.F�       �	� JYc�A�	*

loss�=M�M�       �	��JYc�A�	*

loss��=�{��       �	�FKYc�A�	*

lossq�>�z��       �	Q�KYc�A�	*

lossW�<��?�       �	ŐLYc�A�	*

loss�ߖ=ҕT�       �	7MYc�A�	*

lossxK�=��a       �	�MYc�A�	*

loss8�N>U�F       �	�vNYc�A�	*

loss�D=\��W       �	~OYc�A�	*

losswr�=rC;�       �	b�OYc�A�	*

loss �=緈o       �	�hPYc�A�	*

lossi,>�&��       �	�QYc�A�	*

lossa�=�:�       �	U�QYc�A�	*

loss3�=m��       �	�8RYc�A�	*

loss��4=N�       �	��RYc�A�	*

loss�<
:�       �	�dSYc�A�	*

loss�[�=��I       �	r�SYc�A�	*

loss8��=���       �	:�TYc�A�	*

loss�9�=�M��       �	�*UYc�A�	*

loss_m�<���H       �	7�UYc�A�	*

loss
j=R:��       �	�]VYc�A�	*

loss��a>>�       �	��VYc�A�	*

loss��=���       �	�WYc�A�	*

loss���= �V�       �	&XYc�A�	*

loss�~�=x�	       �	��XYc�A�	*

lossv��=X�J       �	�TYYc�A�	*

loss��<>~	=       �	A�YYc�A�	*

loss/��=��A�       �	��ZYc�A�	*

loss�p{=*Dɞ       �	�![Yc�A�	*

loss
�y=��       �	��[Yc�A�	*

loss&X&=���Q       �	?n\Yc�A�	*

loss�F�=�nE�       �	:]Yc�A�	*

lossň�=�`       �	o�]Yc�A�	*

lossfj�=2�       �	->^Yc�A�	*

lossl��=��Fw       �	�^Yc�A�	*

loss�M�=ڿT�       �	�m_Yc�A�	*

lossݠ�=��2�       �	%`Yc�A�	*

lossmm�<�̚       �	-�`Yc�A�	*

loss
S�<�M�       �	�5aYc�A�	*

lossN�-=Y       �	`�aYc�A�	*

loss� 
=�
��       �	�hbYc�A�	*

loss�J�>�*�       �	�cYc�A�	*

loss��B=�c�       �	z�cYc�A�	*

lossɏE<Q,Z       �	�EdYc�A�	*

lossq�=p��g       �	��dYc�A�	*

loss(��<��	       �	l�eYc�A�	*

loss!=�=��7�       �	9fYc�A�	*

lossV�j=��_d       �	8�fYc�A�	*

loss�Tg=�'C5       �	�sgYc�A�	*

loss�F�=ٓ7       �	hYc�A�	*

loss��T=���       �	D�hYc�A�	*

loss���=�A{       �	�BiYc�A�	*

loss|�=�F�       �	j�iYc�A�	*

loss�"=�{Ɔ       �	�xjYc�A�	*

lossd%�=��Ǘ       �	�kYc�A�	*

loss��=��я       �	H�kYc�A�	*

lossN> >w8?       �	K=lYc�A�	*

lossͼ�=%^��       �	��lYc�A�	*

lossLh�=o�t�       �	DnmYc�A�	*

loss=��=���       �	nYc�A�	*

loss�?>=�>�       �	��nYc�A�	*

loss�
�=]�f[       �	�8oYc�A�	*

loss)�q=��P       �	�oYc�A�	*

loss=q.= ��       �	�qpYc�A�	*

loss���=b0-�       �	>qYc�A�	*

lossl�=��
�       �	��qYc�A�	*

loss��=�a@       �	�?rYc�A�	*

loss�i�=�{%       �	�rYc�A�
*

losst��=@��       �	k�sYc�A�
*

loss��>ݶF,       �	% tYc�A�
*

lossx6�=ԓ�       �	��tYc�A�
*

lossڡ)<���@       �	.ruYc�A�
*

loss-˅=H�5�       �	�vYc�A�
*

loss6�=��       �	i�vYc�A�
*

loss�u�=�1�       �	|HwYc�A�
*

lossh��=(qG�       �	��wYc�A�
*

lossq`�=��r�       �	�yxYc�A�
*

lossi��<il$"       �	�yYc�A�
*

loss���=�y_W       �	Y�yYc�A�
*

lossP�=��       �	�IzYc�A�
*

loss�	=j�p�       �	��zYc�A�
*

loss1��=���       �	�{Yc�A�
*

lossT�>�E�X       �	>&|Yc�A�
*

loss���<~��,       �	4�|Yc�A�
*

loss�k=KiD       �	�T}Yc�A�
*

lossޝ=H��       �	u~Yc�A�
*

loss*l>�?>       �	ܡ~Yc�A�
*

loss�,=�J=       �	GYc�A�
*

loss�'�<?Bg�       �	��Yc�A�
*

loss8=Q2�v       �	o��Yc�A�
*

loss.4&>�']X       �	!�Yc�A�
*

loss�M=T��W       �	���Yc�A�
*

loss�c�=��       �	�^�Yc�A�
*

loss�	>E�4]       �	���Yc�A�
*

loss��<x,k        �	x��Yc�A�
*

loss�G	=ˁ�)       �	�8�Yc�A�
*

loss�)>��       �	{لYc�A�
*

loss�Y�=^oy;       �	s�Yc�A�
*

lossa>�@?Y       �	��Yc�A�
*

loss�Z+=+���       �	���Yc�A�
*

loss�\=S��       �	K�Yc�A�
*

loss��9=i	�       �	|a�Yc�A�
*

loss��=YK��       �	1�Yc�A�
*

loss���<fmF       �	�'�Yc�A�
*

loss��Q=���       �	D��Yc�A�
*

loss厐=����       �	�W�Yc�A�
*

loss��=]q�d       �	��Yc�A�
*

lossC�F=��q-       �	ۇ�Yc�A�
*

loss�
�=H7�b       �	�<�Yc�A�
*

loss|��<\��       �	�ލYc�A�
*

loss���=��)^       �	��Yc�A�
*

loss���=�1E       �	W&�Yc�A�
*

loss�s<�l�       �	
��Yc�A�
*

loss��J=��_S       �	�h�Yc�A�
*

loss2��=bq�z       �	��Yc�A�
*

loss�q�=�߸�       �	"��Yc�A�
*

loss�3�=�x�*       �	�C�Yc�A�
*

loss1�t=��1       �	jݒYc�A�
*

loss�l=w�       �	{��Yc�A�
*

loss�;�=�ç
       �	��Yc�A�
*

loss�/�=��83       �	�ƔYc�A�
*

lossJ@�<1(�       �	�h�Yc�A�
*

loss�3�=�S"       �	7 �Yc�A�
*

loss�:v=�~S�       �	>��Yc�A�
*

lossX��=M�/       �	�;�Yc�A�
*

loss� �<1n-�       �	ԗYc�A�
*

loss �<T�X       �	�m�Yc�A�
*

loss��$=�m�       �	� �Yc�A�
*

loss�\<��>�       �	Ԛ�Yc�A�
*

loss�b==��VX       �	�1�Yc�A�
*

lossSBJ=�[!       �	�(�Yc�A�
*

loss�&#=���       �	=֛Yc�A�
*

loss�l�=`C��       �	�r�Yc�A�
*

loss#U=���Y       �	0�Yc�A�
*

loss�s�=�VJ�       �	���Yc�A�
*

lossT-=2�I�       �	�G�Yc�A�
*

loss�Fq=M���       �	��Yc�A�
*

lossdQ�=HL-       �	k��Yc�A�
*

loss�ظ<���6       �	��Yc�A�
*

lossצ�=,z�^       �	���Yc�A�
*

loss}R�=�O       �	�{�Yc�A�
*

lossM�>��^       �	q�Yc�A�
*

lossV�[=�*��       �	���Yc�A�
*

loss�'�;K�e�       �	�H�Yc�A�
*

lossH2�={�        �	�ݣYc�A�
*

loss$x�=ߎ��       �	�w�Yc�A�
*

loss��o>���T       �	�%�Yc�A�
*

loss��M<���       �	�ǥYc�A�
*

loss�H�=X�n�       �	���Yc�A�
*

loss�y=��       �	!>�Yc�A�
*

lossԛ`=�#�4       �	+ݧYc�A�
*

loss�$Y<���       �	Xs�Yc�A�
*

loss���=�%;�       �	t�Yc�A�
*

loss��P=�8�       �	Y��Yc�A�
*

lossXh<Q�1K       �	�P�Yc�A�
*

lossÂ4=�'S�       �	V�Yc�A�
*

loss��}=�O3P       �	Ė�Yc�A�
*

loss���=$�>       �	2=�Yc�A�
*

loss8z�=�7�       �	(ҬYc�A�
*

loss��=��       �	ke�Yc�A�
*

loss\G�=Qξ�       �	v��Yc�A�
*

loss�Ġ=(�ڀ       �	ٓ�Yc�A�
*

loss]��<�`!-       �	�(�Yc�A�
*

loss`1;=���       �	���Yc�A�
*

loss���=�l�c       �	UQ�Yc�A�
*

loss�8p=1��*       �	��Yc�A�
*

loss�hC=`��       �	��Yc�A�
*

loss��=�$�:       �	��Yc�A�
*

loss��<8��X       �	!��Yc�A�
*

loss �A=U���       �	�I�Yc�A�
*

lossv�b<� �e       �	f��Yc�A�
*

loss�G=�Q�!       �	%��Yc�A�
*

losstK�<DrO�       �	�0�Yc�A�
*

loss�3=@��       �	�˵Yc�A�
*

loss\E=���y       �	�r�Yc�A�
*

loss���=��^?       �	��Yc�A�
*

loss
�=L���       �	�ƷYc�A�
*

loss��2=��v�       �	�]�Yc�A�
*

loss��=��3�       �	b��Yc�A�
*

loss_�;=�8c�       �	F��Yc�A�
*

loss��=�AZ       �	���Yc�A�
*

loss$iP=;�x�       �	V��Yc�A�
*

loss#�3<�m*9       �	�G�Yc�A�
*

loss+�=>׎       �	.�Yc�A�
*

loss%��=goѯ       �	�ʾYc�A�
*

loss�}=G�X>       �	o�Yc�A�
*

loss�w=/�S       �	��Yc�A�
*

loss��|=�Z�7       �	H��Yc�A�
*

loss<ԍ=���V       �	�%�Yc�A�
*

loss. =�%��       �	��Yc�A�
*

loss��=���c       �	y��Yc�A�
*

loss��$=�y��       �	W��Yc�A�
*

loss~S=��       �	�g�Yc�A�
*

loss�<�<HbiJ       �	�
�Yc�A�
*

loss�62>�2��       �	Q��Yc�A�
*

loss�>>�Y       �	\V�Yc�A�*

loss7�=mM��       �	���Yc�A�*

loss;�H= �lk       �	ӈ�Yc�A�*

loss&S=ާ+       �	�>�Yc�A�*

loss�m=.}��       �	d��Yc�A�*

loss��1=!��g       �	���Yc�A�*

loss�h#=��z       �	9�Yc�A�*

loss8�=����       �	D��Yc�A�*

loss�>�=��c�       �	��Yc�A�*

loss�X=]��       �	�#�Yc�A�*

loss�la>�TNh       �	�	�Yc�A�*

lossH� >ps       �	��Yc�A�*

loss�E>���       �	�x�Yc�A�*

loss$�=:Ǚ�       �	��Yc�A�*

lossX�=��?�       �	i:�Yc�A�*

loss׍�=A�T{       �	'��Yc�A�*

loss0�>�܁       �	�q�Yc�A�*

loss��<8a�>       �	o�Yc�A�*

loss_�i=��|)       �	�n�Yc�A�*

loss쥬=%YiT       �	�	�Yc�A�*

loss�w=�9}V       �	���Yc�A�*

lossJ e=F��       �	6=�Yc�A�*

lossZy�=�۹       �	$��Yc�A�*

loss{~K<�އ       �	m�Yc�A�*

lossz|.=�       �	�	�Yc�A�*

loss�ǐ=�G��       �	���Yc�A�*

loss��M=Ĕ�       �	�L�Yc�A�*

loss���=�c!d       �	��Yc�A�*

loss�`�=���d       �	���Yc�A�*

loss��\=�{�       �	B&�Yc�A�*

loss?��=�`֣       �	���Yc�A�*

loss`&>�e�z       �	�u�Yc�A�*

loss==Xc�       �	��Yc�A�*

loss�M�=���I       �	c��Yc�A�*

losso�N=tP�n       �	.T�Yc�A�*

lossF)�=-        �	r��Yc�A�*

loss�/�=���       �	ٕ�Yc�A�*

loss��=���       �	0�Yc�A�*

lossgH�=��
       �	m��Yc�A�*

loss�+L=�E�       �	a�Yc�A�*

loss�@�<�ʼ"       �	I��Yc�A�*

loss�q�=u�)�       �	ߊ�Yc�A�*

loss��U=æ�       �	��Yc�A�*

lossmf�=�ߧk       �	���Yc�A�*

lossH�B=uB,!       �	���Yc�A�*

loss\��=-{ZQ       �	�{�Yc�A�*

loss:d=y��       �	��Yc�A�*

loss]f�<��u�       �	��Yc�A�*

lossx��<��'       �	�F�Yc�A�*

lossT;=��       �	'��Yc�A�*

lossՕ=��lA       �	Su�Yc�A�*

loss�h=^FQ       �	�,�Yc�A�*

loss�}�=��@A       �	H��Yc�A�*

loss�Ȳ<�w�       �	�j�Yc�A�*

loss��=!7��       �	��Yc�A�*

lossxm>7{;       �	Y��Yc�A�*

lossH�=�0�       �	\�Yc�A�*

loss6M	>�pE�       �	L��Yc�A�*

lossV=e'	�       �	F��Yc�A�*

loss�ǹ=3P�/       �	�7�Yc�A�*

loss���=�S��       �	���Yc�A�*

loss�=��       �	p�Yc�A�*

loss�%P=7%��       �	��Yc�A�*

lossRWd=�d       �	��Yc�A�*

loss��=���w       �	|F�Yc�A�*

loss(�Y=5C=�       �	�9�Yc�A�*

loss��=PU��       �	h��Yc�A�*

loss��>Y��p       �	�~�Yc�A�*

loss�ۆ=�T	       �	��Yc�A�*

loss�=]=�{�H       �	���Yc�A�*

lossx��=M�;r       �	�I�Yc�A�*

loss/̼=Ak�%       �	t��Yc�A�*

loss�&t=+c�       �	H��Yc�A�*

loss�-<���f       �	� �Yc�A�*

loss茴<��K       �	���Yc�A�*

loss�y�=��ӝ       �	�g�Yc�A�*

loss��=��E       �	��Yc�A�*

loss�Z�=Ox�       �	���Yc�A�*

lossW��=HN��       �	cC�Yc�A�*

lossd�=�z�Y       �	���Yc�A�*

loss�5=%HTr       �	b��Yc�A�*

loss���<:lȎ       �	�P�Yc�A�*

loss{}X=��؇       �	���Yc�A�*

lossv*%<�$�       �	"��Yc�A�*

loss�!�=F̧       �	+�Yc�A�*

loss8.=���i       �	���Yc�A�*

loss��|<8�9       �	�x Yc�A�*

loss=^T=��w?       �	�Yc�A�*

loss�ݩ=h5L       �	A�Yc�A�*

lossU/�=�U�       �	bNYc�A�*

loss	)/>��4^       �	`�Yc�A�*

loss�>�+�       �	��Yc�A�*

loss��>7�3�       �	�9Yc�A�*

loss�\G=�ɨ       �	h�Yc�A�*

loss��=� D       �	{�Yc�A�*

lossX=���       �	�,Yc�A�*

loss�e�=����       �	F�Yc�A�*

lossv��<g��c       �	��Yc�A�*

loss�ئ=y?{       �	Yc�A�*

loss�C�;�1�b       �	��Yc�A�*

loss���=�S�       �	�Z	Yc�A�*

loss�<�'�)       �	4�	Yc�A�*

loss� =E���       �	��
Yc�A�*

loss-�=�C;B       �	["Yc�A�*

loss�a�=2���       �	y�Yc�A�*

loss ��=��       �	wdYc�A�*

losse�W=0mo>       �	�	Yc�A�*

lossEIT=!*'H       �	T�Yc�A�*

loss�־=��n*       �	�GYc�A�*

loss�p�<3��       �	��Yc�A�*

lossh#{=�#�       �	u�Yc�A�*

loss��q=4.S�       �	A*Yc�A�*

lossC��=��        �	˿Yc�A�*

lossBV#=8/c�       �	�]Yc�A�*

loss���=*���       �	�8Yc�A�*

loss�Q=�bw�       �	��Yc�A�*

loss��<�$'�       �	�{Yc�A�*

lossK֠=^j%Z       �	�.Yc�A�*

loss�'=,`�       �	p�Yc�A�*

loss�E�<x��       �	�gYc�A�*

loss���=�%Ӟ       �	�Yc�A�*

loss��=�s(�       �	��Yc�A�*

lossؒ=�fY�       �	�Yc�A�*

loss1��<�&�       �	�-Yc�A�*

loss�x:<л�]       �	.�Yc�A�*

loss�T8=�2�$       �	�iYc�A�*

loss�6=G�1       �	�Yc�A�*

loss�'�=�2��       �	ӤYc�A�*

loss.�=b)�       �	�BYc�A�*

loss:j=�"�       �	T�Yc�A�*

lossL�*=x�\/       �	zTYc�A�*

loss���=Nq_j       �	��Yc�A�*

lossQ�<p���       �	4�Yc�A�*

loss���<���w       �	u<Yc�A�*

loss��c=�@G       �	��Yc�A�*

lossM>)A�t       �	X� Yc�A�*

loss,�,=T���       �	a5!Yc�A�*

loss`X>H���       �	$�!Yc�A�*

loss$F�=��7       �	�r"Yc�A�*

loss�S�< ���       �	#Yc�A�*

loss}�=��S|       �	��#Yc�A�*

loss�=-�x       �	�D$Yc�A�*

loss�_�=��:l       �	��$Yc�A�*

lossj��=��7�       �	[�%Yc�A�*

loss�1�<�� �       �	g�&Yc�A�*

loss�LE=�7ش       �	�'Yc�A�*

loss�|(=r4_�       �	�l(Yc�A�*

loss�J�=��       �	N)Yc�A�*

loss���<�j�,       �	į)Yc�A�*

loss�Y;=�72I       �	9F*Yc�A�*

lossԡ.=xܲu       �	�*Yc�A�*

loss�P=
�Ǯ       �	�}+Yc�A�*

loss_Q�=����       �	�,Yc�A�*

loss��@=��w       �	%�,Yc�A�*

loss�#=�r�1       �	4I-Yc�A�*

loss?'�<g� �       �	m�-Yc�A�*

lossTv(>#,�       �	�y.Yc�A�*

loss3>S>h-       �	�/Yc�A�*

loss���=�k �       �	�/Yc�A�*

lossH�='�c       �	�Q0Yc�A�*

loss;�B=U(�x       �	>�0Yc�A�*

lossܗ3=�_7�       �	u�1Yc�A�*

losscҗ<Ɯ:�       �	wK2Yc�A�*

lossZQR=8��t       �	��2Yc�A�*

loss��==��       �	��3Yc�A�*

loss�C=�3�J       �	C4Yc�A�*

loss���<��B       �	Ϻ4Yc�A�*

loss��S=���q       �	R5Yc�A�*

lossNH>�I�`       �	 6Yc�A�*

loss t�=Y���       �	�6Yc�A�*

loss��=hR��       �	b�7Yc�A�*

loss�K$=��J�       �	W�8Yc�A�*

loss4L%>�x�;       �	�.9Yc�A�*

losst��<=���       �	B�9Yc�A�*

loss��=���       �	�i:Yc�A�*

loss��<DvV       �	�;Yc�A�*

lossX��< }       �	��;Yc�A�*

loss6�<�\-K       �	2<Yc�A�*

lossy=���       �	��<Yc�A�*

lossGs=I.�       �	��=Yc�A�*

loss� �={�
k       �	1a>Yc�A�*

loss�{�=�p�5       �	a�>Yc�A�*

loss�5�<Z#
�       �	{�?Yc�A�*

lossGr=�d�       �	�m@Yc�A�*

loss@�?=�ysi       �	�1AYc�A�*

loss�=���       �	�BYc�A�*

loss�}@=<Jd�       �	1�BYc�A�*

loss��=�^��       �	�MCYc�A�*

loss�?>��J�       �	��CYc�A�*

loss=�M<���       �	.�DYc�A�*

loss8��<�$�4       �	�MEYc�A�*

lossN�,=狒       �	�*FYc�A�*

lossC�d=�"��       �	O�FYc�A�*

loss�}/=IF��       �	fjGYc�A�*

loss��R=��N;       �	�@HYc�A�*

loss�8Y=7j�       �	��HYc�A�*

loss_��<��B       �	��IYc�A�*

loss�1�=�0�w       �	!!JYc�A�*

loss�O=���n       �	��JYc�A�*

loss#�=(o~�       �	CYKYc�A�*

loss@��=	
�       �	��LYc�A�*

lossؚ�=m�8�       �	ӆMYc�A�*

loss��=�h�       �	�5NYc�A�*

lossDV�=C \�       �	\�NYc�A�*

loss�d�=�Ǵb       �	��OYc�A�*

loss�o=}g7�       �	9PYc�A�*

loss��>�M�       �	<�PYc�A�*

loss$B�=���       �	>xQYc�A�*

loss4�6>�|/       �	eRYc�A�*

lossS#!<�gg�       �	��RYc�A�*

loss
��<WQ�n       �	�SYc�A�*

lossV<r=��       �	�<TYc�A�*

lossI�|=Əօ       �	&�TYc�A�*

lossA�`=��4@       �	��UYc�A�*

loss�eX=�o&�       �	�)VYc�A�*

loss��%=�p"       �	��VYc�A�*

loss
�a>�:�       �	:�WYc�A�*

lossr~c=#�       �	>XYc�A�*

losso32>e�n       �	�XYc�A�*

loss��`=$���       �	tYYc�A�*

loss׌/>��a�       �	�ZYc�A�*

loss�p�=�]@       �	M�ZYc�A�*

loss�ݝ=�|�       �	:\[Yc�A�*

lossT�{=2{�?       �	]�[Yc�A�*

loss�5=�Z�       �	��\Yc�A�*

loss�c=�j�       �	"6]Yc�A�*

loss�@g=�Q6f       �	�]Yc�A�*

loss��=��{        �	��^Yc�A�*

loss]V�<�T       �	�_Yc�A�*

lossa&O=O
�K       �	�_Yc�A�*

loss�Xt=�lVW       �	�N`Yc�A�*

loss��<�|a�       �	W�`Yc�A�*

loss��1=4�       �	�aYc�A�*

lossX>8��       �	AIbYc�A�*

loss�<>H���       �	u�bYc�A�*

loss�\�=F�O"       �	v�cYc�A�*

loss��>$T%m       �	�.dYc�A�*

loss�ĥ<����       �	`�dYc�A�*

lossRՈ=N��       �	fYc�A�*

loss��/>��       �	̲fYc�A�*

loss�\�=�"��       �	�OgYc�A�*

lossf�=���       �	E�gYc�A�*

loss�y�=�,�       �	��hYc�A�*

loss�ɠ=e��       �	NciYc�A�*

lossH�)=��I       �	;�iYc�A�*

loss0Y=�>       �	��jYc�A�*

loss��>��2       �	)YkYc�A�*

loss�*>�N       �	�kYc�A�*

loss�̦=��9d       �	ŏlYc�A�*

loss�C>�L��       �	R*mYc�A�*

loss�Gk=�ͯ�       �	��mYc�A�*

lossHvQ=�yS       �	)[nYc�A�*

loss��=��aI       �	��nYc�A�*

loss�~(=Cv@w       �	��oYc�A�*

loss��L<o،0       �	�hpYc�A�*

loss*e<�9       �	�qYc�A�*

loss��b=�D��       �	ǟqYc�A�*

loss��A=�5cr       �	\:rYc�A�*

loss�8B=����       �	��rYc�A�*

loss;�f=��l       �	%ysYc�A�*

loss-M =x,��       �	�$tYc�A�*

loss�C�=G&�X       �	��tYc�A�*

loss�<6!�s       �	K�uYc�A�*

loss��>Ҟ��       �	y:vYc�A�*

lossmw6=!�:�       �	��vYc�A�*

loss�l�=w(�N       �	wYc�A�*

lossu>���z       �	W'xYc�A�*

loss��$=;�       �	��xYc�A�*

lossF�c<��|       �	@�yYc�A�*

loss�� =}��       �	�(zYc�A�*

loss<�=�8n:       �	t�zYc�A�*

loss?L�<I��       �	k�{Yc�A�*

loss��%>��f       �	�,|Yc�A�*

lossw�>r`RF       �	I}Yc�A�*

loss1��=Y�-       �	x�}Yc�A�*

lossd�k<h��       �	N�~Yc�A�*

loss|�>�R�K       �	�AYc�A�*

loss֯�=����       �	N�Yc�A�*

loss&-=��B       �	�πYc�A�*

loss�x=tKv�       �	p��Yc�A�*

lossީ>�N��       �	K:�Yc�A�*

loss6��<�ǝ7       �	��Yc�A�*

loss��=%���       �	���Yc�A�*

loss��c=�V�[       �	6�Yc�A�*

loss	��=[ܞ�       �	��Yc�A�*

lossh��<\�       �	���Yc�A�*

loss	�=m+��       �	@3�Yc�A�*

lossc�W<Ip|       �	�ՆYc�A�*

lossɖ�<z~��       �	=|�Yc�A�*

loss��=�/�       �	c&�Yc�A�*

loss��=����       �	ʈYc�A�*

loss��U=o/+       �	�r�Yc�A�*

loss1��=���I       �	��Yc�A�*

loss�ab=��{l       �	�ƊYc�A�*

loss���<�+�       �	.v�Yc�A�*

loss}�Z<;���       �	��Yc�A�*

loss��k<!p\       �	+��Yc�A�*

loss,�>)G�	       �	`r�Yc�A�*

lossm�(=32       �	"�Yc�A�*

lossDB�=���m       �	qˎYc�A�*

lossY=�.�       �	�u�Yc�A�*

loss��=DB9�       �	�(�Yc�A�*

loss�=�=���       �	�ؐYc�A�*

loss�X=ǅ0�       �	L��Yc�A�*

loss9=LF       �	�:�Yc�A�*

loss�.%<^���       �	��Yc�A�*

loss���=��H       �	X��Yc�A�*

lossq�<=EQ��       �	~;�Yc�A�*

loss��=V��        �	���Yc�A�*

loss��=�F�w       �	ߦ�Yc�A�*

loss�oR=!��       �	�H�Yc�A�*

loss��/=���       �	��Yc�A�*

lossF=�=:��D       �	Ҍ�Yc�A�*

loss�=��0       �	s,�Yc�A�*

losszG=�U<�       �	wؘYc�A�*

loss�0<�ܢ�       �	���Yc�A�*

loss��=��]�       �	�+�Yc�A�*

loss�l!=��~       �	ҚYc�A�*

loss��=�D.       �	>x�Yc�A�*

loss�d�<�X��       �	�$�Yc�A�*

loss�y�=~�e�       �	yʜYc�A�*

loss.�H=�4�M       �	p�Yc�A�*

loss��1=\>t�       �	��Yc�A�*

loss�=�?k       �	 ��Yc�A�*

loss�w=HD%~       �	�e�Yc�A�*

loss�a$<E��_       �	_�Yc�A�*

loss���;(Z^�       �	[��Yc�A�*

loss���<2�E�       �	�T�Yc�A�*

loss$��<���       �	M�Yc�A�*

loss/g=�&{8       �	���Yc�A�*

loss��o<�!�_       �	~7�Yc�A�*

lossj��<��}�       �	ӣYc�A�*

losss�<!�QW       �	us�Yc�A�*

loss�A;|Ɇ       �	% �Yc�A�*

loss�+&;9o։       �	H�Yc�A�*

loss���:�U�       �	E��Yc�A�*

loss��<��)       �	�?�Yc�A�*

loss�w�=�jޮ       �	n��Yc�A�*

loss(?�<�`�K       �	﬩Yc�A�*

lossu�;�v�       �	�ƪYc�A�*

loss@Z�<�
�       �	�s�Yc�A�*

loss��w>�D�       �	�Yc�A�*

lossEUk<�       �	�I�Yc�A�*

loss�b<���0       �	g�Yc�A�*

loss�w=��Y�       �	��Yc�A�*

loss3f�=���       �	TůYc�A�*

loss �)=��       �	�^�Yc�A�*

lossDq�<�L7       �	�U�Yc�A�*

loss���=eI��       �	���Yc�A�*

loss�T=&Vf       �	���Yc�A�*

loss�zB>�2�       �	gC�Yc�A�*

loss��l=<���       �	��Yc�A�*

lossr@�=�	       �	�}�Yc�A�*

loss�=`}��       �	��Yc�A�*

loss���=�γ       �	_��Yc�A�*

lossnǻ=��$       �	eU�Yc�A�*

loss.y�=�5Kb       �	���Yc�A�*

loss��.=E@ޠ       �	#��Yc�A�*

loss.��=C�0S       �	��Yc�A�*

loss7��<�~�       �	�Yc�A�*

loss�`>��E       �	"��Yc�A�*

loss�<�T��       �	F'�Yc�A�*

lossU4=�H%       �	h?�Yc�A�*

lossa�1=Ak+�       �	8׻Yc�A�*

lossۼ1=�q�       �	E�Yc�A�*

lossQ��<��n�       �	�&�Yc�A�*

loss \=\��       �	"ǽYc�A�*

loss��K<5b��       �	k�Yc�A�*

loss��<X'�       �	��Yc�A�*

loss��E=�[
       �	u��Yc�A�*

lossT�=�p�'       �	
L�Yc�A�*

loss���=���       �	6��Yc�A�*

loss��?<
�mN       �	I��Yc�A�*

loss�R�=�	dp       �	�F�Yc�A�*

loss�f�=�+        �	���Yc�A�*

loss�H<y$�i       �	;�Yc�A�*

loss�=�=Z       �	�T�Yc�A�*

loss�*=	��       �	���Yc�A�*

loss��8<�]M       �	��Yc�A�*

loss�,>=��'       �	1\�Yc�A�*

loss��z=��6�       �	���Yc�A�*

loss&�<8�       �	��Yc�A�*

loss�uU<+�|B       �	G9�Yc�A�*

lossӣ=8�"�       �	5��Yc�A�*

lossa��<PF^       �	��Yc�A�*

lossn�_=� P       �	�{�Yc�A�*

loss�Q9=���       �	W"�Yc�A�*

lossغ=��f       �	��Yc�A�*

loss�U�=��)a       �	�]�Yc�A�*

loss| <=��&�       �	��Yc�A�*

losswG�=f��       �	h��Yc�A�*

loss�-�<��       �	O�Yc�A�*

loss;��<g�       �	���Yc�A�*

loss�>xH�E       �	��Yc�A�*

loss��=��]       �	�o�Yc�A�*

loss�*�=��+       �	��Yc�A�*

loss�u�=��0_       �	���Yc�A�*

loss�x�=��b�       �	;n�Yc�A�*

loss�=���       �	��Yc�A�*

loss�B�=�@nP       �	\��Yc�A�*

loss�=+JR?       �	J�Yc�A�*

lossdܕ=�H'�       �	m��Yc�A�*

loss���=�{Y�       �	4��Yc�A�*

loss�g<Lf       �	�!�Yc�A�*

loss�4�=��       �	��Yc�A�*

loss���=���       �	6Z�Yc�A�*

loss��>�o-       �	Q��Yc�A�*

loss�c=s��F       �	m��Yc�A�*

loss���=,M�       �	='�Yc�A�*

lossӰ�;�TA       �	r��Yc�A�*

loss	~<%�p�       �	CW�Yc�A�*

loss�I=��V3       �	���Yc�A�*

loss���=���       �	q��Yc�A�*

loss��=�Ȕ        �	#,�Yc�A�*

loss�#�=���       �	���Yc�A�*

lossN(=�rQ�       �	Q��Yc�A�*

losss�\=*�2       �	��Yc�A�*

loss
�&=��I�       �	C��Yc�A�*

loss݀v< �>       �	,a�Yc�A�*

loss�4�=|Ԭ�       �	��Yc�A�*

loss�fH=�Z�       �	
��Yc�A�*

loss��I=�X�       �	�7�Yc�A�*

loss�:=�R�N       �	r��Yc�A�*

loss8F=���M       �	�~�Yc�A�*

lossR��=��tD       �	�B�Yc�A�*

loss�S=����       �	���Yc�A�*

lossN�>�2��       �	0��Yc�A�*

loss�w�<5ߧK       �	Ug�Yc�A�*

loss�}=�!��       �	��Yc�A�*

loss�"�;��G       �	��Yc�A�*

loss���=���       �	UP�Yc�A�*

loss� >BhR       �	2 �Yc�A�*

loss�7�=���       �	8��Yc�A�*

losss�l=:-y�       �	�A Yc�A�*

lossȴV=�{�]       �	5@Yc�A�*

loss�+=��l�       �	��Yc�A�*

lossxb�=�I8       �	��Yc�A�*

loss1P�=�WX       �	��Yc�A�*

loss.��=�       �	;�Yc�A�*

loss��=,p�       �	�NYc�A�*

loss�W�=[y�       �		RYc�A�*

lossD��=�T�v       �	d�Yc�A�*

loss.�<�Ge       �	Q�Yc�A�*

lossF�$=�s)       �	�&Yc�A�*

lossʌ}=��V�       �	��Yc�A�*

loss���<��!       �	Cu	Yc�A�*

loss�k>��@1       �	�
Yc�A�*

loss�_�=u�u       �	ظ
Yc�A�*

lossNF�;J���       �	�aYc�A�*

loss.�x;�rv       �	�Yc�A�*

loss�6X<��Y�       �	e�Yc�A�*

loss8��=;�k>       �	pzYc�A�*

loss[S=b��F       �	~Yc�A�*

lossO @=:��       �	M�Yc�A�*

loss=e=h�)�       �	&WYc�A�*

loss���<��       �	��Yc�A�*

loss�D=H%N       �	�Yc�A�*

loss��<�t��       �	�4Yc�A�*

loss�G{=��       �	��Yc�A�*

loss�ST=yL�V       �	��Yc�A�*

loss\=d/�       �	�$Yc�A�*

loss�G>�n�       �	��Yc�A�*

loss/��=��n�       �	�lYc�A�*

losse�=f��       �	IYc�A�*

loss�=�
�       �	��Yc�A�*

loss��+=�͜       �	kYc�A�*

loss3�a=���       �	�Yc�A�*

lossB�=>�Z�       �	6�Yc�A�*

loss���<ebw       �	MYc�A�*

loss�e�<#L��       �	�Yc�A�*

lossŘ�=���;       �	`�Yc�A�*

lossf*�<�Q�{       �	JYc�A�*

loss�=�,�)       �	:�Yc�A�*

loss��s=e��;       �	^�Yc�A�*

loss��=�_�!       �	�Yc�A�*

loss��=��A       �	�Yc�A�*

loss�1<�=�X       �	�jYc�A�*

loss%�=��Mq       �	�4Yc�A�*

lossn)=||��       �	��Yc�A�*

lossݶ�<��       �	�pYc�A�*

loss���=#�       �	� Yc�A�*

losse�<��N"       �	�� Yc�A�*

loss�,�=�C��       �	�h!Yc�A�*

loss#��=��A�       �	="Yc�A�*

lossv=�Ə       �	ö"Yc�A�*

loss܏<"f	�       �	�R#Yc�A�*

lossc��=2��       �	o�#Yc�A�*

lossѕ�=\zf�       �	��$Yc�A�*

loss�j�<���_       �	�0%Yc�A�*

loss-�=ּ��       �	�%Yc�A�*

loss�o�<h{��       �	�j&Yc�A�*

lossч>@�Qv       �	�'Yc�A�*

loss�d=�[j?       �	��'Yc�A�*

lossQ�*=[��w       �	�x(Yc�A�*

loss�\O=����       �	<K)Yc�A�*

loss�j=�-�       �	�*Yc�A�*

loss�mI=uqr�       �	�*Yc�A�*

loss��-=��       �	�?+Yc�A�*

loss��=$6��       �	��+Yc�A�*

loss��<푫�       �	�,Yc�A�*

loss��<7{�       �	՗-Yc�A�*

loss�
0>��q�       �	��.Yc�A�*

loss��y=ލ)       �	4�/Yc�A�*

loss���=j�;n       �	T�0Yc�A�*

loss��<���,       �	��1Yc�A�*

loss��=F�V[       �	 �2Yc�A�*

loss�k <9{�(       �	�94Yc�A�*

loss
��<�#�       �	��4Yc�A�*

loss}�R<��V}       �	T�5Yc�A�*

loss�ݖ<�W[       �	�T6Yc�A�*

loss;y�=��^5       �	g`7Yc�A�*

loss��<b=�_       �	��7Yc�A�*

loss��;=*m#'       �	ϟ8Yc�A�*

loss��=�43a       �	�C9Yc�A�*

loss߻=m���       �	��9Yc�A�*

lossV�3=p�D       �	~:Yc�A�*

loss)��<�Y��       �	:";Yc�A�*

loss��J=_9�       �	��;Yc�A�*

loss6�=��A�       �	�V<Yc�A�*

loss��>S�B�       �	c�<Yc�A�*

loss�@�<��#       �	�=Yc�A�*

loss��=��       �	�9>Yc�A�*

loss�5I=��&/       �	E?Yc�A�*

loss�yE=���       �	�?Yc�A�*

loss��<��}       �	�D@Yc�A�*

lossR�o=�#s       �	L�@Yc�A�*

loss�<�s�>       �	��AYc�A�*

lossx��=��l�       �	�WBYc�A�*

loss.��<@�       �	��BYc�A�*

loss��e=�@5�       �	�CYc�A�*

loss���<	 M       �	&DYc�A�*

loss��?="       �	�DYc�A�*

loss�y�<n�6       �	�MEYc�A�*

loss��<l��1       �	��EYc�A�*

loss�*=�v       �	�FYc�A�*

loss��C=$��9       �	�/GYc�A�*

loss`��=i�Q!       �	��GYc�A�*

loss=�aY�       �	�`HYc�A�*

loss3��<���       �	=IYc�A�*

loss�ʃ=��va       �	�IYc�A�*

loss�p6<'�n       �	�9JYc�A�*

lossd�=^�B       �	^�JYc�A�*

lossO=��s�       �	%�KYc�A�*

lossR�|<[^�       �	O@LYc�A�*

loss(��=J�!       �	C�LYc�A�*

loss�I=�mx       �	�xMYc�A�*

lossrX�="�q       �	�NYc�A�*

lossӭ�=_P�       �	�NYc�A�*

lossO��<qԮ�       �	-@OYc�A�*

loss���;M�
%       �	!!PYc�A�*

lossz�_=]��       �	�PYc�A�*

loss��=_W�1       �	�XQYc�A�*

lossn8=�u       �	�mRYc�A�*

loss�O�=����       �	+SYc�A�*

loss�%o<(�F}       �	��SYc�A�*

lossV�|=�1��       �	mTYc�A�*

loss��;0҇       �	�UYc�A�*

loss��x=�D�8       �	2�UYc�A�*

loss(�<1�A�       �	��VYc�A�*

loss��<Hn�       �	�?WYc�A�*

loss��=m�/�       �	JXYc�A�*

loss�V<���X       �	g�XYc�A�*

loss�2�<
+�M       �	�TYYc�A�*

lossTnC=�I�M       �	��YYc�A�*

loss�*�;�]��       �	^�ZYc�A�*

lossl�<��u�       �	�I[Yc�A�*

loss&]�=c�       �	m\Yc�A�*

loss ��;E��:       �	�\Yc�A�*

loss��*=��v+       �	�G]Yc�A�*

loss8G�=�^]J       �	�]Yc�A�*

loss�]=$�9�       �	�^Yc�A�*

loss�2?=j8y       �	0_Yc�A�*

loss�!l<�t�       �	\�_Yc�A�*

loss��<S���       �	��`Yc�A�*

loss�N<�''       �	gdaYc�A�*

lossl�@< �Jl       �	� bYc�A�*

loss?��<�	(       �	��bYc�A�*

lossax�<���Y       �	k-cYc�A�*

loss�S�<�˰       �	�cYc�A�*

loss��d=�       �	F}dYc�A�*

loss��=�w��       �	((eYc�A�*

loss��!=U��X       �	~�eYc�A�*

lossș/=���       �	"rfYc�A�*

loss��~=��,�       �	0�gYc�A�*

loss��R={�g2       �	W=hYc�A�*

loss�� =�+7I       �	�=iYc�A�*

loss��=���       �	�jYc�A�*

loss��;��#�       �	�jYc�A�*

loss$�A<\yV9       �	�lYc�A�*

loss�׼<�h+       �	-�lYc�A�*

loss���<�`D       �	}�mYc�A�*

loss��2>�V��       �	�`nYc�A�*

lossS_=� ��       �	��oYc�A�*

loss��=#q��       �	йpYc�A�*

loss�� <���       �	��qYc�A�*

loss�S�=�;�w       �	rYc�A�*

loss`�o=�\�>       �	@isYc�A�*

loss=9=�1�H       �	tYc�A�*

lossZ&)=�'�       �	��tYc�A�*

loss��=9�       �	T�uYc�A�*

loss���=⪼~       �	
�vYc�A�*

lossǙ�=5�5�       �	��wYc�A�*

loss���<o�       �	�/xYc�A�*

loss��t<�W�G       �	��xYc�A�*

loss�&=E�L[       �	'�yYc�A�*

loss��x<3��       �	&rzYc�A�*

loss��<C�5       �	6>{Yc�A�*

loss���=���       �	'�{Yc�A�*

loss�J�=��u       �	~�|Yc�A�*

loss;�n=_�       �	-}Yc�A�*

loss3�=s�       �	;�}Yc�A�*

loss�Ͼ=��y        �	�d~Yc�A�*

loss�?=p��?       �	�Yc�A�*

lossF�;�^       �	[�Yc�A�*

lossF�<`���       �	=��Yc�A�*

loss�8=I�[       �	�1�Yc�A�*

loss��>`�>       �	�ɁYc�A�*

loss��=&�=�       �	�a�Yc�A�*

loss��=���8       �	O�Yc�A�*

loss�H<p�u       �	���Yc�A�*

loss(�Q=��#       �	�G�Yc�A�*

loss�M =��G�       �	k�Yc�A�*

loss�8=����       �	ݕ�Yc�A�*

loss��q<z~�       �	}A�Yc�A�*

loss�:=��o�       �	݆Yc�A�*

lossC��=C+Dt       �	_|�Yc�A�*

loss�=Ê�       �	�1�Yc�A�*

loss;�=���f       �	�͈Yc�A�*

loss#~�=��^�       �	{f�Yc�A�*

loss���=�2J�       �	��Yc�A�*

loss4��=T�5+       �		��Yc�A�*

loss�	'>+UÁ       �	�:�Yc�A�*

loss�N�<F���       �	+ۋYc�A�*

lossc��<ʽ��       �	�y�Yc�A�*

loss�>2�6       �	��Yc�A�*

loss��=�i
0       �	V��Yc�A�*

losss�U=�QI~       �	dX�Yc�A�*

loss�[�=�@       �	'��Yc�A�*

loss|#:=zy�y       �	!��Yc�A�*

loss�p�=uqKW       �	40�Yc�A�*

loss{��<.�mC       �	�ҐYc�A�*

loss �o=��^       �	��Yc�A�*

lossR�	=GQH=       �	2�Yc�A�*

lossa�=*EZ0       �	d̒Yc�A�*

loss2��=���       �	[{�Yc�A�*

loss*"�=Z�D_       �	>&�Yc�A�*

loss!;*�1       �	�ΔYc�A�*

loss,UO;�� @       �	�o�Yc�A�*

loss
W=Dm�Y       �	w�Yc�A�*

loss�w�< h�       �	K��Yc�A�*

loss�:�=�=>       �	>\�Yc�A�*

loss|�"=�?[k       �	��Yc�A�*

loss��=h̛�       �	u��Yc�A�*

loss��<H~޾       �	z9�Yc�A�*

loss`"=X��       �	JљYc�A�*

loss�V>�C�       �	nk�Yc�A�*

loss���=4���       �	O�Yc�A�*

loss�z�=m��}       �	s��Yc�A�*

loss�4=����       �	&8�Yc�A�*

lossEs�=���G       �	�ݜYc�A�*

lossE��=��i       �	�v�Yc�A�*

lossD >Y�T�       �	0�Yc�A�*

lossZ�Y=���?       �	&��Yc�A�*

lossLؙ=���J       �	PQ�Yc�A�*

losss�-=X�r~       �	��Yc�A�*

lossa��<,ǐ�       �	Ք�Yc�A�*

loss�<=\G�       �	�@�Yc�A�*

loss1�=ٯ��       �	l�Yc�A�*

losst�>\d�        �	Yc�A�*

loss���=�.�        �	~9�Yc�A�*

loss���<�bg�       �	��Yc�A�*

lossx��=x�;       �	Yc�A�*

loss���<*�Ù       �	���Yc�A�*

loss�'C<m�\Q       �	�=�Yc�A�*

loss7!k<d��\       �	`�Yc�A�*

loss��<sރ)       �	��Yc�A�*

loss���<&�E       �	�ĨYc�A�*

lossr�*=ZT;A       �	���Yc�A�*

loss_3�=�/�       �	#J�Yc�A�*

loss�L�=__��       �	!�Yc�A�*

loss5�=��Bv       �	��Yc�A�*

lossO-*<O��       �	㦬Yc�A�*

loss��;��4       �	+��Yc�A�*

loss��q<"]�       �	���Yc�A�*

loss%R�={�sx       �	~Q�Yc�A�*

loss�<�;��       �	[��Yc�A�*

loss�ɥ;��P0       �	�d�Yc�A�*

loss��;=�A�~       �	&��Yc�A�*

loss$�=�D�       �	s��Yc�A�*

lossM�<���       �	�6�Yc�A�*

lossC-�=�u�<       �	�H�Yc�A�*

loss�>"W       �	�Yc�A�*

loss71�=��8�       �	�~�Yc�A�*

loss���<t        �	\�Yc�A�*

loss��=�/Z�       �	�ɶYc�A�*

lossYz =��J�       �	6u�Yc�A�*

loss��=k'x�       �	�L�Yc�A�*

losso;�<:8l�       �	^��Yc�A�*

loss�&�=I�       �	���Yc�A�*

loss݇�;��       �	�O�Yc�A�*

loss��=o��       �	���Yc�A�*

loss��=�:��       �	��Yc�A�*

lossQ'�=~7y�       �	$D�Yc�A�*

loss�=!���       �	�Yc�A�*

loss�xt=�ȐX       �	�!�Yc�A�*

lossmy�=̇{}       �	C˾Yc�A�*

loss_,�<�Aۆ       �	�n�Yc�A�*

lossZ�S=��b�       �	��Yc�A�*

loss�D-=��D       �	���Yc�A�*

loss�=�}       �	��Yc�A�*

loss���=8PJ       �	�8�Yc�A�*

loss�*>9"J�       �	c��Yc�A�*

loss2C=���o       �	U��Yc�A�*

lossDQ=���       �	w��Yc�A�*

loss�h=����       �	�L�Yc�A�*

loss35r=�pW        �	���Yc�A�*

loss;��=tW-       �	��Yc�A�*

loss��=���       �	Ae�Yc�A�*

loss.��=_̈́�       �	7o�Yc�A�*

loss�w=u��n       �	�Yc�A�*

loss�X>=�Wɨ       �	9��Yc�A�*

loss/w`=��b�       �	���Yc�A�*

loss��=v�(�       �	32�Yc�A�*

loss��=E0��       �	y��Yc�A�*

loss��<�$��       �	�o�Yc�A�*

loss�b�<p       �	�Yc�A�*

loss�N�=n�.�       �	���Yc�A�*

lossR�>+�}       �	�X�Yc�A�*

loss��=��'       �	`�Yc�A�*

loss�L*=�6�       �	���Yc�A�*

loss�@=W���       �	���Yc�A�*

lossl�q=�[�       �	)�Yc�A�*

lossW�,<a�G�       �	g��Yc�A�*

lossz_=�4dO       �	���Yc�A�*

loss?X�<�$�?       �	u<�Yc�A�*

loss� =#d �       �	��Yc�A�*

loss�KW=|��       �	r��Yc�A�*

loss��2>�.�       �	�3�Yc�A�*

loss�/V=�,��       �	���Yc�A�*

loss�]�<�z�(       �	�u�Yc�A�*

lossp�=��       �	�B�Yc�A�*

loss�؄=����       �	���Yc�A�*

lossl�=k���       �	؃�Yc�A�*

loss,G�<_s�5       �	�!�Yc�A�*

loss���<�-�       �	��Yc�A�*

loss$=O\�       �	�W�Yc�A�*

lossmR�=r�?       �	��Yc�A�*

loss&��=&B4�       �	��Yc�A�*

lossTIz<+J0       �	�-�Yc�A�*

loss��<1���       �	��Yc�A�*

loss��<���       �	��Yc�A�*

loss��<r��0       �	�B�Yc�A�*

loss��)=�|u       �	���Yc�A�*

loss�l=�T�Y       �	Jy�Yc�A�*

loss#�=��L�       �	q!�Yc�A�*

loss��=��m]       �	��Yc�A�*

loss�%�=~��3       �	�O�Yc�A�*

loss<ӎ=�Krh       �	���Yc�A�*

loss�|�=án�       �	<��Yc�A�*

loss��=���       �	���Yc�A�*

losso�<�M>,       �	&n�Yc�A�*

loss���<�3�f       �	���Yc�A�*

losswA=�       �	���Yc�A�*

lossh,�<`֊r       �	#1�Yc�A�*

lossw�=��|�       �	d��Yc�A�*

loss3�<��\�       �	 `�Yc�A�*

lossvH�<aD       �	���Yc�A�*

loss/z=�       �	͑�Yc�A�*

loss�h�=��h       �	?�Yc�A�*

loss�w�=��       �	
��Yc�A�*

lossH8c=Uq��       �	Cq�Yc�A�*

lossx�
=U�	�       �	V-�Yc�A�*

loss{��=.`:|       �	���Yc�A�*

loss���=��X       �	�t�Yc�A�*

loss4�<��m       �	�Yc�A�*

loss	>I=P�c       �	N��Yc�A�*

loss65[<P�^E       �	�Yc�A�*

loss}i�<��v       �	�,�Yc�A�*

loss��b=#�a       �	&��Yc�A�*

loss�8=�8�       �	�_�Yc�A�*

loss{\�=��$       �	��Yc�A�*

loss��F<KDny       �	ę�Yc�A�*

lossN��<��)       �	�1�Yc�A�*

loss�!�=l�y       �	���Yc�A�*

loss���<�L�5       �	�b�Yc�A�*

loss��#=O�r       �	��Yc�A�*

lossM`=>62       �	��Yc�A�*

lossCF7=av
�       �	��Yc�A�*

loss�l>��[I       �	���Yc�A�*

loss�0�;RS�s       �	RC�Yc�A�*

loss���<�<�[       �	���Yc�A�*

loss;�B<f��       �	�q�Yc�A�*

lossQP=fC\       �	�
�Yc�A�*

lossP�=���%       �	��Yc�A�*

lossA!=��z�       �	�H�Yc�A�*

lossHX=��       �	o��Yc�A�*

loss슲<��}g       �	e��Yc�A�*

lossM��=A�B�       �	w1 Yc�A�*

loss��m=m�@�       �	�� Yc�A�*

loss�
�=�+?�       �	bhYc�A�*

loss��4=i�%       �	�Yc�A�*

loss7$�==VP�       �	��Yc�A�*

loss���=�gO       �	e�Yc�A�*

loss2`=��m       �	$'Yc�A�*

loss v�=���       �	��Yc�A�*

loss!d=݌�;       �	 ^Yc�A�*

loss[�=ۙo�       �	�Yc�A�*

loss�m�=��{       �	:�Yc�A�*

loss��x=� �       �	�;Yc�A�*

loss�;N�h       �	��Yc�A�*

loss|��<�as�       �	�|Yc�A�*

loss�p=�X+�       �	N+	Yc�A�*

loss�@�<�ȃ]       �	��	Yc�A�*

loss��<�       �	�a
Yc�A�*

loss-v=�d�       �	�GYc�A�*

loss���;�H       �	�SYc�A�*

loss_|�=�C       �	 	Yc�A�*

losssr�<�A"       �	��Yc�A�*

loss��>Rq/�       �	�FYc�A�*

loss�+�<-\��       �	��Yc�A�*

loss,�=
�b�       �	��Yc�A�*

loss�j�<k�i       �	P:Yc�A�*

loss��<��       �	��Yc�A�*

loss�d=����       �	nYc�A�*

loss��o=g�AY       �	Yc�A�*

loss�O=)���       �	�Yc�A�*

loss�G=_�J�       �	�bYc�A�*

loss�1v=,���       �	�JYc�A�*

loss(�=�`�       �	�Yc�A�*

loss��g=��l�       �	��Yc�A�*

loss*'>=��{�       �	w+Yc�A�*

lossM�.=����       �	G�Yc�A�*

loss*E =�%�o       �	mYc�A�*

losst��=5�$T       �	GYc�A�*

loss���=�պ       �	��Yc�A�*

loss��%=�_h       �	u:Yc�A�*

loss8T,=�t�       �	��Yc�A�*

losse��;��       �	�uYc�A�*

loss�e�=�+�       �	Yc�A�*

loss��=1�       �	�Yc�A�*

loss���<}��y       �	�XYc�A�*

loss}��<�Y(�       �	��Yc�A�*

loss�=�T=�       �	��Yc�A�*

loss��=�x��       �	S"Yc�A�*

loss$�<y��       �	��Yc�A�*

loss,N�<�        �	�qYc�A�*

loss�Ht=u���       �	� Yc�A�*

loss���=�j;�       �	J� Yc�A�*

loss���=���       �	&V!Yc�A�*

loss뜂=^F�       �	�"Yc�A�*

loss��=U�0       �	:�"Yc�A�*

loss:��=���       �	�T#Yc�A�*

lossMΛ=�n�       �	��#Yc�A�*

loss��<W���       �	ׅ$Yc�A�*

loss��N<G}�       �	� %Yc�A�*

loss?�	=��R       �	ܹ%Yc�A�*

lossγ=�g�.       �	"S&Yc�A�*

loss1x=^�Л       �	��&Yc�A�*

loss/�=|�       �	ˁ'Yc�A�*

lossV=d�J�       �	�"(Yc�A�*

loss �=�n�       �	��(Yc�A�*

loss�h=<x�       �	��)Yc�A�*

loss�%%<C�[�       �	=�*Yc�A�*

loss=�=���F       �	xE+Yc�A�*

loss�4=N<z       �	,Yc�A�*

lossi�=�P       �	F-Yc�A�*

losst��=&#�       �	G.Yc�A�*

loss]��<��P       �	��.Yc�A�*

loss<�<W�       �	!�/Yc�A�*

loss ��<��ڡ       �	�:0Yc�A�*

loss��p=i       �	]p1Yc�A�*

lossA^%=��l
       �	� 2Yc�A�*

loss�>[a       �	��2Yc�A�*

loss�Ǉ=}�0       �	*r3Yc�A�*

loss4A=�Ư       �	l4Yc�A�*

loss��s=<���       �	�\5Yc�A�*

lossD�}=��:Z       �	9%6Yc�A�*

loss-l�<�P=�       �	��6Yc�A�*

loss�Q"=��w�       �	ڏ7Yc�A�*

lossQY&<��~{       �	�=8Yc�A�*

lossŅ?=;�*�       �	��8Yc�A�*

loss�?=�rXp       �	�~9Yc�A�*

lossr_=3�       �	U2:Yc�A�*

loss�i�<��b       �	��:Yc�A�*

loss�K�=vX�       �	m;Yc�A�*

lossa��;���       �	�<Yc�A�*

lossR��<��       �	i�<Yc�A�*

loss
�;XN�       �	H=Yc�A�*

lossC��<TBA       �	��=Yc�A�*

lossM�+=2���       �	w�>Yc�A�*

loss�*=��       �	p#?Yc�A�*

loss��<0�	�       �	�?Yc�A�*

loss��>~n��       �	LS@Yc�A�*

lossm��=_��Y       �	-�@Yc�A�*

lossxՆ<$�       �	+�AYc�A�*

loss�Y�<��e�       �	taBYc�A�*

loss�=�`v�       �	�CYc�A�*

lossI \=�I�       �	�CYc�A�*

loss���<��(       �	@2DYc�A�*

loss&�>=	|�d       �	��DYc�A�*

loss(�<�w,�       �	�nEYc�A�*

loss��h=sw       �	�FYc�A�*

loss-�i= �ܠ       �	�FYc�A�*

loss��<`�$       �	�NGYc�A�*

lossl�d=}��       �	��GYc�A�*

loss<];<ո�~       �	�~HYc�A�*

loss���=Cڤ�       �	�IYc�A�*

loss]��<����       �	��IYc�A�*

loss���=��y�       �	kEJYc�A�*

loss��:=*�@%       �	T�JYc�A�*

lossV40=?��e       �	k}KYc�A�*

loss[~'=kr�h       �	uLYc�A�*

loss$p�=�/��       �	̶LYc�A�*

loss���<�o�H       �	�SMYc�A�*

loss��'=꟔O       �	�MYc�A�*

lossJ�t<d���       �	ŌNYc�A�*

losso\=m       �	�,OYc�A�*

loss&��<x�,2       �	��OYc�A�*

loss���<��|�       �	�ZPYc�A�*

loss=5�<��+       �	��PYc�A�*

losst؆<���       �	�QYc�A�*

loss��=c�ג       �	9%RYc�A�*

loss�Yf<g��       �	��RYc�A�*

loss���<��lp       �	;SSYc�A�*

loss�@=۸ZC       �	��SYc�A�*

loss8�<�u��       �	c~TYc�A�*

loss���<T�       �	EUYc�A�*

loss��<%�E       �	z�UYc�A�*

loss!�<Bg��       �	JVYc�A�*

loss��G<2�
�       �	\�VYc�A�*

lossVg<M��x       �	�WYc�A�*

loss{'�;��n�       �	�+XYc�A�*

loss;;=?\�       �	a�XYc�A�*

loss��]<QE�6       �	�VYYc�A�*

losssɗ:AW�       �	��YYc�A�*

loss6��:���       �	ˡZYc�A�*

loss��<zyt�       �	�7[Yc�A�*

loss�VB=oᘁ       �	Q�[Yc�A�*

loss/�=F�W�       �	}z\Yc�A�*

loss;y�6�       �	�)]Yc�A�*

loss#n(=��Ej       �	8�]Yc�A�*

losso��>��I       �	��^Yc�A�*

lossi�:<! g�       �	�S_Yc�A�*

loss��:<���       �	�_Yc�A�*

loss�>==�       �	�`Yc�A�*

lossa�=8�7�       �	�6aYc�A�*

loss�NB=�b�       �	�aYc�A�*

loss�=�<=�	       �	tzbYc�A�*

loss�j=��'       �	PPcYc�A�*

lossI"=U�`Z       �	��cYc�A�*

losss��=\	<        �	X�dYc�A�*

loss��>�'�       �	�@eYc�A�*

loss�b=�Ec       �	��eYc�A�*

loss
�[=����       �	=~fYc�A�*

lossH��=�.8/       �	�gYc�A�*

loss��?=�uq�       �	R�gYc�A�*

lossD��=�_�       �	�uhYc�A�*

loss[��=���#       �	iYc�A�*

lossE�`=F��3       �	�jYc�A�*

loss	*="�w=       �	^�jYc�A�*

loss��;=k2&9       �	!YkYc�A�*

loss���<b7��       �	w�kYc�A�*

lossJ)�<}/Ib       �	ٙlYc�A�*

lossWw<=b���       �	�:mYc�A�*

loss=߷<*�v       �	��mYc�A�*

loss#�<onF�       �	fknYc�A�*

loss<�;˙�       �	�oYc�A�*

lossX=��E�       �	�oYc�A�*

lossqL$=���       �	8pYc�A�*

lossX��<��u�       �	1�pYc�A�*

loss�B>zCVZ       �	I�qYc�A�*

lossG�=|��^       �	�qrYc�A�*

lossC)�<#���       �	�sYc�A�*

lossj��=����       �	/�sYc�A�*

loss��=<���       �	!?tYc�A�*

loss�$�;�Y�       �	��tYc�A�*

loss��3=Iuf�       �	y�uYc�A�*

loss!'d=�-W       �	jOvYc�A�*

loss��v<�G        �	��vYc�A�*

loss̅�<
w��       �	�|wYc�A�*

loss��=DϦ       �	�xYc�A�*

loss�}�=&�+-       �	K�xYc�A�*

loss�\�<T�e       �	@�yYc�A�*

loss�3=Ul�:       �	zYc�A�*

lossJ��<��%       �	��zYc�A�*

loss	m�<���       �	�C{Yc�A�*

loss�>E=�-�       �	h�{Yc�A�*

loss�!==:�}�       �	J~|Yc�A�*

loss	�=��,       �	}Yc�A�*

loss�ٙ<	Y�X       �	��}Yc�A�*

lossO�}=ap��       �	�K~Yc�A�*

loss��<Wu�l       �	}�~Yc�A�*

lossC[={_-�       �	f�Yc�A�*

loss��<H=�       �	��Yc�A�*

loss��B=���-       �	+��Yc�A�*

loss��=	¥�       �	B#�Yc�A�*

loss8��<�.       �	��Yc�A�*

loss@�Y=�gk       �	Va�Yc�A�*

lossd^=o5M�       �	�+�Yc�A�*

loss�'%=�L8       �	�ȜYc�A�*

loss�M]<Z�T�       �	Nb�Yc�A�*

loss��=� E�       �	J
�Yc�A�*

loss��D=θ�6       �	&��Yc�A�*

loss���<�q�;       �	B?�Yc�A�*

loss��=I��       �	��Yc�A�*

lossh$�<��!q       �	���Yc�A�*

loss�|=�� �       �	 R�Yc�A�*

loss.-,=��       �	B�Yc�A�*

loss� h=3��       �	��Yc�A�*

loss��<��E�       �	O�Yc�A�*

lossWH<_:       �	5��Yc�A�*

loss�p�<zZc       �	�Q�Yc�A�*

loss��>#�=�       �	-�Yc�A�*

loss�Q=��/u       �	c��Yc�A�*

loss�R�<�j"       �	7�Yc�A�*

lossVs�<�)r       �	��Yc�A�*

loss�	=�mPm       �	kd�Yc�A�*

loss���<m��w       �	.Y�Yc�A�*

loss��<��-       �	e6�Yc�A�*

lossLQ=�PC�       �	�|�Yc�A�*

loss��<�s]�       �	@�Yc�A�*

loss���<�;L       �	ѱ�Yc�A�*

loss}e�<\�z�       �	�H�Yc�A�*

loss���=yӑ1       �	��Yc�A�*

loss$�(=�l4�       �	"Q�Yc�A�*

loss|�<O)��       �	���Yc�A�*

loss}O�=��K       �	㧯Yc�A�*

lossj��<��.�       �	"R�Yc�A�*

loss �=.��'       �	�Yc�A�*

loss�mM<��n       �	���Yc�A�*

loss(�=©�H       �	�V�Yc�A�*

loss>�>-��1       �	H��Yc�A�*

loss�>�#L       �	��Yc�A�*

loss��<	���       �	&W�Yc�A�*

loss���=��أ       �	���Yc�A�*

loss��<b�<`       �	>͵Yc�A�*

loss�p=����       �	#i�Yc�A�*

lossz�=��c�       �	�Yc�A�*

lossJ+>
Ύ       �	÷Yc�A�*

loss/��<���|       �	-Z�Yc�A�*

loss]?�<�/       �	��Yc�A�*

lossl!=*�k       �	:��Yc�A�*

lossC�;�F       �	I+�Yc�A�*

loss���<�tH�       �	�̺Yc�A�*

lossU��<��j       �	�e�Yc�A�*

lossY��<|��       �	B�Yc�A�*

loss�g>�x=       �	���Yc�A�*

lossd�=/q�W       �	�D�Yc�A�*

loss�]<x��       �	᷾Yc�A�*

lossO';,^M)       �	!v�Yc�A�*

loss���<>���       �	���Yc�A�*

loss��|<o�c/       �	�h�Yc�A�*

loss�Q�<�ٷ)       �	J�Yc�A�*

loss�K#=T��       �	Ý�Yc�A�*

loss�
=��c�       �	�4�Yc�A�*

loss�I�<3DE�       �	L�Yc�A�*

loss	�n=��k       �	���Yc�A�*

lossz<*��       �	�^�Yc�A�*

loss)#�<�uz       �	h�Yc�A�*

lossv��<�l=(       �	�S�Yc�A�*

lossHd=V��       �	���Yc�A�*

loss�L�=`x��       �	���Yc�A�*

loss��6>�Wk       �	?�Yc�A�*

loss��=c���       �	���Yc�A�*

loss�J=m�R       �	��Yc�A�*

loss6�g=vp�       �	0�Yc�A�*

loss�Z==V.�       �	���Yc�A�*

loss̍�<�T��       �	0��Yc�A�*

lossJE�<_,A6       �	���Yc�A�*

loss�)�<���       �	�n�Yc�A�*

loss��O=O���       �	W	�Yc�A�*

loss��<�Y�       �	¡�Yc�A�*

losso,�=`K4       �	#J�Yc�A�*

lossS��<��?�       �	\��Yc�A�*

loss�&=]��b       �	�|�Yc�A�*

loss�Ō=��3       �	'�Yc�A�*

loss���<Q�>�       �	���Yc�A�*

losss��<B��       �	�Y�Yc�A�*

loss�e"=�y�V       �	���Yc�A�*

loss��=)��       �	���Yc�A�*

loss5)�=!r�K       �	c&�Yc�A�*

loss�C�<mX�	       �	���Yc�A�*

loss$̽< �Wi       �	��Yc�A�*

loss�0=p,*�       �	�s�Yc�A�*

loss�X=��v�       �	��Yc�A�*

loss�z�<��v�       �	ƾ�Yc�A�*

lossn��=�\m~       �	�f�Yc�A�*

loss_7j=�x8�       �	��Yc�A�*

loss���<A��(       �	���Yc�A�*

loss�vO=M��X       �	�B�Yc�A�*

loss�e=]�	       �	<��Yc�A�*

loss{w�=�rG       �	�u�Yc�A�*

losst��<��
       �	t�Yc�A�*

loss\�;XfJ       �	~��Yc�A�*

loss��;�'�c       �	9F�Yc�A�*

loss���<HƥS       �	���Yc�A�*

loss.$ <,L�h       �	���Yc�A�*

loss%��<�p�       �	
.�Yc�A�*

lossV��<}ͮw       �	L��Yc�A�*

loss4
k<���z       �	p_�Yc�A�*

losss+x<'��       �	���Yc�A�*

lossr�m=���       �	��Yc�A�*

loss��<�\<       �	�9�Yc�A�*

loss��=f�H       �	f0�Yc�A�*

loss��=K�F       �	���Yc�A�*

loss�pP=!�       �	�z�Yc�A�*

loss��P<fp�F       �	��Yc�A�*

lossO(7=h�l�       �	��Yc�A�*

loss̡w=���       �	�l�Yc�A�*

lossfD>W��       �	�Yc�A�*

loss11�=1��       �	T��Yc�A�*

loss���=q,��       �	o�Yc�A�*

loss)t=7Y�~       �	Q�Yc�A�*

loss��=�Y3�       �	U��Yc�A�*

loss�I=����       �	 ��Yc�A�*

loss��=3MY�       �	�H�Yc�A�*

loss��~=�s�       �	>��Yc�A�*

lossA�<�       �	��Yc�A�*

losss�9=T��       �	I0�Yc�A�*

loss�m�<Q�S       �	N��Yc�A�*

loss�V<Q l�       �	�q�Yc�A�*

lossw��=|{�       �	Y�Yc�A�*

loss�ŝ=�f}�       �	x��Yc�A�*

loss��<�"�       �	~Q�Yc�A�*

loss��=s��'       �	���Yc�A�*

loss��=��_�       �	"��Yc�A�*

lossdD�<�pP�       �	��Yc�A�*

loss;�=-f��       �	A��Yc�A�*

lossn�=ݯ��       �	SY�Yc�A�*

loss}�'=~���       �	���Yc�A�*

loss��Y<��;       �	��Yc�A�*

lossav�<&v3]       �	+0�Yc�A�*

loss�Ǯ<�d�       �	O�Yc�A�*

loss���<���       �	�	�Yc�A�*

loss=G�       �	8��Yc�A�*

loss��=e�0       �	;�Yc�A�*

loss�^�=�H�       �	P��Yc�A�*

loss!H>���       �	�6�Yc�A�*

loss==7��       �	>�Yc�A�*

loss�~=ԃ��       �	�B�Yc�A�*

loss1-�;�t~       �	���Yc�A�*

loss�5<�SLy       �	؂�Yc�A�*

loss�=3�[4       �	E,�Yc�A�*

loss�	d=� ��       �	���Yc�A�*

loss.��=V��\       �	Me Yc�A�*

loss�7~=��g       �	�Yc�A�*

loss��.=Mu�       �	5�Yc�A�*

loss��6=����       �	��Yc�A�*

loss��<]���       �	�:Yc�A�*

loss���<�VJ       �	.�Yc�A�*

loss3,�=(��%       �	��Yc�A�*

lossO*�=�w>"       �	:yYc�A�*

loss�K�<`wiA       �	�Yc�A�*

losst]=�>�       �	�VYc�A�*

lossX�@=�2�~       �	[	Yc�A�*

loss&�V=�L
(       �	�+
Yc�A�*

loss3e�;N3i�       �	�:Yc�A�*

lossA��=���_       �	.�Yc�A�*

loss.)=���>       �	�Yc�A�*

loss��2<�Ք�       �	�@Yc�A�*

loss.�=:��       �	b�Yc�A�*

loss��<�>0�       �	�qYc�A�*

loss���<�Hn       �	rYc�A�*

loss/(=�J&       �	E�Yc�A�*

lossAz<2�5�       �	�ZYc�A�*

lossw=����       �	nYc�A�*

loss�V#= CO�       �	�Yc�A�*

lossC;��VL       �	�HYc�A�*

loss���<w�O�       �	��Yc�A�*

loss�>b=/��~       �	�	Yc�A�*

loss=�Y�       �	6�Yc�A�*

loss�g=��5       �	@KYc�A�*

loss��<��-v       �	�Yc�A�*

loss�ƈ;j?       �	O�Yc�A�*

loss�t&=,�?*       �	�WYc�A�*

loss��S=�#��       �	�@Yc�A�*

loss��<0��       �	h�Yc�A�*

loss��k<��]       �	�Yc�A�*

loss�`�;�dac       �	�4Yc�A�*

loss��Z=p��9       �	��Yc�A�*

loss�~�<�?1�       �	4�Yc�A�*

loss&0<=���
       �	�#Yc�A�*

loss�m	=עI       �	иYc�A�*

loss��<Os��       �	�SYc�A�*

loss m�<���       �	��Yc�A�*

lossF�<�WG)       �	G�Yc�A�*

loss���<d��       �		3 Yc�A�*

loss�,3<4 "&       �	k!Yc�A�*

loss�ԃ<<�_�       �	ҧ!Yc�A�*

loss�J=����       �	A"Yc�A�*

loss�;�=u��       �	�"Yc�A�*

loss*�8=�'�:       �	�|#Yc�A�*

loss��,<P�/       �	�$Yc�A�*

lossn��<u�Ȥ       �	Re%Yc�A�*

loss|"=���       �	G&Yc�A�*

loss82=��Ʋ       �	U�&Yc�A�*

lossk�<gs<       �	�:'Yc�A�*

lossq!G<o>om       �	��'Yc�A�*

loss3�P=¢B�       �	m(Yc�A�*

loss�׆=$?��       �	~)Yc�A�*

loss�+�=-�)^       �	T�)Yc�A�*

loss&	=S.       �	�x*Yc�A�*

lossFu�=���       �	��+Yc�A�*

lossŋ�<��n�       �	G;,Yc�A�*

loss�_�<[]Ԕ       �	Y�,Yc�A�*

loss��@=�M=       �	C�-Yc�A�*

loss�f1<�`�       �	t�.Yc�A�*

loss���=��F       �	�z/Yc�A�*

loss���=����       �	�0Yc�A�*

lossXr}=�|�Q       �	B&1Yc�A�*

loss�,H>�&       �	3�1Yc�A�*

loss���=4���       �	S�2Yc�A�*

loss��'=tp�       �	K3Yc�A�*

loss�=8�r�       �	�4Yc�A�*

loss78M=0��       �	�5Yc�A�*

loss�=� 
�       �	}�5Yc�A�*

loss�z=G��       �	�6Yc�A�*

lossP�<L9�u       �	�>7Yc�A�*

loss�dj=S�g       �	8Yc�A�*

loss(��=�)�U       �	��8Yc�A�*

loss��P=L�)5       �	BZ9Yc�A�*

lossh�=ə��       �	��9Yc�A�*

loss�8<#v�k       �	�:Yc�A�*

lossUd�<7:��       �	�,;Yc�A�*

loss<��<���       �	�O<Yc�A�*

loss��[=�6�       �	��<Yc�A�*

lossV�=w)C�       �	��=Yc�A�*

loss�1�=1U��       �	� >Yc�A�*

loss�*�<���       �	��>Yc�A�*

lossMy=��d       �	Z?Yc�A�*

loss���=}-߄       �	D�?Yc�A�*

lossq)�=�n)       �	2�@Yc�A�*

loss�9O<ֱ�       �	&oAYc�A�*

loss�0�<k.GW       �	$BYc�A�*

loss�ZO==��       �	W�BYc�A�*

lossqu�<w�B�       �	TCYc�A�*

loss�9%=c+       �	zVDYc�A�*

lossS�H=r)w�       �	k�DYc�A�*

lossh5�<��*N       �	k�EYc�A�*

loss��|<'M5W       �	H7FYc�A�*

loss���<����       �	��FYc�A�*

loss��<�L�&       �	�wGYc�A�*

loss���<ʀ�#       �	uHYc�A�*

lossRΉ=)��       �	̸HYc�A�*

lossD�=V|�       �	��IYc�A�*

loss��<�1��       �	�JYc�A�*

lossf<���       �	�3KYc�A�*

loss���;�.�       �	=�KYc�A�*

loss���<���       �	bfLYc�A�*

lossx�+<t�\       �	�MYc�A�*

lossO�<@�       �	��MYc�A�*

lossN>z<;�y=       �	l?NYc�A�*

lossvBC=%��       �	��NYc�A�*

lossd�=? �       �	��OYc�A�*

loss�p�<=X��       �	/PYc�A�*

loss�Ԛ= ���       �	�PYc�A�*

loss��=���       �	F_QYc�A�*

loss�{8=�{�       �	#RYc�A�*

loss"< M��       �	�RYc�A�*

loss��=��P       �	�PSYc�A�*

loss�Ǯ<1���       �	p�SYc�A�*

loss��n=��       �	B�TYc�A�*

loss��<Jb�       �	+MUYc�A�*

loss)I2=x��       �	�UYc�A�*

loss���<R�&4       �	R�VYc�A�*

loss3�x=~~G       �	f3WYc�A�*

lossjA=��W�       �	G�WYc�A�*

loss��z=V=��       �	j�XYc�A�*

loss@��=�bv�       �	�YYc�A�*

lossʴ�<ޚ`       �	^�YYc�A�*

loss���<�2!       �	�SZYc�A�*

loss���=�       �	�[Yc�A�*

loss쳗=Еc       �	1�[Yc�A�*

loss��<f
�       �	�L\Yc�A�*

loss�I[<�M�D       �	��\Yc�A�*

loss��<����       �	�]Yc�A�*

loss%&�=�Zj�       �	�^Yc�A�*

loss��==l��       �	��^Yc�A�*

loss6�6=s�S�       �	�n_Yc�A�*

loss�<A=ꖾ�       �	�`Yc�A�*

loss��3=���S       �	��`Yc�A�*

loss�x�<�z{�       �	k�aYc�A�*

lossd��<��       �	�#bYc�A�*

loss���:�[�2       �	��bYc�A�*

loss�"=Ol�       �	gcYc�A�*

lossMw><t�p       �	�
dYc�A�*

lossl�<��fo       �	�dYc�A�*

loss��<<��S       �	�NeYc�A�*

loss�qP=I��=       �	�eYc�A�*

loss ��<L�}8       �	w�gYc�A�*

loss�u=��Oq       �	�9hYc�A�*

loss�ӄ=��E       �	R�hYc�A�*

loss�=����       �	kiYc�A�*

loss<rs=�bUI       �	�jYc�A�*

loss'=�,�       �	Y�jYc�A�*

loss���<kIu       �	�?kYc�A�*

loss��=���U       �	��kYc�A�*

lossx}�=:&(�       �	nllYc�A�*

loss,��=^Ê�       �	SmYc�A�*

loss���;��#r       �	�mYc�A�*

losst^.=���       �	�RnYc�A�*

loss��0<r��k       �	��nYc�A�*

loss�o=\4�M       �	ƇoYc�A�*

loss�*�<�.mE       �	�"pYc�A�*

loss���=���       �	��pYc�A�*

lossM��=T���       �	"�qYc�A�*

lossE�5=���       �	�BrYc�A�*

lossS5=d)�       �	�rYc�A�*

lossdǈ=��)"       �	g�sYc�A�*

loss���<�� N       �	�tYc�A�*

loss��<�`'       �	��tYc�A�*

lossM�B=T_��       �	�XuYc�A�*

loss� =� m�       �	HvYc�A�*

loss���<Ěr       �	úvYc�A�*

lossʏ�=�Lv�       �	R_wYc�A�*

loss��t=��:       �	I�wYc�A�*

lossf%�<9�N�       �	ēxYc�A�*

loss$р=D���       �	Y4yYc�A�*

losshC�=�*tJ       �	(�yYc�A�*

lossKk�<ӟh^       �	�rzYc�A�*

lossSB�<���X       �	��{Yc�A�*

loss �6=j~V|       �	�S|Yc�A�*

lossL�j=/[�       �	E�|Yc�A�*

lossS�=З<�       �	G�}Yc�A�*

loss�F�<ȟ�       �	)%~Yc�A�*

loss�
=6��
       �	A�~Yc�A�*

loss��h=��N�       �	+QYc�A�*

lossN&>�{9+       �	���Yc�A�*

loss
!A=�{��       �	;7�Yc�A�*

loss ]�<��       �	��Yc�A�*

loss;�==�Y�g       �	;��Yc�A�*

lossI�<'hM�       �	G�Yc�A�*

loss�'<��H�       �	D��Yc�A�*

loss�.�;�p�:       �	���Yc�A�*

loss�S/=��c?       �	@�Yc�A�*

loss]tw=�R�       �	څYc�A�*

lossπ�<k`�       �	���Yc�A�*

loss�m�=H��%       �	l&�Yc�A�*

loss��=��J       �	�ɇYc�A�*

loss�-�<&�       �	o�Yc�A�*

loss���=)c�v       �	(C�Yc�A�*

lossJS�<x�Q|       �	�Yc�A�*

loss8s�<Y���       �	I��Yc�A�*

lossjC�;���       �	�d�Yc�A�*

loss���;�X[j       �	p�Yc�A�*

loss`J�<�+�       �	-ΌYc�A�*

loss}d<����       �	8i�Yc�A�*

loss`�L=0��y       �	�	�Yc�A�*

loss�U�;�       �	�ƎYc�A�*

lossMڋ<���       �	/m�Yc�A�*

lossq�=}|��       �	g�Yc�A�*

losso� =���R       �	ȵ�Yc�A�*

loss |.=��u�       �	�]�Yc�A�*

loss�=W�       �	�9�Yc�A�*

loss3i�<c7�       �	 ӒYc�A�*

loss|b=͑�=       �	.r�Yc�A�*

loss���=�"�       �	(G�Yc�A�*

loss*�>�@�~       �	��Yc�A�*

loss���=J�-�       �	�~�Yc�A�*

lossWbE=�CE       �	>?�Yc�A�*

loss��=wXQ�       �	��Yc�A�*

loss-�~<�<:r       �	���Yc�A�*

loss{��<�]��       �	6>�Yc�A�*

loss!�,=X:6�       �	y�Yc�A�*

loss�,�==�}       �	�~�Yc�A�*

loss�X=Q[�<       �	��Yc�A�*

lossAr<=�       �	﬚Yc�A�*

loss��=��]b       �	OW�Yc�A�*

loss:h�=$#       �	G�Yc�A�*

loss\�<�2��       �	>��Yc�A�*

loss|�	=2]��       �	K�Yc�A�*

lossTu�=�;R�       �	&�Yc�A�*

loss� =��-       �	�{�Yc�A�*

lossS��<q�L�       �	~�Yc�A�*

loss�+.=;���       �	���Yc�A�*

loss��"=ip��       �	d]�Yc�A�*

loss1�=���       �	o��Yc�A�*

lossn6<�F�       �	 ��Yc�A�*

loss,a�=���w       �	5'�Yc�A�*

losst*�=J?O�       �	+��Yc�A�*

lossi�U={�       �	Q��Yc�A�*

lossƍ�<�i       �	[?�Yc�A�*

loss31v<Ԃ�x       �	�פYc�A�*

lossn�=U�y|       �	��Yc�A�*

loss
�<�ҵ�       �	��Yc�A�*

lossm(�<pI$�       �	x��Yc�A�*

loss;�o=��	�       �	eT�Yc�A�*

loss�{<��:       �	x��Yc�A�*

loss�v=���U       �	ē�Yc�A�*

lossg�;����       �	�u�Yc�A�*

loss�֜<3Ye       �	o�Yc�A�*

loss�(�</l2�       �	��Yc�A�*

loss
o�<�j$>       �	й�Yc�A�*

lossG�<��@j       �	�\�Yc�A�*

lossIyR=��ٗ       �	S�Yc�A�*

loss�g�=5���       �	n��Yc�A�*

loss�"�<���       �	K�Yc�A�*

loss�^=^��       �	|�Yc�A�*

loss�Ζ=1�!o       �	ő�Yc�A�*

loss�l�=,�       �	V-�Yc�A�*

lossdg�<�̊�       �	l�Yc�A�*

loss���<s\m       �	���Yc�A�*

loss�5�=U([�       �	�G�Yc�A�*

loss�v�<����       �	���Yc�A�*

loss��=A@%�       �	K�Yc�A�*

lossJ��<���       �	�Yc�A�*

lossR_�=S�       �	��Yc�A�*

lossR5?=�!�       �	��Yc�A�*

loss�y-=��"       �	�?�Yc�A�*

lossP�<#�1�       �	D߷Yc�A�*

loss�T<��]�       �	-y�Yc�A�*

loss�u=aF�       �	@�Yc�A�*

loss��=��G       �	���Yc�A�*

loss�;�<�pBp       �	�Q�Yc�A�*

losseؿ=�k       �	�Yc�A�*

loss�=�2�       �	���Yc�A�*

loss]�=����       �	d �Yc�A�*

loss<6�=�a��       �	I��Yc�A�*

loss��3=8�       �	�Q�Yc�A�*

loss�;@=���       �	��Yc�A�*

loss�+�=���K       �	A��Yc�A�*

loss�s<�dE       �	�N�Yc�A�*

loss�8�<��       �	��Yc�A�*

loss�z@=�YMf       �	��Yc�A�*

loss��<8�~       �	D�Yc�A�*

loss_�<J�?�       �	���Yc�A�*

loss�=	�c�       �	y��Yc�A�*

loss-k&=Z�Yl       �	"T�Yc�A�*

loss8��<JB�.       �	���Yc�A�*

loss�ܱ<�Dh&       �	ӟ�Yc�A�*

loss���<0�^       �	\��Yc�A�*

loss�� <�\~�       �	�:�Yc�A�*

lossd�<��9y       �	O��Yc�A�*

loss�5=�(�       �	���Yc�A�*

lossY>�xט       �	�/�Yc�A�*

loss��=�|F�       �	p��Yc�A�*

loss�L=���       �	���Yc�A�*

loss$�V<%�S       �	�V�Yc�A�*

lossL��<���       �	�Yc�A�*

loss�)>o"1�       �	Υ�Yc�A�*

lossMt�<h�       �	�A�Yc�A�*

lossr�,=$a��       �	���Yc�A�*

loss8�`=��N�       �	_z�Yc�A�*

loss�N=��T       �	�Yc�A�*

loss'�=��Tm       �	(��Yc�A�*

loss}�<�s�}       �	�N�Yc�A�*

loss)�R=!d22       �	��Yc�A�*

loss��]=���       �	f��Yc�A�*

loss��=��د       �	/�Yc�A�*

lossE�=N3�       �	l��Yc�A�*

loss �=��Hv       �	\u�Yc�A�*

loss��=3��p       �	?�Yc�A�*

loss��_=aI\       �	���Yc�A�*

loss6�=�9y9       �	�U�Yc�A�*

loss��<�,�       �	��Yc�A�*

loss�4i=����       �	V��Yc�A�*

lossi �<WЯ�       �	O>�Yc�A�*

lossHP=����       �	��Yc�A�*

loss�^=G�˖       �	���Yc�A�*

loss�!=&r��       �	�P�Yc�A�*

loss̓#<y)�	       �	�"�Yc�A�*

loss�Z�<���R       �	���Yc�A�*

loss1;T;{!;       �	�S�Yc�A�*

loss��=���m       �	���Yc�A�*

loss�!=�Nn�       �	���Yc�A�*

loss�m�<��<       �	�Yc�A�*

loss�L>���J       �	,��Yc�A�*

loss۾=z?	       �	@Q�Yc�A�*

loss\:�<yLR       �	}��Yc�A�*

loss���<��:�       �	,��Yc�A�*

loss/��<�~��       �	�W�Yc�A�*

loss3C�<�&@       �	R��Yc�A�*

loss1�= ҙ�       �	���Yc�A�*

losse�>ܾ�       �	�)�Yc�A�*

loss��=�зI       �	���Yc�A�*

loss�<�9�       �	Ui�Yc�A�*

loss�xh=07�8       �	W�Yc�A�*

loss3p�=�B��       �	ˡ�Yc�A�*

loss1
�=M�/�       �	�9�Yc�A�*

lossUP=dX4       �	x��Yc�A�*

losss�=�{�       �	���Yc�A�*

loss*�'=�-       �	��Yc�A�*

loss62=����       �	���Yc�A�*

loss�X=tl!       �	�t�Yc�A�*

loss}�=W1 �       �	w�Yc�A�*

loss�,�<�Ԓ       �	*��Yc�A�*

loss�=�&V�       �	JF�Yc�A�*

loss��2<p�&       �	j��Yc�A�*

lossI�<d��"       �	hw�Yc�A�*

loss4�Z=#Ep�       �	�Yc�A�*

lossF��<���       �	���Yc�A�*

loss��+=�:��       �	�Yc�A�*

loss���=�y��       �	ظ�Yc�A�*

loss��=-/�       �	�`�Yc�A�*

loss���<���2       �	��Yc�A�*

lossH��<N3       �	��Yc�A�*

lossA�<%fo       �	�I�Yc�A�*

loss���=2 �y       �	���Yc�A�*

loss�Q=���       �	'��Yc�A�*

lossT�`=�p�?       �	<�Yc�A�*

loss(=-�G�       �	r�Yc�A�*

lossb�<���9       �	�G�Yc�A�*

loss�m�<�rL�       �	��Yc�A�*

loss�y�<���       �	Y��Yc�A�*

losso=?A��       �	���Yc�A�*

loss��<W���       �	��Yc�A�*

loss� �=���z       �	�R�Yc�A�*

lossZQ�=��K�       �	Q-�Yc�A�*

loss�x=I��q       �	���Yc�A�*

lossS�= I�       �	��Yc�A�*

loss1�?=���       �	�6�Yc�A�*

lossiNm<���       �	���Yc�A�*

lossj=���<       �	k��Yc�A�*

loss�a�<���H       �	+�Yc�A�*

loss�{<ϑ!       �	���Yc�A�*

loss�=��}p       �	%t�Yc�A�*

lossJ$=��       �	��Yc�A�*

lossj�=�/9       �	a� Yc�A�*

loss�3U=�懘       �	PYc�A�*

loss��4=܊�       �	r�Yc�A�*

loss�=7�e�       �	ÝYc�A�*

lossh��=̌��       �	�FYc�A�*

loss�@�<~��@       �	��Yc�A�*

lossZj5=�a�       �	]�Yc�A�*

loss@G=<�@�       �	w-Yc�A�*

loss8W<j        �	�Yc�A�*

loss��:� �l       �	>�Yc�A�*

loss��<僡�       �	�0Yc�A�*

lossmaQ<�"Ѯ       �	o�Yc�A�*

loss͠e<��
l       �	�vYc�A�*

loss��<"<��       �	�	Yc�A�*

loss�e:0q�q       �	�	Yc�A�*

loss�=O�=       �	>Z
Yc�A�*

loss[��:�fri       �	U�
Yc�A�*

lossq}c:φN       �	��Yc�A�*

loss�\%:�tj       �	S<Yc�A�*

loss��=���M       �	�Yc�A�*

loss�r�<��F       �	�pYc�A�*

loss���<�8�       �	�Yc�A�*

loss$�;�uI�       �	=�Yc�A�*

loss���<!�       �	OYc�A�*

loss8E�=��f       �	��Yc�A�*

loss\PH;�a��       �	�Yc�A�*

loss���<C�5       �	�/Yc�A�*

loss,{=x8        �	�Yc�A�*

losst�=#�A�       �	�Yc�A�*

loss@�=%m��       �	t�Yc�A�*

loss���<_�|       �	\9Yc�A�*

lossE�=���       �	;�Yc�A�*

loss4�=?�       �	=~Yc�A�*

loss ��=�y       �	� Yc�A�*

loss�}�=8V�4       �	4�Yc�A�*

loss�=Tfc       �	MeYc�A�*

loss)b=D�d       �	�Yc�A�*

lossM�T=dP�N       �	��Yc�A�*

loss��&<<~�       �	�=Yc�A�*

loss��a=PΒ�       �	��Yc�A�*

loss2;R=Z|�       �	OuYc�A�*

loss��<�Q$�       �	Yc�A�*

loss?�<>
�       �	h�Yc�A�*

loss��=Y��       �	TYc�A�*

loss��<����       �	[�Yc�A�*

loss��<p��       �	q�Yc�A�*

lossY�=7�5       �	N'Yc�A�*

lossz�s<�zg^       �	��Yc�A�*

loss��]<\k��       �	)]Yc�A�*

loss�"�<�Y'+       �	#�Yc�A�*

losslr�<{g{       �	�� Yc�A�*

loss��=�)��       �	
+!Yc�A�*

loss�$=�ޭU       �	#�!Yc�A�*

lossau�<���       �	�w"Yc�A�*

loss��B=y��       �	�#Yc�A�*

loss�Q�<�OE�       �	��#Yc�A�*

losskC=���Y       �	�E$Yc�A�*

loss?�<���1       �	�$Yc�A�*

lossj��;��Q       �	)v%Yc�A�*

loss���=�j"       �	�&Yc�A�*

loss�"�;�-��       �	y�&Yc�A�*

loss�A/<����       �	�H'Yc�A�*

loss�b=��.       �	%�'Yc�A�*

loss�?=���G       �	@�(Yc�A�*

lossO5(=8��       �	 ')Yc�A�*

loss:m<���c       �	��)Yc�A�*

loss���<cn�t       �	[^*Yc�A�*

loss��<}�,�       �	��*Yc�A�*

lossq�n<��_�       �	��+Yc�A�*

loss�d<�J�       �	U3,Yc�A�*

loss$c�<hUo       �	��-Yc�A�*

loss���<�c       �	�(.Yc�A�*

loss$� <�ؑ       �	��.Yc�A�*

loss��=�nQ       �	s/Yc�A�*

loss2�%<���       �	(0Yc�A�*

loss�Q<��݌       �	X�0Yc�A�*

loss$��<�Ǻ       �	�PJYc�A�*

lossjN�<��̀       �	M�JYc�A�*

loss���=u�       �	��KYc�A�*

loss�BR=�3z�       �	�LYc�A�*

loss=�5=�zCQ       �	KMYc�A�*

loss�r�<F���       �	s�MYc�A�*

loss�|�<��؞       �	��NYc�A�*

loss��<P�҂       �	�WOYc�A�*

loss^��=-GMk       �	3�OYc�A�*

loss$�<��N       �	��PYc�A�*

loss��\<;�%�       �	UkQYc�A�*

loss���<̂e       �	(,RYc�A�*

loss��<J@�@       �	0�RYc�A�*

loss�i~=���       �	��SYc�A�*

loss,�2=�7�f       �	@�TYc�A�*

loss�N�=�         �	mTUYc�A�*

loss�d�<��B�       �	mVVYc�A�*

loss��<iXV       �	8�VYc�A�*

loss��<._Y�       �	�WYc�A�*

loss�f�=�/       �	K<XYc�A�*

loss�\7=�KK�       �	��XYc�A�*

loss��=�?3�       �	�xYYc�A�*

lossX�l<2�D        �	0ZYc�A�*

loss���=�=��       �	��ZYc�A�*

loss�׿<����       �	K[Yc�A�*

lossH;�<�v�       �	�[Yc�A�*

lossl�d=�84       �	��\Yc�A�*

lossE�0<�?2^       �	)]Yc�A�*

loss�=(���       �		�]Yc�A�*

lossr.=dŔ�       �	yY^Yc�A�*

loss;�V=�>��       �	�^Yc�A�*

loss�v�<%�N       �	Y�_Yc�A�*

loss�Y=�Jkh       �	q`Yc�A�*

loss�\1= P��       �	��`Yc�A�*

lossf�S<�]�       �	�JaYc�A�*

loss�`=�'�       �	��aYc�A�*

loss��;��,       �	�}bYc�A�*

lossLS<��Λ       �	McYc�A�*

loss���=���i       �	*�cYc�A�*

losss/+=�`3�       �	�<dYc�A�*

lossW�<���       �	��dYc�A�*

lossA��<hй�       �	YieYc�A�*

loss�W�<B���       �	\�eYc�A�*

lossv3-=|�̂       �	�fYc�A�*

loss#F�=��E       �	>?gYc�A�*

loss}�=���       �	��gYc�A�*

loss'��<7#�       �	c�hYc�A�*

loss:C
=*{��       �	�-iYc�A�*

loss��<x�A       �	��iYc�A�*

loss�#�<o3�I       �	6VjYc�A�*

loss�`<S ��       �	R�jYc�A�*

loss8Ui=k׾       �	��kYc�A�*

lossHs=���       �	mlYc�A�*

loss��;>�ֽ�       �	8�mYc�A�*

loss�Rw=��4       �	@�nYc�A�*

lossc>�<��       �	�)oYc�A�*

loss��p;2�L\       �	,�oYc�A�*

loss��;X|p       �	5~pYc�A�*

lossd�q<Vb\       �	&qYc�A�*

loss�I=//�       �	i�qYc�A�*

lossd�-=E��       �	�MrYc�A�*

loss�=��       �	��rYc�A�*

loss�>�;y8
V       �	�sYc�A�*

lossݍx=BI��       �	�$tYc�A�*

loss�&t<���       �	��tYc�A�*

loss�V<���       �	�huYc�A�*

loss�C<:�'       �	bvYc�A�*

loss���<��1�       �	��vYc�A�*

loss�q�=�c�       �	��wYc�A�*

loss?RB=R���       �	80xYc�A�*

loss���<o��Y       �	�xYc�A�*

lossS�:= �K�       �	��yYc�A�*

lossʄ=���       �	� zYc�A�*

loss��<�(�j       �	��zYc�A�*

loss1�c=�i��       �	`s{Yc�A�*

loss�<�֓�       �	|Yc�A�*

loss�ʳ<��?       �	��|Yc�A�*

loss��<��qa       �	@L}Yc�A�*

loss�8<$�p       �	��}Yc�A�*

loss<��<y��       �	��~Yc�A�*

loss�O�<.��       �	b.Yc�A�*

lossۤ9=�k�K       �	��Yc�A�*

loss�^=�p+t       �	�i�Yc�A�*

loss헰;ׯ�       �	��Yc�A�*

loss�)*=�(�(       �	���Yc�A�*

losseHH<���       �	�<�Yc�A�*

loss�ne<X�       �	�ׂYc�A�*

loss=�=����       �	�q�Yc�A�*

loss��m<}|�g       �	t	�Yc�A�*

lossnf�<�={a       �	���Yc�A�*

loss���</}�       �	�~�Yc�A�*

loss��t=��k.       �	7�Yc�A�*

loss�-<"l!       �	}��Yc�A�*

loss��<���       �	�I�Yc�A�*

loss[O2=$��       �	��Yc�A�*

lossZ%�;zE[       �	���Yc�A�*

loss���=��^       �	:W�Yc�A�*

loss���<0��       �	���Yc�A�*

lossxZ�=�A�       �	ę�Yc�A�*

lossA)<�_�6       �	�>�Yc�A�*

loss[3V=�
��       �	�Yc�A�*

lossD�P;��<�       �	;��Yc�A�*

loss�Q�=��I�       �	�%�Yc�A�*

loss�]�<�꧳       �	vƍYc�A�*

lossZ��<[��       �	�\�Yc�A�*

loss�'�=�u�       �	�Yc�A�*

loss��;*���       �	ԁ�Yc�A�*

loss��3=$Gc�       �	e�Yc�A�*

loss�"�=?ͷ       �	��Yc�A�*

loss��\<R�h-       �	�Q�Yc�A�*

losso�h<&3�)       �	��Yc�A�*

loss*�<���       �	�u�Yc�A�*

loss
i=��y       �	�Yc�A�*

loss�5&< ��       �	d�Yc�A�*

losse�{=:�N�       �	��Yc�A�*

loss�ĝ<��W       �	��Yc�A�*

loss�w<-�{�       �	½�Yc�A�*

loss�z=��       �	]�Yc�A�*

loss˽�<��       �	&��Yc�A�*

loss�>.<a�xT       �	֨�Yc�A�*

loss)�{<�}��       �	G�Yc�A�*

loss�M�=%�9�       �	��Yc�A�*

loss�?z=(�Q       �	���Yc�A�*

lossl�<"���       �	V�Yc�A�*

loss&�=q# �       �	k�Yc�A�*

loss� j=��17       �	���Yc�A�*

loss���<��L�       �	�9�Yc�A�*

loss�^�<�5:E       �	�МYc�A�*

loss��=t�       �	>x�Yc�A�*

loss<�	=`--m       �	��Yc�A�*

loss��=ćk       �	�Yc�A�*

losse�y<�V��       �	�`�Yc�A�*

loss]
�<����       �	a��Yc�A�*

loss���<»��       �	���Yc�A�*

loss�5�=,�[N       �	F�Yc�A�*

lossų�<:tuT       �	n��Yc�A�*

loss�j<y=��       �	٢Yc�A�*

lossNZ=K�s       �	�{�Yc�A�*

loss�	=�O�       �	(�Yc�A�*

lossx�X<��kS       �	��Yc�A�*

loss���;��{_       �	�L�Yc�A�*

loss��<'�r�       �	.�Yc�A�*

loss@�l=��C       �	N~�Yc�A�*

loss�j�<c       �	s.�Yc�A�*

loss���</��       �	ѧYc�A�*

lossl��<�"�       �	�v�Yc�A�*

lossT�=�ME~       �	Y��Yc�A�*

loss��<0��       �	�)�Yc�A�*

loss�;��2       �	۪Yc�A�*

loss�E<���       �	���Yc�A�*

loss�$�<t�       �	�#�Yc�A�*

loss
�=)�{       �	W"�Yc�A�*

loss�cT=�^K�       �	(�Yc�A�*

loss�t1=�|x�       �	��Yc�A�*

loss�T=����       �	���Yc�A�*

loss�;Rd�       �	S!�Yc�A�*

loss��l<�~       �	I*�Yc�A�*

loss���=��       �	ȲYc�A�*

lossz�=�Gw�       �	l#�Yc�A�*

loss�s<�o��       �	�ŴYc�A�*

loss#=���d       �	1`�Yc�A�*

loss��='��       �	��Yc�A�*

loss�p"=��e       �	QٶYc�A�*

loss��:i"��       �	]ܷYc�A�*

loss<PH=�yu       �	 z�Yc�A�*

loss�؇<�[y       �	�3�Yc�A�*

loss��;"�       �	�ѹYc�A�*

loss��|=�X<.       �	3p�Yc�A�*

loss�:�<W��;       �	x�Yc�A�*

loss�e<����       �	�ûYc�A�*

loss �=|f�+       �	Eb�Yc�A�*

loss�_�<��a        �	�Yc�A�*

lossV�<���       �	Ú�Yc�A�*

loss��=ㄨ       �	�Q�Yc�A�*

loss��	=j׶�       �	��Yc�A�*

loss �A=Ԝ9       �	*ʿYc�A�*

loss�k�<l�?-       �	�c�Yc�A�*

loss�s=�kλ       �	���Yc�A�*

loss�I=�VU       �	1��Yc�A�*

loss�M�<Y<p�       �	�5�Yc�A�*

lossj= ���       �	�1�Yc�A�*

loss�<���Y       �	��Yc�A�*

loss�"w<y�C       �	�b�Yc�A�*

loss/3<0ri       �	��Yc�A�*

loss�K=*Ao       �	N�Yc�A�*

loss;dD<`U�       �	��Yc�A�*

loss�+�=�*��       �	t��Yc�A�*

lossr;�<G��       �	aT�Yc�A�*

loss��<��4.       �	 ��Yc�A�*

loss�y[=���O       �	�r�Yc�A�*

loss?�=4W��       �	F?�Yc�A�*

loss�b�<-X@�       �	�	�Yc�A�*

loss�<M���       �	��Yc�A�*

loss`F<8Y4�       �	�A�Yc�A�*

loss�2<�F��       �	���Yc�A�*

lossr�<q�8       �	Z��Yc�A�*

lossL.<l�]       �	=�Yc�A�*

loss�/�<��=       �	���Yc�A�*

loss��W=b~}m       �	���Yc�A�*

loss}O><�#�       �	,�Yc�A�*

loss�l�=����       �	���Yc�A�*

lossV��;��;        �	d]�Yc�A�*

lossON=��;�       �	���Yc�A�*

lossu�=�8
       �	���Yc�A�*

loss��<���       �	,�Yc�A�*

loss��<"�F�       �	���Yc�A�*

loss�=��jD       �	;U�Yc�A�*

lossT��=�7�       �	���Yc�A�*

loss���<٥5       �	���Yc�A�*

loss���<&��-       �	q8�Yc�A�*

loss��<g��       �	���Yc�A�*

lossݑg<ӟ�N       �	1{�Yc�A�*

loss{IV<��H       �	� �Yc�A�*

loss�eo<7��       �	���Yc�A�*

lossJYe=ɛ��       �	�Q�Yc�A�*

lossn~B=\�[       �	S��Yc�A�*

lossh��<���       �	���Yc�A�*

lossE��=�;       �	�V�Yc�A�*

loss{�?=��_�       �	(��Yc�A�*

loss���=z���       �	��Yc�A�*

loss�@N<�� �       �	�%�Yc�A�*

loss2FK<q�F�       �	{��Yc�A�*

loss��<ť�       �	�T�Yc�A�*

lossܯO=N�*       �	��Yc�A�*

lossf"w<�LC       �	���Yc�A�*

lossZB�<|��       �	D3�Yc�A�*

loss�-�<�灒       �	���Yc�A�*

loss��;A�X       �	<g�Yc�A�*

loss �<�5�       �	*��Yc�A�*

loss,�=�k�r       �	���Yc�A�*

loss�	[;B��       �	�-�Yc�A�*

loss�F�<�`�2       �	���Yc�A�*

lossA\�<����       �	�b�Yc�A�*

loss��<��       �	��Yc�A�*

lossiu�=W#�Y       �	;��Yc�A�*

lossv��<��D�       �	��Yc�A�*

loss���<�B�V       �	�?�Yc�A�*

loss @�;K6%[       �	q��Yc�A�*

loss�h}=���       �	���Yc�A�*

loss3�7<�=J�       �	�=�Yc�A�*

loss:,�<_09�       �	v��Yc�A�*

loss�!�<���       �	n��Yc�A�*

loss.��<�x8�       �	�i�Yc�A�*

loss]��<.��_       �	9�Yc�A�*

loss<�=��A       �	֬�Yc�A�*

loss���<u�>�       �	�H�Yc�A�*

lossv=3m�]       �	�M�Yc�A�*

loss�a=�z��       �	^��Yc�A�*

lossp<n~N       �	��Yc�A�*

loss[yL<<�Q�       �	0.�Yc�A�*

loss�q=����       �	P�Yc�A�*

loss��=8fϚ       �	Ț�Yc�A�*

loss	p�=�-K       �	74�Yc�A�*

loss��<�6�       �	���Yc�A�*

loss��<�y�       �	Ig�Yc�A�*

loss<t�=�'       �	�
�Yc�A�*

loss�H`=0�&f       �	���Yc�A�*

lossJ4�<�t�z       �	�<�Yc�A�*

loss�l�=����       �	��Yc�A�*

loss���<t�l�       �	z�Yc�A�*

lossօ�;F#       �	��Yc�A�*

lossl<iM       �	;��Yc�A�*

loss��=�N�>       �	$E�Yc�A�*

lossL8�<����       �	���Yc�A�*

loss<d�=n�       �	9|�Yc�A�*

loss�f=<���       �	��Yc�A�*

loss���=KA��       �	���Yc�A�*

loss.I=ܝI       �	W]�Yc�A�*

losshn7=����       �	�&�Yc�A�*

loss�!+=��       �	���Yc�A�*

lossf��<�>�       �	Dj�Yc�A�*

loss$0�<�0�       �	��Yc�A�*

loss�X	=Z��       �	$��Yc�A�*

loss��<��C       �	bN Yc�A�*

lossO��=���       �	�� Yc�A�*

loss�V�=��K�       �	&�Yc�A�*

loss��/=l21       �	=(Yc�A�*

lossN��<H[;!       �	��Yc�A�*

loss���<u�'       �	�^Yc�A�*

loss���<q��       �	v�Yc�A�*

loss�d*;�v�3       �	��Yc�A�*

loss���;q�'       �	0bYc�A�*

loss�#S<�{*       �	�Yc�A�*

loss�	�< �>�       �	ڮYc�A�*

loss.HV=z�5       �	�Yc�A�*

loss�;�<��R�       �	x~Yc�A�*

loss�t=����       �	�&	Yc�A�*

loss};�<#`#D       �	��	Yc�A�*

loss�"=a��       �	~
Yc�A�*

losszE;�a       �	� Yc�A�*

loss��4<�̑�       �	�Yc�A�*

loss&�\=IV?�       �	��Yc�A�*

loss��:(x�       �	�nYc�A�*

lossc��;(�       �	#Yc�A�*

loss�0=UGMI       �	Q�Yc�A�*

lossq�R=ѐ       �	ɓYc�A�*

loss�`_<�%�       �	�7Yc�A�*

loss��= �U'       �	��Yc�A�*

loss��=�E��       �	ZYc�A�*

loss	f=u���       �	+Yc�A�*

loss�h�<ww��       �	(�Yc�A�*

loss��<��Y       �	PYc�A�*

loss��U<��a       �	��Yc�A�*

loss϶<��/�       �	ƉYc�A�*

loss���<O��x       �	�(Yc�A�*

loss��Y=�Sי       �	X�Yc�A�*

loss�� ;�
ұ       �	paYc�A�*

loss�p	=���       �	+�Yc�A�*

loss��0=���       �	K�Yc�A�*

loss���=�R{       �	{-Yc�A�*

losss��<���        �	q�Yc�A�*

loss���<���       �	�Yc�A�*

loss��u=89^�       �	Yc�A�*

loss|X�;=_�       �	��Yc�A�*

lossj�=�&n�       �	�pYc�A�*

loss4P�=��       �	*Yc�A�*

loss���;���       �	|�Yc�A�*

loss���<��pW       �	${Yc�A�*

lossV�=>��%       �	%$Yc�A�*

loss �<���       �	a�Yc�A�*

loss�<=��Շ       �	&mYc�A�*

lossr�=�R!�       �	Z Yc�A�*

lossT�!=0ߣ8       �	!Yc�A�*

loss�;9o�D       �	��!Yc�A�*

loss��m<��<1       �	�^"Yc�A�*

loss�K=�"�<       �	�#Yc�A�*

loss�$�=�P       �	�#Yc�A�*

loss�m}<Z�@       �	fI$Yc�A�*

lossd�W=C��       �	3�$Yc�A�*

lossmy�=S5{�       �	�%Yc�A�*

losse�≮��       �	�H&Yc�A�*

loss�.N<zx�       �	D'Yc�A�*

loss]��;06W=       �	t�'Yc�A�*

loss�1�<Zd<�       �	D�(Yc�A�*

lossR��=��=       �	Ze)Yc�A�*

loss:��<��Xy       �	�*Yc�A�*

loss���<��       �	��*Yc�A�*

loss�e�<�W�       �	&R+Yc�A�*

lossK�<Q,�\       �	�+Yc�A�*

loss@zS<���       �	�4-Yc�A�*

loss��;�&��       �	�.Yc�A�*

loss%��<���       �	`�.Yc�A�*

loss&��;��7       �	�t/Yc�A�*

lossM?{<��L�       �	T70Yc�A�*

loss��=w[u�       �	k1Yc�A�*

lossT��=I_�       �	Z2Yc�A�*

loss��B<Rv�       �	M�2Yc�A�*

loss�n�=i�_�       �	y4Yc�A�*

loss�=3�f       �	��4Yc�A�*

loss�<��]�       �	�R5Yc�A�*

losskd=�K��       �	�-6Yc�A�*

lossŸ�;��E       �	��6Yc�A�*

loss�=ofE�       �	��7Yc�A�*

loss�=��&       �	��8Yc�A�*

lossQ�=���       �	��9Yc�A�*

loss&N1<A���       �	�:Yc�A�*

loss�p=��N�       �	��;Yc�A�*

loss8�<��5       �	�<Yc�A�*

loss�X�=]���       �	Q�=Yc�A�*

loss+,�<����       �	�@>Yc�A�*

loss�Fu=�	\       �	?Yc�A�*

lossB?=��RK       �	B�?Yc�A�*

loss�;=vLU       �	rR@Yc�A�*

loss@;�=Ӭ�d       �	��@Yc�A�*

lossL�=C�W       �	ߋAYc�A�*

loss�fa=Ou��       �	J]BYc�A�*

lossFH;=4ֵ,       �	 CYc�A�*

loss��<z3�V       �	0�CYc�A�*

loss
<�3ۣ       �	6DYc�A�*

lossҾ<9�       �	t�DYc�A�*

loss�p={E�f       �	jhEYc�A�*

loss�L�;�       �	�FYc�A�*

loss8�[<?�qH       �	b�FYc�A�*

loss��<[�,�       �	�DGYc�A�*

lossr�9<�;       �	��GYc�A�*

loss΅=�}�n       �	�HYc�A�*

lossS�<��L�       �	t)IYc�A�*

loss��J=-��       �	�IYc�A�*

loss
t=���       �	jJYc�A�*

loss�$�= �.       �	�KYc�A�*

lossK�<pz��       �	RdLYc�A�*

loss}�<l�       �	~�LYc�A�*

loss��<�֊        �	/�MYc�A�*

loss�R�<��<X       �	�?NYc�A�*

loss�C<h�v       �	��NYc�A�*

lossnNV=�h(       �	)�OYc�A�*

lossW�K=J��       �	�>PYc�A�*

loss�x=nW'p       �	~�PYc�A�*

lossH��;33̤       �	��QYc�A�*

loss�g<���&       �	�RYc�A�*

lossڃ=��Y       �	ȳRYc�A�*

lossB��=�Y�       �	�VSYc�A�*

loss���<u���       �	��SYc�A�*

loss���<wg�       �	v�TYc�A�*

loss�M�<:�a�       �	bKUYc�A�*

lossm�=M�g�       �	��UYc�A�*

lossa��:m/��       �	E�VYc�A�*

loss��<��64       �	E�WYc�A�*

loss��<�Nu       �	KsXYc�A�*

lossF�=�)I�       �	p&YYc�A�*

lossi��<���       �	C�YYc�A�*

loss���<�5*�       �	�wZYc�A�*

loss �.=���       �	r[Yc�A�*

loss,�E<Sg6�       �	O�[Yc�A�*

lossT��=iMhC       �	�{\Yc�A�*

loss��=M��       �	�]Yc�A�*

loss��=�ǜ=       �	w�]Yc�A�*

loss���<�;�       �	�^^Yc�A�*

lossa�<J��       �	J	_Yc�A�*

loss���=�辂       �	��_Yc�A�*

loss~i=kq�.       �	=b`Yc�A�*

loss��?=�y       �	�aYc�A�*

loss/�<�̃�       �	ٰaYc�A�*

loss��s=���       �	�NbYc�A�*

loss!�R=4�"�       �	��bYc�A�*

loss�yB=�v8�       �	0�cYc�A�*

loss��P;�2�       �	>dYc�A�*

lossʂ�<ێ�       �	��dYc�A�*

loss��=Ğ}�       �	(�eYc�A�*

loss��l=���       �	?fYc�A�*

loss�k=HJ��       �	��fYc�A�*

lossk�=�";�       �	�gYc�A�*

loss�Z�<�       �	�HhYc�A�*

lossWb�=b=�       �	��hYc�A�*

loss�t�=B��       �	�{iYc�A�*

lossZwP=k`��       �	�jYc�A�*

loss3m3=Dr+�       �	�jYc�A�*

loss���<�t�       �	VGkYc�A�*

loss�,�<X��       �	H�kYc�A�*

loss�<8��       �	mYc�A�*

lossӛ]=��       �	)�mYc�A�*

loss���<+lݳ       �	 �nYc�A�*

loss��=��       �	�oYc�A�*

loss�<��}�       �	~:pYc�A�*

lossݼ�=��       �	�}qYc�A�*

loss�b3<��h�       �	HrYc�A�*

lossE��<���       �	��rYc�A�*

loss��<���       �	H�sYc�A�*

loss��}<a�8       �	��tYc�A�*

losso:<�IH&       �	%>uYc�A�*

lossd�2=z�~J       �	�vYc�A�*

loss�>�x�       �	��vYc�A�*

lossI	O<�ťN       �	��wYc�A�*

lossTm�<i\6�       �	�rxYc�A�*

lossoF�;���       �	�lyYc�A�*

loss�I<�rӵ       �	��zYc�A�*

loss�� >��I�       �	n�{Yc�A�*

loss�v<Ù�q       �	��|Yc�A�*

loss_�&=f&��       �	.�}Yc�A�*

lossd$�<&�2�       �	�~Yc�A�*

loss�R�<�M
       �	�Yc�A�*

loss�F�<,��       �	t��Yc�A�*

loss�Q�;�f       �	W]�Yc�A�*

loss!�>�,�       �	+�Yc�A�*

loss�1>)��       �	���Yc�A�*

lossa�<_��h       �	uʃYc�A�*

loss��<��       �	{�Yc�A�*

loss&�r=vD�        �	�%�Yc�A�*

lossh��<'���       �	���Yc�A�*

loss��=Yz�       �	���Yc�A�*

lossaN�<}���       �	jO�Yc�A�*

lossV�0<#��       �	[[�Yc�A�*

loss1T<���       �	_�Yc�A�*

lossAB�<�}r       �	���Yc�A�*

loss�=�_�|       �	ᖊYc�A�*

lossB=���       �	AI�Yc�A�*

loss�UW<y�j�       �	��Yc�A�*

loss�#<���       �	w��Yc�A�*

lossJJ0<%{��       �	�$�Yc�A�*

loss���;��4�       �	/��Yc�A�*

loss�J�=���       �	qT�Yc�A�*

loss���<�W�       �	��Yc�A�*

loss7�J=��`�       �	���Yc�A�*

loss�0=�= I       �	 )�Yc�A�*

loss�J
<�>S�       �	&ÐYc�A�*

losss<;��y       �	�Z�Yc�A�*

lossZa�<qi�       �	��Yc�A�*

lossc��;N/��       �	A��Yc�A�*

loss�<M~�y       �	z6�Yc�A�*

loss��=�e�t       �	)͓Yc�A�*

loss&�=��M!       �	wj�Yc�A�*

loss��<Ĺb       �	��Yc�A�*

loss�%=g5Ek       �	m��Yc�A�*

loss\�W=�q�       �	�X�Yc�A�*

loss�={~�m       �	��Yc�A�*

loss��j<��J       �	숗Yc�A�*

loss
k�;Ou�       �	�A�Yc�A�*

loss�t1=�       �	7�Yc�A�*

loss-�;�h�       �	%w�Yc�A�*

loss�=X�       �	��Yc�A�*

lossq�G<��]       �	���Yc�A�*

loss���=�̎f       �	E�Yc�A�*

loss�J<�47       �	ޛYc�A�*

loss��=<��I�       �	9��Yc�A�*

loss�O�<���       �	�1�Yc�A�*

loss
/�<իa%       �	�ŝYc�A�*

loss	u�<!��       �	���Yc�A�*

loss�+�<�W7       �	�1�Yc�A�*

loss��=я��       �	�ΟYc�A�*

loss}G�<�V<#       �	��Yc�A�*

lossm��=;mi       �	���Yc�A�*

loss��;�{�       �	�=�Yc�A�*

loss>��<�$j�       �	��Yc�A�*

lossj��;w���       �	ߊ�Yc�A�*

lossT��<>�q�       �	/�Yc�A�*

loss�g�<B;>       �	��Yc�A�*

lossj0<F��       �	��Yc�A�*

loss�!X<�Ce       �	з�Yc�A�*

lossխ<MS^H       �	�l�Yc�A�*

lossȃ =� kH       �	��Yc�A�*

loss�Z�<ʶy       �	��Yc�A�*

lossf9<T��       �	��Yc�A�*

loss��=��,�       �	�1�Yc�A�*

lossm�<^k"       �	���Yc�A�*

loss4=Z�S       �	ɒ�Yc�A�*

lossd�<;%       �	4-�Yc�A�*

loss�,j<FFG�       �	�ҬYc�A�*

lossdQ�<��9s       �	���Yc�A�*

loss�T<Ś_+       �	��Yc�A�*

loss���<�*U       �	kԯYc�A�*

loss�L<�#9       �	��Yc�A�*

lossM�D<J���       �	��Yc�A�*

loss@բ;�1E       �	=ղYc�A�*

loss6?u;0%�       �	��Yc�A�*

lossh��;h��       �	k��Yc�A�*

lossO�+<��|G       �	�q�Yc�A�*

lossÕ�=���G       �	�
�Yc�A�*

lossEfF<}���       �	6��Yc�A�*

loss�X=07       �	@N�Yc�A�*

loss���<^(|@       �	�Yc�A�*

loss\=K�Ò       �	腸Yc�A�*

loss���<�@       �	2"�Yc�A�*

loss{�<	J(�       �	0��Yc�A�*

loss�F":�2�       �	]R�Yc�A�*

loss�>,<Z3Ͽ       �	��Yc�A�*

loss��<Y�O       �	���Yc�A�*

loss`��<��       �	6"�Yc�A�*

loss��
<�֘�       �	���Yc�A�*

lossJɌ;�;C�       �	�O�Yc�A�*

loss��=SH��       �	��Yc�A�*

loss��6<�Z�       �	*��Yc�A�*

loss���:5IV�       �	�&�Yc�A�*

lossQz�9�u�       �	�ĿYc�A�*

loss ��;����       �	�]�Yc�A�*

loss��=Ϭd       �	��Yc�A�*

lossL�;��|       �	��Yc�A�*

loss�;��W       �	H�Yc�A�*

loss���<0���       �	)��Yc�A�*

lossCa�=@I       �	n��Yc�A�*

loss\~�:�(�       �	�^�Yc�A�*

lossw�<���       �	��Yc�A�*

loss&m�<��̬       �	
��Yc�A�*

loss@=�<��Ed       �	���Yc�A�*

loss��<m
��       �	z8�Yc�A�*

loss�@<�M       �	�w�Yc�A�*

loss�=�o�       �	'�Yc�A�*

loss��<�A�b       �	B��Yc�A�*

lossd�2=�#       �	�s�Yc�A�*

loss��<�3M�       �	&�Yc�A�*

lossS{<��'�       �	��Yc�A�*

lossr��=�e-�       �	R`�Yc�A�*

loss�0b=��d       �	�A�Yc�A�*

loss��X<���       �	B!�Yc�A�*

loss�}=}�B6       �	|��Yc�A�*

loss�Ã=� ��       �	c�Yc�A�*

loss=e:=M=       �	<��Yc�A�*

loss���<����       �	t��Yc�A�*

loss*pZ=�A�       �	/4�Yc�A�*

loss;�i=0$       �	��Yc�A�*

loss�Y�<���       �	%��Yc�A�*

loss�<Ts�       �	'P�Yc�A�*

loss
�<���       �	���Yc�A�*

loss�-;*��       �	՗�Yc�A�*

lossK5�;���       �	>�Yc�A�*

loss{�F<��	n       �	���Yc�A�*

loss\ g=���       �	���Yc�A�*

loss֗=��Z;       �	�%�Yc�A�*

loss��m=yt�       �	���Yc�A�*

loss�B=4@3�       �	�]�Yc�A�*

loss�w;��&       �	��Yc�A�*

loss�U<��$       �	��Yc�A�*

loss}dN=P�g       �	�B�Yc�A�*

loss��;5�Ґ       �	+��Yc�A�*

loss4+d<����       �	#��Yc�A�*

loss�̮<��@       �	('�Yc�A�*

loss�{�<���u       �	7��Yc�A�*

loss{\�<ˬ]�       �	$`�Yc�A�*

loss�1�=�I�       �		��Yc�A�*

loss�]<a��       �	#��Yc�A�*

loss���<{]�	       �	XV�Yc�A�*

loss?C�< �       �	���Yc�A�*

loss�=��)=       �	���Yc�A�*

lossC��<8،u       �	y$�Yc�A�*

losso��<;�^�       �	���Yc�A�*

loss]>t=�Z%�       �	�k�Yc�A�*

loss�}
=��4�       �	��Yc�A�*

loss�q<���       �	@��Yc�A�*

lossV�#=��PT       �	:x�Yc�A�*

loss��"<O&�       �	^�Yc�A�*

loss1�<޳�       �	��Yc�A�*

lossl�6=lK       �	�q Yc�A�*

loss��=�Z       �	'Yc�A�*

loss�=.��9       �	h�Yc�A�*

losse�$=-j�       �	�OYc�A�*

loss�\\=�;'S       �	��Yc�A�*

lossڠ�<�N�?       �	{�Yc�A�*

loss�\1=��]       �	|(Yc�A�*

lossyT=6l       �	��Yc�A�*

lossɽC=�Q&       �	�xYc�A�*

lossX�S=4�       �	wYc�A�*

loss��$<Rm��       �	��Yc�A�*

loss��>=g�!�       �	�Yc�A�*

loss:�>b:��       �	p�Yc�A�*

loss}ə=GA�       �	0K	Yc�A�*

losscu�<%&��       �	X
Yc�A�*

loss��H<��       �	9+Yc�A�*

lossV�j:|��       �	��Yc�A�*

loss}̑<�E|       �	��Yc�A�*

loss�0�<vM�       �	�uYc�A�*

loss�1	=����       �	�Yc�A�*

lossG=�bc       �	N�Yc�A�*

loss��1=�;��       �	�]Yc�A�*

loss��;+[C�       �	��Yc�A�*

lossB7�=��kZ       �	{�Yc�A�*

lossV<<ӷ��       �	]OYc�A�*

loss��<�m�       �	��Yc�A�*

loss#Gx=3�       �	ՕYc�A�*

loss7��;�S�       �	 ;Yc�A�*

loss��o=d��;       �	w�Yc�A�*

loss��;E���       �	�}Yc�A�*

loss�#I<���       �	�!Yc�A�*

loss)�<sT       �	��Yc�A�*

loss/�=/��d       �	�aYc�A�*

loss1{�=ӅTj       �	�Yc�A�*

lossz"=�f�       �	ǡYc�A�*

lossr�=�L{�       �	@Yc�A�*

loss6"z<�@�       �	7�Yc�A�*

loss��<��       �	~Yc�A�*

loss��E=��w       �	� Yc�A�*

loss܍[=��4�       �	<�Yc�A�*

loss��<~"f�       �	�`Yc�A�*

lossÚ<���       �	��Yc�A�*

loss�k�<-k_k       �	c�Yc�A�*

loss�gf<x�D       �	�cYc�A�*

loss� ,=�k7       �	� Yc�A�*

loss@�s='��       �	Z�Yc�A�*

loss�V=�c1�       �	�AYc�A�*

lossKƎ=�!�       �	��Yc�A�*

loss$
Z<ږby       �	Ҋ Yc�A�*

lossz�<�2>�       �	4.!Yc�A�*

loss��<^@(       �	I�!Yc�A�*

loss�ɷ=���r       �	 y"Yc�A�*

loss�o�<)ae       �	G#Yc�A�*

loss��=a�*q       �	I�#Yc�A�*

lossɤ�<�G�       �	�Z$Yc�A�*

loss�-.;�U�       �	Y�$Yc�A�*

lossV�;���G       �	�%Yc�A�*

loss��6;
���       �	RD&Yc�A�*

loss���;���i       �	H�&Yc�A�*

loss���<x�(       �	^�'Yc�A�*

loss uW=��w
       �	\!(Yc�A�*

lossr}<<��N       �	��(Yc�A�*

lossO�G;i]�       �	|)Yc�A�*

lossi�<���       �	�*Yc�A�*

loss��<��e�       �	`�*Yc�A�*

lossd-�<�F��       �	�p+Yc�A�*

loss.�w=�Ql�       �	+,Yc�A�*

loss���<ïwZ       �	>�,Yc�A�*

loss���<�=�       �	֐-Yc�A�*

loss��<�j�       �	�3.Yc�A�*

loss՛<Yw�p       �	�/Yc�A�*

loss���<4P��       �	q<0Yc�A�*

loss�=D.�       �	�1Yc�A�*

loss��<�>g       �	x�1Yc�A�*

lossC��<�R�       �	��2Yc�A�*

loss�X<���       �	5~3Yc�A�*

loss!��;����       �	y 4Yc�A�*

loss���;&�TL       �	��4Yc�A�*

loss4�<���       �	D�5Yc�A�*

loss�<w\��       �	�'6Yc�A�*

loss8�H<u�Ǻ       �	�6Yc�A�*

loss���<7�^       �	��7Yc�A�*

loss_�&=.��       �	�O8Yc�A�*

loss�
4;b�f&       �	|�8Yc�A�*

loss��)=
���       �	7�9Yc�A�*

loss^D;=�&�       �	R&:Yc�A�*

loss6JS=A9�       �	��:Yc�A�*

loss6��<��d       �		q;Yc�A�*

loss��=�/��       �	�<Yc�A�*

lossHϝ</p��       �	G�<Yc�A�*

loss��<��Y�       �	�`=Yc�A�*

loss�e=O�x       �	"�=Yc�A�*

lossh<�Cs�       �	��>Yc�A�*

loss$ =A,�       �		7?Yc�A�*

loss��&=W�       �	��?Yc�A�*

loss�J=l�*       �	�i@Yc�A�*

lossLJ�<	�@B       �	�&AYc�A�*

loss�s�<���       �	�AYc�A�*

lossŚ�=W��g       �	)ZBYc�A�*

lossư�<���       �	��BYc�A�*

lossf��<�}       �	��CYc�A�*

loss�/�;H&�k       �	�-DYc�A�*

loss���=jտ�       �	�.EYc�A�*

loss�q<�&JQ       �	��EYc�A�*

loss}=��U�       �	qFYc�A�*

loss��=#��]       �	RGYc�A�*

lossj�;֑�       �	+�GYc�A�*

loss�6<"���       �	\>HYc�A�*

lossb�=%���       �	��HYc�A�*

loss
&<ps�7       �	�IYc�A�*

loss�qP=�"�       �	�JYc�A�*

lossxg�<�2E�       �	��JYc�A�*

loss���<��       �	�QKYc�A�*

lossc�<X�       �	�LYc�A�*

loss�xh<lX��       �	3�LYc�A�*

loss�e;���       �	�QMYc�A�*

loss���<�o.       �	~NYc�A�*

loss��
=��l�       �	��NYc�A�*

loss�$�;�ݿ       �	wIOYc�A�*

loss�Ɇ<sg�       �	y�OYc�A�*

lossZ�~<�g��       �	��PYc�A�*

lossԮ<?Q�f       �	�0QYc�A�*

loss#�=���U       �	1�QYc�A�*

lossA�=��c       �	�iRYc�A�*

losspM�<g���       �	_SYc�A�*

loss�д=��       �	ΧSYc�A�*

lossQՕ<�
(2       �	�ITYc�A�*

loss��<��"�       �	\�TYc�A�*

lossy�<A�       �		�UYc�A�*

loss�I.=�Ә�       �	�-VYc�A�*

loss��7<!��       �	]�VYc�A�*

loss�`�<rʆ�       �	R�WYc�A�*

loss�!<�RZ�       �	L�XYc�A�*

loss*-|<��U)       �	˟YYc�A�*

lossR�L=`'�       �	<ZYc�A�*

loss}�=�Rp9       �	�ZYc�A�*

loss��<�b�4       �	p�[Yc�A�*

loss�nr<.}t6       �	N\Yc�A�*

loss��n<��4       �	N�\Yc�A�*

loss��"<�xI�       �	̘]Yc�A�*

lossX�K;:��       �	@2^Yc�A�*

loss�e,<p�ٜ       �	��^Yc�A�*

lossN�=J��,       �	Ed_Yc�A�*

loss���=��<`       �	V`Yc�A�*

loss�[�=�m       �	ߥ`Yc�A�*

lossi�*=�B~�       �	�GaYc�A�*

loss�zK<8�w       �	��aYc�A�*

loss�W�;�R�       �	��bYc�A�*

lossj�<(�\       �	`>cYc�A�*

loss�ͻ<W��       �	�dYc�A�*

lossc4�<#��       �	2�dYc�A�*

loss�h=��       �	�?eYc�A�*

loss#d=M��       �	��eYc�A�*

loss�=A*��       �	��fYc�A�*

loss!�V=Ը��       �	�!gYc�A�*

loss���:�H�       �	M�gYc�A�*

loss�9�<2\uh       �	�^hYc�A�*

lossW��=<�w       �	ZiYc�A�*

loss|)m=�_,L       �	~�iYc�A�*

lossZ�Y<;bg�       �	�RjYc�A�*

loss�T�<���       �	�jYc�A�*

lossT�l;<<�|       �	ǀkYc�A�*

loss�D<�m�T       �	�lYc�A�*

loss!h	<��       �	طlYc�A�*

loss:K�<�c��       �	$cmYc�A�*

loss���;/���       �	7�mYc�A�*

loss�*q;٭�V       �	�nYc�A�*

loss�%�<�y�       �	m:oYc�A�*

loss�}�<w�0/       �	�JpYc�A�*

lossAu5=��^�       �	�QqYc�A�*

loss-h�=,.�       �	�!rYc�A�*

lossv��<���       �	��rYc�A�*

lossl�%<ړ^y       �	zsYc�A�*

lossǀ<���Y       �	|atYc�A�*

loss]R8<���       �	�uYc�A�*

lossc�;�C�       �	[�uYc�A�*

losss�<��N:       �	fhvYc�A�*

lossN0�=P
v�       �	�wYc�A�*

lossMž==�G�       �	y�wYc�A�*

lossTs<+r��       �	�uxYc�A�*

loss��M;�p�       �	^yYc�A�*

loss�� =���Y       �	��yYc�A�*

loss&�,=ښ�R       �	�VzYc�A�*

loss2U<�Y��       �	�{Yc�A�*

lossz�_;9�       �	��{Yc�A�*

lossn�;�;`�       �	MH|Yc�A�*

loss	�D=�m��       �	e�|Yc�A�*

loss���<}��/       �	؀}Yc�A�*

lossx=��߮       �	�~Yc�A�*

loss7�<�V�j       �	��~Yc�A�*

loss�(=*��1       �	�VYc�A�*

loss��<-%�       �	��Yc�A�*

loss�;U���       �	���Yc�A�*

loss��<���       �	�$�Yc�A�*

loss�Ɉ<�uh�       �	b��Yc�A�*

loss�}E=����       �	9|�Yc�A�*

loss,OW;fx�       �	�Yc�A�*

loss��=�_       �	b��Yc�A�*

loss{�L<2 �]       �	�\�Yc�A�*

loss:�<K�M       �	Z��Yc�A�*

loss,�C=-{I       �	���Yc�A�*

lossֱ�=MM~       �	R)�Yc�A�*

loss�<$�A@       �	���Yc�A�*

loss�9C=���^       �	ćYc�A�*

loss�c<��|�       �	�_�Yc�A�*

loss`�<b��       �	�Yc�A�*

loss�+=�j�Y       �	]��Yc�A�*

loss�~�=3��       �	;�Yc�A�*

loss���<���       �	�ϊYc�A�*

loss��<�o�       �	Qg�Yc�A�*

lossŁ9=f��G       �	�
�Yc�A�*

loss<�>VO�       �	G��Yc�A�*

loss��;���       �	�Q�Yc�A�*

loss��><`O�       �	��Yc�A�*

loss���=���       �	�׎Yc�A�*

loss'R�=J*E�       �	���Yc�A�*

loss#Hw=�E�C       �	�I�Yc�A�*

loss�Z�=	{�9       �	 ��Yc�A�*

lossh�=����       �	�,�Yc�A�*

loss&B=��t�       �	cђYc�A�*

loss��<bt_�       �	zr�Yc�A�*

loss��M<'X�       �	�Yc�A�*

lossr� =qj�       �	졔Yc�A�*

loss��Y=�h��       �	;�Yc�A�*

loss8�<���Z       �	pϕYc�A�*

lossX��<Md�	       �	ge�Yc�A�*

lossr7�<4^��       �	��Yc�A�*

loss�ѩ<_�gJ       �	��Yc�A�*

loss�&=o�2~       �	#I�Yc�A�*

loss�T�<�\�4       �	QޘYc�A�*

loss��<�N��       �	�q�Yc�A�*

loss�k$<�v{       �	�
�Yc�A�*

loss&@�<^��       �	��Yc�A�*

loss�=Ν!�       �	�F�Yc�A�*

losscD�<6���       �	��Yc�A�*

loss�+<���       �	g��Yc�A�*

loss)�-<-=�X       �	��Yc�A�*

loss��=R�U       �	���Yc�A�*

loss	>���       �	�C�Yc�A�*

loss6��;�u�       �	��Yc�A�*

loss��=�kB       �	��Yc�A�*

loss��Y<u�'       �	�S�Yc�A�*

lossf�A=AAq       �	���Yc�A�*

loss�(.=HZ�       �	���Yc�A�*

loss�3�<X.�       �	K:�Yc�A�*

loss���<b�i       �	(֢Yc�A�*

losst2�;��a7       �	�m�Yc�A�*

loss��=�N��       �	t�Yc�A�*

loss��<>n�       �	���Yc�A�*

loss��J=��       �	�?�Yc�A�*

lossa�<��D#       �	sץYc�A�*

loss[�;0�׆       �	�t�Yc�A�*

loss
	�<�X}       �	('�Yc�A�*

loss�[�;�&��       �	�ѧYc�A�*

loss�ro;CJ�>       �	Ag�Yc�A�*

loss�#<��}�       �	? �Yc�A�*

loss��=�a1       �	���Yc�A�*

lossrua<G�       �	�5�Yc�A�*

lossaŒ=�$|�       �	8ڪYc�A�*

loss��=o�Rc       �	�r�Yc�A�*

loss��><%$       �	�
�Yc�A�*

lossx�<��       �	j��Yc�A�*

loss�ق=� �       �	G�Yc�A�*

loss�Ý<=C@d       �	�ޭYc�A�*

lossX�=9b�       �	�x�Yc�A�*

lossq�m<�q'       �	� �Yc�A�*

loss$H=�S(       �	��Yc�A�*

loss߁�<�wa�       �	7Q�Yc�A�*

loss6��=*C�p       �	��Yc�A�*

lossu>=`G �       �	��Yc�A�*

loss�<<"Q�       �	,I�Yc�A�*

loss7o�<�D�Q       �	��Yc�A�*

lossXOa=��&       �	-��Yc�A�*

lossڅ5=��       �	q=�Yc�A�*

lossrW4=���       �	�Yc�A�*

lossEl�= �       �	�ηYc�A�*

loss@r<����       �	ap�Yc�A�*

lossd�;��I        �	�(�Yc�A�*

loss?��;���5       �	�ѹYc�A�*

loss7�#=0�j�       �	(�Yc�A�*

losskG;�^�       �	�"�Yc�A�*

lossԚ�;{��       �	���Yc�A�*

loss���;�&��       �	�^�Yc�A�*

loss�h�<!f��       �	M��Yc�A�*

loss�X�<]���       �	ߧ�Yc�A�*

lossM�=7�       �	)[�Yc�A�*

loss1=R=75       �	8��Yc�A�*

loss	={�/       �	���Yc�A�*

loss8�<U���       �	F]�Yc�A�*

lossť�<�?��       �	�
�Yc�A�*

lossO��<�
^�       �	���Yc�A�*

losssCY=?O�       �	�@�Yc�A�*

loss}��<G�       �	��Yc�A�*

loss��'<�$�       �	nk�Yc�A�*

lossS	d=��$       �	=�Yc�A�*

lossV#=��       �	���Yc�A�*

loss�8<��g5       �	�1�Yc�A�*

loss��^=_ϣ       �	u��Yc�A�*

loss�{�=U_�       �	ۋ�Yc�A�*

lossM�=�x��       �	�2�Yc�A�*

loss�b;=/�       �	���Yc�A�*

lossM<��Q       �	�&�Yc�A�*

loss��<ԫ3�       �	N�Yc�A�*

loss��<[��3       �	j��Yc�A�*

loss�Q�=����       �	p{�Yc�A�*

loss�/�=�f�n       �	��Yc�A�*

lossj|	<ב'       �	0��Yc�A�*

loss�=	a��       �	I��Yc�A�*

loss��<�H       �	{g�Yc�A�*

loss�]=�<+�       �	��Yc�A�*

lossq�f<t���       �	��Yc�A�*

loss��<SY�       �	�e�Yc�A�*

loss�y�=���O       �	�Yc�A�*

loss�5v<�V3	       �	��Yc�A�*

loss���;��M�       �	���Yc�A�*

loss���=��%!       �	���Yc�A�*

loss�n*;n���       �	e��Yc�A�*

loss3!�=TQ�       �	[%�Yc�A�*

loss���<K�1�       �	���Yc�A�*

loss��<�?�{       �	���Yc�A�*

loss�.�<����       �	�P�Yc�A�*

loss��=q�FC       �	���Yc�A�*

loss�I=�?�       �	Ɏ�Yc�A�*

lossss�;mͼ�       �	�/�Yc�A�*

loss4�/=\]H�       �	���Yc�A�*

loss�=�4�~       �	n�Yc�A�*

loss�ֵ<Y�v�       �	J�Yc�A�*

loss�ƌ<�C�       �	[��Yc�A�*

lossŶ�=����       �	�m�Yc�A�*

loss�	=�.;       �	%�Yc�A�*

loss1j"=����       �	���Yc�A�*

loss�6�=�xX       �	>�Yc�A�*

loss��)<���       �	���Yc�A�*

loss�N=���       �	[y�Yc�A�*

lossJu�=p�       �	��Yc�A�*

lossS�1=K��;       �	Ψ�Yc�A�*

lossLI=5uI       �	JE�Yc�A�*

loss	��<�!�       �	���Yc�A�*

loss���<?i�I       �	�v�Yc�A�*

lossf��;W��       �	'l�Yc�A�*

loss��-;���4       �	.�Yc�A�*

loss��<���%       �	��Yc�A�*

loss��W<n>��       �	$H�Yc�A�*

loss���=��       �	���Yc�A�*

loss�^�=�-��       �	���Yc�A�*

loss���<��e       �	�2�Yc�A�*

loss�8j<(��       �	���Yc�A�*

loss�	�<�̒�       �	*r�Yc�A�*

loss�u0=,b��       �	��Yc�A�*

loss�G0<ɯ��       �	6��Yc�A�*

loss���;�%�       �	b�Yc�A�*

loss�*�;�ե�       �	_�Yc�A�*

loss�=V���       �	ͮ�Yc�A�*

losslp�;�Ν	       �	�I�Yc�A�*

loss�==Xe�       �	���Yc�A�*

loss<5#<�JKl       �	�x�Yc�A�*

losso��<�        �	l&�Yc�A�*

loss8F�<��՗       �	h]�Yc�A�*

loss�=<L�       �	���Yc�A�*

loss�Y�=��L       �	��Yc�A�*

loss�c<��v       �	R,�Yc�A�*

loss�s:<Uc       �	���Yc�A�*

loss�څ<�&       �	l^�Yc�A�*

loss�n�=��>�       �	��Yc�A�*

lossR�<拧�       �	S��Yc�A�*

loss4�-=� �*       �	bI�Yc�A�*

loss�$=shy       �	���Yc�A�*

loss. 1<�D       �	�|�Yc�A�*

loss�#<���#       �	>>�Yc�A�*

loss$Ȫ<]'�b       �	Q��Yc�A�*

loss�+=rF/�       �	1z�Yc�A�*

loss��='�J       �	��Yc�A�*

lossdn<�^�&       �	V��Yc�A�*

loss?E=�ͤ�       �	�\�Yc�A�*

loss�u<y�6       �	'��Yc�A�*

loss��<���d       �	���Yc�A�*

loss��=���Z       �	W"�Yc�A�*

loss�͆<��ˌ       �	D��Yc�A�*

loss��V<�;       �	S\�Yc�A�*

loss(��<���       �	�&�Yc�A�*

loss��;@e�%       �	s��Yc�A�*

loss�@<߂w       �	;V Yc�A�*

loss%�</�x;       �	� Yc�A�*

loss�2�<2��       �	��Yc�A�*

lossŭ;e�       �	�Yc�A�*

lossx�P=�I�       �	A�Yc�A�*

loss31J=p��)       �	bMYc�A�*

loss�i�=|���       �	:�Yc�A�*

lossM��;�*}       �	��Yc�A�*

loss���;w|<�       �	KYc�A�*

loss/'�<��,H       �	B�Yc�A�*

loss��P=_N"�       �	�Yc�A�*

loss_�m<WJ�       �	�9Yc�A�*

loss���=o�lx       �	T�Yc�A�*

loss�u<C�       �	�Yc�A�*

loss}S>��,       �	�*	Yc�A�*

loss:.�:�̃2       �	%�	Yc�A�*

loss�[�<3��       �	�y
Yc�A�*

loss�>�<D2       �	�Yc�A�*

loss��i;�P2       �	��Yc�A�*

loss�;-=�b��       �	'iYc�A�*

loss8�p=b$g       �	�Yc�A�*

lossT�<�'�       �	�Yc�A�*

lossl�_=���       �	+NYc�A�*

loss��D=/���       �	��Yc�A�*

loss^�=�c(�       �	�{Yc�A�*

lossR�< �X       �	8Yc�A�*

loss]r$<Z�a       �	��Yc�A�*

lossL�%=�qC�       �	@lYc�A�*

lossTU�<Sm�w       �	!Yc�A�*

loss�7�<W��       �	�Yc�A�*

loss�]=���       �	VGYc�A�*

lossͣ�;
�	�       �	i�Yc�A�*

lossϡ�<��e
       �	��Yc�A�*

loss�{n=�ն       �	�Yc�A�*

loss���=3B5Y       �	1�Yc�A�*

lossz~<�Y��       �	�oYc�A�*

loss���;�<��       �	Yc�A�*

loss��<
��       �	v�Yc�A�*

loss�G�=�)�       �	EYc�A�*

lossV�c< "��       �	��Yc�A�*

lossn�j=U#       �	&pYc�A�*

lossY�<62��       �	5^Yc�A�*

loss߻�=��)�       �	�Yc�A�*

loss�E=*��       �	��Yc�A�*

loss�?�<|��       �	�&Yc�A�*

loss�9<��y�       �	#�Yc�A�*

loss:�=�H�       �	�RYc�A�*

lossg�=�NCx       �	��Yc�A�*

loss*�=˺d�       �	��Yc�A�*

loss�wK=\	;       �	�(Yc�A�*

loss�;�<?Q�       �	��Yc�A�*

loss��K<o�-       �	�\ Yc�A�*

lossj��=�3��       �	�� Yc�A�*

loss�|=Z�p       �	��!Yc�A�*

loss�į<3]c�       �	�"Yc�A�*

loss׷�<��       �	�"Yc�A�*

loss`�/<\[-\       �	}�#Yc�A�*

loss�L<�O��       �	s�$Yc�A�*

loss��h<#m-�       �	�%Yc�A�*

loss=ܕ=���0       �	!�%Yc�A�*

losslG�=��S�       �	�E&Yc�A�*

lossI�<+���       �	��&Yc�A�*

loss�{�='�0�       �	�s'Yc�A�*

loss�;�Or       �	(Yc�A�*

lossh�k<��u       �	@�(Yc�A�*

loss(e�=V�-�       �	�<)Yc�A�*

loss[�5<a���       �	��)Yc�A�*

loss�GW=`
       �	f�*Yc�A�*

lossV<�<e�6       �	\+Yc�A�*

loss�{�=��>C       �	��+Yc�A�*

loss
�.<��_�       �	�\,Yc�A�*

loss���;�� �       �	"q-Yc�A�*

loss��y=�.J�       �	�.Yc�A�*

loss2�)=�d4�       �		�.Yc�A�*

loss[�=h��_       �	)�/Yc�A�*

loss� �<jZ�X       �	�B0Yc�A�*

loss�e='���       �	p�0Yc�A�*

lossw�/=qwWB       �	Z1Yc�A�*

loss�d�<ӌ�       �	�B2Yc�A�*

loss\G�<����       �	+�2Yc�A�*

loss1k�<%��       �	2r3Yc�A�*

lossŻ<(
I       �	�4Yc�A�*

loss���<,�&�       �	V�4Yc�A�*

lossxݹ<B�.7       �	�@5Yc�A�*

loss�wB=ݯ̲       �	b�5Yc�A�*

losst��=��=       �	�n6Yc�A�*

loss$�<�X>       �	�7Yc�A�*

loss�T�<��G       �	��7Yc�A�*

lossp�<���       �	ao8Yc�A�*

loss�_T=$�|�       �	
9Yc�A�*

loss#��<{�$T       �	�9Yc�A�*

loss�ʲ<uh��       �	�E:Yc�A�*

loss�#�=�2-u       �	��:Yc�A�*

loss�H<�5�       �	cz;Yc�A�*

lossvl;&D��       �	�<Yc�A�*

losse�;�Z��       �	>�<Yc�A�*

loss���<��j       �	�L=Yc�A�*

lossEd<����       �	G�=Yc�A�*

loss(�I=;k�v       �	�>Yc�A�*

loss:�=5�d�       �	t'?Yc�A�*

loss0D=�!(       �	��?Yc�A�*

loss:�<Ȱ�       �	�[@Yc�A�*

loss]L{=3�b       �	��@Yc�A�*

loss���<´5       �	�AYc�A�*

loss���<�}�R       �	�"BYc�A�*

lossat�<��       �	ܹBYc�A�*

loss�ן=c�       �	�NCYc�A�*

loss2�=�#;�       �	��CYc�A�*

loss"9<����       �	��DYc�A�*

loss�%�<�S>�       �	�7EYc�A�*

loss���;�M=       �	��EYc�A�*

loss!��:�`�       �	jFYc�A�*

loss���=�]�{       �	hGYc�A�*

loss$[<��B       �	ʤGYc�A�*

loss���<�G��       �	:<HYc�A� *

loss�K]=g6}       �	��HYc�A� *

loss�ء<� �       �	�mIYc�A� *

loss͐=����       �	�JYc�A� *

loss�#J=�ι�       �	�&KYc�A� *

loss��/=6/H&       �	�KYc�A� *

loss��<<Bxυ       �	�VLYc�A� *

loss��<�7,       �	�MYc�A� *

lossnB<r	r       �	��MYc�A� *

loss�w�<y<J       �	��NYc�A� *

loss��V<գ�       �	l�OYc�A� *

lossnT�<�)�n       �	��PYc�A� *

loss]�<�1       �	�mQYc�A� *

lossM=��(u       �	�RYc�A� *

lossW= ���       �	>�RYc�A� *

loss��<~�Ý       �	0bSYc�A� *

loss|��=+
a       �	/�SYc�A� *

loss��g<t��       �	`�TYc�A� *

loss���<�",�       �	W{UYc�A� *

loss�{�<��_�       �	XVYc�A� *

loss��<# u       �	-�VYc�A� *

loss8mN=" 01       �	sIWYc�A� *

loss��.<�@8[       �	��WYc�A� *

loss��^;�g/[       �	4�XYc�A� *

lossC��<�Ǝ       �	�8YYc�A� *

loss<��<����       �	��YYc�A� *

loss�d�<x���       �	�^ZYc�A� *

loss䉃;T���       �	��ZYc�A� *

lossz�S<c��       �	��[Yc�A� *

loss�H<QR��       �	/�\Yc�A� *

loss(��;: ��       �	(]Yc�A� *

loss�k�;6yLF       �	��]Yc�A� *

loss��F=�G�       �	~�^Yc�A� *

losssR�=�[uD       �	77_Yc�A� *

loss{��;[K�)       �	��_Yc�A� *

loss��<W��_       �	�z`Yc�A� *

loss`�#<���       �	!aYc�A� *

lossR��:��1       �	%�aYc�A� *

loss�J�:^�L       �	�ubYc�A� *

lossQl�<͐       �	ucYc�A� *

loss��<qeVS       �	o�cYc�A� *

loss\b�<{�.�       �	�dYc�A� *

lossqYD<�+       �	XeYc�A� *

lossڕd:��       �	��eYc�A� *

loss:Q�<���E       �	,cfYc�A� *

loss�+;g"��       �	VgYc�A� *

loss�aB;��kR       �	��gYc�A� *

loss=�9���m       �	�QhYc�A� *

loss#T;*��       �	t�hYc�A� *

lossT��;~\       �	��iYc�A� *

losslf<e�w       �	d;jYc�A� *

lossxq�:Dd��       �	��jYc�A� *

lossE��;j�B[       �	��kYc�A� *

lossV��=�br[       �	�WlYc�A� *

lossa�)<ǁ       �	mYc�A� *

loss�K�<��       �	�mYc�A� *

loss��2=/&y0       �	QKnYc�A� *

lossO03=���g       �	�nYc�A� *

lossO�6=�[�\       �	ėoYc�A� *

lossEXB<1�?A       �	�IpYc�A� *

lossIa�=tg�D       �	%$qYc�A� *

lossߑD=�iF�       �	p�qYc�A� *

loss�=;Ֆ}       �	vrYc�A� *

lossfj�<-�ϳ       �	K#sYc�A� *

lossȦi;A�F�       �	��sYc�A� *

loss��s=��˶       �	�etYc�A� *

loss�5X=Of�       �	�
uYc�A� *

loss�=�N{�       �	��uYc�A� *

loss��y<(�c       �	!YvYc�A� *

loss�'=��       �	; wYc�A� *

loss�k�<Xn�       �	�wYc�A� *

loss�+G<7�a�       �	�LxYc�A� *

loss4�=v��       �	��xYc�A� *

lossd�<'�       �	h�yYc�A� *

loss(��;�t�>       �	8zYc�A� *

loss�+T=4��2       �	��zYc�A� *

loss}�<��l~       �	%z{Yc�A� *

lossȾ�;��o*       �	y$|Yc�A� *

loss��I<���$       �	"�|Yc�A� *

loss� f;�W�       �	%]}Yc�A� *

lossT
�<�\0q       �	�~Yc�A� *

lossy�;D	7�       �	3�~Yc�A� *

loss!fQ=�~�       �	iSYc�A� *

loss�+�=kܢO       �	��Yc�A� *

loss4=`ҪI       �	��Yc�A� *

lossH��<!��u       �	z5�Yc�A� *

loss��b;_�:�       �	tҁYc�A� *

loss��;Iv��       �	tz�Yc�A� *

loss�K<]�j�       �	n�Yc�A� *

loss݄�<�Š       �	2��Yc�A� *

loss ��;Fi�D       �	�G�Yc�A� *

loss
yK<����       �	zބYc�A� *

loss&�=���       �	�|�Yc�A� *

loss���<�F�k       �	���Yc�A� *

loss�":<�?r�       �	���Yc�A� *

loss�(�<���       �	��Yc�A� *

lossm{�<��[~       �	9�Yc�A� *

loss	��<���~       �	�ԉYc�A� *

loss�	<F�       �	�n�Yc�A� *

loss׀8=HN       �	��Yc�A� *

lossv$=b���       �	�؋Yc�A� *

loss\��:�y
�       �	:x�Yc�A� *

loss�.=��!H       �	j�Yc�A� *

loss��;�8�D       �	��Yc�A� *

loss*k;�`�5       �	.��Yc�A� *

loss7�;+v�|       �	�Yc�A� *

loss��< y       �	"��Yc�A� *

loss�P=+[��       �	�I�Yc�A� *

loss��=�u�k       �	�Yc�A� *

loss�P=ۏ{�       �	�}�Yc�A� *

loss��=��8       �	��Yc�A� *

loss#�<	)       �	���Yc�A� *

loss��<n
D�       �	YQ�Yc�A� *

lossVI�=:�       �	H��Yc�A� *

lossc}Q=R�y�       �	h��Yc�A� *

loss8k�;���       �	/�Yc�A� *

loss,_2=��>       �	�ԫYc�A� *

loss��L=QF
�       �	�m�Yc�A� *

loss���<����       �	l�Yc�A� *

lossd�o=xț|       �	��Yc�A� *

lossR�^=��B�       �	n5�Yc�A� *

loss���;��F       �	�ծYc�A� *

loss0G;���5       �	��Yc�A� *

loss� -=Ůx�       �	Q۰Yc�A� *

loss�t=�onW       �	�ǱYc�A� *

loss]]F=W��Z       �	ۈ�Yc�A� *

losse�=��pi       �	�f�Yc�A� *

loss�a<�=z�       �	Z��Yc�A� *

loss�=�=a)*�       �	S��Yc�A�!*

loss:�K=vm�V       �	�|�Yc�A�!*

loss��=�O��       �	·Yc�A�!*

loss!��<ׯ��       �	LǸYc�A�!*

loss&(x;��e_       �	�^�Yc�A�!*

loss��8=��t       �	���Yc�A�!*

loss4�X<6��i       �	�F�Yc�A�!*

loss�e�<�)Q�       �	��Yc�A�!*

loss=����       �	4��Yc�A�!*

lossf�@=MM��       �	�D�Yc�A�!*

loss��=}���       �	l�Yc�A�!*

loss���<A�       �	�ȾYc�A�!*

loss%�L=h���       �	`�Yc�A�!*

loss��;+��       �	,�Yc�A�!*

lossJ�<�&#)       �	2��Yc�A�!*

loss84�=E�g�       �	���Yc�A�!*

loss�!�=H�c(       �	wg�Yc�A�!*

loss (�<^�	�       �	7T�Yc�A�!*

loss��<���       �	��Yc�A�!*

lossQ,�;е��       �	���Yc�A�!*

loss�`v<lG�)       �	�0�Yc�A�!*

loss�B�<r�W       �	l��Yc�A�!*

loss���=l��       �	ip�Yc�A�!*

loss�%B=L��       �	#�Yc�A�!*

loss�Bi<+��       �	>��Yc�A�!*

loss�G=�ZC�       �	�W�Yc�A�!*

loss��:��U       �	(��Yc�A�!*

loss(-=D0�       �	Y��Yc�A�!*

loss��=&��       �	��Yc�A�!*

loss��<��B0       �	��Yc�A�!*

loss��=�ʫ�       �	�W�Yc�A�!*

lossM�j=| 2?       �	���Yc�A�!*

loss���;FX��       �	'��Yc�A�!*

loss!��;ʽ�8       �	�&�Yc�A�!*

lossؼ;��Rr       �	���Yc�A�!*

loss��H<����       �	^�Yc�A�!*

loss�q�<,&�       �	�Yc�A�!*

loss(=����       �	���Yc�A�!*

loss��=�÷�       �	eU�Yc�A�!*

lossF��<��       �	w��Yc�A�!*

lossӉ9<&���       �	��Yc�A�!*

loss��=6q��       �	�D�Yc�A�!*

loss�6=o��<       �	���Yc�A�!*

loss/�<�dd       �	���Yc�A�!*

loss�� <Zi�       �	�7�Yc�A�!*

loss��<�"x       �	���Yc�A�!*

loss�O�=�@S�       �	���Yc�A�!*

loss���<Ɛ-       �	�#�Yc�A�!*

loss� =�)       �	!��Yc�A�!*

loss�F�<�^R@       �	g�Yc�A�!*

loss�y=�l       �	:�Yc�A�!*

loss��R=JN>�       �	��Yc�A�!*

loss�|�<Y���       �	T5�Yc�A�!*

lossר;v���       �	���Yc�A�!*

loss$S�;`�E       �	Po�Yc�A�!*

lossZr<@�       �	 �Yc�A�!*

loss�!=ޫ�       �	R��Yc�A�!*

loss&�=d���       �	j�Yc�A�!*

loss��<�       �	.��Yc�A�!*

loss�5Y='T�       �	��Yc�A�!*

loss��r;@�NL       �	H�Yc�A�!*

lossF#1=D��D       �	��Yc�A�!*

loss��<=o�7R       �	=}�Yc�A�!*

loss�T�<�{{       �	g}�Yc�A�!*

loss���=,B((       �	��Yc�A�!*

loss$�=�#�       �	q��Yc�A�!*

loss$\E<)�.l       �	�}�Yc�A�!*

loss��<�(�?       �	�Yc�A�!*

lossSv=i�       �	���Yc�A�!*

loss3�;�9       �	�C�Yc�A�!*

loss'I�<�`�       �	m��Yc�A�!*

lossd�<=KJ�<       �	���Yc�A�!*

loss�:<��;H       �	Z-�Yc�A�!*

lossLq�=C|�`       �	���Yc�A�!*

loss��<�/l       �	3��Yc�A�!*

loss��#=y鯧       �	'1�Yc�A�!*

loss�5�<ӯ�S       �	���Yc�A�!*

loss��;_s�s       �	b�Yc�A�!*

lossɒ�;͐Oj       �	�
�Yc�A�!*

loss��R<�Hϳ       �	5��Yc�A�!*

lossfzC=�33i       �	+M�Yc�A�!*

loss,��<y�՘       �	=��Yc�A�!*

lossT�}<�u�N       �	%��Yc�A�!*

loss��<1��+       �	v8�Yc�A�!*

loss��$<\ƭ       �	 ��Yc�A�!*

loss�u3=a ��       �	|�Yc�A�!*

loss��<r���       �	�Yc�A�!*

loss�3=�`�       �	@�Yc�A�!*

loss�hZ=Vy�       �	���Yc�A�!*

lossr�j<���)       �	z�Yc�A�!*

loss3r?;�6��       �	1@�Yc�A�!*

loss�,�;�X7.       �	���Yc�A�!*

loss���;�@4�       �	ӽ�Yc�A�!*

lossxhW<m܏L       �	�X�Yc�A�!*

loss|�=���       �	 ��Yc�A�!*

loss��5=�% �       �	]��Yc�A�!*

loss7-<�t�       �	B�Yc�A�!*

lossX=ןQ,       �	)&�Yc�A�!*

lossi��<c��       �	���Yc�A�!*

loss�;c�7       �	.p�Yc�A�!*

loss��<�g�       �	��Yc�A�!*

loss��;&u��       �	���Yc�A�!*

loss�ӊ<F�C�       �	�E�Yc�A�!*

loss�C�<.��       �	���Yc�A�!*

loss{o<,��x       �	|�Yc�A�!*

lossd
=P8��       �	K�Yc�A�!*

lossxh|<��       �	d��Yc�A�!*

lossr;<ߚNm       �	{�Yc�A�!*

loss`�H=���       �	 Yc�A�!*

loss��<.X��       �	� Yc�A�!*

loss��<n��C       �	�KYc�A�!*

loss���=)�i<       �	(Yc�A�!*

loss�b=��,       �	�Yc�A�!*

loss��O=@��       �	�Yc�A�!*

loss�?�<��<�       �	�Yc�A�!*

lossw��<Rt�g       �	NYc�A�!*

loss]�<P��       �	��Yc�A�!*

lossK	;��       �	�|Yc�A�!*

loss�U;]N�B       �	"Yc�A�!*

loss�+=���       �	��Yc�A�!*

loss��S=i���       �	;�	Yc�A�!*

loss�v�<h�,d       �	'
Yc�A�!*

loss4�X<T�       �	Z�
Yc�A�!*

loss���<�k��       �	-�Yc�A�!*

loss�/<2�       �	AYc�A�!*

lossְ;���       �	s�Yc�A�!*

loss2�c<��LB       �	�sYc�A�!*

loss���;.��%       �	�XYc�A�!*

loss��<�ML.       �	��Yc�A�"*

lossɐ=^���       �	v�Yc�A�"*

loss���<lSH�       �	� Yc�A�"*

lossF�D=�W��       �	��Yc�A�"*

loss?IF<�z�       �	UhYc�A�"*

lossME<�[�i       �	IYc�A�"*

loss�^�=e-�q       �	�Yc�A�"*

loss!c�=�k�B       �	h[Yc�A�"*

loss|�h<+2       �	L�Yc�A�"*

lossHN<���       �	=�Yc�A�"*

loss'T<[�P�       �	�CYc�A�"*

loss�ȱ;+û       �	��Yc�A�"*

loss�
�:�g�`       �	�wYc�A�"*

loss�2=���       �	�Yc�A�"*

loss=C<{m8]       �	��Yc�A�"*

loss<�;*K�       �	sLYc�A�"*

lossc�Y=\	�Q       �	uYc�A�"*

lossl��;ޝ       �	E�Yc�A�"*

lossM7�<Fl��       �	iQYc�A�"*

loss�|=[C4�       �	��Yc�A�"*

loss��q;� �       �	��Yc�A�"*

loss4d=X��       �	nYc�A�"*

lossx�=��       �	�Yc�A�"*

loss���;5!��       �	�BYc�A�"*

loss7c�=��*       �	��Yc�A�"*

loss�Ď<R�V�       �	\sYc�A�"*

loss��x=�D�,       �	�
Yc�A�"*

lossm��<G%�       �	��Yc�A�"*

loss3��<�GZ       �	�D Yc�A�"*

loss7�;�/�       �	v� Yc�A�"*

loss��<YMs>       �	 �!Yc�A�"*

loss�k�<��a       �	�+"Yc�A�"*

loss�=� ��       �	��"Yc�A�"*

lossv<l���       �	�s#Yc�A�"*

loss��q<�       �	�
$Yc�A�"*

lossJ��<�o#       �	"�$Yc�A�"*

loss�+<���       �	�F%Yc�A�"*

loss}��<f;�-       �	��%Yc�A�"*

loss��c<���       �	��&Yc�A�"*

loss���<}m       �	cA'Yc�A�"*

loss�s<<<b�D       �	�'Yc�A�"*

loss�?<��       �	�~(Yc�A�"*

lossA<�3�       �	m)Yc�A�"*

loss�|�;�&�       �	й)Yc�A�"*

loss�|
<
Ҝ       �	.V*Yc�A�"*

lossm�<�]�       �	+Yc�A�"*

loss�a�<TE��       �	 �+Yc�A�"*

lossaС;�3oI       �	:W,Yc�A�"*

lossR��<<�0�       �	0�,Yc�A�"*

loss:�.=6���       �	-�-Yc�A�"*

loss=�=�r       �	YR.Yc�A�"*

lossWCG<r��       �	[�.Yc�A�"*

loss�=ܧ�c       �	�/Yc�A�"*

lossw��<P�?�       �	�0Yc�A�"*

loss��'<��)       �	�f1Yc�A�"*

loss�a[<2|�       �	�2Yc�A�"*

loss��U=�       �	ݲ2Yc�A�"*

loss:9=�^�A       �	�U3Yc�A�"*

lossT`�<����       �	��3Yc�A�"*

loss3A<� l       �	!�4Yc�A�"*

loss�h<d���       �	�-5Yc�A�"*

loss���<���       �	��5Yc�A�"*

lossm�/<H�ܿ       �	�~6Yc�A�"*

lossN`�<�_0�       �	*7Yc�A�"*

loss��=+�b�       �	}�7Yc�A�"*

loss�X�=��v�       �	=�8Yc�A�"*

loss�Y=&Cg       �	�"9Yc�A�"*

loss�Ї=I��       �	��9Yc�A�"*

loss�h#=����       �	n:Yc�A�"*

loss��<��Q�       �	`;Yc�A�"*

loss�!<: �       �	ǜ;Yc�A�"*

lossM��<ȹZ�       �	�6<Yc�A�"*

loss v�<�d       �	��<Yc�A�"*

lossm1�<5�m�       �	�q=Yc�A�"*

loss��<���,       �	T>Yc�A�"*

loss��<&k��       �	��>Yc�A�"*

loss���<�5       �	ʉ?Yc�A�"*

loss��<�X��       �	�@Yc�A�"*

loss;<Dgۉ       �	��@Yc�A�"*

loss��w<�j��       �	.YAYc�A�"*

loss�n<�"_       �	��AYc�A�"*

lossJ��;�и�       �	@�BYc�A�"*

loss\+�<	[Q       �	�;CYc�A�"*

loss�
�<E�*       �	��CYc�A�"*

loss�Q<�:s�       �	b�DYc�A�"*

loss��<z(7       �	c'EYc�A�"*

lossQ��=�㺨       �	x�EYc�A�"*

loss�Ʉ=���L       �	�}FYc�A�"*

lossM��<��r�       �	*GYc�A�"*

loss�f�<fU#@       �	��GYc�A�"*

loss�A=�%       �	�nHYc�A�"*

loss�Y�<ط�       �	�IYc�A�"*

loss{�=1�|�       �	�IYc�A�"*

loss��p<^1�       �	IJJYc�A�"*

loss�۳<�w�       �	-�JYc�A�"*

loss�;��5�       �	U�KYc�A�"*

loss&Wo<#�T}       �	�3LYc�A�"*

loss{c�<h<��       �	��LYc�A�"*

loss��y<��:�       �	�vMYc�A�"*

loss-��<�մ       �	!NYc�A�"*

lossl}�=:IY�       �	e�NYc�A�"*

loss4S�<���]       �	5^OYc�A�"*

loss���;��f,       �	�PYc�A�"*

loss��=;F3Ǌ       �	��PYc�A�"*

loss���<]j��       �	$EQYc�A�"*

loss�ڣ<�v       �	2�RYc�A�"*

loss�LC=kh'       �	=aSYc�A�"*

lossA	h<Д�H       �		�SYc�A�"*

lossi��<	�Ս       �	��TYc�A�"*

loss$�<�pB�       �	�eUYc�A�"*

loss��=�f       �	�VYc�A�"*

loss�i<.t        �	��VYc�A�"*

lossX`�<�>�       �	�FWYc�A�"*

loss.J9=ז       �	��WYc�A�"*

lossL�;�6�       �	�yXYc�A�"*

loss��#=�a�5       �	:YYc�A�"*

loss�%=ro�       �	��YYc�A�"*

loss�
=�
�y       �	yZYc�A�"*

loss�+=dQP       �	[Yc�A�"*

loss�<�̹       �	�[Yc�A�"*

loss*֥<)'�       �	9a\Yc�A�"*

lossY(=e��       �	��\Yc�A�"*

loss�oH<oQg�       �	��]Yc�A�"*

loss���=r��!       �	�[^Yc�A�"*

loss���<���       �	��^Yc�A�"*

loss7��<��U       �	��_Yc�A�"*

loss�<�� �       �	�8`Yc�A�"*

loss �<�C�Z       �	��`Yc�A�"*

loss-)#=�gn       �	�laYc�A�#*

loss2E<j���       �	�4bYc�A�#*

loss��<1&�       �	�bYc�A�#*

loss��E<��CK       �	�ucYc�A�#*

loss�^�<�ح�       �	�dYc�A�#*

loss��I=p���       �	�dYc�A�#*

loss�t�<	(P       �	[eYc�A�#*

lossV)c=7�p�       �	�fYc�A�#*

lossT�=<��-�       �	�fYc�A�#*

loss�Z�;��       �	igYc�A�#*

loss�d�:(�d       �	}	hYc�A�#*

loss�^�;���!       �	C�hYc�A�#*

lossx��<��       �	+NiYc�A�#*

loss�Z-;�9N       �	-�iYc�A�#*

loss<��<�8��       �	�jYc�A�#*

loss�u�<&:`�       �	`;kYc�A�#*

lossQ�=��)(       �	��kYc�A�#*

loss�>�<�8Ie       �	ΌlYc�A�#*

loss�*=J)�       �	�-mYc�A�#*

losszh=SuYy       �	x�mYc�A�#*

loss��<A��}       �	OvnYc�A�#*

loss�9�=i��       �	� oYc�A�#*

loss_ٶ;/�)!       �	��oYc�A�#*

loss���;8j�^       �	�apYc�A�#*

loss=s�<���       �	*�pYc�A�#*

loss
�[=i�·       �	i�rYc�A�#*

lossh�^=q0$�       �	��sYc�A�#*

lossl��;:�H       �	�XtYc�A�#*

loss�k=��m.       �	�TuYc�A�#*

lossiTN=r���       �	�vYc�A�#*

loss�v=X�(        �	��vYc�A�#*

loss�s7<歮�       �	O[wYc�A�#*

loss�B�<bW�       �	�xYc�A�#*

lossZF=�XxU       �	ϡxYc�A�#*

loss[��;W��       �	v�yYc�A�#*

loss�=<�&��       �	0IzYc�A�#*

lossV~
=�]�       �	u�zYc�A�#*

loss
s3;��f�       �	��{Yc�A�#*

loss�P�<��       �	�-|Yc�A�#*

loss��<OHI       �	hZ}Yc�A�#*

loss3��=��       �	��}Yc�A�#*

lossO�4=@�$       �	��~Yc�A�#*

loss�H=�m       �	AJYc�A�#*

loss�w<j]       �	��Yc�A�#*

loss��;@�o\       �	��Yc�A�#*

loss
�;D��'       �	��Yc�A�#*

lossϘ�<y�L       �	��Yc�A�#*

loss��'<%n2�       �	�g�Yc�A�#*

loss��<<X_�       �	J�Yc�A�#*

loss�a�=�$��       �	��Yc�A�#*

loss6�=����       �	�L�Yc�A�#*

loss�C&=��m�       �	;��Yc�A�#*

loss���;��!       �	��Yc�A�#*

loss��t<:��>       �	�J�Yc�A�#*

loss2F<�(�       �	��Yc�A�#*

loss�Ok=1W�L       �	䅇Yc�A�#*

loss� =��       �	O$�Yc�A�#*

loss�o<��R       �	Yc�A�#*

lossQ/�<�.�o       �	^e�Yc�A�#*

lossy<FH,�       �	��Yc�A�#*

loss���:���<       �	���Yc�A�#*

loss���<�"�       �	U�Yc�A�#*

loss�MK<m�U�       �	R�Yc�A�#*

loss��;#�/�       �	���Yc�A�#*

lossE_�<��n�       �	M/�Yc�A�#*

lossi�j=w#@,       �	�ЍYc�A�#*

loss&?=�-�       �	�v�Yc�A�#*

lossm�<A�s3       �	��Yc�A�#*

loss;u�<"'=^       �	��Yc�A�#*

loss��<��}       �	�_�Yc�A�#*

loss�~�;Vc�       �	�	�Yc�A�#*

loss�0�<P��       �	�9�Yc�A�#*

lossJ�2<l [A       �	��Yc�A�#*

lossD��<�Ȩ�       �	K��Yc�A�#*

loss:��;��]�       �	�5�Yc�A�#*

loss�W�=?��       �	�הYc�A�#*

loss���;�E&       �	�Yc�A�#*

loss��;LH��       �	�!�Yc�A�#*

loss�\[<�s��       �	���Yc�A�#*

loss�b�;�Q�       �	p]�Yc�A�#*

loss�*�<H��       �	 �Yc�A�#*

losslu�<�y��       �	��Yc�A�#*

lossJ��<��       �	Ql�Yc�A�#*

loss}IT=�L��       �	��Yc�A�#*

loss	�'=N��       �	r��Yc�A�#*

lossұ.=nE3W       �	�F�Yc�A�#*

loss��]=���j       �	��Yc�A�#*

loss�N=���       �	��Yc�A�#*

loss�u<��Vo       �	T�Yc�A�#*

loss���<Ps�       �	x��Yc�A�#*

loss!�<)�       �	W�Yc�A�#*

loss�f�<�8(�       �	�S�Yc�A�#*

loss�(T<��       �	k�Yc�A�#*

loss�nM<Z�)       �	jߠYc�A�#*

loss�<�w�       �	�y�Yc�A�#*

loss��<(��8       �	:�Yc�A�#*

loss,[<�:       �	���Yc�A�#*

loss���<�ʑ       �	@N�Yc�A�#*

loss�Kg<�]ά       �	�2�Yc�A�#*

loss�=��Ϫ       �	�դYc�A�#*

loss<�<�ƀ       �	���Yc�A�#*

loss}m'<�}��       �	�*�Yc�A�#*

loss�X=�]L5       �	K˦Yc�A�#*

loss3I�<)%��       �	�t�Yc�A�#*

loss[\�=�һ�       �	��Yc�A�#*

loss���<(��_       �	ܡ�Yc�A�#*

lossH�N=Σv|       �	ND�Yc�A�#*

loss�[�<?M�_       �	��Yc�A�#*

lossFkD<��i       �	M��Yc�A�#*

loss`�p<2L��       �	O#�Yc�A�#*

loss�n+<�@��       �	#��Yc�A�#*

lossv};���`       �	�b�Yc�A�#*

lossx��<�V       �	L��Yc�A�#*

loss
K�<�QS�       �	X��Yc�A�#*

loss�[�<#(Z�       �	�E�Yc�A�#*

loss��<��l       �	s��Yc�A�#*

loss�-5=�/��       �	���Yc�A�#*

loss`h}:�̡�       �	�(�Yc�A�#*

loss;R=0       �	�°Yc�A�#*

loss�"�<��       �	�h�Yc�A�#*

loss��W=�k��       �	L5�Yc�A�#*

loss?�<#�       �	��Yc�A�#*

loss��S<�[f$       �	
��Yc�A�#*

loss��<�       �	c(�Yc�A�#*

lossc�=���       �	:ʴYc�A�#*

loss@&=�{lL       �	j�Yc�A�#*

loss\-�<��       �	��Yc�A�#*

loss�B�<l���       �	�ŷYc�A�#*

loss�۩<���       �	$d�Yc�A�$*

loss�[&<��l}       �	V��Yc�A�$*

loss��?=�J       �	x_�Yc�A�$*

lossfZH<�6�I       �	���Yc�A�$*

loss��<��        �	7��Yc�A�$*

loss P�=� b       �	�#�Yc�A�$*

loss�Ͷ<��3�       �	�Yc�A�$*

loss�d;=ɛ��       �	�R�Yc�A�$*

loss�V�==�q�       �	m��Yc�A�$*

loss��:;'�        �	ݖ�Yc�A�$*

loss�6 <��CB       �	I0�Yc�A�$*

loss��a=+���       �	ϿYc�A�$*

loss��=F�|B       �	"r�Yc�A�$*

loss�L=���       �	8�Yc�A�$*

loss��=@g?�       �	q��Yc�A�$*

loss�:�:��2�       �	�K�Yc�A�$*

loss)b�=sy՘       �	:��Yc�A�$*

loss�e�<N4{�       �	���Yc�A�$*

loss�5t=��cg       �	��Yc�A�$*

lossD��<"-��       �	��Yc�A�$*

loss�s@=�V)6       �	�{�Yc�A�$*

loss\�+<�3�       �	��Yc�A�$*

loss��<) Qj       �	��Yc�A�$*

loss#��<51�       �	�x�Yc�A�$*

loss��'<w��       �	��Yc�A�$*

loss:ӛ<�?�       �	 ��Yc�A�$*

loss��6<mR��       �	�}�Yc�A�$*

loss��=���,       �	_B�Yc�A�$*

loss1��;h~�       �	z�Yc�A�$*

loss�W<����       �	��Yc�A�$*

loss�C�<�Ѯ�       �	UN�Yc�A�$*

loss!�;����       �	��Yc�A�$*

loss��<�[5�       �	�`�Yc�A�$*

loss��=�D��       �	
�Yc�A�$*

loss-m=ǤA(       �	��Yc�A�$*

loss�4D<��RS       �	LR�Yc�A�$*

loss��=����       �	İ�Yc�A�$*

loss��;[�G�       �	wI�Yc�A�$*

loss�d<<rp       �	���Yc�A�$*

lossߦA=��8       �	�x�Yc�A�$*

loss�TE<p*�       �	��Yc�A�$*

loss]Ƣ<7�{Z       �	?��Yc�A�$*

loss��=��4:       �	Gt�Yc�A�$*

loss]�<�ȩ?       �	��Yc�A�$*

loss�<<�W�N       �	-��Yc�A�$*

loss3��;Ifi       �	�h�Yc�A�$*

lossn;�<�EV       �	��Yc�A�$*

loss��='RDs       �	��Yc�A�$*

lossء=�\��       �	�E�Yc�A�$*

loss#��;��3       �	���Yc�A�$*

loss�5=�C�W       �	�v�Yc�A�$*

lossc�=a�i       �	��Yc�A�$*

lossqX#=�g�/       �	]��Yc�A�$*

loss.��<���       �	Vf�Yc�A�$*

loss-r�;qu       �	��Yc�A�$*

loss�b�;Uj       �	���Yc�A�$*

loss�0a<��       �	�b�Yc�A�$*

loss�~�<h� �       �	e��Yc�A�$*

loss:p�<ۓ.       �	���Yc�A�$*

loss�"�< j       �	�-�Yc�A�$*

lossz5S;�&�       �	*��Yc�A�$*

loss �;A=h�       �	�i�Yc�A�$*

loss�� <-~��       �	u �Yc�A�$*

loss�=:�t       �	��Yc�A�$*

loss�z�<��.       �	,b�Yc�A�$*

loss�x�<}�_�       �	P��Yc�A�$*

loss�!\=��ՠ       �	��Yc�A�$*

loss�	�;�n;x       �	�4�Yc�A�$*

loss[��:��        �	s.�Yc�A�$*

loss7b�;��       �	���Yc�A�$*

lossP�<+���       �	�{�Yc�A�$*

loss<�<s�*�       �	��Yc�A�$*

loss��t=[��?       �	���Yc�A�$*

loss}v=;Ѣ�       �	�n�Yc�A�$*

lossZۼ;��l       �	}�Yc�A�$*

loss�NP;	w\	       �	^��Yc�A�$*

loss_��=�^�A       �	v5�Yc�A�$*

loss�'�;a� N       �	-��Yc�A�$*

loss˿�<���~       �	�d�Yc�A�$*

loss�D�<Q0��       �	���Yc�A�$*

loss�&#=5
�       �	P��Yc�A�$*

loss�%�;]�1&       �	lA�Yc�A�$*

loss�=v���       �	���Yc�A�$*

loss�:�;2�.�       �	��Yc�A�$*

loss�_�<�v�       �	��Yc�A�$*

loss��<���       �	�8�Yc�A�$*

lossl=�&Ǆ       �	@�Yc�A�$*

loss;��Fq       �	E�Yc�A�$*

loss��K<��b=       �	���Yc�A�$*

losso�&<S�X       �	���Yc�A�$*

lossc�<��nb       �	��Yc�A�$*

loss!P= ���       �	�T�Yc�A�$*

loss�Y�<L9Z       �	���Yc�A�$*

lossʹY=0�S�       �	u��Yc�A�$*

loss$S;߈��       �	�.�Yc�A�$*

loss��;����       �	;��Yc�A�$*

loss�T;4m�       �	5]�Yc�A�$*

loss�<� ~       �	L��Yc�A�$*

loss�~Q<�q�       �	t��Yc�A�$*

loss�BR<sٔ�       �	p@�Yc�A�$*

loss;,<��       �	���Yc�A�$*

loss\ȸ=�xk       �	�p�Yc�A�$*

loss4�<3؜�       �	d�Yc�A�$*

loss��<�d       �	��Yc�A�$*

loss��< ��o       �	�>�Yc�A�$*

loss��;�T��       �	���Yc�A�$*

loss$EM<^        �	
� Yc�A�$*

loss��<���       �	)>Yc�A�$*

loss!Cw=�+��       �	��Yc�A�$*

loss�֯<FyR�       �	�tYc�A�$*

loss�Ǯ<jK�       �	�
Yc�A�$*

loss���;���       �	@�Yc�A�$*

lossj<�`S       �	�_Yc�A�$*

loss�Lu<���       �	v�Yc�A�$*

lossE�<r��       �	�Yc�A�$*

loss�O:�#�       �	�PYc�A�$*

loss�*=��       �	��Yc�A�$*

loss��9<EM�"       �	�Yc�A�$*

loss3��<�lxa       �	�Yc�A�$*

loss���<�~�       �	��Yc�A�$*

loss��:<���       �	|	Yc�A�$*

lossp$=�g,�       �	F~Yc�A�$*

lossz{.=;        �	 Yc�A�$*

lossB ;�S��       �	��Yc�A�$*

loss9�<�Js       �	gYc�A�$*

loss1��;N���       �	5Yc�A�$*

lossg�:�wܿ       �	��Yc�A�$*

loss� /=�{�       �	�:Yc�A�$*

loss�Y<��c�       �	(�Yc�A�%*

loss�.�;��       �	�Yc�A�%*

loss�z<��"       �	�<Yc�A�%*

loss~�9&xp?       �	�Yc�A�%*

loss��&<v�|
       �	�uYc�A�%*

loss���:�A}Y       �	�6Yc�A�%*

loss�>F9��>�       �	�Yc�A�%*

loss��8k"�       �	�vYc�A�%*

losshL$;��u�       �	Yc�A�%*

lossr��<9_(       �	�Yc�A�%*

loss�4<��bY       �	6VYc�A�%*

loss��:��8�       �	��Yc�A�%*

loss�'�<�yfB       �	(�Yc�A�%*

loss�X=�9�       �	�;Yc�A�%*

loss�f�:Y�?�       �	ȵYc�A�%*

lossd.M<���       �	�ZYc�A�%*

loss�"=�?       �	�Yc�A�%*

lossT{=`/V;       �	�Yc�A�%*

loss?�=���       �	\<Yc�A�%*

loss$�<|�       �	?�Yc�A�%*

lossW?=�E�       �	��Yc�A�%*

loss�5)=o�V�       �	�)Yc�A�%*

loss�H=�{J       �	�Yc�A�%*

lossO�<֦��       �	�hYc�A�%*

loss�; �4�       �	0 Yc�A�%*

lossز<XV��       �	>� Yc�A�%*

lossS6�<,r��       �	8K!Yc�A�%*

loss7��=Y�4       �	��!Yc�A�%*

loss���<O��       �	��"Yc�A�%*

lossE\Q<{��z       �	� #Yc�A�%*

loss�]@=wRGl       �	��#Yc�A�%*

loss7�q<J���       �	�}$Yc�A�%*

loss�l=�R�6       �	+%Yc�A�%*

loss��;���       �	�%Yc�A�%*

lossZ
<��       �	�n&Yc�A�%*

loss���<r�       �	g'Yc�A�%*

loss�=;o�       �	6�'Yc�A�%*

loss��;����       �	�N(Yc�A�%*

loss�T�;gJ��       �	�(Yc�A�%*

loss=M1<v�l       �	y�)Yc�A�%*

lossH�<�썋       �	�e*Yc�A�%*

loss��<�s       �	� +Yc�A�%*

loss���<��J�       �	�+Yc�A�%*

loss� �=��%       �	�I,Yc�A�%*

loss;�Q<LP       �	��,Yc�A�%*

lossMe�<�       �	/�-Yc�A�%*

loss���; �       �	O".Yc�A�%*

lossr@<��       �	#�.Yc�A�%*

loss�$=����       �	yZ/Yc�A�%*

loss�s�;JI�       �	�/Yc�A�%*

loss�*�;~�       �	�0Yc�A�%*

lossl�"<j��       �	�@1Yc�A�%*

loss�?�<��       �	�A2Yc�A�%*

lossL�<�n�       �	,�2Yc�A�%*

loss�y�;	�H�       �	�3Yc�A�%*

loss��<��
�       �	%A4Yc�A�%*

lossx<��/       �	�%5Yc�A�%*

lossB�=q��       �	3�5Yc�A�%*

lossWͭ<�i2       �	�s6Yc�A�%*

lossz�<�F�d       �	;7Yc�A�%*

lossjV�</9��       �	�7Yc�A�%*

lossD�2<[��       �	m8Yc�A�%*

loss�dx<
��z       �	�9Yc�A�%*

loss��<�4:2       �	�9Yc�A�%*

loss%<��P�       �	��:Yc�A�%*

loss���;ݎ��       �	��SYc�A�%*

lossN=^��       �	�mTYc�A�%*

loss]�<N��<       �	UYc�A�%*

loss�ʌ<+�e�       �	��UYc�A�%*

loss�O=�y�       �	q9VYc�A�%*

loss�9b<�$%       �	J�VYc�A�%*

loss>6	=<�l�       �	vlWYc�A�%*

loss�:=k_�       �	XYc�A�%*

loss�VQ=0�ڔ       �	�LYYc�A�%*

loss|��<����       �	c�YYc�A�%*

loss�;����       �	j�ZYc�A�%*

loss�[.=�	�       �	�[Yc�A�%*

loss%2�=�mc(       �	��[Yc�A�%*

loss�w=�Vx�       �	-_\Yc�A�%*

lossv�=jB��       �	T�\Yc�A�%*

lossái=��+!       �	�]Yc�A�%*

loss��F:��       �	�^Yc�A�%*

loss^D<���       �	�_Yc�A�%*

loss�q�<^��=       �	o�_Yc�A�%*

loss,�=���       �	!�`Yc�A�%*

loss) �<��6M       �	�9aYc�A�%*

loss���;w��       �	k�aYc�A�%*

loss�<�-��       �	&sbYc�A�%*

loss!2�<_b!       �	0cYc�A�%*

loss|��;�ֵ       �	X�cYc�A�%*

loss�H�;9��       �	�_dYc�A�%*

lossZ��<�g�7       �	o�dYc�A�%*

loss�MW<�T��       �	��eYc�A�%*

lossW
�<34�       �	�afYc�A�%*

loss��<0K��       �	��fYc�A�%*

loss;�}<���       �	��gYc�A�%*

loss�<��c       �	p@hYc�A�%*

loss�<(�o�       �	�hYc�A�%*

lossFV#=(�       �	1�iYc�A�%*

loss��<� ��       �	��jYc�A�%*

loss��y=P�Ӄ       �	^-kYc�A�%*

lossn��:5��       �	�lYc�A�%*

loss�:=�       �	<3mYc�A�%*

loss.��<0�Ր       �	��mYc�A�%*

loss���<n���       �	�tnYc�A�%*

lossi-�<��       �	�oYc�A�%*

loss�<��y       �	��oYc�A�%*

loss��<I��       �	mpYc�A�%*

loss�G�<��{s       �	>qYc�A�%*

loss�m�<`�&�       �	\�qYc�A�%*

loss �=QH�       �	�BrYc�A�%*

loss(�e<�݆       �	��rYc�A�%*

loss_�=s�	M       �	H�sYc�A�%*

loss*��;�"�_       �	�-tYc�A�%*

loss4��9���[       �	+�tYc�A�%*

loss�+<�[�       �	cuYc�A�%*

loss�8�<�v/�       �	; vYc�A�%*

loss#c�;���       �	��vYc�A�%*

lossA>mn��       �	�'wYc�A�%*

loss�2B<��Y�       �	��wYc�A�%*

loss�k;�w��       �	CWxYc�A�%*

loss��:�U�?       �	��xYc�A�%*

loss��<~��       �	u�yYc�A�%*

loss��m<��+�       �	�$zYc�A�%*

loss���<"Û�       �	4�zYc�A�%*

loss(g�=WJ�/       �	�T{Yc�A�%*

lossj��<�ɪ       �	�{Yc�A�%*

loss�<�:"�G�       �	�M}Yc�A�%*

loss���<c�'       �	m�}Yc�A�&*

loss�;�<�D�       �	A}~Yc�A�&*

loss�c<%���       �	�Yc�A�&*

loss�@@="�;\       �	��Yc�A�&*

loss�I�<¸��       �	�<�Yc�A�&*

loss���<�vi       �	�ЀYc�A�&*

loss�=r�        �	fl�Yc�A�&*

loss�&x=�2��       �	O�Yc�A�&*

loss{��<\�       �	���Yc�A�&*

loss�e�<���       �	5�Yc�A�&*

loss��<Drc�       �	q˃Yc�A�&*

lossYG�<��O       �	a�Yc�A�&*

loss�^�<� ��       �	N	�Yc�A�&*

loss��>;����       �	L��Yc�A�&*

loss���;�(�       �	K<�Yc�A�&*

loss\:6<���       �	(цYc�A�&*

loss2�<�ӣ       �	O��Yc�A�&*

lossū�;;	C^       �	�H�Yc�A�&*

loss��<��       �	V��Yc�A�&*

loss�B=��q_       �	���Yc�A�&*

lossO<^0�V       �	�Yc�A�&*

loss��;��d�       �	� �Yc�A�&*

lossȏ<H۬G       �	���Yc�A�&*

loss�.h=sҊ<       �	Ԟ�Yc�A�&*

loss���<�ػ       �	�a�Yc�A�&*

lossI�C<~N�;       �	��Yc�A�&*

loss=�+<t�(       �	|��Yc�A�&*

loss�<f�`       �	�x�Yc�A�&*

loss@�=�x�U       �	�B�Yc�A�&*

loss�_<ʆ.�       �	N�Yc�A�&*

lossx�K<5��       �	J��Yc�A�&*

loss� =��u�       �	�.�Yc�A�&*

loss��<�ew       �	�ɒYc�A�&*

loss�?<���       �	�z�Yc�A�&*

loss�y�<v˷       �	�Yc�A�&*

lossC�N=7Wu4       �	ܸ�Yc�A�&*

loss�n<��҅       �	3P�Yc�A�&*

loss��;ڜ��       �	�Yc�A�&*

loss�b�:���6       �	�y�Yc�A�&*

loss��<��~�       �	��Yc�A�&*

loss��W;�i��       �	��Yc�A�&*

loss�z<�A:�       �	�>�Yc�A�&*

loss.X�=�C       �	�ԘYc�A�&*

loss��4<'�E       �		n�Yc�A�&*

losslK�<j���       �	.�Yc�A�&*

loss=�F=w�BO       �	:��Yc�A�&*

loss;�T<�<�@       �	G�Yc�A�&*

loss���<���       �	��Yc�A�&*

loss�ځ=W՝�       �	���Yc�A�&*

loss��<�|P       �	2�Yc�A�&*

loss��;��9       �	�ƝYc�A�&*

loss��Y<,<w�       �	i�Yc�A�&*

loss4@<��       �	���Yc�A�&*

loss��=^�       �	���Yc�A�&*

loss��=O~6B       �	l=�Yc�A�&*

loss�i<%�x       �	,�Yc�A�&*

loss�w�<��-m       �	��Yc�A�&*

loss5�<���       �	F"�Yc�A�&*

lossm�x=y�-�       �	�q�Yc�A�&*

loss3r�<[�R�       �	R�Yc�A�&*

loss���<X��       �	�Yc�A�&*

loss�+
<a�hb       �	�,�Yc�A�&*

loss�2<1��       �	ѦYc�A�&*

loss�Jn<�
k6       �	�i�Yc�A�&*

lossz�8<>��       �	��Yc�A�&*

loss�E�<1�I�       �	R��Yc�A�&*

lossWÇ<���       �	.:�Yc�A�&*

loss��S<H�lS       �	��Yc�A�&*

loss���<���       �	���Yc�A�&*

loss�;���       �	�"�Yc�A�&*

loss�$W<Htx�       �	MګYc�A�&*

lossT+<=�՘g       �	�n�Yc�A�&*

loss���<n�ۧ       �	��Yc�A�&*

loss�v�<�j!       �	���Yc�A�&*

loss,&)=�<	\       �	�=�Yc�A�&*

loss��P=��4�       �	�ݮYc�A�&*

loss��a=�{�       �	�x�Yc�A�&*

loss�<$��       �	�5�Yc�A�&*

lossR�B<"A       �	(ѰYc�A�&*

loss�sn=��       �	�q�Yc�A�&*

loss�3b<5��B       �	�>�Yc�A�&*

loss�"=�3�&       �	]��Yc�A�&*

loss��<7���       �	�7�Yc�A�&*

loss�<!j       �	���Yc�A�&*

loss��~<��K�       �	�.�Yc�A�&*

loss��;�-��       �	JζYc�A�&*

loss�:s=e�C+       �	�j�Yc�A�&*

loss�y<;h�       �	��Yc�A�&*

loss���<jz       �	6��Yc�A�&*

loss�cB=A�F       �	�f�Yc�A�&*

lossq/�<�թR       �	�5�Yc�A�&*

loss[_�<lkv       �	�ߺYc�A�&*

loss�NA:�D9h       �	z�Yc�A�&*

loss;X=� Fh       �	8�Yc�A�&*

loss#%=�j:�       �	��Yc�A�&*

loss�n�<ުuk       �	�K�Yc�A�&*

lossEU�<��r       �	��Yc�A�&*

loss��=�P       �	ߊ�Yc�A�&*

loss�<�!       �	�'�Yc�A�&*

loss�K�<`�N�       �	ҿYc�A�&*

loss�:����       �	�k�Yc�A�&*

loss}i=03]       �	��Yc�A�&*

loss��<�       �	A��Yc�A�&*

loss<iJC�       �	�6�Yc�A�&*

lossX=�\ki       �	���Yc�A�&*

loss���;����       �	Zd�Yc�A�&*

loss)��<��P�       �	���Yc�A�&*

lossx16=i�?�       �	��Yc�A�&*

loss�գ;�J��       �	�U�Yc�A�&*

loss��<�^ҡ       �	N��Yc�A�&*

loss�lG<v$�       �	Ͻ�Yc�A�&*

loss�Ĕ;qj�       �	�o�Yc�A�&*

loss8�<'���       �	��Yc�A�&*

lossF��=:;�       �	a��Yc�A�&*

lossxOL<�O+       �	SB�Yc�A�&*

lossV�%= '�/       �	w��Yc�A�&*

loss�՞<&B�       �	5�Yc�A�&*

loss�$�:�޳�       �	��Yc�A�&*

loss��<�a&�       �	���Yc�A�&*

loss⡇<���       �	X:�Yc�A�&*

loss�4;�>c�       �	w�Yc�A�&*

loss�S�;�П       �	B�Yc�A�&*

lossT��<�n       �	ܜ�Yc�A�&*

loss2=\g�       �	�2�Yc�A�&*

lossT�<BD1�       �	�S�Yc�A�&*

loss��=� ��       �	�<�Yc�A�&*

loss`��<r��       �	��Yc�A�&*

losso�w=~�"4       �	�m�Yc�A�&*

loss�L<�:�       �	��Yc�A�'*

lossW�2<��|�       �	U��Yc�A�'*

lossh��;��+�       �	�9�Yc�A�'*

loss��;�Vgb       �	2��Yc�A�'*

loss p�<GQ9       �	.�Yc�A�'*

losssN=/(�        �	���Yc�A�'*

lossN.�;?�1�       �	���Yc�A�'*

loss�n=+�E       �	.U�Yc�A�'*

loss1�g<�UN       �	��Yc�A�'*

lossaT#=��       �	�~�Yc�A�'*

loss��2<f2       �	O�Yc�A�'*

loss��
=K[
       �	��Yc�A�'*

loss}/ =���       �	���Yc�A�'*

loss���<��       �	�^�Yc�A�'*

loss�Ë=�]�       �	V��Yc�A�'*

loss��=j3�       �	r��Yc�A�'*

loss?j�<C���       �	�!�Yc�A�'*

loss2w,=w�T�       �	���Yc�A�'*

loss�y<���       �	fN�Yc�A�'*

loss�S2='��       �	���Yc�A�'*

lossK=�Fl#       �	|�Yc�A�'*

loss�?=ļxm       �	��Yc�A�'*

loss�fI<�CX"       �	���Yc�A�'*

lossv=]~�E       �	�Q�Yc�A�'*

lossU=�G       �	���Yc�A�'*

lossߥf=蒒�       �	l��Yc�A�'*

losso,(=�Q]V       �	�5�Yc�A�'*

losst�B=��A�       �	�t�Yc�A�'*

loss^��=�T�       �	��Yc�A�'*

loss���=Ƴ�       �	���Yc�A�'*

loss=�;Ƣ��       �	�B�Yc�A�'*

lossi�s<:u��       �	P��Yc�A�'*

loss�R�<�Bt2       �	Ox�Yc�A�'*

lossd�<q}B       �	��Yc�A�'*

lossL�	=Q��       �	t��Yc�A�'*

lossZq=��}�       �	�M�Yc�A�'*

lossM��;�]�8       �	N��Yc�A�'*

loss�޳<ѡV�       �	ۈ�Yc�A�'*

loss�]�<���$       �	&�Yc�A�'*

loss��;C�       �	���Yc�A�'*

lossj��<&��C       �	�^�Yc�A�'*

lossH�v<����       �	z��Yc�A�'*

lossb	�<�*'�       �	p��Yc�A�'*

loss�"�= �       �	�D�Yc�A�'*

loss��L=@@��       �	���Yc�A�'*

loss��;��=       �	ۆ�Yc�A�'*

lossN��< �?       �	��Yc�A�'*

loss��<�ڞ4       �	���Yc�A�'*

loss���<��[H       �	a�Yc�A�'*

loss-<g���       �	l��Yc�A�'*

loss
��<%�ܺ       �	'��Yc�A�'*

loss�I�=ǩ��       �	�!�Yc�A�'*

loss�>�=u�d�       �	���Yc�A�'*

loss!�=)&�m       �	�{�Yc�A�'*

loss�cA<��K�       �	;�Yc�A�'*

loss��;���       �	Q-�Yc�A�'*

lossY�<��       �	���Yc�A�'*

lossʺ�;�g��       �	�]�Yc�A�'*

loss��<��^`       �	s�Yc�A�'*

loss�q<Uwm$       �	/�Yc�A�'*

loss� �<RJ?�       �	���Yc�A�'*

loss��E=z�(       �	�T�Yc�A�'*

lossq�?<L�r       �	��Yc�A�'*

loss_;i�       �	��Yc�A�'*

loss�d<�WqK       �	F(�Yc�A�'*

loss_�<-P�       �	���Yc�A�'*

loss�lV<��@�       �	C��Yc�A�'*

lossm��<Tg�j       �	4 Yc�A�'*

lossRi=��       �	� Yc�A�'*

loss`��;Wv�       �	�xYc�A�'*

loss!�?<��b       �	;�Yc�A�'*

loss��<��L�       �	O"Yc�A�'*

loss\��<���V       �	��Yc�A�'*

lossA�'=a�.       �	�Yc�A�'*

loss���:��X       �	#Yc�A�'*

loss� �=�$	�       �	��Yc�A�'*

loss�C=`��Y       �	B�Yc�A�'*

loss�_�=��kB       �	c+Yc�A�'*

loss$Ы=e��       �	��Yc�A�'*

lossO3<�";a       �	eqYc�A�'*

loss��=��":       �	B$	Yc�A�'*

loss���</��       �	��	Yc�A�'*

loss�=Qm��       �	�Z
Yc�A�'*

loss]��=겾�       �	��
Yc�A�'*

loss���<�4�{       �	�Yc�A�'*

lossɨ=}v�       �	X8Yc�A�'*

lossa�<�*+       �	�Yc�A�'*

loss^<��       �	b�Yc�A�'*

loss��=n`�0       �	�mYc�A�'*

loss� "<�fv�       �	�Yc�A�'*

lossjpx;�U�       �	��Yc�A�'*

loss�|<�/��       �	�IYc�A�'*

loss��w=L�`�       �	6�Yc�A�'*

losso*=RA��       �	�Yc�A�'*

lossw�<�tt       �	�JYc�A�'*

loss���=fe�       �	Q�Yc�A�'*

lossZ�h<N<�       �	�qYc�A�'*

lossS�;b��_       �	NYc�A�'*

loss�j\<�[��       �	�Yc�A�'*

loss�;�;���       �	4MYc�A�'*

loss ��<���       �	��Yc�A�'*

lossa0K;���       �	�}Yc�A�'*

loss)��;�!-       �	#Yc�A�'*

loss��<@�>�       �	7�Yc�A�'*

loss<�r<���       �	�\Yc�A�'*

loss��L;�U�        �	j�Yc�A�'*

lossQ<w׽       �	��Yc�A�'*

lossz��<hI|�       �	p|Yc�A�'*

loss�0�<����       �	�Yc�A�'*

loss�n>=�yc!       �	O�Yc�A�'*

loss�_�<
Mp{       �	�GYc�A�'*

loss��;��8       �	��Yc�A�'*

loss/E_=	��       �	FzYc�A�'*

loss@n�<nz�i       �	�Yc�A�'*

loss��<��h�       �	;�Yc�A�'*

loss��:JmL8       �	�CYc�A�'*

loss���<�j3�       �	��Yc�A�'*

loss��<�IȲ       �	�p Yc�A�'*

loss|�_<D�E�       �	�!Yc�A�'*

loss)k�;Um��       �	��!Yc�A�'*

lossՃ�<��h       �	K"Yc�A�'*

losstS<�G       �	4�"Yc�A�'*

lossM�&;��U       �	X�#Yc�A�'*

lossG<�1��       �	�,$Yc�A�'*

loss���<�0i#       �	��$Yc�A�'*

loss�ѓ;��*       �	�i%Yc�A�'*

loss�<���       �	�+&Yc�A�'*

lossdX�<�qoy       �	Y'Yc�A�'*

loss���</�v�       �	��(Yc�A�(*

loss��_<sI�C       �	e7)Yc�A�(*

loss�<�od�       �	�=*Yc�A�(*

lossHf=��R#       �	R�*Yc�A�(*

loss��<��|V       �	�+Yc�A�(*

loss�- =���.       �	+M,Yc�A�(*

loss���<xۚ%       �	d�,Yc�A�(*

loss}=�fX       �	��-Yc�A�(*

loss8�<'ts�       �	�).Yc�A�(*

loss�,�<2ة.       �	l�.Yc�A�(*

loss�k=�<
       �	qt/Yc�A�(*

loss��=g��       �	'0Yc�A�(*

loss!�@<��        �	��0Yc�A�(*

loss�Ԅ<j       �	�Z1Yc�A�(*

lossqٌ<��d�       �	�*2Yc�A�(*

loss1�"==�       �	��2Yc�A�(*

loss�v�<����       �	F^3Yc�A�(*

loss���<���)       �	��3Yc�A�(*

loss�<���       �	 �4Yc�A�(*

loss<�;�C�       �	iU5Yc�A�(*

loss���;Uk��       �	p�5Yc�A�(*

loss��;
�]&       �	��6Yc�A�(*

loss�[<�Ǹ�       �	;m7Yc�A�(*

loss��=���@       �	58Yc�A�(*

loss�+�;�8n'       �	��8Yc�A�(*

loss�ʈ=��͈       �	ʧ9Yc�A�(*

loss���<����       �	F:Yc�A�(*

loss<��       �	y$;Yc�A�(*

lossHks<�w��       �	��;Yc�A�(*

loss�_=�       �	p�<Yc�A�(*

loss�8c<�*�       �	+/=Yc�A�(*

lossJd;��GJ       �	��=Yc�A�(*

loss$�;�3`�       �	�c>Yc�A�(*

loss�\�<Ee�       �	?Yc�A�(*

loss��:��8<       �	��?Yc�A�(*

loss��%=~`�       �	^L@Yc�A�(*

lossrI;���       �	T�@Yc�A�(*

loss�A�;V5�       �	@�AYc�A�(*

loss�<E���       �	"BYc�A�(*

lossA@<��	       �	^�BYc�A�(*

lossv<=�=       �	�YCYc�A�(*

lossHو<^<�       �	�DYc�A�(*

losss͏;.       �	��DYc�A�(*

loss*�=��Bw       �	,JEYc�A�(*

loss��
=g  �       �	��EYc�A�(*

loss��t=)�4.       �	_~FYc�A�(*

loss��=�%p�       �	�GYc�A�(*

loss�� =ī��       �	�GYc�A�(*

loss���<%Ŏ       �	�bHYc�A�(*

loss-�9<�c�       �	�;IYc�A�(*

loss$��<qڽ       �	4�IYc�A�(*

lossX��<��H�       �	��JYc�A�(*

losszR<��C       �	 KYc�A�(*

loss4�;1l�       �	��KYc�A�(*

loss�<�u       �	�SLYc�A�(*

loss�u;����       �	%$NYc�A�(*

lossȰ�=1�8$       �	b�NYc�A�(*

loss��F= ;8p       �	WOYc�A�(*

loss�L�<3�
       �	t�PYc�A�(*

losshڞ<�F�       �	�IQYc�A�(*

lossrWM=�r�K       �	��QYc�A�(*

loss�x�<����       �	�'SYc�A�(*

loss��!=W^0�       �	j�SYc�A�(*

lossĮ=�I��       �	�]TYc�A�(*

loss��>=u��       �	�TYc�A�(*

lossC<?[�r       �	]VYc�A�(*

loss�t�<.��       �	�VYc�A�(*

lossCP=>M��       �	wHWYc�A�(*

lossW�Y<����       �	��WYc�A�(*

loss'"�<��T       �	��XYc�A�(*

lossO�<υH       �	oeYYc�A�(*

loss��<����       �	�ZYc�A�(*

loss�Ny<Г�|       �	��ZYc�A�(*

loss�՞;�׍=       �	�C[Yc�A�(*

loss��<�5d;       �	'�[Yc�A�(*

loss��=��U�       �	O�\Yc�A�(*

loss��>��@�       �	�J]Yc�A�(*

loss�!e;B��       �	��]Yc�A�(*

lossN;
��       �	y�^Yc�A�(*

lossaЍ<�)M�       �	�-_Yc�A�(*

loss���<p�U       �	��_Yc�A�(*

lossJ�<Q�'T       �	�l`Yc�A�(*

loss;kP<��Ѹ       �	�aYc�A�(*

lossn�<E|7T       �	��aYc�A�(*

loss�o=ϗ��       �	�FbYc�A�(*

loss���=zʳ       �	3�bYc�A�(*

loss��,=c�       �	�}cYc�A�(*

loss3��<e2�       �	�3dYc�A�(*

loss�S<9/^�       �	�dYc�A�(*

loss���<RK�        �	�weYc�A�(*

loss���<8�P       �	fYc�A�(*

loss:�l=;л�       �	��fYc�A�(*

loss4��<���`       �	�FgYc�A�(*

loss�@=�^��       �	b�gYc�A�(*

losso,0=3V��       �	3�hYc�A�(*

loss��=����       �	�$iYc�A�(*

loss��r=Dv0�       �	��iYc�A�(*

lossȞ:%�$D       �	=�jYc�A�(*

loss��Y<�篚       �	"5kYc�A�(*

loss���< �~       �	V�kYc�A�(*

lossf	�<���(       �	�ylYc�A�(*

loss�օ<'ěk       �	mYc�A�(*

loss���<�n�       �	��mYc�A�(*

loss,�:ipi�       �	�znYc�A�(*

loss��[=�U��       �	XoYc�A�(*

loss���<N�D�       �	g�oYc�A�(*

lossQa<V�j       �	!VpYc�A�(*

loss�@)=m�v       �	`qYc�A�(*

loss���<��       �	M�qYc�A�(*

lossmZ<��`       �	�>rYc�A�(*

lossVau;S��C       �	GsYc�A�(*

lossŀ=,5��       �	�sYc�A�(*

loss�?8<�[       �	�tYc�A�(*

lossq�<wz�       �	V(uYc�A�(*

loss�4=����       �	6�uYc�A�(*

loss�<ќ��       �	9cvYc�A�(*

loss�@<���$       �	X�vYc�A�(*

lossMHR<cS��       �	�wYc�A�(*

losst�m<~v<4       �	f2xYc�A�(*

loss���<O���       �	)�xYc�A�(*

lossr�;��D�       �	�gyYc�A�(*

loss B�<�T       �	m�yYc�A�(*

loss���<�ڜ       �	=�zYc�A�(*

loss_V=��1       �	pC{Yc�A�(*

lossR�i=��c       �	��{Yc�A�(*

loss��p;_�       �	|Yc�A�(*

lossc<��7       �	�$}Yc�A�(*

loss�-�=W��       �	��}Yc�A�(*

lossA��;���/       �	�y~Yc�A�)*

loss4��<tu�       �	�Yc�A�)*

losshJ�<�B=       �	��Yc�A�)*

loss��<��       �	�U�Yc�A�)*

lossI�<�-O       �	���Yc�A�)*

loss��<y:u       �	��Yc�A�)*

loss�7=D�S       �	�9�Yc�A�)*

loss�:=��J�       �	��Yc�A�)*

loss��=��a       �	г�Yc�A�)*

lossq��;��ln       �	X�Yc�A�)*

loss���<��b       �	���Yc�A�)*

loss�}�<tr�       �	�Yc�A�)*

loss��<}�L�       �	D0�Yc�A�)*

loss|?<���       �	��Yc�A�)*

losscV;���<       �	���Yc�A�)*

loss��<�j%�       �	'K�Yc�A�)*

loss���<�q`       �	O�Yc�A�)*

loss�=t!6       �	䄉Yc�A�)*

lossB=�c�#       �	a�Yc�A�)*

lossp�<�v)e       �	e�Yc�A�)*

loss�7<�[��       �	���Yc�A�)*

loss6��<u�@       �	�N�Yc�A�)*

loss�
;uY�e       �	���Yc�A�)*

loss7�'="�Jr       �	���Yc�A�)*

loss��<���3       �	=�Yc�A�)*

lossq+�<(߲>       �	�؎Yc�A�)*

loss\Y=9�j�       �	
��Yc�A�)*

lossm�;M���       �	�5�Yc�A�)*

loss�~E;��>�       �	��Yc�A�)*

loss&��<�1�       �	.��Yc�A�)*

loss�T&<��LN       �	�)�Yc�A�)*

loss��s<����       �	�ɒYc�A�)*

loss�8#=d��"       �	�l�Yc�A�)*

loss6�=5��       �	�Yc�A�)*

loss
l�;KE       �	~��Yc�A�)*

loss$v<���       �	{N�Yc�A�)*

loss�Li;���#       �	��Yc�A�)*

loss$5=�
��       �	w��Yc�A�)*

lossH�!=��       �	G�Yc�A�)*

loss�ޑ;6�?_       �	��Yc�A�)*

loss�9=���       �	/��Yc�A�)*

loss<o�<(�Z�       �	q �Yc�A�)*

lossH�P=��s       �	 əYc�A�)*

lossS�<e]       �	�`�Yc�A�)*

loss�0�<��       �	���Yc�A�)*

lossI�;]�5       �	ۧ�Yc�A�)*

loss
R&=�u��       �	�Q�Yc�A�)*

loss��<cb�       �	4�Yc�A�)*

lossԂ'=�Е�       �	B��Yc�A�)*

loss���=�J\       �	�/�Yc�A�)*

loss��=I�	       �	�͞Yc�A�)*

loss�Ҷ<����       �	k�Yc�A�)*

loss��J=�Vm       �	>�Yc�A�)*

loss��}=&5@*       �	1�Yc�A�)*

loss�5=;7�       �	��Yc�A�)*

loss�$<�.�)       �	�$�Yc�A�)*

loss��<��n*       �	ĢYc�A�)*

loss��<�v�U       �	�l�Yc�A�)*

loss�@�;W��v       �	��Yc�A�)*

loss=��;�5y       �	b��Yc�A�)*

lossm�;�7��       �	׆�Yc�A�)*

loss<^=��z       �	W"�Yc�A�)*

loss�a=��P       �	��Yc�A�)*

loss7�<�H�v       �	
��Yc�A�)*

loss=7=2�W]       �	O �Yc�A�)*

loss��3;���V       �	�¨Yc�A�)*

loss�<�gt�       �	�a�Yc�A�)*

loss��;u��       �	&��Yc�A�)*

loss��<4s`K       �	8��Yc�A�)*

loss�|#=w�a       �	�@�Yc�A�)*

loss�U+;�Z�       �	��Yc�A�)*

loss��<�       �	���Yc�A�)*

lossz�e<����       �	+3�Yc�A�)*

loss���<���       �	V֭Yc�A�)*

lossFՀ<{�<�       �	�p�Yc�A�)*

lossД#;��       �	��Yc�A�)*

loss�=\<���       �	#��Yc�A�)*

loss\<&=!��       �	��Yc�A�)*

loss��;0Xqb       �	M��Yc�A�)*

loss�N5<* ��       �	ob�Yc�A�)*

loss�Q=B`i       �	r5�Yc�A�)*

loss��=��       �	��Yc�A�)*

lossi�;���(       �	��Yc�A�)*

lossh��;��d�       �	Q��Yc�A�)*

loss��S<��?�       �	���Yc�A�)*

loss��;V<�       �	zU�Yc�A�)*

loss���8�s=       �	�W�Yc�A�)*

lossd�(<�v(       �	���Yc�A�)*

loss��;z]-|       �	�S�Yc�A�)*

lossȿg<[�       �	�/�Yc�A�)*

loss�N�;��0	       �	�ͻYc�A�)*

lossO<�:�*�9       �	h�Yc�A�)*

loss�ZX<�=�       �	�Yc�A�)*

loss�:�)p       �	}��Yc�A�)*

loss\�';�p��       �	�1�Yc�A�)*

loss�o9�[;�       �	�̾Yc�A�)*

loss\/!<��~q       �	�e�Yc�A�)*

loss��<v��       �	���Yc�A�)*

loss8&�;����       �	��Yc�A�)*

loss,q,:�V �       �	�-�Yc�A�)*

loss�?�;)���       �	T��Yc�A�)*

loss�4=v��       �	Yn�Yc�A�)*

loss� ,;'��       �	��Yc�A�)*

loss��<)b��       �	s��Yc�A�)*

lossvP
=�p�q       �	�S�Yc�A�)*

loss���<�Yu       �	f��Yc�A�)*

loss���<HT�"       �	���Yc�A�)*

lossʹ<�:c       �	C�Yc�A�)*

loss-�<ʜ�7       �	��Yc�A�)*

lossfk,=�x       �	${�Yc�A�)*

loss̨�;d���       �	��Yc�A�)*

loss�S3<����       �	H��Yc�A�)*

loss/@;t��       �	�h�Yc�A�)*

loss	��=ax^�       �	l�Yc�A�)*

loss
�=�16�       �	���Yc�A�)*

loss��D=�ZQ1       �	�L�Yc�A�)*

loss ��<t}�       �	t��Yc�A�)*

loss��<��s�       �	��Yc�A�)*

loss�<<^��       �	$��Yc�A�)*

lossa�J<GD'D       �	�K�Yc�A�)*

loss �a=;�^       �	?��Yc�A�)*

loss�Ӫ<꽴�       �	�M�Yc�A�)*

lossc�O<>���       �	V��Yc�A�)*

loss	Y�<a�]R       �	���Yc�A�)*

lossvQ�<�-��       �	 ;�Yc�A�)*

loss�g<%i�z       �	���Yc�A�)*

loss3;�;�'��       �	���Yc�A�)*

loss}�;÷��       �	��Yc�A�)*

loss<3�<�N"�       �	��Yc�A�**

loss=Tq��       �	yx�Yc�A�**

losss�<����       �	e�Yc�A�**

lossz]=�5(       �	���Yc�A�**

lossHJ<�j��       �	�[�Yc�A�**

loss��;/�N4       �	���Yc�A�**

loss#�;��x       �	���Yc�A�**

lossW��;AEB�       �	�`�Yc�A�**

loss\q�<���       �	O�Yc�A�**

loss��<��)       �	���Yc�A�**

loss*u�;l��       �	�S�Yc�A�**

lossw��;�       �	��Yc�A�**

loss�!�<��j        �	���Yc�A�**

loss&>F<�t�N       �	�-�Yc�A�**

loss���<)@�       �	��Yc�A�**

loss�[<�Ϗ6       �	���Yc�A�**

lossA_;	�=       �	N%�Yc�A�**

loss!*�<�*T�       �	���Yc�A�**

loss��;<e�       �	�b�Yc�A�**

lossF��<A��K       �	��Yc�A�**

loss��<���h       �	k��Yc�A�**

loss��I;¶'�       �	�G�Yc�A�**

loss���<��}       �	��Yc�A�**

lossI+Q;x'��       �	Wx�Yc�A�**

loss�q=��w       �	g�Yc�A�**

lossѠ�;���       �	��Yc�A�**

loss��<T�e       �	X Yc�A�**

loss� =\rhG       �	�� Yc�A�**

loss�%<��e�       �	��Yc�A�**

loss��<~�C�       �	�&Yc�A�**

loss��<o��       �	��Yc�A�**

loss�ro<j�Q�       �	W]Yc�A�**

loss���<�s�4       �	�Yc�A�**

loss��=��t�       �	סYc�A�**

loss�6�<���       �	XYc�A�**

lossV�<�-�W       �	��Yc�A�**

lossdb�<B��3       �	ͯYc�A�**

loss��i<�U�       �	�Yc�A�**

loss��<�3�       �	�^Yc�A�**

loss�f�<���       �	u 	Yc�A�**

lossA�4<���z       �	ʥ	Yc�A�**

loss���;.��       �	�H
Yc�A�**

loss��;Sj�-       �	��
Yc�A�**

loss�Z�<�3��       �	tyYc�A�**

loss4��<%�H�       �	�Yc�A�**

lossSR=��       �	�#Yc�A�**

lossWޜ=x�F       �	��Yc�A�**

loss55<��u�       �	o�Yc�A�**

lossv=��<�       �	őYc�A�**

losse��<K���       �	�/Yc�A�**

loss���;9��       �	�Yc�A�**

losse�<<�D       �	�[Yc�A�**

loss8sP;$�n       �	/�Yc�A�**

loss�ը<��y       �	n�Yc�A�**

loss�\�;UBƨ       �	�fYc�A�**

loss�!�<���       �	\Yc�A�**

lossr�=9���       �	��Yc�A�**

loss�;�;�       �	�CYc�A�**

loss(=Iq�       �	��Yc�A�**

loss�O�<=���       �	��Yc�A�**

lossz~&=��Ȳ       �	(|Yc�A�**

lossLG�;J�|�       �	fYc�A�**

loss� }<PV       �	��Yc�A�**

loss�>���R       �	��Yc�A�**

loss��e=�T�m       �	h$Yc�A�**

loss�<!F�       �	�Yc�A�**

lossT&=���3       �	OuYc�A�**

loss�{<�Qc       �	�Yc�A�**

loss==a=׆�       �	�Yc�A�**

lossl�<��^�       �	,IYc�A�**

loss�q�<����       �	��Yc�A�**

loss���<tfv�       �	�zYc�A�**

lossZ;�;!�2+       �	iYc�A�**

loss�O<���       �	ܹYc�A�**

loss\��:��)       �	V Yc�A�**

loss@\<4���       �	�� Yc�A�**

loss�=��       �	!Yc�A�**

loss�fI<�c]�       �	_$"Yc�A�**

lossz�$>��2�       �	��"Yc�A�**

lossw $=����       �	��#Yc�A�**

loss���;�_��       �	]N$Yc�A�**

loss�7�;i�,       �	�$Yc�A�**

losso�;㘈�       �	��%Yc�A�**

loss���<8�t�       �	t(&Yc�A�**

lossݖV=Z��       �	�&Yc�A�**

loss��(=��       �	\t'Yc�A�**

loss,r;Ԟ C       �	
/(Yc�A�**

loss�u;���       �	[�(Yc�A�**

loss֗�=^~{       �	�)Yc�A�**

loss��=B-�       �	)*Yc�A�**

loss?S�;�V��       �	�+Yc�A�**

lossC4f<t� |       �	�+Yc�A�**

loss�K�;f� /       �	i,Yc�A�**

loss��=�Qg�       �	�-Yc�A�**

lossm�<Ȼt�       �	��-Yc�A�**

lossEVU<G�J|       �	�U.Yc�A�**

loss T�<����       �	��.Yc�A�**

lossH��<Nb��       �	D�/Yc�A�**

lossa�9<��|�       �	%0Yc�A�**

loss���<��`        �	��0Yc�A�**

loss6�W<T��T       �	{k1Yc�A�**

loss�;=��3O       �	2Yc�A�**

loss�pp<P���       �	%�2Yc�A�**

loss�ɴ;��n�       �	�O3Yc�A�**

loss%�$<q?0v       �	�m4Yc�A�**

loss�^<���       �	`;5Yc�A�**

losss�/=;fg       �	�5Yc�A�**

loss��<�ߔ       �	7Yc�A�**

loss/i�;�$�       �	��7Yc�A�**

loss��<��       �	�8Yc�A�**

loss2�<$�>�       �	�)9Yc�A�**

lossۿ�<�z^V       �	��9Yc�A�**

loss\�<"���       �	��:Yc�A�**

loss��<p.�       �	8�;Yc�A�**

lossl�<��7       �	�N<Yc�A�**

lossO�=����       �	t�<Yc�A�**

loss���<4�sw       �	�=Yc�A�**

loss��E<�V�O       �	+5>Yc�A�**

loss�'=���       �	J�>Yc�A�**

loss{��<�䮜       �	.r?Yc�A�**

loss%�w<�"�       �	�@Yc�A�**

lossײ<&���       �	m�@Yc�A�**

loss$;o<�K�6       �	'�AYc�A�**

lossh�~=k�mw       �	�HBYc�A�**

loss��+<��&�       �	��BYc�A�**

loss��t;S�p       �	�CYc�A�**

loss.6S:ne��       �	��DYc�A�**

loss�w�<N�q�       �	TUEYc�A�**

loss=J<��1e       �	C FYc�A�+*

loss&�=�F.       �	'�FYc�A�+*

losss��<1���       �	S>GYc�A�+*

loss�D�;� �}       �	��GYc�A�+*

loss!�<m2G       �	�wHYc�A�+*

loss���=��-�       �	?IYc�A�+*

loss��<�4       �	��IYc�A�+*

loss�F\=/��       �	�yJYc�A�+*

lossI��<;�I       �	: KYc�A�+*

loss].#<�Ԡb       �	k�KYc�A�+*

loss6$p<"��b       �	<PLYc�A�+*

loss<%N<�B�P       �	8�LYc�A�+*

losse��:�/O�       �	�MYc�A�+*

loss7�;52�V       �	fJNYc�A�+*

loss�=�tY�       �	�NYc�A�+*

loss��(= @/       �	%�OYc�A�+*

loss�Z=,�       �	��PYc�A�+*

lossO<=���C       �	�IQYc�A�+*

loss�J=��MA       �	��QYc�A�+*

lossʧ<����       �	�(SYc�A�+*

loss�.\<�&Y       �	��SYc�A�+*

loss��<
w3       �	n�TYc�A�+*

loss�J9<1l�g       �	9&UYc�A�+*

lossʽZ<�ɠ�       �	��UYc�A�+*

loss�K<�%u       �	�tVYc�A�+*

loss�«=L��       �	wWYc�A�+*

lossxL-<���7       �	|�WYc�A�+*

lossyʖ<F-j�       �	�QXYc�A�+*

loss�b�=P��       �	4�XYc�A�+*

loss���<䞹       �	��YYc�A�+*

lossCs�<gIA�       �	�2ZYc�A�+*

loss��>�Fǥ       �	��ZYc�A�+*

lossC}H<�ݻ�       �	f�[Yc�A�+*

loss���<���       �	]\Yc�A�+*

loss���<r�0       �	G]Yc�A�+*

loss@g<K!��       �	��]Yc�A�+*

loss�P<�lMQ       �	�A^Yc�A�+*

lossỂ;ðM�       �	��^Yc�A�+*

lossZZ=e��       �	��_Yc�A�+*

loss�1�<Z��       �	F&`Yc�A�+*

loss�߻<-�       �	��`Yc�A�+*

loss.μ=�E)�       �	&qaYc�A�+*

losso��<���        �	�bYc�A�+*

loss8J;oGM�       �	>�bYc�A�+*

lossl�<,�$�       �	�RcYc�A�+*

loss�;r/8K       �	J�cYc�A�+*

loss���<�%d       �	��dYc�A�+*

loss$�<�_$       �	�&eYc�A�+*

loss}!�<7T       �	��eYc�A�+*

lossev�</+=       �	�^fYc�A�+*

loss��<�M�       �	�gYc�A�+*

lossR��<'��       �	ŭgYc�A�+*

loss�nc:)3�E       �	tbhYc�A�+*

lossl0�<8�`       �	ciYc�A�+*

losswV�<zF�       �	��iYc�A�+*

loss_bO<�{��       �	�WjYc�A�+*

loss�<�1��       �	�jYc�A�+*

losse�<��L       �	��kYc�A�+*

loss|�<EOυ       �	_blYc�A�+*

loss�ʏ<o1��       �	�mYc�A�+*

loss�B�;�
�T       �	��mYc�A�+*

loss!nD=�~�       �	=~nYc�A�+*

loss H<<2>       �	a6oYc�A�+*

loss��<X	��       �	��oYc�A�+*

loss��i<��+�       �	B�pYc�A�+*

loss��<q)7/       �	XqYc�A�+*

loss굖<e�+&       �	`rYc�A�+*

loss�L�<귵�       �	��rYc�A�+*

loss��;����       �	YlsYc�A�+*

lossq�=�ϕ       �	A~tYc�A�+*

loss��C<{4'       �	$uYc�A�+*

loss��E;Ր�       �	�uYc�A�+*

loss(|<J�k}       �	��vYc�A�+*

loss�g�=���       �	�kwYc�A�+*

loss�3=z�?�       �	�xYc�A�+*

loss �^<��K       �	E�xYc�A�+*

loss��3;L�       �	�cyYc�A�+*

loss���;�~�!       �	�zYc�A�+*

loss�=��{�       �	ʦzYc�A�+*

lossJ�/<|g\�       �	�{Yc�A�+*

loss,�;�fa,       �	�$|Yc�A�+*

loss�Aa;�bl�       �	��|Yc�A�+*

loss; d<�3��       �	9a}Yc�A�+*

loss��<�}��       �	� ~Yc�A�+*

loss�0�<�]�:       �	�~Yc�A�+*

lossT%�<W���       �	�BYc�A�+*

loss�?j=0\%4       �	��Yc�A�+*

loss?$o=�y�U       �	��Yc�A�+*

loss���;`B�       �	�"�Yc�A�+*

lossdҜ;��[       �	3��Yc�A�+*

lossߴ^<�h��       �	�[�Yc�A�+*

lossݬ;7k�U       �	��Yc�A�+*

loss���;�U,e       �	h��Yc�A�+*

lossgu;�x�]       �	-�Yc�A�+*

lossz�n=����       �	�ĄYc�A�+*

lossJ[=��3N       �	�]�Yc�A�+*

lossњ<j\��       �	Y��Yc�A�+*

loss͏B=X�?�       �	[��Yc�A�+*

loss��e;=w�c       �	+4�Yc�A�+*

loss^=՘`�       �	�އYc�A�+*

loss��<k	�       �	˄�Yc�A�+*

loss�%p<�0       �	�=�Yc�A�+*

loss�_{;���       �	�߉Yc�A�+*

loss]GU=B`H�       �	\��Yc�A�+*

losst��<�uF?       �	�9�Yc�A�+*

lossV�<��       �	���Yc�A�+*

loss��<y��H       �	;��Yc�A�+*

lossz �<n�       �	p'�Yc�A�+*

loss� �<3��I       �	a��Yc�A�+*

loss�`�<���R       �	p�Yc�A�+*

lossc�*<r��_       �	��Yc�A�+*

loss!=o(�       �	и�Yc�A�+*

lossqN7=.f�<       �	t��Yc�A�+*

loss��<��}r       �	ap�Yc�A�+*

loss�=�U�}       �	B
�Yc�A�+*

lossH&�<V�2�       �	N��Yc�A�+*

loss�> =h���       �	T�Yc�A�+*

loss� <↌x       �	��Yc�A�+*

loss��;�h9�       �	���Yc�A�+*

loss�ߟ<�4�       �	ʍ�Yc�A�+*

loss�<8=>�ְ       �	���Yc�A�+*

loss~�<���       �	V�Yc�A�+*

loss��<�Mt�       �	j�Yc�A�+*

loss�>�;��       �	�ÙYc�A�+*

loss<�<��%       �	�a�Yc�A�+*

losswk<�S|�       �	U3�Yc�A�+*

loss�h:<����       �	�ܛYc�A�+*

loss?5�;J޹?       �	K��Yc�A�+*

loss$�;<���       �	�j�Yc�A�,*

lossԾ*=c��       �	�<�Yc�A�,*

loss��<�qb       �	�Z�Yc�A�,*

loss�J�<,��       �	81�Yc�A�,*

loss7�;~;�       �	{נYc�A�,*

loss
�v<?W�Q       �	V�Yc�A�,*

loss�wg<��       �	 �Yc�A�,*

loss�==ﶵ�       �	8��Yc�A�,*

lossO$<m龪       �	e�Yc�A�,*

lossR�=��z�       �	��Yc�A�,*

lossiv='��       �	^��Yc�A�,*

loss�ĺ<G��       �	}Y�Yc�A�,*

lossT6>wV       �	B�Yc�A�,*

loss��{<`�ة       �	㨦Yc�A�,*

lossP�;R�       �	�D�Yc�A�,*

loss�t@=��>�       �	��Yc�A�,*

loss���<��|�       �	H��Yc�A�,*

lossEgc=N��       �	�8�Yc�A�,*

lossO!=f8�       �	�ҩYc�A�,*

lossD�<2Z?�       �	�q�Yc�A�,*

loss��}=����       �	 �Yc�A�,*

loss)$=n���       �	!ʫYc�A�,*

loss��;N-�       �	�m�Yc�A�,*

loss��<���       �	�
�Yc�A�,*

loss'== t�       �	���Yc�A�,*

loss��<woz�       �	�F�Yc�A�,*

loss�~=L2�       �	��Yc�A�,*

loss�<��V�       �	�Yc�A�,*

loss�WY<�3G�       �	�C�Yc�A�,*

loss-�;|�       �	��Yc�A�,*

lossp'<b���       �	Yc�A�,*

loss���<�L�       �	�6�Yc�A�,*

loss@@�<�"�       �	�޲Yc�A�,*

lossE�<���       �	��Yc�A�,*

loss���;����       �	X��Yc�A�,*

loss���<۞��       �	tC�Yc�A�,*

loss�<CXe�       �	���Yc�A�,*

loss�M<��5�       �	>��Yc�A�,*

loss���=�ҫ�       �	�3�Yc�A�,*

losscN<MU��       �	�зYc�A�,*

lossvl<4�       �	m�Yc�A�,*

losss+<0^�       �	��Yc�A�,*

loss?k<ۚ.x       �	��Yc�A�,*

loss �<z�Ϣ       �	N^�Yc�A�,*

lossD]=�9�D       �	��Yc�A�,*

loss�Ʀ<M��       �	`ʻYc�A�,*

loss��H<I
�       �	�y�Yc�A�,*

loss�A=��;       �	䞾Yc�A�,*

loss���<իS�       �	<�Yc�A�,*

loss��a:U�,       �	YڿYc�A�,*

loss՚�=���       �	t�Yc�A�,*

loss�ف;Rh8�       �	�Yc�A�,*

loss
�%<�9�       �	��Yc�A�,*

loss�O�<P�s       �	M�Yc�A�,*

loss�YP=B�&       �	���Yc�A�,*

loss��=�,K       �	���Yc�A�,*

loss?_�;Z��       �	ۆ�Yc�A�,*

loss�M<�s�       �	$(�Yc�A�,*

lossLP$;32�Y       �	���Yc�A�,*

losse�U;��       �	�~�Yc�A�,*

loss��<T�e�       �	!�Yc�A�,*

lossj�;LO�       �	��Yc�A�,*

lossm�$<����       �	KY�Yc�A�,*

lossR��<�k�       �	���Yc�A�,*

loss���<�7�8       �	���Yc�A�,*

loss{��;T[q�       �	�9�Yc�A�,*

loss�_�<� v�       �	���Yc�A�,*

loss�V�<Z��>       �	��Yc�A�,*

loss��v<Y��       �	+�Yc�A�,*

loss4�7=�:;�       �	���Yc�A�,*

loss!΁;��H       �	Cr�Yc�A�,*

loss�9�<U�!       �	��Yc�A�,*

loss���<�-��       �	A��Yc�A�,*

loss���<�X��       �	~W�Yc�A�,*

loss�_�=11�       �	��Yc�A�,*

loss�O�;�Z�T       �	��Yc�A�,*

loss��<2!       �	hY�Yc�A�,*

loss ^�<F��       �	,��Yc�A�,*

lossfP�;ǬRp       �	��Yc�A�,*

loss�R?<f+Y�       �	� �Yc�A�,*

loss[�<\���       �	˽�Yc�A�,*

loss��o=�4�"       �	�c�Yc�A�,*

lossi <�]�H       �	@�Yc�A�,*

loss{u<�|h       �		��Yc�A�,*

loss��=�<�2       �	�j�Yc�A�,*

lossw	<���       �	4�Yc�A�,*

lossq)=f#|       �	��Yc�A�,*

lossT=��غ       �	Zg�Yc�A�,*

loss�x<?%��       �	�"�Yc�A�,*

loss	=�t�       �	���Yc�A�,*

loss�g�<���       �	jl�Yc�A�,*

loss*n==���Q       �	��Yc�A�,*

loss��;��{       �	Ψ�Yc�A�,*

loss� <07�f       �	+M�Yc�A�,*

lossS��<���e       �	���Yc�A�,*

loss�[�<=�I       �	���Yc�A�,*

lossOS�=��Ŭ       �	t&�Yc�A�,*

loss ^�<�Dc�       �	/��Yc�A�,*

lossR$�<����       �	�\�Yc�A�,*

loss�=�{h>       �	s��Yc�A�,*

loss�<���       �	K��Yc�A�,*

loss��<t�A�       �	w.�Yc�A�,*

loss|<���g       �	X��Yc�A�,*

loss��<=�p/�       �	N��Yc�A�,*

loss��<�gV       �		4�Yc�A�,*

loss'n<2x0�       �	��Yc�A�,*

lossx�@<wcI       �	bf�Yc�A�,*

loss6c=��       �	���Yc�A�,*

loss�3�;Ĉk       �	#��Yc�A�,*

loss5|;7j        �	~8�Yc�A�,*

loss�l�<q�=�       �	}��Yc�A�,*

loss!�=���-       �	#f�Yc�A�,*

loss�t�;
�
�       �	�Yc�A�,*

loss�_�==���       �	��Yc�A�,*

loss�0<�w       �	�H�Yc�A�,*

loss��<6��       �	k��Yc�A�,*

loss�cq<J ��       �	G��Yc�A�,*

lossp�=�J;       �	]�Yc�A�,*

loss���;L���       �	���Yc�A�,*

loss�v<P见       �	��Yc�A�,*

loss�F�;rܐ       �	�h�Yc�A�,*

loss?I=��6       �	q�Yc�A�,*

loss�B;S���       �	^��Yc�A�,*

loss^d=�`       �	�A�Yc�A�,*

lossV�+<��q       �	���Yc�A�,*

loss���<�jb0       �	��Yc�A�,*

loss/$�<r�=G       �	��Yc�A�,*

loss��<T��       �	���Yc�A�,*

loss�Xy<��?�       �	~Q�Yc�A�-*

loss,i=<�3��       �	��Yc�A�-*

loss�<��)<       �	,-�Yc�A�-*

loss�OD=�k�       �	Q��Yc�A�-*

loss��D=��ݰ       �	��Yc�A�-*

lossk=�X�N       �	1��Yc�A�-*

loss�W�<Ơ�~       �	Hk�Yc�A�-*

loss;V=�N�4       �	��Yc�A�-*

loss��(<�}�       �	ٴ�Yc�A�-*

loss��=^eO       �	^�Yc�A�-*

loss�8<��h�       �	�Yc�A�-*

loss�;�<�a�^       �	���Yc�A�-*

loss_��:W"hv       �	�~�Yc�A�-*

loss���<`�VW       �	�+�Yc�A�-*

lossti^<R�s\       �	l��Yc�A�-*

loss��;b�Q       �	�u�Yc�A�-*

loss�+�<I��       �	��Yc�A�-*

loss��<{�t       �	_��Yc�A�-*

lossP=�1�X       �	}Z Yc�A�-*

loss4h�<$�g       �	P� Yc�A�-*

loss�+�<�&K       �	1�Yc�A�-*

loss�w�;/�ʓ       �	�0Yc�A�-*

loss�Э<&�W       �	OYc�A�-*

loss�̝<����       �	��Yc�A�-*

lossd1�<�[��       �	v�Yc�A�-*

loss��L;���7       �	�UYc�A�-*

loss�g�=Up��       �	�Yc�A�-*

loss�<�P�       �	6�Yc�A�-*

loss*�<�>�K       �	�WYc�A�-*

loss#[}<�r�j       �	��Yc�A�-*

lossɗ <���m       �	A�	Yc�A�-*

loss�!w<���}       �	��
Yc�A�-*

loss,�N<n�
t       �	�:Yc�A�-*

loss��L<�-�       �	2�Yc�A�-*

loss�q�< &       �	�Yc�A�-*

loss��<���       �	[(Yc�A�-*

lossluv=Jd��       �	u�Yc�A�-*

loss��:���       �	nlYc�A�-*

lossi��;�2�       �	�Yc�A�-*

lossI�&<��ҭ       �	��Yc�A�-*

lossC<-$G       �	j�Yc�A�-*

loss�;��       �	�wYc�A�-*

loss��<>��       �	jYc�A�-*

loss�h=:BY�       �	�Yc�A�-*

lossٶ"<Y� �       �	>uYc�A�-*

loss��<KL5�       �	�/Yc�A�-*

loss���<V��{       �	��Yc�A�-*

loss.[�<��V       �	�nYc�A�-*

loss���<S�       �	�Yc�A�-*

lossw�<Ǧ�O       �	طYc�A�-*

loss_�=�R4�       �	��Yc�A�-*

loss�=QU@�       �	�7Yc�A�-*

loss�*=�_�       �	i�Yc�A�-*

lossP��;a��       �	�Yc�A�-*

loss�@d=���       �	<-Yc�A�-*

loss���<ψ��       �	F�Yc�A�-*

lossKp!=��!       �	�nYc�A�-*

loss%�-;��P       �	GrYc�A�-*

loss�0�;�       �	�Yc�A�-*

lossd�<��       �	ܻYc�A�-*

loss	��<�ː�       �	SYYc�A�-*

lossxpn<��K�       �	��Yc�A�-*

loss��^=�܍s       �	͓Yc�A�-*

lossq�;�E       �	�B Yc�A�-*

loss�z�=�T�       �	� Yc�A�-*

loss.�<���       �	��!Yc�A�-*

loss6$�;��       �	�C"Yc�A�-*

loss?�=�+�       �	��"Yc�A�-*

loss
S�=�]	�       �	D�#Yc�A�-*

loss_!�<���       �	�&$Yc�A�-*

loss�$�<�       �	��$Yc�A�-*

loss�n<����       �	�d%Yc�A�-*

loss�Pm=�:       �	�&Yc�A�-*

loss8�;�b�       �	%�&Yc�A�-*

lossM+=[���       �	^'Yc�A�-*

loss��o<�M`�       �	�(Yc�A�-*

loss��<af       �	�(Yc�A�-*

loss��/<��`       �	LR)Yc�A�-*

loss<�:<%�       �	��)Yc�A�-*

loss~@�<L�U       �	��*Yc�A�-*

lossh�<T8I       �	�;+Yc�A�-*

lossP_"=���(       �	U�+Yc�A�-*

loss\��=�
t�       �	�x,Yc�A�-*

loss8�j=��.s       �	"-Yc�A�-*

loss�M�<6��       �	�.Yc�A�-*

loss�t�<D��       �	��.Yc�A�-*

loss�!;<��*�       �	�[/Yc�A�-*

loss��q=��.G       �	�/Yc�A�-*

loss7Y0<��a       �	v�0Yc�A�-*

loss�՘<v��       �	?1Yc�A�-*

loss�<�<��N       �	�l2Yc�A�-*

loss홆<��x       �	!3Yc�A�-*

loss���;*���       �	�3Yc�A�-*

loss�p;��Q       �	�o4Yc�A�-*

lossM �<�^��       �	`#5Yc�A�-*

loss�+�<�	g�       �	<�5Yc�A�-*

loss>��<��Mh       �	�Y6Yc�A�-*

loss�<z�Y�       �	27Yc�A�-*

loss���<��       �	��7Yc�A�-*

lossW�$=��K�       �	�k8Yc�A�-*

loss���<�A��       �	P9Yc�A�-*

loss_��<��:       �	P�9Yc�A�-*

loss$g�<���]       �	�_:Yc�A�-*

loss��~;Z       �	�;Yc�A�-*

loss�+=d�Tb       �	��;Yc�A�-*

loss��<���;       �	�x<Yc�A�-*

lossP�<S��       �	�=Yc�A�-*

loss<��<Wu�7       �	�#>Yc�A�-*

loss��)<�[!�       �	ؼ>Yc�A�-*

lossɪ=�+lP       �	�S?Yc�A�-*

lossV�R;��"�       �	��?Yc�A�-*

loss/+=j�ǐ       �	@�@Yc�A�-*

loss�5=�l       �	�AYc�A�-*

loss��<x��d       �	ܷAYc�A�-*

loss��=Wv�D       �	�OBYc�A�-*

loss�V�;��       �	��BYc�A�-*

lossWG�:X��       �	oCYc�A�-*

loss�	<���x       �		DYc�A�-*

loss�:6<�>��       �	W�DYc�A�-*

lossjd<UK�       �	�MEYc�A�-*

loss�i
=�a�       �	I�EYc�A�-*

lossry=��=       �	~�FYc�A�-*

lossq�<��       �	�&GYc�A�-*

loss�DC<{ kV       �	'�GYc�A�-*

losscI�=��el       �	�cHYc�A�-*

loss�<M���       �	GIYc�A�-*

loss���<���j       �	�IYc�A�-*

loss���;�       �	z6JYc�A�-*

loss�s<�q~�       �	��JYc�A�.*

loss�<��p       �	�qKYc�A�.*

loss4�<;#G�       �	�$LYc�A�.*

loss|�<���       �	�LYc�A�.*

loss�{�=�ꢑ       �	eVMYc�A�.*

loss��;5K�e       �	[�MYc�A�.*

loss ��<����       �	��NYc�A�.*

loss!Y�;1���       �	PYc�A�.*

lossݺ=���4       �	f�PYc�A�.*

loss;�	<1�{�       �	��QYc�A�.*

loss��<���       �	pRYc�A�.*

loss\�;�ț       �	�SYc�A�.*

lossj�=�5�       �	��SYc�A�.*

loss=:�<F(s(       �	!�TYc�A�.*

loss��<�	"       �	��UYc�A�.*

lossɆ�<�J       �	'�VYc�A�.*

lossq=H<���       �	�WYc�A�.*

loss��;�.�W       �	L�WYc�A�.*

lossTA�<�rP�       �	�YYc�A�.*

loss)�d<ׄR�       �	ܛYYc�A�.*

lossh%={T��       �	w0ZYc�A�.*

loss��<�Պ       �	��ZYc�A�.*

lossZ�<im)�       �	�x[Yc�A�.*

loss�"=�U�>       �	�\Yc�A�.*

loss��=.�       �	_�\Yc�A�.*

loss��=����       �	�K]Yc�A�.*

loss�n=.��       �	��]Yc�A�.*

lossMgO<�
��       �	�^Yc�A�.*

loss�<6�X�       �	)_Yc�A�.*

loss��p=<9�7       �	��_Yc�A�.*

loss�ԓ<���w       �	�``Yc�A�.*

lossHPr<��a�       �	'PaYc�A�.*

loss���<��|p       �	�bYc�A�.*

loss��^;k�	       �	$�bYc�A�.*

loss��<<l�(       �	TWcYc�A�.*

loss�ֿ:��P       �	��cYc�A�.*

loss�t!<�W�       �	m�dYc�A�.*

loss��h<fH��       �	n�eYc�A�.*

loss�RD=�&Gu       �	t'fYc�A�.*

loss?�d<J�I�       �	{�fYc�A�.*

lossw9�;sO��       �	�}gYc�A�.*

lossc�6=7r��       �	hYc�A�.*

lossf��<F�J-       �	o�hYc�A�.*

loss���;]��J       �	iiYc�A�.*

lossQe�<Yކ�       �	�jYc�A�.*

lossXU�;a���       �	�jYc�A�.*

loss�,Y:ӂz�       �	"PkYc�A�.*

loss=��:�o�I       �	
�kYc�A�.*

loss�U�<��       �	��lYc�A�.*

loss���;t�S       �	['mYc�A�.*

loss��<���       �	#�mYc�A�.*

loss�X�9}��       �	�XnYc�A�.*

loss�2�=���       �	XoYc�A�.*

loss3�6:ˏ:a       �	1�oYc�A�.*

lossӜ�8��f       �	�SpYc�A�.*

loss��1;[��       �	��pYc�A�.*

loss$L�<�� �       �	)�qYc�A�.*

loss�	v<�}k       �	5rYc�A�.*

loss��;�p�F       �	�LsYc�A�.*

lossG� ;��^�       �	v�sYc�A�.*

loss��<���       �	5�tYc�A�.*

loss1�I=����       �	HvYc�A�.*

loss/G<0T       �	awYc�A�.*

lossO'�;gVBM       �	77xYc�A�.*

loss*=/S�       �	�yYc�A�.*

loss�;�<\i��       �	�yYc�A�.*

lossA|<9���       �	fzYc�A�.*

loss�a<�S	U       �	N{Yc�A�.*

loss��;��3       �	$�{Yc�A�.*

loss�zC<d{�)       �	rN|Yc�A�.*

lossc3�;/�D�       �	�|Yc�A�.*

loss�=N,MR       �	 �}Yc�A�.*

lossN�;=�ޚ       �	&~Yc�A�.*

loss]mb=e�\q       �	��~Yc�A�.*

loss��=WL;�       �	^Yc�A�.*

loss*��<HdY�       �	�Yc�A�.*

lossH�.<�J4�       �	£�Yc�A�.*

lossx�=�Un       �	cC�Yc�A�.*

loss�=�g�        �	��Yc�A�.*

loss(��;ŗ��       �	��Yc�A�.*

lossj��<`"�       �	L3�Yc�A�.*

loss3d;=��       �	l̃Yc�A�.*

loss`4o<��>       �	Re�Yc�A�.*

loss�*<k"9       �	���Yc�A�.*

loss4c<t{�|       �	j��Yc�A�.*

loss�s�;(�Dx       �	�5�Yc�A�.*

lossxpl;@�9�       �	`̆Yc�A�.*

lossn�-<嶐�       �	n�Yc�A�.*

loss�<�_ZG       �	��Yc�A�.*

loss��.;����       �	>�Yc�A�.*

lossn��<��2�       �	�y�Yc�A�.*

lossa �<^a�^       �	�(�Yc�A�.*

losse^;�nL       �	�Yc�A�.*

loss:B<Y�       �	�X�Yc�A�.*

loss��<�>h       �	g�Yc�A�.*

lossh0";ܳ<�       �	.��Yc�A�.*

loss|0<1h��       �	�'�Yc�A�.*

loss�5g;���       �	̍Yc�A�.*

loss_�;	���       �	$b�Yc�A�.*

loss�X/;���       �	���Yc�A�.*

loss�=<���       �	���Yc�A�.*

loss��=yJkN       �	Y1�Yc�A�.*

loss�_<��       �	�ǐYc�A�.*

lossƾN<h6��       �	��Yc�A�.*

loss���<�R��       �	�C�Yc�A�.*

lossWL<��v       �	�ْYc�A�.*

loss��;�>�       �	�u�Yc�A�.*

lossU�<����       �	C9�Yc�A�.*

loss|l�<#��       �	�ٔYc�A�.*

loss�+ ;���       �	�r�Yc�A�.*

loss`v�;����       �	��Yc�A�.*

loss�p�;��       �	���Yc�A�.*

loss���;�T�       �	GU�Yc�A�.*

lossTL�;�]��       �	@�Yc�A�.*

loss��U<�d}K       �	cծYc�A�.*

loss��<�`�       �		l�Yc�A�.*

loss�
4<>C�       �	���Yc�A�.*

lossS�<>�Gy       �	z�Yc�A�.*

loss�$_=��!�       �	�ʱYc�A�.*

loss*-!<�R?M       �	gd�Yc�A�.*

lossV<~��       �	���Yc�A�.*

loss��<=li       �	ˢ�Yc�A�.*

loss�`<2Aj<       �	TS�Yc�A�.*

loss  ;鞲Z       �	1^�Yc�A�.*

lossC=��z]       �	�u�Yc�A�.*

loss�52=v�#4       �	�+�Yc�A�.*

loss���<��3�       �	��Yc�A�.*

loss!8�<��L`       �	�(�Yc�A�.*

loss���<�}��       �	��Yc�A�/*

loss�{�:��G       �	E��Yc�A�/*

loss��:�0��       �	bi�Yc�A�/*

lossh�B<'��S       �	(�Yc�A�/*

lossJ�<��`)       �	��Yc�A�/*

loss�B�<i��       �	�Yc�A�/*

loss�b�;+�6       �	I��Yc�A�/*

loss���:
�=�       �	���Yc�A�/*

loss$�<��s       �	���Yc�A�/*

loss��;��,       �	�#�Yc�A�/*

loss@�<�PZ�       �	1��Yc�A�/*

loss���<�_;       �	���Yc�A�/*

loss	��<
*(       �	g��Yc�A�/*

loss��e=���       �	�W�Yc�A�/*

lossZVE;`S��       �	���Yc�A�/*

loss��K<[Ut       �	���Yc�A�/*

loss,=��u�       �	�i�Yc�A�/*

lossȒi<�r�       �	�`�Yc�A�/*

lossW%�=��uH       �	�Yc�A�/*

loss�!�<��!       �	��Yc�A�/*

loss�&=,	�t       �	^/�Yc�A�/*

loss�5u<���       �	.��Yc�A�/*

loss��X=�6�       �	l[�Yc�A�/*

lossٞ	=��u       �	o��Yc�A�/*

loss_�<���j       �	���Yc�A�/*

loss��<p�8       �	v�Yc�A�/*

loss��C<���x       �	��Yc�A�/*

loss��<���       �	���Yc�A�/*

lossdH<��       �	.X�Yc�A�/*

loss��<���       �	��Yc�A�/*

loss�-=#�z        �	���Yc�A�/*

loss��o<v��H       �	�~�Yc�A�/*

losst��<PqG/       �	��Yc�A�/*

lossLIN<SA       �	̵�Yc�A�/*

lossl��;�g��       �	rO�Yc�A�/*

loss.�7<,�       �	���Yc�A�/*

lossA�=�0S�       �	h��Yc�A�/*

losso�<�W�       �	�Y�Yc�A�/*

loss4��=Xj46       �	���Yc�A�/*

loss�:�<��iT       �	1��Yc�A�/*

lossQo;���Y       �	Q1�Yc�A�/*

loss�u;�%�       �	:��Yc�A�/*

loss�w�:����       �	�b�Yc�A�/*

loss�ߖ;97|�       �	���Yc�A�/*

loss��<o�a       �	��Yc�A�/*

loss|��=_��       �	�,�Yc�A�/*

lossSf�;��b       �	���Yc�A�/*

loss���=��.       �	�n�Yc�A�/*

loss3[=mP��       �	��Yc�A�/*

loss��<��J|       �	��Yc�A�/*

loss�h<A+       �	1A�Yc�A�/*

lossT�G<9[�       �	L��Yc�A�/*

loss���;����       �	y�Yc�A�/*

lossl��<��E�       �	Z�Yc�A�/*

loss� �<�y�G       �	���Yc�A�/*

loss�sk<R��       �	d>�Yc�A�/*

loss�<i2        �	���Yc�A�/*

loss�<h��	       �	�m�Yc�A�/*

loss�P; �Y       �	��Yc�A�/*

loss*$#< ?/�       �	F��Yc�A�/*

loss�x=Ќ�g       �	/��Yc�A�/*

lossS��<��|.       �	�g�Yc�A�/*

lossq_�<O� �       �	
�Yc�A�/*

loss�J<�;�       �	.��Yc�A�/*

loss��<]���       �	�N�Yc�A�/*

lossL<#=�ȩ7       �	���Yc�A�/*

loss�F<�|6�       �	׆�Yc�A�/*

lossx��<�-�!       �	C�Yc�A�/*

loss�gi: ��       �	7��Yc�A�/*

lossx�;tq%       �	(��Yc�A�/*

lossDz&;k�W�       �	��Yc�A�/*

loss=kp<�`�P       �	E��Yc�A�/*

loss�s<;Cz       �	���Yc�A�/*

loss�ܺ<,]�H       �	I+�Yc�A�/*

loss
MQ<��*N       �	���Yc�A�/*

loss�sH<�x�t       �	5`�Yc�A�/*

loss`�<��       �	]��Yc�A�/*

loss��R;���       �	ѓ�Yc�A�/*

loss��*<��X�       �	j0�Yc�A�/*

loss�F+=���       �	T��Yc�A�/*

loss��q;t ��       �	;��Yc�A�/*

loss�� =�4�       �	h"�Yc�A�/*

loss9��<���       �	{��Yc�A�/*

lossK�=(W6�       �	>\�Yc�A�/*

loss?!�<�eS       �	��Yc�A�/*

loss���<.�J�       �	N��Yc�A�/*

loss�w:��!       �	�M�Yc�A�/*

loss�<�R3p       �	���Yc�A�/*

loss]?<=���       �	F{�Yc�A�/*

loss�!�;y�~       �	���Yc�A�/*

lossR�p=��b       �	���Yc�A�/*

loss
��:���       �	�}�Yc�A�/*

lossH�_;c�`�       �	4�Yc�A�/*

loss�,e<�8I       �	���Yc�A�/*

loss���<ܘ�T       �	�Yc�A�/*

loss�U�<�ՙ
       �	�Yc�A�/*

lossq��<�-�1       �	��Yc�A�/*

loss�dO<z�_y       �	!;�Yc�A�/*

lossfDp;�$�       �	|��Yc�A�/*

loss���:E�       �	�w�Yc�A�/*

lossc}:DYq       �	# Yc�A�/*

loss��<�TW       �	�� Yc�A�/*

loss7�<�ۋ�       �	8MYc�A�/*

loss-%<BR��       �	��Yc�A�/*

loss��<߽U�       �	��Yc�A�/*

loss]O�<_���       �	�=Yc�A�/*

lossoV=`ڊM       �	��Yc�A�/*

loss��T<����       �	BwYc�A�/*

lossH�;���x       �	�Yc�A�/*

loss���<u4��       �	k�Yc�A�/*

loss���;�"vX       �	YYc�A�/*

loss
~�;*q-�       �	��Yc�A�/*

lossqZ;�b�       �	��Yc�A�/*

loss��<�M�       �	9	Yc�A�/*

loss8N&=��I       �	8�	Yc�A�/*

loss5q<.?6�       �	1y
Yc�A�/*

loss��(=?�       �	�Yc�A�/*

loss��<���       �	5�Yc�A�/*

loss<��<�gN�       �	�LYc�A�/*

loss-B9<��/�       �	��Yc�A�/*

loss� <���l       �	�|Yc�A�/*

loss�O�<�7}       �	UYc�A�/*

loss�ew<.��       �	��Yc�A�/*

lossd+)<'�Y       �	$FYc�A�/*

loss�9�<�{�h       �	�DYc�A�/*

loss�UF;�ﶕ       �	��Yc�A�/*

loss;b!<Bk��       �	�Yc�A�/*

loss�2�<e	�       �	I/Yc�A�/*

loss��=7w�R       �	��Yc�A�0*

loss�4�<^��       �	=�Yc�A�0*

lossC��<�FC       �	�Yc�A�0*

loss�<3ŕ       �	g,Yc�A�0*

lossE�;=��J'       �	&�Yc�A�0*

loss~;�B+       �	�ZYc�A�0*

loss��%<z9�       �	9�Yc�A�0*

lossT;9<�8�       �	��Yc�A�0*

loss�{<�K�       �	j0Yc�A�0*

losszX�=-�r�       �	��Yc�A�0*

lossr�<��       �	�sYc�A�0*

lossT=���       �	Yc�A�0*

loss>d;94|�       �	��Yc�A�0*

loss @�:f�"�       �	�6Yc�A�0*

loss�^�<���(       �	K�Yc�A�0*

loss���< >�z       �	msYc�A�0*

loss�P�;5��       �	
Yc�A�0*

lossn_�<4�V�       �	R�Yc�A�0*

loss���;��>       �	R Yc�A�0*

loss+׀<���:       �	*� Yc�A�0*

lossdh<.K=       �	�!Yc�A�0*

loss�)�=��_�       �	�)"Yc�A�0*

loss�.�<����       �	�"Yc�A�0*

loss��K;���B       �	�V#Yc�A�0*

lossX�=/?�       �	��#Yc�A�0*

loss�%0;��yz       �		�$Yc�A�0*

loss��p<�^&s       �	�E%Yc�A�0*

loss=ا=���       �	H�%Yc�A�0*

loss�X;��       �	&s&Yc�A�0*

loss G;�Er       �	�'Yc�A�0*

loss�<KҖ       �	�'Yc�A�0*

lossoF�<b�       �	=(Yc�A�0*

lossE��;��`?       �	h�(Yc�A�0*

loss[�5=��4n       �	{g)Yc�A�0*

lossJ� =è+�       �	X�)Yc�A�0*

loss�-<���O       �	ݕ*Yc�A�0*

loss.�)=��       �	�h+Yc�A�0*

loss P�:
�=       �	�,Yc�A�0*

lossN�h<`��       �	)$-Yc�A�0*

loss#�t<�|�       �	X�-Yc�A�0*

loss�H; �       �	yt.Yc�A�0*

loss��<�m       �	�/Yc�A�0*

loss��;�Y]�       �	T�/Yc�A�0*

loss=r-=��       �	NF0Yc�A�0*

loss\\!<�xR�       �	��0Yc�A�0*

loss�H�<�r=l       �	�}1Yc�A�0*

loss}kB<e��       �	q 2Yc�A�0*

loss4�=W1<�       �	f�2Yc�A�0*

loss�<����       �	�X3Yc�A�0*

loss	�t;�L�       �	��3Yc�A�0*

loss7:�;�(�`       �	T�4Yc�A�0*

lossڞ<;�ozB       �	Z)5Yc�A�0*

loss�<d�~       �	B�5Yc�A�0*

loss׏<� �       �	��6Yc�A�0*

lossG�<s��r       �	;�7Yc�A�0*

loss t�<v� �       �	Ot8Yc�A�0*

lossq��;H�!       �	�
9Yc�A�0*

lossd�<�Ҟ       �	�_:Yc�A�0*

loss<`;���       �	4�:Yc�A�0*

loss��<طs%       �	ۉ;Yc�A�0*

lossLy�;}#+9       �	-!<Yc�A�0*

loss �K<�I       �	+�<Yc�A�0*

loss��<�0       �	�t=Yc�A�0*

lossL��<FkT       �	l
>Yc�A�0*

loss�]<
���       �	Ǟ>Yc�A�0*

lossآ�<���9       �	�5?Yc�A�0*

loss���<ID��       �	c�?Yc�A�0*

lossF��<���       �	pa@Yc�A�0*

loss��<Ah��       �	p	AYc�A�0*

loss��;H       �	ʥAYc�A�0*

loss��|;�TVQ       �	7�CYc�A�0*

loss�Ve<��>�       �	ԂDYc�A�0*

loss�1=��p       �	/EYc�A�0*

lossq�P={!i�       �	��EYc�A�0*

lossď!=��u�       �	�PFYc�A�0*

loss(��=��<       �	��FYc�A�0*

lossfq=�q�B       �	A~GYc�A�0*

loss�,]<�X��       �	^HYc�A�0*

loss��<>�')       �	s�HYc�A�0*

loss1�{<��       �	jPIYc�A�0*

lossQ��=53 w       �	C�IYc�A�0*

loss��H<eb��       �	V�JYc�A�0*

loss�Y�<[!�       �	 FKYc�A�0*

loss&f�;Jf��       �	��KYc�A�0*

loss���;VY&       �	ʉLYc�A�0*

lossO��<�M�       �	�,MYc�A�0*

losst`D<��`K       �	��MYc�A�0*

loss�;P�C�       �	DlNYc�A�0*

loss\X1<�i8       �	�OYc�A�0*

loss2��;�       �	��OYc�A�0*

loss4W==t7�6       �	@hPYc�A�0*

loss���<��       �	�]QYc�A�0*

loss$[�<ύX�       �	7 RYc�A�0*

loss��<R��L       �	��RYc�A�0*

loss��<P�q       �	�7TYc�A�0*

lossȊ#=j<       �	�TYc�A�0*

loss�<<��        �	vnUYc�A�0*

lossW97;#�&       �	�VYc�A�0*

loss|�<�~��       �	��VYc�A�0*

lossqg=ȹ��       �	qXWYc�A�0*

lossZ��<RF�       �	��WYc�A�0*

loss��<c��O       �	`�XYc�A�0*

loss�"<��{       �	i5YYc�A�0*

loss|�<���o       �	�YYc�A�0*

loss�X<�!       �	MjZYc�A�0*

loss|=L���       �	c	[Yc�A�0*

loss '<h���       �	�[Yc�A�0*

lossH:<��<0       �	�@\Yc�A�0*

loss�d�<����       �	n�\Yc�A�0*

lossv��;,��4       �	}y]Yc�A�0*

loss$H�;Ї       �	^Yc�A�0*

loss�]C<��*�       �	a�^Yc�A�0*

lossA�<l�36       �	V_Yc�A�0*

losstE�<l�R�       �	c�_Yc�A�0*

loss���;��h       �	��`Yc�A�0*

loss���;��$�       �	~9aYc�A�0*

lossͯ�<WE֔       �	��aYc�A�0*

loss��U<#�        �	�ebYc�A�0*

loss���<�g��       �	&�bYc�A�0*

loss�Br=�w�s       �	�cYc�A�0*

lossd��;y��       �	�9dYc�A�0*

loss3��<vKn�       �	�dYc�A�0*

loss���;V8s�       �	MeeYc�A�0*

loss��$=Y�P�       �	+�eYc�A�0*

losscgj<S���       �	ɓfYc�A�0*

loss3�<�5�
       �	*gYc�A�0*

loss�=��`�       �	"�gYc�A�0*

loss��<B��       �	UhYc�A�0*

loss�g<F��i       �	 �hYc�A�1*

loss4��<Q�\�       �	�iYc�A�1*

loss1�<t�7�       �	�jYc�A�1*

lossQ>�ܼ�       �	��jYc�A�1*

loss r�<<��       �	�NkYc�A�1*

loss�<w<rVN       �	��kYc�A�1*

loss�g;ǔ�       �	�xlYc�A�1*

loss���<�[mc       �	mYc�A�1*

losso�<]|
       �	��mYc�A�1*

loss��.:�O�z       �	2:nYc�A�1*

lossZ.Z;Nd��       �	��nYc�A�1*

lossis�;˺,�       �	�hoYc�A�1*

loss��Q=����       �	�oYc�A�1*

loss�)=�N��       �	��pYc�A�1*

loss�=�<�WM�       �	~5qYc�A�1*

loss\�9=�i,       �	�rYc�A�1*

loss�a�;)x�       �	ˠrYc�A�1*

loss��=<�f       �	�@sYc�A�1*

loss&��:�١�       �	��sYc�A�1*

loss��Q;{m)#       �	`vtYc�A�1*

loss�b�<�yc       �	�uYc�A�1*

lossܓ�;���       �	!�uYc�A�1*

loss\ʴ:7j       �	�vYc�A�1*

loss�e�<�5(�       �	#NwYc�A�1*

loss�n�<�	       �	_)xYc�A�1*

loss䐧:]w�`       �	�lyYc�A�1*

loss���<�a�       �	�zYc�A�1*

loss��X="Wڳ       �	F�zYc�A�1*

lossɏ&=-M�       �	&W{Yc�A�1*

loss�`
=`�G       �	p�{Yc�A�1*

loss�^�<xmԓ       �	��|Yc�A�1*

loss�u�;�h$       �	9)}Yc�A�1*

lossb�=�VY       �	��}Yc�A�1*

lossc<���,       �	'N~Yc�A�1*

loss!��<A�!�       �	;�~Yc�A�1*

loss�jw;����       �	�}Yc�A�1*

lossJ��<
��       �	n�Yc�A�1*

loss�5<�WP-       �	欀Yc�A�1*

lossu�<o��       �	![�Yc�A�1*

loss�;����       �	��Yc�A�1*

lossJ&I<�$8       �	^��Yc�A�1*

loss��M<�O-       �	p"�Yc�A�1*

loss��;3ZX�       �	��Yc�A�1*

loss
_E<TD��       �	���Yc�A�1*

loss�C6=�:�       �	vS�Yc�A�1*

loss!�A=���       �	���Yc�A�1*

loss��<]ﳪ       �	���Yc�A�1*

lossMw<=���       �	;�Yc�A�1*

loss���<z���       �	�ׇYc�A�1*

loss�'B<_�1       �	�r�Yc�A�1*

lossH�(=r��       �	��Yc�A�1*

loss���<�R�y       �	���Yc�A�1*

lossf��;C�       �	UL�Yc�A�1*

loss�_�;�V��       �	�Yc�A�1*

lossd�=�g��       �	S��Yc�A�1*

loss7�=n�S�       �	�*�Yc�A�1*

loss���;���       �	*ČYc�A�1*

loss�/�<��\�       �	Y�Yc�A�1*

lossE�K=8� �       �	��Yc�A�1*

loss�/�<��Hv       �	y��Yc�A�1*

loss���;���       �	>�Yc�A�1*

loss�al;*4sa       �	��Yc�A�1*

loss���<����       �	���Yc�A�1*

lossA}'=��Y9       �	��Yc�A�1*

loss=��;|�fI       �	.ȑYc�A�1*

lossd��<��T       �	�^�Yc�A�1*

lossL�b=��       �	���Yc�A�1*

loss��i;TQ	l       �	G�Yc�A�1*

loss��:���       �	���Yc�A�1*

loss:/�;���       �	�Q�Yc�A�1*

loss��;Ͽ��       �	[�Yc�A�1*

loss��O<g�w>       �	���Yc�A�1*

loss��=�KW}       �	;�Yc�A�1*

lossS�w=��       �	QؗYc�A�1*

loss!I<Y��       �	�t�Yc�A�1*

loss�<�;���       �	�	�Yc�A�1*

loss�rq=�:	5       �	]��Yc�A�1*

losspe<˫��       �	�=�Yc�A�1*

loss�Q-=���       �	�Yc�A�1*

loss|�v;=gN       �	�t�Yc�A�1*

loss@M<�U�W       �	��Yc�A�1*

lossoۖ<u c�       �	��Yc�A�1*

loss\��<�]�D       �	zR�Yc�A�1*

loss\4]<e���       �	��Yc�A�1*

loss��<�2q�       �	���Yc�A�1*

loss=�Q<Ki�       �	$�Yc�A�1*

loss�t�<�ߘ�       �	@Yc�A�1*

loss��<nk~v       �	�c�Yc�A�1*

lossa��<�j�H       �	@/�Yc�A�1*

loss��o<X�|�       �	��Yc�A�1*

loss&�;�Q=�       �	�{�Yc�A�1*

loss�>�<��T       �	�!�Yc�A�1*

loss4d<=W%��       �	CƣYc�A�1*

loss�=�=�|       �	���Yc�A�1*

loss8�=���$       �	f2�Yc�A�1*

loss�u�<+���       �	åYc�A�1*

loss;b�;�J��       �	�b�Yc�A�1*

loss�}1<�k�       �	���Yc�A�1*

loss<m=���(       �	�ѧYc�A�1*

loss�&O=U�       �	�k�Yc�A�1*

lossV�;o��       �	i�Yc�A�1*

loss,�b<~��       �	���Yc�A�1*

loss -�<isp       �	�R�Yc�A�1*

loss��<H,�       �	��Yc�A�1*

loss-A�<!�d�       �	��Yc�A�1*

loss�^�<F!;�       �	P�Yc�A�1*

loss���<_��       �	h��Yc�A�1*

loss3%J=DӶ�       �	L�Yc�A�1*

lossD[=���       �	��Yc�A�1*

loss�T<��       �	^��Yc�A�1*

loss�̱<��L       �	��Yc�A�1*

loss��<]�       �	O��Yc�A�1*

lossS=��6       �	II�Yc�A�1*

lossI�-;�Uw�       �	T�Yc�A�1*

loss�6\=����       �	:v�Yc�A�1*

lossOk�<0�6�       �	x�Yc�A�1*

lossõn<:̻�       �	���Yc�A�1*

loss���;a
=$       �	�5�Yc�A�1*

loss�G�;ttm       �	۳Yc�A�1*

loss�>�;���^       �	`r�Yc�A�1*

loss]�[<O{       �	��Yc�A�1*

loss��/<��}�       �	���Yc�A�1*

lossw-�<@Y��       �	�?�Yc�A�1*

lossI��;���c       �	�Yc�A�1*

loss79;<x��       �	���Yc�A�1*

loss��b:8��       �	�=�Yc�A�1*

loss���:�8d�       �	�t�Yc�A�1*

loss�}�;����       �	�z�Yc�A�1*

loss�[ <��       �	�2�Yc�A�2*

loss���;u��       �	�ӼYc�A�2*

loss�v�;@�	�       �	�m�Yc�A�2*

loss�=^��2       �	G�Yc�A�2*

lossD
$<�[@       �	��Yc�A�2*

loss�0!=m��;       �	���Yc�A�2*

loss�K�<�BĆ       �	e��Yc�A�2*

loss��=���       �	�@�Yc�A�2*

loss��;�~/~       �	N��Yc�A�2*

loss�<<Q�       �	�m�Yc�A�2*

lossv�#<�!�[       �	��Yc�A�2*

loss�V�<딱�       �	C��Yc�A�2*

loss�|�<YT�Q       �	A�Yc�A�2*

lossnZ;h�       �	I��Yc�A�2*

loss]�*<��}       �	y��Yc�A�2*

loss�=;h�       �	.�Yc�A�2*

loss]�I=����       �	���Yc�A�2*

loss���:L1pv       �	�b�Yc�A�2*

lossr��;$YJ�       �	�	�Yc�A�2*

loss	G�<�rom       �	E��Yc�A�2*

loss��<|��       �	9�Yc�A�2*

lossSL�<ojH       �	i��Yc�A�2*

loss|�<Pޅ       �	(��Yc�A�2*

loss��:3չ
       �	�b�Yc�A�2*

loss�=��d       �	��Yc�A�2*

loss\�G=,��       �	r��Yc�A�2*

loss6��;h���       �	"T�Yc�A�2*

loss]}<y��`       �	���Yc�A�2*

loss�K�<͐�       �	\�Yc�A�2*

lossڽ�;�"�d       �	���Yc�A�2*

lossl0;���$       �	q:�Yc�A�2*

loss��<�#�       �	 ��Yc�A�2*

loss�u$<���       �	���Yc�A�2*

loss�<<��E=       �	�1�Yc�A�2*

loss\M�<SQ��       �	���Yc�A�2*

loss·�=��'       �	G��Yc�A�2*

loss6�=�o�       �	p#�Yc�A�2*

lossR�*=s�I       �	���Yc�A�2*

loss ��<��V       �	���Yc�A�2*

lossx�;,?3�       �	�<�Yc�A�2*

lossn�;�&��       �	p��Yc�A�2*

lossC,w<���{       �	��Yc�A�2*

loss�e�=_��       �	�w�Yc�A�2*

loss&�|<|M�       �	��Yc�A�2*

loss�S�=�Q��       �	���Yc�A�2*

loss��; ��       �	1C�Yc�A�2*

loss	$D<kd��       �	���Yc�A�2*

loss}Ԙ=���       �	�{�Yc�A�2*

loss���<�~�       �	@�Yc�A�2*

loss�<
��N       �	c��Yc�A�2*

lossr2=W�5�       �	M��Yc�A�2*

lossO�D=���N       �	&�Yc�A�2*

loss�;Q��K       �	���Yc�A�2*

loss�BV;v��       �	@Q�Yc�A�2*

loss�o�=I��.       �	F��Yc�A�2*

losso�=��*7       �	�~�Yc�A�2*

loss��n<�h�       �	��Yc�A�2*

lossW�*<vv��       �	ѭ�Yc�A�2*

loss�s<Om��       �	F�Yc�A�2*

loss�g�<�Io       �	���Yc�A�2*

loss?x�<�@�       �	v�Yc�A�2*

loss�A�<��B<       �	��Yc�A�2*

lossS��;?"�       �	���Yc�A�2*

loss;��< l�       �	�4�Yc�A�2*

loss�G=�K��       �	2��Yc�A�2*

loss�r�=~��G       �	1^�Yc�A�2*

loss8݉<��[M       �	#0�Yc�A�2*

loss�Hy<���       �	Q��Yc�A�2*

loss���;�k       �	LT�Yc�A�2*

loss:�=<d��       �	���Yc�A�2*

loss)ފ;�T�       �	��Yc�A�2*

loss5x =�7�!       �	p%�Yc�A�2*

loss�<�
<?       �	���Yc�A�2*

loss�.=e�       �	�U�Yc�A�2*

loss;�<v��       �	5��Yc�A�2*

loss��<]�       �	���Yc�A�2*

loss�[�;�UX�       �	�-�Yc�A�2*

lossQ��;���       �	���Yc�A�2*

loss���;�~P       �	/n�Yc�A�2*

loss�� <���       �	K�Yc�A�2*

loss��\<(5�       �	��Yc�A�2*

loss��y<`j1�       �	YP�Yc�A�2*

loss��<�Ht�       �	G��Yc�A�2*

loss�y�;D�"       �	Q��Yc�A�2*

lossy�<�-�       �	~�Yc�A�2*

loss`�K<5B�       �	F��Yc�A�2*

loss��B<	���       �	K�Yc�A�2*

loss�b�< �       �	���Yc�A�2*

losswd=��Zu       �	Sz�Yc�A�2*

loss�9Z;�p       �	��Yc�A�2*

lossv'X;G)�       �	9��Yc�A�2*

loss�N�<�g~�       �	�x�Yc�A�2*

losss�<ߔ�a       �	�=�Yc�A�2*

loss��:���F       �	e5�Yc�A�2*

loss�V�;�x�       �	���Yc�A�2*

loss/k�<qX�       �	��Yc�A�2*

loss �*<�Y       �	B�Yc�A�2*

loss%�o=�b��       �	���Yc�A�2*

lossL[�<}��]       �	Ț�Yc�A�2*

lossڸu<�{�       �	�@�Yc�A�2*

losstQ�<v�`�       �	n��Yc�A�2*

lossh��<Ϙ       �	�t�Yc�A�2*

lossX�(;��       �	�L Yc�A�2*

lossZX�;�Ɩc       �	G� Yc�A�2*

loss�*Q<�40J       �	�}Yc�A�2*

loss��;�_{       �	�Yc�A�2*

loss���;|�N       �	��Yc�A�2*

loss!Y�<�?^%       �	ipYc�A�2*

loss�،<�ө       �	0Yc�A�2*

loss�r<�!�D       �	p�Yc�A�2*

loss6G<�E��       �	NYc�A�2*

loss�<���*       �	x�Yc�A�2*

loss��f<���       �	�Yc�A�2*

loss���<���       �	LUYc�A�2*

loss��@<I�<N       �	��Yc�A�2*

lossO�</A�       �	��Yc�A�2*

loss|H=�bHD       �	RC	Yc�A�2*

loss}Py=���       �	��	Yc�A�2*

lossR�;��       �	�
Yc�A�2*

loss�U;�ӹ�       �	�;Yc�A�2*

loss��9<0�6_       �	��Yc�A�2*

loss�$:~�e^       �	}yYc�A�2*

lossbn=�B       �	�Yc�A�2*

loss#�4:��ɍ       �	��Yc�A�2*

loss�Ҹ;���       �	�BYc�A�2*

loss_w�<E��       �	{�Yc�A�2*

lossr�< ��       �	�tYc�A�2*

lossxu<�� �       �	�Yc�A�2*

lossi=n���       �	�Yc�A�3*

loss��= F�       �	LYc�A�3*

loss�G<�g       �	��Yc�A�3*

loss��	;��J       �	�Yc�A�3*

loss��=;�1TI       �	�eYc�A�3*

lossH ;�7iv       �	�1Yc�A�3*

loss;H�9�>�       �	S�Yc�A�3*

loss���;kix       �	�`Yc�A�3*

loss��:�2��       �	��Yc�A�3*

lossF<���p       �	�Yc�A�3*

loss�!�<��}[       �	�.Yc�A�3*

loss���9n%\�       �	��Yc�A�3*

loss�?<��       �	�{Yc�A�3*

loss�{;���       �	'Yc�A�3*

lossݨ�9cFy�       �	��Yc�A�3*

loss�9�:�Ň       �	pDYc�A�3*

loss�%�;�ш�       �	��Yc�A�3*

loss:W=[eǑ       �	TtYc�A�3*

loss|�;ޱ1K       �	1
Yc�A�3*

loss�!;��v�       �	f�Yc�A�3*

loss?;��L�       �	K=Yc�A�3*

lossJd='�|       �	�Yc�A�3*

loss�]�:5���       �	jYc�A�3*

loss��<[q�       �	� Yc�A�3*

loss���<�*%�       �	5�Yc�A�3*

loss���<��=�       �	�V Yc�A�3*

lossz݈<�A��       �	�l!Yc�A�3*

loss��f<3�:t       �	�"Yc�A�3*

loss.~,=λ`�       �	��"Yc�A�3*

losso=�u+�       �	�o#Yc�A�3*

loss�lF<�#~       �	,$Yc�A�3*

loss��<�}=�       �	��$Yc�A�3*

lossF;8+�       �	�%Yc�A�3*

lossd<z�F       �	�)&Yc�A�3*

losszE�<U�ױ       �	��&Yc�A�3*

loss�=�;�n�_       �	�c'Yc�A�3*

loss��Q=��Dd       �	*(Yc�A�3*

loss Z�<d@�       �	Ƣ(Yc�A�3*

loss8�<ʕ�       �	�Y)Yc�A�3*

loss�d;�j�       �	��)Yc�A�3*

loss$�k=v��       �	F�*Yc�A�3*

lossmw;q�S�       �	�A+Yc�A�3*

lossҕg;�[0�       �	8�+Yc�A�3*

loss:e<-ׂ�       �	�~,Yc�A�3*

lossE5�;�s'C       �	D0-Yc�A�3*

lossڦo<ę��       �	��-Yc�A�3*

loss���;F��       �	�k.Yc�A�3*

lossT{ ;���9       �	�
/Yc�A�3*

loss�?K<V��       �	�/Yc�A�3*

loss� ;@��       �	6>0Yc�A�3*

lossTܞ<��[       �	�1Yc�A�3*

loss�|-=��&<       �	�1Yc�A�3*

loss��%<B�h       �	Q2Yc�A�3*

loss��<;0<       �	>�2Yc�A�3*

loss0�<���       �	׆3Yc�A�3*

loss�m;�yY       �	�%4Yc�A�3*

lossG�<�㑫       �	b�4Yc�A�3*

loss�; �       �	�`5Yc�A�3*

loss�(�;��d�       �	��5Yc�A�3*

losse�;g��       �	[�6Yc�A�3*

lossZ�=Z��       �	Tp7Yc�A�3*

loss�� <�}6       �	98Yc�A�3*

lossF�;�r,       �	D�8Yc�A�3*

loss�\}<��z       �	�B9Yc�A�3*

loss��;Af�       �	��9Yc�A�3*

lossɜ�;�Cg�       �	z:Yc�A�3*

lossSC<Gg�d       �	G;Yc�A�3*

loss)��=��2       �	��;Yc�A�3*

loss��<~'�1       �	��<Yc�A�3*

lossJw;����       �	Z=Yc�A�3*

lossf��;���8       �	��=Yc�A�3*

loss8#�;��Di       �	P�>Yc�A�3*

loss�NR<z�O       �	"6?Yc�A�3*

loss h�;Ŗ�K       �	��XYc�A�3*

loss�=�<���       �	�cYYc�A�3*

loss���<?z       �	�ZYc�A�3*

lossN7X<wZK}       �	
�ZYc�A�3*

lossrt|<��>       �	�[Yc�A�3*

loss�y�<�[j�       �	��\Yc�A�3*

loss��T<�V��       �	�Y]Yc�A�3*

loss;,�<\�       �	\^Yc�A�3*

lossO*`=O��9       �	R_Yc�A�3*

lossh�<>���       �	l`Yc�A�3*

loss��"<s�(�       �	��`Yc�A�3*

loss �y=��g�       �	�VbYc�A�3*

loss�$<�٩�       �	c(dYc�A�3*

lossRi�</���       �	z�dYc�A�3*

loss�;�<T(��       �	�oeYc�A�3*

lossLR=��       �	b�fYc�A�3*

loss�)<(�	       �	#�gYc�A�3*

lossa�;Tl?h       �	bhYc�A�3*

lossc�R<&
n       �	]�hYc�A�3*

loss=��<vM�       �	 �iYc�A�3*

lossh �<닡�       �	2jYc�A�3*

loss�{�;�NX}       �	��jYc�A�3*

loss��<��       �	E�kYc�A�3*

loss?L�<6Ȝ       �	�+lYc�A�3*

loss:�<c���       �	�lYc�A�3*

lossOڣ<���1       �	tmYc�A�3*

lossxG <'8J�       �	�nYc�A�3*

loss�b�:M�bv       �	�nYc�A�3*

loss��=x��       �	 �oYc�A�3*

loss�<sx�u       �	�MpYc�A�3*

loss�^H</NU�       �	��pYc�A�3*

lossCܼ;y
��       �	��qYc�A�3*

loss��;��]�       �	�~rYc�A�3*

loss��<$��       �	� sYc�A�3*

loss3�<��I�       �	N�sYc�A�3*

loss�@@=��]�       �	��tYc�A�3*

loss���;�Eb       �	n2uYc�A�3*

loss3�{;�NX8       �	�uYc�A�3*

loss�U=�m�       �	�svYc�A�3*

lossl�=4`�       �	uwYc�A�3*

lossϳ�<.�;       �	�wYc�A�3*

loss��F=�s�       �	�xYc�A�3*

loss�;(Oհ       �	��yYc�A�3*

lossˍ�<a�M/       �	�ZzYc�A�3*

loss`��<9��L       �	��zYc�A�3*

loss�R<i��       �	��{Yc�A�3*

losse��;���       �	�|Yc�A�3*

loss��[<eƛ       �	�T}Yc�A�3*

loss�=�& _       �	)~Yc�A�3*

lossa��:u��%       �	��~Yc�A�3*

lossSY�;Qyʭ       �	�=Yc�A�3*

lossf��<O��       �	��Yc�A�3*

loss���;>W��       �	$}�Yc�A�3*

loss��'>
:o�       �	~�Yc�A�3*

loss��D<����       �	��Yc�A�3*

loss��;&��       �	�T�Yc�A�4*

loss�dB:t�       �	���Yc�A�4*

loss���:j���       �	��Yc�A�4*

lossht�<��[�       �	C=�Yc�A�4*

loss8:�<(�h6       �	�Yc�A�4*

losso��=/��       �	꓅Yc�A�4*

loss�/<?�f-       �	�3�Yc�A�4*

loss��:��gU       �	�φYc�A�4*

loss��$=Z��       �	d��Yc�A�4*

loss�R�;PA       �	���Yc�A�4*

loss\V�;��       �	B@�Yc�A�4*

loss1�<���r       �	�ډYc�A�4*

loss0+�<���E       �	\w�Yc�A�4*

lossɽ=!�`       �	�?�Yc�A�4*

loss��$<m��       �	�Yc�A�4*

loss}�<�1 �       �	�Yc�A�4*

loss�3�<M��&       �	�?�Yc�A�4*

lossφ�<�M��       �	3ލYc�A�4*

loss�Ĉ;�ao4       �	:�Yc�A�4*

loss�q�<M���       �	v��Yc�A�4*

loss
��<�P=       �	)%�Yc�A�4*

lossmh<��Z       �	���Yc�A�4*

loss� <!/�       �	!Y�Yc�A�4*

loss�`;Z���       �	V��Yc�A�4*

loss$"j<|�K       �	ڎ�Yc�A�4*

loss��<�tC       �	�)�Yc�A�4*

loss��<�V       �	˓Yc�A�4*

loss�p�<)r_       �	�p�Yc�A�4*

loss���;�^��       �	��Yc�A�4*

loss��"=��by       �	X��Yc�A�4*

lossK�:���       �	�X�Yc�A�4*

lossdv�<ʢ�       �	��Yc�A�4*

lossl�<��ٽ       �	㎗Yc�A�4*

lossd0Y<���       �	*�Yc�A�4*

lossX�:<�?[�       �	�јYc�A�4*

loss+�=�]�       �	�v�Yc�A�4*

loss��<���       �	a�Yc�A�4*

loss��<���       �	c��Yc�A�4*

loss�0S<l��       �	�ޛYc�A�4*

loss-��<LAZ�       �	�Yc�A�4*

losst��<�sd(       �	� �Yc�A�4*

lossAc4=p� �       �	��Yc�A�4*

lossc�=���K       �	Υ�Yc�A�4*

loss��/=��=       �	sJ�Yc�A�4*

loss��@<��>�       �	!�Yc�A�4*

loss�O�;bw�       �	V~�Yc�A�4*

loss�"�;�U��       �	�&�Yc�A�4*

loss�t�<���       �	ܼ�Yc�A�4*

loss�D<sv��       �	�V�Yc�A�4*

loss�9<�*��       �	��Yc�A�4*

loss�.{=����       �	4��Yc�A�4*

loss�<�;!M��       �	��Yc�A�4*

lossQi=��       �	��Yc�A�4*

loss�x=�S�       �	J^�Yc�A�4*

lossͻ<���<       �	���Yc�A�4*

loss]��=q�       �	��Yc�A�4*

loss�A<�\D�       �	�:�Yc�A�4*

loss��}</�I       �	
�Yc�A�4*

lossM��:��-N       �	S��Yc�A�4*

lossn��;qq<�       �	$D�Yc�A�4*

loss�4�<O�N�       �	3ߩYc�A�4*

loss�K�;��q�       �	�w�Yc�A�4*

loss��=���       �	��Yc�A�4*

loss�;��	9       �	���Yc�A�4*

loss���<���s       �	<K�Yc�A�4*

loss�b"=N��2       �	��Yc�A�4*

lossn\�<��[       �	�Yc�A�4*

lossM�R<�e��       �	B�Yc�A�4*

loss���<DE��       �	���Yc�A�4*

loss�;0�
�       �	��Yc�A�4*

lossMĤ;G��B       �	�<�Yc�A�4*

lossA<	;kl �       �	}�Yc�A�4*

loss��A:f/4�       �	u��Yc�A�4*

loss���<2�T(       �	�,�Yc�A�4*

loss
��<>�:�       �	ʲYc�A�4*

losszw0<ot��       �	ji�Yc�A�4*

loss���<�z       �	�
�Yc�A�4*

lossV�<��N�       �	Y��Yc�A�4*

lossf#=Y�t       �	F�Yc�A�4*

loss��<��;&       �	K�Yc�A�4*

loss߆e<9�X.       �	��Yc�A�4*

loss*��<W=ϑ       �	4�Yc�A�4*

loss�M&=@i�&       �	}ϷYc�A�4*

loss��.<�o#�       �	7��Yc�A�4*

loss�� <���       �	+��Yc�A�4*

loss82/;�ޤ�       �	���Yc�A�4*

loss7�:<��ET       �	�&�Yc�A�4*

loss&��<�)N�       �	c�Yc�A�4*

loss�=�!��       �	�ͼYc�A�4*

loss�b=x�0�       �	T��Yc�A�4*

loss ­<�O��       �	5��Yc�A�4*

lossr�;�!N       �	�Yc�A�4*

loss���;=?��       �	]�Yc�A�4*

loss#s�:���       �	��Yc�A�4*

loss��<�s�       �	�D�Yc�A�4*

loss	=�;El�       �	���Yc�A�4*

lossf��<k\�/       �	*��Yc�A�4*

loss�=N�       �	y@�Yc�A�4*

loss�,=��&J       �	L��Yc�A�4*

lossM�<"�R$       �	q��Yc�A�4*

loss���;�p��       �	�*�Yc�A�4*

loss�y�;f}?P       �	���Yc�A�4*

loss��8<{ˏ       �	�h�Yc�A�4*

loss��=��1�       �		�Yc�A�4*

loss��;�Pr�       �	���Yc�A�4*

loss�#m<b�.       �	AD�Yc�A�4*

loss��<6�w       �	���Yc�A�4*

loss��A<����       �	��Yc�A�4*

loss��:���       �	� �Yc�A�4*

loss��<ر!       �	���Yc�A�4*

loss��\<��Π       �	�T�Yc�A�4*

loss�˖;%W�&       �	B��Yc�A�4*

loss̜�<�Y��       �	Ɗ�Yc�A�4*

loss�a/;�Z�       �	@/�Yc�A�4*

lossz�<�Tn�       �	���Yc�A�4*

lossq� =��_�       �	Kx�Yc�A�4*

loss{j�;C���       �	r�Yc�A�4*

loss�Ǯ;T6�<       �	˹�Yc�A�4*

lossZ��;��%       �	2X�Yc�A�4*

loss���:�/*�       �	~��Yc�A�4*

loss/җ=%�p�       �	b��Yc�A�4*

lossK�=�o�       �		6�Yc�A�4*

loss���<���       �	���Yc�A�4*

loss-�k<[�'       �	�o�Yc�A�4*

loss&k�;W�^�       �	J�Yc�A�4*

loss1B�:ɺ7�       �	Z��Yc�A�4*

loss�-4<Io��       �	v�Yc�A�4*

loss/�E=nn��       �	��Yc�A�4*

lossl~�;��       �	~��Yc�A�5*

loss�z�;^~%�       �	і�Yc�A�5*

loss�X<�v�       �	�2�Yc�A�5*

loss�=�ZD�       �	��Yc�A�5*

loss���=8�Ҫ       �	lv�Yc�A�5*

loss���<��'       �	c�Yc�A�5*

lossT�6=V��)       �	���Yc�A�5*

loss�0=/ٺF       �	oH�Yc�A�5*

lossW��;{o�S       �	���Yc�A�5*

lossZ(�<._.�       �	�Yc�A�5*

loss��<���       �	��Yc�A�5*

loss���;�Hǘ       �	��Yc�A�5*

loss��<�@k       �	�^�Yc�A�5*

loss�u�:,?X�       �	���Yc�A�5*

loss�Gn<�x       �	[��Yc�A�5*

loss�f�;�e�A       �	�.�Yc�A�5*

loss��<ȋn       �	��Yc�A�5*

loss�/�<�KG~       �	_]�Yc�A�5*

lossn�\;�7�v       �	�5�Yc�A�5*

loss�Z�<�3j�       �	���Yc�A�5*

loss��;C��9       �	Ui�Yc�A�5*

loss��_<�;1       �		�Yc�A�5*

loss��=J�h       �	���Yc�A�5*

loss
=.�       �	�@�Yc�A�5*

loss�Q=c;r�       �	���Yc�A�5*

loss�'$=\��       �	m��Yc�A�5*

loss�n%<'��       �	p\�Yc�A�5*

lossb��<��!�       �	b��Yc�A�5*

loss��3<�5�       �	���Yc�A�5*

loss�nx<�Tc�       �	WB�Yc�A�5*

losseQ�;sj�       �	���Yc�A�5*

loss�=�(d       �	؂�Yc�A�5*

loss�3=\�C       �	e�Yc�A�5*

loss�D=i��B       �	���Yc�A�5*

loss�L&=�,�       �	XV�Yc�A�5*

losstSn=1uy       �	���Yc�A�5*

loss��%=�$�       �	Ŏ�Yc�A�5*

loss�ܤ<�A�       �	�,�Yc�A�5*

loss�J�:n�."       �	���Yc�A�5*

loss�ID<i�e�       �	�y�Yc�A�5*

loss���<ق\�       �	�b�Yc�A�5*

loss�D=�p�       �	���Yc�A�5*

loss�=����       �	o��Yc�A�5*

loss��;�j��       �	�7�Yc�A�5*

lossE;�.��       �	��Yc�A�5*

loss��<�JOP       �	i�Yc�A�5*

loss], =ѝ       �	`�Yc�A�5*

lossC\�;.f�#       �	���Yc�A�5*

loss62�<�ML       �	C�Yc�A�5*

lossV��;�D�       �	���Yc�A�5*

loss���<:���       �	Po�Yc�A�5*

loss��	=�t�Z       �	c�Yc�A�5*

loss���<0uL       �	��Yc�A�5*

loss`�D=�H�0       �	�A�Yc�A�5*

loss��;�3�       �	��Yc�A�5*

loss�o�=��+       �	C��Yc�A�5*

loss�׾;z2��       �	i:�Yc�A�5*

loss�0<W�ֿ       �	^��Yc�A�5*

loss�4�<b�kk       �	�|�Yc�A�5*

loss�*�<�	p       �	@�Yc�A�5*

loss��G=#&��       �	0��Yc�A�5*

lossf9�<$Un�       �	?V�Yc�A�5*

lossu�<�T�G       �	���Yc�A�5*

loss#��;V�(�       �	3��Yc�A�5*

lossͥ�<��,       �	eR  Yc�A�5*

lossz��;o޲�       �	�  Yc�A�5*

loss��<����       �	؜ Yc�A�5*

loss�'�<�N,�       �	�E Yc�A�5*

loss}�<C��3       �	� Yc�A�5*

loss�� =x�1       �	�� Yc�A�5*

loss���;���       �	y; Yc�A�5*

loss�JW;��E�       �	�� Yc�A�5*

loss�-�;�`0       �	� Yc�A�5*

loss�<�`z       �	)  Yc�A�5*

loss�|\<$q�'       �	%� Yc�A�5*

lossV	j<�0C       �	�x Yc�A�5*

loss:��<`}%N       �	�& Yc�A�5*

lossfq�<�a��       �	w� Yc�A�5*

loss��=*L       �	��	 Yc�A�5*

lossfm�<�+�J       �	`
 Yc�A�5*

loss��<x�t�       �	��
 Yc�A�5*

loss�d=�Zv       �	h Yc�A�5*

loss��;�5��       �	� Yc�A�5*

lossP=�F�       �	�� Yc�A�5*

loss̍�<�b.�       �	� Yc�A�5*

loss��F=V�d       �	> Yc�A�5*

loss5�=<��       �	� Yc�A�5*

loss��_<����       �	�} Yc�A�5*

lossNi=+��       �	�& Yc�A�5*

lossT�Y<�%6`       �	W� Yc�A�5*

loss<�u<i       �	Cu Yc�A�5*

loss���=�x�       �	�! Yc�A�5*

lossǖ=��        �	{� Yc�A�5*

lossMJx=[M��       �	�\ Yc�A�5*

loss�l;;�w�       �	#� Yc�A�5*

loss<�;�-�t       �	!� Yc�A�5*

loss��<)PT       �	�1 Yc�A�5*

lossW�;_��       �	�� Yc�A�5*

lossЭ<��{(       �	�l Yc�A�5*

lossq��<���F       �	�� Yc�A�5*

lossL� <�#�p       �	o. Yc�A�5*

loss`4�<Ut?       �	� Yc�A�5*

loss���=66�       �	� Yc�A�5*

loss��=M-�       �	"P Yc�A�5*

lossj<P���       �	�* Yc�A�5*

loss�M�;�� n       �	h� Yc�A�5*

loss���;�W�f       �	1� Yc�A�5*

loss�#�;'F�       �	&R Yc�A�5*

loss��n<Y&�!       �	�: Yc�A�5*

loss8��;ٔ��       �	0� Yc�A�5*

losst��:U;p       �	5{ Yc�A�5*

loss�w�<�L       �	�  Yc�A�5*

losst�<y�)       �	N�  Yc�A�5*

loss���<��.�       �	'f! Yc�A�5*

lossZ�=�]Z5       �	q" Yc�A�5*

loss��~<���s       �	��" Yc�A�5*

loss+]�=Vٍ�       �	�D# Yc�A�5*

loss�K�<�J�!       �	��# Yc�A�5*

loss{��<x��       �	܂$ Yc�A�5*

loss8��<�h�       �	3% Yc�A�5*

lossa7	=�K�       �	)�% Yc�A�5*

loss��;�k�?       �	�u& Yc�A�5*

loss&f	=o��       �	o' Yc�A�5*

loss4Ug;��W       �	h�' Yc�A�5*

loss�|�<|(n�       �	/M( Yc�A�5*

loss��=�T(~       �	��( Yc�A�5*

lossZ�1;ë�>       �	j�) Yc�A�5*

loss�;U��)       �	o+* Yc�A�5*

lossJ&<�V'�       �	��* Yc�A�6*

loss���<�)9�       �	�b+ Yc�A�6*

loss��<�+y&       �	, Yc�A�6*

loss�3�<�ZO�       �	�, Yc�A�6*

loss�*�=�r��       �	- Yc�A�6*

loss(� ;��<?       �	�. Yc�A�6*

loss.�;���       �	ϻ. Yc�A�6*

lossi=	0��       �	�X/ Yc�A�6*

loss���<h��       �	7�/ Yc�A�6*

loss��<�� �       �	0�0 Yc�A�6*

lossF�c=��x�       �	�@1 Yc�A�6*

loss��|=n���       �	`�1 Yc�A�6*

lossrls<�X
�       �	~�2 Yc�A�6*

loss�nd<����       �	�+3 Yc�A�6*

lossL�<����       �	��3 Yc�A�6*

loss C�<��n0       �	'h4 Yc�A�6*

loss���<*�D       �	p5 Yc�A�6*

loss�d-=���"       �	�5 Yc�A�6*

loss�2�<���       �	�d6 Yc�A�6*

lossҸn<Ra�\       �	�7 Yc�A�6*

lossJG�;x�       �	�7 Yc�A�6*

loss_�q<2�D�       �	P:8 Yc�A�6*

loss��<�;��       �	��8 Yc�A�6*

loss���<����       �	�9 Yc�A�6*

lossM�g<77�       �	&�: Yc�A�6*

lossJ�=�p�       �	rj; Yc�A�6*

lossi�<JJ�       �	xb< Yc�A�6*

loss��<�wG�       �	�$= Yc�A�6*

losst;mt�       �	�= Yc�A�6*

losse!1;���       �	�g> Yc�A�6*

loss�-m;��W       �	�> Yc�A�6*

lossh�9<ϫ%t       �	¡? Yc�A�6*

loss?t�;�^d        �	F@ Yc�A�6*

loss1�=��Ϲ       �	d�@ Yc�A�6*

loss��<�;(.       �	ӄA Yc�A�6*

loss6ތ<�%P�       �	"B Yc�A�6*

lossm��<���       �	��B Yc�A�6*

lossQ�m<TQ_       �	�SC Yc�A�6*

loss�<q�C       �	��C Yc�A�6*

loss�b}<kAq       �	9�D Yc�A�6*

loss��5<
�       �	�6E Yc�A�6*

loss3��<2�k�       �	F�E Yc�A�6*

loss�[;����       �	�dF Yc�A�6*

lossv��;aC�       �	�0G Yc�A�6*

loss�	�:N��Y       �	��G Yc�A�6*

lossVDf;�P^�       �	x_H Yc�A�6*

lossa-<��d       �	��H Yc�A�6*

losst��;~�M       �	��I Yc�A�6*

loss�`==�l�       �	�rJ Yc�A�6*

loss��<\�3�       �	�K Yc�A�6*

loss쀛<b��       �	-�K Yc�A�6*

loss�%N<>e=�       �	\VL Yc�A�6*

loss�Y�<ך       �	��L Yc�A�6*

loss�t�=s��y       �	S�M Yc�A�6*

loss���<�Y�       �	U4N Yc�A�6*

loss�[+=��R       �	��N Yc�A�6*

loss	Ւ;a̓n       �	�gO Yc�A�6*

loss�3<��I       �	r�O Yc�A�6*

lossh�<�`��       �	�P Yc�A�6*

loss��<C��W       �	�tQ Yc�A�6*

lossw$	=X<       �	�R Yc�A�6*

loss/�l<=�M       �	'�R Yc�A�6*

loss4	<Vȷ�       �	K:S Yc�A�6*

loss��;&
�       �	��S Yc�A�6*

loss_RW<RTrC       �	nmT Yc�A�6*

lossZ�:=���       �	�U Yc�A�6*

lossr=��Rx       �	��U Yc�A�6*

loss��<P���       �	lCV Yc�A�6*

loss�]-<}��       �	n�V Yc�A�6*

lossg�=���       �	�X Yc�A�6*

loss�z�<2�!       �	]�X Yc�A�6*

lossRo7<ٜ��       �	�MY Yc�A�6*

loss�@<���       �	��Y Yc�A�6*

loss*Z<!�:2       �	t|Z Yc�A�6*

loss�<j�.�       �	� [ Yc�A�6*

loss��=��g�       �	��[ Yc�A�6*

loss7�_=U�u       �	,�\ Yc�A�6*

loss�Ɣ;t�        �	Tq] Yc�A�6*

loss�s_;-�u       �	^ Yc�A�6*

loss��<6��       �	��^ Yc�A�6*

loss�V:<7�y�       �	{O_ Yc�A�6*

loss���;���       �	
�_ Yc�A�6*

lossQ�<V�       �	ȗ` Yc�A�6*

loss�׼;mY�]       �	U1a Yc�A�6*

loss�ԧ<�~�}       �	J�a Yc�A�6*

loss�~:旼�       �	�jb Yc�A�6*

lossz;��X>       �	�c Yc�A�6*

lossd©;��:       �	i�c Yc�A�6*

loss4>�;V�@       �	�Kd Yc�A�6*

loss�'�;\�>       �	 �d Yc�A�6*

lossq+<�dm       �	P�e Yc�A�6*

loss��-=4��S       �	�(f Yc�A�6*

loss�"<N�)�       �	Q�f Yc�A�6*

loss���<��)�       �	-\g Yc�A�6*

loss�d1=QZ�P       �	��g Yc�A�6*

loss=�<P�3�       �	ڐh Yc�A�6*

lossx#=���       �	�8i Yc�A�6*

loss��;��aV       �	��i Yc�A�6*

loss��<:u{�       �	Wxj Yc�A�6*

loss]�j<�"�$       �	�k Yc�A�6*

loss��=Y\�       �	%�k Yc�A�6*

loss�_=p���       �	nLl Yc�A�6*

loss;&<�u�       �	��l Yc�A�6*

lossA��<6$�o       �	Rm Yc�A�6*

lossA4!=�]�l       �	Yn Yc�A�6*

loss��Q:6S��       �	E�n Yc�A�6*

lossJgy;g��^       �	�Po Yc�A�6*

loss."�<fM       �	��o Yc�A�6*

loss�Q=^���       �	(p Yc�A�6*

loss
׬<��[       �	jq Yc�A�6*

loss�N�<M��       �	(�q Yc�A�6*

losso��:��4�       �		Qr Yc�A�6*

loss1�=(4�       �	�r Yc�A�6*

loss�|�<*�Y       �	��s Yc�A�6*

loss%��<�r�       �	l&t Yc�A�6*

lossq2%=�rx�       �	��t Yc�A�6*

loss�>=
��L       �	�iu Yc�A�6*

loss�'X<fS�       �	 
v Yc�A�6*

loss�<�Ƹ       �	��v Yc�A�6*

loss��<Ҽ��       �	�>w Yc�A�6*

loss�hV<�|       �	��w Yc�A�6*

loss9P;��0       �	�sx Yc�A�6*

lossS��=�)�6       �	P�y Yc�A�6*

lossNh<4��=       �	,Ez Yc�A�6*

loss�r�<+���       �	��z Yc�A�6*

loss� �<wZ%       �	A�{ Yc�A�6*

loss�s"<���       �	W!| Yc�A�6*

loss ,�<\��$       �	��| Yc�A�6*

loss�<���s       �	8i} Yc�A�7*

lossHt=��^       �	�'~ Yc�A�7*

lossX&=�/�       �	+�~ Yc�A�7*

loss��<�/7�       �	g Yc�A�7*

loss�� =��U�       �	F� Yc�A�7*

lossY�;c=��       �	��� Yc�A�7*

loss��<��       �	�?� Yc�A�7*

loss�=�t;}       �	�ׁ Yc�A�7*

lossƆ><��L�       �	3o� Yc�A�7*

lossߜ<��e       �	�
� Yc�A�7*

loss���;)��0       �	��� Yc�A�7*

loss6�?<T��\       �	"P� Yc�A�7*

loss��;��͆       �	=� Yc�A�7*

loss럟;�8�)       �	��� Yc�A�7*

lossz��;�ffR       �	0� Yc�A�7*

loss�X=���       �	�Q� Yc�A�7*

loss}�1<$��H       �	J%� Yc�A�7*

lossr�l<���       �	�� Yc�A�7*

loss�i|<���       �	⒉ Yc�A�7*

lossF?1=K࠶       �	�>� Yc�A�7*

loss1-<e�       �	�؊ Yc�A�7*

loss=���R       �	�o� Yc�A�7*

lossFr�;���{       �	p� Yc�A�7*

loss��;�Ǯ�       �	z�� Yc�A�7*

loss��!=A�@       �	0H� Yc�A�7*

lossF3r<�<>       �	� Yc�A�7*

loss;�q<��kk       �	��� Yc�A�7*

loss轝<��       �	�� Yc�A�7*

lossN�;���$       �	�� Yc�A�7*

lossfv<h\X�       �	LP� Yc�A�7*

loss�S;[ɒ.       �	�� Yc�A�7*

loss o=�	�I       �	~� Yc�A�7*

loss��;��"�       �	�� Yc�A�7*

loss�!�<���Z       �	�Œ Yc�A�7*

loss�6�<šv�       �	�v� Yc�A�7*

lossY�	;c�}       �	'� Yc�A�7*

loss��`;�/��       �	a�� Yc�A�7*

lossW�];VV�$       �	_D� Yc�A�7*

loss�^�;SQ�       �	�ە Yc�A�7*

lossP�;����       �	6w� Yc�A�7*

loss�x�<�9�^       �	�� Yc�A�7*

loss�X=��        �	��� Yc�A�7*

loss6�><[b��       �	c�� Yc�A�7*

loss��;��.i       �	9E� Yc�A�7*

loss(�G;�`ˍ       �	zߚ Yc�A�7*

loss���<Sp�       �	.�� Yc�A�7*

loss��8<��`r       �	M�� Yc�A�7*

loss�p;*��       �	HO� Yc�A�7*

loss	_E=���       �	�� Yc�A�7*

loss�-�;�Y       �	��� Yc�A�7*

loss�v�<�cE�       �	FA� Yc�A�7*

loss�-;QX�       �	��� Yc�A�7*

loss%��<V��N       �	�� Yc�A�7*

loss��;�9>�       �	O<� Yc�A�7*

loss6e<�]E�       �	� Yc�A�7*

lossz;!<y       �	 {� Yc�A�7*

loss��5<G���       �	9&� Yc�A�7*

loss[H&<LXk�       �	�� Yc�A�7*

loss��<3�.s       �	*W� Yc�A�7*

loss@u< ��       �	�� Yc�A�7*

lossT��<lVv{       �	� Yc�A�7*

loss�<혾�       �	�� Yc�A�7*

loss�;�f21       �	�� Yc�A�7*

loss%�{;�bT:       �		N� Yc�A�7*

loss*�<��S       �	B� Yc�A�7*

loss
�q<�C�V       �	8�� Yc�A�7*

lossaa�;L���       �	4,� Yc�A�7*

loss�7�;l�l�       �	ҩ Yc�A�7*

loss�XI;�.r�       �	k� Yc�A�7*

loss�<��Xv       �	-� Yc�A�7*

loss�[<�v��       �	��� Yc�A�7*

lossM�<S�       �	%@� Yc�A�7*

lossi�<�d��       �	� Yc�A�7*

loss���;RW;        �	)z� Yc�A�7*

lossM��;�x�       �	�� Yc�A�7*

loss�]<�       �	��� Yc�A�7*

loss���;���J       �	�I� Yc�A�7*

loss���<K�       �	�߯ Yc�A�7*

loss��<���       �	�� Yc�A�7*

loss��;қ~       �	a� Yc�A�7*

loss-��<��\^       �	��� Yc�A�7*

loss�Z:�       �	�Q� Yc�A�7*

loss���<�ܥ       �	O� Yc�A�7*

losstŁ:/6��       �	4�� Yc�A�7*

loss�A�<�df"       �	E-� Yc�A�7*

loss0Ձ<���L       �	7ƴ Yc�A�7*

loss�Y*=ӏ6�       �	Ec� Yc�A�7*

loss�</�9       �	�� Yc�A�7*

lossx~<&�l�       �	�� Yc�A�7*

loss�'=n5,�       �	�5� Yc�A�7*

loss��#<��7       �	�� Yc�A�7*

loss@�:)|ϩ       �		�� Yc�A�7*

loss�ä;�t�       �	�� Yc�A�7*

loss��G<'��       �	|(� Yc�A�7*

loss���8�P�       �	9� Yc�A�7*

loss���;�h�^       �	�ϼ Yc�A�7*

loss��;;��m�       �	�� Yc�A�7*

loss}Y�;�В�       �	d�� Yc�A�7*

loss0R<�
6       �	�i� Yc�A�7*

loss��9Y���       �	�� Yc�A�7*

loss���;��M�       �	q�� Yc�A�7*

loss
4T:e+ֽ       �	��� Yc�A�7*

lossT�.:5���       �	L� Yc�A�7*

lossc��9��մ       �	��� Yc�A�7*

loss�_�<�'�       �	�w� Yc�A�7*

loss��<��[       �	�� Yc�A�7*

loss���<��C�       �	�� Yc�A�7*

loss��:�I�7       �	5A� Yc�A�7*

loss7�d<�Iln       �	V� Yc�A�7*

loss�O�=���       �	��� Yc�A�7*

lossC�:hnK       �	�N� Yc�A�7*

loss���;d�8U       �	\�� Yc�A�7*

lossf�s<_mN       �	�� Yc�A�7*

lossHTn<��z8       �	�� Yc�A�7*

lossQ�.<�)       �	p�� Yc�A�7*

loss[;<��d       �	jO� Yc�A�7*

lossv�D<�v�       �	��� Yc�A�7*

loss�y;��6�       �	��� Yc�A�7*

loss2��;���       �	�j� Yc�A�7*

loss���<I��       �	{� Yc�A�7*

lossԫZ:0F��       �	G�� Yc�A�7*

loss�!�=Vk�       �	�J� Yc�A�7*

loss{�<;��       �	2�� Yc�A�7*

loss�N�;9F3D       �	�� Yc�A�7*

loss���<i���       �	�� Yc�A�7*

lossi�@=^^       �	�� Yc�A�7*

loss�<�d       �	�U� Yc�A�7*

lossԆ�<�u �       �	�� Yc�A�7*

loss���<��[�       �	��� Yc�A�8*

loss%�Z;PB       �	/1� Yc�A�8*

lossE�>;&��       �	��� Yc�A�8*

loss<�<^N�       �	x_� Yc�A�8*

loss#�<�evP       �	�� Yc�A�8*

loss
I!;��5H       �	̗� Yc�A�8*

loss��<at       �	/0� Yc�A�8*

loss���=kn       �	�� Yc�A�8*

loss���<����       �	�Y� Yc�A�8*

loss�*N;?|Ay       �	3�� Yc�A�8*

lossO�<�'��       �	͐� Yc�A�8*

loss1��<zP�       �	t&� Yc�A�8*

loss˅�<�       �	<�� Yc�A�8*

loss��o<��d       �	�Y� Yc�A�8*

loss�V<Le�)       �	��� Yc�A�8*

lossQ;�m��       �	ܞ� Yc�A�8*

loss,*=� =       �	z5� Yc�A�8*

lossP�
<c��       �		�� Yc�A�8*

loss�J�;�h�       �	�y� Yc�A�8*

loss���;��"9       �	�� Yc�A�8*

loss�W=��L�       �	��� Yc�A�8*

loss��;Vtɯ       �	�6� Yc�A�8*

lossΫ�<�j��       �	��� Yc�A�8*

lossα=���y       �	��� Yc�A�8*

loss��=;։��       �	�� Yc�A�8*

loss�=�W��       �	غ� Yc�A�8*

loss��;�9�       �	 W� Yc�A�8*

lossZ�1<��K       �	��� Yc�A�8*

lossWM�<AV�       �	�� Yc�A�8*

loss�p='�2�       �	/0� Yc�A�8*

loss(�T<��L       �	��� Yc�A�8*

lossQ��:���S       �	�b� Yc�A�8*

losso��;��        �	�� Yc�A�8*

lossd�#=���1