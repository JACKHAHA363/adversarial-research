       �K"	   eYc�Abrain.Event:2�˻�$�     ��c�	��eYc�A"��
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
seed2�ß
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
seed2���
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
seed2��
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
seed2���*
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
T0*
Index0*
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
seed2���
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
valueB *
_output_shapes
: *
dtype0
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
 *   B*
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
 *  �?*
_output_shapes
: *
dtype0
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
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
_output_shapes
: *
T0
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
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
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
_output_shapes
:	�
*
dtype0
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
N"�jB�,     4�+�	�\
eYc�AJ��
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
seed2�ß*
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
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
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
seed2��*
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
_output_shapes
:	�
*
seed2��*
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
#sequential_1/dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
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
:���������@*
seed2���*
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
valueB:*
_output_shapes
:*
dtype0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
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
 *   B*
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
$softmax_cross_entropy_loss_1/Shape_2ShapePlaceholder*
T0*
out_type0*
_output_shapes
:
f
$softmax_cross_entropy_loss_1/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
valueB:*
dtype0*
_output_shapes
:
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
 *  �?*
_output_shapes
: *
dtype0
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
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
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
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
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
valueB@*    *&
_output_shapes
:@*
dtype0
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
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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

Adam/beta1Const*
_output_shapes
: *
dtype0*
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
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0*
use_locking( 
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
$softmax_cross_entropy_loss_1/value:0�h%       ��-	c_=eYc�A*

loss�V@�X�R       ��-	 >eYc�A*

loss�@�c�U       ��-	M�>eYc�A*

lossj�@�+�       ��-	�]?eYc�A*

loss@�i�       ��-	.U@eYc�A*

loss�w@�h�I       ��-	� AeYc�A*

lossL~@��       ��-	#BeYc�A*

loss�� @��       ��-	CeYc�A*

lossH@i2��       ��-	ϤCeYc�A	*

lossnx�?^ϟ�       ��-	�MDeYc�A
*

loss��??���       ��-	��DeYc�A*

loss���?Ј�"       ��-	9	FeYc�A*

loss�N�?���       ��-	��FeYc�A*

loss/y�?��       ��-	$%HeYc�A*

loss�d�?��S.       ��-	R�IeYc�A*

loss�ǒ?� %�       ��-	:@JeYc�A*

loss�}?�'})       ��-	��JeYc�A*

loss+�?�a       ��-	 {KeYc�A*

lossA }?:�4       ��-	�LeYc�A*

loss_��?6��       ��-	e�LeYc�A*

lossN?�6��       ��-	>MeYc�A*

loss�v?k4-6       ��-	��MeYc�A*

lossf�a?O�_       ��-	qNeYc�A*

loss.r�?�
��       ��-	�OeYc�A*

lossW�?ǌ�       ��-	��OeYc�A*

loss߶U?���       ��-	�oPeYc�A*

loss�3?�
�       ��-	QeYc�A*

loss{@x?|��       ��-	I�QeYc�A*

loss�ik?�S�       ��-	xDReYc�A*

loss�dP?�1�       ��-	��ReYc�A*

lossU��?G���       ��-	$�SeYc�A*

loss}�+?�;QB       ��-	�eTeYc�A *

loss�&\?�N�#       ��-	UeYc�A!*

loss���?��       ��-	��UeYc�A"*

loss�4?�       ��-	YNVeYc�A#*

loss�lK?i�dm       ��-	��VeYc�A$*

loss��?�⊐       ��-	?�WeYc�A%*

losso��?G�       ��-	(EXeYc�A&*

loss��R?Bo�       ��-	��XeYc�A'*

loss��`?�>�       ��-	�YeYc�A(*

loss�� ?Q4E�       ��-	�$ZeYc�A)*

lossL�?���       ��-	�ZeYc�A**

lossT�,?�z�       ��-	Ed[eYc�A+*

lossL�)?Fte       ��-	�.]eYc�A,*

loss�(?���       ��-	Z�]eYc�A-*

loss�K)?�	�        ��-	�^eYc�A.*

lossB�?86�       ��-	~_eYc�A/*

loss��?!9#!       ��-	�`eYc�A0*

loss?�h�|       ��-	*�`eYc�A1*

loss���>v��i       ��-	�QaeYc�A2*

lossn��>1� �       ��-	+�aeYc�A3*

lossq�?j�.D       ��-	�beYc�A4*

lossק�>ć�       ��-	�FceYc�A5*

loss=�Q?��8}       ��-	��ceYc�A6*

lossA��>��
�       ��-	��deYc�A7*

loss�I�>��6�       ��-	�geeYc�A8*

loss��>\4c|       ��-	1feYc�A9*

loss'�>�J�O       ��-	A�feYc�A:*

loss��<?}��       ��-	|dgeYc�A;*

loss��>=ʖ�       ��-	heYc�A<*

loss�?~`��       ��-	ӼheYc�A=*

loss��>N�2       ��-	�bieYc�A>*

loss�A�>TyfM       ��-	�jeYc�A?*

loss^�?���^       ��-	t�jeYc�A@*

loss�	?KpC�       ��-	-[keYc�AA*

loss�]�>���>       ��-	��keYc�AB*

loss_�7?��*�       ��-	մleYc�AC*

loss�s?�{S�       ��-	/mmeYc�AD*

loss��(?���       ��-	�"neYc�AE*

loss[�?��       ��-	(�neYc�AF*

loss´?gn&       ��-	soeYc�AG*

lossHJ?�c.       ��-	t'peYc�AH*

loss��>I)�       ��-	��peYc�AI*

loss�
	?y�jb       ��-	vqeYc�AJ*

lossڃ+?�w       ��-	HreYc�AK*

lossw�?�*       ��-	g�reYc�AL*

loss�~g?5
�`       ��-	�_seYc�AM*

loss��M?߷A!       ��-	5teYc�AN*

loss� �>�;�u       ��-	��teYc�AO*

loss���>�]$�       ��-	T�ueYc�AP*

loss	�?��       ��-	�2veYc�AQ*

lossa\?L>Q�       ��-	��veYc�AR*

lossym?�@T       ��-	xweYc�AS*

loss)��>6��       ��-	�xeYc�AT*

lossݞ?;��P       ��-	�xeYc�AU*

loss�V�>�ݥ�       ��-	�LyeYc�AV*

loss&ݘ>6ޮy       ��-	��yeYc�AW*

loss�??ǥk       ��-	#�zeYc�AX*

lossz?��@�       ��-	> {eYc�AY*

lossH��>��v       ��-	��{eYc�AZ*

loss��?� `�       ��-	��|eYc�A[*

lossC�?�1+       ��-	��}eYc�A\*

loss7	?�6�a       ��-	��~eYc�A]*

loss(?٩�       ��-	f.eYc�A^*

lossAL!?.�('       ��-	 �eYc�A_*

lossW*�>ٸ0v       ��-	vn�eYc�A`*

loss�R�>�P^2       ��-		�eYc�Aa*

lossD
?��       ��-	LłeYc�Ab*

loss�>?�ʊ7       ��-	���eYc�Ac*

loss]?/~]�       ��-	[C�eYc�Ad*

loss���>���       ��-	4�eYc�Ae*

loss�g�>���       ��-	q��eYc�Af*

loss�5�>gM�+       ��-	䇆eYc�Ag*

loss���>P{�U       ��-	7��eYc�Ah*

lossFX�>��|7       ��-	��eYc�Ai*

loss��>Ф~        ��-	���eYc�Aj*

loss	��>`��       ��-	2ʋeYc�Ak*

loss�y>B�<�       ��-	=�eYc�Al*

loss�W%?b���       ��-	�'�eYc�Am*

loss8@�>���       ��-	PߎeYc�An*

loss3?�1�       ��-	ӏeYc�Ao*

loss�F�>iT%A       ��-	?p�eYc�Ap*

lossC �>�Z��       ��-	���eYc�Aq*

lossD��>���       ��-	�A�eYc�Ar*

loss�>��R-       ��-	F\�eYc�As*

loss�Q�>R�X�       ��-	���eYc�At*

loss�Ý>�"�z       ��-	ȗ�eYc�Au*

loss�	?�)��       ��-	�8�eYc�Av*

loss�g�>�f��       ��-	��eYc�Aw*

loss��>���       ��-	"��eYc�Ax*

loss��>�;�       ��-	�0�eYc�Ay*

loss3A�>/	�j       ��-	�җeYc�Az*

loss�ڸ>���i       ��-	.u�eYc�A{*

loss�@�>�픆       ��-	a�eYc�A|*

loss�)>�|Q�       ��-	,��eYc�A}*

loss���>���[       ��-	b�eYc�A~*

loss�p?줣G       ��-	��eYc�A*

loss���>
��7       �	Q��eYc�A�*

loss�(�>����       �	@�eYc�A�*

loss���>+��       �	.�eYc�A�*

loss��>�g�       �	3��eYc�A�*

loss�]�>ߵ��       �	�D�eYc�A�*

loss1]5>&���       �	,��eYc�A�*

lossF�D>��6�       �	���eYc�A�*

lossx�>!�5{       �	�(�eYc�A�*

loss���>��ջ       �	CŠeYc�A�*

loss@ݬ>z3�"       �	=d�eYc�A�*

loss8Ԁ>Vz        �	7��eYc�A�*

lossrZ>�a��       �	ɑ�eYc�A�*

loss�!<>,�U       �	�+�eYc�A�*

loss%�2>�3@       �	�ģeYc�A�*

loss�,�>B���       �	x`�eYc�A�*

loss!ɍ>[��W       �	��eYc�A�*

loss@�>2��{       �	E��eYc�A�*

loss�Ė>��M�       �	�;�eYc�A�*

loss��U>�k:�       �	{٦eYc�A�*

loss�z�>BP       �	V}�eYc�A�*

lossh��=�'�       �	r�eYc�A�*

loss�=>�2ێ       �	ﮨeYc�A�*

loss�R�>�q��       �	�K�eYc�A�*

loss�h�>3��i       �	"�eYc�A�*

lossOy�>���       �	�|�eYc�A�*

lossoF�>@�u�       �	D�eYc�A�*

loss�.�>5�       �	 ��eYc�A�*

lossf3N>E��       �	KX�eYc�A�*

loss��->��޷       �	r��eYc�A�*

loss�W�>�R��       �	c��eYc�A�*

loss��>�X�1       �	�4�eYc�A�*

lossO��>�)B�       �	dϮeYc�A�*

loss��>���       �	�i�eYc�A�*

loss��>R���       �	�eYc�A�*

loss >��K
       �	ۤ�eYc�A�*

loss��>�7A)       �	_C�eYc�A�*

loss��'>��*�       �	?�eYc�A�*

lossaޟ>Q+��       �	k��eYc�A�*

loss�j>N�SP       �	�5�eYc�A�*

loss��>�N�o       �	�ֳeYc�A�*

loss��>!Hk;       �	8��eYc�A�*

loss
S9>���v       �	�%�eYc�A�*

lossQ#>2���       �	"��eYc�A�*

lossQ�l>(2|       �	
i�eYc�A�*

loss9>�"�       �	B	�eYc�A�*

loss'�>��3       �	޷eYc�A�*

loss�>�Zc�       �	���eYc�A�*

loss��_>	��       �	�G�eYc�A�*

loss��>T�&�       �	\�eYc�A�*

loss��>����       �	��eYc�A�*

loss�n�>��݌       �	� �eYc�A�*

loss�H�>r���       �	6ͻeYc�A�*

loss�>��|l       �	�g�eYc�A�*

lossس�=O(6g       �	�	�eYc�A�*

lossZ{�>�wp�       �	ᕿeYc�A�*

lossR�v>�n��       �	�A�eYc�A�*

lossjj>mE�,       �	���eYc�A�*

lossIN>����       �	���eYc�A�*

lossMy>:�        �	�g�eYc�A�*

losss�Y>�A�       �	4�eYc�A�*

loss֫�>�4zn       �	��eYc�A�*

loss,;�>��6�       �	�_�eYc�A�*

loss⊙>� �Z       �	_
�eYc�A�*

lossS4�>�P       �	���eYc�A�*

lossmtX>�&��       �	�d�eYc�A�*

loss��T>|: l       �	M�eYc�A�*

lossz�>�Pǯ       �	���eYc�A�*

loss�F>3|��       �	���eYc�A�*

loss\x>AE�8       �	�F�eYc�A�*

lossw��>���       �	���eYc�A�*

loss�6�>E���       �	U��eYc�A�*

losss�>.ƃ       �	�'�eYc�A�*

loss�5#>�!!�       �	���eYc�A�*

loss=�@>��       �	�m�eYc�A�*

loss��>���       �	��eYc�A�*

loss�3 >���I       �	��eYc�A�*

loss��V>O՞       �	0J�eYc�A�*

loss�u[>
��       �	���eYc�A�*

loss̆�=lM��       �	~�eYc�A�*

lossʆC>F��       �	fI�eYc�A�*

loss�>> j�l       �	m��eYc�A�*

lossI`3>�IL�       �	G��eYc�A�*

loss(�~>>��       �	�M�eYc�A�*

loss�C>�i�       �	lB�eYc�A�*

loss.�z>�o|       �	a��eYc�A�*

loss���=��       �	>z�eYc�A�*

loss?�>���%       �	2w�eYc�A�*

lossr��>�p�       �	g�eYc�A�*

lossE`�>� U�       �	���eYc�A�*

loss��x>�(��       �	eQ�eYc�A�*

lossH��>�v'       �	f�eYc�A�*

lossD*�>;,��       �	)��eYc�A�*

loss�3�>��:&       �	�d�eYc�A�*

loss=3	>�F2       �	WC�eYc�A�*

loss8��>E��Y       �	F"�eYc�A�*

loss��>*��<       �	]��eYc�A�*

lossJE�>���       �	��eYc�A�*

loss�>>Ќ]�       �	u �eYc�A�*

loss��>�n�       �	~��eYc�A�*

loss��> ��       �	Ov�eYc�A�*

loss��j>\��       �	�!�eYc�A�*

loss���>��*       �	b��eYc�A�*

lossJ�_>�@nh       �	[�eYc�A�*

loss�)(>3$]       �	��eYc�A�*

loss(�2>�?       �	��eYc�A�*

loss톞>��zW       �	+L�eYc�A�*

loss��>���L       �	���eYc�A�*

loss���>1�8�       �	D��eYc�A�*

loss�9>���b       �	tF�eYc�A�*

loss�lg>unT       �	G��eYc�A�*

loss�J�>u��       �	#��eYc�A�*

loss	��>��?�       �	/�eYc�A�*

lossN>c{�       �	J��eYc�A�*

loss!VG>[tj       �	�v�eYc�A�*

lossz�U>��!�       �	��eYc�A�*

loss�Ʊ>DB�	       �	���eYc�A�*

loss8<p>�]��       �	O[�eYc�A�*

lossh	�>��`       �	���eYc�A�*

loss��`>xg��       �	}��eYc�A�*

loss��5>>�a       �	�>�eYc�A�*

loss�3>�h�       �	��eYc�A�*

loss��F>���       �	���eYc�A�*

loss��g>�]�b       �	,�eYc�A�*

loss��>h���       �	h��eYc�A�*

loss��>��(7       �	�f�eYc�A�*

loss��>��b�       �	��eYc�A�*

loss=�=̂f`       �	ۤ�eYc�A�*

loss�">�'�       �	�H�eYc�A�*

lossX�Y>�ҟ�       �	��eYc�A�*

loss6��>զ�0       �	���eYc�A�*

loss�3�>H       �	��eYc�A�*

loss��*>K��i       �	(��eYc�A�*

loss���>��d       �	�X�eYc�A�*

loss[
n>��9       �	�eYc�A�*

loss{�&>��M+       �	[��eYc�A�*

loss}:�>�aܞ       �	�H�eYc�A�*

loss���>����       �	2��eYc�A�*

loss?*�>-�       �	���eYc�A�*

loss�(>����       �	�'�eYc�A�*

loss���>��
.       �	���eYc�A�*

loss��v>D��       �	b�eYc�A�*

loss=\�>/y�       �	:�eYc�A�*

lossq=>J�J�       �	��eYc�A�*

loss�Y>���7       �	K��eYc�A�*

lossC�W>��ȁ       �	��eYc�A�*

loss|i�>y�7�       �	K�eYc�A�*

loss3�6>����       �	���eYc�A�*

loss�>�*       �	1[�eYc�A�*

lossO��>6�J'       �	4��eYc�A�*

loss��@>���       �	ō�eYc�A�*

loss��&>�M.-       �	�' fYc�A�*

loss&�'>���I       �	� fYc�A�*

loss;b1>k�$       �	�[fYc�A�*

loss��=І�t       �	��fYc�A�*

loss��*>�2r       �	͑fYc�A�*

lossX�!>�>��       �	�7fYc�A�*

lossv3>I|E       �	t�fYc�A�*

loss��>�8�       �	.ufYc�A�*

lossf�h>Ҥn6       �	fYc�A�*

loss��>��sF       �	�fYc�A�*

lossW >paGt       �	W�fYc�A�*

loss_>��e2       �	�/fYc�A�*

loss�)C=��G       �	}�fYc�A�*

loss4�=;�|�       �	efYc�A�*

loss��>����       �	��fYc�A�*

loss�N>���       �		�	fYc�A�*

lossm�>1~}g       �	�A
fYc�A�*

lossP	>{*��       �	��
fYc�A�*

lossǚ> >��       �	"�fYc�A�*

lossM��>�h�z       �	S$fYc�A�*

loss��>~u�       �	s�fYc�A�*

loss��o>R�       �	"SfYc�A�*

loss-��>���       �	��fYc�A�*

loss4}G>h5O�       �	��fYc�A�*

loss�4i>t���       �	"fYc�A�*

losswYO>
@p�       �	��fYc�A�*

lossI�G>��^       �	rnfYc�A�*

lossv��=[��       �	<fYc�A�*

lossV�x>�{ګ       �	R�fYc�A�*

loss�ה=�R��       �	rmfYc�A�*

loss&r�>۽��       �	�fYc�A�*

loss�X�=�~p       �	
�fYc�A�*

loss�X&>y���       �	�7fYc�A�*

loss�k>����       �	��fYc�A�*

loss�Ƈ>�       �	��fYc�A�*

loss�:>��P�       �	}%fYc�A�*

loss��>\�t�       �	��fYc�A�*

loss��=�1n�       �	afYc�A�*

loss��=�K66       �	�fYc�A�*

lossz/'>�:�       �	��fYc�A�*

lossx�>M��C       �	�JfYc�A�*

lossqa>�G�       �	��fYc�A�*

loss�wQ>�	E       �	U�fYc�A�*

lossM=>�殦       �	�&fYc�A�*

loss,l�>���       �	��fYc�A�*

loss��>(C��       �	�mfYc�A�*

loss_8>?	�<       �	�fYc�A�*

loss���>�b�E       �	:�fYc�A�*

lossc��>�Ɩ       �	CTfYc�A�*

loss��O>LU��       �	��fYc�A�*

loss�1>T�7p       �	8�fYc�A�*

lossE>T�D�       �	�A fYc�A�*

loss@�E>!��       �	6� fYc�A�*

loss&>����       �	�!fYc�A�*

loss�9�=���       �	� "fYc�A�*

loss��=R��       �	��"fYc�A�*

loss�d >�A��       �	�b#fYc�A�*

loss΍>U7^�       �	}$fYc�A�*

lossL��=f��7       �	��$fYc�A�*

loss��z>`}       �	/%fYc�A�*

loss��n>	ާ       �	��%fYc�A�*

loss`�->�~�       �	"l&fYc�A�*

loss1Ѓ=���       �	-'fYc�A�*

loss��@>��d�       �	�'fYc�A�*

loss��=R_i�       �	�D(fYc�A�*

loss414>,ʪ       �	��(fYc�A�*

loss=�V>��       �	��)fYc�A�*

lossZ2?��ob       �	'1*fYc�A�*

loss�K> �       �	q�*fYc�A�*

loss���=c&�U       �	Tr+fYc�A�*

loss�)Y>pǰ       �	~,fYc�A�*

loss�;&>���m       �	�<.fYc�A�*

loss��>N�g(       �	2�.fYc�A�*

loss�'	>cqy�       �	��/fYc�A�*

lossv>�Q�W       �	�'0fYc�A�*

loss��	>A��       �	܂1fYc�A�*

lossc=>
�g�       �	�2fYc�A�*

lossJ:>��ֈ       �	��2fYc�A�*

loss@�=���       �	_]3fYc�A�*

loss��4>gE��       �	
�3fYc�A�*

lossͦ>�\��       �	ލ4fYc�A�*

loss�$�=���       �	�$5fYc�A�*

loss��=X�}�       �	b�5fYc�A�*

loss�=?> ���       �	iU6fYc�A�*

loss���>�=0�       �	K�6fYc�A�*

loss�=5>�r�6       �	�|7fYc�A�*

loss���>!�ٕ       �	$d8fYc�A�*

loss�h>�}       �	�8fYc�A�*

loss��*>����       �	 �9fYc�A�*

lossL�;>6%��       �	8.:fYc�A�*

loss} �=E|�N       �	K�:fYc�A�*

loss�@�>��x0       �	i;fYc�A�*

loss%��=IbG       �	
<fYc�A�*

loss�V�=��:�       �	�=fYc�A�*

loss�>�=�       �	�9>fYc�A�*

lossul�>����       �	�?fYc�A�*

loss�Gc>ͤ�u       �	��?fYc�A�*

loss�X8>Y�C       �	��@fYc�A�*

loss�>���u       �	LpAfYc�A�*

lossy��>�i\�       �	�RBfYc�A�*

loss�P!>9��i       �	��BfYc�A�*

loss�&>��       �	��CfYc�A�*

loss@>��}�       �	]DfYc�A�*

loss�${>t�{�       �	n�DfYc�A�*

lossڡn>���V       �	��EfYc�A�*

loss��0>��h%       �	�-FfYc�A�*

loss�mV>���       �	��FfYc�A�*

loss�=0�4�       �	�GfYc�A�*

lossi�=�׳�       �	� HfYc�A�*

loss�71>��1       �	@�HfYc�A�*

loss
�>A:��       �	�QIfYc�A�*

loss� >�x�       �	y�IfYc�A�*

loss@SA>I�x       �	��JfYc�A�*

loss@�Y>?o�       �	�#KfYc�A�*

lossҕ>�k�       �	)�KfYc�A�*

loss��i>l��       �	�kLfYc�A�*

loss��'>�<��       �	hMfYc�A�*

lossXO�>�z�       �	�MfYc�A�*

loss���>B��*       �	�ANfYc�A�*

loss�;>��\       �	�NfYc�A�*

loss��I=�!       �	%wOfYc�A�*

loss�
>)�ذ       �	PfYc�A�*

loss?.�>�O@K       �	n�PfYc�A�*

loss��>U�r       �	�?QfYc�A�*

loss��J>����       �	��QfYc�A�*

loss�o�=��       �	� SfYc�A�*

loss��>�?�       �	��SfYc�A�*

loss�Ί=pVO�       �	M1UfYc�A�*

loss��#>b�h       �	9�UfYc�A�*

loss,]�>���0       �	�VfYc�A�*

loss��@>����       �	�[WfYc�A�*

loss��F>C�f�       �	��WfYc�A�*

loss#_�>��Q>       �	��XfYc�A�*

loss�`#>g'R�       �	kYfYc�A�*

lossç�>�Z�1       �	�ZfYc�A�*

loss��>F�x       �	40[fYc�A�*

loss��>���r       �	�[fYc�A�*

loss�Q>�;ZD       �	x�\fYc�A�*

loss�	P>Y��       �	;o]fYc�A�*

lossV�=ќTu       �	�	^fYc�A�*

loss��=}��o       �	��^fYc�A�*

lossCp>*��       �	DN_fYc�A�*

loss�>Eu       �	��_fYc�A�*

loss�=S��j       �	�`fYc�A�*

lossvu(>>U5�       �	�)afYc�A�*

loss�p$>����       �	Y�afYc�A�*

lossl�>O��       �	�XbfYc�A�*

lossv�B>"D       �	�0cfYc�A�*

loss�>h��8       �	�cfYc�A�*

loss�r>��J       �	fdfYc�A�*

lossa:P>�=�       �	T�dfYc�A�*

loss�&>�J�       �	!�efYc�A�*

loss��!>���       �	a8ffYc�A�*

loss�tg>T�]       �	m�ffYc�A�*

loss�>�:I       �	[|gfYc�A�*

loss���=��       �	ahfYc�A�*

lossF�=>�J�.       �	�hfYc�A�*

loss&=">��̳       �	�WifYc�A�*

lossFC2=�>u       �	8�ifYc�A�*

losse5%>Y�J�       �	m�jfYc�A�*

loss�b>i���       �	9*kfYc�A�*

loss��=����       �	��kfYc�A�*

loss�}>\!X       �	�alfYc�A�*

lossAX�>;�<       �	K mfYc�A�*

loss�-a>��G�       �	��mfYc�A�*

loss$��=!�       �	�2nfYc�A�*

loss�K">��e       �	��nfYc�A�*

loss-CF=谘�       �	wjofYc�A�*

loss,2X>��M       �	ppfYc�A�*

loss���>�T�-       �	�pfYc�A�*

loss?�J>(x��       �	�9qfYc�A�*

lossf��=��WC       �	(�qfYc�A�*

loss㋸>�n�       �	 orfYc�A�*

lossex >����       �	usfYc�A�*

loss؞�=t�       �	�sfYc�A�*

loss냊=.���       �	�8tfYc�A�*

loss�5�>.���       �	��tfYc�A�*

losse��>)r�p       �	�iufYc�A�*

loss}PY>ǧF       �	%vfYc�A�*

loss �>%0        �	�xfYc�A�*

loss��Y>�ԡ       �	a�xfYc�A�*

loss��">#7p{       �	؁yfYc�A�*

loss�#>o�m       �	[#zfYc�A�*

loss� >7
�r       �	��zfYc�A�*

loss�`=����       �	�`{fYc�A�*

lossW�A>>9�       �	P�{fYc�A�*

loss��v>���d       �	o�|fYc�A�*

loss��=j�?Y       �	�:}fYc�A�*

loss�?�=�=�_       �	��}fYc�A�*

loss/�	>Q�cH       �	��~fYc�A�*

loss��=h�
5       �	'�fYc�A�*

loss�J5>�φ�       �	���fYc�A�*

lossIIN=^�r       �	�݁fYc�A�*

loss134>C��       �	��fYc�A�*

loss�3>�B0       �	I�fYc�A�*

lossw��>Zw/�       �	��fYc�A�*

loss�g>����       �	u!�fYc�A�*

loss�'�=�h~�       �	UM�fYc�A�*

lossSW�=V���       �	V�fYc�A�*

loss6�I=˼]�       �	FӇfYc�A�*

loss2��=��       �	���fYc�A�*

loss��>y�T�       �	e��fYc�A�*

loss�Q%>7Ծ       �	�͊fYc�A�*

loss���>ؘ۠       �	`ɋfYc�A�*

loss�5�=v+:{       �	q��fYc�A�*

loss$j>���*       �	\w�fYc�A�*

loss��>�+��       �	T��fYc�A�*

loss�܇=8
       �	�W�fYc�A�*

loss��={��       �	�fYc�A�*

losst->����       �	n��fYc�A�*

loss��*>:��       �	�&�fYc�A�*

loss'�=�.       �	I��fYc�A�*

lossC�_>�Z       �	>\�fYc�A�*

loss��>�X       �	���fYc�A�*

loss�Af>}�       �	\��fYc�A�*

loss%�X=ٷnR       �	H7�fYc�A�*

lossl��=.X�*       �	�ҔfYc�A�*

lossҰ�=���\       �	�f�fYc�A�*

loss!��=Y��d       �	��fYc�A�*

loss�:>?}�       �	F��fYc�A�*

loss�*�=if�D       �	�.�fYc�A�*

loss�=rdD       �	�fYc�A�*

lossڕ>�8~       �	�V�fYc�A�*

loss�.�>���w       �	��fYc�A�*

lossFeU=�        �	���fYc�A�*

loss�0�=o��       �	� �fYc�A�*

loss�� >c�M�       �	vƚfYc�A�*

loss
�T>,^�
       �	��fYc�A�*

lossH��=zA       �	⒜fYc�A�*

lossj->�F�       �	�0�fYc�A�*

loss3M�=�3q_       �	�ߝfYc�A�*

loss��V>@���       �	v��fYc�A�*

loss���=���       �	�.�fYc�A�*

loss���=:��       �	�̟fYc�A�*

loss�>6�_       �	Ii�fYc�A�*

loss��0=t�       �	�	�fYc�A�*

loss�b>2#c       �		��fYc�A�*

loss�1�=�E�1       �	�@�fYc�A�*

loss�=>9Ґ:       �	�ڢfYc�A�*

loss��=4�Z�       �	���fYc�A�*

loss�<�=�:�_       �	Z,�fYc�A�*

loss�%�=!S�       �	�ǤfYc�A�*

loss��=d�6�       �	nl�fYc�A�*

loss+�<��5�       �	��fYc�A�*

lossjG�=,/H�       �	���fYc�A�*

loss6�=���       �	 :�fYc�A�*

loss2r\=ّ�6       �	�ۧfYc�A�*

loss�d�=*��       �	�w�fYc�A�*

loss�U�=c<f       �	��fYc�A�*

loss�9�=���i       �	q��fYc�A�*

loss��J>���       �	2Y�fYc�A�*

loss�N7>�3��       �	,�fYc�A�*

loss�0=��V       �	���fYc�A�*

loss�~8=$5"�       �	�2�fYc�A�*

loss��=�h�       �	�լfYc�A�*

lossZ�=��<�       �	�v�fYc�A�*

loss��q=��o       �	��fYc�A�*

loss�r�=���       �	���fYc�A�*

loss#�<R��K       �	�B�fYc�A�*

loss#7k=�C;       �	�گfYc�A�*

loss�O=�G\�       �	�r�fYc�A�*

losstt=��       �	��fYc�A�*

loss�Q>mY�\       �	B��fYc�A�*

loss���<�<��       �	fO�fYc�A�*

loss!	�<q�+�       �	���fYc�A�*

loss�N�<�z[�       �	��fYc�A�*

loss
x�=��s       �	
0�fYc�A�*

loss�>Y�Ht       �	ϴfYc�A�*

loss'�=���       �	�k�fYc�A�*

loss1}O<�0�K       �	}�fYc�A�*

loss?@�=/��       �	鞶fYc�A�*

lossȟ�>B�0�       �	�8�fYc�A�*

lossI�I=�N�       �	OηfYc�A�*

lossvR�=�Q��       �	�e�fYc�A�*

loss#�">p�B       �	��fYc�A�*

loss�oz>I�M�       �	�fYc�A�*

loss�)�=��2P       �	�$�fYc�A�*

loss
5==�K�       �	û�fYc�A�*

lossJ� >�@�;       �	�R�fYc�A�*

loss�-#>0�	�       �	t�fYc�A�*

loss�?>$8�       �	�|�fYc�A�*

loss �>q�       �	��fYc�A�*

loss��F>|�	z       �	��fYc�A�*

loss$>E9��       �	؛�fYc�A�*

loss��f>���       �	r5�fYc�A�*

loss��>�@e�       �	�ʿfYc�A�*

loss�Tw>��}=       �	$a�fYc�A�*

loss1�j>�x       �	���fYc�A�*

loss{t=��.       �	��fYc�A�*

lossC9�=�j�=       �	�.�fYc�A�*

loss^�>��j�       �	���fYc�A�*

lossҒ�=x��       �	GZ�fYc�A�*

loss��=����       �	��fYc�A�*

loss��6>�6H       �	n��fYc�A�*

loss1� >��x�       �	�N�fYc�A�*

loss�1A= ��~       �	���fYc�A�*

losst0�=���       �	��fYc�A�*

loss�[=�jYW       �	 �fYc�A�*

loss�J>�       �	g��fYc�A�*

loss���=�(*�       �	O�fYc�A�*

loss[�=�F*       �	���fYc�A�*

loss�*=>�fk"       �	��fYc�A�*

loss�Z=�P�       �	;�fYc�A�*

loss��=��3       �	���fYc�A�*

loss=(F>e���       �	M�fYc�A�*

lossܢ�<��^�       �	���fYc�A�*

loss�\�=j���       �	�}�fYc�A�*

loss���=i�nX       �	�fYc�A�*

lossڭ�=��Z�       �	U��fYc�A�*

lossm��=��!!       �	9�fYc�A�*

lossO�>%Y*       �	���fYc�A�*

loss�6�=�w�:       �	Ra�fYc�A�*

loss�B=.1]       �	���fYc�A�*

loss�$r=켊       �	���fYc�A�*

loss�,�=�O^�       �	xz�fYc�A�*

loss��U>��       �	�"�fYc�A�*

loss�=����       �	��fYc�A�*

loss{�I>��       �	�e�fYc�A�*

loss7ɠ=g�z       �	t�fYc�A�*

loss#�=�P$       �	���fYc�A�*

loss���=���       �	JD�fYc�A�*

loss���=t��       �	x��fYc�A�*

lossjs�=���       �	���fYc�A�*

loss��p>K`�       �	*��fYc�A�*

loss��=��F       �	˂�fYc�A�*

loss�C#>���-       �	!�fYc�A�*

loss�!,>'b9       �	J��fYc�A�*

loss��\>W�~�       �	QN�fYc�A�*

lossS>��f       �	���fYc�A�*

loss�F�=Cy�3       �	�y�fYc�A�*

loss<E�=�:2       �	/�fYc�A�*

lossP>ވZF       �	���fYc�A�*

lossm��>@���       �	�n�fYc�A�*

loss�-�=���       �	�	�fYc�A�*

loss
ؼ=g��       �	�B�fYc�A�*

lossi�$>q��K       �	���fYc�A�*

loss*�>�$�M       �	�~�fYc�A�*

loss{��=�:�       �	�fYc�A�*

loss
�[>q�p;       �	���fYc�A�*

loss�ȟ=�0�       �	R�fYc�A�*

lossH�>puܗ       �	���fYc�A�*

loss��=��       �	�|�fYc�A�*

loss�W�>At|       �	I+�fYc�A�*

loss���=,��n       �	���fYc�A�*

lossƻ	>0��       �	�d�fYc�A�*

loss��>�)�       �	%�fYc�A�*

loss��:>��L�       �	;8�fYc�A�*

losso	>q5�       �	���fYc�A�*

loss\7�=��r       �	�v gYc�A�*

loss�ݻ=�0��       �	PgYc�A�*

loss]ش=!��       �	e�gYc�A�*

lossɷ">:���       �	σgYc�A�*

loss37>�En       �	�&gYc�A�*

loss=��=�P��       �	��gYc�A�*

lossW��=8i�       �	��gYc�A�*

lossF >^�\�       �	U�gYc�A�*

loss��>��>       �	gYc�A�*

loss���=���       �	��gYc�A�*

loss��]>k=�       �	��gYc�A�*

loss�ۉ=q1�T       �	�#gYc�A�*

loss��<>M��+       �	��gYc�A�*

loss�~�>l��^       �	�\	gYc�A�*

loss�Yc>i�u       �	s�
gYc�A�*

loss�];>�N��       �	gYc�A�*

loss�|�=��'�       �	��gYc�A�*

loss8�= �E       �	agYc�A�*

loss$P>�!�       �	�RgYc�A�*

loss��^>9ˢ�       �	C�gYc�A�*

lossEY>/$�       �	ŌgYc�A�*

lossB;>�Տ[       �	�#gYc�A�*

loss��5>�<U�       �	��gYc�A�*

loss��>��vr       �	y�gYc�A�*

lossl|�=%\�d       �	�JgYc�A�*

lossqj=[��       �	��gYc�A�*

loss��+>a+�       �	|gYc�A�*

loss��=��c       �	�'gYc�A�*

loss�?�"u&       �	GgYc�A�*

loss�3@>+�       �	�gYc�A�*

loss`^=l�/       �	�bgYc�A�*

loss��=�l�!       �	egYc�A�*

lossW�E=��j�       �	�gYc�A�*

loss.&>��Z       �	IgYc�A�*

lossM,�=3I��       �	��gYc�A�*

loss=	�=����       �	6sgYc�A�*

lossc��=�^,9       �	E*gYc�A�*

lossⵓ=���5       �	��gYc�A�*

lossRB�=��       �	'NgYc�A�*

lossb;>��~       �	C�gYc�A�*

loss���=����       �	��gYc�A�*

lossXG>����       �	^�gYc�A�*

loss��$>%,�6       �	4ggYc�A�*

lossz0�>R4��       �	�gYc�A�*

loss�j�=k�h�       �	�gYc�A�*

loss��>$Ο�       �	�>gYc�A�*

lossx�=m���       �	b�gYc�A�*

loss@��=\���       �	�t gYc�A�*

loss`_+>_�:1       �	�!gYc�A�*

lossz>��8�       �	|G"gYc�A�*

lossI��=��AG       �	�"gYc�A�*

loss�-�=Y�D�       �	��#gYc�A�*

loss�4�>%��'       �	�8$gYc�A�*

loss�˴=>�8�       �	��$gYc�A�*

losse*�=X��~       �	+�%gYc�A�*

loss�_8>/�|�       �	�#&gYc�A�*

loss�=M>Tb��       �	��&gYc�A�*

loss&@/>J��*       �	d'gYc�A�*

loss��=�z	�       �	��'gYc�A�*

loss�\�=��54       �	|�(gYc�A�*

loss�	�=Mn�       �	�1)gYc�A�*

lossV�=	j��       �	5�)gYc�A�*

lossZ_>_�6�       �	g*gYc�A�*

lossԉ�=3�.&       �	d+gYc�A�*

loss�5�=�|��       �	"�+gYc�A�*

loss�\>���P       �	J@,gYc�A�*

lossI�=:J�y       �	��,gYc�A�*

loss���=�g�,       �	`r-gYc�A�*

losss�=R��       �	�.gYc�A�*

lossEX�>ڜ_�       �	��.gYc�A�*

loss��=h_�f       �	?T/gYc�A�*

loss�֦=3Ҁ�       �	[�/gYc�A�*

loss�g�=�r�.       �	��0gYc�A�*

loss�y�>���B       �	T1gYc�A�*

loss?��=����       �	.�1gYc�A�*

loss@Wi=(��       �	�G2gYc�A�*

lossn4=��H       �	w�2gYc�A�*

loss*�>�Mh       �	V�3gYc�A�*

loss|�=��Y       �	�04gYc�A�*

loss�( >��g�       �	�4gYc�A�*

loss/�2>ɩ��       �	F]5gYc�A�*

loss��=��f       �	�5gYc�A�*

loss4=_<Q       �	H�6gYc�A�*

loss�>Uw��       �	E)7gYc�A�*

losssa:>�]�        �	+�7gYc�A�*

loss&M&>��,       �	�X8gYc�A�*

loss[>82��       �	5�8gYc�A�*

loss�X=��W?       �	��9gYc�A�*

loss� �=�f!I       �	�!:gYc�A�*

lossm`�=���       �	�:gYc�A�*

loss�Q�=Sh�       �	?S;gYc�A�*

loss�B�=Qb}.       �	��;gYc�A�*

lossX��=��t       �	Y�<gYc�A�*

loss�!>E���       �	%=gYc�A�*

lossAj�=A�2�       �	��=gYc�A�*

loss�/ >C���       �	�n>gYc�A�*

loss�BQ>7WlT       �	�!?gYc�A�*

loss!3W>z3�f       �	E�?gYc�A�*

losse!>,]$       �	��@gYc�A�*

loss�a�<�d@�       �	5&AgYc�A�*

loss��[>�i�       �	<�AgYc�A�*

loss�s2>'T`�       �	SXBgYc�A�*

loss�y=����       �	=�BgYc�A�*

loss�E>6㧞       �	��CgYc�A�*

loss=JL>־zM       �	��DgYc�A�*

loss��.>肘       �	3mEgYc�A�*

lossF/�=zk3       �	FgYc�A�*

loss6��=q$�       �	k�FgYc�A�*

loss�=�=��h       �	�:GgYc�A�*

loss��>Q{�Y       �	��GgYc�A�*

loss�>�(��       �	�lHgYc�A�*

loss���=q���       �	6IgYc�A�*

loss���=��'�       �	�IgYc�A�*

loss��=��ys       �	�;JgYc�A�*

loss��8=���       �	|�JgYc�A�*

loss�7a=D�       �	YmKgYc�A�*

loss\��=hÖ�       �	uLgYc�A�*

loss� >��A        �	V�LgYc�A�*

loss'��=�@�       �	^0MgYc�A�*

loss��s>�)C!       �	��MgYc�A�*

loss_��=0��}       �	�gNgYc�A�*

loss7>�ޘ�       �	�	OgYc�A�*

lossJ�U=�d�       �	�OgYc�A�*

loss3�D=���       �	�QPgYc�A�*

loss��=���       �	��PgYc�A�*

loss�!0=B�4�       �	6�QgYc�A�*

lossj
>�|	�       �	1RgYc�A�*

lossSM�=��7       �	O�RgYc�A�*

loss��=x"��       �	_zSgYc�A�*

loss;+�=�3(�       �	vTgYc�A�*

lossL�=��2�       �	�TgYc�A�*

loss��=��-�       �	WUgYc�A�*

lossI��=�8�       �	 �UgYc�A�*

lossw!g>��       �	WgYc�A�*

loss���=	�#       �	 �WgYc�A�*

loss� >#Gs�       �	fKXgYc�A�*

loss���=���>       �	��XgYc�A�*

loss��=��x       �	,~YgYc�A�*

loss�J=�Y1       �	ZgYc�A�*

loss.H=N�,�       �	��ZgYc�A�*

loss�Ǭ=y��d       �	�E[gYc�A�*

loss`B�=.��       �	��[gYc�A�*

loss">��.a       �	�\gYc�A�*

lossn�G=eM�       �	�]gYc�A�*

loss/��=���       �	y�]gYc�A�*

loss*I�=����       �	(D^gYc�A�*

loss�1=>,&s       �	��^gYc�A�*

loss�>*���       �	�n_gYc�A�*

lossd�%>�L%H       �	`gYc�A�*

loss(��<yi(�       �	)�`gYc�A�*

loss>LYw       �	,-agYc�A�*

loss��4>�&�a       �	�agYc�A�*

loss�>I�       �	'�bgYc�A�*

loss��=@��       �	�GcgYc�A�*

lossX}%=(���       �	��cgYc�A�*

loss���<)��       �	xdgYc�A�*

loss��=�9z�       �	egYc�A�*

lossӬ�=�2ۏ       �	��egYc�A�*

loss�'=~<�       �	kEfgYc�A�*

loss�"�=�'�       �	��fgYc�A�*

loss�8�=DdD       �	g}ggYc�A�*

loss*��=�� �       �	�ehgYc�A�*

loss��=�D��       �	igYc�A�*

loss��[=�$j�       �	U�igYc�A�*

loss(��=�6�       �	�BjgYc�A�*

loss�� >w4]Q       �	��jgYc�A�*

loss��>`���       �	yykgYc�A�*

loss�d�=Q��       �	�lgYc�A�*

lossc�=(�       �	�lgYc�A�*

loss��$=���       �	 BmgYc�A�*

loss#�=��ya       �	G�mgYc�A�*

lossں>�C|       �	j�ngYc�A�*

loss�b*>�m.u       �	�NogYc�A�*

losse>%6�       �	,�ogYc�A�*

loss�q=��       �	ÂpgYc�A�*

loss(�r=(L�a       �	�)qgYc�A�*

loss<F=�K�       �	��qgYc�A�*

lossx�="~O�       �	��rgYc�A�*

lossX��=c�T       �	h"sgYc�A�*

loss�k�=�M       �	�sgYc�A�*

lossŸp=jhP`       �	�rtgYc�A�*

loss�>��S       �	[ugYc�A�*

loss���=�V�%       �	��ugYc�A�*

loss�^�=��B
       �	�LvgYc�A�*

loss_�m=`��       �	y;wgYc�A�*

lossnn>Uu�       �	ߦxgYc�A�*

loss�?�=���       �	1AygYc�A�*

loss���=�2�/       �	zgYc�A�*

lossEB�=�t�H       �	 �zgYc�A�*

loss4"4>�ccT       �	!V{gYc�A�*

loss��=h�q!       �	Q�}gYc�A�*

loss$�=	�       �	�gYc�A�*

loss��1>��p�       �	T�gYc�A�*

lossC:>��       �	 �gYc�A�*

loss?�>���       �	'��gYc�A�*

loss.h�<Wҍ       �	b��gYc�A�*

loss�3=F��(       �	i;�gYc�A�*

loss
bm>Up�       �	~�gYc�A�*

loss�]�>u��       �	B
�gYc�A�*

loss��=M��       �	���gYc�A�*

loss��=ˬ       �	
��gYc�A�*

lossR��=���       �	���gYc�A�*

loss���=�`4v       �	u=�gYc�A�*

lossC?O=���       �	��gYc�A�*

lossa��=|��       �	���gYc�A�*

lossR�_=�{�       �	�b�gYc�A�*

loss��W=U$�       �	 
�gYc�A�*

loss-��=!O       �	���gYc�A�*

loss��=�E�       �	�L�gYc�A�*

lossƳ>"f��       �	T�gYc�A�*

loss�5�=(       �	�{�gYc�A�*

lossm��=��A       �	^�gYc�A�*

loss8,>Q�q�       �	䡎gYc�A�*

lossM~E>��H       �	~5�gYc�A�*

loss�^=�O       �	5яgYc�A�*

loss�]~= Q��       �	�e�gYc�A�*

loss�> �u"       �	���gYc�A�*

loss_�=��%�       �	gYc�A�*

loss���=N�8�       �	���gYc�A�*

loss�>tfu	       �	V)�gYc�A�*

lossl�=�\)E       �	v��gYc�A�*

loss6�>�'/9       �	XƕgYc�A�*

loss�%�=����       �	�k�gYc�A�*

loss� >�i�       �	-w�gYc�A�*

loss��=?�ii       �	��gYc�A�*

loss�� >�N<       �	S��gYc�A�*

loss- >����       �	IJ�gYc�A�*

loss�8�=���       �	��gYc�A�*

loss�O)= ��       �	}�gYc�A�*

loss�=}C�       �	��gYc�A�*

loss=�đ       �	ɬ�gYc�A�*

loss X�=�=<       �	NE�gYc�A�*

lossHS1>p�!�       �	�ٜgYc�A�*

lossH�N=��&�       �	*��gYc�A�*

loss6��=w�`       �	��gYc�A�*

loss��=|��       �	;:�gYc�A�*

loss�Ş=E���       �	�ПgYc�A�*

loss�>��g+       �	O��gYc�A�*

loss->���       �	_E�gYc�A�*

loss� e>�Z/�       �	eޡgYc�A�*

loss}��=���Y       �	�w�gYc�A�*

loss8<>)�       �	��gYc�A�*

loss��=�}�       �	ӥ�gYc�A�*

loss�Hi>�_��       �	\;�gYc�A�*

lossI��=��U       �	T�gYc�A�*

loss�Ҏ=�S       �	��gYc�A�*

loss��=��O       �	H�gYc�A�*

loss�t�=�
�D       �	ߩ�gYc�A�*

loss��
>(�Mi       �	-?�gYc�A�*

loss�E->�       �	�ڧgYc�A�*

loss�;>���       �	h�gYc�A�*

lossV�=x��+       �	���gYc�A�*

loss���=���       �	��gYc�A�*

loss�=>�G�       �	[��gYc�A�*

lossqc>0x��       �	wN�gYc�A�*

losszC=e"��       �	��gYc�A�*

loss1��=���       �	��gYc�A�*

loss�0�=���       �	��gYc�A�*

loss��=0��       �	|��gYc�A�*

loss!e!>�g�V       �	�L�gYc�A�*

loss�rY=50��       �	즰gYc�A�*

loss��=�n�       �	}=�gYc�A�*

loss���=�-��       �	�ұgYc�A�*

loss�#=���       �	2v�gYc�A�*

loss�X=��U       �	��gYc�A�*

loss�e#=^�y       �	Է�gYc�A�*

loss�>�}0       �	�d�gYc�A�*

loss({=P}�e       �	�gYc�A�*

lossf�g=q��       �	ڨ�gYc�A�*

loss��=���       �	4��gYc�A�*

loss�M�=�e�       �	�"�gYc�A�*

lossP��=��S�       �	d̷gYc�A�*

loss�AB>���       �	"l�gYc�A�*

loss���=���       �	��gYc�A�*

loss��D>�-�       �	��gYc�A�*

loss--�=c�n{       �	㍺gYc�A�*

loss�i�=��{       �	�A�gYc�A�*

loss�=u��       �	'�gYc�A�*

loss���=ɨ�       �	��gYc�A�*

loss3��=S䗥       �	
��gYc�A�*

lossL�(>�&'$       �	�r�gYc�A�*

loss �K=��B�       �	j�gYc�A�*

loss�g">��<�       �	���gYc�A�*

loss��+=�|h�       �	u��gYc�A�*

lossC�>�ݞ1       �	��gYc�A�*

loss=��=���       �	)��gYc�A�*

loss��>�cm&       �	���gYc�A�*

loss/n�=+�M�       �	Tq�gYc�A�*

loss��=��[?       �	r5�gYc�A�*

loss�&�<��d       �	M��gYc�A�*

loss�`">�8Va       �	��gYc�A�*

lossBE�=�$�       �	v4�gYc�A�*

lossWb=��w�       �	���gYc�A�*

lossϿ>�sS[       �	�m�gYc�A�*

loss���=�۶       �	��gYc�A�*

loss�Ċ=@y��       �	��gYc�A�*

lossů�>�'��       �	�h�gYc�A�*

lossf��=(2�,       �	P�gYc�A�*

lossZ��=T�       �	��gYc�A�*

loss��>N�x5       �	f��gYc�A�*

loss\|_>`�       �	�(�gYc�A�*

lossM�=����       �	��gYc�A�*

loss�y1=	       �	�j�gYc�A�*

lossƍ#=&�V
       �	$�gYc�A�*

loss#�>n�       �	d��gYc�A�*

loss!��=���       �	�V�gYc�A�*

loss#DN=zV�       �	�\�gYc�A�*

loss*C@=��<�       �	c+�gYc�A�*

loss��;=M �=       �	���gYc�A�*

loss\>7>�z��       �	�t�gYc�A�*

loss( f=�/�m       �	��gYc�A�*

loss,+>J�wb       �	��gYc�A�*

loss;��=J���       �	t�gYc�A�*

loss���=���\       �	�gYc�A�*

loss��<)�DR       �	ӽ�gYc�A�*

loss�=X�b       �	�^�gYc�A�*

lossf�=�P1^       �	}	�gYc�A�*

loss��>e�[       �	]��gYc�A�*

lossZXP=}�u       �	WA�gYc�A�*

loss���>�u1       �	�=�gYc�A�*

loss�Z>��ٯ       �	���gYc�A�*

loss�.�=�L�       �	u�gYc�A�*

loss||M>w���       �	�gYc�A�*

loss��=ۼ �       �	P��gYc�A�*

lossO�7=�8��       �	.U�gYc�A�*

lossv�E= �k�       �	��gYc�A�*

loss��=�/'@       �	7��gYc�A�*

loss}`=7A��       �	�K�gYc�A�*

loss�o�=�1�^       �	A�gYc�A�*

lossB�>z�       �	.��gYc�A�*

loss�y�<���R       �	�S�gYc�A�*

loss���=w�G       �	�>�gYc�A�*

loss��=�b�       �	 ��gYc�A�*

loss��=�%
)       �	��gYc�A�*

loss��=��\       �	�1�gYc�A�*

loss�0^=�o��       �	���gYc�A�*

loss݌=N�1�       �	�u�gYc�A�*

loss�V�=���       �	�gYc�A�*

lossC)�=B�\w       �	F��gYc�A�*

loss��*>E�7       �	�X�gYc�A�*

loss=h�=��xQ       �	� �gYc�A�*

loss��=g�S�       �	A��gYc�A�*

lossE\=*2;�       �	\9�gYc�A�*

loss4�=��`:       �	���gYc�A�*

loss��s=8uv�       �	Sw�gYc�A�*

loss�|�=��_       �	��gYc�A�*

lossx�>��       �	(��gYc�A�*

loss���=��)       �	�M�gYc�A�*

loss�ů=�6M       �	=��gYc�A�*

loss���=�O�       �	��gYc�A�*

loss�C�= �~�       �	�3�gYc�A�*

lossN|(>5��       �	���gYc�A�*

lossJ�=�Vj        �	Ih�gYc�A�*

loss��=����       �	W�gYc�A�*

loss�>��t�       �	ӟ�gYc�A�*

loss���=I`��       �	�@�gYc�A�*

loss�|D>�;�l       �	��gYc�A�*

loss�Ƹ=�	?D       �	�}�gYc�A�*

loss�
�<��*       �	?�gYc�A�*

loss��I=���T       �	���gYc�A�*

lossI��=�ŖZ       �	j�gYc�A�*

loss�B&>�       �	�gYc�A�*

loss�~�=[�^       �	��gYc�A�*

loss�ٞ=U���       �	�y�gYc�A�*

loss���=�9t       �	�!�gYc�A�*

loss�`>�^C�       �	m��gYc�A�*

loss�7�=���)       �	�t�gYc�A�*

loss�M�=�H�       �	�gYc�A�*

loss�!>ǌ��       �	B��gYc�A�*

loss� �=�       �	kJ�gYc�A�*

loss�;>��>�       �	���gYc�A�*

loss�5�=�`��       �	���gYc�A�*

lossz��=�_��       �	�*�gYc�A�*

loss�f=���       �	���gYc�A�*

loss�G>��"�       �	�f�gYc�A�*

loss���=ɠn�       �	P� hYc�A�*

lossd�,>�n��       �	�hYc�A�*

loss�&�=v[�
       �	u�hYc�A�*

lossx[H=���n       �	�hYc�A�*

loss-�=�}<h       �	%XhYc�A�*

lossv>Q$6       �	�hYc�A�*

loss,>(���       �	��hYc�A�*

loss| >�uA       �	&�hYc�A�*

loss��=�m       �	�hYc�A�*

loss,>ΰ�       �	 �hYc�A�*

loss\��=�2N       �	�ehYc�A�*

loss�:F>5��       �	�	hYc�A�*

loss��>|�DT       �	t�	hYc�A�*

loss�a>?��       �	!\
hYc�A�*

lossӇ>�s       �	�hYc�A�*

loss�J>���z       �	��hYc�A�*

lossv�V=��<�       �	:ZhYc�A�*

lossdC#=�bY       �	�hYc�A�*

loss�>!���       �	ĳhYc�A�*

loss}��=7(e       �	�[hYc�A�*

lossR=� \       �	�hYc�A�*

loss��=5�@%       �	�hYc�A�*

lossf�=��_       �	�]hYc�A�*

loss �>�h�       �	�hYc�A�*

loss*(�=��~D       �	��hYc�A�*

lossw�/>�= �       �	�?hYc�A�*

loss���=1g�       �	P�hYc�A�*

lossv,>e���       �	��hYc�A�*

loss��N=BIwp       �	l"hYc�A�*

loss	t�= 4�h       �	D�hYc�A�*

lossZ�=��       �	3jhYc�A�*

lossų�=J��       �	)hYc�A�*

loss(�=\�+�       �	ҬhYc�A�*

loss���=/�w       �	KhYc�A�*

loss�U�=����       �	��hYc�A�*

loss�`7=Kg��       �	��hYc�A�*

lossa��=S���       �	�3hYc�A�*

loss�yw=<Fg       �	nRhYc�A�*

loss�=5�6�       �	A�hYc�A�*

loss>=~�       �	2�hYc�A�*

loss$��=]�;5       �	�VhYc�A�*

loss�?<>�]z       �	��hYc�A�*

loss
�=�/+       �	�hYc�A�*

loss��=�kl�       �	�?hYc�A�*

loss	��<�ڌ�       �	}�hYc�A�*

loss�!>�ց�       �	��hYc�A�*

loss>+U�       �	lA hYc�A�*

loss�T�=&g��       �	� hYc�A�*

loss�s�=��       �	�|!hYc�A�*

lossr>Q�]�       �	G"hYc�A�*

loss\��=�S�d       �	��"hYc�A�*

loss$=vۼ�       �	�h#hYc�A�*

loss��+=��s�       �	$hYc�A�*

loss�;I>��=�       �	�$hYc�A�*

lossS�f>0M��       �	�[%hYc�A�*

lossݠ�=0	��       �	��%hYc�A�*

loss�s�=�u�       �	��&hYc�A�*

loss6��=��       �	�@'hYc�A�*

loss���=6�W#       �	��'hYc�A�*

loss�R�=>���       �	܄(hYc�A�*

loss��=Sz*�       �	$)hYc�A�*

loss���<�T]       �	O�)hYc�A�*

lossJ֥=��ܟ       �	�c*hYc�A�*

loss��>j�$'       �	��*hYc�A�*

loss�"�=�0       �	�+hYc�A�*

loss�Gh=� b       �	�O,hYc�A�*

loss��=�<De       �	+�,hYc�A�*

loss�`�=�i��       �	��-hYc�A�*

loss!E=k4<8       �	J.hYc�A�*

loss7	�<y4^       �	Q�.hYc�A�*

loss�>Q��       �	�/hYc�A�*

lossh@�=K{�v       �	<0hYc�A�*

loss��>c�       �	��0hYc�A�*

lossx&>���'       �	Ș1hYc�A�*

loss�{r=Cq�       �	�>2hYc�A�*

loss�Ι<���       �	��2hYc�A�*

loss��%=o��       �	>�3hYc�A�*

lossQg=���       �	�?4hYc�A�*

loss��=�C       �	B�4hYc�A�*

loss�3>��       �	"�5hYc�A�*

loss�Z�>��(6       �	�N6hYc�A�*

lossE�="��       �	��6hYc�A�*

loss[
>㏸�       �	`�7hYc�A�*

lossLE>�/v�       �	�V8hYc�A�*

loss�7>R��Y       �	kJ9hYc�A�*

loss&N=�I�        �	 :hYc�A�*

lossl��=&���       �	�:hYc�A�*

loss X	>câo       �	]l;hYc�A�*

lossJ'�=��r�       �	<hYc�A�*

loss�(�=��կ       �	^�<hYc�A�*

lossv�6=���       �	��=hYc�A�*

loss[��=�_<l       �	�A>hYc�A�*

loss|I =����       �	��>hYc�A�*

loss�-�=s�ؐ       �	N�?hYc�A�*

lossJO =2�CS       �	X;@hYc�A�*

loss�=�<*M�       �	��@hYc�A�*

loss�*�=.��$       �	ɎAhYc�A�*

loss��,=���>       �	9BhYc�A�*

loss)�=����       �	��BhYc�A�*

loss4��=}��       �	a�ChYc�A�*

loss��=9�j�       �	�:DhYc�A�*

lossx�=v��       �	��DhYc�A�*

loss���=v�Z�       �	_�EhYc�A�*

loss-�Z=�[�       �	�5FhYc�A�*

loss�`�=��d       �	f�FhYc�A�*

lossh��=S��       �	��GhYc�A�*

lossf>�f�a       �	�AHhYc�A�*

lossR�=x9xt       �	4�HhYc�A�*

loss1��=h�,�       �	��IhYc�A�*

loss���=)�       �	{JhYc�A�*

lossI��<d1�       �	�&KhYc�A�*

loss=r�=""�(       �	R�KhYc�A�*

loss��0=)O       �	�sLhYc�A�*

lossj�>�mq�       �	�MhYc�A�*

loss�Y�=�Z��       �	��MhYc�A�*

lossl<>;bX       �	�hNhYc�A�*

loss���='hd       �	HOhYc�A�*

loss�c�=����       �	z�OhYc�A�*

loss�$�=cF��       �	�qPhYc�A�*

lossj+�=���       �	AQhYc�A�*

lossXM=CP�Z       �	�QhYc�A�*

loss,Æ=��%<       �	dYRhYc�A�*

lossHp�=���M       �	�ShYc�A�*

loss�O=m��       �	�(ThYc�A�*

loss�y�=is;�       �	��ThYc�A�*

lossF��=�_�$       �	~UhYc�A�*

loss��=.���       �	�-VhYc�A�*

loss�> 21       �	 �VhYc�A�*

loss�'>��&�       �	TsWhYc�A�*

loss��v=2B��       �	EXhYc�A�*

lossx-�<>D�       �	�XhYc�A�*

loss�Л=4�D       �	�KYhYc�A�*

lossMf=Ǝ       �	=_ZhYc�A�*

loss���<G�t�       �	]�ZhYc�A�*

loss�s�<��H       �	T�[hYc�A�*

loss�Ҟ<A�2       �	&\hYc�A�*

loss��<��z       �	��\hYc�A�*

loss.P&=�<        �	�V]hYc�A�*

loss�P�<�u��       �	�]hYc�A�*

loss�.�=!���       �	f�^hYc�A�*

loss��<z	�j       �	�_hYc�A�*

loss8Po;NKh       �	0�_hYc�A�*

loss��q<g+ݭ       �	�P`hYc�A�*

loss�8�=�,�       �	)�`hYc�A�*

loss3S�>g���       �	/�ahYc�A�*

loss�'B=_�       �	�!bhYc�A�*

lossx.�;U,-~       �	�bhYc�A�*

lossaQ%=��g       �	��chYc�A�*

loss���>߶��       �	�"dhYc�A�*

loss��<n%�       �	�dhYc�A�*

lossV&=�G�Y       �	PUehYc�A�*

loss�5�=,[1w       �	|�ehYc�A�	*

loss?��=�P�       �	ӢfhYc�A�	*

lossz+�=AA�C       �	�FghYc�A�	*

lossog<=�-�       �	m�ghYc�A�	*

loss�[�=$z       �	��hhYc�A�	*

lossj�=���       �	l"ihYc�A�	*

loss-�="�       �	��ihYc�A�	*

loss]>D&
a       �	�_jhYc�A�	*

lossߙ�=���)       �	4khYc�A�	*

lossWb�=q%S	       �	��khYc�A�	*

lossIm>�L�       �	�`lhYc�A�	*

lossr~�=:o��       �	�	mhYc�A�	*

lossm�>r9ߎ       �	�mhYc�A�	*

lossƉ,>�~_0       �	.�nhYc�A�	*

loss���=��       �	NAohYc�A�	*

lossM��=F��D       �	'�ohYc�A�	*

loss��=@^�R       �	Q�phYc�A�	*

loss���=�WP       �	,aqhYc�A�	*

loss��P=��0        �	orhYc�A�	*

loss<��=J��       �	��rhYc�A�	*

lossf#�=(%R        �	��shYc�A�	*

loss�T<!X-�       �	HnthYc�A�	*

loss�0=!���       �	\ uhYc�A�	*

loss_�8=���8       �	u�uhYc�A�	*

loss��L=�E       �	�vhYc�A�	*

loss��J=x>��       �	�.whYc�A�	*

loss�Y >X+r       �	�whYc�A�	*

loss1S�=&�       �	S�xhYc�A�	*

loss�F�<ɓ��       �	�@yhYc�A�	*

loss�˕=ӿ�       �	��yhYc�A�	*

lossA/�=��-       �	&�zhYc�A�	*

loss�<�lcF       �	}Y{hYc�A�	*

losse��=rb�       �	F
|hYc�A�	*

lossQD�=Dj��       �	�|hYc�A�	*

lossXa=�m��       �	\}hYc�A�	*

lossZ��=��g       �	�~hYc�A�	*

loss�->��7       �	��~hYc�A�	*

loss�->�[(       �	UhYc�A�	*

loss�#�=}�v�       �	/�hYc�A�	*

loss��=�J�e       �	�؀hYc�A�	*

lossY͇=ҥ:b       �	8��hYc�A�	*

loss�3+>�?ra       �	�E�hYc�A�	*

losso$,=Q��G       �	��hYc�A�	*

loss��=g?h>       �	0�hYc�A�	*

loss�G�=~�6       �	�1�hYc�A�	*

loss���<eϵ�       �	u�hYc�A�	*

loss�w�=}#       �	�"�hYc�A�	*

loss�/=K�BV       �	�hYc�A�	*

loss���<���       �	3��hYc�A�	*

loss#�=t@[�       �	�ţhYc�A�	*

loss$�=��F�       �	�a�hYc�A�	*

lossR�N>N3�W       �	z��hYc�A�	*

loss��2>ggv@       �	[��hYc�A�	*

lossK��=r�i       �	�-�hYc�A�	*

lossֳ�=��dq       �	�ɦhYc�A�	*

loss�1>�|��       �	�b�hYc�A�	*

loss��=��{l       �	*��hYc�A�	*

loss�=\׬�       �	3��hYc�A�	*

loss��>�3N�       �	�=�hYc�A�	*

loss��J=�i/\       �	�٩hYc�A�	*

loss���=�p�       �	Z�hYc�A�	*

loss4�C><t^       �	�E�hYc�A�	*

loss�8>9��       �	HݫhYc�A�	*

loss��=Kc�       �	�v�hYc�A�	*

loss�]>g�T�       �	��hYc�A�	*

loss�+�<���V       �	���hYc�A�	*

lossi��=o�%       �	sL�hYc�A�	*

losslJ�=��Y       �	�hYc�A�	*

loss��J>�V�       �	�hYc�A�	*

loss)/�=�r��       �	�f�hYc�A�	*

loss��l>��^�       �	��hYc�A�	*

loss���=��^G       �	7��hYc�A�	*

loss�>�7�       �	JC�hYc�A�	*

loss=,�=�I�       �	L߲hYc�A�	*

loss��=�_n�       �	�y�hYc�A�	*

loss%�P=@|TI       �	p'�hYc�A�	*

loss�A+=�@_       �		ŴhYc�A�	*

loss��=�a�       �	�_�hYc�A�	*

losso|�=a�0       �	���hYc�A�	*

lossV�=6��       �	նhYc�A�	*

loss1�j=����       �	�m�hYc�A�	*

loss��=V�e�       �	P �hYc�A�	*

loss��
>�W[       �	��hYc�A�	*

loss�h�=@��;       �	I.�hYc�A�	*

loss��=��<S       �	�ӹhYc�A�	*

loss/�g=[��&       �	�{�hYc�A�	*

loss�ĉ=��s�       �	�Z�hYc�A�	*

loss��[>��R�       �	f��hYc�A�	*

loss͑>W�
       �	�hYc�A�	*

losssq�=�2{       �	�&�hYc�A�	*

loss�V�=B�       �	�ýhYc�A�	*

loss�F�<cp��       �	���hYc�A�	*

loss�9>�_n�       �	�*�hYc�A�	*

loss��l=�*       �	�ҿhYc�A�	*

loss�bO>���       �	e�hYc�A�	*

lossiƋ=�;l       �	�<�hYc�A�	*

loss�s�=U�Ԗ       �	�Q�hYc�A�	*

loss��=�"LJ       �	���hYc�A�	*

lossh#�<��m       �	���hYc�A�	*

lossE��<�b9~       �	��hYc�A�	*

loss��>�x^2       �	���hYc�A�	*

loss�G&=��,D       �	ޑ�hYc�A�	*

loss�~�>���       �	�;�hYc�A�	*

loss� =ELM�       �	k��hYc�A�	*

loss��Q<����       �	D��hYc�A�	*

loss�&{<�4�       �	��hYc�A�	*

loss�A<���M       �	��hYc�A�	*

loss~՗=�M�       �	���hYc�A�	*

lossSg�=�˦�       �	k*�hYc�A�	*

loss��>8]��       �		��hYc�A�	*

loss�P�=vAc*       �	�a�hYc�A�	*

loss�HF=}��       �	���hYc�A�	*

loss:84=�h�       �	9��hYc�A�	*

loss��=���4       �	�3�hYc�A�	*

loss���=�*�       �	���hYc�A�	*

loss���=$��h       �	�j�hYc�A�	*

lossR�=R��       �	��hYc�A�	*

lossn�^>NWcM       �	���hYc�A�	*

loss �H=�.�       �	�4�hYc�A�	*

loss�p=<���       �	��hYc�A�	*

loss\�>C��       �	�e�hYc�A�	*

loss&3n="a�o       �		��hYc�A�	*

lossɜK=$��       �	��hYc�A�	*

loss7e=j=z�       �	U4�hYc�A�	*

loss�	=��+4       �	���hYc�A�	*

loss�T=���o       �	d�hYc�A�	*

loss���=��M�       �	���hYc�A�	*

lossX�^=��e^       �	��hYc�A�	*

loss���<=��L       �	(�hYc�A�
*

lossz}>ε�       �	<��hYc�A�
*

loss�f>D�W�       �	�V�hYc�A�
*

loss}�=� �?       �	-'�hYc�A�
*

loss|̣=�ڧ3       �	���hYc�A�
*

loss�2=�1B       �	mT�hYc�A�
*

loss�8�=v�ټ       �	c��hYc�A�
*

loss���<t���       �	Ӣ�hYc�A�
*

loss?�
=�`�(       �	MN�hYc�A�
*

loss��=��z3       �	?��hYc�A�
*

loss t-=�s�       �	p|�hYc�A�
*

loss�?I>g�%       �	��hYc�A�
*

lossr�=q�r�       �	��hYc�A�
*

loss�8�=�N�"       �	Jb�hYc�A�
*

loss�.}=4vo       �	*��hYc�A�
*

loss��=˝31       �	���hYc�A�
*

loss�M<Jæ�       �	rO�hYc�A�
*

loss:�=Fݢ�       �	p"�hYc�A�
*

loss�Ab=�u(�       �	I��hYc�A�
*

loss
�v>�5t       �	+k�hYc�A�
*

loss=�=D��       �	��hYc�A�
*

loss؋>=�i       �	��hYc�A�
*

loss���<���       �	RG�hYc�A�
*

loss�T;>���       �	"��hYc�A�
*

loss���=NU4�       �	k~�hYc�A�
*

loss��=�U��       �	��hYc�A�
*

loss��d=Z��       �	�U�hYc�A�
*

loss���<�N1       �	 ��hYc�A�
*

loss���<���       �	ˁ�hYc�A�
*

loss��>�j\�       �	�^�hYc�A�
*

loss��=��\K       �	���hYc�A�
*

lossf�	>A�l        �	���hYc�A�
*

loss��=T�R       �	f�hYc�A�
*

loss��%=�((�       �	���hYc�A�
*

lossH�w<&.C�       �	}��hYc�A�
*

lossv?�<ĸ��       �	�9�hYc�A�
*

loss�D =���       �	���hYc�A�
*

loss�=���a       �	�u�hYc�A�
*

lossn�Z="��       �	�hYc�A�
*

loss䲷=�[�`       �	���hYc�A�
*

loss�
�=��L       �	7��hYc�A�
*

loss|��=J+\       �	�5�hYc�A�
*

loss��t=&A��       �	���hYc�A�
*

loss��}>}       �	;p�hYc�A�
*

loss�>d�yw       �	h�hYc�A�
*

loss��<�϶�       �	ϣ�hYc�A�
*

loss�#�=</[       �	�?�hYc�A�
*

loss�G>h�K       �	���hYc�A�
*

lossK�>p��Z       �	�h�hYc�A�
*

loss��=��
       �	��hYc�A�
*

loss���=�lu       �	,��hYc�A�
*

loss�ɾ=��/�       �	�P�hYc�A�
*

losse
�=����       �	Z��hYc�A�
*

loss�p�=�Io       �	��hYc�A�
*

lossn�e=��S�       �	S"�hYc�A�
*

lossXJ>���       �	���hYc�A�
*

loss�;�=5?�       �	�O�hYc�A�
*

loss�4�='CG�       �	}��hYc�A�
*

lossW�d=G�il       �	,��hYc�A�
*

loss���<�[E�       �	m�hYc�A�
*

loss)�=*%P'       �	���hYc�A�
*

lossJ4\<\��%       �	DM�hYc�A�
*

loss/��=��?.       �	���hYc�A�
*

loss��=Q�{       �	R~�hYc�A�
*

loss@��=���       �	��hYc�A�
*

lossE&�=��       �	���hYc�A�
*

loss�G=���       �	&V iYc�A�
*

loss�q�=�-�       �	�� iYc�A�
*

loss���<���        �	h�iYc�A�
*

loss�#U=�h�       �	lxiYc�A�
*

loss0�=WF'�       �	�WiYc�A�
*

lossh�<f�       �	��iYc�A�
*

loss��=�2vj       �	��iYc�A�
*

loss��:>���!       �	iYc�A�
*

loss���=ڻ�c       �	O�iYc�A�
*

loss�ݗ=zO�       �	�hiYc�A�
*

loss�]�<C�z�       �	L�iYc�A�
*

loss)�i=�B$P       �	x�iYc�A�
*

lossn�	>4T�       �	�<iYc�A�
*

lossN/!>=w(       �	��iYc�A�
*

lossJ-=��6�       �	�y	iYc�A�
*

lossR`=%��       �	
iYc�A�
*

loss7G�<5?��       �	s�
iYc�A�
*

loss�=+�JF       �	P8iYc�A�
*

loss�f�<�a��       �	��iYc�A�
*

loss��]=6��       �	�eiYc�A�
*

loss�`^=�XJ9       �	BiYc�A�
*

loss���<�2b       �	��iYc�A�
*

loss;B=|'Q�       �	�<iYc�A�
*

lossd9C=1��       �	E�iYc�A�
*

loss q�=��h@       �	�iYc�A�
*

loss�~�=�T\�       �	�?iYc�A�
*

loss�-=>�E       �	#�iYc�A�
*

lossW�=/F��       �	{iYc�A�
*

loss/5 >hT��       �	O\iYc�A�
*

loss6��=���       �	�1iYc�A�
*

loss��#=��!!       �	B�iYc�A�
*

loss
;�=��ʫ       �	hiYc�A�
*

loss�J�=�*6       �	\iYc�A�
*

loss?=� ��       �	ݗiYc�A�
*

lossz}=h#d�       �	4iYc�A�
*

loss&&<�N��       �	��iYc�A�
*

loss�ͬ=����       �	WwiYc�A�
*

lossT�=�$       �	p$iYc�A�
*

loss�=61��       �	��iYc�A�
*

loss$��<�Lg       �	�iYc�A�
*

loss�=�zw
       �	VeiYc�A�
*

loss��"=��ƣ       �	&�iYc�A�
*

loss��=\�8�       �	F�iYc�A�
*

loss:�-=���J       �	P5iYc�A�
*

loss3��=�M       �	��iYc�A�
*

lossa4=�v�)       �	&piYc�A�
*

loss<��=�P       �	�	iYc�A�
*

loss�(J=�@o       �	�iYc�A�
*

loss��<�g        �	,FiYc�A�
*

loss�0�<�s�5       �	��iYc�A�
*

losssi0=��mS       �	� iYc�A�
*

lossx�b=�#��       �	U!iYc�A�
*

loss?W�=b(�       �	��!iYc�A�
*

loss�>4��       �	��"iYc�A�
*

loss�Z�={)ir       �	�2#iYc�A�
*

loss�=R>Q�       �	\�#iYc�A�
*

lossJ=㒯�       �	�d$iYc�A�
*

loss�'=1(�p       �	$H%iYc�A�
*

loss�/�=�� �       �	T�%iYc�A�
*

loss�)b<�o�M       �	��&iYc�A�
*

loss3(]=����       �	�I'iYc�A�
*

loss$��=aV�y       �	�'iYc�A�
*

loss�ژ=�/L�       �	h�(iYc�A�*

loss��t=$�0       �	�G)iYc�A�*

lossS�=��L�       �	��)iYc�A�*

loss�=��X�       �	؜*iYc�A�*

loss�4�=�B��       �	L+iYc�A�*

loss�0.=^��d       �	5�+iYc�A�*

loss�ۃ<�e�g       �	�,iYc�A�*

lossL��=}<�p       �	]3-iYc�A�*

loss��>5�M�       �	��-iYc�A�*

loss\1=ˎ�d       �	�p.iYc�A�*

loss�>2�_       �	�/iYc�A�*

loss��=k��y       �	}�/iYc�A�*

lossm�">"�7�       �	�Z0iYc�A�*

lossQ��<�W��       �	z�0iYc�A�*

lossь�=5zY�       �	4�1iYc�A�*

loss��=����       �	�;2iYc�A�*

loss��>Ɗ        �	{�2iYc�A�*

loss#˚=�5�k       �	�~3iYc�A�*

lossD2=T�B�       �	�4iYc�A�*

loss.V�=���       �	�4iYc�A�*

loss�M�=��QH       �	iQ5iYc�A�*

loss|�=Ie��       �	I�5iYc�A�*

loss��=z�Q       �	�6iYc�A�*

loss)��<4���       �	Y27iYc�A�*

loss3R>=�	�       �	��8iYc�A�*

lossSd=�!uP       �	779iYc�A�*

lossh�(=�X�       �	��9iYc�A�*

loss�$�=��       �	em:iYc�A�*

loss���=1�       �	�;iYc�A�*

loss�L�=�ܭ       �	U�;iYc�A�*

loss�Й=X=֟       �	m�<iYc�A�*

lossX�0>��i�       �	4M=iYc�A�*

loss�L�<� �       �	�=iYc�A�*

loss��=)Ho�       �	ׅ>iYc�A�*

loss�w�=6���       �	$?iYc�A�*

loss�=Ǩ�       �	�?iYc�A�*

loss�q8=�g��       �	��@iYc�A�*

loss\G=*�b�       �	��AiYc�A�*

loss���=]e��       �	��BiYc�A�*

loss��<̊��       �	gCiYc�A�*

loss�O)=�t�K       �	5FDiYc�A�*

loss�#�= �2       �	��DiYc�A�*

loss\;�=�WD,       �	v�EiYc�A�*

loss���=+U��       �	�IFiYc�A�*

loss4"�=���       �	��FiYc�A�*

loss?��=5Z�       �	֐GiYc�A�*

loss��K<𮯂       �	f1HiYc�A�*

lossHD�;2��       �	��HiYc�A�*

lossvy�=���h       �	��IiYc�A�*

loss8IF=/�L       �	�JJiYc�A�*

losssf=�Ç�       �	�JiYc�A�*

loss��=���5       �	M�KiYc�A�*

loss
>I�#/       �	� LiYc�A�*

lossC,�=�q�       �	��LiYc�A�*

loss���<���       �	�[MiYc�A�*

loss�()>�Z�       �	'�MiYc�A�*

loss��"=��A|       �	�_OiYc�A�*

loss���=G���       �	��OiYc�A�*

loss��1=��]/       �	��PiYc�A�*

loss)&�=:��I       �	�KQiYc�A�*

loss��=���r       �	��QiYc�A�*

loss~�>m�8�       �	��RiYc�A�*

loss�j�=9{�       �	SzSiYc�A�*

loss�Q=�t��       �	�TiYc�A�*

loss�]�=��M~       �	��TiYc�A�*

loss2�=�C       �	|HUiYc�A�*

lossҾ=;�^       �	�UiYc�A�*

lossHn�=���       �	{ViYc�A�*

loss��=Y�i       �	�&WiYc�A�*

loss�cl=���j       �	��WiYc�A�*

loss��>=��ʇ       �	�SXiYc�A�*

loss���=N���       �	��XiYc�A�*

loss?z�=3�&       �	@�YiYc�A�*

lossOC =�eD�       �	�ZiYc�A�*

loss��==	�i       �	eT[iYc�A�*

loss��=�P��       �	9�[iYc�A�*

loss�C�=�Oo       �	�\iYc�A�*

loss���=��1L       �	�3]iYc�A�*

loss���=E���       �	P�]iYc�A�*

lossC*�<<I�~       �	�_^iYc�A�*

loss�sv=v�2       �	��^iYc�A�*

loss�.\=1�       �	��_iYc�A�*

loss���<69��       �	�.`iYc�A�*

loss��G=i���       �	z�`iYc�A�*

loss�bd=��k       �	�iaiYc�A�*

loss@�<	k       �	�biYc�A�*

lossa�D<fO}       �	ӾbiYc�A�*

losss+=Y���       �	=dciYc�A�*

loss�K	>���       �	N
diYc�A�*

loss�>c=37       �	��diYc�A�*

loss�� >��Һ       �	�OeiYc�A�*

lossr�=shr       �	��eiYc�A�*

loss/y>�d�Z       �	O�fiYc�A�*

loss�$=(�%�       �	M/giYc�A�*

loss��A=��_�       �	��giYc�A�*

loss�,}=jK�       �	�ihiYc�A�*

lossk�	>����       �	�iiYc�A�*

loss,�3=�LP]       �	Z�iiYc�A�*

lossH�
>pc��       �	2;jiYc�A�*

loss�L�<��c       �	��jiYc�A�*

loss
J�={� V       �	�}kiYc�A�*

lossL��<~IeH       �	?liYc�A�*

lossZ{W=5���       �	j�liYc�A�*

loss�;D=�m��       �	6WmiYc�A�*

loss"@=���       �	�miYc�A�*

loss�^h=K5St       �	��niYc�A�*

loss��=J�j       �	_(oiYc�A�*

lossq�=�b}       �	��oiYc�A�*

loss3ؐ=��E2       �	�YpiYc�A�*

loss�u�=z��       �	�qiYc�A�*

loss�=��J�       �	d�qiYc�A�*

lossa� >o��q       �	{JriYc�A�*

loss�Y�=
��       �	�riYc�A�*

loss
��<�Q�
       �	�siYc�A�*

loss��=�T�       �	�tiYc�A�*

lossL�=��K       �	Z�tiYc�A�*

loss���<,=��       �	�TuiYc�A�*

lossHp�=�hh       �	��uiYc�A�*

loss�>�7s�       �	f�viYc�A�*

loss���<I.0J       �	KwiYc�A�*

loss��=���       �	F�wiYc�A�*

loss��=�T�f       �	~xiYc�A�*

loss6��=��`       �	<�yiYc�A�*

loss�y+=Ĥ9�       �	K#ziYc�A�*

loss�c�<�.�*       �	��ziYc�A�*

loss���<����       �	�b{iYc�A�*

loss� �<���g       �	�|iYc�A�*

lossܑ>�I�       �	��|iYc�A�*

loss/�F=vl��       �	M�}iYc�A�*

loss��=��V+       �	~~iYc�A�*

loss
ot=�b       �	ȷ~iYc�A�*

loss��=��       �	�UiYc�A�*

loss�8d<6I-�       �	��iYc�A�*

loss߰p=Ʈ�8       �	犀iYc�A�*

loss�f�=��}P       �	�/�iYc�A�*

loss�o=�F�       �	�ȁiYc�A�*

loss*~]=���       �	*�iYc�A�*

loss�[�>��e       �	��iYc�A�*

loss�N�=��6�       �	�M�iYc�A�*

losss7=���       �	��iYc�A�*

losshS>R+˛       �	w��iYc�A�*

loss^�<�8�       �	��iYc�A�*

lossJ�&=��AL       �	0��iYc�A�*

loss�Ɠ<��z�       �	T�iYc�A�*

loss��V=~o       �	�iYc�A�*

loss��N=��       �	Н�iYc�A�*

loss�R=�L       �	6�iYc�A�*

loss���=A�D_       �	�ˉiYc�A�*

loss<���       �	�l�iYc�A�*

loss�n=dn%       �	��iYc�A�*

loss���<tG�       �	�iYc�A�*

loss2�r='cnr       �	�o�iYc�A�*

loss��J=��]�       �	\�iYc�A�*

lossjF>p7        �	_��iYc�A�*

lossv:?=h��<       �	vO�iYc�A�*

loss��=ڡ�M       �	%�iYc�A�*

loss���=����       �	��iYc�A�*

loss�9�=t~3�       �	���iYc�A�*

loss���<3<��       �	�8�iYc�A�*

loss�U�=�a�       �	�ԑiYc�A�*

loss��==۽�       �	|)�iYc�A�*

lossDA=+��       �	�ȓiYc�A�*

loss��(=t��T       �	���iYc�A�*

loss�3=���       �	(I�iYc�A�*

loss�˂=,$��       �	��iYc�A�*

losswx�=�3��       �	��iYc�A�*

loss�H=B�@�       �	��iYc�A�*

loss}f�=.|aV       �	Ͱ�iYc�A�*

loss6�>��y       �	�`�iYc�A�*

loss3�%>v��       �	P�iYc�A�*

loss|҆=z4ߢ       �	�!�iYc�A�*

loss�gw=�>��       �	�ŚiYc�A�*

loss�g=09R�       �	;m�iYc�A�*

lossz =��BR       �	��iYc�A�*

loss#%�=x��       �	���iYc�A�*

loss)r%=�%       �	�t�iYc�A�*

loss.H=��g       �	t(�iYc�A�*

lossn�U<�0�       �	*ƞiYc�A�*

loss0%=ap Z       �	�֟iYc�A�*

loss6��=��       �	ro�iYc�A�*

loss�<z=�$-�       �	x	�iYc�A�*

loss�Ox=3��       �	��iYc�A�*

loss�ғ=�L�z       �	�@�iYc�A�*

lossd	�=٤��       �	6�iYc�A�*

loss�2�=˝��       �	<��iYc�A�*

loss�S=�%׵       �	!�iYc�A�*

lossZ�K=7��       �	7äiYc�A�*

loss�Q=�J�|       �	�\�iYc�A�*

loss��+>ZR#5       �	��iYc�A�*

lossf�{=��y/       �	+��iYc�A�*

lossEE�<ӣc       �	��iYc�A�*

loss���<����       �	���iYc�A�*

loss�܉=�wyT       �	�O�iYc�A�*

loss�=@db       �	O�iYc�A�*

loss ��=����       �	8��iYc�A�*

loss�4\=D�	       �	.�iYc�A�*

loss��<�d�&       �	-��iYc�A�*

loss��=>���       �	�J�iYc�A�*

lossvZ�=���s       �	Q��iYc�A�*

loss�>�I#�       �	��iYc�A�*

loss�-�=�R       �	�0�iYc�A�*

losswp�=��       �	ɭiYc�A�*

loss��=`sn       �	�^�iYc�A�*

loss���=uʓ�       �	���iYc�A�*

loss扵=�Y}h       �	O��iYc�A�*

loss���=��@�       �	@/�iYc�A�*

lossM&t>�ُ       �	[]�iYc�A�*

loss��=�$��       �	���iYc�A�*

lossEy1>�M�       �	ܟ�iYc�A�*

loss�c=�`Z       �	�@�iYc�A�*

loss��,=A'�       �	G�iYc�A�*

loss=f�=rAE�       �	-��iYc�A�*

loss]ݲ<0A�4       �	�;�iYc�A�*

loss��=���1       �	��iYc�A�*

loss��=)���       �	'��iYc�A�*

loss��=��a       �	��iYc�A�*

loss�><t	y       �	���iYc�A�*

loss���=9���       �	S]�iYc�A�*

lossaD�=UH+�       �	��iYc�A�*

loss`�<=v�       �	���iYc�A�*

loss�:�=�0��       �	g'�iYc�A�*

loss��=��Q       �	��iYc�A�*

loss�E�=�VA�       �	5^�iYc�A�*

loss�#>=���       �	B�iYc�A�*

loss���=���       �	ж�iYc�A�*

loss? L=#^�       �	�Y�iYc�A�*

loss��N=l:%       �	s��iYc�A�*

losss��=��u�       �	Ę�iYc�A�*

loss�S�<��v:       �	[C�iYc�A�*

loss,��=k	�       �	��iYc�A�*

loss�U=�g2V       �	�t�iYc�A�*

lossT�<�b�       �	��iYc�A�*

lossUH=크v       �	
��iYc�A�*

loss4	>1��:       �	���iYc�A�*

loss)g>8��       �	��iYc�A�*

loss�H�=Ȃ+�       �	�I�iYc�A�*

lossJ�L=t�Ĉ       �	��iYc�A�*

loss��7=S���       �	��iYc�A�*

loss}�U=��v]       �	"7�iYc�A�*

loss�6+>�Aı       �	���iYc�A�*

loss3��=�aK�       �	�p�iYc�A�*

loss=
׀       �	`X�iYc�A�*

loss��=B7�       �	���iYc�A�*

loss�T�=��P�       �	���iYc�A�*

loss
9�<+�l�       �	�$�iYc�A�*

loss	q�=*��       �	���iYc�A�*

loss��.>���
       �	�[�iYc�A�*

lossɨV>䋒<       �	|��iYc�A�*

lossxh>A�H       �	7��iYc�A�*

loss�C>��<       �	1(�iYc�A�*

loss6U�=�ټ5       �	���iYc�A�*

loss�h8=(d��       �	y�iYc�A�*

loss���=n;�       �	�iYc�A�*

loss���<��=�       �	��iYc�A�*

lossWF�<�7X�       �	*t�iYc�A�*

loss=a$=UFJ�       �	��iYc�A�*

loss��=����       �	���iYc�A�*

loss�W:=�=n�       �	'P�iYc�A�*

loss���=�8��       �	���iYc�A�*

loss�h�=��Ώ       �	���iYc�A�*

loss�1�=�'��       �	{-�iYc�A�*

loss^G=�Ob       �	.��iYc�A�*

lossOj%=�S-�       �	c�iYc�A�*

loss1��=�c�       �	F	�iYc�A�*

loss�YN=�4PV       �	��iYc�A�*

loss��1=��l       �	�\�iYc�A�*

loss���=��Д       �	Y��iYc�A�*

loss��
=8E�       �	e��iYc�A�*

loss�&�<4j0�       �	
L�iYc�A�*

loss)ؑ<���       �	#��iYc�A�*

loss�*=�G�       �	w��iYc�A�*

loss<yL=jk9�       �	�3�iYc�A�*

loss�->���c       �	���iYc�A�*

loss8�;>����       �	�z�iYc�A�*

lossl�*=K{l�       �	6 �iYc�A�*

loss/��=�ZQ�       �	���iYc�A�*

lossQp�=��"       �	�l�iYc�A�*

lossW��=���R       �	��iYc�A�*

lossȼ�<���x       �	���iYc�A�*

lossd}o=��       �	mX�iYc�A�*

lossFu�=���       �	�iYc�A�*

lossة�=C��a       �	#��iYc�A�*

loss���=�j=       �	�T�iYc�A�*

loss��M=���       �	��iYc�A�*

lossQ�=���D       �	���iYc�A�*

loss(cp=���       �	t&�iYc�A�*

loss_4=B?_�       �	���iYc�A�*

loss3+�<2%>�       �	�g�iYc�A�*

lossz=�<��Qi       �	��iYc�A�*

loss3��=	B�       �	)��iYc�A�*

lossxl�<�	ͤ       �	K>�iYc�A�*

lossli.=p��G       �	���iYc�A�*

lossd��=Ugz       �	���iYc�A�*

lossSZ�=��@       �	=�iYc�A�*

loss%=��
[       �	I��iYc�A�*

lossn%=J���       �	e��iYc�A�*

loss�P�<��&       �	���iYc�A�*

loss)�=���       �	tB�iYc�A�*

lossf�2=�
�z       �	���iYc�A�*

loss)�x=3~�       �	;n�iYc�A�*

loss��T<�l?       �	O�iYc�A�*

loss%R�=_[��       �	��iYc�A�*

loss�އ=1�~       �	�4�iYc�A�*

lossL7E=��       �	���iYc�A�*

loss���=�}��       �	�i�iYc�A�*

loss�p<G���       �	v�iYc�A�*

loss2�=��       �	l��iYc�A�*

loss=ZM=��^u       �	~��iYc�A�*

loss�5�=<[�n       �	�?�iYc�A�*

loss,��=���       �	���iYc�A�*

loss�-�<km       �	�o�iYc�A�*

loss%�=>�t�       �	��iYc�A�*

loss��X=+�|�       �	ɯ�iYc�A�*

lossT+�<�I��       �	MM�iYc�A�*

loss���=^��W       �	��iYc�A�*

loss!�L;�G]       �	ҍ�iYc�A�*

loss�<��@�       �	p'�iYc�A�*

loss�_n=�(>       �	���iYc�A�*

loss�G&=��       �	k�iYc�A�*

loss���=/�O�       �	>�iYc�A�*

loss{@�=>�       �	���iYc�A�*

loss�#>�i��       �	�~�iYc�A�*

loss6N=l��3       �	r�iYc�A�*

lossZ�=��;}       �	 ��iYc�A�*

loss��=v��C       �	�N�iYc�A�*

loss�&=5"v�       �	���iYc�A�*

loss8M�;�jX       �	5� jYc�A�*

loss/�<�ۗt       �	9jYc�A�*

lossE��<�?Ϭ       �	��jYc�A�*

loss/a�;��ef       �	�sjYc�A�*

lossq�h=� �       �	�jYc�A�*

loss>=�å�       �	��jYc�A�*

lossR��=L��w       �	DOjYc�A�*

loss�'<��4�       �	T�jYc�A�*

loss��;�&�;       �	}�jYc�A�*

loss��4<{���       �	0,jYc�A�*

loss4{K=׭M       �	��jYc�A�*

loss���=P�       �	ujYc�A�*

loss�u5=�
�>       �	('jYc�A�*

loss���;3��       �	��jYc�A�*

loss��<'}q�       �	�f	jYc�A�*

loss��>K�
       �	�
jYc�A�*

loss�1<g��i       �	.�
jYc�A�*

loss��=��0       �	�IjYc�A�*

loss��/=�K�       �	��jYc�A�*

loss1�=��s       �	��jYc�A�*

loss�K�<w(       �	�HjYc�A�*

loss�n�<�~Q�       �	��jYc�A�*

lossQ�=@k�S       �	d�jYc�A�*

loss��=N�rL       �	�5jYc�A�*

lossf:=��       �	��jYc�A�*

lossS�=7�!�       �	��jYc�A�*

loss�;�=�       �	HOjYc�A�*

loss=�>�U��       �	��jYc�A�*

loss�� >���       �	єjYc�A�*

loss<mO="s_L       �	�9jYc�A�*

loss�[c=mK��       �	��jYc�A�*

loss���=.,O       �	�jYc�A�*

loss
3x=}���       �	2jYc�A�*

loss]+�=X'X       �	��jYc�A�*

lossjm�=4�h�       �	��jYc�A�*

loss��^=��1       �	�jjYc�A�*

loss��<�D�       �	�jYc�A�*

loss��~=L��       �	��jYc�A�*

loss;�=�ܡ�       �	`�jYc�A�*

loss|�	=��%X       �	9'jYc�A�*

loss�
=�%g       �	z�jYc�A�*

lossh<ܨ�       �	M�jYc�A�*

lossmZD=',p�       �	9GjYc�A�*

lossþm=p��       �	2�jYc�A�*

loss8��=O_u$       �	0�jYc�A�*

loss o~=uF�       �	W"jYc�A�*

loss�5=ڿ�       �	��jYc�A�*

loss���<8U�       �	SujYc�A�*

loss�=�?*�       �	�$ jYc�A�*

loss�Z�;6�H       �	�� jYc�A�*

loss�C=�-5�       �	�m!jYc�A�*

loss�E=��        �	Z"jYc�A�*

lossN�<mjv�       �	1�"jYc�A�*

losso��=��2       �	vS#jYc�A�*

lossťr==�H�       �	N�#jYc�A�*

loss��==�1�       �	͕$jYc�A�*

loss��<��h1       �	2>%jYc�A�*

loss��=s�       �	Y�%jYc�A�*

lossv�8=e�3       �	��&jYc�A�*

loss�Y=�/Α       �	('jYc�A�*

lossn@�=�y��       �	��'jYc�A�*

loss�֚=��P       �	8f(jYc�A�*

loss�=?Ug       �	)jYc�A�*

loss�(�<���       �	4�)jYc�A�*

loss�a�=�}�       �	+5*jYc�A�*

loss�3`<[YE�       �	��*jYc�A�*

loss��M<r��#       �	�e+jYc�A�*

loss��=�s`�       �	��BjYc�A�*

lossl�[=��w�       �	�CjYc�A�*

loss��=�tx       �	Y�DjYc�A�*

lossQ�=�Z�       �	�"EjYc�A�*

lossV*�=pv��       �	��EjYc�A�*

lossb�<b ˁ       �	urFjYc�A�*

losss�=R
�       �	/GjYc�A�*

loss��_=���z       �	+�GjYc�A�*

lossS��=A֓�       �	��HjYc�A�*

losstR�=90?�       �	�IjYc�A�*

loss�=�*��       �	�JjYc�A�*

lossh�H=�eS�       �	�hKjYc�A�*

loss�&�=�mD       �	�LjYc�A�*

lossؿ�=.S&Y       �	_�LjYc�A�*

loss ]>���       �	nMjYc�A�*

loss���=Z�)       �	OuNjYc�A�*

loss���;�m��       �	=OjYc�A�*

loss���=����       �	��OjYc�A�*

loss��U=͓��       �	�QjYc�A�*

lossIM>h��       �	��QjYc�A�*

loss��=�$3       �	�MRjYc�A�*

loss$Ҝ=���       �	�SjYc�A�*

loss�u�<��        �	0�SjYc�A�*

loss3h=	�=�       �	 ]TjYc�A�*

loss��z=���       �	VjYc�A�*

loss�]=R}�       �	O�VjYc�A�*

lossPN=�4�       �	)\WjYc�A�*

lossQP�<�T�       �	��WjYc�A�*

loss��=�G;       �	7�XjYc�A�*

lossć�<O��       �	3YjYc�A�*

lossl�<�b+1       �	��YjYc�A�*

lossÇ�<�,_Z       �	A�ZjYc�A�*

loss�CM=�ݶ]       �	![jYc�A�*

lossSQ	>���>       �	N�[jYc�A�*

loss�==p�       �	'�\jYc�A�*

loss�e=�Z|�       �	<]jYc�A�*

loss�=���^       �	+P^jYc�A�*

loss�˭=��-       �	o�^jYc�A�*

loss��=����       �	��_jYc�A�*

loss`^�=����       �	X�`jYc�A�*

loss{t=O�l       �	A*ajYc�A�*

loss�L=�jz�       �	��ajYc�A�*

loss� �<[�
       �	kbbjYc�A�*

lossL�j=���       �	�bjYc�A�*

loss\H
=�~U       �	�cjYc�A�*

lossD~�=`��d       �	�5djYc�A�*

loss�	�=S�[       �	!�djYc�A�*

loss��=�h�       �	odejYc�A�*

loss�g0=E�       �	{�ejYc�A�*

loss\�<�HƸ       �	B�fjYc�A�*

loss��;=�H�H       �	7gjYc�A�*

loss7>���       �	+�gjYc�A�*

loss��=��       �	ЀhjYc�A�*

loss�>�WX       �	B#ijYc�A�*

loss#!=7a�Y       �	��ijYc�A�*

loss���<��       �	�\jjYc�A�*

lossj��;�-�       �	�jjYc�A�*

lossT�!<�b��       �	��kjYc�A�*

losse$#=�|��       �	7ljYc�A�*

loss
�C=����       �	/�ljYc�A�*

losssL=a�E       �	\vmjYc�A�*

lossf�=��S       �	�njYc�A�*

losso�$=|]>�       �	�njYc�A�*

lossu=����       �	XVojYc�A�*

lossnÁ=�b��       �	��pjYc�A�*

loss�Ҙ=�5^�       �	�$qjYc�A�*

lossf��=�-�2       �	��qjYc�A�*

losse�>���       �	OurjYc�A�*

loss�r>�22[       �	� sjYc�A�*

loss��D=MM�       �	��sjYc�A�*

loss��&=���       �	�tjYc�A�*

loss�C=근
       �	_ujYc�A�*

lossFT�=��&�       �	�ujYc�A�*

loss}�N=<v       �	�vjYc�A�*

losso�=��De       �	h%wjYc�A�*

lossM��=c�       �	_�wjYc�A�*

loss��"=��o       �	>uxjYc�A�*

loss�Y�=�Dh       �	�yjYc�A�*

loss�t6=�J�       �	��yjYc�A�*

loss;�=���       �	�azjYc�A�*

loss��8=g�٬       �	){jYc�A�*

loss�t�=c�8�       �	q�{jYc�A�*

loss��>���-       �	aP|jYc�A�*

losssM�<��X2       �	1�|jYc�A�*

lossZ5B=���       �	��}jYc�A�*

loss�W�<�B��       �	s-~jYc�A�*

loss�V
>����       �	��~jYc�A�*

loss�h�=o��^       �	�cjYc�A�*

lossC�<y�D       �	�jYc�A�*

loss��X=��|�       �	Ֆ�jYc�A�*

loss�Ͻ=��yC       �	�0�jYc�A�*

loss߀�<�$�       �	kӁjYc�A�*

lossJ��=!�I�       �	�x�jYc�A�*

loss@AT=�{       �	�'�jYc�A�*

loss�d�=�X|       �	��jYc�A�*

loss�=�<�-�       �	 ĄjYc�A�*

loss,Wx=��=�       �	���jYc�A�*

loss��<��6�       �	'L�jYc�A�*

loss�a	>�.�0       �	*��jYc�A�*

loss_�W=j9�       �	0��jYc�A�*

loss��=9o��       �	�@�jYc�A�*

loss�2=-k��       �	QڈjYc�A�*

loss��>p.V�       �	�u�jYc�A�*

loss��?=�Η�       �	t�jYc�A�*

lossz��=V?¯       �	ꮊjYc�A�*

loss�Y(>���       �	+M�jYc�A�*

loss��=�<7       �	<��jYc�A�*

loss���<���       �	犌jYc�A�*

loss��=�a_       �	�f�jYc�A�*

loss��Q=8�l       �	E�jYc�A�*

loss���=�U�       �	���jYc�A�*

loss��G=��П       �	LO�jYc�A�*

lossɁ�=�,)       �	��jYc�A�*

lossH�[<�ԇ       �	���jYc�A�*

loss��!=�r�       �	�2�jYc�A�*

loss2�%=��9       �	�ϑjYc�A�*

loss��<a�v       �	Lk�jYc�A�*

loss浟=��o�       �	��jYc�A�*

loss��-=䴒�       �	>��jYc�A�*

loss/ʍ=7�'       �	�U�jYc�A�*

lossqTb=���       �	��jYc�A�*

loss�jI=Ѷ�       �	��jYc�A�*

loss��(>���       �	�;�jYc�A�*

loss�R`=����       �	ߖjYc�A�*

loss�\.<�B       �	�|�jYc�A�*

loss��=�ae       �	5$�jYc�A�*

loss���=�
       �	{��jYc�A�*

lossAc�<N�8|       �	�S�jYc�A�*

losse�=���       �	��jYc�A�*

loss���=wA�       �	ĳ�jYc�A�*

loss��$=?�u       �	<��jYc�A�*

loss�f�==ӂ�       �	e�jYc�A�*

losszm =����       �	�ʜjYc�A�*

loss���<�`c       �	vn�jYc�A�*

loss��=Y"�]       �	��jYc�A�*

loss�O<4Nt       �	雞jYc�A�*

loss�y=u�n�       �	�`�jYc�A�*

loss��<�U�       �	�*�jYc�A�*

loss���<�n��       �	2ʠjYc�A�*

loss�3�<|�c,       �	�b�jYc�A�*

loss�?<�>��       �	��jYc�A�*

loss�6�=����       �	!��jYc�A�*

loss[r=��5       �	P�jYc�A�*

lossoT]=[(�       �	��jYc�A�*

loss*Ҽ=�I&7       �	Ԙ�jYc�A�*

loss7�1=�J�       �	�A�jYc�A�*

loss�p=]��-       �	`�jYc�A�*

loss��;~���       �	!��jYc�A�*

loss�f�;r��       �	]5�jYc�A�*

loss%��< ��<       �	ݧjYc�A�*

loss5�=t��       �	܄�jYc�A�*

loss�H�=f�N       �	U.�jYc�A�*

loss�-�=A�4       �	�֩jYc�A�*

loss���=R���       �	6u�jYc�A�*

loss�I=����       �	�jYc�A�*

lossO~�<��d       �	�jYc�A�*

loss�P-=���       �	[@�jYc�A�*

loss��8>��`       �	ҬjYc�A�*

loss�u>g�Y\       �	�i�jYc�A�*

loss��B=v���       �	F�jYc�A�*

loss߽�=�       �	s��jYc�A�*

losso�=/ԧT       �	G=�jYc�A�*

lossՙ=�L�       �	S�jYc�A�*

losszq�;p��       �	r��jYc�A�*

loss?�"=2��8       �	�B�jYc�A�*

loss'=T1t�       �	�ڱjYc�A�*

loss���<���       �	z�jYc�A�*

loss�0c<�FE}       �	'�jYc�A�*

lossHA=���       �	=��jYc�A�*

loss�s�<�x�       �	T�jYc�A�*

loss �=��       �	��jYc�A�*

loss�< c�       �	���jYc�A�*

loss���=�       �	v�jYc�A�*

losso��=�q�       �	��jYc�A�*

loss�1=���       �	�ŷjYc�A�*

lossf�<�Xg       �	Rd�jYc�A�*

loss1��=�Z �       �	�	�jYc�A�*

loss�Q�=q���       �	/��jYc�A�*

loss҆]<T5��       �	Z��jYc�A�*

loss�[�<h9U       �	�V�jYc�A�*

loss��<.SN       �	���jYc�A�*

loss	�c=�]�7       �	ӟ�jYc�A�*

loss�4�<��2�       �	�8�jYc�A�*

loss/�=�U�       �	�νjYc�A�*

loss%�<s7��       �	Eh�jYc�A�*

lossl]�<��F�       �	��jYc�A�*

loss�L-=]��r       �	���jYc�A�*

loss�Zm=<�4       �	ob�jYc�A�*

lossTEX=�X��       �	�jYc�A�*

loss�$�=T?��       �	���jYc�A�*

loss�V�<��7�       �	1x�jYc�A�*

lossl�e=����       �	���jYc�A�*

loss�	�<Gm        �	tb�jYc�A�*

losso#=R�$�       �	6>�jYc�A�*

loss�$<7�a       �	���jYc�A�*

lossx�?=���4       �	˄�jYc�A�*

losso$�<{�       �	�&�jYc�A�*

lossq��=���C       �	���jYc�A�*

lossDg�=H�(       �	�g�jYc�A�*

loss6�w<����       �	�	�jYc�A�*

loss t�<���v       �	���jYc�A�*

lossl�<5ك�       �	�R�jYc�A�*

losshA=�g       �	���jYc�A�*

loss�C=�V(�       �	���jYc�A�*

loss���=Ҥ�g       �	�2�jYc�A�*

loss܌�<3&O       �	$��jYc�A�*

lossy|�=JOW�       �	���jYc�A�*

loss�|=���       �	,�jYc�A�*

loss�,�=eRJ       �	���jYc�A�*

loss�U={��_       �	�_�jYc�A�*

lossm��<�C}       �	���jYc�A�*

loss��?=�J+       �	���jYc�A�*

loss���<]�G�       �	�f�jYc�A�*

lossҝ�<EX��       �	��jYc�A�*

loss��=vF�
       �	7��jYc�A�*

lossԵ�=���       �	xA�jYc�A�*

lossIw7=Ҿ��       �	j��jYc�A�*

loss�ů=��y�       �	Tt�jYc�A�*

loss�>�gR       �	�jYc�A�*

loss%�X=���=       �	Ҫ�jYc�A�*

lossL�r=�a��       �	�?�jYc�A�*

loss!	�<�T��       �	���jYc�A�*

lossX�=���       �	vk�jYc�A�*

loss�� >T�m=       �	��jYc�A�*

lossa�Z=B�x�       �	%��jYc�A�*

loss�n=�t{�       �	�J�jYc�A�*

loss���=6�       �	-��jYc�A�*

lossi�=x���       �	���jYc�A�*

loss��=����       �	�)�jYc�A�*

loss4�=��/�       �	��jYc�A�*

loss�@<7b�;       �	f��jYc�A�*

loss�{�<���J       �	�V�jYc�A�*

loss`��=�R�       �	y�jYc�A�*

loss׵I=��c8       �	��jYc�A�*

lossT�@=9�:t       �	�H�jYc�A�*

loss�48=�(�       �	��jYc�A�*

loss��I=Fθ�       �	h��jYc�A�*

loss���=f\�?       �	�/�jYc�A�*

loss��2>�/       �	���jYc�A�*

loss���<0Q�       �	(��jYc�A�*

loss�!h<��C       �	�&�jYc�A�*

lossQ��=V߬�       �	r��jYc�A�*

loss���=G��       �	�i�jYc�A�*

losso�>��       �	�z�jYc�A�*

loss��=a�]T       �	f�jYc�A�*

loss��t=��;�       �	��jYc�A�*

lossJ�<Lr8�       �	���jYc�A�*

loss��z<{Y�       �	�^�jYc�A�*

loss��f=_@re       �	���jYc�A�*

lossz��<�PT       �	��jYc�A�*

loss=t�У       �	.7�jYc�A�*

loss��='�c�       �	i:�jYc�A�*

loss���=�ܙ       �	V��jYc�A�*

lossy�<�=�h       �	&o�jYc�A�*

loss=$"�       �	��jYc�A�*

lossWO=�s0       �	֧�jYc�A�*

loss�� =�D       �	D�jYc�A�*

loss�L==5�l*       �	��jYc�A�*

loss��"=b0��       �	�|�jYc�A�*

loss��7=ma
�       �	8N�jYc�A�*

loss��:=W�3�       �	8�jYc�A�*

loss��<�H       �	���jYc�A�*

loss��>f�bb       �	�F�jYc�A�*

loss�PU=����       �	���jYc�A�*

loss+��=��       �		��jYc�A�*

loss�P�=����       �	T�jYc�A�*

loss�2�=Bas       �	���jYc�A�*

loss��=3dWN       �	y[�jYc�A�*

loss)>Q��L       �	~�jYc�A�*

lossܞ�=W3�       �	��jYc�A�*

losslǆ=bv)       �	�?�jYc�A�*

loss��=��[�       �	P��jYc�A�*

lossv9�=�nz       �	��jYc�A�*

lossh��=�	
       �	�,�jYc�A�*

loss��=��gE       �	���jYc�A�*

loss\�=���N       �	�m�jYc�A�*

loss�=t=&��       �	9�jYc�A�*

loss*=X�_       �	Y��jYc�A�*

loss��=գ�|       �	x�jYc�A�*

loss�Q�=Fߵ�       �	��jYc�A�*

loss�><0�)�       �	���jYc�A�*

loss%(<��۝       �	DR�jYc�A�*

lossW%-=X+[�       �	 ��jYc�A�*

loss� �<,V�       �	���jYc�A�*

lossj=�=�6N       �	�& kYc�A�*

loss,R=)�I�       �	�� kYc�A�*

losss=	Jِ       �	PVkYc�A�*

loss�V�<�zē       �	{�kYc�A�*

lossC�</��l       �	"�kYc�A�*

losskS=QH(n       �	i8kYc�A�*

loss�<����       �	��kYc�A�*

loss�֘=��=       �	�skYc�A�*

loss�9-="H�       �	�@kYc�A�*

loss�׌<E|k�       �	l#kYc�A�*

lossi�'=̗7�       �	��kYc�A�*

loss���=�ك�       �	��kYc�A�*

loss �=V��       �	��kYc�A�*

loss�W�=|�V       �	E	kYc�A�*

loss�l�=� �       �	��	kYc�A�*

loss���=����       �	hkYc�A�*

lossP=���       �	��kYc�A�*

loss2Ɇ=p.�       �	vkYc�A�*

loss�@=���       �	QkYc�A�*

losstOh=�=��       �	O�kYc�A�*

loss��m=�0��       �	EkYc�A�*

loss��=J�-�       �	��kYc�A�*

loss��@<%��0       �	�ykYc�A�*

loss��>Z{yV       �	�kYc�A�*

loss�Լ<�x<�       �	��kYc�A�*

loss��k=��       �	�ZkYc�A�*

loss��<9 �Z       �	��kYc�A�*

lossL=m�To       �	�kYc�A�*

loss��,=�hm       �	 :kYc�A�*

lossc��<TL@�       �	-�kYc�A�*

loss="�<6!E)       �	ӃkYc�A�*

loss7b =��[       �	>#kYc�A�*

lossv*\=���b       �	��kYc�A�*

loss��,=�7�       �	�TkYc�A�*

loss�=-=6DN�       �	��kYc�A�*

loss�=�&�o       �	��kYc�A�*

loss�]=�J)�       �	6tkYc�A�*

loss��>]NXp       �	IkYc�A�*

lossխ�=�wݤ       �	5�kYc�A�*

loss�V=W�Ǝ       �	7SkYc�A�*

loss�=˨       �	k�kYc�A�*

lossQk�=?��(       �	�kYc�A�*

lossk�=����       �	�]kYc�A�*

loss��=�@ː       �	PkYc�A�*

loss�D�=�ܺ       �	��kYc�A�*

loss@D�=vtn�       �	�pkYc�A�*

loss�q�<�1>�       �	�kYc�A�*

loss>9=� Z       �	��kYc�A�*

loss.�<<�-ϝ       �	� kYc�A�*

loss�!�=��       �	�X!kYc�A�*

lossf�>��%�       �	��!kYc�A�*

loss��?=��d       �	ٙ"kYc�A�*

lossEV=��-�       �	�>#kYc�A�*

lossWN="9��       �	��#kYc�A�*

loss��=�1       �	�$kYc�A�*

loss�$<��^�       �	�%kYc�A�*

loss��O<�ݭ�       �	��%kYc�A�*

loss*��<�ߑ       �	[&kYc�A�*

lossmgw=oZk       �	v�&kYc�A�*

loss�2.=xӥ       �	0�'kYc�A�*

lossY> �گ       �	C:(kYc�A�*

loss	T�=�S�       �	��(kYc�A�*

loss�D=��׳       �	�k)kYc�A�*

lossXa=��Y       �	:*kYc�A�*

lossa?�<��`�       �	N�*kYc�A�*

lossS�5=��O       �	�9+kYc�A�*

loss&�=zv��       �	��+kYc�A�*

lossƙ�<W=#v       �	@m,kYc�A�*

loss�=�pM       �	
-kYc�A�*

loss_<=K�v       �	>�-kYc�A�*

loss�A�=�@��       �	�M.kYc�A�*

loss|�=#W,       �	�.kYc�A�*

loss!a�<�߀�       �	#�/kYc�A�*

loss-b�=�EL       �	B#0kYc�A�*

lossAǔ=�\��       �	e�0kYc�A�*

loss�o=3k�,       �	b1kYc�A�*

loss�'=����       �	y2kYc�A�*

loss�u�=�!a       �	��2kYc�A�*

loss/�=;M6�       �	�G3kYc�A�*

loss��=�R       �	��3kYc�A�*

loss/�b=d���       �	��4kYc�A�*

loss���<��ɪ       �	�65kYc�A�*

loss��>��WU       �	[�5kYc�A�*

loss�a�<���k       �	x6kYc�A�*

loss���<&�       �	j7kYc�A�*

loss�޽< 8�       �	��7kYc�A�*

loss'�=$��       �	-[8kYc�A�*

loss��<:H�       �	9kYc�A�*

loss�FG=��M       �	I�9kYc�A�*

loss_=����       �	lB:kYc�A�*

loss;=�ZW�       �	��:kYc�A�*

loss��>�o�       �	s;kYc�A�*

loss���=�{��       �	N<kYc�A�*

loss|�7=��W%       �	f�<kYc�A�*

loss� R=�6>       �	�==kYc�A�*

lossi��=Z��y       �	��=kYc�A�*

loss�=e0X�       �	`v>kYc�A�*

loss�=v���       �	�?kYc�A�*

loss��<�f�       �	�?kYc�A�*

loss�y�<�0v	       �	�N@kYc�A�*

loss�^�<�S,�       �	��@kYc�A�*

lossMh�<{4u       �	��AkYc�A�*

loss�[�=��       �	�BBkYc�A�*

loss�>�=㯎�       �	a�BkYc�A�*

loss�η=5���       �	�{CkYc�A�*

loss���<{��       �	NFDkYc�A�*

loss MJ=�,       �	��DkYc�A�*

loss�7b=e4�_       �	�EkYc�A�*

lossĢ=�3�       �	EGkYc�A�*

lossV=���(       �	��GkYc�A�*

loss{��=�؎       �	kIkYc�A�*

loss�g�=7�%       �	�IkYc�A�*

loss*�'=1m��       �	RdJkYc�A�*

loss?�Q<��J=       �	�EKkYc�A�*

loss��<ߩ��       �	�6LkYc�A�*

loss$�U=Zҙ�       �	DoMkYc�A�*

loss��m<��#B       �	NkYc�A�*

loss��Y=��#[       �	ϡNkYc�A�*

loss���<�7�       �	+QOkYc�A�*

loss�2�<]�(B       �	��OkYc�A�*

loss�@�<5       �	�PkYc�A�*

loss�Xf=���'       �	�4QkYc�A�*

loss�9k=r)y�       �	��QkYc�A�*

loss!�=��<       �	�}RkYc�A�*

loss*�=j�B�       �	�$SkYc�A�*

loss��=GK�       �	�SkYc�A�*

loss�N�=�&1r       �	�YTkYc�A�*

loss��e=2�       �	v�TkYc�A�*

lossF`}=x��       �	4�UkYc�A�*

loss�%>VVJ�       �	#JVkYc�A�*

loss4�=�G       �	�VkYc�A�*

loss��=?�[C       �	�WkYc�A�*

lossi��<|��h       �	�2XkYc�A�*

loss��<�Vl�       �	k�XkYc�A�*

loss7&�=��       �	nkYkYc�A�*

loss.��=w�r�       �	�ZkYc�A�*

loss��=�ݍ-       �	��ZkYc�A�*

loss�?>=��7D       �	mX[kYc�A�*

loss=��<uwj�       �	�\kYc�A�*

loss�!�=�/�       �	��\kYc�A�*

lossљu=mw;s       �	�F]kYc�A�*

loss� >��md       �	�?_kYc�A�*

lossJk�=����       �	]�_kYc�A�*

loss!��=�M"�       �	u�`kYc�A�*

loss	R=ue�8       �	χakYc�A�*

lossmKC=C��       �	k+bkYc�A�*

loss*�Q=�r�O       �	>�bkYc�A�*

loss�!=�a�       �	g~ckYc�A�*

loss��=<'|S       �	`dkYc�A�*

loss�
�=�kB~       �	U�dkYc�A�*

loss��O=��       �	S\ekYc�A�*

loss]�-<uo�       �	jfkYc�A�*

loss_;=�R�m       �	1�fkYc�A�*

loss��=ӊ�       �	*RgkYc�A�*

loss��<Tã�       �	F�gkYc�A�*

loss�V�< ((O       �	�hkYc�A�*

loss���=�D�z       �	k'ikYc�A�*

loss_3�=�h��       �	��ikYc�A�*

lossD0�=6E�D       �	߇jkYc�A�*

loss���=6�+L       �	�#kkYc�A�*

loss���<?��       �	�kkYc�A�*

loss�W=��       �	�XlkYc�A�*

loss��=Mo       �	�lkYc�A�*

loss��=�">       �	��mkYc�A�*

loss&�)=)H8?       �	2ZnkYc�A�*

loss��=���       �	��nkYc�A�*

loss.��=�xD�       �	%�okYc�A�*

loss��B<$I�"       �	�0pkYc�A�*

loss�"�<��*z       �	��pkYc�A�*

loss��3=� w�       �	КqkYc�A�*

loss�e(>�@R�       �	5�rkYc�A�*

loss)I�=Jz��       �	�JskYc�A�*

lossP��=8w�       �	��skYc�A�*

loss4�!=ԓ0�       �	��tkYc�A�*

lossX�<6<C       �	�ukYc�A�*

loss�=Pr�       �	�ukYc�A�*

loss�!=bM�       �	�RvkYc�A�*

loss��V<'K�       �	V�vkYc�A�*

lossA%p<�煃       �		�wkYc�A�*

loss-�=�-1!       �	^,xkYc�A�*

lossE?�=���       �	��xkYc�A�*

loss@�2=���       �	�iykYc�A�*

loss*
=����       �	�	zkYc�A�*

lossnD=��       �	�zkYc�A�*

loss�=�       �	�c{kYc�A�*

lossT%9<Rӄ�       �	pw|kYc�A�*

lossJ�X=%0�|       �	�"}kYc�A�*

lossh�<F�A#       �	*�}kYc�A�*

loss�=��       �	i~kYc�A�*

loss��e=�͓`       �	{kYc�A�*

lossSj�<צU�       �	F�kYc�A�*

loss:A�;�w�P       �	�T�kYc�A�*

loss� �<\^�       �	u�kYc�A�*

loss�g�<1�t�       �	��kYc�A�*

loss�[�<�9X�       �	B�kYc�A�*

loss0�>��+Z       �	d�kYc�A�*

lossz�9>��ڋ       �	'��kYc�A�*

loss�Z�<W@dx       �	�%�kYc�A�*

lossF�g=0���       �	
��kYc�A�*

loss:�Y=���`       �	�V�kYc�A�*

loss�5m=��"       �	)�kYc�A�*

lossп<w E        �	��kYc�A�*

loss�	=.�ב       �	C�kYc�A�*

loss�X�=�Pk�       �	й�kYc�A�*

loss�c=����       �	vQ�kYc�A�*

loss 6�=JӸ       �	S�kYc�A�*

loss�4=�"p�       �	�kYc�A�*

loss���=i默       �	�/�kYc�A�*

loss�N<���       �	ɊkYc�A�*

loss�V�<�U�       �	N`�kYc�A�*

loss���<���i       �	���kYc�A�*

lossX��<n�\g       �	���kYc�A�*

loss[�=}�       �	/�kYc�A�*

loss��=MųC       �	�ȍkYc�A�*

loss2�<f���       �	~n�kYc�A�*

loss@��<��ʓ       �	�kYc�A�*

loss[,P=@׊       �	���kYc�A�*

loss�#<{��8       �	C�kYc�A�*

loss�;=Ł�F       �	ېkYc�A�*

loss\w=�G��       �	�u�kYc�A�*

lossC�=�fV       �		S�kYc�A�*

loss�B<8e�       �	^�kYc�A�*

losso��<���=       �	͐�kYc�A�*

loss�y�=��\�       �	9&�kYc�A�*

loss�	�= ��       �	�ǔkYc�A�*

loss�2�==n��       �	w�kYc�A�*

loss� �<җ�Q       �	�2�kYc�A�*

loss��|=]#       �	�ޖkYc�A�*

loss#s�<A�       �	6��kYc�A�*

losst� >[��       �	:�kYc�A�*

lossnY�=�B�       �	W!�kYc�A�*

loss3��=v�ԋ       �	[͙kYc�A�*

loss���<
*       �	�x�kYc�A�*

lossi�<|�{�       �	�%�kYc�A�*

lossw}-=����       �	�؛kYc�A�*

loss�%�=�       �	=��kYc�A�*

loss�N7=^�|�       �	9��kYc�A�*

loss�AP=X��       �	���kYc�A�*

loss� �;��45       �	g|�kYc�A�*

loss�=�[��       �	%!�kYc�A�*

loss6p�<J�       �	W�kYc�A�*

lossE8%=��       �	��kYc�A�*

losshoA=(X�       �	3N�kYc�A�*

loss1@�=���F       �	��kYc�A�*

loss�"�=2�q�       �	ѓ�kYc�A�*

loss�0�<��
       �	�3�kYc�A�*

loss��
<�d��       �	�ѤkYc�A�*

loss�#Q=�S�       �	�l�kYc�A�*

loss�r�<Q��       �	
�kYc�A�*

loss�L�;#o�       �	��kYc�A�*

loss=�><�G��       �	���kYc�A�*

loss!Ug;��a       �	#�kYc�A�*

loss���;-)V�       �	èkYc�A�*

lossؾ�<-�È       �	a�kYc�A�*

loss�{w=�0;n       �	���kYc�A�*

lossoM�=�c��       �	���kYc�A�*

loss��%;u �        �	�H�kYc�A�*

loss�;[�}6       �	�kYc�A�*

loss�%�;�-��       �	=��kYc�A�*

loss��<=�1m       �	���kYc�A�*

loss���=J#��       �	kE�kYc�A�*

loss�=�7Q       �	�kYc�A�*

loss�<��R       �	9��kYc�A�*

loss�\E<�i�       �	TR�kYc�A�*

loss�Υ>��V       �	��kYc�A�*

loss�d�<;���       �	���kYc�A�*

loss�7�;���       �	�2�kYc�A�*

loss��=0��       �	b��kYc�A�*

loss/�o=��Q�       �	���kYc�A�*

loss��=���0       �	?6�kYc�A�*

loss��b<br��       �	δkYc�A�*

lossK�
><��]       �	�i�kYc�A�*

loss.�=��*       �	q �kYc�A�*

loss8��<	ȉ3       �	�նkYc�A�*

lossM=��t       �	�v�kYc�A�*

loss�c=�5D�       �	�kYc�A�*

lossz�=E@��       �	.��kYc�A�*

loss��=cS       �	(C�kYc�A�*

lossm�=����       �	�ݹkYc�A�*

loss�c�=*��       �	���kYc�A�*

loss�ژ=�z��       �	>Z�kYc�A�*

lossȁ4=���       �	U��kYc�A�*

lossZd=�ʟ       �	l��kYc�A�*

lossl��=�_Wt       �	/4�kYc�A�*

loss�r=��%       �	�ʽkYc�A�*

loss�=�A
       �	�g�kYc�A�*

loss�S=�~%       �	*�kYc�A�*

loss҇�<G!r3       �	��kYc�A�*

loss��G<�y�       �	HR�kYc�A�*

loss1�=���       �	��kYc�A�*

loss���<kz8�       �	G��kYc�A�*

lossf�<�|��       �	���kYc�A�*

loss�}�<N�       �	�5�kYc�A�*

loss���=��       �	��kYc�A�*

loss��=�PZ       �	�o�kYc�A�*

loss��<vX�       �	�kYc�A�*

loss.�=��       �	��kYc�A�*

loss*��<9v�}       �	�W�kYc�A�*

loss�U�;v���       �	l�kYc�A�*

loss�==E:       �	M��kYc�A�*

loss��	=zގW       �	�@�kYc�A�*

loss�$l<bz&       �	/��kYc�A�*

loss[qr=:o��       �	�x�kYc�A�*

loss剑=ir�       �	� �kYc�A�*

loss֌�==�       �	\��kYc�A�*

loss�gV;�@��       �	B��kYc�A�*

lossOH=�<O�       �	�^�kYc�A�*

lossv"=�M7N       �	��kYc�A�*

lossȴ�=��4�       �	c��kYc�A�*

lossf��<#�2       �	&5�kYc�A�*

lossF��=
X��       �	r��kYc�A�*

loss��a=��a�       �	=~�kYc�A�*

loss�.=�Y�;       �	)!�kYc�A�*

lossJ,�=W�,�       �	��kYc�A�*

loss�m�<'!'       �	�X�kYc�A�*

loss�GH<��'       �	� �kYc�A�*

loss-I=Z@       �	4H�kYc�A�*

loss|�=
'gB       �	M��kYc�A�*

loss_n�==��$       �	:v�kYc�A�*

lossL��=�#�       �	ˆ�kYc�A�*

lossQd]=�gF~       �	�!�kYc�A�*

lossM1�<>�       �	D��kYc�A�*

loss&�=�|�       �	zS�kYc�A�*

loss���<3���       �	��kYc�A�*

loss3�>=���       �	���kYc�A�*

losst�=�$�       �	z�kYc�A�*

loss7��<�o�A       �	_��kYc�A�*

lossh��=GF�H       �	�J�kYc�A�*

loss(�=�Jm�       �	���kYc�A�*

loss�N�=Vՠg       �	�z�kYc�A�*

loss�=����       �	��kYc�A�*

lossRi�<_"�j       �	���kYc�A�*

loss�Z<x�?       �	�K�kYc�A�*

loss�7=�/�       �	���kYc�A�*

loss|�=9|�        �	 ~�kYc�A�*

loss���=$)��       �	n�kYc�A�*

lossm6=#E�       �	Ѱ�kYc�A�*

lossW�=�{�S       �	�M�kYc�A�*

loss�X=��o�       �	��kYc�A�*

loss�=r=@XR       �	֋�kYc�A�*

loss�>=�ά       �	B&�kYc�A�*

loss�F�<���       �	}��kYc�A�*

loss��N=��p�       �	F}�kYc�A�*

losseY�<��f�       �	��kYc�A�*

lossw=7/�       �	���kYc�A�*

loss:=��b�       �	ۆ�kYc�A�*

loss���<֤?       �	�&�kYc�A�*

loss�P|=S�5)       �	�+�kYc�A�*

loss{��<��A       �	���kYc�A�*

lossݾ=���o       �	�c�kYc�A�*

loss�1=�.�       �	c lYc�A�*

loss8�=�J<�       �	T� lYc�A�*

lossb�;=;��       �	jlYc�A�*

lossn.=h�2�       �	�lYc�A�*

loss; �=�r�       �	��lYc�A�*

loss�M=��       �	�OlYc�A�*

lossJ��=��ֱ       �	��lYc�A�*

loss�6�;.�       �	{�lYc�A�*

loss�v=���       �	�*lYc�A�*

loss�R
=���%       �	�lYc�A�*

lossj�t=/�7�       �	sLlYc�A�*

loss��t=��#}       �	=IlYc�A�*

loss��,=���l       �	��lYc�A�*

loss. =�%�       �	ѭ	lYc�A�*

loss;z8=�uJC       �	�b
lYc�A�*

loss�~ <�C       �	�lYc�A�*

loss�t<8@'       �	a�lYc�A�*

lossA��=}Υ[       �	 �lYc�A�*

loss�V�<�`       �	��lYc�A�*

loss`Y>��3�       �	�9lYc�A�*

loss8�<�0�       �	p�lYc�A�*

loss��,;�2       �	�lYc�A�*

loss��;@R�C       �	�NlYc�A�*

loss2��<=��       �	<�lYc�A�*

loss�	='.       �	��lYc�A�*

loss���=C+Y       �	�PlYc�A�*

loss1��=�rS       �	]�lYc�A�*

loss6�~=PX�d       �	z�lYc�A�*

lossT�<͠x       �	��lYc�A�*

loss��<�C$       �	-�lYc�A�*

loss:[&<�͎       �	�JlYc�A�*

loss���<j1B       �	��lYc�A�*

loss�2=����       �	
�lYc�A�*

loss�'=B��       �	y\lYc�A�*

loss6��=���+       �	�"lYc�A�*

loss*;=_͞�       �	��lYc�A�*

lossm=��7       �	�UlYc�A�*

loss�a�=���       �	|�lYc�A�*

lossj1=       �	2�lYc�A�*

lossQ)=���       �	&4lYc�A�*

loss%=ҹX       �	��lYc�A�*

loss��<*�ï       �	�tlYc�A�*

loss�ޖ=�)5       �	�lYc�A�*

lossr�~=В       �	]�lYc�A�*

lossWL.=��       �	= lYc�A�*

loss�g�<^qK       �	�� lYc�A�*

lossCd2=T0$�       �	wi!lYc�A�*

loss與=�1�~       �	4"lYc�A�*

loss���<&�b�       �	&�"lYc�A�*

loss��W<�P��       �	RG#lYc�A�*

loss�D,=�DF       �	��#lYc�A�*

loss�~<����       �	�s$lYc�A�*

loss�u'=��h�       �	�%lYc�A�*

loss���<��g       �	]�%lYc�A�*

loss�Y=ݫW�       �	P&lYc�A�*

loss���<�M��       �	ga'lYc�A�*

loss{s=D~��       �	U�'lYc�A�*

loss ��<l!       �	��(lYc�A�*

loss��@=��gX       �	�&)lYc�A�*

lossC�'=dz��       �	�)lYc�A�*

loss�p"=G��        �	�U*lYc�A�*

loss{E<_��       �	��*lYc�A�*

lossS�*=�4�       �	�+lYc�A�*

loss;�D=9QN�       �	�",lYc�A�*

lossR��=����       �	b�,lYc�A�*

loss��2=S[J�       �	�W-lYc�A�*

loss!�=�>|       �	_�-lYc�A�*

loss�E=>�O       �	n�.lYc�A�*

loss��=�I�       �	�/lYc�A�*

loss��<��B       �	��/lYc�A�*

loss��<C.�f       �	bL0lYc�A�*

loss���=`�5�       �	�0lYc�A�*

loss���<2���       �	f�1lYc�A�*

loss��<�!�       �	O$2lYc�A�*

loss8>�= �       �	G�2lYc�A�*

loss�ޠ<��)       �	�e3lYc�A�*

loss�A=f���       �	B4lYc�A�*

loss�`�<�vٜ       �	�4lYc�A�*

loss:��=]�#|       �	�E5lYc�A�*

loss�?l<��H       �	��5lYc�A�*

loss:`=}��       �	?�6lYc�A�*

loss_�T=R7��       �	ʍ7lYc�A�*

lossX�<�ԑ�       �	,8lYc�A�*

loss�=���       �	��8lYc�A�*

lossd��=��x&       �	�[9lYc�A�*

loss�5�<��>       �	$�9lYc�A�*

loss�6�<�]�P       �	��:lYc�A�*

loss��+=YxU�       �	JA;lYc�A�*

lossS��= O��       �	-�;lYc�A�*

lossA5V=�ƍ�       �	�<lYc�A�*

losszq*=��i�       �	g+=lYc�A�*

losss�Y=ô�/       �	Y�=lYc�A�*

lossO�?=|R��       �	[`>lYc�A�*

loss�=<���       �	\�>lYc�A�*

lossj��=o<q&       �	��?lYc�A�*

loss|ڛ=뙭c       �	.@lYc�A�*

loss�>=F�       �	*�@lYc�A�*

lossA'=)2�       �	�`AlYc�A�*

loss��=+h>�       �	eBlYc�A�*

loss&��;�HS�       �	��BlYc�A�*

lossxs|=����       �	/PClYc�A�*

loss�%�<hj^       �	��ClYc�A�*

lossÒ�<q;�       �	��DlYc�A�*

loss���<��'       �	�ElYc�A�*

loss>=�L�       �	��ElYc�A�*

loss��d<��       �	+iFlYc�A�*

loss2�j<;�"\       �	�nGlYc�A�*

loss�=���       �	��HlYc�A�*

loss�3*=5W��       �	O�IlYc�A�*

lossH��=y4��       �	�JlYc�A�*

lossv�G=h��@       �	�;KlYc�A�*

loss���<9a�       �	ALlYc�A�*

lossc�=� g2       �	��LlYc�A�*

loss��;���T       �	�"NlYc�A�*

loss���<f��       �	��OlYc�A�*

lossl�%=&v8�       �	EfPlYc�A�*

loss�i�<k��       �	#-QlYc�A�*

loss�=Xx�.       �	�QlYc�A�*

loss-��=��       �	��RlYc�A�*

loss{=�D��       �	.WSlYc�A�*

loss�a=���       �	�SlYc�A�*

lossÀe<f�R�       �	ӟTlYc�A�*

loss�=��)N       �	�NUlYc�A�*

lossa�E='�c       �	)�UlYc�A�*

loss,��==�O�       �	�VlYc�A�*

lossj�`<u5�@       �	�.WlYc�A�*

loss��c=�T�       �	��WlYc�A�*

lossm�="��!       �	.uXlYc�A�*

loss��/=yL��       �	H5YlYc�A�*

lossV�	;�6�       �	��YlYc�A�*

loss<XS=��i�       �	�~ZlYc�A�*

loss
;�=b�n�       �	4.[lYc�A�*

loss��<�L�}       �	:�[lYc�A�*

lossnL�=�|       �	bf\lYc�A�*

loss"�#=-��       �	�]lYc�A�*

loss�Q�<��?       �	7�]lYc�A�*

loss�Rf=mh�       �	>^lYc�A�*

loss���;��.       �	��^lYc�A�*

lossI�=ߍ��       �	E�_lYc�A�*

lossO�/=ꂝ9       �	?T`lYc�A�*

loss�\�;�7�       �	)^alYc�A�*

loss�QG=1�XE       �	�blYc�A�*

loss��=�r��       �	��blYc�A�*

loss)�[=\cV       �	�XclYc�A�*

lossi��<=��       �	��clYc�A�*

loss�A�<9�       �	؝dlYc�A�*

loss�&�<J��       �	MelYc�A�*

loss��M=��S�       �	p�elYc�A�*

loss3N�=�v�4       �	'�flYc�A�*

loss��<0tp�       �	�PglYc�A�*

loss��<:� �       �	-hlYc�A�*

loss}��<���       �	�hlYc�A�*

loss�=��r�       �	�vilYc�A�*

lossED?=�]��       �	!jlYc�A�*

lossIE"=��y       �	]klYc�A�*

loss�w="�JK       �	r�klYc�A�*

loss&	>��r�       �	hllYc�A�*

loss�Y==���I       �	�mlYc�A�*

loss	E�<諽       �	%�mlYc�A�*

lossib�<�਍       �	�HnlYc�A�*

loss�Ϙ<i�|       �	��nlYc�A�*

loss���<)�       �	/�olYc�A�*

losss�=�ھw       �	)plYc�A�*

loss���<�e��       �	 �plYc�A�*

loss1��=#^.�       �	-_qlYc�A�*

lossT��<݆�&       �	��qlYc�A�*

loss��5=3c3g       �	6�rlYc�A�*

loss�)=I�}�       �	�WslYc�A�*

loss( �<L1��       �	'�slYc�A�*

loss�|{==��       �	��tlYc�A�*

loss��<f˶)       �	�=ulYc�A�*

loss䁉=���<       �	��ulYc�A�*

loss�Z=��       �	�xvlYc�A�*

loss��='(_�       �	�wlYc�A�*

loss<�\=�8�<       �	6�wlYc�A�*

lossH�=y*�       �	�SxlYc�A�*

loss�=���s       �	�xlYc�A�*

loss���<7���       �	-�ylYc�A�*

loss���<���       �	�ZzlYc�A�*

loss��=�]��       �	^L{lYc�A�*

loss&-z=?�W�       �	%|lYc�A�*

loss�(>�       �	�|lYc�A�*

loss���<T	�       �	�L}lYc�A�*

loss�'�=�T(       �	�}lYc�A�*

loss��={��       �	�~lYc�A�*

loss�É=�	�;       �	/lYc�A�*

loss,tE<�-2       �	`�lYc�A�*

lossV��<ݺ��       �	�j�lYc�A�*

lossS�m=�?��       �	$	�lYc�A�*

loss�B�=�V��       �	���lYc�A�*

loss��3=&~*       �	�<�lYc�A�*

lossJ[�=��K       �	Ad�lYc�A�*

loss�&�=#��       �	��lYc�A�*

loss�V�=�z�0       �	�τlYc�A�*

loss�S�<��k�       �	�o�lYc�A�*

lossS��=�`�7       �	�A�lYc�A�*

loss�G*<q�W|       �	|�lYc�A�*

loss��x<gt��       �	��lYc�A�*

loss���=���       �	ˈlYc�A�*

loss�U=���       �	3l�lYc�A�*

loss|�1=�a�       �	L�lYc�A�*

loss��q=/Kq       �	*�lYc�A�*

lossQ��<�A       �	��lYc�A�*

lossQ��=��P~       �	ӟ�lYc�A�*

loss��=�˴
       �	�>�lYc�A�*

loss͑:=���       �	o�lYc�A�*

lossE�'=��Z       �	���lYc�A�*

loss�=���p       �	uY�lYc�A�*

loss?]�<��`       �	��lYc�A�*

loss��=�\�       �	y��lYc�A�*

loss��=Z�Ak       �	PQ�lYc�A�*

loss��=�Ud�       �	��lYc�A�*

losseR�<��D�       �	{��lYc�A�*

lossDw�<�KpZ       �	���lYc�A�*

loss^�=|���       �	�-�lYc�A�*

loss,p�<cF�k       �	�˕lYc�A�*

loss�T=C�|       �	i�lYc�A�*

loss:i�=~Y�h       �	�h�lYc�A�*

losse	A=���x       �	\�lYc�A�*

loss���<�N#�       �	k��lYc�A�*

loss��,<�c�e       �	�7�lYc�A�*

loss��<$;�       �	VҙlYc�A�*

loss=c���       �	�o�lYc�A�*

loss�D=��X       �	6�lYc�A�*

lossq�<Z�!       �	$��lYc�A�*

loss�<�=�)^       �	�J�lYc�A�*

loss{��<��6�       �	��lYc�A�*

loss4v=��D�       �	���lYc�A�*

lossj>bP30       �	�+�lYc�A�*

lossq%�<Yn_�       �	ŞlYc�A�*

loss�V=Y �       �	�`�lYc�A�*

loss�(><㛩       �	f��lYc�A�*

loss�Ɉ=GZ}�       �	���lYc�A�*

losse0�={K�       �	Ih�lYc�A�*

lossh�=���;       �	���lYc�A�*

lossߧ�=F�"�       �	N��lYc�A�*

loss��1< S#Y       �	�-�lYc�A�*

loss$�<����       �	�ǣlYc�A�*

loss�*�<�tŵ       �	Ac�lYc�A�*

loss�� =����       �	�
�lYc�A�*

loss-Fl=s��       �	�B�lYc�A�*

loss]��=_���       �	צlYc�A�*

losse�t=!��       �	n�lYc�A�*

lossj�<m2�C       �	O�lYc�A�*

lossNk�=}���       �	Z��lYc�A�*

loss�[=�]1�       �	�7�lYc�A�*

loss�S�;�D��       �	��lYc�A�*

lossΑ=J�pQ       �	I��lYc�A�*

loss��=>�b�       �	��lYc�A�*

lossw�<K ��       �	[��lYc�A�*

loss���=h��%       �	�M�lYc�A�*

loss!�=�       �	"�lYc�A�*

loss���=�K�       �	5z�lYc�A�*

loss��
=[
�       �	�9�lYc�A�*

lossJ��<@/a       �	bڮlYc�A�*

loss�]<ܟ�       �	���lYc�A�*

loss�@<��t�       �	�C�lYc�A�*

loss�;=�˳�       �	0�lYc�A�*

loss\F�<+��       �	囱lYc�A�*

lossqݱ<o^�~       �	�8�lYc�A�*

lossD�}=7>V       �	�ܲlYc�A�*

losst��=�]f       �	~�lYc�A�*

loss� =0�ԑ       �	`�lYc�A�*

lossL��=���F       �	�´lYc�A�*

loss}�x=v(,�       �	�i�lYc�A�*

loss�$�=����       �	��lYc�A�*

loss��< ���       �	���lYc�A�*

loss)�=�8       �	ZJ�lYc�A�*

losst�=�Xǡ       �	|�lYc�A�*

loss���=���       �	���lYc�A�*

lossV�<!:(2       �	�;�lYc�A�*

loss2��=	�)U       �	�޹lYc�A�*

loss��<�Y:x       �	���lYc�A�*

loss�z�=�~b�       �	��lYc�A�*

lossC��<�4��       �	佻lYc�A�*

loss3��<z�<�       �	S^�lYc�A�*

loss��g=b���       �	�lYc�A�*

lossZS$=�聺       �	V��lYc�A�*

loss<�<�l       �	.7�lYc�A�*

lossc�<~��H       �	
۾lYc�A�*

loss#�<�#��       �	W{�lYc�A�*

loss�n<=�/Ļ       �	m �lYc�A�*

loss��<.�'{       �	Y��lYc�A�*

loss���=��=       �	�c�lYc�A�*

loss8 �<�~-.       �	�
�lYc�A�*

loss�=����       �	���lYc�A�*

loss��<���       �	K�lYc�A�*

loss߹�=��>       �	���lYc�A�*

loss�6�=eI �       �	���lYc�A�*

lossC_$=l��       �	��lYc�A�*

loss��<��b       �	�G�lYc�A�*

loss�{a=��ԅ       �	[	�lYc�A�*

loss��!<�^       �	��lYc�A�*

loss�x�<J�l       �	���lYc�A�*

lossC�k=J�       �	���lYc�A�*

loss�>J=��A       �	�/�lYc�A�*

lossn��<�Ђ)       �	]n�lYc�A�*

loss��<*�4       �	,.�lYc�A�*

loss���;E�;9       �	���lYc�A�*

loss��<΢�       �	��lYc�A�*

loss��=?��       �	˼�lYc�A�*

loss�E=��!;       �	�b�lYc�A�*

lossH��<G���       �	ܝ�lYc�A�*

loss[,=Y=w       �	WA�lYc�A�*

lossS�s=�m�x       �	'��lYc�A�*

loss$��<���d       �	{��lYc�A�*

lossҝD<�e       �	��lYc�A�*

loss{��<�\��       �	�=�lYc�A�*

loss(�c=�ѕ3       �	�;�lYc�A�*

loss�)�<���"       �	'��lYc�A�*

loss�'>���       �	���lYc�A�*

lossa�m=0Ƴ�       �	���lYc�A�*

loss!�<y��       �	pD�lYc�A�*

loss�<~=�jB�       �	Sy�lYc�A�*

loss��=��j�       �	��lYc�A�*

lossA�=�b�W       �	�8�lYc�A�*

loss��;j���       �	���lYc�A�*

loss<�X=Uh4>       �	���lYc�A�*

lossf(M=����       �	QO�lYc�A�*

loss�j�<_��       �	���lYc�A�*

loss�=��5�       �	��lYc�A�*

loss���<mt��       �	'J�lYc�A�*

loss�kI<.��       �	P��lYc�A�*

loss.X�<e3\�       �	S��lYc�A�*

loss-��=2���       �	�P�lYc�A�*

loss��<t��       �	��lYc�A�*

losslӆ="�z�       �	e��lYc�A�*

loss��=�*�^       �	�(�lYc�A�*

loss��=Ƥ��       �	���lYc�A�*

loss*��=���J       �	�{�lYc�A�*

lossi`�=�K��       �	��lYc�A�*

lossd��<,Ի�       �	k��lYc�A�*

loss �=�u%J       �	���lYc�A�*

loss:�<���       �	yx�lYc�A�*

loss=�<���x       �	a�lYc�A�*

loss"=�3       �	���lYc�A�*

losshކ=gԨ       �	�W�lYc�A�*

loss�9]=�J��       �	���lYc�A�*

loss���=L��       �	1��lYc�A�*

loss�0X=�]L       �	:�lYc�A�*

lossZ�<X !       �	���lYc�A�*

lossì�=��       �	x�lYc�A�*

lossƌ\={Wћ       �	�lYc�A�*

loss�p,=sc9b       �	���lYc�A�*

loss��,=�Lj�       �	�F�lYc�A�*

loss���=�5�       �	+��lYc�A�*

lossX7=�8       �	ς�lYc�A�*

loss\v`=am	(       �	T�lYc�A�*

loss��<$,�h       �	��lYc�A�*

loss��j<��=       �	Zc�lYc�A�*

loss��<�F~       �	U��lYc�A�*

lossDX�<)�2�       �	ӡ�lYc�A�*

lossȘf=��,�       �	�=�lYc�A�*

loss���<�0
       �	���lYc�A�*

loss�p�;r��F       �	�z�lYc�A�*

loss�ѝ=�F�j       �	Y�lYc�A�*

loss/�g=҉       �	��lYc�A�*

loss�\v<JOC�       �	_D�lYc�A�*

loss)��<m~	!       �	���lYc�A�*

losstKX=�X��       �	G��lYc�A�*

lossv=�t       �	�)�lYc�A�*

lossI��=��.       �	���lYc�A�*

lossjH-=���       �	F|�lYc�A�*

loss���<Oҷ�       �	8N�lYc�A�*

loss�}�<@iQ�       �		��lYc�A�*

loss�h=�Lʥ       �	`��lYc�A�*

loss��<<�AN8       �	�G�lYc�A�*

loss�r�<"�_       �	���lYc�A�*

loss�wU=ٕ       �	��lYc�A�*

losssC�<CI��       �	# mYc�A�*

lossNl9=���       �	� mYc�A�*

lossw�c=;^�       �	%ZmYc�A�*

lossT�=q�z�       �	�mYc�A�*

lossLH=S�       �	��mYc�A�*

lossHl=c��       �	�6mYc�A�*

loss��_=�(O�       �	�mYc�A�*

loss/�=M��t       �	ĴmYc�A�*

loss�u�=�1^�       �	O]mYc�A�*

loss��7=�wW)       �	�\mYc�A�*

loss�R�=|��:       �	�cmYc�A�*

loss(� =����       �	�YmYc�A�*

loss�kf={xΆ       �	�5	mYc�A�*

lossA�!<�+�	       �	j�	mYc�A�*

loss֣�<,Z�v       �	)�
mYc�A�*

loss��<p{�       �	�wmYc�A�*

loss���<�R��       �	CmYc�A�*

loss%%f<�T�       �	�mYc�A�*

loss�,�=�C�W       �	��mYc�A�*

loss.�/<FI'�       �	f�mYc�A�*

lossCp>���*       �	�mYc�A�*

loss�=F�@�       �	úmYc�A�*

loss�z�=T1~       �	��mYc�A�*

loss65a=��T       �	sfmYc�A�*

loss���=��X�       �	]mYc�A�*

loss6�<x W       �	�mYc�A�*

lossj=���       �	� mYc�A�*

loss�x=��Eu       �	�$mYc�A�*

losse=<	yW       �	�`mYc�A�*

loss�-I=Y��}       �	;7mYc�A�*

loss�K�<�D�       �	�@mYc�A�*

loss�P�=��x?       �	{�mYc�A�*

loss}�<k��       �	VcmYc�A�*

lossZ�<��H*       �	�(mYc�A�*

losss�Z=�7M       �	%mYc�A�*

lossC�&<nU-       �	�cmYc�A�*

losss'=�W�k       �	�mYc�A�*

lossh��=ul�g       �		TmYc�A�*

loss:E�=`�       �	 mYc�A�*

lossJ�P=N�9F       �	 � mYc�A�*

lossXP�=�iF�       �	Dk!mYc�A�*

loss{�/<�
ja       �	�
"mYc�A�*

loss�6=UU�       �	�"mYc�A�*

loss#�=��f�       �	C#mYc�A�*

lossq<��v`       �	z�#mYc�A�*

loss@h�<Djhc       �	lz$mYc�A�*

lossd�=�P�       �	�+%mYc�A�*

loss�"�=}Ѡ       �	�%mYc�A�*

lossxE<M�ۍ       �	�r&mYc�A�*

lossOP|<���q       �	�'mYc�A�*

loss�Ma=~t�       �	��'mYc�A�*

loss���=�3       �	(mYc�A�*

loss�?=�PӋ       �	�%)mYc�A�*

lossT�==�ma�       �	��)mYc�A�*

lossx�<÷<9       �	��*mYc�A�*

lossO��<���       �	t)+mYc�A�*

loss]�=���A       �	A�+mYc�A�*

loss���<��       �	��,mYc�A�*

loss��<}{��       �	�9-mYc�A�*

loss���<�j�       �	��-mYc�A�*

lossl-N=�Rm        �	�.mYc�A�*

lossXG2<�F��       �	�4/mYc�A�*

losswe=��Z�       �	e�0mYc�A�*

loss�&W=���       �	�P1mYc�A�*

loss��=�;�
       �	�2mYc�A�*

loss�=Usx       �	u�2mYc�A�*

lossۑ�;����       �	�R3mYc�A�*

loss��#=�@�`       �	�3mYc�A�*

loss�z=�%k�       �	��4mYc�A�*

loss�F+=���       �	i:5mYc�A�*

loss1>�=�s��       �	�5mYc�A�*

lossڶ�<?98�       �	=�6mYc�A�*

loss� =��8m       �	�h7mYc�A�*

loss�	=��w�       �	�8mYc�A�*

lossK��<y��L       �	{�8mYc�A�*

loss͚<�F��       �	�i9mYc�A�*

loss(F�=a�6       �	�:mYc�A�*

loss@�M>�0�       �	ͯ:mYc�A�*

loss�p=a��i       �	��;mYc�A�*

loss�H�<ή�+       �	8<mYc�A�*

loss��>{��       �	��<mYc�A�*

lossf��<nЫ�       �	��=mYc�A�*

loss&��<)��       �	d;>mYc�A�*

loss��<a���       �	K�?mYc�A�*

loss�D�=�W=�       �	�4@mYc�A�*

lossx)�<�a�$       �	��@mYc�A�*

loss��=í��       �	��AmYc�A�*

loss�0�<�� �       �	P�CmYc�A�*

loss��T=�M�N       �	�7DmYc�A�*

losso6�<ZiwG       �	~�DmYc�A�*

loss�9<
�4       �	�-FmYc�A�*

losss�<�B��       �	�
GmYc�A�*

loss��"=�+k       �	M�GmYc�A�*

loss�o�<�s�j       �	��HmYc�A�*

loss�E�<0�_�       �	M�ImYc�A�*

loss��8=~\ف       �	Y�JmYc�A�*

loss���=�L�G       �	,EKmYc�A�*

loss=�h=<�~       �	[�KmYc�A�*

loss�\�<��Jy       �	��LmYc�A�*

lossZ��;3��       �	EMmYc�A�*

loss��<y�I       �	V�MmYc�A�*

loss��?=�*�S       �	�NmYc�A�*

loss*J�<5�L�       �	�SOmYc�A�*

lossX�!=ݼ/        �	dPmYc�A�*

loss��7=·o       �	��PmYc�A�*

loss�ђ=�<=h       �	�`QmYc�A�*

lossj;V=��[       �	* RmYc�A�*

loss��<Ǝ�       �	�RmYc�A�*

loss�:�=�FB       �	�SSmYc�A�*

loss�.L;�A��       �	�SmYc�A�*

loss4��<���       �	w�TmYc�A�*

lossr؀<��)       �	�=UmYc�A�*

loss�w=���       �	�VmYc�A�*

loss�$�<�}T       �	S�VmYc�A�*

loss�N<r��8       �	IGWmYc�A�*

loss2��<�K*Y       �	BXmYc�A�*

lossi��==���       �	~�XmYc�A�*

loss�r�=�!B�       �	DYmYc�A�*

lossd�I<�$��       �	��YmYc�A�*

loss��;(VN�       �	uZmYc�A�*

loss`J=g'�       �	�
[mYc�A�*

loss:��<�T�       �	�[mYc�A�*

loss+�=��^       �	G\mYc�A�*

loss��<7$3-       �	��\mYc�A�*

loss�8�=��       �	�]mYc�A�*

loss|#�=���       �	rk^mYc�A�*

loss
�3=�y\�       �	�_mYc�A�*

loss��l=����       �	V�_mYc�A�*

loss�9�=���	       �	4`mYc�A�*

loss%��<.�O{       �	�/amYc�A�*

lossLO�=�g�       �	�amYc�A�*

loss
��<�H`
       �	�qbmYc�A�*

loss:��;��&�       �	cmYc�A�*

loss���;�-�       �	r3dmYc�A�*

lossT�=��MV       �	^�dmYc�A�*

loss�͇;���6       �	�mfmYc�A�*

loss�]�<�ٞ�       �	�gmYc�A�*

loss(ˍ;�%�~       �	��gmYc�A�*

losss�Q;d��       �	hhmYc�A�*

lossؔ�;�wc�       �	bimYc�A�*

lossEz4=��m9       �	�imYc�A�*

loss�2�=�,2O       �	|cjmYc�A�*

lossO�<�"�q       �	�	kmYc�A�*

loss��;#ře       �	P�kmYc�A�*

loss1��;�;Ƅ       �	WAmmYc�A�*

loss;�h>
��       �	onmYc�A�*

loss�J2<�X6�       �	Y�nmYc�A�*

lossF^C<��       �	b�omYc�A�*

loss��<́��       �	?8pmYc�A�*

loss[�[=�^��       �	��pmYc�A�*

lossj�<��Z�       �	�qmYc�A�*

lossx�q<�R)       �	�PrmYc�A�*

lossF�=O+,�       �	��rmYc�A�*

loss-(�=��O       �	2�smYc�A�*

loss�S=�[6+       �	qVtmYc�A�*

loss9�=nx��       �	�tmYc�A�*

loss��t=�=��       �	g�umYc�A�*

loss�n=��w       �	k`vmYc�A�*

loss���=���       �	wmYc�A�*

loss�m�<$Z/�       �	��wmYc�A�*

loss���=%R�c       �	AgxmYc�A�*

lossUs�=	�۾       �	<ymYc�A�*

lossZ�.=��E       �	ܷymYc�A�*

loss�L�<g�;�       �	�\zmYc�A�*

loss�I=���       �	~�zmYc�A�*

loss�=��t       �	_�{mYc�A�*

loss[`<H�6       �	�y|mYc�A�*

loss3t=��       �	!}mYc�A�*

loss��Y=_�       �	]�}mYc�A�*

loss$�<��J�       �	�h~mYc�A�*

loss6�<] �       �	J	mYc�A�*

lossO�7<�c�%       �	�mYc�A�*

loss|��<1f�       �	�I�mYc�A�*

loss�<cG�Y       �	��mYc�A�*

loss&#=
�>�       �	舁mYc�A�*

loss&��=L�p       �	� �mYc�A�*

lossŬ�<�$�s       �	ػ�mYc�A�*

lossȅ<��~�       �	.V�mYc�A�*

loss��<�k�       �	U/�mYc�A�*

lossF<LF-�       �	�фmYc�A�*

loss���=�C       �	�l�mYc�A�*

loss|�=g��       �	��mYc�A�*

loss��"<���       �	���mYc�A�*

loss*�=t�00       �	JF�mYc�A�*

lossw�=���|       �	��mYc�A�*

loss��=eޘ6       �	���mYc�A�*

loss�\�;��       �	�Q�mYc�A�*

loss�a=[^�x       �	��mYc�A�*

loss�`�<:�       �	���mYc�A�*

loss��<��       �	�$�mYc�A�*

loss�2/<H�b       �	���mYc�A�*

loss'b=EB��       �	hX�mYc�A�*

lossLL=ۤ��       �	�)�mYc�A�*

lossq"�;l=Y�       �	aōmYc�A�*

lossMN�<{}�       �	�h�mYc�A�*

loss��<.�;L       �	�	�mYc�A�*

loss�mo<�U��       �	r��mYc�A�*

loss��=^�&       �	�ǩmYc�A�*

loss:�=e�4^       �	���mYc�A�*

losse�>�XV       �	|E�mYc�A�*

lossZ~�=}���       �	B�mYc�A�*

loss��2=�@r�       �	K��mYc�A�*

loss�5=T�C�       �	6�mYc�A�*

loss���=�0:�       �	�ЭmYc�A�*

loss�r�<<��       �	�~�mYc�A�*

loss��m=��x�       �	�L�mYc�A�*

loss�R�=�Hò       �	���mYc�A�*

loss�ף<�        �	~��mYc�A�*

lossd�~=�Ԑ'       �	�9�mYc�A�*

loss]�#=��'       �	w�mYc�A�*

loss���=�n�       �	ø�mYc�A�*

lossp��=�-�v       �	%[�mYc�A�*

loss�ɓ=��3t       �	���mYc�A�*

loss$W�;ڲ�       �	V�mYc�A�*

lossI��<��,t       �	��mYc�A�*

loss�C=3���       �	'��mYc�A�*

loss X=���+       �	VJ�mYc�A�*

lossG=Vj
�       �	��mYc�A�*

loss�,�=p�SW       �	���mYc�A�*

loss-��;�%Z�       �	�]�mYc�A�*

loss!
`=p3�       �	���mYc�A�*

loss�A�<�&Ɖ       �	:��mYc�A�*

loss@f<.�lH       �	�=�mYc�A�*

loss��^=Fo�       �	@ܻmYc�A�*

loss�,�<��=�       �	���mYc�A�*

loss2(�=?[Z�       �	�U�mYc�A�*

loss)"=����       �	���mYc�A�*

lossc�_=���       �	���mYc�A�*

loss���<�h%`       �	�.�mYc�A�*

loss��;=^p~�       �	ǿmYc�A�*

lossB��=>.T�       �	�e�mYc�A�*

loss
��<	܀       �	�mYc�A�*

lossS{p=�D�       �	;��mYc�A�*

loss-��<\1�w       �	�D�mYc�A�*

losse'=�� �       �	G�mYc�A�*

lossn�=h=��       �	E��mYc�A�*

loss�J�=J��       �	8�mYc�A�*

lossX;=5a�       �	���mYc�A�*

loss���<l�N       �	Cs�mYc�A�*

lossZ�y<c@d�       �	��mYc�A�*

loss�h�<�=��       �	�B�mYc�A�*

loss��v<�'�K       �	���mYc�A�*

loss�%�<)}�       �	1��mYc�A�*

loss�ե<"�j?       �	�o�mYc�A�*

lossM�<�J�       �	��mYc�A�*

loss�z=���       �	�B�mYc�A�*

loss3@O;|\Ռ       �	��mYc�A�*

loss��<��K       �	���mYc�A�*

loss=v�       �	-��mYc�A�*

loss���<e��       �	���mYc�A�*

loss�� >t4��       �	
g�mYc�A�*

loss��=�'nN       �	�mYc�A�*

loss2�;�ض       �	���mYc�A�*

loss��]<��?       �	[|�mYc�A�*

loss� �;�q��       �	S{�mYc�A�*

loss�̚<3l��       �	���mYc�A�*

loss�lZ=UQ�       �	���mYc�A�*

loss�,�<��_�       �	B?�mYc�A�*

loss�'<=�X0       �	��mYc�A�*

lossX��;t/.�       �	y�mYc�A�*

loss��$<�p�F       �		�mYc�A�*

loss�Ϩ<�B��       �	��mYc�A�*

loss�D=DI�b       �	E��mYc�A�*

loss���=���       �	�,�mYc�A�*

loss�S�<���       �	v��mYc�A�*

loss��-=�K�i       �	d�mYc�A�*

loss�a!=
UrF       �	H��mYc�A�*

loss�"%=�p       �	|��mYc�A�*

loss �<�
�       �	H4�mYc�A�*

loss�bs=��g       �	>��mYc�A�*

loss$&�<���       �	g�mYc�A�*

loss0�=�	;�       �	��mYc�A�*

losst��<���       �	r��mYc�A�*

lossh��<�aY       �	l?�mYc�A�*

loss�0=D�G       �	�;�mYc�A�*

loss{��;׵!3       �	+��mYc�A�*

loss m~=b�w�       �	���mYc�A�*

loss�/=����       �	8��mYc�A�*

loss���<f�Ɯ       �	>z�mYc�A�*

loss
��<��`o       �	�&�mYc�A�*

lossv�;�K�       �	���mYc�A�*

loss���<Sl�       �	o�mYc�A�*

loss��/<6�g�       �	��mYc�A�*

loss<UY3{       �	���mYc�A�*

loss��<�$�       �	PT�mYc�A�*

loss��v=s݈8       �	��mYc�A�*

loss�T�<3��       �	��mYc�A�*

lossxd=|��J       �	UM�mYc�A�*

loss�8�<��R�       �	��mYc�A�*

loss�G<6xK`       �	֏�mYc�A�*

loss
a�<��B       �	|,�mYc�A�*

loss%H"=!��M       �	���mYc�A�*

loss�K�<d@p       �	D��mYc�A�*

loss�k%=PxN�       �	+�mYc�A�*

loss��>=߽0       �	���mYc�A�*

loss_��=x�NQ       �	U��mYc�A�*

loss�3~<�+	R       �	�3�mYc�A�*

loss��/<���d       �	��mYc�A�*

loss���<�P�.       �	7��mYc�A�*

lossFC=EL�       �	�C�mYc�A�*

loss�� <TP��       �	���mYc�A�*

loss�Hq=����       �	b��mYc�A�*

losszi)=��I�       �	��mYc�A�*

lossyd�<�0�Y       �	���mYc�A�*

loss7/y=W\�A       �	GY�mYc�A�*

loss�	@=h�x�       �	rO�mYc�A�*

lossY^ =��L       �	 ��mYc�A�*

loss���=�	a       �	H��mYc�A�*

loss�X<U�(X       �	S%�mYc�A�*

loss���<�s{       �	���mYc�A�*

loss�<��[�       �	d�mYc�A�*

loss�O=����       �	��mYc�A�*

loss%y+<m&�w       �	���mYc�A�*

loss-�=�(Š       �	8�mYc�A�*

losss A=	lQ       �	��mYc�A�*

lossi�<+�I�       �	Xr�mYc�A�*

loss/�=ʣ_       �	9�mYc�A�*

loss�~&=�m�       �	7��mYc�A�*

loss��_=���       �	�<�mYc�A�*

lossر�<�>��       �	'��mYc�A�*

loss�k[=���       �	)v nYc�A�*

loss�
<�9       �	�nYc�A�*

loss�.B=Dq�	       �	�nYc�A�*

loss�&X=v��       �	�LnYc�A�*

loss��<�ڒ�       �	D�nYc�A�*

lossW�u=��       �	�nYc�A�*

loss,n�=�)9�       �	.=nYc�A�*

lossW^o=[_+       �	w�nYc�A�*

loss���<X���       �	_�nYc�A�*

loss)q
=��Z       �	�nYc�A�*

loss_�]<��       �	W{nYc�A�*

loss3"�=��       �	�@nYc�A�*

loss`'�<p�Z       �	��nYc�A�*

loss��<=�R�       �	�	nYc�A�*

lossx�<��a       �	%
nYc�A�*

loss�P�<���       �	�
nYc�A�*

lossd]=f�{�       �	�inYc�A�*

loss��<֣�b       �	|(nYc�A�*

loss1��<��       �	�nYc�A�*

loss�~y<��،       �	2rnYc�A�*

loss�M!=<�&�       �	 nYc�A�*

losst̕=����       �	��nYc�A�*

loss��g<~�j�       �	�ZnYc�A�*

lossf[<]���       �	�nYc�A�*

loss���;G�+�       �	ްnYc�A�*

lossd�0<;�f�       �	�NnYc�A�*

lossT3�<���       �	��nYc�A�*

loss��=���L       �	/�nYc�A�*

loss�n_=��$�       �	�*nYc�A�*

loss�cm=�s�       �	&�nYc�A�*

loss��2=&_��       �	�nYc�A�*

loss�=�E;�       �	B�nYc�A�*

loss�s�: q�       �	�MnYc�A�*

loss8��<����       �	�dnYc�A�*

loss{�=���       �	�nYc�A�*

lossJ�n=�|�$       �	ެnYc�A�*

loss���<M�2d       �	�LnYc�A�*

loss<��a�       �	�nYc�A�*

loss��<d:       �	^�nYc�A�*

loss<q�       �	�EnYc�A�*

loss��:nRۮ       �	��nYc�A�*

lossm*M=�I�       �	�$nYc�A�*

loss��#<߈��       �	��nYc�A�*

loss 
�;}V;�       �	�vnYc�A�*

loss�V�<5u@j       �	�nYc�A�*

lossI#�<�I�       �	�W nYc�A�*

loss�� =l?I       �	N�!nYc�A�*

loss��`=q��       �		6"nYc�A�*

loss��=ू       �	��"nYc�A�*

lossj/=@���       �	��#nYc�A�*

loss>W�= ��       �	�&$nYc�A�*

loss��G<9b�D       �	%nYc�A�*

losss<=��{       �	N�%nYc�A�*

lossx��= �       �	�U&nYc�A�*

lossO&�=���       �	��&nYc�A�*

loss1��<-�       �	�'nYc�A�*

lossCh�<�V�       �	;(nYc�A�*

lossl�r<|L       �	��(nYc�A�*

loss�=id0�       �	+m)nYc�A�*

loss��;��;�       �	�*nYc�A�*

loss.[�<�Q��       �	�*nYc�A�*

loss�@%<U?�       �	O;+nYc�A�*

lossN�A=G���       �	��+nYc�A�*

lossmWX=s��       �	8i,nYc�A�*

loss[e�<*h)+       �	!-nYc�A�*

loss;��<;o��       �	g�-nYc�A�*

loss�Et=�+��       �	�.nYc�A�*

loss;��<��#�       �	�B/nYc�A�*

loss���<�0�p       �	��/nYc�A�*

loss�X�<��j       �	��0nYc�A�*

loss�ej<?1�       �	i1nYc�A�*

loss�<MA��       �	4�1nYc�A�*

loss6��<�T9       �	ʇ2nYc�A�*

loss��=���4       �	�'3nYc�A�*

lossO"m=,0�       �	��3nYc�A�*

loss]��=
Gm�       �	�`4nYc�A�*

loss@B�<�r�       �	��4nYc�A�*

loss�=P=4~�6       �	�5nYc�A�*

loss}�q<f�;       �	Nc6nYc�A�*

loss?c#<��R?       �	9	7nYc�A�*

lossV��<S��&       �	��7nYc�A�*

lossm֥<����       �	�D8nYc�A�*

loss�9�<��`       �	n�8nYc�A�*

loss@�r=O��       �	c}9nYc�A�*

loss�p�<hU       �	P4:nYc�A�*

loss��8=�B�       �	��:nYc�A�*

lossZ��=W	E�       �	\�;nYc�A�*

loss1��<��AS       �	�G<nYc�A�*

loss�h�<I���       �	T�<nYc�A�*

loss�W<i��       �	�=nYc�A�*

loss�<�{�U       �	� >nYc�A�*

loss�s=K��j       �	r�>nYc�A�*

loss?�=x�y       �	�}?nYc�A�*

loss�0�=�+٧       �	�@nYc�A�*

loss#�)=I��       �	A�@nYc�A�*

loss L=�a�       �	�bAnYc�A�*

loss�ߒ=z��       �	BnYc�A�*

loss���<�T�D       �	֧BnYc�A�*

loss��$=����       �	�XCnYc�A�*

loss��<��f       �	�CnYc�A�*

loss��y=��Q�       �	�DnYc�A�*

loss�{=�U�       �	b.EnYc�A�*

loss��=!m�       �	F�EnYc�A�*

loss,��=�}��       �	��FnYc�A�*

loss���<���       �	�'GnYc�A�*

lossH0�<��9K       �	F�GnYc�A�*

loss�=<���       �	��HnYc�A�*

loss��;�zN]       �	�@InYc�A�*

loss���<I�C�       �	��InYc�A�*

loss��I=�,�B       �	еJnYc�A�*

lossw�<ﴠ       �	mSKnYc�A�*

loss꨸=x��       �	��KnYc�A�*

loss�I=(�       �	r�LnYc�A�*

lossU�=�=�       �	�zMnYc�A�*

lossC�<7i�       �	{NnYc�A�*

loss+�=M�8,       �	��OnYc�A�*

loss��u<�}n�       �	�1PnYc�A�*

loss��
<?�W@       �	�QnYc�A�*

loss|�0=�:TI       �	 RnYc�A�*

loss2,=H��f       �	��RnYc�A�*

loss2��<�$]�       �	R�SnYc�A�*

loss��v=��S�       �	�mTnYc�A�*

loss;^4=��7�       �	�aUnYc�A�*

lossڍ!=kʅO       �	g+VnYc�A�*

loss��<Ll�       �	�AWnYc�A�*

loss�O=�2�t       �	��WnYc�A�*

loss{̣<1�^       �	W�XnYc�A�*

loss�$=��p�       �	 oYnYc�A�*

loss�N�=���       �	��ZnYc�A�*

loss���=?I�N       �	��[nYc�A�*

loss�f�;�M��       �	��\nYc�A�*

loss��<��*       �	�j]nYc�A�*

lossCS�<�:�       �	G^nYc�A�*

loss��<�z�4       �	ܜ^nYc�A�*

loss���<_��       �	'N_nYc�A�*

loss\q5=j��       �	��_nYc�A�*

lossfF�<����       �	 �anYc�A�*

loss9=�)�       �	��bnYc�A�*

loss���<���       �	hcnYc�A�*

lossq��=8��       �	�dnYc�A�*

lossq=��i�       �	��dnYc�A�*

loss/��=�iD       �	�;enYc�A�*

loss���<�ڴ       �	��enYc�A�*

loss-=�=�Q�u       �	xfnYc�A�*

loss��=��$�       �	gnYc�A�*

loss���=x]       �	u�gnYc�A�*

loss�OM=0�"       �	irhnYc�A�*

loss-Ƌ<�\��       �	�inYc�A�*

lossr+�<W��v       �	�inYc�A�*

loss��t=�q��       �	�@jnYc�A�*

loss�A�<��%       �	��jnYc�A�*

lossT��=}���       �	�rknYc�A�*

lossF��=%'��       �	�KlnYc�A�*

loss���<�.�       �	
�lnYc�A�*

lossr<��B       �	1�mnYc�A�*

lossO��=|s�!       �	r2nnYc�A�*

losse�N=׏$�       �	1�nnYc�A�*

loss0�<-���       �		oonYc�A�*

lossxI�<N28%       �	�pnYc�A�*

loss&�<�ߓ       �	s�pnYc�A�*

loss3ـ=�nR#       �	�fqnYc�A�*

lossQ-�=�0�       �	QrnYc�A�*

loss=�I=��ޏ       �	��rnYc�A�*

losse"*=�b�       �	�psnYc�A�*

lossTϻ<���       �	�tnYc�A�*

loss��<Y$�'       �	Q�tnYc�A�*

loss�+#<X�:       �	�XunYc�A�*

loss6��;T��g       �	�unYc�A�*

loss��=�V�       �	ՖvnYc�A�*

loss��<7w`|       �	�<wnYc�A�*

loss'C�;/ �+       �	�wnYc�A�*

loss��@<��ˮ       �	�sxnYc�A�*

loss=*=J߫�       �	4ynYc�A�*

loss�� =�3�       �	ɯynYc�A�*

loss���=��F�       �	�RznYc�A�*

loss�R;=	+4       �	��znYc�A�*

loss�xz=��       �	ҍ{nYc�A�*

loss^��=����       �	+|nYc�A�*

lossz9�<�<H       �	��|nYc�A�*

loss���<k�CM       �	(b}nYc�A�*

lossI�=y+�f       �	Y�}nYc�A�*

loss]��<�  �       �	��~nYc�A�*

loss�6�=%��       �	�7nYc�A�*

lossA�<�^�?       �	6�nYc�A�*

lossW�n=AV�       �	<f�nYc�A�*

lossv�a<o�        �	���nYc�A�*

loss=9J&�       �	���nYc�A�*

loss8"=&�,�       �	�*�nYc�A�*

loss���<����       �	���nYc�A�*

loss{��=���c       �	�0�nYc�A�*

loss���<u��^       �	�ʄnYc�A�*

loss3��<Ń�       �	�h�nYc�A�*

loss|W>=�iL�       �	��nYc�A�*

loss�5'<�bؓ       �	Ę�nYc�A�*

loss  =W��U       �	�D�nYc�A�*

loss,�s=O�       �		݈nYc�A�*

lossm�<�_�5       �	�w�nYc�A�*

lossQ8�<kf       �	k�nYc�A�*

loss)w�=���       �	R��nYc�A�*

loss�Q�=�<       �	�L�nYc�A�*

lossl�Q=X�z�       �	o�nYc�A�*

lossf�=��W       �	��nYc�A�*

lossO�;=a-��       �	"6�nYc�A�*

lossTD�<o��W       �	�(�nYc�A�*

loss��=�l�       �	�ƎnYc�A�*

lossc�1=�B��       �	b�nYc�A�*

loss��==Z �       �	"�nYc�A�*

lossR-�<����       �	ø�nYc�A�*

loss�g<
��       �	�T�nYc�A�*

lossd��<b�       �	��nYc�A�*

loss	�9=��$       �	U��nYc�A�*

lossa��=�dE       �	9'�nYc�A�*

lossg=�p<6       �	5�nYc�A�*

loss��2=e���       �	g��nYc�A�*

loss�C�<���,       �	N+�nYc�A�*

loss}==!s0       �	_�nYc�A�*

loss� �<�sT�       �	���nYc�A�*

loss��<�drw       �	QN�nYc�A�*

loss�f�<�!       �	���nYc�A�*

loss�l�=��,       �	r��nYc�A�*

loss(�"=�PЬ       �	�A�nYc�A�*

lossfb>=e2       �	�ٙnYc�A�*

loss
�p=H�       �	�n�nYc�A�*

loss_��<w_B�       �	S�nYc�A�*

loss�f@=����       �	4��nYc�A�*

loss�ν<��'       �	�5�nYc�A�*

lossa��<��uF       �	��nYc�A�*

loss���;m畢       �	Z��nYc�A�*

loss��=?�       �	*�nYc�A�*

lossXH�=����       �	ƞnYc�A�*

loss�a�<ve+       �	�q�nYc�A�*

lossa�=���d       �	��nYc�A�*

loss�(<0C�       �	�d�nYc�A�*

lossJ,k<[#       �	���nYc�A�*

loss/8}<�P�       �	���nYc�A�*

loss�O=`3*8       �	LR�nYc�A�*

lossʓ=���6       �	��nYc�A�*

loss.P�<�@2
       �	���nYc�A�*

loss��=�6�v       �	�nYc�A�*

loss��f<�Pr       �	���nYc�A�*

loss1�D=�[        �	�R�nYc�A�*

loss{'(=/Y+�       �	]��nYc�A�*

loss�o�<I
R2       �	���nYc�A�*

loss06=�5�B       �	q�nYc�A�*

lossQ�<#���       �	
�nYc�A�*

loss@6�<>��       �	���nYc�A�*

loss�3?=˵y       �	�;�nYc�A�*

lossI��<��.�       �	�ҪnYc�A�*

loss�^�;v�Z�       �	�h�nYc�A�*

loss�\"=#їX       �	��nYc�A�*

loss��f=�vL	       �	襬nYc�A�*

lossf��<zh �       �	aP�nYc�A�*

loss��G=��C�       �	��nYc�A�*

lossĘ�=����       �	ݔ�nYc�A�*

loss8��=Ug��       �	|+�nYc�A�*

loss�n�=�c%�       �	�įnYc�A�*

loss�8�< �j?       �	[�nYc�A�*

loss6��=�Z       �	��nYc�A�*

loss��>=��B       �	���nYc�A�*

loss�MJ<^b�b       �	F"�nYc�A�*

loss�%�<�P�       �	軲nYc�A�*

loss;b=���|       �	�R�nYc�A�*

loss!u=�Y�       �	W��nYc�A�*

loss�j=���       �	���nYc�A�*

loss@��=�)�A       �	�$�nYc�A�*

lossc<��<�       �	aµnYc�A�*

loss,�<'L$G       �	)Z�nYc�A�*

loss�,�<��1.       �	�z�nYc�A�*

lossAF�<�xF       �	���nYc�A�*

loss�<���        �	���nYc�A�*

loss>�<��K       �	�7�nYc�A�*

lossf5�<����       �	�лnYc�A�*

loss
��=��.�       �	mq�nYc�A�*

loss�o]<=�2�       �	�%�nYc�A�*

loss�ٗ<�-�       �	侽nYc�A�*

loss��<=P�       �	T��nYc�A�*

loss�S`<=/0       �	\W�nYc�A�*

loss��<0��#       �	���nYc�A�*

loss�t�<Jf
�       �	@��nYc�A�*

loss�ؙ=��F_       �	��nYc�A�*

loss <W<E�       �	0/�nYc�A�*

loss���<1�U       �	2��nYc�A�*

loss�(�<o_~       �	m�nYc�A�*

loss�(>�h20       �	��nYc�A�*

loss!�==�D9�       �	m��nYc�A�*

loss��<�G!       �	E�nYc�A�*

loss��R=%Ԇ�       �	U��nYc�A�*

loss�;�=�e�       �	mp�nYc�A�*

loss�]>��       �	_)�nYc�A�*

loss
i�<X��       �	���nYc�A�*

loss��=��q�       �	���nYc�A�*

loss�O~=qu��       �	?��nYc�A�*

loss9�>�ƅ�       �	�)�nYc�A�*

loss���;��b       �	��nYc�A�*

loss��><+�       �	�r�nYc�A�*

loss�Y='ncO       �	��nYc�A�*

loss6q=GRm&       �	ʥ�nYc�A�*

loss?��<�nIy       �	�B�nYc�A�*

loss#�L=��Rm       �	�;�nYc�A�*

loss�o<%���       �	���nYc�A�*

loss�6�=��/.       �	�l�nYc�A�*

loss;�f=��       �	6�nYc�A�*

loss?�=ڷ�{       �	؝�nYc�A�*

loss��<J��       �	�4�nYc�A�*

loss�yI=-[|1       �	
��nYc�A�*

lossc��<'�,       �	Oy�nYc�A�*

loss��/=��^�       �	s�nYc�A�*

loss�&=��~       �	-��nYc�A�*

loss3�p=�`v�       �	<K�nYc�A�*

loss:�M<�<$       �	J��nYc�A�*

loss�k�<��?}       �	��nYc�A�*

loss%O�<�d+�       �	:�nYc�A�*

loss}/=o�N�       �	���nYc�A�*

loss�}�<�&       �	[��nYc�A�*

lossOL�<� ��       �	L8�nYc�A�*

loss��L<�&Zo       �	���nYc�A�*

lossN�y=IG��       �	�q�nYc�A�*

loss�"�=�s|�       �	�	�nYc�A�*

lossj��=+/��       �	Ϟ�nYc�A�*

loss��=���+       �	�:�nYc�A�*

lossmJ@=;�       �	���nYc�A�*

loss���<i�2       �	w�nYc�A�*

loss�3%=�h�       �	�4�nYc�A�*

loss�>�=����       �	c��nYc�A�*

loss�=�ى�       �	�m�nYc�A�*

loss`�!=�dA       �	��nYc�A�*

loss:PN=R��       �	H��nYc�A�*

loss|�=
g       �	)��nYc�A�*

loss�A<5p�9       �	�-�nYc�A�*

loss-�<02�       �	��nYc�A�*

lossFA1=��mJ       �	��nYc�A�*

lossE`�=%L�	       �	�;�nYc�A�*

loss���=�d�       �	c��nYc�A�*

losss��<�쪀       �	;m�nYc�A�*

loss/�<�w�$       �	�t�nYc�A�*

lossJr�<��]�       �	��nYc�A�*

loss�=oJip       �	A��nYc�A�*

lossH�=�a&�       �	P5�nYc�A�*

loss�]�;8j��       �	���nYc�A�*

loss���<�b6�       �	�p�nYc�A�*

loss�ʣ=�m�       �	x	�nYc�A�*

lossr[m<z�	�       �	���nYc�A�*

loss!��<�V�       �	C;�nYc�A�*

loss�3�<$�>k       �	���nYc�A�*

loss�D�<���#       �	���nYc�A�*

loss!_<k�R       �	/��nYc�A�*

loss�;�;a�ޑ       �	%"�nYc�A�*

loss1�b=�a�       �	���nYc�A�*

losss��<Y�v�       �	��nYc�A�*

loss��<q�yn       �	)�nYc�A�*

loss�k=�v��       �	���nYc�A�*

loss��;�4j       �	U�nYc�A�*

loss�7�;���       �	O��nYc�A�*

loss{�m<�x��       �	C��nYc�A�*

loss ��<Mo�U       �	�$�nYc�A�*

loss�P~<��I       �	���nYc�A�*

loss��>�i       �	�d�nYc�A�*

loss +>��i:       �	)�nYc�A�*

loss�ȏ<�^�       �	���nYc�A�*

loss���<cg�       �	�B�nYc�A�*

loss��<��t�       �	���nYc�A�*

loss/I<�dW       �	���nYc�A�*

loss�Ռ=�r�u       �	�4�nYc�A�*

loss��<B�!�       �	���nYc�A�*

loss!7w=��       �	8i�nYc�A�*

loss��#<i�l�       �	��nYc�A�*

loss�=R��       �	���nYc�A�*

loss��=|>1       �	vm�nYc�A�*

loss��#=� �       �	�nYc�A�*

loss�.<���       �	���nYc�A�*

lossx��=����       �	�G�nYc�A�*

lossc�=�J��       �	���nYc�A�*

loss_@D=�pY}       �	dx�nYc�A�*

loss��<%�"�       �	��nYc�A�*

loss�!r<+~p�       �	X��nYc�A�*

lossq��<>�c&       �	�C�nYc�A�*

loss
�Q<�h�       �	L��nYc�A�*

loss���=MeR�       �	�{ oYc�A�*

loss�p<�d       �	�oYc�A�*

lossj�<9t        �	/�oYc�A�*

lossQ�'<Z w�       �	K\oYc�A�*

loss��
=�:�U       �	0�oYc�A�*

loss�<���       �	��oYc�A�*

loss�u<��]�       �	1$oYc�A�*

loss�U�<���       �	˻oYc�A�*

loss�<΃�:       �	�^oYc�A�*

loss�F�=����       �	�oYc�A�*

loss��<�R �       �	_�oYc�A�*

loss�m =㤌�       �	 BoYc�A�*

lossg�<%�LZ       �	�oYc�A�*

lossů�=:��       �	j�oYc�A�*

loss�
�=���       �	f�	oYc�A�*

loss	|�=�d@�       �	�
oYc�A�*

lossoZ�<��D       �	�zoYc�A�*

losso3<���!       �	�,oYc�A�*

loss,=NJ       �	��oYc�A�*

loss
�&=F��       �	�yoYc�A�*

lossЭ<��$r       �	LoYc�A�*

loss�L�<6�       �	��oYc�A�*

loss��:v��       �	�coYc�A�*

lossљ�<�if(       �	WAoYc�A�*

loss!��<O��H       �	s�oYc�A�*

lossR��<g|-       �	�soYc�A�*

loss��P=n�#       �	�oYc�A�*

loss���<7�|{       �	��oYc�A�*

loss���=g�"�       �	rQoYc�A�*

loss�I�<�� 8       �	1	oYc�A�*

losst��;�yg       �	.�oYc�A�*

loss"�<���P       �	�MoYc�A�*

loss�SX< ��T       �	��oYc�A�*

lossغ$;T��       �	��oYc�A�*

loss��=R�R       �	�)oYc�A�*

loss��4<���j       �	�oYc�A�*

loss�D�;X#�^       �	(aoYc�A�*

loss���<�_�)       �	)AoYc�A�*

loss�q�:��Y       �	{�oYc�A�*

loss�@=)y�L       �	��oYc�A�*

loss�a;�ڵ       �	oYc�A�*

loss��m:��_       �	��oYc�A�*

loss6�P:\�       �	�joYc�A�*

loss�;���       �	JoYc�A�*

loss.<�ۄ�       �	�oYc�A�*

lossB<EvF       �	�QoYc�A�*

lossf�a;?��       �	�oYc�A�*

loss�C�<A�t       �	̳oYc�A�*

loss�1Z>�E�!       �	QK oYc�A�*

loss�-<�F�       �	�� oYc�A�*

loss���;)�{�       �	��!oYc�A�*

loss5ؔ<i�5       �	("oYc�A�*

loss�0~=��@�       �	U�"oYc�A�*

loss���<9yz�       �	[�#oYc�A�*

loss4��<��       �	c�$oYc�A�*

lossL�L=P�*%       �	\�%oYc�A�*

lossq��=,���       �	F�&oYc�A�*

loss�m�=3��       �	��'oYc�A�*

loss�O<U!Mq       �	��(oYc�A�*

lossڈ1<���V       �	�)oYc�A�*

loss�>L=�       �	�)*oYc�A�*

loss3�B=�ָv       �	��*oYc�A�*

lossO�<$��       �	�r+oYc�A�*

loss,J=�� �       �	�$,oYc�A�*

losstL=B�        �	��,oYc�A�*

lossȤ"=�*��       �	9�-oYc�A�*

loss�4�<)x�       �	>B.oYc�A�*

loss��G=�=�l       �	�</oYc�A�*

loss�.�<�� :       �	Z�/oYc�A�*

loss�9<�^�       �	-1oYc�A�*

lossS��<�t�       �	��1oYc�A�*

lossZD=����       �	`s2oYc�A�*

lossHlq=�A�N       �	r3oYc�A�*

loss`>�<��V�       �	`�3oYc�A�*

lossEb�:㲄,       �	nn4oYc�A�*

loss�d<t<�w       �	�
5oYc�A�*

losse.~=�l       �	]�5oYc�A�*

loss�*�=���@       �	Kw6oYc�A�*

loss�k�=[�'       �	x(7oYc�A�*

lossJdY<�Ν       �	q8oYc�A�*

loss"=d��       �	P�8oYc�A�*

lossJ�<��       �	9a9oYc�A�*

loss�wL;?��W       �	�:oYc�A�*

loss{`<����       �	��:oYc�A�*

loss周<?{[       �	n;oYc�A�*

loss#��<޴ww       �	�%<oYc�A�*

loss��=����       �	N�<oYc�A�*

loss.�t=w���       �	�=oYc�A�*

losse�<8@4�       �	�&>oYc�A�*

loss�� <����       �	��>oYc�A�*

loss��}<���       �	~q?oYc�A�*

loss#�<c=�       �	�#@oYc�A�*

loss؀ =��       �	�@oYc�A�*

loss6�<���       �	��AoYc�A�*

loss�c=(��       �	�/BoYc�A�*

loss/Kq=��Uy       �	��BoYc�A�*

loss�3=7��       �	�yCoYc�A�*

loss��<4[��       �	�EoYc�A�*

loss��=�H��       �	��EoYc�A�*

lossA�<��f�       �	JaFoYc�A�*

loss�B�<e�&5       �	�^^oYc�A�*

loss�P�=w0�       �	� _oYc�A�*

losstT=z�       �	��_oYc�A�*

loss��=�N�       �	�>`oYc�A�*

loss=�L��       �	c�aoYc�A�*

loss_�L<��i       �	�VboYc�A�*

loss�@=�ҝ       �	coYc�A�*

lossE�
=l:_       �	��doYc�A�*

lossm�z= ݿ�       �	�`eoYc�A�*

loss\�w=uV=       �	y>foYc�A�*

loss��;�3�       �	��foYc�A�*

loss��#=��s4       �	��goYc�A�*

lossn�z=�[��       �	ioYc�A�*

loss���<G:q�       �	��ioYc�A�*

lossm=]T;       �	�koYc�A�*

loss���<����       �	�koYc�A�*

loss�ܡ;ٗz�       �	b�loYc�A�*

loss&�<֐�K       �	��moYc�A�*

loss���<���       �	~�noYc�A�*

loss�IF=^(շ       �	�NooYc�A�*

loss]j�<}��%       �	��ooYc�A�*

loss_��<:��       �	��poYc�A�*

lossAL<���n       �	�qoYc�A�*

loss��%=�ԩ       �	!roYc�A�*

loss���<4\6�       �	�roYc�A�*

loss��p<��=       �	�	toYc�A�*

loss���<�9]�       �	��toYc�A�*

lossffy<���M       �		�uoYc�A�*

lossr�V=%s��       �	�voYc�A�*

loss�=�S��       �	2woYc�A�*

loss;�<φ9�       �	e�woYc�A�*

loss{&�<)W�       �	�xoYc�A�*

lossb�<M�<�       �	�OyoYc�A�*

loss=˟=�a�j       �	l�yoYc�A�*

loss�ş;�ͨ       �	ЙzoYc�A�*

loss	�-=���       �	�f{oYc�A�*

loss���<����       �	�|oYc�A�*

loss�9,=/��M       �	��|oYc�A�*

loss�k�=
%c(       �	�A}oYc�A�*

loss���=���       �	�}oYc�A�*

lossX&�<�Y#�       �	��~oYc�A�*

loss�q=��0i       �	�[oYc�A�*

loss씗<j�&�       �	\�oYc�A�*

lossR�~=G��       �	ԀoYc�A�*

lossWx=�>�       �	Tn�oYc�A�*

loss]g\=���-       �	�oYc�A�*

loss��=	M�       �	���oYc�A�*

loss��w=c���       �	�=�oYc�A�*

loss��q<6��z       �	��oYc�A�*

loss�[;��T�       �	Q��oYc�A�*

lossʬ�<��       �	 �oYc�A�*

loss�,=şp:       �	k��oYc�A�*

loss���<J��W       �	UL�oYc�A�*

loss��=�       �	��oYc�A�*

loss�U�;w���       �	���oYc�A�*

loss��?;���>       �	�Y�oYc�A�*

loss��;��P�       �	]R�oYc�A�*

loss`=�5+       �	�oYc�A�*

loss��=l��>       �	���oYc�A�*

loss��=�h�u       �	&��oYc�A�*

lossa�<�W[�       �	%��oYc�A�*

lossR<Xv��       �	���oYc�A�*

loss�JD<T]4�       �	Mh�oYc�A�*

loss�"�<��D       �	�5�oYc�A�*

loss���<���0       �	Y6�oYc�A�*

loss��+=�\,w       �	0h�oYc�A�*

loss��d=R�0�       �	�oYc�A�*

loss磃<�b�       �	ȒoYc�A�*

loss�t�=��p�       �	"q�oYc�A�*

loss03�<pȂn       �	�oYc�A�*

loss�*=q��I       �	*�oYc�A�*

loss�-a=�~       �	��oYc�A�*

loss��D=���       �	ǹ�oYc�A�*

loss���<n�!	       �	�U�oYc�A�*

loss$�=>I�       �	p$�oYc�A�*

loss��;�~i       �	�oYc�A�*

loss=�<��T       �	�^�oYc�A�*

loss�HW<[H7�       �	h�oYc�A�*

loss��L<dG?F       �	���oYc�A�*

loss�,�<����       �	�F�oYc�A�*

loss��<���G       �	��oYc�A�*

loss�"=e��       �	ϡ�oYc�A�*

lossᧅ=���m       �	'L�oYc�A�*

loss�L6<9�4�       �	��oYc�A�*

lossq�<��
       �	N�oYc�A�*

lossL�W<���       �	��oYc�A�*

lossԵ;<���}       �	�ПoYc�A�*

loss�5N=�a�       �	
j�oYc�A�*

lossAl�<���       �	l��oYc�A�*

loss,�M=��       �	���oYc�A�*

loss�D=�lO;       �	,�oYc�A�*

loss@z�=a�       �	ѣoYc�A�*

loss��'=m=��       �	�n�oYc�A�*

loss��<&��#       �	�oYc�A�*

loss�,5=�!r       �	��oYc�A�*

loss�I�;R��       �	jM�oYc�A�*

loss�Z =?�<�       �	l�oYc�A�*

lossTe=Zu��       �	���oYc�A�*

lossQ�=�l|       �	!!�oYc�A�*

loss���=����       �	o��oYc�A�*

loss�j<�K��       �	"U�oYc�A�*

loss)j�<ʯ�       �	��oYc�A�*

loss�ܪ<`u78       �	ڐ�oYc�A�*

loss�H<Xv^�       �	�R�oYc�A�*

loss��=G���       �	)�oYc�A�*

loss&�4=�>��       �	U��oYc�A�*

losso�k;��       �	��oYc�A�*

loss�<v}n�       �	���oYc�A�*

loss��=��N9       �	'K�oYc�A�*

loss/��<T^�Q       �	��oYc�A�*

lossNU�=<��       �	���oYc�A�*

loss�ݛ<�e��       �	G!�oYc�A�*

loss�!{<�ZJ       �	ðoYc�A�*

loss_:�;^���       �	�[�oYc�A�*

loss�#=r��Z       �	���oYc�A�*

lossRba<=F       �	陲oYc�A�*

loss#�=���       �	28�oYc�A�*

loss��=䁀T       �	%�oYc�A�*

lossq�w=Gj	x       �	N��oYc�A�*

loss5A=H{�       �	8L�oYc�A�*

loss҈�<�<�W       �	���oYc�A�*

loss���<>k�h       �	E��oYc�A�*

loss36a=�m�       �	�I�oYc�A�*

loss
�=?Դ�       �	��oYc�A�*

loss���<T:�E       �	И�oYc�A�*

loss���<���       �	�f�oYc�A�*

loss&v�<�       �	<�oYc�A�*

loss�;== �       �	��oYc�A�*

loss:az=��B�       �	O�oYc�A�*

lossW@�<�       �	��oYc�A�*

loss�[�<�;�H       �	[��oYc�A�*

lossS�<=�%       �	�7�oYc�A�*

lossƠ�=���       �	߽oYc�A�*

lossU� =�8�F       �	�|�oYc�A�*

loss��>!�h�       �	�oYc�A�*

loss,�x<%�       �	���oYc�A�*

loss�=���       �	�G�oYc�A�*

loss�Dg=�e�       �	���oYc�A�*

loss�l=ET�|       �	���oYc�A�*

lossNt<���       �	9�oYc�A�*

lossf�<6��       �	,��oYc�A�*

lossa��<���^       �	J|�oYc�A�*

loss�.�<��       �	��oYc�A�*

loss��N=�<��       �	<��oYc�A�*

loss_1H=�.�       �	h��oYc�A�*

loss�
�<Ii�       �	2�oYc�A�*

loss���<K���       �	���oYc�A�*

loss$�<��[�       �	 a�oYc�A�*

lossa�<LA�(       �	��oYc�A�*

lossV��<)k�/       �	��oYc�A�*

loss%�6<�'�       �	?:�oYc�A�*

loss�V=�j}�       �	��oYc�A�*

loss}a�=vA��       �	ɯ�oYc�A�*

loss٭ =g�%�       �	�I�oYc�A�*

lossd=��b�       �	���oYc�A�*

loss$%�<�Y�;       �	��oYc�A�*

loss_��<Lr�       �	�S�oYc�A�*

loss�l7=!���       �	���oYc�A�*

loss�8�=H�Y       �	ҧ�oYc�A�*

lossE�)<u�~�       �	\U�oYc�A�*

loss�x�<����       �	a��oYc�A�*

loss���<��9z       �	o��oYc�A�*

loss�+�<M+�       �	�E�oYc�A�*

loss@�[<)�iM       �	���oYc�A�*

loss.�&=����       �	F��oYc�A�*

loss��=?�X�       �	�A�oYc�A�*

loss�% <*H-�       �	��oYc�A�*

loss�(z=��T"       �	v��oYc�A�*

loss��<�]S       �	3�oYc�A�*

loss�0<�@GH       �	���oYc�A�*

lossÍA=��;�       �	���oYc�A�*

loss/�j;�G=Z       �	� �oYc�A�*

loss�:�;- z|       �	���oYc�A�*

lossD#D=�%�       �	z�oYc�A�*

loss$�?<�sj�       �	��oYc�A�*

lossɀ=�3�       �	ʩ�oYc�A�*

loss��*=�47       �	�?�oYc�A�*

loss�q�=y��       �	v��oYc�A�*

lossi��<>9�       �	t}�oYc�A�*

loss���<y��       �	<�oYc�A�*

loss%wc;O���       �	ͮ�oYc�A�*

loss5�=��h       �	,d�oYc�A�*

loss֯�;|!X       �	e��oYc�A�*

lossO�E<ƹ�       �	���oYc�A�*

lossM��<a�S       �	�_�oYc�A�*

loss���=����       �	���oYc�A�*

loss�j=��5V       �	6��oYc�A�*

losss+�<g�s�       �	�*�oYc�A�*

loss�<����       �	.��oYc�A�*

lossÖ`=}�       �	ɓ�oYc�A�*

loss���<�>?�       �	�*�oYc�A�*

loss��=o�'�       �	\:�oYc�A�*

loss���;��       �	���oYc�A�*

lossN&�<fR2�       �	�u�oYc�A�*

loss��a<]/       �	�!�oYc�A�*

loss-ո<���l       �	/��oYc�A�*

lossf�;�`#�       �	�a�oYc�A�*

lossLS�<���       �	�
�oYc�A�*

lossx�N=�T]       �	��oYc�A�*

loss�h�<�a]       �	�B�oYc�A�*

lossv͡<�S�F       �	i��oYc�A�*

losse��<&
�E       �	�~�oYc�A�*

lossM��<�       �	��oYc�A�*

loss�=c���       �	Ǹ�oYc�A�*

loss��J<���       �	�Q�oYc�A�*

loss�a�<���	       �	���oYc�A�*

loss�� =���;       �	D��oYc�A�*

loss�ݷ=;�F�       �	t'�oYc�A�*

lossH�=�2w       �	<��oYc�A�*

lossq�<�2       �	%[�oYc�A�*

lossɀx= ��       �	` �oYc�A�*

loss(A#=Z6       �	��oYc�A�*

loss�l'<���       �	#��oYc�A�*

loss'B�<1��F       �	�9�oYc�A�*

loss��+=�Qx�       �	���oYc�A�*

loss��=pƞ�       �	���oYc�A�*

loss�O�<�C��       �	�7�oYc�A�*

loss��=�H       �	��oYc�A�*

loss@�=���       �	Sy�oYc�A�*

loss9`�=��a+       �		�oYc�A�*

loss[[�<JtF       �	���oYc�A�*

loss��b<�F       �	s�oYc�A�*

loss���<�PQ�       �	���oYc�A�*

loss)c�=���E       �	�l�oYc�A�*

lossd==�)�       �	>�oYc�A�*

loss�U=9D�X       �	e��oYc�A�*

loss�f�<�,sL       �	�E�oYc�A�*

loss�w�<h�@       �	���oYc�A�*

loss�B�<g8=�       �	��oYc�A�*

loss��=h[M�       �	"3�oYc�A�*

loss�� <e�r       �	���oYc�A�*

lossj��;5객       �	Kx�oYc�A�*

loss���<��9�       �	+�oYc�A�*

loss��q=?�       �	��oYc�A�*

lossd`={D�C       �	3N pYc�A�*

lossGI=i�~=       �	�� pYc�A�*

loss��=!]��       �	��pYc�A�*

loss�4�=��}z       �	�pYc�A�*

loss�==z��o       �	��pYc�A�*

loss;��:�Đ�       �	WpYc�A�*

loss��b<#�w�       �	�pYc�A�*

loss)�^=o	��       �	��pYc�A�*

lossP[=�v0       �	�(pYc�A�*

loss��<���       �	~�pYc�A�*

lossh��<+�E       �	�hpYc�A�*

loss���<q?@       �	pYc�A�*

lossM$y<�g��       �	ݳpYc�A�*

loss?�+<`��       �	OpYc�A�*

loss��q=퀍�       �	O�pYc�A�*

loss$�<k$��       �	E
pYc�A�*

loss�E=?ӟw       �	��
pYc�A�*

lossAJ�=�~ܩ       �	�pYc�A�*

loss�w�<h��D       �	�KpYc�A�*

loss�|X<�9T�       �	��pYc�A�*

loss/"=�B       �	��pYc�A�*

loss):<�dl       �	�5pYc�A�*

loss��=WDm       �	��pYc�A�*

loss{�w=�Bh�       �	<jpYc�A�*

lossM@�;#\��       �	gGpYc�A�*

loss�
�<MN��       �	��pYc�A�*

lossX��<��       �	X�pYc�A�*

lossL�=�-�X       �		6pYc�A�*

lossݕ�=݃$�       �	�?pYc�A�*

loss,��<񖃉       �	��pYc�A�*

loss�\�<F&�       �	q�pYc�A�*

loss��;O��       �	�6pYc�A�*

lossQ�|=�x7�       �	J�pYc�A�*

lossR�M=B��       �	�lpYc�A�*

loss���=V��`       �	WpYc�A�*

loss�%="]B^       �	�pYc�A�*

loss3vS<�2�       �	��pYc�A�*

loss���=�(�       �	PppYc�A�*

lossڝ=�M��       �	/pYc�A�*

loss
,�=H�B�       �	_�pYc�A�*

loss�*�=1�8       �	&UpYc�A�*

loss�&=���.       �	��pYc�A�*

loss��< �I       �	��pYc�A�*

loss���;��8/       �	�pYc�A�*

loss�L=�ʔ       �		�pYc�A�*

loss��h=����       �	SA pYc�A�*

loss�,�:�yC�       �	�� pYc�A�*

loss3��<.Tn�       �	ڏ!pYc�A�*

lossqy�=8�[�       �	�-"pYc�A�*

loss�VS=ߕ�       �	��"pYc�A�*

lossΪ�<PR�9       �	x_#pYc�A�*

lossF%=N�x�       �	b.$pYc�A�*

loss�_7=3p�n       �	�%pYc�A�*

lossȭ�<���       �	V�%pYc�A�*

loss��v=�i�       �	�R&pYc�A�*

lossM̹< �q�       �	��&pYc�A�*

loss&�C;�B��       �	�'pYc�A�*

lossӣ�=-��c       �	-(pYc�A�*

lossã�<��       �	� )pYc�A�*

loss]D;cM�       �	�"*pYc�A�*

loss�
=�u�O       �	��*pYc�A�*

lossJ�8=� �G       �	we+pYc�A�*

lossI�`<'�,H       �	*�+pYc�A�*

loss�
�<^9�e       �	m�,pYc�A�*

loss-�m=��`�       �	�L-pYc�A�*

loss�$�=LET       �	��-pYc�A�*

loss&�=-G�       �	�.pYc�A�*

loss�'�<�z��       �	�^/pYc�A�*

loss�_<�R��       �	��/pYc�A�*

loss�9d=��F�       �	��0pYc�A�*

lossMC=��h       �	�61pYc�A�*

lossw�=��       �	E�1pYc�A�*

loss}��;V�5�       �	�v2pYc�A�*

loss�x�=ӊg       �	�3pYc�A�*

loss�4�;@��;       �	�3pYc�A�*

loss� (<R���       �	�I4pYc�A�*

loss�<��2I       �	;�4pYc�A�*

lossF�=��/�       �	�5pYc�A�*

loss�Sz=�<A�       �	�6pYc�A�*

loss��<�$1(       �	�6pYc�A�*

lossD��;?2$�       �	g7pYc�A�*

loss!T�<�wJy       �	��7pYc�A�*

loss��V<5h�+       �	��8pYc�A�*

loss,=y<K�'�       �	X99pYc�A�*

loss��=ȳ�
       �	��9pYc�A�*

loss8��<W��       �	an:pYc�A�*

lossIo�<y�B/       �	l;pYc�A�*

loss��=�	�       �	|�;pYc�A�*

loss�?=~"7�       �	�C<pYc�A�*

loss� <��       �	��<pYc�A�*

loss�2_=���       �	P�=pYc�A�*

lossl�=�8�       �	N*>pYc�A�*

loss��m=��f�       �	�>pYc�A�*

loss��<�i�a       �	�p?pYc�A�*

loss~�<f���       �	n@pYc�A�*

loss:�(=�@�       �	��@pYc�A�*

lossV!=Wc�t       �	c^ApYc�A�*

loss��6<��?f       �	�BpYc�A�*

lossH�S<�)��       �	ڨBpYc�A�*

lossAq�<�F�       �	{KCpYc�A�*

lossw�=r�7�       �	��CpYc�A�*

loss.'=@��       �	�DpYc�A�*

loss�y�<�V�W       �	N*EpYc�A�*

loss�\�<N=�       �	{�EpYc�A�*

loss/y=��8�       �	�UFpYc�A�*

lossT�I<d[q�       �	��FpYc�A�*

lossѫ;���       �	�GpYc�A�*

losss�7=J	       �	� HpYc�A�*

lossѬ�<�y�       �	E�HpYc�A�*

loss�0�<�C�0       �	�tIpYc�A�*

lossM�
>ohZ�       �	b.JpYc�A�*

loss*�u=D���       �	��JpYc�A�*

loss�£<8O^�       �	fKpYc�A�*

lossc�=]�^n       �	�LpYc�A�*

lossw='��g       �	��LpYc�A�*

loss���<��@�       �	�JMpYc�A�*

loss�Y�;XC��       �	?�MpYc�A�*

loss�>�<S        �	��NpYc�A�*

loss��<&�F7       �	>$OpYc�A�*

loss��<o*�Z       �	�OpYc�A�*

loss�Y�<t���       �	�`PpYc�A�*

lossE�7<��       �	��PpYc�A�*

lossʉ]<��       �	[�QpYc�A�*

lossD��<0�       �	�/RpYc�A�*

loss�7D<H}^       �	�RpYc�A�*

loss�3�<vو�       �	tSpYc�A�*

lossi��=�-       �	^TpYc�A�*

lossg;�=(�[�       �	a�TpYc�A�*

loss�Ϥ=�.�       �	�GUpYc�A�*

loss=��=h�i       �	��UpYc�A�*

loss�6=wܼ\       �	��VpYc�A�*

loss�t�=���l       �	g+WpYc�A�*

loss��,=0��a       �	��WpYc�A�*

loss�`<sn�       �	�vXpYc�A�*

loss�b<C���       �	�'YpYc�A�*

loss��;��       �	=�YpYc�A�*

loss���<vc��       �	oZpYc�A�*

loss�P=�ylv       �	�[pYc�A�*

loss&#:=O
-g       �	S�[pYc�A�*

loss�W�<���D       �	7m\pYc�A�*

lossdER<��{n       �	]pYc�A�*

loss��z=�M��       �	Ū]pYc�A�*

loss	h�=����       �	L�^pYc�A�*

loss��-=����       �	��_pYc�A�*

loss=�6=�Iq�       �	ro`pYc�A�*

loss#=Z�=       �	@apYc�A�*

loss&ڼ<}�4       �	��apYc�A�*

loss@5<9
�       �	�ZbpYc�A�*

loss8^< y�       �	��bpYc�A�*

lossZY�;�jϓ       �	^�cpYc�A�*

loss3<�i��       �	yYdpYc�A�*

loss�g=;��       �	�dpYc�A�*

loss/j`=Z��       �	ۦepYc�A�*

loss���=9��       �	wIfpYc�A�*

loss/�<[R4I       �	�fpYc�A�*

loss�r?<��wq       �	w+hpYc�A�*

loss]�'=�.�       �	��hpYc�A�*

lossvf<��       �	�jpYc�A�*

loss?�<���       �	�jpYc�A�*

lossɮ�<��$       �	�/kpYc�A�*

lossvD<tZ�       �	�kpYc�A�*

loss�-=�ݪ@       �	iWlpYc�A�*

lossP�<"��X       �	��lpYc�A�*

lossZVm<;W�/       �	��mpYc�A�*

loss���<��*�       �	�npYc�A�*

loss\Ǻ<�T�       �	k�npYc�A�*

loss�2]<���a       �	eSopYc�A�*

loss6�<?j�       �	}ppYc�A�*

loss�;=9U�       �	�qpYc�A�*

loss���<�R�       �	�qpYc�A�*

loss���<y��W       �	��rpYc�A�*

loss��s=֮|�       �	4�spYc�A�*

loss�f9=|Ek�       �	OtpYc�A�*

lossSV�<���       �	e�tpYc�A�*

lossm��<i{�       �	ıupYc�A�*

loss�:�<�i�$       �	 ^vpYc�A�*

loss��$=�e�       �	/�vpYc�A�*

loss�Q�=� ��       �	�wpYc�A�*

loss���<ڸ�       �	O@xpYc�A�*

loss�b�<��~�       �	z�xpYc�A�*

loss�=��
       �	��ypYc�A�*

loss̏�<�*       �	r�zpYc�A�*

loss攚<h�8       �	�'{pYc�A�*

lossFK�<���_       �	��{pYc�A�*

loss=��       �	lz|pYc�A�*

loss��Q<ߏ�       �	M}pYc�A�*

loss���<�B�       �	�}pYc�A�*

losso~�<�|~I       �	�D~pYc�A�*

loss!��<��|�       �	��~pYc�A�*

lossMQ9=gVG�       �	�upYc�A�*

lossj�R=�j��       �	��pYc�A�*

loss��@=Y���       �	���pYc�A�*

loss�TB<�p$�       �	�N�pYc�A�*

loss��<0<��       �	��pYc�A�*

loss��=D���       �	D��pYc�A�*

lossiD�<�y�       �	�'�pYc�A�*

loss���<1B�u       �	3ăpYc�A�*

lossib�<�qX       �	W]�pYc�A�*

loss��<,�d�       �	���pYc�A�*

losslo<P�       �	1��pYc�A�*

loss���<\�p       �	<M�pYc�A�*

loss3�\<#;M�       �	I�pYc�A�*

lossʁ$<�S��       �	6��pYc�A�*

loss�Pm<\w�       �	R*�pYc�A�*

loss��+=�T;�       �	��pYc�A�*

loss
n<���       �	կ�pYc�A�*

lossS�=/w�/       �	eŊpYc�A�*

loss�m�=s��{       �	�y�pYc�A�*

loss�/=��84       �	T�pYc�A�*

loss�FK=$be>       �	���pYc�A�*

loss��;���       �	�ʎpYc�A�*

loss���=��[�       �	�}�pYc�A�*

loss��{=�kq       �	�&�pYc�A�*

loss���<J�}�       �	hːpYc�A�*

loss]�<�?sS       �	3m�pYc�A�*

loss�/�=�ڂ       �	�pYc�A�*

loss㈎=�+ߤ       �	ȳ�pYc�A�*

lossox<�~�        �	�b�pYc�A�*

loss7y�;90��       �	��pYc�A�*

losst)0=�P       �	!��pYc�A�*

loss\u�=��ia       �	zT�pYc�A�*

loss�x;=@2w�       �	��pYc�A�*

lossf��<V9�       �	|��pYc�A�*

loss�s^<+��=       �	`>�pYc�A�*

loss5�;��ޚ       �	�ޗpYc�A�*

loss�O�<�A�I       �	��pYc�A�*

loss��=t�8,       �	��pYc�A�*

lossѪ<�~^       �	Z��pYc�A�*

lossr��;sć�       �	�V�pYc�A�*

loss"V�=*��$       �	Sv�pYc�A�*

loss���<���[       �	�pYc�A�*

loss408<W��       �	ٰ�pYc�A�*

loss��=D��       �	OW�pYc�A�*

loss�1�<��k       �	^�pYc�A�*

loss�G�=Q��       �	��pYc�A�*

lossP;<n�"J       �	�2�pYc�A�*

loss��?=���       �	�ΟpYc�A�*

lossM��;��6       �	�e�pYc�A�*

lossX�<���\       �	3��pYc�A�*

loss�K�=+C�       �	0��pYc�A�*

loss�b<!~=�       �	n4�pYc�A�*

lossj{�<�E�Q       �	SʢpYc�A�*

loss�(�;��'[       �	�d�pYc�A�*

loss��<3/��       �	���pYc�A�*

loss��X<w���       �	O��pYc�A�*

loss&��=���       �	d<�pYc�A�*

loss���=�8��       �	�ԥpYc�A�*

loss��L=4g��       �	rn�pYc�A�*

lossC&�<�	 �       �	��pYc�A�*

loss��2=|���       �	���pYc�A�*

loss*9=N	q       �	C9�pYc�A�*

lossP��<	��a       �	7ިpYc�A�*

loss�|b<19�G       �	$}�pYc�A�*

loss�n=��G       �	��pYc�A�*

loss�%�<���1       �	|��pYc�A�*

loss8�=Լz�       �	�[�pYc�A�*

loss��>=�k�8       �	j��pYc�A�*

lossF"�=�
N       �	���pYc�A�*

loss`0�;�0o8       �	� �pYc�A�*

loss}�=�0�       �	B�pYc�A�*

loss�=���k       �	���pYc�A�*

loss��<4�U       �	�H�pYc�A� *

loss.��<�W��       �	��pYc�A� *

loss
=�r�P       �	z��pYc�A� *

loss �<�'X�       �	�-�pYc�A� *

loss�<�q�       �	�ʱpYc�A� *

loss�)=X�E�       �	k�pYc�A� *

loss�4�;4�I       �	��pYc�A� *

loss�c<��O�       �	﮳pYc�A� *

loss�%<&UQ�       �	�K�pYc�A� *

loss3х=#[�       �	��pYc�A� *

loss F<�n�`       �	���pYc�A� *

loss��<3B�h       �	'L�pYc�A� *

loss/0�<. J       �	���pYc�A� *

lossCބ<�H4       �	���pYc�A� *

loss���<+{�$       �	�v�pYc�A� *

loss�<���z       �	�pYc�A� *

losse=�6��       �	��pYc�A� *

loss�
=T��       �	?��pYc�A� *

lossj�<��ٵ       �	~5�pYc�A� *

loss���<&�%�       �	޻pYc�A� *

loss
V=t�       �	>v�pYc�A� *

lossRV/=��(       �	�pYc�A� *

lossS�<��z       �	O��pYc�A� *

loss��<;@�       �	�H�pYc�A� *

loss���<� �       �	��pYc�A� *

loss�
<��       �	�|�pYc�A� *

loss*��<�FBs       �	��pYc�A� *

lossd�%;ڂ�        �	i��pYc�A� *

loss���<7�e�       �	cF�pYc�A� *

lossV�=ɜl       �	���pYc�A� *

loss���=�w-       �	r��pYc�A� *

loss���<�s\       �	p$�pYc�A� *

lossʧ�<�5p       �	��pYc�A� *

loss垻=��k�       �	3o�pYc�A� *

loss�<Vy�T       �	��pYc�A� *

loss:;7g�|       �	���pYc�A� *

lossҬR=,c��       �	y[�pYc�A� *

loss=�9<p#5h       �	)�pYc�A� *

loss_��<�{?       �	���pYc�A� *

lossX/�<��S       �	�]�pYc�A� *

loss��;���       �	���pYc�A� *

loss���<���       �	e��pYc�A� *

loss�Ҟ<É�K       �	fL�pYc�A� *

loss���9�Dī       �	�~�pYc�A� *

lossS$<��[       �	G!�pYc�A� *

loss�U<qP       �	:��pYc�A� *

loss��a:��Fi       �	�c�pYc�A� *

loss��?:)ڰ	       �	��pYc�A� *

lossRA;�_1       �	W��pYc�A� *

loss!�<�ȟB       �	3��pYc�A� *

loss���<<�.s       �	�,�pYc�A� *

losslB+;�ߋ       �	m��pYc�A� *

loss�u�;�B.�       �	Nd�pYc�A� *

loss�J�==�v�       �	r3�pYc�A� *

lossڝ;:�       �	f�pYc�A� *

loss� �<�V�<       �	G��pYc�A� *

loss�ȁ<�y�s       �	�b�pYc�A� *

lossdD=K.�I       �	���pYc�A� *

loss�.{< ���       �	6�pYc�A� *

loss��<�k�#       �	 ��pYc�A� *

loss=h�       �	�\�pYc�A� *

loss�=��0�       �	��pYc�A� *

loss�Oj=�Y y       �	Χ�pYc�A� *

loss��C=�Y!       �	/Q�pYc�A� *

loss��U<1�R�       �	���pYc�A� *

lossA�e<c�z       �	u��pYc�A� *

lossv>[=��$�       �	-�pYc�A� *

loss�]<�v��       �	��pYc�A� *

lossȐ�<��0       �	 |�pYc�A� *

loss�<51��       �	��pYc�A� *

loss�L<+Ţ�       �	���pYc�A� *

loss|�=���E       �	?V�pYc�A� *

loss�!=9�b�       �	M��pYc�A� *

loss,�!=�       �	���pYc�A� *

loss%.I<Z���       �	�<�pYc�A� *

lossi��<��y       �	��pYc�A� *

loss�-S<2�q       �	�r�pYc�A� *

loss��:��       �	~�pYc�A� *

loss�}�<q��j       �	���pYc�A� *

loss�/0;Ϫ�       �	wJ�pYc�A� *

loss��;{���       �	���pYc�A� *

lossv#�;�8Y�       �	��pYc�A� *

lossOrS=m�b�       �	���pYc�A� *

lossn_=_R�/       �	f�pYc�A� *

loss	;��}       �	� �pYc�A� *

loss�
=u�        �	���pYc�A� *

loss T�=d�F�       �	2�pYc�A� *

lossb�;�8��       �	)��pYc�A� *

loss/�<5�_       �	sc�pYc�A� *

loss�c<-� �       �	; �pYc�A� *

loss(�/=!�R�       �	���pYc�A� *

loss��G=k�       �	f��pYc�A� *

loss��=��k       �	e�pYc�A� *

loss�p<(��       �	$��pYc�A� *

lossv��;�#\       �	Yj�pYc�A� *

loss���<��`       �	��pYc�A� *

loss��4=;.�       �	ծ�pYc�A� *

lossz�Y<9x0       �	0L�pYc�A� *

loss���;�i��       �	t��pYc�A� *

loss|�=]۔       �	��pYc�A� *

loss��p=�r��       �	&�pYc�A� *

loss%�;����       �	й�pYc�A� *

loss|��<}s/0       �	�P�pYc�A� *

loss�Ǭ<E�&g       �	B��pYc�A� *

lossMQ�;�ZzQ       �	���pYc�A� *

loss�<�{K       �	
qYc�A� *

loss���<hqt       �	8qYc�A� *

loss�S=z�X�       �	��qYc�A� *

loss՚<��       �	�BqYc�A� *

loss�H�<J	tN       �	��qYc�A� *

loss!�]=�j��       �	��qYc�A� *

loss{�.=��nA       �	(qYc�A� *

lossc�S<�u�       �	j�qYc�A� *

loss7�<R.#       �	�qYc�A� *

loss�u�<ŗ,       �	k�qYc�A� *

loss��<��	�       �	h#qYc�A� *

losshy?=�O^�       �	��qYc�A� *

loss��<��ǽ       �	�hqYc�A� *

loss��=B��       �	SqYc�A� *

loss�p�<+=I       �	ΥqYc�A� *

loss:w�;%ls�       �	�GqYc�A� *

loss�h�:���)       �	��qYc�A� *

loss�ʇ<W��6       �	J�qYc�A� *

loss���<Md8?       �	t@qYc�A� *

lossA��=�g�       �	�qYc�A� *

loss���=����       �	��qYc�A� *

lossz�=@[�       �	� qYc�A� *

loss�XC<S�u�       �	s�qYc�A� *

lossV��<���z       �	�VqYc�A�!*

loss790=��w       �	x�qYc�A�!*

lossHE�<���       �	:�qYc�A�!*

loss�a�=�xƶ       �	g+qYc�A�!*

loss+=<L1�       �	��qYc�A�!*

loss��=���       �	؀qYc�A�!*

lossэ�<�x�P       �	GqYc�A�!*

lossW{=O{R�       �	��qYc�A�!*

loss�$=��-       �	�uqYc�A�!*

loss�k<�.a       �	,qYc�A�!*

loss�`�=��       �	��qYc�A�!*

loss�%	<�G{       �	}A qYc�A�!*

losswh�=]ҜZ       �	�� qYc�A�!*

loss�<�:�       �	�w!qYc�A�!*

loss�_=4(5       �	I"qYc�A�!*

loss��a=�׻d       �	��"qYc�A�!*

loss�rV=��_v       �	L#qYc�A�!*

loss)4=���       �	�#qYc�A�!*

losst�<���       �	�$qYc�A�!*

loss{��<;�4u       �	ȶ%qYc�A�!*

loss�=S��       �	�v&qYc�A�!*

loss�^�<32G�       �	'qYc�A�!*

loss#\�=(6�       �	ƥ'qYc�A�!*

loss���<���b       �	�;(qYc�A�!*

loss�B,=+9_b       �	��(qYc�A�!*

loss��=\X�<       �	Wx)qYc�A�!*

loss "�;�y��       �	�_*qYc�A�!*

loss�=u�'       �	4�*qYc�A�!*

loss|8�=����       �	��+qYc�A�!*

loss�ϣ<47?�       �	�8,qYc�A�!*

lossN�>?�k�       �	(�,qYc�A�!*

loss���<?�       �	v-qYc�A�!*

loss M�:���       �	7.qYc�A�!*

loss%i};.�%%       �	��.qYc�A�!*

loss���;e�       �	�M/qYc�A�!*

loss�<<��|       �	�/qYc�A�!*

loss ȸ<Rm�       �	j�0qYc�A�!*

loss�^�=0p �       �	@M1qYc�A�!*

lossȥO<�CB�       �	9�1qYc�A�!*

loss��;�wE�       �	��2qYc�A�!*

loss�r<W��       �	k3qYc�A�!*

loss��^=�r       �	�4qYc�A�!*

loss�ʤ<��0       �	u�4qYc�A�!*

loss껞<t݀T       �	rP5qYc�A�!*

loss��<��d       �	��5qYc�A�!*

loss۷�=## �       �	�6qYc�A�!*

loss8S5=���-       �	t'7qYc�A�!*

loss��=2�P       �	��7qYc�A�!*

lossC��<}$       �	�W8qYc�A�!*

loss��=��0�       �	��8qYc�A�!*

lossak{<���g       �	��9qYc�A�!*

loss= �:�       �	(:qYc�A�!*

lossi��<_\D       �	��:qYc�A�!*

lossߩ�;��v�       �	�d;qYc�A�!*

loss�G'=I��       �	��;qYc�A�!*

loss(b�<.+�       �	>�<qYc�A�!*

loss���<8[�!       �	�4=qYc�A�!*

loss��1=t1}�       �	�=qYc�A�!*

loss���=����       �	�o>qYc�A�!*

loss(T�<�}�       �	N?qYc�A�!*

loss]�:<k�]�       �	C�?qYc�A�!*

loss��<hͯ_       �	$H@qYc�A�!*

loss�C�<���j       �	��@qYc�A�!*

loss��<V¤2       �	�{AqYc�A�!*

loss���<�5]�       �	jBqYc�A�!*

lossm��<:���       �	�BqYc�A�!*

loss�H8<6��F       �	*TCqYc�A�!*

loss[��=�auW       �	��CqYc�A�!*

lossxs�<����       �	��DqYc�A�!*

lossd=�.IS       �	�DEqYc�A�!*

loss�+=|堨       �	��EqYc�A�!*

lossiJC=W�p       �	�wFqYc�A�!*

loss �S<��d       �	GqYc�A�!*

loss��<j!�       �	6�GqYc�A�!*

lossq�V<L�9b       �	<JHqYc�A�!*

loss-��=9K]       �	��HqYc�A�!*

lossmJP<b��g       �	��IqYc�A�!*

loss�I�</��$       �	�[JqYc�A�!*

loss�<�\�       �	�<KqYc�A�!*

loss�*�< %p#       �	[�KqYc�A�!*

loss�I=�}��       �	zoLqYc�A�!*

loss��=�5tk       �	�MqYc�A�!*

loss� =�"�       �	��MqYc�A�!*

loss�И<=�A�       �	_ENqYc�A�!*

loss�c�<o%��       �	��NqYc�A�!*

loss&I=�6'       �	4�OqYc�A�!*

loss9<6��T       �	rQPqYc�A�!*

lossL:=&1       �	��PqYc�A�!*

lossc7<�/�       �	 QqYc�A�!*

loss���<=B��       �	�RqYc�A�!*

loss6�V;C�q�       �	ڬRqYc�A�!*

loss���<�
�       �	�DSqYc�A�!*

loss�3<���%       �	��SqYc�A�!*

loss�z�<zx!�       �	�oTqYc�A�!*

loss�,=�U�7       �	JUqYc�A�!*

loss4 @=ym>�       �	��UqYc�A�!*

loss�D�<��Gz       �	�6VqYc�A�!*

loss��M=��       �	��VqYc�A�!*

lossa�c=��ؚ       �	ofWqYc�A�!*

lossQ�<ijv       �	��WqYc�A�!*

loss;��<�`�       �	�XqYc�A�!*

loss�
1<ն��       �	4YqYc�A�!*

lossé�<�Ĵ       �	1�YqYc�A�!*

loss�f=���3       �	�jZqYc�A�!*

loss�ry<����       �	�[qYc�A�!*

lossW�=R�Q�       �	��[qYc�A�!*

loss��<x���       �	�7\qYc�A�!*

loss6!�<mQ       �	�\qYc�A�!*

lossa!=93�       �	bh]qYc�A�!*

loss���<#BA�       �	v�]qYc�A�!*

lossL�p<1gA       �	�^qYc�A�!*

loss+4>Uy��       �	5(_qYc�A�!*

loss�=����       �	r�_qYc�A�!*

loss�?�<i���       �	�}`qYc�A�!*

loss�0=���       �	�aqYc�A�!*

losss=Y��q       �	)�aqYc�A�!*

lossS"	<���       �	�LbqYc�A�!*

loss,�
<7T       �	��bqYc�A�!*

loss���<�7        �	)xcqYc�A�!*

loss&r�;]�x�       �	
.dqYc�A�!*

loss�:=��       �	E�dqYc�A�!*

loss�i=>���       �	��eqYc�A�!*

lossq�k<�XU�       �	��fqYc�A�!*

loss��<�o��       �	��gqYc�A�!*

loss��;�6�8       �	�hqYc�A�!*

loss�i�;*x��       �	O�hqYc�A�!*

loss��<�ن�       �	�\iqYc�A�!*

lossW� =��\       �	� jqYc�A�!*

loss��=��       �	�kqYc�A�"*

loss�|�=TGP       �	��kqYc�A�"*

loss�#'=d�i�       �	�=lqYc�A�"*

loss/�"=����       �	o�lqYc�A�"*

loss���;�u��       �	3kmqYc�A�"*

loss���;��c       �	�nqYc�A�"*

lossX�=볮�       �	�\oqYc�A�"*

lossVh=;#o�       �	�oqYc�A�"*

lossS3]<b��       �	��pqYc�A�"*

lossNN�<��Q�       �	T5qqYc�A�"*

lossn�s<�R-       �	B�qqYc�A�"*

lossr<O��)       �	�hrqYc�A�"*

loss�$J<��A�       �	��rqYc�A�"*

lossY	=�F�w       �	S�sqYc�A�"*

loss��4<b�S        �	
1tqYc�A�"*

loss@�=��ֱ       �	�tqYc�A�"*

loss{ȃ<�%�L       �	�euqYc�A�"*

loss/�;�|n�       �	 vqYc�A�"*

loss�<��c       �	{�vqYc�A�"*

loss�=8T��       �	zRwqYc�A�"*

loss�j"=��+5       �	��wqYc�A�"*

loss�?<�)�       �	0�xqYc�A�"*

loss��H=��7       �	2"yqYc�A�"*

loss�_�;	�c�       �	/�yqYc�A�"*

lossq�<�n��       �	[zqYc�A�"*

loss��|=����       �	_�zqYc�A�"*

loss@�$=�fJ/       �	�{qYc�A�"*

lossCu�=�h�g       �	�$|qYc�A�"*

loss��&<�ǉh       �	��|qYc�A�"*

loss폠:I��       �	U}qYc�A�"*

loss&��<G�       �	��}qYc�A�"*

lossoiL<����       �	�~qYc�A�"*

lossŠU<���       �	�BqYc�A�"*

lossyD<��U�       �	S�qYc�A�"*

loss���<<��       �	ڀqYc�A�"*

loss�=�3�       �	;p�qYc�A�"*

loss��<��%B       �	�C�qYc�A�"*

loss��q=}�h       �	'ۂqYc�A�"*

loss`�3=��e�       �	&r�qYc�A�"*

loss_,H=)-q�       �	�qYc�A�"*

lossi��<Ԓ�       �	6��qYc�A�"*

loss;��       �	�h�qYc�A�"*

loss��<��m�       �	:�qYc�A�"*

loss��<X��       �	���qYc�A�"*

loss}��<�~�       �	�F�qYc�A�"*

loss �;�6[O       �	��qYc�A�"*

loss���=̫c!       �	���qYc�A�"*

loss�d=Fb�       �	L�qYc�A�"*

loss�v"<�$�J       �	��qYc�A�"*

loss���=�s�       �	?��qYc�A�"*

loss$q�;y���       �	 B�qYc�A�"*

lossP<(�       �	֋qYc�A�"*

loss#L/=�i�;       �	#k�qYc�A�"*

loss��<��!'       �	j�qYc�A�"*

loss�p<��i       �	Ѳ�qYc�A�"*

lossc2=�)�       �	\�qYc�A�"*

loss	�<tC       �	��qYc�A�"*

lossz!.=xr@�       �	�ЏqYc�A�"*

lossSE1=�Ec       �	Cq�qYc�A�"*

loss��=^q)�       �	�qYc�A�"*

losse�<�!       �	K��qYc�A�"*

lossѿ�<~<R�       �	^J�qYc�A�"*

lossv�A=}�G�       �	nߒqYc�A�"*

losss��<#���       �	��qYc�A�"*

losső4=�Z�       �	S�qYc�A�"*

loss>�=y�-       �	4�qYc�A�"*

lossd�"=/�@       �	���qYc�A�"*

lossi�<=���       �	|*�qYc�A�"*

loss��=���       �	�ɖqYc�A�"*

loss���<��K�       �	R��qYc�A�"*

loss�2�<G#e�       �	��qYc�A�"*

loss$,A<п1f       �	O��qYc�A�"*

loss�,�=�e�       �	L�qYc�A�"*

loss$�=�֒       �	3�qYc�A�"*

loss�X=K1��       �	Z��qYc�A�"*

loss��}<az��       �	�`�qYc�A�"*

loss���;��o�       �	}�qYc�A�"*

lossh'�<���       �	�qYc�A�"*

loss�V=���       �	%A�qYc�A�"*

lossH�<|���       �	D��qYc�A�"*

loss(|�<��       �	���qYc�A�"*

loss!�M<W�j�       �	�E�qYc�A�"*

loss���<GO�Q       �	��qYc�A�"*

lossmS=S ~       �	��qYc�A�"*

loss�T�<0�p@       �	a��qYc�A�"*

loss�vD<��=&       �	%�qYc�A�"*

loss�6=>N��       �	���qYc�A�"*

loss�=�t       �	V�qYc�A�"*

lossW2<#��       �	���qYc�A�"*

loss�;�;��n       �	77�qYc�A�"*

loss)U�<5���       �	��qYc�A�"*

lossNa\=�-��       �	���qYc�A�"*

loss\��<����       �	oH�qYc�A�"*

loss��%=�>�U       �	�ާqYc�A�"*

loss�ފ<6�v�       �	�{�qYc�A�"*

loss�7�<��y"       �	�qYc�A�"*

lossCeU=��       �	�A�qYc�A�"*

losss��<�ĵ�       �	��qYc�A�"*

lossh�<�vG       �	/��qYc�A�"*

loss��<���       �	-�qYc�A�"*

loss�X=f�U       �	xϬqYc�A�"*

loss���<�v%       �	�u�qYc�A�"*

loss���<�ɠw       �	�.�qYc�A�"*

loss��;�M�       �	�ҮqYc�A�"*

loss�V5<YC~E       �	���qYc�A�"*

losseЀ<���       �	�qYc�A�"*

lossc��<�1.�       �	���qYc�A�"*

lossfw�<Db�       �	L�qYc�A�"*

loss*��=���       �	��qYc�A�"*

lossv��<���       �	W��qYc�A�"*

lossW�=���       �	�?�qYc�A�"*

loss ��<�Ĳ�       �	��qYc�A�"*

lossĸ<K�	       �	��qYc�A�"*

loss�� =ɝ�       �	�+�qYc�A�"*

loss�G�;KiZ       �	͵qYc�A�"*

loss$;=���       �	
e�qYc�A�"*

loss��<��p/       �	� �qYc�A�"*

loss���=D��,       �	f��qYc�A�"*

lossa��<dr�       �	�?�qYc�A�"*

loss��*<�/*       �	vݸqYc�A�"*

lossN_�<���       �	r�qYc�A�"*

loss��I=�<�       �	N�qYc�A�"*

lossځ#=��D�       �	줺qYc�A�"*

loss>�k��       �	�l�qYc�A�"*

loss�ʬ=0���       �	$	�qYc�A�"*

lossSs�=[���       �	&�qYc�A�"*

loss���;��P�       �	|~�qYc�A�"*

loss��>�&i�       �	��qYc�A�"*

loss���<��<�       �	멾qYc�A�#*

loss3��;��&%       �	E�qYc�A�#*

loss��s<�aq�       �	��qYc�A�#*

loss�x�=�N��       �	E��qYc�A�#*

loss,P�<s�       �	L�qYc�A�#*

loss�&�=rw       �	7��qYc�A�#*

losst�<����       �	iU�qYc�A�#*

lossÖE=��8/       �	��qYc�A�#*

loss�٬<(�L       �	���qYc�A�#*

loss��<$W��       �	�'�qYc�A�#*

loss4�;��5�       �	8��qYc�A�#*

lossx�+<��(}       �	~��qYc�A�#*

loss܍i=:K!M       �	&�qYc�A�#*

loss&Q�<ix��       �	0��qYc�A�#*

loss͏R;��       �	Y�qYc�A�#*

loss��><U�       �	 �qYc�A�#*

loss�[�=kgBK       �	��qYc�A�#*

loss�j\<�\�       �	(*�qYc�A�#*

loss�.�<�l	�       �	b��qYc�A�#*

loss�7G=��&�       �	�Z�qYc�A�#*

losse��<X-�       �	���qYc�A�#*

loss��_<Ƭ�       �	5��qYc�A�#*

loss�O�=�G�       �	�m�qYc�A�#*

loss\
<��7       �	�qYc�A�#*

loss�gl=%�}       �	���qYc�A�#*

loss�H=�a��       �	.9�qYc�A�#*

loss.�=1��       �	�&�qYc�A�#*

loss��<J�R       �	�
�qYc�A�#*

loss=.V       �	+��qYc�A�#*

loss��<��%       �	��qYc�A�#*

lossH4<���s       �	���qYc�A�#*

loss��a<#A�       �	���qYc�A�#*

loss�n�<�<y       �	���qYc�A�#*

loss֣3=�Xb�       �	`��qYc�A�#*

loss7z�<���       �	y"�qYc�A�#*

lossN'�<�&       �	Ag�qYc�A�#*

lossl�v=��f)       �	�M�qYc�A�#*

lossVR(<��       �	�Z�qYc�A�#*

lossc��;�~z|       �	�"�qYc�A�#*

lossQm=�τ�       �	���qYc�A�#*

loss7A,<��       �	��qYc�A�#*

loss-��<�z��       �	���qYc�A�#*

loss8�=9S|�       �	��qYc�A�#*

loss�F+=��!       �	4/�qYc�A�#*

loss�s<�\       �	\T�qYc�A�#*

loss�"�=P��       �	���qYc�A�#*

loss?�[=+�       �	ۣ�qYc�A�#*

lossA�<{�a�       �	�=�qYc�A�#*

loss�sB<� �       �	x%�qYc�A�#*

loss�x�<mk�k       �	��qYc�A�#*

lossbM�=��w6       �	y#�qYc�A�#*

loss�KV<���       �	�qYc�A�#*

loss.��<Hf��       �	DP�qYc�A�#*

loss��<T�d       �	�V�qYc�A�#*

loss�=]�#       �	�{�qYc�A�#*

loss-? >^��        �	��qYc�A�#*

losso3�<f�       �	���qYc�A�#*

loss<�<��!       �	�l�qYc�A�#*

loss{m<mR�       �	��qYc�A�#*

loss��_=S)�v       �	-��qYc�A�#*

lossv��<�       �	5]�qYc�A�#*

loss�3<����       �	O�qYc�A�#*

lossM��<�j��       �	���qYc�A�#*

lossO�0=�       �	,H�qYc�A�#*

loss`�.<��$       �	���qYc�A�#*

losseb>Gm�       �	<��qYc�A�#*

lossDdQ<�Y��       �	�-�qYc�A�#*

lossSo�<�AN       �	��qYc�A�#*

loss��X=���O       �	�Z�qYc�A�#*

lossCS�<-��!       �	N��qYc�A�#*

loss�d�<�/Z�       �	���qYc�A�#*

loss��<\ZN       �	j/�qYc�A�#*

lossHY=�X&h       �	�qYc�A�#*

loss8�<Y�k       �	���qYc�A�#*

loss|-�<��
:       �	G;�qYc�A�#*

loss�=����       �	���qYc�A�#*

loss`@t<�3�&       �	r�qYc�A�#*

lossw�<��A�       �	��qYc�A�#*

loss=|!<rc	�       �	���qYc�A�#*

lossCE�<��1�       �	�8�qYc�A�#*

loss:J�<�"��       �	���qYc�A�#*

loss���<�a�	       �	Do�qYc�A�#*

loss��7= ��       �	�qYc�A�#*

loss�-�=��ޫ       �	
��qYc�A�#*

loss�#F=O�V       �	�9�qYc�A�#*

loss�ty=�sÁ       �	���qYc�A�#*

loss��]=D�l       �	�g�qYc�A�#*

loss���<��@.       �	��qYc�A�#*

loss��<F��       �	���qYc�A�#*

loss;_b<��s�       �	='�qYc�A�#*

loss)�b<n��       �	T��qYc�A�#*

loss}G!=�[	B       �	[^ rYc�A�#*

loss�u�;�؋       �	�� rYc�A�#*

loss?��<�"�z       �	�rYc�A�#*

loss�g�<��       �	:ArYc�A�#*

loss���<S�d�       �	3�rYc�A�#*

loss=n߭�       �	xzrYc�A�#*

lossrm�=��_D       �	%rYc�A�#*

loss�=ց^�       �	Q�rYc�A�#*

lossd��=�w       �	�XrYc�A�#*

loss�x�<%`i8       �	�rYc�A�#*

loss6�I<�r�
       �	C�rYc�A�#*

loss��<��'       �	�-rYc�A�#*

loss�_<�_&�       �	��rYc�A�#*

loss��/<�G�=       �	�_rYc�A�#*

loss�s�<���A       �	�rYc�A�#*

loss��<�Kz       �	K�	rYc�A�#*

lossd_o=J���       �	�$
rYc�A�#*

loss���<��D�       �	U�
rYc�A�#*

loss=��<hRjX       �	PTrYc�A�#*

loss��k<�6Y       �	W�rYc�A�#*

loss�0<R���       �	�~rYc�A�#*

loss̷/=ݭ��       �	�rYc�A�#*

loss=��;��0?       �	K�rYc�A�#*

loss�=�Ni�       �	�KrYc�A�#*

lossc/<<}.`~       �	��rYc�A�#*

loss�ۊ=��       �	�urYc�A�#*

loss!��<���]       �	,rYc�A�#*

loss'v�<�^$:       �	��rYc�A�#*

lossv�<Cj�       �	�<rYc�A�#*

loss�R`=���       �	��rYc�A�#*

lossE*�;�       �	yyrYc�A�#*

loss���<NHb�       �	�rYc�A�#*

loss���<Ô�       �	T�rYc�A�#*

lossjm�<�ҳ�       �	<MrYc�A�#*

loss��g<�n.        �	5rYc�A�#*

loss��=���       �	��rYc�A�#*

loss|d	=�*��       �	UrYc�A�#*

losssL=���       �	m rYc�A�$*

lossY=T&3�       �	X�rYc�A�$*

loss���<.c��       �	�NrYc�A�$*

loss&�z=s]oI       �	��rYc�A�$*

lossoO�=@�
�       �	5�rYc�A�$*

lossiB�<kh	E       �	ArYc�A�$*

loss�)=a���       �	��rYc�A�$*

lossA�=� 7       �	mrYc�A�$*

loss*m�=\W^       �	rYc�A�$*

loss�M�;,��.       �	R�rYc�A�$*

loss[<�vKj       �	YrYc�A�$*

loss��=����       �	�rYc�A�$*

loss�B=���       �	��rYc�A�$*

lossѤx<���V       �	2 rYc�A�$*

loss{|/=�C�       �	�� rYc�A�$*

lossDRE=&�;       �	Sx!rYc�A�$*

lossX=rv�       �	�"rYc�A�$*

loss�)=��7�       �	��"rYc�A�$*

loss�D�<��J�       �	,E#rYc�A�$*

loss�#�;��d�       �	�$rYc�A�$*

loss�>��x       �	I�$rYc�A�$*

loss܍=�B��       �	O<%rYc�A�$*

lossqӇ<"*��       �	�%rYc�A�$*

loss�=);��       �	Uk&rYc�A�$*

lossL9�<z�hM       �	J{'rYc�A�$*

loss���;��       �	�(rYc�A�$*

loss�=��C       �	t�(rYc�A�$*

loss�v�< �U�       �	^)rYc�A�$*

loss���<S��a       �	�*rYc�A�$*

lossI�;�o�}       �	h�*rYc�A�$*

loss�<O��^       �	bI+rYc�A�$*

loss��h<;B��       �	,�+rYc�A�$*

loss�)<K�       �	Ő,rYc�A�$*

loss{�
=)��A       �	�8-rYc�A�$*

loss=�=�       �	q�-rYc�A�$*

loss�f=B\��       �	�.rYc�A�$*

loss}�<�m0�       �	�/rYc�A�$*

loss �><��KB       �	��/rYc�A�$*

loss���<�8��       �	q0rYc�A�$*

loss�l�=����       �	T1rYc�A�$*

losss#e=EY8x       �	�1rYc�A�$*

loss�D�<e û       �	%X2rYc�A�$*

loss��<�iy�       �	|�2rYc�A�$*

loss쭶<���       �	 �3rYc�A�$*

lossO�G<�V�	       �	f34rYc�A�$*

loss7�;�]��       �	C�4rYc�A�$*

loss1�=�W�}       �	�`5rYc�A�$*

loss.8.>h��\       �	8�5rYc�A�$*

lossߩ<=�1-�       �	�6rYc�A�$*

lossTs�<���v       �	�<7rYc�A�$*

lossOI<<���v       �	��7rYc�A�$*

loss�<e�        �	�t8rYc�A�$*

loss�e$=�Ձ       �	p"9rYc�A�$*

loss��<. G       �	��9rYc�A�$*

loss��(<;��       �	x_:rYc�A�$*

loss��;����       �	+�:rYc�A�$*

loss}'}=��S�       �	��;rYc�A�$*

loss��<(|ށ       �	�(<rYc�A�$*

loss�a<�\��       �	o�<rYc�A�$*

loss���<��w       �	�Q=rYc�A�$*

loss)8Z<�ǲ�       �	m�=rYc�A�$*

loss֌�<&�o�       �	~>rYc�A�$*

loss�9:��3�       �	�T?rYc�A�$*

loss�7�=N��C       �	d�?rYc�A�$*

loss�י<]��       �	��@rYc�A�$*

loss���<h8       �	�.ArYc�A�$*

loss�=�Q0       �	��ArYc�A�$*

loss�<2S       �	�aBrYc�A�$*

loss�ֹ;~M�       �	7�BrYc�A�$*

loss�:=yL!�       �	�CrYc�A�$*

loss5"=p��<       �	tFDrYc�A�$*

loss�B<�;��       �	U�DrYc�A�$*

loss-�>�֍?       �	�qErYc�A�$*

loss��>f�x       �	�FrYc�A�$*

loss�̊=�l��       �	�FrYc�A�$*

lossvr�<�7�       �	�LGrYc�A�$*

loss�P(<�K�       �	T�GrYc�A�$*

loss�k�=
��       �	�~HrYc�A�$*

loss�ݭ<�z��       �	�IrYc�A�$*

losso�C<�Q��       �	S�IrYc�A�$*

loss��K=�df       �	�JrYc�A�$*

loss)t/=��
       �	�`KrYc�A�$*

loss ~D=r�Jt       �	6LrYc�A�$*

loss�#�<sA�]       �	�MrYc�A�$*

loss�N�=��e*       �	z�MrYc�A�$*

loss���;�(��       �	YOrYc�A�$*

lossc�<=���       �	m�OrYc�A�$*

loss�p�;pd��       �	5~PrYc�A�$*

lossO�<���v       �	�QrYc�A�$*

loss��y=��kr       �	��QrYc�A�$*

loss�k�<�`Ȧ       �	.�RrYc�A�$*

lossI$�<r���       �	�NSrYc�A�$*

loss�^N=���       �	�SrYc�A�$*

loss:V�=l��i       �	g{TrYc�A�$*

loss k_<r$��       �	�$UrYc�A�$*

loss�<=c�ܪ       �	�VrYc�A�$*

loss���;�\b2       �	�VrYc�A�$*

lossW�=wF:�       �	{MWrYc�A�$*

loss�'�<}��       �	��WrYc�A�$*

loss71Z<5jQ�       �	+�XrYc�A�$*

lossi��;�O�q       �	�$YrYc�A�$*

lossqם<��5�       �	D�YrYc�A�$*

loss���<Z+{       �	J]ZrYc�A�$*

losstr<C�       �	(�ZrYc�A�$*

loss [#=q�       �	��[rYc�A�$*

loss1v�;� ?�       �	S!\rYc�A�$*

loss��Z=�:       �	�\rYc�A�$*

lossXG�<X���       �	3P]rYc�A�$*

loss�$	=�{       �	C^rYc�A�$*

loss!�!=e-�<       �	�^rYc�A�$*

loss�}H<K��F       �	h_rYc�A�$*

lossJ�<�EB       �	��_rYc�A�$*

loss<)�=���       �	��`rYc�A�$*

lossU;Y͸0       �	*arYc�A�$*

loss�['=����       �	.�arYc�A�$*

lossȀ;���~       �	[brYc�A�$*

lossN�K<�X3�       �	o�brYc�A�$*

loss48=�i�u       �	ΊcrYc�A�$*

lossd��<��H�       �	�drYc�A�$*

lossϐ�<��U       �	"6erYc�A�$*

loss���<B��       �	��erYc�A�$*

lossv�_=J��e       �	�_frYc�A�$*

lossTxS<�{�       �	0�frYc�A�$*

loss�c�<r7�       �	�grYc�A�$*

loss+^=��J       �	H1hrYc�A�$*

loss� 9<�q�5       �	O�hrYc�A�$*

loss�v;6�       �	�rirYc�A�$*

lossN�=<��x�       �	jrYc�A�$*

lossa�X;�s��       �	=�jrYc�A�%*

loss�$<%�\�       �	~WkrYc�A�%*

loss��C<c�(       �	�lrYc�A�%*

loss�	v;��|l       �	`�lrYc�A�%*

loss=��<@�{q       �	�GmrYc�A�%*

loss �:y޺)       �	��mrYc�A�%*

loss7��9ց85       �	]�nrYc�A�%*

loss!R;1���       �	}"orYc�A�%*

lossdm;M'�~       �	��orYc�A�%*

lossM��<-�u�       �	nprYc�A�%*

lossx�<D��       �	�qrYc�A�%*

loss]BX;�1��       �	O�qrYc�A�%*

lossɋ;�ڄ_       �	�FrrYc�A�%*

loss���="<��       �	K�rrYc�A�%*

loss�,9<[Ka�       �	��srYc�A�%*

loss\.;LOi       �	�GtrYc�A�%*

loss��7=���       �	��trYc�A�%*

loss��d=       �	ߋurYc�A�%*

loss?�<�N,�       �	�.vrYc�A�%*

lossf�*<��       �	��vrYc�A�%*

lossl1=�k}-       �	F`wrYc�A�%*

loss�rD=d���       �	M�wrYc�A�%*

loss|y=4�ZZ       �	Y�xrYc�A�%*

loss��<⦅�       �	$yrYc�A�%*

loss��'<�ꂏ       �	��yrYc�A�%*

loss�=��hK       �	KVzrYc�A�%*

lossڶ$=4`�~       �	p[{rYc�A�%*

loss��<�(D|       �	b�{rYc�A�%*

lossd�<�A!       �	ޑ|rYc�A�%*

lossG��<HC"<       �	�'}rYc�A�%*

loss�=Z       �	a~rYc�A�%*

lossqK�<ҝA�       �	%�~rYc�A�%*

lossd��<Q�9i       �	wKrYc�A�%*

loss�F�<f��p       �	�rYc�A�%*

loss:�<@��       �	��rYc�A�%*

loss�S&=i^�b       �	��rYc�A�%*

loss��L<?��}       �	��rYc�A�%*

loss4�<e�s�       �	M�rYc�A�%*

loss��<U��A       �	�rYc�A�%*

loss��;)���       �	ǁ�rYc�A�%*

loss�<�aT       �	�w�rYc�A�%*

loss���<���       �	��rYc�A�%*

loss�#�=X�Ӏ       �	ҧ�rYc�A�%*

loss�M~=����       �	�G�rYc�A�%*

loss�G<�?�V       �	��rYc�A�%*

loss�þ<�;�J       �	Ő�rYc�A�%*

loss��1=��=Y       �	jj�rYc�A�%*

loss�q�;W�2       �	�rYc�A�%*

loss�<�N�r       �	o��rYc�A�%*

loss�Ʃ<��r       �	L6�rYc�A�%*

loss�;Ӛ��       �	;�rYc�A�%*

loss5�<vw+Z       �	Ί�rYc�A�%*

losso�}=���a       �	��rYc�A�%*

lossC�<�Vy�       �	��rYc�A�%*

loss���<���       �	E��rYc�A�%*

loss
�<,-Q       �	.U�rYc�A�%*

loss�A<c��       �	��rYc�A�%*

loss�g�<�_�O       �	ׅ�rYc�A�%*

losstq�;�4       �	��rYc�A�%*

loss��<d1KK       �	J��rYc�A�%*

losss�u=��U       �	�u�rYc�A�%*

loss���<VJEv       �	�rYc�A�%*

loss{h<^�       �	���rYc�A�%*

loss��Z<��b�       �	�F�rYc�A�%*

loss�=$=�b       �	��rYc�A�%*

loss%O�<���       �	��rYc�A�%*

loss�@=�-Y       �	�|�rYc�A�%*

loss��=s"Le       �	[%�rYc�A�%*

lossOZ=� �       �	���rYc�A�%*

loss��=0�L�       �	�X�rYc�A�%*

loss�9=ώ�       �	�rYc�A�%*

loss<�
=l���       �	���rYc�A�%*

loss�T<[��o       �	�U�rYc�A�%*

loss�: =�U!k       �	5��rYc�A�%*

loss�=%�3       �	�ƲrYc�A�%*

loss��<^EG       �	$^�rYc�A�%*

lossn�.=E0�       �	���rYc�A�%*

loss�.=���       �	���rYc�A�%*

loss��=�� �       �	F$�rYc�A�%*

lossE�<�$��       �	K�rYc�A�%*

loss#�C<]5�       �	&��rYc�A�%*

loss�݉<i@}n       �	���rYc�A�%*

loss��.=���       �	'3�rYc�A�%*

loss<p��       �	)�rYc�A�%*

lossa&�<H|y7       �	�źrYc�A�%*

loss*�<��.e       �	���rYc�A�%*

loss�K�<�En�       �	R�rYc�A�%*

lossN��<�v       �	~��rYc�A�%*

loss�=WXZ       �	̚�rYc�A�%*

loss���<k�ag       �	�4�rYc�A�%*

loss�x<�!�       �	%˾rYc�A�%*

loss��=���       �	�k�rYc�A�%*

loss�A<y�       �	e��rYc�A�%*

loss�='K�       �	j��rYc�A�%*

loss�y�<�Z��       �	���rYc�A�%*

loss�SK<��r�       �	P�rYc�A�%*

loss-w<���       �	˹�rYc�A�%*

loss���<5��       �	�P�rYc�A�%*

loss� :='�s�       �	S��rYc�A�%*

loss�Kj<:��       �	���rYc�A�%*

loss@�B=����       �	_&�rYc�A�%*

loss��;�1       �	b��rYc�A�%*

loss��<�Ύ       �	)��rYc�A�%*

lossQ��=wРH       �	1�rYc�A�%*

loss脔=��N�       �	?��rYc�A�%*

lossET�<���       �	�Y�rYc�A�%*

loss3�<�[�       �	���rYc�A�%*

loss��<n��       �	ׅ�rYc�A�%*

loss�f<zY�~       �	��rYc�A�%*

lossv�<R,��       �	k��rYc�A�%*

loss�p=\^1�       �	"O�rYc�A�%*

loss��<���       �	.�rYc�A�%*

loss.`<U��       �	[��rYc�A�%*

loss���<����       �	�q�rYc�A�%*

loss�c:+�ΰ       �	��rYc�A�%*

loss
�;�B       �	���rYc�A�%*

loss�/W=V��       �	i �rYc�A�%*

loss��= ��v       �	��rYc�A�%*

loss��>GǊ       �	k�rYc�A�%*

loss�.<��be       �	{0�rYc�A�%*

lossÕk:�(       �	{�rYc�A�%*

loss���9�Q�       �	�R�rYc�A�%*

loss�s<�U7       �	�rYc�A�%*

lossXNs<u �       �	�]�rYc�A�%*

loss'=�&ǈ       �	��rYc�A�%*

loss�1�=6(�@       �	0F�rYc�A�%*

loss]��;@��       �	���rYc�A�%*

loss��R<����       �	*��rYc�A�%*

loss��<��r       �	���rYc�A�&*

lossq��<q���       �	o��rYc�A�&*

loss�,<)`�       �	�m�rYc�A�&*

loss
~�<��״       �	�rYc�A�&*

loss��f<r�'�       �	��rYc�A�&*

lossTG�=~U�       �	d��rYc�A�&*

loss^�=�       �	��rYc�A�&*

loss���<��)�       �	9�rYc�A�&*

loss��<Q!��       �	}��rYc�A�&*

loss{�=|%X�       �	�_�rYc�A�&*

lossC�Y<���       �	81�rYc�A�&*

loss�v/=�1z       �	!��rYc�A�&*

loss>�<s��       �	�a�rYc�A�&*

loss��;�Ug�       �	b��rYc�A�&*

lossf�<&V       �	s��rYc�A�&*

loss�$�;b\
~       �	�r�rYc�A�&*

loss�8�<Ŷj       �	�9�rYc�A�&*

lossD�l=y�uI       �	��rYc�A�&*

loss�qL<�Z�       �	���rYc�A�&*

loss%y=��*       �	 z�rYc�A�&*

loss��:H�       �	��rYc�A�&*

lossͼ<�*]�       �	ɮ�rYc�A�&*

lossvH�;M�'       �	�@�rYc�A�&*

loss�=�<��[�       �	���rYc�A�&*

loss�DK<�oM       �	���rYc�A�&*

loss�(<��[T       �	|*�rYc�A�&*

loss��<���       �	M��rYc�A�&*

loss��w=��`�       �	2u�rYc�A�&*

loss���<�4�       �	��rYc�A�&*

loss��<�3Y�       �	���rYc�A�&*

loss%�<qr�?       �	�A�rYc�A�&*

lossi��=(�<       �	>��rYc�A�&*

loss�~;b#��       �	��rYc�A�&*

loss�K=�ݕ�       �	�9�rYc�A�&*

lossL�"<�Z�       �	a��rYc�A�&*

lossSI�=x��       �	}�rYc�A�&*

loss�oE<a-W�       �	��rYc�A�&*

loss�n<�@�       �	���rYc�A�&*

loss2�<�)��       �	v�rYc�A�&*

loss�u<��0�       �	��rYc�A�&*

loss��%<Ӻ�       �	���rYc�A�&*

loss��<<a|�       �	1?�rYc�A�&*

loss��z=���       �	���rYc�A�&*

loss���;ռ�       �	mo�rYc�A�&*

loss-<F�L1       �	�F�rYc�A�&*

lossd��=���       �	���rYc�A�&*

loss��`<�n       �	���rYc�A�&*

loss��=�Xѫ       �	��rYc�A�&*

loss��v<ҫ��       �	ܷ�rYc�A�&*

loss!�<�ke       �	�K�rYc�A�&*

loss���;i        �	��rYc�A�&*

loss@V<�&A       �	1��rYc�A�&*

loss�H�;��?       �	�V�rYc�A�&*

loss�J�=�2�T       �	���rYc�A�&*

loss�Z@=���       �	���rYc�A�&*

loss|g=H�3       �	< sYc�A�&*

loss)k=�D�K       �	�� sYc�A�&*

loss�y�<��^       �	whsYc�A�&*

lossg	=� �       �	�
sYc�A�&*

loss���<]vc       �	��sYc�A�&*

loss_�I<>d��       �	8/sYc�A�&*

loss8�;0Y�       �	/�sYc�A�&*

loss�.�<�4bh       �	�VsYc�A�&*

loss�<aw��       �	�/sYc�A�&*

loss�b*=�Y��       �	��sYc�A�&*

loss��*=FB�       �	��sYc�A�&*

loss��<ƤDN       �	8sYc�A�&*

loss&�.<j�       �	d�sYc�A�&*

lossWM=q9�       �	�tsYc�A�&*

loss>�<c j        �	�	sYc�A�&*

loss� �;�_%�       �	��	sYc�A�&*

loss`�v=��?�       �	�Z
sYc�A�&*

losst0w=M���       �	�*sYc�A�&*

loss�P<��u9       �	#�sYc�A�&*

loss�R[<�d�       �	��sYc�A�&*

lossU<��k       �	'�sYc�A�&*

loss���;���       �	�lsYc�A�&*

loss��;UD�n       �	�sYc�A�&*

loss��<��^       �	H�sYc�A�&*

loss�ū<��WS       �	�	sYc�A�&*

loss���;�Za       �	�sYc�A�&*

loss��`=·�{       �	�;sYc�A�&*

loss��=c:a       �	+�sYc�A�&*

lossd��<��&�       �	=�sYc�A�&*

lossD<�'�       �	�msYc�A�&*

loss3�4;��.�       �	�?sYc�A�&*

loss���<�S3       �	��sYc�A�&*

loss��X<��_�       �	=}sYc�A�&*

loss�s}=��F       �	jsYc�A�&*

loss��3=���       �	��sYc�A�&*

lossf��<3�F       �	SsYc�A�&*

loss��F=���3       �	A�sYc�A�&*

loss,o�;ސ_       �	��sYc�A�&*

loss EK<��v4       �	�'sYc�A�&*

loss�Y=�v�       �	2�sYc�A�&*

lossH�!=��fq       �	�bsYc�A�&*

loss�\n<�g�       �	��sYc�A�&*

loss]~8=8�׮       �	:�sYc�A�&*

loss�$�<�Y�       �	�fsYc�A�&*

loss�3<1
�1       �	� sYc�A�&*

loss�O�<���e       �	?�sYc�A�&*

loss,Vn==6�       �	kD sYc�A�&*

loss���<��6/       �	W�!sYc�A�&*

lossQ�<�"�       �	us"sYc�A�&*

loss���<��k       �	�#sYc�A�&*

lossl�<
�8�       �	*�#sYc�A�&*

lossh�=�
~�       �	1B$sYc�A�&*

losss��<=Qa�       �	��$sYc�A�&*

lossS��;�Ԩp       �	�s%sYc�A�&*

loss���:u�       �	>	&sYc�A�&*

loss��e=�
��       �	ϡ&sYc�A�&*

loss�c�;O, {       �	�7'sYc�A�&*

loss]<�*�       �	��'sYc�A�&*

loss�Y8=Kvf       �	��(sYc�A�&*

loss���=W��       �	+M)sYc�A�&*

loss�p�<��P       �	��)sYc�A�&*

loss�(�<Hh�       �	=~*sYc�A�&*

loss�Ai;IwG�       �	+sYc�A�&*

loss��=%Bu       �	��+sYc�A�&*

loss��=�2��       �	�L,sYc�A�&*

loss�/|<Ң�l       �	T�,sYc�A�&*

loss�X=ڝb�       �	�-sYc�A�&*

lossI��;����       �	".sYc�A�&*

lossOs�=��^       �	�.sYc�A�&*

loss�G`=ē$       �	�M/sYc�A�&*

lossS�;=��1Z       �	.�/sYc�A�&*

lossߣ�<��(�       �	t0sYc�A�&*

loss�/�=��t       �	1sYc�A�&*

loss4�<��       �	`�1sYc�A�'*

loss���<��zt       �	�R2sYc�A�'*

loss�<T2�       �	:�2sYc�A�'*

loss�o�;���B       �	�3sYc�A�'*

lossJ[w<&y��       �	�4sYc�A�'*

loss��\=��R       �	� 5sYc�A�'*

loss4Z==��W       �	̗5sYc�A�'*

loss��<m�2�       �	�.6sYc�A�'*

lossdb'=<���       �	��6sYc�A�'*

lossF�]=�
�       �	�z7sYc�A�'*

loss�"1=Eӟu       �	�&8sYc�A�'*

loss�i�==dF�       �	��8sYc�A�'*

loss-��<ꠅ�       �	�Y9sYc�A�'*

loss,ث;{�K       �	�9sYc�A�'*

loss�}�<�F �       �	��:sYc�A�'*

loss�ą=����       �	/3;sYc�A�'*

loss��h=�R�       �	��;sYc�A�'*

lossmE=݈�A       �	e<sYc�A�'*

loss��=g�/h       �	��<sYc�A�'*

loss|<�@��       �	��=sYc�A�'*

loss�k�<#�k       �	\:>sYc�A�'*

loss�6�<����       �	��>sYc�A�'*

loss�(<��S�       �	�h?sYc�A�'*

loss�9=����       �	�@sYc�A�'*

loss��V=�H?       �	��@sYc�A�'*

lossOy+<k�IQ       �	5AsYc�A�'*

loss��
=��       �	��AsYc�A�'*

loss46�=t�,�       �	hBsYc�A�'*

lossjY=[�       �	yCsYc�A�'*

lossQ��;ZҬ�       �	��CsYc�A�'*

lossZ=�׵�       �	�(DsYc�A�'*

lossCN=^���       �	��DsYc�A�'*

lossH��=g�u�       �	�[EsYc�A�'*

loss=��=��y�       �	h#FsYc�A�'*

lossZ�A=���       �	P�FsYc�A�'*

loss���<�o�       �	cGsYc�A�'*

loss��<�@�       �	��GsYc�A�'*

lossO�<u-M       �	��HsYc�A�'*

loss~�=5}�d       �	#IsYc�A�'*

lossh&<f���       �	��IsYc�A�'*

loss��,<��       �	taJsYc�A�'*

loss+�<�Ԍ       �	!KsYc�A�'*

loss!�i<�h+       �	��KsYc�A�'*

loss��=�{�       �	Y2LsYc�A�'*

lossL�<��Ǧ       �	2�LsYc�A�'*

lossl-�<w��       �	D�MsYc�A�'*

loss.RN<�,B       �	�)NsYc�A�'*

loss��=���       �	��NsYc�A�'*

loss	,�;���        �	kbOsYc�A�'*

loss#QF<�|�J       �	w�OsYc�A�'*

loss_?d=2x�       �	H�PsYc�A�'*

lossm��<g1�e       �	�@QsYc�A�'*

loss���<�7       �	��QsYc�A�'*

losse�<h���       �	}xRsYc�A�'*

lossdڡ<ҸX
       �	�SsYc�A�'*

loss��<�	<       �	`�SsYc�A�'*

loss ��;��e�       �	�DTsYc�A�'*

loss�`=9�^�       �	@�TsYc�A�'*

loss�U<<���8       �	�rUsYc�A�'*

loss��={.�       �	�	VsYc�A�'*

loss�&S=��       �	0�VsYc�A�'*

loss�E=��)w       �	��WsYc�A�'*

loss���;�D�        �	�XsYc�A�'*

loss6�T<pym�       �	�XsYc�A�'*

loss�<�i       �	�PYsYc�A�'*

losse=We       �	��YsYc�A�'*

lossQg<.3�@       �	ԁZsYc�A�'*

loss/�;T���       �	�[sYc�A�'*

loss��<�z�       �	�	\sYc�A�'*

lossw�/<��tU       �	�\sYc�A�'*

loss��<�Fs�       �	�8]sYc�A�'*

lossn��<�`b�       �	F�]sYc�A�'*

loss�k=p[��       �	�e^sYc�A�'*

loss��<�k�       �	Y�^sYc�A�'*

loss�0�;�D�       �	_�_sYc�A�'*

loss�x�=��       �	q`sYc�A�'*

loss\��<����       �	#asYc�A�'*

loss�f=����       �	6�asYc�A�'*

lossRq�<�I�       �	�FbsYc�A�'*

lossư�<fv�&       �	��bsYc�A�'*

loss4�4<U;��       �	XucsYc�A�'*

loss�]C=z�       �	�dsYc�A�'*

lossd�D<ڴ��       �	�dsYc�A�'*

lossh�=A^}w       �	*ResYc�A�'*

loss_�<�5�"       �	WCfsYc�A�'*

loss��<$\       �	8�fsYc�A�'*

loss g�;�]p�       �	T5hsYc�A�'*

lossa�=�p��       �	�hsYc�A�'*

lossi�=�{�       �	��isYc�A�'*

loss�<^���       �	;�jsYc�A�'*

losss�g;�_R       �	ۉksYc�A�'*

loss���<��6R       �	!lsYc�A�'*

loss*�=v��       �	��lsYc�A�'*

loss���<�Yt       �	:WmsYc�A�'*

loss�\�=��8J       �	R�msYc�A�'*

loss�k	=���       �	��nsYc�A�'*

lossæ=X�6       �	aTosYc�A�'*

lossߑ==I&`       �	��osYc�A�'*

lossd�O<��y       �	h�psYc�A�'*

loss��;�[�       �	t)qsYc�A�'*

loss./�<ڍ��       �	1�qsYc�A�'*

lossTG<�_�       �	SwrsYc�A�'*

lossn ;;�F�       �	\;ssYc�A�'*

loss���;=��       �	��ssYc�A�'*

loss���<���       �	�tsYc�A�'*

lossYk<�^Ox       �	6usYc�A�'*

loss�#�<*z��       �	��usYc�A�'*

loss��7=�4��       �	pvsYc�A�'*

lossj��<�ޞ       �	�DwsYc�A�'*

lossy�<���/       �	��wsYc�A�'*

loss�D�<夕�       �	M�xsYc�A�'*

lossN�j<�{<       �	�ysYc�A�'*

loss��<�Y��       �	�ysYc�A�'*

lossIb=�K�8       �	�`zsYc�A�'*

lossmI>=�e�       �	��zsYc�A�'*

lossP<bզ�       �	�{sYc�A�'*

lossi��<,��       �	}?|sYc�A�'*

lossEs[<}i�       �	�1}sYc�A�'*

loss6\�;l		�       �	��}sYc�A�'*

loss��\=-Յo       �	Nc~sYc�A�'*

loss�p=i31�       �	7�~sYc�A�'*

loss��<̀��       �	@�sYc�A�'*

loss�4|<&�I       �	��sYc�A�'*

loss��<L�p,       �	8��sYc�A�'*

loss4�=&]��       �	�&�sYc�A�'*

loss�ܭ;�bY�       �	S΂sYc�A�'*

loss-%�<%�       �	Ig�sYc�A�'*

loss6�=�[<       �	u �sYc�A�'*

loss�/E<E�M�       �	ؚ�sYc�A�(*

lossd͌<��p�       �	3�sYc�A�(*

loss,=����       �	ӅsYc�A�(*

loss$�V=3�g�       �	ڭ�sYc�A�(*

lossѳ=�MW       �	�s�sYc�A�(*

loss�^=���       �	��sYc�A�(*

loss ��=P+Gm       �	���sYc�A�(*

losse��<S���       �	�{�sYc�A�(*

loss�l< ^�n       �	��sYc�A�(*

loss�7�<AN��       �	^��sYc�A�(*

loss�6=�{�       �	�S�sYc�A�(*

loss)΀<kU�       �	E��sYc�A�(*

lossڨ�<���       �	"��sYc�A�(*

loss��1<�T�       �	I-�sYc�A�(*

lossFE�<[�_�       �	U��sYc�A�(*

loss��,=pg�[       �	+܎sYc�A�(*

lossDB<����       �	���sYc�A�(*

loss�"�<�ש�       �	OZ�sYc�A�(*

lossA�<��       �	��sYc�A�(*

loss���<�k��       �	s0�sYc�A�(*

loss�;���        �	l̒sYc�A�(*

loss�';�?	=       �	�k�sYc�A�(*

loss!kM=ܚ�z       �	��sYc�A�(*

loss\J=S��r       �	��sYc�A�(*

lossȏe<f�u       �	���sYc�A�(*

loss���=P4&�       �	�g�sYc�A�(*

loss,��<:�zz       �	>!�sYc�A�(*

lossME0=�`��       �	�ԗsYc�A�(*

loss_�<x5       �	���sYc�A�(*

lossJ�?=R�(�       �	�T�sYc�A�(*

loss�;<�N       �	��sYc�A�(*

lossr4,;P��x       �	���sYc�A�(*

loss�a�<��-d       �	�?�sYc�A�(*

lossҙ�<���       �	�ޛsYc�A�(*

loss=��<�(�       �	x��sYc�A�(*

loss��=��<R       �	4�sYc�A�(*

loss�<L�+�       �	�ΝsYc�A�(*

loss6%<��HA       �	Wx�sYc�A�(*

loss���<�       �	�Q�sYc�A�(*

loss �@<5�;u       �	\�sYc�A�(*

loss���</pP�       �	��sYc�A�(*

loss�[�<r�w"       �	��sYc�A�(*

loss�p�<�.i       �	W��sYc�A�(*

lossK�<u%��       �	���sYc�A�(*

loss�;:=�0u�       �	C9�sYc�A�(*

loss7*�=%P��       �	�ԣsYc�A�(*

loss,�<05�2       �	�n�sYc�A�(*

lossq;�=@��       �	��sYc�A�(*

loss$n�<x7�W       �	���sYc�A�(*

loss =˾�       �	D�sYc�A�(*

lossÞN<'�       �	�ܦsYc�A�(*

loss�1B=��       �	��sYc�A�(*

loss�dw<���       �	�T�sYc�A�(*

lossZq�=�R6�       �	�J�sYc�A�(*

loss<e<X�8�       �	���sYc�A�(*

loss�<���       �	䟪sYc�A�(*

loss_��<;�~�       �	�<�sYc�A�(*

loss�uO=)'	�       �	c�sYc�A�(*

loss��<=����       �	��sYc�A�(*

loss`��<��o       �	+��sYc�A�(*

lossp��<�\��       �	�2�sYc�A�(*

loss	�}<%�U       �	BѮsYc�A�(*

loss�M�<4��?       �	�n�sYc�A�(*

lossW�<�HI"       �	�	�sYc�A�(*

loss�>C<�c       �	ڭ�sYc�A�(*

loss�=�;�w�u       �	vQ�sYc�A�(*

loss�P<+�#	       �	�sYc�A�(*

loss߾~=�M�s       �	���sYc�A�(*

loss`�F=��i�       �	�D�sYc�A�(*

loss|C<N�       �	?�sYc�A�(*

loss��<�1Ռ       �	Ɖ�sYc�A�(*

lossf�;H�|Q       �	O͵sYc�A�(*

loss�	=X�9       �	[x�sYc�A�(*

loss�< �3       �	��sYc�A�(*

loss�Rx=e;�       �	�зsYc�A�(*

loss�z=9�       �	6w�sYc�A�(*

loss�T}=w�
       �	��sYc�A�(*

loss$_T<���       �	^��sYc�A�(*

lossU��<�_An       �	Y�sYc�A�(*

loss�Yl<��/P       �	,�sYc�A�(*

loss��<'       �	��sYc�A�(*

loss���;�&�       �	�)�sYc�A�(*

loss��<��d       �	iȼsYc�A�(*

lossf�h=a�$�       �	�r�sYc�A�(*

loss��<�Mvl       �	4�sYc�A�(*

loss;��<�xC3       �	Z��sYc�A�(*

loss�2=-	w�       �	�Q�sYc�A�(*

loss�~=�,��       �	��sYc�A�(*

loss:�=O���       �	a��sYc�A�(*

loss�l�<�Ѭ�       �	@2�sYc�A�(*

loss�1=B���       �	���sYc�A�(*

lossX~�<\9Ԗ       �	��sYc�A�(*

lossQ�L=S�a       �	�=�sYc�A�(*

loss��d= ��       �	��sYc�A�(*

lossyv�=�$       �	%x�sYc�A�(*

loss��K=����       �	"�sYc�A�(*

loss��<&}��       �	���sYc�A�(*

loss6�z<Xbs       �	�T�sYc�A�(*

lossn�6<0��       �	���sYc�A�(*

loss<��<��       �	��sYc�A�(*

loss�E=�.��       �	�/�sYc�A�(*

loss�I�<>,�       �	^��sYc�A�(*

loss���=�yDs       �	
��sYc�A�(*

lossʦ;ԁs       �	�%�sYc�A�(*

loss�Pt=ݫ       �	��sYc�A�(*

loss��=�-*       �	���sYc�A�(*

loss4�=�So       �	�!�sYc�A�(*

loss�}T<���       �	3��sYc�A�(*

losslIc=��+�       �	���sYc�A�(*

loss�:9=�2^�       �	�&�sYc�A�(*

loss$�9<��       �	��sYc�A�(*

loss,&�=����       �	�g�sYc�A�(*

lossr��<'x       �	��sYc�A�(*

loss�S2<��%�       �	t��sYc�A�(*

loss��<RB]r       �	���sYc�A�(*

loss���<��u       �	Z��sYc�A�(*

lossޗ�<Ŋl�       �	 c�sYc�A�(*

loss���;�m��       �	�M�sYc�A�(*

loss��&=�3KU       �	���sYc�A�(*

loss��;��L�       �	���sYc�A�(*

losszM*<���       �	E�sYc�A�(*

loss���< ��       �	���sYc�A�(*

loss|�d=Sm�8       �	ı�sYc�A�(*

loss���<�c�1       �	H��sYc�A�(*

loss�<��GO       �	gc�sYc�A�(*

loss|E�;�}�:       �	��sYc�A�(*

loss�<=:R��       �	��sYc�A�(*

loss6��=m�s�       �	�\�sYc�A�(*

loss�3<���j       �	��sYc�A�)*

lossl��<[b�       �	��sYc�A�)*

loss�-=��?�       �	,F�sYc�A�)*

loss�u�<@qd�       �	��sYc�A�)*

loss���;LJ�       �	Օ�sYc�A�)*

loss���;�K��       �	�/�sYc�A�)*

loss��<�ӟ�       �	*��sYc�A�)*

loss��=�A-�       �	'��sYc�A�)*

loss�R=�';�       �	�%�sYc�A�)*

loss��<���        �	z��sYc�A�)*

loss̗�;�$a�       �	�f�sYc�A�)*

loss�:n;j��       �	}�sYc�A�)*

loss
O=��X       �	ɫ�sYc�A�)*

lossCV<���       �	�D�sYc�A�)*

loss]q<�d2       �	#��sYc�A�)*

loss�o�;V�S�       �	yv�sYc�A�)*

loss#+=�Uf�       �	 �sYc�A�)*

loss0U<����       �	L��sYc�A�)*

lossҞ+=l�M�       �	K?�sYc�A�)*

loss�	=���       �	+��sYc�A�)*

loss�<<��q       �	R|�sYc�A�)*

loss�>l;WΟ�       �	M�sYc�A�)*

loss�y�;�b�       �	¿�sYc�A�)*

loss�<(m�]       �	�W�sYc�A�)*

lossM�<�E �       �	!�sYc�A�)*

loss �R=p��`       �	���sYc�A�)*

loss-�<�^��       �	�a�sYc�A�)*

loss�WG<�Pu<       �	��sYc�A�)*

loss}B]<�+$�       �	��sYc�A�)*

lossZ7�:q�V�       �	��sYc�A�)*

lossXhR<(�<�       �	���sYc�A�)*

loss�|�<>�L5       �	�3�sYc�A�)*

loss<,�=LǇ�       �	�!�sYc�A�)*

lossD�=B���       �	w��sYc�A�)*

lossD%�<��0l       �		��sYc�A�)*

lossfy�<�	c�       �	�Z�sYc�A�)*

loss�1G=F��       �	���sYc�A�)*

lossH�u=H��       �	���sYc�A�)*

lossU�;T�       �	�)�sYc�A�)*

loss��N;0B��       �	?��sYc�A�)*

lossѾS=���x       �	1|�sYc�A�)*

loss��1<?�O       �	#�sYc�A�)*

lossF�=��]       �	���sYc�A�)*

loss8{�<��.       �	(G�sYc�A�)*

loss��<��I;       �	���sYc�A�)*

loss��r;3:5?       �	�}�sYc�A�)*

loss���<c
A#       �	7�sYc�A�)*

loss`գ<o��       �	���sYc�A�)*

lossE�K<����       �	�Z�sYc�A�)*

lossh@G<�F=       �	o��sYc�A�)*

loss��=��       �	���sYc�A�)*

lossxf�=y��       �	J$�sYc�A�)*

loss�N=P*�v       �	���sYc�A�)*

lossiԇ=���       �	J^�sYc�A�)*

lossJ��;�{T       �	��sYc�A�)*

lossq)F<݈�       �	;��sYc�A�)*

loss�O�;����       �	E tYc�A�)*

loss3�=Oe�k       �	�� tYc�A�)*

lossd��;�|C       �	`wtYc�A�)*

loss��<�C�'       �	jtYc�A�)*

lossl�;O!Q�       �	��tYc�A�)*

lossv�<Ǽ�       �	,GtYc�A�)*

loss><�u�       �	�tYc�A�)*

loss���<���       �	b�tYc�A�)*

lossA��<֕��       �	g'tYc�A�)*

loss��_<+��       �	�tYc�A�)*

losss�w<ʌ2�       �	-_tYc�A�)*

loss�w<}���       �	v�tYc�A�)*

loss��<"���       �	�tYc�A�)*

loss_��<l#A       �	h?tYc�A�)*

loss:�;�?F3       �	�tYc�A�)*

loss�7=���       �	~	tYc�A�)*

lossne�<	��       �	�
tYc�A�)*

loss�z;�-n�       �	�
tYc�A�)*

lossݕ�<�kN�       �	�qtYc�A�)*

lossm<��$�       �	stYc�A�)*

loss�v�;�0��       �	��tYc�A�)*

lossS��=S�0       �	��tYc�A�)*

loss��;)�d*       �	�xtYc�A�)*

loss��<Î��       �	*7tYc�A�)*

loss��<#�       �	�(tYc�A�)*

loss*=�懕       �	��tYc�A�)*

loss 7�<e�       �	�tYc�A�)*

lossN�Q<�1�T       �	��tYc�A�)*

loss�d�<��$j       �	�tYc�A�)*

lossͿ�;´(�       �	��tYc�A�)*

loss*$�:�2�\       �	��tYc�A�)*

loss�3<!s�       �	�CtYc�A�)*

loss�	L;ߖqC       �	�tYc�A�)*

loss��1;B	�u       �	��tYc�A�)*

loss�R<Oۮ       �	�5tYc�A�)*

loss8;�(-5       �	W�tYc�A�)*

lossZ��;���       �	�etYc�A�)*

lossl7;�t�       �	��tYc�A�)*

lossT��9���       �	��tYc�A�)*

loss�K�8-��q       �	�}tYc�A�)*

lossq�1;�[k9       �	1tYc�A�)*

lossL@/<�1)D       �	,�tYc�A�)*

loss���;�u5       �	7ltYc�A�)*

loss�.;�S'       �	�tYc�A�)*

loss=��<���x       �	1�tYc�A�)*

loss:i<�
��       �	�1tYc�A�)*

lossр$;Iͥ_       �	p�tYc�A�)*

loss�a<��pT       �	gc tYc�A�)*

lossLW1=%\       �	�� tYc�A�)*

lossR��<�t'I       �	u�!tYc�A�)*

loss�q<?r�)       �	�+"tYc�A�)*

lossvm<ZicL       �	a�"tYc�A�)*

loss�h�<0uI       �	�d#tYc�A�)*

loss�(G<��Q�       �	J$tYc�A�)*

lossR�<
ɋ1       �	��$tYc�A�)*

loss�B/=7�Xo       �	}?%tYc�A�)*

loss�h;�gB       �	[�%tYc�A�)*

loss�j<<�[AI       �	��&tYc�A�)*

loss���=Y��       �	�C'tYc�A�)*

lossX<���`       �	l�'tYc�A�)*

loss�~�<Rt�-       �	��(tYc�A�)*

loss��\=��*�       �	e4)tYc�A�)*

loss�=�<t�c       �	��)tYc�A�)*

loss�)k<z}       �	��*tYc�A�)*

loss���<ÅR-       �	a�+tYc�A�)*

loss��q<q��       �	�t,tYc�A�)*

lossN�3;e�'�       �	�-tYc�A�)*

loss%]�<�S3       �	8.tYc�A�)*

loss1y�=�m�       �	n�.tYc�A�)*

loss�K;�H�       �	EF/tYc�A�)*

loss�7�;�%L�       �	��/tYc�A�)*

loss�:��c       �	��0tYc�A�)*

loss7u<m�^�       �	�)1tYc�A�**

lossr�Y<�]eh       �	"Q2tYc�A�**

lossj�=KG�       �	F3tYc�A�**

loss�"�=��Hb       �	ӽ3tYc�A�**

loss��<}E��       �	��4tYc�A�**

lossf(=m[c       �	�5tYc�A�**

loss�D<ܡ��       �	��5tYc�A�**

losscrB;�b �       �	!s6tYc�A�**

lossY,=
�D�       �	 7tYc�A�**

loss��<�S�       �	ۤ7tYc�A�**

loss��;�>�       �	�8tYc�A�**

loss�<gՊO       �	�9tYc�A�**

loss�=���j       �	֬9tYc�A�**

loss�< ;$���       �	P�:tYc�A�**

lossv}A<�I�       �	�`;tYc�A�**

loss�;= �x�       �	��;tYc�A�**

lossa<aJJ�       �	��<tYc�A�**

loss��<�^K�       �		3=tYc�A�**

loss�y�<9�K       �	��=tYc�A�**

loss��=w?4       �	�m>tYc�A�**

loss�H�<�f4@       �	�?tYc�A�**

lossmd�;�8�       �	��?tYc�A�**

loss(B�<G�       �	I@tYc�A�**

loss�t;�rb       �	*�@tYc�A�**

loss�[�<r�W�       �	9}AtYc�A�**

loss���;����       �	�(atYc�A�**

loss���<�UK       �	*btYc�A�**

loss��9=��F�       �	��btYc�A�**

lossL��=�d�Y       �	YctYc�A�**

loss���<6�(       �	dtYc�A�**

loss�Z=��        �	�etYc�A�**

lossfO�<��+_       �	�etYc�A�**

loss���<��kc       �	�ftYc�A�**

loss�[=i<	       �	�gtYc�A�**

lossV\=�G��       �	��htYc�A�**

losst$<�V��       �	��itYc�A�**

loss�� =�bc�       �	}=jtYc�A�**

loss��[=p��&       �	�;ktYc�A�**

lossM:t=.t�       �	cdltYc�A�**

lossr�<����       �	wJmtYc�A�**

loss��= ��I       �	�	ntYc�A�**

loss4�;��c       �	��ntYc�A�**

loss�%=o�q       �	�otYc�A�**

loss��	=��o       �	�ptYc�A�**

loss���<ZN�       �	�qtYc�A�**

loss���<���       �	IertYc�A�**

loss?��<����       �	_\stYc�A�**

lossd��;q�_�       �	�NttYc�A�**

loss��<�/܁       �	tButYc�A�**

loss�T�<y�F       �	b�utYc�A�**

loss���<{y�Z       �	 ovtYc�A�**

loss$W�<��B       �	5wtYc�A�**

loss�"�<7�d       �	��wtYc�A�**

loss���<�^��       �	�XxtYc�A�**

loss�D�<�'�       �	(�xtYc�A�**

loss��F<��i}       �	}�ytYc�A�**

loss)��<���*       �	�gztYc�A�**

lossT�y<X��\       �	z�ztYc�A�**

loss@i�=��َ       �	{�{tYc�A�**

loss�L�;��8       �	�`|tYc�A�**

loss���=�)�]       �	��|tYc�A�**

lossXo; �WR       �	��}tYc�A�**

loss���<1EȎ       �	DP~tYc�A�**

loss���=�Z|�       �	��~tYc�A�**

loss8�=C0       �	�}tYc�A�**

loss��/<.&
�       �	j4�tYc�A�**

loss�9�<dtFS       �	�΀tYc�A�**

loss,}�<D�=�       �	e�tYc�A�**

loss�F�<��ky       �	���tYc�A�**

loss�=Es�       �	/��tYc�A�**

loss�A=a�H^       �	�F�tYc�A�**

lossH�=x~M�       �	A�tYc�A�**

loss�Y�<�yj!       �	���tYc�A�**

loss�E�<��b       �	=H�tYc�A�**

loss�	�;�D�z       �	h�tYc�A�**

loss��(<��v       �	���tYc�A�**

lossV��<��xT       �	 C�tYc�A�**

loss��-="��       �	�هtYc�A�**

loss�=x�N        �	"q�tYc�A�**

loss�<��E       �	G�tYc�A�**

losss�;fr	o       �	��tYc�A�**

loss�:4�QV       �	ߋ�tYc�A�**

loss.��:���       �	�$�tYc�A�**

lossQ�'<'�u5       �	���tYc�A�**

lossf<K�WS       �	�W�tYc�A�**

loss�E�<͹P�       �	N�tYc�A�**

loss��;�SS�       �	ׇ�tYc�A�**

loss�@�;�3�       �	 �tYc�A�**

loss�n<`4��       �	���tYc�A�**

lossv�h<f���       �	c^�tYc�A�**

loss&�o;�       �	���tYc�A�**

lossQ�=�ѝ�       �	o��tYc�A�**

lossD��<���)       �	~8�tYc�A�**

loss1�3=;��       �	ґtYc�A�**

loss1��<�3�       �	in�tYc�A�**

losszFq<���       �	�0�tYc�A�**

loss���<gao/       �	`ǓtYc�A�**

loss���<��`       �	yt�tYc�A�**

loss���<���       �	�tYc�A�**

loss;4�<h��h       �	���tYc�A�**

loss#N<��En       �	�B�tYc�A�**

loss�c�<�(^�       �	�ٖtYc�A�**

lossx<!�I�       �	�I�tYc�A�**

loss6v�;��
       �	T�tYc�A�**

lossL�?=�M�       �	��tYc�A�**

loss�L�<�y�,       �	�/�tYc�A�**

loss8N�<�eb       �	�КtYc�A�**

loss�ާ<�O��       �	Yk�tYc�A�**

loss.�<�L�       �	2�tYc�A�**

lossM��<�FT�       �	QݜtYc�A�**

lossJU�; d=�       �	�}�tYc�A�**

loss	�G<���       �	��tYc�A�**

loss*�<�6       �	��tYc�A�**

loss��<�A\{       �	�<�tYc�A�**

loss��h<.�N�       �	E՟tYc�A�**

lossҫ�<�7�       �	�n�tYc�A�**

loss�b�<�y�       �	�tYc�A�**

loss$Q�;��1�       �	���tYc�A�**

loss�<t{ <       �	�>�tYc�A�**

loss1�=�6��       �	�٢tYc�A�**

lossQȔ;�χ�       �	ty�tYc�A�**

loss���<q���       �	��tYc�A�**

loss��<>ܢ�       �	��tYc�A�**

lossGǍ=�C�Z       �	G>�tYc�A�**

loss�+�<�FH       �	�ӥtYc�A�**

loss���<���       �	�r�tYc�A�**

loss⼛;��[       �	[	�tYc�A�**

loss��7<��,<       �	窧tYc�A�**

loss�	#<��<       �	IK�tYc�A�+*

lossX��=J7       �	*�tYc�A�+*

loss��;�/�D       �	{�tYc�A�+*

loss�<*d�       �	��tYc�A�+*

loss�}4<g�<       �	�tYc�A�+*

lossĸr=A?¶       �	���tYc�A�+*

loss��~<q���       �	� �tYc�A�+*

loss �_=^��       �	ػ�tYc�A�+*

loss��;=�WS�       �	*p�tYc�A�+*

loss��	='Ɛ�       �	WA�tYc�A�+*

loss4�!;�Gy       �	�ܮtYc�A�+*

lossh+<�h�U       �	�s�tYc�A�+*

lossI/�;��b�       �	_�tYc�A�+*

lossA��<Tq�       �	e��tYc�A�+*

loss��=�#��       �	�M�tYc�A�+*

loss�N=rm�=       �	
��tYc�A�+*

loss��=(t
�       �	Ū�tYc�A�+*

loss8#�<�8��       �	���tYc�A�+*

loss	�=����       �	�/�tYc�A�+*

lossW��<�9�s       �	�˴tYc�A�+*

loss@b�<
�8       �	�ԵtYc�A�+*

loss��<-f��       �	�p�tYc�A�+*

loss��<�u+,       �	��tYc�A�+*

lossDel<`5��       �	�·tYc�A�+*

loss��<o�Ȏ       �	2r�tYc�A�+*

loss�(=��E       �	��tYc�A�+*

loss��<h��        �	!ȹtYc�A�+*

loss�.T<pe��       �	a��tYc�A�+*

loss�=�="lx�       �	Z/�tYc�A�+*

loss�uA=��_       �	/��tYc�A�+*

loss�;H<�ݓ�       �	���tYc�A�+*

loss!x=���j       �	=,�tYc�A�+*

loss�~s<��       �	R
�tYc�A�+*

loss���<*�ƺ       �	
��tYc�A�+*

loss��=hHu2       �	�Y�tYc�A�+*

loss�ʑ<�d19       �	��tYc�A�+*

loss�m<�Z-       �	���tYc�A�+*

loss���:e@ �       �	�L�tYc�A�+*

lossI��<t�m�       �	V��tYc�A�+*

loss��<Al��       �	���tYc�A�+*

loss�֌=>�Ë       �	T:�tYc�A�+*

loss���<��T3       �	A��tYc�A�+*

loss���<�1=P       �	]p�tYc�A�+*

loss=�<�\c6       �	�	�tYc�A�+*

loss�IJ<��V6       �	U��tYc�A�+*

loss���<���|       �	iT�tYc�A�+*

lossc3=<�ϟ       �	���tYc�A�+*

loss�<=�xn       �	υ�tYc�A�+*

loss�H�=NiE       �	$�tYc�A�+*

loss6�X=+6�       �	8��tYc�A�+*

lossM�i<7�P/       �	Ǟ�tYc�A�+*

lossJX�<�g�w       �	 :�tYc�A�+*

loss�ɻ;t8v       �	���tYc�A�+*

loss&��;'6�D       �	Mj�tYc�A�+*

loss���<ƪ       �	i �tYc�A�+*

loss��=r2�y       �	,��tYc�A�+*

lossTz�;�!"j       �	]4�tYc�A�+*

loss[��<N[�g       �	��tYc�A�+*

loss��<�q�       �	r��tYc�A�+*

loss��9=�h�(       �	�'�tYc�A�+*

loss6.�;9���       �	:��tYc�A�+*

lossEp�<�՗)       �	Tp�tYc�A�+*

lossE�i=l��:       �	V�tYc�A�+*

loss��;Ȝ       �	��tYc�A�+*

lossJ�=V�        �	vR�tYc�A�+*

loss3;Q='|�Z       �	��tYc�A�+*

lossc�
=���l       �	�~�tYc�A�+*

loss,!4=�F{�       �	7�tYc�A�+*

loss��<w�       �	��tYc�A�+*

loss��P=��S*       �	�D�tYc�A�+*

loss&@�=�4       �	���tYc�A�+*

loss���;#�       �	z�tYc�A�+*

loss-��<5�2�       �	�tYc�A�+*

loss(xi=�?E       �	���tYc�A�+*

loss��=�+       �	�G�tYc�A�+*

loss� N<Gýu       �	��tYc�A�+*

loss�ܟ<@hͺ       �	�v�tYc�A�+*

loss��d;��֮       �	#��tYc�A�+*

loss-H_=�t��       �	y!�tYc�A�+*

loss��<��       �	ø�tYc�A�+*

loss��<�P�       �	�O�tYc�A�+*

loss��!<�Qg       �	j��tYc�A�+*

loss�	{;�x�       �	&��tYc�A�+*

loss�u0=š��       �	�'�tYc�A�+*

lossŃ=�ߞ       �	���tYc�A�+*

loss0�=x�       �	�`�tYc�A�+*

loss�0:=�֩�       �	* �tYc�A�+*

loss�O3<��d�       �	)��tYc�A�+*

loss�#�=�^N�       �	��tYc�A�+*

loss��Z<iv�       �	?�tYc�A�+*

lossw�V<�I�       �	���tYc�A�+*

lossE p;q뚣       �	�=�tYc�A�+*

loss��K<P�>       �	���tYc�A�+*

losso�+<ॐb       �	�k�tYc�A�+*

loss(Ȫ<���       �	��tYc�A�+*

loss	@�<�=       �	���tYc�A�+*

loss`vP<��'�       �	Ɏ�tYc�A�+*

lossh�=X���       �	�&�tYc�A�+*

lossA2<L$��       �	��tYc�A�+*

loss�\=ꎁ�       �	S��tYc�A�+*

lossg"=\�sd       �	�1�tYc�A�+*

loss��<cGU       �	j��tYc�A�+*

lossE��<�}5"       �	�=�tYc�A�+*

lossX�=Ң       �	���tYc�A�+*

loss�n<�UB2       �	3m�tYc�A�+*

loss4�=o�W�       �	h�tYc�A�+*

loss�S�<����       �	H��tYc�A�+*

loss�>=	v�^       �	��tYc�A�+*

loss��;�=��       �	A)�tYc�A�+*

loss��R<-�       �	���tYc�A�+*

loss��;��Z�       �	��tYc�A�+*

loss���<N`P�       �	�{�tYc�A�+*

lossv7`=z��       �	e�tYc�A�+*

loss��<7� T       �	���tYc�A�+*

loss�,=����       �	�G�tYc�A�+*

loss᛼=Y��&       �	N��tYc�A�+*

lossn�k=��s�       �	��tYc�A�+*

loss�aZ=
"       �	p"�tYc�A�+*

loss��<3ŢW       �	���tYc�A�+*

loss忿<��        �	���tYc�A�+*

loss�i�=jdc       �	(F�tYc�A�+*

loss��<~�:�       �	���tYc�A�+*

loss([$=��       �	��tYc�A�+*

lossD�=5��;       �	T�tYc�A�+*

lossؗ�<Q_�#       �	���tYc�A�+*

loss��<��S       �	���tYc�A�+*

loss{� ={�ɻ       �	��tYc�A�+*

loss�;v;�2[6       �	���tYc�A�+*

loss�<w�o       �	Ie�tYc�A�,*

lossz��<�OG�       �	6�tYc�A�,*

loss$��<���       �	2��tYc�A�,*

loss4�<0�.G       �	'K uYc�A�,*

loss���<�kk�       �	�uYc�A�,*

loss��<od�D       �	f�uYc�A�,*

loss�s�<�y�u       �	�iuYc�A�,*

loss�
�<����       �	�uYc�A�,*

loss35[;x�a        �	E�uYc�A�,*

loss�<Z���       �	�fuYc�A�,*

loss��y=��;%       �	�uYc�A�,*

loss��<�,%�       �	��uYc�A�,*

loss	&�<D�       �	�GuYc�A�,*

loss+=�-`*       �	��uYc�A�,*

loss���;�~(�       �	��uYc�A�,*

loss|><G�P       �	 �uYc�A�,*

loss��;מI       �	.	uYc�A�,*

lossƘ=0�       �	=�	uYc�A�,*

loss/�y<���       �	�
uYc�A�,*

loss��=��       �	C=uYc�A�,*

lossq=��w�       �	�uYc�A�,*

loss�fU<��D       �	XuYc�A�,*

lossjY<��       �	��uYc�A�,*

loss�^~<��O       �	��uYc�A�,*

loss{�<3-��       �	�*uYc�A�,*

loss큛=\�Z       �	��uYc�A�,*

loss�e�<}D�       �	XouYc�A�,*

loss�KT;Q̙�       �	
uYc�A�,*

loss�u�<s
s�       �	#�uYc�A�,*

loss}��<y�Ԧ       �	�>uYc�A�,*

loss�=G�}       �	��uYc�A�,*

loss�!=X.�       �	tuYc�A�,*

loss���<4r��       �	��uYc�A�,*

loss��}=\8�J       �	�quYc�A�,*

lossq=H���       �	/uYc�A�,*

loss�66=P{�;       �	!�uYc�A�,*

loss��<��c\       �	�iuYc�A�,*

loss�=��i_       �	�	uYc�A�,*

loss�<�k>�       �	@�uYc�A�,*

loss;B�<��C       �	�EuYc�A�,*

loss�mM<3R��       �	�uYc�A�,*

loss�"=�ͨ�       �	F�uYc�A�,*

loss��<{e�       �	@KuYc�A�,*

lossM/P=9�Z�       �	!�uYc�A�,*

loss�=��)       �	ۈuYc�A�,*

loss�(�<��E�       �	�"uYc�A�,*

loss���;���]       �	߿uYc�A�,*

loss���<i�       �	uWuYc�A�,*

loss��E=��V       �	+uYc�A�,*

lossq�V;�us       �	��uYc�A�,*

loss(�W<�ˡ       �	-_ uYc�A�,*

loss�\`<jÉF       �	�� uYc�A�,*

losse�Q<���7       �	P�!uYc�A�,*

loss��Q=�aݯ       �	m"uYc�A�,*

loss��<O�v�       �	W�"uYc�A�,*

lossƝ=�%۠       �	L#uYc�A�,*

loss��&<>I�       �	��#uYc�A�,*

loss,]?<֦��       �	��$uYc�A�,*

loss1��;ʾc�       �	%uYc�A�,*

loss�<"<B2�       �	��%uYc�A�,*

loss�r%=��A       �	�=&uYc�A�,*

loss	�c<fJ��       �	l�&uYc�A�,*

loss�WX;�7�       �	 a'uYc�A�,*

loss��-<�ob       �	E�'uYc�A�,*

loss��=��f�       �	D�(uYc�A�,*

lossa��<�&��       �	z)uYc�A�,*

loss���<���       �	h�)uYc�A�,*

loss�=-&ٙ       �	FE*uYc�A�,*

loss�@=�>HT       �	Qh+uYc�A�,*

loss*pn<p
        �	��+uYc�A�,*

loss�8$<ρ!       �	�,uYc�A�,*

loss��<w^N)       �	�<-uYc�A�,*

lossO��=NQ�       �	 �-uYc�A�,*

loss#(p<��?%       �	�o.uYc�A�,*

loss�7/=81FX       �	&7/uYc�A�,*

loss�>�;;dj�       �	�0uYc�A�,*

loss�T�<�l1       �	i�0uYc�A�,*

loss�*
<Z&�       �	ݔ1uYc�A�,*

loss�K<�+8       �	P72uYc�A�,*

loss�<ۏȣ       �	��2uYc�A�,*

lossN��<��mr       �	V�3uYc�A�,*

loss�O�<�D�       �	�V4uYc�A�,*

loss<�>h       �	��4uYc�A�,*

loss�<���:       �	+�5uYc�A�,*

lossS~�<,Ԍ�       �	�G6uYc�A�,*

lossL5L;�~�Z       �	��6uYc�A�,*

loss��1<��J       �	dY8uYc�A�,*

lossT=j�ݒ       �	��8uYc�A�,*

lossj�z<�       �	��9uYc�A�,*

lossfW�<�d:�       �	�D:uYc�A�,*

loss���=��r�       �	��:uYc�A�,*

loss���=��W�       �	m�;uYc�A�,*

loss蕚<r~m�       �	N)<uYc�A�,*

lossO�P<���       �	��<uYc�A�,*

loss��C=�0�?       �	�m=uYc�A�,*

lossaE�<�ݘ�       �	�>uYc�A�,*

loss`�=��ެ       �	G�>uYc�A�,*

loss�=��Y       �	sI?uYc�A�,*

loss�4=D,E       �	��?uYc�A�,*

loss��<��a�       �	��@uYc�A�,*

loss:=�<�On�       �	�+AuYc�A�,*

loss��;<6M�<       �	��AuYc�A�,*

loss�K<��       �	BwBuYc�A�,*

loss���=��       �	�CuYc�A�,*

loss���;�        �	дCuYc�A�,*

loss6�<�n�       �	jMDuYc�A�,*

lossZϸ<�p�       �	��DuYc�A�,*

loss:|=��O       �	��EuYc�A�,*

loss��<8�k$       �	� FuYc�A�,*

loss/,;[r��       �	E�FuYc�A�,*

loss�<�<fǪu       �	aqGuYc�A�,*

loss6�<]��B       �	`HuYc�A�,*

loss��;t��       �	ԞHuYc�A�,*

loss�@ >�be�       �	�nIuYc�A�,*

loss�~�<ש�3       �	VJuYc�A�,*

loss݄<f��       �	��JuYc�A�,*

loss�s�=5	7       �	�MKuYc�A�,*

loss,=��>       �	2�KuYc�A�,*

lossYz=���       �	�~LuYc�A�,*

loss.}E<S��       �	"MuYc�A�,*

loss_V�<��       �	b�MuYc�A�,*

loss��<���w       �	�pNuYc�A�,*

losso4=��;       �	lOuYc�A�,*

loss�<=�6�       �	1�OuYc�A�,*

lossn
=X���       �	ۅPuYc�A�,*

loss4w<F���       �	�,QuYc�A�,*

loss��	=m3�y       �	J�QuYc�A�,*

lossZ�6=;5�       �	@�RuYc�A�,*

loss�+t=��7�       �	 zSuYc�A�-*

lossWI�<�GI�       �	QTuYc�A�-*

loss���=���       �	H�TuYc�A�-*

lossӿ�=�U,       �	�=UuYc�A�-*

lossL19=���       �	��UuYc�A�-*

lossZ�n<�@W�       �	jVuYc�A�-*

lossl��<���{       �	2 WuYc�A�-*

lossZb=�] �       �	l�WuYc�A�-*

loss�ƌ<�a}       �	�6XuYc�A�-*

loss!{S<1��       �	J�XuYc�A�-*

loss�<rb�p       �	mYuYc�A�-*

loss���<k���       �	�ZuYc�A�-*

loss��<+8�       �	|�ZuYc�A�-*

lossC}v=U���       �	;4[uYc�A�-*

lossS�z<� �)       �	�[uYc�A�-*

loss���<��%       �	mY\uYc�A�-*

loss1�=��       �	k�\uYc�A�-*

loss� �<#?m       �	n�]uYc�A�-*

loss��=��;       �	z^uYc�A�-*

loss���<by�       �	G�^uYc�A�-*

lossj�,=���       �	�C_uYc�A�-*

loss�5=<�;4       �	�_uYc�A�-*

loss��<�:#       �	�u`uYc�A�-*

loss6��;�=�b       �	auYc�A�-*

loss�ݬ;o�m       �	�auYc�A�-*

loss3ʨ;�?i       �	y>buYc�A�-*

loss�F�;�$H>       �	0�buYc�A�-*

lossȌ&=J<f       �	.�cuYc�A�-*

loss���<�k�&       �	�6duYc�A�-*

loss`�K<���G       �	I�duYc�A�-*

lossάA<3v�       �	|euYc�A�-*

loss-�f<p&+       �	-#fuYc�A�-*

lossF(=�L�       �	t�fuYc�A�-*

loss|D�;e2�       �	P�guYc�A�-*

lossΌ\<�
��       �	�7huYc�A�-*

lossї<�f�       �	R�huYc�A�-*

loss�"S=�]       �	:yiuYc�A�-*

loss�f�;2���       �	$juYc�A�-*

lossAE]<��%�       �	��juYc�A�-*

loss�`�<M�	0       �	pkuYc�A�-*

loss�u0<eJ��       �	�luYc�A�-*

losss�;�!�       �	�luYc�A�-*

loss�p=Q��       �	�ImuYc�A�-*

loss��<CeK       �	y�muYc�A�-*

loss6�S<���       �	W�nuYc�A�-*

loss���<S�U       �	/ouYc�A�-*

loss���=,�7       �	��ouYc�A�-*

loss�D�=K�OX       �	<gpuYc�A�-*

lossa==h�j       �	uquYc�A�-*

loss��<j~�       �	�quYc�A�-*

loss���<:Ͱ       �	S@ruYc�A�-*

loss �<�m�b       �	n�ruYc�A�-*

loss,�<"C�       �	#�suYc�A�-*

loss�{�<�U1       �	 tuYc�A�-*

loss���<��@T       �	o�tuYc�A�-*

lossc�<r	��       �	�SuuYc�A�-*

loss	[<=��ب       �	=�uuYc�A�-*

loss�9�;e�K       �	��vuYc�A�-*

loss���;��V�       �	g)wuYc�A�-*

loss��<����       �	FyuYc�A�-*

loss���;���       �	<�yuYc�A�-*

loss�1�;�I�#       �	mrzuYc�A�-*

loss��=xѳh       �	�!{uYc�A�-*

loss3gF<�Y�       �	��{uYc�A�-*

loss�-X=�n'R       �	5{|uYc�A�-*

loss��;=g�       �	�}uYc�A�-*

loss:�==*�       �	q�}uYc�A�-*

lossì_=�i��       �	�H~uYc�A�-*

lossx�=�=��       �	"�~uYc�A�-*

loss�u<�)       �	!suYc�A�-*

lossD�U<�G��       �	i�uYc�A�-*

loss���<N#'�       �	幀uYc�A�-*

lossh;�<�%x8       �	�]�uYc�A�-*

loss\�=��dm       �	���uYc�A�-*

loss�u�;���       �	雂uYc�A�-*

loss���<�#�       �	�0�uYc�A�-*

loss��<�A�L       �	˃uYc�A�-*

loss�<
8�       �	�e�uYc�A�-*

loss)�r<d��       �	���uYc�A�-*

loss�A<�7KZ       �	J��uYc�A�-*

lossUV<a�h_       �	�3�uYc�A�-*

lossV�<x�ܴ       �	�ƆuYc�A�-*

lossǍ#=c��<       �	Z�uYc�A�-*

loss��=@B       �	��uYc�A�-*

loss���<��n�       �	e��uYc�A�-*

lossN6�<�o�       �	�&�uYc�A�-*

loss�"=��&       �	�ΉuYc�A�-*

loss�J�=OY��       �	�e�uYc�A�-*

lossM�.<N-"�       �	��uYc�A�-*

loss��=�U�       �	8��uYc�A�-*

loss�|�<n�r�       �	-?�uYc�A�-*

lossq�P=�!,]       �	)�uYc�A�-*

loss�kY<|;       �	(�uYc�A�-*

loss��=<��       �	��uYc�A�-*

loss.c<1�i       �	5��uYc�A�-*

loss��=�$�i       �		R�uYc�A�-*

lossZ�=�Ω       �	��uYc�A�-*

loss6۬<���       �	��uYc�A�-*

loss)��<~��       �	=H�uYc�A�-*

loss���<(��       �	tE�uYc�A�-*

lossS~<f�&�       �	I��uYc�A�-*

loss���;4��       �	BΓuYc�A�-*

loss�FN;�]�       �	�k�uYc�A�-*

loss�H <��?       �	��uYc�A�-*

loss��g=����       �	���uYc�A�-*

loss�)�<��Wt       �	aT�uYc�A�-*

loss�>�<߫��       �	���uYc�A�-*

loss���<�L��       �	���uYc�A�-*

lossJ �<��;`       �	Y5�uYc�A�-*

loss_�<�T�       �	�ٚuYc�A�-*

loss\g;�#u       �	;q�uYc�A�-*

loss,�=[��       �	��uYc�A�-*

loss�WH<Ȧ�S       �	���uYc�A�-*

lossd9<|حD       �	=C�uYc�A�-*

loss���=V`O%       �	��uYc�A�-*

loss �+<V�ܿ       �	Z��uYc�A�-*

lossR��;]f�       �	k�uYc�A�-*

loss��:�3��       �	�8�uYc�A�-*

lossH<��       �	|ѠuYc�A�-*

loss���<3�       �	�f�uYc�A�-*

loss�#=��^       �	&�uYc�A�-*

loss�ި=��re       �	��uYc�A�-*

loss\׌<y�u       �	7�uYc�A�-*

loss,eh<��4u       �	�̣uYc�A�-*

loss;«�*       �	$c�uYc�A�-*

loss�L�<�1d?       �	���uYc�A�-*

loss��G=s�z�       �	�ΥuYc�A�-*

loss��;S{       �	D��uYc�A�-*

loss��0=K-)       �	�9�uYc�A�.*

lossF��<�N�A       �	[ΧuYc�A�.*

loss �b<!�,
       �	�b�uYc�A�.*

lossW6�<H^e(       �	��uYc�A�.*

loss�o<��/;       �	�ةuYc�A�.*

lossÜ=7d4x       �	rp�uYc�A�.*

loss��<��       �	��uYc�A�.*

loss.-�;��v       �	ᚫuYc�A�.*

loss�]<��Y       �	/6�uYc�A�.*

lossɔ<��9Z       �	BϬuYc�A�.*

loss/�<p;8       �	j�uYc�A�.*

lossGj�<A���       �	��uYc�A�.*

loss���<�c�X       �	���uYc�A�.*

loss{�=����       �	>?�uYc�A�.*

loss�_j;� �       �	�үuYc�A�.*

loss�K�;�?�       �	�p�uYc�A�.*

loss�<�^�       �	��uYc�A�.*

lossLi<*8��       �	m��uYc�A�.*

loss��<�)��       �	3T�uYc�A�.*

loss��,<؀�       �	��uYc�A�.*

loss�"�<w�al       �	�y�uYc�A�.*

lossi�<���       �	f�uYc�A�.*

loss؃�<׎��       �	��uYc�A�.*

loss3�<:�U       �	�M�uYc�A�.*

loss�J�<rO&�       �	d�uYc�A�.*

loss�w;H���       �	�}�uYc�A�.*

lossb=q��       �	Y�uYc�A�.*

loss�Mk<q���       �	-��uYc�A�.*

loss�Q<4���       �	
M�uYc�A�.*

loss��<<x�       �	��uYc�A�.*

loss.�<SZp�       �	�y�uYc�A�.*

loss���<k���       �	��uYc�A�.*

loss#�<9��i       �	���uYc�A�.*

loss;;�d��       �	�P�uYc�A�.*

loss�QP<z[�       �	���uYc�A�.*

loss)Q;;���       �	�ƼuYc�A�.*

loss��<Z�3Y       �	�`�uYc�A�.*

loss���<���E       �	&��uYc�A�.*

loss XO;7`��       �	\��uYc�A�.*

loss��?<�/E�       �	�.�uYc�A�.*

loss�5�<>�       �	qɿuYc�A�.*

loss�,N>^�        �	2s�uYc�A�.*

loss+L=�NHf       �	�	�uYc�A�.*

loss�w<�3"       �	ˢ�uYc�A�.*

loss�P�<#�2�       �	�O�uYc�A�.*

loss</�       �	r��uYc�A�.*

loss���:�?TE       �	���uYc�A�.*

loss��<�-�8       �	]6�uYc�A�.*

loss:H�<E�qv       �	���uYc�A�.*

loss�~;~�;       �	J}�uYc�A�.*

lossm?<�`B\       �	��uYc�A�.*

loss�t�9�|�       �	���uYc�A�.*

loss�<�; %�A       �	~S�uYc�A�.*

loss�J�;Qv��       �	���uYc�A�.*

losst?W9�Pӣ       �	)"�uYc�A�.*

loss��9@�9�       �	���uYc�A�.*

loss�;��n       �	�V�uYc�A�.*

loss��9<���       �	w��uYc�A�.*

loss]=3c+�       �	���uYc�A�.*

lossq
r;�ނ�       �	M�uYc�A�.*

loss3�;��$�       �	V��uYc�A�.*

loss�mQ=���       �	+O�uYc�A�.*

loss�P:Wp�       �	���uYc�A�.*

loss��;�T_E       �	G�uYc�A�.*

loss���<$�EZ       �	��uYc�A�.*

loss��x<�~��       �	!�uYc�A�.*

loss��;��ć       �	��uYc�A�.*

loss�^<#��       �	e�uYc�A�.*

loss옢<J���       �	���uYc�A�.*

loss6jQ=�j�[       �	���uYc�A�.*

lossX;=GB�       �	}<�uYc�A�.*

loss	[�<�VP       �	O�uYc�A�.*

loss�O=F�       �	��uYc�A�.*

loss�o�=7���       �	CU�uYc�A�.*

loss뷉=����       �	,�uYc�A�.*

loss0��<��Zo       �	���uYc�A�.*

loss�Ɗ=��K       �	n�uYc�A�.*

losst7�<�cρ       �	/��uYc�A�.*

lossۄ�<��a       �	���uYc�A�.*

loss��<���       �	�>�uYc�A�.*

loss2	�<`�`�       �	i��uYc�A�.*

lossz��<��.�       �	s-�uYc�A�.*

loss@$E<o���       �	���uYc�A�.*

loss�:)=s�v>       �	˽�uYc�A�.*

loss3ĥ<�9��       �	�[�uYc�A�.*

lossr?�;�Y�       �	4��uYc�A�.*

loss�Z�<s��       �	P��uYc�A�.*

lossf�K<Y��       �	��uYc�A�.*

lossv!�<	�z�       �	1�uYc�A�.*

loss�Z�;J���       �	`��uYc�A�.*

loss��U=�u�       �	�l�uYc�A�.*

loss	��<���%       �	8�uYc�A�.*

loss�U�<~E0�       �	���uYc�A�.*

loss���<�!       �	�G�uYc�A�.*

loss,�;��g�       �	b�uYc�A�.*

loss�x�:�꟮       �	P �uYc�A�.*

loss�Y<Ŗ70       �	R��uYc�A�.*

loss� V;ֹ�       �	�9�uYc�A�.*

loss�3�;���       �	k��uYc�A�.*

lossI��<��V�       �	�m�uYc�A�.*

loss��=:�5�       �	���uYc�A�.*

loss8�<�zY       �	{g�uYc�A�.*

loss9�
;���       �	\�uYc�A�.*

loss[p�< p�P       �	���uYc�A�.*

loss8A;����       �	���uYc�A�.*

lossQ<dB�       �	�]�uYc�A�.*

loss�v%=���       �	k��uYc�A�.*

loss_�2=t��I       �	v��uYc�A�.*

loss)��<v�P�       �	� �uYc�A�.*

loss��u<X��       �	���uYc�A�.*

loss)l�={�l}       �	Ih�uYc�A�.*

loss���:��[�       �	�uYc�A�.*

loss��6<��>T       �	���uYc�A�.*

loss�s�;9H��       �	�QvYc�A�.*

lossa=��i       �	1�vYc�A�.*

loss�$�<��'       �	�vYc�A�.*

loss��<��!       �	uvYc�A�.*

loss,8�<Y��       �	��vYc�A�.*

loss�}E<��m       �	�JvYc�A�.*

loss�9=$Ҕ�       �	y�vYc�A�.*

loss3�"=�ܪ�       �	�}vYc�A�.*

loss�y�<p҂�       �	vYc�A�.*

lossJ�8=_�l�       �	2�vYc�A�.*

loss*^;5��h       �	@jvYc�A�.*

loss��|=Dr>       �	��vYc�A�.*

loss��<�D�       �	�vYc�A�.*

loss�2�<�-u-       �	��vYc�A�.*

loss=��<d{�'       �	kJvYc�A�.*

loss��<:|�Y       �	�vYc�A�/*

lossWX9;M�       �	�~vYc�A�/*

loss`:�<j�$l       �	vYc�A�/*

losstY�<�W�B       �	��vYc�A�/*

loss�=N=[���       �	�gvYc�A�/*

loss�1|<�X�P       �	uvYc�A�/*

loss��G=	��       �	t�vYc�A�/*

loss1�;�F��       �	6vYc�A�/*

loss��&=�_�       �	S�vYc�A�/*

losshb<
�U�       �	�gvYc�A�/*

loss��o=(���       �	��vYc�A�/*

loss�K=�#*       �	)�vYc�A�/*

loss]*=<$xd�       �	/vYc�A�/*

loss���<2       �	l�vYc�A�/*

lossj��<<���       �	�fvYc�A�/*

loss�-<�;�       �	S?vYc�A�/*

loss��;W,>z       �	�vYc�A�/*

loss���;�
G       �	svYc�A�/*

lossǑ=�<C       �	� vYc�A�/*

loss6F<��%�       �	� vYc�A�/*

loss�B=`6�       �	�=!vYc�A�/*

loss���;�|�       �	��!vYc�A�/*

losso�<=3�i       �	�~"vYc�A�/*

lossOx�=�ڗ,       �	4/#vYc�A�/*

loss1�=��       �	^�#vYc�A�/*

loss���<����       �	߇$vYc�A�/*

loss�w=�*I       �	�%%vYc�A�/*

loss��I<���       �	)�%vYc�A�/*

loss�Ғ<QA`|       �	j&vYc�A�/*

loss��\<J%�c       �	�'vYc�A�/*

loss��<�0/�       �	�'vYc�A�/*

lossj�&=��"       �	U(vYc�A�/*

loss�_�<cc       �	��(vYc�A�/*

loss!��=�Ml�       �	0�)vYc�A�/*

loss�l;���:       �	�3*vYc�A�/*

loss�j<W?�       �	��+vYc�A�/*

lossF�%=��1�       �	�q,vYc�A�/*

loss�<�J�W       �	�	-vYc�A�/*

loss,~=�蛔       �	ˢ-vYc�A�/*

loss]Z�<��y       �	X:.vYc�A�/*

lossr{:;  �       �	s�.vYc�A�/*

lossT�m:��47       �	�~/vYc�A�/*

loss7�<0�zM       �	�0vYc�A�/*

loss��;�7       �	�G1vYc�A�/*

loss��!=n3�       �	�x2vYc�A�/*

loss�'�=�vD-       �	�b3vYc�A�/*

lossA��;�� �       �	L�3vYc�A�/*

loss\%<�O+�       �	/�4vYc�A�/*

lossx>�<��F       �	�B5vYc�A�/*

loss��;��M�       �	s�5vYc�A�/*

loss���;)VG�       �	dv6vYc�A�/*

lossl�=�[�       �	Z7vYc�A�/*

lossE�?=��+       �	ʥ7vYc�A�/*

loss�EV=�:��       �	6=8vYc�A�/*

loss-�H=��]       �	�8vYc�A�/*

lossvd=��       �	�i9vYc�A�/*

loss�Y=*�       �	:vYc�A�/*

loss��<>ا�       �	Ƨ:vYc�A�/*

lossf��;����       �	�D;vYc�A�/*

loss.v�<^��       �	L�;vYc�A�/*

loss��;<��a�       �	׉<vYc�A�/*

loss�B�<KDC�       �	=vYc�A�/*

loss;l<�bw�       �	�=vYc�A�/*

loss�Q�;(�       �	�Y>vYc�A�/*

lossR�'<y�a       �	�>vYc�A�/*

loss ��<_}T�       �	�?vYc�A�/*

loss6�=��e:       �	�0@vYc�A�/*

loss�80=a4"q       �	��@vYc�A�/*

lossۅ�:�H�3       �	�_BvYc�A�/*

loss��<���       �	7�BvYc�A�/*

lossI<\Ӗ�       �	F�CvYc�A�/*

loss�;|b�`       �	�,DvYc�A�/*

loss俴<�J	       �	��DvYc�A�/*

lossD�!<n�j�       �	�\EvYc�A�/*

lossd�<e�0f       �	��EvYc�A�/*

loss��<�uF       �	3�FvYc�A�/*

loss���<�=Iw       �	)"GvYc�A�/*

lossh�N<B��A       �	��GvYc�A�/*

loss,X<.c�0       �	nOHvYc�A�/*

loss�e�=�ڡ       �	CTIvYc�A�/*

loss�m<�Q�Z       �	��IvYc�A�/*

lossn�<b/=       �	ƈJvYc�A�/*

loss!��<�F�i       �	�KvYc�A�/*

loss,�<+�       �	�KvYc�A�/*

loss�D�<��P       �	QJLvYc�A�/*

loss_\�<�S3       �	D�LvYc�A�/*

loss.x�;O�,�       �	+�MvYc�A�/*

lossE�=�ـ       �	�>NvYc�A�/*

loss��<Ɲ^�       �	��NvYc�A�/*

loss�h<���w       �	YjOvYc�A�/*

loss��=<uǜ       �	��OvYc�A�/*

lossPИ<п��       �	�PvYc�A�/*

loss��<�,��       �	�(QvYc�A�/*

loss�}�<dg
�       �	NcRvYc�A�/*

loss�p�<F��       �	�hSvYc�A�/*

loss�|�<����       �	�TvYc�A�/*

loss�h=��D�       �	h UvYc�A�/*

loss� �<�� I       �	�UvYc�A�/*

loss���;�.MS       �	��VvYc�A�/*

loss�hM<�m�       �	�=WvYc�A�/*

loss7C;<_0(       �	-wXvYc�A�/*

loss�D<ml,�       �	vYvYc�A�/*

loss��v=8�K       �	��YvYc�A�/*

loss�K�<K,��       �	�CZvYc�A�/*

lossx��<�U       �	u�ZvYc�A�/*

loss�<��5�       �	��[vYc�A�/*

loss���<���       �	XY\vYc�A�/*

loss&û;}}��       �	]vYc�A�/*

loss�=<�Yf�       �	�]vYc�A�/*

loss֬o< �U       �	��^vYc�A�/*

loss��<O�g       �	�d_vYc�A�/*

lossL��<b��       �	�H`vYc�A�/*

loss!�,<{�~�       �	�`vYc�A�/*

loss�t�<�DfW       �	�yavYc�A�/*

loss"+=��       �	bvYc�A�/*

loss.q�<��b�       �	צbvYc�A�/*

loss�%�<#YR�       �	υcvYc�A�/*

loss+�=3�       �	x&dvYc�A�/*

lossy�;�_�       �	��dvYc�A�/*

loss]F�=P�`�       �	�levYc�A�/*

losss݄<A�5�       �	�fvYc�A�/*

lossni<O샷       �	&�fvYc�A�/*

loss-j�<\	       �	�NgvYc�A�/*

loss��<��;�       �	��gvYc�A�/*

loss_��<y�       �	��hvYc�A�/*

loss6��;�Y�V       �	d;ivYc�A�/*

loss��;�-Q�       �	t�ivYc�A�/*

lossF��<d��F       �	gjvYc�A�/*

lossEh�<�ӽ       �	M/kvYc�A�0*

loss�D�=K֚       �	��kvYc�A�0*

loss���<J�o!       �	�lvYc�A�0*

loss��<O�B{       �	:;mvYc�A�0*

loss㖽;鏿�       �	��mvYc�A�0*

lossHx�;atN^       �	�hnvYc�A�0*

loss���<�!u�       �	� ovYc�A�0*

loss�4�<�}t       �	�ovYc�A�0*

loss�ˡ<y���       �	$�pvYc�A�0*

loss�*=C�b       �	P5qvYc�A�0*

loss/I�<���       �	��qvYc�A�0*

loss<�<���       �	�arvYc�A�0*

loss�:s?�       �	n�rvYc�A�0*

loss��=V�       �	!�svYc�A�0*

loss��_< ���       �	�uvYc�A�0*

loss̺6=�/dd       �	=�uvYc�A�0*

loss�<�>�j       �	�UvvYc�A�0*

loss��<~Ɔ       �	��vvYc�A�0*

loss��<S)>O       �	��wvYc�A�0*

loss;�<�Ǳ�       �	�xvYc�A�0*

loss!�;X�v�       �	��xvYc�A�0*

loss��$=��       �	TyvYc�A�0*

loss�}j<�cF�       �	��yvYc�A�0*

loss�><�Ե       �	 �zvYc�A�0*

loss�Y�<�hf       �	�"{vYc�A�0*

loss��;Oy�A       �	/�{vYc�A�0*

loss�R<��e2       �	�^|vYc�A�0*

lossIҦ<���-       �	��|vYc�A�0*

loss��;�2       �	Ւ}vYc�A�0*

loss���;	H�;       �	xa~vYc�A�0*

loss��@=�&@�       �	T�~vYc�A�0*

loss��=%vDw       �	��vYc�A�0*

loss(��<v�-�       �	�1�vYc�A�0*

loss�E=���       �	��vYc�A�0*

loss��J=�E�A       �	���vYc�A�0*

losst�<w{       �	l%�vYc�A�0*

lossdV*<D�)       �	�0�vYc�A�0*

loss��w:��G�       �	�݃vYc�A�0*

lossl�+=1}z       �	w�vYc�A�0*

loss��%<RZ��       �	~�vYc�A�0*

lossF�<-��U       �	���vYc�A�0*

loss��x<�ح�       �	�̆vYc�A�0*

loss���;���       �	vl�vYc�A�0*

loss�u=%B�a       �	�
�vYc�A�0*

loss<�<��Ѧ       �	.��vYc�A�0*

lossC��<*�`%       �	8M�vYc�A�0*

loss��'=Ɇ�]       �	M��vYc�A�0*

lossC�=�g�       �	���vYc�A�0*

lossV�<ݓx,       �	ZE�vYc�A�0*

loss�U�;��~       �	�܋vYc�A�0*

loss��=�nG�       �	:v�vYc�A�0*

loss���<�G�X       �	��vYc�A�0*

loss�J,=G�U�       �	wۍvYc�A�0*

loss�W)<Q� �       �	Gs�vYc�A�0*

lossD�"=> ��       �	�"�vYc�A�0*

lossn��<X��       �	PvYc�A�0*

lossre�<RY��       �	�k�vYc�A�0*

loss�\=��~�       �	��vYc�A�0*

lossxpq;�m:       �	j��vYc�A�0*

lossNo"=�<�h       �	���vYc�A�0*

loss��%=��E       �	4��vYc�A�0*

loss&��;^��       �	$F�vYc�A�0*

lossVv=:��       �	��vYc�A�0*

loss$�<n�<[       �	E��vYc�A�0*

loss��)<�/��       �	��vYc�A�0*

loss�=a�|/       �	�vYc�A�0*

losso/^=\(        �	��vYc�A�0*

loss%��;u�1       �	~��vYc�A�0*

loss���<����       �	���vYc�A�0*

loss��v<�5�       �	�c�vYc�A�0*

lossT� =�/�       �	�6�vYc�A�0*

loss�U�<�Q)�       �	1�vYc�A�0*

loss�]=(<5       �	���vYc�A�0*

loss?e<ej*-       �	:z�vYc�A�0*

loss�si=h�D       �	�k�vYc�A�0*

loss]�=	��       �	���vYc�A�0*

loss�eX=0�X       �	��vYc�A�0*

loss�n�<f+�#       �	7R�vYc�A�0*

loss�S<^���       �	@��vYc�A�0*

loss��	=W���       �	���vYc�A�0*

loss�"=��U       �	�@�vYc�A�0*

loss�T7=3	��       �	�ݥvYc�A�0*

loss��<ܢzF       �	4��vYc�A�0*

lossŚ�<۟��       �	_$�vYc�A�0*

lossVr)<��S       �	§vYc�A�0*

loss�}%<B�a       �	\�vYc�A�0*

loss8��=�%�s       �	f�vYc�A�0*

loss=�A=��-       �	)��vYc�A�0*

loss���;��??       �	jK�vYc�A�0*

lossH�<=�r       �	C �vYc�A�0*

loss��%<���       �	M��vYc�A�0*

loss�Q=���       �	x_�vYc�A�0*

lossW.�=7�       �	l�vYc�A�0*

loss��]<�LQ       �	���vYc�A�0*

loss1�<tk�0       �	7Q�vYc�A�0*

loss�<�
��       �	�0�vYc�A�0*

loss��/;zLz       �	4֯vYc�A�0*

loss��;���       �	�q�vYc�A�0*

loss���<-g       �	��vYc�A�0*

loss���<(MP@       �	�ıvYc�A�0*

lossܙS=�D��       �	Nd�vYc�A�0*

loss�=.���       �	��vYc�A�0*

losssy=��b       �	k��vYc�A�0*

loss	<@<\{O�       �	�H�vYc�A�0*

lossĶ9< o�       �	�vYc�A�0*

lossZ�;=l�|�       �	���vYc�A�0*

loss��<5/�       �	�)�vYc�A�0*

loss��p<�h�       �	L¶vYc�A�0*

loss��<��1       �	�_�vYc�A�0*

loss���<u��       �	T��vYc�A�0*

loss��;W�u       �	��vYc�A�0*

loss;��<gUu       �	<-�vYc�A�0*

loss���</wx@       �	ǹvYc�A�0*

lossSr�<~mz�       �	_^�vYc�A�0*

lossZJ�<��m'       �	���vYc�A�0*

loss���<��xD       �	��vYc�A�0*

loss�Q�<T�3       �	�.�vYc�A�0*

lossa/<, �G       �	�μvYc�A�0*

loss�A<����       �	�l�vYc�A�0*

loss1�<6��       �	)�vYc�A�0*

lossC�<1QDN       �	���vYc�A�0*

loss�_h=��5�       �	P�vYc�A�0*

loss�s~;�g�       �	[	�vYc�A�0*

loss^=Z�͋       �	e��vYc�A�0*

loss�+=EZ{       �	>�vYc�A�0*

lossaI�='I�       �	���vYc�A�0*

loss�Q�<��]       �	�t�vYc�A�0*

loss���<b�D�       �	��vYc�A�0*

loss��!=7���       �	;��vYc�A�1*

loss��<�Q��       �	`<�vYc�A�1*

lossx=�_       �	~��vYc�A�1*

lossag=I*�       �	E��vYc�A�1*

loss'c=���       �	<-�vYc�A�1*

lossH��<���       �	��vYc�A�1*

lossC_C;1�N�       �	�a�vYc�A�1*

loss7=񺩔       �	
,�vYc�A�1*

loss���<�(�       �	���vYc�A�1*

loss���;��       �	�h�vYc�A�1*

lossJJ
;�L��       �	� �vYc�A�1*

lossFŊ<k+bi       �	#��vYc�A�1*

lossR+E<�ь*       �	�7�vYc�A�1*

lossa3�<;;t       �	���vYc�A�1*

loss��=݅��       �	Um�vYc�A�1*

loss�Y==��t       �	-	�vYc�A�1*

loss.� =��       �	���vYc�A�1*

loss�ؘ<p]��       �	8�vYc�A�1*

loss���<����       �	N��vYc�A�1*

loss�<����       �	���vYc�A�1*

loss�3=@sFQ       �	�)�vYc�A�1*

loss��<(���       �	��vYc�A�1*

loss}N�:�~�       �	�t�vYc�A�1*

loss��<^�#       �	��vYc�A�1*

lossk�=�s^       �	�b�vYc�A�1*

loss�4�; �!�       �	��vYc�A�1*

lossJ��<�8Y       �	�0�vYc�A�1*

loss�7=d��       �	��vYc�A�1*

loss� =զ
�       �	���vYc�A�1*

loss�~r<7���       �	�vYc�A�1*

loss���<�,m~       �	1�vYc�A�1*

losst�n;��       �	���vYc�A�1*

loss�=R!s�       �	5��vYc�A�1*

loss��<~C�Y       �	�P�vYc�A�1*

loss(�u=�zt       �	T�vYc�A�1*

loss��;����       �	���vYc�A�1*

loss	8=Q*zR       �	���vYc�A�1*

lossE�<;6�'?       �	���vYc�A�1*

loss�j�<Qȧ       �	!��vYc�A�1*

lossŮM<JcM8       �	A��vYc�A�1*

loss;c�<����       �	=��vYc�A�1*

loss��p<�%��       �	]��vYc�A�1*

loss7=�4�I       �	ޭ�vYc�A�1*

loss��;��+�       �	�S�vYc�A�1*

loss��=Q`�\       �	�vYc�A�1*

loss�.�;S�N       �	ٰ�vYc�A�1*

lossd><�a��       �	�a�vYc�A�1*

loss�d�<�nb       �	���vYc�A�1*

lossʥ�<Я�       �	 F�vYc�A�1*

loss/?z<��       �	q��vYc�A�1*

loss	'�<��GR       �	*��vYc�A�1*

loss&��=^���       �	� �vYc�A�1*

loss�AK<�L       �	��vYc�A�1*

lossh��<ߛ�       �	�S�vYc�A�1*

loss��<P܁�       �	���vYc�A�1*

lossCY<�2 �       �	o�vYc�A�1*

loss��D<�IS       �	e�vYc�A�1*

loss{˾<*�S       �	��vYc�A�1*

loss�=�+�=       �	MN�vYc�A�1*

lossy�<��_       �	)�vYc�A�1*

lossi�<�]��       �	���vYc�A�1*

loss�4�;�VJ�       �	T9�vYc�A�1*

loss{��;���       �	x��vYc�A�1*

loss8�X=�UM       �	���vYc�A�1*

loss�C<��!       �	�X�vYc�A�1*

loss�^�<)L*�       �	���vYc�A�1*

loss�f�;��       �	���vYc�A�1*

loss�(=(kC       �	��vYc�A�1*

loss��<��J�       �	B��vYc�A�1*

lossW�$;�#o�       �	�K�vYc�A�1*

loss�65<�Eb+       �	���vYc�A�1*

loss�p=uP��       �	�v�vYc�A�1*

loss��;�4�^       �	m�vYc�A�1*

loss
Չ=O�)�       �	c��vYc�A�1*

lossjc�<7�Q       �	{N�vYc�A�1*

loss�m=^�vp       �	���vYc�A�1*

lossݒ�<>ea       �	`��vYc�A�1*

loss��;��4       �	�,�vYc�A�1*

lossJ�<���       �	���vYc�A�1*

loss���;|�?Q       �	S\�vYc�A�1*

lossH�<�K�H       �	i��vYc�A�1*

loss�_<���       �	��vYc�A�1*

lossm�Q<s�,�       �	�M�vYc�A�1*

loss���<�	       �	J��vYc�A�1*

lossE;&<�Вv       �	���vYc�A�1*

loss_̠<D�3       �	DO wYc�A�1*

loss�^`<!ѹ6       �	�� wYc�A�1*

loss���<u��       �	�wYc�A�1*

loss�R�<s�p>       �	�BwYc�A�1*

lossH:6<Fl��       �	��wYc�A�1*

loss_^�<,�{;       �	,)wYc�A�1*

loss.��<�.5�       �	2�wYc�A�1*

loss �/=��       �	&nwYc�A�1*

loss�6�=$p�E       �	EwYc�A�1*

loss�ݲ<�͏%       �	��wYc�A�1*

lossJ�=����       �	�QwYc�A�1*

loss�5a<���       �	V�wYc�A�1*

loss)�k<�^�       �	�wYc�A�1*

loss��g<��H       �	�^	wYc�A�1*

lossE��<*��}       �	� 
wYc�A�1*

lossD.�;���+       �	��
wYc�A�1*

loss��=w���       �	�wYc�A�1*

loss�`=���e       �	�-wYc�A�1*

loss��<�Z�y       �	��wYc�A�1*

losso�<b!       �	�~wYc�A�1*

lossf�<
�       �	'wYc�A�1*

loss�CF<C�       �	�wYc�A�1*

loss�='r v       �	?nwYc�A�1*

loss�2�=P~y�       �	1
wYc�A�1*

loss�5�=<�       �	�wYc�A�1*

lossl�g<��d�       �	�YwYc�A�1*

loss�<��#       �	�wYc�A�1*

loss}��<gC�       �	ܛwYc�A�1*

loss���;���       �	�BwYc�A�1*

lossVk�<}���       �	[�wYc�A�1*

lossq��<��Ȼ       �	�3wYc�A�1*

lossX�k=@��       �	�mwYc�A�1*

loss��U;�\�@       �		2wYc�A�1*

lossu�=�
6�       �	��wYc�A�1*

loss��<f��H       �	��wYc�A�1*

lossRXc<E���       �	wfwYc�A�1*

loss�<\���       �	�iwYc�A�1*

loss<��<�e}�       �	��wYc�A�1*

loss�<K<v��       �	p[wYc�A�1*

lossW��<���       �	0wYc�A�1*

loss���;�(
       �	E�wYc�A�1*

loss@Jc<V'r@       �	DiwYc�A�1*

lossB]�<p}fO       �	�rwYc�A�1*

loss�<����       �	�� wYc�A�2*

loss��;��a       �	�"wYc�A�2*

loss?)�<&|ǟ       �	�"wYc�A�2*

loss=�+=5���       �	UN#wYc�A�2*

loss��<+;�       �	�$wYc�A�2*

lossmf�<�7b       �	q�$wYc�A�2*

loss��<#��       �	�%wYc�A�2*

loss%��<����       �	��&wYc�A�2*

loss]5=���       �	�'wYc�A�2*

lossȒF=a[��       �	e7(wYc�A�2*

loss�-�=`:�       �	�)wYc�A�2*

loss���<��qv       �	�)wYc�A�2*

lossN��=�PA�       �	zp*wYc�A�2*

lossO2<���        �	,++wYc�A�2*

loss �=�PH#       �	�3,wYc�A�2*

loss(��<��v�       �	�u-wYc�A�2*

loss��=�E�'       �	�".wYc�A�2*

loss�S�<���G       �	�U/wYc�A�2*

loss)۝<�� x       �	$�0wYc�A�2*

loss,�.<���       �	6�1wYc�A�2*

lossm �;�       �	 W2wYc�A�2*

loss	�f<���z       �	&�2wYc�A�2*

loss�WS=�1�"       �	'�3wYc�A�2*

loss��+<��A�       �	�@4wYc�A�2*

lossOr3=~�K�       �	:�4wYc�A�2*

loss���<�c�       �	�5wYc�A�2*

lossA�<�~       �	�^6wYc�A�2*

loss_��<��]       �	?�6wYc�A�2*

lossMA�<�;r       �	�7wYc�A�2*

loss�&0=��'2       �	�d8wYc�A�2*

loss1��<u�X       �	:wYc�A�2*

losss�<��]�       �	k*;wYc�A�2*

loss�ϟ<��\�       �	Y<wYc�A�2*

loss@�Q<=F�       �	��<wYc�A�2*

loss��<��&�       �	��=wYc�A�2*

loss�5�=��.I       �	!<>wYc�A�2*

loss��e<���R       �	�>wYc�A�2*

loss	< �Ï       �	w?wYc�A�2*

loss1f=<�'~L       �	Q@wYc�A�2*

loss�<О�       �	��@wYc�A�2*

loss���;c2��       �	fMAwYc�A�2*

loss�߸<���)       �	)�AwYc�A�2*

loss�;�=7��       �	o�BwYc�A�2*

loss�W�<�ε       �	CwYc�A�2*

lossl6�<��       �	��CwYc�A�2*

loss� T;�Pc�       �	�WDwYc�A�2*

loss	��=�R�       �	��DwYc�A�2*

loss�C=7��>       �	��EwYc�A�2*

loss�<m�yB       �	�4FwYc�A�2*

loss$��<��"       �	@0GwYc�A�2*

lossҝ<��F       �	�GwYc�A�2*

loss,DB=��`       �	aHwYc�A�2*

loss@62<�0�       �	E�HwYc�A�2*

loss_<��|�       �	*�IwYc�A�2*

loss[�(<�u��       �	FEJwYc�A�2*

loss1 R=�~�       �	m�JwYc�A�2*

loss:l�<�Z �       �	>�KwYc�A�2*

loss���<��m�       �	�2LwYc�A�2*

lossцT<�U��       �	0�LwYc�A�2*

loss͊<< E(�       �	wMwYc�A�2*

loss�<���       �	NNwYc�A�2*

loss���<؜��       �	ߦNwYc�A�2*

loss���<��z       �	�>OwYc�A�2*

loss	�<{�ȧ       �	��OwYc�A�2*

lossZXE=���R       �	��PwYc�A�2*

loss��/<�z�       �	~9QwYc�A�2*

loss7)p<�� (       �	{�QwYc�A�2*

loss}�<cJs�       �	�sRwYc�A�2*

lossQ�<k���       �	d SwYc�A�2*

loss3�.<��       �	s.TwYc�A�2*

loss���<
���       �	 �TwYc�A�2*

loss�8�<�qn       �	�xUwYc�A�2*

loss g<|�>�       �	�VwYc�A�2*

loss��S<P��+       �	��VwYc�A�2*

loss�D�<d,�       �	>wWwYc�A�2*

lossJ�%<����       �	�XwYc�A�2*

loss��M;#>�       �	q�XwYc�A�2*

lossԱ<����       �	3NYwYc�A�2*

loss�`�<lK�       �	��YwYc�A�2*

lossfŬ;��K       �	uxZwYc�A�2*

loss�=��       �	�[wYc�A�2*

loss�6t=��c4       �	�[wYc�A�2*

loss�=�?j�       �	�H\wYc�A�2*

lossᗊ<�Zj       �	+�\wYc�A�2*

loss��y;鷧�       �	ms]wYc�A�2*

lossq�b<��D       �	�^wYc�A�2*

loss_�<Vvň       �	��^wYc�A�2*

loss.�<�H83       �	q:_wYc�A�2*

loss���<
��       �	��_wYc�A�2*

loss��g<)׭       �	�p`wYc�A�2*

loss�=�<���       �	�awYc�A�2*

loss_��<�i       �	��awYc�A�2*

loss)�<듅�       �	YMbwYc�A�2*

loss��
<��I�       �	��bwYc�A�2*

loss,�<`��       �	 �cwYc�A�2*

lossx�:S��       �	/dwYc�A�2*

loss��<�˼       �	>�dwYc�A�2*

lossw��<Ȑ��       �	�gewYc�A�2*

loss4a�<���       �	3�ewYc�A�2*

loss�+�<�N�#       �	!�fwYc�A�2*

loss��;��l       �	�+gwYc�A�2*

lossjV=Tセ       �	ۿgwYc�A�2*

loss
|�;�菅       �	VhwYc�A�2*

loss���;���>       �	��hwYc�A�2*

loss�M^;z䙐       �	i�iwYc�A�2*

loss�-=����       �	�ejwYc�A�2*

loss�
�;����       �	mkwYc�A�2*

lossal�<U�Il       �	��kwYc�A�2*

loss��<�q�       �	�:lwYc�A�2*

loss��=F r�       �	8�lwYc�A�2*

loss&�<�79�       �	%tmwYc�A�2*

lossS?M<��       �	pnwYc�A�2*

loss��"=U'       �	j�nwYc�A�2*

lossBN;�m       �	z9owYc�A�2*

loss�	=N�;�       �	,�owYc�A�2*

loss���<�X-�       �	ݲpwYc�A�2*

loss�;�<���e       �	VJqwYc�A�2*

loss�8_<��k+       �	�qwYc�A�2*

loss줬<�`8;       �	F}rwYc�A�2*

lossQ�;�@}       �	,swYc�A�2*

lossp��<z�       �	��swYc�A�2*

loss�l#;�UV�       �	�itwYc�A�2*

lossW�J<�	l�       �	U0uwYc�A�2*

loss/��:	�:�       �	`�uwYc�A�2*

lossD\<C;�       �	�[vwYc�A�2*

loss�<� �       �	��vwYc�A�2*

lossj=n��       �	֐wwYc�A�2*

loss`��<�,�D       �	UKxwYc�A�2*

loss�:)<���+       �	�xwYc�A�3*

loss�u�<� 9       �	ȗywYc�A�3*

lossJE�<Joa       �	�*zwYc�A�3*

loss:wN;�ޖ       �	�zwYc�A�3*

loss��\<_��v       �	�z{wYc�A�3*

loss	o3;#�*       �	#|wYc�A�3*

loss���:ڧ�       �	ũ|wYc�A�3*

loss:N�;�\\       �	vm}wYc�A�3*

loss{-<�z �       �	�~wYc�A�3*

lossS�`<��7�       �	��~wYc�A�3*

loss���<'\.�       �	w0wYc�A�3*

loss$�7<�̳�       �	��wYc�A�3*

loss�(3=:�,       �	�u�wYc�A�3*

loss�:
;�n
       �	�A�wYc�A�3*

loss�c�:1���       �	K�wYc�A�3*

loss<Ŗ9�T@�       �	U��wYc�A�3*

loss���<_�       �	�f�wYc�A�3*

loss��]=��f       �	:�wYc�A�3*

loss/|B<�5޵       �	��wYc�A�3*

loss��;8�y1       �	�G�wYc�A�3*

lossF��<�       �	��wYc�A�3*

loss���<bY/�       �	��wYc�A�3*

loss܈�;���       �	-"�wYc�A�3*

loss��$;��R       �	ع�wYc�A�3*

loss�4"=3T_�       �	zP�wYc�A�3*

loss��<�G��       �	0��wYc�A�3*

loss�S�;{9�       �	R��wYc�A�3*

loss֋�;�V��       �	�L�wYc�A�3*

loss�_�<'��       �	��wYc�A�3*

loss\��<3F?e       �	cz�wYc�A�3*

loss:�	=�V
m       �	�wYc�A�3*

loss�n)=%�Ѝ       �	���wYc�A�3*

losss	<�2�       �	DM�wYc�A�3*

lossH7�;͔O       �	d�wYc�A�3*

lossx7=o���       �	υ�wYc�A�3*

loss<�f<��D       �	iU�wYc�A�3*

loss���<v�A       �	���wYc�A�3*

loss*� =���       �	R��wYc�A�3*

lossvP<��y       �	R�wYc�A�3*

loss�"N<�*��       �	��wYc�A�3*

loss���<�\�P       �	)��wYc�A�3*

loss���<T�3       �	j�wYc�A�3*

loss��;?	�o       �	�s�wYc�A�3*

lossmآ<�E       �	��wYc�A�3*

loss��;A�;x       �	���wYc�A�3*

loss���;�*       �	Qj�wYc�A�3*

lossb�;��7�       �	��wYc�A�3*

loss�Gf;U�	>       �	���wYc�A�3*

lossc̆;z/�       �	�a�wYc�A�3*

loss&m�;x�        �	V�wYc�A�3*

loss��*=��q        �	-�wYc�A�3*

loss��<_�;�       �	�ךwYc�A�3*

loss�9V;���       �	���wYc�A�3*

loss�z<���~       �	g(�wYc�A�3*

lossӀc<�,T�       �	uȜwYc�A�3*

lossvj�;{|       �	k�wYc�A�3*

loss�F<=,^�       �	6;�wYc�A�3*

loss28;0�M       �	�!�wYc�A�3*

loss?�;����       �	辟wYc�A�3*

loss���<��       �	�c�wYc�A�3*

lossw��<ː�S       �	!�wYc�A�3*

loss�Ψ<�N�v       �	(��wYc�A�3*

loss�<�:-ږ0       �	6�wYc�A�3*

lossq� <��]�       �	`ˢwYc�A�3*

loss��;�}z�       �	�o�wYc�A�3*

loss�A<�/�	       �	M�wYc�A�3*

lossRҽ;j�R�       �	���wYc�A�3*

loss{*<*�lC       �	�N�wYc�A�3*

lossv��<��o       �	+��wYc�A�3*

loss�rU;5-��       �	���wYc�A�3*

lossv�J=mF�H       �	
J�wYc�A�3*

loss,�T;�u�       �	��wYc�A�3*

loss]�<~'       �	wYc�A�3*

loss**>`]B       �	9_�wYc�A�3*

loss=��<pFJ       �	s��wYc�A�3*

lossĵY=w�"�       �	��wYc�A�3*

loss��=����       �	5&�wYc�A�3*

loss��<A�       �	[��wYc�A�3*

loss��}<�>7�       �	ς�wYc�A�3*

loss�7�<�Ji�       �	��wYc�A�3*

lossZN�<��2       �	~��wYc�A�3*

lossI�<�Rj"       �	�M�wYc�A�3*

loss?�=�4�G       �	���wYc�A�3*

lossG0;�唂       �	��wYc�A�3*

loss��S=��5       �	�C�wYc�A�3*

loss�g<��ڤ       �	n��wYc�A�3*

loss�U�<�K��       �	��wYc�A�3*

lossn
�<+do       �	��wYc�A�3*

lossǺ=����       �	��wYc�A�3*

lossC�?;S\'�       �	_��wYc�A�3*

loss&�<�Зk       �	�/�wYc�A�3*

loss���<Gͥ�       �	���wYc�A�3*

loss��<�K�       �	J_�wYc�A�3*

lossx�<@�2       �	���wYc�A�3*

loss���<c�bh       �	З�wYc�A�3*

loss��;��x       �	34�wYc�A�3*

lossi'�<6
�_       �	���wYc�A�3*

loss*SW<�`�       �	�`�wYc�A�3*

loss#Π<���       �	� �wYc�A�3*

loss�l�=�.ћ       �	���wYc�A�3*

lossE�;�I̕       �	�5�wYc�A�3*

loss�><.1t       �	���wYc�A�3*

loss��;I��       �	Me�wYc�A�3*

lossD��<�r       �	 ��wYc�A�3*

loss@F�;�8�       �	Ӡ�wYc�A�3*

loss��;���       �	�:�wYc�A�3*

loss�'3=����       �	���wYc�A�3*

lossx4B;w�??       �	j�wYc�A�3*

loss�.F=�Srn       �	��wYc�A�3*

losssV<Q��4       �	���wYc�A�3*

loss}L0=H���       �	}=�wYc�A�3*

loss�u�=0���       �	x��wYc�A�3*

loss�[@=<�u�       �	&n�wYc�A�3*

losslJ�<1#f       �	�	�wYc�A�3*

lossN];�=W)       �	���wYc�A�3*

lossn��<�p0�       �	K>�wYc�A�3*

lossE'<�c
�       �	M��wYc�A�3*

loss�Д=�o��       �	zq�wYc�A�3*

loss��{<N�s�       �	�	�wYc�A�3*

loss��z<>��       �	ݲ�wYc�A�3*

loss���<��Y�       �	O�wYc�A�3*

loss�Z�<u8       �	���wYc�A�3*

loss
�:���%       �	N~�wYc�A�3*

loss�$<�|��       �	I�wYc�A�3*

lossƻ(=�B�!       �	���wYc�A�3*

loss}�<��o�       �	tD�wYc�A�3*

loss(V�=f��Z       �	E��wYc�A�3*

loss���<�d��       �	H��wYc�A�3*

lossh;�z&�       �	�=�wYc�A�4*

loss��;'NO6       �	k��wYc�A�4*

loss��<�s�7       �	 q�wYc�A�4*

loss�j�;�o��       �	��wYc�A�4*

lossnx{<
��]       �	j��wYc�A�4*

loss�|�<��       �	�=�wYc�A�4*

lossH�< Uj�       �	s��wYc�A�4*

lossd�;B'j�       �	�x�wYc�A�4*

loss
�<ˁ�7       �	0�wYc�A�4*

loss�t�;�X�       �	���wYc�A�4*

lossݠ<e)J2       �	�L�wYc�A�4*

loss:R<�3�       �	��wYc�A�4*

loss�_�<z`�F       �	�7�wYc�A�4*

lossl�K=b�q�       �	}��wYc�A�4*

loss&+�<
�<       �	�h�wYc�A�4*

loss0�<�8d�       �	�T�wYc�A�4*

loss�}�<�Eb�       �	���wYc�A�4*

loss6mF=��fD       �	��wYc�A�4*

lossZ��<�Zt�       �	lC�wYc�A�4*

loss��_<T�?�       �	���wYc�A�4*

lossi��;,e�d       �	%w�wYc�A�4*

loss�PX:�w�       �	���wYc�A�4*

lossQ��<��A       �	��wYc�A�4*

loss(��;���       �	f��wYc�A�4*

loss�<����       �	uU�wYc�A�4*

loss�8h<qTK       �	R��wYc�A�4*

loss�<vu       �	��wYc�A�4*

loss�P=���
       �	���wYc�A�4*

loss���<f�t�       �	�D�wYc�A�4*

loss� =S�        �	T��wYc�A�4*

loss�@�<8��X       �	��wYc�A�4*

lossF��;�㏿       �	�<�wYc�A�4*

loss:[�<����       �	���wYc�A�4*

lossA�=L���       �	:x�wYc�A�4*

lossiy< ��       �	I�wYc�A�4*

lossL�<�K�       �	u��wYc�A�4*

loss&��<�v�       �	���wYc�A�4*

loss&_�<u�L#       �	,��wYc�A�4*

loss_�m<�-"�       �	E)�wYc�A�4*

loss�=�*"D       �	���wYc�A�4*

loss�"K<8�!�       �	}X�wYc�A�4*

loss_�}<Ǆ/       �	���wYc�A�4*

loss�u=��}�       �	m��wYc�A�4*

loss��`=�_4R       �	%# xYc�A�4*

loss��<^O��       �	� xYc�A�4*

loss�<�vp�       �	�fxYc�A�4*

lossd�L<�GG�       �	cxYc�A�4*

lossk�<��       �	f�xYc�A�4*

loss� �;��K�       �	�8xYc�A�4*

loss7t<���)       �	��xYc�A�4*

loss|�Y<���       �	�yxYc�A�4*

loss��:v�-p       �	#xYc�A�4*

loss2<�0ف       �	C�xYc�A�4*

loss��8=�S�       �	nQxYc�A�4*

lossA��;��M�       �	��xYc�A�4*

lossxՠ=�       �	r�xYc�A�4*

loss�u�<�	�       �	� xYc�A�4*

loss^=�#l}       �	��xYc�A�4*

loss�b7<�;��       �	�^	xYc�A�4*

lossŉ<+��(       �	��	xYc�A�4*

loss�}�;���O       �	:�
xYc�A�4*

loss1�<#��	       �	L8xYc�A�4*

loss��h=6�s       �	 �xYc�A�4*

loss�<�$�       �	�nxYc�A�4*

loss�m>����       �	xYc�A�4*

loss6��<�)�       �	��xYc�A�4*

loss��@=�/(m       �	CxYc�A�4*

loss�}-<�Ms%       �		�xYc�A�4*

loss |�<�[L       �	�wxYc�A�4*

loss���;i�1�       �	�xYc�A�4*

loss攈<��M\       �	�xYc�A�4*

loss�a�<�'�       �	FxYc�A�4*

loss�<���T       �	�xYc�A�4*

loss�$�<�E�>       �	|{xYc�A�4*

loss�'�<�l#�       �	bxYc�A�4*

lossO{X<�;B�       �	��xYc�A�4*

lossJ��<�_t�       �	͓xYc�A�4*

loss�Ä<N���       �	�DxYc�A�4*

loss�v=Zz�
       �	�xYc�A�4*

loss��<�	R-       �	�BxYc�A�4*

loss��<8�ot       �	�xYc�A�4*

loss֫�;���       �	^�xYc�A�4*

loss���<�H�       �	`#xYc�A�4*

loss��<����       �	t�xYc�A�4*

loss��#<�1p�       �	�ixYc�A�4*

lossd�:�*�t       �	m xYc�A�4*

loss�H�<��s�       �	4�xYc�A�4*

lossũM<vdt{       �	�<xYc�A�4*

loss;��<(K��       �	�+xYc�A�4*

loss	�y=k٥�       �	��xYc�A�4*

loss�`?<I�j>       �	�xYc�A�4*

lossL�<U10�       �	�� xYc�A�4*

loss�|�;��*       �	\Y!xYc�A�4*

loss�-;!���       �	!"xYc�A�4*

loss�EF<����       �	��"xYc�A�4*

loss���<�y��       �	\�#xYc�A�4*

loss���<��!�       �	^g$xYc�A�4*

loss���<�+��       �	��$xYc�A�4*

loss�HT<��Ub       �	b�%xYc�A�4*

loss�&=~��5       �	5B&xYc�A�4*

lossek�:KGY�       �	�&xYc�A�4*

loss�7�;���2       �	�x'xYc�A�4*

loss� �=��"�       �		(xYc�A�4*

loss��=
�e       �	��(xYc�A�4*

loss�C;<�       �	�f)xYc�A�4*

losshr�<Є��       �	d*xYc�A�4*

loss�-D<��v       �	Y�*xYc�A�4*

lossd�<� @�       �	�B+xYc�A�4*

loss�m;�*��       �	��+xYc�A�4*

lossz)=Қ�       �	t,xYc�A�4*

loss&�N<�"�       �	k-xYc�A�4*

loss&��;��)       �	��-xYc�A�4*

loss�|�<t�V)       �	�J.xYc�A�4*

loss�ϫ;T8!       �	C�.xYc�A�4*

loss���;��vH       �	-|/xYc�A�4*

loss�ݪ<��3�       �	Y0xYc�A�4*

loss�q;�U�       �	C�0xYc�A�4*

loss�i�;\�       �	�D1xYc�A�4*

loss���<C�u�       �	��1xYc�A�4*

loss'�;�ѤB       �	ty2xYc�A�4*

loss 0z<�¨N       �	mT3xYc�A�4*

loss@Τ<�<       �	��3xYc�A�4*

lossxo=W�n       �	L�4xYc�A�4*

lossb_<1��       �	C5xYc�A�4*

lossc��<�� �       �	�5xYc�A�4*

loss���:i��I       �	S6xYc�A�4*

lossZ�j=�SR�       �	K�6xYc�A�4*

loss�<��ޑ       �	��7xYc�A�4*

losso��<(�e       �	F(8xYc�A�5*

loss�E<��       �	��8xYc�A�5*

lossIs(<jPE       �	R9xYc�A�5*

loss;��<�Pi�       �	��9xYc�A�5*

lossd�U<�4�I       �	)�:xYc�A�5*

loss�>=�Q3       �	�0;xYc�A�5*

loss/d=�D        �	��;xYc�A�5*

lossJ�O<���       �	�t<xYc�A�5*

lossiX�<�;}       �	F=xYc�A�5*

loss4b�<QF��       �	^�=xYc�A�5*

loss�]�;��       �	v4>xYc�A�5*

loss�j�<>3��       �	��>xYc�A�5*

loss	��<;u�|       �	9b?xYc�A�5*

loss}{Y<Z�8�       �	#�?xYc�A�5*

loss(\=��^g       �	`�@xYc�A�5*

loss��=�
�       �	�&AxYc�A�5*

lossGj<Q-�       �	��AxYc�A�5*

loss��<L(�:       �	�iBxYc�A�5*

loss��;���       �	
CxYc�A�5*

loss :=u��l       �	}�CxYc�A�5*

loss7*�<���"       �	�XDxYc�A�5*

loss�C=2��;       �	��DxYc�A�5*

loss*�7<��W$       �	=�ExYc�A�5*

loss�=�<�sp       �	W>FxYc�A�5*

loss�#�<2���       �	d�FxYc�A�5*

lossA��<�ה�       �	�GxYc�A�5*

loss͹k=�r       �	�$HxYc�A�5*

loss�+9<k��       �	˼HxYc�A�5*

loss]H�=��_�       �	�`IxYc�A�5*

loss;!\<)��       �	Q�IxYc�A�5*

lossϺk<J6-D       �	�JxYc�A�5*

lossz�=D��*       �	z6KxYc�A�5*

loss��}=_       �	M�KxYc�A�5*

loss���<�KD�       �	�oLxYc�A�5*

lossxc�=C�@e       �	�MxYc�A�5*

lossѹ9=�Z*       �	)�MxYc�A�5*

loss�,�<O��       �	�KNxYc�A�5*

loss,ќ<����       �	�NxYc�A�5*

loss<S�<B	��       �	:wOxYc�A�5*

loss�{g<��ھ       �	�
PxYc�A�5*

lossf�=�N�       �	U�PxYc�A�5*

loss��<�jn�       �	�<QxYc�A�5*

lossl�7=�i�q       �	��QxYc�A�5*

loss�jh=Bή�       �	�pRxYc�A�5*

loss���;Ȑ|�       �	�SxYc�A�5*

loss`/+<3�T       �	��SxYc�A�5*

loss\q�;o�P       �	^MTxYc�A�5*

loss��;��\        �	��TxYc�A�5*

loss��;D�Ǻ       �	�wUxYc�A�5*

loss|!=���       �	�VxYc�A�5*

loss�!=�W�       �	��VxYc�A�5*

losss�"=a��r       �	CWxYc�A�5*

loss�X�<�S       �	�WxYc�A�5*

losslB=c���       �	�yXxYc�A�5*

loss�rc<K	       �	IYxYc�A�5*

loss�'�<�+Z       �	��YxYc�A�5*

lossQZi;���!       �	�DZxYc�A�5*

loss��;�b��       �	��ZxYc�A�5*

loss���<��X�       �	�w[xYc�A�5*

loss�}�<)V�       �	g\xYc�A�5*

loss�$=��m       �	j�\xYc�A�5*

lossX�<	e|�       �	�;]xYc�A�5*

loss��;ؕz�       �	��]xYc�A�5*

loss��=�       �	+�^xYc�A�5*

lossl�b<���,       �	�_xYc�A�5*

lossh�
=�:�       �	ظ_xYc�A�5*

loss���<Ce�       �	�`xYc�A�5*

loss�� <�g}h       �	�KaxYc�A�5*

lossMF{=e�(�       �	��axYc�A�5*

loss҂g<$e�=       �	x~bxYc�A�5*

lossW#M<��$D       �	cxYc�A�5*

loss�0�;�L'       �	�cxYc�A�5*

loss�r�;��F       �	&QdxYc�A�5*

lossm�<�a�       �	��dxYc�A�5*

loss��</r��       �	w�exYc�A�5*

lossC�<.j}       �	�,fxYc�A�5*

loss�u�<�^       �	��fxYc�A�5*

loss7 =2cu       �	�`gxYc�A�5*

loss��	=٢��       �	n�gxYc�A�5*

loss.�^<�P=|       �	��hxYc�A�5*

loss�*;<��       �	�4ixYc�A�5*

losse��<�.��       �	��ixYc�A�5*

loss�:�;��̗       �	�jxYc�A�5*

loss�)=�]w�       �	9kxYc�A�5*

loss�PJ=�S       �	��kxYc�A�5*

loss$d^=��d�       �	SzlxYc�A�5*

loss*�=)�Z       �	� mxYc�A�5*

lossZ�<���       �	��mxYc�A�5*

loss�� =�c�       �	��nxYc�A�5*

losscN<���       �	=+oxYc�A�5*

loss�l�<Jk�q       �	��oxYc�A�5*

loss�= #X       �	VepxYc�A�5*

loss�X�=���       �	�qxYc�A�5*

losstT�<�L�!       �	��qxYc�A�5*

loss@��;\�&�       �	�DrxYc�A�5*

loss]='`��       �	��rxYc�A�5*

loss���<AZ�       �	�vsxYc�A�5*

loss��:�j�       �	�txYc�A�5*

loss�[�;�?�t       �	�txYc�A�5*

loss@\<�L��       �	�HuxYc�A�5*

lossh@=�q�       �	�vxYc�A�5*

loss*�q=<}	       �	?�vxYc�A�5*

loss� �<e�v       �	��wxYc�A�5*

loss3=c�!       �	lxxYc�A�5*

loss�tZ<�-��       �	�yxYc�A�5*

loss%"�<�'��       �	�yxYc�A�5*

loss���;Z H�       �	�RzxYc�A�5*

loss�;;B)2�       �	��zxYc�A�5*

loss�i<vOx       �	V�{xYc�A�5*

lossfxw<�sa�       �	�m|xYc�A�5*

loss�
;�Ħ�       �	�}xYc�A�5*

loss���;��5[       �	x�}xYc�A�5*

loss�P)=L)�>       �	0�~xYc�A�5*

loss��;�S�       �	�mxYc�A�5*

loss���<I�[       �	�xYc�A�5*

loss�E=u��m       �	���xYc�A�5*

loss���<?��       �	�G�xYc�A�5*

loss*�<�O|       �	"�xYc�A�5*

loss�L�<x#       �		�xYc�A�5*

loss�$<�G�       �	"��xYc�A�5*

loss�c=g��       �	I�xYc�A�5*

loss��L<���       �	��xYc�A�5*

loss��Y=�xy       �	�y�xYc�A�5*

loss�f�=�?U       �	#�xYc�A�5*

loss� <0���       �	��xYc�A�5*

loss�)�;1�       �	5@�xYc�A�5*

loss��<1;k�       �	^�xYc�A�5*

loss��!<Z��       �	��xYc�A�5*

loss�e�<�U�       �	1�xYc�A�6*

lossT��<��m       �	ҋ�xYc�A�6*

loss�y�<>�%�       �	2!�xYc�A�6*

lossL�<ָ*       �	4��xYc�A�6*

loss]l�<�X�       �	�P�xYc�A�6*

loss[;�;�(D�       �	��xYc�A�6*

loss��R<�m%3       �	Ƈ�xYc�A�6*

loss� �<&0/       �	�H�xYc�A�6*

loss
S<�?�^       �	��xYc�A�6*

loss���<��       �	C��xYc�A�6*

lossLb�<(�3~       �	�/�xYc�A�6*

lossa�=�$��       �	�ҐxYc�A�6*

loss��<�ϳ�       �	]j�xYc�A�6*

loss�<z�       �	*�xYc�A�6*

lossX�<��k       �	9��xYc�A�6*

loss�B=)��3       �	�X�xYc�A�6*

loss�v�< �۪       �	��xYc�A�6*

loss�=�A"�       �	���xYc�A�6*

lossl�B=ɳ$T       �	;�xYc�A�6*

loss�m(<��1�       �	�ѕxYc�A�6*

loss��J<�Ѩ4       �	�f�xYc�A�6*

loss���;�Ԅ�       �	 �xYc�A�6*

loss��;<� �       �	���xYc�A�6*

loss�v=�1Q�       �	�;�xYc�A�6*

loss��G<~�o\       �	l�xYc�A�6*

loss�۬<��7       �	ӆ�xYc�A�6*

loss�=�;HFH/       �	uW�xYc�A�6*

lossoC=)��       �	�xYc�A�6*

loss�#<{��       �	H��xYc�A�6*

loss��;-)       �	F&�xYc�A�6*

lossTS<c=�N       �	���xYc�A�6*

loss���;ҽf       �	�X�xYc�A�6*

loss��+<�]�l       �	c�xYc�A�6*

loss;�=N�s
       �	���xYc�A�6*

loss��Y=����       �	}@�xYc�A�6*

losso��<�	܋       �	�ԟxYc�A�6*

loss��=�~�       �	@i�xYc�A�6*

loss���<	V�       �	X�xYc�A�6*

loss썌<_���       �	ٱ�xYc�A�6*

loss��<g�A�       �	�I�xYc�A�6*

loss2� =�B       �	��xYc�A�6*

loss�L�<cGt�       �	���xYc�A�6*

loss��.<l$�       �	6��xYc�A�6*

lossd^F=���       �	M1�xYc�A�6*

lossV�8;��o�       �	�ɥxYc�A�6*

loss��s<��J�       �	jh�xYc�A�6*

loss���<�l}�       �	���xYc�A�6*

lossEΥ<�۹       �	���xYc�A�6*

loss}{�<��l�       �	�>�xYc�A�6*

loss1|�<�h��       �	�)�xYc�A�6*

loss���<��0i       �	�ҪxYc�A�6*

loss�4h<��";       �	Dl�xYc�A�6*

loss��^==N�       �	��xYc�A�6*

loss��_=
��K       �	EجxYc�A�6*

loss��<���       �	�v�xYc�A�6*

lossϹ.=��8�       �	�	�xYc�A�6*

loss�H�;����       �	ӡ�xYc�A�6*

lossPD<����       �	�<�xYc�A�6*

loss��&<�V�b       �	�u�xYc�A�6*

loss%�<h���       �	J�xYc�A�6*

lossw�2<��       �	M��xYc�A�6*

loss$Z7=/f`       �	�7�xYc�A�6*

loss���<[�5�       �	t�xYc�A�6*

lossR��;Ki�n       �	���xYc�A�6*

loss]��<�S�7       �	�2�xYc�A�6*

loss�i=�Yw       �	�״xYc�A�6*

loss�(�<��E       �	��xYc�A�6*

loss�߸<v���       �	�>�xYc�A�6*

loss D=��Q       �	ܶxYc�A�6*

losst�/<Ȝ�       �	�x�xYc�A�6*

loss8m#<����       �	��xYc�A�6*

loss�q�;���	       �	I��xYc�A�6*

loss�w6;�MD       �	&W�xYc�A�6*

lossK�<�j�k       �	FD�xYc�A�6*

loss2�<,�~       �	UߺxYc�A�6*

loss��<v,��       �	���xYc�A�6*

loss ��<��       �	��xYc�A�6*

loss4<�Jg       �	���xYc�A�6*

lossHP=�i�&       �	 c�xYc�A�6*

loss /�;cW;�       �	i�xYc�A�6*

lossU�=hT�x       �	٘�xYc�A�6*

loss�z=�I��       �	�5�xYc�A�6*

lossT =f�/�       �	�пxYc�A�6*

loss-EQ<	���       �	wi�xYc�A�6*

loss!}I=7�$       �	�	�xYc�A�6*

lossܖr<�-[�       �	���xYc�A�6*

lossՄ�<9���       �	�G�xYc�A�6*

loss�-<K�M       �	���xYc�A�6*

lossF�=|��       �	$~�xYc�A�6*

lossJ��<�;�e       �	GV�xYc�A�6*

loss�G<�PS	       �	���xYc�A�6*

lossc�N=���       �	{��xYc�A�6*

loss�'�<�U�       �	H�xYc�A�6*

lossU�<O��       �	[��xYc�A�6*

loss1��<1���       �	zV�xYc�A�6*

lossX�=K=,�       �	���xYc�A�6*

lossC�1=�	��       �	���xYc�A�6*

lossW��<��pg       �	j�xYc�A�6*

loss���<��4a       �	�xYc�A�6*

loss��<���       �	A��xYc�A�6*

lossXE�=�zv�       �	�8�xYc�A�6*

loss�_<�4�       �	H��xYc�A�6*

lossv��<����       �	�v�xYc�A�6*

lossWn=Twg�       �	��xYc�A�6*

loss�;=��g       �	���xYc�A�6*

loss_��;����       �	.U�xYc�A�6*

loss!�<Q
��       �	���xYc�A�6*

loss&��<$.       �	��xYc�A�6*

loss��<E       �	�G�xYc�A�6*

lossI.H=��f       �	)��xYc�A�6*

lossw��<ڰlE       �	E��xYc�A�6*

lossZ�*;�       �	�(�xYc�A�6*

loss�?�=����       �	���xYc�A�6*

loss�=�`       �	�f�xYc�A�6*

loss�
�<_,10       �	"��xYc�A�6*

loss��<�@_)       �	,��xYc�A�6*

loss6r�<����       �	%��xYc�A�6*

loss�(c<AҴI       �	���xYc�A�6*

lossW�8<�-9        �	¢�xYc�A�6*

loss�Z�<�4��       �	?s�xYc�A�6*

loss�b�<��6       �	T��xYc�A�6*

loss#��;�ߋ       �	�R�xYc�A�6*

loss��<|B=)       �	w,�xYc�A�6*

lossI<nG�@       �	5��xYc�A�6*

loss���<�C��       �	!#�xYc�A�6*

loss�<�;mP7p       �	���xYc�A�6*

lossf��;��ɺ       �	�`�xYc�A�6*

loss��<���       �	h	�xYc�A�6*

loss���;V�_       �	�$�xYc�A�7*

loss͋*=V���       �	��xYc�A�7*

loss�6�=���       �	h��xYc�A�7*

lossx~*=�X~h       �	���xYc�A�7*

loss;W=����       �	�H�xYc�A�7*

loss��J<Nq�R       �	$��xYc�A�7*

loss�s�<x̷       �	���xYc�A�7*

lossH��<V�-       �	�C�xYc�A�7*

lossLI�;��       �	@��xYc�A�7*

loss���<%˕       �	�|�xYc�A�7*

loss�#[<H��R       �	W#�xYc�A�7*

loss�^�<���4       �	�9�xYc�A�7*

loss�5%<��R�       �	l��xYc�A�7*

lossN��<�[*       �	��xYc�A�7*

lossF�f<�Ӳ       �	:@�xYc�A�7*

losso-=W��       �	�
�xYc�A�7*

loss��.<���       �	���xYc�A�7*

loss�EG<%�7       �	ū�xYc�A�7*

loss�i =@~�       �	<O�xYc�A�7*

loss�d<�`��       �	��xYc�A�7*

lossElJ<�8��       �	a��xYc�A�7*

loss�1�<�Z       �	׆�xYc�A�7*

loss� ;�NT�       �	X9�xYc�A�7*

lossϟ�<,���       �	?��xYc�A�7*

losslY�<���W       �	���xYc�A�7*

loss�|�<Z�\       �	���xYc�A�7*

lossğx<���g       �	�p�xYc�A�7*

loss,ή<
���       �	�
�xYc�A�7*

loss�m�<�«�       �	ʨ�xYc�A�7*

loss&�k<�l`�       �	�@�xYc�A�7*

lossʽ�;���f       �	���xYc�A�7*

lossA�=�K�       �	��xYc�A�7*

lossA�k<ł�       �	��xYc�A�7*

lossm�6<`9xL       �	o��xYc�A�7*

loss�_�<]e�G       �	�S�xYc�A�7*

loss���;�J�9       �	>��xYc�A�7*

lossC�!;,�       �	Ӈ�xYc�A�7*

loss��y<z�y�       �	�$�xYc�A�7*

losse1�;�H�       �	T��xYc�A�7*

loss�d�;�N�       �	yt�xYc�A�7*

loss|�=K�$�       �	Y�xYc�A�7*

loss%��=�4.u       �	?��xYc�A�7*

loss�a�< �A;       �	JE�xYc�A�7*

loss���<���       �	���xYc�A�7*

lossQ��;F_/F       �	�z�xYc�A�7*

loss�F/=l��g       �	R�xYc�A�7*

loss�7<��D'       �	���xYc�A�7*

loss��:����       �	�� yYc�A�7*

losst�^=�[��       �	�+yYc�A�7*

loss���<U��<       �	:�yYc�A�7*

loss�6=�&�E       �	0cyYc�A�7*

loss4Ɠ<g%C�       �	U�yYc�A�7*

loss��*<�;�w       �	�yYc�A�7*

loss)c;�'�:       �	e7yYc�A�7*

loss���;���       �	l�yYc�A�7*

loss��<���       �	ʈyYc�A�7*

losslC'=$B�       �	�8yYc�A�7*

loss���<�+,�       �	��yYc�A�7*

loss4��<0,�N       �	�~yYc�A�7*

lossQ,=t�l       �	�AyYc�A�7*

loss��;*o[       �	��yYc�A�7*

loss�4=�f�       �	R}	yYc�A�7*

loss�i;��ט       �	
yYc�A�7*

lossJ�
<��^       �	�
yYc�A�7*

loss��<rg�G       �	�RyYc�A�7*

lossY;�<Ym�0       �	��yYc�A�7*

loss ��<�L�       �	|�yYc�A�7*

loss��Y<�:k�       �	�3yYc�A�7*

loss�<=5�       �	�yYc�A�7*

loss`Pf=hjI       �	�cyYc�A�7*

loss��.=���       �	�eyYc�A�7*

loss�9Y=Ͼ*       �	�yYc�A�7*

loss���<V&م       �	(�yYc�A�7*

loss��=r�m�       �	AyYc�A�7*

loss���<A�       �	��yYc�A�7*

loss�xn<9E�       �	xyYc�A�7*

loss]�<(��{       �	wyYc�A�7*

lossCH<��T       �	.�yYc�A�7*

lossI�;ĭq�       �	�HyYc�A�7*

loss1-�;"n       �	>�yYc�A�7*

loss�#�<�0�       �	��yYc�A�7*

loss�=��&�       �	��yYc�A�7*

loss<B<����       �	��yYc�A�7*

loss��; �/       �	�$yYc�A�7*

loss�e<|�4@       �	�^yYc�A�7*

lossd�S=��{       �	uyYc�A�7*

loss���;2��W       �	��yYc�A�7*

loss���<��52       �	^IyYc�A�7*

loss2p6=���       �	��yYc�A�7*

loss��@=���       �	�yYc�A�7*

loss|�H<CA�^       �	��yYc�A�7*

loss�n�=�lE       �	�FyYc�A�7*

loss4�=<$|7       �	��yYc�A�7*

loss,�;u�       �	WyyYc�A�7*

loss��3:��Ҷ       �	�x yYc�A�7*

lossa��;ퟕ�       �	�!yYc�A�7*

lossr�=5�?�       �	ݴ!yYc�A�7*

loss��u<����       �	�O"yYc�A�7*

loss��M<���       �	T�"yYc�A�7*

loss�Y�:�!%�       �	��#yYc�A�7*

lossE8<�Sg
       �	6x$yYc�A�7*

lossY�<��       �	U%yYc�A�7*

loss�Z:��~�       �	Q�%yYc�A�7*

loss ��9h��3       �	�S&yYc�A�7*

loss�/s;��^�       �	k�&yYc�A�7*

loss�a�<ޟ��       �	z�'yYc�A�7*

loss��;��%       �	�$(yYc�A�7*

loss��;��L       �	�(yYc�A�7*

loss�I]; ��       �	I�)yYc�A�7*

lossojx={       �	�?*yYc�A�7*

loss��6<�b�K       �	��*yYc�A�7*

loss��~;���h       �	t|+yYc�A�7*

loss,��<�Ҿ       �	,yYc�A�7*

loss{��<'[��       �	h�,yYc�A�7*

lossn��<PM$       �	�L-yYc�A�7*

loss�-*<V]L[       �	��-yYc�A�7*

loss���=�l�       �	��.yYc�A�7*

lossmW=0��       �	�%/yYc�A�7*

loss�{�<O��       �	u0yYc�A�7*

loss5�<�i       �	r�1yYc�A�7*

loss��n<�;       �	~2yYc�A�7*

lossfk�<��l       �	�v3yYc�A�7*

loss�cO=	��       �	�4yYc�A�7*

lossX�<	�       �	6�4yYc�A�7*

loss̞)<P�\h       �	�\5yYc�A�7*

lossq¯<��       �	f�5yYc�A�7*

loss�3,<��m\       �	>?7yYc�A�7*

loss͈�<9M7�       �	��7yYc�A�7*

lossC��<�U�p       �	��8yYc�A�8*

lossW�<�f�       �	�&9yYc�A�8*

loss<��;6���       �	��9yYc�A�8*

loss��<~�˹       �	YQ:yYc�A�8*

loss$�x<L��       �	,�:yYc�A�8*

loss%��:����       �	�;yYc�A�8*

loss��;��       �	�=<yYc�A�8*

loss�%�;�wM*       �	��<yYc�A�8*

loss�g�;T�NO       �	�|=yYc�A�8*

loss�D�;Q���       �	U>yYc�A�8*

loss�)�<�M�       �	�>yYc�A�8*

loss��?=�?��       �	�H?yYc�A�8*

losshI;�"a7       �	��?yYc�A�8*

loss��&<!�2�       �	Ȳ@yYc�A�8*

loss���;�V       �	nOAyYc�A�8*

lossԖ<AZ`       �	:�AyYc�A�8*

loss��<=cs�       �	�~ByYc�A�8*

loss2o�:��       �	uCyYc�A�8*

loss��=[@8       �	ԵCyYc�A�8*

loss�"2=����       �	ODyYc�A�8*

lossC�<�X       �	��DyYc�A�8*

losse��<���       �	��EyYc�A�8*

loss@��:q���       �	M/FyYc�A�8*

loss�Є=��Z�       �	�xGyYc�A�8*

loss��;��v/       �	�HyYc�A�8*

loss$�<`���       �	v�HyYc�A�8*

loss$Y~;�9�)       �	�FIyYc�A�8*

loss ��<��=�       �	��IyYc�A�8*

loss1%5=�Am       �	�JyYc�A�8*

lossы�<�G�       �	iKyYc�A�8*

losso�E<�5.       �	��KyYc�A�8*

loss��c;|y��       �	]PLyYc�A�8*

loss.$�<�-�       �	$�LyYc�A�8*

loss��/<h�� 