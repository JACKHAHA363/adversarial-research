       �K"	  ��Xc�Abrain.Event:2��R!�     9���	�e��Xc�A"��
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
:@*
seed2��1*
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2���*
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
seed2�ַ
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
T0*
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
seed2��
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
_output_shapes
:	�
*
seed2Ʌ�*
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
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2���*
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��`*
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
 *   @*
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
 *   @*
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
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
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
valueB *
_output_shapes
: *
dtype0
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
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
_output_shapes
: 
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
4gradients/softmax_cross_entropy_loss_1/Sum_grad/TileTile7gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape5gradients/softmax_cross_entropy_loss_1/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
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
: "�Aab�,     ���{	�˧�Xc�AJ��
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
:@*
seed2��1
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
seed2�ַ
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
seed2��
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
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2Ʌ�
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
:���������@*
seed2���
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2��`
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
 *   @*
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
valueB: *
dtype0*
_output_shapes
:
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
 *   @*
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
valueB *
dtype0*
_output_shapes
: 
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
9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2RealDiv9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_1#softmax_cross_entropy_loss_1/Select*
_output_shapes
: *
T0
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
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"V
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0H��n       ��-	��ׅXc�A*

loss�@M�       ��-	��؅Xc�A*

loss�`@/p�Z       ��-	ڒمXc�A*

lossjP@���       ��-	b�څXc�A*

loss��?C�1�       ��-	R�ۅXc�A*

lossRS�?�cbm       ��-	`܅Xc�A*

loss���?|       ��-	K:݅Xc�A*

loss��z?}�/�       ��-	��݅Xc�A*

loss��?�3       ��-	�ޅXc�A	*

lossM��?�iV�       ��-	zS߅Xc�A
*

loss1v]?��S       ��-		�߅Xc�A*

loss�1?�qg       ��-	)���Xc�A*

loss�"?ȷڍ       ��-	�9�Xc�A*

loss�8?��G       ��-	[��Xc�A*

loss�,?<pj)       ��-	m��Xc�A*

loss}�?����       ��-	D3�Xc�A*

loss���>����       ��-	���Xc�A*

loss>�?-�Hh       ��-	}y�Xc�A*

loss��?,�v�       ��-	��Xc�A*

loss{M?��       ��-	f��Xc�A*

loss�_?N�	       ��-	�\�Xc�A*

loss�!?gqi        ��-	~��Xc�A*

loss�?��$
       ��-	��Xc�A*

loss�A?�uSU       ��-	�=�Xc�A*

lossv�?�-jh       ��-	���Xc�A*

lossw�>H�Ӹ       ��-	ӆ�Xc�A*

loss ��>ˈ2�       ��-	�.�Xc�A*

loss��?2��       ��-	���Xc�A*

loss%�?���       ��-	s�Xc�A*

lossO 3?=�=�       ��-	��Xc�A*

loss���>&{��       ��-	���Xc�A*

loss�Y�>��;�       ��-	xc�Xc�A *

loss�J?�q�\       ��-	���Xc�A!*

loss�P)?T��4       ��-	��Xc�A"*

loss���>c��       ��-	\8�Xc�A#*

loss��?��Yn       ��-	���Xc�A$*

loss�S�>�
�       ��-	Wy��Xc�A%*

loss�
?BLW       ��-	)#�Xc�A&*

loss�7?����       ��-	[��Xc�A'*

loss��'?��U'       ��-	-z�Xc�A(*

losss"�>�*i       ��-	x)�Xc�A)*

loss3W�>p|�       ��-	��Xc�A**

lossC
�>Y�T       ��-	8j�Xc�A+*

loss��>\{�       ��-	x
��Xc�A,*

loss���>wL�       ��-	����Xc�A-*

loss_6?��A       ��-	�C��Xc�A.*

loss`A�>�2��       ��-	����Xc�A/*

loss���>k]��       ��-	�~��Xc�A0*

lossĚ�>�h��       ��-	� ��Xc�A1*

loss�BR>5ur       ��-	D���Xc�A2*

lossV�>�;�       ��-	�b��Xc�A3*

loss�i�>�} �       ��-	���Xc�A4*

loss�w�>U��       ��-	����Xc�A5*

loss6yJ?���       ��-	\X��Xc�A6*

lossA@q>2ǸK       ��-	��Xc�A7*

lossx �>
���       ��-	����Xc�A8*

loss��{>�ݚ�       ��-	zr��Xc�A9*

lossaE>\��       ��-	���Xc�A:*

loss�~	?�H|�       ��-	S���Xc�A;*

loss��>�Ǜ       ��-	�D��Xc�A<*

loss��>��       ��-	m���Xc�A=*

loss���>bw��       ��-	�} �Xc�A>*

lossL�u>)�5       ��-	��Xc�A?*

loss�G�>!,r�       ��-	y��Xc�A@*

loss�a�>����       ��-	�P�Xc�AA*

loss�v�>=��$       ��-	���Xc�AB*

loss3�?�>�"       ��-	��Xc�AC*

loss�e�>�|       ��-	�6�Xc�AD*

loss�3?�ǜ       ��-	h��Xc�AE*

lossjQ�>XVb�       ��-	�g�Xc�AF*

loss�Fo>
|�p       ��-	���Xc�AG*

loss=�>�x��       ��-	%��Xc�AH*

loss��]>�TE       ��-	=,�Xc�AI*

loss]��>N�       ��-	��Xc�AJ*

lossRP�>p�t�       ��-	(b�Xc�AK*

lossJ5�>���       ��-	{��Xc�AL*

loss��?�&��       ��-	\�	�Xc�AM*

lossQ�+?pp1       ��-	6
�Xc�AN*

loss���>l��       ��-	��
�Xc�AO*

loss���>�6.       ��-	�d�Xc�AP*

loss\f�>�IS�       ��-	��Xc�AQ*

loss;�>=�`       ��-	̵�Xc�AR*

loss��>����       ��-	�U�Xc�AS*

loss6Oe>�)�       ��-	���Xc�AT*

loss��>˳b�       ��-	���Xc�AU*

loss�6]>��h�       ��-	�4�Xc�AV*

loss�~�>-��       ��-	S?�Xc�AW*

loss��>�[l       ��-	O��Xc�AX*

loss蔩>ɪ�*       ��-	��Xc�AY*

loss�2Y>X71�       ��-	�"�Xc�AZ*

lossn��>k�       ��-	/��Xc�A[*

lossXq>�3�}       ��-	(��Xc�A\*

loss<�>*ٷ�       ��-	@�Xc�A]*

loss���>Qԭ�       ��-	���Xc�A^*

loss͞�>�q��       ��-	�~�Xc�A_*

losslm>>��h�       ��-	Y�Xc�A`*

loss�6�>.�Q�       ��-	B��Xc�Aa*

lossl��>���       ��-	wK�Xc�Ab*

loss6�?��S       ��-	��Xc�Ac*

loss&v>R�O�       ��-	){�Xc�Ad*

losss�i>���       ��-	{�Xc�Ae*

loss��>n��       ��-	���Xc�Af*

loss���>E��C       ��-	�D�Xc�Ag*

loss8�*>-ɗ}       ��-	���Xc�Ah*

loss�m>H�o�       ��-	ٲ�Xc�Ai*

loss�D>�{�       ��-	�M�Xc�Aj*

loss�h>-�)       ��-	��Xc�Ak*

loss��*>><�       ��-	�}�Xc�Al*

lossѕ�>g߻�       ��-	��Xc�Am*

loss��>s�o4       ��-	��Xc�An*

loss�>V(��       ��-	�]�Xc�Ao*

loss�h�>.��       ��-	> �Xc�Ap*

loss�dN>�t�       ��-	�� �Xc�Aq*

loss��=S�g:       ��-	�E!�Xc�Ar*

loss%��=d�q�       ��-	2�!�Xc�As*

loss�>ч       ��-	b�"�Xc�At*

loss&�8>NW�       ��-	�$#�Xc�Au*

loss�W>�@�       ��-	��#�Xc�Av*

loss�o>�3��       ��-	�q$�Xc�Aw*

loss�M�>���       ��-	�%�Xc�Ax*

loss��1>�pFu       ��-	��%�Xc�Ay*

lossr�^>�4�       ��-	ȴ&�Xc�Az*

loss�M>7�Y       ��-	�`'�Xc�A{*

loss?�>���%       ��-	q(�Xc�A|*

loss,��=P�1       ��-	l�(�Xc�A}*

loss+Q>�_�       ��-	�])�Xc�A~*

loss��o>�9`k       ��-	c	*�Xc�A*

loss;T�=d^[H       �	c�*�Xc�A�*

loss�P�>81�(       �	��+�Xc�A�*

loss}��>�{��       �	B	-�Xc�A�*

loss��>@ �.       �	*�-�Xc�A�*

loss�OB>~���       �	/�Xc�A�*

lossT�">qA�s       �	�V0�Xc�A�*

loss�V>xv�       �	O�1�Xc�A�*

loss��?>�;`       �	N|2�Xc�A�*

lossߦ�>�B�J       �	�3�Xc�A�*

lossY>�.�J       �	!�3�Xc�A�*

loss�>�Vc-       �	(�4�Xc�A�*

loss�'	>Nk��       �	�?5�Xc�A�*

loss%�=%0�'       �	��5�Xc�A�*

loss��o=4�v       �	��6�Xc�A�*

losszb>hhFR       �	�a7�Xc�A�*

loss��r>��u       �	!8�Xc�A�*

loss4�G>=v�@       �	\�8�Xc�A�*

loss�R]>�<d{       �	�Q9�Xc�A�*

loss[�=K��D       �	E�9�Xc�A�*

loss)�A>,�K       �	>�:�Xc�A�*

loss}�z=(��       �	1�;�Xc�A�*

loss�Ū=�Z�z       �	9<�Xc�A�*

loss�#�=�{K�       �	1�<�Xc�A�*

loss6/>�r0]       �		�=�Xc�A�*

loss�.�>�nź       �	�5>�Xc�A�*

loss���>���       �	.�>�Xc�A�*

loss��N>�D�       �	Q�?�Xc�A�*

lossj�>�t
o       �	&@�Xc�A�*

loss���=Ai�       �	x�@�Xc�A�*

loss���=��vw       �	�lA�Xc�A�*

loss؏P>�.��       �	�B�Xc�A�*

loss�	w>�P�^       �	��B�Xc�A�*

loss�e>�o�`       �	��C�Xc�A�*

loss]"=><U�x       �	U/D�Xc�A�*

loss�D:>{h�A       �	��D�Xc�A�*

loss�f6>c(       �	A~E�Xc�A�*

loss��=��W�       �	rF�Xc�A�*

loss�G@>����       �	��F�Xc�A�*

loss \>P��       �	�G�Xc�A�*

loss3��=��       �	� H�Xc�A�*

loss}�'> �J       �	8�H�Xc�A�*

loss_> +��       �	�YI�Xc�A�*

loss�+>�z	�       �	{�I�Xc�A�*

lossCa5>�2�       �	��J�Xc�A�*

loss�P�=E��       �	�&K�Xc�A�*

lossŚ[>���c       �	>�K�Xc�A�*

lossxY>��+       �	mL�Xc�A�*

loss��v=��Z�       �	XM�Xc�A�*

loss���=GZ�       �	Y�M�Xc�A�*

loss�*�>u��H       �	tCN�Xc�A�*

loss�y�>�Ha       �	��N�Xc�A�*

loss�6�= ,A       �	ytO�Xc�A�*

loss`��=z+       �	NP�Xc�A�*

lossis�=���,       �	��P�Xc�A�*

losstp>1ƕ�       �	�7Q�Xc�A�*

lossh�">��ݎ       �	��Q�Xc�A�*

loss���=���        �	c�R�Xc�A�*

loss��=����       �	�yS�Xc�A�*

loss %>�u�       �	�T�Xc�A�*

loss��*><M�       �	y�T�Xc�A�*

lossMa]>�       �	�BU�Xc�A�*

loss�� >��M�       �	I�U�Xc�A�*

loss�>��       �	�
W�Xc�A�*

loss�FL><�~�       �	¢W�Xc�A�*

lossE*>�w��       �	�TX�Xc�A�*

lossJ��=zauD       �	(�X�Xc�A�*

loss���=��أ       �	��Y�Xc�A�*

loss�`=,C�N       �	�Z�Xc�A�*

loss]:>I��#       �	O�Z�Xc�A�*

loss{�@>�a       �	�I[�Xc�A�*

loss�if>�H       �	��[�Xc�A�*

loss}b�>�c��       �	�|\�Xc�A�*

loss�m�=Z�M�       �	�]�Xc�A�*

loss�B%>��F       �	��]�Xc�A�*

loss	P�=	��k       �	�<^�Xc�A�*

loss���=A�-�       �	,�^�Xc�A�*

loss!�>�       �	
h_�Xc�A�*

loss�5>���$       �	]�_�Xc�A�*

loss���=��       �	m�`�Xc�A�*

loss��>|$B       �	,a�Xc�A�*

loss�>�p��       �	��a�Xc�A�*

loss6c>���       �	�\b�Xc�A�*

lossf1>���G       �	o�b�Xc�A�*

lossZ�>�*�_       �	�c�Xc�A�*

loss�>7Cړ       �	�%d�Xc�A�*

loss24=���       �	r�d�Xc�A�*

lossZ7>��       �	�Xe�Xc�A�*

lossa�L>P7t       �	,�e�Xc�A�*

loss3�G>Sf       �	-�f�Xc�A�*

loss���=��LJ       �	%=g�Xc�A�*

loss�t�>��_       �	��g�Xc�A�*

loss�C�>|,��       �	[{h�Xc�A�*

loss�Ƥ>�p/K       �	~i�Xc�A�*

loss��j=��\j       �	��i�Xc�A�*

lossv,�=mv��       �	@jj�Xc�A�*

loss#Ӌ>%��^       �	Xk�Xc�A�*

loss��k>�Q>Y       �	��k�Xc�A�*

loss���=�I�O       �		5l�Xc�A�*

lossO��=S=�L       �	�Wm�Xc�A�*

lossԜ_>T@gf       �	+�m�Xc�A�*

lossoST>�v�M       �	$Eo�Xc�A�*

loss��=���       �	��o�Xc�A�*

loss�a�>�U�       �	��p�Xc�A�*

loss�R�=�yRT       �	�jq�Xc�A�*

lossh�[=]U��       �	�fr�Xc�A�*

loss��f>uW�       �	�s�Xc�A�*

loss3~>�:A       �	�s�Xc�A�*

loss��C>}�r�       �	�Lt�Xc�A�*

loss�<>��/       �	A�t�Xc�A�*

loss�9�=�MU�       �	K�u�Xc�A�*

lossC@h>�GXk       �	�_v�Xc�A�*

lossZ*>:b�       �	��v�Xc�A�*

loss@�S=P��]       �	^�w�Xc�A�*

loss7�>4+m       �	�Lx�Xc�A�*

loss*v�="�~/       �	� y�Xc�A�*

lossk�>x��}       �	��y�Xc�A�*

loss���=�6��       �	�Mz�Xc�A�*

loss�=}� 1       �	'�z�Xc�A�*

lossO4L>?v       �	��{�Xc�A�*

lossܣ�=��       �	�E|�Xc�A�*

loss���=�4��       �	D�|�Xc�A�*

lossr>9t��       �	s�}�Xc�A�*

loss�V)>ݧ��       �	�~�Xc�A�*

loss[m2>Y}       �	�~�Xc�A�*

loss�
>M_%       �	�U�Xc�A�*

loss�6>���0       �	���Xc�A�*

loss�=r�Rp       �	���Xc�A�*

loss ]�=����       �	8k��Xc�A�*

loss��	>��       �	!��Xc�A�*

loss�>�sh�       �	oԂ�Xc�A�*

loss>hH�       �	x��Xc�A�*

lossj�=|+�       �	�)��Xc�A�*

loss��>�D�       �	C��Xc�A�*

loss���=1��{       �	ɰ��Xc�A�*

lossū�=��        �	�J��Xc�A�*

loss�0
>q�       �	��Xc�A�*

lossd��=�`��       �	����Xc�A�*

loss33=>'���       �	0��Xc�A�*

loss2Z�=E�       �	}ʈ�Xc�A�*

loss(�>�7+       �	0g��Xc�A�*

lossې>;��       �	T ��Xc�A�*

loss�'x>�V�Y       �	���Xc�A�*

loss}��=���%       �	C��Xc�A�*

lossR��=��       �	�苆Xc�A�*

loss�=��U�       �	ʇ��Xc�A�*

loss�E�=G56�       �	�#��Xc�A�*

loss7R�=~���       �	׾��Xc�A�*

loss��5>�G;8       �	]��Xc�A�*

loss�c>/��       �	/���Xc�A�*

lossO�>�#>       �	[��Xc�A�*

loss��=���       �	���Xc�A�*

losso		>���       �	�푆Xc�A�*

loss�jL>��;�       �	敒�Xc�A�*

loss-�=�q       �	y=��Xc�A�*

lossŇ?=c&J}       �	���Xc�A�*

lossE�Z>r��       �	����Xc�A�*

loss��=fl�       �	�5��Xc�A�*

loss@]>�h̪       �	�
��Xc�A�*

loss37>�'       �	"���Xc�A�*

loss\1�=�ֽ       �	�C��Xc�A�*

loss$w�=����       �	&ߗ�Xc�A�*

lossr>:i�       �	�_��Xc�A�*

loss�Vq=4x�V       �	���Xc�A�*

loss��J=��       �	����Xc�A�*

loss�=>"H�I       �	tE��Xc�A�*

lossv�	=L���       �	�曆Xc�A�*

loss��u=�ʒ�       �	8���Xc�A�*

loss0�>;#�       �	���Xc�A�*

lossnIR>����       �	���Xc�A�*

loss�nk>&�n�       �	�P��Xc�A�*

losss;S>��2�       �	�?��Xc�A�*

loss�'>؃Ύ       �	�֟�Xc�A�*

lossO�`>s�)�       �	�u��Xc�A�*

loss-��=�F�0       �	��Xc�A�*

loss7�=��La       �	uȡ�Xc�A�*

loss��_=>�/�       �	坢�Xc�A�*

loss��>۱��       �	�=��Xc�A�*

lossw>W��5       �	ݣ�Xc�A�*

lossŁd>j���       �	�z��Xc�A�*

lossI&=��u�       �	���Xc�A�*

loss3�I>)xɳ       �	�ĥ�Xc�A�*

lossz�=���       �	�l��Xc�A�*

loss��>���&       �	��Xc�A�*

loss��=�KIU       �	���Xc�A�*

loss�	>���G       �	N��Xc�A�*

loss��W>����       �	�ꨆXc�A�*

loss��=���       �	4���Xc�A�*

loss�k�=���       �	�!��Xc�A�*

loss�<>>��       �	���Xc�A�*

lossa��=�VI       �	�R��Xc�A�*

loss�@�=���       �	�髆Xc�A�*

loss�7
>�=U\       �	x}��Xc�A�*

loss�L�=���       �	@��Xc�A�*

lossQ�=k9��       �	����Xc�A�*

loss�I^>�n�       �	TS��Xc�A�*

lossͧ=�TM�       �	���Xc�A�*

loss���=ᴾ�       �	���Xc�A�*

lossT"o>�QX/       �	D5��Xc�A�*

loss}�Q>��H�       �	ʰ�Xc�A�*

loss��>�RaA       �	�^��Xc�A�*

loss��=��]�       �	Q���Xc�A�*

loss��0>���*       �	D���Xc�A�*

loss[��=�Ƕs       �	WA��Xc�A�*

loss$�>���?       �	Eڳ�Xc�A�*

loss���=�$�       �	v��Xc�A�*

loss��N=ϟ[�       �	*��Xc�A�*

lossTF=}�	       �	���Xc�A�*

loss�ft>�$       �	�Q��Xc�A�*

loss&{c=��k       �	-���Xc�A�*

loss�|>�J��       �	ۆ��Xc�A�*

loss8�'>��\       �	��Xc�A�*

loss 	>
[��       �	,���Xc�A�*

loss�Lw=k���       �	{i��Xc�A�*

loss9�=�!�       �	\��Xc�A�*

lossr�]=Ե�\       �	𢺆Xc�A�*

loss�C%>���o       �	�A��Xc�A�*

lossF��=�	c�       �	�ڻ�Xc�A�*

loss�u>�.�@       �	&q��Xc�A�*

lossI>��v       �	���Xc�A�*

loss{CF=�=�       �	����Xc�A�*

loss��j>(��!       �	�?��Xc�A�*

loss}��=��       �	¤��Xc�A�*

loss�?+>q���       �	�G��Xc�A�*

loss� �=�Fݥ       �	����Xc�A�*

loss x>=�@�       �	R���Xc�A�*

lossD>Z��       �	�EXc�A�*

loss�"�=�e�       �	��Xc�A�*

loss�s(>"��       �	��ÆXc�A�*

loss�8=��8�       �	�#ĆXc�A�*

loss��=��Q�       �	:�ĆXc�A�*

losso��=^���       �	ReņXc�A�*

loss���=v�C�       �	_
ƆXc�A�*

loss�h=�ߜ2       �	;�ƆXc�A�*

loss���=��u       �	�DǆXc�A�*

loss�!>S&܂       �	�ȆXc�A�*

loss�T�=mV,L       �	n�ȆXc�A�*

lossd*P>p��6       �	�|ɆXc�A�*

loss��3>�       �	jʆXc�A�*

loss|#>��.�       �	��ʆXc�A�*

loss5s>��?�       �	?RˆXc�A�*

lossh�==�k�       �	(�ˆXc�A�*

loss�(>��S       �	Ŏ̆Xc�A�*

lossJ�>s��a       �	�/͆Xc�A�*

lossW$�=���{       �	0�͆Xc�A�*

lossml	>�[	�       �	�mΆXc�A�*

loss�h�=��s�       �	tφXc�A�*

loss��=+}       �	��φXc�A�*

lossqu >�u�       �	�CІXc�A�*

loss��=��6       �	��ІXc�A�*

loss&�)>[�m7       �	N�цXc�A�*

loss�H>�PN�       �	�T҆Xc�A�*

loss<}�=tbZ2       �	_�҆Xc�A�*

loss�o> �w]       �	^�ӆXc�A�*

loss�[2>��       �	\ԆXc�A�*

lossŕ'>�N'       �	��ԆXc�A�*

loss��>���(       �	M�ՆXc�A�*

loss:��=>�        �	�)ֆXc�A�*

losse�=�ץj       �	�ֆXc�A�*

loss�f>̇P�       �	~�׆Xc�A�*

loss�2�=�%��       �	M2؆Xc�A�*

loss��;>���<       �	��؆Xc�A�*

loss a�=0p��       �	�نXc�A�*

loss!��>�<�r       �	��چXc�A�*

loss�#j>��"�       �	��ۆXc�A�*

loss�k,>vS�Z       �	}�܆Xc�A�*

loss�>��A;       �	�R݆Xc�A�*

loss7�=�hc�       �	��݆Xc�A�*

lossI�	>����       �	wJ߆Xc�A�*

loss!�?>c��v       �	��߆Xc�A�*

loss�(^=�:֡       �	E���Xc�A�*

lossãz=�n�       �	��Xc�A�*

loss{�~=�PQ1       �	Ѳ�Xc�A�*

loss�a=>����       �	'O�Xc�A�*

loss �>/��1       �	���Xc�A�*

loss��=��";       �	�z�Xc�A�*

loss�*�>�ΦO       �	=�Xc�A�*

loss��=A�;q       �	���Xc�A�*

loss@��=G˔�       �	N��Xc�A�*

loss$	�=wZ�~       �	���Xc�A�*

loss��>}o�       �	j�Xc�A�*

losslȑ=�}�       �	O��Xc�A�*

loss4�>>�<       �	�E�Xc�A�*

loss]G>���       �	���Xc�A�*

lossQ��=V�g6       �	<��Xc�A�*

lossJ�>u�I@       �	�*�Xc�A�*

lossv�>�ݓT       �	:��Xc�A�*

loss-U>s�       �	�j�Xc�A�*

loss�fo>��       �	9�Xc�A�*

loss�ű=�6�       �	���Xc�A�*

loss�>�=%�^�       �	���Xc�A�*

loss�#T=����       �	�Q�Xc�A�*

loss��=S��F       �	���Xc�A�*

loss��P>��       �	��Xc�A�*

loss=��=}~��       �	�`��Xc�A�*

loss���=���       �	��Xc�A�*

loss���=�<��       �	��Xc�A�*

loss�O>p��       �	QM�Xc�A�*

lossJU1>8���       �	���Xc�A�*

loss~�>�|`[       �	v��Xc�A�*

loss���=xr�       �	_A�Xc�A�*

loss!F7>j�
L       �	���Xc�A�*

lossd�>!~�       �	Sy��Xc�A�*

loss؄�=�p�       �	���Xc�A�*

loss�Q
>��h<       �	����Xc�A�*

lossd>�n       �	(e��Xc�A�*

lossf�=�[��       �	)��Xc�A�*

loss�m=�wS       �	���Xc�A�*

loss��P>E��L       �	F��Xc�A�*

lossel=t#n�       �	@���Xc�A�*

loss�\�=_���       �	����Xc�A�*

loss�m�=��7_       �	���Xc�A�*

loss�uo=2�L�       �	�+��Xc�A�*

loss~Q�=�4�       �	���Xc�A�*

loss�}k>�^X�       �	Xp��Xc�A�*

loss@�>�;�[       �	{��Xc�A�*

loss
'O=e��K       �	���Xc�A�*

lossh�=�=       �	�Q��Xc�A�*

loss���=��       �	A���Xc�A�*

loss��>Zn��       �	a� �Xc�A�*

loss�f]>u?��       �	�"�Xc�A�*

loss}��=�њ+       �	���Xc�A�*

loss_S= I�       �	�W�Xc�A�*

lossq>�'�       �	P��Xc�A�*

loss3�M>.Q�&       �	З�Xc�A�*

loss�.�=_�zq       �	�-�Xc�A�*

loss
#�=}���       �	���Xc�A�*

loss�8>�!/       �	�a�Xc�A�*

lossX_0>BL�g       �	&��Xc�A�*

loss��=�0	J       �	���Xc�A�*

loss#f5>�թg       �	�(�Xc�A�*

losswE>��M       �	���Xc�A�*

loss`�#>����       �	�V�Xc�A�*

lossv�D>�9       �	A��Xc�A�*

loss��2=p�Ǘ       �	�	�Xc�A�*

loss}�=���       �	�+
�Xc�A�*

loss�<C=˯M�       �	��
�Xc�A�*

loss;.�>�B�       �	�]�Xc�A�*

loss�g=�P��       �	v��Xc�A�*

lossN��=s̲       �	`��Xc�A�*

lossۿ=�g�       �	�(�Xc�A�*

loss#(�=}�i�       �	 ��Xc�A�*

loss%%3>c�#:       �	�h�Xc�A�*

loss���<��z^       �	��Xc�A�*

lossLw>�ti       �	��Xc�A�*

loss<�=�z&�       �	
M�Xc�A�*

lossQ9>V�<       �	��Xc�A�*

loss}5i>�d�       �	n��Xc�A�*

loss�sp=9�I       �	#�Xc�A�*

loss�^L=�w�}       �	���Xc�A�*

loss���<�2��       �	X�Xc�A�*

lossJ�$=��'       �	���Xc�A�*

loss�=x���       �	֌�Xc�A�*

loss��X>�I       �	 (�Xc�A�*

loss��{>�r       �	˾�Xc�A�*

loss���=c-G       �	mX�Xc�A�*

loss�w=��;C       �	���Xc�A�*

losst` >�F`�       �	��Xc�A�*

loss�=�=��+�       �	�*�Xc�A�*

loss���=+��       �	n��Xc�A�*

loss��=�C��       �	h]�Xc�A�*

loss��=EV       �	�n�Xc�A�*

loss�WA=ȡP�       �	�H�Xc�A�*

loss�)>.�i�       �	1D�Xc�A�*

loss�E�=��       �	+��Xc�A�*

lossM��=��\�       �	u��Xc�A�*

loss��=5��       �	g�Xc�A�*

lossB=JzO�       �	��Xc�A�*

loss�x�=���~       �	���Xc�A�*

lossH�E=
�F       �	= �Xc�A�*

lossS>͕�;       �	� �Xc�A�*

loss��E=��j�       �	vk!�Xc�A�*

loss�L�=�nv�       �	� "�Xc�A�*

loss�>�=��3�       �	R�"�Xc�A�*

loss�cH>M~�"       �	0#�Xc�A�*

loss@��<H��N       �	��#�Xc�A�*

lossW�<FeG       �	�a$�Xc�A�*

loss�5=����       �	��$�Xc�A�*

loss��>}N��       �	��%�Xc�A�*

loss�p%=I�
       �	�2&�Xc�A�*

loss�>�6�       �	G�&�Xc�A�*

loss�ɤ=|^%+       �	9a'�Xc�A�*

loss3�>�+       �	��'�Xc�A�*

loss2�.>����       �		�(�Xc�A�*

loss��K=��%3       �	�$)�Xc�A�*

lossW�y=�3�       �	��)�Xc�A�*

lossɎ~=XP       �	�j*�Xc�A�*

loss.V�=xQ//       �	�+�Xc�A�*

lossϽ=u�       �	9�+�Xc�A�*

loss��=B 4�       �	J,�Xc�A�*

loss���=�:�o       �	%�,�Xc�A�*

loss�Ȃ=�CJ�       �	/�-�Xc�A�*

loss�jl=�p	       �	�#.�Xc�A�*

loss)�=FO�Y       �	�>/�Xc�A�*

loss�G4<I�T       �	�$0�Xc�A�*

loss¼<z�&�       �	��0�Xc�A�*

loss��*<���       �	�Z1�Xc�A�*

loss:ɍ=m��       �	2�Xc�A�*

loss=��=fUU,       �	��2�Xc�A�*

lossɱ�<;l       �	iR3�Xc�A�*

lossH/�=F�td       �	��3�Xc�A�*

loss���=���       �	¤4�Xc�A�*

loss,�O>+��       �	�B5�Xc�A�*

loss�qt=c�*�       �	'�5�Xc�A�*

loss`�8<V�,�       �	�~6�Xc�A�*

loss�/>��P       �	7�Xc�A�*

loss�?�<
Q��       �	�7�Xc�A�*

loss��n=iͫ�       �	�F8�Xc�A�*

loss�0�<��@N       �	]�8�Xc�A�*

lossi��<�p�
       �	�y9�Xc�A�*

loss�H�;NuN       �	|:�Xc�A�*

loss�~-=��v       �	��:�Xc�A�*

loss�܅<%�'�       �	6<;�Xc�A�*

loss�0>��C       �	��;�Xc�A�*

loss� @= ���       �	wi<�Xc�A�*

loss^�;�ƣ       �	�=�Xc�A�*

lossZ9A;L�$�       �	��=�Xc�A�*

loss�cZ=�H]       �	�3>�Xc�A�*

loss�E>fKc�       �	��>�Xc�A�*

loss3��=��P�       �	J_?�Xc�A�*

lossfd�;�y��       �	2�?�Xc�A�*

loss���<8t��       �	9�@�Xc�A�*

loss�ȳ>7H�       �	�/A�Xc�A�*

loss�X<<J��2       �	/�A�Xc�A�*

loss���=�X��       �	\B�Xc�A�*

loss�6�=�͘�       �	�C�Xc�A�*

loss3s�>BQ[_       �	c�C�Xc�A�*

lossf
�=dT2       �	'3D�Xc�A�*

loss��[=�;�       �	��D�Xc�A�*

loss��=��6       �	�_E�Xc�A�*

loss�h�=ɶ�E       �	�F�Xc�A�*

loss��>�a�       �	�F�Xc�A�*

loss���=���       �	�2G�Xc�A�*

loss!>�"�2       �	t�G�Xc�A�*

lossJ��>�5�v       �	LmH�Xc�A�*

loss�a>|B�=       �	� I�Xc�A�*

loss��>�S�       �	V�I�Xc�A�*

loss�!C>�Gwe       �	�7J�Xc�A�*

loss�M�=�=��       �	��J�Xc�A�*

loss|0g=��'       �	rpK�Xc�A�*

loss�>�MG�       �	c	L�Xc�A�*

loss ��= ��y       �	��L�Xc�A�*

loss��=�m�E       �	=EM�Xc�A�*

loss
��=fv"�       �	��M�Xc�A�*

loss���=���o       �	N�Xc�A�*

losss�d=����       �	�O�Xc�A�*

loss�E=R-@       �	0�O�Xc�A�*

loss��=.sUN       �	�^P�Xc�A�*

loss@.t=>���       �	v�P�Xc�A�*

loss�mw=�Nq       �	d�Q�Xc�A�*

lossBc�<w�       �	&:R�Xc�A�*

loss�.>��       �	#�R�Xc�A�*

lossx��=ouG�       �	�wS�Xc�A�*

loss@�m=� �=       �	�-T�Xc�A�*

lossV\=�        �	��T�Xc�A�*

lossǓ�=���&       �	�`U�Xc�A�*

loss���<�J�8       �	�U�Xc�A�*

loss_W^=�-�       �	t�V�Xc�A�*

loss�[=��       �	s/W�Xc�A�*

loss�'=f}jr       �	��W�Xc�A�*

loss���=n���       �	0�X�Xc�A�*

losss`�=����       �	�Y�Xc�A�*

loss!X�=<2�       �	��Y�Xc�A�*

loss�r�=�ֽ�       �	FZ�Xc�A�*

loss�6�=���d       �	r�Z�Xc�A�*

loss�=��<       �	n[�Xc�A�*

loss->�#ea       �	\\�Xc�A�*

loss(3,=ܫ��       �	C�\�Xc�A�*

loss��=zNb       �	K]�Xc�A�*

loss� �=���       �	h�]�Xc�A�*

loss�9y=	`|       �	.�^�Xc�A�*

lossX`�=���       �	W#_�Xc�A�*

loss欖==���       �	��_�Xc�A�*

loss?[=E}�	       �	TT`�Xc�A�*

lossAN�=3�B       �	� {�Xc�A�*

lossC7�=~��F       �	�{�Xc�A�*

lossH��=暴k       �	�T|�Xc�A�*

lossc��=K��       �	l�|�Xc�A�*

loss�_�=��w       �	+�}�Xc�A�*

loss���=��       �	R'~�Xc�A�*

loss&�=0-(|       �	H�~�Xc�A�*

loss,�v>5�1       �	�U�Xc�A�*

loss�� >��       �	���Xc�A�*

loss��=	�!v       �	���Xc�A�*

lossTQI=�|�p       �	0��Xc�A�*

loss?�n= )C       �	eǁ�Xc�A�*

loss8]�=���       �	�]��Xc�A�*

loss$��=2��U       �	7���Xc�A�*

lossX�>�@�8       �	���Xc�A�*

loss@�>ڹ�|       �	@3��Xc�A�*

loss�0u=��HF       �	̈́�Xc�A�*

loss��N=�֋U       �	4d��Xc�A�*

lossy�=�)�3       �	���Xc�A�*

loss���>�"X�       �	>���Xc�A�*

loss4�=.˲       �	�L��Xc�A�*

lossܗ�>�w       �	�뇇Xc�A�*

loss�Rs=���h       �	Y���Xc�A�*

lossa��=�Z       �	�0��Xc�A�*

loss��=v��j       �	�Չ�Xc�A�*

loss4�=/�=�       �	Po��Xc�A�*

loss��=?�+3       �	j��Xc�A�*

lossx��=��S       �	Ѱ��Xc�A�*

loss�H>#5       �	�G��Xc�A�*

lossљ�=p���       �	�Xc�A�*

loss��_=�Y       �	���Xc�A�*

loss!
}=���       �	0)��Xc�A�*

loss��
>�=�       �	�Ŏ�Xc�A�*

loss^m>�[�1       �	�d��Xc�A�*

loss���=	R�S       �	���Xc�A�*

loss��X>�+��       �	���Xc�A�*

loss��<�z]
       �	4��Xc�A�*

lossD�>?�n>       �	�ˑ�Xc�A�*

loss�Ec>�Z�d       �	 c��Xc�A�*

loss�1>Sݣ       �	����Xc�A�*

loss�'�=
eӰ       �	G���Xc�A�*

loss�F�=-�       �	���Xc�A�*

loss�Y=)�'       �	����Xc�A�*

loss^G>���}       �	�9��Xc�A�*

loss���=E!��       �	�ږ�Xc�A�*

loss��%>�/�       �	�u��Xc�A�*

loss�Y=Z���       �	��Xc�A�*

loss�`
=���       �	����Xc�A�*

lossL��=!�=       �	�V��Xc�A�*

loss�Y�<�]�       �	�뙇Xc�A�*

loss��h<;�Ѣ       �	����Xc�A�*

loss.�=k��:       �	&��Xc�A�*

losss�>@d�i       �	����Xc�A�*

lossCB�>�àa       �	�B��Xc�A�*

loss��'=(Y       �	�֜�Xc�A�*

loss�j=b�`       �	l��Xc�A�*

loss���<���       �	O��Xc�A�*

loss�J�<�N�       �	Y���Xc�A�*

lossnb> j+       �	/N��Xc�A�*

loss۳�=q3�d       �	L���Xc�A�*

loss��C=�p�       �	����Xc�A�*

loss:;�=���       �	]N��Xc�A�*

loss-�<C�       �	B���Xc�A�*

loss���=m�3+       �	'���Xc�A�*

loss،�=��j�       �	�#��Xc�A�*

loss���=Y��n       �	����Xc�A�*

lossD%>�d"�       �	�`��Xc�A�*

loss^�=���U       �	~���Xc�A�*

loss�>'{�8       �	����Xc�A�*

loss��>0.�]       �	K?��Xc�A�*

loss7do=|�j�       �	A֦�Xc�A�*

lossa�>�w\h       �	Ku��Xc�A�*

loss�E�=dϽ�       �	���Xc�A�*

loss�)�=z��       �	����Xc�A�*

loss�l�=�3��       �	�Y��Xc�A�*

loss?Tm=�j?�       �	����Xc�A�*

loss���=So�$       �	����Xc�A�*

loss;�
>C�v       �	�3��Xc�A�*

losse�>=`��N       �	�ի�Xc�A�*

loss�nj=���       �	�o��Xc�A�*

loss��=#�]�       �	�0��Xc�A�*

loss�E�=��I�       �	���Xc�A�*

loss�p�=
7�X       �	���Xc�A�*

loss�x�<�Y��       �	ӄ��Xc�A�*

lossZ>�=ٶF�       �	%��Xc�A�*

loss}}=�M@�       �	]���Xc�A�*

loss㬨=����       �	�`��Xc�A�*

lossZ2~=�{��       �	FA��Xc�A�*

lossӤ�=�+2�       �	wز�Xc�A�*

loss���<�N�       �	\u��Xc�A�*

loss�� >
��%       �	�f��Xc�A�*

loss���=�r       �	F��Xc�A�*

loss�v�=�#Ϳ       �	ܹ��Xc�A�*

loss,��=�G�       �	LQ��Xc�A�*

loss��=2�H       �	\!��Xc�A�*

loss7�<�h       �	����Xc�A�*

loss6�=��       �	zV��Xc�A�*

lossRH,>��W�       �	����Xc�A�*

loss�O]>��.,       �	>���Xc�A�*

loss��=����       �	B`��Xc�A�*

loss�Ԋ<��y[       �	����Xc�A�*

loss؞�<Jb�S       �	�Xc�A�*

losst�=ϸm�       �	6<��Xc�A�*

loss�P�=�hOG       �	d[��Xc�A�*

loss� �=�{�       �	��Xc�A�*

loss��=K$;n       �	2���Xc�A�*

loss��i=s���       �	F��Xc�A�*

loss�;�<�#Y4       �	3���Xc�A�*

loss�;S>�W?�       �	����Xc�A�*

loss���=Iqu�       �	M��Xc�A�*

loss�>�j��       �	�Xc�A�*

loss��=~e(�       �	(�Xc�A�*

loss�a<=��ܟ       �	�WÇXc�A�*

loss1&�<H!��       �	��ÇXc�A�*

loss�N�<�]�d       �	��ćXc�A�*

lossҵ�=�	       �	�:ŇXc�A�*

loss��=*�W       �	��ŇXc�A�*

loss�:>���       �	�mƇXc�A�*

loss�r=���       �	ǇXc�A�*

loss)��<sV�       �	��ǇXc�A�*

loss���=�#�)       �	 9ȇXc�A�*

lossq�q=2�ɹ       �	W�ȇXc�A�*

loss�=�=S*�       �	qɇXc�A�*

lossW��=\g�z       �	ʇXc�A�*

losso�<�UO�       �	�ʇXc�A�*

loss�cj=�z!       �	�MˇXc�A�*

lossF�>�xQ�       �	��ˇXc�A�*

lossh�X=�BV�       �	�̇Xc�A�*

lossY�=
.�p       �	rN͇Xc�A�*

lossr�=G��V       �	��͇Xc�A�*

lossDp�=���       �	
�·Xc�A�*

loss�C>��q       �	�/χXc�A�*

loss ĭ=߽�       �	��χXc�A�*

loss1�:=ɓ�       �	BwЇXc�A�*

loss�5>�F�Z       �	rчXc�A�*

lossO>�<¼�       �	��чXc�A�*

loss��l=:6�       �	�b҇Xc�A�*

loss��2=}:�U       �	ӇXc�A�*

lossO2�<Z��T       �	�ӇXc�A�*

lossjO�=�{E       �	+LԇXc�A�*

loss!�<[��       �	�ԇXc�A�*

lossiU=�Bך       �	ޑՇXc�A�*

loss��=�Z6       �	:ևXc�A�*

loss(t�=��@       �	:�ևXc�A�*

loss�±=0\,O       �	o�ׇXc�A�*

loss��'=�{{�       �	$؇Xc�A�*

loss�V�=:�~d       �	"�؇Xc�A�*

loss��/<��5j       �	��هXc�A�*

loss͐e<��c�       �	6ڇXc�A�*

losst=(�T*       �	��ڇXc�A�*

loss�T�<;�Z�       �	�ۇXc�A�*

lossHQ�>�<�       �	�i܇Xc�A�*

loss�0>)[u       �	�݇Xc�A�*

loss/f�=���       �	E�݇Xc�A�*

lossp��= />(       �	\9އXc�A�*

loss_T�<W�ƹ       �	�%߇Xc�A�*

loss��B=c;N�       �	��߇Xc�A�*

lossҔ�=��~       �	�]��Xc�A�*

loss�<*>�0�       �	j���Xc�A�*

loss��=��%       �	a��Xc�A�*

lossȹ>�6�       �	�(�Xc�A�*

loss;�==:m?       �	���Xc�A�*

loss�W=���       �	�q�Xc�A�*

loss��R<L9�       �	�	�Xc�A�*

loss��=:=�       �	��Xc�A�*

loss��=&:�       �	�q�Xc�A�*

loss4ʁ=�5L`       �	��Xc�A�*

loss�>�=~¬�       �	���Xc�A�*

loss�vo=K�h]       �	/0�Xc�A�*

loss��h=���7       �	���Xc�A�*

loss� >�^6       �	Dm�Xc�A�*

lossH�<=j%       �	q�Xc�A�*

loss�=-�+       �	J��Xc�A�*

lossq��=*��"       �	0�Xc�A�*

loss'<��       �	���Xc�A�*

loss4�=���       �	�n�Xc�A�*

loss}�>���       �	�Xc�A�*

loss���=����       �	��Xc�A�*

loss/_>$:��       �	T�Xc�A�*

loss��=�Qiz       �	�#�Xc�A�*

losse�<�ȃ       �	���Xc�A�*

loss\w�=>);       �	,��Xc�A�*

loss�s =�\��       �	3��Xc�A�*

lossx�=0�a       �	h���Xc�A�*

loss�f#<v�u       �	Eh�Xc�A�*

loss��=Y�`       �	%�Xc�A�*

loss��_=��!�       �	���Xc�A�*

loss!�=���B       �	O?�Xc�A�*

loss�e�=&Z�&       �	���Xc�A�*

loss���=�|]       �	�~�Xc�A�*

lossf�>�c�3       �	���Xc�A�*

lossa�s=8<]       �	���Xc�A�*

lossJ�=*y�|       �	�R��Xc�A�*

loss�<NLm       �	���Xc�A�*

lossxx�<r�T�       �	1���Xc�A�*

loss�>=y�x       �	nL��Xc�A�*

lossv[G=Z )       �	l���Xc�A�*

loss��=%�cP       �	Q���Xc�A�*

loss RP>���A       �	�8��Xc�A�*

losse�2=l��       �	����Xc�A�*

lossȃ=5��       �	"7��Xc�A�*

loss��z={���       �	6���Xc�A�*

loss�qq=|�/       �	���Xc�A�*

loss:��=:M�o       �	z��Xc�A�*

loss� J=�y?A       �	����Xc�A�*

loss�b�<�p�m       �	�a��Xc�A�*

loss��=ٻT       �	/ �Xc�A�*

lossct3=Z�D       �	4� �Xc�A�*

loss�;�=k2�6       �	�^�Xc�A�*

loss4;�=E��       �	8�Xc�A�*

lossZ�<: -v       �	���Xc�A�*

loss�#�=Iu��       �	7��Xc�A�*

loss�/=�;y       �	�"�Xc�A�*

lossN��=cn�k       �	%��Xc�A�*

loss���=�z��       �	�q�Xc�A�*

loss .�=��^2       �	�
�Xc�A�*

lossl�=s��\       �	 ��Xc�A�*

loss�>T7�R       �	lA�Xc�A�*

loss�8>�gr       �	{��Xc�A�*

lossפ>rr�%       �	���Xc�A�*

lossܥ<k&y�       �	�s	�Xc�A�*

lossJ=�~�       �	�
�Xc�A�*

loss��=��ۢ       �	�
�Xc�A�*

lossT�=��p�       �	�Q�Xc�A�*

loss
�T<9�^�       �	���Xc�A�*

loss��p=��       �	u��Xc�A�*

loss
(�=Ln.       �	L�Xc�A�*

loss��=�6��       �	��Xc�A�*

lossJ�=�[d       �	��Xc�A�*

loss$>��ߡ       �	 '�Xc�A�*

loss)/�<x�+�       �	��Xc�A�*

loss���<���:       �	�_�Xc�A�*

loss\��=���       �	���Xc�A�*

loss�\9=3��       �	���Xc�A�*

loss�j=N��x       �	�0�Xc�A�*

loss�ˑ=P�kQ       �	���Xc�A�*

loss��h=u&�       �	�]�Xc�A�*

lossS�=��q�       �	���Xc�A�*

loss�K�=x�b�       �	.��Xc�A�*

lossN��<v��       �	�$�Xc�A�*

loss݂�<Q��g       �	��Xc�A�*

lossx��=D;       �	�Q�Xc�A�*

lossp��=J��s       �	���Xc�A�*

lossCwI=�j��       �	k��Xc�A�*

loss�]x=<X.�       �	��Xc�A�*

loss_��=Y"�       �	���Xc�A�*

lossD��<�^��       �	�B�Xc�A�*

lossV0W<��/~       �	���Xc�A�*

loss��=Kj�       �	2r�Xc�A�*

loss���=��~�       �	F{�Xc�A�*

lossq��=��0�       �	�9�Xc�A�*

loss�%">M��       �	���Xc�A�*

loss��=�y2�       �	9��Xc�A�*

loss�
�<�X�       �	-��Xc�A�*

loss���<@7��       �	���Xc�A�*

loss���<�7\       �	B	!�Xc�A�*

loss2��=D:�`       �	]�!�Xc�A�*

lossϒ>=r�0�       �	��"�Xc�A�*

loss�ZG=��4}       �	5#�Xc�A�*

loss�n�=�w�       �	��#�Xc�A�*

lossIy�==
�       �	�j$�Xc�A�*

loss��T=��R       �	�%�Xc�A�*

lossb
>�z��       �	,�%�Xc�A�*

loss�<�ǃ%       �	?&�Xc�A�*

loss���=&5�K       �	��&�Xc�A�*

loss)�<�o�       �	|'�Xc�A�*

loss�K�=�M�,       �	:#(�Xc�A�*

loss��c='�٣       �	\�(�Xc�A�*

loss�3L>�D�       �	Fy)�Xc�A�*

loss�`�<�\;�       �	�"*�Xc�A�*

lossm=��I       �	�*�Xc�A�*

loss�y=�$��       �	�R+�Xc�A�*

loss�#�=�kL�       �	��+�Xc�A�*

loss��=[
P�       �	F�,�Xc�A�*

losst��=�/Y       �	I+-�Xc�A�*

loss�>6'2       �	\�-�Xc�A�*

lossw�<���       �	�k.�Xc�A�*

lossX}�;�t       �	�/�Xc�A�*

lossw�=�>�@       �	��/�Xc�A�*

lossv>c�Ǵ       �	Uh0�Xc�A�*

loss4��<3��N       �	B1�Xc�A�*

lossI۟<����       �	��1�Xc�A�*

losso4>~���       �	�_2�Xc�A�*

lossVc�<��h�       �	�
3�Xc�A�*

loss�Z�=�7�z       �	��3�Xc�A�*

loss�
�=��K       �	"R4�Xc�A�*

loss���=+d�       �	��4�Xc�A�*

loss䓮;M��       �	�5�Xc�A�*

loss�7a=l��       �	?96�Xc�A�*

loss��<3ԿE       �	��6�Xc�A�*

loss h<�[       �	�|7�Xc�A�*

loss1!]=���       �	�8�Xc�A�*

loss�t�<�$N/       �	d�8�Xc�A�*

loss&H�<d�L9       �	�V9�Xc�A�*

loss�a6=�?�       �	��9�Xc�A�*

lossm%Q==��c       �	n�:�Xc�A�*

lossL��<���       �	*;�Xc�A�*

loss��A>�y�	       �	ѱ;�Xc�A�*

lossLn>�m�M       �	�<�Xc�A�*

loss1��=��]       �	o-=�Xc�A�*

loss�Ϣ=V��E       �	��=�Xc�A�*

loss@M==RH�       �	bi>�Xc�A�*

lossAZ�<�Z�T       �	l	?�Xc�A�*

loss���=��       �	ͮ?�Xc�A�*

lossT��=�!~       �	
L@�Xc�A�*

loss`	>�%ǳ       �	?�@�Xc�A�*

loss��<�6�y       �	�A�Xc�A�*

loss��>��:       �	�ZB�Xc�A�*

loss\��=����       �	n�B�Xc�A�*

loss��j=���?       �	9�C�Xc�A�*

loss`��<ݶY�       �	j4D�Xc�A�*

lossT�=��_u       �	}�D�Xc�A�*

loss3��=�0�V       �	�fE�Xc�A�*

lossJ��<<�Aj       �	��E�Xc�A�*

lossR�`=���       �	��F�Xc�A�*

loss���=�Zr�       �	-G�Xc�A�*

loss��<���       �	��G�Xc�A�*

loss�I=��@�       �	�J�Xc�A�*

loss<��=a�       �	��J�Xc�A�*

loss&�F=�z�!       �	�UK�Xc�A�*

losss�=w��j       �	o�K�Xc�A�*

lossw�p>���w       �	*�L�Xc�A�*

loss"�>0A�d       �	JM�Xc�A�*

loss��<!+x�       �	 �M�Xc�A�*

loss)�>y���       �	ƢN�Xc�A�*

losssd>[��       �	�MO�Xc�A�*

loss�zk=O�L       �	t�O�Xc�A�*

loss��C=����       �	u�P�Xc�A�*

loss��=�X�q       �	�;Q�Xc�A�*

loss�Ӎ=�       �	��Q�Xc�A�*

loss��=�       �	8�R�Xc�A�*

loss��=�r:�       �	"S�Xc�A�*

lossN	�<�Fp�       �	|�S�Xc�A�*

loss�	=GXmi       �	�oT�Xc�A�*

loss� �=Qۖ�       �	7U�Xc�A�*

lossp֚=��G�       �	��U�Xc�A�*

loss1Se=�@�p       �	�V�Xc�A�*

loss鑲=ɳ	�       �	
.W�Xc�A�*

loss��9=�O       �	��W�Xc�A�*

loss%R*<��3U       �	�gX�Xc�A�*

losss`�<km�       �	�Y�Xc�A�*

loss˴<k��       �	b�Y�Xc�A�*

loss���=2��U       �	HZ�Xc�A�*

loss�=E�z       �	��Z�Xc�A�*

loss]V�>��']       �	�[�Xc�A�*

loss%�M>�?W
       �	Y2\�Xc�A�*

loss$E%=���Z       �	��\�Xc�A�*

loss���= �<J       �	�i]�Xc�A�*

lossO�e=��B{       �	�^�Xc�A�*

loss'"=!�       �	<�^�Xc�A�*

loss���<AO�       �	�_�Xc�A�*

lossNf<ĆY�       �	9�`�Xc�A�*

loss�u�=�T&       �	>?a�Xc�A�*

lossn��=���       �	�Jb�Xc�A�*

loss,n�=���       �	��b�Xc�A�*

loss&~�=T,��       �	��c�Xc�A�*

lossI�=��2       �	�/d�Xc�A�*

loss)�=U�Y�       �	��d�Xc�A�*

loss���=68�       �	�e�Xc�A�*

lossd�=ϔ �       �	�f�Xc�A�*

loss���=ѫ�_       �	��f�Xc�A�*

loss���=�#OX       �	�yg�Xc�A�*

loss.� =p���       �	f3h�Xc�A�*

loss,�=�r�       �	��h�Xc�A�*

loss}�B=DR       �	s�i�Xc�A�*

lossliS=����       �	S#j�Xc�A�*

lossM��=��U       �	}�j�Xc�A�*

lossFG�<= {C       �	&ok�Xc�A�*

lossX=U3f       �	�l�Xc�A�*

loss�:�=}ݲc       �	��l�Xc�A�*

loss�!w=
୬       �	�[m�Xc�A�*

losss�U=6	��       �	H�m�Xc�A�*

loss��[=�Ŀ�       �	�n�Xc�A�*

lossA�;=��o�       �	�yo�Xc�A�*

loss-w@=�ֺ       �	Hp�Xc�A�*

loss/ߧ=�M�+       �	_�p�Xc�A�*

loss�q)>:$�       �	tbq�Xc�A�*

lossp�=��e       �	��r�Xc�A�*

loss2�=�eI
       �	}ws�Xc�A�*

loss��<>�$+7       �	��t�Xc�A�*

loss��`=$V)�       �	vu�Xc�A�*

loss�|�=���/       �	�$v�Xc�A�*

loss�F>_�E�       �	��v�Xc�A�*

loss��=yV��       �	zw�Xc�A�*

loss!�<E�o�       �	S[x�Xc�A�*

loss_��=W$D�       �	��x�Xc�A�*

loss-hz=��       �	��y�Xc�A�*

loss��=��r       �	A,{�Xc�A�*

loss�=�x�       �	u�{�Xc�A�*

loss_��=k3d       �	5|�Xc�A�*

loss�pm=��>       �	� }�Xc�A�*

loss���=�Q�       �	J�}�Xc�A�*

lossF�;=��       �	;m~�Xc�A�*

lossrs=2M�       �	�"�Xc�A�*

loss�=x��M       �	f��Xc�A�*

loss+�>�E��       �	d��Xc�A�*

loss�<�=p        �	����Xc�A�*

loss�d]<o�1       �	[���Xc�A�*

loss�Q=��i       �	�4��Xc�A�*

loss��j=z [O       �	҂�Xc�A�*

loss z�<E       �	l��Xc�A�*

lossZ/`=�5ɉ       �	�
��Xc�A�*

loss4�
>AR�X       �	����Xc�A�*

loss�%=D���       �	HQ��Xc�A�*

loss�U�=V��x       �	�녈Xc�A�*

loss�C=�S\       �	M���Xc�A�*

loss��=jX�       �	���Xc�A�*

loss�>"=��I2       �	���Xc�A�*

loss=y0=��hd       �		T��Xc�A�*

lossNQ�=�u�Z       �	RXc�A�*

loss���=����       �	����Xc�A�*

lossar�=%       �	�5��Xc�A�*

loss��<���       �	ي�Xc�A�*

loss��+>ݥ�G       �	Bv��Xc�A�*

loss�ō=Ξ:1       �	�-��Xc�A�*

loss�F)>`���       �	#1��Xc�A�*

lossh�;(, }       �	�ˍ�Xc�A�*

loss�7	=��;(       �	�t��Xc�A�*

loss��=��+X       �	��Xc�A�*

loss�YM>$+�G       �	ܻ��Xc�A�*

loss�n�<\I��       �	�U��Xc�A�*

loss#I�=^�Lj       �	c}��Xc�A�*

lossv~�<��,       �	���Xc�A�*

lossd��=�ҁ�       �	����Xc�A�*

loss��=���       �	����Xc�A�*

loss�b�=��O�       �	3��Xc�A�*

lossN��=��       �	 ѕ�Xc�A�*

loss���=�E�       �	Yl��Xc�A�*

loss\Wh=���1       �	2��Xc�A�*

loss�1=�B��       �	���Xc�A�*

loss_�=-9�x       �	$_��Xc�A�*

loss7�f=o��       �	�+��Xc�A�*

lossfx6=��s�       �	Tř�Xc�A�*

loss2�B=1       �	�`��Xc�A�*

lossTM�=j!�       �	�J��Xc�A�*

loss�:n=\⥠       �	;ᛈXc�A�*

loss��<��7       �	����Xc�A�*

loss=;=O-0a       �	�&��Xc�A�*

lossR�r=(�C       �	�Ν�Xc�A�*

loss��=ˣ�       �	�c��Xc�A�*

loss�>��b�       �	i���Xc�A�*

loss>s=DqO&       �	����Xc�A�*

loss���=�E�       �	+��Xc�A�*

loss*f�=��X       �	� �Xc�A�*

loss�l�<���       �	x`��Xc�A�*

loss7\�=�;s�       �	����Xc�A�*

loss��/>a)�Y       �	c*��Xc�A�*

loss�}==��9�       �	�ڣ�Xc�A�*

lossCPM=JO-\       �	���Xc�A�*

loss���=�!�       �	1��Xc�A�*

loss���=�=�       �	Rҥ�Xc�A�*

loss`��<D���       �	����Xc�A�*

lossM�c=���       �	�9��Xc�A�*

loss�$�=�i�       �	9Ч�Xc�A�*

lossR�=��o0       �	8e��Xc�A�*

loss�x	=^�K)       �	H���Xc�A�*

loss��t=�f��       �	r���Xc�A�*

lossSV=���       �	><��Xc�A�*

lossf;=��;       �	֪�Xc�A�*

loss���=�       �	�{��Xc�A�*

loss��P=���%       �	���Xc�A�*

loss���<�w��       �	PĬ�Xc�A�*

loss��<"�_       �	�t��Xc�A�*

losst�x=�%�	       �	�"��Xc�A�*

lossyN�=�WYL       �	`ˮ�Xc�A�*

loss(�=M��       �	~t��Xc�A�*

loss9=��       �	���Xc�A�*

loss%�<.��_       �	����Xc�A�*

lossL��=#�<       �	I���Xc�A�*

lossPV�</Z��       �	VF��Xc�A�*

loss;�;>$2.�       �	�K��Xc�A�*

loss7�y=՞S�       �	~��Xc�A�*

loss�6�=m��       �	����Xc�A�*

lossf��=�Z�       �	�W��Xc�A�*

loss$��<y��       �	 ��Xc�A�*

lossO.N<��l�       �	JѶ�Xc�A�*

loss6(�=i�?U       �	}v��Xc�A�*

loss}B�<q�1�       �	�3��Xc�A�*

loss棌<o bf       �	ܸ�Xc�A�*

loss��!>{n       �	υ��Xc�A�*

loss�V>�b��       �	R&��Xc�A�*

loss�i�=~W�       �	bٺ�Xc�A�*

loss��Y=��       �	�z��Xc�A�*

loss��=o�	       �	"��Xc�A�*

loss�S>-Rw       �	R�Xc�A�*

loss��=��oa       �	&���Xc�A�*

lossO2�<��r�       �	@���Xc�A�*

lossJV�=�IA�       �	�]��Xc�A�*

lossm(=qvLG       �	����Xc�A�*

loss Z�=Э	x       �	���Xc�A�*

lossl;|=�x�       �	0��Xc�A�*

loss�ƙ=M�B�       �	G���Xc�A�*

loss�!={3tv       �	kcXc�A�*

loss5n�<�?��       �	FÈXc�A�*

loss�W�<&�o       �	�ÈXc�A�*

loss�5'=�Ō       �	fLĈXc�A�*

lossX�=����       �	��ĈXc�A�*

loss8�<ō       �	8�ňXc�A�*

lossqm=�v�e       �	�ƈXc�A�*

loss�3�=��"       �	1�ƈXc�A�*

loss=�K��       �	WǈXc�A�*

loss6)<%?��       �	 �ǈXc�A�*

loss�z�<�\�"       �	ÃȈXc�A�*

loss��h<x��       �	ɈXc�A�*

lossum�=���       �	��ɈXc�A�*

loss��=�I�       �	�zʈXc�A�*

loss�40=iO�5       �	! ˈXc�A�*

loss�W+=�&�U       �	��ˈXc�A�*

loss M�=�W       �	�b̈Xc�A�*

lossS&=�,G+       �	��̈Xc�A�*

loss��=���2       �	/�͈Xc�A�*

loss)Ҭ=84�       �	pBΈXc�A�*

loss�!*<��a       �	�ΈXc�A�*

loss��<9��t       �	�ψXc�A�*

loss,DO=�fMb       �	�ЈXc�A�*

lossE�=�5��       �	�ЈXc�A�*

loss�?=�S�       �	XXшXc�A�*

loss�� =j|]�       �	�шXc�A�*

loss�+%=���       �	v�҈Xc�A�*

loss��_=#���       �	X7ӈXc�A�*

loss�k�<���       �	^�ӈXc�A�*

lossA˛=�F��       �	�sԈXc�A�*

loss���<�k�<       �	4ՈXc�A�*

loss=Kg<�4       �	ӣՈXc�A�*

loss}��=�       �	:ֈXc�A�*

loss��<F	q�       �	W�ֈXc�A�*

loss�X=�3'�       �	Jy׈Xc�A�*

loss���=��       �	�*؈Xc�A�*

loss��=j[��       �	��؈Xc�A�*

loss��#=*��       �	�WوXc�A�*

loss�D<@�z�       �	��وXc�A�*

loss���=jA=       �	υڈXc�A�*

loss-�<R�v       �	�܈Xc�A�*

lossx��<K�Rp       �	݈Xc�A�*

loss&<�;��       �	��݈Xc�A�*

loss1q<���       �	�eވXc�A�*

loss�;�9u>       �	 ߈Xc�A�*

loss��?<��T�       �	��߈Xc�A�*

loss`>�:�4�+       �	�k��Xc�A�*

lossȊ�=�7^�       �	�	�Xc�A�*

loss�(`<v[v       �	��Xc�A�*

lossm�N;m�e       �	�O�Xc�A�*

loss���:���7       �	��Xc�A�*

loss�x<J�`�       �	��Xc�A�*

lossvx�=���       �	�W�Xc�A�*

loss�F�=,`T       �	I��Xc�A�*

loss�n:Tg�       �	`"�Xc�A�*

lossp�=z��       �	d��Xc�A�*

loss&Y�>z�1$       �	vl�Xc�A�*

loss�6y;��y       �	2�Xc�A�*

lossܰs=k�*       �	���Xc�A�*

loss�b�=Fc�_       �	�\�Xc�A�	*

losso�=o���       �	���Xc�A�	*

loss̲Q=
�       �	��Xc�A�	*

loss��=���       �	C�Xc�A�	*

loss�&>�3t_       �	���Xc�A�	*

loss$�=W���       �	��Xc�A�	*

loss��n=�0i       �	�E�Xc�A�	*

loss�o�=@�/Q       �	X��Xc�A�	*

lossCt�=B���       �	?��Xc�A�	*

loss�e�=ƫ�9       �	�*�Xc�A�	*

loss�l=��       �	L��Xc�A�	*

lossLYA=x}�       �	�[��Xc�A�	*

loss�>vO��       �	I���Xc�A�	*

losshU�=�߳�       �	+��Xc�A�	*

loss-�=�6"�       �	��Xc�A�	*

loss��=��D|       �	ع�Xc�A�	*

loss��>����       �	�P�Xc�A�	*

lossjz	=@v0       �	�Xc�A�	*

loss}��<|�Yo       �	 ��Xc�A�	*

loss_*q=�P��       �	T��Xc�A�	*

loss��V=C��       �		���Xc�A�	*

loss�t�<(��       �	ގ��Xc�A�	*

loss��D<���       �	?U��Xc�A�	*

loss{%=��       �	���Xc�A�	*

loss]&�=�Qk       �	����Xc�A�	*

loss&�=f��       �	GZ��Xc�A�	*

lossz�>-��Z       �	���Xc�A�	*

loss���=��r       �	����Xc�A�	*

loss��<=�cS~       �	ZG��Xc�A�	*

loss�ִ<�       �	#���Xc�A�	*

loss���=nc��       �	i���Xc�A�	*

loss�պ<��д       �	.��Xc�A�	*

lossMp�=���i       �	����Xc�A�	*

lossi�h<|�~       �	�t��Xc�A�	*

loss&3�<1��       �	���Xc�A�	*

loss�
_=(���       �	����Xc�A�	*

loss�3A=ݍ�       �	�y �Xc�A�	*

loss�3�=�x�       �	��Xc�A�	*

loss�t�;Ԏ�       �	���Xc�A�	*

lossC|={��       �	�J�Xc�A�	*

loss�T=1��       �	�.�Xc�A�	*

losse>�=�a       �	���Xc�A�	*

loss��g<y��K       �	9��Xc�A�	*

loss/Σ=L�j       �	(��Xc�A�	*

loss �Z=�-��       �	{2�Xc�A�	*

lossn`z<z��c       �	(��Xc�A�	*

loss嵬=����       �	�n�Xc�A�	*

losso<<D�r       �	?�Xc�A�	*

loss/X=��e       �	b��Xc�A�	*

loss7�O=�ow5       �	�"�Xc�A�	*

lossh�~=�r`       �	�!#�Xc�A�	*

loss�t�=��r       �	o�#�Xc�A�	*

loss���<$�;       �	�$�Xc�A�	*

loss�=o�!T       �	/�%�Xc�A�	*

loss��B=l��       �	l?'�Xc�A�	*

loss&�%=���`       �	�'�Xc�A�	*

loss�r�=��ɹ       �	��(�Xc�A�	*

loss2d	>�)tu       �	))�Xc�A�	*

loss�s>�w��       �	��)�Xc�A�	*

lossf�^=���       �	mY*�Xc�A�	*

loss�N=*gy       �	8�*�Xc�A�	*

loss��=���       �	��+�Xc�A�	*

loss
-E=�H'~       �	�I,�Xc�A�	*

lossm\+=�;       �	��,�Xc�A�	*

loss�Q=H��D       �	��-�Xc�A�	*

loss=<��b       �	�/.�Xc�A�	*

loss@$�<klҘ       �	��.�Xc�A�	*

loss�׎=V�rq       �	�a/�Xc�A�	*

loss,�7>��S       �	U0�Xc�A�	*

lossl[y=9L�       �	��0�Xc�A�	*

lossKx>���       �	�Z1�Xc�A�	*

loss���<F5�Z       �	��1�Xc�A�	*

loss��j=��ƴ       �	��2�Xc�A�	*

loss6.=��b       �	cC3�Xc�A�	*

loss���<%o       �	E�3�Xc�A�	*

lossv=�&�       �	��4�Xc�A�	*

loss�s�<�!       �	�m5�Xc�A�	*

loss=0�=����       �	�$6�Xc�A�	*

losshG�=�ǔ       �	H�6�Xc�A�	*

loss�;=�v�       �	�]7�Xc�A�	*

lossC{='��       �	��7�Xc�A�	*

loss7lC=9��       �	�8�Xc�A�	*

lossb�>��       �	^*9�Xc�A�	*

loss�=��6D       �	��9�Xc�A�	*

loss��&=pgc�       �	Ef:�Xc�A�	*

lossj�<bY�       �	�	;�Xc�A�	*

loss7�=]HqV       �	ø;�Xc�A�	*

loss�EL>�\o"       �	 �<�Xc�A�	*

loss��=��&�       �	�e=�Xc�A�	*

loss%�<?|4�       �	�>�Xc�A�	*

loss���=�L�d       �	}�>�Xc�A�	*

loss}�<3�%�       �	2X?�Xc�A�	*

lossC�<=�S�       �	R�?�Xc�A�	*

loss���=|��       �	Ý@�Xc�A�	*

loss*�>~ F       �	��A�Xc�A�	*

loss*�<s��       �	�sB�Xc�A�	*

loss�|�=�̔�       �		�C�Xc�A�	*

loss�&I=�l��       �	�#D�Xc�A�	*

loss��}<~�       �	�D�Xc�A�	*

lossxH<H+�/       �	(eE�Xc�A�	*

loss���=����       �	�F�Xc�A�	*

loss�T�<h�       �	öF�Xc�A�	*

loss)�A>o��       �	�[G�Xc�A�	*

loss�)�<FW�       �	OH�Xc�A�	*

loss��<]�t       �	z�H�Xc�A�	*

loss��<��S�       �	KI�Xc�A�	*

lossT<"-       �	�I�Xc�A�	*

loss��<�UV�       �	��J�Xc�A�	*

loss��=D67�       �	�HK�Xc�A�	*

loss��a=��       �	v�K�Xc�A�	*

lossIv=A�[q       �	�L�Xc�A�	*

lossGj�=Q�x       �	$M�Xc�A�	*

loss��=���        �	�M�Xc�A�	*

loss�C=��       �	!ZN�Xc�A�	*

loss��V=���       �	��N�Xc�A�	*

lossi�=3t�       �	��O�Xc�A�	*

lossr��=5:w       �	�)P�Xc�A�	*

lossO�=��a�       �	<�P�Xc�A�	*

loss� �='��       �	_Q�Xc�A�	*

loss��M=�^-       �	��Q�Xc�A�	*

loss���=W�V       �	)�R�Xc�A�	*

loss���<1���       �	�JS�Xc�A�	*

loss�=]��0       �	��S�Xc�A�	*

loss���=��       �	2vT�Xc�A�	*

loss�3�<���       �	�U�Xc�A�	*

loss,�=9g�       �	��U�Xc�A�	*

loss�c/=;_�       �	�OV�Xc�A�	*

loss?��=~!�K       �	��V�Xc�A�	*

lossW�f=� �       �	�W�Xc�A�
*

lossN05=Q�4       �	�=X�Xc�A�
*

loss g>�&j       �	aY�Xc�A�
*

loss��J=��W�       �	m�Y�Xc�A�
*

loss��^=~��       �	DoZ�Xc�A�
*

lossI�#=A��J       �	�[�Xc�A�
*

lossmp =݉       �	�Z\�Xc�A�
*

loss��q=dY�>       �	�]�Xc�A�
*

loss(}'=L�ҿ       �	?�]�Xc�A�
*

lossπj=�n�       �	��^�Xc�A�
*

loss�!=���       �	�Q_�Xc�A�
*

loss���<�>�d       �	1`�Xc�A�
*

lossHT�=1�-�       �	��`�Xc�A�
*

lossl�
=�s       �	��a�Xc�A�
*

loss���<�A��       �	z�b�Xc�A�
*

lossj��=�^��       �	�Mc�Xc�A�
*

loss���<.���       �	��c�Xc�A�
*

loss��=@(       �	+�d�Xc�A�
*

loss�w�<�G       �	��e�Xc�A�
*

loss� >nQ�       �	Pqf�Xc�A�
*

loss��J=�:F       �	?g�Xc�A�
*

lossډ�<Eg$(       �	&9h�Xc�A�
*

loss��q=c��       �	�Vi�Xc�A�
*

lossE��=2!       �	0�i�Xc�A�
*

loss�Zz=���       �	��j�Xc�A�
*

loss�=.��h       �	Y1k�Xc�A�
*

lossم>vэI       �	��k�Xc�A�
*

loss��<�:7>       �	�jl�Xc�A�
*

loss���<�T�       �	�m�Xc�A�
*

lossE��=�w��       �	��m�Xc�A�
*

lossI�(=�? ?       �	�n�Xc�A�
*

losswo�=�N��       �	+o�Xc�A�
*

loss0��<�:�~       �	*�o�Xc�A�
*

loss*�<�MD�       �	tap�Xc�A�
*

loss.�p<���       �	Iq�Xc�A�
*

loss�(�<�j       �	p�q�Xc�A�
*

loss���=:�k�       �	�Tr�Xc�A�
*

loss�U=l#��       �	�r�Xc�A�
*

lossr>��O       �	�s�Xc�A�
*

loss��=���       �	.;t�Xc�A�
*

lossH�=r�ƛ       �	��t�Xc�A�
*

loss�f"=|ĉ�       �	�}u�Xc�A�
*

loss�;�=��.B       �	 v�Xc�A�
*

loss-�p=�Y�/       �	w�Xc�A�
*

loss1�<���       �	��w�Xc�A�
*

loss:�<�u>�       �	vPx�Xc�A�
*

loss(	�=���       �	�7y�Xc�A�
*

loss�x=K�#l       �	��y�Xc�A�
*

lossS`�<����       �	�tz�Xc�A�
*

loss�#�=[��h       �	X{�Xc�A�
*

loss
��= :mF       �	��{�Xc�A�
*

loss�C%=�_�       �	�z|�Xc�A�
*

loss)c@<�.�       �	�}�Xc�A�
*

lossNe=Tر       �	~�Xc�A�
*

loss�%�<�!�       �	�~�Xc�A�
*

loss�@�=�aK�       �	�X�Xc�A�
*

loss�Ԋ<F��       �	w��Xc�A�
*

loss?Z�<�-�S       �	9ր�Xc�A�
*

loss���<�J       �	偉Xc�A�
*

loss�v�<|ܔ�       �	�炉Xc�A�
*

loss�[�<�#��       �	덃�Xc�A�
*

loss���<���%       �	f0��Xc�A�
*

loss�=S�X;       �	&���Xc�A�
*

loss�D�=�[a       �	ŭ��Xc�A�
*

lossrJ>g�?�       �	\��Xc�A�
*

loss�r�=��0�       �	���Xc�A�
*

loss�@�<_�w�       �	R���Xc�A�
*

loss�|A=�8�       �	Mf��Xc�A�
*

loss[�<	�       �	c	��Xc�A�
*

loss	7"=�[
�       �	���Xc�A�
*

loss��h=�;�_       �	�]��Xc�A�
*

lossb<���U       �	����Xc�A�
*

loss�|�=�ށ\       �	����Xc�A�
*

loss!��=����       �	AG��Xc�A�
*

loss��=h��       �	C按Xc�A�
*

lossZ��=кc       �	狍�Xc�A�
*

loss�$;����       �	�-��Xc�A�
*

loss�#d=����       �	 ��Xc�A�
*

loss%�j=:�S       �	zŏ�Xc�A�
*

loss�[�=�3��       �	ː�Xc�A�
*

lossiZu<Po        �	]k��Xc�A�
*

loss���=ũ��       �	�	��Xc�A�
*

lossRb�<�я�       �	����Xc�A�
*

lossȠf=�f��       �	h?��Xc�A�
*

loss�I@<%1�       �	Aؓ�Xc�A�
*

losss[z<��       �	us��Xc�A�
*

lossy�<��<�       �	g��Xc�A�
*

loss�A=�ߘ�       �	2���Xc�A�
*

loss�΢=˅�d       �	BB��Xc�A�
*

loss��=˭!+       �	h疉Xc�A�
*

loss=�Q=S�m,       �	J}��Xc�A�
*

loss�,�=�N��       �	���Xc�A�
*

loss���<;���       �	����Xc�A�
*

lossh�<9�g'       �	�_��Xc�A�
*

loss6(>@��l       �	� ��Xc�A�
*

loss�w�;��       �	=���Xc�A�
*

loss�%�=~�Z       �	6>��Xc�A�
*

loss�.=���#       �	7���Xc�A�
*

loss=_= v�Q       �	D���Xc�A�
*

lossm�=5/VV       �	���Xc�A�
*

loss�&�<ra�H       �	�B��Xc�A�
*

loss|�x<4��A       �	�	��Xc�A�
*

loss�=�G#!       �	�٠�Xc�A�
*

lossmƵ<��       �	��Xc�A�
*

loss���;t��       �	����Xc�A�
*

lossn�o<Y���       �	�^��Xc�A�
*

loss��0=E9L%       �	]���Xc�A�
*

lossƗ�=@��       �	�%��Xc�A�
*

loss1��=L��       �	k���Xc�A�
*

loss�_=��       �	�x��Xc�A�
*

loss,U=@�k       �	f��Xc�A�
*

loss��=�+��       �	⧉Xc�A�
*

loss�	�<ߩR�       �	8���Xc�A�
*

loss=��<\�G        �	�X��Xc�A�
*

loss��t<�Q�       �	���Xc�A�
*

loss��B<�?J�       �	G���Xc�A�
*

loss�]k<�aG&       �	[{��Xc�A�
*

loss�De=�uB�       �	�&��Xc�A�
*

loss�rv=7_j       �	�Ŭ�Xc�A�
*

loss��>;�       �	3m��Xc�A�
*

loss�6<�{�       �	?��Xc�A�
*

loss�#�="f��       �	�î�Xc�A�
*

loss*��;3�y       �	<k��Xc�A�
*

loss�M=�h	       �	>��Xc�A�
*

lossf�;=r�67       �	称�Xc�A�
*

lossd=P�N`       �	�O��Xc�A�
*

lossAAN=����       �	�걉Xc�A�
*

lossgu=d��W       �	ۇ��Xc�A�
*

lossC-�<�3^H       �	d#��Xc�A�*

loss/=��       �	Hų�Xc�A�*

loss�Ϻ=E8y_       �	/i��Xc�A�*

loss�E<i�5       �	���Xc�A�*

loss 6=�[�       �	����Xc�A�*

loss��<�<Q�       �	/N��Xc�A�*

lossM��<hǴ�       �	�ﶉXc�A�*

loss�I�=f�h       �	q���Xc�A�*

loss=t�=�[$       �	�.��Xc�A�*

lossX*e=�؎       �	;���Xc�A�*

loss�]�=_��       �	g���Xc�A�*

loss?N�=j���       �	�~��Xc�A�*

loss�:==,��v       �	>!��Xc�A�*

loss(v�<�͢�       �	'���Xc�A�*

lossT`=�Q�i       �	t^��Xc�A�*

loss]'3<D�ML       �	���Xc�A�*

loss��x=���       �	���Xc�A�*

loss�z�<q�+�       �	�U��Xc�A�*

loss�,B=q��       �	��Xc�A�*

loss�=�죓       �	����Xc�A�*

losst{�=��N�       �	�@��Xc�A�*

loss/M�<&� �       �	����Xc�A�*

loss�=����       �	8���Xc�A�*

lossF�<�3�       �	�.Xc�A�*

loss}�;<�       �	��Xc�A�*

loss)U=��Y�       �	[}ÉXc�A�*

loss�<��5�       �	�ĉXc�A�*

lossv�r=�g �       �	
�ĉXc�A�*

loss���<�AT�       �	]ŉXc�A�*

losshX<����       �	jƉXc�A�*

loss�=�]��       �	�ƉXc�A�*

lossȱ=g���       �	&TǉXc�A�*

loss�O<<���s       �	��ǉXc�A�*

lossƒ�<���X       �	�ȉXc�A�*

loss�:=P�       �	�4ɉXc�A�*

loss���=�2��       �	�ɉXc�A�*

loss�/h=��S�       �	�rʉXc�A�*

loss�L�=�%;       �	7ˉXc�A�*

loss[��<��$       �	�ˉXc�A�*

loss��<�^B`       �	9c̉Xc�A�*

loss�=�H9       �	��̉Xc�A�*

loss,w�<5�|a       �	.�͉Xc�A�*

loss�/=N���       �	.VΉXc�A�*

loss��<���       �	��ΉXc�A�*

lossO�n=Ἐh       �	��ωXc�A�*

loss���<�k�       �	|GЉXc�A�*

loss	�<���       �	��ЉXc�A�*

loss�ƨ<���       �	��щXc�A�*

lossq(<.��D       �	�0҉Xc�A�*

lossc�+=��}L       �	[�҉Xc�A�*

loss.9'=4���       �	�jӉXc�A�*

loss��h=���       �	B
ԉXc�A�*

loss�d�=�r̈́       �	w�ԉXc�A�*

loss�s=�a��       �	�ՉXc�A�*

loss)q�<��u       �	^.։Xc�A�*

loss���=��@       �	!�։Xc�A�*

loss�w=�
��       �	�a׉Xc�A�*

loss�E�=���       �	��׉Xc�A�*

loss�I�;;�}       �	�؉Xc�A�*

loss���=7��l       �	@0ىXc�A�*

loss���<�&k�       �	��ىXc�A�*

lossM��=h�"       �	5yډXc�A�*

loss��=�aWP       �	ۉXc�A�*

loss�9=�1}       �	z�ۉXc�A�*

lossL�<9)�       �	`܉Xc�A�*

loss~i=>;�W       �	�݉Xc�A�*

loss�
�<n��W       �	n�݉Xc�A�*

loss�!$>n;��       �	UOމXc�A�*

lossml�=���       �	�F߉Xc�A�*

loss���<g�x�       �	��߉Xc�A�*

loss6o&=Dbj�       �	����Xc�A�*

lossv� >v�Z�       �	�}�Xc�A�*

loss�N.=Z��j       �	�/�Xc�A�*

loss��=�?��       �	���Xc�A�*

loss��r<p��
       �	���Xc�A�*

loss�(=R"�       �	 D�Xc�A�*

loss.}�=�`       �	t��Xc�A�*

loss_�=�SP       �	��Xc�A�*

loss3�=0�?�       �	)?�Xc�A�*

loss�)�<��        �	���Xc�A�*

lossQy�<���       �	r��Xc�A�*

loss=��=�5�N       �	s*�Xc�A�*

lossX��;]n��       �	���Xc�A�*

loss�?�<��F�       �	x}�Xc�A�*

loss�Ԣ=W�8       �	�#�Xc�A�*

loss�b<i0[\       �	���Xc�A�*

loss�O<�;�x       �	k�Xc�A�*

loss��0=:&��       �	��Xc�A�*

loss�;?=+(�       �	9��Xc�A�*

loss�X�<O1r�       �	>[�Xc�A�*

loss�S><ܛN       �	��Xc�A�*

loss�E�=䬲�       �	\��Xc�A�*

loss#�=�d6       �	�W�Xc�A�*

loss���=�BT�       �	*��Xc�A�*

loss���<,[�       �	'���Xc�A�*

loss��<!��       �	�:�Xc�A�*

loss�f=U�)�       �	���Xc�A�*

loss��T=�oL�       �	y��Xc�A�*

loss���=���       �	�/�Xc�A�*

loss�#�;Z�       �	���Xc�A�*

loss�,�=�d�       �	���Xc�A�*

loss'�=��p       �	����Xc�A�*

loss\�~=�$ç       �	����Xc�A�*

loss��<���.       �	MN��Xc�A�*

lossi�<��\�       �	x���Xc�A�*

loss@�B=�1f       �	AG��Xc�A�*

loss(Iw<���       �	��Xc�A�*

loss4T�<d�[       �	K���Xc�A�*

loss�Ƕ=�8�       �	$c��Xc�A�*

loss{�2<�dB5       �	T���Xc�A�*

loss��<:�b8       �	����Xc�A�*

loss�~q=��"-       �	�S��Xc�A�*

loss�w�<��>       �	>"��Xc�A�*

loss���<t	�       �	����Xc�A�*

loss�L�=�.�&       �	L���Xc�A�*

loss�r=Η�5       �	. �Xc�A�*

lossw2�;%�       �	�� �Xc�A�*

loss���<�M�W       �	B��Xc�A�*

loss��x=X"5h       �	�w�Xc�A�*

loss߈f=���       �	��Xc�A�*

loss��$=�(_       �	{��Xc�A�*

loss:B<=��       �	�`�Xc�A�*

loss��a=<��       �	L��Xc�A�*

loss���<ړ��       �	&��Xc�A�*

lossQX=�g�:       �	ZE�Xc�A�*

loss	w<�4�9       �	;��Xc�A�*

loss��<aB�:       �	|�Xc�A�*

loss�%"=��86       �	?�Xc�A�*

loss�|5=K�
       �	���Xc�A�*

loss@`*=��Q       �	>^	�Xc�A�*

loss{u=5��       �	��	�Xc�A�*

loss��S=I�f@       �	��
�Xc�A�*

loss��<�CK�       �	;:�Xc�A�*

loss�[�;P��L       �	���Xc�A�*

losse{�<_:x       �	o~�Xc�A�*

loss3��=�]ti       �	�"�Xc�A�*

loss�+<��"       �	���Xc�A�*

loss���==$�       �	�`�Xc�A�*

lossr�=(�e       �	���Xc�A�*

lossD		=�W�4       �	�-�Xc�A�*

loss��J=�fZ�       �	6��Xc�A�*

lossw�=o(B�       �	��Xc�A�*

loss2�<��{<       �	�Xc�A�*

loss��+=,�       �	W��Xc�A�*

loss!�X=���a       �	�L�Xc�A�*

loss�w�=���       �	���Xc�A�*

loss��A=���       �	z��Xc�A�*

loss%�->�B�"       �	W>�Xc�A�*

lossx��;�\p�       �	)��Xc�A�*

lossP=�!+       �	Ӣ�Xc�A�*

lossAz�<� �{       �	�K�Xc�A�*

loss�$=%P[�       �	-�Xc�A�*

lossS=L�w       �	���Xc�A�*

lossZ�<�g�       �	�s�Xc�A�*

loss�h�<�O�{       �	�Xc�A�*

loss��=��b       �	���Xc�A�*

loss
�2>�1JJ       �	�s�Xc�A�*

lossgl#=��4       �	��Xc�A�*

lossC�=�uh�       �	���Xc�A�*

loss�P>��A       �	���Xc�A�*

loss�+=���{       �	�x�Xc�A�*

loss�=i�       �	 �Xc�A�*

loss�S=�c��       �	� �Xc�A�*

loss1�5=��I       �	(("�Xc�A�*

loss:<z��X       �	8�"�Xc�A�*

loss��<�s�/       �	?p#�Xc�A�*

loss(�=I;I�       �	�A$�Xc�A�*

loss"�=:2�S       �	�#%�Xc�A�*

loss���=�]�       �	��%�Xc�A�*

loss�>�=(�.=       �	ʧ&�Xc�A�*

loss�s=��       �	�P'�Xc�A�*

loss�W,=�e��       �	�(�Xc�A�*

lossf=Y>�v�       �	m�(�Xc�A�*

loss�LK=�ʇ2       �	k)�Xc�A�*

loss��B=L�7       �	� *�Xc�A�*

loss?�=*d�C       �	̘*�Xc�A�*

loss��<X�!D       �	�2+�Xc�A�*

loss�0�<���       �	��+�Xc�A�*

loss<�y=GKx       �	h,�Xc�A�*

loss{<#=?�       �	v�,�Xc�A�*

lossl0�<䈴       �	�-�Xc�A�*

loss�w=��`�       �	2.�Xc�A�*

lossSG=�'e�       �	�.�Xc�A�*

loss� |=��       �	�k/�Xc�A�*

lossH7�<ˣ�       �	�0�Xc�A�*

losshR�<���       �	=�0�Xc�A�*

loss.�S=�F�       �	+31�Xc�A�*

loss��U=�vH       �	��1�Xc�A�*

loss)��=��       �	`2�Xc�A�*

loss !h<?S�       �	�2�Xc�A�*

loss��T=���@       �	B�3�Xc�A�*

loss�3�<o�!�       �	�T4�Xc�A�*

loss��=uB5       �	��4�Xc�A�*

lossĠI<�y�       �	*6�Xc�A�*

lossnUo=�T��       �	��6�Xc�A�*

loss��=��       �	s7�Xc�A�*

loss�B<��f/       �	�%8�Xc�A�*

lossr��=�~��       �	�8�Xc�A�*

lossFʴ<�!}       �	�r9�Xc�A�*

loss�]�<�'�       �	:�Xc�A�*

loss�=��D       �	Ʀ:�Xc�A�*

lossĊ�<���K       �	O>;�Xc�A�*

loss��}=$��       �	^<�Xc�A�*

loss�=�.]*       �	B�<�Xc�A�*

loss��c=u��*       �	�H=�Xc�A�*

loss4��=x�       �	L�=�Xc�A�*

loss��#>b�F�       �	��>�Xc�A�*

lossƧw=T�h       �	?�Xc�A�*

loss[��=�       �	g�?�Xc�A�*

lossf`<�o�>       �	uU@�Xc�A�*

loss��<̧��       �	��@�Xc�A�*

lossȧd=���>       �	0�A�Xc�A�*

lossJ�%=��\F       �	�*B�Xc�A�*

lossֈ�<�}�F       �	w�B�Xc�A�*

loss��p=�yj^       �	�SC�Xc�A�*

loss�G<�?�       �	L�C�Xc�A�*

loss�
>�@]�       �	k�D�Xc�A�*

loss�ĉ=-�M3       �	�0E�Xc�A�*

loss���=V+��       �	!�E�Xc�A�*

loss(B<>8-�       �	siF�Xc�A�*

loss�&L=�_K�       �	G�Xc�A�*

loss�/�<8���       �	!�G�Xc�A�*

loss~�<�	�8       �	YH�Xc�A�*

loss��<�Ƀ7       �	(�H�Xc�A�*

loss8�=w�K�       �	��I�Xc�A�*

loss�f=~ݛ       �	�QJ�Xc�A�*

loss�#>=�J��       �	@�J�Xc�A�*

lossֹ=�x�       �	�K�Xc�A�*

lossӬU=XU	       �	$+L�Xc�A�*

loss�=�J�       �	L�L�Xc�A�*

lossh  <��x�       �	�mM�Xc�A�*

loss}�f=j�Z       �	BN�Xc�A�*

loss��K=)���       �	�N�Xc�A�*

loss��=���       �	�?O�Xc�A�*

loss���=�7�       �	o�O�Xc�A�*

lossv��=^��5       �	;�P�Xc�A�*

loss� =��V        �	�0Q�Xc�A�*

lossw�=�#L�       �	[�Q�Xc�A�*

loss��<��R       �	DlR�Xc�A�*

loss/7>O��z       �	.S�Xc�A�*

loss��l=
�X       �	��S�Xc�A�*

lossጮ<����       �	�ST�Xc�A�*

loss�`�=%�G�       �	��T�Xc�A�*

loss��=8h�       �	͓U�Xc�A�*

loss��<[z��       �	�2V�Xc�A�*

loss���<rާ�       �	��V�Xc�A�*

loss1x�=��\�       �	1xW�Xc�A�*

loss�=�A	�       �	DX�Xc�A�*

loss-�=�S;�       �	K�X�Xc�A�*

lossA=f+
F       �	EY�Xc�A�*

loss�}=�M��       �	'�Y�Xc�A�*

loss~B<�%)       �	+�Z�Xc�A�*

loss0=�+�       �	OZ[�Xc�A�*

lossE�}<�I�       �	��[�Xc�A�*

loss��!<S�#-       �	"�\�Xc�A�*

loss
��;�ك5       �	�]�Xc�A�*

lossn��=dI�E       �	�/^�Xc�A�*

loss��;=�Y��       �	�A_�Xc�A�*

loss�2�<;<       �	p^`�Xc�A�*

loss�/=�3cf       �	l\a�Xc�A�*

loss��M<"]��       �	�b�Xc�A�*

loss�<F�       �	��b�Xc�A�*

lossQ�.<�ǜ�       �	q�c�Xc�A�*

loss?`�=)��       �	�Wd�Xc�A�*

loss��	= s$T       �	?6e�Xc�A�*

loss�9=*�K       �	�f�Xc�A�*

loss�Ԧ=���       �	O�f�Xc�A�*

loss��#<`���       �	xag�Xc�A�*

loss���<k�~       �	m�g�Xc�A�*

loss�Y�;���~       �	~�h�Xc�A�*

loss(�<o�       �	s�i�Xc�A�*

loss��<�鷺       �	c'j�Xc�A�*

loss���=��A.       �	}�j�Xc�A�*

loss(�=�sh�       �	�tk�Xc�A�*

loss�=����       �	=l�Xc�A�*

loss��<��@�       �	��l�Xc�A�*

loss �=4��       �	Sm�Xc�A�*

loss X�<�/�       �	#�m�Xc�A�*

loss8*$=5>       �	B�n�Xc�A�*

lossH�<�+yu       �	�Co�Xc�A�*

loss�P	=K��       �	��o�Xc�A�*

loss��<l�+       �	�p�Xc�A�*

loss��<"͋�       �	�*q�Xc�A�*

lossHq�<��,(       �	j�q�Xc�A�*

loss�=���       �	 �r�Xc�A�*

loss�@<��n       �	d<s�Xc�A�*

loss�Q�<fe�8       �	��s�Xc�A�*

loss�X=Dn       �	Ynt�Xc�A�*

loss��]<�k��       �	U4u�Xc�A�*

lossP�=���R       �	��u�Xc�A�*

loss�%$<�X|       �	�rv�Xc�A�*

lossp=B1^�       �	�w�Xc�A�*

lossSڲ=.�W�       �	T�w�Xc�A�*

loss��%=ƫ�       �	Eex�Xc�A�*

lossȀ=��g:       �		y�Xc�A�*

lossiyL=�#T       �	i�y�Xc�A�*

lossn$=��	       �	�Kz�Xc�A�*

loss/�<�}�k       �	��z�Xc�A�*

loss��7=�_S       �	��{�Xc�A�*

loss�Y�=��W       �	�|�Xc�A�*

lossq�=���       �	��|�Xc�A�*

loss.Ӹ<�.�       �	�}}�Xc�A�*

loss���;u�~n       �	�~�Xc�A�*

loss��<Ǒ�       �	?�~�Xc�A�*

losst�=��Ap       �	�M�Xc�A�*

loss�m�<�I�       �	W��Xc�A�*

loss)��=C��s       �	4���Xc�A�*

loss:h< ��       �	l$��Xc�A�*

lossv@;=]�A`       �	����Xc�A�*

loss8��<���j       �	�X��Xc�A�*

loss��<�.�&       �	��Xc�A�*

loss�=]<�~
       �	����Xc�A�*

loss�=�K-W       �	.9��Xc�A�*

loss�;�<�� �       �	�ބ�Xc�A�*

loss�i=���       �	�v��Xc�A�*

loss,�_:h�Q       �	���Xc�A�*

loss��9=I��       �	����Xc�A�*

loss�U(=x�Q&       �	�B��Xc�A�*

loss�<`�       �	wև�Xc�A�*

losseG6=
�-       �	&s��Xc�A�*

loss�$+<���n       �	���Xc�A�*

lossѶ>S.�4       �	����Xc�A�*

lossWV�<��a�       �	�R��Xc�A�*

loss
nK<�4       �	����Xc�A�*

lossB�=��E       �	���Xc�A�*

loss;�;de�       �	^���Xc�A�*

loss��=�b��       �	�B��Xc�A�*

loss&@�;zYg�       �	�捊Xc�A�*

lossMx�;��v#       �	 Ŏ�Xc�A�*

lossX�<��G       �	Ug��Xc�A�*

loss��<rT~p       �	���Xc�A�*

loss{�;���       �	P���Xc�A�*

loss�;���Q       �	S=��Xc�A�*

loss�I<^�Y\       �	�Ց�Xc�A�*

loss�:����       �	�o��Xc�A�*

loss9:�%N�       �	���Xc�A�*

loss!$T<�B@       �	����Xc�A�*

loss�"x=
3�       �	TU��Xc�A�*

loss�;0=��       �	�锊Xc�A�*

lossZQ_;!�a       �	�~��Xc�A�*

losss�;w|��       �	���Xc�A�*

loss�	�>�@PY       �	�Ȗ�Xc�A�*

loss��L<Bd.�       �	0g��Xc�A�*

losss�=i%B�       �	���Xc�A�*

loss�=0��       �	�$��Xc�A�*

lossD�=&�~,       �	�ę�Xc�A�*

lossoc#=�҉�       �	'k��Xc�A�*

loss&��=[}��       �	�	��Xc�A�*

lossT!�<��       �	歛�Xc�A�*

loss�a=Z�9       �	#I��Xc�A�*

lossdSM<l(�       �	眊Xc�A�*

loss |t=�4��       �	����Xc�A�*

loss��'=���       �	����Xc�A�*

lossv��=��!U       �	a��Xc�A�*

loss�$=�6��       �	���Xc�A�*

loss)�b=eܓ�       �	-���Xc�A�*

lossё=p�       �	6W��Xc�A�*

loss��==��F�       �	f��Xc�A�*

lossN��<d�8e       �	d���Xc�A�*

loss0�=icP       �	+M��Xc�A�*

loss�>�=Fth�       �	!礊Xc�A�*

loss$�= 2��       �	{���Xc�A�*

lossr)�<�8�       �	�_��Xc�A�*

loss#�=��I       �	����Xc�A�*

loss�θ<��"O       �	[���Xc�A�*

losst�<����       �	�1��Xc�A�*

lossϣz<l5       �	ȩ�Xc�A�*

lossM0R<�_�       �	�k��Xc�A�*

loss�^�<�+�       �	��Xc�A�*

lossl��<��1       �	���Xc�A�*

loss� S=X�)�       �	q<��Xc�A�*

losse�K=�^�       �	�Ѭ�Xc�A�*

loss�5=F��I       �	�f��Xc�A�*

lossq"a=���       �	����Xc�A�*

loss�?<Ȁ	z       �	!���Xc�A�*

loss
��:�E�       �	I+��Xc�A�*

loss�R�<M���       �		ů�Xc�A�*

loss)�;��Y       �	����Xc�A�*

lossʑZ<�v��       �	�-��Xc�A�*

loss8�e=7��       �	 ű�Xc�A�*

loss0=�H�O       �	�[��Xc�A�*

lossrS~=^<٪       �	����Xc�A�*

loss�2<���       �	j���Xc�A�*

loss�"�<C��       �	�<��Xc�A�*

loss��t<���
       �	_д�Xc�A�*

loss���=�s{       �	�t��Xc�A�*

loss���<����       �	���Xc�A�*

lossL'�<���       �	:���Xc�A�*

loss#i=^��       �	�K��Xc�A�*

loss���<��4�       �	�㷊Xc�A�*

loss,�=W*nl       �	F{��Xc�A�*

loss�W;�� R       �	���Xc�A�*

lossa��;x�t/       �	����Xc�A�*

loss��<#�)!       �	y�ЊXc�A�*

loss��=37�=       �	HъXc�A�*

lossV>��ƛ       �	I�ъXc�A�*

lossi�>=)$�d       �	.�ҊXc�A�*

loss֚�<�>D       �	B&ӊXc�A�*

loss�S�<߸'       �	��ӊXc�A�*

loss��l=��       �	UPԊXc�A�*

loss�}�=��p       �	�1ՊXc�A�*

loss�^^=�       �	z�ՊXc�A�*

loss
;=G�(       �	�c֊Xc�A�*

loss܌#=@Gҵ       �	T׊Xc�A�*

loss-�<v.       �	M�׊Xc�A�*

loss��~=99�       �		5؊Xc�A�*

loss�T�=[\�       �	��؊Xc�A�*

lossH�V=�)8       �	4gيXc�A�*

loss���<�e}       �	��يXc�A�*

loss}XN:[-�       �	��ڊXc�A�*

lossBa�<��]       �	�#ۊXc�A�*

loss��=�-*�       �	A�ۊXc�A�*

loss�>��0m       �	��܊Xc�A�*

loss\z=��6�       �	Wz݊Xc�A�*

loss��i=���       �	ފXc�A�*

loss��<O�f       �	^�ߊXc�A�*

loss'��=�E�       �	�Q��Xc�A�*

loss&<r�8K       �	����Xc�A�*

loss�=�A�M       �	���Xc�A�*

loss��B<��u�       �	�#�Xc�A�*

lossqcH<d�.�       �	̶�Xc�A�*

loss�W=F�       �	�H�Xc�A�*

loss��<0��       �	���Xc�A�*

loss\��<��       �	j��Xc�A�*

loss�S8<�Ȫ       �	%�Xc�A�*

loss�5<�Ղ�       �	;��Xc�A�*

loss:��=�L��       �	X�Xc�A�*

lossH�|<&�W8       �	���Xc�A�*

lossJ��=Mk�       �	e��Xc�A�*

lossᯅ<*;SA       �	t$�Xc�A�*

lossM��=��(G       �	���Xc�A�*

loss�>E� �       �	x`�Xc�A�*

loss�d�=&L�0       �	Q��Xc�A�*

loss�=�b�       �	���Xc�A�*

loss�7�<j�<W       �	�'�Xc�A�*

lossۙ <p@�       �	���Xc�A�*

loss�$]<�ė�       �	�M�Xc�A�*

loss�M�<g"z�       �	:��Xc�A�*

loss�==k!C@       �	��Xc�A�*

lossʚ�<C��       �	��Xc�A�*

loss�<�#�       �	]��Xc�A�*

loss�_�<78V�       �	�b�Xc�A�*

loss��;���       �	5	��Xc�A�*

loss�S�;rן`       �	k���Xc�A�*

loss΀�=+�Ϙ       �	�5�Xc�A�*

loss���<�G�r       �	{��Xc�A�*

lossTʅ>��%c       �	Su�Xc�A�*

loss�=uui       �	��Xc�A�*

loss��D;~ ى       �	���Xc�A�*

loss���;S�       �	��Xc�A�*

loss�~ ;�8       �	L���Xc�A�*

loss���<��#�       �	����Xc�A�*

loss�<=O�2�       �	�3��Xc�A�*

loss�"3=��6�       �	����Xc�A�*

lossO��<�Z�e       �	�_��Xc�A�*

lossW��<cq|�       �	x���Xc�A�*

loss&J=��4       �	[���Xc�A�*

lossF��<B�u�       �	�e��Xc�A�*

loss
��<�i       �	����Xc�A�*

lossC�=��       �	O���Xc�A�*

loss�=�E��       �	�>��Xc�A�*

lossz�=i�Fu       �	4���Xc�A�*

loss��=A��       �	eo��Xc�A�*

loss#r=��w       �	���Xc�A�*

loss��=2��       �	����Xc�A�*

loss@L�;CM�|       �	a7��Xc�A�*

loss��"=��3       �	����Xc�A�*

loss�"=ڡJS       �	�d �Xc�A�*

loss%�<R=!�       �	r� �Xc�A�*

loss62<�O�f       �	��Xc�A�*

loss�=�	�       �	�#�Xc�A�*

loss���<T�       �	���Xc�A�*

lossL$P=3y��       �	�M�Xc�A�*

loss��=�       �	3��Xc�A�*

loss���=����       �	�u�Xc�A�*

loss-��<C���       �	��Xc�A�*

lossa�;i齹       �	���Xc�A�*

loss��<p�+�       �	M�Xc�A�*

loss�;p���       �	��Xc�A�*

loss��<��u       �	��Xc�A�*

loss*��<�Kyh       �	�9�Xc�A�*

loss���<�<E       �	���Xc�A�*

loss�5=�۹�       �	�`	�Xc�A�*

loss��=��eD       �	o�	�Xc�A�*

loss���=��;       �	x�
�Xc�A�*

lossXr�<��z�       �	�2�Xc�A�*

lossw=�<k�86       �	��Xc�A�*

lossfrQ=���R       �	�^�Xc�A�*

lossA�<,U�K       �	���Xc�A�*

loss%=?���       �	~��Xc�A�*

loss�U�<b�'       �	5$�Xc�A�*

loss�l�=�       �	���Xc�A�*

lossַ�<E}�       �	S�Xc�A�*

lossD5'=���       �	���Xc�A�*

loss3�F;�!m�       �	x�Xc�A�*

loss�^�=�!�       �	L�Xc�A�*

loss�q=��F�       �	;��Xc�A�*

loss��,=9k�       �	�;�Xc�A�*

loss3'>"��7       �	���Xc�A�*

loss�,N;��M�       �	__�Xc�A�*

lossq��;�/�       �	���Xc�A�*

loss�8�<��p       �	\��Xc�A�*

loss�;�<��7�       �	i7�Xc�A�*

loss���= "�{       �	c��Xc�A�*

loss@,�<�SY�       �	9c�Xc�A�*

lossͯ&=Eȩ       �	���Xc�A�*

loss	u�<hCK       �	%��Xc�A�*

loss�݃<�&��       �	A�Xc�A�*

lossȿ;'�K       �	���Xc�A�*

loss��,=5/�L       �	pw�Xc�A�*

lossNȰ=i��       �	�	�Xc�A�*

loss���<Gg��       �	��Xc�A�*

loss�<@u,�       �	rP�Xc�A�*

loss�lh<�       �	��Xc�A�*

lossf�L=�G�       �	���Xc�A�*

loss$�<�䙴       �	�*�Xc�A�*

loss��=�s�       �	f��Xc�A�*

lossri@<mh*�       �	2W�Xc�A�*

loss}��=¡Xl       �	���Xc�A�*

loss�W�<���       �	���Xc�A�*

loss4i�;����       �	�t �Xc�A�*

loss��<�t       �	!�Xc�A�*

loss�{�=��'       �	 "�Xc�A�*

loss�Ƅ<{�{{       �	��"�Xc�A�*

loss�=�,v�       �	�#�Xc�A�*

loss�=�<|s�       �	�?$�Xc�A�*

lossƹ�<Ä-�       �	��$�Xc�A�*

loss��=���       �	��%�Xc�A�*

loss*6�<݁Ȕ       �	�)&�Xc�A�*

loss)f�<��Q�       �	��&�Xc�A�*

loss���<�B8�       �	d'�Xc�A�*

lossX�<\�S�       �	I(�Xc�A�*

loss&�<�=��       �	��(�Xc�A�*

loss��;��:�       �	��)�Xc�A�*

lossJ�<��ܗ       �	�-*�Xc�A�*

lossC�[=� ��       �	��*�Xc�A�*

loss)�=��>�       �	Y+�Xc�A�*

loss�>6���       �	��+�Xc�A�*

losssv=_f4�       �	Ǽ-�Xc�A�*

loss�@t;���       �	vS.�Xc�A�*

loss�|g;>��       �	@�.�Xc�A�*

lossBn<P���       �	؝/�Xc�A�*

loss��<W��v       �	�?0�Xc�A�*

loss�vx;-��       �	}�0�Xc�A�*

loss��
=i�s�       �	��1�Xc�A�*

loss�\=���x       �	H42�Xc�A�*

loss��<U^i       �	'�2�Xc�A�*

loss=�=Yk       �	�q3�Xc�A�*

loss���<�e��       �	t4�Xc�A�*

loss�tc<;f�a       �	ϡ4�Xc�A�*

lossi��<⿁       �	 85�Xc�A�*

loss���=b��       �	��5�Xc�A�*

lossF�<����       �	��6�Xc�A�*

lossR"N=^I�X       �	��7�Xc�A�*

loss�+�<�RD       �	�"8�Xc�A�*

loss-!�<~�u�       �	<�8�Xc�A�*

lossHx#;W*��       �	(`9�Xc�A�*

loss�f�=�t�x       �	��9�Xc�A�*

lossS��<�Y^6       �	�:�Xc�A�*

loss��;��[�       �	�';�Xc�A�*

lossl4�<���       �	8�;�Xc�A�*

loss?��=�_E       �	�^<�Xc�A�*

lossdG)=KPr       �	V�<�Xc�A�*

lossx S=����       �	�=�Xc�A�*

loss��;1�#j       �	�>�Xc�A�*

loss��=<݂��       �	a�>�Xc�A�*

loss���=�!��       �	6;?�Xc�A�*

loss�؍:����       �	G�?�Xc�A�*

loss�j <e��}       �	*�@�Xc�A�*

loss�x=��=�       �	hzA�Xc�A�*

loss��`=ee#�       �	�B�Xc�A�*

loss�)�<%�       �	��B�Xc�A�*

lossw��<��}A       �	VHC�Xc�A�*

loss,�:kɉ�       �	X�C�Xc�A�*

lossT="d��       �	w�D�Xc�A�*

lossr�=���       �	E�Xc�A�*

loss���<��1�       �	�E�Xc�A�*

loss�e�:����       �	�F�Xc�A�*

loss!�+=j�.�       �	/2G�Xc�A�*

loss��=�}       �	��G�Xc�A�*

loss;|L=h��U       �	�vH�Xc�A�*

loss#�=��O�       �	I�Xc�A�*

lossn+N=h�*       �	N�I�Xc�A�*

loss8�=���H       �	��J�Xc�A�*

lossz�~=�j��       �	K#K�Xc�A�*

loss�T�; ?t       �	��K�Xc�A�*

loss�B6;m$�       �	�\L�Xc�A�*

lossȄ�<����       �	��L�Xc�A�*

loss���<��T�       �	�M�Xc�A�*

loss�/=���U       �	)N�Xc�A�*

loss��[=�M�       �	��N�Xc�A�*

loss�=SF*       �	��O�Xc�A�*

loss��-<p��       �	+P�Xc�A�*

lossi�9=��W�       �	+�P�Xc�A�*

loss4=�<�9�$       �	�[Q�Xc�A�*

loss�ݼ<�q�       �	��Q�Xc�A�*

loss�q�;��1       �	�R�Xc�A�*

lossdf�<���       �	�+S�Xc�A�*

loss��4=S��r       �	��S�Xc�A�*

loss[b=�9	       �	0hT�Xc�A�*

loss=A'=��!�       �	D�T�Xc�A�*

loss�Q�<���       �	`�U�Xc�A�*

loss�� =h�       �	=+V�Xc�A�*

loss2�O<�-7�       �	��V�Xc�A�*

lossv�"=��       �	�ZW�Xc�A�*

lossI;%<�&k�       �	��W�Xc�A�*

lossXi<��HT       �	��X�Xc�A�*

lossQ�0=C�       �	;Y�Xc�A�*

lossDgN=��       �	Z�Xc�A�*

loss�e=^��       �	�Z�Xc�A�*

loss�&�<�;T5       �	�2[�Xc�A�*

loss�Rv=qv�8       �	��[�Xc�A�*

loss唶<��       �	�h\�Xc�A�*

lossR��<r��h       �	R]�Xc�A�*

loss�r=�q0       �	��]�Xc�A�*

loss�6<���F       �	�D^�Xc�A�*

loss��=�Z�~       �	��^�Xc�A�*

lossW);�w�       �	_�Xc�A�*

loss6�=��       �	"`�Xc�A�*

lossC�<���       �	c�`�Xc�A�*

loss�;C=+\T�       �	�Ia�Xc�A�*

lossQ��<ĭ       �	?�a�Xc�A�*

loss�Q%;A��       �	�b�Xc�A�*

lossx��<���       �	y#c�Xc�A�*

loss�=�CG       �	��c�Xc�A�*

loss��g=g�       �	�id�Xc�A�*

lossc�=d@�       �	;�d�Xc�A�*

loss��<�ѫ       �	*�e�Xc�A�*

loss�3�=���       �	�%f�Xc�A�*

lossch�< ��9       �	��f�Xc�A�*

loss�;=�
%�       �	^Mg�Xc�A�*

loss��=rH��       �	��g�Xc�A�*

lossi�5=�M&       �	��h�Xc�A�*

loss�(	=���       �	�*i�Xc�A�*

lossz�-==�H�       �	;�i�Xc�A�*

loss���=�i�u       �	Zej�Xc�A�*

loss��"=���       �	�k�Xc�A�*

loss��=��       �	�k�Xc�A�*

lossp�<�+�       �	�8l�Xc�A�*

lossT��<��Kz       �	W�l�Xc�A�*

loss#�<<���       �	�im�Xc�A�*

loss�j�=Tݲ�       �	(n�Xc�A�*

losst�&=Y8�       �	�n�Xc�A�*

loss��=H:F�       �	q=o�Xc�A�*

loss["(>�0��       �	��o�Xc�A�*

loss���<M{[�       �	r�p�Xc�A�*

lossa��;W�3�       �	�Jq�Xc�A�*

loss�;���s       �	�"r�Xc�A�*

loss��!<�V
       �	!�r�Xc�A�*

loss��=�2L�       �	4fs�Xc�A�*

loss1�R<�\��       �	8t�Xc�A�*

loss��<&���       �	r�t�Xc�A�*

loss�4�=sӗ�       �	ADu�Xc�A�*

lossE�1=Wϒ�       �	��u�Xc�A�*

loss� �<�(��       �	/w�Xc�A�*

loss�=&��       �	�w�Xc�A�*

loss�n�=�]2�       �	 px�Xc�A�*

loss@%�<  �       �	y�Xc�A�*

lossv��;��®       �	�y�Xc�A�*

lossdN�=�@Q       �	�Kz�Xc�A�*

lossL	<��p*       �	V�z�Xc�A�*

lossHq�=R~�5       �	��{�Xc�A�*

loss.�P=T��       �	m|�Xc�A�*

loss�=�
Y�       �	�y}�Xc�A�*

lossM)�<�E��       �	~�Xc�A�*

loss 21=_��       �	��~�Xc�A�*

loss�&<?       �	�D�Xc�A�*

loss^�>��:I       �	���Xc�A�*

lossof=3��}       �	�x��Xc�A�*

loss8�?=&�#y       �	���Xc�A�*

loss�S]<�n�S       �	y���Xc�A�*

loss
[�<���_       �	|G��Xc�A�*

loss��U=��8�       �	Z�Xc�A�*

loss��M;�=ٷ       �	Ú��Xc�A�*

loss{�R=8g       �	S@��Xc�A�*

loss�y�;�뀴       �	ل�Xc�A�*

lossJ'V=����       �	3k��Xc�A�*

loss{4?=S��       �	� ��Xc�A�*

loss]>=��x�       �	맆�Xc�A�*

loss��@="��o       �	�?��Xc�A�*

lossӊ�;k��       �	�؇�Xc�A�*

loss�=�u�M       �	rm��Xc�A�*

loss!�;���       �	���Xc�A�*

loss\��;n7�=       �	Cɉ�Xc�A�*

loss|��<M�s{       �	�q��Xc�A�*

loss��0;���       �	��Xc�A�*

loss�ō;���       �	����Xc�A�*

loss8_=^H3       �	탍�Xc�A�*

loss��<��       �	��Xc�A�*

lossJٻ<1EZ�       �	����Xc�A�*

loss��#>|H�       �	����Xc�A�*

loss!�=F�܈       �	𧐋Xc�A�*

loss��@=�i�C       �	}>��Xc�A�*

loss��=p�{       �	Nӑ�Xc�A�*

losso��<�T�       �	�i��Xc�A�*

loss�3�<�Վ�       �	& ��Xc�A�*

loss��~=Q��       �	���Xc�A�*

loss��=�E��       �	C;��Xc�A�*

loss'P>�       �	kԔ�Xc�A�*

loss�Oq;Z�
       �	�h��Xc�A�*

loss�=^��H       �	����Xc�A�*

loss
= 9��       �	ⓖ�Xc�A�*

loss��s<�7�@       �	�]��Xc�A�*

loss�)�<���S       �	���Xc�A�*

loss�r=����       �	����Xc�A�*

losslZ=�hĘ       �	�E��Xc�A�*

lossrH=2�       �	���Xc�A�*

loss$��<D��       �	"���Xc�A�*

loss\5=�֌       �	�Q��Xc�A�*

loss��<b�8       �	%뛋Xc�A�*

loss.�<[��%       �	����Xc�A�*

loss�={���       �	e4��Xc�A�*

lossR�<q��q       �	ڝ�Xc�A�*

loss�W�<6LI�       �	Kt��Xc�A�*

loss��$>�T�"       �	���Xc�A�*

loss���<����       �	V�Xc�A�*

loss[�<�)       �	�Ġ�Xc�A�*

loss��B=�P}'       �	@l��Xc�A�*

loss���=�	�a       �	L8��Xc�A�*

loss%7H<���f       �	�<��Xc�A�*

loss��=D��       �	
���Xc�A�*

loss=#�=D� W       �	[���Xc�A�*

loss�2p=�ӽH       �	G<��Xc�A�*

loss�қ<l�>�       �	�q��Xc�A�*

loss���<��N�       �	�.��Xc�A�*

loss�a)<L�       �	�ħ�Xc�A�*

loss.�<�'�       �	rn��Xc�A�*

loss��=�X=�       �	�+��Xc�A�*

lossx"�<9���       �	m䩋Xc�A�*

loss%�@=/»�       �	����Xc�A�*

loss3c==(��       �	#���Xc�A�*

loss�
=��Z�       �	�:��Xc�A�*

loss:��;��Ʈ       �	9�Xc�A�*

loss��<���_       �	U���Xc�A�*

loss��y=��G       �	JF��Xc�A�*

loss�p<B+_f       �	�箋Xc�A�*

lossƕr=��S       �	����Xc�A�*

lossq�=s       �	D0��Xc�A�*

loss>�=)�b�       �	����Xc�A�*

lossć�<-k��       �	����Xc�A�*

loss\�N=Z�o       �	�8��Xc�A�*

loss�@ =Z�       �	'ܲ�Xc�A�*

lossiE'=�m\       �	r���Xc�A�*

loss��;���       �	5D��Xc�A�*

loss.�<h9&�       �	Eش�Xc�A�*

loss�<G=.�z       �	�p��Xc�A�*

lossE�7=�ct�       �	5z��Xc�A�*

loss��4=3� �       �	�$��Xc�A�*

loss�?]<��z       �	����Xc�A�*

loss��<       �	PT��Xc�A�*

loss	�;����       �	�Xc�A�*

loss�Va<�9��       �	����Xc�A�*

lossV�<
bui       �	;6��Xc�A�*

loss�P,=}�]�       �	�׺�Xc�A�*

loss;��<
�z�       �	�o��Xc�A�*

losss+M<-�       �	S��Xc�A�*

loss�.�=���       �	'���Xc�A�*

loss���=�B�       �	靽�Xc�A�*

lossӬm=o�]       �	|C��Xc�A�*

lossE_�=O�9�       �	�߾�Xc�A�*

loss�W<]��n       �	}v��Xc�A�*

loss�Q)=џ~o       �	���Xc�A�*

loss���<y���       �	 ���Xc�A�*

loss!��<t�7       �	�A��Xc�A�*

loss�7�<�O�J       �	]���Xc�A�*

loss:=q��       �	wXc�A�*

loss�Wq<O�%�       �	fËXc�A�*

loss�<��U       �	c�ËXc�A�*

lossd�=�$!       �	]PċXc�A�*

loss��L=i�\q       �	)�ċXc�A�*

loss}��=�I%�       �	��ŋXc�A�*

loss�J=��       �	HƋXc�A�*

lossɯ^=9��       �	x�ƋXc�A�*

loss踊;3+�       �	mWǋXc�A�*

loss�W<' ;$       �	��ǋXc�A�*

loss���=�NI�       �	E�ȋXc�A�*

losszI!<<�       �	R(ɋXc�A�*

lossw��;���"       �	��ɋXc�A�*

loss��=���       �	M�ʋXc�A�*

loss���<s+�       �	:ˋXc�A�*

lossl�E=G(R�       �	|�ˋXc�A�*

loss�i<��\b       �	�k̋Xc�A�*

loss�^;=��h�       �	:͋Xc�A�*

loss$ɹ<1�       �	��͋Xc�A�*

loss2:�<,79s       �	�C΋Xc�A�*

loss ��<@�M       �	j�΋Xc�A�*

loss�C�<�!��       �	�vϋXc�A�*

loss$�x<DD%       �	�ЋXc�A�*

losso��=hqG�       �	�ЋXc�A�*

loss�}�<��k       �	{IыXc�A�*

loss6�j=1��*       �	��ыXc�A�*

lossm�D<�;8       �	�zҋXc�A�*

loss��g<�t�6       �	�gӋXc�A�*

loss���=W�;�       �	�ԋXc�A�*

loss�FJ<�       �	r�ԋXc�A�*

lossV�=�ebq       �	tCՋXc�A�*

loss�F	<�[��       �	��ՋXc�A�*

lossӇ=��3       �	\w֋Xc�A�*

loss�L�<@���       �	�׋Xc�A�*

loss���<!y��       �	n�׋Xc�A�*

loss��<�"�       �	�B؋Xc�A�*

loss��<ٷ�       �	s�؋Xc�A�*

loss��<M=V       �	�qًXc�A�*

loss<�=�ya       �	�ڋXc�A�*

loss8[�<���       �	�ڋXc�A�*

lossx�*=��       �	,EۋXc�A�*

lossf�7=�l��       �	<�ۋXc�A�*

loss���<eB�       �	�t܋Xc�A�*

loss)�<JeL       �	_݋Xc�A�*

loss\��<�no�       �	ʦ݋Xc�A�*

losssm.<j�k�       �	cAދXc�A�*

loss4> =��!T       �	�1ߋXc�A�*

loss��=��       �	��ߋXc�A�*

loss8�P<���       �	�i��Xc�A�*

loss���=�L�Q       �	k�Xc�A�*

loss��<����       �	���Xc�A�*

loss,k=���%       �	S?�Xc�A�*

loss���<�f�~       �	���Xc�A�*

loss��0=�       �	`w�Xc�A�*

loss֩�<�S�       �	R�Xc�A�*

loss]�=.i��       �	��Xc�A�*

loss�{�;�JF       �	�E�Xc�A�*

loss1�;���       �	���Xc�A�*

loss�?O=�=�-       �	?��Xc�A�*

loss#/<����       �	��Xc�A�*

loss�}�;�-�       �	@m�Xc�A�*

loss=�N<N6��       �	��Xc�A�*

lossq��<�כ�       �	n��Xc�A�*

loss��6=��<�       �	�\�Xc�A�*

loss.<�c��       �	�,�Xc�A�*

loss2[.=~t��       �	� �Xc�A�*

loss�u�<)�k       �	:��Xc�A�*

lossT��;t�RY       �	�N�Xc�A�*

loss.8u=Ttҫ       �	8��Xc�A�*

lossИ=��;�       �	���Xc�A�*

lossf��<���       �	�4�Xc�A�*

loss3f�=� �       �	���Xc�A�*

loss�d�<�r�       �	�}��Xc�A�*

loss�2=��a�       �	��Xc�A�*

loss�">�=�       �	,��Xc�A�*

loss�#~=Ui�       �	/Q�Xc�A�*

loss��=��       �	���Xc�A�*

loss*�L=���a       �	_��Xc�A�*

lossA�=��        �	P�Xc�A�*

loss9$<rŸ(       �	���Xc�A�*

loss�15<H���       �	 ��Xc�A�*

lossC�=or�       �	����Xc�A�*

lossi
=�*�2       �	-Z��Xc�A�*

lossd�=\�-�       �	����Xc�A�*

loss4h9=�Au       �	����Xc�A�*

loss�� =�+��       �	m���Xc�A�*

loss�0Q="[�       �	�,��Xc�A�*

loss㓨=��rC       �	C���Xc�A�*

loss��<�zp       �	`��Xc�A�*

loss���;^�g	       �	�S��Xc�A�*

loss�|L<�#       �	����Xc�A�*

lossG�>be��       �	���Xc�A�*

loss���<��P�       �	�I��Xc�A�*

loss�	�=�б       �	>���Xc�A�*

loss�h;�fi       �	b���Xc�A�*

loss�?S<�B1�       �	�$ �Xc�A�*

lossȣ�=z�ݾ       �	� �Xc�A�*

loss��&<}A�       �	�U�Xc�A�*

loss(ܯ=eH�       �	0��Xc�A�*

loss�Ǘ<PR��       �	���Xc�A�*

loss�Hp<�h"       �	�0�Xc�A�*

loss+�<�U¡       �	d��Xc�A�*

losst�<�lU       �	?n�Xc�A�*

loss�3�<���       �	�	�Xc�A�*

loss s�:ao�       �	���Xc�A�*

loss�H=�+�       �	�C�Xc�A�*

lossc�K<& �       �	C��Xc�A�*

loss_y!=ւ��       �	��Xc�A�*

loss\Z1>C�H[       �	.	�Xc�A�*

lossf�9=]^�       �	��	�Xc�A�*

loss1[=�[F�       �	��
�Xc�A�*

loss��=�9.�       �	�<�Xc�A�*

loss��)=/���       �	p��Xc�A�*

loss=`�<N��<       �	��Xc�A�*

loss��<�r��       �	('�Xc�A�*

loss+n�=Y��       �	]��Xc�A�*

loss<�=���       �	.p�Xc�A�*

lossX݌<Z,�p       �	��Xc�A�*

loss߆�<Wao�       �	#��Xc�A�*

loss_S�<�B��       �	?��Xc�A�*

lossq�<��(�       �	;5�Xc�A�*

lossT�6<�;��       �	e��Xc�A�*

loss�;�<����       �	���Xc�A�*

loss�ڍ<���       �	K:�Xc�A�*

loss�?=�XI       �	1��Xc�A�*

lossg�<KH��       �	ޒ�Xc�A�*

loss�Q�<rg�J       �	�:�Xc�A�*

loss�T�=2�ˌ       �	���Xc�A�*

loss,P�<?�u       �	^��Xc�A�*

loss���<��R�       �	R(�Xc�A�*

loss�t�<���       �	��Xc�A�*

loss�4=v�i       �	Wx�Xc�A�*

loss۬<��fv       �	D�Xc�A�*

loss
�;I��r       �	���Xc�A�*

lossZ�= 5�s       �	YO�Xc�A�*

lossHe�<6�gj       �	���Xc�A�*

loss��i=2�p�       �	��Xc�A�*

lossn��<8ͽg       �	A�Xc�A�*

loss��<�)�X       �	F�Xc�A�*

loss�6�<_�       �	��Xc�A�*

loss�a <� &�       �	K�Xc�A�*

lossh7i=hgh�       �	���Xc�A�*

loss-";=.e       �	l��Xc�A�*

lossQW=i''�       �	; �Xc�A�*

lossQ�=��\U       �	C!�Xc�A�*

loss��	<�j�,       �	�!�Xc�A�*

lossa'�<d�D       �	��"�Xc�A�*

loss;|�<Uh�       �	˞#�Xc�A�*

losss�</���       �	c_$�Xc�A�*

loss���<�
       �	s�%�Xc�A�*

loss�1;�dn       �	+1&�Xc�A�*

loss���<\�g       �	F�&�Xc�A�*

loss�͡<��~Y       �	�j'�Xc�A�*

loss7�<�f>w       �	0(�Xc�A�*

loss�#�<|�(       �	6�(�Xc�A�*

loss�=Wd_&       �	F[)�Xc�A�*

loss3s�=���       �	m*�Xc�A�*

lossò�<�a%�       �	a�*�Xc�A�*

loss�=<�j��       �	�J+�Xc�A�*

loss2��<�XmE       �	�,�Xc�A�*

loss��<>��       �	ȶ,�Xc�A�*

loss�"r:x��       �	�W-�Xc�A�*

loss9΄<�b�(       �	��-�Xc�A�*

lossx�:
�H       �	=�/�Xc�A�*

loss���:,mj       �	�50�Xc�A�*

lossr��;�(�7       �	��0�Xc�A�*

loss!�:��"�       �	��1�Xc�A�*

loss���<�U��       �	�'2�Xc�A�*

loss�G%=�t$�       �	��2�Xc�A�*

loss��r;���(       �	1_3�Xc�A�*

loss��8NEW�       �	�3�Xc�A�*

loss/Ԫ:��C�       �	��4�Xc�A�*

loss��>�9��       �	�O5�Xc�A�*

loss�	�<���X       �	��5�Xc�A�*

loss�*:0�5       �	w�6�Xc�A�*

loss�V=�,��       �	Z)7�Xc�A�*

loss��>5��'       �	6�7�Xc�A�*

lossL��;�Tą       �	(b8�Xc�A�*

loss�Q=C,��       �	��8�Xc�A�*

loss�B<�=_�       �	��9�Xc�A�*

loss��!=E�Y&       �	<0:�Xc�A�*

losso�#=�˘       �	��:�Xc�A�*

lossH';��p#       �	g;�Xc�A�*

loss��v<c/
�       �	.<�Xc�A�*

loss���=��t       �	�<�Xc�A�*

loss��0=6ͅ�       �	7O=�Xc�A�*

loss��C=.e��       �	�/>�Xc�A�*

loss�4=s,�       �	U�>�Xc�A�*

losszz=���       �	t\?�Xc�A�*

loss��P=���~       �	#�?�Xc�A�*

loss��4=fs�R       �	�@�Xc�A�*

lossH�i=��S       �	�-A�Xc�A�*

lossS٘=P���       �	;�A�Xc�A�*

loss��<���       �	�^B�Xc�A�*

loss�-<1!Ip       �	k(C�Xc�A�*

loss�"==�       �	��C�Xc�A�*

lossDʓ<i%ږ       �	�fD�Xc�A�*

loss�<;�2�       �	_	E�Xc�A�*

loss�=�7�#       �	��E�Xc�A�*

loss��	<��6�       �	$EF�Xc�A�*

loss�yV;�H�u       �	��F�Xc�A�*

loss�L�<"W]:       �	�G�Xc�A�*

loss��;~���       �	mH�Xc�A�*

lossI�;���       �	��H�Xc�A�*

loss~j=���       �	�^I�Xc�A�*

lossH��;�j��       �	"�I�Xc�A�*

loss��D=�T��       �	��J�Xc�A�*

loss���<�8��       �	>K�Xc�A�*

loss%p<Έ�^       �	�K�Xc�A�*

loss���<@⸡       �	UmL�Xc�A�*

loss!��;��M�       �	 
M�Xc�A�*

lossNKP=�ud       �	%�M�Xc�A�*

loss�E*=,��P       �	�NN�Xc�A�*

lossE��;A���       �	��N�Xc�A�*

lossC�<���       �	��O�Xc�A�*

lossf��<Y7��       �	�%P�Xc�A�*

lossv=E<T�       �	x�P�Xc�A�*

loss���;����       �	?�Q�Xc�A�*

loss��<��x        �	�9R�Xc�A�*

loss��@=�R;       �	T�R�Xc�A�*

loss8��<��       �	j�S�Xc�A�*

loss���:��[*       �	T�Xc�A�*

loss^�<�3{F       �	�T�Xc�A�*

loss��=9P��       �	AgU�Xc�A�*

loss���;`���       �	��U�Xc�A�*

loss5g=���       �	�V�Xc�A�*

loss���:�.��       �	U2W�Xc�A�*

lossDp�<0T&       �	��W�Xc�A�*

loss5�;�#�       �	�Hr�Xc�A�*

loss�e)=2�2�       �	��r�Xc�A�*

loss�_�=�՚�       �	:vs�Xc�A�*

loss�v�=���V       �	5t�Xc�A�*

lossCZ =�o       �	�t�Xc�A�*

lossƎ�=i#g       �	.7u�Xc�A�*

loss�(�<կV       �	��u�Xc�A�*

loss.�=�L,       �	!tv�Xc�A�*

loss8�s=�4�       �	�
w�Xc�A�*

lossX�?=�eB       �	C�w�Xc�A�*

loss��=��5       �	xEx�Xc�A�*

loss���<�5g       �	��x�Xc�A�*

loss�0e=�u0u       �	�uy�Xc�A�*

loss��=�9D�       �	+1z�Xc�A�*

lossh��=�[�       �	��z�Xc�A�*

loss6S=��E       �	7n{�Xc�A�*

loss8Ɉ;rx�Q       �	�|�Xc�A�*

loss��;�=�       �	�|�Xc�A�*

lossnT<mH�       �	7R}�Xc�A�*

loss{E�=ҹ@       �	2�}�Xc�A�*

loss�C�<@ p-       �	��~�Xc�A�*

lossa�~=�TT�       �	w1�Xc�A�*

loss�Y�<2*�|       �	G��Xc�A�*

loss��(=W�n       �	d��Xc�A�*

loss�n=)��       �	����Xc�A�*

loss\&=@R�       �	}���Xc�A�*

loss
��<S���       �	�/��Xc�A�*

loss�K�<jG��       �	�ʂ�Xc�A�*

loss �%=f.��       �	�d��Xc�A�*

loss�a=�Mɷ       �	:��Xc�A�*

loss�0=�:li       �	J���Xc�A�*

lossz�<����       �	I0��Xc�A�*

lossr�;�CA0       �	T���Xc�A�*

lossT�=�c��       �	(���Xc�A�*

lossQ:=�@�       �	D1��Xc�A�*

loss�D�=����       �	Oˇ�Xc�A�*

loss��<��Gz       �	�n��Xc�A�*

lossض�<r]��       �	x��Xc�A�*

loss(H�=�äZ       �	H���Xc�A�*

loss���<���X       �	�;��Xc�A�*

loss/U=��       �	��Xc�A�*

loss*��<���       �	)���Xc�A�*

loss(�<3�^;       �	To��Xc�A�*

loss�ɝ<Ǉڪ       �	}��Xc�A�*

loss��=��Z�       �	Z���Xc�A�*

losshN�<o�!7       �	:��Xc�A�*

loss\��<� �8       �	ӎ�Xc�A�*

loss:�a=��k       �	bh��Xc�A�*

loss�H�=���)       �	m���Xc�A�*

lossR<����       �	J���Xc�A�*

lossE��;"��       �	1��Xc�A�*

loss�t\=U�ٱ       �	đ�Xc�A�*

lossS��<�f%       �	�Y��Xc�A�*

loss?�n>NT�       �	�Xc�A�*

lossI��=��@Y       �	���Xc�A�*

loss}��;��
`       �	>��Xc�A�*

loss���;���W       �	���Xc�A�*

loss�� ;=�"       �	๕�Xc�A�*

lossA��;k��       �	�X��Xc�A�*

loss��=��	�       �	����Xc�A�*

loss!�<<�(_       �	����Xc�A�*

loss�X<s��f       �	�:��Xc�A�*

loss}�D;���3       �	И�Xc�A�*

lossW�=G��d       �	�b��Xc�A�*

lossf�{<���       �	H���Xc�A�*

loss�,=�>�(       �	����Xc�A�*

loss\�6>�Y(       �	���Xc�A�*

loss<�<Y�(9       �	���Xc�A�*

lossn��=
�        �	�P��Xc�A�*

loss�/=`�       �	)蜌Xc�A�*

loss��M<?�<l       �	�~��Xc�A�*

loss��<��Ȯ       �	!��Xc�A�*

loss2��<I�r       �	PŞ�Xc�A�*

loss�O�<#�       �	�e��Xc�A�*

lossTgN=	#�       �	�V��Xc�A�*

loss��<H�@       �	��Xc�A�*

loss�0}<)��>       �	񄡌Xc�A�*

lossd��<��A       �	��Xc�A�*

loss�n=C��       �	i���Xc�A�*

loss
f=�;�       �	/���Xc�A�*

loss��<�g)�       �	�J��Xc�A�*

lossW�=���       �	x�Xc�A�*

loss�Y�<��5       �	���Xc�A�*

loss՟=��6       �	�)��Xc�A�*

loss�=�<�$@�       �	���Xc�A�*

loss�|<y��       �	w���Xc�A�*

loss]^C=���}       �	9B��Xc�A�*

lossVe�<�!�       �	⩌Xc�A�*

loss�O=#�1       �	���Xc�A�*

loss��z<��F�       �	�.��Xc�A�*

loss�f)=�IA       �		ƫ�Xc�A�*

loss�`<�qe       �	�o��Xc�A�*

loss�ZX<����       �	E���Xc�A�*

loss!�<�ah       �	��Xc�A�*

loss��<l�,u       �	��Xc�A�*

lossV�4=CpC]       �	���Xc�A�*

loss�6�=�5o�       �	�:��Xc�A�*

loss�<:�d�       �	�װ�Xc�A�*

loss��>+��       �	`���Xc�A�*

lossF!�;���)       �	�)��Xc�A�*

loss�H;��
�       �	�ǲ�Xc�A�*

loss�B<R]�       �	�d��Xc�A�*

losso��<f���       �	�&��Xc�A�*

loss�,�=���       �	/���Xc�A�*

loss���<i!��       �	�Y��Xc�A�*

loss�Q�=��M       �	����Xc�A�*

loss��:�ȃ�       �	*���Xc�A�*

loss�t<<;F�j       �	�,��Xc�A�*

loss�>T�H       �	䷌Xc�A�*

lossm�<�|�0       �	�긌Xc�A�*

loss�HH=�)?a       �	\���Xc�A�*

loss�=&��       �	�1��Xc�A�*

loss�X<�qL       �	gҺ�Xc�A�*

loss���;�g�       �	5��Xc�A�*

lossd`=���       �	 ��Xc�A�*

loss�iI;ߧg�       �	׼�Xc�A�*

loss���=8��6       �	Xu��Xc�A�*

loss���=Ȏ�+       �	cb��Xc�A�*

loss�I=r78&       �	,f��Xc�A�*

loss���<uGH>       �	���Xc�A�*

lossă
=���'       �	�)��Xc�A�*

loss�e�<��R�       �	����Xc�A�*

loss�~�<��&       �	��Xc�A�*

loss���<���p       �	�:ÌXc�A�*

loss:��;ŧ�       �	Z�ÌXc�A�*

loss�z�<�0;?       �	�zČXc�A�*

loss)��<���       �	�!ŌXc�A�*

lossH<q��'       �	�ŌXc�A�*

loss}�-=Sh�       �	��ƌXc�A�*

loss� <��@"       �	�8ǌXc�A�*

loss�\�<�y��       �	�ǌXc�A�*

loss���<���P       �	E�ȌXc�A�*

loss/ͻ<�94       �	�YɌXc�A�*

loss�� <��5d       �	#�ɌXc�A�*

loss:�;=�1z�       �	��ʌXc�A�*

loss��<�a�       �	�;ˌXc�A�*

loss@�E=�<*       �	�V̌Xc�A�*

loss=S<EW	       �	��̌Xc�A�*

loss�	Z=�*�{       �	!�͌Xc�A�*

loss, �<�       �	RΌXc�A�*

loss䝊;Z+v�       �	��ΌXc�A�*

lossx��<�xV�       �	}�όXc�A�*

loss��<�v�       �	amЌXc�A�*

loss�i=?�       �	�ьXc�A�*

loss���<�ڧ       �	M�ьXc�A�*

lossR�b<~�J�       �	=ҌXc�A�*

loss]�-<I���       �	��ҌXc�A�*

lossS��;��L9       �	��ӌXc�A�*

loss_�<�~�       �	�DԌXc�A�*

loss���<͋�`       �	'�ԌXc�A�*

lossdol<�qy�       �	B�ՌXc�A�*

lossiC{=��M       �	!Z֌Xc�A�*

loss�=��c4       �	��֌Xc�A�*

loss��I=R�D       �	]�׌Xc�A�*

loss�P�<FD�x       �	�K،Xc�A�*

loss_a:�K!{       �	�،Xc�A�*

loss
�+<w�A       �	:�ٌXc�A�*

loss��=�w       �	�7ڌXc�A�*

loss�$W=�J�q       �	��ڌXc�A�*

loss�n=�f�       �	�|یXc�A�*

lossD�=�ۥ�       �	�"܌Xc�A�*

loss�n�<��d	       �	8�܌Xc�A�*

lossr�c<��J7       �	Me݌Xc�A�*

loss|�:n5�       �	�ތXc�A�*

loss=�6<7�z       �	j�ތXc�A�*

loss���;����       �	P8ߌXc�A�*

loss`��;��p       �	��ߌXc�A�*

loss�|<�Ƅo       �	�|��Xc�A�*

loss&�<P�Ƕ       �	�3�Xc�A�*

loss�::=:6�       �	���Xc�A�*

loss:X�<����       �	���Xc�A�*

loss�<է��       �	Z�Xc�A�*

loss��Y;o��       �	!�Xc�A�*

loss��<JL�n       �	���Xc�A�*

lossI_�;�0�X       �	�g�Xc�A�*

lossA&=~+%       �	W�Xc�A�*

loss���;|�e-       �	���Xc�A�*

loss��4=(ï�       �	g}�Xc�A�*

loss�=f�<�       �	5'�Xc�A�*

loss�y�< E�       �	���Xc�A�*

lossI��:$G@�       �	fl�Xc�A�*

loss%��<�Y�       �	��Xc�A�*

loss<����       �	���Xc�A�*

loss��;u�}
       �	�G�Xc�A�*

loss��<���J       �	q��Xc�A�*

loss�`�;��{       �	8��Xc�A�*

loss�c�=��s       �	~�Xc�A�*

loss��=.�m       �	���Xc�A�*

loss֣8=��-       �	P�Xc�A�*

loss�3�<���       �	���Xc�A�*

loss ��=M}��       �	��Xc�A�*

lossi�z<��q       �	9��Xc�A�*

loss��c;���       �	����Xc�A�*

loss�.<Fݖ7       �	oe�Xc�A�*

loss�0<���       �	� �Xc�A�*

loss��<����       �	��Xc�A�*

loss&�A=o�_       �	f1�Xc�A�*

loss4\C=4�e*       �	���Xc�A�*

loss�f=|���       �	�m�Xc�A�*

lossq��;)�:       �	���Xc�A�*

loss��<�"W       �	 ���Xc�A�*

loss�w<��֧       �	�L��Xc�A�*

loss=�h<���       �	����Xc�A�*

lossM �<C�U       �	'���Xc�A�*

loss��<}+|       �	�)��Xc�A�*

loss쒬<p��       �	����Xc�A�*

loss���<�]       �	�~��Xc�A�*

loss.��=F���       �	�,��Xc�A�*

loss�P=��"�       �	����Xc�A�*

loss�zS<zX��       �	,~��Xc�A�*

lossc40=z��       �	%��Xc�A�*

loss��=5f       �	v���Xc�A�*

loss)��;�`=+       �	�i��Xc�A�*

loss6��<�k�l       �	�Y��Xc�A�*

loss��=K#D       �	���Xc�A�*

loss�(>ۡ6       �	ף��Xc�A�*

lossm{�=�LP       �	G; �Xc�A�*

lossυ�<8�V@       �	N� �Xc�A�*

loss��=��J.       �	P��Xc�A�*

loss�~�<�[+�       �	0�Xc�A�*

loss�X�<3��2       �	���Xc�A�*

loss}�=4X       �	�d�Xc�A�*

lossߒ�<Aae,       �	���Xc�A�*

loss,�<��\;       �	-��Xc�A�*

lossjL�;3       �	]2�Xc�A�*

loss$1�<^/a       �	���Xc�A�*

loss�=J<VaV�       �	u��Xc�A�*

loss���<��I       �	A+�Xc�A�*

lossNM�<�r�       �	���Xc�A�*

loss���<!~       �	4h�Xc�A�*

loss2��<x���       �	 	�Xc�A�*

loss�J<	�a4       �	|�	�Xc�A�*

loss�&r<�8>�       �	�3
�Xc�A�*

loss6)=�//8       �	�
�Xc�A�*

loss1\�=k��       �	0g�Xc�A�*

lossKL�<�F*�       �	���Xc�A�*

lossh*�< �<       �	���Xc�A�*

loss}ճ<��       �	�6�Xc�A�*

loss
��=��#�       �	��Xc�A�*

loss�%Y<9�l       �	�i�Xc�A�*

loss�<<$��9       �	��Xc�A�*

losse��<b�       �	{��Xc�A�*

loss�٘<kΥ       �	&U�Xc�A�*

loss��H=AL��       �	���Xc�A�*

lossN�==�ğ�       �	z��Xc�A�*

loss���=�;�       �	�.�Xc�A�*

loss��><��       �	�M�Xc�A�*

lossou�<�F7�       �	(��Xc�A�*

loss�9=��B       �	Ί�Xc�A�*

loss۷�<[�        �	w1�Xc�A�*

loss
��<e ��       �	���Xc�A�*

loss���<�b��       �	�v�Xc�A�*

loss�� =fLei       �	��Xc�A�*

loss�(F<oT�n       �	u��Xc�A�*

loss���:Y�	�       �	�E�Xc�A�*

lossZ�B<G���       �	��Xc�A�*

loss���<h���       �	�{�Xc�A�*

loss�<s^�       �	��Xc�A�*

loss�}<Tq       �	���Xc�A�*

loss��<�Lad       �	��Xc�A�*

loss{�<��       �	�,�Xc�A�*

lossZ�=V/�       �	���Xc�A�*

loss!�<o˴l       �	�}�Xc�A�*

loss�X<��$       �	l&�Xc�A�*

lossw�6=+�~{       �	���Xc�A�*

loss@W<U��       �	"��Xc�A�*

lossH$ =�ֻ,       �	�) �Xc�A�*

lossf0�<��~       �	�� �Xc�A�*

loss��=�ҍ       �	��!�Xc�A�*

loss�h�< ���       �	o�"�Xc�A�*

lossC�==���U       �	PS#�Xc�A�*

lossi� =���K       �	��#�Xc�A�*

loss݀&=x�"R       �	Ք$�Xc�A�*

loss�U<���       �	�3%�Xc�A�*

loss�\>댕�       �	_�%�Xc�A�*

loss�d>[��9       �	q&�Xc�A�*

loss|!<�       �	�'�Xc�A�*

loss��@;���       �	F�'�Xc�A�*

loss��=bN�F       �	+�(�Xc�A�*

loss\�=��|a       �	�$)�Xc�A�*

lossQl�:��it       �	�)�Xc�A�*

loss��;�0nm       �	SX*�Xc�A�*

loss�=��'�       �	P�*�Xc�A�*

lossI�(=�OP)       �	 �+�Xc�A�*

loss�՜<�pn       �	�0,�Xc�A�*

loss�r=����       �	��,�Xc�A�*

loss��<��؁       �	~-�Xc�A�*

loss��8<ߋ�T       �	�.�Xc�A�*

lossR�=lkJn       �	��.�Xc�A�*

loss:=hֲ�       �	�W/�Xc�A�*

loss�rV<�q�z       �	4�/�Xc�A�*

loss���=�ppl       �	��0�Xc�A�*

loss�A�<q�       �	C=1�Xc�A�*

lossj�&;��       �	��1�Xc�A�*

lossSF�<r˩       �	r2�Xc�A�*

loss���<�        �	-	3�Xc�A�*

loss�/;��<       �	,�3�Xc�A�*

lossQ°;h�@v       �	/54�Xc�A�*

loss�\=ˁ��       �	x�4�Xc�A�*

loss���=5�L�       �	nl5�Xc�A�*

loss��;=�7	g       �	�6�Xc�A�*

loss�q�;-XG       �	J�6�Xc�A�*

lossR,;<�M��       �	�:7�Xc�A�*

loss��=#�:�       �	��7�Xc�A�*

lossV�D=�<�&       �	�u8�Xc�A�*

loss]�@=��L�       �	�19�Xc�A�*

loss�t�<8�7       �	��9�Xc�A�*

losst��<��؝       �	j:�Xc�A�*

loss�%�<w�       �	�	;�Xc�A�*

loss	�=�K0       �	��;�Xc�A�*

loss��!<]��       �	�E<�Xc�A�*

loss�Fn<����       �	��<�Xc�A�*

lossH��=��>�       �	�}=�Xc�A�*

loss�6�;��       �	�>�Xc�A�*

loss�	�<U��       �	�>�Xc�A�*

loss)�h=)o��       �	"l?�Xc�A�*

lossTj=���D       �	1@�Xc�A�*

loss=�-<�~B       �	��@�Xc�A�*

loss���=s��       �	�uA�Xc�A�*

lossԋ�<�Ă       �	v4B�Xc�A�*

lossḃ<���.       �	��B�Xc�A�*

loss�6�=2Q       �	�C�Xc�A�*

lossK=�ZE�       �	OD�Xc�A�*

lossF�<�E       �	�D�Xc�A�*

loss�@�=ۛY       �	y�E�Xc�A�*

loss��=o��       �	-F�Xc�A�*

loss��<���       �	��F�Xc�A�*

loss1�|<�U��       �	�kG�Xc�A�*

loss ��<QE��       �	�#H�Xc�A�*

lossE+�=�T�F       �	4�H�Xc�A�*

lossw��<_9�       �	�WI�Xc�A�*

lossm�=���I       �	��I�Xc�A�*

loss3n%<m;N�       �	�J�Xc�A�*

loss�<�;����       �	:$K�Xc�A�*

loss��$<G)17       �	0�K�Xc�A�*

lossk�<7j       �	�LL�Xc�A�*

loss�D�</��       �	��L�Xc�A�*

loss#�=�`9       �	A�M�Xc�A�*

loss��=	�k�       �	&N�Xc�A�*

loss!/@;@�,�       �	�N�Xc�A�*

loss���;o}:A       �	�PO�Xc�A�*

loss�G�<�U�       �	KxQ�Xc�A�*

loss&�<*i�s       �	�jR�Xc�A�*

loss�]*=)��       �	,S�Xc�A�*

loss1��=��6i       �	~�S�Xc�A�*

loss)�='cB       �	�lT�Xc�A�*

lossM��;�m�       �	�U�Xc�A�*

loss[�=�l��       �	عU�Xc�A�*

lossZ';<'���       �	GXV�Xc�A�*

lossi˿<�uvE       �	��V�Xc�A�*

loss�,�;%�\P       �	��W�Xc�A�*

loss��<&8+�       �	#kX�Xc�A�*

losszD�<T)X�       �	AY�Xc�A�*

loss���<�~�       �	�Y�Xc�A�*

loss�b3=�K�k       �	fLZ�Xc�A�*

loss/i=�MR<       �	2�Z�Xc�A�*

loss��<�#3       �	'�[�Xc�A�*

lossGu<6���       �	_\\�Xc�A�*

loss��=���       �	I]�Xc�A�*

lossE��<[�}�       �	C�]�Xc�A�*

loss�o�=jr*�       �	JA^�Xc�A�*

loss��9<�Z[       �	I�^�Xc�A�*

loss��]<&��       �	x{_�Xc�A�*

loss��=��>�       �	2"`�Xc�A�*

lossg�<����       �	+4a�Xc�A�*

lossvC�<�=�       �	}b�Xc�A�*

lossvp=��b       �	P�b�Xc�A�*

loss�9�;�+S
       �	�c�Xc�A�*

loss.�<��       �	�id�Xc�A�*

losshZ=�=	*       �	*e�Xc�A�*

lossQ��<
��P       �	��e�Xc�A�*

loss.]=/�%�       �	)xf�Xc�A�*

loss��<pT�       �	=g�Xc�A�*

loss=L�<�h>�       �	H�g�Xc�A�*

loss�՗=Em��       �	\;h�Xc�A�*

loss�d=)�%W       �	��h�Xc�A�*

lossj�=��       �	�ii�Xc�A�*

lossN�o=a��%       �	]�i�Xc�A�*

loss@u=C�A       �	s�j�Xc�A�*

loss6��<"�*       �	Cpk�Xc�A�*

loss�Yj<��;�       �	ol�Xc�A�*

loss<����       �	%�l�Xc�A�*

loss�~�=q       �	#Im�Xc�A�*

loss��=�;�       �	�m�Xc�A�*

lossń<��8�       �	�~n�Xc�A�*

lossS�=o �       �	+o�Xc�A�*

loss/��<V�p       �	��o�Xc�A�*

loss߄m=.�\�       �	�@p�Xc�A�*

losswa�<h���       �	��p�Xc�A�*

loss���;�%��       �	��q�Xc�A�*

loss���;��^�       �	�r�Xc�A�*

lossz��<��       �	��r�Xc�A�*

loss�#A=��       �	.Vs�Xc�A�*

loss��<T��       �	v�s�Xc�A�*

loss��k=/���       �	ԛt�Xc�A�*

loss8t�=�]�       �	Ktu�Xc�A�*

lossݰ�:32�       �	�&v�Xc�A�*

loss�(�<
k�K       �	��v�Xc�A�*

loss��'<�__~       �	�|w�Xc�A�*

loss�+}<ᄶP       �	�&x�Xc�A�*

loss�	�;���       �	��x�Xc�A�*

loss�;�z�X       �	�iy�Xc�A�*

loss}xm=�/XX       �	�z�Xc�A�*

loss	��<��
�       �	�z�Xc�A�*

loss��+=�{Y       �	C{�Xc�A�*

lossAr=X��       �	n�{�Xc�A�*

loss��<!e�       �	Ed}�Xc�A�*

loss:��<L�F�       �	�}�Xc�A�*

loss�&�<��       �	 �~�Xc�A�*

loss
ʌ<J�Sn       �	.7�Xc�A�*

lossx�=*ۧ�       �	
��Xc�A�*

loss
A�=��       �	臀�Xc�A�*

loss��X;��       �	�.��Xc�A�*

loss!��<C�>       �	Xc�A�*

loss��2=v:p�       �	狂�Xc�A�*

loss��=�@y4       �	 )��Xc�A�*

loss���;z��-       �	�ƃ�Xc�A�*

lossB�;s>6�       �	�p��Xc�A�*

losst�<"���       �	5	��Xc�A�*

lossR�$=]bOr       �	r���Xc�A�*

lossY�<�L�       �	?U��Xc�A�*

lossO��< N�       �	��Xc�A�*

loss��#;��       �	����Xc�A�*

loss�ur=���2       �	@/��Xc�A�*

loss���;�Uv�       �	!Ɉ�Xc�A�*

loss1E<��       �	B`��Xc�A�*

lossm��<�~�       �	����Xc�A�*

loss'�!<�R�       �	l���Xc�A�*

losst��<*�&       �	�F��Xc�A�*

loss�=�;���K       �	�鋍Xc�A�*

loss ��<��       �	�~��Xc�A�*

loss��)<�q�l       �	V��Xc�A�*

loss+c<���       �	�Xc�A�*

loss��=2/�#       �	h���Xc�A�*

loss�K
='�}�       �	�:��Xc�A�*

loss2�<�V       �	�ۏ�Xc�A�*

loss��<7�       �	�w��Xc�A�*

loss{�T<�[�O       �	���Xc�A�*

loss3I(=J���       �	����Xc�A�*

loss�p<�f�       �	eQ��Xc�A�*

lossA�@<���       �	�咍Xc�A�*

loss��=X�%       �	���Xc�A�*

loss)�<�wϦ       �	~��Xc�A�*

loss�#+=/�E�       �	R�Xc�A�*

loss��=�	/       �	L���Xc�A�*

lossr::=c��       �	�(��Xc�A�*

loss��j=����       �	����Xc�A�*

loss!,�;	�و       �	�S��Xc�A�*

loss3�3=�C��       �	����Xc�A�*

lossc�=N��       �	.���Xc�A�*

loss}�	>ɫ�        �	V*��Xc�A�*

loss�z�<�ms       �	����Xc�A�*

lossm�q<�/�       �	Va��Xc�A�*

loss��^<�O8�       �	����Xc�A�*

loss�M�;s���       �	���Xc�A�*

loss��=M�]�       �	I+��Xc�A�*

loss`)= v�       �	��Xc�A�*

loss�1=��s�       �	�X��Xc�A�*

loss���<1��       �	�띍Xc�A�*

loss�xZ=��G�       �		���Xc�A�*

loss�`!<��o�       �	�!��Xc�A�*

lossJ�#<���       �	A���Xc�A�*

lossh:�<H>q�       �	vR��Xc�A�*

losss�=�$�~       �	����Xc�A�*

loss<۔=M�G�       �	+���Xc�A�*

loss���<���       �	襢�Xc�A�*

loss~�<F�65       �	�b��Xc�A�*

loss;�Y<�d�       �	����Xc�A�*

lossF�==��z        �	\ʤ�Xc�A�*

lossGB<|��       �	^h��Xc�A�*

loss*.�=+p+�       �	���Xc�A�*

loss�2y<WW�       �	#���Xc�A�*

loss��=�Y�<       �	�<��Xc�A�*

loss@�=����       �	k֧�Xc�A�*

loss��<y���       �	�n��Xc�A�*

loss_�9<���T       �	�#��Xc�A�*

loss	2y;衅�       �	�ǩ�Xc�A�*

loss���<��9�       �	�\��Xc�A�*

lossR��<gOQ       �	I���Xc�A�*

loss�$>i�t       �	����Xc�A�*

loss���=�.U�       �	0/��Xc�A�*

loss�<#*7�       �	[Ҭ�Xc�A�*

loss?#R<��"�       �	h魍Xc�A�*

lossV�@=�W       �	ƅ��Xc�A�*

losst�-=�mA[       �	E.��Xc�A�*

loss�=Ի�p       �	�ͯ�Xc�A�*

loss.ϙ<I��O       �	���Xc�A�*

loss�T=fv/       �	5���Xc�A�*

loss�<���X       �	BZ��Xc�A�*

loss=ܗ<�̔       �		���Xc�A�*

loss�o<��e       �	6峍Xc�A�*

loss��.<�Jf�       �	e���Xc�A�*

loss��	=����       �	�/��Xc�A�*

lossH�j<�PUk       �	1͵�Xc�A�*

loss.�S<��Y       �	o��Xc�A�*

loss���<ʅqS       �	g��Xc�A�*

loss��<��H�       �	ߧ��Xc�A�*

loss��r<y�Fq       �	WZ��Xc�A�*

lossqg=�+�       �	�*��Xc�A�*

loss�Y�<�Ė�       �	�ѹ�Xc�A�*

loss�;�<|���       �	u��Xc�A�*

loss��F;��       �	>=��Xc�A�*

lossJ�	<k��       �	3ọXc�A�*

loss�TV=Zﱤ       �	ڍ��Xc�A�*

loss���<}1�J       �	.7��Xc�A�*

lossf0=:Z�D       �	�ڽ�Xc�A�*

loss�`H=P��c       �	_}��Xc�A�*

loss؜�<�a$       �	�!��Xc�A�*

loss�,<�C+�       �	P㿍Xc�A�*

lossxK�<3qX       �	P���Xc�A�*

loss
��;��S       �	�%��Xc�A�*

loss76=���A       �	���Xc�A�*

loss�=�� ?       �	^hXc�A�*

loss���=�ܡb       �	[ÍXc�A�*

loss���;r�l       �	ӠÍXc�A�*

lossZ��<C�       �	)_čXc�A�*

loss�s�<�-��       �	4ōXc�A�*

lossd��<�5��       �	�ƍXc�A�*

loss$��<��Y       �	G�ƍXc�A�*

loss@�z=�       �	��ǍXc�A�*

loss��N<Ef�       �	�ȍXc�A�*

loss�L{<F�:C       �	@�ȍXc�A�*

loss RF;p61s       �	7qɍXc�A�*

loss��;����       �	ʍXc�A�*

loss!�V=�`5�       �	y�ʍXc�A�*

loss��<�G1       �	lwˍXc�A�*

lossk�<� �       �	O#̍Xc�A�*

lossƔ'=[�0l       �	��̍Xc�A�*

loss	��=g��       �	�l͍Xc�A�*

loss�"�<�ɔ�       �	�	΍Xc�A�*

loss�iv;O�A"       �	:�΍Xc�A�*

loss���<ꙇf       �	rPύXc�A�*

loss�ZA;�u�       �	��ύXc�A�*

loss,�>:���       �	��ЍXc�A�*

loss��!=w�       �	>%эXc�A�*

loss���:3�4       �	r�эXc�A�*

loss��;_1�E       �	�]ҍXc�A�*

loss��F=�Z5�       �	3�ҍXc�A�*

loss�6":Z�       �	ՑӍXc�A�*

loss�#�<:»       �	�9ԍXc�A�*

loss�l<c1�       �	X�ԍXc�A�*

lossN{:G��       �	�ՍXc�A�*

loss�&Q9�e�       �	-֍Xc�A�*

loss�51<cѨ        �	�֍Xc�A�*

loss��	=q���       �	qq׍Xc�A�*

loss�=�Q��       �	)!؍Xc�A�*

loss��;`��       �	 �؍Xc�A�*

loss-��;��w       �	�kٍXc�A�*

loss9�=Vy�^       �	kڍXc�A�*

loss�VG<�f       �	9�ڍXc�A�*

loss��<��W�       �	�cۍXc�A�*

loss�@�;dF�"       �	�܍Xc�A�*

lossO!=�s�       �	�܍Xc�A�*

loss�p�<�iN�       �	�pݍXc�A�*

lossTb+;I%p�       �	uލXc�A�*

loss�%�<�Sr�       �	�ލXc�A�*

loss�L�<�U�k       �	V}ߍXc�A�*

lossF�I;�s�       �	+��Xc�A�*

loss�͚<���       �	����Xc�A�*

loss��;mT��       �	�{�Xc�A�*

loss�0�<��v       �	���Xc�A�*

loss1�<���u       �	� �Xc�A�*

loss��<ܪ�s       �	��Xc�A�*

loss	w=4n��       �	h�Xc�A�*

loss��=����       �	�*�Xc�A�*

loss�#=��#C       �	w��Xc�A�*

loss���;1"�       �	�|�Xc�A�*

loss	�G=��,       �	��Xc�A�*

loss�s�<��R�       �	���Xc�A�*

loss6�*<��       �	���Xc�A�*

lossҐT=U4       �	�G�Xc�A�*

loss���<�%_�       �	���Xc�A�*

loss�M;�?ht       �	��Xc�A�*

loss��<���       �	4/�Xc�A�*

loss��<�Nm�       �	���Xc�A�*

loss��=h���       �	o�Xc�A�*

loss|��=�ֺ
       �	�'�Xc�A�*

loss�w=��jx       �	���Xc�A�*

lossl-A=�<r�       �	i�Xc�A�*

loss�:�<�I>       �	�
��Xc�A�*

lossx��<F�=       �	8���Xc�A�*

loss�A<p���       �	��Xc�A�*

loss��<J�G�       �	�D�Xc�A�*

loss�(<O�       �	���Xc�A�*

loss��;�Q��       �	:��Xc�A�*

lossQ��<�8��       �	'2�Xc�A�*

loss��0=�*:�       �	d��Xc�A�*

loss�G=�5       �	�x��Xc�A�*

lossf��<X�        �	w/��Xc�A�*

loss��;�2}�       �	w���Xc�A�*

lossR(=��c�       �	����Xc�A�*

loss�/<&Y7�       �	�E��Xc�A�*

loss��n=���       �	����Xc�A�*

loss�_='��       �	���Xc�A�*

loss��;�g��       �	�\��Xc�A�*

loss�E=��G       �	���Xc�A�*

loss!�$;�d�c       �	W���Xc�A�*

loss��<��5�       �	u��Xc�A�*

loss�M�:ˁ�V       �	�,��Xc�A�*

loss��;�c�       �	����Xc�A�*

loss8 <��z       �	��Xc�A�*

loss�h=.T��       �	b��Xc�A�*

loss�t=c�       �	�A�Xc�A�*

loss���<qc��       �	���Xc�A�*

loss�u</��|       �	Ov�Xc�A�*

loss��u<���        �	�
�Xc�A�*

lossi��<��͐       �	���Xc�A�*

loss�1=�M�y       �	wI�Xc�A�*

loss�K�=G��x       �	m��Xc�A�*

loss��{=6\�       �	�x�Xc�A�*

lossW�v<���       �	� �Xc�A�*

loss�A<����       �	� �Xc�A�*

lossW%6=~�       �	�C!�Xc�A�*

loss{4=54/       �	&�"�Xc�A�*

loss��V<<�       �	�Z#�Xc�A�*

lossi$�<{���       �	�$�Xc�A�*

lossf�;����       �	�$�Xc�A�*

lossB+=��[       �	��%�Xc�A�*

lossQ�=<��~       �	�&�Xc�A�*

loss&��=�0��       �	<N'�Xc�A�*

loss��+<v��m       �	�'�Xc�A�*

lossҦ�=       �	�(�Xc�A�*

loss��;J       �	_F)�Xc�A�*

loss,�	=J�Y@       �	!*�Xc�A�*

loss�`<��a�       �	�+�Xc�A�*

loss�Q^<~��       �	,�Xc�A�*

loss�-&=1KbH       �	B-�Xc�A�*

loss��D<|��n       �	L�-�Xc�A�*

lossv��<����       �	�/�Xc�A�*

loss6I=�'!�       �	 �/�Xc�A�*

loss��;q?��       �	_a0�Xc�A�*

lossf��;m�IT       �	U�1�Xc�A�*

loss��+<�Me       �	��2�Xc�A�*

loss�k=��Z       �	zT3�Xc�A�*

loss8��<K��       �	I�3�Xc�A�*

loss�
�=A��       �	m�4�Xc�A�*

loss���;�w       �	ǂ5�Xc�A�*

loss=�A=���t       �	�6�Xc�A�*

loss���=s���       �	�6�Xc�A�*

loss��<�       �	�p7�Xc�A�*

lossw$=0�c�       �	�8�Xc�A�*

loss�n=�Q~>       �	{�8�Xc�A�*

lossi�:
W)       �	�c9�Xc�A�*

loss`�	=|���       �	��9�Xc�A�*

loss��y<
��       �	��:�Xc�A�*

loss���=��r       �	�B;�Xc�A�*

lossX� =L�Q        �	��;�Xc�A�*

loss�(#<�wy       �	��<�Xc�A�*

loss6(}<Ym!�       �	�=�Xc�A�*

loss\�L;�i��       �	S�=�Xc�A�*

lossw��;���       �	$E>�Xc�A�*

lossdN=J�Y�       �	��>�Xc�A�*

loss�R <�,׀       �	,�?�Xc�A�*

loss?�?>/���       �	�@�Xc�A�*

loss��<,�       �	=�@�Xc�A�*

loss��m<�;J�       �	XVA�Xc�A�*

loss��{;�du       �	,�A�Xc�A�*

loss'q<�8�       �	J�B�Xc�A�*

loss)b<=��T�       �	#/C�Xc�A�*

loss�?"=�x�8       �	�C�Xc�A�*

loss
^@=_��       �	YlD�Xc�A�*

loss�.�<N�B�       �	�UE�Xc�A�*

loss#�;z/&+       �	��E�Xc�A�*

loss�0�<�i�R       �	O�F�Xc�A�*

loss��<�Bv       �	�.G�Xc�A�*

loss*�<���       �	��G�Xc�A�*

lossz��=CB��       �	�dH�Xc�A�*

loss�SJ<�n0�       �	��H�Xc�A�*

loss�ޕ=2]�6       �	V�I�Xc�A�*

loss�*�;��f       �	�lJ�Xc�A�*

loss�=a��X       �	�K�Xc�A�*

loss�+=8fΊ       �	g�K�Xc�A�*

loss�a{<9扔       �	5L�Xc�A�*

loss/�<b=�       �	
�L�Xc�A�*

loss� �<S��       �	OuM�Xc�A�*

lossO��;���J       �	1N�Xc�A�*

loss���;)��[       �	7�N�Xc�A�*

lossȀ;���       �	�eO�Xc�A�*

lossF�C=��ڜ       �	�P�Xc�A�*

loss���<��G       �	��P�Xc�A�*

loss�� =_>*�       �	;6Q�Xc�A�*

lossR�@=2�K       �	v�Q�Xc�A�*

lossa��<����       �	{R�Xc�A�*

loss�n]:d��       �	�S�Xc�A�*

loss��k<���       �	"�S�Xc�A�*

lossr��;:w��       �	\ZT�Xc�A�*

losssm=��p�       �	�U�Xc�A�*

loss���<R�~�       �	�.V�Xc�A�*

loss���<��	       �	o�V�Xc�A�*

loss��<�a�       �	�X�Xc�A�*

loss���;:��_       �	��X�Xc�A�*

loss���<1�	<       �	�lY�Xc�A�*

loss��f<&?ef       �	^Z�Xc�A�*

loss�"=�-�       �	,�Z�Xc�A�*

loss���<����       �	�X[�Xc�A�*

loss.��;��B�       �	O\�Xc�A�*

loss�I =~�       �	��\�Xc�A�*

loss�g_<Y@�       �	`t]�Xc�A�*

loss&�1=o���       �	�k^�Xc�A�*

loss�6;�~A�       �	�_�Xc�A�*

loss<<p3�       �	�_�Xc�A�*

loss�)�;���       �	�`�Xc�A�*

loss�u+=s�=.       �	�La�Xc�A�*

loss%=�5�y       �	�a�Xc�A�*

loss>�=( �;       �	��b�Xc�A�*

loss���<��D       �	�}c�Xc�A�*

loss<��:a,^\       �	~;d�Xc�A�*

loss`�+;G�m�       �	�d�Xc�A�*

loss�׋=�l�       �	e�e�Xc�A�*

loss�*�<�i��       �	�7f�Xc�A�*

loss-8�=���       �	(g�Xc�A�*

loss[x�<��B       �	�g�Xc�A�*

loss2��<�Ly{       �	�]h�Xc�A�*

loss��;D��-       �	[i�Xc�A�*

loss�=<�\�:       �	t	j�Xc�A�*

lossi�;
���       �	�j�Xc�A�*

lossߞ�<|       �	@Pk�Xc�A�*

lossÅ=(���       �	��k�Xc�A�*

loss��<p���       �	��l�Xc�A�*

loss���<!&'�       �	%�m�Xc�A�*

loss2O<����       �	h�n�Xc�A�*

loss�g<S?j       �	r5o�Xc�A�*

lossL�;]��H       �	�o�Xc�A�*

lossr��<�Co�       �	$}p�Xc�A�*

loss�K�<#��V       �	Xq�Xc�A�*

loss���=m}�]       �	��q�Xc�A�*

loss��:<�l       �	��r�Xc�A�*

losst#><�m       �	�<s�Xc�A�*

loss��=w�y       �	$�s�Xc�A�*

loss�.S<P�A       �	�tt�Xc�A�*

loss@�S=�� _       �	Mu�Xc�A�*

lossfÁ<f۔�       �	��u�Xc�A�*

lossD4;=^(�;       �	�Pv�Xc�A�*

loss6�q;����       �	�w�Xc�A�*

loss�c�=ě*G       �	̶w�Xc�A�*

loss��n=ǫ��       �	Px�Xc�A�*

loss<ł<���       �	��x�Xc�A�*

loss�
�;�瀕       �	��y�Xc�A�*

lossE��;T:�r       �	�Gz�Xc�A�*

loss��<�Cn       �	��z�Xc�A�*

loss�HO;x��       �	��{�Xc�A�*

loss��6<���       �	�>|�Xc�A�*

loss�xV=��to       �	"}�Xc�A�*

loss*�<)iX�       �	��}�Xc�A�*

loss��=�y1       �	�X~�Xc�A�*

lossx��<�L3+       �	
�~�Xc�A�*

loss�w�;E���       �	��Xc�A�*

loss�{;�ѭE       �	�8��Xc�A�*

loss�6'<�Y�h       �	;߀�Xc�A�*

loss��<T�O(       �	�ҁ�Xc�A�*

lossPs;��l       �	�h��Xc�A�*

lossi��<-[\0       �	��Xc�A�*

loss�:�<ūF�       �	����Xc�A�*

loss�=���<       �	kE��Xc�A�*

loss�L�<��=       �	Q݄�Xc�A�*

loss(��;0���       �	�t��Xc�A�*

loss-�0<��\a       �	���Xc�A�*

losssC5=���       �	w���Xc�A�*

losscG�=%�r       �	�6��Xc�A�*

loss�1�<���       �	҇�Xc�A�*

lossV�<Z���       �	̈�Xc�A�*

loss{�<m��       �	�f��Xc�A�*

lossó�<`��$       �	����Xc�A�*

lossZj;+���       �	c���Xc�A�*

loss.r<��       �	~5��Xc�A�*

lossZ�*<��       �	׋�Xc�A�*

loss�ć;h��       �	�w��Xc�A�*

loss-.<�p�       �	4��Xc�A�*

loss�h�<���B       �	���Xc�A�*

loss�V]=}�j�       �	�K��Xc�A�*

loss�|�<�R       �	q㎎Xc�A�*

loss�_�;��2O       �	�|��Xc�A�*

loss;�Z;�?�       �	���Xc�A�*

loss㏡<g�#m       �	����Xc�A�*

loss�@(;uj~       �	�O��Xc�A�*

loss��B;{��       �	�Xc�A�*

loss�D�<m���       �	͔��Xc�A�*

loss�<Fļ�       �	�8��Xc�A�*

loss�	�<���p       �	Г�Xc�A�*

loss��Y=�3E�       �	�m��Xc�A�*

loss��;��       �	���Xc�A�*

lossOi�<E9�       �	B���Xc�A�*

loss�~<P��       �	�Q��Xc�A�*

loss��<L�H       �	��Xc�A�*

lossI]�</Z��       �	W���Xc�A�*

lossV%;��(�       �	�6��Xc�A�*

loss�r=����       �	M٘�Xc�A�*

lossOar=㵙�       �	괙�Xc�A�*

loss�X�<[I��       �	�K��Xc�A�*

lossn�J=��5       �	I���Xc�A�*

loss�ޙ=ws�       �	ꓛ�Xc�A�*

loss*�<t]@�       �	o/��Xc�A�*

loss@�;@d�       �	�Ɯ�Xc�A�*

loss�4<|���       �	1^��Xc�A�*

loss��<�M�H       �	/���Xc�A�*

lossC��;�V��       �	tϞ�Xc�A�*

loss��<��֯       �	f��Xc�A�*

loss��?=�3�       �	���Xc�A�*

loss�9=��
�       �	R���Xc�A�*

loss��<N��       �	g`��Xc�A�*

loss7f�=��       �	����Xc�A�*

loss��c<�p�.       �	���Xc�A�*

loss��$<o�k       �	�i��Xc�A�*

loss�R$<���O       �	?��Xc�A�*

loss&΁<Ϗ:q       �	f���Xc�A�*

loss��7=~U��       �	����Xc�A�*

loss��=��"�       �	F(��Xc�A�*

loss�Q/=I۩�       �	Ϧ�Xc�A�*

loss�gL<Q�>       �		q��Xc�A�*

loss��=8~�       �	��Xc�A�*

loss��N<�BWq       �	,���Xc�A�*

lossU�=�I؞       �	7R��Xc�A�*

loss��;�;h�       �	쩎Xc�A�*

lossI�=Pm��       �	Ƈ��Xc�A�*

loss.d=���       �	V*��Xc�A�*

loss��^=J��       �	�ǫ�Xc�A�*

loss�4>=����       �	|a��Xc�A�*

lossc��<SI;       �	���Xc�A�*

loss�,)=l��w       �	����Xc�A�*

loss�'�=��#T       �	1y��Xc�A�*

loss��d<7�>�       �	'��Xc�A�*

loss�G<�O\�       �	6���Xc�A�*

lossO�;�U?�       �	�G��Xc�A�*

loss���=t	�m       �	�ܰ�Xc�A�*

loss���;��Ϣ       �	ⱎXc�A�*

loss��$=���)       �	�y��Xc�A�*

losso�(=Ү�       �	�&��Xc�A�*

loss��`=��,�       �	ӿ��Xc�A�*

loss��'=o��x       �	 W��Xc�A�*

loss T'<$-�       �	=Xc�A�*

losso.4<��x�       �	f���Xc�A�*

loss�< �|(       �	��Xc�A�*

lossC:�<�n�       �	F���Xc�A�*

loss��<c��       �	�U��Xc�A�*

loss;�4=�|��       �	�뷎Xc�A�*

loss��s=�e&�       �	����Xc�A�*

lossxV@=e��,       �	�,��Xc�A�*

loss�s0=s�(       �	���Xc�A�*

loss�0=4Q̱       �	K���Xc�A�*

loss��<�ㄐ       �	�O��Xc�A�*

loss�=B;n��i       �	8���Xc�A�*

loss��<R��       �	穼�Xc�A�*

loss���<d�       �	ML��Xc�A�*

loss&��<4H|�       �	d꽎Xc�A�*

loss7��<�D�       �	����Xc�A�*

lossc�v;I���       �	Ql��Xc�A�*

loss�{�=���       �	���Xc�A�*

loss!��;�Va^       �	'���Xc�A�*

lossF�[<ωK7       �	K��Xc�A�*

losss�=:	��       �	����Xc�A�*

loss�s�<pwS5       �	G�Xc�A�*

loss!��;�lt}       �	^*ÎXc�A�*

loss�%o<4���       �	7�ÎXc�A�*

loss��,;ɬ7       �	dĎXc�A�*

loss�	;xyx~       �	z�ĎXc�A�*

loss9�:,�x�       �	��ŎXc�A�*

loss�ʜ<���       �	W@ƎXc�A�*

lossZ)�<>_�       �	��ƎXc�A�*

loss��<A�Ύ       �	t�ǎXc�A�*

lossF��<$;|�       �	@1ȎXc�A�*

lossE�V=>49&       �	��ȎXc�A�*

lossJF-=03�       �	iɎXc�A�*

loss��<�q,�       �	�ʎXc�A�*

loss�|=�"!D       �	m�ʎXc�A�*

loss?9�=�#�       �	NAˎXc�A�*

loss��G:���       �	b�ˎXc�A�*

loss���</B��       �	qu̎Xc�A�*

loss��<О�       �	�͎Xc�A�*

lossAڍ<J�hp       �	�͎Xc�A�*

loss�4t=�\i
       �	IeΎXc�A�*

lossV��<���4       �	@�ΎXc�A�*

loss!�<�]�m       �	S�ώXc�A�*

loss��2=.�       �	�&ЎXc�A�*

loss8x�;03w       �	\�ЎXc�A�*

loss��=%Y�l       �	�bюXc�A�*

lossE+E=�       �	p%ҎXc�A�*

lossvʹ<�O<       �	��ҎXc�A�*

lossL�;�PH�       �	oeӎXc�A�*

loss�=��&       �	ԎXc�A�*

loss =�3*       �	��ԎXc�A�*

loss)��:�ET�       �	�QՎXc�A�*

loss%��<��0�       �	��ՎXc�A�*

loss{�<*��       �	&�֎Xc�A�*

loss�_�<K��       �	�F׎Xc�A�*

loss�_�<�o��       �	�׎Xc�A�*

loss��
=V+s�       �	�~؎Xc�A�*

lossR��<�l`�       �	�&َXc�A�*

loss�6I<�O�       �	j�َXc�A�*

lossvd<�D�       �	�ZڎXc�A�*

loss6�<v��       �	X�ڎXc�A�*

loss�]�:��|       �	��ێXc�A�*

loss�ݑ<�]�#       �	�@܎Xc�A�*

loss[��;�7��       �	E�܎Xc�A�*

loss��C<z�Ю       �	��ݎXc�A�*

loss�=5��       �	y;ގXc�A�*

loss�k<�FZ�       �	��ގXc�A�*

lossi0<7'�       �	�uߎXc�A�*

lossFq�=�77C       �	���Xc�A�*

loss�3�<a�       �	����Xc�A�*

loss�˘=���.       �	�W�Xc�A�*

loss��<M���       �	���Xc�A�*

lossa�W</B>�       �	���Xc�A�*

lossha7<+���       �	;��Xc�A�*

loss�=$<�O       �	���Xc�A�*

loss-.=��,       �	���Xc�A�*

loss���= &       �	���Xc�A�*

loss6^3;�"��       �	�%�Xc�A�*

lossK=���       �	���Xc�A�*

loss0�<����       �	�c�Xc�A�*

loss��<���       �	*��Xc�A�*

lossv�p=���/       �	��Xc�A�*

loss�v�=]���       �	*;�Xc�A�*

loss8�=�6A�       �	z��Xc�A�*

lossH��;��x       �	�x�Xc�A�*

loss��	<.�E%       �	�+�Xc�A�*

loss��=<'ԗ       �	*��Xc�A�*

losss�c<�"�       �	�a�Xc�A�*

loss@a�<4[��       �	e�Xc�A�*

loss�=$l�h       �	<��Xc�A�*

loss�#=(m��       �	1?��Xc�A�*

loss+�=4�a       �	����Xc�A�*

loss�E�=���       �	܀�Xc�A�*

lossX,1=���       �	��Xc�A�*

loss&~;{'�>       �	���Xc�A�*

loss��=Qp�c       �	ԁ�Xc�A�*

loss��=�}պ       �	�0�Xc�A�*

loss���<b ��       �	G��Xc�A�*

loss�d=룱�       �	�u��Xc�A�*

loss�w3=��R       �	q��Xc�A�*

loss(h�<��V       �	���Xc�A�*

loss7��<��3       �	�p��Xc�A�*

loss/��<B[��       �	���Xc�A�*

loss��<��@�       �	9���Xc�A�*

lossq�<<�T�       �	l^��Xc�A�*

loss�E�=ȼ�       �	�H��Xc�A�*

lossZQI;��XI       �	A���Xc�A�*

loss��.=rP       �	-���Xc�A�*

lossl\�<
�_�       �	pD��Xc�A�*

losss��=/f       �	*���Xc�A�*

lossLV<���7       �	����Xc�A�*

loss��:��s       �	�$��Xc�A�*

loss���<�B�       �	���Xc�A�*

lossN�==�ƐW       �	�[��Xc�A�*

loss�l�;���f       �	����Xc�A�*

lossg�=b<c       �	�� �Xc�A�*

loss��=�l�       �	�6�Xc�A�*

lossV�<j~       �	���Xc�A�*

loss���<����       �	{�Xc�A�*

loss�=�,}�       �	��Xc�A�*

lossd{g<�e�       �	��Xc�A�*

loss�;��H!       �	�`�Xc�A�*

loss���;�;Gs       �	���Xc�A�*

loss|�e=X/�Q       �	͔�Xc�A�*

loss*:<��g$       �	P5�Xc�A�*

lossB[=���       �	��Xc�A�*

lossm%;��j�       �	�z�Xc�A�*

loss���<�0Mb       �	��Xc�A�*

lossL�;o��       �	|��Xc�A�*

lossԃ<���O       �	TU	�Xc�A�*

loss�f�<�)w/       �	��	�Xc�A�*

lossT�
=�H�7       �	V�
�Xc�A�*

loss�>b<�g��       �	�"�Xc�A�*

loss�V�<YE       �	I��Xc�A�*

loss�e1=�x�       �	/O�Xc�A�*

loss���<Ȁ��       �	<��Xc�A�*

lossa��<���V       �	=��Xc�A�*

lossVd =��A�       �	:�Xc�A�*

loss�׫<+��       �	���Xc�A�*

lossl7:=ߤ<'       �	�{�Xc�A�*

loss�N=�F�       �	��Xc�A�*

loss(�=qu       �	'��Xc�A�*

loss`�;4�se       �	�^�Xc�A�*

loss,��=��'�       �	t
�Xc�A�*

loss�j�<���       �	U��Xc�A�*

loss�d?<P�Wp       �	^��Xc�A�*

lossD�< ��       �	�A�Xc�A�*

loss���<�:��       �	���Xc�A�*

loss،=���a       �	��Xc�A�*

loss��< ��s       �	g`�Xc�A�*

loss=��<.t�f       �	4��Xc�A�*

loss�o=��       �	��Xc�A�*

loss���<�_�       �	�"�Xc�A�*

loss�,>{"C�       �	���Xc�A�*

loss�]"<1��       �	�\�Xc�A�*

loss/.[<b�xl       �	���Xc�A�*

loss�zM=�D�       �	 ��Xc�A�*

lossO�<��+]       �	l&�Xc�A�*

loss`&;ܺ��       �	j��Xc�A�*

lossA�;���       �	k�Xc�A�*

loss��<�9�       �	��Xc�A�*

loss쩁<mܱ]       �	ػ�Xc�A�*

loss�AM=`��       �	(d�Xc�A�*

lossc><�J*:       �	��Xc�A�*

loss�ְ<����       �	ղ�Xc�A�*

loss-�d;ű       �	�O �Xc�A�*

loss{��=�Y_�       �	�� �Xc�A�*

loss�C<�I�]       �	.�!�Xc�A�*

loss�]�:�a^�       �	�f"�Xc�A�*

loss
��;���       �	�2#�Xc�A�*

loss��<���       �	��#�Xc�A�*

loss<b<Ƌ��       �	�_$�Xc�A�*

loss))R;~�S       �	n%�Xc�A�*

loss�H<��gR       �	��%�Xc�A�*

loss`�;-�       �	|&�Xc�A�*

lossV��<�	|       �	�)'�Xc�A�*

loss%~�<9�_K       �	��'�Xc�A�*

loss���<�W��       �	��(�Xc�A�*

loss6�"<���d       �	Â)�Xc�A�*

lossJj�;4��       �	VI*�Xc�A�*

loss��T<�a�       �	��*�Xc�A�*

loss�=��E       �	�#,�Xc�A�*

loss�M=�D��       �	��,�Xc�A�*

loss�08=�c��       �	ds-�Xc�A�*

loss�q=��[�       �	�.�Xc�A�*

loss��<��yR       �	>�.�Xc�A�*

lossV
#<��'       �	GW/�Xc�A�*

loss�׌;M��3       �	��/�Xc�A�*

loss�k�:�;6       �	F�0�Xc�A�*

loss<��<�f�       �		71�Xc�A�*

loss��q=�U��       �	�1�Xc�A�*

loss�S<}|M       �	?t2�Xc�A�*

loss���<�3YO       �	D3�Xc�A�*

loss��;�       �	y�3�Xc�A�*

lossM�=�>       �	a4�Xc�A�*

loss2�v<�r��       �	�5�Xc�A�*

loss��<Ho �       �	��5�Xc�A�*

lossx�+<f�       �	�p6�Xc�A�*

loss���<��(�       �	r7�Xc�A�*

loss�W�;{��       �	��7�Xc�A�*

lossh��;��)       �	�S8�Xc�A�*

loss��G<(��$       �	�8�Xc�A�*

loss��=$
s       �	C�9�Xc�A�*

loss�r�<G�`�       �	�):�Xc�A�*

loss�f�;�#W-       �	�;�Xc�A�*

loss��0=�k'       �	l"<�Xc�A�*

lossܱ<j<2       �	>=�Xc�A�*

loss�C<!��       �	��=�Xc�A�*

loss@|*=�Y�T       �	0�>�Xc�A�*

losse�<ku�       �	�X?�Xc�A�*

loss�@=]�=�       �	��?�Xc�A�*

loss���<Kp7�       �	Q�@�Xc�A�*

loss�ѳ<.ƥ       �	�A�Xc�A�*

loss��>���M       �	]�A�Xc�A�*

loss;db=�H#       �	)[B�Xc�A�*

loss�f�<̀�       �	��B�Xc�A�*

loss�lu<׎�c       �	V�C�Xc�A�*

loss�p<`k�       �	u;D�Xc�A�*

loss�.<Y�@I       �	rE�Xc�A�*

loss��D<=�A       �	@�E�Xc�A�*

losse�M<0a�       �	�zF�Xc�A�*

loss���=]���       �	!G�Xc�A�*

loss�P;��`       �	Z�G�Xc�A�*

loss�};�#�R       �	=_H�Xc�A�*

loss��
=��       �	��H�Xc�A�*

lossM�,=��j       �	��I�Xc�A�*

loss�<ru��       �	�4J�Xc�A�*

loss-��;$o��       �	�J�Xc�A�*

loss���<���C       �	d�K�Xc�A�*

loss1f=��N�       �	^�L�Xc�A�*

loss�J�=ڲ��       �	�+M�Xc�A�*

loss�H�<����       �	p�M�Xc�A�*

loss�j;EӢ#       �	�rN�Xc�A�*

loss'�;N�g�       �	N�O�Xc�A�*

lossHI=�7D       �	[CP�Xc�A�*

lossv	<��{       �	}�P�Xc�A�*

lossD��<l�B       �	__R�Xc�A�*

lossI/�<Y�%�       �	b�R�Xc�A�*

loss���;hh�       �	��S�Xc�A�*

loss(��;�n\<       �	�;T�Xc�A�*

loss΍<2w|3       �	��T�Xc�A�*

loss�u=�H��       �	�|U�Xc�A�*

loss���;q�M       �	CV�Xc�A�*

loss:k�<��       �	�8W�Xc�A�*

loss�VR<�e��       �	�GX�Xc�A�*

loss�u�<��Z}       �	��X�Xc�A�*

lossͥE=-�       �	|Y�Xc�A�*

loss�J�;����       �	/Z�Xc�A�*

loss�%!=�e�_       �	��Z�Xc�A�*

loss!��;��l       �	�N[�Xc�A�*

loss7��=�g��       �	:�[�Xc�A�*

loss
��<=��^       �	܄\�Xc�A�*

loss�z&=y��       �	!]�Xc�A�*

lossWen<�%RN       �	Ժ]�Xc�A�*

loss�ac<&�R       �	\X^�Xc�A�*

loss��<���b       �	_�Xc�A�*

loss\��<����       �	=�_�Xc�A�*

loss4�a=Α$�       �	�L`�Xc�A�*

loss��<����       �	��`�Xc�A�*

lossjg�< ��@       �	l|a�Xc�A�*

loss��3=]�&       �	�b�Xc�A�*

loss�'<�o~       �	��b�Xc�A�*

lossO��<Bz��       �	xFc�Xc�A�*

loss)�;��\�       �	��c�Xc�A�*

loss���<�7�v       �	e5e�Xc�A�*

loss��d=�y9       �	��e�Xc�A�*

loss!=}�5N       �	�lf�Xc�A�*

loss�Q=�'��       �	g�Xc�A�*

loss�Z�<"K)R       �	^h�Xc�A�*

lossR�<�G�g       �	��h�Xc�A�*

loss\f�;x�{�       �	tBi�Xc�A�*

loss�	2=�=�*       �	��i�Xc�A�*

lossJ;p;��8z       �	�yk�Xc�A�*

loss�<����       �	0l�Xc�A�*

loss��:�'       �	p�l�Xc�A�*

loss	��<�       �	)\m�Xc�A�*

loss�<<	��       �	��m�Xc�A�*

loss���<V!J       �	��n�Xc�A�*

lossv,�;���       �	 :o�Xc�A�*

loss��<�A        �	Ip�Xc�A�*

loss�7�<�Y��       �	�p�Xc�A�*

lossL�<o�K|       �	h@q�Xc�A�*

loss�ʟ<�:A�       �	#�q�Xc�A�*

lossl�R<�?+       �	�}r�Xc�A�*

loss���<=��       �		s�Xc�A�*

loss�]@<�c�q       �	�s�Xc�A�*

loss�l�<a,�p       �	gHt�Xc�A�*

loss�_e<Q&��       �	 u�Xc�A�*

loss��*<d�a       �	�u�Xc�A�*

loss���;�v=�       �	jLv�Xc�A�*

lossEp�;T[�       �	�v�Xc�A�*

loss.�1;��       �	dww�Xc�A�*

loss
L�;E��.       �	�Fx�Xc�A�*

loss��9�9q       �	n�x�Xc�A�*

loss�͏;�_L       �	�ty�Xc�A�*

loss�x�<���       �	hz�Xc�A�*

lossB�;J�X       �	�F{�Xc�A�*

loss#|<��q       �	�	|�Xc�A�*

loss��%;�R`�       �	�|�Xc�A�*

lossE�=.�~�       �	�}�Xc�A�*

losse	�<�E��       �	J@~�Xc�A�*

loss��q<�f�       �	��~�Xc�A�*

loss/�<=�a\       �	)z�Xc�A�*

loss[g< aV       �	e��Xc�A�*

loss�=:��'c       �	����Xc�A�*

loss=�};UAl�       �	�@��Xc�A�*

loss@�X:тC�       �	^�Xc�A�*

lossC|d;ߧV�       �	���Xc�A�*

lossU��<���y       �	G>��Xc�A�*

loss#�$:���       �	�Ճ�Xc�A�*

loss��:j�1       �	N|��Xc�A�*

loss�aU<�F[E       �	S ��Xc�A�*

loss��y9��'       �	�Å�Xc�A�*

losslѐ:����       �	�`��Xc�A�*

loss�Y�;O6��       �	����Xc�A�*

lossӹ�;�P�3       �	d���Xc�A�*

loss��v<�&^�       �	7R��Xc�A�*

lossL�:�<w�       �	{��Xc�A�*

lossT��:��w`       �	���Xc�A�*

loss��=�R�       �	�R��Xc�A�*

lossO1�:���       �	_늏Xc�A�*

lossx��=ڣ*�       �	���Xc�A�*

losst��<�ė�       �	�"��Xc�A�*

loss"]�<`���       �	����Xc�A�*

lossNڨ<B�VT       �	aQ��Xc�A�*

loss���;cf�       �	a���Xc�A�*

loss���<l��       �	󑎏Xc�A�*

loss�!=���4       �	o*��Xc�A�*

loss�]�;Rx��       �	ˏ�Xc�A�*

loss���<_��}       �	�c��Xc�A�*

lossJ� <���;       �	����Xc�A�*

loss���<���       �	C���Xc�A�*

loss��=>#�       �	�a��Xc�A�*

loss)r;��^�       �	v���Xc�A�*

loss�f'=q�S7       �	����Xc�A�*

loss���<�i�       �	�$��Xc�A�*

loss�G�;g�w�       �	
���Xc�A�*

lossVQ�<x��       �	�\��Xc�A�*

loss(ޛ<?�o       �	�Xc�A�*

loss�ә<�B�       �	����Xc�A�*

loss�G:<���       �	T��Xc�A�*

lossjd6=I�i       �	���Xc�A�*

loss��< f,       �	�E��Xc�A�*

loss�3;{ػ�       �	ژ�Xc�A�*

lossҬ<�A�       �	Gq��Xc�A�*

loss�8<�vv       �	%��Xc�A�*

loss�N<��o       �	Н��Xc�A�*

loss�i�:��e�       �	z6��Xc�A�*

loss)Q+=�}L}       �	o֛�Xc�A�*

loss^�=?#�       �	����Xc�A�*

loss��<��<z       �	�'��Xc�A�*

loss`��9�s�       �	2˝�Xc�A�*

lossy�=К��       �	f��Xc�A�*

lossO��:����       �	���Xc�A�*

lossEM�;(�^       �	����Xc�A�*

loss'R ;`���       �	�P��Xc�A�*

loss�x<i�[�       �	����Xc�A�*

lossw��<�8�R       �	䞡�Xc�A�*

loss���<��,       �	�H��Xc�A�*

loss�I�;�L�r       �	x�Xc�A�*

loss���<E�է       �	$���Xc�A�*

loss߄p<�\        �	B@��Xc�A�*

loss�ۂ=���-       �	�餏Xc�A�*

loss�7<;��D       �	����Xc�A�*

lossE��<��`$       �	/o��Xc�A�*

loss��<���       �	kE��Xc�A�*

loss�j�=��B       �	,��Xc�A�*

loss�� <`��       �	Ψ�Xc�A�*

loss,�#<��GQ       �	)���Xc�A�*

loss�e=�Z0       �	mV��Xc�A�*

lossZL�;��       �	����Xc�A�*

loss�<(?�       �	�FďXc�A�*

lossf.=�Q       �	G�ďXc�A�*

lossھ�=�6K       �	j�ŏXc�A�*

loss�Xf<�Df       �	3mƏXc�A�*

loss鹝;G�b       �	�CǏXc�A�*

loss��< 2       �	��ǏXc�A�*

loss��<��Ҵ       �	�yȏXc�A�*

loss��<G��       �	MɏXc�A�*

lossw�=R)�       �	��ɏXc�A�*

lossWZ<�c-<       �	QJʏXc�A�*

loss{T�<\��?       �	��ʏXc�A�*

loss� T=����       �	u�ˏXc�A�*

loss��t=i�!�       �	�/̏Xc�A�*

loss�S
<[��       �	h�̏Xc�A�*

loss�k�<k�}       �	��͏Xc�A�*

lossn��=)��8       �	SvΏXc�A�*

losss�:��a�       �	fϏXc�A�*

loss�#�;K��       �	�ϏXc�A�*

loss�c�<}�.�       �	�QЏXc�A�*

loss�;�<�s�       �	��ЏXc�A�*

loss6�d<jA�       �	\�яXc�A�*

lossV�=��4�       �	Y3ҏXc�A�*

loss̧�;w2b       �	>�ҏXc�A�*

loss ��<��       �	uӏXc�A�*

loss�
X;k��P       �	AԏXc�A�*

lossC��;�b[       �	�ԏXc�A�*

loss�9<6g��       �	�CՏXc�A�*

lossmA<+8X       �	��ՏXc�A�*

loss�=�w�X       �	�֏Xc�A�*

loss,�<3�`       �	K׏Xc�A�*

loss}]=fH�~       �	k�׏Xc�A�*

loss�<��8       �	�Y؏Xc�A�*

loss;��<���H       �	��؏Xc�A�*

loss�=�31$       �	ُ̚Xc�A�*

loss��Z=�׽�       �	�6ڏXc�A�*

loss���<}�3�       �	��ڏXc�A�*

lossXp�;ZD�u       �	�ۏXc�A�*

loss(�/=\�       �	�܏Xc�A�*

loss͖�< �       �	�܏Xc�A�*

loss��J=���       �	QݏXc�A�*

loss���<76]�       �	�#ޏXc�A�*

loss%�;0"�.       �	&�ޏXc�A�*

loss��p;���       �	kߏXc�A�*

loss�Y|<અ�       �	�	��Xc�A�*

lossׂ[<�~�       �	m���Xc�A�*

loss�},=>���       �	�D�Xc�A�*

loss���;�գx       �	���Xc�A�*

loss��H<
��e       �	*��Xc�A�*

lossMQ�;O<V�       �	�/�Xc�A�*

loss,W;�A'�       �	���Xc�A�*

lossZƺ;'P�       �	��Xc�A�*

loss�Y8<�P�6       �	B��Xc�A�*

lossΝ�;ڑ/       �	���Xc�A�*

loss�7n>�7��       �	Y�Xc�A�*

losso��<+JA�       �	���Xc�A�*

loss�&;����       �	�>�Xc�A�*

loss7��:MK�g       �	s��Xc�A�*

loss�9�70       �	�p�Xc�A�*

loss2nL<<��       �	!��Xc�A�*

loss:��=�}p'       �	��Xc�A�*

loss�Ǣ=�k/P       �	�i�Xc�A�*

loss(B�;��b�       �	��Xc�A�*

loss�~<����       �	���Xc�A�*

loss���<����       �	�5�Xc�A�*

loss�Sf;_Hl#       �	���Xc�A�*

loss8�F;l��       �	�j��Xc�A�*

lossf�W=B4��       �	G�Xc�A�*

loss���<9<��       �	@��Xc�A�*

loss
݇=��       �	C9�Xc�A�*

loss`68=��;�       �	���Xc�A�*

lossJm�<�0�       �	��Xc�A�*

loss�=����       �	�,�Xc�A�*

loss��;��p       �	��Xc�A�*

loss&<�$       �	$^��Xc�A�*

loss�3<�<7�       �	@���Xc�A�*

lossJw&<���       �	T���Xc�A�*

loss�l�;7�[q       �	�#��Xc�A�*

loss���<�B:       �	����Xc�A�*

loss��<��us       �	ca��Xc�A�*

loss&��<�uL       �	���Xc�A�*

loss@�<=�3       �	����Xc�A�*

loss�|�<�c��       �	�F��Xc�A�*

loss4p<��B�       �	����Xc�A�*

loss�ƣ:�@�B       �	����Xc�A�*

lossA�w=�E]I       �	�]��Xc�A�*

lossx�`=�[q       �	����Xc�A�*

loss��<���R       �	ō��Xc�A�*

loss�}<��'       �	�'��Xc�A�*

losss��=�(��       �	���Xc�A�*

loss�<4d�<       �	T��Xc�A�*

loss�=�z�       �	9���Xc�A�*

lossX��<d���       �	#� �Xc�A�*

lossdT�<c�       �	CX�Xc�A�*

loss�ՠ<��nd       �	%�Xc�A�*

loss�p5="؝�       �	��Xc�A�*

loss4�<�q�0       �	76�Xc�A�*

loss4�=�q��       �	���Xc�A�*

loss��<kT|�       �	�e�Xc�A�*

loss���<{�j       �	���Xc�A�*

lossj��;/��Q       �	���Xc�A�*

loss� �;�*�#       �	�Z�Xc�A�*

loss�Ҙ:.Mf�       �	���Xc�A�*

loss��<��#�       �	���Xc�A�*

lossi�=�       �	c+�Xc�A�*

loss* =�M       �	>��Xc�A�*

loss)�<C�A�       �	�v	�Xc�A�*

loss��4<�h�7       �	�
�Xc�A�*

loss5;(d       �	*�
�Xc�A�*

loss�y�<z ��       �	�?�Xc�A�*

lossP��<�W3a       �	���Xc�A�*

loss�D=���       �	�t�Xc�A�*

loss6��;��!;       �	��Xc�A�*

loss�TC<O��       �	���Xc�A�*

loss8�;[Ɯ�       �	7S�Xc�A�*

loss�{�<�ɎD       �	���Xc�A�*

loss|�;	�|       �	Z��Xc�A�*

loss}S�<���       �	D�Xc�A�*

loss=��<��       �	���Xc�A�*

loss�;��<�       �	��Xc�A�*

loss�e�:;��|       �	'�Xc�A�*

lossz
b<�Ն       �	��Xc�A�*

loss�K2<��g       �	Ho�Xc�A�*

loss،t;�s�       �	]�Xc�A�*

lossW�;�}��       �	��Xc�A�*

lossԬ�;A���       �	�V�Xc�A�*

loss��s<±\�       �	��Xc�A�*

loss|\<�HO       �	C��Xc�A�*

loss3��;�b�-       �	�)�Xc�A�*

lossf;=�+�       �	��Xc�A�*

loss���< �j       �	�m�Xc�A�*

loss���<
��t       �	�
�Xc�A�*

loss�g�;+�]�       �	���Xc�A�*

loss��<�M/       �	C=�Xc�A�*

loss�#�<�U       �	���Xc�A�*

loss�!=�éK       �	=��Xc�A�*

lossݝ<�{S�       �	�2�Xc�A�*

loss�+=�\D       �	)��Xc�A�*

loss4�H<۰�       �	w�Xc�A�*

loss��;���       �	�#�Xc�A�*

loss;<���#       �	��Xc�A�*

losshp�;u�&j       �	�_�Xc�A�*

lossL��;!y:�       �	���Xc�A�*

loss��<�2�&       �	� �Xc�A�*

loss���<�>�       �	:<!�Xc�A�*

lossf!<��       �	��!�Xc�A�*

loss�u�;�b�       �	U�"�Xc�A�*

lossXΙ;�P�       �	�#�Xc�A�*

loss3Iw;���       �	�r$�Xc�A�*

loss���;���       �	%�Xc�A�*

lossI\<[�7U       �	��%�Xc�A�*

loss<{\�       �	�N&�Xc�A�*

loss�d�=��Y�       �	��&�Xc�A�*

loss��I<�?�y       �	
�'�Xc�A�*

lossb<�ױ       �	�z(�Xc�A�*

loss��=0p[       �	$)�Xc�A�*

lossI�f:�       �	��)�Xc�A�*

loss��;��       �	�c*�Xc�A�*

loss�6<���O       �	�^+�Xc�A�*

loss�/]<D�U       �	�,�Xc�A�*

lossS��;��       �	̲,�Xc�A�*

loss�vL=3Z�       �	^d-�Xc�A�*

loss�<5�"       �	X�-�Xc�A�*

loss9B=�6�       �	��.�Xc�A�*

loss}�i;�j�/       �	G:/�Xc�A�*

loss�T�;���Y       �	��/�Xc�A�*

loss&~M<b(�       �	s�0�Xc�A�*

loss���:��Ul       �	F"1�Xc�A�*

lossM��<)�|       �	a�1�Xc�A�*

lossl��;���       �	�2�Xc�A�*

loss{�3=�i�8       �	�3�Xc�A�*

loss���=F��D       �	��3�Xc�A�*

loss
��<���       �	B^4�Xc�A�*

lossL3;^�       �	 5�Xc�A�*

loss�$<	뵱       �	�@6�Xc�A�*

loss:��:ZN�       �	o�6�Xc�A�*

loss�l=,y�       �	�p7�Xc�A�*

loss�m�<s[��       �	�g8�Xc�A�*

loss �~<X�ǳ       �	�8�Xc�A�*

loss#+G<��       �	 �9�Xc�A�*

loss{�U=B�~�       �	I0:�Xc�A�*

loss���:qcL�       �	�:�Xc�A�*

loss�Q=3�o       �	 �;�Xc�A�*

loss�D�;�	�c       �	X;<�Xc�A�*

losss*�<u�s       �	M�<�Xc�A�*

loss���:�h       �	��=�Xc�A�*

loss���;���l       �		�>�Xc�A�*

lossO�A=���%       �	�"?�Xc�A�*

lossa�>=���2       �	$�?�Xc�A�*

lossM	=�\L       �	�F@�Xc�A�*

loss=��<�~��       �	�@�Xc�A�*

loss���=N�s�       �	�zA�Xc�A�*

loss�tc<xO(M       �	<B�Xc�A�*

loss��<Fb�       �	��B�Xc�A�*

loss},b<�Į�       �	��C�Xc�A�*

loss̒�;\tS;       �	'jD�Xc�A�*

loss���<�0�4       �	R
E�Xc�A�*

loss[��;���E       �	��E�Xc�A�*

loss�?�<&�u+       �	�HF�Xc�A�*

losse�<�C6�       �	��F�Xc�A�*

loss�<�%�       �	B�G�Xc�A�*

loss�.�<�̰Y       �	�nH�Xc�A�*

loss4d<�cRB       �	�I�Xc�A�*

loss��<�ѵ�       �	��I�Xc�A�*

loss�i7<4ts7       �	�OJ�Xc�A�*

loss�C�<g@�       �	7�J�Xc�A�*

loss�#=��~-       �	�K�Xc�A�*

loss�D=br�       �	�?L�Xc�A�*

lossR�7='+��       �	~9M�Xc�A�*

lossF�e<G�ו       �	;�M�Xc�A�*

loss4r"=�bn       �	ڌN�Xc�A�*

loss�+{<�HU        �	�.O�Xc�A�*

loss��<̔��       �	E�O�Xc�A�*

loss#_�<:�=�       �	��P�Xc�A�*

lossow|;��k�       �	d<Q�Xc�A�*

loss�3W=��RR       �	:�Q�Xc�A�*

loss��0=��?       �	g�R�Xc�A�*

lossjP=���       �	�ES�Xc�A�*

lossU!=S�?H       �	��S�Xc�A�*

loss���=d�ۉ       �	��T�Xc�A�*

loss�/\<e$�       �	V-U�Xc�A�*

loss`�<�G�       �	��U�Xc�A�*

lossS-)<��w�       �	��V�Xc�A�*

lossE��;7z��       �	cBW�Xc�A�*

loss)f=򴳕       �	��W�Xc�A�*

lossA[;����       �	.�X�Xc�A�*

lossE��;��ș       �	CZ�Xc�A�*

loss�++=�rr       �	k�Z�Xc�A�*

lossh�5<|T]�       �	�W[�Xc�A�*

loss���<��o       �	��[�Xc�A�*

loss_w<�kS       �	��\�Xc�A�*

loss�5</V��       �	�#]�Xc�A�*

lossN�|;�I�       �	�1^�Xc�A�*

loss;�k=����       �	%�^�Xc�A�*

loss�)<[��       �	�b_�Xc�A�*

loss
Q=ox�U       �	��_�Xc�A�*

loss��<W��       �	�`�Xc�A�*

loss���;�q�e       �	}xa�Xc�A�*

loss��l<�٤       �	�b�Xc�A�*

lossA�;�       �	ͯb�Xc�A�*

loss%�a:芗        �	�Ec�Xc�A�*

loss}Kp<'cZ       �	��c�Xc�A�*

loss3�G<6���       �	lxd�Xc�A�*

loss��<q�e       �	v4e�Xc�A�*

loss��<WBÇ       �	��e�Xc�A�*

lossM�<�՛$       �	w�f�Xc�A�*

loss;Q�;q'�       �	�g�Xc�A�*

loss	�S<1J��       �	��i�Xc�A�*

loss �;g5P�       �	�j�Xc�A�*

loss��<���       �	�j�Xc�A�*

lossŽ�<�$�       �	4Kk�Xc�A�*

loss�ʀ<(z��       �	�Qm�Xc�A�*

loss��<w���       �	��m�Xc�A�*

lossj�6<%��       �	~�n�Xc�A�*

loss�5<Le1�       �	�Po�Xc�A�*

lossr�5;��       �	)�o�Xc�A�*

loss�;I�}       �	�~p�Xc�A�*

lossI�<���       �	eq�Xc�A�*

lossQ�<��       �	��q�Xc�A�*

loss���<TL�x       �	Ir�Xc�A�*

loss�n:=���       �	��r�Xc�A�*

loss}��<�~=�       �	J}s�Xc�A�*

lossvi@<����       �	Z+t�Xc�A�*

loss컣<�2!       �	�t�Xc�A�*

loss�K�<����       �	wu�Xc�A�*

lossOG=�g�       �	�v�Xc�A�*

loss̇:	#��       �	p�v�Xc�A�*

losss��<6�X�       �	�Vw�Xc�A�*

loss�	<��       �	��w�Xc�A�*

lossfyT=��:       �	��x�Xc�A�*

loss*=���4       �	9y�Xc�A�*

loss��<�՞       �	��y�Xc�A�*

lossŪ�<T`b       �	Wwz�Xc�A�*

lossx��<b,u       �	('{�Xc�A�*

loss<o�<���|       �	*�{�Xc�A�*

loss���="ޛ�       �	e|�Xc�A�*

loss�3�=n�N`       �		�|�Xc�A�*

loss�F:<�P�       �	��}�Xc�A�*

losslU.<"D�       �	~Q~�Xc�A�*

loss��<-��       �	Q�~�Xc�A�*

losso�+=�J       �	���Xc�A�*

lossf3�:kW�s       �	A���Xc�A�*

loss1k�;�.�       �	�Y��Xc�A�*

loss��V<��%       �	����Xc�A�*

loss�J�<94�       �	����Xc�A�*

loss�1�=���>       �	�M��Xc�A�*

lossG=�R<j       �	�惐Xc�A�*

loss�6S=���       �	O���Xc�A�*

loss̆�<�?�       �	�)��Xc�A�*

loss�r<	�        �	cЅ�Xc�A�*

lossź9<�@v�       �	[{��Xc�A�*

loss�ʫ:_; �       �	]R��Xc�A�*

loss��=��       �	3���Xc�A�*

loss�7z;"�o�       �	����Xc�A�*

lossz��;:c�$       �	�4��Xc�A�*

loss�Rb=�T�$       �	Ή�Xc�A�*

loss�<��4j       �	j��Xc�A�*

loss���<wI�       �	�8��Xc�A�*

loss���<2rn)       �	q⋐Xc�A�*

loss҄C<J �d       �	]���Xc�A�*

lossD��=:<v�       �	�"��Xc�A�*

loss4	Y<���       �	ӽ��Xc�A�*

loss���<؛��       �	�R��Xc�A�*

loss�w<�B}k       �	�玐Xc�A�*

loss�<��g       �	ԁ��Xc�A�*

loss�u�<��P       �	.W��Xc�A�*

loss�zv=�]�       �	A���Xc�A�*

loss��e;�{       �	����Xc�A�*

loss�)�<F���       �	O$��Xc�A�*

loss1��<z��       �	ג�Xc�A�*

loss�9�<>/l       �	�j��Xc�A�*

lossz�J;��s�       �	m��Xc�A�*

loss��@<��!�       �	6���Xc�A�*

loss�d=Dpe6       �	|D��Xc�A�*

lossAs2<S�7       �	3���Xc�A�*

lossT�<��       �	�#��Xc�A�*

lossOK�=���       �	�ɗ�Xc�A�*

loss8�<��       �	�a��Xc�A�*

loss�<\��       �	��Xc�A�*

loss
��<���8       �	����Xc�A�*

loss�3W<O��       �	�/��Xc�A�*

lossx#T<���       �	̚�Xc�A�*

loss�_�=D�
Y       �	�w��Xc�A�*

loss���<�2�       �	��Xc�A�*

lossQ�;I~C       �	���Xc�A�*

lossH�= ��O       �	�R��Xc�A�*

loss@)=ODm�       �	�P��Xc�A�*

loss`S�<<���       �	���Xc�A�*

loss��`=΄Q       �	����Xc�A�*

loss��;=��ȝ       �	2��Xc�A�*

loss��4=���       �	�ؠ�Xc�A�*

lossiA=���[       �	ퟡ�Xc�A�*

loss�C�<d-vn       �	�@��Xc�A�*

loss���;��X_       �	�٢�Xc�A�*

loss��4;4�	Q       �	Hp��Xc�A�*

loss̦�=� �&       �	A��Xc�A�*

lossë�;���|       �	;���Xc�A�*

loss�k�<�Q�       �	+���Xc�A�*

loss�|�<���       �	����Xc�A�*

loss!_�<m�_'       �	bL��Xc�A�*

loss�w";���'       �	#��Xc�A�*

loss���;����       �	Z٨�Xc�A�*

loss/u�;���Q       �	"é�Xc�A�*

loss�F�=1�DK       �	s���Xc�A�*

lossX�<��m,       �	�'��Xc�A�*

lossϟ�=��^Q       �	b/��Xc�A�*

loss�#=��,�       �	�c��Xc�A�*

loss�<>z�        �	J��Xc�A�*

loss6)=p�v       �	�ͮ�Xc�A�*

loss�d=<��       �	���Xc�A�*

loss�^�<u�a       �	����Xc�A�*

loss��<��-�       �	)z��Xc�A�*

loss}��;-j��       �	�4��Xc�A�*

loss��6=�0\       �	�Xc�A�*

loss�F<|
�L       �	����Xc�A�*

lossM'�=���z       �	_��Xc�A�*

loss3�p;��       �	���Xc�A�*

loss03=��       �	µ�Xc�A�*

lossf��;�Y]       �	�d��Xc�A�*

lossq�D<ۏ�0       �	���Xc�A�*

loss��<"�!       �	؟��Xc�A�*

losse��<J,C        �	�=��Xc�A�*

loss�^<^�u�       �	'۸�Xc�A�*

lossO
<t���       �	$|��Xc�A�*

loss���=z1�       �	��Xc�A�*

loss�5/=�C�       �	`���Xc�A�*

loss#��<T��       �	�G��Xc�A�*

loss�z=�$/       �	��Xc�A�*

loss�6�<է�       �	����Xc�A�*

losst�;�nux       �	�n��Xc�A�*

lossQ��<���       �	�ҿ�Xc�A�*

loss�u/<�4�       �	*s��Xc�A�*

lossj7�<Y�2�       �	���Xc�A�*

lossì�;�הH       �	���Xc�A�*

lossDJ�<vkc       �	@�Xc�A�*

loss*ނ<���       �	V)ĐXc�A�*

loss��E;ak��       �	��ĐXc�A�*

loss���;=�)�       �	3�ŐXc�A�*

loss1�<sP��       �	RǐXc�A�*

loss�"]=���:       �	öǐXc�A�*

loss��<7JD7       �	�SȐXc�A�*

lossl�W;~��N       �	)ɐXc�A�*

loss��=sѷ       �	�ɐXc�A�*

lossT��=�>�       �	�@ʐXc�A�*

loss�[�<u0L@       �	+�ʐXc�A�*

loss��,;]
�       �	uwːXc�A�*

loss��{=!w�       �	�̐Xc�A�*

loss|{<�       �	J�̐Xc�A�*

loss�=q\/�       �	8k͐Xc�A�*

lossw��:P)�       �	[ΐXc�A�*

lossE	�;f5��       �	��ΐXc�A�*

loss��]<��\       �	4dϐXc�A�*

loss;�<��C�       �	lАXc�A�*

loss�a�<;[��       �	��АXc�A�*

lossObD=�jg�       �	zPѐXc�A�*

loss�v<B�;�       �	��ѐXc�A�*

loss�<-�_�       �	$�ҐXc�A�*

loss�T+:�
��       �	K>ӐXc�A�*

loss	�X<�8E�       �	H�ӐXc�A�*

loss�;נM	       �	�ԐXc�A�*

lossh��;d,F�       �	A,ՐXc�A�*

loss�z3<-�@n       �	E�ՐXc�A�*

loss��;J��       �	�w֐Xc�A�*

lossϸ�<�Y�       �	�!אXc�A�*

loss� =�*)�       �	K�אXc�A�*

loss=���        �	rؐXc�A�*

loss�l<�g�       �	YِXc�A�*

loss�.<#\��       �	�ِXc�A�*

loss���;��48       �	�dڐXc�A�*

loss��;�1       �	�ېXc�A�*

loss�BO=]0�
       �	��ېXc�A�*

lossh0�<:�٪       �	�TܐXc�A�*

lossD>C=����       �	�ܐXc�A�*

loss���;�Ku       �	�ݐXc�A�*

lossƗ�<��m       �	AސXc�A�*

lossO�=���       �	��ސXc�A�*

lossR@�;�P       �	�ߐXc�A�*

loss|$�;�{@.       �	c)��Xc�A�*

loss��7;���       �	~���Xc�A�*

loss��<χ�i       �	�h�Xc�A�*

loss�V<50��       �	[�Xc�A�*

loss;�~<α       �	��Xc�A�*

loss{�<��ү       �	8K�Xc�A�*

lossi^�:�<�M       �	���Xc�A�*

loss�I=��L       �	���Xc�A�*

loss�K�<���w       �	�!�Xc�A�*

loss��<�Q       �	6"�Xc�A�*

loss�w�;�9�        �	r�Xc�A�*

loss?�<gc>       �	�P�Xc�A�*

loss��<���(       �	��Xc�A�*

lossu
;�ճK       �	���Xc�A�*

loss!��;�T�       �	��Xc�A�*

lossr<W1�"       �	k-�Xc�A�*

loss�\7<x�9       �	K��Xc�A�*

loss�^<8�!�       �	fj�Xc�A�*

lossw=��HL       �	W�Xc�A�*

lossl�<M�@D       �	@��Xc�A�*

loss�&<�&�       �	`x�Xc�A�*

lossO`<�yq       �	@�Xc�A�*

loss*�s<�Z��       �	��Xc�A�*

loss�&�<��       �	�f�Xc�A�*

loss��'=�{��       �	��Xc�A�*

loss8'=��p       �	���Xc�A�*

loss�)�<�#�       �	IL�Xc�A�*

loss2�<W��K       �	��Xc�A�*

loss�[<��f�       �	Y��Xc�A�*

loss}�<����       �	$_��Xc�A�*

loss)�@=���       �	���Xc�A�*

loss�>�;��S       �	-���Xc�A�*

loss(�<�L�       �	�L��Xc�A�*

loss���<(BX       �	I���Xc�A�*

loss���=�&	       �	����Xc�A�*

loss�/;r�7G       �	%A��Xc�A�*

loss-*�:����       �	���Xc�A�*

loss�M<�uP       �	N{��Xc�A�*

loss���;�*�T       �	\:��Xc�A�*

lossU��=
��       �	����Xc�A�*

lossH��;��T       �	a���Xc�A�*

loss�<����       �	�+��Xc�A�*

loss_�;�Hn/       �	?5��Xc�A�*

loss<�<�I       �	1���Xc�A�*

loss�j�;�2       �	W���Xc�A�*

lossEc�:gȐX       �	�V��Xc�A�*

loss���;h_T       �	����Xc�A�*

loss���=�;?       �	ˡ �Xc�A�*

loss*<���       �	�N�Xc�A�*

loss���<�qK       �	���Xc�A�*

loss�n;�VF�       �	���Xc�A�*

loss��'<9s��       �	t�Xc�A�*

loss�[K;|�~       �	��Xc�A�*

loss���;ct�j       �	���Xc�A�*

loss��=$�ķ       �	�]�Xc�A�*

loss��`<�/�M       �	]��Xc�A�*

loss��}<Ԡk       �	c��Xc�A�*

loss��;��&�       �	G�Xc�A�*

lossڄ<;�t�       �	���Xc�A�*

lossRF=��d�       �	��	�Xc�A�*

loss6o�9�6s       �	w1
�Xc�A�*

loss,�=���       �	)�
�Xc�A�*

lossf�b<,7��       �	/j�Xc�A�*

loss�HQ= tr       �	�a�Xc�A�*

loss F�=:��a       �	���Xc�A�*

loss\�`=s���       �	e��Xc�A�*

lossF_;�^�       �	{��Xc�A�*

losss*�:�b~,       �	�(�Xc�A�*

loss*�J=��E+       �	���Xc�A�*

loss�hD<�i��       �	�e�Xc�A�*

lossF�A<�n        �	q�Xc�A�*

loss���<���=       �	u��Xc�A�*

loss��;�       �	�J�Xc�A�*

loss�[�<B��       �	((�Xc�A�*

lossH��<��#/       �	���Xc�A�*

lossC�X<�ѱ       �	Tq�Xc�A�*

lossj�<�P�       �	W�Xc�A�*

loss;�`<>��\       �	���Xc�A�*

lossL=<�       �	��Xc�A�*

loss�=��Q       �	g�Xc�A� *

loss�e?=n��       �	d�Xc�A� *

loss��;٢��       �	��Xc�A� *

loss\{='��       �	?�Xc�A� *

losss~�=!(�       �	��Xc�A� *

loss:�=�       �	Xu�Xc�A� *

loss��6<v��)       �	s�Xc�A� *

lossw�<���E       �	i��Xc�A� *

loss�&(=�T�c       �	^G�Xc�A� *

lossa<��M]       �	~��Xc�A� *

loss��;�>�       �	$��Xc�A� *

loss;,�<��*�       �	�&�Xc�A� *

loss�{;���       �	���Xc�A� *

loss�}�<�ҟ�       �	Nd�Xc�A� *

loss��<g�H�       �	���Xc�A� *

loss&�;F��_       �	� �Xc�A� *

loss=��<�2<�       �	<.!�Xc�A� *

lossH��;_��       �	��!�Xc�A� *

loss�v=��l�       �	z"�Xc�A� *

loss�� <�V�       �	7#�Xc�A� *

loss�=hW       �	 �#�Xc�A� *

losssZ�<V�t�       �	R$�Xc�A� *

loss��k<4��       �	��$�Xc�A� *

loss8:�;�:e       �	V�%�Xc�A� *

lossʔ�<�H�       �	n4&�Xc�A� *

loss4n�;�;a�       �	4h'�Xc�A� *

loss-�r<��X       �	�)(�Xc�A� *

loss=e�:�$}&       �	u�(�Xc�A� *

lossz��:��8J       �	�])�Xc�A� *

lossC*�<��%H       �	m�)�Xc�A� *

loss�p�<��V�       �	*�*�Xc�A� *

loss���<�xs�       �	M+�Xc�A� *

lossƍ!=غ�       �	�+�Xc�A� *

loss��<F�o�       �	e�,�Xc�A� *

lossJ��<��o�       �	+l-�Xc�A� *

loss�<�<��AY       �	�.�Xc�A� *

loss�9�<� �&       �	
�.�Xc�A� *

loss�Y+:�P�T       �	�5/�Xc�A� *

loss�ٸ:g�-       �	��/�Xc�A� *

loss�;�ӊc       �	*p0�Xc�A� *

lossd��:�p��       �	|1�Xc�A� *

loss@�Y<}�B�       �	��1�Xc�A� *

loss=��;Oz�       �	~W2�Xc�A� *

loss��9ɋև       �	�53�Xc�A� *

loss���:�A݃       �	6�3�Xc�A� *

loss|�/=�ԒH       �	kf4�Xc�A� *

loss�}�:j�ֺ       �	a�4�Xc�A� *

lossJa;W�3#       �	 �5�Xc�A� *

loss�P>:_���       �	�66�Xc�A� *

lossӯ�;ÜP�       �	_�6�Xc�A� *

loss��X<l���       �	�l7�Xc�A� *

loss	G9:c�<.       �	O8�Xc�A� *

loss
-
=- u�       �	��8�Xc�A� *

lossND>����       �	�79�Xc�A� *

lossC�T;�,��       �	E�9�Xc�A� *

losss�=��       �	2t:�Xc�A� *

loss�i:= �~�       �	�;�Xc�A� *

lossN�c<==i�       �	 �;�Xc�A� *

loss�\<MKm       �	><�Xc�A� *

loss�t�<��T       �	��<�Xc�A� *

lossI-�=����       �	Xp=�Xc�A� *

lossna�=�JH�       �	O>�Xc�A� *

loss��;N>�       �	�>�Xc�A� *

loss�0�;A���       �	iT?�Xc�A� *

lossgą:2�'{       �	��?�Xc�A� *

loss�{=*qk       �	�@�Xc�A� *

loss�zN=���       �	gGA�Xc�A� *

loss�
;����       �	��A�Xc�A� *

loss$݆=z
       �	"�B�Xc�A� *

loss�k-=�6.-       �	��C�Xc�A� *

loss��=A.�       �	�6D�Xc�A� *

lossn);���       �	�D�Xc�A� *

lossHu&=j�g[       �	.qE�Xc�A� *

loss���;C��N       �	�0F�Xc�A� *

loss��<�K
9       �	O�F�Xc�A� *

loss��<��I�       �	�`G�Xc�A� *

loss:�<���       �	w�G�Xc�A� *

lossq)�:
�S�       �	ɒH�Xc�A� *

loss�q�<y:�m       �	4,I�Xc�A� *

loss
�; ��       �	��I�Xc�A� *

loss��<�E?       �	\XJ�Xc�A� *

lossd��<3���       �	A�J�Xc�A� *

lossA��<hѐ�       �	j�K�Xc�A� *

loss�[�<>���       �	�(L�Xc�A� *

loss?Å<�V       �	��L�Xc�A� *

lossܻ*<����       �	jM�Xc�A� *

loss:s;��m�       �	��M�Xc�A� *

loss�1;Wc��       �	ԘN�Xc�A� *

lossz3�<8~�)       �	�8O�Xc�A� *

loss��;"�)h       �	��O�Xc�A� *

loss� �<~���       �	amP�Xc�A� *

loss�K�;D�ޓ       �	�Q�Xc�A� *

loss�=�Zr�       �	��Q�Xc�A� *

loss��;0�q       �	4JR�Xc�A� *

loss��:��A�       �	��R�Xc�A� *

loss�m�;-k�O       �	=�S�Xc�A� *

losstS�;�mV�       �	�T�Xc�A� *

lossd�<u<�       �	��T�Xc�A� *

loss���;RW(       �	SxU�Xc�A� *

lossT˷<�q�       �	=V�Xc�A� *

lossd!�<]���       �	P�V�Xc�A� *

loss�g�;�T�       �	AW�Xc�A� *

loss�5�<�j        �	��W�Xc�A� *

lossJ��;~��       �	qX�Xc�A� *

loss��;�ZF�       �	o+Y�Xc�A� *

loss᎕<�Ju       �	T�t�Xc�A� *

loss�
V<�:       �	4du�Xc�A� *

lossl�<��       �	��u�Xc�A� *

loss��<�       �	��v�Xc�A� *

loss�]6=9 �       �	s0w�Xc�A� *

loss�a�<�?       �	��w�Xc�A� *

lossS�=�
W:       �	�wx�Xc�A� *

lossOܳ<���;       �	�y�Xc�A� *

loss�9=O}ϱ       �	��y�Xc�A� *

lossX�;�k(�       �	�Bz�Xc�A� *

lossC�<%�S�       �	` {�Xc�A� *

loss}r=O-�^       �	�{�Xc�A� *

lossK�=,��       �	x)|�Xc�A� *

loss �5<s'/       �	׿|�Xc�A� *

loss�pG<U|2�       �	9�}�Xc�A� *

loss�)7=�H��       �	j�~�Xc�A� *

lossM�9�8l2       �	@�Xc�A� *

loss��5<�W��       �	1��Xc�A� *

loss\�<�A       �	˅��Xc�A� *

loss��1={��       �	} ��Xc�A� *

lossX:�<wK6r       �	1΁�Xc�A� *

loss���=/;�9       �	�{��Xc�A� *

loss貈;�}�       �	3��Xc�A� *

loss�w�<�Q��       �	����Xc�A�!*

loss��<���       �	�[��Xc�A�!*

loss�k(=�oO       �	����Xc�A�!*

loss`/k<{���       �	m���Xc�A�!*

loss6��;=��       �	$��Xc�A�!*

loss��<�q��       �	����Xc�A�!*

loss�I�<�GB       �	|a��Xc�A�!*

loss3�<��w       �	e��Xc�A�!*

lossr@<��>l       �	���Xc�A�!*

loss�1\<�q�       �	'���Xc�A�!*

lossc9=֗�Z       �	+/��Xc�A�!*

loss�]�;T��U       �	qʊ�Xc�A�!*

loss�=�Ω       �	Bv��Xc�A�!*

losswF;�       �	���Xc�A�!*

loss�<=G<{�       �	tό�Xc�A�!*

loss�׏=El�}       �	ir��Xc�A�!*

loss*��<����       �	�1��Xc�A�!*

loss�	�<m�k       �	5ӎ�Xc�A�!*

lossC��;^@�       �	�n��Xc�A�!*

lossqߋ<F���       �	���Xc�A�!*

lossI�<��       �	;���Xc�A�!*

lossv�'=���s       �	�E��Xc�A�!*

lossW��=T���       �	O呑Xc�A�!*

loss��1=�y�}       �	舒�Xc�A�!*

lossXM<�A�       �	2<��Xc�A�!*

lossN�;���g       �	ӓ�Xc�A�!*

loss�F�9����       �	2r��Xc�A�!*

losser�:1���       �	F
��Xc�A�!*

loss�:%=���       �	���Xc�A�!*

loss@~�:w�M}       �	�Q��Xc�A�!*

loss�!�=��Pg       �	  ��Xc�A�!*

loss���<�/7       �	b���Xc�A�!*

lossS{�:�:�       �	K���Xc�A�!*

loss?�@;Y�       �	�5��Xc�A�!*

loss�}�:����       �	�י�Xc�A�!*

lossv�L<�͹q       �	���Xc�A�!*

loss,^�=15�3       �	6���Xc�A�!*

lossf�f=��a9       �	�A��Xc�A�!*

loss��$;o�;\       �	8ל�Xc�A�!*

loss���;Ԩ@�       �	nn��Xc�A�!*

loss跆<�	w)       �	u��Xc�A�!*

lossʶ�<fY�?       �	����Xc�A�!*

loss�F;�z�       �	S��Xc�A�!*

loss(+�<�jyg       �	�Xc�A�!*

loss���< x_�       �	a���Xc�A�!*

loss��=��*       �	�U��Xc�A�!*

loss��O<����       �	�롑Xc�A�!*

loss.nV=���Y       �	����Xc�A�!*

loss1�E=ơ�I       �	U��Xc�A�!*

loss�h<��E�       �	�Xc�A�!*

loss�0=Pr��       �	����Xc�A�!*

loss���<		�       �	���Xc�A�!*

loss�(=�        �	����Xc�A�!*

loss_�;n�PU       �	F[��Xc�A�!*

loss��;���       �	����Xc�A�!*

losss$B=!0Bc       �	�V��Xc�A�!*

lossj�<�yn�       �	��Xc�A�!*

loss�'�<��       �	R���Xc�A�!*

loss�N=��(�       �	�T��Xc�A�!*

loss�Q�<��%H       �	�쪑Xc�A�!*

loss3�o:� ��       �	겫�Xc�A�!*

loss�8�<�c�       �	�L��Xc�A�!*

loss?��:A��       �	1ꬑXc�A�!*

loss{�;Y���       �	uɭ�Xc�A�!*

loss[O
<ɒ �       �	�e��Xc�A�!*

loss!�<I�z       �	k��Xc�A�!*

loss�q<�df       �	����Xc�A�!*

loss��G<��n�       �	�8��Xc�A�!*

lossH�7<� �       �	*ⰑXc�A�!*

loss	��<��ru       �	�w��Xc�A�!*

loss =�S�       �	�6��Xc�A�!*

loss�2<GWk�       �	Բ�Xc�A�!*

loss�Wa<��j�       �	?p��Xc�A�!*

lossfB�=fRP       �	�
��Xc�A�!*

loss
�<���&       �	����Xc�A�!*

loss�K�=���       �	F���Xc�A�!*

loss���<���       �	w/��Xc�A�!*

lossQz;�.�       �	XŶ�Xc�A�!*

loss�<;)n8\       �	�]��Xc�A�!*

loss�<S��g       �	o���Xc�A�!*

loss6��<P���       �	C���Xc�A�!*

loss�]�<|@��       �	 C��Xc�A�!*

loss\�==H<��       �	�ṑXc�A�!*

loss.�;�       �	�w��Xc�A�!*

loss�%':�R       �	/��Xc�A�!*

loss��=�29;       �	w���Xc�A�!*

loss}�p<7��       �	�U��Xc�A�!*

loss��<��Z       �	�꼑Xc�A�!*

loss�Q�<p���       �	^���Xc�A�!*

loss��=�a�       �	n��Xc�A�!*

loss<A�;2�~       �	����Xc�A�!*

loss1��<���       �	�S��Xc�A�!*

loss�"<�0p        �	�쿑Xc�A�!*

loss1f�=�MX�       �	?��Xc�A�!*

lossZ�C=����       �	���Xc�A�!*

loss��0<��
       �	��Xc�A�!*

loss�7";ʊ&       �	�;ÑXc�A�!*

loss3!�<��[�       �	K�ÑXc�A�!*

loss|<�<w``       �	g}đXc�A�!*

loss.��<<{-       �	őXc�A�!*

loss�9<�k�       �	��őXc�A�!*

loss���:jf       �	uUƑXc�A�!*

loss�L�<ę�       �	�ƑXc�A�!*

loss��I=1�J       �	��ǑXc�A�!*

loss��!<.�J       �	(+ȑXc�A�!*

loss��=C"4       �	~�ȑXc�A�!*

lossO�2<��D       �	�eɑXc�A�!*

lossD�}<F��       �	M�ɑXc�A�!*

loss�<n���       �	�ʑXc�A�!*

loss��=D���       �	I.ˑXc�A�!*

loss�e�<����       �	��ˑXc�A�!*

loss��%=��g       �	[{̑Xc�A�!*

loss�A@<�ty       �	+͑Xc�A�!*

loss��M=�Bi�       �	 �͑Xc�A�!*

lossz��;�YN�       �	�[ΑXc�A�!*

loss���;D�J       �	E�ΑXc�A�!*

loss_Cu<S�&O       �	��ϑXc�A�!*

loss���;��g`       �	�"БXc�A�!*

loss�h�;�       �	��БXc�A�!*

lossx��<�{��       �	��ёXc�A�!*

loss�'=8�ld       �	�1ґXc�A�!*

loss4	<��{�       �	}�ґXc�A�!*

loss�A<ۃR       �	�kӑXc�A�!*

loss;[�:�       �	�ԑXc�A�!*

loss��{;zͥ       �	��ԑXc�A�!*

loss��2<��;�       �	]PՑXc�A�!*

loss�.�<�b.�       �	��ՑXc�A�!*

losst4m;~T��       �	�֑Xc�A�!*

loss�~c<:"�       �	F]בXc�A�"*

loss�=鋑�       �	+�בXc�A�"*

loss�c<�@/�       �	6�ؑXc�A�"*

loss}��;�2�g       �	f0ّXc�A�"*

lossZfT<�GTs       �	��ّXc�A�"*

loss�,�;� �~       �	�kڑXc�A�"*

loss��i<�5�       �	ۑXc�A�"*

lossT3�<��?       �	3�ۑXc�A�"*

loss�=�6�       �	�MݑXc�A�"*

loss�YW=���       �	��ݑXc�A�"*

loss.�+<6~�       �	��ޑXc�A�"*

loss��<_q�+       �	5ߑXc�A�"*

loss���9t�0�       �	p�ߑXc�A�"*

loss�'=P���       �	�z��Xc�A�"*

lossl�<@]�&       �	��Xc�A�"*

lossR��;��h       �	���Xc�A�"*

loss�ju='�~       �	�>�Xc�A�"*

lossё�;,s��       �	���Xc�A�"*

loss*/=�w/�       �	<l�Xc�A�"*

loss�.=��"�       �	�7�Xc�A�"*

loss�q�<O�(-       �	o��Xc�A�"*

lossbZ<�W��       �	qt�Xc�A�"*

loss�d7;���M       �	R�Xc�A�"*

loss��S:O>�       �	{��Xc�A�"*

lossI��<tz��       �	1{�Xc�A�"*

loss��=p?��       �	u>�Xc�A�"*

loss|T<\�ι       �	���Xc�A�"*

lossTi=zO       �	���Xc�A�"*

loss���<���       �	"�Xc�A�"*

loss�G:Bmi�       �	��Xc�A�"*

loss��<�Hs�       �	ga�Xc�A�"*

loss��4<~�[       �	�Xc�A�"*

loss��<�Ӷ       �	J��Xc�A�"*

loss;��;9ֶ�       �	?5�Xc�A�"*

loss�i&;�us�       �	��Xc�A�"*

loss���=���g       �	)��Xc�A�"*

lossV=�3       �	�R�Xc�A�"*

lossݸ�;ϑ~�       �	���Xc�A�"*

loss4Wq=Ug�       �	���Xc�A�"*

loss]Y!=�޿       �	N+�Xc�A�"*

loss|��;�#j       �	q��Xc�A�"*

loss��;�_NF       �	�d�Xc�A�"*

loss�<���       �	��Xc�A�"*

lossjǐ<�\�       �	���Xc�A�"*

losss��;�C�~       �	G=��Xc�A�"*

lossD��:a�"e       �	b���Xc�A�"*

loss=s=Y�:^       �	�t��Xc�A�"*

loss���<�-�       �	W
��Xc�A�"*

loss�3�;�B       �	D���Xc�A�"*

loss#=e�j�       �	L��Xc�A�"*

loss`_B;L֊�       �	����Xc�A�"*

loss
��<-w+i       �	����Xc�A�"*

loss�sc<_�ɞ       �	a8��Xc�A�"*

loss��y<���       �	���Xc�A�"*

loss�N<��J       �	�r��Xc�A�"*

loss�H=���       �	���Xc�A�"*

lossx)�<ղ�	       �	!���Xc�A�"*

loss��<�M9�       �	
M��Xc�A�"*

loss�aI<��2g       �	V���Xc�A�"*

loss�c�;�I;E       �	j���Xc�A�"*

losse�=���a       �	�&��Xc�A�"*

loss&':K_�|       �	����Xc�A�"*

lossJ�%<[oO       �	e��Xc�A�"*

loss �h=���       �	  �Xc�A�"*

loss�.=fk�       �	�� �Xc�A�"*

loss�=�~��       �	�2�Xc�A�"*

loss�N<�E�{       �	���Xc�A�"*

loss4
=���       �	 t�Xc�A�"*

loss׌�<5�.�       �	)�Xc�A�"*

loss��d<0���       �	���Xc�A�"*

loss�5�;JeS       �	pA�Xc�A�"*

lossc$�;q��       �	���Xc�A�"*

lossq�<}ӍG       �	�p�Xc�A�"*

loss$�Z<��w       �	��Xc�A�"*

loss�Ɨ<�7�       �	��Xc�A�"*

loss�$=R� $       �	@j�Xc�A�"*

loss:<��ע       �	T �Xc�A�"*

loss���<�)�$       �	8��Xc�A�"*

lossܚ�;kA�       �	Q�	�Xc�A�"*

loss��f<����       �	/P
�Xc�A�"*

loss,�<;�WXY       �	�Xc�A�"*

loss,b<�x\�       �	*��Xc�A�"*

loss�~#<"��       �	�M�Xc�A�"*

loss���<57�       �	���Xc�A�"*

loss�)�<��.�       �	���Xc�A�"*

loss[�=X�l       �	EI�Xc�A�"*

loss8%�=�?*K       �	���Xc�A�"*

loss�6q<^'e       �	~��Xc�A�"*

loss�S�<;]u       �	4�Xc�A�"*

loss/+;K��       �	��Xc�A�"*

lossS՗;�!��       �	�w�Xc�A�"*

loss3�A;�X��       �	��Xc�A�"*

loss�<��       �	��Xc�A�"*

loss9?<ܥ�b       �	O\�Xc�A�"*

loss��<<%V�       �	��Xc�A�"*

loss�4<�:Z�       �	и�Xc�A�"*

loss��<7P.5       �	S^�Xc�A�"*

loss�k�<�UZx       �	�
�Xc�A�"*

loss�a�<���       �	"��Xc�A�"*

lossTy=�k��       �	�?�Xc�A�"*

lossi <`h�       �	D��Xc�A�"*

loss�>=�v�       �	M��Xc�A�"*

loss�i:z�       �	B%�Xc�A�"*

loss�A,:��Eg       �	��Xc�A�"*

losszM:;F�       �	�\�Xc�A�"*

lossT	t<�u��       �	z��Xc�A�"*

loss�+A<�ҳ       �	6��Xc�A�"*

loss��B=h�B       �	�7�Xc�A�"*

loss�`}<>��       �	��Xc�A�"*

loss��y<��}       �	�h�Xc�A�"*

loss��&<����       �	��Xc�A�"*

lossRL�<��       �	���Xc�A�"*

loss,&=��݀       �	u;�Xc�A�"*

lossL�=�r+F       �	��Xc�A�"*

loss�)5;J��       �	�n �Xc�A�"*

lossc�<x��       �	F@!�Xc�A�"*

loss]��<Iu*�       �	U�!�Xc�A�"*

lossQ~�; /͚       �	�p"�Xc�A�"*

loss�y1==�       �	_#�Xc�A�"*

loss(=�=�       �	�#�Xc�A�"*

loss�kH<�eL       �	�5$�Xc�A�"*

loss���<�ó�       �	5�$�Xc�A�"*

loss�<��7�       �	d%�Xc�A�"*

loss�=�=��       �	��%�Xc�A�"*

lossX��<���\       �	��&�Xc�A�"*

loss��<Z�E0       �	�'�Xc�A�"*

loss��x;�"Y       �	��(�Xc�A�"*

loss(\<L       �	�P)�Xc�A�"*

loss��<�e��       �	j*�Xc�A�#*

loss7-�;����       �	�+�Xc�A�#*

lossʥ�<E�QB       �	�+�Xc�A�#*

loss@��;��z       �	�<,�Xc�A�#*

loss[��<k�&�       �	��,�Xc�A�#*

lossV�4=�N��       �	�q-�Xc�A�#*

loss�}U=U��       �	.�Xc�A�#*

loss�%�<bYݷ       �	k�.�Xc�A�#*

loss�<�;���       �	�]/�Xc�A�#*

loss��;t&�       �	�0�Xc�A�#*

loss/��;�|��       �	ӡ0�Xc�A�#*

loss�h�:����       �	�F1�Xc�A�#*

loss��=����       �	��1�Xc�A�#*

loss
#�<{��       �	��2�Xc�A�#*

loss��=� m       �	(3�Xc�A�#*

loss�=�<^�n       �	��3�Xc�A�#*

lossI�Q<7i��       �	b4�Xc�A�#*

loss��c<Nf��       �	��4�Xc�A�#*

loss ��;gbg       �	��5�Xc�A�#*

loss[Y�<c�K�       �	�06�Xc�A�#*

loss�K={�       �	z�6�Xc�A�#*

loss�X=H6N7       �	�[7�Xc�A�#*

loss�<6\�       �	d8�Xc�A�#*

lossd��<x��F       �	p�8�Xc�A�#*

lossS��=zs�A       �	�J9�Xc�A�#*

loss�@�<&��       �	z�9�Xc�A�#*

loss�$=���	       �	�:�Xc�A�#*

lossz�";��I       �	5%;�Xc�A�#*

loss�H�;-
?       �	��;�Xc�A�#*

lossd,<�a       �	�X<�Xc�A�#*

loss�"<k��       �	i=�Xc�A�#*

loss7wi<��lc       �	=>�Xc�A�#*

loss�Ņ;&y3       �	��>�Xc�A�#*

loss�e�<~��       �	x?�Xc�A�#*

loss�op;���       �	�@�Xc�A�#*

loss�3�;PR]       �	��@�Xc�A�#*

loss��i=��B�       �	�uA�Xc�A�#*

loss�{�;��t�       �	B�Xc�A�#*

lossOm�;�e@       �	ƿB�Xc�A�#*

loss*6�<��TP       �	}YC�Xc�A�#*

loss��v=u�       �	b/D�Xc�A�#*

lossZ��<�\��       �	7�D�Xc�A�#*

loss��K=}M��       �	҉E�Xc�A�#*

loss�Z�<�F��       �	�F�Xc�A�#*

loss�O�:�ʏ�       �	QOG�Xc�A�#*

loss�30=��#|       �	[�G�Xc�A�#*

loss%�<ܪv3       �	��H�Xc�A�#*

loss�T*<L�2>       �	7�I�Xc�A�#*

lossL��<���b       �	�<J�Xc�A�#*

loss���<��I�       �	m�J�Xc�A�#*

loss�<�.       �	�K�Xc�A�#*

loss��K<bU�       �	�&L�Xc�A�#*

loss��<�AS       �	��L�Xc�A�#*

losscµ;���       �	/jM�Xc�A�#*

loss�;���-       �	TN�Xc�A�#*

lossVt<��1�       �	��N�Xc�A�#*

loss��.:c9u       �	�9O�Xc�A�#*

lossᎵ<=��-       �	k�O�Xc�A�#*

loss��;`��       �	rP�Xc�A�#*

loss{,�<>���       �	Q�Xc�A�#*

loss�H:�69[       �	��Q�Xc�A�#*

loss��s:Bȃ�       �	�RR�Xc�A�#*

loss8Zu;D�       �	��R�Xc�A�#*

lossl��;�zӀ       �	ǄS�Xc�A�#*

loss=<�y�9       �	�T�Xc�A�#*

lossZ2�=M!�<       �	�T�Xc�A�#*

loss��=�r�       �	�OU�Xc�A�#*

loss�v<&��       �	�%V�Xc�A�#*

loss!R=��)�       �	+�V�Xc�A�#*

loss��'<�?�       �	�SW�Xc�A�#*

loss2�M;����       �	��W�Xc�A�#*

loss���:�sf�       �	؛X�Xc�A�#*

loss&4l<�%�       �	�4Y�Xc�A�#*

lossJ�=6���       �	�Z�Xc�A�#*

lossWhs<��x�       �	��Z�Xc�A�#*

loss���<���E       �	;6[�Xc�A�#*

loss�hu;��8(       �	�[�Xc�A�#*

loss�q�;��       �	�j\�Xc�A�#*

lossI�;��>       �	u]�Xc�A�#*

loss���;r�ȏ       �	��]�Xc�A�#*

lossrA=�hb�       �	�R^�Xc�A�#*

lossq x<f��       �	�_�Xc�A�#*

loss��=V�١       �	ɬ_�Xc�A�#*

loss�%=K�O�       �	�N`�Xc�A�#*

loss���<)�n3       �	��`�Xc�A�#*

loss�ZM<����       �	w�a�Xc�A�#*

loss4=<�hx�       �	�Jb�Xc�A�#*

losst�+=¹�M       �	��b�Xc�A�#*

loss�l�<�9*       �	��c�Xc�A�#*

lossQ��<i+��       �	5Dd�Xc�A�#*

lossĢ<�AU       �	_�d�Xc�A�#*

loss���<W�ZL       �	̚e�Xc�A�#*

loss�P;�9(.       �	>f�Xc�A�#*

loss,��<�U.�       �	��f�Xc�A�#*

lossm��<�       �	1�g�Xc�A�#*

loss�M�=W�r�       �	.�h�Xc�A�#*

loss8��<ym�       �	�.i�Xc�A�#*

loss�i�<�	�(       �	t�i�Xc�A�#*

loss^=]:m�       �	uj�Xc�A�#*

loss��=X�6       �	;k�Xc�A�#*

loss�=y�ٲ       �	��k�Xc�A�#*

lossEb9;�8z�       �	�il�Xc�A�#*

loss��1=��a�       �	;m�Xc�A�#*

loss�"�=R�ӯ       �	��m�Xc�A�#*

loss�<�a7       �	?sn�Xc�A�#*

loss� �:���       �	$o�Xc�A�#*

lossp�<p��       �	�o�Xc�A�#*

loss I<�t��       �	?p�Xc�A�#*

loss�<J���       �	��p�Xc�A�#*

loss�Z;ѿ?       �	��q�Xc�A�#*

lossc');��5c       �	�,r�Xc�A�#*

loss��<vn h       �	��r�Xc�A�#*

lossXAY;{ �       �	�is�Xc�A�#*

loss��;�"�}       �	Ot�Xc�A�#*

loss�)=�!�       �	�t�Xc�A�#*

loss�G�<�=��       �	Zu�Xc�A�#*

loss��;)�87       �	��u�Xc�A�#*

lossV�K:���=       �	�v�Xc�A�#*

loss� <�^p�       �		4w�Xc�A�#*

loss�<��8       �	��w�Xc�A�#*

loss�Wj=5E,�       �	�qx�Xc�A�#*

loss淂<)@�(       �	gy�Xc�A�#*

loss!��;���       �	��y�Xc�A�#*

loss�M=���P       �	Iz�Xc�A�#*

loss���<����       �	W�z�Xc�A�#*

loss�_=�G�-       �	a�{�Xc�A�#*

loss�_�;�s��       �	�B|�Xc�A�#*

loss�
=�8@�       �	�|�Xc�A�#*

loss���;���S       �	��}�Xc�A�$*

lossx��<uN��       �	0~�Xc�A�$*

lossI��<�J�Q       �	�~�Xc�A�$*

lossHyt<���Z       �	��Xc�A�$*

loss�,�=j��       �	�S��Xc�A�$*

loss�Y�;c�2       �	�怒Xc�A�$*

loss�=�x�       �	���Xc�A�$*

loss8�<�0U8       �	½��Xc�A�$*

loss��%<Ӥ7K       �	zU��Xc�A�$*

lossv"#;���c       �	�탒Xc�A�$*

loss��:����       �	���Xc�A�$*

loss�"<	B��       �	6��Xc�A�$*

loss	Kl<o$�       �	5���Xc�A�$*

loss�=<���       �	H���Xc�A�$*

loss��<G_��       �	T:��Xc�A�$*

loss�֡:}<��       �	=҇�Xc�A�$*

loss�%g<���#       �	n��Xc�A�$*

loss )r<��}       �	e��Xc�A�$*

loss$y0=N�g6       �	����Xc�A�$*

lossm!<���`       �	����Xc�A�$*

loss�H=穾�       �	����Xc�A�$*

loss�W<M&Y       �		���Xc�A�$*

loss[0<����       �	1]��Xc�A�$*

loss=�;cȷ�       �	����Xc�A�$*

loss}>�;�c�F       �	%���Xc�A�$*

loss*ˁ;�Q?*       �	�=��Xc�A�$*

loss]��<9.�       �	׏�Xc�A�$*

loss��F=v��       �	mo��Xc�A�$*

loss���<bxM�       �	���Xc�A�$*

loss��*<L#��       �	~���Xc�A�$*

loss1c<�2       �	rP��Xc�A�$*

lossr�<�}�       �	�Xc�A�$*

loss�7<��Ɩ       �	%���Xc�A�$*

lossI�O<��`}       �	�<��Xc�A�$*

loss3^=��).       �	���Xc�A�$*

loss�m�;=�n       �	c���Xc�A�$*

loss���=dr�Z       �	.X��Xc�A�$*

loss��'=.�@�       �	��Xc�A�$*

lossiT2<�`�       �	ǝ��Xc�A�$*

losstB>kp�       �	�2��Xc�A�$*

loss���<�(       �	2˘�Xc�A�$*

loss�?J<��/o       �	�}��Xc�A�$*

loss{�;�       �	��Xc�A�$*

loss�Q=x�G       �	>���Xc�A�$*

lossf��<�?z       �	�X��Xc�A�$*

loss%�;�g��       �	gXc�A�$*

loss�D�;_aԼ       �	����Xc�A�$*

loss//A<�"�       �	X��Xc�A�$*

loss�./=���]       �	峝�Xc�A�$*

loss@Od<E�{G       �	MK��Xc�A�$*

loss���;��Hy       �	����Xc�A�$*

lossJ�#<��o       �	t��Xc�A�$*

lossA=��v�       �	���Xc�A�$*

lossv]a;�_�       �	����Xc�A�$*

lossR�;���       �	:@��Xc�A�$*

lossT�;��k�       �	Lޡ�Xc�A�$*

loss���<��W       �	^���Xc�A�$*

loss�� </o       �	e��Xc�A�$*

loss��<��t�       �	г��Xc�A�$*

lossiU�:n��N       �	�K��Xc�A�$*

loss�O�;�k��       �	Dߤ�Xc�A�$*

loss�|#=�d�       �	2w��Xc�A�$*

loss���;���{       �	���Xc�A�$*

loss�ќ=� �       �	&���Xc�A�$*

loss�=<r���       �	�P��Xc�A�$*

lossۖ�<5t�       �	[Xc�A�$*

loss�F�;ZJy       �	���Xc�A�$*

loss���:?�#�       �	�:��Xc�A�$*

loss��.;@��       �	LR��Xc�A�$*

loss$�;RG)       �	����Xc�A�$*

loss�A�<��h�       �	�¬�Xc�A�$*

loss=x�;e��       �	����Xc�A�$*

lossM��=��ى       �	�N��Xc�A�$*

loss�?1=�	�t       �	v���Xc�A�$*

loss�_�;}v�e       �	����Xc�A�$*

loss_w�;�fn       �	Eְ�Xc�A�$*

lossIO�;R�       �	����Xc�A�$*

loss�9/=�&��       �	o+��Xc�A�$*

lossE��=�VkZ       �	�Ȳ�Xc�A�$*

lossߪ�<��}�       �	r���Xc�A�$*

lossب�<.p�       �	�ô�Xc�A�$*

loss��&<����       �	�k��Xc�A�$*

lossA8�<�EFA       �	���Xc�A�$*

loss�K�;	1mD       �	����Xc�A�$*

loss�2�;M��       �	�V��Xc�A�$*

loss���:��t       �	T���Xc�A�$*

loss���<��`1       �	� ��Xc�A�$*

loss��M<o5��       �	S˹�Xc�A�$*

loss�M�<��R�       �	Hk��Xc�A�$*

losst��<53��       �	���Xc�A�$*

loss�w<�o�       �	����Xc�A�$*

loss�3=;��       �	�p��Xc�A�$*

lossD;��       �	���Xc�A�$*

loss�D<2���       �	ܺ��Xc�A�$*

loss�W�:>�(�       �	�Ծ�Xc�A�$*

loss\4�;�u^	       �	_z��Xc�A�$*

lossdx+;ǵ��       �	7��Xc�A�$*

lossB�=�eI       �	����Xc�A�$*

loss��;Q��I       �	4���Xc�A�$*

loss4�\<�<&[       �	W&Xc�A�$*

loss��;F�oP       �	�ÒXc�A�$*

loss0A�;S��       �	��ÒXc�A�$*

loss�?�=�W�       �	�IĒXc�A�$*

loss���;�v�       �	�ĒXc�A�$*

loss�84=�0Ɏ       �	H�ŒXc�A�$*

lossj=[j�       �	�*ƒXc�A�$*

loss��<	�       �	��ƒXc�A�$*

loss$�v<7N�P       �	�fǒXc�A�$*

loss��<ư4       �	AȒXc�A�$*

loss M<@8�       �	�ɒXc�A�$*

loss�=�;��:�       �	�ɒXc�A�$*

loss�><��?S       �	t
˒Xc�A�$*

loss���</��       �	��˒Xc�A�$*

loss��;��h       �	@L̒Xc�A�$*

loss�:;�$JN       �	-]͒Xc�A�$*

lossɁ:�1B�       �	��͒Xc�A�$*

loss��;���       �	�ΒXc�A�$*

loss�a�<h�       �	�0ϒXc�A�$*

loss�/=Ӑ�:       �	]�ϒXc�A�$*

loss��<E�Տ       �	qZВXc�A�$*

loss�l�;���       �	32ђXc�A�$*

loss\l�<�0�C       �	��ђXc�A�$*

loss��;�?Z�       �	�cҒXc�A�$*

loss��:&��O       �	? ӒXc�A�$*

loss.H+=���       �	��ӒXc�A�$*

loss!�t::�T       �	�sԒXc�A�$*

loss�:��`W       �	p
ՒXc�A�$*

loss�3�<S�v       �	<�ՒXc�A�$*

loss�,;����       �	+Q֒Xc�A�%*

loss�(;�K�       �	��֒Xc�A�%*

lossx̢;Y�       �	@�גXc�A�%*

loss�<���       �	�FؒXc�A�%*

loss](j:���       �	7�ؒXc�A�%*

loss��;G�?�       �	�uْXc�A�%*

loss�`�:1�Q�       �	�
ڒXc�A�%*

loss.�G8�Z�=       �	ӢڒXc�A�%*

loss��<xP�       �	�EےXc�A�%*

loss,]�<ȟ6�       �	m�ےXc�A�%*

lossqQM<��	�       �	w�ܒXc�A�%*

loss�L�:k��L       �	�ݒXc�A�%*

loss�2�<KX*       �	��ݒXc�A�%*

lossB�<(e4       �	iQޒXc�A�%*

loss��;2���       �	R�ޒXc�A�%*

lossɉ8=�h�       �	��ߒXc�A�%*

losswµ;�1��       �	�1��Xc�A�%*

loss @<�Z|�       �	����Xc�A�%*

loss���<����       �	�j�Xc�A�%*

loss��<"�?       �	��Xc�A�%*

loss��6=a�w       �	!��Xc�A�%*

loss��<h�5�       �	)]�Xc�A�%*

loss#F�;�;r�       �	� �Xc�A�%*

lossA`�:�       �	ʩ�Xc�A�%*

loss���<���       �	H�Xc�A�%*

lossV�V=t�f       �	`��Xc�A�%*

loss�yA=�l�W       �	t}�Xc�A�%*

loss�;Y�e�       �	�Xc�A�%*

loss�s�<X�c       �	���Xc�A�%*

lossn�@==i:�       �	�X�Xc�A�%*

lossRB=;[i�       �	y�Xc�A�%*

lossJ�E<4~       �	��Xc�A�%*

loss�g�<3�F       �	���Xc�A�%*

loss�:{;��k;       �	*��Xc�A�%*

loss���<��T       �	�#�Xc�A�%*

loss�}�<���       �	���Xc�A�%*

loss=Nd��       �	z�Xc�A�%*

loss��;��        �	��Xc�A�%*

loss�<F	��       �	�-�Xc�A�%*

loss׽�;Ui��       �	�n�Xc�A�%*

loss���;�9X       �	��Xc�A�%*

lossl�;!W��       �	���Xc�A�%*

loss`Xo=�b�^       �	W��Xc�A�%*

loss��y=Q��d       �	����Xc�A�%*

lossE��<��#       �	���Xc�A�%*

loss|3S;)��       �	N��Xc�A�%*

loss{� <��E�       �	����Xc�A�%*

loss���;��E�       �	����Xc�A�%*

loss�"=�	p�       �	�E��Xc�A�%*

loss� <���       �	 ���Xc�A�%*

lossvS#<;F.�       �	c���Xc�A�%*

loss,E^<Tؽ�       �	�8��Xc�A�%*

lossx1�<l���       �	����Xc�A�%*

loss�� <��W       �	����Xc�A�%*

lossvJ�9,��i       �	+��Xc�A�%*

loss[9<��D       �	*���Xc�A�%*

lossC�b;v:"�       �	�m��Xc�A�%*

loss2J&<�n�       �	I��Xc�A�%*

lossM��;��V       �	W �Xc�A�%*

loss�a<;�m�k       �	_�Xc�A�%*

loss��=��G       �	��Xc�A�%*

loss<��2&       �	�V�Xc�A�%*

loss(��<[��       �	3��Xc�A�%*

loss�'�:�r       �	��Xc�A�%*

loss�*=��}       �	#M�Xc�A�%*

loss7�#=k2�       �	n�Xc�A�%*

loss��<�:�>       �	 �Xc�A�%*

loss2�<�[J�       �	�� �Xc�A�%*

loss/}�<�I��       �	eo!�Xc�A�%*

loss�Lu<m �L       �	�'"�Xc�A�%*

loss�-L=�ϊ       �	t�"�Xc�A�%*

loss�E�;Ot��       �	̙#�Xc�A�%*

loss�ص<ex�c       �	"O$�Xc�A�%*

loss�<	\*       �	��$�Xc�A�%*

lossl~<K!       �	L�%�Xc�A�%*

loss���<C�?       �	�Y&�Xc�A�%*

loss!;!=-p��       �	�'�Xc�A�%*

loss�j�<�EU�       �	�'�Xc�A�%*

loss�s�<�J�q       �	U�(�Xc�A�%*

loss��\< ���       �	EJ)�Xc�A�%*

loss��\;!>J�       �	��)�Xc�A�%*

loss&y�:��Ĕ       �	��*�Xc�A�%*

loss`�;�J�       �	ѓ+�Xc�A�%*

loss���;�65[       �	d@,�Xc�A�%*

loss��<�b       �	��,�Xc�A�%*

lossv�4<Y�/�       �	A�-�Xc�A�%*

loss�=�߉X       �	�K.�Xc�A�%*

losst��:��       �	��.�Xc�A�%*

loss�;7=���       �	��/�Xc�A�%*

loss1s;��:"       �	Q20�Xc�A�%*

loss�7<E%J       �	Q�0�Xc�A�%*

loss�+Q<.z��       �	ǀ1�Xc�A�%*

lossf=̪y       �	E/2�Xc�A�%*

loss.8-=��)L       �	(�2�Xc�A�%*

loss���<Ü�       �	�y3�Xc�A�%*

loss��@<��       �	�4�Xc�A�%*

loss ��;_�;       �	ȵ4�Xc�A�%*

loss��<]~�       �	�Q5�Xc�A�%*

lossv��<�۰l       �	[�5�Xc�A�%*

loss��:��G�       �	p�6�Xc�A�%*

loss{F�<�       �	�57�Xc�A�%*

loss��T;��       �	@�7�Xc�A�%*

loss��{<���t       �	�s8�Xc�A�%*

loss�m�<R�A�       �	c9�Xc�A�%*

loss}�<h�r�       �	]�9�Xc�A�%*

loss\�	=�?<�       �	O<:�Xc�A�%*

loss[ڍ<)+��       �	�;�Xc�A�%*

lossMץ:3Elu       �	%�;�Xc�A�%*

lossNW<G>��       �	ё<�Xc�A�%*

lossC{�<�*       �	�*=�Xc�A�%*

loss;��;�H�       �	"�=�Xc�A�%*

loss���<h� 4       �	�Z>�Xc�A�%*

loss�ŕ<���       �	��>�Xc�A�%*

loss3��;+.�       �	��?�Xc�A�%*

loss�9�;a;�       �	>@�Xc�A�%*

loss8��<��=       �	��@�Xc�A�%*

loss�E�<��       �	*oA�Xc�A�%*

loss�}<A&�l       �	�4B�Xc�A�%*

loss�H>��y�       �	��B�Xc�A�%*

loss�1�<FS�       �	�nC�Xc�A�%*

loss�/�;��G�       �	�D�Xc�A�%*

lossܿ�:l|>�       �	6�D�Xc�A�%*

lossƢr:����       �	>�E�Xc�A�%*

loss��H=F���       �	�uF�Xc�A�%*

loss��=�       �	F"G�Xc�A�%*

loss��&=S���       �	+�G�Xc�A�%*

lossM�E;\���       �	GZH�Xc�A�%*

lossN�;l��       �	 I�Xc�A�%*

loss��=vz�P       �	��I�Xc�A�&*

lossJ��;��(       �	�CJ�Xc�A�&*

loss�Gr:��&       �	K�Xc�A�&*

loss�
2<AL�       �	f�K�Xc�A�&*

loss��*<�!{�       �	�BL�Xc�A�&*

lossLՌ;.���       �	��L�Xc�A�&*

lossY&=_�       �	~�M�Xc�A�&*

loss�X<)H�I       �	5_N�Xc�A�&*

loss#F<t3       �	D�N�Xc�A�&*

loss�S=�m       �	�P�Xc�A�&*

loss�2m;jT�       �	��P�Xc�A�&*

lossO'<s9�`       �	�gQ�Xc�A�&*

loss���:��\^       �	�@R�Xc�A�&*

loss;��<�L-�       �	��R�Xc�A�&*

loss4^�;x�       �	1|S�Xc�A�&*

loss�xi<�3�       �	�6T�Xc�A�&*

lossz
=ᜠ�       �	��T�Xc�A�&*

lossC�<-5(�       �	KvU�Xc�A�&*

loss���<QE~�       �	eV�Xc�A�&*

loss�u�;$x&       �	��V�Xc�A�&*

loss��<����       �	XYW�Xc�A�&*

lossO�e<�8p       �	RX�Xc�A�&*

losst9�:+lO�       �	��X�Xc�A�&*

loss�@=ו�       �	�>Y�Xc�A�&*

loss��;�;є       �	�Z�Xc�A�&*

loss��<��h       �	�Z�Xc�A�&*

loss\�<��=�       �	,G[�Xc�A�&*

loss���;S���       �	K\�Xc�A�&*

loss��=�5�       �	�\�Xc�A�&*

loss��G<ff�       �	G]�Xc�A�&*

losss��<!u0       �	��]�Xc�A�&*

loss�N�<����       �	�._�Xc�A�&*

loss;��;	�#H       �	��_�Xc�A�&*

loss���<��       �	t`�Xc�A�&*

lossf��=B���       �	�a�Xc�A�&*

loss}�<iJ�\       �	��a�Xc�A�&*

loss�;�;Q�^       �	H�b�Xc�A�&*

loss�m�;�J�       �	>Zc�Xc�A�&*

lossfǢ;�ej�       �	�c�Xc�A�&*

loss<��I       �	8�d�Xc�A�&*

loss���<��%�       �	u;e�Xc�A�&*

loss&�<_��       �	3�e�Xc�A�&*

loss��<�$F�       �	W�f�Xc�A�&*

lossa��;���1       �	�Rg�Xc�A�&*

loss�,�;���       �	(�g�Xc�A�&*

loss�C�<2��       �	%!i�Xc�A�&*

loss:��;Qg��       �	0cj�Xc�A�&*

loss���<��       �	�	k�Xc�A�&*

loss)�<v�X�       �	ol�Xc�A�&*

loss8��<A$�^       �	r�l�Xc�A�&*

loss���;����       �	uWm�Xc�A�&*

loss�ř<��K2       �	�n�Xc�A�&*

loss�j:y�K       �	�n�Xc�A�&*

loss�=��c       �	�to�Xc�A�&*

loss�"�<����       �	&6p�Xc�A�&*

lossS�;Up       �	S�p�Xc�A�&*

loss��<ɯ!       �	Zgq�Xc�A�&*

lossK�<w���       �	�}r�Xc�A�&*

lossT�)<xI�<       �	�s�Xc�A�&*

loss�<�٘�       �	g�t�Xc�A�&*

loss=�<ڪ��       �	Ɔu�Xc�A�&*

loss�t�;���       �	u9v�Xc�A�&*

lossX.u=��a�       �	eqw�Xc�A�&*

lossC�Y<ے�|       �	�ox�Xc�A�&*

loss�;s�       �	�&y�Xc�A�&*

loss���<t�       �	~�y�Xc�A�&*

lossߘ�;��X       �	^�z�Xc�A�&*

loss�m�<�rD�       �	�){�Xc�A�&*

loss���;�6y       �	��{�Xc�A�&*

loss�2D<{�@(       �	6t|�Xc�A�&*

loss�wq<�Rm       �	�}�Xc�A�&*

loss�L=H�!       �	��}�Xc�A�&*

loss��L<	#�       �	#k~�Xc�A�&*

loss�f0=KP�Z       �	��Xc�A�&*

loss�'�;�O'       �	Ϻ�Xc�A�&*

loss�]M;�m       �	'݀�Xc�A�&*

loss��=Κ��       �	hy��Xc�A�&*

lossԊ�;b41�       �	s*��Xc�A�&*

loss�\=�0�T       �	�ǂ�Xc�A�&*

loss|<�
��       �	�_��Xc�A�&*

lossIm<�(�H       �	h��Xc�A�&*

loss��-<��&       �	h��Xc�A�&*

loss<��<^da       �	����Xc�A�&*

loss�T);{�       �	<M��Xc�A�&*

loss�q<[(��       �	<���Xc�A�&*

lossL�<W�1!       �	]���Xc�A�&*

loss�[=B�9       �	b��Xc�A�&*

lossH�J;�6��       �	]���Xc�A�&*

loss߾L<���       �	}��Xc�A�&*

loss��_=Z�kd       �	]���Xc�A�&*

loss�L�<jE|�       �	O>��Xc�A�&*

lossݩb=�ε]       �	M֋�Xc�A�&*

loss�-,;�`�       �	����Xc�A�&*

lossя�<1�DA       �	����Xc�A�&*

loss�<�q�       �	{3��Xc�A�&*

loss؄�<�g+       �	�Ў�Xc�A�&*

lossN?7<�}       �	rn��Xc�A�&*

loss�W�=ԋ�       �	Q��Xc�A�&*

loss�<�v4       �	����Xc�A�&*

loss܆�<
~�^       �	Ed��Xc�A�&*

loss�ˋ:R�^Z       �	���Xc�A�&*

loss��< N�       �	����Xc�A�&*

loss߮?;���g       �	�c��Xc�A�&*

loss<�;�af       �	���Xc�A�&*

loss���:Δ�&       �	����Xc�A�&*

loss�k�<��[�       �	�c��Xc�A�&*

loss��<Ҋ�       �	���Xc�A�&*

loss䵂=�ĝ�       �	CŖ�Xc�A�&*

lossh/\;�-Q       �	]o��Xc�A�&*

loss (d;OS��       �	���Xc�A�&*

loss��T=r��d       �	7Ę�Xc�A�&*

loss�{�;f��%       �	"o��Xc�A�&*

loss1^�;�v�z       �	Y��Xc�A�&*

loss�U�<:G�|       �	�ɚ�Xc�A�&*

loss���<�J2�       �	x{��Xc�A�&*

lossy<'�w�       �	Y��Xc�A�&*

loss)W=����       �	�Ü�Xc�A�&*

loss�k�:�y       �	�d��Xc�A�&*

loss���<'I�       �	���Xc�A�&*

lossa.�<��z       �	K˞�Xc�A�&*

lossT_�;1 ��       �	@���Xc�A�&*

lossDWJ<�r6U       �	D6��Xc�A�&*

loss���<N���       �	.㠓Xc�A�&*

loss�	=b��P       �	C���Xc�A�&*

loss���<|]       �	H7��Xc�A�&*

loss}��<~�!       �	�ۢ�Xc�A�&*

loss���<���s       �	V���Xc�A�&*

loss̩�==�i�       �	:$��Xc�A�&*

loss�K<l0�       �	�ä�Xc�A�'*

loss4ΐ;�6       �	�f��Xc�A�'*

loss�:<ھ�       �	�	��Xc�A�'*

losshy�:�$�       �	u���Xc�A�'*

loss��<��n       �	mS��Xc�A�'*

loss�r�;���       �	Q���Xc�A�'*

loss��<�X��       �	�+��Xc�A�'*

loss�SN=k�;       �	7é�Xc�A�'*

loss�e<df�       �	֪�Xc�A�'*

lossyi	=7�       �	n��Xc�A�'*

loss=��;���       �	1��Xc�A�'*

loss
"�;:��       �	좬�Xc�A�'*

loss�`�;A��       �	�?��Xc�A�'*

loss<�K<�.�       �	}뭓Xc�A�'*

loss�"�;"       �	����Xc�A�'*

loss�T<[��       �	H7��Xc�A�'*

loss�)Y<�*ld       �	�ޯ�Xc�A�'*

loss{U�<���       �	܀��Xc�A�'*

lossz<z���       �	W"��Xc�A�'*

lossT:�=s��y       �	ۿ��Xc�A�'*

loss���=c�X       �	 b��Xc�A�'*

loss�`?:ٌ�V       �	���Xc�A�'*

loss�s�;[��L       �	����Xc�A�'*

loss��<��!�       �	�E��Xc�A�'*

loss�_<"+�b       �	�ⴓXc�A�'*

loss�>1=f̀�       �	䂵�Xc�A�'*

lossά�=6���       �	e9��Xc�A�'*

loss�>g=�m��       �	�Զ�Xc�A�'*

loss<N=�{�       �	q��Xc�A�'*

loss2�<���       �	F��Xc�A�'*

loss�;�٧       �	����Xc�A�'*

loss�� <�!�^       �	jP��Xc�A�'*

loss)aF=c�>       �	���Xc�A�'*

loss��:�Ba       �	~���Xc�A�'*

loss��:sXE       �	M��Xc�A�'*

loss-l=<�)       �	����Xc�A�'*

loss�g�;t       �	����Xc�A�'*

loss:8<�y�7       �	l%��Xc�A�'*

loss�b�<���       �	eľ�Xc�A�'*

loss`�,<	���       �	�c��Xc�A�'*

loss�h�:�]��       �	9��Xc�A�'*

loss�Z�<d�       �	K���Xc�A�'*

loss��';#kUl       �	nh��Xc�A�'*

loss)E�<=�^       �	"���Xc�A�'*

loss��<9��       �	�Xc�A�'*

loss��3<OD       �	�;ÓXc�A�'*

lossy2<7��       �	��ÓXc�A�'*

loss_�=\2�       �	��ēXc�A�'*

loss��;b��s       �	9DœXc�A�'*

loss�w!;�CS�       �	��œXc�A�'*

loss?�<r�KX       �	��ƓXc�A�'*

loss�x<!]��       �	R)ǓXc�A�'*

loss�%�;:��B       �	*�ǓXc�A�'*

losse�?<9�N       �	�_ȓXc�A�'*

lossW�\;�	�       �	��ȓXc�A�'*

loss���<d��       �	P�ɓXc�A�'*

loss��?;��3�       �	�AʓXc�A�'*

loss�S�;����       �	��ʓXc�A�'*

loss��
=&��       �	�|˓Xc�A�'*

lossɴ�<A�]|       �	�b̓Xc�A�'*

loss��<�_��       �	��̓Xc�A�'*

loss/z�<N�P�       �	-�͓Xc�A�'*

loss�	�:�u�F       �	hAΓXc�A�'*

loss���<���       �	,�ΓXc�A�'*

loss��O;:�kq       �	@�ϓXc�A�'*

lossd�J=`�       �	,EГXc�A�'*

loss`�M<W��b       �	7�ГXc�A�'*

loss!�^=��D�       �	�|ѓXc�A�'*

loss�h5<멓�       �	LғXc�A�'*

losst�=XDq�       �	�ғXc�A�'*

loss#��;�Ġe       �	2sӓXc�A�'*

loss�p�<.)��       �	�ԓXc�A�'*

loss��<2�pz       �	�ԓXc�A�'*

loss,Y=y��]       �	S^ՓXc�A�'*

loss�95�5�       �	!֓Xc�A�'*

loss[=^zA       �	b�֓Xc�A�'*

loss�b=���       �	�5דXc�A�'*

loss�	=xYE"       �	��דXc�A�'*

loss�<����       �	�mؓXc�A�'*

lossϾ&=��f       �	~6ٓXc�A�'*

loss�r;{�V       �	Z�ٓXc�A�'*

loss�	�<��b�       �	huړXc�A�'*

loss��*<aw�       �	,ۓXc�A�'*

loss���=k�V       �	P�ۓXc�A�'*

lossr@�<�
h�       �	�bܓXc�A�'*

loss� �<�lJ�       �	[	ݓXc�A�'*

loss���;�r�       �	j�ݓXc�A�'*

loss��T<��?�       �	�8ޓXc�A�'*

loss�/=Jvy       �	��ޓXc�A�'*

loss�ң;Yy�       �	��ߓXc�A�'*

loss��V<���       �	p?��Xc�A�'*

loss(�=�J�Z       �	����Xc�A�'*

loss��<9�o�       �	��Xc�A�'*

loss�NE=�bi|       �	5)�Xc�A�'*

loss>�=t�       �	���Xc�A�'*

loss�	�<�l;       �	Ln�Xc�A�'*

loss7 ;A��&       �	��Xc�A�'*

loss��<��$       �	���Xc�A�'*

loss;�;m�       �	�C�Xc�A�'*

lossax�;���P       �	y��Xc�A�'*

loss�3< _�H       �	̛�Xc�A�'*

lossx�;K�       �	�8�Xc�A�'*

loss�2;j��	       �	�&�Xc�A�'*

loss�6=��h       �	<��Xc�A�'*

loss���<��       �	t|�Xc�A�'*

loss�$�;�>�       �	�B�Xc�A�'*

loss_�=�oI&       �	���Xc�A�'*

loss�5=���k       �	�(�Xc�A�'*

loss@�k=~��X       �	��Xc�A�'*

loss��.='%�       �	?�Xc�A�'*

loss$��;���T       �	���Xc�A�'*

loss�T�<q�;�       �	H��Xc�A�'*

lossj<���       �	���Xc�A�'*

loss�<����       �	�O�Xc�A�'*

lossv�<=���       �	���Xc�A�'*

lossOB�;:>-       �	��Xc�A�'*

lossθ�;~��       �	a�Xc�A�'*

loss���<p���       �	���Xc�A�'*

losse <��h       �	T��Xc�A�'*

loss��5<��K       �	����Xc�A�'*

losss�;��
       �	���Xc�A�'*

loss=|�<��Q�       �	�/��Xc�A�'*

loss��;�p�+       �	����Xc�A�'*

lossG/;�K�       �	Z��Xc�A�'*

lossŘ<>��       �	����Xc�A�'*

loss��<��U       �	����Xc�A�'*

loss�`;����       �	�&��Xc�A�'*

loss2��<BDN�       �	wH��Xc�A�'*

loss]p-<�3�p       �	����Xc�A�(*

loss�u�<��       �	����Xc�A�(*

loss6==�ʀ       �	o)��Xc�A�(*

loss	��<�o�1       �	����Xc�A�(*

loss�;�j3       �	�v��Xc�A�(*

loss�`<�@�       �	���Xc�A�(*

loss��=�Y�       �	е��Xc�A�(*

loss�;~;�a��       �	HQ��Xc�A�(*

loss�0^<w$�       �	%���Xc�A�(*

loss��=$�       �	�� �Xc�A�(*

losss=�y�       �	h �Xc�A�(*

loss���<����       �	'��Xc�A�(*

lossgT<<�G       �	�]�Xc�A�(*

loss�r�;���k       �	b��Xc�A�(*

loss��:�%�R       �	~��Xc�A�(*

lossNK=ǫt�       �	�#�Xc�A�(*

loss��<M��       �	��Xc�A�(*

lossQ�5=/n       �	���Xc�A�(*

loss�z<���       �	J%�Xc�A�(*

loss��;o��       �	P�Xc�A�(*

loss=wf<ͯ�       �	R��Xc�A�(*

loss��"=	�Ls       �	O�Xc�A�(*

loss?�<Y\b       �	���Xc�A�(*

loss��;Ʊ�2       �	��	�Xc�A�(*

lossC�<l��9       �	0
�Xc�A�(*

lossCM]=,)z�       �	��
�Xc�A�(*

loss� ]=+L       �	6u�Xc�A�(*

loss6-�;<�+       �	��Xc�A�(*

loss}d�<��7�       �	6��Xc�A�(*

loss�K�;8af;       �	,I�Xc�A�(*

loss3*< 6       �	���Xc�A�(*

loss`);(��       �	k~�Xc�A�(*

lossj��<�<��       �	�Xc�A�(*

loss���<��;�       �	���Xc�A�(*

lossW<�v�       �	�P�Xc�A�(*

loss�t�<�h6       �	���Xc�A�(*

loss['8<��H       �	��Xc�A�(*

loss6��<ӆ֥       �	-#�Xc�A�(*

loss�UK<�x�       �	���Xc�A�(*

loss_Q=��g       �	+L�Xc�A�(*

loss�$�<��&�       �	���Xc�A�(*

loss�;���       �	���Xc�A�(*

lossJGi<���       �	�#�Xc�A�(*

loss/�M<@�$r       �	ǹ�Xc�A�(*

loss�-�<��       �	/P�Xc�A�(*

loss(��<�g�F       �	��Xc�A�(*

loss�|�<��5P       �	���Xc�A�(*

loss!�=$	�       �	6u�Xc�A�(*

lossse�<n1L}       �	N�Xc�A�(*

loss#�<����       �	���Xc�A�(*

lossͷ%<C��       �	���Xc�A�(*

lossa�6<��       �	��Xc�A�(*

loss��:˂0       �	e��Xc�A�(*

loss��;j���       �	�J�Xc�A�(*

loss�M<��K�       �	d��Xc�A�(*

loss���<��v�       �	7��Xc�A�(*

loss�m<�e       �	w.�Xc�A�(*

loss\��;➃�       �	���Xc�A�(*

loss�w�<��Z�       �	vm �Xc�A�(*

lossQw�<��7d       �	�!�Xc�A�(*

loss?<E�#       �	��!�Xc�A�(*

lossc�;��K�       �	�4"�Xc�A�(*

loss��<\Bs       �	L�"�Xc�A�(*

lossV?�=˴|�       �	��#�Xc�A�(*

lossڵ�<S�C�       �	�$�Xc�A�(*

lossjz�<˦s.       �	��$�Xc�A�(*

loss��=l���       �	�S%�Xc�A�(*

lossU�<|�       �	�%�Xc�A�(*

loss\�;�aLv       �	�1'�Xc�A�(*

loss��;e;]B       �	��'�Xc�A�(*

loss���;�/�       �	f(�Xc�A�(*

lossu�<7~       �	T�(�Xc�A�(*

loss89�<2�՗       �	F�)�Xc�A�(*

loss��;/7t       �	��*�Xc�A�(*

loss��<@���       �	��+�Xc�A�(*

loss���;!�&       �	R�,�Xc�A�(*

loss�[�;q�       �	��-�Xc�A�(*

loss�9�:"H'       �	a�.�Xc�A�(*

loss��=`��       �	��/�Xc�A�(*

loss���:��\�       �	��0�Xc�A�(*

loss�
t<@�       �	7�1�Xc�A�(*

loss�~r<=��       �	��2�Xc�A�(*

loss�jo<�       �	�B3�Xc�A�(*

lossCQ<���       �	c^4�Xc�A�(*

loss�<<�/��       �	/5�Xc�A�(*

loss�;=<�g)       �	9�5�Xc�A�(*

loss��;�ob       �	z6�Xc�A�(*

loss�G�<ݟǛ       �	�7�Xc�A�(*

lossxi�:N���       �	��7�Xc�A�(*

loss�<с \       �	�b8�Xc�A�(*

loss���</L�k       �	&9�Xc�A�(*

loss�2�<�dU�       �	��9�Xc�A�(*

loss�8=�)ҿ       �	�g:�Xc�A�(*

loss=��       �	 ;�Xc�A�(*

loss��=��9        �	��;�Xc�A�(*

loss�'�=S##�       �	�P<�Xc�A�(*

loss��;�)�       �	�7=�Xc�A�(*

loss%�:N �       �	+�=�Xc�A�(*

loss�S<)s6�       �	��>�Xc�A�(*

loss�q=`�~F       �	�'?�Xc�A�(*

loss���<R<�       �	��?�Xc�A�(*

loss�7<q\ѓ       �	��@�Xc�A�(*

loss��<f�5       �	5(A�Xc�A�(*

loss�0:�훚       �	B�A�Xc�A�(*

lossq*�=�pڇ       �	�sB�Xc�A�(*

loss�,�;a�s       �	\ C�Xc�A�(*

lossL�e<��H       �	��C�Xc�A�(*

loss��;Dn�       �	[}D�Xc�A�(*

lossg<�s�       �	$E�Xc�A�(*

loss}�;��6�       �	��E�Xc�A�(*

lossS��;�T�B       �	�F�Xc�A�(*

loss�KQ<-       �	�;G�Xc�A�(*

lossNJ=���       �	��G�Xc�A�(*

loss��;��i�       �	0�H�Xc�A�(*

loss �
<���       �	�SI�Xc�A�(*

loss�U<��3       �	o�I�Xc�A�(*

loss��<��0h       �	S�J�Xc�A�(*

loss�<���       �	l=K�Xc�A�(*

loss4��;;�JC       �	Y�K�Xc�A�(*

loss�=d�       �	�wL�Xc�A�(*

loss։<��4�       �	�#M�Xc�A�(*

loss�J3<��       �	R�M�Xc�A�(*

loss�,�< g<�       �	N�Xc�A�(*

loss�[�<^m��       �	)O�Xc�A�(*

lossq%�<�qL�       �	Q�O�Xc�A�(*

lossӍ�<Is��       �	v�P�Xc�A�(*

loss�2<�(�       �	,)Q�Xc�A�(*

loss��=28Sg       �	��Q�Xc�A�(*

loss:+�<Ro�       �	�iR�Xc�A�)*

loss�(�;�g�K       �	iS�Xc�A�)*

lossԲ�;�>�       �	U�S�Xc�A�)*

loss��=/� �       �	fT�Xc�A�)*

loss>�#<�t�       �	�U�Xc�A�)*

lossq+;�K�l       �	�U�Xc�A�)*

losstn=��6       �	iVV�Xc�A�)*

loss'E=ox�{       �	��V�Xc�A�)*

lossR��;_���       �	V�W�Xc�A�)*

loss�=����       �	�HX�Xc�A�)*

loss}�-<:�Va       �	��X�Xc�A�)*

lossjN�;�Aj�       �	.�Y�Xc�A�)*

lossT�Z<�a�       �	Q2Z�Xc�A�)*

loss��=`�Xu       �	��Z�Xc�A�)*

loss6�k;��_       �	$~[�Xc�A�)*

loss�{�;]�8�       �	�C\�Xc�A�)*

loss���=n�5*       �	c�\�Xc�A�)*

loss�t�<y$/       �	Ę]�Xc�A�)*

loss3\�<8w�_       �	IL^�Xc�A�)*

loss���;B�w0       �	��^�Xc�A�)*

lossm60= �MC       �	�_�Xc�A�)*

losso�<����       �	�N`�Xc�A�)*

loss&�<���d       �	�b�Xc�A�)*

loss
d{=��i�       �	 �b�Xc�A�)*

loss�ZP<����       �	�Vc�Xc�A�)*

loss�yL<�բi       �	��c�Xc�A�)*

loss!|�<d��o       �	U�d�Xc�A�)*

loss�Z{;���       �	�Le�Xc�A�)*

loss��;�4       �	�f�Xc�A�)*

lossԓ\;�nv.       �	J�f�Xc�A�)*

lossD_<��UN       �	�Ug�Xc�A�)*

loss�<j-��       �	��g�Xc�A�)*

loss[m3=��y|       �		�h�Xc�A�)*

lossK=O[�       �	�Ni�Xc�A�)*

loss#6<�D��       �	��j�Xc�A�)*

loss�g�;Q��0       �	��k�Xc�A�)*

loss�=l��       �	Tol�Xc�A�)*

loss��;��#�       �	�Bm�Xc�A�)*

lossF�T<�!@m       �	��m�Xc�A�)*

losst��<�1\       �	��n�Xc�A�)*

loss�`�<��j       �	_Do�Xc�A�)*

loss�w<�`>�       �	��o�Xc�A�)*

loss-0�<�{/�       �	'�p�Xc�A�)*

lossE`�<:#-�       �	�Iq�Xc�A�)*

loss}G<9p��       �	�q�Xc�A�)*

loss惃:�(��       �	f�r�Xc�A�)*

loss��<~Q�       �	'�s�Xc�A�)*

loss��.<��\�       �	AFt�Xc�A�)*

loss#��<�G       �	(�t�Xc�A�)*

loss�yt<��h       �	��u�Xc�A�)*

loss�
h<4�)%       �	�uv�Xc�A�)*

loss��<�;�       �	Yw�Xc�A�)*

losswa�=Z�`       �	��w�Xc�A�)*

loss�x<��=       �	�rx�Xc�A�)*

loss�:t%�       �	#y�Xc�A�)*

lossѝ$;J�l�       �	&�y�Xc�A�)*

lossc=󞟠       �	6vz�Xc�A�)*

loss� :<u��       �	z{�Xc�A�)*

lossۻ0<oql�       �	�{�Xc�A�)*

lossҽ�<(��:       �	�f|�Xc�A�)*

lossK�;���       �	<}�Xc�A�)*

loss�d�;�;�       �	V�}�Xc�A�)*

loss.�;��       �	 y~�Xc�A�)*

loss��;�#�       �	b�Xc�A�)*

loss��<$Fj,       �	��Xc�A�)*

loss��<+���       �	���Xc�A�)*

loss6P
=8��       �	ü��Xc�A�)*

lossh�<��^       �	�a��Xc�A�)*

loss��y<��x�       �	O��Xc�A�)*

loss���<�N1�       �	P���Xc�A�)*

loss��;P,�       �	PP��Xc�A�)*

loss���;48�       �	%A��Xc�A�)*

loss�Ġ<0)       �	*ᅔXc�A�)*

loss��<o��       �	N��Xc�A�)*

lossjzt;��sN       �	*��Xc�A�)*

lossڜ�:�ri       �	mɇ�Xc�A�)*

loss��;���       �	�e��Xc�A�)*

loss��<K_       �	T��Xc�A�)*

loss�&<�?�M       �	Ü��Xc�A�)*

loss]C;<�D��       �	�5��Xc�A�)*

loss4�*<�U�       �	�Ί�Xc�A�)*

loss�U�<���       �	���Xc�A�)*

loss ː<��       �	C��Xc�A�)*

loss���:� �r       �	�䌔Xc�A�)*

loss*��;�_vW       �	�}��Xc�A�)*

lossr�4<f���       �	��Xc�A�)*

loss�c�<�AIm       �	|���Xc�A�)*

lossc��;x�ʭ       �	M��Xc�A�)*

loss��=��x!       �	���Xc�A�)*

loss�|<�/A       �	��Xc�A�)*

loss��;��0-       �	����Xc�A�)*

loss�^A;8��o       �	kI��Xc�A�)*

loss��<f���       �	�ᒔXc�A�)*

loss�r�<���       �	�~��Xc�A�)*

lossQB�8N�E'       �	O#��Xc�A�)*

loss��9�i'�       �	���Xc�A�)*

loss��|;�bd5       �	�c��Xc�A�)*

loss��;����       �	U���Xc�A�)*

loss26,<�t`d       �	����Xc�A�)*

loss0:��       �	XW��Xc�A�)*

losse3�;��	�       �	�#��Xc�A�)*

loss��<טA_       �	����Xc�A�)*

loss�{�;8*u       �		T��Xc�A�)*

loss��Y=��w�       �	 �Xc�A�)*

lossT	_<(t       �	����Xc�A�)*

loss3Ѕ;�zi       �	5%��Xc�A�)*

loss�S�<XE�       �	2˛�Xc�A�)*

loss�ȕ;Q��       �	 b��Xc�A�)*

loss?��;Hvi       �	����Xc�A�)*

loss�=O;�xs       �	ꑝ�Xc�A�)*

lossHY�:�m��       �	�'��Xc�A�)*

loss �+<{4�9       �	Y���Xc�A�)*

loss���<���       �	�V��Xc�A�)*

loss���=��֨       �	9�Xc�A�)*

lossm�<�D'       �	�נ�Xc�A�)*

loss�=�;�g�
       �	}��Xc�A�)*

lossE�=���P       �	���Xc�A�)*

loss�zg=|�ذ       �	���Xc�A�)*

loss�B'=R�*�       �	�K��Xc�A�)*

loss�B�;���A       �	�ᣔXc�A�)*

loss���<~�-       �	g{��Xc�A�)*

loss�mu<J���       �	E��Xc�A�)*

loss�^;�wt7       �	K���Xc�A�)*

loss���<�(k
       �	�C��Xc�A�)*

loss�oG<rz��       �	ߦ�Xc�A�)*

loss�O�;?�l-       �	�y��Xc�A�)*

loss�m<�7��       �	���Xc�A�)*

loss�}�<+�w�       �	����Xc�A�)*

loss���<��       �	C��Xc�A�**

loss�C�;��       �	���Xc�A�**

loss�'d</Z{       �	����Xc�A�**

lossP�='� �       �	M���Xc�A�**

loss_=�<k4k�       �	,��Xc�A�**

loss�<�:=       �	ڬ�Xc�A�**

lossz��;W�`9       �	qu��Xc�A�**

loss��;�Pd�       �	���Xc�A�**

loss��;40sw       �	ϡ��Xc�A�**

lossi(�;���       �	�8��Xc�A�**

lossV}�;��ީ       �	�Я�Xc�A�**

loss�=@phc       �	Af��Xc�A�**

loss��<nV��       �	o��Xc�A�**

lossw&�<�~
�       �	�Ա�Xc�A�**

loss�7�:�dx�       �	�m��Xc�A�**

loss���;��       �	�	��Xc�A�**

loss���;H!�]       �	%���Xc�A�**

loss�<[���       �	wM��Xc�A�**

loss_��:5k�       �	���Xc�A�**

loss��(<E�k�       �	���Xc�A�**

loss_ļ<-��       �	{2��Xc�A�**

lossvi <�       �	�ض�Xc�A�**

loss��[=��T       �	Cr��Xc�A�**

loss�q�:�7%y       �	
��Xc�A�**

lossgӛ;�,�       �	ҧ��Xc�A�**

loss�F;#���       �	��ДXc�A�**

loss�H;�'�=       �	|�єXc�A�**

lossI��;�F�b       �	^/ҔXc�A�**

loss\Tk<ݷ�y       �	��ҔXc�A�**

loss]�U;�l��       �	,gӔXc�A�**

loss�̦<��       �	�ӔXc�A�**

loss��<<B�h       �	ՑԔXc�A�**

loss���=���       �	�'ՔXc�A�**

loss.��<L�ʍ       �	��ՔXc�A�**

loss�5
=r�k�       �	6[֔Xc�A�**

lossF�Z<QtIn       �	H�ؔXc�A�**

lossTn-<��'�       �	�MٔXc�A�**

loss�Gd<���       �	�ٔXc�A�**

lossn-h;���       �	ѯڔXc�A�**

lossC�n<N��       �	�U۔Xc�A�**

loss֘+;��!	       �	�۔Xc�A�**

loss�g�:��       �	�ܔXc�A�**

loss���;�%��       �	�EݔXc�A�**

loss�k<<� ��       �	`�ݔXc�A�**

loss��=s�8       �	�ޔXc�A�**

lossfu�;�b��       �	2ߔXc�A�**

loss)�^=F�0       �	#�ߔXc�A�**

lossҽ�:�x�       �	Wv��Xc�A�**

loss��<T�       �	�"�Xc�A�**

loss�=o��       �	��Xc�A�**

losslŮ; �.       �	�m�Xc�A�**

loss�J<Ȃ�M       �	y�Xc�A�**

loss�?<�B �       �	��Xc�A�**

loss�ށ=M�"*       �	�x�Xc�A�**

loss�	<e���       �	n�Xc�A�**

loss��G;��M�       �	ܷ�Xc�A�**

loss8=o��C       �	�R�Xc�A�**

loss��.<�_�       �	���Xc�A�**

loss�� =�a�p       �	v��Xc�A�**

loss��1<�-�       �	�%�Xc�A�**

loss�`�<���       �	H��Xc�A�**

lossw�Y<��И       �	h��Xc�A�**

loss}�<�H�       �	���Xc�A�**

lossD(=౦a       �	�P�Xc�A�**

lossz�=��Z       �	e�Xc�A�**

loss�<S��       �	{��Xc�A�**

loss$�a;!��Q       �	���Xc�A�**

lossh�;�׍�       �	�t�Xc�A�**

lossIH<�zb_       �	7��Xc�A�**

loss-=a       �	}�Xc�A�**

lossº<���       �	\��Xc�A�**

lossxЇ<�9#       �	ٕ�Xc�A�**

lossF <�=�       �	U/�Xc�A�**

loss  �<X�8       �	���Xc�A�**

lossJ��:?Փ6       �	G��Xc�A�**

loss�
�:�ţ�       �	o+��Xc�A�**

loss��=Ȩ-       �	P���Xc�A�**

lossZ��<Af|a       �	Z��Xc�A�**

loss��n>J��L       �	x���Xc�A�**

loss��V=��Dw       �	v���Xc�A�**

lossN�:<��       �	̚��Xc�A�**

lossv�A;[�'�       �	�5��Xc�A�**

loss���:5��       �	t���Xc�A�**

loss�-<�?7       �	Ym��Xc�A�**

loss
c<[MkD       �	
��Xc�A�**

losskI=��       �	����Xc�A�**

loss�(�;�w��       �	+P��Xc�A�**

loss���:�,       �	���Xc�A�**

lossu�;kG�       �	����Xc�A�**

lossl�<�_7�       �	i��Xc�A�**

loss!�w:%+h       �	���Xc�A�**

lossן=���       �	b���Xc�A�**

loss�v�;�k��       �	�7��Xc�A�**

loss�I=�U�       �	9���Xc�A�**

loss��4<q}\       �	�j �Xc�A�**

lossa�B= |��       �	��Xc�A�**

lossl(j<gi�#       �	Ω�Xc�A�**

loss���<�J�{       �	�@�Xc�A�**

loss1|;�}�       �	���Xc�A�**

loss�Ֆ<'�V�       �	���Xc�A�**

loss;M�;��rA       �	���Xc�A�**

loss`j�:�n3H       �	'�Xc�A�**

loss�ze=q<��       �	���Xc�A�**

loss���<�2��       �	S��Xc�A�**

lossof�<��[M       �	=�Xc�A�**

loss�52=&��R       �	��Xc�A�**

lossJ�=75KQ       �	<��Xc�A�**

loss���;^A�x       �	?9	�Xc�A�**

loss�s:t�~       �	4�	�Xc�A�**

loss���<4�a       �	Χ
�Xc�A�**

lossƅ�<���       �	�'�Xc�A�**

losss-�<݆ͻ       �	^I�Xc�A�**

lossC��<�ys       �	���Xc�A�**

loss�۸<����       �	�/�Xc�A�**

loss`�<�ői       �	�t�Xc�A�**

lossZ�g=��~1       �	?�Xc�A�**

lossqhh<~�y�       �	��Xc�A�**

loss�5<B6��       �	2q�Xc�A�**

loss$)	<}m�       �	3�Xc�A�**

lossW��< �&       �	���Xc�A�**

loss�g�;I���       �	�u�Xc�A�**

lossT<=
Ue9       �	I�Xc�A�**

lossەg<��Qq       �	��Xc�A�**

loss�;�<*[�@       �	Id�Xc�A�**

loss�5�;�4�{       �	9)�Xc�A�**

loss$�;2�?       �	���Xc�A�**

loss3�U;�sO�       �	�e�Xc�A�**

loss&�M=K�'       �	��Xc�A�**

loss�4�<S�       �	[��Xc�A�+*

lossFs<�\�       �	N*�Xc�A�+*

loss4)�;�ðl       �	N��Xc�A�+*

lossT�<����       �	{��Xc�A�+*

loss��;���y       �	i�Xc�A�+*

loss�ƀ=�71       �	���Xc�A�+*

losso�;�|X       �	�'�Xc�A�+*

loss��=ԯ3       �	n��Xc�A�+*

loss��c=��ŵ       �	�]�Xc�A�+*

loss�
?=���r       �	&  �Xc�A�+*

loss��;oY!-       �	D� �Xc�A�+*

lossti�<�p��       �	6�!�Xc�A�+*

loss��;0��J       �	�)"�Xc�A�+*

loss��>�*�       �	U�"�Xc�A�+*

loss%\�<�	z       �	GW#�Xc�A�+*

lossr�,=v��*       �	��#�Xc�A�+*

loss�;���/       �	<�$�Xc�A�+*

loss�R�<���       �	�#%�Xc�A�+*

loss4�F=q:�+       �	3�%�Xc�A�+*

loss�/�<]1�       �	�p&�Xc�A�+*

loss$�=<�$��       �	�'�Xc�A�+*

loss�w4;��       �	ɰ'�Xc�A�+*

loss==���t       �	fM(�Xc�A�+*

loss��;9e       �	��(�Xc�A�+*

lossL�|;�E       �	3�)�Xc�A�+*

loss���<�J(       �	�,*�Xc�A�+*

loss��4=�L?�       �	2�*�Xc�A�+*

loss��<��@       �	t�+�Xc�A�+*

losst��<|� �       �	�6,�Xc�A�+*

lossƉ=��V�       �	��,�Xc�A�+*

loss���<�?�       �	�p.�Xc�A�+*

lossX}�<x7�       �	�/�Xc�A�+*

loss��<%�y�       �	��/�Xc�A�+*

loss&�<��       �	sd0�Xc�A�+*

loss�:�;���       �	Y�0�Xc�A�+*

loss}�;~:��       �	�1�Xc�A�+*

lossJ�7<�j�       �	M22�Xc�A�+*

loss�@;�Y�       �	��2�Xc�A�+*

lossWz�<�J�       �	Gu3�Xc�A�+*

loss��1<��A�       �	o4�Xc�A�+*

loss��e=�-Wf       �	��4�Xc�A�+*

loss=�<!�3u       �	�N5�Xc�A�+*

lossD�
<���%       �	��5�Xc�A�+*

loss���:���~       �	ȶ6�Xc�A�+*

loss��0;��]�       �	�]7�Xc�A�+*

lossV�;����       �	��7�Xc�A�+*

loss���<.)�       �	֎8�Xc�A�+*

loss�;dBQd       �	39�Xc�A�+*

lossoO�=�t�       �	��9�Xc�A�+*

loss�&<�.       �	^�:�Xc�A�+*

lossӇ�<���       �	g(;�Xc�A�+*

loss��;(��>       �	�;�Xc�A�+*

loss��;۫	%       �	c<�Xc�A�+*

loss�!<��(�       �	��<�Xc�A�+*

loss�$w=@t��       �	V�=�Xc�A�+*

loss�C�;��W       �	�F>�Xc�A�+*

loss�VY=�!5z       �	�>�Xc�A�+*

lossH�=���:       �	^�?�Xc�A�+*

loss�>V<T��       �	�&@�Xc�A�+*

loss=tE=��m;       �	�@�Xc�A�+*

loss��d:Ƥ��       �	PmA�Xc�A�+*

loss֝0=��r�       �	�B�Xc�A�+*

loss܍�;)iw=       �	��B�Xc�A�+*

loss*`2<��9�       �	�jC�Xc�A�+*

lossA��;Zp�       �	OD�Xc�A�+*

loss%�;-�GT       �	,�D�Xc�A�+*

loss���<-ycQ       �	fOE�Xc�A�+*

loss��=e�@        �	��E�Xc�A�+*

lossֿo<��H       �	��F�Xc�A�+*

loss�-<�8a       �	�4G�Xc�A�+*

loss|C	<.T>�       �	��G�Xc�A�+*

loss�g&=SY5       �	tH�Xc�A�+*

loss���<wu�       �	wI�Xc�A�+*

loss��K<K�C-       �	�I�Xc�A�+*

lossC�<��"       �	�cJ�Xc�A�+*

loss�=a�F       �	.�J�Xc�A�+*

loss��=^O�       �	$L�Xc�A�+*

lossi��9[ѝ�       �	!WM�Xc�A�+*

loss��<f�(&       �	F�M�Xc�A�+*

loss���;�[�       �	�N�Xc�A�+*

loss%��;�Y�        �	�;O�Xc�A�+*

lossZy�:�}�w       �	<�O�Xc�A�+*

loss6�m<�3�7       �	 �P�Xc�A�+*

loss��=�p        �	v2Q�Xc�A�+*

loss��Y=��8$       �	�R�Xc�A�+*

lossE�6=I�]�       �	�hS�Xc�A�+*

loss���<���H       �	�T�Xc�A�+*

losss�)=!r��       �	b�T�Xc�A�+*

loss36f<X�x       �	�;U�Xc�A�+*

loss�T2<X���       �	8�U�Xc�A�+*

loss�J;O���       �	�(W�Xc�A�+*

lossw:�:�&       �	U�W�Xc�A�+*

loss�S�;�b�^       �	\UX�Xc�A�+*

loss���;3�z�       �	��X�Xc�A�+*

losss�<d�=u       �	t�Y�Xc�A�+*

loss�=}��       �	UZ�Xc�A�+*

loss�;�:�}y       �	��Z�Xc�A�+*

loss7�=#���       �	r�[�Xc�A�+*

loss�2G<�5�       �	�$\�Xc�A�+*

loss�AI</�       �	��\�Xc�A�+*

loss�/<�m       �	��]�Xc�A�+*

loss�3e<�/B�       �	9*^�Xc�A�+*

lossN��;�� �       �	Y�^�Xc�A�+*

loss�$�<I�Dm       �	�X_�Xc�A�+*

loss�Hc=:�k�       �	�)`�Xc�A�+*

lossa�==Gy"V       �	C�`�Xc�A�+*

loss:Q�<X �       �	�]a�Xc�A�+*

lossy�;��)=       �	��a�Xc�A�+*

lossc[)<�"��       �	\�b�Xc�A�+*

loss�Le;i0�       �	`c�Xc�A�+*

loss��;��       �	Fd�Xc�A�+*

loss1p=����       �	��d�Xc�A�+*

lossL��<O!�       �	�<e�Xc�A�+*

loss�=�0+~       �	��e�Xc�A�+*

lossc	�=7���       �	�f�Xc�A�+*

loss�j=�D�       �	M.g�Xc�A�+*

loss�E�<��       �	��g�Xc�A�+*

loss�=ɑ�K       �	�mh�Xc�A�+*

loss;�;��Y       �	Gi�Xc�A�+*

lossZ�;�J��       �	/�i�Xc�A�+*

loss���<\       �	�:j�Xc�A�+*

loss�-�;Q�B       �	��j�Xc�A�+*

lossǙ:ʇQ�       �	�l�Xc�A�+*

loss$!�;ٲ�d       �	*�l�Xc�A�+*

loss���<gp>�       �	lwm�Xc�A�+*

loss��=��T�       �	�n�Xc�A�+*

loss�12;��)�       �	;�n�Xc�A�+*

loss�	<��       �	��o�Xc�A�+*

loss��:mF��       �	
q�Xc�A�,*

loss�<*<�I�E       �	a�q�Xc�A�,*

loss�)6<��P       �	��r�Xc�A�,*

loss�9<�|��       �	w�t�Xc�A�,*

lossde�<(�
�       �	%Au�Xc�A�,*

loss,��<�~��       �	
-v�Xc�A�,*

loss3�<Ѣ��       �	(cw�Xc�A�,*

losso-�;�g       �	�x�Xc�A�,*

loss{ٛ:���       �	�.y�Xc�A�,*

loss���;2י�       �	�y�Xc�A�,*

loss�|'<����       �	t`z�Xc�A�,*

loss��c;Y3I       �	�{�Xc�A�,*

loss#,�<�F+       �	��{�Xc�A�,*

loss)m@<rM4�       �	Cq|�Xc�A�,*

loss;��<u�h.       �	=}�Xc�A�,*

loss?]�;��Q       �	O�}�Xc�A�,*

loss�JW;>C�       �	�Q~�Xc�A�,*

lossƬ<�QԀ       �	^�~�Xc�A�,*

loss�t�< ���       �	���Xc�A�,*

loss�!<;���       �	M��Xc�A�,*

loss�9=�7�       �	L��Xc�A�,*

loss���;pqn�       �	灕Xc�A�,*

loss!<�;��ݻ       �	����Xc�A�,*

loss��:Ƙo�       �	g(��Xc�A�,*

lossGe<j�F       �	"Ã�Xc�A�,*

loss��/<��       �	�Z��Xc�A�,*

lossh� <��       �	t$��Xc�A�,*

loss)8<��U       �	����Xc�A�,*

loss	J�<B��       �	dW��Xc�A�,*

loss�)�<�Z�0       �	Q��Xc�A�,*

loss��;<zCP       �	���Xc�A�,*

loss|��<>�w�       �	�b��Xc�A�,*

loss���<�|7	       �	���Xc�A�,*

loss�G=��l       �	p���Xc�A�,*

lossF%;K��       �	�^��Xc�A�,*

loss&�=* <�       �	�
��Xc�A�,*

lossaJ<�2cO       �	?���Xc�A�,*

lossw2�;�x]A       �	����Xc�A�,*

loss��N<_�z       �	�C��Xc�A�,*

loss�=�T�       �	�䍕Xc�A�,*

loss���<8�       �	���Xc�A�,*

lossԳ=\�T       �	"��Xc�A�,*

lossG[<1�A       �	����Xc�A�,*

loss��=���(       �	P��Xc�A�,*

lossM�q=�g       �	 ��Xc�A�,*

lossoϙ<䘲�       �	w���Xc�A�,*

loss=��;��       �	�=��Xc�A�,*

loss�V|<[�k�       �	�ڒ�Xc�A�,*

loss&�C=����       �	���Xc�A�,*

lossSbb;�z<c       �	���Xc�A�,*

loss���:�I�       �	O���Xc�A�,*

lossC��<��       �	�D��Xc�A�,*

loss��7<�x�       �	���Xc�A�,*

loss��<��       �	Ҫ��Xc�A�,*

lossNB�<O\{�       �	JC��Xc�A�,*

loss�<���#       �	�ᗕXc�A�,*

lossZ��:���       �	���Xc�A�,*

loss!b�<�       �	��Xc�A�,*

loss��,<����       �	y晕Xc�A�,*

loss�'<���       �	A~��Xc�A�,*

lossjH<C�U[       �	���Xc�A�,*

loss�'�:d0o       �	q���Xc�A�,*

loss��";�[(�       �	4G��Xc�A�,*

loss�v<��       �	����Xc�A�,*

loss��;���       �	9���Xc�A�,*

loss-o�;��       �	G>��Xc�A�,*

loss�)<��T�       �	Uٞ�Xc�A�,*

loss�h:=�E�       �	By��Xc�A�,*

lossT�<cS{       �	+��Xc�A�,*

loss�݃<�0(�       �	���Xc�A�,*

loss6��;m�7       �	u���Xc�A�,*

losss~�=���.       �	+2��Xc�A�,*

loss�K=�Γ�       �	�Ԣ�Xc�A�,*

loss�q?<� B       �	x��Xc�A�,*

loss�v%=��       �	*��Xc�A�,*

loss�Z�<v�6�       �	���Xc�A�,*

loss�GE<��n       �	Z��Xc�A�,*

loss��<DC"�       �	��Xc�A�,*

lossD-�;A:�       �	����Xc�A�,*

loss(|�;��j       �	N(��Xc�A�,*

loss4�0;���       �	�ȧ�Xc�A�,*

loss���<�Ҟ�       �	�a��Xc�A�,*

loss(�;_�(�       �	�	��Xc�A�,*

loss��';�4z0       �	a���Xc�A�,*

loss[��<�TB�       �	_@��Xc�A�,*

loss#�;�ʨ�       �	�媕Xc�A�,*

losseԅ<)K�t       �	����Xc�A�,*

loss2��<EU��       �	걬�Xc�A�,*

loss3(/<˱��       �	]���Xc�A�,*

loss�V�<�.t�       �	z��Xc�A�,*

loss�"$=~,,�       �	vQ��Xc�A�,*

lossL<a|��       �	����Xc�A�,*

loss��:�&1�       �	|���Xc�A�,*

loss�`=
F�       �	�7��Xc�A�,*

loss�-=�       �	�᱕Xc�A�,*

loss�<0^�       �	�~��Xc�A�,*

loss�=I�J       �	�'��Xc�A�,*

loss`�^=^ %L       �	�ⳕXc�A�,*

loss�F`<���       �	⒴�Xc�A�,*

losse�<W�)�       �	29��Xc�A�,*

loss.[�<����       �	�ܵ�Xc�A�,*

lossT";5A��       �	�X��Xc�A�,*

loss�6;ǃb�       �	����Xc�A�,*

loss�<�LP/       �	R���Xc�A�,*

loss��u;��j       �	C9��Xc�A�,*

loss~�=��       �	�๕Xc�A�,*

loss�o�;m|a�       �	Ό��Xc�A�,*

lossK=��?p       �	2��Xc�A�,*

loss��:�m       �	EԻ�Xc�A�,*

loss���;I�E|       �	�r��Xc�A�,*

loss��N;��0j       �	.��Xc�A�,*

lossD�p<�-��       �	Z���Xc�A�,*

loss���; ��       �	�\��Xc�A�,*

loss}h�<��[       �	X��Xc�A�,*

loss3y=l��       �	i���Xc�A�,*

loss�&�;cpy       �	(G��Xc�A�,*

loss;:=Ϣ�       �	����Xc�A�,*

loss(�^<��Z       �	ĕ��Xc�A�,*

lossT�<xi�       �	�6Xc�A�,*

loss��;�v��       �	��Xc�A�,*

loss]ˣ9[:{F       �	؀ÕXc�A�,*

loss��<�\U`       �	x'ĕXc�A�,*

loss�� <].�       �	e�ĕXc�A�,*

loss��X=W���       �	cƕXc�A�,*

loss���:�l�d       �	ǕXc�A�,*

loss��<Ѓ��       �	��ǕXc�A�,*

losso��<*ݏ       �	�oȕXc�A�,*

lossH�	<�I       �	�ɕXc�A�,*

loss|0<�˻�       �	G�ɕXc�A�-*

loss%S�;Z2�       �	�rʕXc�A�-*

lossߤ#<��&U       �	�˕Xc�A�-*

lossͿ\=F�#       �	w�˕Xc�A�-*

loss���< �1�       �	�`̕Xc�A�-*

loss��=�c�
       �	��̕Xc�A�-*

loss�Օ<�n�d       �	��͕Xc�A�-*

lossA*=GO�P       �	YOΕXc�A�-*

loss(c<�F��       �	��ΕXc�A�-*

loss�)<�u��       �	M�ϕXc�A�-*

loss�}�;�z�B       �	>>ЕXc�A�-*

lossB<�	��       �	��ЕXc�A�-*

lossH�v;#k�:       �	�ѕXc�A�-*

loss]H�=���       �	-%ҕXc�A�-*

loss��"<f���       �	��ҕXc�A�-*

loss�5=�sH       �	:wӕXc�A�-*

loss�Y�<x�ua       �	�ԕXc�A�-*

loss��;3�MC       �	��ԕXc�A�-*

loss�[)=����       �	]NՕXc�A�-*

loss�&=.χ�       �	/�ՕXc�A�-*

loss�p}</�U�       �	՗֕Xc�A�-*

loss	Y;�       �	W^וXc�A�-*

loss��=%�b       �	�וXc�A�-*

loss�T=�4<       �	��ؕXc�A�-*

loss��<�},.       �	�*ٕXc�A�-*

loss���:�vP�       �	��ٕXc�A�-*

loss��<pU�       �	x^ڕXc�A�-*

loss��<<�       �	�ەXc�A�-*

lossD�@=��       �	��ەXc�A�-*

lossRu=�j(�       �	�oݕXc�A�-*

loss_.�<?DKG       �	ޕXc�A�-*

lossj�;�p!       �	�ޕXc�A�-*

losso�x<VB7       �	^KߕXc�A�-*

loss���:Z�3        �	��ߕXc�A�-*

loss@V6=���       �	����Xc�A�-*

losse�=ē%�       �	��Xc�A�-*

loss�n�<�o�       �	��Xc�A�-*

loss#��9�{z�       �	�N�Xc�A�-*

loss��<:�q�G       �	���Xc�A�-*

loss�K�<��'�       �	ӆ�Xc�A�-*

loss�bp<�њe       �	�Xc�A�-*

loss'�<W�/       �	ض�Xc�A�-*

lossNM<hI1�       �	vR�Xc�A�-*

loss�<BV�_       �	���Xc�A�-*

loss,��<ܫ�       �	���Xc�A�-*

loss(4=�%��       �	��Xc�A�-*

lossZ<K�Y�       �	`��Xc�A�-*

loss|��<`�B�       �	�G�Xc�A�-*

loss�B<�O{�       �	���Xc�A�-*

lossHRH=�sf�       �	��Xc�A�-*

loss�=<����       �	��Xc�A�-*

loss]�=X�       �	>��Xc�A�-*

loss���<�0��       �	(`�Xc�A�-*

loss���;nf�	       �	��Xc�A�-*

loss���<��       �	���Xc�A�-*

lossu�=>�u       �	��Xc�A�-*

loss���;�6�       �	VJ�Xc�A�-*

loss�ȧ;2�hU       �	���Xc�A�-*

lossݾ:t��       �	���Xc�A�-*

loss{��<���q       �	P�Xc�A�-*

lossCӀ<t%��       �	���Xc�A�-*

loss���<�,��       �	���Xc�A�-*

lossa F<S�'       �	(C�Xc�A�-*

lossSv�<�^;�       �	T��Xc�A�-*

loss�Ф<��       �	`��Xc�A�-*

lossʛ�<U��9       �	q<��Xc�A�-*

loss���;����       �	e���Xc�A�-*

loss�A`<@���       �	���Xc�A�-*

loss?Z=��m�       �	�-��Xc�A�-*

lossG��;Q^�       �	���Xc�A�-*

loss)��<D$y�       �	����Xc�A�-*

lossֳ%<W�/       �	,H��Xc�A�-*

loss��t;a��       �	!���Xc�A�-*

loss��b;�]�L       �	2���Xc�A�-*

loss�o�<r��n       �	z9��Xc�A�-*

loss�,<� �       �	Y���Xc�A�-*

loss
�	<	�i�       �	=���Xc�A�-*

loss�s&<V��$       �	����Xc�A�-*

lossP�;m8�       �	H��Xc�A�-*

losst�<R~%       �	>���Xc�A�-*

loss�?<�:�       �	U���Xc�A�-*

loss��;�4C       �	Q���Xc�A�-*

loss�D�<��J]       �	O# �Xc�A�-*

lossú�;�[�b       �	�� �Xc�A�-*

lossMb�<�\�       �	\q�Xc�A�-*

lossB��<�bm       �	�<�Xc�A�-*

lossc9\<V�N�       �	E��Xc�A�-*

loss�]�<+J|3       �	��Xc�A�-*

lossg,�<X@�       �	�0�Xc�A�-*

loss
[�;%��       �	���Xc�A�-*

lossWz{<�%��       �	�{�Xc�A�-*

lossH�A=�y�       �	^g�Xc�A�-*

lossxC�<{
"       �	$��Xc�A�-*

loss�k�;c��       �	g(�Xc�A�-*

lossv�;r       �	!��Xc�A�-*

loss�A;�=e       �	|	�Xc�A�-*

loss9='Zar       �	�&
�Xc�A�-*

lossO��<�B�       �	��
�Xc�A�-*

loss�۟;�VT�       �	5�Xc�A�-*

lossr��;*���       �	�&�Xc�A�-*

loss��<g\��       �	9+�Xc�A�-*

loss��O;�ߧ�       �	���Xc�A�-*

loss��T;k�       �	^��Xc�A�-*

loss�;���       �	E)�Xc�A�-*

loss���=k�       �	���Xc�A�-*

loss�;��L       �	�r�Xc�A�-*

loss�9O<���       �	G�Xc�A�-*

loss(L%;�!ba       �	r��Xc�A�-*

loss��;@N       �	�k�Xc�A�-*

lossrN=8\�<       �	�Xc�A�-*

loss�[�;���       �	>��Xc�A�-*

loss��t=χh?       �	||�Xc�A�-*

loss��<�(�V       �	�.�Xc�A�-*

loss���;
i�       �	|��Xc�A�-*

losse<6��q       �	�D�Xc�A�-*

lossm�<E�{v       �	��Xc�A�-*

loss� t< ��[       �	J��Xc�A�-*

loss1~�;?��!       �	q:�Xc�A�-*

loss���<��~       �	���Xc�A�-*

lossn$�<��       �	���Xc�A�-*

loss�5=5��O       �	_B�Xc�A�-*

loss���<!Q       �	���Xc�A�-*

loss�j�<�m��       �	���Xc�A�-*

lossW�4;{ZL�       �	�C�Xc�A�-*

loss�w�;+hi       �	F��Xc�A�-*

lossXq�<L�$n       �	p��Xc�A�-*

loss�ck<���       �	q8�Xc�A�-*

lossz�E<>]�$       �	@��Xc�A�-*

lossq�<�H�|       �	߇ �Xc�A�.*

loss�b�;>�}�       �	�-!�Xc�A�.*

lossA�7;o3�       �	t�!�Xc�A�.*

loss,�;��       �	By"�Xc�A�.*

loss�^<ȅFN       �	�$#�Xc�A�.*

losss�;��       �	*�#�Xc�A�.*

lossy�<Wz/       �	kf$�Xc�A�.*

loss�N�<O��c       �	�%�Xc�A�.*

loss#�D<�֊�       �	>�%�Xc�A�.*

lossv+�<q��A       �	ff&�Xc�A�.*

loss2�<���Q       �	p'�Xc�A�.*

losse�2=�T2�       �	��'�Xc�A�.*

loss8ͅ</9�       �	is(�Xc�A�.*

loss��6=�� �       �	)�Xc�A�.*

lossM�J:Xk�\       �	
�)�Xc�A�.*

loss���;�s�       �	�[*�Xc�A�.*

lossd<�:��D�       �	3�*�Xc�A�.*

loss)�;B��b       �	c�+�Xc�A�.*

lossD�;h&��       �	/0,�Xc�A�.*

loss��=
��       �	-�Xc�A�.*

loss3d�:Xɣ�       �	��-�Xc�A�.*

loss~�<�5�.       �	|.�Xc�A�.*

lossv�><�4K�       �	��/�Xc�A�.*

loss)�N:F��D       �	��0�Xc�A�.*

loss�²;�c�k       �	��1�Xc�A�.*

lossUڍ<��N       �	��2�Xc�A�.*

loss,= �z�       �	i�3�Xc�A�.*

loss&^B;�S�       �	�Q4�Xc�A�.*

loss�X=5�G�       �	l�4�Xc�A�.*

loss���<iC]       �	U�5�Xc�A�.*

loss�[0<�w�       �	�g6�Xc�A�.*

loss���;��'�       �	��6�Xc�A�.*

lossRa�;�9�       �	��7�Xc�A�.*

loss<��:&�0       �	.Y8�Xc�A�.*

loss�[;=��c       �	�8�Xc�A�.*

loss~�:�CP       �	t�9�Xc�A�.*

lossȦ?;�6�       �	�-:�Xc�A�.*

loss�h�<��       �	z�:�Xc�A�.*

loss�U<��v�       �	y];�Xc�A�.*

lossef�<���r       �	��;�Xc�A�.*

loss���;��       �	��<�Xc�A�.*

loss�H�</�       �	�2=�Xc�A�.*

loss��=�pD       �	��=�Xc�A�.*

loss�n�:�ɸ4       �	ׄ>�Xc�A�.*

lossc�s<�Q       �	m?�Xc�A�.*

loss���:f       �	��?�Xc�A�.*

loss��99�[�       �	�[@�Xc�A�.*

loss�
z;q���       �	��@�Xc�A�.*

lossl��:�ZM/       �	��A�Xc�A�.*

loss;�Т       �	bB�Xc�A�.*

loss��;D�1|       �	i�B�Xc�A�.*

lossײַ9��H�       �	�D�Xc�A�.*

loss�ذ;7mB�       �	ܷD�Xc�A�.*

loss�Q�<�$Q�       �	�fE�Xc�A�.*

loss/�
9&^>�       �	�yF�Xc�A�.*

loss!\:�*�Z       �	ZG�Xc�A�.*

loss(�<�8Р       �	��G�Xc�A�.*

loss�%�<{ .h       �	MgH�Xc�A�.*

lossV�0<�(�       �	a�H�Xc�A�.*

loss]B:��e       �	��I�Xc�A�.*

loss�ș;��       �	?5J�Xc�A�.*

loss��=+51n       �	��J�Xc�A�.*

loss���;E8x       �	�dK�Xc�A�.*

loss	�=�M��       �	�KL�Xc�A�.*

loss1�D;T       �	.UM�Xc�A�.*

lossT��;�%e       �	�M�Xc�A�.*

loss���<���       �	�N�Xc�A�.*

loss���:S�G�       �	l$O�Xc�A�.*

loss�!;p�n       �	�O�Xc�A�.*

losscK:��X       �	�WP�Xc�A�.*

loss�
�<�Ʌa       �	��P�Xc�A�.*

loss	;Ϙ҅       �	�Q�Xc�A�.*

loss�:$:�alf       �	�mR�Xc�A�.*

loss�~,<Ɖ\?       �	�yS�Xc�A�.*

loss�U\<���u       �	\T�Xc�A�.*

losszʒ<r�["       �	��T�Xc�A�.*

loss�W=i�+V       �	�rU�Xc�A�.*

lossʯ<�
JI       �	�V�Xc�A�.*

loss�c<%��       �	�V�Xc�A�.*

loss.��;�4�^       �	�[W�Xc�A�.*

loss�c<��Ϳ       �	�W�Xc�A�.*

loss	z;� �       �	��X�Xc�A�.*

loss��Y;�BL       �	4�Y�Xc�A�.*

loss���<F)�       �	�=Z�Xc�A�.*

loss<�S<�M�       �	!�Z�Xc�A�.*

loss�kW<J1Ԋ       �	�[�Xc�A�.*

loss
.&<M��       �	-\�Xc�A�.*

loss
��;S˸        �	��\�Xc�A�.*

loss=,�<��-       �	�n]�Xc�A�.*

lossؙD=�_u       �	�^�Xc�A�.*

loss��<S�       �	��^�Xc�A�.*

loss��=�[�M       �	�e_�Xc�A�.*

loss���<.���       �	�`�Xc�A�.*

loss\[<����       �	0�`�Xc�A�.*

lossh��;p2B�       �	nMa�Xc�A�.*

loss@��:�h       �	N�a�Xc�A�.*

loss�?<�#�)       �	�b�Xc�A�.*

loss=�<��       �	�c�Xc�A�.*

lossW��;؃�2       �	B"d�Xc�A�.*

loss)F�<�J�       �	��d�Xc�A�.*

loss�<�q�       �	�xe�Xc�A�.*

loss�&1;��CC       �	$f�Xc�A�.*

loss��$:���\       �	��f�Xc�A�.*

losswU�;�7�       �	�fg�Xc�A�.*

loss�H�<��+�       �	�h�Xc�A�.*

loss*; �/       �	p�h�Xc�A�.*

lossA"<����       �	+Qi�Xc�A�.*

loss�4;~A�'       �	�i�Xc�A�.*

loss|h=��V�       �	H�j�Xc�A�.*

loss!�^;cfR�       �	ZFk�Xc�A�.*

loss���<M�\1       �	��k�Xc�A�.*

loss1;Ns�       �	REm�Xc�A�.*

loss���<-���       �	��m�Xc�A�.*

loss�3"=�y�       �	q9��Xc�A�.*

lossM��<l�hp       �	W̆�Xc�A�.*

loss�B�<��:y       �	]p��Xc�A�.*

loss��<;��       �	36��Xc�A�.*

loss��O<a/       �	�ш�Xc�A�.*

loss�F�<���       �	���Xc�A�.*

lossq-�;Κy       �	n��Xc�A�.*

loss�� =�6�       �	ĳ��Xc�A�.*

loss=h�(�       �	8L��Xc�A�.*

loss_o�<�Ҹ       �	�茖Xc�A�.*

loss��S<��ޱ       �	L���Xc�A�.*

loss�j<[� �       �	�0��Xc�A�.*

lossXӆ<��       �	�Վ�Xc�A�.*

loss@gF;�a�       �	�z��Xc�A�.*

lossHϰ<�=�       �	`��Xc�A�.*

lossL<-;�l       �	�Ð�Xc�A�/*

lossw��9;e.&       �	'h��Xc�A�/*

lossR�;�+Um       �	���Xc�A�/*

lossuK<��>       �	����Xc�A�/*

loss���<@,i�       �	�H��Xc�A�/*

loss�y;w*4       �	�ꓖXc�A�/*

loss�«<��       �	;���Xc�A�/*

loss��:
���       �	�0��Xc�A�/*

loss��<x�2�       �	Օ�Xc�A�/*

loss��<����       �	y��Xc�A�/*

lossㄜ;��z`       �	�#��Xc�A�/*

loss��<�mO;       �	ທ�Xc�A�/*

loss�Qj;콉T       �	iQ��Xc�A�/*

lossL�=j��       �	�瘖Xc�A�/*

loss�n=����       �	4���Xc�A�/*

loss'�;;ˍ�       �	r��Xc�A�/*

loss�;�N�f       �	1���Xc�A�/*

loss��0;�b��       �	_\��Xc�A�/*

lossq9"=��K�       �	���Xc�A�/*

loss#V�:\4��       �	˜��Xc�A�/*

loss���<tt�       �	�=��Xc�A�/*

lossR7.;HF�h       �	#۝�Xc�A�/*

loss�`�<x���       �	����Xc�A�/*

loss�t�<��T�       �	+4��Xc�A�/*

loss�\�<R��       �	�ԟ�Xc�A�/*

lossja<���\       �	죠�Xc�A�/*

lossp��;�G�>       �	�@��Xc�A�/*

loss���;� �@       �	ᢖXc�A�/*

loss��/<��C       �	�y��Xc�A�/*

loss�	=tE�       �	���Xc�A�/*

loss@a;tݘ�       �	Ḥ�Xc�A�/*

loss���<p�˱       �	\X��Xc�A�/*

loss��
=AE��       �	v���Xc�A�/*

loss�`�;Zwq	       �	����Xc�A�/*

lossʪ 9��s1       �	�N��Xc�A�/*

loss�&F:��Q�       �	����Xc�A�/*

loss��<D!��       �	P���Xc�A�/*

loss�ƛ<�ksN       �	�T��Xc�A�/*

loss�i>T���       �	r���Xc�A�/*

loss;7�<�/��       �	J���Xc�A�/*

lossD7;ZK��       �	9d��Xc�A�/*

loss��;q�7v       �	���Xc�A�/*

loss{K�:=Q�6       �	����Xc�A�/*

lossA��;Dv�
       �	�{��Xc�A�/*

loss��<U�k�       �	�B��Xc�A�/*

lossO�; t       �	w���Xc�A�/*

loss_ǧ;W�t�       �	�P��Xc�A�/*

lossq�T:q�       �	y��Xc�A�/*

loss��v<��       �	����Xc�A�/*

loss.*�;��       �	�X��Xc�A�/*

loss�<��١       �	U���Xc�A�/*

loss��G<4�0�       �	i���Xc�A�/*

loss�]�;)��       �	 W��Xc�A�/*

loss t<<0y       �	��Xc�A�/*

loss\<;��       �	,���Xc�A�/*

loss��z<}!Z]       �	�w��Xc�A�/*

loss�m�=)��       �	�5��Xc�A�/*

loss׌<�;S       �	��Xc�A�/*

loss��;��w       �	A���Xc�A�/*

loss�<b}       �	�D��Xc�A�/*

lossJ<
��       �	꺖Xc�A�/*

lossV�:��       �	+��Xc�A�/*

lossו�:�Ɏ4       �	f���Xc�A�/*

lossS�<SE       �	dw��Xc�A�/*

loss��<����       �	���Xc�A�/*

loss�d=���       �	���Xc�A�/*

loss{��<O4�=       �	F_��Xc�A�/*

loss��<O�       �	nj��Xc�A�/*

loss�%T:�:ǟ       �	b��Xc�A�/*

loss�m�<8VZ       �	����Xc�A�/*

loss�c�;_���       �	�SXc�A�/*

loss���<��M)       �	�2ÖXc�A�/*

loss,[�;�s2�       �	|�ÖXc�A�/*

lossn'=��pd       �	rmĖXc�A�/*

losse�Y<�Λ�       �	�ŖXc�A�/*

lossXù;ۡ       �	�ŖXc�A�/*

loss���<�4rk       �	2;ƖXc�A�/*

loss�C%<>�y       �	��ƖXc�A�/*

lossI�&<#"<�       �	ZgǖXc�A�/*

lossc�=����       �	m�ǖXc�A�/*

loss7Ȥ:v5�A       �	'�ȖXc�A�/*

lossXQ�<�w2       �	5AɖXc�A�/*

loss�&<	ZqL       �	�ɖXc�A�/*

lossZ�l<+��4       �	��ʖXc�A�/*

loss}��<�Rě       �	�˖Xc�A�/*

loss�{/;�:       �	fi̖Xc�A�/*

loss�ؠ:��       �	�͖Xc�A�/*

loss�M�;���       �	��͖Xc�A�/*

losso_�<h�.�       �	�VΖXc�A�/*

loss_W�<�kC�       �	��ΖXc�A�/*

loss�0<��$�       �	øϖXc�A�/*

loss�\�:�M�       �	m�ЖXc�A�/*

loss�J:���}       �	�fіXc�A�/*

losst�=�       �	t	ҖXc�A�/*

loss��[=��1�       �	*�ҖXc�A�/*

lossmgP<4���       �	oFӖXc�A�/*

loss�$<�|W       �	��ӖXc�A�/*

loss=�<8ӗ       �	+�ԖXc�A�/*

loss���:��p�       �	�ՖXc�A�/*

loss^�<·��       �	V�ՖXc�A�/*

loss��;��       �	F]֖Xc�A�/*

loss��<q[0�       �	��֖Xc�A�/*

loss���=�v       �	��זXc�A�/*

loss%�<%��i       �	QؖXc�A�/*

loss'��;���       �	��ؖXc�A�/*

lossئ =~���       �	�ٖXc�A�/*

loss v�<�g�       �	�ږXc�A�/*

loss�@G;^Lz�       �	�ږXc�A�/*

loss[��;�`U�       �	&�ۖXc�A�/*

loss�9%:}_Z�       �	�,ܖXc�A�/*

lossC�w<�J�K       �	"�ܖXc�A�/*

loss���<\�l       �	�tݖXc�A�/*

lossz��:۝2       �	ޖXc�A�/*

loss�|�<����       �	J�ޖXc�A�/*

loss��!;g�@>       �		PߖXc�A�/*

losst=����       �	��ߖXc�A�/*

loss��<|��       �	~���Xc�A�/*

lossjR<�q{�       �	rR�Xc�A�/*

loss�4;�m       �	���Xc�A�/*

lossﺟ=�ܕ       �	���Xc�A�/*

loss$�;�|	       �	�8�Xc�A�/*

loss�Ё<�sȮ       �	��Xc�A�/*

lossX�<�ףH       �	y��Xc�A�/*

loss�B=��њ       �	��Xc�A�/*

lossQ��<��5       �	z4�Xc�A�/*

loss/�;��p&       �	�@�Xc�A�/*

loss�ź;��K�       �	���Xc�A�/*

loss�o$<^�@8       �	�q�Xc�A�/*

loss���<fՑA       �	��Xc�A�0*

loss�*"=#y=       �	���Xc�A�0*

loss�h3<�2A�       �	�4�Xc�A�0*

loss\�<��'�       �	���Xc�A�0*

loss];�t��       �	�c�Xc�A�0*

loss8h�; 1]       �	� �Xc�A�0*

loss���<�>�       �	��Xc�A�0*

loss��#<@��       �	ޑ�Xc�A�0*

loss��n=x��p       �	R*�Xc�A�0*

loss��<�W�.       �	1a�Xc�A�0*

lossT��<�>��       �	t��Xc�A�0*

loss�p�<-���       �	j��Xc�A�0*

loss���:O �       �	�Y�Xc�A�0*

loss���;�i��       �	��Xc�A�0*

loss/d-<B�Ɇ       �	ŭ�Xc�A�0*

loss�أ;�X�c       �	�D��Xc�A�0*

loss�g�<�j`       �	����Xc�A�0*

loss�Z&=��6       �	�z��Xc�A�0*

loss��*<{C{p       �	���Xc�A�0*

loss��<!��       �	���Xc�A�0*

lossJ�7:! F?       �	]P��Xc�A�0*

lossU=A�x�       �	����Xc�A�0*

loss%�v:i��       �	���Xc�A�0*

loss@�:p��K       �	+��Xc�A�0*

loss�"=6;�       �	����Xc�A�0*

loss�<��       �	
���Xc�A�0*

loss	�=�^	�       �	���Xc�A�0*

loss��<D`m�       �	n���Xc�A�0*

loss@��;aw|�       �	?n��Xc�A�0*

loss-�F;�a�@       �	���Xc�A�0*

loss;K�;-U��       �	ͱ��Xc�A�0*

loss�̸:�D|       �	?X��Xc�A�0*

loss���;Τf�       �	����Xc�A�0*

lossc�
<�x�W       �	ǂ��Xc�A�0*

loss�c<hC�)       �	� �Xc�A�0*

loss�Z�<<{�v       �	>� �Xc�A�0*

loss��%=�|       �	�K�Xc�A�0*

loss;k�92:�M       �	���Xc�A�0*

lossT�<F���       �	_��Xc�A�0*

loss���;]=��       �	�Z�Xc�A�0*

loss��;*��S       �	���Xc�A�0*

loss��:k$>�       �	��Xc�A�0*

loss���::�       �	s*�Xc�A�0*

loss�m=3�66       �	���Xc�A�0*

lossc��=����       �	[�Xc�A�0*

lossL�"<N�6a       �	J��Xc�A�0*

lossHN�<:i��       �	���Xc�A�0*

lossV��=�i��       �	�f�Xc�A�0*

loss�6�<`@�       �	K	�Xc�A�0*

loss��<;c�*=       �	J�	�Xc�A�0*

loss���<���t       �	�7
�Xc�A�0*

loss_��:�j��       �	��
�Xc�A�0*

loss��;��+"       �	Pn�Xc�A�0*

lossfv�;+�D�       �	
�Xc�A�0*

loss(=�0�       �	;��Xc�A�0*

loss�M�<��@�       �	��Xc�A�0*

loss!<P;����       �	��Xc�A�0*

loss$-�<�,	p       �	R��Xc�A�0*

loss:��;��[       �	�P�Xc�A�0*

loss�X�<޴�w       �	.7�Xc�A�0*

loss�o�< ��       �	���Xc�A�0*

loss�x�<�(f)       �	�k�Xc�A�0*

loss��8<����       �	��Xc�A�0*

lossN�<��!]       �	J��Xc�A�0*

loss�j=�R
       �	j0�Xc�A�0*

lossȘ)<y׌       �	P��Xc�A�0*

loss�?�<��       �	F\�Xc�A�0*

loss֞�;e,�       �	4��Xc�A�0*

lossJ<-�       �	���Xc�A�0*

lossRP�:V�)       �	�5�Xc�A�0*

lossq�<��*n       �	���Xc�A�0*

loss3��<���       �	;m�Xc�A�0*

loss��=7���       �	�Xc�A�0*

losss3%=��g
       �	���Xc�A�0*

loss=��~�       �	t@�Xc�A�0*

loss؁=A�       �	��Xc�A�0*

lossy�<֙��       �	���Xc�A�0*

loss&Q<���W       �	I�Xc�A�0*

loss;�<y�        �	���Xc�A�0*

lossᆁ;D�չ       �	�t�Xc�A�0*

loss���<ݼ�       �	wK�Xc�A�0*

lossC�=��       �	���Xc�A�0*

lossm�:1j�0       �	��Xc�A�0*

loss�;=�7f�       �	� �Xc�A�0*

loss�<[�x�       �	�� �Xc�A�0*

loss9�<�nV&       �	gD!�Xc�A�0*

loss�
];R�l       �	8�!�Xc�A�0*

loss��1<$�jB       �	�p"�Xc�A�0*

loss��;cջ�       �	�#�Xc�A�0*

loss���<���       �	��#�Xc�A�0*

loss|#�;���       �	.8$�Xc�A�0*

loss�W�<��       �	��$�Xc�A�0*

loss�;=�C�<       �	�j%�Xc�A�0*

loss���;�        �	�&�Xc�A�0*

loss8�J<qJ�&       �	�&�Xc�A�0*

losslp�;%�       �	�0'�Xc�A�0*

loss|׷9G���       �	��'�Xc�A�0*

loss�<���       �	�`(�Xc�A�0*

loss�-�;�?d�       �	D�(�Xc�A�0*

loss��<��       �	��)�Xc�A�0*

loss�(=�jk*       �	G!*�Xc�A�0*

lossl$< (�       �	ٵ*�Xc�A�0*

lossrl#;���p       �	�K+�Xc�A�0*

loss�C%<�)�*       �	�+�Xc�A�0*

loss|x;���       �	:v,�Xc�A�0*

loss�f;%|�       �	�-�Xc�A�0*

loss�ş<_��&       �	)�-�Xc�A�0*

loss��~=��h       �	�b.�Xc�A�0*

loss�s�<��P�       �	#�/�Xc�A�0*

lossT/i;f�w       �	]�0�Xc�A�0*

loss7(;��        �	�2�Xc�A�0*

loss6�):I�c�       �	��2�Xc�A�0*

lossw1+;W@�       �	3�Xc�A�0*

loss��<D�E       �	�#4�Xc�A�0*

loss��<@>-�       �	� 5�Xc�A�0*

loss/��<��G       �	I6�Xc�A�0*

lossI�;��5t       �	��6�Xc�A�0*

loss
ϩ=�Ͼ2       �	�i7�Xc�A�0*

loss:|�;ľ9�       �	�8�Xc�A�0*

loss�8�<����       �	K>9�Xc�A�0*

loss���<�ΐ�       �	%�9�Xc�A�0*

loss�vE<���       �	u�:�Xc�A�0*

lossS;��+       �	�0;�Xc�A�0*

loss��<�'�&       �	��;�Xc�A�0*

loss�6�;�_��       �	�i<�Xc�A�0*

lossI�u;F_��       �	�=�Xc�A�0*

loss�p�<�IA�       �	 �=�Xc�A�0*

loss���<��/       �	�p>�Xc�A�0*

loss�ɂ<JWR�       �	�?�Xc�A�1*

loss��<ZV�       �	H�?�Xc�A�1*

loss�p�<4�w�       �	$E@�Xc�A�1*

lossQM=��E�       �	��@�Xc�A�1*

lossL�=u��j       �	��A�Xc�A�1*

lossX#<�$�       �	8-B�Xc�A�1*

loss��U;��]�       �	��B�Xc�A�1*

loss?y�<���       �	KuC�Xc�A�1*

lossEr0=ң��       �	�
D�Xc�A�1*

loss6�,;L�7       �	�D�Xc�A�1*

loss�~�;H��.       �	:;E�Xc�A�1*

loss��;����       �	!�E�Xc�A�1*

loss�I<�챆       �	B�F�Xc�A�1*

loss��);�D�       �	�pG�Xc�A�1*

loss���<9��X       �	�H�Xc�A�1*

loss��<D�?       �	��H�Xc�A�1*

lossj�;��y�       �	LRI�Xc�A�1*

lossbf=���       �	��I�Xc�A�1*

loss�;�)$       �	�J�Xc�A�1*

lossM��:�/O�       �	�YK�Xc�A�1*

lossoa&<=A        �	��K�Xc�A�1*

loss��J;�azp       �	L�L�Xc�A�1*

loss��;+D�J       �	'M�Xc�A�1*

lossܵ�<2P^�       �	I�M�Xc�A�1*

loss�0=�$�       �	VN�Xc�A�1*

loss���;��`6       �	��N�Xc�A�1*

loss��=���       �	��O�Xc�A�1*

lossi�<�a��       �	�%P�Xc�A�1*

loss��=���N       �	J�P�Xc�A�1*

loss�� <�4��       �	��Q�Xc�A�1*

lossZ�y;��3�       �	O R�Xc�A�1*

loss��<X���       �	z�R�Xc�A�1*

loss��<i>       �	UhS�Xc�A�1*

lossxF�<�R       �	�T�Xc�A�1*

lossz�A= �4       �	R�T�Xc�A�1*

lossE�o<T�,�       �	!YU�Xc�A�1*

loss��*<	�%�       �	��U�Xc�A�1*

loss�D={��       �	��V�Xc�A�1*

loss:��;��mg       �	M2W�Xc�A�1*

lossK�;�}'�       �	��W�Xc�A�1*

lossV��;����       �	mqX�Xc�A�1*

loss`�<�g�       �	NY�Xc�A�1*

loss/6;G�?n       �	�Y�Xc�A�1*

loss�,n;���       �	LRZ�Xc�A�1*

loss��<T�0       �	��Z�Xc�A�1*

loss[J;�5�<       �	.�[�Xc�A�1*

loss�x;=F�       �	 b\�Xc�A�1*

loss}Tk<i���       �	�]�Xc�A�1*

loss3/<-U       �	��]�Xc�A�1*

loss���<_�       �	�<^�Xc�A�1*

lossZQ�<��O(       �	7�^�Xc�A�1*

loss��r<��.|       �	�u_�Xc�A�1*

loss��;�Mc       �	�`�Xc�A�1*

lossiB<���       �	��`�Xc�A�1*

lossS��;��Y~       �	g~a�Xc�A�1*

loss\�;�si       �	�b�Xc�A�1*

lossK�<�0�F       �	��b�Xc�A�1*

loss�A�<X�X�       �	iSc�Xc�A�1*

lossH�0<8F�e       �	F�c�Xc�A�1*

loss��2=v�ư       �	@�d�Xc�A�1*

loss��<����       �	�#e�Xc�A�1*

loss�� <W       �	=�e�Xc�A�1*

lossͳ�;�ɟ<       �	�nf�Xc�A�1*

losse�/<�_:       �	kEg�Xc�A�1*

loss��:y�d       �	��g�Xc�A�1*

loss�X<�bF�       �	��h�Xc�A�1*

loss�u;���       �	�0i�Xc�A�1*

loss���<�2Q       �	8�i�Xc�A�1*

loss��I;���       �	Ttj�Xc�A�1*

loss�[B:� _       �	{k�Xc�A�1*

lossa��;�]�       �	�k�Xc�A�1*

lossŰF<��VR       �	�Fl�Xc�A�1*

loss��;���y       �	r�l�Xc�A�1*

loss3��<*k*]       �	�m�Xc�A�1*

loss�&<=��FD       �	�'n�Xc�A�1*

loss��;���x       �	5�n�Xc�A�1*

lossK.=�K�       �	 �o�Xc�A�1*

loss2�<p��       �	�{p�Xc�A�1*

lossQ��<!��i       �	KXq�Xc�A�1*

loss�E)<E�a9       �	z r�Xc�A�1*

lossa}�9E�t�       �	&�r�Xc�A�1*

loss��=qX�Z       �	-^s�Xc�A�1*

loss���;�-�}       �	st�Xc�A�1*

loss���=͇(       �	p�t�Xc�A�1*

loss�Ɣ;�=��       �	�u�Xc�A�1*

loss�k<]i>       �	�Qv�Xc�A�1*

loss�;��v&       �	�w�Xc�A�1*

loss�<����       �	]�w�Xc�A�1*

lossNbf;���       �	_Ex�Xc�A�1*

losse�Y=�0�       �	��x�Xc�A�1*

loss<a9<'ڶ#       �	�y�Xc�A�1*

loss��:�=�       �	�$z�Xc�A�1*

loss��=�*Z       �	��z�Xc�A�1*

loss��<�%l�       �	�\{�Xc�A�1*

losss�<����       �	��|�Xc�A�1*

lossST=-[��       �	�(}�Xc�A�1*

lossf�;<���       �	r�}�Xc�A�1*

loss;n�;�~aj       �	F[~�Xc�A�1*

loss��;Ep�       �	��~�Xc�A�1*

loss|�<ɺ�J       �	:��Xc�A�1*

loss%0�;�d�       �	�+��Xc�A�1*

loss��C<�
       �	qƀ�Xc�A�1*

lossә <(�\i       �	R_��Xc�A�1*

loss�֥;��       �	���Xc�A�1*

loss K|<,u�1       �	����Xc�A�1*

loss�;?M��       �	����Xc�A�1*

lossQx�<�a       �	�H��Xc�A�1*

loss�@�=q/d�       �	I�Xc�A�1*

loss���;Y!�U       �	����Xc�A�1*

loss�<o���       �	~5��Xc�A�1*

lossh�<Z�~       �	\䆗Xc�A�1*

loss��<ui -       �	����Xc�A�1*

loss]�%<�j�       �	�'��Xc�A�1*

loss42�;3��M       �	�B��Xc�A�1*

loss��.<:."�       �	nۉ�Xc�A�1*

loss�;�c�}       �	�q��Xc�A�1*

loss�;���       �	���Xc�A�1*

lossxd�:���4       �	����Xc�A�1*

loss�a�;o%{�       �	�h��Xc�A�1*

loss��E;�C��       �	���Xc�A�1*

loss��;�H��       �	<��Xc�A�1*

lossݩ�:�%��       �	ޯ��Xc�A�1*

loss@�<��w�       �	6���Xc�A�1*

loss�Bc;_��|       �	�[��Xc�A�1*

loss��<<��&        �	���Xc�A�1*

loss��;F��       �	'���Xc�A�1*

lossO�:�i�X       �	K[��Xc�A�1*

loss�Up;��M�       �	M���Xc�A�1*

lossx��;` �       �	�ē�Xc�A�2*

loss���;|��Y       �	 _��Xc�A�2*

loss}	.=u���       �	���Xc�A�2*

loss?	=_�       �	K���Xc�A�2*

loss�:-;n��Y       �	�(��Xc�A�2*

loss��T<%:�       �	�Ŗ�Xc�A�2*

lossڡ <�O��       �	$`��Xc�A�2*

loss�z�<��       �	����Xc�A�2*

lossE<;@���       �	u���Xc�A�2*

lossI�;��8�       �	K?��Xc�A�2*

loss���;/ä       �	nޙ�Xc�A�2*

lossM˼<i\�       �	yu��Xc�A�2*

lossq=���?       �	R��Xc�A�2*

loss�-�<��4�       �	�曗Xc�A�2*

lossv'�<�O�_       �	�z��Xc�A�2*

loss�2�<�s�Z       �	���Xc�A�2*

lossܭ�<�1��       �	q���Xc�A�2*

loss��<r�C       �	�H��Xc�A�2*

loss-	�:���       �	&���Xc�A�2*

loss��m=Ä��       �	����Xc�A�2*

loss�A<6�h�       �	X��Xc�A�2*

loss%�P<X+v$       �	W���Xc�A�2*

lossϏ�<1W��       �	�B��Xc�A�2*

lossS��9P�       �	��Xc�A�2*

loss@��<�B�       �	ۊ��Xc�A�2*

loss��z<�kPY       �	�&��Xc�A�2*

loss�Ԭ;��Y�       �	&ã�Xc�A�2*

loss���;��&Y       �	�f��Xc�A�2*

loss��h<�)��       �	���Xc�A�2*

loss_	;���)       �	ѭ��Xc�A�2*

loss,�:� 'M       �	�J��Xc�A�2*

loss[E=����       �	6覗Xc�A�2*

loss�h;Ia�}       �	z���Xc�A�2*

loss�7X;Uר�       �	�4��Xc�A�2*

loss6�=	P��       �	Vը�Xc�A�2*

loss�<m��       �	����Xc�A�2*

lossW?�<�@4       �	4��Xc�A�2*

loss���;@T@�       �	�ت�Xc�A�2*

loss� <���       �	�o��Xc�A�2*

lossEAA<ƚ�       �	p��Xc�A�2*

loss�;����       �	����Xc�A�2*

loss�v4;۝1       �	�Q��Xc�A�2*

loss\;=��!       �	Xc�A�2*

lossr`<ܰF�       �	?���Xc�A�2*

loss0ޠ<¦�       �	.��Xc�A�2*

loss_(P<@��       �	 į�Xc�A�2*

loss��<@�l       �	![��Xc�A�2*

loss�
N<���       �	bh��Xc�A�2*

loss]s;xk�       �	9���Xc�A�2*

loss��;�e�U       �	aq��Xc�A�2*

loss�<�;6���       �	����Xc�A�2*

lossT6=@�5�       �	_y��Xc�A�2*

loss7HE;#��       �	�$��Xc�A�2*

loss��;T=�A       �	�϶�Xc�A�2*

lossQh�;a-�       �	�η�Xc�A�2*

lossM�.>^e�       �	΋��Xc�A�2*

loss��n<.�"       �	J&��Xc�A�2*

loss׬;�[�       �	�Թ�Xc�A�2*

loss �_<��        �	����Xc�A�2*

lossS<��P�       �	�'��Xc�A�2*

loss�L0<jA�       �	fڻ�Xc�A�2*

loss!F�;�E<W       �	g~��Xc�A�2*

loss��;7ئ]       �	{.��Xc�A�2*

loss��J;�B��       �	+i��Xc�A�2*

loss���<+���       �	���Xc�A�2*

loss`?�;t��       �	���Xc�A�2*

lossџ�<��Y�       �	;���Xc�A�2*

loss�&,<INN       �	�G��Xc�A�2*

loss���<�<��       �	h!Xc�A�2*

loss��o; �i|       �	Q�Xc�A�2*

loss�֒;�       �	v�×Xc�A�2*

loss2x=��X�       �	�CėXc�A�2*

loss%�=T!��       �	�ŗXc�A�2*

loss;+�b       �	W"ƗXc�A�2*

loss�}r=���       �	�ƗXc�A�2*

loss/��;��2-       �	��ǗXc�A�2*

loss��<[�D�       �	BwȗXc�A�2*

loss}h�:5��b       �	�ɗXc�A�2*

loss�|;�{�v       �	�2ʗXc�A�2*

loss��;�ց       �	��ʗXc�A�2*

loss�'&=.Gb�       �	��˗Xc�A�2*

lossq�L=���       �	O]̗Xc�A�2*

loss� =�ך       �	 ͗Xc�A�2*

loss�B<�%C       �	��͗Xc�A�2*

loss1�[;�(��       �	JzΗXc�A�2*

loss�3O;e���       �	ϗXc�A�2*

loss�C<na        �	i�ϗXc�A�2*

lossLmC==��k       �	�uЗXc�A�2*

lossO�.<��`       �	��їXc�A�2*

loss��^<P+�*       �	�SҗXc�A�2*

loss)r�:��       �	R�ӗXc�A�2*

loss��J;.�EN       �	�ԗXc�A�2*

loss1�.=-�y       �	y�ԗXc�A�2*

lossqA�;/,�       �	H֗Xc�A�2*

loss��-<0p�       �	s�֗Xc�A�2*

loss�<���n       �	pחXc�A�2*

loss�a<�Mo       �	ؗXc�A�2*

loss�=�^�       �	�.ٗXc�A�2*

loss=r�<gy�       �	��ٗXc�A�2*

loss��<����       �	P�ڗXc�A�2*

loss}ס<vU�}       �	LOۗXc�A�2*

loss��<���%       �	��ۗXc�A�2*

lossq*:�&�+       �	9�ܗXc�A�2*

loss-� <�;�       �	dvݗXc�A�2*

lossV
u;]��!       �	�ޗXc�A�2*

lossko<$#��       �	�ޗXc�A�2*

loss��;��7       �	5yߗXc�A�2*

loss�^�<�b3�       �	�_��Xc�A�2*

lossm��;$�T       �	p�Xc�A�2*

lossX��<���R       �	���Xc�A�2*

lossO�;���       �	(c�Xc�A�2*

loss�PQ:&���       �	K�Xc�A�2*

loss�P<<ڇ�       �	կ�Xc�A�2*

lossM��:Qf��       �	�M�Xc�A�2*

loss�$�=���       �	P��Xc�A�2*

lossQۅ<dn��       �	��Xc�A�2*

lossV��<�B��       �	�G�Xc�A�2*

loss!f=}��       �	���Xc�A�2*

loss	�+<�e�       �	��Xc�A�2*

loss��;�~�v       �	d�Xc�A�2*

loss�><�d��       �	���Xc�A�2*

lossT� =\��;       �	>[�Xc�A�2*

loss�'�<OL)       �	L��Xc�A�2*

loss��9��M�       �	W��Xc�A�2*

loss-@\<h�If       �	�-�Xc�A�2*

loss��K=ݵ�N       �	2��Xc�A�2*

loss}3�;LG��       �	0��Xc�A�2*

loss�G<�2�       �	ZF�Xc�A�2*

loss4�W;��       �	��Xc�A�3*

loss/?�=�P}r       �	�"�Xc�A�3*

losso�<��]�       �	���Xc�A�3*

loss�;�ے6       �	���Xc�A�3*

lossn��<?6f*       �	�~�Xc�A�3*

loss���9����       �	_��Xc�A�3*

lossP{9��H�       �	��Xc�A�3*

loss1�<,�i�       �	_@��Xc�A�3*

loss�;Ӂ�       �	R��Xc�A�3*

loss�C�:X`��       �	����Xc�A�3*

loss#�H<���       �	�,��Xc�A�3*

loss�9]���       �	�A��Xc�A�3*

lossX�:�@f       �	iQ��Xc�A�3*

lossI-�<��Ҿ       �	���Xc�A�3*

loss���8�@��       �	����Xc�A�3*

loss&�9<2V!       �	�W��Xc�A�3*

loss'
�9M��
       �	'���Xc�A�3*

loss���;�uԫ       �	����Xc�A�3*

lossy<l��       �	D��Xc�A�3*

loss�><����       �	(���Xc�A�3*

lossz;�Y�$       �	����Xc�A�3*

loss�h'=�U%�       �	L���Xc�A�3*

lossʵ�9
_A�       �	[} �Xc�A�3*

loss,�=�Dm0       �	�K�Xc�A�3*

lossW�T;���P       �	���Xc�A�3*

lossݨ�;(R�`       �	<��Xc�A�3*

lossm�^<�@       �	���Xc�A�3*

lossq�;�Ƶ`       �	�[�Xc�A�3*

loss3��<_�
       �	���Xc�A�3*

lossi(�<'��e       �	��Xc�A�3*

loss��;�<��       �	�H�Xc�A�3*

loss�^�<0U�       �	@��Xc�A�3*

loss�ɺ<�E��       �	���Xc�A�3*

loss�
�<��*	       �	��Xc�A�3*

loss�h�;�1SU       �	ע	�Xc�A�3*

loss]s�;S�6�       �	�;
�Xc�A�3*

loss*�l=��nG       �	�
�Xc�A�3*

loss�<��       �	�o�Xc�A�3*

loss!�<��Ŋ       �	��Xc�A�3*

loss��<��d�       �	��Xc�A�3*

loss�`=<ӂ_       �	�H�Xc�A�3*

lossfx�;�7�;       �	���Xc�A�3*

loss��;��Y�       �	��Xc�A�3*

loss�}�<l��       �	���Xc�A�3*

loss�+�<�-��       �	G�Xc�A�3*

loss'/<Ώ�       �	,��Xc�A�3*

loss�O<�c�        �	G��Xc�A�3*

lossWP<}�Ղ       �	.�Xc�A�3*

loss��;�!2#       �	�Xc�A�3*

loss�:�Þ�       �	��Xc�A�3*

losskS�<���       �	�C�Xc�A�3*

loss��<�+ �       �	�D�Xc�A�3*

loss�M=`���       �	��Xc�A�3*

loss}X<fS�m       �	���Xc�A�3*

lossd`G<�0�       �	6�Xc�A�3*

loss�@K:�A�       �	X��Xc�A�3*

loss�+�;$4&       �	�b�Xc�A�3*

loss�;<̭��       �	���Xc�A�3*

loss���;��%X       �	>��Xc�A�3*

lossF�<����       �	�*�Xc�A�3*

loss</�H       �	���Xc�A�3*

loss,ċ;���-       �	uU�Xc�A�3*

loss ��9%j8       �	>�Xc�A�3*

loss���:u�/2       �	���Xc�A�3*

loss��#=�L-�       �	�w�Xc�A�3*

loss��;��L�       �	�
�Xc�A�3*

lossx;SnB       �	���Xc�A�3*

lossXVe;]��'       �	5�Xc�A�3*

loss@ *=随C       �	���Xc�A�3*

loss�q0;�FM       �	Na �Xc�A�3*

loss�r<v!}h       �	� �Xc�A�3*

loss�:�b       �	�!�Xc�A�3*

loss���;�j�       �	�a"�Xc�A�3*

loss%��;+�2�       �	�`@�Xc�A�3*

loss��;5��       �	�<A�Xc�A�3*

loss3�="v       �	B�Xc�A�3*

loss��<��1       �	��B�Xc�A�3*

loss)��;�� �       �	^�C�Xc�A�3*

lossus<���       �	��D�Xc�A�3*

loss� < �*       �	qsE�Xc�A�3*

loss�p�<����       �	�F�Xc�A�3*

loss��<)�P�       �	��F�Xc�A�3*

lossZ�;�u       �	�RG�Xc�A�3*

lossE��<��D�       �	J�G�Xc�A�3*

loss�4x<4:��       �	��H�Xc�A�3*

loss�c=�	       �	$I�Xc�A�3*

loss%=�y8       �	ػI�Xc�A�3*

loss���;Ƨ��       �	�hJ�Xc�A�3*

loss�W�=��Ю       �	�K�Xc�A�3*

loss�ǒ9�3p       �	�K�Xc�A�3*

loss��<ܯ��       �	�4L�Xc�A�3*

loss�#�;'�C�       �	��L�Xc�A�3*

loss�U<���       �	.tM�Xc�A�3*

losst�;0       �	�N�Xc�A�3*

loss0��<���l       �	�O�Xc�A�3*

loss
�B<)�^       �	׿O�Xc�A�3*

loss�A=�fe�       �	lP�Xc�A�3*

loss�<W�;       �	Q�Xc�A�3*

loss�K&;I�       �	��Q�Xc�A�3*

loss;��;�2u�       �	w�R�Xc�A�3*

loss=��<97M       �	�'S�Xc�A�3*

loss֠I=��U�       �	��S�Xc�A�3*

loss�.�<Ҕo       �	x{T�Xc�A�3*

loss�k�;���s       �	�U�Xc�A�3*

lossu�;���       �	&�U�Xc�A�3*

lossQ;$QH�       �	�lV�Xc�A�3*

loss?�@=�e�z       �	�W�Xc�A�3*

losst�9;�%��       �	��W�Xc�A�3*

loss�n=�T��       �	&RX�Xc�A�3*

losse��;�z�d       �	��X�Xc�A�3*

loss��?<�-��       �	ԜY�Xc�A�3*

lossX�)=RN8       �	�BZ�Xc�A�3*

loss��<�:�k       �	a�Z�Xc�A�3*

loss���<�|��       �	+�[�Xc�A�3*

loss���:�0       �	�/\�Xc�A�3*

loss�H:<8Q�       �	c�\�Xc�A�3*

lossĔ�<�fn�       �	�p]�Xc�A�3*

lossi��<��cL       �	r^�Xc�A�3*

loss�P�;jz�H       �	�z_�Xc�A�3*

lossn�_;�{
F       �	�`�Xc�A�3*

lossL�<��)T       �	��a�Xc�A�3*

loss#ϣ;�|�y       �	q<b�Xc�A�3*

loss.��9��D�       �	�b�Xc�A�3*

loss��:GS��       �	�c�Xc�A�3*

lossl��<��       �	b1d�Xc�A�3*

loss�	<�-��       �	�e�Xc�A�3*

loss�BX>�/��       �	��e�Xc�A�3*

loss�h=��Jt       �	Yf�Xc�A�3*

loss��;�QI�       �	��f�Xc�A�4*

lossq!D<T�_       �	��g�Xc�A�4*

lossϪ;�r��       �	�@h�Xc�A�4*

loss���<��ټ       �	!�h�Xc�A�4*

loss��-<��%.       �	&�i�Xc�A�4*

lossX��;��       �	�pj�Xc�A�4*

loss(;�!sy       �	�k�Xc�A�4*

loss*�o:��B=       �	��k�Xc�A�4*

losse�<5.�M       �	jml�Xc�A�4*

loss(��;*F�       �	�m�Xc�A�4*

loss��;�j�       �	��m�Xc�A�4*

loss�@=0��       �	Jn�Xc�A�4*

lossw�:B�c�       �	��n�Xc�A�4*

loss��;oZ�       �	Œo�Xc�A�4*

lossL��<H���       �	�1p�Xc�A�4*

loss@;[<srͫ       �	��p�Xc�A�4*

loss!/=>\�       �	��q�Xc�A�4*

lossA5�<�Ha       �	�Or�Xc�A�4*

loss��1;ǈ       �	40s�Xc�A�4*

loss/��;Ex@$       �	� t�Xc�A�4*

loss��+;d�       �	�Vu�Xc�A�4*

loss�7�:ж��       �	xEv�Xc�A�4*

loss�r�<"y=       �	Aw�Xc�A�4*

loss�W=����       �	D�w�Xc�A�4*

loss6=d��/       �	�y�Xc�A�4*

loss)��<p`d       �	�:z�Xc�A�4*

loss׿�<�lt       �	��z�Xc�A�4*

loss�\<��
e       �	x�{�Xc�A�4*

loss��3:�O�c       �	�|�Xc�A�4*

loss��<`��       �	��}�Xc�A�4*

loss���:�|A       �	k~�Xc�A�4*

lossw|=���       �	�|�Xc�A�4*

loss�E[<��v       �	k���Xc�A�4*

loss;�n<��(       �	h\��Xc�A�4*

loss]�~<`R��       �	in��Xc�A�4*

loss���;� �       �	�*��Xc�A�4*

loss\i=��       �	�Ń�Xc�A�4*

loss/�;����       �	^e��Xc�A�4*

lossI� <��o�       �	����Xc�A�4*

loss}3�<�k#[       �	$���Xc�A�4*

lossw �;cE3�       �	f��Xc�A�4*

loss��=���       �	��Xc�A�4*

loss<��<��       �	����Xc�A�4*

loss�{Z=�\�       �	iQ��Xc�A�4*

loss�aw;?��?       �	t수Xc�A�4*

lossMX�;Y�M�       �	/���Xc�A�4*

loss���:�ŗ       �	�=��Xc�A�4*

loss�ː<��lv       �	݊�Xc�A�4*

lossD�<��Hb       �	�}��Xc�A�4*

losssA�<"%A       �	��Xc�A�4*

loss*��;�K�       �	ď�Xc�A�4*

loss���:��S�       �	Nc��Xc�A�4*

loss:��l       �	/��Xc�A�4*

loss�=�_�G       �	����Xc�A�4*

loss��;cv�F       �	�J��Xc�A�4*

loss�H=����       �	.㏘Xc�A�4*

loss�w�;���i       �	����Xc�A�4*

loss�(<;W1�       �	1B��Xc�A�4*

loss��;Ց&�       �	�ؑ�Xc�A�4*

lossA �<2�{�       �	ep��Xc�A�4*

loss<�G;5�xr       �	���Xc�A�4*

loss���<@�I�       �	���Xc�A�4*

loss0='B�       �	�6��Xc�A�4*

loss�v�;�r?M       �	�ϔ�Xc�A�4*

loss|ϴ<�*       �	�u��Xc�A�4*

lossī�;g>�       �	���Xc�A�4*

loss1�<���       �	 ���Xc�A�4*

loss��<����       �	pC��Xc�A�4*

lossc�;�I       �	����Xc�A�4*

loss��:���	       �	�~��Xc�A�4*

loss��<�p�       �	��Xc�A�4*

loss���;�ڔ4       �	񽙘Xc�A�4*

lossHl=��ć       �	X��Xc�A�4*

loss��<<�Ľ       �	���Xc�A�4*

loss�=����       �	���Xc�A�4*

loss�}�<��#�       �	�9��Xc�A�4*

loss��;?�g�       �	�М�Xc�A�4*

loss\Zx<��t3       �	r��Xc�A�4*

loss���;<�y       �	���Xc�A�4*

loss�=��|       �	t���Xc�A�4*

loss;j�;|�`�       �	�O��Xc�A�4*

lossעE=��ܲ       �	���Xc�A�4*

loss���;�~       �	K���Xc�A�4*

loss�h�::��       �	^G��Xc�A�4*

loss;$s<{7��       �	�塘Xc�A�4*

loss���;�-+       �	Z��Xc�A�4*

loss~,<�#�I       �	1$��Xc�A�4*

loss���;�"�L       �	Ͼ��Xc�A�4*

lossH3=���       �	W��Xc�A�4*

loss�$�<����       �	���Xc�A�4*

loss��[<9�{       �	����Xc�A�4*

lossR_3;��{�       �	4���Xc�A�4*

loss���;����       �	���Xc�A�4*

lossqw�;%SI       �	����Xc�A�4*

loss�(	=��l       �	b��Xc�A�4*

loss�!<ֳ��       �	e ��Xc�A�4*

loss*M=?��       �	,���Xc�A�4*

loss�g<g|;       �	�=��Xc�A�4*

loss���;�h�*       �	�٪�Xc�A�4*

loss:�;���       �	)x��Xc�A�4*

lossF��9��       �	D��Xc�A�4*

loss:�;6�V       �	 ���Xc�A�4*

loss`փ=���D       �	�H��Xc�A�4*

loss�q�<����       �	�Xc�A�4*

lossA�<e�5�       �	���Xc�A�4*

loss
^=/��p       �	�<��Xc�A�4*

loss��<�}\s       �	'���Xc�A�4*

loss�M�=�&       �	�ɲ�Xc�A�4*

losst� :���       �	(��Xc�A�4*

lossLr�=��n       �	���Xc�A�4*

loss�{�:��<8       �	Di��Xc�A�4*

loss��+;7�<       �	�e��Xc�A�4*

lossM�k;�Z�       �	*��Xc�A�4*

loss�:�;��ʢ       �	ܻ��Xc�A�4*

lossT�m=�r       �	K[��Xc�A�4*

lossTB�<���       �	
���Xc�A�4*

loss۫y;���       �	"޹�Xc�A�4*

loss�9:�T��       �	s��Xc�A�4*

loss�M�<��/�       �	���Xc�A�4*

loss�\�:�>c       �	���Xc�A�4*

loss���;���       �	s���Xc�A�4*

lossx��<��,;       �	�8��Xc�A�4*

loss���<�o�&       �	����Xc�A�4*

loss���<x�s       �	0���Xc�A�4*

loss,�=;8�
       �	75��Xc�A�4*

loss���99%Y       �	���Xc�A�4*

loss��"<�H       �	;���Xc�A�4*

loss�U�;{�1�       �	�K��Xc�A�4*

loss��<;�*�       �	b���Xc�A�5*

loss�|8<\�˴       �	��Xc�A�5*

loss�<�I/s       �	�SØXc�A�5*

loss�J=?:N�       �	��ØXc�A�5*

loss��5=�i�       �	�ĘXc�A�5*

loss�3�<��y       �	|GŘXc�A�5*

loss`��<��       �	+�ŘXc�A�5*

loss
-�<� {�       �	�vƘXc�A�5*

loss��[<�67a       �	ǘXc�A�5*

lossy4<�7�       �	��ǘXc�A�5*

loss�W$<�F	       �	�MȘXc�A�5*

lossc�;/�}       �	H�ȘXc�A�5*

lossX̹;���n       �	�zɘXc�A�5*

lossqm\;p�y�       �	�`ʘXc�A�5*

loss�j|<6�B�       �	��ʘXc�A�5*

loss��<�?�K       �	p�˘Xc�A�5*

lossVB�;� 	g       �	74̘Xc�A�5*

loss� <�O0�       �	��̘Xc�A�5*

loss.RJ;]�       �	Fy͘Xc�A�5*

loss��=���       �	�ΘXc�A�5*

loss�J�;�.�        �	��ΘXc�A�5*

loss��$<��       �	�TϘXc�A�5*

loss]A<'��z       �	��ϘXc�A�5*

loss��O<$�/       �	G�ИXc�A�5*

loss�G�=_       �	�7јXc�A�5*

lossCn!=&ȅ�       �	�'ԘXc�A�5*

loss�aN<�1x�       �	��ԘXc�A�5*

loss���;�,�N       �	�n՘Xc�A�5*

loss �j<ڀǲ       �	�֘Xc�A�5*

loss���:��X       �	��֘Xc�A�5*

loss��<��`       �	xCטXc�A�5*

lossO�,=�{��       �	��טXc�A�5*

loss�*=h8�c       �	�ؘXc�A�5*

loss��k<oy&�       �	�Y٘Xc�A�5*

loss�Q*<��ʆ       �	�ژXc�A�5*

loss�==;�c       �	�ژXc�A�5*

lossR8�<׋��       �	�OۘXc�A�5*

loss�u,<�b�C       �	�ۘXc�A�5*

loss���;��       �	g�ܘXc�A�5*

loss�]P<`��{       �	�4ݘXc�A�5*

loss\�<X�&�       �	��ݘXc�A�5*

loss�ǡ:٭h�       �	^�ޘXc�A�5*

loss���:%�q*       �	hߘXc�A�5*

lossc4?=�#       �	��ߘXc�A�5*

loss'�!<��|z       �	����Xc�A�5*

loss� P<�UU�       �	�1�Xc�A�5*

loss3�:�"��       �	4��Xc�A�5*

losss�W<Qfɇ       �	�y�Xc�A�5*

lossM��;/D��       �	%�Xc�A�5*

loss��<3(�w       �	(��Xc�A�5*

loss�Y2;B1\�       �	���Xc�A�5*

loss3x= ���       �	K>�Xc�A�5*

loss;"E=_o�a       �	T��Xc�A�5*

loss�W�<�)�J       �	���Xc�A�5*

loss�ѝ<S�J�       �	JA�Xc�A�5*

loss���<I�A�       �	=��Xc�A�5*

lossT:xܓn       �	ڏ�Xc�A�5*

loss�3�:�7�Z       �	�+�Xc�A�5*

loss�@&<~�       �	���Xc�A�5*

loss�?<y��K       �	�i�Xc�A�5*

loss|Bp=�@u:       �	y�Xc�A�5*

loss�k<��+J       �	z��Xc�A�5*

lossU�<��&m       �	�G�Xc�A�5*

lossT}�;�t=�       �	h��Xc�A�5*

lossVf1;���O       �	'��Xc�A�5*

lossi�/;J��_       �	I.�Xc�A�5*

loss�ޙ<Ч��       �	(��Xc�A�5*

lossAJ2=���@       �	�z�Xc�A�5*

loss8�=��lk       �	z�Xc�A�5*

loss|�;�/�       �	��Xc�A�5*

loss���;"�D�       �	�r�Xc�A�5*

lossͩ(:��"       �	��Xc�A�5*

lossx��;ܬt[       �	>��Xc�A�5*

loss�~�<�6�        �	b��Xc�A�5*

loss��;�-'       �	�A��Xc�A�5*

loss�k�<9��       �	M���Xc�A�5*

loss<�<f3�u       �	����Xc�A�5*

loss�e�<�=],       �	����Xc�A�5*

lossX͢<{�$�       �	$d��Xc�A�5*

loss�L<
���       �	K;��Xc�A�5*

loss��.=�`)       �	����Xc�A�5*

loss9D�<c       �	����Xc�A�5*

loss:*){�       �	Y��Xc�A�5*

loss��#=��T�       �	�^��Xc�A�5*

loss��;S�\�       �	M0��Xc�A�5*

loss�?�<�>s�       �	����Xc�A�5*

loss�<��A�       �	i��Xc�A�5*

loss��=Z�K�       �	����Xc�A�5*

loss
2�<�7µ       �	����Xc�A�5*

lossrKE=��ײ       �	Q� �Xc�A�5*

loss*s<���        �	]�Xc�A�5*

loss��<8�^A       �	��Xc�A�5*

loss�+�<7�       �	��Xc�A�5*

lossX�(<� ��       �	���Xc�A�5*

loss�
�;z��       �	a��Xc�A�5*

loss �;r�,       �	=D�Xc�A�5*

loss|q�<��       �	P��Xc�A�5*

loss�^�: ��       �	+��Xc�A�5*

loss��;9�R       �	`r�Xc�A�5*

loss��;�_`�       �	�	�Xc�A�5*

loss�m <UJa<       �	'�	�Xc�A�5*

loss�̦;dC%       �	se
�Xc�A�5*

loss���=��c       �	��Xc�A�5*

loss�8�;#�6       �	��Xc�A�5*

loss�;28!       �	�V�Xc�A�5*

lossI�N<�\�       �	D��Xc�A�5*

loss�v<���w       �	��Xc�A�5*

loss��9���       �	���Xc�A�5*

loss8�b;��$       �	fi�Xc�A�5*

loss?Ϛ:	e       �	%�Xc�A�5*

loss��:�x�       �	ͱ�Xc�A�5*

loss @�<���       �	jP�Xc�A�5*

loss6	=>�R       �	�Q�Xc�A�5*

lossH�;Kau�       �	��Xc�A�5*

loss��;O&       �	���Xc�A�5*

loss7%�<~�J'       �	�-�Xc�A�5*

loss{�w=n3�i       �	V��Xc�A�5*

loss��M<|�N�       �	���Xc�A�5*

loss�];�R�4       �	�)�Xc�A�5*

lossqj�<_�       �	?��Xc�A�5*

losstY�<�E�       �	���Xc�A�5*

lossdU<V�t       �	�c�Xc�A�5*

loss
z=DG�       �	��Xc�A�5*

loss=e$;��W�       �	c��Xc�A�5*

lossZRh<���]       �	;�Xc�A�5*

lossXp< zw�       �	U��Xc�A�5*

loss��Z<gN�       �	��Xc�A�5*

lossI��;�Xr       �	d!�Xc�A�5*

lossʥ�;1Jy       �	���Xc�A�6*

lossP��<��Y�       �	ff�Xc�A�6*

lossƜ�<�s��       �	��Xc�A�6*

lossD�;F�\�       �	¢�Xc�A�6*

loss�ˆ<�{�       �	I �Xc�A�6*

loss�9R;$V=�       �	�� �Xc�A�6*

lossq{�;
3�h       �	��!�Xc�A�6*

loss�#@=]�H�       �	,"�Xc�A�6*

loss&�;[C2       �	�"�Xc�A�6*

loss��M<E1V�       �	t}#�Xc�A�6*

loss�<\=w���       �	�!$�Xc�A�6*

lossR�<:��       �	�$�Xc�A�6*

loss�ׅ;�       �	Z%�Xc�A�6*

loss߽.=,��       �	��%�Xc�A�6*

loss�Y�<3��1       �	��&�Xc�A�6*

loss���<!-�       �	H5'�Xc�A�6*

loss��&<����       �	��'�Xc�A�6*

lossdv�<�d��       �	�y(�Xc�A�6*

lossE�M=���       �	3)�Xc�A�6*

losso�< ��z       �	W�)�Xc�A�6*

loss:�<	%�G       �	�V*�Xc�A�6*

loss! n;*���       �	(�*�Xc�A�6*

loss�T5;4.�3       �	>�+�Xc�A�6*

loss��#<����       �	�0,�Xc�A�6*

loss+�:�t�       �	H�,�Xc�A�6*

loss?5�<%*;�       �	�|-�Xc�A�6*

loss	��;�ɚ       �	p$.�Xc�A�6*

loss��;���q       �	{�.�Xc�A�6*

loss|X<f�o�       �	i/�Xc�A�6*

loss��o:j@jw       �	N
0�Xc�A�6*

loss�7y;�!       �	`�0�Xc�A�6*

loss��<��Y*       �	RG1�Xc�A�6*

loss��;@��x       �	C�1�Xc�A�6*

loss�l=R�Ҋ       �	�z2�Xc�A�6*

loss�_�<ށ��       �	�3�Xc�A�6*

loss�R�;�[�       �	b�3�Xc�A�6*

loss��=�Z,p       �	PU4�Xc�A�6*

loss3�;p�@       �	��4�Xc�A�6*

loss:Dl<?��;       �	�5�Xc�A�6*

loss(g�:���       �	�)6�Xc�A�6*

loss�t�<��       �	z�6�Xc�A�6*

loss��.<�}K       �	|�7�Xc�A�6*

loss��<��       �	�b8�Xc�A�6*

lossS��;�G�       �	d9�Xc�A�6*

loss�2;�K��       �	��9�Xc�A�6*

loss|1�<o,��       �	L8:�Xc�A�6*

loss�w;�G��       �	��:�Xc�A�6*

lossFl�;=A{       �	-|;�Xc�A�6*

loss�b�;�*��       �	C<�Xc�A�6*

losss��;��(       �	��<�Xc�A�6*

lossC!�;Xb6       �	�m=�Xc�A�6*

loss�';�
�C       �	Z>�Xc�A�6*

loss���<[R�I       �	8�>�Xc�A�6*

loss�=�Q�       �	�a?�Xc�A�6*

loss��e=���2       �	�@�Xc�A�6*

loss!R?=��(�       �	��@�Xc�A�6*

loss�$<�m�       �	�bA�Xc�A�6*

lossq�;f�6�       �	�hB�Xc�A�6*

lossx�`<{i |       �	�	C�Xc�A�6*

loss��<��ɶ       �	��C�Xc�A�6*

loss��:#D 5       �	?SD�Xc�A�6*

loss�߈;�N       �	P�D�Xc�A�6*

loss�J<��es       �	l�E�Xc�A�6*

loss|��;�	PV       �	�9F�Xc�A�6*

loss��;;I[       �	��F�Xc�A�6*

loss�S�;ߦ��       �	+nG�Xc�A�6*

lossZх=M�ĭ       �	BH�Xc�A�6*

loss���<���+       �	o�H�Xc�A�6*

loss�8<v��       �	+5I�Xc�A�6*

loss���;��q       �	��I�Xc�A�6*

loss��<
�       �	�bJ�Xc�A�6*

lossQg�<,��       �	�K�Xc�A�6*

loss���;-� �       �	�L�Xc�A�6*

lossj�;x71�       �	��L�Xc�A�6*

loss��D<�֝�       �	
hM�Xc�A�6*

loss_�<��       �	z N�Xc�A�6*

loss%W�;@�g�       �	�N�Xc�A�6*

loss�z:^'��       �	��O�Xc�A�6*

loss*�;ͽ�       �	�P�Xc�A�6*

loss�0<s��m       �	D�P�Xc�A�6*

lossX�<�_�       �	�eQ�Xc�A�6*

lossqy�;�G+n       �	+�Q�Xc�A�6*

loss��|<�ޞH       �	%�R�Xc�A�6*

loss�;��       �	�FS�Xc�A�6*

lossAŮ<�7       �	��S�Xc�A�6*

loss,Щ:.X�       �	��T�Xc�A�6*

loss��;*��G       �	�#U�Xc�A�6*

loss̃&;)Rj       �	�U�Xc�A�6*

lossA.<��I'       �	�VV�Xc�A�6*

lossV�;�֞�       �	�V�Xc�A�6*

losss��:TV�       �	'�W�Xc�A�6*

lossZII=
dͬ       �	OvX�Xc�A�6*

loss;l�;=)g�       �	wY�Xc�A�6*

lossn�x=�1��       �	m�Y�Xc�A�6*

loss��<�V��       �	�EZ�Xc�A�6*

loss=�f= ��       �	��Z�Xc�A�6*

loss�;[�F�       �	��[�Xc�A�6*

loss�SU<��=�       �	�7\�Xc�A�6*

loss,;�@�o       �	��\�Xc�A�6*

lossH�\<�Q�J       �	]�Xc�A�6*

loss;a�<�C`       �	�'^�Xc�A�6*

loss�5W;�"       �	j�^�Xc�A�6*

loss� �<�{�       �	Ք_�Xc�A�6*

loss-�=�(.�       �	�6`�Xc�A�6*

loss)xL<�l       �	g�`�Xc�A�6*

lossEQ�:%~
�       �	yua�Xc�A�6*

loss�a�:��(       �	b�Xc�A�6*

loss��<u=�       �	Ŭb�Xc�A�6*

loss�<�
��       �	�Zc�Xc�A�6*

lossT�<�R]u       �	w�c�Xc�A�6*

loss'|<9��1       �	��d�Xc�A�6*

loss_!�:���r       �	2se�Xc�A�6*

loss���<[A.J       �	�f�Xc�A�6*

lossp�=R��       �	^�f�Xc�A�6*

loss[��<5�|	       �	S^g�Xc�A�6*

loss�M�;-�i�       �	z h�Xc�A�6*

loss��+<���       �	R�h�Xc�A�6*

loss�<q�/       �	}>i�Xc�A�6*

loss$Z:E       �	^�i�Xc�A�6*

loss�e�;���       �	��j�Xc�A�6*

loss���;^���       �	�k�Xc�A�6*

loss�h�;��|N       �	��k�Xc�A�6*

loss7�<T"?       �	�^l�Xc�A�6*

loss�.�;��M       �	3�l�Xc�A�6*

lossWN�;�)�0       �	_�m�Xc�A�6*

loss(�;�D�0       �	�7n�Xc�A�6*

loss�<�f�B       �	��n�Xc�A�6*

lossS�.<�
@       �	n�o�Xc�A�6*

loss�<8Die       �	gCp�Xc�A�7*

loss��<�Ǟ       �	5�p�Xc�A�7*

loss��=�Ƃ6       �	:�q�Xc�A�7*

lossX��;��@�       �	|r�Xc�A�7*

loss�k<3'].       �	�$s�Xc�A�7*

loss�Wp<�Q��       �	#�s�Xc�A�7*

loss�q�<�Q�       �	�`t�Xc�A�7*

loss�A�=r1i�       �	�u�Xc�A�7*

lossH�;�f�       �	��u�Xc�A�7*

loss�I(<�`�       �	�Jv�Xc�A�7*

loss���;N���       �	��v�Xc�A�7*

loss�=�\       �	��w�Xc�A�7*

loss�b�<H1n�       �	@x�Xc�A�7*

loss�T3;��d�       �	��x�Xc�A�7*

loss,�<�*��       �	ʈy�Xc�A�7*

loss��;j
&x       �	/z�Xc�A�7*

loss��,<Fz�v       �	��z�Xc�A�7*

loss�Ņ;�(K'       �	�}{�Xc�A�7*

lossa
<}���       �	|(|�Xc�A�7*

loss\oh<py�       �	q�|�Xc�A�7*

loss�؄<h,i       �	.q}�Xc�A�7*

loss�U�;/[V1       �	�~�Xc�A�7*

lossmZ�:hC�:       �	��~�Xc�A�7*

loss���;�K(�       �	�J�Xc�A�7*

loss���<k޸|       �	���Xc�A�7*

loss2Z~;�       �	.���Xc�A�7*

lossD�=�.�       �	�0��Xc�A�7*

loss;��<�X       �	B́�Xc�A�7*

lossn��=w��       �	�m��Xc�A�7*

loss��k;n�       �	���Xc�A�7*

loss��};?��       �	���Xc�A�7*

loss�]W<'��       �	>[��Xc�A�7*

loss�m<���       �	����Xc�A�7*

loss��E;(��       �	I���Xc�A�7*

loss�=��y�       �	�<��Xc�A�7*

loss�&;k��       �	j݆�Xc�A�7*

loss���:1k��       �	x���Xc�A�7*

loss��:�T�&       �	O!��Xc�A�7*

loss��;k5Yu       �	ň�Xc�A�7*

loss�6�;j�e       �	�]��Xc�A�7*

loss�!_=%��d       �	��Xc�A�7*

loss���< �,       �	 ���Xc�A�7*

loss1B8;T(�       �	}��Xc�A�7*

loss�"�:J�V�       �	�֋�Xc�A�7*

loss�3<297�       �	�j��Xc�A�7*

lossܭ$;c��       �	`��Xc�A�7*

loss��'<ډ#       �	���Xc�A�7*

loss�!�<W�_       �	�;��Xc�A�7*

loss��0<�De       �	�ҏ�Xc�A�7*

loss}G�;'��       �	�u��Xc�A�7*

losst��:�       �	#��Xc�A�7*

lossL�<(���       �	禑�Xc�A�7*

lossw�<0y8�       �	%@��Xc�A�7*

loss���;l���       �	�Ԓ�Xc�A�7*

lossVa�<��=�       �	�w��Xc�A�7*

loss���<�Ͻ       �	���Xc�A�7*

lossW!�<����       �	����Xc�A�7*

loss�/=�Y       �	b��Xc�A�7*

lossɶ<:�R       �	����Xc�A�7*

loss�=a���       �	�Xc�A�7*

loss��;w/:2       �	�7��Xc�A�7*

loss|��<�^��       �	�̗�Xc�A�7*

loss��;;��       �	����Xc�A�7*

loss(�-;�}a~       �	�J��Xc�A�7*

loss=�C;��~p       �	��Xc�A�7*

loss�1h<Ys�       �	����Xc�A�7*

loss'�<��5       �	�:��Xc�A�7*

loss}g�<ù�U       �	&⛙Xc�A�7*

loss�A;dE��       �	񃜙Xc�A�7*

lossҨ�;R��       �	�#��Xc�A�7*

lossE�;�E�[       �	DÝ�Xc�A�7*

loss�:/�_       �	�b��Xc�A�7*

loss$�I<>ǡ       �	;���Xc�A�7*

loss��x:⎲�       �	����Xc�A�7*

loss& �<���9       �	���Xc�A�7*

loss�;x��9       �	�0��Xc�A�7*

loss��'<���!       �	�ڡ�Xc�A�7*

loss6>\<����       �	V��Xc�A�7*

loss���;�T�       �	�%��Xc�A�7*

lossZ<p��:       �	(���Xc�A�7*

lossvK�;��       �	�S��Xc�A�7*

loss��<+��9       �	(��Xc�A�7*

loss�N/;Nh�       �	�¥�Xc�A�7*

loss8��;b��       �	B_��Xc�A�7*

loss�S&;�]Q�       �		���Xc�A�7*

lossL98=��&�       �	!���Xc�A�7*

loss��<��       �	�(��Xc�A�7*

loss�z�<U�o�       �	�¨�Xc�A�7*

loss���;�4�P       �	�Z��Xc�A�7*

lossK5 <��.       �	Z�Xc�A�7*

loss�=$4��       �	׉��Xc�A�7*

lossB�<'c�7       �	,��Xc�A�7*

loss�k=<�qe       �	-ѫ�Xc�A�7*

loss78}:���       �	�l��Xc�A�7*

loss��L9}W�       �	c
��Xc�A�7*

loss/f�;d3a       �	a���Xc�A�7*

loss�?;���{       �	�>��Xc�A�7*

lossZr�:Sy��       �	�ޮ�Xc�A�7*

loss eJ<R��!       �	V���Xc�A�7*

loss�hV:���       �	�Ű�Xc�A�7*

loss��;mI|�       �	�c��Xc�A�7*

lossäP=9��       �	g��Xc�A�7*

loss ��9Y�5�       �	�'��Xc�A�7*

loss�ƫ7�>�       �	�,��Xc�A�7*

loss���9�&�-       �	vⴙXc�A�7*

loss*�;S���       �	�嵙Xc�A�7*

loss�� <���       �	J|��Xc�A�7*

loss°:�8t'       �	���Xc�A�7*

lossO(:���       �	󬷙Xc�A�7*

lossC40=n�d       �	�_��Xc�A�7*

loss_h:�%�       �	8���Xc�A�7*

loss:9=MF<       �	F���Xc�A�7*

loss|<:��       �	Va��Xc�A�7*

loss\v3<�~ڼ       �	���Xc�A�7*

lossΝ!=�'       �	ߣ��Xc�A�7*

loss�@�;�8�       �	�7��Xc�A�7*

lossQ��< ��       �	WѼ�Xc�A�7*

lossIgR;~NP}       �	bg��Xc�A�7*

loss�;�N�       �	����Xc�A�7*

lossK+�;�Xq�       �	���Xc�A�7*

loss���;v�       �	J%��Xc�A�7*

lossx��:[�2�       �	�⿙Xc�A�7*

loss�{<"�       �	�w��Xc�A�7*

lossۉ�;*�,       �	B
��Xc�A�7*

loss���=J���       �	؝��Xc�A�7*

loss���<&|v�       �	�8Xc�A�7*

loss�9;Z�       �	��Xc�A�7*

loss�*�:%�/       �	iÙXc�A�7*

loss��<j�       �	IęXc�A�8*

lossJ��;�D��       �	m�ęXc�A�8*

lossF��;K;       �	�{řXc�A�8*

loss��<�կ       �	ƙXc�A�8*

loss��h<���       �	ղƙXc�A�8*

lossX�:�f��       �	{OǙXc�A�8*

loss�<T�[�       �	p�ǙXc�A�8*

loss���<����       �	�șXc�A�8*

lossh�<�W       �	�WəXc�A�8*

lossEZ;���*       �	��əXc�A�8*

lossE��<F��,       �	;�ʙXc�A�8*

loss��F= ��       �	�%˙Xc�A�8*

loss�F%=���[       �	��˙Xc�A�8*

loss�8�; ��       �	P�̙Xc�A�8*

lossD7<��-�       �	�*͙Xc�A�8*

loss�,s<W�y*       �	��͙Xc�A�8*

loss/�;dK�^       �	[�ΙXc�A�8*

loss���:�i��       �	��ϙXc�A�8*

loss��U;7N�T       �	)^ЙXc�A�8*

loss ��;B�+O       �	�.љXc�A�8*

loss}L�<��       �	-�љXc�A�8*

lossȺ)<,ϡ_       �	�`ҙXc�A�8*

loss�I:N� �       �	��ҙXc�A�8*

loss�T0;�J�M       �	��әXc�A�8*

lossU�;���       �	=)ԙXc�A�8*

lossR*"=(��       �	��ԙXc�A�8*

loss���;�S��       �	oՙXc�A�8*

loss���<%��       �	�$֙Xc�A�8*

loss\
 <��k       �	�֙Xc�A�8*

loss(��:�E�       �	�yיXc�A�8*

loss��= /��       �	7ؙXc�A�8*

lossIxS<�6�       �	̶ؙXc�A�8*

loss���;2g�       �	�KٙXc�A�8*

lossE�N<��h�