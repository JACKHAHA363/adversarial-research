       �K"	  @�Yc�Abrain.Event:2��	�5�     Or�	��i�Yc�A"��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2��t*
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
seed2�ߧ*
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
seed2�ً
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
new_axis_mask *
ellipsis_mask *

begin_mask 
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
seed2���
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2�ݸ
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
seed2�Ǯ
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
seed2���*
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
shrink_axis_mask *
ellipsis_mask *

begin_mask *
end_mask
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
seed2���
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
 *  �B*
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
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0
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
valueB *
_output_shapes
: *
dtype0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
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
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"N�����     �>�	�*m�Yc�AJ��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2��t*
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2�ߧ
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2�ً*
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
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
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
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
seed2�ݸ*
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
seed2�Ǯ
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
:���������@*
seed2���
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
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
seed2���
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
 *  �B*
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
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: *
T0
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
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
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
7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
_output_shapes
: *
T0
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
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
T0*
out_type0*
_output_shapes
:
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
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
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*'
_output_shapes
:���������
*
T0
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
N*
T0*/
_output_shapes
:���������@*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
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
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
_output_shapes
:*
out_type0*
T0
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
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
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
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*
T0*/
_output_shapes
:���������@
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
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
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
_output_shapes
:*
out_type0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:@
�
Agradients/sequential_1/conv2d_1/convolution_grad/tuple/group_depsNoOpE^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputF^gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter
�
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*
T0
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
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *"
_class
loc:@conv2d_1/kernel*
shared_name 
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
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *
_output_shapes
: *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta2_power/readIdentitybeta2_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
j
zerosConst*
dtype0*&
_output_shapes
:@*%
valueB@*    
�
conv2d_1/kernel/Adam
VariableV2*
shared_name *
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
l
zeros_1Const*&
_output_shapes
:@*
dtype0*%
valueB@*    
�
conv2d_1/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_1/kernel*
shared_name *&
_output_shapes
:@*
shape:@
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
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
VariableV2*&
_output_shapes
:@@*
dtype0*
shape:@@*
	container *"
_class
loc:@conv2d_2/kernel*
shared_name 
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
zeros_5Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam_1
VariableV2*
shared_name *
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
	container *
dtype0* 
_class
loc:@conv2d_2/bias*
shared_name *
_output_shapes
:@*
shape:@
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
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
T
zeros_7Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam_1
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
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
	container *
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
b
zeros_9Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
�
dense_1/kernel/Adam_1
VariableV2*
shared_name *
shape:���*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
dtype0*
	container 
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
zeros_11Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam_1
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
dtype0*
_output_shapes
:	�
*
valueB	�
*    
�
dense_2/kernel/Adam_1
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
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
T0
U
zeros_14Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam
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
zeros_15Const*
_output_shapes
:
*
dtype0*
valueB
*    
�
dense_2/bias/Adam_1
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *
_class
loc:@dense_2/bias*
shared_name 
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
dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0*
use_locking( 
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
�
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonFgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
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
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""
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
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"�
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0�}�       ��-	bK��Yc�A*

lossH^@��?       ��-	����Yc�A*

loss�3@e�)F       ��-	0���Yc�A*

loss|�@x�       ��-	�D��Yc�A*

loss8@���       ��-	�랧Yc�A*

loss�@�&�       ��-	󑟧Yc�A*

loss�Z@�/       ��-	cE��Yc�A*

loss�@���p       ��-	(�Yc�A*

loss�@�%�       ��-	����Yc�A	*

loss�D
@�c|       ��-	@O��Yc�A
*

losso�@�6Nl       ��-	���Yc�A*

loss��?l��       ��-	eģ�Yc�A*

loss*�?����       ��-	Cs��Yc�A*

loss6�?�/ɠ       ��-	���Yc�A*

loss�f�?]Ed       ��-	�ʥ�Yc�A*

loss)��?2u�       ��-	zq��Yc�A*

loss�M�?���^       ��-	C��Yc�A*

lossjd�?*�(       ��-	����Yc�A*

lossi�?���^       ��-	{g��Yc�A*

loss�Т?�&!g       ��-	_��Yc�A*

loss��?�U�       ��-	)���Yc�A*

loss���?T�#f       ��-	\Y��Yc�A*

loss���?:�Л       ��-	�ī�Yc�A*

lossW�?߅Q       ��-	(c��Yc�A*

lossݰ�?�"��       ��-	>��Yc�A*

loss�k?�.�       ��-	M���Yc�A*

lossJ�k?�X�        ��-	�]��Yc�A*

losso��?�yH       ��-	����Yc�A*

loss��?����       ��-	����Yc�A*

loss��? T��       ��-	�9��Yc�A*

loss�8k?�f�f       ��-	�ְ�Yc�A*

loss�K?�k�       ��-	S���Yc�A *

loss��r?=p�M       ��-	K��Yc�A!*

loss�^z?�7p.       ��-	�岧Yc�A"*

loss�t?=�{       ��-	@���Yc�A#*

loss8��?��+�       ��-	�s��Yc�A$*

loss*$?mLѸ       ��-	���Yc�A%*

loss�l�? �M-       ��-	2���Yc�A&*

lossj��?�T       ��-	�c��Yc�A'*

loss��z?�)�       ��-	 ���Yc�A(*

lossR8?m`       ��-	���Yc�A)*

loss�?qc�^       ��-	R��Yc�A**

losss�E?S�$<       ��-	�6��Yc�A+*

losshA?й�       ��-	�ҹ�Yc�A,*

loss�.Q?���	       ��-	�n��Yc�A-*

loss��G?�R:	       ��-	S��Yc�A.*

losse�@?:�m       ��-	못�Yc�A/*

loss?��q       ��-	�D��Yc�A0*

lossv�?��x       ��-	LἧYc�A1*

lossN�+?�w'r       ��-	�ݽ�Yc�A2*

loss
-?b 9�       ��-	����Yc�A3*

loss]�,?[x�9       ��-	M��Yc�A4*

lossΈ?U��L       ��-	�꿧Yc�A5*

loss�\^?���       ��-	{���Yc�A6*

loss��>��g�       ��-	G���Yc�A7*

loss��	?��hU       ��-	*§Yc�A8*

loss3�?eew�       ��-	��§Yc�A9*

loss_�
?�-6�       ��-	�XçYc�A:*

loss��$?�n       ��-	�çYc�A;*

loss���>�<c       ��-	��ħYc�A<*

loss
��>a�       ��-	�&ŧYc�A=*

loss@?��p�       ��-	j�ŧYc�A>*

loss��
?�8h       ��-	�PƧYc�A?*

loss$�+?UUk       ��-	��ƧYc�A@*

loss��?O��       ��-	ޏǧYc�AA*

loss*��>��(e       ��-	{1ȧYc�AB*

loss�F<?�Y-�       ��-	�ȧYc�AC*

loss?>? ��$       ��-	�aɧYc�AD*

loss�N?��Y�       ��-	v�ɧYc�AE*

lossf�
?R�D       ��-	ђʧYc�AF*

loss�4%?d��       ��-	y"˧Yc�AG*

loss��?ԫ�#       ��-	��˧Yc�AH*

loss,� ?t2�3       ��-	�M̧Yc�AI*

loss�h
?F�e       ��-	r�̧Yc�AJ*

loss [?�o�       ��-	�tͧYc�AK*

loss�'?���       ��-	�	ΧYc�AL*

loss�D?�#\�       ��-	�ΧYc�AM*

loss�H?V���       ��-	2UϧYc�AN*

loss�}?�O��       ��-	'�ϧYc�AO*

loss�Y�>t��
       ��-	!�ЧYc�AP*

lossF�I?�a�       ��-	-&ѧYc�AQ*

loss�ԅ?�/��       ��-	��ѧYc�AR*

loss�=?���       ��-	W`ҧYc�AS*

loss���>�c5`       ��-	�ӧYc�AT*

loss�:?S�v�       ��-	��ӧYc�AU*

loss�e�>q�(       ��-	�OԧYc�AV*

loss<��>��͊       ��-	��ԧYc�AW*

loss�?!_C{       ��-	&�էYc�AX*

loss�t"?��5�       ��-	�'֧Yc�AY*

loss>�>��6       ��-	�֧Yc�AZ*

loss�.?�aJ       ��-	�gקYc�A[*

loss�?��4N       ��-	mاYc�A\*

loss'?�iD�       ��-	��اYc�A]*

loss��
?yf       ��-	C9٧Yc�A^*

losss�.?LE|�       ��-	g�٧Yc�A_*

loss1!�>+�       ��-	�rڧYc�A`*

lossl�? !?�       ��-	MۧYc�Aa*

loss��A?�f1�       ��-	�ۧYc�Ab*

loss�$4?V�w1       ��-	[?ܧYc�Ac*

loss�s!?�F�       ��-	^�ܧYc�Ad*

lossL�
?i��;       ��-	�yݧYc�Ae*

loss���>|��       ��-	�ާYc�Af*

loss3V!?�Y޼       ��-	��ާYc�Ag*

loss�d�>�=8       ��-	�JߧYc�Ah*

loss�u�>έT       ��-	��ߧYc�Ai*

loss8;�>?{       ��-	k�Yc�Aj*

lossw��>td
k       ��-	7��Yc�Ak*

loss��>�:�!       ��-	�?�Yc�Al*

loss�,;?ɏZt       ��-	,��Yc�Am*

loss��)?Qaa�       ��-	��Yc�An*

lossJd?��        ��-	��Yc�Ao*

loss�E$?�RV�       ��-	���Yc�Ap*

loss:��>F+�b       ��-	�z�Yc�Aq*

loss�m�>B��       ��-	8�Yc�Ar*

loss��>SSP�       ��-	Υ�Yc�As*

loss�	�>zd�       ��-	A�Yc�At*

lossc��>�I�       ��-	A��Yc�Au*

loss]y?lެc       ��-	�k�Yc�Av*

lossH	?nf��       ��-	�Yc�Aw*

lossq[? 1h�       ��-	ݙ�Yc�Ax*

loss�?�7�       ��-	31�Yc�Ay*

loss�|�>B�       ��-	���Yc�Az*

loss���>0q4�       ��-	�Y�Yc�A{*

loss_�?��:       ��-	k��Yc�A|*

loss�8�>!�6       ��-	+���Yc�A}*

loss9�>L�<�       ��-	O�Yc�A~*

lossXE1?;�       ��-	���Yc�A*

lossZ��>�[ǧ       �	w��Yc�A�*

loss���>�.��       �	��Yc�A�*

loss�o�>&       �	���Yc�A�*

lossa	�>�UC       �	̴�Yc�A�*

lossD&�>��	�       �	��Yc�A�*

lossq�o>Ǩ�       �	�"�Yc�A�*

loss��>�wk�       �	^��Yc�A�*

lossJ�>H�f       �	�`��Yc�A�*

loss4��>��^O       �	'���Yc�A�*

loss���>D�       �	c���Yc�A�*

loss�ֵ>Ќ�j       �	����Yc�A�*

loss#�>	�;       �	P���Yc�A�*

loss�f�>H��G       �	wM��Yc�A�*

loss�[B>��!>       �	����Yc�A�*

loss:?�>F       �	�"��Yc�A�*

loss4��>�֒       �	a���Yc�A�*

loss)��>=�"�       �	\t��Yc�A�*

loss���>+F�~       �	���Yc�A�*

loss���>�:       �	y���Yc�A�*

loss<^ ?ˑ��       �	�O��Yc�A�*

loss&�A>��N       �	����Yc�A�*

loss�o>*�A�       �	q� �Yc�A�*

loss���>��{       �	d!�Yc�A�*

loss��>��:�       �	|��Yc�A�*

loss���>�)@�       �	H�Yc�A�*

loss��?A�@?       �	���Yc�A�*

lossn��>zSIW       �	�i�Yc�A�*

loss�˖>ݞ#�       �	\�Yc�A�*

loss��o>�ͬj       �	<��Yc�A�*

losss$�>֮P�       �	�C�Yc�A�*

loss8��>�_.       �	���Yc�A�*

loss)�)?׈�o       �	͔�Yc�A�*

loss��>7��'       �	^/�Yc�A�*

loss��?t�       �	��Yc�A�*

lossŮ�>F�-�       �	q	�Yc�A�*

losst�>���       �	$
�Yc�A�*

loss9w>� �C       �	��
�Yc�A�*

loss�+�>���"       �	ǃ�Yc�A�*

lossf|�>p?^f       �	��Yc�A�*

lossݸ�>�       �	/��Yc�A�*

lossc�>��[�       �	\W�Yc�A�*

loss��x>fፑ       �	���Yc�A�*

loss8,>�6       �	�Yc�A�*

loss,Q�>��       �	���Yc�A�*

loss&@m>����       �	�=�Yc�A�*

loss�>��'=       �	��Yc�A�*

loss���>�r}�       �	���Yc�A�*

loss�^a>���       �	!�Yc�A�*

losss��>��       �	���Yc�A�*

loss.5?q���       �	��Yc�A�*

loss�>��}�       �	|�Yc�A�*

loss�p�>��       �	K:�Yc�A�*

lossȮ>ǿ�       �	��Yc�A�*

loss[R>���K       �	��Yc�A�*

loss���>�c�$       �	 %�Yc�A�*

loss� �>��8�       �	R��Yc�A�*

loss�L�>��<       �	LP�Yc�A�*

loss��V>���       �	���Yc�A�*

loss=C�>��Ǐ       �	��Yc�A�*

loss��>b��       �	��Yc�A�*

loss��>0C       �	 ��Yc�A�*

loss���>��t       �	y]�Yc�A�*

lossĦ�>/0��       �	�!�Yc�A�*

loss�.�>�k�       �	���Yc�A�*

loss��t>�+$       �	t�Yc�A�*

loss�̓>rS7       �	�Yc�A�*

loss���>��       �	'��Yc�A�*

loss�X>�3c#       �	�6 �Yc�A�*

loss\��>�I��       �	�� �Yc�A�*

loss@ź>={�       �	�_!�Yc�A�*

loss���>|�'P       �	.="�Yc�A�*

loss"?9��       �	��"�Yc�A�*

lossu�>)ӏ       �	V}#�Yc�A�*

lossg�>U]�       �	�%$�Yc�A�*

loss���>��       �	'�$�Yc�A�*

loss�}�>�`�W       �	GT%�Yc�A�*

loss�q|>J�[�       �	��%�Yc�A�*

loss�R�>�:�       �	��&�Yc�A�*

loss�HZ>A�3       �	�B'�Yc�A�*

loss��>I
�L       �	��'�Yc�A�*

loss���>��s�       �	)y(�Yc�A�*

loss��>�eD�       �	�)�Yc�A�*

loss�o�>Qd�       �	u*�Yc�A�*

lossM�Q>���       �	�+�Yc�A�*

loss7�>"c ?       �	y�+�Yc�A�*

loss�\>b�Ro       �	��,�Yc�A�*

loss���>6s�x       �	�Z-�Yc�A�*

loss��>n�~_       �	�-�Yc�A�*

loss}��>���       �	ۣ.�Yc�A�*

loss�H�>S��       �	�M/�Yc�A�*

loss���>�;��       �	��/�Yc�A�*

lossWK?�h5       �	)�0�Yc�A�*

loss���>�5�       �	�.1�Yc�A�*

loss1J>��ݤ       �	K�1�Yc�A�*

loss�
}>��8       �	�p2�Yc�A�*

loss���>�);Y       �	s3�Yc�A�*

lossD|�><Wξ       �	��3�Yc�A�*

loss?>���       �	h�4�Yc�A�*

loss=>oC�       �	�I5�Yc�A�*

lossz�B>�Fq       �	<�5�Yc�A�*

loss�N�>(�,
       �	�6�Yc�A�*

loss��>i�%       �	�47�Yc�A�*

lossj��>0���       �	��7�Yc�A�*

loss��Y>���       �	�8�Yc�A�*

loss��>���T       �	�C9�Yc�A�*

loss�d�>���a       �	��9�Yc�A�*

lossZQ>ab�       �	t�:�Yc�A�*

loss��>�h;       �	�A;�Yc�A�*

loss.�s>�?�       �	��;�Yc�A�*

loss���>�ѩM       �	ޑ<�Yc�A�*

loss�,�>�,�D       �	�==�Yc�A�*

loss�W�>�#6�       �	��=�Yc�A�*

loss}]3>?�       �	�t>�Yc�A�*

loss�>�̺�       �	�?�Yc�A�*

loss㒢>�b��       �	�?�Yc�A�*

loss2��>(\w       �	PV@�Yc�A�*

loss�>����       �	w�@�Yc�A�*

loss1}�>/��7       �	�A�Yc�A�*

loss��>v�N}       �	� B�Yc�A�*

loss��6>�^x�       �	�B�Yc�A�*

loss�A>�3_�       �	emC�Yc�A�*

loss�M�>�Kq�       �	qD�Yc�A�*

loss.x�>�t��       �	;�D�Yc�A�*

loss���>���       �	�gE�Yc�A�*

loss��>^>@�       �	�F�Yc�A�*

lossL�>&U:       �	�F�Yc�A�*

lossj`;>��:       �	�>G�Yc�A�*

loss#=�>8�-(       �	��G�Yc�A�*

loss��_>��       �	�oH�Yc�A�*

loss�ѥ>̂�       �	RI�Yc�A�*

lossO�>�~2       �	��I�Yc�A�*

lossv�V>%�׳       �	VIJ�Yc�A�*

lossW��>���j       �	��J�Yc�A�*

loss`?�>+��       �	�K�Yc�A�*

loss�c>�R��       �	��L�Yc�A�*

lossM<?�
-4       �	�4M�Yc�A�*

lossᝧ>ry_n       �	�N�Yc�A�*

loss`^ ?jze       �	\�N�Yc�A�*

lossX�>��t�       �	�KO�Yc�A�*

loss��>\��       �	��O�Yc�A�*

loss���>~�u�       �	�P�Yc�A�*

lossu�>劬?       �	�$Q�Yc�A�*

loss��z>��-�       �	�Q�Yc�A�*

loss8�>���       �	?VR�Yc�A�*

lossѣ�>����       �	t�R�Yc�A�*

loss|�k>�B��       �	#�S�Yc�A�*

losszߨ>���/       �	vT�Yc�A�*

lossc%�>�c��       �	��T�Yc�A�*

loss��>��       �	�HU�Yc�A�*

loss��V>{J'       �	��U�Yc�A�*

loss2>7��       �	�{V�Yc�A�*

loss*�>�q��       �	�W�Yc�A�*

loss�aH>Ʌ�o       �	G�W�Yc�A�*

loss�>I�I       �	�NX�Yc�A�*

loss�+L>�ZU       �	��X�Yc�A�*

loss t>@C�K       �	BxY�Yc�A�*

loss:S�>�J��       �	�Z�Yc�A�*

loss���>\9Ao       �	P�Z�Yc�A�*

loss�܉>���       �	�<[�Yc�A�*

loss��P>�}-       �	��[�Yc�A�*

loss�v�=H;�5       �	�v\�Yc�A�*

loss�q�>��ok       �	>%]�Yc�A�*

loss8�#>0�#       �	,�]�Yc�A�*

loss\��=��&+       �	>]_�Yc�A�*

loss���>��%�       �	R�_�Yc�A�*

loss�>�ܴ,       �	H�`�Yc�A�*

loss��> .(       �	!a�Yc�A�*

losss1;>��Vo       �	�a�Yc�A�*

loss@B�>��b�       �	Jbb�Yc�A�*

loss̼�>qlΥ       �	Gc�Yc�A�*

loss�u�>IO       �		�c�Yc�A�*

loss<�>�|GX       �	�Cd�Yc�A�*

loss�>�gf#       �	��d�Yc�A�*

lossif>9Ŭ�       �	Jye�Yc�A�*

loss��o>9;"�       �	�f�Yc�A�*

loss�Fu>�O�       �	ȴf�Yc�A�*

lossv�W>�o5m       �	LTg�Yc�A�*

loss��	>�H�{       �	��g�Yc�A�*

loss��>#xz       �	C�h�Yc�A�*

loss��>�)`M       �	2!i�Yc�A�*

loss#��>2�{       �	��i�Yc�A�*

loss?6>�z[�       �	�nj�Yc�A�*

loss^Z>�g��       �	}	k�Yc�A�*

loss�>$P       �	+�k�Yc�A�*

loss���>"���       �	�;l�Yc�A�*

lossq�`>A{��       �	��l�Yc�A�*

loss�/>�~w       �	��m�Yc�A�*

lossğ>��7�       �	4,n�Yc�A�*

loss�Fo>�\20       �	��n�Yc�A�*

lossX�>�05       �	�bo�Yc�A�*

loss-rV= �       �	��o�Yc�A�*

loss2��><ד�       �	��p�Yc�A�*

loss��>ѹv       �	�:q�Yc�A�*

loss�+>��w       �	��q�Yc�A�*

loss�:�>�w�       �	zr�Yc�A�*

loss��>�4�j       �	�s�Yc�A�*

lossu��>�3"�       �	��s�Yc�A�*

loss��?lL�M       �	�Xt�Yc�A�*

loss��?��β       �	@�t�Yc�A�*

loss[��>��e       �	%�u�Yc�A�*

lossD�I>��v"       �	{-v�Yc�A�*

lossR?Z>Ћ�o       �	?�v�Yc�A�*

loss��>���       �	O\w�Yc�A�*

loss���>	q^       �	Rx�Yc�A�*

loss�G>Z��       �	"�x�Yc�A�*

loss���=�>]�       �	Ay�Yc�A�*

loss�؈>-ක       �	��y�Yc�A�*

loss���>�W       �	�}z�Yc�A�*

loss���=�a(-       �	�{�Yc�A�*

loss���>3{C       �	 �{�Yc�A�*

loss��>��|8       �	�Q|�Yc�A�*

loss�t>P��       �	��|�Yc�A�*

loss'�=�&�v       �	��}�Yc�A�*

loss>e�>~�J       �	�~�Yc�A�*

lossF��=��       �	!�~�Yc�A�*

loss�S>�`�7       �	�K�Yc�A�*

lossςR>�ۑW       �	���Yc�A�*

loss.�?=*ϩ       �	���Yc�A�*

loss�A�>���       �	P��Yc�A�*

loss�B>/I��       �	g���Yc�A�*

loss���>�H9�       �	vR��Yc�A�*

loss��+>��w/       �	��Yc�A�*

loss��>��"o       �		���Yc�A�*

lossE�f>s�       �	���Yc�A�*

lossa�->�GC�       �	t���Yc�A�*

loss+�>Y�l�       �	���Yc�A�*

loss�X>B��       �	���Yc�A�*

loss�p>S-��       �	�;��Yc�A�*

loss�u+>&>�       �	pЇ�Yc�A�*

lossE�v>'Q��       �	�i��Yc�A�*

lossvi>��~       �	���Yc�A�*

loss:=g>�m<I       �	����Yc�A�*

loss�w>��_	       �	 *��Yc�A�*

loss��:>XS�D       �	�Ǌ�Yc�A�*

loss�Á>6c��       �	���Yc�A�*

lossV>�       �	�N��Yc�A�*

loss���>4�@       �	G挨Yc�A�*

loss�L�>[\I       �	%y��Yc�A�*

lossoX>��D]       �	���Yc�A�*

loss6pF>�e1       �	帎�Yc�A�*

lossL!>E�#       �	�c��Yc�A�*

loss��_>3��M       �	���Yc�A�*

loss�J/>���       �	���Yc�A�*

loss�$�>}sSq       �	RE��Yc�A�*

lossm��>�:�       �	�ݑ�Yc�A�*

lossa'�>(PW       �	x��Yc�A�*

loss Eb>%�       �	��Yc�A�*

loss��>f�;+       �	����Yc�A�*

losst{�>@�       �	�?��Yc�A�*

loss}�j>�~��       �	|є�Yc�A�*

loss?[�>ݠ@2       �	�`��Yc�A�*

loss��s>�5j@       �	����Yc�A�*

loss�T�>��'c       �	����Yc�A�*

loss�UD>��T]       �	T��Yc�A�*

loss���>��I�       �	J���Yc�A�*

lossPu>��       �	YL��Yc�A�*

losss�s>g4�5       �	���Yc�A�*

loss@>�=���       �	����Yc�A�*

loss�>ɘ�       �	����Yc�A�*

loss�� >M5��       �	���Yc�A�*

loss��_>�1uY       �	����Yc�A�*

loss�.>�1$       �	�V��Yc�A�*

loss�{^>,�>B       �	�Yc�A�*

loss6t`>⾹	       �	[���Yc�A�*

lossִg>�R��       �	:램Yc�A�*

lossh>j�A"       �	�$��Yc�A�*

loss?Z{>5��H       �	+���Yc�A�*

lossm�G>-o�~       �	Z���Yc�A�*

loss�њ>��I       �	�Ң�Yc�A�*

lossZw> !y�       �	�s��Yc�A�*

lossٍ>�?       �	rP��Yc�A�*

loss_�>>#ߙ       �	�a��Yc�A�*

loss�8�>&��v       �	,,��Yc�A�*

loss�Y>I        �	Tp��Yc�A�*

loss EJ>���_       �	Mj��Yc�A�*

loss8�g>>�ڧ       �	����Yc�A�*

loss?�=w��       �	p��Yc�A�*

loss���=�l\�       �	a8��Yc�A�*

lossF�>��J       �	�K��Yc�A�*

loss��l>���       �	��Yc�A�*

loss�3�>�J��       �	���Yc�A�*

loss��C>E�A�       �	���Yc�A�*

lossg��>�$�       �	�I��Yc�A�*

loss��y>K�       �	f/��Yc�A�*

loss�=W>��%w       �	rk��Yc�A�*

loss7#S>L��       �	���Yc�A�*

loss�º>S��       �	޲�Yc�A�*

loss�xn>;X�B       �	j���Yc�A�*

loss`?�>�z       �	K��Yc�A�*

lossf�u=��_�       �	�Ҵ�Yc�A�*

loss��2>����       �	�j��Yc�A�*

lossN<�>�o       �	E��Yc�A�*

lossׇ6>����       �	����Yc�A�*

loss�$�=�JN3       �	�B��Yc�A�*

lossX�>Gpk       �	�۷�Yc�A�*

loss�R\>��;�       �	�q��Yc�A�*

loss:�>@a�       �	N��Yc�A�*

loss���>���       �	Ƨ��Yc�A�*

loss\%�>�z       �	>��Yc�A�*

loss��@>�?�       �	RԺ�Yc�A�*

loss<ơ>CO�Q       �	m���Yc�A�*

loss�x">��       �	V(��Yc�A�*

loss�4 >��$�       �		���Yc�A�*

loss��>�"�e       �	uW��Yc�A�*

loss��2>`�O�       �	-뽨Yc�A�*

loss{�V>��ر       �	�~��Yc�A�*

loss�C$>�6��       �	���Yc�A�*

lossN>iS3       �	����Yc�A�*

loss��=����       �	�Q��Yc�A�*

loss��/>a��       �	����Yc�A�*

loss�� >�78�       �	><¨Yc�A�*

loss-�=6���       �	��¨Yc�A�*

lossc%�=�#�p       �	ièYc�A�*

lossa��>�g�v       �	�ĨYc�A�*

loss�e�>8�7�       �	ɯĨYc�A�*

loss%;>�?�a       �	�hŨYc�A�*

loss�z>#�&       �	�
ƨYc�A�*

loss���=K��       �	^�ƨYc�A�*

loss<�>Q�       �	HSǨYc�A�*

lossx��>~y*Q       �	L7ȨYc�A�*

lossXh>�b"�       �	W�ȨYc�A�*

loss�A�=I;       �	�zɨYc�A�*

loss�ӑ>X���       �	KʨYc�A�*

loss��>���C       �	��ʨYc�A�*

loss�O�=�̦       �	�s˨Yc�A�*

loss.>U
}       �	*�̨Yc�A�*

lossgv>�7��       �	��ͨYc�A�*

lossl��>q��       �	�ΨYc�A�*

loss�ѱ>\A�%       �	�*ϨYc�A�*

lossh��>�!{       �	��ϨYc�A�*

loss��[>�ȋ�       �	�ѨYc�A�*

loss���>�H�n       �	�ѨYc�A�*

loss�=U>h.�K       �	�bҨYc�A�*

loss�l	>�Yw�       �	Y�ҨYc�A�*

lossې�=F�       �	��ӨYc�A�*

loss8%>F��|       �	81ԨYc�A�*

loss�"o>m��       �	��ԨYc�A�*

loss�=Rջ�       �	��ըYc�A�*

loss�p>E0>�       �	�:֨Yc�A�*

loss	,>�g�       �	��֨Yc�A�*

loss
S?>$�@X       �	WzרYc�A�*

loss��`>���       �	�بYc�A�*

loss1�=�S       �	��بYc�A�*

lossnY�>�r       �	�a٨Yc�A�*

loss��:>���#       �	7�٨Yc�A�*

lossG�>L�       �	ǝڨYc�A�*

lossLm>�U�;       �	�`ۨYc�A�*

loss�X�=%�<,       �	��ۨYc�A�*

loss�eq=����       �	��ܨYc�A�*

loss��=D�       �	�ݨYc�A�*

loss��>��)�       �	�MިYc�A�*

loss��>�?cT       �	7�ިYc�A�*

loss�5�>y ��       �	��ߨYc�A�*

loss$��>�q�f       �	τ�Yc�A�*

loss)2�>q�(�       �	��Yc�A�*

lossrt&>�[��       �	 ��Yc�A�*

loss�[>�\��       �	9��Yc�A�*

loss�5
>��C       �	�r�Yc�A�*

lossJ5">���       �	��Yc�A�*

loss��J>DH/       �	���Yc�A�*

loss�҇>�]��       �	�t�Yc�A�*

lossL�F>fk�X       �	V�Yc�A�*

loss���>_�       �	���Yc�A�*

loss�>{���       �	�{�Yc�A�*

loss��>-(h       �	�H�Yc�A�*

loss��>�%�       �	���Yc�A�*

lossd��=��       �	��Yc�A�*

lossd>Et�(       �	Z)�Yc�A�*

loss]Z>!��u       �	i��Yc�A�*

loss���>�09       �	�_�Yc�A�*

loss�9>E�k       �	� �Yc�A�*

loss�>�#��       �	���Yc�A�*

loss�-n>z�[1       �	IG��Yc�A�*

loss��>�T%       �	U���Yc�A�*

loss�>��[K       �	*��Yc�A�*

lossq��=+#�       �	�i�Yc�A�*

loss)(G>�̡m       �	�	�Yc�A�*

loss��J>J���       �	���Yc�A�*

loss��/>Y�<        �	�P�Yc�A�*

losst�\>����       �	��Yc�A�*

lossԏ>&+�6       �	ޏ�Yc�A�*

loss��Y>�1L�       �	D5�Yc�A�*

loss��;>�_��       �	S"��Yc�A�*

loss��=q9g       �	ʋ��Yc�A�*

loss�N�=V �       �	�B��Yc�A�*

loss�R�=��WI       �	"��Yc�A�*

loss
�F>o��       �	u���Yc�A�*

loss�0>Ug��       �	�n��Yc�A�*

loss��m>���       �	��Yc�A�*

lossH�>N��       �	Ĳ��Yc�A�*

loss��>�y'       �	�O��Yc�A�*

loss��=Ӊk�       �	���Yc�A�*

lossR�>� ,�       �	���Yc�A�*

loss���=S�       �	^��Yc�A�*

loss�>���_       �	����Yc�A�*

loss��=�% �       �	����Yc�A�*

loss
�=E�       �	�H��Yc�A�*

loss��'>Tr�       �	!���Yc�A�*

loss���=��{�       �	� �Yc�A�*

loss/�9>�w��       �	�?�Yc�A�*

loss�7->��4�       �	��Yc�A�*

loss�A�>Wd��       �	���Yc�A�*

loss�Ť=k��M       �	�B�Yc�A�*

lossaA=[��       �	��Yc�A�*

loss�i>\Op�       �	��Yc�A�*

loss���=�V�       �	)=�Yc�A�*

lossq��<����       �	>��Yc�A�*

loss���=���       �	��Yc�A�*

loss�<�=���T       �	A�Yc�A�*

loss���=��^�       �	���Yc�A�*

loss��8=�\A�       �	��Yc�A�*

loss��T=a||�       �	y=	�Yc�A�*

loss�M�>��[�       �	��	�Yc�A�*

loss�ͭ<^[A       �	ep
�Yc�A�*

loss��<�FN5       �	<�Yc�A�*

loss��/=Q��       �	g��Yc�A�*

loss�->y�*       �	�c�Yc�A�*

lossi��>=Gg       �	-
�Yc�A�*

lossZ�!>�3͍       �	��Yc�A�*

loss0�<-߇�       �	�N�Yc�A�*

loss+�=	r�       �	��Yc�A�*

lossVY?��a}       �	d��Yc�A�*

losstx=_�}       �	�*�Yc�A�*

lossv��>s�|�       �	{��Yc�A�*

loss��>�c5       �	�`�Yc�A�*

loss|B�>�Q�       �	���Yc�A�*

loss�$>ijv       �	���Yc�A�*

loss}��=�%e�       �	�+�Yc�A�*

loss�q->uP�M       �	���Yc�A�*

loss�Bi>R��j       �	LU�Yc�A�*

loss��%>D|�*       �	��Yc�A�*

loss��8>j뽢       �	j��Yc�A�*

loss��>�{�.       �	|a�Yc�A�*

loss�mk>�w�{       �	e��Yc�A�*

loss�;�>�55       �	���Yc�A�*

loss��N>�ҷ�       �	�L�Yc�A�*

loss���>�%/�       �	2��Yc�A�*

loss�S>��h       �	��Yc�A�*

loss�>�V-�       �	J)�Yc�A�*

loss�K>ꢜ�       �	��Yc�A�*

loss��>��@       �	�v�Yc�A�*

lossF��=nm0       �	�r�Yc�A�*

loss���='A=       �	Y�Yc�A�*

loss��>��44       �	ܺ�Yc�A�*

lossE�g>�d>l       �	6\�Yc�A�*

loss���=�;�f       �	� �Yc�A�*

lossO4�=�M�R       �	� �Yc�A�*

loss�lq=�R�       �	l!�Yc�A�*

loss�g�>�\�       �	�"�Yc�A�*

loss��=b�~       �	�"�Yc�A�*

loss%O9>CV��       �	�P#�Yc�A�*

loss�i>)�>�       �	��#�Yc�A�*

lossq��=��Ŕ       �	��$�Yc�A�*

lossj�>fCC�       �	(�%�Yc�A�*

lossU��>e�m�       �	�c&�Yc�A�*

loss���=3��F       �	'�Yc�A�*

lossc� >���       �	؟'�Yc�A�*

loss3� >��       �	�:(�Yc�A�*

loss���=���       �	�(�Yc�A�*

loss�Cl>���j       �	�w)�Yc�A�*

loss�>��k       �	�*�Yc�A�*

loss��C>�U~|       �	��*�Yc�A�*

loss$u�=[eC       �	i+�Yc�A�*

loss���=G_f       �	E,�Yc�A�*

loss�:S>1���       �	�,�Yc�A�*

loss�-R>
���       �	�]-�Yc�A�*

lossM�
>�_,       �	�.�Yc�A�*

loss�r;>1��T       �	�.�Yc�A�*

lossfv>4�       �	�b/�Yc�A�*

loss:uE>��iH       �	:<0�Yc�A�*

loss��b>�q}�       �	��0�Yc�A�*

loss��>FTs       �	Ԃ1�Yc�A�*

loss�W�=ε(�       �	�/2�Yc�A�*

loss=�b>$��W       �	ıO�Yc�A�*

loss=�E>�%a       �	�oP�Yc�A�*

loss��><�p       �	�Q�Yc�A�*

loss�*G>���J       �	��Q�Yc�A�*

lossF�'>�Z�'       �	�RR�Yc�A�*

loss��=z�e�       �	>�R�Yc�A�*

loss��P>g(7       �	��S�Yc�A�*

loss��0>2�	       �	iT�Yc�A�*

loss">"�!+       �	��T�Yc�A�*

loss��>5I�<       �	�LU�Yc�A�*

lossH��=F��       �	*�U�Yc�A�*

loss4�=��       �	�vV�Yc�A�*

loss�>�/i�       �	�W�Yc�A�*

loss ;W>G�[       �	t�W�Yc�A�*

losssJ>����       �	SX�Yc�A�*

lossqS6>��	C       �	_�X�Yc�A�*

loss��
>e�L       �	��Y�Yc�A�*

lossJ�=�Y��       �	c*Z�Yc�A�*

loss��>58&       �	��Z�Yc�A�*

loss���>X��       �	c[�Yc�A�*

loss��5>`���       �	M\�Yc�A�*

loss�'_>T�)v       �	a�\�Yc�A�*

loss��?>�U       �	�A]�Yc�A�*

loss�ӂ>�)�       �	�]�Yc�A�*

loss�Tk>Q�S�       �	1�^�Yc�A�*

lossV��=��L9       �	�M_�Yc�A�*

loss� >9)       �	�Z`�Yc�A�*

losst�=�T�       �	a�Yc�A�*

loss�>+��       �	Ҩa�Yc�A�*

loss�S>o��       �	8�c�Yc�A�*

lossH�&>��~0       �	lyd�Yc�A�*

loss�2�=�i`�       �	�e�Yc�A�*

loss͉E>�"�       �	Cf�Yc�A�*

loss��>S�       �	,�f�Yc�A�*

loss��=����       �	�og�Yc�A�*

loss�3�>'x       �	h�Yc�A�*

loss\��=��C�       �	�Ki�Yc�A�*

loss�h>e��"       �	��i�Yc�A�*

loss���>�x�       �	��j�Yc�A�*

lossL��>"2�5       �	�Ek�Yc�A�*

lossj��>��       �	�#m�Yc�A�*

loss���=��/       �	��m�Yc�A�*

loss��=h�	�       �	%tn�Yc�A�*

loss)�Q>���       �	eo�Yc�A�*

lossW(>����       �	�o�Yc�A�*

loss��^>�H�e       �	�`p�Yc�A�*

loss2�y>jN       �	>q�Yc�A�*

loss;>mMЯ       �	ߦq�Yc�A�*

loss�� >e�T       �	�Lr�Yc�A�*

loss��l=q�?S       �	l�r�Yc�A�*

loss���=7��@       �	�s�Yc�A�*

loss�w>/ݪ	       �	B!t�Yc�A�*

loss3�>#nت       �	��t�Yc�A�*

loss��H?a�<�       �	Qu�Yc�A�*

loss�3>���       �	��u�Yc�A�*

lossR^g=s̋       �	�}v�Yc�A�*

loss�,=_쨭       �	�w�Yc�A�*

lossO�>����       �	��w�Yc�A�*

loss��5>��       �	؀x�Yc�A�*

loss��>1*�d       �	5%y�Yc�A�*

loss��$>f�I�       �	��y�Yc�A�*

loss&>�
R#       �	hZz�Yc�A�*

loss1��=���       �	��z�Yc�A�*

loss4�+>>Vp�       �	��{�Yc�A�*

loss�ǳ=f�%       �	�|�Yc�A�*

lossQP>��=       �	S�|�Yc�A�*

loss�L>Uw*�       �	�d}�Yc�A�*

loss!�	>Y�P       �	�}�Yc�A�*

lossD��>���       �	l�~�Yc�A�*

loss�@>��ڒ       �	�6�Yc�A�*

loss���=<��       �	���Yc�A�*

losszW�=�I:       �	�p��Yc�A�*

loss�# >�<my       �	���Yc�A�*

loss��>�w�       �	���Yc�A�*

loss�Z>'��       �	XS��Yc�A�*

loss>���       �	���Yc�A�*

loss�ӣ>|�T       �	󍃩Yc�A�*

loss͒�>/���       �	@4��Yc�A�*

loss厡=JP�       �	gք�Yc�A�*

loss�v�=U�       �	���Yc�A�*

loss
.>��X       �	���Yc�A�*

loss���>�װ       �	հ��Yc�A�*

loss���=��_�       �	�I��Yc�A�*

loss�f�=OO�{       �	�Yc�A�*

lossd��=f�a       �	m���Yc�A�*

lossM�>7�q�       �	F#��Yc�A�*

loss)�%>&u@m       �	w���Yc�A�*

loss3��=y*{       �	&Q��Yc�A�*

lossoZ4>:W��       �	芩Yc�A�*

loss�>1>fj       �	����Yc�A�*

loss�r>ᆮG       �	���Yc�A�*

loss�%2>��       �	2��Yc�A�*

loss-W >�؝       �	�Í�Yc�A�*

loss�*>���       �	�o��Yc�A�*

loss�ғ>�L��       �	����Yc�A�*

loss���=��)       �	��Yc�A�*

loss�
>$��       �	���Yc�A�*

loss��@>��-       �	����Yc�A�*

loss���>���9       �	����Yc�A�*

loss,O>�63       �	�C��Yc�A�*

loss_h�=����       �	�㓩Yc�A�*

loss�H�=A���       �	B���Yc�A�*

loss��(>~*�       �	UQ��Yc�A�*

loss4�=��Jv       �	@���Yc�A�*

loss� >�M       �	����Yc�A�*

loss�d&>/�1J       �	�,��Yc�A�*

lossK�=<&[       �	�Yc�A�*

loss�p�=���q       �	�X��Yc�A�*

lossA�>�wē       �	|Yc�A�*

loss�.+>W�	�       �	���Yc�A�*

lossTj2>ߕs�       �	AF��Yc�A�*

loss�^�=��<�       �	|(��Yc�A�*

loss�IJ>\7       �	����Yc�A�*

loss�>]9�       �	�\��Yc�A�*

lossi�=z�V
       �	Z�Yc�A�*

lossRN�=���?       �	덝�Yc�A�*

lossX>>�db       �	J)��Yc�A�*

losst}>ƋO�       �	�Ξ�Yc�A�*

lossb%>o^=G       �	�l��Yc�A�*

losst�=2�l       �	���Yc�A�*

loss"�>�"��       �	)���Yc�A�*

loss�EP>$��       �	}���Yc�A�*

loss��)>b��       �	f0��Yc�A�*

lossE�/>T�F�       �	 墩Yc�A�*

loss��7=�K�@       �	~ã�Yc�A�*

loss8J�=�>X�       �	���Yc�A�*

loss�@C>3�@�       �	�2��Yc�A�*

loss{7�=Oī�       �	2ʥ�Yc�A�*

loss�C>��ߌ       �	ge��Yc�A�*

loss'G>ƪ��       �	� ��Yc�A�*

loss��G>s0�v       �	ܞ��Yc�A�*

loss�>��&       �	�@��Yc�A�*

loss>U�-�       �	�ݨ�Yc�A�*

lossE�7==N��       �	���Yc�A�*

loss}�b>U�H`       �	�*��Yc�A�*

loss��=>	�g\       �	�Ϊ�Yc�A�*

loss]/z>"68L       �	�n��Yc�A�*

loss�=��#�       �	���Yc�A�*

loss�+>&�Ӡ       �	Ҧ��Yc�A�*

lossO��=�qr       �	�o��Yc�A�*

loss�ʩ=b�       �	���Yc�A�*

loss �I>�+&       �	�̮�Yc�A�*

loss��@>�u��       �	Ae��Yc�A�*

loss�N->�9       �	o��Yc�A�*

loss�%t>���_       �	2���Yc�A�*

loss\n�=���       �	K>��Yc�A�*

loss�<(>�{!Z       �	�Ա�Yc�A�*

loss��6=���       �	�s��Yc�A�*

loss[&�=Z���       �	J��Yc�A�*

loss=,>��4       �	Z���Yc�A�*

loss!�	>@��<       �	B��Yc�A�*

loss��>1��       �	�״�Yc�A�*

loss6#>f�^9       �	�l��Yc�A�*

loss
�>�FzQ       �	��Yc�A�*

lossd�1>_8*�       �	���Yc�A�*

lossȗm=t`6       �	����Yc�A�*

loss�=3>b�9�       �	B]��Yc�A�*

loss�pj>��D       �	/���Yc�A�*

loss�{�>0!��       �	���Yc�A�*

loss�>�-�       �	'N��Yc�A�*

loss3�=M`�       �	꺩Yc�A�*

lossƕ>�+�p       �	����Yc�A�*

loss���=��P       �	Y���Yc�A�*

loss
�e=�9�R       �	=E��Yc�A�*

losst�=R��Y       �	�ܽ�Yc�A�*

loss�U�=��>�       �	����Yc�A�*

loss4�>\t�       �	�-��Yc�A�*

lossm�=��v       �	�ο�Yc�A�*

loss� �=���*       �	�i��Yc�A�*

losst�0>�.$'       �	9
��Yc�A�*

loss���=E�A�       �	v���Yc�A�*

loss�!4=n���       �	�;©Yc�A�*

loss��>��        �	#�©Yc�A�*

loss>���       �	�wéYc�A�*

lossq�m=͔�+       �	jĩYc�A�*

loss��>���       �	��ĩYc�A�*

loss*��>k��       �	2WũYc�A�*

loss��+>�R�       �	��ũYc�A�*

loss���=64r�       �	L�ƩYc�A�*

lossS�=��C�       �	:"ǩYc�A�*

loss���=��]*       �	ȩYc�A�*

loss��>#��       �	��ȩYc�A�*

loss�9�=h��       �	�=ɩYc�A�*

loss��\=�wSb       �	z�ɩYc�A�*

loss=J�=Y���       �	RʩYc�A�*

lossL� >���       �	�˩Yc�A�*

loss�3>�<       �	-�˩Yc�A�*

losse�>��Xx       �	@L̩Yc�A�*

lossV�=����       �	��̩Yc�A�*

loss�U5>���       �	��ͩYc�A�*

loss��F>�~	       �	l@ΩYc�A�*

lossf>UX�       �	9�ΩYc�A�*

loss���=Ys�       �	�kϩYc�A�*

loss���=���       �	/ЩYc�A�*

loss-�u=���       �	B�ЩYc�A�*

loss�|�=�-o       �	/mѩYc�A�*

loss��>���       �	`ҩYc�A�*

loss��>�{0       �	E�ҩYc�A�*

loss�Պ>jk�       �	�pөYc�A�*

loss�A�=���       �	�ԩYc�A�*

loss�b>�� �       �	a�ԩYc�A�*

loss�b�=27��       �	
dթYc�A�*

loss϶�=��Ā       �	�֩Yc�A�*

lossW��=h��N       �	�֩Yc�A�*

lossOۨ=����       �	�?שYc�A�*

lossW��=D�ǅ       �	�שYc�A�*

loss�G>:!��       �	��ةYc�A�*

lossݺ�=��       �	�+٩Yc�A�*

loss�ap=Pz�       �	��٩Yc�A�*

lossd�0>E��?       �	%zکYc�A�*

loss�u�=Q�       �	�۩Yc�A�*

loss��>�Ot<       �	��۩Yc�A�*

loss��=z�h       �	C<ܩYc�A�*

loss��b=�VR       �	b�ܩYc�A�*

loss��L>~�a       �	epݩYc�A�*

loss��=�P�       �	-ZީYc�A�*

loss{l>�(:�       �	ZߩYc�A�*

lossV��>Tǐ�       �	v��Yc�A�*

lossni�>�*�9       �	�J�Yc�A�*

loss!�r>`hbB       �	2�Yc�A�*

loss1&=!۠B       �	|'�Yc�A�*

lossã�=�p�       �	H��Yc�A�*

loss�}K>��(�       �	���Yc�A�*

loss��W>�@��       �	U��Yc�A�*

loss#P�<����       �	\�Yc�A�*

lossF�>��>]       �	�>�Yc�A�*

loss)$�=E�H�       �	*��Yc�A�*

lossק�=�JO}       �	���Yc�A�*

loss���=���       �	q��Yc�A�*

loss���=���       �	c*�Yc�A�*

loss�v=�B�       �	��Yc�A�*

loss��K=�SMe       �	��Yc�A�*

lossv�>o7��       �	7��Yc�A�*

loss���=@>�Z       �	D��Yc�A�*

loss� >%��P       �	���Yc�A�*

loss ø=�Ȃ�       �	�l�Yc�A�*

lossM�=4��{       �	��Yc�A�*

loss.f>>�ü       �	��Yc�A�*

loss�o!>�4       �	/��Yc�A�*

loss���=joP�       �	�'�Yc�A�*

lossL�=F^�       �	8��Yc�A�*

loss���=4���       �	,���Yc�A�*

loss!\�=�؎�       �	@M��Yc�A�*

lossȝ�=��Pr       �	���Yc�A�*

loss2�=y�!1       �	j��Yc�A�*

loss��o>ج2       �	II��Yc�A�*

loss6_�=��H�       �	����Yc�A�*

loss4Ԧ=~~��       �	���Yc�A�*

losss�->�i�       �	����Yc�A�*

loss�w�=Tl|r       �	>y��Yc�A�*

loss�U>l �       �	V���Yc�A�*

lossr`D>�Hw        �	�O��Yc�A�*

loss A�=,��       �	'���Yc�A�*

loss��=��       �	����Yc�A�*

loss��r=�,q       �	-Z��Yc�A�*

loss�QE=�B��       �	B �Yc�A�*

lossO(1>)���       �	 � �Yc�A�*

loss�iU>�+7       �	jL�Yc�A�*

loss&ٷ=��j�       �	1��Yc�A�*

lossC��=��(�       �	���Yc�A�*

loss<#>w���       �	T�Yc�A�*

lossC�=�O�       �	.��Yc�A�*

loss畗>��Wb       �	�u�Yc�A�*

loss���=T�       �	5�Yc�A�*

loss�u`>���0       �		��Yc�A�*

loss�e�=�=f       �	�A�Yc�A�*

loss )>�\�%       �	���Yc�A�*

loss_I>���       �	{�Yc�A�*

loss$H�>SdAc       �	��Yc�A�*

loss�T�=���       �	ڨ�Yc�A�*

loss�]>6,       �	i;	�Yc�A�*

lossRH�=��v�       �	�3
�Yc�A�*

lossѐ�=dA&�       �	��
�Yc�A�*

loss�>�+�{       �	j�Yc�A�*

loss�>��VN       �	R�Yc�A�*

lossZe>\s@       �	ߥ�Yc�A�*

lossqU�=����       �	%=�Yc�A�*

loss/b�=W�p�       �	���Yc�A�*

lossh�1>�       �	<i�Yc�A�*

lossz�=&A<�       �	��Yc�A�*

loss��4=���5       �	c��Yc�A�*

loss x�=g^&�       �	�0�Yc�A�*

loss��>�ZY       �	���Yc�A�*

loss���=+'Cq       �	@h�Yc�A�*

loss#�>vc�n       �	��Yc�A�*

loss�L�=7]j       �	���Yc�A�*

loss�@�=�B       �	�o�Yc�A�*

loss��=\�       �	��Yc�A�*

loss<6>"�l`       �	Ϊ�Yc�A�*

loss}cq=FЏ(       �	S>�Yc�A�*

loss@TP=I�a�       �	���Yc�A�*

loss{��=�0��       �	|{�Yc�A�*

lossh#=!��       �	��Yc�A�*

loss��<ޟ(       �	���Yc�A�*

lossϹ{=���$       �	�E�Yc�A�*

loss�l$>>�x       �	��Yc�A�*

lossz�>�^       �	q��Yc�A�*

loss�f><��       �	?W�Yc�A�*

lossC�>�А       �	r��Yc�A�*

lossL$4>���       �	��Yc�A�*

loss �=6"��       �	�p�Yc�A�*

lossX��=0O�s       �	��Yc�A�*

loss�
�=�fm#       �	��Yc�A�*

loss��/>r���       �	�T�Yc�A�*

lossƍ�=Ӵ&       �	~��Yc�A�*

lossS��>�˱�       �	� �Yc�A�*

loss�J=jC��       �	�� �Yc�A�*

loss��#>S��       �	J!�Yc�A�*

loss�L�<��n       �	/2"�Yc�A�*

loss}��=¹u       �	�V#�Yc�A�*

loss�� >@�[       �	�#�Yc�A�*

loss�F)>e��~       �	��$�Yc�A�*

lossl�/>f�K�       �	��%�Yc�A�*

loss���=���       �	4&�Yc�A�*

loss�HM=��7       �	��&�Yc�A�*

loss]�<>Q6F       �	�v'�Yc�A�*

lossC�>C��7       �	(�Yc�A�*

loss^�=�A��       �	ǻ(�Yc�A�*

loss��]>/��       �	�b)�Yc�A�*

lossjC>���       �	�*�Yc�A�*

loss�xj=�؛3       �	��*�Yc�A�*

lossCK7>Ӂ�       �	uX+�Yc�A�*

loss��>�(��       �	��+�Yc�A�*

loss�O�=���       �	ڏ,�Yc�A�*

loss�>�{f?       �	=)-�Yc�A�*

loss�>�c�       �	"�-�Yc�A�*

loss�3>S�#       �	}x.�Yc�A�*

loss�@�=��G�       �	� /�Yc�A�*

loss��
>Ml��       �	j�/�Yc�A�*

loss]V>G���       �	�W0�Yc�A�*

loss� �=}5(m       �	��0�Yc�A�*

loss�z>��Vx       �	�1�Yc�A�*

lossHne=�	l       �	 '2�Yc�A�*

loss�~�=�8       �	d�2�Yc�A�*

loss y<>ͣ       �	N�3�Yc�A�*

loss�� >��r       �	n14�Yc�A�*

losso�>j��       �	d�4�Yc�A�*

loss���=�ӄ       �	
f5�Yc�A�*

loss� �=�D7�       �	6�Yc�A�*

loss�Z;=�9��       �	�.7�Yc�A�*

loss�"�=��H       �	��7�Yc�A�*

lossq�x=mh��       �	nm8�Yc�A�*

loss@u=Q��`       �	�9�Yc�A�*

loss�r$>U��A       �	)�9�Yc�A�*

loss^��>���       �	Pq:�Yc�A�*

loss��_>�Zh       �	;�Yc�A�*

loss��u=X��       �	��;�Yc�A�*

loss��o>/��a       �	�C<�Yc�A�*

losst]m=�|}i       �	��<�Yc�A�*

loss��>Yr(E       �	M�=�Yc�A�*

loss$��=�Jߡ       �	%>�Yc�A�*

loss�1�=�.�u       �	��>�Yc�A�*

lossWH�=t��       �	�Y?�Yc�A�*

loss���=��+       �	A�?�Yc�A�*

lossx0#>�wn       �	��@�Yc�A�*

loss�H=�4��       �	d$A�Yc�A�*

loss��=@Ce>       �	��A�Yc�A�*

lossי�=����       �	>\B�Yc�A�*

loss���=��M�       �	�VC�Yc�A�*

loss�,x=����       �	��C�Yc�A�*

loss�>�:��       �	x�D�Yc�A�*

loss��>e�       �	�nE�Yc�A�*

losstd�=m�[       �	F�Yc�A�*

loss�5U>�!�       �	�F�Yc�A�*

loss AV>3[�7       �	=G�Yc�A�*

loss�Z�=sF�!       �	�G�Yc�A�*

loss/X>�oX�       �	�{H�Yc�A�*

loss��>/ӛ�       �	rI�Yc�A�*

loss�>���       �		�I�Yc�A�*

loss�i=u�	       �	�]J�Yc�A�*

loss��>����       �	K�Yc�A�*

lossO{�=� ��       �	��K�Yc�A�*

loss�ܟ=E��       �	�AL�Yc�A�*

loss@��=��4�       �	��L�Yc�A�*

lossD>	I�       �	7qM�Yc�A�*

loss���>�} �       �	�N�Yc�A�*

loss[1>fx��       �	,�N�Yc�A�*

lossuP><K�       �	XWO�Yc�A�*

loss[�=�Oe+       �	��O�Yc�A�*

loss��>?��       �	��P�Yc�A�*

loss�t>�!�       �	KQ�Yc�A�*

losse',>�_�       �	�cR�Yc�A�*

loss�.>CG�u       �	��R�Yc�A�*

loss���=%��h       �	��S�Yc�A�*

loss��T=]l�       �	�U�Yc�A�*

lossfW�=X��       �	��U�Yc�A�*

lossF�>�+y�       �	�IV�Yc�A�*

lossm��=����       �	�W�Yc�A�*

loss=BO=5>�       �	ߨW�Yc�A�*

lossLP>R�i       �	�MX�Yc�A�*

lossU8>�14�       �	p�X�Yc�A�*

loss��=8/+�       �	ʊY�Yc�A�*

lossa4>�e�Z       �	w�Z�Yc�A�*

loss��">�5HV       �	� [�Yc�A�*

lossV�>����       �	��[�Yc�A�*

loss��{>�sޮ       �	/j\�Yc�A�*

loss��w={.A       �	�]�Yc�A�*

loss���=��K       �	�]�Yc�A�*

loss���=qYn       �	.S^�Yc�A�*

loss��M>6       �	�_�Yc�A�*

lossn�=6�#       �	�`�Yc�A�*

loss{�9>��{F       �	ޫ`�Yc�A�*

loss=Z>�V�j       �	$Ea�Yc�A�*

loss6�u=�G�       �	n�a�Yc�A�*

loss��[=��Xq       �	�xb�Yc�A�*

loss�k>vh\�       �	�c�Yc�A�*

lossUP�>RnÊ       �	>�c�Yc�A�*

loss8�>I<       �	�Pd�Yc�A�*

loss%	>[Sk�       �	�d�Yc�A�*

lossO��=1���       �	4�e�Yc�A�*

loss >�s       �	!f�Yc�A�*

loss�n>�d(Y       �	Ϻf�Yc�A�*

lossS��=_ab�       �	�`g�Yc�A�*

losso��>���       �	ih�Yc�A�*

lossj��>�3�t       �	<�h�Yc�A�*

losszۀ>X]       �	�Fi�Yc�A�*

lossB�<�{}�       �	�i�Yc�A�*

loss��y=5�6�       �	d�j�Yc�A�*

loss��>>>�s       �	�4k�Yc�A�*

loss��=:ˬ�       �	��k�Yc�A�*

loss�(�=L��       �	�l�Yc�A�*

loss�>+9�       �	�(m�Yc�A�*

loss���=�#��       �	��m�Yc�A�*

loss�&Z>O*�       �	�sn�Yc�A�*

lossѧ>��	�       �	�o�Yc�A�*

loss<�/>����       �	ݲo�Yc�A�*

loss B�=�2��       �	8fp�Yc�A�*

loss#�5>�I        �	�q�Yc�A�*

loss��=a�͒       �	\�q�Yc�A�*

lossi�=x��       �	�Pr�Yc�A�*

loss�B)>ǎ�       �	��r�Yc�A�*

loss}^�=e���       �	��s�Yc�A�*

lossv*�=�]u       �	�$t�Yc�A�*

loss@�>�b�       �	��t�Yc�A�*

lossn��=�\�       �	�su�Yc�A�*

loss8�]='C       �	v�Yc�A�*

loss@ot=���       �	W�v�Yc�A�*

loss:�=�S�       �	l�w�Yc�A�*

loss%��<.�M        �	I,x�Yc�A�*

loss{��=FK��       �	z�x�Yc�A�*

loss/j�>��       �	�`y�Yc�A�*

loss��>��#       �	��y�Yc�A�*

lossl޵=8XZ�       �	��z�Yc�A�*

loss�_>EÁi       �	Y�{�Yc�A�*

lossmv�=(��       �	�m|�Yc�A�*

loss1 >d�0�       �	-}�Yc�A�*

loss
l�>�3~]       �	s�}�Yc�A�*

lossi��=
�׷       �	fL~�Yc�A�*

loss��=�9�       �	P�~�Yc�A�*

loss��Q>�Rp       �	��Yc�A�*

loss|z>=�=G       �	�%��Yc�A�*

loss�)@=�,u       �	����Yc�A�*

lossH�A=�F�       �	�S��Yc�A�*

loss.x>򴅪       �	yꁪYc�A�*

loss!�H>g]O�       �	P���Yc�A�*

loss�PG>�`�Y       �	B&��Yc�A�*

loss�o�=���c       �	Cȃ�Yc�A�*

loss��=� ��       �	�g��Yc�A�*

loss��= F�       �	Q��Yc�A�*

loss��H>q�^L       �	L���Yc�A�*

loss�s�=߇��       �	wI��Yc�A�*

loss�>=��i�       �	�t��Yc�A�*

loss�)�= �X�       �	��Yc�A�*

loss���=�D�       �	����Yc�A�*

lossS� =58Z�       �	�U��Yc�A�*

loss���=Zr�j       �	���Yc�A�*

loss=���a       �	���Yc�A�*

loss|:�=EZ#       �	�4��Yc�A�*

loss� {=g��       �	=֋�Yc�A�*

losslFp=���       �	t��Yc�A�*

lossT�>n�       �	S"��Yc�A�*

loss�>z+��       �	!ʍ�Yc�A�*

loss}�=�gy%       �	�m��Yc�A�*

loss��:>���J       �	���Yc�A�*

lossC��=�ը�       �	����Yc�A�*

loss1�U=�^:       �	�b��Yc�A�*

loss��=�b��       �	����Yc�A�*

loss��=X���       �	9���Yc�A�*

loss��>Ѐ�       �	r7��Yc�A�*

loss�M=>���       �	A֒�Yc�A�*

loss��>�9{i       �	�s��Yc�A�*

loss1��=?D�       �	{��Yc�A�*

loss�6�=>���       �	ɭ��Yc�A�*

loss>�>��(\       �	wN��Yc�A�*

loss�ժ=*�[L       �	:镪Yc�A�*

lossׁv=��M       �	����Yc�A�*

lossٮ>�^��       �	X9��Yc�A�*

loss(u>bQhm       �	�՗�Yc�A�*

loss�d�=^�l
       �	�m��Yc�A�*

loss6�S>�.�*       �	X��Yc�A�*

lossr�=��       �	훙�Yc�A�*

loss���=�3	
       �	]1��Yc�A�*

loss���=� ��       �	aƚ�Yc�A�*

lossO�^=Ӽ`       �	]��Yc�A�*

loss�'�=G~�       �	'���Yc�A�*

loss�=5Zt�       �	~���Yc�A�*

loss3%>���`       �	t$��Yc�A�*

loss��=��       �	�Yc�A�*

loss�~=�A��       �	_��Yc�A�*

loss��>_hx       �	Y���Yc�A�*

loss��.>ؘ��       �	�ҟ�Yc�A�*

lossͣI=�`'{       �	dx��Yc�A�*

lossF�>�I\�       �	&��Yc�A�*

loss�г=xo2)       �	@��Yc�A�*

loss��0>+NU�       �	�䢪Yc�A�*

loss��=�.-^       �	����Yc�A�*

loss��=�c�       �	:$��Yc�A�*

loss��Z=��]o       �	T��Yc�A�*

loss�{�=���8       �	}ʥ�Yc�A�*

loss�>�\       �	k���Yc�A�*

loss?�=����       �	Y��Yc�A�*

lossT�>b��       �	���Yc�A�*

loss�7=U'²       �	����Yc�A�*

loss\�;>v��       �	�Y��Yc�A�*

loss��=g9\:       �	ff��Yc�A�*

loss�X>��       �	�
��Yc�A�*

loss�V�=u��E       �	Ძ�Yc�A�*

loss]��=j�e?       �	>]��Yc�A�*

loss2�q=����       �	+���Yc�A�*

loss�>����       �	����Yc�A�*

loss�φ=�f~�       �	EF��Yc�A�*

loss4H�=�6p       �	���Yc�A�*

lossr>�<O�wW       �	囯�Yc�A�*

lossq�=AQ��       �	�7��Yc�A�*

lossf��=��)       �	k԰�Yc�A�*

loss��=-|4�       �	n��Yc�A�*

loss�=� ��       �	���Yc�A�*

loss>�[]c       �	����Yc�A�*

loss�F >���%       �	�<��Yc�A�*

loss\��<D!��       �	�г�Yc�A�*

loss���<��/�       �	h��Yc�A�*

loss�ק=���       �	G��Yc�A�*

lossd.=�Ե       �	����Yc�A�*

loss�<�1C       �	d:��Yc�A�*

loss���<O't       �	�鶪Yc�A�*

lossA��<=�;�       �	����Yc�A�*

loss�+�<���
       �	��Yc�A�*

loss��<7�P       �	ȳ��Yc�A�*

loss���;� i-       �	@M��Yc�A�*

loss��=[E�       �	�鹪Yc�A�*

loss��=�-J�       �	���Yc�A�*

loss���;�bU       �	���Yc�A�*

loss���;����       �	���Yc�A�*

loss�Ny<�{cR       �	MJ��Yc�A�*

lossxN>�]+       �	+߼�Yc�A�*

lossLkO=^/@       �	�u��Yc�A�*

loss���;:�_       �	���Yc�A�*

lossj��=ʔ4�       �	����Yc�A�*

loss;?�DK       �	�I��Yc�A�*

loss�h�<��W       �	�⿪Yc�A�*

lossi��>����       �	w���Yc�A�*

loss,x�==�c       �	�W��Yc�A�	*

loss�@">� .       �	����Yc�A�	*

loss��= ��:       �	Y�ªYc�A�	*

loss��<{�?z       �	t)êYc�A�	*

loss��>$,]       �	��êYc�A�	*

loss���=�c�`       �	�cĪYc�A�	*

loss�+
>�۰�       �	@ŪYc�A�	*

loss��Q>ź       �	<�ŪYc�A�	*

loss�L>N��       �	�tƪYc�A�	*

lossS<>|�NB       �	<ǪYc�A�	*

loss��T>��z�       �	�ǪYc�A�	*

loss?��=�
�P       �	�=ȪYc�A�	*

loss��D>��'v       �	8�ȪYc�A�	*

lossNmN>cU=       �	mqɪYc�A�	*

loss2��=�Ҹ�       �	9	ʪYc�A�	*

losse��=��w"       �	j�ʪYc�A�	*

loss���=��tP       �	v6˪Yc�A�	*

lossI��=G+       �	�˪Yc�A�	*

loss��=�q)�       �	�a̪Yc�A�	*

loss���=�W;       �	��̪Yc�A�	*

loss���=6���       �	��ͪYc�A�	*

loss5.=�;f�       �	l?ΪYc�A�	*

loss�C�=E�N5       �	�ΪYc�A�	*

lossנR=�b�       �	�uϪYc�A�	*

loss:�=[�r-       �	5ЪYc�A�	*

loss�W�=z�=       �	`�ЪYc�A�	*

loss}y>����       �	�JѪYc�A�	*

loss.�a>��i       �	?�ѪYc�A�	*

loss�k�=SbX       �	�{ҪYc�A�	*

lossH��=�)       �	OӪYc�A�	*

loss՚=c�k�       �	w�ӪYc�A�	*

losstWG=��G/       �	�iԪYc�A�	*

loss=]�=��E?       �	��ԪYc�A�	*

loss�"w=j���       �	F�ժYc�A�	*

loss6��<0�r;       �	^+֪Yc�A�	*

loss��=��r       �	��֪Yc�A�	*

loss1/>�=o       �	P�תYc�A�	*

loss8:�=���       �	�#تYc�A�	*

loss8�=Ύ�       �	��تYc�A�	*

loss%I=��z�       �	uX٪Yc�A�	*

loss��_=_�n�       �	��٪Yc�A�	*

loss�y>��I�       �	c�ڪYc�A�	*

loss�ү=VA0o       �	�3۪Yc�A�	*

loss���=�n@�       �	��۪Yc�A�	*

lossI]>9���       �	{�ܪYc�A�	*

loss*�-=�&G6       �	�[ݪYc�A�	*

loss	
�=�5F�       �	g�ݪYc�A�	*

loss���=lS��       �	S�ުYc�A�	*

loss�i=���;       �	o,ߪYc�A�	*

loss���=7Y       �	�p�Yc�A�	*

loss�E>��\       �	8�Yc�A�	*

lossi�>��>�       �	���Yc�A�	*

loss�M�=���       �	R�Yc�A�	*

loss)�=V*       �	��Yc�A�	*

loss	#8=���n       �	X��Yc�A�	*

lossZ$�=�*�<       �	L3�Yc�A�	*

loss��=N���       �	���Yc�A�	*

loss:;>:�gl       �	k�Yc�A�	*

loss���=��?b       �	�	�Yc�A�	*

loss��H=�%�       �	���Yc�A�	*

loss@	S=�yVc       �	jL�Yc�A�	*

lossx;
>��G       �	���Yc�A�	*

loss���=@<�       �	��	�Yc�A�	*

loss_�=�s�       �	�"
�Yc�A�	*

loss�9)>W�h       �	�
�Yc�A�	*

loss1�<� �       �	O[�Yc�A�	*

loss��=_���       �	���Yc�A�	*

loss(7�=8�       �	׾�Yc�A�	*

loss�>o>�-~�       �	*U�Yc�A�	*

loss㳧=�&w�       �	l	�Yc�A�	*

loss��+>X9 1       �	7��Yc�A�	*

loss}��=ߧ��       �	�H�Yc�A�	*

loss��>�g�       �	���Yc�A�	*

loss$��=�D��       �	Ք�Yc�A�	*

loss�`=�%l�       �	R+�Yc�A�	*

lossݩ>�lO�       �	��Yc�A�	*

loss20�=�D�I       �	2Z�Yc�A�	*

loss���=]�]8       �	���Yc�A�	*

lossѥ�=]=��       �	���Yc�A�	*

loss�Z�=����       �	�.�Yc�A�	*

loss���=��       �	���Yc�A�	*

loss�->!痜       �	]��Yc�A�	*

lossm8i>��E�       �	�+�Yc�A�	*

loss�u=�^��       �	���Yc�A�	*

lossFA>��0�       �	�s�Yc�A�	*

loss�J=�       �	�Yc�A�	*

loss���=���       �	���Yc�A�	*

loss/2�>���       �	�[�Yc�A�	*

lossN�H>٢`6       �	&��Yc�A�	*

loss�I�=&��3       �	���Yc�A�	*

loss�#�=M@�       �	C�Yc�A�	*

loss�=��o       �	��Yc�A�	*

loss�b>%�4�       �	�~�Yc�A�	*

loss�@�=�9mn       �	��Yc�A�	*

loss�B>.�&r       �	h��Yc�A�	*

losse��=1��        �	vT�Yc�A�	*

loss8,�=��	F       �	���Yc�A�	*

lossN��<%g��       �	Ȕ�Yc�A�	*

loss=�^\�       �	�Y �Yc�A�	*

loss@��<쏀.       �	�?!�Yc�A�	*

loss=5�=8}��       �	��!�Yc�A�	*

loss�=h8       �	e�"�Yc�A�	*

loss
8�>�]�B       �	�@#�Yc�A�	*

loss��=���       �	U�#�Yc�A�	*

lossv��<:��       �	��$�Yc�A�	*

loss�%�=��{>       �	"%�Yc�A�	*

loss�U=��d�       �	կ%�Yc�A�	*

loss�9�=4�       �	I&�Yc�A�	*

loss���=����       �	��&�Yc�A�	*

loss��E>.�!�       �	<�'�Yc�A�	*

loss�Sa=_��       �	�(�Yc�A�	*

loss��=u_�       �	��(�Yc�A�	*

loss&	>�}��       �	��)�Yc�A�	*

lossA�=�OH�       �	#*�Yc�A�	*

loss�==b�rH       �	[�*�Yc�A�	*

lossH>�_��       �	[+�Yc�A�	*

loss3~�=
�N       �	��+�Yc�A�	*

loss)J>W�:`       �	�,�Yc�A�	*

loss�0>�ik       �	~s-�Yc�A�	*

lossT��=<��       �	�.�Yc�A�	*

loss�,�=W���       �	S�.�Yc�A�	*

loss��=�e}�       �	<L/�Yc�A�	*

lossܤ�=��&d       �	q�/�Yc�A�	*

lossƳ�=2P�X       �	��0�Yc�A�	*

loss�]�=T��2       �	�1�Yc�A�	*

lossg�=��S�       �	��1�Yc�A�	*

loss2�M>�C�N       �	�2�Yc�A�	*

lossvM{=Mg�a       �	�Q3�Yc�A�	*

loss�ʽ=+�y�       �	��3�Yc�A�
*

loss�p>���       �	ı4�Yc�A�
*

loss�w+>b�H       �	D�5�Yc�A�
*

loss�>/Q�!       �	,D6�Yc�A�
*

loss2M�=?�#L       �	G7�Yc�A�
*

loss�݋=XmF�       �	�7�Yc�A�
*

loss��=0��4       �	��8�Yc�A�
*

lossn��=S�J�       �	�K9�Yc�A�
*

loss$�
=�/�       �	��9�Yc�A�
*

loss�t�=��W�       �	Ō:�Yc�A�
*

loss|�u=M��       �	�);�Yc�A�
*

lossE��=�0V�       �	��;�Yc�A�
*

loss��=���F       �	�k<�Yc�A�
*

loss6�P=c���       �	!=�Yc�A�
*

lossT�I>�x/3       �	=�=�Yc�A�
*

loss7�)>�{       �	X8>�Yc�A�
*

losssg�=%��       �	9�>�Yc�A�
*

loss(7�=�J~�       �	�m?�Yc�A�
*

loss���=��       �	�@�Yc�A�
*

loss�z=>���       �	ڮ@�Yc�A�
*

loss�)?>/g��       �	�VA�Yc�A�
*

lossS,�=A'�       �	T6B�Yc�A�
*

loss�us=:`�       �	f�C�Yc�A�
*

loss̶@>S�.�       �	a6D�Yc�A�
*

loss��=¸L�       �	iE�Yc�A�
*

loss �=7���       �	)	F�Yc�A�
*

loss��T>�	       �	ŪF�Yc�A�
*

loss���<����       �	$CG�Yc�A�
*

loss!Q�=��#r       �	f�G�Yc�A�
*

lossr�>N���       �	�pH�Yc�A�
*

loss��=x��       �	I�Yc�A�
*

loss	� >'�
?       �	�I�Yc�A�
*

loss�q�=֦��       �	�^J�Yc�A�
*

loss��=��       �	j�J�Yc�A�
*

loss+6=P���       �	��K�Yc�A�
*

loss�;t=F�r�       �	�(L�Yc�A�
*

lossH_�=��0�       �	��L�Yc�A�
*

loss�ܨ=4uT       �	�M�Yc�A�
*

lossW��=*9T�       �	�N�Yc�A�
*

loss|��=fȭ       �	�N�Yc�A�
*

loss�ޮ=�c�W       �	�qO�Yc�A�
*

loss���=�os       �	�P�Yc�A�
*

losss��=f���       �	��P�Yc�A�
*

loss�\>��       �	�^Q�Yc�A�
*

loss�ȫ=mW��       �	RR�Yc�A�
*

loss��=�/��       �	ƢR�Yc�A�
*

lossX�>q��       �	9CS�Yc�A�
*

lossߕ�=���e       �	�S�Yc�A�
*

lossK;=)��       �	��T�Yc�A�
*

lossR�E>�gP       �	�*U�Yc�A�
*

loss��=Ϳ�       �	G�U�Yc�A�
*

loss���=;�K�       �	'jV�Yc�A�
*

lossр�=f��       �	+W�Yc�A�
*

loss4'�=*�       �	��W�Yc�A�
*

loss�D=3/ϰ       �	u�X�Yc�A�
*

loss;��=�bCs       �	�TY�Yc�A�
*

loss�6�=��z�       �	
�Y�Yc�A�
*

loss%�=ՏH�       �	�Z�Yc�A�
*

lossE��<4�(       �	DO[�Yc�A�
*

loss�
�=*��       �	��[�Yc�A�
*

lossC�X=���       �	Ό\�Yc�A�
*

loss���<ݠ       �	0,]�Yc�A�
*

loss��Q>�_*       �	�]�Yc�A�
*

loss��>|�\       �	ĕ^�Yc�A�
*

loss���=�+�       �	1_�Yc�A�
*

loss�n>��       �	,�_�Yc�A�
*

lossH%G=8*e�       �	�w`�Yc�A�
*

loss���=L?�       �	�a�Yc�A�
*

loss��<���       �	��a�Yc�A�
*

lossh}�<]R/I       �	}Yb�Yc�A�
*

loss%@�=��,s       �	�^c�Yc�A�
*

lossQӃ=A�W       �	p	d�Yc�A�
*

lossJm;>~���       �	Ʀd�Yc�A�
*

loss��>v��       �	Ve�Yc�A�
*

loss��>g�p       �	6f�Yc�A�
*

loss<�>�b�       �	��f�Yc�A�
*

loss�<]W�       �	�<g�Yc�A�
*

loss��=5�3       �	e�g�Yc�A�
*

loss�)>d]�>       �	�h�Yc�A�
*

loss�>��+�       �	�i�Yc�A�
*

loss�K�=X��       �	��i�Yc�A�
*

loss;ǫ=T�       �	�Tj�Yc�A�
*

loss$9�=�3{       �	��j�Yc�A�
*

loss߫�=g�
-       �	T�k�Yc�A�
*

loss!c<���       �	�Ol�Yc�A�
*

lossϠ�=sw{        �	�l�Yc�A�
*

loss22�=���/       �	�m�Yc�A�
*

loss��=6��       �	QLn�Yc�A�
*

loss!�
>�u       �	��n�Yc�A�
*

loss��=����       �	o�o�Yc�A�
*

lossi��<����       �	IGp�Yc�A�
*

loss�%�=;��       �	\�p�Yc�A�
*

loss�H�<3��       �	��q�Yc�A�
*

loss�=�=��V�       �	1r�Yc�A�
*

loss6��=����       �	�r�Yc�A�
*

loss�>�<�yjl       �	�}s�Yc�A�
*

loss���=�A       �	�t�Yc�A�
*

loss\I�>@RR       �	��t�Yc�A�
*

loss�+>�
�       �	ofu�Yc�A�
*

loss�=����       �	v�Yc�A�
*

loss�u=#~�.       �	üv�Yc�A�
*

loss2�~<���R       �	_w�Yc�A�
*

loss�>(*�       �	9x�Yc�A�
*

loss��=*�       �	�x�Yc�A�
*

lossN==^�J       �	^y�Yc�A�
*

loss��b=B��       �	�z�Yc�A�
*

lossQP�=G5`       �	�z�Yc�A�
*

loss��=��       �	2�{�Yc�A�
*

loss+�=�qr�       �	�}�Yc�A�
*

lossm�=�-       �	��}�Yc�A�
*

lossxT�<�y@       �	JB~�Yc�A�
*

loss	>�L�       �	��~�Yc�A�
*

losss$�=}4a�       �	`��Yc�A�
*

loss�=��Ox       �	A��Yc�A�
*

lossx�=$��       �	�ۀ�Yc�A�
*

loss�=vs�R       �	b���Yc�A�
*

lossW�=7_e       �	�+��Yc�A�
*

lossxY�=욕Q       �	�т�Yc�A�
*

loss�V�=̦P�       �	 ~��Yc�A�
*

loss�>K�F�       �	r��Yc�A�
*

lossf}P=w��K       �	����Yc�A�
*

lossҬ=d�P�       �	|_��Yc�A�
*

lossX�9=�IL       �	 ��Yc�A�
*

lossX�5=�ea�       �	����Yc�A�
*

loss�r=�I�       �	4��Yc�A�
*

loss��3=�}'3       �	�؇�Yc�A�
*

loss�=�-��       �	�q��Yc�A�
*

loss<�>���       �	A��Yc�A�
*

lossfi=1>�       �	�d��Yc�A�*

lossfϓ=���       �	���Yc�A�*

lossX�'>Ě��       �	U���Yc�A�*

loss���=�%e       �	�7��Yc�A�*

loss���=�i�       �	0׌�Yc�A�*

loss�1=P\'       �	�l��Yc�A�*

loss$5"=�ý;       �	k��Yc�A�*

loss}u�=܋��       �	ƥ��Yc�A�*

lossg��=�VA�       �	sK��Yc�A�*

lossA�9>L�6�       �	$폫Yc�A�*

loss��>2c��       �	L���Yc�A�*

loss�F<>�z�       �	P9��Yc�A�*

loss�E*>ag�       �	Dޑ�Yc�A�*

loss)�=����       �	s��Yc�A�*

lossS��=OvN^       �		��Yc�A�*

loss�/�=�K%b       �	+���Yc�A�*

lossn�(>�-�       �	�8��Yc�A�*

loss�kZ=���       �	�ϔ�Yc�A�*

loss���=��       �	�e��Yc�A�*

loss̽�=�YX�       �	P���Yc�A�*

loss���=�yYz       �	>���Yc�A�*

loss\>4�[h       �	�-��Yc�A�*

loss�h�=��E       �	~ė�Yc�A�*

loss�`b=߿�k       �	(e��Yc�A�*

loss�G=��       �	Q���Yc�A�*

loss�$>@���       �	�ę�Yc�A�*

loss�ť=��A       �	Hn��Yc�A�*

lossү�=ϫG�       �	#��Yc�A�*

loss�f�=P��        �	鷛�Yc�A�*

loss�>�=���       �	.W��Yc�A�*

loss1��=�¸v       �	����Yc�A�*

loss4�>~�Wq       �	���Yc�A�*

loss4$�=�K�       �	0��Yc�A�*

loss�Z=�o�[       �	�͞�Yc�A�*

loss�B>���y       �	nk��Yc�A�*

loss�q�=����       �	���Yc�A�*

loss�g1=�:       �	����Yc�A�*

loss\��=�Ɉ[       �	�\��Yc�A�*

lossjq>�B�[       �	N�Yc�A�*

loss��b=<�\       �	T���Yc�A�*

loss�|�=�Z��       �	�飫Yc�A�*

loss]��=E�U-       �	I���Yc�A�*

loss���=�m��       �	%��Yc�A�*

loss��='sY       �	[쥫Yc�A�*

lossn�9>`�Y       �	A���Yc�A�*

loss$��=U�C�       �	�q��Yc�A�*

loss��X=Pa�       �	>	��Yc�A�*

loss��a=\D       �	%���Yc�A�*

loss;�;=P0$�       �	EI��Yc�A�*

loss�z�=�.��       �	D੫Yc�A�*

loss���=�'�H       �	\w��Yc�A�*

loss�@�=&��       �	����Yc�A�*

loss��=Q�N&       �	�.��Yc�A�*

loss�V=��o       �	�Ȭ�Yc�A�*

loss3 �=�:h�       �	g`��Yc�A�*

loss�c>�7�       �	 ���Yc�A�*

loss��=�l�       �	����Yc�A�*

loss�7�=�}zE       �	C8��Yc�A�*

loss�Y�=!n��       �	կ�Yc�A�*

loss#�=�5�       �	�l��Yc�A�*

loss���=����       �	���Yc�A�*

loss�5B>�<�       �	K���Yc�A�*

loss�=>�m       �	�Q��Yc�A�*

loss��=Y9�1       �	���Yc�A�*

loss#��==K       �	N���Yc�A�*

loss��O=�j       �	�E��Yc�A�*

loss�U�=�ȡ       �	b���Yc�A�*

loss�g�="���       �	9���Yc�A�*

loss3b>u܀       �	�B��Yc�A�*

loss�$=���       �	`綫Yc�A�*

loss�k=�]@�       �	����Yc�A�*

lossI��=��b       �	�/��Yc�A�*

loss�8>G�r�       �	ʸ�Yc�A�*

lossf�]<��>       �	Q���Yc�A�*

lossM�=e���       �	C8��Yc�A�*

loss�E>�H��       �	�Ϻ�Yc�A�*

loss���=96       �	���Yc�A�*

loss��=���       �	]4��Yc�A�*

loss�ڹ=y�O       �	G伫Yc�A�*

loss��=�}��       �	����Yc�A�*

loss��<A~��       �	軾�Yc�A�*

loss�I�=��Aw       �	�k��Yc�A�*

loss�=׎y       �	)��Yc�A�*

loss��<B;�       �	����Yc�A�*

lossP>n�       �	p[��Yc�A�*

loss� �<�ۉ,       �	& «Yc�A�*

lossd�a=��;       �	��«Yc�A�*

loss#�=��m       �	�CëYc�A�*

lossHc�=D�
$       �	��ëYc�A�*

loss\* >�ǻp       �	u�īYc�A�*

loss��>�s�       �	�@ūYc�A�*

loss�z(>��W�       �	!ƫYc�A�*

lossu�">�/s�       �	�ƫYc�A�*

loss
r�=�B�X       �	�IǫYc�A�*

loss�=n��       �	WȫYc�A�*

loss��~=VBn�       �	D�ȫYc�A�*

loss�<�=�
�       �	BɫYc�A�*

lossm�=���=       �	�ɫYc�A�*

lossf
>����       �	��ʫYc�A�*

lossY�=��N	       �	x(˫Yc�A�*

lossܓ�=d��       �	��˫Yc�A�*

lossؤ;=���H       �	�b̫Yc�A�*

loss���=��0�       �	�ͫYc�A�*

lossT�>�|��       �	ϣͫYc�A�*

loss7��=�R�>       �	�EΫYc�A�*

loss�n�=Q�\       �	��ΫYc�A�*

loss���<��       �	ܟϫYc�A�*

loss�7=��Q       �	�>ЫYc�A�*

loss�{�=���       �	��ЫYc�A�*

loss)=�LuX       �	�ѫYc�A�*

loss��
=���T       �	7ҫYc�A�*

loss�>�CLs       �	�ҫYc�A�*

loss[��=g�f�       �	�QӫYc�A�*

lossNQ=K�o�       �	,�ӫYc�A�*

loss=�='7       �	�ԫYc�A�*

loss�V�=3�Ͼ       �	*իYc�A�*

lossx3=9�d
       �	��իYc�A�*

loss�"�=ݑ�       �	6Y֫Yc�A�*

loss��+>�n?       �	=�֫Yc�A�*

loss��=k|e9       �	|�׫Yc�A�*

loss�|=��       �	�.ثYc�A�*

loss��=~�q�       �	��ثYc�A�*

loss�`6>ȹ�       �	�}٫Yc�A�*

loss���=�i�       �	�ګYc�A�*

loss4�0=�]�.       �	��ګYc�A�*

loss�b=�&��       �	�I۫Yc�A�*

loss�/k=�H�       �	?�۫Yc�A�*

loss߂N>�(�       �	�|ܫYc�A�*

loss�� =A
�       �	�ݫYc�A�*

loss#n2>Bd       �	ؼݫYc�A�*

loss;=Β�       �	iUޫYc�A�*

loss�(0>&(t�       �	
�ޫYc�A�*

loss��[<7s�       �	ҏ߫Yc�A�*

loss� r=%,��       �	�&�Yc�A�*

loss�K�=0��       �	���Yc�A�*

loss?#>m*G�       �	���Yc�A�*

loss�˭=t���       �	
/�Yc�A�*

loss�5!>�*u�       �	���Yc�A�*

loss���=W;�       �	��Yc�A�*

lossVޖ=m[y7       �	Q�Yc�A�*

loss�V>8�%Q       �	���Yc�A�*

loss��o=���       �	.��Yc�A�*

loss���=���"       �	N)�Yc�A�*

lossz5E=��:       �	���Yc�A�*

lossPN�=+n�X       �	Zc�Yc�A�*

lossK��=�,S�       �	;�Yc�A�*

loss��=�C��       �	$��Yc�A�*

loss���=�u�       �	;�Yc�A�*

lossԍ!=폢�       �	���Yc�A�*

loss���<����       �	7p�Yc�A�*

loss4E=��]C       �	��Yc�A�*

lossث�=+f H       �	��Yc�A�*

loss�ݕ=c�       �	�F�Yc�A�*

loss�}�=>�I       �	���Yc�A�*

loss\�m=#�|�       �	����Yc�A�*

loss��[=���       �	1�Yc�A�*

loss(t>�E�M       �	���Yc�A�*

loss��>��R6       �	�h�Yc�A�*

loss���=|�v�       �	:�Yc�A�*

loss�)>؜��       �	���Yc�A�*

lossI-H=��^       �	�J�Yc�A�*

loss=a�=�RZ]       �	���Yc�A�*

lossBߛ=;&I�       �	Ja��Yc�A�*

loss
>�o��       �	���Yc�A�*

loss;��=D��       �	����Yc�A�*

lossm�=�)��       �	6<��Yc�A�*

lossN��=�b0       �	k���Yc�A�*

loss��=r��]       �	y��Yc�A�*

loss���=u�/u       �	� ��Yc�A�*

loss�T >8й       �	L���Yc�A�*

loss��a=���3       �	Na��Yc�A�*

loss�#=��[       �	Z��Yc�A�*

loss�<�=�D�:       �	 ���Yc�A�*

loss՛�=�P�       �	�M��Yc�A�*

loss��=[��       �	����Yc�A�*

loss���=�Ҵ       �	r���Yc�A�*

loss�V�=`4�       �	{f��Yc�A�*

lossȰd<RA#�       �	��Yc�A�*

loss*W�=t�w       �	���Yc�A�*

loss~�=��y�       �	~Q �Yc�A�*

lossm�
>
H       �	�
�Yc�A�*

loss�٭=+�<       �	���Yc�A�*

loss�xf=��7       �	?�Yc�A�*

lossM:.>��0�       �	���Yc�A�*

loss�D=�mp        �	�q�Yc�A�*

lossV�X=2�       �	��Yc�A�*

loss���=��Z*       �	,��Yc�A�*

lossH�u=W       �	6>�Yc�A�*

loss��>��       �	�+�Yc�A�*

loss(�(=!ꔚ       �	��Yc�A�*

loss��>*%�       �	�Z�Yc�A�*

loss=�Y=Ӿ��       �	��Yc�A�*

loss�Y�=3���       �	
��Yc�A�*

loss��O=QO�       �	�	�Yc�A�*

lossX	>��       �	��	�Yc�A�*

loss&�=Z_�       �	|G
�Yc�A�*

loss�*A=,{4�       �	��
�Yc�A�*

lossi��=ɡG�       �	�m�Yc�A�*

lossT��=q�       �	��Yc�A�*

loss���=2�&Z       �	Q��Yc�A�*

losszm�=Ku�       �	�E�Yc�A�*

loss̇o=�f�D       �	���Yc�A�*

loss�[&>S�{       �	�~�Yc�A�*

loss�g�=��Y�       �	 �Yc�A�*

lossŊ�=��       �	��Yc�A�*

lossƠ=�3k�       �	�S�Yc�A�*

loss�C5>��r�       �	]��Yc�A�*

lossf�6>�x4�       �	u��Yc�A�*

lossz�=�U
�       �	,�Yc�A�*

loss�i<��       �	���Yc�A�*

loss��= U�w       �	�Z�Yc�A�*

loss���=��#/       �	���Yc�A�*

loss�^�=@=r�       �	��Yc�A�*

loss��k=�8��       �	�Yc�A�*

loss�;	>?���       �	��Yc�A�*

loss��7=g��       �	�Z�Yc�A�*

loss�@&>1��>       �	���Yc�A�*

loss$2>s�n       �	���Yc�A�*

lossR�=n��       �	:[�Yc�A�*

lossͅ�=[D�       �	��Yc�A�*

loss��>?� -       �	���Yc�A�*

loss� ='�H�       �	�D�Yc�A�*

lossQ�y=��Y�       �	���Yc�A�*

loss�Z�=���       �	���Yc�A�*

loss�݋=X�6�       �	�b�Yc�A�*

loss�H=����       �	U��Yc�A�*

loss�L
=��[2       �	��Yc�A�*

loss �=��C       �	�B�Yc�A�*

loss`�'=�G��       �	���Yc�A�*

loss؊<[˧*       �	���Yc�A�*

loss��<��ʽ       �	�0 �Yc�A�*

lossM�<:VG       �	j!�Yc�A�*

loss.;]<��d       �	�."�Yc�A�*

loss���=�W       �	��"�Yc�A�*

lossvF�=�P�       �	y�#�Yc�A�*

loss�>��N       �	�$�Yc�A�*

lossM'�=�x�T       �	%�Yc�A�*

lossV��=�Vf�       �	�%�Yc�A�*

loss���=�d).       �	j&�Yc�A�*

loss%`�>����       �	�'�Yc�A�*

loss.E�=ʜ��       �	u�'�Yc�A�*

loss�='PN�       �	�_(�Yc�A�*

lossS�>���       �	\)�Yc�A�*

lossAԧ=�a+�       �	��)�Yc�A�*

loss?q=���h       �	w/*�Yc�A�*

loss܄g=��g       �	�*�Yc�A�*

loss3F>�Ҁ�       �	�+�Yc�A�*

loss�$>\=r5       �	m�,�Yc�A�*

loss��=�M|�       �	�\-�Yc�A�*

losswE>w���       �	& .�Yc�A�*

loss|nN=�� L       �	9�.�Yc�A�*

loss�8�=�"�       �	�3/�Yc�A�*

loss���=�K       �	��/�Yc�A�*

lossU=��S�       �	��0�Yc�A�*

lossl#= �1�       �	�1�Yc�A�*

loss��=�":�       �	k�1�Yc�A�*

loss��S>���       �	S]2�Yc�A�*

loss�Օ=�c�Q       �	3�Yc�A�*

loss2��=��       �	&�3�Yc�A�*

loss��=�6mI       �	�@4�Yc�A�*

loss�j/=��^       �	m�4�Yc�A�*

loss�,�==�2�       �	p}5�Yc�A�*

losscu<�N:       �	/6�Yc�A�*

lossJ� >�顀       �	$G7�Yc�A�*

lossIp=P5t       �	*�7�Yc�A�*

lossډ >XK��       �	K�8�Yc�A�*

loss�)>W�A�       �	�39�Yc�A�*

loss�C8=��<B       �	��9�Yc�A�*

loss0s<ќ�n       �	E�:�Yc�A�*

loss��d=��       �	�&;�Yc�A�*

loss�׍<�^6       �	��;�Yc�A�*

loss�^�<�v�       �	o�<�Yc�A�*

lossݗ>9J��       �	+3=�Yc�A�*

loss���>!p�{       �	��=�Yc�A�*

lossZ��=��y^       �	@�>�Yc�A�*

loss�(=��       �	�)?�Yc�A�*

loss�M>���       �	��?�Yc�A�*

loss4>�M(       �	�t@�Yc�A�*

loss�WE=���       �	A�Yc�A�*

lossJ��=ׇ�       �	b�A�Yc�A�*

loss&�>Ci��       �	6[B�Yc�A�*

loss�"'=T��       �	�C�Yc�A�*

loss���=����       �	�C�Yc�A�*

loss�5>�68       �	ȳD�Yc�A�*

loss���=�K       �	-[E�Yc�A�*

loss��B<)JN       �	�F�Yc�A�*

lossxY�=��5       �	$�F�Yc�A�*

loss!�\=����       �	�(G�Yc�A�*

loss#D$=\ߖ�       �	�G�Yc�A�*

loss?�=]\o�       �	�NH�Yc�A�*

loss�M�<G�9�       �	.�H�Yc�A�*

loss�>j=xִ�       �	�}I�Yc�A�*

loss��=�`�g       �	sJ�Yc�A�*

loss�'�=���8       �	�J�Yc�A�*

loss}�=��q,       �	�MK�Yc�A�*

lossӅ=>_       �	?�K�Yc�A�*

loss���=����       �	FxL�Yc�A�*

loss���=6V�       �	M�Yc�A�*

loss��]=�Bf       �	�M�Yc�A�*

loss�%>���       �	�UN�Yc�A�*

lossO��=`qp       �	��N�Yc�A�*

loss}:�=��       �	߉O�Yc�A�*

loss/&>h�$)       �	S"P�Yc�A�*

loss �<jC�       �	��P�Yc�A�*

loss4��=�� �       �	�]Q�Yc�A�*

loss(�<��       �	��Q�Yc�A�*

loss8��=���q       �	Q�R�Yc�A�*

loss��^=�v��       �	�CS�Yc�A�*

lossY#>ҳ�r       �	��S�Yc�A�*

loss�^�=B��7       �	��T�Yc�A�*

loss&�=���~       �	#.U�Yc�A�*

loss�s=GH�>       �	��U�Yc�A�*

loss�%�=tA�w       �	NcV�Yc�A�*

loss���=f��e       �	��V�Yc�A�*

lossH��=�/�4       �	��W�Yc�A�*

loss�Ԑ<.=�       �	�6X�Yc�A�*

loss��?=A�Q�       �	 VY�Yc�A�*

loss�e=�/ۜ       �	��Y�Yc�A�*

loss�ݏ=
f�       �	v�Z�Yc�A�*

loss{��<�	�       �	H3[�Yc�A�*

lossFH">�Y�       �	6�[�Yc�A�*

loss�)
>p@�       �	��\�Yc�A�*

loss�ߐ=}%�       �	5)]�Yc�A�*

loss3~�<B��       �	��]�Yc�A�*

loss���=�?,       �	�o^�Yc�A�*

loss�=��DU       �	�D_�Yc�A�*

loss@��<�Z�       �	��_�Yc�A�*

loss�q=o� �       �	�a�Yc�A�*

lossC��<�� �       �	��a�Yc�A�*

loss�|�<���       �	CUb�Yc�A�*

loss�k=��"       �	�b�Yc�A�*

lossJ� =Q�H;       �	��c�Yc�A�*

loss��N==�U       �	�Ad�Yc�A�*

loss��j<�7�       �	�d�Yc�A�*

lossSJ�;��ix       �	�e�Yc�A�*

loss��;Sak       �	�f�Yc�A�*

lossQ�d=�c �       �	��f�Yc�A�*

loss���=P�/�       �	,bg�Yc�A�*

loss/U]=�+|       �	*�g�Yc�A�*

loss��D<�L�a       �	��h�Yc�A�*

loss&�=?w"Q       �	E-i�Yc�A�*

loss��>TY�N       �	V�i�Yc�A�*

loss�y;�y�       �	Ihj�Yc�A�*

loss;Zp>�c�5       �	��j�Yc�A�*

loss�b�=MK��       �	C�k�Yc�A�*

loss
$>>2W�}       �	1%l�Yc�A�*

loss z=N��'       �	;�l�Yc�A�*

lossb�<��m       �	Mgm�Yc�A�*

loss[�=���O       �	qn�Yc�A�*

loss �>��bM       �	j�n�Yc�A�*

loss4$�=D-�       �	D�o�Yc�A�*

loss��=�G       �	��p�Yc�A�*

loss�ў=@*ڽ       �	m�q�Yc�A�*

loss��->y��       �	4r�Yc�A�*

loss��$>4�%�       �	
�r�Yc�A�*

lossq�V='2��       �	Nzs�Yc�A�*

loss�?>��	       �	9+t�Yc�A�*

loss7u:>m��s       �	J�t�Yc�A�*

lossj��=�Ճ�       �	�zu�Yc�A�*

loss�!=���       �	�]v�Yc�A�*

loss�>����       �	�w�Yc�A�*

loss���=���       �	��w�Yc�A�*

loss���<��4�       �	#Jx�Yc�A�*

loss��=ef]8       �	��x�Yc�A�*

loss�r�=|��       �	}�y�Yc�A�*

lossl��<���       �	��z�Yc�A�*

losse�6=��       �	�!{�Yc�A�*

lossr�<%[�       �	��{�Yc�A�*

lossv�=��       �	��|�Yc�A�*

loss���<N]^       �	�^}�Yc�A�*

lossg4>LI       �	�}�Yc�A�*

loss�^	>�c       �	]�~�Yc�A�*

lossϒz=У*       �	�R�Yc�A�*

loss���=Bk�       �	���Yc�A�*

loss���=�       �	����Yc�A�*

losso�<�K        �	3��Yc�A�*

loss
=B�4       �	4ځ�Yc�A�*

loss���<�L�       �	2v��Yc�A�*

losst>\=���       �	�	��Yc�A�*

loss�U�=u��       �	����Yc�A�*

loss�%>Iس�       �	E��Yc�A�*

lossc��=�&s�       �	�ބ�Yc�A�*

loss}..<�Ŏ�       �	ۅ��Yc�A�*

loss;��<�X�_       �	���Yc�A�*

lossD!�=���       �	^���Yc�A�*

loss ��=.�q�       �	���Yc�A�*

loss7��=��v�       �	㥈�Yc�A�*

loss�:�=��d       �	�I��Yc�A�*

loss�s�=��l       �	2劬Yc�A�*

loss��=o���       �	k���Yc�A�*

lossE�R="\�       �	����Yc�A�*

loss�M4=����       �	2>��Yc�A�*

loss���<��%       �	�؍�Yc�A�*

loss.��=�{3       �	����Yc�A�*

loss�Q`>�        �	>\��Yc�A�*

loss �&>5�Mr       �	���Yc�A�*

loss��>>+)Q       �	ˡ��Yc�A�*

loss���=R��       �	�?��Yc�A�*

lossa�=�T       �	N��Yc�A�*

loss{�=h��       �	¦��Yc�A�*

loss���< ƙ       �	�K��Yc�A�*

loss^�>�,�s       �	��Yc�A�*

loss���=�E��       �	_���Yc�A�*

loss8|�<g�       �	�{��Yc�A�*

loss)�:=�2�)       �	z��Yc�A�*

lossx�v=2��L       �	���Yc�A�*

loss���=&$I�       �	�곬Yc�A�*

lossHօ=��d       �	��Yc�A�*

loss�Z>?��?       �	U1��Yc�A�*

loss��<m��       �	 (��Yc�A�*

loss���=e�       �	���Yc�A�*

loss�I=��       �	�Ҹ�Yc�A�*

loss��d>v��       �	I��Yc�A�*

loss�=rr�       �	ךּ�Yc�A�*

loss}�&>M�BK       �	�i��Yc�A�*

loss=3=��       �	X��Yc�A�*

loss�!>�1�       �	8���Yc�A�*

loss�)=u�c       �	���Yc�A�*

lossZ��=��       �	����Yc�A�*

lossT�>�{       �	�d��Yc�A�*

loss�.P=�5-f       �	���Yc�A�*

loss���=r��z       �	���Yc�A�*

loss��>�1�       �	O=��Yc�A�*

losso�=ѭs       �	V¬Yc�A�*

loss��<���2       �	:�¬Yc�A�*

loss�{�=B���       �	�IìYc�A�*

loss�`;>�+a�       �	��ìYc�A�*

loss-�<=o䢳       �	8�ĬYc�A�*

loss=6�=��(       �	�4ŬYc�A�*

loss�p�<�dO       �	�ŬYc�A�*

loss�@=Ɓi�       �	�iƬYc�A�*

lossϱ�>�
�       �	�-ǬYc�A�*

loss)�>�{��       �	~�ǬYc�A�*

losss�>��z       �	�ZȬYc�A�*

loss6Q�==���       �	��ȬYc�A�*

loss���<J�~�       �	�ɬYc�A�*

loss�ʧ=�C3       �	�+ʬYc�A�*

loss���=�v�       �	��ʬYc�A�*

loss���=�'�F       �	x�ˬYc�A�*

loss2��=J��       �	�̬Yc�A�*

lossb� =�EN�       �	�$ͬYc�A�*

loss���=b��       �	��ͬYc�A�*

loss8�<4�I�       �	�`άYc�A�*

loss�=���       �	+�άYc�A�*

loss��=I��       �	��ϬYc�A�*

loss=(d�       �	��ЬYc�A�*

lossV2�>�ե       �	 ѬYc�A�*

loss���=����       �	��ѬYc�A�*

lossa-�<tF�       �	K�ҬYc�A�*

loss&[�;n,��       �	�HӬYc�A�*

loss�e�<ܾ�       �	@�ӬYc�A�*

loss��=���       �	vԬYc�A�*

lossm�w=���       �	.VլYc�A�*

loss`@O>j� �       �	+֬Yc�A�*

loss�c�=f��?       �	��֬Yc�A�*

loss=����       �	��׬Yc�A�*

loss�L�=�0fc       �	�HجYc�A�*

lossVx=��u       �	P٬Yc�A�*

loss�f#>ċ�D       �	A�٬Yc�A�*

loss�=�@̴       �	�QڬYc�A�*

loss4��=�6 d       �	D�ڬYc�A�*

lossac�=���       �	ʤ۬Yc�A�*

loss<�=ư��       �	�=ܬYc�A�*

loss��Q=BP       �	��ܬYc�A�*

losso[M=� p       �	�ݬYc�A�*

loss�v=7��c       �	2�ެYc�A�*

loss�b�=ᠸ       �	�g߬Yc�A�*

loss\�=�[       �	.�Yc�A�*

loss̷�<wqt|       �	���Yc�A�*

loss3��=���       �	�L�Yc�A�*

loss@�="S�       �	�E�Yc�A�*

loss\��=]v��       �	��Yc�A�*

loss�
�=��6       �	?��Yc�A�*

loss���=�\��       �	�>�Yc�A�*

loss�C�=�%�"       �	^��Yc�A�*

lossc�=����       �	�n�Yc�A�*

loss}H�<P�%       �	�3�Yc�A�*

lossԘ.=haS�       �	y��Yc�A�*

loss�^�=�%*�       �	 a�Yc�A�*

loss��	=�\��       �	��Yc�A�*

loss�ә=i�k�       �	��Yc�A�*

losso�6=䱟       �	�9�Yc�A�*

loss}�=��T�       �	1��Yc�A�*

loss�A�=���       �	Q��Yc�A�*

loss���=��A�       �	�,�Yc�A�*

loss���<��.       �	B��Yc�A�*

loss�</=�a�       �	�t��Yc�A�*

loss�	O>��"       �	��Yc�A�*

loss�v2=��I�       �	׽�Yc�A�*

loss�gL=� �       �	�P�Yc�A�*

lossJ��=��-       �	���Yc�A�*

loss@C>(�       �	6��Yc�A�*

loss�G=��       �	8-�Yc�A�*

loss�p�<q��       �	���Yc�A�*

loss&��<]�oX       �	�u�Yc�A�*

loss�=�=�U�s       �	�Yc�A�*

loss�a{=5w/�       �	��Yc�A�*

loss�@O=,-�A       �	h\��Yc�A�*

loss%5>��<�       �	;���Yc�A�*

loss�	=���0       �	Z���Yc�A�*

loss��<A^9       �	�A��Yc�A�*

loss��N>ͱU�       �	r���Yc�A�*

lossּ�=��p       �	�v��Yc�A�*

loss�+�=4���       �	���Yc�A�*

lossԜ�=��k�       �	����Yc�A�*

loss��=s�8       �	�D��Yc�A�*

loss��T=�9�       �	]���Yc�A�*

lossW9=��e}       �	w��Yc�A�*

lossv1;=O�Q~       �	8��Yc�A�*

loss�=�}Z       �	r���Yc�A�*

loss�S�=Da>       �	H��Yc�A�*

loss���=�;�       �	����Yc�A�*

loss�7&=�C��       �		���Yc�A�*

loss�BL=tT�       �	�P��Yc�A�*

loss<�p==:UL       �	����Yc�A�*

loss�>����       �	����Yc�A�*

lossA�=�,�
       �	� �Yc�A�*

lossl5Q<�       �	9� �Yc�A�*

loss�$=	}       �	�K�Yc�A�*

loss`�=)��       �	T��Yc�A�*

loss(.�=(��       �	���Yc�A�*

loss!�+>�n�       �	|(�Yc�A�*

loss�Ȏ=2J��       �	���Yc�A�*

lossLP=��8       �	�`�Yc�A�*

loss~I=���<       �	o��Yc�A�*

loss�7S=7WGZ       �	���Yc�A�*

loss��<M�       �	�f�Yc�A�*

lossRm�=�X�       �	��Yc�A�*

lossu�=��|\       �	���Yc�A�*

loss��<X[�k       �	�]�Yc�A�*

loss��<x�ǧ       �	O	�Yc�A�*

loss��H=A�\�       �	ˠ	�Yc�A�*

loss�k=��       �	�7
�Yc�A�*

loss�N<�/�G       �	��
�Yc�A�*

loss��=�36�       �	m�Yc�A�*

loss�=����       �	��Yc�A�*

loss.�=�J,�       �	֭�Yc�A�*

loss�_�=�p�       �	�I�Yc�A�*

lossڝ�<�#��       �	���Yc�A�*

loss��=�7��       �	3��Yc�A�*

loss�=E2��       �	1(�Yc�A�*

loss$��<��=�       �	���Yc�A�*

loss>9=��2B       �	�a�Yc�A�*

lossd�[=e��       �	��Yc�A�*

loss�=T'��       �	W��Yc�A�*

loss�>��t�       �	6x�Yc�A�*

loss�w�=<�}       �	3�Yc�A�*

loss��=5��       �	��Yc�A�*

lossI��;��;       �	L��Yc�A�*

lossl}=�"�       �	=E�Yc�A�*

lossWB�=<�       �	���Yc�A�*

loss�S~>!Tp       �	���Yc�A�*

loss�q�=���	       �	o.�Yc�A�*

loss�j)>�Y��       �	o��Yc�A�*

lossj��=�e        �	+m�Yc�A�*

loss�G=���       �	�Yc�A�*

loss�h�<ƈ|�       �	i��Yc�A�*

loss�|�=����       �	G�Yc�A�*

lossh^=�E        �	���Yc�A�*

lossD�A="2;\       �	-{�Yc�A�*

loss��=����       �	
/�Yc�A�*

losspC=�L�       �	S��Yc�A�*

loss��=^N�       �	��Yc�A�*

loss��e=��kb       �	'�Yc�A�*

loss���< n��       �	ͭ�Yc�A�*

loss�B=>ʾT       �	�D�Yc�A�*

loss�8�=d��       �	���Yc�A�*

loss��=p���       �	�q �Yc�A�*

loss��>�@m       �	�!�Yc�A�*

loss���=�`WS       �	s�!�Yc�A�*

loss2I�=��$2       �	�6"�Yc�A�*

loss�K=�p^u       �	r�"�Yc�A�*

loss�n
=u�`Q       �	Gt#�Yc�A�*

loss�i5<N�G       �	xz$�Yc�A�*

loss�W{=��yf       �	+�%�Yc�A�*

loss�P=iR�       �	~&�Yc�A�*

loss�A=R�o       �	[�&�Yc�A�*

loss��=r�u�       �	�K'�Yc�A�*

loss �=��-       �	��'�Yc�A�*

loss(��=��d       �	w(�Yc�A�*

loss&.�<<��       �	�	)�Yc�A�*

loss	�)=n*�       �	G�)�Yc�A�*

loss�!=���       �	�O*�Yc�A�*

lossb�>PG       �	��*�Yc�A�*

loss�t=U�T�       �	z�+�Yc�A�*

loss?��=�#�       �	
,,�Yc�A�*

loss���<��4       �	��,�Yc�A�*

loss��4<��e       �	ob-�Yc�A�*

loss;�D=k�       �	y.�Yc�A�*

lossĸe=�90�       �	Ǟ.�Yc�A�*

losszZ�=|$�       �	4/�Yc�A�*

lossU� >}��       �	U�/�Yc�A�*

loss���=u���       �	��0�Yc�A�*

lossA��=�� �       �	fO1�Yc�A�*

loss�<�Z       �	��1�Yc�A�*

loss�h=L��       �	��2�Yc�A�*

loss��l=���       �	�$3�Yc�A�*

loss�H�<�VD       �	��3�Yc�A�*

loss�t[=�Y�$       �	�`4�Yc�A�*

loss��d=�;f       �	��4�Yc�A�*

lossc�=n�_]       �	ݙ5�Yc�A�*

loss૪=���       �	�36�Yc�A�*

loss;��=�I	       �	J�7�Yc�A�*

loss��'=Fph	       �	4i8�Yc�A�*

loss� =��G       �	^9�Yc�A�*

loss��!=��m       �	ݱ9�Yc�A�*

loss47=��T~       �	=_:�Yc�A�*

loss��>�,�       �	��;�Yc�A�*

loss�}�=�t��       �	+2<�Yc�A�*

loss=>J2��       �	��<�Yc�A�*

loss {>�:DV       �	=�=�Yc�A�*

losso�">ٕ       �	�P>�Yc�A�*

loss͞�=��L|       �	��>�Yc�A�*

loss�ِ=$��>       �	ĳ?�Yc�A�*

loss���<K�!�       �	�M@�Yc�A�*

lossmn>��       �	.�@�Yc�A�*

loss
S>𕂟       �	�~A�Yc�A�*

loss���<>_j�       �	�B�Yc�A�*

loss��0=�:       �	q�B�Yc�A�*

loss`R�=Dw]       �	v�C�Yc�A�*

loss�"�=OF+�       �	�"D�Yc�A�*

loss�W=�/$�       �	+�D�Yc�A�*

lossa>廭=       �	oeE�Yc�A�*

loss}�=�S�w       �	iF�Yc�A�*

loss��=y0O       �	�F�Yc�A�*

lossv4�=,qS�       �	eG�Yc�A�*

loss��D=}r6       �	H�Yc�A�*

loss 1�=�%�D       �	K�H�Yc�A�*

losst�=��޶       �	YMI�Yc�A�*

loss�U�=�܊       �	��I�Yc�A�*

lossW��=؍E       �	'�J�Yc�A�*

loss��=��7       �	&WK�Yc�A�*

loss�"y<,߾X       �	�
L�Yc�A�*

lossM.=��       �	h�L�Yc�A�*

lossM�u=��Q       �	�OM�Yc�A�*

loss3��=��$�       �	*:N�Yc�A�*

loss@�{=���g       �	��N�Yc�A�*

loss���=?�       �	ѓO�Yc�A�*

loss��=���G       �	^P�Yc�A�*

loss���=�yb�       �	�)Q�Yc�A�*

loss�Y�<���       �	��Q�Yc�A�*

loss�>�Ԝ"       �	�qR�Yc�A�*

loss�И=�#�       �	�S�Yc�A�*

loss1�9=�3       �	�S�Yc�A�*

loss
K�=�RN�       �	�?T�Yc�A�*

lossW�}=&�]�       �	��T�Yc�A�*

loss�d�<֠8�       �	(|U�Yc�A�*

loss���<(D��       �	V�Yc�A�*

loss�=7=lT_       �	@�V�Yc�A�*

loss�=/�       �	ZW�Yc�A�*

loss�>�}(�       �	T�W�Yc�A�*

lossJb=�x�}       �	��X�Yc�A�*

loss�n�=��b�       �	!>Y�Yc�A�*

loss���=�N#�       �	��Y�Yc�A�*

loss���<��U�       �	�Z�Yc�A�*

loss`��=h�mh       �	�[�Yc�A�*

lossd=���       �	y]\�Yc�A�*

loss�2�=��0�       �	]�Yc�A�*

loss$�K=��F�       �	��]�Yc�A�*

loss�Y�=�B��       �	�Y^�Yc�A�*

loss���=.�~�       �	b�^�Yc�A�*

loss��>�(�       �	H�_�Yc�A�*

loss;�=�`��       �	J_`�Yc�A�*

lossoA=���       �	_�a�Yc�A�*

loss���=�b,Y       �	�>b�Yc�A�*

losssB�=h��       �	;�b�Yc�A�*

losst�=!�b�       �	]�c�Yc�A�*

lossb>t��       �	�d�Yc�A�*

loss��>��       �	he�Yc�A�*

loss��=��       �	4f�Yc�A�*

lossR�&=5�V�       �	*g�Yc�A�*

loss.}�=�Ƃ_       �	��g�Yc�A�*

lossq��=�m�       �	n�h�Yc�A�*

losso��<d[�       �	h i�Yc�A�*

loss�<��#       �	��i�Yc�A�*

loss<�=�ۻ�       �	�pj�Yc�A�*

loss)�>}�Ĺ       �	�Lk�Yc�A�*

lossRH>���m       �	�l�Yc�A�*

loss$nv=K��       �	��l�Yc�A�*

loss뗜=W�z�       �	�Vm�Yc�A�*

lossxw*<H�       �	C�m�Yc�A�*

loss{�E=�Vt�       �	�n�Yc�A�*

lossd4?<,Ջv       �	�?o�Yc�A�*

loss�7�<y.       �	�o�Yc�A�*

lossm��=��r2       �	A�p�Yc�A�*

loss�k�;���$       �	%"q�Yc�A�*

loss�I�<����       �	Q�q�Yc�A�*

loss0d=9�       �	�Tr�Yc�A�*

loss��>n�g�       �	D�r�Yc�A�*

loss��=��)	       �	��s�Yc�A�*

loss>���>       �	#Jt�Yc�A�*

loss��>�6�.       �	A�t�Yc�A�*

loss85�=�4$�       �	x�u�Yc�A�*

loss}f�<��(:       �	�Dv�Yc�A�*

loss���=r��       �	)�v�Yc�A�*

lossŜ^=�{       �	Ɗw�Yc�A�*

loss�P�=��r�       �	�2x�Yc�A�*

loss���=O^�4       �	v�x�Yc�A�*

loss��<>aM�>       �	Ӄy�Yc�A�*

loss��;���M       �	\z�Yc�A�*

loss��:>�L�        �	v�z�Yc�A�*

loss��/=��?{       �	1^{�Yc�A�*

loss���=��H       �	S|�Yc�A�*

loss|,�=R+�%       �	n�|�Yc�A�*

loss)a�=�3��       �	�@}�Yc�A�*

loss�Z�=R)��       �	z�}�Yc�A�*

loss�AE=f�2�       �	^�~�Yc�A�*

loss��`=!� w       �	��Yc�A�*

loss�#>+I�       �	g��Yc�A�*

lossE�P=*��       �	bր�Yc�A�*

loss��x=l�e       �	�m��Yc�A�*

loss�=��{�       �	U��Yc�A�*

lossV�=���       �	����Yc�A�*

lossm��<�7>@       �	ߤ��Yc�A�*

loss*�S>�o�       �	tB��Yc�A�*

lossDP�<H+6r       �	�䄭Yc�A�*

lossw�X<^y`)       �	Ƈ��Yc�A�*

losse̠=��       �	k(��Yc�A�*

loss�]>U;�1       �	�Ć�Yc�A�*

loss%�`<�Fw^       �	����Yc�A�*

loss��6=u��r       �	�$��Yc�A�*

loss�=�*�       �	]È�Yc�A�*

loss<"/>�J       �	�^��Yc�A�*

loss��=���T       �	���Yc�A�*

loss�:�<���       �	u���Yc�A�*

loss$-=��`�       �	k*��Yc�A�*

loss�A=!�p3       �	����Yc�A�*

loss\�7>~��       �	h��Yc�A�*

loss�
�<I~�x       �	����Yc�A�*

loss��=��i�       �	ݔ��Yc�A�*

loss�ǿ=Z�
       �	�4��Yc�A�*

lossO�=�<:       �	�ˎ�Yc�A�*

loss�;Ilh_       �	�b��Yc�A�*

loss6�H=\�       �	y��Yc�A�*

loss���=4kNi       �	����Yc�A�*

loss�E�=y�6�       �	Y6��Yc�A�*

lossa*~=Q��F       �	�ϑ�Yc�A�*

loss!�[>����       �	Ie��Yc�A�*

loss��K>�m��       �	���Yc�A�*

lossO��<F��       �	j���Yc�A�*

loss��=k,4�       �	}@��Yc�A�*

lossv�h=t�Y       �	�蔭Yc�A�*

loss���=Q�       �	����Yc�A�*

losso4�=�Ru~       �	J%��Yc�A�*

losseG�=5�W       �	쾖�Yc�A�*

loss!�a=||       �	`Z��Yc�A�*

loss��J=^:�Y       �	��Yc�A�*

loss6"�=����       �	P���Yc�A�*

loss�<ڶRL       �	�%��Yc�A�*

loss[�<!��r       �	����Yc�A�*

loss=��=|^�       �	�V��Yc�A�*

loss��=��y       �	* ��Yc�A�*

losslW�=B[�-       �	ꖛ�Yc�A�*

loss�γ=�6��       �	/��Yc�A�*

loss�8=K�Ґ       �	"��Yc�A�*

loss�'[<��F1       �	A���Yc�A�*

loss{ B>?��Q       �	�V��Yc�A�*

loss���=N�3       �	L���Yc�A�*

loss\i�=�^ԛ       �	z���Yc�A�*

loss_�	>�/�s       �	!>��Yc�A�*

loss��8=6��D       �	�꠭Yc�A�*

lossWz�<$�j       �	���Yc�A�*

loss1j>=�j�       �	���Yc�A�*

loss]WE=st       �	�¢�Yc�A�*

loss��=�%��       �	6s��Yc�A�*

loss�f�=�+�       �	&��Yc�A�*

loss��|=�R�       �	ä�Yc�A�*

loss�=�S�       �	�g��Yc�A�*

loss���=��B�       �	��Yc�A�*

loss��*>�Kx       �	����Yc�A�*

loss���=��       �	q9��Yc�A�*

loss�OF=����       �	�ҧ�Yc�A�*

losst�=���       �	�p��Yc�A�*

loss�&M=B���       �	<M��Yc�A�*

lossac�=��4�       �	&䩭Yc�A�*

lossR'�=(Z�       �	�{��Yc�A�*

loss�H�=�}��       �	���Yc�A�*

loss�0�<3Ľl       �	����Yc�A�*

loss|3�=4��4       �	�W��Yc�A�*

loss�3�=��P�       �	�쬭Yc�A�*

lossl�=� �U       �	o���Yc�A�*

loss�u�<Dv�       �	H��Yc�A�*

loss��=�̖�       �	5���Yc�A�*

loss,��=��T�       �	vO��Yc�A�*

loss݈=�7?       �	语Yc�A�*

lossK��=��       �	���Yc�A�*

loss��=͓�3       �	����Yc�A�*

lossldA=��!]       �	-[��Yc�A�*

lossw>ϭ��       �	��Yc�A�*

lossE=�O�       �	
���Yc�A�*

loss��K=s��       �	�+��Yc�A�*

loss��<��9       �	:̴�Yc�A�*

lossr�=��}/       �	Cr��Yc�A�*

loss�\�=����       �	m��Yc�A�*

loss�ϝ<����       �	:維Yc�A�*

loss��=��K�       �	䆷�Yc�A�*

loss��=��       �	�$��Yc�A�*

lossn�4=�녾       �	.Ƹ�Yc�A�*

loss�%=�,�F       �	/l��Yc�A�*

loss-��=S�       �	K��Yc�A�*

loss0ʡ=GH��       �	Q���Yc�A�*

lossTnp=`6�'       �	U��Yc�A�*

loss��>S�҅       �	Y���Yc�A�*

loss���=G��-       �	R*��Yc�A�*

loss��=m�DH       �	Oɽ�Yc�A�*

losslt=�w��       �	l���Yc�A�*

loss�=�G       �	K��Yc�A�*

lossA�=b�{       �	4f��Yc�A�*

loss�^�=���z       �	O��Yc�A�*

loss�D<$�l4       �	C­Yc�A�*

loss��	={c�K       �	��­Yc�A�*

lossHD�=Z�T�       �	nRíYc�A�*

loss�&>���N       �	��íYc�A�*

loss#�=��d$       �	�ĭYc�A�*

loss%��=X��P       �	�PŭYc�A�*

loss���=��       �	 �ŭYc�A�*

lossdD>��ߊ       �	]�ƭYc�A�*

lossԐu=�My       �	$%ǭYc�A�*

loss]�=�       �	#�ǭYc�A�*

loss�͞=�D;       �	%YȭYc�A�*

loss�>��M�       �	��ȭYc�A�*

loss�w�=�=��       �	��ɭYc�A�*

loss�&�=�_��       �	&ʭYc�A�*

loss#s�=6���       �	7�ʭYc�A�*

loss�Y�=+��S       �	�[˭Yc�A�*

loss(��=�,B7       �	�˭Yc�A�*

loss��M=���       �	��̭Yc�A�*

loss-Ĝ=�J��       �	\TͭYc�A�*

loss�P0=b��       �	N�ͭYc�A�*

loss�=��H       �	��έYc�A�*

lossf-=�(j       �	�ϭYc�A�*

loss�
�<E�       �	.�ϭYc�A�*

loss�}C=&G�       �	FAЭYc�A�*

lossW3>W�3�       �	��ЭYc�A�*

loss:5�=g�       �	�xѭYc�A�*

loss�\1=�       �	xҭYc�A�*

loss��=%��       �	��ҭYc�A�*

lossܞ�<"�J       �	WAӭYc�A�*

loss�L%=و�       �	��ӭYc�A�*

lossx�R>�Gu       �	=|ԭYc�A�*

loss��<@o       �	 խYc�A�*

loss�7=�u)�       �	_�խYc�A�*

loss}��=ۭת       �	M֭Yc�A�*

losswќ=���Q       �	�׭Yc�A�*

loss{<��{       �	u�׭Yc�A�*

lossr��<�-�       �	]SحYc�A�*

loss�@�=��@       �	n�حYc�A�*

loss�J>t}�       �	p�٭Yc�A�*

loss��=���       �	�0ڭYc�A�*

loss��=Ňu        �	u�ڭYc�A�*

loss8}<=s_�       �	�jۭYc�A�*

losse�/=#;|       �	
ܭYc�A�*

loss4��=7�~       �	 �ܭYc�A�*

loss�F0=��mg       �	)>ݭYc�A�*

loss�<w�K       �	I�ݭYc�A�*

loss�Ӿ<]Oz>       �	ɪޭYc�A�*

loss�>�+,�       �	MK߭Yc�A�*

loss��=�w�       �	_�߭Yc�A�*

loss�B=��w.       �	ӄ�Yc�A�*

loss�	=',+h       �	c'�Yc�A�*

loss*)V=���8       �	��Yc�A�*

loss�^=��       �	f�Yc�A�*

loss!<NB*�       �	. �Yc�A�*

loss*�G>uۼ�       �	z��Yc�A�*

loss\8�=���a       �	�x�Yc�A�*

loss���=���       �	~8�Yc�A�*

loss�(�=�qCR       �	%�Yc�A�*

loss�PT<.>��       �	M��Yc�A�*

loss���<��<B       �	&��Yc�A�*

loss�t=�%��       �	/l�Yc�A�*

loss���<~��A       �	�n�Yc�A�*

loss��<^ت�       �	f�Yc�A�*

loss�X1>���       �	~:�Yc�A�*

loss�">�h�       �	�x�Yc�A�*

loss���=�U�       �	,E��Yc�A�*

lossA��=�LQ       �	^���Yc�A�*

loss��>r��I       �	���Yc�A�*

loss
�=�}R�       �	ʦ�Yc�A�*

loss��==.%T       �	�b�Yc�A�*

loss�2�<���       �	� �Yc�A�*

loss�S�=��M�       �	p��Yc�A�*

lossN=���       �	|}�Yc�A�*

loss�Ӗ=��}�       �	��Yc�A�*

lossW=)J7b       �	��Yc�A�*

lossa�!=�R�(       �	�C��Yc�A�*

loss�L�<b[%       �	MI��Yc�A�*

loss1,M<(�       �	D4��Yc�A�*

loss="�<.�.<       �	\���Yc�A�*

loss38)<�GK	       �	���Yc�A�*

loss��=�~��       �	*��Yc�A�*

loss��<O��0       �	ٲ��Yc�A�*

loss(�!=s7/       �	X��Yc�A�*

loss��=����       �	g���Yc�A�*

loss���={L�       �	8���Yc�A�*

loss�ڵ<�z�       �	���Yc�A�*

lossR�<
EiR       �	����Yc�A�*

loss�*�=͢��       �	�K��Yc�A�*

loss4��=�6+       �	=���Yc�A�*

lossJ}
=�a�       �	$���Yc�A�*

lossњ�=���       �	�9��Yc�A�*

loss.��<���a       �	���Yc�A�*

loss�P>"z/Z       �	�C �Yc�A�*

lossFW=rc��       �	U� �Yc�A�*

lossd�=��	?       �	^��Yc�A�*

lossoR�=@��       �	�!�Yc�A�*

loss� _<�8�       �	���Yc�A�*

loss�R<=�D�       �	�]�Yc�A�*

lossi�G=��       �	���Yc�A�*

loss
�2>��       �	���Yc�A�*

loss��h=���       �	^0�Yc�A�*

lossO؄=�Quh       �	���Yc�A�*

loss��)=1ی#       �	Di�Yc�A�*

loss
�k= k�       �	S�Yc�A�*

loss@�E<�Զ       �	?��Yc�A�*

lossI�=D/��       �	o��Yc�A�*

loss=�K<��\�       �	�	�Yc�A�*

loss?�W=XOyC       �	)�	�Yc�A�*

loss� >=�z�       �	_E
�Yc�A�*

loss�/=Pt�K       �	��
�Yc�A�*

lossO�=��N�       �	�~�Yc�A�*

loss��c=���3       �	C�Yc�A�*

loss�>G��7       �	���Yc�A�*

loss�0?<�~-�       �	�P�Yc�A�*

lossK �<�qR       �	��Yc�A�*

lossa��=LY��       �	ɰ�Yc�A�*

loss=n�;ӿ}       �	;V�Yc�A�*

loss6;�d       �	
d�Yc�A�*

loss�9<;j�       �	��Yc�A�*

lossMq�=p�Ml       �	 ��Yc�A�*

loss��"=R�"       �	N�Yc�A�*

loss�{=�;o       �	m��Yc�A�*

loss��<T\��       �	��Yc�A�*

lossW�W=�6�       �	��Yc�A�*

loss�i;˩3�       �	B��Yc�A�*

loss�;5�f�       �	G�Yc�A�*

loss��#;w       �	���Yc�A�*

loss!�;	�g�       �	���Yc�A�*

loss���=כ�       �	,+�Yc�A�*

loss�L=�[�       �	i��Yc�A�*

lossE�I;��<�       �	�[�Yc�A�*

loss���=ͦ�       �	���Yc�A�*

lossI;�>�O9k       �	7��Yc�A�*

loss��'<\��       �	�'�Yc�A�*

loss?`;>�;4?       �	N��Yc�A�*

loss��=�|-g       �	x�Yc�A�*

loss�ۊ>T�!f       �	]�Yc�A�*

losss��<V�"k       �	5��Yc�A�*

loss]�=�q�q       �	�M�Yc�A�*

loss�D�=���       �	��Yc�A�*

lossMb�=��       �	Y��Yc�A�*

losslͮ=�[�       �	� �Yc�A�*

loss�_�=HS�       �	��Yc�A�*

loss���=����       �	Uk �Yc�A�*

loss���=9"�@       �	�!�Yc�A�*

loss9)>�E       �	��!�Yc�A�*

lossw)�=��        �	T"�Yc�A�*

loss��=���       �	��"�Yc�A�*

lossŋ<>b���       �	�#�Yc�A�*

lossӥ=xWy       �	��$�Yc�A�*

loss[�=6�+�       �	�&�Yc�A�*

loss_�l=l�V        �	��&�Yc�A�*

loss�I�<��:M       �	�e'�Yc�A�*

loss��1=�LY�       �	 �'�Yc�A�*

loss��==�F       �	ę(�Yc�A�*

loss��k=��3       �	�6)�Yc�A�*

loss���<�\8
       �	V�)�Yc�A�*

loss��<�[�       �	�k*�Yc�A�*

loss�t�<��
       �	@+�Yc�A�*

loss�uE=�p�       �	`�+�Yc�A�*

loss=���       �	�F,�Yc�A�*

lossO"�=��km       �	��,�Yc�A�*

lossO�=�Z��       �	�|-�Yc�A�*

loss��
=�]�       �	�.�Yc�A�*

loss�%�=����       �	�.�Yc�A�*

lossڹ�=�+�J       �	JB/�Yc�A�*

loss׵�<ߎ��       �	��/�Yc�A�*

loss�k=Td[       �	��0�Yc�A�*

loss.�C=��U�       �	B@1�Yc�A�*

lossv$r=��ߍ       �	b�1�Yc�A�*

loss�n�=m��G       �	7�2�Yc�A�*

loss{��=�9�a       �	�`3�Yc�A�*

lossS�D=#�O�       �	�4�Yc�A�*

loss3.�<��x       �	?�4�Yc�A�*

lossl4h<����       �	�A5�Yc�A�*

loss��=��38       �	��5�Yc�A�*

lossῊ=���(       �	��6�Yc�A�*

loss���<NH9�       �	�,7�Yc�A�*

loss��=�B�       �	x�7�Yc�A�*

loss/�s=���:       �	�v8�Yc�A�*

loss��u<�Ue       �	<9�Yc�A�*

loss���=t��[       �	�9�Yc�A�*

loss�<F��>       �	/R:�Yc�A�*

loss�)2=�pyM       �	,�:�Yc�A�*

loss��D=�ng�       �	�=R�Yc�A�*

loss�=� �K       �	��R�Yc�A�*

loss�K�=UZ       �	�rS�Yc�A�*

loss�V�=BM5       �	�T�Yc�A�*

loss�@B=�g�s       �	~�T�Yc�A�*

loss�^.=9m��       �	�EU�Yc�A�*

loss�Բ=�U�       �	��U�Yc�A�*

lossأ�<H�r�       �	lyV�Yc�A�*

lossm��=ȗ|]       �	W�Yc�A�*

lossO�=��ė       �	��W�Yc�A�*

loss�K�<���       �	�EX�Yc�A�*

loss&�$<`߽       �	��X�Yc�A�*

loss�uA=���       �	�Y�Yc�A�*

loss���=\C��       �	�LZ�Yc�A�*

loss�q=r�A       �	��Z�Yc�A�*

loss��={�s�       �	�\�Yc�A�*

loss?�<�z;I       �	¦\�Yc�A�*

loss��T=�
�       �	�y]�Yc�A�*

loss`�s<wi3       �	�^�Yc�A�*

lossT�f>o�^�       �	��^�Yc�A�*

lossr��=�8�       �	�P_�Yc�A�*

lossJ�=�(K�       �	�_�Yc�A�*

lossi��=͕��       �	P�`�Yc�A�*

loss)@�=� �       �	s-a�Yc�A�*

loss��+=@C�O       �	��a�Yc�A�*

loss!�X=ף�       �	�}b�Yc�A�*

loss���<+�       �	F"c�Yc�A�*

lossѿ�<���1       �	��c�Yc�A�*

loss��=�n       �	�jd�Yc�A�*

loss���=ß��       �	'�e�Yc�A�*

loss�Q�=Z�v       �	��f�Yc�A�*

loss��.=��^       �	�Xg�Yc�A�*

loss��&=�%h       �	�.h�Yc�A�*

loss%=>_��       �	��h�Yc�A�*

loss�Ȗ=ܤ�       �	hvi�Yc�A�*

loss���=B�,W       �	�Jj�Yc�A�*

lossw��;]+s       �	g�k�Yc�A�*

lossai=��h       �	t(l�Yc�A�*

lossO�)>�x�       �	�l�Yc�A�*

lossM{$><M�       �	�`m�Yc�A�*

loss�3�=����       �	��m�Yc�A�*

lossq=*>;�       �	l�n�Yc�A�*

lossNgD<w�       �	�To�Yc�A�*

loss3�=���h       �	��o�Yc�A�*

loss��=���       �	��p�Yc�A�*

lossɡ=�+}�       �	i�q�Yc�A�*

loss[�=����       �	=,r�Yc�A�*

loss��=v��       �	��r�Yc�A�*

loss�CY=zoG6       �	W^s�Yc�A�*

loss���;��~       �	��s�Yc�A�*

loss.�<5�a�       �	1�t�Yc�A�*

loss1}0=E5�Z       �	�7u�Yc�A�*

loss�<q�p�       �	��u�Yc�A�*

loss���=�s��       �	vv�Yc�A�*

loss^�<S���       �	�Iw�Yc�A�*

loss��<Ht��       �	y�w�Yc�A�*

loss3<8��Z       �	��x�Yc�A�*

loss��:<8�ڱ       �	�!y�Yc�A�*

loss��p=}^��       �	��y�Yc�A�*

loss���=���       �	�kz�Yc�A�*

loss��Q> �Υ       �	�{�Yc�A�*

loss
ť=[�ƍ       �	��{�Yc�A�*

loss��=�K�0       �	3R|�Yc�A�*

loss:��=��Z       �	��|�Yc�A�*

loss��a=��z�       �	�}�Yc�A�*

loss���<;[W�       �	 ;~�Yc�A�*

loss�=´<f       �	1�~�Yc�A�*

lossNo�<���       �	���Yc�A�*

loss�*�=Τ�l       �	\ ��Yc�A�*

loss�L�=�) �       �	8���Yc�A�*

loss��{=��-�       �	ap��Yc�A�*

lossܕ�<�FF?       �	���Yc�A�*

loss|��=;�CN       �	����Yc�A�*

loss(qa=��       �	�D��Yc�A�*

lossHǧ<��}�       �	;��Yc�A�*

loss�D�<�j��       �	۾��Yc�A�*

loss`�(=�1�U       �	kc��Yc�A�*

loss��e=	�`�       �	�
��Yc�A�*

loss�9A=}6z�       �	����Yc�A�*

loss��W=,7��       �	�_��Yc�A�*

lossﴒ=�*{b       �	����Yc�A�*

lossxOk=�(s       �	F���Yc�A�*

loss��:=�KN       �	�6��Yc�A�*

lossО<����       �	�ɉ�Yc�A�*

loss��S=�+��       �	k��Yc�A�*

loss��=$lMh       �	���Yc�A�*

loss[�[<¼       �	氋�Yc�A�*

lossg�<��$�       �	�P��Yc�A�*

lossA�O=�u4e       �	�댮Yc�A�*

loss�]F=�(��       �	r���Yc�A�*

lossD��=�C       �	k(��Yc�A�*

lossqI>�!�       �	�Ȏ�Yc�A�*

loss�Bm=���       �	�a��Yc�A�*

loss��,>�w�:       �	<���Yc�A�*

loss�)�=�� �       �	g���Yc�A�*

loss��=��_�       �	�5��Yc�A�*

lossHU�=(new       �	T䑮Yc�A�*

lossw�2=����       �	y��Yc�A�*

loss���=�y       �	&��Yc�A�*

loss���=�N       �	ø��Yc�A�*

lossi�<��       �	fO��Yc�A�*

loss�=��q�       �	S씮Yc�A�*

loss�M�=�E�       �	���Yc�A�*

loss`d�<��       �	28��Yc�A�*

loss�"�=�Av       �	5Ӗ�Yc�A�*

loss��>�B8P       �	�o��Yc�A�*

lossO��<�@�        �	���Yc�A�*

loss/�<�l��       �	D���Yc�A�*

lossj�=rE�(       �	�D��Yc�A�*

loss=r/=���       �	z♮Yc�A�*

losss��=*23�       �	J}��Yc�A�*

loss�(�<�(��       �	��Yc�A�*

loss�I�=���       �	+Û�Yc�A�*

lossA�!=l��7       �	�]��Yc�A�*

loss2��<f�!       �	���Yc�A�*

loss$�<�cW�       �	ᙝ�Yc�A�*

loss�H=��       �	�7��Yc�A�*

lossB�=���       �	�Ӟ�Yc�A�*

loss��=X�       �	�j��Yc�A�*

loss%�K=�Nq�       �	���Yc�A�*

loss�=Q�ð       �	����Yc�A�*

loss�	�=G j       �	Y���Yc�A�*

loss��>��G       �	����Yc�A�*

loss�k�<�ڱ�       �	�$��Yc�A�*

loss���</�`�       �	����Yc�A�*

lossza�=��c       �	W_��Yc�A�*

lossi4=,��{       �	#���Yc�A�*

lossOk=���       �	B���Yc�A�*

loss,7�=5Q�C       �	-��Yc�A�*

loss4�/=vHGY       �	�Ȧ�Yc�A�*

loss���<�m�       �	�e��Yc�A�*

loss�O�=W��c       �	!��Yc�A�*

loss#�)=�n�       �	����Yc�A�*

loss�+�<:"�       �	?��Yc�A�*

loss�l�=�҅       �		ߩ�Yc�A�*

loss�>F=�t�p       �	Sz��Yc�A�*

loss�	+=�e|�       �	K:��Yc�A�*

loss]�r<У�k       �	髮Yc�A�*

losso��;;?"�       �	a���Yc�A�*

loss4=�3F4       �	�"��Yc�A�*

lossїu<v���       �	�í�Yc�A�*

loss�=ʑ��       �	�0��Yc�A�*

lossZ�=�k��       �	XƯ�Yc�A�*

lossv҆=�a       �	�`��Yc�A�*

loss ��=�Q:�       �	P���Yc�A�*

lossDl�<�~�       �	����Yc�A�*

loss���=B���       �	�1��Yc�A�*

lossӏ<R8�0       �	�ǲ�Yc�A�*

loss�n=���       �	�^��Yc�A�*

loss�=�ɒ       �	T���Yc�A�*

loss��<c4�s       �	����Yc�A�*

lossɻ>=�2       �	L3��Yc�A�*

loss'B=�;¹       �	����Yc�A�*

lossvz=�G       �	,���Yc�A�*

loss�ؗ=��/�       �	�6��Yc�A�*

loss�"�;���b       �	�ͷ�Yc�A�*

loss�f=m��       �	0f��Yc�A�*

loss@��=�=;       �	����Yc�A�*

lossv	
>F�wC       �	����Yc�A�*

loss,�<���3       �	�)��Yc�A�*

lossٴ=-I�       �	轺�Yc�A�*

loss�h�=���       �	X��Yc�A�*

loss��/=/"�       �	���Yc�A�*

loss=�<S��b       �	T���Yc�A�*

loss��g=�z�&       �	�+��Yc�A�*

loss�=���@       �	�ý�Yc�A�*

lossC2�<,��       �	B]��Yc�A�*

loss�U= ��       �	 ���Yc�A�*

lossd��<8e]�       �	�i��Yc�A�*

loss�(=%���       �	AJ��Yc�A�*

loss/Q�=Ȏ��       �	����Yc�A�*

loss?�<��:{       �	�}®Yc�A�*

loss��=C��       �	�îYc�A�*

lossɟ�=q��       �	��îYc�A�*

loss�\=9��       �	"OĮYc�A�*

lossC��<�飹       �	�ŮYc�A�*

loss W�=p]��       �	��ŮYc�A�*

loss�i�=���       �	apƮYc�A�*

loss��s=��4       �	�ǮYc�A�*

lossj��<��:       �	�ǮYc�A�*

losst�<���       �	t{ȮYc�A�*

lossl=bHӳ       �	�ɮYc�A�*

loss\�<���       �	��ɮYc�A�*

lossOA	=�I�       �	FʮYc�A�*

loss-;=`���       �	��ʮYc�A�*

lossd�A<�<��       �	Q�ˮYc�A�*

loss���<�>�7       �	�̮Yc�A�*

loss���<�f"x       �	i�̮Yc�A�*

loss�{r=��j       �	inͮYc�A�*

lossA�=�7xY       �	uήYc�A�*

lossV)�=����       �	��ήYc�A�*

loss�	�<��$V       �	��ϮYc�A�*

loss] �<�Sy       �	�+ЮYc�A�*

loss���<����       �	��ЮYc�A�*

lossvt<H؋       �	�qѮYc�A�*

loss�Ĝ<w��       �	ӮYc�A�*

loss��=]��w       �	�ӮYc�A�*

loss�Hb=BBaK       �	$}ԮYc�A�*

loss#�>��&�       �	�#ծYc�A�*

lossJ�S=f���       �	��ծYc�A�*

loss{��<{v2       �	�g֮Yc�A�*

loss�a�<����       �	��֮Yc�A�*

lossV�<L���       �	 �׮Yc�A�*

loss= 0=�	�t       �	2خYc�A�*

lossX6�<er�y       �	��خYc�A�*

loss=�F�       �	�cٮYc�A�*

loss�>U�       �	 �ٮYc�A�*

loss���=��J�       �	E�ڮYc�A�*

loss��=L63�       �	!>ۮYc�A�*

loss���=��       �	Z�ۮYc�A�*

loss�=\{�       �	��ܮYc�A�*

loss�Bd=K�֑       �	0LݮYc�A�*

loss���<j��:       �	W�ݮYc�A�*

loss�M<L��X       �	��ޮYc�A�*

loss��=.���       �	��߮Yc�A�*

losshU�==�~C       �	�8�Yc�A�*

loss,�p>���Y       �	���Yc�A�*

lossZD&>�z��       �	p�Yc�A�*

loss>�p/>       �	g�Yc�A�*

loss�Z7=�6       �	ƥ�Yc�A�*

loss�>�<w®�       �	>�Yc�A�*

loss�o>��?       �	���Yc�A�*

loss�m=$�2!       �	��Yc�A�*

loss��n=��HZ       �	@�Yc�A�*

lossJ�}<o�       �	���Yc�A�*

loss|�`=���       �	���Yc�A�*

loss�ث=\Y.       �	Q�Yc�A�*

loss���=:��i       �	��Yc�A�*

loss&��=���B       �	��Yc�A�*

lossd��=�R�o       �	,J�Yc�A�*

loss�	7=�w�       �	���Yc�A�*

loss�<Y�w`       �	$��Yc�A�*

loss�pi=B���       �	7�Yc�A�*

loss�P=�1-       �	��Yc�A�*

loss��=�.>�       �	�o�Yc�A�*

loss���=UO�!       �	���Yc�A�*

loss��=/���       �	����Yc�A�*

loss��[=̀k�       �	 W�Yc�A�*

loss��=�r�4       �	c��Yc�A�*

loss�<Y�Ɔ       �	���Yc�A�*

loss���<S8Z�       �	+�Yc�A�*

losss֖<9�       �	z��Yc�A�*

lossl�=DCۡ       �	�c�Yc�A�*

loss)��<����       �	6�Yc�A�*

loss���=YA       �	5��Yc�A�*

lossWK
=�[At       �	�.�Yc�A�*

lossv@w<h�R�       �	���Yc�A�*

loss��=h�(r       �	�^��Yc�A�*

lossX��=8L��       �	]���Yc�A�*

losss=R�&       �	:���Yc�A�*

lossNծ=�b       �	�%��Yc�A�*

lossI��=�o��       �	���Yc�A�*

lossU�=�Oa�       �	�]��Yc�A�*

loss�6g=c�)g       �	����Yc�A�*

lossv<�<�
gB       �	����Yc�A�*

loss�Ex<���b       �	�6��Yc�A�*

loss��:=��W       �	����Yc�A�*

loss�V'=q���       �	){��Yc�A�*

loss0�=!V��       �	'��Yc�A�*

loss�
#=Jg��       �	����Yc�A�*

lossn�<tw�       �	�k��Yc�A�*

loss�v<{KCc       �	���Yc�A�*

lossi��=%��b       �	���Yc�A�*

lossE=�2��       �	G���Yc�A�*

loss_%E=U|�       �	I/��Yc�A�*

lossr?J<{�R�       �	C���Yc�A�*

loss�>kY�       �	�^ �Yc�A�*

loss��<�7�       �	�� �Yc�A�*

loss[�&>�E�       �	���Yc�A�*

loss{F�<Yֹ]       �	)�Yc�A�*

loss܄"=�r       �	;��Yc�A�*

loss�=�o       �	�X�Yc�A�*

loss�/=��Dw       �	#��Yc�A�*

loss8B�=�w�       �	��Yc�A�*

loss��,>L�(       �	�;�Yc�A�*

loss���=��1       �	-��Yc�A�*

lossb�<K8�       �	֐�Yc�A�*

lossw��< ��       �	O>�Yc�A�*

lossL0>�*�       �	���Yc�A�*

loss�|^=h��E       �	�v�Yc�A�*

lossۊ�<>�       �	bN	�Yc�A�*

loss�-�=g���       �	#�	�Yc�A�*

loss`��=���O       �	ѕ
�Yc�A�*

loss֎�=�;�       �	�4�Yc�A�*

loss >x=�ϴ�       �	���Yc�A�*

loss�<=��       �	�w�Yc�A�*

loss��=�>o       �	"�Yc�A�*

loss[��<Q�       �	0��Yc�A�*

loss��<�m1�       �	.T�Yc�A�*

loss��<D:�       �	3��Yc�A�*

loss�-�<��
�       �	���Yc�A�*

loss�:�=Tb       �	�2�Yc�A�*

loss��<|�=�       �	��Yc�A�*

loss\�<"\YN       �	�m�Yc�A�*

loss�=ح�7       �	��Yc�A�*

loss�R�=����       �	���Yc�A�*

lossq��<�o�&       �	o��Yc�A�*

losst�l=����       �	���Yc�A�*

lossK��=��@�       �	+3�Yc�A�*

loss4�b=��.�       �	���Yc�A�*

loss��c=~��       �	:u�Yc�A�*

loss�a	=z��h       �	.�Yc�A�*

loss�=�E��       �	���Yc�A�*

lossE�=����       �	m�Yc�A�*

loss� �=���       �	��Yc�A�*

loss,>�A��       �	ס�Yc�A�*

loss}%�<����       �	i7�Yc�A�*

loss���=��S�       �	d��Yc�A�*

loss[6=��       �	��Yc�A�*

lossܨ�=�#��       �	� �Yc�A�*

loss�_�<����       �	D��Yc�A�*

losse6�=Eз�       �	�e�Yc�A�*

loss��{=莆       �	��Yc�A�*

lossxa=�]G       �	��Yc�A�*

loss)D�<d��       �	q��Yc�A�*

loss�>�       �	�B �Yc�A�*

loss���=p�       �	� �Yc�A�*

loss�[�<A��       �	am!�Yc�A�*

loss���=�}       �	|"�Yc�A�*

loss0p=9|�       �	ؼ"�Yc�A�*

loss�=_�'�       �	�b#�Yc�A�*

loss��>����       �	�$�Yc�A�*

losss�= ���       �	Ψ$�Yc�A�*

lossd/�<�B�       �	E%�Yc�A�*

loss��=XW*       �	�;&�Yc�A�*

loss��>�}Ԓ       �	>'�Yc�A�*

loss��=��J       �	|�'�Yc�A�*

loss��=+�+C       �	@N(�Yc�A�*

loss�y=��       �	��(�Yc�A�*

loss��>�I�       �	��)�Yc�A�*

loss��H=-O�       �	�H*�Yc�A�*

loss��<V^��       �	��*�Yc�A�*

loss�'=�X       �	�w+�Yc�A�*

lossb� =[�R�       �	�,�Yc�A�*

loss��=G��       �	��,�Yc�A�*

loss C=���*       �	�G-�Yc�A�*

loss	>�=��       �	w�-�Yc�A�*

loss�=��u�       �	){.�Yc�A�*

loss���=�_�       �	/�Yc�A�*

loss���<�[�       �	�/�Yc�A�*

loss�m<�S)S       �	&R0�Yc�A�*

loss\nA<�'�'       �	�0�Yc�A�*

loss}�:=��f       �	7�1�Yc�A�*

loss�ۍ=���       �	�52�Yc�A�*

loss�D>���       �	��2�Yc�A�*

loss_�>/%l�       �	ɓ3�Yc�A�*

loss:׬<IL�       �	s/4�Yc�A�*

loss�Z�=k���       �	�4�Yc�A�*

loss��W=���O       �	[5�Yc�A�*

lossҭ=����       �	,�5�Yc�A�*

loss��R<"z       �	*�6�Yc�A�*

lossAϚ<ȡ��       �	757�Yc�A�*

loss51>Ю	n       �	%�7�Yc�A�*

lossH�j=O�v�       �	�w8�Yc�A�*

loss%e=I�&       �	�9�Yc�A�*

loss���<b]w�       �	�9�Yc�A�*

loss2c%=�x��       �	�J:�Yc�A�*

lossmZ=�D�       �	��:�Yc�A�*

loss�
�=�F2       �	3�;�Yc�A�*

loss��P<�@�       �	�%<�Yc�A�*

loss�+=e^o       �	��<�Yc�A�*

lossD�A=��.       �	�~=�Yc�A�*

loss��<����       �	�>�Yc�A�*

lossu*�=���       �	д>�Yc�A�*

lossʻ�=_TƦ       �	ü?�Yc�A�*

lossط�=�r&       �	�X@�Yc�A�*

loss��=�c��       �	��@�Yc�A�*

lossq�J=���       �	��A�Yc�A�*

loss@e�<b���       �	B�Yc�A�*

lossDZ�=� ��       �		�B�Yc�A�*

loss��|=�G-�       �	�^C�Yc�A�*

losse�c<RP<e       �	�
D�Yc�A�*

loss��=nK��       �	��D�Yc�A�*

lossأ�;G���       �	�{E�Yc�A�*

loss.\=?I�k       �	�F�Yc�A�*

loss��>��c       �	��F�Yc�A�*

loss��>�P�       �	CG�Yc�A�*

loss"E=�mg       �	^�G�Yc�A�*

lossς=��3       �	J}H�Yc�A�*

lossu� >��"�       �	�I�Yc�A�*

loss�]=�%4J       �	��I�Yc�A�*

loss��<ܹ��       �	�QJ�Yc�A�*

loss�8P>À�-       �	? K�Yc�A�*

loss;��<[�H�       �	�K�Yc�A�*

loss��<~X��       �	&7L�Yc�A�*

loss�E�=�
HV       �	��L�Yc�A�*

loss7�3=9S~       �	^�M�Yc�A�*

loss��w=�\ۦ       �	�$N�Yc�A�*

loss\��<�z4H       �	��N�Yc�A�*

loss	i�=�u       �	�nO�Yc�A�*

loss��u=�E��       �	/P�Yc�A�*

loss��=�:#       �	��P�Yc�A�*

loss�l�=�½       �	�XQ�Yc�A�*

loss=�=.B�       �	'�Q�Yc�A�*

lossq|f=@r�       �	m�R�Yc�A�*

loss��a>��p       �	A)S�Yc�A�*

loss�^�<���       �	3�S�Yc�A�*

loss�=�!A        �	�[T�Yc�A�*

loss��7=X��n       �	=�T�Yc�A�*

loss<wu=��@�       �	�U�Yc�A�*

loss�z=r�K       �	=+V�Yc�A�*

loss�y�=?�       �	D�V�Yc�A�*

loss��=c�ؔ       �	�`W�Yc�A�*

loss�Ar<�Z�       �	@�W�Yc�A�*

loss�x�<�GF       �	�X�Yc�A�*

loss1�=���       �	L6Y�Yc�A�*

loss�">ñK       �	��Y�Yc�A�*

loss/�m=�K�       �	_�Z�Yc�A�*

lossj�X=�A�       �	�B[�Yc�A�*

loss�m;=��ȏ       �	��[�Yc�A�*

lossf�=��F�       �	�s\�Yc�A�*

loss�i->��       �	
]�Yc�A�*

loss_C=D~��       �	P�]�Yc�A�*

lossx+%>�`v�       �	WC^�Yc�A�*

loss�9�=W�b       �	��^�Yc�A�*

lossA��=l3B       �	s_�Yc�A�*

lossf=uB*       �	-
`�Yc�A�*

loss���<�ݪ�       �	��`�Yc�A�*

lossjб=>��       �	H5a�Yc�A�*

loss6~�=W���       �	rb�Yc�A�*

lossȩ�<���W       �	u�b�Yc�A�*

loss��=��       �	ILc�Yc�A�*

lossDW�=�B        �	[�c�Yc�A�*

lossXj�=V��       �	;�d�Yc�A�*

loss�G=��9�       �	|)e�Yc�A�*

loss�8�=I�        �	0�e�Yc�A�*

lossT�=�e�       �	+�f�Yc�A�*

loss��>(.}       �	1zg�Yc�A�*

loss}��<��b       �	��h�Yc�A�*

loss�^=�=!       �	�Oi�Yc�A�*

loss�۸=z��       �	�vj�Yc�A�*

loss��X=L!�       �	Rak�Yc�A�*

lossb��=ԗ�       �	�k�Yc�A�*

lossI�&=G�       �	�l�Yc�A�*

loss�?�<�U[       �	q�m�Yc�A�*

loss�D�<�s^N       �	!;n�Yc�A�*

loss�~"=����       �	��n�Yc�A�*

loss	��<��       �	ۋo�Yc�A�*

loss���<Eg�A       �	v4p�Yc�A�*

losst3�<�ϗ/       �	��p�Yc�A�*

lossڟ�=:�       �	�q�Yc�A�*

loss��>U]�&       �	-r�Yc�A�*

loss��a=���g       �	4�r�Yc�A�*

loss ��=a7�\       �	�s�Yc�A�*

lossa�<��4W       �	��t�Yc�A�*

lossi�1=��~       �	�)u�Yc�A�*

lossμ>-�       �	��u�Yc�A�*

loss�|�=1j�       �	�jv�Yc�A�*

loss��=��Rs       �	$w�Yc�A�*

loss��e=�H�7       �	�w�Yc�A�*

loss�7�='���       �	PUx�Yc�A�*

loss
i<Y��!       �	��x�Yc�A�*

loss��W<W���       �	�y�Yc�A�*

loss��=Oߧ�       �	#/z�Yc�A�*

loss�O�=��W       �	3�z�Yc�A�*

lossԒ�=���       �	�[{�Yc�A�*

loss�6�=C%�       �	�|�Yc�A�*

loss!�=@�>|       �	ܻ|�Yc�A�*

loss3J�<CWI       �	�U}�Yc�A�*

lossW\&=�|��       �	��}�Yc�A�*

lossT��<��       �	��~�Yc�A�*

loss��Z<FY6N       �	�4�Yc�A�*

lossg=�<ND�>       �	��Yc�A�*

loss��1=>\'�       �	�a��Yc�A�*

loss{ey=�I�       �	N&��Yc�A�*

lossO<呂?       �	<���Yc�A�*

loss�N�<�벂       �	R��Yc�A�*

loss{b&=֍f)       �	�킯Yc�A�*

loss��=�F�       �	<���Yc�A�*

losso7�<Ō�t       �	C��Yc�A�*

loss��>c��D       �	F���Yc�A�*

loss���=/��       �	�V��Yc�A�*

loss� ^=�N�3       �	I���Yc�A�*

loss���=~ۡ       �	����Yc�A�*

loss���;8�L       �	�!��Yc�A�*

loss-�<�d��       �	쾇�Yc�A�*

loss�Z<=��       �	�f��Yc�A�*

loss\�J=��p�       �	���Yc�A�*

loss"<�f�F       �	˟��Yc�A�*

loss���=m=�%       �	�3��Yc�A�*

loss�4E>_�       �	:͊�Yc�A�*

lossf��=,5�P       �	�e��Yc�A�*

loss?�S=���       �	���Yc�A�*

loss�(�=b@�       �	򘌯Yc�A�*

lossj@(>"P��       �	�v��Yc�A�*

loss�"=�Y�p       �	V��Yc�A�*

loss���=����       �	����Yc�A�*

loss�ڤ=�Hy�       �	�;��Yc�A�*

loss��4<�;       �	ᏯYc�A�*

loss�Z=�       �	Q���Yc�A�*

lossć,=�˖       �	���Yc�A�*

loss��=��.�       �	9���Yc�A�*

loss/l*=+�       �	N��Yc�A�*

loss�-=H��<       �	�撯Yc�A�*

lossx�<r���       �	􉓯Yc�A�*

loss	<�N       �	#��Yc�A�*

loss��;=��M       �	FӔ�Yc�A�*

loss��<��%{       �	vp��Yc�A�*

loss�;=��A�       �	���Yc�A�*

loss�=�AV       �	K���Yc�A�*

loss���=��#       �	�B��Yc�A�*

loss�Y�<��uM       �	"ޗ�Yc�A�*

loss��D=��
       �	:u��Yc�A�*

loss���<��UJ       �	c��Yc�A�*

loss��k=Y�\�       �	<���Yc�A�*

lossW�V<MgCb       �	�;��Yc�A�*

lossA��=�v�       �	�ۚ�Yc�A�*

losst��<*���       �	�t��Yc�A�*

loss� �=�2��       �	��Yc�A�*

lossOx�=��Do       �	ö��Yc�A�*

loss�}=/G�6       �	GT��Yc�A�*

loss7�Z=���       �	�ꝯYc�A�*

lossr�<�D�       �	#���Yc�A�*

loss��=�
�       �	���Yc�A�*

loss@H/=xX��       �	-���Yc�A�*

loss\��=<�       �	�V��Yc�A�*

lossq��=��=�       �	0���Yc�A�*

loss8{p=v�T�       �	���Yc�A�*

loss��<ֹ1w       �	T:��Yc�A�*

loss�Y=R��       �	�+��Yc�A�*

losse�<�1�       �	Fϣ�Yc�A�*

loss�g#=�K9�       �	Ym��Yc�A�*

loss�l�;�F��       �	
��Yc�A�*

loss�d�<��`�       �	���Yc�A�*

lossWs�<i�.       �	Y��Yc�A�*

loss�N�=$EBz       �	�H��Yc�A�*

lossh?p=�
�       �	����Yc�A�*

loss�, >W���       �	ﮨ�Yc�A�*

loss��E>ev�       �	�J��Yc�A�*

lossW*&<a�!�       �	���Yc�A�*

losslU3=����       �	�U��Yc�A�*

lossj=��%�       �	]���Yc�A�*

loss���;]��       �	����Yc�A�*

loss�ƚ;=�`       �	�`��Yc�A�*

losssB�;bW�s       �	���Yc�A�*

loss豜<��p       �	�ⰯYc�A�*

loss�uh<��C       �	����Yc�A�*

loss�K�<喢3       �	�۲�Yc�A�*

lossS�;G�v�       �	�z��Yc�A�*

loss�<�é�       �	�r��Yc�A�*

loss�Z];�CH�       �	���Yc�A�*

loss�:4�~       �	���Yc�A�*

lossM�f;��       �	�Q��Yc�A�*

loss���<�۩�       �	)#��Yc�A�*

loss��=&�z       �	T��Yc�A�*

lossٕ=�l�       �	����Yc�A�*

loss1~;��`       �	x|��Yc�A�*

loss�[�<�q��       �	+i��Yc�A�*

loss��>eD��       �	��Yc�A�*

loss�2"<��o       �	���Yc�A�*

loss1�\>�b&�       �	RH��Yc�A�*

loss���=��       �	'���Yc�A�*

loss��=�
��       �	���Yc�A�*

loss��=�dlw       �	�M��Yc�A�*

lossOt\=�J�t       �	��Yc�A�*

lossE%>�J�       �	����Yc�A�*

loss���=��N       �	R+��Yc�A�*

loss�7H=�v*       �	����Yc�A�*

loss)y=*hC$       �	�v¯Yc�A�*

loss�2=
>-�       �	ZïYc�A�*

loss��=Ͻ��       �	жïYc�A�*

loss+ŝ=�{�       �	�fįYc�A�*

loss��=42Ҵ       �	�ůYc�A�*

lossò�=^�h8       �	�ƯYc�A�*

loss�J�=�9�       �	s�ƯYc�A�*

loss2�D=(-��       �	*RǯYc�A�*

loss��=�0��       �	��ǯYc�A�*

loss��>���~       �	%�ȯYc�A�*

loss��<d�I       �	�(ɯYc�A�*

loss���<ծ�       �	s�ɯYc�A�*

loss�+�=����       �	�ZʯYc�A�*

loss,��<?_	:       �	��ʯYc�A�*

loss@�O<�^��       �	�˯Yc�A�*

lossi1=�b�Z       �	1̯Yc�A�*

lossol<<z�^5       �	w�̯Yc�A�*

lossM4"=N\�       �	\vͯYc�A�*

loss��v<�M��       �	uίYc�A�*

lossE��=g�F�       �	)�ίYc�A�*

loss���=B��       �	.sϯYc�A�*

loss�c�<��P�       �	ЯYc�A�*

lossf��=��H�       �	��ЯYc�A�*

loss,�U=ߤ�@       �	�^ѯYc�A�*

loss��<뮄D       �	|үYc�A�*

loss��<��=~       �	��үYc�A�*

loss�9�<����       �	@iӯYc�A�*

lossm�<j�\�       �	�ԯYc�A�*

loss�Jw=��X       �	H�ԯYc�A�*

lossE��=�K��       �	�wկYc�A�*

loss��=C���       �	�7֯Yc�A�*

loss�<ox��       �	u�֯Yc�A�*

loss��	=�vg3       �	!�ׯYc�A�*

loss<&��       �	�@دYc�A�*

loss��<4ρ�       �	��دYc�A�*

loss�(�<�`��       �	�ٯYc�A�*

loss��<j`��       �	�:گYc�A�*

loss���=%�a;       �	��گYc�A�*

loss��;t�Pw       �	e�ۯYc�A�*

lossDF=�b�c       �	�*ܯYc�A�*

loss��<@3fC       �	}�ܯYc�A�*

loss��<���a       �	�vݯYc�A�*

loss���<���M       �	�T��Yc�A�*

lossڿb=�Q��       �	����Yc�A�*

lossV�k=����       �	����Yc�A�*

loss�?=���       �	�%��Yc�A�*

lossan=�*c�       �	����Yc�A�*

loss�ʎ<���       �	p���Yc�A�*

losszח=���       �	K�Yc�A�*

loss�:=�C)b       �	X��Yc�A�*

loss���=�¹�       �	hB�Yc�A�*

loss��>��]T       �	���Yc�A�*

loss��<Զ,�       �	�j�Yc�A�*

loss�l=���;       �	�0�Yc�A�*

loss�.=�T�       �	�A�Yc�A�*

lossX�=�t�       �	{��Yc�A�*

lossX��<�/4       �	���Yc�A�*

lossW4`=	+"E       �	���Yc�A�*

loss�<c�s�       �	R)�Yc�A�*

lossd`=��E�       �	��Yc�A�*

loss �<4ן�       �	��	�Yc�A�*

lossz/>}��       �	�=
�Yc�A�*

loss.�==�P'<       �	4�
�Yc�A�*

loss� �=uaݷ       �	�n�Yc�A�*

lossj��<��       �	f2�Yc�A�*

lossA��=��       �	�#�Yc�A�*

loss4��<ǧ       �	ǽ�Yc�A�*

loss���<ZX{1       �	&W�Yc�A�*

lossm{�=�Y�       �	���Yc�A�*

lossMsd<���=       �	ɏ�Yc�A�*

lossݕ=`�       �	'�Yc�A�*

loss6w5=���D       �	���Yc�A�*

loss�4=M��       �	�]�Yc�A�*

loss�E�<e�b�       �	���Yc�A�*

loss�1�<��ut       �	H��Yc�A�*

lossA>�{��       �	�N�Yc�A�*

loss�c�<���       �	���Yc�A�*

loss_k=��2&       �	s��Yc�A�*

loss��<���       �	�E�Yc�A�*

loss�ǚ=�DR�       �	�K�Yc�A�*

loss��4>yȹ       �	���Yc�A�*

loss���=�y�l       �	��Yc�A�*

lossn%=� ��       �	�7�Yc�A�*

loss�D�=,��9       �	��Yc�A�*

loss�!�<L䂄       �	��Yc�A�*

loss66z=���       �	�H�Yc�A�*

loss�=;R�       �	W��Yc�A�*

lossa�=��F$       �	Û�Yc�A�*

losst�=�
6T       �	,E�Yc�A�*

loss�|�<B��       �	���Yc�A�*

loss@n=;u��       �	���Yc�A�*

loss��:|��x       �	�r�Yc�A�*

lossS��=�*�       �	. �Yc�A�*

loss?�= R��       �	�� �Yc�A�*

lossO߇<*u�       �	V!�Yc�A�*

loss��!>	��       �	�!�Yc�A�*

loss��E=���K       �	D�"�Yc�A�*

lossqf;����       �	�7#�Yc�A�*

lossW{<�NtC       �	��#�Yc�A�*

loss���;F\�       �	�v$�Yc�A�*

loss@H�=`��       �	�%�Yc�A�*

loss��=?Hm       �	Ҭ%�Yc�A�*

loss��F=m�       �	�F&�Yc�A�*

loss��<=Z�q,       �	W�&�Yc�A�*

loss�&@<-ទ       �	��'�Yc�A�*

losss�B=�x+       �	�4(�Yc�A�*

lossM
1<�g9�       �	_�(�Yc�A�*

loss{	=�M�       �	ͬ)�Yc�A�*

loss���=����       �	aO*�Yc�A�*

loss�p=(a�       �	�*�Yc�A�*

loss��=�~�w       �	r�+�Yc�A�*

loss�#�<���        �	N,�Yc�A�*

loss���<��       �	9�,�Yc�A�*

loss	{�<Y�	       �	��-�Yc�A�*

loss(u�<C/Ԅ       �	�1.�Yc�A�*

loss`u
=��$       �	��.�Yc�A�*

loss��*=_HL�       �	�s/�Yc�A�*

loss��G=,Y/�       �	�0�Yc�A�*

loss��<j�"-       �	�0�Yc�A�*

loss�gH=K��C       �	XY1�Yc�A�*

loss(H=���       �	w�1�Yc�A�*

loss���<�N�       �	.�2�Yc�A�*

loss���=d�k.       �	v53�Yc�A�*

lossϼ=y:{       �	j�3�Yc�A�*

loss��R=m��g       �	|4�Yc�A�*

loss7�+<���#       �	�5�Yc�A�*

loss�؁<���       �	�5�Yc�A�*

loss��<$��       �	�6�Yc�A�*

loss;�`<E4'�       �	*7�Yc�A�*

loss,! =��       �	��7�Yc�A�*

lossa3�<E��       �	�P8�Yc�A�*

lossl>=�'>�       �	��8�Yc�A�*

loss���=��A�       �	�9�Yc�A�*

loss���=WJ'Y       �	�+:�Yc�A�*

loss�:�<���       �	v�:�Yc�A�*

lossMœ=1B7�       �	�Z;�Yc�A�*

lossv��=�A)�       �	��;�Yc�A�*

loss�0=P6�6       �	q�<�Yc�A�*

lossd�='�k       �	�$=�Yc�A�*

loss�6�=�� �       �	ؼ=�Yc�A�*

loss ��=[�;�       �	SY>�Yc�A�*

lossv��<o��       �	O?�Yc�A�*

loss�m�<I[�       �	 �?�Yc�A�*

loss��<��U�       �	�@�Yc�A�*

loss!<�=���       �	@A�Yc�A�*

loss�Ҷ;�_��       �	��A�Yc�A�*

loss��=ؚ�y       �	C�Yc�A�*

loss��]='���       �	|�C�Yc�A�*

lossL��<�HM�       �	�qD�Yc�A�*

lossnwF=�6�r       �	*E�Yc�A�*

loss���=a��"       �	 �E�Yc�A�*

loss�+ =p�#�       �	"nF�Yc�A�*

loss64X=KИ�       �	G�Yc�A�*

lossϖ<J-P�       �	ҨG�Yc�A�*

loss4�=潦x       �	�LH�Yc�A�*

loss���<T���       �	��H�Yc�A�*

loss�f�<�8�@       �	{�I�Yc�A�*

loss%}Y=��N9       �	^J�Yc�A�*

loss�=sxwC       �	� K�Yc�A�*

loss!l�=�M?       �	ȗK�Yc�A�*

lossq?�=j;�       �	j2L�Yc�A�*

loss2)=��K|       �	 �L�Yc�A�*

loss�=�l!+       �	��M�Yc�A�*

loss���<��
       �	{-N�Yc�A�*

loss<�=�~A       �	;�N�Yc�A�*

loss-=�T��       �	FaO�Yc�A�*

loss��><��?       �	�&P�Yc�A�*

lossz!�=��ш       �	��P�Yc�A�*

loss�c=�ٝ       �	[]Q�Yc�A�*

loss6�d=M;��       �	+�Q�Yc�A�*

loss�=���       �	#�R�Yc�A�*

loss���=jB�j       �	�HS�Yc�A�*

loss�^S<+
�1       �	��S�Yc�A�*

loss�K==�)       �	��T�Yc�A�*

loss�Ν<�
�       �	(EU�Yc�A�*

lossɔE=���       �	H�U�Yc�A�*

loss�$H=���Y       �	�xV�Yc�A�*

lossJ6�<��P       �	VW�Yc�A�*

loss�b�<��3�       �	�W�Yc�A�*

loss�l�<.�n       �	�AX�Yc�A�*

loss�=�n1�       �	��X�Yc�A�*

loss;_<�B�       �	�pY�Yc�A�*

loss_=7�w�       �	�
Z�Yc�A�*

loss	~,=�Oe       �	f�Z�Yc�A�*

loss��=G�D�       �	;8[�Yc�A�*

loss�(�<��       �	��[�Yc�A�*

loss=T�=H���       �	Ã\�Yc�A�*

loss�hK<����       �	l%]�Yc�A�*

loss�%�=^7�R       �	T�]�Yc�A�*

loss�Q|<��'_       �	�c^�Yc�A�*

loss��Z=����       �	X_�Yc�A�*

loss�D�<�2�       �	r�_�Yc�A�*

loss}��<`�O�       �	5C`�Yc�A�*

loss��.=����       �	m�`�Yc�A�*

loss]QB=�oP�       �	�a�Yc�A�*

lossd>���       �	�b�Yc�A�*

losst��=�&�       �	ȶb�Yc�A�*

loss��>;L0�       �	�Mc�Yc�A�*

lossl�<[�-       �	��c�Yc�A�*

loss�<|>�q�)       �	Y�d�Yc�A�*

loss4��=\�       �	�!e�Yc�A�*

loss+�=�-lt       �	��e�Yc�A�*

loss_�A=��C       �	ҏf�Yc�A�*

loss�=��b�       �	�*g�Yc�A�*

loss�C�=��w�       �	v�g�Yc�A�*

loss�A�<���       �	�mh�Yc�A�*

loss#�=<z8\       �	�i�Yc�A�*

lossvd=cq9�       �	��i�Yc�A�*

lossq#<T��       �	B@j�Yc�A�*

loss�Y=�^m�       �	!k�Yc�A�*

loss�^=8+W�       �	Y�k�Yc�A�*

loss�R=M���       �	[]l�Yc�A�*

loss�+j=Y1��       �	#�l�Yc�A�*

loss_�<�2s�       �	�m�Yc�A�*

loss��<��       �	q;n�Yc�A�*

loss6�=����       �	��n�Yc�A�*

loss��<bs�I       �	�po�Yc�A�*

loss*�x=I-�f       �	�p�Yc�A�*

loss6<�=��a)       �	m�p�Yc�A�*

loss㿃=��`�       �	DQq�Yc�A�*

loss���<��6�       �	��q�Yc�A�*

lossXc=KPp       �	"�r�Yc�A�*

lossX� =OT)�       �	.�s�Yc�A�*

lossjb�=w2&       �	~qt�Yc�A�*

loss���<�d�/       �	u�Yc�A�*

loss4\�<�b*-       �	��u�Yc�A�*

loss�!�<R�ƺ       �	�jv�Yc�A�*

lossD^=`�&�       �	�w�Yc�A�*

loss]�=��y�       �	��w�Yc�A�*

lossaP<TmX�       �	�Vx�Yc�A�*

loss�|'=5��       �	�y�Yc�A�*

loss���=$�_3       �	I�y�Yc�A�*

loss�8=�1sY       �	VDz�Yc�A�*

loss���<�)�x       �	f�z�Yc�A�*

lossX/<�-	(       �	�{�Yc�A�*

loss���<��g�       �	�>|�Yc�A�*

lossw=9�N�       �	�|�Yc�A�*

loss�S�=��       �	�|}�Yc�A�*

loss!Jd<0.g       �	�~�Yc�A�*

losss_�=��L�       �	İ~�Yc�A�*

loss�-=~Dg�       �	I�Yc�A�*

loss�;�<�)��       �	���Yc�A�*

loss|=y��       �	g���Yc�A�*

loss�x�<MUj       �	�*��Yc�A�*

lossw�	=�\ۦ       �	����Yc�A�*

loss��<ą�       �	�V��Yc�A�*

loss���<m��       �	�&��Yc�A�*

loss�n<�/�?       �	[��Yc�A�*

loss���=�$RV       �	o���Yc�A�*

loss�=k�t�       �	h<��Yc�A�*

loss:F=Y       �	oم�Yc�A�*

lossXA�<ԥ��       �	Lq��Yc�A�*

loss���<k)       �	f��Yc�A�*

loss��=�v+       �	�ć�Yc�A�*

loss�G�;;�Sd       �	����Yc�A�*

loss&�<3�Ԡ       �	I*��Yc�A�*

loss���=���       �	�剰Yc�A�*

loss_!s=��d       �	�,��Yc�A�*

loss\%>>V�a       �	���Yc�A�*

loss�v�=�N��       �	ץ��Yc�A�*

loss�R>��'`       �	�:��Yc�A�*

loss�=}y}       �	�э�Yc�A�*

loss��<�g�       �	�j��Yc�A�*

lossT��<�       �	O��Yc�A�*

lossF%�=�A}       �	6���Yc�A�*

loss��=
��       �	�Q��Yc�A�*

loss��;�v7       �	 Yc�A�*

lossi�=�	��       �	㉑�Yc�A�*

lossN�R=<	\�       �	�$��Yc�A�*

loss?�Z=H�V       �	����Yc�A�*

loss}��< ���       �	�a��Yc�A�*

lossM�L=cɉ�       �	���Yc�A�*

loss�%1<�?ҕ       �	����Yc�A�*

loss?= 8D�       �	�<��Yc�A�*

loss�#?<���       �	֕�Yc�A�*

loss�E<L���       �	�z��Yc�A�*

loss-Ɉ=8�t       �	���Yc�A�*

loss���=��0       �	q藰Yc�A�*

lossWB8=c��       �	ρ��Yc�A�*

loss�=A��       �	���Yc�A�*

loss^V�=0�+x       �	����Yc�A�*

lossFµ<�$��       �	OX��Yc�A�*

loss-q�<��i�       �	����Yc�A�*

loss-Ss=�Kö       �	����Yc�A�*

loss:�N=���       �	�7��Yc�A�*

loss:�=��0w       �	�Yc�A�*

lossQ�}=��y       �	u���Yc�A�*

loss�E=x+��       �	�9��Yc�A�*

loss7=(���       �	Dݞ�Yc�A�*

loss-�3=�^�       �	��Yc�A�*

loss[D+=9��       �	����Yc�A�*

lossJJ�<K2��       �	�O��Yc�A�*

losseWh=��S�       �	{���Yc�A�*

loss��=�_'       �	:���Yc�A�*

loss�A�=���       �	"3��Yc�A�*

lossiK�<��T8       �	�أ�Yc�A�*

loss<7�<�8]       �	�}��Yc�A�*

loss�=��L       �	���Yc�A�*

lossĭ�<O7�8       �	ȶ��Yc�A�*

loss���=�R       �	-\��Yc�A�*

loss̃=�b�       �	(
��Yc�A�*

loss4Dm=>��       �	�O��Yc�A�*

loss.�=#�E�       �	�-��Yc�A�*

loss�<)��       �	aߩ�Yc�A�*

lossV7�=5�8       �	uw��Yc�A�*

loss]=o�"       �	D��Yc�A�*

lossj�<����       �	v���Yc�A�*

lossۧ�=z��Y       �	}>��Yc�A�*

loss��=N70       �	�֬�Yc�A�*

loss��{=�0       �	�l��Yc�A�*

lossa{�=b�i�       �	���Yc�A�*

loss'=8<�V       �	���Yc�A�*

loss��"=�og�       �	�9��Yc�A�*

loss��!=>��y       �	�ܯ�Yc�A�*

loss��+=�N�       �	��Yc�A�*

lossa�=y7�       �	!��Yc�A�*

loss`4�=[�D       �	<���Yc�A�*

loss���=��D       �	B_��Yc�A�*

loss��<D��       �	jl��Yc�A�*

lossi�5<���       �	*��Yc�A�*

loss�V�=���       �	����Yc�A�*

loss��<*��       �	*��Yc�A�*

loss%�&<ܵ�E       �	Cȵ�Yc�A�*

loss���;#
'       �	�\��Yc�A�*

loss�³=ivI�       �	$�Yc�A�*

loss��=혖;       �	㍷�Yc�A�*

loss}�w=<�       �	&��Yc�A�*

lossq�<=�mm       �	�Ÿ�Yc�A�*

loss�E�=�Lq       �	�[��Yc�A�*

loss���<˻̫       �	����Yc�A�*

loss���<)�~�       �	���Yc�A�*

loss�X;x��       �	�1��Yc�A�*

loss-ҋ;?�E&       �	�̻�Yc�A�*

loss��<���       �	#e��Yc�A�*

loss�w=Q�w       �	����Yc�A�*

lossJ��<�tc~       �	ݔ��Yc�A�*

loss��<��       �	NC��Yc�A�*

loss���<z���       �	,׾�Yc�A�*

losst$n=r�v9       �	Ƣ��Yc�A�*

lossMQ�=��       �	�:��Yc�A�*

loss���=�D��       �	���Yc�A�*

loss��m=b�\�       �	j���Yc�A�*

lossl�;�<]�       �	�1ðYc�A�*

loss��C=����       �	��ðYc�A�*

loss	��<�M7       �	V�İYc�A�*

loss?�=��b
       �	iŰYc�A�*

loss ��=���       �	�ŰYc�A�*

loss���=R�{       �	rưYc�A�*

loss}�A<���r       �	�ǰYc�A�*

loss6��=���        �	�ZȰYc�A�*

loss�G"<hB       �	��ȰYc�A�*

losse��=�
v       �	��ɰYc�A�*

lossZ=]l       �	�@ʰYc�A�*

loss#+=D2e       �	��ʰYc�A�*

lossx��=wbi       �	ۇ˰Yc�A�*

loss��;=d�t6       �	� ̰Yc�A�*

loss@a#< ��       �	j�̰Yc�A�*

lossCC�=��g�       �	�ZͰYc�A�*

loss�h�<hԀ�       �	��ͰYc�A�*

loss�?�<K�       �	ߊΰYc�A�*

loss �j=��2�       �	m9ϰYc�A�*

loss�[�<���       �	J�ϰYc�A�*

loss���<8`�       �	�aаYc�A�*

loss��=�}a�       �	k�аYc�A�*

losskz�=_�%       �	H�ѰYc�A�*

loss
e�;��#�       �	) ҰYc�A�*

loss��<�bu"       �	<�ҰYc�A�*

loss]A�=���O       �	uVӰYc�A�*

loss���<6L��       �	��ӰYc�A�*

lossz%$=-az�       �	��԰Yc�A�*

lossݒ�=�f��       �	�,հYc�A�*

losswe�=_߱z       �	��հYc�A�*

loss.&>=�C�       �	�WְYc�A�*

lossҫ�<N~�       �	��ְYc�A�*

loss$��<���^       �	ЀװYc�A�*

loss�w=���9       �	�ذYc�A�*

loss�S�=���       �	J�ذYc�A�*

lossno�=��Z       �	.YٰYc�A�*

loss�g+=Ϭ��       �	V�ٰYc�A�*

lossv�=:�k{       �	>�ڰYc�A�*

losso�8=��       �	o*۰Yc�A�*

loss��;�A)�       �	��۰Yc�A�*

lossR�<+O�       �	��ܰYc�A�*

loss�i=�4f�       �	�BݰYc�A�*

loss�	>"��G       �	(�ݰYc�A�*

loss�=��[�       �	�fްYc�A�*

lossC��=��!�       �	�ްYc�A�*

loss��>��l       �	��߰Yc�A�*

loss��=���]       �	�D�Yc�A�*

lossmQ�=�L9       �	'��Yc�A�*

loss�<Jۖ       �	Lp�Yc�A�*

loss�u�=S��       �	��Yc�A�*

loss�8<�W6       �	[��Yc�A�*

loss�pP=T�z�       �	�/�Yc�A�*

loss� S= @�       �	��Yc�A�*

loss��x=�.r       �	�X�Yc�A�*

loss��=�T
       �	���Yc�A�*

loss���<��z�       �	��Yc�A�*

loss�q<�7>�       �	{�Yc�A�*

lossx�6=*@Ϟ       �	֪�Yc�A�*

loss_Ċ=\0G&       �	1?�Yc�A�*

lossqL.=�k��       �	���Yc�A�*

loss�V%=�?{�       �	���Yc�A�*

loss���<��=�       �	b��Yc�A�*

loss���<��&�       �	���Yc�A�*

lossc��=����       �	�h�Yc�A�*

lossoS�=/4��       �	9
�Yc�A�*

lossX�"=M�'&       �	V��Yc�A�*

loss�͐=_�       �	�O��Yc�A�*

loss�Ŏ<5��U       �	����Yc�A�*

loss�==# �	       �	���Yc�A�*

loss-��<����       �	�)�Yc�A�*

loss_=�s��       �	4��Yc�A�*

loss��<��̣       �	�v�Yc�A�*

loss��<�	�b       �	�Yc�A�*

loss�0�;qUTf       �	��Yc�A�*

loss��=����       �	j��Yc�A�*

loss��%=��*a       �	�N�Yc�A�*

loss[:�=�~       �	���Yc�A�*

loss�|1=B9�       �	S{��Yc�A�*

loss�H�<5�       �	���Yc�A�*

loss7��=��p�       �	J���Yc�A�*

loss���<�a�       �	(I��Yc�A�*

loss��<� �       �	����Yc�A�*

loss�W�=�վ�       �	Jy��Yc�A�*

loss���=�yɜ       �	��Yc�A�*

loss��=\���       �	&���Yc�A�*

loss?�=,��       �	B@��Yc�A�*

loss�=̏�^       �	����Yc�A�*

loss��=��e�       �	����Yc�A�*

loss	��<����       �	_'��Yc�A�*

loss�N><&q��       �	4���Yc�A�*

loss�:�=���       �	]S��Yc�A�*

loss%I�=�<y       �	�(��Yc�A�*

loss鋗<(T�K       �	����Yc�A�*

loss=��<nï�       �	mV��Yc�A�*

losss�{=+)�       �	���Yc�A�*

loss*Z�=���5       �	7���Yc�A�*

loss�rd<��.       �	E* �Yc�A�*

lossc��<5a       �	��Yc�A�*

loss��</o��       �	���Yc�A�*

loss=��=�\]       �	pB�Yc�A�*

lossXHJ<�y��       �	��Yc�A�*

loss���<�V��       �	Dj�Yc�A�*

loss�x= g{�       �	���Yc�A�*

loss���<�˾q       �	��Yc�A�*

lossV={�       �	�?�Yc�A�*

lossP}�=?�       �	���Yc�A�*

loss9D�=�3�       �	�g�Yc�A�*

loss;] =*K��       �	���Yc�A�*

lossX%�=sd�m       �	O��Yc�A�*

loss�p6=C�	�       �	fO�Yc�A�*

loss!�U=fC�t       �	���Yc�A�*

loss�l=����       �	�	�Yc�A�*

lossE$�<�J       �	3
�Yc�A�*

losst��=p.&n       �	��
�Yc�A�*

lossTY\=��vh       �	���Yc�A�*

loss�_�=�j        �	�F�Yc�A�*

loss�g(<Y��4       �	�E�Yc�A�*

loss��<<�݃       �	���Yc�A�*

loss��=��nl       �	�~�Yc�A�*

loss#>L�C�       �	sg�Yc�A�*

lossהU=G˗�       �	]��Yc�A�*

loss61�=g��       �	���Yc�A�*

loss��L<�tg�       �	8/�Yc�A�*

loss�2>��        �	��Yc�A�*

loss��==����       �	�[�Yc�A�*

loss�H=�5�       �	}�Yc�A�*

loss/w=���       �	3��Yc�A�*

losst�[=u��       �	�:�Yc�A�*

loss"�=�]�-       �	J��Yc�A�*

loss
v�<�^�       �	��Yc�A�*

loss�(=0�p       �	�"�Yc�A�*

loss��1=3I.Z       �	���Yc�A�*

loss�UL<���       �	)z�Yc�A�*

loss�;?=(ǯJ       �	��Yc�A�*

loss�s=��K�       �	���Yc�A�*

loss��<���       �	_`�Yc�A�*

loss.�=f)�       �	��Yc�A�*

lossR3�;�۾        �	֭�Yc�A�*

loss��H<�x       �	�S�Yc�A�*

lossi�<�%�T       �	���Yc�A�*

loss��r=�LN�       �	���Yc�A�*

loss ��=� �       �	��Yc�A�*

lossL��<kke�       �	p��Yc�A�*

loss�=U=ڛ_       �	�\�Yc�A�*

lossM`<<��/�       �	���Yc�A�*

loss)�<�,5�       �	k��Yc�A�*

loss��1>N�}&       �	�2 �Yc�A�*

loss)"�=j��8       �	�� �Yc�A�*

loss $�<���E       �	bh!�Yc�A�*

loss�m=l� d       �	��!�Yc�A�*

lossE�=���       �	Ĕ"�Yc�A�*

loss�;eQ-3       �	,#�Yc�A�*

lossNc�<�ba�       �	��#�Yc�A�*

lossF��=.�jn       �	�T$�Yc�A�*

loss.*=>�!S       �	��$�Yc�A�*

loss廴=m|�6       �	Z�%�Yc�A�*

loss��i=��6       �	X&�Yc�A�*

loss�A�<�Q��       �	�&�Yc�A�*

loss�sc='M�       �	8I'�Yc�A�*

loss��m=�2(�       �	K(�Yc�A�*

lossn�;��"B       �	1)�Yc�A�*

lossp��<��       �	�)�Yc�A�*

loss`U�<��       �	wK*�Yc�A�*

loss��=M�       �	�F+�Yc�A�*

lossfU�<����       �	�0,�Yc�A�*

losss� <n��y       �	I�,�Yc�A�*

loss��7=�dK�       �	�-�Yc�A�*

loss�t6=𕾓       �	�;.�Yc�A�*

loss���=�!`�       �	��.�Yc�A�*

loss�Z(<���       �	�/�Yc�A�*

loss{=.>��A�       �	F�0�Yc�A�*

loss=|<�V��       �	c{1�Yc�A�*

loss�is=]��       �	�"2�Yc�A�*

loss��y=!�{�       �	Y�2�Yc�A�*

loss�ݽ<���^       �	%�3�Yc�A�*

lossX�<J�)       �	�4�Yc�A�*

lossr�;"�;       �	wL5�Yc�A�*

loss���=�T       �	��5�Yc�A�*

lossτA=\W��       �	L�6�Yc�A�*

loss���=jz�       �	�7�Yc�A�*

loss�(�=����       �	�T8�Yc�A�*

loss�b=��F       �	*9�Yc�A�*

losst^2=����       �	�2:�Yc�A�*

loss0=��ݗ       �	�;�Yc�A�*

loss@�=�&`�       �	�;�Yc�A�*

loss,��<1�*`       �	��<�Yc�A�*

lossq��<���U       �	�e=�Yc�A�*

lossMφ=DZ�o       �	zU>�Yc�A�*

lossڲ�;i-�       �	�Z?�Yc�A�*

loss�=z�       �	+@�Yc�A�*

lossCb=s%�m       �	|�@�Yc�A�*

loss䵃=k���       �	�\A�Yc�A�*

loss���;˫��       �	��A�Yc�A�*

loss���=�S�       �	_�B�Yc�A�*

loss?o=:�	�       �	f0C�Yc�A�*

lossi� =���       �	��C�Yc�A�*

loss[�A=N��       �	�\D�Yc�A�*

loss�R�<�J�z       �	��D�Yc�A�*

loss ��<^��       �	H�E�Yc�A�*

loss	J�=�.ٚ       �	KF�Yc�A�*

lossu�=�">R       �	��F�Yc�A�*

loss�Po<���       �	�NG�Yc�A�*

loss��&=�5��       �	��G�Yc�A�*

loss��M<q��       �	�uH�Yc�A�*

loss�Q�=K[��       �	�&I�Yc�A�*

loss
�r=� ��       �	��I�Yc�A�*

loss���< $��       �	1\J�Yc�A�*

loss1-�<����       �	��J�Yc�A�*

loss3��=_�       �	��K�Yc�A�*

loss��=��2o       �	�OL�Yc�A�*

loss���<+#�       �	��L�Yc�A�*

loss}@=��~       �	��M�Yc�A�*

lossM=����       �	�FN�Yc�A�*

loss�i:=�Oz}       �	��N�Yc�A�*

losso��<�d�       �	�O�Yc�A�*

loss��=��X       �	�$P�Yc�A�*

loss{7�=(%��       �	��P�Yc�A�*

loss���<C��       �	�fQ�Yc�A�*

loss���<��hw       �	��Q�Yc�A�*

loss��6=x:��       �	o�R�Yc�A�*

loss#�<N��6       �	�SS�Yc�A�*

lossޛ�=^�Z       �	g�S�Yc�A�*

loss0/�<���       �	n�T�Yc�A�*

loss7��<AU�E       �	CU�Yc�A�*

loss���<����       �	c�U�Yc�A�*

lossl�(=�=       �	�JV�Yc�A�*

loss���<8i��       �	��V�Yc�A�*

losssl=P��       �	��W�Yc�A�*

loss�ˈ=�{ I       �	�!X�Yc�A�*

loss|��;)���       �	��X�Yc�A�*

loss}�=�
��       �	�UY�Yc�A�*

loss�)s=/4��       �	<�Y�Yc�A�*

loss&D;��       �	_�Z�Yc�A�*

loss��:K�<       �	a4[�Yc�A�*

loss3�*<{��t       �	|�[�Yc�A�*

lossW��<���x       �	�q\�Yc�A�*

loss���<O���       �	�]�Yc�A�*

lossvl�<�       �	؞]�Yc�A�*

loss��<gA��       �	�4^�Yc�A�*

lossl �;�Z��       �	��^�Yc�A�*

loss�[�;&`��       �	0�_�Yc�A�*

loss���:�aF�       �	T8`�Yc�A�*

lossä:Q�h�       �	��`�Yc�A�*

loss���<���       �	�a�Yc�A�*

loss�C�=r�_:       �	�,b�Yc�A�*

loss��<���[       �	C�b�Yc�A�*

loss�y#;k��Z       �	0bc�Yc�A�*

loss�=��       �	��d�Yc�A�*

loss�D�=B,�       �	Ke�Yc�A�*

loss!��;�6v       �	9�e�Yc�A�*

loss�=�=�|/-       �	�Mf�Yc�A�*

lossj��=��       �	��f�Yc�A�*

loss�==R3�-       �	��g�Yc�A�*

lossH՟<V�vZ       �	�Gh�Yc�A�*

loss�l�<��/u       �	�1i�Yc�A�*

loss�~�=��E       �	�i�Yc�A�*

loss��=_�ک       �	�j�Yc�A�*

loss!<r�I�       �	��k�Yc�A�*

loss#�&=��       �	��l�Yc�A�*

lossz�Z=4��       �	Jm�Yc�A�*

loss
N#>Ę��       �	��m�Yc�A�*

lossZ��=����       �	��n�Yc�A�*

loss�]g=��U�       �	�*o�Yc�A�*

loss�n=��U�       �	S�o�Yc�A�*

loss��=�r�1       �	ɓp�Yc�A�*

loss�u$=��*       �	�5q�Yc�A�*

loss� �=��       �	��q�Yc�A�*

loss8��=���       �	;rr�Yc�A�*

lossHթ<t�H}       �	9
s�Yc�A�*

loss9C
=@��       �	�s�Yc�A�*

loss�QZ=��(       �	mt�Yc�A�*

loss�jm<�2^       �	su�Yc�A�*

lossX<���       �	��u�Yc�A�*

lossNj�<C��       �	�Rv�Yc�A�*

lossz�;�u|       �	�v�Yc�A�*

loss���=��E�       �	)�w�Yc�A�*

loss�<GN       �	�Ux�Yc�A�*

losszS�=b��       �	��x�Yc�A�*

loss=�=J��       �	��y�Yc�A�*

loss�;��P       �	�+z�Yc�A�*

loss|P�=�f8       �	7�z�Yc�A�*

loss���<�(��       �	�^{�Yc�A�*

lossl^;kH�       �	/|�Yc�A�*

lossM�=�5�+       �	:�|�Yc�A�*

loss!Қ<�X�f       �	$c}�Yc�A�*

loss!�=��r       �	��}�Yc�A�*

loss�X=�Ԭ       �	)�~�Yc�A�*

lossJ�=K��"       �	Z-�Yc�A�*

loss��k=w;       �	���Yc�A�*

loss�d<f�Oy       �	_���Yc�A�*

loss[�2=�� e       �	,��Yc�A�*

loss���<^�o       �	-Ё�Yc�A�*

lossm4=��t       �	�d��Yc�A�*

loss��<?��M       �	����Yc�A�*

loss=��=W
*       �	f���Yc�A�*

loss�<�=�g�       �	.9��Yc�A�*

lossN5<��       �	�܄�Yc�A�*

loss<:_=Hn�       �	�r��Yc�A�*

loss*˟<ή�       �	���Yc�A�*

losscX<���.       �	྆�Yc�A�*

loss���<�jϲ       �	r3��Yc�A�*

loss�mM=A�"2       �	WΠ�Yc�A�*

losstc�=х��       �	zo��Yc�A�*

lossm�<t	j       �	[��Yc�A�*

loss�i=��       �	����Yc�A�*

loss�=���       �	�:��Yc�A�*

loss���<��(E       �	\裱Yc�A�*

loss_٧<<�8�       �	^���Yc�A�*

lossK�=���M       �	���Yc�A�*

loss���=���       �	̳��Yc�A�*

loss�ǂ<ș�d       �	����Yc�A�*

loss�F�<���       �	n3��Yc�A�*

loss׎�=Sɵ       �	lͧ�Yc�A�*

loss[�=�r�       �	{f��Yc�A�*

loss�'�<��&       �	G��Yc�A�*

loss�t=���o       �	���Yc�A�*

loss,�<��JA       �	%t��Yc�A�*

loss���<��?       �	�S��Yc�A�*

loss&N=2�E�       �	�쫱Yc�A�*

lossS �=���       �	.㬱Yc�A�*

loss��;=�tu�       �	����Yc�A�*

loss�Ŧ=��n�       �	�;��Yc�A�*

lossz��<���       �	 $��Yc�A�*

loss?�=� ˈ       �	�ٯ�Yc�A�*

loss#M�<]���       �	Y���Yc�A�*

loss�0h=�Hv       �	�]��Yc�A�*

lossdt�<��=       �	�K��Yc�A�*

loss�=S���       �	����Yc�A�*

loss�z�<U�p       �	Ƌ��Yc�A�*

loss��=�f�E       �	6Y��Yc�A�*

loss:�<ػ       �	໵�Yc�A�*

loss�3=#�P       �	b���Yc�A�*

loss8O�<��"�       �	ZE��Yc�A�*

loss���=9�       �	/��Yc�A�*

loss2�F<;�u-       �	�Yc�A�*

lossvĴ<a��       �	r���Yc�A�*

loss\P�<�h��       �	�ú�Yc�A�*

loss��&=���X       �	����Yc�A�*

loss/]�=��       �	�Yc�A�*

loss!(�=�;@       �	�
��Yc�A�*

loss1��<�       �	!���Yc�A�*

loss��=�'cj       �	�X��Yc�A�*

loss��C<{y�6       �	���Yc�A�*

lossMv�<�<n�       �	����Yc�A�*

lossn��=        �	�W��Yc�A�*

lossR�>�3       �	Q±Yc�A�*

loss���<�X^       �	(�±Yc�A�*

loss�q,=?�Q�       �	��ñYc�A�*

loss�,a<�f�;       �	0,ıYc�A�*

lossW+�:��       �	��ıYc�A�*

lossʈQ<�)�1       �	]űYc�A�*

loss��;=�P�       �	GƱYc�A�*

loss�Q�<7�z�       �	` ǱYc�A�*

loss2��=�1�       �	�ǱYc�A�*

loss�Ϯ<��7�       �	�8ȱYc�A�*

loss)Fb;JX�k       �	��ȱYc�A�*

loss��;��       �	c{ɱYc�A�*

loss�z�:D�       �	�ʱYc�A�*

loss�i�<8&�       �	��ʱYc�A�*

loss�S"=kua       �	_a˱Yc�A�*

lossZ��=uՍ�       �	u̱Yc�A�*

loss�v�;�=+�       �	�̱Yc�A�*

loss�7S<rf�F       �	JEͱYc�A�*

loss�by=}#��       �	�ͱYc�A�*

loss�I�;�:<       �	�}αYc�A�*

lossAT�;���       �	�ϱYc�A�*

loss��=W�3       �	�ϱYc�A�*

loss6�<⋒       �	'JбYc�A�*

loss�#8=yhN�       �	��бYc�A�*

loss�>i���       �	�zѱYc�A�*

losso5=���       �	�ұYc�A�*

losshv�<C!G       �	��ұYc�A�*

lossi�=9��       �	�PӱYc�A�*

loss��<J�V4       �	��ӱYc�A�*

lossj�=4G�R       �	�ԱYc�A�*

lossD��<����       �	nձYc�A�*

loss�rx<��!s       �	��ձYc�A�*

loss�� =�K       �	YֱYc�A�*

loss�e;=7�9.       �	�ֱYc�A�*

lossMǏ=u�52       �	�ױYc�A�*

lossDQ�<w'�6       �	0FرYc�A�*

loss��=���        �	��رYc�A�*

loss~�=��P       �	T�ٱYc�A�*

lossn�<<� �       �	@2ڱYc�A�*

loss��=�0�r       �	�ڱYc�A�*

lossa��<�4r�       �	�۱Yc�A�*

lossJL=���       �	�KܱYc�A�*

loss�P=�c�       �	W�ܱYc�A�*

lossfS=��J�       �	�ݱYc�A�*

loss`�<�v�       �	�ޱYc�A�*

loss��>�s|       �	�ޱYc�A�*

loss���=���       �	S^߱Yc�A�*

lossҟG=�A@       �	�<�Yc�A�*

lossTA�<l"l       �	��Yc�A�*

loss�K�=���       �	��Yc�A�*

loss��g=ѣ
#       �	3�Yc�A�*

loss�o==Pσ       �	p��Yc�A�*

loss;o=�h1�       �	�Q�Yc�A�*

loss���=��       �	Y��Yc�A�*

loss�cJ=���<       �	���Yc�A�*

loss�Ɇ;G��r       �	.7�Yc�A�*

loss\�w;mL��       �	l��Yc�A�*

loss =M�L�       �	5�Yc�A�*

loss�></K.       �	��Yc�A�*

loss�9z=�xQ       �	~��Yc�A�*

loss1^N=��       �	�C�Yc�A�*

loss6v�<����       �	'��Yc�A�*

loss�M,<�dT�       �	q�Yc�A�*

loss6 �=$8�[       �	W�Yc�A�*

loss[J�<��       �	�5�Yc�A�*

loss�c�=sA�S       �	T�Yc�A�*

loss��4=O�K       �	��Yc�A�*

loss�Z[=���&       �	w�Yc�A�*

loss�v�;+=�       �	��Yc�A�*

loss&��<lx�       �	���Yc�A�*

lossф<�l_C       �	G�Yc�A�*

loss�$�=h���       �	X��Yc�A�*

lossm�s=�%Zt       �	Y5�Yc�A�*

loss��6=a��r       �		��Yc�A�*

loss�"�<���       �	ڮ�Yc�A�*

loss��<���       �	�a��Yc�A�*

lossԻ�<��7�       �	i��Yc�A�*

loss[��= 5�
       �	��Yc�A�*

loss�5= 6?Y       �	����Yc�A�*

loss;�;h��       �	����Yc�A�*

lossS�^=!ɓ�       �	U���Yc�A�*

loss16�<"»>       �	nO��Yc�A�*

loss�*=F���       �	\��Yc�A�*

loss͎=&o&�       �	ǡ��Yc�A�*

lossf�=J�       �	�A��Yc�A�*

losszծ<���       �		���Yc�A�*

loss#o=����       �	T���Yc�A�*

lossD<o9�       �	�y��Yc�A�*

loss��<�uF       �	.��Yc�A�*

lossw�r=�O       �	���Yc�A�*

lossR�<2P�       �	�Z��Yc�A�*

lossO=4�h�       �	H���Yc�A�*

loss�c�<�	��       �	� �Yc�A�*

loss�@=/�M       �	 q�Yc�A�*

loss�7�;FR�       �	��Yc�A�*

lossn+�<3�H�       �	���Yc�A�*

lossTA*=>z}       �	�Yc�A�*

lossf�==��ٌ       �	��Yc�A�*

loss�Hp=���       �	7p�Yc�A�*

lossti>JnI       �	��Yc�A�*

loss���<�T�       �	��Yc�A�*

loss�=����       �	M�Yc�A�*

loss�3�;�c�       �	�Yc�A�*

loss��`<���       �	U��Yc�A�*

loss�kl=�p�       �	S<	�Yc�A�*

loss$��<i(       �	+�	�Yc�A�*

loss�ҡ<^Is       �	�u
�Yc�A�*

loss��=�2       �	��Yc�A�*

loss7Fe<�m-       �	a��Yc�A�*

loss�=��M       �	��Yc�A�*

loss��<�Δ~       �	�-�Yc�A�*

loss�wZ=	ꡟ       �	r�Yc�A�*

loss���=Џo�       �	<��Yc�A�*

loss��=�Y�	       �	���Yc�A�*

loss��<����       �	s+�Yc�A�*

loss� �=<�	       �	"��Yc�A�*

loss}�!=�^�       �	�a�Yc�A�*

loss��=A� �       �	��Yc�A�*

lossiG�;��N�       �	D��Yc�A�*

lossw��<q��       �	�|�Yc�A�*

loss�O1=n��       �	C�Yc�A�*

loss��<�4��       �	)?�Yc�A�*

lossy�<�{e       �	e��Yc�A�*

lossh<m�       �	��Yc�A�*

loss�<��$       �	��Yc�A�*

lossw�X="q       �	̵�Yc�A�*

loss<�R;� �       �	�f�Yc�A�*

loss�-�<���M       �	X�Yc�A�*

loss�ؗ=Xd,m       �	ۣ�Yc�A�*

loss��<i<0�       �	�B�Yc�A�*

loss(�=�$0       �	���Yc�A�*

loss=�=-�ag       �	���Yc�A�*

loss�{�=~L��       �	<3�Yc�A�*

lossd�o<���       �	`��Yc�A�*

loss#��<����       �	>v�Yc�A�*

loss�&�<w_N       �	y�Yc�A�*

loss@4=��g�       �	���Yc�A�*

loss)<����       �	��Yc�A�*

lossŜ<2+�a       �	�# �Yc�A�*

loss��<ϓoy       �	�� �Yc�A�*

loss��==AK       �	�Z!�Yc�A�*

loss�jv=%�       �	��!�Yc�A�*

loss�^�=�I�%       �	��"�Yc�A�*

loss��<,��       �	�`#�Yc�A�*

loss�Hs=��N       �	?$�Yc�A�*

loss��=C���       �	��$�Yc�A�*

loss�J	=�׿b       �	�E%�Yc�A�*

loss��8=���       �	M.&�Yc�A�*

lossD�q<$^;l       �	F�&�Yc�A�*

loss�;;Y&�       �	�r'�Yc�A�*

lossC�==���r       �	�'(�Yc�A�*

loss{ԙ<���       �	0�(�Yc�A�*

loss�&=����       �	�z)�Yc�A�*

loss�߽=�_zc       �	�I*�Yc�A�*

loss�<���       �	z�*�Yc�A�*

loss�n=���K       �	��+�Yc�A�*

loss��<]��R       �	�/,�Yc�A�*

loss�K�;O��T       �		-�Yc�A�*

lossm+H=Z,       �	ɮ-�Yc�A�*

loss��<�M�       �	nN.�Yc�A�*

loss21=m��       �	��.�Yc�A�*

loss��=w,��       �	/�Yc�A�*

loss��$=� R�       �	�/0�Yc�A�*

loss��=ы�C       �	��0�Yc�A�*

loss��<h�&       �	2�Yc�A�*

loss��K=�       �	4,3�Yc�A�*

lossCΦ=C&b�       �	b�3�Yc�A�*

loss�O�;�#O       �	�}4�Yc�A�*

lossZ��;�3��       �	�5�Yc�A�*

lossѕ�=�W       �	��5�Yc�A�*

losso�=�t�*       �	ԁ6�Yc�A�*

loss��>G2 �       �	�7�Yc�A�*

loss�+>�΢<       �	g�7�Yc�A�*

loss��=C	D       �	�h8�Yc�A�*

loss.�_=z�&�       �	�9�Yc�A�*

lossfs�<v�a�       �	J�9�Yc�A�*

loss>ĉ<���Y       �	�_:�Yc�A�*

lossG
='O��       �	7;�Yc�A�*

lossL��=|i��       �	��;�Yc�A�*

loss�1�;��       �	�`<�Yc�A�*

loss<ڈ=L{c<       �	�=�Yc�A�*

loss?�B=Gs��       �	��=�Yc�A�*

loss0=e9K�       �	cE>�Yc�A�*

loss7(<7��       �	�>�Yc�A�*

loss]TD=���       �	]�?�Yc�A�*

loss���<e�~�       �	E/@�Yc�A�*

loss��c<�xV�       �	��@�Yc�A�*

loss<�=��u9       �	]jA�Yc�A�*

loss�6�<���3       �	�B�Yc�A�*

loss`�=l~��       �	�B�Yc�A�*

loss�@�<��       �	b�C�Yc�A�*

loss8M=�jϿ       �	�#D�Yc�A�*

lossㄼ=i��       �	o�D�Yc�A�*

loss�/�=_���       �	D�E�Yc�A�*

loss���<l}3"       �	g+F�Yc�A�*

lossq�/=�!r       �	��F�Yc�A�*

loss|�'=��Zh       �	;mG�Yc�A�*

loss��{=��       �	�H�Yc�A�*

loss �<7](       �	��H�Yc�A�*

loss	7<C�qS       �	P9I�Yc�A�*

losscڿ<���f       �	`�I�Yc�A�*

lossт�<���       �	��J�Yc�A�*

loss�<�-�       �	K�Yc�A�*

loss���<z��       �	�K�Yc�A�*

loss��<$��       �	(~L�Yc�A�*

loss��=�#i�       �	�M�Yc�A�*

loss��<_�'�       �	�M�Yc�A�*

loss�w=���       �	*WN�Yc�A�*

loss�?;�7       �	A�N�Yc�A�*

loss���<�28x       �	ёO�Yc�A�*

loss�� <c�y       �	
.P�Yc�A�*

losso��<�K�       �	2�P�Yc�A�*

lossE[}<O��       �	��Q�Yc�A�*

lossZ�4=Th�U       �	��R�Yc�A�*

loss��=R�l.       �	�ES�Yc�A�*

loss�>X<��UI       �	}�S�Yc�A�*

loss6Nb<E�Z�       �	-�T�Yc�A�*

loss(�=��+H       �	�7U�Yc�A�*

loss�@=����       �	 V�Yc�A�*

lossmĕ=�X��       �	N�V�Yc�A�*

loss���<�OW�       �	PSW�Yc�A�*

lossaCU=:g�=       �	��W�Yc�A�*

loss@�<����       �	��X�Yc�A�*

loss&�7=��4�       �	�-Y�Yc�A�*

loss@6�=v���       �	�Y�Yc�A�*

loss�@�=s�       �	�iZ�Yc�A�*

lossזq=�       �	,[�Yc�A�*

loss�%�=x���       �	�[�Yc�A�*

loss��="��       �	d[\�Yc�A�*

loss	�=z�       �	f�\�Yc�A�*

loss��=O1E"       �	��]�Yc�A�*

lossal=�X1       �	�2^�Yc�A�*

loss�N�;��<�       �	.�^�Yc�A�*

loss�#=��       �	�i_�Yc�A�*

loss�4<�W?�       �	�`�Yc�A�*

loss���;`��       �	��`�Yc�A�*

lossHS7<"��#       �	�Qa�Yc�A�*

loss&��<BZl       �	��a�Yc�A�*

loss=@�<�"b       �	��b�Yc�A�*

loss�Z�=��d       �	�@c�Yc�A�*

loss��8=��s       �	j�c�Yc�A�*

loss
u�=#���       �	[|d�Yc�A�*

loss��=3?A       �	2e�Yc�A�*

loss�rn=�CL�       �	l�e�Yc�A�*

loss)��;���       �	��f�Yc�A�*

loss2�<�2�       �	wKg�Yc�A�*

loss$�=	J|�       �	 �g�Yc�A�*

lossRxd;��@       �	G�h�Yc�A�*

loss,�;[C�9       �	�Li�Yc�A�*

loss3��<��H�       �	��i�Yc�A�*

lossz�>�w�       �	��j�Yc�A�*

lossZ�<"��C       �	�bk�Yc�A�*

loss 5�<k]�       �	��k�Yc�A�*

lossl�>���       �	��l�Yc�A�*

loss�ϑ=��<       �	�9m�Yc�A�*

loss��='۸K       �	�m�Yc�A�*

loss3Σ=1r�       �	�nn�Yc�A�*

loss�	=����       �	�o�Yc�A�*

loss���=�h��       �	I�o�Yc�A�*

loss7�=����       �	7p�Yc�A�*

loss�^;> ���       �	��p�Yc�A�*

loss��;2       �	%xq�Yc�A�*

loss=�=�<�       �	�=r�Yc�A�*

loss�e&<�       �	��r�Yc�A�*

loss$�u=����       �	H�s�Yc�A�*

lossS4 =c��       �	�>t�Yc�A�*

loss��8=��/       �	��t�Yc�A�*

loss5�=����       �	��u�Yc�A�*

loss/��;��V       �	Mv�Yc�A�*

loss�%�<�RE       �	��v�Yc�A�*

loss��=D��       �	ҋw�Yc�A�*

losss��<:N�       �	_&x�Yc�A�*

lossH3=���       �	��x�Yc�A�*

lossI��=�9m       �	�[y�Yc�A�*

loss��0=��_P       �	��y�Yc�A�*

lossD܊<F�q*       �	��z�Yc�A�*

loss(e
>o��       �	�,{�Yc�A�*

loss1�=��%V       �	%�{�Yc�A�*

lossIG�<"��'       �	��|�Yc�A�*

loss��=ͮ��       �	�'}�Yc�A�*

loss8Z�=G��l       �	T�}�Yc�A�*

loss%0=��z�       �	�m~�Yc�A�*

loss���<�7M       �	�
�Yc�A�*

lossۻ�=�U^�       �	���Yc�A�*

loss�A=�Ϡ       �	�9��Yc�A�*

loss�8=Y��       �	|Ӏ�Yc�A�*

lossڰ=Xd�       �	圁�Yc�A�*

loss܇�<3��       �	T;��Yc�A�*

loss�ѻ<JXD       �	ق�Yc�A�*

loss�>0T�       �	ӄ��Yc�A�*

loss��=���       �	*��Yc�A�*

loss�{=���       �	!˄�Yc�A�*

loss���<�
��       �	F��Yc�A�*

loss_�	=#��6       �	�ކ�Yc�A�*

lossM�;�s�       �	�v��Yc�A�*

loss�k<���3       �	f/��Yc�A�*

loss�H�<4+       �	�ۈ�Yc�A�*

loss��=���       �	w��Yc�A�*

loss��=��>>       �	��Yc�A�*

loss�s�=�Bx�       �	2���Yc�A�*

loss~�>`k�       �	�`��Yc�A�*

loss�UA=$ ^�       �	���Yc�A�*

loss��=(�       �	�Yc�A�*

loss;JP=���       �	�1��Yc�A�*

loss7	=*�n$       �	~Ǎ�Yc�A�*

loss�w9=3W�^       �	�i��Yc�A�*

loss��<T*4�       �	  ��Yc�A�*

loss )Z=��Qe       �	 ���Yc�A�*

loss�Ҷ<�{       �	w.��Yc�A�*

lossr]=�Jn       �	����Yc�A�*

loss�"�<'��i       �	[���Yc�A�*

lossD<�k       �	K:��Yc�A�*

lossf�<����       �	�Ғ�Yc�A�*

loss���<ҽ       �	{��Yc�A�*

loss|��<X��#       �	+��Yc�A�*

losssb=�	N�       �	����Yc�A�*

loss�z�<�H��       �	�F��Yc�A�*

lossOX=&��d       �	���Yc�A�*

lossjH�=�^m�       �	�Ö�Yc�A�*

loss��=C<��       �	\��Yc�A�*

loss��\=v��       �	H���Yc�A�*

lossD��=�c{5       �	����Yc�A�*

loss�H<��]       �	@��Yc�A�*

loss�0=�^�o       �	Yߙ�Yc�A�*

loss��=�M)t       �	[z��Yc�A�*

lossl,�=����       �	�8��Yc�A�*

loss��1=B��^       �	ܛ�Yc�A�*

lossư�;�^0�       �	S{��Yc�A�*

loss�>�<)$��       �	���Yc�A�*

loss��R=��7       �	󯝲Yc�A�*

lossD�=�z��       �	�S��Yc�A�*

loss���=u��       �	&���Yc�A�*

loss�=�-w�       �	裟�Yc�A�*

loss�(�<��`       �	.���Yc�A�*

loss� �<�t�       �	ZJ��Yc�A�*

losss��=�g�        �	�졲Yc�A�*

lossq~�=����       �	���Yc�A�*

loss���=Y*:�       �	�=��Yc�A�*

loss9_�<�2�       �	�ݣ�Yc�A�*

loss��<�6V       �	W|��Yc�A�*

lossE+�=�-�       �	�"��Yc�A�*

loss,2N=d��4       �	�ĥ�Yc�A�*

loss�"�=~DHF       �	�h��Yc�A�*

loss��;a�Y*       �	���Yc�A�*

loss�;�3 �       �	񼧲Yc�A�*

loss!�><�k��       �	�_��Yc�A�*

lossf�=��       �	* ��Yc�A�*

loss�!=6M�p       �	3���Yc�A�*

loss�v�<e� �       �	\Y��Yc�A�*

lossݏ=*�Z       �	�l��Yc�A�*

loss�~�=�/�       �	���Yc�A�*

loss=��=���*       �	衬�Yc�A�*

lossd�V<6�;9       �	�B��Yc�A�*

loss�i�<��@�       �	@ݭ�Yc�A�*

loss��g<w:�{       �	�x��Yc�A�*

loss��Y<�i�       �	���Yc�A�*

loss�=�M       �	i���Yc�A�*

loss 5�<����       �	�A��Yc�A�*

loss��C<ꎩ�       �	
ٰ�Yc�A�*

lossZ�p=~��}       �	.u��Yc�A�*

lossC�=��5�       �	���Yc�A�*

lossi�,=غ�       �	0��Yc�A�*

loss�`=S��       �	Xȳ�Yc�A�*

loss���<�Qޡ       �	%y��Yc�A�*

loss�7=U��       �	���Yc�A�*

loss`�i<�0�r       �	����Yc�A�*

loss�ܐ=�#t�       �	�T��Yc�A�*

loss��<re҅       �	�網Yc�A�*

loss�L=�Y��       �	ƈ��Yc�A�*

loss�.R=೵�       �	��Yc�A�*

lossQ.=��{(       �	永�Yc�A�*

loss�2<��է       �	�D��Yc�A�*

loss1iK<[N�3       �	�ݹ�Yc�A�*

loss9�<0       �	~r��Yc�A�*

loss-[=j�q�       �	���Yc�A�*

loss�J�<b�=�       �	Ѳ��Yc�A�*

lossq��=�sza       �	�J��Yc�A�*

loss��"=\���       �	�༲Yc�A�*

loss��R=.%+�       �	-{��Yc�A�*

loss>.�=O�i�       �	"��Yc�A�*

lossI�p=�p��       �	����Yc�A�*

loss[*R<S5��       �	V��Yc�A�*

lossb�>h<��       �	E�Yc�A�*

loss`��<�t��       �	���Yc�A�*

loss�@<�˧       �	�,��Yc�A�*

lossq+=����       �	����Yc�A�*

loss�w�<6��       �	�a²Yc�A�*

loss�J%<L|�       �	Y�²Yc�A�*

loss�	�<����       �	�òYc�A�*

loss� �=v�M       �	�HŲYc�A�*

loss��<�@�       �	��ŲYc�A�*

loss��<�dD5       �	4�ƲYc�A�*

losslyE<�w�K       �	�CǲYc�A�*

loss�f�;((��       �	�ǲYc�A�*

lossxNU=q��       �	0�ȲYc�A�*

loss�>����       �	�WɲYc�A�*

loss�.>�~�       �	�ɲYc�A�*

loss��=">��       �	�ʲYc�A�*

loss���=�W��       �	4-˲Yc�A�*

loss8��<3�`�       �	X�˲Yc�A�*

loss���<���       �	�e̲Yc�A�*

lossT��=1n]�       �	|ͲYc�A�*

loss���<���       �	��ͲYc�A�*

loss��S=?��       �	�UβYc�A�*

loss��V=�1�       �	��βYc�A�*

loss���=�(H�       �	��ϲYc�A�*

lossfu(<�       �	�dвYc�A�*

lossؠ=���       �	�ѲYc�A�*

loss��n=�q�       �	�ѲYc�A�*

lossdh-=�ӏc       �	�MҲYc�A�*

loss��?=Ɛ�2       �	��ҲYc�A�*

loss�j=�	�       �	a�ӲYc�A�*

loss�""=�m._       �	W$ԲYc�A�*

lossӥ<N��       �	��ԲYc�A�*

loss�(<z�%       �	a�ղYc�A�*

lossz�<��R7       �	c(ֲYc�A�*

loss�ć;:*��       �	�ֲYc�A�*

loss)�<���       �	�aײYc�A�*

lossŋ�=B�er       �	��ײYc�A�*

loss��,=�6�       �	k�زYc�A�*

loss`!�<���       �	29ٲYc�A�*

loss�5�<����       �	ڲYc�A�*

loss���<���       �	�ڲYc�A�*

loss��<�	�       �	f۲Yc�A�*

lossgw�<=��q       �	ܲYc�A�*

loss��=�6Y�       �	�ܲYc�A�*

lossf�~<r���       �	�SݲYc�A�*

loss�{�<�t7y       �	�ݲYc�A�*

loss��,=���       �	n�޲Yc�A�*

loss�<<���d       �	�:߲Yc�A�*

lossTx<�=,3       �	��߲Yc�A�*

loss���<���       �	�y�Yc�A�*

loss�#=���       �	v�Yc�A�*

loss��<Q��       �	<��Yc�A�*

lossã+>t1L       �	\�Yc�A�*

lossn��=��P       �	���Yc�A�*

loss���<��/       �	��Yc�A�*

loss�=�9�       �	|,�Yc�A�*

lossY\=kD�+       �	���Yc�A�*

loss]�P=�j�}       �	���Yc�A�*

lossu�<��
h       �	R'�Yc�A�*

loss��h=�F�       �	���Yc�A�*

loss
��=ء>�       �	�h�Yc�A�*

loss�o�<�@�       �	��Yc�A�*

loss'!=��?@       �	+��Yc�A�*

loss�l�<�x��       �	�F�Yc�A�*

loss�Ж=~b�|       �	/��Yc�A�*

lossm�<Ѵ��       �	���Yc�A�*

loss;��=.�'c       �	�p�Yc�A�*

loss��b=��       �	F�Yc�A�*

lossH�<�! �       �	��Yc�A� *

loss&H_=^�sq       �	�G��Yc�A� *

lossK��;�ߩc       �	YQ�Yc�A� *

loss�=�,4       �	���Yc�A� *

lossJ��=����       �	���Yc�A� *

loss��S=u?,       �	<-�Yc�A� *

loss��<�e�       �	F��Yc�A� *

loss/�<�ՠ�       �	p�Yc�A� *

loss�n=��       �	g�Yc�A� *

loss22�=9З       �	S��Yc�A� *

loss��;��S       �	�S�Yc�A� *

loss�5�<d��       �	2��Yc�A� *

loss���<����       �	ѱ��Yc�A� *

loss۶�=�N\       �	�R��Yc�A� *

loss�K�<@�g�       �	$���Yc�A� *

loss#b�;���       �	����Yc�A� *

loss׿+=�g�{       �	�9��Yc�A� *

lossΒ�<#���       �	����Yc�A� *

loss�>=�"�       �	I���Yc�A� *

lossa�-=� �       �	���Yc�A� *

lossT�[<+���       �	g���Yc�A� *

loss=�=7��       �	�[��Yc�A� *

loss=�D<X�"       �	Q���Yc�A� *

lossJ�+=r��}       �	A���Yc�A� *

loss��J=#��4       �	6��Yc�A� *

loss�+�<��P>       �	����Yc�A� *

loss��"<^}       �	jm��Yc�A� *

loss�;�/�        �	���Yc�A� *

loss�;X�S       �	.���Yc�A� *

lossz�<�;�]       �	�B��Yc�A� *

loss���<��a�       �	j���Yc�A� *

loss�K<G�g       �	w �Yc�A� *

lossn�D=�(SZ       �	w�Yc�A� *

lossͺi==�       �	���Yc�A� *

lossaD�;���       �	�v�Yc�A� *

loss<�=�	�C       �	��Yc�A� *

loss_��<c��       �	��Yc�A� *

lossT�h<b?�Z       �	WB�Yc�A� *

loss���<�=��       �	��Yc�A� *

lossƇ<9��L       �	zr�Yc�A� *

loss�Ԙ<���       �	��Yc�A� *

loss�k�<�!ص       �	��Yc�A� *

loss�K�;�L̻       �	�=�Yc�A� *

loss25>:���       �	R��Yc�A� *

loss*�=PA�       �	���Yc�A� *

loss� <h��       �	x�	�Yc�A� *

loss`�z;dRc]       �	3l
�Yc�A� *

loss���:?�\       �	?�Yc�A� *

loss�1;t���       �	l��Yc�A� *

lossZ<],c^       �	z��Yc�A� *

loss��<�}D       �	r�Yc�A� *

loss֓r;�'ý       �	w�Yc�A� *

loss�va<�ۚ       �	���Yc�A� *

loss.�=>�M��       �	�F�Yc�A� *

lossd��;\O       �	��Yc�A� *

lossH�>e�       �	���Yc�A� *

loss��=�4m�       �	K:�Yc�A� *

loss��w<�h�H       �	75�Yc�A� *

lossi2�<�nG1       �	��Yc�A� *

loss-��;D��       �	/��Yc�A� *

loss8�=w<�B       �	�e�Yc�A� *

loss�?�=���T       �	"�Yc�A� *

lossF?�<���       �	m��Yc�A� *

loss��=���.       �	���Yc�A� *

loss�j�;�M$�       �	/�Yc�A� *

loss�En=)�m5       �	Н�Yc�A� *

loss?)�=u�{       �	�3�Yc�A� *

lossJ��<gz�       �	���Yc�A� *

lossAZ�=<�<�       �	�d�Yc�A� *

lossҎR=<�	Y       �	���Yc�A� *

lossTů<����       �	p��Yc�A� *

loss��<�e	�       �	"Q�Yc�A� *

loss�>=�`es       �	�
�Yc�A� *

loss�~�<�N^       �	;��Yc�A� *

lossAt�<ۮ9�       �	/��Yc�A� *

loss=\=�`�s       �	u �Yc�A� *

loss�ظ<Y�{       �	�� �Yc�A� *

loss��5;}���       �	}]!�Yc�A� *

loss ��<�]��       �	��!�Yc�A� *

loss��<ԥ�       �	��"�Yc�A� *

lossI݂=^:)�       �	�1#�Yc�A� *

lossArU;�S��       �	��#�Yc�A� *

loss���=�Nr~       �	�v$�Yc�A� *

loss�k=�C�       �	4%�Yc�A� *

lossw�s<h�>�       �	��%�Yc�A� *

loss�Ь<Ϟ�       �	Hk&�Yc�A� *

lossV�;�ޝX       �	1'�Yc�A� *

loss#�<�ǎ5       �	��'�Yc�A� *

lossoy�<��t�       �	{O(�Yc�A� *

loss�U�<~�b       �	6�(�Yc�A� *

lossR�=�[�       �	��)�Yc�A� *

lossh�-<��r�       �	*�Yc�A� *

loss1�0=H,3�       �	�*�Yc�A� *

loss��=��^�       �	P�+�Yc�A� *

loss�.;a��       �	}],�Yc�A� *

loss�4=''�       �	j�,�Yc�A� *

lossS~]=.R�D       �	\�-�Yc�A� *

loss���<b��       �	�3.�Yc�A� *

lossȑ�<�'a�       �	�.�Yc�A� *

lossm߿<1H;       �	�l/�Yc�A� *

loss�=��G       �	(
0�Yc�A� *

loss��;�TO       �	��0�Yc�A� *

loss��S<�D��       �	�K1�Yc�A� *

loss�76<O�+�       �	��1�Yc�A� *

loss
�<$��n       �	��2�Yc�A� *

loss=���       �	�O�Yc�A� *

loss��K=:���       �	g�P�Yc�A� *

loss/�o=ے��       �	�<Q�Yc�A� *

loss��*=��V�       �	��Q�Yc�A� *

loss���<=��       �	�rR�Yc�A� *

loss��={$�I       �	kS�Yc�A� *

loss�\�<v%�)       �	>�S�Yc�A� *

loss-G=�J       �	�RT�Yc�A� *

loss���=��       �	��T�Yc�A� *

loss���=�j��       �	��U�Yc�A� *

losse��<m�J       �	n1V�Yc�A� *

loss�z�<b�+z       �	�V�Yc�A� *

loss�w=�{ٖ       �	�qW�Yc�A� *

loss�� =����       �	�X�Yc�A� *

loss���<�L��       �	t�X�Yc�A� *

loss��N=��.       �	IHY�Yc�A� *

loss�G�;wW��       �	�Y�Yc�A� *

loss8y�<�~�       �	{Z�Yc�A� *

loss;e3<���       �	�[�Yc�A� *

loss�|�=��B
       �	��[�Yc�A� *

loss�
�<R�;       �	�Y\�Yc�A� *

loss�UW>��       �	v2]�Yc�A� *

loss�;�Q       �	��]�Yc�A� *

lossF�=�S��       �	:u^�Yc�A�!*

loss3�
=�7��       �	B	_�Yc�A�!*

loss��<K�߼       �	Y�_�Yc�A�!*

lossF��=)^�,       �	�?`�Yc�A�!*

loss���<��`       �	��`�Yc�A�!*

loss��_< ��       �	!xa�Yc�A�!*

loss8z:=u�L}       �	�!b�Yc�A�!*

lossx�=JE��       �	��b�Yc�A�!*

loss��y=;2js       �	\Wc�Yc�A�!*

lossD��=�P��       �	9�c�Yc�A�!*

lossL�>��1�       �	�d�Yc�A�!*

loss���<[/       �	�&e�Yc�A�!*

loss�D=yj5       �	��e�Yc�A�!*

loss#�;�j=�       �	�Zf�Yc�A�!*

loss�$v<��:       �	"�f�Yc�A�!*

loss%�,>z�My       �	�Sh�Yc�A�!*

loss�ؙ=ն�       �	y�h�Yc�A�!*

loss.	<�?��       �	��i�Yc�A�!*

loss�8�=D�N       �	B"j�Yc�A�!*

loss�d0<��       �	�j�Yc�A�!*

losso[=����       �	�[k�Yc�A�!*

loss���<���j       �	.9l�Yc�A�!*

loss.Dd=�}@�       �	r�l�Yc�A�!*

lossmW�<א7�       �	$�m�Yc�A�!*

loss�Y�<�O�       �	Όn�Yc�A�!*

lossC|< ��       �	�"o�Yc�A�!*

loss�;��p        �	�o�Yc�A�!*

loss��%<�F�       �	�Zp�Yc�A�!*

loss͗u<��       �	��p�Yc�A�!*

loss:�7<��¢       �	؛q�Yc�A�!*

loss��>��#�       �	��s�Yc�A�!*

lossԉ	=����       �	xbt�Yc�A�!*

lossT��<�x4+       �	:u�Yc�A�!*

lossr��; X       �	��u�Yc�A�!*

loss��;�-��       �	�\v�Yc�A�!*

loss
)=���       �	w�Yc�A�!*

lossO� =-$�       �	2�w�Yc�A�!*

loss�E�=@u�       �	HRx�Yc�A�!*

loss�p_=��ݦ       �	�x�Yc�A�!*

lossX�<�O#T       �	�y�Yc�A�!*

loss8wp=Ϭ�       �	Kz�Yc�A�!*

loss#�y<$]E       �	��z�Yc�A�!*

loss��0<�n�       �	~�{�Yc�A�!*

lossCl<$��i       �	�%|�Yc�A�!*

loss\7/=p)��       �	,�|�Yc�A�!*

lossݸ�=5*""       �	�i}�Yc�A�!*

loss��=Cwr�       �	
�~�Yc�A�!*

lossͷ�<�28�       �	*��Yc�A�!*

loss�̆<W�~l       �	|���Yc�A�!*

loss��<��       �	�d��Yc�A�!*

loss���<5��f       �		���Yc�A�!*

loss�p�<e��       �	6���Yc�A�!*

loss8
J<}��       �	�3��Yc�A�!*

loss�F�<#�Va       �	�ȃ�Yc�A�!*

loss�?=�E       �	�^��Yc�A�!*

loss���;�N3�       �	����Yc�A�!*

loss#YW=<�`�       �	;ǅ�Yc�A�!*

loss8,E=&��       �	^c��Yc�A�!*

lossqd�<^�^�       �	["��Yc�A�!*

loss�_�<�֧       �	���Yc�A�!*

loss!�|;�R��       �	����Yc�A�!*

loss��;���       �	Q��Yc�A�!*

loss�<�3�       �	����Yc�A�!*

loss-�<6l)�       �	�N��Yc�A�!*

lossf�=!d��       �	�&��Yc�A�!*

loss;�<����       �	����Yc�A�!*

loss'�<Z�Y       �	7���Yc�A�!*

losss�5=�%?       �	[#��Yc�A�!*

loss�5�<ڀ5G       �	����Yc�A�!*

loss���=7��3       �	�Y��Yc�A�!*

loss�=���       �	f���Yc�A�!*

lossʓm=M�       �	;���Yc�A�!*

loss���<k��       �	2"��Yc�A�!*

loss��U=7�&	       �	+���Yc�A�!*

lossv��<�F��       �	!Z��Yc�A�!*

loss�F(=`�       �	��Yc�A�!*

loss<k	]~       �	Z���Yc�A�!*

loss�*�;��]�       �	���Yc�A�!*

losszm<�Q�o       �	����Yc�A�!*

loss��=H��       �	�=��Yc�A�!*

losso��;tŇJ       �	9Օ�Yc�A�!*

loss��X<g�       �	�o��Yc�A�!*

loss�;�<��n�       �	���Yc�A�!*

loss$nL<��B       �	����Yc�A�!*

loss:�4<�.X�       �	�G��Yc�A�!*

loss�n=�'*\       �	3☳Yc�A�!*

loss.C=6#AB       �	����Yc�A�!*

loss���=�Z       �	-"��Yc�A�!*

lossۼ<�]O       �	ܺ��Yc�A�!*

loss�;�<�^M�       �	�Q��Yc�A�!*

loss�&@<e��       �	6図Yc�A�!*

lossi�)<+�R!       �	9}��Yc�A�!*

lossI=���A       �	���Yc�A�!*

lossZ�N=	}X       �	����Yc�A�!*

loss�i=��%y       �	�Z��Yc�A�!*

losskc!=����       �	-Yc�A�!*

loss�p�=A�i       �	j���Yc�A�!*

lossK�=�X�5       �	���Yc�A�!*

lossRy=��O       �	����Yc�A�!*

loss�=��       �	fI��Yc�A�!*

loss�3=g}��       �	�ޡ�Yc�A�!*

loss��@;�w��       �	�t��Yc�A�!*

lossɠ�<`N{�       �	/3��Yc�A�!*

loss���=F��V       �	�ţ�Yc�A�!*

loss��?;��c       �	�_��Yc�A�!*

loss�X=�K��       �	U���Yc�A�!*

loss�-&=�#	�       �	ȗ��Yc�A�!*

loss�{�<�1�       �	!?��Yc�A�!*

loss`��<0a �       �	�ݦ�Yc�A�!*

lossJ �<�
�6       �	�o��Yc�A�!*

loss�|@;L<]�       �	���Yc�A�!*

lossH-�=wAgV       �	Ҧ��Yc�A�!*

loss֚\<��5�       �	�=��Yc�A�!*

loss�(�<���f       �	1ѩ�Yc�A�!*

loss;��<�a�       �	sh��Yc�A�!*

loss�z�<���       �	� ��Yc�A�!*

loss:vx<X�       �	����Yc�A�!*

loss�V;�n�       �	�Ь�Yc�A�!*

loss`>�<A�0�       �	ȷ��Yc�A�!*

loss�=*1�B       �	�I��Yc�A�!*

loss	�<�=�       �	�)��Yc�A�!*

lossl�.=?��       �	KͰ�Yc�A�!*

lossX�=@Ja�       �	gc��Yc�A�!*

loss�4=�|��       �	����Yc�A�!*

loss
�<J�M�       �	W���Yc�A�!*

loss'�<�[~�       �	�.��Yc�A�!*

loss��=v�       �	����Yc�A�!*

loss���<5G�A       �	�d��Yc�A�!*

lossl��<�v��       �	�
��Yc�A�"*

lossz��=��!�       �	[���Yc�A�"*

lossy�=r�\       �	&���Yc�A�"*

loss�<�;Z��       �	g,��Yc�A�"*

loss'�;ݝ+       �	]÷�Yc�A�"*

loss&�=�\��       �	]��Yc�A�"*

loss�rI<�a�       �	�︳Yc�A�"*

losssi<P޽       �	{���Yc�A�"*

lossd�=-p]�       �	�G��Yc�A�"*

loss��=�4��       �	�ں�Yc�A�"*

loss%g�<�/1       �	Xq��Yc�A�"*

lossl��<:;       �	W��Yc�A�"*

loss$Ff<��H       �	�ռ�Yc�A�"*

loss�d�=�QVZ       �	�g��Yc�A�"*

loss���<Q�       �	����Yc�A�"*

loss$�=�.��       �	�*��Yc�A�"*

loss_.�:#��       �	7¿�Yc�A�"*

lossȝ�;d�/�       �	�[��Yc�A�"*

loss�=v���       �	����Yc�A�"*

loss�q"=Ҕ�       �	s���Yc�A�"*

loss��=;����       �	jN³Yc�A�"*

loss�E�<w:G       �	�³Yc�A�"*

loss��q=�	��       �	ŌóYc�A�"*

loss�?�;���       �	�(ĳYc�A�"*

loss�{<��fc       �	!�ĳYc�A�"*

loss߱X=Q*��       �	ߋųYc�A�"*

lossf�=��z6       �	�sƳYc�A�"*

loss	N�<��N�       �	�ǳYc�A�"*

loss��=W��       �	-�ǳYc�A�"*

loss$˱:�b       �	NGȳYc�A�"*

loss���<<ᝬ       �	��ȳYc�A�"*

lossW7n;�n�       �	�ɳYc�A�"*

loss<�<#���       �	KʳYc�A�"*

loss��;�a<A       �	��ʳYc�A�"*

loss��=*=8       �	�X˳Yc�A�"*

loss��X=��qh       �	:̳Yc�A�"*

loss��<�%)�       �	W�̳Yc�A�"*

loss�G=v�       �	E*ͳYc�A�"*

loss�z{=���       �	��ͳYc�A�"*

lossF��=}�P7       �	w�γYc�A�"*

loss/�=       �	�TϳYc�A�"*

loss@��;���.       �	S�ϳYc�A�"*

loss��<��3=       �	A�гYc�A�"*

loss��:)��W       �	t$ѳYc�A�"*

lossWo�<�q��       �	��ѳYc�A�"*

loss��?<��5       �	�RҳYc�A�"*

loss��4<"��Q       �	��ҳYc�A�"*

loss�y{=���       �	x|ӳYc�A�"*

loss��]<[55,       �	&ԳYc�A�"*

loss�X0=�r�-       �	��ԳYc�A�"*

loss��*=Uׇ�       �	�[ճYc�A�"*

loss?��<%���       �	�ճYc�A�"*

loss�7:=m�
�       �	�ֳYc�A�"*

loss��=�k��       �	�,׳Yc�A�"*

lossv��<�9u6       �	T�׳Yc�A�"*

loss-V�=��dh       �	�WسYc�A�"*

loss!̄=��       �	l�سYc�A�"*

loss{�G=� �       �	'�ٳYc�A�"*

loss$�=��k       �	�$ڳYc�A�"*

loss�.=�݈c       �	��ڳYc�A�"*

loss�k^<�@X�       �	g۳Yc�A�"*

loss7Z�;_��C       �	D�۳Yc�A�"*

loss2#=��H�       �	�ܳYc�A�"*

loss���<#�t�       �	�/ݳYc�A�"*

loss�eU=�"�       �	q�ݳYc�A�"*

loss��=��u       �	�g޳Yc�A�"*

loss��=q�P�       �	��޳Yc�A�"*

lossZ��=�0y�       �	ٕ߳Yc�A�"*

loss�1�<��l#       �	*�Yc�A�"*

loss��<g*�v       �	L��Yc�A�"*

loss��;­'       �	��Yc�A�"*

loss��=�݁2       �	-�Yc�A�"*

loss��=��       �	���Yc�A�"*

lossXB�;�j       �	mq�Yc�A�"*

loss
�=*��m       �	�Yc�A�"*

loss��<��       �	���Yc�A�"*

loss$8j=��>�       �	4�Yc�A�"*

loss)��<� V�       �	��Yc�A�"*

loss�k�=I��4       �	^h�Yc�A�"*

loss�kj<�� 9       �	���Yc�A�"*

loss���;V<��       �	y��Yc�A�"*

lossߛX<���       �	/m�Yc�A�"*

loss-�)<h.�       �	��Yc�A�"*

loss�?>=�L�]       �	F��Yc�A�"*

loss���;q7Y       �	5(�Yc�A�"*

loss�0=�ZJ       �	���Yc�A�"*

loss-5=�p[       �	eo�Yc�A�"*

loss�0(=���       �	��Yc�A�"*

loss�E�<a	o�       �	��Yc�A�"*

loss�=mz�U       �	�g��Yc�A�"*

loss?�!=R;�p       �	����Yc�A�"*

loss&i=�K�       �	���Yc�A�"*

loss�"]=\λ       �	�+�Yc�A�"*

loss�E�=nY2�       �	*�Yc�A�"*

loss��t=|y�       �	���Yc�A�"*

loss�]=*׎�       �	�o�Yc�A�"*

lossi7�<*�;�       �	+�Yc�A�"*

loss��y=e�>I       �	���Yc�A�"*

loss�@4<��b       �	?T��Yc�A�"*

loss\h�<��2p       �	����Yc�A�"*

lossL:m=��(       �	����Yc�A�"*

loss��=�`       �	'��Yc�A�"*

loss:_�;�|s�       �	����Yc�A�"*

loss��;�yNG       �	�A��Yc�A�"*

loss&V�<\�hr       �	����Yc�A�"*

loss/<V�q       �	�|��Yc�A�"*

loss#߾<�s�       �	�%��Yc�A�"*

loss9L=3С       �	M���Yc�A�"*

loss,�@=<�<       �	�`��Yc�A�"*

loss�a=k�       �	���Yc�A�"*

loss��;����       �	���Yc�A�"*

loss�b7= xO       �	!\��Yc�A�"*

loss�� =wW��       �	����Yc�A�"*

loss��<�J�       �	�:��Yc�A�"*

lossl)W<�K�<       �	����Yc�A�"*

loss��7=KD?k       �	�e��Yc�A�"*

loss���<>d�W       �	a���Yc�A�"*

loss��=����       �	� �Yc�A�"*

loss
Ob=�.�X       �	=�Yc�A�"*

loss�S<h-iX       �	;��Yc�A�"*

loss�[�<�]       �	���Yc�A�"*

loss$�<�ٓ       �	�'�Yc�A�"*

loss���=���       �	N��Yc�A�"*

loss���=M�R�       �	DN�Yc�A�"*

lossx�:=8��^       �	A��Yc�A�"*

loss��=Y���       �	A��Yc�A�"*

loss9h�;:��!       �	v3�Yc�A�"*

loss
�,=��3       �	 ��Yc�A�"*

lossȓ$=Bg	L       �	�Z�Yc�A�#*

lossi  =|�X�       �	:#�Yc�A�#*

loss�[<����       �	���Yc�A�#*

lossb<����       �	+N	�Yc�A�#*

loss��<���       �	!�	�Yc�A�#*

loss�m+=����       �	��
�Yc�A�#*

loss�!<È'       �	/�Yc�A�#*

loss��<�Jk�       �	���Yc�A�#*

loss�}�<��9       �	HS�Yc�A�#*

loss�0�;��       �	p��Yc�A�#*

lossv�;M���       �	���Yc�A�#*

loss@�o:���!       �	n1�Yc�A�#*

lossa=^�:�       �	���Yc�A�#*

lossؑ�;����       �	kb�Yc�A�#*

lossxC.:N�s�       �	\�Yc�A�#*

loss<6�<;�
       �	��Yc�A�#*

loss�G�=m�ER       �	�=�Yc�A�#*

loss��<��'�       �	��Yc�A�#*

loss-�o=�7       �	���Yc�A�#*

loss���=ڛL�       �	���Yc�A�#*

lossɖ�<����       �	vn�Yc�A�#*

loss�y	<K�9�       �	]�Yc�A�#*

loss��==l�j       �	��Yc�A�#*

loss��	=�|��       �	eR�Yc�A�#*

losss=K�Sv       �	���Yc�A�#*

losss(=�!§       �	h��Yc�A�#*

loss�!>�_�       �	H7�Yc�A�#*

loss���;��       �	,��Yc�A�#*

loss�@9=���S       �	an�Yc�A�#*

loss��<=s+�       �	��Yc�A�#*

loss�$<��b�       �	4��Yc�A�#*

loss�a�<����       �	]3�Yc�A�#*

lossR]j=��(       �	���Yc�A�#*

loss\�>=��`�       �	�i�Yc�A�#*

loss��3<�C�!       �	��Yc�A�#*

loss�)&<��3�       �	˜�Yc�A�#*

loss�b>���       �	;:�Yc�A�#*

loss�<���F       �	���Yc�A�#*

lossZ�;�U�       �	m�Yc�A�#*

lossm��<��5�       �	1 �Yc�A�#*

lossE��=��       �	�� �Yc�A�#*

loss��<���4       �	��!�Yc�A�#*

lossc2z=���       �	�"�Yc�A�#*

loss��_=����       �	5�"�Yc�A�#*

loss/ <}(�       �	�E#�Yc�A�#*

loss!wV<\ʀw       �	��#�Yc�A�#*

loss�'=��       �	w�$�Yc�A�#*

lossS�=j��Q       �	�E%�Yc�A�#*

loss�t=4d��       �	j�%�Yc�A�#*

lossc�'=Q�q       �	҉&�Yc�A�#*

loss�hH=ۤ       �	�-'�Yc�A�#*

loss�=;�'<       �	��'�Yc�A�#*

loss�t�<��?       �	0d(�Yc�A�#*

loss�(�<@���       �	\�(�Yc�A�#*

loss_�<v�
       �	k�)�Yc�A�#*

loss  �=�0,�       �	<*�Yc�A�#*

lossj�=/�-        �	��*�Yc�A�#*

lossH�-=.ߛ�       �	xz+�Yc�A�#*

lossV�;wqy       �	G ,�Yc�A�#*

loss@��=��N       �	�-�Yc�A�#*

loss��n<��!       �	��-�Yc�A�#*

loss��<"�A�       �	$`.�Yc�A�#*

lossS�;'*v�       �	�/�Yc�A�#*

loss_�K<i��       �	v�/�Yc�A�#*

loss���;U"�       �	�>0�Yc�A�#*

loss�K>���       �	��0�Yc�A�#*

loss���=�#D`       �	�\1�Yc�A�#*

loss?2<���Q       �	D�1�Yc�A�#*

lossA�=zo�"       �	��2�Yc�A�#*

loss��=�h4�       �	�<3�Yc�A�#*

losso�	=���       �	�#4�Yc�A�#*

loss<<�h c       �	�4�Yc�A�#*

loss��<g��1       �	�U5�Yc�A�#*

loss ī=��Y�       �	��5�Yc�A�#*

loss�+=�"�7       �	 �6�Yc�A�#*

loss��=��       �	bK7�Yc�A�#*

lossD�=l�Y�       �	%�7�Yc�A�#*

loss���<E�ՠ       �	`�8�Yc�A�#*

loss�\A={��       �	�79�Yc�A�#*

loss��=-�       �	�9�Yc�A�#*

loss��@=;�       �	��:�Yc�A�#*

loss�	�<w�c       �	�+;�Yc�A�#*

loss��=O#�       �	��;�Yc�A�#*

loss�jY=ڥw�       �	�q<�Yc�A�#*

loss�5>\�B       �	�=�Yc�A�#*

lossF�,=}�n�       �	y�=�Yc�A�#*

loss�#�<8h/6       �	�T>�Yc�A�#*

loss2P=�;�e       �	Q�>�Yc�A�#*

lossc$A=��R       �	�?�Yc�A�#*

loss�"M<�#	       �	�D@�Yc�A�#*

loss�T=��=       �	��@�Yc�A�#*

loss͛�<l�|�       �	�}A�Yc�A�#*

loss&S=,��.       �	�B�Yc�A�#*

loss�j=��       �	��B�Yc�A�#*

loss�;�<ݙ�6       �	�MC�Yc�A�#*

loss67=�g�a       �	&�C�Yc�A�#*

loss�A=����       �	��D�Yc�A�#*

lossf�=���       �	�kE�Yc�A�#*

loss��=���       �	\F�Yc�A�#*

loss�Ԩ<Q�F       �	D�F�Yc�A�#*

loss�M=UA�<       �	�CG�Yc�A�#*

lossط�<M&�)       �	�G�Yc�A�#*

loss���<'1�       �	SxH�Yc�A�#*

loss�t�=���       �	/I�Yc�A�#*

loss1��<�_       �	�I�Yc�A�#*

lossO��<�D=�       �	9J�Yc�A�#*

lossf��=��%6       �	�K�Yc�A�#*

loss��7=}�       �	V�K�Yc�A�#*

loss���=35��       �	�SL�Yc�A�#*

lossj�=�s
\       �	��L�Yc�A�#*

loss���<�~3n       �	��M�Yc�A�#*

lossd��<�'w�       �	�+N�Yc�A�#*

lossʽ�<+��       �	yO�Yc�A�#*

loss�$�<K�D       �	e�O�Yc�A�#*

loss���=D�f�       �	�GP�Yc�A�#*

loss0=��}�       �	v�P�Yc�A�#*

loss-�=� LX       �	JyQ�Yc�A�#*

loss�B�<?���       �	�R�Yc�A�#*

loss) �<<       �	��R�Yc�A�#*

loss�U3<����       �	�ES�Yc�A�#*

lossL=9~<       �	G�S�Yc�A�#*

loss�=�;_�^�       �	�T�Yc�A�#*

loss4�V<�2       �	6>U�Yc�A�#*

loss�m�<�_{       �	h�U�Yc�A�#*

loss�ϩ<�2:�       �	��V�Yc�A�#*

loss��+=NJ       �	�/W�Yc�A�#*

lossM.�<d��       �	�X�Yc�A�#*

lossfq>&Ċ       �	�X�Yc�A�#*

loss��<���k       �	fNY�Yc�A�$*

loss;'k<�uO�       �	>�Y�Yc�A�$*

loss�ڴ=�%�       �	�Z�Yc�A�$*

loss�)=J�       �	�[�Yc�A�$*

loss��>�J �       �	s�[�Yc�A�$*

lossly=�\^�       �	�]\�Yc�A�$*

losse+�<�Q��       �	��\�Yc�A�$*

loss�t�=��/�       �	��]�Yc�A�$*

loss� Y='�91       �	�:^�Yc�A�$*

lossJ|�;V^R       �	��^�Yc�A�$*

loss��<��       �	�_�Yc�A�$*

loss���<�($J       �	0a�Yc�A�$*

loss���<�t61       �	ݵa�Yc�A�$*

loss�5�<��       �	�cb�Yc�A�$*

loss�	=��       �	c�Yc�A�$*

loss OA<ᶐ       �	��d�Yc�A�$*

loss��=�[��       �	�Ae�Yc�A�$*

lossF��<=F3�       �	��e�Yc�A�$*

lossX��=��D�       �	t�f�Yc�A�$*

lossF�=� b       �	JCg�Yc�A�$*

lossڲ=ۖ��       �	;�g�Yc�A�$*

loss�#=�tf�       �	��h�Yc�A�$*

loss�<���~       �	�%i�Yc�A�$*

loss���<�yw_       �	��i�Yc�A�$*

loss�F=�1�G       �	B�j�Yc�A�$*

loss�0_<z�Br       �	�3k�Yc�A�$*

loss�y�<Tb�       �	m�k�Yc�A�$*

loss�lu<TI       �	ƈl�Yc�A�$*

loss\W�<ŷ�       �	{1m�Yc�A�$*

lossM�K<A�V�       �	��m�Yc�A�$*

loss-`<��_�       �	rn�Yc�A�$*

loss�c<D�ܾ       �	x�o�Yc�A�$*

loss�B;=��g       �	c'p�Yc�A�$*

loss�_�=��r�       �	5�p�Yc�A�$*

loss�Ϭ=�_�0       �	&r�Yc�A�$*

lossu�>�?��       �	1�r�Yc�A�$*

loss�=�/s�       �	Qs�Yc�A�$*

loss�I;P�       �	��s�Yc�A�$*

loss)9�<�? `       �	�t�Yc�A�$*

loss��j=YI��       �	�u�Yc�A�$*

lossjr�;$:��       �	 �u�Yc�A�$*

lossLq�<���       �	�av�Yc�A�$*

lossZm<>?��       �	�w�Yc�A�$*

loss���=�Ϧ       �	��w�Yc�A�$*

loss�+[<���       �	�dx�Yc�A�$*

loss��"=�>S�       �	gy�Yc�A�$*

loss�z�<�C�        �	�y�Yc�A�$*

loss�j�=���       �	DLz�Yc�A�$*

loss�1=�'&       �	��z�Yc�A�$*

loss���<�9��       �	ș{�Yc�A�$*

loss߅<=?�D�       �	e6|�Yc�A�$*

loss��<���       �	��|�Yc�A�$*

lossx�U=h�       �	x~}�Yc�A�$*

loss�+=b�K       �	: ~�Yc�A�$*

loss�H.<�0�       �	�~�Yc�A�$*

lossS�<+%�       �	�X�Yc�A�$*

loss�.>E@�       �	9��Yc�A�$*

lossH<�?DV       �	i���Yc�A�$*

loss���;U��       �	�,��Yc�A�$*

loss��H=�^xo       �	K��Yc�A�$*

loss
&�<U��       �	����Yc�A�$*

loss�ԕ;�bn       �	g_��Yc�A�$*

loss�!=<��^       �	P���Yc�A�$*

loss�b=l���       �	����Yc�A�$*

loss�Cz<��	�       �	�:��Yc�A�$*

loss��<j�JH       �	~ᅴYc�A�$*

loss�?|=�.�       �	�z��Yc�A�$*

losse�<p�m(       �	��Yc�A�$*

loss�v�;D5 �       �	"Ǉ�Yc�A�$*

loss<�B�`       �	����Yc�A�$*

loss%��<��%�       �	�!��Yc�A�$*

loss�l<l�       �	!ˉ�Yc�A�$*

lossN�>�&P}       �	�h��Yc�A�$*

lossq�T>�Q�       �	Q��Yc�A�$*

loss8�5=�~n�       �	ܺ��Yc�A�$*

lossvQ =F'+�       �	����Yc�A�$*

loss\^�=o���       �	%��Yc�A�$*

loss���<ּ�       �	ɍ�Yc�A�$*

loss��t<K�o       �	�g��Yc�A�$*

loss�қ=K��:       �		��Yc�A�$*

loss[1�=.�J       �	����Yc�A�$*

loss���<?y_       �	X��Yc�A�$*

loss#�<�弋       �	�$��Yc�A�$*

loss[�2=fr       �	�ϒ�Yc�A�$*

loss ?=�(��       �	�t��Yc�A�$*

loss�;�(�       �	�)��Yc�A�$*

loss8�1<�7�       �	�ʔ�Yc�A�$*

loss��<��~�       �	ę��Yc�A�$*

loss��;Q��H       �	5��Yc�A�$*

lossˋ =�<�       �	0��Yc�A�$*

loss�3<����       �	�Η�Yc�A�$*

loss;d<�       �	�j��Yc�A�$*

loss�?=mFT�       �	9��Yc�A�$*

loss�D�=��&       �	Ψ��Yc�A�$*

loss ��;o2��       �	)A��Yc�A�$*

loss���<8��t       �	ך�Yc�A�$*

loss��}<R(�l       �	�o��Yc�A�$*

loss�*=��٘       �	���Yc�A�$*

loss
�<Րz�       �	����Yc�A�$*

loss���<�lC�       �	 F��Yc�A�$*

loss�
!<�9�       �	᝴Yc�A�$*

loss�l�<H9^�       �	�x��Yc�A�$*

loss���<L�a�       �	8��Yc�A�$*

lossF�W<C��       �	�ğ�Yc�A�$*

loss�ǐ<5 �u       �	d]��Yc�A�$*

lossq1=�E�       �	��Yc�A�$*

loss�C=�~�n       �	r���Yc�A�$*

loss�<�FC       �	,(��Yc�A�$*

loss��*<�7�n       �	8���Yc�A�$*

loss��K=T�6       �	�R��Yc�A�$*

loss�u�<D��A       �	�죴Yc�A�$*

loss�C�;�D��       �	����Yc�A�$*

losscmG=�|�o       �	7��Yc�A�$*

loss���<��q       �	����Yc�A�$*

loss��<�i��       �	�G��Yc�A�$*

loss���;��L�       �	�⦴Yc�A�$*

loss�<��2;       �	���Yc�A�$*

loss��= 3%�       �	���Yc�A�$*

lossjH�<ON��       �	,���Yc�A�$*

loss��F<U֍�       �	 ^��Yc�A�$*

lossl�<���       �	���Yc�A�$*

loss@��=��,       �	O���Yc�A�$*

loss�Ū<ӂ�I       �	YN��Yc�A�$*

losscc<��       �	�Yc�A�$*

loss�HL=$���       �	0���Yc�A�$*

loss�B<o�%v       �	�>��Yc�A�$*

loss\�<��*g       �	;���Yc�A�$*

losspB;g�       �	����Yc�A�$*

loss0*<+��H       �	�ᱴYc�A�%*

lossv�
=	�_�       �	�}��Yc�A�%*

loss	�.=� �       �	"��Yc�A�%*

loss�h:����       �	o���Yc�A�%*

lossp�<��W       �	�R��Yc�A�%*

loss�2�<��-0       �	��Yc�A�%*

loss��K9#Ñ�       �	ޒ��Yc�A�%*

loss1�:�jwK       �	81��Yc�A�%*

loss_)r=r��       �	�϶�Yc�A�%*

loss b%=���       �	Y÷�Yc�A�%*

lossN�;ʳ       �	p[��Yc�A�%*

loss�t�:]�       �	A�Yc�A�%*

loss�K;��֏       �	����Yc�A�%*

loss(�->m�{       �	`!��Yc�A�%*

loss.��9�s       �	ö��Yc�A�%*

loss�4;>u���       �	�K��Yc�A�%*

loss�J	=5�S       �		⻴Yc�A�%*

lossJ��<�7�       �	w��Yc�A�%*

loss��A<򞾚       �	q=��Yc�A�%*

loss��;��$       �	Bҽ�Yc�A�%*

loss-W&=G�u       �	�i��Yc�A�%*

loss�0�=���8       �	����Yc�A�%*

lossal5=�}       �	h���Yc�A�%*

loss��<� w       �	x)��Yc�A�%*

loss���<�L�       �	����Yc�A�%*

lossTy=Ϲ�       �	2w��Yc�A�%*

loss�Uk=5u��       �	�´Yc�A�%*

loss/E�<E       �	�´Yc�A�%*

lossB�=�>k       �	BôYc�A�%*

loss�W]=1mP       �	��ôYc�A�%*

loss�K=�	�       �	?nĴYc�A�%*

loss�G�<�]�5       �	oŴYc�A�%*

loss͑�=���	       �	8�ŴYc�A�%*

loss]��<I4�       �	6:ƴYc�A�%*

loss���;�q�       �	|�ƴYc�A�%*

loss[��=\       �	�nǴYc�A�%*

loss/�<���       �	l	ȴYc�A�%*

loss�	<�B�n       �	��ȴYc�A�%*

loss�B�<�\       �	;ɴYc�A�%*

loss�ͮ;7A�       �	t�ɴYc�A�%*

loss:<�}�       �	��ʴYc�A�%*

loss�6%={�8�       �	�}˴Yc�A�%*

loss�a�=-g,       �	�.̴Yc�A�%*

loss��=�3�       �	�ʹYc�A�%*

loss���<���       �	��ʹYc�A�%*

loss���<&f       �	�EδYc�A�%*

loss�l;I<�-       �	��δYc�A�%*

losszJ;����       �	��ϴYc�A�%*

loss���<�'��       �	5$дYc�A�%*

loss���<+�}       �	��дYc�A�%*

loss&�<��L�       �	p\ѴYc�A�%*

lossr�2=�.a       �	��ѴYc�A�%*

loss��=1<j       �	��ҴYc�A�%*

loss��<&*��       �	[]ӴYc�A�%*

loss�fY<l�6       �	EԴYc�A�%*

loss{f�<&2��       �	�ԴYc�A�%*

loss�<�0�       �	JմYc�A�%*

loss{;g=)�K       �	��մYc�A�%*

lossd��<�`ZN       �	�tִYc�A�%*

lossFl�=�wz       �	W״Yc�A�%*

loss*Ou=��Z�       �	��״Yc�A�%*

loss�)<]/�       �	%<شYc�A�%*

loss��z=�D�q       �	9�شYc�A�%*

loss���:y��       �	wfٴYc�A�%*

lossA�;j��       �	��ٴYc�A�%*

losszK =����       �	9���Yc�A�%*

loss�=��#       �	2q��Yc�A�%*

lossa#=�}�       �	��Yc�A�%*

losst�=pgZQ       �	`���Yc�A�%*

lossݢ=Ƈ�       �	�K��Yc�A�%*

loss;�v=��u       �	����Yc�A�%*

loss��<,܉�       �	����Yc�A�%*

loss1i�<wv"�       �	�)��Yc�A�%*

loss�F=�Cu�       �	����Yc�A�%*

loss_>y=�̐�       �	c|��Yc�A�%*

lossH�=��1       �	�(��Yc�A�%*

loss�X<>w       �	����Yc�A�%*

loss�1<��       �	zo��Yc�A�%*

loss�ɚ=��\u       �	���Yc�A�%*

loss� �<�t&�       �	W���Yc�A�%*

lossQ	�=��+\       �	�H��Yc�A�%*

loss��Q;�f�       �	L���Yc�A�%*

lossA�W<f��>       �	8� �Yc�A�%*

loss��<����       �	��Yc�A�%*

losse�)>֠��       �	���Yc�A�%*

loss���<N*�$       �	�J�Yc�A�%*

loss.�=
���       �	���Yc�A�%*

loss�;��z�       �	���Yc�A�%*

lossT��<�ɖ       �	X�Yc�A�%*

loss���<�Q�       �	���Yc�A�%*

loss�:�<~�\W       �	>��Yc�A�%*

loss�-=\�u@       �	g+�Yc�A�%*

lossFVb;۟5�       �	��Yc�A�%*

loss��<�)]       �	'i�Yc�A�%*

loss���;�l'       �	��Yc�A�%*

loss��8<43�A       �	���Yc�A�%*

loss�~�<�1+       �	�E	�Yc�A�%*

loss�-n</cI       �	�@
�Yc�A�%*

loss�}O=��/       �	D�
�Yc�A�%*

loss�ȣ<�+       �	&p�Yc�A�%*

loss@!=�[�       �	)�Yc�A�%*

loss`��;>�2�       �	���Yc�A�%*

losst��<̩�        �	2�Yc�A�%*

loss�k�=��S�       �	`�Yc�A�%*

loss���=w���       �	��Yc�A�%*

loss,�Y=���       �	�C�Yc�A�%*

loss�d�<�5$       �	���Yc�A�%*

loss��<`_H       �	:w�Yc�A�%*

loss�Ns=���       �	�Yc�A�%*

loss.��=H�_       �	t��Yc�A�%*

loss{+.=�Ү       �	?��Yc�A�%*

lossC}=uzs.       �	%>�Yc�A�%*

loss���<�.�       �	5��Yc�A�%*

loss�� =�O       �	si�Yc�A�%*

loss�n�;y�T
       �	T�Yc�A�%*

losss\�<�c'       �	��Yc�A�%*

loss��=�uz�       �	�D�Yc�A�%*

losso�<�Y�]       �	���Yc�A�%*

lossvC�=/��M       �	�
�Yc�A�%*

loss�q�<��U�       �	S��Yc�A�%*

lossp�<�[}c       �	�Q�Yc�A�%*

loss��:D���       �	���Yc�A�%*

loss�y:�?�       �	j��Yc�A�%*

loss6 <}�Gi       �	�:�Yc�A�%*

lossv�j=�F�o       �	���Yc�A�%*

loss�R>����       �	2v�Yc�A�%*

loss�D<S>�}       �	��Yc�A�%*

lossZ�=��*�       �	���Yc�A�%*

loss�s=V�NO       �	L�Yc�A�&*

loss�<O�       �	e��Yc�A�&*

loss�9b=�~7�       �	���Yc�A�&*

loss�8�;#�K�       �	}$ �Yc�A�&*

lossQ��<�ɞ       �	{� �Yc�A�&*

lossL��<+g�       �	�T!�Yc�A�&*

loss��;] R�       �	��!�Yc�A�&*

loss���<*��V       �	@�"�Yc�A�&*

loss1A�;�픫       �	� #�Yc�A�&*

lossQΨ;�k�       �	��#�Yc�A�&*

lossĬ�=�@��       �	b$�Yc�A�&*

loss�CS<�x��       �	��$�Yc�A�&*

lossDO<�AhX       �	��%�Yc�A�&*

losst�<���P       �	s-&�Yc�A�&*

loss���<�I)       �	m�&�Yc�A�&*

loss�"<NX�       �	�\'�Yc�A�&*

losshW=��"       �	��'�Yc�A�&*

loss�>U=w��{       �	3�(�Yc�A�&*

loss���=q��       �	��)�Yc�A�&*

loss���<7z�       �	S$*�Yc�A�&*

lossN0;��g�       �	�*�Yc�A�&*

loss3��<�Zv�       �	LQ+�Yc�A�&*

loss�_<�%m�       �	��+�Yc�A�&*

loss <l�f�       �	 �,�Yc�A�&*

loss$P;���p       �	�I-�Yc�A�&*

loss2]<� ��       �	�.�Yc�A�&*

loss�= 9�b       �	�/�Yc�A�&*

loss:�;=;��       �	A�/�Yc�A�&*

loss�W,=o�O�       �	,1�Yc�A�&*

loss���<�(�       �	��1�Yc�A�&*

loss�i�=%�`�       �	@l2�Yc�A�&*

lossCwB<��F<       �	�3�Yc�A�&*

loss�;�Rq�       �	��3�Yc�A�&*

loss7��<d�8�       �	�F4�Yc�A�&*

loss@R�<� 7]       �	^�4�Yc�A�&*

loss���=i7��       �	��5�Yc�A�&*

loss
�-<~&>�       �	�:6�Yc�A�&*

lossˍ=dG}�       �	o�6�Yc�A�&*

loss;-N<�C�^       �	�|7�Yc�A�&*

loss{Y_=9� �       �	38�Yc�A�&*

loss���;�[,	       �	�8�Yc�A�&*

loss���<����       �	��9�Yc�A�&*

loss�nS>��Mz       �	>B:�Yc�A�&*

loss��"<½�       �	)�:�Yc�A�&*

loss6��;O���       �	)�;�Yc�A�&*

loss]E�=h?$       �	78<�Yc�A�&*

loss��m<��12       �	��<�Yc�A�&*

lossə�=c�       �	�n=�Yc�A�&*

loss�wx=�I(2       �	->�Yc�A�&*

lossj�<0G�u       �	j�>�Yc�A�&*

loss}E�:���D       �	�;?�Yc�A�&*

loss��x;t���       �	~�?�Yc�A�&*

loss>�<��+       �	�@�Yc�A�&*

loss�	=e���       �	FzA�Yc�A�&*

loss��<z��       �	S B�Yc�A�&*

loss褯<o���       �	R�B�Yc�A�&*

loss=[�<Πa       �	�NC�Yc�A�&*

lossj0<�0:       �	y�C�Yc�A�&*

loss׶=J�P       �	0�D�Yc�A�&*

loss*�*=(h��       �	�E�Yc�A�&*

loss��<j4�       �	�E�Yc�A�&*

loss(	�<��Ad       �	�LF�Yc�A�&*

loss��\=���1       �	��F�Yc�A�&*

loss�T=`H[       �	R�G�Yc�A�&*

loss�Y4=�'v�       �	�ZH�Yc�A�&*

loss��=�NZ       �	v�H�Yc�A�&*

loss�q?=�B�       �	��I�Yc�A�&*

loss�Ř<�U�       �	k+J�Yc�A�&*

loss��s<Y(r#       �	��J�Yc�A�&*

loss��<j�+l       �	Y�K�Yc�A�&*

loss�\=>-"Y       �	�L�Yc�A�&*

loss�_`=A��       �	ճL�Yc�A�&*

loss��=[D~[       �	QJM�Yc�A�&*

loss(2 <�!�m       �	e�M�Yc�A�&*

loss��u=i`��       �	T�N�Yc�A�&*

loss�[<��3       �	cO�Yc�A�&*

loss("�<s�(�       �	g*P�Yc�A�&*

loss�;�;���       �	�KQ�Yc�A�&*

loss`Њ<(��       �	�Q�Yc�A�&*

loss�==X��       �	�R�Yc�A�&*

loss%��<�]��       �	�&S�Yc�A�&*

lossb8�=~��W       �	�T�Yc�A�&*

lossR��<Ӂ�.       �	?U�Yc�A�&*

loss`"�=f��       �	��U�Yc�A�&*

lossݛ;#0��       �	ڎV�Yc�A�&*

loss�z�<Gл       �	�-W�Yc�A�&*

loss�O�=���       �	h�W�Yc�A�&*

loss�p]=��I       �	ΥX�Yc�A�&*

loss��<\mh        �	�:Y�Yc�A�&*

loss��=:z�       �	��Y�Yc�A�&*

lossE�<�d       �	GqZ�Yc�A�&*

loss@l=�)�       �	,([�Yc�A�&*

loss�
;��       �	v�[�Yc�A�&*

loss(:v<�i�       �	&�\�Yc�A�&*

loss]Ua=!�?'       �	�8^�Yc�A�&*

loss�K�=�l�m       �	��^�Yc�A�&*

loss͟�<�g       �	"o_�Yc�A�&*

lossO�G=j�6�       �	�`�Yc�A�&*

loss�0=/�]&       �	Φ`�Yc�A�&*

loss��\=p��       �	�ha�Yc�A�&*

loss���; d�       �	b�Yc�A�&*

loss<��=/���       �	~�b�Yc�A�&*

loss��<��N�       �	�\c�Yc�A�&*

loss�$�;��%�       �	v�c�Yc�A�&*

lossXO�<-��o       �	��d�Yc�A�&*

loss-Wm=�v8i       �	G8e�Yc�A�&*

loss	S=�!��       �	�e�Yc�A�&*

loss�e =���0       �	�kf�Yc�A�&*

loss�+?<�)��       �	�g�Yc�A�&*

loss�E<p�-       �	��g�Yc�A�&*

loss���<���       �	�\h�Yc�A�&*

loss�w�;�;�n       �	��h�Yc�A�&*

loss�b�=Q_ӊ       �	��i�Yc�A�&*

loss�^2=��$       �	S�j�Yc�A�&*

lossJz-=��|       �		2k�Yc�A�&*

loss��;���       �	`�k�Yc�A�&*

lossX��;�"�       �	�]l�Yc�A�&*

loss�B<D�       �	�l�Yc�A�&*

loss��<�a[R       �	z�m�Yc�A�&*

lossT��<��dC       �	�4n�Yc�A�&*

lossq_<�?�       �	��n�Yc�A�&*

loss�ŵ<�c E       �	�qo�Yc�A�&*

loss�]�<P��-       �	�kp�Yc�A�&*

loss�3�=��O�       �	�q�Yc�A�&*

lossH2�;�I       �	Ƣq�Yc�A�&*

lossR�<�!5�       �	r7r�Yc�A�&*

lossV�2<z��       �	�r�Yc�A�&*

lossv�m<cF�       �	{�s�Yc�A�&*

loss��=�/       �	�@t�Yc�A�'*

loss�b�<z�l'       �	�t�Yc�A�'*

loss��G<�~=!       �	czu�Yc�A�'*

loss`�;�� �       �	v�Yc�A�'*

lossV�<=�Ak3       �	h�v�Yc�A�'*

loss��b=o��       �	Lw�Yc�A�'*

loss,��<�F��       �	��w�Yc�A�'*

lossLM=ΛbU       �	7�x�Yc�A�'*

loss@X=�z|       �	�}y�Yc�A�'*

loss��<=p,��       �	�z�Yc�A�'*

loss��< +�D       �	�z�Yc�A�'*

loss1��<:B��       �	�P{�Yc�A�'*

loss��I=�t       �	$�{�Yc�A�'*

loss�h�<��Hr       �	�|�Yc�A�'*

lossA*�<$[��       �	L}�Yc�A�'*

loss��=Adh=       �	�}�Yc�A�'*

lossz2�=y7�!       �	NF~�Yc�A�'*

loss�Z=�Ρ       �	b�~�Yc�A�'*

loss��f<�H,       �	�o�Yc�A�'*

loss�ٻ<�}        �	��Yc�A�'*

loss8��<icr/       �	����Yc�A�'*

loss�L�<A��_       �	�]��Yc�A�'*

lossL1m<�|�2       �	��Yc�A�'*

loss�ϋ=�L�       �	嗂�Yc�A�'*

loss��=,G��       �	FA��Yc�A�'*

lossnV�=ϖ�       �	⃵Yc�A�'*

loss�I�=QS       �	����Yc�A�'*

loss&^=����       �	�1��Yc�A�'*

loss�=W�}u       �	兵Yc�A�'*

loss��2=1��       �	���Yc�A�'*

loss���=�y�U       �	���Yc�A�'*

lossT��<��l�       �	)���Yc�A�'*

loss)��=�t?       �	�[��Yc�A�'*

loss��|<���       �	<���Yc�A�'*

loss�9�<Ϋ��       �	����Yc�A�'*

loss�r=�z       �	W"��Yc�A�'*

loss3�<�,y�       �	����Yc�A�'*

loss�d=)�L       �	����Yc�A�'*

lossCR�<J�H�       �	7���Yc�A�'*

loss�v�;I���       �	Q��Yc�A�'*

loss�I;(�os       �	����Yc�A�'*

loss@L=��{       �	���Yc�A�'*

losscTT=��a%       �	BC��Yc�A�'*

loss�!<��       �	�؏�Yc�A�'*

loss��=j\�       �	�k��Yc�A�'*

loss�9<�tT       �	���Yc�A�'*

loss��&=AXX�       �	I���Yc�A�'*

loss��=Iɼ       �	�8��Yc�A�'*

loss�T<�\�       �	>!��Yc�A�'*

loss�/�;[��       �	����Yc�A�'*

loss!?�=F���       �	J^��Yc�A�'*

loss��< �       �	<���Yc�A�'*

lossJ�s=���       �	^���Yc�A�'*

loss66_<�d"�       �	�;��Yc�A�'*

loss@�=����       �	=Ԗ�Yc�A�'*

loss��<R��       �	�{��Yc�A�'*

loss�P,<��5�       �	���Yc�A�'*

losst��<��I�       �	0���Yc�A�'*

loss��=B��       �	�V��Yc�A�'*

loss�M9=��q�       �	��Yc�A�'*

loss���=m_�       �	>���Yc�A�'*

lossX�= #)       �	�+��Yc�A�'*

loss�cC=I�>�       �	ț�Yc�A�'*

lossa.	<�0��       �	�]��Yc�A�'*

lossH��;�e�       �	\��Yc�A�'*

loss�H=���       �	����Yc�A�'*

loss��&<���       �	{M��Yc�A�'*

loss�!�<졔�       �	�Yc�A�'*

loss:�<HK�W       �	΋��Yc�A�'*

loss@�E=��g�       �	>#��Yc�A�'*

loss��;����       �	Ǹ��Yc�A�'*

loss[>=wF�       �	�W��Yc�A�'*

lossj �<�.��       �	@���Yc�A�'*

loss�&=��r       �	X���Yc�A�'*

loss�yz=�w��       �	1$��Yc�A�'*

loss�o*=sY5�       �	�ģ�Yc�A�'*

lossĎ<����       �	BZ��Yc�A�'*

lossc�{=gU�+       �	Yc�A�'*

loss�gq<���/       �	f���Yc�A�'*

loss�8=��       �	*��Yc�A�'*

loss��<���|       �	væ�Yc�A�'*

loss�a�<Q�O�       �	BZ��Yc�A�'*

loss��<���6       �	����Yc�A�'*

lossz�>�t       �	���Yc�A�'*

loss���<�j	       �	*��Yc�A�'*

lossľK;|*       �	u̩�Yc�A�'*

loss�tB<�reI       �	g��Yc�A�'*

loss�h=L��       �	 ��Yc�A�'*

loss�=X��       �	����Yc�A�'*

loss�O�:f�"�       �	�D��Yc�A�'*

loss�/C<�E�       �	
׬�Yc�A�'*

lossL	=h3�       �	�m��Yc�A�'*

loss��_=] /       �	W>��Yc�A�'*

loss���=TR�j       �	���Yc�A�'*

loss���<��8�       �	���Yc�A�'*

loss�u=[��       �	E���Yc�A�'*

loss�\[<�       �	�\��Yc�A�'*

loss�?�;u��@       �	4���Yc�A�'*

losshW�;<�       �	����Yc�A�'*

loss�gt;��xC       �	�>��Yc�A�'*

loss䛇<��j       �	*೵Yc�A�'*

loss?�;�t��       �	˂��Yc�A�'*

lossO�H;i-��       �	���Yc�A�'*

lossG=x��       �	����Yc�A�'*

loss�d>�{�       �	�Q��Yc�A�'*

lossi_=H~�        �	춵Yc�A�'*

lossH�<�       �	���Yc�A�'*

loss_X=����       �	%���Yc�A�'*

lossOT==�*`e       �	�B��Yc�A�'*

loss�C#<]���       �	�湵Yc�A�'*

loss}��<|�S�       �	���Yc�A�'*

lossH<���       �	"5��Yc�A�'*

loss���=D�K       �	ѻ�Yc�A�'*

loss�U�="�w�       �	�s��Yc�A�'*

loss00>��y�       �	E��Yc�A�'*

lossEN�<�0�|       �	xս�Yc�A�'*

loss�Փ<M� �       �	]k��Yc�A�'*

lossϸ=<è5       �	���Yc�A�'*

loss"<���       �	����Yc�A�'*

loss�<t�`       �	�>��Yc�A�'*

loss�=�ڂ�       �	����Yc�A�'*

loss��=�Rn       �	}u��Yc�A�'*

loss��;���       �	IµYc�A�'*

loss�qd<�94(       �	�µYc�A�'*

loss���<��0�       �	�NõYc�A�'*

lossv��<c;�^       �	;�õYc�A�'*

loss1��<|��       �	O�ĵYc�A�'*

loss��=t,�\       �	�+ŵYc�A�'*

lossʻu=zD'#       �	��ŵYc�A�(*

loss�ȱ;�G�       �	`ƵYc�A�(*

loss��=���0       �	j�ƵYc�A�(*

loss��=�6��       �	�ǵYc�A�(*

loss#�i<dHz       �	6ȵYc�A�(*

loss(^�<%Lv       �	N�ȵYc�A�(*

lossc��=#J�%       �	+nɵYc�A�(*

loss��9<rXH�       �	�8ʵYc�A�(*

loss1e�=�ڲ�       �	_�ʵYc�A�(*

loss/%�<c��       �	�e˵Yc�A�(*

lossO�=ڝ��       �	��˵Yc�A�(*

lossN�/<=P�       �	1�̵Yc�A�(*

loss���<�� �       �	��͵Yc�A�(*

loss� �;�1Y�       �	E-εYc�A�(*

loss!Ɉ<�z�       �	u�εYc�A�(*

loss�c'=���       �	��ϵYc�A�(*

loss}vu<(&       �	
1еYc�A�(*

lossz$=��rh       �	�ѵYc�A�(*

lossX��<�wʎ       �	@�ѵYc�A�(*

loss��=yͬ�       �	�KҵYc�A�(*

loss��;����       �	|�ҵYc�A�(*

lossC"P<����       �	��ӵYc�A�(*

loss��<ՈBb       �	�*ԵYc�A�(*

loss��=+�Є       �	��ԵYc�A�(*

loss��A=�@�?       �		�յYc�A�(*

loss�H�=�Jh�       �	�PֵYc�A�(*

lossg�=�N�       �	�׵Yc�A�(*

lossn�<��j[       �	k�׵Yc�A�(*

loss�n�=�y��       �	��صYc�A�(*

lossi�<��       �	�gٵYc�A�(*

loss]h�<��[�       �	��ٵYc�A�(*

loss#2�<�~&�       �	`�ڵYc�A�(*

loss�U�<�%�       �	�*۵Yc�A�(*

loss|��=+��c       �	e�۵Yc�A�(*

loss���<�:#�       �	hXܵYc�A�(*

loss�j�<"Gl-       �	J�ܵYc�A�(*

loss���<=`��       �	6�ݵYc�A�(*

loss��<��!�       �	O>޵Yc�A�(*

losso�<=���       �	4�޵Yc�A�(*

loss�;I��       �	�nߵYc�A�(*

lossS�9=Un9t       �	%�Yc�A�(*

lossڕ�<U=�       �	g��Yc�A�(*

loss���<��"�       �	U3�Yc�A�(*

loss���;@�Ŷ       �	q��Yc�A�(*

loss��=�EaB       �	x_�Yc�A�(*

loss��<����       �	b0�Yc�A�(*

loss�b�=�Х       �	���Yc�A�(*

lossR�X<r��+       �	7o�Yc�A�(*

loss��0<^]�k       �	��Yc�A�(*

loss`�;XWV�       �	x��Yc�A�(*

lossZ�6<����       �	�<�Yc�A�(*

loss$D�<�Z :       �	d��Yc�A�(*

loss�9�;@�Y�       �	Ŏ�Yc�A�(*

loss��; �       �	�5�Yc�A�(*

lossz��<�s�       �	��Yc�A�(*

loss��~=�J�       �	���Yc�A�(*

loss�2=�~��       �	�4�Yc�A�(*

loss�R=���       �	���Yc�A�(*

loss��<;�5�       �	1{�Yc�A�(*

loss��<:�f�       �	� �Yc�A�(*

loss�g=��`�       �	���Yc�A�(*

lossvY=IB4l       �	l]��Yc�A�(*

loss�y)=��!       �	�Yc�A�(*

lossJ�>����       �	>��Yc�A�(*

loss_D�<����       �	ji�Yc�A�(*

loss�n�;���       �	�Yc�A�(*

loss���=��kF       �	!��Yc�A�(*

loss�y�=����       �	x�Yc�A�(*

loss�b�="��i       �	���Yc�A�(*

loss�:!=��X`       �	�_�Yc�A�(*

loss�;���M       �	���Yc�A�(*

loss�+='��       �	����Yc�A�(*

loss���<s�        �	�g��Yc�A�(*

loss���<2��x       �	���Yc�A�(*

lossDwa<І��       �	����Yc�A�(*

loss�'�;f��       �	�V��Yc�A�(*

loss[�=�T�:       �	���Yc�A�(*

loss�R�;s4�Q       �	���Yc�A�(*

loss���<�mk�       �	/��Yc�A�(*

loss���<���X       �	����Yc�A�(*

loss�1=��       �	�x��Yc�A�(*

loss��&<|�Q�       �	�$��Yc�A�(*

loss-6V=C�;       �	Q���Yc�A�(*

losso` >_�U�       �	KZ��Yc�A�(*

lossd2�;����       �	����Yc�A�(*

lossd�=���       �	����Yc�A�(*

loss*� =(���       �	)Y��Yc�A�(*

loss/�=�&       �	M���Yc�A�(*

loss�X=�^�S       �	����Yc�A�(*

loss�2=��wj       �	�3 �Yc�A�(*

loss��J=���{       �	� �Yc�A�(*

lossI~=Ct7       �	���Yc�A�(*

lossJ_E=���       �	�'�Yc�A�(*

loss!Z�<�0W�       �	��Yc�A�(*

loss�C=�t       �	qr�Yc�A�(*

loss�L<�0��       �	�(�Yc�A�(*

loss	z<H>�       �	���Yc�A�(*

loss	�^=�0       �	eo�Yc�A�(*

loss�.�<~"�       �	��Yc�A�(*

lossF�=���H       �	z��Yc�A�(*

lossꈭ=t�Jh       �	�T�Yc�A�(*

lossX�t<I���       �	��Yc�A�(*

loss ��=�m�       �	v��Yc�A�(*

lossx�<�<�       �	�T	�Yc�A�(*

loss�=GE9       �	f�	�Yc�A�(*

loss��%=0�       �	>�
�Yc�A�(*

loss�>�<#��,       �	��Yc�A�(*

lossh׎<���X       �	[C�Yc�A�(*

loss��$=�	       �	F��Yc�A�(*

loss	��<��a�       �	��Yc�A�(*

loss%'=���       �	F"�Yc�A�(*

loss��<��:e       �	�x�Yc�A�(*

loss���<�4�       �	��Yc�A�(*

loss��<��נ       �	���Yc�A�(*

loss���<�A�       �	���Yc�A�(*

loss��=|�:       �	��Yc�A�(*

lossi��<�{�       �	c^�Yc�A�(*

loss%��<'P�       �	��Yc�A�(*

loss���<�)��       �	e��Yc�A�(*

loss�W�;;T�       �	�A�Yc�A�(*

loss?�s<�+       �	��Yc�A�(*

loss�`�=t\y(       �	c��Yc�A�(*

loss��=��F       �	�<�Yc�A�(*

loss�^x=uđ�       �	��Yc�A�(*

loss�>�=�P��       �	���Yc�A�(*

lossa�3<v;�       �	�H�Yc�A�(*

lossZEf=m�       �	���Yc�A�(*

lossT�R=m���       �	���Yc�A�(*

loss�>1=���       �	�#�Yc�A�)*

loss��n<*'^        �	5��Yc�A�)*

loss�8�<)���       �	h��Yc�A�)*

loss���=���L       �	�)�Yc�A�)*

loss���;�p��       �	���Yc�A�)*

loss��<9.l�       �	�`�Yc�A�)*

loss
$K=����       �	�Yc�A�)*

loss�Z�= vP       �	Z��Yc�A�)*

lossJ0-<(�       �	�4 �Yc�A�)*

losst�<W�a�       �	�� �Yc�A�)*

losss�={b6�       �	W�!�Yc�A�)*

loss��B<�J^       �	eP"�Yc�A�)*

loss�%�<͟ԧ       �	�#�Yc�A�)*

lossQR�<n�q       �	�^$�Yc�A�)*

loss�1K<��4       �	�%�Yc�A�)*

loss�<V�n       �	�&�Yc�A�)*

lossR{=�]��       �	Ҭ&�Yc�A�)*

loss�<:�       �	G'�Yc�A�)*

lossf��<G��d       �	�c(�Yc�A�)*

loss�~<�"8       �	0)�Yc�A�)*

loss�1<�WT       �	��)�Yc�A�)*

loss�F�<cl       �	�L*�Yc�A�)*

loss��;\�bt       �	�*�Yc�A�)*

loss�G�=��m�       �	y�+�Yc�A�)*

loss���<bV��       �	~R,�Yc�A�)*

lossڤ=5�U       �	P�,�Yc�A�)*

loss&�0=Ӭ��       �	հ-�Yc�A�)*

lossX��;R�y�       �	YO.�Yc�A�)*

loss3�;�'��       �	>�.�Yc�A�)*

loss:M�;�_F�       �	:0�Yc�A�)*

loss��=�b       �	��0�Yc�A�)*

loss�kG<L ��       �	�2�Yc�A�)*

loss}W`=�8�       �	�?3�Yc�A�)*

loss�ޜ=O1�       �	4�Yc�A�)*

loss}^�=U��       �	]55�Yc�A�)*

loss!x�<�O�       �	$�5�Yc�A�)*

lossx4�<ó�       �	M�6�Yc�A�)*

loss�/=r�M       �	�37�Yc�A�)*

loss�M�<\)       �	p�7�Yc�A�)*

loss�0�;����       �	��8�Yc�A�)*

loss�]�=LQ�O       �	��9�Yc�A�)*

loss�w�;�<u�       �	z9:�Yc�A�)*

loss̲i=�2�Q       �	_�:�Yc�A�)*

loss�R�<s�       �	��;�Yc�A�)*

loss��
<�X       �	�6<�Yc�A�)*

loss8��;��#P       �	��<�Yc�A�)*

lossc}�<�2�{       �	��=�Yc�A�)*

loss� <X�+       �	-�>�Yc�A�)*

loss��=�Y�Q       �	3�?�Yc�A�)*

loss3�=�Z,�       �	R�@�Yc�A�)*

loss1��;V�       �	kGA�Yc�A�)*

lossh��<�B2       �	��A�Yc�A�)*

loss�X=O��       �	%vB�Yc�A�)*

lossȱ�<\:��       �	 C�Yc�A�)*

loss�p<�1o�       �	Z�C�Yc�A�)*

lossV�<���       �	�>D�Yc�A�)*

loss(ř;��Ea       �	t�D�Yc�A�)*

loss�G�<����       �	�fE�Yc�A�)*

loss=.7��       �	z�E�Yc�A�)*

lossF�<���p       �	�F�Yc�A�)*

lossRz�<�?�H       �	�kG�Yc�A�)*

loss�)<O�'�       �	TH�Yc�A�)*

loss�=�x@�       �	l�H�Yc�A�)*

loss���;.�C�       �	+I�Yc�A�)*

loss�~�<E�        �	�J�Yc�A�)*

losse5]<p�X�       �	��J�Yc�A�)*

loss�S�=�0N       �	��K�Yc�A�)*

loss�G_<*&�i       �	�?L�Yc�A�)*

loss1��<��D�       �	Z�L�Yc�A�)*

loss��=��]       �	�mM�Yc�A�)*

loss�z<$��       �	�eN�Yc�A�)*

loss�]<\R�       �	r�N�Yc�A�)*

lossw;�=��Yv       �	�O�Yc�A�)*

loss�U�<���,       �	�eP�Yc�A�)*

loss\)<x�:V       �	��P�Yc�A�)*

loss��;\�F       �	��Q�Yc�A�)*

loss���<��)g       �	MS�Yc�A�)*

lossu�=�L�       �	��S�Yc�A�)*

losslt�<l�       �	�:T�Yc�A�)*

lossז<<v���       �	p�T�Yc�A�)*

loss���;ʗu%       �	sfU�Yc�A�)*

loss%QO=c        �	�V�Yc�A�)*

loss��P<��       �	y<W�Yc�A�)*

lossQ�;Ą�       �	x�W�Yc�A�)*

loss��4=�7�4       �	wgX�Yc�A�)*

lossL�;���       �	�X�Yc�A�)*

lossi}<�P �       �	��Y�Yc�A�)*

loss`�<
ܫ�       �	�(Z�Yc�A�)*

loss�Ş<�ʹ�       �	��Z�Yc�A�)*

loss�A;�
��       �	{h[�Yc�A�)*

loss�-�<s%��       �	T�[�Yc�A�)*

loss	�+;7>��       �	&�\�Yc�A�)*

loss�r<�J�#       �	�>]�Yc�A�)*

lossɮY=a�C       �	1�]�Yc�A�)*

loss�&9Q��       �	I�^�Yc�A�)*

loss
�;����       �	�7_�Yc�A�)*

loss4h<��H       �	��_�Yc�A�)*

lossF��<ֻ��       �	lz`�Yc�A�)*

loss��<�*       �	�a�Yc�A�)*

loss���:mY��       �	��a�Yc�A�)*

loss�o�;Q&w&       �	O]b�Yc�A�)*

lossM9�=)qj       �	b�b�Yc�A�)*

loss�_;����       �	X�c�Yc�A�)*

loss N.>��}       �	c&d�Yc�A�)*

loss��"=y�@�       �	��d�Yc�A�)*

loss�n�<q�g�       �	�e�Yc�A�)*

loss|v�=_��Z       �	b,f�Yc�A�)*

loss"�<4q�       �	a�f�Yc�A�)*

loss$��<�k       �	dXg�Yc�A�)*

loss�_�<�8�       �	
�g�Yc�A�)*

lossqM�;���       �	��h�Yc�A�)*

loss_Y4=��h       �	#ji�Yc�A�)*

lossT=�#�V       �	4j�Yc�A�)*

loss@��<й{s       �	��j�Yc�A�)*

loss��o=.���       �	�Jk�Yc�A�)*

lossq�=�#��       �	��k�Yc�A�)*

lossAz�=���|       �	6ul�Yc�A�)*

loss�rm=��r       �	�m�Yc�A�)*

losst��<ع��       �	˻m�Yc�A�)*

loss���;�W��       �	}Yn�Yc�A�)*

loss�H<9�,       �	��n�Yc�A�)*

loss�I�;�hH^       �	��o�Yc�A�)*

loss���;!���       �	��p�Yc�A�)*

loss�^=�$��       �	�Tq�Yc�A�)*

loss��c=�E�       �	r�Yc�A�)*

loss(�N;���       �	&�r�Yc�A�)*

loss��^<Rs��       �	us�Yc�A�)*

loss�_<��Ģ       �	�Pt�Yc�A�)*

loss�!�<�a��       �	�5u�Yc�A�**

lossSq#<X3��       �	��u�Yc�A�**

loss�b�=�ܵH       �	lv�Yc�A�**

loss^|�=;�2�       �	+Pw�Yc�A�**

loss��+<I�=       �	+jx�Yc�A�**

loss]�=M<:�       �	�
y�Yc�A�**

lossO� =Z<u       �	o�y�Yc�A�**

loss��;
�\       �	�z�Yc�A�**

lossJW�<����       �	�s{�Yc�A�**

loss7�D<�i�       �	�|�Yc�A�**

loss�;-��       �	�|�Yc�A�**

lossH��<��h1       �	�X}�Yc�A�**

loss���<
NV�       �	/�}�Yc�A�**

loss��</{��       �	��~�Yc�A�**

lossj��<
^6       �	�0�Yc�A�**

loss���<�rj       �	���Yc�A�**

loss)�C<�sɨ       �	8���Yc�A�**

loss-��<���:       �	X��Yc�A�**

loss���;mq�)       �	m��Yc�A�**

loss�G�;��r       �	�˂�Yc�A�**

lossx��=Z��       �	kf��Yc�A�**

lossCy�;�`4j       �	���Yc�A�**

lossl:=��       �	����Yc�A�**

loss�"t<����       �	z6��Yc�A�**

loss��<�t       �	�҅�Yc�A�**

loss�%�=ޅja       �	�ȡ�Yc�A�**

loss�2�<���-       �	#f��Yc�A�**

loss#�N=�r|       �	����Yc�A�**

loss�U�=��Ӌ       �	Kȣ�Yc�A�**

lossD2�<9>�x       �	�[��Yc�A�**

loss�N<�E-�       �	��Yc�A�**

loss�k�<<���       �	H���Yc�A�**

loss���;�       �	k-��Yc�A�**

loss���<i��       �	]æ�Yc�A�**

lossD�=	",J       �	Z��Yc�A�**

lossHA�;ɍ�       �	���Yc�A�**

lossOr0</m�S       �	E���Yc�A�**

loss�	=��SP       �	�q��Yc�A�**

lossvZ<7 0�       �	�
��Yc�A�**

loss�)=NH�       �	8���Yc�A�**

loss��;_�Z
       �	�3��Yc�A�**

loss��:����       �	���Yc�A�**

loss�[k<�숙       �	���Yc�A�**

loss�kD<��({       �	hB��Yc�A�**

lossSv�<���7       �	v㭶Yc�A�**

loss���<lX       �	,���Yc�A�**

loss<��=��D       �	j4��Yc�A�**

loss��<���5       �	�1��Yc�A�**

loss��J=Us�       �	�߰�Yc�A�**

loss��T<�~�`       �	j���Yc�A�**

lossM?�<�X�       �	�U��Yc�A�**

loss R<��Ƽ       �	����Yc�A�**

loss�!;ᄅ$       �	0*��Yc�A�**

loss��;��٪       �	մ�Yc�A�**

lossӶ=�t       �	mr��Yc�A�**

loss�_<��E�       �	!��Yc�A�**

losst*�<�$       �	����Yc�A�**

loss�[�<���       �	�d��Yc�A�**

loss��=i2i       �	;���Yc�A�**

loss��<�B,�       �	]���Yc�A�**

loss`��<�;b       �	�<��Yc�A�**

loss��<�q�       �	�빶Yc�A�**

loss��;#�	�       �	����Yc�A�**

loss��=����       �	.��Yc�A�**

loss1�=����       �	o���Yc�A�**

loss$�#=�N�H       �	iS��Yc�A�**

loss�ކ<&p�       �	��Yc�A�**

loss؆�:��o       �	Ɏ��Yc�A�**

loss�Ȉ=��*=       �	�,��Yc�A�**

loss}�=���       �	7Ŀ�Yc�A�**

loss�� <@v6^       �	_��Yc�A�**

loss��y<R�)       �	���Yc�A�**

lossy�<+�o�       �	����Yc�A�**

loss�L;�N       �	�j¶Yc�A�**

lossl��<����       �	v3öYc�A�**

lossZ�<	�K       �	��öYc�A�**

loss��;���       �	�ĶYc�A�**

loss��;��#�       �	�,ŶYc�A�**

loss��=
��        �	��ŶYc�A�**

lossr
=X_xs       �	�lƶYc�A�**

loss�F�<A�l�       �	�ǶYc�A�**

loss�j$::��       �	˺ǶYc�A�**

loss]q�:&�~       �	�lȶYc�A�**

loss��;8_�       �	�ɶYc�A�**

loss���<d�L�       �	�ɶYc�A�**

loss̐�=K� �       �	?UʶYc�A�**

loss#�=� ��       �	��ʶYc�A�**

lossFz<$'?�       �	l�˶Yc�A�**

loss���<w�d       �	4K̶Yc�A�**

loss�_;�p��       �	O�̶Yc�A�**

lossė<�~�h       �	��ͶYc�A�**

loss<����       �	�ζYc�A�**

lossI�;�C�       �	4϶Yc�A�**

loss�Ӆ=#ī       �	�жYc�A�**

loss���<SD�]       �	��жYc�A�**

loss��<Q�k�       �	�OѶYc�A�**

loss��!=�|
       �	�&ҶYc�A�**

loss�?�<F}�       �	��ҶYc�A�**

loss���<�+       �	�kӶYc�A�**

loss�l;���       �	k)ԶYc�A�**

loss|J_;�F��       �	��ԶYc�A�**

loss,|;�c�"       �	ֶYc�A�**

loss�ڂ<:�`       �	¦ֶYc�A�**

lossM?�<���       �	K?׶Yc�A�**

loss�g=       �	��׶Yc�A�**

lossT�;�g�       �	�rضYc�A�**

loss��<�"�       �	�ٶYc�A�**

loss�'=�(�       �	��ٶYc�A�**

lossA�^<z�N�       �	�UڶYc�A�**

lossI��:뿷�       �	G۶Yc�A�**

loss�e�;j�	       �	�۶Yc�A�**

loss{Ҏ<ː�b       �	AFܶYc�A�**

loss���;{       �	��ܶYc�A�**

loss�j�;���I       �	�ݶYc�A�**

loss��a<;�       �	�8޶Yc�A�**

loss�Ʊ<&G٘       �	KZ�Yc�A�**

loss�QQ=����       �	P�Yc�A�**

lossS̲<ل��       �	˝�Yc�A�**

loss.�h<+��       �	�8�Yc�A�**

loss�-�=�.       �	9��Yc�A�**

loss,T~<�H�       �	Ho�Yc�A�**

lossw!�<���V       �	E�Yc�A�**

losss �<���:       �	���Yc�A�**

loss(?�=/��       �	)[�Yc�A�**

loss��.<K��       �	��Yc�A�**

lossE��:��.       �	���Yc�A�**

loss��<��D       �	kH�Yc�A�**

lossE��<L��       �	���Yc�A�**

loss�L<�i޿       �	���Yc�A�+*

loss�X <�o�       �	�8�Yc�A�+*

lossr=���       �	���Yc�A�+*

loss�;*��       �	\s�Yc�A�+*

loss��<�ty�       �	<�Yc�A�+*

loss��z=�3	       �	���Yc�A�+*

loss3ڹ<���       �	+M�Yc�A�+*

loss��<�(��       �	5��Yc�A�+*

lossd��=�VE�       �	Έ��Yc�A�+*

losspD�<tz�       �	�$�Yc�A�+*

loss�v�;�sI       �	k��Yc�A�+*

loss���<7_>�       �	Xp�Yc�A�+*

loss��m:c��       �	��Yc�A�+*

loss��;�<��       �	¾�Yc�A�+*

loss��&=��yn       �	qZ�Yc�A�+*

lossM�<?��       �	`�Yc�A�+*

losszS<���!       �	��Yc�A�+*

loss��%= h       �	�[�Yc�A�+*

loss�|=��o�       �	��Yc�A�+*

loss$p�<�)�J       �	���Yc�A�+*

loss��;L:�       �	Ad��Yc�A�+*

lossp@�<�N�D       �	g��Yc�A�+*

loss���=�2�?       �	����Yc�A�+*

loss�Z+=o���       �	Pp��Yc�A�+*

lossan<x>)B       �	���Yc�A�+*

losstFA=Y��       �	���Yc�A�+*

lossH�=C�       �	{j��Yc�A�+*

loss8۩<!Y�       �	���Yc�A�+*

losswD=���       �	r���Yc�A�+*

loss��C=B���       �	-@��Yc�A�+*

loss�̸<�
�a       �	����Yc�A�+*

lossh��=j��       �	k���Yc�A�+*

lossO��;B�2       �	���Yc�A�+*

lossc�<U���       �	����Yc�A�+*

loss��<�JG�       �	�L��Yc�A�+*

loss6�j;?U^�       �	����Yc�A�+*

lossSX;i��>       �	8���Yc�A�+*

loss&��;��Mv       �	� �Yc�A�+*

lossR~0=��'       �	p� �Yc�A�+*

loss�M;='[Jr       �	�F�Yc�A�+*

loss�/S=�ɱ�       �	Q��Yc�A�+*

loss��=����       �	.s�Yc�A�+*

loss��9=ѝ�       �	+�Yc�A�+*

loss��<��W       �	��Yc�A�+*

loss���;Z��       �	Hp�Yc�A�+*

loss��7<0"�b       �	��Yc�A�+*

loss!�C=����       �	{��Yc�A�+*

lossFU=q�,       �	:�Yc�A�+*

loss=J��       �	<��Yc�A�+*

loss�h�;K"       �	�z�Yc�A�+*

loss��<��$~       �	��Yc�A�+*

losseP=���       �	~��Yc�A�+*

loss�;�       �	y<	�Yc�A�+*

loss�<OO8       �	��	�Yc�A�+*

loss��E=�L�       �	eq
�Yc�A�+*

loss8=�x\j       �	��Yc�A�+*

loss�ۗ<�03       �	_��Yc�A�+*

loss�=ds�       �	R�Yc�A�+*

lossJ=%�7       �	���Yc�A�+*

loss���<��	       �	ӣ�Yc�A�+*

loss{��;��"*       �	(F�Yc�A�+*

loss�zV=M+(�       �	��Yc�A�+*

lossm��<�73o       �	��Yc�A�+*

lossҚ�<��~       �	A��Yc�A�+*

loss��3=�k�       �	_'�Yc�A�+*

loss��"<�8�q       �	��Yc�A�+*

loss�X;���       �	�|�Yc�A�+*

loss�:l=wZ�       �	� �Yc�A�+*

loss��'<� �       �	���Yc�A�+*

loss�/�<{���       �	�Y�Yc�A�+*

loss�$<�       �	��Yc�A�+*

loss��U<M�g       �	Ǜ�Yc�A�+*

loss���=V��        �	�5�Yc�A�+*

lossx��<�A�O       �	U��Yc�A�+*

loss_�G=�=�+       �	��Yc�A�+*

loss_(!=�Y�       �	K�Yc�A�+*

lossc��;0GΏ       �	���Yc�A�+*

lossT��:h~2�       �	v��Yc�A�+*

loss���;�	��       �	�'�Yc�A�+*

loss���<_R]�       �	���Yc�A�+*

loss���;��-n       �	<j�Yc�A�+*

loss�0<W���       �	" �Yc�A�+*

lossW��<&�s�       �	���Yc�A�+*

lossDy< D�       �	s-�Yc�A�+*

loss��<cR�F       �	L��Yc�A�+*

lossCv;=���       �	�]�Yc�A�+*

loss�l=���       �	V��Yc�A�+*

loss`�<2B��       �	��Yc�A�+*

lossT�Q<�a��       �	& �Yc�A�+*

lossN5M;�I�       �	�� �Yc�A�+*

lossͻn;l��       �	ji!�Yc�A�+*

loss{5:;"T       �	��!�Yc�A�+*

loss���;& �{       �	�"�Yc�A�+*

lossaP�<	I       �	�8#�Yc�A�+*

loss�=�#�?       �	X�#�Yc�A�+*

loss�R<���*       �	��$�Yc�A�+*

lossjy�;�s�6       �	�%�Yc�A�+*

loss�yj<��       �	��%�Yc�A�+*

lossM* <���s       �	�G&�Yc�A�+*

lossi!�;�ik       �	q�&�Yc�A�+*

loss�®<D¨�       �	D�'�Yc�A�+*

loss�ً<���       �	�(�Yc�A�+*

loss�Y=ì��       �	=�(�Yc�A�+*

loss݅�=$0��       �	K)�Yc�A�+*

loss�8�<��r       �	��)�Yc�A�+*

loss	yo<>�ּ       �	Gu*�Yc�A�+*

lossI<<%!�e       �	l
+�Yc�A�+*

loss��<�%?�       �	Z�+�Yc�A�+*

loss;|;�4�       �	�A,�Yc�A�+*

loss��<Ž�       �	I�,�Yc�A�+*

loss�d<z���       �	�w-�Yc�A�+*

loss�+=���       �	�#.�Yc�A�+*

loss�L=$��G       �	��.�Yc�A�+*

loss��)>�rR�       �	��/�Yc�A�+*

loss���=��       �	�0�Yc�A�+*

loss.H}= �NZ       �	�R1�Yc�A�+*

lossѫ<=��}       �	�1�Yc�A�+*

loss��<����       �	S�2�Yc�A�+*

loss���;�!�       �	^H3�Yc�A�+*

loss&s�<���6       �	.�3�Yc�A�+*

loss}-�=^�?       �	Ū4�Yc�A�+*

loss�QC;���       �	2�5�Yc�A�+*

loss�i<���       �	�6�Yc�A�+*

loss�y;���m       �	Q17�Yc�A�+*

lossAc=�Ѳk       �	��7�Yc�A�+*

loss:
�<b��^       �	J�8�Yc�A�+*

lossL��<%*�a       �	��9�Yc�A�+*

loss�A<>�}�       �	w0:�Yc�A�+*

loss,�R=D��       �	�;�Yc�A�,*

loss�x=Al��       �	.<�Yc�A�,*

loss�s�;��D�       �	��<�Yc�A�,*

loss4g=5��~       �	[x=�Yc�A�,*

loss�<R�,       �	1#>�Yc�A�,*

loss���<,;�       �	�>?�Yc�A�,*

loss��<�]��       �	/�?�Yc�A�,*

lossPW�<�숼       �	�@�Yc�A�,*

lossR�;܅e       �	) A�Yc�A�,*

loss#=���       �	��A�Yc�A�,*

loss�\�<�{�       �	�fB�Yc�A�,*

loss�p<��0       �	k'C�Yc�A�,*

lossv0G=＇       �	��C�Yc�A�,*

loss�5=��l"       �	[�D�Yc�A�,*

loss� =�QS�       �	W�E�Yc�A�,*

loss�|&<�Y	�       �	�EF�Yc�A�,*

loss%5�;�Ex�       �	e�F�Yc�A�,*

loss�a�<���       �	�xG�Yc�A�,*

loss�pu<þ�u       �	�*H�Yc�A�,*

loss��*=��l�       �	o�H�Yc�A�,*

loss�pz= �0       �	ӃI�Yc�A�,*

loss�<�ec4       �	!J�Yc�A�,*

loss��:���       �	�J�Yc�A�,*

lossR��;�C�       �	�PK�Yc�A�,*

losst�3=ٶ�        �	�K�Yc�A�,*

loss$F=� k7       �	){L�Yc�A�,*

loss�;�;�8y�       �	HM�Yc�A�,*

loss�uo=}@�        �	��M�Yc�A�,*

loss1�"=0��       �	�NN�Yc�A�,*

loss��t=�mO�       �	��N�Yc�A�,*

loss�o�<�n&"       �	|�O�Yc�A�,*

loss)?.<��       �	$P�Yc�A�,*

loss5��<�Ã       �	��P�Yc�A�,*

loss�K�<W^�       �	W^Q�Yc�A�,*

lossb <�m8I       �	� R�Yc�A�,*

loss���<ѣ��       �	�R�Yc�A�,*

loss��=|���       �	�6S�Yc�A�,*

lossm�=.@r       �	��S�Yc�A�,*

loss���<����       �	�T�Yc�A�,*

loss�J2=Fl�D       �	�TU�Yc�A�,*

loss��4=�B       �	�U�Yc�A�,*

lossV�*=s��       �	T�V�Yc�A�,*

loss�T�=`��M       �	�W�Yc�A�,*

loss���=Ǩ�       �	4�W�Yc�A�,*

lossZ��=�� +       �	)ZX�Yc�A�,*

lossI�<U��        �	n�X�Yc�A�,*

lossvib;F;Z
       �	2�Y�Yc�A�,*

loss���=I0��       �	#KZ�Yc�A�,*

loss��N<k�       �	��Z�Yc�A�,*

loss���<���       �	�[�Yc�A�,*

loss�8];�'�       �	\�Yc�A�,*

losstXn<�i�)       �	��\�Yc�A�,*

lossA��<�@:�       �	�]]�Yc�A�,*

loss� �=d��I       �	kI^�Yc�A�,*

lossl=�<��       �	!�^�Yc�A�,*

loss��u=��       �	��_�Yc�A�,*

loss �n<l��@       �	F"`�Yc�A�,*

loss4��;j��(       �	��`�Yc�A�,*

loss��<c&��       �	�ea�Yc�A�,*

losse)<5io�       �	�a�Yc�A�,*

lossw��;낪Z       �	�b�Yc�A�,*

lossa;�;�1�W       �	9*c�Yc�A�,*

loss!L�;�=t6       �	�c�Yc�A�,*

lossXk�;RU�       �	[d�Yc�A�,*

loss��<J���       �	8e�Yc�A�,*

loss�[B<4�*       �	H�e�Yc�A�,*

loss�>=V��_       �	8f�Yc�A�,*

losscP�<#֡       �	,�f�Yc�A�,*

lossC-�<;��a       �	��g�Yc�A�,*

loss3�(=D�
       �	�wh�Yc�A�,*

lossm�%=���       �	�
i�Yc�A�,*

lossA�A=�!C%       �	{�i�Yc�A�,*

loss-�=��[�       �	�<j�Yc�A�,*

losscRC=�       �	�j�Yc�A�,*

loss)j>����       �	�kk�Yc�A�,*

loss�;:5�       �	2Ul�Yc�A�,*

loss�<"�X       �	)�l�Yc�A�,*

lossX-=��J�       �	 �m�Yc�A�,*

loss��<#`nY       �	o*n�Yc�A�,*

loss��=��E       �	F�n�Yc�A�,*

loss_�<e�C�       �	�oo�Yc�A�,*

loss[�<���'       �	xp�Yc�A�,*

loss�H<�v��       �	N�p�Yc�A�,*

loss���;yY�       �	��q�Yc�A�,*

loss��
<��:�       �	h[r�Yc�A�,*

loss�k�;'͜�       �	�s�Yc�A�,*

loss#�F;���M       �	W>t�Yc�A�,*

lossfQ�=�{k       �	,�t�Yc�A�,*

loss��=��xd       �	�u�Yc�A�,*

loss:M�<��g�       �	6 v�Yc�A�,*

lossw�A=%@-�       �	��v�Yc�A�,*

loss�|�<Z��       �	Uw�Yc�A�,*

loss԰<�L        �	8�w�Yc�A�,*

lossHW=�Z       �	��x�Yc�A�,*

loss8�3=�q       �	*7y�Yc�A�,*

lossO��;K�r�       �	��y�Yc�A�,*

lossL�<r�Ѿ       �	��z�Yc�A�,*

loss]c�<�}       �	fK{�Yc�A�,*

lossF�<w�l       �	��{�Yc�A�,*

loss��W<9�{t       �	w|�Yc�A�,*

loss�><:��       �	}�Yc�A�,*

loss�}><����       �	_�}�Yc�A�,*

loss���;T��       �	ep~�Yc�A�,*

loss��=��w�       �	��Yc�A�,*

loss埸<��       �	*��Yc�A�,*

lossZ\:=�9       �	S@��Yc�A�,*

loss&�(=�V       �	�Ԁ�Yc�A�,*

loss�?=P|WG       �	k��Yc�A�,*

lossQ�5;�q%h       �	$��Yc�A�,*

loss�R�:���       �	 ���Yc�A�,*

loss�B�<�D-�       �	@M��Yc�A�,*

loss��<s��       �	 僷Yc�A�,*

lossɪ�<4�1�       �	����Yc�A�,*

loss�p�=\�3(       �	���Yc�A�,*

loss��=�:�       �	����Yc�A�,*

lossm�	< N7�       �	DQ��Yc�A�,*

loss���=B�S;       �	����Yc�A�,*

loss�Cd<�ǖ       �	p���Yc�A�,*

loss��V<lt�       �	�C��Yc�A�,*

loss.';��v�       �	�߈�Yc�A�,*

loss��<D��p       �	����Yc�A�,*

loss��U=ŷ�       �	%$��Yc�A�,*

loss` B;{Ө!       �	+���Yc�A�,*

loss3�=9�h       �	kf��Yc�A�,*

loss�
�=TH       �	���Yc�A�,*

loss�}=I��       �	n���Yc�A�,*

losseT�;����       �	�?��Yc�A�,*

losst<Ӯ!       �	/፷Yc�A�,*

lossx��<���       �	܃��Yc�A�-*

loss��;�V�       �	�=��Yc�A�-*

lossӕ[<�h��       �	Mڏ�Yc�A�-*

loss�n&<�ц       �	1x��Yc�A�-*

loss�I$=H�n       �	���Yc�A�-*

loss��<K��       �	����Yc�A�-*

loss�8�<���       �	�X��Yc�A�-*

loss�]�<���T       �	�L��Yc�A�-*

loss�<��b�       �	`䓷Yc�A�-*

loss4ٗ<xpL�       �	=���Yc�A�-*

lossL<��"{       �	i7��Yc�A�-*

loss3t�<�t(�       �	����Yc�A�-*

losst
<|�'�       �	*ǖ�Yc�A�-*

loss�6�;!���       �	탗�Yc�A�-*

loss~�<p��       �	X��Yc�A�-*

lossՈ =�&��       �	_���Yc�A�-*

loss�S<�أU       �	4K��Yc�A�-*

loss_�=<^J�k       �	�㙷Yc�A�-*

lossx�<<jA�       �	)y��Yc�A�-*

loss�HT<�0�       �	5��Yc�A�-*

loss`yi=?F�       �	�Yc�A�-*

lossW=�i��       �	�X��Yc�A�-*

loss���<#���       �	�윷Yc�A�-*

loss3�j=�-��       �	b���Yc�A�-*

loss?_Z<�Dv�       �	L��Yc�A�-*

loss���<�d3       �	󮞷Yc�A�-*

loss�W�=z�+�       �	�G��Yc�A�-*

loss��P=��$B       �	C៷Yc�A�-*

loss�S�=2�q�       �	�v��Yc�A�-*

loss�=�:�b       �	���Yc�A�-*

loss�=_��       �	����Yc�A�-*

losse�'=�E       �	�C��Yc�A�-*

lossme<K��       �	nݢ�Yc�A�-*

loss�l<9�,�       �	�~��Yc�A�-*

loss�;Y=�}1       �	���Yc�A�-*

loss��=`ア       �	[���Yc�A�-*

lossƊ�=�Y�       �	�F��Yc�A�-*

lossdv=b
       �	�ݥ�Yc�A�-*

loss@T�< Z'�       �	w��Yc�A�-*

loss�<�`�       �	^��Yc�A�-*

loss)�=@z#       �	U���Yc�A�-*

loss�T�;e�К       �	����Yc�A�-*

loss�p�<���       �	�-��Yc�A�-*

lossM�=�㲉       �	�Ʃ�Yc�A�-*

loss6(7</�i       �	�_��Yc�A�-*

loss���=��       �	
���Yc�A�-*

loss��;<4�T�       �	����Yc�A�-*

loss�B=��S�       �	1��Yc�A�-*

loss�a�<ƁZf       �	~ɬ�Yc�A�-*

lossiqu<�"S5       �	�b��Yc�A�-*

lossNv="�&�       �	����Yc�A�-*

loss�{h=��        �	����Yc�A�-*

loss2�=O]u       �	�H��Yc�A�-*

lossr1�<��P       �	�ܯ�Yc�A�-*

loss(X�<��t       �	�t��Yc�A�-*

loss�c=PXS       �	)<��Yc�A�-*

loss*��=�ʲ�       �	fܱ�Yc�A�-*

loss-�g;C(�       �	�q��Yc�A�-*

loss7#�<���       �	���Yc�A�-*

lossޙ�<Ҷ�       �	ᷳ�Yc�A�-*

loss{�?=�x�       �	�Q��Yc�A�-*

loss�5&=6	�[       �	�Yc�A�-*

loss�MB=�3�M       �	y���Yc�A�-*

lossD|;�	-�       �	 (��Yc�A�-*

lossf��=�Q��       �	�ٶ�Yc�A�-*

loss�5�<��"|       �	����Yc�A�-*

loss�8�<��_`       �	�$��Yc�A�-*

loss�a=�j�       �	�ø�Yc�A�-*

loss��"=О��       �	�c��Yc�A�-*

loss��<vep       �	�	��Yc�A�-*

loss��;v,-�       �	ߣ��Yc�A�-*

loss��Y<m��e       �	
I��Yc�A�-*

loss{<�G�       �	k�Yc�A�-*

lossTy9<��       �	 ���Yc�A�-*

loss�W6<��       �	�2��Yc�A�-*

loss�a<|��Z       �	ӽ�Yc�A�-*

loss��6=s:q+       �	�o��Yc�A�-*

loss�z#<��       �	���Yc�A�-*

loss�o=u��v       �	3���Yc�A�-*

loss�M�<�-L       �	�A��Yc�A�-*

loss�w�;����       �	j���Yc�A�-*

loss��<]@<       �	�r��Yc�A�-*

loss��B=�v�       �	�·Yc�A�-*

loss;�<�o�       �	��·Yc�A�-*

loss�k=H��}       �	&8÷Yc�A�-*

loss���;_P�w       �	��÷Yc�A�-*

loss��=����       �	�ķYc�A�-*

loss�6�=��       �	�ŷYc�A�-*

loss��<��I       �	
�ŷYc�A�-*

loss��<��%m       �	�WƷYc�A�-*

loss;a=ޣ�=       �	��ƷYc�A�-*

loss��L=�\X�       �	��ǷYc�A�-*

losso:�w M       �	�ȷYc�A�-*

loss
[;����       �	�ȷYc�A�-*

loss�%�=����       �	d?ɷYc�A�-*

loss��)>�y�       �	��ɷYc�A�-*

losslE=
�,�       �	�xʷYc�A�-*

loss=E=#�;S       �	�	˷Yc�A�-*

loss��<gg�r       �	g�˷Yc�A�-*

loss&:�;�~x8       �	�1̷Yc�A�-*

lossr2�;���       �	.�̷Yc�A�-*

lossd=;W�0       �	�\ͷYc�A�-*

lossJ	$=�H]       �	��ͷYc�A�-*

loss/��<�T�       �	b�ηYc�A�-*

loss�ǟ=�0RX       �	�ϷYc�A�-*

loss��;=�D�-       �	зYc�A�-*

loss�/<��L�       �	��зYc�A�-*

loss�sD<y�5�       �	�WѷYc�A�-*

loss n+<��x       �	�lҷYc�A�-*

loss�XD<{ے�       �	�	ӷYc�A�-*

loss�x�;��K       �	��ӷYc�A�-*

loss]��=#�       �	t�ԷYc�A�-*

loss�e�;�)*}       �	�}շYc�A�-*

loss{�<�x�       �	DoַYc�A�-*

loss�X�=���       �	�B׷Yc�A�-*

loss�-D;愾�       �	.طYc�A�-*

loss�L�<�@3K       �	��طYc�A�-*

loss��7<��t       �	~�ٷYc�A�-*

lossR¹;Y���       �	odڷYc�A�-*

lossڜT;��Ҙ       �	6۷Yc�A�-*

loss��=�        �	�۷Yc�A�-*

loss��=nߝM       �	dܷYc�A�-*

lossܑs<1=��       �	�ݷYc�A�-*

loss���<Q�b�       �	��ݷYc�A�-*

lossW!*<�	\.       �	�j޷Yc�A�-*

lossF	=
/��       �	�	߷Yc�A�-*

loss��;E�       �	Z�߷Yc�A�-*

lossKw;�э�       �	"6�Yc�A�-*

loss8�<j'�I       �	��Yc�A�.*

loss�A�;~y��       �	sf�Yc�A�.*

loss��<�h>l       �	�Yc�A�.*

loss���<�d��       �	���Yc�A�.*

loss�6=w�b^       �	�A�Yc�A�.*

loss��;�MN       �	5��Yc�A�.*

loss�%�=a�y�       �	v��Yc�A�.*

loss;��C�       �	JF�Yc�A�.*

loss|�;�Z!M       �	r��Yc�A�.*

loss,�b<KhM�       �	Tt�Yc�A�.*

loss�DP<���       �	
�Yc�A�.*

loss���<��       �	
��Yc�A�.*

loss4{=����       �	�J�Yc�A�.*

loss��<����       �	z��Yc�A�.*

loss7�	<h�Ot       �	�y�Yc�A�.*

loss
�D={t�       �	��Yc�A�.*

lossZ� =���       �	{��Yc�A�.*

loss�aG=HL��       �	�=�Yc�A�.*

loss��5<�մ3       �	}��Yc�A�.*

loss4Y�<��>�       �	#��Yc�A�.*

lossFS�:���       �	��Yc�A�.*

loss.]4<�Q�y       �	S���Yc�A�.*

loss�eC=�1��       �	eQ�Yc�A�.*

loss�C<�K��       �	���Yc�A�.*

loss�ʾ<E�Q       �	m��Yc�A�.*

lossaj�;VY�.       �	[a�Yc�A�.*

loss��=���       �	�:�Yc�A�.*

loss<�= �3�       �	�+�Yc�A�.*

losst��=ԓ�n       �	�B�Yc�A�.*

loss��g<Cc{       �	w��Yc�A�.*

loss=�~<���P       �	���Yc�A�.*

losso�=�X�i       �	?���Yc�A�.*

loss`]-=�e]        �	�y��Yc�A�.*

loss�g;��_       �	KW��Yc�A�.*

loss�_�;�h�1       �	�:��Yc�A�.*

loss���;����       �	��Yc�A�.*

lossύ<����       �	
���Yc�A�.*

lossع�<0���       �	=a��Yc�A�.*

loss���<�<�       �	v���Yc�A�.*

loss�:<U߅       �	����Yc�A�.*

loss�\=��'       �	><��Yc�A�.*

loss��=3��       �	i���Yc�A�.*

loss�8<���       �	��Yc�A�.*

loss}ґ<��i�       �	�#��Yc�A�.*

loss�>�<Y(՟       �	����Yc�A�.*

lossϋ;��e       �	0g��Yc�A�.*

lossx:ҏ1       �	A �Yc�A�.*

loss��+;���       �	� �Yc�A�.*

loss�٥<9p�       �	�a�Yc�A�.*

loss:�u;j��       �	��Yc�A�.*

loss�W6<L2�|       �	���Yc�A�.*

loss8�Z:���O       �	�;�Yc�A�.*

loss$e<���       �	���Yc�A�.*

lossMn;.3ڨ       �	��Yc�A�.*

loss�[39rEw       �	�%�Yc�A�.*

loss���9f�!k       �	���Yc�A�.*

loss;�>;�X(�       �	S]�Yc�A�.*

lossđg<qD�8       �	���Yc�A�.*

loss���;
��@       �	)��Yc�A�.*

loss�/:��4�       �	�*�Yc�A�.*

loss,�;�       �	���Yc�A�.*

loss���=���       �	��	�Yc�A�.*

loss;'x;a3.7       �	��
�Yc�A�.*

loss�U>�r�`       �	it�Yc�A�.*

loss���;�u�]       �	9
�Yc�A�.*

loss	
Z=qt       �	w��Yc�A�.*

lossH�c<�_8�       �	�=�Yc�A�.*

loss�H�<�g�       �	z��Yc�A�.*

loss/Y = ZWG       �	mt�Yc�A�.*

loss3͡</PB�       �	N�Yc�A�.*

loss�;7=8�~       �	n��Yc�A�.*

loss?��<�?�u       �	&8�Yc�A�.*

lossT�"=2��       �	���Yc�A�.*

loss���=�a�       �	�h�Yc�A�.*

loss�L*=�ݿ�       �	�N�Yc�A�.*

loss�:<�A�'       �	$��Yc�A�.*

losswx+=n��       �	Ɖ�Yc�A�.*

loss7��<��       �	�%�Yc�A�.*

loss�Ob=��X3       �	���Yc�A�.*

loss^��<�:�A       �	�d�Yc�A�.*

loss7!=x��       �		�Yc�A�.*

lossX��;�(7�       �	ܛ�Yc�A�.*

loss/ g;�Sy       �	
.�Yc�A�.*

loss ؖ<��<&       �	���Yc�A�.*

loss:c=~�       �	N_�Yc�A�.*

loss��<��s       �	��Yc�A�.*

loss��;W@.�       �	N��Yc�A�.*

loss[��;+f       �	҉�Yc�A�.*

loss@=�H[       �	+0�Yc�A�.*

lossR)<�E�       �	���Yc�A�.*

lossD;�=˂\       �	|_�Yc�A�.*

loss��=Zd�       �	��Yc�A�.*

loss8�<��O�       �	b��Yc�A�.*

loss�\J=����       �	L�Yc�A�.*

loss�<y�m�       �	��Yc�A�.*

lossHn;`<��       �	:v�Yc�A�.*

loss�<��       �	� �Yc�A�.*

loss|�3<=ݶ�       �	J� �Yc�A�.*

lossB3	<���       �	�K!�Yc�A�.*

loss��-={�	�       �	t�!�Yc�A�.*

loss��x=��l�       �	�"�Yc�A�.*

loss���<���       �	5$#�Yc�A�.*

loss��;ѠD�       �	�#�Yc�A�.*

loss� �;�փ       �	�V$�Yc�A�.*

loss	�<�%�       �	��$�Yc�A�.*

loss2gL<�Ȟb       �	U�%�Yc�A�.*

lossK��=���E       �	�&�Yc�A�.*

lossj;>O4Q�       �	}�&�Yc�A�.*

lossJ��<&�p       �	�B'�Yc�A�.*

loss�}T;?~��       �	�'�Yc�A�.*

loss��=~��4       �	6s(�Yc�A�.*

lossž�:Aq�,       �	O)�Yc�A�.*

loss�T<�iJ|       �	��)�Yc�A�.*

loss�ɋ=nM(^       �	�YB�Yc�A�.*

loss%%=�\�t       �	x�B�Yc�A�.*

loss�i=�0g�       �	w�C�Yc�A�.*

loss��<fp�       �	�kD�Yc�A�.*

lossO��<w M�       �	E�Yc�A�.*

losse�=[���       �	��E�Yc�A�.*

loss�A_<cw       �	�PF�Yc�A�.*

loss�xc=�}4       �	��F�Yc�A�.*

loss���<K5=g       �	�~G�Yc�A�.*

losswX:=F5KC       �	�H�Yc�A�.*

loss�#;K�Y�       �	�H�Yc�A�.*

lossZ��<��o       �	�HI�Yc�A�.*

loss&��<�c�       �	�$J�Yc�A�.*

loss�b.<�9�       �	��J�Yc�A�.*

loss��M<n6�       �	WK�Yc�A�.*

loss�Q=G �       �	��K�Yc�A�/*

loss��:�&^;       �	��L�Yc�A�/*

loss$s�;pot�       �	�VM�Yc�A�/*

loss��<!K�       �	�	N�Yc�A�/*

loss=1-o       �	��N�Yc�A�/*

loss$<�;J6�C       �	W|O�Yc�A�/*

loss���<�VA�       �	sP�Yc�A�/*

loss6�*<�*U�       �	7�P�Yc�A�/*

lossI1�=�&h�       �	[@Q�Yc�A�/*

lossI�j<<�ka       �	��Q�Yc�A�/*

loss���<a�N�       �	6�R�Yc�A�/*

loss�2�<:ˏ�       �	��S�Yc�A�/*

loss8Y�<\�       �	x�T�Yc�A�/*

loss���<4�
       �	*:U�Yc�A�/*

loss�<0�(@       �	�V�Yc�A�/*

loss��Y=�)��       �	�V�Yc�A�/*

loss_�S<����       �	`XW�Yc�A�/*

loss�ܟ;\'�R       �	��W�Yc�A�/*

lossI$G=���y       �	0�X�Yc�A�/*

loss�@$=�VQ�       �	�oY�Yc�A�/*

loss3��<dCК       �	!Z�Yc�A�/*

loss�
<�[&�       �	�Z�Yc�A�/*

lossH�<b��5       �	MM[�Yc�A�/*

lossSt�<��F�       �	u�[�Yc�A�/*

loss ~�=�c�       �	\�Yc�A�/*

loss�6=I�2       �	�^]�Yc�A�/*

loss��=pZ�g       �	{�]�Yc�A�/*

loss<�<�[77       �	Ҋ^�Yc�A�/*

loss��<�!�       �	�_�Yc�A�/*

loss1Ð<����       �	�_�Yc�A�/*

lossۘ1=���       �	�S`�Yc�A�/*

loss���<�a       �	l�`�Yc�A�/*

loss���;%Pb�       �	�a�Yc�A�/*

loss�l�<�,7       �	b�Yc�A�/*

loss�<�՚�       �	U�b�Yc�A�/*

loss$��:�ǻ�       �	�ec�Yc�A�/*

loss<�=�z��       �	��c�Yc�A�/*

lossd�<H��y       �	e�d�Yc�A�/*

loss���=�E+8       �	2e�Yc�A�/*

loss�%�<�%�]       �	�e�Yc�A�/*

loss��9n%�       �	Sg�Yc�A�/*

loss
��:�75       �	�g�Yc�A�/*

loss���;.�]       �	w�h�Yc�A�/*

loss1��;�[w       �	� i�Yc�A�/*

loss�\�=�C�v       �	C�i�Yc�A�/*

lossW��=��>y       �	�fj�Yc�A�/*

loss͏
<�+��       �	Sk�Yc�A�/*

loss��<�       �	�k�Yc�A�/*

lossҴf=��ٲ       �	v4l�Yc�A�/*

loss�x<���       �	��l�Yc�A�/*

lossn2;d       �	�bm�Yc�A�/*

loss~�<���       �	��m�Yc�A�/*

loss��=�J,       �	G�n�Yc�A�/*

loss��<�F�N       �	�*o�Yc�A�/*

loss&f<e\&       �	K�o�Yc�A�/*

lossl�9=u!K       �	�p�Yc�A�/*

loss$�+=O�5�       �	�<q�Yc�A�/*

loss)�=d��       �	�r�Yc�A�/*

loss�;��\       �	�r�Yc�A�/*

loss�<�1�v       �	��s�Yc�A�/*

lossj�;35�v       �	J}t�Yc�A�/*

loss�`=P]0e       �	�u�Yc�A�/*

loss*�X<�}I�       �	�[v�Yc�A�/*

loss���;Q��o       �	�Gw�Yc�A�/*

loss��< ��T       �	@�w�Yc�A�/*

lossԿ=Bֹ       �	�yx�Yc�A�/*

loss�z=e��G       �	�y�Yc�A�/*

loss��<b�@       �	��y�Yc�A�/*

loss���;��O�       �	¾z�Yc�A�/*

loss�	=�7�       �	j{�Yc�A�/*

lossfP�<���       �	�|�Yc�A�/*

loss^&<��tS       �	��|�Yc�A�/*

lossn��;pz�       �	�}�Yc�A�/*

loss�>�<���       �	'L~�Yc�A�/*

loss��@=i%�       �	��~�Yc�A�/*

loss��_<[���       �	�|�Yc�A�/*

lossm#�<H�{5       �	f��Yc�A�/*

loss(�=B�,       �	a���Yc�A�/*

loss��;�؅       �	<M��Yc�A�/*

lossiq=<X-�       �	ꁸYc�A�/*

loss�:;�y��       �	����Yc�A�/*

loss i3<h�       �	�%��Yc�A�/*

losss�f<ξW�       �	kԃ�Yc�A�/*

loss`WP=�7F       �	8j��Yc�A�/*

loss���<S�}/       �	���Yc�A�/*

lossx�<<����       �	U���Yc�A�/*

lossn�:�a�       �	^i��Yc�A�/*

loss_<�z|�       �	��Yc�A�/*

loss�
3<����       �	����Yc�A�/*

lossV�;�%Pd       �	C���Yc�A�/*

lossFv=<f�*5       �	�2��Yc�A�/*

loss���;X`խ       �	�܉�Yc�A�/*

lossn�;9�q       �	1}��Yc�A�/*

loss��/=��+�       �	l"��Yc�A�/*

loss{�L<u$/       �	����Yc�A�/*

lossױ>=!�]�       �	DO��Yc�A�/*

loss�VY=�:�       �	xYc�A�/*

loss��<���W       �	����Yc�A�/*

loss�js;o�̎       �		4��Yc�A�/*

lossA:�<_1{       �	�׎�Yc�A�/*

loss���;�l��       �	�v��Yc�A�/*

loss�I=��";       �	N��Yc�A�/*

loss�==����       �	F���Yc�A�/*

loss�<b!�       �	�F��Yc�A�/*

loss1�<# �       �	�]��Yc�A�/*

loss*kK<]�P       �	�T��Yc�A�/*

lossD;<���       �	�!��Yc�A�/*

loss�%
=ۆ�       �	c���Yc�A�/*

loss�E(=�       �	�P��Yc�A�/*

loss���:"       �	�蕸Yc�A�/*

lossM��<)�L       �	=���Yc�A�/*

loss���<����       �	�h��Yc�A�/*

lossw��;�V!       �	���Yc�A�/*

loss�=����       �	���Yc�A�/*

loss��<=-��       �		��Yc�A�/*

loss`'�<�̴       �	0՚�Yc�A�/*

loss�M�<S�       �	o��Yc�A�/*

loss�3< �wf       �	'��Yc�A�/*

lossr^t<���       �	 ���Yc�A�/*

lossC�G=> �&       �	�~��Yc�A�/*

loss��m<5�~�       �	���Yc�A�/*

loss�f<<�h�U       �	u���Yc�A�/*

loss�;��BG       �	�R��Yc�A�/*

loss\��;&y�H       �	o���Yc�A�/*

loss1Bc;4e(�       �	ލ��Yc�A�/*

loss;��:	%D�       �	�W��Yc�A�/*

losssF	<��Ъ       �	�Yc�A�/*

lossɓu=
eW�       �	
���Yc�A�/*

loss��3=�*Ee       �	���Yc�A�0*

loss�)�<ì�       �	����Yc�A�0*

loss��8=�{vp       �	!Y��Yc�A�0*

lossd�;����       �	�1��Yc�A�0*

loss��:�+��       �	�ȥ�Yc�A�0*

losso7�<���       �	-]��Yc�A�0*

loss���;��       �	��Yc�A�0*

loss�\�;��n       �	㍧�Yc�A�0*

loss-o�<��2�       �	;5��Yc�A�0*

lossXڀ=4!�&       �	&娸Yc�A�0*

loss1u�<�IfG       �	����Yc�A�0*

loss�W@=�T�e       �	6!��Yc�A�0*

loss;a;�:       �	�ƪ�Yc�A�0*

loss�_I<�q8�       �	�m��Yc�A�0*

lossd�<�dI�       �	�o��Yc�A�0*

loss]�=�	�       �	� ��Yc�A�0*

losss31<�dvx       �	k���Yc�A�0*

loss�Q�;{�֡       �	P��Yc�A�0*

lossv!�;��H&       �	�许Yc�A�0*

loss��I=�bg�       �	����Yc�A�0*

lossn�<\6�       �	���Yc�A�0*

loss�}�<Tvц       �	����Yc�A�0*

loss��<�W�q       �	�K��Yc�A�0*

loss�<4��E       �	�q��Yc�A�0*

loss��<u/{C       �	����Yc�A�0*

loss�;�R3�       �	�M��Yc�A�0*

lossc��<$�r       �	�h��Yc�A�0*

loss��=)       �	o��Yc�A�0*

losse<¨       �	�䶸Yc�A�0*

loss��z;��7       �	/᷸Yc�A�0*

lossWSr=���t       �	����Yc�A�0*

loss��<����       �	X;��Yc�A�0*

losse�.=?k�       �	�⹸Yc�A�0*

loss1��<I#       �	����Yc�A�0*

loss��<r���       �	CU��Yc�A�0*

loss�E$;��[�       �	�껸Yc�A�0*

lossÐ�<w��       �	8���Yc�A�0*

lossѼ�:��
�       �	�>��Yc�A�0*

loss@PY<�U       �	r޽�Yc�A�0*

loss�!=̂r       �	�w��Yc�A�0*

loss���;Qa�L       �	���Yc�A�0*

loss=�6<�:�K       �	>���Yc�A�0*

loss�5�;���       �	�H��Yc�A�0*

lossq4=�rl�       �	.���Yc�A�0*

loss�Q�<DPF       �	w��Yc�A�0*

loss-�<KU�       �	�¸Yc�A�0*

loss8}#<ݝ�\       �	1�¸Yc�A�0*

loss�AD=�d�       �	�]øYc�A�0*

loss���<$S*"       �	ĸYc�A�0*

loss�P<��VK       �	Z�ĸYc�A�0*

lossJ�:,�       �	29ŸYc�A�0*

loss��:ơ,\       �	c�ŸYc�A�0*

loss�B�;{��+       �	�ƸYc�A�0*

loss<�]'       �	�(ǸYc�A�0*

loss/ �<�{�       �	D�ǸYc�A�0*

loss�V�=�i#;       �	�_ȸYc�A�0*

loss<w;��Z       �	#�ȸYc�A�0*

loss �U<�F��       �	�ʸYc�A�0*

loss��<���       �	��ʸYc�A�0*

loss�C�;�I�E       �	�3˸Yc�A�0*

loss{-�;�~�M       �	��˸Yc�A�0*

loss8i�<�@       �	y\̸Yc�A�0*

lossd��; ��]       �	�_͸Yc�A�0*

loss7�=�F;       �	��͸Yc�A�0*

lossD�V=�g�       �	��θYc�A�0*

loss_�<L�P�       �	�JϸYc�A�0*

loss扄<�$��       �	�ϸYc�A�0*

loss��S=6n��       �	,�иYc�A�0*

losst�<�^�       �	�OѸYc�A�0*

loss��=�s�       �	�ѸYc�A�0*

loss.Q�<�U�       �	ҸYc�A�0*

loss��X=X�o�       �	�ӸYc�A�0*

loss��=�4]       �	W�ӸYc�A�0*

loss���=���       �	sfԸYc�A�0*

loss��<�)6�       �	�ոYc�A�0*

loss`+=�7@       �	��ոYc�A�0*

lossV�W=t�       �	�yָYc�A�0*

loss;kh;����       �	�׸Yc�A�0*

loss\[�=Yo�       �	��׸Yc�A�0*

loss��;o�Z       �	�?ظYc�A�0*

loss�dS==�       �	b�ظYc�A�0*

lossR��;ʰ�{       �	3mٸYc�A�0*

loss�!=K��       �	�ڸYc�A�0*

loss��<M���       �	�ڸYc�A�0*

loss�i�;NJ�       �	�.۸Yc�A�0*

loss#u<أ@       �	%�۸Yc�A�0*

loss�1�<gV��       �	9aܸYc�A�0*

loss]ҵ:��C       �	��ܸYc�A�0*

loss��<o~5�       �	��ݸYc�A�0*

loss܃<֖,�       �	&޸Yc�A�0*

lossh�;'&�1       �	Q�޸Yc�A�0*

loss�A�<q��       �	�U߸Yc�A�0*

loss:�Y<Ռi       �	Y�߸Yc�A�0*

loss�t)<�Qu�       �	���Yc�A�0*

loss�+=pGO�       �	�k�Yc�A�0*

lossԴ�<^�ż       �	!�Yc�A�0*

loss�u�;�G��       �	؞�Yc�A�0*

loss��l<�,��       �	�B�Yc�A�0*

loss��"=1e��       �	���Yc�A�0*

lossHi�=˻�^       �	q��Yc�A�0*

loss�v�<Av       �	�Q�Yc�A�0*

lossx�@=�f�       �	|��Yc�A�0*

loss��{<q�&       �	���Yc�A�0*

lossSz�;�K�P       �	�+�Yc�A�0*

losss�;)��       �	���Yc�A�0*

loss4.�<�D       �	�t�Yc�A�0*

loss�	�<Hڗ�       �	��Yc�A�0*

lossV�7=qˠ       �	[��Yc�A�0*

loss���<�2�L       �	�L�Yc�A�0*

loss��<Ѻ�a       �	d��Yc�A�0*

loss�}V:�{i�       �	$~�Yc�A�0*

loss��><�&K       �	���Yc�A�0*

lossJ��<��(       �	l���Yc�A�0*

loss�V<�[��       �	jN�Yc�A�0*

loss%D=3��       �	���Yc�A�0*

loss
�<$91       �	�{�Yc�A�0*

loss$l/=�!.       �	�F�Yc�A�0*

lossZ9�<�c��       �	)�Yc�A�0*

losstT-=Z*�8       �	C�Yc�A�0*

loss:>�;F[.       �	
��Yc�A�0*

loss��<���&       �	Xo�Yc�A�0*

loss��<Dm��       �	�:��Yc�A�0*

loss�O�<�k-       �	����Yc�A�0*

loss���=5G9�       �	�l��Yc�A�0*

loss���<9��       �	��Yc�A�0*

loss�r=;���       �	}���Yc�A�0*

loss�
=�}��       �	2��Yc�A�0*

loss�]$<�X�       �	���Yc�A�0*

lossrq�<�\ib       �	�m��Yc�A�1*

loss��<�kF       �	���Yc�A�1*

lossq$C=���       �	����Yc�A�1*

loss�T=*��       �	E��Yc�A�1*

lossܓ�=2�&�       �	� ��Yc�A�1*

lossB:�;�;��       �	���Yc�A�1*

lossg�<�BQ       �	74��Yc�A�1*

loss�f=2�{       �	���Yc�A�1*

loss���;��ŵ       �	���Yc�A�1*

loss�I;Ȑa       �	�F��Yc�A�1*

loss�7S;��%T       �	$���Yc�A�1*

loss�A�<�EYC       �	���Yc�A�1*

loss��$<���C       �	h �Yc�A�1*

loss��<T��       �	��Yc�A�1*

loss���;5�ݔ       �	`��Yc�A�1*

lossgp=�t<       �	�L�Yc�A�1*

loss�Cz;�J�       �	��Yc�A�1*

loss�2<S,�       �	���Yc�A�1*

loss7�=�Q%�       �	��Yc�A�1*

loss3�I:լ��       �	p��Yc�A�1*

lossD��;�       �	�S�Yc�A�1*

loss�!<�V��       �	.�Yc�A�1*

loss�F�:Y�\       �	i��Yc�A�1*

loss�ao<�YC       �	^�Yc�A�1*

loss��$=�:?       �	'��Yc�A�1*

lossQH;���       �	��Yc�A�1*

lossݤ�<9-�U       �	�6	�Yc�A�1*

lossD�:=b��       �	�	�Yc�A�1*

loss$%�<����       �	b
�Yc�A�1*

loss%��;�Xj       �	�
�Yc�A�1*

loss��=O��i       �	l��Yc�A�1*

loss��;�;s�       �	Q1�Yc�A�1*

loss�a�=��       �	9��Yc�A�1*

loss�s�<�       �	�i�Yc�A�1*

loss�0)=����       �	��Yc�A�1*

loss��\<)���       �	���Yc�A�1*

loss-t[=+r�       �	`?�Yc�A�1*

lossI��=Z��       �	g��Yc�A�1*

loss̕�<k�i�       �	g�Yc�A�1*

loss頂<Lhl�       �	& �Yc�A�1*

loss��-=*�X�       �	-��Yc�A�1*

loss�%=%��+       �	�1�Yc�A�1*

lossH�m<���&       �	"��Yc�A�1*

lossjY�;ϱC?       �	�[�Yc�A�1*

loss?@�=�8�       �	��Yc�A�1*

loss�;<<xc�       �	<3�Yc�A�1*

lossR�k<�ۢ�       �	���Yc�A�1*

loss�Kz=����       �	׈�Yc�A�1*

loss-��:]�d       �	Pp�Yc�A�1*

loss���<���       �	rO�Yc�A�1*

loss�;!=�n�
       �	�Yc�A�1*

loss�Yu=���       �	:��Yc�A�1*

lossظ;�       �	:u�Yc�A�1*

loss#3�<����       �	1#�Yc�A�1*

loss���=08��       �	X��Yc�A�1*

loss1��<��;       �	x��Yc�A�1*

loss�ff<r�k�       �	s��Yc�A�1*

lossx�<�,�       �	�C�Yc�A�1*

loss�5=|Y��       �	��Yc�A�1*

loss:�<pk�       �	���Yc�A�1*

loss�xY;�ېI       �	�' �Yc�A�1*

loss>�<��I       �	�� �Yc�A�1*

loss��<r�%       �	/�!�Yc�A�1*

loss��=%�a�       �	� "�Yc�A�1*

lossN��<Z���       �	f�"�Yc�A�1*

lossD9�<W�֎       �	Nd#�Yc�A�1*

loss���<��i       �	�$�Yc�A�1*

loss� =6�)k       �	��$�Yc�A�1*

lossr��;��|�       �	�?%�Yc�A�1*

lossJ��;$�       �	��%�Yc�A�1*

loss�w�=a�$�       �	�|&�Yc�A�1*

loss�|�;��       �	�'�Yc�A�1*

loss�~�;����       �	T�'�Yc�A�1*

loss�S<=C       �	�{(�Yc�A�1*

loss��=���       �	�)�Yc�A�1*

losss,9<N�\|       �	�)�Yc�A�1*

loss��<E��       �	3N*�Yc�A�1*

loss�H1=�5       �	O�*�Yc�A�1*

lossZӞ=|�С       �	 �+�Yc�A�1*

lossd�R;+_�{       �	T;,�Yc�A�1*

lossz�<�D]�       �	��,�Yc�A�1*

loss/�;�:��       �	Ӈ-�Yc�A�1*

loss4{<�tRk       �	�f.�Yc�A�1*

loss�&�<��c�       �	�.�Yc�A�1*

loss�=�,�p       �	l�/�Yc�A�1*

loss�7=��'@       �	��0�Yc�A�1*

loss�v<w�Y�       �	�]1�Yc�A�1*

loss'7�<D��       �	P�1�Yc�A�1*

loss:��<h�#Q       �	��2�Yc�A�1*

loss��=�P�       �	�4�Yc�A�1*

loss�C<��5C       �	��4�Yc�A�1*

loss��;��r       �	W�5�Yc�A�1*

loss�3�=�q��       �	86�Yc�A�1*

loss&�=S�p       �	�6�Yc�A�1*

loss%�=�ژ       �	�n7�Yc�A�1*

loss*)=ݻ�t       �	.8�Yc�A�1*

loss[�<q��       �	�8�Yc�A�1*

lossì&;�J4       �	&Q9�Yc�A�1*

loss\�p=����       �	��9�Yc�A�1*

loss�F<w�FK       �	#�:�Yc�A�1*

lossw�?='�       �	�1;�Yc�A�1*

loss\%�;�d�"       �	1�;�Yc�A�1*

lossO^�;(�ԥ       �	~p<�Yc�A�1*

lossĵ6<t�@       �	ʈ=�Yc�A�1*

loss[	�=��V       �	&>�Yc�A�1*

lossW��=LH       �	��>�Yc�A�1*

loss�Re<y��&       �	�y?�Yc�A�1*

loss	˺<�Ä       �	sK@�Yc�A�1*

loss�=i	�/       �	��@�Yc�A�1*

lossw�<z�D3       �	�wA�Yc�A�1*

loss�,�<ԉ��       �	�B�Yc�A�1*

loss�eS=j���       �	�B�Yc�A�1*

loss`��=�m        �	�KC�Yc�A�1*

lossh_�<����       �	��C�Yc�A�1*

loss o=��       �	��D�Yc�A�1*

loss��<o�g�       �	?oE�Yc�A�1*

loss \3=CY       �	�F�Yc�A�1*

loss��= FE�       �	��F�Yc�A�1*

loss��<�>�       �	�CG�Yc�A�1*

loss;~�;hf�;       �	u�G�Yc�A�1*

lossA�<Nh`�       �	Z�H�Yc�A�1*

loss�:�<��       �	�I�Yc�A�1*

lossQ��<�C.       �	s�I�Yc�A�1*

lossE_B<(�       �	g_J�Yc�A�1*

loss�^=8u       �	��J�Yc�A�1*

loss�!<���       �	�K�Yc�A�1*

loss���;�ߥP       �	Z-L�Yc�A�1*

loss_�<R��       �	��L�Yc�A�1*

loss�\=��       �	:XM�Yc�A�2*

loss<9<�9�       �	��M�Yc�A�2*

lossƙf=���       �	uO�Yc�A�2*

loss6_a<Kzh       �	��O�Yc�A�2*

lossdX�;dI       �	�8P�Yc�A�2*

loss��O=�X�I       �	��P�Yc�A�2*

lossA'<�=2�       �	VfQ�Yc�A�2*

loss�ow<Q��a       �	kR�Yc�A�2*

loss��<��fc       �	�S�Yc�A�2*

loss7y<r�Ѵ       �	��S�Yc�A�2*

loss#�=;       �	aOT�Yc�A�2*

loss���=*N�       �	φU�Yc�A�2*

loss�)=�M��       �	�#V�Yc�A�2*

lossFtX<*�       �	6?W�Yc�A�2*

loss鼏<��a�       �	�X�Yc�A�2*

loss{�(=�:w�       �	J�X�Yc�A�2*

loss�P�=����       �	0Y�Yc�A�2*

loss�vj;`��       �	��Y�Yc�A�2*

loss� �;[��       �	��Z�Yc�A�2*

loss;=lS�       �	t�[�Yc�A�2*

lossͩW<���	       �	�.\�Yc�A�2*

loss��;o�D       �	��\�Yc�A�2*

loss��=���       �	�d]�Yc�A�2*

loss=IU;<�rI       �	X�]�Yc�A�2*

loss,cE=��8       �	#�^�Yc�A�2*

lossz�R=`�`z       �	"4_�Yc�A�2*

loss(�F=���       �	��_�Yc�A�2*

losss �;�7�       �	=d`�Yc�A�2*

loss��<]J�"       �	�a�Yc�A�2*

loss��;�!�[       �	�a�Yc�A�2*

loss�I;��o       �	^Jb�Yc�A�2*

lossv9�<����       �	��b�Yc�A�2*

loss*I�<���k       �	&�c�Yc�A�2*

loss��#<`�_       �	�1d�Yc�A�2*

loss��<E9�       �	\�d�Yc�A�2*

lossQf=j���       �	Vce�Yc�A�2*

loss��<<��A       �	�f�Yc�A�2*

lossZ�s<�~$�       �	��f�Yc�A�2*

loss�K�<�R�p       �	�^g�Yc�A�2*

loss|�=<��+�       �	%h�Yc�A�2*

loss�%9<��͟       �	<�h�Yc�A�2*

loss�В=�j��       �	Hi�Yc�A�2*

loss��=EWW       �	�:j�Yc�A�2*

loss��O=�y��       �	��j�Yc�A�2*

loss0A=��>y       �	<ik�Yc�A�2*

loss�=g;��C^       �	Bl�Yc�A�2*

loss��=uL2�       �	��l�Yc�A�2*

loss/e@=%	�n       �	�Fm�Yc�A�2*

loss��<[��n       �	_�m�Yc�A�2*

loss8�m<�S�       �	f�n�Yc�A�2*

loss�P�<z���       �	V,o�Yc�A�2*

loss��#<�lHd       �	!�o�Yc�A�2*

lossm�;��       �	�np�Yc�A�2*

loss��<�*�       �	q�Yc�A�2*

loss��=|u��       �	/�q�Yc�A�2*

loss�
<��
�       �	2Xr�Yc�A�2*

lossCY�<Z�%       �	I�r�Yc�A�2*

loss��;km�       �	i�s�Yc�A�2*

loss��<�Ԏ       �	�st�Yc�A�2*

loss̴:<�M@�       �	�u�Yc�A�2*

loss��7<���       �	Q�u�Yc�A�2*

loss�H�<H�s�       �	�Pv�Yc�A�2*

loss���<&���       �	:�v�Yc�A�2*

loss���;򡨹       �	�x�Yc�A�2*

loss�u>��K�       �	9�x�Yc�A�2*

loss�և;[       �	�Ky�Yc�A�2*

loss�&C=Rj�       �	q�y�Yc�A�2*

loss�x;<s�B7       �	4�z�Yc�A�2*

loss�Ǌ<���       �	�V{�Yc�A�2*

lossֱ�;XE�       �	��{�Yc�A�2*

loss�d�;�|��       �	��|�Yc�A�2*

loss&��=Y���       �	"5}�Yc�A�2*

loss�`<�v��       �	��}�Yc�A�2*

losssy<N�       �	{k~�Yc�A�2*

loss::5=�}g�       �	��Yc�A�2*

loss�%<�k       �	���Yc�A�2*

loss)�1;7T=�       �	�S��Yc�A�2*

loss4<�-�       �	O끹Yc�A�2*

loss�(<��       �	ߊ��Yc�A�2*

loss�<�=D       �	! ��Yc�A�2*

lossn-Q=�~�1       �	����Yc�A�2*

lossp��<�{�       �	X��Yc�A�2*

loss�"w=�s��       �	=Yc�A�2*

lossCǀ;q�&       �	����Yc�A�2*

loss3��<�W�r       �	0J��Yc�A�2*

loss1�=����       �	�܆�Yc�A�2*

loss �F<�I8�       �	����Yc�A�2*

loss��%<�Cv       �	 %��Yc�A�2*

loss
�M<38/�       �	Ɉ�Yc�A�2*

loss� �<�C��       �	�k��Yc�A�2*

loss�=�`       �	~ ��Yc�A�2*

loss���;Z/c�       �	>���Yc�A�2*

loss�u<���       �	-��Yc�A�2*

lossͻ�;ƿZU       �	Gʋ�Yc�A�2*

lossMH�<ɫ{r       �	Ra��Yc�A�2*

loss�s&<��M       �	�-��Yc�A�2*

loss:�=Ԫ       �	�ȍ�Yc�A�2*

loss�+�<�fcH       �	e��Yc�A�2*

loss�={;��/{       �	1?��Yc�A�2*

loss��<ӔE�       �	,Տ�Yc�A�2*

lossW�.=�G�       �	�k��Yc�A�2*

lossC̎=C�n�       �	���Yc�A�2*

lossA�[;<2�       �	z���Yc�A�2*

lossA<F<q�0�       �	C=��Yc�A�2*

loss��U<�ߟA       �	�Ӓ�Yc�A�2*

loss3��<����       �	ge��Yc�A�2*

loss��b;�b��       �	����Yc�A�2*

loss�"8=�a"b       �	�.��Yc�A�2*

loss��p;2�Q�       �	oӕ�Yc�A�2*

loss��>=�L�       �	�}��Yc�A�2*

lossV��<1v-�       �	j��Yc�A�2*

lossn��<s5)       �	����Yc�A�2*

loss��=�B)       �	�T��Yc�A�2*

lossw��;�(~�       �	����Yc�A�2*

loss!�0=�Ul       �	<���Yc�A�2*

loss�@�<A�       �	�K��Yc�A�2*

loss�<�"XG       �	F��Yc�A�2*

loss�<�}ӏ       �	g���Yc�A�2*

lossh>=�;�        �	�P��Yc�A�2*

loss�p;�i~5       �	뜹Yc�A�2*

loss���<|�h       �	臝�Yc�A�2*

lossΔ�;xx^�       �	EI��Yc�A�2*

loss�� =�X�       �	�O��Yc�A�2*

loss��;��       �	O培Yc�A�2*

lossZ��;�NX       �	�w��Yc�A�2*

loss��d=KQ��       �	���Yc�A�2*

loss@8�;ļ��       �	h���Yc�A�2*

loss�(=l���       �	�A��Yc�A�2*

losslb�<2s       �	�5��Yc�A�3*

loss�+=M�]       �	�ɣ�Yc�A�3*

lossD) ;Ÿ=�       �	�^��Yc�A�3*

loss�~�;�t�V       �	9F��Yc�A�3*

loss�=!HWL       �	#ܥ�Yc�A�3*

loss+�
;�9�u       �	�v��Yc�A�3*

lossA��9< ]       �	���Yc�A�3*

loss��<I��       �	����Yc�A�3*

loss�{<t��~       �	�;��Yc�A�3*

loss�5�;X�i       �	�Ҩ�Yc�A�3*

loss���;#��       �	Dn��Yc�A�3*

lossSk�9��V       �	p��Yc�A�3*

loss�<�-�7       �	~���Yc�A�3*

loss��_; �Y�       �	FA��Yc�A�3*

lossDY�9�T       �	nܫ�Yc�A�3*

loss�,	:j�+       �	�p��Yc�A�3*

loss咙;!��       �	y��Yc�A�3*

lossm/�<��
#       �	R���Yc�A�3*

loss��T<	���       �	tB��Yc�A�3*

loss(��9����       �	V׮�Yc�A�3*

loss���<� ��       �	�m��Yc�A�3*

lossa�q=Y6�C       �	��Yc�A�3*

loss!��:���       �	���Yc�A�3*

loss��>���       �	����Yc�A�3*

loss+#=;M�-       �	�;��Yc�A�3*

loss�=[��       �	�Yc�A�3*

loss�G�<��z       �	J���Yc�A�3*

lossD��<b��       �	�O��Yc�A�3*

lossƆ<?       �	\��Yc�A�3*

loss�'=.n�       �	
L��Yc�A�3*

lossJ+=g;:�       �	V�Yc�A�3*

loss$�O=�G�       �	F��Yc�A�3*

loss.�;�O       �	���Yc�A�3*

lossֈ<�$��       �	�f��Yc�A�3*

loss���;��/�       �	~��Yc�A�3*

loss��=�ʒ       �	-_��Yc�A�3*

loss���<��       �	����Yc�A�3*

loss�i-=�<>       �	F���Yc�A�3*

loss��<���       �	�+��Yc�A�3*

lossj5�<y�G�       �	'���Yc�A�3*

loss�=VQa�       �	�u��Yc�A�3*

loss��<�J˕       �	V��Yc�A�3*

loss,��;� �       �	
���Yc�A�3*

loss1>M<_f�       �	�:��Yc�A�3*

loss�.�<��H       �	����Yc�A�3*

loss���:��N       �	\v¹Yc�A�3*

loss��<��3       �	�ùYc�A�3*

loss�l�<�	`       �	�ĹYc�A�3*

loss;>/<,�g       �	��ĹYc�A�3*

loss��<00�       �	�SŹYc�A�3*

loss[��=�-�B       �	��ŹYc�A�3*

lossE[�=���[       �	}�ƹYc�A�3*

loss�V=�\��       �	�*ǹYc�A�3*

loss}��;����       �	U�ǹYc�A�3*

loss#��;��       �	qYȹYc�A�3*

lossd�d;�%�       �	<ɹYc�A�3*

loss�D{<�NҶ       �	�ɹYc�A�3*

loss�X=��_�       �	�ZʹYc�A�3*

loss<�R<�whl       �	8�ʹYc�A�3*

lossZz�<���       �	?�˹Yc�A�3*

loss�g�<��Դ       �	�&̹Yc�A�3*

loss#�,<%]��       �	��̹Yc�A�3*

loss�J�;��A'       �	�f͹Yc�A�3*

lossV�<�2�P       �	*ιYc�A�3*

loss]�g<;.�       �	r�ιYc�A�3*

lossHR�;����       �	�YϹYc�A�3*

loss
{�<ԗ|+       �	��ϹYc�A�3*

loss}�=�!��       �	�йYc�A�3*

loss�!=x��5       �	0ѹYc�A�3*

loss�A�:�^�`       �	e�ѹYc�A�3*

loss��<xL��       �	�mҹYc�A�3*

loss,��:�x�       �	�ӹYc�A�3*

loss��<����       �	#�ӹYc�A�3*

loss�!�<       �	��Yc�A�3*

loss3�<�n�3       �	���Yc�A�3*

loss��r<9���       �	/�Yc�A�3*

loss��,<?�       �	���Yc�A�3*

loss���;[$�       �	�'��Yc�A�3*

loss���=I��       �	e���Yc�A�3*

loss@�3<G x�       �	���Yc�A�3*

loss��K=�jm�       �	�@�Yc�A�3*

loss$F�<���       �	X��Yc�A�3*

loss��<St)       �	z��Yc�A�3*

loss��<3�A       �	� �Yc�A�3*

loss�"==�	�B       �	���Yc�A�3*

losst<�x       �	F�Yc�A�3*

loss��)<�       �	��Yc�A�3*

loss�}�<��c       �	ϟ�Yc�A�3*

loss �n<��L       �	�7��Yc�A�3*

loss��:�	{       �	e ��Yc�A�3*

loss��<�=�$       �	V���Yc�A�3*

lossl:A<��       �	H4��Yc�A�3*

loss݌9=y\       �	.���Yc�A�3*

loss��<=�
�f       �	͏��Yc�A�3*

lossi�=>�       �	�'��Yc�A�3*

loss>=r���       �	����Yc�A�3*

loss��s=�6e�       �	����Yc�A�3*

lossrlT<5��       �	�#��Yc�A�3*

loss�G�;��5#       �	����Yc�A�3*

loss��=��ɢ       �	Y��Yc�A�3*

loss�C;Y���       �	M���Yc�A�3*

loss�I�<m�=�       �	X���Yc�A�3*

loss��<��:       �	���Yc�A�3*

loss���;���1       �	����Yc�A�3*

loss7�=�[��       �	L��Yc�A�3*

loss�Ҙ;x�p       �	8H��Yc�A�3*

loss��<^W �       �	����Yc�A�3*

loss��<�W�       �	�z �Yc�A�3*

losst�<a��y       �	��Yc�A�3*

loss�?;/S�       �	7��Yc�A�3*

loss���<�o       �	Fa�Yc�A�3*

loss�b9=A�H       �	�j�Yc�A�3*

lossf��<9�       �	6�Yc�A�3*

loss���;��e\       �	Ӡ�Yc�A�3*

loss<���       �	P:�Yc�A�3*

loss	R|;F��       �	���Yc�A�3*

loss�׊<
�6�       �	�z�Yc�A�3*

loss���;��MC       �	��Yc�A�3*

loss�7!=�,e�       �	&��Yc�A�3*

losso�O<�X.�       �	�@�Yc�A�3*

loss\�<̑��       �	���Yc�A�3*

loss;�S<
.4�       �	��	�Yc�A�3*

lossejS;���       �	q�
�Yc�A�3*

lossCʑ<^��       �	�)�Yc�A�3*

loss3?f=3k~       �	;��Yc�A�3*

loss�zD;@�&�       �	_�Yc�A�3*

loss�p�=�)��       �	L�Yc�A�3*

loss�7�<4l       �	���Yc�A�3*

loss��5;G!�K       �	@Q�Yc�A�4*

loss�Ǎ;� k�       �	��Yc�A�4*

loss�A�;)=��       �	Ĕ�Yc�A�4*

lossO<��Q�       �	gD�Yc�A�4*

loss�c<��*       �	/��Yc�A�4*

lossWw�=�b       �	Þ�Yc�A�4*

loss��;qf%�       �	�>�Yc�A�4*

loss�L?;����       �	���Yc�A�4*

loss�v�<�R�3       �	�z�Yc�A�4*

loss7��<�Z       �	a�Yc�A�4*

lossB=E�[�       �	��Yc�A�4*

loss�E@=*V�O       �	fN�Yc�A�4*

loss���<_g�F       �	���Yc�A�4*

loss�\�;����       �	Z~�Yc�A�4*

loss���<l�       �	_&�Yc�A�4*

loss��<TG�t       �	��Yc�A�4*

loss��<bh7�       �	{��Yc�A�4*

loss7�<�T'G       �	Փ�Yc�A�4*

loss�9=GfZ`       �	@1�Yc�A�4*

lossi��<�\�       �	���Yc�A�4*

loss���;w�tM       �	F}�Yc�A�4*

lossh=��p       �	��Yc�A�4*

loss�=<�Nc       �	��Yc�A�4*

loss2�H;�/�*       �	���Yc�A�4*

lossi|;_���       �	�b�Yc�A�4*

loss1r(<W�	       �	���Yc�A�4*

loss\�b<���       �	r1 �Yc�A�4*

losst(<�u�       �	G� �Yc�A�4*

loss`�b<n���       �	�^!�Yc�A�4*

loss�6x=RW�]       �	�!�Yc�A�4*

loss.;�nb       �	u�"�Yc�A�4*

losslfM=���f       �	%=#�Yc�A�4*

loss�<<l�\       �	��#�Yc�A�4*

loss�7B<E�       �	ђ$�Yc�A�4*

loss��<���       �	�+%�Yc�A�4*

loss�Ky<8'8�       �	��%�Yc�A�4*

loss��<�,+       �	�q&�Yc�A�4*

lossI��;T5/#       �	'�Yc�A�4*

loss8NP;q�N       �	��'�Yc�A�4*

loss�~�;�4       �	wf(�Yc�A�4*

lossF=�&o�       �	*)�Yc�A�4*

loss_��<M��       �	�)�Yc�A�4*

loss���;���       �	<*�Yc�A�4*

loss�pA<r���       �	��*�Yc�A�4*

loss�g�:
��J       �	"p+�Yc�A�4*

loss�8<+�J       �	�,�Yc�A�4*

loss�{�<�Y�       �	�,�Yc�A�4*

loss���<��1�       �	�_-�Yc�A�4*

loss���;�r;       �	X�-�Yc�A�4*

loss�a=��       �	s�.�Yc�A�4*

loss���;���       �	�:/�Yc�A�4*

loss$�;H���       �	��/�Yc�A�4*

loss�;�;��        �	U�0�Yc�A�4*

lossF�C<�k�X       �	'1�Yc�A�4*

lossw�;�d�1       �	�1�Yc�A�4*

loss
��<E��       �	w�2�Yc�A�4*

lossav�<�h��       �	d 3�Yc�A�4*

loss�=��S�       �	��3�Yc�A�4*

loss���:F�K       �	�^4�Yc�A�4*

loss�ǽ;���       �	W'5�Yc�A�4*

loss�~-<?{-D       �	��5�Yc�A�4*

lossFѠ<H~%�       �	�6�Yc�A�4*

loss�c�<�R�c       �	�~7�Yc�A�4*

lossh�v;���       �	��8�Yc�A�4*

loss.��=Bb�       �	�o9�Yc�A�4*

loss$�0;��       �	w:�Yc�A�4*

loss�K�<o��       �	�:�Yc�A�4*

loss�x=�5�       �	�o;�Yc�A�4*

loss�{X;}��       �	�<�Yc�A�4*

loss(�M;�M�i       �	t�<�Yc�A�4*

loss�=��       �	�R=�Yc�A�4*

lossn��;�F�}       �	��=�Yc�A�4*

loss���;L5       �	?�>�Yc�A�4*

loss1�=|=��       �	�,?�Yc�A�4*

losss��;��S6       �	��?�Yc�A�4*

lossk;ł�7       �	)]@�Yc�A�4*

loss̀S<�9+�       �	�@�Yc�A�4*

loss���<N�,�       �	��A�Yc�A�4*

lossZ(>;��F�       �	S=B�Yc�A�4*

lossb�<��       �	L�B�Yc�A�4*

loss�đ<�0       �	0�C�Yc�A�4*

lossOA+;�9$L       �	�D�Yc�A�4*

loss��;s�'       �	��D�Yc�A�4*

lossv�<��#�       �	vQE�Yc�A�4*

loss�u-;rՏ�       �	F�E�Yc�A�4*

loss!��;l�@�       �	��F�Yc�A�4*

loss�<���a       �	6 G�Yc�A�4*

loss��d=z�r       �	�G�Yc�A�4*

loss�(�<����       �	XH�Yc�A�4*

loss,E=��       �	9�H�Yc�A�4*

lossqX�<�]�       �	�J�Yc�A�4*

loss�@;Z�47       �	p�J�Yc�A�4*

losscf<qv+       �	kIK�Yc�A�4*

loss�zR:�JA       �	�K�Yc�A�4*

loss��<�<=
       �	SzL�Yc�A�4*

lossZ��<S֯U       �	ZM�Yc�A�4*

lossas�<����       �	��M�Yc�A�4*

lossRky<�f<8       �	@QN�Yc�A�4*

lossJ[-<C�       �	W�N�Yc�A�4*

lossϝ=8�       �	]�O�Yc�A�4*

loss{*�:����       �	W#P�Yc�A�4*

lossq� <Fx�       �	��P�Yc�A�4*

loss�"q=�z�'       �	�iQ�Yc�A�4*

loss�h'=O���       �	�R�Yc�A�4*

loss�5<�e�q       �	O�R�Yc�A�4*

lossT"}=X]�I       �	=dS�Yc�A�4*

loss�a�<��O=       �	��S�Yc�A�4*

loss���;3R�U       �	<�T�Yc�A�4*

lossWI:;�I       �	2�U�Yc�A�4*

lossa7=��t       �	.V�Yc�A�4*

loss2%<5�       �	��V�Yc�A�4*

loss��:���       �	�gW�Yc�A�4*

loss=|/<���       �	e�W�Yc�A�4*

loss��<�*}-       �	�X�Yc�A�4*

loss��8=�6�u       �	�MY�Yc�A�4*

loss7Y�<N�!�       �	d�Y�Yc�A�4*

lossû�<TZjv       �	w�Z�Yc�A�4*

loss���<f�N       �	�I[�Yc�A�4*

loss�u�<-���       �	C�[�Yc�A�4*

lossF��;В�e       �	o�\�Yc�A�4*

loss��<��       �	y"]�Yc�A�4*

loss�=�F��       �	߿]�Yc�A�4*

lossI$=L�ȭ       �	�_^�Yc�A�4*

loss�^�;@\h�       �	Q�^�Yc�A�4*

loss��|=��,       �	�_�Yc�A�4*

lossĝ�9+f9�       �	�3`�Yc�A�4*

lossm:�<]-�m       �	��`�Yc�A�4*

loss��<�;u       �	�pa�Yc�A�4*

loss�=��)�       �	+b�Yc�A�5*

loss2	�;j��B       �	�b�Yc�A�5*

loss��:@F�       �	Tc�Yc�A�5*

loss��9<(�Ɖ       �	A�c�Yc�A�5*

lossv _;��f
       �	]�d�Yc�A�5*

lossD�6<�Uk       �	ze�Yc�A�5*

loss��7=�^�       �	�e�Yc�A�5*

lossQ$�<T+`        �	�Rf�Yc�A�5*

loss��<�(�       �	��f�Yc�A�5*

loss�=��R�       �	��g�Yc�A�5*

loss�N<~h�       �	�,h�Yc�A�5*

loss��;ح�       �	��h�Yc�A�5*

loss�3<V���       �	2ti�Yc�A�5*

loss��%=�'��       �	j�Yc�A�5*

loss�^<d6I       �	>�j�Yc�A�5*

loss��<�+*       �	fMk�Yc�A�5*

lossfĩ:=�)       �	��k�Yc�A�5*

loss4/=���       �	��l�Yc�A�5*

lossJ��;��;'       �	�.m�Yc�A�5*

loss��;I�I       �	t�m�Yc�A�5*

loss���<��2       �	�vn�Yc�A�5*

loss�`�;�e�#       �	�o�Yc�A�5*

loss濉;A���       �	��o�Yc�A�5*

loss�M=z�f�       �	�Yp�Yc�A�5*

loss��Q;}��       �	?�p�Yc�A�5*

loss <��       �	�q�Yc�A�5*

loss�y�=��+N       �	�;r�Yc�A�5*

lossa�A<��e       �	��r�Yc�A�5*

loss�?�;���       �	�s�Yc�A�5*

loss�ʩ:��|       �	��t�Yc�A�5*

loss�O;ἀC       �	=v�Yc�A�5*

loss�=uv�       �	ܡv�Yc�A�5*

loss-ɯ<�U6       �	�@w�Yc�A�5*

loss�e>pd�f       �	��w�Yc�A�5*

lossj�=H؝c       �	��x�Yc�A�5*

loss��=�$v�       �	 y�Yc�A�5*

loss�F<{���       �	y�y�Yc�A�5*

loss��<}���       �		Sz�Yc�A�5*

loss��5;; �l       �	1�z�Yc�A�5*

lossFc�<�T�       �	j�{�Yc�A�5*

loss,�<%�Z�       �	}!|�Yc�A�5*

loss�=<Ҧ;�       �	�|�Yc�A�5*

loss�<0���       �	��}�Yc�A�5*

lossj��<L�(�       �	�~�Yc�A�5*

loss�" <#l�       �	W�~�Yc�A�5*

loss��;�i��       �	UN�Yc�A�5*

loss��2=��Z�       �	���Yc�A�5*

lossAL;;��}       �	_���Yc�A�5*

loss��0;��C       �	�j��Yc�A�5*

lossA0=�.�       �	���Yc�A�5*

loss&�;
�.~       �	g���Yc�A�5*

lossb4<^l�K       �	O��Yc�A�5*

loss�DH<1���       �	���Yc�A�5*

loss~C<�ݙ�       �	ꔄ�Yc�A�5*

loss�@�<�*�       �	 7��Yc�A�5*

lossHH�<%�9       �	f܅�Yc�A�5*

loss7R2<����       �	�~��Yc�A�5*

lossW�R<��       �	*��Yc�A�5*

loss�$<�%�Z       �	����Yc�A�5*

loss��;ȡgt       �	�]��Yc�A�5*

loss��<�3�       �	h��Yc�A�5*

loss�U�<��.�       �	/���Yc�A�5*

lossZ?<���q       �	�=��Yc�A�5*

loss#GZ<��^�       �	8ڊ�Yc�A�5*

loss&==�Q"       �	�s��Yc�A�5*

lossJ!)=�� �       �	�K��Yc�A�5*

lossi�V<��       �	N�Yc�A�5*

loss�<^,        �	ގ��Yc�A�5*

loss�J2=��       �	,+��Yc�A�5*

loss_s�;3�       �	�Ȏ�Yc�A�5*

lossf��9M��>       �	p]��Yc�A�5*

loss��3;�i�       �	��Yc�A�5*

loss���:9��       �	����Yc�A�5*

loss���;)-/b       �	�4��Yc�A�5*

lossx];-��       �	�ʑ�Yc�A�5*

loss �;�R�       �	�ݒ�Yc�A�5*

lossn˔;4S��       �	yv��Yc�A�5*

loss���:1o�       �	���Yc�A�5*

loss��<O�և       �	����Yc�A�5*

lossI==�K�!       �	�=��Yc�A�5*

loss�=��f�       �	>蕺Yc�A�5*

lossS��<?�       �	���Yc�A�5*

loss�~<��9�       �	�-��Yc�A�5*

lossΰ�=]%�1       �	Ș�Yc�A�5*

losse&<e^<'       �	�d��Yc�A�5*

lossR��<�ٙ       �	����Yc�A�5*

loss#�h=E��       �	���Yc�A�5*

loss/�<<Y�V�       �	.8��Yc�A�5*

loss?D<�\�       �	H1��Yc�A�5*

loss:�<�TX�       �	�Ȝ�Yc�A�5*

loss�l�;`:^�       �	i��Yc�A�5*

loss�=�ԩ
       �	� ��Yc�A�5*

losstL�<����       �	񝞺Yc�A�5*

loss��;+�"H       �	�v��Yc�A�5*

loss?�	<�R�F       �	���Yc�A�5*

lossTG�;f<�       �	����Yc�A�5*

loss�ם=��ո       �	EK��Yc�A�5*

loss4�;�d��       �	9Yc�A�5*

loss�#<hX�       �	����Yc�A�5*

lossnSG<!}f       �	�.��Yc�A�5*

loss@{�;v��       �	Iգ�Yc�A�5*

loss/9n<�ξ       �	�t��Yc�A�5*

loss�6�<�S�c       �	���Yc�A�5*

loss]D�<���       �	�Yc�A�5*

loss���;m�,_       �	����Yc�A�5*

loss�a�;�2        �	�/��Yc�A�5*

lossW�;5��       �	�ӧ�Yc�A�5*

loss�#�;Qt��       �	�z��Yc�A�5*

loss�DC=���%       �	v��Yc�A�5*

loss*�<��o�       �	b���Yc�A�5*

loss�G<��H�       �	�g��Yc�A�5*

loss�Z*=<B�(       �	6��Yc�A�5*

lossN�/<Q�%�       �	����Yc�A�5*

loss|��;p�tt       �	~5��Yc�A�5*

loss�^�<]�m?       �	�Ь�Yc�A�5*

loss�=�=�?[�       �	�n��Yc�A�5*

loss��%=�ʦX       �	�	��Yc�A�5*

losss��;	P�z       �	����Yc�A�5*

loss9P�<�ݧ�       �	�U��Yc�A�5*

loss;��;�Q�a       �	 '��Yc�A�5*

loss�؍==�x       �	�ư�Yc�A�5*

loss=�{��       �	|_��Yc�A�5*

loss,��=���y       �	���Yc�A�5*

loss�Z�<�!F�       �	����Yc�A�5*

lossR!0=��       �	�I��Yc�A�5*

loss���;Tŏ�       �	~��Yc�A�5*

loss�� =J1V       �	$��Yc�A�5*

loss��<GU�C       �	���Yc�A�5*

loss��H<���       �	���Yc�A�6*

loss_�=d���       �	����Yc�A�6*

loss�I�<���Q       �	�U��Yc�A�6*

lossH�U=/|Z       �	N��Yc�A�6*

lossc5<��\A       �	1칺Yc�A�6*

loss�,�<3�M�       �	Q���Yc�A�6*

lossz��<4o-       �	2��Yc�A�6*

lossj��<����       �	����Yc�A�6*

loss�J�=]V�&       �	�\��Yc�A�6*

loss��<4n]�       �	��Yc�A�6*

loss�Z�=����       �	����Yc�A�6*

loss�� =��       �	l#��Yc�A�6*

lossT�k;USgI       �	���Yc�A�6*

loss�P=��Gx       �	OY��Yc�A�6*

lossq�+=��       �	��Yc�A�6*

lossIË;�r��       �	N���Yc�A�6*

loss��;�       �	U/��Yc�A�6*

lossO|B<m��       �	�ºYc�A�6*

loss�<j�       �	P�ºYc�A�6*

loss��[<�d��       �	u=úYc�A�6*

loss�=��       �	��úYc�A�6*

lossڂb<!&	�       �	�nĺYc�A�6*

loss���<ԋ�       �	�źYc�A�6*

lossT��=L!�X       �	�źYc�A�6*

loss���;	���       �	�,ƺYc�A�6*

loss�7�<��0�       �	��ƺYc�A�6*

lossa�;�)�       �	�cǺYc�A�6*

loss=�<��d�       �	�ǺYc�A�6*

loss;�n:��4       �	Z�ȺYc�A�6*

loss��w;�ez       �	�4ɺYc�A�6*

lossj�k<�{�       �	��ɺYc�A�6*

loss]* =�z��       �	�oʺYc�A�6*

loss-��<;�       �	�˺Yc�A�6*

loss �=�;-�       �		�˺Yc�A�6*

loss1��=�v�       �	-?̺Yc�A�6*

lossa�:\�fq       �	+�̺Yc�A�6*

loss!i<���t       �	�tͺYc�A�6*

loss
3a<��qg       �	�κYc�A�6*

lossa�r=�fI       �	'�κYc�A�6*

loss�6#<�&c       �	i6ϺYc�A�6*

losslAg;Y:�       �	�ϺYc�A�6*

loss7�s=�x%�       �	+jкYc�A�6*

loss$/�=����       �	�
ѺYc�A�6*

loss��<�X�0       �	��ѺYc�A�6*

lossE��;:1�       �	0JҺYc�A�6*

losse�;�?       �	��ҺYc�A�6*

loss�+<˒��       �	��ӺYc�A�6*

loss���;�>�       �	qWԺYc�A�6*

loss���;���       �	9պYc�A�6*

lossC�D<@��       �	\ZֺYc�A�6*

loss�Z=�f�G       �	�#׺Yc�A�6*

loss1��;JY|T       �	��׺Yc�A�6*

loss�e|=�ո�       �	�rغYc�A�6*

loss"%=y-��       �	�ٺYc�A�6*

loss��9<��R       �	]4ںYc�A�6*

losso�Q<x�Ȼ       �	5�ںYc�A�6*

loss�-;��
@       �	�iۺYc�A�6*

loss�m�;k��h       �	ܺYc�A�6*

losszR�<Q���       �	-�ܺYc�A�6*

loss� #=�٩�       �	i�ݺYc�A�6*

loss-B@=���G       �	SX޺Yc�A�6*

loss=�<7�R       �	`�ߺYc�A�6*

loss�d%;��ķ       �	���Yc�A�6*

lossU�<?���       �	�/�Yc�A�6*

loss���<�eC       �	g��Yc�A�6*

loss�0S;.�I#       �	c�Yc�A�6*

loss��<;4Lv       �	�Yc�A�6*

loss֯�< o�       �	ެ�Yc�A�6*

loss�8=8�       �	�E�Yc�A�6*

loss��0<����       �	3��Yc�A�6*

loss:��<�)sJ       �	Ts�Yc�A�6*

loss�Yk=��~       �	��Yc�A�6*

loss���;�o�c       �	���Yc�A�6*

loss���:����       �	7T�Yc�A�6*

lossí�=���U       �	t��Yc�A�6*

loss��M<���       �	|��Yc�A�6*

loss,/<X�w`       �	U�Yc�A�6*

loss���:�*       �	���Yc�A�6*

loss��;���       �	?�Yc�A�6*

loss�~�<�"S�       �	���Yc�A�6*

loss��<]��       �	x�Yc�A�6*

loss)U=ҟ�       �	8�Yc�A�6*

loss��B<�gx       �	 ��Yc�A�6*

lossȽv<�b�U       �	�V��Yc�A�6*

loss��K=��q�       �	[���Yc�A�6*

loss�]�:�pR       �	���Yc�A�6*

loss��=�Dk�       �	(��Yc�A�6*

loss��;�E�        �	���Yc�A�6*

loss(�'=]��       �	��Yc�A�6*

loss;�<���       �	D��Yc�A�6*

loss�%=>��       �	W�Yc�A�6*

loss([�<S�.�       �	x(�Yc�A�6*

lossi�;EK�:       �	���Yc�A�6*

loss��p=�~v       �	�1��Yc�A�6*

loss��;� �R       �	����Yc�A�6*

loss_>,=z(c)       �	؛��Yc�A�6*

loss%�{=�j�       �	�4��Yc�A�6*

loss11=jZ       �	Z���Yc�A�6*

losss�<E*�       �	uu��Yc�A�6*

lossaj<�Vq       �	u!��Yc�A�6*

loss=��<����       �	����Yc�A�6*

loss<.�<�BL       �	����Yc�A�6*

loss{�M=�Mh1       �	�+��Yc�A�6*

loss��T=	�Va       �	����Yc�A�6*

loss
^=h%3n       �	Ot��Yc�A�6*

loss�:;���$       �	���Yc�A�6*

loss�<GϹu       �	����Yc�A�6*

loss�f<�1Y       �	�`��Yc�A�6*

lossj�;��       �	����Yc�A�6*

loss���<�z��       �	}���Yc�A�6*

loss���=��rg       �	�2 �Yc�A�6*

loss���;��       �	�� �Yc�A�6*

loss0= B��       �	�c�Yc�A�6*

loss�?�<A;       �	�;�Yc�A�6*

loss-��=_�T       �	���Yc�A�6*

lossŧ�;�fڕ       �	~�Yc�A�6*

lossc�=W"\       �	�c�Yc�A�6*

loss��;3�i7       �	���Yc�A�6*

lossh�Q=�׸       �	��Yc�A�6*

loss�`(=[���       �	$)�Yc�A�6*

loss'Y=,F5@       �	[��Yc�A�6*

loss��<��PK       �	�|�Yc�A�6*

loss-��<S/F       �	��Yc�A�6*

lossX�<���       �	���Yc�A�6*

loss�6�<�r��       �	|a	�Yc�A�6*

loss|��<Xdz       �	O
�Yc�A�6*

loss=ps;hs�j       �	j�
�Yc�A�6*

loss�==��,V       �	G�Yc�A�6*

loss�q=$��       �	)��Yc�A�7*

loss�s�=t�I       �	��Yc�A�7*

loss���=��i       �	e��Yc�A�7*

loss4]*=��-�       �	'�Yc�A�7*

loss�?�=@BP       �	g��Yc�A�7*

loss�R6;����       �	�r�Yc�A�7*

loss�<��SW       �	�Yc�A�7*

loss�}�=�f�1       �	n��Yc�A�7*

lossP-<�V       �	�E�Yc�A�7*

lossJX<�k��       �	���Yc�A�7*

loss�c�<:C=       �	���Yc�A�7*

loss!�f<��       �	c%�Yc�A�7*

lossē�;���3       �	���Yc�A�7*

lossx<�[��       �	�g�Yc�A�7*

loss3�<YτF       �	�Yc�A�7*

loss*&]<��H       �	��Yc�A�7*

loss�A4=�Dը       �	1]�Yc�A�7*

loss�s�;蕬�       �	���Yc�A�7*

losse�\<֫��       �	���Yc�A�7*

loss ��<_       �	�*�Yc�A�7*

loss�`H<�=��       �	���Yc�A�7*

loss>�<Jt��       �	�\�Yc�A�7*

loss�}%<k'��       �	���Yc�A�7*

loss}��;/e,       �	���Yc�A�7*

loss=�=�%�       �	��Yc�A�7*

loss�d�<�]�l       �	�&�Yc�A�7*

lossm�)=:s&       �	���Yc�A�7*

loss���<���       �	�^�Yc�A�7*

loss5�;oW�       �	���Yc�A�7*

loss��<�^q�       �	Ƌ�Yc�A�7*

loss�3�<�ݴ�       �	�(�Yc�A�7*

loss���=�X�       �	j��Yc�A�7*

loss�V)=��*       �	�Y �Yc�A�7*

loss c�:��i       �	�� �Yc�A�7*

loss���<��9       �	�!�Yc�A�7*

loss��<<��       �	&"�Yc�A�7*

loss��<A?m�       �	��"�Yc�A�7*

loss���:���       �	TW#�Yc�A�7*

loss�<"r^E       �	��#�Yc�A�7*

loss�5�<u�8       �	Y�$�Yc�A�7*

lossw�=漦�       �	P%�Yc�A�7*

loss#>={�݋       �	��%�Yc�A�7*

loss5�<K�%       �	�I&�Yc�A�7*

loss!��;x�ac       �	L�&�Yc�A�7*

loss�	C<�h��       �	��'�Yc�A�7*

loss�}�<g�u       �	�8(�Yc�A�7*

loss�kM<�8�       �	��(�Yc�A�7*

loss��}<li*       �	bg)�Yc�A�7*

loss\�=���,       �	*�Yc�A�7*

lossH��;Ǆ*k       �	�*�Yc�A�7*

lossRE<<D+�U       �	�D+�Yc�A�7*

loss�͉<�/��       �	��+�Yc�A�7*

loss�K�;�9�       �	Q�,�Yc�A�7*

lossF�=�|+o       �	B&-�Yc�A�7*

lossW�<�QS       �	��-�Yc�A�7*

loss��<����       �	�f.�Yc�A�7*

lossa��;#��       �	O/�Yc�A�7*

loss��=!R��       �	̷/�Yc�A�7*

loss�i!;o8	       �	�X0�Yc�A�7*

lossD��=�-�i       �	��0�Yc�A�7*

loss���;US��       �	�1�Yc�A�7*

loss�ߵ<�K��       �	S=2�Yc�A�7*

loss]ſ:_���       �	N�2�Yc�A�7*

loss	�j=u6�       �	�s3�Yc�A�7*

loss�l�<���       �	M4�Yc�A�7*

lossO=�=+��       �	��4�Yc�A�7*

loss�A�;/I.w       �	_
6�Yc�A�7*

loss�*�<P �       �	��6�Yc�A�7*

loss�p�=��       �	�Z7�Yc�A�7*

lossS��<h}�       �	�8�Yc�A�7*

lossc1<E���       �	Q�8�Yc�A�7*

lossE�<�p�P       �	Rb9�Yc�A�7*

loss���<ӓ��       �	�
:�Yc�A�7*

lossn��;�sJ       �	r�:�Yc�A�7*

loss�ٽ<ʩ�       �	�=;�Yc�A�7*

lossxE�;5�0       �	�;�Yc�A�7*

lossZWw=xJ��       �	Q�<�Yc�A�7*

lossi��<��n#       �	�]=�Yc�A�7*

loss�q�;<t+*       �	��=�Yc�A�7*

loss�2}<vC^:       �	��>�Yc�A�7*

loss��<�m       �	�*?�Yc�A�7*

loss���;�·       �	��?�Yc�A�7*

lossY�<ș	       �	b@�Yc�A�7*

loss�*<;5�U       �	��@�Yc�A�7*

loss��<C�B       �	�A�Yc�A�7*

loss�
_<��W�       �	�;B�Yc�A�7*

loss��;�         �	E�B�Yc�A�7*

loss6h�:�g       �	qC�Yc�A�7*

loss�2#<�߸k       �	�D�Yc�A�7*

loss�U�='��       �	s�D�Yc�A�7*

lossem�<�Rei       �	\uE�Yc�A�7*

lossf�<x̰�       �	0F�Yc�A�7*

loss$�(=:���       �	�F�Yc�A�7*

loss��K;��       �	�QG�Yc�A�7*

loss�4�9��
r       �	��G�Yc�A�7*

loss�`�;�*x       �	hyH�Yc�A�7*

lossS�:��       �	5I�Yc�A�7*

loss!js;���       �	ܞI�Yc�A�7*

lossEY6<sp�       �	_�J�Yc�A�7*

loss�j[:��q       �	Z/K�Yc�A�7*

lossG%=�]p       �	��K�Yc�A�7*

loss�5;�=�       �	$bL�Yc�A�7*

loss��:y�       �	3�L�Yc�A�7*

loss�[�;l$1       �	НM�Yc�A�7*

loss�K<C���       �	�1N�Yc�A�7*

loss+�;@��       �	��N�Yc�A�7*

loss��<� �       �	�O�Yc�A�7*

loss��9��y�       �	HP�Yc�A�7*

loss3�<����       �	K�P�Yc�A�7*

loss��>#�       �	�JQ�Yc�A�7*

loss�:�;⡛       �	[R�Yc�A�7*

loss20�=Ե�       �	"5S�Yc�A�7*

loss=m�<U��       �	i�S�Yc�A�7*

loss"{�<9u�3       �	�eT�Yc�A�7*

loss�<Eo_�       �	�AU�Yc�A�7*

loss�k�<.��       �	�U�Yc�A�7*

loss�<!�       �	ǛV�Yc�A�7*

losssj�=���       �	B>W�Yc�A�7*

loss�<�}�W       �	�X�Yc�A�7*

lossWJ<`N
       �	H�X�Yc�A�7*

loss�=t��       �	AY�Yc�A�7*

lossS�_=ᷲ�       �	]�Y�Yc�A�7*

loss�L=A        �	�~Z�Yc�A�7*

loss�T<%d�       �	[�Yc�A�7*

loss�Ȩ<�A��       �	��[�Yc�A�7*

loss�8�=�X�       �	 �\�Yc�A�7*

loss��z=Z�p       �	%=]�Yc�A�7*

loss�"�<۫;�       �	��]�Yc�A�7*

loss�=��Y�       �	�q^�Yc�A�8*

loss���<1���       �	-
_�Yc�A�8*

loss��;{��)       �	w�_�Yc�A�8*

loss�R=�        �	C<`�Yc�A�8*

lossRES<���+       �	&�`�Yc�A�8*

loss��Y=�}6�       �	za�Yc�A�8*

loss�zo<WD��       �	#b�Yc�A�8*

lossE��:����       �	�b�Yc�A�8*

loss�J%=�Y�0       �	6;d�Yc�A�8*

lossۀ�:�k��       �	x�d�Yc�A�8*

loss���=-	�       �	fe�Yc�A�8*

loss��<�[;L       �	g
f�Yc�A�8*

loss�L�;g��!       �	�f�Yc�A�8*

loss�K<� �|       �	�yg�Yc�A�8*

lossi�`<	��       �	�h�Yc�A�8*

loss�/�:��C        �	��h�Yc�A�8*

lossS �<[���       �	�Ai�Yc�A�8*

loss�X�;G��7       �	;�i�Yc�A�8*

lossl;B���       �	}j�Yc�A�8*

loss�W�<F��       �	�k�Yc�A�8*

loss���<w,�'       �	W�k�Yc�A�8*

loss$�<2VF�       �	�Pl�Yc�A�8*

loss|4;�y��       �	��l�Yc�A�8*

loss�Mv;U|�       �	;�m�Yc�A�8*

loss�"=�F�       �	�$n�Yc�A�8*

loss�.�<ډf       �	#�n�Yc�A�8*

loss�b�;d�       �	�So�Yc�A�8*

lossd�Y<�!a       �	�o�Yc�A�8*

loss�W:=V�,�       �	�p�Yc�A�8*

loss]>�:��p       �	�q�Yc�A�8*

lossM%<��M       �	��q�Yc�A�8*

loss�V�:G�:       �		Tr�Yc�A�8*

loss�R�;D��       �	�+s�Yc�A�8*

loss?�;����