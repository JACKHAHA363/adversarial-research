       �K"	  @Ofc�Abrain.Event:2mM�"�     �])�	e�bOfc�A"��
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
:@@*
seed2��*
T0*
seed���)*
dtype0
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
seed2���
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
seed2͡�
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
*
seed2��a
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
 *  �?*
dtype0*
_output_shapes
: 
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��*
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
:����������*
seed2���*
T0*
seed���)*
dtype0
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
 *  �@*
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
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/RealDiv_1RealDiv:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Neg-sequential_1/dropout_1/cond/dropout/keep_prob*
T0*/
_output_shapes
:���������@
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
valueB@*    *
dtype0*&
_output_shapes
:@
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
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "�o�:�,     nKqz	��fOfc�AJ��
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
:@*
seed2���*
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
seed2��
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
seed2͡�
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
_output_shapes
:	�
*
seed2��a*
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
seed2��
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
seed2���
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
divRealDivnum_correct/readadd*
_output_shapes
: *
T0
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0֚��       ��-	���Ofc�A*

loss�N@!谜       ��-	�}�Ofc�A*

loss�3@w+.�       ��-	�"�Ofc�A*

loss��@���       ��-	��Ofc�A*

loss�� @É��       ��-	YڝOfc�A*

loss���?s���       ��-	�|�Ofc�A*

loss�+�?�rY�       ��-	�(�Ofc�A*

loss��?���       ��-	�ӟOfc�A*

loss]��?�Q�       ��-	�o�Ofc�A	*

loss4{�?���k       ��-	��Ofc�A
*

loss0��?j.1�       ��-	cաOfc�A*

loss�-p?���l       ��-	χ�Ofc�A*

lossvw?zƭD       ��-	�4�Ofc�A*

loss�Ex?/!�g       ��-	3�Ofc�A*

loss/~2?k2�       ��-	���Ofc�A*

loss�(f?qi}>       ��-	�5�Ofc�A*

loss�?~+F�       ��-	��Ofc�A*

loss�H?�D�       ��-	��Ofc�A*

loss�?wt       ��-	Q��Ofc�A*

loss�u?��       ��-	!t�Ofc�A*

lossVT/?T��       ��-	[&�Ofc�A*

lossL;?'�Ē       ��-	�ΩOfc�A*

loss�'?UDV       ��-	�ɫOfc�A*

loss2�N?-5�       ��-	�{�Ofc�A*

loss�v.?l�qu       ��-	�)�Ofc�A*

loss^X?��S_       ��-	�ȮOfc�A*

loss�B,?O�pR       ��-	\U�Ofc�A*

loss�(6?�A
d       ��-	$�Ofc�A*

lossS	G?�Ij       ��-	�زOfc�A*

loss��.?p��;       ��-	���Ofc�A*

loss�m;?�>�K       ��-	m;�Ofc�A*

loss8�?R��}       ��-	���Ofc�A *

loss�f?�8       ��-	�6�Ofc�A!*

loss��Q?�,u"       ��-	�ݶOfc�A"*

loss.R�>��S       ��-	���Ofc�A#*

loss+�?�ܢF       ��-	�,�Ofc�A$*

loss,��>�E�C       ��-	�׸Ofc�A%*

lossdS5?�8z       ��-	�w�Ofc�A&*

loss��L?��
�       ��-	+�Ofc�A'*

loss��'?	#i�       ��-	f��Ofc�A(*

loss�
?(H�       ��-	ݖ�Ofc�A)*

loss�>6�%       ��-	f3�Ofc�A**

loss\�>�s       ��-	;o�Ofc�A+*

loss��?n U,       ��-	��Ofc�A,*

loss��'?�`1       ��-	iƿOfc�A-*

loss[>?��       ��-	N~�Ofc�A.*

loss7�>��j       ��-	�-�Ofc�A/*

lossN�>s�1"       ��-	1��Ofc�A0*

loss�w�>��nD       ��-	-z�Ofc�A1*

lossx�>a��       ��-	��Ofc�A2*

lossH��>��+r       ��-	E��Ofc�A3*

lossQt�>vD��       ��-	,d�Ofc�A4*

loss���>����       ��-	�	�Ofc�A5*

loss�i[?Q�       ��-	P��Ofc�A6*

loss!��>]s�R       ��-	�T�Ofc�A7*

lossTy�>`"��       ��-	��Ofc�A8*

lossF�>',�Y       ��-	E��Ofc�A9*

loss�)�>��       ��-	"6�Ofc�A:*

loss}?��8r       ��-	��Ofc�A;*

loss�S�>�E|       ��-	��Ofc�A<*

loss�D�>�^R        ��-	t'�Ofc�A=*

lossqO�>>�       ��-	���Ofc�A>*

loss
�>([T       ��-	�q�Ofc�A?*

loss��>j���       ��-	��Ofc�A@*

loss� �>3ۿ?       ��-	$�Ofc�AA*

loss(��>Ĭ�       ��-	���Ofc�AB*

loss�m!?ɰ��       ��-	ǡ�Ofc�AC*

loss��>
P-?       ��-	�W�Ofc�AD*

lossVw?a���       ��-	���Ofc�AE*

loss�{q>�p}�       ��-	؝�Ofc�AF*

loss#��>CSZ       ��-	A�Ofc�AG*

lossK�>˙��       ��-	���Ofc�AH*

lossDŠ>��`o       ��-	Ҏ�Ofc�AI*

loss�]�>���       ��-	���Ofc�AJ*

loss~�?�%t        ��-	�.�Ofc�AK*

loss\��>KdMP       ��-	q��Ofc�AL*

loss��?/�       ��-	jm�Ofc�AM*

loss��?��"       ��-	|�Ofc�AN*

loss�>��       ��-	��Ofc�AO*

loss�ǆ>�ta�       ��-	GY�Ofc�AP*

loss���>}�%�       ��-	=�Ofc�AQ*

loss�?^ 7       ��-	���Ofc�AR*

loss��>�^       ��-	��Ofc�AS*

lossI�f>��]K       ��-	�O�Ofc�AT*

loss��>]�/C       ��-	���Ofc�AU*

loss�>����       ��-	�G�Ofc�AV*

loss[ǡ>��       ��-	=+�Ofc�AW*

lossJ�>���       ��-	�3�Ofc�AX*

lossqn�>\l�       ��-	�(�Ofc�AY*

loss/.�>,���       ��-	J^�Ofc�AZ*

loss�>v0u�       ��-	�h�Ofc�A[*

loss=N�>�L g       ��-	��Ofc�A\*

loss|Ԩ>#�I�       ��-	]��Ofc�A]*

lossa$�>b��       ��-	_C�Ofc�A^*

loss�߼>���       ��-	���Ofc�A_*

loss��i>4-`�       ��-	s�Ofc�A`*

loss�v>��       ��-	,�Ofc�Aa*

loss\�>^�+�       ��-	&��Ofc�Ab*

loss�� ?mp�J       ��-	��Ofc�Ac*

lossV��>�#E       ��-	N_�Ofc�Ad*

loss�o_>�K��       ��-	���Ofc�Ae*

loss��L>�T
�       ��-	f��Ofc�Af*

loss�#�>��:�       ��-	�I�Ofc�Ag*

loss�SC>�1��       ��-	���Ofc�Ah*

loss�>t>��	�       ��-	�Ofc�Ai*

loss[7�>f-��       ��-	�#�Ofc�Aj*

loss4� >��       ��-	���Ofc�Ak*

loss�A\>pA       ��-	Z�Ofc�Al*

loss� �>��*�       ��-	���Ofc�Am*

loss���>r���       ��-	q��Ofc�An*

loss��>ª       ��-	�W�Ofc�Ao*

loss4U�>\#�G       ��-	 �Ofc�Ap*

loss��>t�2&       ��-	j��Ofc�Aq*

loss̺�>81v?       ��-	�R�Ofc�Ar*

loss�L>{�z       ��-	��Ofc�As*

loss�&�>5���       ��-	��Ofc�At*

loss�YJ>�G       ��-	,�Ofc�Au*

loss��>�}^       ��-	:�Ofc�Av*

loss2i�>w�%       ��-	���Ofc�Aw*

loss �r>��=�       ��-	�H�Ofc�Ax*

loss�
�>F ��       ��-	P��Ofc�Ay*

loss@��>j�       ��-	��Ofc�Az*

loss�kd>�/��       ��-	j��Ofc�A{*

lossڒ�>;-�       ��-	9*�Ofc�A|*

loss\��=����       ��-	��Ofc�A}*

loss�<>�7�       ��-	�n�Ofc�A~*

loss��>m|Y       ��-	�Ofc�A*

lossF/a>BԷ9       �	���Ofc�A�*

lossH�>�1��       �	w�Ofc�A�*

loss�R�>����       �	��Ofc�A�*

loss�7H>,��        �	���Ofc�A�*

loss/�>DC��       �	7q�Ofc�A�*

loss��>���       �	�'�Ofc�A�*

loss��=a��        �	���Ofc�A�*

loss�R>�bR�       �	%y Pfc�A�*

lossT�>4��       �	�$Pfc�A�*

lossf΂>�,��       �	D�Pfc�A�*

loss��=O�5.       �	AcPfc�A�*

lossI@5>�g#       �	�Pfc�A�*

loss�E,>E"�8       �	!�Pfc�A�*

loss	b�=��$l       �	�YPfc�A�*

loss]Ĕ>
q:�       �	/�Pfc�A�*

loss�R>E�E�       �	��Pfc�A�*

loss�_>�a�A       �	�<Pfc�A�*

loss&��>���       �	=�Pfc�A�*

loss�5�=be�       �	mPfc�A�*

lossK,�>-�/�       �	�Pfc�A�*

loss�k�=L���       �	I�Pfc�A�*

lossh�=U��       �	�<	Pfc�A�*

loss]><>���       �	Pfc�A�*

loss�i>�/#,       �	0�Pfc�A�*

lossng�>�}�~       �	_Pfc�A�*

loss]c�>�Z�k       �	b�Pfc�A�*

loss(��>�&g        �	M�Pfc�A�*

loss�#>��n�       �	_&Pfc�A�*

loss��=H�j�       �	"�Pfc�A�*

loss��:>��J�       �	\Pfc�A�*

loss!�r>��"       �	JPfc�A�*

loss���>�L}       �	+�Pfc�A�*

loss�!>����       �	6Pfc�A�*

loss���>�e�|       �	-�Pfc�A�*

loss$�>>B�B6       �	sfPfc�A�*

lossfiB>����       �	*Pfc�A�*

loss�&>�Bۛ       �	�Pfc�A�*

lossm�g>�       �	&4Pfc�A�*

lossF� >�S�       �	�Pfc�A�*

loss�$>���       �	7nPfc�A�*

loss�~L>�sr�       �	�Pfc�A�*

loss(��=��/�       �	��Pfc�A�*

loss��2>�#I\       �	�IPfc�A�*

loss>1�X       �	T�Pfc�A�*

loss_�=�3�q       �	c�Pfc�A�*

lossG>G'       �	�.Pfc�A�*

loss{(V>�^�       �	�Pfc�A�*

loss��=U���       �	6rPfc�A�*

loss��D>H^�       �	�Pfc�A�*

loss�3 ?�z�M       �	}�Pfc�A�*

loss�Y>��L�       �	gEPfc�A�*

loss|'>�5       �	�Pfc�A�*

loss��=�08�       �	yyPfc�A�*

loss-�,=�gYX       �	ZPfc�A�*

loss�@z>���       �	��Pfc�A�*

loss�e[>[��       �	�>Pfc�A�*

loss
>h�:�       �	��Pfc�A�*

lossEK>�j
       �	�s Pfc�A�*

loss1I�=��,\       �	�!Pfc�A�*

loss?~>����       �	�!Pfc�A�*

loss�,�>�|-d       �	xE"Pfc�A�*

loss
">R��-       �	��"Pfc�A�*

loss��#>V�tA       �	�#Pfc�A�*

loss�G>�$W	       �	h�$Pfc�A�*

loss{�>��       �	�I%Pfc�A�*

loss���=W�;       �	�%Pfc�A�*

lossz>���\       �	/�&Pfc�A�*

loss.\�=I��       �	$'Pfc�A�*

loss<�>Zv�       �	<�'Pfc�A�*

loss�n>�B��       �	�Z(Pfc�A�*

loss� K>�Va       �	�(Pfc�A�*

loss8D�>���        �	�)Pfc�A�*

loss��?>�&��       �	�'*Pfc�A�*

loss�IX>����       �	�*Pfc�A�*

lossW�=#���       �	�f+Pfc�A�*

lossfR�=f��       �	�+Pfc�A�*

loss�>ɟ[�       �	��,Pfc�A�*

lossV� >���f       �	�:-Pfc�A�*

loss|�>m��e       �	�-Pfc�A�*

losse=c=�c*�       �	^�.Pfc�A�*

loss�~>��D       �	�O/Pfc�A�*

lossV�>�긐       �	]0Pfc�A�*

lossc
>k��y       �	ެ0Pfc�A�*

lossw!>^�\3       �	��1Pfc�A�*

loss���=@~`       �	u�2Pfc�A�*

lossH˼=o$S�       �	aS3Pfc�A�*

lossR$�=���V       �	�4Pfc�A�*

loss��>>�Q�       �	ԝ4Pfc�A�*

loss��Z>�tE�       �	�35Pfc�A�*

lossL�P>Un�       �	��5Pfc�A�*

loss��>E��       �	B{6Pfc�A�*

lossW:�>��       �	/�7Pfc�A�*

loss�K�>�:��       �	Gw8Pfc�A�*

lossVt=8��       �	�9Pfc�A�*

loss���=d��       �	�	:Pfc�A�*

loss�*�>@���       �	 �:Pfc�A�*

loss\�>&�l�       �	EK;Pfc�A�*

lossJ�y=y |w       �	%�;Pfc�A�*

loss��(>�j��       �	��<Pfc�A�*

loss��C>kzv�       �	�'=Pfc�A�*

loss-J>Hw�       �	 �=Pfc�A�*

loss��9>j�V       �	�a>Pfc�A�*

loss�:Y>���8       �	��>Pfc�A�*

losst >��O�       �	R�?Pfc�A�*

loss-۩=2��       �	�7@Pfc�A�*

lossd />�5l�       �	�@Pfc�A�*

loss��+>v*�       �	g}APfc�A�*

loss� r>[Β'       �	BPfc�A�*

loss��>�{�w       �	̴BPfc�A�*

loss;��=ꍦ�       �	@OCPfc�A�*

loss誉>.�       �	zQDPfc�A�*

loss��3>Xo�       �	��DPfc�A�*

loss�L�=;��B       �	�EPfc�A�*

lossnj&>��{=       �	�3FPfc�A�*

lossI�>:r       �	B�FPfc�A�*

loss�}C>}���       �	�hGPfc�A�*

loss�!k>��)       �	�HPfc�A�*

loss��>��D       �	��HPfc�A�*

lossWAY>�|
       �	dYIPfc�A�*

lossJ��=$�Z�       �	�JPfc�A�*

loss֬�=l)*       �	�JPfc�A�*

loss�M>���>       �	��KPfc�A�*

loss�>6FtG       �	-"LPfc�A�*

loss<�?>w�k       �	��LPfc�A�*

loss��>�C��       �	mMPfc�A�*

lossl)>�L�S       �	&NPfc�A�*

loss��=��c       �	�NPfc�A�*

lossc��=�e��       �	�bOPfc�A�*

loss!�">T�)�       �	XPPfc�A�*

loss�xC>���b       �	?�PPfc�A�*

lossYX>�       �	�JQPfc�A�*

loss�s>�e�.       �	J�QPfc�A�*

loss�E>�G�C       �	z�RPfc�A�*

lossĈ>O}|       �	�&SPfc�A�*

lossv��=�@�|       �	��SPfc�A�*

loss���>_ �       �	Z�TPfc�A�*

loss�d�=/`       �	#,UPfc�A�*

lossv�o>L�]       �	��UPfc�A�*

loss2��= ���       �	�tVPfc�A�*

loss���>�	X�       �	�WPfc�A�*

lossW�^>)e�       �	ܻWPfc�A�*

loss�ʭ>���S       �	�gXPfc�A�*

loss� �=g��j       �	�	YPfc�A�*

lossN��=Z��Z       �	F�YPfc�A�*

loss��i>�k�       �	wZPfc�A�*

loss��	>?iXE       �	�2[Pfc�A�*

loss�M>��       �	~�[Pfc�A�*

loss1�R>7�zu       �	[�\Pfc�A�*

lossD;�>w�q�       �	/R]Pfc�A�*

loss*9�=�c       �	�^Pfc�A�*

loss�PK>^�       �	��^Pfc�A�*

loss��H>X�ʀ       �	�_Pfc�A�*

loss��H>�@�       �	Wx`Pfc�A�*

loss��=?���       �	�+aPfc�A�*

loss���=1p4�       �	4�aPfc�A�*

lossTp�>��       �	r�bPfc�A�*

loss��>}׽5       �	�9cPfc�A�*

loss晅>�*��       �	F�cPfc�A�*

loss�K>��=       �	�dPfc�A�*

loss�8>&�9�       �	&WePfc�A�*

loss�J�=���1       �	�fPfc�A�*

loss�=A�*r       �	��fPfc�A�*

loss�d{=���       �	��gPfc�A�*

loss�:=���       �	�XhPfc�A�*

loss2i�=�R3       �	�iPfc�A�*

lossWM==���H       �	��iPfc�A�*

loss��j=�7@       �	�,kPfc�A�*

loss2��=A`��       �	��kPfc�A�*

lossC�I>��$       �	O�lPfc�A�*

loss���=��       �	�TmPfc�A�*

lossN�t>.�E       �	ZnPfc�A�*

lossTa>��       �	�=oPfc�A�*

loss��l> ���       �	��oPfc�A�*

loss%��=�       �	ͰpPfc�A�*

loss���=�~�       �	�WqPfc�A�*

lossr�=e�B�       �	rPfc�A�*

loss��>,|�       �	k�rPfc�A�*

loss=	>��@J       �	��sPfc�A�*

loss�!�>�C/�       �	�@tPfc�A�*

loss=�<��ہ       �	��tPfc�A�*

lossϤ>p���       �	��uPfc�A�*

loss�G=~U8       �	-vPfc�A�*

loss��=�p��       �	��vPfc�A�*

loss��=^쐌       �	_wPfc�A�*

lossOd:>��-�       �	.�wPfc�A�*

lossRD�=Z���       �	n�xPfc�A�*

loss�/X=���       �	�RyPfc�A�*

lossV�?=���       �	��yPfc�A�*

loss�m>t=��       �	��zPfc�A�*

loss�(�=��QZ       �	1z{Pfc�A�*

loss��Z=L]j�       �	�|Pfc�A�*

loss�-:>�q�x       �	��|Pfc�A�*

loss�9>x��       �	X}Pfc�A�*

loss�C=sj"p       �	i~Pfc�A�*

loss��D>|�9       �	�~Pfc�A�*

loss�`>���r       �	NPfc�A�*

lossRB>�D;�       �	��Pfc�A�*

lossk��>��       �	���Pfc�A�*

loss�y�>�W�q       �	�N�Pfc�A�*

losswp2>��`*       �	��Pfc�A�*

loss(Ky=v~�       �	狂Pfc�A�*

lossXcW>J�U       �	�(�Pfc�A�*

loss_aG>{%d       �	�҃Pfc�A�*

loss�A�=K7��       �	pz�Pfc�A�*

loss���=*�       �	��Pfc�A�*

loss�L�=eV       �	׽�Pfc�A�*

loss��=�>�       �	�e�Pfc�A�*

loss�.><��       �	��Pfc�A�*

lossZ%>	<.       �	d��Pfc�A�*

loss�|m>�wA�       �	�Q�Pfc�A�*

loss-�>�'F       �	��Pfc�A�*

lossn9>�Q$       �	���Pfc�A�*

loss��=,��       �	�e�Pfc�A�*

loss��G=���       �	��Pfc�A�*

loss��f=ی�c       �	ϣ�Pfc�A�*

lossʀS>I�ئ       �	�N�Pfc�A�*

loss�w�=�C��       �	��Pfc�A�*

lossH>�>\E*:       �	(��Pfc�A�*

loss�=4>@�|�       �	RG�Pfc�A�*

loss��=�u��       �	��Pfc�A�*

loss�ь>��E"       �	b��Pfc�A�*

lossԈ�=��\J       �	�D�Pfc�A�*

lossw�>#|       �	���Pfc�A�*

lossm[�=��l       �	���Pfc�A�*

loss�>�=���       �	"U�Pfc�A�*

lossk>�FX�       �	3��Pfc�A�*

loss�7>pH3h       �	ɫ�Pfc�A�*

lossX�>>�,(       �	�\�Pfc�A�*

loss��^=�n_�       �	��Pfc�A�*

loss��=�/�       �	���Pfc�A�*

lossF��=t�Nc       �	L�Pfc�A�*

loss�nv=V��       �	��Pfc�A�*

losshs�=�^�       �	���Pfc�A�*

loss��	>>���       �	o+�Pfc�A�*

loss�@>W���       �	RјPfc�A�*

loss*<�=t��m       �	>u�Pfc�A�*

loss�A>����       �	�Pfc�A�*

lossY
>�m�o       �	/��Pfc�A�*

loss�>�=�f��       �	B��Pfc�A�*

lossa\a>�"�M       �	X9�Pfc�A�*

lossRP�={K3       �	7�Pfc�A�*

loss�>��`|       �	B{�Pfc�A�*

lossZ�=v��I       �	]�Pfc�A�*

loss(\S>/i�(       �	ᴞPfc�A�*

loss�\>*�n�       �	�L�Pfc�A�*

loss�(�=-K�       �	@��Pfc�A�*

loss�>�=�G"�       �	���Pfc�A�*

loss{�=�l       �	�@�Pfc�A�*

loss2�>�Z       �		�Pfc�A�*

lossDU>7�l       �	���Pfc�A�*

loss�`�=U�w       �	�(�Pfc�A�*

loss>g>s�       �	lУPfc�A�*

loss]D>G�Gf       �	�p�Pfc�A�*

loss�9�>s��       �	��Pfc�A�*

loss�9�>M�/       �	���Pfc�A�*

lossv�<=nS�C       �	�W�Pfc�A�*

lossO��=�`       �	���Pfc�A�*

loss��u=�8�:       �	WΧPfc�A�*

lossr��=�~�_       �	�h�Pfc�A�*

losswU>rK��       �	��Pfc�A�*

lossaj>f�D       �	���Pfc�A�*

loss��>���y       �	�=�Pfc�A�*

loss[�m>��       �	�תPfc�A�*

lossN�/>ё�j       �	�p�Pfc�A�*

loss���=mF��       �	��Pfc�A�*

loss�3>����       �	��Pfc�A�*

loss��>c��       �	q;�Pfc�A�*

loss�t�=�"OU       �	߭Pfc�A�*

lossZu>�S�H       �	�v�Pfc�A�*

loss�A�=�gԦ       �	��Pfc�A�*

loss	�v=tc:�       �	H��Pfc�A�*

loss_�<>��       �	@j�Pfc�A�*

loss�K>"B��       �	��Pfc�A�*

loss61>,��       �	R��Pfc�A�*

lossF:>�z%>       �	�8�Pfc�A�*

lossm�>%�KI       �	/�Pfc�A�*

loss���<�ـ       �	�{�Pfc�A�*

loss;A�=hv�       �	. �Pfc�A�*

loss]>�iy       �	P�Pfc�A�*

lossxzJ>d�        �	�{�Pfc�A�*

loss�;�=�c'       �	�)�Pfc�A�*

loss?�;>��]       �	.�Pfc�A�*

loss��@>5��N       �	j��Pfc�A�*

loss�g>��       �	V-�Pfc�A�*

losszoP>p�t       �	�иPfc�A�*

loss�>���       �	�f�Pfc�A�*

loss��>>d       �	g�Pfc�A�*

loss]2>�j8�       �	��Pfc�A�*

loss�>�8�       �	O@�Pfc�A�*

loss$�M=��'       �	DݻPfc�A�*

loss�bZ=_��       �	�r�Pfc�A�*

loss�!>�[��       �	�	�Pfc�A�*

loss�)->$���       �	���Pfc�A�*

loss�x�=`�/�       �	�5�Pfc�A�*

loss2�= �)       �	�޾Pfc�A�*

loss�_�=�s��       �	{��Pfc�A�*

loss{ό>v�C�       �	Z+�Pfc�A�*

lossx2>���_       �	���Pfc�A�*

loss@@_>�e�-       �	Ql�Pfc�A�*

loss1'�=q�2       �	^�Pfc�A�*

loss�'�=��d       �	x��Pfc�A�*

loss�^>S���       �	~V�Pfc�A�*

losst�==�       �	���Pfc�A�*

loss�->'b       �	���Pfc�A�*

loss� �=R�Z�       �	�+�Pfc�A�*

loss)�=���       �	V��Pfc�A�*

loss�p>B,A�       �	�s�Pfc�A�*

loss�l>Lv��       �	*�Pfc�A�*

loss�j=�î�       �	���Pfc�A�*

lossEo0>��       �	y�Pfc�A�*

loss6j>��b       �	\�Pfc�A�*

loss��c=T%�V       �	���Pfc�A�*

loss���=��       �	�Q�Pfc�A�*

loss_}>�-�       �	��Pfc�A�*

loss�_^>�/       �	L��Pfc�A�*

loss��=0�'       �	�C�Pfc�A�*

loss[�=�u�       �	���Pfc�A�*

loss�=�r9�       �	�Pfc�A�*

lossl3/>�mo       �	N)�Pfc�A�*

lossa�>-։       �	���Pfc�A�*

loss�r�=���       �	#e�Pfc�A�*

loss��=�	Y�       �	F
�Pfc�A�*

loss��]>��       �	ɮ�Pfc�A�*

lossFj>>��K�       �	Q�Pfc�A�*

lossLٜ=�}�T       �	k��Pfc�A�*

loss�P=R�y$       �	W��Pfc�A�*

lossM�>��+       �	�7�Pfc�A�*

loss:�>*�+�       �	���Pfc�A�*

lossl�>7�"�       �	x�Pfc�A�*

loss[�!>��h       �	�Pfc�A�*

loss([>7A�S       �	'��Pfc�A�*

loss��=� [�       �	1]�Pfc�A�*

lossn��=B�%�       �	��Pfc�A�*

loss[�=��&�       �	}��Pfc�A�*

lossYف=�S�       �	/�Pfc�A�*

loss7�=})'�       �	���Pfc�A�*

lossz�@>��g�       �	g�Pfc�A�*

loss�T�=H~�X       �	��Pfc�A�*

loss���=��)       �	A��Pfc�A�*

loss��=�06A       �	�2�Pfc�A�*

lossc�n=���@       �	�c�Pfc�A�*

loss�b>EU�a       �	�N�Pfc�A�*

loss�=�<���e       �	���Pfc�A�*

losss.�=#�4�       �	#��Pfc�A�*

loss`{�=1qO�       �	p%�Pfc�A�*

loss��S>B�;�       �		��Pfc�A�*

loss�5D>��d       �	)Y�Pfc�A�*

loss�+�=e|*8       �	I��Pfc�A�*

loss*��=xnu�       �	~��Pfc�A�*

loss�;]=���$       �	�2�Pfc�A�*

loss��>]�ث       �	 ��Pfc�A�*

loss7}=��^g       �	�h�Pfc�A�*

loss�0>��       �	��Pfc�A�*

loss��>=fc7       �	5��Pfc�A�*

loss*$�=C`��       �	<2�Pfc�A�*

loss3(>)[��       �	y��Pfc�A�*

lossԢ�=�Q(       �	Jb�Pfc�A�*

loss�6�=��~       �	���Pfc�A�*

loss*��=��y       �	���Pfc�A�*

losst2�=��,�       �	�T�Pfc�A�*

loss��\>���^       �	8��Pfc�A�*

loss��]=�D~�       �	*��Pfc�A�*

loss��/>�Į       �	F&�Pfc�A�*

lossW�=ZVPe       �	���Pfc�A�*

loss�Bo>�£       �	|c�Pfc�A�*

loss@HL=�M�+       �	���Pfc�A�*

loss�==p��       �	ѕ�Pfc�A�*

loss\:�=4؊�       �	�1�Pfc�A�*

loss�X=(��V       �	)��Pfc�A�*

loss=�>2��5       �	�n�Pfc�A�*

loss0��=-&*�       �	��Pfc�A�*

lossS�=h㗜       �	e��Pfc�A�*

loss��o>��o�       �	�c�Pfc�A�*

loss�]�=]�u0       �	&�Pfc�A�*

loss�X=y�       �	���Pfc�A�*

loss�߈=�ɝC       �	�X�Pfc�A�*

loss�$�=J��       �	4��Pfc�A�*

lossg'>]{�       �	���Pfc�A�*

losshÙ=|��       �	a�Pfc�A�*

loss�<	>��,�       �	�Pfc�A�*

loss_GI=j       �	���Pfc�A�*

lossaO>��g�       �	�N�Pfc�A�*

loss!m=܃��       �	Q�Pfc�A�*

loss��g=���       �	���Pfc�A�*

loss���=I���       �	iV�Pfc�A�*

loss���=�^�e       �	���Pfc�A�*

lossME>��A       �	ҏ�Pfc�A�*

lossӞ=��K       �	�,�Pfc�A�*

loss3{@>R��l       �	]��Pfc�A�*

loss�J�=�ckK       �	(�Pfc�A�*

lossΐ�=�M�       �	��Pfc�A�*

loss]��=A~C\       �	Թ�Pfc�A�*

loss��=ؘ_�       �	�_�Pfc�A�*

loss{ �=��j�       �	�Pfc�A�*

loss�v�=^Ep�       �	��Pfc�A�*

lossᎏ<J���       �	}>�Pfc�A�*

loss��=�FQ�       �	Z��Pfc�A�*

loss���=�       �	p Qfc�A�*

lossk�= "i       �	1Qfc�A�*

lossV��=��u       �	��Qfc�A�*

lossS�O>�'j�       �	�;Qfc�A�*

loss,>J&޾       �	"�Qfc�A�*

loss�4�<S�qF       �	V~Qfc�A�*

loss��=/qu       �	�Qfc�A�*

loss^�=��W       �	��Qfc�A�*

loss�2D=���       �	4hQfc�A�*

loss���<�>��       �	�	Qfc�A�*

loss�=9$�?       �	 �Qfc�A�*

loss�#=����       �	�BQfc�A�*

lossMT=RF       �	<�Qfc�A�*

lossc_=��f�       �	�uQfc�A�*

lossF։<�KX9       �	�	Qfc�A�*

loss7Ǯ=���       �	;�	Qfc�A�*

loss-�e<ܿ]q       �	�<
Qfc�A�*

lossF��;���       �	c�
Qfc�A�*

loss _�<caD       �	�gQfc�A�*

loss� �=����       �	r�Qfc�A�*

lossbP>���       �	��Qfc�A�*

loss��=<��t       �	�3Qfc�A�*

loss���;���p       �	q�Qfc�A�*

loss#�1=�;nG       �	�jQfc�A�*

loss��?
^�       �	Qfc�A�*

loss�m�<����       �	��Qfc�A�*

loss� =Y/l�       �	�fQfc�A�*

loss���=�+�       �	��Qfc�A�*

lossq�
>O��       �	�Qfc�A�*

loss�(J=Ϸ.<       �	,+Qfc�A�*

loss���=�3~�       �	+�Qfc�A�*

loss_�=�%��       �	UQfc�A�*

lossci�=�p       �	��Qfc�A�*

lossT�=�y2$       �	?�Qfc�A�*

loss&�=�)ą       �	(*Qfc�A�*

loss�p,>��Yf       �	��Qfc�A�*

loss6?d>��׀       �	�cQfc�A�*

lossMu7>h�)       �	@�Qfc�A�*

loss��=_C       �	�Qfc�A�*

loss�4u>i�       �	�)Qfc�A�*

loss$m/>�-M       �	Q�Qfc�A�*

loss$��=&�       �	UQfc�A�*

lossq��=��`C       �	B�Qfc�A�*

lossAg�=2Z�       �	{�Qfc�A�*

loss���=+�(�       �	u!Qfc�A�*

lossJ�L=�g�\       �	�Qfc�A�*

loss[*�=p�o       �	�TQfc�A�*

lossEV>�A	       �	��Qfc�A�*

loss�;�<�!       �	YQfc�A�*

lossܣ2=�h��       �	�2Qfc�A�*

lossf_�<��>o       �	�h Qfc�A�*

loss��=w�0�       �	؃!Qfc�A�*

loss*57=I�@       �	{h"Qfc�A�*

loss��>v�:(       �	;�#Qfc�A�*

loss~>i�c�       �	�O$Qfc�A�*

lossʹt=�/Mw       �	�%Qfc�A�*

loss���=)�z�       �	��%Qfc�A�*

loss�X>���7       �	Ja&Qfc�A�*

loss#�<�4[�       �	�'Qfc�A�*

lossI�=�\       �	��'Qfc�A�*

loss�K=�N       �	\V(Qfc�A�*

lossi3f=�U�1       �	��(Qfc�A�*

lossS >�["�       �	��)Qfc�A�*

loss���=��D*       �	�<*Qfc�A�*

loss���=�]�       �	M�*Qfc�A�*

loss�.=*��       �	�s+Qfc�A�*

loss��u=�E�       �	�	,Qfc�A�*

loss�=�=`mF       �	#�,Qfc�A�*

loss���=aɡ�       �	1?-Qfc�A�*

loss���=�h��       �	�-Qfc�A�*

loss�ì=�50�       �	du.Qfc�A�*

loss��=����       �	�"/Qfc�A�*

loss4�=[��       �	��/Qfc�A�*

lossϝ+>�
x       �	�b0Qfc�A�*

loss���=�Ao�       �	�1Qfc�A�*

loss.�d=P2(       �	8�1Qfc�A�*

loss�>^y��       �	��KQfc�A�*

loss�h�=��M       �	YLQfc�A�*

loss�]O>-i�       �	�LQfc�A�*

lossѭ>R<�       �	��MQfc�A�*

lossq2�=TȊ3       �	�$NQfc�A�*

loss<Ab=m���       �	U�NQfc�A�*

loss4��=���?       �	�WOQfc�A�*

loss�m>��N;       �	��PQfc�A�*

loss�ƙ=eףW       �	�RQfc�A�*

loss��2>Gٔ       �	l"SQfc�A�*

lossaJ=
 !0       �	��SQfc�A�*

loss�v�=��EY       �	�mTQfc�A�*

loss7b>�َ       �	�UQfc�A�*

loss-��=�J��       �	��UQfc�A�*

loss�;=;Wg�       �	�kVQfc�A�*

loss�M>\�$Y       �	�WQfc�A�*

loss�T�=�N/Y       �	�XQfc�A�*

lossLM|=Ȋ��       �	��XQfc�A�*

loss�J[=��e�       �	��YQfc�A�*

loss��>�j�       �	�.ZQfc�A�*

loss�Ż=V��       �	�Y[Qfc�A�*

loss��>9.8�       �	��[Qfc�A�*

loss�~>!x<�       �	�\Qfc�A�*

lossӽ>>��       �	n�]Qfc�A�*

loss21�=d��       �	�{^Qfc�A�*

lossJ=T�2       �	�H_Qfc�A�*

loss���=�44�       �	��_Qfc�A�*

loss.g=���       �	o�`Qfc�A�*

loss\�>y���       �	aQfc�A�*

loss��H=Y�x�       �	�aQfc�A�*

lossQS=끭�       �	�SbQfc�A�*

loss�<@=_�	       �	�bQfc�A�*

loss���=���       �	��cQfc�A�*

loss�i>��r0       �	�7dQfc�A�*

loss�uo=�&<       �	x�dQfc�A�*

loss4��=�=�       �	�deQfc�A�*

lossOAT=Zёz       �	/�eQfc�A�*

loss�J>�l�       �	˝fQfc�A�*

loss%s>u�bu       �	�5gQfc�A�*

loss��2>�,ͼ       �	/�gQfc�A�*

loss�S�=���{       �	�~hQfc�A�*

loss���=��Ѯ       �	$�jQfc�A�*

loss�z<wW��       �	�FkQfc�A�*

loss�7�=O5�       �	n�lQfc�A�*

loss��=�fc�       �	wfmQfc�A�*

loss�1E>(cMC       �	x	nQfc�A�*

loss_)�=f�       �	&�nQfc�A�*

loss1��=#�5:       �	�IoQfc�A�*

losss�=��p�       �	��oQfc�A�*

loss�}%=��G�       �	\�pQfc�A�*

loss,�<��A�       �	Q-qQfc�A�*

loss2I�=0��       �	��qQfc�A�*

lossn��=0|Az       �	Y�rQfc�A�*

loss!L�>���       �	�sQfc�A�*

lossmj�=�*       �	{�sQfc�A�*

loss��<��1�       �	��tQfc�A�*

loss}�=X��       �	˻uQfc�A�*

loss���<��B�       �	�YvQfc�A�*

loss2�=H�ne       �		owQfc�A�*

loss���=�uY,       �	�xQfc�A�*

loss���=J��Q       �	��xQfc�A�*

loss���=�J��       �	�;yQfc�A�*

loss��=�7�       �	�yQfc�A�*

lossI�=��       �	WvzQfc�A�*

loss���=o�       �	{Qfc�A�*

loss�}�={�       �	Ū{Qfc�A�*

lossh��=e��[       �	�T|Qfc�A�*

loss�
(>� }T       �	m�|Qfc�A�*

loss��g>�@��       �	�}}Qfc�A�*

loss|
>�P�       �	/~Qfc�A�*

loss}wL=�^Z       �	r�~Qfc�A�*

loss�>�M       �	�iQfc�A�*

loss[�]=v�/�       �	H�Qfc�A�*

loss���=~I5�       �	W�Qfc�A�*

loss|�>)�~�       �	֍�Qfc�A�*

loss$�=�f,�       �	}!�Qfc�A�*

loss��L>�2�D       �	���Qfc�A�*

lossS��>���       �	Q�Qfc�A�*

lossD��<b~�{       �	�"�Qfc�A�*

loss�߂=.z�7       �	���Qfc�A�*

loss�B�=�#       �	�ǅQfc�A�*

loss�jG>Y�7C       �	iT�Qfc�A�*

loss<�)>����       �	o��Qfc�A�*

lossz�b=$��M       �	^��Qfc�A�*

loss`[�=M�P       �	�F�Qfc�A�*

loss��=9�       �	l�Qfc�A�*

loss�">�-��       �	6��Qfc�A�*

loss�\�=�@u       �	�5�Qfc�A�*

loss��j=��C�       �	���Qfc�A�*

loss�\>���       �	���Qfc�A�*

loss(4>��t       �	rj�Qfc�A�*

loss���=��        �	��Qfc�A�*

lossqh�=�yC       �	���Qfc�A�*

loss��=�n	       �	Ja�Qfc�A�*

lossrr>��       �	��Qfc�A�*

loss�Wk=n�5�       �	���Qfc�A�*

lossҠ�=�`I;       �	���Qfc�A�*

lossE�=9�6�       �	XU�Qfc�A�*

loss.�*>���W       �	��Qfc�A�*

losss`�=�-s|       �	q��Qfc�A�*

lossd�<�3�       �	=��Qfc�A�*

loss��=���o       �	��Qfc�A�*

loss���==���       �	�Qfc�A�*

loss�̑={=��       �	�Z�Qfc�A�*

loss�_>�V!o       �	o�Qfc�A�*

loss&�C>Ȉ�~       �	L��Qfc�A�*

loss�?=�v�       �	�"�Qfc�A�*

loss8�Z=r��       �	ݙQfc�A�*

loss�>*�K       �	h��Qfc�A�*

loss\~>�a�K       �	
-�Qfc�A�*

loss�>~
.       �	pЛQfc�A�*

loss+�=m\��       �	�g�Qfc�A�*

loss�� =���C       �	�"�Qfc�A�*

lossd��<nz8.       �	��Qfc�A�*

lossYh�=~�ދ       �	�?�Qfc�A�*

loss=��=�v�       �	hw�Qfc�A�*

loss��>�#       �	��Qfc�A�*

loss�>�$�       �	�S�Qfc�A�*

loss�zM=*]gD       �	o�Qfc�A�*

loss��=���y       �	��Qfc�A�*

lossSJ�=�,K       �	A��Qfc�A�*

lossH��=��{       �	�إQfc�A�*

loss��=�C9�       �	�}�Qfc�A�*

loss��=\LV�       �	G�Qfc�A�*

loss�q<�5�       �	�ҧQfc�A�*

loss�i�=����       �	(~�Qfc�A�*

loss�-�=�B�b       �	�{�Qfc�A�*

lossHh=�_�       �	D�Qfc�A�*

loss*.>�       �	t#�Qfc�A�*

lossgl�=��2       �	{��Qfc�A�*

losss��=,I��       �	���Qfc�A�*

loss�=�9�v       �	��Qfc�A�*

loss�9|=���       �	��Qfc�A�*

loss8q�<3�       �	�I�Qfc�A�*

loss3��=K��a       �	ӈ�Qfc�A�*

lossFWb=)�D       �	�#�Qfc�A�*

loss��>����       �	b�Qfc�A�*

loss�46=)��t       �	X�Qfc�A�*

loss/u$=N>^V       �	e޳Qfc�A�*

loss���<�<
)       �	o�Qfc�A�*

lossY��<S�]�       �	>��Qfc�A�*

loss�]N=�!��       �	ڍ�Qfc�A�*

loss1(>v�       �	 F�Qfc�A�*

loss�O>�js�       �	l&�Qfc�A�*

loss�=���x       �	��Qfc�A�*

loss;��<�z�A       �	}��Qfc�A�*

loss$>b1�       �	'M�Qfc�A�*

loss)a�<��k�       �	���Qfc�A�*

lossV�<F�       �	��Qfc�A�*

loss��r=ߴ"?       �	�U�Qfc�A�*

loss�F=%.�       �	S�Qfc�A�*

losshM�=^�_�       �	��Qfc�A�*

loss/L=��U       �	���Qfc�A�*

loss;J�=�F�       �	ع�Qfc�A�*

loss�=�=#�Ev       �	z��Qfc�A�*

lossw*	=CZ4       �	}A�Qfc�A�*

loss/��<���$       �	j��Qfc�A�*

loss-2>sU       �	���Qfc�A�*

loss��=�|�       �	7��Qfc�A�*

loss�}f=T%�       �	���Qfc�A�*

lossܐ�=���       �	7��Qfc�A�*

loss �=�E�       �	�Qfc�A�*

loss��=��4�       �	U��Qfc�A�*

lossT^T=ھ=�       �	7��Qfc�A�*

loss�۫=���       �	K��Qfc�A�*

lossӜ>=�g��       �	�_�Qfc�A�*

loss�DW=��2�       �	��Qfc�A�*

loss��s=��=�       �	�E�Qfc�A�*

loss��W=��D       �	���Qfc�A�*

lossɟ�=���       �	�}�Qfc�A�*

loss*��=����       �	t~�Qfc�A�*

lossT=wPw�       �	r�Qfc�A�*

loss���=X�       �	���Qfc�A�*

loss��^=<��I       �	���Qfc�A�*

loss h�<Y��       �	�6�Qfc�A�*

lossC+|=�["A       �	_��Qfc�A�*

lossm��=��F       �	�t�Qfc�A�*

loss���=>8Kg       �	Y�Qfc�A�*

loss&�>���       �	�&�Qfc�A�*

lossV�0=�Ud�       �	���Qfc�A�*

loss���<����       �	�a�Qfc�A�*

loss���=2m��       �	r��Qfc�A�*

loss��=ե��       �	ș�Qfc�A�*

lossM�s={��       �	�4�Qfc�A�*

loss��<�1       �	w��Qfc�A�*

loss �=�?��       �	Gu�Qfc�A�*

lossj�=�0�       �	�Qfc�A�*

loss��s=l˪r       �	ݱ�Qfc�A�*

loss��=H㯚       �	�X�Qfc�A�*

loss��='�+h       �	:�Qfc�A�*

lossҶ>1h�       �	���Qfc�A�*

loss��<�>	       �	�Qfc�A�*

loss�I�=�d��       �	���Qfc�A�*

loss�p=��3f       �	���Qfc�A�*

loss�$	=�:       �	}�Qfc�A�*

loss,=L��       �	��Qfc�A�*

loss���=�M-
       �	�'�Qfc�A�*

loss�kN= ��I       �	��Qfc�A�*

loss��P>���       �	+��Qfc�A�*

loss��=х��       �	���Qfc�A�*

losswd�=C,�?       �	�[�Qfc�A�*

loss<�q=��\,       �	h#�Qfc�A�*

loss�r=���       �	���Qfc�A�*

loss�}=)Ob�       �	�g�Qfc�A�*

loss툕=Us        �	��Qfc�A�*

loss��=�B       �	p��Qfc�A�*

loss	BW=��j       �	75�Qfc�A�*

loss�(�=M^^�       �	���Qfc�A�*

lossID�=��       �	�m�Qfc�A�*

loss��=.�Ez       �	��Qfc�A�*

loss��=Ӟ�#       �	b��Qfc�A�*

loss���<�y�H       �	���Qfc�A�*

lossϑ�<%a�       �	�U�Qfc�A�*

loss.�b=շ�       �	���Qfc�A�*

loss/�=�Q;       �	�k�Qfc�A�*

loss���=d x       �	��Qfc�A�*

lossTȰ=�       �	Ӥ�Qfc�A�*

loss� 2>0�y�       �	G�Qfc�A�*

loss$�=I���       �	,��Qfc�A�*

loss)�C>vC�       �	��Qfc�A�*

loss�C�<l�>R       �	�A�Qfc�A�*

loss���=�e�F       �	���Qfc�A�*

loss{ئ=��r       �	Z��Qfc�A�*

loss�>γkI       �	�"�Qfc�A�*

loss��<�z�-       �	���Qfc�A�*

lossd�=S�b�       �	�b�Qfc�A�*

loss���=8�ӆ       �	��Qfc�A�*

loss4k�=/.�       �	A��Qfc�A�*

loss!=�,       �	.;�Qfc�A�*

loss�I�=;�T�       �	^��Qfc�A�*

loss���<��<�       �	Q��Qfc�A�*

lossc�=�\��       �	~�Qfc�A�*

loss��=���$       �	��Qfc�A�*

lossͽ�=�uF       �	GV�Qfc�A�*

loss[W�=G0�       �	1��Qfc�A�*

loss�:�=h�;       �	���Qfc�A�*

loss w�=E#�t       �	~�Qfc�A�*

loss8E>�R��       �	@��Qfc�A�*

loss6��=��A       �	�Z Rfc�A�*

loss�iF=�2��       �	�� Rfc�A�*

lossʕ�=����       �	L�Rfc�A�*

loss���=�/S�       �	�,Rfc�A�*

loss[�>ńE�       �	t�Rfc�A�*

lossۆ�=Ն�       �	�pRfc�A�*

lossL�=$�0�       �	�Rfc�A�*

loss���=���       �	I�Rfc�A�*

loss=DRU�       �	YRfc�A�*

loss(�X=0q�?       �	��Rfc�A�*

loss��=�d�       �	m�Rfc�A�*

loss@Ɋ=V�       �	�)Rfc�A�*

loss�=�^R~       �	�Rfc�A�*

loss���=v�Tb       �	@iRfc�A�*

loss�&b=z���       �	s	Rfc�A�*

loss_(�<�� �       �	�	Rfc�A�*

loss�S�<N�       �	_E
Rfc�A�*

loss}��<       �	"�
Rfc�A�*

loss*>�'�       �	�|Rfc�A�*

lossSբ=}���       �	�Rfc�A�*

loss��i=\��       �	F�Rfc�A�*

loss^Q>�~u4       �	~RRfc�A�*

loss%�=f���       �	��Rfc�A�*

loss�U�<�m�       �	/�Rfc�A�*

loss0�=��&�       �	�(Rfc�A�*

loss��=�Q2       �	��Rfc�A�*

lossH7�=C f#       �	F`Rfc�A�*

loss[\:=�E�       �	��Rfc�A�*

loss&��=Ps�       �	�Rfc�A�*

loss*��=�Wh       �	�.Rfc�A�*

loss�F>/*��       �	��Rfc�A�*

loss�{�=��)�       �	�aRfc�A�*

lossl��=�f       �	�Rfc�A�*

loss2<J=6�u�       �	C�Rfc�A�*

lossn��=r.�       �	�,Rfc�A�*

loss�=q��6       �	�Rfc�A�*

loss�>�=R�ae       �	�mRfc�A�*

lossX'>k�<r       �	ARfc�A�*

loss��N=��Q�       �	J�Rfc�A�*

loss�m<���       �	�^Rfc�A�*

loss� >��B�       �	��Rfc�A�*

loss-�=8�$�       �	~�Rfc�A�*

loss���<%~=       �	�@Rfc�A�*

loss3�=}�A�       �	��Rfc�A�*

lossd83>7MU-       �	�Rfc�A�*

loss�{�=��T       �	�Rfc�A�*

loss_�4>��.P       �	l�Rfc�A�*

loss�>u���       �	LRfc�A�*

loss���=��^       �	�Rfc�A�*

loss�)�=%���       �	��Rfc�A�*

loss4E=7Z;�       �	3�Rfc�A�*

loss��<�q��       �	\w Rfc�A�*

loss�ظ<��6�       �	#!Rfc�A�*

loss���=�
�)       �	�0"Rfc�A�*

loss��<�)�       �	�#Rfc�A�*

loss:�<Y�       �	��#Rfc�A�*

loss�4�<sG�/       �	f�$Rfc�A�*

lossLE�=Q��/       �	�%Rfc�A�*

loss�/>Tfn�       �	��%Rfc�A�*

loss���=�ş�       �	��&Rfc�A�*

loss��>���~       �	�'Rfc�A�*

lossY�=@�+�       �	��'Rfc�A�*

loss���=d��e       �	�h(Rfc�A�*

loss�ũ=�q��       �	x
)Rfc�A�*

lossa�D=��%�       �	�)Rfc�A�*

loss�i�=�G�       �	?V*Rfc�A�*

loss,Z�=�aO       �	��*Rfc�A�*

lossHi�=U��#       �	_�+Rfc�A�*

loss��T<s���       �	�1,Rfc�A�*

losse�>�W�       �	:�,Rfc�A�*

loss��=c�T       �	�f-Rfc�A�*

loss�q=-F�#       �	[.Rfc�A�*

loss�Ü=�p�1       �	�/Rfc�A�*

loss�\�=�3]       �	Q�/Rfc�A�*

loss��=��.b       �	=0Rfc�A�*

loss��E=f�Wp       �	�0Rfc�A�*

lossWe<�       �	o�1Rfc�A�*

lossxV�=�D�       �	%<2Rfc�A�*

loss߷�=���!       �	4�2Rfc�A�*

loss�Q=���P       �	�j3Rfc�A�*

loss�!>�t�       �	~4Rfc�A�*

loss�x�=��
       �	��4Rfc�A�*

loss��3=V��       �	5A5Rfc�A�*

loss�;/>�Ӱ�       �	g�5Rfc�A�*

loss�{�=3g��       �	]m6Rfc�A�*

lossK9=�[Y       �	t7Rfc�A�*

loss�5�=���       �	U�7Rfc�A�*

loss3�8>\���       �	�58Rfc�A�*

lossM��=�q6       �	��8Rfc�A�*

loss]}q=       �	�r9Rfc�A�*

loss�S�=����       �	�:Rfc�A�*

loss�c�=h���       �	z�:Rfc�A�*

lossx�]=��2(       �	,D;Rfc�A�*

loss�=��h�       �	��;Rfc�A�*

lossWx�<l2�       �	��=Rfc�A�*

loss�Q2=d�Yt       �	�r>Rfc�A�*

loss�i>�L'}       �	?Rfc�A�*

loss�k=v��Q       �	��?Rfc�A�*

loss��N=9�-P       �	4M@Rfc�A�*

loss;�'>�`�       �	�@Rfc�A�*

loss��v=ⳳ�       �	߇ARfc�A�*

loss�%<�\f       �	�"BRfc�A�*

lossf�A=��4�       �	��BRfc�A�*

loss�Z=
��       �	�jCRfc�A�*

loss���=�$��       �	kDRfc�A�*

loss�L�=wC       �	£DRfc�A�*

lossL\>��6�       �	�MERfc�A�*

loss���=��Bq       �	��ERfc�A�*

loss��=����       �	�FRfc�A�*

lossV�=O��       �	:GRfc�A�*

loss�=ˏ_       �	;HRfc�A�*

loss�x�=���       �	�HRfc�A�*

loss�=�a�       �	BzIRfc�A�*

loss�{=�<��       �	�JRfc�A�*

loss�}=.�9�       �	��JRfc�A�*

loss"7=r�'�       �	BKRfc�A�*

lossz�=W��       �	��KRfc�A�*

loss�K}=,�       �	�wLRfc�A�*

lossS^%=��X�       �	oMRfc�A�*

loss�=�'�K       �	�MRfc�A�*

lossR�c=��E       �	�=NRfc�A�*

lossCv�=7��       �	c�NRfc�A�*

loss4ŧ=�t��       �	sORfc�A�*

loss�?=O`!�       �	PRfc�A�*

loss�=ox�       �	R�PRfc�A�*

lossZ~�=^���       �	�XQRfc�A�*

loss��=A       �	��QRfc�A�*

lossn��=���Q       �	��RRfc�A�*

loss��=1�S�       �	d#SRfc�A�*

loss�ؗ<(N�t       �	�SRfc�A�*

loss�=��g�       �	�OTRfc�A�*

loss��=S��       �	T�TRfc�A�*

loss7q=�ܚ�       �	�yURfc�A�*

lossO!>�T~       �	�VRfc�A�*

loss�7=��       �	;�VRfc�A�*

loss`|�=W��_       �	h?WRfc�A�*

loss�=�/       �	��WRfc�A�*

lossp�>;�       �	0�XRfc�A�*

loss��>�]Z�       �	�YRfc�A�*

loss��
>'�T       �	1�YRfc�A�*

loss�DQ=��       �	fIZRfc�A�*

lossj!>�*9       �	��ZRfc�A�*

loss��#=�r]       �	Jy[Rfc�A�*

loss.�=�09�       �	�\Rfc�A�*

loss��=��i�       �	��\Rfc�A�*

loss�<��       �	�@]Rfc�A�*

loss�=P�       �	6^Rfc�A�*

loss��8=���`       �	$�^Rfc�A�*

loss��]=
��       �	=�_Rfc�A�*

loss�˶=� 	&       �	w�`Rfc�A�*

loss��=�}t�       �	d=aRfc�A�*

loss���=d��;       �	` bRfc�A�*

loss=Ê=��Mp       �	øbRfc�A�*

loss1�=g7�       �	�UcRfc�A�*

loss���=-��       �	p�cRfc�A�*

loss�ۯ=��_       �	�dRfc�A�*

loss��/=I3	�       �	d;eRfc�A�*

lossX=A>R
K       �	�fRfc�A�*

lossK"
=$��h       �	��fRfc�A�*

loss�Lg=c@#       �	!�gRfc�A�*

loss �8=q*|A       �	rphRfc�A�*

lossS|>�>��       �	_�iRfc�A�*

lossZ�J=�>�       �	�|jRfc�A�*

lossIi==�Q�       �	3�kRfc�A�*

loss��Z=}b�       �	I�lRfc�A�*

loss�H4=��ۃ       �	�]mRfc�A�*

loss��P=ʙ�       �	�	nRfc�A�*

loss䠨=>�=�       �	'�nRfc�A�*

lossW�=��>�       �	`pRfc�A�*

loss��=��u       �	2�pRfc�A�*

loss=4��E       �	��qRfc�A�*

loss(*>�R̈́       �	�?rRfc�A�*

lossԮ�=�=q�       �	��rRfc�A�*

loss��>|,��       �	!xsRfc�A�*

loss��=�]�-       �	,tRfc�A�*

loss�
�=g��       �	�tRfc�A�*

loss�.>��Rh       �	=IuRfc�A�*

lossW�>{y�x       �	*�uRfc�A�*

loss��D<`r�[       �	!vvRfc�A�*

loss i=+F�       �	�NwRfc�A�*

loss�=Z�       �	��wRfc�A�*

loss3�`=p�BV       �	�xRfc�A�*

loss��<=u�       �	ٗyRfc�A�*

loss �=Z0�       �	�-zRfc�A�*

lossm�=ŵ&�       �	*�zRfc�A�*

lossZ}<>�E�       �	��{Rfc�A�*

lossd��=�s�       �	X�|Rfc�A�*

loss�=>0�T       �	�(}Rfc�A�*

loss=��=��"       �	/�}Rfc�A�*

loss[��=�E@       �	mU~Rfc�A�*

loss*�=���       �	��~Rfc�A�*

loss��@=�T�       �	��Rfc�A�*

loss�=@/       �	�Rfc�A�*

lossd�=�\��       �	Rfc�A�*

loss�9�<0{Kk       �	F�Rfc�A�*

loss3^S=�/R�       �	�فRfc�A�*

loss��t=x>x       �	zq�Rfc�A�*

loss�=�<�`J       �	R�Rfc�A�*

lossT|�<^���       �	���Rfc�A�*

lossv�U<�r�       �	�=�Rfc�A�*

losssi=SHr~       �	�фRfc�A�*

loss�=��͈       �	h�Rfc�A�*

loss%f�=�b       �	]��Rfc�A�*

lossj	&>ؖ�O       �	��Rfc�A�*

loss)�='*��       �	�@�Rfc�A�*

loss�*�=��       �	��Rfc�A�*

lossCp�;gJ�       �	�|�Rfc�A�*

loss�!�=kr �       �	��Rfc�A�*

loss�k>��&       �	שּRfc�A�*

lossƐ=�U|       �	RE�Rfc�A�*

loss���=C]�       �	�ڊRfc�A�*

loss�6(=���       �	<��Rfc�A�*

loss��`=7�Y~       �	��Rfc�A�*

losst�J=��>f       �	S��Rfc�A�*

loss&�<3�4#       �	9F�Rfc�A�*

loss��=�vo�       �	��Rfc�A�*

loss��=�i�J       �	-��Rfc�A�*

loss�]�=���       �	U1�Rfc�A�*

loss��>�b4       �	�ЏRfc�A�*

loss�v=���       �	Ku�Rfc�A�*

lossԙX=l�Z       �	��Rfc�A�*

loss�r�=F�N       �	���Rfc�A�*

loss��=�]�_       �	EK�Rfc�A�*

loss��=�c�       �	G�Rfc�A�*

loss!�a=�߃�       �	��Rfc�A�*

lossA�=�p�^       �	`�Rfc�A�*

loss�W�=����       �	˹�Rfc�A�*

loss+�<�Ġ       �	TV�Rfc�A�*

loss@{k=,��       �	��Rfc�A�*

loss��=���l       �	b��Rfc�A�*

lossԍ�=i���       �	��Rfc�A�*

loss�</��g       �	���Rfc�A�*

loss ��=uʫ�       �	�V�Rfc�A�*

loss�X4=�H��       �	s�Rfc�A�*

loss팻=����       �	���Rfc�A�*

lossZ1�=.�lU       �	�'�Rfc�A�*

loss��?=��       �	ŚRfc�A�*

lossֶ�=�18C       �	[�Rfc�A�*

loss}q=����       �	=�Rfc�A�*

loss�J�=�ʭ�       �	��Rfc�A�*

loss25=Z[��       �	�!�Rfc�A�*

loss�q�>Z*��       �	亝Rfc�A�*

loss�`�>�k�T       �	�U�Rfc�A�*

loss���=����       �	 �Rfc�A�*

loss)�*=<�c       �	���Rfc�A�*

lossX|=�L�       �	�Rfc�A�*

loss���=@+��       �	��Rfc�A�*

loss�S=3��]       �	T �Rfc�A�*

lossAS�<��       �	��Rfc�A�*

loss�=�Y�       �	*��Rfc�A�*

loss@�3=w���       �	K�Rfc�A�*

lossO�C>u��       �	>�Rfc�A�*

loss=�=��V�       �	τ�Rfc�A�*

loss	J�=`�-�       �	j�Rfc�A�*

loss�Q�<~#�       �	���Rfc�A�*

loss��,=��p       �	D�Rfc�A�*

loss�>�=���e       �	��Rfc�A�*

loss�m=D���       �	���Rfc�A�*

loss<�>VS       �	F��Rfc�A�*

lossV�<�Z�C       �	+M�Rfc�A�*

loss�	
=B`��       �	��Rfc�A�*

lossZ�<>#'       �	��Rfc�A�*

lossay�=X�tF       �	��Rfc�A�*

loss�<="�N�       �	o��Rfc�A�*

loss1�q=Ҩz�       �	'i�Rfc�A�*

losse	]=K�$F       �	�,�Rfc�A�*

loss��s=:��
       �	�ƮRfc�A�*

loss��=�c"�       �	�d�Rfc�A�*

loss��N=MD��       �	��Rfc�A�*

lossh�=��       �	s��Rfc�A�*

loss�~=?��       �	=F�Rfc�A�*

loss(*=Cpr�       �	�Rfc�A�*

loss�<p<s�|T       �	���Rfc�A�*

loss�OH=ELVx       �	t(�Rfc�A�*

loss)�7="1o{       �	�³Rfc�A�*

loss�F\=���       �	�a�Rfc�A�*

loss�;U=["��       �	���Rfc�A�*

lossxT�=tF�Q       �	���Rfc�A�*

loss��,=�\�       �	�(�Rfc�A�*

loss���<��p�       �	��Rfc�A�*

lossQf=��°       �	ߊ�Rfc�A�*

loss���=L�       �	�#�Rfc�A�*

loss�e�<6v��       �	���Rfc�A�*

lossJgv=���       �	���Rfc�A�*

loss���;�-;       �	kG�Rfc�A�*

loss��N=���W       �	�ݺRfc�A�*

loss�T�<%�%       �	�q�Rfc�A�*

loss��<.a        �	�Rfc�A�*

loss�^< �o�       �	���Rfc�A�*

loss,D>ǖb       �	}<�Rfc�A�*

loss�8>��)       �	ҽRfc�A�*

loss�=0��A       �	Ul�Rfc�A�*

loss�@\=
>�<       �	��Rfc�A�*

loss�6�=��       �	���Rfc�A�*

lossW��<8�ъ       �	�2�Rfc�A�*

lossw�8:����       �	%��Rfc�A�*

lossO��;
o��       �	�`�Rfc�A�*

loss��;<� #;       �	I��Rfc�A�*

loss��7=z�        �	��Rfc�A�*

loss�s@<U��)       �	U.�Rfc�A�*

loss�<��>N       �	���Rfc�A�*

loss��J=� ��       �	�T�Rfc�A�*

loss�U=c��}       �	���Rfc�A�*

lossQ�:C�       �	Έ�Rfc�A�*

loss:o<*���       �	9%�Rfc�A�*

loss���<�'       �	T��Rfc�A�*

loss�;�=Rl�       �	^e�Rfc�A�*

loss�%=F�Z       �	���Rfc�A�*

loss��;�Ic}       �	>��Rfc�A�*

lossa�$=W��       �	�/�Rfc�A�*

loss��>Uk�       �	���Rfc�A�*

loss�<�;4�N�       �	�g�Rfc�A�*

loss���<L��       �	�Rfc�A�*

loss��=ګ��       �	���Rfc�A�	*

loss�� >���       �	W]�Rfc�A�	*

loss�UF=V�#�       �	�	�Rfc�A�	*

loss�;K=d��       �	���Rfc�A�	*

loss�=�;I       �	}=�Rfc�A�	*

loss��=0;��       �	���Rfc�A�	*

loss�n�=�v        �	�t�Rfc�A�	*

lossK�=��<�       �	_&�Rfc�A�	*

loss䟗=� �7       �	���Rfc�A�	*

loss�>�)�6       �	(c�Rfc�A�	*

loss&��=�{       �	$
�Rfc�A�	*

loss���=/�Ά       �	v��Rfc�A�	*

lossX��=��9#       �	+N�Rfc�A�	*

loss^>>d��       �	���Rfc�A�	*

loss�!=��w�       �	4��Rfc�A�	*

loss�!>�K�       �	��Rfc�A�	*

loss��>�� �       �	���Rfc�A�	*

lossA;�<B.Z       �	6W�Rfc�A�	*

loss��<�S��       �	7��Rfc�A�	*

lossC��=���       �	���Rfc�A�	*

loss/��=��i�       �	a4�Rfc�A�	*

losstt<���U       �	���Rfc�A�	*

loss��3=|�f�       �	��Rfc�A�	*

lossW><�       �	�7�Rfc�A�	*

loss�J=ǧmb       �	u��Rfc�A�	*

loss#�,=Q       �	�p�Rfc�A�	*

loss���=�|F�       �	��Rfc�A�	*

lossK>���       �	*��Rfc�A�	*

loss���<��%       �	�B�Rfc�A�	*

lossߚ?=b�̜       �	���Rfc�A�	*

loss��Q=���       �	��Rfc�A�	*

loss��<       �	hA�Rfc�A�	*

loss**x=`��<       �	%�Rfc�A�	*

loss��=�)1       �	���Rfc�A�	*

loss!=<�Nk       �	ޫ�Rfc�A�	*

lossJ��=       �	PS�Rfc�A�	*

loss�	�=��.&       �	�b�Rfc�A�	*

lossn#{=�"��       �	%Y�Rfc�A�	*

loss��<��O�       �	��Rfc�A�	*

loss3�<W��<       �	���Rfc�A�	*

loss�O�={O�       �	�/�Rfc�A�	*

lossR9�=�6��       �	���Rfc�A�	*

loss=��=�5f�       �	���Rfc�A�	*

loss�f�=��I       �	�o�Rfc�A�	*

loss�>ج;       �	��Rfc�A�	*

loss�g�<!N��       �	���Rfc�A�	*

loss�1�=�V�       �	�W�Rfc�A�	*

loss@�<��       �		��Rfc�A�	*

loss�i<�
�       �	��Rfc�A�	*

losswsO=g-��       �	�jSfc�A�	*

loss߆U=q:��       �	�Sfc�A�	*

losshʱ=�&��       �	��Sfc�A�	*

loss�">�O]1       �	
1Sfc�A�	*

loss�@=���       �	��Sfc�A�	*

lossC�/=~�`       �	��Sfc�A�	*

lossa�=��       �	�Sfc�A�	*

lossD[�=ZO"�       �	`�Sfc�A�	*

loss���=qH�b       �	ESfc�A�	*

loss���=�n�N       �	��Sfc�A�	*

lossr=Z���       �	Ww	Sfc�A�	*

loss�5=�͍\       �	V
Sfc�A�	*

lossѨ�=��!]       �	t�
Sfc�A�	*

lossה=�p��       �	�ISfc�A�	*

loss,H:=iP]       �	�Sfc�A�	*

loss���=���       �	wSfc�A�	*

loss��<;"cq       �	�
Sfc�A�	*

lossv�=qׄ       �	��Sfc�A�	*

loss�`�<~c��       �	?Sfc�A�	*

loss�i>��S^       �	q�Sfc�A�	*

losswQd=%ZFm       �	OySfc�A�	*

loss~��=�T��       �	Sfc�A�	*

loss�8 =p�	       �	5�Sfc�A�	*

loss�=���       �	�FSfc�A�	*

loss���<=ь�       �	�Sfc�A�	*

lossq�=�e       �	ƈSfc�A�	*

loss �=%���       �	3nSfc�A�	*

loss�{�<B�xB       �	�Sfc�A�	*

loss��'= �|�       �	n�Sfc�A�	*

loss��I=3ǝ"       �	�ASfc�A�	*

loss�Y�<��S�       �	��Sfc�A�	*

loss6��<?��       �	mpSfc�A�	*

lossMX=�x
       �	�Sfc�A�	*

loss��(>:V$�       �	��Sfc�A�	*

loss ��< �b       �	�CSfc�A�	*

loss���=�K��       �	��Sfc�A�	*

lossZ�=Ţh       �	niSfc�A�	*

loss%�#>�7N�       �	n�Sfc�A�	*

loss .�=�X       �	��Sfc�A�	*

loss$3�=�Mm�       �	�,Sfc�A�	*

loss��b=�\�|       �	��Sfc�A�	*

lossM�=yo�'       �	�\Sfc�A�	*

loss3��<E��       �	��Sfc�A�	*

loss��=|�r}       �	��Sfc�A�	*

loss�u�=��\       �	��Sfc�A�	*

loss��,>�@U6       �	��Sfc�A�	*

loss��<C��       �	S< Sfc�A�	*

loss��=�o�_       �	k� Sfc�A�	*

loss�`�=t�`�       �	�!Sfc�A�	*

loss��<v\�       �	?"Sfc�A�	*

lossM��<%��m       �	��"Sfc�A�	*

loss���=j7��       �	Qk#Sfc�A�	*

lossx8�<��D       �	��#Sfc�A�	*

lossڡ'>wD�e       �	g�$Sfc�A�	*

loss�O;=���c       �	�2%Sfc�A�	*

lossX�<D�       �	�&Sfc�A�	*

loss�`�;*B|       �	פ&Sfc�A�	*

loss;�-<w S7       �	�B'Sfc�A�	*

loss�=��R�       �	��'Sfc�A�	*

lossMS�=1�       �	�(Sfc�A�	*

lossa��=m�V       �	�)Sfc�A�	*

loss#�<<b"�       �	��)Sfc�A�	*

loss$cj<�!v       �	�a*Sfc�A�	*

loss^�<�=��       �	t+Sfc�A�	*

loss?�E=Ya9       �	+�+Sfc�A�	*

loss�-�<�
�L       �	}=,Sfc�A�	*

loss�ŷ=�g��       �	-Sfc�A�	*

loss��o=9��-       �	Ͱ-Sfc�A�	*

loss1{�=� �       �	J^.Sfc�A�	*

loss�&�=
��       �	k�.Sfc�A�	*

loss�b>=����       �	A�/Sfc�A�	*

lossH�m=ʯo�       �	�H0Sfc�A�	*

loss;m=zcJ       �	=�0Sfc�A�	*

lossR�8=y,>�       �	[�1Sfc�A�	*

loss}�i=�խU       �	S@2Sfc�A�	*

loss��= yy�       �	��2Sfc�A�	*

loss��=����       �	��3Sfc�A�	*

loss6�>�ż�       �	?4Sfc�A�	*

loss��<�N�4       �	6�4Sfc�A�	*

loss�Q�=�1�       �	�5Sfc�A�
*

loss���=�k��       �	�*6Sfc�A�
*

loss�X�=���       �	��6Sfc�A�
*

loss�y;=e]l[       �	r7Sfc�A�
*

lossfL=�b�       �	s8Sfc�A�
*

loss���=΄��       �	q�8Sfc�A�
*

loss�=ͣ��       �	IL9Sfc�A�
*

loss���=���       �	Y�9Sfc�A�
*

loss.ө=0j�       �	��:Sfc�A�
*

loss��=       �	�`;Sfc�A�
*

loss���<*n��       �	3�;Sfc�A�
*

loss�;=0���       �	 �<Sfc�A�
*

loss_d�=1�V       �	`#=Sfc�A�
*

loss��=�ű�       �	a�=Sfc�A�
*

loss a=��b�       �	bj>Sfc�A�
*

loss��=9=�       �	�?Sfc�A�
*

loss�D7=�H��       �	z�?Sfc�A�
*

loss���<'W��       �	5^@Sfc�A�
*

lossC90=���       �	r�@Sfc�A�
*

loss��>;�8:       �	I�ASfc�A�
*

lossl��<9�v       �	P8BSfc�A�
*

loss6]
=ŗ�+       �	��BSfc�A�
*

loss���<�[.       �	~nCSfc�A�
*

loss��<@�A&       �	1DSfc�A�
*

lossO��<�"O       �	�DSfc�A�
*

lossm�?=�5       �	�KESfc�A�
*

loss�>?^�       �	x�ESfc�A�
*

loss-<%�'       �	�FSfc�A�
*

loss	�7=� �       �	5GSfc�A�
*

loss���=�[δ       �	��GSfc�A�
*

loss��<��sk       �	L�HSfc�A�
*

loss�>��       �	.7ISfc�A�
*

lossm�=��WT       �	��ISfc�A�
*

loss���=f$aU       �	b�JSfc�A�
*

lossܹ�<]}�A       �	�rKSfc�A�
*

loss�ph=@Jn7       �	�LSfc�A�
*

loss�S#<715       �	}�LSfc�A�
*

loss�ϑ=R�7       �	?TMSfc�A�
*

loss���=j��       �	��MSfc�A�
*

loss��g=.)�       �	��NSfc�A�
*

loss�X=�)r       �	)OSfc�A�
*

lossL��<C��       �	��OSfc�A�
*

lossȧ�=�X�^       �	�UPSfc�A�
*

loss �=,
       �	;�PSfc�A�
*

loss3�=�A�       �	��QSfc�A�
*

loss���<P�3       �	n1RSfc�A�
*

loss��=�p�       �	��RSfc�A�
*

lossd�<Gmw�       �	N~SSfc�A�
*

loss�q�<���6       �	�TSfc�A�
*

loss@��=Acw       �	 �TSfc�A�
*

loss�]=��V�       �	4LUSfc�A�
*

lossF�=��(       �	��USfc�A�
*

lossҴ]=��ֈ       �	g{VSfc�A�
*

loss)�==�p7D       �	�"WSfc�A�
*

lossv3�<��       �	U�WSfc�A�
*

loss	�>�/M       �	�UXSfc�A�
*

losse� =@���       �	��XSfc�A�
*

lossܼ=o&�z       �	T�YSfc�A�
*

lossn =��       �	�2ZSfc�A�
*

losst��<g���       �	�ZSfc�A�
*

loss=u�<�kP�       �	_{[Sfc�A�
*

loss���<4G&�       �	�!\Sfc�A�
*

loss\��<ru�0       �	E�\Sfc�A�
*

lossJP=�T�       �	�Y]Sfc�A�
*

loss�>�?+�       �	A�]Sfc�A�
*

loss���=��k�       �	��^Sfc�A�
*

lossN��<�D       �	�+_Sfc�A�
*

lossE	�=x�r-       �	`V`Sfc�A�
*

loss�I<�Dݼ       �	PaSfc�A�
*

loss�փ=�g�       �	:�aSfc�A�
*

lossV��<8���       �	^�bSfc�A�
*

lossXm�<��{?       �	�8cSfc�A�
*

lossTc�<��z;       �	eodSfc�A�
*

loss�?D=q+K       �	tAeSfc�A�
*

loss<�_=�T��       �	��eSfc�A�
*

loss7�=�e       �	�wfSfc�A�
*

loss�u�;4	�\       �	�gSfc�A�
*

loss
s=m��h       �	h�gSfc�A�
*

loss���=���       �	@MhSfc�A�
*

loss|�=���"       �	��hSfc�A�
*

lossr�U<�zAx       �	ߧiSfc�A�
*

loss�|�=�Xy�       �	EjSfc�A�
*

loss�<�,-�       �	��jSfc�A�
*

loss\J=��;;       �	��kSfc�A�
*

loss�X<*�%F       �	�;lSfc�A�
*

loss�c=�D3�       �	��lSfc�A�
*

loss{*�<�v�       �	5{mSfc�A�
*

lossq�*<{�y�       �	�nSfc�A�
*

loss��=M���       �	��nSfc�A�
*

loss��p<�x��       �	MJoSfc�A�
*

loss-�=1�       �	��oSfc�A�
*

lossT�/=��@�       �	ӅpSfc�A�
*

lossMi<�9G1       �	��qSfc�A�
*

loss]w<�?<       �	�prSfc�A�
*

loss��[=�F��       �	<sSfc�A�
*

loss���<$�q       �	NtSfc�A�
*

loss�+=�@�       �	��tSfc�A�
*

lossO��=�ަ�       �	�$vSfc�A�
*

loss���=�jH�       �	i�vSfc�A�
*

lossR"7=��3�       �	�fwSfc�A�
*

loss�U<�9��       �	�xSfc�A�
*

losslc�;�t��       �	�xSfc�A�
*

loss��=��c�       �	cFySfc�A�
*

loss�]=��߇       �	�ySfc�A�
*

loss���;�J+�       �	'�zSfc�A�
*

loss�e�<��J       �	�&{Sfc�A�
*

loss�7=���       �	��{Sfc�A�
*

loss��X= ���       �	]o|Sfc�A�
*

loss&̝<�aA       �	�}Sfc�A�
*

lossڂ�<(�C       �	��}Sfc�A�
*

loss�ԁ=Uj��       �	�P~Sfc�A�
*

loss5�>
PY       �	��~Sfc�A�
*

loss��+=͝7W       �	{�Sfc�A�
*

loss�d-<�{�S       �	u�Sfc�A�
*

loss�ü<QW       �	G�Sfc�A�
*

loss?�<!�z�       �	��Sfc�A�
*

loss1�<-�b�       �	fM�Sfc�A�
*

loss��<v��       �	)�Sfc�A�
*

loss��=�P6       �	��Sfc�A�
*

lossğ�=j�Q       �	S$�Sfc�A�
*

loss�<>�b       �	��Sfc�A�
*

loss=��=��        �	~R�Sfc�A�
*

lossD\	=L2Ƌ       �	O�Sfc�A�
*

lossD =Ώ�F       �	��Sfc�A�
*

loss��=m6Nm       �	�*�Sfc�A�
*

loss+<��]       �	ɇSfc�A�
*

loss��;���       �	�`�Sfc�A�
*

loss�c<P��x       �	��Sfc�A�
*

lossI�=>���       �	���Sfc�A�*

lossa�g=�u�        �	'/�Sfc�A�*

lossSpL=��#r       �	uɊSfc�A�*

loss�a=�Lw       �	`�Sfc�A�*

loss�é=�i       �	���Sfc�A�*

loss���<ۆ��       �	K��Sfc�A�*

loss6˵<:ǐ       �	o.�Sfc�A�*

lossϦ=i@-�       �	�ɍSfc�A�*

loss8S>�
�       �	oe�Sfc�A�*

loss��=�<h�       �	� �Sfc�A�*

loss��=��       �	��Sfc�A�*

loss:�=��       �	�?�Sfc�A�*

loss�UV=���n       �	'ېSfc�A�*

loss�O#<��\       �	�s�Sfc�A�*

loss�A�=x��l       �	c�Sfc�A�*

lossa��<���       �	��Sfc�A�*

loss<��=�[��       �	I�Sfc�A�*

loss e3=��?�       �	��Sfc�A�*

loss!�==KP]�       �	(��Sfc�A�*

lossR!y=��|&       �	�Sfc�A�*

loss8�X=�	       �	g��Sfc�A�*

loss��8=s�K       �	�P�Sfc�A�*

loss;��<
��a       �	W�Sfc�A�*

loss�2�<���       �	��Sfc�A�*

loss&<�-J       �	�'�Sfc�A�*

loss36�=b��X       �	K˘Sfc�A�*

loss��<}}��       �	1z�Sfc�A�*

lossؒ]=Q@)�       �	�)�Sfc�A�*

loss&�d=ޕ       �	�ƚSfc�A�*

loss�=��Ѳ       �	�e�Sfc�A�*

loss�> H��       �	���Sfc�A�*

loss#�=o�|       �	ɓ�Sfc�A�*

loss�ڒ<�9��       �	�5�Sfc�A�*

loss�+=\�       �	$ѝSfc�A�*

loss:�=0��g       �	�x�Sfc�A�*

losslwG=[��H       �	��Sfc�A�*

loss��F=����       �	��Sfc�A�*

lossc�[=d�?       �	J�Sfc�A�*

loss��=�'��       �	4�Sfc�A�*

lossf��=~[�       �	T��Sfc�A�*

losse=*=2�i�       �	�"�Sfc�A�*

loss5��=Bj�{       �	0��Sfc�A�*

lossO��<�<E:       �	4��Sfc�A�*

loss�\=$��       �	��Sfc�A�*

lossw�$>�&]       �	ꮤSfc�A�*

lossX��<��^       �	�O�Sfc�A�*

lossn�@<ZL�j       �	��Sfc�A�*

loss(,=���       �	�Sfc�A�*

loss��<6��       �	�%�Sfc�A�*

loss�2=+Ҩ�       �	�˧Sfc�A�*

loss6�C=F��       �	�k�Sfc�A�*

loss\�=h.��       �	(�Sfc�A�*

loss{�=��t�       �	3��Sfc�A�*

loss�s�=�4|�       �	�F�Sfc�A�*

loss�R�<�-Cu       �	e�Sfc�A�*

losszR>��y�       �	�}�Sfc�A�*

lossj�=4I��       �	��Sfc�A�*

loss��=��T       �	���Sfc�A�*

lossC�G<�vN       �	�N�Sfc�A�*

loss�i�=�k�       �	��Sfc�A�*

loss|�Y=T~��       �	Q��Sfc�A�*

loss͗�=#�       �	5$�Sfc�A�*

loss��=��Hi       �	���Sfc�A�*

loss~e=u�(B       �	:X�Sfc�A�*

loss�Pm=Ma{�       �	s�Sfc�A�*

lossS�='��       �	���Sfc�A�*

loss�=��Y       �	�&�Sfc�A�*

loss�y;=NMu�       �	ͲSfc�A�*

loss�v=�!��       �	�g�Sfc�A�*

lossVt�<w$f�       �	)�Sfc�A�*

loss��=��`       �	���Sfc�A�*

lossV��=,��M       �	Cq�Sfc�A�*

loss��=�2H       �	�Sfc�A�*

loss#i<�}�)       �	���Sfc�A�*

lossRx�<n��6       �	�>�Sfc�A�*

lossE�#=q�n8       �	��Sfc�A�*

loss�:6<��=`       �	���Sfc�A�*

loss�K>y@�       �	�+�Sfc�A�*

loss�=P�       �	��Sfc�A�*

lossM@L=B��       �	HźSfc�A�*

loss/�T=-���       �	/i�Sfc�A�*

lossp��<cw�o       �	��Sfc�A�*

loss ��;A	vJ       �	���Sfc�A�*

lossM��;T���       �	O�Sfc�A�*

loss�ǔ=����       �	��Sfc�A�*

loss���;��U       �	��Sfc�A�*

loss�><r�=       �	�/�Sfc�A�*

lossT�)=h�&�       �	�ɿSfc�A�*

loss��}=�?�       �	\q�Sfc�A�*

lossй<�m�[       �	U�Sfc�A�*

loss���=o��"       �	��Sfc�A�*

lossVC>݂ѹ       �	�A�Sfc�A�*

loss7#�=�t��       �	���Sfc�A�*

loss�1=�<�       �	Lq�Sfc�A�*

loss�=�1��       �	�
�Sfc�A�*

loss��<��~�       �	ʧ�Sfc�A�*

loss(ֵ=��K�       �	�[�Sfc�A�*

lossW:�<�o�       �	���Sfc�A�*

lossd��=f}��       �	���Sfc�A�*

lossX�=<u�       �	Z-�Sfc�A�*

loss7V=�r�       �	��Sfc�A�*

loss�<��۪       �	�V�Sfc�A�*

loss�<�<�A�       �	���Sfc�A�*

loss���<��b�       �	%��Sfc�A�*

loss[xt=��ve       �	�2�Sfc�A�*

loss�R=���+       �	���Sfc�A�*

loss/T<�u�8       �	Lm�Sfc�A�*

loss�-"=Z�D       �	�=�Sfc�A�*

lossAj�=�g�       �	C��Sfc�A�*

loss]�n<K 77       �	��Sfc�A�*

lossn!;=v�}g       �	R(�Sfc�A�*

lossic�=%�^]       �	��Sfc�A�*

loss�6I=�t       �	f�Sfc�A�*

loss�<��       �	ܛ�Sfc�A�*

lossOa�=����       �	�?�Sfc�A�*

loss�OA=ג       �	<��Sfc�A�*

lossdEE==�F       �	�y�Sfc�A�*

loss�9N=��       �	`#�Sfc�A�*

loss�=` ��       �	��Sfc�A�*

loss�I�=��T       �	-^�Sfc�A�*

lossh��=�Ɖ       �	M��Sfc�A�*

lossx�U=<�       �	`��Sfc�A�*

lossn��=.O	�       �	�*�Sfc�A�*

loss�YE=C�?S       �	'��Sfc�A�*

loss]��<�8qc       �	ka�Sfc�A�*

loss_^�<���       �	n�Sfc�A�*

loss��<	#�       �	h��Sfc�A�*

loss�O=��<�       �	�R�Sfc�A�*

loss��c=����       �	���Sfc�A�*

loss���<a�9�       �	���Sfc�A�*

lossn͒=:C�       �	��Sfc�A�*

lossԧ�=��_       �	4��Sfc�A�*

loss�H-<+���       �	sJ�Sfc�A�*

losss~W;"��a       �	���Sfc�A�*

loss N=y��       �	x�Sfc�A�*

loss���=�g�       �	��Sfc�A�*

lossR>�=4ݩ�       �	���Sfc�A�*

lossxD->����       �	�B�Sfc�A�*

loss���=a��       �	n��Sfc�A�*

loss\��<$�.�       �	G;�Sfc�A�*

loss��x=�j�       �	@��Sfc�A�*

losstW<?��Q       �	�x�Sfc�A�*

loss��=y�ub       �	��Sfc�A�*

loss:.�<���Q       �	���Sfc�A�*

loss��<���       �	E�Sfc�A�*

loss4X>=#��9       �	��Sfc�A�*

loss�.3=���       �	�u�Sfc�A�*

loss1��=�c       �	��Sfc�A�*

loss6�<k��O       �	��Sfc�A�*

loss��=�7�{       �	�N�Sfc�A�*

loss�
=�-�$       �	\��Sfc�A�*

losso8=f       �	�y�Sfc�A�*

losso�=sy��       �	��Sfc�A�*

loss[�=�Uf[       �	���Sfc�A�*

loss���<�t�.       �	�P�Sfc�A�*

loss�<� O       �	���Sfc�A�*

losss�=y�NI       �	c�Sfc�A�*

loss�?�=��F       �	���Sfc�A�*

loss�� =B��s       �	���Sfc�A�*

loss7�=g5�       �	�4�Sfc�A�*

loss�|=<�y�       �	���Sfc�A�*

lossa�W<��       �	��Sfc�A�*

loss��=�/�Y       �	
.�Sfc�A�*

loss!��=�>�       �	|(�Sfc�A�*

loss��	=����       �	q��Sfc�A�*

loss8�=ӬiR       �	���Sfc�A�*

loss�=`q�{       �	�@�Sfc�A�*

loss�3�=H��       �	���Sfc�A�*

loss�	�=��1�       �	��Sfc�A�*

loss���=
�Xo       �	�.�Sfc�A�*

loss��'=ے        �	���Sfc�A�*

lossS�
=�M�h       �	{/�Sfc�A�*

loss�ư=u�T�       �	���Sfc�A�*

loss�z=Q��       �	���Sfc�A�*

loss��2=ش)       �	�e�Sfc�A�*

loss�N2=W	�       �	�Sfc�A�*

loss�8=M�       �	7��Sfc�A�*

loss��;���+       �	,J�Sfc�A�*

loss��=��u       �	���Sfc�A�*

loss�Ռ=�U<       �	��Sfc�A�*

lossF5�<���b       �	v8�Sfc�A�*

lossH<�}�o       �	���Sfc�A�*

loss��u=���       �	y Tfc�A�*

loss���=Z�w�       �	?Tfc�A�*

loss�f=��	       �	
�Tfc�A�*

lossC��<�]��       �	RaTfc�A�*

lossLl4=(V�       �	}Tfc�A�*

loss�:�=�[��       �	y�Tfc�A�*

loss���=-�S\       �	�UTfc�A�*

loss$i�<��+&       �	��Tfc�A�*

loss��;�Q@�       �	��Tfc�A�*

lossqL�<ܫ��       �	s0Tfc�A�*

loss�<Y4�<       �	
�Tfc�A�*

loss^L�<��c�       �	7oTfc�A�*

lossN�<��       �	�Tfc�A�*

loss���=1��&       �	b�Tfc�A�*

lossx/�<�85�       �	!:	Tfc�A�*

loss���=�דs       �	p�	Tfc�A�*

loss䖼=�Q.�       �	�s
Tfc�A�*

loss1�C=M�;       �	�Tfc�A�*

lossVq�<Z�       �	�Tfc�A�*

loss��=�r0�       �	�NTfc�A�*

loss��=\H�Y       �	p�Tfc�A�*

loss�H�=4e`       �	��Tfc�A�*

loss}ػ=�?,�       �	�6Tfc�A�*

loss4�=�T��       �	��Tfc�A�*

lossr�u=�_       �	UjTfc�A�*

loss��=]��k       �	�Tfc�A�*

loss3�p=�v�       �	��Tfc�A�*

lossC�N<���       �	i8Tfc�A�*

lossW#=/�       �	l�Tfc�A�*

lossSoq=��qA       �	hTfc�A�*

lossX�C=�sm%       �	��Tfc�A�*

lossOs<%څ1       �	y�Tfc�A�*

loss=>Z=�5r       �	�(Tfc�A�*

loss��=Br�W       �	��Tfc�A�*

loss�^&>z6       �	tbTfc�A�*

loss�'>�f��       �	'�Tfc�A�*

loss���=�c��       �	��Tfc�A�*

lossDb=��>r       �	/5Tfc�A�*

loss���=��#j       �	�Tfc�A�*

lossz�]=V��3       �	�iTfc�A�*

loss�8U=�(��       �	VTfc�A�*

loss)�=�6��       �	z�Tfc�A�*

loss�,7=Z��i       �	MTfc�A�*

loss��	<اX:       �	��Tfc�A�*

loss��F<�#i�       �	�Tfc�A�*

loss�!=1�p       �	�4Tfc�A�*

loss+='8       �	�Tfc�A�*

lossG =���&       �	pTfc�A�*

loss���<'4�       �	�Tfc�A�*

loss��<��       �	�Tfc�A�*

lossMu�<ɰ�       �	VTfc�A�*

loss mH>�CKi       �	^�Tfc�A�*

loss�g�=�a�       �	�� Tfc�A�*

loss��=�%)       �	�b!Tfc�A�*

loss{(�=)Eh-       �	&�!Tfc�A�*

loss���;��W�       �	��"Tfc�A�*

loss�9*=����       �	�#Tfc�A�*

loss;)>���       �	,�$Tfc�A�*

lossqT�<z��d       �	�%Tfc�A�*

loss�=��/�       �	� &Tfc�A�*

lossή�=	�Uj       �	O?'Tfc�A�*

loss���<�u�W       �	��'Tfc�A�*

lossq0�<o��       �	�r(Tfc�A�*

loss�4G=���S       �	�)Tfc�A�*

lossGH�=�lK       �	��)Tfc�A�*

loss��e=:`��       �	�G*Tfc�A�*

loss�=����       �	��*Tfc�A�*

loss�t=s���       �	6v+Tfc�A�*

loss�:�=y���       �	�,Tfc�A�*

loss"��<L�O�       �	��,Tfc�A�*

loss:E�=�� �       �	�h-Tfc�A�*

loss�K�<opE       �	J.Tfc�A�*

loss��<Ցk�       �	��.Tfc�A�*

lossf�<aDt       �	�2/Tfc�A�*

loss3i�=~�o�       �	��/Tfc�A�*

loss�$=MW68       �	of0Tfc�A�*

losslj�<�zI       �	��0Tfc�A�*

loss��=�0       �	��1Tfc�A�*

loss�٢<�{�|       �	�m2Tfc�A�*

loss��<�_�F       �	G3Tfc�A�*

loss1�1<��t       �	Y�3Tfc�A�*

loss_�=ީS>       �	�B4Tfc�A�*

loss��(=I��'       �	��4Tfc�A�*

loss��R=�V       �	�t5Tfc�A�*

loss��>�p�_       �	6Tfc�A�*

lossmJ=Y�Z�       �	~�6Tfc�A�*

loss���<��Ȕ       �	�A7Tfc�A�*

loss�<3��       �	q�7Tfc�A�*

loss&� =OD�       �	
�8Tfc�A�*

loss��<�.�       �	�9Tfc�A�*

loss��D>:�\7       �	x�9Tfc�A�*

loss��$>���A       �	�:Tfc�A�*

loss<�4=7��#       �	�";Tfc�A�*

loss��	<v�^�       �	��;Tfc�A�*

loss|�=�� �       �	�\<Tfc�A�*

loss�V�<q�       �	��<Tfc�A�*

lossē�<5���       �	ĕ=Tfc�A�*

loss_�=T��s       �	�->Tfc�A�*

lossJ�Y=#F�       �	X�>Tfc�A�*

loss]B�=��-~       �	�[?Tfc�A�*

lossϊ<=�a,       �	��?Tfc�A�*

loss��D=?M	�       �	��@Tfc�A�*

lossq�<O@	       �	t)ATfc�A�*

loss���;.8?5       �	��ATfc�A�*

loss|��<)��I       �	�\BTfc�A�*

loss�c9<C; �       �	��BTfc�A�*

loss.�a<0*'       �	W�CTfc�A�*

loss��=u+/       �	A.DTfc�A�*

losst�<�       �	L�DTfc�A�*

loss�.e=�6�_       �	�aETfc�A�*

loss-��=A5^�       �	 �ETfc�A�*

loss\>}=�Q       �	ٖFTfc�A�*

loss��=8j��       �	8-GTfc�A�*

loss�Ѹ<�4\�       �	��GTfc�A�*

loss�ٔ<[��M       �	�[HTfc�A�*

lossͽ�=��9�       �	��HTfc�A�*

loss�^
=��C       �	y�ITfc�A�*

loss��<$��       �	�0JTfc�A�*

loss�'9<�nM       �	G�JTfc�A�*

lossZ�=7e��       �	�`KTfc�A�*

loss�5=b5�z       �	s�KTfc�A�*

loss�0<�ev�       �	P�LTfc�A�*

loss�` >��s�       �	�%MTfc�A�*

loss���<�M�?       �	�MTfc�A�*

loss�J
>쐜�       �	�ZNTfc�A�*

loss?�e<HZ�       �	{�NTfc�A�*

loss��_=�jx       �	��OTfc�A�*

lossa�=c��       �	c&PTfc�A�*

loss�p�=��HZ       �	˼PTfc�A�*

lossf�;����       �	�XQTfc�A�*

loss�>�=�e       �	��QTfc�A�*

loss�P=�L9<       �	��RTfc�A�*

lossX~�<oԼ0       �	9&STfc�A�*

loss�U�;bs�       �	ԻSTfc�A�*

loss[X�<�|bd       �	�aTTfc�A�*

loss�x�=o�.       �	D�TTfc�A�*

lossj;�<�誢       �	��UTfc�A�*

loss��=��Wu       �	�=VTfc�A�*

lossM$�=]Q�m       �	��VTfc�A�*

loss�9�=n��       �	i�WTfc�A�*

loss�sJ<TO�       �	f0XTfc�A�*

lossY�;�ީ�       �	+�XTfc�A�*

lossĖ=����       �	N~YTfc�A�*

loss�r;��e�       �	�#ZTfc�A�*

loss� <̓�r       �	��ZTfc�A�*

loss���;�V��       �	�o[Tfc�A�*

loss^e�<�<H�       �	�\Tfc�A�*

loss���;���5       �	�\Tfc�A�*

loss[��<�fs       �	#N]Tfc�A�*

loss�[<�&�(       �	��]Tfc�A�*

loss�<pV�       �	΋^Tfc�A�*

loss �{;^�A       �	�(_Tfc�A�*

loss��:��͕       �	��_Tfc�A�*

loss��;��L       �	vm`Tfc�A�*

loss$Ŗ<�'$W       �		aTfc�A�*

loss��=<���       �	A�aTfc�A�*

loss�<���       �	�bTfc�A�*

lossj#;o���       �	�(cTfc�A�*

lossh�<dWc       �	vdTfc�A�*

lossԯ,>��M�       �	deTfc�A�*

loss|�!;��R�       �	}[fTfc�A�*

loss�#=�ƻb       �	#jgTfc�A�*

lossR�H=��k       �	�UhTfc�A�*

loss���<��+@       �	9FiTfc�A�*

loss��=JX�       �	��jTfc�A�*

loss�x2=$;�       �	X�kTfc�A�*

lossx��=��       �	��lTfc�A�*

loss\��=�r�       �	�lmTfc�A�*

loss�t�=(9��       �	�-nTfc�A�*

loss��`=�s��       �	!soTfc�A�*

loss�NK=ރ�H       �	DpTfc�A�*

loss�I6=S        �	�qTfc�A�*

loss&��=��9       �	��qTfc�A�*

loss�`F=󷣨       �	xbrTfc�A�*

lossX�=!|       �	�rTfc�A�*

loss��p=�k
R       �	H�sTfc�A�*

loss�=jl
       �	��tTfc�A�*

loss}=�1�U       �	�,uTfc�A�*

loss{�D=��RY       �	R�uTfc�A�*

losse@=��Ԝ       �	IhvTfc�A�*

loss��s<�_5       �	�wTfc�A�*

loss�cO<��p       �	��wTfc�A�*

lossC>==���       �	*WxTfc�A�*

loss�<�<z��       �	��xTfc�A�*

loss���;O�vv       �	�yTfc�A�*

loss��?;��	       �	�%zTfc�A�*

loss�Z=[���       �	��zTfc�A�*

loss���<�\��       �	c{Tfc�A�*

lossf�=z�8�       �	  |Tfc�A�*

loss��=I�2x       �	��|Tfc�A�*

loss}v~<��;�       �	:<~Tfc�A�*

loss�|=H�n�       �	��~Tfc�A�*

lossW=s��v       �	�yTfc�A�*

loss!0�;�8%�       �	p&�Tfc�A�*

lossh<�ǭ       �	�̀Tfc�A�*

loss��=�˞Z       �	�r�Tfc�A�*

loss��+=}�:       �	��Tfc�A�*

lossL�=3@��       �	���Tfc�A�*

loss3�&=�lZ       �	�Z�Tfc�A�*

loss�|^=�n       �	&��Tfc�A�*

lossA�T<j���       �	0��Tfc�A�*

lossX�<ˇ�Q       �	T5�Tfc�A�*

loss�s2=�       �	�ӅTfc�A�*

loss�
�<tP*�       �	�x�Tfc�A�*

loss1��<����       �	*�Tfc�A�*

lossѩy=��       �	FӇTfc�A�*

lossO��=�E�       �	`t�Tfc�A�*

loss �<*�(�       �	o�Tfc�A�*

loss���=t��       �	��Tfc�A�*

loss_�N<neD       �	;S�Tfc�A�*

lossWO�<WU}�       �	��Tfc�A�*

loss �=��u"       �	i��Tfc�A�*

loss�_�=�X�       �	���Tfc�A�*

loss�9�=��uX       �	z��Tfc�A�*

lossm�'=�F��       �	�/�Tfc�A�*

loss�j�=��X�       �	�צTfc�A�*

loss���<<�x       �	�w�Tfc�A�*

loss�61=R=9�       �	�!�Tfc�A�*

loss�Qx=���       �	BϨTfc�A�*

losss�/=G�/       �	}w�Tfc�A�*

lossA6�=�q��       �	:!�Tfc�A�*

loss�<SH&�       �	r��Tfc�A�*

loss���=���]       �	c�Tfc�A�*

loss�L�=PT�       �	 �Tfc�A�*

loss��=,&��       �	�Tfc�A�*

loss:5�=��p       �	P:�Tfc�A�*

loss���=Z,�       �	�խTfc�A�*

loss�Sn;ًY�       �	&q�Tfc�A�*

lossM�;x[��       �	M��Tfc�A�*

loss�T�<k��v       �	!�Tfc�A�*

lossw=]��P       �	E��Tfc�A�*

loss�đ=-<�       �	�Y�Tfc�A�*

loss�Y�=��       �	#��Tfc�A�*

lossDY=$��9       �	=��Tfc�A�*

loss��==��       �	B?�Tfc�A�*

loss��=��-�       �	�ܳTfc�A�*

loss�C�</�m       �	�t�Tfc�A�*

losse�m=)r�{       �	�ȵTfc�A�*

loss��7=���       �	�a�Tfc�A�*

lossf��=�9b�       �		�Tfc�A�*

loss��=G)8       �	���Tfc�A�*

loss�x=��+5       �	I�Tfc�A�*

loss�t�<i(p       �	�1�Tfc�A�*

loss!�"=���       �	�-�Tfc�A�*

lossr��=V�|�       �	0G�Tfc�A�*

loss{$C<��Y�       �	��Tfc�A�*

loss+�=���       �	��Tfc�A�*

loss���<F�        �	.=�Tfc�A�*

loss�F
=�o>�       �	��Tfc�A�*

loss
��=�� s       �	x�Tfc�A�*

loss-!�<'       �	4�Tfc�A�*

loss%X�<��uu       �	���Tfc�A�*

loss`��<1�m       �	�h�Tfc�A�*

lossW;<Ō�       �	��Tfc�A�*

lossT\K=#1aK       �	\��Tfc�A�*

loss�T�=d�T       �	�G�Tfc�A�*

loss�Ye=����       �	���Tfc�A�*

lossHwB=tN;       �	ǃ�Tfc�A�*

lossD�9=X�9       �	#�Tfc�A�*

loss#�{<X��4       �	>��Tfc�A�*

loss�|�;��D�       �	�i�Tfc�A�*

loss"$�<Ѩ       �	��Tfc�A�*

loss|x_=��^y       �	%��Tfc�A�*

losss!"<
-       �	�H�Tfc�A�*

loss���=�K�       �	���Tfc�A�*

lossl&�<��       �	���Tfc�A�*

loss͖�;FU�       �	&�Tfc�A�*

loss��R:���       �	���Tfc�A�*

loss�<7��+       �	�f�Tfc�A�*

loss�g'=��       �	>�Tfc�A�*

lossO��=J�<       �	���Tfc�A�*

lossR�v=����       �	�f�Tfc�A�*

lossf�6<�h�       �	 �Tfc�A�*

lossr?<�;�{       �	J��Tfc�A�*

lossLu�<��:       �	�4�Tfc�A�*

lossA�<d�`       �	W��Tfc�A�*

loss��e=7ƃO       �	Dn�Tfc�A�*

loss��=��$�       �	$
�Tfc�A�*

loss6x�<��M       �	}��Tfc�A�*

lossWs�=�1��       �	CW�Tfc�A�*

loss�9�=F�͒       �	���Tfc�A�*

loss�,�<w{�       �	���Tfc�A�*

loss�B�<h9       �	�;�Tfc�A�*

loss�!�<n�r�       �	<��Tfc�A�*

loss��=
��[       �	�}�Tfc�A�*

loss�<=Б�       �	�*�Tfc�A�*

loss.A�;�+?)       �	���Tfc�A�*

loss���<�C       �	F~�Tfc�A�*

loss�+�<=Ԋ       �	�"�Tfc�A�*

loss�=�oAa       �	���Tfc�A�*

loss�>s=�X.�       �	j�Tfc�A�*

lossJ)�<��f       �	�B�Tfc�A�*

loss��*=a���       �	���Tfc�A�*

lossU�=IߒT       �	U��Tfc�A�*

loss��Y<�?3       �	w,�Tfc�A�*

loss�.�<ud�t       �	���Tfc�A�*

loss�	=r���       �		k�Tfc�A�*

loss�ɠ=��v-       �	��Tfc�A�*

lossHh�<!��~       �	0��Tfc�A�*

loss���<���P       �	?8�Tfc�A�*

loss2N5=�KZ8       �	r��Tfc�A�*

losspH=m�5       �	H��Tfc�A�*

loss_�n=��       �	���Tfc�A�*

loss�ܳ<���       �	q��Tfc�A�*

lossh�=��       �	ʤ�Tfc�A�*

loss� �=���       �	���Tfc�A�*

loss}��<���r       �	���Tfc�A�*

loss��<<9��       �	U�Tfc�A�*

loss.�4=��#�       �	��Tfc�A�*

losst*,=L��       �	|��Tfc�A�*

loss��=�E�w       �	�R�Tfc�A�*

loss}A <���#       �	@��Tfc�A�*

loss@	�<�	��       �	�;�Tfc�A�*

lossڞ=�v�_       �	���Tfc�A�*

loss)C:<97wT       �	�g�Tfc�A�*

loss&�?=)R��       �	�9�Tfc�A�*

loss�q�=��+�       �	C��Tfc�A�*

lossA F;�CV       �	��Tfc�A�*

loss_�<]0�       �	Jz�Tfc�A�*

loss�ϓ=���M       �	��Tfc�A�*

lossG�<`ئ       �	X��Tfc�A�*

loss��/=���       �	�b�Tfc�A�*

loss K�<u�r�       �	z��Tfc�A�*

loss��=�w�       �	B��Tfc�A�*

losst�)<|4q       �	�2�Tfc�A�*

loss�Ѻ<�(1       �	}��Tfc�A�*

lossM�=;���-       �	���Tfc�A�*

loss�*�<&�g�       �	@��Tfc�A�*

loss!�`=M��       �	D�Tfc�A�*

loss�{�<��>       �	m��Tfc�A�*

loss�l�<�O7       �	�Tfc�A�*

loss�LK<���c       �	���Tfc�A�*

lossOt�=�`       �	�c�Tfc�A�*

loss5 �=D"B       �	���Tfc�A�*

loss�J=���J       �	�1�Tfc�A�*

loss�ѕ<yo�B       �	��Tfc�A�*

loss���<5Ȉ       �	�o�Tfc�A�*

loss�S= �}�       �	.:�Tfc�A�*

loss_8m<C-�       �		��Tfc�A�*

loss�D�=�-N�       �	o� Ufc�A�*

loss��K<�n�       �	�Ufc�A�*

loss�4<��P�       �	O�Ufc�A�*

loss��1=�6`'       �	bLUfc�A�*

loss}�=����       �	q�Ufc�A�*

lossX}�;�uh       �	��Ufc�A�*

lossQ �=�Cd_       �	�Ufc�A�*

loss���<��lt       �	k�Ufc�A�*

lossV|m=�׸i       �	'PUfc�A�*

lossiy�;g�z       �	��Ufc�A�*

lossJy<���       �	g�Ufc�A�*

lossI�k=_l�       �	lAUfc�A�*

losshG<j3�       �	��Ufc�A�*

loss���<���       �	.�Ufc�A�*

loss�b3<I(��       �	;7	Ufc�A�*

loss&>�;\0�       �	C�	Ufc�A�*

lossA�=���`       �	��
Ufc�A�*

loss���<u'|2       �	�=Ufc�A�*

loss�>>R4N       �	��Ufc�A�*

loss�;>�Mf       �	��Ufc�A�*

loss%P=�Ѓ�       �	�>Ufc�A�*

loss�/�<G��       �	2�Ufc�A�*

loss8�	=���U       �	��Ufc�A�*

loss�p�<|���       �	�1Ufc�A�*

loss�ka=���       �	��Ufc�A�*

loss�)�<�x"       �	��Ufc�A�*

lossQ��="ң|       �	�#Ufc�A�*

loss�ɚ;mD�       �	��Ufc�A�*

loss]�<��       �	vUfc�A�*

loss�ޟ=�       �	�&Ufc�A�*

loss�!�=�m!       �	*�Ufc�A�*

loss��<k@ӣ       �	W|Ufc�A�*

loss��u=S��       �	�?Ufc�A�*

loss䵇<'n��       �	�Ufc�A�*

loss-�=ٷF       �	_�Ufc�A�*

loss?�<*
T�       �	�EUfc�A�*

loss�"�=[��       �	��Ufc�A�*

loss�ʱ<� �       �	��Ufc�A�*

loss�-�;���       �	]Ufc�A�*

loss�Z�=]��`       �	$|Ufc�A�*

loss�6=%��       �	D4Ufc�A�*

loss U�<�j�       �	��Ufc�A�*

loss�9=W."       �	H�Ufc�A�*

loss���;��       �	2VUfc�A�*

loss�=�<X�<H       �	�Ufc�A�*

lossh�=n�9       �	��Ufc�A�*

loss��+<Ӽ�^       �	GsUfc�A�*

loss�9�=�E��       �	� Ufc�A�*

loss%�=�07�       �	c� Ufc�A�*

loss��=A���       �	py!Ufc�A�*

loss-�=�U�B       �	C"Ufc�A�*

loss1D<����       �	mW#Ufc�A�*

loss �;U���       �	R$Ufc�A�*

loss>�=�>�       �	I%Ufc�A�*

loss6�<"�       �	��%Ufc�A�*

loss�x�;��p       �	D�&Ufc�A�*

loss���<��[       �	,b'Ufc�A�*

losss3�=�ZrH       �	,(Ufc�A�*

lossT��=;S��       �	*�(Ufc�A�*

loss�<c�W�       �	�)Ufc�A�*

loss��<�,��       �	^�*Ufc�A�*

loss,�Q=�Rzx       �	�+Ufc�A�*

loss�
�=�;k       �	�+Ufc�A�*

lossWD=�{v       �	�T,Ufc�A�*

lossT�=��G       �	�,Ufc�A�*

loss�K<}�#       �	�-Ufc�A�*

loss�j�;�3��       �	�H.Ufc�A�*

loss��<�5�       �	��.Ufc�A�*

loss l-=s�(       �	�/Ufc�A�*

loss�w�=�qP       �	�#0Ufc�A�*

loss�b=r�G�       �	"�0Ufc�A�*

lossS	<4x        �	�X1Ufc�A�*

loss��H=`��       �	f�1Ufc�A�*

loss���;<��       �	�2Ufc�A�*

loss	�)=����       �	c(3Ufc�A�*

loss�m�<Z��`       �	Z�3Ufc�A�*

loss{>�<8�'       �	-]4Ufc�A�*

loss�ϗ<!�       �	o�4Ufc�A�*

lossxcL=��7%       �	��5Ufc�A�*

loss֊�= ��d       �	_(6Ufc�A�*

loss�=#���       �	˻6Ufc�A�*

loss��R=��       �	N7Ufc�A�*

lossB==�$�j       �	C�7Ufc�A�*

lossA�<#�8�       �	�u8Ufc�A�*

loss;�<��       �	�9Ufc�A�*

loss%�
=y���       �	�9Ufc�A�*

loss��=��       �	�6:Ufc�A�*

loss��=���       �	G�:Ufc�A�*

lossV��=�l|Q       �	�a;Ufc�A�*

loss�$�=��FN       �	��;Ufc�A�*

loss��=/��       �	?�<Ufc�A�*

lossW^�=�	�       �	�"=Ufc�A�*

lossr�<L�       �	��=Ufc�A�*

loss<,�<��ac       �	�L>Ufc�A�*

loss�E�=o�u�       �	B�>Ufc�A�*

loss��=	��{       �	+�?Ufc�A�*

loss���;����       �	�X@Ufc�A�*

loss��z<�s�        �	nAUfc�A�*

loss��|=����       �	}�AUfc�A�*

loss�a=�\�=       �	�WBUfc�A�*

loss�f<~�M�       �	�CUfc�A�*

loss�bJ<k�OY       �	��DUfc�A�*

lossq��;�%4       �	�AEUfc�A�*

loss��<bs'#       �	n�EUfc�A�*

loss�C�<���       �	��FUfc�A�*

loss�	=�Gv�       �	�.GUfc�A�*

loss��\=@��       �	1�GUfc�A�*

loss�i�<��       �	{HUfc�A�*

losst=�.�r       �	�IUfc�A�*

lossf׹=7�x       �	/�IUfc�A�*

loss���=�d        �	caJUfc�A�*

loss���<���       �	��JUfc�A�*

loss�w<�rP&       �	�KUfc�A�*

losss�
=�\��       �	xBLUfc�A�*

loss�8=���"       �	M�MUfc�A�*

lossB�=B��       �	��NUfc�A�*

losss��<�'F6       �	Y4OUfc�A�*

lossW�{<���       �	�OUfc�A�*

loss9s <E��       �	eQUfc�A�*

lossn{r=�W�#       �	RUfc�A�*

losswd=Cߏ       �	�RUfc�A�*

loss��<�^��       �	kJSUfc�A�*

loss$�%=<��       �	��SUfc�A�*

loss�=%B�       �	��TUfc�A�*

losst�=4m��       �	�)UUfc�A�*

lossLz�;�OG�       �	��UUfc�A�*

loss�t<?S1<       �	mpVUfc�A�*

lossq� =��j        �	�
WUfc�A�*

loss�!�<���       �	ӣWUfc�A�*

loss��W=H,r       �	>XUfc�A�*

loss��i<g��P       �	�XUfc�A�*

loss�J=	�t�       �	�vYUfc�A�*

loss�5�<P~7V       �	DZUfc�A�*

loss��$<bs��       �	��ZUfc�A�*

loss$2=:B��       �	Q[Ufc�A�*

lossh��<f�       �	��[Ufc�A�*

lossb�=�(�0       �	�\Ufc�A�*

loss=��<�ܹ       �	�"]Ufc�A�*

loss{>�<���       �	#�]Ufc�A�*

loss�8=����       �	�Y^Ufc�A�*

loss�{=���       �	>_Ufc�A�*

loss\��=��vE       �	:�_Ufc�A�*

loss�_=���       �	wJ`Ufc�A�*

loss�7m=;�p�       �	��`Ufc�A�*

lossI[�<-�f*       �	`�aUfc�A�*

lossO��<7B�       �	�,bUfc�A�*

loss\��=�᤾       �	�fcUfc�A�*

lossRc>��       �	�]dUfc�A�*

loss�n�<��)V       �	�GeUfc�A�*

loss�z;;`M@u       �	?�fUfc�A�*

loss!Ż<P*P�       �	kcgUfc�A�*

loss���=J[�B       �	��gUfc�A�*

loss@<<�)!       �	��hUfc�A�*

lossƅ�;&Q       �	˃iUfc�A�*

loss��<�jK       �	\jUfc�A�*

lossa��<�D�e       �	G�jUfc�A�*

lossxЌ=�qY1       �	�nkUfc�A�*

loss�q>C
��       �	lUfc�A�*

loss�.�<���       �	U�lUfc�A�*

lossEέ=��b       �	�cmUfc�A�*

loss��<Yi�K       �	.�mUfc�A�*

loss�<(;�-��       �	N�nUfc�A�*

loss�ٖ<���/       �	�>oUfc�A�*

loss��K=�{��       �	��oUfc�A�*

loss���<��l       �	�pUfc�A�*

loss�&z<7��E       �	�-qUfc�A�*

lossqЅ<DFu       �	��qUfc�A�*

loss�NC=��s       �	�qrUfc�A�*

loss���<8-�       �	�sUfc�A�*

loss���<{1�@       �		�sUfc�A�*

lossT�>���;       �	p�tUfc�A�*

loss߮M=��¸       �	7OuUfc�A�*

loss�s(=&M�       �	d�uUfc�A�*

loss4�=8�8�       �	�ewUfc�A�*

loss0�=�1i�       �	ZxUfc�A�*

loss�8L=�c�]       �	�,yUfc�A�*

loss�G=�ʔq       �	��yUfc�A�*

loss���=���       �	duzUfc�A�*

loss��(<�|       �	"{Ufc�A�*

lossv�y=,˾�       �	T�{Ufc�A�*

lossW,=<�H�       �	�g|Ufc�A�*

loss2�<}��6       �	}Ufc�A�*

loss��<x��[       �	x�}Ufc�A�*

lossϹ
=.��       �	F[~Ufc�A�*

lossh'�<�Kx"       �	��Ufc�A�*

loss�KH<)gő       �	�<�Ufc�A�*

loss��J<��K�       �	�߀Ufc�A�*

loss��=��8~       �	���Ufc�A�*

lossJ�<PNU�       �	:[�Ufc�A�*

losshE%=��bF       �	�X�Ufc�A�*

loss3/�=yyϔ       �	��Ufc�A�*

loss
w	=��H(       �	S��Ufc�A�*

lossS��<1�a       �	mW�Ufc�A�*

loss���=}�       �	�Ufc�A�*

lossk�=�V��       �	}��Ufc�A�*

loss.h�<0       �	㎈Ufc�A�*

loss$=��o       �	6:�Ufc�A�*

loss<g�=Ô[x       �	eމUfc�A�*

loss�ğ=5�N       �	�{�Ufc�A�*

lossat�<�F,]       �	X�Ufc�A�*

loss*��<R�KL       �	V��Ufc�A�*

lossO^=o}=b       �	S�Ufc�A�*

loss��=��       �	c�Ufc�A�*

lossԞ�<(�I       �	���Ufc�A�*

loss]`�<y�F\       �	F(�Ufc�A�*

lossWn�<��O       �	iǎUfc�A�*

loss���=�T\       �	_�Ufc�A�*

loss4I4=�.B�       �	���Ufc�A�*

loss4�k=q�U�       �	��Ufc�A�*

loss� ~=EDjz       �	>=�Ufc�A�*

loss7q�=_��       �	��Ufc�A�*

loss��;6�       �	递Ufc�A�*

lossYT�<�\�K       �	� �Ufc�A�*

loss��<\'YO       �	���Ufc�A�*

loss��(=�f�       �	OɔUfc�A�*

lossۥ=�vq       �	c�Ufc�A�*

loss}�=<�v       �	���Ufc�A�*

loss='n       �	ĕ�Ufc�A�*

loss�a�<�͸�       �	�/�Ufc�A�*

lossU�=�D2       �	�͗Ufc�A�*

loss��<=<��[       �	�e�Ufc�A�*

lossƩ�<ՙ       �	u�Ufc�A�*

loss��<�@�       �	5��Ufc�A�*

loss��<̃e�       �	9�Ufc�A�*

loss@D�<cR�       �	OΚUfc�A�*

loss�*�<?�I       �	�j�Ufc�A�*

loss�+�=���v       �	I�Ufc�A�*

lossϫ�<����       �	Ū�Ufc�A�*

lossi��;��yH       �	B�Ufc�A�*

loss`t	<�.�E       �	���Ufc�A�*

lossK�<���O       �	�{�Ufc�A�*

loss<��<���       �	��Ufc�A�*

lossϠ�=����       �	7��Ufc�A�*

loss�<�\�       �	vP�Ufc�A�*

loss%Q
=��Ɠ       �	��Ufc�A�*

loss��c=���       �	���Ufc�A�*

loss�1�=�s,       �	�/�Ufc�A�*

loss<=��ڛ       �	FѢUfc�A�*

loss��=@i��       �	�w�Ufc�A�*

loss�|w=�5F       �	bj�Ufc�A�*

loss�D1<q>�       �	��Ufc�A�*

loss$W�<�81o       �	���Ufc�A�*

lossVa�=���       �	ܦUfc�A�*

loss�M�<>hK�       �	ʨ�Ufc�A�*

loss&Kh=��       �	:�Ufc�A�*

lossfsz<J3o       �	���Ufc�A�*

lossn]?<)�׌       �	֪Ufc�A�*

loss�
=H(b�       �	M��Ufc�A�*

lossV��=�'Va       �	6r�Ufc�A�*

loss�D"=�$&       �	�r�Ufc�A�*

loss`�}<bd6       �	��Ufc�A�*

loss�Q=[��       �	�%�Ufc�A�*

loss#�=B"܍       �	��Ufc�A�*

loss��=N��       �	r°Ufc�A�*

loss��Z<���       �	g��Ufc�A�*

lossE��<�@gE       �	�X�Ufc�A�*

loss �A<�7Ł       �	t)�Ufc�A�*

loss��}=�m��       �	v³Ufc�A�*

loss>'=�#/~       �	l[�Ufc�A�*

lossX�;=�#$       �	#�Ufc�A�*

loss�:�=��pe       �	��Ufc�A�*

lossi(�;NP��       �	c��Ufc�A�*

loss̞<�xa�       �	�c�Ufc�A�*

loss���=94��       �	��Ufc�A�*

loss3��<��       �	K��Ufc�A�*

loss��<'��O       �	�U�Ufc�A�*

loss3�<��D       �	G�Ufc�A�*

loss�r�=��       �	���Ufc�A�*

loss�G�:Q���       �	�Y�Ufc�A�*

loss�	�<1��       �	���Ufc�A�*

loss�Z<a�       �	b��Ufc�A�*

lossx��=^��       �	M�Ufc�A�*

loss��<��M       �	��Ufc�A�*

loss�s=>��!       �	���Ufc�A�*

loss�Z�=b�,g       �	�[�Ufc�A�*

loss��v=kx�       �	��Ufc�A�*

loss*�P=v �       �	���Ufc�A�*

lossW>�<c4�e       �	:�Ufc�A�*

lossp�=��Y�       �	
��Ufc�A�*

loss3�<1� �       �	�{�Ufc�A�*

loss�di=�`�       �	�Ufc�A�*

loss��=a��       �	d��Ufc�A�*

loss�S=���w       �	�T�Ufc�A�*

loss76�=2���       �	N��Ufc�A�*

loss�v�<�^�        �	y��Ufc�A�*

loss?�=���V       �	32�Ufc�A�*

loss�pg=��E       �	K��Ufc�A�*

lossʬ�=ؕ��       �	Xu�Ufc�A�*

loss���; Ǧ�       �	��Ufc�A�*

lossTJy<���       �	֧�Ufc�A�*

loss��b=p*7       �	H�Ufc�A�*

loss*��<{$.       �	���Ufc�A�*

lossj�;�k�j       �	�~�Ufc�A�*

loss2�T=6��a       �	'�Ufc�A�*

loss6'�;s�I'       �	��Ufc�A�*

loss�:�=	�ܡ       �	�G�Ufc�A�*

lossY	=-�0�       �	���Ufc�A�*

loss��=gP�       �	�x�Ufc�A�*

loss)�<aJ�       �	��Ufc�A�*

lossM`-=U��       �	���Ufc�A�*

lossM=�kK
       �	�M�Ufc�A�*

loss�ʜ<��؂       �	��Ufc�A�*

lossl�=�$��       �	1x�Ufc�A�*

loss[��<&�~�       �	�Ufc�A�*

loss��<u��       �	��Ufc�A�*

loss��=8��B       �	pA�Ufc�A�*

loss���<ܔ;T       �	���Ufc�A�*

loss@��<��0       �	[��Ufc�A�*

lossf=��G�       �	n3�Ufc�A�*

lossR^C<�0��       �	���Ufc�A�*

lossm��;��y-       �	���Ufc�A�*

lossH<,��?       �	�!�Ufc�A�*

loss{-�=�B�       �	7��Ufc�A�*

lossfa=1�3^       �	�p�Ufc�A�*

lossX_�<�#9       �	��Ufc�A�*

loss�],=1ld�       �	���Ufc�A�*

lossCTp;��@o       �	nm�Ufc�A�*

loss�]�<kU�<       �	B�Ufc�A�*

loss)�>��       �	���Ufc�A�*

loss8�=�/�s       �	YN�Ufc�A�*

loss{Ei<�gw       �	���Ufc�A�*

loss�yh=����       �	A��Ufc�A�*

loss�V�=��R       �	�7�Ufc�A�*

loss�V�<9	��       �	���Ufc�A�*

lossl�<jS�       �	��Ufc�A�*

loss|]�= ��       �	b/�Ufc�A�*

loss���=��       �	E��Ufc�A�*

loss��=ѫU�       �	��Ufc�A�*

loss���=�kv
       �	�F�Ufc�A�*

lossچK<�d       �	���Ufc�A�*

loss��9=��Nj       �	ѕ�Ufc�A�*

loss��1=c�\c       �	B��Ufc�A�*

lossE�<�?B       �	��Ufc�A�*

lossl*3;G�T       �	<��Ufc�A�*

loss6 <�_��       �	W\�Ufc�A�*

loss���=c��       �	�'�Ufc�A�*

loss��=Q뮆       �	+��Ufc�A�*

loss�k<�x�8       �	���Ufc�A�*

loss8��<>       �	[&�Ufc�A�*

loss�Hk<O6[       �	l��Ufc�A�*

loss��:<X(:       �	�s�Ufc�A�*

loss̂�<k��S       �	�Ufc�A�*

loss�=\�u       �	`��Ufc�A�*

lossR��<���       �	�v�Ufc�A�*

loss��=�5�       �	Wx�Ufc�A�*

lossZ��=���       �	$%�Ufc�A�*

loss�-<�a|m       �	F��Ufc�A�*

lossm�<��"%       �	��Ufc�A�*

loss��;��       �	�)�Ufc�A�*

loss��&=�@��       �	g��Ufc�A�*

loss��<���`       �	�{�Ufc�A�*

loss���=�5�       �	�*�Ufc�A�*

lossV8k>�Ua       �	���Ufc�A�*

lossr��<:�       �	k�Ufc�A�*

lossI�<풄�       �	w�Ufc�A�*

loss�8=&�#?       �	���Ufc�A�*

loss^v=��-�       �	�\�Ufc�A�*

loss-/�<͜T�       �	`�Ufc�A�*

lossZO�<*2]       �	K��Ufc�A�*

loss�U�=y�M�       �	�a�Ufc�A�*

loss�a�<��#�       �	*��Ufc�A�*

loss�j=Q�=       �	��Ufc�A�*

loss��N<�+Y�       �	:�Ufc�A�*

lossM�=/���       �	b��Ufc�A�*

loss�x<��0�       �	�|�Ufc�A�*

lossF�<��       �	?�Ufc�A�*

loss�%�=�w�?       �	���Ufc�A�*

loss�d =����       �	�r�Ufc�A�*

lossZ�*=Q��^       �	��Ufc�A�*

loss$G�;�ۺ)       �	d��Ufc�A�*

loss��=j��       �	bL�Ufc�A�*

loss߁=����       �	X��Ufc�A�*

lossN�/=��       �	k��Ufc�A�*

lossv<��:b       �	� Vfc�A�*

losss�<�x}       �	��Vfc�A�*

loss���<�ŏ�       �	~�Vfc�A�*

loss���=_�s        �	$^Vfc�A�*

loss��<	��       �	]�Vfc�A�*

loss6Z�<�ҽg       �	��Vfc�A�*

loss���<>I,)       �	YOVfc�A�*

loss�N=]h�       �	1�Vfc�A�*

loss �K=ϿnR       �	��Vfc�A�*

losss��<���<       �	�"Vfc�A�*

loss���<�;	t       �	��Vfc�A�*

loss���;��KA       �	fVfc�A�*

loss��=�mJ       �	��Vfc�A�*

lossTq<4{�'       �	Z�	Vfc�A�*

loss:�=>+��       �	S@
Vfc�A�*

loss_i$=�m�       �	#�
Vfc�A�*

loss���<�YK�       �	�yVfc�A�*

loss�{<D�O        �	<Vfc�A�*

lossÍr=�/�       �	�Vfc�A�*

lossv��<U�$M       �	�dVfc�A�*

loss�ݿ<���?       �	�Vfc�A�*

lossR@�;
�u�       �	ެVfc�A�*

loss�Rq<�N       �	�RVfc�A�*

loss��<��`       �	��Vfc�A�*

loss3��;=�       �	˟Vfc�A�*

loss���;2΢       �	l?Vfc�A�*

lossZ==���       �	7�Vfc�A�*

loss@w�=�u�       �	w�Vfc�A�*

loss�<˲Z|       �	?Vfc�A�*

loss�ZN=L"�       �	J�Vfc�A�*

loss�3�=�If       �	YVfc�A�*

loss&M;Ğ'       �	��Vfc�A�*

lossK7�:Eq��       �	��Vfc�A�*

loss�@<��?s       �	Z.Vfc�A�*

loss��;�A-       �	$�Vfc�A�*

loss��;&/o       �	�tVfc�A�*

loss��;��>�       �	�Vfc�A�*

lossQN�:/r�       �	�Vfc�A�*

loss�@�;?��_       �	[]Vfc�A�*

loss�M�;� ��       �	�Vfc�A�*

loss��;�4W'       �	�Vfc�A�*

loss��:�Ӏ�       �	n1Vfc�A�*

lossd��<&�>       �	x�Vfc�A�*

loss?=ݣ!�       �	�xVfc�A�*

loss�[<+ڌ�       �	/Vfc�A�*

loss�|�:�ȃ�       �	�Vfc�A�*

loss1/�;�ghM       �	�UVfc�A�*

loss�'>�}�`       �	��Vfc�A�*

lossS;���n       �	y�Vfc�A�*

loss��y<����       �	YO!Vfc�A�*

lossnt�=Jd       �	�"Vfc�A�*

loss6ϱ<0��       �	��"Vfc�A�*

loss��h<9�3       �	:�#Vfc�A�*

loss̕<�p       �	�-$Vfc�A�*

lossn�=.^�t       �	�%Vfc�A�*

loss���=�X�       �	+�%Vfc�A�*

loss ��<U�c^       �	F	'Vfc�A�*

lossa2=m�
       �	(�'Vfc�A�*

lossHo;=Ba��       �	�a(Vfc�A�*

loss�+S=0<�$       �	�	)Vfc�A�*

loss�9�=B�       �	��)Vfc�A�*

loss��0=]�FL       �	U*Vfc�A�*

loss�4M=)n��       �	�*Vfc�A�*

loss��'=N��$       �	Ȕ+Vfc�A�*

lossY=T���       �	&�,Vfc�A�*

loss�S�;�D(V       �	�c-Vfc�A�*

loss	��=��3       �	�-Vfc�A�*

loss3��<D�       �	D�.Vfc�A�*

loss���;E�       �	rP/Vfc�A�*

loss�\�<�a	4       �	K0Vfc�A�*

loss��= ��,       �	L�0Vfc�A�*

loss�D<8^R       �	y1Vfc�A�*

loss��;o�=>       �	(2Vfc�A�*

loss]E�;���D       �	��2Vfc�A�*

loss�%�<��%       �	�>3Vfc�A�*

loss
#�<�� @       �	��3Vfc�A�*

loss��#=�D�       �	�$5Vfc�A�*

losswo=@�W�       �	Ǽ5Vfc�A�*

loss!\X;XL       �	`W6Vfc�A�*

loss�G�=��v�       �	 7Vfc�A�*

loss*�<���       �	�7Vfc�A�*

loss
W=;���       �	�38Vfc�A�*

loss��<R�g       �	��8Vfc�A�*

loss�O%<�W�       �	Z�9Vfc�A�*

losso͝<Vx!�       �	T::Vfc�A�*

lossږ�<��)       �	X�:Vfc�A�*

loss|�O=�W�       �	��;Vfc�A�*

loss���<��N	       �	�#<Vfc�A�*

loss|��:����       �	6�<Vfc�A�*

lossT%�<�lxT       �	Kt=Vfc�A�*

lossWc�<��oU       �	�>Vfc�A�*

loss�}O<���B       �	�>Vfc�A�*

loss"H<�i�       �	�HAVfc�A�*

lossM��=E���       �	]�AVfc�A�*

loss?L�=$Hp)       �	ɫBVfc�A�*

loss��j<䚈       �	�KCVfc�A�*

lossm==҆�g       �	%�CVfc�A�*

loss�z;GZ��       �	�DVfc�A�*

lossj�3<<�$:       �	�3EVfc�A�*

lossΟ�<��I�       �	�\Vfc�A�*

loss�PY=10j�       �	�B]Vfc�A�*

loss���<y9d�       �	^�]Vfc�A�*

loss@�.=���       �	~^Vfc�A�*

loss�=��l       �	�_Vfc�A�*

loss���<0cml       �	 �_Vfc�A�*

loss�w�<�!~       �	jP`Vfc�A�*

lossS�F=`��Z       �	��`Vfc�A�*

loss���=�Q	       �	A�aVfc�A�*

lossTkQ=�@�F       �	�bVfc�A�*

loss &<���4       �	��bVfc�A�*

loss1�=��,       �	7�cVfc�A�*

loss\�X=i|<�       �	ѕdVfc�A�*

lossA��<��.B       �	�=eVfc�A�*

loss:�t=����       �	��eVfc�A�*

lossqVN=b^(       �	njfVfc�A�*

loss;��       �	 gVfc�A�*

loss.�;��2       �	1�gVfc�A�*

loss�B<M:��       �	r4hVfc�A�*

loss�>=�杓       �	��hVfc�A�*

loss�A<Y��%       �	��iVfc�A�*

loss*�=G�p�       �	RbjVfc�A�*

loss��p<��D       �	��jVfc�A�*

loss%-f=��O2       �	�kVfc�A�*

loss�=eX[�       �	p�lVfc�A�*

loss�, =����       �	*:mVfc�A�*

loss�Į<)ސ�       �	0�mVfc�A�*

loss��^=�j��       �	V�nVfc�A�*

lossK=���       �	Y�oVfc�A�*

loss/:<�QH?       �	�6pVfc�A�*

lossZf=�;[�       �	��pVfc�A�*

loss}�<�w�       �	�lqVfc�A�*

loss <jK�       �	1rVfc�A�*

lossWZ�=q��       �	��rVfc�A�*

loss|� =!��       �	isVfc�A�*

loss=�\=���       �	�tVfc�A�*

loss)�^<����       �	��tVfc�A�*

loss��%=w���       �	�>uVfc�A�*

lossqv>��}       �	C�uVfc�A�*

lossB	=O�5�       �	"�vVfc�A�*

loss}A�==��       �	�%wVfc�A�*

loss��<+��E       �	��wVfc�A�*

lossM�!=��~       �	�[xVfc�A�*

loss��<sP�       �	��xVfc�A�*

loss��{<Wa�       �	��yVfc�A�*

loss�3J=�Fz       �	 $zVfc�A�*

loss���<B��       �	˺zVfc�A�*

loss�u>=�Po�       �	SZ{Vfc�A�*

loss}��;^�o�       �	�{Vfc�A�*

loss���<kE��       �	�|Vfc�A�*

loss��~<�p�       �	� }Vfc�A�*

loss�3-=��I6       �	��}Vfc�A�*

loss7}�<���       �	8H~Vfc�A�*

lossC��=i>3       �	��~Vfc�A�*

loss{�=��	       �	x{Vfc�A�*

loss[�<��0Q       �	q�Vfc�A�*

loss/:.<ɩ��       �	���Vfc�A�*

lossh�p<���       �	I�Vfc�A�*

lossSB�<Ov`�       �	U݁Vfc�A�*

lossN=g��       �	�q�Vfc�A�*

loss���=Զ��       �	)�Vfc�A�*

loss�<�_A�       �	���Vfc�A�*

lossI��:�0       �	T6�Vfc�A�*

lossa�&<��1�       �	BЄVfc�A�*

lossF�e<�6U�       �	g�Vfc�A�*

loss�P�=6�@�       �	?��Vfc�A�*

loss{cQ=��       �	���Vfc�A�*

loss��=MfB�       �	�>�Vfc�A�*

loss���=�M��       �	'އVfc�A�*

loss�_�<!�6�       �	�z�Vfc�A�*

loss�<G"�R       �	��Vfc�A�*

lossԴ/= �\5       �	d��Vfc�A�*

lossܯ�<`A       �	�F�Vfc�A�*

lossj{�<��-�       �	��Vfc�A�*

loss�K=!���       �	���Vfc�A�*

loss��;6D��       �	�-�Vfc�A�*

loss�/=e���       �	�ʌVfc�A�*

loss��B=�e��       �	/j�Vfc�A�*

loss�C=�XTs       �	��Vfc�A�*

loss��<�9�       �	ۣ�Vfc�A�*

loss�9\=R       �	�I�Vfc�A�*

loss,�D=��/       �	q�Vfc�A�*

loss,�=�L��       �	F|�Vfc�A�*

loss��<��:       �	3�Vfc�A�*

loss&(�;N��       �	���Vfc�A�*

loss�9<����       �	eT�Vfc�A�*

loss���<�SS�       �	N�Vfc�A�*

loss���<wi^       �	���Vfc�A�*

loss�e�<a`�       �	��Vfc�A�*

lossȍ�=��n       �	���Vfc�A�*

loss��:=�\k       �	Q�Vfc�A�*

lossv9,=���-       �	��Vfc�A�*

loss��=��       �	y�Vfc�A�*

loss[y�=���       �	��Vfc�A�*

loss�<�=���       �	`��Vfc�A�*

loss}��;���"       �	�v�Vfc�A�*

loss��,=�OI       �	��Vfc�A�*

loss�QG<u�       �	稙Vfc�A�*

loss(M�=W�@�       �	VF�Vfc�A�*

loss���<p�0!       �	��Vfc�A�*

loss��F;�J       �	��Vfc�A�*

lossO_�;���       �	!�Vfc�A�*

lossH�=-v�       �	ĵ�Vfc�A�*

loss1�=��T       �	�K�Vfc�A�*

loss��8=���       �	E�Vfc�A�*

lossT"}=�>�       �	���Vfc�A�*

lossHu;���       �	w.�Vfc�A�*

lossV<c=��       �	_ҟVfc�A�*

loss9�=���O       �	\v�Vfc�A�*

loss�v6<�fX�       �	z�Vfc�A�*

loss�]k=��       �	���Vfc�A�*

loss�4�<Z��       �	$a�Vfc�A�*

loss��=c#)       �	�f�Vfc�A�*

loss�>�;{Hq�       �	�Vfc�A�*

loss�q:=�7~�       �	�T�Vfc�A�*

loss�K�;�[�       �	���Vfc�A�*

loss�Ј<���X       �	���Vfc�A�*

loss�]�=��l       �	�1�Vfc�A�*

loss�V=��+Y       �	2̧Vfc�A�*

loss�.�;TM�"       �	Lk�Vfc�A�*

lossf�=Pv�r       �	��Vfc�A�*

lossH��<��$E       �	�ΩVfc�A�*

loss}}:=�h�}       �	�v�Vfc�A�*

loss�9�<�u�       �	��Vfc�A�*

lossFy�;Ʋ�@       �	���Vfc�A�*

loss�{�=�l       �	�X�Vfc�A�*

loss!��=�&��       �	���Vfc�A�*

loss�xP<��$       �	���Vfc�A�*

loss�q=�)�Z       �	�G�Vfc�A�*

loss��=���r       �	m�Vfc�A�*

loss#L+;�VB       �	&��Vfc�A�*

lossf�8<�KQ       �	�%�Vfc�A�*

loss���<�Q�       �	�ǰVfc�A�*

loss�aP<��X�       �	{j�Vfc�A�*

loss���=B�       �	p�Vfc�A�*

loss�w
<�j<�       �	���Vfc�A�*

lossJ��<���       �	<P�Vfc�A�*

lossO5�<\y�,       �	)��Vfc�A�*

lossv��<�0       �	��Vfc�A�*

loss:�<�x��       �	:#�Vfc�A�*

loss���;͍�{       �	<��Vfc�A�*

loss��\<���       �	F^�Vfc�A�*

loss�(
=(�re       �	�
�Vfc�A�*

loss��=f��!       �	���Vfc�A�*

loss�/=3cx�       �	iQ�Vfc�A�*

loss6�<��r
       �	���Vfc�A�*

loss�=��%q       �	���Vfc�A�*

loss4��;��-       �	�L�Vfc�A�*

loss�\�<�j�       �	���Vfc�A�*

loss��=cL�       �	��Vfc�A�*

loss�(N<L
N       �	���Vfc�A�*

loss�]=7^       �	�5�Vfc�A�*

loss�\=����       �	S�Vfc�A�*

lossz��<D �       �	?��Vfc�A�*

loss &=X�       �	�C�Vfc�A�*

lossa�:zK�       �	�ܿVfc�A�*

loss� �<Ϧ��       �	��Vfc�A�*

loss|b�=Vn�D       �	��Vfc�A�*

loss��=F|8       �	���Vfc�A�*

loss̇2=1f�!       �	�h�Vfc�A�*

loss��T=t�|        �		�Vfc�A�*

lossTba<B`8       �	���Vfc�A�*

loss
�Z=����       �	%z�Vfc�A�*

loss�rz;$���       �	��Vfc�A�*

loss�c�=>1       �	��Vfc�A�*

losseA�<3{�       �	K�Vfc�A�*

loss�%;SΪ�       �	u�Vfc�A�*

loss�K=��       �	|��Vfc�A�*

loss��.<���       �	�7�Vfc�A�*

loss�� <��+r       �	>��Vfc�A�*

loss��<3%       �	���Vfc�A�*

loss���;?�K�       �	�-�Vfc�A�*

loss��<B��Z       �	T��Vfc�A�*

loss�I=�zRr       �	�^�Vfc�A�*

loss�� <y%H�       �	H��Vfc�A�*

lossC8_< - �       �	9a�Vfc�A�*

loss{�<�Xg�       �	g�Vfc�A�*

loss��f="�B�       �	k��Vfc�A�*

lossiIm=�7�!       �	�\�Vfc�A�*

lossd��<�A�}       �	R��Vfc�A�*

lossR`�;��       �	~��Vfc�A�*

loss�J�=42�g       �	N%�Vfc�A�*

lossZ~=�0}       �	�J�Vfc�A�*

loss�9�;6��       �	��Vfc�A�*

lossq��<$n�       �	���Vfc�A�*

losst�s<�0�       �	��Vfc�A�*

loss}D=�ϣk       �	���Vfc�A�*

loss(�*=I�%
       �	�&�Vfc�A�*

loss�׵<x��       �	���Vfc�A�*

loss�6T=�G�       �	�P�Vfc�A�*

loss>C�=�'TF       �	���Vfc�A�*

loss�DX<a�y       �	���Vfc�A�*

loss��q<qw�       �	(+�Vfc�A�*

loss�&x=�\Lv       �	���Vfc�A�*

loss�Ć;��&       �	9a�Vfc�A�*

lossj=��       �	���Vfc�A�*

lossn.�<Ǝr�       �	��Vfc�A�*

lossj_/=\q:�       �	�-�Vfc�A�*

loss���<1���       �	���Vfc�A�*

loss���<@Kw�       �	Ze�Vfc�A�*

loss�~-=�:��       �	��Vfc�A�*

loss�Ux=z�-�       �	I��Vfc�A�*

loss�d�;k��       �	a8�Vfc�A�*

loss�j�=O��       �	���Vfc�A�*

loss�x�<?�l�       �	.p�Vfc�A�*

loss(	<�r�~       �	�	�Vfc�A�*

lossCi<sA�       �	��Vfc�A�*

lossa==���       �	�I�Vfc�A�*

lossǫ=�#m�       �	���Vfc�A�*

loss��=3�       �	���Vfc�A�*

loss�t�<��j�       �	��Vfc�A�*

loss]>=>�H�       �	�"�Vfc�A�*

loss���<���"       �	I��Vfc�A�*

loss�V�;��U       �	1^�Vfc�A�*

loss�4=p���       �	Y��Vfc�A�*

loss���=J�G�       �	��Vfc�A�*

loss��=>m�h       �	�*�Vfc�A�*

lossrMX=�hq       �	n��Vfc�A�*

lossŲ>=��O       �	U�Vfc�A�*

loss�=\��       �	���Vfc�A�*

loss!��<vBn       �	G��Vfc�A�*

loss(=J)�       �	8�Vfc�A�*

loss�sq<�~K       �	��Vfc�A�*

loss@��<��P       �	���Vfc�A�*

lossc@w<�.D�       �	
1�Vfc�A�*

lossϠ9<�6[�       �	���Vfc�A�*

lossCSD=�Y#m       �	 u�Vfc�A�*

loss��<����       �	��Vfc�A�*

lossx0�<���       �	ػ�Vfc�A�*

loss�:�=�"bH       �	XX�Vfc�A�*

loss��<;�G       �	9��Vfc�A�*

loss���<�ZUF       �	���Vfc�A�*

lossV��<f�]�       �	d�Vfc�A�*

loss��<��s&       �	c��Vfc�A�*

loss���<��|�       �	�r�Vfc�A�*

loss杒<%�t�       �	9	�Vfc�A�*

loss֜�=W��T       �	^��Vfc�A�*

loss �==���4       �	�5�Vfc�A�*

loss\��=%�"       �	���Vfc�A�*

lossv�;B���       �	wd�Vfc�A�*

loss%��<�
M       �	^G�Vfc�A�*

loss�?M<��,       �	+��Vfc�A�*

lossF	7=E�N       �	���Vfc�A�*

loss���=R:�       �	(e�Vfc�A�*

loss�Q�<"��Z       �	q�Vfc�A�*

loss�B<p�I�       �	u��Vfc�A�*

loss�|�<��       �	OW�Vfc�A�*

lossL�`<˝X&       �	���Vfc�A�*

loss��,=k�       �	���Vfc�A�*

loss�k�<㡥       �	j3�Vfc�A�*

loss���<����       �	F��Vfc�A�*

loss��'>={g�       �	B�Vfc�A�*

loss)��<��t       �	���Vfc�A�*

loss��><���       �	Q� Wfc�A�*

lossF�;S	�       �	�#Wfc�A�*

loss`A�<��a�       �	F�Wfc�A�*

loss�	q=2�Ua       �	){Wfc�A�*

loss��<�*:       �	o+Wfc�A�*

lossN�<�=��       �	��Wfc�A�*

loss��<+�ޤ       �	tWfc�A�*

loss.��<���y       �	tCWfc�A�*

loss�)�=�� �       �	��Wfc�A�*

loss!?=$F?       �	 |Wfc�A�*

loss�.=?�iA       �	%!Wfc�A�*

loss�q[=�:�?       �	��Wfc�A�*

loss2~<[A��       �	�pWfc�A�*

lossCp�<�ص]       �	�	Wfc�A�*

loss}z=�+�
       �	�	Wfc�A�*

lossq��=e[       �	�a
Wfc�A�*

lossM��<�\�F       �	a�
Wfc�A�*

lossɝ<F^�&       �	��Wfc�A�*

loss#{&=`%f       �	a7Wfc�A�*

loss\H==�5       �	��Wfc�A�*

loss�ڀ<�ES       �	�nWfc�A�*

loss�	�=�h��       �	RWfc�A�*

lossm2�=�-�x       �	��Wfc�A�*

loss��=�i[       �	NWfc�A�*

loss#�s<
�P       �	t�Wfc�A�*

loss6	�=��-       �	~�Wfc�A�*

loss�<==,�|\       �	�Wfc�A�*

loss\�q;@'�l       �	�Wfc�A�*

loss1\<�1       �	UWfc�A�*

lossxM�<���!       �	��Wfc�A�*

loss��4=)��       �	�Wfc�A�*

loss��5=�}�(       �	(Wfc�A�*

loss�+>�p�       �	\�Wfc�A�*

loss��<�c�j       �	�_Wfc�A�*

lossv�A<�	�|       �	�Wfc�A�*

loss�8<��Xc       �	�Wfc�A�*

loss��:Z}ǒ       �	)%Wfc�A�*

loss!�<H���       �	ϼWfc�A�*

lossL�=̍�$       �	�UWfc�A�*

loss�"�;�+"�       �	��Wfc�A�*

loss.<����       �	b�Wfc�A�*

loss��E<���`       �	Wfc�A�*

lossL��<�n��       �	�Wfc�A�*

loss<�ks�       �	�QWfc�A�*

loss#>�Ao�       �	5�Wfc�A�*

lossx��=�p`       �	q�Wfc�A�*

loss8��=-5       �	i7Wfc�A�*

loss�+!<�p)3       �	_�Wfc�A�*

loss�)�<���       �	iWfc�A�*

lossV\<��ݶ       �	WWfc�A�*

loss��>��d       �	*�Wfc�A�*

losso�7=ta*�       �	�< Wfc�A�*

lossA�>��       �	� Wfc�A�*

loss��;�%�=       �	[|!Wfc�A�*

loss�7>Dc�6       �	�"Wfc�A�*

loss\4�<���       �	ɰ"Wfc�A�*

loss]�~<����       �	�P#Wfc�A�*

loss�=i�(N       �	��#Wfc�A�*

lossƠ�<�e�       �	 �$Wfc�A�*

lossW��<��R       �	��%Wfc�A�*

losso1<%�3       �	Y�&Wfc�A�*

loss��2=��       �	zQ'Wfc�A�*

loss۵C=Zy/�       �	��'Wfc�A�*

loss�d�<$��       �	/�(Wfc�A�*

lossDn8=��Qw       �	P�)Wfc�A�*

lossZ��=C���       �	��*Wfc�A�*

loss��=њ��       �	JF+Wfc�A�*

loss��<5xN       �	y!,Wfc�A�*

lossfJ�=�!;�       �	�,Wfc�A�*

loss?�=�7��       �	�W-Wfc�A�*

loss��<�!�x       �	7q.Wfc�A�*

loss.RM<�MX�       �	
/Wfc�A�*

loss ��<�}�S       �	��/Wfc�A�*

lossF�=���       �	`:0Wfc�A�*

loss`�i=�+�]       �	��0Wfc�A�*

loss�=��O
       �	Yk1Wfc�A�*

loss�= XT       �	!2Wfc�A�*

lossLN=���j       �	Ý2Wfc�A�*

loss�O<�{��       �	�43Wfc�A�*

loss���;]^�       �	>^4Wfc�A�*

loss�b�<�F        �	9c5Wfc�A�*

loss ��=N6'�       �	��5Wfc�A�*

loss�$=���#       �	��6Wfc�A�*

loss�L=�=n�       �	F&7Wfc�A�*

lossR�<��       �	��7Wfc�A�*

loss���<' �       �	r8Wfc�A�*

loss�܈<�D'�       �	49Wfc�A�*

loss� ;W��       �	[�9Wfc�A�*

loss��&=a�8�       �	#h:Wfc�A�*

loss�Mp<&^��       �	�	;Wfc�A�*

loss��<���       �	=�;Wfc�A�*

loss��=��E�       �	]S<Wfc�A�*

loss]�@=�+Rs       �	S�<Wfc�A�*

loss�:�<�� g       �	��=Wfc�A�*

loss,/=��e�       �	�/>Wfc�A�*

losse�O<ZIk        �	x�>Wfc�A�*

loss��=���       �	�{?Wfc�A�*

loss���<�7�       �	%@Wfc�A�*

loss[�G=�~)       �	�@Wfc�A�*

lossI��<��A)       �	wdAWfc�A�*

loss��j=�Wc�       �	�BWfc�A�*

loss	�J=���       �	C�BWfc�A�*

lossGm<fqc!       �	�mCWfc�A�*

lossM_<��w       �	�DWfc�A�*

lossOi�<S�
       �	>�DWfc�A�*

loss�~�<W��`       �	�REWfc�A�*

loss��+<:0��       �	��EWfc�A�*

loss���<,��       �	��FWfc�A�*

lossq�;�GX       �	D0GWfc�A�*

lossa��<_z~w       �	��GWfc�A�*

loss���=��>�       �	~HWfc�A�*

lossB�=;}��       �	�$IWfc�A�*

lossdȰ=��0�       �	��IWfc�A�*

lossC=#gp�       �	�tJWfc�A�*

lossCh=3�Wm       �	�KWfc�A�*

loss�X\<���\       �	'�KWfc�A�*

loss0=Z���       �	iWLWfc�A�*

loss�'�<1&�       �	�LWfc�A�*

loss6��<���       �	�MWfc�A�*

loss��<���       �	:@NWfc�A�*

loss�<WZ��       �	��NWfc�A�*

loss���<"U�       �	�vOWfc�A�*

loss=Ͳ=�]F       �	�PWfc�A�*

loss��=�q�       �	P�PWfc�A�*

loss*)~=*��       �		NQWfc�A�*

loss��<Q|s1       �	��QWfc�A�*

loss2�9=����       �	~�RWfc�A�*

loss�  </�zY       �	{2SWfc�A�*

loss��=�5l�       �	�SWfc�A�*

lossz�;PR�	       �	&mTWfc�A�*

loss�s�;W{Cr       �	�UWfc�A�*

lossQ*�;��Gs       �	Y�UWfc�A�*

loss�=ӳb       �	�8VWfc�A�*

loss��i<fD�       �	��VWfc�A�*

loss�b�=�M'�       �	7lWWfc�A�*

loss.Xy={��T       �	$
XWfc�A�*

lossM�; Y$�       �	��XWfc�A�*

loss7Pl=�       �	�;YWfc�A�*

lossX2?=��C       �	��YWfc�A�*

lossN��<)1�       �	irZWfc�A�*

loss
F=\*��       �	�[Wfc�A�*

loss�C=�[�t       �	��[Wfc�A�*

loss'�=�%Q$       �	�F\Wfc�A�*

loss��;_�_�       �	�\Wfc�A�*

loss��:�� z       �	�y]Wfc�A�*

loss�T;J+�       �	� ^Wfc�A�*

loss��;�t[       �	��^Wfc�A�*

lossr/;Sk�       �	�m_Wfc�A�*

loss�=O���       �	�`Wfc�A�*

losse��<_��_       �	ȶ`Wfc�A�*

lossT3C;�Ͻ       �	�VaWfc�A�*

loss�X>�iS       �	�bWfc�A�*

loss:�<�q�J       �	3�bWfc�A�*

loss?�b=Tb��       �	<lcWfc�A�*

loss�z�=�3�I       �	�dWfc�A�*

loss�tS<۔�       �	�eWfc�A�*

loss��]=ߑ��       �	�eWfc�A�*

loss�%"=,�U�       �	GTfWfc�A�*

loss��z=/��(       �	`gWfc�A�*

loss�<'�       �	�gWfc�A�*

loss��+=��7X       �	�JhWfc�A�*

loss8e�<Nי�       �	��hWfc�A�*

losse�g=�-�       �	��iWfc�A�*

loss��p:��>f       �	?6jWfc�A�*

lossGo<So�Z       �	��jWfc�A�*

loss�P�=�`�       �	�rkWfc�A�*

loss�=����       �	jlWfc�A�*

lossQ��;U�       �	�lWfc�A�*

loss}�p=A�R]       �	gmWfc�A�*

loss��;���       �	+nWfc�A�*

loss�d�==�G�       �	�nWfc�A�*

losst >N��?       �	�_oWfc�A�*

lossl�>���       �	�pWfc�A�*

loss��=��w       �	��pWfc�A�*

loss O3<��D       �	j�qWfc�A�*

loss}'�<nW��       �	A,rWfc�A�*

loss��H<~^�       �	{�rWfc�A�*

loss�t=x��{       �	s�sWfc�A�*

loss��Q<�j        �	qtWfc�A�*

loss�'<@�P       �	ԺtWfc�A�*

loss��R<���o       �	�UuWfc�A�*

lossm��=m.�       �	B'vWfc�A�*

loss���<0��]       �	��vWfc�A�*

loss*S	=LK��       �	�hwWfc�A�*

loss;M�=$-�       �	uxWfc�A�*

loss��Y<�;�       �	_�xWfc�A�*

loss��<b��       �	�4yWfc�A�*

loss:�~<���       �	��yWfc�A�*

lossm�=0��       �	�pzWfc�A�*

loss�[�<�@�9       �	[{Wfc�A�*

loss��B=4��       �	հ{Wfc�A�*

loss��;�:S2       �	8K|Wfc�A�*

loss@0=�/��       �	��|Wfc�A�*

loss�L>v��;       �	�y}Wfc�A�*

loss1��<�/       �	�~Wfc�A�*

loss�j=��t�       �	��~Wfc�A�*

loss),�=ؓ�5       �	HSWfc�A�*

lossl=z=�ɌO       �	��Wfc�A�*

loss
X�;�cUI       �	��Wfc�A�*

loss�ߴ<����       �	�F�Wfc�A�*

loss,�<���       �	4��Wfc�A�*

loss/5 >�b8       �	H��Wfc�A�*

loss��<���       �	D��Wfc�A�*

loss��=�Y��       �	�(�Wfc�A�*

loss)2~=#Ӗ-       �	�˅Wfc�A�*

loss�H~<��r�       �	�r�Wfc�A�*

loss���=�%IY       �	�Wfc�A�*

loss���;*ι�       �	�Wfc�A�*

loss*�<��b       �	O]�Wfc�A�*

loss��<��d       �	��Wfc�A�*

loss�/P=~c�S       �	���Wfc�A�*

loss��R=��       �	�V�Wfc�A�*

loss F�<<�Z�       �	M�Wfc�A�*

loss&	�<$F�8       �	W��Wfc�A�*

loss �	=J��       �	Q2�Wfc�A�*

loss�;=��9       �	ӌWfc�A�*

loss�F<��DU       �	�s�Wfc�A�*

loss�zR=�W�       �	��Wfc�A�*

loss=�<C�+h       �	���Wfc�A�*

lossW��<9'�7       �	�X�Wfc�A�*

loss�>cN�       �	
��Wfc�A�*

lossX"=�į       �	ݖ�Wfc�A�*

loss
�%<�x�[       �	�4�Wfc�A�*

lossnI�;���"       �	�БWfc�A�*

loss�V�<��       �	�l�Wfc�A�*

loss��Q<$��       �	��Wfc�A�*

lossx^�=�L�       �	���Wfc�A�*

loss6��=O �       �	5C�Wfc�A�*

loss'�<�ي_       �	�ޔWfc�A�*

lossv�<�o+       �	�u�Wfc�A�*

loss��&=�6%       �	Z�Wfc�A�*

loss��=)l��       �	!��Wfc�A�*

loss�9g<���e       �	�A�Wfc�A�*

loss��<��       �	YߗWfc�A�*

loss	�=��       �	�u�Wfc�A�*

loss�s�<���       �	��Wfc�A�*

loss�=�R��       �	�ęWfc�A�*

loss��<��        �	s�Wfc�A�*

loss	�=|1C�       �	�&�Wfc�A�*

lossW<�	       �	]ݛWfc�A�*

lossVq =iol3       �	���Wfc�A�*

lossn�<�{4z       �	$H�Wfc�A�*

lossO��=G]       �	�Wfc�A�*

lossx�<�r��       �	���Wfc�A�*

loss;P&<긗�       �	�8�Wfc�A�*

loss]4�<a�4       �	^۟Wfc�A�*

loss��=�+ރ       �	A~�Wfc�A�*

loss�H�<9��       �	��Wfc�A�*

loss��[;�D>�       �	��Wfc�A�*

loss�)p=�5*�       �	:\�Wfc�A�*

loss��N<�K$       �	r��Wfc�A�*

loss�Z6=��       �	���Wfc�A�*

loss���=���b       �	�2�Wfc�A�*

loss���<�֒�       �	X<�Wfc�A�*

loss��7<��       �	$�Wfc�A�*

lossEq�<����       �	ʤ�Wfc�A�*

loss���<e"8       �	`9�Wfc�A�*

loss���;�(�       �	�ЧWfc�A�*

loss	��<�
2o       �	wf�Wfc�A�*

loss�<W6,       �	��Wfc�A�*

lossq�$=��"       �	���Wfc�A�*

loss��<�Ĳ�       �	�9�Wfc�A�*

loss=��;!)       �	SϪWfc�A�*

lossӻN=U
�       �	�f�Wfc�A�*

lossNF�;���       �	���Wfc�A�*

loss��<r(\       �	ޓ�Wfc�A�*

loss�=X�|       �	�5�Wfc�A�*

lossS�<h�$       �	�ЭWfc�A�*

lossڎ�<Ӷ       �	o�Wfc�A�*

lossQ7:� ��       �	��Wfc�A�*

loss�:�;���       �	���Wfc�A�*

lossx�b=��]�       �	GU�Wfc�A�*

loss!=���       �	���Wfc�A�*

loss<�=<]F��       �	Y3�Wfc�A�*

lossfz=����       �	��Wfc�A�*

loss��=���N       �	+ڳWfc�A�*

lossC
�<�4.j       �	/o�Wfc�A�*

lossd�<.�:"       �	��Wfc�A�*

loss\��;�?�       �	���Wfc�A�*

loss���;�aFt       �	�)�Wfc�A�*

loss�t>��P       �	���Wfc�A�*

loss	��:�b<       �	KV�Wfc�A�*

loss��:Ϧ�7       �	5	�Wfc�A�*

loss�()<b@c       �	���Wfc�A�*

lossW�;F7��       �	�5�Wfc�A�*

lossI� ;k>(       �	�չWfc�A�*

loss/h6<�U       �	�l�Wfc�A�*

lossy�<Q):F       �	:�Wfc�A�*

loss�"A:���k       �	<��Wfc�A�*

lossj9;k��h       �	WA�Wfc�A�*

loss�><��g�       �	��Wfc�A�*

loss���<�O$q       �	��Wfc�A�*

loss�#-<�+��       �	�&�Wfc�A�*

loss$j"9�;k�       �	�¾Wfc�A�*

loss�T�<����       �	�a�Wfc�A�*

lossE��=��׽       �	��Wfc�A�*

loss4|8;jUӴ       �		��Wfc�A�*

loss2E�<�� �       �	�g�Wfc�A�*

lossC^�<�P�       �	X7�Wfc�A�*

loss�8s<\��        �	L��Wfc�A�*

loss��=)       �	D��Wfc�A�*

loss6k�<�H�K       �	b.�Wfc�A�*

lossX}=׉��       �	z��Wfc�A�*

loss�`�=�vbU       �	���Wfc�A�*

loss�'<>��;       �	�/�Wfc�A�*

loss��%=j��       �	��Wfc�A�*

loss�=�j       �	C��Wfc�A�*

loss�R=�T�       �	�/�Wfc�A�*

lossx�<P���       �	��Wfc�A�*

lossʉ�<�J��       �	�z�Wfc�A�*

loss�}�<�-��       �	��Wfc�A�*

lossX�<G�s�       �	÷�Wfc�A�*

lossм�<���	       �	Nb�Wfc�A�*

lossf�<^��q       �	S�Wfc�A�*

loss![=��(       �	"��Wfc�A�*

loss�=��K       �	�Q�Wfc�A�*

loss]ߵ<H� �       �	��Wfc�A�*

loss&t<r��!       �	ƣ�Wfc�A�*

loss��	<��       �	pD�Wfc�A�*

loss�k<6��v       �	2��Wfc�A�*

loss>I�<m�1       �	{��Wfc�A�*

loss��:��u       �	A*�Wfc�A�*

loss��)<��#Y       �	-��Wfc�A�*

lossA�<0���       �	�|�Wfc�A�*

lossxt*=,s��       �	�&�Wfc�A�*

loss�"�=���<       �	u��Wfc�A�*

loss.�=&���       �	�z�Wfc�A�*

loss�#<q�L�       �	�"�Wfc�A�*

loss���;��F       �	2��Wfc�A�*

lossKT;Ș�v       �	�a�Wfc�A�*

loss�

=th}       �	��Wfc�A�*

loss䤞;[�WD       �	��Wfc�A�*

lossO��<��K        �	�R�Wfc�A�*

loss��5<��C�       �	K�Wfc�A�*

loss��J=/�+       �	��Wfc�A�*

loss��n=R��       �	MH�Wfc�A�*

loss���:��|J       �	��Wfc�A�*

loss��Z<��qb       �	���Wfc�A�*

loss�#=9��       �	�4�Wfc�A�*

loss�g=�@�       �	��Wfc�A�*

loss-(F=n}�\       �	�}�Wfc�A�*

loss�6<%tO{       �	6�Wfc�A�*

lossE�q=��[�       �	���Wfc�A�*

loss ܈<bV��       �	 `�Wfc�A�*

lossMo�<mJ�       �	��Wfc�A�*

loss�� =R:k       �	W��Wfc�A�*

loss΅<"feE       �	�[�Wfc�A�*

lossDR�<	�<       �	���Wfc�A�*

lossh-!=dV�V       �	X Xfc�A�*

loss@Nb=�8�       �	�� Xfc�A�*

loss���;��/       �	�Xfc�A�*

loss�٫<��J       �	�UXfc�A�*

loss�:=7Y�`       �	�Xfc�A�*

lossN+6<@|�       �	�Xfc�A�*

lossǴ<��Y       �	O[Xfc�A�*

loss��C=��       �	��Xfc�A�*

loss�Χ<��٠       �	�Xfc�A�*

loss�K�<��U�       �	�XXfc�A�*

loss��w=���       �	�Xfc�A�*

loss=�=�O1       �	��Xfc�A�*

loss*�=�ɉ       �	CXfc�A�*

loss��;���-       �	��Xfc�A�*

loss�=>�5�       �	|�	Xfc�A�*

losss�;#���       �	�G
Xfc�A�*

loss�}B<P11�       �	>�
Xfc�A�*

losst<�6�       �	�Xfc�A�*

loss��<�E       �	k-Xfc�A�*

loss���<���       �	��Xfc�A�*

loss��4=���       �	;mXfc�A�*

loss�%�;ye7�       �	�Xfc�A�*

loss|E=��O�       �	?�Xfc�A�*

loss��=�^��       �	IKXfc�A�*

loss�S�<_���       �	[�Xfc�A�*

loss�?i<?a��       �	��Xfc�A�*

loss�M�<�>��       �	h Xfc�A�*

lossZK>=S�	E       �	��Xfc�A�*

lossA�=Y��       �	SZXfc�A�*

loss@�<lv�       �	��Xfc�A�*

loss�P�<����       �	ۉXfc�A�*

loss��<CU5o       �	\ Xfc�A�*

loss� �=;(XW       �		�Xfc�A�*

loss�<	<҉       �	d[Xfc�A�*

loss�w=,���       �	f�Xfc�A�*

loss��:;��.�       �	ߌXfc�A�*

loss�t=����       �	�#Xfc�A�*

loss=��=�*@�       �	��Xfc�A�*

lossq/�=���       �	�SXfc�A�*

loss��=^xT5       �	 �Xfc�A�*

loss� =��F       �	ԁXfc�A�*

lossRM�;G�~�       �	�Xfc�A�*

loss�R=n�       �	(�Xfc�A�*

loss=e<#���       �	OXfc�A�*

loss|�-=��       �	��Xfc�A�*

loss]Ϯ<�;       �	�Xfc�A�*

loss���=���       �	�HXfc�A�*

lossi�=�{ԥ       �	]�Xfc�A�*

lossq+�:ˁ�       �	�wXfc�A�*

lossA�;b�j�       �	�qXfc�A�*

loss��z=��       �	J Xfc�A�*

loss�&<uJ��       �	�� Xfc�A�*

loss��b=���       �	�K!Xfc�A�*

loss	a<2��       �	�!Xfc�A�*

loss���;�`�       �	H�"Xfc�A�*

lossA�^;�b�       �	. #Xfc�A�*

losss�:ܑ�<       �	��#Xfc�A�*

loss��<`�       �	�$Xfc�A�*

loss���=��yF       �	 R%Xfc�A�*

loss�w=��ti       �	9�%Xfc�A�*

loss^�<ڤ��       �	��'Xfc�A�*

loss�F�=���       �	��(Xfc�A�*

lossT�<�R�       �	��)Xfc�A�*

lossу�;���       �	p�*Xfc�A�*

loss:ā<{ݎ       �	S=+Xfc�A�*

loss	�%<,�7�       �	C,Xfc�A�*

loss�:�<�Ȗ7       �	�Q-Xfc�A�*

loss�/`=�sN�       �	�!.Xfc�A�*

loss$?�<R�KI       �	��.Xfc�A�*

loss��<�4��       �	qt/Xfc�A�*

loss:m=����       �	<00Xfc�A�*

loss8��<ͤ�:       �	)1Xfc�A�*

lossg�=}��       �	=�1Xfc�A�*

loss���<�J�V       �	��2Xfc�A�*

loss]��;}-`       �	[�3Xfc�A�*

loss{��;[f�       �	�4Xfc�A�*

loss!p=��5�       �	y"5Xfc�A�*

loss U@<�3��       �	��5Xfc�A�*

loss�==����       �	�c6Xfc�A�*

lossT�<l��       �	[7Xfc�A�*

loss��=i0       �	��7Xfc�A�*

loss�&=Q��p       �	d@8Xfc�A�*

loss .<�ULf       �	/�8Xfc�A�*

lossXx�;��
U       �	�u9Xfc�A�*

loss{�<��       �	g:Xfc�A�*

lossX�`<>ғ4       �	��:Xfc�A�*

lossz#<��       �	�U;Xfc�A�*

lossMQ1=��       �	[�;Xfc�A�*

lossf̀<J��4       �	�<Xfc�A�*

loss��Q=���       �	�'=Xfc�A�*

lossx�N=��       �	��=Xfc�A�*

loss t�<]��       �	�Z>Xfc�A�*

loss���<�D%P       �	]S@Xfc�A�*

lossHJ�=����       �	3�@Xfc�A�*

loss�F�;��`       �	��AXfc�A�*

loss��=���       �	�BXfc�A�*

loss}b�<A���       �	MICXfc�A�*

loss�Ww=	�        �	g�CXfc�A�*

loss=&p��       �	H�DXfc�A�*

losso
\;=B��       �	�gEXfc�A�*

loss���;�M*4       �	p�FXfc�A�*

loss�VF=��R�       �	�}GXfc�A�*

lossƗ<�ߌ       �	HHXfc�A�*

lossosi=���       �	��HXfc�A�*

loss��<hf       �	�?IXfc�A�*

loss���;�8       �	��IXfc�A�*

loss��`=� ��       �	�vJXfc�A�*

loss�~1=p�_�       �	~KXfc�A�*

loss���<���       �	}�KXfc�A�*

loss�l>���       �	�FLXfc�A�*

losst�<G�1        �	��LXfc�A�*

loss7�=k6�       �	�xMXfc�A�*

losss�;Lb(�       �	�	NXfc�A�*

loss�n�<�w�       �	q�NXfc�A�*

loss��5<�c�l       �	�GOXfc�A�*

loss{f�<�n�W       �	#�OXfc�A�*

loss��&=[�@d       �	�{PXfc�A�*

loss�:�<Y��       �	�hQXfc�A�*

lossst<QuN       �	pBRXfc�A�*

losst�<~lѩ       �	�RXfc�A�*

loss���<� �       �	${SXfc�A�*

losslg�<��       �	�TXfc�A�*

lossکa=1Ol�       �	 �TXfc�A�*

loss�<�Ȑ�       �	aSUXfc�A�*

lossH~=ϟIV       �	:�UXfc�A�*

loss6r<�2�       �	A�VXfc�A�*

lossDnx<l�       �	�WXfc�A�*

lossz�J=�=m�       �	��WXfc�A�*

lossv��;T�Ȗ       �	�LXXfc�A�*

lossZ_�<���@       �	�XXfc�A�*

loss�@=�V��       �	|YXfc�A�*

loss��	=jaϙ       �	ZXfc�A�*

loss(�w;<��J       �	��ZXfc�A�*

lossIdx=���t       �	9C[Xfc�A�*

loss�?k<�#@�       �	H�[Xfc�A�*

loss�r=KOGl       �	�y\Xfc�A�*

loss)��;Uђ\       �	�]Xfc�A�*

loss�<W�       �	e�]Xfc�A�*

loss�7�=��        �	?^Xfc�A�*

loss.ӗ<V��       �	��^Xfc�A�*

lossl\<�3�       �	ji_Xfc�A�*

loss�'[<j��S       �	`Xfc�A�*

lossfU�=N��7       �	5�`Xfc�A�*

loss��=ڂ�A       �	I.aXfc�A�*

lossF�<̩�H       �	��aXfc�A�*

loss�O�;ݳL       �	>YbXfc�A�*

loss��;��^�       �	D�bXfc�A�*

loss�	=q��>       �	��cXfc�A�*

loss��D=�P��       �	_@dXfc�A�*

loss��;�DV�       �	q�dXfc�A�*

lossѢ=���4       �	��eXfc�A�*

loss�C�<>�a�       �	��fXfc�A�*

loss���<�/Ÿ       �	$CgXfc�A�*

loss�]K=�wY       �	2hXfc�A�*

loss�o�:�kT7       �	o�hXfc�A�*

loss�H�;N��       �	:iXfc�A�*

loss<��=X��       �	��iXfc�A�*

loss��=)�       �	zpjXfc�A�*

loss���<���       �	�kXfc�A�*

loss_��<`�       �	}�kXfc�A�*

loss/%?<�N"�       �	��lXfc�A�*

lossá~<�_c�       �	{1mXfc�A�*

loss#��;�:TV       �	��mXfc�A�*

loss8��<�	�       �	"onXfc�A�*

loss��<}��/       �	�oXfc�A�*

loss�Z�;DHVb       �	��oXfc�A�*

loss�=p�(n       �	��pXfc�A�*

loss?�;�[�       �	|qXfc�A�*

loss�C<�r�       �	PrXfc�A�*

loss��=�#�       �	B�rXfc�A�*

loss{-L=��+f       �	�GsXfc�A�*

loss =�T�y       �	��sXfc�A�*

lossV�=��x       �	�otXfc�A�*

loss��=�>�       �	uXfc�A�*

loss�4R=)g��       �	R�uXfc�A�*

loss�aP=���       �	n5vXfc�A�*

loss�'�<���       �	�vXfc�A�*

lossW<7e       �	F~wXfc�A�*

lossj��;�Y8�       �	�xXfc�A�*

lossя;�K�       �	�xXfc�A�*

loss���<H�;�       �	�WyXfc�A�*

loss{+<[��       �	�yXfc�A�*

loss�d=7r��       �	�zXfc�A�*

loss��.=����       �	x�{Xfc�A�*

loss�E<�dx�       �	_|Xfc�A�*

loss��O=�Z�G       �	��|Xfc�A�*

loss)&=�K4       �	�}Xfc�A�*

loss	y=����       �	|G~Xfc�A�*

loss[E�<8V�       �	��~Xfc�A�*

loss�z�<k���       �	�Xfc�A�*

loss4D<��V8       �	�.�Xfc�A�*

lossڳ;�M�       �	,׀Xfc�A�*

lossw-q<_x�       �	�n�Xfc�A�*

loss��<��?       �	{�Xfc�A�*

loss���;��q[       �	)��Xfc�A�*

loss�Y< �       �	�Y�Xfc�A�*

loss�"=�Pq�       �	L��Xfc�A�*

loss8<th       �	R��Xfc�A�*

lossCy�;��w�       �	D4�Xfc�A�*

lossZ1=Lɭ#       �	@݅Xfc�A�*

loss�'�<���       �	A��Xfc�A�*

lossv�]=�m       �	��Xfc�A�*

lossA�
=߻�        �	�ćXfc�A�*

loss	�S;_�uY       �	fj�Xfc�A�*

lossd��;�F{       �	1
�Xfc�A�*

loss�uB<ML�j       �	��Xfc�A�*

lossSY�=��\n       �	�L�Xfc�A�*

loss��K=Mo>�       �	��Xfc�A�*

loss]��<�%�       �	���Xfc�A�*

loss��<=��*       �	� �Xfc�A�*

loss��<�7�       �	���Xfc�A�*

loss�͐<�q*       �	�Z�Xfc�A�*

loss�H�;WO��       �	��Xfc�A�*

loss�'=�rt       �	���Xfc�A�*

lossl7=k�xh       �	.:�Xfc�A�*

loss{#=oS       �	ڏXfc�A�*

lossک=�J8�       �	�Xfc�A�*

loss&}=V�c�       �	��Xfc�A�*

loss�1�=1��       �	׿�Xfc�A�*

loss ?�;ܲ�       �	�d�Xfc�A�*

loss�@=I�B�       �	%�Xfc�A�*

loss7x<#&       �	���Xfc�A�*

loss��N<�5�       �	�H�Xfc�A�*

loss� �;!�h       �	��Xfc�A�*

lossiZ�<�^B       �	�|�Xfc�A�*

lossQT=�F�?       �	��Xfc�A�*

lossE��<��       �	)�Xfc�A�*

loss��J<ZnQ       �	街Xfc�A�*

loss��c=�u       �	�;�Xfc�A�*

lossXf;Ŕ�       �	�јXfc�A�*

loss��;��A       �	k�Xfc�A�*

loss�4L<�V��       �	]�Xfc�A�*

loss�L<$>��       �	���Xfc�A�*

loss��<r�	y       �	F`�Xfc�A�*

loss�=v�(       �	��Xfc�A�*

loss3s�<'�'2       �	r��Xfc�A�*

loss�e= ܻ        �	A�Xfc�A�*

loss�=��L�       �	�ߝXfc�A�*

lossA;�"       �	�s�Xfc�A�*

loss�X�<H/�       �	��Xfc�A�*

loss��<����       �	���Xfc�A�*

loss��!=��"B       �	[D�Xfc�A�*

lossG��<cR�~       �	�ݠXfc�A�*

loss�H�=����       �	�u�Xfc�A�*

loss�M�<!��       �	Z�Xfc�A�*

loss�#<g��       �	��Xfc�A�*

loss��;pΈ7       �	0I�Xfc�A�*

loss�ܬ<Q�b�       �	��Xfc�A�*

loss���;��(       �	���Xfc�A�*

loss�@=|�q�       �	,I�Xfc�A�*

loss��
>��       �	���Xfc�A�*

loss�)�;Ȯm	       �	-��Xfc�A�*

lossl��:oj)�       �	���Xfc�A�*

lossr�e;7�,<       �	G�Xfc�A�*

loss��<�t�       �	�N�Xfc�A�*

lossKh�<xzt       �	f2�Xfc�A�*

loss��<�       �	�تXfc�A�*

loss��<@-?x       �	���Xfc�A�*

loss2	<-���       �	���Xfc�A�*

loss0 #=��\       �		��Xfc�A�*

loss�jH<$9�$       �	�<�Xfc�A�*

losse�<Q� �       �	��Xfc�A�*

loss��N=�5q       �	���Xfc�A�*

lossl��<����       �	k��Xfc�A�*

loss�E�:w��       �	q�Xfc�A�*

loss�=�a       �	L�Xfc�A�*

loss���<85�1       �	��Xfc�A�*

lossU�=H�%�       �	���Xfc�A�*

lossԻ<JU3�       �	�U�Xfc�A�*

loss���<B��h       �	���Xfc�A�*

loss�f�;W��c       �	`��Xfc�A�*

losss�<���       �	b.�Xfc�A�*

loss�A�<��*�       �	�ҶXfc�A�*

lossŝ�=[�U       �	o�Xfc�A�*

loss[�^=��/?       �	 �Xfc�A�*

loss��<w�       �	b��Xfc�A�*

loss�Y�;4cs�       �	&:�Xfc�A�*

loss�vJ=	��       �	�йXfc�A�*

loss
��<��ܰ       �	Hn�Xfc�A�*

loss�q;4{B       �	)�Xfc�A�*

loss�;۳��       �	J��Xfc�A�*

loss�u7<g�7       �	�3�Xfc�A�*

loss8	�;,o�       �	 ʼXfc�A�*

loss`�K=K���       �	�^�Xfc�A�*

lossѓ�=���       �	+��Xfc�A�*

loss��m=0�$       �	���Xfc�A�*

lossnj�;b��       �	BZ�Xfc�A�*

loss�)<k���       �	��Xfc�A�*

loss���:�Ă       �	��Xfc�A�*

loss���;�%A       �	��Xfc�A�*

loss��<|��       �	W��Xfc�A�*

loss�]=��F       �	�M�Xfc�A�*

lossԗ�:Q���       �	='�Xfc�A�*

loss��R<��>X       �	ϻ�Xfc�A�*

loss�FZ=�A�+       �	�]�Xfc�A�*

loss\�'=�δ�       �	{��Xfc�A�*

loss{N�<�o~7       �	w��Xfc�A�*

lossܙ�=k��       �	{O�Xfc�A�*

lossR��<�4�       �	���Xfc�A�*

loss��<[. �       �	,��Xfc�A�*

loss��<юu       �	�>�Xfc�A�*

loss;�q<��I       �	��Xfc�A�*

lossԵ�=�H��       �	���Xfc�A�*

lossl~�<��$�       �	�T�Xfc�A�*

loss���=P>v       �	���Xfc�A�*

loss��;�       �	���Xfc�A�*

loss��<�n       �	��Xfc�A�*

lossܺ<�b��       �	��Xfc�A�*

loss��=/[�       �	�L�Xfc�A�*

loss�J�; wh�       �	���Xfc�A�*

loss:*�<����       �	���Xfc�A�*

loss�`�<�2C       �	�8�Xfc�A�*

loss}Na<��Z       �	g��Xfc�A�*

loss���<��L{       �	m�Xfc�A�*

loss���=#֞c       �	��Xfc�A�*

loss;p=�;       �	ݗ�Xfc�A�*

loss�.�<����       �	8/�Xfc�A�*

lossd�<�t�       �	*��Xfc�A�*

loss�Q=M�
�       �	h�Xfc�A�*

loss;<<�s�D       �	&�Xfc�A�*

lossI�=�[       �	_��Xfc�A�*

loss�q1=<B�y       �	�4�Xfc�A�*

loss0��=��       �	���Xfc�A�*

loss���<hk:@       �	�g�Xfc�A�*

loss�)�=$C�!       �	"��Xfc�A�*

loss��<4�+�       �	5��Xfc�A�*

loss�F�<Pa#       �	v5�Xfc�A�*

loss�t= ��       �	e��Xfc�A�*

loss��<�P��       �	~�Xfc�A�*

loss�ـ<��uh       �	�Xfc�A�*

lossMt<Hp��       �	���Xfc�A�*

lossF�;P�
�       �	I�Xfc�A�*

loss��<!�        �	n��Xfc�A�*

loss���<m2];       �	x�Xfc�A�*

lossvUJ<�[�       �	��Xfc�A�*

lossz�2=��`�       �	���Xfc�A�*

loss��<��2       �	X<�Xfc�A�*

loss�Ո=2��       �	|��Xfc�A�*

loss}Nq;qHݓ       �	Uk�Xfc�A�*

loss$u�;p���       �	��Xfc�A�*

loss�*�<ӯ��       �	=��Xfc�A�*

loss[�$=]<2�       �	H5�Xfc�A�*

loss�j�;�~C6       �	���Xfc�A�*

loss�<>lv�       �	�_�Xfc�A�*

lossߜ3=7|��       �	6�Xfc�A�*

loss���<��˹       �	���Xfc�A�*

loss�"1=$ey�       �	H3�Xfc�A�*

loss@�5=��%�       �	���Xfc�A�*

loss#�<*�P       �	�a�Xfc�A�*

loss�gb<c��,       �	��Xfc�A�*

lossM�<��1       �	\��Xfc�A�*

loss'I�=#�j�       �	B��Xfc�A�*

loss0.=O�x�       �	k*�Xfc�A�*

loss�Ŕ<iT�       �	���Xfc�A�*

lossz�<�N       �	���Xfc�A�*

lossMI�;�x��       �	���Xfc�A�*

loss��=OX�<       �	��Xfc�A�*

loss�ĵ<�c8       �	�c�Xfc�A�*

losstЬ=�Y       �	*�Xfc�A�*

loss�e=O�N       �	��Xfc�A�*

loss�:�<�6�6       �	d[�Xfc�A�*

loss=N�<�֥�       �	/��Xfc�A�*

lossܦp=��       �	���Xfc�A�*

loss��M=Tc��       �	�I�Xfc�A�*

lossߓ}=�(��       �	�-�Xfc�A�*

loss��<w?��       �	���Xfc�A�*

loss	��<�d>       �	T��Xfc�A�*

lossA�<����       �	�4�Xfc�A�*

loss_@<�A��       �	���Xfc�A�*

loss*�[<�.v�       �	~�Xfc�A�*

lossqX=<IWh�       �	� �Xfc�A�*

lossJ2=�       �	Ի�Xfc�A�*

loss_B�<UL�       �	2U�Xfc�A�*

loss��A<����       �	+��Xfc�A�*

lossjw<H��       �	���Xfc�A�*

loss_�D=�j��       �	�G�Xfc�A�*

loss��J<�B[�       �	�O�Xfc�A�*

loss6�	=*w��       �	�3�Xfc�A�*

loss���=^��Q       �	R��Xfc�A�*

lossv5<�        �	nm�Xfc�A�*

loss��<%�ԁ       �	|��Xfc�A�*

loss�[<	8�p       �	*�Xfc�A�*

loss�<FG��       �	k��Xfc�A�*

loss�lx;��s�       �	p[�Xfc�A�*

loss���<�N�	       �	 �Xfc�A�*

loss��D=�3��       �	̚�Xfc�A�*

loss��-<���L       �	+4 Yfc�A�*

loss�W<�� �       �	�� Yfc�A�*

loss�L.<�5``       �	_Yfc�A�*

loss]��<LO�s       �	�Yfc�A�*

lossò^<g��       �	�Yfc�A�*

lossJ+�;~n�y       �	��Yfc�A�*

lossnV�<E�٣       �	�#Yfc�A�*

loss ;?=	R�       �	��Yfc�A�*

lossSN=y�h       �	��Yfc�A�*

loss���;��%       �	�Yfc�A�*

lossry%=XM�q       �	WBYfc�A�*

loss}//<F�        �	��Yfc�A�*

loss�=� �}       �	"qYfc�A�*

loss14O<���       �	9		Yfc�A�*

lossx<u<��z       �	�,
Yfc�A�*

lossI��<���       �	��
Yfc�A�*

loss��)<	(�*       �	0cYfc�A�*

loss��<���       �	Yfc�A�*

loss�n�<�o�c       �	��Yfc�A�*

loss�=;���       �	�lYfc�A�*

loss78=ߋ&^       �	�Yfc�A�*

loss}[ <}Ī       �	��Yfc�A�*

loss�{:<��O�       �	dYfc�A�*

lossv�<ZRLc       �	)Yfc�A�*

losseY=�B*       �	�Yfc�A�*

loss�{<��       �	h[Yfc�A�*

lossM��<�oN       �	�	Yfc�A�*

loss���<��       �	�Yfc�A�*

loss}y>c�/�       �	TTYfc�A�*

loss=ĸ;��~       �	�Yfc�A�*

loss��;(�)       �	��Yfc�A�*

loss]��=��}       �	�KYfc�A�*

loss�L=`�Q�       �	�Yfc�A�*

lossVC�<@�}�       �	��Yfc�A�*

lossnX=��I       �	.Yfc�A�*

loss<b);j>ƛ       �	��Yfc�A�*

loss��?=��s       �	!uYfc�A�*

loss��=�R�       �	nYfc�A�*

loss��=�<�       �	��Yfc�A�*

loss@j�<,��       �	irYfc�A�*

loss���<�ۇ�       �	�Yfc�A�*

loss�:�;]Lc'       �	ƥYfc�A�*

loss�T�;Lg�v       �	PQYfc�A�*

lossCa4=I��1       �	��Yfc�A�*

loss�GR<,��Y       �	�� Yfc�A�*

loss�:�;��       �	;:!Yfc�A�*

losso��<��       �	�!Yfc�A�*

lossO�8=h�yz       �	0�"Yfc�A�*

loss6r<U㤴       �	32#Yfc�A�*

loss:h<�y�V       �	*�#Yfc�A�*

lossʟ_<�'�3       �	�$Yfc�A�*

lossi|�;X���       �	�*%Yfc�A�*

loss@Mz<�       �	_�%Yfc�A�*

loss
m-=Cؕ�       �	b�'Yfc�A�*

loss�;�=b+��       �	��(Yfc�A�*

loss�B=<�w       �	b*Yfc�A�*

loss �=.
T       �	��*Yfc�A�*

loss�Ѯ;�%L�       �	�+Yfc�A�*

loss1L�<��>       �	+�,Yfc�A�*

loss�S�=cJ�       �	�X-Yfc�A�*

loss�Q=�;��       �	K.Yfc�A�*

lossUj�<�	q       �	8�.Yfc�A�*

lossß�<Z�N�       �	U/Yfc�A�*

lossC	�=�[�9       �	�0Yfc�A�*

loss��<g:�       �	��0Yfc�A�*

loss�<�@�&       �	�a1Yfc�A�*

loss���=�j       �	��1Yfc�A�*

loss ��=�o       �	#�2Yfc�A�*

loss7=	<,�       �	rN3Yfc�A�*

loss�A6<S�|       �	��3Yfc�A�*

loss�N<A)�       �	��4Yfc�A�*

loss*�<���       �	�;5Yfc�A�*

loss�"=Kڔ�       �	n�5Yfc�A�*

loss2�u;	@=�       �	��6Yfc�A�*

loss�!;Oۃ       �	k7Yfc�A�*

loss2V</�+F       �	�	8Yfc�A�*

loss���<,�!�       �	��8Yfc�A�*

loss;�==�2       �	�>9Yfc�A�*

loss��<����       �	&�9Yfc�A�*

loss`a�;i�       �	��:Yfc�A�*

loss.Z�;;�0k       �	F(;Yfc�A�*

lossS�<�;�       �	��;Yfc�A�*

loss��<4	'o       �	�n<Yfc�A�*

lossZ�w=�hS       �	�=Yfc�A�*

loss��d=1�}Q       �	]�=Yfc�A�*

loss<we<���       �	�Z>Yfc�A�*

loss�@�<���E       �	8M?Yfc�A�*

lossv�<?�U       �	�@Yfc�A�*

loss�A�<ս8A       �	�qAYfc�A�*

loss��L;�P��       �	ZBYfc�A�*

loss��;Q�/       �	��BYfc�A�*

loss��U<��]�       �	;TCYfc�A�*

loss~�=�k�5       �	r�CYfc�A�*

loss�}<=�
y'       �	�DYfc�A�*

lossJ��<5�[       �	�MEYfc�A�*

lossl�;�~�+       �	��EYfc�A�*

loss�2e=��K�       �	D�FYfc�A�*

loss�=�H�       �	k+GYfc�A�*

lossXK(=��Z       �	�GYfc�A�*

loss��><=1�       �	�IJYfc�A�*

loss�=�|:�       �	��JYfc�A�*

loss�#><��M       �	��KYfc�A�*

loss�:�<5�\�       �	�9LYfc�A�*

loss��_<��E       �	�NYfc�A�*

loss-��<�\�B       �	��NYfc�A�*

loss��;U�(O       �	�eOYfc�A�*

loss�A*=0/X�       �	cPYfc�A�*

loss-�=\�Tn       �	)�PYfc�A�*

loss`,
=1�K�       �	�MQYfc�A�*

loss G�<W���       �	2�QYfc�A�*

loss���;u�Us       �	��RYfc�A�*

loss�L�;?�H       �	�'SYfc�A�*

lossn�=P0_�       �	2�SYfc�A�*

lossn��<Y,       �	�lTYfc�A�*

loss���;�d<�       �	�UYfc�A�*

loss��a<�B��       �	8�UYfc�A�*

loss���<յj�       �	2<VYfc�A�*

loss�p�=����       �	D�VYfc�A�*

lossX3�<��       �	�XYfc�A�*

lossv6=��^       �	J�XYfc�A�*

loss�BP;6��       �	�SYYfc�A�*

loss�EO<���       �	0�YYfc�A�*

lossD�=���Q       �	΍ZYfc�A�*

loss�>�;0+�.       �	�/[Yfc�A�*

loss��2=�:B�       �	\�[Yfc�A�*

loss��Y<�+��       �	n\Yfc�A�*

loss��4=�!�       �	:]Yfc�A�*

loss�ӂ<&��       �	4�]Yfc�A�*

loss��9<�G�D       �	�c^Yfc�A�*

lossI�X<)�	�       �	�S`Yfc�A�*

loss.��<�O�O       �	l	aYfc�A�*

loss���<Z3ך       �	1�aYfc�A�*

loss��=�I��       �	�ZbYfc�A�*

lossEN><H�|       �	ScYfc�A�*

lossQp�<��"�       �	�cYfc�A�*

lossT�:��       �	�KdYfc�A�*

loss�W<!4ȋ       �	��dYfc�A�*

loss=$�<}��;       �	˄eYfc�A�*

loss���;@?,�       �	2!fYfc�A�*

loss�^�<͉��       �	G�fYfc�A�*

loss�ʖ<=��y       �	��gYfc�A�*

loss���=���?       �	�>hYfc�A�*

loss�5,<8ұ       �	7 iYfc�A�*

loss(5�<�       �	�jYfc�A�*

loss�;�0
Z       �	%�jYfc�A�*

loss��N;	5�       �	�lYfc�A�*

loss��8!�h       �	��lYfc�A�*

loss"� <�kW�       �	�amYfc�A�*

loss=�-;�_�       �	,-nYfc�A�*

loss��;�0}       �	rjoYfc�A�*

loss���;��1       �	+pYfc�A�*

loss��j:n�-�       �	Y�pYfc�A�*

loss�)�<�be       �	�aqYfc�A�*

loss˓;�*K�       �	q�rYfc�A�*

loss<��9x���       �	k�sYfc�A�*

loss���9V�J       �	O�tYfc�A�*

loss�K:ͥ�       �	��uYfc�A�*

loss�[�<��       �	�lvYfc�A�*

loss�s+<�s˴       �	�PwYfc�A�*

losse��9�5�       �	�wYfc�A�*

loss��;iCj�       �	�xYfc�A�*

loss+�>T��       �	��yYfc�A�*

loss	�;�f       �	GqzYfc�A�*

loss���;0[Z�       �	�{Yfc�A�*

loss��Y=�`�y       �	��{Yfc�A�*

loss_<=�[       �	ۤ|Yfc�A�*

lossI�<R<7�       �	��}Yfc�A�*

loss�F-<�S��       �	�K~Yfc�A�*

loss�7e<�f�       �	�tYfc�A�*

loss �D=��i       �	Yfc�A�*

loss��;e�       �	�t�Yfc�A�*

loss��<\�ʄ       �	���Yfc�A�*

loss�k;��5�       �	�M�Yfc�A�*

loss�3<���       �	��Yfc�A�*

lossf=���M       �	w��Yfc�A�*

loss#5[<XEuu       �	y!�Yfc�A�*

loss�0=3��j       �	�Yfc�A�*

loss�_�;j���       �	MM�Yfc�A�*

loss�8�<P��       �	��Yfc�A�*

loss?<����       �	ǁ�Yfc�A�*

loss��=��{       �	C�Yfc�A�*

loss�=���       �	���Yfc�A�*

loss3�<.þ       �	M�Yfc�A�*

lossT�<?       �	G�Yfc�A�*

loss��+=*/�'       �	j��Yfc�A�*

lossH��:��v�       �	P�Yfc�A�*

loss��;XD�j       �	W��Yfc�A�*

lossr��;�H�       �	#L�Yfc�A�*

loss�%"<z��       �	��Yfc�A�*

loss�<w��       �	���Yfc�A�*

loss���< h	N       �	#�Yfc�A�*

loss[DQ=��       �	�ĎYfc�A�*

loss���;D��Q       �	�̏Yfc�A�*

loss�<==�YZ       �	���Yfc�A�*

lossz֔;D�s�       �	��Yfc�A�*

loss���;�f�       �	�Yfc�A�*

loss�/�<��t       �	�K�Yfc�A�*

lossS�<�H�U       �	y�Yfc�A�*

lossV��<ݨ�1       �	$��Yfc�A�*

loss�!&=�e�P       �	?�Yfc�A�*

loss��<��x�       �	���Yfc�A�*

lossS�=�HX�       �	UM�Yfc�A�*

loss�5�<$M/o       �	�Yfc�A�*

lossz�{=ɭ��       �	؂�Yfc�A�*

lossC�<�j7�       �	��Yfc�A�*

loss;�(=]���       �	>��Yfc�A�*

loss�M_<?�*       �	*X�Yfc�A�*

loss>�<;]�o       �	���Yfc�A�*

loss�1E=ې.=       �	ʋ�Yfc�A�*

loss�(,=Q&8       �	/�Yfc�A�*

loss5 �=!+QG       �	�ԛYfc�A�*

loss�P�:���!       �	�y�Yfc�A�*

loss-�x<��       �	��Yfc�A�*

loss��q=����       �	Z��Yfc�A�*

loss�J=�5Vp       �	�8�Yfc�A�*

loss�=���       �	xеYfc�A�*

loss��<'�h       �	j�Yfc�A�*

loss��<��s�       �	��Yfc�A�*

loss��#<k�qh       �	+��Yfc�A�*

loss���<S��       �	?�Yfc�A�*

loss��=TkS       �	�۸Yfc�A�*

loss���<qwkw       �	9}�Yfc�A�*

lossq	5<��	       �	��Yfc�A�*

loss���<�r�       �	(��Yfc�A�*

loss#�9<�?4D       �	mV�Yfc�A�*

loss�N=Z$*       �	���Yfc�A�*

lossiXe<��p       �	���Yfc�A�*

loss�Z=4>yS       �	�A�Yfc�A�*

loss`�<�,�       �	2�Yfc�A�*

loss�z6:w}��       �	���Yfc�A�*

loss�Hy<K�)       �	�,�Yfc�A�*

loss
=E<��(       �	ȿYfc�A�*

loss�8=��       �	�n�Yfc�A�*

loss>�;�RƓ       �	�+�Yfc�A�*

loss�2*=o��N       �	 �Yfc�A�*

loss���;���       �	���Yfc�A�*

lossa%�<$#�       �	���Yfc�A�*

lossǉ<F�       �	��Yfc�A�*

loss��	<��ع       �	F��Yfc�A�*

lossד�<�}�       �	&n�Yfc�A�*

lossz�'<���?       �	.��Yfc�A�*

loss���<��c       �	�I�Yfc�A�*

loss��=!9��       �	��Yfc�A�*

loss�V�;xTӬ       �	U��Yfc�A�*

loss��=�s�       �	��Yfc�A�*

loss`HP=<pJS       �	A+�Yfc�A�*

loss��<�z�       �	��Yfc�A�*

lossa(=([Y       �	�a�Yfc�A�*

loss�͈<	�LJ       �	�Yfc�A�*

loss�u�;:XU�       �	/��Yfc�A�*

lossO��<RQ�#       �	d;�Yfc�A�*

loss�v>;R�       �	���Yfc�A�*

loss&�:=<��Z       �	�k�Yfc�A�*

lossW$><+��&       �	:�Yfc�A�*

loss�:;=�}>�       �	Ę�Yfc�A�*

loss=a=3��       �	<�Yfc�A�*

loss�o<�.э       �	t��Yfc�A�*

loss�v<L�Ϻ       �	Mj�Yfc�A�*

loss�A�;�߇�       �	��Yfc�A�*

loss�]<a��       �	��Yfc�A�*

loss�<y�+       �	Q��Yfc�A�*

loss���<���       �	�=�Yfc�A�*

loss��:�{*       �	=��Yfc�A�*

losstD�<���       �	�i�Yfc�A�*

loss��<~<W\       �	!�Yfc�A�*

loss�;���p       �	���Yfc�A�*

loss�ZM=�C"�       �	�H�Yfc�A�*

loss�0<q��       �	��Yfc�A�*

loss��<0nlq       �	�w�Yfc�A�*

lossh!�:�Tψ       �	[
�Yfc�A�*

loss��:�Yߨ       �	��Yfc�A�*

losso�<��       �	�7�Yfc�A�*

loss)=�q�       �	���Yfc�A�*

losscZ�=W       �	f�Yfc�A�*

loss�b�;s7�       �	��Yfc�A�*

loss6��:��X(       �	ӥ�Yfc�A�*

loss�ʄ<���"       �	AD�Yfc�A�*

loss��<��_�       �	M��Yfc�A�*

loss���:���       �	�q�Yfc�A�*

loss�,�;�<�'       �	�Yfc�A�*

loss8A�;�<��       �	���Yfc�A�*

loss�80=�+T       �	#-�Yfc�A�*

loss(��<�ȃ�       �	���Yfc�A�*

lossz��<֑շ       �	�\�Yfc�A�*

loss�,�<�~       �	{��Yfc�A�*

loss�E�<}ǭ�       �	ۉ�Yfc�A�*

loss��?<��(�       �	$�Yfc�A�*

loss�N<{���       �	U��Yfc�A�*

loss)��<V��       �	f�Yfc�A�*

loss�Y<"�f�       �	C��Yfc�A�*

lossx��<�l6       �	|��Yfc�A�*

loss��<@1�I       �	]2�Yfc�A�*

lossJ�<v'�       �	��Yfc�A�*

loss��;=�j�       �	���Yfc�A�*

loss{�<]�4W       �	�?�Yfc�A�*

loss�TC=��0       �	Z��Yfc�A�*

lossL�|<޽�       �	�x�Yfc�A�*

loss<����       �	��Yfc�A�*

loss�-�<9�j       �	��Yfc�A�*

loss��U=��       �	��Yfc�A�*

loss�`�;A�=�       �	`��Yfc�A�*

loss/#<i��       �	�N�Yfc�A�*

loss���<c��       �	���Yfc�A�*

loss:�k<�3��       �	Ւ�Yfc�A�*

loss��=�$�o       �	0�Yfc�A�*

lossm�=J�o       �	��Yfc�A�*

loss�}�<)�Y�       �	_�Yfc�A�*

losst*�=/�!       �	Y��Yfc�A�*

lossc�<4�       �	���Yfc�A�*

lossN��=�Ǚ�       �	�2�Yfc�A�*

lossqJ�<�5�e       �	� �Yfc�A�*

loss���=B�T�       �	��Yfc�A�*

loss��<X>�       �	u��Yfc�A�*

lossA�;,�]�       �	�L�Yfc�A�*

loss��+;ݫ�B       �	�8�Yfc�A�*

loss%w�;1u�       �	���Yfc�A�*

loss��<�       �	yx�Yfc�A�*

loss��|<'|�       �	�#�Yfc�A�*

lossm?�<"_�R       �	���Yfc�A�*

loss%`<���"       �	}�Yfc�A�*

loss���<��n       �	��Yfc�A�*

loss?�=Ƕd&       �	��Yfc�A�*

lossT�+<@��v       �	�M�Yfc�A�*

loss��C=�vl.       �	���Yfc�A�*

loss���<Q@�       �	}��Yfc�A�*

loss6s=��V�       �	�c�Yfc�A�*

loss�H�; ��       �	���Yfc�A�*

loss��<9&       �	|��Yfc�A�*

loss=C�<Yzio       �	j4�Yfc�A�*

loss#�p<���f       �	!��Yfc�A�*

loss4T�<�ME       �	�e Zfc�A�*

loss4�<=Bba       �	X Zfc�A�*

loss�<��w�       �	S�Zfc�A�*

loss��;�}�       �	N)Zfc�A�*

lossLK2=F�*�       �	1�Zfc�A�*

lossV!!<ع��       �	��Zfc�A�*

lossa�=�=\4       �	�zZfc�A�*

loss���;a�:       �	�>Zfc�A�*

loss��=$��U       �	��Zfc�A�*

loss�!�<��[�       �	�xZfc�A�*

lossIMe<��       �	�Zfc�A�*

loss��=�}H�       �	G�Zfc�A�*

loss@q^<n,��       �	K<	Zfc�A�*

lossޡ<
�Щ       �	ji
Zfc�A�*

lossQ�<g��       �	�Zfc�A�*

loss*"�<|1�P       �	��Zfc�A�*

loss=#@�       �	��Zfc�A�*

loss��=+���       �	��Zfc�A�*

lossY�=LN�)       �	�1Zfc�A�*

loss{']=�Ճ�       �	��Zfc�A�*

loss3��<*I�       �	ǃZfc�A�*

loss� �<P$"�       �	%Zfc�A�*

lossF�>=����       �	��Zfc�A�*

loss x�;�jH�       �	quZfc�A�*

loss)��;���       �	3Zfc�A�*

loss��<J��       �	��Zfc�A�*

loss�,I<X��       �	7RZfc�A�*

lossP��=;��G       �	��Zfc�A�*

loss�
W< F�f       �	��Zfc�A�*

loss="�<��2       �	�!Zfc�A�*

loss���;s�`d       �	��Zfc�A�*

lossX��;�Z4k       �	yZZfc�A�*

lossJʳ<q�J�       �	$�Zfc�A�*

loss���;u�I       �	?�Zfc�A�*

loss/(�;�*Լ       �	pzZfc�A�*

loss,ވ<��O�       �	vZfc�A�*

loss8$8=� #a       �	M�Zfc�A�*

loss��>=& c       �	^eZfc�A�*

loss8��;GL       �	�
Zfc�A�*

loss=7<�n�z       �	y�Zfc�A�*

loss��r<`��       �	�JZfc�A�*

loss��4= {<n       �	��Zfc�A�*

loss���;���6       �	ׇZfc�A�*

loss�E=6)`       �	�#Zfc�A�*

loss�)�;�mY       �	�Zfc�A�*

loss�C1=�dfe       �	i Zfc�A�*

loss�j<놚�       �	w!Zfc�A�*

loss9�!=d��       �	7�!Zfc�A�*

loss�<�S       �	�T"Zfc�A�*

loss}ñ:��       �	R�"Zfc�A�*

loss3I�=j��       �	��#Zfc�A�*

loss�~9<��<�       �	�8$Zfc�A�*

loss�<���w       �	.�$Zfc�A�*

loss�'={�5�       �	 �%Zfc�A�*

losslq�;�PD�       �	�4&Zfc�A�*

loss|N�;��r       �	��&Zfc�A�*

loss��F<��       �	r�'Zfc�A�*

lossf$�<S��       �	��(Zfc�A�*

loss�T�<퓖�       �	�u)Zfc�A�*

loss�v<�Q�|       �	ƅ*Zfc�A�*

loss��<�n}�       �	�%+Zfc�A�*

loss���<��a       �	w�,Zfc�A�*

loss)�r;ө�J       �	}w-Zfc�A�*

lossb.;d��a       �	'1.Zfc�A�*

loss���<���H       �	N�.Zfc�A�*

lossm� =Y'c�       �	��/Zfc�A�*

lossl�:_��       �	}!0Zfc�A�*

loss\�@==��       �	��0Zfc�A�*

loss�K�< ��       �	�w1Zfc�A�*

loss��=����       �	72Zfc�A�*

loss�2�<>       �	��2Zfc�A�*

lossȘ0=u��       �	7O3Zfc�A�*

lossz��<~�.       �	��3Zfc�A�*

loss_}�=�7��       �	�x4Zfc�A�*

lossJ��<�l)       �	 65Zfc�A�*

loss�G�<�}o       �	L�5Zfc�A�*

loss��<��]�       �	)^6Zfc�A�*

loss�m�;����       �	E�6Zfc�A�*

loss}�<��$�       �	��7Zfc�A�*

loss��(;7O�       �	y"8Zfc�A�*

loss�<�:�N       �	��8Zfc�A�*

loss*�;�/��       �	/N9Zfc�A�*

loss��W<��k       �	�:Zfc�A�*

loss�C�;ˣBe       �	B�:Zfc�A�*

loss�ǝ;�0q^       �	�.;Zfc�A�*

loss'�;yO�@       �	��;Zfc�A�*

loss��<,oC       �	yX<Zfc�A�*

loss-q<.���       �	��<Zfc�A�*

loss�<�8�       �	��=Zfc�A�*

loss��{<:�0       �	�>Zfc�A�*

loss醸<�;|       �	��>Zfc�A�*

loss�ě<3@�M       �	4J?Zfc�A�*

loss���=��       �	v�?Zfc�A�*

lossm?�<��l       �	��@Zfc�A�*

loss�D=j�;       �	�"AZfc�A�*

lossqE;�*�       �	ϺAZfc�A�*

lossJ�&;�u2       �	�QBZfc�A�*

loss�!R<Nd�       �	��BZfc�A�*

loss�w=x?�       �	��CZfc�A�*

lossf_=S�Z�       �	�?DZfc�A�*

loss�+=\(�m       �	��DZfc�A�*

lossS�=^��)       �	)�EZfc�A�*

loss/�:=��       �	�-FZfc�A�*

lossmV
<Y��       �	`�FZfc�A�*

loss��V=��eD       �	I�GZfc�A�*

loss�/<z�       �	�HZfc�A�*

loss�|=
TDK       �	��HZfc�A�*

loss!��;��#�       �	aQIZfc�A�*

loss��E<EI�"       �	�IZfc�A�*

loss��=�m�       �	ڌJZfc�A�*

loss$�=�=��       �	�%KZfc�A�*

loss��;�gy       �	M�KZfc�A�*

loss{h�;�V7n       �	\uLZfc�A�*

loss��;�g܉       �	MZfc�A�*

lossHg;����       �	�MZfc�A�*

loss���=HSu       �	GNZfc�A�*

loss��<��U       �	��NZfc�A�*

loss"��<F�h�       �	�sOZfc�A�*

loss��=<���       �	�PZfc�A�*

loss��<�O�       �	��PZfc�A�*

loss�3�<����       �	d>QZfc�A�*

loss�*=8�D�       �	V�QZfc�A�*

loss�<X;O�       �	~qRZfc�A�*

loss�U�<8�7       �	�SZfc�A�*

losso�+<U+       �	�SZfc�A�*

loss�)S<�(�z       �	i9TZfc�A�*

lossO��<m��J       �	b�TZfc�A�*

loss��=�z
�       �	�nUZfc�A�*

loss��;�ײ       �	]�VZfc�A�*

loss�HX;�}�B       �	�-WZfc�A�*

loss�v=��g       �	�WZfc�A�*

lossh�	=�A�y       �	O�XZfc�A�*

losse�+<ʩ�L       �	�7YZfc�A�*

loss�e�<��ԍ       �	��YZfc�A�*

loss(�K<d/7+       �	��ZZfc�A�*

loss�0<ٔ�>       �	�l[Zfc�A�*

loss��;��'       �	4\Zfc�A�*

loss�N�;��	       �	��\Zfc�A�*

loss�_�;���       �	��]Zfc�A�*

loss�Z�<Gʲ�       �	C8^Zfc�A�*

loss��;䗘�       �	#�^Zfc�A�*

loss���<K�!       �	�y_Zfc�A�*

loss�7=��       �	�`Zfc�A�*

lossb�	=&S��       �	N�`Zfc�A�*

loss�u==t�°       �	B]aZfc�A�*

loss�I<3�!.       �	��aZfc�A�*

lossf��;y       �	y�bZfc�A�*

loss��<l�\       �	�/cZfc�A�*

loss�o'<f��9       �	�dZfc�A�*

loss=ڎ�Y       �	ޏfZfc�A�*

loss��g=�ު       �	$+gZfc�A�*

loss=��<w�h�       �	��gZfc�A�*

loss/�=�:=b       �	�YhZfc�A�*

loss��<x��       �	'�hZfc�A�*

loss]�U<�P�}       �	��iZfc�A�*

loss�.=����       �	-kZfc�A�*

loss�ɖ<y�S       �	�kZfc�A�*

loss�Ew=]��3       �	��lZfc�A�*

loss�^O=ǡ(�       �	_�mZfc�A�*

lossf7�;�R��       �	M�nZfc�A�*

loss���;�s�       �	�doZfc�A�*

loss�E�<b���       �	r1pZfc�A�*

loss��<�c��       �	�mqZfc�A�*

lossZ��;�;��       �	YrZfc�A�*

loss��n<k��       �	�sZfc�A�*

lossE��<]�[^       �	B�sZfc�A�*

loss��;}�ӽ       �	��tZfc�A�*

loss�q2=���       �	�muZfc�A�*

loss�VM<��s�       �	�vZfc�A�*

loss��m<����       �	��vZfc�A�*

loss��;�A{�       �	�WwZfc�A�*

lossʯ�<5ۯ3       �	/�wZfc�A�*

loss#r_;��{9       �	 �xZfc�A�*

lossH�<�5       �	�:yZfc�A�*

loss)�w<���       �	��yZfc�A�*

loss��W;g��u       �	�|zZfc�A�*

loss��<3c�       �	�{Zfc�A�*

loss�!�<e� �       �	��{Zfc�A�*

loss�l�=NNK�       �	�R|Zfc�A�*

loss0:�=�ݻ�       �	t�|Zfc�A�*

lossj�;���       �	L�}Zfc�A�*

lossʔ�<����       �	�(~Zfc�A�*

lossq	�<kջ�       �	��~Zfc�A�*

loss��;E��F       �	x^Zfc�A�*

loss{~�;�q��       �	��Zfc�A�*

loss7��;��2�       �	b��Zfc�A�*

lossĖ�=zдy       �	�;�Zfc�A�*

loss�$=!���       �	�ՁZfc�A�*

loss�W=�n�T       �	�l�Zfc�A�*

loss�{;�G��       �	B�Zfc�A�*

loss�=�?j�       �	���Zfc�A�*

loss��%=�!:�       �	�5�Zfc�A�*

loss@�<4F�       �	�фZfc�A�*

loss��<1V       �	3��Zfc�A�*

losst�<@��       �	�J�Zfc�A�*

loss��X<[;m
       �	���Zfc�A�*

loss�;^��V       �	�v�Zfc�A�*

lossZ�;���       �	k�Zfc�A�*

loss�%>=�12�       �	��Zfc�A�*

lossd��<���p       �	|F�Zfc�A�*

loss�=:S       �	��Zfc�A�*

loss;��<���       �	���Zfc�A�*

loss���<��o       �	�'�Zfc�A�*

loss`T<{[v�       �	H��Zfc�A�*

lossff;=�y�Y       �	F�Zfc�A�*

loss�/\=�Y�       �	sI�Zfc�A�*

lossӦ�<�_��       �	�Zfc�A�*

lossr� =S��       �	���Zfc�A�*

loss?�I<�L�{       �	�z�Zfc�A�*

loss���<`oô       �	��Zfc�A�*

loss���;�Aߗ       �	"��Zfc�A�*

loss��#=��m       �	c��Zfc�A�*

lossI��<dv��       �	�r�Zfc�A�*

lossA�@<ͯ0�       �	|�Zfc�A�*

loss�Z�;֧��       �	ҧ�Zfc�A�*

loss��;;��       �	�G�Zfc�A�*

loss7�%<w�       �	���Zfc�A�*

lossJ9�=�ڰB       �	%��Zfc�A�*

loss��;�M�       �	0*�Zfc�A�*

loss���<�?L       �	˾�Zfc�A�*

lossJ X=����       �	fi�Zfc�A�*

loss��<��M       �	G �Zfc�A�*

loss��<7x�       �	ȕ�Zfc�A�*

loss�:�8��       �	�/�Zfc�A�*

lossMb�;@�W�       �	A֚Zfc�A�*

lossi�;<,ǃ	       �	�|�Zfc�A�*

loss��Q;10��       �	� �Zfc�A�*

loss*d�=���~       �	���Zfc�A�*

loss�M'=�ڲ8       �	阝Zfc�A�*

loss�.�<���       �	9�Zfc�A�*

lossWO�<2t�       �	�՞Zfc�A�*

loss�OP=��7       �	qr�Zfc�A�*

lossN�k<���       �	g&�Zfc�A�*

loss�<);O�       �	�ĠZfc�A�*

loss�H<&�B�       �	3��Zfc�A�*

loss�TR<E�m       �	V+�Zfc�A�*

loss�}�<<���       �	E-�Zfc�A�*

loss�]<��/       �	ȣZfc�A�*

loss��<�ZQ�       �	_a�Zfc�A�*

loss��;�S]�       �	��Zfc�A�*

loss�go;?J}�       �	���Zfc�A�*

loss=	B:X       �	�P�Zfc�A�*

loss�u�<���y       �	��Zfc�A�*

loss�g�<��bR       �	a��Zfc�A�*

loss�s,<����       �	�f�Zfc�A�*

loss{�j=�[�g       �	Z�Zfc�A�*

lossE�}=�OU       �	���Zfc�A�*

loss#�=V<8       �	�:�Zfc�A�*

loss�LE<H��e       �	ժZfc�A�*

loss�.�=͌q�       �	8i�Zfc�A�*

loss�j<N��       �	���Zfc�A�*

loss55<���       �	}��Zfc�A�*

lossy�<B!�       �	;�Zfc�A�*

loss�j<O|m�       �	��Zfc�A�*

lossiaS;a��       �	Ӈ�Zfc�A�*

loss���<��       �	�&�Zfc�A�*

loss%��<GFR�       �	���Zfc�A�*

lossњd<���       �	�V�Zfc�A�*

loss\�<i�GN       �	Rd�Zfc�A�*

loss�ּ<!���       �	��Zfc�A�*

loss��<"��       �	w��Zfc�A�*

losssZ=��W       �	B�Zfc�A�*

loss\YG=+ӕ       �	��Zfc�A�*

loss�=�x�       �	��Zfc�A�*

loss=,��Z       �	`�Zfc�A�*

lossH�R<��T       �	(��Zfc�A�*

loss�L�;��c       �	&V�Zfc�A�*

loss=N*;���W       �	k�Zfc�A�*

loss�U�<.�^       �	L��Zfc�A�*

loss�p<*��^       �	�"�Zfc�A�*

lossA��=���o       �	���Zfc�A�*

loss��M;\t6�       �	�Q�Zfc�A�*

loss�И;���       �	$�Zfc�A�*

loss�
�;��-�       �	'��Zfc�A�*

loss��P<�	k       �	!�Zfc�A�*

loss4�V;�d�       �	g��Zfc�A�*

lossа<�eQ       �	YO�Zfc�A�*

loss(.�<ܧ�c       �	6�Zfc�A�*

lossZ�<=ð��       �	4��Zfc�A�*

lossF�;��4y       �	77�Zfc�A�*

loss��;	��K       �	};Zfc�A�*

loss�σ<J�t�       �	�c�Zfc�A�*

loss�A�<�%�O       �	���Zfc�A�*

loss��9<���       �	!��Zfc�A�*

loss1m<GbMc       �	�%�Zfc�A�*

loss��F=���       �	��Zfc�A�*

loss@=Y;^g�Y       �	Y��Zfc�A�*

loss��=��`       �	�G�Zfc�A�*

loss�<�mL-       �	%��Zfc�A�*

loss-(�<���*       �	��Zfc�A�*

lossE��<| �       �	��Zfc�A�*

loss��<�4^       �	���Zfc�A�*

loss��<c��[       �	P�Zfc�A�*

loss_g<c}�       �	���Zfc�A�*

loss`Ӱ<Sn�       �	y��Zfc�A�*

loss�=1�       �	�6�Zfc�A�*

loss12=��;�       �	s��Zfc�A�*

lossz�V=��/�       �	|�Zfc�A�*

lossKK=��m~       �	�!�Zfc�A�*

loss�o%;���       �	���Zfc�A�*

loss�;*Y4j       �	�v�Zfc�A�*

lossd�<��j       �	�Zfc�A�*

loss��C<�w�       �	Q��Zfc�A�*

loss�3�;i�ޅ       �	�{�Zfc�A�*

loss��=U�-+       �	 '�Zfc�A�*

loss��M<�5�V       �	���Zfc�A�*

loss6&J=&J��       �	��Zfc�A�*

lossVf=��N6       �	$�Zfc�A�*

loss�69=(^b�       �	���Zfc�A�*

loss�	�<�Y�       �	{�Zfc�A�*

lossw��<��       �	}�Zfc�A�*

loss�m�;J��       �	���Zfc�A�*

loss���<�h��       �	�f�Zfc�A�*

loss��=,��p       �	(�Zfc�A�*

loss�T�;�RЈ       �	���Zfc�A�*

loss�4V<��Es       �	M�Zfc�A�*

loss�s�<M�?�       �	x��Zfc�A�*

loss;�H=��p8       �	Ɗ�Zfc�A�*

loss�=����       �	�#�Zfc�A�*

loss=�<��h+       �	���Zfc�A�*

loss���<;FQ�       �	�v�Zfc�A�*

loss��<ZmԘ       �	e�Zfc�A�*

lossZ��<¬s       �	V��Zfc�A�*

loss��=;��{       �	W^�Zfc�A�*

loss���=�>�       �	��Zfc�A�*

loss�|=�#�       �	ܻ�Zfc�A�*

loss�S�<u��       �	�[�Zfc�A�*

loss��m;ݲ��       �	8��Zfc�A�*

loss���< �$�       �	z��Zfc�A�*

lossn�R=��	�       �	�S�Zfc�A�*

loss��R<yH�i       �	@��Zfc�A�*

loss���<���       �	J��Zfc�A�*

loss2�L=�)w/       �	
.�Zfc�A�*

loss$�<�2��       �	R��Zfc�A�*

loss�iw<���m       �	�z�Zfc�A�*

loss'�<���       �	U�Zfc�A�*

loss-r)=J,�        �	|��Zfc�A�*

loss��=$���       �	{M�Zfc�A�*

loss�e�=�^��       �	���Zfc�A�*

loss�G�<jl^y       �	���Zfc�A�*

loss�H<7�       �	�7�Zfc�A�*

lossr<,�       �	���Zfc�A�*

lossZxm=���t       �	-��Zfc�A�*

loss��;�_��       �	D�Zfc�A�*

loss��<Z��.       �	-C�Zfc�A�*

loss\'<����       �	V�Zfc�A�*

loss|�/=��C�       �	��Zfc�A�*

loss���<�s�1       �	�H�Zfc�A�*

loss��< J��       �	�@�Zfc�A�*

lossꁳ:H�~�       �	\>�Zfc�A�*

loss��
<Y��       �	�~�Zfc�A�*

losstdC<0���       �	�w�Zfc�A�*

lossw<�>?�       �	�q�Zfc�A�*

loss�?N=%Ŧ@       �	��Zfc�A�*

loss�<;�%�2       �	z��Zfc�A�*

lossS��=���       �	H��Zfc�A�*

lossζ=+�       �	��Zfc�A�*

loss3��<�9��       �	&��Zfc�A�*

loss���<XT2R       �	���Zfc�A�*

loss�i�;�Jq       �	���Zfc�A�*

loss)��<��r       �	w��Zfc�A�*

loss)�D<�$       �	��Zfc�A�*

lossZ �=���;       �	k�Zfc�A�*

loss��*=R}�       �	MH�Zfc�A�*

loss��W<���       �	+3�Zfc�A�*

loss�L-< �~t       �	w��Zfc�A�*

loss[y�:�|��       �	��Zfc�A�*

loss�@=۲j>       �	5�Zfc�A�*

lossP(�<��       �	Kw [fc�A�*

loss �<aC^       �	�I[fc�A�*

loss�Ø<6<�J       �	I+[fc�A�*

loss��;�̀\       �	j[fc�A�*

loss��=�       �	wH[fc�A�*

loss�E<��8�       �	s�[fc�A�*

losstf�<~��       �	u�[fc�A�*

loss��D;�+�       �	�6[fc�A�*

loss[��<���       �	��[fc�A�*

loss��<�k}       �	��[fc�A�*

loss1eq<����       �	9&[fc�A� *

lossA�
=�)"2       �	\�[fc�A� *

loss�I�;S-�       �	0
[fc�A� *

loss��0=�I�       �	X�
[fc�A� *

lossl�^=�XU       �	>B[fc�A� *

loss���<$�p�       �	B�[fc�A� *

loss���;S �       �	]�[fc�A� *

loss}@<���9       �	 &[fc�A� *

loss\��<K��       �	��[fc�A� *

loss�A�<;ci       �	�Q[fc�A� *

loss�L=}8��       �	��[fc�A� *

loss<'@<#̶z       �	.�[fc�A� *

lossf�;��       �	<-[fc�A� *

lossH��<���       �	�[fc�A� *

loss)�;���       �	B\[fc�A� *

loss��=X�       �	-	[fc�A� *

loss��v<�_0       �	3�[fc�A� *

loss�_<? �(       �	F[fc�A� *

loss�G=J��       �	:�[fc�A� *

loss�̦<'h�       �	��[fc�A� *

lossqR<���       �	�![fc�A� *

loss�=��z�       �	�[fc�A� *

loss�X<�ʂ       �	.V[fc�A� *

loss��<�/ZQ       �	]�[fc�A� *

loss�:=:�       �	$�[fc�A� *

loss���</Ħ       �	�3[fc�A� *

loss}�4=���       �	�[fc�A� *

loss
�Q<�,��       �	0e[fc�A� *

loss�<h���       �	�[fc�A� *

loss�{�<��       �	'�[fc�A� *

loss�|�<+>ʛ       �	�9[fc�A� *

loss��;�&�s       �	��[fc�A� *

loss�Q�;{�7       �	�j[fc�A� *

loss��=G�d�       �	�[fc�A� *

loss�M;��m       �	<�[fc�A� *

loss�$;]ʴ�       �	T[fc�A� *

loss\�)<DvyG       �	��[fc�A� *

loss+T;�4��       �	/�[fc�A� *

loss���9x�x       �	� [fc�A� *

loss6S;���O       �	[� [fc�A� *

lossN�'<��<�       �	QJ![fc�A� *

loss���;k�}P       �	r�![fc�A� *

loss�I�;,R�+       �	/�"[fc�A� *

loss�p�9��*�       �	%#[fc�A� *

loss	��;��|       �	��#[fc�A� *

loss,l�<ƒ_�       �	�W$[fc�A� *

lossr��:���       �	�%[fc�A� *

loss?�l;�H��       �	u�%[fc�A� *

loss�c6:<���       �	wN&[fc�A� *

loss��2<�b�l       �	$�&[fc�A� *

lossU�<)�"V       �	@�'[fc�A� *

loss�0:5���       �	�#([fc�A� *

loss5��<�j       �	F�([fc�A� *

loss4�e=�5�       �	�j)[fc�A� *

loss-\;�Us�       �	2*[fc�A� *

loss���;�-:�       �	��*[fc�A� *

lossѕ<�-W       �	�B+[fc�A� *

lossf�)<c]��       �	��+[fc�A� *

loss^��<^|�       �	\s,[fc�A� *

loss��_=�Ĳr       �	f-[fc�A� *

loss�)�=�5%�       �	��-[fc�A� *

lossځ�<�>"       �	�Z.[fc�A� *

loss�-�=�c�P       �	�.[fc�A� *

loss\�p<�viT       �	��/[fc�A� *

lossz��<�x��       �	�>0[fc�A� *

loss/��=w��       �	f�0[fc�A� *

lossĈ0<���       �	�z1[fc�A� *

loss,m<�k�       �	2[fc�A� *

loss��;=���       �	��2[fc�A� *

loss��j=q!�T       �	�D3[fc�A� *

loss��P<�Gp       �	/�3[fc�A� *

loss��e<#�Bf       �	:u4[fc�A� *

loss/Yu=�       �	
5[fc�A� *

loss_SU<a�!�       �	��5[fc�A� *

loss��5;�*�       �	�F6[fc�A� *

loss��~<���       �	��6[fc�A� *

lossT�<sIA       �	f�7[fc�A� *

loss $�:&�k�       �	68[fc�A� *

loss��:O	w       �	�8[fc�A� *

loss�V�<#�"       �	_b9[fc�A� *

loss�̴=�ڮ�       �	X�9[fc�A� *

loss���<�j�       �	(�:[fc�A� *

loss:��<�[�       �	?;[fc�A� *

loss��D=��;       �	��;[fc�A� *

loss��;͐�       �	�p<[fc�A� *

loss�L�<���       �	k=[fc�A� *

loss��;���       �	�=[fc�A� *

loss@��<7ꡧ       �	
K>[fc�A� *

loss:�<�K�       �	~�>[fc�A� *

loss�4D<����       �	�y?[fc�A� *

lossoɻ<GB�k       �	�@[fc�A� *

lossXoK=���       �	ɯ@[fc�A� *

loss�r�<�0Z�       �	2WA[fc�A� *

lossl:<�(�       �	��A[fc�A� *

loss��&<�(��       �	�B[fc�A� *

loss8$�<�H�C       �	R,C[fc�A� *

loss%[<T1�D       �	��C[fc�A� *

loss���<���P       �	keD[fc�A� *

lossĴ�;��^       �	U�D[fc�A� *

lossa/
=)b       �	��E[fc�A� *

loss��R=�- �       �	�5F[fc�A� *

lossr�;/���       �	s�F[fc�A� *

lossќ�<��#       �	?pG[fc�A� *

lossC��:@��       �	H[fc�A� *

loss??c;ׯe�       �	z�H[fc�A� *

lossd�i;�A[�       �	la[fc�A� *

loss��<�L��       �	�b[fc�A� *

losscK�<)��       �	�b[fc�A� *

loss���;��P�       �	�*c[fc�A� *

loss�<C=QrY�       �	��c[fc�A� *

loss%l<<�N�       �	"Rd[fc�A� *

loss�`4<A��       �	��d[fc�A� *

loss��)=���       �	G�e[fc�A� *

loss��w=q�P       �	J&f[fc�A� *

loss���<���       �	��f[fc�A� *

loss	�;�G��       �	Qig[fc�A� *

loss��-=:�;�       �	dh[fc�A� *

lossn��<`0�E       �	ۧh[fc�A� *

loss߻�<͝�V       �	,Ji[fc�A� *

lossTZ�<1<��       �	��i[fc�A� *

lossa��<]q��       �	ׄj[fc�A� *

loss�l�;�2�X       �	&k[fc�A� *

loss�	�;�l�       �	[�k[fc�A� *

loss"z=8X��       �	!Vl[fc�A� *

loss_.=,���       �	��l[fc�A� *

lossS+<&1��       �	Z�m[fc�A� *

loss���<��+~       �	�4n[fc�A� *

loss<�6/       �	��n[fc�A� *

lossv��=a��       �	�\o[fc�A�!*

loss��<*��r       �	��o[fc�A�!*

loss귡<���       �	�p[fc�A�!*

loss(��<�w�N       �	�$q[fc�A�!*

loss�=��ډ       �	�q[fc�A�!*

loss\�$=�f8�       �	�Or[fc�A�!*

loss���;g[        �	�r[fc�A�!*

loss1O�;eQ��       �	��s[fc�A�!*

loss
��;Ei       �	(t[fc�A�!*

loss�;��
�       �	��t[fc�A�!*

loss_�D=��       �	�Xu[fc�A�!*

loss�}�=~�h�       �	1�u[fc�A�!*

loss���<���i       �	8�v[fc�A�!*

lossJ�;㉎C       �	�,w[fc�A�!*

loss�8=�qr�       �	x[fc�A�!*

loss�eL=��R       �	(�x[fc�A�!*

lossĈ�<�A�       �	�^y[fc�A�!*

loss�,<	M       �	f�y[fc�A�!*

lossGM�<�j�b       �	@�z[fc�A�!*

loss�{�;�z�       �	�H{[fc�A�!*

loss�y�<�J��       �	��{[fc�A�!*

loss~9�;��	       �	�|[fc�A�!*

lossV <�m׀       �	!}[fc�A�!*

loss���;.��C       �	�}[fc�A�!*

lossj��<���       �	�Y~[fc�A�!*

loss4�;ԍ�8       �	^�~[fc�A�!*

lossx�g:p�k7       �	��[fc�A�!*

loss�,<���{       �	�Ȁ[fc�A�!*

loss7a,=M��       �	
e�[fc�A�!*

loss���;�eKn       �	5�[fc�A�!*

loss\��=0�       �	8��[fc�A�!*

lossW�+;�DW_       �	y$�[fc�A�!*

loss}��<�ߧ       �	C�[fc�A�!*

loss;U;{eH       �	c{�[fc�A�!*

lossԖ�:G�f�       �	a�[fc�A�!*

lossۯ�<G_U       �	C��[fc�A�!*

loss�^<?�       �	VE�[fc�A�!*

loss�$={���       �	��[fc�A�!*

loss��;.�v       �	�z�[fc�A�!*

lossH�;���       �	�2�[fc�A�!*

loss`��<�t�       �	+ۉ[fc�A�!*

loss2��<��A�       �	���[fc�A�!*

loss�>�<Ȫ�-       �	@1�[fc�A�!*

loss�+
=}�!�       �	�m�[fc�A�!*

loss�F%<`D-       �	H�[fc�A�!*

loss)�<��	       �	�ō[fc�A�!*

loss��<���       �	k�[fc�A�!*

loss�¼<�Ǘ       �	J�[fc�A�!*

loss�f�<��G�       �	���[fc�A�!*

loss$<X�T       �	L�[fc�A�!*

loss`j=���       �	}�[fc�A�!*

loss��;q{��       �	��[fc�A�!*

lossx�;��8X       �	:�[fc�A�!*

loss7��;D�6�       �	���[fc�A�!*

lossm�;�F.�       �	�O�[fc�A�!*

loss�ic<��       �	$�[fc�A�!*

loss1��<���#       �	��[fc�A�!*

loss���<���t       �	��[fc�A�!*

loss��<���       �	_��[fc�A�!*

lossV"�<{��       �	S\�[fc�A�!*

lossV��;�29�       �	�[fc�A�!*

lossZ�n;l��       �	���[fc�A�!*

loss�(5<�	a�       �	r�[fc�A�!*

loss��@<�q	       �	h��[fc�A�!*

loss��2<˕/�       �	MJ�[fc�A�!*

loss@�>;s��       �	�[fc�A�!*

lossw7�;F���       �	�x�[fc�A�!*

loss;��;��"�       �	�[fc�A�!*

loss&;=A�j       �	"��[fc�A�!*

loss�Pg;�YW�       �	�?�[fc�A�!*

loss(�B<�%׫       �	�_�[fc�A�!*

lossJPH=M���       �	G�[fc�A�!*

loss��<?*�       �	��[fc�A�!*

loss�`�<&�       �	�9�[fc�A�!*

loss8�I=�Ď�       �	Zٟ[fc�A�!*

loss㴳=^=�       �	�q�[fc�A�!*

loss&��<��j�       �	��[fc�A�!*

lossR%<m�H�       �	��[fc�A�!*

loss�zw:�3�d       �	G�[fc�A�!*

loss@x'=ʫ��       �	@ߢ[fc�A�!*

loss[�;N��q       �	�{�[fc�A�!*

loss�};���       �	��[fc�A�!*

loss��<q�҈       �	GȤ[fc�A�!*

loss��;�Vz       �	�p�[fc�A�!*

loss�7�;aR9       �	��[fc�A�!*

loss�f�=�/�,       �	 ��[fc�A�!*

lossi�k<��,       �	�G�[fc�A�!*

loss�i[=u�       �	J�[fc�A�!*

lossF#�<���       �	/��[fc�A�!*

loss��<ƶ7�       �	��[fc�A�!*

loss�h�;ّ�9       �	�A�[fc�A�!*

loss� <�Й       �	u�[fc�A�!*

lossq�n97�D�       �	���[fc�A�!*

loss�;E��       �	v�[fc�A�!*

lossR �<�Z(�       �	���[fc�A�!*

loss�c<�hWd       �	X9�[fc�A�!*

loss�#<}s�y       �	�ˮ[fc�A�!*

loss���;�#y�       �	-_�[fc�A�!*

lossg[=�s�B       �	Z�[fc�A�!*

loss���;a�        �	ƨ�[fc�A�!*

loss��|=��R       �	n�[fc�A�!*

loss��2:d�3�       �	��[fc�A�!*

loss��=`�|�       �	U��[fc�A�!*

loss��<ԡ,�       �	kF�[fc�A�!*

loss�� <jDX       �	#ܳ[fc�A�!*

loss�<_nܘ       �	�q�[fc�A�!*

lossc��<+���       �	��[fc�A�!*

loss��k<$��       �	���[fc�A�!*

loss�<3Wa$       �	G�[fc�A�!*

lossx#C=�ƣ�       �	�߶[fc�A�!*

loss�9<&�M       �	�w�[fc�A�!*

loss�xl=\���       �	�	�[fc�A�!*

loss$��;P���       �	��[fc�A�!*

losssj$<�V��       �	�:�[fc�A�!*

loss_�<C��       �	�й[fc�A�!*

loss̓�;gO�D       �	f�[fc�A�!*

loss�F�<�sr       �	���[fc�A�!*

loss{
�;��V�       �	���[fc�A�!*

loss�=�DN       �	�u�[fc�A�!*

losse��<�슮       �	��[fc�A�!*

loss��R<���       �	$��[fc�A�!*

lossY$=�~x~       �	�]�[fc�A�!*

loss�c6=��       �	���[fc�A�!*

loss`y0;Q��       �	��[fc�A�!*

loss�-B;�;F�       �	0F�[fc�A�!*

loss@��;"�       �	���[fc�A�!*

loss��x<[d%�       �	'��[fc�A�!*

loss��;��7p       �	d�[fc�A�!*

loss-�=�"��       �	��[fc�A�"*

lossM�=���       �	<��[fc�A�"*

loss�,9=D�d       �	S��[fc�A�"*

loss���<C�z       �	�G�[fc�A�"*

loss�Z1;D0�;       �	��[fc�A�"*

lossI`�:��P'       �	8��[fc�A�"*

loss$��<n<D�       �		p�[fc�A�"*

lossN=M:       �	�[fc�A�"*

loss�D�;�i�e       �	*��[fc�A�"*

loss��<���z       �	C<�[fc�A�"*

loss�@�:W��3       �	g��[fc�A�"*

loss�< ���       �	Xp�[fc�A�"*

loss��<�4�`       �	Y�[fc�A�"*

loss�'�<�q�<       �	��[fc�A�"*

loss.�O;I݋�       �	�n�[fc�A�"*

loss
Yi;��~�       �	��[fc�A�"*

loss��<݃��       �	���[fc�A�"*

loss�< <C��       �	hZ�[fc�A�"*

lossd�?<S�       �	'�[fc�A�"*

loss�Ԯ=�@e       �	)��[fc�A�"*

loss�$Z<�
�       �	�w�[fc�A�"*

loss�;+<�ƶ       �	M�[fc�A�"*

loss�b�<�
�h       �	���[fc�A�"*

lossdb�<�JR       �	�d�[fc�A�"*

loss��d=hri       �	�[fc�A�"*

loss�*<���l       �	ʤ�[fc�A�"*

loss٦�<i`O       �	�@�[fc�A�"*

loss0�;�C�       �	��[fc�A�"*

losst�<�8�       �	j��[fc�A�"*

lossX�><B�6�       �	-#�[fc�A�"*

loss{�=�i	�       �	���[fc�A�"*

loss� ==�R�       �	XW�[fc�A�"*

lossҐ`;*���       �	���[fc�A�"*

loss���;�T��       �	���[fc�A�"*

loss��:�"pp       �	�5�[fc�A�"*

loss��<� C�       �	���[fc�A�"*

loss_}�<��2       �	{�[fc�A�"*

loss�U�<O|E       �	��[fc�A�"*

loss�^U;�&B       �	���[fc�A�"*

loss��=���       �	\�[fc�A�"*

lossH�[<�W�<       �	
��[fc�A�"*

loss�<�-��       �	n��[fc�A�"*

loss஗<����       �	�)�[fc�A�"*

loss]�;�E       �	���[fc�A�"*

loss�w�<��       �	7m�[fc�A�"*

loss8��<0�       �	j�[fc�A�"*

lossR=�(       �	��[fc�A�"*

loss��=}��       �	ZJ�[fc�A�"*

loss_�5<��a?       �	6��[fc�A�"*

loss/<�<%�6�       �	�{�[fc�A�"*

loss<��:٬�3       �	�[fc�A�"*

lossq,<*��3       �	:��[fc�A�"*

lossII=�ݯ�       �	�E�[fc�A�"*

loss,J7;`�y�       �	���[fc�A�"*

loss��;C��]       �	6v�[fc�A�"*

loss&ǳ<^� �       �	�[fc�A�"*

lossI��<4�<       �	¢�[fc�A�"*

loss㴓<�c�       �	`;�[fc�A�"*

loss2<<���       �	N��[fc�A�"*

loss=q<��;;       �	�o�[fc�A�"*

loss�=�       �	D��[fc�A�"*

loss��<�C]�       �	&��[fc�A�"*

lossm5<���       �	%��[fc�A�"*

loss��<QT,       �	��[fc�A�"*

loss� [=��U       �	���[fc�A�"*

loss�z�<�u�2       �	5]�[fc�A�"*

loss>HG,�       �	і�[fc�A�"*

losss5�="q>       �	�+�[fc�A�"*

losso�@=�6 �       �	2��[fc�A�"*

loss��<Ά       �	h��[fc�A�"*

lossn�5<�q4       �	9�[fc�A�"*

loss<~<�oִ       �	���[fc�A�"*

loss�-<k9}       �	v��[fc�A�"*

loss�)=��Qn       �	#1�[fc�A�"*

loss�2=M��       �	>��[fc�A�"*

loss�P=����       �	Uj�[fc�A�"*

loss�[<a
��       �	��[fc�A�"*

loss}}�;��y�       �	^��[fc�A�"*

loss?2�=�,��       �	d;�[fc�A�"*

loss<1<�Z��       �	���[fc�A�"*

loss�[�<Ұ�       �	�~�[fc�A�"*

loss؃�<+���       �	;�[fc�A�"*

loss �o=,��       �	���[fc�A�"*

loss���;5�tO       �	܀�[fc�A�"*

lossCFO<R1ĭ       �	W&�[fc�A�"*

loss���<� H       �	���[fc�A�"*

lossߝ�<�s�<       �	�]�[fc�A�"*

loss�'�<�S�U       �	��[fc�A�"*

loss�n;
3�       �	"��[fc�A�"*

loss�.O<�~t�       �	�G�[fc�A�"*

losse��<7�+v       �	���[fc�A�"*

loss�gJ<���       �	�~ \fc�A�"*

loss؀c=�ST       �	#\fc�A�"*

lossR�a=7y��       �	��\fc�A�"*

lossh�=<;E��       �	�d\fc�A�"*

loss�\�;�xx       �	<�\fc�A�"*

lossU�;I��       �	k�\fc�A�"*

lossEM�<�7�       �	�0\fc�A�"*

lossʚ�<U$��       �	X�\fc�A�"*

loss��<(�kA       �	q\fc�A�"*

lossh��;6V�       �	+\fc�A�"*

loss^=2Kk3       �	��\fc�A�"*

loss߸�:�ة       �	6V\fc�A�"*

lossX�M;Ӥo�       �	�\fc�A�"*

loss��<�l��       �	ٰ\fc�A�"*

lossH(=/��       �	vS	\fc�A�"*

loss���<	�n.       �	(�	\fc�A�"*

loss�n�<�Y��       �	r�
\fc�A�"*

loss���<����       �	�$\fc�A�"*

loss���<e!��       �	ܹ\fc�A�"*

loss"��;��(o       �	uV\fc�A�"*

loss�=oգ:       �	\fc�A�"*

lossUc�<�~�       �	��\fc�A�"*

lossn�V=70kD       �	�@\fc�A�"*

loss3��;D@�       �	��\fc�A�"*

loss�s�=���
       �	�\fc�A�"*

loss�!f<�GJ       �	'\fc�A�"*

lossZ�t=��l�       �	��\fc�A�"*

loss1:�<WP9E       �	si\fc�A�"*

loss,A<-�f�       �	H\fc�A�"*

loss�)=1as�       �	��\fc�A�"*

loss��<�;��       �	�l\fc�A�"*

loss��n<��@�       �	<\fc�A�"*

loss�<�媂       �	��\fc�A�"*

lossrQi=ED��       �	d\fc�A�"*

lossŀ<��g       �	&�\fc�A�"*

loss�A ;��	_       �	Q�\fc�A�"*

loss�P�=p��#       �	�>\fc�A�"*

loss��=4��p       �	��\fc�A�#*

lossrcW<���       �	A�\fc�A�#*

loss�C<�N�E       �	�"\fc�A�#*

loss�5<���       �	�\fc�A�#*

loss���<��[       �	�e\fc�A�#*

lossJ<=s�!�       �	��\fc�A�#*

loss�<���       �	�\fc�A�#*

loss��<�2�       �	U.\fc�A�#*

lossk�<�ʕB       �	��\fc�A�#*

losscG�<�b�       �	8h\fc�A�#*

loss�4<ť%       �	��\fc�A�#*

loss�]�;(��I       �	b�\fc�A�#*

loss��e<�A y       �	�H\fc�A�#*

loss�b;���       �	��\fc�A�#*

loss�>�: ��a       �	i� \fc�A�#*

loss`L�<ݲp�       �	�.!\fc�A�#*

loss���<Wx6N       �	��!\fc�A�#*

loss��;>i�       �	�e"\fc�A�#*

lossE�=���       �	#\fc�A�#*

loss�=M�1+       �	��#\fc�A�#*

loss��=��A       �	?;$\fc�A�#*

loss��=�X�G       �	��$\fc�A�#*

losss�b<�]�X       �	�l%\fc�A�#*

lossq�&<R�|�       �	�&\fc�A�#*

lossa+�<�]�       �	~�&\fc�A�#*

loss�5=���       �	�C'\fc�A�#*

loss�h4=`�4=       �	��'\fc�A�#*

loss�yi;A�%e       �	?o(\fc�A�#*

lossn�)=g��       �	)\fc�A�#*

lossp.�<�1'       �	C�)\fc�A�#*

loss/o�;��ʚ       �	��*\fc�A�#*

loss�U�;i���       �	 �+\fc�A�#*

loss�s<�DL�       �	��,\fc�A�#*

lossʰ�<]�P       �	��-\fc�A�#*

lossd��;����       �	L�.\fc�A�#*

lossj_�;U�d       �	��/\fc�A�#*

loss��9<�ٙ       �	1\fc�A�#*

loss3��<$�.       �	��1\fc�A�#*

loss�
=t�m       �	93\fc�A�#*

loss���<o�i�       �	u�3\fc�A�#*

loss���=���<       �	Y�4\fc�A�#*

loss� .<�Q�X       �	6\fc�A�#*

lossͷ�<��O�       �	�6\fc�A�#*

loss/}<��`�       �	M�7\fc�A�#*

loss��;iЁ�       �	9�8\fc�A�#*

lossne:<Du�       �	�9\fc�A�#*

losss��<��!       �	O�:\fc�A�#*

loss+]=
�#       �	si;\fc�A�#*

loss��L<����       �	"7<\fc�A�#*

loss=|:=;w�       �	1�<\fc�A�#*

lossl�(=��S       �	�l=\fc�A�#*

lossB�<��H       �	>\fc�A�#*

loss��u;�e�       �	��>\fc�A�#*

loss��;����       �	sL?\fc�A�#*

loss_��;����       �	;�?\fc�A�#*

loss=G=w�~       �	lz@\fc�A�#*

loss���<Xj��       �	~A\fc�A�#*

loss��<���q       �	�A\fc�A�#*

loss-<�<�*.Z       �	ZKB\fc�A�#*

lossF��<?�4�       �	)�B\fc�A�#*

loss��;�g<�       �	ۇC\fc�A�#*

lossई<C�       �	�D\fc�A�#*

lossD�	;��       �	�LE\fc�A�#*

loss|v�<�gR�       �	�jF\fc�A�#*

loss.q�;%3=       �	�G\fc�A�#*

lossM��=�!�       �	¥G\fc�A�#*

loss�"=Z       �	�DH\fc�A�#*

loss��J<��V       �	?�H\fc�A�#*

loss I=K��       �	�I\fc�A�#*

loss�6<8��^       �	O"J\fc�A�#*

loss�/�<P��       �	ؼJ\fc�A�#*

loss効;N�ܖ       �	@iK\fc�A�#*

lossM��<�SɊ       �	[L\fc�A�#*

loss�T�=�!n�       �	��L\fc�A�#*

loss��<���       �	�>M\fc�A�#*

loss܇N=�        �	��M\fc�A�#*

loss;�;[�ya       �	��N\fc�A�#*

lossl�u<��k       �	(aO\fc�A�#*

lossT��<ܱA�       �	��O\fc�A�#*

loss@��<%;C�       �	��P\fc�A�#*

loss6�A=�y�       �	%;Q\fc�A�#*

loss9��<2�ڙ       �	�Q\fc�A�#*

loss(�<�Q��       �	}R\fc�A�#*

loss���<���       �	�%S\fc�A�#*

loss�Z|=lI<       �	;�S\fc�A�#*

loss,�=7�|       �	bT\fc�A�#*

loss8~=��S�       �	U\fc�A�#*

loss�V4=�`-       �	ܞU\fc�A�#*

loss�!�;��g�       �	�9V\fc�A�#*

loss�q<D3��       �	�V\fc�A�#*

loss�(�;	�$N       �	yvW\fc�A�#*

loss�<V<PWX       �	oGX\fc�A�#*

lossx�(;`8/�       �	��X\fc�A�#*

lossr_<���       �	ۆY\fc�A�#*

loss��2<��X=       �	9%Z\fc�A�#*

loss2=2�a       �	ϺZ\fc�A�#*

lossx\G<�'�t       �	c[\fc�A�#*

loss��<o�       �	6\\fc�A�#*

lossO�<�;R�       �	�\\fc�A�#*

loss�@<����       �	�9]\fc�A�#*

loss_hY<C���       �	�]\fc�A�#*

loss���<>k�r       �	�i^\fc�A�#*

loss)�h<����       �	d_\fc�A�#*

lossA;<&7t       �	�_\fc�A�#*

lossa��;^�c{       �	e�`\fc�A�#*

loss/��:��<       �	�+a\fc�A�#*

loss�.�<JE�       �	��a\fc�A�#*

loss@z�<�f�j       �	�]b\fc�A�#*

lossAb�<$>1X       �	��b\fc�A�#*

loss�%<;�V       �	Օc\fc�A�#*

loss`l�;�v�       �	/5d\fc�A�#*

lossϘ;��.       �	��d\fc�A�#*

lossߗW=�i�       �	��e\fc�A�#*

loss�t;���2       �	75f\fc�A�#*

loss7J<<= �       �	��f\fc�A�#*

loss*}�<�+h�       �	Zhg\fc�A�#*

lossl��<���       �	h\fc�A�#*

loss�<2l8       �	M�h\fc�A�#*

lossdh�<@�|       �	xFi\fc�A�#*

loss��;��       �	��i\fc�A�#*

loss��4<ה       �	��j\fc�A�#*

loss���;t,`       �	]7k\fc�A�#*

loss���<O��       �	{�k\fc�A�#*

lossQ�<:��       �	�wl\fc�A�#*

losstfp<Y�@       �	�m\fc�A�#*

loss]�<y�HU       �	l�m\fc�A�#*

loss�d�;�s$       �	�Mn\fc�A�#*

loss��=��q�       �	��n\fc�A�#*

losst�;��;       �	�~o\fc�A�$*

loss��W<����       �	�p\fc�A�$*

loss��B<�gd^       �	<�p\fc�A�$*

loss���<Xq�c       �	`Vq\fc�A�$*

loss�Y<mH�       �	0�q\fc�A�$*

loss�B;��X�       �	��r\fc�A�$*

loss�{<�U��       �	�&s\fc�A�$*

lossR=���+       �	~9t\fc�A�$*

lossD��<��q�       �	2�t\fc�A�$*

loss��3<�s�\       �	9v\fc�A�$*

loss@��;�E\�       �	N�v\fc�A�$*

loss�m=ךɐ       �	Xpw\fc�A�$*

loss��;�       �	�x\fc�A�$*

loss8�;��}\       �	��x\fc�A�$*

loss��|=f�       �	MNy\fc�A�$*

losso�<I���       �	��y\fc�A�$*

loss��<0f@�       �	�z\fc�A�$*

loss8��=ts�W       �	r6{\fc�A�$*

losso<8�
       �	0�{\fc�A�$*

loss��=O��       �	� }\fc�A�$*

loss�[�<��A       �	��}\fc�A�$*

loss�wL;��       �	L~\fc�A�$*

loss�؍;K�l       �	��~\fc�A�$*

losswur=��#       �	I�\fc�A�$*

lossv�;9['       �	|)�\fc�A�$*

loss��U<�1       �	̀\fc�A�$*

loss�ޣ;%�u�       �	Ts�\fc�A�$*

loss���;ݾ@R       �	j�\fc�A�$*

loss�@p;S�u\       �	b��\fc�A�$*

lossx�V<��|G       �	�b�\fc�A�$*

loss�5<�am�       �	t�\fc�A�$*

loss�;���       �	��\fc�A�$*

lossˇ<}s�       �	W��\fc�A�$*

loss���<�]��       �	HS�\fc�A�$*

loss�_)=����       �	8�\fc�A�$*

loss��<���       �	Ӡ�\fc�A�$*

loss�F=��:       �	�P�\fc�A�$*

loss �;l=��       �	g�\fc�A�$*

loss�<}��<       �	��\fc�A�$*

loss(<=6�K�       �	#1�\fc�A�$*

loss0��<'!A�       �	�Њ\fc�A�$*

loss�]<b��P       �	�n�\fc�A�$*

loss�Os;}�pC       �	��\fc�A�$*

loss���=Ձ��       �	���\fc�A�$*

losse`�;O$tn       �	�X�\fc�A�$*

loss,�"=z1u       �	��\fc�A�$*

lossW��<��       �	-��\fc�A�$*

loss1.Q=��<;       �	�4�\fc�A�$*

loss�=���       �	4׏\fc�A�$*

loss��_=>$Ǖ       �	Pr�\fc�A�$*

loss�<M��       �	h	�\fc�A�$*

loss!AQ<�ۻ5       �	���\fc�A�$*

loss]=Η�       �	/6�\fc�A�$*

loss���;s       �	�͒\fc�A�$*

loss�e<u���       �	�o�\fc�A�$*

loss_h�;}�y�       �	�\fc�A�$*

lossX�=�P�R       �	���\fc�A�$*

loss�D�=e=؇       �	X<�\fc�A�$*

loss�
5<�r	o       �	�ѕ\fc�A�$*

lossx�;�b�       �	{g�\fc�A�$*

lossxՔ=Rs�       �	���\fc�A�$*

loss_QC;P\�]       �	0��\fc�A�$*

loss�[<�x       �	�;�\fc�A�$*

lossEoh=�W�-       �	��\fc�A�$*

lossT�b<��V�       �	���\fc�A�$*

lossR�'=��5V       �	�*�\fc�A�$*

lossO @=%�\       �	7ƚ\fc�A�$*

loss�c�<��P       �	Qj�\fc�A�$*

lossVc�;��       �	��\fc�A�$*

loss�+�<׮�       �	S��\fc�A�$*

loss���;*;��       �	�T�\fc�A�$*

loss��;	        �	\�\fc�A�$*

loss3P=�o"�       �	졞\fc�A�$*

loss�;�=b��d       �	�=�\fc�A�$*

loss�|=pU\�       �	��\fc�A�$*

loss��<P
�       �	�y�\fc�A�$*

loss	��<�B�       �	�\fc�A�$*

losst,�=�V��       �	��\fc�A�$*

loss:G�<�܁P       �	KY�\fc�A�$*

loss�<�aE       �	�\fc�A�$*

lossH+R<�+��       �	��\fc�A�$*

losso�<�Px       �	p(�\fc�A�$*

loss6��<�ʴ�       �	���\fc�A�$*

loss���</��       �	�[�\fc�A�$*

loss/H<����       �	���\fc�A�$*

loss��;x6�       �	���\fc�A�$*

lossɆ;��C�       �	�-�\fc�A�$*

loss�B5<c�Wq       �	iȧ\fc�A�$*

loss2�<F�A�       �	�]�\fc�A�$*

loss�r_<�(�/       �	��\fc�A�$*

loss�ߵ;U���       �	0��\fc�A�$*

loss���<��S       �	 B�\fc�A�$*

loss���<N�       �	1�\fc�A�$*

lossa��< ��       �	y��\fc�A�$*

loss�&B;��OB       �	fL�\fc�A�$*

loss�5<���       �	'�\fc�A�$*

loss�@�;��
       �	׭\fc�A�$*

loss?C@=�-�=       �	�r�\fc�A�$*

lossmV�<OH�       �	��\fc�A�$*

loss�<޷�       �	y��\fc�A�$*

loss<�;e.��       �	�R�\fc�A�$*

lossz��<��D       �	*�\fc�A�$*

lossA�x<P��       �	���\fc�A�$*

loss�v�;��ʓ       �	VH�\fc�A�$*

loss�X�<īj}       �	��\fc�A�$*

loss�z%=�t�j       �	�{�\fc�A�$*

lossxr2=��C       �	M�\fc�A�$*

lossAÚ<_e�       �	窴\fc�A�$*

loss�g/=���       �	�B�\fc�A�$*

loss�K�<��x       �	�׵\fc�A�$*

loss!�<��3       �	Bw�\fc�A�$*

loss8e�:����       �	(�\fc�A�$*

loss-h=�n��       �	���\fc�A�$*

loss���;}��       �	S>�\fc�A�$*

loss�G�<� <C       �	�Ӹ\fc�A�$*

loss��99S<       �	#k�\fc�A�$*

loss}�:���       �	��\fc�A�$*

lossDb�<��˭       �	���\fc�A�$*

lossw-Z;&7�m       �	�8�\fc�A�$*

lossTsE=`z��       �	"�\fc�A�$*

lossI@�<���       �	���\fc�A�$*

loss�O"=���       �	�'�\fc�A�$*

loss�NL<A�B�       �	���\fc�A�$*

loss���<
Vh       �	?V�\fc�A�$*

lossX�<zZ�       �	���\fc�A�$*

loss��<���       �	���\fc�A�$*

lossy��<���9       �	S]�\fc�A�$*

loss��<��V`       �	���\fc�A�$*

loss({�<�ִ�       �	��\fc�A�%*

lossZ�Q<���       �	H��\fc�A�%*

loss�O<��N�       �	�9�\fc�A�%*

loss��=�;G�       �	v��\fc�A�%*

loss��-<�7[       �	x|�\fc�A�%*

loss���;���       �	<�\fc�A�%*

loss-�!8�s�I       �	�Z�\fc�A�%*

loss�|:=k��       �	���\fc�A�%*

loss�<5�S       �	���\fc�A�%*

loss��7<���       �	�e�\fc�A�%*

loss2p7;5`�r       �	J�\fc�A�%*

lossl(1:ߍ-,       �	i��\fc�A�%*

lossf6�;b=�       �	�R�\fc�A�%*

loss�p\<;D��       �	_��\fc�A�%*

loss�P�;��       �	���\fc�A�%*

loss1M*<L�       �	C:�\fc�A�%*

loss�09<"�$�       �	}��\fc�A�%*

loss���;��Q�       �	od�\fc�A�%*

lossg�
<���       �	��\fc�A�%*

lossK^<f�[;       �	.��\fc�A�%*

lossퟒ<�@`(       �	���\fc�A�%*

loss�.=-]�       �	p�\fc�A�%*

loss���;"k3       �	��\fc�A�%*

loss82<��>       �	ˠ�\fc�A�%*

loss,+�;!�)l       �	���\fc�A�%*

loss=�}<۱�       �	Z��\fc�A�%*

loss�-�<a���       �	�;�\fc�A�%*

loss��
;ª��       �	���\fc�A�%*

loss(<h4IJ       �	_}�\fc�A�%*

loss?U>=�:�       �	�.�\fc�A�%*

loss\��<AK�       �	h��\fc�A�%*

loss��<�t��       �	e�\fc�A�%*

loss��B=��       �	H��\fc�A�%*

loss}�<t��       �	��\fc�A�%*

loss;��;$�s       �	�/�\fc�A�%*

loss��<\:�o       �	���\fc�A�%*

loss��<�B{�       �	�n�\fc�A�%*

loss3!�:;�~`       �	�
�\fc�A�%*

loss�hm<i���       �	���\fc�A�%*

loss�}�;�;q�       �	>�\fc�A�%*

lossvp=��1<       �	5��\fc�A�%*

lossR�c<�i�       �	�m�\fc�A�%*

lossS��<+��       �	p�\fc�A�%*

loss��/=����       �	���\fc�A�%*

lossVX<-�oC       �	@�\fc�A�%*

loss��;=�/�       �	���\fc�A�%*

lossÔ;<�XT�       �	�~�\fc�A�%*

loss�m;��JH       �	��\fc�A�%*

loss�B�;$�	       �	���\fc�A�%*

loss��<��@�       �	S�\fc�A�%*

loss��i<=���       �	0��\fc�A�%*

lossa��<��       �	���\fc�A�%*

loss��<v��k       �	�+�\fc�A�%*

loss8�<=��P       �	C��\fc�A�%*

loss_t;5\!       �	�e�\fc�A�%*

loss-w;�Ey       �	h	�\fc�A�%*

loss��<J���       �	*��\fc�A�%*

loss�m%=r�       �	RD�\fc�A�%*

loss7��;_K��       �	��\fc�A�%*

loss�Xi<v�{�       �	���\fc�A�%*

lossj��<�>}�       �	��\fc�A�%*

loss�I�<��_       �	���\fc�A�%*

loss%$�<��j�       �	�P�\fc�A�%*

loss��<��Gq       �	��\fc�A�%*

lossH%�;'�x'       �	���\fc�A�%*

lossK�;7Cv       �	0�]fc�A�%*

loss�q�=���6       �	��]fc�A�%*

loss��V=ϳG7       �	']fc�A�%*

loss���;���       �	��]fc�A�%*

loss�O�<Ze@       �	n]fc�A�%*

loss��<�܊O       �	b]fc�A�%*

loss7�=8�p
       �	�]fc�A�%*

lossL��<�>]       �	:Z]fc�A�%*

loss�zq=��       �	/�]fc�A�%*

loss���;����       �	��]fc�A�%*

lossu�<�)�       �	�8 ]fc�A�%*

loss�B�<�uQ�       �	�� ]fc�A�%*

lossJX=]�r�       �	�!]fc�A�%*

losso/�<��zW       �	�6"]fc�A�%*

lossZa�<�O@0       �	��"]fc�A�%*

loss�	�;L�       �	�q#]fc�A�%*

lossJ�;:�       �	�$]fc�A�%*

loss���;�K�       �	I�$]fc�A�%*

loss��K<.��       �		S%]fc�A�%*

loss�N=��.�       �	��%]fc�A�%*

loss�!=��o�       �	;�&]fc�A�%*

lossf"�<7��       �	J@']fc�A�%*

loss�n�;3Xp       �	�']fc�A�%*

loss�>S=�n_       �	�m(]fc�A�%*

loss�zc<Ɨrv       �	))]fc�A�%*

lossl��;�W�       �	n�)]fc�A�%*

lossFD!<*��       �	C*]fc�A�%*

loss>p=kV�b       �	�.+]fc�A�%*

loss��<�Z�       �	
�+]fc�A�%*

loss�3H=�8�       �	�,]fc�A�%*

loss/#�;��(       �	*�-]fc�A�%*

loss�g�;��dg       �	
j.]fc�A�%*

loss�=;��r       �	�Q/]fc�A�%*

loss�D�<�r�       �	�0]fc�A�%*

losszw;JJg3       �	��0]fc�A�%*

loss q=�޶�       �	��1]fc�A�%*

loss߆�;�)K�       �	͕2]fc�A�%*

loss��<5���       �	��3]fc�A�%*

loss�ީ=�H˶       �	�4]fc�A�%*

loss�ϵ<��O       �	�5]fc�A�%*

loss��<}�C       �	��5]fc�A�%*

loss�h=o	,�       �	�6]fc�A�%*

loss�+1;�u�D       �	Q07]fc�A�%*

loss���<�p��       �	��7]fc�A�%*

loss�$�;����       �	-x8]fc�A�%*

loss���<����       �	9]fc�A�%*

lossѢ=p���       �	)�9]fc�A�%*

loss�n<�!       �	�^:]fc�A�%*

losseJ�;b"       �	4�;]fc�A�%*

loss?aQ:�n��       �	'<]fc�A�%*

loss���;Ɩ:&       �	��<]fc�A�%*

loss)|=�M�D       �	��=]fc�A�%*

loss�t;�o�       �	�)>]fc�A�%*

loss?k=��       �	�?]fc�A�%*

loss ��;h�I       �	g�?]fc�A�%*

loss?3�;��H       �	W\@]fc�A�%*

lossA��:jp�#       �	A]fc�A�%*

loss�Dz;����       �	�A]fc�A�%*

loss/3@<�h��       �	�B]fc�A�%*

loss�x=�u       �	�0C]fc�A�%*

lossx=:w�I       �	s�C]fc�A�%*

loss���<��N�       �	9|D]fc�A�%*

loss�g;�"*�       �	x'E]fc�A�%*

lossg�
<�qej       �	��E]fc�A�&*

loss��<���       �	yF]fc�A�&*

loss%�2;yj�J       �	+G]fc�A�&*

loss�QW=R�S       �	��G]fc�A�&*

loss���;�(�m       �	taH]fc�A�&*

lossO��<`��       �	�I]fc�A�&*

loss�3=0�<E       �	�I]fc�A�&*

loss;��<?���       �	%;J]fc�A�&*

loss�E0<(,(       �	��J]fc�A�&*

loss�g�;       �	muK]fc�A�&*

lossz��<����       �	aL]fc�A�&*

lossŮ;���       �	�L]fc�A�&*

lossٞ<'Mh�       �	)^M]fc�A�&*

loss;�;
&�       �	v�M]fc�A�&*

loss���;�       �	��N]fc�A�&*

loss��a<����       �	jlO]fc�A�&*

loss]�=�Cێ       �	�P]fc�A�&*

loss@qf;�P��       �	.�P]fc�A�&*

loss?u�=���       �	�MQ]fc�A�&*

loss��<��~       �	��Q]fc�A�&*

lossQ��;�Jc�       �	��R]fc�A�&*

loss�A;Go�J       �	�BS]fc�A�&*

loss��X<$U�       �	&�S]fc�A�&*

lossOИ;�\�       �	��T]fc�A�&*

loss."<!u�\       �	l$U]fc�A�&*

loss��<���       �	��U]fc�A�&*

loss�<+'e       �	�YV]fc�A�&*

lossL{<Tx       �	��V]fc�A�&*

loss�?�<���       �	W�W]fc�A�&*

loss��;�Ҙ       �	j0X]fc�A�&*

lossm�q;���T       �	��X]fc�A�&*

loss�a=��7�       �	J�Y]fc�A�&*

loss$|n<��/       �	G�Z]fc�A�&*

loss�=}�9U       �	�,[]fc�A�&*

loss�/<c;�"       �	��[]fc�A�&*

lossi��=9��       �	Rc\]fc�A�&*

loss�hZ;x�*       �	��\]fc�A�&*

loss��;���       �	��]]fc�A�&*

loss�:�OǸ       �	w.^]fc�A�&*

loss,Y<�1;       �	"�^]fc�A�&*

losssT�<�e@       �	lZ_]fc�A�&*

loss�m<d�Z�       �	��_]fc�A�&*

loss�-:<��n
       �	��`]fc�A�&*

loss� <߿��       �	J$a]fc�A�&*

loss��~;'w        �	'�a]fc�A�&*

loss�=�G�       �	Rb]fc�A�&*

loss��<�b3       �	�b]fc�A�&*

loss��=l0�9       �	��c]fc�A�&*

loss���<{w�       �	�d]fc�A�&*

loss��<����       �	%�d]fc�A�&*

loss�#�<�ݮe       �	�Ge]fc�A�&*

loss�9�<x��T       �	 �e]fc�A�&*

loss.��:���       �	�{f]fc�A�&*

loss��=<F��:       �	� g]fc�A�&*

loss�33<t�A�       �	_�g]fc�A�&*

lossĠ�<��\       �	ih]fc�A�&*

lossK7<=�v       �	�i]fc�A�&*

loss���;�;��       �	йi]fc�A�&*

loss��<E(Ձ       �	�Tj]fc�A�&*

loss ��;WK�`       �	S�j]fc�A�&*

loss���;�Y       �	��k]fc�A�&*

loss,��:��/       �	�-l]fc�A�&*

loss�#�<�aF       �	��l]fc�A�&*

loss�&�<gk�u       �	Ram]fc�A�&*

loss�%;Q)��       �	�Hn]fc�A�&*

loss�T;Ol
�       �	��n]fc�A�&*

loss�s�:�4"�       �	�q]fc�A�&*

loss1B|<�p�       �	/�q]fc�A�&*

loss��<ëo       �	jr]fc�A�&*

loss�ǐ<uK�       �	�s]fc�A�&*

loss�,i;¼!W       �	]�s]fc�A�&*

loss�(T=+o�       �	�Gt]fc�A�&*

lossZ�<��lA       �	v�t]fc�A�&*

loss�H�<t�2@       �	Q�u]fc�A�&*

loss���:�oW       �	F%v]fc�A�&*

loss&b�<�#�       �	�v]fc�A�&*

lossd*�<��Z-       �	�dw]fc�A�&*

loss��;E�C�       �	�
x]fc�A�&*

lossʃ�;DH�&       �	?�x]fc�A�&*

loss֤�<��H+       �	mSy]fc�A�&*

loss��@=BFb       �	��y]fc�A�&*

loss�(=�y��       �	��z]fc�A�&*

loss�ͯ<��V�       �	�/{]fc�A�&*

lossˡ;/�Y       �	��{]fc�A�&*

loss y<;N�)5       �	�s|]fc�A�&*

loss\4a:�Na       �	�(}]fc�A�&*

lossC3�<=�       �	�}]fc�A�&*

loss�E;�owC       �	�f~]fc�A�&*

loss�=�*>       �	� ]fc�A�&*

loss���;��:�       �	H4�]fc�A�&*

lossM	<�l       �	�π]fc�A�&*

loss&%�<6�K�       �	@l�]fc�A�&*

loss��:���R       �	y�]fc�A�&*

lossj�;J��       �	���]fc�A�&*

lossa��=�r       �	�R�]fc�A�&*

loss�
�=�`X�       �	���]fc�A�&*

loss���;!;       �	\��]fc�A�&*

loss���<{ Q�       �	i7�]fc�A�&*

losss�<�wq2       �	~�]fc�A�&*

loss��<xb[       �	���]fc�A�&*

loss@Z�;�*�       �	�4�]fc�A�&*

loss�^�<k��       �	�ڇ]fc�A�&*

loss�ȩ;����       �	�{�]fc�A�&*

lossH�@:���B       �	�&�]fc�A�&*

loss*��<���U       �	̉]fc�A�&*

loss�<L���       �	�q�]fc�A�&*

loss��?;^��       �	��]fc�A�&*

lossJ�=p��X       �	���]fc�A�&*

loss6&<޸!w       �	�T�]fc�A�&*

lossO�;��R       �	$��]fc�A�&*

loss�-<NY�       �	���]fc�A�&*

loss3<�<�*�       �	�&�]fc�A�&*

loss��=2��0       �	�]fc�A�&*

loss��-<q�        �	-^�]fc�A�&*

lossX<<�c��       �	���]fc�A�&*

loss
�N<��~�       �	���]fc�A�&*

loss�>K;`�X�       �	�*�]fc�A�&*

loss��=;Z�0       �	I��]fc�A�&*

lossZ@=7��$       �	�W�]fc�A�&*

loss�:9;��83       �	��]fc�A�&*

loss&%=��Y�       �	2��]fc�A�&*

lossD�<P�       �	1'�]fc�A�&*

lossʅ:��m7       �	���]fc�A�&*

loss���<���       �	�Q�]fc�A�&*

loss�<	�-�       �	��]fc�A�&*

loss�?<'F        �	[}�]fc�A�&*

lossƦ�<��       �	��]fc�A�&*

loss$��<��@       �	���]fc�A�&*

loss$��;���       �	:W�]fc�A�'*

loss(�: ��       �	��]fc�A�'*

loss<D���       �	
��]fc�A�'*

lossFM;Ɣ�       �	R(�]fc�A�'*

loss ¢;Nj       �	�՚]fc�A�'*

loss?�{<|��       �	;��]fc�A�'*

lossR��<.���       �	�9�]fc�A�'*

loss�q�<��J�       �	��]fc�A�'*

loss�a<@�B�       �	e��]fc�A�'*

loss�i�<���       �	�X�]fc�A�'*

loss^^!;1Q7�       �	��]fc�A�'*

loss�Ȟ;gw�b       �	dϟ]fc�A�'*

loss�u�;����       �	腠]fc�A�'*

lossvȲ<>�`       �	><�]fc�A�'*

loss?�'<�0�       �	g�]fc�A�'*

lossE�<2k�5       �	���]fc�A�'*

lossR�K=X�v]       �	Z�]fc�A�'*

loss�ʇ=B-pG       �	��]fc�A�'*

lossqB<�	�       �	໤]fc�A�'*

lossį=�O�       �	d�]fc�A�'*

loss\�U<���       �	x�]fc�A�'*

lossm$;4���       �	̸�]fc�A�'*

lossO�;���       �	�[�]fc�A�'*

loss<W�<֛�-       �	P�]fc�A�'*

loss;�J=6L�       �	�Ϩ]fc�A�'*

loss;��<�yw�       �	�s�]fc�A�'*

loss���<��x2       �		�]fc�A�'*

lossQ0�<��+8       �	V��]fc�A�'*

lossa#=.{�       �	�Z�]fc�A�'*

loss�i;U	��       �	�0�]fc�A�'*

loss�j<�	       �	�ά]fc�A�'*

loss�Y�<��Wg       �	���]fc�A�'*

loss���<�[�2       �	�]fc�A�'*

loss���<��V       �	1��]fc�A�'*

loss��p<h3�\       �	/ð]fc�A�'*

loss@�<:�y       �	���]fc�A�'*

loss�ڰ<J��v       �	�1�]fc�A�'*

loss�8<�$4V       �	! �]fc�A�'*

lossݭ<Zq]       �	-̳]fc�A�'*

loss5Ҁ;Vg{       �	�p�]fc�A�'*

loss��:07oR       �	�f�]fc�A�'*

loss��=�L�X       �	jN�]fc�A�'*

lossρ�<\��       �	(+�]fc�A�'*

lossr��<�<�       �	:�]fc�A�'*

lossh1=��*       �	5��]fc�A�'*

loss���;��R       �	���]fc�A�'*

lossq8[<#r9�       �	�"�]fc�A�'*

lossJS=e�       �	���]fc�A�'*

loss�f�;Ȩ7       �	�f�]fc�A�'*

loss26�;^k%e       �	�0�]fc�A�'*

loss28<\q�u       �	U�]fc�A�'*

loss�eC<�x�G       �	���]fc�A�'*

lossl$=��       �	��]fc�A�'*

loss;t�<��       �	��]fc�A�'*

loss���<�)��       �	`��]fc�A�'*

loss�p�;E�<q       �	gE�]fc�A�'*

lossʋ=��2�       �	���]fc�A�'*

loss��==`�^�       �	l��]fc�A�'*

loss �D<�Dh�       �	zp�]fc�A�'*

loss(܃<�       �	\=�]fc�A�'*

loss�=�Gk�       �	�q�]fc�A�'*

loss�?�<���       �	�_�]fc�A�'*

loss=��;�       �	���]fc�A�'*

loss�	�<��       �	��]fc�A�'*

loss��;nd�       �	���]fc�A�'*

lossE��<�O��       �	h>�]fc�A�'*

loss�\<+jM       �	Vd�]fc�A�'*

loss���=��vX       �	k+�]fc�A�'*

loss��;��1       �	���]fc�A�'*

loss;�;iN�H       �	�t�]fc�A�'*

lossċi<���q       �	��]fc�A�'*

loss��B<W���       �	,��]fc�A�'*

lossv�<qx1�       �	�T�]fc�A�'*

loss#Ŋ<���u       �	f��]fc�A�'*

loss�I,;e�2�       �	���]fc�A�'*

lossZ�H<F�X;       �	�A�]fc�A�'*

loss�M=(�XK       �	���]fc�A�'*

loss�'O<�       �	���]fc�A�'*

lossH�;��h�       �	*�]fc�A�'*

loss�]<%��       �	���]fc�A�'*

loss�[ <kq       �	k�]fc�A�'*

loss�=�>       �	��]fc�A�'*

loss��<��       �	z��]fc�A�'*

loss���=v�"B       �	�F�]fc�A�'*

lossqle=�w�       �	;��]fc�A�'*

loss��;�f<       �	f��]fc�A�'*

loss}�$;��I�       �	!�]fc�A�'*

loss[~<���	       �	���]fc�A�'*

loss��%=��zl       �	rS�]fc�A�'*

loss�|0<�-��       �	F��]fc�A�'*

loss�i�<����       �	7��]fc�A�'*

loss�1�:��)@       �	O#�]fc�A�'*

loss?�v<<:��       �	���]fc�A�'*

lossR>�<�щ       �	�m�]fc�A�'*

loss�'<V�.�       �	��]fc�A�'*

loss�<���       �	���]fc�A�'*

loss���=��+       �	�B�]fc�A�'*

loss�)�<�#4n       �	���]fc�A�'*

loss�CT;j4�       �	ur�]fc�A�'*

loss�;� �       �	��]fc�A�'*

loss=d�<�<��       �	4��]fc�A�'*

lossM��:g�       �	j2�]fc�A�'*

loss�*;H4n�       �	���]fc�A�'*

lossf��;�33�       �	�a�]fc�A�'*

loss�=����       �	P��]fc�A�'*

losss��;��~       �	���]fc�A�'*

lossw#�<�C��       �	.�]fc�A�'*

loss1��<��~�       �	���]fc�A�'*

loss�x�<Z1 �       �	�i�]fc�A�'*

loss��0<�7Q       �	e��]fc�A�'*

lossX}�;b	�"       �	K��]fc�A�'*

loss�s�;�+ɶ       �	�%�]fc�A�'*

loss�Qb=!3       �	o��]fc�A�'*

loss%@�;Z��5       �	�T�]fc�A�'*

lossO�z=��rH       �	K��]fc�A�'*

lossŤ�;�;��       �	N��]fc�A�'*

loss�<��D       �	��]fc�A�'*

lossos�;kF
       �	��]fc�A�'*

loss���;���       �	�^�]fc�A�'*

loss�\�<�KZ�       �	���]fc�A�'*

loss��1<1�ӊ       �	A�]fc�A�'*

loss
K�<-��O       �	Gv�]fc�A�'*

lossY<���       �	�]fc�A�'*

loss̲�<�S0       �	C�]fc�A�'*

lossq�=Jױ*       �	]��]fc�A�'*

loss.��;%�bs       �	l[�]fc�A�'*

lossC�g<��v�       �	=��]fc�A�'*

loss�t�<*�ɾ       �	���]fc�A�'*

loss�=l�        �	K"�]fc�A�(*

loss;�(<��o       �	��]fc�A�(*

loss��~<�Π�       �	�p�]fc�A�(*

lossV$�<o�       �	5�]fc�A�(*

lossO�:
 K�       �	��]fc�A�(*

loss��*<����       �	�4�]fc�A�(*

loss/�,=�1	       �	a��]fc�A�(*

loss��m<��       �	��]fc�A�(*

lossR�k<Q��       �	+�]fc�A�(*

lossSN<	�|�       �	��]fc�A�(*

lossV��<����       �	�d�]fc�A�(*

loss�y�<C�	�       �	 �]fc�A�(*

loss/}5;����       �	}��]fc�A�(*

loss��"<A�#�       �	�3�]fc�A�(*

lossO�O<Λ�q       �	%��]fc�A�(*

lossI\=��[4       �	jg�]fc�A�(*

lossh$�;j���       �	���]fc�A�(*

lossa)�<6�(m       �	d��]fc�A�(*

loss�"�;\0�C       �	M/�]fc�A�(*

loss�=X��       �	���]fc�A�(*

loss��V<�,�       �	�h�]fc�A�(*

loss�˭:�X��       �	?��]fc�A�(*

loss�Ġ;y��-       �	֪�]fc�A�(*

loss\|�;S��       �	iS ^fc�A�(*

loss�9H;�P�T       �	@� ^fc�A�(*

loss���=rv�l       �	�b^fc�A�(*

loss��4=++@-       �	D�^fc�A�(*

loss��E<S���       �	�^fc�A�(*

loss��=��       �	l=^fc�A�(*

loss$��<�=�1       �	�^fc�A�(*

lossӅ<(��       �	!x^fc�A�(*

loss{i�:<߯�       �	�^fc�A�(*

loss��<�0�       �	|�^fc�A�(*

loss��!<�5�       �	�R^fc�A�(*

loss=/M<@7�?       �	d�^fc�A�(*

loss$B<��/e       �	��^fc�A�(*

loss��<��1�       �	C	^fc�A�(*

loss.v<��)�       �	�	^fc�A�(*

lossq4;܄c�       �	��
^fc�A�(*

loss*c�<	�F       �	�^fc�A�(*

loss3�b<jST       �	EG^fc�A�(*

loss6C�;����       �	]�^fc�A�(*

lossnX�<���       �	��^fc�A�(*

loss�ѩ<Ð�       �	�%^fc�A�(*

loss���<��       �		5^fc�A�(*

loss3[c<����       �	G�^fc�A�(*

loss���<1�s�       �	�^fc�A�(*

lossv�<[<��       �	�%^fc�A�(*

lossۥ�;끫�       �	��^fc�A�(*

loss��=5�;       �	�^^fc�A�(*

loss=0C<O{�       �	^fc�A�(*

loss���<?'�       �	(^fc�A�(*

loss�2<��6�       �	%�^fc�A�(*

loss���<�H@       �	UN^fc�A�(*

loss��<�I2       �	��^fc�A�(*

loss|��<�N�       �	ʍ^fc�A�(*

lossS<I<�a��       �	w.^fc�A�(*

lossC��<���w       �	��^fc�A�(*

lossW��<j�2�       �	s^fc�A�(*

loss���<�7��       �	�^fc�A�(*

loss?$�<���       �	-`^fc�A�(*

loss���;U��       �	&�^fc�A�(*

loss��><�"�>       �	��^fc�A�(*

loss��<R��        �	kF^fc�A�(*

loss�QR;��Q       �	z�^fc�A�(*

loss ��:y;�t       �	s�^fc�A�(*

loss[@�<F[_       �	�^fc�A�(*

lossvȩ<F��H       �	�^fc�A�(*

loss�r<�	�e       �	�a^fc�A�(*

lossɼ:��       �	* ^fc�A�(*

loss��;���       �	�� ^fc�A�(*

loss�h�;磌�       �	F!^fc�A�(*

loss���;�`��       �	��!^fc�A�(*

loss,fC= >/       �	��"^fc�A�(*

lossLUJ<h�2�       �	/#^fc�A�(*

lossTd<n��       �	O�#^fc�A�(*

loss.�<�4!�       �	3N$^fc�A�(*

loss�;�\bD       �	�$^fc�A�(*

loss1��;�V|(       �	��%^fc�A�(*

loss?h�<�;��       �	cC&^fc�A�(*

loss�(�=�U       �	�/'^fc�A�(*

loss��;�I�       �	��'^fc�A�(*

lossirE=3�c�       �	�r(^fc�A�(*

loss.=�|q8       �	�)^fc�A�(*

loss�S=��3       �	D�)^fc�A�(*

loss��?=��       �	�^*^fc�A�(*

loss�A�<�E%�       �	�)+^fc�A�(*

loss��=�	)       �	��+^fc�A�(*

loss�0T<�/g�       �	�r,^fc�A�(*

loss]��;�&�v       �	v-^fc�A�(*

loss�{<C�       �	�/^fc�A�(*

loss�=7=�H�       �	��/^fc�A�(*

loss}c�<�E�       �	�j0^fc�A�(*

loss��=�ޝ       �	)#1^fc�A�(*

loss�l=BY         �	�2^fc�A�(*

loss�=�9       �	�&3^fc�A�(*

loss�T�<~��       �	��3^fc�A�(*

loss�6�:_^�       �	��4^fc�A�(*

loss��<Ǥ�       �	R�5^fc�A�(*

loss-=�0�       �	5C6^fc�A�(*

loss��a=���       �	ς7^fc�A�(*

lossrG�; Y83       �	v�8^fc�A�(*

loss�=wu:�       �	֎9^fc�A�(*

loss�;���       �	b1:^fc�A�(*

loss��[=;��       �	!>;^fc�A�(*

loss�=�3O�       �	��;^fc�A�(*

lossR��=��z�       �	`�<^fc�A�(*

loss���<=��)       �	�C=^fc�A�(*

loss�� <Fb��       �	\�=^fc�A�(*

loss
l�<AȞ?       �	�>^fc�A�(*

loss�w,;�.
�       �	<-?^fc�A�(*

lossĴ�=�m��       �	:�?^fc�A�(*

lossX<�t�       �	x@^fc�A�(*

loss?*/;����       �	 A^fc�A�(*

loss�û;W�ë       �	��A^fc�A�(*

loss
=_<��7       �	fB^fc�A�(*

loss��<���T       �	�C^fc�A�(*

loss/��<��9�       �	��C^fc�A�(*

loss1N�<���       �	VdD^fc�A�(*

loss�A�;�^~�       �	&E^fc�A�(*

loss�|N=���       �	��E^fc�A�(*

loss��`<���       �	a8F^fc�A�(*

loss��N=e�<�       �	v�F^fc�A�(*

loss:��<{_^N       �	G^fc�A�(*

loss�z�<��@       �	�!H^fc�A�(*

losswU�:*CZ�       �	��H^fc�A�(*

loss��=���       �	kfI^fc�A�(*

loss���<۾��       �	�J^fc�A�(*

loss!9<��;       �	�J^fc�A�)*

lossv�<D�w:       �	�EK^fc�A�)*

lossc��<{�       �	��K^fc�A�)*

loss?�:=Adj�       �	E�L^fc�A�)*

loss��;Z�       �	GM^fc�A�)*

lossӫ�;͉�%       �	��M^fc�A�)*

loss�Ge=����       �	]�N^fc�A�)*

loss���<��g       �	Y0O^fc�A�)*

loss�k=;��       �	�O^fc�A�)*

loss8U�<B��       �	�P^fc�A�)*

loss/�;F��{       �	�3Q^fc�A�)*

loss��f<�	s       �	��Q^fc�A�)*

loss�%�<����       �	\rR^fc�A�)*

loss�։<�}��       �	�S^fc�A�)*

loss�@�:��7       �	�S^fc�A�)*

loss��;�bS       �	��T^fc�A�)*

losse�<�3        �	ӇU^fc�A�)*

loss�=H�W       �	s-V^fc�A�)*

loss��?<��ɥ       �	��V^fc�A�)*

lossD6m<@Q?d       �	�lW^fc�A�)*

loss)�b<5-�S       �	�X^fc�A�)*

loss}�U;%��I       �	̷X^fc�A�)*

losstM�;��-       �	K[Y^fc�A�)*

loss��<|�9�       �	�Z^fc�A�)*

loss�<�<B�Eh       �	ߣZ^fc�A�)*

loss�/�;��l�       �	G[^fc�A�)*

loss��[=�R�o       �	��[^fc�A�)*

loss\�=�%�       �	��\^fc�A�)*

loss�S$;���i       �	�!]^fc�A�)*

loss�&�:&E"       �	�]^fc�A�)*

loss���<^	��       �	hX^^fc�A�)*

loss���<m�b       �	 _^fc�A�)*

loss>;<���       �	/�_^fc�A�)*

loss�=�?��       �	�<`^fc�A�)*

lossͷ[=�2��       �	�*a^fc�A�)*

lossd�<��B       �	�a^fc�A�)*

lossq�T:>F|�       �	�db^fc�A�)*

lossa��<��}       �	qc^fc�A�)*

lossv2e< Y�       �	�c^fc�A�)*

loss���:z�["       �	0Hd^fc�A�)*

loss�=|���       �	��d^fc�A�)*

lossR�};���       �	8�e^fc�A�)*

loss�<��bs       �	�!f^fc�A�)*

loss
3	=V�@       �	ϻf^fc�A�)*

loss2c<���U       �	�Vg^fc�A�)*

loss�X�;��H        �	��g^fc�A�)*

loss��<e��"       �	��h^fc�A�)*

loss`W\<�|�       �	�'i^fc�A�)*

loss{�L=�^��       �	F�j^fc�A�)*

loss�#�<��=       �	'�k^fc�A�)*

loss�;�v�       �	2!l^fc�A�)*

loss~;�<c��       �	
em^fc�A�)*

lossq�+=�{b2       �	��m^fc�A�)*

loss�o =s	.       �	WCo^fc�A�)*

loss��;��~       �	�fp^fc�A�)*

lossD�:Z&7       �	GXq^fc�A�)*

loss���;LwT       �	>{r^fc�A�)*

lossz�$=\��       �	ڪs^fc�A�)*

loss<z�<p�9�       �	�Mt^fc�A�)*

loss��;��m�       �	4�t^fc�A�)*

loss��];��b�       �	�v^fc�A�)*

loss�Kd<L���       �	i�v^fc�A�)*

loss7M<Sc��       �	��w^fc�A�)*

loss	9�;~���       �	�Jx^fc�A�)*

loss�	�<O���       �	�	y^fc�A�)*

loss�2<iz�       �	��y^fc�A�)*

loss�7=�z       �	�Dz^fc�A�)*

losscM�<�Ac{       �	�z^fc�A�)*

loss`I�;��`�       �	u|^fc�A�)*

lossΥ�<�W.Y       �	��|^fc�A�)*

loss���;�=7       �	��}^fc�A�)*

lossMw	=�^�       �	�.~^fc�A�)*

loss@*=վ��       �	k�^fc�A�)*

lossט<�r�       �	S��^fc�A�)*

lossi��<�N��       �	2ȁ^fc�A�)*

loss|��9n=��       �	�Ղ^fc�A�)*

lossVٻ;Q�        �	n��^fc�A�)*

loss|h�<ï�       �	 s�^fc�A�)*

lossHV<�E�       �	�
�^fc�A�)*

losst�<�)Q       �	n��^fc�A�)*

loss�=�;���V       �	�^fc�A�)*

loss3z=�nF
       �	��^fc�A�)*

loss���;�S�3       �	}@�^fc�A�)*

losst5�:�D��       �	i�^fc�A�)*

loss&V�<�w��       �	�^fc�A�)*

loss(��:��       �	L��^fc�A�)*

loss�=�84x�       �	in�^fc�A�)*

loss���:;���       �	��^fc�A�)*

loss�9�;Q�       �	���^fc�A�)*

loss�t;��~n       �	PQ�^fc�A�)*

loss��;�*y�       �	>�^fc�A�)*

loss��9޾��       �	`��^fc�A�)*

loss_�; ��5       �	)�^fc�A�)*

loss�G�;��ޢ       �	Ɛ^fc�A�)*

loss%k�8h�P'       �	�i�^fc�A�)*

loss_�79�U�7       �	�^fc�A�)*

loss�-:�Y       �		��^fc�A�)*

lossX��<���       �	�E�^fc�A�)*

loss�L;�
�       �	�ܓ^fc�A�)*

loss�{�8PT�t       �	s�^fc�A�)*

loss�<+;�f       �	6 �^fc�A�)*

lossQ-=Ƚ��       �	���^fc�A�)*

loss���:��7�       �	Fa�^fc�A�)*

loss�#A<L���       �	h�^fc�A�)*

loss��<t©�       �	ղ�^fc�A�)*

loss֐�;���       �	)\�^fc�A�)*

loss*k�<�C       �	���^fc�A�)*

loss�.�;#@�N       �	-��^fc�A�)*

loss�h<���z       �	�/�^fc�A�)*

losstS�;)`ҋ       �	_Κ^fc�A�)*

lossv�<y�w�       �	�h�^fc�A�)*

lossʙ�<平�       �	��^fc�A�)*

lossD�)<�؆       �	-��^fc�A�)*

loss�-<#���       �	�a�^fc�A�)*

loss�˥<m��\       �	���^fc�A�)*

loss�6?=���       �	���^fc�A�)*

lossh=�<�v�       �	M-�^fc�A�)*

lossu<��Q       �	6ʟ^fc�A�)*

lossvV�<���+       �	4e�^fc�A�)*

lossɩk<�%(1       �	���^fc�A�)*

loss�-�<�       �	��^fc�A�)*

lossI��<�w�`       �	P9�^fc�A�)*

loss7�;��z       �	��^fc�A�)*

lossWT�<	��-       �	��^fc�A�)*

lossR9�;�0k�       �	�$�^fc�A�)*

lossX�k<)�F       �	^ؤ^fc�A�)*

loss�o:e�w�       �	Z�^fc�A�)*

losso+�:P���       �	'�^fc�A�)*

loss���<3�\       �	�Ц^fc�A�**

loss�1�;O�^|       �	�}�^fc�A�**

loss��'=a�H�       �	9'�^fc�A�**

losst��=���       �	���^fc�A�**

loss$F;�/O�       �	h]�^fc�A�**

loss)؈;	�       �	��^fc�A�**

lossq�=���h       �	'��^fc�A�**

loss���;��
       �	@�^fc�A�**

loss�<� Zx       �	��^fc�A�**

loss85�:��o�       �	 ��^fc�A�**

lossT�<-�S-       �	�5�^fc�A�**

loss7M<\%)i       �	@/�^fc�A�**

loss��^=�Z��       �	�#�^fc�A�**

loss�=�:Zfk%       �	�ͯ^fc�A�**

loss�;���S       �	[y�^fc�A�**

losso�;�A�       �	�c�^fc�A�**

lossh�=�?>       �	��^fc�A�**

loss�o<0���       �	[{�^fc�A�**

loss��x;>q��       �	���^fc�A�**

lossRa�=.V�       �	�>�^fc�A�**

loss��<l)w3       �	A(�^fc�A�**

lossT�=�x��       �	�q�^fc�A�**

loss�~=�!U       �	�H�^fc�A�**

loss��:���       �	�[�^fc�A�**

loss��;%�ui       �	ޒ�^fc�A�**

loss֐�<�٬�       �	^/�^fc�A�**

loss ?�<����       �	���^fc�A�**

loss�iR=?i/�       �	Dj�^fc�A�**

loss��<s�eB       �	?��^fc�A�**

loss�`=�g       �	���^fc�A�**

loss;��;��@t       �	]1�^fc�A�**

loss��=���       �	���^fc�A�**

loss{��<)�D}       �	b�^fc�A�**

loss-}�<Ъ��       �	b��^fc�A�**

loss�h<���       �	���^fc�A�**

lossC��;���       �	�&�^fc�A�**

loss�z>=N�t       �	���^fc�A�**

loss�.�<e�6�       �	\�^fc�A�**

loss�'<<��       �	D��^fc�A�**

loss���<{.��       �	��^fc�A�**

loss�$A=́��       �	�-�^fc�A�**

loss9�;^�.�       �	�j�^fc�A�**

loss�w\<[J,       �	m�^fc�A�**

lossë|<�['       �	S��^fc�A�**

loss�/�<�uh       �	�(�^fc�A�**

lossz%:</O*       �	2��^fc�A�**

lossl1i<P]p�       �	�\�^fc�A�**

loss�7;�J�@       �	���^fc�A�**

loss�l�<귎�       �	Ҋ�^fc�A�**

loss�(<�M��       �	O$�^fc�A�**

lossIZ@<��Р       �	���^fc�A�**

loss�@�<�C�       �	[`�^fc�A�**

loss�t�<<j5�       �	���^fc�A�**

loss�to<����       �	��^fc�A�**

loss�%;�hH       �	~5�^fc�A�**

loss�=��u       �	JC�^fc�A�**

loss���;�V�e       �	���^fc�A�**

loss8�T;��       �	 ~�^fc�A�**

lossBW=�8��       �	��^fc�A�**

loss ��:�=�       �	ͯ�^fc�A�**

lossr<=�=�c       �	�V�^fc�A�**

lossϵ;�[mH       �	��^fc�A�**

loss\6<k?>F       �	��^fc�A�**

loss(��<FѸ       �	�E�^fc�A�**

lossAL=D�:�       �	���^fc�A�**

loss.t<�jB�       �	���^fc�A�**

loss\N;=���       �	�&�^fc�A�**

loss�%S;;7�0       �		��^fc�A�**

loss��<5�ʊ       �	G��^fc�A�**

lossj��<�ܰt       �	�_�^fc�A�**

lossT�6<����       �	. �^fc�A�**

loss�`E<J3�       �	���^fc�A�**

loss�� =c}	       �	O�^fc�A�**

loss��<'\��       �	��^fc�A�**

loss�iZ:��\       �	΍�^fc�A�**

lossr�E<���       �	8�^fc�A�**

loss;p�<&l       �	���^fc�A�**

loss�6;�_�       �	�o�^fc�A�**

lossC�_=�`��       �	��^fc�A�**

loss�QA<��t@       �	��^fc�A�**

lossι;�       �	=C�^fc�A�**

lossIr�:�Ў^       �	���^fc�A�**

loss�T�:�C�       �	���^fc�A�**

loss%Vf<*<��       �	�{�^fc�A�**

loss|-�<5F       �	�$ _fc�A�**

loss��;8_}C       �	ǹ _fc�A�**

lossX@�;M:2       �	�W_fc�A�**

loss$0=	��       �	��_fc�A�**

lossf�<�XQ       �	m�_fc�A�**

loss�7<ނ�<       �	�&_fc�A�**

loss˱:� ��       �	��_fc�A�**

loss^b<%]��       �	�^_fc�A�**

loss���<g���       �	��_fc�A�**

loss� �=vW�q       �	K�_fc�A�**

lossi�9=整�       �	g+_fc�A�**

loss�sJ<�H-�       �	��_fc�A�**

loss�[�<��$�       �	�{_fc�A�**

loss�-=�V��       �	�_fc�A�**

loss�,�;�Hk       �	&	_fc�A�**

lossr�X<�X       �	U�	_fc�A�**

loss�I�;W-�       �	TS
_fc�A�**

lossvW <��G       �	M�
_fc�A�**

lossؗ<�~V	       �	1�_fc�A�**

lossM@�;���       �	�_fc�A�**

loss��A=��|�       �	FD_fc�A�**

loss��<�Pm       �	�_fc�A�**

loss�x�<<j��       �	o_fc�A�**

loss��<����       �	�_fc�A�**

lossѾ�;{�63       �	��_fc�A�**

loss;wp;���H       �	)w_fc�A�**

loss �<)\�       �	_fc�A�**

loss-��;��J�       �	ϣ_fc�A�**

losse@<T��K       �	�6_fc�A�**

loss*��;S��       �	��_fc�A�**

loss�t�<:�X       �	Cr_fc�A�**

loss� *<F��       �	9_fc�A�**

loss���<bl|�       �	Q�_fc�A�**

loss���;?��       �	�@_fc�A�**

losssh<di�i       �	��_fc�A�**

loss=��<�$F\       �	nk_fc�A�**

loss�,<,���       �	�_fc�A�**

loss���<}�M       �	#�_fc�A�**

lossӳ�<T��       �	`;_fc�A�**

loss��K=���+       �	��_fc�A�**

lossڌ�:�X�       �	�m_fc�A�**

loss���;Buց       �	�_fc�A�**

lossH�:����       �	b�_fc�A�**

loss�;�	��       �	�5_fc�A�**

lossNB;6W�
       �	��_fc�A�+*

loss=[�;n��       �	wj_fc�A�+*

loss`U4<+��       �	�_fc�A�+*

loss�K:�9M       �	�_fc�A�+*

loss��=���       �	�1_fc�A�+*

loss�-=1�.�       �	��_fc�A�+*

loss�=2P��       �	ji_fc�A�+*

loss�5n=7b�c       �	 _fc�A�+*

loss&[�<�+7       �	d� _fc�A�+*

loss�<�jt>       �	J)!_fc�A�+*

loss� ;Yc�       �	��!_fc�A�+*

loss|I�<g�K       �	�3#_fc�A�+*

lossW�1:A�'�       �	��#_fc�A�+*

loss�}<vb�N       �	�}$_fc�A�+*

lossM�<��Q�       �	�%_fc�A�+*

loss�F.<9C��       �	�%_fc�A�+*

lossG�<Ç�       �	^d&_fc�A�+*

loss��_<�y�X       �	��&_fc�A�+*

loss/A=4��       �	V�'_fc�A�+*

loss
�<�x�       �	9(_fc�A�+*

lossa�=`*F       �	��(_fc�A�+*

lossNrZ;-�2�       �	�w)_fc�A�+*

loss��<o���       �	T�*_fc�A�+*

loss�<��       �	d>+_fc�A�+*

lossq��;|�\       �	k�,_fc�A�+*

lossȺ�<e/�"       �	�6-_fc�A�+*

loss�7<ڶ��       �	��-_fc�A�+*

loss�k�:��T�       �	��._fc�A�+*

loss�6<���B       �	S/_fc�A�+*

lossZ��;~�d       �	$�/_fc�A�+*

loss��;�ϩK       �	�0_fc�A�+*

loss�J�<��_-       �	�]1_fc�A�+*

loss�{;�Cm       �	9
2_fc�A�+*

lossT�L<���       �	�2_fc�A�+*

loss,:�;��a       �	�g3_fc�A�+*

loss��n=��I�       �	]�3_fc�A�+*

lossM܉=2��       �	��4_fc�A�+*

lossJ�;�HY�       �	�K5_fc�A�+*

loss�E<~:��       �	j�5_fc�A�+*

loss2�<1���       �	�z6_fc�A�+*

loss�D�<�サ       �	�7_fc�A�+*

lossqn�=���       �	"�7_fc�A�+*

loss2��<�tHc       �	u=8_fc�A�+*

loss�5{<!Tr       �	�8_fc�A�+*

lossq�O;�+f       �	 z9_fc�A�+*

loss� =��<u       �	�:_fc�A�+*

loss觕<�`%t       �	��:_fc�A�+*

loss�l=~w�e       �	�:;_fc�A�+*

loss`j"=�L       �	�;_fc�A�+*

loss|��<S;60       �	'i<_fc�A�+*

loss-,=z6ר       �	=_fc�A�+*

loss�1=.�       �	ܜ=_fc�A�+*

losse�9:1'��       �	�7>_fc�A�+*

loss�6�;'m�}       �	��>_fc�A�+*

loss�^�<љo�       �	�k?_fc�A�+*

lossƑ�<d͹       �	�@_fc�A�+*

lossM�m<|�Y$       �	�@_fc�A�+*

loss�ϝ<�e�       �	�7A_fc�A�+*

loss�9>;�f��       �	(�A_fc�A�+*

lossӂ=��"       �	SxB_fc�A�+*

loss�Y5<�gy)       �	�C_fc�A�+*

loss ��<'l>_       �	��C_fc�A�+*

loss��%<Mi(       �	WAD_fc�A�+*

loss hC:c�y�       �	�D_fc�A�+*

lossx[�<�P��       �	<�E_fc�A�+*

lossx8�;��       �	�F_fc�A�+*

loss }�;��)�       �	�F_fc�A�+*

lossqn�<�f�       �	qYG_fc�A�+*

loss��;�       �	H_fc�A�+*

loss�	c;�?�>       �	��I_fc�A�+*

loss)Ŵ<�>V{       �	��J_fc�A�+*

loss�R(<	�       �	W`K_fc�A�+*

lossT�<�P�       �	G L_fc�A�+*

lossn��;�YK�       �	˞L_fc�A�+*

loss�&P=���       �	�3M_fc�A�+*

loss��<�V       �	<JN_fc�A�+*

loss�WG;!+��       �	O_fc�A�+*

lossM�/;�h�       �	7�O_fc�A�+*

lossM�=IK��       �	[DP_fc�A�+*

loss��;��B�       �	X�P_fc�A�+*

loss8|�;��i       �	�{Q_fc�A�+*

loss�j�<,?f       �	UR_fc�A�+*

lossZ X;|�y�       �	K�R_fc�A�+*

loss��=�$       �	�XS_fc�A�+*

loss�V=�r       �	��S_fc�A�+*

lossMi<od=�       �	�T_fc�A�+*

loss�9_<��C�       �		3U_fc�A�+*

loss��=�X�       �	��U_fc�A�+*

lossT2�<�RY�       �	iV_fc�A�+*

loss�;I�;�       �	�W_fc�A�+*

loss��<i���       �	�W_fc�A�+*

loss��<:Y�m       �	�:X_fc�A�+*

loss{��<����       �	��X_fc�A�+*

loss�c@;��Ϝ       �	ȘY_fc�A�+*

lossR9a<	++�       �	_CZ_fc�A�+*

loss�ٜ:Pn6�       �	��Z_fc�A�+*

loss�{:=���R       �	�~[_fc�A�+*

loss
��;%7w       �	v\_fc�A�+*

loss�B;����       �	�\_fc�A�+*

loss��;�zS       �	X]_fc�A�+*

loss���<4��       �	��]_fc�A�+*

loss\Re<�C�o       �	ƅ^_fc�A�+*

lossa72;����       �	�*__fc�A�+*

loss��:<�A�       �	��__fc�A�+*

loss�m�:���j       �	�q`_fc�A�+*

lossL8�<�P�:       �	Aa_fc�A�+*

lossN=����       �	�a_fc�A�+*

loss.�<�9�R       �	�?b_fc�A�+*

lossz�<�N�?       �	�b_fc�A�+*

loss�X;
�q�       �	�nc_fc�A�+*

loss��)<O�b�       �	
d_fc�A�+*

loss��`<Z�#/       �	��d_fc�A�+*

lossdp�=�A��       �	�@e_fc�A�+*

lossS�4='F!       �	�e_fc�A�+*

loss�I=�̈�       �	�vf_fc�A�+*

loss.'L=�o�y       �	�g_fc�A�+*

loss\ >ќu�       �	`�g_fc�A�+*

loss+v<ZS�       �	Lh_fc�A�+*

lossePH<~~��       �	��h_fc�A�+*

loss��<���       �	t|i_fc�A�+*

loss}�\<���;       �	'j_fc�A�+*

loss�<F�1�       �	��j_fc�A�+*

loss
��:�X,�       �	�Gk_fc�A�+*

loss<��<�;Y       �	a�k_fc�A�+*

loss��;��|+       �	{l_fc�A�+*

lossZ56<Ñ"2       �	c*m_fc�A�+*

lossTb<�!��       �	|an_fc�A�+*

loss=�<V 0       �		2o_fc�A�+*

lossr;����       �	��o_fc�A�,*

losss��<��e       �	�rp_fc�A�,*

loss��b<����       �	�q_fc�A�,*

loss���<��}#       �	��q_fc�A�,*

loss��/=淕|       �	�Qr_fc�A�,*

loss�;�<_Tũ       �	��r_fc�A�,*

loss�J=�Ɂ       �	+�s_fc�A�,*

loss���<wK�a       �	N*t_fc�A�,*

loss��;����       �	G�t_fc�A�,*

loss�G8<,s2f       �	Ktu_fc�A�,*

loss�=�)�       �	(v_fc�A�,*

losse�;{P��       �	�v_fc�A�,*

lossN��<���H       �	>w_fc�A�,*

loss��;@���       �	Y�w_fc�A�,*

loss6��<+�       �	vx_fc�A�,*

loss�)-<�?s       �	�y_fc�A�,*

loss,RA<5u11       �	T�y_fc�A�,*

lossJ�h<nÚ       �	�Bz_fc�A�,*

loss�<;�Ŭ       �	'�z_fc�A�,*

loss,�)<��4�       �	2v{_fc�A�,*

loss� =��       �	5|_fc�A�,*

loss��0<G9>       �	�|_fc�A�,*

lossCL?;ӱ��       �	;}_fc�A�,*

losst�;j�       �	��}_fc�A�,*

loss��<���9       �	zo~_fc�A�,*

loss�ݴ<P��(       �	y_fc�A�,*

loss�.<��       �	��_fc�A�,*

loss���<B��       �	�R�_fc�A�,*

loss@��<���M       �	�_fc�A�,*

loss�O<E�h�       �	/��_fc�A�,*

loss�)=�X�K       �	��_fc�A�,*

loss�<���       �	9��_fc�A�,*

loss?T�;�y��       �	�V�_fc�A�,*

loss���<@T_�       �	*p�_fc�A�,*

loss�U=;�%�D       �	f�_fc�A�,*

lossWH�<��       �	Ҭ�_fc�A�,*

loss*��<�>�       �	
I�_fc�A�,*

loss�q=Ա��       �	�_fc�A�,*

loss�@=3�C       �	�x�_fc�A�,*

loss{�^<���       �	��_fc�A�,*

loss[�;~lVL       �	o��_fc�A�,*

lossW�<��       �	a6�_fc�A�,*

loss��<Ezz�       �	_ω_fc�A�,*

lossr��=�6�       �	jl�_fc�A�,*

loss�*<�r��       �	��_fc�A�,*

losswN�:M2��       �	{��_fc�A�,*

loss�:d�Q       �	�6�_fc�A�,*

lossL'�;C�y�       �	JҌ_fc�A�,*

loss���<����       �	�k�_fc�A�,*

lossѿY;�vF�       �	��_fc�A�,*

lossI�;�e�       �	4��_fc�A�,*

loss@�,;:E��       �	�N�_fc�A�,*

loss��;����       �	�_fc�A�,*

lossʜ�<u��6       �	f��_fc�A�,*

loss-b�;o�@       �	��_fc�A�,*

lossTs�<pXP       �	`��_fc�A�,*

loss�n_<��       �	�G�_fc�A�,*

loss��?<��(e       �	�r�_fc�A�,*

loss�+�<q�       �	R�_fc�A�,*

loss��C:;[��       �	��_fc�A�,*

losst� <���       �	,F�_fc�A�,*

loss��:c��       �	��_fc�A�,*

loss�Z�:6*�       �	R�_fc�A�,*

loss�E<��`q       �	��_fc�A�,*

loss�=��gi       �	4��_fc�A�,*

lossoj�<
!f       �	�c�_fc�A�,*

loss-Z�<��       �	H��_fc�A�,*

loss�'?=�M8       �	���_fc�A�,*

loss}�<�Q�       �	{.�_fc�A�,*

loss���;24�9       �	H_fc�A�,*

loss�6<�p��       �	.V�_fc�A�,*

loss��T=�B       �	��_fc�A�,*

loss�L=�Ճ       �	A��_fc�A�,*

loss�M7<t�٘       �	�"�_fc�A�,*

lossuq=V��y       �	���_fc�A�,*

loss�2p;��       �	Z�_fc�A�,*

loss��<-l%       �	M��_fc�A�,*

loss�<��)�       �	)��_fc�A�,*

loss�;�1�       �	�-�_fc�A�,*

loss���;�"       �	�¡_fc�A�,*

loss�<��:       �	�`�_fc�A�,*

loss|�<j$�       �	��_fc�A�,*

lossiut<�'��       �	���_fc�A�,*

loss%��;po�       �	�<�_fc�A�,*

loss���<W�S�       �	�Ϥ_fc�A�,*

loss�mH;>�`r       �	4g�_fc�A�,*

lossf��<���y       �	H��_fc�A�,*

loss��>1Ƥ9       �	И�_fc�A�,*

loss��a=�OB�       �	���_fc�A�,*

loss�O<�!��       �	ke�_fc�A�,*

lossS8�<	�_�       �	��_fc�A�,*

losse/4<�,��       �	�ȩ_fc�A�,*

lossHtz;5�S�       �	d�_fc�A�,*

loss|N�<jUd       �	���_fc�A�,*

loss���<�M۬       �	���_fc�A�,*

loss��<�%SJ       �	�V�_fc�A�,*

lossqa|<���b       �	��_fc�A�,*

loss�X=��XX       �	���_fc�A�,*

loss���<$N��       �	.9�_fc�A�,*

loss���<ǁ�       �	�ݮ_fc�A�,*

lossB�<3�k8       �	��_fc�A�,*

loss_�;;h@�=       �	o,�_fc�A�,*

loss��D<Y4D�       �	̰_fc�A�,*

loss�8=^4g�       �	,��_fc�A�,*

loss�	�=w��U       �	;�_fc�A�,*

loss�py<$�3Z       �	ز_fc�A�,*

loss�R�<7��       �	�ʳ_fc�A�,*

loss�j=GG{4       �	���_fc�A�,*

loss��+<9+�       �	�e�_fc�A�,*

lossZB�;4��       �	V��_fc�A�,*

loss��=���       �	I׷_fc�A�,*

loss�D
<#�@;       �	Xq�_fc�A�,*

loss��+;PB�K       �	+�_fc�A�,*

loss��=��$&       �	�й_fc�A�,*

loss\�0=��$       �	ֺ_fc�A�,*

loss�{X=)��       �	�ʻ_fc�A�,*

loss�a=̿)       �	.��_fc�A�,*

lossf!E<IK�       �	�b�_fc�A�,*

lossxZ<M
\       �	�Z�_fc�A�,*

loss�M�:�bB       �	O˿_fc�A�,*

loss�H(<Y��       �	���_fc�A�,*

lossI�<�Q��       �	E��_fc�A�,*

lossE�l<���<       �	��_fc�A�,*

losso��<�/n�       �	���_fc�A�,*

loss��<J)b       �	�[�_fc�A�,*

lossT��;�\(       �	��_fc�A�,*

lossd�;S�;       �	F��_fc�A�,*

losse�5=���       �	���_fc�A�,*

loss�lK;�ʐi       �	�N�_fc�A�-*

loss�<<���4       �	�K�_fc�A�-*

loss<;*;���       �	�~�_fc�A�-*

loss�
=��ɜ       �	3�_fc�A�-*

loss��<�jZJ       �	@h�_fc�A�-*

loss�E >�Q�I       �	9��_fc�A�-*

loss͝�<���       �	�b�_fc�A�-*

loss�mq<��c       �	ͭ�_fc�A�-*

loss�N�<<&��       �	��_fc�A�-*

loss�K<ØT)       �	���_fc�A�-*

loss�t�;�^E
       �	���_fc�A�-*

loss�!=3m�K       �	^+�_fc�A�-*

loss���;q*Ӊ       �	 ��_fc�A�-*

loss�^=<~�\       �	�a�_fc�A�-*

loss�M<�Aϊ       �	�`�_fc�A�-*

loss���<��       �	�&�_fc�A�-*

lossR�><W��       �	C�_fc�A�-*

lossq'@=�*u1       �	[��_fc�A�-*

loss\l=�2       �	Nd�_fc�A�-*

loss�G<��z�       �	[�_fc�A�-*

lossI�<��/       �	-�_fc�A�-*

loss���;��N!       �	�4�_fc�A�-*

loss�s�;~�c       �	���_fc�A�-*

lossQl�;�Tn�       �	f�_fc�A�-*

loss�e�;���       �	7T�_fc�A�-*

loss�e<d�C�       �	ǁ�_fc�A�-*

lossws�<���       �	�P�_fc�A�-*

lossDf�<�,       �	U��_fc�A�-*

loss��>=g���       �	���_fc�A�-*

lossͤy;��k       �	���_fc�A�-*

loss��;};��       �	��_fc�A�-*

loss��<G�       �	��_fc�A�-*

loss��L<|��7       �	��_fc�A�-*

loss�!�;�Z       �	�T�_fc�A�-*

loss6`�=~�a�       �	���_fc�A�-*

loss@�=M�zk       �	m�_fc�A�-*

lossD�$<1b��       �	��_fc�A�-*

loss��:��	�       �	���_fc�A�-*

loss�׻;3xQ]       �	��_fc�A�-*

loss��;��Ŀ       �	!!�_fc�A�-*

lossL�<���       �	`��_fc�A�-*

loss1c@<Tb�       �	%t�_fc�A�-*

loss�y�<}Z       �	/l�_fc�A�-*

loss
��<`	>�       �	�X�_fc�A�-*

loss�|�;��?D       �	>��_fc�A�-*

loss �$<q>}�       �	؞�_fc�A�-*

loss�; ��7       �	���_fc�A�-*

loss���<*Zc�       �	~��_fc�A�-*

loss'/<7�       �	��_fc�A�-*

loss۸*<g>f0       �	���_fc�A�-*

loss�d <d��z       �	���_fc�A�-*

loss��<f$       �	�w�_fc�A�-*

loss{��<�[��       �	��_fc�A�-*

loss��;�i6�       �	-�_fc�A�-*

loss�	=��W       �	�1�_fc�A�-*

loss��<�c�=       �	�	�_fc�A�-*

loss2G�<*�U�       �	���_fc�A�-*

lossIVD;"{w�       �	Gu�_fc�A�-*

loss�;�       �	=��_fc�A�-*

losso=��@(       �	|�_fc�A�-*

lossqk<O��
       �	�]�_fc�A�-*

loss�';���       �	�� `fc�A�-*

loss�w=�X�       �	�J`fc�A�-*

loss$�;�IЈ       �	V`fc�A�-*

lossl�=d�W�       �	5�`fc�A�-*

loss�&=M2�}       �	�d`fc�A�-*

loss*�=���|       �	c`fc�A�-*

loss�<����       �	ĵ`fc�A�-*

loss&��;�; �       �	ge`fc�A�-*

loss{�r=+�       �	�`fc�A�-*

losss��:Y�       �	�`fc�A�-*

loss8V�<sY+~       �	Ec`fc�A�-*

loss���<��b       �	�`fc�A�-*

loss&D<���       �	v�`fc�A�-*

lossr�;X�C9       �	^L	`fc�A�-*

loss���;	e��       �	E�	`fc�A�-*

loss]<�<1�@x       �	�
`fc�A�-*

lossx=�;_>��       �	�D`fc�A�-*

loss{�<��       �	�`fc�A�-*

lossQO�;�&=       �	�`fc�A�-*

loss��;;�<Q       �	n0`fc�A�-*

loss:��<�X>]       �	f�`fc�A�-*

loss�^�<C� �       �	
�`fc�A�-*

lossC�u<���       �	�+`fc�A�-*

loss��<;ќ        �	�`fc�A�-*

loss��;2�jx       �	��`fc�A�-*

loss��a<z�i       �	�=`fc�A�-*

loss!M=���9       �	��`fc�A�-*

loss�߆<�G�G       �	}`fc�A�-*

lossr :<5��H       �	�`fc�A�-*

lossXzL<\e�       �	��`fc�A�-*

loss:�n=�}S�       �	�]`fc�A�-*

loss�Y;R[�       �	Y�`fc�A�-*

loss,��;T��       �	
�`fc�A�-*

loss|D�;�4a�       �	�i`fc�A�-*

loss���=kb�       �	�	`fc�A�-*

loss�ַ<|�N       �	*�`fc�A�-*

lossk��<�fk�       �	�E`fc�A�-*

lossI<�ڴ       �	��`fc�A�-*

loss�(.=�lA/       �	Wz`fc�A�-*

loss��<�$�       �	�3`fc�A�-*

loss�E<�l       �	��`fc�A�-*

loss���;{���       �	^g`fc�A�-*

lossTʷ;�k�       �	`fc�A�-*

lossvpV=�m!�       �	8�`fc�A�-*

loss
=�U       �	�Y`fc�A�-*

loss��<3��       �	�`fc�A�-*

loss��:U�]�       �	ۦ`fc�A�-*

loss��<@I�       �	C`fc�A�-*

loss���;`���       �	W�`fc�A�-*

loss��<�G�       �	;� `fc�A�-*

loss�H=�a�j       �	2!`fc�A�-*

loss��;���       �	,�!`fc�A�-*

loss��<9�$�       �	o"`fc�A�-*

loss�>=�;��       �	�#`fc�A�-*

loss��=�+�       �	�-$`fc�A�-*

loss\�:��g�       �	>�$`fc�A�-*

loss�h;=��5       �	jm%`fc�A�-*

loss��<���       �	&`fc�A�-*

loss��X<s���       �	)�&`fc�A�-*

lossl=[pT       �	/'`fc�A�-*

loss?�=�lF       �	�'`fc�A�-*

lossd@�;��9�       �	�t(`fc�A�-*

loss�4�;2M       �	x)`fc�A�-*

loss.�,<T��       �	)�)`fc�A�-*

loss� <#cq       �	�Q*`fc�A�-*

lossm�<���       �	�*`fc�A�-*

lossę�;w�wv       �	��+`fc�A�-*

loss���<V���       �	L,`fc�A�.*

loss�)�;�mF`       �	��,`fc�A�.*

losswZ<�4dK       �	N�-`fc�A�.*

loss?�==��f4       �	�T.`fc�A�.*

lossz��;�*�        �	�t/`fc�A�.*

lossܗv:nԊ)       �	�u0`fc�A�.*

loss�(i=�И       �	�1`fc�A�.*

loss���;��xq       �	�?2`fc�A�.*

loss�P=�5       �	��2`fc�A�.*

loss�t�<��"       �	a�3`fc�A�.*

loss��;꽠&       �	�O4`fc�A�.*

loss�5a<�琈       �	��4`fc�A�.*

loss��= tL�       �	i�5`fc�A�.*

loss��{<���F       �	��6`fc�A�.*

loss���:`��r       �	<7`fc�A�.*

loss��[;���       �	��7`fc�A�.*

lossʝ<רWD       �	��8`fc�A�.*

loss��"<ʂ��       �	{I9`fc�A�.*

lossAy=�a_�       �	��9`fc�A�.*

loss) <EO�       �	ڐ:`fc�A�.*

lossa}�:�7��       �	76;`fc�A�.*

loss�O�<̝�       �	��;`fc�A�.*

loss�<�P�d       �	�v<`fc�A�.*

loss�xR;V�k       �	�=`fc�A�.*

loss�p)<���T       �	��=`fc�A�.*

loss*<��       �	�g>`fc�A�.*

loss�Sl<�cK       �	
?`fc�A�.*

loss�l�;�8��       �	m�?`fc�A�.*

loss\�7<8L�       �	fI@`fc�A�.*

loss�z=,!�       �	��@`fc�A�.*

loss���;DS�m       �	��A`fc�A�.*

loss�&�:<~09       �	�*B`fc�A�.*

lossx�=r3�]       �	��B`fc�A�.*

loss�c1<���&       �	�hC`fc�A�.*

lossn�N<�p4�       �	�
D`fc�A�.*

losso�d;��h�       �	��D`fc�A�.*

loss(�:G��       �	�DE`fc�A�.*

loss=�<j��       �	��E`fc�A�.*

loss���;��o+       �	*rF`fc�A�.*

loss��h<��(C       �	�G`fc�A�.*

loss��<'�       �	ŪG`fc�A�.*

loss�o�<<[��       �	gGH`fc�A�.*

loss �L<�X��       �	��H`fc�A�.*

loss�-<=��b       �	/�I`fc�A�.*

lossI(<CV�       �	Z)J`fc�A�.*

lossf�;�T'�       �	��J`fc�A�.*

loss�<���       �	cK`fc�A�.*

loss�w�:�,��       �	�L`fc�A�.*

loss@��;��       �	ÛL`fc�A�.*

loss���;`���       �	H4M`fc�A�.*

loss?��<��܄       �	Q�M`fc�A�.*

loss�_9��'       �	�rN`fc�A�.*

loss�-�;���       �	5O`fc�A�.*

loss��<�a��       �	��O`fc�A�.*

loss�cv9-�Խ       �	�@P`fc�A�.*

loss[�8(z�       �	U�P`fc�A�.*

loss�C;�g��       �	qvQ`fc�A�.*

loss��;�O�j       �	�R`fc�A�.*

loss-�;h3��       �	z�R`fc�A�.*

loss���8G�h        �	�nS`fc�A�.*

loss�K�;4	�       �	�T`fc�A�.*

loss��<�]�!       �	��T`fc�A�.*

lossHD�:��1+       �		NU`fc�A�.*

loss(�<��
�       �	��U`fc�A�.*

loss6(�<���       �	;�V`fc�A�.*

loss7��<IҪM       �	�%W`fc�A�.*

loss4�;���W       �	��W`fc�A�.*

loss[ς;�L�(       �	:wX`fc�A�.*

lossa3#=�_��       �	7Y`fc�A�.*

loss]W�;kL��       �	иY`fc�A�.*

lossz��:����       �	�TZ`fc�A�.*

loss/�<t�-       �	��Z`fc�A�.*

loss��:`Y5N       �	@�[`fc�A�.*

lossEk�<�:�       �	�\`fc�A�.*

losss�=PQ��       �	a2]`fc�A�.*

lossl��;[4�       �	��]`fc�A�.*

loss�N=b	�        �	�^^`fc�A�.*

lossM�s<_��       �	�^`fc�A�.*

loss���<.���       �	��_`fc�A�.*

loss��<3�s�       �	�.``fc�A�.*

loss���</4�&       �	��``fc�A�.*

loss��X<�P�       �	�]a`fc�A�.*

lossx�:�4�Q       �	��a`fc�A�.*

loss�a�<+�E       �	��b`fc�A�.*

lossr��<]�<�       �	)Ac`fc�A�.*

loss��=�G�       �	w�c`fc�A�.*

loss$�O=����       �	Grd`fc�A�.*

loss��;{�9       �	>e`fc�A�.*

loss�n�=>�F       �	��e`fc�A�.*

loss6i;�8@       �	�Ff`fc�A�.*

lossEu=��       �	��f`fc�A�.*

loss��<�[�       �	Ōg`fc�A�.*

loss�6�<�ecN       �	�+h`fc�A�.*

loss�<���       �	��h`fc�A�.*

loss�U<܆�x       �	|ci`fc�A�.*

loss˻;}�c�       �	� j`fc�A�.*

loss\��;'���       �	�j`fc�A�.*

loss���:���       �	�5k`fc�A�.*

loss��C<�+�       �	�k`fc�A�.*

loss4<K+F       �	@hl`fc�A�.*

loss���<&��       �	\ m`fc�A�.*

loss�i;���/       �	Ԝm`fc�A�.*

loss���:}��       �	�3n`fc�A�.*

loss�Z#<�s0�       �	�n`fc�A�.*

lossv��;r&�       �	�go`fc�A�.*

loss4_�<2���       �	ۆp`fc�A�.*

loss�K!<킭�       �	nq`fc�A�.*

loss�O@=8ϗv       �	zr`fc�A�.*

lossI��<���       �	�et`fc�A�.*

lossV(;|�4�       �	Mu`fc�A�.*

loss;jW=�bs       �	5�u`fc�A�.*

loss���:l���       �	�Wv`fc�A�.*

loss�)�:��       �	�w`fc�A�.*

loss�B�<xj�       �	���`fc�A�.*

loss#ں<�'*�       �	�*�`fc�A�.*

loss��=/� U       �	���`fc�A�.*

lossD|y=FO�       �	8.�`fc�A�.*

loss�0=`76"       �	�Ӓ`fc�A�.*

loss� <K��x       �	��`fc�A�.*

lossLa�<���       �	�,�`fc�A�.*

loss ݽ<��?       �	�˔`fc�A�.*

loss��E=��o�       �	��`fc�A�.*

loss��P;�/g       �	@��`fc�A�.*

loss�:>��       �	��`fc�A�.*

lossF'�< ,S5       �	-@�`fc�A�.*

loss }O=���&       �	g�`fc�A�.*

lossv0<����       �	��`fc�A�.*

loss���;wқ       �	M.�`fc�A�.*

loss�<4���       �	�̚`fc�A�/*

lossi�:�n��       �	�i�`fc�A�/*

loss��;�.�<       �	>�`fc�A�/*

loss��5=,��       �	蟜`fc�A�/*

lossF
�<"��       �	�R�`fc�A�/*

loss@�3<%H[B       �	���`fc�A�/*

lossf*i<��W�       �	Ҫ�`fc�A�/*

loss��#;����       �	rR�`fc�A�/*

loss��<��h       �	��`fc�A�/*

loss��g<]���       �	Y��`fc�A�/*

lossSX3<v���       �	�Q�`fc�A�/*

loss�L<0n�:       �	 �`fc�A�/*

lossB8<	o&�       �	Ū�`fc�A�/*

loss�|�<Țp�       �	�S�`fc�A�/*

loss��H<�{�N       �	���`fc�A�/*

lossI��;��$       �	=��`fc�A�/*

lossa�;�I       �	�5�`fc�A�/*

loss($�;�.�       �	T�`fc�A�/*

loss�#=m-0O       �	�{�`fc�A�/*

loss��<��`       �	RH�`fc�A�/*

lossz��<V�m�       �	��`fc�A�/*

loss�H ;�3m       �	j��`fc�A�/*

loss$Ɯ=�K       �	}"�`fc�A�/*

loss�o�< H��       �	�Ҫ`fc�A�/*

loss}'[<��:*       �	F|�`fc�A�/*

loss.�r;fTQO       �	�*�`fc�A�/*

loss�Э;h�^�       �	�Ǭ`fc�A�/*

loss��<,�qc       �	�c�`fc�A�/*

loss)�<����       �	N
�`fc�A�/*

loss8
<���/       �	��`fc�A�/*

loss�<ɏ>�       �	KW�`fc�A�/*

loss��{<��       �	��`fc�A�/*

loss
B�;���G       �	�ʰ`fc�A�/*

loss��<��iF       �	�f�`fc�A�/*

loss`��9��?f       �	��`fc�A�/*

loss��;��D|       �	���`fc�A�/*

loss<�=M�       �	���`fc�A�/*

loss��;O7��       �	2=�`fc�A�/*

lossp�<��       �	�ڴ`fc�A�/*

loss��W;H	       �	!s�`fc�A�/*

loss�1�:_��m       �	��`fc�A�/*

loss
�9֯       �	;ö`fc�A�/*

loss���9����       �	�p�`fc�A�/*

lossz_�;���       �	�.�`fc�A�/*

loss�=��$	       �	TŸ`fc�A�/*

loss��f<�m�-       �	o�`fc�A�/*

lossj��:��l	       �	�1�`fc�A�/*

loss�;{M�       �	��`fc�A�/*

loss��`<�O       �	�+�`fc�A�/*

loss�X;��u�       �	*�`fc�A�/*

loss�=�p �       �	ʩ�`fc�A�/*

loss�� =W��       �	�]�`fc�A�/*

loss�X<A,
N       �	Yi�`fc�A�/*

lossx�u=����       �	��`fc�A�/*

loss���<����       �	���`fc�A�/*

lossc^�;~�N       �	� �`fc�A�/*

loss]rU<���g       �	��`fc�A�/*

loss�<<�90       �	�`fc�A�/*

loss�ô;�8@�       �	���`fc�A�/*

loss6$t<ܒBm       �	I��`fc�A�/*

loss���;v��*       �	���`fc�A�/*

loss�J;��       �	d�`fc�A�/*

loss�E�:_t�l       �	���`fc�A�/*

lossf�<na�       �	�S�`fc�A�/*

loss��;���       �	��`fc�A�/*

loss�d�;S�h       �	��`fc�A�/*

loss��^<W��       �	�7�`fc�A�/*

loss\f�;iI8       �	=��`fc�A�/*

lossN�*<�T�;       �	_A�`fc�A�/*

loss��;l�c       �	1��`fc�A�/*

loss���;EC޷       �	���`fc�A�/*

loss�)�:zݗ       �	�)�`fc�A�/*

loss:i�<�`       �	���`fc�A�/*

lossoI=��y       �	}]�`fc�A�/*

loss_ϡ;��C       �	n��`fc�A�/*

loss�=�Jɯ       �	b��`fc�A�/*

lossl =T�       �	�r�`fc�A�/*

loss�g;��L       �	S�`fc�A�/*

loss��d;�	*       �	���`fc�A�/*

lossqbb=�t��       �	��`fc�A�/*

loss�P^;+�       �	O#�`fc�A�/*

loss�yA=�}��       �	x��`fc�A�/*

loss;P�;d�_�       �	fl�`fc�A�/*

loss��=P�C       �	�	�`fc�A�/*

loss}�;�AҾ       �	���`fc�A�/*

loss|R7;zZ��       �	�N�`fc�A�/*

loss�\�:��       �	_��`fc�A�/*

loss
�;`^a�       �	Ӈ�`fc�A�/*

loss#5�<���z       �	�m�`fc�A�/*

loss���;L[       �	U�`fc�A�/*

loss��s=��       �	F��`fc�A�/*

lossW�l:�n       �	LU�`fc�A�/*

loss'�;	D�       �	X�`fc�A�/*

loss1�P=�b �       �	��`fc�A�/*

lossoH�;T�57       �	�@�`fc�A�/*

loss��(=�ӳD       �	v��`fc�A�/*

loss�c<���       �	��`fc�A�/*

loss� 	<�q�S       �	�l�`fc�A�/*

loss��\<��_k       �	�`fc�A�/*

loss���<D�h�       �	=��`fc�A�/*

lossnhi:R�mr       �	qW�`fc�A�/*

losso�;��.7       �	���`fc�A�/*

loss��;�O�       �	���`fc�A�/*

loss�h+<"�|       �	-��`fc�A�/*

lossl�;��+�       �	�3�`fc�A�/*

loss	�z;�_�       �	���`fc�A�/*

loss��<ڭ��       �	=}�`fc�A�/*

loss|��;��a       �	t'�`fc�A�/*

loss��z;��:�       �	���`fc�A�/*

loss�`�9Sbb@       �	�_�`fc�A�/*

loss��B=`_s�       �	���`fc�A�/*

loss᭜</�4Q       �	h��`fc�A�/*

loss�p�<���       �	C<�`fc�A�/*

loss4qm< �eN       �	e��`fc�A�/*

loss���<SP@       �	���`fc�A�/*

loss=}�<uq��       �	�A�`fc�A�/*

loss
�</�@X       �	���`fc�A�/*

lossȷ�<D��c       �	)��`fc�A�/*

loss:0�;�Qs       �	NE�`fc�A�/*

losse��<BE��       �	���`fc�A�/*

loss6�B=�cbD       �	ɒ�`fc�A�/*

lossq�4<݅f�       �	�9�`fc�A�/*

loss��;�X       �	���`fc�A�/*

lossx_�;�       �	Ԁ�`fc�A�/*

loss:��<mƊ2       �	$�`fc�A�/*

loss櫥<�(�       �	D�`fc�A�/*

loss=9<�ޮ       �	+�`fc�A�/*

lossu�<c:�"       �	�`fc�A�/*

loss`0�<�]��       �	7��`fc�A�0*

lossn-`=o�'2       �	�i�`fc�A�0*

lossx<��       �	#�`fc�A�0*

loss{i;F��       �	��`fc�A�0*

lossEZ_;y�N�       �	�j�`fc�A�0*

losse��<҂ p       �		k�`fc�A�0*

loss��K<��       �	��`fc�A�0*

loss��;.��       �	���`fc�A�0*

loss:<齥       �	YO�`fc�A�0*

lossя�<�k�       �	���`fc�A�0*

loss$<�8H       �	��`fc�A�0*

loss���<l5�A       �	��`fc�A�0*

loss:�R:$�Z�       �	�R�`fc�A�0*

lossk��;t��       �	���`fc�A�0*

lossX��<��%       �	R��`fc�A�0*

loss�=�	-f       �	N�`fc�A�0*

loss}�{<z���       �	�o afc�A�0*

lossM�=#�<       �	fafc�A�0*

loss=�;Y$�       �	c�afc�A�0*

lossƷ<��mc       �	@Qafc�A�0*

loss~�;ٱ       �	`�afc�A�0*

loss���<u<��       �	&�afc�A�0*

lossSS<t6�       �	�6afc�A�0*

loss��_:�9�!       �	��afc�A�0*

loss�u=	�       �	%uafc�A�0*

lossd�<Ĝ��       �	afc�A�0*

loss���:=�|_       �	e�afc�A�0*

loss*֬<��j       �	Nafc�A�0*

loss�{;~��       �	a�afc�A�0*

loss���;���       �	){afc�A�0*

loss`4�;��       �	.	afc�A�0*

loss�.<(p�       �	E�	afc�A�0*

loss�$<xl}       �	�R
afc�A�0*

loss?j�<��:�       �	V�
afc�A�0*

loss���<���       �	��afc�A�0*

loss3��<g�+�       �	afc�A�0*

loss\%;���Z       �	Yafc�A�0*

lossJ:�8�8��       �	"�afc�A�0*

lossD=�,�       �	��afc�A�0*

loss���;��Y       �	{afc�A�0*

loss��/:o�       �	�afc�A�0*

loss�{�<�P�       �	��afc�A�0*

loss3�:��>       �	�_afc�A�0*

loss+�<���       �	��afc�A�0*

lossf�Y<����       �	��afc�A�0*

lossN�<�>�       �	3afc�A�0*

loss���=��;       �	�afc�A�0*

loss�<���       �	&mafc�A�0*

loss�%<�a�       �	`afc�A�0*

loss�;�G+       �	|�afc�A�0*

loss�,l<���       �	T5afc�A�0*

loss*%u<��/       �	:�afc�A�0*

loss���<�g�       �	hafc�A�0*

loss�%�;�U��       �	 afc�A�0*

loss�=���       �	�afc�A�0*

loss��;��s%       �	5'afc�A�0*

loss@3�;Uq�       �	ؼafc�A�0*

loss��<��       �	Tafc�A�0*

loss���;%��       �	E�afc�A�0*

loss ��;�Ú       �	�afc�A�0*

loss��<P�o�       �	�<afc�A�0*

lossW<%;�]�K       �	��afc�A�0*

loss��-;1˩       �	�nafc�A�0*

losslt�<F�1       �	� afc�A�0*

lossh"<�GZ       �	S� afc�A�0*

loss�V<x;��       �	K!afc�A�0*

loss�0�;��       �	�!afc�A�0*

loss֥�<-��(       �	�"afc�A�0*

loss8�<��       �	.#afc�A�0*

lossq��<a��d       �	�#afc�A�0*

loss@�<�I-       �	Cu$afc�A�0*

loss
�y<�#�       �	�%afc�A�0*

loss�*�=0�qC       �	�%afc�A�0*

loss[ �<��\       �	9F&afc�A�0*

loss4|�=ރ��       �	��&afc�A�0*

loss�^5=�^&�       �	{'afc�A�0*

loss�==���       �	Z(afc�A�0*

loss֡S;F��       �	��(afc�A�0*

loss&�<]��       �	�b)afc�A�0*

losst��<��       �	\ *afc�A�0*

loss ߢ<�1j       �	s�*afc�A�0*

lossj_�;ː�|       �	�8+afc�A�0*

loss�<�Y$x       �	I�+afc�A�0*

loss�r<���       �	�v,afc�A�0*

loss��<L��       �	 -afc�A�0*

loss]�; K�)       �	ܼ-afc�A�0*

loss�<���       �	"S.afc�A�0*

loss�2�:r^}�       �	-�.afc�A�0*

loss�a=[���       �	Á/afc�A�0*

lossE�2={3@�       �	\0afc�A�0*

loss��=�Z       �	�0afc�A�0*

loss�J�;3�g       �	e�1afc�A�0*

loss���;$%�H       �	��2afc�A�0*

loss�?�<v�[       �	�84afc�A�0*

lossD�<�,�g       �	�!5afc�A�0*

loss��<� .       �	��5afc�A�0*

loss���<:�s       �	�L7afc�A�0*

loss-��<4�܁       �	B�7afc�A�0*

loss��<����       �	'�8afc�A�0*

loss��1<�1�        �	�%9afc�A�0*

lossک�=�TE4       �	��9afc�A�0*

loss�W�<]�d(       �	�z:afc�A�0*

loss�'�=/���       �	�;afc�A�0*

loss.�;	�"       �	��=afc�A�0*

loss�;��|W       �	ܡ>afc�A�0*

lossz,�<΅2       �	�K?afc�A�0*

loss]�<�S�       �	m�?afc�A�0*

loss�Õ<6���       �	�@afc�A�0*

loss�}�<0I�       �	�.Aafc�A�0*

loss�j<�Py�       �	Bafc�A�0*

lossZ'�:o��       �	ƣBafc�A�0*

loss��2;ޜ�       �	�;Cafc�A�0*

loss��<m���       �	��Cafc�A�0*

loss���<c�A       �	�{Dafc�A�0*

loss��V<��6�       �	$Eafc�A�0*

loss
~�<�P       �	��Eafc�A�0*

loss�E�<}%=v       �	OYFafc�A�0*

loss�%<.zt�       �	�Gafc�A�0*

loss=U�<��       �	'�Gafc�A�0*

loss���;��       �	�;Hafc�A�0*

loss��;3�       �	��Hafc�A�0*

loss��q<�|U�       �	x{Iafc�A�0*

loss��;�C�       �	HJafc�A�0*

loss�C�<��%�       �	��Jafc�A�0*

loss���<󟚪       �	�MKafc�A�0*

loss$�?<?�       �	��Kafc�A�0*

loss��=W��       �	�}Lafc�A�0*

loss��<{w'�       �	MMafc�A�0*

losso�'<c��       �	O�Mafc�A�1*

loss���<8�       �	
jNafc�A�1*

loss�GH=B       �	Oafc�A�1*

loss6ӭ=��f       �	y�Oafc�A�1*

loss���<�	�       �	�NPafc�A�1*

loss�5;F�G       �	&Qafc�A�1*

lossE?A;g�|�       �	�Qafc�A�1*

lossA�<�.�       �	~URafc�A�1*

loss�ZD=����       �	�Rafc�A�1*

loss�wl;^%       �	��Safc�A�1*

loss��;��+%       �	o.Tafc�A�1*

losslY?<�Q�X       �	��Tafc�A�1*

loss���;�]�L       �	HmUafc�A�1*

lossx��<�f�       �	�Vafc�A�1*

loss�3�<��c	       �	��Vafc�A�1*

loss|\�<#�V       �	v2Wafc�A�1*

loss���;�ZR�       �	�Wafc�A�1*

loss��O<���       �	�`Xafc�A�1*

loss���;�N<�       �	'�Xafc�A�1*

loss�C;0G�       �	m�Yafc�A�1*

loss��.<Yuf�       �	+Zafc�A�1*

loss��<{H�y       �	+�Zafc�A�1*

loss�]k:U��V       �	�Y[afc�A�1*

loss�'@<���       �	�[afc�A�1*

lossQON<[�       �	ӄ\afc�A�1*

lossT(9<U�M�       �	e]afc�A�1*

loss�;<j1j       �	)�]afc�A�1*

lossњ"=�K�       �	$F^afc�A�1*

loss2e�<R?q^       �	�^afc�A�1*

loss���;��kK       �	�_afc�A�1*

loss!dG<<V��       �	"`afc�A�1*

loss�̖;��rF       �	й`afc�A�1*

loss���<��       �	"Raafc�A�1*

lossC<ǂ'�       �	�aafc�A�1*

lossn��=��߬       �	,�bafc�A�1*

loss���<�{
s       �	�cafc�A�1*

loss�IN=fB \       �	�cafc�A�1*

loss��';W�y       �	�Pdafc�A�1*

lossn�;,�kg       �	0�dafc�A�1*

loss�B;��       �	O�eafc�A�1*

losstL�;��T�       �	{2fafc�A�1*

loss��<�i�       �	�fafc�A�1*

lossή�;1�       �	�ogafc�A�1*

lossX��;��/r       �	�hafc�A�1*

lossO�z<���       �	��hafc�A�1*

lossL��;Ü�P       �	�Liafc�A�1*

loss���<���j       �	�Pjafc�A�1*

loss��<s�       �	��jafc�A�1*

loss�=Ħ��       �	D�kafc�A�1*

lossӮq<��a�       �	vRlafc�A�1*

loss/0�<�$�L       �	l�lafc�A�1*

loss~�<� �\       �	{�mafc�A�1*

loss��'<��9       �	� nafc�A�1*

loss)��;|�1       �	w�nafc�A�1*

lossn\<���K       �	�Qoafc�A�1*

lossJ��<�!       �	�oafc�A�1*

loss!Uh<J��       �	�pafc�A�1*

loss��<P[�@       �	�qafc�A�1*

lossi��<��Ё       �	@�qafc�A�1*

loss�C�<)�_�       �	�Xrafc�A�1*

loss��`;O��A       �	Z/safc�A�1*

loss�EB;3A�       �	�tafc�A�1*

loss�rr;�L       �	I�tafc�A�1*

loss�Z�<^-       �	Vuafc�A�1*

loss�T:����       �	*vafc�A�1*

loss��n<$       �	�vafc�A�1*

loss�6$<���       �	r3wafc�A�1*

lossv&=(Bc       �	_�wafc�A�1*

loss��:��A       �	Dnxafc�A�1*

loss���<_ �       �	kyafc�A�1*

lossIV=;���       �	b�yafc�A�1*

losse2�;�/�9       �	�:zafc�A�1*

loss�,7<8QGR       �	��zafc�A�1*

loss���=	�s�       �	Mg{afc�A�1*

loss��7=����       �	�|afc�A�1*

loss�`�<3��b       �	c�|afc�A�1*

loss7�<�       �	P:}afc�A�1*

loss�NI;��C       �	��}afc�A�1*

lossD� <d@�       �	�d~afc�A�1*

loss^<";i�O2       �	y!afc�A�1*

loss�W<�Ɍ�       �	��afc�A�1*

loss���=��       �	7m�afc�A�1*

loss��$=�oX�       �	h�afc�A�1*

loss�\<<�m'       �	Z��afc�A�1*

loss��;���       �	�A�afc�A�1*

loss��;4]��       �	�ڂafc�A�1*

lossX��<E�{7       �	r�afc�A�1*

lossl��<c!�B       �	�#�afc�A�1*

loss�G<�w�@       �	+Äafc�A�1*

loss�G�;��.       �	�i�afc�A�1*

loss$
�;3*��       �	���afc�A�1*

lossz�'<uW�       �	���afc�A�1*

losse�A=�8ْ       �	�2�afc�A�1*

lossZɹ<.2�       �	F·afc�A�1*

loss�?<����       �	�c�afc�A�1*

loss %W=Z03�       �	���afc�A�1*

loss��;�`       �	0��afc�A�1*

loss�5<��G*       �	1^�afc�A�1*

loss���<���       �	qW�afc�A�1*

loss�,�;��       �	+��afc�A�1*

loss?J\9S��K       �	���afc�A�1*

lossw�-<��S       �	�\�afc�A�1*

loss���<J��5       �	��afc�A�1*

loss�2�:e/I�       �	��afc�A�1*

lossm��<�Ȧn       �	�3�afc�A�1*

loss%L=�O��       �	6ʏafc�A�1*

lossC�r<�'y,       �	�e�afc�A�1*

loss��<�B�O       �	8N�afc�A�1*

loss�^�<Pa|(       �	%�afc�A�1*

lossAv]<@+c       �	3��afc�A�1*

loss�g<K�`�       �	G�afc�A�1*

loss�p�<~�L�       �	��afc�A�1*

loss�=�;�	�N       �	�{�afc�A�1*

lossQF�<(�i�       �	4�afc�A�1*

loss�<�	�-       �	�afc�A�1*

lossi3;��C�       �	���afc�A�1*

loss� �<K�׳       �	�%�afc�A�1*

loss�<F���       �	軗afc�A�1*

loss�d�;1�Y       �	�Q�afc�A�1*

loss���;��e       �	S�afc�A�1*

loss�=!<1�j�       �	Z��afc�A�1*

loss���;=6��       �	@�afc�A�1*

losse��<i��       �	p��afc�A�1*

loss[�?=:{<�       �	�M�afc�A�1*

loss�6=#i�1       �	y�afc�A�1*

loss��:j��       �	��afc�A�1*

loss��;P�e9       �	��afc�A�1*

lossR=�vȮ       �	1��afc�A�1*

loss�4<�	ۛ       �	�K�afc�A�2*

lossO@�:�2ȶ       �	m�afc�A�2*

loss��=:��       �	y�afc�A�2*

lossE&=�d��       �	Z�afc�A�2*

loss� �<�Ρ�       �	v��afc�A�2*

loss@��<��lA       �	�<�afc�A�2*

lossTi�;!�C       �	ӡafc�A�2*

loss�±<�#4�       �	�d�afc�A�2*

loss}�;�
&       �	^��afc�A�2*

loss���;3ʖ       �	���afc�A�2*

loss�u�;�S9�       �	��afc�A�2*

loss%�<��i�       �	���afc�A�2*

loss�m�<hD��       �	�G�afc�A�2*

loss�1�;Ï�       �	ݥafc�A�2*

loss
�L<T5\       �	�s�afc�A�2*

lossS��<ee�       �	��afc�A�2*

lossM��<�={       �	���afc�A�2*

loss��:�%�       �	HR�afc�A�2*

lossm�w;� v       �	��afc�A�2*

losst\7=���       �	���afc�A�2*

loss�B�;M�       �	q!�afc�A�2*

loss6"�<	̎[       �	g��afc�A�2*

loss{T�=?U��       �	�N�afc�A�2*

losst�;#��       �	K�afc�A�2*

loss�]=���       �	{�afc�A�2*

loss=C�<}X�X       �	��afc�A�2*

loss��<���       �	Ψ�afc�A�2*

loss��<�:�:       �	>�afc�A�2*

loss���<pNޅ       �	oخafc�A�2*

loss��Y;Z��]       �	*o�afc�A�2*

loss��7:+Vr�       �	5�afc�A�2*

loss�1=�L#M       �	+��afc�A�2*

loss��$;Lny+       �	�7�afc�A�2*

loss�u<Si^�       �	���afc�A�2*

loss�$(<zFm       �	>��afc�A�2*

lossAu�<��\D       �	y@�afc�A�2*

lossvJ�<�?��       �	c'�afc�A�2*

loss���;È�?       �	��afc�A�2*

loss�Kt<Y'O�       �	���afc�A�2*

loss�=% �       �	�E�afc�A�2*

loss�Ah;�q{H       �	��afc�A�2*

loss�qk<���_       �	�ŷafc�A�2*

loss��=3`-�       �	o��afc�A�2*

loss���<�Ĥz       �	;4�afc�A�2*

loss�!$=�l
       �	�afc�A�2*

lossΩC;1��n       �	㈻afc�A�2*

loss��4<�?��       �	�'�afc�A�2*

loss�=�}m       �	I��afc�A�2*

loss�89<����       �	b.�afc�A�2*

loss`��<%���       �	��afc�A�2*

loss���;B��       �	`�afc�A�2*

loss��<�|�       �	��afc�A�2*

loss��h;���        �	i�afc�A�2*

loss[�)<��eJ       �	+�afc�A�2*

loss�O<�L�       �	���afc�A�2*

loss��<.੢       �	9��afc�A�2*

lossNh�<��$�       �	�;�afc�A�2*

loss��*<=�       �	0��afc�A�2*

lossCj�;ט\       �	�l�afc�A�2*

lossL�;)r�[       �	h�afc�A�2*

loss?�=���7       �	��afc�A�2*

losso��<o���       �	�>�afc�A�2*

loss�<���       �	��afc�A�2*

loss:+{<1�}F       �	l{�afc�A�2*

loss�==4�Sq       �	�afc�A�2*

loss1=��t\       �	p��afc�A�2*

loss�q=�٧�       �	FA�afc�A�2*

loss��<CD\f       �	A��afc�A�2*

loss�G�;z�       �	*��afc�A�2*

loss���;`�(p       �	8-�afc�A�2*

lossc�;/!�H       �	p��afc�A�2*

loss�ǫ<6�_�       �	���afc�A�2*

loss,c$;d��>       �	� �afc�A�2*

loss�2�;���e       �	N��afc�A�2*

loss]VK=����       �	�K�afc�A�2*

loss_"�<@�Uc       �	q��afc�A�2*

loss��r;���       �	�v�afc�A�2*

loss�F�;�\�       �	�
�afc�A�2*

losso�<��       �	���afc�A�2*

loss��<���w       �	U0�afc�A�2*

loss�]c=:�tv       �	J��afc�A�2*

loss�D=�I�G       �	��afc�A�2*

loss���;�dx7       �	��afc�A�2*

lossV� ;��a�       �	��afc�A�2*

loss�j�;����       �	wH�afc�A�2*

loss��D<���       �	���afc�A�2*

lossn�=K��       �	Ou�afc�A�2*

loss���;���       �	��afc�A�2*

loss� �<�y��       �	���afc�A�2*

loss�^6<#�9�       �	C�afc�A�2*

loss\��;}S       �	 ��afc�A�2*

lossM�:<ᴐ       �	�z�afc�A�2*

lossV��;���       �	�afc�A�2*

loss\�:�U�       �	i��afc�A�2*

loss�*�;��`       �	IH�afc�A�2*

loss�m2;x(	�       �	>��afc�A�2*

lossd�<��!1       �	���afc�A�2*

loss�G�;�Yd�       �	v7�afc�A�2*

loss��C;P��O       �	[��afc�A�2*

loss�_<�{3       �	�f�afc�A�2*

loss`Zg<d�O       �	E�afc�A�2*

lossi�E<x:��       �	���afc�A�2*

loss取;�of�       �	�4�afc�A�2*

loss�)#;/�m�       �	���afc�A�2*

loss��2<�Oq�       �	�k�afc�A�2*

loss��$=F�W�       �	B�afc�A�2*

lossF�{<y���       �	��afc�A�2*

loss�<��̔       �	zP�afc�A�2*

lossd;�S8�       �	���afc�A�2*

loss��=���       �		��afc�A�2*

loss@,$=�(�       �	�"�afc�A�2*

lossŴ�;`�c       �	��afc�A�2*

loss��E<{<U�       �	�Y�afc�A�2*

loss� �;��[       �	���afc�A�2*

loss)�'=���       �	/��afc�A�2*

loss)��;�MՁ       �	S"�afc�A�2*

loss��;��~�       �	��afc�A�2*

loss� =���       �	mT�afc�A�2*

loss�� <A!�       �	���afc�A�2*

loss ��:��       �	(��afc�A�2*

loss$�=�"u�       �	H�afc�A�2*

lossfԓ<���E       �	d��afc�A�2*

loss��)<J��j       �	�B�afc�A�2*

loss��@:q��       �	V�afc�A�2*

loss��<���P       �	t��afc�A�2*

loss0p<����       �	���afc�A�2*

lossLڒ;'���       �	"3�afc�A�2*

lossʐy;��C       �	=��afc�A�2*

lossRq<����       �	�}�afc�A�3*

loss��>���       �	��afc�A�3*

loss��5;����       �	Z��afc�A�3*

loss	�9�U       �	��afc�A�3*

loss�l/<�S�       �	���afc�A�3*

loss!Ŗ:x�Bc       �	#��afc�A�3*

loss!~�8�_�       �	B$�afc�A�3*

loss|�;S���       �	���afc�A�3*

loss,4]:�ȯ�       �	ԁ�afc�A�3*

loss���;H|(�       �	$'�afc�A�3*

loss�]};��D-       �	C��afc�A�3*

lossM09��ͤ       �	f�afc�A�3*

lossi�Z:�k�       �	��afc�A�3*

loss�S�;�%��       �	H��afc�A�3*

lossL�Z7p� ?       �	�^�afc�A�3*

lossܑ�;Ú�       �	f��afc�A�3*

loss?�T:W&oJ       �	���afc�A�3*

loss�c>;��54       �	>�afc�A�3*

loss�w;�*6       �	���afc�A�3*

lossdI8Q¼       �	��afc�A�3*

loss 2�::�u�       �	E/�afc�A�3*

loss[��=�\w�       �	���afc�A�3*

loss��:v �       �	�~ bfc�A�3*

loss�Ad=uD.B       �	�bfc�A�3*

loss�m!<��z       �	��bfc�A�3*

loss�<YDϞ       �	��bfc�A�3*

lossK�;(�Y�       �	z�bfc�A�3*

loss�F�;��S       �	T7bfc�A�3*

lossL�=�/�l       �	
�bfc�A�3*

loss֚; b       �	�bfc�A�3*

loss��6;��       �	��bfc�A�3*

loss�W=B���       �	�>bfc�A�3*

loss1Z�:�vO�       �	U�bfc�A�3*

loss�>�<}9       �	1zbfc�A�3*

loss߿4<��\-       �		bfc�A�3*

lossX"�;��F       �	o�	bfc�A�3*

loss��<<&�       �	�[
bfc�A�3*

loss&��<a*��       �	 	bfc�A�3*

lossO<j�}       �	M�bfc�A�3*

loss�/�<\�       �	�5bfc�A�3*

lossꋒ<�|�c       �	K�bfc�A�3*

loss�:<�U��       �	Քbfc�A�3*

loss?�;#vp       �	�,bfc�A�3*

loss�
�;�4�       �	��bfc�A�3*

lossm8<�M�       �	Ǹbfc�A�3*

loss��;�r�       �	��bfc�A�3*

lossr�w;�)�S       �	�;bfc�A�3*

loss���<{H��       �	��bfc�A�3*

loss���<v        �	�bfc�A�3*

loss��<�]�B       �	�=bfc�A�3*

loss�6<<�<z       �	rnbfc�A�3*

loss��E=f~	{       �	Qbfc�A�3*

loss��;L��s       �	a�bfc�A�3*

loss䓥;�k�       �	-[bfc�A�3*

lossZ��:ܙUS       �	ebfc�A�3*

lossy�;�]W       �	�bfc�A�3*

lossAm><\��!       �	K9bfc�A�3*

lossT�+;�p{       �	��bfc�A�3*

lossA�<��       �	N{bfc�A�3*

loss��;��nH       �	�bfc�A�3*

loss���<F�9�       �	��bfc�A�3*

loss)Q�<�]�h       �	H�bfc�A�3*

lossj��;hI�       �	�(bfc�A�3*

loss�;4,�       �	��bfc�A�3*

loss 
�;�ƚ       �	-\bfc�A�3*

lossO�<�V<       �	c�bfc�A�3*

lossO5�;X�[       �	u�bfc�A�3*

loss��;�꼾       �	(bfc�A�3*

lossr��<�q�       �	Y�bfc�A�3*

loss�9<���       �	�W bfc�A�3*

loss�<B�W�       �	E� bfc�A�3*

lossF��:��α       �	��!bfc�A�3*

loss)V;8��k       �	�1"bfc�A�3*

lossM��;�Z��       �		5Bbfc�A�3*

loss�<_�A       �	�Bbfc�A�3*

lossl�;vx"�       �	��Cbfc�A�3*

loss�uo;L��b       �	��Dbfc�A�3*

loss��<��Cl       �	,DEbfc�A�3*

loss�b<+�n       �	c|Fbfc�A�3*

loss$��<��q       �	��Gbfc�A�3*

loss�=�(       �	�yHbfc�A�3*

lossJ;�;�I�       �	�dIbfc�A�3*

loss���;�V�       �	HJbfc�A�3*

loss�=�:3��       �	I�Jbfc�A�3*

lossx�=����       �	�Kbfc�A�3*

loss0ß<q�B       �	^�Lbfc�A�3*

lossʙ�<��y�       �	]�Mbfc�A�3*

loss̈́<��=�       �	��Nbfc�A�3*

loss�}<6�J       �	��Obfc�A�3*

loss���;5��&       �	S�Pbfc�A�3*

lossA�;B��       �	|`Qbfc�A�3*

loss�,�;���       �	�Rbfc�A�3*

loss�W�<Q`)       �	W�Rbfc�A�3*

loss�<��d       �	�Sbfc�A�3*

loss_�
=�"y�       �	��Tbfc�A�3*

loss�:;:��       �	+Vbfc�A�3*

lossML�<�6�       �		�Vbfc�A�3*

loss��b<k�H�       �	��Wbfc�A�3*

lossչ<�r�       �	�LXbfc�A�3*

loss�j<"�J�       �	�Ybfc�A�3*

loss��D<.��       �	9*Zbfc�A�3*

loss�B�==���       �	��Zbfc�A�3*

losslO�;(���       �	��[bfc�A�3*

loss�z�;��R�       �	A�\bfc�A�3*

loss��<;�p?       �	�g]bfc�A�3*

loss؀�9����       �	O�^bfc�A�3*

losst[)<�[گ       �	8_bfc�A�3*

loss���:Q��       �	�>`bfc�A�3*

loss�ܽ<����       �	��abfc�A�3*

loss��;AG7       �	��bbfc�A�3*

loss�~�<=�\�       �	%�cbfc�A�3*

loss^0�<��j       �	��dbfc�A�3*

loss�7<�#       �	ђebfc�A�3*

loss-z�<~�Ơ       �	�fbfc�A�3*

loss=�;}8y       �	'hbfc�A�3*

lossd�<A��7       �	��hbfc�A�3*

loss��,=�.;       �	�qibfc�A�3*

loss��O;�0S       �	�3jbfc�A�3*

loss@�<��(�       �	gkbfc�A�3*

loss�.)<p�1       �	��lbfc�A�3*

lossW��<�#`	       �	�imbfc�A�3*

loss�<q=y�       �	��nbfc�A�3*

loss�wW9	��d       �	�obfc�A�3*

lossh�<��Z       �	��pbfc�A�3*

loss�$�<��y�       �	�qbfc�A�3*

loss�<��t       �	��rbfc�A�3*

lossF5�<��*�       �	��sbfc�A�3*

loss��;c��m       �	�~tbfc�A�3*

lossWi�<�'<s       �	�&ubfc�A�4*

loss�";��ٖ       �	0�ubfc�A�4*

loss��49v��       �	��vbfc�A�4*

loss���<�Mr       �	N(wbfc�A�4*

loss<ZB<��C�       �	S�wbfc�A�4*

loss��v<��2       �	�oxbfc�A�4*

lossrb(;�ި!       �	�ybfc�A�4*

loss[�:PO��       �	�?zbfc�A�4*

loss���;�K?       �	��zbfc�A�4*

loss^F�;��|�       �	͒{bfc�A�4*

loss���:��\�       �	B>|bfc�A�4*

loss�rH=�n�       �	��|bfc�A�4*

lossؙr<c]u	       �	��}bfc�A�4*

loss�8�<��TD       �	#~bfc�A�4*

lossԐ�<~��       �	��~bfc�A�4*

loss��E<6���       �	�bbfc�A�4*

loss��;��.       �	͏�bfc�A�4*

loss*�:�#�'       �	�-�bfc�A�4*

loss�<WZ       �	}ʁbfc�A�4*

lossH�S;�G`�       �	+m�bfc�A�4*

loss��;�JD       �	�bfc�A�4*

lossH�:���       �	פ�bfc�A�4*

loss��E=YN*�       �	�:�bfc�A�4*

lossS��;�f�p       �	���bfc�A�4*

loss��;#��)       �	�|�bfc�A�4*

lossz�<��0u       �	��bfc�A�4*

loss�*�<�Zn�       �	2��bfc�A�4*

lossɰ�;�}�t       �	wN�bfc�A�4*

loss�|M<�!P       �	�bfc�A�4*

loss���:f�        �	l��bfc�A�4*

loss���;*b(�       �	�3�bfc�A�4*

lossaي:��eG       �	�̉bfc�A�4*

lossT�G=���       �	1��bfc�A�4*

loss*h#=�+i#       �	�~�bfc�A�4*

loss
��<�lY       �	��bfc�A�4*

loss��K<�8j       �	J��bfc�A�4*

loss].�<l��>       �	|�bfc�A�4*

lossc(<r���       �	���bfc�A�4*

loss��;^@��       �	V�bfc�A�4*

loss�=}�K�       �	<��bfc�A�4*

loss�b�;JL�       �	���bfc�A�4*

loss���<�_I�       �	�1�bfc�A�4*

loss�><�@ݩ       �	S͑bfc�A�4*

loss@� >,���       �	\q�bfc�A�4*

loss�*|:��s	       �	|
�bfc�A�4*

lossi��;��z�       �	���bfc�A�4*

loss@�<���       �	>A�bfc�A�4*

lossaƩ; ���       �	�ؔbfc�A�4*

loss��u;���       �	�z�bfc�A�4*

lossP�=u�C�       �	 �bfc�A�4*

loss���<�b�       �	:$�bfc�A�4*

lossI��:e�3R       �	E��bfc�A�4*

lossQl�;%+�       �	�a�bfc�A�4*

loss�3=�~       �	���bfc�A�4*

loss;+�;���$       �	ȗ�bfc�A�4*

loss���=c$��       �	u>�bfc�A�4*

loss�̐<�l/       �	wۚbfc�A�4*

loss��<&h�-       �	ux�bfc�A�4*

loss�\;:)�q       �	��bfc�A�4*

lossxo�;{4�       �	�ɜbfc�A�4*

loss.��8��N�       �	�h�bfc�A�4*

losssa;g��q       �	(�bfc�A�4*

loss�<Jv�       �	��bfc�A�4*

loss���<iPx       �	.S�bfc�A�4*

lossf\r<��|^       �	��bfc�A�4*

loss�I�;چJ(       �	�2�bfc�A�4*

lossT��<r9       �	͡bfc�A�4*

lossfm�;5Z-       �	�n�bfc�A�4*

loss�G<7�N�       �	��bfc�A�4*

loss�e�:y&|       �	���bfc�A�4*

lossMR =�YN�       �	��bfc�A�4*

loss�k�< Ě       �	�6�bfc�A�4*

loss�s�;�^0       �	Rץbfc�A�4*

loss�e<�x�>       �	Ts�bfc�A�4*

lossFa�<��c]       �	C�bfc�A�4*

loss>+;q�89       �	�çbfc�A�4*

lossX�<�U^[       �	h�bfc�A�4*

loss�P/=�O�       �	S�bfc�A�4*

lossO��;�       �	^��bfc�A�4*

lossvӚ<���H       �	�H�bfc�A�4*

loss$U�<	�E3       �	F��bfc�A�4*

loss��&=�ӣ$       �	���bfc�A�4*

lossa�:~D��       �	�"�bfc�A�4*

lossϮ9<B@       �	a��bfc�A�4*

loss}�2=��	�       �	<��bfc�A�4*

loss�:;�١       �	�ݮbfc�A�4*

lossVJT<�R
       �	w��bfc�A�4*

loss�_<��       �	W%�bfc�A�4*

lossJ��<,MŚ       �	RҰbfc�A�4*

loss6��<#       �	V�bfc�A�4*

loss�f�<�       �	V,�bfc�A�4*

lossa��<p��n       �	�Ѳbfc�A�4*

loss;�\<)��f       �	��bfc�A�4*

loss��v;9�u�       �	�ƴbfc�A�4*

loss��<�,��       �	�t�bfc�A�4*

loss��<o>w�       �	 ^�bfc�A�4*

loss�
|<���       �	81�bfc�A�4*

loss�<<�Z�I       �	bڷbfc�A�4*

lossO�#=Ƅ�4       �	�{�bfc�A�4*

loss��<߭�       �	���bfc�A�4*

loss�};A@!�       �	�,�bfc�A�4*

loss4��;�N�A       �	պbfc�A�4*

loss7=^��       �	4��bfc�A�4*

loss��=���H       �	�4�bfc�A�4*

loss�k<:�f       �	s۽bfc�A�4*

loss�u�<5v��       �	���bfc�A�4*

loss���:ݻ��       �	��bfc�A�4*

loss톃<��TN       �	O��bfc�A�4*

loss�*<��H       �	�n�bfc�A�4*

loss�=�z�       �	�bfc�A�4*

loss��+;�9�n       �	��bfc�A�4*

loss��:�-�q       �	fK�bfc�A�4*

loss��w=r�       �	���bfc�A�4*

lossb`:��v�       �	��bfc�A�4*

loss{�<���       �	�+�bfc�A�4*

loss�=v��       �	c��bfc�A�4*

loss�#8;_G       �	�r�bfc�A�4*

loss��;�       �	R�bfc�A�4*

loss�uw;�fj       �	:��bfc�A�4*

loss�t�;�fȹ       �	wL�bfc�A�4*

loss�n<�f��       �	x��bfc�A�4*

loss���<(Ь       �	���bfc�A�4*

lossZ�;U�6x       �	�%�bfc�A�4*

loss���;�_��       �	��bfc�A�4*

lossހ;C]7       �	Lp�bfc�A�4*

loss�r:����       �	��bfc�A�4*

loss��<��       �	_��bfc�A�4*

loss�<�<E�       �	R_�bfc�A�4*

loss!V;� �       �	��bfc�A�5*

loss��.<s[a4       �	6��bfc�A�5*

lossƇ�:YX�u       �	{J�bfc�A�5*

loss�6�<�A��       �	���bfc�A�5*

lossV\�<x%       �	n��bfc�A�5*

lossX�<���       �	+�bfc�A�5*

lossX<=	e�"       �	��bfc�A�5*

loss��[<3�i�       �	�]�bfc�A�5*

loss�8�<�y��       �	���bfc�A�5*

loss\�:;3i'       �	ђ�bfc�A�5*

lossiz�;�֙5       �	�'�bfc�A�5*

lossȢ�;�L�X       �	���bfc�A�5*

lossM5�=�Ͽ       �	*T�bfc�A�5*

loss��y<k��4       �	���bfc�A�5*

loss8�>l���       �	���bfc�A�5*

loss�U<43L�       �	$*�bfc�A�5*

loss�d�;�O�~       �	[��bfc�A�5*

loss���<>��       �	�p�bfc�A�5*

lossO� =�w�2       �	n�bfc�A�5*

loss@A�;Zk��       �	g��bfc�A�5*

loss!�><d�%f       �	�X�bfc�A�5*

lossL�W<����       �	��bfc�A�5*

loss;{:<dƟK       �	��bfc�A�5*

lossE<�e��       �	�7�bfc�A�5*

loss5��<��       �	��bfc�A�5*

loss���<)R�$       �	.r�bfc�A�5*

lossW>�<5�*       �	�	�bfc�A�5*

lossi�}=���       �	��bfc�A�5*

lossc<6�f�       �	 ;�bfc�A�5*

loss���;���P       �	���bfc�A�5*

loss8�;�W.A       �	}v�bfc�A�5*

loss��<�	,       �	��bfc�A�5*

loss�j.=�[��       �	���bfc�A�5*

loss4E=�/�       �	5C�bfc�A�5*

lossS˫<0�/       �	���bfc�A�5*

loss4�=�7#W       �	m�bfc�A�5*

loss��=	�(a       �	 �bfc�A�5*

loss�2,<�"�       �	V��bfc�A�5*

loss�c�<���%       �	<.�bfc�A�5*

lossא�<��Pz       �	���bfc�A�5*

lossrI�;�R-       �	�U�bfc�A�5*

lossۼ�<��F       �	C��bfc�A�5*

lossvVu;lkI�       �	��bfc�A�5*

lossW��<��Zh       �	YP�bfc�A�5*

loss�4�<�e       �	���bfc�A�5*

loss�O�;X���       �	x��bfc�A�5*

loss��F<dM��       �	A�bfc�A�5*

loss�]/;�Q��       �	@��bfc�A�5*

loss�A6;�J6w       �	)x�bfc�A�5*

loss&8�;S���       �	k�bfc�A�5*

loss�2T<�� 6       �	¡�bfc�A�5*

loss�j�;��(�       �	�F�bfc�A�5*

loss�S�<��       �	���bfc�A�5*

loss7�;^�ȼ       �	��bfc�A�5*

loss��C<�:t1       �	�bfc�A�5*

loss7�y<l�c�       �	��bfc�A�5*

loss�K�;cLZ�       �	wK�bfc�A�5*

lossf��;:�'       �	���bfc�A�5*

loss���;��ڥ       �	��bfc�A�5*

loss�7�<f��)       �	��bfc�A�5*

losst>�<���       �	2��bfc�A�5*

lossf�><$���       �	�D�bfc�A�5*

loss(�l<�g:�       �	e��bfc�A�5*

loss2t�;ԩ�       �	}�bfc�A�5*

loss1�<S)       �	U�bfc�A�5*

loss�Y<��'K       �	>��bfc�A�5*

loss��;ꍔ�       �	�P�bfc�A�5*

loss��=L�ڣ       �	��bfc�A�5*

loss�r�;�t�(       �	H��bfc�A�5*

loss���<���       �	�a�bfc�A�5*

loss)O; ��       �	X��bfc�A�5*

lossfU;;�K       �	-��bfc�A�5*

loss��<��#~       �	z6�bfc�A�5*

loss(`�<b���       �	���bfc�A�5*

loss@�@<s��       �	m�bfc�A�5*

loss���=����       �	d�bfc�A�5*

loss��f=��0~       �	���bfc�A�5*

loss�X/;DP��       �	q;�bfc�A�5*

loss���<Ė�       �	���bfc�A�5*

lossRU�;}�s:       �	x}�bfc�A�5*

loss���;�µ       �	u�bfc�A�5*

loss��-<k�vj       �	���bfc�A�5*

loss�i<1��1       �	�O�bfc�A�5*

lossiG<L�7�       �	���bfc�A�5*

loss�fl=G_�       �	^� cfc�A�5*

loss�_=���       �	�7cfc�A�5*

loss,ae<g\՜       �	��cfc�A�5*

loss��Q<5���       �	[acfc�A�5*

loss���<CJMz       �	�cfc�A�5*

loss`no<}h�       �	��cfc�A�5*

lossQč<	�4       �	�|cfc�A�5*

loss�(f=�tzv       �	�cfc�A�5*

loss)Gj<a�g       �	�cfc�A�5*

loss_:�<� n�       �	�Wcfc�A�5*

losstx�:]xt#       �	x�cfc�A�5*

lossMސ<��3�       �	7�cfc�A�5*

loss$d�<�zB�       �	R'cfc�A�5*

loss��;�*       �	��cfc�A�5*

loss�lO<�P�       �	Qk	cfc�A�5*

loss��(;=��       �	|
cfc�A�5*

loss�v�=2��       �	ޯ
cfc�A�5*

loss��=�>�       �	l^cfc�A�5*

loss��<���       �	��cfc�A�5*

lossoE�<���       �	��cfc�A�5*

loss�4k;T>ݝ       �	]5cfc�A�5*

loss��x<'�.�       �	:�cfc�A�5*

loss{V�;0[XD       �	jcfc�A�5*

loss}v*:��uo       �	Kcfc�A�5*

loss�į<3�u       �	��cfc�A�5*

lossq`;5���       �	78cfc�A�5*

loss ��:�P�:       �	�cfc�A�5*

losst�X;n��       �	mscfc�A�5*

lossgL�<א�       �	$cfc�A�5*

loss��&<���       �	��cfc�A�5*

loss�f,<���       �	�:cfc�A�5*

lossh
~=/>��       �	��cfc�A�5*

loss�<D<��}�       �	҉cfc�A�5*

lossD:C<y�0       �	�%cfc�A�5*

loss��;#t��       �	K�cfc�A�5*

loss�A<m��a       �	�^cfc�A�5*

lossc�F=����       �	��cfc�A�5*

loss���;+U�       �	��cfc�A�5*

loss��N=���*       �	�]cfc�A�5*

loss��;-���       �	R�cfc�A�5*

loss� �;8ı�       �	��cfc�A�5*

loss�<�W��       �	*;cfc�A�5*

loss���;Ջ_       �	L�cfc�A�5*

loss�6w;260�       �	cfc�A�5*

loss6��;v_	�       �	zcfc�A�6*

loss�}<kHs�       �	��cfc�A�6*

loss�4�;ѷM�       �	�Ncfc�A�6*

loss{�;���       �	M�cfc�A�6*

loss!�<@j"       �	_�cfc�A�6*

loss�q�:�&�       �	�4cfc�A�6*

loss �<���       �	�cfc�A�6*

lossU��<�%�       �	`w cfc�A�6*

losso6X="([       �	b!cfc�A�6*

lossS}�<�o�       �	��!cfc�A�6*

loss1�=<��!       �	{N"cfc�A�6*

lossf^�<،��       �	p]#cfc�A�6*

losswx�:��        �	��#cfc�A�6*

loss�<$L�q       �	3�$cfc�A�6*

lossQ�O<�g        �	%cfc�A�6*

lossؑ�<��:       �	��%cfc�A�6*

loss8L�<��F�       �	dX&cfc�A�6*

loss�<q�       �	�8'cfc�A�6*

loss?WL=�1X(       �	S�'cfc�A�6*

lossT�;sߝ       �	+k(cfc�A�6*

loss�P<'�1�       �	�)cfc�A�6*

loss�<�)1�       �	R�)cfc�A�6*

loss���<�,2       �	7*cfc�A�6*

loss�H=l�(U       �	��*cfc�A�6*

loss?�;�C�       �	ro+cfc�A�6*

loss��=�o�       �	l,cfc�A�6*

loss�e<�MX	       �	N�,cfc�A�6*

lossEج<�0ρ       �	�4-cfc�A�6*

losswL�:w�,�       �	��-cfc�A�6*

loss�Me:��@       �	��.cfc�A�6*

loss�:���J       �	� /cfc�A�6*

loss�߮=��:       �	h�/cfc�A�6*

loss!a<���       �	�b0cfc�A�6*

loss�-n=�@^�       �	+�0cfc�A�6*

lossF~*=]y        �	ސ1cfc�A�6*

loss�"�<O�³       �	�*2cfc�A�6*

loss{ߙ<9�       �	7�2cfc�A�6*

loss�S�;�㐴       �	�\3cfc�A�6*

loss�7<�N�5       �	$4cfc�A�6*

lossM��;�h��       �	��4cfc�A�6*

lossܼw<w,;6       �	0�5cfc�A�6*

loss�(B=�3��       �	a86cfc�A�6*

loss��]<CZ       �	�s7cfc�A�6*

loss�h�<�?|S       �	/8cfc�A�6*

loss�/M<���       �	�"9cfc�A�6*

lossET<A�?�       �	c:cfc�A�6*

loss�g�;+޼       �	'2;cfc�A�6*

loss	��<����       �	g�;cfc�A�6*

lossW�9;���-       �	��<cfc�A�6*

loss��p;p�'#       �	B�=cfc�A�6*

lossR�F;�j��       �	��>cfc�A�6*

loss���=:�֙       �	��?cfc�A�6*

loss�z=��m�       �	�6@cfc�A�6*

loss��<�m       �	KAcfc�A�6*

loss��<��8       �	%�Acfc�A�6*

loss��'<��e�       �	e�Bcfc�A�6*

loss��f<��4�       �	3lCcfc�A�6*

lossWZ<� �P       �	ϡDcfc�A�6*

loss���<s���       �	�BEcfc�A�6*

loss�w�;��f	       �	�Fcfc�A�6*

lossm��;�E	`       �	�>Gcfc�A�6*

loss�<^<�       �	�vHcfc�A�6*

loss}�i<I       �	�,Icfc�A�6*

loss�v�;:Gk�       �	W�Icfc�A�6*

loss��<Z��        �	�xJcfc�A�6*

loss�5�;���O       �	�Kcfc�A�6*

loss(/�<jxK       �	ѯKcfc�A�6*

loss��)<�g�       �	�ELcfc�A�6*

loss�g�<��6       �	�Lcfc�A�6*

loss�m;�5"�       �	��Mcfc�A�6*

loss���<�W�]       �	�ENcfc�A�6*

lossJ��;0r�       �	��Ncfc�A�6*

loss�y <��g       �	�~Ocfc�A�6*

lossH�:�YT       �	)"Pcfc�A�6*

lossW<�
�H       �	��Pcfc�A�6*

lossμ�;Ϯ�y       �	:ZQcfc�A�6*

lossm,�<0Uy�       �	�Rcfc�A�6*

loss�?::�'_8       �	ԷRcfc�A�6*

lossX/4;��       �	�]Scfc�A�6*

loss[E�;B�       �	/�Scfc�A�6*

loss�0<j��       �	��Tcfc�A�6*

loss2;�	��       �	�Ucfc�A�6*

lossWD	<��m�       �	�#Vcfc�A�6*

loss��<����       �	E�Vcfc�A�6*

loss!��;DS.       �	/NWcfc�A�6*

loss�#D:9n       �	�Wcfc�A�6*

loss�:�(kY       �	ӅXcfc�A�6*

loss7�:���       �	�Ycfc�A�6*

loss}Q�;�+1�       �	��Ycfc�A�6*

loss[�0;�$       �	�NZcfc�A�6*

loss���;C*��       �	��Zcfc�A�6*

loss�$8<��<�       �	�[cfc�A�6*

loss�a�;F��'       �	�\cfc�A�6*

loss���<;��~       �	>�\cfc�A�6*

loss17�;�`��       �	9C]cfc�A�6*

loss�0h<:�|�       �	=�]cfc�A�6*

lossɥa<B3�       �	am^cfc�A�6*

lossm��<��n       �	\_cfc�A�6*

loss��u;&��       �	M�_cfc�A�6*

loss�Z�<�       �	6:`cfc�A�6*

loss��Z=7��       �	��`cfc�A�6*

loss\�=O(�       �	�gacfc�A�6*

loss]r�<)_��       �	�Qbcfc�A�6*

loss�A�<L'x       �	x�bcfc�A�6*

loss:Lp=�c��       �	߈ccfc�A�6*

loss��9���       �	&dcfc�A�6*

loss�P�:����       �	��dcfc�A�6*

loss�o<e_F�       �	:Wecfc�A�6*

loss';*��       �	��ecfc�A�6*

loss֏<?\�5       �	T�fcfc�A�6*

lossF�=����       �	�(gcfc�A�6*

loss�k`;9�_�       �	¿gcfc�A�6*

loss؃=u�M       �	�Zhcfc�A�6*

loss���<�0|u       �	��hcfc�A�6*

loss���<A_߾       �	��icfc�A�6*

lossM��<$�x�       �	d#jcfc�A�6*

loss*��<N�Eg       �	��jcfc�A�6*

loss�(U<���       �	�[kcfc�A�6*

loss��;�MV       �	�kcfc�A�6*

lossR=�m�       �	'�lcfc�A�6*

loss�}�;p�7�       �	�mcfc�A�6*

loss(�+; �h4       �	֬mcfc�A�6*

lossO��;Le�B       �	�Dncfc�A�6*

loss_��;}/�       �	o�ncfc�A�6*

loss�E<X,vM       �	�oocfc�A�6*

loss��;��       �	�pcfc�A�6*

loss��<��-9       �	��pcfc�A�6*

loss�"&;tR~       �	�8qcfc�A�6*

losse�%;�b�       �	Q�qcfc�A�7*

lossr[�<��l;       �	�prcfc�A�7*

loss)>B=��       �	oscfc�A�7*

loss�;8<bTuo       �	?tcfc�A�7*

lossH�<���       �	�ucfc�A�7*

loss�U+<����       �	�vcfc�A�7*

loss|��;���       �	4�vcfc�A�7*

loss���<��h�       �	kcwcfc�A�7*

lossiG=P6(�       �	1xcfc�A�7*

loss�Uh<�!�3       �	��xcfc�A�7*

lossR�?;���       �	/oycfc�A�7*

loss��e=[��       �	�zcfc�A�7*

loss{�};;��       �	��zcfc�A�7*

loss�2�;9��       �	�{cfc�A�7*

loss���:k�A       �	*|cfc�A�7*

loss�8<���u       �	��|cfc�A�7*

loss���<85~�       �	�l}cfc�A�7*

loss���<��'�       �	#~cfc�A�7*

lossP<�
�       �	p�~cfc�A�7*

loss�@�;�^       �	�Scfc�A�7*

lossm�<#���       �	��cfc�A�7*

lossDA�<�\�       �	.��cfc�A�7*

loss�;|ؖ�       �	,)�cfc�A�7*

loss�*e;��z"       �	�ȁcfc�A�7*

loss��O<��i       �	�r�cfc�A�7*

lossh�=��9h       �	��cfc�A�7*

lossH��;UF       �	���cfc�A�7*

loss��<Էa{       �	�-�cfc�A�7*

loss�O,<~ƨ       �	�Ņcfc�A�7*

losso�<V1�       �	l[�cfc�A�7*

lossc�=�"s       �	��cfc�A�7*

loss ��<�C	;       �	�ԇcfc�A�7*

lossw(;`�)       �	�o�cfc�A�7*

loss�V/<滗�       �	SX�cfc�A�7*

loss���=D�e�       �		��cfc�A�7*

loss�]Q<$�x�       �	��cfc�A�7*

loss(�<|d E       �	a4�cfc�A�7*

loss��=^��       �	+܋cfc�A�7*

loss��Y;PT�+       �	x�cfc�A�7*

lossXv9<����       �	��cfc�A�7*

lossW��<ځ�       �	F��cfc�A�7*

loss �=��"       �	�b�cfc�A�7*

lossl1�<7*.       �	�A�cfc�A�7*

loss�#;y�       �	��cfc�A�7*

loss��<��y�       �	��cfc�A�7*

loss��<{�pP       �	�)�cfc�A�7*

loss���<B|��       �	�"�cfc�A�7*

loss��;�G��       �	�͓cfc�A�7*

loss�P!=�5Q�       �	���cfc�A�7*

loss}9'<|�p:       �	�.�cfc�A�7*

loss���=���       �	
֕cfc�A�7*

lossvr�<S��p       �	��cfc�A�7*

loss�U�<�Ȓ�       �	)�cfc�A�7*

loss��;���       �	Tɗcfc�A�7*

loss��;��-       �	Hp�cfc�A�7*

loss�Y=�^�V       �	��cfc�A�7*

lossz+�<ʜ�       �	��cfc�A�7*

loss�Jl<���       �	h[�cfc�A�7*

loss��;z<�       �	��cfc�A�7*

lossz�@<���-       �	`�cfc�A�7*

loss,1�<2��       �	$��cfc�A�7*

loss�fO=H`�       �	�S�cfc�A�7*

loss.ρ;:7l       �	��cfc�A�7*

loss��=h6*H       �	E��cfc�A�7*

loss� K<j�7�       �	['�cfc�A�7*

lossv��<l0       �	���cfc�A�7*

loss��O<�ͩ       �	ke�cfc�A�7*

loss�a<���r       �	��cfc�A�7*

loss�%�;��H�       �	O��cfc�A�7*

loss��F<:��       �	6V�cfc�A�7*

lossh��<�]�i       �	s��cfc�A�7*

lossZ�"<�~��       �	���cfc�A�7*

lossmf*=A�|       �	a�cfc�A�7*

loss�(&;�e�>       �	��cfc�A�7*

lossy��<�u�       �	&ƥcfc�A�7*

lossJ�q</ڛ�       �	�h�cfc�A�7*

loss�5�<�y�       �	��cfc�A�7*

lossEg<���       �	O��cfc�A�7*

loss}�<;K�i`       �	\�cfc�A�7*

lossv��<!ɦ       �	U��cfc�A�7*

loss:��<�|�	       �	��cfc�A�7*

loss2�;�9�       �	�1�cfc�A�7*

lossQ�<irp       �	�Ѫcfc�A�7*

loss(Ϯ:N��U       �	k�cfc�A�7*

loss�*�;��       �	A�cfc�A�7*

loss$Y=<����       �	���cfc�A�7*

loss�@;#�~p       �	�[�cfc�A�7*

loss{�;QL��       �	�cfc�A�7*

loss�(�<<���       �	2��cfc�A�7*

lossž�<D�6�       �	vS�cfc�A�7*

loss��;_�u�       �	3��cfc�A�7*

loss/m!;��z�       �	���cfc�A�7*

lossF�8<1��
       �	�Y�cfc�A�7*

losshT;Of�       �	�
�cfc�A�7*

lossO';�X̕       �	���cfc�A�7*

loss�:?;��U"       �	�O�cfc�A�7*

lossա;���       �	��cfc�A�7*

lossEp<<1�6       �	?��cfc�A�7*

loss}�<2��q       �	0�cfc�A�7*

loss,��9q#8       �	^صcfc�A�7*

loss#�;�lk       �	|{�cfc�A�7*

loss/�;�`�+       �	
.�cfc�A�7*

loss�K�7'S�       �	cԷcfc�A�7*

loss�>9���       �	J}�cfc�A�7*

loss���9b�\       �	&�cfc�A�7*

loss�7t;)�7       �	��cfc�A�7*

loss�(8;�.A�       �	|a�cfc�A�7*

loss��9���$       �	��cfc�A�7*

loss��@;t�a�       �	���cfc�A�7*

loss�rO=���       �	GX�cfc�A�7*

loss��:��       �	���cfc�A�7*

loss��_<�Ql       �	T��cfc�A�7*

loss,1�<�	�       �	�O�cfc�A�7*

loss��<���       �	?��cfc�A�7*

loss�j�;
'�I       �	���cfc�A�7*

loss�U�;�}�       �	�>�cfc�A�7*

loss�`}=�,�       �	>��cfc�A�7*

lossz̏<���       �	'��cfc�A�7*

loss`�;*�ǆ       �	 �cfc�A�7*

loss18<QI:y       �	��cfc�A�7*

loss�<K5�o       �	�^�cfc�A�7*

lossd9;�g�       �	��cfc�A�7*

lossf�<��N       �	D��cfc�A�7*

loss$7�;˕�       �	oJ�cfc�A�7*

lossx�
=��=       �	���cfc�A�7*

loss�"�;���$       �	n��cfc�A�7*

loss���<֫�\       �	�#�cfc�A�7*

loss�5�;t(�}       �	r��cfc�A�7*

lossC><"P�       �	�U�cfc�A�8*

loss��x< ���       �	���cfc�A�8*

loss-v:q���       �	E��cfc�A�8*

loss��;�ɖ�       �	z�cfc�A�8*

loss���:���       �	���cfc�A�8*

loss�&;���       �	nO�cfc�A�8*

loss�ET;�٤l       �	��cfc�A�8*

loss�jT:x^EC       �	��cfc�A�8*

loss;�<���v       �	|)�cfc�A�8*

loss&��;j�m       �	.��cfc�A�8*

loss�=\;#�       �	�b�cfc�A�8*

loss.P�<���       �	~��cfc�A�8*

lossX��;�+��       �	5��cfc�A�8*

lossѤ;��c       �	�4�cfc�A�8*

lossW��:��0�       �	p��cfc�A�8*

loss��:�<       �	Gw�cfc�A�8*

loss��;��G�       �	��cfc�A�8*

lossw1);���       �	��cfc�A�8*

loss�ݵ<|Y�       �	�G�cfc�A�8*

loss��<���       �	���cfc�A�8*

lossс�<�X��       �	�w�cfc�A�8*

loss<�:��
K       �	$�cfc�A�8*

loss�d';�k�u       �	��cfc�A�8*

loss�&=z)t@       �	�U�cfc�A�8*

lossa��<q[�       �	���cfc�A�8*

loss͜�<�\�g       �	��cfc�A�8*

lossʎ�:;R�       �	��cfc�A�8*

lossؠL<EQf�       �	,��cfc�A�8*

loss��)=6 y7       �	aS�cfc�A�8*

losstƏ<���       �	Ic�cfc�A�8*

loss�Oo<��K       �	6�cfc�A�8*

lossXQ:�լ�       �	N��cfc�A�8*

lossV��:�OE       �	]4�cfc�A�8*

loss4�|<O%@�