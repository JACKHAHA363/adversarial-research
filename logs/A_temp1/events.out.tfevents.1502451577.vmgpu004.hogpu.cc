       �K"	  @�ec�Abrain.Event:2i�C1�     $�	&�x�ec�A"��
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
 *�x=*
_output_shapes
: *
dtype0
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2ևa*
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
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2��"*
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
shrink_axis_mask *
ellipsis_mask *

begin_mask *
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
:����������*
seed2��q
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
_output_shapes
:	�
*
seed2��c*
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
:���������@*
seed2���
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask *
T0*
Index0*
end_mask*
_output_shapes
:*
ellipsis_mask *

begin_mask 
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
seed2��R
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
ToFloatCastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

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
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
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
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: 
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: 
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
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
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
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
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:���������*
T0
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
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
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
valueB *
_output_shapes
: *
dtype0
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
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: 
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
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_1_grad/Reshape*'
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
valueB@@*    *&
_output_shapes
:@@*
dtype0
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
 *w�+2*
_output_shapes
: *
dtype0
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"��i���     �S�	�*|�ec�AJ��
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
seed2ևa
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
:���������@*
seed2��"*
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
seed2���
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
 *  �?*
dtype0*
_output_shapes
: 
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��q*
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
seed2��c
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
:���������@*
seed2���*
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
shrink_axis_mask *
ellipsis_mask *

begin_mask *
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
:����������*
seed2��R
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
 *  �?*
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
valueB *
_output_shapes
: *
dtype0
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
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
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
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
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
Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
valueB *
_output_shapes
: *
dtype0
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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
_output_shapes
: 
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0	�,       ��-	����ec�A*

loss�@}���       ��-	�N��ec�A*

loss=c@���       ��-	����ec�A*

lossO@�{1       ��-	)ϭ�ec�A*

loss���?�Jޱ       ��-	�r��ec�A*

lossn��?�wH       ��-	���ec�A*

loss�.�?�,�       ��-	���ec�A*

loss���?U��       ��-	�I��ec�A*

loss�?EdT2       ��-	���ec�A	*

lossJ�l?-ր�       ��-	x&��ec�A
*

loss=��?AΰW       ��-	�Ʋ�ec�A*

lossħ6?��       ��-	�\��ec�A*

loss;DG?�\��       ��-	����ec�A*

loss�S?�_�C       ��-	mT��ec�A*

loss��-?{L2e       ��-	W��ec�A*

loss)kE?��eu       ��-	����ec�A*

lossi��>r�$       ��-	_)��ec�A*

loss�K?v�X�       ��-	����ec�A*

loss3s�>����       ��-	kc��ec�A*

loss� ?LX�<       ��-	���ec�A*

lossa)?'~�$       ��-	|���ec�A*

loss,�*?~�p       ��-	_B��ec�A*

loss~?�0-�       ��-	<ܺ�ec�A*

loss)0?��_       ��-	ur��ec�A*

lossl�?a.A       ��-	0��ec�A*

loss�=�>�?nD       ��-	����ec�A*

loss��>�x�R       ��-	fJ��ec�A*

loss��?Er�/       ��-	���ec�A*

lossq�&?�	�i       ��-	����ec�A*

loss�?���       ��-	a��ec�A*

loss[��>rߚ�       ��-	:���ec�A*

loss��>�/T
       ��-	KV��ec�A *

loss���>�s�y       ��-	���ec�A!*

loss?CB�       ��-	���ec�A"*

loss�
�>m��       ��-	����ec�A#*

loss��>N]�L       ��-	G��ec�A$*

lossLĝ>�.�?       ��-	]���ec�A%*

loss͎?�3��       ��-	B"��ec�A&*

loss�d?O>2       ��-	����ec�A'*

loss��?��w	       ��-	8e��ec�A(*

loss3�>Fj��       ��-	���ec�A)*

lossA)�>�@�       ��-	����ec�A**

loss���>��       ��-	T7��ec�A+*

loss�ָ>�T�2       ��-	����ec�A,*

loss�E�>st6U       ��-	?p��ec�A-*

loss��?�q`�       ��-	t��ec�A.*

loss\��>��>�       ��-	h���ec�A/*

loss�>��о       ��-	�K��ec�A0*

loss���>m�       ��-	����ec�A1*

lossQ�_>�A��       ��-	�|��ec�A2*

loss��;>2͸       ��-	��ec�A3*

lossX��>E��       ��-	1��ec�A4*

loss�f>�Y�>       ��-	5���ec�A5*

loss�"v?�0�       ��-	o~��ec�A6*

loss�Cb>h�6�       ��-	�$��ec�A7*

loss�>>��&       ��-	@���ec�A8*

loss!�6>�@i/       ��-	�^��ec�A9*

lossA�B>��S�       ��-	���ec�A:*

loss`��>�"��       ��-	i���ec�A;*

loss���>��U�       ��-	�N��ec�A<*

loss���>?5��       ��-	����ec�A=*

loss�no>��
       ��-	{���ec�A>*

loss��V>)<�c       ��-	�L��ec�A?*

loss��>�f�X       ��-	����ec�A@*

lossTТ>����       ��-	Q���ec�AA*

lossL(C>��
       ��-	�K��ec�AB*

loss���>(/��       ��-	(���ec�AC*

loss�d�>s��       ��-	G���ec�AD*

loss�p?d%i�       ��-	�1��ec�AE*

loss�
b>�N��       ��-	���ec�AF*

lossdhP>��?       ��-	{���ec�AG*

loss �>s�       ��-	�0��ec�AH*

losssbc>z�;�       ��-	f���ec�AI*

loss}ʫ>{��r       ��-	���ec�AJ*

loss�h�>-\       ��-	�%��ec�AK*

loss���>'N       ��-	����ec�AL*

loss�o?�0f       ��-	b��ec�AM*

loss6o?�s�s       ��-	��ec�AN*

loss��->g��       ��-	���ec�AO*

loss֎> 0e8       ��-	MH��ec�AP*

loss��>��f�       ��-	z���ec�AQ*

loss���>�I��       ��-	ly��ec�AR*

loss,�>�F��       ��-	%��ec�AS*

loss}>x�t�       ��-	'���ec�AT*

lossW\�>����       ��-	�l��ec�AU*

loss�+�>��Z�       ��-	(��ec�AV*

loss�x>�%��       ��-	����ec�AW*

loss)��>�(x       ��-	�[��ec�AX*

loss�
?��Ch       ��-	�	��ec�AY*

loss��X> ��       ��-	����ec�AZ*

lossm��>�Ux!       ��-	�P��ec�A[*

loss��X>��E       ��-	����ec�A\*

lossnc>��Ne       ��-	����ec�A]*

loss��>gՕ       ��-	����ec�A^*

loss�k�>6���       ��-	�X��ec�A_*

loss]�d>�V��       ��-	����ec�A`*

lossAC�>1 �n       ��-	���ec�Aa*

lossd}�>�h       ��-	����ec�Ab*

loss[��>q�~R       ��-	�j��ec�Ac*

loss3�~>�|��       ��-	c��ec�Ad*

loss�|>H.��       ��-	J���ec�Ae*

lossA	>!���       ��-	X��ec�Af*

loss<��>Wl�       ��-	"���ec�Ag*

loss�>��\       ��-	�B��ec�Ah*

loss�6�>��!p       ��-	����ec�Ai*

loss�v�>̀G�       ��-	���ec�Aj*

lossc�=j�$s       ��-	]7��ec�Ak*

loss{�=�N��       ��-	Y���ec�Al*

loss��>��wc       ��-	���ec�Am*

lossntz>U��       ��-	D��ec�An*

lossfj�>��       ��-	����ec�Ao*

loss3�>ٜ�       ��-	���ec�Ap*

loss{!<>C!$       ��-	�-��ec�Aq*

lossfDb=�ZS       ��-	����ec�Ar*

loss$O>s�n�       ��-	=d��ec�As*

lossR�&>x�e       ��-	����ec�At*

lossq�'>j�       ��-	 ���ec�Au*

lossfo�>_���       ��-	+2��ec�Av*

loss�>_>>c,       ��-	����ec�Aw*

lossv�/>�ҩ%       ��-	^f��ec�Ax*

loss��x>�7�       ��-	 ��ec�Ay*

loss�>�ר       ��-	����ec�Az*

loss���>@��@       ��-	�9��ec�A{*

loss�iR>v\.       ��-	w���ec�A|*

loss��Z=1Q+       ��-	]l��ec�A}*

lossx�?>JH�'       ��-	,��ec�A~*

loss��l>���       ��-	;���ec�A*

losss&�=#"s�       �	�@ �ec�A�*

lossch�>jU�       �	�L�ec�A�*

losse�>+��a       �	�Q�ec�A�*

lossf�f>��g       �	��ec�A�*

loss �H>���+       �	Զ�ec�A�*

lossd�>nҨ       �	1\�ec�A�*

lossi�=R�u       �	q�ec�A�*

loss��>�M��       �	 ��ec�A�*

loss_�,>��W       �	i�ec�A�*

loss�\V>@=�       �	��ec�A�*

lossȩ�=���	       �	s��ec�A�*

loss1->G�#G       �	�b	�ec�A�*

loss���=�b       �	
�ec�A�*

loss@K�=Am�       �	!�ec�A�*

lossR��=�V       �	��ec�A�*

loss�J[>H�8�       �	�;�ec�A�*

loss���=����       �	���ec�A�*

loss	�d>HP�T       �	�y�ec�A�*

loss6W4>����       �	�H�ec�A�*

loss[�[>��^x       �	���ec�A�*

lossXR�=�M�       �	�}�ec�A�*

loss�&�=� ��       �	�ec�A�*

lossC�>N��l       �	i��ec�A�*

loss� >}�[       �	S@�ec�A�*

loss�,h>��C       �	;��ec�A�*

loss�C�>]��&       �	��ec�A�*

lossv�>�T�       �	2 �ec�A�*

loss��)>�+�7       �	���ec�A�*

loss0�=�m       �	l[�ec�A�*

loss�J�=�b@�       �	���ec�A�*

lossQIE>>��       �	ٙ�ec�A�*

lossS9>��K�       �	d?�ec�A�*

lossKp>P��       �	���ec�A�*

loss�?}>�J2M       �	��ec�A�*

lossM��=��jJ       �	��ec�A�*

lossQ0>��       �	o��ec�A�*

lossl�=uR��       �	Z�ec�A�*

loss7 >vQi       �	��ec�A�*

lossa��=r�       �	͏�ec�A�*

losslt�=0���       �	W'�ec�A�*

loss�9A>e�       �	��ec�A�*

loss���=�̮#       �	=c�ec�A�*

lossja=��y"       �	\ �ec�A�*

loss�P�=��       �	���ec�A�*

lossj�c=�pX       �	5�ec�A�*

loss\�>Rw�       �	O��ec�A�*

lossYr>���\       �	d��ec�A�*

loss\si=x��T       �	�& �ec�A�*

lossM
>恫]       �	� �ec�A�*

loss6�>�@y�       �	�R!�ec�A�*

lossx�>�-U�       �	-�!�ec�A�*

loss���=DRV�       �	I�"�ec�A�*

loss�>�=m �       �	�"#�ec�A�*

lossC΂=+ �       �	��#�ec�A�*

loss��+>z{	�       �	�i$�ec�A�*

loss��>>n�޿       �	�%�ec�A�*

loss)u=�8�       �	2�%�ec�A�*

loss �>�E��       �	UO&�ec�A�*

loss,�>�U_       �	��&�ec�A�*

lossۊ=��D�       �	k�'�ec�A�*

loss�>���       �	2 (�ec�A�*

loss=1h��       �	A�(�ec�A�*

loss!A>Tր�       �	B\)�ec�A�*

loss=�h>��Β       �	� *�ec�A�*

lossi�T>�kp�       �	�*�ec�A�*

lossZ'�=l =T       �	�0+�ec�A�*

loss �>��_D       �	�.,�ec�A�*

loss��<��       �	wg-�ec�A�*

lossEd�=�P��       �	�S.�ec�A�*

losso�<>�PL+       �	G�.�ec�A�*

loss��=-Ի       �	k~/�ec�A�*

loss`im>+��       �	�0�ec�A�*

loss���=@�Â       �	�1�ec�A�*

loss��`>���       �	�|2�ec�A�*

lossZ��=�L�o       �	�!3�ec�A�*

loss=t�=%�^�       �	��3�ec�A�*

loss�>�r�@       �	�4�ec�A�*

loss�V<>2       �	��5�ec�A�*

lossҗ=V��Z       �	�e6�ec�A�*

lossε>�4�       �	-7�ec�A�*

lossƕ�=ێ�       �	��7�ec�A�*

loss�X�=�?       �	D�8�ec�A�*

lossl$#>��(a       �	9�9�ec�A�*

loss�c�=��       �	a:�ec�A�*

loss.M�=�`u       �	�;�ec�A�*

loss�2�=��D       �	��;�ec�A�*

loss�Tx=��       �	��<�ec�A�*

loss�d$>.H�#       �	�=�ec�A�*

loss45>�:��       �	]>�ec�A�*

loss6�>�rJ�       �	b?�ec�A�*

loss���>|���       �	��?�ec�A�*

lossv�x>����       �	��@�ec�A�*

loss�v�>�e�t       �	?�A�ec�A�*

loss�"�=����       �	��B�ec�A�*

loss�k>�w+�       �	i�C�ec�A�*

loss��>J�#       �	��D�ec�A�*

lossn?Y>.�q       �	�bE�ec�A�*

loss�2�=���U       �	�F�ec�A�*

loss!�6>�Μ       �	؛F�ec�A�*

losskk>�Z[Z       �	s.G�ec�A�*

lossK�>�HG}       �	v�G�ec�A�*

loss2�=>�\       �	mVH�ec�A�*

loss�)>C�=�       �	��H�ec�A�*

losss\>�ߦ�       �	�I�ec�A�*

loss���=I��       �	 J�ec�A�*

loss�F�>ȋ�}       �	z�J�ec�A�*

lossc>�Q�M       �	\XK�ec�A�*

lossϛ.>Rv��       �	F�K�ec�A�*

loss���=��S�       �	9~L�ec�A�*

lossc�">聓�       �	�M�ec�A�*

loss�g>.��w       �	"�M�ec�A�*

lossqoI>�J�g       �	�O�ec�A�*

loss@1�=Y��       �	��O�ec�A�*

loss���=ZT�:       �	�BP�ec�A�*

loss���=�E\�       �	��P�ec�A�*

loss��5>h�k       �	'jQ�ec�A�*

lossM��=�d       �	:R�ec�A�*

loss�j$>,       �	�R�ec�A�*

loss��=����       �	�2S�ec�A�*

lossaE�=�4�}       �	��S�ec�A�*

loss���=Vh�       �	�cT�ec�A�*

loss=��>m�3�       �	��T�ec�A�*

loss5�=����       �	�U�ec�A�*

loss���=���       �	x'V�ec�A�*

loss�yL>�=?�       �	@4W�ec�A�*

lossA>�ނ       �	��W�ec�A�*

loss�_T=�>v;       �	�UY�ec�A�*

losst O=0ZI        �	��Y�ec�A�*

loss[R�=�#�       �	/�Z�ec�A�*

loss���=�wp�       �	IL[�ec�A�*

loss�j>�D�       �	!\�ec�A�*

loss�.�=��K       �	T�\�ec�A�*

loss1�R>#�8       �	T�]�ec�A�*

loss�|�=S�s�       �	~7^�ec�A�*

loss�Vs=/�       �	I�^�ec�A�*

loss@�2>����       �	��_�ec�A�*

lossO!x=�֋�       �	[^`�ec�A�*

loss}��>7���       �	��`�ec�A�*

loss�I�=��?       �	|�a�ec�A�*

lossj>_h�       �	�<b�ec�A�*

loss�i>P�A�       �	�b�ec�A�*

loss��>�Ee3       �	�sc�ec�A�*

loss47>����       �	�d�ec�A�*

lossڄ�=Kc�i       �	@�d�ec�A�*

loss}��=V�L       �	lCe�ec�A�*

lossi�>B���       �	��e�ec�A�*

loss�G>��T�       �	}f�ec�A�*

loss��>WG�       �	�g�ec�A�*

loss���>ya       �	V�g�ec�A�*

loss���=�̏       �	��h�ec�A�*

loss�Ϥ=GO�R       �	��i�ec�A�*

lossH�B>}ѳP       �	f2j�ec�A�*

loss>>;�O�       �	��j�ec�A�*

loss��=[4��       �	dk�ec�A�*

loss��q=v�)       �	�(l�ec�A�*

loss�Ӵ=�Dx�       �	��l�ec�A�*

lossjg�=8���       �	0)n�ec�A�*

loss�p>�|?�       �	&�n�ec�A�*

loss_��=;�~�       �	��o�ec�A�*

losso�=���       �	�ep�ec�A�*

loss�{�=�{��       �	Ԟq�ec�A�*

loss�>>p'x       �	��r�ec�A�*

lossڢ<	���       �	Ws�ec�A�*

loss�& =��s       �	� t�ec�A�*

lossOt�=�3       �	��t�ec�A�*

loss�rr=���J       �	�Xu�ec�A�*

lossM��=��p�       �	A�u�ec�A�*

loss@��<��u�       �	Y�v�ec�A�*

loss��`>»7�       �	�$w�ec�A�*

loss���=�=d       �	��w�ec�A�*

lossR��=��G�       �	5]x�ec�A�*

loss�>���B       �	�x�ec�A�*

loss�I>ط��       �	}�y�ec�A�*

loss��=亱p       �	�)z�ec�A�*

loss�6�=�.52       �	��z�ec�A�*

loss[��='�       �	�{�ec�A�*

lossF��=����       �	�#|�ec�A�*

lossT�=)�NU       �	�|�ec�A�*

loss*nA>Fґ       �	[}�ec�A�*

loss���<��       �	+�}�ec�A�*

loss�r�=�xw       �	��~�ec�A�*

loss��=�       �	j�ec�A�*

loss:4>���       �	���ec�A�*

loss���= �MQ       �	����ec�A�*

lossNu�=���       �	;6��ec�A�*

loss��=
�w�       �	�Ӂ�ec�A�*

loss{ �=����       �	�x��ec�A�*

loss���<3���       �	i��ec�A�*

loss�N%>^�;�       �	}���ec�A�*

loss?�9=$�!       �	�F��ec�A�*

lossv�-=~��       �	�܄�ec�A�*

loss1�\>��T�       �	dv��ec�A�*

loss#|�=�       �	4��ec�A�*

loss�� =�A\       �	���ec�A�*

lossowe>��l       �	6?��ec�A�*

loss��=�       �	v��ec�A�*

loss���=���       �	����ec�A�*

loss,&+>�{_�       �	 ŉ�ec�A�*

lossV��>����       �	D���ec�A�*

lossld�=I���       �	�&��ec�A�*

loss�ٴ=.Y       �	�ȋ�ec�A�*

loss�	>����       �	�f��ec�A�*

loss��F>N�P       �	h	��ec�A�*

loss��>l~�       �	ʥ��ec�A�*

loss��=М��       �	�)��ec�A�*

loss�M�=+��       �	�Ə�ec�A�*

lossn�=7^�~       �	Dm��ec�A�*

loss�J>�*B       �	���ec�A�*

lossř�=Ϗ~k       �	ɑ�ec�A�*

loss��#>jm       �	�h��ec�A�*

loss?{>,�#       �	1��ec�A�*

loss�M�=�S       �	���ec�A�*

loss�D�<�qF�       �	p@��ec�A�*

loss_ �=���       �	�ޔ�ec�A�*

loss���=�1��       �	Sx��ec�A�*

loss��>� G�       �	���ec�A�*

loss�t>�U�       �	����ec�A�*

loss&/�>�oݙ       �	�k��ec�A�*

loss*1>j�?       �	:��ec�A�*

loss���=
�V�       �	8���ec�A�*

lossx@B>w���       �	�8��ec�A�*

lossW�=��       �	E֙�ec�A�*

lossSa=bU�       �	�r��ec�A�*

loss���=k�ʝ       �	���ec�A�*

lossm��=%3�W       �	E���ec�A�*

loss�(�=(�Q       �	�^��ec�A�*

loss[S�=�I�       �	n���ec�A�*

loss��>7I        �	.���ec�A�*

loss�e=����       �	�)��ec�A�*

lossf =~?�<       �	�Ԟ�ec�A�*

lossO�w=���       �	r��ec�A�*

loss��=���       �	���ec�A�*

lossE�=�L8d       �	����ec�A�*

lossd�&>�k��       �	�S��ec�A�*

loss~�>��       �	Q���ec�A�*

loss�>($¤       �	6���ec�A�*

lossW�?>�n�k       �	U��ec�A�*

lossѼ�=�:s�       �	����ec�A�*

lossۻ�=,��       �	����ec�A�*

losso#>z 1M       �	R)��ec�A�*

lossj�=�,       �	a¥�ec�A�*

loss�(�=�Z�+       �	�n��ec�A�*

loss�b5=���s       �	���ec�A�*

loss�z�=xn��       �	İ��ec�A�*

lossnĢ=��I�       �	�^��ec�A�*

loss8 >���       �	���ec�A�*

lossr>e߾�       �	K���ec�A�*

loss$�=w�\       �	GW��ec�A�*

loss�-<>�q�       �	P��ec�A�*

lossiNi>�,T       �	����ec�A�*

loss�d�=7SW�       �	�¬�ec�A�*

loss���=(ԅ0       �	Eb��ec�A�*

loss�R>E�U       �	���ec�A�*

loss-!*>��M       �	���ec�A�*

loss��>]��       �	'j��ec�A�*

losss4>�v�^       �	_	��ec�A�*

lossd2�=6�hH       �	����ec�A�*

loss
=���       �	�j��ec�A�*

loss+�>ڧ�Q       �	G��ec�A�*

loss���=�,)�       �	��ec�A�*

loss��n>RA�       �	�)��ec�A�*

lossQ>qb@;       �	s���ec�A�*

lossD>g��a       �	�Q��ec�A�*

loss���=j�p�       �	~��ec�A�*

loss>�==N]�       �	ux��ec�A�*

loss
�=|�Y�       �	9��ec�A�*

loss� �=�7�#       �	����ec�A�*

lossi6�=�,�       �	5��ec�A�*

loss��1>Aܵ�       �	�Ƿ�ec�A�*

loss$h�=��n       �	Rb��ec�A�*

loss� 8=D�(       �	]���ec�A�*

loss�q>[ޚ�       �	;���ec�A�*

loss�>Uu�}       �	�:��ec�A�*

loss>k��       �	�ݺ�ec�A�*

lossq��=m-9       �	\s��ec�A�*

loss�L>�}BO       �	1��ec�A�*

loss���=8C��       �	�A��ec�A�*

lossE0�=�J�?       �	Qؽ�ec�A�*

loss��>>��       �	�t��ec�A�*

loss�=��!"       �	�	��ec�A�*

loss�N�=?0�s       �	I���ec�A�*

loss�	>���       �	 6��ec�A�*

loss�'>� J       �	p���ec�A�*

loss�=w=od       �	����ec�A�*

loss��D>>ǕL       �	�j��ec�A�*

loss��>fr�       �	���ec�A�*

lossW]>݁9       �	{M��ec�A�*

loss[Jw>�       �	c��ec�A�*

loss�h
>��a?       �	����ec�A�*

loss���<���       �	���ec�A�*

loss��&=��:�       �	2���ec�A�*

loss���=)�H       �	}%��ec�A�*

loss�Y'>����       �	�.��ec�A�*

loss��<�tm@       �	���ec�A�*

loss*I�=�+�w       �	���ec�A�*

loss��=j�l�       �	�G��ec�A�*

lossD�i>O��       �	���ec�A�*

lossn�=ꋹQ       �	n2��ec�A�*

loss;F�>�Mf�       �	����ec�A�*

loss�{�=q2��       �	�A��ec�A�*

loss*1>Q�,       �	o���ec�A�*

losso��=���       �	���ec�A�*

loss���=Sy�       �	oK��ec�A�*

loss��>�}�       �	���ec�A�*

loss�	�=~(�       �	z��ec�A�*

loss�f=����       �	����ec�A�*

lossØ�=)\��       �	.���ec�A�*

lossZ��=B�j.       �	|��ec�A�*

loss�o=Y	��       �	+���ec�A�*

loss�3K=�-�       �	�(��ec�A�*

loss���<�!�m       �	����ec�A�*

lossR��<�k}�       �	"l��ec�A�*

loss��c=��8�       �	��ec�A�*

loss�a@>KC�C       �	���ec�A�*

loss#�>�on�       �	fI��ec�A�*

loss ¸=���       �	����ec�A�*

loss3>�x�       �	o��ec�A�*

loss��i<F�kn       �	���ec�A�*

lossa��=iE~�       �	���ec�A�*

loss<�>+�vs       �	�[��ec�A�*

lossӎ=	+��       �	k��ec�A�*

loss�4�=��R�       �	2���ec�A�*

loss�5%>���!       �	�J��ec�A�*

losso.>�e�O       �	����ec�A�*

lossh�
=����       �	����ec�A�*

losslM>ۼ�       �	i��ec�A�*

loss�D>�K�3       �	-���ec�A�*

loss<˦>P�q       �	<N��ec�A�*

losss�+>��=�       �	����ec�A�*

loss|�>Ť*�       �	����ec�A�*

lossң==Fh�       �	�(��ec�A�*

lossP3>���       �	����ec�A�*

loss:zA>ȑ��       �	�b��ec�A�*

loss4�=ccO�       �	|��ec�A�*

lossT�U=�*j�       �	m���ec�A�*

loss�ol=��o       �	_F��ec�A�*

lossl�> O�       �	����ec�A�*

loss+ >�_A       �	���ec�A�*

loss�C[=��c       �	;��ec�A�*

loss�O=�"�       �	����ec�A�*

loss2�+=��6�       �	n��ec�A�*

lossc�G=��o       �	��ec�A�*

loss�S=�/�       �	p���ec�A�*

loss�*d>�u�t       �	���ec�A�*

loss�	�=�ӝz       �	�E��ec�A�*

loss�?>��Q�       �	v���ec�A�*

loss�]+>C���       �	%v��ec�A�*

loss�q==꣫�       �	|H��ec�A�*

lossC�'=4�8       �	q���ec�A�*

loss�+�<!a�       �	A~��ec�A�*

loss2��=��]       �	����ec�A�*

loss�c=U��       �	u���ec�A�*

loss><I��       �	"9��ec�A�*

loss��_>N�z�       �	����ec�A�*

lossg^>��z�       �	a���ec�A�*

lossՀ�=��]�       �	�'��ec�A�*

loss���=>1�       �	���ec�A�*

loss��=����       �	:]��ec�A�*

loss�"�=&�qD       �	r���ec�A�*

lossY��=PGP       �	R���ec�A�*

loss�a�= ��x       �	���ec�A�*

losseKC=����       �	�5��ec�A�*

loss�' >Ks,	       �	
���ec�A�*

loss8Z�=�	�p       �	�w��ec�A�*

lossz�->s�{       �	b��ec�A�*

lossq�<N��       �	q���ec�A�*

lossv��=8�}�       �	�C��ec�A�*

loss��5=0ބd       �	�F��ec�A�*

lossj=Ɛ}       �	����ec�A�*

lossV��=�so'       �	�q��ec�A�*

losse�<ds��       �	���ec�A�*

loss]��=� ��       �	ȱ��ec�A�*

loss,)�=����       �	�U �ec�A�*

losstٮ=.�N�       �	5� �ec�A�*

lossƯ<��       �	���ec�A�*

lossf66=��eF       �	��ec�A�*

loss���<�Y�'       �	4��ec�A�*

loss��'>"���       �	�h�ec�A�*

loss. �<3�x       �	3��ec�A�*

loss���=lѓ/       �	+��ec�A�*

loss}	�=@RD�       �	D��ec�A�*

lossͿ�=ғm       �	�p�ec�A�*

loss�5>4Er       �	A�ec�A�*

loss4x�=6"{C       �	O��ec�A�*

loss��=��6b       �	(I	�ec�A�*

loss9C='Q��       �	c�	�ec�A�*

loss�\�=�"|�       �	m�
�ec�A�*

lossf�=�b��       �	6�ec�A�*

lossC�>ֵh       �	���ec�A�*

lossN[�=P�x       �	�l�ec�A�*

loss�u~=��NN       �	��ec�A�*

loss�Ě=<�7�       �	q��ec�A�*

loss�9�=DS
�       �	I�ec�A�*

lossF;{=�p       �	���ec�A�*

loss6Z=���,       �	_}�ec�A�*

lossc�s=���d       �	U�ec�A�*

loss��M=���0       �	���ec�A�*

loss,�=A�5*       �	xB�ec�A�*

lossT$�=� 4       �	���ec�A�*

lossD�a=�?       �	`u�ec�A�*

loss׽�=�b�       �	��ec�A�*

loss��,>@Tq       �	P��ec�A�*

loss�53<�LH       �	�>�ec�A�*

loss�C�<�,�u       �	V��ec�A�*

loss��=w�4t       �	n�ec�A�*

lossS=G��       �	�$�ec�A�*

loss�R<��M       �	}��ec�A�*

lossh�=6Z��       �	kc�ec�A�*

loss��5<�GT       �	���ec�A�*

loss�q�<G˧S       �	���ec�A�*

lossT�<�?7       �	�S�ec�A�*

loss�J�<i���       �	���ec�A�*

loss#��=�o       �	��ec�A�*

loss�,�;ա�       �	 S�ec�A�*

loss�d<�P       �	���ec�A�*

loss�;�\�       �	d��ec�A�*

loss]�=���i       �	&6�ec�A�*

loss<�2>���       �	���ec�A�*

loss�q=�|�	       �	���ec�A�*

lossf<��       �	�C�ec�A�*

lossj8�={R��       �	���ec�A�*

loss���>A�S�       �	E� �ec�A�*

loss#s<�0�       �	�!�ec�A�*

loss��u>vu,       �	l�!�ec�A�*

loss���=��       �	ZH"�ec�A�*

lossE�s>�@:.       �	D�"�ec�A�*

loss�f=�o�*       �	�#�ec�A�*

loss�`=�s .       �	�i$�ec�A�*

loss��>��<       �	� %�ec�A�*

lossA��=��uB       �	�%�ec�A�*

lossO�:>Ѕ�        �	g�&�ec�A�*

loss�� >���       �	ut'�ec�A�*

lossv�>�}�       �	F(�ec�A�*

loss$
>�A       �	Y�(�ec�A�*

lossdk1>:���       �	c�)�ec�A�*

loss
�=%��M       �	�*�ec�A�*

loss�{+>J�'c       �	7+�ec�A�*

loss�hk>>���       �	��+�ec�A�*

loss�C�=a��       �	&T,�ec�A�*

lossL>�=��g�       �	W�-�ec�A�*

loss��>�        �	��.�ec�A�*

loss��<���       �	Y/�ec�A�*

loss�0�=M�=�       �	�	0�ec�A�*

loss���=��       �	��0�ec�A�*

loss��>Epv       �	��1�ec�A�*

loss3�=��:�       �	�2�ec�A�*

loss:�B=y�>'       �	@�3�ec�A�*

loss�#=�y.       �	}4�ec�A�*

loss�9w=D�       �	&5�ec�A�*

loss��=4�C       �	0�5�ec�A�*

loss��=��cZ       �	7�ec�A�*

loss�:�=��       �	�7�ec�A�*

lossMB=i�{       �	9
9�ec�A�*

loss)�=߅�p       �	�B:�ec�A�*

loss�_=����       �	��:�ec�A�*

loss$c�<�=       �	x�;�ec�A�*

loss(�=y��       �	ǀ<�ec�A�*

lossq�=���       �	�=�ec�A�*

loss�<$d{       �	�=�ec�A�*

loss��=u7)y       �	�M>�ec�A�*

loss�U2>v&@"       �	B�>�ec�A�*

lossA��=�m�h       �	�?�ec�A�*

loss�b�=�a��       �	�@�ec�A�*

lossC�=�       �	�@�ec�A�*

loss!e=�mv5       �	�UA�ec�A�*

loss)ǝ=��/<       �	��A�ec�A�*

loss{+=]W�       �	#�B�ec�A�*

loss���=�T�q       �	>C�ec�A�*

loss���=��z       �	��C�ec�A�*

loss���=LӨ�       �	%vD�ec�A�*

loss#v�=1}       �	hE�ec�A�*

loss<��<�qb       �	gF�ec�A�*

loss&X�<���I       �	Z�F�ec�A�*

loss!�>��1�       �	��d�ec�A�*

loss���=��8       �	�e�ec�A�*

loss ��=c4�       �	�&f�ec�A�*

loss|�=Ηo^       �	;�f�ec�A�*

loss_�S>� ��       �	`g�ec�A�*

loss�%:=���       �	z h�ec�A�*

loss=����       �	��h�ec�A�*

lossO<>.z��       �	�Gi�ec�A�*

loss��a>,bEV       �	�hj�ec�A�*

loss��A>���       �		k�ec�A�*

loss���=N��       �	c�k�ec�A�*

lossS�!>QF       �	�Fl�ec�A�*

loss���=V+��       �	��l�ec�A�*

loss�ph=�u�       �	I�m�ec�A�*

lossF>1s��       �	�?n�ec�A�*

lossJ>�*"W       �	;no�ec�A�*

loss��<!f��       �	l#p�ec�A�*

loss��=�Ό       �	��p�ec�A�*

loss�ʘ=B���       �	sr�ec�A�*

lossfk�>�2       �	��r�ec�A�*

loss�5T=,�*�       �	��s�ec�A�*

loss��>�iO�       �	yt�ec�A�*

loss�)�=��oZ       �	ku�ec�A�*

loss�� >��I�       �	�v�ec�A�*

lossl=�Z�       �	9Bw�ec�A�*

loss֖=�y��       �	�x�ec�A�*

loss��=:g�       �	��x�ec�A�*

loss-%�<�g*       �	/�y�ec�A�*

loss��>ёhr       �	�qz�ec�A�*

loss�*�=��9�       �	H6{�ec�A�*

loss."�=�Q        �	9^|�ec�A�*

loss7�-=c8�Y       �	6}�ec�A�*

loss��q=1S       �	,�}�ec�A�*

loss�$>{��S       �	�5~�ec�A�*

loss�A=$0!�       �	c�~�ec�A�*

loss��=�Ԕ�       �	�n�ec�A�*

loss���<�1L       �	N��ec�A�*

loss�I2=%x=�       �	Φ��ec�A�*

loss�)P>�
�U       �	�>��ec�A�*

lossi*�=�%6       �	bׁ�ec�A�*

loss��=��Z�       �	
ւ�ec�A�*

loss���=~鰚       �	em��ec�A�*

loss��^=��y�       �	���ec�A�*

loss �=G�;o       �	�Մ�ec�A�*

loss��=�G�]       �	�s��ec�A�*

loss���=0Z^�       �	k��ec�A�*

loss�f�<_I�p       �	멆�ec�A�*

lossq�>@�''       �	5A��ec�A�*

loss�	=r�Z       �	pw��ec�A�*

loss���<u�9*       �	���ec�A�*

lossϛ�<���       �	���ec�A�*

lossN>5��P       �	�{��ec�A�*

losss&�=_���       �	���ec�A�*

loss[9�>����       �	����ec�A�*

loss:b=���       �	DL��ec�A�*

loss�Ό<k��"       �	����ec�A�*

lossT*=�T�       �	�A��ec�A�*

losse�<K�u�       �	���ec�A�*

lossc��={u��       �	���ec�A�*

loss�V�= u��       �	����ec�A�*

loss��>���       �	m9��ec�A�*

loss��=�nf�       �	@ّ�ec�A�*

loss�=*���       �	6u��ec�A�*

loss�@�=j��        �	rl��ec�A�*

loss���=q��       �	���ec�A�*

loss�
=���       �	,���ec�A�*

loss���=C�Z�       �	�4��ec�A�*

loss�>#ȼ       �	�Z��ec�A�*

lossv�6>�@�*       �	��ec�A�*

loss���=�6       �	쇗�ec�A�*

loss{�K=�凸       �	���ec�A�*

loss��=�xg       �	o���ec�A�*

loss��=�D       �	HP��ec�A�*

loss�c�=B��6       �	���ec�A�*

loss_��=<���       �	Wy��ec�A�*

loss���=z�XD       �	���ec�A�*

loss�:_>�[zs       �	���ec�A�*

loss�@I>S�       �	'N��ec�A�*

loss)1(=H`M�       �	h��ec�A�*

loss3��=�02"       �	e���ec�A�*

loss7ٜ=9\kH       �	H��ec�A�*

loss '�=��J       �	�ڞ�ec�A�*

loss���=�V��       �	�q��ec�A�*

loss�ۇ<!X�       �	���ec�A�*

loss�Y=��5h       �	歠�ec�A�*

lossZ��=��+       �	�S��ec�A�*

lossA�=M[#e       �	���ec�A�*

lossH{�=�\�       �	0���ec�A�*

loss���=r���       �	dY��ec�A�*

loss�E�<�%i       �	����ec�A�*

loss;ش=n$�O       �	7���ec�A�*

lossL�=�ф�       �	G���ec�A�*

loss{ b=mI$�       �	)Y��ec�A�*

loss��=@)5�       �	3è�ec�A�*

loss�>PBE[       �	�h��ec�A�*

loss��B=Rn       �	g��ec�A�*

loss6�=�ƣ       �	}���ec�A�*

loss5��<��       �	3S��ec�A�*

loss!>"�       �	9��ec�A�*

loss���=���       �	���ec�A�*

lossC+�<��       �	5���ec�A�*

loss'�=��0G       �	�ɮ�ec�A�*

loss��=d3�       �	�~��ec�A�*

loss�VK=�9�       �	�H��ec�A�*

lossl�=~�O       �	���ec�A�*

loss?`>����       �	���ec�A�*

lossj��<��#~       �	����ec�A�*

loss$ƥ<�       �	-%��ec�A�*

loss��&>��S       �	���ec�A�*

loss��=��"�       �	Kȴ�ec�A�*

loss��=X��       �	>���ec�A�*

loss�h=f0i\       �	Y��ec�A�*

loss�e�=Xً�       �	_��ec�A�*

lossQcN=��<<       �	!��ec�A�*

loss��=Yt�;       �	���ec�A�*

loss��=5>��       �	����ec�A�*

lossx�G=V���       �	�/��ec�A�*

loss��=�c�       �	2ʺ�ec�A�*

loss�Ig=Tn�       �	,d��ec�A�*

loss�e=m2       �	����ec�A�*

loss�+�=I��       �	ۋ��ec�A�*

loss�N�=����       �	�!��ec�A�*

loss-5o=�N32       �	�ec�A�*

loss��=g�"       �	�L��ec�A�*

loss	p�<��-�       �	���ec�A�*

loss#,�<�!I�       �	�w��ec�A�*

loss|��=��E       �	��ec�A�*

loss$!�=y,�       �	<���ec�A�*

loss��{=�(�       �	#I��ec�A�*

loss���=�W�
       �	����ec�A�*

loss)�)=�;p�       �	c&��ec�A�*

loss� �=��?q       �	a4��ec�A�*

loss���=�H&�       �	����ec�A�*

loss�R4=��H8       �	o���ec�A�*

loss<`j>�+;�       �	 R��ec�A�*

loss۶=��v�       �	����ec�A�*

loss\,~=�       �	�#��ec�A�*

losst~=�xN�       �	=���ec�A�*

loss-x<jI�u       �	F���ec�A�*

loss��-=q���       �	���ec�A�*

lossIHp<��       �	q���ec�A�*

loss֪�=iu-�       �	B?��ec�A�*

lossOd�=x��       �	����ec�A�*

loss�`T>|��3       �	�d��ec�A�*

loss�P�=�O�       �	� ��ec�A�*

loss�~,=2���       �	����ec�A�*

loss�=���       �	Dk��ec�A�*

losse+=���       �	N��ec�A�*

lossJ	=����       �	D���ec�A�*

loss��=��Rg       �	 F��ec�A�*

lossN�<�	}�       �	����ec�A�*

lossH%�=�!��       �	�w��ec�A�*

loss�(�=$ж       �	���ec�A�*

loss���=ʈ��       �	���ec�A�*

lossV��=��,       �	�?��ec�A�*

loss��;�       �	����ec�A�*

loss�=1?�       �	�h��ec�A�*

loss	�=�)��       �	r���ec�A�*

lossŌ�=�"��       �	���ec�A�*

loss?Y=10F
       �	s��ec�A�*

loss�=��e7       �	��ec�A�*

losso�=Hw�b       �	���ec�A�*

loss�[=���       �	;p��ec�A�*

loss���;�Z��       �	���ec�A�*

loss(�=��v       �	81��ec�A�*

loss�'=�a"�       �	����ec�A�*

loss��;=�2�        �	�j��ec�A�*

losszh>MK�       �	m���ec�A�*

lossn*?=���       �	lA��ec�A�*

loss��=���       �	����ec�A�*

loss�	>�K       �	�u��ec�A�*

loss�ܼ;����       �	��ec�A�*

loss�IU=?�b�       �	ޫ��ec�A�*

loss��=;��       �	�D��ec�A�*

lossz�<U��r       �	����ec�A�*

loss���=���       �	6r��ec�A�*

lossN��=}�@U       �	�
��ec�A�*

loss��=H��       �	Ӥ��ec�A�*

lossWQ�=L�       �	h>��ec�A�*

loss4� =�m��       �	����ec�A�*

loss��R;B��h       �	Uj��ec�A�*

loss�\~=K�       �	~��ec�A�*

lossߦ�=�z�       �	����ec�A�*

lossyw�;�Q�F       �	>=��ec�A�*

lossa0W=b�u�       �	����ec�A�*

lossz�<�a�       �	�j��ec�A�*

loss���=�!]       �	���ec�A�*

loss���=ima"       �	R���ec�A�*

loss*�<��.y       �	�K��ec�A�*

loss[,�=�l�m       �	����ec�A�*

loss��'>��۔       �	H���ec�A�*

loss��N=��D*       �	TV��ec�A�*

loss^�<�
5�       �	����ec�A�*

lossd(+=&�:       �	R|��ec�A�*

loss�[�<f���       �	x*��ec�A�*

lossM��=��T�       �	���ec�A�*

loss��=ޯ3�       �	b���ec�A�*

loss)U�=(���       �	%��ec�A�*

loss#Z�="�1m       �	h���ec�A�*

loss���=���       �	�d��ec�A�*

lossrt/=�%�       �	��ec�A�*

loss%��<�`�       �	����ec�A�*

loss��=H���       �	PP��ec�A�*

loss�D�=
. g       �	���ec�A�*

loss��<�~��       �	(���ec�A�*

loss�P =���       �	���ec�A�*

loss i�=	!��       �	���ec�A�*

loss�>_=�       �	[��ec�A�*

lossh�&=�7       �	/���ec�A�*

loss̅>ó��       �	!���ec�A�*

loss_D�=�{i*       �	6��ec�A�*

lossJŁ=M�       �	t���ec�A�*

loss[@`=霗%       �	�j��ec�A�*

loss]
�=��C,       �	_��ec�A�*

loss���=	�3�       �	����ec�A�*

lossf��=�}       �	4��ec�A�*

loss��E>Nk�       �	����ec�A�*

lossl�[>�Җ       �	�a��ec�A�*

loss$�=<��       �	����ec�A�*

loss:�Y>́Q�       �	�� �ec�A�*

loss9}<��       �	�*�ec�A�*

lossc��<�7       �	��ec�A�*

loss5$�=s��       �	3S�ec�A�*

loss�i >ס6X       �	��ec�A�*

loss���<U@n�       �	0�ec�A�*

lossD�=0m       �	��ec�A�*

loss!u�=O�X9       �	o��ec�A�*

loss��=�x��       �	M�ec�A�*

loss�i=j���       �	0��ec�A�*

losssVJ=���f       �	��ec�A�*

lossJ��<p��t       �	���ec�A�*

lossv�<���       �	�;�ec�A�*

loss��=܅@W       �	l��ec�A�*

loss;�r=�߁       �	�	�ec�A�*

loss���=�*�I       �	m9
�ec�A�*

lossxwL=,��       �	A�
�ec�A�*

loss�+=V	�       �	n�ec�A�*

loss�0M>��	$       �	�
�ec�A�*

loss8Nn>(u�       �	���ec�A�*

loss�9M<���       �	Pn�ec�A�*

loss���<�Sw       �	��ec�A�*

loss^.�=�s��       �	���ec�A�*

lossS�=X᳐       �	1�ec�A�*

loss��=*��A       �	��ec�A�*

loss�r�=���       �	,a�ec�A�*

loss7=Q�ɧ       �	%=�ec�A�*

loss�)=="2�       �	R��ec�A�*

loss�X�=/xV�       �	Zb�ec�A�*

loss!�=��x7       �	���ec�A�*

loss��=��Jw       �	&��ec�A�*

loss���=�G4       �	��ec�A�*

lossۆ�=��+6       �	N��ec�A�*

loss �=>܄Q       �	K�ec�A�*

loss��K=9�       �	���ec�A�*

loss$��;�}�       �	Cs�ec�A�*

loss��!=@騏       �	��ec�A�*

loss�=p�l       �	N��ec�A�*

loss(��=c�!�       �	�6�ec�A�*

loss��G=��^E       �	_��ec�A�*

loss҅&=Y��L       �	9b�ec�A�*

loss�U,=�A�f       �	���ec�A�*

loss��!=�	��       �	1��ec�A�*

loss�U�=G�h�       �	]7�ec�A�*

loss�N=3x       �	��ec�A�*

lossn�=�ˎ�       �	Ou�ec�A�*

lossT��<��ɧ       �	w�ec�A�*

lossF��=��'�       �	_��ec�A�*

loss�`�=��m       �	�W�ec�A�*

loss%�>>�߽B       �	���ec�A�*

loss�a=��%       �	���ec�A�*

loss�"�=)�@       �	�5 �ec�A�*

loss �a=-��6       �	)� �ec�A�*

loss��=*��       �	D�!�ec�A�*

loss1��<f�{�       �	 "�ec�A�*

lossV�Q>F�	M       �	�"�ec�A�*

loss��=��(�       �	�R#�ec�A�*

loss)�=Bu=+       �	��#�ec�A�*

loss�6*<�A       �	/�$�ec�A�*

loss!>���h       �	"%�ec�A�*

loss�}�=�� K       �	9�%�ec�A�*

lossA��;�LG|       �	�L&�ec�A�*

loss_�*<�#}(       �	��&�ec�A�*

loss���=?��       �	^�'�ec�A�*

loss�j=n��       �	�(�ec�A�*

loss�N>��a       �	�(�ec�A�*

loss��=�'�       �	b)�ec�A�*

loss6�=+N       �	V�)�ec�A�*

loss��<��b       �	��*�ec�A�*

lossй=G/��       �	9)+�ec�A�*

loss;�<��4�       �	*�+�ec�A�*

lossJ�d<��%�       �	Ee,�ec�A�*

loss�N�=�,�       �	�-�ec�A�*

loss/�)<M�y�       �		�-�ec�A�*

loss�:<��o�       �	`;.�ec�A�*

loss�S�<Dg3       �	)�.�ec�A�*

loss�@P=Br�       �	Eh/�ec�A�*

lossi�O=�$�6       �	�0�ec�A�*

loss�}>�t��       �	r�0�ec�A�*

loss��=�^],       �	@Q1�ec�A�*

loss���=��j       �	BA2�ec�A�*

losso]=�Z�       �	�B3�ec�A�*

loss��E=W���       �	��3�ec�A�*

loss6!=�us�       �	@�4�ec�A�*

loss�>ɵ�       �	�5�ec�A�*

lossq�=�%�        �	Ou6�ec�A�*

loss2�9>B���       �	�7�ec�A�*

loss�I�<����       �	�/8�ec�A�*

loss���=��=�       �	9�ec�A�*

loss��o=�l)�       �	_E:�ec�A�*

loss�i]=#nZ       �	��;�ec�A�*

loss�k4=�4�       �	l?<�ec�A�*

losso=�       �	�<�ec�A�*

loss�,�=�;       �	�=�ec�A�*

losst�=<۟a       �	[">�ec�A�*

lossW�=�]��       �	��>�ec�A�*

loss}�=��d�       �	�Y?�ec�A�*

lossMZ =�L�z       �	��?�ec�A�*

lossL�=��*       �	��@�ec�A�*

loss�y=z8vK       �	�2A�ec�A�*

lossZC=�<       �	J�A�ec�A�*

loss�c�<���       �	�rB�ec�A�*

lossj�=��_w       �	cC�ec�A�*

lossiN�=>���       �	]�C�ec�A�*

loss�?-=Xl��       �	�DD�ec�A�*

loss��D=��g`       �	��D�ec�A�*

loss�k�=e�       �	�zE�ec�A�*

loss�j3=:z�       �	��F�ec�A�*

loss$�=�q��       �	.G�ec�A�*

loss��>=�%��       �	p�G�ec�A�*

loss��=yD��       �	�xH�ec�A�*

loss�y-<�^�>       �	7I�ec�A�*

loss�Τ=�R�       �	��I�ec�A�*

lossF�<݄�	       �	/�J�ec�A�*

loss��<��l       �	+K�ec�A�*

loss��	>���T       �	�L�ec�A�*

lossM�!=l\       �		�L�ec�A�*

loss�M�=�"�       �	��M�ec�A�*

loss^�=:YA       �	GXN�ec�A�*

lossi��=L =�       �	�N�ec�A�*

loss�x�<R�K�       �	�O�ec�A�*

lossO�=2���       �	�*P�ec�A�*

lossS��=qi]5       �	Z�P�ec�A�*

loss�ȥ=j��;       �	�oQ�ec�A�*

losswU�=�HH       �	{R�ec�A�*

loss;>�^�'       �	W�R�ec�A�*

lossm%=��       �	�[S�ec�A�*

lossÅV=
_`�       �	��S�ec�A�*

loss�)>Lzv�       �	��T�ec�A�*

lossXF=vǴ       �	,U�ec�A�*

loss���<N��       �	�U�ec�A�*

loss�2=���       �	?WV�ec�A�*

loss#�`=U�uo       �	��V�ec�A�*

loss6|C=ٔ��       �	�W�ec�A�*

loss�˿=
C�       �	�X�ec�A�*

loss)�>i�       �	ʥX�ec�A�*

loss��<P�X�       �	P:Y�ec�A�*

loss�=�cMX       �	��Y�ec�A�*

loss��<��       �	�eZ�ec�A�*

loss
�0=8Pc_       �	� [�ec�A�*

loss��=]�|r       �	p�[�ec�A�*

loss��h=�3TE       �	n3\�ec�A�*

lossC�X=��h       �	e�\�ec�A�*

lossD��<�e       �	0]�ec�A�*

loss��=�}�8       �	}!^�ec�A�*

lossܳX=��9�       �	�^�ec�A�*

loss���=۲�I       �	Z_�ec�A�*

lossu�=B �M       �	-�_�ec�A�*

lossWP�<Ώ��       �	l�`�ec�A�*

loss*�G=�[Su       �	7Ta�ec�A�*

lossT��<+��       �	��a�ec�A�*

loss��6=�d       �	��b�ec�A�*

loss?c >��O       �	�c�ec�A�*

loss�4�=�)       �	��c�ec�A�*

loss�z�=U�A       �	�Sd�ec�A�*

lossx��<�Z5�       �	9�d�ec�A�*

loss>6��       �	Cf�ec�A�*

lossi�=��=B       �	�f�ec�A�*

loss��^=��       �	�g�ec�A�*

loss_�,=y�d�       �	X<h�ec�A�*

lossֳ�=w�Q�       �	4�h�ec�A�*

loss8�=����       �	-�i�ec�A�*

lossڹ5=���       �	�k�ec�A�*

loss��
>���r       �	ɭk�ec�A�*

losssS�<��e�       �	�Ll�ec�A�*

loss�h�<�k       �	�!m�ec�A�*

loss݋�=��.�       �	�m�ec�A�*

loss��<qu��       �	�Pn�ec�A�*

lossQެ=#�        �	* o�ec�A�*

loss��<�a!�       �	z�o�ec�A�*

lossܖ>�
�       �	J�p�ec�A�*

loss��=eQ5�       �	*rq�ec�A�*

loss�Ή=�Z�       �	��r�ec�A�*

lossE��=\�       �	�:s�ec�A�*

lossE��<6o�       �	]�s�ec�A�*

lossDJ�<��K       �	�ut�ec�A�*

loss8)>,��z       �	Hu�ec�A�*

loss�B�;�#��       �	��u�ec�A�*

loss��;�.4       �	�av�ec�A�*

loss֗k=��f       �	��v�ec�A�*

loss�=EC       �	��w�ec�A�*

loss���<Wz��       �	/6x�ec�A�*

lossX��={֝       �	B�x�ec�A�*

lossM�d>�e�
       �	�hy�ec�A�*

loss�d/=���D       �	z�ec�A�*

loss��8=�\m       �	�z�ec�A�*

losso�=��       �	�S{�ec�A�*

loss��W=��.       �	�{�ec�A�*

loss�0H=�Sr       �	j�|�ec�A�*

losso,�=+a�       �	� }�ec�A�*

losst�=/� �       �	|�}�ec�A�*

loss�>n���       �	�M~�ec�A�*

loss�>�K�       �	��~�ec�A�*

loss��
=���       �	vp�ec�A�*

lossX�>��L�       �	���ec�A�*

loss�}�=~ӧ!       �	 ���ec�A�*

losss��=���       �	0ԁ�ec�A�*

loss�+:<��ݢ       �	�l��ec�A�*

lossl�<����       �	��ec�A�*

lossV��=)���       �	u;��ec�A�*

loss.k�=�їe       �	Mۄ�ec�A�*

loss	��=���       �	x��ec�A�*

loss��>qϠ�       �	���ec�A�*

loss�w@=��       �	ު��ec�A�*

losss�>�y��       �	FB��ec�A�*

loss4��=���)       �	�އ�ec�A�*

lossX̺=6���       �	�~��ec�A�*

lossߘb=Ipsj       �	���ec�A�*

loss�;�=bY�W       �	S���ec�A�*

loss���<3H�       �	�d��ec�A�*

loss�r�;��B�       �	3��ec�A�*

lossc=g��w       �	���ec�A�*

loss���=T�       �	tЌ�ec�A�*

loss���<��       �	�e��ec�A�*

losst=�d5       �	[��ec�A�*

loss$�}=��|       �	���ec�A�*

lossh2�=��@       �	z5��ec�A�*

loss���<i���       �	B��ec�A�*

loss�i%=�\K       �	,��ec�A�*

loss�k{<O       �	���ec�A�*

loss_�I<i��i       �	$���ec�A�*

loss侫=��       �	�R��ec�A�*

lossLݜ=΁"�       �	���ec�A�*

loss{�7=z4��       �	���ec�A�*

loss�*>q�B�       �	��ec�A�*

loss�M�<�-��       �	����ec�A�*

loss�?�<0�k       �	Y��ec�A�*

lossa�w>�r�       �	����ec�A�*

loss�`8=n�       �	S���ec�A�*

lossӽp=��g�       �	v4��ec�A�*

loss��=�[Q       �	�ї�ec�A�*

loss�.=�l�        �	����ec�A�*

loss� R=�/a       �	���ec�A�*

lossc�<����       �	M���ec�A�*

loss�,>�G�       �	�R��ec�A�*

lossv�=-��       �	W��ec�A�*

lossl5)>��G�       �	k���ec�A�*

loss�vn=	��       �	���ec�A�*

loss&��=�Q       �	����ec�A�*

lossR�#=Ap}�       �	oI��ec�A�*

loss�[�=� ��       �	7ߝ�ec�A�*

loss;T�<{���       �	dx��ec�A�*

loss��<4��       �	���ec�A�*

loss��=�F&f       �	/���ec�A�*

loss�^B>N�j�       �	�:��ec�A�*

loss<�=����       �	�֠�ec�A�*

loss�Ռ=�N�       �	<k��ec�A�*

loss��U<o�	�       �	���ec�A�*

lossH�/=
�(�       �	����ec�A�*

loss1�6=+-Ҍ       �	�R��ec�A�*

losszo�;0V<�       �	-��ec�A�*

loss:��=t�v       �	����ec�A�*

lossm��<�I-W       �	%!��ec�A�*

lossS��=@���       �	����ec�A�*

loss&�=6&n�       �	�P��ec�A�*

loss]��<t&�       �	W��ec�A�*

lossѯ=��7@       �	Y���ec�A�*

loss��2=F�c�       �	%��ec�A�*

lossIVp=�f��       �	��ec�A�*

loss���<�;�c       �	�n��ec�A�*

loss��9>얣�       �	���ec�A�*

loss��>	�       �	����ec�A�*

lossֱx=��~�       �	P:��ec�A�*

loss�:@=v�iY       �	Ҭ�ec�A�*

loss�E�=7�3       �	�i��ec�A�*

loss,��=���       �	���ec�A�*

loss��X<�O�A       �	q���ec�A�*

lossE��<����       �	�%��ec�A�*

loss��Y=_���       �	4֯�ec�A�*

loss��j=�u.�       �	�ǰ�ec�A�*

loss�7�=��ƪ       �	}���ec�A�*

loss�ib=Z���       �	ײ�ec�A�*

lossAi�=Z�ޒ       �	�'��ec�A�*

loss�\<v��8       �	���ec�A�*

loss�u=�
�;       �	5	��ec�A�*

loss$�=�2Z       �	����ec�A�*

loss�<	���       �	 T��ec�A�*

loss�<>�^��       �	&���ec�A�*

lossmL�<a�       �	f���ec�A�*

loss���=B��       �	�G��ec�A�*

loss���=��1+       �	����ec�A�*

loss{�=D\�L       �	�%��ec�A�*

loss��<�,,�       �	u˻�ec�A�*

losswP<�Hr[       �	2���ec�A�*

loss�N-=�ғ�       �	�\��ec�A�*

loss��|=�!��       �	e��ec�A�*

loss=R�<�r��       �	T ��ec�A�*

loss=��=��$p       �	<���ec�A�*

loss��\=F�v$       �	�9��ec�A�*

loss�;=�l�       �	B���ec�A�*

loss�S�=n���       �	'l��ec�A�*

loss�<�^�       �	�^��ec�A�*

lossh	y=���       �	����ec�A�*

loss�b=Q��       �	*���ec�A�*

loss]'v=�U�6       �	�)��ec�A�*

loss���<x!         �	7���ec�A�*

loss���=����       �	#f��ec�A�*

loss#��=V%��       �	����ec�A�*

loss���<�w�       �	���ec�A�*

lossZb=en�       �	�4��ec�A�*

lossſ/=v?:       �	 ���ec�A�*

lossi=���P       �	�~��ec�A�*

lossR-�<�K��       �	���ec�A�*

loss;�;X�v�       �	���ec�A�*

loss<�=|9       �	B��ec�A�*

loss�|-=�අ       �	m���ec�A�*

loss�<f��!       �	�v��ec�A�*

loss�,�<�{v       �	� ��ec�A�*

loss�}�=�7��       �	@���ec�A�*

loss|��>���       �	�_��ec�A�*

loss?��=��       �	����ec�A�*

loss\i<��U       �	����ec�A�*

loss�H�=��/k       �	�t��ec�A�*

loss�=#���       �	�L��ec�A�*

loss��=�j�       �	����ec�A�*

lossݑ\<vE�k       �	;���ec�A�*

loss{�<o^/�       �	~Q��ec�A�*

loss]��;	�eZ       �	���ec�A�*

loss�P�;+{j       �	����ec�A�*

loss��:F.��       �	S#��ec�A�*

loss��M=��       �	����ec�A�*

loss��;�О�       �	X��ec�A�*

loss�,;Ф��       �	Z���ec�A�*

loss�H�:�Cb       �	c���ec�A�*

loss�#�;z�       �	 ���ec�A�*

lossn�o=Fǀ*       �	-^��ec�A�*

loss�)=2kR       �	����ec�A�*

loss��;��,�       �	߇��ec�A�*

lossXX=�d_�       �	� ��ec�A�*

loss8z4>�<�       �	���ec�A�*

loss��;o��d       �	�R��ec�A�*

losseP>K�!�       �	����ec�A�*

lossR�E=����       �	���ec�A�	*

loss���=40/       �	 ��ec�A�	*

loss;%=ZprN       �	�/��ec�A�	*

lossR��=��p       �	/���ec�A�	*

loss��`=�n�       �	<��ec�A�	*

lossS��=5�#�       �	w���ec�A�	*

losse�>.,�<       �	mp��ec�A�	*

loss*�>g��F       �	/��ec�A�	*

loss�/<=�Xt�       �	����ec�A�	*

lossjL�=��5       �	�F��ec�A�	*

loss�W=l�#       �		���ec�A�	*

loss��:=/��5       �	�t��ec�A�	*

loss�J�=����       �	W	��ec�A�	*

loss�tr=ׇ��       �	����ec�A�	*

loss|��<��m�       �	<��ec�A�	*

loss���<ڌA�       �	e���ec�A�	*

loss�1�=>�sF       �	hz��ec�A�	*

lossz��=�=�       �	����ec�A�	*

loss���<Z���       �	�\��ec�A�	*

loss(��=:�Z       �	w���ec�A�	*

loss�o�=F�9�       �	̛��ec�A�	*

loss�Y"<c��       �	�@��ec�A�	*

loss��<x��       �	����ec�A�	*

loss�yH<�jW	       �	O���ec�A�	*

lossć=g>_       �	�Q��ec�A�	*

loss��m=��(�       �	o���ec�A�	*

lossa7�=�?f       �	���ec�A�	*

lossڢ�=�,u�       �	�<��ec�A�	*

lossC��<6;,       �	.��ec�A�	*

loss���=#u       �	����ec�A�	*

loss��<��j�       �	_`��ec�A�	*

loss�f�<�md       �	�`��ec�A�	*

loss�J<d�&Y       �	�u��ec�A�	*

loss���<$5�       �	���ec�A�	*

loss�� =�؁�       �	-���ec�A�	*

loss�Y�= ��       �	}y��ec�A�	*

loss �=.-V�       �	�~��ec�A�	*

loss\��==E�       �	@1��ec�A�	*

lossz�'=��Y�       �	;���ec�A�	*

loss6�v=�A2~       �	\��ec�A�	*

loss�2�<�X�       �	� ��ec�A�	*

loss�!*=S���       �	I���ec�A�	*

loss��#=�'Z�       �	C��ec�A�	*

loss��
>P�?t       �	���ec�A�	*

loss���=�|       �	�p��ec�A�	*

loss�R�<���       �	>��ec�A�	*

loss_�i=4Q�       �	����ec�A�	*

loss��i<Z�.�       �	�1��ec�A�	*

loss���<"��       �	����ec�A�	*

loss��=�R �       �	*��ec�A�	*

loss��=�[��       �	h[�ec�A�	*

loss�)=�u       �	���ec�A�	*

lossR	={&]e       �	ڏ�ec�A�	*

loss8l�=�	       �	1�ec�A�	*

loss&�=<T��       �	���ec�A�	*

loss.4�=�a%�       �	�q�ec�A�	*

loss���=��]n       �	��ec�A�	*

loss��>�qP]       �	i��ec�A�	*

loss6��<��       �	9_�ec�A�	*

lossS�=6?4�       �	�ec�A�	*

loss�v=�9�       �	l��ec�A�	*

loss*b�<�.�       �	��ec�A�	*

loss4u�=�qqF       �	���ec�A�	*

loss4�=��       �	^H�ec�A�	*

loss�r�=�<       �	��ec�A�	*

loss�0�<�S��       �	qv �ec�A�	*

loss�m(=�/�       �	!�ec�A�	*

lossʤ =���       �	+�!�ec�A�	*

lossZ��=�qz       �	�9"�ec�A�	*

lossè=���       �	��"�ec�A�	*

loss��">c.�*       �	�k#�ec�A�	*

loss�&[=���       �	%$�ec�A�	*

loss,�=�jIK       �	8�$�ec�A�	*

loss`�$=㉕�       �	�5%�ec�A�	*

loss�=R�S�       �	7�%�ec�A�	*

lossEJp=���       �	�v&�ec�A�	*

loss��5<m�G       �	�'�ec�A�	*

loss�ħ=��J       �	ͯ'�ec�A�	*

lossy=!f<       �	7U(�ec�A�	*

loss�}�<��       �	t�(�ec�A�	*

loss�@�=�n�6       �	Y�)�ec�A�	*

loss�'a<2���       �	�#*�ec�A�	*

loss�ݘ=�-F�       �	�*�ec�A�	*

loss�׈<NG[]       �	uW+�ec�A�	*

loss� =�5�'       �	Z�+�ec�A�	*

loss2#�<��n       �	#�,�ec�A�	*

lossmY�=>^C\       �	:-�ec�A�	*

loss_��=��S       �	Z�-�ec�A�	*

loss���=�)�Y       �	�P.�ec�A�	*

loss�Ȉ=�L��       �	��.�ec�A�	*

losstm�=��       �	W|/�ec�A�	*

loss�6<�#A�       �	0�ec�A�	*

losst't=�P]$       �	S�0�ec�A�	*

loss�Հ=y���       �	��1�ec�A�	*

losso=�&X       �	�2�ec�A�	*

loss�e�=�]       �	��3�ec�A�	*

loss�X{=�;       �	34�ec�A�	*

loss3+�=I΢�       �	p�4�ec�A�	*

loss�Ox;=��W       �	AG5�ec�A�	*

loss�<e�^�       �	8�5�ec�A�	*

loss>��=�[�;       �	jj6�ec�A�	*

loss���<�5�6       �	�7�ec�A�	*

loss!�=`|n       �	/�7�ec�A�	*

lossV}�<"a�       �	�B8�ec�A�	*

loss���;ċ��       �	��8�ec�A�	*

loss���;��}�       �	�{9�ec�A�	*

loss� �;���L       �	g:�ec�A�	*

loss��s<Ԑï       �	:�:�ec�A�	*

loss�J=��       �	JE;�ec�A�	*

loss� >���       �	z�;�ec�A�	*

loss��=H�О       �	߇<�ec�A�	*

lossD5�;�y�Z       �	&=�ec�A�	*

loss��D=�jF�       �	L�=�ec�A�	*

loss��_<O�<       �	1a>�ec�A�	*

lossi��<��f#       �	'�>�ec�A�	*

loss�G=�J�g       �	Ԛ?�ec�A�	*

lossz�=�]       �	�B@�ec�A�	*

loss�6>A�C�       �	4�@�ec�A�	*

loss(r�=|<o'       �	��A�ec�A�	*

loss��7=�a#�       �	yB�ec�A�	*

loss�Ϙ=l�B�       �	�B�ec�A�	*

losse�=z�#       �	�UC�ec�A�	*

lossJ�t==���       �	��C�ec�A�	*

loss�ۑ=��?�       �	S�D�ec�A�	*

loss��
=���       �		7E�ec�A�	*

loss�J=%\�2       �	u�E�ec�A�	*

loss��=���       �	`F�ec�A�	*

loss�3�<�&�*       �	��F�ec�A�	*

losse_�<��X       �	%�G�ec�A�
*

lossʤ�<Ks��       �	</H�ec�A�
*

lossl�=����       �	��H�ec�A�
*

lossO�S=S�q�       �	��I�ec�A�
*

loss�X�=k�],       �	n�J�ec�A�
*

loss���<��I       �	 CK�ec�A�
*

lossM)�=���       �	��K�ec�A�
*

loss�=��$�       �	?qL�ec�A�
*

loss�[�=K:�L       �	� M�ec�A�
*

loss�Q<-��       �	��M�ec�A�
*

loss�sW<,�/       �	�)N�ec�A�
*

loss�"u=g��       �	�N�ec�A�
*

loss%Nk=�}��       �	aRO�ec�A�
*

loss���<����       �	��O�ec�A�
*

loss�R�<���       �	�yP�ec�A�
*

loss�Z=��p	       �	�Q�ec�A�
*

lossFW=���       �	ۤQ�ec�A�
*

loss:�=ˁ�       �	[BR�ec�A�
*

loss���<��	       �	(�R�ec�A�
*

loss);>��G�       �	rS�ec�A�
*

loss%B<CpJL       �	yT�ec�A�
*

lossi��;5�}�       �	c�T�ec�A�
*

loss�[�<'�9�       �	�3U�ec�A�
*

loss:�i<�]F�       �	��U�ec�A�
*

loss}A�= �o:       �	"lV�ec�A�
*

loss?�<��j\       �	��V�ec�A�
*

loss�?R=�w�I       �	��W�ec�A�
*

lossC��;�w�       �	}#X�ec�A�
*

loss\�;6��a       �	��X�ec�A�
*

lossh��=��ʬ       �	UKY�ec�A�
*

loss�Y�<RӜV       �	n�Y�ec�A�
*

lossz?g=��       �	uZ�ec�A�
*

loss��<�}e       �	5[�ec�A�
*

loss;'�<],�       �	��[�ec�A�
*

loss>/<gIX(       �	]5\�ec�A�
*

lossA�k=5
��       �	��\�ec�A�
*

loss$ǐ<�k�       �	c_]�ec�A�
*

lossrdj<@���       �	��]�ec�A�
*

lossQ�=���       �	�^�ec�A�
*

loss^M=@�7       �	_�ec�A�
*

loss��o<���       �	��_�ec�A�
*

lossTi�=�0E�       �	�Z`�ec�A�
*

losss�W=؇       �	��`�ec�A�
*

lossF{=�\��       �	��a�ec�A�
*

loss8k=�+�       �	�$b�ec�A�
*

loss� �<�P �       �	��b�ec�A�
*

loss���=�ˡ       �	�Kc�ec�A�
*

lossÉS=��       �	C�c�ec�A�
*

loss�"n<�Q��       �	td�ec�A�
*

loss��=7~�       �	�e�ec�A�
*

loss�u=�l       �	ߨe�ec�A�
*

loss/�"=�D��       �	�>f�ec�A�
*

loss�<�<Ō�X       �	0�f�ec�A�
*

loss�=�Z~       �	"og�ec�A�
*

loss�x�;ޫ�S       �	�h�ec�A�
*

loss�$�=ob7       �	�h�ec�A�
*

loss�G�<�TG       �	�Ui�ec�A�
*

loss8�=�%�       �	o�i�ec�A�
*

lossƽ�<8�wo       �	��j�ec�A�
*

loss�~;���       �	k�ec�A�
*

lossv�<��̢       �	��k�ec�A�
*

losso<��       �	�Ll�ec�A�
*

loss#b�<�-se       �	��l�ec�A�
*

lossdy�=YPlF       �	ʉm�ec�A�
*

loss��=���       �	8,n�ec�A�
*

loss_c�<�g�       �	��n�ec�A�
*

loss�p�<�U_       �	=eo�ec�A�
*

loss�P�<eEy�       �	]�o�ec�A�
*

loss��=��       �	6�p�ec�A�
*

lossnO< tP�       �	�sq�ec�A�
*

loss���<Ɓ�       �	�'r�ec�A�
*

lossm��<Jh�6       �	7�r�ec�A�
*

loss�^=�f)�       �	x�s�ec�A�
*

loss��~=Rz�       �	�Yt�ec�A�
*

loss�b�=���;       �	� u�ec�A�
*

loss��<OvB       �	�u�ec�A�
*

loss�̃:�-��       �	9Fv�ec�A�
*

lossT�=�M�j       �	��v�ec�A�
*

lossA�	>qE�       �	U�w�ec�A�
*

loss}�;=L�`        �	.x�ec�A�
*

lossĪ�;{�o�       �	=�x�ec�A�
*

loss��o=�+��       �	�wy�ec�A�
*

lossM�<o3�       �	z�ec�A�
*

lossqvJ=�#��       �	��z�ec�A�
*

loss$5`<U��?       �	�P{�ec�A�
*

lossi\�=���4       �	1�{�ec�A�
*

loss߻=FiK       �	*�|�ec�A�
*

loss{�[<lx�       �	�3}�ec�A�
*

lossTo�<�y�       �	��}�ec�A�
*

loss&�=�?       �	f~�ec�A�
*

loss�#c<����       �	�~�ec�A�
*

loss<�=a��m       �	Փ�ec�A�
*

lossm�#<�KP�       �	=*��ec�A�
*

loss���;ÿ�       �	#���ec�A�
*

lossw�=HD;�       �	�P��ec�A�
*

loss��=�zRm       �	 ��ec�A�
*

lossIx8<['       �	����ec�A�
*

loss��%=�e��       �	�*��ec�A�
*

loss��1=�c�       �	�ă�ec�A�
*

lossse=f/F       �	,a��ec�A�
*

lossNS�=H^��       �	1	��ec�A�
*

loss��:���e       �	Y���ec�A�
*

loss���;�l<�       �	�H��ec�A�
*

loss/H�<���C       �	���ec�A�
*

loss���<�O��       �	�z��ec�A�
*

loss%��<Xwt       �	+��ec�A�
*

loss�7`=/Y }       �	 ���ec�A�
*

loss@��=D       �	A��ec�A�
*

lossd�<yQ�       �	�݉�ec�A�
*

loss�nH=���n       �	�v��ec�A�
*

loss�\r=~��       �	���ec�A�
*

loss`�=�|!       �	Oϋ�ec�A�
*

loss�ck<�ŀ       �	�b��ec�A�
*

lossf c<�ۛ       �	{���ec�A�
*

loss�+�=���       �	����ec�A�
*

loss��`;n4�       �	���ec�A�
*

loss��Z<��7       �	w���ec�A�
*

lossR[==����       �	h��ec�A�
*

lossF�=����       �	���ec�A�
*

loss��W=SW�       �	��ec�A�
*

loss�<�%�Z       �	Q��ec�A�
*

lossd�=e�B;       �	iŒ�ec�A�
*

lossc�=�A4       �	zq��ec�A�
*

lossw�=���v       �	���ec�A�
*

lossCGZ=���[       �	d˔�ec�A�
*

loss߲C<S Ma       �	�j��ec�A�
*

loss�Hc<���4       �	'��ec�A�
*

lossi��=W2�       �	.Ŗ�ec�A�
*

loss��	>�!�B       �	���ec�A�*

loss�V=��7       �	���ec�A�*

lossf�8=,�?�       �	c���ec�A�*

loss6ޮ<�w�>       �	�L��ec�A�*

lossSۈ<,�Q�       �	8���ec�A�*

loss��<�"��       �	���ec�A�*

loss�2<�:��       �	�*��ec�A�*

loss��:=�٩�       �	7ƛ�ec�A�*

loss�O<=i_�       �	�Y��ec�A�*

lossXc>��o       �	���ec�A�*

loss��=���*       �	���ec�A�*

loss���=����       �	�"��ec�A�*

loss���=gY�       �	ܺ��ec�A�*

lossd�G=C       �	d��ec�A�*

loss�VL<�S�V       �	���ec�A�*

loss3p=�uK       �	䡠�ec�A�*

lossԙI=mt�       �	�?��ec�A�*

loss�<���       �	�ӡ�ec�A�*

loss;\=|�1       �	N{��ec�A�*

lossR��<)UI       �	E��ec�A�*

loss�
=I�R       �	����ec�A�*

lossV�l=g�5�       �	�A��ec�A�*

loss??=�?�       �	��ec�A�*

lossI�v<N#~f       �	�~��ec�A�*

loss�D�<���       �	���ec�A�*

loss�!�=�o
T       �	#���ec�A�*

loss�:=���       �	�R��ec�A�*

lossQ�=�B��       �	m��ec�A�*

loss���<�|F
       �	{���ec�A�*

losss&�<�>�       �	���ec�A�*

loss���=�J�A       �	����ec�A�*

loss��=�Kۓ       �	�=��ec�A�*

loss�u�<��5�       �	T;��ec�A�*

loss�P<�1Z       �	֫�ec�A�*

loss௷<�e�       �	w��ec�A�*

loss��=�U��       �	����ec�A�*

loss��(=�
g       �	�<��ec�A�*

loss�(/=�H��       �	�Ю�ec�A�*

loss��<Ƽ��       �	���ec�A�*

lossO�=���q       �	!ɰ�ec�A�*

lossT�<���       �	j���ec�A�*

loss�&=�F�w       �	j���ec�A�*

lossC�<��       �	2��ec�A�*

loss/Y=�`��       �	����ec�A�*

loss>d���       �	�P��ec�A�*

loss:j�=$`(]       �	��ec�A�*

lossd�`<��(       �	�~��ec�A�*

loss��<�3~       �	I��ec�A�*

loss���<׮#y       �	d���ec�A�*

loss.<=R��       �	�B��ec�A�*

loss�2�<��=O       �	�ָ�ec�A�*

loss=DA]       �	�<��ec�A�*

lossZ �=�R,       �	�Һ�ec�A�*

loss���<`:��       �	�r��ec�A�*

lossly=����       �	@��ec�A�*

loss��=��;|       �	Ե��ec�A�*

loss��=����       �	�[��ec�A�*

loss��1=�D�_       �	����ec�A�*

lossSa�;g��p       �	����ec�A�*

loss�K=Z(�       �	J'��ec�A�*

loss�hb=���       �	v���ec�A�*

loss㙘=MJ       �	hZ��ec�A�*

loss��<�H�p       �	����ec�A�*

loss�,�;�L�h       �	����ec�A�*

lossOej=��|�       �	tE��ec�A�*

lossJ�=�z       �	����ec�A�*

loss��=
/�q       �	ǂ��ec�A�*

losseo^=���       �	7��ec�A�*

loss�2
>�0A�       �	����ec�A�*

loss0=���M       �	?��ec�A�*

loss@��;�J��       �	���ec�A�*

lossq�=�80       �	�i��ec�A�*

lossCĔ=����       �	����ec�A�*

lossx��<����       �	���ec�A�*

loss�c<�j�       �	��ec�A�*

loss=��<sȰ�       �	����ec�A�*

loss=g�;$�n�       �	�K��ec�A�*

loss�&=�	�       �	����ec�A�*

lossE3�=f
��       �	�9��ec�A�*

loss�Z�=p a�       �	�7��ec�A�*

loss��s<H�޷       �	���ec�A�*

loss4I�<{�       �	*t��ec�A�*

losse��;�F�       �	���ec�A�*

lossT��<~�%       �	{���ec�A�*

loss3)h=%U��       �	�[��ec�A�*

loss3\�;��       �	���ec�A�*

loss1��;��|       �	����ec�A�*

losse_�<o�/       �	����ec�A�*

lossT��=̭       �	�_��ec�A�*

loss)=��/       �	����ec�A�*

loss�� >Dm}!       �	����ec�A�*

lossb�>��J	       �	�7��ec�A�*

loss-q�<o��       �	j���ec�A�*

loss"=m��       �	�}��ec�A�*

loss)��<vh��       �	���ec�A�*

loss!�<7؏�       �	����ec�A�*

lossH��=���{       �	���ec�A�*

loss��2=��8L       �	�0��ec�A�*

loss���=W���       �	g���ec�A�*

loss���;:��-       �	�|��ec�A�*

lossb��=R�       �	���ec�A�*

loss���<��J       �	����ec�A�*

lossDȤ<�U��       �	�_��ec�A�*

loss��#<O�:]       �	���ec�A�*

loss=E�_�       �	����ec�A�*

loss�P�<�x��       �	�6��ec�A�*

loss7=��`       �	<���ec�A�*

loss��_<@Ą"       �	Lo��ec�A�*

loss��=�1�n       �	J��ec�A�*

loss�O�<�+7c       �	]���ec�A�*

lossN�G=��:�       �	�@��ec�A�*

loss�E=ٴX       �	����ec�A�*

loss�Ȭ<���       �	�l��ec�A�*

lossߙ<ŊȮ       �	A��ec�A�*

loss��>�Zb/       �	���ec�A�*

loss��7=�0	       �	�H��ec�A�*

loss�V�<��4       �	���ec�A�*

losswZ]=-��       �	?t��ec�A�*

loss��=��ɒ       �	N	��ec�A�*

lossf�F=&�g�       �	0���ec�A�*

loss�Z�=��       �	/4��ec�A�*

loss��=ʾ�       �	����ec�A�*

loss��=��       �	_��ec�A�*

loss��o<���       �	����ec�A�*

loss��<�z��       �	v���ec�A�*

loss��<��#       �	S!��ec�A�*

loss���=�l��       �	@���ec�A�*

loss�ʫ=9S�       �	F^��ec�A�*

loss݇=��I       �	����ec�A�*

lossdv0=%�&�       �	O���ec�A�*

loss;�%=���       �	R,��ec�A�*

loss5�=��m�       �	���ec�A�*

lossR�<0��       �	�f��ec�A�*

loss?�=!N�       �	����ec�A�*

loss�e�=*�(�       �	����ec�A�*

loss��S=�y�       �	y:��ec�A�*

lossRKE=a1�       �	V���ec�A�*

loss�3>��DL       �	�i��ec�A�*

lossQ/�<��       �	2<��ec�A�*

lossOE?<�R9�       �	E,��ec�A�*

loss�C�=���}       �	K���ec�A�*

loss���<��x�       �	����ec�A�*

lossx� =�o��       �	۾��ec�A�*

lossQ�Q<<��o       �	B_��ec�A�*

loss��=��H       �	4���ec�A�*

lossN�%=�6_�       �	�;��ec�A�*

loss8)	=���        �	:��ec�A�*

lossx�=DP��       �	����ec�A�*

loss��=�.,       �	+4��ec�A�*

loss*��<�W��       �	����ec�A�*

loss�t*<�=�       �	�g��ec�A�*

lossD�=���       �	���ec�A�*

loss!��=��2�       �	����ec�A�*

loss�Sb={���       �	�8��ec�A�*

loss��a<G�       �	U���ec�A�*

lossw;�<5��       �	��ec�A�*

loss6q�=	�)�       �	�#��ec�A�*

lossik�=1BC       �	���ec�A�*

loss��P<��i       �	Q��ec�A�*

loss�L>2�h�       �	����ec�A�*

lossȹ�<�A8       �	΍ �ec�A�*

loss
�<r�W�       �	�-�ec�A�*

loss��<n�]       �	!��ec�A�*

lossR��<Y͈{       �	�e�ec�A�*

losswH�<�f�       �	"��ec�A�*

lossp�=Y1��       �	��ec�A�*

lossx=W?       �	�6�ec�A�*

loss�iH<c�n       �	���ec�A�*

lossH'=r�}�       �	m�ec�A�*

loss���=k9�       �	�ec�A�*

loss��=�'�       �	@��ec�A�*

loss�s9=[,;       �	�A�ec�A�*

loss�:=b��       �	���ec�A�*

loss�a�<���       �	�q�ec�A�*

lossH�=��       �	=
	�ec�A�*

loss�>�o�       �	z�	�ec�A�*

lossWGK<mh"�       �	:;
�ec�A�*

loss�oW<���i       �	��
�ec�A�*

loss2�N=]��       �	Yj�ec�A�*

loss��_=�Y       �	���ec�A�*

loss�=�e�       �	���ec�A�*

loss��/<��|�       �	-�ec�A�*

loss�><F�}G       �	?��ec�A�*

loss��.=�ٯ�       �	�W�ec�A�*

loss��p<�       �	���ec�A�*

losshK�<7���       �	���ec�A�*

loss�:�<$s�       �	$�ec�A�*

loss�!o=s	��       �	���ec�A�*

loss�U�=��|�       �	�[�ec�A�*

loss�BM<'�+       �	���ec�A�*

loss`�;�^��       �	���ec�A�*

loss�|�<N �0       �	B"�ec�A�*

lossc#=E/��       �	���ec�A�*

loss�V<@��7       �	�X�ec�A�*

loss+�=��i       �	Q��ec�A�*

loss<^�=�ӣ�       �	��ec�A�*

loss#C'= ���       �	%�ec�A�*

lossX(X=c�g       �	I��ec�A�*

loss�a�<D�dr       �	GX�ec�A�*

loss��4=Z	"       �	N��ec�A�*

loss��=��i       �	��ec�A�*

lossL�<�tJ�       �	#�ec�A�*

lossXBK=g���       �	l��ec�A�*

loss��I=��vr       �	�E�ec�A�*

loss��=�|�       �	'��ec�A�*

loss0u�<����       �	�p�ec�A�*

loss��n=�*�h       �	_
�ec�A�*

lossb_�=39�       �	��ec�A�*

loss�ֵ=?�Q       �	=�ec�A�*

loss?u<.��       �	���ec�A�*

loss ��<�G�       �	�j�ec�A�*

loss�*=�6�       �	��ec�A�*

loss��=� ��       �	���ec�A�*

loss&d6<5J2       �	�? �ec�A�*

loss&�<j'�>       �	I� �ec�A�*

loss�Z <�y�       �	Dl!�ec�A�*

loss�=�=�Ѧ;       �	�"�ec�A�*

lossWL�=��nW       �	N�"�ec�A�*

lossj��=���'       �	*8#�ec�A�*

loss��=�x       �	��#�ec�A�*

losssr�=LW�       �	g$�ec�A�*

lossX�=z�"       �	��$�ec�A�*

loss��<�sc       �	��%�ec�A�*

losscN+=�i5�       �	q &�ec�A�*

loss��d<B�0�       �	�&�ec�A�*

loss��#=�aQk       �	bN'�ec�A�*

loss�c�<�3��       �	T�'�ec�A�*

loss8BV=T�s'       �	_z(�ec�A�*

lossRё<CZ       �	)�ec�A�*

loss�S�<���       �	��)�ec�A�*

loss�&�<����       �	�J*�ec�A�*

loss�N<os��       �	��*�ec�A�*

lossn�<���=       �	��+�ec�A�*

losss$�<٦j�       �	",�ec�A�*

loss�
l=ڎ��       �	��,�ec�A�*

loss�eZ=8��       �	F-�ec�A�*

loss�%�=ܩ�       �	i�-�ec�A�*

loss2��;���       �	�t.�ec�A�*

loss�c=5�1�       �	�/�ec�A�*

loss���=i���       �	��/�ec�A�*

loss��<����       �	`�0�ec�A�*

loss<F{�4       �	P1�ec�A�*

loss\-=�>��       �	�1�ec�A�*

lossI7=w��X       �	�2�ec�A�*

lossMu+=G|�p       �	O�3�ec�A�*

loss=�<�M       �	<�4�ec�A�*

lossW�l=���{       �	Ot5�ec�A�*

loss�x,= oF�       �	]�6�ec�A�*

loss1=�*�       �	��7�ec�A�*

loss=G
=��&V       �	�L8�ec�A�*

loss��n<�ӽ�       �	a�9�ec�A�*

loss���<�z��       �	�1:�ec�A�*

loss �<[�E       �	�\;�ec�A�*

lossm�9=�%6T       �	�,<�ec�A�*

lossEM�:��;       �	I�<�ec�A�*

loss-ڐ< c:m       �	>�=�ec�A�*

lossQ��=��&       �	�`>�ec�A�*

loss�}i<�4o6       �	�)?�ec�A�*

loss�&<<<B�,       �	��?�ec�A�*

loss#9=o^wP       �	Û@�ec�A�*

lossW�e<�O=       �	�;A�ec�A�*

losszɸ=b�f�       �	&�A�ec�A�*

lossX"�;�k{       �	܀B�ec�A�*

loss��+>���       �	�aC�ec�A�*

loss�-<VD�X       �	�D�ec�A�*

loss���<�^�<       �	�D�ec�A�*

lossHP6=m�?z       �	`9E�ec�A�*

loss\=����       �	��E�ec�A�*

loss���=�h       �	Z�F�ec�A�*

loss��l=:U�       �	5)G�ec�A�*

loss�[=��
       �	��G�ec�A�*

lossf�;d��       �	�kH�ec�A�*

loss�&B>[~F�       �	;I�ec�A�*

loss��5>��       �	$�I�ec�A�*

loss���=dՖ�       �	�YJ�ec�A�*

loss�-�=�5��       �	��J�ec�A�*

lossQ�=#$yl       �	,�K�ec�A�*

loss�t�=���       �	�4L�ec�A�*

loss���<M'�       �	��L�ec�A�*

loss�D,=0��       �	�eM�ec�A�*

loss)�:=eP�3       �	:N�ec�A�*

loss=�C=��       �	t�N�ec�A�*

loss} �<|���       �	�,O�ec�A�*

loss�?�<'�J       �	��O�ec�A�*

loss9R=s�       �	8gP�ec�A�*

loss}��<}c0       �	�Q�ec�A�*

loss�q?=աm       �	��Q�ec�A�*

loss)R=��L       �	�DR�ec�A�*

loss�q�<T�~�       �	��R�ec�A�*

loss��=>�Z       �	SyS�ec�A�*

loss��<q/�       �	rT�ec�A�*

lossH��=5[h       �	C�T�ec�A�*

loss�~g=��ʐ       �	�BU�ec�A�*

loss{��=D��       �	��U�ec�A�*

loss2I�<S��N       �	I�V�ec�A�*

lossf��<xcN�       �	GW�ec�A�*

loss��G<w�˂       �	��W�ec�A�*

loss�5#=��B       �	�bX�ec�A�*

loss���<Mؚ       �	�Y�ec�A�*

loss?Q=��m       �	U�Y�ec�A�*

loss��1=����       �	ZFZ�ec�A�*

loss䐩=ѺP       �	��Z�ec�A�*

loss쑍=�!x       �	�{[�ec�A�*

lossm�<:`�       �	\�ec�A�*

lossX��=�Z��       �	��\�ec�A�*

losst��<��g}       �	G<]�ec�A�*

loss[El=�TȬ       �	��]�ec�A�*

loss��%=�(��       �	�b^�ec�A�*

loss�ǀ=�R       �	9�^�ec�A�*

loss� Q=��@�       �	��_�ec�A�*

loss���<���       �	�`�ec�A�*

lossZ��;@�}       �	��`�ec�A�*

loss�T�<�B�(       �	�Wa�ec�A�*

lossZ�/<+*ج       �	��a�ec�A�*

lossx�=�Y��       �	�|b�ec�A�*

loss�X�;��M�       �	c�ec�A�*

loss�"=�9�       �	�c�ec�A�*

loss�/=����       �	35d�ec�A�*

loss��;���@       �	��d�ec�A�*

loss�`A=+M�       �	>^e�ec�A�*

loss��(=�F�       �	8�e�ec�A�*

loss&Z=���       �		�f�ec�A�*

loss?��<B{��       �	�g�ec�A�*

loss�O<t{��       �	��g�ec�A�*

lossS>�'+1       �	fNh�ec�A�*

lossy�;��n       �	v�h�ec�A�*

loss� �<����       �	uxi�ec�A�*

loss�{<,[bb       �	9j�ec�A�*

loss�_[=����       �	2�j�ec�A�*

lossJ�'<fv�Q       �	�Wk�ec�A�*

loss4z=&�       �	s�k�ec�A�*

loss�T,;��lF       �	��l�ec�A�*

loss�H;.P�#       �	�5m�ec�A�*

loss�~�:ݾ��       �	�m�ec�A�*

lossV�;gP�       �	s�n�ec�A�*

loss��:R���       �	�$o�ec�A�*

loss`N;�,-       �		�o�ec�A�*

loss��-= �>       �	\p�ec�A�*

loss�چ=��6'       �	8�p�ec�A�*

loss�p�:��[*       �	a�q�ec�A�*

loss���<�ݭw       �	�^r�ec�A�*

loss])>)GL       �	T:s�ec�A�*

loss��;y��w       �	�t�ec�A�*

loss��2>�V3       �	Y2u�ec�A�*

loss��="!�?       �	��u�ec�A�*

loss���=�        �	=�v�ec�A�*

loss���=�⊒       �	�/w�ec�A�*

loss��=޴       �	��w�ec�A�*

loss`zv=�,@       �	ax�ec�A�*

loss
�?=Ԁ)       �	��x�ec�A�*

loss���<���\       �	��y�ec�A�*

loss$ߗ=)	�       �	�Gz�ec�A�*

lossC�#=q[�       �	3�z�ec�A�*

loss��=	o�)       �	V�{�ec�A�*

lossZ��=��d�       �	�u|�ec�A�*

loss���<Á �       �	r}�ec�A�*

loss���=_�)�       �	[�}�ec�A�*

loss��=l$c       �	qU~�ec�A�*

loss�J=�pD�       �	��~�ec�A�*

loss��=��|�       �	1��ec�A�*

loss��=6���       �	�-��ec�A�*

loss��;�Db�       �	�Ȁ�ec�A�*

loss;�;��       �	J`��ec�A�*

loss9p�=�.^=       �	z���ec�A�*

lossn�`=	B��       �	���ec�A�*

lossn�&<>�:~       �	;7��ec�A�*

loss��W<��l       �	�փ�ec�A�*

loss�^�;_��>       �	�|��ec�A�*

loss}��<�q       �	��ec�A�*

lossl�;mZ��       �	����ec�A�*

loss��=븿&       �	MN��ec�A�*

loss���=���       �	���ec�A�*

lossQӂ<x�	       �	S���ec�A�*

loss|l=g� Z       �	�0��ec�A�*

lossç=����       �	�Έ�ec�A�*

loss�^=I�       �	����ec�A�*

lossx�9=m8       �	x)��ec�A�*

loss��<<��X       �	�P��ec�A�*

loss��Y=� 3       �	���ec�A�*

loss#�<���       �	~���ec�A�*

loss��=��K�       �	F��ec�A�*

loss-�S=MC6M       �	���ec�A�*

loss��"<'��       �	����ec�A�*

loss��<�;,       �	�.��ec�A�*

loss7V=L�6       �	qɏ�ec�A�*

lossd>��!       �	+i��ec�A�*

loss!d><vIW       �	y��ec�A�*

loss��(== �       �	V���ec�A�*

loss���=Z!�       �	�c��ec�A�*

lossF�;Ŧ       �	(F��ec�A�*

lossX_=y�x�       �	�w��ec�A�*

loss4�<%��Y       �	���ec�A�*

lossJ=*�e       �	����ec�A�*

loss�!= �u�       �	�P��ec�A�*

loss��D=9�e�       �	��ec�A�*

loss��=c;�E       �	aO��ec�A�*

loss��f=~���       �	I��ec�A�*

loss�-�<@���       �	x���ec�A�*

loss��g<=�>�       �	�K��ec�A�*

loss�Qj<W�5        �	؂��ec�A�*

loss��A=1"�M       �	����ec�A�*

loss��b=w׎�       �	�W��ec�A�*

lossh��=ȴ�H       �	����ec�A�*

loss��;�Ro�       �	����ec�A�*

loss�u�=��m�       �	v6��ec�A�*

loss�8=8��{       �	���ec�A�*

loss�.=�9M       �	g���ec�A�*

loss�=��'�       �	i5��ec�A�*

loss�h�=?g��       �	_Ӹ�ec�A�*

loss?<�'�u       �	�x��ec�A�*

loss�^�<̼2�       �	m ��ec�A�*

loss�]= �f�       �	����ec�A�*

loss,�<�$�y       �	.W��ec�A�*

loss�{�=���       �	���ec�A�*

loss1>�`�A       �	&���ec�A�*

lossf�e<,σ�       �	�)��ec�A�*

loss���=�L�}       �	�ҽ�ec�A�*

loss[�<�۳�       �	zo��ec�A�*

loss׻�<Xe��       �	��ec�A�*

loss���=��8�       �	���ec�A�*

loss�E�<OO��       �	p>��ec�A�*

loss='�<(�T       �	N���ec�A�*

loss���=��       �	�l��ec�A�*

lossg�=�U|�       �	m��ec�A�*

loss3�<O�
       �	і��ec�A�*

lossE<AI��       �	�/��ec�A�*

loss;*.>e7�C       �	���ec�A�*

loss62�=hSӴ       �	
g��ec�A�*

loss81>���       �	.��ec�A�*

loss���;��}~       �	
I��ec�A�*

loss$�<=�6;�       �	>���ec�A�*

lossf�>���       �	���ec�A�*

loss�[c=����       �	�%��ec�A�*

loss�=�`0       �	����ec�A�*

lossh8�<(3A�       �	�\��ec�A�*

loss�vm<�O�       �	W���ec�A�*

loss��=[9�P       �	#���ec�A�*

loss�,=�$�K       �	!W��ec�A�*

loss�:X>����       �	����ec�A�*

loss틊=N��       �	����ec�A�*

loss��2=���       �	{2��ec�A�*

loss�`^=�]m�       �	���ec�A�*

loss��a;�q'�       �	in��ec�A�*

loss�|�<�Fq�       �	r��ec�A�*

loss1C=o��       �	^���ec�A�*

loss���<���9       �	�^��ec�A�*

loss��1>n
��       �	 ��ec�A�*

loss�t<Wq:       �	����ec�A�*

loss��:*h�       �	YN��ec�A�*

loss�h�<��Y       �	���ec�A�*

loss>�:$�U�       �	.���ec�A�*

loss��;Z8<�       �	VG��ec�A�*

loss�O=�uG       �	����ec�A�*

loss;��=����       �	$��ec�A�*

lossN�< �2�       �	3��ec�A�*

loss�<`�        �	���ec�A�*

loss�&	=u��       �	�]��ec�A�*

loss#�<��Y|       �	^���ec�A�*

loss���=�,O       �	ڌ��ec�A�*

loss��u=����       �	y!��ec�A�*

loss=z*=M �       �	E���ec�A�*

loss\+%=����       �	���ec�A�*

loss��"=�Q�       �	F���ec�A�*

loss� <�d�       �	QN��ec�A�*

loss�f�<��F       �	����ec�A�*

loss��<���       �	9|��ec�A�*

loss�<ڤ       �	����ec�A�*

loss�DM<t�s'       �	;��ec�A�*

lossҎ'<�u�R       �	����ec�A�*

lossV�<֨z�       �	Ot��ec�A�*

lossx@S=��q       �	���ec�A�*

loss'<�lQ�       �	���ec�A�*

loss8)7=-o       �	����ec�A�*

loss���=���       �	[��ec�A�*

lossf�=׃�W       �	s���ec�A�*

loss\5=�X�Q       �	����ec�A�*

loss؁;���       �	���ec�A�*

loss�-/<��i�       �	#J��ec�A�*

loss�ي;����       �	B���ec�A�*

loss�,�<1+DE       �	Ύ��ec�A�*

loss�ţ;EL��       �	40��ec�A�*

loss���<�?%j       �	����ec�A�*

loss�Y�<�}�       �	Á��ec�A�*

lossmr=q�]       �	2��ec�A�*

loss�P=O]_       �	g���ec�A�*

loss�˩<DX��       �	�W��ec�A�*

loss��;��7	       �	����ec�A�*

loss^�<=V��       �	$���ec�A�*

lossn��:��F�       �	*��ec�A�*

lossAS�="�DS       �	����ec�A�*

loss�2'<&�b       �	2W��ec�A�*

loss�N=�[+�       �	����ec�A�*

loss!S�<,%ok       �	B���ec�A�*

loss!v(<{�B       �	o~��ec�A�*

loss/_<��;�       �	O ��ec�A�*

loss<�=?�~�       �	f���ec�A�*

loss�)<̾)S       �	 X��ec�A�*

loss,��<��J       �	����ec�A�*

lossxJ�=H��       �	�y��ec�A�*

loss�?�;q��       �	�X��ec�A�*

loss�.�<�Lh8       �	�T��ec�A�*

loss`y�=���       �	����ec�A�*

loss�T_<8�       �	p?��ec�A�*

loss3=��RZ       �	�N��ec�A�*

loss���=�I�?       �	V���ec�A�*

loss��<]��4       �	Lo��ec�A�*

loss2��:{���       �	�+��ec�A�*

loss��=��=       �	]���ec�A�*

lossk<��c@       �	���ec�A�*

loss�O=!�q       �	$���ec�A�*

loss��f=6V�!       �	'J�ec�A�*

loss�]<Ȟ��       �	���ec�A�*

loss��<a4�        �	���ec�A�*

loss�!�<��õ       �	K#�ec�A�*

loss@I�=��6�       �	���ec�A�*

loss��<��D       �	�`�ec�A�*

lossA}<94�:       �	���ec�A�*

loss|̻<�K       �	���ec�A�*

loss�	�<����       �	�]�ec�A�*

loss$�=�*       �	���ec�A�*

loss`� =z��       �	p��ec�A�*

lossS��=���       �	�:�ec�A�*

lossv�=ۄ��       �	���ec�A�*

lossq�?=ccm�       �	Ll	�ec�A�*

lossy�	<m��       �	�
�ec�A�*

loss�64=0%�       �	��
�ec�A�*

loss��<�X�       �	׊�ec�A�*

loss��
>P%       �	`"�ec�A�*

loss1��<���       �	���ec�A�*

loss@�;�
6�       �	�T�ec�A�*

loss%/�<N�O�       �	���ec�A�*

loss$ü;��s�       �	���ec�A�*

loss���;����       �	���ec�A�*

loss�t6</��       �	uX�ec�A�*

lossn-J<�S��       �	 ��ec�A�*

loss,R�<]��d       �	��ec�A�*

loss��>+{�       �	B��ec�A�*

loss;߇=
R�$       �	�G�ec�A�*

loss��*<����       �	H��ec�A�*

loss{P�<I��       �	�s�ec�A�*

loss�;q�+�       �	��ec�A�*

loss��<d+�        �	ж�ec�A�*

loss1�R<=jf�       �	�`�ec�A�*

loss���<��̔       �	h�ec�A�*

loss?�(==*u�       �	���ec�A�*

loss���<q<��       �	�=�ec�A�*

loss��m=u��       �	-�ec�A�*

loss���=��e�       �	��ec�A�*

loss�7)<�^W"       �	�2�ec�A�*

loss�'�<��3�       �	���ec�A�*

lossf�< \��       �	�a�ec�A�*

loss�nQ=��d       �	��ec�A�*

lossd��;�k�       �	���ec�A�*

loss, �< Hl6       �	�/�ec�A�*

loss��
=	h�'       �	��ec�A�*

loss�=�.�       �	�o�ec�A�*

lossU7::�<       �	A�ec�A�*

loss6�=���       �	���ec�A�*

loss�</���       �	$E �ec�A�*

loss�<A�`�       �	� �ec�A�*

loss�p+=:�p       �	�y!�ec�A�*

loss�n�<�;��       �	j"�ec�A�*

lossWZ= �z�       �	O�"�ec�A�*

loss�K�<��G�       �	4L#�ec�A�*

loss�j�;}��D       �	&�#�ec�A�*

loss���<���       �	�$�ec�A�*

loss��=z�־       �	�T%�ec�A�*

loss*��;1��m       �	�%�ec�A�*

loss>�=����       �	�&�ec�A�*

loss�=�E��       �	N^'�ec�A�*

lossW�	< !K�       �	O(�ec�A�*

loss�u�=h��       �	��(�ec�A�*

loss�r�;D[�       �	}@)�ec�A�*

loss1Cf;7�n�       �	�)�ec�A�*

loss���<:&�c       �	J*�ec�A�*

loss��0<ؼX�       �	�j+�ec�A�*

loss��<��C       �	u,�ec�A�*

loss��U;@~��       �	a�,�ec�A�*

lossx��;�*�$       �	�H-�ec�A�*

loss��=(�1�       �	��-�ec�A�*

loss�^='6�H       �	܁.�ec�A�*

loss`V<��xQ       �	(a0�ec�A�*

loss�-?=I�x�       �	D�0�ec�A�*

loss���=9b��       �	f�1�ec�A�*

loss���<�3��       �	�42�ec�A�*

loss��7<('�       �	0�2�ec�A�*

loss�� ;���S       �	��3�ec�A�*

loss��<Ҫ�Z       �	Ϣ4�ec�A�*

loss�C<�6_�       �	"T5�ec�A�*

loss3�=t=��       �	��6�ec�A�*

loss�{/=%[��       �	��7�ec�A�*

loss ��=��:       �	�j8�ec�A�*

lossQ��<(#0m       �	I�9�ec�A�*

loss��<@�q       �	�:�ec�A�*

loss��;��qC       �	�;�ec�A�*

loss7_<1�8       �	�t<�ec�A�*

loss���< ��       �	�=�ec�A�*

loss���;�cp       �	�=�ec�A�*

lossk��=�L%�       �	vR>�ec�A�*

loss�=���6       �	�>�ec�A�*

loss�='�Y�       �	�?�ec�A�*

loss��I=�<l.       �	5@�ec�A�*

loss�}=]��N       �	f�@�ec�A�*

loss���;�4       �	�wA�ec�A�*

loss�\�=뿀�       �	zB�ec�A�*

lossW.�;̺�       �	��B�ec�A�*

lossCb<���       �	�PC�ec�A�*

loss�=	j�X       �	��C�ec�A�*

loss&X=$)�       �	��D�ec�A�*

loss1t>��9�       �	l"E�ec�A�*

loss�23=.j��       �	��E�ec�A�*

loss�4�=oX1f       �	.SF�ec�A�*

loss-#i=�ʖ�       �	��F�ec�A�*

loss�Ҙ<2���       �	��G�ec�A�*

loss�d<ב�2       �	rH�ec�A�*

loss��<&��       �	ͫH�ec�A�*

lossc�z=,C�H       �	 FI�ec�A�*

loss;p�<I�B       �	D�I�ec�A�*

lossOf�;���       �	 {J�ec�A�*

loss�:8=���       �	P�K�ec�A�*

loss�r=Z���       �	y;L�ec�A�*

lossR<�>�7       �	��L�ec�A�*

loss��;qk,.       �	�zM�ec�A�*

loss:�;�笠       �	GN�ec�A�*

loss
�:=��r�       �	$O�ec�A�*

loss�^�=���       �	̸O�ec�A�*

loss�/Q<$$2       �	�WP�ec�A�*

loss�4�=.(�       �	�P�ec�A�*

lossOKC<�d&       �	��Q�ec�A�*

losss��<Z�K�       �	2XR�ec�A�*

lossQѐ=����       �	�R�ec�A�*

loss��U=�r�       �	��S�ec�A�*

loss@�0<��ع       �	L6T�ec�A�*

lossz��<��n       �	�T�ec�A�*

loss8]�=n�       �	\�U�ec�A�*

loss���<��C?       �	,-V�ec�A�*

lossP�<v(��       �	��V�ec�A�*

loss�h,=��3�       �	�fW�ec�A�*

loss=c�<���A       �	WX�ec�A�*

loss*�<%�@@       �	q�X�ec�A�*

loss�|W<o �S       �	B_Y�ec�A�*

loss�Y�<��+&       �	Z�ec�A�*

loss��<�/       �	H�Z�ec�A�*

lossň=����       �	kH[�ec�A�*

loss|?X=���N       �	��[�ec�A�*

loss�M%=���       �	ҋ\�ec�A�*

lossr՝;&+��       �	�0]�ec�A�*

lossV��;H~p�       �	W>^�ec�A�*

loss�I�=f�W       �	+�^�ec�A�*

loss6�=�\       �	�w`�ec�A�*

loss'e<e|��       �	�a�ec�A�*

loss���;2���       �	�"b�ec�A�*

loss}H�=��l�       �	=�b�ec�A�*

loss!C=J|��       �	-Zc�ec�A�*

loss�*�<�2W       �	+�c�ec�A�*

loss
��<�C��       �	؛d�ec�A�*

loss���<��o�       �	�8e�ec�A�*

loss���<.�@�       �	��e�ec�A�*

loss��$<6g       �	�{f�ec�A�*

loss�(=����       �	vg�ec�A�*

loss�o�<�-&       �	ٰg�ec�A�*

lossM�B=��Ȅ       �	.Vh�ec�A�*

lossV$=Bǂ!       �	��h�ec�A�*

loss���<5���       �	Hj�ec�A�*

loss\��<:m��       �	&�j�ec�A�*

loss�=5�9       �	�?k�ec�A�*

loss��=;b�:       �	[�k�ec�A�*

loss�]�=}�D�       �	�rl�ec�A�*

loss��<�Y4       �	Lm�ec�A�*

loss��;���       �	��m�ec�A�*

loss��;� �       �	
in�ec�A�*

loss���=𛣸       �	�
o�ec�A�*

lossxԈ=�#��       �	?�o�ec�A�*

lossʩ<w�       �	QJp�ec�A�*

loss���<���       �	��p�ec�A�*

loss�E�<�Q{        �	ƈq�ec�A�*

loss��={\��       �	p'r�ec�A�*

loss�=
�]       �	��r�ec�A�*

loss\u{;�]       �	TTs�ec�A�*

loss�g�<����       �	�s�ec�A�*

loss�;�0'       �	0�t�ec�A�*

loss�"=��2       �	�tu�ec�A�*

loss��:Al�o       �	�v�ec�A�*

loss	6;dǾ�       �	S�v�ec�A�*

loss���=Ӈ��       �	k'w�ec�A�*

losss��<�Ŀ�       �	��w�ec�A�*

loss�/�;�k�       �	nOx�ec�A�*

loss��=H��d       �	�x�ec�A�*

loss*�x=,�=v       �	��y�ec�A�*

loss}'�<���       �	]4z�ec�A�*

lossa^}=a"!�       �	I�z�ec�A�*

loss�>Cy��       �	K|�ec�A�*

loss�HH=��7       �	:]}�ec�A�*

loss��=���       �	��}�ec�A�*

lossԢ<ۋ�M       �	��~�ec�A�*

lossʲ:<�Ά       �	�=�ec�A�*

loss&��=�m�       �	���ec�A�*

lossH� =�elT       �	�q��ec�A�*

loss�h^=<�Y       �	���ec�A�*

loss*Թ;��O�       �	ʥ��ec�A�*

loss��<g���       �	~7��ec�A�*

loss��=��&       �	Tɂ�ec�A�*

loss�n�<��r3       �	�Y��ec�A�*

loss�jB=�L#�       �	���ec�A�*

loss� �<�Ro       �	f���ec�A�*

loss�@4=�ѻ5       �	���ec�A�*

loss��<�@'#       �	����ec�A�*

loss���<je��       �	+P��ec�A�*

loss+2=��q       �	���ec�A�*

loss}��:H�3b       �	N{��ec�A�*

lossf%<7�       �	��ec�A�*

loss��<��%       �	:���ec�A�*

loss���<]��       �	�L��ec�A�*

loss�=.�׉       �	���ec�A�*

loss�M=�M0       �	c{��ec�A�*

lossF�3=��]~       �	���ec�A�*

lossܝ�:f�kh       �	Y���ec�A�*

loss�<gTbO       �	ڎ��ec�A�*

loss���=�s��       �	h"��ec�A�*

loss��=��"       �	W���ec�A�*

loss���=��#       �	�~��ec�A�*

lossV��;Xg>�       �	�0��ec�A�*

loss�xh=�.�       �	_я�ec�A�*

loss���<a�       �	�t��ec�A�*

lossL=����       �	���ec�A�*

losswe.<��Т       �	����ec�A�*

loss��<ƾ�       �	3Q��ec�A�*

loss�N�<�Z��       �	���ec�A�*

loss�5b</g+5       �	�}��ec�A�*

lossV�V=�#        �	^��ec�A�*

lossw�<x-       �	Ͼ��ec�A�*

loss�3}=���j       �	����ec�A�*

loss1�8:Ȋ>�       �	�5��ec�A�*

loss�q�;!�:       �	�ؖ�ec�A�*

loss*n.<�D`�       �	5y��ec�A�*

lossC�<�Ʃ*       �	���ec�A�*

lossT��<mp�G       �	����ec�A�*

lossza�=�$       �	�E��ec�A�*

loss �	=�h70       �	0ٙ�ec�A�*

loss�Q�;�q��       �	!s��ec�A�*

loss}-�<����       �	p
��ec�A�*

loss(S=�Z��       �	���ec�A�*

lossZW<f       �	T7��ec�A�*

loss�F�;��:�       �	dʜ�ec�A�*

losszaL<����       �	�^��ec�A�*

loss�S�=_e��       �	����ec�A�*

lossM�3;$�i�       �	��ec�A�*

loss���=��X�       �	"��ec�A�*

loss �x:rK�       �	A���ec�A�*

loss��K<����       �	�J��ec�A�*

loss�q;�]       �	U۠�ec�A�*

loss�=Uo_       �	�n��ec�A�*

lossEH,=0b�       �	,��ec�A�*

loss �<k��       �	���ec�A�*

lossG�<BX�       �	�:��ec�A�*

loss���=� #       �	5У�ec�A�*

loss�R�<U��       �	c��ec�A�*

loss	3�<l��       �	����ec�A�*

loss�d�=0��       �	c���ec�A�*

loss�Dn=j5Z�       �	k+��ec�A�*

lossC"�;v���       �	Ӿ��ec�A�*

loss��=���X       �	.X��ec�A�*

loss�-=F��       �	���ec�A�*

loss�}�<�U	       �	���ec�A�*

loss�<a=�Q�?       �	���ec�A�*

loss�Y<09�       �	���ec�A�*

loss㇉<N�C       �		O��ec�A�*

loss��=�]�       �	���ec�A�*

lossQ*=�h��       �	�{��ec�A�*

lossִ�=��iL       �	a��ec�A�*

loss&�=:�N       �	���ec�A�*

loss��i<��>�       �	�E��ec�A�*

loss�5=}�       �	Yڭ�ec�A�*

loss�G�;E&Q�       �	�j��ec�A�*

loss�X=t��Q       �	R��ec�A�*

loss�*>>�N�       �	ס��ec�A�*

loss���;��       �	y=��ec�A�*

loss�U=���       �	�Ӱ�ec�A�*

loss��<�x&4       �	���ec�A�*

loss�g�<���       �	$��ec�A�*

loss���<���       �	���ec�A�*

loss}Y<<7c�       �	bM��ec�A�*

loss��<vQ�       �	���ec�A�*

loss:��<�Ͻ#       �	�|��ec�A�*

loss��=���;       �	�3��ec�A�*

lossQ�<� O       �	�Ƶ�ec�A�*

loss*�=��ٗ       �	�\��ec�A�*

loss���<2'��       �	���ec�A�*

loss���=�9��       �	���ec�A�*

loss4�^<�g2H       �	+��ec�A�*

loss��;f�e       �	Q���ec�A�*

loss�#z;&��       �	 T��ec�A�*

loss�M<�r�"       �	-��ec�A�*

loss�=yb=�       �	�~��ec�A�*

loss�ݭ<� N       �	T��ec�A�*

loss�g
>�2��       �	����ec�A�*

loss��=X�If       �	�U��ec�A�*

lossC5�=�S�P       �	x��ec�A�*

lossZS=����       �	Ⓗ�ec�A�*

loss��<��8       �	'��ec�A�*

loss��=g�o       �	j���ec�A�*

loss�:=]�i)       �	?R��ec�A�*

lossq�6<i�.�       �	u��ec�A�*

loss�`{<��}       �	����ec�A�*

loss��Z=1���       �	� ��ec�A�*

loss�e�=݅q�       �	ܷ��ec�A�*

loss���=Q�$�       �	N��ec�A�*

loss��B=6f��       �	����ec�A�*

loss�J�<=��       �	����ec�A�*

loss#�0;E�w�       �	���ec�A�*

loss��<v�y�       �	����ec�A�*

loss�=V��       �	�T��ec�A�*

loss�*�=���c       �	&���ec�A�*

lossQ �<�A��       �	p���ec�A�*

loss�9�<�S�       �	�4��ec�A�*

loss�_/<ެ��       �	���ec�A�*

loss?��=���p       �	!\��ec�A�*

loss:�=�&#j       �	g���ec�A�*

loss��=#U�       �	����ec�A�*

loss�]"<�.�       �	�/��ec�A�*

loss�t=�3�       �	����ec�A�*

lossx�^<��       �	�j��ec�A�*

loss�iN;;���       �	\��ec�A�*

loss�]$={T�       �	����ec�A�*

loss��
<�͊'       �	4/��ec�A�*

loss���<��!=       �	;���ec�A�*

lossA�h=(\��       �	�x��ec�A�*

loss�U:=��|       �	���ec�A�*

loss��<�3h�       �	����ec�A�*

loss��k<iL       �	����ec�A�*

loss獆=R,K�       �	�*��ec�A�*

loss!��;��9-       �	Y���ec�A�*

lossZ�/<(sx       �	Y��ec�A�*

loss{�+=,m       �	z���ec�A�*

loss���<��R�       �	���ec�A�*

loss��!<D�       �	�"��ec�A�*

loss\�%<5�_       �	<���ec�A�*

lossV8X<��_�       �	T��ec�A�*

lossv0s<~�{�       �	cD��ec�A�*

loss$è=6,^�       �	Y���ec�A�*

lossE �<��_6       �	)w��ec�A�*

lossx��<��!�       �	���ec�A�*

loss�ϵ<���       �	o���ec�A�*

lossI%_=�       �	V��ec�A�*

loss �;�M�r       �	����ec�A�*

loss4��;�]�       �	}���ec�A�*

lossvz�<�t�       �	@��ec�A�*

loss���< 7M�       �	����ec�A�*

loss/�<u��       �	�l��ec�A�*

lossa�_=���5       �	L��ec�A�*

lossôF=�-a�       �	6���ec�A�*

loss�yg<�x��       �	�C��ec�A�*

lossna=$B��       �	$���ec�A�*

loss��<$v�)       �	�l��ec�A�*

loss�=�|
       �	d��ec�A�*

loss�V�;���       �	����ec�A�*

loss}��=ۮ;�       �	�4��ec�A�*

loss���<Z`~       �	����ec�A�*

loss�5�;�y��       �	;n��ec�A�*

loss3I?=���       �	���ec�A�*

lossn"'=,\�[       �	���ec�A�*

loss�5�;';       �	ԙ��ec�A�*

lossII�:�2;,       �	�,��ec�A�*

lossE�=*�B       �	����ec�A�*

loss�r<���       �	����ec�A�*

lossė,=CE��       �	�L��ec�A�*

loss���;���       �	$���ec�A�*

lossE�<N�܎       �	����ec�A�*

loss��;Z	�       �	%��ec�A�*

lossf��<�]�P       �	m��ec�A�*

loss�>=EGl�       �	��ec�A�*

loss��<�8V       �	����ec�A�*

lossnG3>�(�s       �	��ec�A�*

lossD�/=��k       �	h#��ec�A�*

loss�1�=��lS       �	8���ec�A�*

lossM%�;:b       �	�X��ec�A�*

loss,j�<l6�       �	^���ec�A�*

loss�]�=��Do       �	D���ec�A�*

loss�E�<V���       �	�y��ec�A�*

loss׹R;��4�       �	���ec�A�*

loss�x=���       �	Ѳ��ec�A�*

loss�hR;]�z�       �	IJ��ec�A�*

loss�_%=�5�       �	����ec�A�*

lossFh�<t�$}       �	����ec�A�*

loss��=ɲ\	       �	.���ec�A�*

loss�(= <]�       �	����ec�A�*

loss?�C<���]       �	D6��ec�A�*

lossXѢ<�P�9       �	����ec�A�*

loss��=)o�       �	t{��ec�A�*

loss3��<ՎF       �	���ec�A�*

loss���<i3-�       �	����ec�A�*

loss&F=�!C�       �	C��ec�A�*

loss̾�<���N       �	���ec�A�*

loss���<(��       �	�n��ec�A�*

loss�%9<К߳       �	!��ec�A�*

loss�>D<�n�G       �	R���ec�A�*

loss1)=�OO�       �	�,��ec�A�*

loss4�/=	>��       �	m���ec�A�*

loss��<��p       �	h��ec�A�*

loss��q<'딭       �	����ec�A�*

loss\�I<�gQ       �	)� �ec�A�*

loss���<.a�3       �	�(�ec�A�*

loss��J<�&V       �	j��ec�A�*

lossZ��;��_�       �	V�ec�A�*

loss��<���q       �	���ec�A�*

loss��<m�~       �	�}�ec�A�*

loss���=RN6I       �	��ec�A�*

loss�6�;R�!�       �	���ec�A�*

loss	�<G��S       �	2<�ec�A�*

lossh6=�~�       �	���ec�A�*

loss�Y<��       �	�i�ec�A�*

loss��:U���       �	��ec�A�*

loss���<5�5�       �	ʥ�ec�A�*

loss���<G�W�       �	:@�ec�A�*

lossC��;�y�       �	I��ec�A�*

lossT��:�k�2       �	1{	�ec�A�*

loss��<I;��       �	6
�ec�A�*

lossq�<�>K       �	4�
�ec�A�*

loss*�=�^�       �	GY�ec�A�*

loss���<�.�       �	3��ec�A�*

loss�l�<�﫠       �	J��ec�A�*

lossc�=^S�       �	�1�ec�A�*

loss�N;FƎ�       �	���ec�A�*

loss�4=i�8       �	a�ec�A�*

loss���=,~-&       �	Z��ec�A�*

lossq90:�       �	���ec�A�*

loss���:�I��       �	�ec�A�*

loss��;rJ�       �	��ec�A�*

loss�I;�+�1       �	O�ec�A�*

lossXz=>��       �	6��ec�A�*

loss��q;+�N�       �	��ec�A�*

loss]m:���       �	��ec�A�*

loss�'�<�}��       �	V��ec�A�*

loss}IK9���       �	
��ec�A�*

losstժ98�Tj       �	��ec�A�*

lossٷ�9�n�       �	��ec�A�*

loss��	=L��       �	�C�ec�A�*

loss� �;����       �	���ec�A�*

losss�;Z	       �	ˁ�ec�A�*

loss�+;�7��       �	��ec�A�*

lossR{�<-|��       �	.��ec�A�*

loss���=Uu�       �	�A�ec�A�*

loss@� ;��S�       �	Z��ec�A�*

loss��=n��        �	]l�ec�A�*

loss)��<�ܺ�       �	S�ec�A�*

loss���=��?       �	��ec�A�*

loss@�W=�w3�       �	=�ec�A�*

loss�<�<H�D       �	��ec�A�*

losse��=�5��       �	@l�ec�A�*

loss1<&=gl��       �	0G�ec�A�*

loss�}l=���       �	���ec�A�*

loss��W<Z�9I       �	�t�ec�A�*

loss�=h�X       �	5 �ec�A�*

loss��z=g       �	�� �ec�A�*

loss_��<��i�       �	m7!�ec�A�*

loss���<UwKM       �	��!�ec�A�*

lossYd=M���       �	��"�ec�A�*

loss���<17�	       �	��#�ec�A�*

loss:�^=gIw       �	��$�ec�A�*

lossF��<�('�       �	o-%�ec�A�*

loss=�m=%_�%       �	�%�ec�A�*

loss�<��͢       �	�p&�ec�A�*

loss��;5��]       �	�'�ec�A�*

loss�S�<��tb       �	��'�ec�A�*

lossB=�A�       �	UN(�ec�A�*

loss%V�<YfrC       �	��(�ec�A�*

loss��<���       �	'�)�ec�A�*

lossp\�;/M�       �	�i*�ec�A�*

loss�=�J	;       �	+�ec�A�*

lossVF;=~)�E       �	w�+�ec�A�*

loss��=��'k       �	�=,�ec�A�*

lossjXm=g	       �	/�-�ec�A�*

loss�Ct</7�b       �	�{.�ec�A�*

loss�f�<?��O       �	W%/�ec�A�*

loss��;蹯�       �	��0�ec�A�*

loss@��::�a       �	��1�ec�A�*

loss��<1���       �	�22�ec�A�*

loss��$<3�b       �	�2�ec�A�*

loss�8=/z       �	Wv3�ec�A�*

loss� �<�a�       �	�4�ec�A�*

loss8B=L���       �	ܺ4�ec�A�*

loss�b�<��%u       �	�R5�ec�A�*

lossD#<d�E       �	��5�ec�A�*

loss��<�;|�       �	-7�ec�A�*

loss) 1<�ĺp       �	�7�ec�A�*

loss�<�       �	W�8�ec�A�*

loss
��;�94�       �	/N9�ec�A�*

loss�U=���       �	ˁ:�ec�A�*

loss\i=SQ       �	,*;�ec�A�*

loss��<�W�       �	m�;�ec�A�*

loss�=�3�       �	L�<�ec�A�*

loss2Z�;�^�       �	��=�ec�A�*

loss�-3<��0       �	��>�ec�A�*

loss��<���'       �	N�W�ec�A�*

loss
�I=�6�/       �	�ZX�ec�A�*

loss�=Rw`w       �	�X�ec�A�*

loss��<�9F       �	U�Y�ec�A�*

loss��<���       �	Z�ec�A�*

loss�<=���       �	-�Z�ec�A�*

loss��=��
�       �	�M[�ec�A�*

lossM،=vA(2       �	d�[�ec�A�*

loss���=��;�       �	��\�ec�A�*

lossw�Z=�
}�       �	i]�ec�A�*

loss �l;�U�       �	��]�ec�A�*

loss<�::�       �	��^�ec�A�*

losst��<���
       �	5_�ec�A�*

loss慕=��_       �	2�_�ec�A�*

lossH��<��(�       �	Zb`�ec�A�*

lossFM�=��s�       �	.a�ec�A�*

lossoc�:� ,�       �	�a�ec�A�*

loss�=r<�J;[       �	3b�ec�A�*

losso��;%�       �	��b�ec�A�*

loss?��<p�       �	{gc�ec�A�*

loss%�;=7SL       �	d�ec�A�*

loss�K�=t�       �	��d�ec�A�*

loss�H�;����       �	�5e�ec�A�*

lossc��=u1�       �	�	g�ec�A�*

loss-ט<�QҠ       �	�g�ec�A�*

loss��<���       �	�Xh�ec�A�*

loss�E�;�[l!       �	?i�ec�A�*

loss�-�;OR�       �	�i�ec�A�*

loss�z0=_$��       �	Fj�ec�A�*

loss�ԟ=�_n       �	��j�ec�A�*

loss��;�wC       �	/�k�ec�A�*

loss,-�;��       �	�l�ec�A�*

loss�=�rů       �	�l�ec�A�*

loss[m<�mU�       �	�Fm�ec�A�*

loss�
<E+��       �	'�m�ec�A�*

loss�\�=xb�       �	Krn�ec�A�*

lossr��;�VP�       �	-o�ec�A�*

lossn.=Tȝ�       �	��o�ec�A�*

loss��;=t���       �	EHp�ec�A�*

loss��+>^ˤ�       �	�p�ec�A�*

lossA�2;)
Z       �	�vq�ec�A�*

loss���<��       �	*r�ec�A�*

loss��<[�7       �	%�r�ec�A�*

loss�ʀ<���U       �	0Is�ec�A�*

loss��<����       �		�s�ec�A�*

loss�F =�O       �	�st�ec�A�*

loss�&�<Ǟ�l       �	h	u�ec�A�*

loss\�(=^"2       �	�u�ec�A�*

loss��<���X       �	��v�ec�A�*

loss}�:<E��       �	ρw�ec�A�*

loss��p<;{�,       �	��x�ec�A�*

lossZ;?=��o       �	�ey�ec�A�*

loss��:g�B�       �	��z�ec�A�*

loss�.$>MHI�       �	�M{�ec�A�*

loss�M�<B��       �	�|�ec�A�*

loss �~:�Tu�       �	�|�ec�A�*

loss�8�:�O�r       �	��}�ec�A�*

loss�l <2"9p       �	��~�ec�A�*

loss߷[<�U�       �	ܟ�ec�A�*

loss��=�ޛ�       �	`;��ec�A�*

lossl%�=���&       �	�׀�ec�A�*

loss�i�;��(       �	|���ec�A�*

loss�O�;ʡ�.       �	�V��ec�A�*

lossHĠ=N��       �	���ec�A�*

loss� <���a       �	ō��ec�A�*

loss��<��-�       �	�&��ec�A�*

loss$��<!��E       �	�Ƅ�ec�A�*

loss_�=y�t�       �	�b��ec�A�*

loss!�=�m�       �	r���ec�A�*

lossjF�=	-�c       �	��ec�A�*

lossH�<K��       �	�8��ec�A�*

loss���;Տ��       �	�ч�ec�A�*

loss|4<v�B�       �	Ee��ec�A�*

loss6%C<�<�       �	\��ec�A�*

loss�$�;q���       �	ᚉ�ec�A�*

loss���<�(R]       �	�.��ec�A�*

loss�+�<{�Q       �	�Ί�ec�A�*

loss�
P<��+       �	�k��ec�A�*

lossW��<4�'�       �	@��ec�A�*

loss�)|=����       �	�S��ec�A�*

lossJ�=R���       �	F��ec�A�*

lossi�<fM�%       �	Z���ec�A�*

loss�p=ik$>       �	���ec�A�*

loss�H<I�       �	J���ec�A�*

loss#;�>.�       �	3R��ec�A�*

lossn��:	�        �	���ec�A�*

loss�(�=N�x       �	����ec�A�*

loss���<u���       �	�W��ec�A�*

loss8d!<<\
�       �	A���ec�A�*

lossx�=,�L�       �	"���ec�A�*

loss�v=%��       �	(,��ec�A�*

loss}ߵ;p�yF       �	��ec�A�*

loss�ӽ:�t�       �	y[��ec�A�*

lossd�<O_�       �	0���ec�A�*

loss��n<��5q       �	/���ec�A�*

lossD��;�Ij       �	�.��ec�A�*

loss3�=Q��       �	pИ�ec�A�*

loss�o�;z�:�       �	}��ec�A�*

loss��=�3�       �	{��ec�A�*

loss�ET;�l[�       �	����ec�A�*

loss.�;.GwE       �	�U��ec�A�*

lossm;6秿       �	$��ec�A�*

loss7N�;�yN�       �	����ec�A�*

loss$�G<0]�       �	�8��ec�A�*

loss�c0<h$�       �	ݝ�ec�A�*

loss��=#�w        �	 z��ec�A�*

loss���;J��       �	���ec�A�*

loss��J<?NCa       �	����ec�A�*

loss���<�J(�       �	>��ec�A�*

loss3�Z=
���       �	�ޠ�ec�A�*

loss�UB=�-k�       �	�z��ec�A�*

loss�^z=*�ܡ       �	���ec�A�*

loss=�=�a       �	����ec�A�*

loss��H<��q       �	:@��ec�A�*

lossdI�<��w       �	�ӣ�ec�A�*

lossǄ<j���       �	�j��ec�A�*

loss=ѕ;��J�       �	\���ec�A�*

loss��<�a�J       �	���ec�A�*

loss;�<=��       �	�.��ec�A�*

loss���;���       �	�æ�ec�A�*

lossUt<�p��       �	����ec�A�*

loss�\�=yE'C       �	�P��ec�A�*

loss�?U=�<�       �	���ec�A�*

loss}�<�n�       �	����ec�A�*

loss4
.<d�"       �	�#��ec�A�*

loss��<=�+�       �	���ec�A�*

loss�<���       �	�~��ec�A�*

loss�O'=�:�       �	3��ec�A�*

loss��=|��q       �	����ec�A�*

loss.�<��-       �	J_��ec�A�*

loss�L�<���       �	����ec�A�*

loss���<��\z       �	��ec�A�*

loss��=ґV�       �	,��ec�A�*

loss�L<�qb�       �		ï�ec�A�*

loss�
=ht`�       �	�b��ec�A�*

loss$H*<uB#       �	����ec�A�*

lossJ��<h���       �	����ec�A�*

loss�*�<?6s       �	�.��ec�A�*

lossE�6<:_�?       �	3Ĳ�ec�A�*

lossA�D<��t�       �	�Y��ec�A�*

lossɇ�:�       �	����ec�A�*

loss}z�=E�
�       �	����ec�A�*

loss�d�<��       �	ʊ��ec�A�*

lossQ�=��a�       �	���ec�A�*

loss���<�8*K       �	���ec�A�*

lossS�?=T��       �	����ec�A�*

loss,kZ;D�S9       �	�]��ec�A�*

loss��d;t��<       �	����ec�A�*

lossiא<5�       �	���ec�A�*

losst0=Ov]       �	�%��ec�A�*

loss�n/;0�i       �	���ec�A�*

loss��<-,�h       �	PS��ec�A�*

loss�'E<�.M       �	K��ec�A�*

losss?=�V       �	�y��ec�A�*

loss���<˧��       �	�!��ec�A�*

loss��$;�v       �	F���ec�A�*

loss�N�=2_�R       �	�L��ec�A�*

loss�X�=?�Z�       �	*��ec�A�*

loss
4=��q�       �	����ec�A�*

loss�>l<m3��       �	���ec�A�*

loss,�S<f9��       �	����ec�A�*

loss��G;����       �	R��ec�A�*

loss��9=�1��       �	c���ec�A�*

loss�2<+b       �	���ec�A�*

loss�V�=2/�       �	`���ec�A�*

loss���9��I�       �	�/��ec�A�*

loss{h<��d       �	o���ec�A�*

loss���< ��       �	 s��ec�A�*

lossMZD;��	H       �	�	��ec�A�*

loss�G=�$\�       �	L���ec�A�*

loss�D?=o�0�       �	?��ec�A�*

loss�N<�J�       �	���ec�A�*

loss��<N��       �	 z��ec�A�*

loss�_=L�l�       �	���ec�A�*

lossO
�;���}       �	I���ec�A�*

loss�$�;�,�$       �	�d��ec�A�*

loss�)@<8��       �	���ec�A�*

loss}�;v�       �	^���ec�A�*

loss@
>=��f�       �	�A��ec�A�*

lossH��:�o)�       �	����ec�A�*

loss�:v�d�       �	�z��ec�A�*

loss��<����       �	��ec�A�*

loss��	<�80!       �	[���ec�A�*

lossړV<��Ə       �	���ec�A�*

loss�Y)<}s�k       �	{1��ec�A�*

loss��F<M���       �	p���ec�A�*

loss�=�%)�       �	J���ec�A�*

lossA"8;	ݙ�       �	���ec�A�*

lossw�<�bn�       �	0-��ec�A�*

lossHDz<���       �	z���ec�A�*

loss�
>��
=       �	�Z��ec�A�*

loss���<��       �	���ec�A�*

loss#��;6��7       �	ڎ��ec�A�*

loss�}�<�|y*       �	$��ec�A�*

lossB�":B-�       �	k���ec�A�*

loss��<�쇫       �	:X��ec�A�*

loss=��;mSlG       �	����ec�A�*

loss�-#<�k�       �	R��ec�A�*

loss�E-=��Z       �	���ec�A�*

loss�f<Ȋ3       �	�v��ec�A�*

loss���<ZFu�       �	��ec�A�*

lossʹ�;%,��       �	���ec�A�*

loss�� <)�R�       �	�>��ec�A�*

lossZ��<<B       �	R���ec�A�*

loss�7<~�ˮ       �	�f��ec�A�*

loss�<��<,       �	����ec�A�*

loss1�=���       �	����ec�A�*

loss���=���h       �	�*��ec�A�*

loss79=�{�       �	����ec�A�*

loss�
>U�       �	�V��ec�A�*

loss\��;D�D+       �	����ec�A�*

loss,qm<0���       �	ޑ��ec�A�*

lossD�,=�Rli       �	�)��ec�A�*

loss$0=�C>       �	���ec�A�*

loss�<o�       �	�T��ec�A�*

loss%��=ȇ��       �	����ec�A�*

loss��=Xh�       �	����ec�A�*

lossJo�=-�X�       �	���ec�A�*

loss��=-MsE       �	����ec�A�*

lossX�=��       �	rN��ec�A�*

lossZ��<3��,       �	���ec�A�*

loss���;��~       �	�z��ec�A�*

loss��V<�6^<       �	���ec�A�*

loss�.=U��3       �	n���ec�A�*

loss�{�<�m;2       �	�P��ec�A�*

lossw�2=_\�       �	����ec�A�*

lossM۔<]Z       �	lx��ec�A�*

lossW'�=���       �	k��ec�A�*

loss@<�F�%       �	w���ec�A�*

loss�e<���g       �	v7��ec�A�*

loss <�i��       �	!���ec�A�*

lossӘ�=�Bya       �	fi��ec�A�*

loss%p*<����       �	i���ec�A�*

loss�E=D��       �	p���ec�A�*

loss85�;3ux       �	]1��ec�A�*

lossŢ=	���       �	���ec�A�*

lossl�&<�U�L       �	�_��ec�A�*

loss��=�M6       �	� ��ec�A�*

loss�K=e�`P       �	)���ec�A�*

loss� )<�� P       �	�3��ec�A�*

lossV0�;�0��       �	����ec�A�*

loss���<�s��       �	�l��ec�A�*

loss���<5V}�       �	A��ec�A�*

lossT4�<�]X       �	,��ec�A�*

loss��B=Yp��       �	e���ec�A�*

loss8��<5�       �	�e��ec�A�*

loss�C<w25x       �	����ec�A�*

loss7<���       �	����ec�A�*

lossL��<�\��       �	�`��ec�A�*

loss��9=��A�       �	�1��ec�A�*

lossi�g=�K�?       �	���ec�A�*

lossr��=J;9       �	0J��ec�A�*

loss��=08�       �	�!��ec�A�*

loss��7;�.��       �	����ec�A�*

loss�,;&r9       �	��ec�A�*

loss�C�:���       �	���ec�A�*

loss{FZ=1k�       �	�y�ec�A�*

loss�'(<�f�        �	0�ec�A�*

lossLk<O��6       �	���ec�A�*

loss��=�W%l       �	C��ec�A�*

loss�U-<j'�       �	B��ec�A�*

loss|8�=^ 	       �	=�ec�A�*

loss8�<��9       �	#�ec�A�*

lossX(&=� ,       �	���ec�A�*

loss[��<(��P       �	=�	�ec�A�*

loss珘;��       �	��
�ec�A�*

loss��P=T�       �	���ec�A�*

loss��>=G���       �	_^�ec�A�*

loss��`=ם�       �	�E�ec�A�*

loss�R=��u�       �	<��ec�A�*

loss��x<=�'       �	~n�ec�A�*

loss݉<��ğ       �	.7�ec�A�*

loss�b�<��       �	��ec�A�*

loss�9L=��E       �	��ec�A�*

loss �=8�M       �	�y�ec�A�*

loss�,�<���u       �	W��ec�A�*

loss�H%<�k
7       �	���ec�A�*

loss��;;'͐       �	]j�ec�A�*

loss#M>���       �	���ec�A�*

loss$oX=�:       �	�A�ec�A�*

loss��<-��       �	%�ec�A�*

lossIV�<��M�       �	W=�ec�A�*

loss���<ce-       �	�V�ec�A�*

lossS�=)-@       �	��ec�A�*

loss*j=mYSJ       �	��ec�A�*

lossu�=��b6       �	Q��ec�A�*

lossH�<$��X       �	�;�ec�A�*

loss4��;&�Q.       �	���ec�A�*

loss4��=��W&       �	���ec�A�*

loss���<}r�       �	h� �ec�A�*

loss���:���g       �	<�!�ec�A�*

loss��f<���Z       �	W|"�ec�A�*

loss�Z2<��R       �	�#�ec�A�*

lossOkS;3.��       �	7�#�ec�A�*

loss%T�;cr-u       �	��$�ec�A�*

lossM�=-B!Y       �	�%%�ec�A�*

lossXr=�Z       �	��%�ec�A�*

loss: �=�k��       �	Bv&�ec�A�*

loss��3=�̌�       �	s'�ec�A�*

loss��U=}�MM       �	l%(�ec�A�*

loss�\�<�o}i       �	��(�ec�A�*

loss�X�=yN]       �	G�)�ec�A�*

loss/f=1 �       �	��*�ec�A�*

loss[��=���E       �	{+�ec�A�*

loss�m�=�0�       �	o/,�ec�A�*

lossNL�=�~K�       �	k�,�ec�A�*

lossO(�;��8�       �	*t-�ec�A�*

loss}��=���       �	�.�ec�A�*

loss�$�;U�       �	<�.�ec�A�*

loss��<�       �	�=/�ec�A�*

loss�|�<��8�       �	��/�ec�A�*

lossmcK=���9       �	�t0�ec�A�*

lossjIt=�9��       �	�1�ec�A�*

lossMf]<-�}*       �	ũ1�ec�A�*

lossm�e=m^       �	=2�ec�A�*

lossT��=��S�       �	��2�ec�A�*

loss�nV<.݁�       �	Dl3�ec�A�*

loss�CP<�RN�       �	d4�ec�A�*

loss��]<Ux|       �	 �4�ec�A�*

loss�o�<��͞       �	|,5�ec�A�*

loss�nh<��x       �	$�5�ec�A�*

loss�:�<��(S       �	jj6�ec�A�*

loss�b;=�~zg       �	�U7�ec�A�*

loss��<0\��       �	0�7�ec�A�*

lossD?�<#�       �	�8�ec�A�*

loss@qW=�       �	Q19�ec�A�*

loss�(0<��l�       �	��9�ec�A�*

lossR�4=B�       �	�r:�ec�A�*

loss���<�ڢ�       �	A;�ec�A�*

lossv�=ү��       �	ڨ;�ec�A�*

loss6J4<��An       �	C<�ec�A�*

loss�ѳ;�]�       �	��<�ec�A�*

loss�a�<3b�       �	�p=�ec�A�*

lossk <'�D       �	�>�ec�A�*

loss_�\<��W       �	��>�ec�A�*

loss!t�<�Pw�       �	��?�ec�A�*

loss߼)=_�       �	�3@�ec�A�*

loss1�`<�gE�       �	0�@�ec�A�*

lossin=m���       �	qtA�ec�A�*

lossH�:/ !,       �	�B�ec�A�*

loss��;Tv�       �	_�B�ec�A�*

loss!=Y3��       �	�RC�ec�A�*

loss<!�<��       �	��C�ec�A�*

loss=��<l�/       �	ގD�ec�A�*

loss��=�d��       �	�1E�ec�A�*

loss׿�<8�#       �	��E�ec�A�*

loss���<���       �	@jF�ec�A�*

loss�f�<��       �	GG�ec�A�*

loss��K=�g�a       �	��G�ec�A�*

loss��^=כ��       �	-BH�ec�A�*

loss ϰ<��^�       �	��H�ec�A�*

loss�d�<���       �	�wI�ec�A�*

loss2V=#A��       �	�J�ec�A�*

loss��<LM]�       �	=�J�ec�A�*

loss3>�=ԩ�.       �	�mK�ec�A�*

losss�<(Y2       �	�	L�ec�A�*

loss�%�;&b��       �	(�L�ec�A�*

lossu�<j��       �	6M�ec�A�*

loss<���       �	DMN�ec�A�*

loss#�=�c       �	��N�ec�A�*

losscy�;�u�       �	q�O�ec�A�*

loss��;<��q       �	�"P�ec�A�*

loss�{<\��/       �	J�P�ec�A�*

lossL�!=�K7m       �	eQQ�ec�A�*

loss#�!=�f0i       �	��Q�ec�A�*

loss�?�<���b       �	��R�ec�A�*

lossG��<�ZQd       �	 %S�ec�A�*

loss�M<����       �	��S�ec�A�*

loss@$�<�B       �	�VT�ec�A�*

loss�&<9궠       �	:�T�ec�A�*

loss�Y;;��       �	X�U�ec�A�*

loss��={��       �	�$V�ec�A�*

loss��=�D�F       �	>�V�ec�A�*

loss]~=�`�       �	�TX�ec�A�*

loss��;{X�       �	��X�ec�A�*

loss��><W��       �	@�Y�ec�A�*

loss8"�<N"L       �	N&Z�ec�A�*

loss�1=ԉ9       �	�Z�ec�A�*

loss�=��շ       �	�j[�ec�A�*

losseJ�<cq�       �	�[�ec�A�*

loss�_=}�L�       �	�\�ec�A�*

loss�q<���       �	�>]�ec�A�*

lossc(�=�pvD       �	��]�ec�A�*

loss��<Ђ�,       �	~t^�ec�A�*

loss�;�W��       �	O_�ec�A�*

loss�o`=d�BM       �	��_�ec�A�*

loss�o< �-�       �	�A`�ec�A�*

loss��<��?V       �	��`�ec�A�*

loss�֝:�:�p       �	-za�ec�A�*

loss�:��8�       �	b�ec�A�*

loss3X�<��vg       �	��b�ec�A�*

loss��<'>r�       �	Adc�ec�A�*

lossN4<v�       �	d�ec�A�*

loss���;�>�_       �	T�d�ec�A�*

loss�+p<�f�'       �	zQe�ec�A�*

lossΓ�;�/�       �	��e�ec�A�*

loss%�;�jw�       �	7�f�ec�A�*

loss��<���       �	5*g�ec�A�*

loss�s;w�,6       �	H�g�ec�A�*

loss�;�j�T       �	�ah�ec�A�*

loss}>k<e�oD       �	s�h�ec�A�*

loss$	�<�8�       �	%�i�ec�A�*

loss:��<�9-1       �	*j�ec�A�*

loss��;��Q9       �	��j�ec�A�*

loss�(�=	�3�       �	�\k�ec�A�*

lossL�[=�O�G       �	��k�ec�A�*

loss��F<�]��       �	��l�ec�A�*

loss]g=[�Tz       �	!m�ec�A�*

loss���;Vt�       �	:n�ec�A�*

lossy_�=����       �	��n�ec�A�*

lossw;�<Ui�       �	�do�ec�A�*

loss�m=��U�       �	Y�o�ec�A�*

loss�Z�;=�v       �	`�p�ec�A�*

lossC��=�g�       �	�/q�ec�A�*

lossW�P<A D       �	.�q�ec�A�*

loss��=�a       �	��r�ec�A�*

loss*7�;�$s=       �	Hps�ec�A�*

lossT��;�[��       �	%t�ec�A�*

loss�<�e��       �	.�t�ec�A�*

loss�9n=7�'       �	�_u�ec�A�*

lossQHO;8m��       �	�u�ec�A�*

loss���<�k+A       �	�v�ec�A�*

lossX�<�[��       �	2w�ec�A�*

lossd��=QK�       �	i�w�ec�A�*

loss��S=QZ]0       �	ɓx�ec�A�*

loss��
=�b~�       �	>�y�ec�A�*

loss���<[�<�       �	��z�ec�A�*

loss� ?<m�       �	+�{�ec�A�*

loss�:�J�       �	`W|�ec�A�*

lossw)�=�D��       �	�	}�ec�A�*

loss�^�<�1ӹ       �	Ҧ}�ec�A�*

lossl�5=�,�       �	4�~�ec�A�*

losstL�:iK��       �	&m�ec�A�*

loss���<2��       �	���ec�A�*

loss�2�<ٳ�       �	n���ec�A�*

lossoz�<���       �	�8��ec�A�*

loss���;P�L       �	�́�ec�A�*

loss���<��y'       �	ga��ec�A�*

loss��^<$�?�       �	b���ec�A�*

loss���<V���       �	����ec�A�*

loss�l=2L�       �	O#��ec�A�*

loss���=�>2b       �	z�ec�A�*

loss�t�<���       �	Ad��ec�A�*

loss���=�jó       �	���ec�A�*

lossA�"=Ru^�       �	j���ec�A�*

lossd^<����       �	�B��ec�A�*

loss�=t��       �	#ڇ�ec�A�*

loss���<�(�6       �	s��ec�A�*

lossf��<i��       �	�
��ec�A�*

loss�G=pj�[       �	����ec�A�*

loss >-_��       �	�7��ec�A�*

lossS";�N#�       �	�Ҋ�ec�A�*

loss/@�<�&�1       �	�g��ec�A�*

loss[�=��d�       �	����ec�A�*

loss�%=)2?Y       �	˞��ec�A�*

loss%=͜�4       �	�7��ec�A�*

loss�l<�u��       �	�ԍ�ec�A�*

loss��H<t�c�       �	�t��ec�A�*

loss�<Gt�       �	�&��ec�A�*

lossAq�=X�AV       �	=���ec�A�*

loss�%�;�{e�       �	�N��ec�A�*

loss��<�L�       �	D���ec�A�*

loss�y�;���       �	����ec�A�*

lossC�=>W�H       �	
K��ec�A�*

loss���<��.       �	e��ec�A�*

lossO!=E�4�       �	1~��ec�A�*

loss%�6;[z�       �	���ec�A�*

loss�g�<5�"�       �	�L��ec�A�*

loss��;3�ds       �	F��ec�A�*

loss�Ö;�#�       �	����ec�A�*

losst��=�}h�       �	g&��ec�A�*

loss�թ<�{�       �	0���ec�A�*

loss���<�f�       �	�X��ec�A�*

lossL�<8�       �	���ec�A�*

loss��,<N��O       �	V���ec�A�*

loss��; T��       �	�֛�ec�A�*

loss�<^���       �	�p��ec�A�*

loss��e<���       �	$��ec�A�*

loss��<g�ɷ       �	ޭ��ec�A�*

loss�JC=�Ƞ�       �	F��ec�A�*

loss�fB=Ski       �	Lݞ�ec�A�*

lossq�O<>�       �	Ԝ��ec�A�*

loss��o<+�       �	�3��ec�A�*

loss�b!=�s�>       �	�>��ec�A�*

loss�X=�09O       �	�ߢ�ec�A�*

lossn#�;�W��       �	�w��ec�A�*

lossQ<ϝ��       �	���ec�A�*

loss��5=�N�&       �	����ec�A�*

loss��<��%�       �	0L��ec�A�*

loss)�=#�h�       �	���ec�A�*

loss�2�<W�a�       �	����ec�A�*

loss?L<��9�       �	�&��ec�A�*

loss��;*0�       �	����ec�A�*

loss��=�C�       �	�S��ec�A�*

loss�v�;uV#�       �	1��ec�A�*

loss�R�=�o��       �	I���ec�A�*

loss��e<�?�       �	���ec�A�*

loss�:;v�       �	>���ec�A�*

loss���<9$&       �	�I��ec�A�*

lossV��<9�׍       �	�߫�ec�A�*

loss/ =�6�       �	�y��ec�A�*

loss�;        �	H5��ec�A�*

lossj,�<����       �	ڭ�ec�A�*

loss�`<f���       �	�u��ec�A�*

loss�]�=;@]�       �	���ec�A�*

loss��<��9o       �	6���ec�A�*

lossI5=26OZ       �	�V��ec�A�*

lossd�=~ �Y       �	����ec�A�*

lossS�;m���       �	o���ec�A�*

loss��;�y:       �	X;��ec�A�*

loss� =*(�       �	�߲�ec�A�*

lossd�<A#Ik       �	�x��ec�A�*

loss�v�<Ua�       �	逵�ec�A�*

lossO�=�LZ       �	�.��ec�A�*

loss��;� ξ       �	�Ƕ�ec�A�*

loss쁍<��p}       �	#e��ec�A�*

lossR2=(.;"       �	�ec�A�*

lossj)9<0$�       �	^J��ec�A�*

loss�
<�yg�       �	N%��ec�A�*

loss�O=_]�       �	�z��ec�A�*

lossL�;Ƣ!�       �	$E��ec�A�*

loss�%#=��|       �	�'��ec�A�*

lossT`:�rM�       �	߿�ec�A�*

loss�t<�X2       �	U���ec�A�*

loss�l
=Yp��       �	ӽ��ec�A�*

lossl��;���I       �	`��ec�A�*

loss�ִ<�aƧ       �	���ec�A�*

loss�=A�W       �	���ec�A�*

lossN�R=h�g�       �	+N��ec�A�*

lossH�:��       �	����ec�A�*

loss�|<=�5�       �	a���ec�A�*

loss�fm=B�P�       �	(��ec�A�*

loss�IO<OY��       �	����ec�A�*

loss��:DZO�       �	o��ec�A�*

lossIfw:�o-�       �	p��ec�A�*

lossr�:��L       �	���ec�A�*

loss��;怙e       �	ӽ��ec�A�*

loss4��<�F�       �	����ec�A�*

loss�.�:JeG^       �	�Y��ec�A�*

loss�s�;s.�       �	���ec�A�*

loss��9l�[i       �	j���ec�A�*

loss�v8�n�       �	�D��ec�A�*

loss;w8��X       �	����ec�A�*

lossF<uZ       �	i���ec�A�*

loss�þ;�ӄ�       �	�/��ec�A�*

loss��;�	       �	y��ec�A�*

loss<1�9U<:3       �	�)��ec�A�*

lossOE�;��4�       �	����ec�A�*

loss$��=�1�       �	�l��ec�A�*

loss�w�:2� �       �	2��ec�A�*

lossZ,S>�y�       �	,���ec�A�*

loss��G<��S�       �	����ec�A�*

lossT�=��"�       �	�1��ec�A�*

lossHr�<C��v       �	����ec�A�*

loss̭�;U�       �	���ec�A�*

lossT۰=��       �	���ec�A�*

loss:h= �()       �	����ec�A�*

loss��c=���       �	����ec�A�*

loss��e<*��       �	k���ec�A�*

loss{h�; ��       �	�,��ec�A�*

loss�B�<E��       �	n���ec�A�*

lossL�=��W       �	A���ec�A�*

loss X=j$��       �	zQ��ec�A�*

loss��S=K�c�       �	=���ec�A�*

loss���<�ؒ�       �	F���ec�A�*

lossc�t=�,�       �	h?��ec�A�*

loss�=@n�:       �	\>��ec�A�*

loss!�?=7�ߓ       �	N���ec�A�*

loss��<��5       �	���ec�A�*

loss/m	<����       �	�M��ec�A�*

lossW�<���       �	e��ec�A�*

lossq��<1�)�       �	(��ec�A�*

loss�^�:��       �	���ec�A�*

loss\��;Y��       �	jg��ec�A�*

lossɁ<Nky       �	8��ec�A�*

loss�N<6[�%       �	���ec�A�*

loss�Ð;EN�1       �	2r��ec�A�*

loss]�c=w��       �	� ��ec�A�*

loss6�.=�pk       �	����ec�A�*

loss���<� ߹       �	�y��ec�A�*

loss���<�~Q�       �	�"��ec�A�*

losspo�;t�O       �	l���ec�A�*

loss V6;@�b�       �	����ec�A�*

loss�m�;���       �	�\��ec�A�*

loss-�=�դ�       �	���ec�A�*

loss?~z<�B       �	b���ec�A�*

loss�4<,t\       �	�f��ec�A�*

loss�=e=��       �	:��ec�A�*

loss��:�&.       �	֧��ec�A�*

loss</U<9
c       �	�J��ec�A�*

loss�J-=b��       �	����ec�A�*

loss8��;��Ϗ       �	s���ec�A�*

loss<�<N���       �	���ec�A�*

lossFk=��!�       �	=���ec�A�*

loss�Q =�6X       �	�S��ec�A�*

loss�?i=,��W       �	����ec�A�*

lossz�N=���       �	����ec�A�*

lossV�=��R�       �	 &��ec�A�*

loss���;��       �	���ec�A�*

loss�4�;�,��       �	2w��ec�A�*

loss�n�;;6��       �	�~�ec�A�*

loss�<�ݞ�       �	?�ec�A�*

loss�Ǆ= ��(       �	R��ec�A�*

loss:�E<�       �	�M�ec�A�*

loss�=v+=]       �	��ec�A�*

loss���:W��[       �	�|�ec�A�*

loss&8�;���1       �	Y�ec�A�*

loss�=j�N�       �	��ec�A�*

lossIϠ=�z�        �	���ec�A�*

loss�� <��-�       �	���ec�A�*

loss��<�Gu�       �	��ec�A�*

loss?<�2��       �	bM�ec�A�*

loss�tz=\�h;       �	��ec�A�*

loss�u<[d�       �	���ec�A�*

loss�Oo<�;ԭ       �	�e�ec�A�*

loss���;�Qԙ       �	�ec�A�*

loss�V9�	       �	���ec�A�*

loss�l�<	+h�       �	ds�ec�A�*

loss�-�<�&�N       �	c�ec�A�*

loss���;�ᇮ       �	v�ec�A�*

loss/[�;�$�Z       �	V��ec�A�*

loss�ʣ=�k�       �	�U�ec�A�*

loss7��;<�P       �	R��ec�A�*

loss���=o�)�       �	ʋ�ec�A�*

loss��U;j�ڶ       �	�"�ec�A�*

loss�q;$^       �	���ec�A�*

loss�-?=�b��       �	�V �ec�A�*

loss�1?<�2��       �	�� �ec�A�*

loss��X<^P��       �	��!�ec�A�*

loss�E<��:�       �	�"�ec�A�*

loss���;͍��       �	��"�ec�A�*

losszo�;���       �	�k#�ec�A�*

loss���;6�a       �	�$�ec�A�*

lossN�<� ��       �	_�$�ec�A�*

loss��<�s&�       �	�2%�ec�A�*

loss@��<���       �	[�%�ec�A�*

lossa�Q;��m�       �	�l&�ec�A�*

loss߷�<&�7�       �	_'�ec�A�*

loss�ɤ=�XbF       �	��'�ec�A�*

loss�$�<��Z        �	qU(�ec�A�*

loss�{=�{       �	��(�ec�A�*

lossJ��<]���       �	��)�ec�A�*

loss�Z�;��G       �	�>*�ec�A�*

loss�^h=�h��       �	��*�ec�A�*

loss��;<���g       �	�|+�ec�A�*

loss�i�=ZZ       �	D1,�ec�A�*

loss��
<W���       �	��,�ec�A�*

loss��<���       �	�i-�ec�A�*

lossS�n;��P       �	.�ec�A�*

loss;Ov:
��       �	D�.�ec�A�*

lossm;;�E       �	�;/�ec�A�*

loss�}�=/T��       �	g�/�ec�A�*

lossᄄ<ѐ�       �	ir0�ec�A�*

lossEʇ=^*��       �	�
1�ec�A�*

loss���<��       �	T�1�ec�A�*

loss�C�;>M�       �	Fz2�ec�A�*

loss�!:V�U
       �	R3�ec�A�*

loss�~<��\       �	�3�ec�A�*

loss���;|�0O       �	�64�ec�A�*

loss��<=���       �	��4�ec�A�*

loss���=h4�       �	�c5�ec�A�*

loss��Q;�Ʃ�       �	��5�ec�A�*

loss&N�;՞$       �	֌6�ec�A�*

loss>+=KW�       �	t$7�ec�A�*

lossE	�<Iu�       �	Q�7�ec�A�*

loss��>;���n       �	�P8�ec�A�*

loss�n8=�/}G       �	9�ec�A�*

loss[�;2*�       �	��9�ec�A�*

loss ��=f�	       �	�;�ec�A�*

loss��J<
�$X       �	=�;�ec�A�*

loss���;���I       �	�B<�ec�A�*

loss�p�<��f       �	��<�ec�A�*

loss�U�<���1       �	w�=�ec�A�*

lossx<ؤ;D       �	�>�ec�A�*

loss�<�i+N       �	�>�ec�A�*

loss�;���:       �	PS?�ec�A�*

loss��<�Cvl       �	�?�ec�A�*

loss���<���7       �	M�@�ec�A�*

loss�Z�<TqL       �	�"A�ec�A�*

loss��]<��^3       �	ظA�ec�A�*

loss��m;mWň       �	fNB�ec�A�*

loss�f�</%       �	�B�ec�A�*

lossQ�P<��       �	ҎC�ec�A�*

loss7�;���       �	�/D�ec�A�*

loss�}�;�J       �	%�D�ec�A�*

loss�G:�ؒ       �	�eE�ec�A�*

loss�1�<�m1       �	F�ec�A�*

loss��{<ۦ�       �	��F�ec�A�*

loss�d<+Fp*       �	FCG�ec�A�*

loss� <Y�       �	*�G�ec�A�*

loss	{=N��       �	RH�ec�A�*

lossݙ=��{       �	LI�ec�A�*

lossM��;r���       �	��I�ec�A�*

loss u�<��je       �	UJ�ec�A�*

loss��D=�|[�       �	o�J�ec�A�*

loss3�G;Z�$�       �	ҏK�ec�A�*

lossiY�;_�4X       �	�DL�ec�A�*

loss�`=4��
       �	��L�ec�A�*

loss��6=8׍       �	��M�ec�A�*

loss�;�&��       �	N�ec�A�*

loss�=A"��       �	ȲN�ec�A�*

lossCl{<��;�       �	�IO�ec�A�*

lossl��;�B��       �	?�O�ec�A�*

lossM��<���]       �	�zP�ec�A�*

loss/I(<9�B?       �	�Q�ec�A�*

loss��=�6��       �	X�Q�ec�A�*

loss�S;<���       �	�?R�ec�A�*

loss�j!;��L       �	�XS�ec�A�*

lossZ8�<Ū��       �	��S�ec�A�*

loss82<���8       �	܄T�ec�A�*

loss���;D
Bk       �	�U�ec�A�*

loss��
=����       �	��U�ec�A�*

loss�\]=����       �	wV�ec�A�*

loss��_9�yр       �	nW�ec�A�*

loss��<I6�       �	��W�ec�A�*

loss[�\=��>=       �	gGX�ec�A�*

loss.�F=��b       �	[�X�ec�A�*

loss{�6=��u�       �	��Y�ec�A�*

loss�Ϫ;G�1m       �	pAZ�ec�A�*

loss��<��"|       �	M�Z�ec�A�*

loss�]�;'*�       �	s[�ec�A�*

loss@��<�q�       �	0\�ec�A�*

lossO٣<y u�       �	�\�ec�A�*

loss�<T=%Ѵ       �	5@]�ec�A�*

lossDo�9C�
�       �	��]�ec�A�*

loss���=���       �	Q�^�ec�A�*

lossdQ<�*9       �	�__�ec�A�*

lossz6�;ⵐ�       �	\�_�ec�A�*

loss�;�<�YI�       �	o�`�ec�A�*

loss�=1GA�       �	*Va�ec�A�*

loss  =JΊS       �	��a�ec�A�*

loss�L�<=��:       �	��b�ec�A�*

loss;�P=;��q       �	(Gc�ec�A�*

lossն=��       �	��c�ec�A�*

loss�Q�=�*:6       �	Ûd�ec�A�*

loss)�=w�       �	�@e�ec�A�*

loss���<j:�       �	K�e�ec�A�*

loss�Z;/�I�       �	}�f�ec�A�*

loss*4O;-]$       �	�4g�ec�A�*

loss�e^<�k�       �	w�g�ec�A�*

loss��<9l�       �	yh�ec�A�*

loss���;Z/`       �	�*i�ec�A�*

loss�[�;V,T       �	2�i�ec�A�*

lossM6;n��Y       �	3pj�ec�A�*

losse�=�2p       �	�k�ec�A�*

loss���<��|�       �	�k�ec�A�*

loss��<'S       �	~Wl�ec�A�*

loss��9�:u�       �	gHm�ec�A�*

loss�֋<A��       �	��m�ec�A�*

lossE�4;Z��+       �	�n�ec�A�*

loss�+�<	���       �	�%o�ec�A�*

loss�c�;�	[       �	=�o�ec�A�*

loss��<Xq�=       �	f�p�ec�A�*

loss!�<�F�o       �	�,q�ec�A�*

loss�!=�l�       �	��q�ec�A�*

loss7�;��       �	��r�ec�A�*

lossH��<"B��       �	�Js�ec�A�*

loss���<0�!�       �	%u�ec�A�*

lossm0�=��C       �	�u�ec�A�*

loss><��w       �	�[v�ec�A�*

loss_ =�Ǯ       �	uw�ec�A�*

loss)T!;�܍�       �	2�w�ec�A�*

loss��=	=��       �	�0y�ec�A�*

lossMA(:���       �	��y�ec�A�*

loss^C"=ϲ�       �	�z�ec�A�*

loss?i�:~o       �	
g{�ec�A�*

loss��t<��N       �	�T|�ec�A�*

lossx`�;�ۍ       �	e8}�ec�A�*

losst��;��	z       �	f~�ec�A�*

losso�;���       �	nQ�ec�A�*

lossZ=ׇa�       �	n��ec�A�*

loss���;��th       �	�ƀ�ec�A�*

lossxS�:adD�       �	���ec�A�*

loss�MV=wIr�       �	��ec�A�*

loss(:��       �	����ec�A�*

loss�eF;"���       �	P��ec�A�*

loss1�p=�       �	ж��ec�A�*

loss��T=�&�S       �	a��ec�A�*

loss��<#�       �	5
��ec�A�*

loss���;,f)       �	����ec�A�*

loss_��:?{�       �	�Q��ec�A�*

lossv�a=:��       �	����ec�A�*

losso*�<�US�       �	���ec�A�*

loss�{H<kD��       �	|H��ec�A�*

loss�4K:�(.       �	���ec�A�*

loss��:4�h�       �	���ec�A�*

losso-�=*�ܷ       �	�.��ec�A�*

loss�<x�       �	�ь�ec�A�*

loss��L<Xbi�       �	�s��ec�A�*

loss��w<��       �	���ec�A�*

lossL�A<�a��       �	,-��ec�A�*

lossJ)<�j��       �	ڐ�ec�A�*

loss��<���       �	���ec�A�*

loss�7�=����       �	o)��ec�A�*

lossV��=��x       �	�Ӓ�ec�A�*

loss\�8<!���       �	�|��ec�A�*

loss;'�:��#       �	9*��ec�A�*

loss��h<ߧ$       �	�Ք�ec�A�*

loss�vS=�_fe       �	�z��ec�A�*

loss �;N       �	�)��ec�A�*

lossDC;8S�?       �	Ֆ�ec�A�*

loss15�<&|       �	9}��ec�A�*

loss���<�P�       �	�*��ec�A�*

loss\
=���       �	*Ř�ec�A�*

loss�)�<��Zb       �	�q��ec�A�*

loss��<Qx�       �	m��ec�A�*

loss;%u=��S/       �	����ec�A�*

lossF;�3~�       �	�d��ec�A�*

loss��P<�B�j       �	���ec�A�*

loss��:�5<       �	U���ec�A�*

loss���:�Gv       �	d>��ec�A�*

loss|�<�v�       �	�ٝ�ec�A�*

loss6�R<���       �	 z��ec�A�*

loss��R<�	8�       �	���ec�A�*

loss/��<`�C�       �	���ec�A�*

loss	�H<E�]       �	�U��ec�A�*

loss	k1=J��       �	���ec�A�*

loss�t=G^wi       �	֎��ec�A�*

loss-CB=��       �	�,��ec�A�*

lossN�=�|s�       �	�Ң�ec�A�*

lossx��;虵       �	/k��ec�A�*

loss)�5;w���       �	. ��ec�A�*

loss��;�Y2�       �	����ec�A�*

loss��u=���>       �	Q2��ec�A�*

lossZ�S;;��V       �	*ʥ�ec�A�*

lossh�Q=@p�j       �	�b��ec�A�*

lossq<O��       �	���ec�A�*

lossr=�<�d�~       �	���ec�A�*

loss�0;�A�1       �	�=��ec�A�*

loss�;��L       �	����ec�A�*

lossj��;��tA       �	}���ec�A�*

loss_SV=[xgW       �	-'��ec�A�*

loss�|�<t��       �	����ec�A�*

lossC=^Ƴ       �	U��ec�A�*

loss;�=k�       �	���ec�A�*

loss잗;s�Zs       �	�~��ec�A�*

lossR�^<�ݼg       �	��ec�A�*

loss��B;�8-(       �	����ec�A�*

lossؘ0:9oW       �	B��ec�A�*

loss���;|�W�       �	bخ�ec�A�*

loss΁<D/d       �	�v��ec�A�*

loss���;
       �	���ec�A�*

loss1 W=�CrB       �	����ec�A�*

loss��=�Z�       �	K��ec�A�*

loss�3�;^X��       �	���ec�A�*

loss7<�-��       �	pz��ec�A�*

lossV�;;�-       �	Z,��ec�A�*

lossaA;�[�u       �	�ʳ�ec�A�*

lossz��<��A�       �	wf��ec�A�*

loss��8<N�       �	�
��ec�A�*

lossmy<��       �	����ec�A�*

loss֧X>�L��       �	UQ��ec�A�*

loss��<�.��       �	����ec�A�*

loss��t;S=�       �	%���ec�A�*

lossi�H<�r�P       �	�;��ec�A�*

loss��n<-G�D       �	oظ�ec�A�*

lossGR	<���       �	ޏ��ec�A�*

loss�d�<�Oϳ       �	�+��ec�A�*

loss!s<p|��       �	�ֺ�ec�A�*

loss�:<;ն       �	E���ec�A�*

loss�<=����       �	E+��ec�A�*

lossf#�<�HI       �	$���ec�A�*

loss�0�<YC�       �	�a��ec�A�*

lossX��<rP`       �	���ec�A�*

loss�=���t       �	ץ��ec�A�*

loss���:����       �	ZG��ec�A�*

loss%W=��4C       �	����ec�A�*

loss�n�<�       �	8���ec�A�*

loss���=/BZ�       �	)��ec�A�*

loss�o�=�J`       �	����ec�A�*

loss��X<\1�       �	Zh��ec�A�*

loss���<�a       �	�	��ec�A�*

loss&�;d��r       �	����ec�A�*

loss�oT=T��,       �	IJ��ec�A�*

loss~�=_5V       �	����ec�A�*

loss�`�<d�;�       �	ۆ��ec�A�*

loss���<0���       �	�"��ec�A�*

loss|��:$vc       �	˿��ec�A�*

loss��G<�p       �	 a��ec�A�*

lossl"�=��eC       �	:��ec�A�*

loss��w<��`       �	Y���ec�A�*

lossVl{<)��;       �	�?��ec�A�*

loss�P<W�J-       �	k���ec�A�*

lossZQ<�j}�       �	�k��ec�A�*

loss�^c=��6�       �	y��ec�A�*

loss�f:�t.       �	1���ec�A�*

loss���<�_d       �	0-��ec�A�*

lossL��;98$       �	@���ec�A�*

loss(�	<i�#       �	m��ec�A�*

loss��c;eO��       �	F
��ec�A�*

loss@�6;�:�       �	���ec�A�*

loss	�;���       �	�U��ec�A�*

loss��;�e       �	����ec�A�*

loss-��<�1��       �	y���ec�A�*

loss�#�<S���       �	�-��ec�A�*

loss���;�O�       �	����ec�A�*

lossì�<�h�       �	�v��ec�A�*

lossj�<9:I       �	^��ec�A�*

loss�f�=�^s       �	����ec�A�*

loss��<��=       �	�R��ec�A�*

loss-�J<qt"�       �	����ec�A�*

lossj<]uv       �	����ec�A�*

loss��=�f�       �	�,��ec�A�*

loss�>�=cb�       �	K���ec�A�*

loss�W=��^�       �	c_��ec�A�*

loss���=�~�h       �	����ec�A�*

lossq��<�^{�       �	Ĕ��ec�A�*

loss���<)80y       �	n1��ec�A�*

loss�<��j       �	X���ec�A�*

loss;+@<�t�       �	|b��ec�A�*

loss�@;�R�@       �	����ec�A�*

loss�~o;F��       �	̛��ec�A�*

lossZ�&=ӗ:�       �	0/��ec�A�*

loss��P;�:��       �	X���ec�A�*

loss|׆;�N��       �	Mj��ec�A�*

loss|�=�z�\       �	���ec�A�*

loss�P�;^ޥ�       �	���ec�A�*

lossr��<�.��       �		���ec�A�*

lossz��<���       �	A��ec�A�*

loss�O�<0c�       �	����ec�A�*

lossߠ�<Ÿ�       �	Xt��ec�A�*

lossqK=�C�       �	 ��ec�A�*

loss�Q4=�)<�       �	2���ec�A�*

loss[d�;�:��       �	EF��ec�A�*

lossڡ$=5J]�       �	����ec�A�*

loss���<�8{U       �	���ec�A�*

loss�l�<S�k�       �	_B��ec�A�*

loss^�<�<H�       �	o���ec�A�*

lossb�=����       �	�n��ec�A�*

loss{��=��$�       �	y��ec�A�*

loss�T�;Ӏ��       �	���ec�A�*

lossW�0=�s�       �	<-��ec�A�*

loss���;2���       �	���ec�A�*

loss(+M<4��.       �	�[��ec�A�*

lossz�=�%�I       �	x���ec�A�*

loss|H�;�Vp       �	X���ec�A�*

loss&�<��|�       �	d!��ec�A�*

loss=��<:(%_       �	i���ec�A�*

loss��
=_�e       �	�d��ec�A�*

lossQ`�;z���       �	���ec�A�*

loss��;��j       �	ܝ��ec�A�*

loss�d�<���       �	m9��ec�A�*

loss���<�j�       �	����ec�A�*

loss���<d�       �	�e��ec�A�*

loss]�=6^4E       �	"���ec�A�*

loss���={ְ�       �	:���ec�A�*

lossoo;����       �	�-��ec�A�*

loss�H<��F       �	����ec�A�*

loss�x=I�d       �	����ec�A�*

loss�� =-b&�       �	�p��ec�A�*

losst�x<��*�       �	���ec�A�*

loss��/;-��p       �	~���ec�A�*

loss=�=�Fj�       �	WC��ec�A�*

loss��J=w���       �	#���ec�A�*

loss}Ĺ=I��~       �	iq��ec�A�*

losss'�:Lp�o       �	���ec�A�*

lossS�M<>��       �	>���ec�A�*

lossN��;&�n       �	�R��ec�A�*

loss��;����       �	)���ec�A�*

loss��k=�q�       �	ȴ��ec�A�*

loss��<�$h�       �	����ec�A�*

lossŴ<�M��       �	�#��ec�A�*

loss#�;X�c       �	���ec�A�*

loss�=MJ�=       �	����ec�A�*

lossK�<Yk	       �	DL��ec�A�*

lossӊ�<�6i�       �	����ec�A�*

loss��[=���       �	����ec�A�*

lossE�<�ܞE       �	A( �ec�A�*

loss�
x;o��       �	�� �ec�A�*

loss/�;�.��       �	XW�ec�A�*

lossV\�<V��z       �	���ec�A�*

lossri�:�2�       �	��ec�A�*

loss�
1<:�       �	�0�ec�A�*

loss��;�*�       �	���ec�A�*

lossN�=��       �	
d�ec�A�*

lossh�q<�hY�       �	7��ec�A�*

loss�<�+�~       �	���ec�A�*

loss�f�<@��       �	f3�ec�A�*

loss-DC=���_       �	��ec�A�*

loss!o�=�:{j       �	�^�ec�A�*

lossܸ�;l�       �	e��ec�A�*

loss�ȫ<�>�&       �	$�	�ec�A�*

loss x(>�3��       �	>
�ec�A�*

lossmP�<�~��       �	��
�ec�A�*

loss���;�bN�       �	��ec�A�*

loss^#=W��       �	u"�ec�A�*

loss�(<w�H~       �	/��ec�A�*

lossHt�<2���       �	Cq�ec�A�*

loss��=v�j:       �	�R�ec�A�*

loss�%�;@]r!       �	��ec�A�*

loss2�=�Œ�       �	���ec�A�*

lossv�=��^�       �	3�ec�A�*

loss���<��u       �	���ec�A�*

lossc\�;N�C�       �	4f�ec�A�*

loss���;�O�"       �	��ec�A�*

loss�=���        �	���ec�A�*

loss#�_;�.       �	j1�ec�A�*

lossf�9;�?+m       �	`��ec�A�*

loss�TE;َ        �	xb�ec�A�*

loss3=*<T       �	���ec�A�*

lossC~�:�l�`       �	���ec�A�*

loss(oV=��l�       �	�K�ec�A�*

loss	
=(�I       �	��ec�A�*

loss�<5���       �	���ec�A�*

loss-� ;)Z!       �	0�ec�A�*

loss��=c%f�       �	A��ec�A�*

lossn��<��{       �	�|�ec�A�*

loss�2<����       �	��ec�A�*

lossq�d<śW�       �	���ec�A�*

loss&^<)��2       �	FC�ec�A�*

loss�!w<�A�       �	���ec�A�*

loss��=5f��       �	�{�ec�A�*

loss�=p�H       �	r�ec�A�*

loss�z�;_ȳ       �	!��ec�A�*

loss�a�=���l       �	�P�ec�A�*

loss�w�;��Nh       �	1��ec�A�*

loss�@�:���       �	"��ec�A�*

loss#eL:#�}       �	�" �ec�A�*

loss�=�#B        �	Z� �ec�A�*

loss��=g>�       �	a!�ec�A�*

loss��:�9�M       �	��!�ec�A�*

loss6�<W�       �	��"�ec�A�*

loss�_�9A�f       �	wM#�ec�A�*

loss��=�	`�       �	t�#�ec�A�*

lossƘ�<��Tw       �	b�$�ec�A�*

loss�
<Y���       �	$D%�ec�A�*

loss���;��_       �	�%�ec�A�*

loss΢,<c��       �	��&�ec�A�*

lossaq <��V       �	�F'�ec�A�*

loss�<p�i�       �	��'�ec�A�*

loss|�<�}�       �	͒(�ec�A�*

loss���:�"�       �	�5)�ec�A�*

loss�z�<��\       �	B�)�ec�A�*

loss%��;��t       �	=*�ec�A�*

loss��<��=�       �	�+�ec�A�*

loss�0�; AI�       �	��+�ec�A�*

loss�*$<(uL]       �	�P,�ec�A�*

lossh��;��xW       �	��,�ec�A�*

loss��;��uz       �	N�-�ec�A�*

lossv��:	�*       �	�6.�ec�A�*

lossv�d=9>�O       �	M�.�ec�A�*

loss�y�=Ss9       �	�y/�ec�A�*

loss��&<Z���       �	X0�ec�A�*

loss\��=�:��       �	��0�ec�A�*

loss�s�;�P|[       �	�^1�ec�A�*

loss�9�<D1�u       �	��1�ec�A�*

loss��$=7ڍ       �	N�2�ec�A�*

lossV�<�Km       �	�=3�ec�A�*

lossaz<�^       �	��3�ec�A�*

loss	̍;GOO       �	�y4�ec�A�*

lossK,�=A�G       �	�5�ec�A�*

lossxy�:�%        �	C�5�ec�A�*

lossT�;+���       �	�|6�ec�A�*

loss���;k<�7       �	�*7�ec�A�*

loss�$N=��|       �	��7�ec�A�*

loss�t�<��        �	~r8�ec�A�*

loss�&�;7��       �	\9�ec�A�*

loss� �;�D��       �	~�9�ec�A�*

loss�@~<�`i       �	8�:�ec�A�*

loss�=� �       �	��;�ec�A�*

lossa+=}��       �	\V<�ec�A�*

loss�ƞ<G~6r       �	�=�ec�A�*

lossM�<]��       �	��=�ec�A�*

loss���=�#\9       �	K>�ec�A�*

loss(��;L���       �	Z�>�ec�A�*

lossf]<����       �	ޓ?�ec�A�*

lossa�;�6��       �	�;@�ec�A�*

loss��<҃�       �	��@�ec�A�*

loss��!:���       �	�B�ec�A�*

lossTt�<�Q�       �	4.C�ec�A�*

loss�k}=2F�       �	Q�C�ec�A�*

loss���:UDX       �	JE�ec�A�*

loss�='��       �	�E�ec�A�*

loss;m�;�*       �	��F�ec�A�*

loss��<��       �	-BG�ec�A�*

losst�A<��       �	�G�ec�A�*

loss��D:�S��       �	��H�ec�A�*

loss��;Ʒ�       �	$aI�ec�A�*

loss���<��d�       �	]�I�ec�A�*

loss��^=&�m�       �	%�J�ec�A�*

loss�R�=�L�W       �	6vK�ec�A�*

loss<�F=?u��       �	\UL�ec�A�*

lossѬ=���       �	��L�ec�A�*

loss��<`�}       �	�M�ec�A�*

lossG�=I�ji       �	28N�ec�A�*

lossN��<�z       �	�N�ec�A�*

loss10�;r̈       �	�vO�ec�A�*

loss�!=L<�T       �	�P�ec�A�*

loss��)<�X>�       �	��P�ec�A�*

lossű�=�6��       �	�ZQ�ec�A�*

loss<+�;y�       �	��Q�ec�A�*

loss�K�;�fQ�       �	w�R�ec�A�*

loss�w%;Z�       �	�T�ec�A�*

loss��<��       �	��T�ec�A�*

losswS�;���n       �	XU�ec�A�*

loss���<����       �	\V�ec�A�*

loss��=��       �	�V�ec�A�*

loss���<��xx       �	�;W�ec�A�*

loss���<��Q�       �	x�W�ec�A�*

loss^*�;��db       �	vmX�ec�A�*

loss�׿<�.�9       �	OY�ec�A�*

lossj�;���+       �	��Y�ec�A�*

lossЉ;�^�       �	!:Z�ec�A�*

loss�A;Z�T�       �	�Z�ec�A�*

loss�x;�x��       �	Gr[�ec�A�*

lossr׮<�8�       �	\�ec�A�*

lossA�<�g�9       �	��\�ec�A�*

loss��;���X       �	�=]�ec�A�*

lossZ��<=T��       �	H�]�ec�A�*

loss��A=*��       �	%z^�ec�A�*

loss�<�٥        �	�_�ec�A�*

lossT��<�r6�       �	�_�ec�A�*

loss%��:j=ƭ       �	tD`�ec�A�*

loss�g<,��F       �	Y�`�ec�A�*

loss��<$�4       �	�ra�ec�A�*

lossf�";�GE       �	�b�ec�A�*

loss��=t��|       �	؞b�ec�A�*

loss���<q�+       �	2c�ec�A�*

loss�ʶ;1��t       �	��c�ec�A�*

loss-�T<�~��       �	�ed�ec�A�*

loss�%�:�.O       �	��d�ec�A�*

lossK<�6�m       �	�e�ec�A�*

loss׶,;E��       �	m f�ec�A�*

loss{E�;�D��       �	>�f�ec�A�*

lossW�`;Q�҄       �	=Fg�ec�A�*

loss�	�;�J��       �	��g�ec�A�*

loss�#<�bk       �	��h�ec�A�*

loss�.�=Z`��       �	�Ji�ec�A�*

loss �<�C�       �	��i�ec�A�*

loss��=7�z       �	uj�ec�A�*

loss�<D<�&       �	0k�ec�A�*

lossV�Q=S��       �	N�k�ec�A�*

loss���: I�u       �	Zdl�ec�A�*

loss��a9�Y"s       �	�m�ec�A�*

lossO<ϣ�g       �	�m�ec�A�*

loss��:{?�       �	|dn�ec�A�*

loss���;�J�S       �	o�ec�A�*

lossʛ�:W�~       �	��o�ec�A�*

loss�f9z�V�       �	�Xp�ec�A�*

lossL��;7JR�       �	��p�ec�A�*

lossZ�[:?��       �	�q�ec�A�*

loss���;��H       �	�Mr�ec�A�*

loss�Q�9ݔܸ       �	x�s�ec�A�*

loss�%�;����       �	N�u�ec�A�*

loss�F�=1�10       �	�<v�ec�A�*

loss=��;I�-�       �	D�v�ec�A�*

loss3N�;���       �	�w�ec�A�*

loss�c�9ޞ'       �	�@x�ec�A�*

loss�L;=�#�       �	��x�ec�A�*

loss��|:lο�       �	K�y�ec�A�*

lossfM�=��y       �	j0z�ec�A�*

loss���<g"       �	T�z�ec�A�*

loss�1<:(l�       �	?|�ec�A�*

loss��:;�b�       �	Ǽ|�ec�A�*

loss�jv<�[��       �	�^}�ec�A�*

loss���=OX��       �	��}�ec�A�*

loss6��<�m�       �	�~�ec�A�*

losss�=�s�       �	�H�ec�A�*

loss̩3=�_�       �	���ec�A�*

losss[�;v_�        �	z���ec�A�*

lossz��;��R�       �	T7��ec�A�*

lossJ��<�H       �	'؁�ec�A�*

loss��<B��,       �	�q��ec�A�*

loss�ߴ<c���       �	�
��ec�A�*

loss׿4=�e�       �	���ec�A�*

loss4U4<w�T�       �	c���ec�A�*

loss�{!<��d�       �	P��ec�A�*

losso�3;�<�       �	̴��ec�A�*

loss{��;X�?A       �	�J��ec�A�*

loss��1<� ��       �	a���ec�A�*

loss�;� 8       �	�}��ec�A�*

lossa�t=w�       �	�)��ec�A�*

lossm��:~š�       �	̈�ec�A�*

loss��<hJ�       �	"p��ec�A�*

loss���:"�l       �	���ec�A�*

loss��;��p�       �	�Ċ�ec�A�*

loss�=�<r~��       �	�m��ec�A�*

loss�t�<�/�n       �	Χ��ec�A�*

loss��=�bҝ       �	�Y��ec�A�*

loss)B=�b��       �	���ec�A�*

loss[{�<���N       �	ꮎ�ec�A�*

lossႹ;� �       �	�W��ec�A�*

loss�sX<)*�x       �	���ec�A�*

loss1�;R(�       �	����ec�A�*

lossē�;���       �	�Y��ec�A�*

loss��=��V       �	.��ec�A�*

loss�)�<��<       �	M���ec�A�*

lossry�<��T5       �	�C��ec�A�*

loss�b�<�whg       �	
���ec�A�*

lossS�[:��c/       �	����ec�A�*

loss	�b:���A       �	bL��ec�A�*

loss�Q�:S�W}       �	���ec�A�*

loss?U=q��       �	���ec�A�*

loss��3:w�       �	gC��ec�A�*

loss��E; ��N       �	�ܗ�ec�A�*

loss�>_��       �	�v��ec�A�*

loss��:;����       �	Q��ec�A�*

loss�*<��Q�       �	¾��ec�A�*

loss��/;3�       �	�i��ec�A�*

loss��;#���       �	���ec�A�*

loss���<��        �	�3��ec�A�*

loss�(9=�Ю]       �	ҵ�ec�A�*

lossT�<&c��       �	�p��ec�A�*

lossqA�=�Kx       �	���ec�A�*

loss,��;�
�       �	����ec�A�*

loss(��;�"�W       �	�O��ec�A�*

lossDD�<�K       �	5��ec�A�*

loss��m<B� �       �	j���ec�A�*

loss�͆=A؝�       �	-"��ec�A�*

loss���<�b�       �	4���ec�A�*

loss�:s�B�       �	ɑ��ec�A�*

lossL�<�4��       �	�7��ec�A�*

lossfOK<1���       �	߼�ec�A�*

loss�*<%�dg       �	ۈ��ec�A�*

loss��o=q��       �	�c��ec�A�*

loss��<շF       �	�n��ec�A�*

loss�'�9`i�       �	�E��ec�A�*

loss��<�#�       �	�%��ec�A�*

lossx�=�F       �	���ec�A�*

loss�q�=n�U�       �	ӥ��ec�A�*

loss1e�;D�H       �	K[��ec�A�*

loss�+�=����       �	z���ec�A�*

lossnu�:p(�       �	b1��ec�A�*

loss��<k�e�       �	f���ec�A�*

loss�Ph=.�       �	���ec�A�*

loss�y;>*Z       �	����ec�A�*

lossx�<%��       �	�e��ec�A�*

lossGH�;]�E�       �	���ec�A�*

loss�P
=j���       �	&���ec�A�*

loss)�S<�bU       �	�d��ec�A�*

loss}�;�Ogk       �	ǝ��ec�A�*

loss�u�;�&��       �	y@��ec�A�*

loss�	�;qP �       �	����ec�A�*

loss*��<_��6       �	ʇ��ec�A�*

loss�-?<*�       �	/5��ec�A�*

loss���=Hʏ       �	�?��ec�A�*

loss��;��^       �	6w��ec�A�*

loss�$=��       �	���ec�A�*

loss�:
=:9�       �	����ec�A�*

loss�^e<��       �	*T��ec�A�*

loss��<�c�D       �	x���ec�A�*

lossc��=6	��       �	Ό��ec�A�*

loss_�b<�\�       �	�-��ec�A�*

loss��d< �       �	.���ec�A�*

loss��:zr��       �	�d��ec�A�*

loss-]�=#�N       �	?���ec�A�*

loss��Z=��Ov       �	��ec�A�*

loss��;���\       �	4���ec�A�*

loss�S|=��C�       �	wH��ec�A�*

lossn�O:&�G       �	k��ec�A�*

loss��;��|�       �	����ec�A�*

loss��=/�B�       �	�:��ec�A�*

loss�I�;�KjD       �	����ec�A�*

loss�>��g       �	����ec�A�*

loss��<��       �	�(��ec�A�*

loss�<=u�X       �	����ec�A�*

loss�L�:,       �	�[��ec�A�*

loss��:gĐt       �	����ec�A�*

lossт<�L�D       �	o���ec�A�*

lossf��<g8[�       �	a��ec�A�*

loss*�=i�,       �	O���ec�A�*

loss��Q<���       �	�F��ec�A�*

loss��;BC��       �	n���ec�A�*

loss�RQ<��ߘ       �	�{��ec�A�*

loss��=0��*       �	���ec�A�*

loss*ߣ;r�       �	����ec�A�*

loss���</b�       �	n��ec�A�*

loss�=�h��       �	���ec�A�*

lossO�=|��       �	�[��ec�A�*

loss\��<��F�       �	�	��ec�A�*

loss�;Ac~       �	���ec�A�*

loss���;�m�       �	sL��ec�A�*

loss#b�<
[c�       �	����ec�A�*

loss���<����       �	7���ec�A�*

loss��=zo�       �	Ϣ��ec�A�*

loss���<���       �	�:��ec�A�*

loss�A:` �       �	)���ec�A�*

loss�\�;k	qo       �	/n��ec�A�*

lossz8<q��       �	���ec�A�*

loss_��<�.�       �	ޫ��ec�A�*

loss���<�p�       �	4K��ec�A�*

lossO�"=��v2       �	����ec�A�*

loss��3<?$!�       �	���ec�A�*

loss3��:*{��       �	��ec�A�*

loss��:c�4N       �	-���ec�A�*

loss��<�N�       �	^M��ec�A�*

loss���<��       �	 ���ec�A�*

loss/F�<����       �	H��ec�A�*

loss[/]<b�!i       �	x���ec�A�*

loss <=�<�       �	�_��ec�A�*

lossS h<��       �	G��ec�A�*

loss[�<O��       �	���ec�A�*

loss��;9�0	       �	�F��ec�A�*

loss�<����       �	!���ec�A�*

loss�l�=��B*       �	�{��ec�A�*

loss4"�:�bL       �	s��ec�A�*

loss8 �<l���       �	����ec�A�*

loss�<�AB^       �	�G��ec�A�*

lossa6�=A��0       �	����ec�A�*

loss���;�Y��       �	zp��ec�A�*

loss��;)��       �	y?��ec�A�*

lossiS�:|^n+       �	��ec�A�*

loss�&;�       �	-���ec�A�*

loss;W��       �	؃��ec�A�*

loss���:�m.       �	*q��ec�A�*

loss֌e<?(j�       �	� �ec�A�*

loss�T�<����       �	ɬ �ec�A�*

loss���;R`�       �	�T�ec�A�*

loss�s�=d�P]       �	���ec�A�*

lossc��<:�B�       �	,��ec�A�*

lossAbL<�ҝ?       �	��ec�A�*

loss}P�<$�*       �	�(�ec�A�*

loss���<�;=       �	���ec�A�*

lossÞ�:��_       �	�W�ec�A�*

loss w�<"��o       �	���ec�A�*

loss*P�;0�       �	���ec�A�*

lossOC;�Ē       �	S �ec�A�*

loss�F7=z��P       �	��ec�A�*

loss��:���        �	�_�ec�A�*

losss�=	��F       �	���ec�A�*

loss~�
=���       �	��	�ec�A�*

loss�B%=���;       �	�w
�ec�A�*

loss-y
<���       �	��ec�A�*

loss���;įS�       �	���ec�A�*

lossd7:\���       �	�\�ec�A�*

lossm�f<Z�       �	� �ec�A�*

loss<���       �	��ec�A�*

loss�8=ar&^       �	�;�ec�A�*

loss���<�       �	
��ec�A�*

loss�<�4�       �	�r�ec�A�*

loss�j�<6ٕ       �	4�ec�A�*

loss��=����       �	���ec�A�*

lossϭ<�L�?       �	�F�ec�A�*

lossg"�<\4��       �	7��ec�A�*

lossQ�<       �	���ec�A�*

loss�i�;�Ӏ       �	��ec�A�*

loss	�:P�k�       �	ٲ�ec�A�*

loss���:����       �	~T�ec�A�*

lossO�:F�'       �	!#�ec�A�*

loss�=�:Ͷ�q       �	���ec�A�*

loss�|�;��       �	�b�ec�A�*

loss��;򜭜       �	�ec�A�*

loss���;nL̯       �	�V�ec�A�*

loss��%<2翠       �	U��ec�A�*

loss�{"=�m�       �	���ec�A�*

loss4�A<�o       �	�;�ec�A�*

lossj�=��       �	<��ec�A�*

loss̪N=i��S       �	�q�ec�A�*

loss}15<8��       �	k�ec�A�*

lossnNJ=�9&       �	���ec�A�*

loss���:0kB�       �	8J�ec�A�*

loss�R�=O��       �	��ec�A�*

lossG!<XQR�       �	Y��ec�A�*

loss}:<M�q       �	�� �ec�A�*

lossJ=��|$       �	Ú!�ec�A�*

loss:+�:���t       �	!>"�ec�A�*

lossď�<�f:�       �	c�"�ec�A�*

loss�c�<K���       �	zp#�ec�A�*

loss�v!=���       �	=
$�ec�A�*

lossp��;��<�       �	��$�ec�A�*

loss��J=�"(       �	bN%�ec�A�*

losso�;�W	       �	o�%�ec�A�*

loss{[);[bY       �	L�&�ec�A�*

loss�#�<ǳ�       �	I-'�ec�A�*

lossx�L=���       �	VI(�ec�A�*

loss�+,;W9��       �	u�(�ec�A�*

lossܔ�:1M�B       �	�)�ec�A�*

loss���=��       �	�D*�ec�A�*

lossJL<r�z       �	��*�ec�A�*

loss3��:s�eU       �	��+�ec�A�*

loss�u5=��e       �	1%,�ec�A�*

loss�j�<�},�       �	A�,�ec�A�*

loss"~�9���k       �	�R-�ec�A�*

loss�]�;+���       �	��-�ec�A�*

loss�N�:xI�"       �	��.�ec�A�*

lossâ�<�
�       �	v/�ec�A�*

loss���;3���       �	��/�ec�A�*

loss���<d
�g       �	�K0�ec�A�*

lossM!7=��2       �	��0�ec�A�*

loss���:�y�       �	�z1�ec�A�*

loss���< :`�       �	�E2�ec�A�*

lossv>�<�Q�
       �	<�2�ec�A�*

loss͙Y;��/A       �	p@4�ec�A�*

loss���:è�       �	�z5�ec�A�*

loss[~�:�֐�       �	6�ec�A�*

loss4�8:���       �	��6�ec�A�*

loss�θ:#��       �	�`7�ec�A�*

lossm�;�*�       �	}	8�ec�A�*

loss�w�:ʅx�       �	��8�ec�A�*

loss��j<�7�#       �	<9�ec�A�*

loss��z=r�RX       �	0�9�ec�A�*

lossFU=��f       �	}u:�ec�A�*

loss6;�=��       �	,;�ec�A�*

lossn6,=n(�       �	B�;�ec�A�*

loss��:�[��       �	#I<�ec�A�*

loss���:n 	�       �	�&=�ec�A�*

loss�;��       �	��=�ec�A�*

lossԟ�<�ːd       �	O?�ec�A�*

loss�[�=!\       �	�@�ec�A�*

lossA = ��B       �	y]A�ec�A�*

loss�:�;���v       �	PB�ec�A�*

lossv��=d�       �	�B�ec�A�*

loss��<B�       �	�{C�ec�A�*

loss���;y���       �	�ED�ec�A�*

loss�$<��K       �	��D�ec�A�*

loss1Zv<���W       �	w�E�ec�A�*

loss�m=T�X       �	�F�ec�A�*

loss��n=�kU�       �	��F�ec�A�*

lossoS<Q�U�       �	3RG�ec�A�*

loss���: sj       �	��G�ec�A�*

lossd?�;,LV�       �	;�H�ec�A�*

loss�S�:��}       �	�I�ec�A�*

loss��;�Z       �	e�I�ec�A�*

loss�e&;�zҽ       �	�`J�ec�A�*

lossCv�=�oT�       �	3�J�ec�A�*

loss 9 =*�Ѫ       �	Z�K�ec�A�*

loss���=:�kl       �	�:L�ec�A�*

lossA;/=h�x       �	1�L�ec�A�*

loss*V2=�c��       �	AgM�ec�A�*

loss�3�<��r�       �	�N�ec�A�*

loss���<I&�N       �	�N�ec�A�*

losss��;6�*f       �	3O�ec�A�*

loss2=�;��       �	y�O�ec�A�*

loss��X<�ɑj       �	�gP�ec�A�*

lossEf�;��=Y       �	��P�ec�A�*

loss�#R=ҢF       �	��Q�ec�A�*

loss3e=�Qc�       �	c'R�ec�A�*

loss�I�=����       �	�R�ec�A�*

loss)p�;)�Q       �	S[S�ec�A�*

lossڲe:��O       �	�PT�ec�A�*

lossJ��<���       �	�T�ec�A�*

loss,�q;��       �	�wU�ec�A�*

loss!�c<�~8       �	�V�ec�A�*

lossl��;�[��       �	��W�ec�A�*

loss��D<��z�       �	CX�ec�A�*

lossv��;aC�       �	��X�ec�A�*

loss�ʢ;�#=        �	AfY�ec�A�*

loss�J=�X>4       �	2Z�ec�A�*

loss��A=���d       �	�Z�ec�A�*

loss�Tv;_}%       �	��[�ec�A�*

loss0W=i���       �	Й\�ec�A�*

lossϜ�;���R       �	�:]�ec�A�*

loss��<Y��       �	��]�ec�A�*

loss	�V<3S       �	��^�ec�A�*

loss�L�;�f�       �	�'_�ec�A�*

lossD��<]v"�       �	v�_�ec�A�*

lossv�;��3�       �	��`�ec�A�*

loss�(�;�Z߻       �	�1b�ec�A�*

loss4B=�I7       �	��b�ec�A�*

lossԇ5;p��       �	�`c�ec�A�*

lossv5�=چ��       �	d�ec�A�*

lossB>�%�R       �	�d�ec�A�*

lossl��<X?�       �	B]e�ec�A�*

loss�X�9�T}       �	��e�ec�A�*

loss�v<�s|m       �	i�f�ec�A�*

loss���;I�G       �	[$g�ec�A�*

loss��:����       �	��g�ec�A�*

loss�%:;B��q       �	�Xh�ec�A�*

loss-N�=t��       �	��h�ec�A�*

loss�@;y��G       �	i�i�ec�A�*

lossd��<86�>       �	'3j�ec�A�*

losse�l<d�4$       �	g�j�ec�A�*

lossO�<7�&�       �	3nk�ec�A�*

loss��=@
       �	Bl�ec�A�*

loss�y!;8W�U       �	��l�ec�A�*

lossˀ<�݊       �	sJm�ec�A�*

loss�7>=���       �	��m�ec�A�*

lossϧ�<%RE�       �	>xn�ec�A�*

loss�#<�ҫ�       �	<o�ec�A�*

lossŌ<Kk�       �	ظo�ec�A�*

lossz;�<�v��       �	�Sp�ec�A�*

loss��(<��P5       �	C�p�ec�A�*

loss�Ik;�Æ>       �	�zq�ec�A�*

loss70.;����       �	�r�ec�A�*

losst��<�u�4       �	U�r�ec�A�*

loss��:��       �	�9s�ec�A�*

lossc�q;8/�       �	��s�ec�A�*

loss���;JX4�       �	�ft�ec�A�*

loss��J=48�       �	��t�ec�A�*

loss*��<QT��       �	��u�ec�A�*

lossŊ:���       �	+v�ec�A�*

lossK�:C4��       �	<�v�ec�A�*

lossroo=��u       �	&Rw�ec�A�*

loss��<�o��       �	5�w�ec�A�*

loss��<_�e�       �	��x�ec�A�*

lossg�=nW�|       �	�y�ec�A�*

loss{�=��v       �	��y�ec�A�*

loss���<��       �	�Dz�ec�A�*

loss1\+<�6�0       �	m�z�ec�A�*

loss�ZT;͎�       �	��{�ec�A�*

lossL�O<�Gz�       �	p(|�ec�A�*

lossD�9=�_�       �	�|�ec�A�*

loss?�<�{c       �	/�}�ec�A�*

loss/�g;�ש�       �	y�~�ec�A�*

lossR�<]�Rk       �	r5�ec�A�*

loss6�2;�xG�       �	���ec�A�*

loss�#";(�V^       �	�o��ec�A�*

loss��=j-�k       �	W��ec�A�*

loss���=m���       �	ʣ��ec�A�*

loss��<\5'T       �	!:��ec�A�*

lossnc)=���       �	-ς�ec�A�*

loss�y#;�d�/       �	�c��ec�A�*

loss�=��       �	%��ec�A�*

loss��<n�]\       �	(���ec�A�*

loss���<�t�r       �	�1��ec�A�*

loss:�<Ș��       �	�̅�ec�A�*

losso�<y6r|       �	�f��ec�A�*

lossWB|=(��8       �	���ec�A�*

loss1�;*��       �	����ec�A�*

loss�"�:'�f�       �	�@��ec�A�*

losslא;���       �	�ֈ�ec�A�*

lossR�;�0�       �	i��ec�A�*

lossU��<;f�	       �	e���ec�A�*

lossK�<�Xv�       �	����ec�A�*

loss�4�;��b�       �	|+��ec�A�*

lossZu<��fA       �	n���ec�A�*

loss)�d;P^+]       �	U��ec�A�*

loss��+;�S�{       �	���ec�A�*

lossT�=<j�-�       �	Ԁ��ec�A�*

loss��;%K��       �	u"��ec�A�*

loss�]�<���       �	3Ď�ec�A�*

loss�]=�x^       �	�h��ec�A�*

loss��8;�� �       �	���ec�A�*

loss�/#;"�y�       �	����ec�A�*

loss; =���       �	�L��ec�A�*

loss��9<Ӗ�z       �	����ec�A�*

loss֮�;#�T       �	����ec�A�*

loss�K,<��{�       �	\;��ec�A�*

loss��<;[D�       �	����ec�A�*

lossD��=��5�       �	���ec�A�*

lossq[<����       �	����ec�A�*

loss�K�<{j�B       �	F\��ec�A�*

loss#�:�L��       �	���ec�A�*

loss�r�:Mz^       �	����ec�A�*

loss��=���&       �	P4��ec�A�*

loss�d�<�*~4       �	Aؘ�ec�A�*

lossZ�<˘A       �	�u��ec�A�*

loss��<��!H       �	)���ec�A�*

loss��<��;q       �	�&��ec�A�*

lossEv�:���j       �	�!��ec�A�*

lossN�)<��@       �	�М�ec�A�*

loss���<���|       �	����ec�A�*

lossة�;e�H�       �	� ��ec�A�*

loss�by=hu?_       �	ʞ�ec�A�*

loss���=�s�       �	/o��ec�A�*

loss�"�<P@15       �	Z��ec�A�*

lossW�;���0       �	���ec�A�*

loss�o�<ͽ�\       �	�K��ec�A�*

loss�0�<R�q       �	��ec�A�*

loss���<�ޑ�       �	����ec�A�*

loss7��;����       �	k)��ec�A�*

loss��<;x 9       �	>ͣ�ec�A�*

lossƑ=�t�       �	j��ec�A�*

loss�{<2���       �	���ec�A�*

loss�m=�-5l       �	����ec�A�*

loss�b�;Y%-9       �	{N��ec�A�*

loss��o;p�R]       �	���ec�A�*

loss��@<|�       �	����ec�A�*

loss���<�"��       �	[$��ec�A�*

lossZ,;�&�8       �	b���ec�A�*

loss�ܯ<���       �	�N��ec�A�*

lossm~<�]-^       �	$���ec�A�*

loss:�C:�#N�       �	���ec�A�*

loss��="�a       �	k)��ec�A�*

loss��=�c}�       �	U���ec�A�*

loss�%=�z       �	5]��ec�A�*

loss���=����       �	� ��ec�A�*

lossƢ;Y��G       �	ꗭ�ec�A�*

loss�Z�<�?�9       �	
/��ec�A�*

loss�ρ;����       �	B̮�ec�A�*

lossv�%=/�       �	�e��ec�A�*

loss���;�0�       �	����ec�A�*

loss�|�;����       �	⑰�ec�A�*

lossNO<�3       �	_)��ec�A�*

loss�P<�R'       �	�ɱ�ec�A�*

loss�;
<��{       �	B\��ec�A�*

loss�ut;���       �	���ec�A�*

loss��=$��l       �	�ec�A�*

loss��=����       �	�#��ec�A�*

lossÎ<-l/       �	����ec�A�*

lossE�_;�N��       �	�N��ec�A�*

lossC(c<p��%       �	O��ec�A�*

loss��>�[Q       �	-z��ec�A�*

lossX�;��       �	���ec�A�*

loss�ؙ:�.�       �	3���ec�A�*

loss&;^<Xy��       �	;:��ec�A�*

loss�2�;_p0�       �	�ϸ�ec�A�*

loss�=�<GX�       �	8f��ec�A�*

loss���<� �'       �	v���ec�A�*

losse�;�8�X       �	����ec�A�*

loss���<Y�6�       �	�0��ec�A�*

lossí<H`y�       �	 Ȼ�ec�A�*

lossv�:��-       �	�Y��ec�A�*

loss
��<m>�(       �	)���ec�A�*

lossJ<<��Տ       �	߈��ec�A�*

loss=5y<�d�j       �	c)��ec�A�*

loss%ǎ;���       �	����ec�A�*

lossm��;m4��       �	�P��ec�A�*

loss��;	� �       �	t��ec�A�*

loss��;z��       �	@���ec�A�*

loss|k1<���       �	�!��ec�A�*

lossC�<��5       �	Y���ec�A�*

lossu=�;�       �	`��ec�A�*

loss!��;���       �	����ec�A�*

loss�f ;	�I�       �	���ec�A�*

loss`��<	�~       �	l#��ec�A�*

loss���<Ā9�       �	����ec�A�*

losss�;���       �	]N��ec�A�*

losss�<��j�       �	*���ec�A�*

lossl�w;�m�       �	�y��ec�A�*

loss�6=�C0�       �	k,��ec�A�*

loss|��=��L       �	k���ec�A�*

lossp<�7�/       �	7l��ec�A�*

lossϣ�;a�~�       �	���ec�A�*

loss��<0��.       �	|���ec�A�*

loss��<���X       �	,��ec�A�*

lossJ��<)�F�       �	����ec�A�*

loss�=�ʌt       �	�S��ec�A�*

lossn�=h��m       �	����ec�A�*

loss��=��       �	Þ��ec�A�*

loss ӱ:R�F       �	G:��ec�A�*

loss!&Z=r�       �	����ec�A�*

loss���:�rP[       �	����ec�A�*

loss��<\�q�       �	���ec�A�*

loss�Z�<��Y�       �	����ec�A�*

loss���<����       �	�K��ec�A�*

loss�;r���       �	%���ec�A�*

loss�g\<�l       �	���ec�A�*

loss}؀<��e       �	O��ec�A�*

loss$ڹ;�3J       �	����ec�A�*

loss:'�<�ܑ�       �	�f��ec�A�*

loss���<��֟       �	2���ec�A�*

loss}/�:���       �	���ec�A�*

loss�m�<�?       �	����ec�A�*

lossf�i=f�k       �	d@��ec�A�*

lossע`="�|�       �	��ec�A�*

loss���;���s       �	���ec�A�*

loss�=M�{�       �	֬��ec�A�*

lossl'><gp�(       �	�F��ec�A�*

loss[J<a�<�       �	O��ec�A�*

loss>/�=*���       �	����ec�A�*

lossW��;;��       �	X9��ec�A�*

loss��L=v��       �	j���ec�A�*

lossT�J=xSS�       �	p}��ec�A�*

loss6�<�p$+       �	���ec�A�*

lossA��<_�%       �	)���ec�A�*

loss.��=FTp�       �	�F��ec�A�*

loss�@�;|��       �	r���ec�A�*

loss6h�<�M�O       �	)u��ec�A�*

loss�!<[���       �	�	��ec�A�*

loss0��<0 R       �	ʧ��ec�A�*

loss��:=s��       �	����ec�A�*

losscn=yY"       �	���ec�A�*

loss�m�;��       �	S���ec�A�*

loss�\�=��PG       �	�E��ec�A�*

lossh<�,o       �	����ec�A�*

loss�ҕ<	V�       �	����ec�A�*

loss��;E;gG       �	 (��ec�A�*

loss��t<�n�.       �	���ec�A�*

loss�~�<iR�E       �	�j��ec�A�*

loss�<�L��       �	���ec�A�*

loss��<�P�       �	���ec�A�*

lossƽ�;�X�       �	����ec�A�*

loss1%=��Uq       �	}<��ec�A�*

loss�?;���       �	����ec�A�*

loss�1;*���       �	�r��ec�A�*

lossv��<&s

       �	���ec�A�*

lossʏ�;��=       �	����ec�A�*

loss�=�o1       �	�6��ec�A�*

loss��;�*       �	x���ec�A�*

loss74n=��+�       �	�e��ec�A�*

loss��<]y��       �	����ec�A�*

loss�3=%��k       �	n���ec�A�*

loss&��<xŸ�       �	�_��ec�A�*

loss��=��P�       �	h��ec�A�*

loss��<㼆�       �	����ec�A�*

loss���:�}_       �	�?��ec�A�*

loss �V<���p       �	����ec�A�*

loss��1;n��       �	S���ec�A�*

loss��<*��       �	�1��ec�A�*

loss]<=�e       �	����ec�A�*

lossd��:n˲2       �	�q��ec�A�*

loss!��:Zр�       �	��ec�A�*

loss�K�;��       �	����ec�A�*

loss}��9��Ƅ       �	�A��ec�A�*

lossJ<A<V��       �	����ec�A�*

loss��r;6�.8       �	Bv��ec�A�*

loss��;<�ӹ�       �	J��ec�A�*

lossԡ�;�J�       �	/���ec�A�*

loss7�;w� �       �	�:��ec�A�*

loss���=
�'/       �	����ec�A�*

loss��<���i       �	�o��ec�A�*

lossn�':o�)       �	�
��ec�A�*

loss��;=ߣ��       �	r���ec�A�*

loss���9.5��       �	[��ec�A�*

loss'*�;7���       �	�^��ec�A� *

loss3N�<԰�       �	� �ec�A� *

loss�m�;�ēp       �	>� �ec�A� *

loss@"(<��W       �	�G�ec�A� *

loss-��<�7`z       �	��ec�A� *

lossx��<���       �	\v�ec�A� *

lossSC9tV       �	��ec�A� *

lossq�;Y��       �	ץ�ec�A� *

loss��=t��       �	T9�ec�A� *

loss7��=ICՏ       �	p��ec�A� *

lossH*�=f�~�       �	�l�ec�A� *

loss��;:Soa       �	[�ec�A� *

loss�s�:�d#       �	p��ec�A� *

loss�=�^/       �	
-�ec�A� *

loss� �<x��       �	P��ec�A� *

loss�}4;J�k       �	Y�ec�A� *

loss�K�<=!0       �	���ec�A� *

lossa#�=,pM       �	��	�ec�A� *

loss��<k�$       �	�!
�ec�A� *

loss{�(<�O|�       �	c�
�ec�A� *

loss	��<r��       �	�Q�ec�A� *

loss��<n�C       �	���ec�A� *

loss?<�WF�       �	���ec�A� *

loss�e<q-�       �	P�ec�A� *

loss#6�<�zW�       �	˻�ec�A� *

loss�<�h?L       �	�Y�ec�A� *

loss��q;S�E�       �	���ec�A� *

loss��a:}��&       �	���ec�A� *

loss-Q<vhȜ       �	'1�ec�A� *

lossf��<ф�u       �	��ec�A� *

loss�i	;��f�       �	�}�ec�A� *

loss`�/<�0�       �	�ec�A� *

lossː<A��B       �	��ec�A� *

loss��<4�       �	�A�ec�A� *

loss�c�<���       �	���ec�A� *

loss��;�BQ       �	;o�ec�A� *

loss��<����       �	p	�ec�A� *

loss�*:n�C       �	N��ec�A� *

lossa��;xN+       �	�.�ec�A� *

lossxc�;vO�I       �	���ec�A� *

lossI[�:�OIk       �	���ec�A� *

loss#�;�=�n       �	N(�ec�A� *

lossn�;:�8�       �	���ec�A� *

loss�p+8��       �	Z��ec�A� *

loss��w<̽59       �	���ec�A� *

loss��88���       �	�:�ec�A� *

loss�"9 L��       �	���ec�A� *

loss��;���       �	_&�ec�A� *

loss���:[�       �	w��ec�A� *

lossﱕ=����       �	_�ec�A� *

loss �;��F       �	���ec�A� *

loss�5X8R��       �	��ec�A� *

loss�o<2Z6       �	+4 �ec�A� *

loss�2M=�s�7       �	9� �ec�A� *

loss=�:t|8       �	�!�ec�A� *

loss��!>6y�0       �	�"�ec�A� *

loss�o<u�       �	L#�ec�A� *

loss8�;��       �	��#�ec�A� *

losso#�:�{X�       �	B�$�ec�A� *

losshr�:7�+       �	B%�ec�A� *

loss��}<��       �	+�%�ec�A� *

loss�t�<Ϸ��       �	�q&�ec�A� *

lossHcQ;����       �	+'�ec�A� *

lossޞ�;�C6z       �	��'�ec�A� *

loss�';lЅT       �	�c(�ec�A� *

loss?a=�5h       �	.�(�ec�A� *

loss�N�=�x�v       �	�)�ec�A� *

loss���:�t�       �	�0*�ec�A� *

loss�:<$��       �	��*�ec�A� *

loss�E<�X7~       �	��+�ec�A� *

loss H=���;       �	�(,�ec�A� *

loss6�#=<=�i       �	G�,�ec�A� *

loss�G�=��QR       �	�`-�ec�A� *

loss���;f|V        �	��-�ec�A� *

loss��v;
#-�       �	y�.�ec�A� *

loss��$=��]�       �	�$/�ec�A� *

loss�T<���U       �	��/�ec�A� *

loss]m�;�i       �	�O0�ec�A� *

loss���:���       �	��0�ec�A� *

lossU<�N�       �	G�1�ec�A� *

loss�=ز�       �	)2�ec�A� *

losss�^<�Q=�       �	��2�ec�A� *

loss�7=�d       �	�[3�ec�A� *

loss��<H��       �	��3�ec�A� *

lossન<e�       �	i�4�ec�A� *

loss�'�;4���       �	I05�ec�A� *

loss �v; �%�       �	w�5�ec�A� *

loss�v�;@�Pg       �	�r6�ec�A� *

loss��N:6F�       �	�7�ec�A� *

loss�!;�(P�       �	�7�ec�A� *

lossZs#;�X�r       �	rR8�ec�A� *

loss�C�;䢬       �	��8�ec�A� *

loss&�K=�j       �	_�9�ec�A� *

loss:!�=̚�       �	n0:�ec�A� *

loss�j�:Q9U�       �	�:�ec�A� *

loss�̰<�|�       �	�p;�ec�A� *

lossNgY=��       �	0<�ec�A� *

lossm�;�Ԍ       �	�<�ec�A� *

loss��;z8S�       �	Ŭ=�ec�A� *

lossn&=�n�       �	�B>�ec�A� *

loss�1�<��       �	��>�ec�A� *

lossm��;��       �	�t?�ec�A� *

lossb�=kL��       �	�	@�ec�A� *

loss8Zb:���       �	��@�ec�A� *

loss�@�;���       �	;9A�ec�A� *

loss�P�<V       �	�XX�ec�A� *

lossI=��r�       �	�MY�ec�A� *

loss�S�=�v+       �	jZ�ec�A� *

lossCQ	=�X�       �	��Z�ec�A� *

loss�<[L3�       �	�o[�ec�A� *

loss�\\; �t�       �	:\�ec�A� *

loss��*=��sT       �	��\�ec�A� *

lossE�z;�aWK       �	��]�ec�A� *

loss��<Sg�       �	�:^�ec�A� *

loss=(͒�       �	��^�ec�A� *

loss�ޘ<*�L       �	÷_�ec�A� *

loss4E<���       �	��`�ec�A� *

lossmٕ<�t�       �	�a�ec�A� *

lossN��=gl�l       �	�0b�ec�A� *

lossF�6<�u�       �	5�b�ec�A� *

loss�e;��Ȍ       �	5�c�ec�A� *

lossi�:��       �	�Pd�ec�A� *

loss��M=�c�       �	e�ec�A� *

loss3{<|Et       �	�!f�ec�A� *

loss�9�<�1       �	]�f�ec�A� *

loss��;!���       �	8�g�ec�A� *

loss���<u|��       �	�h�ec�A� *

loss�;x"       �	%�h�ec�A� *

loss}Ld=���       �	Ni�ec�A�!*

loss+X�;|��
       �	��i�ec�A�!*

loss��;����       �	)�j�ec�A�!*

loss��;8�P       �	�]k�ec�A�!*

loss]=fAe       �	��k�ec�A�!*

loss2V;j���       �	*�l�ec�A�!*

loss{S�<R�       �	5*m�ec�A�!*

loss��=lw�       �	�m�ec�A�!*

loss<��       �	_n�ec�A�!*

loss=:<E a�       �	H�n�ec�A�!*

lossDE�<���#       �	�o�ec�A�!*

lossM#;9���       �	�`p�ec�A�!*

loss��;t�9       �	
�p�ec�A�!*

loss�O�:r� �       �	��q�ec�A�!*

loss��<F��       �	D�r�ec�A�!*

loss�^<�#U       �	��s�ec�A�!*

loss��y<Q&Bw       �	��t�ec�A�!*

loss��<�GH�       �	�Bu�ec�A�!*

loss��9<�Z�       �	��u�ec�A�!*

lossl��9��|       �	eov�ec�A�!*

loss�S�<ёJ       �	�w�ec�A�!*

loss�V�<2�       �	$�w�ec�A�!*

loss:o<��(       �	bNx�ec�A�!*

loss�0/=4��       �		�x�ec�A�!*

loss�7�<Fo��       �	��y�ec�A�!*

lossl{G;�'�       �	Y5z�ec�A�!*

loss_ʶ;0�       �	'�z�ec�A�!*

loss��n<�:��       �	�n{�ec�A�!*

losszn7<�)d�       �	�|�ec�A�!*

lossj� ;��       �	}�ec�A�!*

loss�*�<8,��       �	�}�ec�A�!*

loss��s;����       �	��~�ec�A�!*

loss%po;8p�       �	9��ec�A�!*

loss8�<��$       �	ۈ��ec�A�!*

loss#�:u�kG       �		N��ec�A�!*

loss�_	<xnP       �	���ec�A�!*

loss8M0=s~cg       �	K��ec�A�!*

loss'��=�X�E       �	[��ec�A�!*

lossd#�:�b�       �	ҋ��ec�A�!*

loss=�u;Ɂ_;       �	�#��ec�A�!*

loss�;;t7��       �	8���ec�A�!*

loss$_s:qq�B       �	gb��ec�A�!*

loss��n;�N�S       �	����ec�A�!*

loss�b<O�#v       �	ސ��ec�A�!*

loss��:S�_       �	'.��ec�A�!*

loss�);!�       �	�̈�ec�A�!*

loss�<
�%�       �	Nb��ec�A�!*

loss7O$;(��       �	Q1��ec�A�!*

lossf�5<����       �	:��ec�A�!*

lossᨕ<��'       �	E���ec�A�!*

loss&<�G"       �	X��ec�A�!*

loss�L;���W       �	4���ec�A�!*

loss�B�<���r       �	S��ec�A�!*

lossW2:9��       �	���ec�A�!*

loss� <���g       �	����ec�A�!*

loss�B�<#�f       �	�#��ec�A�!*

lossq݌<IЙ       �	�ec�A�!*

loss��<�΋\       �	:���ec�A�!*

loss*�<�^��       �	
H��ec�A�!*

loss,q�<n��9       �	�ݑ�ec�A�!*

lossmb�8��`       �	�p��ec�A�!*

loss��<�\;M       �	���ec�A�!*

lossw��<����       �	���ec�A�!*

loss�5<��5u       �	�=��ec�A�!*

lossX@�<���9       �	BД�ec�A�!*

loss��:����       �	�p��ec�A�!*

loss�;��j@       �	g
��ec�A�!*

loss���;1.R1       �	����ec�A�!*

loss�=L���       �	�/��ec�A�!*

loss�CE;���L       �	�K��ec�A�!*

loss;��<�Q�4       �	�t��ec�A�!*

lossD{�=���@       �	ŏ��ec�A�!*

lossnß<��t�       �	�;��ec�A�!*

loss���<\),�       �	���ec�A�!*

lossBd='A��       �	�6��ec�A�!*

loss1�;��T<       �	�ޝ�ec�A�!*

loss�<���B       �	1���ec�A�!*

loss��3<84�       �	�Y��ec�A�!*

loss�;����       �	���ec�A�!*

losst��<��_�       �	<��ec�A�!*

loss׍,;��+�       �	[��ec�A�!*

loss���;;��Q       �	I���ec�A�!*

loss4�='� �       �	���ec�A�!*

loss@<�R��       �	Ͱ��ec�A�!*

loss�":���       �	oJ��ec�A�!*

loss��O<�2�P       �	���ec�A�!*

loss��A=�]       �	o���ec�A�!*

loss\-s<��       �	���ec�A�!*

loss�7"=���)       �	H¦�ec�A�!*

lossu�<U��       �	�\��ec�A�!*

loss��
<��Ӟ       �	2 ��ec�A�!*

loss��2:8��L       �	휨�ec�A�!*

loss��p<>�0�       �	�>��ec�A�!*

loss8�<���       �	���ec�A�!*

lossAg';��g       �	����ec�A�!*

loss��<����       �	�(��ec�A�!*

loss�J2;�nd       �	�ī�ec�A�!*

lossj�!;�D�T       �	u��ec�A�!*

loss��<����       �	i��ec�A�!*

loss#;sea�       �	���ec�A�!*

loss��<�a �       �	ˆ��ec�A�!*

loss��_9͂u�       �	)��ec�A�!*

lossN��:qy       �	����ec�A�!*

loss��3;$���       �	Z*��ec�A�!*

loss�<��T�       �	M���ec�A�!*

loss�"�;`�m�       �	T��ec�A�!*

loss���;�xD8       �	���ec�A�!*

lossa9�:D��       �	���ec�A�!*

loss�rk;�&       �	>���ec�A�!*

loss�ZV:��a       �	�V��ec�A�!*

loss���::?�/       �	����ec�A�!*

lossC��;��       �	����ec�A�!*

loss&P�:�KS       �	�:��ec�A�!*

loss���="�eD       �	
ַ�ec�A�!*

loss))+;F%       �	Lq��ec�A�!*

loss��:Of�       �	Q��ec�A�!*

loss�&�<�N�       �	ٯ��ec�A�!*

loss��I:ɏWi       �	�J��ec�A�!*

loss��N<�\~�       �	���ec�A�!*

loss��=,���       �	���ec�A�!*

loss�ٓ<�|�       �	��ec�A�!*

loss���<}�'�       �	ɒ��ec�A�!*

loss�`=���p       �	A)��ec�A�!*

loss	�=n	�I       �	!Ⱦ�ec�A�!*

loss3�+;.�}       �	wi��ec�A�!*

loss3�<m%��       �	q��ec�A�!*

loss%�k;B��        �	Q���ec�A�!*

lossX,)<Ca8       �	�9��ec�A�!*

loss��;��{q       �	����ec�A�"*

loss%^<��PF       �	s��ec�A�"*

loss��=16m'       �	���ec�A�"*

loss���;��O<       �	*���ec�A�"*

loss�p9s��       �	�?��ec�A�"*

loss�f<_��E       �	v���ec�A�"*

loss��3<���%       �	����ec�A�"*

loss��B<����       �	J$��ec�A�"*

loss�S<}��       �	O���ec�A�"*

loss�'�<����       �	�u��ec�A�"*

lossIۡ:JE��       �	���ec�A�"*

loss,�0<�|��       �	=���ec�A�"*

loss��9T�q�       �	�Z��ec�A�"*

loss�d�<�&��       �	;���ec�A�"*

loss-2I;5�s       �	����ec�A�"*

lossOA�9ڶ�F       �	�4��ec�A�"*

loss�-�9�g��       �	z���ec�A�"*

loss�/,='F�       �	k���ec�A�"*

loss���<����       �	R���ec�A�"*

loss�k�=���       �	w���ec�A�"*

loss3O89ѿ}�       �	�(��ec�A�"*

loss�;�[�t       �	C���ec�A�"*

loss5�<����       �	����ec�A�"*

lossIWT;,��       �	B��ec�A�"*

loss��+=�͛�       �	����ec�A�"*

loss;�<��3       �	.���ec�A�"*

loss<3�=V)�e       �	�-��ec�A�"*

loss��<�Hx�       �	����ec�A�"*

loss�z�<��X       �	ji��ec�A�"*

loss���:�k       �	�,��ec�A�"*

loss��=���       �	���ec�A�"*

loss��@;����       �	n���ec�A�"*

lossҪ�:!�0�       �	bN��ec�A�"*

lossݖ�:���       �	.���ec�A�"*

loss�><]W6       �	���ec�A�"*

loss�٘<H�}�       �	�9��ec�A�"*

loss��D=��ʻ       �	y#��ec�A�"*

losshY_=�^2�       �	$���ec�A�"*

loss��<!�i�       �	f��ec�A�"*

lossu;�l       �	����ec�A�"*

loss|:�<�w��       �	%���ec�A�"*

loss.��<��       �	����ec�A�"*

loss��(<�-��       �	s���ec�A�"*

lossS�:��T       �	�m��ec�A�"*

lossԇ:��g       �	��ec�A�"*

loss8�;�c�       �	I���ec�A�"*

loss]:I<��       �	m6��ec�A�"*

loss��;cXQ|       �	����ec�A�"*

loss��\;WO�       �	�_��ec�A�"*

lossx�V=r�t�       �	����ec�A�"*

loss��1<D��,       �		���ec�A�"*

loss�>�=��'       �	^+��ec�A�"*

loss��;�8�       �	G���ec�A�"*

loss#{�<9��*       �	�[��ec�A�"*

loss��E=��=�       �	���ec�A�"*

loss�Ҥ<�ZCy       �	w���ec�A�"*

loss�;v=�L��       �	:#��ec�A�"*

loss���<;,0|       �	_���ec�A�"*

loss	�x<N�t       �		Q��ec�A�"*

loss�Κ;� �k       �	}���ec�A�"*

loss�<;��       �	ρ��ec�A�"*

loss��<.h^       �	�Y��ec�A�"*

loss��;�A�       �	����ec�A�"*

loss�l<���       �	e���ec�A�"*

lossPo=VZ�       �	_��ec�A�"*

lossN{>WY�       �	���ec�A�"*

lossm�b=L��:       �	����ec�A�"*

lossmV=E�Z       �	�#��ec�A�"*

lossߍ�<4       �	k���ec�A�"*

loss��;<mV �       �	]��ec�A�"*

loss&�6;�n&M       �	���ec�A�"*

loss;�<v��       �	����ec�A�"*

loss	�<����       �	+��ec�A�"*

loss��7=u�2#       �	"���ec�A�"*

loss	a<��H       �	�X��ec�A�"*

lossr/�:Rf3       �	���ec�A�"*

loss�C$<��
�       �	_%��ec�A�"*

loss�1<��+       �	����ec�A�"*

loss@��;�4�	       �	6W��ec�A�"*

loss�#�<6܇       �	����ec�A�"*

loss��;���       �	����ec�A�"*

loss�{�;u��-       �	���ec�A�"*

lossG/;���       �	����ec�A�"*

lossp�;QeC�       �	����ec�A�"*

loss��;�@��       �	MN��ec�A�"*

loss]7�;���0       �	����ec�A�"*

loss/��<S�s       �	#���ec�A�"*

loss��H;��;�       �	�?��ec�A�"*

loss��;��ɼ       �	����ec�A�"*

lossc[,<\Xt       �	h% �ec�A�"*

loss��F=H¡�       �	� �ec�A�"*

loss�M-=m���       �	���ec�A�"*

loss�0Y=��s       �	�i�ec�A�"*

loss��<�
S�       �	���ec�A�"*

loss�E=� 2�       �	�O�ec�A�"*

loss�%�9���       �	���ec�A�"*

loss�<�98�#�       �	,��ec�A�"*

loss�yo<��%       �	���ec�A�"*

lossZ[(:��       �	M��ec�A�"*

lossI~<RŻ�       �	D�ec�A�"*

lossF�k;#�k'       �	c��ec�A�"*

loss�4�<��/�       �	n	�ec�A�"*

lossF��9����       �	r�
�ec�A�"*

loss�1;�V�       �	�A�ec�A�"*

loss��y<tӳ       �	Q��ec�A�"*

lossܕ;מz�       �	Bx�ec�A�"*

loss8v|;�?�       �	r�ec�A�"*

loss�Z�<0A�%       �	h��ec�A�"*

loss���<]r�       �	K�ec�A�"*

lossD�P='\�?       �	���ec�A�"*

loss�-�;��+�       �	��ec�A�"*

lossi�<6Lf       �	r4�ec�A�"*

lossCT�9��#       �	���ec�A�"*

lossD�k<�38�       �	�v�ec�A�"*

loss1:0�l�       �	��ec�A�"*

loss3�;��o       �	��ec�A�"*

loss��<���       �	I�ec�A�"*

loss(:�=�e�       �	%��ec�A�"*

loss
��<@��]       �	j��ec�A�"*

lossQ�:ۻ݀       �	?8�ec�A�"*

loss�D<�g�       �	���ec�A�"*

loss�8=m��       �	r�ec�A�"*

lossf�
<���g       �	t�ec�A�"*

loss��=�=I       �	���ec�A�"*

lossj-�<�v]
       �	�M�ec�A�"*

loss�I�<���       �	���ec�A�"*

loss��M:�v4�       �	%��ec�A�"*

loss2<y��       �	�;�ec�A�"*

loss���<`���       �	��ec�A�#*

lossN+:�{2�       �	���ec�A�#*

loss]�:��E?       �	r1�ec�A�#*

lossd�u:RM}J       �	�$�ec�A�#*

loss)FX;g4�       �	���ec�A�#*

loss���;ӉC4       �	i�ec�A�#*

loss�f�<�,�       �	(�ec�A�#*

loss6�=�r5�       �	���ec�A�#*

loss��L<M��p       �	@i �ec�A�#*

loss�W9:�My       �	b!�ec�A�#*

lossh�:×�       �	v�!�ec�A�#*

loss��B:{z:|       �	�a"�ec�A�#*

loss��D;�8R�       �	<#�ec�A�#*

loss�`;9 l+�       �	��#�ec�A�#*

loss�^!<��	G       �	O[$�ec�A�#*

lossm�;��I       �	��$�ec�A�#*

loss��p<��*�       �	ܛ%�ec�A�#*

losscN<5ʪ       �	�7&�ec�A�#*

loss��(<}��       �	9'�ec�A�#*

loss��<d�#x       �	t�'�ec�A�#*

loss��<3���       �	�(�ec�A�#*

loss=�;K���       �	�-)�ec�A�#*

lossͻ;c��9       �	��)�ec�A�#*

loss�a<:��       �	a*�ec�A�#*

loss1�f<�2%�       �	[+�ec�A�#*

loss��<B��[       �	X�+�ec�A�#*

loss�=i�y       �	;R,�ec�A�#*

loss�j�;����       �	o�,�ec�A�#*

loss�S#;�=�       �	K�-�ec�A�#*

loss�;ݧ �       �	B.�ec�A�#*

lossڙ==,�j�       �	��.�ec�A�#*

loss���;N�V�       �	j�/�ec�A�#*

lossL�=��I@       �	�"0�ec�A�#*

lossL�<mW��       �	@�0�ec�A�#*

loss�J�:�OZ�       �	HS1�ec�A�#*

loss�V�<,�U       �	$�1�ec�A�#*

lossΠb=*���       �	��2�ec�A�#*

loss=v�:J�+�       �	�23�ec�A�#*

loss#�><T�X       �	��3�ec�A�#*

loss�=�2       �	p`4�ec�A�#*

lossQV>=�F�       �	��4�ec�A�#*

loss�}�< ��v       �	��5�ec�A�#*

lossx�=,^�       �	>%6�ec�A�#*

loss��<^�       �	o�6�ec�A�#*

loss��%<�@��       �	��7�ec�A�#*

loss��<vn$�       �	�98�ec�A�#*

loss"	=N�F       �	�8�ec�A�#*

loss�ZE<d_�|       �	Dj9�ec�A�#*

lossh�;� A       �	�:�ec�A�#*

loss�U�<�W�       �	��:�ec�A�#*

loss	��;�q�]       �	�w;�ec�A�#*

loss���<��
3       �	�3<�ec�A�#*

loss��< 3*�       �	5�=�ec�A�#*

loss�<'�H�       �	9�>�ec�A�#*

loss�S�;|e�       �	j2?�ec�A�#*

loss1�<�8�       �	��?�ec�A�#*

loss  e9Q�l       �	�|@�ec�A�#*

loss��<1���       �	�A�ec�A�#*

loss2�=4c�s       �	�GB�ec�A�#*

loss��<w3�@       �	��B�ec�A�#*

loss��:6�       �	,D�ec�A�#*

loss�v�;��c       �	��D�ec�A�#*

loss?T�:^];�       �	�ME�ec�A�#*

loss��<|�@�       �	S�E�ec�A�#*

loss�Cq9��i       �	9F�ec�A�#*

loss@��=�^8�       �	�G�ec�A�#*

loss�;�I?�       �	akH�ec�A�#*

loss,�q;��g>       �	�I�ec�A�#*

loss�7B=�K�'       �	��I�ec�A�#*

loss�P�<�%4g       �	�GJ�ec�A�#*

loss�$�<��       �	��J�ec�A�#*

lossX�<�To	       �	|K�ec�A�#*

loss�D:z��       �	L�ec�A�#*

lossV��=z�h�       �	�L�ec�A�#*

loss��<2<\       �	3RM�ec�A�#*

loss?}�;ő"       �	P�M�ec�A�#*

loss���;��^�       �	s�N�ec�A�#*

loss��;&R�C       �	�8O�ec�A�#*

loss�/�=����       �	S�O�ec�A�#*

loss�<:9����       �	s�P�ec�A�#*

loss/*�:a��       �	�RQ�ec�A�#*

loss���=A�&�       �	f�Q�ec�A�#*

loss��u<-���       �	��R�ec�A�#*

lossa`f<��x}       �	�1S�ec�A�#*

loss�[�;c%m�       �	�S�ec�A�#*

loss�q�<�8�	       �	�hT�ec�A�#*

loss��E=r#�       �	x	U�ec�A�#*

lossђ<k�W�       �	|�U�ec�A�#*

loss(�<�0�       �	�1V�ec�A�#*

loss֔�<W8F�       �	�V�ec�A�#*

lossC�c=2�Q�       �	�uW�ec�A�#*

loss��*<4@�       �	�X�ec�A�#*

loss�O�<z��       �	��X�ec�A�#*

loss�͞<�_#�       �	GY�ec�A�#*

losslA:!�ي       �	a[�ec�A�#*

loss�>^;���       �	~�[�ec�A�#*

loss��=a���       �	��\�ec�A�#*

losse�=hdfh       �	��]�ec�A�#*

loss���;E�?�       �	B"^�ec�A�#*

loss�bf;����       �	<�^�ec�A�#*

loss�I�<x���       �	�^_�ec�A�#*

loss�z:�p?f       �	:`�ec�A�#*

loss&�:���}       �	O�`�ec�A�#*

loss��=�u�       �	ra�ec�A�#*

loss��;hݢ`       �	�b�ec�A�#*

loss�,<Q�       �	��b�ec�A�#*

losss�=�hb       �	�:c�ec�A�#*

loss;& <���)       �	W�d�ec�A�#*

lossh�<���k       �	Tse�ec�A�#*

loss��=I|       �	�f�ec�A�#*

loss�b�:
0f�       �	�f�ec�A�#*

loss,��<�[	�       �	�mg�ec�A�#*

loss!6=c/3�       �	A�h�ec�A�#*

lossII�;��\\       �	i�ec�A�#*

loss��<s��       �	��i�ec�A�#*

loss�P�;��a       �	vj�ec�A�#*

loss���<��0\       �	k�ec�A�#*

loss��9G���       �	G�k�ec�A�#*

loss���;~�|       �	�Kl�ec�A�#*

lossu|;�L�       �	 �m�ec�A�#*

loss��<�a�       �	�n�ec�A�#*

loss�'�:�m��       �	@/o�ec�A�#*

lossIԥ;[�ڣ       �	��o�ec�A�#*

loss�g�;�9m       �	){p�ec�A�#*

loss&Ka<_�T8       �	�q�ec�A�#*

loss�:>���       �	|�q�ec�A�#*

loss��<4�#�       �	�_r�ec�A�#*

loss��?<]L:       �	��r�ec�A�#*

loss���9�M�V       �	��s�ec�A�$*

lossx`<���       �	\8t�ec�A�$*

loss��={�}/       �	��t�ec�A�$*

loss���<_Z�       �	�ou�ec�A�$*

loss*�=���       �	�v�ec�A�$*

loss��;f�+       �	�v�ec�A�$*

loss��P<t���       �	3Tw�ec�A�$*

loss��7<ݴ֨       �	�w�ec�A�$*

losst�(<�Z��       �	��x�ec�A�$*

loss<N��}       �	�<y�ec�A�$*

lossV�U:oVk       �	��y�ec�A�$*

lossT,�<	�       �	�|z�ec�A�$*

loss�E=Io�       �	�{�ec�A�$*

loss)�;�+J�       �	x�{�ec�A�$*

loss�l�;�|`       �	XU|�ec�A�$*

loss�_:W%��       �	��|�ec�A�$*

loss䵚=�r�k       �	.�}�ec�A�$*

loss��<��^�       �	¦~�ec�A�$*

lossV�<��       �	�b�ec�A�$*

loss�OZ<��:�       �	�v��ec�A�$*

loss@��<����       �	���ec�A�$*

loss���<n\1
       �	L���ec�A�$*

lossq&<�2��       �	C=��ec�A�$*

loss	X;�9^>       �	�ۂ�ec�A�$*

loss�`�;�ղt       �	!t��ec�A�$*

lossEy4<�l       �	n��ec�A�$*

loss#r�;���       �	[���ec�A�$*

lossAׄ<�h�!       �	!\��ec�A�$*

lossKz=Q��&       �		��ec�A�$*

loss��=��S       �	����ec�A�$*

lossC$�;�p�}       �	�6��ec�A�$*

lossdj�<CS       �	�v��ec�A�$*

loss�h�;*���       �	0��ec�A�$*

lossH�a;7�       �	����ec�A�$*

loss�N`=�}[       �	�@��ec�A�$*

loss��;�       �	؊�ec�A�$*

loss�IT;�� �       �	s��ec�A�$*

loss���9��s�       �	j��ec�A�$*

loss:�<�u�       �	����ec�A�$*

loss�4�<��       �	�N��ec�A�$*

loss���9���       �	��ec�A�$*

loss]@�<����       �	"���ec�A�$*

lossAl <�:�       �	K#��ec�A�$*

loss��<���*       �	vď�ec�A�$*

loss�y'<��ԋ       �	�a��ec�A�$*

loss�x�<��[�       �	����ec�A�$*

lossjb�;��y       �	����ec�A�$*

loss�5�=����       �	�0��ec�A�$*

loss���;2���       �	W̒�ec�A�$*

lossf�<nL��       �	?q��ec�A�$*

lossA�=ViB       �	���ec�A�$*

loss�ߋ:�W��       �	3���ec�A�$*

lossi��<QSf1       �	�?��ec�A�$*

loss���;()<�       �	�ڕ�ec�A�$*

loss��z;~
Y       �	R��ec�A�$*

lossa�1;��4       �	���ec�A�$*

lossv7�=B��       �	����ec�A�$*

lossǨ�;rt��       �	�l��ec�A�$*

loss�$�<���       �	���ec�A�$*

loss=	k9����       �	;T��ec�A�$*

lossrO<ẙ�       �	n���ec�A�$*

losso.�:��y       �		���ec�A�$*

loss9�8ȹt9       �	F��ec�A�$*

loss|R=�i�Y       �	���ec�A�$*

lossdz�;*FNq       �	����ec�A�$*

loss��9;��w       �	(,��ec�A�$*

loss_D<`Y��       �	Ҟ�ec�A�$*

loss�cJ<<*��       �	�v��ec�A�$*

loss}�^<���K       �	<��ec�A�$*

loss�<�<7���       �	����ec�A�$*

loss��#;��
       �	�C��ec�A�$*

loss�5:jXm       �	���ec�A�$*

losshR>����       �	倢�ec�A�$*

loss�t;�u�       �	T��ec�A�$*

loss��;wڷ       �	���ec�A�$*

loss�6<�F;�       �	O��ec�A�$*

loss
i�<�}��       �	� ��ec�A�$*

lossҪ1=$�f�       �	A���ec�A�$*

lossX�;#7^       �	82��ec�A�$*

loss�9=4�       �	 ��ec�A�$*

loss�*Q<���{       �	����ec�A�$*

loss!�;�jv�       �	�6��ec�A�$*

loss
#=ʞ?�       �	�ި�ec�A�$*

loss�4�<�A�       �	v���ec�A�$*

lossO}�;���+       �	]1��ec�A�$*

lossf29�Ù       �	a��ec�A�$*

loss�P`<�v�       �	���ec�A�$*

loss޸	;�ȳ�       �	*S��ec�A�$*

loss�s�;�4?g       �	F��ec�A�$*

lossSP0<�>�B       �	3���ec�A�$*

lossŠ<�7�       �	l$��ec�A�$*

lossU5=9a�       �	�Ʈ�ec�A�$*

loss(-=/h��       �	�h��ec�A�$*

loss���<���g       �	�	��ec�A�$*

loss�@; ��;       �	ګ��ec�A�$*

lossꞬ<�'��       �	�I��ec�A�$*

loss���;t&��       �	�
��ec�A�$*

lossTB�:��)J       �	X���ec�A�$*

loss�6r<Dd��       �	�A��ec�A�$*

loss���;�u�(       �	�A��ec�A�$*

lossm�?<�_��       �	���ec�A�$*

loss�J<��?l       �	ސ��ec�A�$*

loss�g=��m       �	8.��ec�A�$*

loss��9����       �	!ͷ�ec�A�$*

loss&�E<�i       �	�g��ec�A�$*

loss���9w���       �	��ec�A�$*

loss���<�P#�       �	����ec�A�$*

lossڲ�;J^       �	�5��ec�A�$*

lossr�=�5W       �	�Һ�ec�A�$*

loss
*�;vϞ�       �	o��ec�A�$*

loss���:��3j       �	c��ec�A�$*

loss��:];�F       �	ƣ��ec�A�$*

loss�h<6��       �	5��ec�A�$*

loss਑;�T�,       �	&��ec�A�$*

lossY<�9       �	b���ec�A�$*

loss��d:�Z�B       �	u��ec�A�$*

loss��=�-�(       �	����ec�A�$*

loss:�<���z       �	����ec�A�$*

loss��;~�ǡ       �	<���ec�A�$*

loss�	�<�F�       �	���ec�A�$*

loss-��9���R       �	a3��ec�A�$*

loss ��;\���       �	����ec�A�$*

lossf<1'��       �	�}��ec�A�$*

loss��;b�       �	���ec�A�$*

loss_B=Z�*       �	���ec�A�$*

loss��;����       �	�?��ec�A�$*

loss{��9	��       �	����ec�A�$*

loss��:��J       �	�{��ec�A�$*

loss<�9k.��       �	m��ec�A�%*

loss�Ru;'i[5       �	l���ec�A�%*

loss���:3Vz       �	�L��ec�A�%*

lossXR:��3       �	����ec�A�%*

loss���:7��       �	����ec�A�%*

loss#q8:8*��       �	���ec�A�%*

loss/g�8�       �	-���ec�A�%*

lossX�9Eݗ       �	K��ec�A�%*

loss_�;axf�       �	3���ec�A�%*

loss�!;��       �	����ec�A�%*

loss��<}pb�       �	h%��ec�A�%*

loss�D~:�e�       �	���ec�A�%*

loss�v�<�"D�       �	Ҧ��ec�A�%*

lossxz�=�>i�       �	B��ec�A�%*

lossZ�m973�       �	����ec�A�%*

loss��=t��a       �	t}��ec�A�%*

loss�z<XR�       �	h ��ec�A�%*

loss�I�<g��       �	ظ��ec�A�%*

loss�=�9�       �	�a��ec�A�%*

loss��=����       �	�T��ec�A�%*

loss@�<�X�       �	����ec�A�%*

lossƻN:��PG       �	
���ec�A�%*

loss�7~<UAX       �	;:��ec�A�%*

lossZW�:x})�       �	F��ec�A�%*

lossTH�<����       �	����ec�A�%*

loss�G<<�7       �	*���ec�A�%*

loss�`�<���;       �	-C��ec�A�%*

lossJ��<ˀ��       �	3���ec�A�%*

loss|�<����       �	�s��ec�A�%*

loss5�;>u<       �	�	��ec�A�%*

loss!�<u4       �	���ec�A�%*

loss���9��P�       �	�D��ec�A�%*

loss{�=z�M       �	J���ec�A�%*

lossb��:�ݓ�       �	;���ec�A�%*

loss�	�;�       �	�(��ec�A�%*

loss�A=�E�       �	]���ec�A�%*

loss&��;ѫ�j       �	O\��ec�A�%*

loss�_:���Z       �	�W��ec�A�%*

loss�f9Հ��       �	 ���ec�A�%*

lossO�<�p�       �	0���ec�A�%*

loss���=p�5       �	���ec�A�%*

loss�m�<P��       �	ҧ��ec�A�%*

loss%d=�(�       �	J��ec�A�%*

loss ��<�7�        �	����ec�A�%*

lossq�^;��%�       �	�t��ec�A�%*

lossO�@;�0�=       �	���ec�A�%*

loss�!<y��       �	n���ec�A�%*

loss��T=�scm       �	�E��ec�A�%*

loss
JF<�V?�       �	`���ec�A�%*

lossqc92�       �	C ��ec�A�%*

loss��<I�D       �	r���ec�A�%*

loss1�=��Q:       �	�X��ec�A�%*

loss8�+<��       �	����ec�A�%*

loss�c</$��       �	����ec�A�%*

loss��;�S�       �	Y2��ec�A�%*

lossNu:;�Vj�       �	R���ec�A�%*

loss�l< ��V       �	t��ec�A�%*

loss_�p;_25�       �	���ec�A�%*

loss�<;PG�       �	f���ec�A�%*

loss��K<)K(       �	�=��ec�A�%*

loss��=��       �	o���ec�A�%*

loss�p,;�Y9N       �	+i��ec�A�%*

loss�L�;��%�       �	a���ec�A�%*

loss�L(<�g�u       �	����ec�A�%*

loss�2�;j�F       �	�"��ec�A�%*

loss;���       �	��ec�A�%*

loss'x�=�1�       �	ܡ�ec�A�%*

loss)��;���       �	�5�ec�A�%*

loss6�;h���       �	���ec�A�%*

loss_F =��/	       �	�x�ec�A�%*

loss�q<�X��       �	k�ec�A�%*

loss�̡<g�&�       �	���ec�A�%*

loss�Q><����       �	�H�ec�A�%*

loss�� =���Y       �	���ec�A�%*

lossa�<EQ�       �	|{�ec�A�%*

loss�;7:@       �	��ec�A�%*

loss�UL="c�G       �	{��ec�A�%*

loss���<L��7       �	B>�ec�A�%*

lossN�V=�M��       �	���ec�A�%*

loss#�=��^       �	�p�ec�A�%*

loss�Y;�w~       �	��ec�A�%*

loss��:��G�       �	1��ec�A�%*

lossDY�=z>��       �	�B�ec�A�%*

loss�r�<}�ad       �	+��ec�A�%*

lossL�~<F �       �	�x�ec�A�%*

loss3)<`5̪       �	��ec�A�%*

loss��;>���7       �	v��ec�A�%*

loss �:��(3       �	QN�ec�A�%*

loss�tq=h�R�       �	���ec�A�%*

lossOr;�t�n       �	��ec�A�%*

loss9
=�S"�       �	��ec�A�%*

lossH�
<9̌m       �	��ec�A�%*

lossz�:[[F�       �	@Q�ec�A�%*

loss}�o<��k       �	�5 �ec�A�%*

loss�� =�.J       �	�� �ec�A�%*

loss�EC;"r�W       �	�b!�ec�A�%*

loss,��<ŏ
2       �	�"�ec�A�%*

loss��~<b�($       �	��"�ec�A�%*

loss&P�;�w��       �	0#�ec�A�%*

loss�I;�V:>       �	`�#�ec�A�%*

loss�t=�:�7       �	,e$�ec�A�%*

loss3n�;�,/�       �	 %�ec�A�%*

loss�Ą=�:�       �	/�%�ec�A�%*

loss��<��A       �	�L&�ec�A�%*

loss�Kf<�\Z       �	��&�ec�A�%*

loss�%<� n�       �	Й'�ec�A�%*

loss�{;Z�*�       �	�0(�ec�A�%*

loss���;��ƹ       �	,�(�ec�A�%*

lossTQ<[�>]       �	�)�ec�A�%*

loss���<d�}X       �	Bv*�ec�A�%*

loss�Z�<{́       �	�+�ec�A�%*

lossf�V=OǾ       �	�+�ec�A�%*

loss�4#=^�       �	~T,�ec�A�%*

loss�.|;��3y       �	8�,�ec�A�%*

loss
�:H��'       �	|�-�ec�A�%*

loss�Ԗ<�N�j       �	�U.�ec�A�%*

loss̀/<:�BA       �	�.�ec�A�%*

loss0�;,�>       �	΋/�ec�A�%*

loss|�=y��       �	�#0�ec�A�%*

loss�dI=�97�       �	��0�ec�A�%*

loss�49?�x�       �	�o1�ec�A�%*

loss���;?H�       �	G2�ec�A�%*

loss�c�9N�77       �	Ú2�ec�A�%*

loss�#;��G       �	�/3�ec�A�%*

loss��E<�d-�       �	�3�ec�A�%*

loss�=��       �	Y4�ec�A�%*

loss���<��o       �	�4�ec�A�%*

loss٬�<!�6�       �	r�5�ec�A�%*

loss�=�;=�AS       �	6�ec�A�&*

loss6��;7�e)       �	ٱ6�ec�A�&*

loss�FC:%�J       �	�F7�ec�A�&*

lossx%;ݴ^�       �	��7�ec�A�&*

lossLD�<�5�       �	'k8�ec�A�&*

lossDK�<-MV�       �	��8�ec�A�&*

lossVd�<���       �	C�9�ec�A�&*

lossm�5=ň�       �	6!:�ec�A�&*

loss�,<h�_       �	Ǽ:�ec�A�&*

loss�;qg�       �	�_;�ec�A�&*

loss���;�j�       �	�<�ec�A�&*

lossS@o=\Q{�       �	)�<�ec�A�&*

lossHI}<���s       �	�c=�ec�A�&*

loss���<M��       �	H�>�ec�A�&*

loss�_;�V�(       �	�>?�ec�A�&*

lossZ�<�x       �	��@�ec�A�&*

loss&��;Q�@A       �	�oA�ec�A�&*

loss��,=�C b       �	�TB�ec�A�&*

loss�R�<Sd{�       �	��B�ec�A�&*

loss�� ;2�׽       �	s�C�ec�A�&*

loss�a�<�a�u       �	�$D�ec�A�&*

loss.��;��       �	��D�ec�A�&*

lossN�;�C7       �	�[E�ec�A�&*

loss�xN<MV�c       �	��E�ec�A�&*

loss&Ö<����       �	r�F�ec�A�&*

loss���<�mxi       �	BBG�ec�A�&*

loss3�:����       �	*�G�ec�A�&*

loss��=dyW       �	|H�ec�A�&*

loss�F�<�A{       �	EI�ec�A�&*

loss�e�<���       �	��I�ec�A�&*

loss}h<���       �	y?J�ec�A�&*

lossxa�<%��       �	5�J�ec�A�&*

loss�,b;(D;       �	ZfK�ec�A�&*

loss�H=��A]       �	�K�ec�A�&*

loss �;�&0�       �	��L�ec�A�&*

loss��<��       �	&:M�ec�A�&*

lossW�:�^�        �	��M�ec�A�&*

loss �Q;�c�q       �	�}N�ec�A�&*

loss�:@�       �	�O�ec�A�&*

lossn(<;8�,*       �	��O�ec�A�&*

loss�s�<( \�       �	�_P�ec�A�&*

loss<�%=r4_�       �		�P�ec�A�&*

loss�Z�:Z�H�       �	��Q�ec�A�&*

loss;۴9�x�       �	;5R�ec�A�&*

loss&s6:�R�;       �	��R�ec�A�&*

loss���;�aB�       �	{S�ec�A�&*

loss,@<�=:�       �	�T�ec�A�&*

loss��N;%�       �	�T�ec�A�&*

lossAs;Z�@       �	"RU�ec�A�&*

loss�4%;5\�       �	6�U�ec�A�&*

loss���<"0@�       �	��V�ec�A�&*

loss�Ba:tF��       �	�'W�ec�A�&*

loss�m=��       �	�EX�ec�A�&*

loss��6;��{       �	��X�ec�A�&*

loss��<���       �	�wY�ec�A�&*

loss��i;|��       �	jZ�ec�A�&*

loss��;���       �	�Z�ec�A�&*

loss0�;       �	�D[�ec�A�&*

loss���;�t�-       �	��[�ec�A�&*

loss;��;���       �	�\�ec�A�&*

lossW��<YW$       �	�"]�ec�A�&*

loss x:���       �	7�]�ec�A�&*

loss�^.:u7�]       �	�^^�ec�A�&*

loss ��=J�4�       �	�^�ec�A�&*

loss=;%<���       �	��_�ec�A�&*

loss���<��w�       �	�"`�ec�A�&*

loss6�;4!�       �	�`�ec�A�&*

loss8Fr=f�{`       �	XWa�ec�A�&*

loss8��:$$R       �	��a�ec�A�&*

loss���;�mX-       �	��b�ec�A�&*

loss�8;��T�       �	#�c�ec�A�&*

loss4��<�nID       �	9+d�ec�A�&*

loss���;1}��       �	.�d�ec�A�&*

lossXӦ;�7~�       �	�ge�ec�A�&*

lossV�l<<�W       �	��e�ec�A�&*

loss;��9�C�       �	ޑf�ec�A�&*

loss�3�;?yE*       �	^.g�ec�A�&*

loss{��:�       �	��g�ec�A�&*

loss��<��Q/       �	�Xh�ec�A�&*

lossC��<qy�       �	t�h�ec�A�&*

lossF+�;�Ҕ       �	ѐi�ec�A�&*

loss
Q�<�q	       �	5j�ec�A�&*

lossӧJ;�3j       �	��j�ec�A�&*

loss�'�:Y%r       �	0dk�ec�A�&*

lossa:��|(       �	��k�ec�A�&*

loss$�:�:�       �	�l�ec�A�&*

loss�2�;�l��       �	�%m�ec�A�&*

loss�q�:d�Ӧ       �	��m�ec�A�&*

loss�z<J�       �	1\n�ec�A�&*

loss�0=-p       �	��n�ec�A�&*

lossn�;7:i�       �	��o�ec�A�&*

loss��;�E�       �	�7p�ec�A�&*

lossu�:p_�&       �	[�p�ec�A�&*

loss<A��       �	�eq�ec�A�&*

loss
v�;�N�j       �	��q�ec�A�&*

loss�=���       �	S�r�ec�A�&*

loss�jE<x�Z�       �	�as�ec�A�&*

loss-�;�Ⲕ       �	�t�ec�A�&*

loss�8:Ə(�       �	�t�ec�A�&*

loss�q	=�J9U       �	u;u�ec�A�&*

loss�Ż9���       �	/�u�ec�A�&*

loss�ǌ<ػAR       �	��v�ec�A�&*

loss��^;4ð:       �	�pw�ec�A�&*

lossڋ�;��T�       �	3�x�ec�A�&*

loss���<A@�n       �	�Ay�ec�A�&*

lossF<�݁z       �	C�y�ec�A�&*

loss�><���       �	�z�ec�A�&*

loss�{�<�_f�       �	O#{�ec�A�&*

lossm��=����       �	v�{�ec�A�&*

loss�P�;��`�       �	si|�ec�A�&*

losss*�<��l6       �	}�ec�A�&*

loss,�9غ       �	��}�ec�A�&*

lossT`3<�Ƴ�       �	P9~�ec�A�&*

loss`��<킒J       �	:�~�ec�A�&*

loss�B<Y7�       �	Vb�ec�A�&*

loss��;�~1�       �	T��ec�A�&*

loss�w;�x�       �	�ŀ�ec�A�&*

loss!<;�a;�       �	�_��ec�A�&*

loss��<���       �	���ec�A�&*

loss�:�;�L�       �	���ec�A�&*

lossi?�:*I�       �	ڐ��ec�A�&*

loss���=K��       �	f-��ec�A�&*

loss��J9���       �	�e��ec�A�&*

loss��N<kH�c       �	���ec�A�&*

loss{\2;H���       �	<0��ec�A�&*

loss_g:*�;       �	��ec�A�&*

lossM�;3�o0       �	z9��ec�A�&*

loss�~<�gp�       �	5щ�ec�A�&*

loss�@<�%�       �	<���ec�A�'*

lossI8�;���       �	�8��ec�A�'*

loss*�+;��2       �	%ϋ�ec�A�'*

loss�VL=�,�w       �	zn��ec�A�'*

lossvȆ;���       �	X ��ec�A�'*

loss�h`=���       �	B���ec�A�'*

loss��F=P	f	       �	�$��ec�A�'*

loss:�P=�{[       �	鶎�ec�A�'*

loss3�]:����       �	&r��ec�A�'*

loss(�=mK!       �	���ec�A�'*

lossܫ-<�
�8       �	5���ec�A�'*

lossa��:��       �	~9��ec�A�'*

loss��U;���       �	�Α�ec�A�'*

loss��;C	��       �	Eh��ec�A�'*

loss��:y��       �	���ec�A�'*

loss�=P!�       �	��ec�A�'*

loss�3
<�{gi       �	�1��ec�A�'*

loss�l<�6��       �	�ǔ�ec�A�'*

loss1. <'GS7       �	]o��ec�A�'*

lossV�=k{�       �	J
��ec�A�'*

lossi��;�r�       �	����ec�A�'*

loss���;��#       �	.:��ec�A�'*

lossQ�=�t�       �	sؗ�ec�A�'*

loss}R�<�i��       �	hu��ec�A�'*

loss��$<�if�       �	S	��ec�A�'*

loss��[=�zd�       �	���ec�A�'*

loss_�=T�p       �	�3��ec�A�'*

loss��,=�ba       �	�ǚ�ec�A�'*

loss��<��       �	�b��ec�A�'*

loss:+�=��ͩ       �	���ec�A�'*

lossÇ�;_�c       �	���ec�A�'*

loss\�<S���       �	�%��ec�A�'*

loss�E�;2O�       �	���ec�A�'*

loss�5;�*B       �	 U��ec�A�'*

loss���<zWp       �	-��ec�A�'*

loss�#<�]9k       �	܁��ec�A�'*

lossL=	Z�       �	���ec�A�'*

loss�#];�9�r       �	����ec�A�'*

loss-�B;J��       �	>��ec�A�'*

loss_�;�C
       �	�ѡ�ec�A�'*

loss�m�:����       �	{i��ec�A�'*

lossW@v<�4�R       �	����ec�A�'*

loss��<f��%       �	����ec�A�'*

loss\�8;"�͖       �	%��ec�A�'*

loss�ף<�u��       �	���ec�A�'*

losst�;,YA@       �	]Q��ec�A�'*

lossj�<:�4       �	4��ec�A�'*

loss��>�!*       �	���ec�A�'*

loss<�:Ц�6       �	�"��ec�A�'*

loss�<a}nV       �	f���ec�A�'*

lossMs�<��;:       �	gc��ec�A�'*

lossC�:;&)�       �	��ec�A�'*

loss�h�<�3�=       �	(���ec�A�'*

loss$��;Dט       �	}@��ec�A�'*

lossj��;X:R       �	j۪�ec�A�'*

lossz�F=�{��       �	R��ec�A�'*

lossmu�;s��       �	%��ec�A�'*

lossJ)�:�p�       �	j���ec�A�'*

lossw|?<���       �	9_��ec�A�'*

lossR�<
'�       �	����ec�A�'*

loss���;.\��       �	ڏ��ec�A�'*

loss���<��.       �	�,��ec�A�'*

loss��<	lv5       �	�ȯ�ec�A�'*

loss 	2:�q�       �	0b��ec�A�'*

loss��:수       �	D���ec�A�'*

loss��"<��       �	����ec�A�'*

loss��;/k9N       �	�8��ec�A�'*

loss�2�<a�&       �	.ʲ�ec�A�'*

loss�5�;�>�       �	g��ec�A�'*

loss\r�9���       �	�
��ec�A�'*

loss�!;� a�       �	���ec�A�'*

loss	�J;�h��       �	B��ec�A�'*

loss_� <��L       �	�ߵ�ec�A�'*

loss��>;�U��       �	g���ec�A�'*

lossC>w;��       �	���ec�A�'*

loss�׀;�%�       �	���ec�A�'*

loss��;�>c�       �	�D��ec�A�'*

loss�,J<�m�       �	���ec�A�'*

loss�b�<�0       �	W���ec�A�'*

loss
%j<\��       �	�}��ec�A�'*

loss׵[;�,�       �	���ec�A�'*

loss��9��["       �	����ec�A�'*

loss�r�<!�u;       �	�I��ec�A�'*

loss�U�<꧌k       �	���ec�A�'*

loss~<�       �	4���ec�A�'*

loss7ٖ:�       �	"��ec�A�'*

loss%M�:Ee��       �		��ec�A�'*

loss�װ;��v       �	���ec�A�'*

loss�`�:F�op       �	.��ec�A�'*

loss���;d�T�       �	.���ec�A�'*

loss���<���       �	Kr��ec�A�'*

loss-:�ʆ       �	8���ec�A�'*

loss��0;�x
&       �	"��ec�A�'*

lossaTu:*��       �	�,��ec�A�'*

loss�RP;9�	�       �	����ec�A�'*

loss�]q;�8       �	F{��ec�A�'*

loss:��9r`�4       �	E��ec�A�'*

loss�ǈ<x���       �	I���ec�A�'*

loss��;]blk       �	�5��ec�A�'*

loss��8��{       �	����ec�A�'*

loss�4k=��r�       �	q��ec�A�'*

lossO�:�7aw       �	��ec�A�'*

loss �9i&X�       �	����ec�A�'*

loss���<J-e�       �	=C��ec�A�'*

loss��:k���       �	Y���ec�A�'*

loss�g�<���       �	D���ec�A�'*

loss�M�<2��       �	� ��ec�A�'*

loss�'�=��s       �	j���ec�A�'*

loss�\=���       �	�Y��ec�A�'*

loss_�9'���       �	����ec�A�'*

loss{��;?��       �	����ec�A�'*

lossl1�<��       �	 ��ec�A�'*

loss>�=��       �	����ec�A�'*

lossb�:���       �	�}��ec�A�'*

loss��=��_       �	���ec�A�'*

loss�.9�Mq�       �	���ec�A�'*

lossL&}<c�       �	�F��ec�A�'*

loss*�=<�VK       �	����ec�A�'*

loss��9I|y       �	�q��ec�A�'*

loss�@<�]U       �	`��ec�A�'*

loss�AL<��6       �	���ec�A�'*

loss�g�=��y�       �	^M��ec�A�'*

lossZs�:}�j�       �	���ec�A�'*

loss$w:�&��       �	����ec�A�'*

loss} ;�nl�       �	� ��ec�A�'*

loss�5|9W��_       �	����ec�A�'*

loss��H;��       �	l]��ec�A�'*

loss2�<#B�       �	���ec�A�'*

lossq�p<2�n       �	J��ec�A�(*

loss�+�;��c�       �	F���ec�A�(*

loss>�=_�:       �	;���ec�A�(*

loss��p<$�f       �	�(��ec�A�(*

loss:�;�J�g       �	����ec�A�(*

lossoD�;�&�       �	Ww��ec�A�(*

loss�7<c�       �	�5��ec�A�(*

loss-��:��:       �	0���ec�A�(*

loss�@=�?�'       �	E���ec�A�(*

loss@�o<R��v       �	�A��ec�A�(*

lossT��<��i       �	����ec�A�(*

loss�Ò<5��       �	����ec�A�(*

lossה;#���       �	ta��ec�A�(*

loss��/<���}       �	�M��ec�A�(*

loss�m�<�^ {       �	q���ec�A�(*

lossO�=��K       �	Έ��ec�A�(*

loss�В9��[�       �	�)��ec�A�(*

loss\�Z<_E*       �	����ec�A�(*

lossEj=<��       �	0���ec�A�(*

loss�_=ߢ�f       �	�(��ec�A�(*

lossC�=8+��n       �	���ec�A�(*

loss��): 1��       �	�n��ec�A�(*

loss�W$;�="       �		��ec�A�(*

loss2�;�       �	9���ec�A�(*

loss)�<�hg�       �	�Q��ec�A�(*

loss�
=��       �	����ec�A�(*

loss��w<�h�f       �	�~��ec�A�(*

loss��:7 �]       �	�!��ec�A�(*

loss��u:���       �	����ec�A�(*

loss�X;<*>       �	 $��ec�A�(*

loss�V;�	M       �	#���ec�A�(*

loss&��:(_Lw       �	|��ec�A�(*

loss�&�;�k/�       �	>���ec�A�(*

loss�";b٥       �	eP��ec�A�(*

lossl�4=wrK       �	����ec�A�(*

loss�O�;��L�       �	ѐ��ec�A�(*

loss��G;�       �	�*��ec�A�(*

loss��#<E��       �	J���ec�A�(*

loss)��<��Z       �	�g��ec�A�(*

loss�	�:Î�       �	��ec�A�(*

loss�	=7�2w       �	?���ec�A�(*

loss��;�[�@       �	�J��ec�A�(*

loss;��:�6       �	����ec�A�(*

loss��g<{��Q       �	�S��ec�A�(*

loss2ܩ=��       �	����ec�A�(*

loss�;r;��gj       �	���ec�A�(*

loss;�<	��d       �	�3��ec�A�(*

loss��;s|�       �	0���ec�A�(*

loss���:�y?2       �	ߋ��ec�A�(*

loss8K�<6���       �	�-��ec�A�(*

loss��<x��       �	����ec�A�(*

loss3R�;����       �	�b��ec�A�(*

loss$h;] [       �	^.�ec�A�(*

loss%�;���0       �	ݴ�ec�A�(*

loss��:[�BR       �	�c�ec�A�(*

loss��<�       �	W��ec�A�(*

loss�W,<�+#�       �	F�ec�A�(*

loss.�<�4       �	 ��ec�A�(*

loss�	<0�Na       �	���ec�A�(*

loss�gH<�TPp       �	v�ec�A�(*

loss|�;��y       �	%��ec�A�(*

loss��:����       �	�	�ec�A�(*

loss-�;�O�       �	��	�ec�A�(*

lossz= $�5       �	�L
�ec�A�(*

loss�Ǣ;K�V        �	��
�ec�A�(*

loss�3D=�+@       �	���ec�A�(*

lossR��=*��[       �	�,�ec�A�(*

loss���<���       �	���ec�A�(*

lossT��;3��       �	�q�ec�A�(*

lossΒ<:���Y       �	=�ec�A�(*

loss<>;�
       �	X��ec�A�(*

loss�� <*��       �	_A�ec�A�(*

loss��;d|�       �	���ec�A�(*

loss̃<f�/�       �	s�ec�A�(*

loss���;�c��       �	c
�ec�A�(*

loss�3q<ۊz�       �	���ec�A�(*

lossF�2;����       �	�4�ec�A�(*

loss��{8���       �	���ec�A�(*

loss�R�;i�U       �	Z�ec�A�(*

loss�!V;�i{       �	��ec�A�(*

loss���<h�4�       �	֋�ec�A�(*

loss�@=�'�.       �	t#�ec�A�(*

loss�ɗ=`v�       �	��ec�A�(*

losss�5<ܒ�       �	S�ec�A�(*

loss�3<�J       �	���ec�A�(*

loss�D�<�5�       �	~�ec�A�(*

loss��	=n��       �	��ec�A�(*

loss���;�N��       �	���ec�A�(*

loss��;��c;       �	DO�ec�A�(*

loss��;���       �	L��ec�A�(*

loss� �;
K؍       �	Z��ec�A�(*

lossh�;��x       �	�ec�A�(*

lossɧo<����       �	���ec�A�(*

loss�6�:H��       �	kH�ec�A�(*

loss'��<N�       �	i��ec�A�(*

loss���<S=       �	���ec�A�(*

loss��;���=       �	��ec�A�(*

loss���8-��       �	���ec�A�(*

loss�%99�}�       �	I�ec�A�(*

lossTF�;��q       �	���ec�A�(*

loss�'<��i�       �	!=!�ec�A�(*

loss6�><���       �	��!�ec�A�(*

loss���9xV�       �	��"�ec�A�(*

loss��8\�k�       �	`x#�ec�A�(*

lossD�w<{;�       �	�$�ec�A�(*

loss�=={���       �	��$�ec�A�(*

lossē$=�̋�       �	+L%�ec�A�(*

loss�6K<uO]       �	�%�ec�A�(*

lossj
:/��       �	��&�ec�A�(*

lossM��:g�w       �	''�ec�A�(*

loss��:�F��       �	�(�ec�A�(*

lossxt=��8=       �	��(�ec�A�(*

loss�_�<m��       �	�5)�ec�A�(*

lossf��9R`�"       �	��)�ec�A�(*

loss�	�;ݝ��       �	��*�ec�A�(*

lossDs#<��y       �	�/+�ec�A�(*

loss`Ү:Y���       �	�+�ec�A�(*

loss��;        �	h,�ec�A�(*

loss��<r���       �	��,�ec�A�(*

loss�/<���       �	ڒ-�ec�A�(*

loss�׷<�͞�       �	�4.�ec�A�(*

loss�g�;%[       �	��.�ec�A�(*

loss_d�= a��       �	�{/�ec�A�(*

loss%�;g���       �	�)0�ec�A�(*

loss�r=Wb�       �	��0�ec�A�(*

loss�:��i       �	1^1�ec�A�(*

loss�	g=��       �	Z�1�ec�A�(*

loss�9�=q��       �	e�2�ec�A�(*

loss׾�;ǫ��       �	�J3�ec�A�)*

loss��B;*e�       �	��3�ec�A�)*

lossc�</c��       �	~�4�ec�A�)*

loss��@=Y�l�       �	�*5�ec�A�)*

lossA�/;4�       �	 �5�ec�A�)*

loss%�=��?{       �	�e6�ec�A�)*

loss=;q,��       �	[	7�ec�A�)*

loss�q�;E��B       �	ݴ7�ec�A�)*

loss��=1��C       �	�X8�ec�A�)*

loss���;b� �       �	��8�ec�A�)*

loss�ȡ;,���       �	h�9�ec�A�)*

loss-�<�D�(       �	 9:�ec�A�)*

lossNk<�\�x       �	��:�ec�A�)*

loss���:�m�F       �	N|;�ec�A�)*

loss��;�#%       �	�<�ec�A�)*

loss{�T<���       �	�<�ec�A�)*

loss&�=1���       �	Eb=�ec�A�)*

loss�^=���|       �	>�ec�A�)*

loss�Q:�O
.       �	��>�ec�A�)*

loss��O;�6<       �	}]?�ec�A�)*

loss��;j�~�       �	v�?�ec�A�)*

loss�4;���n       �	��@�ec�A�)*

loss�k�97��       �	PA�ec�A�)*

loss~�="3�       �	}�A�ec�A�)*

loss�<}9Ц       �	��C�ec�A�)*

loss��7;j(�n       �	`#D�ec�A�)*

loss��?=�(i       �	�D�ec�A�)*

lossMg8��y       �	�pE�ec�A�)*

loss\u;q�/#       �	�	F�ec�A�)*

loss�e2;��M       �	�F�ec�A�)*

loss�+<�(��       �	�HG�ec�A�)*

loss:-<���       �	�G�ec�A�)*

loss��==��C       �	�I�ec�A�)*

loss v<�]��       �	=�I�ec�A�)*

lossZ�8=�i�       �	HOJ�ec�A�)*

loss��;�<�       �	,�J�ec�A�)*

loss?W=���       �	K�ec�A�)*

loss�#=���       �	�5L�ec�A�)*

lossiW�<2&�t       �	��L�ec�A�)*

loss���:�m��       �	�hM�ec�A�)*

loss}m:<ۗb�       �	& N�ec�A�)*

loss`�:��f       �	[�N�ec�A�)*

loss]��;�^�L       �	�O�ec�A�)*

lossӘC=����       �	�P�ec�A�)*

loss"��;�I9=       �	ΩP�ec�A�)*

loss]�+:��w       �	wJQ�ec�A�)*

loss7�;�m �       �	�Q�ec�A�)*

loss
�<���       �	�{R�ec�A�)*

lossv�}=��@�       �	"S�ec�A�)*

loss�g�;���       �	k�S�ec�A�)*

loss��<��X�       �	DmT�ec�A�)*

lossZj|=.�       �		U�ec�A�)*

loss��a<�1�|       �	��U�ec�A�)*

loss;/��       �	 :V�ec�A�)*

lossq�w:c�       �	��V�ec�A�)*

lossi==D��       �	��W�ec�A�)*

loss1^L<����       �	N(X�ec�A�)*

loss��;��P�       �	z�X�ec�A�)*

loss�ҏ;`|�       �	�cY�ec�A�)*

loss�e�;�1�t       �	�Y�ec�A�)*

loss�@�:�YF       �	�Z�ec�A�)*

loss��@=D,b       �	�X[�ec�A�)*

lossO�;PD       �	2W\�ec�A�)*

loss�;u�Ŏ       �	�&]�ec�A�)*

loss�\X<$�k%       �	B^�ec�A�)*

loss��;b��       �	r�^�ec�A�)*

loss#9�;}��       �	o�_�ec�A�)*

loss� <�Fܸ       �	]`�ec�A�)*

lossM��<�1̡       �	��`�ec�A�)*

loss�]=���       �	�Ta�ec�A�)*

loss/O4:�#~       �	��a�ec�A�)*

loss�z�:�Y�       �	?�b�ec�A�)*

lossj�S;��!       �	�#c�ec�A�)*

loss��<���k       �	طc�ec�A�)*

loss���:�z*�       �	HOd�ec�A�)*

loss��=:�RO       �	�d�ec�A�)*

loss��<;���.       �	�ue�ec�A�)*

lossR)�<�(�
       �	5f�ec�A�)*

loss���;ٝ\�       �	<�f�ec�A�)*

loss$;��9�       �	�?g�ec�A�)*

loss/��<���        �	y"h�ec�A�)*

loss�=33�       �	�h�ec�A�)*

loss��0:1}-h       �	Nai�ec�A�)*

loss�$�:��_0       �	��i�ec�A�)*

loss�e=M�i       �	��j�ec�A�)*

loss {�;��       �	"�k�ec�A�)*

lossdJ�8z�m�       �	�/l�ec�A�)*

loss�N�96�-       �	U�l�ec�A�)*

loss��c:��X       �	ywm�ec�A�)*

loss�k�;��       �	�n�ec�A�)*

loss���:P��       �	��n�ec�A�)*

lossqo�7���       �	i�o�ec�A�)*

loss�Ո<^�<       �	n1p�ec�A�)*

lossro<��-s       �	��p�ec�A�)*

lossf�88��       �	��q�ec�A�)*

loss�^9f��       �	Xr�ec�A�)*

loss�i�8���Z       �	S�r�ec�A�)*

loss:�<]�        �	�Vs�ec�A�)*

loss�:�� �       �	��s�ec�A�)*

loss��y9��{�       �	f�t�ec�A�)*

losszB;��u       �	2u�ec�A�)*

loss�\�<ZU�       �	{�u�ec�A�)*

loss�9(ŀ�       �	�qv�ec�A�)*

loss�ݱ=�X��       �	kw�ec�A�)*

loss��|;���	       �	��w�ec�A�)*

lossj(�:רK�       �	Zx�ec�A�)*

loss�};hp�       �	��x�ec�A�)*

loss	�:Y��       �	��y�ec�A�)*

loss���<�Q�       �	��z�ec�A�)*

lossC��<��n       �	/1{�ec�A�)*

lossy"�;Ȣ��       �	��{�ec�A�)*

loss��w<w���       �	b|�ec�A�)*

loss�x�;��"       �	��|�ec�A�)*

loss�,9<=�*�       �	F�}�ec�A�)*

lossC�=���
       �	:~�ec�A�)*

loss�Cf:��t�       �	(�~�ec�A�)*

loss�a�;��,H       �	Yi�ec�A�)*

loss��<���       �	>��ec�A�)*

loss"��<2�C@       �	ע��ec�A�)*

loss��5<؋       �	�=��ec�A�)*

loss���=�l""       �	9Ձ�ec�A�)*

loss<m;�v�i       �	ӡ��ec�A�)*

loss�/<!�20       �	�:��ec�A�)*

loss���;�أ�       �	�փ�ec�A�)*

loss�1=<���       �	�q��ec�A�)*

loss_�9kGp       �	���ec�A�)*

lossl�<��       �	쥅�ec�A�)*

loss��;� �       �	Ӈ�ec�A�)*

loss���<����       �	?q��ec�A�**

loss_��=N	�       �	���ec�A�**

loss���<J��       �	����ec�A�**

loss��n<Q��       �	W���ec�A�**

loss�c�<iRM�       �	C7��ec�A�**

loss�T<@CS-       �	�ϋ�ec�A�**

loss�h�<�น       �	�m��ec�A�**

lossS��9�n�       �	���ec�A�**

loss��X=mCz�       �	4ٍ�ec�A�**

loss4Y
;�*�       �	ލ��ec�A�**

losse�e:���       �	�%��ec�A�**

loss�L6=>כ        �	幏�ec�A�**

loss�o =!�.�       �	R��ec�A�**

loss��;���R       �	���ec�A�**

lossv�:V��       �	���ec�A�**

loss,�;���       �	$��ec�A�**

loss6g<�qm       �	�ƒ�ec�A�**

loss�.6;Ir       �	�k��ec�A�**

loss�,;��r       �	u!��ec�A�**

loss>&;��W�       �	̔�ec�A�**

loss\=�T�v       �	n��ec�A�**

lossP�:q[��       �	4��ec�A�**

losse�l<��$�       �	Ȗ�ec�A�**

loss���:�<o       �	{j��ec�A�**

lossWb<6�;;       �	���ec�A�**

loss_�$:�Wys       �	�2��ec�A�**

loss�)*<��1[       �	�Ʈ�ec�A�**

loss?�;zGnZ       �	�]��ec�A�**

loss�t;�b�@       �	7���ec�A�**

loss�� ;����       �	9%��ec�A�**

loss�s7=����       �	M���ec�A�**

loss��;�X�p       �	�`��ec�A�**

loss��~=�"��       �	���ec�A�**

loss�
<���       �	3���ec�A�**

loss�u=k�9H       �	�2��ec�A�**

loss��;[J�       �	�մ�ec�A�**

loss��<1��       �	�w��ec�A�**

lossb>�)�       �	C��ec�A�**

lossA+�;�r$r       �	,Ӷ�ec�A�**

lossb�;����       �	jk��ec�A�**

loss:K>;�W]�       �	���ec�A�**

lossA=�:l��,       �	����ec�A�**

losst1z<�r��       �	J��ec�A�**

lossgs<���       �	zĺ�ec�A�**

loss���=�/�/       �	_`��ec�A�**

loss�֛=�J$       �	N��ec�A�**

loss��j<��wT       �	Z���ec�A�**

loss\,�<2�       �	\V��ec�A�**

lossd��=)J��       �	O��ec�A�**

loss�n=̢/       �	����ec�A�**

loss�K�;.�3�       �	\V��ec�A�**

loss4i�<��       �	����ec�A�**

loss��<��2       �	c���ec�A�**

loss1��;mپ�       �	82��ec�A�**

loss��4<m�       �	&���ec�A�**

lossn��;��Y       �	����ec�A�**

loss��=���       �	`���ec�A�**

loss-�<ҵ�)       �	7���ec�A�**

loss�I�<-�n       �	����ec�A�**

lossQ6<$P|9       �	K��ec�A�**

lossϗ�<�#�       �	@���ec�A�**

loss{*J<x�de       �	`��ec�A�**

loss��=�Ѣ�       �	�N��ec�A�**

loss��<s��       �	Q.��ec�A�**

loss�:j<�W�       �	a���ec�A�**

lossCWt=�ք�       �	)���ec�A�**

lossdnC:͂u�       �	����ec�A�**

loss `E;��n|       �	����ec�A�**

loss�Y�;n�       �	5{��ec�A�**

loss�;��I       �	y#��ec�A�**

lossH�u<�(�'       �	����ec�A�**

loss�G=�R��       �	����ec�A�**

loss�;$3��       �	���ec�A�**

loss�,<��E       �	�@��ec�A�**

loss���:�z+j       �	����ec�A�**

loss�5�;5Ka�       �	���ec�A�**

loss�b6=s��z       �	�o��ec�A�**

loss�ҹ;�ЗN       �	���ec�A�**

loss_�=���       �	-���ec�A�**

loss�$�:���       �	�P��ec�A�**

loss��;�֗�       �	����ec�A�**

lossv�S:h��       �	m���ec�A�**

loss��D<�Н|       �	o+��ec�A�**

loss i=�XFR       �	����ec�A�**

lossw�;�bC#       �	�g��ec�A�**

loss�=����       �	���ec�A�**

loss�V<:]/��       �	���ec�A�**

loss-#+;X��P       �	T9��ec�A�**

loss%!�<���       �	����ec�A�**

loss�J:�Qy	       �	z��ec�A�**

loss�Ƚ<�/ [       �	���ec�A�**

loss�;=�m       �	K���ec�A�**

loss�0><�/h:       �	�P��ec�A�**

loss���<(U�       �	����ec�A�**

loss���;����       �	����ec�A�**

loss6!�<e�i.       �	d"��ec�A�**

loss��:$�3#       �	����ec�A�**

loss��;��       �	�a��ec�A�**

loss=`B^3       �	n���ec�A�**

loss�=�;]-9       �	���ec�A�**

loss�m<6�D?       �	�7��ec�A�**

loss��9�6�       �	����ec�A�**

loss%�;�3�       �	�p��ec�A�**

loss���;�k¨       �	���ec�A�**

loss�<�=���E       �	Ѯ��ec�A�**

loss���<�&L       �	|D��ec�A�**

loss�]<��       �	����ec�A�**

loss@�<F���       �	�)��ec�A�**

loss=��8�.E       �	e���ec�A�**

loss�L<��r       �	����ec�A�**

loss��:g��f       �	�&��ec�A�**

lossO'�:�:�O       �	'���ec�A�**

loss�US=�Ѓ�       �	�Z��ec�A�**

loss*�;7S��       �	����ec�A�**

loss/�.:Ie �       �	����ec�A�**

lossM�:$x�\       �	?��ec�A�**

loss�� =e���       �	����ec�A�**

loss�)$;�y�       �	'L��ec�A�**

loss�v3<P�o       �	����ec�A�**

loss�]�;�~ON       �	���ec�A�**

loss��?<���       �	���ec�A�**

loss'[=���V       �	غ��ec�A�**

loss�Ȱ<���       �	�_��ec�A�**

lossD��<��z       �	����ec�A�**

loss<�;�?��       �	f���ec�A�**

lossps;�       �	RF��ec�A�**

loss���;�˨�       �	����ec�A�**

loss  Q:�$       �	���ec�A�**

loss!��:�[�G       �	�5��ec�A�+*

loss�Ke:���       �	)���ec�A�+*

lossag:���-       �	�p��ec�A�+*

loss�p�:��BV       �	�%��ec�A�+*

lossr�_:�|h�       �	����ec�A�+*

lossq�
=#�*�       �	ga��ec�A�+*

loss�O�;3qbY       �	����ec�A�+*

loss1;�$��       �	���ec�A�+*

lossAs!<3�;�       �	NC��ec�A�+*

loss{�?=XW�       �	����ec�A�+*

loss�z�;��8       �	�~��ec�A�+*

loss�=B.       �	���ec�A�+*

loss��9��ہ       �	����ec�A�+*

losssnN<y�b�       �	_��ec�A�+*

lossc-=<��       �	����ec�A�+*

loss�?�<9�k       �	M���ec�A�+*

loss���<iց       �	�@ �ec�A�+*

loss�&<�W;�       �	X� �ec�A�+*

lossv��<\f�       �	0��ec�A�+*

loss��;�A��       �	/��ec�A�+*

loss�P<.^V�       �	~��ec�A�+*

losshP�:�ż�       �	wL�ec�A�+*

loss|Z=<�zμ       �	���ec�A�+*

lossV�<�6/�       �	e��ec�A�+*

lossч<}�_       �	��ec�A�+*

lossM�k<��       �	+�ec�A�+*

loss�S=�p��       �	�
�ec�A�+*

loss�9M=��,M       �	�
�ec�A�+*

loss �.<��H       �	L�ec�A�+*

loss���<�l�)       �	e��ec�A�+*

lossM�:��.       �	�|�ec�A�+*

loss�O�<Q,��       �	�&�ec�A�+*

lossh��;�a�5       �	��ec�A�+*

lossU��;��|       �	7n�ec�A�+*

loss��9;���       �	�ec�A�+*

loss�o;��D�       �	���ec�A�+*

loss�0�:��	       �	�a�ec�A�+*

loss�M�9�A��       �	��ec�A�+*

loss$�h<���       �	��ec�A�+*

lossͿ=�E�L       �	�?�ec�A�+*

lossOL�<ַ�e       �	<��ec�A�+*

lossl��=/�       �	�y�ec�A�+*

lossi�U:�Hɉ       �	�ec�A�+*

loss�o8q�       �	R��ec�A�+*

loss��P<3���       �	�R�ec�A�+*

loss�U9�kH�       �	x��ec�A�+*

loss���:�^��       �	��ec�A�+*

loss}'E:�y}       �	=+�ec�A�+*

loss���<6i��       �	���ec�A�+*

loss<�
;$r       �	�g�ec�A�+*

lossQ�r<�0��       �	0�ec�A�+*

lossm�;*m�       �	���ec�A�+*

lossj9
<S^,�       �	�@�ec�A�+*

loss.s;gՇ�       �	���ec�A�+*

loss�/�<D�G       �	^��ec�A�+*

loss1�<�/�Q       �	� �ec�A�+*

loss��1;���       �	b��ec�A�+*

lossQ�=��S       �	Qk�ec�A�+*

lossT�<�v�]       �	��ec�A�+*

loss0�:���       �	k��ec�A�+*

loss�K;�^�       �	�X�ec�A�+*

loss6�=_���       �	<��ec�A�+*

loss?z3=>�7       �	�� �ec�A�+*

lossxI9����       �	]4!�ec�A�+*

lossE�<�$P�       �	p�!�ec�A�+*

loss�ɖ:U��i       �	�"�ec�A�+*

loss5�:6���       �	�#�ec�A�+*

loss�Wr;�V�&       �	�3$�ec�A�+*

loss���:�FC�       �	��$�ec�A�+*

lossu1�;�.^)       �	�%�ec�A�+*

loss%X�=e�[�       �	�/&�ec�A�+*

loss��!<t,i"       �	��&�ec�A�+*

loss�>9ܖ�       �	Sv'�ec�A�+*

loss��<�T��       �	.(�ec�A�+*

loss��<���       �	T�(�ec�A�+*

loss�C<�%<X       �	�l)�ec�A�+*

loss#�:R�]       �	*�ec�A�+*

loss��X:�~�s       �	�*�ec�A�+*

lossa�E=S?�D       �	�W+�ec�A�+*

loss��G=@px       �	�,�ec�A�+*

loss�\�;�n�       �	ɭ,�ec�A�+*

lossj�J;�.�2       �	��-�ec�A�+*

loss),�:��^�       �	�E.�ec�A�+*

loss��=���       �	j�.�ec�A�+*

loss�L�;G�       �	��/�ec�A�+*

lossb=GCB       �	,.0�ec�A�+*

loss�#<KM�       �	
�0�ec�A�+*

lossCؠ=2�J�       �	�w1�ec�A�+*

lossrw<M�?�       �	�2�ec�A�+*

loss�p�<l��       �	�2�ec�A�+*

lossQY�;@�x       �	 c3�ec�A�+*

loss
,�;���       �	�4�ec�A�+*

loss�V�:q�}�       �	�4�ec�A�+*

loss���:0;�.       �	�S5�ec�A�+*

loss��=Im�I       �	��5�ec�A�+*

lossxwq<��"       �	o�6�ec�A�+*

loss��<�k��       �	�H7�ec�A�+*

loss`��;���       �	�v8�ec�A�+*

lossR��;�T�       �	�M9�ec�A�+*

loss�W9����       �	K�9�ec�A�+*

loss�+�;[�V       �	�;�ec�A�+*

lossf =
+[�       �	Z�;�ec�A�+*

loss��:���       �	t{<�ec�A�+*

lossA��<��p�       �	|�=�ec�A�+*

loss!0<��$�       �	=>�ec�A�+*

loss��<b/z�       �	w�>�ec�A�+*

loss2^�<��2�       �	�v?�ec�A�+*

lossi�=���       �	@�ec�A�+*

loss�9�<���	       �	m�@�ec�A�+*

loss��
;�DU�       �	�lA�ec�A�+*

loss �E;O�h�       �	
B�ec�A�+*

loss�Q�:��P�       �	h�B�ec�A�+*

loss��d<ɫ�
       �	�E�ec�A�+*

loss�U�=�Z،       �	�F�ec�A�+*

lossX�=Ƅ�7       �	�F�ec�A�+*

loss�<Ad��       �	�G�ec�A�+*

loss��u=�f��       �	��H�ec�A�+*

lossZ�;8�7�       �	p`I�ec�A�+*

losss
�<�"6�       �	hJ�ec�A�+*

lossz��:6�r�       �	��J�ec�A�+*

loss_g;��[�       �	�kK�ec�A�+*

loss��<�%�c       �	L�ec�A�+*

lossJjH;����       �	D�L�ec�A�+*

loss���<��A       �	�jM�ec�A�+*

lossD�:;c�ɲ       �	7N�ec�A�+*

loss��m<�r��       �	��N�ec�A�+*

loss�H�;)�`       �	�~O�ec�A�+*

loss�:8}�       �	d#P�ec�A�+*

loss��#9z�/�       �	��P�ec�A�,*

loss�<I�D�       �	��Q�ec�A�,*

loss&[<�       �	^.R�ec�A�,*

loss\�<T�       �	h�R�ec�A�,*

lossv��:�T*u       �	QkS�ec�A�,*

loss�<����       �	�T�ec�A�,*

loss8�A=S��       �	��T�ec�A�,*

loss��;��/       �	�dU�ec�A�,*

loss�Z9;8��       �	�V�ec�A�,*

lossn	4<4�[�       �		�V�ec�A�,*

loss'=�n��       �	�iW�ec�A�,*

loss�(<L�v�       �	�X�ec�A�,*

lossj�F<� S       �	j�X�ec�A�,*

loss�u�;��#�       �	�VY�ec�A�,*

loss,��9g��       �	+�Y�ec�A�,*

lossM�<�s�       �	��Z�ec�A�,*

lossF@�;��=       �	SB[�ec�A�,*

lossrN;��:r       �	D�[�ec�A�,*

loss�$<����       �	�\�ec�A�,*

loss@2=ZN**       �	1%]�ec�A�,*

lossͤ*=	ʖ       �	��]�ec�A�,*

lossd�";�I߮       �	~t^�ec�A�,*

loss��9U���       �	�$_�ec�A�,*

loss���9�gg       �	��_�ec�A�,*

lossN��:H�?       �	mp`�ec�A�,*

lossf$�:���       �	�a�ec�A�,*

loss��9:T�L�       �	E�a�ec�A�,*

loss@;�;ϙ��       �	�{b�ec�A�,*

loss��-=���a       �	Lc�ec�A�,*

loss<��i       �	V�c�ec�A�,*

loss��<zrA       �	S[d�ec�A�,*

loss�*<USs�       �	��d�ec�A�,*

loss��<C^��       �	��e�ec�A�,*

loss��8;�w7<       �	�;f�ec�A�,*

loss=Wr:�@uX       �	w�f�ec�A�,*

lossz �:���       �	?�g�ec�A�,*

lossX��<��
;       �	1h�ec�A�,*

loss�<_�E�       �	�h�ec�A�,*

loss�<'y��       �	){i�ec�A�,*

loss#q<���       �	j�ec�A�,*

loss\B�<_�       �	��j�ec�A�,*

loss���<�	       �	OWk�ec�A�,*

loss�%;z���       �	��k�ec�A�,*

loss���>=��       �	��l�ec�A�,*

loss+p"=���H       �	�2m�ec�A�,*

loss=F+9iYF�       �	��m�ec�A�,*

loss%{88'�       �	�nn�ec�A�,*

loss�5;6�"       �	�o�ec�A�,*

loss�*�<��n       �	��o�ec�A�,*

loss��Z:e}7/       �	mWp�ec�A�,*

loss��=�gj       �	C q�ec�A�,*

loss��<�kw�       �	��q�ec�A�,*

loss��=a~��       �	q�r�ec�A�,*

loss�׬=t�^{       �	(s�ec�A�,*

loss	�;@M8R       �	�%t�ec�A�,*

loss�"=���       �	6�t�ec�A�,*

loss�
t;t^��       �	uu�ec�A�,*

loss��<��       �	Mv�ec�A�,*

loss���;�r       �	b�v�ec�A�,*

loss]�;��       �	��w�ec�A�,*

loss��<]��       �	[Cx�ec�A�,*

loss�q�9��=�       �	��x�ec�A�,*

loss�x;�JK       �	P�y�ec�A�,*

loss��;gȪ�       �	�.z�ec�A�,*

loss��<L2Ѷ       �	�z�ec�A�,*

losst�<<�kw       �	�f{�ec�A�,*

loss`Y�<�G�       �	S|�ec�A�,*

loss)�<��V       �	�|�ec�A�,*

loss8�;�@�        �	�9}�ec�A�,*

lossI�<��6f       �	]o~�ec�A�,*

loss��;_g�       �	��ec�A�,*

loss�&;���       �	��ec�A�,*

loss�� <�L(�       �	UN��ec�A�,*

loss�`�<2��       �	���ec�A�,*

loss���;��C%       �	Έ��ec�A�,*

loss�T=����       �	) ��ec�A�,*

loss� �=�-��       �	���ec�A�,*

loss"�;%��8       �	o���ec�A�,*

loss�v=;��L\       �	�ׄ�ec�A�,*

loss�K:���;       �	Cƅ�ec�A�,*

loss�r�:l�{        �	�h��ec�A�,*

loss�1><�R'        �	0��ec�A�,*

loss��b:aZ�       �	ٵ��ec�A�,*

losst0)=��J6       �	�U��ec�A�,*

loss�@+<%�,�       �	i���ec�A�,*

loss
b:o��       �	М��ec�A�,*

loss=C9;����       �	h<��ec�A�,*

lossq�I:���       �	S��ec�A�,*

loss�i�<��F�       �	����ec�A�,*

loss�<=<�Yi�       �	E)��ec�A�,*

loss��;�W�e       �	�ƌ�ec�A�,*

loss��<'o�(       �	f��ec�A�,*

loss\��8��       �	 ��ec�A�,*

lossa�;}�D�       �	�ec�A�,*

loss��<�^#       �	�4��ec�A�,*

lossv�=��       �	�ҏ�ec�A�,*

loss�d�:��       �	�q��ec�A�,*

loss���:���       �		��ec�A�,*

loss�\<^���       �	ퟑ�ec�A�,*

lossw�I<�7�       �	�<��ec�A�,*

loss�$�;g"[�       �	-ђ�ec�A�,*

loss��9X�J       �	Dm��ec�A�,*

loss:V;P%��       �	���ec�A�,*

loss��4<3I�       �	Ϥ��ec�A�,*

loss E=WI��       �	�7��ec�A�,*

loss�3�<�u�"       �	̕�ec�A�,*

lossL�O<4~V       �	=`��ec�A�,*

loss܈*=��"       �	���ec�A�,*

loss���9 ��       �		���ec�A�,*

loss��;��7k       �	N��ec�A�,*

loss&	p;G��       �	2��ec�A�,*

loss�O�9Z��       �	����ec�A�,*

loss�"<��v}       �	N&��ec�A�,*

loss���=�E}�       �	���ec�A�,*

loss�� <���B       �	�N��ec�A�,*

loss��~;�2'O       �	��ec�A�,*

losssB�<ۅ�e       �	,���ec�A�,*

loss��N=O��_       �	j/��ec�A�,*

loss�k�::O6       �	՝�ec�A�,*

loss�9mX�<       �	�}��ec�A�,*

loss]�; 2�j       �	����ec�A�,*

loss/_==L��8       �	�P��ec�A�,*

lossZ��:����       �	����ec�A�,*

lossq s<�O6�       �	����ec�A�,*

loss�":m�{�       �	��ec�A�,*

loss��J;<ӳ�       �	|���ec�A�,*

loss�t
=�w�       �	���ec�A�,*

lossh�;)r��       �	T���ec�A�,*

loss#͂;�       �	�=��ec�A�-*

loss��*:#D�       �	�ܦ�ec�A�-*

lossA�:�lU�       �	J��ec�A�-*

loss%��:���       �	� ��ec�A�-*

lossc5&=YR0r       �	L���ec�A�-*

loss�>=�+S�       �	]n��ec�A�-*

lossMyK<*�
       �	~��ec�A�-*

loss��;�"�i       �		Ī�ec�A�-*

loss��:�+�       �	]k��ec�A�-*

loss\��;w0W       �	���ec�A�-*

loss�ī<���       �	)���ec�A�-*

loss��e<+3�J       �	���ec�A�-*

loss�3}<�	@       �	�)��ec�A�-*

lossT�A;�f3�       �	~Ǯ�ec�A�-*

loss�%�<���>       �	�\��ec�A�-*

loss)I;�i��       �	����ec�A�-*

loss���;����       �	����ec�A�-*

loss,W=�X�       �	^��ec�A�-*

loss��=<H� �       �	0��ec�A�-*

loss/��<��8�       �	����ec�A�-*

loss4(P<��w@       �	m��ec�A�-*

loss��+:��b}       �	����ec�A�-*

loss�P�:H��       �	�X��ec�A�-*

loss/�==
��        �	��ec�A�-*

lossJ><��       �	M���ec�A�-*

loss<ι:��+%       �	/��ec�A�-*

loss��<M���       �	y���ec�A�-*

loss��":!�7p       �	-B��ec�A�-*

loss�=���p       �	�շ�ec�A�-*

loss_Z;1��j       �	Ul��ec�A�-*

loss?�;і^�       �	��ec�A�-*

loss��w=��]�       �	����ec�A�-*

loss��k;q.&       �	i:��ec�A�-*

loss���;0�c�       �	Mٺ�ec�A�-*

loss�"�<ܖ�{       �	�w��ec�A�-*

loss�n�<�O       �	���ec�A�-*

loss���=����       �	�Ƽ�ec�A�-*

loss���9���       �	 ^��ec�A�-*

lossZ::�#�O       �	(���ec�A�-*

loss�\@;�vy�       �	Ĕ��ec�A�-*

loss���;i�E�       �	3��ec�A�-*

lossu�<��'       �	dʿ�ec�A�-*

lossq�<QN8       �	�h��ec�A�-*

loss��A=�E�       �	��ec�A�-*

loss���=ob3�       �	:���ec�A�-*

lossx�'<w]ld       �	�4��ec�A�-*

loss�E�<}��       �	h���ec�A�-*

loss��;���*       �	����ec�A�-*

loss%~�<Y��w       �	-��ec�A�-*

loss�O�;�q�       �	!��ec�A�-*

loss4t:2���       �	B%��ec�A�-*

lossxB�<3]�       �	g���ec�A�-*

lossx{=�y�       �	����ec�A�-*

loss,^;<f�K�       �	\���ec�A�-*

loss\<`<��Z�       �	]��ec�A�-*

lossd =O|XL       �	�7��ec�A�-*

loss�Wv:0R��       �	+���ec�A�-*

loss��;BUC       �	3���ec�A�-*

loss�iP<2�^h       �	3���ec�A�-*

loss�G�:uK       �	ۆ��ec�A�-*

loss��'<��,J       �	�(��ec�A�-*

loss�9#�P�       �	���ec�A�-*

loss�T<B���       �	(b��ec�A�-*

loss:Ӌ:$#��       �	� ��ec�A�-*

loss��;吉       �	g���ec�A�-*

loss�@�;���       �	e4��ec�A�-*

loss_�<l��       �	J���ec�A�-*

loss? ;�i��       �	�r��ec�A�-*

loss�t<vMi�       �	��ec�A�-*

lossS��<2�;�       �	����ec�A�-*

loss@��;V�{       �	�`��ec�A�-*

loss��<`V�       �	$��ec�A�-*

loss�1�:4XMt       �	_���ec�A�-*

loss��;W�C       �	PS��ec�A�-*

lossdm.<�ԃ       �	����ec�A�-*

loss8�<)��       �	����ec�A�-*

loss ç;���       �	76��ec�A�-*

loss-+�:���       �	l���ec�A�-*

loss��<�k��       �	�k��ec�A�-*

loss���:���       �	B��ec�A�-*

loss~y=K%=       �	���ec�A�-*

loss�:�=���m       �	4��ec�A�-*

loss�B;��m�       �	t���ec�A�-*

loss���<qR        �	�i��ec�A�-*

lossl�,=��"Z       �	y��ec�A�-*

lossmi4:��Er       �	ܺ��ec�A�-*

loss�@�<�eY       �	�a��ec�A�-*

loss�^�<���       �	7���ec�A�-*

loss�R;ɼ*4       �	s���ec�A�-*

loss���=H&��       �	?5��ec�A�-*

loss�"�=P8�5       �	����ec�A�-*

loss���<}6��       �	@h��ec�A�-*

loss�`m<
�V       �	`��ec�A�-*

loss���;Ȫ+<       �	����ec�A�-*

loss�^�=��<       �	29��ec�A�-*

lossr7\=ņ>/       �	g���ec�A�-*

losst�=߯��       �	�y��ec�A�-*

loss�X>:���       �	�#��ec�A�-*

loss�;:�[%       �	�:��ec�A�-*

loss���;��       �	����ec�A�-*

loss��<��-�       �	x��ec�A�-*

lossC�*<b�       �	���ec�A�-*

loss(�w;U��R       �	E���ec�A�-*

lossC�;k       �	�g��ec�A�-*

lossN��<�b6       �	%t��ec�A�-*

loss4�<+$�       �	3��ec�A�-*

lossͮ%: �ˈ       �	���ec�A�-*

loss�_�;��       �	7P��ec�A�-*

loss���:�4��       �	à��ec�A�-*

loss�R=<��iZ       �	;��ec�A�-*

loss�m�;p�O       �	=���ec�A�-*

loss��X>"}U       �	�{��ec�A�-*

loss$�<ߛ.�       �	���ec�A�-*

lossQ�;���       �	ƿ��ec�A�-*

loss�4>��i�       �	R`��ec�A�-*

loss\F�;$�	�       �	����ec�A�-*

lossj��<����       �	����ec�A�-*

lossC��<5�z�       �	!=��ec�A�-*

loss���<�ݸ�       �	���ec�A�-*

loss^�<z���       �	���ec�A�-*

lossv�=큚       �	 ��ec�A�-*

loss��V<�ρ�       �	���ec�A�-*

lossIIG<��F       �	�[��ec�A�-*

loss�� ;
y�       �	����ec�A�-*

loss,��<d�֘       �	 ���ec�A�-*

loss���<,���       �	�*��ec�A�-*

loss��k=�7�^       �	���ec�A�-*

loss7�<���       �	�f��ec�A�-*

loss��;ɶ"       �	�	��ec�A�.*

loss�:=����       �	v���ec�A�.*

loss|[=<Pך�       �	�I��ec�A�.*

loss�B�;
���       �	T���ec�A�.*

loss�Z�;o�'�       �	���ec�A�.*

lossaF:9O5       �	'��ec�A�.*

loss))�<�7��       �	����ec�A�.*

lossZ�R;��!       �	`V��ec�A�.*

loss��=� �X       �	����ec�A�.*

lossW";M�K�       �	�� �ec�A�.*

loss��<>�P       �	�<�ec�A�.*

loss��=�W\       �	���ec�A�.*

loss��;��&�       �	�s�ec�A�.*

losse��<�wK       �	(�ec�A�.*

lossn%K;����       �	��ec�A�.*

loss��;��Ϻ       �	A��ec�A�.*

loss�uA=��O       �	(c�ec�A�.*

loss��/<T���       �	��ec�A�.*

loss�<^<v��`       �	��ec�A�.*

loss|'<�A6�       �	h^�ec�A�.*

loss N�<1G��       �	���ec�A�.*

loss56�<4�^3       �	-��ec�A�.*

loss�DF9����       �	�0	�ec�A�.*

loss8<֡�       �	_�	�ec�A�.*

loss�3"<AB�       �	�o
�ec�A�.*

lossԩ;��H?       �	�
�ec�A�.*

lossɑW= T�       �	ǝ�ec�A�.*

loss7\�;j?�v       �	*��ec�A�.*

loss���;*�(�       �	WA�ec�A�.*

loss��|=��%�       �	���ec�A�.*

lossD>�:�K�d       �	/o�ec�A�.*

loss��(;�:J�       �	��ec�A�.*

loss��;/ �       �	¥�ec�A�.*

loss���;�T�C       �	/O�ec�A�.*

loss!��:����       �	���ec�A�.*

loss+��:��O�       �	��ec�A�.*

loss;�3:�H�       �	�:�ec�A�.*

lossL�t<2$j       �	��ec�A�.*

loss}��;9�       �	�s�ec�A�.*

loss�(�<�v�       �	��ec�A�.*

lossR�0<��%�       �	6 �ec�A�.*

loss�I�;�� 
       �	���ec�A�.*

loss��m:w�r       �	aU�ec�A�.*

loss�u�:}7dy       �	p��ec�A�.*

loss�0�;/�)�       �	u��ec�A�.*

loss<�`<��B       �	�+�ec�A�.*

lossj�:_j�k       �	r��ec�A�.*

loss@�.:�eU       �	�V�ec�A�.*

loss��.9���       �	���ec�A�.*

loss� y:u        �	7��ec�A�.*

loss$�9;��C�       �	!�ec�A�.*

lossc4!:W�Y       �	F�ec�A�.*

loss4�z;�W��       �	�5�ec�A�.*

loss8o�9�O)       �	l��ec�A�.*

loss�@:7~��       �	�s�ec�A�.*

loss�X�:	��       �	�
�ec�A�.*

loss�r�8��[�       �	X��ec�A�.*

loss��2=t~^!       �	�A �ec�A�.*

loss�\c<5_��       �	�� �ec�A�.*

lossz��7C&�j       �	1x!�ec�A�.*

loss��:��]Z       �	"�ec�A�.*

loss�Kn<]�,�       �	��"�ec�A�.*

loss2"M9����       �	�]#�ec�A�.*

lossE�?=oC]       �	��#�ec�A�.*

loss��S<�ty�       �	�$�ec�A�.*

loss
��=-�,_       �	z9%�ec�A�.*

loss��<��!       �	x�%�ec�A�.*

loss%%<Y}0       �	�j&�ec�A�.*

loss�c�:���       �	�'�ec�A�.*

loss��<~��)       �	V�'�ec�A�.*

loss�ߪ;
��       �	�\(�ec�A�.*

loss���;��Χ       �	})�ec�A�.*

lossZT�;!x��       �	8�)�ec�A�.*

loss�w<;n��       �	U0*�ec�A�.*

lossj�W<אW"       �	��*�ec�A�.*

loss;��ʏ       �	�~+�ec�A�.*

loss�"]<(2y       �	R*,�ec�A�.*

loss)�<�kb�       �	Q-�ec�A�.*

lossZ��:�I=       �	��-�ec�A�.*

loss��:�+ q       �	\>.�ec�A�.*

lossM<�.{�       �	��.�ec�A�.*

loss�y�;��]#       �	=|/�ec�A�.*

lossqCT:�n_       �	�0�ec�A�.*

losse�\:��       �	�.1�ec�A�.*

loss�[A;��.       �	Y�1�ec�A�.*

loss��;�^V       �	�2�ec�A�.*

lossb�;*�Ԅ       �	�73�ec�A�.*

loss���:�.W       �	��3�ec�A�.*

loss�B<���       �	$�4�ec�A�.*

loss�'�<�'�       �	�5�ec�A�.*

loss��x<���        �	��5�ec�A�.*

loss �=���R       �	�_6�ec�A�.*

loss�P�9���       �	��6�ec�A�.*

loss�0�;�L�H       �	A�7�ec�A�.*

loss�[�9�E��       �	�K8�ec�A�.*

loss��;|��       �	��8�ec�A�.*

loss��;;�$T       �	g}9�ec�A�.*

loss�
�;�5A�       �	� :�ec�A�.*

lossN�0;�%��       �	��:�ec�A�.*

loss�<���       �	�`;�ec�A�.*

loss	�w=�JB�       �	<�ec�A�.*

loss䓀;��#q       �	��<�ec�A�.*

lossddo:�.�       �	WB=�ec�A�.*

loss�[�;R�       �	��=�ec�A�.*

lossa�<I8�       �	z>�ec�A�.*

loss �=0M'       �	s?�ec�A�.*

losssM;��       �	�?�ec�A�.*

loss�n�;љ��       �	K@�ec�A�.*

lossF}�;�-��       �	��@�ec�A�.*

lossn��;	U3�       �	��A�ec�A�.*

lossؗ�<�-	�       �	�!B�ec�A�.*

loss� }909��       �	�5C�ec�A�.*

loss�˄=��       �	��C�ec�A�.*

loss:�:�H�4       �	�`�ec�A�.*

loss��<�6�       �	�a�ec�A�.*

lossƬl;y��       �	J�a�ec�A�.*

loss�S�;u!�       �	6�b�ec�A�.*

loss�gg:����       �	�zc�ec�A�.*

loss���9����       �	�d�ec�A�.*

lossȧ�;��
�       �	ȶd�ec�A�.*

loss4�;hC[*       �	Ne�ec�A�.*

loss ��<��,       �	��e�ec�A�.*

loss|5=>)R1       �	4�f�ec�A�.*

lossb<d�       �	Pg�ec�A�.*

lossD6P<�f�       �	��g�ec�A�.*

loss�9�;�$/+       �	Kh�ec�A�.*

lossh*x<����       �	��h�ec�A�.*

loss�q�<��,       �	�i�ec�A�.*

loss���;�Rq�       �	�/j�ec�A�/*

loss� .9Xv�}       �	n�j�ec�A�/*

loss
4s;&>�       �	Swk�ec�A�/*

loss�e<\�@�       �	�l�ec�A�/*

loss��W=�]L<       �	-�l�ec�A�/*

loss�t�;bÖ       �	Hm�ec�A�/*

loss��=����       �		o�ec�A�/*

loss��:��r[       �	�o�ec�A�/*

loss,�=Ȃ��       �	��p�ec�A�/*

loss�� ;�	�.       �	9&q�ec�A�/*

loss��<_��       �	��q�ec�A�/*

loss:@�<P{F�       �	[r�ec�A�/*

lossi7/:��       �	��r�ec�A�/*

loss�&�;!i�       �	ގs�ec�A�/*

lossQ�|;7���       �	�9t�ec�A�/*

loss��:��H�       �	,�t�ec�A�/*

loss�B;b�       �	�nu�ec�A�/*

loss�ю;8��       �	1v�ec�A�/*

loss��n=��NB       �	��v�ec�A�/*

loss�0�;�*~�       �	M�w�ec�A�/*

loss��@<`?�:       �	\sx�ec�A�/*

loss7��;��v       �	ky�ec�A�/*

loss�WM=�׭�       �		�y�ec�A�/*

loss#= 2ܰ       �	�@z�ec�A�/*

lossq�S<<��%       �	�{�ec�A�/*

loss9J�:�Q(       �	�{�ec�A�/*

loss�BD<�l?       �	/N|�ec�A�/*

lossi�<�Pl�       �	5�|�ec�A�/*

loss�;v�X�       �	�}�ec�A�/*

loss��;.�m%       �	�Q~�ec�A�/*

loss��2=o���       �	�~�ec�A�/*

loss��\<=s�       �	���ec�A�/*

loss!�;�[�       �	#M��ec�A�/*

lossׇ�:��D�       �	>��ec�A�/*

loss��9����       �	
��ec�A�/*

loss�QE;	�Pv       �	*���ec�A�/*

loss�sR<��mD       �	�)��ec�A�/*

loss�9ŏt�       �	MN��ec�A�/*

loss�Z5=��̟       �	�%��ec�A�/*

loss���;$��        �	��ec�A�/*

loss�h)9#[
}       �	2���ec�A�/*

loss��+:�/�:       �	���ec�A�/*

loss}�?9׷�       �	Z*��ec�A�/*

loss`�9;�\��       �	�a��ec�A�/*

loss7�w=}�)       �	v���ec�A�/*

loss�O�=�>|       �	���ec�A�/*

loss��d;ޚ1�       �	=~��ec�A�/*

loss�A=�R��       �	�I��ec�A�/*

loss,�c;{       �	]7��ec�A�/*

lossx(5;� �V       �	ԏ�ec�A�/*

loss:�<�mu       �	�ܐ�ec�A�/*

loss�BO:4��       �	�j��ec�A�/*

loss��@<��L�       �	uU��ec�A�/*

loss�=�F��       �	�	��ec�A�/*

lossi�-:����       �	B���ec�A�/*

loss�^:ׅ"       �	�S��ec�A�/*

loss}�I<�I�z       �	���ec�A�/*

lossQ��9W'�F       �	G���ec�A�/*

loss���:E�7W       �	*��ec�A�/*

loss?�S<��m�       �	�×�ec�A�/*

lossv		;�0ˑ       �	�^��ec�A�/*

lossh9�:_�N>       �	����ec�A�/*

loss �<�J       �	ڍ��ec�A�/*

loss�)C<o�       �	o+��ec�A�/*

loss���:���H       �	ɚ�ec�A�/*

loss@��:5��       �	Qh��ec�A�/*

loss��R<���       �	���ec�A�/*

loss�O�:���       �	���ec�A�/*

lossy�;��;�       �	;��ec�A�/*

loss,��;P!��       �	N՝�ec�A�/*

loss֣-:6�P       �	Xu��ec�A�/*

loss;}�: *�       �	���ec�A�/*

loss-5;;/D�       �	����ec�A�/*

loss薦<�n�       �	�D��ec�A�/*

lossѭ>=!T�.       �	�ܠ�ec�A�/*

lossT�J;q��       �	Nz��ec�A�/*

lossR�;Y��       �	��ec�A�/*

loss�^�8��&       �	N���ec�A�/*

loss�&;c�~.       �	�m��ec�A�/*

loss���<���       �	є��ec�A�/*

loss�ME<�}       �	K;��ec�A�/*

loss/>�< ��       �	�֥�ec�A�/*

loss# +:�:��       �	Kt��ec�A�/*

loss�D<A`�?       �	�w��ec�A�/*

loss<{Y<K��       �	���ec�A�/*

lossv��9gْ�       �	9���ec�A�/*

lossI�:��^�       �	�T��ec�A�/*

loss֘=��       �	����ec�A�/*

lossn�z:�q�       �	���ec�A�/*

loss��;w��       �	7Q��ec�A�/*

loss;}Ew�       �	g��ec�A�/*

loss���:�!?a       �	���ec�A�/*

loss��9��       �	���ec�A�/*

loss�J=�p}/       �	����ec�A�/*

loss�5;!8c       �	h"��ec�A�/*

loss�)�<�2�       �	���ec�A�/*

loss ~q;߻c       �	'L��ec�A�/*

lossS6�;I%o�       �	)��ec�A�/*

loss�:�9V�:�       �	�w��ec�A�/*

loss�Z=.���       �	���ec�A�/*

loss�<<l�1       �	B̴�ec�A�/*

loss!��<��y       �	�v��ec�A�/*

loss�<�V `       �	���ec�A�/*

loss3 �;��3       �	��ec�A�/*

loss�/�9�1�!       �	$^��ec�A�/*

loss���;S���       �	���ec�A�/*

loss}�=|���       �	���ec�A�/*

lossL�<ha3X       �	(-��ec�A�/*

loss�KL<��       �	9й�ec�A�/*

loss#&<<���       �	�d��ec�A�/*

loss�_�;I7�       �	���ec�A�/*

loss��?;C3rW       �	d���ec�A�/*

lossq��9=��       �	�#��ec�A�/*

loss��<K�%       �	F���ec�A�/*

lossܾ	;�RL       �	7��ec�A�/*

loss�ɬ;��hf       �	lξ�ec�A�/*

losst��:ss�       �	sc��ec�A�/*

loss:0�<y]�       �	[��ec�A�/*

loss��s<}��       �	E���ec�A�/*

loss��<d5��       �	6?��ec�A�/*

loss�#<t��       �	
L��ec�A�/*

loss;<�BG�       �	B���ec�A�/*

loss���;�3��       �	L���ec�A�/*

loss�h�9���O       �	s*��ec�A�/*

loss7��<��vw       �	���ec�A�/*

loss�"�;��T�       �	����ec�A�/*

lossOW�<���       �	{K��ec�A�/*

loss� �;�4Z�       �	a��ec�A�/*

loss��:�TG�       �	����ec�A�0*

lossm�:����       �	����ec�A�0*

loss �<U�%�       �	gD��ec�A�0*

lossSE;\��%       �	���ec�A�0*

loss�<��\>       �	����ec�A�0*

loss�z�:�7ˀ       �	[?��ec�A�0*

loss��F;���g       �	����ec�A�0*

loss�2�;hљ       �	P���ec�A�0*

loss<2�;:Vg:       �	�F��ec�A�0*

lossi�t;X��X       �	����ec�A�0*

loss,f�;ٙ׏       �	9}��ec�A�0*

loss-'P=E_�k       �	���ec�A�0*

loss#l�;��$       �	����ec�A�0*

loss���;��#Z       �	�N��ec�A�0*

loss�?�=)��+       �	)���ec�A�0*

loss
�S;c��       �	����ec�A�0*

loss��:�m��       �	~��ec�A�0*

lossp;�]~�       �	����ec�A�0*

losszR$9k�C}       �	�V��ec�A�0*

loss��;�p�       �	1���ec�A�0*

loss�\�8�F�j       �	є��ec�A�0*

loss$�y;[���       �	�2��ec�A�0*

lossj;\9A�=�       �	S���ec�A�0*

loss�w9���P       �	�e��ec�A�0*

loss� :�k�-       �	���ec�A�0*

loss�qA;�H�B       �	����ec�A�0*

loss�e9<�TΒ       �	R,��ec�A�0*

loss`��<����       �	M���ec�A�0*

loss��:�!߂       �	�f��ec�A�0*

lossm<��E(       �	����ec�A�0*

loss��<*A�       �	��ec�A�0*

lossQ;)��       �	���ec�A�0*

lossF�^=�r       �	�H��ec�A�0*

loss���<p�f        �	���ec�A�0*

loss;/;��P       �	:���ec�A�0*

losshPE;��b�       �	�2��ec�A�0*

loss
.K9�~Û       �	����ec�A�0*

loss1�5:��c�       �	'l��ec�A�0*

loss��l<u���       �	���ec�A�0*

loss��:k�_       �	(���ec�A�0*

lossA��;�>�       �	�4��ec�A�0*

loss5B�9�        �	5���ec�A�0*

loss)Ձ9�(f       �	�e��ec�A�0*

loss;s<:��W�       �	���ec�A�0*

loss'�;`I�       �	z���ec�A�0*

loss3@�<�H�       �	z���ec�A�0*

loss�7H=����       �	|��ec�A�0*

loss]�$;�s4�       �	�<��ec�A�0*

lossd�%;�i�       �	����ec�A�0*

losso��9��       �	v���ec�A�0*

loss�g=�"�       �	S���ec�A�0*

loss�2�9b���       �	�:��ec�A�0*

lossZ�;�?�       �	����ec�A�0*

loss��<���q       �	Ym��ec�A�0*

loss��u<�E�       �	���ec�A�0*

loss�A�;�O��       �	p���ec�A�0*

lossM�p<����       �	0F��ec�A�0*

lossh<Q�       �	i���ec�A�0*

loss8�8;)�.1       �	����ec�A�0*

loss��; ���       �	Mj��ec�A�0*

loss��:�W��       �	����ec�A�0*

loss�[;�l�       �	�r��ec�A�0*

lossi1�;)"��       �	1��ec�A�0*

lossXZ=aP��       �	|���ec�A�0*

loss.u9�%`       �	�5��ec�A�0*

loss��)<O��       �	����ec�A�0*

loss-�B<���       �	�l��ec�A�0*

loss��9=��M        �	u��ec�A�0*

lossx0�:���N       �	���ec�A�0*

lossj�+<�$8�       �	G��ec�A�0*

loss<.�<���       �	����ec�A�0*

loss	�%=�_�       �	|���ec�A�0*

loss=���&       �	ʍ��ec�A�0*

loss���=��       �	�&��ec�A�0*

lossc �;8%�       �	���ec�A�0*

loss�=	�`       �	LT��ec�A�0*

lossV=����       �	����ec�A�0*

loss�8�<0͞!       �	w���ec�A�0*

loss#:�:�=�!       �	��ec�A�0*

lossG;���v       �	F���ec�A�0*

loss��=ֿ�       �	�R��ec�A�0*

loss}�	=�'<�       �	,���ec�A�0*

loss68�:�0RX       �	q���ec�A�0*

lossI�;���m       �	U2 �ec�A�0*

lossz�:bm��       �	�� �ec�A�0*

loss/�;0��Q       �	�g�ec�A�0*

loss�:��'L       �	!�ec�A�0*

loss�:��9�       �	ș�ec�A�0*

lossf�;��?       �	�z�ec�A�0*

loss�@x<���       �	���ec�A�0*

lossw��<\	�       �	�(�ec�A�0*

loss.�<<��R       �	^��ec�A�0*

loss��:o�x�       �	TU�ec�A�0*

loss@	(<�A�       �	���ec�A�0*

loss�A<5/U       �	<��ec�A�0*

loss֫�;Dc�       �	�I�ec�A�0*

loss#v:��       �	[��ec�A�0*

loss�h�;"bjt       �	{�	�ec�A�0*

loss���9���       �	
�ec�A�0*

loss!�8<؊��       �	Է
�ec�A�0*

loss�z{;� x�       �	�O�ec�A�0*

loss:�;�<��       �	~��ec�A�0*

loss�E�;X~��       �	_��ec�A�0*

loss��9<��k�       �	"8�ec�A�0*

losso�<� Q�       �	���ec�A�0*

loss��9<z��       �	ڏ�ec�A�0*

lossue<��       �	�5�ec�A�0*

loss��:�S       �	���ec�A�0*

lossf�)<E��       �	�u�ec�A�0*

loss���:y�t       �	 �ec�A�0*

lossȯ;�9�       �	���ec�A�0*

loss-��8~ɔ       �	"T�ec�A�0*

loss68�:��&�       �	��ec�A�0*

losse�;_��C       �	��ec�A�0*

loss�}�7��E       �	)�ec�A�0*

lossjB�:Ԭ       �	<��ec�A�0*

lossu�<ᡀ�       �	�W�ec�A�0*

lossIZ�8&���       �	��ec�A�0*

loss��|<�.       �	L��ec�A�0*

loss�<�ƺ�       �	1$�ec�A�0*

loss�g:����       �	��ec�A�0*

loss#c<�ǋ�       �	R`�ec�A�0*

losscs9W�n%       �	���ec�A�0*

loss���;��W       �	��ec�A�0*

loss�7�<;�b�       �	Z-�ec�A�0*

loss�E�<��V�       �	:��ec�A�0*

loss��H<���       �	l�ec�A�0*

lossG�:D)�"       �	��ec�A�0*

lossl
�:_:V�       �	J��ec�A�1*

lossT:��)0       �	i9�ec�A�1*

loss���=0�-�       �	<��ec�A�1*

loss26V;�4ڦ       �	G��ec�A�1*

lossc��;���       �	�$�ec�A�1*

loss���:bw��       �	���ec�A�1*

loss��;E���       �	`� �ec�A�1*

loss�;��^       �	�N!�ec�A�1*

loss{h�<B<       �	��!�ec�A�1*

loss6�9��"       �	��"�ec�A�1*

loss���<ݪUF       �	F&#�ec�A�1*

loss�-�;ą7�       �	w�#�ec�A�1*

lossw@W:t.         �	�T$�ec�A�1*

lossw�;,4V�       �	��$�ec�A�1*

loss��w9���       �	#�%�ec�A�1*

loss��<\�R       �	-�&�ec�A�1*

loss頹9�ʯ�       �	8�'�ec�A�1*

losst�:W���       �	s+(�ec�A�1*

lossX{�;kK��       �	}�(�ec�A�1*

loss�W�:���R       �	G�)�ec�A�1*

loss�]�;9O|Y       �	P5*�ec�A�1*

lossT��9��<       �	�*�ec�A�1*

lossD�9��1       �	��+�ec�A�1*

lossLX�9Iv(       �	�(,�ec�A�1*

loss��<;="�       �	]�,�ec�A�1*

loss���;S�(       �	�e-�ec�A�1*

loss6��:�E�       �	.�ec�A�1*

loss�y=Th�       �	Н.�ec�A�1*

loss|�;��\       �	�6/�ec�A�1*

loss/�<��Hp       �	��/�ec�A�1*

lossIe�:Ů�g       �	�h0�ec�A�1*

loss���:)U��       �	��0�ec�A�1*

loss|M<��-       �	��1�ec�A�1*

loss�W�;Uc�       �	�/2�ec�A�1*

loss��=�:]       �	r�2�ec�A�1*

loss��A8���       �	�[3�ec�A�1*

loss�@:��k       �	��3�ec�A�1*

loss1��;s%�       �	�4�ec�A�1*

loss�X�=I�7       �	`5�ec�A�1*

loss�z;�E�2       �	��5�ec�A�1*

loss;c><�H��       �	K6�ec�A�1*

loss�H;)�,�       �	;�6�ec�A�1*

loss&F<<HƵ       �	�z7�ec�A�1*

lossD;�~l       �	�8�ec�A�1*

loss3%Z<���       �	P�8�ec�A�1*

loss�Z�9�       �	�?9�ec�A�1*

loss �n;!�g�       �	��9�ec�A�1*

loss��@=&��       �	�k:�ec�A�1*

lossZң;p%�       �	�;�ec�A�1*

loss���:�s��       �	N�;�ec�A�1*

lossX�<��3�       �	�C<�ec�A�1*

loss�5�:��9�       �	e�<�ec�A�1*

loss&K�;�nj$       �	�|=�ec�A�1*

loss^��;��(       �	�>�ec�A�1*

loss�<Ⱦ"       �	4�>�ec�A�1*

loss�%c;�T4�       �	�t?�ec�A�1*

loss`��<��!�       �	�@�ec�A�1*

loss�";�I��       �	¢@�ec�A�1*

loss\8<*�KB       �	�:A�ec�A�1*

lossԨ�9A�       �	��A�ec�A�1*

loss�}�;[���       �	�fB�ec�A�1*

loss��q;c�2       �	��C�ec�A�1*

lossf�_<�(�       �	�"E�ec�A�1*

lossh?�=եK       �	|�E�ec�A�1*

loss�ۃ9��       �	�mF�ec�A�1*

loss�,<|�tA       �	5(G�ec�A�1*

loss��8�b       �	�ZH�ec�A�1*

loss�?'=�c-�       �	�I�ec�A�1*

loss t]9G��^       �	�J�ec�A�1*

loss%U�:�d�       �	�)K�ec�A�1*

lossa��9�[�       �	��K�ec�A�1*

loss�'�<�ЎM       �	w�L�ec�A�1*

loss1�:��y-       �	�^M�ec�A�1*

loss��=���       �	�M�ec�A�1*

loss{��:��@k       �	7�N�ec�A�1*

lossL��9܉�q       �	B�O�ec�A�1*

losss�%<�       �	�/P�ec�A�1*

loss���;�	       �	+Q�ec�A�1*

lossCv�;����       �	��Q�ec�A�1*

loss��9���       �	%ZR�ec�A�1*

loss�fp;-��       �	��R�ec�A�1*

lossl��<q�7�       �	��S�ec�A�1*

loss�o�:���       �	�.T�ec�A�1*

loss�S<�{Z�       �	�U�ec�A�1*

loss�:�;�:��       �	�/W�ec�A�1*

loss*�)<�0��       �	��W�ec�A�1*

loss7X�;ڑQ�       �	i�X�ec�A�1*

loss8�;[#'       �	1(Y�ec�A�1*

loss��9%�       �	��Y�ec�A�1*

loss̖:R2a�       �	�_Z�ec�A�1*

losst�&</���       �	[�ec�A�1*

loss�_)=h�O�       �	Z�[�ec�A�1*

lossF�Y<�g_�       �	�]\�ec�A�1*

loss�ů=��       �	�	]�ec�A�1*

loss唁<I�!�       �	ߣ]�ec�A�1*

loss,,�<Y�[!       �	�?^�ec�A�1*

loss擗:��v       �	�^�ec�A�1*

lossh�;�7       �	�v_�ec�A�1*

losst�;L���       �	`�ec�A�1*

lossr�B<�w�       �	��`�ec�A�1*

loss':<�2sr       �	�ra�ec�A�1*

loss���9I��       �	sb�ec�A�1*

lossO-�9�B��       �	!�b�ec�A�1*

loss[�<�ʮ�       �	�Ic�ec�A�1*

loss�F�9���       �	(�c�ec�A�1*

lossfP�<���       �	i�d�ec�A�1*

loss<ꐒ�       �	�'e�ec�A�1*

lossl�};Í|�       �	i�e�ec�A�1*

lossn�*=<�a�       �	m�f�ec�A�1*

loss��=kA�@       �	�g�ec�A�1*

loss(1Z=o�o       �	�wh�ec�A�1*

lossf�=:^�       �	�i�ec�A�1*

loss�,�9  �       �	I�i�ec�A�1*

loss���80�v�       �	�rj�ec�A�1*

loss��<Ǚؗ       �	�(k�ec�A�1*

loss*ܥ:&<�3       �	<�k�ec�A�1*

loss��<�=��       �	�Wl�ec�A�1*

loss���:�ۦg       �	�l�ec�A�1*

lossJ�<��;�       �	;�m�ec�A�1*

loss��;�m�       �	�(n�ec�A�1*

loss�07;A�d       �	��n�ec�A�1*

loss_h=)e       �	p]o�ec�A�1*

loss�_<l��       �	Y�o�ec�A�1*

loss�p&=�̊�       �	ٙp�ec�A�1*

loss��;��       �	�4q�ec�A�1*

loss��9�Ou�       �	�q�ec�A�1*

loss���;�Փ;       �	�gr�ec�A�1*

lossכ�:oDl+       �	H�r�ec�A�1*

loss��Y<9���       �	��s�ec�A�2*

loss� ;�6��       �	0Ht�ec�A�2*

loss��:�;�(       �	��t�ec�A�2*

loss��/<��       �	0v�ec�A�2*

loss�}�9^]Э       �	��v�ec�A�2*

loss]
3=o+��       �	�Aw�ec�A�2*

lossx �;+�W�       �	��w�ec�A�2*

loss��Q<�q�;       �	�ux�ec�A�2*

loss��:�ʑa       �	(y�ec�A�2*

loss�;TA��       �	ߦy�ec�A�2*

loss��Z=�jy       �	xGz�ec�A�2*

loss�d=�{6A       �	��z�ec�A�2*

loss���<�up�       �	&�{�ec�A�2*

losss��9�� �       �	(C|�ec�A�2*

loss���<��u       �	.�|�ec�A�2*

loss��p<�k�       �	�z}�ec�A�2*

loss�<�O{�       �	�~�ec�A�2*

loss?�=9Z�X�       �	��~�ec�A�2*

lossJ�9�*x       �	�H�ec�A�2*

lossld�<�y       �	���ec�A�2*

loss�U�;����       �	R��ec�A�2*

lossi1�<s�=       �	*��ec�A�2*

loss@w�<���)       �	���ec�A�2*

lossz��:{��       �	�U��ec�A�2*

loss��j=܎4�       �	���ec�A�2*

loss2�B:V�9        �	���ec�A�2*

losss˂;%�vs       �	~���ec�A�2*

loss8�a;Ir       �	�-��ec�A�2*

loss�H�:ٯ�Y       �	�ȅ�ec�A�2*

loss��&;ѝ�       �	͆�ec�A�2*

lossƯ4:� �       �	�w��ec�A�2*

loss�^;���u       �	
��ec�A�2*

lossĲ`:]y�M       �	KV��ec�A�2*

loss�S:X"|�       �	����ec�A�2*

loss=�<HԷH       �	D���ec�A�2*

loss�5�;j���       �	9F��ec�A�2*

loss��~<��;�       �	m���ec�A�2*

loss�{A<?O��       �	�K��ec�A�2*

loss���;O��7       �	B��ec�A�2*

loss�g:P�P�       �	���ec�A�2*

loss���;EL��       �	���ec�A�2*

loss��y;-SV       �	N���ec�A�2*

loss�g:߀?�       �	�Q��ec�A�2*

lossM#�<t4{       �	)��ec�A�2*

loss��:��D�       �	ԁ��ec�A�2*

loss��.;�`��       �	u;��ec�A�2*

lossf&�=&�\       �	�ؓ�ec�A�2*

lossq~�<C��h       �	�o��ec�A�2*

loss��;�y�H       �	|E��ec�A�2*

loss_�<�ץ       �	�ݕ�ec�A�2*

lossREF;3E�1       �	ur��ec�A�2*

loss3��9���;       �	�K��ec�A�2*

loss�o^;I���       �	3��ec�A�2*

loss.5<��x�       �	}x��ec�A�2*

loss'�<��       �	�!��ec�A�2*

lossR��;��M7       �	f���ec�A�2*

lossD�	:��D       �	HQ��ec�A�2*

lossѽ�<�Y�       �	���ec�A�2*

loss�W�;ՠ~c       �	o���ec�A�2*

loss�<(�       �	�D��ec�A�2*

loss��=���X       �		��ec�A�2*

loss(�H;麡�       �	����ec�A�2*

loss@�E;~�(�       �	½��ec�A�2*

loss��O;c�U�       �	|_��ec�A�2*

loss�^Q=%�L�       �	��ec�A�2*

loss�t�<�=E�       �	����ec�A�2*

loss�_�;h1j�       �	V��ec�A�2*

loss8$s;yD��       �	��ec�A�2*

loss̷;q���       �	����ec�A�2*

loss�@<���       �	AE��ec�A�2*

lossd�E:�5^       �	U/��ec�A�2*

lossd41=����       �	Eؤ�ec�A�2*

loss&ֻ9�`�       �	Y���ec�A�2*

loss��<�1=�       �	&R��ec�A�2*

loss�;<v�       �	W��ec�A�2*

loss y�:��1�       �	 ���ec�A�2*

lossZ��;��       �		8��ec�A�2*

loss��x8�q]       �	�=��ec�A�2*

lossU�=�xF~       �	�ة�ec�A�2*

lossO��;����       �	����ec�A�2*

losszUd;���y       �	[���ec�A�2*

lossV]<�G;�       �	G��ec�A�2*

loss��<t�|]       �	7��ec�A�2*

loss��b;ri�       �	�}��ec�A�2*

loss�P;c�/(       �	X��ec�A�2*

loss߂<�N��       �	ȵ��ec�A�2*

loss�	;عp       �	���ec�A�2*

loss��O;m*U�       �	#��ec�A�2*

loss�<W�o�       �	���ec�A�2*

loss��Q<���~       �	)���ec�A�2*

loss�۷<���       �	�ײ�ec�A�2*

lossr�:���L       �	F~��ec�A�2*

lossE��:�K�       �	i��ec�A�2*

loss*'>:�z�F       �	YĴ�ec�A�2*

loss�Ƹ9���       �	
���ec�A�2*

loss��<�Ή       �	5���ec�A�2*

loss�t�<��       �	�?��ec�A�2*

loss�+�:�S3�       �	�ݷ�ec�A�2*

loss���;ܛ03       �	����ec�A�2*

loss�Q�;�>'       �	N(��ec�A�2*

losscwN<���I       �	�Ĺ�ec�A�2*

losst�^:�R�       �	�`��ec�A�2*

loss���:�X�       �	'���ec�A�2*

loss1�<�V�       �	%���ec�A�2*

loss��<�6       �	c+��ec�A�2*

loss�FV:d;�       �	yʼ�ec�A�2*

loss��;�*�       �	b��ec�A�2*

loss��<7��	       �	{���ec�A�2*

loss���9ͱ&E       �	��ec�A�2*

lossS�><��%       �	�*��ec�A�2*

lossx�;Hi�o       �	�̿�ec�A�2*

loss�ȩ<I�D�       �	�n��ec�A�2*

loss�(�;b|V       �	l��ec�A�2*

loss.=6<.$�/       �	����ec�A�2*

loss�.-=P�       �	�4��ec�A�2*

loss݊u;��T       �	C���ec�A�2*

loss��Y<��f       �	J^��ec�A�2*

lossn'<�r�       �	����ec�A�2*

loss�δ9� ,       �	r���ec�A�2*

loss�;@��       �	�!��ec�A�2*

loss��Q<>>��       �	����ec�A�2*

lossT��9�-��       �	�V��ec�A�2*

lossV�=Dܔ       �	�c��ec�A�2*

loss)}�8tK       �	���ec�A�2*

loss�:��       �	9���ec�A�2*

loss�q�<�_       �	�.��ec�A�2*

loss��<rژY       �	�K��ec�A�2*

loss?�m9Z�'       �	3O��ec�A�2*

loss�TU9�u�O       �	c���ec�A�3*

loss��<ɂ��       �	����ec�A�3*

loss��9���d       �	8K��ec�A�3*

loss�x�9�Q*�       �	F���ec�A�3*

loss�W�<a��       �	a���ec�A�3*

loss�O�9����       �	v8��ec�A�3*

losso'9&P��       �	k���ec�A�3*

loss�s{9K�3�       �	ys��ec�A�3*

lossԚ9` l}       �	���ec�A�3*

loss���:AS�       �	ӽ��ec�A�3*

loss�8<����       �	__��ec�A�3*

lossD�q7'35'       �	+���ec�A�3*

loss���:݆V       �	���ec�A�3*

loss�L0;��?�       �	X:��ec�A�3*

loss� �7uIC�       �	����ec�A�3*

loss�[c8��p       �	�p��ec�A�3*

loss�/�:�+ �       �	y��ec�A�3*

loss8� :Np��       �	����ec�A�3*

loss��8�Ұ�       �	�9��ec�A�3*

lossc?9���       �	����ec�A�3*

lossQpy:h���       �	�s��ec�A�3*

lossR�o<�L��       �	k��ec�A�3*

loss��Q9y"Y
       �	_���ec�A�3*

loss��T=)��=       �	�Y��ec�A�3*

loss�\=��(�       �	����ec�A�3*

lossmrV<~�c       �	���ec�A�3*

loss���<JP�h       �	%��ec�A�3*

lossqǢ;��eQ       �	����ec�A�3*

loss�J�;���a       �	�l��ec�A�3*

loss�K�;0�2       �	T��ec�A�3*

loss<zl;T���       �	����ec�A�3*

loss`�(;}G��       �	<0��ec�A�3*

loss�^:15��       �	.���ec�A�3*

loss:�Z9�XC�       �	�b��ec�A�3*

lossT�:�M9       �	����ec�A�3*

loss��x:2wd       �	����ec�A�3*

lossR�<�#�S       �	=(��ec�A�3*

loss��X;)�ek       �	����ec�A�3*

loss�w�:1m       �	we��ec�A�3*

lossc��:�Xvy       �	���ec�A�3*

lossQ,�;3�Z�       �	]���ec�A�3*

lossƜ:b�1�       �	�D��ec�A�3*

loss�9�X��       �	����ec�A�3*

lossè�:�}Y       �	_y��ec�A�3*

loss,aj=�s�A       �	9��ec�A�3*

loss�!�;-y>�       �	���ec�A�3*

loss MI:�@*        �	�E��ec�A�3*

losso9�6       �	w���ec�A�3*

loss,=���       �	P��ec�A�3*

loss?��9򨕀       �	[���ec�A�3*

lossJ��9	�k�       �	�D��ec�A�3*

lossܘ�<x�-�       �	����ec�A�3*

lossC�< 4��       �	xz��ec�A�3*

loss�ܬ:���Y       �	���ec�A�3*

loss��;��       �	����ec�A�3*

loss4�;9\��U       �	�?��ec�A�3*

loss��<euǶ       �	����ec�A�3*

loss�oL;ű��       �	���ec�A�3*

loss�K<��l�       �	~��ec�A�3*

loss�t<�ދ�       �	f���ec�A�3*

loss�c�;er�       �	5{��ec�A�3*

loss�3;���
       �	���ec�A�3*

loss�+;�p2�       �	����ec�A�3*

loss?�9:P��       �	�K��ec�A�3*

loss�C�<s�s       �	����ec�A�3*

loss�q�;�>��       �	�s��ec�A�3*

loss;+;\���       �	
��ec�A�3*

loss7��< ��       �	����ec�A�3*

loss�ڻ:���2       �	>��ec�A�3*

loss%��<8��       �	c���ec�A�3*

loss�g;ؒ��       �	�k��ec�A�3*

loss��;/ڵ       �	J��ec�A�3*

losswς;~��       �	*���ec�A�3*

loss14�;�?=I       �	ǝ�ec�A�3*

loss��;*�       �	�3�ec�A�3*

lossl&�=��D�       �	���ec�A�3*

loss@��;�'Oe       �	-^�ec�A�3*

losszM�<?��       �	���ec�A�3*

losstK<��F_       �	@��ec�A�3*

lossѰU;@}�       �	�ec�A�3*

loss���<��V       �	���ec�A�3*

lossh �<6fC       �	oE�ec�A�3*

loss�9¼�       �	���ec�A�3*

loss�;�C>L       �	l�ec�A�3*

loss�]�:��P�       �	T �ec�A�3*

loss�_�9�`x�       �	���ec�A�3*

loss`�%<�s�       �	"3�ec�A�3*

loss1��:w�^       �	��ec�A�3*

loss��c<ɚi�       �	bg�ec�A�3*

loss�J.9�6�       �	`�ec�A�3*

loss�:�� �       �	 ��ec�A�3*

loss��;��x�       �	@2�ec�A�3*

loss�p�<PM��       �	z��ec�A�3*

loss۽`<wp#�       �	�^�ec�A�3*

loss�*<L��{       �	-	�ec�A�3*

loss�k�:5��p       �	+��ec�A�3*

loss��0=}�ky       �	�6�ec�A�3*

loss��:v�~�       �	C��ec�A�3*

lossi�:;��       �	�r �ec�A�3*

losso0= g��       �	h!�ec�A�3*

loss��96�h       �	��!�ec�A�3*

loss6��;���       �	?"�ec�A�3*

loss��:"�}W       �	J�"�ec�A�3*

lossÁ�:���       �	I�#�ec�A�3*

lossd�":5ފj       �	($�ec�A�3*

loss5v
;�48	       �	
�$�ec�A�3*

loss��Q<��*�       �	�Q%�ec�A�3*

loss��:_�?       �	��%�ec�A�3*

loss�!<`��q       �	dy&�ec�A�3*

loss��s:ю�S       �	�'�ec�A�3*

loss�;<X�u�       �	�'�ec�A�3*

loss�4;�>�l       �	�@(�ec�A�3*

loss�*<��fx       �	��(�ec�A�3*

loss�g:\/�!       �	�)�ec�A�3*

loss;��:m�1�       �	�*�ec�A�3*

loss��<�r�       �	s�*�ec�A�3*

loss�4�;0GO       �	^+�ec�A�3*

lossn=4<��g7       �	��+�ec�A�3*

loss��E<��/c       �	j�,�ec�A�3*

loss�
�8^�f�       �	1A-�ec�A�3*

lossT�'<�3�       �	�.�ec�A�3*

loss�Q"9���5       �	.�.�ec�A�3*

loss�5
:V&       �	��/�ec�A�3*

loss��n;31
�       �	�|0�ec�A�3*

lossw�<)��       �	=)1�ec�A�3*

loss��;��       �	��1�ec�A�3*

loss���<紂       �	Vc2�ec�A�3*

loss.�;��|F       �	F3�ec�A�3*

loss�̔:9��       �	�3�ec�A�4*

loss��c9�%]�       �	y<4�ec�A�4*

lossA��8����       �	��4�ec�A�4*

loss���9�?c       �	Lo5�ec�A�4*

loss�<�;|       �	 6�ec�A�4*

losscd=�C�       �	g�6�ec�A�4*

loss�*�8t��       �	�,7�ec�A�4*

loss���8c`G�       �	{�7�ec�A�4*

loss���:�Ć       �	!V8�ec�A�4*

loss�L�<��v       �	��8�ec�A�4*

loss�M:�cB�       �	V}9�ec�A�4*

loss��)=7��       �	b:�ec�A�4*

loss��:�Ȋ1       �	��:�ec�A�4*

loss��;���       �	E;�ec�A�4*

loss��=�v       �	f�;�ec�A�4*

loss�"�9F�       �	�s<�ec�A�4*

loss�� =��w       �	|=�ec�A�4*

lossH�M:�?�       �	~�=�ec�A�4*

loss[�:=��       �	@>�ec�A�4*

loss@Y�9����       �	�>�ec�A�4*

loss܋9Z�       �	�j?�ec�A�4*

loss���;�٠E       �	�@�ec�A�4*

loss��`;��Zy       �	��@�ec�A�4*

loss
�O:��       �	O;A�ec�A�4*

loss �9J�3       �	+�A�ec�A�4*

loss^�=��       �	,}B�ec�A�4*

loss��!=���       �	�0C�ec�A�4*

lossx&�9�f       �	��C�ec�A�4*

loss}H&9�lo       �	KwD�ec�A�4*

loss�M�9:l       �	]E�ec�A�4*

loss4r<3e�R       �	z�E�ec�A�4*

loss�i�9��       �	�vF�ec�A�4*

loss�^�:���1       �	�*G�ec�A�4*

loss�<�|{       �	��G�ec�A�4*

loss�Y9����       �	�cH�ec�A�4*

loss�;�?�x       �	��H�ec�A�4*

loss �<����       �	�I�ec�A�4*

lossc\!:�MS       �	A�J�ec�A�4*

loss��<���;       �	�SK�ec�A�4*

lossD3�;`$�       �	�-L�ec�A�4*

loss�T9����       �	jM�ec�A�4*

loss�c=gP��       �	ݱM�ec�A�4*

loss_p�;"�ud       �	ИN�ec�A�4*

loss�V�9����       �	3O�ec�A�4*

lossn(,:��j       �	#�O�ec�A�4*

loss#�:M��-       �	N�P�ec�A�4*

loss7�4�]       �	�RQ�ec�A�4*

lossw&9�IX       �	C<R�ec�A�4*

lossk_<�,��       �	^S�ec�A�4*

lossl <?��
       �	�T�ec�A�4*

loss!��8�h�       �	K�T�ec�A�4*

loss���<�ڦ       �	�V�ec�A�4*

lossdÁ9�{�       �	y�V�ec�A�4*

loss|G=�=�       �	��W�ec�A�4*

loss���;<Ё       �	&�X�ec�A�4*

loss\��;z���       �	�Y�ec�A�4*

lossj3z<G��H       �	�nZ�ec�A�4*

loss�Sz;w�R�       �	/[�ec�A�4*

loss��8:� y�       �	�O\�ec�A�4*

loss�}�;�j��       �	�6]�ec�A�4*

loss���7�ph�       �	��]�ec�A�4*

loss���7�l#T       �	�_�ec�A�4*

lossE�;/ad�       �	U`�ec�A�4*

lossj��;u��       �	sa�ec�A�4*

loss��;�0]5       �	�Qb�ec�A�4*

loss��p<�7��       �	*;c�ec�A�4*

loss��6<Y���       �	�d�ec�A�4*

loss�g�;�?nl       �	�d�ec�A�4*

loss�M�;�3i       �	ǹe�ec�A�4*

loss��t:����       �	Adf�ec�A�4*

loss̾$<>�L�       �	�g�ec�A�4*

loss�i:��wl       �	�g�ec�A�4*

loss}�I;�[X       �	�lh�ec�A�4*

lossc\n;>}�       �	�i�ec�A�4*

losse�;���"       �	&�i�ec�A�4*

loss$�g<ո��       �	AIj�ec�A�4*

loss-re=�V�       �	,�j�ec�A�4*

lossn�F=�+��       �	��k�ec�A�4*

loss��<X�r3       �	�?l�ec�A�4*

lossh�9=4u$       �	��l�ec�A�4*

lossIԭ<O�@       �	�rm�ec�A�4*

loss�y�<��dD       �	n�ec�A�4*

loss
��;0(�       �	B�n�ec�A�4*

loss#)=�A�O       �	~To�ec�A�4*

lossA��:�	�       �	��o�ec�A�4*

loss��9�D�       �	іp�ec�A�4*

loss�n�<���       �	6q�ec�A�4*

loss�j<�]D       �	��q�ec�A�4*

loss�8�;*,��       �	?qr�ec�A�4*

loss,��<���       �	s�ec�A�4*

loss�;�n�n       �	��s�ec�A�4*

loss(��;�.��       �	vRt�ec�A�4*

loss�<��       �	��u�ec�A�4*

loss�";��5�       �	�dv�ec�A�4*

lossl�:�O�       �	�w�ec�A�4*

lossL�=�ٽ`       �	��w�ec�A�4*

loss�k�<LGm       �	�Kx�ec�A�4*

loss� <�ҏ       �	#�x�ec�A�4*

lossP�>�C-H       �	Ԝy�ec�A�4*

loss$މ<�\%d       �	REz�ec�A�4*

lossI�U9��       �	��z�ec�A�4*

loss��H9N13�       �	��{�ec�A�4*

loss�L�=daa       �	�+|�ec�A�4*

loss�.�=ܐ�w       �	��|�ec�A�4*

losso��=��~�       �	�o}�ec�A�4*

lossNV�<,�4]       �	�~�ec�A�4*

lossj��;��@       �	ޭ~�ec�A�4*

loss���:��       �	bL�ec�A�4*

loss�xs9�(h       �	���ec�A�4*

loss�=T�vH       �	f���ec�A�4*

loss��}:�u��       �	� ��ec�A�4*

loss�!E:��SX       �	���ec�A�4*

lossL=�:K$z       �	zP��ec�A�4*

lossJ��<�b�	       �	���ec�A�4*

loss�f�;zN�       �	����ec�A�4*

loss�P:���       �	ZE��ec�A�4*

loss��G:Bb�*       �	���ec�A�4*

loss��U=W�z       �	���ec�A�4*

loss�S+<|eC�       �	�8��ec�A�4*

loss�I<u:�       �	����ec�A�4*

lossi's=T��       �	m��ec�A�4*

lossD&<{��       �	����ec�A�4*

loss7N�<��W       �	�F��ec�A�4*

loss4�<@"��       �	�߉�ec�A�4*

loss�i�;���x       �	���ec�A�4*

loss�Y�9�_K<       �	c^��ec�A�4*

losss �<��H�       �	���ec�A�4*

loss<f�;����       �	C���ec�A�4*

loss���;��t%       �	y$��ec�A�5*

loss��
;u��1       �	����ec�A�5*

loss�W<>��       �	�O��ec�A�5*

loss��;Ҙв       �	y��ec�A�5*

loss]^�;F�R       �	����ec�A�5*

lossqE,<���       �	�N��ec�A�5*

loss<h(<o\f�       �	���ec�A�5*

loss�nB<r$#�       �	r���ec�A�5*

loss���=)	       �	*��ec�A�5*

loss�I}<*'G       �	�ǒ�ec�A�5*

loss|��<p��4       �	P���ec�A�5*

loss�pU:����       �	m<��ec�A�5*

loss��l:UiG       �	JΔ�ec�A�5*

loss4�I<ѮB�       �	�`��ec�A�5*

loss(>�<c=�       �	��ec�A�5*

lossz�"=��Kt       �	���ec�A�5*

loss���:!轹       �	$���ec�A�5*

loss���:j�       �	}!��ec�A�5*

loss�b6<0ך�       �	����ec�A�5*

loss;Wos�       �	^i��ec�A�5*

loss� =�~�Z       �	����ec�A�5*

loss�|J=�,        �	����ec�A�5*

lossV�<�)��       �	�V��ec�A�5*

loss8��:�
R       �	{���ec�A�5*

loss���<J<8r       �	獜�ec�A�5*

loss\?�98>��       �	O#��ec�A�5*

loss1��=. �       �	���ec�A�5*

loss�I�:�+pY       �	�\��ec�A�5*

lossf�7:~)H       �	���ec�A�5*

loss��';n���       �	~���ec�A�5*

loss{�59e�~�       �	�w��ec�A�5*

loss��<�H��       �	���ec�A�5*

loss_��=7�H�       �	&��ec�A�5*

loss���=���       �	�z��ec�A�5*

lossH�+=��       �	���ec�A�5*

loss@ы<r�       �	D���ec�A�5*

loss	~�<�sF-       �	:���ec�A�5*

lossր�=3�ˇ       �	ƥ�ec�A�5*

loss�;���       �	�^��ec�A�5*

loss��;��`E       �	����ec�A�5*

loss�)�<�R)       �	>���ec�A�5*

loss�3:���       �	�-��ec�A�5*

loss��m9� �e       �	�˨�ec�A�5*

loss^�:�j==       �	�g��ec�A�5*

loss��;�]J       �	=)��ec�A�5*

lossc�$<	�7�       �	ǽ��ec�A�5*

loss�|�<�Ds       �	�Z��ec�A�5*

loss]K�:E(Qv       �	r5��ec�A�5*

loss�z:�>       �	__��ec�A�5*

lossy�=�ȕ�       �	���ec�A�5*

loss.�<���       �	����ec�A�5*

loss!�=��_r       �	�a��ec�A�5*

loss>�;�e�       �	�j��ec�A�5*

loss�B�:��6�       �	G��ec�A�5*

lossT�/=ڟ"�       �	�ݱ�ec�A�5*

loss�kh;H.Q�       �	����ec�A�5*

loss���8�Z��       �	�y��ec�A�5*

loss8�\:��S�       �	���ec�A�5*

loss=h�;t�       �	�Ŵ�ec�A�5*

loss@�
:��2       �	�`��ec�A�5*

loss��=/Y�       �	f/��ec�A�5*

loss��I;�V�       �	*��ec�A�5*

loss�-j;캁J       �	J}��ec�A�5*

losso��=6�'       �	v��ec�A�5*

loss2��9#�       �	����ec�A�5*

loss�=�D:m       �	mV��ec�A�5*

loss��y;0��       �	���ec�A�5*

loss�ʿ:FfF       �	����ec�A�5*

loss��
:���}       �	3��ec�A�5*

lossd��<:���       �	^ڻ�ec�A�5*

loss=��9p�K       �	Q���ec�A�5*

loss��?9ʋ��       �	�0��ec�A�5*

lossS8�;~}�       �	���ec�A�5*

loss�[�;���       �	���ec�A�5*

loss�=!7��       �	_'��ec�A�5*

loss��A<ALφ       �	˿�ec�A�5*

loss *�:�H{Y       �	zq��ec�A�5*

loss�@�<���Y       �	���ec�A�5*

lossDù;��C       �	8���ec�A�5*

loss�أ;�v�X       �	�`��ec�A�5*

loss�FL<2K       �	r���ec�A�5*

loss.=z��       �	(���ec�A�5*

loss`=�9�L%�       �	�:��ec�A�5*

loss�<e�3#       �	=���ec�A�5*

loss��<�3��       �	�o��ec�A�5*

loss��;q"r�       �	�
��ec�A�5*

loss��;[��       �	z���ec�A�5*

lossѬ�<>��m       �	�K��ec�A�5*

loss�7�<E׾�       �	_	��ec�A�5*

losso�=���       �	t��ec�A�5*

loss��<'�       �	�C��ec�A�5*

loss�I�<f�Y       �	����ec�A�5*

loss��^=���       �	���ec�A�5*

loss_v8����       �	y=��ec�A�5*

loss,l5;���)       �	����ec�A�5*

lossf��<ϻ�V       �	mr��ec�A�5*

loss�B<1W�       �	���ec�A�5*

loss��;<Z�       �	E���ec�A�5*

loss��%<K�       �	v���ec�A�5*

lossE��;c���       �	���ec�A�5*

loss_�=;UX       �	����ec�A�5*

loss݊�:yӖ�       �	����ec�A�5*

loss!�=���       �	((��ec�A�5*

lossE�O<���       �	4���ec�A�5*

loss���;�� Q       �	|��ec�A�5*

loss�\<��u       �	�o��ec�A�5*

loss���;�!��       �	s��ec�A�5*

loss��;
�б       �	�"��ec�A�5*

lossl2u=��S�       �	UM��ec�A�5*

loss�:�;�       �	����ec�A�5*

loss�i�:��:$       �	���ec�A�5*

loss�;
���       �	z���ec�A�5*

loss��:�/�       �	�J��ec�A�5*

loss���<��ȟ       �	|���ec�A�5*

loss�g<���/       �	�4��ec�A�5*

loss�J�<m6�y       �	<���ec�A�5*

lossx(�<<P3       �	=���ec�A�5*

lossAs�;�az)       �	���ec�A�5*

loss?~�<�Im�       �	����ec�A�5*

loss�,�;x�y       �	�F��ec�A�5*

loss,�B=C���       �	����ec�A�5*

loss8S�:��M(       �	{���ec�A�5*

loss�^L<تDs       �	�S��ec�A�5*

loss�]�9��&�       �	l���ec�A�5*

loss�-;\%/�       �	ɑ��ec�A�5*

loss�t<4�       �	�.��ec�A�5*

lossR�<b6�}       �	q���ec�A�5*

loss��<�A%       �	{j��ec�A�5*

loss��;�%c�       �	�
��ec�A�6*

loss?R;���       �	���ec�A�6*

loss�(�:W+��       �	
I��ec�A�6*

loss���;)�xx       �	����ec�A�6*

lossc� <Gf�       �	���ec�A�6*

loss4�K<[�p�       �	*��ec�A�6*

loss�u<<@X�?       �	���ec�A�6*

loss�=�<#9��       �	{N��ec�A�6*

losstI\<�y<,       �	���ec�A�6*

loss��;.A�]       �	����ec�A�6*

loss�%�<b�N       �	I���ec�A�6*

loss���=Q�Z�       �	\��ec�A�6*

loss��::'�F       �	9���ec�A�6*

lossk�9,��       �	x^��ec�A�6*

loss��=a��       �	i���ec�A�6*

loss��<5��       �	D���ec�A�6*

loss�<����       �	`>��ec�A�6*

lossR��;���       �	����ec�A�6*

loss6�==N       �	�s��ec�A�6*

loss3I�;��9�       �	��ec�A�6*

loss}\�;X,�       �	v���ec�A�6*

loss2L;n�`       �	pA��ec�A�6*

loss��\;��m       �	<���ec�A�6*

loss#р<]�#Y       �	D���ec�A�6*

loss�C�=��0       �	���ec�A�6*

loss���=�{�       �	���ec�A�6*

loss�K=P8�}       �	�Q��ec�A�6*

loss�t
<��P       �	2���ec�A�6*

lossJ�@:�]�	       �	�|��ec�A�6*

loss��9:�E0       �	���ec�A�6*

loss
o<���       �	���ec�A�6*

lossx�<��h       �	�C��ec�A�6*

lossaB�;
5W       �	����ec�A�6*

loss�M�<�K:S       �	z��ec�A�6*

loss�>�tz&       �	� �ec�A�6*

lossJL�;u���       �	� �ec�A�6*

loss|��<@�3       �	{J�ec�A�6*

loss�J�;� ��       �	���ec�A�6*

loss��y<q���       �	���ec�A�6*

loss<�T;�e�       �	m�ec�A�6*

loss��=��*       �	/��ec�A�6*

loss#�-<�b��       �	ni�ec�A�6*

loss��u:����       �	� �ec�A�6*

loss��:�#A       �	z��ec�A�6*

loss?��;�b4>       �	�K�ec�A�6*

losseל<�;zm       �	���ec�A�6*

lossF�:؊       �	�#�ec�A�6*

loss}��;���       �	�		�ec�A�6*

loss%�=�8��       �	i
�ec�A�6*

loss��S<��2�       �	�
�ec�A�6*

lossj�;SR�        �	�W�ec�A�6*

lossi�t;�)�       �	�<�ec�A�6*

loss\2�=(���       �	o��ec�A�6*

loss��;��.J       �	/m�ec�A�6*

loss.�4<�[L�       �	�	�ec�A�6*

loss��6<��}       �	zn�ec�A�6*

lossO�<�%�x       �	��ec�A�6*

loss���:~��_       �	��ec�A�6*

loss���;�v�       �	S�ec�A�6*

loss�S�<�~�       �	��ec�A�6*

loss�:���@       �	���ec�A�6*

lossH<�a�       �	�@�ec�A�6*

losst)<?�>'       �	��ec�A�6*

loss��N=�`�6       �	=�ec�A�6*

lossB�=wo (       �	6!�ec�A�6*

loss��=h�y�       �	���ec�A�6*

loss(C�;hw`T       �	e�ec�A�6*

loss�0�<��       �	.�ec�A�6*

lossHZ:���       �	/��ec�A�6*

loss���:�d>u       �	A�ec�A�6*

loss3�;�*�       �	���ec�A�6*

loss��>N~��       �	e��ec�A�6*

lossd�<�9��       �	�%�ec�A�6*

loss��:D�Jo       �	�;�ec�A�6*

loss���:�#>       �	T��ec�A�6*

loss��E=o���       �	R��ec�A�6*

lossX�;Y��e       �	%]�ec�A�6*

losszI8;�~7�       �	� �ec�A�6*

lossW�&;��R�       �	u��ec�A�6*

loss���<��	�       �	t� �ec�A�6*

loss��m;?��       �	$B!�ec�A�6*

loss�<	�(�       �	b�!�ec�A�6*

lossG=:��       �	q"�ec�A�6*

lossHȋ;u       �	0#�ec�A�6*

lossH:W�j�       �	f�#�ec�A�6*

loss���8n�       �	�F$�ec�A�6*

loss�{7:;�       �	L�$�ec�A�6*

loss�
:�`��       �	�v%�ec�A�6*

loss9�;����       �	#&�ec�A�6*

loss���<���       �	��&�ec�A�6*

loss��;����       �	�['�ec�A�6*

loss�;i�8�       �	\(�ec�A�6*

loss1�=�h��       �	��(�ec�A�6*

lossz=�<�?j       �	��)�ec�A�6*

loss�<ƈb�       �	*�+�ec�A�6*

loss��;�vL�       �	&T,�ec�A�6*

loss_�<
�       �	�,�ec�A�6*

loss�H'=V�       �	Ƣ-�ec�A�6*

loss��<���h       �	z8.�ec�A�6*

loss{0�<yt��       �	��/�ec�A�6*

loss��<����       �	�0�ec�A�6*

loss[�@<4�b�       �	�&1�ec�A�6*

loss��=�Bc       �	�1�ec�A�6*

loss���;q")       �	F`2�ec�A�6*

lossP<����       �	��2�ec�A�6*

loss��<�{ٳ       �	�3�ec�A�6*

loss�=���       �	M04�ec�A�6*

loss��<]�t       �	Y�4�ec�A�6*

loss��<q��       �	�V5�ec�A�6*

loss0	�; <�L       �	��5�ec�A�6*

lossߨ
<`'�       �	��6�ec�A�6*

loss��8Bs�&       �	�.7�ec�A�6*

loss�<��3       �	a�7�ec�A�6*

loss��7=�h�       �	�i8�ec�A�6*

loss#�v<�҇Z       �	� 9�ec�A�6*

lossM�<\��c       �	�9�ec�A�6*

loss&W_=}"       �		6:�ec�A�6*

loss�<F&:       �	V�:�ec�A�6*

lossԘ�:=�3       �	!v;�ec�A�6*

lossڽ�:�Ԍ�       �	�<�ec�A�6*

lossSeF=�^��       �	��<�ec�A�6*

loss��*<��&#       �	�b=�ec�A�6*

lossuo;�P       �	� >�ec�A�6*

loss��"<N�o|       �	0�>�ec�A�6*

loss3+U=���h       �	�A?�ec�A�6*

lossr��<U!f       �	��?�ec�A�6*

loss}�<H�a2       �	2w@�ec�A�6*

loss�5<�g��       �	�A�ec�A�6*

loss�@<=K�       �	ڬA�ec�A�7*

loss��= �j�       �	<NB�ec�A�7*

lossTsl<f]y�       �	�B�ec�A�7*

loss�-�<_~�       �	u�C�ec�A�7*

loss`)�<���       �	�*D�ec�A�7*

loss��<:��t        �	"�D�ec�A�7*

lossN�;��       �	^E�ec�A�7*

loss�K<���       �	��E�ec�A�7*

loss_h:I       �	ڍF�ec�A�7*

loss�ԯ;�t^       �	"G�ec�A�7*

loss ^;�,       �	��G�ec�A�7*

loss���=���       �	"UH�ec�A�7*

lossY ;�m��       �		I�ec�A�7*

lossz�<��~/       �	J�ec�A�7*

loss���:v��O       �	��J�ec�A�7*

loss���<"���       �	lK�ec�A�7*

loss7�9�ĵ       �	�L�ec�A�7*

loss�*!:���       �	@�L�ec�A�7*

loss�	�9�re�       �	�7M�ec�A�7*

loss i<���       �	}�M�ec�A�7*

loss��<����       �	akN�ec�A�7*

loss�З:ԅs       �	nO�ec�A�7*

loss�$�7ӧB       �	0�O�ec�A�7*

loss,�c;Q�       �	KYP�ec�A�7*

loss;�=zE}�       �	�P�ec�A�7*

loss�_R:2)-       �	��Q�ec�A�7*

lossX�(<(�E\       �	,R�ec�A�7*

lossLQ�=?e��       �	��R�ec�A�7*

lossh9<B�0       �	�jS�ec�A�7*

loss�;��A       �	�T�ec�A�7*

loss��Q9�ݼ�       �	ܝT�ec�A�7*

loss;K=%/;       �	(EU�ec�A�7*

lossl�:l\�       �	��U�ec�A�7*

loss�O+:-�gB       �	��V�ec�A�7*

loss�<<�Z       �	�0W�ec�A�7*

loss��:8�rn       �	E�W�ec�A�7*

loss�5�;t{�q       �	uvX�ec�A�7*

lossӂ�:P�7�       �	)Y�ec�A�7*

loss�η;k�%       �	Y�Y�ec�A�7*

loss}��:��	�       �	gZ�ec�A�7*

lossZk=���       �	I[�ec�A�7*

loss�$=�f��       �	��[�ec�A�7*

loss�i;G�       �	�n\�ec�A�7*

loss��);!�<       �	�]�ec�A�7*

losssE<��=`       �	^�]�ec�A�7*

losshb;��Yy       �	t\^�ec�A�7*

loss�v�;uwP�       �	�_�ec�A�7*

loss�<�h��       �	��_�ec�A�7*

lossadm<��4       �	�H`�ec�A�7*

lossN:�;�9�       �	��`�ec�A�7*

loss�o=/$׋       �	$�a�ec�A�7*

loss&O;t��G       �	�b�ec�A�7*

loss��M<����       �	��b�ec�A�7*

lossO��8^R^       �	�{c�ec�A�7*

lossXc<(~�       �	�d�ec�A�7*

loss�?�<���       �	�d�ec�A�7*

loss��<˕�)       �	�ye�ec�A�7*

lossX�<v�W�       �	�/f�ec�A�7*

loss��;$���       �	=�f�ec�A�7*

loss�<�YͲ       �	�hg�ec�A�7*

loss�)�<f���       �	`h�ec�A�7*

loss�o�<~�bg       �	k�h�ec�A�7*

losss�9���{       �	73i�ec�A�7*

lossnJ;"9o       �	��i�ec�A�7*

loss���;���_       �	�Zj�ec�A�7*

lossð�;W�       �	��k�ec�A�7*

lossD�<��7.       �	k�l�ec�A�7*

lossi9�;���K       �	 7m�ec�A�7*

lossv�":�^_       �	��m�ec�A�7*

loss|σ:l*��       �	Ppn�ec�A�7*

loss�%5:�Ù       �	�o�ec�A�7*

loss�No<�8�U       �	؟o�ec�A�7*

loss�x�<Ǌ"�       �	D2p�ec�A�7*

loss�;_:���4       �	��p�ec�A�7*

loss�
�;��8       �	�`q�ec�A�7*

loss;|`9ɱ       �	�q�ec�A�7*

lossJ��;�=�       �	�r�ec�A�7*

loss�f=B'�'       �	w,s�ec�A�7*

loss!g�:�xW�       �	)t�ec�A�7*

loss�?�91�e       �	��t�ec�A�7*

loss&P:E>       �	�cu�ec�A�7*

loss�$x8�agX       �	�u�ec�A�7*

loss�=�:�޿�       �	�v�ec�A�7*

lossmi+:��$       �	J(w�ec�A�7*

loss���:A��       �	
�w�ec�A�7*

loss�$�<��iO       �	�gx�ec�A�7*

loss��:@��F       �	��x�ec�A�7*

loss���9�2U       �	��y�ec�A�7*

loss�P;l&��       �	w,z�ec�A�7*

lossߍL<��}       �	��z�ec�A�7*

loss�(:���       �	�f{�ec�A�7*

lossl�	:K»x       �	e�{�ec�A�7*

lossj+}:	��       �	��|�ec�A�7*

loss��,;�m�e       �	 }�ec�A�7*

loss*h88��!�       �	1�}�ec�A�7*

loss�q�<��Q�       �	 T~�ec�A�7*

loss�D4:�       �	:�~�ec�A�7*

loss�!�:{�i/       �	��ec�A�7*

loss�h~<����       �	�"��ec�A�7*

loss3{	;1ო       �	����ec�A�7*

losss��:��7�       �	�N��ec�A�7*

lossc09�3>!       �	���ec�A�7*

loss!��8�(�5       �	�v��ec�A�7*

loss��8�G��       �	�	��ec�A�7*

losswȕ8��ߧ       �	a���ec�A�7*

loss�#a:��       �	�=��ec�A�7*

loss@��;r��       �	�ф�ec�A�7*

losso��7-�@       �	+m��ec�A�7*

loss@�;ʫD�       �	���ec�A�7*

loss��R=}��*       �	ӣ��ec�A�7*

lossl�:��mD       �	�=��ec�A�7*

loss��-=8� i       �	�ч�ec�A�7*

loss��`;
$B       �	���ec�A�7*

losso:(       �	c���ec�A�7*

loss�Oq<�N;`       �	�s��ec�A�7*

loss]l�;G:cI       �	-��ec�A�7*

loss:(�=��zX       �	����ec�A�7*

loss7��:�HR�       �	ē��ec�A�7*

lossaB:/��(       �	d\��ec�A�7*

loss��=?y�!       �	@��ec�A�7*

loss-)<H�Zq       �	�ώ�ec�A�7*

loss_�<;Ύ:g       �	Z���ec�A�7*

loss��/=���       �	+��ec�A�7*

loss��):JWpf       �	���ec�A�7*

lossؐ�;�l8       �	n���ec�A�7*

loss�:6���       �	34��ec�A�7*

loss�};��       �	��ec�A�7*

lossS��:��j�       �	n���ec�A�7*

loss~�=���       �	8N��ec�A�8*

loss6�=;�15       �	����ec�A�8*

loss�|�:#a�       �	����ec�A�8*

loss!`D=�mF@       �	�5��ec�A�8*

loss���;V��       �	�ߖ�ec�A�8*

loss��C;oΕ}       �	(���ec�A�8*

lossN;�;{���       �	^+��ec�A�8*

loss�3<��$L       �	�ɘ�ec�A�8*

loss��;v\G�       �	�n��ec�A�8*

loss7$Z:�6(       �	
��ec�A�8*

losszM#;nX�       �	����ec�A�8*

loss��q<9��       �	;T��ec�A�8*

loss�~;�r       �	���ec�A�8*

loss��%<G#       �	���ec�A�8*

loss��:As�       �	��ec�A�8*

loss�T�;.��       �	���ec�A�8*

loss�|�<�G\       �	]R��ec�A�8*

lossLH�;1n�z       �	���ec�A�8*

loss}ʣ;'-�{       �	���ec�A�8*

loss��<�w�8       �	*��ec�A�8*

lossn�C;v��       �	����ec�A�8*

loss�y<���       �	<K��ec�A�8*

lossSZ�:cSv�       �	����ec�A�8*

loss$/T;�r�#       �	禢�ec�A�8*

loss_�;� �       �	�N��ec�A�8*

loss��;*�V5       �	h��ec�A�8*

lossMV�;u:t"       �	␤�ec�A�8*

loss��;3�&       �	�?��ec�A�8*

loss�	�;	�N       �	Z���ec�A�8*

loss�6�;�ɃO       �	i���ec�A�8*

loss�
.=�r��       �	zR��ec�A�8*

loss̆�8��Fm       �	����ec�A�8*

loss6Cl9&5:       �	D���ec�A�8*

loss�-�;�Q}