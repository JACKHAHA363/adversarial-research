       �K"	  �!Yc�Abrain.Event:2�D�4�     ���(	���!Yc�A"��
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
seed2�ȵ*
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
:���������@*
seed2ޖv
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
valueB:*
_output_shapes
:*
dtype0
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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2���
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
:����������*
seed2Ȣz*
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
_output_shapes
:	�
*
seed2㉟*
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
seed2ħ�*
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
valueB:*
_output_shapes
:*
dtype0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2��*
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
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
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
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
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
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:���������
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
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*#
_output_shapes
:���������*
T0
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
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:���������
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
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
gradients/div_1_grad/mulMul9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshapegradients/div_1_grad/RealDiv_2*'
_output_shapes
:���������
*
T0
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
: "1�V���     >�x�	֩�!Yc�AJ��
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2�ȵ*
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
:���������@*
seed2ޖv*
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
:���*
seed2���
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2Ȣz*
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
 *̈́U>*
_output_shapes
: *
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�
*
seed2㉟*
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
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2ħ�*
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
seed2��
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
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
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
;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
_output_shapes
: *
T0
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
valueB *
dtype0*
_output_shapes
: 
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
valueB:*
_output_shapes
:*
dtype0
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
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
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

~
gradients/div_1_grad/RealDiv_1RealDivgradients/div_1_grad/Negdiv_1/y*
T0*'
_output_shapes
:���������

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
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"0
losses&
$
"softmax_cross_entropy_loss/value:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0����       ��-	<M "Yc�A*

loss&_@��?2       ��-	.� "Yc�A*

lossy@\_j       ��-	�!"Yc�A*

lossF@%�       ��-	zT""Yc�A*

losso�@��       ��-	�""Yc�A*

loss��@���       ��-	��#"Yc�A*

loss��@O�O       ��-	�N$"Yc�A*

loss?�@s%�       ��-	8�$"Yc�A*

loss�n@.l�       ��-	 �%"Yc�A	*

loss�?�m��       ��-	�J&"Yc�A
*

lossA�?��4�       ��-	��&"Yc�A*

loss3ɹ?�{�       ��-	�'"Yc�A*

loss��?�}�       ��-	E("Yc�A*

lossq �?=��       ��-	��("Yc�A*

lossm��?�Ɏ�       ��-	�)"Yc�A*

loss���?|h*�       ��-	�e*"Yc�A*

loss�j?\�W       ��-	�+"Yc�A*

loss=7�?�	�       ��-	��+"Yc�A*

loss%�_?[�M�       ��-		o,"Yc�A*

loss��?���(       ��-	�-"Yc�A*

loss�xq?���       ��-	ٵ-"Yc�A*

lossWXO?��i�       ��-	GT."Yc�A*

loss�w=?d��       ��-	��."Yc�A*

loss��?����       ��-	��/"Yc�A*

loss�e? �/       ��-	+50"Yc�A*

loss� `?����       ��-	I�0"Yc�A*

loss?�<?�{cU       ��-	�{1"Yc�A*

loss�I?���       ��-	�)2"Yc�A*

loss�`?Dy�       ��-	0�2"Yc�A*

loss��~??�x�       ��-	3"Yc�A*

lossRGd?�y��       ��-	�'4"Yc�A*

losssu/?�̂�       ��-	(�4"Yc�A *

lossJF:?p��       ��-	;�5"Yc�A!*

loss,?�       ��-	~S6"Yc�A"*

lossl�9?VQL       ��-	Z�6"Yc�A#*

lossx�>?T�xS       ��-	��7"Yc�A$*

loss�� ?RА�       ��-	2Y8"Yc�A%*

loss��/?��A:       ��-	0)9"Yc�A&*

loss��?�+s�       ��-	��9"Yc�A'*

lossx�Z?Q\|M       ��-	(b:"Yc�A(*

loss�E ?����       ��-	hz;"Yc�A)*

losss�?�ң       ��-	�<"Yc�A**

loss��?�3�       ��-	Ψ<"Yc�A+*

loss)$?ؑL
       ��-	�<="Yc�A,*

lossq�(?����       ��-	�="Yc�A-*

loss���>1�N       ��-	�>"Yc�A.*

loss��?>�B/       ��-	�d?"Yc�A/*

loss���>��'D       ��-	Vd@"Yc�A0*

lossZ?N�I�       ��-	��A"Yc�A1*

loss4i�>���       ��-	�}B"Yc�A2*

loss���>gHO       ��-	� C"Yc�A3*

loss7�	?,�@       ��-	+�C"Yc�A4*

lossX��>vAۦ       ��-	&WD"Yc�A5*

loss,�K?��fP       ��-	<�D"Yc�A6*

lossEe�>'*]       ��-	؟E"Yc�A7*

loss�o�>�H�P       ��-	�HF"Yc�A8*

loss��>��v�       ��-	0�F"Yc�A9*

loss�*�>C���       ��-	��G"Yc�A:*

loss��1?�q�)       ��-	t$H"Yc�A;*

loss���>�� #       ��-	n�H"Yc�A<*

loss��?t���       ��-	qXI"Yc�A=*

loss�e�>5��(       ��-	��I"Yc�A>*

loss�>�>7#S       ��-	��J"Yc�A?*

loss{�?5=M�       ��-	�0K"Yc�A@*

loss��>���       ��-	%L"Yc�AA*

loss85�>�       ��-	M�L"Yc�AB*

lossQAM?w���       ��-	MHM"Yc�AC*

loss�<(?w��       ��-	��M"Yc�AD*

lossg�?eU�       ��-	T�N"Yc�AE*

loss�}�>��y       ��-	)@O"Yc�AF*

loss��>8k��       ��-	�O"Yc�AG*

lossV��>
�!       ��-	�qP"Yc�AH*

lossN/�>ۦ�       ��-	Q"Yc�AI*

lossJ��>**�,       ��-	�Q"Yc�AJ*

loss��?�i       ��-	�YR"Yc�AK*

loss
]?AOy�       ��-	��R"Yc�AL*

loss��J?u׵�       ��-	�S"Yc�AM*

loss�!'?��|       ��-	�4T"Yc�AN*

loss���>vz��       ��-	o�T"Yc�AO*

lossH�>n��       ��-	�mU"Yc�AP*

loss��?I�C�       ��-	9V"Yc�AQ*

lossv�%?ɇ�E       ��-	{�V"Yc�AR*

loss&#?�U^       ��-	�>W"Yc�AS*

loss`̰>�MT       ��-	9�W"Yc�AT*

loss���>�4K       ��-	voX"Yc�AU*

lossI�>�ó�       ��-	Y"Yc�AV*

loss��>b�7        ��-	ܛY"Yc�AW*

lossx��>/7W       ��-	{0Z"Yc�AX*

loss���>n�Wy       ��-	�7["Yc�AY*

lossf��>�mL       ��-	��["Yc�AZ*

loss��?�v�       ��-	�f\"Yc�A[*

loss���>]P�U       ��-	�\"Yc�A\*

lossÊ�>A���       ��-	��]"Yc�A]*

loss �>D�P�       ��-	2^"Yc�A^*

lossh??ˌ��       ��-	��^"Yc�A_*

loss#��>�H�M       ��-	�j_"Yc�A`*

loss���>���       ��-	r`"Yc�Aa*

lossv�?s�&�       ��-	�a"Yc�Ab*

loss��-?m�j       ��-	�a"Yc�Ac*

lossP�>o�5�       ��-	\;b"Yc�Ad*

lossn��>�6�       ��-	,�b"Yc�Ae*

losssݐ>��2�       ��-		lc"Yc�Af*

lossh�?��V�       ��-	�d"Yc�Ag*

loss��>z�|       ��-	76e"Yc�Ah*

loss9�>�}u       ��-	��e"Yc�Ai*

loss�А>mEu)       ��-	��f"Yc�Aj*

loss�r�>� �!       ��-	hZg"Yc�Ak*

loss��]>�׽)       ��-	��g"Yc�Al*

loss�
?�A�       ��-	/�h"Yc�Am*

loss�׽>��       ��-	��i"Yc�An*

loss��?���       ��-	�kj"Yc�Ao*

lossCd�>].��       ��-	`k"Yc�Ap*

loss�߅>�ZɄ       ��-	��k"Yc�Aq*

loss��:>V��}       ��-	�+l"Yc�Ar*

loss�o�>��R=       ��-	߿l"Yc�As*

loss���>���       ��-	vSm"Yc�At*

loss�<�>t�s       ��-	��m"Yc�Au*

loss6l�>6Bɩ       ��-	(�n"Yc�Av*

loss���>�V$       ��-	Do"Yc�Aw*

loss��>����       ��-	6�o"Yc�Ax*

loss��>�]�W       ��-	�Cp"Yc�Ay*

loss%�>NM��       ��-	4�p"Yc�Az*

lossry�>�;�5       ��-	�nq"Yc�A{*

losshʧ>ă�       ��-	r"Yc�A|*

loss@��=L3��       ��-	g�r"Yc�A}*

lossW�>�_�       ��-	#/s"Yc�A~*

loss��>g�       ��-	��s"Yc�A*

loss���>�       �	=�t"Yc�A�*

loss�ͮ>}��       �	��u"Yc�A�*

loss�޷>�ܭ�       �	oJv"Yc�A�*

loss#d�>��       �	�v"Yc�A�*

loss�>��7�       �	rw"Yc�A�*

loss�_">c�u�       �	-&x"Yc�A�*

loss�E#>�&��       �	y"Yc�A�*

loss�V>�(��       �	��y"Yc�A�*

lossQ�>��Z       �	C�z"Yc�A�*

loss��> 	�`       �	�{"Yc�A�*

loss�SZ>����       �	%$|"Yc�A�*

lossZx>+(d       �	}"Yc�A�*

loss�K�=zo��       �	�H~"Yc�A�*

loss��!>�u�       �	�~"Yc�A�*

loss���>��       �	�w"Yc�A�*

loss[̢>�lO�       �	���"Yc�A�*

loss���>2;�6       �	�?�"Yc�A�*

lossa��>9��       �	y�"Yc�A�*

loss�?J>���p       �	���"Yc�A�*

loss��>�3��       �	4+�"Yc�A�*

loss���=��Zl       �	σ"Yc�A�*

loss<^>Ș��       �	�i�"Yc�A�*

loss�B�>�j�       �	~ �"Yc�A�*

loss��M>�Jh�       �	1��"Yc�A�*

loss�4�>����       �	3�"Yc�A�*

lossNB�>?��       �	'J�"Yc�A�*

loss�/�>�S��       �	��"Yc�A�*

loss��">���|       �	��"Yc�A�*

loss�&>&�$�       �	8��"Yc�A�*

loss�o>�*��       �	�W�"Yc�A�*

loss�>��ӈ       �	�o�"Yc�A�*

loss�R�>�mF       �	%�"Yc�A�*

loss�E�>�Xov       �	�b�"Yc�A�*

loss��>��W�       �	f1�"Yc�A�*

lossV�>����       �	ˎ"Yc�A�*

lossnw>ˆ�R       �	���"Yc�A�*

loss�9>���2       �	��"Yc�A�*

loss��>:İ�       �	��"Yc�A�*

loss$q>�SL!       �	�H�"Yc�A�*

loss��j>A���       �	
ۑ"Yc�A�*

loss^1�>�X��       �	�q�"Yc�A�*

loss
X{>ЫV       �	V�"Yc�A�*

loss�J>�s�       �	��"Yc�A�*

loss��/>���       �	�I�"Yc�A�*

lossfJ2>���       �	ݔ"Yc�A�*

lossZ}>�?�K       �	�n�"Yc�A�*

lossM�>w��+       �	��"Yc�A�*

losss�%>[��Y       �	?��"Yc�A�*

loss&@�>E��       �	#K�"Yc�A�*

lossJ6?�*TJ       �	��"Yc�A�*

loss��^>J �Z       �	��"Yc�A�*

loss�pA>��̚       �	y$�"Yc�A�*

loss�i�>�̷       �	���"Yc�A�*

loss�7�=����       �	\�"Yc�A�*

lossX:�>|{ҫ       �	���"Yc�A�*

loss��F>+<�       �	���"Yc�A�*

loss�,>�c�B       �	��"Yc�A�*

lossn�>�9�       �	鹜"Yc�A�*

lossS}�>]Uύ       �	XV�"Yc�A�*

loss�:.>4Hou       �	��"Yc�A�*

loss�ǘ>�l%o       �	w��"Yc�A�*

lossDҍ>&�b       �	y$�"Yc�A�*

loss �>�R��       �	��"Yc�A�*

loss���>o
�<       �	���"Yc�A�*

loss��A>67�       �	 S�"Yc�A�*

loss�Y<>~�d/       �	(�"Yc�A�*

lossnq�>���       �	a��"Yc�A�*

lossz]>���       �	%�"Yc�A�*

losso�?>��       �	@ܣ"Yc�A�*

loss�\�>^G!
       �	�@�"Yc�A�*

loss�O�>���f       �	�ޥ"Yc�A�*

loss�I�>��ͩ       �	2u�"Yc�A�*

loss�̃>��d       �	�"Yc�A�*

loss�|W>njY�       �	���"Yc�A�*

loss��)>����       �	�<�"Yc�A�*

lossQme>��>       �	��"Yc�A�*

loss��N>�,�)       �	�w�"Yc�A�*

loss�C�><<�2       �		�"Yc�A�*

lossT��=_��[       �	�ê"Yc�A�*

loss���>	**S       �	hY�"Yc�A�*

loss9�>V�\1       �	��"Yc�A�*

lossCC%>���\       �	p��"Yc�A�*

loss�%�>#��       �	�*�"Yc�A�*

loss�-(>����       �	'��"Yc�A�*

loss��>�)       �	&��"Yc�A�*

loss�>{��*       �	t$�"Yc�A�*

loss �>� xJ       �	���"Yc�A�*

loss�>�       �	wN�"Yc�A�*

loss��>���       �	k�"Yc�A�*

loss���>�{�       �	Υ�"Yc�A�*

loss�I�>��z       �	�=�"Yc�A�*

loss[��>���       �	�ճ"Yc�A�*

loss���>,xY       �	ѭ�"Yc�A�*

loss@�=��'�       �	%\�"Yc�A�*

loss�S+>���       �	~9�"Yc�A�*

loss}�>[ :       �	�Ͷ"Yc�A�*

loss��>��=       �	vk�"Yc�A�*

loss��>$��       �	���"Yc�A�*

loss*�
>&x3�       �	W��"Yc�A�*

lossf�/> f~$       �	�=�"Yc�A�*

loss���><�k2       �	��"Yc�A�*

lossp��>X}@�       �	���"Yc�A�*

lossۓ(>|�n       �	�"Yc�A�*

loss�(O>���       �	���"Yc�A�*

loss�_�=���_       �	`��"Yc�A�*

loss���>*Û       �	�,�"Yc�A�*

lossj�8>%`�       �	.�"Yc�A�*

loss�.�>>X       �	i��"Yc�A�*

lossS��=���       �	�F�"Yc�A�*

loss��:>�       �	�"Yc�A�*

loss�> O�o       �	@��"Yc�A�*

loss���>:!�       �	s,�"Yc�A�*

loss�8M>3�k�       �	R��"Yc�A�*

lossD�>���       �	s�"Yc�A�*

lossq^>�b�       �	��"Yc�A�*

loss�9�>g0ov       �	���"Yc�A�*

lossT�]>FwkK       �	3m�"Yc�A�*

loss(U�>��	2       �	i�"Yc�A�*

loss�Ϩ>.���       �	��"Yc�A�*

loss��?>fIE�       �	�1�"Yc�A�*

loss̊>�L?�       �	���"Yc�A�*

loss4�>��       �	b�"Yc�A�*

lossه>�h�       �	��"Yc�A�*

loss���>�.       �	ޑ�"Yc�A�*

loss���>	��l       �	,(�"Yc�A�*

loss���>1��       �	���"Yc�A�*

loss�k=̖a�       �	~V�"Yc�A�*

loss
�=���       �	�y�"Yc�A�*

lossF�P>��       �	��"Yc�A�*

loss�H>[#Dh       �	���"Yc�A�*

loss�C�>�4��       �	AI�"Yc�A�*

loss��M>+���       �	���"Yc�A�*

loss��>��/W       �	���"Yc�A�*

lossds>D�       �	�"Yc�A�*

loss��>Z���       �	��"Yc�A�*

lossZ/�>v�ۉ       �	�Q�"Yc�A�*

loss��n>�5ؘ       �	�"Yc�A�*

lossb�>�T�       �	q��"Yc�A�*

loss{Y>a�T       �	#g�"Yc�A�*

lossO�6>P��       �	S�"Yc�A�*

loss��F>�yV       �	ܡ�"Yc�A�*

loss�N�>�	       �	�=�"Yc�A�*

loss�!E>"�e       �	#��"Yc�A�*

loss�~�>9�#�       �	�s�"Yc�A�*

loss)G>�9�       �	F�"Yc�A�*

loss��>�?	       �	���"Yc�A�*

loss��@>eeu        �	�k�"Yc�A�*

loss{Lr>�LG2       �	��"Yc�A�*

lossņ�>�D�       �	;��"Yc�A�*

loss=�H>����       �	�|�"Yc�A�*

loss9@>���       �	D�"Yc�A�*

loss�\�>nq�%       �	��"Yc�A�*

loss0N>���       �	�^�"Yc�A�*

loss���=��_       �	��"Yc�A�*

loss��H>G�|       �	5��"Yc�A�*

loss,Y\>�vu       �	 B�"Yc�A�*

loss�>mL`       �	���"Yc�A�*

loss[MS>�woV       �	���"Yc�A�*

loss��+>�Sз       �	��"Yc�A�*

lossĉ&>?�,@       �	��"Yc�A�*

loss�i�=�ףK       �	S��"Yc�A�*

loss8�W>��a�       �	�d�"Yc�A�*

loss@�n=��a       �	̙�"Yc�A�*

loss3Ǟ=`S�6       �	�S�"Yc�A�*

loss��Y>�At       �	��"Yc�A�*

loss<�>�IX       �	�9�"Yc�A�*

lossA@>n�"       �	�X�"Yc�A�*

loss��>>�r       �	B��"Yc�A�*

lossi�>�v+V       �	���"Yc�A�*

loss��>�&	�       �	;n�"Yc�A�*

loss��>��{       �	e��"Yc�A�*

loss<Ý>?ӳ       �	�?�"Yc�A�*

loss�Et>\�lo       �	��"Yc�A�*

loss8-S>�E��       �	���"Yc�A�*

loss���=�pY�       �	���"Yc�A�*

loss�5f>e���       �	���"Yc�A�*

loss��>܃pL       �	�l�"Yc�A�*

lossA3>�w�Q       �	�z�"Yc�A�*

loss6Տ>���-       �	��"Yc�A�*

loss���=BK�       �	d��"Yc�A�*

loss͈>��!R       �	�I�"Yc�A�*

loss@�V=I\�       �	X��"Yc�A�*

loss#��>9�w       �	o~�"Yc�A�*

lossZ�>��ɵ       �	'�"Yc�A�*

lossO�_>����       �	���"Yc�A�*

lossU>���)       �	nR�"Yc�A�*

loss�_1>Y!��       �	^��"Yc�A�*

lossn��=v4L6       �	t�"Yc�A�*

loss�/+>P��E       �	*��"Yc�A�*

loss<9�=���v       �	-C�"Yc�A�*

loss
��=ފ޿       �	���"Yc�A�*

loss���>5�R       �	]��"Yc�A�*

loss��;>����       �	�.�"Yc�A�*

loss�=eu       �	��"Yc�A�*

loss|9s>�ˊ5       �	�e�"Yc�A�*

loss�[>�e6       �	��"Yc�A�*

loss�K>&�R�       �	��"Yc�A�*

loss,��>�'&       �	.�"Yc�A�*

loss���>'�'�       �	a��"Yc�A�*

loss�o5>��       �	+i�"Yc�A�*

loss�s�=U`��       �	& #Yc�A�*

loss� J>+���       �	t� #Yc�A�*

loss�u�>�\�       �	2#Yc�A�*

loss�v>���       �	K�#Yc�A�*

loss�9�=0�@�       �	�i#Yc�A�*

loss@Y�=�|       �	 #Yc�A�*

loss<׸=��       �	Ւ#Yc�A�*

loss7�>8�}�       �	�&#Yc�A�*

loss��=� In       �	��#Yc�A�*

loss�;�>�k8,       �	`V#Yc�A�*

loss�no>��L       �	S�#Yc�A�*

loss��G>-��       �	
�#Yc�A�*

loss;��=�e^       �	#Yc�A�*

loss��E><`[�       �	�#Yc�A�*

lossC}�=�>��       �	�[#Yc�A�*

loss.">1��W       �	��#Yc�A�*

lossH��=�	��       �	ڑ	#Yc�A�*

loss�Q�>���       �	�6
#Yc�A�*

loss@�>�e&r       �	��
#Yc�A�*

loss�?=w~9       �	�_#Yc�A�*

loss>>�CT       �	
�#Yc�A�*

loss��>��7�       �	k�#Yc�A�*

loss$i>���       �	�0#Yc�A�*

loss���=�*+       �	x�#Yc�A�*

loss���=X5�(       �	"m#Yc�A�*

loss:F >ثE       �	�#Yc�A�*

loss�&)>���       �	��#Yc�A�*

lossA�c>���       �	+L#Yc�A�*

lossf��=�3�       �	g�#Yc�A�*

loss�.>WU�{       �	Ί#Yc�A�*

loss��=T���       �	�#Yc�A�*

loss��=D>�       �	��#Yc�A�*

loss�>>-�7�       �	�i#Yc�A�*

loss6�u> Z��       �	� #Yc�A�*

loss(��=�       �	��#Yc�A�*

loss:��=n �_       �	�K#Yc�A�*

loss
,?���`       �	~�#Yc�A�*

lossՐ>��O       �	��#Yc�A�*

lossQ\d>����       �	�#Yc�A�*

loss��)>����       �	��#Yc�A�*

loss���=�6�Q       �	&W#Yc�A�*

loss�W�=X�       �	�#Yc�A�*

loss4�=T(-�       �	J�#Yc�A�*

lossW��=ѱ|�       �	�k#Yc�A�*

lossQw%>V�P       �	|
#Yc�A�*

lossq`>����       �	X�#Yc�A�*

lossܲ@>�G�       �	=E#Yc�A�*

loss�&6>-���       �	��#Yc�A�*

loss]f�>��(       �	}u#Yc�A�*

loss&�0>�1�       �	�	#Yc�A�*

loss@PC>
A�       �	Q�#Yc�A�*

lossV��=)��       �	{I#Yc�A�*

loss��>gtW�       �	��#Yc�A�*

loss}xr>6?�       �	�� #Yc�A�*

loss��>���K       �	�!#Yc�A�*

lossd�K>L��       �	B!"#Yc�A�*

loss)$>�I�       �	^�"#Yc�A�*

loss<
>o��       �	�o##Yc�A�*

loss;��=o�-G       �	�($#Yc�A�*

loss��S>ф@       �	�%#Yc�A�*

loss�O>�g��       �	�&#Yc�A�*

loss��>�i��       �	�&#Yc�A�*

loss�/>�;       �	�^'#Yc�A�*

loss���>f
�,       �	�Y(#Yc�A�*

loss�f->���       �	��)#Yc�A�*

loss��>�,�h       �	LT*#Yc�A�*

loss���>����       �	�+#Yc�A�*

loss_��=~۾       �	d�+#Yc�A�*

loss�3>	��       �	AJ,#Yc�A�*

loss;�=M��       �	�.-#Yc�A�*

loss��>[h	�       �	ƿ-#Yc�A�*

loss��>�;GC       �	y�.#Yc�A�*

loss�0�><���       �	�[/#Yc�A�*

loss�>�π       �	�\0#Yc�A�*

losst�5>i?�       �	e�0#Yc�A�*

loss��m>s5.�       �	(�1#Yc�A�*

loss�`>\��       �	Ց2#Yc�A�*

loss�.�=@.       �	͒3#Yc�A�*

loss��j>���       �	�24#Yc�A�*

loss��>�/�       �	G�4#Yc�A�*

loss͹@>�s�       �	�`5#Yc�A�*

loss�>��/�       �	C�6#Yc�A�*

lossw�c> R`�       �	Y7#Yc�A�*

loss��=��e       �	|�7#Yc�A�*

loss��m>�Y��       �	f�8#Yc�A�*

loss3y$>�e�4       �	�9#Yc�A�*

loss�h�>�Hx       �	Q:#Yc�A�*

loss�1�>[���       �	>�:#Yc�A�*

losssG�>�:xe       �	 {;#Yc�A�*

loss��1=Y�'v       �	�<#Yc�A�*

loss�1�=�%j�       �	ߨ<#Yc�A�*

loss
��>kqA�       �	<=#Yc�A�*

losst�&>#q.�       �	��=#Yc�A�*

lossN��=�_�c       �	5{>#Yc�A�*

loss->A��       �	5?#Yc�A�*

loss�6>wF�       �	�?#Yc�A�*

loss�!�>S�@6       �	O<@#Yc�A�*

loss�TA>���       �	��@#Yc�A�*

loss�?^3<�       �	�zA#Yc�A�*

loss;��=(��       �	5B#Yc�A�*

loss&�_>RpM�       �	0�B#Yc�A�*

lossm�>�xL       �	�5C#Yc�A�*

loss��6>���       �	>�C#Yc�A�*

loss��U>(�@       �	�zD#Yc�A�*

loss�C >�NI       �	2E#Yc�A�*

lossȜ>Ŷ       �	��E#Yc�A�*

loss_`�=�);�       �	�kF#Yc�A�*

loss��>����       �	�G#Yc�A�*

loss]�=F��R       �	��G#Yc�A�*

loss�4�=.8��       �	�bH#Yc�A�*

loss�->�Mj�       �	��H#Yc�A�*

loss�s=����       �	k�I#Yc�A�*

lossE��=�S�       �	�~J#Yc�A�*

loss�M�>�X�)       �	�K#Yc�A�*

loss���>��;       �	T�K#Yc�A�*

lossmV�=�jS�       �	h@L#Yc�A�*

loss:�>�3f       �	'�L#Yc�A�*

loss�_=�h�y       �	qM#Yc�A�*

lossg��>
���       �	�N#Yc�A�*

lossR;�>'��       �	�N#Yc�A�*

loss�l$>���       �	<O#Yc�A�*

loss 6�=OƵ�       �	��O#Yc�A�*

loss&�>��%       �	�tP#Yc�A�*

loss)WM>E���       �	�Q#Yc�A�*

loss�D�=5�T�       �	R#Yc�A�*

loss�q�=��[L       �	�R#Yc�A�*

loss��Z>��       �	�@T#Yc�A�*

loss�s>0�0       �	'�T#Yc�A�*

lossc�]>����       �	�V#Yc�A�*

loss��>D1K[       �	C�V#Yc�A�*

loss��7>��u�       �	MW#Yc�A�*

lossv :>+�       �	��W#Yc�A�*

loss.�s>qʾ       �	�zX#Yc�A�*

lossi��=].�       �	Y#Yc�A�*

loss�=�JM�       �	ĴY#Yc�A�*

lossɁ�=#��       �	�NZ#Yc�A�*

loss�*>�'�       �	��Z#Yc�A�*

loss급=�i       �	�[#Yc�A�*

loss-L�=�Oq       �	�1\#Yc�A�*

lossVg�=^c*       �	}�\#Yc�A�*

loss�	�=7 !2       �	�b]#Yc�A�*

loss�E>�\+       �	7 ^#Yc�A�*

loss��=�8       �	Ք^#Yc�A�*

losso��>       �	F&_#Yc�A�*

loss$e.>�9(�       �	(`#Yc�A�*

loss�0�>��6       �	�`#Yc�A�*

loss��V>��^q       �	�ka#Yc�A�*

loss�=�b��       �	�b#Yc�A�*

loss<��=`��L       �	X�b#Yc�A�*

loss���=D��       �	��c#Yc�A�*

lossl�?>�PG�       �	Iid#Yc�A�*

loss&T>�)�       �	�ye#Yc�A�*

lossx�v>��:0       �	�kf#Yc�A�*

lossN�>Q �       �	`g#Yc�A�*

loss0,>0#       �	�g#Yc�A�*

loss;&>�:2�       �	��h#Yc�A�*

loss�>>#8:       �	�i#Yc�A�*

lossnO><b��       �	Mfj#Yc�A�*

loss7p�=��4       �	["k#Yc�A�*

loss��A>樌�       �	x�k#Yc�A�*

loss	si>�3J�       �	!Wl#Yc�A�*

loss��=Z�ь       �	��l#Yc�A�*

loss��,>aƇ       �	��m#Yc�A�*

loss��c>�LW�       �	
+n#Yc�A�*

loss/�P>���       �	;�n#Yc�A�*

loss$y<=��!       �	�p#Yc�A�*

loss��`=�aKO       �	�cq#Yc�A�*

loss4�=�X
�       �	�r#Yc�A�*

loss��:=��A       �	�r#Yc�A�*

loss��>˓�p       �	�Ss#Yc�A�*

loss�'=d���       �	{�s#Yc�A�*

loss���=���Z       �	ܛt#Yc�A�*

lossJ��=��E�       �	�Gu#Yc�A�*

loss�g>�%��       �	��u#Yc�A�*

loss�=�R       �	�v#Yc�A�*

loss�vk=���       �	 bw#Yc�A�*

lossd�>d�م       �	L�w#Yc�A�*

loss��>L�!�       �	�:y#Yc�A�*

loss�>CCn       �	Y6z#Yc�A�*

loss_��=g��       �	I{#Yc�A�*

loss���=R�C�       �	�{#Yc�A�*

loss��8>}?       �	`|#Yc�A�*

loss��=��`�       �	�|#Yc�A�*

loss��}=�o��       �	͕}#Yc�A�*

loss���=w�	       �	@/~#Yc�A�*

loss���=�黙       �	��~#Yc�A�*

loss$(
>��4       �	�h#Yc�A�*

loss��>���       �	\<�#Yc�A�*

loss�pw>UR�       �	|ҁ#Yc�A�*

loss
��=L�       �	�v�#Yc�A�*

lossm�>6��{       �	I�#Yc�A�*

loss���=�E�       �	U��#Yc�A�*

loss��=u$�5       �	W@�#Yc�A�*

loss��b=�7�W       �	���#Yc�A�*

loss]R�=��U?       �	�t�#Yc�A�*

loss22�=5��_       �	2��#Yc�A�*

loss��=��3       �	�,�#Yc�A�*

lossP+>x�_O       �	俈#Yc�A�*

loss��=���K       �	�T�#Yc�A�*

losstp�=���       �	I��#Yc�A�*

loss��E>6�       �	���#Yc�A�*

loss��Z>g�:S       �	)&�#Yc�A�*

losseD=���       �	�'�#Yc�A�*

loss|!�=G|       �	
-�#Yc�A�*

loss=�>�û       �	͎#Yc�A�*

loss�'=}��       �	�i�#Yc�A�*

loss���<H��       �	c	�#Yc�A�*

lossdlJ=ت�Q       �	���#Yc�A�*

loss#�W=8��       �	K<�#Yc�A�*

loss�V�<'F�u       �	ב#Yc�A�*

loss�3
=���       �	�k�#Yc�A�*

loss<��=Ω       �	?9�#Yc�A�*

loss��P>0t�       �	Yړ#Yc�A�*

loss��y<��mt       �	!x�#Yc�A�*

lossgE<{I5u       �	�M�#Yc�A�*

loss�A&<����       �	��#Yc�A�*

loss��>>�#�       �	��#Yc�A�*

loss;}g>�m(       �	_&�#Yc�A�*

loss��=�ͤS       �	��#Yc�A�*

loss�y<a�t�       �	B[�#Yc�A�*

lossM�=��       �	���#Yc�A�*

loss�J?��[       �	y��#Yc�A�*

loss5��<2b�k       �	�E�#Yc�A�*

lossd�>�g�G       �	�ۚ#Yc�A�*

lossd��=@r��       �	�r�#Yc�A�*

lossd?>���       �	R
�#Yc�A�*

loss(��=��
�       �	&Ȝ#Yc�A�*

loss�'=�X�L       �	`u�#Yc�A�*

loss��8>Mω       �	V�#Yc�A�*

loss�1>K٧�       �	ߥ�#Yc�A�*

loss��I>e�Yn       �	X;�#Yc�A�*

loss�->�C'�       �	�ҟ#Yc�A�*

lossؒD>׭��       �	�k�#Yc�A�*

loss=�_>`|?       �	��#Yc�A�*

lossC�0>(D�Z       �	���#Yc�A�*

loss�=�au       �	��#Yc�A�*

loss �v>S��w       �	���#Yc�A�*

loss~�>dN�|       �	o�#Yc�A�*

lossn/�=D�{J       �	�q�#Yc�A�*

loss�4�=$�Y       �	� �#Yc�A�*

loss�r>s�O�       �	�A�#Yc�A�*

lossQ�=o7i�       �	;�#Yc�A�*

lossL׆=Օ�       �	x|�#Yc�A�*

loss_�=g�U_       �	cѩ#Yc�A�*

loss)�;>@�Z       �	�v�#Yc�A�*

lossl(=��7u       �	�N�#Yc�A�*

lossJ�=AC�A       �	R�#Yc�A�*

lossC=����       �	���#Yc�A�*

loss�+>��-a       �	�$�#Yc�A�*

loss3��<�{�       �	���#Yc�A�*

losse�>�%O       �	ʉ�#Yc�A�*

losss�=!�k�       �	�*�#Yc�A�*

loss�̧=4&-G       �	�Ȱ#Yc�A�*

loss�I�=��E�       �	�ұ#Yc�A�*

loss/�>Β�       �	%x�#Yc�A�*

loss�WL=��p�       �	�,�#Yc�A�*

loss}�=؃��       �	�˳#Yc�A�*

loss!�b=�	`       �	ys�#Yc�A�*

loss���=�F       �		�#Yc�A�*

loss��E>�v²       �	0��#Yc�A�*

loss.M>���       �	�}�#Yc�A�*

loss%�=\�       �	��#Yc�A�*

loss�F�=X*��       �	��#Yc�A�*

loss���=p�       �	�?�#Yc�A�*

loss�?>���       �	�ݸ#Yc�A�*

loss�~>��A�       �	1{�#Yc�A�*

lossMU�=wd       �	j�#Yc�A�*

loss��=�qC�       �	Χ�#Yc�A�*

lossYD>���       �	p[�#Yc�A�*

lossü�=�C��       �	���#Yc�A�*

losss��=M�\8       �	��#Yc�A�*

loss�
>�Y��       �	!�#Yc�A�*

loss��=��       �	��#Yc�A�*

lossS(>w�i?       �	 7�#Yc�A�*

loss��=�߆�       �	���#Yc�A�*

loss1Jc>�I�k       �	[}�#Yc�A�*

lossaM>A�ho       �	{�#Yc�A�*

loss*Q>>ym�       �	]��#Yc�A�*

loss�ҹ=� �7       �	�C�#Yc�A�*

loss�-'>��,?       �	���#Yc�A�*

lossR`1>[��       �	���#Yc�A�*

loss�$�=�J�       �	t&�#Yc�A�*

loss=�L>Ÿ�       �	(�#Yc�A�*

lossJ��=��d�       �	)��#Yc�A�*

lossd�=��p       �	rP�#Yc�A�*

lossF�=�: �       �	g��#Yc�A�*

loss�H�=+��       �	���#Yc�A�*

loss�p=�n       �	�0�#Yc�A�*

loss��>�9       �	C��#Yc�A�*

loss���=e���       �	�[�#Yc�A�*

loss�E�=�2�#       �	���#Yc�A�*

loss���=����       �	��#Yc�A�*

lossJn�>��&       �	#1�#Yc�A�*

loss��'>C��       �	���#Yc�A�*

loss&)>�	       �	���#Yc�A�*

lossS��=����       �	4��#Yc�A�*

loss��b>3+0       �	fg�#Yc�A�*

loss,�%>x�5-       �	�.�#Yc�A�*

loss�U�=��p�       �	�D�#Yc�A�*

loss&�=2�<       �	���#Yc�A�*

loss�(�=<I�       �	���#Yc�A�*

loss	��=����       �	��#Yc�A�*

loss�>�"Q       �	��#Yc�A�*

loss��=��#       �	�9�#Yc�A�*

loss��;=?ښ       �	�#Yc�A�*

loss�4>~ؽ       �	���#Yc�A�*

loss�}�>e,n�       �	���#Yc�A�*

loss%�=:�щ       �	(�#Yc�A�*

loss]�/>�8�       �	���#Yc�A�*

loss(�Z=�7�       �	eq�#Yc�A�*

lossE�U>���d       �	v�#Yc�A�*

loss��>��S�       �	���#Yc�A�*

loss�1�>��֜       �	W�#Yc�A�*

loss<8!>���R       �	���#Yc�A�*

loss}�=N�X;       �	x��#Yc�A�*

lossd�u=$�<g       �	j3�#Yc�A�*

loss�~
>J��       �	���#Yc�A�*

loss�Q>�):       �		q�#Yc�A�*

loss$�>�sƍ       �	l	�#Yc�A�*

loss�5�=���e       �	���#Yc�A�*

loss�p�=�*�       �	���#Yc�A�*

loss=��=���       �	 &�#Yc�A�*

loss�3�<j��O       �	q��#Yc�A�*

loss�7�<K�-       �	�n�#Yc�A�*

loss�@>�O+�       �	Q�#Yc�A�*

lossrx�=�M�       �	$��#Yc�A�*

loss}:?���$       �	�U�#Yc�A�*

loss�f=1       �	��#Yc�A�*

losst�=���n       �	���#Yc�A�*

loss�X=$>�       �	�I�#Yc�A�*

lossś5=�P8)       �	���#Yc�A�*

losso�>���       �	���#Yc�A�*

loss#�=YI'       �	���#Yc�A�*

loss�d�=�X�       �	�p�#Yc�A�*

lossO�=3_^       �	� $Yc�A�*

lossr�=N���       �	=� $Yc�A�*

loss��>�-^|       �	��$Yc�A�*

lossc;�=7j�Y       �	�$Yc�A�*

loss���=).	R       �	$Yc�A�*

loss\�1>T�a       �	��$Yc�A�*

loss/��=��A       �	mS$Yc�A�*

loss�i�>��4j       �	T�$Yc�A�*

loss�8>9��       �	V�$Yc�A�*

losss��=�L2e       �	�H$Yc�A�*

loss�x>�n.       �	��$Yc�A�*

loss�X>^�	�       �	֏$Yc�A�*

loss\��=3e       �	�5	$Yc�A�*

loss�>� ]�       �	,�	$Yc�A�*

lossL�">��M}       �	��
$Yc�A�*

loss�iI>�q�       �	B�$Yc�A�*

loss��P>�X��       �	+L$Yc�A�*

loss�^>6�Z       �	<1$Yc�A�*

lossWq�=Y)f�       �	��$Yc�A�*

losssQ>T���       �	p_$Yc�A�*

loss�+e>$i[�       �	j�$Yc�A�*

lossγ�=�]*�       �	$�$Yc�A�*

lossa�=�E       �	�6$Yc�A�*

lossJ��=��Z�       �	�$Yc�A�*

loss�2�=�M��       �	4�$Yc�A�*

loss�&�=�B�       �	e9$Yc�A�*

loss�
>�'�       �	o�$Yc�A�*

loss16�=��'M       �	�p$Yc�A�*

loss�[�=�8�       �	�o$Yc�A�*

loss�6�=~�$       �	I$Yc�A�*

loss�>�{��       �	��$Yc�A�*

loss8{�=����       �	ס$Yc�A�*

loss��>D{��       �	RE$Yc�A�*

loss�L�>����       �	!�$Yc�A�*

loss5h�=�F       �	J~$Yc�A�*

loss@�>���       �	-#$Yc�A�*

loss�>�=�#b�       �	g�$Yc�A�*

loss��{>��       �	B[$Yc�A�*

loss}�=Rg��       �	�-$Yc�A�*

lossL�2=��q{       �	��$Yc�A�*

loss:ވ=���       �	nm$Yc�A�*

lossR6>n/�       �	�$Yc�A�*

lossꤋ=S�       �	��$Yc�A�*

loss���=x��0       �	�/$Yc�A�*

lossTG>��}       �	��$Yc�A�*

loss�99==tQ       �	yZ$Yc�A�*

losss,6=.�O       �	?�$Yc�A�*

loss6!2>K�\�       �	� $Yc�A�*

loss�O�=����       �	8.!$Yc�A�*

loss�kZ>*��K       �	��!$Yc�A�*

loss��=]�w       �	q"$Yc�A�*

loss�=��aS       �	�&#$Yc�A�*

loss�K�<�h�,       �	�$$Yc�A�*

loss��y=����       �	
�$$Yc�A�*

loss��=(�G�       �	��%$Yc�A�*

loss��*>SG2       �	d�&$Yc�A�*

loss(�>���       �	>�'$Yc�A�*

loss\'�=�8�       �	�Y($Yc�A�*

loss�s�=�(Z       �	XS)$Yc�A�*

loss?��=�5|       �	2 *$Yc�A�*

losst�>�BZ       �	��*$Yc�A�*

loss6�>Y`t       �	�+$Yc�A�*

loss�>�4��       �	�'-$Yc�A�*

loss���<�Tj
       �	�.$Yc�A�*

loss�>T��       �	v�.$Yc�A�*

loss�q�=���       �	@�/$Yc�A�*

lossR��=��v`       �	�F1$Yc�A�*

loss��9>���       �	��1$Yc�A�*

loss���=���       �	��2$Yc�A�*

loss��=	��       �	�3$Yc�A�*

lossf�>5�j�       �	�N4$Yc�A�*

loss`��=c��       �	�5$Yc�A�*

lossF�<F�W�       �	96$Yc�A�*

lossp�>��       �	�7$Yc�A�*

loss�y�=+��b       �	�.8$Yc�A�*

loss��>4�       �	��8$Yc�A�*

loss��=���       �	�j9$Yc�A�*

loss��=5 3       �	�=:$Yc�A�*

loss2(�=*�+�       �	2X;$Yc�A�*

loss�
==S       �	 <$Yc�A�*

loss�!�=���Z       �	& =$Yc�A�*

loss��T>�qб       �	��=$Yc�A�*

loss -�=����       �	��>$Yc�A�*

lossmY>D��j       �	��?$Yc�A�*

lossW�t=��       �	mq@$Yc�A�*

lossvdR>q�8       �	?�A$Yc�A�*

loss��y<y;��       �	��B$Yc�A�*

lossf`]=�-�       �	��C$Yc�A�*

lossrF�=����       �	�D$Yc�A�*

loss�5�=��       �	�-E$Yc�A�*

loss��2>�B�Z       �	O�E$Yc�A�*

loss�L�=�8�u       �	�jF$Yc�A�*

lossd�]=�S�L       �	_G$Yc�A�*

loss%	>f��       �	��G$Yc�A�*

loss?B<��        �	5EH$Yc�A�*

lossQZ=u\�E       �	4�H$Yc�A�*

loss�v#>����       �	�I$Yc�A�*

lossߚN>���r       �	B$J$Yc�A�*

lossy�=�Ӷ\       �	k�J$Yc�A�*

loss��=�-�       �	��K$Yc�A�*

loss�^�=g/=       �	C9L$Yc�A�*

loss�f�=fK6V       �	+�L$Yc�A�*

loss͵ =,�       �	,~M$Yc�A�*

loss[J�=-`       �	#N$Yc�A�*

loss
�=��       �	��N$Yc�A�*

loss%Î=>�i       �	�[O$Yc�A�*

loss=�=����       �	��O$Yc�A�*

loss�x�=Y�/�       �	]�P$Yc�A�*

loss� y=j�'       �	s�Q$Yc�A�*

loss�b~=�VY       �	� R$Yc�A�*

loss��=$�       �	��R$Yc�A�*

loss(t�=�jN�       �	rjS$Yc�A�*

loss��8>t���       �	T$Yc�A�*

lossۮ=�>G       �	H�T$Yc�A�*

loss1�>����       �	KU$Yc�A�*

loss�;@>ZG��       �	��U$Yc�A�*

loss���=�c+�       �	��V$Yc�A�*

loss ��=��$o       �	�jW$Yc�A�*

loss�y�=ķx#       �	bX$Yc�A�*

loss�^�<�݊       �	.�X$Yc�A�*

loss=��=�B5`       �	�nY$Yc�A�*

loss��=K�ن       �	�Z$Yc�A�*

lossh�8=� ?n       �	$�Z$Yc�A�*

loss�]>��R�       �	�i[$Yc�A�*

lossD��=y�       �	�\$Yc�A�*

loss�D�=nNu       �	��\$Yc�A�*

loss���=���2       �	~n]$Yc�A�*

lossM&�=��S�       �	�^$Yc�A�*

lossN7�=�B�       �	C�^$Yc�A�*

loss�$'>ٜh       �	|_$Yc�A�*

loss���=m6^       �	�`$Yc�A�*

lossKϙ=��&�       �	��`$Yc�A�*

lossh�o=�/��       �	Eda$Yc�A�*

lossԌ=I���       �	�9b$Yc�A�*

loss۸�=	�E       �	0c$Yc�A�*

lossI�">�(�	       �	��c$Yc�A�*

loss��=$��m       �	qrd$Yc�A�*

loss슋=�4{       �	e$Yc�A�*

loss���=����       �	6�e$Yc�A�*

loss�
H=ł�       �	�vf$Yc�A�*

loss�%=���!       �	p%g$Yc�A�*

loss�<�d�8       �	��g$Yc�A�*

loss��=���i       �	�dh$Yc�A�*

lossO͝=<�8�       �	�i$Yc�A�*

loss�>�=�,��       �	�i$Yc�A�*

lossUI>���j       �	SYj$Yc�A�*

loss(;�=���       �	k$Yc�A�*

loss��=,�u       �	�l$Yc�A�*

loss���=n��       �	��l$Yc�A�*

loss���=���X       �	6Wm$Yc�A�*

lossZ�v=I�a       �	�n$Yc�A�*

lossA�@<��jx       �	O�n$Yc�A�*

loss,"=���=       �	�Yo$Yc�A�*

loss	>d�#�       �	��o$Yc�A�*

lossL�=�FZ       �	I�p$Yc�A�*

loss�aQ>���D       �	gFq$Yc�A�*

loss�֋>�:y�       �	��q$Yc�A�*

lossT�]>�
B�       �	߇r$Yc�A�*

loss�P�>��͹       �	N+s$Yc�A�*

lossf�S=Of.3       �	��s$Yc�A�*

loss�}[=h�	/       �	nlt$Yc�A�*

loss��$>���       �	u$Yc�A�*

loss�[<>g�j�       �	ıu$Yc�A�*

loss��	=�\6�       �	}�v$Yc�A�*

lossܣp=<^�@       �	�]w$Yc�A�*

lossj�>t�+�       �	��w$Yc�A�*

loss�ܸ=��u�       �	b�x$Yc�A�*

loss��U=t��       �	�Ly$Yc�A�*

lossxx,>.�s�       �	A�y$Yc�A�*

loss�:D=��o.       �	��z$Yc�A�*

loss�B0=��:
       �	~:{$Yc�A�*

loss�->�1��       �	��{$Yc�A�*

lossz�>:���       �	�|$Yc�A�*

loss��I>���a       �	t'}$Yc�A�*

loss?�9>��W"       �	h�}$Yc�A�*

loss���=f��       �	Re~$Yc�A�*

loss�=">�4�^       �	�$Yc�A�*

loss"2>�CL�       �	̘$Yc�A�*

loss|T,=���\       �	I/�$Yc�A�*

loss�C�=^�@       �	�ɀ$Yc�A�*

loss�v�=���       �	(d�$Yc�A�*

loss�$7>�'{l       �	���$Yc�A�*

lossVVa=�h3       �	���$Yc�A�*

lossW�=غ�O       �	,)�$Yc�A�*

loss�,>&S�       �	s��$Yc�A�*

lossa�=�x�       �	�h�$Yc�A�*

loss��= ps       �	J�$Yc�A�*

loss|�=2�Λ       �	\��$Yc�A�*

loss.k�=[�0R       �	�L�$Yc�A�*

loss�>��@       �	i �$Yc�A�*

loss��=�m�1       �	���$Yc�A�*

loss.y�=�M�       �	�4�$Yc�A�*

lossV��<�*T       �	�҈$Yc�A�*

loss6�s=�L4�       �	Qk�$Yc�A�*

lossd��=�*�       �	��$Yc�A�*

loss��=D�]�       �	���$Yc�A�*

loss�T�=���       �	X:�$Yc�A�*

lossi>�N�G       �	�ۋ$Yc�A�*

loss��=�ȱ�       �	�t�$Yc�A�*

loss��5>e���       �	��$Yc�A�*

loss�̀=�C       �	/��$Yc�A�*

lossS�$>��t�       �	\W�$Yc�A�*

loss?�=��A�       �	�$Yc�A�*

loss6�D>���]       �	׉�$Yc�A�*

loss�[�=*	��       �	�/�$Yc�A�*

lossWM*>ã�j       �	_Ԑ$Yc�A�*

loss��	> {>       �	ro�$Yc�A�*

loss��j>o�B�       �	+�$Yc�A�*

loss�>k҄N       �	���$Yc�A�*

lossa��=l?��       �	$_�$Yc�A�*

loss:��=�~[�       �	��$Yc�A�*

losss�=���       �	���$Yc�A�*

loss d=��       �	?;�$Yc�A�*

loss��>��\#       �	Օ$Yc�A�*

loss�Z>k���       �	�Ė$Yc�A�*

lossndg=l�_�       �	�_�$Yc�A�*

loss�<�f��       �	%�$Yc�A�*

loss�a�=ed�       �	,��$Yc�A�*

loss�Z8>�s9       �	VD�$Yc�A�*

loss��<tKУ       �	X�$Yc�A�*

loss���<�       �	���$Yc�A�*

lossC�>�7�       �	�)�$Yc�A�*

loss�N�='2       �	�ɛ$Yc�A�*

loss���=��J       �	j�$Yc�A�*

loss��=OJ�       �	��$Yc�A�*

loss�_�=����       �	ʧ�$Yc�A�*

loss��=�PS5       �	A�$Yc�A�*

loss��a=�0i       �	�Ԟ$Yc�A�*

loss��<JC�m       �	ds�$Yc�A�*

loss��!=}�c;       �	I�$Yc�A�*

loss�n>»�       �	㦠$Yc�A�*

lossO�>=CJr       �	fg�$Yc�A�*

loss��=�%�*       �	��$Yc�A�*

loss��!=�E�
       �	x��$Yc�A�*

loss�J�=�v(       �	}A�$Yc�A�*

loss��0>`f�       �	Yܣ$Yc�A�*

loss��?>��a�       �	w�$Yc�A�*

loss,>a�,       �	��$Yc�A�*

loss�E�>�H�       �	k��$Yc�A�*

loss�s%=R�@	       �	�\�$Yc�A�*

loss �r=����       �	���$Yc�A�*

lossX�=	��N       �	s��$Yc�A�*

loss�y�=��3�       �	���$Yc�A�*

loss�|�=2�o�       �	/O�$Yc�A�*

loss�m>H�       �	�E�$Yc�A�*

loss�_�<��|       �	���$Yc�A�*

lossc'7>`���       �	���$Yc�A�*

loss��h=(�L       �	ir�$Yc�A�*

loss�*>B�X�       �	(�$Yc�A�*

loss ad=�QC�       �	ٲ�$Yc�A�*

lossT�.>Ɲ�       �	]O�$Yc�A�*

loss�(?>�[+       �	���$Yc�A�*

loss�=��2       �	ܡ�$Yc�A�*

loss���<u��       �	�g�$Yc�A�*

losse>rK       �	_�$Yc�A�*

loss��/=|r�       �	`��$Yc�A�*

loss��=m�c�       �	,H�$Yc�A�*

loss�K> �M�       �	`�$Yc�A�*

lossq��=*�=�       �	���$Yc�A�*

loss�B�=���       �	I.�$Yc�A�*

loss�x&>�xշ       �	�Ѵ$Yc�A�*

lossd�F>���       �	Dl�$Yc�A�*

lossc�>TD@�       �	F�$Yc�A�*

loss�r>1��       �	�Ӷ$Yc�A�*

loss�X!>�2��       �	is�$Yc�A�*

loss.�Y=N��       �	�$Yc�A�*

loss}[=���       �	@��$Yc�A�*

lossp�=�t��       �	O]�$Yc�A�*

loss�C/>�=�       �	��$Yc�A�*

loss0��=1�GW       �	`��$Yc�A�*

loss}�e=��       �	�G�$Yc�A�*

loss��9=d�[�       �	��$Yc�A�*

loss�fw=
҃�       �	ˆ�$Yc�A�*

loss3�2>N�Y�       �	��$Yc�A�*

loss�Dn=X��       �	���$Yc�A�*

loss���= ��?       �	"l�$Yc�A�*

loss=��=���Y       �	 �$Yc�A�*

loss��=�Fc       �	[��$Yc�A�*

loss�7=��       �	w1�$Yc�A�*

loss�^�=�.~       �	���$Yc�A�*

lossV�<��m       �	|d�$Yc�A�*

lossNݼ=Z�       �	�$�$Yc�A�*

loss��(>��       �	w��$Yc�A�*

loss�l�>��d�       �	,c�$Yc�A�*

loss�W>т�       �	��$Yc�A�*

loss:�=E
�       �	��$Yc�A�*

lossm2>>M�W       �	cA�$Yc�A�*

loss�+�=F���       �	P��$Yc�A�*

loss��=�%n�       �	��$Yc�A�*

loss�\�=w�А       �	�*�$Yc�A�*

loss�Z�=T�@_       �	���$Yc�A�*

loss�f>��,�       �	i�$Yc�A�*

loss���=5��g       �	�$Yc�A�*

lossE�G>�u��       �	���$Yc�A�*

loss)�;=�Y^#       �	;��$Yc�A�*

loss�8o=�E       �	h$�$Yc�A�*

loss��i=�#       �	P��$Yc�A�*

loss�>S��       �	�c�$Yc�A�*

loss�!�=+��       �	2�$Yc�A�*

loss��=��       �	��$Yc�A�*

loss�ҥ=���       �	�.�$Yc�A�*

loss�="�n       �	���$Yc�A�*

loss��l>ۼ��       �	ٵ�$Yc�A�*

lossC�I>�qd        �	�d�$Yc�A�*

loss��=ɠk�       �	^�$Yc�A�*

loss�=��?�       �	"��$Yc�A�*

loss�;�=���       �	�i�$Yc�A�*

loss�%=����       �	��$Yc�A�*

loss��=>E�       �	U��$Yc�A�*

loss	�>���;       �	W>�$Yc�A�*

loss��=k(M�       �	��$Yc�A�*

lossẴ=w�<
       �	'��$Yc�A�*

loss�kw=�D1       �	��$Yc�A�*

lossA��=�v]%       �	���$Yc�A�*

lossH�->i�(       �	�v�$Yc�A�*

lossҟ>�@1�       �	��$Yc�A�*

loss-�d=%��W       �	���$Yc�A�*

loss�h=�,       �	�n�$Yc�A�*

loss��=��       �	��$Yc�A�*

lossZ1>�_��       �	נ�$Yc�A�*

lossw�=�=ɟ       �	;8�$Yc�A�*

lossi�/>��U�       �	��$Yc�A�*

lossW�B=d��       �	�f�$Yc�A�*

loss���<B��s       �	��$Yc�A�*

loss܌O=N6�       �	���$Yc�A�*

loss�=���       �	D4�$Yc�A�*

loss1�=3a       �	���$Yc�A�*

loss&}�=Uf�%       �	8h�$Yc�A�*

loss�>�= )�       �	���$Yc�A�*

losst�=W�       �	Ǹ�$Yc�A�*

lossH� >��p       �	#M�$Yc�A�*

loss��=��)�       �	���$Yc�A�*

loss�=/>"        �	ۅ�$Yc�A�*

loss�^}=}OH       �	v�$Yc�A�*

loss�>��d�       �	Fy�$Yc�A�*

loss���= ��       �	M�$Yc�A�*

loss���=uF��       �	���$Yc�A�*

lossN̾<�͹�       �	X�$Yc�A�*

lossf<�=1gk       �	Q�$Yc�A�*

loss��=`��       �	��$Yc�A�*

lossZ~>��Z       �	�W�$Yc�A�*

loss�$�=���T       �	��$Yc�A�*

lossƳO=�9       �	"��$Yc�A�*

lossd�=��       �	�*�$Yc�A�*

loss�{�=*�M�       �	��$Yc�A�*

loss>)���       �	=`�$Yc�A�*

lossM�>����       �	W�$Yc�A�*

loss6��=c*u       �	���$Yc�A�*

lossFc�=9���       �	�=�$Yc�A�*

loss���=jC [       �	���$Yc�A�*

loss��=��^a       �	~��$Yc�A�*

loss�{�=t��       �	�`�$Yc�A�*

loss�>Q%��       �	U��$Yc�A�*

lossftO>d��       �	���$Yc�A�*

loss�&Z>�       �	A+�$Yc�A�*

loss�h�<1��       �	��$Yc�A�*

lossn3=�[�e       �	�V�$Yc�A�*

lossR�5>$� �       �	-��$Yc�A�*

lossN>�r�       �	���$Yc�A�*

loss=�s=��<       �	P�$Yc�A�*

loss(Q�=�a�       �	?��$Yc�A�*

loss���=�f��       �	�?�$Yc�A�*

loss�?�>��w�       �	���$Yc�A�*

loss�F>��8       �	j�$Yc�A�*

lossF�r>sm=�       �	  �$Yc�A�*

loss��=ø+Q       �	B��$Yc�A�*

lossT.>g��D       �	6�$Yc�A�*

lossX�^=���       �	��$Yc�A�*

losss�=�W�Z       �	�n�$Yc�A�*

lossa=rj��       �	iU�$Yc�A�*

loss�r>�|3Y       �	f��$Yc�A�*

loss[��=����       �	P��$Yc�A�*

loss� �=��<       �	�h�$Yc�A�*

loss5>�P��       �	�$Yc�A�*

loss��R=?jS�       �	�E�$Yc�A�*

loss�k$=���       �	*�$Yc�A�*

loss��x=<g]/       �	x��$Yc�A�*

loss�T=u9�J       �	=a %Yc�A�*

loss�w'=f��       �	]N%Yc�A�*

lossnN>_��       �	�%Yc�A�*

loss��>��p       �	�%Yc�A�*

loss�=���       �	z6%Yc�A�*

loss�h(>���D       �	!�%Yc�A�*

lossCa<;�a       �	�m%Yc�A�*

lossݧ�=/>�       �	�%Yc�A�*

lossj�>>��       �	V�%Yc�A�*

loss�� >X���       �	�Q%Yc�A�*

loss��a=��9�       �	�%Yc�A�*

lossd�=-��       �	И%Yc�A�*

lossO��=���/       �	�>%Yc�A�*

loss�^>=;��F       �	��%Yc�A�*

loss�C=Ʃq       �	�~	%Yc�A�*

loss=�R>5#��       �	�
%Yc�A�*

loss���=�1�%       �	�
%Yc�A�*

loss���=*P       �	<P%Yc�A�*

loss�
>9��(       �	��%Yc�A�*

loss��x=TP�       �	�%Yc�A�*

losst#�=S�$�       �	�)%Yc�A�*

loss�.�=b�j`       �	��%Yc�A�*

loss��=)>@�       �	n%Yc�A�*

loss\�e=�܊[       �	^%Yc�A�*

lossAi�<0 �       �	�%Yc�A�*

loss�1>�z�       �	�H%Yc�A�*

loss]o=MQ�V       �	*�%Yc�A�*

lossW�6=�4��       �	�z%Yc�A�*

loss��e="�[m       �	�%Yc�A�*

lossZy�=	:1�       �	ö%Yc�A�*

loss���=#\=�       �	HS%Yc�A�*

loss�K�=[^Jt       �	�(%Yc�A�*

loss�(+>�E��       �	�%Yc�A�*

loss�1m=�m7�       �	�V%Yc�A�*

loss3��=4/k�       �	�%Yc�A�*

lossl#W>�W��       �	�g%Yc�A�*

loss�ݝ<��j       �	�%Yc�A�*

loss6X=�H%       �	U�%Yc�A�*

lossiY�<��.�       �	~9%Yc�A�*

loss %=��EC       �	��%Yc�A�*

loss�U=㛲{       �	'j%Yc�A�*

loss���=�B��       �	p�%Yc�A�*

loss�G�>r+�       �	�/%Yc�A�*

loss��{=�Ջ�       �	��%Yc�A�*

loss_�O=���A       �	^i%Yc�A�*

loss�
�=4�iD       �	 %Yc�A�*

lossJ��=V��\       �	@�%Yc�A�*

loss�Y=SS�a       �	�@%Yc�A�*

lossXݚ=���       �	r�%Yc�A�*

loss4�>�}8�       �	lx %Yc�A�*

loss��.=�u�\       �	!%Yc�A�*

lossT��=�pö       �	0�!%Yc�A�*

lossq}�=!�2G       �	a8"%Yc�A�*

loss�|�=[i�       �	F�"%Yc�A�*

loss �D=�c��       �	Zc#%Yc�A�*

loss}�e=jIFd       �	�>$%Yc�A�*

loss3�=!(v�       �	��$%Yc�A�*

loss@�=:�t       �	;�%%Yc�A�*

loss!ݸ=n9��       �	`Y&%Yc�A�*

loss���<�+�       �	T:'%Yc�A�*

loss���=��٪       �	~o(%Yc�A�*

loss���=���       �	�,)%Yc�A�*

loss��=z	�h       �	�)%Yc�A�*

loss��=�/�8       �	g*%Yc�A�*

loss�js=��$       �		+%Yc�A�*

loss�R�=��       �	�+%Yc�A�*

loss��>Ha�.       �	��,%Yc�A�*

loss_=�6       �	�.-%Yc�A�*

loss��>͘,I       �	vm.%Yc�A�*

loss���=��N       �	�A/%Yc�A�*

loss@'>���       �	 �/%Yc�A�*

loss�aa>/�4_       �	+�0%Yc�A�*

loss݁�=��s       �	�l1%Yc�A�*

loss���=�e�B       �	�2%Yc�A�*

lossJ{�<'�b       �	g�2%Yc�A�*

lossM�=���       �	�\3%Yc�A�*

loss�7"=���       �	z�3%Yc�A�*

loss��=;%8�       �	�4%Yc�A�*

loss�ȁ=�$UZ       �	�F5%Yc�A�*

lossỀ=��       �	DR6%Yc�A�*

loss�}�<�;��       �	4.7%Yc�A�*

loss���=h��       �	��7%Yc�A�*

loss�/=2T       �	�8%Yc�A�*

loss)��=��&%       �	IK9%Yc�A�*

loss�M=E�       �	��9%Yc�A�*

lossL=O�B�       �	F�:%Yc�A�*

loss�x�=���[       �	r1;%Yc�A�*

loss��J=���       �	~�;%Yc�A�*

loss&�]=#�       �	B]<%Yc�A�*

loss� �=0}�       �	o�<%Yc�A�*

loss�G>�֑`       �	=%Yc�A�*

loss��<��N]       �	�>%Yc�A�*

loss�d<�*�       �	˼>%Yc�A�*

lossq?�=�Z׀       �	�P?%Yc�A�*

loss<�=���       �	&�?%Yc�A�*

loss���;�P�       �	c{@%Yc�A�*

loss<���       �	VA%Yc�A�*

loss���<ټpu       �	H�A%Yc�A�*

loss|�<qv�"       �	e8B%Yc�A�*

loss�w=x.:       �	S�B%Yc�A�*

lossz�<�`�       �	�aC%Yc�A�*

loss��=Y��W       �	��C%Yc�A�*

lossF�;G�4�       �		�D%Yc�A�*

loss�׈;��0'       �	t&E%Yc�A�*

loss��;�l       �	�E%Yc�A�*

loss�<��       �	�QF%Yc�A�*

losss�y>(�l�       �	�F%Yc�A�*

loss�ob>A���       �	I�G%Yc�A�*

loss�8�<u�$       �	+H%Yc�A�*

lossW��=�B�       �	ګH%Yc�A�*

loss	��>_km       �	`?I%Yc�A�*

loss�:/<`�!       �	��I%Yc�A�*

lossa�>Y\Hc       �	�iJ%Yc�A�*

loss:��=y��       �	�K%Yc�A�	*

loss�N>+���       �	#�K%Yc�A�	*

loss��u=��U�       �	�8L%Yc�A�	*

lossr�p=͑z�       �	`�L%Yc�A�	*

losst�=(%W�       �	mM%Yc�A�	*

loss���=����       �	`N%Yc�A�	*

loss�8�=,P�       �	��N%Yc�A�	*

loss2ž=Q�-�       �	�;O%Yc�A�	*

loss
�=��       �	_�O%Yc�A�	*

loss�8>gDZ.       �	8jP%Yc�A�	*

lossQ��=�#�       �	� Q%Yc�A�	*

lossHY�=��b�       �	��Q%Yc�A�	*

loss�>Zk�       �	�:R%Yc�A�	*

loss(+>�$v$       �	��R%Yc�A�	*

lossm�0=�d��       �	�sS%Yc�A�	*

lossR8e=�@�       �	�T%Yc�A�	*

loss�%�=�gB       �	T�T%Yc�A�	*

loss'ߢ=��       �	~QU%Yc�A�	*

loss:��<��       �	2�U%Yc�A�	*

lossO�=�N�       �	"�V%Yc�A�	*

loss��=��v       �	wNW%Yc�A�	*

loss�77=��       �	�X%Yc�A�	*

loss*(=%a��       �	d�X%Yc�A�	*

loss���<Fj��       �	�FY%Yc�A�	*

loss�H=�m�       �	q�Y%Yc�A�	*

loss�KJ=�z       �	Q�Z%Yc�A�	*

loss;53>ܭ�b       �	�[%Yc�A�	*

loss�F>J�       �	/\%Yc�A�	*

loss��]=�"۰       �	�\%Yc�A�	*

loss��;=i��0       �	t]%Yc�A�	*

loss;��=/B��       �	w^%Yc�A�	*

lossF5�<�Ā       �	��^%Yc�A�	*

lossju=�~x       �	=E_%Yc�A�	*

loss��4=H��       �	��_%Yc�A�	*

lossc��<2��-       �	{`%Yc�A�	*

loss�e�=�ǗH       �	�a%Yc�A�	*

lossfi�=��8       �	;�a%Yc�A�	*

loss%��=��϶       �	xEb%Yc�A�	*

loss��1=�       �	Y�b%Yc�A�	*

loss<W=Rw��       �	\sc%Yc�A�	*

loss�S�=����       �	~�d%Yc�A�	*

loss�A�=�!y�       �	�?e%Yc�A�	*

lossv|�=�	�       �	�1f%Yc�A�	*

loss�N2=
���       �	��f%Yc�A�	*

loss#��=_@�       �	#�g%Yc�A�	*

loss��v=���       �	Y�h%Yc�A�	*

loss� >�       �	!=i%Yc�A�	*

lossQ�o=���        �	e�i%Yc�A�	*

loss&TH=T�E	       �	sk%Yc�A�	*

loss��=����       �	ob�%Yc�A�	*

loss�#�=�^�Z       �	\�%Yc�A�	*

loss��c>�w�       �	���%Yc�A�	*

loss���=/��-       �	�R�%Yc�A�	*

loss=���       �	��%Yc�A�	*

loss�Dg=� ��       �	���%Yc�A�	*

loss�A=��s�       �	*�%Yc�A�	*

loss��=T
�Q       �	�ˌ%Yc�A�	*

loss
d>��_       �	Mh�%Yc�A�	*

loss_h>W
��       �	qY�%Yc�A�	*

loss��E={�t       �	��%Yc�A�	*

loss(��=�B1       �	���%Yc�A�	*

loss,S=���f       �	�M�%Yc�A�	*

lossB9>�![t       �	`�%Yc�A�	*

loss���=�       �	���%Yc�A�	*

loss��=ɯq5       �	�#�%Yc�A�	*

loss
<H��       �	qʒ%Yc�A�	*

loss�xx=0ђ�       �	�p�%Yc�A�	*

loss���<��߷       �	��%Yc�A�	*

loss'>6��       �	ޮ�%Yc�A�	*

lossh*�=���       �	"Q�%Yc�A�	*

losso-	>���l       �	��%Yc�A�	*

loss93=�xN       �	˂�%Yc�A�	*

loss�G�=׆�       �	�+�%Yc�A�	*

loss��=�KL�       �	PǗ%Yc�A�	*

loss$��<�Np�       �	fi�%Yc�A�	*

loss�|�=��h�       �	9
�%Yc�A�	*

loss��=��_�       �	ʧ�%Yc�A�	*

losse��=�F�        �	�E�%Yc�A�	*

lossWp_>-��\       �	_�%Yc�A�	*

loss��<���}       �	v��%Yc�A�	*

lossf�w=h�       �	o-�%Yc�A�	*

loss��=�Ի       �	mʜ%Yc�A�	*

loss�6>�_�       �	�h�%Yc�A�	*

loss���<z{3[       �	^�%Yc�A�	*

loss�8�=���       �	j��%Yc�A�	*

lossq�<�kP�       �	�_�%Yc�A�	*

losst{�=�C6�       �	��%Yc�A�	*

loss�Q�>��       �	ũ�%Yc�A�	*

loss��>�       �	�D�%Yc�A�	*

lossnt6>��"A       �	�ޡ%Yc�A�	*

loss%�a=��       �	�~�%Yc�A�	*

loss�R1=u0.�       �	��%Yc�A�	*

loss�߼=K�\f       �	��%Yc�A�	*

loss$!�=����       �	���%Yc�A�	*

loss|�>�o       �	m�%Yc�A�	*

loss$�e=�r�       �	���%Yc�A�	*

lossa5d=���       �		N�%Yc�A�	*

loss闗=~�y�       �	C�%Yc�A�	*

loss�\�<͛��       �	V�%Yc�A�	*

loss F�<���T       �	@�%Yc�A�	*

loss��F=<]]�       �	t��%Yc�A�	*

loss��=�~�       �	iQ�%Yc�A�	*

loss���>��       �	b��%Yc�A�	*

loss8�W=���(       �	T��%Yc�A�	*

loss�t<N��       �	k(�%Yc�A�	*

loss�i�;�r       �	Kɫ%Yc�A�	*

loss�A�<���       �	+j�%Yc�A�	*

loss���=v|       �	�%Yc�A�	*

loss�ݖ=��       �	���%Yc�A�	*

loss�|�=#�\       �	@N�%Yc�A�	*

loss#��=�.0;       �	��%Yc�A�	*

lossݑ=��"E       �	���%Yc�A�	*

loss�v�=����       �	�'�%Yc�A�	*

loss3�I= �'�       �	.ǰ%Yc�A�	*

loss]�w=��Gx       �	�j�%Yc�A�	*

loss�;�=y6��       �	��%Yc�A�	*

lossl&?>��l�       �	9��%Yc�A�	*

loss�>7�]�       �	Y3�%Yc�A�	*

loss���=�=�       �	G̳%Yc�A�	*

loss�r=BI5)       �	�m�%Yc�A�	*

loss�{�=Gv�       �	��%Yc�A�	*

loss�p_=�_�       �	H��%Yc�A�	*

loss�7�=N�2=       �	�C�%Yc�A�	*

lossC��=��+�       �	��%Yc�A�	*

loss���<�h��       �	:��%Yc�A�	*

loss���=��Š       �	�/�%Yc�A�	*

loss:= >��C�       �	�̸%Yc�A�	*

losszY=����       �	zl�%Yc�A�	*

lossѵx=H�       �	�%Yc�A�
*

loss�t=$$1]       �	�κ%Yc�A�
*

loss���=�ٞI       �	p��%Yc�A�
*

loss ��=1:{�       �	�Q�%Yc�A�
*

loss���<�$��       �	���%Yc�A�
*

loss��=3���       �	���%Yc�A�
*

loss ȷ<�/��       �	]S�%Yc�A�
*

lossAg =�l��       �	���%Yc�A�
*

loss���=!|��       �	퟿%Yc�A�
*

loss<��=�ŵ       �	�=�%Yc�A�
*

loss�D=p���       �	���%Yc�A�
*

lossJ,�=ۨ�       �	t�%Yc�A�
*

loss!_�=��i�       �	���%Yc�A�
*

loss��}=����       �	E+�%Yc�A�
*

loss�'�=�-��       �	d��%Yc�A�
*

loss���=��s       �	���%Yc�A�
*

losscw�<T��1       �	Q-�%Yc�A�
*

lossݡ�=�I       �	���%Yc�A�
*

lossX�6=-)|�       �	l_�%Yc�A�
*

lossݺ>��ܮ       �	f��%Yc�A�
*

loss�
>���       �	��%Yc�A�
*

loss�6=Ӎȣ       �	�2�%Yc�A�
*

loss��<E�̈       �	���%Yc�A�
*

loss�3�=��       �	�e�%Yc�A�
*

loss�ݗ=[��       �	�%Yc�A�
*

loss�U:=��\�       �	̲�%Yc�A�
*

loss��=��M�       �	�Q�%Yc�A�
*

loss��'<�[�*       �	V��%Yc�A�
*

loss��1=���       �	��%Yc�A�
*

loss�'>坈�       �	�0�%Yc�A�
*

loss�|=F�+O       �	��%Yc�A�
*

lossR��=@'�*       �	���%Yc�A�
*

loss��
=�[�Y       �	�B�%Yc�A�
*

lossA�=v+`�       �	���%Yc�A�
*

lossf=��9       �	�x�%Yc�A�
*

loss.WN=X$��       �	n�%Yc�A�
*

lossV?N=J/       �	F��%Yc�A�
*

loss�~=9&?�       �	�W�%Yc�A�
*

loss��=�H(q       �	7��%Yc�A�
*

loss$Uk=M8�       �	w��%Yc�A�
*

loss�R�=Gm��       �	�L�%Yc�A�
*

lossF�=�H�       �	�#�%Yc�A�
*

loss�#�=���       �	��%Yc�A�
*

lossj�="1&       �	�Y�%Yc�A�
*

loss\�=1r�3       �	���%Yc�A�
*

loss�;�k�       �	���%Yc�A�
*

lossϸ4=�C�       �	�E�%Yc�A�
*

loss�ea=
�p       �	���%Yc�A�
*

loss���< c�       �	v�%Yc�A�
*

loss��#> 3*�       �	��%Yc�A�
*

loss�);=)*)�       �	��%Yc�A�
*

loss�bd=��H�       �	�A�%Yc�A�
*

loss�J�==��       �	���%Yc�A�
*

lossί�=(,       �	���%Yc�A�
*

loss�P=9��       �	]2�%Yc�A�
*

lossD��=�u�       �	T��%Yc�A�
*

loss�+8=^,�       �	5b�%Yc�A�
*

loss�L=�&�       �	7��%Yc�A�
*

loss�=<��       �	��%Yc�A�
*

loss@�=� !�       �	*7�%Yc�A�
*

loss�.=�#h�       �	N��%Yc�A�
*

lossqnD<�_�       �	wg�%Yc�A�
*

loss��<q0��       �	��%Yc�A�
*

loss	z�=�9�       �	��%Yc�A�
*

loss��=tN)       �	.<�%Yc�A�
*

loss���=��{       �	���%Yc�A�
*

lossW=Y]�       �	*��%Yc�A�
*

loss��=B�       �	�s�%Yc�A�
*

loss��;֭��       �	%Z�%Yc�A�
*

loss�|�<�!Xm       �	f��%Yc�A�
*

lossa�8=��cl       �	e��%Yc�A�
*

loss?�<�7�J       �	��%Yc�A�
*

lossi<{=��'^       �	�4�%Yc�A�
*

loss(��=h� 4       �	&4�%Yc�A�
*

lossÂ�=�<�?       �	D��%Yc�A�
*

lossf��=���)       �	ׇ�%Yc�A�
*

loss��<�K�       �	Z-�%Yc�A�
*

loss	F�=q���       �	)��%Yc�A�
*

loss�	>�z(       �	�k�%Yc�A�
*

loss��+>�9�b       �	��%Yc�A�
*

losso4=��k�       �	D��%Yc�A�
*

loss*d�=�8�N       �	G�%Yc�A�
*

loss��9=	Ő       �	m��%Yc�A�
*

loss�"=r       �	��%Yc�A�
*

lossVp�<����       �	��%Yc�A�
*

loss�5=Q���       �	J��%Yc�A�
*

loss|��=m�       �	~��%Yc�A�
*

lossC�$=� �       �	�(�%Yc�A�
*

loss� �=�#��       �	���%Yc�A�
*

loss��=���O       �	%Z�%Yc�A�
*

loss[r=>��       �	���%Yc�A�
*

loss�'�=��>�       �	k��%Yc�A�
*

loss4��<[���       �	v8�%Yc�A�
*

loss��(=�&�e       �	���%Yc�A�
*

loss�85>ztc0       �	Mi�%Yc�A�
*

loss��<l�QM       �	��%Yc�A�
*

loss`��=?eg�       �	��%Yc�A�
*

loss؞�=�G�^       �	�.�%Yc�A�
*

lossaד=V/��       �	��%Yc�A�
*

loss�X1=�xO0       �	��%Yc�A�
*

lossNư<�zY�       �	@N�%Yc�A�
*

loss�g�<�9��       �	O�%Yc�A�
*

loss	��=?�2       �	��%Yc�A�
*

loss��<	���       �	PU�%Yc�A�
*

lossH�7<��4]       �	}��%Yc�A�
*

lossY�	=�t�       �	���%Yc�A�
*

loss�?=(]��       �	S%�%Yc�A�
*

loss�;<=,���       �	��%Yc�A�
*

lossR*=N{�l       �	�T &Yc�A�
*

loss۬S=s1�       �	&� &Yc�A�
*

losst!�=5�S�       �	��&Yc�A�
*

lossڋ>���q       �	4-&Yc�A�
*

lossj> vA       �	T�&Yc�A�
*

loss�H�<2gp�       �	Uh&Yc�A�
*

lossx�=ڑp9       �	m&Yc�A�
*

loss;G%<�5�S       �	��&Yc�A�
*

loss_��=!�D�       �	�3&Yc�A�
*

loss���<��f       �	O�&Yc�A�
*

loss�>v=Z��>       �	Dj&Yc�A�
*

lossH�=��'       �	�
&Yc�A�
*

loss�j�=���       �	ϣ&Yc�A�
*

loss���=[�m�       �	K:&Yc�A�
*

loss�x7=�K�       �	��&Yc�A�
*

loss��p= �w`       �	�z
&Yc�A�
*

lossm�>�V�n       �	s&Yc�A�
*

loss�S�<��l       �	��&Yc�A�
*

lossK�=��Ag       �	�I&Yc�A�
*

loss��p>&�3�       �	��&Yc�A�
*

loss�ɘ=��       �	7�&Yc�A�*

lossJ�=2�Ě       �	�r&Yc�A�*

loss��=��2�       �	&Yc�A�*

losss�=���       �	�&Yc�A�*

lossc��=�%%%       �	0K&Yc�A�*

loss��<�ٿ       �	��&Yc�A�*

loss��C=N�+�       �	qv&Yc�A�*

loss,�=�{�"       �	�&Yc�A�*

lossNq�=E	\K       �	��&Yc�A�*

loss}PG>%�8�       �	�j&Yc�A�*

losst��=,N��       �	�&Yc�A�*

loss�G>��       �	��&Yc�A�*

loss�o�=N��       �	{J&Yc�A�*

loss2�<:�<�       �	��&Yc�A�*

loss;<�=U��g       �	��&Yc�A�*

loss�=9j�       �	�&Yc�A�*

loss`B�=��R       �	B�&Yc�A�*

loss��<.�       �	f&Yc�A�*

lossJ�S=
X�       �	nl&Yc�A�*

loss5	�=g�Y       �	�&Yc�A�*

loss�\�=fax�       �	��&Yc�A�*

loss&=]�e�       �	�1&Yc�A�*

loss���=~G,       �	B
&Yc�A�*

loss=�[�d       �	��&Yc�A�*

loss8��<���       �	�:&Yc�A�*

lossZc�=2��q       �	[�&Yc�A�*

loss�2�<�       �	�}&Yc�A�*

loss���=���       �	�&Yc�A�*

loss��=!��       �	Ω&Yc�A�*

loss!+Z=y�i�       �	�= &Yc�A�*

loss�T:=/V\       �	�� &Yc�A�*

loss���=̓t       �	�x!&Yc�A�*

loss���<Щ�C       �	�"&Yc�A�*

loss-��=d��D       �	 �"&Yc�A�*

loss�(6=�FG       �	�$&Yc�A�*

loss!�=Ӭ       �	��$&Yc�A�*

loss}F=�7{/       �	�d%&Yc�A�*

loss�i�=	!��       �	r�%&Yc�A�*

losst�=��>       �	��&&Yc�A�*

loss�=�c�       �	9)'&Yc�A�*

lossl��=��|'       �	M�'&Yc�A�*

loss� = ���       �	�W(&Yc�A�*

lossl�*=�[cR       �	��(&Yc�A�*

lossi��=�+�       �	��)&Yc�A�*

loss�=���       �	y*&Yc�A�*

loss��=�'       �	)�*&Yc�A�*

loss���<��>{       �	v�+&Yc�A�*

loss��=g���       �	Uj,&Yc�A�*

loss�
=��/A       �	�-&Yc�A�*

loss�j�=t6�6       �	�-&Yc�A�*

loss�X`=��S       �	�E.&Yc�A�*

loss�7y=׍�       �	KX/&Yc�A�*

loss��=�tp�       �	�/&Yc�A�*

lossz��=V�w�       �	�1&Yc�A�*

lossot=ծ�       �	�1&Yc�A�*

loss�>��"�       �	G2&Yc�A�*

loss�a=N��c       �	��2&Yc�A�*

loss$�=�3q       �	:w3&Yc�A�*

loss��<��mP       �	�4&Yc�A�*

loss�ͬ=h׶�       �	m�4&Yc�A�*

loss�i@=yt)�       �	�N5&Yc�A�*

loss���=��^�       �	��5&Yc�A�*

loss�Ɠ=�^       �	�6&Yc�A�*

lossI6 =��N�       �	�?7&Yc�A�*

losst��=�ʕ�       �	��7&Yc�A�*

loss���=3���       �	�8&Yc�A�*

loss-��=��ю       �	�9&Yc�A�*

loss`�>>N�       �	��9&Yc�A�*

loss4>6���       �	�a:&Yc�A�*

lossi�0=��fu       �	D�:&Yc�A�*

loss}�=��T       �	!�;&Yc�A�*

losso��=9+G       �	�m<&Yc�A�*

lossԱ�=�$�U       �	�=&Yc�A�*

loss*��<v�̑       �	�=&Yc�A�*

lossC�=Jj�       �	�W>&Yc�A�*

loss8 >�u�       �	7�>&Yc�A�*

loss|϶=���       �	��?&Yc�A�*

losss߭=D2�}       �	$F@&Yc�A�*

lossM�R=���@       �	\�@&Yc�A�*

lossOm=}~��       �	��A&Yc�A�*

losskו=>a̠       �	�*B&Yc�A�*

loss�N�=�;	�       �	��B&Yc�A�*

loss1�<�^       �	EhC&Yc�A�*

loss۷;d��(       �	�D&Yc�A�*

lossJ>�=���       �	��D&Yc�A�*

loss%<)��       �	�DE&Yc�A�*

loss��<��/i       �	��E&Yc�A�*

loss���<�v-�       �	�F&Yc�A�*

loss�6y=���       �	/2G&Yc�A�*

loss��=�NQY       �	�3H&Yc�A�*

lossC��=�D       �	y�H&Yc�A�*

loss:\->�,�       �	�cI&Yc�A�*

lossѽ0>��~j       �	3lJ&Yc�A�*

loss�C|=2a��       �	K&Yc�A�*

loss�Y=��J�       �	��K&Yc�A�*

loss$%9=���        �	
1L&Yc�A�*

loss�ei=�5       �	{�L&Yc�A�*

lossꌜ=�Ǌ5       �	�oM&Yc�A�*

loss�*:>T8m�       �	�N&Yc�A�*

lossS<���!       �	q�N&Yc�A�*

lossя�=U���       �	�CO&Yc�A�*

loss��<�x|�       �	��O&Yc�A�*

lossv,�=+�G       �	?rP&Yc�A�*

lossS=(3�       �	�?Q&Yc�A�*

loss�´=���       �	��Q&Yc�A�*

loss!k>V�;       �	M�R&Yc�A�*

lossC&=9�"       �	�2S&Yc�A�*

lossi�<�Z>       �	��S&Yc�A�*

loss`�=y�3       �	V}T&Yc�A�*

lossS��=�       �	�U&Yc�A�*

loss���=��5�       �	*�U&Yc�A�*

loss��=���       �	�gV&Yc�A�*

loss2{�=W�W       �	��V&Yc�A�*

loss��i<�ㅈ       �	�W&Yc�A�*

loss�,>��m2       �	�-X&Yc�A�*

loss{,`=��)�       �	��X&Yc�A�*

loss���<�%,        �	LlY&Yc�A�*

loss�wB=�� U       �	�Z&Yc�A�*

loss��=�9D>       �	��Z&Yc�A�*

loss�k+=i��=       �	�F[&Yc�A�*

lossJj=f��       �	��[&Yc�A�*

lossMO=��z       �	\u\&Yc�A�*

lossi�=>r�F�       �	5]&Yc�A�*

loss�<� T       �	��]&Yc�A�*

loss���<�<M       �	K^&Yc�A�*

loss�k�=I[       �	��^&Yc�A�*

loss��=ߜ]       �	�z_&Yc�A�*

lossz�=j,�l       �	`&Yc�A�*

loss��d=�{�       �	M�`&Yc�A�*

lossS1J=0���       �	�9a&Yc�A�*

lossך(=	;;�       �	�a&Yc�A�*

loss���=�j       �	l{b&Yc�A�*

lossٛ#<N�c       �	�c&Yc�A�*

loss�K�<P,��       �	ȶc&Yc�A�*

loss�n�<��ۖ       �	eQd&Yc�A�*

loss�z�=PE�       �	"e&Yc�A�*

loss��o=0EN�       �	��e&Yc�A�*

loss�U�>���       �	�
g&Yc�A�*

loss��>��@X       �	��g&Yc�A�*

loss��!=����       �	��h&Yc�A�*

loss(��=�g�       �	��i&Yc�A�*

loss��=c�-       �	nQj&Yc�A�*

loss۟�=�t       �	��j&Yc�A�*

loss�2�<���N       �	�l&Yc�A�*

loss}�H=,�W       �	��l&Yc�A�*

loss*ǧ=��L]       �	M�m&Yc�A�*

loss��8=���       �	% n&Yc�A�*

lossE7�=��eU       �	ePo&Yc�A�*

lossw�<�O�       �	�p&Yc�A�*

loss;�<�� 7       �	�Vq&Yc�A�*

loss���=`��       �	p�r&Yc�A�*

loss�*C=��U       �	��s&Yc�A�*

lossv�<)�       �	�t&Yc�A�*

loss|Zt=�Ch�       �	u&Yc�A�*

loss� =Y=R�       �	�u&Yc�A�*

loss��=A�Q�       �	K\v&Yc�A�*

loss�\>�+�Q       �	@�v&Yc�A�*

lossX��=����       �	%�w&Yc�A�*

lossߋ=e�D�       �	g~x&Yc�A�*

loss�ݾ=	ޗM       �	 y&Yc�A�*

loss�%�=S'�}       �	)�y&Yc�A�*

loss�K�<���t       �	�wz&Yc�A�*

loss.��=�i��       �	k{&Yc�A�*

loss��n=�!�       �	@�{&Yc�A�*

loss<A�=��/       �	!X|&Yc�A�*

lossΑ="i�        �	��|&Yc�A�*

loss��P<��{1       �	8�}&Yc�A�*

loss�I=k�۹       �	a~&Yc�A�*

loss�!�=d8@�       �	~�~&Yc�A�*

loss*�=���       �	��&Yc�A�*

loss�i=��
       �	k)�&Yc�A�*

loss��=T�       �	�4�&Yc�A�*

loss� �=9�A�       �	���&Yc�A�*

lossla�=u[��       �	ꗂ&Yc�A�*

loss�[�=��       �	Q.�&Yc�A�*

loss� > ��       �	P&Yc�A�*

loss0��=>�E�       �	�X�&Yc�A�*

losso+	<m�tR       �	g�&Yc�A�*

loss�L>BD|       �	.��&Yc�A�*

loss���=�Fb�       �	I*�&Yc�A�*

loss� >?��9       �	vƆ&Yc�A�*

loss%��<��c�       �	�_�&Yc�A�*

loss�L=��U;       �	bI�&Yc�A�*

lossm�9=D���       �	��&Yc�A�*

loss@j�=�       �	���&Yc�A�*

loss�ئ=���y       �	L�&Yc�A�*

loss��=]���       �	:��&Yc�A�*

loss��=����       �	oF�&Yc�A�*

loss6K>9�C       �	Yߋ&Yc�A�*

loss�H�;�r��       �	J{�&Yc�A�*

loss��<�Gv�       �	��&Yc�A�*

loss�X�<8�\�       �	h��&Yc�A�*

lossq�=�-�_       �	,G�&Yc�A�*

loss�.u=��%�       �	�ݎ&Yc�A�*

lossx>^=z@�       �	
��&Yc�A�*

loss�T>���       �	�#�&Yc�A�*

loss�m�<�xv       �	�Đ&Yc�A�*

loss�WZ=���       �	�^�&Yc�A�*

lossh~N=�˨0       �	�	�&Yc�A�*

loss*3 >ั       �	흒&Yc�A�*

loss��=��J)       �	8�&Yc�A�*

loss[b�=�q       �	�ړ&Yc�A�*

losse��=�L6�       �	�{�&Yc�A�*

loss�|>�ڹ=       �	<�&Yc�A�*

lossɎ�=�`�       �	���&Yc�A�*

lossd��<{وn       �	C�&Yc�A�*

loss��=G���       �	�ۖ&Yc�A�*

lossp�=���       �	6s�&Yc�A�*

loss��>�C4`       �	��&Yc�A�*

loss�K=խ�w       �	G��&Yc�A�*

lossz5�==�P�       �	NE�&Yc�A�*

loss߹I=��5       �	�&Yc�A�*

loss� >�0�,       �	���&Yc�A�*

loss���<��3�       �	��&Yc�A�*

loss�^�=Il�(       �	2��&Yc�A�*

loss���<U��]       �	�G�&Yc�A�*

loss�2�=�Ф       �	�ڜ&Yc�A�*

loss㭘= h�       �	�m�&Yc�A�*

lossm><���       �	��&Yc�A�*

loss旗=K!�Y       �	��&Yc�A�*

loss
>4>�Ϳ�       �	B?�&Yc�A�*

loss]�=A�w�       �	�G�&Yc�A�*

loss\<�=�JQ�       �	�۠&Yc�A�*

losso!C=�?�       �	���&Yc�A�*

lossX�=�|b^       �	�%�&Yc�A�*

loss�
�<� �h       �	Q��&Yc�A�*

lossEq?=۪�&       �	�V�&Yc�A�*

loss��[=&��9       �	d�&Yc�A�*

loss�%�=��w       �	�~�&Yc�A�*

loss�j�<%$��       �	�"�&Yc�A�*

loss<t=6��       �	]��&Yc�A�*

losss�c<Ip��       �	��&Yc�A�*

loss��<\�)Q       �	�T�&Yc�A�*

lossH>%e�       �	��&Yc�A�*

loss�j�> o��       �	���&Yc�A�*

lossa8=�@̌       �	�w�&Yc�A�*

loss��%>���       �	��&Yc�A�*

loss\i�<U0)�       �	���&Yc�A�*

loss���=�n�       �	YM�&Yc�A�*

lossa>�W�       �	���&Yc�A�*

loss���<}"�;       �	��&Yc�A�*

loss=�<���       �	CY�&Yc�A�*

loss��>�,       �	Z�&Yc�A�*

loss���=�e]�       �	͓�&Yc�A�*

loss�K<�C�V       �	M,�&Yc�A�*

loss�P�<���e       �	���&Yc�A�*

loss?>�R       �	?��&Yc�A�*

loss�Q>O�       �	B�&Yc�A�*

loss�'�=�W"       �	Rײ&Yc�A�*

loss}��=����       �	�l�&Yc�A�*

lossފ�=���*       �	�&Yc�A�*

loss�7i=M)Ŧ       �	���&Yc�A�*

loss$��=�bZd       �		2�&Yc�A�*

lossU|=��J�       �	eƵ&Yc�A�*

loss��=,?M�       �	:Z�&Yc�A�*

lossᕲ<EQD       �	��&Yc�A�*

lossm��=�       �	ꑷ&Yc�A�*

loss��W=���o       �	x%�&Yc�A�*

loss�=c�B)       �	��&Yc�A�*

loss\K�<c�w�       �	�U�&Yc�A�*

loss�=#: �       �	F�&Yc�A�*

loss�x�=�P/       �	f��&Yc�A�*

loss���<�z       �	 �&Yc�A�*

loss{G
>�y:�       �	��&Yc�A�*

lossv*c=�W�       �	3Q�&Yc�A�*

loss;q=m"��       �	��&Yc�A�*

loss��=K�       �	-y�&Yc�A�*

loss}�=5�f�       �	'�&Yc�A�*

loss�	l=��[�       �	W��&Yc�A�*

loss��=��H�       �	xD�&Yc�A�*

lossO�=f�E�       �	+ڿ&Yc�A�*

loss�7<�w��       �	�o�&Yc�A�*

loss�M!>r�       �	��&Yc�A�*

loss�>�М       �	̲�&Yc�A�*

lossC��=@'vf       �	�H�&Yc�A�*

loss�>�=`vl�       �	���&Yc�A�*

loss�[>I�=�       �	Bw�&Yc�A�*

lossy-�=m���       �	$�&Yc�A�*

loss�2�<qG	�       �	���&Yc�A�*

loss-)=@�)�       �	���&Yc�A�*

loss��=��U'       �	�,�&Yc�A�*

lossᷜ<9�P       �	a��&Yc�A�*

loss�,=�̣       �	g�&Yc�A�*

loss6�=�K�:       �	 ��&Yc�A�*

losswDX=��֊       �	ɒ�&Yc�A�*

loss��<���       �	�$�&Yc�A�*

loss�QY=D?/�       �	��&Yc�A�*

loss��<Q<�       �	o�&Yc�A�*

loss3_
=rP�*       �	��&Yc�A�*

loss�f�=?��U       �	��&Yc�A�*

loss�a><J       �	�H�&Yc�A�*

loss��=��       �	���&Yc�A�*

lossԉ�=ƴ	C       �	Bz�&Yc�A�*

loss�=�:       �	��&Yc�A�*

loss�["=x\�        �	<��&Yc�A�*

loss�X0=�@{       �	�<�&Yc�A�*

loss��=��ڻ       �	���&Yc�A�*

loss�t>���       �	�j�&Yc�A�*

loss\��<���f       �	��&Yc�A�*

loss�a�=�(.�       �	��&Yc�A�*

loss=�=��       �	�K�&Yc�A�*

lossf��=��#�       �	���&Yc�A�*

loss#��=e��       �	�z�&Yc�A�*

loss�P�<l^�1       �	��&Yc�A�*

lossno$={�B�       �	��&Yc�A�*

lossa\3<G��       �	%[�&Yc�A�*

loss�d�=.�[�       �	+��&Yc�A�*

lossd!^=��Ѹ       �	q��&Yc�A�*

lossc�<=k���       �	(�&Yc�A�*

lossn�d=9e��       �	���&Yc�A�*

lossh��<���       �	�b�&Yc�A�*

lossj�=����       �	��&Yc�A�*

loss�$�=��k       �	���&Yc�A�*

losss@<]�       �	=�&Yc�A�*

loss��=���       �	,��&Yc�A�*

loss:P�;}e�       �	�k�&Yc�A�*

loss=j7��       �	�&Yc�A�*

loss)6Z=e���       �	J��&Yc�A�*

loss�?]<�<�       �	�4�&Yc�A�*

loss/�U=�P�       �	���&Yc�A�*

loss8�=��ش       �	Fa�&Yc�A�*

loss>�ZDJ       �	���&Yc�A�*

loss�R�<���       �	���&Yc�A�*

loss��=u�F�       �	�h�&Yc�A�*

loss�8L=�ڝS       �	��&Yc�A�*

loss���<�A
�       �	e��&Yc�A�*

loss��:<M���       �	�F�&Yc�A�*

loss<�<�k@       �	��&Yc�A�*

loss_ߪ<
��       �	ׇ�&Yc�A�*

loss)f=��:       �	ɮ�&Yc�A�*

loss}=r6ur       �	�L�&Yc�A�*

lossᾲ<��R>       �	d��&Yc�A�*

loss��_=cVY       �	�Y�&Yc�A�*

lossJ�<���?       �	r7�&Yc�A�*

loss�p;��8K       �	s��&Yc�A�*

loss��%;��       �	.��&Yc�A�*

loss	y�<�"z       �	΋�&Yc�A�*

loss|�|=����       �	�j�&Yc�A�*

loss�_�<61�       �	�'�&Yc�A�*

loss��:*�Y�       �	^��&Yc�A�*

loss�à<�p       �	W�&Yc�A�*

lossw:�>��F       �	���&Yc�A�*

loss��;{�9*       �	��&Yc�A�*

loss�W�>P�+6       �	�~�&Yc�A�*

loss��=��q       �	tD�&Yc�A�*

lossR��=�ڥ       �	�G�&Yc�A�*

loss_�u=�܇�       �	�3�&Yc�A�*

loss[t�<��d       �	��&Yc�A�*

loss�8V=����       �	���&Yc�A�*

loss�6�=��s�       �	}=�&Yc�A�*

loss��=�G3�       �	��&Yc�A�*

loss\��=�?�       �	`��&Yc�A�*

loss� =�-�d       �	���&Yc�A�*

loss��>�u��       �	�k�&Yc�A�*

loss課=>ؚ        �	�?�&Yc�A�*

loss\Pl= �6�       �	���&Yc�A�*

lossq�=���U       �	.��&Yc�A�*

lossj� >���v       �	^0�&Yc�A�*

loss��w=��Ŋ       �	~6�&Yc�A�*

loss�=Y�R       �	���&Yc�A�*

loss석=����       �	 ��&Yc�A�*

loss$�t=�ՙ       �	�g�&Yc�A�*

loss���<�U\g       �	��&Yc�A�*

loss�TQ=HNI�       �	r��&Yc�A�*

lossܠ�<���N       �	\ 'Yc�A�*

lossV.
=�}:       �	j'Yc�A�*

loss[@'=��+�       �	L�'Yc�A�*

loss�~j<��       �	<�'Yc�A�*

lossq�O=,J4       �	Zb'Yc�A�*

loss�,�<�Xo�       �	��'Yc�A�*

loss��=&b߬       �	IH'Yc�A�*

loss�� >�C��       �	��'Yc�A�*

lossm��<1�4       �	��'Yc�A�*

loss��<y$��       �	�x'Yc�A�*

loss�Qi=E�S       �	�G'Yc�A�*

loss���;fm6       �	�(	'Yc�A�*

loss�.=3'<�       �	\�	'Yc�A�*

lossLI"=��}�       �	��
'Yc�A�*

loss4�I<;n��       �	�9'Yc�A�*

loss;Ϊ='R��       �	�'Yc�A�*

loss��=�&~�       �	[&'Yc�A�*

loss��=���       �	�X'Yc�A�*

lossE��<*��       �	�'Yc�A�*

lossjOY<�?��       �	C�'Yc�A�*

loss��C<��       �	܁'Yc�A�*

loss�=���       �	��'Yc�A�*

loss�y.=��%�       �	z�'Yc�A�*

loss6&7=�"�U       �	M�'Yc�A�*

loss#�=���       �	�'Yc�A�*

lossE��<��S       �	��'Yc�A�*

loss���=��f�       �	��'Yc�A�*

losss�&=��ˊ       �	nR'Yc�A�*

loss|�<ݢ�>       �	l�'Yc�A�*

loss_i�=�,�^       �	��0'Yc�A�*

loss�(�=��W|       �	�U1'Yc�A�*

loss8^,>��R       �	J�1'Yc�A�*

loss,��=�ɠ       �	r�2'Yc�A�*

loss�=�<"�@       �	3'Yc�A�*

loss<d+�       �	��3'Yc�A�*

lossm>=�H(A       �	L4'Yc�A�*

loss�@s=L��       �	��4'Yc�A�*

lossn��=U��	       �	w�5'Yc�A�*

loss�>?���       �	�6'Yc�A�*

lossr��<\9=�       �	F�6'Yc�A�*

lossa�W=�L�       �	fK7'Yc�A�*

loss��>���       �	n�7'Yc�A�*

loss��;=hD�y       �	Ku8'Yc�A�*

lossB��=^�@       �	�
9'Yc�A�*

loss܉&>��       �	�9'Yc�A�*

lossje-=�//�       �	sH:'Yc�A�*

loss�<�\�       �	�;'Yc�A�*

lossĆ=^��R       �	�;'Yc�A�*

loss@�:>���:       �	NA<'Yc�A�*

loss#� =���/       �	��<'Yc�A�*

loss��=r�       �	�u='Yc�A�*

loss=�=S���       �	|)>'Yc�A�*

loss./�=��Ir       �	C�>'Yc�A�*

lossz�4=۽q`       �	�A'Yc�A�*

loss�S=뇁p       �	2�A'Yc�A�*

loss}�!=J��       �	��B'Yc�A�*

loss��<^ b�       �	��D'Yc�A�*

lossROL=K��D       �	�LE'Yc�A�*

lossO��=�D�       �	��E'Yc�A�*

loss�ߠ=�d       �	��F'Yc�A�*

lossi��<r�m�       �	�.G'Yc�A�*

loss}�N=}n�       �	��G'Yc�A�*

lossIv2>&V_       �	�lH'Yc�A�*

lossF�<=�:       �	dI'Yc�A�*

loss��!=���q       �	�I'Yc�A�*

loss+t=�Љ�       �	t@J'Yc�A�*

loss�Ӂ= L       �	��J'Yc�A�*

loss�Q8>C��+       �	�iK'Yc�A�*

lossi�#>�ww       �	�%L'Yc�A�*

loss�n�=�,�f       �	�L'Yc�A�*

lossns=���       �	�SM'Yc�A�*

lossS��<4�S?       �	��M'Yc�A�*

loss{Vb=��Y       �	�N'Yc�A�*

loss�r4=)v·       �	�UO'Yc�A�*

loss��=��q�       �	��O'Yc�A�*

loss���<��P�       �	��P'Yc�A�*

lossA�k=J�`�       �	
0Q'Yc�A�*

loss�E=xچ�       �	�Q'Yc�A�*

loss(o�<i��+       �	�R'Yc�A�*

lossR0F=���       �	�5S'Yc�A�*

lossLr=� m       �	��S'Yc�A�*

loss��<k��|       �	DlT'Yc�A�*

loss�;6>�w�       �	�U'Yc�A�*

loss.��<$��g       �	<�U'Yc�A�*

loss2�}<xm�       �	P8V'Yc�A�*

loss��<�J,
       �	>�V'Yc�A�*

loss�)< ؛J       �	�cW'Yc�A�*

loss-5�=KV�l       �	��W'Yc�A�*

lossQ�>=o��       �	�X'Yc�A�*

loss%�o>g`�       �	(-Y'Yc�A�*

lossݬ=�,��       �	 �Y'Yc�A�*

loss!8�<ӱ��       �	�VZ'Yc�A�*

loss��=���       �	��Z'Yc�A�*

loss��<z<n�       �	�['Yc�A�*

lossΚg=ԁ��       �	�\'Yc�A�*

loss�==ɌRZ       �	I�\'Yc�A�*

loss{�=� ��       �	�T]'Yc�A�*

lossWj�=����       �	�^'Yc�A�*

loss��=[�fQ       �	k�^'Yc�A�*

losssX=d���       �	JB_'Yc�A�*

loss�֬=s���       �	��_'Yc�A�*

loss�==�՝�       �	�`'Yc�A�*

loss*�=x�f�       �	�a'Yc�A�*

loss�fZ=Zu8�       �	M�a'Yc�A�*

loss�
6<�t"�       �	�Pb'Yc�A�*

loss�>�<���       �	W�b'Yc�A�*

loss��=K��       �	��c'Yc�A�*

loss�q<<�{�T       �	d'Yc�A�*

loss~��=d2�]       �	W�d'Yc�A�*

loss7۪=G�.�       �	�Te'Yc�A�*

loss�"�=@���       �	mf'Yc�A�*

loss3�=+!�       �	�g'Yc�A�*

loss6D�<�v�$       �	E�g'Yc�A�*

lossi�=�V9�       �	AEh'Yc�A�*

lossWX�<�RnG       �	Y�h'Yc�A�*

loss.iU<��D       �	�i'Yc�A�*

loss�C�<��iA       �	��j'Yc�A�*

loss$��<�|��       �	__k'Yc�A�*

loss�8=^G<�       �	�l'Yc�A�*

lossO{�=;U�{       �	T�l'Yc�A�*

loss���=�
J�       �	�Jm'Yc�A�*

loss��b=�92       �	׉n'Yc�A�*

loss�h�< \��       �	7o'Yc�A�*

loss?>�j�       �	��o'Yc�A�*

loss/��<-��       �	��p'Yc�A�*

lossR=5=�Q8�       �	�0q'Yc�A�*

loss��<�Ҡ�       �	t�q'Yc�A�*

loss�#�=x���       �	��r'Yc�A�*

lossR�q=��       �	P7s'Yc�A�*

loss$��<2�b\       �	�s'Yc�A�*

loss��<%�*<       �	zpt'Yc�A�*

loss��=R0��       �	�Ku'Yc�A�*

loss j�=��o        �	K�u'Yc�A�*

loss%�$=�dQ�       �	��v'Yc�A�*

loss���=?\       �	8-w'Yc�A�*

loss�Ru=�u       �	_�w'Yc�A�*

lossҐK=��KV       �	�mx'Yc�A�*

loss���=��:       �	y'Yc�A�*

loss��=W8c�       �	�y'Yc�A�*

lossHL>���G       �		�z'Yc�A�*

lossj
=�%V�       �	�e{'Yc�A�*

loss���<}ހ�       �	. |'Yc�A�*

loss�L�<i;�       �	��|'Yc�A�*

lossl�=ͅ�s       �	�f}'Yc�A�*

loss���;��       �	�~'Yc�A�*

loss�`=k!       �	Ǜ~'Yc�A�*

losslkT=��s�       �	2>'Yc�A�*

loss�͵=��4�       �	�'Yc�A�*

loss!�=j}       �	l�'Yc�A�*

loss�2�=9�=�       �	m�'Yc�A�*

loss9�<��3       �	���'Yc�A�*

lossD�|=��o�       �	���'Yc�A�*

lossHqp=�_�       �	�H�'Yc�A�*

loss.��<�a�Y       �	W�'Yc�A�*

loss،=�t,�       �	�4�'Yc�A�*

loss�[�<�c[�       �	��'Yc�A�*

lossrPc=��
�       �	���'Yc�A�*

loss���=O�[�       �	x)�'Yc�A�*

loss�;i=����       �	. �'Yc�A�*

loss�[R=�_?�       �	���'Yc�A�*

loss�p�=ܺ��       �	i��'Yc�A�*

lossMCA=��0*       �	*s�'Yc�A�*

loss[��<
(�       �	��'Yc�A�*

loss�֔=>I��       �	���'Yc�A�*

lossE�)=�D�:       �	�O�'Yc�A�*

losslS�=���\       �	-�'Yc�A�*

loss@ �<,x�       �	I��'Yc�A�*

loss!�;��       �	���'Yc�A�*

loss$	�<L!#2       �	tD�'Yc�A�*

loss���:QT��       �	.�'Yc�A�*

lossN��<�o�b       �	l��'Yc�A�*

loss�#�=�܃�       �	�K�'Yc�A�*

lossm��<&��t       �	��'Yc�A�*

loss��<E�L       �	沒'Yc�A�*

loss|��<��F�       �	K�'Yc�A�*

lossE2�=&W��       �	G�'Yc�A�*

loss@ͼ;!xb�       �	�x�'Yc�A�*

loss�<'��       �	H�'Yc�A�*

lossC�_=\�       �	Ვ'Yc�A�*

loss��;�_�1       �	Ln�'Yc�A�*

loss_�=��y       �	��'Yc�A�*

loss��:<C���       �	ı�'Yc�A�*

losst��=�$�"       �	�F�'Yc�A�*

loss��e=����       �	��'Yc�A�*

loss��(<�^_�       �	߉�'Yc�A�*

loss7{<4I�F       �	�_�'Yc�A�*

loss0�>���5       �	��'Yc�A�*

loss�
>����       �	�ě'Yc�A�*

loss WT=�eN�       �	�[�'Yc�A�*

loss'�=���       �	�'Yc�A�*

loss\�=� "�       �	=��'Yc�A�*

lossn�s=��G�       �	IJ�'Yc�A�*

lossű<��<       �	���'Yc�A�*

loss�B7=�yǎ       �	!w�'Yc�A�*

lossaP=��"_       �	�	�'Yc�A�*

losss$<,]��       �	���'Yc�A�*

losszC=tՊ�       �	G>�'Yc�A�*

loss��5=���       �	�'Yc�A�*

loss�$T=���V       �	ʈ�'Yc�A�*

loss�q�=RNtf       �	�)�'Yc�A�*

loss�h=��ҵ       �	���'Yc�A�*

loss�WC=���f       �	�]�'Yc�A�*

loss�k�=\�       �	���'Yc�A�*

loss-�b;L��~       �	V�'Yc�A�*

loss]�.=y�c       �	X��'Yc�A�*

loss)�	>���       �	�,�'Yc�A�*

losst��=��       �	�ǧ'Yc�A�*

loss�Bp<��B       �	Zb�'Yc�A�*

loss)n<);��       �	O�'Yc�A�*

loss!�<#u��       �	W��'Yc�A�*

loss(�=z�oX       �	�-�'Yc�A�*

loss!��=ΊD�       �	̪'Yc�A�*

loss�L<ԛ2�       �	�`�'Yc�A�*

lossF><�rT       �	r��'Yc�A�*

loss�e*=S��F       �	�'Yc�A�*

loss��=٭�%       �	R&�'Yc�A�*

loss�� =d��:       �	/ĭ'Yc�A�*

loss��<,��       �	�]�'Yc�A�*

losse�=�c��       �	���'Yc�A�*

loss�W�=!q��       �	'Yc�A�*

losss��<�C�       �	�/�'Yc�A�*

loss �=<e��       �	K̰'Yc�A�*

lossߙ�<���_       �	i�'Yc�A�*

lossE��<�ul       �	��'Yc�A�*

loss4@t<�[Q       �	˞�'Yc�A�*

lossFe�=z@�       �	L6�'Yc�A�*

loss5==Z �       �	,ӳ'Yc�A�*

loss:�=��R�       �	�k�'Yc�A�*

loss/�	=�?�k       �	�
�'Yc�A�*

loss�I=��@       �	���'Yc�A�*

lossvP�<��"�       �	�>�'Yc�A�*

loss���=���       �	�Զ'Yc�A�*

loss�J�=!�(�       �	u�'Yc�A�*

lossI�<"e3       �	"�'Yc�A�*

lossiY4=ܞ#J       �	�¸'Yc�A�*

loss�Ί=�bi       �	�`�'Yc�A�*

lossf��=E�O�       �	���'Yc�A�*

lossd?�<	�o=       �	��'Yc�A�*

loss3��<֫D?       �	B�'Yc�A�*

loss�Q�<��P�       �	ڻ'Yc�A�*

loss�n�<�GΊ       �	2s�'Yc�A�*

loss!>�<x�E       �	u�'Yc�A�*

lossH��<��o�       �	TŽ'Yc�A�*

loss�o�=] �P       �	�c�'Yc�A�*

loss���=�2>       �	��'Yc�A�*

loss�ߵ=��f1       �	���'Yc�A�*

loss!�P>
��       �	rQ�'Yc�A�*

lossiE1>+�U       �	(��'Yc�A�*

loss�¤=�QJ�       �	���'Yc�A�*

loss&º<V�
       �	V,�'Yc�A�*

loss��n=@mɱ       �	 ��'Yc�A�*

loss��Q=�Ua�       �	�a�'Yc�A�*

loss�V\=�K�j       �	��'Yc�A�*

loss���<���       �	+n�'Yc�A�*

loss.�=9       �	��'Yc�A�*

loss�;�=�y=       �	��'Yc�A�*

loss�܍=CZS       �	EG�'Yc�A�*

loss/8=��q)       �	)��'Yc�A�*

lossi �=	���       �	��'Yc�A�*

losss��<
��       �	,�'Yc�A�*

lossC�<�@��       �	O��'Yc�A�*

losst��=��]       �	Tp�'Yc�A�*

loss%�<�w��       �	��'Yc�A�*

loss�!>�#ۭ       �	���'Yc�A�*

loss�9=�s       �	�_�'Yc�A�*

loss�XT=�x       �	��'Yc�A�*

losss��<ߤ�       �	���'Yc�A�*

loss,�U=	sq       �	�N�'Yc�A�*

loss��<P;UU       �	F��'Yc�A�*

loss�-=Wk�?       �	ސ�'Yc�A�*

lossCx
=|��       �	�,�'Yc�A�*

lossۄ=D��       �	���'Yc�A�*

loss�,|=���j       �	�b�'Yc�A�*

loss&�=���       �	X��'Yc�A�*

lossy^�=V��:       �	��'Yc�A�*

lossIWB=l��       �	R��'Yc�A�*

lossfM<:��        �	�'Yc�A�*

loss��=g9,R       �	���'Yc�A�*

lossE�`=8��h       �	�e�'Yc�A�*

loss`r>q��`       �		�'Yc�A�*

loss���=�)       �	/��'Yc�A�*

loss�7=�8�g       �	�=�'Yc�A�*

loss�<L�N�       �	���'Yc�A�*

loss��< ���       �	���'Yc�A�*

loss�A,<��dG       �	G�'Yc�A�*

lossꖓ=2��A       �	J��'Yc�A�*

lossZ�?=A=kd       �	���'Yc�A�*

lossi�H=V�       �	WB�'Yc�A�*

loss�Ez=����       �	���'Yc�A�*

loss��=��Y�       �	�~�'Yc�A�*

loss4��<�mY�       �	-!�'Yc�A�*

loss�k#=���:       �	,��'Yc�A�*

loss�:�=���B       �	XV�'Yc�A�*

loss���=����       �	���'Yc�A�*

lossL��<&�ѻ       �	*��'Yc�A�*

loss�&�=S��       �	I��'Yc�A�*

lossZ�0=��|�       �	�D�'Yc�A�*

loss�i�=\�I       �	0��'Yc�A�*

loss׋�=�lN�       �	���'Yc�A�*

lossqu�<�:}       �	�I�'Yc�A�*

loss,�=���       �	��'Yc�A�*

loss�.=�>��       �	h��'Yc�A�*

loss��<����       �	�K�'Yc�A�*

loss�X>D���       �	��'Yc�A�*

lossA�=¹�       �	��'Yc�A�*

lossZ�<$��       �	���'Yc�A�*

loss��	<�        �	0��'Yc�A�*

loss��=B���       �	K;�'Yc�A�*

losst�|=�h��       �	��'Yc�A�*

loss��< ś       �	���'Yc�A�*

loss\�<B)ҭ       �	d��'Yc�A�*

lossJ��=m�t�       �	���'Yc�A�*

lossf�<��G       �	��'Yc�A�*

lossẔ=D�x0       �	!�'Yc�A�*

loss��=`>�X       �	���'Yc�A�*

loss�X=�y��       �	Ve�'Yc�A�*

lossq��<E:V�       �	��'Yc�A�*

lossk3=榹`       �	�h�'Yc�A�*

loss�^�<���^       �	A�'Yc�A�*

loss,X)<����       �	��'Yc�A�*

loss!�=X��Z       �	vR�'Yc�A�*

lossQ7�<�ZK       �	H��'Yc�A�*

loss���<aj�       �	��'Yc�A�*

loss��a=�E�       �	oE�'Yc�A�*

loss '�=jY�       �	y��'Yc�A�*

loss�<�R��       �	��'Yc�A�*

losso.�=Z?̛       �	�2�'Yc�A�*

loss���=��>�       �	W��'Yc�A�*

loss/��=�vu       �	Z�'Yc�A�*

loss��=I���       �	d�'Yc�A�*

lossf�<�"%�       �	���'Yc�A�*

loss)�h<A'�z       �	�g�'Yc�A�*

loss��=l=�       �	��'Yc�A�*

lossܯ+=H &�       �	�0�'Yc�A�*

lossA�,>/        �	���'Yc�A�*

lossP�;� �]       �	�e�'Yc�A�*

lossR��=���       �	.�'Yc�A�*

loss�{=����       �	)��'Yc�A�*

loss���=�K�       �	�P (Yc�A�*

loss��f=�ʵ       �	�� (Yc�A�*

loss��=��6p       �	��(Yc�A�*

loss�<=�¯
       �	lC(Yc�A�*

loss�<q��       �	��(Yc�A�*

loss�'1<J�k2       �	��(Yc�A�*

lossz��=���0       �	!(Yc�A�*

loss܁j=��       �	#�(Yc�A�*

loss��<w���       �	�\(Yc�A�*

loss(�=�a�       �	@�(Yc�A�*

loss��=��0d       �	}�(Yc�A�*

loss��=�;��       �	�[(Yc�A�*

loss;�=�B��       �	n�(Yc�A�*

loss��=H{       �	,�(Yc�A�*

loss��'=dE�       �	6=	(Yc�A�*

loss��#=�J�       �	��	(Yc�A�*

lossV]b=��+       �	�v
(Yc�A�*

loss���<Y�Fp       �	Q(Yc�A�*

loss��=R�l�       �	��(Yc�A�*

loss��=�+��       �	�?(Yc�A�*

loss��3>
��       �	��(Yc�A�*

loss�C�<����       �	+k(Yc�A�*

loss�kP<�%�]       �	�(Yc�A�*

loss	��<�5��       �	A�(Yc�A�*

loss�B=�0Xt       �	�(Yc�A�*

loss��=w�l       �	�5(Yc�A�*

loss�w]=#��#       �	*�(Yc�A�*

loss3�.=h<p       �	8�(Yc�A�*

loss毹=�_�c       �	�"(Yc�A�*

loss��H=Ĥ<1       �	��(Yc�A�*

loss�4 <~}m�       �	bj(Yc�A�*

loss�"u<U|ޙ       �	t(Yc�A�*

loss?]=��<�       �	p�(Yc�A�*

loss�!�=��       �	S(Yc�A�*

loss�F=����       �	��(Yc�A�*

loss�=���F       �	��(Yc�A�*

loss�C�=�F�       �	i:(Yc�A�*

losslߎ<FpT|       �	��(Yc�A�*

loss�_�=��O       �	�(Yc�A�*

lossL��=3�y�       �	9((Yc�A�*

loss�D�=�0��       �	h�(Yc�A�*

lossr4j=/�?&       �	�f(Yc�A�*

loss#J
=�)i}       �	7 (Yc�A�*

loss�?q=a�%{       �	5�(Yc�A�*

lossb<�8s       �	r5(Yc�A�*

losse�=_��o       �	��(Yc�A�*

lossĽ-<�&qq       �	�l(Yc�A�*

loss��<mR�7       �	W(Yc�A�*

loss�D�=T�x       �	��(Yc�A�*

loss-�=�GC�       �	JE (Yc�A�*

loss;V=�y��       �		!(Yc�A�*

loss��=H��       �	D�!(Yc�A�*

loss�_:=�ĊO       �	�@"(Yc�A�*

lossI(x<��0       �	/�"(Yc�A�*

loss/�>@B@       �	}y#(Yc�A�*

loss��Q=����       �	�$(Yc�A�*

loss���<Bq3       �	��$(Yc�A�*

loss�s�=3��       �	�^%(Yc�A�*

loss4^,=�~�u       �	��%(Yc�A�*

loss�2�<�{��       �	��&(Yc�A�*

loss}(3=�~'C       �	�K'(Yc�A�*

loss��~=�KK�       �	�J((Yc�A�*

losszo-<B�A       �	�)(Yc�A�*

loss�|�<�D4       �	�3*(Yc�A�*

loss�<�<>���       �	/�*(Yc�A�*

loss�H�=���       �	[�+(Yc�A�*

loss�@=��d       �	*�,(Yc�A�*

loss��=��I       �	40-(Yc�A�*

loss���<_/��       �	��-(Yc�A�*

lossiA=���y       �	�f.(Yc�A�*

loss��>5�}       �	�)/(Yc�A�*

loss#�=>W�+       �	�/(Yc�A�*

losse�:=VP]�       �	l[0(Yc�A�*

loss�M�=�	�       �	��0(Yc�A�*

loss�*=?�5�       �	�1(Yc�A�*

loss���;1�1       �	�B2(Yc�A�*

lossE��=�r�       �	�83(Yc�A�*

loss ZF=[��       �	_4(Yc�A�*

loss;	�=�
?       �	
�4(Yc�A�*

loss��<&?       �	�=5(Yc�A�*

loss�S�<5˝       �	d�5(Yc�A�*

lossr��=	3       �	��6(Yc�A�*

loss?��=�eV       �	J}7(Yc�A�*

loss!#=E��r       �	�8(Yc�A�*

loss��W=�'��       �	!�8(Yc�A�*

loss�
�<�F       �	4K9(Yc�A�*

loss��E>�g�X       �	��9(Yc�A�*

lossz��<\���       �	��:(Yc�A�*

loss��!=�{S       �	p";(Yc�A�*

loss�%=���G       �	p�;(Yc�A�*

loss�9=��\       �	��<(Yc�A�*

loss�_�<g�6       �	&=(Yc�A�*

loss��<�q�       �	��=(Yc�A�*

loss@ �=A3�&       �	-@?(Yc�A�*

losst��<��:       �	@(Yc�A�*

loss�e�<���       �	¥@(Yc�A�*

loss��X=x<       �	�?A(Yc�A�*

loss�]=	�FV       �	/�A(Yc�A�*

loss��?=����       �	��B(Yc�A�*

loss�S9=��So       �	=DC(Yc�A�*

losst؇=z�i       �	��C(Yc�A�*

lossz'=j���       �	=|D(Yc�A�*

loss�rF=����       �	?�E(Yc�A�*

loss���<�`��       �	�|F(Yc�A�*

loss���=p���       �	�G(Yc�A�*

loss�.�=:��       �	��G(Yc�A�*

lossM
�=]�D7       �	PH(Yc�A�*

loss]��<i��       �	��H(Yc�A�*

loss#jB=;��       �	E�I(Yc�A�*

loss�m�=/��       �	["J(Yc�A�*

lossys=zss       �	��J(Yc�A�*

loss�]7<�Ha^       �	�K(Yc�A�*

loss9�=� 
       �	+L(Yc�A�*

lossA��<���       �	�L(Yc�A�*

loss�D�=���       �	%\M(Yc�A�*

losss��=�5�R       �	��M(Yc�A�*

loss�u>BP�;       �	N�N(Yc�A�*

loss�3 =�p{       �	n2O(Yc�A�*

loss�*�=& �       �	~�O(Yc�A�*

lossZI=^�       �	�]P(Yc�A�*

loss�B=�:+9       �	w�P(Yc�A�*

loss{\5=���       �	�Q(Yc�A�*

loss�_�<�F2       �	IR(Yc�A�*

loss�~�<�p��       �	m�R(Yc�A�*

loss4*�<8~��       �	�S(Yc�A�*

loss��Y=	rA�       �	T(Yc�A�*

loss��<W6)       �	u�T(Yc�A�*

lossE�<a�R       �	�UU(Yc�A�*

loss��<S2G�       �	S�U(Yc�A�*

lossXe�<��X�       �	w�V(Yc�A�*

loss�P�<~o�       �	o*W(Yc�A�*

loss�T�=/<�       �	�W(Yc�A�*

lossL��=I��       �	_`X(Yc�A�*

loss���<T+%4       �	-Y(Yc�A�*

lossZ>�I�d       �	հY(Yc�A�*

lossԺX;�w�       �	PZ(Yc�A�*

loss�&9=���       �	$�Z(Yc�A�*

loss��>#�I       �	�[(Yc�A�*

lossV�<2�j�       �	A(\(Yc�A�*

loss11=�q       �	?�\(Yc�A�*

loss��>s�       �	�a](Yc�A�*

lossN�=���       �	~�](Yc�A�*

lossV1<ϡ��       �	d�^(Yc�A�*

lossP<���       �	$*_(Yc�A�*

loss��=:��e       �	r�_(Yc�A�*

loss*�=;��       �	jk`(Yc�A�*

loss���=|��T       �	�a(Yc�A�*

losss�I=Ah)[       �	U�a(Yc�A�*

loss<�<'��a       �	X7b(Yc�A�*

loss)�q<����       �	S�b(Yc�A�*

lossdff=����       �	�kc(Yc�A�*

loss��`=^��       �	,d(Yc�A�*

loss���<�:?L       �	Z�d(Yc�A�*

loss.��;W���       �	�e(Yc�A�*

loss���=���z       �	��f(Yc�A�*

lossD��;#�3�       �		g(Yc�A�*

loss��y<x���       �	��g(Yc�A�*

loss�X�<=I>�       �	1^h(Yc�A�*

loss&UV=ev��       �	^�h(Yc�A�*

losss�<�ߠ6       �	�i(Yc�A�*

loss�N�;�'j]       �	�Mj(Yc�A�*

loss�f�=���I       �	��j(Yc�A�*

loss.�g=�� �       �	��l(Yc�A�*

loss�I�=� �       �	�<m(Yc�A�*

lossGp=����       �	�"o(Yc�A�*

loss�Җ=�ċ       �	0�o(Yc�A�*

loss�C�;�L;       �	�`p(Yc�A�*

lossÐ=��       �	rq(Yc�A�*

loss�=���2       �	Mr(Yc�A�*

loss�=���       �	ũr(Yc�A�*

loss��=Wz�       �	�Ms(Yc�A�*

loss\L�=[؀       �	�s(Yc�A�*

loss��==W�       �	�t(Yc�A�*

loss��K<�|u       �	�u(Yc�A�*

loss]I�=qT'�       �	x�u(Yc�A�*

loss`�[=Su�C       �	�Kv(Yc�A�*

lossI�=O��Z       �	��v(Yc�A�*

loss���;�
/?       �	�w(Yc�A�*

loss!�>��
       �	�$x(Yc�A�*

lossO3�;,���       �	U�x(Yc�A�*

loss��=P�V       �	 _y(Yc�A�*

loss	�=i:�       �	��y(Yc�A�*

loss��l==В       �	ۤz(Yc�A�*

lossn0�;|�       �	�H{(Yc�A�*

loss��{<#0]�       �	��{(Yc�A�*

loss i<��:�       �	+�|(Yc�A�*

lossT:�<e��n       �	�%}(Yc�A�*

lossr�Y=F�       �	L�}(Yc�A�*

loss�ߎ<�F"�       �	s�~(Yc�A�*

loss��=���        �	�@(Yc�A�*

loss�;'>��{       �	��(Yc�A�*

loss��=6K?2       �	膀(Yc�A�*

loss@�:<��       �	d"�(Yc�A�*

loss`R�<�z|       �	r��(Yc�A�*

loss�($<D
}�       �	�X�(Yc�A�*

loss�{:=����       �	��(Yc�A�*

lossA]<B�D       �	R��(Yc�A�*

lossH�.=d\��       �	�"�(Yc�A�*

loss1��;�_��       �	�̄(Yc�A�*

loss�މ=��d       �	�n�(Yc�A�*

loss��4=���       �	W	�(Yc�A�*

loss�D<>Z�w       �	S��(Yc�A�*

loss�=��b�       �	���(Yc�A�*

loss���<��\�       �	�$�(Yc�A�*

loss;��=Z��       �	�Ĉ(Yc�A�*

loss�]O=
apX       �	�]�(Yc�A�*

loss�X/=�,��       �	v��(Yc�A�*

loss�#�=�_��       �	���(Yc�A�*

loss-Y�<�y�       �	H6�(Yc�A�*

loss�eB<�T,;       �	�͋(Yc�A�*

loss��=c��u       �	Qh�(Yc�A�*

lossD�h=���       �	 ��(Yc�A�*

loss�O�<���       �	���(Yc�A�*

loss	
�;~R       �	{0�(Yc�A�*

lossq�Q<�"@       �	Ȏ(Yc�A�*

loss��<I;��       �	�`�(Yc�A�*

loss�>�<+�)       �	���(Yc�A�*

loss�3%=��e       �	-��(Yc�A�*

loss4]�=%[�       �	t)�(Yc�A�*

lossx@4>(�H�       �	���(Yc�A�*

loss_�<�i�       �	-Z�(Yc�A�*

loss1<����       �	v��(Yc�A�*

loss�rJ=�f       �	���(Yc�A�*

lossu�<�7w       �	.�(Yc�A�*

loss
�:�I��       �	̔(Yc�A�*

loss<vN�       �	t��(Yc�A�*

loss�<�	�       �	�t�(Yc�A�*

loss�
�<S��       �	�=�(Yc�A�*

loss:q�;�Ah       �	wڗ(Yc�A�*

loss���:�ۆC       �	:x�(Yc�A�*

loss�x=D��       �	��(Yc�A�*

loss�;�n�V       �	1Ι(Yc�A�*

lossƿ�:΅N       �	�f�(Yc�A�*

loss��`;DE^        �	%�(Yc�A�*

loss2`1<���       �	���(Yc�A�*

loss?��<Cw	T       �	.�(Yc�A�*

loss��u<&       �	���(Yc�A�*

loss�.�;��Oh       �	�Y�(Yc�A�*

loss�[=��3�       �	��(Yc�A�*

loss�h�>p�,w       �	�~�(Yc�A�*

loss��<��ϑ       �	U�(Yc�A�*

loss�Z�=���       �	H��(Yc�A�*

loss�3�<�>D       �	:X�(Yc�A�*

loss���=�E<       �	���(Yc�A�*

loss�2�<Ø�R       �	�Z�(Yc�A�*

loss�f=!V`�       �	��(Yc�A�*

lossL��=oG_�       �	��(Yc�A�*

loss��=W�!�       �	"�(Yc�A�*

loss��h=|�       �	��(Yc�A�*

loss=�u=oܖ�       �	(��(Yc�A�*

loss���<f�̚       �	e7�(Yc�A�*

loss��%>m2�       �	Ѧ(Yc�A�*

loss\	V=���        �	'f�(Yc�A�*

lossٴ=kVv�       �	(�(Yc�A�*

loss�&K=�M��       �	ؼ�(Yc�A�*

loss�x>u�,�       �	h�(Yc�A�*

loss��<�\þ       �	��(Yc�A�*

loss�l�<3<�       �	*��(Yc�A�*

loss�L�=���p       �	^*�(Yc�A�*

loss��R=�}��       �	��(Yc�A�*

loss��<[��       �	U�(Yc�A�*

loss{}0=��	       �	��(Yc�A�*

loss�V,=ڡ:6       �	���(Yc�A�*

loss@_�;�4��       �	�)�(Yc�A�*

lossnJ�<��       �	n��(Yc�A�*

loss�Ų;;C=f       �	�V�(Yc�A�*

loss���=;�mk       �	}�(Yc�A�*

loss�U�<�;�       �	V��(Yc�A�*

loss���=x7�C       �	b��(Yc�A�*

loss��=O�M       �	�%�(Yc�A�*

lossy
<�̷[       �	��(Yc�A�*

loss��=*ϓ^       �	S[�(Yc�A�*

loss���<�Ǿs       �	��(Yc�A�*

loss���<&�_       �	��(Yc�A�*

loss_�=R�e       �	�(Yc�A�*

loss3-4<dS׸       �	���(Yc�A�*

loss�{<�:ik       �	\9�(Yc�A�*

loss�Y=�aF�       �	�̶(Yc�A�*

loss<�=S��X       �	
g�(Yc�A�*

loss �0=����       �	�0�(Yc�A�*

loss7<�x�       �	�ʸ(Yc�A�*

loss�]L<�vu       �	R`�(Yc�A�*

loss�Q�<^�[o       �	��(Yc�A�*

loss���=��$-       �	���(Yc�A�*

loss�Æ<8C!       �	;8�(Yc�A�*

loss7�_=���       �	:ͻ(Yc�A�*

lossD��=}�$H       �	�`�(Yc�A�*

lossa $<g�0       �	���(Yc�A�*

loss��=-T�       �	�(Yc�A�*

loss�[<r�       �	�(Yc�A�*

loss]V�;�8*       �	F��(Yc�A�*

loss�==<�ߛ       �	��(Yc�A�*

loss�>S=Ųt�       �	���(Yc�A�*

lossō�=��m�       �	�O�(Yc�A�*

loss��j=7���       �	���(Yc�A�*

loss�l=�̤9       �	Ő�(Yc�A�*

lossW�<<�!       �	���(Yc�A�*

loss\�e=*��        �	"o�(Yc�A�*

loss�;A=��}�       �	�(Yc�A�*

loss�j=K$��       �	���(Yc�A�*

loss��=���       �	'M�(Yc�A�*

loss�c�<��       �	���(Yc�A�*

loss�3=�]hC       �	n��(Yc�A�*

loss�Jg=u�}       �	��(Yc�A�*

lossLV�=��       �	��(Yc�A�*

loss��= �'Q       �	p�(Yc�A�*

loss�X=7(       �	O�(Yc�A�*

loss��<S��       �	,��(Yc�A�*

losse�=@3U�       �	�W�(Yc�A�*

loss�6=Ĕ��       �	t��(Yc�A�*

loss�vu=�=�L       �	M��(Yc�A�*

lossħ=���       �	a�(Yc�A�*

loss�f�=��R       �	x��(Yc�A�*

loss�,�<[���       �	���(Yc�A�*

loss�D�=+���       �	�/�(Yc�A�*

loss}/�<elF:       �	e��(Yc�A�*

losss�,=��       �	�Y�(Yc�A�*

loss��x=#�       �	0��(Yc�A�*

loss�H�<��N�       �	���(Yc�A�*

loss�W=�!F'       �	�!�(Yc�A�*

loss�zW=بwX       �	g��(Yc�A�*

loss�B<���0       �	QL�(Yc�A�*

loss��=��)       �	��(Yc�A�*

loss�T�<!��~       �	���(Yc�A�*

lossBl�=�D��       �	�X�(Yc�A�*

lossV�X<�yH       �	*�(Yc�A�*

loss 4=����       �	մ�(Yc�A�*

loss!��; ��       �	kG�(Yc�A�*

loss��`=0�d       �	���(Yc�A�*

loss��>5��       �	|}�(Yc�A�*

loss��=�:%�       �	��(Yc�A�*

lossoC�</�T       �	���(Yc�A�*

loss�L<~�w�       �	�F�(Yc�A�*

loss�;%<�B�>       �	���(Yc�A�*

loss�И<!k�       �	t��(Yc�A�*

loss��=2]��       �	^K�(Yc�A�*

loss�˝=Up�       �	���(Yc�A�*

lossu�;�0 �       �	Ƨ�(Yc�A�*

lossj��<:�o       �	�U�(Yc�A�*

lossͺ�<'$h       �	���(Yc�A�*

loss�);�U��       �	1��(Yc�A�*

loss��;NQ7t       �	3�(Yc�A�*

lossE�5=�y/       �	���(Yc�A�*

loss
w<e�       �	�f�(Yc�A�*

loss _�>6�@�       �	P �(Yc�A�*

lossh<r��3       �	��(Yc�A�*

lossn��:�T�0       �	�R�(Yc�A�*

loss�;"m       �	$& )Yc�A�*

loss���;.GJX       �	� )Yc�A�*

loss*�<9���       �	�V)Yc�A�*

loss��=��%       �	��)Yc�A�*

loss�>2� �       �	��)Yc�A�*

loss斕=��z       �	�=)Yc�A�*

loss:�;<.�       �	%)Yc�A�*

loss8qJ=�a��       �	ŭ)Yc�A�*

loss���<��p       �	�Y)Yc�A�*

loss̘;>(T�       �	4�)Yc�A�*

lossE�=�Y�       �	E�)Yc�A�*

lossH�-=��%       �	wL)Yc�A�*

loss�V�=��       �	�)Yc�A�*

loss��=���       �	֍)Yc�A�*

loss�v:=�+>H       �	7	)Yc�A�*

loss��=�Z�       �	��	)Yc�A�*

loss�6q<�{�       �	�{
)Yc�A�*

loss�P	=V#ys       �	�)Yc�A�*

loss�δ;#�e       �	G�)Yc�A�*

loss��Y=˲;�       �	EH)Yc�A�*

lossd�;��       �	L�)Yc�A�*

loss�r�<60;�       �	�u)Yc�A�*

loss�_�;��`�       �	�)Yc�A�*

lossӒc=a��2       �	��)Yc�A�*

loss_�_=0��8       �	ca)Yc�A�*

lossŴ�<����       �	f�)Yc�A�*

lossN�<���       �	_�)Yc�A�*

loss;��<�rb       �	$D)Yc�A�*

loss�̐<m=��       �	q�)Yc�A�*

loss�t�<|�@�       �	؂)Yc�A�*

lossS�.=UM��       �	�)Yc�A�*

loss�<S\J       �	�)Yc�A�*

loss_�q<�a       �	=F)Yc�A�*

loss���<�]~�       �	[�)Yc�A�*

loss4��<���{       �	�|)Yc�A�*

loss�i�=�(�       �	�)Yc�A�*

loss���=�0J       �	��)Yc�A�*

loss*��=��e�       �	�A)Yc�A�*

loss.x�=R��,       �	&�)Yc�A�*

loss��<k5�h       �	�)Yc�A�*

loss]+�=�
�       �	�&)Yc�A�*

loss���<$�       �	��)Yc�A�*

lossZ�=M��       �	2�)Yc�A�*

loss6`x=b9�Y       �	�5)Yc�A�*

lossMA�<�/8�       �	s�)Yc�A�*

loss���;�=�       �	�r)Yc�A�*

loss�^�<vQQ�       �	M)Yc�A�*

loss���< #��       �	�)Yc�A�*

loss/��=X��       �	�R)Yc�A�*

lossmƬ=qд�       �	N�)Yc�A�*

loss!�;����       �	D�)Yc�A�*

lossS��<�i�       �	Y3 )Yc�A�*

loss��=��?       �	R� )Yc�A�*

loss�˧<�0b       �	n!)Yc�A�*

loss�G�=�+=�       �	g)")Yc�A�*

loss�XR=:K�       �	��")Yc�A�*

lossİ`=}&�       �	�c#)Yc�A�*

lossJ��;��[       �	�$)Yc�A�*

lossw�@=�$�K       �	7%)Yc�A�*

loss�޵<��T�       �	2�%)Yc�A�*

loss�3R<<��}       �	��&)Yc�A�*

loss-�k=�<�9       �	�;')Yc�A�*

loss^�=*oˢ       �	�()Yc�A�*

loss�cX<���       �	�()Yc�A�*

lossI�)=ϒ��       �	~p))Yc�A�*

lossg�=#:؉       �	D*)Yc�A�*

loss���<�_       �	d�*)Yc�A�*

lossdV!=o��       �	Y�+)Yc�A�*

loss�Y<��.       �	�.,)Yc�A�*

loss�]�<mPq       �	,�,)Yc�A�*

loss�џ=ކ��       �	�x-)Yc�A�*

loss߯<�
<<       �	� .)Yc�A�*

lossͤ=�s76       �	 �.)Yc�A�*

loss�9=�       �	n/)Yc�A�*

loss)�`<�$�       �	�0)Yc�A�*

loss6�p=�9�       �	��0)Yc�A�*

loss�IM=���       �	�b1)Yc�A�*

loss�]<*�ّ       �	�2)Yc�A�*

loss{O�=ޖ�       �	Y�2)Yc�A�*

loss�d<�|�0       �	�@3)Yc�A�*

loss��W=C%       �	��3)Yc�A�*

lossw��<�d�       �	�z4)Yc�A�*

loss��Y;*�q�       �	�%5)Yc�A�*

loss�=H� �       �	a�5)Yc�A�*

loss�QE=��8       �	(�6)Yc�A�*

loss6__<�:��       �	qs7)Yc�A�*

loss��+=ï�       �	�8)Yc�A�*

loss��->��,       �	��8)Yc�A�*

lossѡ=�4B�       �	�[9)Yc�A�*

lossc_<�K�=       �	��9)Yc�A�*

loss�W=&9�z       �	��:)Yc�A�*

loss��;���       �	<)Yc�A�*

loss��<9��r       �	H�<)Yc�A�*

losst��<7˸       �	�W=)Yc�A�*

loss
TQ<Xz�q       �	8�=)Yc�A�*

loss%o@=g��       �	n�>)Yc�A�*

loss��X=H[�<       �	JA?)Yc�A�*

losse�=="Y��       �	�@)Yc�A�*

loss	�=e3       �	��@)Yc�A�*

loss��;	o�p       �	��A)Yc�A�*

loss��<:ɥ4       �	�B)Yc�A�*

loss��2=�m�       �	�TC)Yc�A�*

loss�ɲ=I1�       �	��C)Yc�A�*

loss[�%=��d5       �	l�D)Yc�A�*

loss�]�=��2{       �	�\E)Yc�A�*

lossLT=ـ�       �	7�F)Yc�A�*

loss�}=�_�q       �	�G)Yc�A�*

lossm��<��q�       �	!H)Yc�A�*

loss��=��kF       �	��H)Yc�A�*

loss�ς<mŜ�       �	�~I)Yc�A�*

loss)�'<L�ϛ       �	�'J)Yc�A�*

loss�J2<98~�       �	��J)Yc�A�*

loss�Ca='��       �	Z�K)Yc�A�*

loss��=a���       �	�:L)Yc�A�*

loss�=�F��       �	M�L)Yc�A�*

loss��%<*q?�       �	a�M)Yc�A�*

loss�%�<y��       �	uuN)Yc�A�*

loss�:q=��~�       �	*O)Yc�A�*

loss��L=��`�       �	t�P)Yc�A�*

lossq�N=��L       �	�;Q)Yc�A�*

loss�я=ڂ(�       �	*�Q)Yc�A�*

lossׁ:=&qw�       �	]�R)Yc�A�*

loss��;� �       �	l"S)Yc�A�*

loss֤�<>[	�       �	��S)Yc�A�*

loss��<��       �	uT)Yc�A�*

lossm�&=�^��       �	�U)Yc�A�*

loss�w�=�cr�       �	��U)Yc�A�*

loss��<<�;�U       �	`�V)Yc�A�*

loss��<�+�       �	NGW)Yc�A�*

loss���;k�Z�       �	��W)Yc�A�*

loss��{=I�3_       �	ҋX)Yc�A�*

lossJ�_=r%�       �	�"Y)Yc�A�*

loss�1�<��5�       �	w�Y)Yc�A�*

loss�ޙ=z]�       �	�YZ)Yc�A�*

loss:B7= �~�       �	y[)Yc�A�*

loss�'=���,       �	,�[)Yc�A�*

loss��#=o*B       �	f0\)Yc�A�*

loss���<7�M2       �	��\)Yc�A�*

loss�7�;J]w>       �	�p])Yc�A�*

loss#�Q<���       �	�^)Yc�A�*

loss��X=;-p       �	��^)Yc�A�*

lossf�=_��       �	�v_)Yc�A�*

loss�a�=8���       �	`)Yc�A�*

lossy=����       �	6�`)Yc�A�*

loss
�<�v_r       �	�Sa)Yc�A�*

loss��<7�]       �	��a)Yc�A�*

loss��<��g       �	*�b)Yc�A�*

loss,+=˰y�       �	�+c)Yc�A�*

loss�ܶ<2%�#       �	�d)Yc�A�*

loss5-�<�+}#       �	Q�d)Yc�A�*

lossȶ=_��       �	�<e)Yc�A�*

lossA�h<y]�f       �	x�e)Yc�A�*

lossm�=����       �	�of)Yc�A�*

loss:�=��       �	kg)Yc�A�*

loss��=��%       �	�g)Yc�A�*

lossz΃<~?:1       �	>?h)Yc�A�*

lossf�C<'��Z       �	O�h)Yc�A�*

lossIu�;o��       �	"�i)Yc�A�*

loss�3�=���i       �	�j)Yc�A�*

loss-=�1Q       �	0Jk)Yc�A�*

loss�.>?���       �	]�k)Yc�A�*

loss|>�=+�       �	El)Yc�A�*

lossԔ�=dk�:       �	Cm)Yc�A�*

loss�փ=z
��       �	,�m)Yc�A�*

loss��<]W-r       �	YRn)Yc�A�*

lossW�={N��       �	��n)Yc�A�*

loss��=y�>       �	m�o)Yc�A�*

lossh��=`��       �	�#p)Yc�A�*

lossBV<mE��       �	=�p)Yc�A�*

loss�!=h���       �	^�q)Yc�A�*

loss��=>O��       �	jmr)Yc�A�*

loss���<�Ӱ�       �	k)s)Yc�A�*

loss�=9�E       �	��s)Yc�A�*

loss�j=�C�3       �	#gt)Yc�A�*

loss�"�;c$�}       �	7�t)Yc�A�*

loss���;%�G       �	1�u)Yc�A�*

loss&��<���       �	y;v)Yc�A�*

loss:��<ޢ?E       �	��v)Yc�A�*

loss
P_=3k       �	njw)Yc�A�*

loss�]�<�TL       �	� x)Yc�A�*

loss��=��:k       �	:�x)Yc�A�*

loss��3<��~�       �	�4y)Yc�A�*

loss��>����       �	��y)Yc�A�*

loss
t�<e���       �	�mz)Yc�A�*

loss��<�7�C       �	x
{)Yc�A�*

loss�%=>�Q"       �	��{)Yc�A�*

loss1R+=��	       �	mt|)Yc�A�*

loss�ʁ=�(       �	�`})Yc�A�*

loss� Y=&pϒ       �		�})Yc�A�*

lossf#�<)w�       �	5�~)Yc�A�*

loss=B<�p�R       �	2)Yc�A�*

loss�h�<Bh�1       �	-%�)Yc�A�*

loss6C�<����       �	���)Yc�A�*

loss��<;m�       �	�O�)Yc�A�*

loss!�==�y�       �	��)Yc�A�*

loss�#�<��]       �	`��)Yc�A�*

loss#�>=e� K       �	�,�)Yc�A�*

loss���<7i%t       �	�σ)Yc�A�*

lossN;jaٯ       �	W|�)Yc�A�*

loss�A<h�,D       �	�L�)Yc�A�*

loss\l�<w ��       �	��)Yc�A�*

loss�_=�D��       �	ڍ�)Yc�A�*

lossr
@=8�+�       �	�2�)Yc�A�*

loss�	�<����       �	V�)Yc�A�*

lossoQ=�(>       �	���)Yc�A�*

lossI�[<��       �	:�)Yc�A�*

loss��=Ym2       �	_҉)Yc�A�*

loss	d<���       �	�p�)Yc�A�*

loss\^)=f;��       �	��)Yc�A�*

loss	<��ڝ       �	ѭ�)Yc�A�*

loss�u�=x���       �	FE�)Yc�A�*

loss�ɥ<�|9       �	��)Yc�A�*

loss<}�=wAf�       �	$��)Yc�A�*

loss3��<_q�f       �	 �)Yc�A�*

losszj=�q�z       �	GƎ)Yc�A�*

loss]�3=�f�       �	|�)Yc�A�*

loss�%z<���@       �	�)Yc�A�*

loss�ф=�T�       �	/Ð)Yc�A�*

loss��>c G�       �	�_�)Yc�A�*

loss��>�	M       �	D��)Yc�A�*

loss�?�<�5��       �	��)Yc�A�*

loss�#;|h�       �	
1�)Yc�A�*

loss�ŵ=���       �	�Ǔ)Yc�A�*

loss���<�r�L       �	^�)Yc�A�*

loss;D���       �	���)Yc�A�*

loss#�;�V	       �	�)Yc�A�*

loss�=�Ѫ@       �	P��)Yc�A�*

loss�mj<�3߆       �	�(�)Yc�A�*

loss�=� 
<       �	��)Yc�A�*

loss��=����       �	�O�)Yc�A�*

loss�Q=��c�       �	\�)Yc�A�*

loss�er<|a�       �	���)Yc�A�*

loss�=Ʃ��       �	(�)Yc�A�*

loss��<�,�H       �	��)Yc�A�*

loss��5;s,��       �	���)Yc�A�*

loss7�=d��P       �	'�)Yc�A�*

loss>��:U��       �	PÜ)Yc�A�*

loss%i�:֪a�       �	���)Yc�A�*

loss�k*<��       �	�K�)Yc�A�*

loss�<c�:       �	��)Yc�A�*

loss1_'=WJ�l       �	7��)Yc�A�*

lossH�<8	��       �	q!�)Yc�A�*

loss��j=3�       �	$��)Yc�A�*

loss$�=����       �	8M�)Yc�A�*

loss�N�<��ޥ       �	��)Yc�A�*

lossW�t=�G��       �	�s�)Yc�A�*

lossZ�Y=$��~       �	��)Yc�A�*

loss���=���`       �	X��)Yc�A�*

loss���=(Ƒ       �	[B�)Yc�A�*

loss��=��J       �	N��)Yc�A�*

loss��;3�K�       �	���)Yc�A�*

lossт�=r�5�       �	5�)Yc�A�*

loss�S<���       �	�Ӧ)Yc�A�*

lossS:v=<Pq�       �	�o�)Yc�A�*

loss�"=�P�x       �	�	�)Yc�A�*

loss��:=:fkx       �	��)Yc�A�*

loss�[e=�6mp       �	-@�)Yc�A�*

loss�n<$љ\       �	b۩)Yc�A�*

loss��<Hc��       �	*t�)Yc�A�*

loss/>'�RS       �	��)Yc�A�*

loss.�q<��U       �	���)Yc�A�*

loss7=�<��a       �	�:�)Yc�A�*

loss��6=�!�       �	�֬)Yc�A�*

lossx|�<]��       �	�m�)Yc�A�*

loss	ǐ< � �       �	�	�)Yc�A�*

loss���=�j:        �	ץ�)Yc�A�*

loss�U�=�Tc       �	�<�)Yc�A�*

lossG<&�=�       �	�ׯ)Yc�A�*

loss��=��x       �	Gr�)Yc�A�*

loss:*=R=5       �	�)Yc�A�*

loss<G&=[wi       �	���)Yc�A�*

loss�J=</�       �	�K�)Yc�A�*

loss�V=	�J�       �	��)Yc�A�*

lossT� >�\?�       �	���)Yc�A�*

loss�2X=n��B       �	#2�)Yc�A�*

loss��-=m.�       �	=)�)Yc�A�*

lossc�D<w�#       �	�Ƶ)Yc�A�*

lossij=6pHj       �	xc�)Yc�A�*

loss��=nd�       �	y�)Yc�A�*

loss��J=�v�x       �	��)Yc�A�*

lossJ�T=���1       �	�=�)Yc�A�*

lossf�;<��GA       �	��)Yc�A�*

loss��={��       �	g��)Yc�A�*

loss��G;����       �	��)Yc�A�*

lossd�=Q'KK       �	�к)Yc�A�*

lossR�;V�B�       �	�l�)Yc�A�*

lossf��=�v       �	��)Yc�A�*

lossd=���[       �	|��)Yc�A�*

lossHn�=�l�f       �	�d�)Yc�A�*

loss�ĥ=����       �	���)Yc�A�*

loss�� =(��       �	_��)Yc�A�*

loss�#�<���       �	�.�)Yc�A�*

loss�h�=����       �	��)Yc�A�*

lossuQ=���z       �	���)Yc�A�*

loss��-<�<�N       �	
M�)Yc�A�*

loss�I�<U?��       �	��)Yc�A�*

loss�M�=���       �	֫�)Yc�A�*

loss�Ot=�VN�       �	���)Yc�A�*

lossL�W=��       �	�j�)Yc�A�*

lossTm<U�9�       �	� �)Yc�A�*

loss;*
<��>�       �	,��)Yc�A�*

losssZ�<�u��       �	<0�)Yc�A�*

loss��<�1x       �	���)Yc�A�*

loss���<[>IF       �	)Z�)Yc�A�*

loss�,=u(|       �	c	�)Yc�A�*

losset�<�!غ       �	0��)Yc�A�*

loss �<��N       �	�4�)Yc�A�*

loss�,�≠�D       �	���)Yc�A�*

loss�[�<fk�       �	�`�)Yc�A�*

loss)��=<Ҥ�       �	���)Yc�A�*

lossT�>�1       �	���)Yc�A�*

lossT�	=�#�       �	�-�)Yc�A�*

loss��=��1�       �	"��)Yc�A�*

loss*�=��       �	�h�)Yc�A�*

lossI��<?x       �	t�)Yc�A�*

loss���<�\�       �	D��)Yc�A�*

loss�T�<��4�       �	�?�)Yc�A�*

loss���;ʨ�%       �	��)Yc�A�*

loss��P=�kP       �	[z�)Yc�A�*

lossi�|=��,       �	�)Yc�A�*

loss�z�=Y��       �	��)Yc�A�*

loss��U=4�h       �	ގ�)Yc�A�*

losssw�<��F       �	�>�)Yc�A�*

loss-�<ߪa�       �	d�)Yc�A�*

loss�+�=[%�       �	,��)Yc�A�*

loss��=I��       �	�\�)Yc�A�*

loss�+�=�� �       �	���)Yc�A�*

lossI��<�"�       �	��)Yc�A�*

loss�C<Էi�       �	-\�)Yc�A�*

lossa�=5J~       �	���)Yc�A�*

lossd�<�P       �	Н�)Yc�A�*

loss�<�aep       �	�6�)Yc�A�*

loss Y�=��j       �	���)Yc�A�*

loss;<v�N'       �	��)Yc�A�*

loss*=�Ƚ�       �	�*�)Yc�A�*

lossŁ�<�j�       �	���)Yc�A�*

loss��+=�Rģ       �	�k�)Yc�A�*

loss�Z�="�a�       �	��)Yc�A�*

loss��7=�4�j       �	���)Yc�A�*

loss�ч=�	w�       �	�{�)Yc�A�*

loss���<t!       �	I�)Yc�A�*

loss�0<1\�y       �	���)Yc�A�*

loss�= f�       �	�A�)Yc�A�*

loss�!�<�Rѕ       �	���)Yc�A�*

loss��o;�J��       �	U��)Yc�A�*

loss�<L�L\       �	�&�)Yc�A�*

loss�L=[�
g       �	���)Yc�A�*

loss责=]�3�       �	Lm�)Yc�A�*

loss��=޻�       �	��)Yc�A�*

loss[�<JGT       �	���)Yc�A�*

loss��=B��       �	|��)Yc�A�*

loss�&�=�t�}       �	�:�)Yc�A�*

loss{s=��       �	���)Yc�A�*

lossX��=k�;       �	ɑ�)Yc�A�*

loss1��=]�[�       �	�;�)Yc�A�*

lossO�=�H(�       �	��)Yc�A�*

lossۺ7=�/4       �	O��)Yc�A�*

loss6�=��A�       �	���)Yc�A�*

loss�c"=��H�       �	�;�)Yc�A�*

loss��q=Ӥ��       �	���)Yc�A�*

loss��;:}�,       �	���)Yc�A�*

loss��.<r���       �	�y�)Yc�A�*

loss��2=)��v       �	_$�)Yc�A�*

lossw��=���`       �	*��)Yc�A�*

lossl.�;ߦ�       �	k�)Yc�A�*

loss1��<��       �	B�)Yc�A�*

loss�g�<��>�       �	���)Yc�A�*

loss���=>g       �	��)Yc�A�*

loss���=�㾶       �	�B�)Yc�A�*

losss�>�Q�       �	���)Yc�A�*

lossR?%=��|g       �	���)Yc�A�*

lossnc�=j�hz       �	�\�)Yc�A�*

loss7��<��n       �	c�)Yc�A�*

loss���<i�?       �	9��)Yc�A�*

loss4�<J�=       �	�l�)Yc�A�*

loss��<B�       �	�g�)Yc�A�*

loss|�"=�ͥ�       �	,�)Yc�A�*

lossə�<��       �	���)Yc�A�*

loss`Sj=E��       �	�l�)Yc�A�*

loss&(�<��k#       �	�9�)Yc�A�*

loss;$N=��$       �	��)Yc�A�*

loss�<慯�       �	��)Yc�A�*

loss�"e<H��,       �	G!�)Yc�A�*

loss��)=�gb�       �	
��)Yc�A�*

lossc�=Yg�       �	1\�)Yc�A�*

loss(g�=땂�       �	W#�)Yc�A�*

loss6�O=�f       �	���)Yc�A�*

loss<q=xk0       �	x~�)Yc�A�*

loss��b<���       �	%$ *Yc�A�*

loss�p�=��ؓ       �	�� *Yc�A�*

lossT��=�Jr�       �	��*Yc�A�*

loss��<�7��       �	3O*Yc�A�*

lossT��<t	       �	��*Yc�A�*

lossy�=�*2(       �	y�*Yc�A�*

loss�q=;��       �	LQ*Yc�A�*

loss&
�;e���       �	?�*Yc�A�*

loss�CA;��xY       �	��*Yc�A�*

lossl'>X�       �	Zc*Yc�A�*

loss΅>?��m       �	d*Yc�A�*

loss��<g�r       �	ߨ*Yc�A�*

loss[��<��Be       �	�G*Yc�A�*

lossS�B=�"y       �	��*Yc�A�*

lossn}=����       �	��	*Yc�A�*

loss���<��       �	�1
*Yc�A�*

loss���;w<:        �	��
*Yc�A�*

loss8��<��P        �	�{*Yc�A�*

loss��;���c       �	*Yc�A�*

loss>�=_�        �	7�*Yc�A�*

loss��=���       �	Tq*Yc�A�*

loss}7<ڎ2�       �	�&*Yc�A�*

lossP�<����       �	f�*Yc�A�*

lossM=��       �	ӣ*Yc�A�*

loss�~�=*�       �	�W*Yc�A�*

loss�(,<(?9(       �	g*Yc�A�*

loss׵=v��e       �	�*Yc�A�*

loss�]d<���       �	0�*Yc�A�*

loss )�=~�*�       �	�j*Yc�A�*

loss���=��h       �	]*Yc�A�*

lossa��<A�1       �	��*Yc�A�*

lossgd�<��0�       �	:�*Yc�A�*

loss1d�;���       �	�@*Yc�A�*

lossM�<e�       �	%�*Yc�A�*

loss�p�;�f��       �	ɓ*Yc�A�*

lossj�=���       �	,H*Yc�A�*

loss]�=�]       �	�*Yc�A�*

loss�Х=- *�       �	�*Yc�A�*

lossL�<���A       �	�T*Yc�A�*

loss�S�;��b�       �		*Yc�A�*

loss���<��Bk       �	s�*Yc�A�*

loss>��<c<{;       �	�*Yc�A�*

loss�l.<ki�>       �	�D*Yc�A�*

loss�ɘ=VE
5       �	��*Yc�A�*

loss%�E<��       �	��*Yc�A�*

loss\}=Nf֥       �	�)*Yc�A�*

loss��%=�_�       �	F�*Yc�A�*

lossVO=Pi       �	�l *Yc�A�*

loss�?;p2�#       �	!*Yc�A�*

loss�t�=�_P�       �	6�!*Yc�A�*

lossЃ=m*D�       �	�M"*Yc�A�*

loss&�<�ڻ       �	��"*Yc�A�*

loss.��<Wl�[       �	-�#*Yc�A�*

loss�5<L�       �	f�$*Yc�A�*

lossC�O<�!�V       �	�f%*Yc�A�*

loss��Z=�9       �	�<&*Yc�A�*

loss��=��U       �	<�&*Yc�A�*

loss|�<<�H>Q       �	�'*Yc�A�*

lossdy<��{       �	�&(*Yc�A�*

loss�|<�W�       �	2�(*Yc�A�*

loss�2=���       �	eo)*Yc�A�*

loss7N�<�ʉ4       �	q�**Yc�A�*

loss�J=>�2�       �	{+*Yc�A�*

loss�&B<�>��       �	m,*Yc�A�*

loss�x�=_�       �	��,*Yc�A�*

loss�3�=~�)1       �	�s-*Yc�A�*

loss��P<�X       �	h!.*Yc�A�*

lossA!=�3�       �	��.*Yc�A�*

lossj:�<�1u       �	�/*Yc�A�*

lossZ�=8��s       �	�0*Yc�A�*

loss�Ja=�f       �	�0*Yc�A�*

loss�V0=枾       �	�]1*Yc�A�*

loss�pr=宋�       �	�2*Yc�A�*

loss\N`<7[�       �	-�2*Yc�A�*

losss,�<>��       �	`Z3*Yc�A�*

lossD$=$u,�       �	�3*Yc�A�*

lossF�T<��/       �	 �4*Yc�A�*

loss_�^=�       �	�05*Yc�A�*

loss�=;�g,�       �	u�5*Yc�A�*

loss�=[P       �	j6*Yc�A�*

lossƴ<�i�z       �	�7*Yc�A�*

lossHj�;��Kq       �	�8*Yc�A�*

loss��<��       �	B�8*Yc�A�*

loss�G)=�bn       �	rN9*Yc�A�*

loss�L=��Z       �	��9*Yc�A�*

loss���=dYZ       �	�,;*Yc�A�*

loss̀�;�+�E       �	��;*Yc�A�*

loss�ׯ<�� �       �	�e<*Yc�A�*

lossm7<S�\o       �	=*Yc�A�*

lossZ�:��       �	;�=*Yc�A�*

loss�/<���       �	�B>*Yc�A�*

loss�N�;n�%       �	�?*Yc�A�*

loss��{;�~qQ       �	 �?*Yc�A�*

loss�-�;*>hH       �	U@*Yc�A�*

loss��;
O[       �	��@*Yc�A�*

loss7�@=�ǃ       �	��A*Yc�A�*

lossƆ<V���       �	�"B*Yc�A�*

loss[8c:�v��       �	��B*Yc�A�*

loss8*;�'�       �	�VC*Yc�A�*

loss�g=�p��       �	�D*Yc�A�*

loss6�w=$y9       �	��D*Yc�A�*

loss�D�;4���       �	nRE*Yc�A�*

loss?��:o3�X       �	�E*Yc�A�*

lossB��=`w�       �	�F*Yc�A�*

lossZr>8�an       �	��G*Yc�A�*

loss3�;Ƨ��       �	qH*Yc�A�*

loss�hT>�!V�       �	3I*Yc�A�*

loss�(*=#*!K       �	��I*Yc�A�*

loss��g=�u�_       �	�DJ*Yc�A�*

loss�ȫ=�T�       �	��J*Yc�A�*

loss8�;y��       �	�nK*Yc�A�*

loss]%�=-|��       �	6L*Yc�A�*

loss��I=�X��       �	9�L*Yc�A�*

loss���=��R       �	�,M*Yc�A�*

loss��G=
~�       �	�M*Yc�A�*

loss�a�<��L�       �	�YN*Yc�A�*

loss�R=n$�;       �	�N*Yc�A�*

loss7Xf=��=       �	�O*Yc�A�*

lossHe=�F�       �	�,P*Yc�A�*

lossO�*=�.�       �	D�P*Yc�A�*

loss.�=�{�       �	|~Q*Yc�A�*

loss�D[={��_       �	�!R*Yc�A�*

loss��t<cF `       �	1�R*Yc�A�*

loss�^�=�� 0       �	�LS*Yc�A�*

loss��:=&�5�       �	�S*Yc�A�*

loss3F<��       �	�|T*Yc�A�*

lossT�K<ޗ��       �	�U*Yc�A�*

loss���<���       �	��U*Yc�A�*

lossB<2�       �	8KV*Yc�A�*

loss��<��+       �	��V*Yc�A�*

loss��r;���       �	(�W*Yc�A�*

lossIA�=�tc$       �	�<X*Yc�A�*

loss��=}�7       �	f�X*Yc�A�*

loss�I;=��z       �	�tY*Yc�A�*

lossx�=�>J       �	+Z*Yc�A�*

loss&�<j�K�       �	_�Z*Yc�A�*

loss��H=�fh�       �	�Q[*Yc�A�*

loss�|�<���       �	�[*Yc�A�*

loss�%�;zQ�i       �	*�\*Yc�A�*

loss$��<��U#       �	�,]*Yc�A�*

lossJVl<y�fA       �	��]*Yc�A�*

loss�X<o��       �	�{^*Yc�A�*

lossi I=G��       �	�_*Yc�A�*

loss��=E        �	��_*Yc�A�*

loss��I=��	k       �	�`*Yc�A�*

loss�	L=4���       �	�^a*Yc�A�*

loss�	)=�jh�       �	��a*Yc�A�*

loss�P�<rA�       �	B�b*Yc�A�*

loss�U=�K��       �	|'c*Yc�A�*

lossv�<�]Q�       �	nd*Yc�A�*

lossZ9=�N��       �	a�d*Yc�A�*

lossDَ=��[�       �	�_e*Yc�A�*

loss���;�ңM       �	4�e*Yc�A�*

loss���<�t*�       �	
�f*Yc�A�*

loss�'�;�̈       �	2g*Yc�A�*

loss�<�hg|       �	��g*Yc�A�*

loss��"=C�z�       �	I�*Yc�A�*

loss�q�=�_        �	J�*Yc�A�*

loss�Ր=�װ4       �	G�*Yc�A�*

lossL>�=��>H       �	���*Yc�A�*

loss�e�<�G��       �	+i�*Yc�A�*

losss�&=�ݵ9       �	��*Yc�A�*

lossde=P�L�       �	��*Yc�A�*

loss3X"=���S       �	xD�*Yc�A�*

loss�j=	�q�       �	��*Yc�A�*

loss���=�h�:       �	��*Yc�A�*

loss7β<2��       �	�&�*Yc�A�*

loss�<!;�       �	�ċ*Yc�A�*

loss�<�:��       �	�l�*Yc�A�*

lossn9=9��~       �	��*Yc�A�*

loss{�=��V�       �	���*Yc�A�*

loss�e~=i_y       �	j��*Yc�A�*

loss�(�;��)       �	(,�*Yc�A�*

loss$Q�;���%       �	Ώ*Yc�A�*

lossm=�IH%       �	l�*Yc�A�*

loss�8�=(zu�       �	Xt�*Yc�A�*

loss���;�ϋ�       �	B%�*Yc�A�*

loss��=�  �       �	"Ē*Yc�A�*

lossO��;>�Q       �	wj�*Yc�A�*

loss�Ś=��l�       �	���*Yc�A�*

loss$��<\�g�       �	�B�*Yc�A�*

loss��<��?�       �	~�*Yc�A�*

lossFD�=��P�       �	2ʖ*Yc�A�*

lossIG�<��S7       �	�_�*Yc�A�*

loss�A2=��#       �	�6�*Yc�A�*

loss��=J�m       �	�Й*Yc�A�*

lossVB<x��       �	/m�*Yc�A�*

loss4��<޷i�       �	��*Yc�A�*

lossQSi<>ݠ       �	���*Yc�A�*

loss���=��̫       �	zT�*Yc�A�*

loss�<�7?       �	�R�*Yc�A�*

losse]�<��4�       �	E�*Yc�A�*

lossQ��<j8�       �	L��*Yc�A�*

loss��+=��       �	�M�*Yc�A�*

loss� �=욦�       �	��*Yc�A�*

loss���=�+�       �	�z�*Yc�A�*

loss�=�y�       �	��*Yc�A�*

loss��=��H       �	�ɡ*Yc�A�*

loss6� <jZ݇       �	b��*Yc�A�*

loss���<8��o       �	��*Yc�A�*

loss���=����       �	��*Yc�A�*

loss\E=$�6       �	,g�*Yc�A�*

loss$(�=���       �	���*Yc�A�*

lossS�N=�먈       �	֍�*Yc�A�*

lossWlT=<�=�       �	N%�*Yc�A�*

loss�(t:P,N�       �	R�*Yc�A�*

loss]h<{��:       �	z��*Yc�A�*

loss��1=�*(       �	�2�*Yc�A�*

loss�8�<��y�       �	��*Yc�A�*

loss �T=���       �	���*Yc�A�*

loss�h,<�/�       �	�+�*Yc�A�*

loss�m�;lz�       �	�*Yc�A�*

loss�}Y;�`��       �	�«*Yc�A�*

loss�j�;BmFd       �	6w�*Yc�A�*

loss��<��H3       �	��*Yc�A�*

loss<�*=�~�       �	c��*Yc�A�*

loss���=v�W       �	l_�*Yc�A�*

lossO�O<�a       �	�W�*Yc�A�*

loss�x&=��W�       �	��*Yc�A�*

loss�6�<l�R}       �	��*Yc�A�*

lossvn�<��ey       �	�*Yc�A�*

loss=�sD�       �	g�*Yc�A�*

loss�Ջ<�ל       �	��*Yc�A�*

loss51�<+��       �	aó*Yc�A�*

loss$3�=�:��       �	�ش*Yc�A�*

loss�f�<�I       �	Xu�*Yc�A�*

loss�<,V�       �	&:�*Yc�A�*

loss8�=��w�       �	���*Yc�A�*

lossΗ�=Rq��       �	�η*Yc�A�*

lossj��<�;.-       �	䠸*Yc�A�*

loss�>�p       �	O>�*Yc�A�*

lossҰ{<���L       �	��*Yc�A�*

loss.�=���       �	��*Yc�A�*

loss��]=?��       �	���*Yc�A�*

loss&I<�_TK       �	�E�*Yc�A�*

lossO�=�N�5       �	�ؼ*Yc�A�*

loss�.�<���       �	/m�*Yc�A�*

lossD
=boQ       �	�	�*Yc�A�*

lossr��=�a�       �	<��*Yc�A�*

loss�MB=q��       �	a7�*Yc�A�*

loss3��<���G       �	˿*Yc�A�*

loss��=pZv       �	�]�*Yc�A�*

loss�Cl=��       �	���*Yc�A�*

lossH&�<����       �	���*Yc�A�*

lossKV<��!       �	t�*Yc�A�*

lossc�<����       �	t	�*Yc�A�*

losse�=Y���       �	=��*Yc�A�*

loss8�&=(U       �	�2�*Yc�A�*

lossi��<XȾ�       �	*��*Yc�A�*

loss��<	�       �	�k�*Yc�A�*

loss�=˃l�       �	�:�*Yc�A�*

loss}x�;[���       �	���*Yc�A�*

loss��==^���       �	a�*Yc�A�*

lossj��<>(n       �	���*Yc�A�*

loss��=�^��       �	 ��*Yc�A�*

loss6�&<��>�       �	�L�*Yc�A�*

lossf =9���       �	���*Yc�A�*

loss��<�;�~       �	i��*Yc�A�*

loss�Yq=]�6       �	��*Yc�A�*

loss��<�       �	O��*Yc�A�*

loss�� =l�l�       �	�B�*Yc�A�*

lossQ�=�=�       �	]��*Yc�A�*

loss��R;a�d@       �	�{�*Yc�A�*

lossZ�?<��       �	{�*Yc�A�*

loss��=?&.�       �	��*Yc�A�*

loss��M=#���       �	�I�*Yc�A�*

loss���=�F�       �	���*Yc�A�*

loss��<�@	       �	��*Yc�A�*

loss���<�u�       �	��*Yc�A�*

loss,l�:<ӯ�       �	x��*Yc�A�*

lossQj�<�He       �	fK�*Yc�A�*

loss ?;��        �	)��*Yc�A�*

loss���=i�       �	P��*Yc�A�*

loss%v^=p0r       �	"�*Yc�A�*

loss�U='tݔ       �	b��*Yc�A�*

lossnbO<�\-C       �	+O�*Yc�A�*

loss�N=���       �	.��*Yc�A�*

lossx�E=Q̒C       �	�v�*Yc�A�*

loss�C�<9�{�       �	��*Yc�A�*

loss��<o���       �	5��*Yc�A�*

loss��+=蘭       �	�d�*Yc�A�*

loss��b=��,�       �	+��*Yc�A�*

loss&7=aX�r       �	2��*Yc�A�*

loss}�=GO#'       �	�3�*Yc�A�*

loss�P�=�	�2       �	���*Yc�A�*

loss��6=��2\       �	�k�*Yc�A�*

loss��0=�'�       �	�*Yc�A�*

lossk�<��)A       �	���*Yc�A�*

loss�g=Y%�Q       �	`<�*Yc�A�*

loss�<��g       �	O��*Yc�A�*

loss=`�0�       �		��*Yc�A�*

loss)F9=�-\D       �	�C�*Yc�A�*

loss4�=��_�       �	-��*Yc�A�*

loss��<ê�       �	i��*Yc�A�*

loss��<qs~       �	�&�*Yc�A�*

loss��<��       �	���*Yc�A�*

loss#�;@J�d       �	{�*Yc�A�*

loss��?<E�v�       �	��*Yc�A�*

loss�J�<��o�       �	)��*Yc�A�*

loss��=����       �	�Q�*Yc�A�*

loss
��=p�_B       �	d��*Yc�A�*

loss���<�6"       �	0��*Yc�A�*

loss?��<�.�       �	)A�*Yc�A�*

lossѾ$<����       �	e��*Yc�A�*

loss���<w��       �	v�*Yc�A�*

loss��=�0��       �	,�*Yc�A�*

loss �=�S�       �	���*Yc�A�*

loss3��<�|��       �	�9�*Yc�A�*

loss��<)�j       �	���*Yc�A�*

lossc=[.��       �	��*Yc�A�*

lossn�t=�3X       �	�I�*Yc�A�*

loss�p�;�_��       �	�5�*Yc�A�*

lossI�g=U�P�       �	���*Yc�A�*

loss�5S<Iw0|       �	Dn�*Yc�A�*

loss�pq=�F��       �	��*Yc�A�*

loss��;�T�       �	��*Yc�A�*

loss� h=4d�?       �	��*Yc�A�*

lossA��<�l��       �	1�*Yc�A�*

loss���<�iw       �	I�*Yc�A�*

loss/C;��i       �	���*Yc�A�*

lossE-�<��       �	���*Yc�A�*

lossM�<�#       �	�D�*Yc�A�*

lossԃu;�#g       �	���*Yc�A�*

loss7�Y=3�"�       �	R}�*Yc�A�*

loss�R=�bP�       �	T�*Yc�A�*

loss�Q=ܶ�       �	��*Yc�A�*

loss�T�<l�t�       �	�N�*Yc�A�*

lossx7;v;�       �	\��*Yc�A�*

loss�;��       �	>x�*Yc�A�*

loss��=k|��       �	��*Yc�A�*

loss�u;�-�)       �	���*Yc�A�*

lossJ�;9�       �	�S�*Yc�A�*

loss�i^=��|�       �	s��*Yc�A�*

lossyG=ɍ=�       �	��*Yc�A�*

loss��f=Ɨ��       �	&��*Yc�A�*

loss��s<d�s       �	���*Yc�A�*

loss�jD:6i�       �	�9�*Yc�A�*

loss��<>Bv       �	���*Yc�A�*

loss3�<+5j5       �	�w�*Yc�A�*

loss�!<��V       �	*��*Yc�A�*

lossL��;��2       �	_y +Yc�A�*

loss�	�<F�4z       �	+Yc�A�*

loss�@=qO�       �	h�+Yc�A�*

loss���<,��       �	gE+Yc�A�*

loss(�<s�hm       �	��+Yc�A�*

loss,ì<�t,o       �	��+Yc�A�*

loss�@i=�$OK       �	��+Yc�A�*

lossM�=�m�       �	Gw+Yc�A�*

loss9��<ފ��       �	+Yc�A�*

loss�w�<V0       �	ӥ+Yc�A�*

loss�1�9�@��       �	�=+Yc�A�*

lossq/<�QC_       �	��+Yc�A�*

lossaϸ<�A�       �	�o+Yc�A�*

loss*G�<��\�       �	U	+Yc�A�*

loss�x�=S�2       �	E�	+Yc�A�*

lossd��<߁�R       �	�S
+Yc�A�*

loss��=�t�       �	��
+Yc�A�*

lossն�<���       �	r�+Yc�A�*

loss g=�=G       �	Y+Yc�A�*

loss���=��Y�       �	�E+Yc�A�*

loss�[=�5�O       �	s�+Yc�A�*

lossU<`�S�       �	b�+Yc�A�*

loss�2�<v=�T       �	+Yc�A�*

lossSS�;���       �	ӿ+Yc�A�*

losst��<G\       �	�9+Yc�A�*

loss@�==$��T       �	7�+Yc�A�*

loss�7=�%E       �	�t+Yc�A�*

loss�i<he2�       �	Z+Yc�A�*

loss.�*<��l       �	��+Yc�A�*

loss�4)<Z�       �	�\+Yc�A�*

loss���=��AC       �	��+Yc�A�*

loss��;=��B       �	*�+Yc�A�*

loss�r>~;4o       �	�j+Yc�A�*

loss?� >� Y       �	�+Yc�A�*

loss��:=��j       �	��+Yc�A�*

loss_�=a���       �	0+Yc�A�*

loss)*=;{��       �	��+Yc�A�*

loss��;b1�       �	�Z+Yc�A�*

loss��<҂��       �	��+Yc�A�*

lossA�m=m""�       �	��+Yc�A�*

lossnb�<Ȥ��       �	�+Yc�A�*

loss
 =H��       �	��+Yc�A�*

lossQ�<R]�       �	�B+Yc�A�*

loss��a<���       �	�+Yc�A�*

loss�k�<	��       �	�k+Yc�A�*

lossx�}<�XJ�       �	�+Yc�A�*

loss�N<b$a�       �	g�+Yc�A�*

loss�-�<$�V�       �	�.+Yc�A�*

loss��=pk~       �	M�+Yc�A�*

loss�b<�M|�       �	�b +Yc�A�*

loss�Ί<��-\       �	�� +Yc�A�*

loss}��<����       �	��!+Yc�A�*

loss�L�=�&��       �	m6"+Yc�A�*

loss�=�]�j       �	��"+Yc�A�*

lossitS='�       �	n#+Yc�A�*

loss�/=�?:[       �	�$+Yc�A�*

loss���<Z��       �	՗$+Yc�A�*

loss�^!<e��       �	+%+Yc�A�*

lossׁi=��
0       �	��%+Yc�A�*

lossZ�p=gi��       �	[_&+Yc�A�*

lossmЙ<��q       �	�&+Yc�A�*

lossE�<�?�       �	5�'+Yc�A�*

loss�K�<��j       �	�E(+Yc�A�*

loss<+��m       �	Q�(+Yc�A�*

loss!.=�8�       �	8�)+Yc�A�*

loss{�=�J       �	�e*+Yc�A�*

lossm'�<.���       �	�++Yc�A�*

loss2H�=#��       �	�A,+Yc�A�*

loss�'
=�zDl       �	!�,+Yc�A�*

loss�ό;yc�&       �	�-+Yc�A�*

loss�;<�d�2       �	�P.+Yc�A�*

loss���<�j�-       �	6�.+Yc�A�*

lossU�<�̨�       �	��/+Yc�A�*

loss #�;5c�x       �	�B0+Yc�A�*

loss��l<���       �	��0+Yc�A�*

loss��<��(�       �	��1+Yc�A�*

loss �<�[V       �	�2+Yc�A�*

lossA��<[�{       �	x�2+Yc�A�*

lossN[(=��D       �	PQ3+Yc�A�*

loss�l�=t�:.       �	��3+Yc�A�*

loss��.=��jW       �	��4+Yc�A�*

lossr��<֙�       �	�S5+Yc�A�*

lossAb�<h�;�       �	��5+Yc�A�*

loss&�<G7�       �	K�6+Yc�A�*

loss3�=����       �	`Z7+Yc�A�*

lossi�N=ӭ�       �	��7+Yc�A�*

loss@�<�҉�       �	і8+Yc�A�*

loss���;���h       �	6:9+Yc�A�*

loss��=�+�k       �	�:+Yc�A�*

lossO=��c       �	`�:+Yc�A�*

loss`=���       �	�W;+Yc�A�*

lossI��=UH�       �	�;+Yc�A�*

loss �<�>�H       �	A�<+Yc�A�*

loss�Ě;���       �	�i=+Yc�A�*

loss�R�=�7�       �	�>+Yc�A�*

lossW��<0��       �	�>+Yc�A�*

lossZ�9;N�y       �	�O?+Yc�A�*

loss�<���       �	��?+Yc�A�*

loss1�;5��       �	��@+Yc�A�*

loss���<���       �	u!A+Yc�A�*

loss*�g=���       �	z�A+Yc�A�*

loss���=p�3�       �	�WB+Yc�A�*

loss�=kwϛ       �	��B+Yc�A�*

loss�f7<�T`'       �	!�C+Yc�A�*

lossl�<�2�m       �	�)D+Yc�A�*

loss�³;jY��       �	b�D+Yc�A�*

loss)�;�rr�       �	�bE+Yc�A�*

lossfB�=RF�A       �	lF+Yc�A�*

loss���:��;,       �	��F+Yc�A�*

lossVv�<��G       �	ӠG+Yc�A�*

loss+s�;2�8�       �	��H+Yc�A�*

loss-��=预'       �	�2I+Yc�A�*

loss�p9<�[�:       �	~�I+Yc�A�*

loss��G=*�_!       �	F|J+Yc�A�*

loss<u�=�Oe       �	K+Yc�A�*

loss�Q2=�zD�       �	y�K+Yc�A�*

loss*w<��x       �	xFL+Yc�A�*

loss�<x
       �	��L+Yc�A�*

loss�C;UH	�       �	taN+Yc�A�*

lossM��<h$"       �	&O+Yc�A�*

loss��<4Ջ       �	؜O+Yc�A�*

loss1��=!Ӝ�       �	;9P+Yc�A�*

loss��q<~� �       �	��P+Yc�A�*

lossRv�=%��F       �	�xQ+Yc�A�*

loss �=�`�       �	aR+Yc�A�*

loss��<N��n       �	İR+Yc�A�*

loss ��<2�'       �	IS+Yc�A�*

loss���=q5D       �	��S+Yc�A�*

lossh2�=8�8       �	�~T+Yc�A�*

loss�*<�t��       �	G!U+Yc�A�*

lossͰ�;C"-2       �	��U+Yc�A�*

lossa��=���e       �	$�V+Yc�A�*

loss>R�<� ��       �	�nW+Yc�A�*

loss���;k�       �	�X+Yc�A�*

lossch�< N(       �	��X+Yc�A�*

losst1N=C'|       �	ADY+Yc�A�*

loss�D=3+�       �	Q�Y+Yc�A�*

loss���=��L�       �	w�Z+Yc�A�*

loss_n=�TQ       �	�%[+Yc�A�*

loss�<o^P       �	��[+Yc�A�*

lossi�R=�~�       �	�\+Yc�A�*

loss�p�=�Г�       �	d]+Yc�A�*

loss�#�<�P��       �	��]+Yc�A�*

loss�W-<�5�       �	�Y^+Yc�A�*

losse�c<���r       �	��^+Yc�A�*

loss���=���       �	D�_+Yc�A�*

loss�,�<U*.�       �	C`+Yc�A�*

loss܉�<럥�       �	Ǻ`+Yc�A�*

loss`��<4k}�       �	�Na+Yc�A�*

lossT�I<~k�       �	��a+Yc�A�*

loss-^�=6�;Z       �	Svb+Yc�A�*

losse� =��͸       �	�c+Yc�A�*

loss6jo=[��M       �	��c+Yc�A�*

lossE�<l�o       �	�Ad+Yc�A�*

loss��X=#m@�       �	��d+Yc�A�*

loss��D<K�+       �	}xe+Yc�A�*

loss��<>`�m       �	sf+Yc�A�*

loss��=㦼q       �	��f+Yc�A�*

lossCU�<~2�       �	�=g+Yc�A�*

loss�e�<aĹ�       �	|�g+Yc�A�*

loss��>�/�y       �	�mh+Yc�A�*

loss��=Ф�I       �	�i+Yc�A�*

loss ]�<E��H       �	6�i+Yc�A�*

loss��=�z�v       �	Jj+Yc�A�*

lossͷ[=$(��       �	?k+Yc�A�*

loss�'�<_�Oj       �	3�k+Yc�A�*

loss�E�;o�       �	��l+Yc�A�*

loss{�&<����       �	܄m+Yc�A�*

loss��H<�?��       �	)n+Yc�A�*

loss���<�Gy�       �	��n+Yc�A�*

loss�E=J���       �	Dko+Yc�A�*

lossϔ�<��L$       �	�
p+Yc�A�*

lossR�;��p       �	e�p+Yc�A�*

loss�R/<+S�       �	�fq+Yc�A�*

loss�
=���       �	�r+Yc�A�*

loss��=��z       �	~�r+Yc�A�*

lossO��<�3       �	�Gs+Yc�A�*

lossf��;4;]       �	��s+Yc�A�*

loss@J�<9       �	��t+Yc�A�*

loss���=�5��       �	u+Yc�A�*

lossow
=��5�       �	t�u+Yc�A�*

loss�}d=�%��       �	�Nv+Yc�A�*

lossP�=�ў�       �	��v+Yc�A�*

loss�Ry<'�qh       �	C�w+Yc�A�*

loss峧<�&�       �	�^x+Yc�A�*

loss��<��!       �	�x+Yc�A�*

lossj��<;�_�       �	�y+Yc�A�*

loss6�6=O��       �	�4z+Yc�A�*

loss�Г<nE2�       �	G�z+Yc�A�*

loss���<�]6       �	�z{+Yc�A�*

lossi�<Yô�       �	Q|+Yc�A�*

loss1��<�ڬ�       �	1�|+Yc�A�*

loss�٤=��q�       �	��}+Yc�A�*

loss�<��c&       �	q~+Yc�A�*

losse]�<U۳9       �	 �~+Yc�A�*

loss���=�y��       �	wM+Yc�A�*

lossi��=��       �	)�+Yc�A�*

loss?�<]6^�       �	Q��+Yc�A�*

loss�:�=�       �	��+Yc�A�*

loss��
=h�|�       �	���+Yc�A�*

loss�Q�;i�8V       �	�P�+Yc�A�*

losso�`=H��s       �	��+Yc�A�*

loss���<DIA       �	���+Yc�A�*

loss� i=zk|       �	��+Yc�A�*

loss�G=V��       �	K��+Yc�A�*

loss��=�.i       �	AD�+Yc�A�*

lossI��<��8�       �	0م+Yc�A�*

loss4{�<�-�       �	"q�+Yc�A�*

losso��<>b�$       �	��+Yc�A�*

loss/�<�Ps�       �	~��+Yc�A�*

lossxnl<��=       �	@�+Yc�A�*

loss�ۏ=g�WO       �	��+Yc�A�*

loss���:� 2       �	:��+Yc�A�*

loss�T�=��       �	\V�+Yc�A�*

loss,6<��:       �	J�+Yc�A�*

loss�r�=�a��       �	6��+Yc�A�*

loss���;Θ�a       �	�'�+Yc�A�*

loss@V�<`ao       �	���+Yc�A�*

loss"Y=����       �	�]�+Yc�A�*

loss��<��z�       �	���+Yc�A�*

loss�}=:IU1       �	���+Yc�A�*

loss��D<�&��       �	|(�+Yc�A�*

loss��=����       �	���+Yc�A�*

loss��=�24       �	W`�+Yc�A�*

loss���=Ht�       �	���+Yc�A�*

loss��v=V��[       �	Ő�+Yc�A�*

loss�<8�f       �	�*�+Yc�A�*

loss|_�=�*�R       �	?Œ+Yc�A�*

lossa��=D]V$       �	_�+Yc�A�*

loss�X�=c�1       �	<��+Yc�A�*

loss2�==q�{       �	���+Yc�A�*

lossHg�=�+��       �	2�+Yc�A�*

lossj\�<P	�(       �	�ȕ+Yc�A�*

loss�;�`�       �	f�+Yc�A�*

loss�f�<?Ѵ       �	v��+Yc�A�*

lossI=�X�       �	�ۗ+Yc�A�*

loss�cu<g�m�       �	�v�+Yc�A�*

loss�=&m {       �	`�+Yc�A�*

loss���;k�X�       �	��+Yc�A�*

loss���=�:l�       �	�o�+Yc�A�*

loss1;=UE       �	��+Yc�A�*

loss��1=Y��       �	t��+Yc�A�*

loss�Y=�P�:       �	�R�+Yc�A�*

loss���<R$�       �	��+Yc�A�*

loss�Y�=��Ё       �	���+Yc�A�*

loss��<b�       �	WB�+Yc�A�*

loss�F<���       �	�ݞ+Yc�A�*

loss�=
K��       �	��+Yc�A�*

loss�=i�       �	�+Yc�A�*

loss<����       �	��+Yc�A�*

loss��-=�7�       �	�f�+Yc�A�*

loss�jH=G���       �	��+Yc�A�*

loss'<���       �	ͮ�+Yc�A�*

loss��<�o�       �	�S�+Yc�A�*

loss��#<WΉ�       �	n��+Yc�A�*

loss��<n]�        �	��+Yc�A�*

loss4Ve=��B       �	�L�+Yc�A�*

loss��=i�       �	k�+Yc�A�*

loss`�<1���       �	[��+Yc�A�*

loss�į=2��w       �	�=�+Yc�A�*

loss�<,�~F       �	��+Yc�A�*

loss�(=xz��       �	���+Yc�A�*

loss��=P�T\       �	S$�+Yc�A�*

loss�e�<�M�       �	�Щ+Yc�A�*

loss\Y<��OA       �	�}�+Yc�A�*

loss͌�<���       �	.�+Yc�A�*

loss��D=pV�n       �	���+Yc�A�*

loss
w<��W�       �	NԬ+Yc�A�*

lossr�7;Q�(U       �	l��+Yc�A�*

loss��I=��Z�       �	8��+Yc�A�*

loss�C=��       �	�]�+Yc�A�*

loss� �<6��v       �	��+Yc�A�*

loss�=2=�?�O       �	�ư+Yc�A�*

loss�]�<���{       �	��+Yc�A�*

loss<�f<�L��       �	�+Yc�A�*

loss/�=� ܩ       �	!ʲ+Yc�A�*

lossN�K<��M       �	�g�+Yc�A�*

lossr�B<�6��       �	��+Yc�A�*

loss�9�<�'s-       �	���+Yc�A�*

loss5��=�JA�       �	�=�+Yc�A�*

lossLY<�Ad       �	׵+Yc�A�*

lossf��<�2JG       �	o�+Yc�A�*

lossW�<^p��       �	y�+Yc�A�*

loss�ݿ;��S       �	D�+Yc�A�*

loss&��=��/�       �	yx�+Yc�A�*

loss�-';�W�       �	u �+Yc�A�*

loss��#>"���       �	A��+Yc�A�*

loss8�`<�˙�       �	�_�+Yc�A�*

loss�60<�8)\       �	��+Yc�A�*

loss��<>ށ��       �	Z��+Yc�A�*

loss4��;ġ^�       �	�9�+Yc�A�*

loss�d5=���       �	��+Yc�A�*

loss�x;쓤s       �	ۦ�+Yc�A�*

lossѝ=r�]       �	�D�+Yc�A�*

loss([V<����       �	�ܾ+Yc�A�*

lossd�q=�\�|       �	�w�+Yc�A�*

lossv��=�Q�W       �	��+Yc�A�*

loss���=�r��       �	ڭ�+Yc�A�*

loss�p<c�v�       �	�D�+Yc�A�*

lossA��=��U�       �	���+Yc�A�*

loss��=Pr�       �	�u�+Yc�A�*

loss���<�o��       �	��+Yc�A�*

lossj+K<F��Y       �	��+Yc�A�*

loss�>��&�       �	B�+Yc�A�*

lossV<F<v�       �	'��+Yc�A�*

loss1��<8�n       �	io�+Yc�A�*

loss�T<�,��       �	��+Yc�A�*

loss��=t+��       �	��+Yc�A�*

lossM�@<=(�G       �	[B�+Yc�A�*

lossDE)=QmR�       �	G��+Yc�A�*

loss/�=��g       �	��+Yc�A�*

loss��=]*�U       �	��+Yc�A�*

loss�:�=~?Q       �	���+Yc�A�*

loss>��<.���       �	ep�+Yc�A�*

lossC^�<c&n       �	�	�+Yc�A�*

loss���=AՅ�       �	��+Yc�A�*

lossډ�<���       �	v7�+Yc�A�*

lossJ)^<ԾPk       �	���+Yc�A�*

lossT��;���       �	�`�+Yc�A�*

lossfҕ< �Nj       �	���+Yc�A�*

lossI;={�2�       �	���+Yc�A�*

loss�!�<K5�       �	�7�+Yc�A�*

loss��<�?�       �	���+Yc�A�*

loss��K=Դ:       �	Eg�+Yc�A�*

loss�&�<Q���       �	��+Yc�A�*

loss�)=�!��       �	,��+Yc�A�*

loss*z�;7�ub       �	�5�+Yc�A�*

loss�>=v�       �	���+Yc�A�*

loss<%<���       �	%w�+Yc�A�*

loss�;=YB�       �	�
�+Yc�A�*

lossa�=g{��       �	���+Yc�A�*

loss��<=��~       �	�U�+Yc�A�*

loss2pE={(��       �	i��+Yc�A�*

loss�K=���       �	b��+Yc�A�*

loss��<Z���       �	�?�+Yc�A�*

loss4Ln=�3"x       �	"��+Yc�A�*

loss8�=�u       �	���+Yc�A�*

loss-�=��       �	�N�+Yc�A�*

lossj�;�T�/       �	��+Yc�A�*

loss}�=5
�       �	���+Yc�A�*

loss�^=�j       �	�a�+Yc�A�*

lossx�;��Iq       �	!�+Yc�A�*

loss�<!Y֕       �	Ω�+Yc�A�*

loss q<��I>       �	�M�+Yc�A�*

loss|�C=E���       �	,��+Yc�A�*

loss�m<�/�       �	���+Yc�A�*

loss�X<���!       �	�+Yc�A�*

loss3��<���       �	մ�+Yc�A�*

loss�C;��,�       �	�^�+Yc�A�*

loss�:[�mA       �	#��+Yc�A�*

loss�>�:�v��       �	���+Yc�A�*

loss.��;5W(       �	<k�+Yc�A�*

loss\{�;�In�       �	}	�+Yc�A�*

loss
�<��       �	t��+Yc�A�*

loss�2�;�ED       �	U�+Yc�A�*

lossM�-=��VA       �	���+Yc�A�*

lossB�:����       �	���+Yc�A�*

lossG�	:^       �	�/�+Yc�A�*

loss�0;�#�       �	%��+Yc�A�*

loss�H';�d?       �	Ox�+Yc�A�*

loss�:=���F       �	�5�+Yc�A�*

loss7Q	<��       �	���+Yc�A�*

loss�iD;PG��       �	Ӡ�+Yc�A�*

loss�h�=bqL�       �	�9�+Yc�A�*

losso��=h�e�       �	���+Yc�A�*

loss�Ch;2�=       �	]k�+Yc�A�*

loss�b>şc�       �	�!�+Yc�A�*

lossn4�<g�9Q       �	ظ�+Yc�A�*

lossV"�=^m��       �	�P�+Yc�A�*

loss̳�<�B�       �	i��+Yc�A�*

loss� <�,g�       �	�}�+Yc�A�*

lossq�>q���       �	��+Yc�A�*

lossq7=�
�       �	���+Yc�A�*

loss�� =�x<       �	tD�+Yc�A�*

loss=�=\��n       �	���+Yc�A�*

loss�R�;� �%       �	[��+Yc�A�*

loss ��=ڠ�[       �	0�+Yc�A�*

loss/4�=��C]       �	���+Yc�A�*

lossDaZ=���       �	1a�+Yc�A�*

losss�/<L�nY       �	��+Yc�A�*

loss�c=3VԶ       �	=��+Yc�A�*

loss�/<:(U�       �	:>�+Yc�A�*

loss�4<=�*       �	���+Yc�A�*

lossa��=h�V�       �	���+Yc�A�*

lossJ-2<����       �	3�+Yc�A�*

loss`�<���c       �	��+Yc�A�*

loss���<��       �	MH�+Yc�A�*

loss�<1���       �	@��+Yc�A�*

lossv0k<�C�p       �	�t�+Yc�A�*

loss�d<Ϩ�       �	9�+Yc�A�*

loss#�;�� �       �	+��+Yc�A�*

loss��l=�*�a       �	�E�+Yc�A�*

loss`f=�=a       �	6��+Yc�A�*

loss�7�=����       �	�y�+Yc�A�*

loss[�}=O_       �	��+Yc�A�*

loss��<��N       �	$��+Yc�A�*

loss��M=���D       �	�M�+Yc�A�*

loss��;%nL@       �	��+Yc�A�*

lossn/4<�}-�       �	��+Yc�A�*

loss���;}�*�       �	v ,Yc�A�*

loss�O�<rF�       �	� ,Yc�A�*

lossN}:</�Z�       �	|`,Yc�A�*

lossv1`<�FW       �	�,Yc�A�*

loss�5�=r�       �	�,Yc�A�*

loss�˙<tJ       �	�,Yc�A�*

lossR�;���;       �	�S,Yc�A�*

loss���<VJ�       �	5�,Yc�A�*

loss.�=k��P       �	��,Yc�A�*

loss3S�;t�~H       �	�#,Yc�A�*

lossM�;��=       �	��,Yc�A�*

loss�S�<�DM       �	DR,Yc�A�*

loss;h=���       �	!�,Yc�A�*

losslJ�;"��(       �	e�,Yc�A�*

loss���<��HP       �	�A	,Yc�A�*

loss'��;#9Ge       �	s�	,Yc�A�*

loss<$d<Xv��       �	�t
,Yc�A�*

loss��=	���       �	�.$,Yc�A�*

losseԂ=;��G       �	��$,Yc�A�*

lossWXu=�%[h       �	�b%,Yc�A�*

lossr�T<�#3       �	� &,Yc�A�*

loss��=�,�       �	�&,Yc�A�*

lossj(*<@&{       �	�_',Yc�A�*

loss%��<d~YZ       �	l(,Yc�A�*

lossLG�<�r�       �	4�(,Yc�A�*

loss�=�
9�       �	�<),Yc�A�*

lossp�=ֆ��       �	�),Yc�A�*

lossz*�<ٓ�       �	A�*,Yc�A�*

lossa��<�(	�       �	in+,Yc�A�*

loss��"=����       �	�,,Yc�A�*

loss�x<w@$       �	�,,Yc�A�*

loss�
�<�>       �	�-,Yc�A�*

loss�Ҭ<rM)�       �	e.,Yc�A�*

loss�;���       �	8�.,Yc�A�*

loss]�=o�j�       �	�\/,Yc�A�*

lossN�<���T       �	��/,Yc�A�*

lossѸI=����       �	��0,Yc�A�*

loss�#?<[��N       �	�@1,Yc�A�*

loss��=%k,       �	�1,Yc�A�*

loss�E<�S	       �	~t2,Yc�A�*

loss�F�=��t       �	�3,Yc�A�*

loss���<���       �	<�3,Yc�A�*

lossr��;�sZ       �	�[4,Yc�A�*

loss쉄<�,��       �	7�4,Yc�A�*

lossŧ�<�Qj:       �	؜5,Yc�A�*

loss��X<�߫�       �	y=6,Yc�A�*

loss�M�<��ͪ       �	P�6,Yc�A�*

loss�f7=޴G_       �	�7,Yc�A�*

loss��<Ӽ�       �	78,Yc�A�*

loss��*=�s�       �	��8,Yc�A�*

loss&��=T�+       �	�p9,Yc�A�*

loss0�=ڃ�       �	8:,Yc�A�*

lossv��=�=+�       �	:�:,Yc�A�*

loss.�O;�&`L       �	�L;,Yc�A�*

loss�>�<!�/M       �	��;,Yc�A�*

lossM<>�D��       �	�|<,Yc�A�*

loss�ȶ<��       �	�=,Yc�A�*

loss��<�U��       �	Ū=,Yc�A�*

lossIM�<,�&�       �	U�>,Yc�A�*

lossM��;%U�a       �	�"?,Yc�A�*

loss��<{�n�       �	a�?,Yc�A�*

lossc��<>Iwy       �	a@,Yc�A�*

loss6�5=��}       �	T�@,Yc�A�*

loss�=m��       �	�A,Yc�A�*

loss�C4=ը��       �	�:B,Yc�A�*

lossNJ�<���       �	��B,Yc�A�*

lossvd�<U`��       �	�C,Yc�A�*

loss��;p���       �	r4D,Yc�A�*

lossL��<�=!1       �	R�D,Yc�A�*

loss�.C<��m       �	\sE,Yc�A�*

loss��=���       �	vF,Yc�A�*

loss]�-=R��a       �	5�F,Yc�A�*

loss	 �:�(�m       �	��G,Yc�A�*

loss?�_:�|͗       �	�H,Yc�A�*

loss`��;HD�       �	�OI,Yc�A�*

loss7S=�G�       �	��I,Yc�A�*

loss;�.=Ȫ(       �	׈J,Yc�A�*

loss�b	>C��       �	�-K,Yc�A�*

loss|�<<�2w       �	��K,Yc�A�*

loss�(<���6       �	usL,Yc�A�*

loss�$=kY�Z       �	M,Yc�A�*

loss��=z<0%       �	�M,Yc�A�*

lossj�;�7�       �	�SN,Yc�A�*

loss��f=Nk¢       �	N�N,Yc�A�*

loss�M�<g(�       �	&�O,Yc�A�*

loss�t=��T�       �	%P,Yc�A�*

loss7�U=�x>�       �	T;Q,Yc�A�*

lossa#�<�V�       �	��Q,Yc�A�*

loss�k�<@��f       �	�uR,Yc�A�*

lossAy�<�2�       �	�S,Yc�A�*

lossέ==B�J       �	�S,Yc�A�*

lossh�3=w���       �	D�T,Yc�A�*

lossı<�Z�E       �	�!U,Yc�A�*

lossv�(<󃻉       �	�U,Yc�A�*

loss߃q<�I�e       �	�RV,Yc�A�*

loss��T<?�r       �	��V,Yc�A�*

loss�X3<2in       �	�W,Yc�A�*

lossnQ�<�V��       �	XX,Yc�A�*

losshs[<I��'       �	-�X,Yc�A�*

loss��8=��T�       �	JY,Yc�A�*

loss�Y�;"f�$       �	 �Y,Yc�A�*

loss&�<�먗       �	Z�Z,Yc�A�*

loss(�<�5��       �	B%[,Yc�A�*

loss*�t=�       �	.�[,Yc�A�*

losss�=/�w�       �	�a\,Yc�A�*

loss�~�<wA�T       �		],Yc�A�*

loss=�<�P>�       �	��],Yc�A�*

loss�==�]�p       �	�R^,Yc�A�*

loss(>=�!u       �	 �^,Yc�A�*

loss3�;sY��       �	�_,Yc�A�*

loss��<z�k�       �	�!`,Yc�A�*

loss�j=E�x�       �	��`,Yc�A�*

lossIf=<5<�       �	Oa,Yc�A�*

loss�?>���       �	��a,Yc�A�*

lossĒd<�9�B       �	Vb,Yc�A�*

lossm�=)^��       �	�c,Yc�A�*

loss�#�;-��       �	��c,Yc�A�*

loss�<̠�       �	oGd,Yc�A�*

loss��:7���       �	.e,Yc�A�*

lossA��=��y�       �	
f,Yc�A�*

loss��<*�hq       �	��f,Yc�A�*

loss�4�<�;TZ       �	(�g,Yc�A�*

loss3܃=�r�y       �	T8h,Yc�A�*

loss��<�y�,       �	��h,Yc�A�*

lossp{<�~c%       �	�i,Yc�A�*

lossE%Y=��TA       �	(j,Yc�A�*

loss��==�x       �	��j,Yc�A�*

loss	F=rl�K       �	Bvk,Yc�A�*

loss���<7S�y       �	�sl,Yc�A�*

loss�t<����       �	�^m,Yc�A�*

lossu�<a�]�       �	��n,Yc�A�*

loss���;���       �	��o,Yc�A�*

loss��;��{       �	�5p,Yc�A�*

loss�3T<��T�       �	��p,Yc�A�*

loss��=]P�       �	p�q,Yc�A�*

loss!��<ko�       �	�>r,Yc�A�*

loss8�;7��       �	�s,Yc�A�*

lossz2=�D��       �	-�s,Yc�A�*

loss��<�_r�       �	\Ut,Yc�A�*

lossZ�=��       �	�
u,Yc�A�*

loss�`�;qan       �	�u,Yc�A�*

loss��=�I       �	 ]v,Yc�A�*

loss�s;=K�֖       �		�v,Yc�A�*

lossO��<��\?       �	Χw,Yc�A�*

losso�I;Y�       �	x_x,Yc�A�*

loss =L�       �	�y,Yc�A�*

lossWk==���       �	��y,Yc�A�*

loss���;u��@       �	sdz,Yc�A�*

loss� S=��df       �	'{,Yc�A�*

loss��;���       �	��{,Yc�A�*

lossEP�;+�Ng       �	�h|,Yc�A�*

loss���<�5'       �	�},Yc�A�*

loss�=@<�G>}       �	��},Yc�A�*

loss��<�?T        �	o�~,Yc�A�*

loss n�<+C��       �	�O,Yc�A�*

loss�q1<��|�       �	�,Yc�A�*

loss��;;WfI;       �	��,Yc�A�*

lossfv;�M�9       �	�D�,Yc�A�*

lossE9=
G��       �	J�,Yc�A�*

loss�{�<{rP�       �	΍�,Yc�A�*

loss\��=%nT]       �	g,�,Yc�A�*

lossF+�=��֫       �	Cǃ,Yc�A�*

loss�Z�<�w�l       �	�b�,Yc�A�*

loss��<9�F:       �	���,Yc�A�*

loss�+�:�W7       �	\��,Yc�A�*

loss��m<��>}       �	�+�,Yc�A�*

loss]��=��8�       �	]ņ,Yc�A�*

loss�S<d�0       �	�q�,Yc�A�*

loss���;�Vi       �	N�,Yc�A�*

lossi86=إ`       �	���,Yc�A�*

loss�j8=�2�,       �	o�,Yc�A�*

loss���=�\�=       �	�@�,Yc�A�*

loss]`�:��       �	8ي,Yc�A�*

loss���<�Ŋu       �	�v�,Yc�A�*

loss��.<J?ҙ       �	��,Yc�A�*

loss���=���       �	ɰ�,Yc�A�*

lossA~;�F4�       �	#L�,Yc�A�*

lossq�<�͐7       �	0��,Yc�A�*

losssBc;�ms�       �	Й�,Yc�A�*

lossΰ�<�6       �	�>�,Yc�A�*

loss@4�:���       �	�ؐ,Yc�A�*

loss�-�<g��)       �	�{�,Yc�A�*

lossd��<����       �	(�,Yc�A�*

loss�X<i��       �	�˒,Yc�A�*

loss��<x��~       �	+l�,Yc�A�*

loss���<UJ/b       �	t�,Yc�A�*

loss��;���       �	<��,Yc�A�*

loss���<���       �	�>�,Yc�A�*

loss���<�r4�       �	i�,Yc�A�*

loss�k;�*�       �	��,Yc�A�*

loss|	=�?ť       �	��,Yc�A�*

loss���;���%       �	毗,Yc�A�*

loss���<�h��       �	�I�,Yc�A�*

loss��1=����       �	i�,Yc�A�*

loss�M=�:�       �	�|�,Yc�A�*

loss% �<��B8       �	�,Yc�A�*

lossx��:��       �	$��,Yc�A�*

lossR':�j�       �	�I�,Yc�A�*

loss�|F<9��.       �	"ߛ,Yc�A�*

loss2��<���q       �	�v�,Yc�A�*

loss�;Oq��       �	��,Yc�A�*

loss�L�;��@       �	��,Yc�A�*

loss�ZL;xU:�       �	�L�,Yc�A�*

lossx�e=F]p`       �	�,Yc�A�*

loss��<���j       �	��,Yc�A�*

loss��<�K�C       �	U��,Yc�A�*

lossA�<˱o�       �	W=�,Yc�A�*

lossX|�=+a%k       �	sס,Yc�A�*

loss��=�ျ       �	r�,Yc�A�*

loss]z<<8�       �	}	�,Yc�A�*

loss�'�;�}�       �	Φ�,Yc�A�*

lossQ�;��ӻ       �	B�,Yc�A�*

loss�C=���G       �	�ݤ,Yc�A�*

loss�P<�Ƕ       �	�{�,Yc�A�*

loss�<��y       �	��,Yc�A�*

lossd'�<��`P       �	Y��,Yc�A�*

loss�D<�7hA       �	�z�,Yc�A�*

lossA�<���       �	.�,Yc�A�*

loss��V<�d+�       �	�Ψ,Yc�A�*

loss�7O=T�       �	/k�,Yc�A�*

loss-�D=�q       �	��,Yc�A�*

lossCm;u�       �	���,Yc�A�*

loss&�v<�Ln       �	Uk�,Yc�A�*

loss���=�8��       �	��,Yc�A�*

loss�<
I       �	"¬,Yc�A�*

loss&�I<�/       �	��,Yc�A�*

loss@� <���       �	m�,Yc�A�*

loss�/<��       �	�7�,Yc�A�*

loss�x�;��C�       �	1ѯ,Yc�A�*

lossC�<P�*       �		o�,Yc�A�*

lossA�<�X?�       �	��,Yc�A�*

loss��<e�%�       �	���,Yc�A�*

loss�\�=K�       �	�>�,Yc�A�*

loss`�=-�       �	 �,Yc�A�*

loss>wk�       �	t�,Yc�A�*

loss���=0       �	��,Yc�A�*

loss���<�z'       �	,Yc�A�*

loss��<NJ�       �	 W�,Yc�A�*

lossJ��<z��N       �	F�,Yc�A�*

loss�<2�       �	T��,Yc�A�*

loss�/�<B�5�       �	<1�,Yc�A�*

lossE<�_1D       �	�ӷ,Yc�A�*

loss�h�<ρ�2       �	\v�,Yc�A�*

lossT<�cʂ       �	�*�,Yc�A�*

loss���<u`'       �	xϹ,Yc�A�*

loss�;��"�       �	�v�,Yc�A�*

loss;�<�t�       �	��,Yc�A�*

lossE�A<~V�       �	W��,Yc�A�*

losss�u;C�+z       �	�[�,Yc�A�*

loss&�'=a�       �	��,Yc�A�*

loss���<����       �	��,Yc�A�*

losswc�=����       �	�Q�,Yc�A�*

loss�4�<,�Q�       �	D��,Yc�A�*

loss�<����       �	���,Yc�A�*

loss�ټ<��RY       �	�E�,Yc�A�*

loss�؇=5�c�       �	p��,Yc�A�*

loss!��:���o       �	���,Yc�A�*

loss�>�:\�       �	��,Yc�A�*

loss#=>�       �	���,Yc�A�*

loss*�<�*       �	0f�,Yc�A�*

loss��	=�F�       �	y�,Yc�A�*

lossHY�<2�       �	#��,Yc�A�*

loss�1�=��e       �	D�,Yc�A�*

loss{��:�ǣ�       �	2��,Yc�A�*

loss��<Ζ�       �	�{�,Yc�A�*

loss�=}�8k       �	y �,Yc�A�*

lossb�<��Z�       �	p(�,Yc�A�*

loss���<��<�       �	���,Yc�A�*

loss,��;E�nY       �	
e�,Yc�A�*

loss��=�t�       �	�#�,Yc�A�*

loss ;��Xf       �	���,Yc�A�*

lossieU;�+$       �	�x�,Yc�A�*

loss�Uq<����       �	��,Yc�A�*

loss^X<��'"       �	W��,Yc�A�*

loss9$<���2       �	O�,Yc�A�*

loss4�&<�       �	���,Yc�A�*

loss��D=�Fc�       �	,�,Yc�A�*

loss[��<�|�       �	��,Yc�A�*

loss:�;�ti       �	���,Yc�A�*

lossPw=�C8       �	�c�,Yc�A�*

lossq�<�{       �	z��,Yc�A�*

loss�u=��S�       �	-��,Yc�A�*

loss��1<�;t~       �	)Z�,Yc�A�*

loss��<��       �	���,Yc�A�*

loss�Ȳ<^Me�       �	���,Yc�A�*

loss�^x=U�       �	?8�,Yc�A�*

loss�=�{��       �	���,Yc�A�*

loss��e<θ�T       �	�e�,Yc�A�*

loss��;�,�T       �	m�,Yc�A�*

loss�e=�,�       �	v��,Yc�A�*

loss�=�n�       �	�<�,Yc�A�*

lossN�=��       �	��,Yc�A�*

loss��%= ���       �	�i�,Yc�A�*

lossf�}<P[��       �	y�,Yc�A�*

loss�@�;1\        �	(��,Yc�A�*

loss���<�1�       �	�7�,Yc�A�*

loss3"�;�R       �	)��,Yc�A�*

lossX�d<��       �	�p�,Yc�A�*

lossjγ<)��       �	�	�,Yc�A�*

lossڛ�<#mb�       �	��,Yc�A�*

loss�l=ٰ2�       �	d=�,Yc�A�*

loss��=�8�       �	s��,Yc�A�*

loss/L=`���       �	���,Yc�A�*

loss��<ێ�       �	��,Yc�A�*

loss:�=��#       �	.�,Yc�A�*

loss:4�:l       �	��,Yc�A�*

loss	h�;>��f       �	u9�,Yc�A�*

lossw��<��<A       �	��,Yc�A�*

loss8�l<�5       �	-|�,Yc�A�*

loss�a=��.       �	]�,Yc�A�*

lossRp;��A7       �	���,Yc�A�*

loss��S;�[       �	�O�,Yc�A�*

lossޑ=����       �	���,Yc�A�*

loss&��;t�	       �	"��,Yc�A�*

loss�)�<�Т       �	-�,Yc�A�*

loss͛�=6(:       �	.��,Yc�A�*

lossZ��<�E>       �	:��,Yc�A�*

loss���<�)�n       �	7R�,Yc�A�*

loss�s=�gon       �	���,Yc�A�*

loss
�`;3aP�       �	���,Yc�A�*

loss�Q=���       �	��,Yc�A�*

loss�N=5��       �	���,Yc�A�*

loss]ܫ=�ƻ�       �	iU�,Yc�A�*

loss��;@ t�       �	��,Yc�A�*

lossDN=s}�0       �	��,Yc�A�*

loss��;w!�;       �	D�,Yc�A�*

loss*<4= �Uz       �	g
�,Yc�A�*

loss}�=
�é       �	��,Yc�A�*

lossF<���T       �	��,Yc�A�*

loss�w=IU��       �	��,Yc�A�*

loss�"�<D���       �	Ĵ�,Yc�A�*

loss���<bad       �	�d�,Yc�A�*

loss_�=ޚip       �	��,Yc�A�*

loss�"?;-�/>       �	ҏ�,Yc�A�*

loss���<����       �	U.�,Yc�A�*

lossb�=`Q{H       �	���,Yc�A�*

lossn��<�5�Z       �	�l�,Yc�A�*

lossO�6=B1.�       �	u�,Yc�A�*

loss��2=��>       �	ʣ�,Yc�A�*

lossq�a<t���       �		P�,Yc�A�*

loss�&�;g�/#       �	���,Yc�A�*

loss���=9���       �	r��,Yc�A�*

loss.�=�|F       �	�,Yc�A�*

lossT�><��Lt       �	c��,Yc�A�*

loss7�	<Dᘳ       �	N�,Yc�A�*

loss�B<C��o       �	���,Yc�A�*

loss��=>�?<       �	$}�,Yc�A�*

loss���;+�       �	U�,Yc�A�*

lossl��<ސJ�       �	���,Yc�A�*

loss���< -h       �	���,Yc�A�*

loss�̠<: �5       �	�4�,Yc�A�*

loss�a=27J�       �	���,Yc�A�*

lossf�M=Ƈ'R       �	�l�,Yc�A�*

loss.��=�T`e       �	� -Yc�A�*

loss��:�=�       �	� -Yc�A�*

loss�2�<���       �	}<-Yc�A�*

loss(M�;}��s       �	^�-Yc�A�*

loss[� <[C�>       �	�z-Yc�A�*

loss�ř;* �       �	�-Yc�A�*

loss�ۂ=�Z��       �	�-Yc�A�*

lossH�<金�       �	�D-Yc�A�*

loss�]�=�A}�       �	�-Yc�A�*

lossl��=���       �	��-Yc�A�*

loss��<$x�q       �	D-Yc�A�*

loss ��<S$�       �	��-Yc�A�*

loss{�R=R���       �	{-Yc�A�*

lossL�]<�~�j       �	C-Yc�A�*

loss8@;V&1       �	E�-Yc�A�*

loss*r�=��       �	��	-Yc�A�*

loss���=-	��       �	^/
-Yc�A�*

loss`��<�*�(       �	v�
-Yc�A�*

loss��;=9vNc       �	�[-Yc�A�*

loss��<���       �	t#-Yc�A�*

loss��%=D��       �	�-Yc�A�*

lossE3=��yv       �	�]-Yc�A�*

lossQ�<����       �	Y�-Yc�A�*

lossr(j<�M�\       �	}�-Yc�A�*

loss6�=��       �	AF-Yc�A�*

loss��<���       �	�-Yc�A�*

lossS��<f�p�       �	��-Yc�A�*

loss�<�`t�       �	i-Yc�A�*

loss/N=.x�       �	'�-Yc�A�*

loss�=���!       �	Ɏ-Yc�A�*

loss�F=�=��       �	�(-Yc�A�*

loss8y2<�:=
       �	;�-Yc�A�*

loss���;���5       �	�Y-Yc�A�*

loss��;��s       �	w�-Yc�A�*

loss3�{<��m�       �	��-Yc�A�*

lossJ�<p �       �	"n-Yc�A�*

loss�/S<Y�*       �	�-Yc�A�*

loss�x�;��}       �	�-Yc�A�*

loss�@*=�f�       �	m:-Yc�A�*

loss_�[<�.��       �	/�-Yc�A�*

loss&�=
�0       �	��-Yc�A�*

loss��=��Z�       �	$*-Yc�A�*

loss(1=��u�       �	��-Yc�A�*

loss�'�;��EJ       �	��-Yc�A�*

lossZg<��C       �	G-Yc�A�*

loss�'=��%�       �	��-Yc�A�*

loss���=F?;       �	�~-Yc�A�*

loss��E=Mk�        �	-Yc�A�*

loss�u�:2(�Y       �	G�-Yc�A�*

lossq��;����       �	W|-Yc�A�*

lossqIC=Z��m       �	� -Yc�A�*

lossO@=Csn�       �	d� -Yc�A�*

loss�	�<"���       �	Ad!-Yc�A�*

loss�
�:�i8�       �	7�!-Yc�A�*

loss�$=�<�       �	D�"-Yc�A�*

lossv�D<߷V       �	O=#-Yc�A�*

lossW�}<S�Y       �	��#-Yc�A�*

loss��d=��	       �	�~$-Yc�A�*

loss��<�
�       �	�%-Yc�A�*

loss6L=�.4:       �	�%-Yc�A�*

loss�]�:�>��       �	t{&-Yc�A�*

loss(�F=�X�       �	#'-Yc�A�*

losso�l<V�h       �	ͮ'-Yc�A�*

loss���<��       �	N(-Yc�A�*

lossEaP<��-       �	J�(-Yc�A�*

loss�EU<���G       �	��)-Yc�A�*

loss�H�<�	=       �	Zg*-Yc�A�*

loss|�H=TA3j       �	x+-Yc�A�*

loss�e=�9�       �	U�+-Yc�A�*

losswA�=b��       �	�E,-Yc�A�*

lossI18=~q�       �	G�,-Yc�A�*

loss���<dF��       �	��--Yc�A�*

loss])�<(��S       �	�F.-Yc�A�*

loss���<
��       �	�.-Yc�A�*

losss��<�@f       �	�z/-Yc�A�*

loss�`B={�:       �	�0-Yc�A�*

loss8]�<r�k
       �	ڭ0-Yc�A�*

loss��<���I       �	@K1-Yc�A�*

loss���=���       �	B�1-Yc�A�*

lossE.�<c�p       �	��2-Yc�A�*

lossOg;/�       �	+L3-Yc�A�*

loss[��<���       �	g�3-Yc�A�*

loss�#�<�GM       �	&�4-Yc�A�*

loss*��=��ڛ       �	'5-Yc�A�*

loss�wD=�/}b       �	˿5-Yc�A�*

loss��=���       �	�\6-Yc�A�*

lossF�3=SY��       �	n�6-Yc�A�*

loss o�=       �	��7-Yc�A�*

loss}��<�F�       �	�+8-Yc�A�*

loss
ʱ=���E       �	z�8-Yc�A�*

loss�1�<�Ͳf       �	�\9-Yc�A�*

lossڛ=2��       �	b�9-Yc�A�*

loss��=< !��       �	.�:-Yc�A�*

loss�:�<�?�8       �	r1;-Yc�A�*

loss~�=#�./       �	:�;-Yc�A�*

lossv>\=Y       �	�g<-Yc�A�*

lossR=�;�]M�       �	�=-Yc�A�*

loss���<[�       �	�=-Yc�A�*

losss�=V��       �	=D>-Yc�A�*

lossA�H<�R       �	��>-Yc�A�*

lossF��;���       �	i�?-Yc�A�*

lossJx�<��i       �	�0@-Yc�A�*

loss)�$<ٚ��       �	��@-Yc�A�*

loss��)<��        �	�A-Yc�A�*

loss$��<]��       �	!#B-Yc�A�*

loss���=O�f\       �	��B-Yc�A�*

lossfF�<"Pp�       �	1zC-Yc�A�*

loss��D=�.~�       �	;D-Yc�A�*

lossC��<.[%       �	�D-Yc�A�*

lossĬ<�e�5       �	HOE-Yc�A�*

loss�L�=y���       �	��E-Yc�A�*

loss��7<0�I�       �	M�F-Yc�A�*

loss�i"<}ϲ�       �	�"G-Yc�A�*

loss��=���O       �	�G-Yc�A�*

loss��M=�N�       �	)�H-Yc�A�*

loss|[�:��t       �	-I-Yc�A�*

loss*�;�ܫ�       �	�I-Yc�A�*

lossl)=A�       �	��J-Yc�A�*

loss���=�?�       �	�3K-Yc�A�*

lossn�=�X�v       �	s�K-Yc�A�*

lossN<�<�!�       �	R�L-Yc�A�*

loss�6<ݸ��       �	�N-Yc�A�*

loss7c<*�@�       �	��N-Yc�A�*

lossC�=u]�       �	��O-Yc�A�*

loss K='c
�       �	�:P-Yc�A�*

loss�8@;7s��       �	,JQ-Yc�A�*

loss��:��l       �	-�Q-Yc�A�*

loss섵<̮��       �	ŪR-Yc�A�*

lossd�=��מ       �	oIS-Yc�A�*

loss6A�;���       �	�S-Yc�A�*

loss��;�b��       �	��T-Yc�A�*

lossac�<J 8       �	u!U-Yc�A�*

lossiD<�(��       �	��U-Yc�A�*

loss1��;�B�       �	�SV-Yc�A�*

loss���=g��       �	� W-Yc�A�*

loss�<&�Hr       �	��W-Yc�A�*

loss�JZ<��.	       �	�/X-Yc�A�*

lossx�>=�       �	u�X-Yc�A�*

lossj�\<^��       �	5_Y-Yc�A�*

loss�=�T�       �	��Y-Yc�A�*

loss��E<T#�       �	�Z-Yc�A�*

lossd�=�,�       �	�.[-Yc�A�*

loss��<�x��       �	�[-Yc�A�*

loss}�V>[���       �	�~\-Yc�A�*

loss���=f��       �	�]-Yc�A�*

loss��<m8�       �	W�]-Yc�A�*

lossTM�<0��       �	�O^-Yc�A�*

loss�5=�.K�       �	��^-Yc�A�*

losst��=��^       �	N}_-Yc�A�*

loss�g;�<��       �	a`-Yc�A�*

lossc7�;O�&M       �	��`-Yc�A�*

loss��<���       �	�Ra-Yc�A�*

loss�;��ZE       �	�a-Yc�A�*

loss�6 =��3       �	��b-Yc�A�*

loss��;ץν       �	�;c-Yc�A�*

loss��<TB�/       �	7�c-Yc�A�*

lossI�;{<�R       �	�d-Yc�A�*

loss���<z�j�       �	8/e-Yc�A�*

loss&d�<��Q�       �	W�e-Yc�A�*

loss�h<(�       �	Q�f-Yc�A� *

loss)&=�&�c       �	> g-Yc�A� *

loss�M�<�T(b       �	��g-Yc�A� *

loss	q=@�u{       �	ah-Yc�A� *

loss���<�૦       �	��h-Yc�A� *

lossi�<���d       �	��i-Yc�A� *

loss���;�Ւ       �	W@j-Yc�A� *

loss|��;��Ʈ       �	0�j-Yc�A� *

loss	��<�}5       �	еk-Yc�A� *

loss��=;�rJ       �	�Nl-Yc�A� *

lossH��<��)�       �	��l-Yc�A� *

lossO�)<	�\       �	��m-Yc�A� *

lossKŋ;���L       �	E)n-Yc�A� *

loss�=���       �	��n-Yc�A� *

lossú,<Q��V       �	w�o-Yc�A� *

loss�	�;���W       �	�p-Yc�A� *

loss���<�Ň       �	�\q-Yc�A� *

loss}�<����       �	�r-Yc�A� *

loss�h=����       �	��r-Yc�A� *

loss��u<���       �	�bs-Yc�A� *

loss#T=.v��       �	H�s-Yc�A� *

lossN�"=)�\       �	@�t-Yc�A� *

loss-{�=8��W       �	�Su-Yc�A� *

loss�.=��Q�       �	s�u-Yc�A� *

loss�Qt<O={�       �	��v-Yc�A� *

loss��;�1�'       �	T�w-Yc�A� *

loss��<�	d�       �	U�x-Yc�A� *

lossMV�:��       �	�y-Yc�A� *

loss��;g��       �	��y-Yc�A� *

loss��w<x��t       �	Gz-Yc�A� *

loss��C<A@��       �	��z-Yc�A� *

loss[�<����       �	�y{-Yc�A� *

lossjg�=Gc�       �	�4|-Yc�A� *

loss܇D=�k\       �	��|-Yc�A� *

loss�= �M       �	��}-Yc�A� *

lossl��;^O\�       �	�#~-Yc�A� *

loss��	=Ӓ2�       �	��~-Yc�A� *

loss��l;N�       �	C�-Yc�A� *

loss�O#:yz�       �	0�-Yc�A� *

loss/=�"b       �	O΀-Yc�A� *

loss:(:�2��       �		l�-Yc�A� *

loss9��<�@�p       �	��-Yc�A� *

loss���<�3�i       �	�ǂ-Yc�A� *

loss��o:��0       �	�g�-Yc�A� *

loss���<�8K�       �	 ��-Yc�A� *

loss��*;Y���       �	��-Yc�A� *

loss�$;���       �	�3�-Yc�A� *

loss���9bN �       �	�х-Yc�A� *

loss�\�;�}Ty       �	ܛ�-Yc�A� *

lossYW�<P�
*       �	xB�-Yc�A� *

loss9Œ<�Dp�       �	��-Yc�A� *

loss�19��\       �	�-Yc�A� *

loss��;Qm��       �	�m�-Yc�A� *

loss���<�7��       �	t�-Yc�A� *

loss�S;��j       �	o�-Yc�A� *

loss�P>[��:       �	a�-Yc�A� *

loss�6�<�bVz       �	s��-Yc�A� *

loss
�x=�ޑ�       �	cb�-Yc�A� *

loss���;0�y�       �	��-Yc�A� *

loss&:�:U4�p       �	���-Yc�A� *

loss��)<�Z�/       �	PS�-Yc�A� *

loss�4�={zٖ       �	��-Yc�A� *

loss�И=�r��       �	z��-Yc�A� *

lossSp�<���x       �	:#�-Yc�A� *

loss�JG=[ۘ�       �	���-Yc�A� *

lossڡ<>�B;        �	�O�-Yc�A� *

loss �=�M��       �	N�-Yc�A� *

lossF��;�RS       �	τ�-Yc�A� *

loss;�a=� �       �	�-Yc�A� *

loss4�=BA*       �	��-Yc�A� *

loss�ϣ<����       �	M�-Yc�A� *

lossC}X<e�       �	m��-Yc�A� *

loss��D=���       �	��-Yc�A� *

loss# �<����       �	�(�-Yc�A� *

lossΎ<OϯP       �	���-Yc�A� *

loss��3=�ʒ�       �	�X�-Yc�A� *

loss8�/=6i��       �	��-Yc�A� *

loss�ӆ<��_       �	ݔ�-Yc�A� *

losslܽ<g        �	�,�-Yc�A� *

loss	)<҈��       �	���-Yc�A� *

loss.�y=�H�\       �	�U�-Yc�A� *

lossdy�=i��       �	��-Yc�A� *

loss�A�<oW       �	˃�-Yc�A� *

loss���=��:�       �	�$�-Yc�A� *

lossSv�;���m       �	Ͼ�-Yc�A� *

loss�8�=�L�x       �	TU�-Yc�A� *

lossAձ<�|��       �	��-Yc�A� *

losst�3<&Huj       �	'��-Yc�A� *

loss��'<���       �	`�-Yc�A� *

loss�.B=3{{       �	[��-Yc�A� *

lossK<�،       �	��-Yc�A� *

lossL��<�Un�       �	��-Yc�A� *

lossl�g=G
�=       �	N��-Yc�A� *

loss��	=q�k0       �	�Q�-Yc�A� *

loss��<r�6F       �	��-Yc�A� *

loss|,�<�T�       �	7��-Yc�A� *

lossT~<�Ͳ       �	M,�-Yc�A� *

loss��<�q�       �	lϥ-Yc�A� *

lossx�k<PQ`6       �	�q�-Yc�A� *

loss���<Q�l�       �	��-Yc�A� *

lossQ*I=g^�!       �	]��-Yc�A� *

loss�ͅ:��
�       �	�B�-Yc�A� *

loss%�<��K�       �	�ݨ-Yc�A� *

loss��l;A%�r       �	4�-Yc�A� *

loss,��:�(�\       �	Ω�-Yc�A� *

lossfi�<�7-       �	wf�-Yc�A� *

loss1�@=��pU       �	��-Yc�A� *

loss��<}��       �	��-Yc�A� *

losss{�<� �       �	�9�-Yc�A� *

lossl%�<�d+       �	�v�-Yc�A� *

lossœ�<�I       �	��-Yc�A� *

loss�:�<�*e�       �	��-Yc�A� *

loss�<_�       �	8��-Yc�A� *

losslȻ=v       �	��-Yc�A� *

lossA�?=6�5�       �	���-Yc�A� *

lossXw<|Fy@       �	|e�-Yc�A� *

loss�$<��9�       �	G �-Yc�A� *

lossV�<�Z�D       �	���-Yc�A� *

loss��3=]��       �	�6�-Yc�A� *

loss�((=�x^       �	w�-Yc�A� *

lossn��<Q��       �	B��-Yc�A� *

loss(�<��.�       �	'L�-Yc�A� *

loss�[�:p/       �	e��-Yc�A� *

loss�<:b
�       �	��-Yc�A� *

loss��\=*��/       �	��-Yc�A� *

loss1)<<�[��       �	t��-Yc�A� *

loss�s�=�Jw�       �	��-Yc�A� *

loss�ic<_/�       �	��-Yc�A� *

loss��1=i��       �	�8�-Yc�A�!*

loss�|�;�oE       �	���-Yc�A�!*

lossw+=|       �	���-Yc�A�!*

lossQa�<u^��       �	s��-Yc�A�!*

loss���;]�vc       �	7U�-Yc�A�!*

loss��p=����       �	���-Yc�A�!*

lossSZ<�z�s       �	��-Yc�A�!*

loss�L�=�	�n       �	�D�-Yc�A�!*

loss��=@H�R       �	w��-Yc�A�!*

lossh�;8�Q       �	2q�-Yc�A�!*

loss�}�<�A�       �	��-Yc�A�!*

loss;��;o�M       �	��-Yc�A�!*

loss+U=�La       �	r�-Yc�A�!*

loss�C�;*�-m       �	��-Yc�A�!*

loss�>V<<휖       �	���-Yc�A�!*

loss��>Qg��       �	sL�-Yc�A�!*

loss@^=}�~       �	���-Yc�A�!*

loss�5<���       �	���-Yc�A�!*

lossL��=Kr�9       �	�*�-Yc�A�!*

loss�u�<+��       �	���-Yc�A�!*

loss��p<�"u       �	*U�-Yc�A�!*

loss��=��J�       �	���-Yc�A�!*

loss_Ԡ<|՞*       �	s��-Yc�A�!*

loss��y<���       �	��-Yc�A�!*

loss�Η;�?&       �	���-Yc�A�!*

loss��<t!J�       �	v�-Yc�A�!*

loss���;�5g       �	�-Yc�A�!*

loss��:��       �	r��-Yc�A�!*

loss���;�	;-       �	<J�-Yc�A�!*

lossj��;��-       �	���-Yc�A�!*

loss�T >���T       �	Q��-Yc�A�!*

lossL��<:fU       �	{/�-Yc�A�!*

loss��;�/�:       �	{�-Yc�A�!*

loss��:�2Y       �	��-Yc�A�!*

loss�+m:�E(�       �	]��-Yc�A�!*

loss��^=�7�
       �	�!�-Yc�A�!*

loss�<=1��J       �	ǻ�-Yc�A�!*

loss��=�r��       �	Z�-Yc�A�!*

loss�;�t$�       �	��-Yc�A�!*

loss�8�;�3�        �	���-Yc�A�!*

loss$B=KD�6       �	���-Yc�A�!*

loss�@=ϗjX       �	Ou�-Yc�A�!*

loss�w\<���       �	�"�-Yc�A�!*

loss&��<rM�R       �	��-Yc�A�!*

loss��';�p�       �	���-Yc�A�!*

lossa��<�2�       �	��-Yc�A�!*

lossh=�<��       �	/i�-Yc�A�!*

loss��<�V�       �	��-Yc�A�!*

loss�b<�$a�       �	���-Yc�A�!*

losstA�:��z       �	]��-Yc�A�!*

loss��<YDf"       �	3T�-Yc�A�!*

loss2��<�ȹ       �	���-Yc�A�!*

lossD�;&1W�       �	0��-Yc�A�!*

losswh�<�?gj       �	r�-Yc�A�!*

loss��<u�       �	��-Yc�A�!*

loss��v<{K�       �	�X�-Yc�A�!*

loss�7<��vP       �	���-Yc�A�!*

loss���;E���       �	j��-Yc�A�!*

loss�� =7�9�       �	�0�-Yc�A�!*

loss�.R<�3�       �	L��-Yc�A�!*

lossZ4;-�T�       �	�_�-Yc�A�!*

loss�k�;��~+       �	���-Yc�A�!*

lossI�;�Ջ=       �	��-Yc�A�!*

loss�l;�|       �	=�-Yc�A�!*

lossXO�=!���       �	���-Yc�A�!*

lossmC�<0��       �	�v�-Yc�A�!*

loss��<(���       �	� .Yc�A�!*

lossx	�<�60       �	� .Yc�A�!*

loss��<�jտ       �	�@.Yc�A�!*

lossW��;�Z�       �	��.Yc�A�!*

losssq�<�!Ѕ       �	�l.Yc�A�!*

loss�q�=檀�       �	.Yc�A�!*

loss���<�"!       �	u�.Yc�A�!*

loss�r@<C��       �	�2.Yc�A�!*

lossR�\<a�>       �	��.Yc�A�!*

loss��+=CV}�       �	�d.Yc�A�!*

lossV�=:�x?       �	�.Yc�A�!*

loss�=���V       �	ګ.Yc�A�!*

loss�f$<�ךy       �	oI.Yc�A�!*

loss�M=gZ��       �	(�.Yc�A�!*

loss*7�<A ��       �	�!	.Yc�A�!*

loss�L�<��M       �	M�	.Yc�A�!*

lossL�=�z       �	l
.Yc�A�!*

losssN�:*��?       �	.Yc�A�!*

losso�n<F���       �	ȳ.Yc�A�!*

loss~{�=
���       �	��.Yc�A�!*

loss�i<�*2�       �	W%.Yc�A�!*

loss�;/�       �	��.Yc�A�!*

lossVF�<��
       �	M�.Yc�A�!*

lossJ��<�fR�       �	9.Yc�A�!*

lossA?�;��D       �	?�.Yc�A�!*

loss��<�	�        �	�.Yc�A�!*

loss4��<6DJ�       �	P.Yc�A�!*

loss�y4;���!       �	��.Yc�A�!*

loss4�=�t       �	��.Yc�A�!*

loss�&i<��r@       �	�.Yc�A�!*

loss<��;�u�       �	��.Yc�A�!*

loss�*�< �.�       �	{.Yc�A�!*

loss<ĩ<� �       �	z.Yc�A�!*

loss��E;9��       �	�.Yc�A�!*

lossTF�<��       �	@Q.Yc�A�!*

loss=��<�ͨ�       �	�.Yc�A�!*

loss�qQ<��fV       �	l�.Yc�A�!*

lossl�=��[       �	�J.Yc�A�!*

loss��<K��       �	��.Yc�A�!*

lossw��<P���       �	}�.Yc�A�!*

loss�
i;u���       �	�).Yc�A�!*

lossz�=��$)       �	��.Yc�A�!*

loss�<d8��       �	�d.Yc�A�!*

loss#�<���       �	��.Yc�A�!*

loss��+<d_4       �	I�.Yc�A�!*

losshV�<��7�       �	�2.Yc�A�!*

loss��<�)�       �	��.Yc�A�!*

loss�'<+}�       �	�f.Yc�A�!*

loss
�s<��x�       �	�.Yc�A�!*

loss3��;#��       �	�� .Yc�A�!*

loss���<�a�^       �	�(!.Yc�A�!*

loss��;F)H.       �	"�!.Yc�A�!*

loss-B�;��j�       �	SZ".Yc�A�!*

loss���<��4       �	��".Yc�A�!*

loss���<ac       �	׈#.Yc�A�!*

loss�!=7R'       �	�!$.Yc�A�!*

loss���;�.��       �	�$.Yc�A�!*

lossb<~���       �	�[%.Yc�A�!*

lossV4�;HEbV       �	�&.Yc�A�!*

loss��<}�:       �	��&.Yc�A�!*

loss�0L<낲�       �	jL'.Yc�A�!*

loss�q�<�h�+       �	`�'.Yc�A�!*

loss(�;si{�       �	9}(.Yc�A�"*

loss�r�<��"       �	�).Yc�A�"*

loss3<�<ӦSk       �	��).Yc�A�"*

loss�`�<(x��       �	Hk*.Yc�A�"*

loss���9/b!�       �	+.Yc�A�"*

loss��;��J�       �	Ƥ+.Yc�A�"*

loss(�<w�_�       �	=I,.Yc�A�"*

loss��Y=1Nf�       �	a�,.Yc�A�"*

loss��;]��       �	�r-.Yc�A�"*

loss�=b�u�       �	h..Yc�A�"*

loss�2�;˳w�       �	�T/.Yc�A�"*

lossX,=K6�       �	��/.Yc�A�"*

loss"��;�ϑ       �	��0.Yc�A�"*

loss �;�^2       �	�>1.Yc�A�"*

loss+�;�s       �	��1.Yc�A�"*

lossT�%:n�%;       �	��2.Yc�A�"*

lossEa;d�\�       �	ݔ3.Yc�A�"*

loss&V�<�89%       �	b14.Yc�A�"*

loss�u�;~�ځ       �	��4.Yc�A�"*

loss���;�|�       �	Y�5.Yc�A�"*

loss��;�B-�       �	�*6.Yc�A�"*

lossC�<;�N��       �	�6.Yc�A�"*

lossng�=�:P[       �	ka7.Yc�A�"*

loss���;@�v
       �	qt8.Yc�A�"*

loss�B�;W��A       �	�9.Yc�A�"*

loss�ł=X��@       �	:.Yc�A�"*

loss�6~=�{Wr       �	��:.Yc�A�"*

lossf�=��       �	�A;.Yc�A�"*

loss�2;����       �	Z�;.Yc�A�"*

lossE!�;��~,       �	5�<.Yc�A�"*

lossA��<��       �	�<=.Yc�A�"*

loss��;��*�       �	�>.Yc�A�"*

loss��y<��P�       �	r�>.Yc�A�"*

loss :;;M}�       �	�:?.Yc�A�"*

loss��;��D       �	]�?.Yc�A�"*

loss(�=^ۯT       �	�u@.Yc�A�"*

loss_H<��        �	�	A.Yc�A�"*

loss�d�=����       �	j�A.Yc�A�"*

loss��e<���       �	]5B.Yc�A�"*

loss�I.=�e�I       �	��B.Yc�A�"*

loss�R�=i*"       �	�zC.Yc�A�"*

loss-;Ns�       �	��E.Yc�A�"*

loss�!=9�Q}       �	�F.Yc�A�"*

loss;�';���^       �	.7G.Yc�A�"*

loss835;-��       �	��G.Yc�A�"*

loss��0=
�Y       �	.�H.Yc�A�"*

loss��<���u       �	jI.Yc�A�"*

loss1��<���       �	�J.Yc�A�"*

loss�J<|p��       �	иJ.Yc�A�"*

lossZH�=�1�[       �	�K.Yc�A�"*

lossܔD<z���       �	�|L.Yc�A�"*

lossG9�<�+       �	X<M.Yc�A�"*

loss$��<�E8�       �	![N.Yc�A�"*

lossy<�{+       �	��N.Yc�A�"*

loss=C�=�t��       �	��O.Yc�A�"*

loss���=}��I       �	l�P.Yc�A�"*

lossT)<c��8       �	��Q.Yc�A�"*

loss_��<Xȧ       �	3oR.Yc�A�"*

loss_��=��e       �	�S.Yc�A�"*

lossl9�;@�/�       �	�^T.Yc�A�"*

lossVγ;$l��       �	ZU.Yc�A�"*

lossI��<�2��       �	��U.Yc�A�"*

loss��$;�Z�       �	�bV.Yc�A�"*

loss�d�;Ga1k       �	�	W.Yc�A�"*

loss�/�<�]Hk       �	x�W.Yc�A�"*

losst5q=je��       �	�\X.Yc�A�"*

loss�*�=�j       �	�Y.Yc�A�"*

loss�=!�v       �	��Y.Yc�A�"*

loss�r�<Y��2       �	�MZ.Yc�A�"*

loss���;����       �	�Z.Yc�A�"*

loss_߂;Wĉ�       �	��[.Yc�A�"*

loss�
{=VyB       �	*\.Yc�A�"*

loss�oH>�K       �	��\.Yc�A�"*

loss��=ƽ��       �	�b].Yc�A�"*

loss�Z=9��       �	��].Yc�A�"*

lossA�q<�O4�       �	-�^.Yc�A�"*

loss��<)�Ջ       �	�4_.Yc�A�"*

loss��=�SJ       �	��_.Yc�A�"*

loss��<�U��       �	b�`.Yc�A�"*

loss�7=Lν       �	�a.Yc�A�"*

lossa�B<���       �	#�a.Yc�A�"*

loss1p<ۏ�N       �	$db.Yc�A�"*

loss>G
Go       �	)c.Yc�A�"*

lossC�Z= O�       �	�c.Yc�A�"*

lossN7�=���       �	�[d.Yc�A�"*

lossD�j=�uqW       �	D�d.Yc�A�"*

losss	�<*�n       �	x�e.Yc�A�"*

loss@<�;��`�       �	�f.Yc�A�"*

lossF�<<g�k�       �	81g.Yc�A�"*

loss��k<��8       �	!�g.Yc�A�"*

loss6�=��,w       �	�fh.Yc�A�"*

loss?�A=�e��       �	i.Yc�A�"*

loss��<�V�?       �	��i.Yc�A�"*

loss��}=]��       �	)?j.Yc�A�"*

loss1E�<�Ng       �	��j.Yc�A�"*

lossK=��=�       �	}yk.Yc�A�"*

loss�{=�qD       �	�l.Yc�A�"*

loss��r<���$       �	��l.Yc�A�"*

loss��=8�H       �	�Mm.Yc�A�"*

loss��<��6�       �	K�m.Yc�A�"*

loss�ӭ<�;��       �	V�n.Yc�A�"*

loss�̹<V���       �	�co.Yc�A�"*

loss�B�;@~�       �	��p.Yc�A�"*

loss�i�;�_Ѧ       �	vpq.Yc�A�"*

loss!�Y<�^KV       �	�Ar.Yc�A�"*

loss��A<�"E
       �	�s.Yc�A�"*

loss���;�u+       �	p�s.Yc�A�"*

loss���<@�H       �	��t.Yc�A�"*

loss��S<���       �	�}u.Yc�A�"*

losse�=��D�       �	�Mv.Yc�A�"*

loss*�H<\7f�       �	� w.Yc�A�"*

loss3^�=�әE       �	�Vx.Yc�A�"*

loss�;�/o       �	��x.Yc�A�"*

loss\��<0���       �	E�y.Yc�A�"*

loss�@�<Zj<j       �	u�z.Yc�A�"*

loss��@=�W�k       �	�j{.Yc�A�"*

loss�a=<��>       �	yW|.Yc�A�"*

lossw2J=���v       �	��}.Yc�A�"*

loss��<Pi�       �	�F~.Yc�A�"*

loss���;�e8�       �	m�~.Yc�A�"*

loss��w<rB�       �	-�.Yc�A�"*

loss�]�<�P#       �	O��.Yc�A�"*

loss��=޵��       �	2�.Yc�A�"*

lossm�=p�}8       �	��.Yc�A�"*

lossn��=�%�y       �	��.Yc�A�"*

loss��<����       �	w.�.Yc�A�"*

loss;H֬       �	eă.Yc�A�"*

loss�]e=l*��       �	�{�.Yc�A�"*

loss�a�<O)\�       �	a�.Yc�A�#*

losse�z;�ݶ�       �	�Ʌ.Yc�A�#*

loss�;U�r       �	�b�.Yc�A�#*

loss)<{h�8       �	#�.Yc�A�#*

loss��;���       �	���.Yc�A�#*

loss�0V=�7~       �	�W�.Yc�A�#*

loss2�=�0g�       �	��.Yc�A�#*

loss��<x�t       �	���.Yc�A�#*

loss��<z��J       �	�<�.Yc�A�#*

lossq|�;t'T       �	��.Yc�A�#*

loss��F=�M�       �	���.Yc�A�#*

loss�Z�<��9t       �	P7�.Yc�A�#*

loss�/;=��\P       �	V׌.Yc�A�#*

loss��:Cc\b       �	pz�.Yc�A�#*

losscb:�u,"       �	��.Yc�A�#*

loss@�;�`�k       �	�ώ.Yc�A�#*

lossH�m=�ɇ       �	Ho�.Yc�A�#*

lossĉ�<a��       �	n�.Yc�A�#*

loss=#�=W~�H       �	(��.Yc�A�#*

losst�M=M�O       �	�]�.Yc�A�#*

loss2�=(�A       �	��.Yc�A�#*

loss�k�;�U�       �	ڪ�.Yc�A�#*

losswL<�       �	�M�.Yc�A�#*

loss#�><�fru       �	��.Yc�A�#*

loss��=ur��       �	 ��.Yc�A�#*

lossȧ=%y|=       �	�<�.Yc�A�#*

lossE��=�%
       �	wٕ.Yc�A�#*

lossa�`<���       �	�z�.Yc�A�#*

lossre~<d�4s       �	�.Yc�A�#*

loss|`=��7       �	众.Yc�A�#*

loss�W<U!6�       �	W�.Yc�A�#*

loss옯;���Y       �	���.Yc�A�#*

lossw�\<�Z       �	��.Yc�A�#*

loss��=�A�       �	iR�.Yc�A�#*

loss�=�;>�P       �	<��.Yc�A�#*

lossμ�=�EZo       �	��.Yc�A�#*

loss� Z=6�       �	9�.Yc�A�#*

loss�ɼ:���       �	u�.Yc�A�#*

loss��;�WH�       �	A��.Yc�A�#*

lossi�h=��)b       �	��.Yc�A�#*

lossH׆<�H!       �	���.Yc�A�#*

loss��u;��Z�       �	p\�.Yc�A�#*

lossh`�<k��:       �	d�.Yc�A�#*

loss0�=e�K�       �	���.Yc�A�#*

loss邾;�M�       �	�W�.Yc�A�#*

lossR'=�       �	*�.Yc�A�#*

loss]��=�w͚       �	���.Yc�A�#*

loss��<�>�`       �	<M�.Yc�A�#*

loss*@=	�       �	��.Yc�A�#*

lossȥ�=�T;       �	o��.Yc�A�#*

loss�#�<�=�       �	�=�.Yc�A�#*

loss��<j4�       �	�ݥ.Yc�A�#*

loss�/<7��       �	�y�.Yc�A�#*

loss�Q;.`,6       �	/�.Yc�A�#*

loss��<Q�c       �	���.Yc�A�#*

lossJ+!=[e�       �	�P�.Yc�A�#*

loss���;�j�       �	��.Yc�A�#*

lossv!=H��~       �	��.Yc�A�#*

loss��O<Ϧ�       �	�n�.Yc�A�#*

loss�$�=��       �	>	�.Yc�A�#*

loss�T�:A��3       �	/��.Yc�A�#*

loss�=���u       �	�W�.Yc�A�#*

loss�@<hʇ       �	��.Yc�A�#*

loss
�;�Z-{       �	\��.Yc�A�#*

loss���<�]�       �	@4�.Yc�A�#*

loss��>Y���       �	jڮ.Yc�A�#*

lossȽ�=��X4       �	��.Yc�A�#*

loss$��;z8C�       �	��.Yc�A�#*

loss�i=hKo       �	Ֆ�.Yc�A�#*

loss
�$=6��       �	�6�.Yc�A�#*

loss(��<[�-�       �	�ڲ.Yc�A�#*

loss���:��#+       �	��.Yc�A�#*

loss&��<���       �	w��.Yc�A�#*

loss�m�<���J       �	�A�.Yc�A�#*

loss [0<��'�       �	}�.Yc�A�#*

lossEk�<��       �	{��.Yc�A�#*

loss�!�<oPޡ       �	) �.Yc�A�#*

lossmY�<�y�Y       �	�ø.Yc�A�#*

lossͩ�<{;E/       �	0h�.Yc�A�#*

loss�;]���       �	�.Yc�A�#*

loss���;Ӈ�       �	��.Yc�A�#*

loss�_�<��       �	�k�.Yc�A�#*

loss��O<X�\�       �	:�.Yc�A�#*

loss�V=6��       �	Ӽ�.Yc�A�#*

loss䰟=q�t       �	�_�.Yc�A�#*

loss��M=��0�       �	��.Yc�A�#*

lossM��;{~       �	��.Yc�A�#*

lossh+�;��       �	0d�.Yc�A�#*

loss���<�&�       �	���.Yc�A�#*

lossz0�;�̣�       �	���.Yc�A�#*

loss�ͅ<�Q[�       �	3�.Yc�A�#*

loss�DD=<��       �	���.Yc�A�#*

loss�P=+)�       �	ep�.Yc�A�#*

loss1q�<����       �	�M�.Yc�A�#*

loss�� <ł[�       �	���.Yc�A�#*

loss-	�;��	V       �	��.Yc�A�#*

loss�ܼ<f]y       �	�+�.Yc�A�#*

loss{��=-&�       �	K��.Yc�A�#*

loss�0<� �B       �	4f�.Yc�A�#*

loss�N�<����       �	? �.Yc�A�#*

lossޟ�<��L�       �	��.Yc�A�#*

loss��d<�=�C       �	�>�.Yc�A�#*

loss��9<D�ɵ       �	#��.Yc�A�#*

loss��>y�S       �	���.Yc�A�#*

loss_f�;�ީ       �	�.�.Yc�A�#*

loss�ix=R��       �	���.Yc�A�#*

loss�4=�5��       �	���.Yc�A�#*

loss�U<.:B       �	�M�.Yc�A�#*

loss�S@<)��       �	��.Yc�A�#*

lossϽ#<�r       �	��.Yc�A�#*

lossM�;k�       �	'�.Yc�A�#*

loss��<�9�       �	���.Yc�A�#*

loss��<W(@M       �	8i�.Yc�A�#*

loss��;PinA       �	E�.Yc�A�#*

loss���<4�0�       �	���.Yc�A�#*

loss=�a<�`�B       �	�j�.Yc�A�#*

loss�� <�h�       �	��.Yc�A�#*

loss���;n���       �	R��.Yc�A�#*

lossNQ<�x       �	C9�.Yc�A�#*

loss��<G�       �	���.Yc�A�#*

loss�ɕ<o�       �	���.Yc�A�#*

loss�8�;q��       �	�d�.Yc�A�#*

loss��=�
i1       �	��.Yc�A�#*

loss��=����       �	���.Yc�A�#*

losso�o=���A       �	�G�.Yc�A�#*

loss\��<:���       �	���.Yc�A�#*

losss�#=Ht,�       �	.v�.Yc�A�#*

loss0�=�44       �	��.Yc�A�#*

loss���<ϳ]�       �	ӥ�.Yc�A�$*

loss�K�<��ק       �	Qg�.Yc�A�$*

lossW��<7�)W       �		��.Yc�A�$*

loss�k�<�V�t       �	��.Yc�A�$*

loss��=؎        �	�2�.Yc�A�$*

loss==Ad��       �	W��.Yc�A�$*

loss p�<��X�       �	ka�.Yc�A�$*

loss���=��6       �	1�.Yc�A�$*

loss��m<�O�       �	���.Yc�A�$*

loss84<�ͣ�       �	j��.Yc�A�$*

loss���;g?�       �	��.Yc�A�$*

lossF��<1��       �	���.Yc�A�$*

loss��<Ƹ�       �	S�.Yc�A�$*

loss �8<���k       �	��.Yc�A�$*

loss��;n�ԥ       �	f.�.Yc�A�$*

loss׺�;���p       �	}��.Yc�A�$*

loss�l�=��&�       �	Ho�.Yc�A�$*

loss�n	<d       �	��.Yc�A�$*

lossԍ=�w[�       �	Z��.Yc�A�$*

loss�p7;���       �	 8�.Yc�A�$*

lossS�3<`��       �	���.Yc�A�$*

loss͚Z=�S       �	Yj�.Yc�A�$*

lossR�;C�Ք       �	�.Yc�A�$*

loss��<&$f�       �	���.Yc�A�$*

loss���<'>       �	3�.Yc�A�$*

lossL̘<��s       �	���.Yc�A�$*

lossC�<��U       �	Na�.Yc�A�$*

loss���<��Ė       �	oc�.Yc�A�$*

loss��A=�t�       �	���.Yc�A�$*

loss�Ѱ<��       �	���.Yc�A�$*

lossߤ=Z� Q       �	�3�.Yc�A�$*

loss�L<yz       �	d��.Yc�A�$*

lossF1�;e[�|       �	�f�.Yc�A�$*

loss��<D�c�       �	2��.Yc�A�$*

loss���=���3       �	{��.Yc�A�$*

losst�=q��       �	:�.Yc�A�$*

loss�(�=���2       �	{��.Yc�A�$*

loss���:�yr�       �	��.Yc�A�$*

loss�<�-K�       �	�A�.Yc�A�$*

loss/��<- ��       �	x*�.Yc�A�$*

loss�k;t�`�       �	��.Yc�A�$*

loss_��<xA�       �	a�.Yc�A�$*

loss�!�<�_       �	pa�.Yc�A�$*

loss�tI<���       �	�`�.Yc�A�$*

losst��:  �       �	;T�.Yc�A�$*

loss�;aڗ       �	��.Yc�A�$*

lossT#�;��T       �	J��.Yc�A�$*

loss�<Ts��       �	���.Yc�A�$*

loss�<f=2�Vu       �	�7�.Yc�A�$*

loss�{X=�G�m       �	A��.Yc�A�$*

losst=�I�       �	?n�.Yc�A�$*

loss�f�<\�m�       �	1
�.Yc�A�$*

loss��<���C       �	֬�.Yc�A�$*

loss%�3<�Ċ3       �	^��.Yc�A�$*

loss ��<����       �	�& /Yc�A�$*

loss@6;;a�       �	�� /Yc�A�$*

lossmw2=E]W7       �	\/Yc�A�$*

losscØ<�
Ǘ       �	��/Yc�A�$*

loss�z<����       �	Ֆ/Yc�A�$*

lossϲ<x�C:       �	�Y/Yc�A�$*

lossy�;p ʃ       �	��/Yc�A�$*

loss> �;c{|�       �	�E/Yc�A�$*

loss6�":�H�       �	��/Yc�A�$*

loss��S=�H�[       �	 �/Yc�A�$*

loss�>;+���       �	X8/Yc�A�$*

lossӪ�<"       �	J�/Yc�A�$*

loss��=F�[�       �	�m/Yc�A�$*

lossc/V<MaK       �		/Yc�A�$*

lossI!<���       �	E�	/Yc�A�$*

lossL�;<)�>       �	;
/Yc�A�$*

lossj�<�]       �	��
/Yc�A�$*

loss\��;C�ڇ       �	�v/Yc�A�$*

lossn�A>	"5M       �	{//Yc�A�$*

loss�o<Ql�e       �	��/Yc�A�$*

loss�v�<���       �	Yn/Yc�A�$*

loss��^<g$Ӂ       �	3�/Yc�A�$*

loss��=AM}�       �	p>/Yc�A�$*

loss�Li<y�$�       �	��/Yc�A�$*

loss6�B<��g       �	dx/Yc�A�$*

loss�j7<��[�       �	�/Yc�A�$*

loss��&=�|�       �	̵/Yc�A�$*

loss|�<SW{       �	�Q/Yc�A�$*

loss�J�<
C��       �	��/Yc�A�$*

loss�E�=�U�       �	��/Yc�A�$*

loss�in<\�a       �	;/Yc�A�$*

lossӄ[:��       �	�>/Yc�A�$*

lossH�C=C�qT       �	��/Yc�A�$*

loss襛:!�f�       �	R/Yc�A�$*

loss��)<-H&�       �	/Yc�A�$*

loss��<���       �	��/Yc�A�$*

lossO�b<�P�       �	��/Yc�A�$*

lossc�=C`�       �	�|/Yc�A�$*

loss=���       �	a/Yc�A�$*

loss�ˢ<�c��       �	��/Yc�A�$*

loss�M�;�Qwi       �	�w/Yc�A�$*

lossxq*<��x       �	%#/Yc�A�$*

lossWf�;��Y�       �	a�/Yc�A�$*

loss=a{="%ƻ       �	�Y/Yc�A�$*

loss�)2;��Q       �	�/Yc�A�$*

loss�<�P3�       �	*�/Yc�A�$*

loss Y5<�|�       �	�(/Yc�A�$*

loss��=so�       �	H�/Yc�A�$*

loss�K�=����       �	] /Yc�A�$*

loss��<�J}>       �	�� /Yc�A�$*

lossŰ=]�V�       �	��!/Yc�A�$*

loss�7=<���       �	�."/Yc�A�$*

lossM+�;�Z��       �	 �"/Yc�A�$*

lossEX�<-�h�       �	 b#/Yc�A�$*

loss�<��       �	�$/Yc�A�$*

loss�	<�Ƨ�       �	׿$/Yc�A�$*

loss��;�Z;�       �	g%/Yc�A�$*

loss5�=���       �	A&/Yc�A�$*

loss��;61�       �	M�&/Yc�A�$*

loss�r�:p�C�       �	�R'/Yc�A�$*

lossN�:<s|��       �	l�'/Yc�A�$*

lossT�<��h�       �	�(/Yc�A�$*

lossM��;1~       �	�)/Yc�A�$*

loss?d<�ox�       �	�)/Yc�A�$*

lossJ/<�=;�       �	DM*/Yc�A�$*

loss�j;��       �	w+/Yc�A�$*

loss�>3<��{       �	��+/Yc�A�$*

losss�w=&Pk�       �	�N,/Yc�A�$*

loss*=��Md       �	��,/Yc�A�$*

loss���;��~       �	��-/Yc�A�$*

loss�$�=�g��       �	3./Yc�A�$*

lossC�3;�&��       �	x�./Yc�A�$*

loss�:�,.       �	nN0/Yc�A�$*

lossm��:wԃ�       �	1/Yc�A�$*

lossT�[<�Z�       �	��1/Yc�A�%*

loss!�:/<�U       �	f2/Yc�A�%*

loss;EP;��,       �	H3/Yc�A�%*

loss�}$;,x�8       �	̶3/Yc�A�%*

losswss;?�e�       �	�l4/Yc�A�%*

loss�vK9��n       �	�5/Yc�A�%*

loss,�9�Y�       �	�5/Yc�A�%*

loss�z�8@��       �	�W6/Yc�A�%*

loss��2;��=<       �	�6/Yc�A�%*

lossX�D=�Z�       �	��7/Yc�A�%*

loss���;�.��       �	JF8/Yc�A�%*

lossE6;���       �	��8/Yc�A�%*

loss���<Bx�$       �	�s9/Yc�A�%*

lossR��=�V�       �	d:/Yc�A�%*

loss���:
�n�       �	Z�:/Yc�A�%*

lossvE>pB�       �	�/;/Yc�A�%*

loss��3;ڡ{<       �	&�;/Yc�A�%*

loss��<�A�O       �	GV</Yc�A�%*

loss��=��B       �	�=/Yc�A�%*

loss��G;"��        �	l�=/Yc�A�%*

loss�[�=�Ya�       �	V,>/Yc�A�%*

loss��u=(���       �	j�>/Yc�A�%*

loss|�;����       �	�b?/Yc�A�%*

lossJ��;k#�_       �	��?/Yc�A�%*

lossf+�<�V�y       �	�@/Yc�A�%*

loss\��=mU�F       �	K A/Yc�A�%*

loss�y&<�3p       �	��A/Yc�A�%*

loss��<�%7       �	�KB/Yc�A�%*

loss��_<�zE5       �	FAC/Yc�A�%*

lossd�=;�mh       �	p[D/Yc�A�%*

lossDۓ<k�G�       �	C�D/Yc�A�%*

loss���<��@       �	J�E/Yc�A�%*

loss	\=4�X�       �	�*F/Yc�A�%*

loss��d<��Q       �	�F/Yc�A�%*

losse&;�B�T       �	=_G/Yc�A�%*

loss���<�\W�       �	��G/Yc�A�%*

loss��<�/	,       �	 �H/Yc�A�%*

lossw\�;ݵi�       �	:I/Yc�A�%*

loss��*<x�P'       �	YJ/Yc�A�%*

loss�w=2E�,       �	��J/Yc�A�%*

loss�n-=�s)�       �	�KK/Yc�A�%*

loss�m�:����       �	��K/Yc�A�%*

loss�!.=���       �	6wL/Yc�A�%*

loss�e9=���`       �	#M/Yc�A�%*

loss
��:s�G       �	ߤM/Yc�A�%*

loss���<�.��       �	�;N/Yc�A�%*

loss(&'<�T�        �	�N/Yc�A�%*

loss�Ov<�F'�       �	��O/Yc�A�%*

loss�,�<�
cz       �	H6P/Yc�A�%*

lossS�F<��y�       �	�P/Yc�A�%*

loss7Τ;�E�z       �	3kQ/Yc�A�%*

loss�<����       �	��Q/Yc�A�%*

loss	� =ɍB[       �	�R/Yc�A�%*

loss�\�<`V�<       �	 AS/Yc�A�%*

loss8);���       �	��S/Yc�A�%*

loss���<c�\?       �	�vT/Yc�A�%*

loss';mZ��       �	�U/Yc�A�%*

loss.^e=��t�       �	�U/Yc�A�%*

loss���<�RbG       �	�>V/Yc�A�%*

loss�9�=�[f       �	V�V/Yc�A�%*

lossc�=��       �	LoW/Yc�A�%*

lossM�5;�Y       �	WX/Yc�A�%*

lossNKW=���       �	��X/Yc�A�%*

loss$�<�ؓD       �	�;Y/Yc�A�%*

lossEQ�<���       �	��Y/Yc�A�%*

lossɣ�;��Y�       �	�	s/Yc�A�%*

loss�&<�GM�       �	Q�s/Yc�A�%*

loss�%Z=8�       �	:]t/Yc�A�%*

loss��G=ü�       �	�t/Yc�A�%*

loss�p�<"��       �	��u/Yc�A�%*

loss��< R��       �	Rv/Yc�A�%*

loss���=�cr<       �	y�v/Yc�A�%*

lossϧ9=ٲ��       �	E�w/Yc�A�%*

loss=H,=�5�       �	�x/Yc�A�%*

loss-�/=BQm       �	A�x/Yc�A�%*

lossA�<V���       �	�Ty/Yc�A�%*

loss&iM=�G�%       �	<�y/Yc�A�%*

losso
<3�jc       �	y�z/Yc�A�%*

lossH��<~U�       �	b0{/Yc�A�%*

lossc��;f���       �	�{/Yc�A�%*

lossV��<p'%�       �	�h|/Yc�A�%*

loss��;�\Y       �	P�|/Yc�A�%*

loss;wY;�w��       �	�}/Yc�A�%*

loss���;����       �	�x~/Yc�A�%*

loss� �<��
�       �	H/Yc�A�%*

loss�X�;��|       �	��/Yc�A�%*

loss��5=�ai       �	�J�/Yc�A�%*

loss8$Z;
õ�       �	��/Yc�A�%*

loss�*=q�k<       �	���/Yc�A�%*

loss��<��Ԉ       �	/6�/Yc�A�%*

loss�.^;���       �	�ׂ/Yc�A�%*

loss�+e<��\       �	E��/Yc�A�%*

loss ��:���       �	q�/Yc�A�%*

loss��M=ي       �	Ǆ/Yc�A�%*

loss��K;�Y�       �	'l�/Yc�A�%*

loss�?�;\5�5       �	��/Yc�A�%*

loss;��;�Jd       �	���/Yc�A�%*

loss���<����       �	�I�/Yc�A�%*

lossct<q�#       �	G�/Yc�A�%*

lossf��; �OJ       �	���/Yc�A�%*

loss3<e�n�       �	H�/Yc�A�%*

loss=��:�>�       �	É/Yc�A�%*

lossZܘ<�I       �	ak�/Yc�A�%*

loss1v(=�� n       �	�/Yc�A�%*

loss�d�=胃3       �	O��/Yc�A�%*

loss0Ð<�@ǡ       �	�Y�/Yc�A�%*

loss�}=��#       �	�@�/Yc�A�%*

loss�u:L� w       �	ݍ/Yc�A�%*

loss�̨<w�6:       �	���/Yc�A�%*

lossߏe<9�ߡ       �	 A�/Yc�A�%*

loss[|�<�?v�       �	o؏/Yc�A�%*

lossr��<��z       �	�p�/Yc�A�%*

lossEX�<7�?       �	��/Yc�A�%*

loss*P}:�^��       �	SΑ/Yc�A�%*

loss}�;݊�t       �	�p�/Yc�A�%*

loss);ui��       �	o�/Yc�A�%*

loss)0<��\u       �	ߣ�/Yc�A�%*

losssҺ<��K       �	Yn�/Yc�A�%*

loss� �=4Rɨ       �	��/Yc�A�%*

lossnj;�Y�       �	ⱕ/Yc�A�%*

loss�t;Z���       �	7��/Yc�A�%*

loss>Y:�{۷       �	M0�/Yc�A�%*

lossiT�;(<;       �	�֗/Yc�A�%*

loss�?�< t�       �	`r�/Yc�A�%*

loss�q<�Є       �	��/Yc�A�%*

loss:��=Ma       �	��/Yc�A�%*

loss�4�:xi�H       �	r2�/Yc�A�%*

loss��(<?�fO       �	eƚ/Yc�A�%*

loss��L=Ch0�       �	rm�/Yc�A�&*

loss�
U;76��       �	,�/Yc�A�&*

loss#��;����       �	��/Yc�A�&*

lossFE�;F�o�       �	�R�/Yc�A�&*

loss,Q�;4�,       �	���/Yc�A�&*

loss]��;� ��       �	n��/Yc�A�&*

loss�(<��e       �	�H�/Yc�A�&*

loss-n�<�g�       �	��/Yc�A�&*

loss�0�;���       �	m��/Yc�A�&*

lossmA=��&�       �	%�/Yc�A�&*

losst�;	�Y�       �	Bҡ/Yc�A�&*

loss�S=H�+�       �	!s�/Yc�A�&*

loss؄�;@�       �	 �/Yc�A�&*

lossa�<+��       �	Gǣ/Yc�A�&*

loss2s�<_��       �	an�/Yc�A�&*

lossrh;e��       �	,�/Yc�A�&*

loss�ق=cy}f       �	,��/Yc�A�&*

loss}�<N�U       �	-[�/Yc�A�&*

loss%�J<��cl       �	3��/Yc�A�&*

loss.��;��;       �	໧/Yc�A�&*

loss��;ah'        �	�c�/Yc�A�&*

lossza<>��       �	��/Yc�A�&*

loss��0=*?�       �	��/Yc�A�&*

lossiE�<{-Y,       �	3��/Yc�A�&*

loss�8W<�.       �	}v�/Yc�A�&*

lossDz=~%w       �	N�/Yc�A�&*

lossJ�Y<o��       �	�̬/Yc�A�&*

loss��;���       �	���/Yc�A�&*

loss�	�=	q`�       �	^-�/Yc�A�&*

loss](=P:       �	M֮/Yc�A�&*

loss�e&<����       �	�o�/Yc�A�&*

lossj�6=�fO       �	�3�/Yc�A�&*

loss�n<��       �	�װ/Yc�A�&*

loss�կ<�|�       �	�ɱ/Yc�A�&*

lossC��<�
�       �	0f�/Yc�A�&*

loss�| =��Y       �	p	�/Yc�A�&*

loss��1;����       �	穳/Yc�A�&*

loss-��;f��
       �	Z��/Yc�A�&*

loss���9ů�\       �	$E�/Yc�A�&*

loss��=Z#�$       �	�ܵ/Yc�A�&*

loss]<�&�       �	�t�/Yc�A�&*

lossi�:���       �	s�/Yc�A�&*

lossx�=��       �	�η/Yc�A�&*

lossb/"<D��R       �	Do�/Yc�A�&*

loss��==��       �	��/Yc�A�&*

loss�V=O���       �	���/Yc�A�&*

loss3�;����       �	:t�/Yc�A�&*

loss�v=�Z?       �	��/Yc�A�&*

loss�_<#TJs       �	�Ļ/Yc�A�&*

lossH=�g       �	Xo�/Yc�A�&*

loss�A�:I_�       �	�/Yc�A�&*

lossݻ�;�<1       �	r�/Yc�A�&*

loss�<l]�       �	���/Yc�A�&*

loss)�)=�O,       �	yʿ/Yc�A�&*

lossc/W=oI��       �	�v�/Yc�A�&*

loss��<6��]       �	�/Yc�A�&*

loss.=�;%�}       �	��/Yc�A�&*

loss�9�<f5p�       �	�\�/Yc�A�&*

loss��]=�l9       �	n��/Yc�A�&*

loss%K�<�5D�       �	d��/Yc�A�&*

lossM��<�Kt�       �	�3�/Yc�A�&*

losse�';�Df       �	:��/Yc�A�&*

loss\��;X�<+       �	j�/Yc�A�&*

loss�?6=Ԭ��       �	�/Yc�A�&*

loss��<����       �	Н�/Yc�A�&*

loss��"=���)       �	�F�/Yc�A�&*

lossx�r<����       �	���/Yc�A�&*

loss4��;�_í       �	#��/Yc�A�&*

loss��<���M       �	O!�/Yc�A�&*

loss�@�<N�h       �	g&�/Yc�A�&*

loss	m7;�|4�       �	Ͻ�/Yc�A�&*

loss�4=zl�p       �	�S�/Yc�A�&*

loss��;���[       �	���/Yc�A�&*

loss��;���       �	'��/Yc�A�&*

loss��;��       �	��/Yc�A�&*

loss��+<9�t�       �	9��/Yc�A�&*

loss�@�<%��x       �	�L�/Yc�A�&*

loss�v�;���       �	n��/Yc�A�&*

lossv�;g���       �	��/Yc�A�&*

loss��k<r�       �	�2�/Yc�A�&*

losss""=�r7U       �	���/Yc�A�&*

loss� =�V�       �	�m�/Yc�A�&*

loss7=�S��       �	O�/Yc�A�&*

loss7|�<�V6�       �	���/Yc�A�&*

lossTy�<�"�       �	�o�/Yc�A�&*

loss��;G���       �	aq�/Yc�A�&*

loss�;<y�`h       �	U�/Yc�A�&*

loss���;�.�       �	���/Yc�A�&*

loss�^�<D9�y       �	}A�/Yc�A�&*

loss��`;���i       �	�/Yc�A�&*

lossvv�<��T�       �	��/Yc�A�&*

loss;$=�m�R       �	�E�/Yc�A�&*

loss�7�:K��7       �	���/Yc�A�&*

loss���<i`�       �	Cu�/Yc�A�&*

lossD8�;F	w�       �	�/Yc�A�&*

loss.�C<���       �	��/Yc�A�&*

loss!�Z;I���       �	]R�/Yc�A�&*

loss��g=��       �	���/Yc�A�&*

lossw3<{�d       �	.u�/Yc�A�&*

loss�}5<�E0       �	
�/Yc�A�&*

loss�m�:w�<       �	b��/Yc�A�&*

loss�j�;5�:�       �	�8�/Yc�A�&*

loss�MI:�{]Z       �	��/Yc�A�&*

loss4�;�ֆ       �	3j�/Yc�A�&*

lossEB�<=��#       �	��/Yc�A�&*

loss?V�<��m       �	��/Yc�A�&*

loss�D=�
�       �	%=�/Yc�A�&*

loss#3=�Ҭ}       �	,��/Yc�A�&*

loss�׿:h)�[       �	3l�/Yc�A�&*

loss��c;��)E       �	 �/Yc�A�&*

lossƻ	<nl$�       �	$��/Yc�A�&*

loss�� <!9       �	5�/Yc�A�&*

loss��<�X|       �	���/Yc�A�&*

loss�T�;�q��       �	�]�/Yc�A�&*

loss�5l=�� �       �	� �/Yc�A�&*

lossA�\;���       �	���/Yc�A�&*

loss�Q><��-�       �	�@�/Yc�A�&*

loss/�;|��s       �	���/Yc�A�&*

loss���;P�A�       �	Á�/Yc�A�&*

loss
#i<�M�       �	�/Yc�A�&*

loss̛�<BH       �	j��/Yc�A�&*

lossD��:��       �	�u�/Yc�A�&*

lossF�:��.�       �	j�/Yc�A�&*

loss3iG<r�2�       �	���/Yc�A�&*

lossq�B<�l��       �	�H�/Yc�A�&*

lossr~W<���&       �	i��/Yc�A�&*

loss��;=�v��       �	��/Yc�A�&*

loss�I�<�ڶ�       �	��/Yc�A�&*

losss��<�*��       �	c��/Yc�A�'*

loss)l�<!J�A       �	PR�/Yc�A�'*

lossiT<�p       �	d��/Yc�A�'*

loss`�<;���       �	ʊ�/Yc�A�'*

loss�7=���?       �	jM�/Yc�A�'*

loss 4<A���       �	8��/Yc�A�'*

lossMZ�;v|�M       �	���/Yc�A�'*

loss/"=�       �	�H�/Yc�A�'*

loss�t�<��r�       �	���/Yc�A�'*

loss��|<��       �	�T�/Yc�A�'*

loss�*;�i>       �	UM�/Yc�A�'*

loss�è;'�       �	��/Yc�A�'*

loss��{;/�"�       �	tA�/Yc�A�'*

lossT��<�uD�       �	;5�/Yc�A�'*

loss�~<���&       �	���/Yc�A�'*

lossh�>=x.�       �	��/Yc�A�'*

loss�B=��kP       �	�B�/Yc�A�'*

loss,f�<3E/f       �	�/Yc�A�'*

loss���=3�	       �	���/Yc�A�'*

loss��;�*z       �	fN�/Yc�A�'*

loss1�;3̈́       �	���/Yc�A�'*

loss���<1�B@       �	.��/Yc�A�'*

lossֆ�<8z�       �	b��/Yc�A�'*

lossd[]<�5�d       �	�� 0Yc�A�'*

loss$�<�ܷ�       �	l0Yc�A�'*

lossn	="#�l       �	�	0Yc�A�'*

lossLI[=���       �	�0Yc�A�'*

loss��=�ed        �	B0Yc�A�'*

loss=n�=��        �	��0Yc�A�'*

lossD2/<	��       �	Ԁ0Yc�A�'*

loss��<���i       �	�0Yc�A�'*

loss�.�=Q�|�       �	��0Yc�A�'*

loss-��=]]�l       �	�\0Yc�A�'*

lossD�<^�ǩ       �	*�0Yc�A�'*

loss�Ӎ=��1�       �	:�0Yc�A�'*

losslt�<{D�Y       �	�-0Yc�A�'*

loss�l�;���       �	3�0Yc�A�'*

loss�<ϖ��       �	�`	0Yc�A�'*

loss��u;A
��       �	
0Yc�A�'*

lossp�:��       �	��
0Yc�A�'*

loss��<Z1#!       �	[^0Yc�A�'*

loss)�b<$���       �	 0Yc�A�'*

loss�;y(�       �	ܡ0Yc�A�'*

lossd�=R(�       �	�0Yc�A�'*

lossk�<ś��       �	�'0Yc�A�'*

lossio<�R�!       �	U�0Yc�A�'*

lossH3�<��       �	��0Yc�A�'*

loss�|�=y�       �	�90Yc�A�'*

loss�`<�Y�       �	*�0Yc�A�'*

loss���<�g~       �	��0Yc�A�'*

loss��<���       �	,0Yc�A�'*

loss���<�^#       �	��0Yc�A�'*

loss�ɞ<c�"       �	L�0Yc�A�'*

loss	CF<=�
�       �	Y40Yc�A�'*

lossl��;���|       �	_�0Yc�A�'*

loss���;�_�       �	g�0Yc�A�'*

loss#,=Ҫ�       �	�d0Yc�A�'*

loss�͌=ߜ1�       �	e 0Yc�A�'*

loss��{<���r       �	\�0Yc�A�'*

loss_+�<hp��       �	#,0Yc�A�'*

loss�]<���S       �	��0Yc�A�'*

loss�Y<f�u�       �	x_0Yc�A�'*

loss��2<��L�       �	�0Yc�A�'*

loss�N�;��f�       �	`�0Yc�A�'*

losst�O;��{�       �	�E0Yc�A�'*

loss;��:��       �	#�0Yc�A�'*

loss�k�<�P��       �	�p0Yc�A�'*

loss�8�;����       �	�0Yc�A�'*

lossO�=�,s�       �	�0Yc�A�'*

loss۲=B� {       �	�C0Yc�A�'*

loss<.<���       �	b�0Yc�A�'*

loss옗;/KS�       �	`s0Yc�A�'*

loss@��;�6xM       �	� 0Yc�A�'*

loss_~�=rJ�a       �	�� 0Yc�A�'*

loss�<!���       �	3!0Yc�A�'*

loss�i�=G3l       �	��!0Yc�A�'*

lossF{�<�       �	Z"0Yc�A�'*

loss�O{=����       �	��"0Yc�A�'*

loss�h=��L       �	8�#0Yc�A�'*

loss�dp<A�$K       �	�$0Yc�A�'*

lossm��<��Z       �	C�$0Yc�A�'*

loss�ʢ;Jv;�       �	�R%0Yc�A�'*

loss�r:<�ƣ�       �	��%0Yc�A�'*

loss%{=�H       �	�y&0Yc�A�'*

loss��s=� �?       �	o'0Yc�A�'*

loss�/�;���       �		�'0Yc�A�'*

loss�#�:����       �	�;(0Yc�A�'*

loss�j�;��?�       �	R�(0Yc�A�'*

loss�N�<UC�       �	�e)0Yc�A�'*

lossŶ�;Qc       �	�8*0Yc�A�'*

loss39=Q/�{       �	��*0Yc�A�'*

losse�<�p        �	Lq+0Yc�A�'*

loss�;?s�$       �	�,0Yc�A�'*

lossD&*<Xϸx       �	��,0Yc�A�'*

loss	_$=n�p       �	6r-0Yc�A�'*

lossE�<[��
       �	2<.0Yc�A�'*

loss�I =�?��       �	M�.0Yc�A�'*

loss�"j<,)Z       �	/o/0Yc�A�'*

loss X:��%8       �	x_00Yc�A�'*

loss�Z{;zW�       �	�m10Yc�A�'*

lossV�<c�M       �	�i20Yc�A�'*

loss���:H��K       �	�,30Yc�A�'*

lossvi;[:[       �	�,40Yc�A�'*

lossǨ<�=�       �	d�40Yc�A�'*

loss $=�+v       �	<�50Yc�A�'*

loss�0<��{C       �	�i60Yc�A�'*

loss��=�~q       �	s070Yc�A�'*

lossM��<�Ό       �	-�70Yc�A�'*

lossT�==��       �	�m80Yc�A�'*

loss�ǡ<����       �	90Yc�A�'*

lossͶ�=mJP       �	��90Yc�A�'*

lossv~�;m�<�       �	�U:0Yc�A�'*

lossɓ�;�5��       �	�:0Yc�A�'*

lossO~T<���       �	��;0Yc�A�'*

loss�!=Ս�E       �	q;<0Yc�A�'*

loss6�;�4        �	��<0Yc�A�'*

loss��<��h�       �	�s=0Yc�A�'*

loss_�_<j���       �	�>0Yc�A�'*

loss��:6�       �	��>0Yc�A�'*

lossHS[<b�Z�       �	=D?0Yc�A�'*

lossأ=����       �	��?0Yc�A�'*

loss�94=Y?:z       �	.t@0Yc�A�'*

loss��Q:��       �	f/A0Yc�A�'*

loss��;��       �	&�A0Yc�A�'*

lossv��<�gL1       �	$^B0Yc�A�'*

loss��";CP%�       �	iC0Yc�A�'*

loss?�<>B�       �	�C0Yc�A�'*

loss���<���       �	%>D0Yc�A�'*

loss��;j�[       �	��D0Yc�A�(*

lossn��<C�A�       �	�F0Yc�A�(*

loss}��<?sk       �	��F0Yc�A�(*

loss��<�%�       �	 `G0Yc�A�(*

lossa�:�ir       �	T�G0Yc�A�(*

loss�;�Yf�       �	.�H0Yc�A�(*

lossg8�<+q�X       �	NI0Yc�A�(*

loss���<��k�       �	��I0Yc�A�(*

loss��0;4��       �	_�J0Yc�A�(*

loss�%�<�3n�       �	qK0Yc�A�(*

loss6��<����       �	�L0Yc�A�(*

loss�J�;���n       �	L�L0Yc�A�(*

lossa��;��        �	�@M0Yc�A�(*

loss+�:b�L       �	��M0Yc�A�(*

loss�! <D��       �	NzN0Yc�A�(*

loss��<"v��       �	!O0Yc�A�(*

loss���<�1E�       �	��O0Yc�A�(*

loss�݊;�e��       �	�GP0Yc�A�(*

loss�;��m       �	Q�P0Yc�A�(*

loss:8n<�d�       �	UmQ0Yc�A�(*

loss�zR;���]       �	�R0Yc�A�(*

loss֯#;<�A       �	�R0Yc�A�(*

loss
�@;g�tX       �	�6S0Yc�A�(*

loss��<i��       �	��S0Yc�A�(*

loss�c:<�       �	W_T0Yc�A�(*

loss�i�<�Cَ       �	U0Yc�A�(*

loss�}�<5���       �	�U0Yc�A�(*

lossvc;�.�       �	�>V0Yc�A�(*

lossb�<��6�       �	�W0Yc�A�(*

loss���<���       �	m�W0Yc�A�(*

loss��L=��       �	�X0Yc�A�(*

loss�ed:��Bz       �	�>Y0Yc�A�(*

lossSZ�<�G�       �	��Y0Yc�A�(*

loss=��Y       �	?sZ0Yc�A�(*

loss"��<~�!       �	�[0Yc�A�(*

loss�=|�Y       �	o�[0Yc�A�(*

loss���:g�j9       �	�b\0Yc�A�(*

lossS
<����       �	�]0Yc�A�(*

loss?~.<����       �	|�]0Yc�A�(*

loss%B�=wz��       �	�8^0Yc�A�(*

loss�(<V�M�       �	�^0Yc�A�(*

lossYv=?�       �	�^_0Yc�A�(*

loss��a=CU�       �	��_0Yc�A�(*

loss:D�<��W       �	*�`0Yc�A�(*

loss�z�<�un       �	�(a0Yc�A�(*

lossѓ
<�M>       �	&�a0Yc�A�(*

loss�p<
�W       �	
�b0Yc�A�(*

loss6�6>����       �	cac0Yc�A�(*

loss���<���       �	��c0Yc�A�(*

loss��	<�t�O       �	K�d0Yc�A�(*

lossGG�<}i�`       �	�7e0Yc�A�(*

loss�#�<z�1U       �	��e0Yc�A�(*

loss�wn;ym�       �	�mf0Yc�A�(*

lossO�-;��`�       �	g0Yc�A�(*

loss���;ƶ��       �	Y�g0Yc�A�(*

loss�`<����       �	�h0Yc�A�(*

loss�)�=�       �	�Di0Yc�A�(*

loss��P=�p��       �	(�i0Yc�A�(*

loss�D=��M�       �	��j0Yc�A�(*

loss!�j<���       �	F@k0Yc�A�(*

loss8�;�Gg       �	�k0Yc�A�(*

loss>+�;Ni�s       �	$�l0Yc�A�(*

loss�e=J��r       �	�m0Yc�A�(*

loss;J�=%Z�       �	��m0Yc�A�(*

loss���;�m0       �	�Tn0Yc�A�(*

lossZ��:bJ��       �	��n0Yc�A�(*

loss�x�<�j�       �	��o0Yc�A�(*

loss:��<yK��       �	6!p0Yc�A�(*

loss�u�;�w�3       �	��p0Yc�A�(*

loss�a�:��_�       �	CTq0Yc�A�(*

loss�:�;�-?�       �	��q0Yc�A�(*

loss�N6=!8[?       �	2s0Yc�A�(*

loss,�<=p���       �	Y�s0Yc�A�(*

loss�U<���       �	�t0Yc�A�(*

loss4,�<��|�       �	1\u0Yc�A�(*

loss{�<�}1�       �	�v0Yc�A�(*

loss�ml=���<       �	��v0Yc�A�(*

lossS��:D�:�       �	�x0Yc�A�(*

loss�P�<�+       �	�x0Yc�A�(*

loss�0N<�"��       �	|ay0Yc�A�(*

loss��<J6       �	^�y0Yc�A�(*

loss��;�k�       �	G�z0Yc�A�(*

lossߌ�<Χ	�       �	�l{0Yc�A�(*

loss��;��K       �	�.|0Yc�A�(*

loss���<M��       �	6:}0Yc�A�(*

loss�\�=I5;       �	�K~0Yc�A�(*

loss��X;��v�       �	�~0Yc�A�(*

loss�rV<���t       �	�C�0Yc�A�(*

loss3Y�<�!�       �		�0Yc�A�(*

loss�j<!N       �	ﬁ0Yc�A�(*

loss8Z�;jb��       �	�U�0Yc�A�(*

lossݤ�<���       �	�0Yc�A�(*

loss�ة=[��       �	���0Yc�A�(*

loss��;�{�       �	�-�0Yc�A�(*

loss��$=2�K�       �	n0Yc�A�(*

loss���;&F��       �	(d�0Yc�A�(*

loss�%=�a�r       �	C��0Yc�A�(*

loss���;E��       �	k��0Yc�A�(*

lossD�&<|�<�       �	i5�0Yc�A�(*

loss}܂=2�G       �	�Շ0Yc�A�(*

lossM�<b�       �	/n�0Yc�A�(*

loss�8;���Q       �	��0Yc�A�(*

loss-�<Et�t       �	Z��0Yc�A�(*

loss�&�;)2i�       �	"4�0Yc�A�(*

loss�>~=�)k       �	�Ȋ0Yc�A�(*

lossxH�<��6       �	$��0Yc�A�(*

loss�C4=���       �	>=�0Yc�A�(*

lossr�;s��p       �	P7�0Yc�A�(*

lossu2=|L#�       �	�0Yc�A�(*

lossjiS<�fm~       �	m��0Yc�A�(*

loss���;H�Py       �	�5�0Yc�A�(*

loss�<�W;G       �	�ߏ0Yc�A�(*

loss��-;�Z�       �	���0Yc�A�(*

loss��;JN8�       �	�.�0Yc�A�(*

lossz�<=�"       �	�
�0Yc�A�(*

lossl�R<E���       �	���0Yc�A�(*

loss�7=��gF       �	�N�0Yc�A�(*

lossg";�i�       �	J�0Yc�A�(*

lossF�;�W�       �	*��0Yc�A�(*

loss$[0;��ye       �	#1�0Yc�A�(*

loss�	�<:	z       �	5ҕ0Yc�A�(*

loss_�=2�       �	6w�0Yc�A�(*

lossg</��       �	��0Yc�A�(*

loss���<B�.       �	;Ɨ0Yc�A�(*

loss�=ځ�       �	Eg�0Yc�A�(*

loss]��9�+�&       �	I�0Yc�A�(*

loss��+<���       �	���0Yc�A�(*

loss�=0��       �	Z�0Yc�A�(*

loss�*�;�"�v       �	���0Yc�A�)*

loss4�<�m;�       �	���0Yc�A�)*

loss��;๬       �	fJ�0Yc�A�)*

lossd{<S�R       �	`�0Yc�A�)*

loss���;�#j       �	��0Yc�A�)*

loss�<	Ls       �	���0Yc�A�)*

lossc#�<uq�        �	�3�0Yc�A�)*

lossئ>zL�`       �	Qݟ0Yc�A�)*

loss ��;6QV6       �	�|�0Yc�A�)*

lossl�<<�DT�       �	��0Yc�A�)*

loss܃S<�Iȱ       �	yΡ0Yc�A�)*

loss��!;'�߆       �	rn�0Yc�A�)*

lossN>�<���       �	 �0Yc�A�)*

loss��:��P       �	vߣ0Yc�A�)*

lossZSZ;,�}�       �	�x�0Yc�A�)*

lossU�<�Z��       �	��0Yc�A�)*

loss���=�A��       �	9�0Yc�A�)*

loss	-<� �       �	NԦ0Yc�A�)*

loss0i =�n^N       �	Dj�0Yc�A�)*

loss`J<�M�#       �	��0Yc�A�)*

loss\��: �#	       �	��0Yc�A�)*

loss�J	=$v�D       �	�,�0Yc�A�)*

loss��:�c       �	!̩0Yc�A�)*

loss��>��+f       �	�c�0Yc�A�)*

loss�<H<26pt       �	���0Yc�A�)*

loss�h�;f�       �	���0Yc�A�)*

lossх=-Ȏk       �	A)�0Yc�A�)*

loss�o<��ۑ       �	�¬0Yc�A�)*

loss�8<'�&�       �	�d�0Yc�A�)*

lossq�e;��2�       �	�#�0Yc�A�)*

lossI'=��zc       �	,��0Yc�A�)*

loss:�w<=��8       �	GX�0Yc�A�)*

loss���=��        �	^��0Yc�A�)*

loss֔�<��(�       �	���0Yc�A�)*

lossxT=}��       �	M,�0Yc�A�)*

lossX��<G��`       �	pͱ0Yc�A�)*

lossj�9����       �	�r�0Yc�A�)*

loss-�;���       �	�E�0Yc�A�)*

loss��;��;       �	��0Yc�A�)*

loss�#�;���U       �	��0Yc�A�)*

loss�|g=G/�       �	a��0Yc�A�)*

lossqF);��"�       �	�#�0Yc�A�)*

loss�2�<�#�       �	Ƕ0Yc�A�)*

loss��.;��P       �	3l�0Yc�A�)*

loss%�y=*��e       �	��0Yc�A�)*

loss��!;����       �	̶�0Yc�A�)*

loss�1y;	�N�       �	]P�0Yc�A�)*

loss�R;?1mE       �	���0Yc�A�)*

loss�'�:;�       �	�$�0Yc�A�)*

loss�2z;�5�Q       �		ƻ0Yc�A�)*

loss��t;����       �	�f�0Yc�A�)*

lossw(Z<I��X       �	v��0Yc�A�)*

loss.N�<�%��       �	9��0Yc�A�)*

loss��=��9�       �	�3�0Yc�A�)*

lossc�<�(�}       �	Cʾ0Yc�A�)*

lossF,=_XEN       �	c�0Yc�A�)*

lossP4:B���       �	�0Yc�A�)*

loss{:�=���[       �	?��0Yc�A�)*

lossD�q;�Wd2       �	A�0Yc�A�)*

loss���<�` q       �	7��0Yc�A�)*

lossT��;1f\{       �	���0Yc�A�)*

loss�=�@ܩ       �	�`�0Yc�A�)*

loss�'J="�       �	H��0Yc�A�)*

loss#a�<;1;�       �	���0Yc�A�)*

loss�	=�7�       �	E/�0Yc�A�)*

loss�{�<��       �	���0Yc�A�)*

loss(�<PX�I       �	�c�0Yc�A�)*

loss�L,=�v       �	���0Yc�A�)*

loss*�<�       �	c��0Yc�A�)*

loss��|=f��}       �	�>�0Yc�A�)*

lossx��<�a�`       �	���0Yc�A�)*

lossҗ�:jՔ_       �	:z�0Yc�A�)*

loss{��;�埽       �	�V�0Yc�A�)*

loss�w�:����       �	��0Yc�A�)*

loss��<:��       �	"��0Yc�A�)*

loss�J�;0�	       �	V+�0Yc�A�)*

loss�%d<	A��       �	�J�0Yc�A�)*

lossρ�<�Ϲ�       �	���0Yc�A�)*

losst$ <Te       �	a��0Yc�A�)*

lossv�<Śrj       �	�;�0Yc�A�)*

loss��<#�N�       �	���0Yc�A�)*

loss?�a=��O�       �	�'�0Yc�A�)*

loss�O;��2�       �	��0Yc�A�)*

loss��B;����       �	�v�0Yc�A�)*

loss.-�<ԏ~�       �	��0Yc�A�)*

loss�i�<���       �	���0Yc�A�)*

loss�875}O       �	KY�0Yc�A�)*

lossm:�Y       �	H��0Yc�A�)*

loss3�3:���       �	���0Yc�A�)*

lossF�;^D0       �	X<�0Yc�A�)*

lossV�d;�HM       �	t��0Yc�A�)*

loss"V�9	I��       �	ڌ�0Yc�A�)*

loss.nJ;Q�       �	�0�0Yc�A�)*

loss(��;�S�       �	���0Yc�A�)*

loss���8���       �	�n�0Yc�A�)*

loss�a8�:�       �	�0Yc�A�)*

loss3�<���       �	ޮ�0Yc�A�)*

lossa22<�WkB       �	YR�0Yc�A�)*

loss��&<&Mc       �	���0Yc�A�)*

loss7p1:�_W       �	C��0Yc�A�)*

loss���<
p�       �	�.�0Yc�A�)*

lossh*�=#���       �	t��0Yc�A�)*

lossŞ9f��       �	�t�0Yc�A�)*

loss:��=�9��       �	{�0Yc�A�)*

loss�`�=ɮ�i       �	X��0Yc�A�)*

loss�=���       �	W?�0Yc�A�)*

loss���;	+       �	���0Yc�A�)*

loss8:@���       �	m�0Yc�A�)*

loss�f>V�_�       �	)�0Yc�A�)*

loss�i%<@�^�       �	Ǡ�0Yc�A�)*

loss�v:=_W�       �	5B�0Yc�A�)*

lossa�;��       �	���0Yc�A�)*

loss3��:pN\;       �	v�0Yc�A�)*

loss<�=�$��       �	xF�0Yc�A�)*

loss�c�<9���       �	U�0Yc�A�)*

loss�c�<�~x       �	��0Yc�A�)*

loss��
=l&�\       �	���0Yc�A�)*

loss�G=�T�       �	�=�0Yc�A�)*

loss�h=,�n       �	���0Yc�A�)*

lossÓ�<s�C�       �	9��0Yc�A�)*

loss��3=��&!       �	=��0Yc�A�)*

loss״R=3�F)       �	zr�0Yc�A�)*

lossc�;�,^�       �	�'�0Yc�A�)*

loss_�<)�Z�       �	h��0Yc�A�)*

lossL<����       �	Uj�0Yc�A�)*

loss!�/;��       �	�0Yc�A�)*

loss_�:ZJ�       �	[��0Yc�A�)*

loss���;,%�       �	�M�0Yc�A�)*

loss���=~�=       �	���0Yc�A�**

loss�<9�t       �	��0Yc�A�**

lossҗ%<]��{       �	�2�0Yc�A�**

loss��<�p       �	f��0Yc�A�**

loss'F�;N�W�       �	�v�0Yc�A�**

loss%�L<d���       �	��0Yc�A�**

loss��B;s�J�       �	[��0Yc�A�**

loss�&;�x~G       �	6r�0Yc�A�**

loss�(U<>>�       �	��0Yc�A�**

loss�e$=H��       �	���0Yc�A�**

loss��=W_f�       �	S@�0Yc�A�**

loss1e
=���       �	���0Yc�A�**

loss9=�bV       �	�o�0Yc�A�**

loss�b�<Ț��       �	��0Yc�A�**

lossؖ<2*c�       �	���0Yc�A�**

loss7�;���       �	�2�0Yc�A�**

loss!e<M�׽       �	���0Yc�A�**

loss�t<-
d       �	�b�0Yc�A�**

loss@3�:Ј�:       �	j��0Yc�A�**

loss?u�<xw3�       �	�o�0Yc�A�**

loss2I=����       �	��0Yc�A�**

loss� ;N+��       �	���0Yc�A�**

loss�l�;D��{       �	�E�0Yc�A�**

loss��<�/��       �	���0Yc�A�**

loss�d�<��i�       �	���0Yc�A�**

loss�<���       �	c�1Yc�A�**

lossK;�/��       �	DQ1Yc�A�**

loss�p�<&�       �	��1Yc�A�**

loss�;�1Q}       �	��1Yc�A�**

loss��k=�q*�       �	e�1Yc�A�**

loss!�&<3�׽       �	O�1Yc�A�**

loss�!a<`�n       �	�E 1Yc�A�**

loss��S=�5��       �	� 1Yc�A�**

loss�~�<B~�3       �	�|!1Yc�A�**

loss&'_<�ht�       �	#"1Yc�A�**

loss���:$St!       �	��"1Yc�A�**

loss�D<d2��       �	.U#1Yc�A�**

lossh��<󋇱       �	�$1Yc�A�**

lossT�;���{       �	9�$1Yc�A�**

loss7K<-��       �	�h%1Yc�A�**

lossZ(T;#��y       �	C&1Yc�A�**

lossf��9Pt�       �	��&1Yc�A�**

lossȋ	<�L       �	�='1Yc�A�**

loss�<��a�       �	��'1Yc�A�**

lossD|1<�`�9       �	�(1Yc�A�**

loss��<o�bc       �	?)1Yc�A�**

loss�R<՜�X       �	�)1Yc�A�**

loss�,�;j�N       �	��*1Yc�A�**

lossX;x<��N       �	|'+1Yc�A�**

loss�%�<�{n�       �	��+1Yc�A�**

lossvW�:��       �	Wz,1Yc�A�**

loss��s<�!�H       �	C-1Yc�A�**

lossv�=2|�       �	s�-1Yc�A�**

loss��=��P�       �	�U.1Yc�A�**

lossɓ%<Д׉       �	��.1Yc�A�**

lossq�<�#.       �	P�/1Yc�A�**

loss�s�=��       �	�"01Yc�A�**

loss+;�%\�       �	%�01Yc�A�**

loss
�=��1       �	V�11Yc�A�**

loss�p�<et2�       �	�!21Yc�A�**

loss�=b��1       �	��21Yc�A�**

loss=��<�]A       �	*r31Yc�A�**

loss�j@<2��       �	*�41Yc�A�**

loss
��=��S       �	,F51Yc�A�**

loss��<˖am       �	J�61Yc�A�**

loss��:�z       �	=�71Yc�A�**

loss�F<����       �	�}81Yc�A�**

loss�r4=�p�W       �	�-91Yc�A�**

lossKm;�2z       �	P�91Yc�A�**

loss�T�<T,�       �	D�:1Yc�A�**

loss�Ǭ<�h�T       �	[%;1Yc�A�**

lossx�j;}>       �	��;1Yc�A�**

loss��'=A�̓       �	�_<1Yc�A�**

loss;�;�7       �	. =1Yc�A�**

loss��;�:�3       �	)�=1Yc�A�**

loss�d=:gӪo       �	->1Yc�A�**

loss�NG=��@       �	��>1Yc�A�**

loss���;Lt��       �	m?1Yc�A�**

loss���=:�JQ       �	]S@1Yc�A�**

loss�b;3Fj
       �	��@1Yc�A�**

lossx\m:)�к       �	8�A1Yc�A�**

loss�Yo;$:&$       �	<KB1Yc�A�**

loss��9B�]�       �	z�B1Yc�A�**

loss��<'�x(       �	�{C1Yc�A�**

loss7x�<rU       �	mD1Yc�A�**

lossM'�=�5o       �	̴D1Yc�A�**

loss��><6���       �	�GE1Yc�A�**

loss��_;�/��       �	��E1Yc�A�**

loss7E�:h�DD       �	��F1Yc�A�**

loss��;/�`+       �	kHG1Yc�A�**

loss�`�;��N        �	��G1Yc�A�**

lossrJ=!;��       �	CqH1Yc�A�**

lossx�<�Q�1       �	F@I1Yc�A�**

loss���=�h       �	�I1Yc�A�**

loss�8�<�.       �	�pJ1Yc�A�**

loss�e<�o��       �	>K1Yc�A�**

lossױ^;O��x       �	�K1Yc�A�**

loss���<���       �	<LL1Yc�A�**

loss'!<aG�:       �	�+M1Yc�A�**

lossx�X=T���       �	��M1Yc�A�**

lossa7�;f>��       �	�gN1Yc�A�**

lossɄa<��t       �	rO1Yc�A�**

loss��<��C       �	�
P1Yc�A�**

lossw�<<����       �	�P1Yc�A�**

lossr=���       �	�EQ1Yc�A�**

lossMf�;����       �	��Q1Yc�A�**

loss��g<��45       �	��R1Yc�A�**

loss�D<�l�       �	�9S1Yc�A�**

lossH�:�4�l       �	��S1Yc�A�**

loss��:S��       �	\U1Yc�A�**

loss�)�<�c��       �	��U1Yc�A�**

loss���;~��'       �	��V1Yc�A�**

loss�<��~j       �	�1W1Yc�A�**

lossA�/;�o:�       �	��W1Yc�A�**

loss�3<�ȥ�       �	&pX1Yc�A�**

loss�<Z&o�       �	Y1Yc�A�**

lossE <�Ĺ`       �	�Y1Yc�A�**

lossC<�,.�       �	:Z1Yc�A�**

loss3�<�*�e       �	(�Z1Yc�A�**

lossů�<Ci��       �	�k[1Yc�A�**

loss�><v��       �	U\1Yc�A�**

loss j2=�/�       �	�\1Yc�A�**

loss���;�f�       �	�T]1Yc�A�**

loss��=>Ey{       �	e�]1Yc�A�**

loss��T=��y�       �	��^1Yc�A�**

loss��;x5       �	�l_1Yc�A�**

loss���<$�>       �	I`1Yc�A�**

loss��<q��       �	��`1Yc�A�**

lossT�[;�v5       �	+Ka1Yc�A�+*

loss�#�:=�m*       �	;�a1Yc�A�+*

loss�uF<�^�       �	t~b1Yc�A�+*

lossP��;� �-       �	�%c1Yc�A�+*

lossC�<|���       �	x�c1Yc�A�+*

loss�e*=��r7       �	�kd1Yc�A�+*

loss��*;C��       �	^e1Yc�A�+*

lossX��<˲�       �	��e1Yc�A�+*

loss2�F=��B       �	�;f1Yc�A�+*

loss�J�<@�0o       �	n�f1Yc�A�+*

loss��;8��[       �	σg1Yc�A�+*

loss�X;���       �	f-h1Yc�A�+*

lossd�V<���K       �	��h1Yc�A�+*

loss�:< ��e       �	��i1Yc�A�+*

lossw�a=�y"       �	j0j1Yc�A�+*

loss�6,=����       �	b�j1Yc�A�+*

loss$��;Sz�W       �	��k1Yc�A�+*

loss���;OM@�       �	L5l1Yc�A�+*

loss�Z<I9��       �	f�l1Yc�A�+*

loss��;n�x       �	�m1Yc�A�+*

loss���;��Z�       �	W!n1Yc�A�+*

loss�͂;A�(�       �	G�n1Yc�A�+*

loss�"'<޹��       �	c_o1Yc�A�+*

loss�jD<�b]       �	/�o1Yc�A�+*

lossi��<� �p       �	[�p1Yc�A�+*

loss<#�<��k�       �	�9q1Yc�A�+*

losshq=�E{       �	4�q1Yc�A�+*

loss���:�0C`       �	zrr1Yc�A�+*

loss�	=Б��       �	�s1Yc�A�+*

loss�ݭ<���       �	7�s1Yc�A�+*

loss\5�:���       �	u>t1Yc�A�+*

loss��<y�_�       �	I�t1Yc�A�+*

loss��=��$       �	�qu1Yc�A�+*

loss�$<�G       �	�v1Yc�A�+*

loss/�;&"       �	��v1Yc�A�+*

loss$r%=�/��       �	a8w1Yc�A�+*

lossa�;U��:       �	,�w1Yc�A�+*

loss��;U�}3       �	?�x1Yc�A�+*

loss�7�;t���       �	O@y1Yc�A�+*

loss�B}=�86�       �	��y1Yc�A�+*

loss���;t�!�       �	�nz1Yc�A�+*

lossT
�<Е�       �	�
{1Yc�A�+*

loss<�v��       �	ȱ{1Yc�A�+*

loss�۶;�f�       �	=`|1Yc�A�+*

loss�Ss<����       �	��|1Yc�A�+*

lossc��9)?       �	��}1Yc�A�+*

loss��:=I�       �	�)~1Yc�A�+*

lossi��;z3J�       �	��~1Yc�A�+*

loss�\8<�8i       �	�X1Yc�A�+*

loss�i=���       �	��1Yc�A�+*

loss��H<Y5�       �	攀1Yc�A�+*

loss�i�<�"
�       �	�-�1Yc�A�+*

loss>�;�'?       �	Á1Yc�A�+*

loss��9/wgi       �	�˂1Yc�A�+*

loss>��=�n��       �	��1Yc�A�+*

loss�A=o��       �	2U�1Yc�A�+*

lossa׀;{m�       �	��1Yc�A�+*

losse�<��m�       �	I��1Yc�A�+*

loss��<��x@       �	���1Yc�A�+*

loss��<�m�       �	2�1Yc�A�+*

lossГ:b�'�       �	Ǉ1Yc�A�+*

loss)�<A_.�       �	@h�1Yc�A�+*

loss&�);Ȇ	;       �	C�1Yc�A�+*

lossҨ(:{Q�9       �	K��1Yc�A�+*

loss�);p6b�       �	�0�1Yc�A�+*

loss<l;�A       �	w׊1Yc�A�+*

loss��f<��)�       �	-{�1Yc�A�+*

loss(h;3���       �	��1Yc�A�+*

loss�3;w;4       �	C��1Yc�A�+*

loss �R=���       �	�Q�1Yc�A�+*

losso}<#�h�       �	y�1Yc�A�+*

loss��:��6�       �	6��1Yc�A�+*

loss�D�;�       �	��1Yc�A�+*

loss_�2=�	�       �	���1Yc�A�+*

lossV�)=^t�)       �	^*�1Yc�A�+*

loss&�g:�
��       �	~ȑ1Yc�A�+*

loss��<X��q       �	�`�1Yc�A�+*

loss%o1;���6       �	w��1Yc�A�+*

loss��;zg�       �	���1Yc�A�+*

loss?Nm<�0�$       �	 7�1Yc�A�+*

loss8��:����       �	[Ҕ1Yc�A�+*

loss��=��[e       �	j4�1Yc�A�+*

loss�Kc:A�U       �	�̖1Yc�A�+*

loss���<Ǧ�<       �	�d�1Yc�A�+*

lossW_=LóQ       �	q�1Yc�A�+*

loss��$;,V��       �	�̘1Yc�A�+*

loss@��<�[       �	al�1Yc�A�+*

lossh2�=�m�]       �	u�1Yc�A�+*

loss��;:uX�       �	^��1Yc�A�+*

loss�Z$;x���       �	�6�1Yc�A�+*

loss.q�;�r�'       �	�̛1Yc�A�+*

loss
��9�p�}       �	e�1Yc�A�+*

loss���;��2h       �	���1Yc�A�+*

lossא<<���{       �	���1Yc�A�+*

loss���<3�:�       �	�<�1Yc�A�+*

loss���<Q�\       �	�Ӟ1Yc�A�+*

loss?N\;��qw       �	3j�1Yc�A�+*

loss�m<���4       �	��1Yc�A�+*

loss�v�;�}��       �	z��1Yc�A�+*

lossx�<�X<�       �	hB�1Yc�A�+*

losss3&<�_<�       �	nۡ1Yc�A�+*

lossq�<�i       �	E��1Yc�A�+*

lossR%�;�� *       �	3�1Yc�A�+*

loss�h=�G��       �	���1Yc�A�+*

lossf?�<m�l       �	aR�1Yc�A�+*

loss�l<��Q       �	��1Yc�A�+*

loss�"
=��iX       �	�x�1Yc�A�+*

lossȮQ<��%�       �	$�1Yc�A�+*

loss\�;�>�       �	禦1Yc�A�+*

loss�r<e�+       �	�?�1Yc�A�+*

lossZo�9�#\�       �	n�1Yc�A�+*

lossc��;����       �	v��1Yc�A�+*

lossTJB<b�VB       �	�F�1Yc�A�+*

loss��X> �"�       �	��1Yc�A�+*

lossnJ<8�d       �	Q��1Yc�A�+*

loss��=���;       �	{.�1Yc�A�+*

loss@�=��!	       �	�ǫ1Yc�A�+*

losse��;˔       �	^�1Yc�A�+*

loss��!=l�v       �	%�1Yc�A�+*

loss�\�;���       �	P��1Yc�A�+*

loss�h�<H]�       �	�U�1Yc�A�+*

loss��;�<�z       �	b��1Yc�A�+*

loss=��;��ns       �	��1Yc�A�+*

loss��Q<6�~~       �	�/�1Yc�A�+*

lossl�;p}mu       �	�װ1Yc�A�+*

loss&6;C��       �	Xp�1Yc�A�+*

lossi��;����       �	6�1Yc�A�+*

loss��<�\T        �	N��1Yc�A�+*

loss��<��#4       �	Q3�1Yc�A�,*

loss��]<ȍV       �	�P�1Yc�A�,*

loss|��<l       �	9&�1Yc�A�,*

loss*��<ă�m       �	.�1Yc�A�,*

lossJ\�;�pyC       �	�0�1Yc�A�,*

loss�Yi;q��A       �	*ʷ1Yc�A�,*

losst�=Y�!       �	~�1Yc�A�,*

loss&:<M�͈       �	��1Yc�A�,*

loss��:nk��       �	�̹1Yc�A�,*

loss _~: w�$       �	l�1Yc�A�,*

loss��;H�o�       �	S�1Yc�A�,*

lossr;�;q��       �	�»1Yc�A�,*

lossÙ�<�9��       �	�t�1Yc�A�,*

loss���<����       �	��1Yc�A�,*

loss���;�Ê       �	���1Yc�A�,*

loss��c:Y��7       �	fK�1Yc�A�,*

loss/;��       �	k�1Yc�A�,*

loss7�2=BS�       �	���1Yc�A�,*

lossEH�;/�Y�       �	�5�1Yc�A�,*

loss��T;T��k       �	���1Yc�A�,*

loss�JH<�5�       �	�b�1Yc�A�,*

loss�� =8��K       �	���1Yc�A�,*

loss ө9�&S       �	��1Yc�A�,*

loss��;̋��       �	B%�1Yc�A�,*

loss熉<7��W       �	ܺ�1Yc�A�,*

loss6|2=�ꗏ       �	�R�1Yc�A�,*

loss
�<���o       �	:��1Yc�A�,*

lossr�;ޙ��       �	V��1Yc�A�,*

loss�w<���f       �	��1Yc�A�,*

loss�Յ</��\       �	���1Yc�A�,*

loss�ĕ<7o�       �	�?�1Yc�A�,*

loss�&<a��t       �	w��1Yc�A�,*

loss�<��\       �	m�1Yc�A�,*

loss�t<YHoI       �	K�1Yc�A�,*

lossfz�<C��       �	U��1Yc�A�,*

lossTIE;G˨       �	*r�1Yc�A�,*

loss;2<�l�I       �	1�1Yc�A�,*

loss��=X�       �	��1Yc�A�,*

loss�qz<$��	       �	6=�1Yc�A�,*

loss.�P<[���       �	M��1Yc�A�,*

loss� <!���       �	�o�1Yc�A�,*

loss��>;��y       �	��1Yc�A�,*

loss&|�=*�X�       �	R��1Yc�A�,*

loss���<l ��       �	<1�1Yc�A�,*

loss�ņ=�<       �	��1Yc�A�,*

loss�Y/<�)<�       �	Nc�1Yc�A�,*

lossd�<M�W�       �	��1Yc�A�,*

loss��=M�O�       �	r��1Yc�A�,*

losss�s;w�c       �	s.�1Yc�A�,*

loss��9e6ʦ       �	r��1Yc�A�,*

loss�o7;�b�V       �	�W�1Yc�A�,*

loss%�:J�)�       �	��1Yc�A�,*

loss�;�5r�       �	_��1Yc�A�,*

lossq�W<��f6       �	O�1Yc�A�,*

loss�7;�E@       �	���1Yc�A�,*

loss��<���?       �	^��1Yc�A�,*

loss�;���N       �		�1Yc�A�,*

lossăZ=�͝�       �	���1Yc�A�,*

loss3�2:K�af       �	�C�1Yc�A�,*

loss=ɀ:x�o2       �	7��1Yc�A�,*

lossm��;����       �	�t�1Yc�A�,*

loss�S�<pR�       �	��1Yc�A�,*

loss�!;�4��       �	���1Yc�A�,*

lossF=���]       �	�Q�1Yc�A�,*

loss��o=*��       �	~��1Yc�A�,*

loss�Y<5s�	       �	=�1Yc�A�,*

lossB�<��       �	��1Yc�A�,*

loss{��;<�(       �	1��1Yc�A�,*

loss��2=5��       �	U�1Yc�A�,*

lossN�<�$x       �	���1Yc�A�,*

loss&�4<W�yo       �	`��1Yc�A�,*

loss4T;�NА       �	$�1Yc�A�,*

loss�<��-A       �	=��1Yc�A�,*

loss�H3;��       �	@L�1Yc�A�,*

loss���="Im        �	X��1Yc�A�,*

lossjB�;u��0       �	�w�1Yc�A�,*

loss�;��B�       �	I�1Yc�A�,*

loss��k;o��       �	���1Yc�A�,*

loss��<����       �	�t�1Yc�A�,*

loss��;���z       �	J�1Yc�A�,*

loss�<��/       �	���1Yc�A�,*

lossz��:�A��       �	F?�1Yc�A�,*

loss�:<*�
J       �	���1Yc�A�,*

loss{��;Q"�Z       �	ni�1Yc�A�,*

loss�C�<�&�n       �	��1Yc�A�,*

loss}��;��z       �	)��1Yc�A�,*

loss�0=�s��       �	=+�1Yc�A�,*

loss:�<��       �	D��1Yc�A�,*

loss ��<4?L       �	�S�1Yc�A�,*

loss=�\<���       �	���1Yc�A�,*

loss�H>�Zٮ       �	=��1Yc�A�,*

lossO.�<�BD       �	~7�1Yc�A�,*

loss�y�:ŝ��       �	$��1Yc�A�,*

loss��+<Ti�2       �	�o�1Yc�A�,*

loss��;���       �	��1Yc�A�,*

loss�d�;�[��       �	
��1Yc�A�,*

loss�jT;o��       �	{K�1Yc�A�,*

loss&hK=ѱ��       �	���1Yc�A�,*

lossiR3=I�"�       �	H��1Yc�A�,*

loss��;�G|       �	'3�1Yc�A�,*

loss�<G�;�       �	���1Yc�A�,*

loss��;_�4       �	Qj�1Yc�A�,*

loss�(<0
`       �	}�1Yc�A�,*

losso>�<���B       �	D��1Yc�A�,*

loss�c�<�Kl       �	-@�1Yc�A�,*

loss�>y=��2       �	���1Yc�A�,*

loss���<���5       �	�n�1Yc�A�,*

loss�=Ӂ�$       �	�'�1Yc�A�,*

loss���;7G�       �	4��1Yc�A�,*

loss��d;�=�/       �	�S�1Yc�A�,*

loss�&<�h�D       �	[��1Yc�A�,*

loss��V=���       �	$��1Yc�A�,*

loss]��;b��       �	2�1Yc�A�,*

lossab=:��G       �	���1Yc�A�,*

loss�5=RO&       �	�I�1Yc�A�,*

lossTCd<�� @       �	?��1Yc�A�,*

lossai�=��	�       �	΍�1Yc�A�,*

lossJ��<�Ǉ       �	�#�1Yc�A�,*

loss�,< �:&       �	���1Yc�A�,*

loss�S;��5       �	�j�1Yc�A�,*

loss�
<?�       �	t�1Yc�A�,*

loss&��=�v��       �	i��1Yc�A�,*

loss�:��+�       �	�_ 2Yc�A�,*

loss�]n=��n       �	j� 2Yc�A�,*

loss���;ɇ��       �	 �2Yc�A�,*

loss���<�B�U       �	�82Yc�A�,*

losst�<�QO       �	v�2Yc�A�,*

lossF��=�Y�       �	��2Yc�A�,*

loss��;<��}       �	4.2Yc�A�-*

loss�H�;�J��       �	�2Yc�A�-*

loss��];A�`g       �	��2Yc�A�-*

loss�Ib<x=_       �	Y32Yc�A�-*

loss)�m=h{��       �	r�2Yc�A�-*

loss=YH<qP��       �	�y2Yc�A�-*

lossL3'=��(�       �	�2Yc�A�-*

loss���<��D       �	E�2Yc�A�-*

lossw�<�M��       �	c	2Yc�A�-*

lossZ%"=�X�       �	

2Yc�A�-*

loss%�=��T       �	X�
2Yc�A�-*

loss���<�w��       �	N�2Yc�A�-*

loss��<��o�       �	�s2Yc�A�-*

loss)�n<��/�       �	�2Yc�A�-*

loss�/<_�ٝ       �	��2Yc�A�-*

loss���;7Ċ�       �	�C2Yc�A�-*

lossDӈ<�m��       �	,�2Yc�A�-*

loss���<�3�       �	�u2Yc�A�-*

loss촞<5��V       �	2Yc�A�-*

lossƔK;���       �	ȱ2Yc�A�-*

loss^x=�]K{       �	RG2Yc�A�-*

loss��9<�(�       �	l�2Yc�A�-*

loss���<(׻�       �	�~2Yc�A�-*

loss�>��x�       �	�02Yc�A�-*

loss��?=z��       �	�D2Yc�A�-*

loss�Fv<2��?       �	D�2Yc�A�-*

loss�)<�yd       �	�2Yc�A�-*

loss�4�;.�8[       �	D�2Yc�A�-*

lossd��=�=�       �	�U2Yc�A�-*

lossT�!<+Z�       �	��2Yc�A�-*

loss�.�:W{�n       �	��2Yc�A�-*

loss�"'<Fb�&       �	ǃ2Yc�A�-*

loss ��;O���       �	�2Yc�A�-*

lossj<AAhE       �	�I2Yc�A�-*

loss
0�<��l.       �	O�2Yc�A�-*

loss�D<�;e       �	��2Yc�A�-*

loss�=��+       �	F"2Yc�A�-*

loss|`;lY�l       �	��2Yc�A�-*

loss|�<`��       �	�{2Yc�A�-*

lossӻ�;�q       �	�2Yc�A�-*

loss/�=�V��       �	�2Yc�A�-*

loss�Ğ;�Õ`       �	ZG 2Yc�A�-*

loss���;��j       �	�� 2Yc�A�-*

loss��;��!       �	ݖ!2Yc�A�-*

loss�a�:�;��       �	�*"2Yc�A�-*

loss��"=���G       �	R�"2Yc�A�-*

loss���;���       �	vq#2Yc�A�-*

loss��J=&Ǣ       �	Fa$2Yc�A�-*

loss�A�;�A�i       �	�$2Yc�A�-*

lossE�/;�B       �	��%2Yc�A�-*

loss2+|<>8,       �	�F&2Yc�A�-*

loss8�n<1��       �	U�&2Yc�A�-*

loss*�5=���       �	Sv'2Yc�A�-*

loss��<)�%E       �	�(2Yc�A�-*

loss�<�D"8       �	s�(2Yc�A�-*

lossQa�<�ʛ�       �	�?)2Yc�A�-*

lossm�<��       �	��)2Yc�A�-*

lossmQi;s��}       �	�i*2Yc�A�-*

loss�l <�T�5       �	�*2Yc�A�-*

losscd=n_Ƹ       �	��+2Yc�A�-*

loss��[<xb       �	w�,2Yc�A�-*

loss���;�5�       �	�-2Yc�A�-*

loss���<˳7'       �	C�-2Yc�A�-*

loss
:;%7�       �	xC.2Yc�A�-*

loss;nB=B8Gu       �	
�.2Yc�A�-*

losss��<�>9�       �	k/2Yc�A�-*

loss� V<n8�T       �	L�/2Yc�A�-*

lossa��<�8�       �	8�02Yc�A�-*

loss�"=�ί�       �	�=12Yc�A�-*

loss��<܍�.       �	��12Yc�A�-*

loss
;���       �	m22Yc�A�-*

loss=�;A��       �	��22Yc�A�-*

loss�=��f       �	��32Yc�A�-*

loss���<��a       �	qZ42Yc�A�-*

lossJ��<�YM�       �	:$52Yc�A�-*

lossc�<Հ1�       �	R�52Yc�A�-*

loss�w�<�jpj       �	yt62Yc�A�-*

loss.�<޿�       �	�:72Yc�A�-*

loss���;�GI�       �	�82Yc�A�-*

loss���;A�W�       �	��82Yc�A�-*

loss\�<�Θ�       �	?:92Yc�A�-*

lossf"�=ܙ-e       �	k�92Yc�A�-*

lossZ�<�� �       �	in:2Yc�A�-*

loss�7=��4�       �	�7;2Yc�A�-*

lossK��<LS6k       �	�o<2Yc�A�-*

lossC�;��p,       �	e=2Yc�A�-*

losst��;�l6�       �	r>2Yc�A�-*

lossS�=�,�       �	�?2Yc�A�-*

loss�|=�H��       �	�I@2Yc�A�-*

loss�8�<=�       �	1(A2Yc�A�-*

loss{�;��#       �	��A2Yc�A�-*

loss�y<^�       �	]�B2Yc�A�-*

lossQ <Q~>       �	XC2Yc�A�-*

loss���;g��r       �	��C2Yc�A�-*

lossʒ<��0       �	��D2Yc�A�-*

loss���=jU�5       �	��E2Yc�A�-*

loss1�E<�xJ+       �	8kF2Yc�A�-*

loss�[�;���       �	� G2Yc�A�-*

loss���;4U       �	�H2Yc�A�-*

loss��;�~��       �	��H2Yc�A�-*

loss��<;OȐ       �	��I2Yc�A�-*

loss��;A �       �	�IJ2Yc�A�-*

loss�<7f8h       �	(K2Yc�A�-*

lossZ��;߷~       �	{L2Yc�A�-*

loss0		>����       �	>�L2Yc�A�-*

loss�q<Pt        �	�NM2Yc�A�-*

lossT$�:s�       �	�N2Yc�A�-*

loss�K�:��6       �	CO2Yc�A�-*

loss	Z7;J"b       �	��O2Yc�A�-*

loss�r�<���       �	I�P2Yc�A�-*

loss�^;ƚ��       �	^�Q2Yc�A�-*

lossy^=՝�f       �		TR2Yc�A�-*

loss�j&:��&       �	��R2Yc�A�-*

loss��<ؙ��       �	�S2Yc�A�-*

loss��<�l�6       �	��T2Yc�A�-*

loss�>�;O-.�       �	S=U2Yc�A�-*

lossN_�<��       �	7�U2Yc�A�-*

loss��y9�n��       �	|V2Yc�A�-*

loss�C=��o4       �	I0W2Yc�A�-*

loss_�n;�v�_       �	�UX2Yc�A�-*

loss�a�=����       �	1�X2Yc�A�-*

loss �=�9�       �	�Y2Yc�A�-*

loss��K<�8Yy       �	^JZ2Yc�A�-*

lossWg;[�b�       �	z�Z2Yc�A�-*

loss�[=[��C       �	�x[2Yc�A�-*

loss�݉=�ł       �	�\2Yc�A�-*

loss���<<�#       �	��\2Yc�A�-*

loss�Z=����       �	Mi]2Yc�A�-*

loss� =0I��       �	�]2Yc�A�.*

lossq�<�h��       �	P�^2Yc�A�.*

loss�X~<�"ǅ       �	y\_2Yc�A�.*

loss&�==���{       �	�_2Yc�A�.*

loss��J<WnG       �	!�`2Yc�A�.*

losss�=;�_�d       �	7Ra2Yc�A�.*

loss!��<R>�}       �	��a2Yc�A�.*

loss�B�:����       �	5�b2Yc�A�.*

loss��<�g0�       �	�Qc2Yc�A�.*

loss.y=�Ii�       �	�c2Yc�A�.*

loss�|<�׭B       �	΋d2Yc�A�.*

loss�@"=k��       �	�e2Yc�A�.*

loss<4j=.O�.       �	J�e2Yc�A�.*

loss{=�<��k�       �	ʊf2Yc�A�.*

lossvVc;����       �	�g2Yc�A�.*

loss�~<���       �	�g2Yc�A�.*

lossf�;�f`�       �	�h2Yc�A�.*

loss���<�Fs       �	:$i2Yc�A�.*

loss�U�<��<       �	��i2Yc�A�.*

loss�9;-$�a       �	�Tj2Yc�A�.*

loss�z<bsIl       �	=�j2Yc�A�.*

loss��=4�)�       �	Y�k2Yc�A�.*

loss�7=T5�p       �	� l2Yc�A�.*

loss,';:��       �	c�l2Yc�A�.*

lossv��<#H_j       �	Pqm2Yc�A�.*

loss���;���       �	�n2Yc�A�.*

loss%�D<*��/       �	Ƣn2Yc�A�.*

loss A�<�\�       �	�<o2Yc�A�.*

lossZ�<��`9       �	g�o2Yc�A�.*

loss1m\<R�,/       �	]lp2Yc�A�.*

loss}�<�/��       �	q2Yc�A�.*

loss��V<�x�       �	Ǡq2Yc�A�.*

lossVIs<�%*�       �	�:r2Yc�A�.*

loss��<���!       �	��r2Yc�A�.*

loss@�}<2[7�       �	�s2Yc�A�.*

loss:a�:$�<�       �	�.t2Yc�A�.*

loss��2;� �       �	*�t2Yc�A�.*

loss�%A<��A�       �	�Zu2Yc�A�.*

loss��;��[�       �	Q�u2Yc�A�.*

loss?<ID��       �	�v2Yc�A�.*

loss �,=�%�       �	�+w2Yc�A�.*

loss-&c=|w�       �	��w2Yc�A�.*

loss2];�ɷc       �	Otx2Yc�A�.*

loss��;�`�       �	y2Yc�A�.*

lossv}W<d�&,       �	��y2Yc�A�.*

loss�@;O�
       �	Jz2Yc�A�.*

lossX�Z;5�Ŗ       �	1�z2Yc�A�.*

lossE�u;z<�       �	ƈ{2Yc�A�.*

loss��D<�߲       �	�F|2Yc�A�.*

loss�3G;}n٢       �	*�|2Yc�A�.*

loss�N	<�6�       �	��}2Yc�A�.*

lossJ!�:L�K�       �	C�~2Yc�A�.*

loss��[<v�͎       �	P:2Yc�A�.*

lossJ��<��x*       �	b�2Yc�A�.*

loss�ͳ9���'       �	��2Yc�A�.*

loss�a�9_��       �	�h�2Yc�A�.*

loss�_�<
%M       �	c�2Yc�A�.*

loss�=�-�       �	÷�2Yc�A�.*

lossAz�;�
s$       �	yY�2Yc�A�.*

loss3008
��       �	�X�2Yc�A�.*

loss��H91�       �	���2Yc�A�.*

loss\�7<�?U!       �	���2Yc�A�.*

loss�
<�r~j       �	 T�2Yc�A�.*

losshR�=��y       �	T��2Yc�A�.*

loss��=�1ܓ       �	���2Yc�A�.*

lossZ�{=� H       �	v7�2Yc�A�.*

loss��<E�K       �	%�2Yc�A�.*

loss�:� �       �	4��2Yc�A�.*

loss��]=:��       �	�F�2Yc�A�.*

loss��=Ƚ��       �	I�2Yc�A�.*

loss� �;Z�t�       �	���2Yc�A�.*

loss��<�ူ       �	�9�2Yc�A�.*

loss���;�E�       �	�،2Yc�A�.*

loss	��<N;�        �	ꑍ2Yc�A�.*

lossz�<�ο+       �	A,�2Yc�A�.*

loss�?�;A�       �	�ю2Yc�A�.*

loss��F<��O�       �	v�2Yc�A�.*

loss��=���'       �	��2Yc�A�.*

loss �< (]       �	Ѱ�2Yc�A�.*

loss�%<~6��       �	�M�2Yc�A�.*

loss�]�<�!F�       �	x�2Yc�A�.*

loss�M;آRL       �	���2Yc�A�.*

loss�V�:.Y�       �	YL�2Yc�A�.*

loss{U`<�LY       �	^�2Yc�A�.*

loss�)U<Q���       �	t��2Yc�A�.*

loss3-@<�Q�       �	�=�2Yc�A�.*

loss���<����       �	U��2Yc�A�.*

loss�
x;����       �	���2Yc�A�.*

loss�0�<�r�       �	RG�2Yc�A�.*

loss])G:=h�       �	�ۗ2Yc�A�.*

loss��
=��s�       �	�s�2Yc�A�.*

loss,P<=B��       �	�2Yc�A�.*

lossv��<��>�       �	6��2Yc�A�.*

loss���<�v�       �	�B�2Yc�A�.*

loss̊;	��       �	Eך2Yc�A�.*

loss6�f;�E��       �	s�2Yc�A�.*

loss��(<���@       �	��2Yc�A�.*

loss���<=�lj       �	� �2Yc�A�.*

loss ��:�~!�       �	��2Yc�A�.*

lossr��<���       �	�.�2Yc�A�.*

lossc��<�k�!       �	Ğ2Yc�A�.*

lossq�K;JҜ[       �	�[�2Yc�A�.*

loss���<�g��       �	��2Yc�A�.*

loss��=5z       �	 ��2Yc�A�.*

loss���<~G�       �	s0�2Yc�A�.*

lossʾ�;nG�       �	��2Yc�A�.*

loss�2;�1O?       �	���2Yc�A�.*

losse�	<���2       �	v7�2Yc�A�.*

loss��1=����       �	�Σ2Yc�A�.*

loss�`;�1��       �	�q�2Yc�A�.*

loss/�<F�w       �	�	�2Yc�A�.*

loss�ϫ<��Э       �	�ߥ2Yc�A�.*

loss�p;�CԼ       �	�2Yc�A�.*

loss{��;�1�       �	[&�2Yc�A�.*

loss���=Y�v       �	a��2Yc�A�.*

loss�!V;�)C       �	W�2Yc�A�.*

loss�V�<)�M�       �	��2Yc�A�.*

loss��v;�       �	@��2Yc�A�.*

loss4q;cL�       �	\ �2Yc�A�.*

loss5#<�zs       �	>��2Yc�A�.*

lossA�!;��c       �	0J�2Yc�A�.*

loss���<��       �	v��2Yc�A�.*

lossw K<Â��       �	w��2Yc�A�.*

loss�̿:��_+       �	�"�2Yc�A�.*

lossH:�<��       �	M��2Yc�A�.*

loss��O=҆k       �	_b�2Yc�A�.*

loss�0�<sd+)       �	���2Yc�A�.*

loss�=�ce�       �	���2Yc�A�.*

loss�� <�?�a       �	�b�2Yc�A�/*

loss��$;��;�       �	� �2Yc�A�/*

loss�H	=RS3       �	���2Yc�A�/*

loss�?'<`�Lb       �	�:�2Yc�A�/*

loss{2�<�zn�       �	V��2Yc�A�/*

loss��;d�e�       �	�l�2Yc�A�/*

loss��;L{�       �	#�2Yc�A�/*

loss�o	<��       �	��2Yc�A�/*

loss-��<�~P,       �	�n�2Yc�A�/*

loss��d=`��|       �	�2Yc�A�/*

loss� =�w��       �	���2Yc�A�/*

loss�N�</�q       �	߇�2Yc�A�/*

loss���<���       �	(�2Yc�A�/*

loss�$<3��g       �	���2Yc�A�/*

loss��p<E��4       �	}\�2Yc�A�/*

lossa;=_C�       �	���2Yc�A�/*

lossj��:��a4       �	��2Yc�A�/*

loss{�j;ac�       �	 |�2Yc�A�/*

lossKQ�=�``*       �	:�2Yc�A�/*

loss�{�;�*�       �	���2Yc�A�/*

lossT�<[ ��       �	~�2Yc�A�/*

lossf��;Fa��       �	��2Yc�A�/*

lossq�<��ȍ       �	���2Yc�A�/*

loss�,;�ް�       �	�P�2Yc�A�/*

lossV�;J�{       �	���2Yc�A�/*

loss���;�%�       �	���2Yc�A�/*

loss��;�ɷ�       �	�;�2Yc�A�/*

lossx�`=����       �	���2Yc�A�/*

loss�(9;3��1       �	�o�2Yc�A�/*

loss��<���<       �	��2Yc�A�/*

loss�&;�n�       �	+��2Yc�A�/*

losssm�<��B       �	C7�2Yc�A�/*

loss��;iܴ�       �	0�2Yc�A�/*

lossݎ:T4�       �	>��2Yc�A�/*

loss�":�yt       �	o��2Yc�A�/*

lossVG�;B5$k       �	d�2Yc�A�/*

loss�)|=��?�       �	��2Yc�A�/*

loss(��<M�-       �	6Y�2Yc�A�/*

loss�O/<�h��       �	���2Yc�A�/*

loss�?0<(Ru       �	 ��2Yc�A�/*

lossX��:���       �	�"�2Yc�A�/*

loss���;����       �	���2Yc�A�/*

loss]�?8�f;�       �	P�2Yc�A�/*

loss���:휼       �	A��2Yc�A�/*

loss{��=��9{       �	���2Yc�A�/*

loss�b�<�XI       �	5b�2Yc�A�/*

loss�b�;'8��       �	*�2Yc�A�/*

loss�M,<�Nd�       �	��2Yc�A�/*

loss8)<��P�       �	�m�2Yc�A�/*

loss7<���e       �	��2Yc�A�/*

loss7�:=��V       �	���2Yc�A�/*

lossFҴ<U��       �	jO�2Yc�A�/*

loss��3;���       �	%��2Yc�A�/*

loss<z��       �	F��2Yc�A�/*

loss�ۈ={�[       �	�2�2Yc�A�/*

lossE��<��IA       �	-��2Yc�A�/*

loss[�<,U,�       �	�i�2Yc�A�/*

loss�D<�0�       �	>�2Yc�A�/*

loss%k<T�+�       �	{g�2Yc�A�/*

losso��;�`�       �	���2Yc�A�/*

lossڄ�;�YL�       �	�P�2Yc�A�/*

lossC;Jʐ       �	��2Yc�A�/*

lossi�<��Ǡ       �	�d�2Yc�A�/*

loss�6�;C �       �	HQ�2Yc�A�/*

loss��<�0��       �	>��2Yc�A�/*

loss�!�;���Z       �	���2Yc�A�/*

loss�F�<�(�       �	�7�2Yc�A�/*

loss�
M<@&h<       �	���2Yc�A�/*

loss�2�8����       �	ݶ�2Yc�A�/*

loss8�;[,�       �	eS�2Yc�A�/*

loss�a<&5��       �	��2Yc�A�/*

loss���=��]       �	���2Yc�A�/*

loss��$:uS,9       �	6�2Yc�A�/*

loss �P<�jE�       �	9��2Yc�A�/*

loss�fr;�X�       �	d�2Yc�A�/*

lossVl�<��0$       �	���2Yc�A�/*

lossF�^;��t       �	}��2Yc�A�/*

loss�QQ:�C�       �	�# 3Yc�A�/*

lossưO<�?U`       �	� 3Yc�A�/*

loss�^P=�WXv       �	��3Yc�A�/*

lossJ�:9fl�       �	�(3Yc�A�/*

loss�0J<{+�       �	��3Yc�A�/*

loss?�	<�_�z       �	�W3Yc�A�/*

loss��:=C�v�       �	�13Yc�A�/*

loss3�e;���\       �	��3Yc�A�/*

loss}:[0(�       �	\3Yc�A�/*

loss���:�]^       �	~3Yc�A�/*

loss�
�;�l5�       �	O�3Yc�A�/*

loss���;j�D�       �		63Yc�A�/*

lossdX�;yT��       �	%�3Yc�A�/*

loss�FU=�5��       �	Id3Yc�A�/*

loss�Ϙ<)���       �	�3Yc�A�/*

loss��:}z        �	Ú	3Yc�A�/*

loss`�<��"       �	�0
3Yc�A�/*

lossU�;�]1�       �	��
3Yc�A�/*

loss�r�;(BL�       �	b3Yc�A�/*

loss)�=<�t�1       �	8�3Yc�A�/*

loss�<u��       �	׉3Yc�A�/*

lossn��;���k       �	W!3Yc�A�/*

loss���;y,�       �	k�3Yc�A�/*

loss���<EIL�       �	IH3Yc�A�/*

lossX;ř79       �	"�3Yc�A�/*

lossF��<E◟       �	,�3Yc�A�/*

loss��;����       �	�3Yc�A�/*

loss�'P;u(��       �	`�3Yc�A�/*

lossD<�:,D�       �	�K3Yc�A�/*

loss�=);�       �	|�3Yc�A�/*

loss<Q�;��]H       �	W�3Yc�A�/*

loss7��:B��       �	�,3Yc�A�/*

loss�:���       �	��3Yc�A�/*

loss�-<���       �	~�3Yc�A�/*

loss�5<����       �	�)3Yc�A�/*

loss��;YS�       �	b�3Yc�A�/*

loss6C<V��       �	�a3Yc�A�/*

loss�=�;�{��       �	P3Yc�A�/*

loss\��;	���       �	S�3Yc�A�/*

loss�<@E�       �	j�3Yc�A�/*

loss`� <�604       �	�
3Yc�A�/*

loss�*=[�y�       �	�3Yc�A�/*

loss�+�=B�M       �	J3Yc�A�/*

lossP�
;��tV       �	��3Yc�A�/*

loss�Q1<[dK�       �	,}3Yc�A�/*

loss&�>;'}�N       �	�3Yc�A�/*

loss�[L:zn.�       �	��3Yc�A�/*

loss}�3;�Sy       �	O@3Yc�A�/*

loss`��;et�       �	`�3Yc�A�/*

lossf�;�D��       �	ƈ 3Yc�A�/*

loss���<R���       �	�%!3Yc�A�/*

loss��<9�u1       �	��!3Yc�A�0*

lossS�:<�nC�       �	�~"3Yc�A�0*

loss�/!<��       �	�#3Yc�A�0*

loss��&<"���       �	u�#3Yc�A�0*

loss���:H�Y       �	�B$3Yc�A�0*

lossX$d;LU�       �	��$3Yc�A�0*

loss0Z=n;�       �	Ln%3Yc�A�0*

loss6�8;Sʲ       �	�&3Yc�A�0*

loss�;�       �	�&3Yc�A�0*

lossd�a=��K�       �	}A'3Yc�A�0*

loss���<q���       �	��'3Yc�A�0*

lossũ�<^
�s       �	%v(3Yc�A�0*

loss�uW:��R       �	2)3Yc�A�0*

loss�pq:5�t)       �	A�)3Yc�A�0*

losssB�<�1       �	6t*3Yc�A�0*

lossp|�=3��.       �	+3Yc�A�0*

lossߑ{=c�8�       �	D�+3Yc�A�0*

loss��<L�       �	�p,3Yc�A�0*

lossx�';��jr       �	�	-3Yc�A�0*

loss�"�<
W��       �	ß-3Yc�A�0*

loss�5;�3�       �	H2.3Yc�A�0*

loss"T�<8�        �	�.3Yc�A�0*

loss�N;��Q�       �	�b/3Yc�A�0*

loss�g;�CZ�       �	W	03Yc�A�0*

loss�	=�Nai       �	��03Yc�A�0*

loss��:���       �	gH13Yc�A�0*

loss{�><��rX       �	z�13Yc�A�0*

loss.*=�� �       �	r�23Yc�A�0*

loss#g<��8�       �	633Yc�A�0*

lossף[;���       �	�33Yc�A�0*

loss���;*�ҵ       �	�43Yc�A�0*

loss?_:U&�       �	�&53Yc�A�0*

loss�%=w4�       �	�e63Yc�A�0*

loss�]=���	       �	K73Yc�A�0*

lossS�I=9�Y�       �	�183Yc�A�0*

loss�[�;���`       �	�83Yc�A�0*

loss
��<�a��       �	|93Yc�A�0*

loss�:�u�       �	0:3Yc�A�0*

loss-�z9�J6�       �	q�:3Yc�A�0*

loss��:���X       �	>�;3Yc�A�0*

lossN�:���y       �	�<3Yc�A�0*

losss :O��,       �	�=3Yc�A�0*

lossl�U:�,��       �	}�>3Yc�A�0*

loss�c�<��       �	�[?3Yc�A�0*

loss�|;�k@       �	]@3Yc�A�0*

loss�~�=�v�       �	��@3Yc�A�0*

lossI��<�`	       �	�*B3Yc�A�0*

loss-z�<h��       �	�^C3Yc�A�0*

loss��;���A       �	��D3Yc�A�0*

loss��h;���t       �	�E3Yc�A�0*

loss�`7;6A�G       �	�F3Yc�A�0*

loss�V:����       �	�4H3Yc�A�0*

loss�3=<�B�'       �	��H3Yc�A�0*

lossiQ�;�}��       �	��I3Yc�A�0*

loss�;{?Ρ       �	�J3Yc�A�0*

loss̊�<�~��       �	��K3Yc�A�0*

loss7S�<�;��       �	ƢL3Yc�A�0*

lossQ�<�?�       �	ogM3Yc�A�0*

loss��<Р?}       �	V�N3Yc�A�0*

loss!��<@Q!       �	�nO3Yc�A�0*

lossH~�; ��       �	[P3Yc�A�0*

loss1I=��>~       �	IQ3Yc�A�0*

loss�q;�9��       �	�RR3Yc�A�0*

loss� =r��       �	��S3Yc�A�0*

loss��;/夲       �	W`T3Yc�A�0*

loss�4=w S�       �	0U3Yc�A�0*

loss��&;�`=       �	,�U3Yc�A�0*

loss�*B<\��       �	�vV3Yc�A�0*

lossQib=�},l       �	�:W3Yc�A�0*

loss�;�/��       �	�:X3Yc�A�0*

lossz�V<\z�       �	�Y3Yc�A�0*

lossAwp=~�{�       �	PZ3Yc�A�0*

loss���<�ԇ�       �	�2[3Yc�A�0*

loss:�=>o�       �	��[3Yc�A�0*

loss�C�=���       �	!�\3Yc�A�0*

loss_a�<P?�       �	H�]3Yc�A�0*

loss�k�<Fİ�       �	u _3Yc�A�0*

loss3�;�Ji       �	�V`3Yc�A�0*

lossa�r;�4X@       �	�a3Yc�A�0*

loss3Q�:���       �	�b3Yc�A�0*

loss���;��-�       �	�Cc3Yc�A�0*

loss7o<�A �       �	��c3Yc�A�0*

loss�|=\�O�       �	6�d3Yc�A�0*

loss��;J�{T       �	'�e3Yc�A�0*

losst��;�t[�       �	g3Yc�A�0*

loss�� =t=m�       �	�g3Yc�A�0*

loss��8=��.�       �	�h3Yc�A�0*

loss/A=��п       �	C i3Yc�A�0*

loss�9�<�P�       �	��i3Yc�A�0*

losst�<���       �	��j3Yc�A�0*

loss�~;60kH       �	�Ck3Yc�A�0*

loss�0:<K۬s       �	��k3Yc�A�0*

losss<��d�       �	��l3Yc�A�0*

loss���;��aF       �	�Zm3Yc�A�0*

loss�5�;�>k�       �	�n3Yc�A�0*

lossM��<�Ģn       �	Q�n3Yc�A�0*

loss�d<�s       �	�Bo3Yc�A�0*

lossm�<R�\       �	�o3Yc�A�0*

lossw0�="��       �	�yp3Yc�A�0*

loss{�w<t�-�       �	nq3Yc�A�0*

lossVƚ<�r�       �	G�q3Yc�A�0*

loss�ָ<6�       �	�Sr3Yc�A�0*

loss���<��%
       �	w�r3Yc�A�0*

lossV��:�Ҿ       �	,�s3Yc�A�0*

loss���;Ž�/       �	ZEt3Yc�A�0*

loss�+�<F,h�       �	�Yu3Yc�A�0*

loss��;IR       �	c*v3Yc�A�0*

loss��<d���       �	�Pw3Yc�A�0*

loss�r;.��?       �	x3Yc�A�0*

loss��V<��F;       �	��x3Yc�A�0*

loss�!�8�6O       �	|,y3Yc�A�0*

loss43&<f���       �	��y3Yc�A�0*

loss�S2=�{yI       �	_zz3Yc�A�0*

loss�-<icY�       �	�{3Yc�A�0*

loss��!<��Yv       �	A�{3Yc�A�0*

loss��=\���       �	�N|3Yc�A�0*

lossf/�<W��h       �	��|3Yc�A�0*

loss%[=\�j�       �	�x}3Yc�A�0*

loss�+T<��B       �	=~3Yc�A�0*

losso��<\y#       �	��~3Yc�A�0*

loss�<���       �	73Yc�A�0*

loss�'�;DQ�       �	p�3Yc�A�0*

loss���:���       �	�i�3Yc�A�0*

loss=��<W�#�       �	��3Yc�A�0*

loss�<��oh       �	���3Yc�A�0*

loss� <��'�       �	�@�3Yc�A�0*

loss�N<G�#�       �	�Ԃ3Yc�A�0*

loss��><�Sb�       �	�j�3Yc�A�0*

loss!�;S�L:       �	�3Yc�A�1*

loss~� =��2       �	擄3Yc�A�1*

loss7ZQ=V��x       �	�&�3Yc�A�1*

loss��=����       �	���3Yc�A�1*

loss&b�;�%�y       �	�T�3Yc�A�1*

loss��;<P9�       �	J�3Yc�A�1*

loss܄�;�v��       �	E��3Yc�A�1*

loss� �<a�"       �	`�3Yc�A�1*

loss���<�u��       �	���3Yc�A�1*

loss4��<��       �	i��3Yc�A�1*

lossˠ<��J�       �	O"�3Yc�A�1*

loss��N<O�$�       �	8��3Yc�A�1*

lossO<$,       �	�b�3Yc�A�1*

loss2ۥ<�{��       �	e��3Yc�A�1*

loss< ?<ʟ�n       �	�ߍ3Yc�A�1*

loss.'=6"ry       �	 �3Yc�A�1*

loss�r�;qaW       �	�x�3Yc�A�1*

loss�J8=�J�       �	A�3Yc�A�1*

losss�;���z       �	ƥ�3Yc�A�1*

loss�Q�;r'[�       �	*;�3Yc�A�1*

loss��[<7ϯ1       �	�ϑ3Yc�A�1*

lossF�:�j       �	e�3Yc�A�1*

loss�!�:�˗�       �	q�3Yc�A�1*

lossDX< 6       �	���3Yc�A�1*

lossʺE=Π       �	�,�3Yc�A�1*

loss���;���       �	;Ĕ3Yc�A�1*

loss�]�<�/       �	���3Yc�A�1*

loss �=;y3       �	� �3Yc�A�1*

loss��@<���g       �	9�3Yc�A�1*

loss<[�'       �	�"�3Yc�A�1*

loss���;�|đ       �	���3Yc�A�1*

lossZ?6;�Ik�       �	�f�3Yc�A�1*

loss�=n=�پ       �	;��3Yc�A�1*

losss�w<��       �	���3Yc�A�1*

loss͹=j�       �	�B�3Yc�A�1*

loss@�9<H�       �	��3Yc�A�1*

loss�y�;�w�       �	���3Yc�A�1*

loss�l�;o�k       �	�)�3Yc�A�1*

loss���;�iu�       �	�ŝ3Yc�A�1*

loss�M;ݘ#�       �	�b�3Yc�A�1*

loss���<��LR       �	���3Yc�A�1*

loss��g;d���       �	b��3Yc�A�1*

loss=P@;�        �	���3Yc�A�1*

loss��,<2zv       �	�N�3Yc�A�1*

loss�ǃ:.�I�       �	�R�3Yc�A�1*

loss�_:	y       �	��3Yc�A�1*

loss��<;�*       �	�3Yc�A�1*

loss��<�dK�       �	��3Yc�A�1*

loss�e�;\���       �	��3Yc�A�1*

loss́<k�       �	O�3Yc�A�1*

loss�ΰ<6�%�       �	��3Yc�A�1*

lossӛ<��޽       �	=��3Yc�A�1*

loss��9QBm\       �	�,�3Yc�A�1*

loss��=+�F�       �	�̧3Yc�A�1*

loss�m�<\A6       �	Tp�3Yc�A�1*

loss�X�;A��j       �	��3Yc�A�1*

lossGu�;v�\\       �	���3Yc�A�1*

loss�B<��S       �	@P�3Yc�A�1*

loss�=��]_       �	}�3Yc�A�1*

loss��P<P��       �	τ�3Yc�A�1*

loss���;bַ       �	`�3Yc�A�1*

loss�;�ѷT       �	ú�3Yc�A�1*

loss3DJ;6�7�       �	3T�3Yc�A�1*

loss3<�ȬU       �	��3Yc�A�1*

loss�~<;��       �	��3Yc�A�1*

loss:�:� ��       �	V*�3Yc�A�1*

loss��A<�^       �	�3Yc�A�1*

lossz,N=^��       �	嘰3Yc�A�1*

loss::}:p{t�       �	u:�3Yc�A�1*

lossX��9N$�       �	�ұ3Yc�A�1*

loss�T
<����       �	�n�3Yc�A�1*

loss���:R�k       �	�e�3Yc�A�1*

lossʄ�:r�r)       �	��3Yc�A�1*

loss���=7u��       �	���3Yc�A�1*

lossa~�=�
a�       �	;�3Yc�A�1*

loss�7=�2Di       �	H�3Yc�A�1*

loss��:�֛       �	���3Yc�A�1*

loss�=>ᯐ       �	�&�3Yc�A�1*

lossl��;X�kY       �	�ķ3Yc�A�1*

loss�'=���0       �	@��3Yc�A�1*

loss��<�pm�       �	��3Yc�A�1*

loss),-=X 8k       �	�+�3Yc�A�1*

loss�;h�$�       �	aź3Yc�A�1*

loss�=F	��       �	$^�3Yc�A�1*

loss��7<4(5v       �	���3Yc�A�1*

loss�E�<�#��       �	���3Yc�A�1*

losss��;C-       �	be�3Yc�A�1*

lossM��:^܃       �	v��3Yc�A�1*

loss�R;�TG       �	���3Yc�A�1*

losstK�<{(��       �	�/�3Yc�A�1*

loss�9;z��f       �	�ÿ3Yc�A�1*

loss]�(=cW��       �	XY�3Yc�A�1*

lossO��=��-�       �	��3Yc�A�1*

loss3�#<�x!�       �	z��3Yc�A�1*

loss�g=�o��       �	Uk�3Yc�A�1*

loss�W<=މ�       �	�
�3Yc�A�1*

loss��7;�J�K       �	��3Yc�A�1*

lossΑ	;f��N       �	�>�3Yc�A�1*

loss�;�v{�       �	��3Yc�A�1*

lossɧ�<CŒ       �	���3Yc�A�1*

loss���<�9�       �	d�3Yc�A�1*

lossl�:8�7       �	���3Yc�A�1*

loss�=u)ZF       �	ݙ�3Yc�A�1*

loss��<�<�       �	�0�3Yc�A�1*

loss,|<>�W6       �	���3Yc�A�1*

loss���<�&�        �	�g�3Yc�A�1*

loss֌<�n9t       �	� �3Yc�A�1*

loss��;�Zܓ       �	F��3Yc�A�1*

loss�$=w��       �	�d�3Yc�A�1*

loss �<F�       �	<�3Yc�A�1*

loss��[;��       �	���3Yc�A�1*

loss.w�=��ݨ       �	�k�3Yc�A�1*

loss,� :��Q       �	��3Yc�A�1*

loss��t: �6�       �	W��3Yc�A�1*

loss�V�<P�t       �	�)�3Yc�A�1*

loss��:�4"z       �	j��3Yc�A�1*

lossL3�<�٭       �	���3Yc�A�1*

lossX��:��c�       �	9%�3Yc�A�1*

loss|� ;�hV�       �	_��3Yc�A�1*

loss���;���       �	:��3Yc�A�1*

lossz�'<5dT&       �	�&�3Yc�A�1*

loss6Ѷ:&�M�       �	��3Yc�A�1*

loss��I;s>�       �	}��3Yc�A�1*

lossf��;%ކ�       �	�J�3Yc�A�1*

lossFP�<s       �	���3Yc�A�1*

loss�4:�E��       �	�x�3Yc�A�1*

loss���;=
1�       �	��3Yc�A�1*

losssq�;IDD       �	j/�3Yc�A�1*

loss�<�H       �	n��3Yc�A�2*

loss�8$<#(ݱ       �	}]�3Yc�A�2*

loss��2<��VL       �	���3Yc�A�2*

lossZ�_=U;��       �	���3Yc�A�2*

loss/<�T3�       �	D�3Yc�A�2*

loss�R<�F23       �	T��3Yc�A�2*

lossՑ<��A	       �	��3Yc�A�2*

loss��=����       �	�5�3Yc�A�2*

lossx|=��       �	:��3Yc�A�2*

loss���;;+��       �	Ze�3Yc�A�2*

loss68p=w��?       �	�a�3Yc�A�2*

loss���<ULsc       �	<��3Yc�A�2*

loss�/�=;,��       �	���3Yc�A�2*

loss��<#@�=       �	�#�3Yc�A�2*

loss
��<���       �	���3Yc�A�2*

loss�3=�C��       �	�l�3Yc�A�2*

loss�e<Ư.�       �	� �3Yc�A�2*

loss��:�O�       �	���3Yc�A�2*

loss�^�<�KF       �	�0�3Yc�A�2*

lossf�;�       �	�3Yc�A�2*

loss�=���       �	^��3Yc�A�2*

losszTh<EWʻ       �	�/�3Yc�A�2*

loss�N"<{�K�       �	���3Yc�A�2*

loss�RM:�]h�       �	�W�3Yc�A�2*

loss�T�<�8	8       �	(��3Yc�A�2*

loss��=+�B       �	H��3Yc�A�2*

loss܏)<��-       �	��3Yc�A�2*

loss�´<)       �	���3Yc�A�2*

losst��<X�u�       �	�L�3Yc�A�2*

loss%��<ԖV�       �	G��3Yc�A�2*

lossD[9<��R7       �	��3Yc�A�2*

lossf<+<ا�!       �	�#�3Yc�A�2*

loss��
<���       �	
��3Yc�A�2*

lossZ�T;`�       �	�Q�3Yc�A�2*

loss��:�_u^       �	0��3Yc�A�2*

loss�#<����       �	���3Yc�A�2*

losst;�<ux		       �	f-�3Yc�A�2*

lossp�;�oɸ       �	���3Yc�A�2*

loss$��:�{�T       �	�f�3Yc�A�2*

loss3.�<�C�g       �	��3Yc�A�2*

loss��C;�t��       �	���3Yc�A�2*

loss�ȯ<4=ީ       �	M��3Yc�A�2*

loss2CR<_<:�       �	��3Yc�A�2*

lossAY�<�1       �	��3Yc�A�2*

loss#b=t���       �	�M�3Yc�A�2*

loss
&S:r��r       �	��3Yc�A�2*

loss]��<�\Q�       �	%��3Yc�A�2*

loss΃<=}���       �	YR�3Yc�A�2*

loss�g�;��h�       �	l��3Yc�A�2*

loss
G.;~�       �	��3Yc�A�2*

lossL�<_���       �	~R�3Yc�A�2*

lossT��;��(�       �	I��3Yc�A�2*

loss�Z�<��\�       �	(��3Yc�A�2*

loss��:P�3       �	V�3Yc�A�2*

loss��5<��J�       �	G�3Yc�A�2*

loss���=r�J�       �	B��3Yc�A�2*

loss�ɫ<]�       �	cz�3Yc�A�2*

loss�@<��I�       �	�N�3Yc�A�2*

loss��<t�       �	 �3Yc�A�2*

loss�B�<�w�&       �	3 4Yc�A�2*

loss)�y<͠�       �	q� 4Yc�A�2*

lossq�`=��"�       �	��4Yc�A�2*

loss�xG:�=�       �	<34Yc�A�2*

loss4��;����       �	A4Yc�A�2*

loss3�<bXcG       �	��4Yc�A�2*

loss8v<3���       �	�4Yc�A�2*

lossi`w;|�^u       �	Dj4Yc�A�2*

lossS�;�5Q�       �	�44Yc�A�2*

loss���;��       �	�84Yc�A�2*

lossA�+<4F       �	5�4Yc�A�2*

loss�,�;Qќ       �	�4Yc�A�2*

loss��%=>�J       �	0�	4Yc�A�2*

loss�z�;=5�       �	�3
4Yc�A�2*

loss�U�<􉝙       �	��
4Yc�A�2*

loss]/=ř;�       �	�k4Yc�A�2*

loss�%�<����       �	. 4Yc�A�2*

loss�;<���       �		�4Yc�A�2*

loss���<�u�       �	I4Yc�A�2*

lossR�=U�V       �	u�4Yc�A�2*

loss���;簛�       �	d4Yc�A�2*

lossw��<�3�s       �	�4Yc�A�2*

loss｡<��4�       �	�@4Yc�A�2*

loss���;kֶ�       �	,�4Yc�A�2*

loss��:����       �	�o4Yc�A�2*

loss�:�<H]Ǭ       �	�4Yc�A�2*

loss�X"=�=��       �	��4Yc�A�2*

lossf�6<ُ�       �	6Y4Yc�A�2*

loss��:���       �	��4Yc�A�2*

lossf�<���X       �	Ĕ4Yc�A�2*

loss��7<F�+       �	�(4Yc�A�2*

loss��j<rIP4       �	��4Yc�A�2*

loss���;N"�D       �	V4Yc�A�2*

loss׈�;��0       �	c�4Yc�A�2*

loss!t�9y0�       �	e�4Yc�A�2*

loss��<N�&�       �	�'4Yc�A�2*

loss��3;���       �	r�4Yc�A�2*

loss�A�;���       �	�e4Yc�A�2*

loss0�;�'8       �	��4Yc�A�2*

lossnH�<�`��       �	Н4Yc�A�2*

loss��;Q�7�       �	H34Yc�A�2*

loss6��<�{�3       �	r�4Yc�A�2*

loss���=�.        �	)\4Yc�A�2*

loss��:��9       �	� 4Yc�A�2*

loss��;+�3?       �	#�4Yc�A�2*

loss	6�;`���       �	�N4Yc�A�2*

loss��<�C�       �	��4Yc�A�2*

loss�1<�xk       �	��4Yc�A�2*

lossh2H=��       �	� 4Yc�A�2*

loss��;+���       �	l� 4Yc�A�2*

lossS�x<;u8       �	�O!4Yc�A�2*

loss�w�<}�(�       �	.�!4Yc�A�2*

loss{z�;��       �	�}"4Yc�A�2*

loss��#<'g       �	~#4Yc�A�2*

loss�[�:��U       �	��#4Yc�A�2*

loss��
=<�r�       �	�]$4Yc�A�2*

loss/�;�}x       �	�.%4Yc�A�2*

loss ΃<�s;�       �	��%4Yc�A�2*

loss���;��N�       �	�f&4Yc�A�2*

lossV��<�� u       �	��&4Yc�A�2*

lossA�e;���       �	��'4Yc�A�2*

loss��;�j��       �	E/(4Yc�A�2*

loss{�h<�RW@       �	[�(4Yc�A�2*

loss�7[;w�       �	be)4Yc�A�2*

lossM(:��N4       �	��)4Yc�A�2*

lossM<:/��       �	��*4Yc�A�2*

loss���<�)K�       �	�,+4Yc�A�2*

loss״�;*'<       �	8�+4Yc�A�2*

loss�?�:���-       �	��,4Yc�A�2*

loss��;6�|�       �	7-4Yc�A�3*

lossLC�=�D       �	��-4Yc�A�3*

loss�q9S=�       �	6r.4Yc�A�3*

loss�0;e3�       �	
/4Yc�A�3*

loss
�<�^�K       �	E�/4Yc�A�3*

loss،C:E��       �	�<04Yc�A�3*

loss��:�[�Z       �	�04Yc�A�3*

loss���:Ӂ�       �	�l14Yc�A�3*

loss�5�;r�'"       �	�24Yc�A�3*

lossi]7<3m�       �	Z�24Yc�A�3*

loss\Q<���       �	�?34Yc�A�3*

lossJx�9�JM       �	k�34Yc�A�3*

loss3N�;��"
       �	�s44Yc�A�3*

loss�@�:����       �	�54Yc�A�3*

loss�'�7��:       �	�64Yc�A�3*

lossQ��9�?r�       �	�64Yc�A�3*

loss<'�;Noa       �	��74Yc�A�3*

loss���;ۖ��       �	�I84Yc�A�3*

loss��;cQ+       �	�N94Yc�A�3*

loss!M@<����       �	�,:4Yc�A�3*

loss��;
�]�       �	R�:4Yc�A�3*

loss��M=�1]       �	u�;4Yc�A�3*

lossv��:�*]
       �	��<4Yc�A�3*

loss.8�=����       �	�M=4Yc�A�3*

lossZW�;͒[�       �	?�=4Yc�A�3*

lossah2=G�       �	Ow>4Yc�A�3*

loss��;�xf7       �	n?4Yc�A�3*

loss��"=��1E       �	�?4Yc�A�3*

loss!��;Z(�`       �	�B@4Yc�A�3*

lossY�;�WE�       �	��@4Yc�A�3*

loss�+!<~��       �	2tA4Yc�A�3*

loss��0<(#S       �	�
B4Yc�A�3*

loss}��;�x       �	,�B4Yc�A�3*

loss�'<�0M�       �	�}C4Yc�A�3*

lossϰJ=W@3       �	�D4Yc�A�3*

lossҫ:��"�       �	��D4Yc�A�3*

loss�G;sf�W       �	�SE4Yc�A�3*

loss?�<���       �	�E4Yc�A�3*

loss�=rԦ       �	Y�F4Yc�A�3*

lossΓ�:�ng�       �	�"G4Yc�A�3*

loss?�
<���        �	+�G4Yc�A�3*

loss���:�2       �	)[H4Yc�A�3*

lossq��:%��1       �	x�H4Yc�A�3*

lossF=;�o�B       �	��I4Yc�A�3*

loss�>�<ö       �	�0J4Yc�A�3*

loss��;��fT       �	��J4Yc�A�3*

loss	�:�P�>       �	�rK4Yc�A�3*

loss$;}ٿ�       �	�L4Yc�A�3*

loss�|�;O��       �	��L4Yc�A�3*

loss=?!=��E�       �	�QM4Yc�A�3*

loss��>
�lK       �	%�M4Yc�A�3*

loss$�$=n[:e       �	
�N4Yc�A�3*

lossȒ\<�u?       �	 O4Yc�A�3*

loss�;*��s       �	h�O4Yc�A�3*

lossj�^:�e�1       �	�QP4Yc�A�3*

losse�:]A"�       �	��P4Yc�A�3*

loss��;��1x       �	��Q4Yc�A�3*

loss�٘;8�O       �	"R4Yc�A�3*

loss`+�;�7?       �	��R4Yc�A�3*

loss�
=�Y�       �	�NS4Yc�A�3*

loss�<�<�I�       �	 �S4Yc�A�3*

loss���;���       �	�T4Yc�A�3*

loss�k�:,`Ӈ       �	G=U4Yc�A�3*

loss�W=���       �	^�U4Yc�A�3*

loss�A<q�O       �	nV4Yc�A�3*

loss��;�"o*       �	�W4Yc�A�3*

loss���<���       �	I�W4Yc�A�3*

loss�k6=&�M�       �	l?X4Yc�A�3*

loss��]=˲-�       �	R�X4Yc�A�3*

loss{�c;����       �	��Y4Yc�A�3*

loss֥%=+�       �	ADZ4Yc�A�3*

loss�ѯ<�@�       �	��Z4Yc�A�3*

lossi�t;_@�        �	��[4Yc�A�3*

lossa�&<<H�k       �	ٗt4Yc�A�3*

loss]̧=�m��       �	U.u4Yc�A�3*

lossy=���)       �	�1v4Yc�A�3*

loss��;8��x       �	&�v4Yc�A�3*

lossȄ<���\       �	��w4Yc�A�3*

lossNo�<Z��5       �	w�x4Yc�A�3*

lossZ�y=����       �	�y4Yc�A�3*

lossm��=��3�       �	G�y4Yc�A�3*

loss���;b���       �	kFz4Yc�A�3*

lossnӝ;�\V�       �	w�z4Yc�A�3*

loss��O;my       �	�{4Yc�A�3*

loss/s^;��h]       �	|4Yc�A�3*

loss8	�:���       �	�|4Yc�A�3*

lossa��<6[^6       �	iT}4Yc�A�3*

lossx��:��q�       �	��}4Yc�A�3*

loss��=X�!       �	"�~4Yc�A�3*

lossTj(9k�|       �	�$4Yc�A�3*

loss��;��       �	v�4Yc�A�3*

loss�"�<�f       �	�`�4Yc�A�3*

loss�<A�|�       �	���4Yc�A�3*

loss4�<V��       �	��4Yc�A�3*

loss��X=.�.       �	k+�4Yc�A�3*

loss
�e;��K�       �	u�4Yc�A�3*

loss�(>�;�N       �	}�4Yc�A�3*

lossN�:���       �	��4Yc�A�3*

loss傛<v��       �	7��4Yc�A�3*

loss�c=;�L�       �	2:�4Yc�A�3*

loss��;!��3       �	�ͅ4Yc�A�3*

loss�7x<I���       �	_�4Yc�A�3*

loss�G<�/[�       �	��4Yc�A�3*

lossVW<6
��       �	*��4Yc�A�3*

lossrW�=�،�       �	��4Yc�A�3*

loss�\�<���       �	y��4Yc�A�3*

loss�7#=���       �	sI�4Yc�A�3*

lossy�;���       �	O�4Yc�A�3*

loss-d�<�[�       �	��4Yc�A�3*

loss��:�f>       �	�1�4Yc�A�3*

loss��=��9�       �	�ȋ4Yc�A�3*

loss�Gb=���       �	a�4Yc�A�3*

losss��<��       �	��4Yc�A�3*

lossچ�;	��       �	���4Yc�A�3*

loss榾;(�{       �	r7�4Yc�A�3*

lossqg"<ٿ�       �	6̎4Yc�A�3*

loss<֖��       �	b�4Yc�A�3*

lossTV~;���\       �	���4Yc�A�3*

loss���<���>       �	���4Yc�A�3*

loss���;ݧ��       �	�2�4Yc�A�3*

loss߷<�aQ�       �	eő4Yc�A�3*

loss�/;����       �	�`�4Yc�A�3*

loss�4,;�Wڕ       �	9�4Yc�A�3*

loss��Z;�"�       �	̷�4Yc�A�3*

loss��9<NXЌ       �	3��4Yc�A�3*

loss��$< ���       �	8g�4Yc�A�3*

lossA'0=4�^       �	;��4Yc�A�3*

lossW�O<*��/       �	u��4Yc�A�3*

loss��;��       �	�=�4Yc�A�4*

loss)�:}=0       �	�4Yc�A�4*

losst�<�o�       �	��4Yc�A�4*

loss�3<p�)�       �	�+�4Yc�A�4*

lossL`4=�E�       �	�Ι4Yc�A�4*

loss�=;�       �	<k�4Yc�A�4*

loss7A�;T%u�       �	��4Yc�A�4*

loss�29;@�
�       �	�ś4Yc�A�4*

loss�t_<���       �	Rc�4Yc�A�4*

loss�;��b       �	���4Yc�A�4*

loss�+�:h�h�       �	:��4Yc�A�4*

loss�ٯ:��Q       �	�E�4Yc�A�4*

loss�^�;9Ť�       �	�4Yc�A�4*

lossQ�j=��       �	D��4Yc�A�4*

loss��<9+�       �	�7�4Yc�A�4*

losss<��       �	�Ҡ4Yc�A�4*

loss	�e<�Z;n       �	Yk�4Yc�A�4*

loss�[�<S��       �	��4Yc�A�4*

lossM=;���D       �	OϢ4Yc�A�4*

loss��{<�i�
       �	0g�4Yc�A�4*

lossE�9;-lj       �	��4Yc�A�4*

loss�3:���Q       �	���4Yc�A�4*

lossn}$;���       �	<O�4Yc�A�4*

loss�V�;�Փ       �	��4Yc�A�4*

loss�̇=K�W�       �	��4Yc�A�4*

lossԾ;��_u       �	+�4Yc�A�4*

loss�E<���       �	#ڧ4Yc�A�4*

loss�/�<�k�       �	;r�4Yc�A�4*

loss=�+;�h�       �	O�4Yc�A�4*

loss�I<:z��s       �	���4Yc�A�4*

loss�NC;O_B\       �	T6�4Yc�A�4*

loss$>�:4X�       �	hΪ4Yc�A�4*

loss�w�<:�y�       �	f��4Yc�A�4*

loss,�K<1��       �	x%�4Yc�A�4*

lossk	<
�2�       �	���4Yc�A�4*

lossCG�<To+l       �	mT�4Yc�A�4*

loss��)<H}!�       �	��4Yc�A�4*

lossFT�;��U�       �	͒�4Yc�A�4*

loss�"<P�=�       �	�*�4Yc�A�4*

lossmao<0@       �	<��4Yc�A�4*

losso�<�A��       �	�V�4Yc�A�4*

loss���<,��{       �	���4Yc�A�4*

loss�<�VС       �	���4Yc�A�4*

loss�"�<DG	B       �	��4Yc�A�4*

loss���<�{�v       �	!��4Yc�A�4*

loss�Br<oK�       �	�F�4Yc�A�4*

lossV�;h�3       �	�ݳ4Yc�A�4*

loss���;BK�}       �	H��4Yc�A�4*

losssA�;a�*       �	�4Yc�A�4*

lossCh<S�u       �	ǵ4Yc�A�4*

loss?�=��e        �	�l�4Yc�A�4*

lossIU;�\�#       �	b�4Yc�A�4*

loss�& :
ي�       �	D��4Yc�A�4*

lossDa<��{�       �	���4Yc�A�4*

lossݞ�;��<�       �	�)�4Yc�A�4*

loss��;Qb#�       �	Hù4Yc�A�4*

loss�0�=㫒       �	}Z�4Yc�A�4*

lossR=�u�       �	��4Yc�A�4*

lossTe�<�]i�       �	���4Yc�A�4*

loss�F�;�l��       �	�<�4Yc�A�4*

loss��<:H�b       �	�Ӽ4Yc�A�4*

loss�H�;�*I       �	�{�4Yc�A�4*

lossV�K=����       �	<�4Yc�A�4*

lossuۘ<KAI�       �	��4Yc�A�4*

lossJ�=èj       �	B^�4Yc�A�4*

lossx-<��
1       �	�4Yc�A�4*

loss��6;+[ck       �	���4Yc�A�4*

lossE��;Iǰd       �	.;�4Yc�A�4*

loss��K:
-�       �	���4Yc�A�4*

loss��:p���       �		o�4Yc�A�4*

lossV��:=OG       �	$
�4Yc�A�4*

loss���;��&        �	���4Yc�A�4*

lossN�<�eC�       �	M�4Yc�A�4*

loss�d�<]��       �	��4Yc�A�4*

loss���;��t�       �	Mh�4Yc�A�4*

loss�<d���       �	���4Yc�A�4*

lossĒ�<�!�o       �	���4Yc�A�4*

loss���<2�ٺ       �	#0�4Yc�A�4*

loss�:��>�       �	���4Yc�A�4*

loss!��;       �	�\�4Yc�A�4*

lossO"Z<�w�       �	���4Yc�A�4*

loss���<IAbb       �	_��4Yc�A�4*

loss�:�Q       �	�j�4Yc�A�4*

loss�<��-i       �	�4Yc�A�4*

lossz��:�߻�       �	ͱ�4Yc�A�4*

loss� W;9M�       �	�I�4Yc�A�4*

loss�#n<%�(�       �	�"�4Yc�A�4*

lossa&=�w=       �	�V�4Yc�A�4*

loss,��;���       �	f��4Yc�A�4*

loss���<9�q       �	���4Yc�A�4*

loss�<�Wͷ       �	j��4Yc�A�4*

loss<8�;���       �	�V�4Yc�A�4*

loss!5;ܵe       �	���4Yc�A�4*

loss���:xZ��       �	w��4Yc�A�4*

loss�b<�_�       �	�=�4Yc�A�4*

loss��O;�ܿd       �	���4Yc�A�4*

loss�Bf;�J��       �	�{�4Yc�A�4*

loss�$�<�
�       �	��4Yc�A�4*

loss��%;�m��       �	!��4Yc�A�4*

loss���<��g       �	��4Yc�A�4*

loss�5O90���       �	���4Yc�A�4*

loss� �;��b       �	c��4Yc�A�4*

lossnĊ=�n��       �		��4Yc�A�4*

lossD�;K��%       �	�'�4Yc�A�4*

lossW�<�j
�       �	���4Yc�A�4*

loss�\=���       �	X��4Yc�A�4*

lossA��:M`A       �	U��4Yc�A�4*

lossA��<�鴖       �	7U�4Yc�A�4*

loss��9)t�^       �	��4Yc�A�4*

loss4��<���       �	���4Yc�A�4*

loss�ͳ<��֬       �	z�4Yc�A�4*

lossr&�;����       �	a�4Yc�A�4*

loss#��;8���       �	���4Yc�A�4*

loss.Ր:��j�       �	�c�4Yc�A�4*

loss��:�F��       �	��4Yc�A�4*

lossM��<�i�       �	��4Yc�A�4*

loss �>;eӧ|       �	���4Yc�A�4*

lossOň;C�*       �	5_�4Yc�A�4*

lossQ��=����       �	���4Yc�A�4*

lossj�;rV�8       �	-��4Yc�A�4*

loss��:�!�       �	�2�4Yc�A�4*

lossM-�<J \�       �	F��4Yc�A�4*

loss/�;Φi�       �	�t�4Yc�A�4*

loss���;�?�k       �	��4Yc�A�4*

loss�j�;�b�(       �	)��4Yc�A�4*

loss.�;?�u       �	�)�4Yc�A�4*

loss���<"�       �	���4Yc�A�4*

loss���;�M�       �	Ȕ�4Yc�A�4*

loss�D<#       �	�0�4Yc�A�5*

loss�O?<~� �       �	���4Yc�A�5*

loss;�;�>�       �	��4Yc�A�5*

loss�<�P�K       �	�#�4Yc�A�5*

loss��^=m���       �	���4Yc�A�5*

loss&*y<���)       �	Tq�4Yc�A�5*

loss�<=Gn�       �	��4Yc�A�5*

loss���<��X       �	-��4Yc�A�5*

loss���;h:z       �	�L�4Yc�A�5*

loss{��;�W#T       �	Y��4Yc�A�5*

loss��<G��       �	�y�4Yc�A�5*

loss �>:�9�       �	f�4Yc�A�5*

loss{�M;T�8�       �	���4Yc�A�5*

loss�I=��
       �	�K�4Yc�A�5*

loss�:]<׹[n       �	���4Yc�A�5*

lossh��=9=�>       �	���4Yc�A�5*

loss�Et;�}\       �	�"�4Yc�A�5*

loss�@<=��}�       �	B_�4Yc�A�5*

loss���;B��       �	o-�4Yc�A�5*

loss��:�**9       �	N|�4Yc�A�5*

loss*5;��,�       �	�4Yc�A�5*

loss�5�;�dS       �	��4Yc�A�5*

loss�);m06�       �	>��4Yc�A�5*

loss�j|=ME�       �	3��4Yc�A�5*

lossxy(<���       �	2��4Yc�A�5*

loss�z;��ߪ       �	H��4Yc�A�5*

loss�mG=p�Q�       �	9`�4Yc�A�5*

loss5s=P�^        �	 5Yc�A�5*

loss��:��       �	5Yc�A�5*

loss��?=��w       �	��5Yc�A�5*

loss�l;錯       �	t�5Yc�A�5*

loss�t=a��X       �	/05Yc�A�5*

loss�%�=��       �	!�5Yc�A�5*

loss-�<=Yג       �	�{5Yc�A�5*

loss/�<��/       �	/5Yc�A�5*

lossT�=�w�       �	#h5Yc�A�5*

loss:*=O���       �	�=5Yc�A�5*

lossv��:r�E�       �	�A5Yc�A�5*

loss��/;�=�       �	��5Yc�A�5*

loss~(;ߊ'       �	�	5Yc�A�5*

loss�< Y.&       �	4�
5Yc�A�5*

loss�,1=��       �	�x5Yc�A�5*

loss��w;s!Rc       �	ё5Yc�A�5*

loss�<7nl�       �	I5Yc�A�5*

loss�O = ���       �	��5Yc�A�5*

lossS1?;�ͼ<       �	��5Yc�A�5*

loss��:�2p�       �	Z~5Yc�A�5*

lossi6�:Z�v       �	�A5Yc�A�5*

loss��.:;�]B       �	��5Yc�A�5*

loss�ur=��wo       �	5Yc�A�5*

loss/<��C�       �	�	5Yc�A�5*

lossQ3<o�       �	��5Yc�A�5*

loss`k0=��qC       �	�\5Yc�A�5*

lossQ�U<̗��       �	W	5Yc�A�5*

loss(M?=o�ɂ       �	�>5Yc�A�5*

loss��?;�n�)       �	�Q5Yc�A�5*

loss|�:�V�       �	�5Yc�A�5*

loss��w;��I�       �	4�5Yc�A�5*

lossOk�;��V       �	p|5Yc�A�5*

loss!��9���N       �	�5Yc�A�5*

lossDJY<�Q��       �	5�5Yc�A�5*

loss� =��`       �	�F5Yc�A�5*

loss��<�w�Z       �	'�5Yc�A�5*

lossΧ<~�K�       �	��5Yc�A�5*

loss:��<��@<       �	` 5Yc�A�5*

lossYO�<��g�       �	��5Yc�A�5*

loss?R|;�w�       �	��5Yc�A�5*

lossUW<kp&�       �	�Z5Yc�A�5*

lossH�^<1�d�       �	� 5Yc�A�5*

loss��<}Z�       �	�� 5Yc�A�5*

loss_�::�&!       �	h=!5Yc�A�5*

lossߞ�9��       �	s�!5Yc�A�5*

loss�s;�4{"       �	pw"5Yc�A�5*

lossR��<�/�       �	#5Yc�A�5*

loss�i`:Y�k       �	�#5Yc�A�5*

loss��<LE^       �	�P$5Yc�A�5*

loss�M=��#       �	��$5Yc�A�5*

loss
J5<�_�b       �	��%5Yc�A�5*

loss��=
5-       �	��&5Yc�A�5*

loss��=�p�       �	� '5Yc�A�5*

loss
8�<�#O�       �	��'5Yc�A�5*

lossA2�<��\       �	�N(5Yc�A�5*

loss=Ԋ:����       �	m�(5Yc�A�5*

loss�A�=�k��       �	�x)5Yc�A�5*

loss��<�c&�       �	*5Yc�A�5*

loss�`�<��A^       �	%�*5Yc�A�5*

lossJ�<����       �	DQ+5Yc�A�5*

loss�ĥ:��{t       �	��+5Yc�A�5*

lossH�;���       �	u�,5Yc�A�5*

lossr>�:�tI       �	J(-5Yc�A�5*

losscn�<��K�       �	'�-5Yc�A�5*

lossoH�=s_�       �	�P.5Yc�A�5*

lossń�;w�77       �	7�.5Yc�A�5*

loss�
0<��Ф       �	y�/5Yc�A�5*

lossx¸:73�       �	�B05Yc�A�5*

loss���<��w       �	��05Yc�A�5*

loss*і<���       �	��15Yc�A�5*

loss�R�:F��       �	�B25Yc�A�5*

lossf/�9�)�U       �	�635Yc�A�5*

loss��<���m       �	-�35Yc�A�5*

loss`��<%��       �	_b45Yc�A�5*

lossC�<�g�Y       �	~�45Yc�A�5*

loss2��:�=�       �	��55Yc�A�5*

loss�Л<�>v       �	�*65Yc�A�5*

loss�� :t�"&       �	u:75Yc�A�5*

lossN�0:�-�       �	[�75Yc�A�5*

lossմ;���       �	�v85Yc�A�5*

loss���;:�z       �	M95Yc�A�5*

lossǎ<���O       �	h�95Yc�A�5*

loss�>�9C �       �	�Q:5Yc�A�5*

loss��:Se��       �	�:5Yc�A�5*

loss_,�<���#       �	ʌ;5Yc�A�5*

lossY�=c��^       �	�%<5Yc�A�5*

loss <�1s       �	5�<5Yc�A�5*

lossfP!=pG�o       �	�r=5Yc�A�5*

lossZ}=E�>�       �	=>5Yc�A�5*

loss��<�%��       �	ձ>5Yc�A�5*

loss�P�=��}^       �	�L?5Yc�A�5*

loss�d�=��       �	 �?5Yc�A�5*

loss�I;C�&C       �	3�@5Yc�A�5*

loss��<��e\       �	�5A5Yc�A�5*

loss4��<�'�`       �	��A5Yc�A�5*

loss�]=Πa�       �	2rB5Yc�A�5*

lossZ�;���x       �	�C5Yc�A�5*

loss*<ݨ�2       �	U�C5Yc�A�5*

loss�R�:ӗٲ       �	!;D5Yc�A�5*

loss��=�[��       �	��D5Yc�A�5*

loss��<�|co       �	gE5Yc�A�5*

loss���:!���       �	�F5Yc�A�6*

loss�ʵ;[P�       �	O�F5Yc�A�6*

loss蒃;_�}<       �	AFG5Yc�A�6*

loss)rE;q��       �	%�G5Yc�A�6*

loss�=[��c       �	��H5Yc�A�6*

loss�D<u#       �	�2I5Yc�A�6*

loss�o�;�@�       �	)�I5Yc�A�6*

lossRm9=�"T�       �	�yJ5Yc�A�6*

loss<5X;��@�       �	ZK5Yc�A�6*

loss���<�1-�       �	�K5Yc�A�6*

loss�|�=l��       �	�FL5Yc�A�6*

loss�<�&.�       �	 M5Yc�A�6*

loss	Q�:hwZD       �	]N5Yc�A�6*

lossT�!;�B��       �	��N5Yc�A�6*

loss%0q<z�U~       �	0cO5Yc�A�6*

lossw,�<��Q       �	�P5Yc�A�6*

loss)�[<���       �	נP5Yc�A�6*

loss3BB<�'�T       �	�=Q5Yc�A�6*

loss�d=�o�       �	+�Q5Yc�A�6*

loss��:����       �	�YS5Yc�A�6*

loss���:��       �	��S5Yc�A�6*

loss_�;��d       �	��T5Yc�A�6*

loss�q?:���       �	�U5Yc�A�6*

loss`�=T�m%       �	0.V5Yc�A�6*

loss�72;�y=       �	�AW5Yc�A�6*

loss� <7|�4       �	A�W5Yc�A�6*

loss1a<FO�       �	*pX5Yc�A�6*

loss��<�?~       �	i�Y5Yc�A�6*

loss���:���       �	��Z5Yc�A�6*

loss�Q;�Y�       �	b-[5Yc�A�6*

loss��a<�0�       �	��[5Yc�A�6*

lossqE;���V       �	�\\5Yc�A�6*

lossCp:ݪ[       �	M�\5Yc�A�6*

loss|=��?�       �	�]5Yc�A�6*

loss��;���        �	�P^5Yc�A�6*

loss/M�;��I�       �	��^5Yc�A�6*

loss���;�OD       �	6�_5Yc�A�6*

loss@=T�s�       �	M2`5Yc�A�6*

loss��P=>�M�       �	1�`5Yc�A�6*

loss���;V�F       �	aka5Yc�A�6*

losss�$<RbB�       �	�b5Yc�A�6*

lossl�F:�4;       �	��b5Yc�A�6*

losss�<\�H       �	xBc5Yc�A�6*

lossXs�<j��       �	"�c5Yc�A�6*

loss<3�<X��       �	5{d5Yc�A�6*

loss���;�h
       �	*8e5Yc�A�6*

lossط�<Ϡ�       �	�lf5Yc�A�6*

lossQ�5:o_��       �	!g5Yc�A�6*

loss��<��OX       �	Ûg5Yc�A�6*

loss؋?=E��=       �	q:h5Yc�A�6*

loss��*<�S�J       �	��h5Yc�A�6*

loss���<�ۄ       �	3�i5Yc�A�6*

loss�r%=��       �	yj5Yc�A�6*

loss&��=�V�       �	��j5Yc�A�6*

loss
�<��V       �	�wk5Yc�A�6*

loss��x<$k�       �	�l5Yc�A�6*

loss�E=�i��       �	�l5Yc�A�6*

loss�v�9�jS       �	�dm5Yc�A�6*

loss-�<%�3       �	�
n5Yc�A�6*

loss�g=��E       �	��n5Yc�A�6*

loss�	�<��u�       �	o5Yc�A�6*

loss��<�{�       �	1#p5Yc�A�6*

loss���<󟢮       �	��p5Yc�A�6*

lossT��;�h_       �	nmq5Yc�A�6*

lossT3;�m�       �		r5Yc�A�6*

loss�;-"�       �	�r5Yc�A�6*

loss6�<�4�z       �	�Ss5Yc�A�6*

loss$Ĵ<�а6       �	��s5Yc�A�6*

loss��;{a       �	`�t5Yc�A�6*

loss�;8܂�       �	:Au5Yc�A�6*

loss-�o;d�Rs       �	j.v5Yc�A�6*

loss� >7�z^       �	��v5Yc�A�6*

loss;?�:D�G       �	�w5Yc�A�6*

loss%h<��Q       �	"6x5Yc�A�6*

loss��<�R.       �	�"y5Yc�A�6*

loss��x<xO�       �	�y5Yc�A�6*

loss�D�;��F�       �	A�z5Yc�A�6*

loss\�=:_92       �	|�{5Yc�A�6*

loss׭{;B<6�       �	��|5Yc�A�6*

lossT�E<r�z       �	�m}5Yc�A�6*

loss���;��1�       �	�V~5Yc�A�6*

lossd�+<DZ�*       �	�5Yc�A�6*

loss$ˁ<���r       �	_�5Yc�A�6*

loss\ҁ==��b       �	/0�5Yc�A�6*

loss�7�=fuA�       �	Ё5Yc�A�6*

loss)�; )�       �	Ѱ�5Yc�A�6*

loss��Q:�c��       �	�L�5Yc�A�6*

lossl�<D���       �	��5Yc�A�6*

loss�	;e�4W       �	���5Yc�A�6*

loss��:4�]�       �	�Ȇ5Yc�A�6*

loss?8�<(�T       �	�(�5Yc�A�6*

lossW�;/���       �	_҈5Yc�A�6*

lossn;cv5       �	T�5Yc�A�6*

lossŻ�<v3�       �	��5Yc�A�6*

loss�;����       �	'K�5Yc�A�6*

loss�<;�4>�       �	��5Yc�A�6*

lossyh<Z��       �	w��5Yc�A�6*

loss?�!=���<       �	K �5Yc�A�6*

losssd><�t.�       �	��5Yc�A�6*

lossI�;����       �	b��5Yc�A�6*

loss�7q=F�
�       �	_'�5Yc�A�6*

loss �!;P��       �	
�5Yc�A�6*

lossc`7<DJ��       �	���5Yc�A�6*

lossh��<e�f(       �	�D�5Yc�A�6*

loss~7<����       �	�ڑ5Yc�A�6*

loss��w<��9       �	�p�5Yc�A�6*

loss|\s;�z�       �	��5Yc�A�6*

loss��"<y'��       �	�ѓ5Yc�A�6*

loss��=�R�       �	h�5Yc�A�6*

loss(;����       �	��5Yc�A�6*

lossk��=��X       �	i��5Yc�A�6*

loss��o;�t       �	1@�5Yc�A�6*

loss�Т<���Q       �	�Ԗ5Yc�A�6*

loss�1=gE�       �	c|�5Yc�A�6*

loss�<ɫ�       �	��5Yc�A�6*

loss�:�<.���       �	q�5Yc�A�6*

losszV�;���       �	A��5Yc�A�6*

loss!c=l��B       �	�3�5Yc�A�6*

lossb,;]&�^       �	�ښ5Yc�A�6*

lossx=<����       �	ě5Yc�A�6*

loss��;<�]V       �	8i�5Yc�A�6*

lossO�N;�pD]       �	�5Yc�A�6*

loss@o*=8p:       �	���5Yc�A�6*

lossi�<�"�       �	W@�5Yc�A�6*

loss��;�.o�       �	��5Yc�A�6*

loss�O;���       �	~��5Yc�A�6*

loss�3<#�       �	g,�5Yc�A�6*

lossS��;���       �	0Ԡ5Yc�A�6*

loss$1f;g��I       �	rm�5Yc�A�7*

loss�~�<Dicl       �	W
�5Yc�A�7*

lossɒy<�ņ       �	��5Yc�A�7*

loss�<���       �	O=�5Yc�A�7*

loss}h<�K"       �	Mڣ5Yc�A�7*

loss�*:����       �	~s�5Yc�A�7*

loss/wH=��`       �	��5Yc�A�7*

lossl �<ԫ��       �	��5Yc�A�7*

loss|&�<�'�       �	�D�5Yc�A�7*

loss���:�&�I       �	D�5Yc�A�7*

loss}�3=+��       �	�.�5Yc�A�7*

loss��B=���       �	�Ǩ5Yc�A�7*

loss*�;�+��       �	d[�5Yc�A�7*

loss@n�;i�Y�       �	���5Yc�A�7*

lossv�;8u,u       �	s��5Yc�A�7*

loss�+<��Ω       �	�H�5Yc�A�7*

lossv��;����       �	��5Yc�A�7*

lossȬG<���       �	���5Yc�A�7*

loss�v<��       �	6�5Yc�A�7*

loss���;�8�       �	6��5Yc�A�7*

lossZ�`<�(�j       �	M0�5Yc�A�7*

loss��;&�kw       �	�ͯ5Yc�A�7*

loss�)V<�6��       �	�g�5Yc�A�7*

loss��:�aI       �	{�5Yc�A�7*

loss�t�=Cd�       �	֭�5Yc�A�7*

loss�&�<؛       �	�O�5Yc�A�7*

loss��<7��       �	T�5Yc�A�7*

loss(R�;����       �	�|�5Yc�A�7*

loss�"�<?#d;       �	v�5Yc�A�7*

lossd�;�}a�       �	���5Yc�A�7*

loss�P:����       �	{I�5Yc�A�7*

lossM��=h'��       �	��5Yc�A�7*

loss�Ie:�+�       �	s��5Yc�A�7*

loss�ż;e��       �	��5Yc�A�7*

loss(5�;���       �	�[�5Yc�A�7*

loss/��<�|�       �	k�5Yc�A�7*

loss��y:.dn       �	e��5Yc�A�7*

loss�?09
Y�       �	�D�5Yc�A�7*

loss��<lu1       �	N�5Yc�A�7*

loss���;W�
�       �	���5Yc�A�7*

loss�'�<{+Ae       �	�E�5Yc�A�7*

loss�c<婾�       �	wؼ5Yc�A�7*

loss�6X;|M_�       �	Tt�5Yc�A�7*

loss���<i�TG       �	��5Yc�A�7*

loss���<}z��       �	��5Yc�A�7*

loss�1�<�8��       �	�F�5Yc�A�7*

loss@%&<��v       �	�ۿ5Yc�A�7*

loss��w<$�@�       �	ty�5Yc�A�7*

loss���<\J       �	��5Yc�A�7*

loss���<�$[�       �	Զ�5Yc�A�7*

loss�N�<˩w�       �	N�5Yc�A�7*

loss�b�;L��       �	W$�5Yc�A�7*

loss��T:M�       �	`9�5Yc�A�7*

loss���9M
�&       �	���5Yc�A�7*

losso[T<�;��       �	rk�5Yc�A�7*

loss1��:ظ��       �	P�5Yc�A�7*

lossm/�<Bw�       �	�A�5Yc�A�7*

loss��;�w	8       �	"��5Yc�A�7*

loss��E<�N8�       �	�u�5Yc�A�7*

lossa8N<'���       �	��5Yc�A�7*

loss�6< :m       �	���5Yc�A�7*

loss�>r<���       �	�C�5Yc�A�7*

loss�W�<5��]       �	���5Yc�A�7*

loss���<>9O\       �	Xp�5Yc�A�7*

loss��8;,B       �	F�5Yc�A�7*

lossS�<
a�       �	Й�5Yc�A�7*

lossF�<����       �	Q3�5Yc�A�7*

lossJ=�;�#P       �	��5Yc�A�7*

loss��:����       �	6?�5Yc�A�7*

loss���=/Ԩ       �	���5Yc�A�7*

loss��(<	��       �	yy�5Yc�A�7*

loss���9�#�       �	I+�5Yc�A�7*

loss@f;�~�e       �	e��5Yc�A�7*

loss��3<�AT�       �	Ul�5Yc�A�7*

loss@�;�T��       �	��5Yc�A�7*

lossx3�;uf{       �	i��5Yc�A�7*

loss���;X��       �	VD�5Yc�A�7*

lossA��=;ҟ�       �	I��5Yc�A�7*

loss�"^:�ׯg       �	~t�5Yc�A�7*

loss�@�;!DC       �	�
�5Yc�A�7*

loss���<���7       �	��5Yc�A�7*

loss+:9#7I       �	�<�5Yc�A�7*

loss�V;q7\�       �	���5Yc�A�7*

loss��95�$�       �	R~�5Yc�A�7*

loss�p;��       �	�5Yc�A�7*

loss���;�<�\       �	��5Yc�A�7*

loss�^;$       �	�L�5Yc�A�7*

loss��q;A�.�       �	d��5Yc�A�7*

loss���:��+�       �	��5Yc�A�7*

loss�+�=@���       �	y!�5Yc�A�7*

loss1�;����       �	R��5Yc�A�7*

losst!�<�T�%       �	m�5Yc�A�7*

losst��<x!"r       �	��5Yc�A�7*

lossX��;�m	P       �	]��5Yc�A�7*

lossD`�:�]�       �	�9�5Yc�A�7*

losso�<�xB       �	[��5Yc�A�7*

loss�J;��       �	���5Yc�A�7*

loss�D:]���       �	�K�5Yc�A�7*

lossRT<�Y�       �	���5Yc�A�7*

loss!�:�m.       �	c~�5Yc�A�7*

lossb>�;�~       �	n�5Yc�A�7*

lossh(88��
y       �	Ϻ�5Yc�A�7*

loss�J:����       �	�T�5Yc�A�7*

loss*2�8�       �	���5Yc�A�7*

lossw�;p��       �	���5Yc�A�7*

loss���<��u       �	 *�5Yc�A�7*

loss_'<b���       �	��5Yc�A�7*

losst
;t�m       �	��5Yc�A�7*

loss�cj=H�       �	�5�5Yc�A�7*

lossP��=~1`f       �	���5Yc�A�7*

loss<��;��~       �	�l�5Yc�A�7*

loss֓�=W�a       �	��5Yc�A�7*

lossx��;g��       �	f��5Yc�A�7*

loss��<���"       �	{M�5Yc�A�7*

loss*�]<�1       �	���5Yc�A�7*

loss�5<XXN       �	��5Yc�A�7*

loss]��=Z�Y�       �	]5�5Yc�A�7*

loss��$=�B^�       �	��5Yc�A�7*

lossJ�M<�>��       �	u�5Yc�A�7*

loss�=#H�W       �	��5Yc�A�7*

loss.(�:^��       �	O��5Yc�A�7*

loss�N<�r]�       �	�_�5Yc�A�7*

loss��:<I       �	���5Yc�A�7*

loss���<��Ɵ       �	J��5Yc�A�7*

loss�Z<JQS�       �	80�5Yc�A�7*

loss��<��       �	h��5Yc�A�7*

loss���<D+��       �	�b�5Yc�A�7*

loss(c>I�7       �	 �5Yc�A�7*

lossp=i|o�       �	���5Yc�A�8*

loss���:#K�       �	��5Yc�A�8*

loss��:�H�       �	!�5Yc�A�8*

lossv��;�.�-       �	$��5Yc�A�8*

loss8�T;���i       �	HO�5Yc�A�8*

loss�Z:"W�       �	*��5Yc�A�8*

loss�;�1       �	�z�5Yc�A�8*

loss頯;�a��       �	��5Yc�A�8*

loss��z<0���       �	��5Yc�A�8*

loss6�;;��       �	�D�5Yc�A�8*

loss�<�:�       �	���5Yc�A�8*

loss�e�<@��       �	�}�5Yc�A�8*

lossV$;�       �	f�5Yc�A�8*

loss��e<�3�       �	��5Yc�A�8*

loss��<�כ       �	�G�5Yc�A�8*

loss��;�޺�       �	���5Yc�A�8*

loss��;��       �	m��5Yc�A�8*

loss4ˏ<���       �	)?�5Yc�A�8*

loss��;=��       �	��5Yc�A�8*

loss�.@;p��[       �	hv 6Yc�A�8*

loss��<*�C�       �	�6Yc�A�8*

loss�i�<�$�       �	/�6Yc�A�8*

loss=��9t��'       �	)@6Yc�A�8*

lossM;<�J��       �	��6Yc�A�8*

loss��?<���       �	�v6Yc�A�8*

loss�[<µT       �	�#6Yc�A�8*

lossC�c=
Vݘ       �	��6Yc�A�8*

lossN=��E       �	�`6Yc�A�8*

losskv�<�/�       �	q�6Yc�A�8*

loss�:;�A�H       �	V�6Yc�A�8*

loss��<�\��       �	�;6Yc�A�8*

lossw�;���       �	,�6Yc�A�8*

loss�;x7�       �	i6Yc�A�8*

loss1�Z;,�q]