       �K"	   8fc�Abrain.Event:2� �z1�     $�	�8fc�A"��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2Ҟ�
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
:@@*
seed2�}*
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
seed2��
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
seed2��{
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
seed2��M*
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
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2���
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
seed2��*
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
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
valueB
 *  �@*
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
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*#
_output_shapes
:���������*
Index0*
T0
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
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
_output_shapes
: *
T0
N
PlaceholderPlaceholder*
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
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
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
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
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

Tdim0*
T0*'
_output_shapes
:���������
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
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
valueB"      @   @   *
_output_shapes
:*
dtype0
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
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "�lk���     �S�	��8fc�AJ��
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
seed2Ҟ�*
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2�}*
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
seed2��
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
 *�3z<*
_output_shapes
: *
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2��{*
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
seed2��M
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
seed2���*
T0*
seed���)*
dtype0
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
seed2��*
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
new_axis_mask *
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
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
L
div_1/yConst*
valueB
 *  �@*
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
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
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
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
T0*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
T0
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
T0*
_output_shapes
: *
Tshape0
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
T0*
_output_shapes
: *
Tshape0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*
_output_shapes
: *H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
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
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
_output_shapes
: *
dtype0*
valueB 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
�
3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
_output_shapes
:*
out_type0*
T0
�
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
T0*
_output_shapes
:*
out_type0
�
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
T0*#
_output_shapes
:���������*
Tshape0
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
T0*
_output_shapes
: *
Tshape0
�
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*#
_output_shapes
:���������*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*
T0
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
�
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
T0*
_output_shapes
: *
Tshape0
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
T0*#
_output_shapes
:���������*
Tshape0
�
Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
�
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*#
_output_shapes
:���������*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*
T0
�
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
T0*
_output_shapes
:*
out_type0
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
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
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*'
_output_shapes
:���������*
T0*

Tdim0
�
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDimsBgradients/softmax_cross_entropy_loss/xentropy_grad/PreventGradient*0
_output_shapes
:������������������*
T0
|
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
_
gradients/div_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
o
gradients/div_1_grad/NegNegsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
T0*
_output_shapes
: *
Tshape0
m
%gradients/div_1_grad/tuple/group_depsNoOp^gradients/div_1_grad/Reshape^gradients/div_1_grad/Reshape_1
�
-gradients/div_1_grad/tuple/control_dependencyIdentitygradients/div_1_grad/Reshape&^gradients/div_1_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape
�
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
_output_shapes
: *1
_class'
%#loc:@gradients/div_1_grad/Reshape_1*
T0
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape
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
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
T0
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
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
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
gradients/Shape_1Shapegradients/Switch:1*
T0*
_output_shapes
:*
out_type0
Z
gradients/zeros/ConstConst*
_output_shapes
: *
dtype0*
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:����������*
Tshape0*
T0
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*
T0
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
:gradients/sequential_1/dropout_2/cond/dropout/div_grad/SumSum>gradients/sequential_1/dropout_2/cond/dropout/div_grad/RealDivLgradients/sequential_1/dropout_2/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
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
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
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
2gradients/sequential_1/dropout_2/cond/mul_grad/SumSum2gradients/sequential_1/dropout_2/cond/mul_grad/mulDgradients/sequential_1/dropout_2/cond/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
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
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:����������*
T0
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1**
_output_shapes
:����������: *
N*
T0
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
N*
T0*(
_output_shapes
:����������*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad
�
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*
T0*(
_output_shapes
:����������
�
7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
_output_shapes	
:�*
data_formatNHWC*
T0
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*
T0
�
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad*
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
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
T0
�
:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradSwitch5gradients/sequential_1/flatten_1/Reshape_grad/Reshape#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape
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
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
_output_shapes
:*
out_type0
\
gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
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
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_1/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*
T0*/
_output_shapes
:���������@*
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*
T0
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
T0*/
_output_shapes
:���������@*
Tshape0
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
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
T0*/
_output_shapes
:���������@*
Tshape0
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
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1*
T0
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
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������@*
T0
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*
N*
T0*1
_output_shapes
:���������@: 
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*/
_output_shapes
:���������@*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
T0
�
6gradients/sequential_1/activation_2/Relu_grad/ReluGradReluGradgradients/AddN_1sequential_1/activation_2/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
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
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   
�
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*&
_output_shapes
:@@*
paddingVALID*
T0*
use_cudnn_on_gpu(
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
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
T0
�
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         @   
�
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*&
_output_shapes
:@*
paddingVALID*
T0*
use_cudnn_on_gpu(
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
VariableV2*
shared_name *
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
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
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*"
_class
loc:@conv2d_1/kernel
�
beta2_power
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
zeros_1Const*
dtype0*&
_output_shapes
:@*%
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
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
T
zeros_2Const*
_output_shapes
:@*
dtype0*
valueB@*    
�
conv2d_1/bias/Adam
VariableV2*
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
dtype0*
	container 
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
zeros_3Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_1/bias/Adam_1
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_1/bias*
shared_name 
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
l
zeros_5Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam_1
VariableV2*
	container *
dtype0*"
_class
loc:@conv2d_2/kernel*
shared_name *&
_output_shapes
:@@*
shape:@@
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
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
T
zeros_6Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam
VariableV2*
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
�
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
VariableV2*
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
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
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
b
zeros_8Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
�
dense_1/kernel/Adam
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
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adamzeros_8*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
_output_shapes	
:�*
dtype0*
shape:�*
	container *
_class
loc:@dense_1/bias*
shared_name 
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
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
_
zeros_12Const*
_output_shapes
:	�
*
dtype0*
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
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adamzeros_12*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
zeros_14Const*
_output_shapes
:
*
dtype0*
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
zeros_15Const*
_output_shapes
:
*
dtype0*
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
dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
%Adam/update_conv2d_2/kernel/ApplyAdam	ApplyAdamconv2d_2/kernelconv2d_2/kernel/Adamconv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking( 
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
Adam/beta2Adam/epsilonEgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0*
use_locking( 
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0�f��       ��-	��B8fc�A*

loss�K@@)��       ��-	��C8fc�A*

loss��@��=�       ��-	�D8fc�A*

loss�L	@+ô�       ��-	E8fc�A*

loss6��?#C       ��-	MMF8fc�A*

lossv��?��1,       ��-	�)G8fc�A*

loss��?�#�       ��-	�G8fc�A*

loss[��?Q۫[       ��-	ͯH8fc�A*

loss�W�?6��       ��-	��I8fc�A	*

loss��?Y#�       ��-	�5J8fc�A
*

loss��?$Dl       ��-	�'K8fc�A*

loss��R?@]e       ��-	<1L8fc�A*

loss�:?Y[�a       ��-	r�L8fc�A*

lossd��?��#       ��-	��M8fc�A*

loss͹t?G,�a       ��-	e7N8fc�A*

loss@Vd?�*H�       ��-	VO8fc�A*

loss�f?097       ��-	�0P8fc�A*

loss��P?x8)       ��-	�Q8fc�A*

loss�&?2�3�       ��-	��Q8fc�A*

loss}[]?�'       ��-	��R8fc�A*

loss��S?���       ��-	��S8fc�A*

loss�gC?.t�v       ��-	�T8fc�A*

loss�B?ޏ�       ��-	��U8fc�A*

loss�U^?h���       ��-	rQV8fc�A*

loss�8?lk�       ��-	�3W8fc�A*

loss6-*?�^�       ��-	�W8fc�A*

loss��$?�qՋ       ��-	�X8fc�A*

loss��[?)�_       ��-	Y8fc�A*

loss��M?���       ��-	=dZ8fc�A*

loss/�??���       ��-	�I[8fc�A*

loss��6?C�       ��-	�\8fc�A*

lossF?���       ��-	��\8fc�A *

loss�J?[���       ��-	BA]8fc�A!*

loss��:?x���       ��-	^�]8fc�A"*

loss\?����       ��-	�u^8fc�A#*

loss�ID?wp�)       ��-	�_8fc�A$*

loss��>w��x       ��-	X�_8fc�A%*

loss�,?�Y��       ��-	�R`8fc�A&*

lossFe\?�d�8       ��-	��`8fc�A'*

loss�@>?n��       ��-	�a8fc�A(*

lossW�?q�%�       ��-	�b8fc�A)*

loss-�>���       ��-	��b8fc�A**

loss\H�>j]8s       ��-	=�c8fc�A+*

lossr3?��v�       ��-	��d8fc�A,*

lossm�?hI�I       ��-	�We8fc�A-*

loss�K?�W�'       ��-	��e8fc�A.*

lossJ��>)?�9       ��-	��f8fc�A/*

losshί>���       ��-	&g8fc�A0*

loss]��>`��       ��-	׾g8fc�A1*

loss���>l=j       ��-	�bh8fc�A2*

loss��>��       ��-	�ri8fc�A3*

loss�>v܉x       ��-	�j8fc�A4*

loss`�>�?6w       ��-	M�j8fc�A5*

loss�m&?��7�       ��-	p^k8fc�A6*

lossq��>��a%       ��-	��k8fc�A7*

lossu��>m9�a       ��-	��l8fc�A8*

loss�|>g$       ��-	�km8fc�A9*

lossh,d>3�Q       ��-	�n8fc�A:*

loss�`?l�y       ��-	��n8fc�A;*

losst��>���M       ��-	HPo8fc�A<*

lossO��>�'�2       ��-	��o8fc�A=*

loss�؁>�5�)       ��-	��p8fc�A>*

loss�
n>Fʄ@       ��-	�7q8fc�A?*

loss��>���       ��-	��q8fc�A@*

loss���>�J��       ��-	x}r8fc�AA*

loss,�O>� wZ       ��-	Us8fc�AB*

loss��?���       ��-	�s8fc�AC*

lossj�>�
       ��-	�Wt8fc�AD*

loss�57?E��s       ��-	F�t8fc�AE*

loss��>���       ��-	e�u8fc�AF*

loss'؜>��d       ��-	�v8fc�AG*

loss2
�>��>       ��-	�v8fc�AH*

losss��>���n       ��-	Kw8fc�AI*

loss���>�BN�       ��-	��w8fc�AJ*

lossi5?�*i       ��-	��x8fc�AK*

lossr��>3iɀ       ��-	my8fc�AL*

loss��?��e�       ��-	��y8fc�AM*

loss7s?hً;       ��-	gz8fc�AN*

loss���>��ô       ��-	�{8fc�AO*

loss�>�g��       ��-	��{8fc�AP*

loss���>5Vlx       ��-	'P|8fc�AQ*

loss\�
?����       ��-	t�|8fc�AR*

lossƟ�>��k       ��-	!�}8fc�AS*

loss�w>�%�!       ��-	�J~8fc�AT*

losslo�>C�;       ��-	��~8fc�AU*

loss�\�>���*       ��-	y�8fc�AV*

lossl�>^%�       ��-	_�8fc�AW*

lossC��>�Dx�       ��-	�$�8fc�AX*

loss���>w�Y       ��-	��8fc�AY*

lossiR>��       ��-	G�8fc�AZ*

loss��>H�s�       ��-	L��8fc�A[*

loss�>s^       ��-	���8fc�A\*

loss�Hv>�b�       ��-	�n�8fc�A]*

loss��>����       ��-	�~�8fc�A^*

loss;�>��m�       ��-	^,�8fc�A_*

loss�Ow>�+�H       ��-	�׉8fc�A`*

loss��>6�r=       ��-	��8fc�Aa*

loss�&�>%�??       ��-	�$�8fc�Ab*

lossvP?���r       ��-	ދ8fc�Ac*

loss���>�9C       ��-	߉�8fc�Ad*

lossf�a>��       ��-	�3�8fc�Ae*

loss�j2>�8�       ��-	Eڍ8fc�Af*

loss�7�>j,�l       ��-	���8fc�Ag*

loss�Za>9��       ��-	k-�8fc�Ah*

loss�̀>'�9�       ��-	�͏8fc�Ai*

lossפ�>���]       ��-	Yj�8fc�Aj*

loss�V>�S       ��-	d�8fc�Ak*

loss8�k>�G       ��-	���8fc�Al*

loss�̾>�n�       ��-	!V�8fc�Am*

loss��>)�y       ��-	���8fc�An*

loss�?7�K       ��-	��8fc�Ao*

loss�[�>���       ��-	HP�8fc�Ap*

loss�+>�`       ��-	k�8fc�Aq*

loss;�5>���f       ��-	�y�8fc�Ar*

loss��*>�:       ��-	#.�8fc�As*

lossD>˿��       ��-	��8fc�At*

loss�?>vݗ$       ��-	�ʘ8fc�Au*

loss-5Z>u�vD       ��-	x��8fc�Av*

loss؊�>�hzi       ��-	t��8fc�Aw*

lossx�l>����       ��-	Y�8fc�Ax*

loss��M>�2G�       ��-	��8fc�Ay*

lossٺ�>��T       ��-	���8fc�Az*

loss�f�>�H�       ��-	P�8fc�A{*

loss��m>�5C�       ��-	P��8fc�A|*

loss���=ߔ�D       ��-	���8fc�A}*

loss(f>t�b       ��-	�ӟ8fc�A~*

loss ��>��       ��-	p�8fc�A*

loss�8N>���       �	V�8fc�A�*

loss37�>L�-�       �	���8fc�A�*

loss$b>�K       �	�=�8fc�A�*

loss�8�>`WL*       �	?�8fc�A�*

loss�Ri>��?�       �	 }�8fc�A�*

lossSр>�Lz       �	�p�8fc�A�*

loss���=���       �	�'�8fc�A�*

loss��f>�x��       �	�ʥ8fc�A�*

loss,R�>��2�       �	�s�8fc�A�*

loss�Yg>�0�       �	���8fc�A�*

loss�#�=�Ȳ,       �	cG�8fc�A�*

lossX>MD��       �	C�8fc�A�*

lossH��=v�Xr       �	���8fc�A�*

lossi�=��9[       �	
-�8fc�A�*

loss\ߑ>�"YL       �	 Ī8fc�A�*

loss3��>��k�       �	�h�8fc�A�*

loss��:>0o&       �	��8fc�A�*

loss��D>O_Mo       �	���8fc�A�*

loss��>l�b�       �	�V�8fc�A�*

lossf�w>o@�       �	t�8fc�A�*

loss��=l�ʴ       �	�8fc�A�*

loss��;>YuK�       �	�&�8fc�A�*

loss��=>%N�       �	�ȯ8fc�A�*

loss�6>���x       �	ca�8fc�A�*

loss�)�>����       �	��8fc�A�*

loss��>6?p�       �	Н�8fc�A�*

loss��U>��b�       �	��8fc�A�*

loss#}:>��5f       �	���8fc�A�*

loss�~=6̅�       �	LP�8fc�A�*

lossG0>o/��       �	��8fc�A�*

lossK~>�M{       �	腵8fc�A�*

loss��>�i,       �	��8fc�A�*

loss�*>�\9=       �	ǻ�8fc�A�*

lossE�>j�(       �	iS�8fc�A�*

loss�n�=E0)�       �	��8fc�A�*

loss�QT>,�"       �	���8fc�A�*

loss�|�=�mL�       �	&�8fc�A�*

loss�]>����       �	?ȹ8fc�A�*

lossX�8>�*]l       �	��8fc�A�*

loss�$>@�9       �	���8fc�A�*

lossI��>��2�       �		�8fc�A�*

losss>d%D�       �	���8fc�A�*

loss}��=�0ז       �	�V�8fc�A�*

lossQT!>S�1       �	��8fc�A�*

lossFWj=`��       �	��8fc�A�*

loss�(S>����       �	Zb�8fc�A�*

lossր�>�'��       �	���8fc�A�*

loss���=����       �	h��8fc�A�*

loss)qN>O��       �	�0�8fc�A�*

loss
�>����       �	Ug�8fc�A�*

loss��>TH       �	���8fc�A�*

lossT�S>�t�       �	�m�8fc�A�*

lossm��=p       �	�_�8fc�A�*

loss���=�Ń�       �	���8fc�A�*

lossv�y>�`��       �	���8fc�A�*

loss�}K>(&b#       �	���8fc�A�*

loss;u�=��S       �	�&�8fc�A�*

lossZ,J>���m       �	���8fc�A�*

loss]W*>��O       �	+n�8fc�A�*

loss��=���       �	^�8fc�A�*

loss��3>���R       �	:��8fc�A�*

loss: 3>�\F�       �	iS�8fc�A�*

loss�q>n<��       �	���8fc�A�*

loss
�c>�_'       �	���8fc�A�*

lossCZ7>F�'�       �	RD�8fc�A�*

loss->Q.K�       �	��8fc�A�*

loss�p1>%���       �	�|�8fc�A�*

loss��=����       �	��8fc�A�*

loss�M>O�Od       �	���8fc�A�*

losslc>C��       �	���8fc�A�*

loss��h>���       �	O$�8fc�A�*

loss�o>/��       �	���8fc�A�*

loss/�->_��|       �	5]�8fc�A�*

loss��M>0���       �	Q��8fc�A�*

loss%��=� �       �	]��8fc�A�*

lossc%�=d��       �	U2�8fc�A�*

loss��7>P��       �	"��8fc�A�*

lossH�C>���       �	���8fc�A�*

lossE�=C�u�       �	Z��8fc�A�*

loss�%>�ع�       �	�7�8fc�A�*

loss�'G>���       �	���8fc�A�*

loss#=>eC�/       �	?t�8fc�A�*

loss�ԇ>�!��       �	�
�8fc�A�*

loss$�.>!D��       �	H��8fc�A�*

loss���=��@�       �	QO�8fc�A�*

loss�;>L,��       �	���8fc�A�*

loss�>m�ޖ       �	;��8fc�A�*

loss�R>9��I       �	�/�8fc�A�*

loss��>v�)�       �	���8fc�A�*

loss��>K$D       �	Hj�8fc�A�*

losst@�>��*       �	��8fc�A�*

lossdy�>HR�K       �	؛�8fc�A�*

loss�g�>�uG       �	�9�8fc�A�*

loss3m�=�@W�       �	���8fc�A�*

loss;��=b��T       �	�l�8fc�A�*

loss7��>Í.�       �	v�8fc�A�*

loss�?T>��>       �	ū�8fc�A�*

loss���=! ��       �	�J�8fc�A�*

lossz��=��A�       �	"��8fc�A�*

loss8n>85��       �	Sv�8fc�A�*

loss��>8�!�       �	�
�8fc�A�*

loss#�E>�$t�       �	|��8fc�A�*

lossR�L>W|�W       �	�8�8fc�A�*

loss#�%>�",       �	}��8fc�A�*

loss�<L=�x�       �	���8fc�A�*

loss!\�>Tz�       �	"4�8fc�A�*

lossfuH>GH�       �	>��8fc�A�*

lossa�U>��I       �	�k�8fc�A�*

loss[/>#Tj�       �	��8fc�A�*

loss=��=
�x       �	1��8fc�A�*

loss�t�>��       �	�5�8fc�A�*

loss��\>k�TW       �	���8fc�A�*

loss��<>�z("       �	8g�8fc�A�*

loss2>9�#�       �	���8fc�A�*

loss@}Q>�t7       �	u��8fc�A�*

loss�v^>�R9�       �	1�8fc�A�*

loss>� �       �	J��8fc�A�*

loss�'N>m�g�       �	g�8fc�A�*

lossc�z>�Q�       �	��8fc�A�*

loss�-�=��|�       �	���8fc�A�*

lossv\>^g       �	�>�8fc�A�*

loss�=>ZÒ       �	��8fc�A�*

loss�>D�l�       �	5�8fc�A�*

lossWTH>��|D       �	��8fc�A�*

loss�p>k+q�       �	��8fc�A�*

loss�
>I.w^       �	�Z�8fc�A�*

loss��D=RFp       �	���8fc�A�*

lossH�^=F���       �	H��8fc�A�*

loss�0�=��)       �	�$�8fc�A�*

lossۢ8>��       �	~��8fc�A�*

loss��G>�P       �	/M�8fc�A�*

loss�=�� t       �	���8fc�A�*

loss>��!T       �	k��8fc�A�*

loss�l�=��X       �	�8�8fc�A�*

loss-�>�v!       �	���8fc�A�*

lossӦ�>#��       �	ɓ�8fc�A�*

loss�X>�T�*       �	a6 9fc�A�*

loss`�>#�0f       �	�� 9fc�A�*

lossR��=��m       �	�z9fc�A�*

loss���=V)��       �	�9fc�A�*

loss���=a[�       �	��9fc�A�*

lossT��>�J��       �	��9fc�A�*

loss�G>L��z       �	9�9fc�A�*

loss��A>L�       �	|�9fc�A�*

loss7�8>���K       �	q9fc�A�*

loss�2�=�ya       �	X�9fc�A�*

loss1��=Y�J2       �	��9fc�A�*

lossmt#>�l�\       �	��	9fc�A�*

lossXQ�>���       �	9fc�A�*

loss}�>�ɽ,       �	��9fc�A�*

lossŏ=�/ބ       �	�c9fc�A�*

lossq�=�,��       �	�9fc�A�*

loss�7;>�z�\       �	��9fc�A�*

loss�ɯ=T��1       �	\:9fc�A�*

loss�>2�       �	��9fc�A�*

loss�}>�㺬       �	)�9fc�A�*

loss^8>e�BE       �	�-9fc�A�*

lossq1�=���       �	i�9fc�A�*

loss�y>���?       �	�`9fc�A�*

loss��>5I�       �	� 9fc�A�*

loss=�O=���       �	�9fc�A�*

loss��
>i�K�       �	�C9fc�A�*

loss�c'=;�V�       �	�9fc�A�*

loss�X=�j?f       �	I�9fc�A�*

loss�>��b       �	�9fc�A�*

loss��=��
�       �	
�9fc�A�*

lossK�=�S��       �	�U9fc�A�*

loss���=�       �	��9fc�A�*

loss�1>+t-�       �	��9fc�A�*

loss�b>wI��       �	�%9fc�A�*

lossh��>':ˣ       �	b�9fc�A�*

loss�_>qtr       �	�#9fc�A�*

loss���>a1N�       �	~�9fc�A�*

loss��>��e       �	�h9fc�A�*

loss�L�=P�9"       �	�9fc�A�*

loss��=x�Yg       �	��9fc�A�*

loss	1	>E.�`       �	0I9fc�A�*

loss���=��r       �	�9fc�A�*

loss��o>}��f       �	��9fc�A�*

loss�=�<)���       �	�9fc�A�*

lossh�$>�s[�       �	�9fc�A�*

loss���=ϳ�v       �	] 9fc�A�*

loss_2�=�!�%       �	� 9fc�A�*

loss�q�= �"       �	2�!9fc�A�*

lossɦ�=���       �	��"9fc�A�*

loss���=�i�       �	[$#9fc�A�*

loss��=8㾯       �	��#9fc�A�*

lossv*�=��@       �	Mj$9fc�A�*

lossp9>'�:       �	�%9fc�A�*

loss�$�=����       �	��%9fc�A�*

lossn�W=so][       �	Q3&9fc�A�*

loss��Z>���       �	z�&9fc�A�*

loss�{�>�P,�       �	�_'9fc�A�*

loss���=���       �	�(9fc�A�*

loss3K�>Ԟ�       �	��(9fc�A�*

loss��$>O���       �	�:)9fc�A�*

loss��{=ȡ��       �	U�)9fc�A�*

lossSq�>�       �	�|*9fc�A�*

loss\�f>xu�       �	C+9fc�A�*

loss��R>�Ъ0       �	N�+9fc�A�*

loss��=��T�       �	N,9fc�A�*

loss�{>Fǃ`       �	B�,9fc�A�*

lossUU>w���       �	΍-9fc�A�*

lossJ̵=�ie�       �	�K.9fc�A�*

loss��=�+�       �	>[/9fc�A�*

loss���=�{��       �	��/9fc�A�*

lossFq�=x       �	]�09fc�A�*

loss2Z>�[E�       �	$19fc�A�*

losse9�=^H�}       �	Ժ19fc�A�*

loss��i>�{Π       �	�V29fc�A�*

lossL>>�G�(       �	V�29fc�A�*

loss3P�=�=H       �	�39fc�A�*

loss��==�;�Q       �	+49fc�A�*

loss�=�A��       �	v�49fc�A�*

loss�P=��v'       �	>^59fc�A�*

loss%��=�<~A       �	�59fc�A�*

lossV$�=��U       �	p�69fc�A�*

losso��>��=y       �	�=79fc�A�*

loss49>���       �	?�79fc�A�*

loss�(�=^�:H       �	989fc�A�*

loss�</>�x�       �	��99fc�A�*

lossq�">��x�       �	NE:9fc�A�*

lossT�k>t��0       �	=(;9fc�A�*

loss4>hj�r       �	��;9fc�A�*

loss@�=H��       �	#f<9fc�A�*

loss�>>�#       �	�
=9fc�A�*

loss7�>F>Z       �	��=9fc�A�*

loss���=]��       �	fK>9fc�A�*

loss8=/=�Y       �	�>9fc�A�*

loss��|=�
dX       �	Ό?9fc�A�*

loss�5�=�]��       �	�'@9fc�A�*

loss\�*>|��       �	�2A9fc�A�*

loss�?�=	��       �	7�A9fc�A�*

loss���=I�2        �	SvB9fc�A�*

loss\3�=���       �	x�C9fc�A�*

loss�C�=�^�       �	�D9fc�A�*

loss���>~���       �	؟E9fc�A�*

loss�#>c��l       �	�GF9fc�A�*

loss���=?�G       �	K�F9fc�A�*

lossjU>����       �	��G9fc�A�*

loss�w>>�y       �	vH9fc�A�*

lossO=L= lnQ       �	�H9fc�A�*

lossȅ=^���       �	?SI9fc�A�*

loss�6>1�       �	6�I9fc�A�*

lossz�">$j�       �	�J9fc�A�*

loss�><�ڗ       �	6K9fc�A�*

loss�a�=�NO�       �	��K9fc�A�*

loss���="<�A       �	�PL9fc�A�*

loss.k>^�?       �	5�L9fc�A�*

loss
F>>�P^�       �	ՖM9fc�A�*

loss4j�=���o       �	Q3N9fc�A�*

loss���=�7��       �	��N9fc�A�*

lossZ�H>����       �	�}O9fc�A�*

loss(�I>�a>�       �	��P9fc�A�*

loss��h>J$       �	c|Q9fc�A�*

loss"[>���       �	� R9fc�A�*

losse۾=9ͼ       �	��R9fc�A�*

loss_�Q=��d�       �	weS9fc�A�*

loss�-�=���y       �	�T9fc�A�*

loss�>ٵ�       �	�T9fc�A�*

loss���=r�z       �	�U9fc�A�*

loss�d2>)�       �	kDW9fc�A�*

lossDNA>��DD       �	2�W9fc�A�*

loss�J>�6 Y       �	��X9fc�A�*

loss� �=rs,�       �	�"Y9fc�A�*

loss�s>����       �	��Y9fc�A�*

loss��'>@�-�       �	��Z9fc�A�*

losso >�2        �	�k[9fc�A�*

loss�z>'���       �	t\9fc�A�*

loss��?=w�N       �	��\9fc�A�*

loss�p�=#
�y       �	��]9fc�A�*

loss���=���9       �	>^^9fc�A�*

loss�d>*rhY       �	 _9fc�A�*

loss��=�.�       �	<�_9fc�A�*

lossf�$>��j       �	J`9fc�A�*

loss�H>4F�c       �	��`9fc�A�*

lossQ��=��(�       �	��a9fc�A�*

lossx��=�\�e       �	,Gb9fc�A�*

loss��>9Wo       �	��b9fc�A�*

loss�->0���       �	"�c9fc�A�*

losss<>���       �	e4d9fc�A�*

loss6O�=V���       �	��d9fc�A�*

lossȨ+>�
�0       �	��e9fc�A�*

loss�>���       �	&4f9fc�A�*

loss��>j5!       �	;�f9fc�A�*

loss��=4�V�       �	o�g9fc�A�*

loss4ن>�Evk       �	-h9fc�A�*

loss[�?>�^��       �	+�h9fc�A�*

loss=�2>���       �	�i9fc�A�*

lossx�0=,娪       �	�#j9fc�A�*

loss¨=�l�       �	�j9fc�A�*

lossWKa>{HsP       �	�xk9fc�A�*

loss�"K>R~=       �	F(l9fc�A�*

loss6l�=U�v�       �	��l9fc�A�*

lossD�>	Q�       �	~m9fc�A�*

loss1M�=Ӕ�       �	�1n9fc�A�*

loss��>���[       �	��n9fc�A�*

lossM�=,�~�       �	�{o9fc�A�*

losswM>��O�       �	~p9fc�A�*

loss�)>>>�)�       �	�p9fc�A�*

lossq�>j���       �	-[q9fc�A�*

lossT*�=g��       �	�q9fc�A�*

lossѫ�=k�_�       �	ޓr9fc�A�*

loss	�>�U.h       �	�2s9fc�A�*

lossʉ�=�;<	       �	y�s9fc�A�*

loss�W=����       �	�ft9fc�A�*

loss�K�=�J�       �	��t9fc�A�*

loss�$G>���$       �	|�u9fc�A�*

loss�^�=�2�B       �	�?v9fc�A�*

lossv��=���b       �	��v9fc�A�*

loss͐�=�+�#       �	rw9fc�A�*

loss(��=�8f       �	Jx9fc�A�*

loss���=��'       �	¢x9fc�A�*

loss�0m>-ڽ       �	*7y9fc�A�*

losslfN>LJh7       �	d�y9fc�A�*

lossj��=iߑ+       �	Vez9fc�A�*

lossE�P>%�ȩ       �	��z9fc�A�*

loss|jw=�}       �	��{9fc�A�*

loss#)>�YoC       �	\<|9fc�A�*

loss4ϛ>F)M       �	�|9fc�A�*

loss�mR>b+[       �	�h}9fc�A�*

loss��=��       �	��}9fc�A�*

loss�Y#>��        �	��~9fc�A�*

loss�>�"0�       �	�.9fc�A�*

loss	�>���       �	p�9fc�A�*

loss��=2%#�       �	g�9fc�A�*

loss��R>]���       �	���9fc�A�*

loss�H�>:"{       �	u��9fc�A�*

loss-�^>Y̟�       �	�5�9fc�A�*

loss0~>t��k       �	ۂ9fc�A�*

loss���=��.�       �	*t�9fc�A�*

loss��=����       �	�	�9fc�A�*

loss-�={�]       �	Ǡ�9fc�A�*

loss&��<m���       �	�A�9fc�A�*

loss��=&K�       �	���9fc�A�*

loss3(o=����       �	i��9fc�A�*

loss��y>��D�       �	a�9fc�A�*

loss�m�=��i        �	Y��9fc�A�*

loss� �=��O�       �	n��9fc�A�*

loss)��=�I7       �	~:�9fc�A�*

loss��>A���       �	8܉9fc�A�*

loss���=p�       �	Ct�9fc�A�*

loss@��=��p�       �	��9fc�A�*

loss��5>,�!�       �	���9fc�A�*

loss)^>�6f       �	 <�9fc�A�*

loss�A>���:       �	�׌9fc�A�*

lossr�'>�Bqw       �	�p�9fc�A�*

lossH�=��       �	��9fc�A�*

loss�L�<��+       �	���9fc�A�*

loss�|~=q�	e       �	�-�9fc�A�*

lossn9�=Յ�       �	�9fc�A�*

loss�x=���       �	�X�9fc�A�*

loss D>h7q�       �	��9fc�A�*

loss.��>����       �	ӈ�9fc�A�*

lossW>�X��       �	W'�9fc�A�*

loss;�8=R��[       �	��9fc�A�*

lossQ�>�	�       �	9d�9fc�A�*

loss[Ͽ=�NNl       �	/��9fc�A�*

loss��=
>S       �	���9fc�A�*

loss3��=�k۾       �	.�9fc�A�*

loss�>O`x       �	I֕9fc�A�*

loss��=�ߟ�       �	vp�9fc�A�*

loss�R'>�Ǝ�       �	��9fc�A�*

loss.P�=�~q       �	��9fc�A�*

lossɎI>��0)       �	AF�9fc�A�*

lossaN_=QKd�       �	p�9fc�A�*

lossM�<(��A       �	���9fc�A�*

loss�nY=>�q�       �	1'�9fc�A�*

loss�Z=^�8�       �	�՚9fc�A�*

loss��S>�)˕       �	�p�9fc�A�*

lossI=,=���d       �	��9fc�A�*

loss��G=�S9       �	&��9fc�A�*

lossS$>6s�       �	}A�9fc�A�*

loss��=����       �	]�9fc�A�*

lossl�/=� �x       �	伞9fc�A�*

lossEmc=���       �	�W�9fc�A�*

loss�x�=���       �	���9fc�A�*

loss��	>���T       �	��9fc�A�*

lossVV�=E�@�       �	��9fc�A�*

loss,�E>%�!�       �	�C�9fc�A�*

loss�Ō=z(V       �	�9fc�A�*

loss�Ae>$�p       �	��9fc�A�*

loss��D>��A�       �	C�9fc�A�*

lossE�/=Y�6       �	0Ԥ9fc�A�*

loss��=i&��       �	6x�9fc�A�*

loss�S�=4�<`       �	��9fc�A�*

loss��>ax�       �	;Ʀ9fc�A�*

loss%K�=Rn�T       �	Ug�9fc�A�*

loss�H>�*b�       �	��9fc�A�*

loss�>��Ѡ       �	���9fc�A�*

lossC�=8��       �	$F�9fc�A�*

loss!�=���~       �	ߩ9fc�A�*

loss �=m\V       �	Oy�9fc�A�*

loss$�<��xU       �	8�9fc�A�*

loss���=�:9       �	��9fc�A�*

loss�?�<RL�       �	ު�9fc�A�*

loss@�K=�-�V       �	�B�9fc�A�*

loss�F�=N��Y       �	0׭9fc�A�*

lossi�=���
       �	�i�9fc�A�*

loss�r�=ct8�       �	��9fc�A�*

loss�C�=;8~B       �	f��9fc�A�*

lossr**>�,7�       �	gF�9fc�A�*

loss� �=�Ct�       �	K�9fc�A�*

loss�w<�y�       �	��9fc�A�*

lossߏ=_��{       �	S{�9fc�A�*

lossD)�<J�V�       �	P�9fc�A�*

loss���<=�j       �	��9fc�A�*

losseV=�J�       �	�D�9fc�A�*

loss��<�H��       �	�մ9fc�A�*

loss�P=�7��       �	Qi�9fc�A�*

lossH��<��M�       �	���9fc�A�*

lossN�P=Q��       �	��9fc�A�*

losslC>�l:       �	��9fc�A�*

lossͱH;��A       �	��9fc�A�*

loss�E<����       �	YP�9fc�A�*

loss���;t�)�       �	��9fc�A�*

loss�j�=�QB�       �	s��9fc�A�*

loss�>,���       �	]�9fc�A�*

loss�)>"�K�       �	Ŭ�9fc�A�*

loss�l<���Z       �	5B�9fc�A�*

loss��=�7�R       �	�ڻ9fc�A�*

loss��>X)&�       �	?q�9fc�A�*

loss�H�;���r       �	�
�9fc�A�*

lossF�q>��q       �	���9fc�A�*

loss��$>;\�       �	�G�9fc�A�*

loss�]�=�V�       �	W�9fc�A�*

loss�>�=
9S       �	��9fc�A�*

loss-�<=��X       �	@�9fc�A�*

lossa�=��5       �	g��9fc�A�*

lossOU�=|tYS       �	�V�9fc�A�*

loss4�3>΢(       �	n��9fc�A�*

loss�;�=;)C�       �	h��9fc�A�*

loss��=k���       �	)�9fc�A�*

loss���>�zx       �	L��9fc�A�*

loss5�>�ɉ�       �	.V�9fc�A�*

lossxP>P1]c       �	�Z�9fc�A�*

loss�U>��W�       �	n�9fc�A�*

lossL�>���       �	f��9fc�A�*

loss�tx=i�)�       �	3��9fc�A�*

loss$�>şQe       �	Ƈ�9fc�A�*

loss{ �=z�       �	H�9fc�A�*

loss|=��^N       �	T��9fc�A�*

loss�̓=�JB�       �	x��9fc�A�*

loss,��=:h�       �	i��9fc�A�*

loss�%>�D�       �	S=�9fc�A�*

loss�\`=r�M�       �	���9fc�A�*

loss��=e��       �	�t�9fc�A�*

loss�==���V       �	��9fc�A�*

loss7!�=E�Ĕ       �	Χ�9fc�A�*

loss)��=(�d       �	�<�9fc�A�*

lossH� >����       �	�g�9fc�A�*

loss�d>f�
�       �	���9fc�A�*

loss���=�I/       �	2��9fc�A�*

loss��.=�/
       �	�/�9fc�A�*

lossā=��u       �	
J�9fc�A�*

loss���<��       �	 ��9fc�A�*

loss���={N�7       �	
��9fc�A�*

loss(�,=�Y�g       �	�D�9fc�A�*

loss���<�B�       �	���9fc�A�*

loss�G>"� �       �	%��9fc�A�*

losswl�=w���       �	�7�9fc�A�*

loss��>gV�       �	��9fc�A�*

loss:]�=�~�E       �	���9fc�A�*

loss��C=LV\d       �	2�9fc�A�*

loss76�=k��d       �	[��9fc�A�*

loss&�>���&       �	fj�9fc�A�*

lossN��=�h�u       �	�9fc�A�*

loss�/[=��u�       �	'��9fc�A�*

loss��>8
�       �	6=�9fc�A�*

loss��=�?�       �	���9fc�A�*

lossO��=�B�       �	dt�9fc�A�*

loss�Z�=����       �	[�9fc�A�*

loss�Ԭ=�Ž�       �	@��9fc�A�*

loss��>� ��       �	�9�9fc�A�*

loss� > �e       �	8��9fc�A�*

loss�+>uS�       �	am�9fc�A�*

loss�x�=ه�       �	\�9fc�A�*

loss}T�=�f�       �	���9fc�A�*

loss4>=#4       �	�*�9fc�A�*

loss�0>:-y       �	���9fc�A�*

loss��$>_       �	PS�9fc�A�*

loss���=���       �	��9fc�A�*

loss��">l��       �	� :fc�A�*

loss���=���       �	n:fc�A�*

loss׹�<���       �	�:fc�A�*

loss,<�=�J       �	��:fc�A�*

loss�ȑ=e�Q�       �	lZ:fc�A�*

loss�ݣ<AX)       �	�:fc�A�*

loss|�>�(�       �	��:fc�A�*

lossxN�=���8       �	�]:fc�A�*

loss�9�=!��       �	:fc�A�*

loss�*�=�=��       �	F�:fc�A�*

loss8c�>zc�       �	�:fc�A�*

loss
�'>4��[       �	y�:fc�A�*

loss)&D>gXbw       �	�k	:fc�A�*

loss��>y?�       �	�
:fc�A�*

loss��1>��       �	ۢ
:fc�A�*

loss���=�"�       �	�I:fc�A�*

loss]X�<���       �	q:fc�A�*

loss1��=�.(       �	��:fc�A�*

lossA�J=�摋       �	%:fc�A�*

loss%j�=�|Z�       �	n�:fc�A�*

loss�=�C��       �	~n:fc�A�*

lossj�=�w��       �	#:fc�A�*

loss&�=���A       �	n�:fc�A�*

loss@O�=W��r       �	^�:fc�A�*

loss�CA>,��       �	Dn:fc�A�*

loss��=)��       �	�8:fc�A�*

loss2-�=<w`%       �	(:fc�A�*

loss��=���       �	F�:fc�A�*

loss��=�Y��       �	��:fc�A�*

loss ��>�qq�       �	]�:fc�A�*

loss��K>o�Y�       �	��:fc�A�*

loss4��==�A�       �	�:fc�A�*

loss���=��       �	e�:fc�A�*

lossD^�=�� `       �	��:fc�A�*

loss�()>�P        �	m:fc�A�*

loss# �=���       �	�6:fc�A�*

loss��>=��C       �	�7:fc�A�*

loss�q�=i��       �	\q:fc�A�*

lossw=s�Yq       �	�l :fc�A�*

lossL*�=�       �	#�!:fc�A�*

loss��=�Ǒ       �	!�":fc�A�*

loss��=>��       �	s�#:fc�A�*

loss��=:�/       �	�$:fc�A�*

losssw�=�l�1       �	!�%:fc�A�*

loss/!�>Z�d       �	s�&:fc�A�*

loss��7=�T�&       �	�W':fc�A�*

loss���<���       �	�':fc�A�*

loss�,=3�E       �	J�(:fc�A�*

loss��=((��       �	s/):fc�A�*

loss�7�=L@Y�       �	p�):fc�A�*

loss�4�=��e       �	dw*:fc�A�*

loss��">�f��       �		+:fc�A�*

loss�
�=�[��       �	��+:fc�A�*

loss�p�=�νI       �	�G,:fc�A�*

lossv�7=�D�r       �	��,:fc�A�*

loss�϶=U%�u       �	�{-:fc�A�*

loss�r=O�:       �	�.:fc�A�*

loss}|>:�]       �	��.:fc�A�*

lossm��=]�Q       �	0I/:fc�A�*

loss�T>Ʈ!�       �	��/:fc�A�*

lossf|�=��0�       �	w�0:fc�A�*

loss���=4MC�       �	�%1:fc�A�*

lossfo�=Ѓ�N       �	��1:fc�A�*

loss�T�=ĭ�       �	�a2:fc�A�*

loss��=�_kI       �	��2:fc�A�*

loss�>M�       �	��3:fc�A�*

loss���=���       �	Ps4:fc�A�*

lossѐf>`�Qf       �	5:fc�A�*

loss�I>X��       �	@�5:fc�A�*

loss 
=Jآ       �	�G6:fc�A�*

loss)��<���       �	��6:fc�A�*

loss��=��ֺ       �	H�7:fc�A�*

loss�)�=q��       �	�8:fc�A�*

loss!=�=��:�       �	�8:fc�A�*

loss*D=�I�v       �	=I9:fc�A�*

losst�=�#�       �	��9:fc�A�*

lossW��=�9)�       �	�v::fc�A�*

loss���=�[i       �	�;:fc�A�*

loss���=��       �	�;:fc�A�*

loss#ҩ='�LN       �	�D<:fc�A�*

loss�	-=��N�       �	<�<:fc�A�*

lossԒ>��T       �	p�=:fc�A�*

loss��>mTZ       �	�0>:fc�A�*

loss�<�<��UV       �	��>:fc�A�*

loss��=��B       �	/n?:fc�A�*

loss���=m���       �	�	@:fc�A�*

loss�9k=1م�       �	��@:fc�A�*

loss/)�=4N��       �	��A:fc�A�*

loss�З=Gi�;       �	*TB:fc�A�*

loss�<<>#�t       �	��B:fc�A�*

loss3m>ǹI{       �	ЗC:fc�A�*

loss%�J=Q ��       �	8D:fc�A�*

loss�*=��.       �	1�D:fc�A�*

loss=J�=�kG       �	�E:fc�A�*

loss�T�=S�(�       �	*�F:fc�A�*

loss���=��	�       �	�G:fc�A�*

loss�6>��.�       �	P9H:fc�A�*

lossn��<�=��       �	�I:fc�A�*

loss��=�)v�       �	�@J:fc�A�*

loss�J>�u��       �	�K:fc�A�*

loss�R�=�W�J       �	h<L:fc�A�*

loss���=�X�       �	�M:fc�A�*

loss�h�=��H       �	~�N:fc�A�*

loss��
=Th�       �	�tO:fc�A�*

loss�=Ym��       �	�P:fc�A�*

loss\]1=og       �	��Q:fc�A�*

lossl|=>�       �	�R:fc�A�*

loss�N�=	�Qa       �	�JS:fc�A�*

loss\l�=��       �	l�S:fc�A�*

loss]�=��į       �	��T:fc�A�*

loss�4�=�[�F       �	u?U:fc�A�*

loss6:�=oXS.       �	��U:fc�A�*

loss튐=u?r       �	͒V:fc�A�*

loss��l=ޮы       �	f2W:fc�A�*

loss�=t=O|<E       �	��W:fc�A�*

loss#�	=����       �	X:fc�A�*

lossi��=�'�?       �	&Y:fc�A�*

loss�(;>�6�s       �	�Y:fc�A�*

loss�=X�J       �	�cZ:fc�A�*

loss1�>���       �	� [:fc�A�*

loss�J�=0��       �	��[:fc�A�*

lossc�==���       �	7\:fc�A�*

loss�%_=}��%       �	l�\:fc�A�*

lossq�<�q�       �	�d]:fc�A�*

loss�2�<�2       �	T�]:fc�A�*

loss���=&v�
       �	��^:fc�A�*

lossߪ�<�U�       �	�5_:fc�A�*

loss�|�=�5g
       �	6�_:fc�A�*

lossi��=���9       �	�i`:fc�A�*

loss���=+��       �	Oa:fc�A�*

loss8��<��_�       �	��a:fc�A�*

loss�Ϣ<�J\       �	`b:fc�A�*

loss��g=����       �	��b:fc�A�*

loss�k�=�(�s       �	��c:fc�A�*

loss�p>����       �	7d:fc�A�*

loss�>,k�       �	/�d:fc�A�*

loss��=��S       �	��e:fc�A�*

loss&��=OG��       �	�?f:fc�A�*

loss�<�9�B       �	8�f:fc�A�*

loss���<��+       �	�yg:fc�A�*

loss�8�<��G�       �	&h:fc�A�*

loss�h</�x       �	{�h:fc�A�*

lossEv�=��"j       �	�ci:fc�A�*

lossoD�=�T�       �	Cj:fc�A�*

loss=��=i�`�       �	�j:fc�A�*

loss�=9�<�       �	�bk:fc�A�*

lossۃ�<q$l�       �	%\l:fc�A�*

lossM0>`?�H       �	��l:fc�A�*

lossB��=Vޚ       �	,�m:fc�A�*

loss��.>�A       �	�Gn:fc�A�*

loss�w�<eݵ        �	N�n:fc�A�*

loss|L�=[\       �	�o:fc�A�*

loss��>['W�       �	�q:fc�A�*

loss�==	a�       �	T�q:fc�A�*

loss��Y<e�?m       �	��r:fc�A�*

loss�ŗ=�^f�       �	[`s:fc�A�*

loss �=R�E       �	b�t:fc�A�*

loss���<���       �	�%u:fc�A�*

lossg=ϧW       �	q�u:fc�A�*

lossJ�P=kPJ�       �	&nv:fc�A�*

loss�1?>��S~       �	8w:fc�A�*

loss�x�=�4e�       �	�w:fc�A�*

lossช<����       �	Vx:fc�A�*

loss,�<��S       �	�x:fc�A�*

losst3>����       �	��y:fc�A�*

losso�	=��%       �	�\z:fc�A�*

lossd=��Hk       �	��z:fc�A�*

loss� �=�R֞       �	&�{:fc�A�*

loss�	>Nt�       �	�#|:fc�A�*

loss�b==pn2       �	ػ|:fc�A�*

lossD�< A�       �	�R}:fc�A�*

loss��5<]n�w       �	$~:fc�A�*

loss6�=�V��       �	׿~:fc�A�*

loss���=��"�       �	Z:fc�A�*

loss]��<��hi       �	M�:fc�A�*

loss���<�       �	���:fc�A�*

loss�m�=�h       �	z�:fc�A�*

lossF�=�M4       �	���:fc�A�*

loss��2=�2�=       �	VI�:fc�A�*

loss`��=h��       �	�܂:fc�A�*

loss��-=��;       �	zq�:fc�A�*

loss�)>)7m\       �	��:fc�A�*

lossZ��=��8Y       �	���:fc�A�*

loss\t!=MC�s       �	�3�:fc�A�*

loss
�i<���       �	b�:fc�A�*

loss�V�<lh�       �	��:fc�A�*

loss�=���[       �	b��:fc�A�*

loss��>\� �       �	�g�:fc�A�*

loss��=���       �	�)�:fc�A�*

lossX�>���       �	y"�:fc�A�*

loss4=�O��       �	��:fc�A�*

loss��z=��0       �	�=�:fc�A�*

loss|r�=�%0       �	�3�:fc�A�*

loss�;=��!       �	ʍ:fc�A�*

lossw��=���       �	�g�:fc�A�*

lossJ��=�*�       �	%�:fc�A�*

lossIz~=�G�\       �	w��:fc�A�*

lossHw�=��Y�       �	�@�:fc�A�*

lossF1�=���       �	��:fc�A�*

lossDWf=A9�       �	���:fc�A�*

lossV=�9}       �	��:fc�A�*

losszX�=joL�       �	���:fc�A�*

loss�L=���       �	�H�:fc�A�*

lossZ@I=#��       �	{ܓ:fc�A�*

lossOI%=�dh�       �	�o�:fc�A�*

lossqFO=�wm)       �	�
�:fc�A�*

lossD��=H��       �	稕:fc�A�*

loss�>{+}�       �	C�:fc�A�*

lossm_>��od       �	�ږ:fc�A�*

loss &>Hi��       �	!s�:fc�A�*

loss/�v>E�)       �	J	�:fc�A�*

loss�<�y,       �	裘:fc�A�*

loss�7�=1���       �	�:�:fc�A�*

loss� f=�B>       �	�Ι:fc�A�*

lossT�>��C$       �	sf�:fc�A�*

lossfS=���        �	���:fc�A�*

loss��='Ъ�       �	u��:fc�A�*

lossw�l>�A&       �	|e�:fc�A�*

lossh��={�       �	���:fc�A�*

loss�=)`a�       �	>��:fc�A�*

loss6�x=j�e�       �	4.�:fc�A�*

loss�/N=6�R       �	d˞:fc�A�*

loss@��<��>]       �	���:fc�A�*

loss��2>�F��       �	d<�:fc�A�*

lossM��=��       �	�:fc�A�*

loss�> >Rl       �	��:fc�A�*

lossF8=�3#4       �	�&�:fc�A�*

loss�be=
Wg       �	\̢:fc�A�*

loss��=���       �	�_�:fc�A�*

loss.Q
>e�       �	��:fc�A�*

loss�	�<4�A�       �	%��:fc�A�*

loss:��=��A�       �	Yi�:fc�A�*

loss���=NZ�        �	�:fc�A�*

lossR8�=�+       �	��:fc�A�*

loss�]=�FD�       �	:>�:fc�A�*

lossa��=ܬ�       �	�է:fc�A�*

loss�>�*ڥ       �	
��:fc�A�*

loss�H�=���*       �	@2�:fc�A�*

loss�@=���       �	
ک:fc�A�*

loss��=�
/�       �	3��:fc�A�*

loss6��=ï�S       �	fI�:fc�A�*

loss�V�=B�SH       �	J�:fc�A�*

loss.S>�N�       �	6��:fc�A�*

loss�L\=Κ�Z       �	�:�:fc�A�*

lossw^-=���P       �	��:fc�A�*

loss<�W=�:��       �	;��:fc�A�*

lossl�=�C       �		2�:fc�A�*

lossznQ=P~�       �	nޯ:fc�A�*

loss� �=.y�D       �	��:fc�A�*

loss)S)=`C�       �	9%�:fc�A�*

loss_I�=a�        �	�̱:fc�A�*

lossZZ}=j�;�       �	�w�:fc�A�*

loss&UC=$L��       �	��:fc�A�*

lossf�>���       �	x��:fc�A�*

lossȕs<	g2�       �	 ]�:fc�A�*

loss�.�=uĞl       �	��:fc�A�*

lossC�>���_       �	)��:fc�A�*

loss���=�UI        �	�Q�:fc�A�*

loss��=7^       �	���:fc�A�*

loss�NE>H,[       �	ӄ�:fc�A�*

loss�o>�K�]       �	��:fc�A�*

lossQ��<���s       �	���:fc�A�*

lossԵM=�/~       �	2V�:fc�A�*

lossL*�=&'_�       �	,�:fc�A�*

loss�q�=Bs=+       �	���:fc�A�*

loss��|=��uW       �	}%�:fc�A�*

lossvp>�ԛ       �	���:fc�A�*

losse��<��|�       �	yX�:fc�A�*

loss�/�=��T|       �	���:fc�A�*

loss㋑=_��       �	��:fc�A�*

loss2��=7(��       �	@4�:fc�A�*

loss�t�=v[��       �	�о:fc�A�*

loss��=�o|/       �	�|�:fc�A�*

loss���=ޱzu       �	�:fc�A�*

loss$A�<�}��       �	��:fc�A�*

loss��=�ԕp       �	�E�:fc�A�*

loss�ҭ=(��       �	!��:fc�A�*

lossF$�=�f       �	���:fc�A�*

lossX<��N#       �	a�:fc�A�*

loss"�=�_{�       �	̲�:fc�A�*

loss0Z=^��       �	^H�:fc�A�*

lossHӢ<u��       �	���:fc�A�*

loss�?�='{�       �	�u�:fc�A�*

loss3��<�;2       �	��:fc�A�*

loss�͈<m�&N       �	@��:fc�A�*

loss��d={��       �	�x�:fc�A�*

loss��#>��`�       �	��:fc�A�*

loss��m=���       �	��:fc�A�*

loss���=�B@�       �	\T�:fc�A�*

loss��=~�       �	x��:fc�A�*

lossB�>���       �	%��:fc�A�*

loss�l�<+�l       �	�2�:fc�A�*

loss�!�=�/�x       �	���:fc�A�*

lossk�=�T3       �	s�:fc�A�*

loss�c�=�� 
       �	��:fc�A�*

loss�>n=�Y}       �	���:fc�A�*

loss&/J>��Ap       �	O>�:fc�A�*

loss�G=4S0       �	��:fc�A�*

loss�X�=�R*�       �	�q�:fc�A�*

loss֧�=iO(       �	�
�:fc�A�*

loss�}=��B       �	���:fc�A�*

loss��6=�CQR       �	_F�:fc�A�*

loss5��=��L       �	���:fc�A�*

loss F�=�H^�       �	0��:fc�A�*

loss�Ύ=�I�=       �	J%�:fc�A�*

loss��u=�E�P       �	ܻ�:fc�A�*

lossW/�=��2       �	�S�:fc�A�*

loss���=3M��       �	���:fc�A�*

lossʜ�=�K�       �	6��:fc�A�*

loss��=C	��       �	0.�:fc�A�*

loss�ٙ=G�&#       �	)��:fc�A�*

loss	F=z��       �	�l�:fc�A�*

lossɃ4>�JS       �		�:fc�A�*

loss�ˑ=�%       �	��:fc�A�*

loss�(�=hw       �	�@�:fc�A�*

loss�)�=#`z       �	f��:fc�A�*

lossL�<>Nƈ#       �	.r�:fc�A�*

lossS��=$�7�       �	��:fc�A�*

lossNz�=��B       �	���:fc�A�*

lossT-�=p� �       �	�H�:fc�A�*

loss�$>ħ��       �	��:fc�A�*

loss���=����       �	/��:fc�A�*

loss.*q=�A��       �	���:fc�A�*

loss�ʼ<X�       �	u�:fc�A�*

loss�|\=��0�       �	��:fc�A�*

loss�8�=(�[       �	[��:fc�A�*

loss��=��9
       �	�b�:fc�A�*

loss�"�=`��       �	�Z�:fc�A�*

loss��=���       �	�h�:fc�A�*

lossn�>�G#       �	5	�:fc�A�*

loss�f<�P�       �	���:fc�A�*

loss���<Ipx�       �	�Y�:fc�A�*

lossw�=���L       �	���:fc�A�*

lossi��=\0~�       �	��:fc�A�*

loss��S=���2       �	�+�:fc�A�*

loss�^>QV@b       �	���:fc�A�*

loss��=&��s       �	eV�:fc�A�*

losss=:���       �	��:fc�A�*

loss%��=��c       �	Q��:fc�A�*

loss�+m=|K+       �	:$�:fc�A�*

loss���=���       �	;��:fc�A�*

lossx7=!��Y       �	�f�:fc�A�*

lossʓ\=xt�        �	��:fc�A�*

loss$�=G���       �	���:fc�A�*

loss�I�=��E       �	�7�:fc�A�*

loss�Q=f&�       �	K��:fc�A�*

loss�5�=��~       �	�_�:fc�A�*

loss��d<�X0�       �	���:fc�A�*

lossA�b=��RJ       �	q��:fc�A�*

lossHVz=�&��       �	�$�:fc�A�*

loss�=��ze       �	ж�:fc�A�*

loss�m=}� *       �	�L�:fc�A�*

lossW��=�k�       �	��:fc�A�*

lossd�U=�d�m       �	uw�:fc�A�*

loss��>=1�d       �	��:fc�A�*

loss��=�;       �	��:fc�A�*

loss���<�-�       �	78�:fc�A�*

losso.�=���=       �	���:fc�A�*

loss�.=X�       �	�d�:fc�A�*

loss��=4��X       �	��:fc�A�*

loss�=��n+       �	��:fc�A�*

lossm�s=m��       �	1E�:fc�A�*

loss함=!��       �	G��:fc�A�*

lossɤ9=��       �	�|�:fc�A�*

loss��<��l       �	+�:fc�A�*

loss�>�=È.�       �	p��:fc�A�*

loss�g<>>xFQ       �	�T�:fc�A�*

loss���=���       �	p��:fc�A�*

loss&=���       �	��:fc�A�*

loss��=>Q%~       �	$�:fc�A�*

loss�7�=0�^�       �	Q��:fc�A�*

lossL�'=v���       �	aP�:fc�A�*

loss�Ü=i�       �	���:fc�A�*

loss�l>75       �	�y�:fc�A�*

loss\�S=.&       �	� ;fc�A�*

loss�@�<�J�       �	�� ;fc�A�*

loss��=��       �	 X;fc�A�*

loss���=�x�       �	O;fc�A�*

loss!��=pN�*       �	ß;fc�A�*

loss��=���n       �	D;fc�A�*

loss�SR<�ԼL       �	�;fc�A�*

lossJ1=�1��       �	$�;fc�A�*

loss�>���d       �	5z;fc�A�*

loss�0=��       �	R(;fc�A�*

lossإ�=��       �	k�;fc�A�*

lossRK=��6       �	�q;fc�A�*

loss��>&0l�       �	;fc�A�*

lossM{�;A��       �	��;fc�A�*

loss42�=��z       �	
;fc�A�*

lossM�g=��       �	�9;fc�A�*

lossqa�=E��       �	�;fc�A�*

loss �:=�=�       �	��;fc�A�*

loss��>v�/�       �	K[;fc�A�*

loss��=1gf`       �	��;fc�A�*

loss���<Ɔ�(       �	��;fc�A�*

losso�j=��p       �	yw;fc�A�*

lossZ�==фfj       �	�h;fc�A�*

loss2�=�u�       �	�;fc�A�*

loss��[=��u�       �	�|;fc�A�*

loss���=�x       �	=+;fc�A�*

lossO!>=�`       �	a;fc�A�*

losst�=�ɖ�       �	5F;fc�A�*

loss�I%>
��       �	 ;;fc�A�*

lossl�=V��       �	�1;fc�A�*

loss3�9>�ՓN       �	��;fc�A�*

loss�#>�p��       �	S�;fc�A�*

loss\��=���       �	>�;fc�A�*

loss���<���       �	0);fc�A�*

loss<��<���       �	�;fc�A�*

loss{��=j!`       �	�o;fc�A�*

loss9�>�K�       �	�;fc�A�*

loss�<�<M��p       �	�<;fc�A�*

loss`>�&�       �	.U;fc�A�*

loss7w/=��       �	�N;fc�A�*

loss�Ǘ>�$       �	��;fc�A�*

loss�	�=~!       �	�,!;fc�A�*

loss��=p닏       �	s�";fc�A�*

lossy�=���       �	��#;fc�A�*

loss)�)=�\       �	�9$;fc�A�*

loss�qu=n��1       �	��$;fc�A�*

loss���<Nn@�       �	ܸ%;fc�A�*

lossd\>L2       �	c&;fc�A�*

lossW�=�Ɲ       �	�!';fc�A�*

loss��=ň�       �	��';fc�A�*

loss�c=���       �	�Y(;fc�A�*

losscC�=��Yz       �	V+);fc�A�*

loss�B�<���       �	3�);fc�A�*

loss�u�<ݡ�S       �	��*;fc�A�*

loss!�j<բ�g       �	F�+;fc�A�*

loss��=��       �	�h,;fc�A�*

loss]��<r�J       �	;9.;fc�A�*

loss�� >�e�       �	�.;fc�A�*

loss�>�}�k       �	@�/;fc�A�*

loss��=���J       �	�+0;fc�A�*

lossQ_'>P�(�       �	��0;fc�A�*

loss�@�<(�<       �	�j1;fc�A�*

loss�a�=�/�"       �	�2;fc�A�*

lossV�`>���       �	��2;fc�A�*

lossŻ�=�asi       �	�Z3;fc�A�*

loss��#=��A       �	3�3;fc�A�*

loss{�J=�L%�       �	��4;fc�A�*

loss	��=�eH       �	2:5;fc�A�*

loss���<�v2       �	��5;fc�A�*

lossJ\�<�:�r       �	g|6;fc�A�*

loss�\�=J}�        �	77;fc�A�*

loss�&�=����       �	��7;fc�A�*

loss�>�y�6       �	�N8;fc�A�*

lossXW�=���       �	��8;fc�A�*

loss�f�=��r       �	؂9;fc�A�*

loss24=��       �	:;fc�A�*

lossI��=AWeQ       �	ĳ:;fc�A�*

loss	R�<��       �	IJ;;fc�A�*

loss���<���       �	��;;fc�A�*

loss ҋ<G�=       �	x<;fc�A�*

loss�%>����       �	�=;fc�A�*

lossB�=o�	v       �	��=;fc�A�*

loss)!�<;�%       �	G;>;fc�A�*

lossH��=�6��       �	��>;fc�A�*

lossÆ=���_       �	rp?;fc�A�*

lossW�<����       �	�
@;fc�A�*

loss��<n	�       �	�@;fc�A�*

lossEF�=!{��       �	�CA;fc�A�*

loss�4=�J�       �	��A;fc�A�*

loss�*�=8��       �	�uB;fc�A�*

loss���=;�       �	IC;fc�A�*

loss��=���,       �	z�C;fc�A�*

loss�6<�*a�       �	�@D;fc�A�*

loss�7=��q�       �	��D;fc�A�*

loss2�<`��`       �	 zE;fc�A�*

loss��6=G��S       �	sF;fc�A�*

lossz>����       �	q�F;fc�A�*

loss�'>r�%       �	}%H;fc�A�*

loss�we=ڱ�       �	s�H;fc�A�*

loss�6�=:�1�       �	�J;fc�A�*

loss�%�=Wv�       �	�HK;fc�A�*

loss;��=Gǳ�       �	� L;fc�A�*

lossF�
=!�!�       �	o�L;fc�A�*

loss_�<U]��       �	�OM;fc�A�*

loss�W�=�b+6       �	*�N;fc�A�*

loss/>.=�t��       �	�eO;fc�A�*

loss���=� �       �	�P;fc�A�*

loss��=�B�       �	��P;fc�A�*

loss!cI=�1��       �	f�Q;fc�A�*

lossL>L<�^x       �	p�R;fc�A�*

loss��=M�\�       �	�CS;fc�A�*

lossL*=�2;�       �	b�T;fc�A�*

lossZ�<�83�       �	�U;fc�A�*

loss3��=�1�d       �	�DV;fc�A�*

lossB�<�r�       �	��V;fc�A�*

loss�B=V��       �	f�W;fc�A�*

loss>�.��       �	�Y;fc�A�*

loss�e!=O�r�       �	*�Y;fc�A�*

loss�գ<v��       �	1EZ;fc�A�*

loss� =�nX'       �	��Z;fc�A�*

loss��L<O��       �	�s[;fc�A�*

loss�w�=��s�       �	�
\;fc�A�*

loss��G=��       �	H�\;fc�A�*

lossv�=���f       �	6<];fc�A�*

losss- =Ϻ�       �	��];fc�A�*

loss���=�-�       �	�k^;fc�A�*

loss,F�=��       �	�_;fc�A�*

losssV�<���       �	��_;fc�A�*

lossؾ>-��       �	<`;fc�A�*

lossZ�=YS�       �	��`;fc�A�*

loss��=��|�       �	}ya;fc�A�*

lossl²<���       �	*b;fc�A�*

loss�y=�\��       �	��b;fc�A�*

lossE�>��       �	HRc;fc�A�*

losst�W=/�       �	��c;fc�A�*

loss��f=m��       �	��d;fc�A�*

loss$-�=j��       �	�+e;fc�A�*

lossؽ$=q�&�       �	��e;fc�A�*

loss��<[W�       �	�ff;fc�A�*

loss \�<C�$'       �	F	g;fc�A�*

lossZK7=�SB       �	�g;fc�A�*

loss�p0=t��       �	Vh;fc�A�*

loss'U=�yL�       �	i;fc�A�*

loss{�c=n":       �	��i;fc�A�*

lossg�=s=$       �	Pj;fc�A�*

loss�6�=`<Y       �	��j;fc�A�*

loss��<�4       �	Ęk;fc�A�*

lossn�$=�.��       �	�8l;fc�A�*

loss�O�=��J�       �	a�l;fc�A�*

loss#M�<ҙ�       �	g�m;fc�A�*

lossQ�;��"       �	J%n;fc�A�*

loss8��<LK%       �	@�n;fc�A�*

lossJ;�<�0�V       �	io;fc�A�*

lossp*=��k�       �	wp;fc�A�*

loss�N�;���X       �	%�p;fc�A�*

lossM�<f9�D       �	AIq;fc�A�*

loss8\8=m���       �	��q;fc�A�*

loss=�	;���{       �	Όr;fc�A�*

loss��;;N��       �	�.s;fc�A�*

lossM2�;M�5�       �	q�s;fc�A�*

losss:}<:���       �	qst;fc�A�*

lossߪM=I�v�       �	@u;fc�A�*

loss���<S��       �	��u;fc�A�*

loss�"�:Y̾�       �	F{v;fc�A�*

loss���=��
       �	8w;fc�A�*

loss��T>�C�       �	%�w;fc�A�*

loss�l�;z���       �	$�x;fc�A�*

lossJ�>3[��       �	�$y;fc�A�*

loss},�=��Z�       �	��y;fc�A�	*

loss��=�u>2       �	�pz;fc�A�	*

lossNQO<��}       �	+{;fc�A�	*

loss/G =��/�       �	Z�{;fc�A�	*

lossw�=o�֍       �	�]|;fc�A�	*

loss��=bv��       �	��|;fc�A�	*

loss�=ؐB�       �	ٓ};fc�A�	*

loss\�=��!       �	b/~;fc�A�	*

loss�W�=HU�#       �	�~;fc�A�	*

lossH�V>�o;0       �	|_;fc�A�	*

loss)�=�i       �	]�;fc�A�	*

loss�_	>��I       �	攀;fc�A�	*

loss��=�V�       �	�1�;fc�A�	*

loss	��=&0       �	`ˁ;fc�A�	*

loss��j=8�~       �	z�;fc�A�	*

loss��<��,       �	��;fc�A�	*

loss?��=��R�       �	���;fc�A�	*

loss�==����       �	�T�;fc�A�	*

loss-��<���       �	��;fc�A�	*

lossOR=�)}       �	׆�;fc�A�	*

loss���=[@�       �	`!�;fc�A�	*

loss�+�;�:�.       �	��;fc�A�	*

loss�D\=��c�       �	\��;fc�A�	*

lossD��<��M       �	}@�;fc�A�	*

loss�5=tm��       �	��;fc�A�	*

loss�)=�F�       �	#�;fc�A�	*

loss��>aτ�       �		��;fc�A�	*

lossa��=���       �	���;fc�A�	*

lossC��<~��       �	�'�;fc�A�	*

loss�2=g.0�       �	dΌ;fc�A�	*

loss3D<�Ӥ�       �	�j�;fc�A�	*

lossA��;[�X       �	�;fc�A�	*

loss�w0=
z��       �	'��;fc�A�	*

loss���=I��       �	���;fc�A�	*

lossH"�;�<��       �	N&�;fc�A�	*

loss�=���:       �	�ɐ;fc�A�	*

loss��>�;��       �	:u�;fc�A�	*

loss��=i�       �	��;fc�A�	*

lossŖ�<��Q�       �	aƒ;fc�A�	*

lossTkG=őn       �	j�;fc�A�	*

lossؙ�=�jR�       �	��;fc�A�	*

lossF�>�Ye"       �	���;fc�A�	*

loss{��<-%       �	iR�;fc�A�	*

lossqȫ=��@:       �	z��;fc�A�	*

loss4y�=p�؈       �	���;fc�A�	*

loss.��<�d�L       �	�R�;fc�A�	*

loss���=v>       �	���;fc�A�	*

loss#x�<x|�5       �	"��;fc�A�	*

lossV��<�7g       �	jm�;fc�A�	*

loss��>����       �	�U�;fc�A�	*

loss���=-ErY       �	R�;fc�A�	*

loss)L�=,�<9       �	��;fc�A�	*

loss��6=!��       �	F'�;fc�A�	*

loss��<m)��       �	���;fc�A�	*

loss�n�<0R       �	W_�;fc�A�	*

loss|�=���       �	7��;fc�A�	*

loss���=A��       �	���;fc�A�	*

loss�]>ָyl       �	�-�;fc�A�	*

loss���=�;��       �	z¸;fc�A�	*

loss���<~-��       �	���;fc�A�	*

loss�P=C�p�       �	?S�;fc�A�	*

loss�n6=����       �	��;fc�A�	*

lossW~�=�0       �	ς�;fc�A�	*

loss@�<�+�       �	P�;fc�A�	*

loss�>��j�       �	B�;fc�A�	*

loss{)<@
�       �	���;fc�A�	*

lossF�=H�G|       �	휾;fc�A�	*

loss�g=����       �	u��;fc�A�	*

loss ��= \q�       �	�U�;fc�A�	*

loss�Um=.       �	.W�;fc�A�	*

loss�$>�F�&       �	��;fc�A�	*

loss��2=@TK�       �	���;fc�A�	*

loss�Â=U���       �	�*�;fc�A�	*

loss8�>�7��       �	���;fc�A�	*

loss�u:=HP�,       �	en�;fc�A�	*

lossx��=���       �	W
�;fc�A�	*

loss���<�m��       �	���;fc�A�	*

loss\�x=\�y<       �	���;fc�A�	*

lossčs=u^�       �	��;fc�A�	*

loss���<2G~�       �	���;fc�A�	*

loss���=����       �	���;fc�A�	*

loss��=\ �s       �	%A�;fc�A�	*

loss��>��j@       �	�x�;fc�A�	*

loss��"=n,ʕ       �	�-�;fc�A�	*

loss��)=��]5       �	���;fc�A�	*

loss�}�<�:x�       �	�w�;fc�A�	*

loss�;=�#�       �	E�;fc�A�	*

loss(>���       �	:��;fc�A�	*

lossK�=bv�/       �	��;fc�A�	*

loss�<��,       �	�\�;fc�A�	*

lossq�=�o��       �	���;fc�A�	*

lossэ<b)O       �	���;fc�A�	*

lossQnl=����       �	[#�;fc�A�	*

loss{�=c��'       �	,��;fc�A�	*

loss�=x��       �	<L�;fc�A�	*

lossz�.=�+�       �	���;fc�A�	*

loss���=���       �	S{�;fc�A�	*

lossyU�=��       �	���;fc�A�	*

lossa�<�^�e       �	`�;fc�A�	*

loss��<,�
�       �	���;fc�A�	*

lossq��=+a       �	��;fc�A�	*

loss�=���       �	��;fc�A�	*

loss�e�>�gJ       �	�I�;fc�A�	*

loss*�!='ǂ�       �	X��;fc�A�	*

lossz��;�/t       �	 {�;fc�A�	*

lossJ��<q!�a       �	z�;fc�A�	*

lossa�R<GV�       �	A��;fc�A�	*

loss��C=�*       �	Ԃ�;fc�A�	*

loss� �=3X�       �	��;fc�A�	*

loss}g>�s<       �	ݵ�;fc�A�	*

loss69�=Pj�       �	*T�;fc�A�	*

loss!��<�vF       �	B!�;fc�A�	*

lossv�E='�;       �	#��;fc�A�	*

loss@0�=�a�c       �	��;fc�A�	*

lossfՏ=��"       �	�z�;fc�A�	*

loss	Y=YQ��       �	�;fc�A�	*

loss!�i=�TbG       �	��;fc�A�	*

lossJ�>ԍ��       �	���;fc�A�	*

loss�{�=��$�       �	�C�;fc�A�	*

lossvp'=��FP       �	P��;fc�A�	*

loss}� =��       �	�~�;fc�A�	*

loss�F�=�a��       �	��;fc�A�	*

loss=Zp=4{u�       �	|��;fc�A�	*

loss�D�=��!       �	3T�;fc�A�	*

loss��t<�M{b       �	���;fc�A�	*

loss�!	>����       �	]��;fc�A�	*

loss��=*<       �	&�;fc�A�	*

loss:^=�>=Z       �	��;fc�A�	*

loss�U�=e{�L       �	X�;fc�A�
*

lossO�c=v�+       �	���;fc�A�
*

lossP�>�=W�       �	'��;fc�A�
*

lossF�M=q���       �	2 �;fc�A�
*

lossb�=�;|M       �	J��;fc�A�
*

loss_�A=�^#�       �	���;fc�A�
*

lossCȌ<J�Kj       �	3�;fc�A�
*

loss�5i=�J�       �	
��;fc�A�
*

loss�	=Ao�J       �	>u�;fc�A�
*

loss��=��       �	�
�;fc�A�
*

loss��=uEg�       �	u��;fc�A�
*

loss���<t�       �	�C�;fc�A�
*

lossח�=����       �	M��;fc�A�
*

loss�#�=���       �	�s�;fc�A�
*

loss�3�<�.<�       �	U�;fc�A�
*

loss��h=��g       �	U��;fc�A�
*

loss/=�]�       �	�<�;fc�A�
*

loss��=n�@z       �	���;fc�A�
*

loss�!�<��5^       �	�j�;fc�A�
*

loss�rR>G:|       �	���;fc�A�
*

lossWCD=W�{�       �	��;fc�A�
*

loss�JL=�~o�       �	�=�;fc�A�
*

losssG�<W_��       �	4��;fc�A�
*

losss�Z=�aN       �	�y�;fc�A�
*

losse_<���u       �	��;fc�A�
*

loss#Y =*5�=       �	�J�;fc�A�
*

loss/�>�/��       �	o��;fc�A�
*

lossþ<����       �	���;fc�A�
*

loss��<�V��       �	�I�;fc�A�
*

losswͮ=>�       �	+��;fc�A�
*

lossJ>=WА       �	-<fc�A�
*

loss/��<+�r{       �	d�<fc�A�
*

loss$�T=)%�       �	�d<fc�A�
*

losso�={TCU       �	r�<fc�A�
*

loss�<�J�       �	��<fc�A�
*

loss}(�=���       �	`=<fc�A�
*

loss��c=f�y       �	��<fc�A�
*

loss>	=�^       �	1x<fc�A�
*

lossC��=w�e       �	<fc�A�
*

loss�ћ=53l       �	ܻ<fc�A�
*

loss�(<�m       �	�R<fc�A�
*

loss��g=&�       �	�p	<fc�A�
*

loss�2b=�n��       �	�
<fc�A�
*

loss�L�=w�Sw       �	!�
<fc�A�
*

lossg=� �       �	��<fc�A�
*

losst3<�yR       �	t<fc�A�
*

loss��=h��}       �	k<fc�A�
*

loss�|M=w�'       �	y<fc�A�
*

loss�ў=99��       �	��<fc�A�
*

loss��">N��       �	*�<fc�A�
*

lossZk=�:0       �	,E<fc�A�
*

loss��<�h�       �	�,<fc�A�
*

loss�*�=%�|}       �	�/<fc�A�
*

loss��`=�=Y�       �	��<fc�A�
*

loss�V�<�c]�       �	��<fc�A�
*

loss/�=�
 �       �	½<fc�A�
*

lossI�<sR/p       �	7P<fc�A�
*

loss�,�=�5`L       �	�<fc�A�
*

loss��
=B(�       �	J�<fc�A�
*

loss��%=�+o.       �	d�<fc�A�
*

loss�B�<�?j�       �	��<fc�A�
*

loss���;s�h�       �	c�<fc�A�
*

loss!D3=8�9\       �	�<fc�A�
*

loss�g�=5��       �	>&<fc�A�
*

loss� �="q>%       �	#�<fc�A�
*

loss��=t       �	�c<fc�A�
*

lossIv<<oH[       �	��<fc�A�
*

loss��<��Hi       �	��<fc�A�
*

loss���<�d%       �	k�<fc�A�
*

lossi��<��S�       �	�� <fc�A�
*

loss��T=2I�T       �	�!<fc�A�
*

loss�l=�j�       �	3�"<fc�A�
*

loss��>��^�       �	��#<fc�A�
*

loss���=�T�y       �	B�$<fc�A�
*

loss��n=�>v@       �	6t%<fc�A�
*

loss��N=�Ĳ`       �	�&<fc�A�
*

loss�f�;����       �	V�&<fc�A�
*

loss�ϕ=��       �	<�'<fc�A�
*

loss�>���       �	�))<fc�A�
*

lossl*>���       �	� *<fc�A�
*

lossJ2=���       �	��*<fc�A�
*

loss��#==�z�       �	Ӆ+<fc�A�
*

loss0�=���       �	�,<fc�A�
*

loss��2<���8       �	[�,<fc�A�
*

lossV��<��[�       �	bL-<fc�A�
*

loss(�=�Z+       �	�-<fc�A�
*

loss��A=�Wc       �	�.<fc�A�
*

lossg��<!(CJ       �	�!/<fc�A�
*

lossr��<qS��       �	��/<fc�A�
*

lossp�"=��       �	�_0<fc�A�
*

loss�0�<�W       �	�,1<fc�A�
*

loss���=K��U       �	H�1<fc�A�
*

loss�n�<�+�K       �	�{2<fc�A�
*

loss��u=a��       �	^3<fc�A�
*

loss}J�=`�*>       �	\�3<fc�A�
*

loss���<j�2       �	)>4<fc�A�
*

loss65�<�s@�       �	��4<fc�A�
*

lossa�>ԭ�       �	u"6<fc�A�
*

lossAd�=r�Ū       �	F�6<fc�A�
*

loss�u�<���       �	�V7<fc�A�
*

lossF�<�L�       �	�7<fc�A�
*

loss�*T;��e?       �	�8<fc�A�
*

loss:�~=�[�E       �	G:9<fc�A�
*

loss�wH=�%       �	��9<fc�A�
*

loss_w�<���       �	�e:<fc�A�
*

loss�U�<����       �	Qi;<fc�A�
*

loss@h}=M��?       �	�<<fc�A�
*

losscנ==���       �	/�<<fc�A�
*

lossn$7=���<       �	�U=<fc�A�
*

loss�x�=w��\       �	��=<fc�A�
*

loss:��=#W9'       �	3�><fc�A�
*

loss{,a=�M�Z       �	�%?<fc�A�
*

lossB�=V
       �	i�?<fc�A�
*

loss{�<f>�<       �	�c@<fc�A�
*

lossڠr<;�Q�       �	A<fc�A�
*

loss�M�</���       �	ʤA<fc�A�
*

loss�=�=O�+       �	FBB<fc�A�
*

loss�o�=7�ĭ       �	m�B<fc�A�
*

loss��A=�3��       �	}zC<fc�A�
*

losse�>��6�       �	8D<fc�A�
*

loss�RU=��0�       �	��D<fc�A�
*

lossX�b=Z���       �	sJE<fc�A�
*

loss�U3<�J��       �	*�E<fc�A�
*

loss�f<4P��       �	��F<fc�A�
*

loss��.=X9�       �	�%G<fc�A�
*

loss��w<xD�       �	k�G<fc�A�
*

loss}=K��b       �	`H<fc�A�
*

loss�ǯ=�Ȱ       �	��H<fc�A�
*

loss�ǫ=��;?       �	��I<fc�A�*

loss�P=�G       �	*7J<fc�A�*

loss� �=�]�       �	��J<fc�A�*

loss4�<LR�       �	epK<fc�A�*

loss��=J �       �	L<fc�A�*

loss�d<����       �	ŪL<fc�A�*

loss\<�<��\       �	�RM<fc�A�*

loss݌	>SvE       �	��M<fc�A�*

loss��=n]��       �	<�N<fc�A�*

loss�>�F��       �	qO<fc�A�*

loss'ۏ=PJ`       �	��O<fc�A�*

lossD��=��"       �	'�P<fc�A�*

loss���=)=       �	�Q<fc�A�*

loss=�=y!�$       �	�Q<fc�A�*

lossu<C�qU       �	0HR<fc�A�*

loss@�=×�       �	�R<fc�A�*

loss8n/=��'-       �	S<fc�A�*

loss/9<-�k�       �	�T<fc�A�*

loss)�%=V���       �	v�T<fc�A�*

loss���=���       �	9cU<fc�A�*

loss)��=2�       �	�V<fc�A�*

lossr<h�r�       �	L�V<fc�A�*

loss���=����       �	tAW<fc�A�*

lossQx<� �Y       �	3�W<fc�A�*

loss��<���B       �	}uX<fc�A�*

loss���=��'@       �	�Y<fc�A�*

lossT4=��l       �	>�Y<fc�A�*

lossXi�=��/�       �	�MZ<fc�A�*

loss���=�ͻ_       �	��Z<fc�A�*

loss$�<���2       �	g�[<fc�A�*

loss>��=Tw*       �	\<fc�A�*

lossa��=����       �	e�\<fc�A�*

loss���<�j�n       �	�|]<fc�A�*

loss�x0<�Y�       �		^<fc�A�*

loss`�X=�u�       �	(�^<fc�A�*

loss��<����       �	�J_<fc�A�*

loss�Ɍ=���X       �	��_<fc�A�*

loss0�=       �	��`<fc�A�*

loss���<H���       �	�$a<fc�A�*

loss��o=�jß       �	��a<fc�A�*

loss���<�       �	^b<fc�A�*

lossط�<�M��       �	��b<fc�A�*

loss1�<�E�       �	�c<fc�A�*

loss�6U=��KL       �	�cd<fc�A�*

lossס�=��̆       �	� e<fc�A�*

loss꒬=�|��       �	ުe<fc�A�*

lossx7�:ߏP�       �	�Nf<fc�A�*

loss=�L;2�CN       �	��f<fc�A�*

loss��<m�@       �	�g<fc�A�*

loss�y�<ꢣ�       �	6"h<fc�A�*

loss6yb=�RP�       �	<�h<fc�A�*

lossߪ><�B��       �	,ci<fc�A�*

loss�>�=��wV       �	 j<fc�A�*

loss�=�Ǒ�       �	��j<fc�A�*

loss}҃=;(��       �	xCk<fc�A�*

loss���=b�1�       �	��k<fc�A�*

loss���<�t)�       �	l<fc�A�*

loss\�=�7h       �	)$m<fc�A�*

lossr��<��       �	��m<fc�A�*

loss��=ozf�       �	\Tn<fc�A�*

loss�<�5Gq       �	]�n<fc�A�*

lossD�=@Ue�       �	ƨo<fc�A�*

loss���=B�4�       �	�Wp<fc�A�*

loss��$=�G��       �	��p<fc�A�*

loss^=v�O       �	��q<fc�A�*

loss�Z�=���       �	�6r<fc�A�*

loss�[�=Sn�       �	��r<fc�A�*

loss��=�0       �	vqs<fc�A�*

loss�V�=2
�:       �	Hu<fc�A�*

lossp8=,�9I       �	��u<fc�A�*

loss�=l�^       �	�Xv<fc�A�*

loss�͹=p^��       �	w�v<fc�A�*

loss �>��Z�       �	��w<fc�A�*

lossI��<<�@�       �	9x<fc�A�*

loss<�U<K���       �	N�x<fc�A�*

loss)�<����       �	Hpy<fc�A�*

lossL>=&��       �	jz<fc�A�*

loss�B�=��(�       �	ız<fc�A�*

loss�a�=Xzm�       �	P{<fc�A�*

lossAf=�w��       �	��{<fc�A�*

loss�w<�?\�       �	��|<fc�A�*

loss��< ���       �	S#}<fc�A�*

loss���<���{       �	�}<fc�A�*

lossC�(<�>�e       �	g�~<fc�A�*

loss)�K=#oxs       �	!�<fc�A�*

loss_��;�
~       �	�3�<fc�A�*

lossO)�;s��       �	[Ѐ<fc�A�*

loss�H<�h��       �	-|�<fc�A�*

loss���=��m�       �	�<fc�A�*

loss�<�1v       �	��<fc�A�*

loss�I>��.1       �	eS�<fc�A�*

lossM�5>�G2^       �	��<fc�A�*

lossF=�=��p       �	���<fc�A�*

lossD��=��r�       �	L5�<fc�A�*

loss�Ҫ=/��K       �	���<fc�A�*

loss�8�<0�R       �	���<fc�A�*

losse�-=G��E       �	<�<fc�A�*

loss�x=ɞ�       �	�Շ<fc�A�*

losscn>yLæ       �	���<fc�A�*

loss��a<w��#       �	�M�<fc�A�*

loss��=d�       �	��<fc�A�*

loss���<tW�        �		��<fc�A�*

loss�u=N�	�       �	�$�<fc�A�*

loss�ڈ<&n�       �	�<fc�A�*

loss�]f=\��       �	�X�<fc�A�*

lossa(�=zo�$       �	���<fc�A�*

lossӮ<����       �	���<fc�A�*

lossổ=���       �	d"�<fc�A�*

loss��=դ��       �	Զ�<fc�A�*

loss��<^T�       �	�J�<fc�A�*

loss�*T=gȶ�       �	���<fc�A�*

loss,�<��3w       �	hu�<fc�A�*

loss� 7=��D       �	��<fc�A�*

loss�VP=��       �	ۧ�<fc�A�*

lossv�]>�J       �	�=�<fc�A�*

loss�W�<  ԯ       �	|Ԓ<fc�A�*

loss� <J&?1       �	�h�<fc�A�*

loss��}=�B       �	��<fc�A�*

loss�0�=�`[�       �	���<fc�A�*

loss�/]=�T�6       �	���<fc�A�*

losso��==��	       �	J(�<fc�A�*

lossP�=�|�I       �	�<fc�A�*

loss���=bs�       �	�Q�<fc�A�*

loss��E= �1V       �	��<fc�A�*

loss�;,<h�*       �	���<fc�A�*

loss��<�۸�       �	��<fc�A�*

loss48o=�V�C       �	s��<fc�A�*

loss�P�=�/]q       �	���<fc�A�*

lossJ�=t���       �	2�<fc�A�*

loss��G=��f�       �	{��<fc�A�*

loss�g=�u�       �	�N�<fc�A�*

loss%_E=R�       �	��<fc�A�*

lossӾ�;(��       �	�}�<fc�A�*

loss6k=��e       �	��<fc�A�*

loss��=�BH       �	x��<fc�A�*

loss�{>=����       �	�Q�<fc�A�*

loss2"�=%r�d       �	,�<fc�A�*

lossOV�=��^$       �	���<fc�A�*

loss��=E�%c       �	�6�<fc�A�*

lossT��<�1v       �	ա<fc�A�*

loss�c�=G���       �	6s�<fc�A�*

loss*.e=�:��       �	-	�<fc�A�*

lossM��<f�       �	���<fc�A�*

loss��;i@�       �	�Y�<fc�A�*

loss���<)���       �	X �<fc�A�*

loss�Ҵ=�M`d       �	Y��<fc�A�*

loss2yP=cGS�       �	֬�<fc�A�*

loss�y�=d�!       �	�{�<fc�A�*

loss�4�<plL       �	��<fc�A�*

loss?*=��	�       �	ک�<fc�A�*

loss�)b=��z       �	�F�<fc�A�*

loss��<�$�       �	.�<fc�A�*

loss�Bt=D;        �	σ�<fc�A�*

loss@��<��$       �	9'�<fc�A�*

lossk;<-��       �	�ʫ<fc�A�*

loss��=�r��       �	�d�<fc�A�*

loss�>@�"�       �	c	�<fc�A�*

loss�ґ=B���       �	<fc�A�*

loss���=6���       �	,J�<fc�A�*

loss(*q<R̰�       �	X�<fc�A�*

lossW�]<Q��0       �	Ή�<fc�A�*

lossx��<�6�       �	c'�<fc�A�*

lossw�u<�s@       �	 ư<fc�A�*

lossZm=��*       �	d�<fc�A�*

loss��==��J�       �	��<fc�A�*

loss�@�<���t       �	���<fc�A�*

loss��'=��"�       �	y$�<fc�A�*

loss�l=ٜ�       �	轳<fc�A�*

lossr��=�'��       �	�U�<fc�A�*

loss�~8>����       �	29�<fc�A�*

lossJc�<�0<�       �	cѶ<fc�A�*

loss�� =��݈       �	Cs�<fc�A�*

loss-��= ��       �	��<fc�A�*

loss-==bOݵ       �	0��<fc�A�*

loss�p�=r��       �	�9�<fc�A�*

loss[&�=�@@       �	�׹<fc�A�*

loss�%�<����       �	���<fc�A�*

loss�D*<��8       �	g'�<fc�A�*

loss��C=[���       �	༻<fc�A�*

loss���<��\2       �	�P�<fc�A�*

loss���=�>m       �	d�<fc�A�*

loss���<�ݤ	       �	���<fc�A�*

loss
�S=�
��       �	}$�<fc�A�*

loss��<x]z�       �	���<fc�A�*

lossԮ=��5       �	�Ϳ<fc�A�*

lossj�=��|       �	�e�<fc�A�*

loss�\�<]zD�       �	��<fc�A�*

loss���=vL�       �	9��<fc�A�*

loss��Z>�Q�       �	 6�<fc�A�*

lossY�;���`       �	���<fc�A�*

loss͙�<˙��       �	/n�<fc�A�*

loss�=C]�D       �	~�<fc�A�*

loss��=��8G       �	p��<fc�A�*

loss}!=т�#       �	�2�<fc�A�*

loss��D=��:1       �	���<fc�A�*

loss��>�Z�        �	`�<fc�A�*

lossL�7=Kk��       �	���<fc�A�*

losstm�=`K�s       �	Ҏ�<fc�A�*

lossM�<J��       �	�<�<fc�A�*

loss�L!=n��       �	���<fc�A�*

lossW�e=��J�       �	��<fc�A�*

lossl�^={?{>       �	���<fc�A�*

loss�^=�Pp�       �	���<fc�A�*

loss�h�=^�ן       �	Mi�<fc�A�*

lossR��=�-�&       �	~�<fc�A�*

loss��(=Ӱ�C       �	�H�<fc�A�*

loss?�>���       �	���<fc�A�*

loss 8=ڼ��       �	D1�<fc�A�*

loss�*
>,��       �	���<fc�A�*

loss:�;��5M       �	�|�<fc�A�*

loss��:<zoD)       �	��<fc�A�*

loss[�==n���       �	���<fc�A�*

losse	>�u%}       �	���<fc�A�*

lossc*�;�5�F       �	B'�<fc�A�*

lossMtR=$��       �	���<fc�A�*

lossr�=5(�j       �	�.�<fc�A�*

loss�G�=E%       �	��<fc�A�*

lossb�=���       �	�'�<fc�A�*

loss�s�=���       �	Gw�<fc�A�*

lossT/=˝��       �	��<fc�A�*

loss�C�=4��       �	��<fc�A�*

loss$�O=q�       �	"��<fc�A�*

lossP�=�5fe       �	�g�<fc�A�*

loss��=,�o�       �	�<fc�A�*

loss�=T=7 �       �	���<fc�A�*

lossn=~=�w��       �	>�<fc�A�*

loss\�=�}       �	=��<fc�A�*

loss� =M��K       �	�o�<fc�A�*

lossW��<�*<�       �	��<fc�A�*

loss	��<���       �	3��<fc�A�*

loss1�E< ju�       �	>^�<fc�A�*

loss��z<q+�h       �	��<fc�A�*

lossLj<ǏQ�       �	��<fc�A�*

loss��=�(ɘ       �	�I�<fc�A�*

loss]�$>T|�i       �	���<fc�A�*

loss��8=íV       �	ޏ�<fc�A�*

loss4��=.���       �	o/�<fc�A�*

loss�=�<�[q       �	���<fc�A�*

lossHӈ<KG(�       �	�i�<fc�A�*

loss�?�=�pX�       �	�	�<fc�A�*

loss��<+��D       �	��<fc�A�*

loss�5=]B��       �		��<fc�A�*

loss��=҇�       �	0e�<fc�A�*

loss�>8��L       �	�y�<fc�A�*

loss��<;��       �	P�<fc�A�*

lossE{K=�w�       �	��<fc�A�*

lossV(�=�x��       �	5`�<fc�A�*

lossn��=�F��       �	��<fc�A�*

loss���<���u       �	@��<fc�A�*

lossQ%6=��_�       �	�H�<fc�A�*

loss%�<|M�       �	��<fc�A�*

loss�]<>]6H       �	ٓ�<fc�A�*

loss2�=���       �	�0�<fc�A�*

loss#�H=�a�^       �	��<fc�A�*

lossܶ�<mY^�       �	�x�<fc�A�*

lossL��<��M       �	��<fc�A�*

lossS#�=j"�Q       �	���<fc�A�*

loss�-�<a˟2       �	�^�<fc�A�*

loss%j�=�%��       �	��<fc�A�*

loss/�<"�1E       �	���<fc�A�*

loss)��=��       �	MI�<fc�A�*

loss�x�=��o�       �	;��<fc�A�*

loss�)�;�~       �	�}�<fc�A�*

lossN�x=L�t       �	��<fc�A�*

loss 	�<�f��       �	 ��<fc�A�*

loss�\=�0�O       �	\U�<fc�A�*

lossNT$>b�       �	��<fc�A�*

lossa�!<�؁       �	+��<fc�A�*

loss�o�=^�pE       �	�!�<fc�A�*

loss�j�<��_       �	��<fc�A�*

loss::=7v�T       �	�P�<fc�A�*

lossw�;<�K7�       �	���<fc�A�*

loss��=2��4       �	(~ =fc�A�*

lossS  >p~��       �	�=fc�A�*

loss(�0=�\CO       �	O�=fc�A�*

loss�>�<L���       �	�C=fc�A�*

lossͽ>�絅       �	��=fc�A�*

loss�[�=�hǄ       �	)w=fc�A�*

loss�a<�Es7       �	�=fc�A�*

loss�^�<4�c�       �	�=fc�A�*

loss!�=o��c       �	:A=fc�A�*

lossq��<��q*       �	��=fc�A�*

loss�Ί=4B��       �	]�=fc�A�*

loss��Z=	��N       �	�%=fc�A�*

loss;�5=�§       �	�=fc�A�*

lossLt�<���       �	�g=fc�A�*

loss��8=�E�       �		=fc�A�*

loss��s<�x4D       �	��	=fc�A�*

losso4l<�?��       �	�S
=fc�A�*

loss��=} �       �	��
=fc�A�*

loss
_q<�-Nf       �	g�=fc�A�*

loss_>=��d       �	��=fc�A�*

loss�}�<T/q�       �	��=fc�A�*

loss[
�<J<�       �	/M=fc�A�*

lossO�<���       �	R�=fc�A�*

lossV�=^��       �	s�=fc�A�*

loss
�=�EE       �	#�=fc�A�*

loss��<�       �	��=fc�A�*

loss���<"&�0       �	du=fc�A�*

loss�}�=��p`       �	�C=fc�A�*

loss�=�}hv       �	��=fc�A�*

loss��=,��       �	x�=fc�A�*

loss8�<��.       �	^�=fc�A�*

loss��=<Kl�       �	�	=fc�A�*

loss�S�=U��       �	�#=fc�A�*

lossj�\<K��       �	!==fc�A�*

loss&/=S�K�       �	��=fc�A�*

lossCf=U��       �	И=fc�A�*

lossvZk={��Y       �	�Z=fc�A�*

loss=dp�       �	�=fc�A�*

loss�&= C        �	�=fc�A�*

losss�i<i��C       �	{�=fc�A�*

lossT�P=�!u       �	�J=fc�A�*

loss�*0=�X       �	��=fc�A�*

lossڇ'=Zhh6       �	ٗ=fc�A�*

lossvsC;���       �	�? =fc�A�*

loss�kT=9�c       �	6� =fc�A�*

loss�ĸ<�nB=       �	χ!=fc�A�*

loss��S=*�r�       �	�)"=fc�A�*

loss��<�q�       �	��"=fc�A�*

loss(w=�4�       �	�i#=fc�A�*

loss�Y>��a       �	�	$=fc�A�*

lossI�<���       �	X�$=fc�A�*

loss��)<���_       �	|H%=fc�A�*

loss?��=�}�       �	��%=fc�A�*

loss�b<�3��       �	M�&=fc�A�*

loss��%<�X�|       �	�*'=fc�A�*

loss�v;Ozp�       �	m�'=fc�A�*

loss���;�g��       �	Rd(=fc�A�*

loss��;*<��       �	��(=fc�A�*

loss��<t�t       �	�)=fc�A�*

loss�X;��S       �	o+*=fc�A�*

loss�|�<#ʂv       �	��*=fc�A�*

loss�r�:�I�       �	�]+=fc�A�*

loss@�9����       �	��+=fc�A�*

loss�g:X{       �	ۊ,=fc�A�*

lossv��<,���       �	�,-=fc�A�*

loss�c�=�B�T       �	��-=fc�A�*

loss���< ͗       �	*T.=fc�A�*

lossO�3;�n�#       �	��.=fc�A�*

loss5�#<i�       �	G�/=fc�A�*

lossR�/>#E�       �	�H0=fc�A�*

loss�,i;�x       �	��0=fc�A�*

lossή>��`       �	�v1=fc�A�*

loss3Ax<�i�       �	�2=fc�A�*

loss��=j��       �	�2=fc�A�*

lossC�p=W��/       �	�S3=fc�A�*

loss47c=���s       �	p�3=fc�A�*

loss���=�ix!       �	||4=fc�A�*

loss���=)2(�       �	5=fc�A�*

lossū='(_�       �	X�5=fc�A�*

lossr��=x�       �	K6=fc�A�*

loss��=d�       �	Q�6=fc�A�*

loss���=f@Q       �	�y7=fc�A�*

loss�4�=�.d�       �	�8=fc�A�*

loss8��<���       �	ʩ8=fc�A�*

loss�Y�=�^Q       �	tC9=fc�A�*

loss��=��?       �	0�9=fc�A�*

lossk�=3KI�       �	�p:=fc�A�*

loss6o=����       �	�;=fc�A�*

loss�<�=�)R�       �	#�;=fc�A�*

loss�a=����       �	�6<=fc�A�*

loss&"�<��a9       �	��<=fc�A�*

loss�;=�Ni�       �	�o==fc�A�*

loss�|<�o��       �	\>=fc�A�*

lossr)=9�_�       �	ϟ>=fc�A�*

loss���<~�u�       �	9?=fc�A�*

loss�b�;���       �	��?=fc�A�*

loss��=�
<|       �	l@=fc�A�*

lossNM�<B��       �	�A=fc�A�*

loss���=��5*       �	&�A=fc�A�*

loss%��=�C       �	�SB=fc�A�*

lossj�<� �       �	��B=fc�A�*

loss���=�MQ       �	w�C=fc�A�*

lossX� <��n=       �	c&D=fc�A�*

loss��<��N       �	e�D=fc�A�*

loss��<�a       �	�cE=fc�A�*

loss]11=<��1       �	��E=fc�A�*

loss�a�<I[�J       �	�F=fc�A�*

loss{�0=����       �	%#G=fc�A�*

lossr5�=tg	       �	��G=fc�A�*

lossnW�=�JS       �	kbH=fc�A�*

loss�	�;�u�m       �	�I=fc�A�*

loss�i=�<:       �	�QJ=fc�A�*

loss.��<	?u       �	r�K=fc�A�*

loss_K=�Q2       �	�L=fc�A�*

loss�S~<�޿�       �	�<M=fc�A�*

loss��a=��$�       �	��M=fc�A�*

loss�r�=dw��       �	SzN=fc�A�*

loss到<��^       �	O=fc�A�*

loss��P=D25�       �	��O=fc�A�*

loss��<H��       �	mYP=fc�A�*

loss��<>�       �	��P=fc�A�*

loss�[O=(�H       �	1{j=fc�A�*

loss]��=����       �	W"k=fc�A�*

loss_�t=_8�b       �	W�k=fc�A�*

loss���=��       �	�ul=fc�A�*

loss���< ���       �	m=fc�A�*

loss�z�<���       �	��m=fc�A�*

lossT��=���s       �	�\n=fc�A�*

loss��<��k       �	�n=fc�A�*

loss� P>��YX       �	@�o=fc�A�*

loss��>!>U       �	�Dp=fc�A�*

loss��<q�נ       �		�p=fc�A�*

loss&#�=BWW       �	��q=fc�A�*

losstS(=�	�       �	r=fc�A�*

loss�j$>x=
       �	&5t=fc�A�*

loss$�n=�~��       �	��t=fc�A�*

loss ��='�W�       �	b�u=fc�A�*

loss��);k���       �	�_v=fc�A�*

loss��q<B�c�       �	�Rw=fc�A�*

loss�ۺ<-g�E       �	!�x=fc�A�*

losss�= )��       �	�sy=fc�A�*

lossƝJ=Gu�]       �	�z=fc�A�*

loss(x�=��       �	7�z=fc�A�*

loss��<g���       �	3�{=fc�A�*

loss6��='C�       �	�)|=fc�A�*

lossqe�<�v�J       �	��|=fc�A�*

lossz��<��(�       �		m}=fc�A�*

loss!��<.c��       �	�~=fc�A�*

lossva�<�N�D       �	�~=fc�A�*

loss���=a鋱       �	�H=fc�A�*

loss
BI=�3��       �	u�=fc�A�*

lossXe�<)e��       �	�|�=fc�A�*

lossf��=��V|       �	B"�=fc�A�*

loss�DM=\P9�       �	�Ɂ=fc�A�*

lossȳ�=c(]�       �	Rd�=fc�A�*

loss�1a=��QN       �	�	�=fc�A�*

loss��f=��       �	e��=fc�A�*

loss*[�<Ƙ�       �	4K�=fc�A�*

loss��=b�[       �	��=fc�A�*

loss��=�'+       �	���=fc�A�*

loss���<r"�       �	�=�=fc�A�*

loss��N=K��R       �	�=fc�A�*

loss&�3=��Q       �	��=fc�A�*

loss�p,=doQ�       �	�'�=fc�A�*

lossT�N=���       �	Ј=fc�A�*

lossg6=7�       �	vq�=fc�A�*

lossd8�=���Q       �	#�=fc�A�*

loss�[�<4H�       �	���=fc�A�*

lossf��=��j       �	6�=fc�A�*

lossyג=�-��       �	b��=fc�A�*

loss��;��       �	P9�=fc�A�*

loss��<j�       �	�(�=fc�A�*

loss۷U=�t�w       �	@=fc�A�*

lossp'<u�       �	�Y�=fc�A�*

loss8��=�*$�       �	���=fc�A�*

losst2�=T�|�       �	���=fc�A�*

losst�<Ә�       �	%>�=fc�A�*

loss���;1x�       �	&��=fc�A�*

loss5�:R�.c       �	,��=fc�A�*

losswn�=ӕ�n       �	%!�=fc�A�*

loss�Հ=��       �	<��=fc�A�*

loss�Y1>�u��       �	�_�=fc�A�*

loss=��<iu�       �	���=fc�A�*

loss�U�<��       �	���=fc�A�*

loss�p^=v �       �	�<�=fc�A�*

lossؗA=א)�       �	�ז=fc�A�*

loss�/o<)V       �	Dm�=fc�A�*

lossO$�<L�xQ       �	��=fc�A�*

loss8*�<�F��       �	���=fc�A�*

loss��a=}V�       �	�:�=fc�A�*

lossT�=�lF�       �	|ә=fc�A�*

loss�r^<K��\       �	.q�=fc�A�*

loss��<�D       �	d�=fc�A�*

lossz�I=��L       �	U��=fc�A�*

loss�Be=6ۅ�       �	(I�=fc�A�*

loss���<��o�       �	��=fc�A�*

loss��< �r�       �	ᘝ=fc�A�*

lossf,i<���       �	�0�=fc�A�*

loss�E�<е�       �	CȞ=fc�A�*

loss3C�<����       �	�]�=fc�A�*

loss��<�P�       �	R�=fc�A�*

lossC�=�<W,       �	=ՠ=fc�A�*

loss�f=�r�       �	�l�=fc�A�*

lossħ�=ڨ�9       �	��=fc�A�*

loss��<s(       �	j��=fc�A�*

loss�w=��.9       �	B?�=fc�A�*

loss��8<r��       �	�գ=fc�A�*

loss~=^�S-       �	�j�=fc�A�*

loss%�S<LQɀ       �	. �=fc�A�*

loss�K<_I�Z       �	�ĥ=fc�A�*

loss!GK=�1V       �	Va�=fc�A�*

loss�jA=vL�d       �	o��=fc�A�*

lossI�F=�	V       �	Y�=fc�A�*

loss{�=�WZz       �	���=fc�A�*

lossg9�=J��       �	#K�=fc�A�*

loss6�=���D       �	��=fc�A�*

loss�'�<��       �	֍�=fc�A�*

loss�>̾n�       �	�(�=fc�A�*

loss��z=ō�       �	[ͫ=fc�A�*

lossH��=� O1       �	�l�=fc�A�*

loss_4=��^F       �	F	�=fc�A�*

loss?D=��       �	H��=fc�A�*

lossW�=�T�       �	sH�=fc�A�*

loss�'�<��       �	�ܮ=fc�A�*

loss��+=�aLa       �	iq�=fc�A�*

loss�C=ԒA       �	��=fc�A�*

loss�â=>��,       �	���=fc�A�*

loss���<\1��       �	�@�=fc�A�*

loss8;�=���       �	��=fc�A�*

loss���=��Z�       �	܄�=fc�A�*

loss�s�<�{�       �	�&�=fc�A�*

loss�<ZeEN       �	[γ=fc�A�*

loss�K�<>G�       �	�n�=fc�A�*

lossJr-=�t��       �	��=fc�A�*

lossu<�
��       �	-��=fc�A�*

loss�=y?}�       �	�K�=fc�A�*

loss=E\=���f       �	_�=fc�A�*

loss�z�<|�~       �	k��=fc�A�*

loss7�h=��O       �	�B�=fc�A�*

lossn�<2�Y�       �	L�=fc�A�*

loss�2Z<��e[       �	�}�=fc�A�*

loss��=�v�>       �	P�=fc�A�*

loss�1�=N�)       �	$��=fc�A�*

loss�Z7<���       �	�R�=fc�A�*

lossݼ=���       �	��=fc�A�*

loss!��<;�6�       �	e��=fc�A�*

loss\�X=Lɝ�       �	�/�=fc�A�*

losstV�=�x
       �	 ɽ=fc�A�*

loss'�<��W       �	�e�=fc�A�*

lossn..=3R�       �	i�=fc�A�*

loss�#=��_       �	8��=fc�A�*

loss��M=�-�       �	-@�=fc�A�*

loss;'<�VM       �	��=fc�A�*

lossL�<_z       �	�|�=fc�A�*

loss
s=#��       �	�=fc�A�*

loss���=V���       �	���=fc�A�*

lossJӭ<7�c       �	�G�=fc�A�*

loss�ϒ<�K�_       �	&�=fc�A�*

loss��=�n       �	ݖ�=fc�A�*

loss�ft;���       �	�<�=fc�A�*

lossmd�;��b�       �	j��=fc�A�*

loss��;!�m�       �	���=fc�A�*

loss�2Y<�ى       �	�y�=fc�A�*

loss��=߉�       �	?�=fc�A�*

loss��=AE��       �	ٳ�=fc�A�*

loss�-�=ouU       �	S]�=fc�A�*

lossr"=Qz�e       �	j��=fc�A�*

loss	A�=?�f       �	���=fc�A�*

lossE��;�
w�       �	���=fc�A�*

loss8��;�w[�       �	�=fc�A�*

loss�[k=�78!       �	
��=fc�A�*

lossn<?�w        �	g��=fc�A�*

lossw�<4�)       �	���=fc�A�*

loss�ѭ<��i       �	Zh�=fc�A�*

loss��<1ru�       �	��=fc�A�*

lossأQ=Գ�       �	��=fc�A�*

loss���;A�2�       �	�I�=fc�A�*

loss&Rx;G��"       �	��=fc�A�*

loss�e>~��       �	K��=fc�A�*

loss��W=x��       �	�.�=fc�A�*

lossh�_=��!�       �	���=fc�A�*

loss�l�;ĥ�       �	l�=fc�A�*

loss(��:�J�!       �	�=fc�A�*

loss�u�<Y���       �	=��=fc�A�*

lossI~;���       �	�P�=fc�A�*

loss��=!���       �	x��=fc�A�*

lossL�<[GB       �	���=fc�A�*

loss��=       �	_��=fc�A�*

loss{��<��       �	�.�=fc�A�*

loss�u=�D�       �	H��=fc�A�*

loss$]=6�       �	~W�=fc�A�*

loss[u�<���       �	���=fc�A�*

loss�1<>J�L       �	
��=fc�A�*

loss��<5���       �	b�=fc�A�*

lossnT�=�H�o       �	���=fc�A�*

loss�@$;Ҁ��       �	U�=fc�A�*

loss�0!=]*�       �	ٵ�=fc�A�*

lossᛔ=I�i       �	uV�=fc�A�*

loss�ל=�̀[       �	���=fc�A�*

loss2>��E�       �	H��=fc�A�*

lossӴ;@�˙       �	�%�=fc�A�*

loss���:���Q       �	e��=fc�A�*

loss��d=�w<�       �	Z�=fc�A�*

lossߠ=#�0&       �	c��=fc�A�*

loss�?�<P��       �	f��=fc�A�*

loss��;���R       �	�'�=fc�A�*

loss��m<Ƶ0w       �	*��=fc�A�*

loss���=C��       �	jl�=fc�A�*

loss)+=/��       �	�=fc�A�*

loss�<d�       �	Y��=fc�A�*

lossL_�<�b?�       �	�j�=fc�A�*

lossݡ�=ݠ�       �	��=fc�A�*

lossTE�<�S       �	>��=fc�A�*

lossx�2=ҽ�q       �	�X�=fc�A�*

loss��=��H4       �	D��=fc�A�*

losst��;��>�       �	b��=fc�A�*

loss��z=�1       �	�=�=fc�A�*

loss��B>�J�C       �	���=fc�A�*

loss��A=���*       �	G��=fc�A�*

loss	̝=�|�       �	�4�=fc�A�*

lossJ^<��;       �	���=fc�A�*

loss�a=�"�%       �	i�=fc�A�*

loss�=�Yb�       �	��=fc�A�*

loss&=�_�\       �	K��=fc�A�*

loss| V=�ů�       �	�Q�=fc�A�*

loss�<]�?       �	��=fc�A�*

loss9=��4       �	ė�=fc�A�*

loss�=y+(�       �	y��=fc�A�*

loss���=�꒎       �	���=fc�A�*

loss$X=�;A�       �	g)�=fc�A�*

loss���=�	5?       �	4��=fc�A�*

loss�M�<����       �	*r�=fc�A�*

lossoZ=��       �	&�=fc�A�*

loss�`~<��x�       �	��=fc�A�*

lossF	�<�S�       �	�V�=fc�A�*

loss��U=]�2       �	���=fc�A�*

lossy��=!EyV       �	��=fc�A�*

loss�">�2�       �	�/�=fc�A�*

loss<g�=}���       �	���=fc�A�*

lossh)�=(�;�       �	�q�=fc�A�*

lossi.=�ź�       �	�=fc�A�*

loss��A=��$`       �	+��=fc�A�*

loss��g<�+�1       �	a4�=fc�A�*

lossc��<���       �	��=fc�A�*

loss�w�=�\       �	Gu�=fc�A�*

loss�z�<k�M       �	��=fc�A�*

loss*4=���       �	���=fc�A�*

loss�X'=/���       �	�] >fc�A�*

lossI�R=�9 �       �	� >fc�A�*

loss�o<��>�       �	C�>fc�A�*

loss�w�=�G�U       �	(H>fc�A�*

losst<���       �	^�>fc�A�*

lossD�;,	@       �	K�>fc�A�*

lossQ�<o��       �	�_>fc�A�*

loss�<Q2��       �	;�>fc�A�*

loss�c=���\       �	��>fc�A�*

lossl�<餗r       �	�:>fc�A�*

loss���=@[       �	��>fc�A�*

loss���<mwB�       �	*�>fc�A�*

loss(f�=_��       �	�)>fc�A�*

lossT��<O���       �	J�>fc�A�*

loss�G<[�       �	�s	>fc�A�*

lossD�=�;�U       �	�
>fc�A�*

lossV�>{D�       �	?�
>fc�A�*

loss��=�j=�       �	#i>fc�A�*

loss�	�=�RX       �	z�>fc�A�*

lossS��<���/       �	�>fc�A�*

loss�c�;���M       �	�=>fc�A�*

loss1G=y;,�       �	s�>fc�A�*

loss��=�H�v       �	?s>fc�A�*

loss-�<��l�       �	�>fc�A�*

loss��R=�:��       �	��>fc�A�*

losst�V=����       �	;:>fc�A�*

loss��F<nӃ$       �	��>fc�A�*

loss|P�<K�`K       �	�x>fc�A�*

lossw�;�ɞ       �	>fc�A�*

loss�<�<%�,       �	X�>fc�A�*

loss�[<����       �	jL>fc�A�*

loss=���       �	��>fc�A�*

loss
�.=��@�       �	��>fc�A�*

lossʷW=v�n�       �	�>fc�A�*

loss���<��       �	��>fc�A�*

loss$|o=�f��       �	"q>fc�A�*

loss��Z=}U��       �	#>fc�A�*

loss�'�=���       �	��>fc�A�*

loss6��</�bi       �	be>fc�A�*

loss���<m���       �	>>fc�A�*

loss��=�i�&       �	X�>fc�A�*

loss�DA=v��`       �	�I>fc�A�*

loss�c='�-�       �	��>fc�A�*

loss�5�<}�OE       �	�P>fc�A�*

loss,dP<��2�       �	R�>fc�A�*

losstJ<σ<       �	�r>fc�A�*

loss
6=Z�T�       �	.>fc�A�*

lossazO<�BW       �	��>fc�A�*

lossä�=�6n)       �	TR >fc�A�*

loss�P�=o:2�       �	�>!>fc�A�*

loss��<C�Nk       �	j�!>fc�A�*

loss�A�:0�       �	AI#>fc�A�*

lossH	�<�v�a       �	�$>fc�A�*

loss��u=$�g       �	��$>fc�A�*

lossc�<��`�       �	GY%>fc�A�*

loss)�<��       �	��%>fc�A�*

loss��C=�r��       �	ę&>fc�A�*

loss�?�<��}u       �	BB'>fc�A�*

loss�`=Bm35       �	}�'>fc�A�*

loss@�=y���       �	H�(>fc�A�*

loss�`�<ah4�       �	�%)>fc�A�*

loss�{�<,g�U       �	�)>fc�A�*

loss4zm=��?       �	�X*>fc�A�*

loss��q;R8�       �	��*>fc�A�*

lossQ�`;��M       �	G�+>fc�A�*

lossE��<��       �	+,>fc�A�*

loss��;�D��       �	��,>fc�A�*

lossJPZ;��       �	_->fc�A�*

lossd�<$��       �	�.>fc�A�*

loss�)=%�1Z       �	�.>fc�A�*

loss�_=�eX       �	�0>fc�A�*

loss.�=/>�,       �	f�0>fc�A�*

losslEM=�*�W       �	�R1>fc�A�*

lossR_=�{��       �	�1>fc�A�*

loss��=s�bX       �	ѕ2>fc�A�*

loss�N-<5TG4       �	=3>fc�A�*

loss��_<P(n       �	��3>fc�A�*

loss��M=p���       �	}4>fc�A�*

loss(�k=cg6[       �	�"5>fc�A�*

loss��J>>���       �	�5>fc�A�*

loss���;�e       �	�a6>fc�A�*

loss�f=I�6       �	87>fc�A�*

lossE��<Y1�I       �	��7>fc�A�*

lossV�=���y       �	�y8>fc�A�*

loss��<�i��       �	)9>fc�A�*

loss$]�<'q)       �	^�9>fc�A�*

loss�<�8       �	�~:>fc�A�*

loss=<[�A4       �	�&;>fc�A�*

loss��8<�i:       �	b�;>fc�A�*

loss}|=S$�       �	Kw<>fc�A�*

lossĿ�:�옪       �	i=>fc�A�*

loss���;hB�       �	��=>fc�A�*

loss�f�<yz       �	ga>>fc�A�*

loss�$�<�/�       �	&�>>fc�A�*

loss�0=0;�       �	h�?>fc�A�*

loss7��=�,       �	�2@>fc�A�*

loss�{=�٨       �	��@>fc�A�*

loss㪦;q��u       �	qA>fc�A�*

loss��=�EBV       �	ZB>fc�A�*

loss.�(=�fgi       �	G�B>fc�A�*

lossc= �O�       �	HC>fc�A�*

loss�v;=/]��       �	h�C>fc�A�*

loss�<=
��0       �	��D>fc�A�*

loss.eF=�N�       �	LE>fc�A�*

lossW�)<l@�&       �	
�E>fc�A�*

loss��Q<��       �	ZeF>fc�A�*

lossVC =Μ       �	?G>fc�A�*

loss܅�;�-��       �	X�G>fc�A�*

loss��G=��R       �	�DH>fc�A�*

loss�~�=��j       �	i�H>fc�A�*

loss��=G
��       �	�I>fc�A�*

loss͆_=���       �	�J>fc�A�*

lossM��=d�ֳ       �	cK>fc�A�*

loss�o�:i��v       �	tL>fc�A�*

lossD^<c:w~       �	�L>fc�A�*

lossW$[<����       �	�OM>fc�A�*

losszX�=W[�_       �	��M>fc�A�*

loss���<Q�OU       �	8�N>fc�A�*

loss�w=rr��       �	H4O>fc�A�*

lossd)�=r�M       �	��O>fc�A�*

loss�f<���       �	�}P>fc�A�*

loss��=�B��       �	�Q>fc�A�*

loss�=�.�Y       �	��Q>fc�A�*

loss��k=��+       �	�aR>fc�A�*

lossHh�:�Bk�       �	.S>fc�A�*

loss� =��`�       �	 �S>fc�A�*

lossC==n�T       �	6WT>fc�A�*

loss.J=6��W       �	Y�T>fc�A�*

lossT�>>���       �	J�U>fc�A�*

loss���;���       �	&QV>fc�A�*

loss߷�;;#�       �	
�V>fc�A�*

lossτ�<�9�X       �	6�W>fc�A�*

loss�6�;@��       �	�,X>fc�A�*

loss�E=��B       �	��X>fc�A�*

lossZ�>=P�Y       �	�cY>fc�A�*

lossn5x< i'l       �	�
Z>fc�A�*

lossdT�<�3A�       �	%�Z>fc�A�*

loss.��=��       �	�B[>fc�A�*

loss/P�<�0��       �	��[>fc�A�*

lossE��=�&_�       �	�z\>fc�A�*

loss�>HJ4�       �	]>fc�A�*

loss_��<��.       �	&�]>fc�A�*

loss�}<b)F�       �	=^>fc�A�*

loss�Ӈ<?K@       �	^�^>fc�A�*

loss���<�y�       �	~n_>fc�A�*

lossg�=�Ɨ       �	�`>fc�A�*

lossc<,�9�       �	��`>fc�A�*

loss@P�<sx�        �	�3a>fc�A�*

loss�U=�#       �	��a>fc�A�*

lossa?=���       �	Pmb>fc�A�*

loss�"�=�%��       �	4c>fc�A�*

loss��=$���       �	Ŭc>fc�A�*

loss�C=G�M@       �	�Td>fc�A�*

loss�r4=�2       �	��d>fc�A�*

loss��'=^��E       �	&�e>fc�A�*

lossI:�<���       �	+�f>fc�A�*

lossP�=�}�       �	��g>fc�A�*

loss�<��K�       �	�/h>fc�A�*

loss�!<�X�G       �	VFi>fc�A�*

loss}�=S�E�       �	��i>fc�A�*

loss���<�)�       �	$|j>fc�A�*

loss�|�=�і�       �	�k>fc�A�*

loss�O�=��oR       �	u�k>fc�A�*

lossR��:�       �	�Fl>fc�A�*

loss�&=1�ˏ       �	��l>fc�A�*

loss�A�<���       �	��m>fc�A�*

lossX�=�o�c       �	jn>fc�A�*

lossW/�<]�i�       �	4-o>fc�A�*

loss���<�;,       �	��o>fc�A�*

loss���=�`j       �	-yp>fc�A�*

loss<W�:Yh       �	q>fc�A�*

loss�s�<H�T�       �	��q>fc�A�*

loss��e<��U�       �	)�r>fc�A�*

lossad=ߐu       �	(bs>fc�A�*

loss��s<��G       �	�2u>fc�A�*

lossA~<����       �	��w>fc�A�*

loss �=Kd��       �	%@x>fc�A�*

loss.Ͷ<�@b�       �	��x>fc�A�*

loss��=Q⤘       �	�y>fc�A�*

loss�=���       �	=,z>fc�A�*

loss<=��mY       �	��z>fc�A�*

loss��= B�       �	
j{>fc�A�*

lossv�&<=ܮ       �	6|>fc�A�*

loss�9=k��S       �	_	~>fc�A�*

loss��=HK�       �	��~>fc�A�*

loss�}�=�B�       �	-A>fc�A�*

lossAK�<�e�       �	��>fc�A�*

loss�'�=eޠs       �	���>fc�A�*

loss��h=ȥ��       �	��>fc�A�*

loss�OW=��L�       �	q��>fc�A�*

loss�x�;���d       �	SA�>fc�A�*

loss$Kc<-Q`M       �		7�>fc�A�*

lossh1=���       �	�܄>fc�A�*

loss�W6=!�:#       �	���>fc�A�*

lossWQF<X�y�       �	��>fc�A�*

lossVp�=N�
�       �	Q��>fc�A�*

loss�|<�S�H       �	�`�>fc�A�*

loss2��=�'o       �	 a�>fc�A�*

loss�=�7�       �	� �>fc�A�*

loss�ї=w5"�       �	���>fc�A�*

loss�h3=���       �	�֊>fc�A�*

loss�=s��       �	���>fc�A�*

lossnb<����       �	���>fc�A�*

loss=F <a/S       �	��>fc�A�*

loss��-<��=       �	t��>fc�A�*

loss죟;��#       �	�j�>fc�A�*

lossL<a���       �	a�>fc�A�*

loss��<��       �	ZJ�>fc�A�*

lossl�r<���       �	�2�>fc�A�*

loss��<Ã�       �	1Ғ>fc�A�*

loss��<Fe�       �	�h�>fc�A�*

loss�~i<���       �	�>fc�A�*

loss�W= u|8       �	�є>fc�A�*

loss��3<�!�       �	i�>fc�A�*

loss��o=�       �	��>fc�A�*

lossxݡ=M���       �	��>fc�A�*

lossۿ�<%���       �	TR�>fc�A�*

loss(��={L�i       �	��>fc�A�*

loss�~E<<��       �	�>fc�A�*

loss[�"=�j�5       �	d$�>fc�A�*

lossZj>�J`4       �	�ƙ>fc�A�*

loss�)=�=#I       �	�h�>fc�A�*

loss��<yn�&       �	��>fc�A�*

loss�L>cj       �	'��>fc�A�*

loss�K�<�1�v       �	�I�>fc�A�*

loss��b<����       �	��>fc�A�*

loss���;=��       �	���>fc�A�*

loss�=^�(       �	�;�>fc�A�*

loss�{�=�:�,       �	�>fc�A�*

loss�Qb<�*P       �	���>fc�A�*

loss��=̤�M       �	�(�>fc�A�*

loss���<���w       �	A֠>fc�A�*

loss�(B<s(C�       �	uu�>fc�A�*

loss���=�p�;       �	� �>fc�A�*

loss�-X=��:�       �	vâ>fc�A�*

lossO[<r�t       �	�e�>fc�A�*

lossSW<hAf~       �	q�>fc�A�*

loss�>)�װ       �	9Ҥ>fc�A�*

loss���<��h       �	�p�>fc�A�*

loss�h�<1��1       �	�>fc�A�*

loss�͛=�	-       �	Y��>fc�A�*

loss6�;�	Y�       �	5@�>fc�A�*

lossTMt<��M0       �	�ۧ>fc�A�*

loss*+4;3<[�       �	�r�>fc�A�*

lossm��=!�       �	=�>fc�A�*

lossZ�<��	       �	��>fc�A�*

loss��8=�]       �	
I�>fc�A�*

loss���<9��e       �	�ު>fc�A�*

loss�*�<��D'       �	Ox�>fc�A�*

loss�U�<d�P       �	��>fc�A�*

loss���;���       �	`��>fc�A�*

lossh�'=0�q-       �	�K�>fc�A�*

lossQ�;+K&p       �	��>fc�A�*

loss)��=r�       �	9~�>fc�A�*

loss�2�=;�=       �	��>fc�A�*

loss�Re=�(50       �	���>fc�A�*

loss��\<�حt       �	�C�>fc�A�*

loss���=	<CY       �	E�>fc�A�*

loss�)�=I�       �	���>fc�A�*

loss>@<w
��       �	�8�>fc�A�*

loss��<���       �	�β>fc�A�*

loss2U�<��N       �	'f�>fc�A�*

lossO}�<"���       �	
�>fc�A�*

lossO�\=I�
       �	���>fc�A�*

lossm�<�.��       �	�A�>fc�A�*

loss�� =���       �	Dݵ>fc�A�*

loss�(=,<�       �	*t�>fc�A�*

loss�B=�#$�       �	��>fc�A�*

lossѫ9=	�M�       �	���>fc�A�*

lossj��<	��       �	�3�>fc�A�*

lossL=�V"       �	�ĸ>fc�A�*

loss%2�<pؒ       �	�f�>fc�A�*

loss�		=ǉ�Q       �	C��>fc�A�*

loss��@=�W.       �	���>fc�A�*

loss�=j=��       �	�(�>fc�A�*

loss��I<��       �	Ի�>fc�A�*

loss��<�o}�       �	O�>fc�A�*

loss��=$.|\       �	i�>fc�A�*

loss��K=CG�       �	$��>fc�A�*

loss�:�<��       �	�>fc�A�*

loss�\�<V�_�       �	з�>fc�A�*

loss���;S��       �	N�>fc�A�*

loss���<|�s#       �	� �>fc�A�*

lossλ%=����       �	���>fc�A�*

loss��)<���       �	�:�>fc�A�*

loss��3=62C       �	=��>fc�A�*

loss�Α=�<q�       �	�g�>fc�A�*

lossO"�=r>�&       �	��>fc�A�*

loss��<�1�       �	8��>fc�A�*

loss���<T���       �	&8�>fc�A�*

lossc�=�2�       �	y��>fc�A�*

loss.0�<3��       �	�i�>fc�A�*

lossz^K=;���       �	��>fc�A�*

lossX�<I��       �	?��>fc�A�*

loss�<>�       �	?�>fc�A�*

loss��=�O�V       �	s��>fc�A�*

loss���;��]�       �	7l�>fc�A�*

loss�4F=N�_�       �	G �>fc�A�*

lossn{�<��q       �	��>fc�A�*

loss��<-H       �	�/�>fc�A�*

loss���;�ɇ        �	���>fc�A�*

loss�3<g�G       �	�_�>fc�A�*

lossMw�=�)3�       �	s��>fc�A�*

lossO�#<�f�!       �	d��>fc�A�*

loss��B=pK�r       �	v3�>fc�A�*

loss�ni=hx�       �	>��>fc�A�*

lossl̓<o�,\       �	�b�>fc�A�*

loss
�:� i�       �	s��>fc�A�*

loss�D<m�Y       �	���>fc�A�*

loss3D�<��|�       �	%�>fc�A�*

loss�
�;#��	       �	s��>fc�A�*

loss��I;CL�       �	�S�>fc�A�*

loss.#�9�
�       �	���>fc�A�*

loss4/2<hM��       �	�y�>fc�A�*

lossE�+<U�$M       �	��>fc�A�*

loss�;���)       �	Y��>fc�A�*

loss�$�:��}�       �	�E�>fc�A�*

loss4Z4<�+s�       �	���>fc�A�*

lossD[�<�JW�       �	D��>fc�A�*

lossq/<y��       �	��>fc�A�*

loss�:t|       �	���>fc�A�*

loss��<(��<       �	_F�>fc�A�*

loss��>���       �	@��>fc�A�*

loss�u�:N�}[       �	c{�>fc�A�*

loss�>8�%�       �	8�>fc�A�*

loss(�=l�*�       �	]��>fc�A�*

loss�h�=7��       �	�J�>fc�A�*

lossT�=$A�       �	@��>fc�A�*

loss�<Q���       �	ۢ�>fc�A�*

loss�}3=��8G       �	S>�>fc�A�*

loss<!=�[��       �	���>fc�A�*

loss���=?���       �	�o�>fc�A�*

loss���<�_H�       �	�>fc�A�*

loss���<t�t�       �	j��>fc�A�*

loss���=�z�g       �	u;�>fc�A�*

loss�=���       �	���>fc�A�*

loss߆�=�~Q       �	<��>fc�A�*

loss�cX=��^�       �	�L�>fc�A�*

loss�2:=RF��       �	��>fc�A�*

lossa�<��\�       �	_��>fc�A�*

loss�G�=�#(N       �	9G�>fc�A�*

loss��=\Z��       �	���>fc�A�*

loss%u�<o�r�       �	��>fc�A�*

loss!7�;�D�       �	�#�>fc�A�*

loss��<)'	�       �	���>fc�A�*

loss�<����       �	q��>fc�A�*

lossH�<���D       �	R`�>fc�A�*

lossy<��C[       �	���>fc�A�*

loss�<ΰ�       �	T��>fc�A�*

loss<#�;����       �	�k�>fc�A�*

loss�O<�[!q       �	� �>fc�A�*

loss}�==��       �	�7�>fc�A�*

loss��0=+Aj       �	|��>fc�A�*

loss1:�;��U       �	 ��>fc�A�*

loss��a=jZJ�       �	�)�>fc�A�*

loss���;��-(       �	"��>fc�A�*

loss��K<}֧y       �	�i�>fc�A�*

lossf��<��v       �	��>fc�A�*

loss���:C���       �	��>fc�A�*

loss�v�<�o       �	0G�>fc�A�*

loss��=4�O�       �	^��>fc�A�*

lossn�S=�4�       �	���>fc�A�*

loss`P�<�0�       �	 )�>fc�A�*

loss��o<����       �	3��>fc�A�*

losst�=O��"       �		��>fc�A�*

lossO�=T���       �	�H�>fc�A�*

loss�;��,�       �	)��>fc�A�*

loss��`;�ߘ�       �	g��>fc�A�*

loss\�=r�ؚ       �	�2�>fc�A�*

loss��J={	_       �	��>fc�A�*

loss���:���       �	 p�>fc�A�*

loss��c=Qa       �	O�>fc�A�*

loss{�<uDu�       �	��>fc�A�*

lossԂ<�M�       �	�=�>fc�A�*

lossm �<�U��       �	n�?fc�A�*

lossvy�=2$�       �	�"?fc�A�*

loss�{�<D���       �	��?fc�A�*

loss7��=T��@       �	;T?fc�A�*

lossF�=��5�       �	$�?fc�A�*

loss��/=�R��       �	
�?fc�A�*

loss�ݰ<AU�)       �	�&?fc�A�*

loss;I9=d�       �	��?fc�A�*

loss6�/=�=       �	 U?fc�A�*

loss�3�;FE�       �	��?fc�A�*

loss�N ;:��       �	<�?fc�A�*

loss]��<�6�_       �	�?fc�A�*

loss{�<�r�~       �	-�?fc�A�*

loss&q=�Nl       �	L?fc�A�*

loss�s=ܘ��       �	]�?fc�A�*

loss؟�=p@e�       �	�z?fc�A�*

loss�z.;ҥ#       �	�?fc�A�*

loss4�;�l��       �	Υ?fc�A�*

loss_5=d��       �	}>?fc�A�*

lossJ
�=)R       �	��?fc�A�*

loss;�d<]��       �	�r?fc�A�*

loss�G >�7�       �	=?fc�A�*

loss�<�9�       �	+�?fc�A�*

loss�٠=�N)�       �	> ?fc�A�*

lossx+i<أP       �	A� ?fc�A�*

loss�
<����       �	s�!?fc�A�*

loss�[�<��؊       �	n"?fc�A�*

lossE=P(!       �	�"?fc�A�*

loss�SZ=�v�C       �	NF#?fc�A�*

loss�7%<\���       �	3�#?fc�A�*

loss�=W�)       �	ǀ$?fc�A�*

loss#�;ط�       �	�"%?fc�A�*

loss�y�<��D       �	D�%?fc�A�*

loss���=�ҕe       �	F`&?fc�A�*

loss�(D<��       �	��&?fc�A�*

lossh�&=n��       �	�'?fc�A�*

lossF��;�^(�       �	�?(?fc�A�*

loss �"=ͻ�       �	��(?fc�A�*

lossd-M>��ŷ       �	�)?fc�A�*

loss�7�=<���       �	v*?fc�A�*

lossso<����       �	�*?fc�A�*

loss��M=�p�A       �	�P+?fc�A�*

lossc��<�r_%       �	p�+?fc�A�*

lossq��<�K��       �	3�,?fc�A�*

loss=;M=_�       �	&-?fc�A�*

loss��=�S�g       �	n�-?fc�A�*

lossq��<ђ,�       �	�a.?fc�A�*

losshc�<�"�       �	* /?fc�A�*

lossJ(�<�J%       �	c�/?fc�A�*

lossv�:�D?�       �	�00?fc�A�*

loss2C{;&R��       �	��0?fc�A�*

loss[�=���       �	=a1?fc�A�*

lossC�<��       �	��1?fc�A�*

loss�ʡ=�w��       �	7�2?fc�A�*

loss�<��E       �	?3?fc�A�*

loss�`�;4�>|       �	��3?fc�A�*

loss�~r;�qp�       �	*U4?fc�A�*

loss���;<-3B       �	j5?fc�A�*

loss6��<g�|Z       �	6?fc�A�*

loss��=W<�       �	�6?fc�A�*

loss�A�=���9       �	�X7?fc�A�*

loss���<u�       �	�7?fc�A�*

loss$5:<+:�       �	O�8?fc�A�*

loss���<����       �	�-9?fc�A�*

lossͭG<B�@       �	��9?fc�A�*

lossf��=:uW       �	ut:?fc�A�*

lossA}�<r3�       �	�;?fc�A�*

loss=3"=es       �	��;?fc�A�*

loss� =��Ta       �	�T<?fc�A�*

loss��=]�>�       �	P�<?fc�A�*

loss
k�<�E
       �	6?>?fc�A�*

loss���<u�
�       �	%�>?fc�A�*

loss/��<^h3        �	֏??fc�A�*

lossQ�e=�B@�       �	,.@?fc�A�*

loss�ޫ<o���       �	F�@?fc�A�*

loss�g<Թ��       �	��A?fc�A�*

loss�9�;���G       �	��B?fc�A�*

loss�@=�{��       �	�4C?fc�A�*

loss8x<ܺM�       �	,�C?fc�A�*

loss(i<|Zl       �	�pD?fc�A�*

loss?�*<����       �	 E?fc�A�*

loss<^=��       �	��E?fc�A�*

loss��<�H�\       �	�QF?fc�A�*

loss�F�9�P       �	��F?fc�A�*

loss�;�t6       �	R�G?fc�A�*

lossZ��;�f6       �	�H?fc�A�*

lossM=���       �	��H?fc�A�*

loss��<��       �	K?fc�A�*

loss�W�<#7�       �	aPL?fc�A�*

loss��*;Uf[       �	��M?fc�A�*

loss(1==�xu       �	�MN?fc�A�*

loss��=�*t       �	n�N?fc�A�*

loss���<����       �	��O?fc�A�*

loss}�<�$o       �	�P?fc�A�*

lossG�<��*�       �	ԛQ?fc�A�*

loss ٌ;��a�       �	�R?fc�A�*

loss���=ߧ�#       �	�nS?fc�A�*

loss�1�<nOKG       �	F�T?fc�A�*

loss��=zny       �	{NU?fc�A�*

loss�`�;pѐd       �	}AV?fc�A�*

loss3�L<�٢       �	��V?fc�A�*

loss�t�;�)�       �	�W?fc�A�*

loss���<덿�       �	T7X?fc�A�*

loss��;�d��       �	[?fc�A�*

lossq=In�       �	�[?fc�A�*

loss%�8=7G��       �	�L\?fc�A�*

lossM�:��       �	i�\?fc�A�*

loss8[�<F;8�       �	�|]?fc�A�*

loss�	>�Z�       �	^?fc�A�*

loss@`4=�5       �	~�^?fc�A�*

loss�5�=�a�       �	rQ_?fc�A�*

lossH�'=�}��       �	
�_?fc�A�*

loss�-=�-�       �	�`?fc�A�*

lossF�#<��d       �	�0a?fc�A�*

loss��)=R�5�       �	`�a?fc�A�*

loss�M�<p6�6       �	�ib?fc�A�*

lossQSV<��r�       �	�c?fc�A�*

loss�%Z=b>�       �	E�c?fc�A�*

lossT�=V��       �	 :d?fc�A�*

lossA߰=�E#       �	��d?fc�A�*

loss���<	�#       �	�e?fc�A�*

loss��=��d       �	((f?fc�A�*

loss�X=�]�       �	�f?fc�A�*

loss�e�<`;O�       �	�Xg?fc�A�*

loss�\<���       �	|�g?fc�A�*

loss�]=Ex�       �	��h?fc�A�*

loss0�=b�8#       �	�$i?fc�A�*

loss7�f<�Pŝ       �	�i?fc�A�*

loss�S=�f��       �	!Vj?fc�A�*

loss��=܂8�       �	(k?fc�A�*

loss���<�S5       �	��k?fc�A�*

loss�B�<�L��       �	�tl?fc�A�*

loss(i<���/       �	�m?fc�A�*

loss��x=���/       �	��m?fc�A�*

lossM��=��L�       �	fin?fc�A�*

lossh^/=��s�       �	o?fc�A�*

loss��=�~�       �	p?fc�A�*

loss���;e
"�       �	��p?fc�A�*

loss操<(��       �	�Uq?fc�A�*

loss�f;���       �	�q?fc�A�*

loss�];��       �	}�r?fc�A�*

lossQ�Z<'{b       �	�,s?fc�A�*

loss��=��       �	`�s?fc�A�*

loss�ѫ<C/4       �	"�t?fc�A�*

loss���;-�_       �	9Dv?fc�A�*

loss,�K=S8       �	��w?fc�A�*

lossĩY<��B       �	�1y?fc�A�*

loss,��<+�       �	y�y?fc�A�*

loss�>�;l �       �		pz?fc�A�*

loss���<_IS5       �	�{?fc�A�*

loss4ɉ<��       �	˹{?fc�A�*

lossH=�<����       �	�Y|?fc�A�*

loss-�=:l�       �	Z�|?fc�A�*

loss�Į<�D~;       �	�}?fc�A�*

loss�,=m;>       �	�<~?fc�A�*

loss�;Z���       �	��~?fc�A�*

loss��<d���       �	�z?fc�A�*

loss�:=��^       �	*�?fc�A�*

lossŖ�=�VF       �	���?fc�A�*

loss=<$m&�       �	V�?fc�A�*

loss���<�ɐC       �	{�?fc�A�*

loss��<�O�|       �	��?fc�A�*

lossob�<��5       �	�Q�?fc�A�*

losssG�;	p��       �	��?fc�A�*

loss�Ӥ=��K�       �	ɐ�?fc�A�*

loss�η;��-�       �	+N�?fc�A�*

lossԥ=Ƞ �       �	��?fc�A�*

lossA��=ى[       �	�?fc�A�*

loss�F<��*�       �	�@�?fc�A�*

loss�+�<� s2       �	'ڇ?fc�A�*

losse'#=���?       �	x�?fc�A�*

loss��#=^{       �	q!�?fc�A�*

loss-�;\u�       �	zĉ?fc�A�*

loss�b{=��A       �	5b�?fc�A�*

loss�=�(��       �	��?fc�A�*

lossx�R=�Ճ�       �	'��?fc�A�*

loss���<>�8       �	NA�?fc�A�*

losss�;#���       �	F
�?fc�A�*

lossjjK=��        �	9��?fc�A�*

loss �<>URe       �	�|�?fc�A�*

loss��<�?u       �	�s�?fc�A�*

loss#��=-L @       �	��?fc�A�*

losscf�<��       �	\��?fc�A�*

loss�+^<	���       �	�D�?fc�A�*

lossYG�=�\�)       �	�ב?fc�A�*

loss�6<�\�~       �	�n�?fc�A�*

loss�&1=�@I�       �	O�?fc�A�*

loss��<�+H       �	c��?fc�A�*

loss�:=��-�       �	'.�?fc�A�*

lossRU<���y       �	�Ô?fc�A�*

lossX
�=��<�       �	Wx�?fc�A�*

loss�kd<B��R       �	��?fc�A�*

lossϏ�<"�       �	ٵ�?fc�A�*

losss��;v��       �	g�?fc�A�*

lossl�><@�R;       �	&��?fc�A�*

loss2�<�	�       �	h��?fc�A�*

lossVF�=�n��       �	�/�?fc�A�*

loss�Xk=�K��       �	Й?fc�A�*

loss�ȳ=\x �       �	{g�?fc�A�*

loss�;�<�q�       �	�?fc�A�*

loss ��<[�J       �	��?fc�A�*

loss��=�<��       �	hA�?fc�A�*

loss�M<q{��       �	;�?fc�A�*

lossz�=dA�       �	j��?fc�A�*

loss��<�7~       �	%"�?fc�A�*

lossw=E:�8       �	�?fc�A�*

loss:iQ=�W�       �	�a�?fc�A�*

loss@?`=��^       �	��?fc�A�*

loss�\�<}�|�       �	נ�?fc�A�*

loss,FS<���o       �	�@�?fc�A�*

loss�З=4��       �	aݡ?fc�A�*

losst�<�z�       �	�|�?fc�A�*

loss�Gg<n�:       �	��?fc�A�*

losssQC=�{�       �	���?fc�A�*

lossys�<Jyh�       �	�n�?fc�A�*

loss��= a�       �	O�?fc�A�*

loss��=�?"       �	Xɥ?fc�A�*

loss�kG>��{�       �	�V�?fc�A�*

loss�A�=�-|Y       �	U��?fc�A�*

loss��|=�z�       �	���?fc�A�*

loss�1<=M3;       �	�G�?fc�A�*

loss/��;��v�       �	!��?fc�A�*

lossoh=���       �	O�?fc�A�*

loss�a=�#u�       �	E�?fc�A�*

loss��%<�5�       �	��?fc�A�*

lossS�<��^       �	+4�?fc�A�*

loss��<C9�       �	�а?fc�A�*

loss��<$
�       �	&r�?fc�A�*

loss��G=w��G       �	�?fc�A�*

loss*4<ĵ�       �	��?fc�A�*

loss�P�<�r�8       �	/R�?fc�A�*

lossʛ<�7�
       �	��?fc�A�*

loss�<;�       �	��?fc�A�*

loss�<��       �	�"�?fc�A�*

loss�Ge<��<�       �	b��?fc�A�*

loss
8=K���       �	%[�?fc�A�*

loss�]<���7       �	j��?fc�A�*

loss�)=�$��       �	w��?fc�A�*

loss�X=�r�+       �	�D�?fc�A�*

loss��;ꈭ-       �	��?fc�A�*

loss��<{���       �	��?fc�A�*

loss�ٯ=����       �	�*�?fc�A�*

loss�ձ<	�$       �	�κ?fc�A�*

lossȈ*=�)�       �	uw�?fc�A�*

lossWǅ=���       �	��?fc�A�*

lossan <�⎽       �	*��?fc�A�*

loss��e<�>�       �	�M�?fc�A�*

loss��&=s^`       �	���?fc�A�*

loss�8�<1/J       �	S��?fc�A�*

lossvg<l��       �	&Q�?fc�A�*

loss�߭<-w��       �	���?fc�A�*

lossTx,=��-       �	Ή�?fc�A�*

loss�<�<�rU       �	-%�?fc�A�*

loss��G;�/��       �	���?fc�A�*

loss��;$!�~       �	�s�?fc�A�*

loss��^;r$°       �	��?fc�A�*

loss,�<Q�p       �	\��?fc�A�*

lossv{-<X��K       �	��?fc�A�*

losslW;���       �	 �?fc�A�*

loss��<hM�P       �		��?fc�A�*

loss��8=���       �	"��?fc�A�*

lossgD�;����       �	�3�?fc�A�*

loss���<�ܕ       �	4��?fc�A�*

lossfQ�=��yd       �	_~�?fc�A�*

loss}�=�~�v       �	� �?fc�A�*

lossC2<���       �	��?fc�A�*

loss�=��U&       �	�q�?fc�A�*

loss�w�<D!=�       �	��?fc�A�*

lossxN=�ާ       �	���?fc�A�*

loss��4=`\T       �	��?fc�A�*

lossq(�<��mb       �	T��?fc�A�*

loss=�<���       �	�,�?fc�A�*

loss�Y<�S       �	M�?fc�A�*

loss���=�HW       �	��?fc�A�*

loss��=��8�       �	q��?fc�A�*

loss?=�os�       �	���?fc�A�*

loss7�<�&R       �	��?fc�A�*

loss�W�;bIy       �	���?fc�A�*

loss�o�=N�1       �	�E�?fc�A�*

losslT#=�J�       �	4�?fc�A�*

loss`�\;��ŵ       �	���?fc�A�*

losso-!={_�       �	�F�?fc�A�*

loss	��<���       �	���?fc�A�*

loss���<m6�E       �	t�?fc�A�*

loss3�<��c/       �	��?fc�A�*

loss�ݯ=�dE�       �	��?fc�A�*

loss�x=�^��       �	"6�?fc�A�*

loss�p�;W��       �	���?fc�A�*

lossq�;�Q}+       �	�|�?fc�A�*

loss��<�X-       �	��?fc�A�*

loss�<���       �	T��?fc�A�*

loss�h�<�j�       �	5A�?fc�A�*

loss�7:�$}O       �	0��?fc�A�*

lossM�7;��Վ       �	it�?fc�A�*

losso�;���)       �	��?fc�A�*

loss��=l�CS       �	
��?fc�A�*

loss�iK<�)       �	z4�?fc�A�*

loss�<��K       �	C��?fc�A�*

loss���=�q�       �	b�?fc�A�*

loss׷b=����       �	'��?fc�A�*

lossa�<��V
       �	Փ�?fc�A�*

loss�gv< ���       �	g(�?fc�A�*

loss�;>���       �	��?fc�A�*

loss��<ܩ2       �	�Q�?fc�A�*

loss�%[=�SV�       �	i�?fc�A�*

loss��>��\       �	���?fc�A�*

loss�G)<,p��       �	/�?fc�A�*

loss�_�=�P"6       �	 ��?fc�A�*

lossf�W<�3�       �	"o�?fc�A�*

loss���;��U�       �	��?fc�A�*

losse�<+L�       �	v��?fc�A�*

lossA�<G�R       �	�A�?fc�A�*

lossHF==Gf�       �	���?fc�A�*

losse\F=kw*�       �	�u�?fc�A�*

loss|��;He       �	��?fc�A�*

losss�=o+��       �	W��?fc�A�*

loss|HA;�<�       �	�L�?fc�A�*

loss��?=L�2�       �	t��?fc�A�*

loss�.�=����       �	���?fc�A�*

lossb��;Q��J       �	*�?fc�A�*

loss1��;v��       �	���?fc�A�*

loss<J�=��oL       �	�k�?fc�A�*

loss��<b��       �	p
�?fc�A�*

loss���:��Re       �	��?fc�A�*

loss��+=~��u       �	���?fc�A�*

loss�q�=�j�       �	:x�?fc�A�*

loss��{=M�<�       �	��?fc�A�*

loss� �<~y       �	a��?fc�A�*

lossS��<�k�-       �	�?�?fc�A�*

lossA�=_c�       �	2��?fc�A�*

loss�� =Ⱥ��       �	�~�?fc�A�*

loss3�_=�a�       �	�?fc�A�*

loss�+/<�r%O       �	~��?fc�A�*

loss.�<&��#       �	�@�?fc�A�*

loss�{�=̅@�       �	���?fc�A�*

loss�<+]x       �	�h�?fc�A�*

loss��=�9��       �	Y��?fc�A�*

loss�e�<��>       �	y��?fc�A�*

loss�p�<J�QL       �	�=�?fc�A�*

loss?<ab       �	��?fc�A�*

lossל=�-B       �	�g�?fc�A�*

loss�1<���       �	��?fc�A�*

lossE�L<�:��       �	4�?fc�A�*

loss)��<kX       �	���?fc�A�*

loss:y�=���G       �	��?fc�A�*

loss�.�=���       �	C�?fc�A�*

loss�$=,�Q�       �	f��?fc�A�*

loss���=[�M(       �	X @fc�A�*

loss4 ==r��       �	�� @fc�A�*

loss)�<㵬�       �	ݕ@fc�A�*

loss#4E<2Ia       �	�1@fc�A�*

lossv!�<1%d�       �	��@fc�A�*

loss��=�7�       �	�h@fc�A�*

loss
�o<�RmQ       �	�2@fc�A�*

loss[qE=Pd4       �	�@fc�A�*

loss�5�<��P�       �	�o@fc�A�*

lossf�</<       �	�@fc�A�*

lossX3�<�6B       �	#g@fc�A�*

loss���<�w�       �	�	@fc�A�*

loss�.�<��       �	�	@fc�A�*

loss7=2�S�       �	�j
@fc�A�*

loss�p=��E       �	k@fc�A�*

loss4n&<Ne
       �	ݳ@fc�A�*

loss�D=�M��       �	�W@fc�A�*

loss�j�=�PYI       �	�>@fc�A�*

loss�F�<�n�Z       �	��@fc�A�*

loss�	=�@T       �	�q@fc�A�*

loss�Z<'��       �	�g@fc�A�*

loss�e=��?       �	�@fc�A�*

loss{R<H���       �	ܡ@fc�A�*

lossn�T=�1       �	�7@fc�A�*

loss���;x��S       �	O�@fc�A�*

loss ��<gb��       �	g@fc�A�*

loss<V�,       �	�@fc�A�*

loss� �=+��,       �	��@fc�A�*

lossfyB=iGO       �	&9@fc�A�*

lossS�=�j��       �	��@fc�A�*

lossv�=j_       �	�s@fc�A�*

loss���<���       �	�@fc�A�*

loss�4>eJI�       �	o�@fc�A�*

loss�E<�K�=       �	�'@fc�A�*

loss��<l
       �	��@fc�A�*

loss]/�=e
��       �	�y@fc�A�*

lossf
�;@Z2�       �	!@fc�A�*

loss ��<��܎       �	��@fc�A�*

loss��k=7��       �	|�@fc�A�*

loss�,�<�P�       �	,@fc�A�*

loss_o^=��)�       �	I�@fc�A�*

lossʽ�<�z��       �	c|@fc�A�*

lossɹ�;#nt�       �	�* @fc�A�*

loss'=Di       �	� @fc�A�*

loss�#==��       �	�t!@fc�A�*

loss�a{<��R�       �	�"@fc�A�*

loss�z=�d
       �	�"@fc�A�*

lossz_�<#g�       �	�#@fc�A�*

loss�=̄^X       �	�3%@fc�A�*

losstE�;��       �	��%@fc�A�*

loss��2<�Y�
       �	-`(@fc�A�*

lossC ]<%k       �	F)@fc�A�*

loss+�!=�Ҹ:       �	�0*@fc�A�*

loss�W<�\�       �	��*@fc�A�*

loss��R<K�w       �	�v+@fc�A�*

loss�d�=�G�       �	�,@fc�A�*

loss���;w�#u       �	��,@fc�A�*

loss8�9=��wb       �	@K-@fc�A�*

lossưQ=Q��       �	&�-@fc�A�*

losssH�=n��       �	ǂ.@fc�A�*

loss|]h<��8       �	�%1@fc�A�*

loss��:<|Q�I       �	s�1@fc�A�*

loss��<��3       �	�Z2@fc�A�*

loss��e<<�f�       �	73@fc�A�*

lossr�=�P��       �	��3@fc�A�*

loss�&<�EH�       �	S^4@fc�A�*

loss.�<M��       �	�5@fc�A�*

losss\�<�$�r       �	��5@fc�A�*

losst�W<D6�       �	�i6@fc�A�*

lossڝ<�Xr�       �	�7@fc�A�*

loss0�<�ȈI       �	��7@fc�A�*

loss8�3=�t>�       �	h\8@fc�A�*

losstA�=��       �	&�8@fc�A�*

loss���;+貎       �	4�9@fc�A�*

loss�s�;~uq       �	�G:@fc�A�*

loss�Ew=��C       �	��:@fc�A�*

loss�J=��8       �	ޫ;@fc�A�*

loss�[<B���       �	�P<@fc�A�*

loss�GJ=�>�       �	��<@fc�A�*

loss<6,��       �	Ę=@fc�A�*

loss[PL<�,�       �	6;>@fc�A�*

lossx�S<�PXm       �	^�>@fc�A�*

lossS�!<����       �	hu?@fc�A�*

loss{�==wj�k       �	�@@fc�A�*

loss*�*=C��       �	6�@@fc�A�*

loss�E�<>�H       �	ZdA@fc�A�*

loss'�;��{       �	�B@fc�A�*

losscE{=�v�@       �	��B@fc�A�*

loss���<�(       �	KC@fc�A�*

loss���;��       �	�C@fc�A�*

loss�f�;�y�       �	o~D@fc�A�*

loss���;�*Է       �	�E@fc�A�*

loss!H<Q�8       �	�F@fc�A�*

loss!~=���[       �	`�F@fc�A�*

lossjb1=8D       �	МG@fc�A�*

loss�+=gu�~       �	�=H@fc�A�*

lossW
�=&@��       �	BAI@fc�A�*

loss;F�;繲&       �	�I@fc�A�*

loss�}h<L0i       �	؁J@fc�A�*

lossl�=;�nw       �	�$K@fc�A�*

loss6�o=��2�       �	u�K@fc�A�*

loss4�<�d       �	lL@fc�A�*

lossOs=��s       �	�M@fc�A�*

loss��<��<�       �	��M@fc�A�*

loss��;@�       �	/�N@fc�A�*

loss�`�;�v�y       �	��O@fc�A�*

loss!�^=�ó       �	�uP@fc�A�*

loss�Q==a�\       �	�"Q@fc�A�*

lossa�I=}�_       �	��Q@fc�A�*

loss�+=���       �	�R@fc�A�*

loss�]=���       �	�%S@fc�A�*

loss��<�	w�       �	��S@fc�A�*

loss(D=�3fN       �	�tT@fc�A�*

lossʕ=�!��       �	^U@fc�A�*

loss�]"<4�Ц       �	o�U@fc�A�*

loss:^�;7��n       �	�[V@fc�A�*

loss	�>���       �	�V@fc�A�*

loss�/<'#l�       �	I�W@fc�A�*

loss浵;��(       �	�;X@fc�A�*

lossWܦ=x3�        �	��X@fc�A�*

loss��</ o�       �	�xY@fc�A�*

loss��\=�Ap       �	�Z@fc�A�*

lossiM�:����       �	��Z@fc�A�*

loss1z�=��       �	bK[@fc�A�*

lossQ�T<�yo�       �	��[@fc�A�*

loss�Gz=��       �	�|\@fc�A�*

loss�� >��f       �	U]@fc�A�*

loss�(<?��C       �	��]@fc�A�*

lossD�(<�G�       �	^M^@fc�A�*

loss��<����       �	��_@fc�A�*

loss=oҼ       �	t�`@fc�A�*

lossw�<��+A       �	Z/a@fc�A�*

lossd�=�+~       �	$�a@fc�A�*

losss��=.���       �	pb@fc�A�*

loss�y=��p       �	c@fc�A�*

lossA��<�eF�       �	��c@fc�A�*

loss��=��&	       �	(}d@fc�A�*

loss�χ=��\R       �	�e@fc�A�*

loss\�:<Q���       �	��e@fc�A�*

lossT�K=�r�       �	�Vf@fc�A�*

loss마=q       �	��f@fc�A�*

loss��:<�t]�       �	�g@fc�A�*

lossS��<�f�       �	�5h@fc�A�*

lossd�"=m^�o       �	�h@fc�A�*

loss|&+=h�^.       �	>zi@fc�A�*

lossM(�;b���       �	S j@fc�A�*

loss��D=���1       �	��j@fc�A�*

loss�i<�I�       �	Кk@fc�A�*

loss�֏<\/3�       �	-@l@fc�A�*

loss�Ұ<��Ν       �	��l@fc�A�*

loss�c<tS~       �	�m@fc�A�*

loss��*=����       �	PUn@fc�A�*

loss��7=SO�       �	P o@fc�A�*

loss�]=X!��       �	�jp@fc�A�*

loss�A<OXP       �	�q@fc�A�*

losst��;��H       �	�gr@fc�A�*

loss:nn<e#�       �	�s@fc�A�*

loss�ZX=��4p       �	;�s@fc�A�*

loss�ݓ<M3�        �	��t@fc�A�*

loss�̏<���J       �	�#u@fc�A�*

loss.�<��       �	Z�u@fc�A�*

lossI��<?h�6       �	HRv@fc�A�*

loss_ٹ<E�b�       �	��v@fc�A�*

loss�n=4��       �	��x@fc�A�*

lossl�q<��       �	�gy@fc�A�*

lossZ�i<��J       �	'Jz@fc�A�*

loss�_�<��F       �	R�z@fc�A�*

losspo<.�~�       �	J
|@fc�A�*

losso�o=��=�       �	��|@fc�A�*

loss��=&̰       �	�~}@fc�A�*

loss�a�;��H�       �	�~@fc�A�*

loss��<Y�b�       �	�;@fc�A�*

lossW(<���       �	��@fc�A�*

loss/� <K&)�       �	�y�@fc�A�*

loss/r�<M,�       �	w�@fc�A�*

loss���;���       �	���@fc�A�*

loss@��<���       �	;U�@fc�A�*

loss�0w=^��(       �	�'�@fc�A�*

lossī�<��8�       �	h˃@fc�A�*

lossR�;y�:�       �	fh�@fc�A�*

loss��;B�#       �	B	�@fc�A�*

loss���=rMf7       �	���@fc�A�*

loss�HQ<W���       �	�D�@fc�A�*

loss�yO<�vb�       �	��@fc�A�*

loss��<�!�       �	ԁ�@fc�A�*

loss� j;��       �	��@fc�A�*

loss�)(:'7L!       �	���@fc�A�*

loss˥;���       �	W�@fc�A�*

loss ��;����       �	t�@fc�A�*

loss�?x;���       �	��@fc�A�*

loss���<�|��       �	-"�@fc�A�*

loss��m;�       �	���@fc�A�*

loss�k?;I��       �	3O�@fc�A�*

lossQv7:��       �	��@fc�A�*

loss��*:����       �	~�@fc�A�*

loss�	�9`�]!       �	��@fc�A�*

lossB�:Fh�R       �	�׎@fc�A�*

loss1�h<��l�       �	�Ï@fc�A�*

loss�cE<~�       �	�l�@fc�A�*

lossZ�9�J�       �	�M�@fc�A�*

loss��%<�v,�       �	��@fc�A�*

loss*�O>s	��       �	�b�@fc�A�*

loss�M<'�#�       �	;��@fc�A�*

loss�md>�B        �	��@fc�A�*

loss>�<��H       �	�J�@fc�A�*

loss��=�.�       �	%�@fc�A�*

loss�l=+�IK       �	��@fc�A�*

lossZ;v�       �	��@fc�A�*

loss���=��y$       �	[��@fc�A�*

lossG�=Z�ĳ       �	�L�@fc�A�*

loss:G=�	T�       �	2�@fc�A�*

lossڈ�<�y]l       �	�|�@fc�A�*

loss&�<o��       �	F[�@fc�A�*

lossČ=1|zi       �	���@fc�A�*

loss�0�<���       �	��@fc�A�*

loss���<���       �	s-�@fc�A�*

lossq��<��.       �	�Ɯ@fc�A�*

loss��3=�DN�       �	�[�@fc�A�*

losss�=n�v       �	m�@fc�A�*

loss܉b<�2��       �	�T�@fc�A�*

loss]xZ=7�#       �	N�@fc�A�*

lossTD�<�>       �	<��@fc�A�*

loss�q�;}�       �	6!�@fc�A�*

loss)1�<\b�       �	��@fc�A�*

lossH�y<u�2�       �	�M�@fc�A�*

loss��Z:�E�o       �	2�@fc�A�*

loss��<��-       �	�}�@fc�A�*

losso��;
?#�       �	��@fc�A�*

loss!�u<w�\l       �	���@fc�A�*

loss�8N=�>       �	DN�@fc�A�*

loss�ػ<~45       �	\�@fc�A�*

loss���=t��C       �	@��@fc�A�*

loss;<�m       �	#0�@fc�A�*

loss�t�<�L"z       �	qȧ@fc�A�*

loss]��<�xɁ       �	�e�@fc�A�*

loss���<r�?g       �	��@fc�A�*

loss'�=�7o�       �	[��@fc�A�*

loss��<
��B       �	�Q�@fc�A�*

loss���<9�k       �	d�@fc�A�*

lossL�<c�5�       �	}��@fc�A�*

loss��=nE�       �	kI�@fc�A�*

loss�2�<���       �	��@fc�A�*

loss�JQ<��       �	ǀ�@fc�A�*

loss\<���       �	�&�@fc�A�*

losst�<��A�       �	3î@fc�A�*

loss�(R<���       �	-\�@fc�A�*

loss��<�V       �	U��@fc�A�*

loss�9B=����       �	���@fc�A�*

loss�&�=.��g       �	T5�@fc�A�*

loss�w/<�̊k       �	�ѱ@fc�A�*

loss��=[��5       �	@j�@fc�A�*

loss�ڇ<��9q       �	6�@fc�A�*

loss���<W��L       �	B��@fc�A�*

loss��;�m �       �	2�@fc�A�*

lossx��<����       �	[��@fc�A�*

lossȇG=��8       �	�I�@fc�A�*

loss�+<X�       �	��@fc�A�*

loss� H=�\�       �	p}�@fc�A�*

loss��;{�ԩ       �	J��@fc�A�*

lossm~K=8��)       �	��@fc�A�*

loss�2=/�v       �	��@fc�A�*

loss\�=;Z��       �	 '�@fc�A�*

loss���=p��       �	4��@fc�A�*

loss�]�;���o       �	b�@fc�A�*

loss�Z{<P���       �	��@fc�A�*

loss���=�|       �	��@fc�A�*

loss* �;�'�(       �	-�@fc�A�*

loss��<�_�       �	Y��@fc�A�*

loss%��=Nu�x       �	W|�@fc�A�*

loss���:�#��       �	w�@fc�A�*

loss�r�<�j'�       �	��@fc�A�*

loss�n<x��N       �	���@fc�A�*

lossw_�<n�C�       �	��@fc�A�*

loss�/<�ǝ       �	�e�@fc�A�*

loss��=v�3�       �	T�@fc�A�*

loss�e�;���I       �	���@fc�A�*

loss	ۆ=;�H�       �	�B�@fc�A�*

loss�=O=�       �	]��@fc�A�*

loss�<
^��       �	��@fc�A�*

loss`��<�i        �	e�@fc�A�*

loss�e�;&A �       �	���@fc�A�*

loss=�<���       �	I-�@fc�A�*

lossƈ=΋�       �	���@fc�A�*

loss��< L��       �	Xp�@fc�A�*

loss��;"�"       �	@�@fc�A�*

loss�.�:�zM       �	T��@fc�A�*

loss��<Ċy       �	߉�@fc�A�*

loss2�b<�EN       �	�*�@fc�A�*

lossa�<\ ��       �	��@fc�A�*

loss.GK;-���       �	�u�@fc�A�*

loss�q�<��       �	D�@fc�A�*

loss#ݍ=՜��       �	!��@fc�A�*

loss@ͽ<U���       �	�R�@fc�A�*

loss�_<8�j�       �	���@fc�A�*

loss;��<Ue!       �	ƥ�@fc�A�*

loss��;:�#�       �	M�@fc�A�*

loss�z�<D�$W       �	o��@fc�A�*

lossg�;w�D       �	��@fc�A�*

lossvm�=��H       �	�2�@fc�A�*

loss\�;н?�       �	��@fc�A�*

lossV�=�&�x       �	jm�@fc�A�*

lossȽ�:&�W       �	��@fc�A�*

losszi�;^Ż�       �	��@fc�A�*

loss$�;	P�       �	2=�@fc�A�*

loss1S�<K��       �	b��@fc�A�*

loss%2<�� �       �	�q�@fc�A�*

loss�K�=af�j       �	�@fc�A�*

loss�<[�<�       �	��@fc�A�*

loss���;�sg       �	�\�@fc�A�*

loss�wZ;�
�       �	��@fc�A�*

loss�H�:8��#       �	V��@fc�A�*

lossr|K;�'Ũ       �	�:�@fc�A�*

loss�;=UE��       �	Z��@fc�A�*

loss�)>�J$�       �	5{�@fc�A�*

loss�?;��       �	4�@fc�A�*

loss��0=��I�       �	+��@fc�A�*

loss,z�<��B�       �	oF�@fc�A�*

loss�<:��\       �	)��@fc�A�*

lossY�;,�v       �	�@fc�A�*

loss��=b�,�       �	Y�@fc�A�*

loss���;��΍       �	!��@fc�A�*

loss�W1=�;       �	�F�@fc�A�*

lossh< 9�r       �	���@fc�A�*

loss_l�;C��9       �	�n�@fc�A�*

loss�u<��<�       �	��@fc�A�*

loss�<_���       �	^��@fc�A�*

loss$�b=��r�       �	�0�@fc�A�*

loss���<}4l�       �	���@fc�A�*

lossѴ�<a���       �	>Z�@fc�A�*

lossZ�<"@�       �	���@fc�A�*

loss�8�:qPm       �	��@fc�A�*

loss�i-;��       �	X7 Afc�A�*

lossj#D=�rA�       �	�� Afc�A�*

lossQL=blb�       �	�]Afc�A�*

loss��<�Ө�       �	8�Afc�A�*

loss}�z<:��       �	��Afc�A�*

lossc�= Zļ       �	�HAfc�A�*

losso�5<��Y
       �	(�Afc�A�*

loss)�m<lod�       �	U�Afc�A�*

loss�1-;�(�r       �	}"Afc�A�*

loss$C<�H�"       �	E�Afc�A�*

loss�'�<ふ       �	8�Afc�A�*

loss�=uC        �	Afc�A�*

lossT)=M9��       �	s�Afc�A�*

lossظY<�yFu       �	aRAfc�A�*

loss{e�<�/�       �	��Afc�A�*

loss"<Z��       �	��	Afc�A�*

loss�+=�       �	�
Afc�A�*

loss��D;�`       �	]�
Afc�A�*

lossǲ=�'fP       �	�zAfc�A�*

loss���:lԂ}       �	�Afc�A�*

loss� <��S       �	��Afc�A�*

loss`�I<=���       �	
MAfc�A�*

loss:�l;�Q/       �	Afc�A�*

loss��;����       �	�Afc�A�*

lossE�=��g       �	S�Afc�A�*

loss)�<���       �	�SAfc�A�*

loss[�.=�i�(       �	��Afc�A�*

loss�4=����       �	��Afc�A�*

loss�N�<0ͥ�       �	�AAfc�A�*

loss�;<��2�       �	��Afc�A�*

loss�|�<%S
       �	�Afc�A�*

lossM�1<��ۄ       �	�Afc�A�*

loss��<�n��       �	f�Afc�A�*

lossX��<(�I       �	�VAfc�A�*

loss.�<�ix       �	�Afc�A�*

loss�g$;e#B.       �	�Afc�A�*

lossj�<�#�       �	�>Afc�A�*

lossr��<'��D       �	��Afc�A�*

loss2c�<|�Xc       �	�uAfc�A�*

loss���<H�_�       �	�Afc�A�*

loss�v<���       �	��Afc�A�*

loss��;	_s?       �	CSAfc�A�*

loss_��;�\e       �	��Afc�A�*

loss�ħ=���       �	K�Afc�A�*

loss
'==C4�       �	܄Afc�A�*

loss�F=�4�A       �	5%Afc�A�*

losst�={ݛ0       �	A�Afc�A�*

loss�{h<w���       �	UAfc�A�*

loss�B'<�QZ�       �	k�Afc�A�*

loss6F�<j�S�       �	@�Afc�A�*

loss���<	t��       �	>  Afc�A�*

loss\� <��[�       �	Q� Afc�A�*

loss�a�;���       �	_\!Afc�A�*

loss�=/�R       �	��!Afc�A�*

lossX��=F9��       �	�"Afc�A�*

loss�M�<-/��       �	�+#Afc�A�*

loss�ah=E���       �	S�#Afc�A�*

loss��+<�mL       �	ji$Afc�A�*

loss�<�<�P�h       �	�%Afc�A�*

losst��;ПH*       �	�%Afc�A�*

lossڼ�<��:�       �	Bv&Afc�A�*

loss)};�:G�       �	3'Afc�A�*

loss\�<�*z       �	�'Afc�A�*

loss]��<�3[V       �	ǻ(Afc�A�*

loss�^=��s�       �	 ^)Afc�A�*

loss̈́=oΨ�       �	��)Afc�A�*

lossM��=�ә       �	��*Afc�A�*

loss�`�<V�>       �	P8+Afc�A�*

loss@[K<� G�       �	A�+Afc�A�*

loss�E\;�4��       �	mq,Afc�A�*

loss-O=A
3�       �	�	-Afc�A�*

lossA&<��`       �	'�-Afc�A�*

loss�'�;λ=�       �	�<.Afc�A�*

loss�Y�<�D"�       �	�.Afc�A�*

loss��=	�       �	Yi/Afc�A�*

loss݊=� ��       �	�0Afc�A�*

lossxt�<��       �	��0Afc�A�*

loss�D�9�#>�       �	PT1Afc�A�*

loss�ٴ:-�1       �	s�1Afc�A�*

loss}mf<�y=�       �	ٓ2Afc�A�*

loss�Kl<cZ|       �	�/3Afc�A�*

loss�`�<�l       �	��3Afc�A�*

lossu�=EFi�       �	g4Afc�A�*

loss.�Y<�D��       �	�	5Afc�A�*

loss�zc<�|�w       �	��5Afc�A�*

lossؐ�;�O       �	�>6Afc�A�*

loss�>�=AD]�       �	��6Afc�A�*

lossb�<pl�H       �	3p7Afc�A�*

loss��N<��.       �	�8Afc�A�*

loss�ɕ;���       �	�8Afc�A�*

loss��,=���       �	$+9Afc�A�*

loss7��;� G       �	b�9Afc�A�*

loss��s=���       �	KW:Afc�A�*

loss��s;�f�       �	>�:Afc�A�*

loss�!<t{	       �	Ȕ;Afc�A�*

loss��s=Y��H       �	-<Afc�A�*

loss=�<�\]�       �	�<Afc�A�*

loss�j^<s��Z       �	�r=Afc�A�*

lossU=���       �	/3>Afc�A�*

loss��=��J       �	��>Afc�A�*

loss�	<4��       �	`r?Afc�A�*

lossh� ;����       �	�#@Afc�A�*

loss��:Ƌ�z       �	��@Afc�A�*

loss2�<6��@       �	�`AAfc�A�*

loss@*�<��Y�       �	��AAfc�A�*

lossD&�;yH       �	X�BAfc�A�*

lossq�;zא�       �	,CAfc�A�*

loss�;�;
���       �	4�CAfc�A�*

loss���:˥i�       �	.rDAfc�A�*

lossAH�;��       �	�EAfc�A�*

loss}�;�<�       �	ƤEAfc�A�*

loss}��<�V�       �	<FAfc�A�*

loss���<�7(�       �	o�FAfc�A�*

losst�;��9�       �	 pGAfc�A�*

loss-*�;uR�       �	5HAfc�A�*

loss��W<���?       �	H�HAfc�A�*

loss8E<G9       �	9IAfc�A�*

loss���:��]�       �	s�IAfc�A�*

lossR��;��G       �	�rJAfc�A�*

lossl��<ZH��       �	�KAfc�A�*

loss,4�;���1       �	� LAfc�A�*

loss��;>�M�       �	��LAfc�A�*

loss��(=?�       �	�\MAfc�A�*

loss!�<���       �	,�MAfc�A�*

loss���;���       �	OAfc�A�*

loss���<R@#       �	j�OAfc�A�*

loss�E;E�k�       �	xFPAfc�A�*

loss�/$=?��       �	�PAfc�A�*

loss��G=Ic�*       �	�sQAfc�A�*

loss(�0=��d�       �	ZRAfc�A�*

lossdB,<�m       �	@�RAfc�A�*

loss��x<���       �	�>SAfc�A�*

lossc<���%       �	��SAfc�A�*

loss��,;vю       �	FxTAfc�A�*

loss"h�<<�O�       �	\UAfc�A�*

loss�I�;�#��       �	J�UAfc�A�*

lossM��<z�u_       �	�JVAfc�A�*

loss0�<�҃/       �	��VAfc�A�*

loss�_>�.��       �	�wWAfc�A�*

lossUJ=x�߬       �	�XAfc�A�*

loss� >���       �	X�XAfc�A�*

loss��<�̒�       �	�>YAfc�A�*

loss
�<��F       �	�LZAfc�A�*

loss��<��D       �	��ZAfc�A�*

loss��K=�z�       �	�[Afc�A�*

losss�e<���p       �	�:\Afc�A�*

loss�z:~��       �	w�\Afc�A�*

loss8Zm=LBH�       �	~]Afc�A�*

loss/�=S'E�       �	 %^Afc�A�*

loss�,=�       �	��^Afc�A�*

loss��=��t�       �	�\_Afc�A�*

loss��A<��       �	b�_Afc�A�*

loss@nd;�$^       �	6�`Afc�A�*

loss�4�:V���       �	B'aAfc�A�*

loss7J�=���w       �	_�aAfc�A�*

loss��p<݈gi       �	ObAfc�A�*

loss�]�<<#J�       �	\�bAfc�A�*

lossQ�L=�v�H       �	�xcAfc�A�*

loss�Z�<���       �	�dAfc�A�*

lossL=��;�       �	H�dAfc�A�*

loss��=�c�N       �	�seAfc�A�*

loss��B;xZ�%       �	�fAfc�A�*

loss�i�<����       �	q�fAfc�A�*

lossm
�<K�j�       �	�BgAfc�A�*

loss+G�<��F       �	c�gAfc�A�*

loss��=�bJ3       �	��hAfc�A�*

lossS��<�Y3        �	J%iAfc�A�*

lossi�o=�ϙ       �	��iAfc�A�*

loss!v!<�*�        �	�`jAfc�A�*

loss�r�<�$3$       �	$kAfc�A�*

loss�<磫�       �	v�kAfc�A�*

loss�Ϲ;+��3       �	DlAfc�A�*

loss��o=GU��       �	;�lAfc�A�*

loss�ؕ=�.V       �	�mAfc�A�*

loss��%=��%�       �	�.nAfc�A�*

loss��M;7���       �	��nAfc�A�*

loss=�;�	:       �	�[oAfc�A�*

loss!�<�P+G       �	��oAfc�A�*

loss֪�<`zC       �	R�pAfc�A�*

loss*e�<�ۑ�       �	X7qAfc�A�*

loss�<�g�       �	_�qAfc�A�*

loss�6=��d�       �	P�rAfc�A�*

lossE�/<�        �	�>sAfc�A�*

lossA!�=���
       �	R�sAfc�A�*

lossb��;�S��       �	htAfc�A�*

lossE��;3'��       �	�uAfc�A�*

loss ��;I��O       �	�uAfc�A�*

loss� �<�1�b       �	\TvAfc�A�*

loss��#=e="       �	$�vAfc�A�*

loss���;te<�       �	��wAfc�A�*

loss.�=�}i       �	6?xAfc�A�*

lossX�:=��       �	��xAfc�A�*

loss���;N�k�       �	jhyAfc�A�*

loss�3�=�X��       �	x	zAfc�A�*

loss/�<��U�       �	��zAfc�A�*

loss\��;� \�       �	>{Afc�A�*

loss���=�0�*       �	A�{Afc�A�*

loss
#�<&�,G       �	y|Afc�A�*

loss�<�m       �	�$}Afc�A�*

lossW�:��O       �	!�}Afc�A�*

loss��=9�i       �	Me~Afc�A�*

loss�ȇ<�e�       �	�Afc�A�*

loss�N�;���       �	��Afc�A�*

loss�s�;>D�#       �	�F�Afc�A�*

loss?Y�:.���       �	��Afc�A�*

loss�I�=1��       �	���Afc�A�*

losshh�<���       �	�(�Afc�A�*

loss��z;B��I       �	O˂Afc�A�*

loss�&�<�w��       �	cd�Afc�A�*

loss7s�<��YK       �	��Afc�A�*

loss W�<˄9�       �	ș�Afc�A�*

loss瑈<H5m�       �	�3�Afc�A�*

loss�Q<Z��(       �	!˅Afc�A�*

loss�<�լ[       �	�e�Afc�A�*

loss��<�u �       �	H��Afc�A�*

loss�R;���       �	W��Afc�A�*

loss��=���j       �	n1�Afc�A�*

loss�D=o�*�       �	CʈAfc�A�*

loss|��;�_�       �	�f�Afc�A�*

loss�y(<�q��       �	���Afc�A�*

loss���=w0��       �	���Afc�A�*

loss�q==H���       �	�'�Afc�A�*

lossr�0<��A�       �	 ȋAfc�A�*

loss��Z<;��       �	_a�Afc�A�*

lossSq�<����       �	���Afc�A�*

loss��d<ǸN�       �	���Afc�A�*

loss�=�@^       �	�8�Afc�A�*

loss�0�=���+       �	�͎Afc�A�*

loss�cb;A�|p       �	Nz�Afc�A�*

loss��=��M       �	W"�Afc�A�*

lossa��;��Fa       �	�ΑAfc�A�*

loss_�Z<H�       �	��Afc�A�*

loss��<I�u�       �	I/�Afc�A�*

loss���<3-       �	ݓAfc�A�*

lossH�=���T       �	<��Afc�A�*

losshr!;��5�       �	�-�Afc�A�*

loss~�=QA�c       �	gӕAfc�A�*

loss� =~y�       �	�ݖAfc�A�*

loss��<�i[       �	܁�Afc�A�*

lossØ�<}�O�       �	�ؘAfc�A�*

lossҕn<��wU       �	��Afc�A�*

loss�Ť;�<?r       �	��Afc�A�*

loss,^�<�Ef�       �	�l�Afc�A�*

lossJP�=��S/       �	�Afc�A�*

lossx�m<��ݶ       �	��Afc�A�*

loss7
h<&��z       �	B`�Afc�A�*

loss��U<3�z�       �	K�Afc�A�*

loss�p�<X!.�       �	8��Afc�A�*

loss��<�h��       �	�A�Afc�A�*

loss��=��#       �	ZK�Afc�A�*

loss��<*|A�       �	��Afc�A�*

lossLɁ=���q       �	�~�Afc�A�*

loss��`<��[       �	Y�Afc�A�*

loss1��:�ۀ       �	!��Afc�A�*

loss
�;J�E       �	�D�Afc�A�*

lossD�<��"�       �	�ޣAfc�A�*

lossHI�< �I]       �	�y�Afc�A�*

loss^��<�
��       �	T�Afc�A�*

lossn=�B�3       �	鶥Afc�A�*

loss�\�<��
	       �	�C�Afc�A�*

loss�M�<$R�       �	:�Afc�A�*

loss}o�;�}d�       �	hϨAfc�A�*

lossL�";y�`       �	d�Afc�A�*

lossW��<፱       �	:�Afc�A�*

loss�R�=S1N4       �	Л�Afc�A�*

lossOmu=ɺU       �	(d�Afc�A�*

loss�|;=�<�m       �	�
�Afc�A�*

loss!��;���u       �	y��Afc�A�*

loss�-2=$f9�       �	D�Afc�A�*

lossa\�<�=��       �	ty�Afc�A�*

loss�)G=H�N/       �	��Afc�A�*

loss�c=7ua�       �	�įAfc�A�*

loss���<n6H�       �	�\�Afc�A�*

loss�1;S�	_       �	���Afc�A�*

loss���=��LW       �	O��Afc�A�*

lossS�m<>��       �	�:�Afc�A�*

loss�N=�0)       �	fزAfc�A�*

lossR
R<��]�       �	���Afc�A�*

lossd��;���       �	�O�Afc�A�*

loss��j<�ϕ       �	��Afc�A�*

loss�%�;)�q7       �	���Afc�A�*

loss�A<tdz�       �	�8�Afc�A�*

loss�h�<�"l       �	�ԶAfc�A�*

loss�Č</��       �	�z�Afc�A�*

lossS��<n�#�       �	
�Afc�A�*

loss���<��       �	���Afc�A�*

loss���<xA�       �	�B�Afc�A�*

lossH�="ݳ@       �	IֹAfc�A�*

lossk�=��y�       �	�u�Afc�A�*

loss��<ZmA�       �	r�Afc�A�*

loss_@b<��V        �	5��Afc�A�*

lossH�<=zwf       �	�Y�Afc�A�*

loss=�;mD��       �	���Afc�A�*

loss��V;���a       �	���Afc�A�*

loss�"=�L��       �	�5�Afc�A�*

loss�$�;�>��       �	�ϾAfc�A�*

loss׏�<��˄       �	n�Afc�A�*

loss>'�<�h�       �	
�Afc�A�*

lossS�=[57�       �	[��Afc�A�*

loss|�<���       �	L��Afc�A�*

loss�/6=1��       �	�!�Afc�A�*

loss��==����       �	���Afc�A�*

loss���;�\D�       �	���Afc�A�*

loss$'<B�       �	P9�Afc�A�*

loss��>��Br       �	���Afc�A�*

loss�|�<�7H�       �	ڮ�Afc�A�*

loss�d�:�b�+       �	rM�Afc�A�*

loss4��={+�       �	���Afc�A�*

loss/�<���       �	�y�Afc�A�*

loss�i�=U{��       �	��Afc�A�*

loss\� <Ҽr�       �	��Afc�A�*

lossl,e<\�FK       �	:A�Afc�A�*

loss�h=lU�       �	���Afc�A�*

loss�NQ=d�K       �	���Afc�A�*

lossep�=(b9P       �	�"�Afc�A�*

loss���=�	�(       �	��Afc�A�*

lossہ=�+�       �	�S�Afc�A�*

loss:-�=A��       �	/��Afc�A�*

loss�l�:��m       �	���Afc�A�*

loss��F<P� n       �	(�Afc�A�*

loss��<��q       �	��Afc�A�*

loss��<�"R�       �	~�Afc�A�*

loss=�1<�^`�       �	��Afc�A�*

loss��<�kV       �	��Afc�A�*

loss�T�<��g       �	qU�Afc�A�*

lossq��<4l��       �	t��Afc�A�*

loss�=�ʌ�       �	���Afc�A�*

loss�|<֧�       �	C�Afc�A�*

loss�R=1-L"       �	)��Afc�A�*

loss�ܥ=���.       �	=G�Afc�A�*

lossg�<����       �	���Afc�A�*

loss}��<�_�       �	+l�Afc�A�*

loss��V=��       �	h�Afc�A�*

lossѽ�=I�       �	���Afc�A�*

loss��]=���a       �	Y0�Afc�A�*

loss*x�=u0#�       �	;��Afc�A�*

lossD=_�͜       �	k+�Afc�A�*

loss��0=��(       �	[��Afc�A�*

loss�d�:k�>       �	�s�Afc�A�*

loss�<"�t       �	�Q�Afc�A�*

loss���<7޿�       �	��Afc�A�*

lossf��<8P��       �	a��Afc�A�*

lossEZ�<ʽ�z       �	#.�Afc�A�*

lossP�=��u       �	��Afc�A�*

loss.�P<o��]       �	�V�Afc�A�*

loss]
>=���4       �	[��Afc�A�*

loss��=��r       �	��Afc�A�*

lossDd=���       �	��Afc�A�*

loss�= [/�       �	���Afc�A�*

loss�܆=�(i�       �	�U�Afc�A�*

loss�6�<����       �	���Afc�A�*

loss���<`�
       �	$��Afc�A�*

losso)R<y�J       �	e�Afc�A�*

lossv�;ۙoK       �	���Afc�A�*

loss�tj;�>�       �	�L�Afc�A�*

loss;+�<��V�       �	���Afc�A�*

lossw=^�AS       �	�.�Afc�A�*

loss�B[<� %H       �	X��Afc�A�*

loss��g<�;��       �	'l�Afc�A�*

loss��<Mlܛ       �	_�Afc�A�*

lossW<-=��HG       �	���Afc�A�*

lossC�_<��       �	|_�Afc�A�*

loss=�W=�(�       �	U��Afc�A�*

losswl<\��       �	���Afc�A�*

loss�-�<�S��       �	�,�Afc�A�*

lossc9�<��+�       �	��Afc�A�*

loss��;2X�       �	&V�Afc�A�*

lossD��<����       �	&��Afc�A�*

loss��w=�	��       �	 ��Afc�A�*

loss�.<�5��       �	�#�Afc�A�*

loss���;��~%       �	��Afc�A�*

lossI�<�b-       �	wM�Afc�A�*

loss��U=��A       �	���Afc�A�*

loss��<*�       �	|��Afc�A�*

losss�-=����       �	��Afc�A�*

loss�*=��ɔ       �	��Afc�A�*

loss&�k<��X�       �	|C�Afc�A�*

loss�g�=�ŹY       �	���Afc�A�*

loss�/=WK+�       �	�u�Afc�A�*

lossJ��;��O       �	�
�Afc�A�*

loss���;?Q�7       �	��Afc�A�*

loss�-�<��s       �	?;�Afc�A�*

lossآ.<O��$       �	^��Afc�A�*

lossh�:D�B�       �	3o�Afc�A�*

loss줭;E �       �	h�Afc�A�*

loss\W�=���D       �	Υ�Afc�A�*

lossQC.;�\�;       �	@�Afc�A�*

loss���;p�       �	9��Afc�A�*

lossʰ<���       �	�o�Afc�A�*

lossŲ<OU��       �	h�Afc�A�*

loss�=�<r�       �	
��Afc�A�*

losszȩ;��@       �	�.�Afc�A�*

lossW��=N�+6       �	`��Afc�A�*

loss�^E<���       �	�g�Afc�A�*

loss��i<�vfV       �	e��Afc�A�*

loss�3{='|�       �	��Afc�A�*

loss��$;C���       �	' Bfc�A�*

loss�&<P}G�       �	
� Bfc�A�*

loss
aA;Y#�       �	DPBfc�A�*

loss��n<�N3       �	��Bfc�A�*

loss%�;�8��       �	
�Bfc�A�*

loss�CX<�ѐ       �	#Bfc�A�*

loss��>���       �	��Bfc�A�*

loss��Y=R���       �	�bBfc�A�*

loss3$<k�R       �	�Bfc�A�*

lossaW<���       �	:�Bfc�A�*

loss�d=�2�       �	�ABfc�A�*

loss�>@<��`       �	��Bfc�A�*

loss��t<�=B       �	�mBfc�A�*

lossq�l=ށ��       �	}Bfc�A�*

loss��<k碊       �	��Bfc�A�*

loss7t<]��r       �	�0	Bfc�A�*

loss�<�j;       �	��	Bfc�A�*

loss�r<?&�       �	�x
Bfc�A�*

loss�&�;Hy�       �	QBfc�A�*

loss�B�=1�K�       �	�Bfc�A�*

loss��;��        �	�LBfc�A�*

loss�]v=��,       �	��Bfc�A�*

loss`Y�=�xt�       �	"�Bfc�A�*

loss@�n;b[�       �	�*Bfc�A�*

lossu�<!���       �	�Bfc�A�*

loss��f=���       �	�hBfc�A�*

loss��~<Q�x       �	o�Bfc�A�*

lossZj;&ӹ-       �	�%Bfc�A�*

loss\8=v��9       �	��Bfc�A�*

loss�;�O:�       �	eBfc�A�*

loss�g<Nye       �	!Bfc�A�*

lossn)�;�0�       �	ʨBfc�A�*

loss��H<�K��       �	%@Bfc�A�*

loss��h;A��j       �	a�Bfc�A�*

loss��b=D�<�       �	dvBfc�A�*

lossܐ�=�x�V       �	�Bfc�A�*

loss �;9
�       �	��Bfc�A�*

lossE�<�Sp{       �	3PBfc�A�*

loss;��;�`       �	�Bfc�A�*

loss��<VqE�       �	J{Bfc�A�*

lossI� =��       �	�Bfc�A�*

loss�i<zA�       �	�Bfc�A�*

loss�-=r�R       �	:<Bfc�A�*

loss22<ex@�       �	V�Bfc�A�*

loss��;Ɣ�L       �	9{Bfc�A�*

loss��<�FDM       �	�Bfc�A�*

loss�s�;9���       �	��Bfc�A�*

loss�<�gP0       �	/OBfc�A�*

loss[��;�ge       �	��Bfc�A�*

loss�D�;|y�;       �	�0Bfc�A�*

loss>�=�\�C       �	D�Bfc�A�*

loss��<��       �	�v Bfc�A�*

loss3��;ҥ,{       �	 !Bfc�A�*

loss.��<�P�       �	`�!Bfc�A�*

lossf�<:��       �	�d"Bfc�A�*

lossO�k< ��J       �	* #Bfc�A�*

lossa��;�2�M       �	ҫ#Bfc�A�*

loss���=LV�       �	J]$Bfc�A�*

loss
�:[S�       �	�%Bfc�A�*

loss��+9rF       �	ծ%Bfc�A�*

loss(z;��²       �	�L&Bfc�A�*

lossb�:�Co�       �	 �&Bfc�A�*

loss�8�;jy��       �	��'Bfc�A�*

loss7��;k��5       �	9�(Bfc�A�*

loss�p9�n3       �	=|)Bfc�A�*

lossz<�A       �	*Bfc�A�*

loss7_�8���       �	��*Bfc�A�*

lossO�
:u1?�       �	�F+Bfc�A�*

loss�Q8��݉       �	?�+Bfc�A�*

loss�%;5��       �	�v,Bfc�A�*

loss\J=��(e       �	��-Bfc�A�*

loss(�A<��*       �	ms.Bfc�A�*

loss�Sn;Cr3       �	�/Bfc�A�*

loss�N#=��c�       �	��0Bfc�A�*

lossv�6>���       �	�+1Bfc�A�*

loss#nm95��       �	_�1Bfc�A�*

loss�;�=��{j       �	��3Bfc�A�*

loss]g=��U       �	��4Bfc�A�*

loss�z?=k�%,       �	gF5Bfc�A�*

loss�A�<~�;       �	:�5Bfc�A�*

loss�r�;oּ�       �	ס6Bfc�A�*

loss-�<�5��       �	�M7Bfc�A�*

loss�_=��.       �	T�7Bfc�A�*

loss=�T;�F�#       �	D�8Bfc�A�*

lossy�<�WP}       �	0e:Bfc�A�*

loss�?=��#�       �	;Bfc�A�*

loss�٤<k+.f       �	��;Bfc�A�*

loss�qk=�Ҡ       �	�P<Bfc�A�*

loss�#
=D���       �	� =Bfc�A�*

lossN=S       �	Ϊ=Bfc�A�*

lossX�{=��       �	�M>Bfc�A�*

loss4��<ZyKZ       �	��>Bfc�A�*

loss�9�<Z�	�       �	�?Bfc�A�*

lossE/S=��~�       �	�D@Bfc�A�*

loss���<�Po       �	��@Bfc�A�*

loss��< �1       �	��ABfc�A�*

loss8�;�       �	Q3BBfc�A�*

lossq��<�6�       �	=�BBfc�A�*

lossl�<�N��       �	�|CBfc�A�*

loss2
-<�C�K       �	$DBfc�A�*

loss\~;ӝ�Q       �	��DBfc�A�*

loss��<`X~�       �	'hEBfc�A�*

loss{��:��       �	�FBfc�A�*

loss��k=��+       �	��FBfc�A�*

loss��<�h�C       �	J@GBfc�A�*

loss�n;��oQ       �	v�GBfc�A�*

loss�
<��V       �	��HBfc�A�*

loss�;-�Y       �	F%IBfc�A�*

losss�q:�o1       �	��IBfc�A�*

lossHW=U9�       �	�_JBfc�A�*

loss���<U�S�       �	[KBfc�A�*

loss�S�;Ty�       �	��KBfc�A�*

lossF?�;kKm       �	�ILBfc�A�*

loss	_�<>�^�       �	��LBfc�A�*

loss�P�<&<��       �	U�MBfc�A�*

lossnv�:c-�       �	F"NBfc�A�*

loss�W�<��r�       �	��NBfc�A�*

losss�Z<-/�       �	[^OBfc�A�*

loss��R<�-�       �	��OBfc�A�*

loss��;��m       �	��PBfc�A�*

loss<��<�C|�       �	�~QBfc�A�*

loss��=�á'       �	�RBfc�A�*

losse1 <@{�y       �	W?SBfc�A�*

lossҴ�<�w��       �	U�SBfc�A�*

lossǆ;��r       �	��TBfc�A�*

lossW��<���       �	�GUBfc�A�*

loss�=eE�       �	�aoBfc�A�*

lossܷ�<��X�       �	�SpBfc�A�*

loss�-�=��       �	��pBfc�A�*

losse��<����       �	��qBfc�A�*

loss���<̕L�       �	utrBfc�A�*

loss�`<��       �	sBfc�A�*

loss�%�<2޴�       �	��sBfc�A�*

lossW��;/lt�       �	�tBfc�A�*

loss�Ո<Y�6�       �	�puBfc�A�*

loss3�_<� �]       �	<vBfc�A�*

loss�<�       �	��vBfc�A�*

loss��;��<K       �	�`wBfc�A�*

loss �=.ү        �	NxBfc�A�*

loss���<ϴAZ       �	��xBfc�A�*

loss���;[H��       �	�fyBfc�A�*

loss*�;�7��       �	�zBfc�A�*

lossz�:�̺       �	�zBfc�A�*

loss��h=�F)�       �	gb{Bfc�A�*

loss�<��z        �	�|Bfc�A�*

lossDvH<N"1       �	��|Bfc�A�*

loss;_�;�\SC       �	. ~Bfc�A�*

loss�7�=��y       �	¥~Bfc�A�*

losss�-;O���       �	AHBfc�A�*

loss�z^=g��b       �	A�Bfc�A�*

loss�t&<��`�       �	ٕ�Bfc�A�*

loss��<��f       �	s0�Bfc�A�*

lossx��;1��j       �	)́Bfc�A�*

loss1�<���3       �	�h�Bfc�A�*

loss��<M��	       �	�
�Bfc�A�*

loss�H=�.m       �	���Bfc�A�*

loss�<;����       �	t`�Bfc�A�*

loss���;��>>       �	}�Bfc�A�*

loss�l <(>�       �	��Bfc�A�*

loss$H�<�@       �	�=�Bfc�A�*

loss�=�~�       �	�ކBfc�A�*

loss@v�=o0��       �	�}�Bfc�A�*

loss��;�4��       �	K#�Bfc�A�*

loss���=_=�       �	"ƈBfc�A�*

loss��?=~G�       �	vo�Bfc�A�*

loss��c=6s�       �	�ϊBfc�A�*

lossi�;��

       �	 t�Bfc�A�*

lossjq�;��h       �	&�Bfc�A�*

loss��<�.�       �	7ǌBfc�A�*

loss�!O<Kd�       �	�r�Bfc�A�*

loss��_=�e�       �	U�Bfc�A�*

loss��V=����       �	��Bfc�A�*

loss�g�;�߯�       �	�a�Bfc�A�*

loss�`=d��       �	��Bfc�A�*

loss��%<2��"       �	���Bfc�A�*

loss��;#e�       �	��Bfc�A�*

loss���;��Y�       �	�+�Bfc�A�*

lossZ��=�[D       �	1ϒBfc�A�*

loss�jk<8&�P       �	C��Bfc�A�*

loss�O>3Ȉb       �	M��Bfc�A�*

loss[�;H&l�       �	D�Bfc�A�*

loss呀:�r�       �	��Bfc�A�*

loss]��:`I|       �	<��Bfc�A�*

loss?[.:�jC       �	kF�Bfc�A�*

lossi6d;�CQ       �	�Bfc�A�*

loss�q�=���       �	|��Bfc�A�*

loss ��=d��       �	2q�Bfc�A�*

loss�l"=��
       �	��Bfc�A�*

loss��<:��S       �	)��Bfc�A�*

loss}��<�       �	�U�Bfc�A�*

loss]��<K3��       �	:y�Bfc�A�*

loss��4=���       �	��Bfc�A�*

loss�g�<�˒       �	GƟBfc�A�*

loss�?N<y��       �	*q�Bfc�A�*

loss��B=�H �       �	�!�Bfc�A�*

lossx�0<f��,       �	�ȡBfc�A�*

lossX =�tnq       �	q�Bfc�A�*

loss���<p��       �	A�Bfc�A�*

lossL8=_�C�       �	���Bfc�A�*

loss�Q<u�`�       �	
��Bfc�A�*

loss�7-=q�v�       �	�=�Bfc�A�*

loss$�n<.aD�       �	e�Bfc�A�*

loss��;ٯ;       �	���Bfc�A�*

loss�c8<���S       �	�*�Bfc�A�*

loss�:4R��       �	\ʧBfc�A�*

loss�d==��o       �	h�Bfc�A�*

loss-t�<��3       �	��Bfc�A�*

loss}s =t٧3       �	.��Bfc�A�*

loss���<���       �	N�Bfc�A�*

loss���:A �       �	 �Bfc�A�*

loss�F6<P0��       �	�w�Bfc�A�*

lossv�n;v�)�       �	��Bfc�A�*

loss�h�<܈��       �	���Bfc�A�*

loss�i�<��I       �	�V�Bfc�A�*

lossj��<%�?\       �	�Bfc�A�*

loss,<i)vY       �	&��Bfc�A�*

loss؄�<�
1�       �	Ѳ�Bfc�A�*

loss��;WC�       �	PQ�Bfc�A�*

loss���;P��       �	��Bfc�A�*

loss�;4S9�       �	~��Bfc�A�*

loss���<,˽�       �	�$�Bfc�A�*

loss^y<�\�       �	��Bfc�A�*

loss�"=��B       �	�L�Bfc�A�*

loss��<�Ϊ�       �	�Bfc�A�*

loss	�=�&ef       �	D��Bfc�A�*

loss�g�<0ӛ(       �	�Bfc�A�*

losst�
<��:	       �	ۿ�Bfc�A�*

lossQ�;� Hz       �	\W�Bfc�A�*

lossfeE=�I�       �	���Bfc�A�*

loss�l7=�~��       �	���Bfc�A�*

loss�D<��;�       �	�&�Bfc�A�*

lossl6=��Z       �	�øBfc�A�*

lossΌ;��ҽ       �	t`�Bfc�A�*

lossC��:���M       �	��Bfc�A�*

loss��^=����       �	3��Bfc�A�*

lossۉ�;���       �	W>�Bfc�A�*

loss�z�=و�       �	�Bfc�A�*

loss�<�3c�       �	1~�Bfc�A�*

loss,�=��N�       �	��Bfc�A�*

loss�P|<|G#�       �	���Bfc�A�*

loss�x�<���U       �	�S�Bfc�A�*

loss��: 	Q       �	9�Bfc�A�*

loss�9�<!�E�       �	Ŏ�Bfc�A�*

loss]��=�l��       �	x%�Bfc�A�*

loss�+=���       �	��Bfc�A�*

loss�&-=�	��       �	�Q�Bfc�A�*

loss ��<e/�       �	[��Bfc�A�*

lossW��<�Y       �	
��Bfc�A�*

lossW�e<�1{�       �	�)�Bfc�A�*

loss�<E�r       �	���Bfc�A�*

lossC�;Yiې       �	5^�Bfc�A�*

lossZ�3<�-w       �	���Bfc�A�*

lossjL�;� �t       �	F��Bfc�A�*

lossC|;�Ey       �	�-�Bfc�A�*

loss.��<mZ	       �	d��Bfc�A�*

loss=�Լ�       �	/n�Bfc�A�*

lossZϪ;⊳�       �	��Bfc�A�*

lossO+<�q��       �	��Bfc�A�*

loss�-�;�_       �	2V�Bfc�A�*

loss�Ai<R���       �	���Bfc�A�*

loss.I�<��?       �	+��Bfc�A�*

loss��;C��O       �	�&�Bfc�A�*

loss�q�<�Z�       �	��Bfc�A�*

loss��V;YVu�       �	˄�Bfc�A�*

lossZ�:7��       �		�Bfc�A�*

loss���<�C��       �	���Bfc�A�*

lossYM<M��!       �	�N�Bfc�A�*

loss���<�,�^       �	���Bfc�A�*

losse��<����       �	ۋ�Bfc�A�*

loss�̪<j@       �	�,�Bfc�A�*

lossl��<l=C�       �	��Bfc�A�*

lossdR�<U�F       �	���Bfc�A�*

loss�z;;fy       �	A��Bfc�A�*

loss�;6;����       �	���Bfc�A�*

losslRw;S�R       �	�m�Bfc�A�*

loss�w:K��e       �	���Bfc�A�*

lossV��;Ԟ��       �	�!�Bfc�A�*

loss�V<�)�r       �	JA�Bfc�A�*

loss�(?<���       �	T��Bfc�A�*

lossb��=s}}n       �	dy�Bfc�A�*

lossC�h<���       �	(�Bfc�A�*

loss�V�9��~�       �	ߩ�Bfc�A�*

loss6=9;+�gy       �	��Bfc�A�*

loss��<*�       �	�0�Bfc�A�*

loss!A<ns�       �	2��Bfc�A�*

lossç�;�)�        �	���Bfc�A�*

loss�vt<�Ϊ�       �	6�Bfc�A�*

loss�S�;��:�       �	\��Bfc�A�*

loss�_=�ǋY       �	�s�Bfc�A�*

lossl�>:/b�       �	{�Bfc�A�*

lossD�\<PW       �	^��Bfc�A�*

loss�h=FQ	�       �	�Z�Bfc�A�*

loss/�;!<D�       �	���Bfc�A�*

loss��=��Y�       �	f��Bfc�A�*

loss��&;���V       �	/�Bfc�A�*

loss%�;~��d       �	z��Bfc�A�*

lossJ@W;f���       �	h�Bfc�A�*

lossCB;���D       �	�Bfc�A�*

lossH�v<w�n       �	���Bfc�A�*

lossju�;L�M       �	�H�Bfc�A�*

loss�?�<�*9       �	���Bfc�A�*

lossV.A;{�,$       �	���Bfc�A�*

loss�ͱ<q�{�       �	rP�Bfc�A�*

loss��L;��܊       �	J��Bfc�A�*

loss�/=�l�       �	�#�Bfc�A�*

loss%��:#@�h       �	���Bfc�A�*

loss�ɰ;��D�       �	���Bfc�A�*

lossJ�;����       �	��Bfc�A�*

loss9�<B�=S       �	%�Bfc�A�*

loss�
m<��_       �	q��Bfc�A�*

loss�L@;�d�e       �	���Bfc�A�*

loss:��;�P*�       �	E��Bfc�A�*

loss���=���       �	"T�Bfc�A�*

lossZ2=�Ox       �	���Bfc�A�*

loss
(<Y�\h       �	A��Bfc�A�*

loss�uT;��n9       �	�u�Bfc�A�*

loss-�>=�H��       �	[�Bfc�A�*

loss^�<��d=       �	���Bfc�A�*

loss�+
<u��       �	���Bfc�A�*

loss1)_<�7�3       �	8,�Bfc�A�*

loss7�5=���I       �	���Bfc�A�*

loss䌱<\��       �	�W�Bfc�A�*

lossO��:Ϣ �       �	Z��Bfc�A�*

lossqq="��       �	��Bfc�A�*

lossz�='i�       �	t��Bfc�A�*

lossC&�:U�3�       �	>^�Bfc�A�*

loss�ŗ<�T5�       �	���Bfc�A�*

loss3^a;���       �	���Bfc�A�*

loss��=W�+0       �	G9�Bfc�A�*

loss��h=��Rf       �	>��Bfc�A�*

loss�Uc;���       �	��Bfc�A�*

loss?o�;o�rc       �	ܟ�Bfc�A�*

loss�V%=�4�&       �	?�Bfc�A�*

loss۰?<�Zn�       �	���Bfc�A�*

loss�
1=���=       �	7o�Bfc�A�*

lossZ<�<��5�       �	J�Bfc�A�*

loss/�<��0T       �	��Bfc�A�*

loss�E�;�=       �	�= Cfc�A�*

loss�E<�
5�       �	�� Cfc�A�*

loss2��<��       �	(�Cfc�A�*

loss�7�<�Ҝ       �	W'Cfc�A�*

loss���=2�H       �	L�Cfc�A�*

loss���=@l-       �	�ZCfc�A�*

loss�Ҹ=ܖW       �	'�Cfc�A�*

loss�:==��ȉ       �	!�Cfc�A�*

loss��	=�ʸ:       �	�<Cfc�A�*

loss��;46       �	��Cfc�A�*

loss��<��       �	Cfc�A�*

loss&i<X�4+       �	N�Cfc�A�*

loss�Qd=>�;G       �	�VCfc�A�*

loss�;��C^       �	��Cfc�A�*

loss��<�W�       �	^�	Cfc�A�*

loss���<���       �	q<
Cfc�A�*

lossAJ�<��N)       �	��
Cfc�A�*

loss�;y<{wI�       �	��Cfc�A�*

loss��
<��C        �	/5Cfc�A�*

loss#�<���2       �	��Cfc�A�*

loss���;3�[�       �	ǀCfc�A�*

losscI.=|�<r       �	�!Cfc�A�*

loss�=x�Ɠ       �	P�Cfc�A�*

loss��<9;��       �	�cCfc�A�*

loss�@�<�6��       �	Cfc�A�*

loss)�&=�0_+       �	�Cfc�A�*

loss&�<�Q��       �	W=Cfc�A�*

lossӕM=��N�       �	��Cfc�A�*

loss�D�;�Y;�       �	�Cfc�A�*

loss�I<;{�xj       �	;�Cfc�A�*

loss��<d\�       �	5^Cfc�A�*

loss��	=����       �	�;Cfc�A�*

loss[w<k��       �	,�Cfc�A�*

lossm =y���       �	��Cfc�A�*

loss��V;�6��       �	q�Cfc�A�*

loss�݈<��>�       �	v4Cfc�A�*

loss?��=����       �	��Cfc�A�*

lossH=��IH       �	gbCfc�A�*

lossv��;�^Ў       �	�
Cfc�A�*

loss�f<g-d\       �	��Cfc�A�*

losso+=j߳�       �	�ZCfc�A�*

loss���;��S'       �	Y�Cfc�A�*

loss>
=�K       �	�Cfc�A�*

losso��:F#Db       �	�9Cfc�A�*

losss�<]pZ�       �	��Cfc�A�*

loss��<��F       �	�zCfc�A�*

loss�?|<4���       �	?Cfc�A�*

lossD��:�T	�       �	��Cfc�A�*

loss��=���       �	�f Cfc�A�*

lossd��<ЋZi       �	�!Cfc�A�*

loss�q5;�3�       �	I�!Cfc�A�*

loss@�<���       �	_a"Cfc�A�*

lossffa<�`�       �	�#Cfc�A�*

lossY�;��´       �	L�#Cfc�A�*

loss���:w��       �	�H$Cfc�A�*

loss�!=%Nq       �	��$Cfc�A�*

loss���=�ę�       �	�%Cfc�A�*

loss�3=S�8�       �	�2&Cfc�A�*

loss��=��       �	�&Cfc�A�*

loss�;j��-       �	zo'Cfc�A�*

loss �i<��%       �	�(Cfc�A�*

loss��<�9��       �	U�(Cfc�A�*

loss z!<�#��       �	u>)Cfc�A�*

loss���=X��L       �	��)Cfc�A�*

loss7_�=�e�D       �	�*Cfc�A�*

loss�,<.lP       �	�:+Cfc�A�*

loss��$;���       �	��+Cfc�A�*

loss��=8b	       �	{,Cfc�A�*

loss�=�&��       �	�-Cfc�A�*

loss�-+;�6�$       �	5�-Cfc�A�*

loss�k'<ct��       �	�.Cfc�A�*

loss��#<�Q	D       �	�n/Cfc�A�*

loss]p=��       �	�0Cfc�A�*

loss��<D�χ       �	�1Cfc�A�*

loss��<#�M�       �	�1Cfc�A�*

loss3=NkR       �	�?2Cfc�A�*

loss���<l�ߡ       �	��2Cfc�A�*

loss�u<J[��       �	��3Cfc�A�*

loss�&x=�s��       �	|`4Cfc�A�*

loss��:�ܩ�       �	o�4Cfc�A�*

loss���<�BA�       �	l�5Cfc�A�*

loss6�0:f�i�       �	s+6Cfc�A�*

loss��:/\r       �	��6Cfc�A�*

loss��p<k�I�       �	�j7Cfc�A�*

loss���;Qu^�       �	��8Cfc�A�*

loss�G:=��j�       �	29Cfc�A�*

lossŢ�=���(       �	��9Cfc�A�*

loss&=IQq�       �	�[:Cfc�A�*

loss;�[=aD-�       �	��:Cfc�A�*

loss�6)=�'       �	�;Cfc�A�*

loss)��<�S�       �	�C<Cfc�A�*

loss<�6;`�L       �	d�<Cfc�A�*

loss�|�=�´       �	�=Cfc�A�*

loss[��=E�?~       �	s+>Cfc�A�*

loss=���C       �	��>Cfc�A�*

loss��;.9|       �	�l?Cfc�A�*

loss�'�<��uf       �	K@Cfc�A�*

lossF�;xw�`       �	7�@Cfc�A�*

loss6u
<�^�       �	@MACfc�A�*

losse�[<_ʱt       �	��ACfc�A�*

loss��<j�       �	x�BCfc�A�*

loss�{�<o.^U       �	�8CCfc�A�*

loss��<j��;       �	@�CCfc�A�*

loss/;N��T       �	�DCfc�A�*

loss7�f=B_�       �	O ECfc�A�*

loss�f:�o��       �	��ECfc�A�*

lossi!;���       �	cFCfc�A�*

loss�e�=�        �	OGCfc�A�*

loss�?=�T�C       �	X�GCfc�A�*

loss�	<-��       �	3NHCfc�A�*

loss|�+<�iJ�       �	6�HCfc�A�*

lossx��;7��       �	&�ICfc�A�*

loss��;��       �	((JCfc�A�*

loss�<dN$�       �	��JCfc�A�*

loss�!|=����       �	�cKCfc�A�*

lossC+�<� ��       �	��KCfc�A�*

loss{�<�X��       �	ҫLCfc�A�*

loss�=�<���       �	�EMCfc�A�*

loss�w=�p�v       �	j�MCfc�A�*

loss]c�<�2=�       �	�xNCfc�A�*

lossH�6;����       �	IOCfc�A�*

loss�;���d       �	[�OCfc�A�*

loss�9=S�0�       �	GPCfc�A�*

lossun<C9�       �	��PCfc�A�*

loss��X;��Ŕ       �	\wQCfc�A�*

lossQ�V=\��|       �	�&RCfc�A�*

loss���<pk�       �	C�RCfc�A�*

loss?�J=�چR       �	!�SCfc�A�*

loss��:U�       �	� UCfc�A�*

loss��;S��       �	��UCfc�A�*

loss`��;|ǔ       �	2�VCfc�A�*

losslk�=j��       �	�MWCfc�A�*

loss�<�;ZAe       �	��WCfc�A�*

loss�[�=d	es       �	C�XCfc�A�*

loss�x�=dӫ       �	�4YCfc�A�*

loss�|;�(Z�       �	�4ZCfc�A�*

loss��J<����       �	��ZCfc�A�*

loss��=����       �	k[Cfc�A�*

loss*"�<LY��       �	u\Cfc�A�*

loss���;h�9�       �	��\Cfc�A�*

lossn�;<�C��       �	�u]Cfc�A�*

loss��8=!q�       �	�^Cfc�A�*

loss���<�ùU       �	�^Cfc�A�*

lossJ�<�U�       �	%X_Cfc�A�*

lossC�D=����       �	9�_Cfc�A�*

loss�y<ǽt       �	�`Cfc�A�*

loss�,=]��       �	�,aCfc�A�*

lossO"g=��L�       �	.�aCfc�A�*

loss2��<�`�       �	��bCfc�A�*

loss�k<KĞ�       �	�scCfc�A�*

loss)�;4���       �	�dCfc�A�*

loss|�; �Tp       �	��dCfc�A�*

loss��<@%f       �	LeCfc�A�*

loss�	=1]�       �	X�eCfc�A�*

loss7��;�cf�       �	��fCfc�A�*

loss��==r߳       �	�gCfc�A�*

loss�;^��       �	U�gCfc�A�*

loss�Ҹ;��{E       �	�DiCfc�A�*

lossq�S;�[>       �	0kCfc�A�*

lossc�Z=V�>�       �	��kCfc�A�*

loss˹	=#��       �	�{lCfc�A�*

loss��?;'mOs       �	�dmCfc�A�*

lossn�<���       �	#�mCfc�A�*

loss���<uG�       �	X�nCfc�A�*

loss�m�<�K       �	_$oCfc�A�*

loss<�];ln       �	A�oCfc�A�*

loss@U<� ?�       �	�QpCfc�A�*

loss�<w���       �	��pCfc�A�*

loss�{=�؜�       �	�qCfc�A�*

loss�7<B֢       �	�:rCfc�A�*

loss��n=%%       �	��rCfc�A�*

lossf�	=I~��       �	?osCfc�A�*

loss0�;���       �	5tCfc�A�*

loss��?=V���       �	��tCfc�A�*

lossޚ=5��       �	1BuCfc�A�*

loss;��<��       �	��uCfc�A�*

loss���=�j�Z       �	p}vCfc�A�*

loss��<� "       �	��wCfc�A�*

loss���:I9��       �	�fxCfc�A�*

lossc4�<c��       �	7�xCfc�A�*

loss�G<��t       �	q�yCfc�A�*

loss�0<P�k�       �	s-zCfc�A�*

loss3E�;AQ=(       �	�zCfc�A�*

loss�<t��8       �	�[{Cfc�A�*

loss�+K=c���       �	�{Cfc�A�*

lossQC;�:Zc       �	�|Cfc�A�*

loss�F�<y�       �	�'}Cfc�A�*

loss|P�;4v/c       �	'�}Cfc�A�*

lossq�=�b\G       �	\Z~Cfc�A�*

lossZ�<2��t       �	�~Cfc�A�*

loss�Q�;V�I       �	��Cfc�A�*

loss:1-<�7�       �	�>�Cfc�A�*

loss��z;!��       �	�ԀCfc�A�*

loss ��=�,�p       �	�h�Cfc�A�*

loss��<|\G       �	�^�Cfc�A�*

lossћ�<5���       �	R�Cfc�A�*

loss`u�;���       �	ĵ�Cfc�A�*

loss�1	<&Y��       �	�T�Cfc�A�*

lossG[<��R�       �	(�Cfc�A�*

loss���<��/       �	���Cfc�A�*

lossOfU=�       �	7�Cfc�A�*

loss��Y=�*�       �	��Cfc�A�*

loss��<lsO�       �	��Cfc�A�*

loss�M�<􈅦       �	ND�Cfc�A�*

loss��<=OJ�       �	��Cfc�A�*

loss|2<._*�       �	�Cfc�A�*

lossd<=���6       �	��Cfc�A�*

loss`�;��s�       �	+��Cfc�A�*

loss��	=�"�H       �	�U�Cfc�A�*

loss?�`;M�C       �	x�Cfc�A�*

loss�:=UCތ       �	v��Cfc�A�*

lossբ:}�׌       �	��Cfc�A�*

loss���<@:��       �	���Cfc�A�*

loss�G=�z\       �	|b�Cfc�A�*

lossԲ<1�\�       �	���Cfc�A�*

loss�N�<�cǘ       �	-��Cfc�A�*

loss�<_͉/       �	�1�Cfc�A�*

loss�4<h�d4       �	.ȐCfc�A�*

loss�?�<p�:�       �	�a�Cfc�A�*

loss��Z< �^�       �	�Cfc�A�*

loss*v�<^Z�a       �	�ǒCfc�A�*

loss#�<�Ϟ       �	{g�Cfc�A�*

loss�`!<�5P�       �	z��Cfc�A�*

lossE�;ӶM       �	^ՔCfc�A�*

lossz�<��0M       �	�Cfc�A�*

loss�?�:�۾�       �	}%�Cfc�A�*

lossI�;�Uv�       �	ǖCfc�A�*

loss�r�<.b�       �	n�Cfc�A�*

loss���;���       �	��Cfc�A�*

loss�X=�ړ5       �	���Cfc�A�*

loss�2=|i�~       �	wM�Cfc�A�*

loss���;��       �	{��Cfc�A�*

loss�,y=Ԯ       �	���Cfc�A�*

loss+;��`       �	\8�Cfc�A�*

loss0)<��p       �	�ۛCfc�A�*

loss	K=���       �	�y�Cfc�A�*

loss:�?;\(�       �	m�Cfc�A�*

loss3�<�Z�       �	麝Cfc�A�*

loss�R�<�}�~       �	CV�Cfc�A�*

loss_��<�D~       �	Q��Cfc�A�*

loss���:7��#       �	i��Cfc�A�*

lossj�<�o��       �	|)�Cfc�A�*

loss7�<�Q�G       �	Q��Cfc�A�*

lossi^�<"jF       �	�T�Cfc�A�*

loss��<�Qf       �	��Cfc�A�*

loss�=�l�       �	���Cfc�A�*

loss#��<��f�       �	�"�Cfc�A�*

loss��Y=<d��       �	��Cfc�A�*

lossm��;:U5�       �	�L�Cfc�A�*

loss���:�f��       �	T�Cfc�A�*

loss�d=�bz       �	+��Cfc�A�*

loss{�9��       �	>&�Cfc�A�*

loss	Em<苅�       �	̦Cfc�A�*

loss��D;o�       �	�j�Cfc�A�*

loss3�T;��	n       �	2 �Cfc�A�*

loss�y<$+       �	��Cfc�A�*

loss�>S;�?��       �	�)�Cfc�A�*

loss!2=pƨF       �	4��Cfc�A�*

loss�8:�2�       �	qX�Cfc�A�*

loss�&�=�6       �	��Cfc�A�*

loss�O�<V��A       �	%��Cfc�A�*

loss]L�;u�g�       �	b0�Cfc�A�*

loss��G=vP�       �	�ɬCfc�A�*

loss[zP<�Eg       �	Gv�Cfc�A�*

loss��8<!:s�       �	p}�Cfc�A�*

loss�U�;�Q��       �	��Cfc�A�*

losss"�<e��       �	~ȯCfc�A�*

loss�+:���       �	
1�Cfc�A�*

lossM`=T���       �	B�Cfc�A�*

loss�'�=�Q��       �	6��Cfc�A�*

loss��=�&       �	V-�Cfc�A�*

loss7��<��#       �	��Cfc�A�*

loss��<���p       �	y:�Cfc�A�*

loss}|(=���`       �	9ֵCfc�A�*

loss�+�;����       �	��Cfc�A�*

loss	Jz<g�R�       �	Q�Cfc�A�*

loss��<V��[       �	��Cfc�A�*

loss��;�ob�       �	@��Cfc�A�*

loss�-�;
d�|       �	a�Cfc�A�*

loss|$=�M�       �	I��Cfc�A�*

loss���<P��       �	�`�Cfc�A�*

lossI��9�7��       �	��Cfc�A�*

loss#��=�M       �	^��Cfc�A�*

loss)�<�       �	�9�Cfc�A�*

lossѥ;m3_       �	0׼Cfc�A� *

loss��*<��       �	�u�Cfc�A� *

loss�{-=LEw�       �	�Cfc�A� *

lossI<4\��       �	;��Cfc�A� *

loss<ԁ=�]A@       �	�=�Cfc�A� *

loss���<e�+A       �	�տCfc�A� *

loss�X<ˋXj       �	�k�Cfc�A� *

loss��P=�K��       �	��Cfc�A� *

lossb=en��       �	j��Cfc�A� *

loss��<�<�       �	�2�Cfc�A� *

lossF�<IV��       �	7��Cfc�A� *

loss�Ɓ<
�6u       �	.Y�Cfc�A� *

loss$�<t�       �	c��Cfc�A� *

lossv�y;{�D       �	���Cfc�A� *

loss�=�q�       �	�Cfc�A� *

loss�?<l��       �	N��Cfc�A� *

loss�\�<�7��       �	�Q�Cfc�A� *

loss��;���S       �	���Cfc�A� *

loss�%`<�Y��       �	�~�Cfc�A� *

lossʝ�<�       �	�Cfc�A� *

loss�Й<�~B,       �	N��Cfc�A� *

loss0��=桰{       �	�L�Cfc�A� *

loss.�<qD       �	�X�Cfc�A� *

loss �N=�#A       �	���Cfc�A� *

lossK�<��Y       �	 ��Cfc�A� *

loss��9�ɹ
       �	�(�Cfc�A� *

loss�c<)lS�       �	T��Cfc�A� *

loss��;X�ŭ       �	h]�Cfc�A� *

loss��;�!�0       �	���Cfc�A� *

lossͅ�<��2�       �	'��Cfc�A� *

loss�s�;B&�       �	�:�Cfc�A� *

loss}�<�}       �	���Cfc�A� *

lossJE7=�{m       �	�o�Cfc�A� *

loss�4z=�R �       �	�Cfc�A� *

loss��5=�Yл       �	H��Cfc�A� *

loss�<�ݮ�       �	/R�Cfc�A� *

loss��<�C9P       �	5�Cfc�A� *

loss���;/]�       �	{��Cfc�A� *

loss\1:��/�       �	���Cfc�A� *

losso,%;�#��       �	@��Cfc�A� *

lossz^+:>�m�       �	^/�Cfc�A� *

loss��&;��$�       �	��Cfc�A� *

loss��<���r       �	���Cfc�A� *

loss��9=����       �	#1�Cfc�A� *

lossv6;�Y�       �	��Cfc�A� *

lossce9�^'�       �	7��Cfc�A� *

loss��:Uɾ�       �	�b�Cfc�A� *

loss�o8x�m�       �	�(�Cfc�A� *

loss�;�s��       �	  �Cfc�A� *

lossd7=�ifu       �	���Cfc�A� *

loss�+�=\Sj       �	N��Cfc�A� *

lossT:���       �	m<�Cfc�A� *

lossz�;)^�       �	M��Cfc�A� *

loss<��=�ob�       �	Xr�Cfc�A� *

loss�nE<�S�S       �	��Cfc�A� *

loss�S(>��       �	O��Cfc�A� *

loss�NO<�&�       �	���Cfc�A� *

loss�F7=H@�       �	b0�Cfc�A� *

lossL߿<�a:A       �	���Cfc�A� *

lossz��<��r       �	�r�Cfc�A� *

lossvQ�<c�k       �	D�Cfc�A� *

loss��U=�(׍       �	��Cfc�A� *

loss���;&@��       �	2U�Cfc�A� *

loss�%:=�PzN       �	E��Cfc�A� *

loss��<)�I�       �	2��Cfc�A� *

losso*=gY��       �	&�Cfc�A� *

loss��B=#H_�       �	���Cfc�A� *

loss=��;D���       �	mX�Cfc�A� *

loss�E�<��P       �	���Cfc�A� *

lossZ��;��A       �	O��Cfc�A� *

lossg�<�꡺       �	
/�Cfc�A� *

loss ��:��H       �	=��Cfc�A� *

loss�
,=G'G�       �	V~�Cfc�A� *

lossTbk<�'�D       �	��Cfc�A� *

loss��i;�%�       �	���Cfc�A� *

lossn�5<�¢~       �	R��Cfc�A� *

losse�r<���T       �	�}�Cfc�A� *

loss{+<|E2�       �	('�Cfc�A� *

loss�Q;`}|       �	\��Cfc�A� *

loss;��<�~��       �	�p�Cfc�A� *

loss��=z!       �	4�Cfc�A� *

loss;*C<�M�       �	���Cfc�A� *

loss�	w<����       �	�F�Cfc�A� *

lossO��<I4�       �	��Cfc�A� *

lossH<�U�       �	��Cfc�A� *

lossQ
K=�N\a       �	�F�Cfc�A� *

loss��<#�	i       �	t�Cfc�A� *

loss���;U�{       �	���Cfc�A� *

loss�-�;Ԝj       �	rQ�Cfc�A� *

loss��=���x       �	6��Cfc�A� *

lossd.<sdX�       �	g}�Cfc�A� *

lossE=>Y!�       �	b�Cfc�A� *

lossht=l��       �	��Cfc�A� *

loss���<�s�       �	=�Cfc�A� *

loss�;��D       �	F��Cfc�A� *

loss*{�=3���       �	+k�Cfc�A� *

loss��P<m�}       �	��Cfc�A� *

loss<3�<�p       �	N��Cfc�A� *

loss�\G;:퉝       �		T�Cfc�A� *

loss�t=�*�       �	���Cfc�A� *

loss�<�y��       �	���Cfc�A� *

loss\�;?�h>       �	(+�Cfc�A� *

loss��7<��4`       �	;��Cfc�A� *

loss���;���       �	�Z�Cfc�A� *

lossܥA;��\       �	��Cfc�A� *

loss�F�;Tl-Y       �	��Dfc�A� *

loss�$[=��n       �	~�Dfc�A� *

loss��<�p�       �	8�Dfc�A� *

loss!�=%�0�       �	\ZDfc�A� *

lossUK=d'�!       �	j�Dfc�A� *

loss���;��#       �	1�Dfc�A� *

loss��3=5D9�       �	'/Dfc�A� *

loss��?;��->       �	��Dfc�A� *

loss	XJ=s?��       �	|aDfc�A� *

loss��x=�zOy       �	� Dfc�A� *

lossR�v;��T�       �	��Dfc�A� *

loss���=SP�       �	+4 Dfc�A� *

loss�
�<�$�z       �	�� Dfc�A� *

lossL��<T��       �	�n!Dfc�A� *

loss4�u<+oac       �	�	"Dfc�A� *

loss]�G;���Y       �	��"Dfc�A� *

loss�9�9�,��       �	8I#Dfc�A� *

loss���;S�M       �	��#Dfc�A� *

loss|�<��       �	\q$Dfc�A� *

loss��=w|r       �	�%Dfc�A� *

loss���<�07�       �	¤%Dfc�A� *

loss/I�=�	�       �	�C&Dfc�A� *

loss�-�;:c��       �	i�&Dfc�A� *

lossC�d=�e��       �	�~'Dfc�A�!*

loss�nJ<�o:P       �	H(Dfc�A�!*

loss��V<Pk�r       �	��(Dfc�A�!*

loss,9�;J�x�       �	�U)Dfc�A�!*

loss��<��2�       �	<�)Dfc�A�!*

loss&4�<��6F       �	��*Dfc�A�!*

lossn��<u��e       �	�/+Dfc�A�!*

loss��;e�d�       �	$�+Dfc�A�!*

loss�^�;`]��       �	O�,Dfc�A�!*

loss�u�<N�h�       �	�3-Dfc�A�!*

lossE�r=�S_       �	��-Dfc�A�!*

lossK:�;	��       �	)x.Dfc�A�!*

loss�Q<���       �	�/Dfc�A�!*

lossN�4<�G�T       �	��/Dfc�A�!*

losssq<��t       �	+�0Dfc�A�!*

loss�]�<��u       �	h\1Dfc�A�!*

loss���<c���       �	�2Dfc�A�!*

lossdz�;���       �	h�2Dfc�A�!*

losss�:ΘJ       �	�(3Dfc�A�!*

loss$
�;rxB        �	%�3Dfc�A�!*

loss�n�;�WEM       �	�5Dfc�A�!*

loss���;T܆C       �	ɰ5Dfc�A�!*

loss�S<����       �		T6Dfc�A�!*

loss���;��n       �	��6Dfc�A�!*

lossJ�:<`7       �	�7Dfc�A�!*

loss9�;ҵ4       �	�J8Dfc�A�!*

loss|:5VB�       �	jK9Dfc�A�!*

lossIn�:��       �	N�9Dfc�A�!*

loss��<F^��       �	��:Dfc�A�!*

lossҞS:;u��       �	�,;Dfc�A�!*

loss���<{F�        �	�;Dfc�A�!*

lossM��<�/��       �	�o<Dfc�A�!*

loss�D<o��       �	R=Dfc�A�!*

loss횈9�2�       �	�=Dfc�A�!*

loss� �9[4x       �	�D>Dfc�A�!*

loss`=l'Q�       �	]�>Dfc�A�!*

loss��="l��       �	Sx?Dfc�A�!*

loss��=@Tʈ       �	@Dfc�A�!*

loss�t�9ʐ0W       �	�@Dfc�A�!*

loss,�:�ז�       �	�BADfc�A�!*

loss�֌<�-4       �	a�ADfc�A�!*

loss��$<ڱ:       �	{BDfc�A�!*

loss��^;Zx��       �	�CDfc�A�!*

losswR<ZH��       �	��CDfc�A�!*

lossd�;��%�       �	$`DDfc�A�!*

loss��=Z��       �	��DDfc�A�!*

loss��<��       �	c(FDfc�A�!*

loss
)i<�6s�       �	Q�FDfc�A�!*

loss	��<t1�       �	PTGDfc�A�!*

loss��A<s��:       �	$�GDfc�A�!*

lossB=<�-}       �	�HDfc�A�!*

loss�q�<�:��       �	qqIDfc�A�!*

lossA�#;�|�z       �	4JDfc�A�!*

loss�M;����       �	c�JDfc�A�!*

loss[�;�}�#       �	�MKDfc�A�!*

loss�7�;rQ3-       �	��KDfc�A�!*

loss�ۏ<j�       �	�LDfc�A�!*

loss���<���^       �	�!MDfc�A�!*

loss��=�
)       �	$�MDfc�A�!*

loss�H�<��       �	kHNDfc�A�!*

lossMw�;��?       �	��NDfc�A�!*

loss�u;<��       �	F{ODfc�A�!*

loss�؇:�̓�       �	�PDfc�A�!*

loss�N~<����       �	��PDfc�A�!*

loss�A�=�.       �	K<QDfc�A�!*

loss��Y<���       �	��QDfc�A�!*

loss}rP<��       �	{hRDfc�A�!*

loss3)�;�:C�       �	;�SDfc�A�!*

lossR��<�,R       �	6=TDfc�A�!*

loss_x<Cq-|       �	��TDfc�A�!*

loss_�;�x��       �	��UDfc�A�!*

loss�'�;m�c�       �	o)VDfc�A�!*

lossc�{9�RD�       �	��VDfc�A�!*

loss��5=mc�o       �	`rWDfc�A�!*

loss)e<F�,�       �	XDfc�A�!*

loss�*d=8���       �	��XDfc�A�!*

loss�L=a�y       �	dYDfc�A�!*

loss��<�*       �	^�ZDfc�A�!*

loss��&;O�?       �	h�[Dfc�A�!*

loss�w:/\��       �	*�\Dfc�A�!*

loss$�(;��LS       �	n1]Dfc�A�!*

loss�Na:�vq       �	��]Dfc�A�!*

lossFo=��       �	z^Dfc�A�!*

loss���;\XS       �	�_Dfc�A�!*

loss�#�:v��&       �	�_Dfc�A�!*

lossjX�;|<��       �	�Y`Dfc�A�!*

lossH>�<�q{�       �	<�`Dfc�A�!*

lossI<�l�i       �	a�aDfc�A�!*

lossa:?=�vd       �	nLbDfc�A�!*

loss���=1���       �	�bDfc�A�!*

loss� =�M�       �	l�cDfc�A�!*

lossoǔ;Ui[�       �	�8dDfc�A�!*

loss���<���^       �	�dDfc�A�!*

loss�NC<���       �	�eDfc�A�!*

loss�
^=�dh�       �	F&fDfc�A�!*

lossLE<	k�       �	�fDfc�A�!*

loss��_=��       �	�_gDfc�A�!*

loss��;A�͌       �	��gDfc�A�!*

loss��<��)�       �	�hDfc�A�!*

loss�C�;��E       �	$*iDfc�A�!*

lossd�5=TX��       �	<�iDfc�A�!*

lossL`�<[�yj       �	�YjDfc�A�!*

lossMZ�;Ѱ��       �	��jDfc�A�!*

loss�x<j�       �	<�kDfc�A�!*

loss�5<}��       �	�nlDfc�A�!*

loss�w�<�a��       �	�mDfc�A�!*

loss�tQ;���       �	t�mDfc�A�!*

lossZ��:��-p       �	/nDfc�A�!*

lossvBZ<�E��       �	�nDfc�A�!*

loss]�;����       �	XWoDfc�A�!*

loss6�<����       �	9�oDfc�A�!*

loss���<X;��       �	b�pDfc�A�!*

loss:%<�3�L       �	7qDfc�A�!*

loss��,<�ô�       �	6�qDfc�A�!*

loss�8);q��0       �	�ErDfc�A�!*

loss��x;�1|�       �	��rDfc�A�!*

lossť�9���D       �	�sDfc�A�!*

loss��s:!W�Y       �	>!tDfc�A�!*

loss1�K;>+-       �	��tDfc�A�!*

loss��;
�h       �	�SuDfc�A�!*

loss*<�Np!       �	I�uDfc�A�!*

loss3�<x��       �	�vDfc�A�!*

lossZ7�;��M�       �	�wDfc�A�!*

loss�<��'9       �	ȴwDfc�A�!*

loss/��<Z���       �	OWxDfc�A�!*

loss t�:a�&       �	x�xDfc�A�!*

loss;o�<��/       �	ߌyDfc�A�!*

loss�j ;n�       �	�zDfc�A�!*

loss�23;����       �	��zDfc�A�"*

loss��e=��xf       �	@N{Dfc�A�"*

loss���<��Om       �	��{Dfc�A�"*

loss.<D�       �	B�|Dfc�A�"*

loss��9�`@�       �	�-}Dfc�A�"*

lossF=�:���5       �	r�}Dfc�A�"*

loss�.�;�iC6       �	`Y~Dfc�A�"*

loss���=��D�       �	k�~Dfc�A�"*

loss��9�J�       �	r�Dfc�A�"*

loss)�	<�@i       �	�Dfc�A�"*

loss�t/<o��       �	/��Dfc�A�"*

losso]<�J�A       �	�[�Dfc�A�"*

lossQ"�8�yn/       �	J��Dfc�A�"*

loss�tg;ys�}       �	���Dfc�A�"*

loss�M�:����       �	��Dfc�A�"*

lossH��<�Ku�       �	���Dfc�A�"*

lossExI<X��1       �	(~�Dfc�A�"*

lossd�;!���       �	��Dfc�A�"*

loss	�,<$׻�       �	��Dfc�A�"*

loss��<��       �	iŇDfc�A�"*

loss��;Iu��       �	�i�Dfc�A�"*

loss��4<�h=r       �	F�Dfc�A�"*

loss`m<
���       �	⮉Dfc�A�"*

loss/�}:��3Q       �	S�Dfc�A�"*

loss���;t"�4       �	�Dfc�A�"*

loss�u4<�#��       �	���Dfc�A�"*

lossc)<�xH7       �	�*�Dfc�A�"*

losso(+<u�a       �	�ҌDfc�A�"*

loss;qH;�)�       �		q�Dfc�A�"*

loss�e4;�Q�       �	b�Dfc�A�"*

lossA��<��       �	X��Dfc�A�"*

lossRE�;�B       �	�H�Dfc�A�"*

loss�.�9�
0�       �	��Dfc�A�"*

lossi;"oS       �	~�Dfc�A�"*

loss��;R�       �	��Dfc�A�"*

loss9Z�<P���       �	г�Dfc�A�"*

losswM�:��x�       �	IK�Dfc�A�"*

lossS��<�In&       �	a�Dfc�A�"*

loss�:�;F��a       �	ȴ�Dfc�A�"*

loss
SK=���8       �	c{�Dfc�A�"*

loss9O�<w��       �	��Dfc�A�"*

loss�@=FT�       �	H�Dfc�A�"*

loss��:��S       �	��Dfc�A�"*

loss4's9��)       �	���Dfc�A�"*

loss�n�;����       �	��Dfc�A�"*

lossq�:[�)_       �	�~�Dfc�A�"*

loss\��;h'       �	H�Dfc�A�"*

loss<a�;D��C       �	
��Dfc�A�"*

loss��9.���       �	�Y�Dfc�A�"*

loss� �<ˮ�/       �	��Dfc�A�"*

loss��
<Ib�       �	���Dfc�A�"*

loss�B�;���$       �	}<�Dfc�A�"*

loss�;;K�*�       �	֝Dfc�A�"*

loss�f;�-�       �	�l�Dfc�A�"*

loss56�<g��       �	�	�Dfc�A�"*

loss���<�@��       �	��Dfc�A�"*

loss@��:cs�!       �	�@�Dfc�A�"*

loss8�]:*Y       �	�נDfc�A�"*

loss�� :���H       �	�n�Dfc�A�"*

loss���;����       �	[_�Dfc�A�"*

loss�@;�Q       �	��Dfc�A�"*

loss6>�;ᝠ       �	���Dfc�A�"*

loss��;DE       �	���Dfc�A�"*

loss�ߕ<\�b�       �	%w�Dfc�A�"*

loss�J=�|,
       �	��Dfc�A�"*

lossdy�=����       �	E��Dfc�A�"*

loss���<�T��       �	B_�Dfc�A�"*

lossJ�
<,�       �	P�Dfc�A�"*

loss�B9=Dٽ�       �	k��Dfc�A�"*

loss��<(�N       �	e4�Dfc�A�"*

loss��g9�0G       �	 ѩDfc�A�"*

loss�o;�{��       �	`t�Dfc�A�"*

lossc2C<yaq�       �	���Dfc�A�"*

loss�":,,Á       �	@��Dfc�A�"*

loss�¤=VF9       �	�5�Dfc�A�"*

losst!B=�?�       �	JѭDfc�A�"*

loss��_<��       �	[}�Dfc�A�"*

loss��x<T��       �	�$�Dfc�A�"*

loss��<�'�       �	�ɯDfc�A�"*

lossk`<6��A       �	�k�Dfc�A�"*

loss;�q<��1�       �	��Dfc�A�"*

lossN��;1�;       �	곱Dfc�A�"*

lossD��<E��       �	V�Dfc�A�"*

lossN�=�Pz�       �	�
�Dfc�A�"*

loss��};ד�o       �	���Dfc�A�"*

loss�N5=�HTn       �	�N�Dfc�A�"*

lossI �=�"9       �	��Dfc�A�"*

loss� <>@�       �	��Dfc�A�"*

loss�=;�j��       �	/�Dfc�A�"*

loss�dE;2���       �	�նDfc�A�"*

loss@Ѣ<k7Q�       �	~o�Dfc�A�"*

lossq�<�vZ�       �	��Dfc�A�"*

loss�?�<�F@O       �	@��Dfc�A�"*

loss t�<���       �	�:�Dfc�A�"*

loss�	�;%��       �	|ԹDfc�A�"*

loss�H<��`       �	J��Dfc�A�"*

loss�R<ېL�       �	�I�Dfc�A�"*

loss���;�(�s       �	`�Dfc�A�"*

lossROs;X!S�       �	댼Dfc�A�"*

loss�o�<�ͳ       �	E+�Dfc�A�"*

loss��m=�h�       �	XƽDfc�A�"*

loss�҈<�X�       �	�[�Dfc�A�"*

loss�%�;8���       �	3��Dfc�A�"*

loss919EwK       �	���Dfc�A�"*

loss�};����       �	�4�Dfc�A�"*

loss϶a;�+��       �	���Dfc�A�"*

lossZ:9<��N|       �	|c�Dfc�A�"*

loss_��;*��       �	���Dfc�A�"*

loss<�<f�N�       �	���Dfc�A�"*

lossJ��<jR��       �	�*�Dfc�A�"*

loss���:��+:       �	a��Dfc�A�"*

lossr��;����       �	g�Dfc�A�"*

loss�+&:<�U	       �	r��Dfc�A�"*

loss�r�;�#�       �	���Dfc�A�"*

loss�j�<�<dL       �	�8�Dfc�A�"*

loss*fr=�m48       �	���Dfc�A�"*

loss�=��        �	"p�Dfc�A�"*

loss_=�="�       �	�
�Dfc�A�"*

loss���<m�       �	���Dfc�A�"*

loss33�:��J       �	�D�Dfc�A�"*

loss���<
�f�       �	���Dfc�A�"*

lossj�z<�`�       �	 ~�Dfc�A�"*

lossFJA<���+       �	�Dfc�A�"*

loss���<R��       �	���Dfc�A�"*

lossaGF=R�k�       �	XW�Dfc�A�"*

loss�!;5��<       �	���Dfc�A�"*

loss��w9���       �	���Dfc�A�"*

loss;,><��       �	'�Dfc�A�"*

loss{k�<���       �	��Dfc�A�#*

lossE��:z�k8       �	�[�Dfc�A�#*

lossW��;{Nrp       �	��Dfc�A�#*

loss��<�O�        �	���Dfc�A�#*

loss�Ն=�&�       �	�5�Dfc�A�#*

loss�j;��L       �	��Dfc�A�#*

loss��<۶}       �	�n�Dfc�A�#*

lossҢS<��F       �	�f�Dfc�A�#*

loss��:���       �	�Dfc�A�#*

loss��d<eY��       �	���Dfc�A�#*

loss�?>;�N��       �	�F�Dfc�A�#*

loss���:�a��       �	���Dfc�A�#*

loss6׌;�o-'       �	9��Dfc�A�#*

loss:H�:Eh�       �	�>�Dfc�A�#*

loss@'�9���       �	���Dfc�A�#*

loss�4�;K,$�       �	���Dfc�A�#*

loss
�:pP��       �	�H�Dfc�A�#*

loss	S�</�<       �	��Dfc�A�#*

loss��<�ؠ}       �	��Dfc�A�#*

loss�(=��`       �	�M�Dfc�A�#*

lossh�f<�9�k       �	��Dfc�A�#*

loss8��;4}�       �	���Dfc�A�#*

loss�K<TUv�       �	�/�Dfc�A�#*

loss�<K<R�.D       �	C��Dfc�A�#*

losst݉=��v       �	�a�Dfc�A�#*

loss���<��u       �	o�Dfc�A�#*

lossD]�=��U       �	���Dfc�A�#*

loss�):��X�       �	wL�Dfc�A�#*

lossdzs<�w�6       �	���Dfc�A�#*

losshvW;Cs0       �	���Dfc�A�#*

loss[��;p�       �	��Dfc�A�#*

loss�\:���       �	���Dfc�A�#*

lossV�<2fx[       �	�L�Dfc�A�#*

lossX�=<�Ld#       �	���Dfc�A�#*

lossy<3΅�       �	|�Dfc�A�#*

loss���<��       �	#�Dfc�A�#*

loss.~3;��C�       �	 ��Dfc�A�#*

loss��=
09�       �	�A�Dfc�A�#*

loss�c=<h���       �	#��Dfc�A�#*

loss�=<>�D�       �	�l�Dfc�A�#*

loss_�;�5!       �	6�Dfc�A�#*

loss�5<�G�Z       �	���Dfc�A�#*

loss)��<�_�       �	K<�Dfc�A�#*

loss&=�;1S~�       �	@��Dfc�A�#*

loss�Ţ=��Y�       �	ђ�Dfc�A�#*

lossr�(<QC�y       �	"7�Dfc�A�#*

loss<��=P�       �	��Dfc�A�#*

lossh��;�ڿ       �	:��Dfc�A�#*

loss��|=�vP�       �	�F�Dfc�A�#*

loss���<�\��       �	���Dfc�A�#*

loss���</:       �	�o�Dfc�A�#*

loss-�<[_��       �	��Dfc�A�#*

loss���<K�%�       �	+��Dfc�A�#*

loss�;+�       �	fK�Dfc�A�#*

loss�|;�<�       �	Y��Dfc�A�#*

lossJ_�<q�t}       �	Ƌ�Dfc�A�#*

loss�G�<����       �	�"�Dfc�A�#*

loss�$�<���p       �	l>�Dfc�A�#*

loss.��<�Rw       �	=+�Dfc�A�#*

loss!l�=Ic�^       �	���Dfc�A�#*

loss$��;�8Ϭ       �	�}�Dfc�A�#*

loss��/<{xb       �	`�Dfc�A�#*

lossԓ,=����       �	��Dfc�A�#*

loss���<w���       �	/��Dfc�A�#*

loss���<A� �       �	�C�Dfc�A�#*

loss�x =�4�       �	;��Dfc�A�#*

loss�j><��'�       �	p|�Dfc�A�#*

loss�S<��       �	���Dfc�A�#*

loss��J<Idƺ       �	�c�Dfc�A�#*

lossA�=fO�       �	l�Dfc�A�#*

loss��<ykB"       �	��Dfc�A�#*

lossj{�:O�x�       �	J�Dfc�A�#*

loss�9<1�U       �	R��Dfc�A�#*

lossl�>=P        �	?��Dfc�A�#*

lossF�:�<�       �	JA Efc�A�#*

loss.��:`�       �	�� Efc�A�#*

lossIR�:� ��       �	0�Efc�A�#*

loss�G,<(3��       �	��Efc�A�#*

lossi�Y<�$xp       �	�Efc�A�#*

loss��9<��x>       �	�_Efc�A�#*

loss�z�:���*       �	
Efc�A�#*

loss�C;G3)�       �	q�Efc�A�#*

loss|��;7m       �	�hEfc�A�#*

lossWֆ=�FQ�       �	FEfc�A�#*

loss��#=�Q       �	�Efc�A�#*

loss�3�<濔N       �	q8	Efc�A�#*

loss�C�<����       �	��	Efc�A�#*

lossO��<0�)       �	�WEfc�A�#*

loss8��;m}�a       �	WEfc�A�#*

loss46 =���@       �	�Efc�A�#*

loss�|n;Z�[       �	�MEfc�A�#*

lossH2<^��       �	d�Efc�A�#*

loss�ia;����       �	�Efc�A�#*

loss5<��1       �	�3Efc�A�#*

loss#B�;�7�       �	Y�Efc�A�#*

loss ��;���_       �	��Efc�A�#*

loss�K;�5�       �	%Efc�A�#*

loss?�<uf��       �	?�Efc�A�#*

lossۥ<x?=       �	ZdEfc�A�#*

loss��<I��E       �	�Efc�A�#*

lossܜ�;���$       �	Efc�A�#*

loss%�;l��h       �	v�Efc�A�#*

loss{ڹ;��       �	�\Efc�A�#*

loss�Ԧ=`��       �	��Efc�A�#*

lossZ@^<��m       �	ӄEfc�A�#*

loss�յ;*�+       �	�.Efc�A�#*

loss(^=��l�       �	��Efc�A�#*

loss�i�<\�[A       �	>^Efc�A�#*

loss[{�<��kf       �	��Efc�A�#*

lossME�;V��       �	)�Efc�A�#*

loss�<�"       �	��Efc�A�#*

loss�@<��W&       �	bEfc�A�#*

loss�u< r@�       �	Efc�A�#*

loss�<$�p=       �	3�Efc�A�#*

lossC�=�H�1       �	{MEfc�A�#*

loss+�<e��+       �	��Efc�A�#*

loss��=&�&�       �	��Efc�A�#*

loss�oE:z�@�       �	6 Efc�A�#*

lossL��;��o�       �	�� Efc�A�#*

loss��
<2���       �	Tt!Efc�A�#*

loss��<�s�       �	�"Efc�A�#*

loss��	;�?�|       �	�"Efc�A�#*

lossli<Nd��       �	d<#Efc�A�#*

lossv�;(�C       �	��#Efc�A�#*

loss���<�̈́�       �	�o$Efc�A�#*

loss:=<Ԏ9       �	%Efc�A�#*

loss�S�=��       �	_�%Efc�A�#*

loss��;�c��       �	2V&Efc�A�#*

loss�L�<P�?       �	��&Efc�A�$*

loss�[;�
C       �	&�'Efc�A�$*

loss��
;ji7       �	6"(Efc�A�$*

loss��n=�E�       �	=�(Efc�A�$*

loss�9^=/d�"       �	 R)Efc�A�$*

loss�<��
       �	��)Efc�A�$*

lossN�\=hFw@       �	T�*Efc�A�$*

loss#=1��       �	&+Efc�A�$*

loss�a�<�n$       �	�+Efc�A�$*

loss��	=�+��       �	B\,Efc�A�$*

lossdFe<���       �	,�,Efc�A�$*

loss!�;���*       �	�-Efc�A�$*

loss�;�<�絢       �	�_.Efc�A�$*

loss,[�:�B�V       �	�/Efc�A�$*

loss�-<ƭ8O       �	
�/Efc�A�$*

loss)`;8ߛh       �	4/0Efc�A�$*

loss8�%=�n&       �	��0Efc�A�$*

loss~<��X�       �	g1Efc�A�$*

loss��;�V��       �	>2Efc�A�$*

loss��o:�&7       �	(�2Efc�A�$*

lossDA�;i��       �	/3Efc�A�$*

loss���<�Bv�       �	:<4Efc�A�$*

lossE�:V-8�       �	��4Efc�A�$*

loss��<���       �	�l5Efc�A�$*

loss=�b<�G"0       �	�6Efc�A�$*

loss��6<E�@�       �	�6Efc�A�$*

loss-C<��͜       �	�;7Efc�A�$*

lossϱ�;��5O       �	�#8Efc�A�$*

losswƨ<�Ͻ�       �	��8Efc�A�$*

loss���;!B�       �	�m9Efc�A�$*

lossp�<ο�e       �	X:Efc�A�$*

loss��;��;       �	��:Efc�A�$*

loss� =���*       �	<3;Efc�A�$*

loss�c=b�y       �	��;Efc�A�$*

losst�y;�L       �	dt<Efc�A�$*

loss���;H       �	�=Efc�A�$*

lossL�=C�D       �	.>Efc�A�$*

loss�L�;0�       �	q�>Efc�A�$*

lossr�<V�5�       �	l?Efc�A�$*

loss}kv<�       �	�@Efc�A�$*

loss�v�;�9�+       �	m�@Efc�A�$*

loss
�<�z#�       �	JFAEfc�A�$*

loss���<�&�       �	��AEfc�A�$*

loss���<�h66       �	�BEfc�A�$*

loss��;ˆ/       �	CEfc�A�$*

lossL�<e��9       �	��CEfc�A�$*

lossƤ.=�(��       �	l\DEfc�A�$*

loss�u <���       �	��DEfc�A�$*

loss/8�<����       �	߉EEfc�A�$*

loss8�v:��9�       �	� FEfc�A�$*

loss`=�<���}       �	p�FEfc�A�$*

lossX�<G�m       �	jjGEfc�A�$*

loss��;�_&�       �	�HEfc�A�$*

loss�G�9ԛ       �	�HEfc�A�$*

lossc��:�P��       �	�7IEfc�A�$*

loss΀�9�\n8       �	�JEfc�A�$*

lossni�=�KG�       �	�XKEfc�A�$*

lossί�;���       �	��KEfc�A�$*

loss�� <r'��       �	��LEfc�A�$*

lossɱ�<*廀       �	�MEfc�A�$*

lossŘ�:�r�R       �	�NEfc�A�$*

loss�d};�#�       �	� OEfc�A�$*

loss��d;P���       �	ԶOEfc�A�$*

loss�'=��8\       �	4LPEfc�A�$*

loss��*<��       �	�PEfc�A�$*

loss)�=<J�.       �	�zQEfc�A�$*

loss�8==� ,       �	[$REfc�A�$*

loss��;E�Mc       �	�REfc�A�$*

loss�6<��,       �	�_SEfc�A�$*

loss���9��d       �	��SEfc�A�$*

loss�K�<r>�,       �	��TEfc�A�$*

loss���:�8       �	��UEfc�A�$*

loss�e�;֊��       �	�VEfc�A�$*

lossm��<�s(.       �	�VEfc�A�$*

loss�<��|g       �	x%XEfc�A�$*

loss�ޅ:B�RP       �	��XEfc�A�$*

loss���<� �       �	�ZEfc�A�$*

loss�^Y=���       �	 :[Efc�A�$*

loss$W<���       �	��[Efc�A�$*

loss�o(:��       �	��\Efc�A�$*

loss�eG=���       �	^�]Efc�A�$*

losslY�;�_�       �	.�^Efc�A�$*

lossJ�c=�       �	�._Efc�A�$*

loss{1�<���       �	��_Efc�A�$*

loss�Q:]��       �	�`Efc�A�$*

loss،O<w�P�       �	~aEfc�A�$*

loss�9�:&NJ       �	M�aEfc�A�$*

loss@�h<D��       �	�bbEfc�A�$*

loss�Ԏ<(�9       �	��bEfc�A�$*

loss4��=V�o       �	.�cEfc�A�$*

lossd�T;qȵ       �	OYdEfc�A�$*

lossi<}nu�       �	keEfc�A�$*

loss�=xz�a       �	�eEfc�A�$*

lossΠG;���       �	!XfEfc�A�$*

lossT�N:��L�       �	f�fEfc�A�$*

loss��<�A�       �	ݗgEfc�A�$*

lossW�;�V�       �	�9hEfc�A�$*

loss�`�=ϼ�e       �	0�hEfc�A�$*

loss�_�;\O��       �	�miEfc�A�$*

loss*�;)��\       �	�jEfc�A�$*

loss�=K��S       �	��jEfc�A�$*

lossA�=���2       �	{KkEfc�A�$*

loss���<�n�>       �	��kEfc�A�$*

loss:m:�l�       �	��lEfc�A�$*

loss��;Z�[       �	�(mEfc�A�$*

loss�>�<��,J       �	��mEfc�A�$*

loss a<Sծ1       �	5anEfc�A�$*

lossF0�<��+       �	�oEfc�A�$*

lossv=�8fT       �	ƨoEfc�A�$*

loss�<=��       �	�KpEfc�A�$*

loss;�J\       �	��pEfc�A�$*

loss��)<s��z       �	�qEfc�A�$*

loss��;��o       �	}^rEfc�A�$*

lossI�:��oA       �	; sEfc�A�$*

loss�J�;=��       �	��sEfc�A�$*

loss/�:Ɍ��       �	��tEfc�A�$*

loss��<UU��       �	JCuEfc�A�$*

loss�'�;��ڴ       �	��uEfc�A�$*

loss���<ƌ�       �	[zvEfc�A�$*

loss�g�<#/.�       �	nwEfc�A�$*

lossv��<�h�       �	��wEfc�A�$*

loss
c=��>J       �	$DxEfc�A�$*

lossCb�;�2�{       �	��xEfc�A�$*

loss��!<:M�       �	��yEfc�A�$*

loss\.3=U�       �	n1zEfc�A�$*

loss ��:��j       �	��zEfc�A�$*

loss�?;1��       �	
f{Efc�A�$*

loss̅=;F_|       �	��{Efc�A�$*

loss��9z=+�       �	W�|Efc�A�%*

lossq^8:4�F�       �	�*}Efc�A�%*

loss�*�:�mr       �	"�}Efc�A�%*

loss�r�:���       �	�X~Efc�A�%*

loss-��;|��k       �	{�~Efc�A�%*

lossڑ3<�20�       �	��Efc�A�%*

loss ��9����       �	B$�Efc�A�%*

loss6�8\ �       �	��Efc�A�%*

loss�G<R�       �	_^�Efc�A�%*

loss�<���       �	���Efc�A�%*

loss)�<<�8       �	���Efc�A�%*

loss�a�8k�       �	�8�Efc�A�%*

loss|�+=4�j)       �	x҃Efc�A�%*

loss�!=����       �	�o�Efc�A�%*

lossZ��:���       �	m�Efc�A�%*

lossԦ�=�1�       �	_��Efc�A�%*

lossU�<��o(       �	/2�Efc�A�%*

lossJ3�<�]�s       �	�؆Efc�A�%*

lossv�=a�       �	�z�Efc�A�%*

lossA��;Nڭ�       �	��Efc�A�%*

lossv#�;����       �	2��Efc�A�%*

lossl�;���       �	�C�Efc�A�%*

loss-�;<c�i5       �	މEfc�A�%*

loss��(=(>K       �	lx�Efc�A�%*

lossnE5<��{j       �	�Efc�A�%*

loss�;�<e�       �	���Efc�A�%*

lossNJ5<�U��       �	 T�Efc�A�%*

loss��b<��ܭ       �	8�Efc�A�%*

loss�h�; `       �	Efc�A�%*

loss�g<�m�       �	 (�Efc�A�%*

lossV^�<��e�       �	&ŎEfc�A�%*

loss���;��B       �	�`�Efc�A�%*

loss6=�c�l       �	���Efc�A�%*

loss7�p<7��       �	���Efc�A�%*

lossH�`<�d�2       �	n0�Efc�A�%*

loss�S�<%=�       �	�ǑEfc�A�%*

loss3�<�x��       �	q�Efc�A�%*

lossTQ�;����       �	��Efc�A�%*

lossT�<Ey�       �	Ǻ�Efc�A�%*

lossF�$;yd       �	W�Efc�A�%*

loss��7<��C       �	��Efc�A�%*

loss}�<R�Ow       �	��Efc�A�%*

loss�H;=����       �	���Efc�A�%*

loss#�<E���       �	�a�Efc�A�%*

lossV<'��       �	��Efc�A�%*

loss͹t<�1w        �	���Efc�A�%*

loss�_;���h       �	]S�Efc�A�%*

loss��{;k3�       �	���Efc�A�%*

loss�Ei;��}�       �	d��Efc�A�%*

loss��4<�E       �	.:�Efc�A�%*

loss��<eO       �	ݛEfc�A�%*

loss�+=W�m       �	ϜEfc�A�%*

lossAH =Z�:       �	�h�Efc�A�%*

loss�Y�:{���       �	%�Efc�A�%*

loss.Zg:�fz       �	?��Efc�A�%*

lossm��<5�       �	�E�Efc�A�%*

loss��<�U�f       �	��Efc�A�%*

loss��I;�5 )       �	8��Efc�A�%*

loss�D�:�q�*       �	R*�Efc�A�%*

loss(S	;�-
       �	�ɡEfc�A�%*

loss�=f��       �	�i�Efc�A�%*

loss���:�gȠ       �	6�Efc�A�%*

loss/�=;�7       �	���Efc�A�%*

loss@G�;�[�       �	rO�Efc�A�%*

lossߩ$;��4_       �	��Efc�A�%*

loss@��:��
       �		��Efc�A�%*

loss��:�u=       �	���Efc�A�%*

lossc�<�̙       �	C�Efc�A�%*

loss�j<jB�[       �	>��Efc�A�%*

lossDe�;rU�G       �	��Efc�A�%*

loss�E; ��M       �	3�Efc�A�%*

loss,~n<���       �	c��Efc�A�%*

lossJ��<5^�c       �	x�Efc�A�%*

loss��g=mm��       �	i�Efc�A�%*

lossC�=b'�       �	��Efc�A�%*

loss�<�:��3z       �	J��Efc�A�%*

lossA5�;�V�       �	�Y�Efc�A�%*

loss�]�<"M��       �	��Efc�A�%*

loss���:MY��       �	m��Efc�A�%*

lossh�U<r{�|       �	�O�Efc�A�%*

loss?��<t�޺       �	��Efc�A�%*

loss�D�8]-�       �	���Efc�A�%*

lossIv,<I6�       �	�9�Efc�A�%*

loss*�;�Z��       �	���Efc�A�%*

loss*�<P��q       �	5}�Efc�A�%*

loss1L<)tW�       �	� �Efc�A�%*

lossX�W=O�Dt       �	���Efc�A�%*

loss׽�:F���       �	���Efc�A�%*

loss��=�m)       �	a7�Efc�A�%*

loss���:��	       �	^��Efc�A�%*

lossl��<�y{�       �	�y�Efc�A�%*

loss_<��V       �	}$�Efc�A�%*

loss/�4<��<�       �	���Efc�A�%*

lossʬ�;d���       �	3k�Efc�A�%*

loss��(=�5�h       �	��Efc�A�%*

loss&�;��F       �	���Efc�A�%*

loss��g=|/��       �	dY�Efc�A�%*

loss4��<�2�       �	D�Efc�A�%*

lossAi=%#��       �	���Efc�A�%*

loss�+|;|͠u       �	���Efc�A�%*

loss�dq=�_       �	���Efc�A�%*

loss�A�;���       �	I�Efc�A�%*

lossS8<��       �	}��Efc�A�%*

loss��;DcY       �	C��Efc�A�%*

loss/�=��J�       �	�)�Efc�A�%*

loss\<.��       �	G��Efc�A�%*

lossq׬:uQL�       �	Eg�Efc�A�%*

loss�,�:�^\       �	��Efc�A�%*

loss��;U�e       �		��Efc�A�%*

loss�d�;�6�       �	$��Efc�A�%*

lossR;"<�x       �	M2�Efc�A�%*

loss��N<����       �	��Efc�A�%*

loss�j@;�O       �	�_�Efc�A�%*

loss��/:�:��       �	���Efc�A�%*

loss L�90�E�       �	u��Efc�A�%*

loss�:��       �	,-�Efc�A�%*

losse�e<2&n�       �	���Efc�A�%*

loss���:���       �	y�Efc�A�%*

loss��<��$�       �	��Efc�A�%*

lossd�:>0J�       �	C��Efc�A�%*

lossƞ�<�C*       �	�E�Efc�A�%*

loss�n2;1���       �	o��Efc�A�%*

lossh-�;�R�c       �	P��Efc�A�%*

loss��K=8��       �	�5�Efc�A�%*

loss�V�<�rFV       �	�(�Efc�A�%*

lossƓ�=�YZ       �	2��Efc�A�%*

lossF�*;m]��       �	�i�Efc�A�%*

loss�Lh<�2��       �	�Efc�A�%*

loss9q<L3�       �	���Efc�A�&*

loss�d�;�\G�       �	!@�Efc�A�&*

lossh1<t�fA       �	{��Efc�A�&*

loss��=C�       �	Lk�Efc�A�&*

lossv��;V��       �	�Efc�A�&*

loss"��=�_!�       �	���Efc�A�&*

lossNe<RID�       �	��Efc�A�&*

loss=��<����       �	���Efc�A�&*

loss��<��       �	X�Efc�A�&*

lossv�/=N�[       �	��Efc�A�&*

loss��3;���       �	Ja�Efc�A�&*

loss(��;�:F       �	&�Efc�A�&*

lossv�<����       �	F��Efc�A�&*

lossj��;�v�       �	���Efc�A�&*

loss�q�:�i@       �	�?�Efc�A�&*

loss�k�<_3�       �	���Efc�A�&*

loss���<�h�       �	9��Efc�A�&*

lossye=B��       �	2�Efc�A�&*

loss`=	v��       �	���Efc�A�&*

loss�&�<�3��       �	�c�Efc�A�&*

loss=<[<��X�       �	2Z�Efc�A�&*

loss%G�:A��       �	j��Efc�A�&*

loss���<仱u       �	��Efc�A�&*

loss�^0=LO�       �	M,�Efc�A�&*

loss�+!=Қ�       �	���Efc�A�&*

loss8{K;vF+�       �	�\�Efc�A�&*

loss��<
���       �	�p�Efc�A�&*

loss[6�<g}��       �	� Ffc�A�&*

loss�!<E��       �	S� Ffc�A�&*

loss��<� �k       �	�RFfc�A�&*

loss\��<���-       �	��Ffc�A�&*

lossAs=�dO       �	t�Ffc�A�&*

loss)�9�$^J       �	�;Ffc�A�&*

lossA@�<���t       �	M�Ffc�A�&*

loss�Ю<+�D{       �	[AFfc�A�&*

loss@�@=��)�       �	��Ffc�A�&*

loss!ѡ<�Tj_       �	�Ffc�A�&*

loss	�J;$C�       �	�%Ffc�A�&*

loss�@;j�       �	�Ffc�A�&*

loss��<	U}<       �	�W	Ffc�A�&*

losssT<=vJo�       �	0L
Ffc�A�&*

loss�=Z�~�       �	G�
Ffc�A�&*

loss��q=�<;�       �	T�Ffc�A�&*

loss-Z�:�7        �	�*Ffc�A�&*

loss��:��       �	L�Ffc�A�&*

loss;��=x5�&       �	�dFfc�A�&*

lossW(A<{��D       �	. Ffc�A�&*

loss�C =O�O0       �	�Ffc�A�&*

lossW]k<ާ��       �	�;Ffc�A�&*

loss�N=R^=
       �	(�Ffc�A�&*

loss*�x:v}G       �	keFfc�A�&*

loss��<�輑       �	��Ffc�A�&*

loss: :_fa�       �	q�Ffc�A�&*

loss�=��e�       �	p%Ffc�A�&*

loss���=s}_       �	�Ffc�A�&*

loss	�<3�/       �	;RFfc�A�&*

loss�d<՚sl       �	��Ffc�A�&*

loss���;v�        �	 {Ffc�A�&*

lossW��:�U       �	�Ffc�A�&*

loss#s<�
�+       �	v�Ffc�A�&*

lossv#�;�7�       �	IFfc�A�&*

loss�[�:�[16       �	��Ffc�A�&*

loss��<$��k       �	b�Ffc�A�&*

loss	�N=��(s       �	w�Ffc�A�&*

loss*�=�3�       �	M�Ffc�A�&*

loss|��<y e       �	��Ffc�A�&*

loss��;1X��       �	>xFfc�A�&*

loss}c0<<c       �	�Ffc�A�&*

lossE:B=�rjG       �	��Ffc�A�&*

loss2j =�Ȥ       �	�DFfc�A�&*

lossr�"<=O��       �	��Ffc�A�&*

loss%�z<��2s       �	�qFfc�A�&*

lossٿ <��       �	�	Ffc�A�&*

loss{5&<���d       �	8 Ffc�A�&*

loss���:k[�       �	�� Ffc�A�&*

loss�n_<e�\       �	N|!Ffc�A�&*

lossC�:/Vi>       �	""Ffc�A�&*

loss�jR;�H�u       �	d�"Ffc�A�&*

loss���;�G       �	#N#Ffc�A�&*

lossZ�h<���       �	��#Ffc�A�&*

loss{�<PA�0       �	q�$Ffc�A�&*

lossX�;+d��       �	k-%Ffc�A�&*

loss3�:�^-L       �	��%Ffc�A�&*

lossm�<w�ڲ       �	[&Ffc�A�&*

loss�C�:����       �	�&Ffc�A�&*

loss��;��?       �	��'Ffc�A�&*

loss��;��ݒ       �	�6(Ffc�A�&*

loss���:�       �	!�(Ffc�A�&*

loss���<~��v       �	3j)Ffc�A�&*

loss��=�㡾       �	�*Ffc�A�&*

loss��
<��2       �	�*Ffc�A�&*

loss��q=J��       �	�^+Ffc�A�&*

loss���8���A       �	,Ffc�A�&*

loss�L9���d       �	��,Ffc�A�&*

loss<��;�/�       �	�B-Ffc�A�&*

loss)�x;���w       �	+�-Ffc�A�&*

loss��	<r��P       �	(~.Ffc�A�&*

loss��<���4       �	u"/Ffc�A�&*

loss���:�ؓ(       �	X�/Ffc�A�&*

loss'�=���-       �	$b0Ffc�A�&*

lossV��9 ��7       �	��0Ffc�A�&*

loss��<o���       �	ʣ1Ffc�A�&*

lossV��:��;       �	�<2Ffc�A�&*

loss3bM:�       �	��2Ffc�A�&*

losseB�:(J?       �	c|3Ffc�A�&*

loss�
�;TO       �	4Ffc�A�&*

loss�uc;ϔH�       �	�4Ffc�A�&*

loss,H<���       �	�<5Ffc�A�&*

lossD��<7c��       �	c�5Ffc�A�&*

loss\!8<n�'�       �	6Ffc�A�&*

loss���:x�P�       �	7Ffc�A�&*

loss�I:{m?       �	/�7Ffc�A�&*

loss�L�;3�e�       �	�m8Ffc�A�&*

loss��=oC~       �	u9Ffc�A�&*

lossÊ�<n�6�       �	��9Ffc�A�&*

loss���;�>       �	�:Ffc�A�&*

loss�P\:���       �	�r;Ffc�A�&*

loss��8�K��       �	<Ffc�A�&*

loss�U�;����       �	��<Ffc�A�&*

losshs�;�L�d       �	�B=Ffc�A�&*

loss͚�:�T��       �	��=Ffc�A�&*

loss?�:'��<       �	�_?Ffc�A�&*

loss��<L�?�       �	/@Ffc�A�&*

lossۣ=�y0q       �	��@Ffc�A�&*

lossDoi;��I       �	TRAFfc�A�&*

loss,��<'B�       �	�AFfc�A�&*

loss��<	z�t       �	L�BFfc�A�&*

losse��<��>�       �	�&CFfc�A�&*

lossa��;��       �	LDFfc�A�'*

loss���<Z�Q�       �	�DFfc�A�'*

loss��n;��Q       �	�NEFfc�A�'*

loss#G?9Ar��       �	FFfc�A�'*

loss	��:*�;�       �	i�FFfc�A�'*

loss��	<�wU)       �	�dGFfc�A�'*

loss-l�=l�       �	�HFfc�A�'*

loss�?=�$R�       �	f�HFfc�A�'*

loss��<>`�N       �	VFIFfc�A�'*

loss�і<0�N�       �	|�IFfc�A�'*

loss�<'`�       �	&�JFfc�A�'*

losso�<U�8}       �	�(KFfc�A�'*

lossd�~9G�+�       �	��KFfc�A�'*

loss��<�{�       �	�[LFfc�A�'*

loss��h<6y �       �	��LFfc�A�'*

loss��F=�e��       �	O�MFfc�A�'*

loss�Q=�rO       �	f0NFfc�A�'*

loss-�<@fP�       �	��NFfc�A�'*

lossl�=<Xe       �	�[OFfc�A�'*

lossl}�<,�c�       �	N�OFfc�A�'*

loss���:���       �	��PFfc�A�'*

loss�P�<��@�       �	��RFfc�A�'*

loss�,|<N�-�       �	�SFfc�A�'*

loss���<�x�       �	�TFfc�A�'*

lossFa�<�"ZM       �	��TFfc�A�'*

loss�8S=�4=2       �	;TUFfc�A�'*

lossdw?=
��d       �	��UFfc�A�'*

loss)�=K�r       �	��VFfc�A�'*

loss1$1<�       �	5zWFfc�A�'*

loss�";���       �	XFfc�A�'*

loss�-�:t)�       �	��XFfc�A�'*

lossǈ<�.]       �	]OYFfc�A�'*

loss���;���Y       �	=�YFfc�A�'*

lossT{�=#
u       �	��ZFfc�A�'*

lossWk�<ꕪ=       �	-[Ffc�A�'*

loss(��=a�Q#       �	��[Ffc�A�'*

loss\}p<]+       �	�r\Ffc�A�'*

loss�)�:��\i       �	]Ffc�A�'*

loss�H<�7ch       �	_�]Ffc�A�'*

loss/n�;�Ʃ�       �	�Z^Ffc�A�'*

loss���;G��S       �	/�^Ffc�A�'*

loss慑;`;>�       �	��_Ffc�A�'*

loss�3[;9��w       �	�R`Ffc�A�'*

lossZ`�<r�       �	I�`Ffc�A�'*

loss웂=%�a�       �	��aFfc�A�'*

loss ��;�[�       �	�rbFfc�A�'*

lossq�<�J��       �	[�cFfc�A�'*

loss�#b<ޤN       �	&RdFfc�A�'*

loss|+;0�!       �	��dFfc�A�'*

loss�< wp�       �	��eFfc�A�'*

loss�}n<Mq�       �	n2fFfc�A�'*

lossf�<[D��       �	��fFfc�A�'*

loss��<�ϣ�       �	
fgFfc�A�'*

loss��q<y/f�       �	��gFfc�A�'*

loss/]�<��{�       �	��hFfc�A�'*

loss#�<]#��       �	�)iFfc�A�'*

loss�_<�K(�       �	��iFfc�A�'*

loss]��;��y8       �	�[jFfc�A�'*

lossq7=h�       �	��jFfc�A�'*

lossS�D;�y�       �	�kFfc�A�'*

lossA �<�_c       �	�.lFfc�A�'*

loss�T<��b       �	��lFfc�A�'*

loss��:��       �	�omFfc�A�'*

loss�x�8*�i       �	{nFfc�A�'*

loss�K�:�UP�       �	B�nFfc�A�'*

loss��<�R<       �	.�oFfc�A�'*

loss�W;�{��       �	�xpFfc�A�'*

loss	�a<A��@       �	�qFfc�A�'*

loss�t<��Y�       �	E�qFfc�A�'*

loss��:VY       �	?UrFfc�A�'*

loss|R�:Ѯ�       �	��rFfc�A�'*

loss<��:%-Q       �	؂sFfc�A�'*

loss!S�:�V�       �	�tFfc�A�'*

loss�2�=��O       �	o�uFfc�A�'*

loss8*b;:�       �	��vFfc�A�'*

loss*Y<Yl(�       �	�DwFfc�A�'*

loss�m<D��       �	��wFfc�A�'*

loss��	=K�۪       �	��xFfc�A�'*

loss�2�<%�]�       �	��yFfc�A�'*

loss���<���       �	w+zFfc�A�'*

loss���<Z��6       �	w�zFfc�A�'*

loss�%�:={�       �	YP{Ffc�A�'*

loss�"=*�X�       �	��{Ffc�A�'*

loss�A�;i>�L       �	�|Ffc�A�'*

lossZ�+<s��       �	["}Ffc�A�'*

loss;�?<���       �	��}Ffc�A�'*

lossA�8:��       �	�d~Ffc�A�'*

loss��<�E       �	� Ffc�A�'*

loss@I"=[Pz       �	��Ffc�A�'*

lossD�<;�ƫ�       �	v2�Ffc�A�'*

losslt=� �c       �	�рFfc�A�'*

loss�(@;�c�       �	�k�Ffc�A�'*

loss�<���W       �	�	�Ffc�A�'*

lossCĝ< �?�       �	b��Ffc�A�'*

loss�m�;휤T       �	�8�Ffc�A�'*

loss<�<5���       �	ԃFfc�A�'*

lossY��:��;�       �	@i�Ffc�A�'*

lossN�(<�UP       �	�Ffc�A�'*

loss];^��}       �	⫅Ffc�A�'*

loss�/�:��_�       �	L�Ffc�A�'*

loss�\�<X�Tl       �	��Ffc�A�'*

loss*=�A'�       �	�~�Ffc�A�'*

lossT��9�S       �	��Ffc�A�'*

loss�=<�0��       �	Ffc�A�'*

loss�|�<Yi��       �	�]�Ffc�A�'*

lossc]�;�Ķ�       �	���Ffc�A�'*

loss�{�<ʆ�C       �	J��Ffc�A�'*

loss��<���       �	{1�Ffc�A�'*

loss�=UE�v       �	 ƋFfc�A�'*

loss�T0:K��       �	�e�Ffc�A�'*

losscQ�;�,d�       �	��Ffc�A�'*

loss��3;O���       �	n��Ffc�A�'*

loss!ȯ:�/B�       �	S?�Ffc�A�'*

lossw7}:�PM       �	EڎFfc�A�'*

lossn� >e��%       �	�q�Ffc�A�'*

lossvH<Eľ�       �	��Ffc�A�'*

loss&�<��       �	���Ffc�A�'*

loss4�<?�       �	4M�Ffc�A�'*

loss�<)�       �	��Ffc�A�'*

lossi�Z<c��9       �	���Ffc�A�'*

lossߩb;��f.       �	/0�Ffc�A�'*

loss�c=����       �	�ƓFfc�A�'*

loss\�;�>�       �	0b�Ffc�A�'*

loss�z�;�?II       �	J�Ffc�A�'*

loss w�<�jK�       �	c��Ffc�A�'*

lossNi;5,       �	�m�Ffc�A�'*

loss��F;�@.L       �	�P�Ffc�A�'*

loss�"<d       �	��Ffc�A�'*

loss! <8��n       �	2�Ffc�A�(*

loss��q;��'w       �	ӡ�Ffc�A�(*

loss�nY=��x       �	���Ffc�A�(*

losse�m=�JO       �	'��Ffc�A�(*

lossV��9�zU       �	�Ffc�A�(*

loss���;�:Sf       �	���Ffc�A�(*

loss���;��ߥ       �	�(�Ffc�A�(*

loss-��<eƿ       �	�a�Ffc�A�(*

lossԏv<m��       �	��Ffc�A�(*

loss�G;p�V�       �	!!�Ffc�A�(*

losst�=3��4       �	��Ffc�A�(*

loss��;�\��       �	���Ffc�A�(*

loss�Q:(!
       �	1B�Ffc�A�(*

loss���9��ߨ       �	V�Ffc�A�(*

loss
�;SYD       �	gF�Ffc�A�(*

loss�(�<��U3       �	V�Ffc�A�(*

loss�9f;y�8/       �	�ΦFfc�A�(*

loss�S,=�d;�       �	�r�Ffc�A�(*

loss{��;��n0       �	o�Ffc�A�(*

loss;��=��d�       �	7��Ffc�A�(*

loss�%;�`��       �	�=�Ffc�A�(*

loss
r;�'�       �	۩Ffc�A�(*

loss�n5;&m<c       �	?r�Ffc�A�(*

loss1�<�l�       �	9�Ffc�A�(*

lossTګ;�+��       �	2��Ffc�A�(*

loss�=^RA3       �	�V�Ffc�A�(*

loss���;�"'
       �	��Ffc�A�(*

loss?��:��M.       �	o��Ffc�A�(*

loss���<K�Iu       �	��Ffc�A�(*

lossn�=�G�a       �	���Ffc�A�(*

lossJ�;�       �	�N�Ffc�A�(*

loss��:*�       �	!�Ffc�A�(*

loss��:a��V       �	��Ffc�A�(*

loss`��=��B�       �	T�Ffc�A�(*

loss�\�;��3�       �	K"�Ffc�A�(*

loss�φ=��1       �	༲Ffc�A�(*

loss�_�;�d�        �	t��Ffc�A�(*

lossK!=S��       �	�8�Ffc�A�(*

loss�R�;�3�       �	eߴFfc�A�(*

lossCŘ:���       �	���Ffc�A�(*

loss�C�=�*�       �	�ֶFfc�A�(*

loss)��<�b       �	�v�Ffc�A�(*

lossC��:���       �	��Ffc�A�(*

loss�]c;m&L       �	�¸Ffc�A�(*

loss�hy=���       �	Vb�Ffc�A�(*

lossWI<�~��       �	%�Ffc�A�(*

loss�[=��j�       �	Ʀ�Ffc�A�(*

lossM�!<����       �	�F�Ffc�A�(*

loss"͔:�N�-       �	��Ffc�A�(*

loss��<@I[#       �	 ~�Ffc�A�(*

loss��=B�       �	��Ffc�A�(*

loss�1%=þ       �	��Ffc�A�(*

loss���;��       �	�L�Ffc�A�(*

loss�<C<L.�       �	;�Ffc�A�(*

loss��;�7�       �	��Ffc�A�(*

loss�`�<>��v       �	�-�Ffc�A�(*

lossmZ=c�S�       �	���Ffc�A�(*

loss��<ű)�       �	�n�Ffc�A�(*

loss���<B��       �	��Ffc�A�(*

loss�;���       �	���Ffc�A�(*

loss��g;}�	       �	zU�Ffc�A�(*

lossr;� ��       �	���Ffc�A�(*

loss�\�<�k$&       �	��Ffc�A�(*

lossv�= �W�       �	�-�Ffc�A�(*

loss�|{=~�IX       �	���Ffc�A�(*

loss���:�L�3       �	Um�Ffc�A�(*

loss%�(<�Ŷ       �	�Ffc�A�(*

loss��<��[�       �	��Ffc�A�(*

loss�n=�#�;       �	yY�Ffc�A�(*

lossx��;M���       �	 �Ffc�A�(*

lossPs<�/�Z       �	���Ffc�A�(*

loss�qu<
�R�       �	�X�Ffc�A�(*

loss��<s       �	� �Ffc�A�(*

loss�R�;���       �	��Ffc�A�(*

loss�	=��       �	]��Ffc�A�(*

loss�S;3��K       �	�j�Ffc�A�(*

loss���;(27       �	�Ffc�A�(*

loss .�:��       �	x��Ffc�A�(*

lossƾ:���       �	�X�Ffc�A�(*

loss�S;��/       �	!!�Ffc�A�(*

loss�mC;��X�       �	���Ffc�A�(*

lossh��<��g       �	�\�Ffc�A�(*

loss{Ӹ:L��'       �	���Ffc�A�(*

loss��W=�t
       �	���Ffc�A�(*

loss�=�;���-       �	�.�Ffc�A�(*

loss�|c<dO�>       �	��Ffc�A�(*

loss��;�%��       �	�a�Ffc�A�(*

loss	A�<��6�       �	���Ffc�A�(*

loss�c�;D��       �	;��Ffc�A�(*

loss��;h�       �	�'�Ffc�A�(*

loss�	�<W0��       �	���Ffc�A�(*

loss�
�<D�p{       �	t\�Ffc�A�(*

lossڜ(=��       �	���Ffc�A�(*

lossO=۴M�       �		��Ffc�A�(*

loss�	a<���>       �	!"�Ffc�A�(*

loss_h[<+T Z       �	-��Ffc�A�(*

loss)\&=�B�.       �	#j�Ffc�A�(*

loss}�9�4�       �	>�Ffc�A�(*

loss��u:؃��       �	��Ffc�A�(*

loss{a�<6�!�       �	�9�Ffc�A�(*

loss��<�ov       �	9��Ffc�A�(*

lossN?�;)z��       �	���Ffc�A�(*

loss-?�<F�x�       �	,-�Ffc�A�(*

loss�O:۩LM       �	���Ffc�A�(*

loss�q�=�B�       �	�U�Ffc�A�(*

loss�t�=ڡ�t       �	���Ffc�A�(*

lossĵ="͖�       �	��Ffc�A�(*

loss��!<<f�O       �	6�Ffc�A�(*

loss�i$<���J       �	f��Ffc�A�(*

loss��M<���j       �	�i�Ffc�A�(*

loss���<�1L�       �	��Ffc�A�(*

loss`�<5�       �	R��Ffc�A�(*

loss�#;y�*       �	�4�Ffc�A�(*

loss�1Y<6ӣS       �	���Ffc�A�(*

loss���<��k�       �	�f�Ffc�A�(*

loss�g�=I�P       �	���Ffc�A�(*

loss\<f��d       �	���Ffc�A�(*

loss#��<��8�       �	�)�Ffc�A�(*

lossBڃ<��s>       �	���Ffc�A�(*

loss�t<�r��       �	d�Ffc�A�(*

lossb"<4o��       �	� �Ffc�A�(*

lossf��<�4�       �	���Ffc�A�(*

loss���<�O�       �	&5�Ffc�A�(*

loss�/=�]	�       �	���Ffc�A�(*

losse�b==d�       �	vo�Ffc�A�(*

loss�#�;���       �	��Ffc�A�(*

loss��E<�s�       �	���Ffc�A�(*

lossV�P=%��       �	VI�Ffc�A�(*

lossI<i��        �	���Ffc�A�)*

loss)�;>��       �	(��Ffc�A�)*

loss��;Q `       �	`!�Ffc�A�)*

loss���=�#�       �	
��Ffc�A�)*

loss�	>;��9L       �	Ed�Ffc�A�)*

loss*&�;H���       �	>	�Ffc�A�)*

loss�A�;?n       �	���Ffc�A�)*

lossh�%>r�7       �	�E�Ffc�A�)*

lossٳ�<��n�       �	&��Ffc�A�)*

loss��V<����       �	S{�Ffc�A�)*

loss�-�<&��       �	;��Ffc�A�)*

loss�5�;���       �	Փ�Ffc�A�)*

loss��;��w       �	�2�Ffc�A�)*

lossm2<*%�d       �	���Ffc�A�)*

lossN��:Zgql       �	Oy�Ffc�A�)*

lossI�9���       �	��Ffc�A�)*

loss#�+>���`       �	���Ffc�A�)*

lossL�;��       �	eU�Ffc�A�)*

loss��9�G��       �	���Ffc�A�)*

loss!n�;��a�       �	q��Ffc�A�)*

loss@vZ<� $       �	
.�Ffc�A�)*

loss�̙<;?       �	.��Ffc�A�)*

lossq�M9y��p       �	�b�Ffc�A�)*

loss�+�=y7�       �	�"�Ffc�A�)*

loss�=z:��y       �	4��Ffc�A�)*

lossP�='��*       �	�U�Ffc�A�)*

loss$�;��       �	���Ffc�A�)*

loss @�:,��       �	��Ffc�A�)*

loss��;���D       �	D4 Gfc�A�)*

loss� �:�^2       �	�� Gfc�A�)*

losse�5<�G��       �	�|Gfc�A�)*

loss5â;��       �	W#Gfc�A�)*

loss��<�1�'       �	��Gfc�A�)*

lossľ�;�'��       �	cGfc�A�)*

loss�{;P>��       �	Gfc�A�)*

loss�X;�\YX       �	�Gfc�A�)*

lossDi�=��ҳ       �	?Gfc�A�)*

lossL��=�Ө       �	��Gfc�A�)*

loss��;W��q       �	��Gfc�A�)*

loss9$!;;��       �	�?Gfc�A�)*

loss�]=��       �	-�Gfc�A�)*

loss???<�xY�       �	1�Gfc�A�)*

loss)�9<�N�$       �	=C	Gfc�A�)*

lossj�w;d�D�       �	��	Gfc�A�)*

loss�U(;Etġ       �	�
Gfc�A�)*

loss�0;hh^�       �	Q2Gfc�A�)*

loss{��;���x       �	"Gfc�A�)*

loss���:�4+       �	j�Gfc�A�)*

loss"�:Q3       �	hGfc�A�)*

loss �@<�Mn�       �	�Gfc�A�)*

lossC
<, N�       �	*�Gfc�A�)*

loss6[�;�	�       �	PGfc�A�)*

loss�Z.=x)1       �	I�Gfc�A�)*

lossMʺ<���3       �	x�Gfc�A�)*

loss���<�>2       �	�IGfc�A�)*

loss
5�<��0�       �	p�Gfc�A�)*

loss��<OF       �	~�Gfc�A�)*

lossH6�;��       �	�$Gfc�A�)*

loss��<���       �	��Gfc�A�)*

lossO�<���       �	S]Gfc�A�)*

loss�ce<����       �	��Gfc�A�)*

lossE�;���       �	.�Gfc�A�)*

lossDN=<�@�R       �	��Gfc�A�)*

loss�;~yݘ       �	�Gfc�A�)*

loss��<����       �	��Gfc�A�)*

loss�h`;u/x�       �	�2Gfc�A�)*

loss�KG; b       �	c�Gfc�A�)*

loss�`�<5�sh       �	E�Gfc�A�)*

loss��<f�*@       �	3Gfc�A�)*

loss�M=��q       �	�Gfc�A�)*

loss�K�<	��6       �	ZGfc�A�)*

loss܇J;jo��       �	��Gfc�A�)*

loss%�V<���}       �	�Gfc�A�)*

lossF ;�8��       �	�EGfc�A�)*

lossw�<)&�,       �	��Gfc�A�)*

loss}�;�r�       �	ƊGfc�A�)*

loss!��;��       �	9( Gfc�A�)*

loss[��<-VX`       �	�� Gfc�A�)*

lossO��=��3C       �	�d!Gfc�A�)*

loss�ky;1���       �	["Gfc�A�)*

loss%Y�;��ҥ       �	r�"Gfc�A�)*

loss��<�h�c       �	�$Gfc�A�)*

loss즟<t��2       �	�$Gfc�A�)*

loss^�:��       �	��%Gfc�A�)*

loss�3
<��̤       �	T6&Gfc�A�)*

loss#�0:J�`@       �	��&Gfc�A�)*

loss;+ :�?I[       �	�|'Gfc�A�)*

loss�~�9G�(3       �	�)Gfc�A�)*

loss?�9o�       �	�F*Gfc�A�)*

loss7'<I�ʣ       �	>�*Gfc�A�)*

loss�f<^��8       �	��+Gfc�A�)*

loss�L�9�{�       �	6;,Gfc�A�)*

lossNl<G��       �	�,Gfc�A�)*

lossV�S:���       �	-w-Gfc�A�)*

loss�:	`��       �	U.Gfc�A�)*

loss�J+8lA8`       �	'�.Gfc�A�)*

lossF�:��1�       �	�]/Gfc�A�)*

loss4�;H̩�       �	��/Gfc�A�)*

loss#��;YE�       �	t�0Gfc�A�)*

loss��8,�S
       �	k1Gfc�A�)*

loss��;ubT�       �	�2Gfc�A�)*

loss=P�=�O�       �	$�2Gfc�A�)*

loss�'�;|t��       �	��3Gfc�A�)*

loss��=ғ��       �	�.4Gfc�A�)*

loss���:�GNO       �	��4Gfc�A�)*

loss<h;�
��       �	qv5Gfc�A�)*

losss��<@�=	       �	@6Gfc�A�)*

loss�x�<����       �	��6Gfc�A�)*

lossϋ�<�W�H       �	8i7Gfc�A�)*

lossj�<���       �	�
8Gfc�A�)*

loss��<��       �	�8Gfc�A�)*

loss��=W��       �	�D9Gfc�A�)*

loss��<<>�o�       �	L�9Gfc�A�)*

loss��J=��c\       �	mt:Gfc�A�)*

loss���<Û�"       �	�;Gfc�A�)*

loss�� <+ٜ       �	&�;Gfc�A�)*

loss]<�;gg��       �	�F=Gfc�A�)*

loss�iM;��       �	}@Gfc�A�)*

loss���<�#:�       �	�@Gfc�A�)*

loss�<�{��       �	�~AGfc�A�)*

loss��a<\�C       �	�@BGfc�A�)*

loss7K=:+�k       �	��BGfc�A�)*

loss�C�;�,       �	v�CGfc�A�)*

loss���<aԪ�       �	GDGfc�A�)*

loss�p�<��       �	~�DGfc�A�)*

loss
B<���       �	}yEGfc�A�)*

loss�X:��k       �	FGfc�A�)*

loss���:��4       �	d�FGfc�A�)*

loss7�t=��
{       �	�GGGfc�A�**

loss�$;(�o       �	��GGfc�A�**

loss7:(<���c       �	�HGfc�A�**

loss�0<���       �	IGfc�A�**

lossMF;1A	W       �	)�JGfc�A�**

loss��<��m       �	�/KGfc�A�**

loss��<'��M       �	^�KGfc�A�**

loss�#�<�M�       �	yuLGfc�A�**

loss	M�;��(       �	�MGfc�A�**

lossҶ�<e/�       �	��MGfc�A�**

loss�T]<��K       �	ONGfc�A�**

losss<�d��       �	��NGfc�A�**

lossߧ<�|       �	܀OGfc�A�**

loss��S:�=#       �	B#PGfc�A�**

loss��:���       �	A�PGfc�A�**

loss�;�a�       �	�RQGfc�A�**

lossD��;��o       �	��QGfc�A�**

losscC=����       �	��RGfc�A�**

loss�d;��u       �	*SGfc�A�**

loss�ʾ<:�O�       �	.�SGfc�A�**

loss=*7�       �	?oTGfc�A�**

loss�^�9�a��       �	^UGfc�A�**

lossR��;qz       �	x�UGfc�A�**

loss	m�;<��       �	��VGfc�A�**

lossM#;:�^R       �	ƊWGfc�A�**

lossv<����       �	I��Gfc�A�**

lossD=@
�       �	�)�Gfc�A�**

loss��=a���       �	�΅Gfc�A�**

loss7θ;-�       �	 s�Gfc�A�**

loss��<LGV       �	=�Gfc�A�**

lossn9:A�u,       �	毇Gfc�A�**

loss*�M=
�-       �	�Q�Gfc�A�**

loss��<C$�       �	)�Gfc�A�**

loss�s<&n��       �	w��Gfc�A�**

lossRħ<�~�Y       �	q�Gfc�A�**

lossι\<�"&�       �	=��Gfc�A�**

loss�v+;=j�k       �	O�Gfc�A�**

loss*�=��;       �	}�Gfc�A�**

loss1��<�+��       �	��Gfc�A�**

loss6P;��%       �	Cs�Gfc�A�**

loss��<=)i       �	�Gfc�A�**

loss4��;�&+�       �	�ÎGfc�A�**

lossM�=��g       �	�_�Gfc�A�**

loss%��;�c)       �	d�Gfc�A�**

loss��\;�^�A       �	@��Gfc�A�**

loss=z<'S�       �	��Gfc�A�**

loss�=���       �	V*�Gfc�A�**

loss�M�:��_       �	)˒Gfc�A�**

loss��]=�2��       �	[��Gfc�A�**

lossJ~�;�Cw�       �	�T�Gfc�A�**

lossS��;���       �	��Gfc�A�**

loss%;j       �	]��Gfc�A�**

loss��U;t���       �	�Gfc�A�**

loss���<��       �	ع�Gfc�A�**

loss1�;� �       �	y[�Gfc�A�**

losshs>=���       �	��Gfc�A�**

loss#Z�:{�<�       �	1	�Gfc�A�**

loss���;̑ש       �	-B�Gfc�A�**

lossQ��<7�Zh       �	o�Gfc�A�**

loss4�;���       �	C=�Gfc�A�**

loss��/<$K2�       �	"�Gfc�A�**

lossW�I;��       �	��Gfc�A�**

lossQ;9�b�       �	�O�Gfc�A�**

loss�;ICP       �	��Gfc�A�**

lossv�=]�z�       �	Z��Gfc�A�**

loss���<H}��       �	T�Gfc�A�**

loss��;u�g�       �	��Gfc�A�**

loss��<.��R       �	�]�Gfc�A�**

loss@ܮ;-�`       �	���Gfc�A�**

loss�z=8��       �	돢Gfc�A�**

loss{�X<m�n       �	�)�Gfc�A�**

loss��=Q���       �	�ģGfc�A�**

loss-�;yM��       �	�\�Gfc�A�**

lossR��<0Ӯ       �	M�Gfc�A�**

loss�5�:0��       �	Gfc�A�**

loss�1V;l��       �	{/�Gfc�A�**

loss�4g<w��M       �	iæGfc�A�**

lossT��;Znx       �	c�Gfc�A�**

loss(s<AiT`       �	)�Gfc�A�**

lossnj�;8�@@       �	���Gfc�A�**

loss�5s;<�g       �	d:�Gfc�A�**

loss�\�;>�1�       �	שGfc�A�**

loss y9blA4       �	�|�Gfc�A�**

lossS�<�pC�       �	S#�Gfc�A�**

loss�C=���       �	�ɫGfc�A�**

lossaz=a��       �	�h�Gfc�A�**

lossN�r;=���       �	�	�Gfc�A�**

loss߉:��C)       �	\��Gfc�A�**

loss)�:C�}�       �	yW�Gfc�A�**

loss��;/ʀ�       �	��Gfc�A�**

loss</.<��'�       �	'��Gfc�A�**

loss�:�;���.       �	�>�Gfc�A�**

loss�H�:�N+       �	HݰGfc�A�**

lossh��<ldz       �	���Gfc�A�**

loss�8�= �2D       �	('�Gfc�A�**

loss��;b���       �	�R�Gfc�A�**

lossɪD<?�A�       �	$�Gfc�A�**

loss�U�;���N       �	Ւ�Gfc�A�**

lossx<h���       �	�-�Gfc�A�**

loss��:7�^       �	
׵Gfc�A�**

lossC�`<:�       �	Q��Gfc�A�**

loss2zD<s|�#       �	�,�Gfc�A�**

loss�2<^S�/       �	ܷGfc�A�**

losse��;���       �	���Gfc�A�**

loss.	S<}�+�       �	�&�Gfc�A�**

loss���:%-��       �	�ǹGfc�A�**

loss|â;004       �	�c�Gfc�A�**

loss.�;4�Pq       �	S�Gfc�A�**

loss���;�8       �	䟻Gfc�A�**

loss�~+<_�:       �	>�Gfc�A�**

loss�;�;�/B�       �	�ؼGfc�A�**

lossA��;ČD       �	�u�Gfc�A�**

loss�0�:5w��       �	��Gfc�A�**

loss��<X��       �	.��Gfc�A�**

loss�%==cq       �	�F�Gfc�A�**

loss� �<�~#i       �	�߿Gfc�A�**

loss$�;R�o       �	-|�Gfc�A�**

loss�f�: ¨G       �	Z�Gfc�A�**

loss��<���       �	���Gfc�A�**

lossvȈ<���       �	 A�Gfc�A�**

loss<��:^%�       �	���Gfc�A�**

loss�F�<q�q�       �	�l�Gfc�A�**

loss��T;v��       �	��Gfc�A�**

loss�rt=!ޛ       �	���Gfc�A�**

loss�$I<��l�       �	D�Gfc�A�**

lossļ=<yLC�       �	���Gfc�A�**

loss��O;�*8&       �	�t�Gfc�A�**

lossX��;4G       �	נ�Gfc�A�**

lossv�x;y       �	�F�Gfc�A�+*

lossvʶ;~EL�       �	��Gfc�A�+*

loss���:j���       �	���Gfc�A�+*

loss�M�9)�Q�       �	a�Gfc�A�+*

loss�G�<ļu�       �	���Gfc�A�+*

loss ��=�_�       �	��Gfc�A�+*

lossؕC<#���       �	+�Gfc�A�+*

lossTe;���       �	!��Gfc�A�+*

loss�d-<��O�       �	$c�Gfc�A�+*

loss��=Tt��       �	���Gfc�A�+*

loss�n�:�(       �	��Gfc�A�+*

loss�.~<� |       �	�#�Gfc�A�+*

loss�ϲ:Ȳ�z       �	��Gfc�A�+*

loss �a:5��       �	�S�Gfc�A�+*

loss��=��H       �	���Gfc�A�+*

loss��<��#�       �	r��Gfc�A�+*

lossN��<�J_       �	��Gfc�A�+*

loss'h�<EQ	�       �	ٱ�Gfc�A�+*

loss6
=2�ݼ       �	�I�Gfc�A�+*

loss#�;4v�q       �	��Gfc�A�+*

loss͌<LPN[       �	�r�Gfc�A�+*

loss{&;�{A�       �	j�Gfc�A�+*

loss�5�<9�/       �	J��Gfc�A�+*

loss7�;�R�A       �	�H�Gfc�A�+*

lossa��;��}�       �	3��Gfc�A�+*

loss�x=nK�3       �	���Gfc�A�+*

loss���<��B       �	���Gfc�A�+*

loss��;e���       �	�l�Gfc�A�+*

loss�Q�;=K�W       �	ni�Gfc�A�+*

loss&�L;%̧       �	���Gfc�A�+*

loss�
;;���       �	9_�Gfc�A�+*

loss.�Z;0�       �	y�Gfc�A�+*

loss`��;�6k�       �	in�Gfc�A�+*

lossnޚ:�'_�       �	�?�Gfc�A�+*

lossR�Y<����       �	���Gfc�A�+*

loss�5<_|9       �	i��Gfc�A�+*

loss��C:_\��       �	|H�Gfc�A�+*

loss�E�:C���       �	��Gfc�A�+*

loss=��:�I?	       �	ip�Gfc�A�+*

loss(L< 9�n       �	��Gfc�A�+*

lossd�;%�S�       �	ƥ�Gfc�A�+*

loss���<�0y       �	�W�Gfc�A�+*

loss�<Սٓ       �	���Gfc�A�+*

lossA�;�I�       �	��Gfc�A�+*

loss�E�;RQ�       �	�O�Gfc�A�+*

loss7s;��E!       �	Q-�Gfc�A�+*

loss(3�:��[       �	��Gfc�A�+*

loss �;KJ�       �	`��Gfc�A�+*

loss��<�	       �	px�Gfc�A�+*

lossMR'=�8��       �	�~�Gfc�A�+*

loss�$3;�,       �	i;�Gfc�A�+*

lossLݭ<J         �	���Gfc�A�+*

loss���9�|�       �	ɒ�Gfc�A�+*

losstS�9��k       �	���Gfc�A�+*

loss�
0<{�%�       �	�/�Gfc�A�+*

loss@iD=A`C       �	��Gfc�A�+*

lossM�;��8       �	�r�Gfc�A�+*

loss��i<��i       �	t�Gfc�A�+*

loss]��9��       �	��Gfc�A�+*

loss`u<�+#       �	�A�Gfc�A�+*

loss�s;�B       �	���Gfc�A�+*

loss
-�;9��>       �	�t�Gfc�A�+*

lossl�$;�g�*       �	g�Gfc�A�+*

loss�d;ږÜ       �	���Gfc�A�+*

loss_p�<��2_       �	�>�Gfc�A�+*

loss@<�<GgW�       �	x��Gfc�A�+*

losszn�:߫       �	�h�Gfc�A�+*

losshW$=��O�       �	&�Gfc�A�+*

loss��3:Hú�       �	ߥ�Gfc�A�+*

loss| �;"K�       �	�I�Gfc�A�+*

loss=�q�       �	��Gfc�A�+*

loss$%=�%�       �	i��Gfc�A�+*

loss3�c=Ax�@       �	A+�Gfc�A�+*

lossaC<>;��       �	���Gfc�A�+*

lossQ0�=�a�       �	Yn�Gfc�A�+*

loss��::�       �	��Gfc�A�+*

loss�A;ܒ3       �	���Gfc�A�+*

loss#��;e-�Q       �	�@�Gfc�A�+*

lossC�;�0�       �	j��Gfc�A�+*

loss�# ;GG�G       �	� Hfc�A�+*

loss�n�9GM#       �	�Hfc�A�+*

loss��:��&�       �	�Hfc�A�+*

loss�4<1=�       �	�THfc�A�+*

loss��<��`       �	k�Hfc�A�+*

loss�0<AX3       �	H�Hfc�A�+*

loss��&=��v       �	h%Hfc�A�+*

loss�/�<o��       �	Q�Hfc�A�+*

lossQ�=���       �	l^Hfc�A�+*

loss��<�
�       �	��Hfc�A�+*

loss�G4<���       �	8�Hfc�A�+*

lossLI�;��       �	 :Hfc�A�+*

lossđ�8k��m       �	x�Hfc�A�+*

loss<h�<YkZ�       �	nnHfc�A�+*

loss .<��E�       �		Hfc�A�+*

lossL�X:�br       �	̸	Hfc�A�+*

loss��=$�w       �	�r
Hfc�A�+*

loss.S;�zB�       �	0Hfc�A�+*

lossR��<��       �	ӼHfc�A�+*

loss���;����       �	�YHfc�A�+*

lossr�:x��       �	4�Hfc�A�+*

loss_�<ֺ�?       �	!�Hfc�A�+*

loss�l�=���       �	�EHfc�A�+*

loss`i�<��       �	��Hfc�A�+*

loss�-=�1       �	9}Hfc�A�+*

lossw��< ���       �	7Hfc�A�+*

loss�V�;�Mм       �	�Hfc�A�+*

lossoL�;��^�       �	MMHfc�A�+*

loss{<�:jky       �	�Hfc�A�+*

lossHIA<@��l       �	T�Hfc�A�+*

loss��":?��       �	'Hfc�A�+*

loss��<<�X�       �	q�Hfc�A�+*

loss���<6��       �	W_Hfc�A�+*

loss0Q�=��2�       �	��Hfc�A�+*

loss�qd<��       �	��Hfc�A�+*

loss�	=�{��       �	�Hfc�A�+*

lossD!=����       �	:�Hfc�A�+*

loss�=���       �	��Hfc�A�+*

loss��=����       �	Hfc�A�+*

lossm��;�d�       �	��Hfc�A�+*

loss���;��{�       �	6�Hfc�A�+*

loss�,�;�O8�       �	ÂHfc�A�+*

loss͜�:�v�       �	�YHfc�A�+*

loss�+d;�F       �	/�Hfc�A�+*

loss<�:�Y�       �	��Hfc�A�+*

loss%�,<C�<�       �	dHfc�A�+*

loss\��;]�r       �	�0Hfc�A�+*

loss6�:�r�       �	'.Hfc�A�+*

loss��9ٸ#       �	��Hfc�A�+*

loss�X�;& Q       �	�� Hfc�A�,*

loss}�<���       �	
"Hfc�A�,*

loss�s�;G[B:       �	 �"Hfc�A�,*

loss-��<�H4_       �	��#Hfc�A�,*

loss �y<5N��       �	�R$Hfc�A�,*

loss���<<�8�       �	;%Hfc�A�,*

loss2ڳ<lP5�       �	�&Hfc�A�,*

lossN��;���8       �	)�&Hfc�A�,*

loss��:��>�       �	xc'Hfc�A�,*

lossC�|;��Z       �	��(Hfc�A�,*

loss$�<�f��       �	�%)Hfc�A�,*

loss.=�Hk�       �	�)Hfc�A�,*

loss4��<A��z       �	>\*Hfc�A�,*

loss*��<�Y�|       �	y+Hfc�A�,*

loss���<�Ώb       �	��+Hfc�A�,*

loss~�<-�=�       �	�E,Hfc�A�,*

loss=�x=Ur�        �	��,Hfc�A�,*

loss,P�<�.*3       �	�x-Hfc�A�,*

loss�c;���       �	=.Hfc�A�,*

loss�J=>B�o       �	Ƨ.Hfc�A�,*

lossV��<tz4       �	@/Hfc�A�,*

loss�W"<���        �	'�/Hfc�A�,*

loss�m9;�� |       �	�o0Hfc�A�,*

loss�ϟ:�eɞ       �	11Hfc�A�,*

loss��h=�;t�       �	��1Hfc�A�,*

loss��:Y`�Q       �	12Hfc�A�,*

loss��<%�hy       �	��2Hfc�A�,*

loss�= ��       �	t_3Hfc�A�,*

loss&�$<Y�/�       �	��3Hfc�A�,*

loss�Ñ<���>       �	dW5Hfc�A�,*

loss��i;S{|       �	��5Hfc�A�,*

loss�i=0~$7       �	b�6Hfc�A�,*

loss��i<�ȹ�       �	�7Hfc�A�,*

loss|��; V�$       �	��7Hfc�A�,*

loss�΅;A�       �	�W8Hfc�A�,*

loss�Z�<�)<;       �	��8Hfc�A�,*

loss�e+<�+Q&       �	�9Hfc�A�,*

loss;��;�8i       �	�:Hfc�A�,*

loss�zn;���j       �	n�:Hfc�A�,*

losss$i<�y<�       �	:W;Hfc�A�,*

lossk�=Wέ-       �	��;Hfc�A�,*

loss1��<��       �	��<Hfc�A�,*

loss�uH<���       �	m;=Hfc�A�,*

loss��Q=�u5       �	l�=Hfc�A�,*

lossN�<#XB�       �	�e>Hfc�A�,*

loss�0;��#       �	��>Hfc�A�,*

loss�vK;4���       �	ٓ?Hfc�A�,*

loss��<Xt��       �	�7@Hfc�A�,*

loss7�;��4�       �	$�@Hfc�A�,*

lossE�:'�o       �	�mAHfc�A�,*

loss���;ㄻ�       �	�	BHfc�A�,*

loss��0;���j       �	�BHfc�A�,*

loss�6=�2ǧ       �	bJCHfc�A�,*

loss��=�M�       �	��CHfc�A�,*

lossߵ;��5r       �	��DHfc�A�,*

loss���<#�       �	t$EHfc�A�,*

loss�.�<�+�'       �	r�EHfc�A�,*

lossk;��(�       �	�|GHfc�A�,*

loss�E;�F��       �	�HHfc�A�,*

loss��;XgB�       �	<�HHfc�A�,*

loss���;$R~X       �	�qIHfc�A�,*

lossM�a:62�       �	�	JHfc�A�,*

lossI5�9��\T       �	\�JHfc�A�,*

loss�
�<̳nV       �	�aKHfc�A�,*

loss�HX=���!       �	�KHfc�A�,*

loss��E<�J�	       �	-�LHfc�A�,*

loss��=r�!       �	D5MHfc�A�,*

loss�M<��z       �	��MHfc�A�,*

lossx��< �U�       �	=bNHfc�A�,*

loss{��<�:�A       �	��NHfc�A�,*

lossdR':V7��       �	��OHfc�A�,*

loss}9T:]�N�       �	�7PHfc�A�,*

loss���<z��       �	��PHfc�A�,*

loss�a�;]��'       �	eQHfc�A�,*

loss�9P=�mD{       �	� RHfc�A�,*

lossq��<V��       �	�RHfc�A�,*

lossa{s;ĉv�       �	�4SHfc�A�,*

loss�+=s��3       �	!�SHfc�A�,*

lossߺ�<ӟU       �	�$UHfc�A�,*

lossO5<��9       �	�UHfc�A�,*

loss3QP<���H       �	�YVHfc�A�,*

loss��9<�nw       �	�WHfc�A�,*

lossIt�;��i�       �	�"XHfc�A�,*

loss1�c;j�M       �	��XHfc�A�,*

loss ;��׻       �	LkYHfc�A�,*

lossm�<��P       �	8ZHfc�A�,*

loss1R�:�s �       �	=�ZHfc�A�,*

loss�|�<%A��       �	�Z[Hfc�A�,*

lossĖ�<�ϭ       �	p$\Hfc�A�,*

loss��;���       �	n�\Hfc�A�,*

loss�ǰ<x���       �	�V]Hfc�A�,*

loss��<�#=S       �	x�]Hfc�A�,*

loss���<K�%�       �	��^Hfc�A�,*

loss���<64�p       �	_Hfc�A�,*

loss�`=�blq       �	�_Hfc�A�,*

loss �;���       �	�Y`Hfc�A�,*

loss.��<�uf       �	��aHfc�A�,*

lossX�:&��S       �	�VbHfc�A�,*

loss#�w<g�       �	_�bHfc�A�,*

losszW�<���       �	��cHfc�A�,*

lossf�*<0�j       �	,,dHfc�A�,*

losseC�;�`_�       �	e�dHfc�A�,*

loss��;-Q^P       �	feHfc�A�,*

loss#"+=�YS       �	�fHfc�A�,*

loss#V�;KhЏ       �	Y�fHfc�A�,*

loss׈.=��_�       �	1[gHfc�A�,*

lossR��;^Gs       �	Y�gHfc�A�,*

loss{��<}�k�       �	��hHfc�A�,*

losst��:@~�e       �	�0iHfc�A�,*

loss_��:���6       �	��iHfc�A�,*

loss�K�;ݏ&�       �	�^jHfc�A�,*

loss�IJ<4��c       �	(kHfc�A�,*

losst)�=���u       �	]�kHfc�A�,*

loss�j�=��C�       �	>lHfc�A�,*

lossn_�<ze��       �	]�lHfc�A�,*

loss�;<$       �	�tmHfc�A�,*

loss���<�.1       �	�nHfc�A�,*

loss�k2<I�       �	�nHfc�A�,*

loss�X.=�/8�       �	�KoHfc�A�,*

loss�A;��$>       �	O�oHfc�A�,*

loss��<g���       �	=�pHfc�A�,*

loss��<�y �       �	�#qHfc�A�,*

loss�A><ÿ<       �	��qHfc�A�,*

loss�Qk;��|�       �	�erHfc�A�,*

loss�\;hv�I       �	�sHfc�A�,*

lossrB�;!c�	       �	�sHfc�A�,*

loss	߬9�}K*       �	BCtHfc�A�,*

loss�h�;�D�       �	e�tHfc�A�,*

lossb�<ssw�       �	�zuHfc�A�-*

loss=��<��       �	.vHfc�A�-*

loss�ej;�Fp�       �	��vHfc�A�-*

loss3I[;�,�       �	(awHfc�A�-*

loss�rQ=�T�_       �	7�wHfc�A�-*

loss�,=t�kW       �	\�xHfc�A�-*

loss?2P;�S|�       �	�yHfc�A�-*

loss��=��       �	'�yHfc�A�-*

loss���<���       �	�ZzHfc�A�-*

loss�Z�:��       �	��zHfc�A�-*

losse�t;lK��       �	��{Hfc�A�-*

lossH�< �-       �	�1|Hfc�A�-*

loss�	:z�-z       �	��|Hfc�A�-*

loss��W<wO��       �	d}Hfc�A�-*

loss��;Y�'�       �	~�}Hfc�A�-*

loss��;���       �	�~Hfc�A�-*

loss��6<옗       �	�8Hfc�A�-*

loss�\;K���       �	g�Hfc�A�-*

loss�H9<�iM�       �	]��Hfc�A�-*

loss�0�:
Gg�       �	=F�Hfc�A�-*

loss�<�n�       �	T�Hfc�A�-*

loss�:a=7܏�       �	(��Hfc�A�-*

loss6m�<a:n       �	�!�Hfc�A�-*

loss?H=�9ǯ       �	Է�Hfc�A�-*

loss�U.;M��}       �	�L�Hfc�A�-*

loss$��;(��       �	?��Hfc�A�-*

loss\�=p7�       �	|ԅHfc�A�-*

loss ��;��w       �	�h�Hfc�A�-*

loss���<�!W       �	��Hfc�A�-*

loss#M:7�%L       �	��Hfc�A�-*

loss1��;�N�)       �	�2�Hfc�A�-*

loss�=K;�Wճ       �	�ǉHfc�A�-*

loss4;��1�       �	qY�Hfc�A�-*

loss�(2<�7B       �	��Hfc�A�-*

losssq9='���       �	�z�Hfc�A�-*

lossJ��<��       �	9�Hfc�A�-*

loss$_�=q-��       �	j��Hfc�A�-*

loss�_v:����       �	�R�Hfc�A�-*

loss�4r=3a       �	S�Hfc�A�-*

loss-;-���       �	f��Hfc�A�-*

loss��;"�Š       �	&�Hfc�A�-*

loss���9�/�       �	/��Hfc�A�-*

loss@N/<�d�h       �	�`�Hfc�A�-*

loss̟D<�K�\       �	���Hfc�A�-*

loss��<�{=       �	̖�Hfc�A�-*

lossqX�<K/.�       �	�1�Hfc�A�-*

loss�=�ǃ       �	d˒Hfc�A�-*

lossvX <B?g�       �	�p�Hfc�A�-*

loss�+^<�+��       �	��Hfc�A�-*

lossT��<���f       �	��Hfc�A�-*

loss��O<�h�R       �	&R�Hfc�A�-*

loss�ǅ=J�V�       �	��Hfc�A�-*

loss���=�%��       �	G��Hfc�A�-*

lossц!<)���       �	�c�Hfc�A�-*

loss��=��"�       �	�Hfc�A�-*

loss�'<"��       �	f��Hfc�A�-*

loss�,<��t       �	y[�Hfc�A�-*

loss/1�<�Q:z       �	��Hfc�A�-*

loss�7;}E       �	ס�Hfc�A�-*

loss�=_�"\       �	y?�Hfc�A�-*

loss�ט<Z���       �	�ݜHfc�A�-*

loss��;`��       �	Kv�Hfc�A�-*

loss>�<��l       �	��Hfc�A�-*

loss�;g��       �	Y��Hfc�A�-*

loss��=���       �	�G�Hfc�A�-*

losslڋ=��z       �	��Hfc�A�-*

loss�?�;��m       �	���Hfc�A�-*

loss�r4<,)       �	�%�Hfc�A�-*

lossa��;p:��       �	�͡Hfc�A�-*

loss%�=DԦ5       �	sg�Hfc�A�-*

loss�O�9�d       �	���Hfc�A�-*

loss�:��T�       �	R��Hfc�A�-*

loss	��;X�4,       �	�=�Hfc�A�-*

loss�8�;q�       �	��Hfc�A�-*

loss
��;���8       �	�y�Hfc�A�-*

loss�@;,�n�       �	��Hfc�A�-*

loss�<2�:8       �	���Hfc�A�-*

loss n�:O��       �	�G�Hfc�A�-*

loss��<WG�X       �	v�Hfc�A�-*

loss��<r*P       �	�}�Hfc�A�-*

loss��;�u�.       �	e�Hfc�A�-*

loss��F=�'��       �	)��Hfc�A�-*

lossi5�;z�a       �	�M�Hfc�A�-*

loss�"=�V�       �	��Hfc�A�-*

loss�=�"�       �	��Hfc�A�-*

loss��;�>��       �	7�Hfc�A�-*

loss;�=�9       �	u��Hfc�A�-*

loss4��<7�(�       �	Q�Hfc�A�-*

loss�K<�B�       �	m�Hfc�A�-*

loss�4<�"�       �	3��Hfc�A�-*

loss �;��=       �	�/�Hfc�A�-*

loss!%J<(��       �	f��Hfc�A�-*

lossZh�<�}�       �	���Hfc�A�-*

lossW"U;�*b-       �	�*�Hfc�A�-*

losse�u:���       �	t@�Hfc�A�-*

loss��0<K�={       �	�ԲHfc�A�-*

loss�J;��֞       �	�h�Hfc�A�-*

lossR�;���       �	i�Hfc�A�-*

lossy<A9�*       �	d!�Hfc�A�-*

loss*W�;W       �	���Hfc�A�-*

loss�=L<�       �	K�Hfc�A�-*

lossR�<]�       �	��Hfc�A�-*

loss�;2�Vf       �	x�Hfc�A�-*

loss�P�:�s9       �	��Hfc�A�-*

loss���=�H/�       �	���Hfc�A�-*

loss	��:�5M�       �	�5�Hfc�A�-*

lossѾ�:AJ��       �	�͹Hfc�A�-*

loss&�@<-%�=       �	oe�Hfc�A�-*

loss1h�;z���       �	��Hfc�A�-*

lossc�;uk#�       �	G��Hfc�A�-*

lossj�:���[       �	&�Hfc�A�-*

loss�="��       �	s��Hfc�A�-*

loss���<�-h�       �	�Z�Hfc�A�-*

loss��<�T�       �	*��Hfc�A�-*

loss���<���       �	˜�Hfc�A�-*

loss��9ѴZ�       �	>�Hfc�A�-*

loss��;��G       �	n߿Hfc�A�-*

loss�.�90�^9       �	���Hfc�A�-*

losst�3<Ȼ�       �	n4�Hfc�A�-*

loss}o;FS�       �	���Hfc�A�-*

loss�P�<S��       �	�n�Hfc�A�-*

loss Dt<��       �	t�Hfc�A�-*

loss�?;��       �	���Hfc�A�-*

loss.=���8       �	I�Hfc�A�-*

loss��}9"��       �	F��Hfc�A�-*

losse�B</|_       �	*��Hfc�A�-*

losse��<��S�       �	�(�Hfc�A�-*

loss�;v<h�I�       �	��Hfc�A�-*

loss��,<���       �	>^�Hfc�A�.*

lossܬV<(\�       �	E��Hfc�A�.*

lossb�=52�]       �	ʌ�Hfc�A�.*

lossZ��;���6       �	�4�Hfc�A�.*

loss�b$;�(       �	J��Hfc�A�.*

loss���:��0       �	�k�Hfc�A�.*

loss|�&=����       �	��Hfc�A�.*

loss�~:���       �	���Hfc�A�.*

loss��<`�$       �	F[�Hfc�A�.*

loss�;�;�$�       �	���Hfc�A�.*

loss�S<]Y�       �	���Hfc�A�.*

lossZ�<a��*       �	L�Hfc�A�.*

lossf�<��4<       �	\��Hfc�A�.*

loss�O=��       �	�G�Hfc�A�.*

lossj�?:�(г       �	G��Hfc�A�.*

loss;ߟ9���       �	Jz�Hfc�A�.*

loss��<(�I�       �	]�Hfc�A�.*

loss��;��f�       �	���Hfc�A�.*

loss�1a<���       �	�H�Hfc�A�.*

loss
�;N,Y       �	���Hfc�A�.*

lossV�<7XU�       �	V��Hfc�A�.*

loss�M�<Y`I�       �	(�Hfc�A�.*

loss�G�<�e��       �	O��Hfc�A�.*

lossd�:+-;       �	an�Hfc�A�.*

loss(;�>Y�       �	5	�Hfc�A�.*

loss��U<� n\       �	��Hfc�A�.*

lossQ],< J��       �	�9�Hfc�A�.*

loss��;����       �	�[�Hfc�A�.*

loss!:<"N       �	B	�Hfc�A�.*

lossH��:I��       �	���Hfc�A�.*

loss?w9ۇ�       �	V��Hfc�A�.*

loss&[;~�|       �	 p�Hfc�A�.*

loss.��:�)�&       �	�	�Hfc�A�.*

loss73�:�/I       �	���Hfc�A�.*

loss\��:dy�U       �	q<�Hfc�A�.*

loss��9J�O       �	���Hfc�A�.*

loss6��:��f?       �	�r�Hfc�A�.*

loss��<�L3       �	��Hfc�A�.*

loss�BZ<i"��       �	���Hfc�A�.*

lossT==��S       �	�5�Hfc�A�.*

lossQ1�<��!       �	���Hfc�A�.*

lossx�^<-;��       �	�c�Hfc�A�.*

loss��<�Vf�       �	���Hfc�A�.*

lossE�:��ƒ       �	���Hfc�A�.*

loss�n�<G C       �	�!�Hfc�A�.*

loss��;�V�T       �	���Hfc�A�.*

lossR�e8o���       �	�|�Hfc�A�.*

loss��:����       �	�o�Hfc�A�.*

loss���:6��       �	h�Hfc�A�.*

loss�`o;tXe       �	���Hfc�A�.*

loss���;����       �	f�Hfc�A�.*

loss�2�94bς       �	[�Hfc�A�.*

loss�o�=T(A{       �	��Hfc�A�.*

losso!R9�U/~       �	�F�Hfc�A�.*

loss��7�ĥ       �	6��Hfc�A�.*

loss��8��:�       �	B��Hfc�A�.*

loss���:Y�v�       �	�6�Hfc�A�.*

lossh`<K�>       �	���Hfc�A�.*

loss���9�sq       �	�k�Hfc�A�.*

loss�v9m�$s       �	�	�Hfc�A�.*

lossӥl;�H�       �	���Hfc�A�.*

loss)|<(�L�       �	;T�Hfc�A�.*

loss�"; F-       �	^��Hfc�A�.*

loss��d=";       �	y��Hfc�A�.*

loss)�<zEx�       �	XW�Hfc�A�.*

loss�kW=uP!�       �	���Hfc�A�.*

lossvB
<�G��       �	Ɏ�Hfc�A�.*

loss�c�:|��       �	J(�Hfc�A�.*

loss�G;��k1       �	"��Hfc�A�.*

lossC�=�]l       �	�j�Hfc�A�.*

loss}�=�e��       �	��Hfc�A�.*

loss>�;�h#q       �	��Hfc�A�.*

loss�<��c       �	�B�Hfc�A�.*

lossȃF=z��m       �	p��Hfc�A�.*

loss�g{<��       �	���Hfc�A�.*

loss软9��\�       �	�,�Hfc�A�.*

loss?n<��*r       �	���Hfc�A�.*

loss N�;�9-       �	�M�Hfc�A�.*

loss*�;n�<�       �	���Hfc�A�.*

lossj�
;d2�;       �	��Hfc�A�.*

losso�;=��0/       �	%<�Hfc�A�.*

loss���:a�9�       �	���Hfc�A�.*

loss�:,���       �	�n�Hfc�A�.*

lossF-<L���       �	��Hfc�A�.*

loss2��<�G��       �	���Hfc�A�.*

loss���;2=D       �	�B�Hfc�A�.*

loss�:�*ݸ       �	)��Hfc�A�.*

loss�O�;O���       �	؀�Hfc�A�.*

lossF��;����       �	�& Ifc�A�.*

lossi|j;a�4V       �	e� Ifc�A�.*

lossTb=;R�       �	0eIfc�A�.*

loss �
=Ξȑ       �	�Ifc�A�.*

loss�C";«�       �	<�Ifc�A�.*

loss���<�d�c       �	�EIfc�A�.*

loss�(�;�Q%       �	��Ifc�A�.*

loss�2�<�CvZ       �	WyIfc�A�.*

loss��;�g       �	sIfc�A�.*

loss_�';s��       �	VIfc�A�.*

lossoC�;���       �	7�Ifc�A�.*

lossI��;J��       �	BIfc�A�.*

loss�<Y��!       �	��Ifc�A�.*

loss�!R<\�5F       �	�qIfc�A�.*

loss,�<�>�       �	d	Ifc�A�.*

lossDC�:Rf�       �	��	Ifc�A�.*

loss��H:��0s       �	�Q
Ifc�A�.*

loss���=���       �	9�
Ifc�A�.*

lossn}!:k`       �	3�Ifc�A�.*

loss��@;%~��       �	Y2Ifc�A�.*

loss�<_�       �	��Ifc�A�.*

loss\�;���~       �	=eIfc�A�.*

loss�<L
/]       �	qIfc�A�.*

loss6;H��/       �	��Ifc�A�.*

lossO�<�Lj�       �	�XIfc�A�.*

loss��:�"7       �	#f/Ifc�A�.*

lossD��<|���       �	 0Ifc�A�.*

loss4��<O���       �	��0Ifc�A�.*

loss��=KB�L       �	5B1Ifc�A�.*

loss��p:ڰէ       �	i�1Ifc�A�.*

loss��:q��       �	�v2Ifc�A�.*

losss��;���V       �	,3Ifc�A�.*

loss�~H=ߠ��       �	��3Ifc�A�.*

lossRv�<��}       �	~4Ifc�A�.*

loss�4q<���       �	S"5Ifc�A�.*

lossa�>;�Β       �	��5Ifc�A�.*

loss Ox<<V�       �	Ul6Ifc�A�.*

loss$�G;M�qV       �	e7Ifc�A�.*

loss;�o;�yc       �	j�7Ifc�A�.*

loss��[=���       �	�G8Ifc�A�.*

loss�{�<�C�       �	��8Ifc�A�/*

loss��;[��       �	?�9Ifc�A�/*

loss6ƾ9��^�       �	/:Ifc�A�/*

loss	O�;��rj       �	��:Ifc�A�/*

lossH�2=!r��       �	m;Ifc�A�/*

loss\��<+���       �	
<Ifc�A�/*

loss��<�&       �	6�<Ifc�A�/*

loss�`�:�R�3       �	+P=Ifc�A�/*

loss!�<<pS�@       �	K�=Ifc�A�/*

loss�c�:ۜ�
       �	�R?Ifc�A�/*

lossX�9M��	       �	F�?Ifc�A�/*

lossNr�<�L�/       �	��@Ifc�A�/*

lossA�F;"R�       �	�0AIfc�A�/*

loss��;�I~        �	\�AIfc�A�/*

lossjg�;+w��       �	�_BIfc�A�/*

loss��:��       �	�CIfc�A�/*

loss/s�:Pع+       �	g�CIfc�A�/*

lossT3<�|��       �	�5DIfc�A�/*

loss�+"=e��       �	��DIfc�A�/*

loss��):2ׄ�       �	/�EIfc�A�/*

loss@��<jEW�       �	�FIfc�A�/*

loss[^�9(��"       �	6�FIfc�A�/*

lossE^�;�T��       �	4MGIfc�A�/*

loss�r<k(*a       �	T�GIfc�A�/*

loss�m:�4{�       �	A}HIfc�A�/*

loss�i�;i.W�       �	�IIfc�A�/*

loss�e�;��B�       �	ձIIfc�A�/*

loss
��8�γ�       �	�EJIfc�A�/*

loss.�c;�"��       �	��JIfc�A�/*

lossDJe<���a       �	�wKIfc�A�/*

lossA��<�3c       �	l
LIfc�A�/*

losse�<��V�       �	P�LIfc�A�/*

loss�±<�z�       �	�=MIfc�A�/*

loss�N�9yz/"       �	R�MIfc�A�/*

loss��92qZ�       �	�lNIfc�A�/*

loss?_:ʘ^�       �	J'OIfc�A�/*

lossE =�3       �	��OIfc�A�/*

loss\�9Lr�)       �	?RPIfc�A�/*

lossm{=��       �	��PIfc�A�/*

loss�:h;9�h�       �	#�QIfc�A�/*

loss���:��W       �	*RIfc�A�/*

losss<ATr�       �	��RIfc�A�/*

loss��9�m       �	IdSIfc�A�/*

loss���:Rt}�       �	7�SIfc�A�/*

loss"=����       �	��TIfc�A�/*

lossT,=LQ:�       �	ŎUIfc�A�/*

losspw�<��FQ       �	�)VIfc�A�/*

lossR�-9:�r       �	��VIfc�A�/*

loss�=�WU�       �	�WIfc�A�/*

loss���;�oi�       �	GXIfc�A�/*

loss`eq:%�d       �	�XIfc�A�/*

lossj��<�i�	       �	�ZYIfc�A�/*

loss��8=���       �	h[Ifc�A�/*

loss'O�;��X�       �	;\Ifc�A�/*

loss�k�<H�,�       �	X�\Ifc�A�/*

loss\�9���       �	��]Ifc�A�/*

loss��={v�)       �	Á^Ifc�A�/*

lossa��;
.�       �	�q_Ifc�A�/*

lossUǏ;A�e       �	w+`Ifc�A�/*

loss�aJ;����       �	>�`Ifc�A�/*

loss���:����       �	�|aIfc�A�/*

loss�o;QJ��       �	h!bIfc�A�/*

lossj�:x�Ai       �	$�bIfc�A�/*

losst�;��|�       �	>{cIfc�A�/*

loss)�u;���       �	mdIfc�A�/*

loss�\�<A�!�       �	}�dIfc�A�/*

loss]�U=}/�       �	"meIfc�A�/*

lossQI�:Z���       �		fIfc�A�/*

loss��Q8�݂�       �	|�fIfc�A�/*

lossn=���2       �	LPgIfc�A�/*

loss��g<1��r       �	h�gIfc�A�/*

loss�du<�
(       �	��hIfc�A�/*

loss鉙;��*/       �	a7iIfc�A�/*

loss�3=d
�       �	��iIfc�A�/*

loss,_=;�}N       �	�zjIfc�A�/*

loss�S�:�I��       �	�kIfc�A�/*

loss@0X=�z       �	}�kIfc�A�/*

loss�<���       �	�VlIfc�A�/*

lossU �<�I�h       �		�lIfc�A�/*

loss	UW=L*��       �	l�mIfc�A�/*

loss�d�<�d       �	�6nIfc�A�/*

lossC�7<�J^�       �	�nIfc�A�/*

loss�8�:��bx       �	coIfc�A�/*

loss�.�;����       �	� pIfc�A�/*

loss�VN=��C       �	��pIfc�A�/*

lossv�p:MIZ�       �	FEqIfc�A�/*

loss`y�<�O       �	��qIfc�A�/*

loss�ջ=?�/�       �	�xrIfc�A�/*

loss�;V�s       �	�sIfc�A�/*

lossvQ<?bR       �	�sIfc�A�/*

loss�W:@H��       �	VItIfc�A�/*

loss�<;�5!�       �	�tIfc�A�/*

loss�qN;+��f       �	ǀuIfc�A�/*

lossi�0=����       �	�vIfc�A�/*

lossnX>;�8�       �	A�vIfc�A�/*

loss�V/<���       �	��wIfc�A�/*

loss���<؉Kt       �	PnxIfc�A�/*

loss��C<z[f�       �	yIfc�A�/*

lossn�';}��       �	,�yIfc�A�/*

lossL/�<�Xp�       �	�]zIfc�A�/*

loss�1:`h/�       �	N{{Ifc�A�/*

lossH��;_��       �	�|Ifc�A�/*

loss~;SI��       �	d�|Ifc�A�/*

loss���;
�pt       �	#J}Ifc�A�/*

loss��;Le��       �	?�}Ifc�A�/*

loss�h�<�e��       �	)�~Ifc�A�/*

losstF\<d�j       �	�hIfc�A�/*

lossQ�;�fh       �	��Ifc�A�/*

lossg* ;�;R       �	1��Ifc�A�/*

loss��u:ԝ<       �	�6�Ifc�A�/*

loss��;���0       �	��Ifc�A�/*

loss��&;����       �	g|�Ifc�A�/*

loss�5;�@D       �	H�Ifc�A�/*

loss�<���_       �	Ifc�A�/*

loss��;��C       �	�K�Ifc�A�/*

lossH��<���T       �	&�Ifc�A�/*

lossv��;Qү       �	 ~�Ifc�A�/*

loss<U=���       �	�M�Ifc�A�/*

lossa��;5�5       �	��Ifc�A�/*

loss]0�;�W�       �	+��Ifc�A�/*

loss''"<4��D       �	W%�Ifc�A�/*

lossX�3;VG��       �	!ˈIfc�A�/*

lossg;k(       �	�a�Ifc�A�/*

loss=�<g@s       �	/��Ifc�A�/*

loss҃�=��!�       �	�Ifc�A�/*

lossx9�:�6�B       �	�1�Ifc�A�/*

loss�=�]�`       �	uˋIfc�A�/*

loss̱4<T�ܒ       �	�`�Ifc�A�/*

lossŭ�;�Ӕ        �	e��Ifc�A�0*

loss%SP=U:��       �	���Ifc�A�0*

loss�צ<�b��       �	1@�Ifc�A�0*

loss�Ft<_C�       �	��Ifc�A�0*

loss;�9%HK�       �	
��Ifc�A�0*

lossI49y���       �	��Ifc�A�0*

loss�n;u�"       �	���Ifc�A�0*

loss췞9��0       �	VE�Ifc�A�0*

loss]�:��       �	wۑIfc�A�0*

loss}��:�Ho�       �	�p�Ifc�A�0*

loss�<��W4       �	��Ifc�A�0*

loss�V�:�?�4       �	#��Ifc�A�0*

loss#{E9R_�%       �	�8�Ifc�A�0*

loss��;�V��       �	�ΔIfc�A�0*

lossȂ.=ʹ�       �	%z�Ifc�A�0*

loss/�=bO�       �	��Ifc�A�0*

loss�VG;��as       �	1��Ifc�A�0*

loss��<iRt       �	�O�Ifc�A�0*

loss1��;�J�       �	`�Ifc�A�0*

loss��;W~       �	���Ifc�A�0*

loss�7�S��       �	���Ifc�A�0*

loss�d4=|��e       �	��Ifc�A�0*

lossx�8�@��       �	\��Ifc�A�0*

loss$|;|��       �	BA�Ifc�A�0*

loss�#J=`��       �	�ޜIfc�A�0*

loss��;&�<       �	.u�Ifc�A�0*

lossc5:ج��       �	�Ifc�A�0*

loss�81=�P�b       �	 X�Ifc�A�0*

lossI &;~ܝI       �	%�Ifc�A�0*

loss#��;*P�       �	<��Ifc�A�0*

loss�j�;|\�       �	OW�Ifc�A�0*

loss�<�;���       �	��Ifc�A�0*

loss��<@���       �	ҋ�Ifc�A�0*

losslC�<�49�       �	� �Ifc�A�0*

loss��=�       �	/��Ifc�A�0*

loss�&�:�]       �	�V�Ifc�A�0*

loss#�F:�#e�       �	��Ifc�A�0*

lossh�9w��       �	,�Ifc�A�0*

loss���<@��       �	]�Ifc�A�0*

lossw�<04�G       �	��Ifc�A�0*

loss�V<kv\       �	�m�Ifc�A�0*

loss\�;�dߞ       �		�Ifc�A�0*

lossM�:�4C       �	�Ifc�A�0*

loss�5�;�o�       �	ǁ�Ifc�A�0*

loss.��:W�K�       �	��Ifc�A�0*

loss�~<߽��       �	���Ifc�A�0*

loss��*<��5E       �	�[�Ifc�A�0*

loss��K<ɁU�       �	��Ifc�A�0*

loss*W�:���K       �	n��Ifc�A�0*

lossξ:��
�       �	�(�Ifc�A�0*

loss���:����       �	�ȯIfc�A�0*

lossf�<U0��       �	Zb�Ifc�A�0*

lossK�9���e       �	e �Ifc�A�0*

lossz��<��       �	��Ifc�A�0*

loss!o�<����       �	�E�Ifc�A�0*

loss��< �#�       �	P�Ifc�A�0*

lossv�~<����       �	���Ifc�A�0*

loss;T�<�R]�       �	�õIfc�A�0*

lossw۔:��       �	|`�Ifc�A�0*

lossi�P;��-�       �	%�Ifc�A�0*

loss3�;!�J@       �	ퟷIfc�A�0*

lossZ&%:Qc��       �	"9�Ifc�A�0*

loss�n�:�#�       �	���Ifc�A�0*

lossr�v=�Es       �	���Ifc�A�0*

loss��b<VA�$       �	L�Ifc�A�0*

loss5<u]ڨ       �	'�Ifc�A�0*

loss۳X<]�*�       �	2ƻIfc�A�0*

loss�;�Sr�       �	�e�Ifc�A�0*

loss;�n9YU�       �	 �Ifc�A�0*

lossVe�<�ٞ�       �	x��Ifc�A�0*

loss��;Եz|       �	�1�Ifc�A�0*

loss���;��+       �	ξIfc�A�0*

lossJ�<��l�       �	�d�Ifc�A�0*

loss[�=>�$�	       �	���Ifc�A�0*

loss�V�=�j�       �	��Ifc�A�0*

losswq�<��a�       �	�0�Ifc�A�0*

lossb=��       �	`��Ifc�A�0*

loss��/;���H       �	�d�Ifc�A�0*

loss��;���{       �	���Ifc�A�0*

loss�f�;���?       �	d��Ifc�A�0*

loss�f�=7��%       �	A.�Ifc�A�0*

loss��;!�`�       �	e��Ifc�A�0*

lossů&<�dG�       �	`�Ifc�A�0*

loss�>=b�H1       �	���Ifc�A�0*

loss��;�f��       �	9��Ifc�A�0*

lossa<��{       �	#0�Ifc�A�0*

lossϊ�<ʋ��       �	���Ifc�A�0*

lossjJ�:xv�v       �	jj�Ifc�A�0*

loss̉E;6�       �	���Ifc�A�0*

loss���<(:VX       �	��Ifc�A�0*

loss]�.:�͏�       �	B'�Ifc�A�0*

loss���;�м       �	M��Ifc�A�0*

loss&�*::�Q�       �	2U�Ifc�A�0*

loss��<��`~       �	:��Ifc�A�0*

loss�4�<�       �	�x�Ifc�A�0*

loss�-j=?=�       �	�Ifc�A�0*

loss��<OU�)       �	2��Ifc�A�0*

loss�*�=G��       �	@�Ifc�A�0*

lossh(�:�4�       �	���Ifc�A�0*

loss��T; �"�       �	���Ifc�A�0*

loss��c=?�,       �	 9�Ifc�A�0*

lossO��:�@       �	A��Ifc�A�0*

loss��<S�ݷ       �	zr�Ifc�A�0*

lossC��:7q��       �	�Ifc�A�0*

loss��<�+�       �	���Ifc�A�0*

loss_��=�l��       �	6>�Ifc�A�0*

loss��<7���       �	_
�Ifc�A�0*

lossی%<���       �	�0�Ifc�A�0*

loss
�<���       �	���Ifc�A�0*

loss4�<�T��       �	sg�Ifc�A�0*

lossݍ<�m5�       �	���Ifc�A�0*

losse�j:��iG       �	���Ifc�A�0*

loss_;�:�Ǆ�       �	:=�Ifc�A�0*

loss��b<W�`P       �	�
�Ifc�A�0*

loss�<��Y       �	׾�Ifc�A�0*

lossO�;"C`       �	�_�Ifc�A�0*

loss��<�7	�       �	���Ifc�A�0*

loss��;���       �	1��Ifc�A�0*

loss��G="HU8       �	8�Ifc�A�0*

loss��<aᆌ       �	���Ifc�A�0*

lossw�5<�l�       �	t^�Ifc�A�0*

loss {�:T=�)       �	3��Ifc�A�0*

loss�.s:��G       �	��Ifc�A�0*

loss2�^<ie�       �	�-�Ifc�A�0*

lossrŐ<��|o       �	���Ifc�A�0*

loss�b<�]��       �	~T�Ifc�A�0*

lossf��<jT��       �	���Ifc�A�0*

lossIT3;T�A�       �	��Ifc�A�0*

loss��q<��&       �	���Ifc�A�1*

loss��;1u��       �	$�Ifc�A�1*

lossf��9�T}       �	���Ifc�A�1*

lossfG=7�O�       �	R�Ifc�A�1*

loss�:=*��w       �	��Ifc�A�1*

lossmt�9��f�       �	1y�Ifc�A�1*

loss&�9c￪       �	��Ifc�A�1*

loss
��<�y��       �	��Ifc�A�1*

loss+��;m-�       �	�5�Ifc�A�1*

loss�:��[       �	b��Ifc�A�1*

loss�\p<Q�       �	m�Ifc�A�1*

loss�;I�ל       �	a��Ifc�A�1*

loss���<3s�        �	���Ifc�A�1*

loss�&;��>       �	y@�Ifc�A�1*

loss,x�<��       �	��Ifc�A�1*

losst�<�bmW       �	�u�Ifc�A�1*

loss���<��Mu       �	��Ifc�A�1*

loss��:�*Fb       �	g��Ifc�A�1*

loss�$d;�+�Y       �	eT�Ifc�A�1*

loss$�<Ib_h       �	1��Ifc�A�1*

loss��:B���       �	���Ifc�A�1*

loss_/:�Dk       �	���Ifc�A�1*

loss�|:��       �	.;�Ifc�A�1*

lossS]�<p��       �	��Ifc�A�1*

loss��a<XGUF       �	�p�Ifc�A�1*

lossS2�<�u�N       �	l�Ifc�A�1*

loss��-<3�d�       �	#��Ifc�A�1*

loss���<�L�       �	�6�Ifc�A�1*

loss�v�;6�'�       �	���Ifc�A�1*

lossZ��;��0�       �	�d�Ifc�A�1*

loss��c<-�B=       �	w��Ifc�A�1*

loss�	`:	��h       �	���Ifc�A�1*

loss�1"<mX��       �	�(�Ifc�A�1*

loss{H$<vM�b       �	���Ifc�A�1*

loss(�&=ѫ��       �	�[�Ifc�A�1*

loss��;��>       �	���Ifc�A�1*

loss�P=9���       �	Ė�Ifc�A�1*

loss�2;����       �	<-�Ifc�A�1*

loss9�:����       �	"��Ifc�A�1*

loss��4:1n`�       �	S\�Ifc�A�1*

lossX�;�5n!       �	���Ifc�A�1*

lossl�+<3k��       �	��Ifc�A�1*

loss<7�;u�       �	,(�Ifc�A�1*

loss}T;���       �	���Ifc�A�1*

loss`Y�<���       �	�U�Ifc�A�1*

loss��}9�<��       �	���Ifc�A�1*

loss�V�:̏W'       �	��Ifc�A�1*

loss��;3�       �	,* Jfc�A�1*

loss(�<
��S       �	�� Jfc�A�1*

loss
B<)���       �	�UJfc�A�1*

loss��<����       �	��Jfc�A�1*

lossWc�;�8�\       �	��Jfc�A�1*

lossY;=&�1       �	_%Jfc�A�1*

lossV�<Xh�=       �	b�Jfc�A�1*

lossX�%<롱       �	�dJfc�A�1*

loss3:�<���       �	z�Jfc�A�1*

loss�I�<���       �	��Jfc�A�1*

lossd�p<a7�       �	r1Jfc�A�1*

loss��+:X���       �	�Jfc�A�1*

lossV1;�Ը�       �	�bJfc�A�1*

lossO�L=?pn'       �	��Jfc�A�1*

loss��3<u�y�       �	ڍJfc�A�1*

loss�;ǀC       �	�%	Jfc�A�1*

loss�Å=5/�P       �	׽	Jfc�A�1*

loss��.;ǰ�       �	�O
Jfc�A�1*

loss�?=�Z�       �	��
Jfc�A�1*

loss[��;���{       �	(~Jfc�A�1*

loss�=|ā:       �	�Jfc�A�1*

loss|G�8j�N�       �	`�Jfc�A�1*

loss:!9f*J       �	8LJfc�A�1*

loss�P8<�P�       �	��Jfc�A�1*

loss���<E�L       �	_Jfc�A�1*

lossi]A<�D~�       �	Jfc�A�1*

loss4��=��U       �	z�Jfc�A�1*

loss-r<�4�	       �	]Jfc�A�1*

loss�;����       �	��Jfc�A�1*

loss<��;����       �	�Jfc�A�1*

loss���=�Mn6       �	Y2Jfc�A�1*

lossM�=<�5W       �	��Jfc�A�1*

loss�FW:{<       �	+lJfc�A�1*

lossu<�L�W       �	qJfc�A�1*

loss�SV=�@iL       �	0�Jfc�A�1*

loss��	;S�[w       �	:Jfc�A�1*

lossA�;�_/       �	�Jfc�A�1*

lossҳ7:e:-       �	KsJfc�A�1*

lossC��;��A�       �	�Jfc�A�1*

loss�q�<ł�?       �	��Jfc�A�1*

loss3�P<�-i]       �	�:Jfc�A�1*

loss	t�<��v�       �	�Jfc�A�1*

lossH�*<m%�       �	�Jfc�A�1*

loss��;�
�       �	Jfc�A�1*

loss�!�=�LL&       �	.�Jfc�A�1*

loss�j�<�~�       �	�fJfc�A�1*

loss��c=��,�       �	�Jfc�A�1*

loss�;��R       �	s�Jfc�A�1*

loss�g=I:       �	�BJfc�A�1*

losst�s=6gL�       �	R�Jfc�A�1*

loss!Z <�ɳ�       �	~pJfc�A�1*

lossC��:d@1U       �	�Jfc�A�1*

lossT��<@2c�       �	��Jfc�A�1*

lossl��9��=�       �	!@ Jfc�A�1*

loss1�;M�D       �	�� Jfc�A�1*

lossc_r;�>�@       �	cz!Jfc�A�1*

loss��$;;�)G       �	�"Jfc�A�1*

loss��T<�e�       �	פ"Jfc�A�1*

loss-��:�:       �	:#Jfc�A�1*

lossc�R=�w�J       �	�#Jfc�A�1*

loss�f�< ��       �	�c$Jfc�A�1*

lossf��;�&�       �	�$Jfc�A�1*

loss(#&<��~       �	�%Jfc�A�1*

loss��<K���       �	�(&Jfc�A�1*

loss��=��u�       �	��&Jfc�A�1*

lossiƳ:�Cb�       �	�T'Jfc�A�1*

loss�6;@�       �	`�'Jfc�A�1*

loss$�<1ڳ�       �	$�(Jfc�A�1*

loss��Y<��f       �	q)Jfc�A�1*

loss=	@R�       �	0�)Jfc�A�1*

loss$�<&0       �	�W*Jfc�A�1*

lossc=�;B��       �	��*Jfc�A�1*

loss��:i�_       �	P�+Jfc�A�1*

loss���;A_��       �	��,Jfc�A�1*

loss�;�q�[       �	�G-Jfc�A�1*

lossXx=#1�       �	,.Jfc�A�1*

loss���:[:H       �	L�.Jfc�A�1*

loss�`�;T��r       �	�Z/Jfc�A�1*

loss��: L�       �	��/Jfc�A�1*

loss�<魜�       �	Ӈ0Jfc�A�1*

loss�;��6�       �	�1Jfc�A�1*

loss���:��        �	��2Jfc�A�2*

loss���;�ic�       �	�v3Jfc�A�2*

loss��;��E       �	�4Jfc�A�2*

loss��5=�@��       �	��5Jfc�A�2*

loss!ʅ;m'�X       �	�{6Jfc�A�2*

loss��;�8�s       �	7Jfc�A�2*

loss(�<��3       �	�8Jfc�A�2*

loss-�<���v       �	T�8Jfc�A�2*

lossrE/=>�       �	vP9Jfc�A�2*

loss`��;�!^%       �	�:Jfc�A�2*

loss
߫;.	4       �	ܜ<Jfc�A�2*

loss[l<y���       �	�6=Jfc�A�2*

loss-/B=cH�       �	��=Jfc�A�2*

loss$d�:i��       �	Vf>Jfc�A�2*

loss��"=e�b       �	�0?Jfc�A�2*

loss�X�<��I�       �	��?Jfc�A�2*

loss&�;!xk�       �	�b@Jfc�A�2*

loss���<�S�p       �	�AJfc�A�2*

lossF'�;r�j�       �	��AJfc�A�2*

loss|\�:Yz/'       �	��BJfc�A�2*

loss.0�<͓��       �	y<CJfc�A�2*

loss3��<�\��       �	H�CJfc�A�2*

loss&q�9����       �	~DJfc�A�2*

loss�Y�:x�^�       �	�EJfc�A�2*

loss���=��Q       �	��EJfc�A�2*

lossʺ<�#˒       �	�lFJfc�A�2*

lossA�=�N�       �	�GJfc�A�2*

lossh�<���	       �	b�GJfc�A�2*

loss���<ڮ�       �	6HJfc�A�2*

loss���;���       �	�HJfc�A�2*

loss�]�:IS       �	,gIJfc�A�2*

loss1�`<!�n�       �	�IJfc�A�2*

loss�q;H��       �	2�JJfc�A�2*

loss��;7R��       �	�&KJfc�A�2*

loss�p=u]       �	��KJfc�A�2*

loss���<�΄E       �	;SLJfc�A�2*

losstA�;FY�b       �	F�LJfc�A�2*

lossRx�<Z��       �	|�MJfc�A�2*

lossC>�<�X�       �	2NJfc�A�2*

loss)�< %��       �	�NJfc�A�2*

lossv�;�p8Z       �	�MOJfc�A�2*

loss+ <��@)       �	�OJfc�A�2*

loss��<�@�=       �	[�PJfc�A�2*

loss�ͧ;��       �	i5QJfc�A�2*

loss�R�;2�       �	�QJfc�A�2*

lossqD�9�ݠ�       �	`�RJfc�A�2*

lossp#<��1       �	�SJfc�A�2*

lossN��=С�       �	y!TJfc�A�2*

loss�8!<��       �	��TJfc�A�2*

loss�TN<�!e       �	 oUJfc�A�2*

loss��t<��       �	�
VJfc�A�2*

loss<I!<��]�       �	3�VJfc�A�2*

lossH��;	l�9       �	�BWJfc�A�2*

loss��S<l��       �	r�WJfc�A�2*

loss���;<%Mc       �	}XJfc�A�2*

loss���=�K<i       �	�YJfc�A�2*

lossS�{=�x�       �	#�YJfc�A�2*

lossJ�.<<0|A       �	(bZJfc�A�2*

loss��j<"���       �	`�\Jfc�A�2*

loss=9�:pfB}       �	qT]Jfc�A�2*

lossU�;i^�}       �	��]Jfc�A�2*

loss�i,<��Ծ       �	��^Jfc�A�2*

loss�H�<~n��       �	B?_Jfc�A�2*

lossT!=�cH�       �	&�_Jfc�A�2*

losst5�<��       �	Ӆ`Jfc�A�2*

loss�1:�1��       �	%#aJfc�A�2*

loss24;�ҏ       �	e5bJfc�A�2*

lossRKS<��u       �	M�bJfc�A�2*

loss#�:^�L       �	�zcJfc�A�2*

loss�F<��iI       �	�dJfc�A�2*

loss�N:�
iz       �	�dJfc�A�2*

loss�{!=
5�;       �	�heJfc�A�2*

loss��}:�W{o       �	�fJfc�A�2*

loss<� <�̼       �	ݶfJfc�A�2*

lossj��;��Mf       �	�SgJfc�A�2*

loss��,;.���       �	=�iJfc�A�2*

loss_	�:���&       �	'jJfc�A�2*

loss�3\:���       �	��jJfc�A�2*

loss�"�;�^s?       �	ӈkJfc�A�2*

loss@[4<L���       �	��lJfc�A�2*

losso��=���p       �	�[mJfc�A�2*

lossqw<s
�       �	�?nJfc�A�2*

loss(�1<���       �	@�nJfc�A�2*

lossW�2;�]�       �	��oJfc�A�2*

loss��	9�tۚ       �	x(pJfc�A�2*

lossque<����       �	�pJfc�A�2*

loss��x=	~�,       �	#�qJfc�A�2*

loss%�&;�?ڕ       �	�trJfc�A�2*

loss���<�מ�       �	�sJfc�A�2*

lossT�#;"���       �	�sJfc�A�2*

loss�xo<�i       �	�ntJfc�A�2*

lossJ�;P��5       �	�uJfc�A�2*

loss�>�;6<rH       �	�fvJfc�A�2*

lossw�;g��       �	wJfc�A�2*

loss�z�:(�BH       �	�wJfc�A�2*

losssҗ:�5�       �	�LxJfc�A�2*

loss���;�	9�       �	_�xJfc�A�2*

loss�ج<�|�^       �	�1zJfc�A�2*

loss���<��7�       �	��zJfc�A�2*

loss�bt=�$��       �	��{Jfc�A�2*

loss�J�=��        �	��|Jfc�A�2*

loss�nR=�,1       �	g(}Jfc�A�2*

lossM:|G�_       �	��}Jfc�A�2*

lossSi�;��N�       �	�[~Jfc�A�2*

lossH�;�.Ji       �	~�~Jfc�A�2*

lossi�y=px�=       �	$�Jfc�A�2*

loss�n;b�F       �	�9�Jfc�A�2*

lossO�<�(��       �	�ЀJfc�A�2*

lossEf<;���       �	�z�Jfc�A�2*

loss�=<FڂU       �	��Jfc�A�2*

lossCw�<�e�       �	���Jfc�A�2*

loss4d�;�j�       �	}��Jfc�A�2*

lossɽ�:�jN�       �	9*�Jfc�A�2*

loss�:,X��       �	���Jfc�A�2*

loss?�<���       �	�d�Jfc�A�2*

loss֡�; ���       �	e�Jfc�A�2*

loss@x[<v�^I       �	k��Jfc�A�2*

loss ��;�m       �	�.�Jfc�A�2*

loss�$:��6       �	ećJfc�A�2*

loss)��;7j�+       �	[�Jfc�A�2*

loss��;��PV       �	��Jfc�A�2*

losssK;�R�       �	�}�Jfc�A�2*

loss��/;�ƻd       �	: �Jfc�A�2*

lossT:�       �	.ŊJfc�A�2*

loss�ت<$!ҫ       �	ao�Jfc�A�2*

lossL�L<N_s�       �	F�Jfc�A�2*

loss�6:V�f       �	��Jfc�A�2*

lossA^<Õ(�       �	T9�Jfc�A�2*

loss#�0<���       �	�ЍJfc�A�3*

lossI$=���@       �	m�Jfc�A�3*

loss��;;i��       �	��Jfc�A�3*

loss�3;���       �	Y��Jfc�A�3*

loss�h�<$�w:       �	[@�Jfc�A�3*

loss6�H;R�~       �	�אJfc�A�3*

loss2��;�'�7       �	zl�Jfc�A�3*

loss�C;J��       �	��Jfc�A�3*

lossy�:��U�       �	|��Jfc�A�3*

lossd(<��4       �	�2�Jfc�A�3*

loss��;�M��       �	�ÓJfc�A�3*

loss/k8���       �	Y�Jfc�A�3*

loss;e�:�:R       �	��Jfc�A�3*

lossE��7�3�|       �	탕Jfc�A�3*

loss�Aj8�R�p       �	��Jfc�A�3*

loss�8���       �	ꮖJfc�A�3*

lossK��9h�D�       �	D�Jfc�A�3*

loss���:��Xs       �	EחJfc�A�3*

lossH^;�$�'       �	n�Jfc�A�3*

loss_0�8��7�       �	�\�Jfc�A�3*

loss��80��       �	C�Jfc�A�3*

loss�b�=�d�       �	՚Jfc�A�3*

loss1_:j��%       �	��Jfc�A�3*

loss]��=I���       �	�ڜJfc�A�3*

loss�9:I��       �	��Jfc�A�3*

loss_��<p�@�       �		�Jfc�A�3*

loss���9��*�       �	h=�Jfc�A�3*

loss���;u�/>       �	\��Jfc�A�3*

loss��:=Ot��       �	�ãJfc�A�3*

lossTz�<��       �	r�Jfc�A�3*

lossA �;���       �	���Jfc�A�3*

loss�Pz<�ی�       �	kJ�Jfc�A�3*

lossFt�:�6       �	���Jfc�A�3*

loss�Ƀ<�#        �	=�Jfc�A�3*

loss��G<h�c       �	�I�Jfc�A�3*

loss�I�<��        �	_�Jfc�A�3*

loss�<~&h�       �	|&�Jfc�A�3*

loss>F�="��=       �	�ܫJfc�A�3*

loss4��<
1Z)       �	�ѬJfc�A�3*

loss?Q�;\�zO       �	�
�Jfc�A�3*

loss?{=ӂ��       �	}�Jfc�A�3*

lossJ�<y,��       �	��Jfc�A�3*

loss�+k:-�        �	��Jfc�A�3*

lossv�8;��       �	���Jfc�A�3*

loss �;����       �	�<�Jfc�A�3*

loss�}\;�&�|       �	%�Jfc�A�3*

loss	��9�I�       �	��Jfc�A�3*

loss&�];"|�m       �	�Jfc�A�3*

loss��;�كj       �	��Jfc�A�3*

loss���;*�z       �	��Jfc�A�3*

loss���<��!       �	��Jfc�A�3*

lossd��:���R       �	�ڸJfc�A�3*

loss��L:p ��       �	L��Jfc�A�3*

loss�M<h!��       �	Q�Jfc�A�3*

loss3Rs<�̉       �	gG�Jfc�A�3*

loss��<n��       �	�Jfc�A�3*

loss��<U&��       �	ټJfc�A�3*

loss)7�;�)��       �	gֽJfc�A�3*

loss��:E�       �	y�Jfc�A�3*

loss4\<���       �	��Jfc�A�3*

loss|��<����       �	eſJfc�A�3*

loss=�3<�I�`       �	8��Jfc�A�3*

loss�/<;Z���       �	*�Jfc�A�3*

loss@�0;�]��       �	F��Jfc�A�3*

loss�ĭ:�O        �	�Y�Jfc�A�3*

lossԢ};Ca��       �	���Jfc�A�3*

loss�=
�N<       �	\��Jfc�A�3*

lossA�S=+�@�       �	�1�Jfc�A�3*

loss�3�<����       �	���Jfc�A�3*

loss�#�8�ĝa       �	�i�Jfc�A�3*

loss�A|<�xF5       �	��Jfc�A�3*

loss`E:lNK�       �	���Jfc�A�3*

loss��<����       �	�?�Jfc�A�3*

loss��<�ٲ�       �	�E�Jfc�A�3*

lossjI�<Hm�       �	*��Jfc�A�3*

loss��;�/�l       �	�w�Jfc�A�3*

loss�an<	z��       �	���Jfc�A�3*

loss�0<)?�H       �	.S�Jfc�A�3*

loss�x0=���       �	y��Jfc�A�3*

loss��=a�]       �	�x�Jfc�A�3*

loss���;�O4       �	N(�Jfc�A�3*

loss��<���       �	^��Jfc�A�3*

loss�O�;�qi�       �	fO�Jfc�A�3*

loss�t;�=�       �	;��Jfc�A�3*

loss�I<ȴ�       �	�z�Jfc�A�3*

loss3��=չkH       �	��Jfc�A�3*

loss�e;���W       �	ۦ�Jfc�A�3*

loss��7:ܨE       �	�:�Jfc�A�3*

loss���:�[I�       �	p��Jfc�A�3*

loss3� 8ƽ�V       �	C��Jfc�A�3*

loss1Ci9����       �	)\�Jfc�A�3*

loss6`�;�ޝ}       �	e�Jfc�A�3*

loss�@�<�}�O       �	$��Jfc�A�3*

lossAs�:��>�       �	�J�Jfc�A�3*

loss��<��$       �	���Jfc�A�3*

loss�= :K-M=       �	��Jfc�A�3*

loss�T�;W��N       �	H2�Jfc�A�3*

loss��<dBW�       �	3��Jfc�A�3*

loss��H:i �G       �	�u�Jfc�A�3*

loss�s�:�XE�       �	�/�Jfc�A�3*

losse/;���       �	���Jfc�A�3*

loss���<}\p       �	�l�Jfc�A�3*

loss(,<E4��       �	��Jfc�A�3*

loss���;�q��       �	e��Jfc�A�3*

loss��;,+h;       �	�7�Jfc�A�3*

lossd�^</��       �	\��Jfc�A�3*

loss��;��E�       �	�l�Jfc�A�3*

loss3_�:� 	�       �	��Jfc�A�3*

lossl��:X��       �	Ͻ�Jfc�A�3*

loss��0<�:�       �	Z�Jfc�A�3*

loss�Zl<
8�I       �	���Jfc�A�3*

loss��S;V�)        �	���Jfc�A�3*

lossF�<���       �	�9�Jfc�A�3*

lossJ�R;rt�t       �	���Jfc�A�3*

lossOq�;�k1�       �	�6�Jfc�A�3*

loss��;����       �	:��Jfc�A�3*

loss|�!<��       �	�i�Jfc�A�3*

losst�:��
�       �	((�Jfc�A�3*

losss|<Q��n       �	FE�Jfc�A�3*

lossJ�D<�Ǚ|       �	B��Jfc�A�3*

lossY�;G�c       �	���Jfc�A�3*

lossxB:cI`       �	��Jfc�A�3*

loss	��9-�P       �	���Jfc�A�3*

loss�8 =�        �	D Kfc�A�3*

lossL!<6��R       �	s� Kfc�A�3*

loss�wb:᳣       �	s�Kfc�A�3*

loss@P<KE!       �	�1Kfc�A�3*

loss���;��El       �	.�Kfc�A�3*

loss��9R�{       �	YKfc�A�4*

lossqC�;�%͜       �	��Kfc�A�4*

loss#�69�D�       �	A~Kfc�A�4*

loss�N:�Nq�       �	MKfc�A�4*

lossv�:Nq[:       �	D�Kfc�A�4*

lossᛢ=�r�8       �	~;Kfc�A�4*

loss�\�;�u��       �	��Kfc�A�4*

loss�%G;3�
m       �	2rKfc�A�4*

loss$��:���P       �	S	Kfc�A�4*

loss,;�hH�       �	�Kfc�A�4*

loss��^:7d��       �	�?	Kfc�A�4*

loss#��<�       �	?�	Kfc�A�4*

loss��:����       �	�y
Kfc�A�4*

loss1:<�u�       �	Kfc�A�4*

loss�Ь=��n�       �	"�Kfc�A�4*

loss�}L<�Iz�       �	h>Kfc�A�4*

lossTzZ; ���       �	�Kfc�A�4*

loss���<�U?J       �	oKfc�A�4*

loss3~d8��[�       �	�Kfc�A�4*

loss	�)<�	       �	U�Kfc�A�4*

loss��;{"'�       �	�>Kfc�A�4*

loss2��8�͞�       �	��Kfc�A�4*

loss:�;��       �	�uKfc�A�4*

loss��u:	x9       �	wKfc�A�4*

loss�;<�6��       �	�Kfc�A�4*

loss��=p��       �	l[Kfc�A�4*

loss�R<�]��       �	L�Kfc�A�4*

loss�/s;��       �	͓Kfc�A�4*

loss*6<Ԭ�       �	�4Kfc�A�4*

loss�q9tƫ�       �	#�Kfc�A�4*

loss��;� qw       �	�xKfc�A�4*

loss���9���       �	wKfc�A�4*

loss��P<�ޙ2       �	��Kfc�A�4*

loss
�;	�N       �	!VKfc�A�4*

loss{:da��       �	TKfc�A�4*

loss�e<�O       �	��Kfc�A�4*

loss�u<q04�       �	&7Kfc�A�4*

lossԨ�;TKv       �	"QKfc�A�4*

lossagN;?�f�       �	�Kfc�A�4*

loss��<7.9�       �	��Kfc�A�4*

loss:��;�Z�a       �	~;Kfc�A�4*

loss��=*�       �	�Kfc�A�4*

loss|i};�f�       �	+�Kfc�A�4*

loss&��<�X�e       �	�!Kfc�A�4*

lossI�j9�#+�       �	��Kfc�A�4*

loss�L{<A)�       �	<kKfc�A�4*

loss#�E:�9��       �	� Kfc�A�4*

loss�:����       �	�� Kfc�A�4*

lossᒌ;�D�_       �	M!Kfc�A�4*

lossc��9?sn0       �	��!Kfc�A�4*

loss�'n;s]cu       �	ׅ"Kfc�A�4*

lossl�:2�D       �	!##Kfc�A�4*

loss���=OO��       �	Z�#Kfc�A�4*

lossn�V;���p       �	+h$Kfc�A�4*

loss��.:M/��       �	%Kfc�A�4*

loss�A=o�       �	ŭ%Kfc�A�4*

loss8\=L���       �	fL&Kfc�A�4*

loss�Q8=�wT       �	��&Kfc�A�4*

loss�u�:e�ס       �	�'Kfc�A�4*

losse��;��P�       �	�4(Kfc�A�4*

loss���;ƛ��       �	�(Kfc�A�4*

loss3X�:�Q��       �	ut)Kfc�A�4*

loss�(=���q       �	,*Kfc�A�4*

lossC�;�hr       �	^�*Kfc�A�4*

lossA�T<����       �	�\+Kfc�A�4*

loss�1�:�ս�       �	�,Kfc�A�4*

lossNV?=$Y��       �	��,Kfc�A�4*

loss�ȷ;u�>b       �	%Z-Kfc�A�4*

loss�^~; x`�       �	��-Kfc�A�4*

loss a�<&��w       �	F�.Kfc�A�4*

lossa��;���       �	@Q/Kfc�A�4*

loss�;��O       �	F�/Kfc�A�4*

lossM�e;1x�8       �	�0Kfc�A�4*

loss�BZ<�rŻ       �	M�1Kfc�A�4*

loss�[�;�J��       �	y!2Kfc�A�4*

lossk <��͗       �	,�2Kfc�A�4*

loss���:��r�       �	�U3Kfc�A�4*

loss�^;P�<       �	
�3Kfc�A�4*

lossnk�<���       �	��4Kfc�A�4*

loss�j�<��=�       �	�(5Kfc�A�4*

loss�1�<�y�O       �	��5Kfc�A�4*

loss=	+<P�R       �	D�6Kfc�A�4*

loss���9m7�       �	:7Kfc�A�4*

lossA;��V�       �	��7Kfc�A�4*

loss��*;DfZ&       �	�h8Kfc�A�4*

loss�{�8���       �	�8Kfc�A�4*

loss���;��>�       �	Ȕ9Kfc�A�4*

loss1�1<���Q       �	�e:Kfc�A�4*

loss�Q<�wN�       �	v;Kfc�A�4*

loss�O
<l��}       �	��;Kfc�A�4*

loss��U;.���       �	K<Kfc�A�4*

loss�3�82��       �	e�<Kfc�A�4*

loss39n:�Y�C       �	y=Kfc�A�4*

loss|��;�*�       �	�>Kfc�A�4*

lossL>�9[�C       �	�?Kfc�A�4*

loss�I�9mH�d       �	�?Kfc�A�4*

loss+��<mE��       �	D�@Kfc�A�4*

loss�f�:���       �	u"AKfc�A�4*

loss8�;�H�       �	b�AKfc�A�4*

loss�2<�;ck       �	yXBKfc�A�4*

loss���7D��       �	��BKfc�A�4*

loss㱚:V�D       �	��CKfc�A�4*

loss�~<ޕ6       �	�"DKfc�A�4*

loss�n<��       �	_�DKfc�A�4*

lossB+<��B�       �	�NEKfc�A�4*

losss�:�nA       �	��EKfc�A�4*

loss/�:���       �	}FKfc�A�4*

loss��;+)�7       �	bGKfc�A�4*

loss��9�b�       �	��GKfc�A�4*

loss�e�;�0�       �	�LHKfc�A�4*

loss%ݙ;$E�       �	��HKfc�A�4*

loss�Pe9�<       �	/�IKfc�A�4*

lossϢ@<��       �	�0JKfc�A�4*

lossh�.:��,�       �	��JKfc�A�4*

lossz!.:i�       �	l�KKfc�A�4*

loss&�:�Bp       �	�TLKfc�A�4*

lossa#9���2       �	��LKfc�A�4*

loss�1x9��Q�       �		�MKfc�A�4*

loss ��;��p       �	#NKfc�A�4*

loss��9x1^S       �	6�NKfc�A�4*

lossﳐ;��?�       �	�oOKfc�A�4*

lossX��=Y�:       �	 PKfc�A�4*

loss�7;�g�       �	�PKfc�A�4*

loss�p=2S��       �	�WQKfc�A�4*

loss=�F:8�v\       �	ORKfc�A�4*

loss��:�T�v       �	_�RKfc�A�4*

loss�S<B��       �	&USKfc�A�4*

loss���:�Ar�       �	b�SKfc�A�4*

loss��	;�U�       �	B�TKfc�A�5*

loss���:��51       �	33UKfc�A�5*

loss�R�:�U��       �	��UKfc�A�5*

loss�u[<|1�       �	dtVKfc�A�5*

loss�U[;�U�       �	�WKfc�A�5*

loss�(<�B^S       �	C�WKfc�A�5*

loss�e=突       �	�KXKfc�A�5*

loss��;�P�p       �	ZGYKfc�A�5*

loss�^�9&�G�       �	�YKfc�A�5*

loss�;����       �	մZKfc�A�5*

lossH��<� K0       �	��[Kfc�A�5*

loss{�e8�||C       �	 $]Kfc�A�5*

lossi�;;�f�h       �	��]Kfc�A�5*

loss;V:x�(,       �	c�^Kfc�A�5*

loss���8ԞQ�       �	��_Kfc�A�5*

loss�|Z=��"       �	��`Kfc�A�5*

loss.�:�k       �	k�aKfc�A�5*

loss,�	<�X"�       �	��bKfc�A�5*

loss	a�<���       �	�KcKfc�A�5*

loss)u�;�6�       �	��cKfc�A�5*

loss�g;,���       �	?�dKfc�A�5*

loss��$=�eޗ       �	s�eKfc�A�5*

loss�;�H       �	�fKfc�A�5*

lossV@�:��       �	��fKfc�A�5*

loss޽;0�d�       �	a�hKfc�A�5*

loss���;�9��       �	XriKfc�A�5*

lossJ)�;eݢ�       �	�jKfc�A�5*

loss�U;W�I�       �	��jKfc�A�5*

loss�ܘ:ݰt�       �	MLkKfc�A�5*

loss�6:��G<       �	��kKfc�A�5*

loss��;���       �	��lKfc�A�5*

loss읦<L��       �	�"mKfc�A�5*

loss���<���O       �	s�mKfc�A�5*

loss�\ =-�EU       �	|`nKfc�A�5*

loss�Bj=Xk��       �	Q�nKfc�A�5*

loss�=]c�       �	N�oKfc�A�5*

loss:;K�W^       �	�6pKfc�A�5*

lossc�+:I�       �	�pKfc�A�5*

loss��?:<9A�       �	b�qKfc�A�5*

lossL�=����       �	�rKfc�A�5*

loss�1-9���       �	l�rKfc�A�5*

loss:[�9fl4       �	�esKfc�A�5*

loss�J:����       �	��sKfc�A�5*

loss3�Z<�+w�       �	Z�tKfc�A�5*

loss@u�<I9�       �	�tuKfc�A�5*

lossK<��S�       �	�vKfc�A�5*

loss �<ހ��       �	��vKfc�A�5*

loss�˟8��j       �	/PwKfc�A�5*

lossP-�9E��       �	�wKfc�A�5*

loss��;���       �	[�xKfc�A�5*

lossq<�;�\i       �	�ryKfc�A�5*

loss;��:Y�w       �	zKfc�A�5*

loss�K�:s�ߢ       �	 �zKfc�A�5*

loss�M�=���t       �	�|Kfc�A�5*

loss�Ռ=��y       �	o�|Kfc�A�5*

loss�\�<fp)�       �	��}Kfc�A�5*

loss#�;gf�       �	W|~Kfc�A�5*

lossV <��w       �	�:Kfc�A�5*

loss��:H)��       �	J�Kfc�A�5*

loss)8Y:|1��       �	�c�Kfc�A�5*

lossJ��<��We       �	!Y�Kfc�A�5*

loss6)=��R_       �	/��Kfc�A�5*

loss��<�Xjl       �	�Kfc�A�5*

loss���;�        �	r3�Kfc�A�5*

loss��:�i       �	�̃Kfc�A�5*

lossm�<}�i�       �	
i�Kfc�A�5*

loss�G�<��J       �	�Kfc�A�5*

loss\G^<��       �	=��Kfc�A�5*

lossψ$=��       �	:�Kfc�A�5*

loss(@<g��       �	�چKfc�A�5*

lossI`�:�qq       �	�|�Kfc�A�5*

loss��0:��D/       �	��Kfc�A�5*

loss��><	F�o       �	��Kfc�A�5*

loss�A�:˃cd       �	�S�Kfc�A�5*

loss�3<�)}�       �	��Kfc�A�5*

loss��&<�EO       �	f��Kfc�A�5*

loss�8m<A��T       �	�Kfc�A�5*

lossc��;�Q�       �	ⰋKfc�A�5*

loss�OH=��W�       �	
J�Kfc�A�5*

loss�}<wT(d       �	zߌKfc�A�5*

loss�g�;
�	       �	�s�Kfc�A�5*

loss�\;@}c       �	�Kfc�A�5*

lossqU�9#tk�       �	���Kfc�A�5*

loss�#�;���~       �	�D�Kfc�A�5*

loss<�'<(hM�       �	�܏Kfc�A�5*

loss*��;��Տ       �	r�Kfc�A�5*

loss�/=-��       �	J	�Kfc�A�5*

lossx!;��.�       �	���Kfc�A�5*

loss{;�<r���       �	�3�Kfc�A�5*

lossC?=�Wn       �	+�Kfc�A�5*

loss�
;f5ܭ       �	�ϔKfc�A�5*

loss�!=@��V       �	���Kfc�A�5*

loss��5=�;       �	�"�Kfc�A�5*

loss��<���       �	̖Kfc�A�5*

lossA{�<��F�       �	p�Kfc�A�5*

loss��:�פ3       �	��Kfc�A�5*

loss�O=��|�       �	ı�Kfc�A�5*

loss?�:(n��       �	eQ�Kfc�A�5*

loss��</.0O       �	>�Kfc�A�5*

loss?-�:�ܕ�       �	=��Kfc�A�5*

lossMG<,�m�       �	@�Kfc�A�5*

loss�t�:��z       �	�Kfc�A�5*

loss�=7+1�       �	tҜKfc�A�5*

loss7��<�+(       �	�n�Kfc�A�5*

loss%�::3�e]       �	Q2�Kfc�A�5*

loss�޴:{�jE       �	{N�Kfc�A�5*

loss�_i;�IZ}       �	��Kfc�A�5*

lossO��:3Kh�       �	G�Kfc�A�5*

loss##<��k�       �	z��Kfc�A�5*

lossR�J9��
�       �	I��Kfc�A�5*

loss1v:�|t�       �	>%�Kfc�A�5*

loss�0�:�<�       �	Q-�Kfc�A�5*

loss��;|6�       �	^��Kfc�A�5*

loss�6C<����       �	A��Kfc�A�5*

loss=,�;b��       �	Hj�Kfc�A�5*

loss��X<�d\       �	�j�Kfc�A�5*

loss�;���#       �	���Kfc�A�5*

lossf�;Ƥ9�       �	Z��Kfc�A�5*

loss�&=k�N�       �	ꖪKfc�A�5*

lossƔ<��I       �	Ho�Kfc�A�5*

loss��<��       �	�"�Kfc�A�5*

lossD\I=ᎈ)       �	5�Kfc�A�5*

loss��-=�hf.       �	b֭Kfc�A�5*

loss	G�:�L�       �	Q��Kfc�A�5*

loss�h):�}��       �	!W�Kfc�A�5*

lossr19�)�       �	��Kfc�A�5*

loss==�9�R��       �	��Kfc�A�5*

loss��f;�s��       �	�C�Kfc�A�5*

loss߯e;.�0K       �	 �Kfc�A�6*

loss,&<Bv+�       �	dy�Kfc�A�6*

loss�=<0�P�       �	.�Kfc�A�6*

lossN�<�9g       �	^��Kfc�A�6*

loss�sf;Ӻ
�       �	�m�Kfc�A�6*

loss�Ƽ<.W�n       �	8�Kfc�A�6*

loss�,�;��,g       �	2��Kfc�A�6*

lossM;>oN3       �	�R�Kfc�A�6*

loss��:��}�       �	���Kfc�A�6*

loss_�!<�$-       �	���Kfc�A�6*

loss��I<��R�       �	4f�Kfc�A�6*

lossV�J;/��>       �	���Kfc�A�6*

lossH��7��G4       �	>B�Kfc�A�6*

loss��;2�2L       �	���Kfc�A�6*

loss�8;ނ�>       �	�3�Kfc�A�6*

lossQn�:���J       �	cнKfc�A�6*

loss��Z<Ϗ��       �	*q�Kfc�A�6*

loss��<5��       �	��Kfc�A�6*

loss�ć<���       �	S��Kfc�A�6*

loss(˰:9���       �	�M�Kfc�A�6*

loss�A�:��       �	l��Kfc�A�6*

loss�<�A7�       �	!��Kfc�A�6*

loss�\8��       �	~5�Kfc�A�6*

lossH��;yr5m       �	���Kfc�A�6*

loss62*;㖷       �	�s�Kfc�A�6*

lossv��<UT�]       �	#�Kfc�A�6*

loss�n\:ߞ       �	g��Kfc�A�6*

loss��<X�Z�       �	W�Kfc�A�6*

loss�_�9�׆�       �	��Kfc�A�6*

loss,�;��q       �	Ҏ�Kfc�A�6*

loss��=:dY��       �	,+�Kfc�A�6*

loss���:���Q       �	���Kfc�A�6*

loss��Y:"��       �	jg�Kfc�A�6*

loss�
=R��       �	���Kfc�A�6*

loss���;u���       �	��Kfc�A�6*

loss��;]n�u       �	�3�Kfc�A�6*

loss��:#�l       �	���Kfc�A�6*

loss<C7<�nU�       �	�b�Kfc�A�6*

loss���;�d�M       �	��Kfc�A�6*

lossf6A;w�x       �	���Kfc�A�6*

lossf˃9���       �	�A�Kfc�A�6*

loss�G�<&��       �	���Kfc�A�6*

loss��S;9��w       �	}v�Kfc�A�6*

loss��<s�
�       �	��Kfc�A�6*

loss��*:��o       �	���Kfc�A�6*

loss8�<���       �	�D�Kfc�A�6*

lossȭ�=F8@       �	��Kfc�A�6*

loss�\�9�[�       �	Po�Kfc�A�6*

loss� �:3��4       �	.�Kfc�A�6*

loss�YO<�i~�       �	��Kfc�A�6*

loss�G:vމ]       �	�2�Kfc�A�6*

loss �:j�ib       �	���Kfc�A�6*

loss#�<;�       �	�k�Kfc�A�6*

loss��<�b�Z       �	��Kfc�A�6*

lossrg#<���       �	M��Kfc�A�6*

loss��<p�L       �	tz�Kfc�A�6*

loss\�F;p�kn       �	� �Kfc�A�6*

lossv�.:�B%�       �	���Kfc�A�6*

loss=�<��(�       �	�u�Kfc�A�6*

loss@�:�+       �	R�Kfc�A�6*

loss��W9o.b\       �	B��Kfc�A�6*

loss
^�:x��?       �	���Kfc�A�6*

loss[:�:�H�F       �	���Kfc�A�6*

loss,�q;WX~?       �	'�Kfc�A�6*

loss;
�;�M�       �	_�Kfc�A�6*

loss��;zn;<       �	��Kfc�A�6*

lossh�<����       �	kJ�Kfc�A�6*

loss��	<��k       �	��Kfc�A�6*

loss*s9� �       �	�?�Kfc�A�6*

lossܽ�9���U       �	���Kfc�A�6*

loss� =��       �	��Kfc�A�6*

lossX��=V�f       �	�(�Kfc�A�6*

loss��l<�J�]       �	���Kfc�A�6*

loss�e8��       �	w�Kfc�A�6*

lossL� =��
       �	4�Kfc�A�6*

loss8)9<X(&t       �	���Kfc�A�6*

loss�0<�'R�       �	{I�Kfc�A�6*

losswC<k!R       �	���Kfc�A�6*

loss�9�.��       �	�}�Kfc�A�6*

lossr�!=}�2       �	q�Kfc�A�6*

loss)Y{:~%�       �	���Kfc�A�6*

loss��9'�;*       �	r�Kfc�A�6*

loss���:�i0       �	��Kfc�A�6*

lossTh�<�~1�       �	b��Kfc�A�6*

loss�~=	�j       �	j�Kfc�A�6*

lossG|:mb��       �	��Kfc�A�6*

loss�[�:ė�f       �	6��Kfc�A�6*

loss�2�;^���       �	+Q�Kfc�A�6*

loss!	�;Z}�       �	���Kfc�A�6*

losss��:��_       �	���Kfc�A�6*

loss�@�<q�e(       �	r1�Kfc�A�6*

loss�c�;d6`�       �	��Kfc�A�6*

losszV<F�.5       �	�k�Kfc�A�6*

loss��=���I       �	R�Kfc�A�6*

lossb��<�_��       �	ߦ�Kfc�A�6*

lossH�~<b�T�       �	�=�Kfc�A�6*

loss��s;A��       �	���Kfc�A�6*

loss���:�ޗu       �	q�Kfc�A�6*

loss��';�h�       �	��Kfc�A�6*

loss#�<BIKJ       �	��Kfc�A�6*

loss̠�<,��       �	�4�Kfc�A�6*

lossQ�:���4       �	��Kfc�A�6*

loss�>Q;�wع       �	Cr�Kfc�A�6*

loss�U�;��A�       �	S	�Kfc�A�6*

loss�CC=`��       �	ݲ�Kfc�A�6*

loss�]8Z�WV       �	�J�Kfc�A�6*

loss�M:f��!       �	�,�Kfc�A�6*

loss[';��)       �	���Kfc�A�6*

loss�D=�02	       �	�q�Kfc�A�6*

loss9�9��l�       �	�Kfc�A�6*

loss(�%:���       �	��Kfc�A�6*

lossH�;��Z9       �	%[�Kfc�A�6*

loss�7�;7�wt       �	)	 Lfc�A�6*

loss�V�;��f       �	� Lfc�A�6*

loss�2�;�vN7       �	hXLfc�A�6*

loss6�:�ͪ       �	��Lfc�A�6*

lossdJ�=�B�       �	��Lfc�A�6*

loss��o<�a4       �	�4Lfc�A�6*

loss��9�̬2       �	y�Lfc�A�6*

loss� =�d�P       �	�jLfc�A�6*

loss�4:�hi�       �	Lfc�A�6*

loss�ӻ9vD~       �	u�Lfc�A�6*

loss�;��       �	�PLfc�A�6*

lossT@;�FH       �	S�Lfc�A�6*

loss<�;C�z`       �	��Lfc�A�6*

loss�;5;�oC�       �	�Lfc�A�6*

loss�B�;gJ~F       �	عLfc�A�6*

loss f:;U��       �	mY	Lfc�A�6*

loss��;\��       �	��	Lfc�A�7*

loss߾<g���       �	ȕ
Lfc�A�7*

loss1=�;*YiH       �	P4Lfc�A�7*

loss�x3==_o       �	2�Lfc�A�7*

loss�0�=�m        �	aLfc�A�7*

loss�N�8Ŭ��       �	�Lfc�A�7*

loss��<�~��       �	��Lfc�A�7*

loss��!<����       �	�@Lfc�A�7*

loss)�8���t       �	��Lfc�A�7*

loss�+�;�X�       �	�vLfc�A�7*

loss(�(;�ҡ       �	Lfc�A�7*

loss#
�:�*E       �	��Lfc�A�7*

loss�7<��       �	�SLfc�A�7*

lossMN�:�G�       �	��Lfc�A�7*

losse��=���K       �	.�Lfc�A�7*

loss���<�&:z       �	F&Lfc�A�7*

loss���<�W�       �	a�Lfc�A�7*

loss�u<(�,       �	[]Lfc�A�7*

loss��;�|8b       �	#�Lfc�A�7*

loss�װ;��e       �	v�Lfc�A�7*

lossa�<�>H�       �	�&Lfc�A�7*

loss� �:�J1�       �	�Lfc�A�7*

loss��9����       �	ߦLfc�A�7*

loss��;�7��       �	�@Lfc�A�7*

loss���=(7�@       �	�Lfc�A�7*

loss�ZV<��j       �	��Lfc�A�7*

lossA~+<Am?�       �	z�Lfc�A�7*

loss�#�;����       �	aLfc�A�7*

loss�k�:%�t�       �	�Lfc�A�7*

lossה�9�       �	r�Lfc�A�7*

lossX�^8��kl       �	��Lfc�A�7*

loss�-�=�[�       �	6�Lfc�A�7*

loss.��;��Q�       �	ޯLfc�A�7*

lossO=;��       �	8�"Lfc�A�7*

loss�mr=��
       �	H#Lfc�A�7*

loss��;��>(       �	��#Lfc�A�7*

loss�p�<C��y       �	X�$Lfc�A�7*

loss���:�g��       �	l#&Lfc�A�7*

loss4�<=�       �	U�&Lfc�A�7*

loss4rF<����       �	6r'Lfc�A�7*

loss��=��       �	�5(Lfc�A�7*

lossIV$>����       �	�[)Lfc�A�7*

lossj�};Y��=       �	ٓ*Lfc�A�7*

loss/��<���L       �	�j+Lfc�A�7*

loss�ݹ:�u�       �	.�,Lfc�A�7*

loss&�H=�B^       �	�.-Lfc�A�7*

loss�s�;E��       �	�-Lfc�A�7*

loss�|�:�M�#       �	�.Lfc�A�7*

lossVV�<��8       �	�b/Lfc�A�7*

loss;�<��=|       �	�0Lfc�A�7*

loss���;�Z�       �	��0Lfc�A�7*

loss�;��P       �	��1Lfc�A�7*

loss,�;�sj       �	��2Lfc�A�7*

lossv��:E�=�       �	�g3Lfc�A�7*

loss��3;�       �	�14Lfc�A�7*

loss��:In�N       �	�5Lfc�A�7*

loss��,=WcC       �	26Lfc�A�7*

loss��;:�hk       �	��7Lfc�A�7*

lossq;6�f1       �	�	9Lfc�A�7*

loss0V<P�Y�       �	��9Lfc�A�7*

loss�ʋ;e�c       �	ZJ:Lfc�A�7*

loss��g<�8�       �	�	;Lfc�A�7*

loss��:����       �	��<Lfc�A�7*

lossCR<Y)��       �	�>Lfc�A�7*

loss,�;8�&       �	�P?Lfc�A�7*

lossI`=�       �	4�?Lfc�A�7*

loss%��;7��       �	S�@Lfc�A�7*

loss�@�<�=	       �	�5ALfc�A�7*

loss�s;R(��       �	]�ALfc�A�7*

loss7��<_���       �	NzBLfc�A�7*

loss�qp<	q�v       �	*CLfc�A�7*

loss�":�ᰖ       �	R�CLfc�A�7*

lossl+[<��N       �	[DLfc�A�7*

loss��9ʫ+�       �	��DLfc�A�7*

lossh;(A�\       �	��ELfc�A�7*

loss��b;NH��       �	j0FLfc�A�7*

loss�/=d߬�       �	��FLfc�A�7*

loss���<7C�       �	�|GLfc�A�7*

loss8�;/��       �	HHLfc�A�7*

loss�Ε:ed       �	̲HLfc�A�7*

loss��=[=�       �	TnILfc�A�7*

loss�U�9���       �	5JLfc�A�7*

loss2�;���)       �	=�JLfc�A�7*

loss3U�8U��%       �	Y2KLfc�A�7*

loss�H�<Fl�       �	��KLfc�A�7*

loss�h
<���       �	�cLLfc�A�7*

loss �9�T�       �	/�LLfc�A�7*

lossC<%�       �	7�MLfc�A�7*

loss�=l]�       �	�!NLfc�A�7*

losso�F=��       �	��NLfc�A�7*

lossI!<=[�+       �	wJOLfc�A�7*

loss��;�LN�       �	��OLfc�A�7*

loss��8<����       �	�-QLfc�A�7*

loss��N9�$_�       �	��QLfc�A�7*

loss�E�9�c��       �	iRLfc�A�7*

lossٶ�;}DX       �	�SLfc�A�7*

lossD0:�K>R       �	�SLfc�A�7*

loss��;E�0
       �	HTLfc�A�7*

loss9f�;�)�       �	�)ULfc�A�7*

loss[��8j��\       �	G�ULfc�A�7*

loss�D=��1       �	hVLfc�A�7*

loss��: �Gt       �	6WLfc�A�7*

loss�p�8�q*       �	��WLfc�A�7*

loss�ϯ7~\��       �	.:XLfc�A�7*

lossN�;�Ҷ_       �	g�XLfc�A�7*

loss��;'�A�       �	�kYLfc�A�7*

loss���;{/	�       �	�ZLfc�A�7*

loss���;��U       �	��ZLfc�A�7*

loss� ;�#�       �	L7[Lfc�A�7*

loss\��=�2�       �	��[Lfc�A�7*

loss�9;��       �	_~\Lfc�A�7*

loss��=��#�       �	,d]Lfc�A�7*

losshc =�<��       �	)^Lfc�A�7*

loss` r;�:       �	�_Lfc�A�7*

loss(�:�e�~       �	ø_Lfc�A�7*

lossu�;o�^�       �	�v`Lfc�A�7*

loss���;���       �	�aLfc�A�7*

lossE��<�47       �	B�aLfc�A�7*

loss��<�G       �	IbLfc�A�7*

loss[7;Cb,�       �	"�bLfc�A�7*

loss�=�:�l       �	ycLfc�A�7*

lossר%<O~��       �	/dLfc�A�7*

loss��&=��-       �	X�dLfc�A�7*

loss�<�:e*z�       �	GXeLfc�A�7*

lossv �<���       �	V�eLfc�A�7*

loss��$=�Q�;       �	��fLfc�A�7*

loss�n;;�u��       �	�2gLfc�A�7*

loss�"�;Ս�x       �	��gLfc�A�7*

loss�u!<E~d�       �	m�iLfc�A�8*

loss��:���       �	�ojLfc�A�8*

lossx;�8��xp       �	kLfc�A�8*

loss��<b���       �	�kLfc�A�8*

loss���<$ٜ       �	�blLfc�A�8*

lossX�}9t@�       �	�mLfc�A�8*

loss�E�9�3�|       �	��mLfc�A�8*

loss�۲:��.�       �	�@nLfc�A�8*

loss�3.=�@(�       �	R�nLfc�A�8*

loss ��<���s       �	�{oLfc�A�8*

loss�dy<�|*n       �	�FpLfc�A�8*

loss��s<�l7       �	�pLfc�A�8*

lossl8:�m?R       �	 sqLfc�A�8*

lossj\w=���       �	S	rLfc�A�8*

loss��<�1�       �	�rLfc�A�8*

loss�f:n��       �	K?sLfc�A�8*

loss	#:�P֙       �	��sLfc�A�8*

loss���;.�{�       �	\vtLfc�A�8*

loss0��;�\Ȉ       �	��uLfc�A�8*

loss4�<"L��       �	�HvLfc�A�8*

loss#�;у"�       �	{�wLfc�A�8*

loss�u-:/Ž�       �	�zxLfc�A�8*

lossڱ�:��<       �	�yLfc�A�8*

loss�as;�       �	ȷyLfc�A�8*

loss/t�<o�W       �	NbzLfc�A�8*

loss��2;"g�=       �	�{Lfc�A�8*

loss�F=c g       �	��{Lfc�A�8*

loss��<�z:       �	��|Lfc�A�8*

lossR=�C�       �	��}Lfc�A�8*

lossO-;�R��       �	�~Lfc�A�8*

loss�ie<��)       �	I.Lfc�A�8*

loss�K<㐗J       �	��Lfc�A�8*

loss�9<��G�       �	�x�Lfc�A�8*

loss���;�n,