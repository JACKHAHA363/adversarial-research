       �K"	  �nXc�Abrain.Event:2+�$(4�     ���(	a��nXc�A"��
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
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
seed2���
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
:*
Index0*
T0*
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
seed2���*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2��>
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
seed2���*
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
:���������@*
seed2˺*
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
 *   @*
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
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: *
T0
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
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
�
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
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
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
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
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
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
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
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
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�

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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
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
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"k����     >�x�	�4�nXc�AJ��
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
valueB"      *
_output_shapes
:*
dtype0
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
:���������@*
seed2���*
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
:����������*
seed2��>
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
seed���)*
T0*
dtype0*
_output_shapes
:	�
*
seed2���
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
seed2˺
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
divRealDivnum_correct/readadd*
T0*
_output_shapes
: 
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
PlaceholderPlaceholder*
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
:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
_output_shapes
: *
T0
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
Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
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
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
valueB@*    *&
_output_shapes
:@*
dtype0
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
valueB@*    *
dtype0*
_output_shapes
:@
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0��>�       ��-	���nXc�A*

loss�m@��y       ��-	$��nXc�A*

lossϴ@�G
�       ��-	Xr�nXc�A*

loss�/
@�'�       ��-		�nXc�A*

loss���?S��n       ��-	���nXc�A*

loss!��?��!<       ��-	J_�nXc�A*

lossm��?�-��       ��-	��nXc�A*

lossT
�?���       ��-	��nXc�A*

loss&�?g*]        ��-	kb�nXc�A	*

loss���?�T�       ��-	��nXc�A
*

loss��A?[���       ��-	`��nXc�A*

loss��?.}       ��-	Y�nXc�A*

loss��Q?�]*       ��-	��nXc�A*

loss��C?Uv(�       ��-	���nXc�A*

loss��"?vw�!       ��-	�V�nXc�A*

loss��M?��       ��-	0f oXc�A*

loss@��>��"       ��-	�oXc�A*

loss��;?��5�       ��-	�oXc�A*

loss�B?0��       ��-	*XoXc�A*

lossi;C?�5�u       ��-	��oXc�A*

loss�'?�s��       ��-	8�oXc�A*

lossr�?�q��       ��-	�=oXc�A*

loss��2?�<A       ��-	]�oXc�A*

loss�R?�f��       ��-	u�oXc�A*

loss��,?���       ��-	)oXc�A*

loss�Q?`j�D       ��-	��oXc�A*

loss���>��(       ��-	�WoXc�A*

loss��1?9N�        ��-	|�oXc�A*

loss�/?T�z       ��-	ҊoXc�A*

loss!��>iq�4       ��-	�!	oXc�A*

lossN�?�8        ��-	��	oXc�A*

lossS��>-�k       ��-	[^
oXc�A *

lossF�?G,?k       ��-	e�
oXc�A!*

loss��D?� r~       ��-	��oXc�A"*

lossq��>`\Td       ��-	�>oXc�A#*

loss��?9�%       ��-	��oXc�A$*

lossxx�>�f$�       ��-	�voXc�A%*

loss��?4���       ��-	9oXc�A&*

loss�;@? ��       ��-	n�oXc�A'*

loss�6"?��̋       ��-	?oXc�A(*

loss���>=WQ�       ��-	��oXc�A)*

loss��>v)�       ��-	��oXc�A**

loss�Ƽ>���W       ��-	�MoXc�A+*

loss�>��+       ��-	�oXc�A,*

loss�P�>��^�       ��-	|�oXc�A-*

losse?}�}       ��-	&6oXc�A.*

lossw|�>��Z       ��-	[�oXc�A/*

loss)��>wn�       ��-	<hoXc�A0*

loss�G�>�l"        ��-	�oXc�A1*

loss�>I��O       ��-	��oXc�A2*

loss >��%n       ��-	�uoXc�A3*

loss�?���
       ��-	�oXc�A4*

loss
��>�;E�       ��-	�oXc�A5*

loss1�#?��X       ��-	�SoXc�A6*

loss��>��TM       ��-	�oXc�A7*

loss�QO>���t       ��-	 �oXc�A8*

loss�E>>w��       ��-	�ooXc�A9*

loss�\�>��]       ��-	�oXc�A:*

loss�Z?1�;       ��-	�oXc�A;*

loss��>&6       ��-	PoXc�A<*

loss@�>�DV�       ��-	��oXc�A=*

loss�7S>��?-       ��-	��oXc�A>*

loss9E�>����       ��-	SoXc�A?*

lossCF�>xG       ��-	��oXc�A@*

loss�p>��յ       ��-	d�oXc�AA*

lossoUW>��       ��-	�( oXc�AB*

loss��?��ݐ       ��-	 � oXc�AC*

loss��>���       ��-	^!oXc�AD*

loss��?�m��       ��-	G"oXc�AE*

lossEj|>:��       ��-	/�"oXc�AF*

loss��>DO�b       ��-	l=#oXc�AG*

lossQ�>���       ��-	�#oXc�AH*

loss�c�>�7       ��-	5�$oXc�AI*

loss��>�9y       ��-	�=%oXc�AJ*

lossJ^�>��(�       ��-	��%oXc�AK*

lossi�p>��       ��-	qr&oXc�AL*

loss-�>����       ��-	�'oXc�AM*

lossv�>���       ��-	�'oXc�AN*

loss�/C>��C       ��-	}?(oXc�AO*

lossHyS>�&       ��-	;�(oXc�AP*

loss,��>OteV       ��-	||)oXc�AQ*

loss$��>����       ��-	�*oXc�AR*

loss�;�>!��       ��-	��*oXc�AS*

loss�)E>�X�j       ��-	~Q+oXc�AT*

loss���>���       ��-	��+oXc�AU*

loss���>?�       ��-	A�,oXc�AV*

loss��;>�b�#       ��-	\-oXc�AW*

loss��>I�T~       ��-	��-oXc�AX*

loss��>����       ��-	U.oXc�AY*

lossr1>O1��       ��-	��.oXc�AZ*

lossD�>�S:       ��-	l�/oXc�A[*

loss�r�>"�/        ��-	�=0oXc�A\*

lossa�,>ؗ�J       ��-	L�0oXc�A]*

lossY*�>)D�v       ��-	�|1oXc�A^*

loss���>U���       ��-	�2oXc�A_*

loss��?>H��T       ��-	�2oXc�A`*

loss��>��       ��-	Q3oXc�Aa*

loss�i�>g>&       ��-	��3oXc�Ab*

loss��&?��m       ��-	�4oXc�Ac*

lossmk�>H��       ��-	a5oXc�Ad*

loss���=t&�       ��-	4H6oXc�Ae*

lossU;>:.�       ��-	s7oXc�Af*

loss��>0.�5       ��-	|8oXc�Ag*

loss��	>,=�a       ��-	[�8oXc�Ah*

loss!��=�D�       ��-	8�9oXc�Ai*

loss��>��b�       ��-	Ig:oXc�Aj*

loss~V>7���       ��-	�;oXc�Ak*

loss8��=�(�>       ��-	�M<oXc�Al*

loss��>jI�       ��-	��=oXc�Am*

loss.�q>ǅq�       ��-	2w>oXc�An*

loss��>S�3>       ��-	]?oXc�Ao*

loss�X�>v���       ��-	��?oXc�Ap*

loss�`><ћI       ��-	ob@oXc�Aq*

loss�S�=1Yx�       ��-	�AoXc�Ar*

lossRF�="@�       ��-	ҨAoXc�As*

lossjZ">I��Y       ��-	+PBoXc�At*

loss�]>S:q       ��-	�BoXc�Au*

loss7">�Zz        ��-	��CoXc�Av*

loss^��>Lf�       ��-	,EDoXc�Aw*

loss�>�r�O       ��-	$�DoXc�Ax*

loss��*>*�E8       ��-	��EoXc�Ay*

lossc@>�<�       ��-	�FoXc�Az*

losso�A>o�       ��-	ıFoXc�A{*

loss��M>��^�       ��-	0IGoXc�A|*

loss/b=��͵       ��-	��GoXc�A}*

loss�N�>k�E�       ��-	�}HoXc�A~*

loss`%�>��k�       ��-	nIoXc�A*

loss*��=�n7�       �	�aJoXc�A�*

loss^�> c>�       �	��JoXc�A�*

loss��>�/�       �	�KoXc�A�*

loss�>">�X�       �	@LoXc�A�*

loss�V>��       �	��LoXc�A�*

loss�
>�       �	`wMoXc�A�*

loss�^�=�nNx       �	ZNoXc�A�*

loss�{d>%�       �	^�NoXc�A�*

loss3 �>���       �	
fOoXc�A�*

loss�bi>�:��       �	�PoXc�A�*

loss�&�=hP       �	ʧPoXc�A�*

loss�S�=x���       �	LQoXc�A�*

loss�|=�hO       �	��QoXc�A�*

loss؆�=��}       �	V}RoXc�A�*

loss�tl>���       �	. SoXc�A�*

loss�C>o��=       �	j�SoXc�A�*

loss+y>W~�G       �	AgToXc�A�*

lossjMt><)W       �	�#UoXc�A�*

loss1p�=��@       �	��UoXc�A�*

loss<  >T��T       �	�iVoXc�A�*

loss732=Q03       �	�WoXc�A�*

lossÀ >��/�       �	r�WoXc�A�*

loss �S>cktz       �	`XXoXc�A�*

loss��>�|�Z       �	�XoXc�A�*

loss���>j���       �	z�YoXc�A�*

loss���>d�x�       �	I/ZoXc�A�*

lossQ�5>�&,�       �	X�ZoXc�A�*

loss�:>r`�k       �	�Z[oXc�A�*

loss.�H=��{�       �	��[oXc�A�*

lossb�>[[ش       �	׆\oXc�A�*

loss���>%��       �	 ]oXc�A�*

loss�'}>f�q       �	˹]oXc�A�*

loss�_>���b       �	�T^oXc�A�*

loss_�|>}s8       �	Q�^oXc�A�*

lossW�	>S�2�       �	��_oXc�A�*

loss�oC>Z��}       �	F%`oXc�A�*

loss���=�P�
       �	�?aoXc�A�*

loss~�>���(       �	��aoXc�A�*

loss-�6>a=b       �	.tboXc�A�*

loss6�>TR޶       �	�coXc�A�*

loss�V>6�       �	��coXc�A�*

loss�>�]�       �	F[doXc�A�*

loss��l>#�ս       �	w�doXc�A�*

loss�D>��       �	`�eoXc�A�*

loss�W�=�N��       �	�.foXc�A�*

lossڄ<>�v��       �	��foXc�A�*

lossŸh>מ�       �	�hgoXc�A�*

loss��=��t�       �	�hoXc�A�*

lossdr>� �o       �	ÝhoXc�A�*

loss)N�>�N��       �	�0ioXc�A�*

loss��">ԁ	�       �	%�ioXc�A�*

loss��>��
!       �	U�joXc�A�*

loss�=�nI        �	8gkoXc�A�*

loss�[=:���       �	�loXc�A�*

loss�$e>��e       �	ͮloXc�A�*

lossN�=�Q�       �	�noXc�A�*

loss���=�и       �	�noXc�A�*

lossD�$>��}|       �	�QooXc�A�*

loss(�=���A       �	��ooXc�A�*

loss�gl>����       �	N�poXc�A�*

loss'�>+��       �	y=qoXc�A�*

loss� />Ӝ\�       �	��qoXc�A�*

loss��&>�A       �	WvroXc�A�*

lossz>��w       �	|,soXc�A�*

loss�3>�Z7       �	toXc�A�*

lossi'�=�ń       �	�toXc�A�*

loss�s(>�;ƅ       �	�EuoXc�A�*

loss��H=e�J       �	�voXc�A�*

loss�WZ>.>�       �	��voXc�A�*

loss��{>��\j       �	��xoXc�A�*

loss6F>���e       �	�nyoXc�A�*

loss�U�>x
8       �	WzoXc�A�*

lossz">B�_       �	��zoXc�A�*

loss�U�=�b*�       �	�>{oXc�A�*

loss��=�{��       �	�v|oXc�A�*

loss��+>��c       �	�}oXc�A�*

loss4�<>b�z       �	��}oXc�A�*

loss\�->{2�H       �	V~oXc�A�*

loss�"�=C��j       �	��~oXc�A�*

lossv�>�Fyw       �	��oXc�A�*

loss�3>>@�}�       �	�"�oXc�A�*

lossh!>5"z�       �	��oXc�A�*

loss$J>�=       �	�Q�oXc�A�*

loss�� >ڄw       �	��oXc�A�*

loss��>1M�f       �	w��oXc�A�*

loss}��=��l       �	�!�oXc�A�*

loss�=r�cM       �	o�oXc�A�*

lossJ|N>�P�       �	���oXc�A�*

loss�"0>�V��       �	m��oXc�A�*

loss��d>g�H!       �	�&�oXc�A�*

loss 3> [       �	��oXc�A�*

loss 	�>��-%       �	�Q�oXc�A�*

loss�B�>�V
       �	!!�oXc�A�*

loss�v�=ioޟ       �	��oXc�A�*

lossx��=��4�       �	t��oXc�A�*

loss��i><w�n       �	�M�oXc�A�*

lossfG>u��       �	6�oXc�A�*

loss��#=�5       �	�{�oXc�A�*

lossc��=�.|       �	E�oXc�A�*

losssU(>�,�w       �	H��oXc�A�*

loss��p>��c       �	?;�oXc�A�*

loss��=��O       �	�ӍoXc�A�*

loss��>x��       �	Hp�oXc�A�*

loss�[�=5��       �	x�oXc�A�*

loss�ւ=�d(       �	��oXc�A�*

loss&K|>#���       �	�C�oXc�A�*

lossFf[>����       �	�ܐoXc�A�*

losscօ>5Pĵ       �	�{�oXc�A�*

loss�>�b��       �	��oXc�A�*

loss���=�L/�       �	G��oXc�A�*

loss��>8,W�       �	�>�oXc�A�*

loss��Z>cAm       �	$ՓoXc�A�*

loss�j�=>n�(       �	i�oXc�A�*

loss�F+>��=       �	��oXc�A�*

loss�?*>+��	       �	�L�oXc�A�*

loss��G>!�W�       �	�)�oXc�A�*

loss��>��I�       �	��oXc�A�*

lossL�D>�|z�       �	�f�oXc�A�*

loss�
>c�M       �	���oXc�A�*

loss-��=�G�       �	G��oXc�A�*

loss�6	>bB�       �	J%�oXc�A�*

loss�z�>�\�       �	Ͼ�oXc�A�*

loss�)>��K       �	�b�oXc�A�*

loss��>��:       �	j��oXc�A�*

lossc�>��}=       �	a��oXc�A�*

loss���==�S�       �	�!�oXc�A�*

lossl�*=7�]       �	���oXc�A�*

loss�#�=a��*       �	�S�oXc�A�*

lossHM}=���       �	#��oXc�A�*

loss�t%>�N�       �	ß�oXc�A�*

loss�RZ>6JP�       �	�B�oXc�A�*

loss薾=|s�       �	��oXc�A�*

lossW��=�UD       �	�ڡoXc�A�*

loss\f�=*��       �	pz�oXc�A�*

loss���=��k`       �	P�oXc�A�*

loss,�u>]Z�       �	W=�oXc�A�*

loss\a�=a��_       �	��oXc�A�*

lossv?]>�v��       �	D��oXc�A�*

lossx8�=n`(       �	�*�oXc�A�*

loss�V>�Y�       �	�ԦoXc�A�*

loss�>6Wq�       �	�y�oXc�A�*

loss���>E�G�       �	$�oXc�A�*

loss���=����       �	�4�oXc�A�*

loss!ˉ=��       �	��oXc�A�*

loss�@>Giy       �	W?�oXc�A�*

loss���=��o       �	�ޫoXc�A�*

lossg�=-u�c       �	�}�oXc�A�*

lossʏ>�y�       �	��oXc�A�*

loss�i>=�h       �	J��oXc�A�*

loss�W�=U�       �	{O�oXc�A�*

loss���=U���       �	���oXc�A�*

loss�>���)       �	̛�oXc�A�*

loss\�=�ԓS       �	RC�oXc�A�*

loss��=ynʘ       �	��oXc�A�*

losst\>Q��X       �	_��oXc�A�*

loss�L�>�s1       �	�:�oXc�A�*

loss�>�v}�       �	��oXc�A�*

loss��5>���M       �	���oXc�A�*

loss��%>� :       �	ޑ�oXc�A�*

loss�>B�       �	�'�oXc�A�*

lossl��=��/        �	�ŵoXc�A�*

loss3ߥ=d�3�       �	�o�oXc�A�*

loss���<X_��       �	A�oXc�A�*

loss�=>�C�       �	�G�oXc�A�*

loss��>P��       �	�oXc�A�*

loss�~�<Y�z�       �	��oXc�A�*

lossPW>�K�       �	8�oXc�A�*

loss�c�=�"�       �	ڻoXc�A�*

lossiN>�Xb�       �	r�oXc�A�*

loss6�b>��,       �	�oXc�A�*

loss4D>U���       �	&��oXc�A�*

loss�I_>�
kJ       �	�?�oXc�A�*

loss�>b>?��`       �	�վoXc�A�*

loss���=�vW�       �	�q�oXc�A�*

loss���=�cP       �	�%�oXc�A�*

loss2�=-1�       �	j��oXc�A�*

lossI��=as�"       �	�u�oXc�A�*

lossfS�=(��j       �	��oXc�A�*

loss�@>4�.3       �	��oXc�A�*

losseۅ=�[��       �	�>�oXc�A�*

lossH6*>�2��       �	���oXc�A�*

loss���=���       �	!��oXc�A�*

loss��=σ�       �	'�oXc�A�*

loss���={�E�       �	8��oXc�A�*

loss�0>��j       �	mX�oXc�A�*

loss��->wڊ�       �	=��oXc�A�*

loss�D�=S��       �	���oXc�A�*

loss��=����       �	�4�oXc�A�*

loss.��=��{       �	%��oXc�A�*

loss_�=&Bl       �	=_�oXc�A�*

loss!�=]��       �	���oXc�A�*

loss��*>l>&e       �	Ց�oXc�A�*

loss�_>7Kqq       �	Z)�oXc�A�*

lossz�0=w$Z       �	���oXc�A�*

loss"�>$�/       �	S[�oXc�A�*

lossN�)>��C       �	[��oXc�A�*

loss3�=���z       �	��oXc�A�*

loss x>��w       �	f�oXc�A�*

loss�q�>�8?�       �	D��oXc�A�*

lossqu�=A-?�       �	mU�oXc�A�*

loss	�=��67       �	���oXc�A�*

loss`�I>D�N�       �	ڌ�oXc�A�*

loss�{.>�|_       �	�"�oXc�A�*

loss��=\>       �	��oXc�A�*

loss�H�=��V�       �	�I�oXc�A�*

lossa�=����       �	���oXc�A�*

lossQ%=�B       �	~t�oXc�A�*

loss��K>�FU       �	��oXc�A�*

loss1~�=4�	       �	��oXc�A�*

loss�	>F~��       �	�)�oXc�A�*

loss�5�=���       �	s��oXc�A�*

loss�c�=?I@�       �	�W�oXc�A�*

loss=��       �	<j�oXc�A�*

lossa��=�o�       �	*��oXc�A�*

loss�`=�_�       �	1��oXc�A�*

loss�A>~�}       �	�-�oXc�A�*

loss��=��j       �	*�oXc�A�*

loss���>����       �	���oXc�A�*

lossHM)>X>�P       �	YR�oXc�A�*

loss�X�=��~�       �	&��oXc�A�*

loss��2>��D       �	@��oXc�A�*

lossv[>��D       �	a��oXc�A�*

loss7�L>s~i       �	�L�oXc�A�*

lossc�?=��<�       �	ip�oXc�A�*

loss��@=���z       �	��oXc�A�*

loss$�=�rǜ       �	���oXc�A�*

loss�##>0�z       �	�w�oXc�A�*

loss�l>�i��       �	p�oXc�A�*

loss�Y�<�i;�       �	b��oXc�A�*

loss���=�T��       �	���oXc�A�*

loss��=�&%�       �	8�oXc�A�*

loss��=_l       �	d��oXc�A�*

loss���=���l       �	�e�oXc�A�*

loss���=�(�v       �	�oXc�A�*

loss���=v~P�       �	?��oXc�A�*

loss�=e�l       �	]��oXc�A�*

loss`Z�>ei       �	.�oXc�A�*

loss�N�>��û       �	��oXc�A�*

loss��=�3�!       �	���oXc�A�*

loss�4>�	=       �	�K�oXc�A�*

lossS}u=��$M       �	��oXc�A�*

loss/��=4?��       �	���oXc�A�*

loss��=_�B�       �	K!�oXc�A�*

loss��>�Ɖf       �	й�oXc�A�*

loss�`D>�X0�       �	#N�oXc�A�*

loss���=*&�(       �	���oXc�A�*

loss8��=��h       �	��oXc�A�*

lossa�f>`��       �	�oXc�A�*

lossa49>I�k       �	���oXc�A�*

loss7d�=�b��       �	�N�oXc�A�*

loss��=��       �	)��oXc�A�*

loss��=#ݷ�       �	���oXc�A�*

loss�>��F�       �	�%�oXc�A�*

loss#�>''��       �	���oXc�A�*

loss@�K>��'I       �	�Z�oXc�A�*

loss|�2>_�*9       �	���oXc�A�*

losssD�=b4�       �	���oXc�A�*

lossq��=�=e�       �	*R�oXc�A�*

lossV5>�K��       �	n�oXc�A�*

loss���=[       �	��oXc�A�*

losssJ>�y�       �	z��oXc�A�*

lossZ��=K�Q�       �	\U�oXc�A�*

loss�#r>�dl       �	DO�oXc�A�*

loss7' >v��       �	u=�oXc�A�*

loss��=���       �	�W�oXc�A�*

loss�,>���;       �	��oXc�A�*

loss��=L�%�       �	1B�oXc�A�*

loss	�>�c�       �	���oXc�A�*

loss���>�M�       �	��oXc�A�*

lossˎ=PL       �	���oXc�A�*

loss�B>�㿲       �	�4 pXc�A�*

loss@ �=�*��       �	� pXc�A�*

loss��8>Y�[       �	6xpXc�A�*

loss鉶=�yz0       �	wpXc�A�*

loss'>	�u        �	��pXc�A�*

loss��H>>s@       �	�gpXc�A�*

lossc�=xX!       �	2pXc�A�*

loss$l}=�+P       �	��pXc�A�*

loss�(�=�I�       �	C8pXc�A�*

loss��>V��x       �	�pXc�A�*

loss�c?>5�       �	!spXc�A�*

lossC��=��ܾ       �	pXc�A�*

loss_;~>M@P�       �	�pXc�A�*

lossj{>�G       �	�+	pXc�A�*

losslV>���4       �	)�	pXc�A�*

lossJ�=-�1       �	�n
pXc�A�*

lossKK�>�,��       �	�pXc�A�*

lossBL>��MF       �	�pXc�A�*

loss�OJ> �[(       �	x`pXc�A�*

loss���<��)G       �	��pXc�A�*

loss�{�=�d�       �	��pXc�A�*

loss��(>12�U       �	kEpXc�A�*

loss�U>b�N       �	�pXc�A�*

loss�<�W�       �	mppXc�A�*

lossec�=S��       �	qpXc�A�*

loss�=vEʂ       �	T�pXc�A�*

loss��Y>�eT�       �	1CpXc�A�*

loss���=
�       �	u�pXc�A�*

losszG2>��       �	{pXc�A�*

loss�=�Bם       �	pXc�A�*

loss�t5>.b]       �	��pXc�A�*

lossxQ�=��6�       �	f�pXc�A�*

loss��=*Sя       �	�'pXc�A�*

lossW��=I@8       �	�pXc�A�*

lossY��=B5P�       �	o�pXc�A�*

loss!1�=�ya       �	�>pXc�A�*

losse�=Q�x�       �	r�pXc�A�*

lossv�	>��Ar       �	JpXc�A�*

loss�D4=W~a       �	TpXc�A�*

lossȼ�=WZ��       �	��pXc�A�*

loss�&=�7�L       �	�VpXc�A�*

loss�`=�OS       �	��pXc�A�*

loss�=[�       �	�pXc�A�*

loss�T>��4�       �	�$pXc�A�*

loss��D>|d/       �	%�pXc�A�*

loss�7>�i�       �	�bpXc�A�*

loss��=o�A%       �	�pXc�A�*

loss�<u       �	��pXc�A�*

loss�4>^�k       �	�8pXc�A�*

loss�̼>�Û�       �	��pXc�A�*

loss&�>8D)�       �	�f pXc�A�*

loss=���        �	h!pXc�A�*

loss��[>zp?       �	��!pXc�A�*

loss�nO>ӂ8G       �	5"pXc�A�*

lossԿ�=�\��       �	6�"pXc�A�*

lossc:7=����       �	W`#pXc�A�*

loss�)>*�j       �	��#pXc�A�*

loss7g>�jK       �	Ü$pXc�A�*

loss`��=�lv�       �	*;%pXc�A�*

lossX�<>���       �	��%pXc�A�*

loss���=#��       �	�o&pXc�A�*

loss��=7��       �	�'pXc�A�*

loss�:>�;�t       �	I�'pXc�A�*

loss�#�=I�-       �	w0(pXc�A�*

loss̐i=����       �	��(pXc�A�*

lossF��=i��       �	�)pXc�A�*

loss_g>��xG       �	O=*pXc�A�*

loss��=���U       �	��*pXc�A�*

loss��u=
Ǳ�       �	�+pXc�A�*

loss֓=���       �	�>,pXc�A�*

loss���=6�"H       �	�,pXc�A�*

loss�6>&��<       �	fl-pXc�A�*

loss?t<V�-�       �	O.pXc�A�*

loss�~>�݋�       �	��.pXc�A�*

loss�.�=��A�       �	�5/pXc�A�*

loss�$*>����       �	��/pXc�A�*

lossܙR>p2E       �	Zb0pXc�A�*

lossƩ�=��6       �	W
1pXc�A�*

lossr��=v��Y       �	��1pXc�A�*

loss���<)wP�       �	�V2pXc�A�*

loss���=$X       �	��2pXc�A�*

losskϘ=e��        �	:�3pXc�A�*

loss�_>����       �	�?4pXc�A�*

lossxi�>֯8U       �	��4pXc�A�*

loss���=�l��       �	�v5pXc�A�*

loss�\�=���;       �	6pXc�A�*

loss��=}��       �	7pXc�A�*

lossm	�=��{�       �	��7pXc�A�*

lossE�@=�       �	�w8pXc�A�*

loss�̒=�e�       �	�J9pXc�A�*

lossڧ0>���       �	��9pXc�A�*

loss-�=8��Q       �	 |:pXc�A�*

loss$�*>:Ɋu       �	j;pXc�A�*

loss��>Z�I       �	��;pXc�A�*

loss�5�=>pL       �	NE<pXc�A�*

loss��a=�h�Z       �	�<pXc�A�*

loss1��=���       �	M�=pXc�A�*

loss�ٺ=�i{`       �	7>pXc�A�*

lossE�=1��       �	}�>pXc�A�*

loss�Z�=��vU       �	B?pXc�A�*

loss�qe=�r<�       �	��?pXc�A�*

loss��=?        �	�k@pXc�A�*

loss���=��ֺ       �	� ApXc�A�*

lossR�=6�       �	>�ApXc�A�*

loss�$�<��0
       �	(IBpXc�A�*

loss-�-=&��o       �	WCpXc�A�*

lossL�Y=Z�P�       �	��CpXc�A�*

lossR2>E ww       �	ʉDpXc�A�*

loss�=7H�       �	"EpXc�A�*

loss���=dO�       �	iFpXc�A�*

loss��=�F��       �	9�FpXc�A�*

loss��->�k       �	�YGpXc�A�*

lossP>�\�       �	�HpXc�A�*

loss1ڍ=�-ߢ       �	��HpXc�A�*

lossOz,>*�W�       �	i6IpXc�A�*

loss<%=X�e       �	��IpXc�A�*

loss���=��X        �	��JpXc�A�*

loss��=ȏh�       �	n1KpXc�A�*

loss��=��8�       �	��KpXc�A�*

lossZW >�k��       �	JbLpXc�A�*

loss\>"���       �	� MpXc�A�*

lossl1�=0>D�       �	(�MpXc�A�*

lossE\�=}Ң       �	3NpXc�A�*

lossX�=)�o�       �	��NpXc�A�*

loss?��=|	a       �	F_OpXc�A�*

loss�	�<J�J        �	��OpXc�A�*

loss�iC=�}E       �	��PpXc�A�*

loss�kd=D��       �	>&QpXc�A�*

loss}��=5�߃       �	��QpXc�A�*

lossW�Y=�6}(       �	+NRpXc�A�*

lossF">C�0�       �	e�RpXc�A�*

lossڕ>,.l       �	{SpXc�A�*

lossJ�<D��b       �	fTpXc�A�*

loss�D�<}�ی       �	~�TpXc�A�*

lossjm>[�	v       �	Z�UpXc�A�*

lossd.=g<��       �	TVpXc�A�*

lossZ2=J�       �	��WpXc�A�*

loss�/=w�9       �	?;XpXc�A�*

loss�!R=]���       �	��XpXc�A�*

loss��!<���       �	�hYpXc�A�*

loss���=�X]�       �	ZpXc�A�*

loss�e;=���5       �	Y�ZpXc�A�*

loss�j�=�#       �	�D[pXc�A�*

loss��;;�       �	�\pXc�A�*

loss�C=p�N�       �	��\pXc�A�*

loss ��;c�Z�       �	G9]pXc�A�*

loss�ñ=�7-       �	��]pXc�A�*

losss>���       �	+l^pXc�A�*

loss{�=转~       �	�_pXc�A�*

loss[��;A���       �	��_pXc�A�*

loss�U�=�Q6�       �	�0`pXc�A�*

loss���>���g       �	�	apXc�A�*

loss���<��       �	�apXc�A�*

loss�Ԓ>�E�       �	iTbpXc�A�*

lossqf�=�y`       �	��bpXc�A�*

loss��=>�Jս       �	cpXc�A�*

lossf"�=       �	�!dpXc�A�*

loss�X�=Dq��       �	/epXc�A�*

loss��>�12       �	5fpXc�A�*

loss�ٹ=J�ld       �	 �fpXc�A�*

loss���=��P       �	�YgpXc�A�*

loss���=�q
       �	�gpXc�A�*

loss�>|V��       �	ЙhpXc�A�*

loss HB>Y��       �	40ipXc�A�*

loss4P�>Â;�       �	e�ipXc�A�*

lossj>|�2D       �	�jpXc�A�*

loss*�$>ڷ�P       �	XkpXc�A�*

loss �9>���       �	;�kpXc�A�*

loss�P�=3�&       �	3klpXc�A�*

loss[��=�h�       �	�	mpXc�A�*

loss]��=@=       �	W�mpXc�A�*

loss
']=��#�       �	ܹnpXc�A�*

loss�;�=*��       �	�hopXc�A�*

loss���=�O��       �	 ppXc�A�*

loss�)�=�l>�       �	U�ppXc�A�*

loss�V0=�v��       �	IKqpXc�A�*

lossXK�=-���       �	��qpXc�A�*

loss��<=hZ�       �	��rpXc�A�*

loss�G=y�H       �	�*spXc�A�*

loss:�R=�e�       �	�spXc�A�*

loss<H
>�,�       �	�rtpXc�A�*

lossz�=Qw#       �	LupXc�A�*

lossJ4�=�c�       �	߿upXc�A�*

loss1ͧ='v�i       �	��vpXc�A�*

loss�0=��       �	FawpXc�A�*

loss6�q<Eg?W       �	�expXc�A�*

loss_[�=��$       �	ypXc�A�*

lossЪ=ak       �	~�ypXc�A�*

losss7J=ǳ�       �	�{pXc�A�*

loss_��=�(��       �	�{pXc�A�*

loss"�>�w��       �	��|pXc�A�*

lossհ>2��        �	��}pXc�A�*

loss�O�<+�k       �	>�~pXc�A�*

lossc��=�b�       �	ݙpXc�A�*

lossx5
=?�Q       �	�M�pXc�A�*

lossiJ�=�]2~       �	��pXc�A�*

loss��=��"       �	
�pXc�A�*

loss���=�	+�       �	���pXc�A�*

loss4��=��t.       �	�9�pXc�A�*

loss���=���       �	G8�pXc�A�*

loss^	>�r�       �	��pXc�A�*

loss#`�<�J��       �	E/�pXc�A�*

loss|��=l��C       �	?X�pXc�A�*

loss/��=h�3&       �	�@�pXc�A�*

loss�)�=9S��       �	�ݧpXc�A�*

loss��=`�,�       �	lv�pXc�A�*

loss-�>�Z��       �	w�pXc�A�*

loss3�>u@X       �	���pXc�A�*

lossL�=�C=j       �	y;�pXc�A�*

loss�|=�9:       �	$ҪpXc�A�*

loss��=���!       �	Dj�pXc�A�*

lossȱE>�U�       �	��pXc�A�*

losss�?>8��       �	Ϣ�pXc�A�*

loss�a=�K�       �	�9�pXc�A�*

loss��<_���       �	��pXc�A�*

loss�V
>�Z�       �	�w�pXc�A�*

loss1>wB/�       �	��pXc�A�*

loss�>]���       �	���pXc�A�*

losso">c�p�       �	$C�pXc�A�*

loss?��<�.��       �	@۰pXc�A�*

loss��>����       �	qs�pXc�A�*

loss{��=$\&       �	��pXc�A�*

lossx�3>���       �	��pXc�A�*

lossx!>�ެl       �	�/�pXc�A�*

loss	�>����       �	�ѳpXc�A�*

loss1�=�Y�9       �	�i�pXc�A�*

loss{C>�v[3       �	\�pXc�A�*

loss�=fYP2       �	���pXc�A�*

loss6>�<��0       �	�H�pXc�A�*

loss�&>r���       �	L�pXc�A�*

loss
]�<�M@W       �	���pXc�A�*

loss >`;       �	���pXc�A�*

loss���=����       �	HR�pXc�A�*

loss%��=f��*       �	M�pXc�A�*

lossȞ=�%�       �	R��pXc�A�*

loss��=M޷T       �	�q�pXc�A�*

loss�c6>���m       �	K�pXc�A�*

loss4�=OԦ�       �	�¼pXc�A�*

loss`X>JXY       �	�g�pXc�A�*

loss�ײ<t�.       �	ka�pXc�A�*

loss_\�=1N6�       �	,�pXc�A�*

loss��l>���       �	O��pXc�A�*

loss�p>DP��       �	�X�pXc�A�*

loss��>h��       �	��pXc�A�*

lossI�=4�       �	P��pXc�A�*

loss���<��       �	�d�pXc�A�*

loss.��=�Ws�       �	��pXc�A�*

loss�W%>����       �	��pXc�A�*

loss�w(>L�h�       �	ע�pXc�A�*

loss��>1W       �	�D�pXc�A�*

loss��=��m       �	Q��pXc�A�*

loss���=��h       �	}y�pXc�A�*

loss�<a���       �	��pXc�A�*

loss��<��wi       �	���pXc�A�*

loss���=Ԕ�[       �	�G�pXc�A�*

lossRa�=:��-       �	���pXc�A�*

loss��]>2��p       �	�v�pXc�A�*

loss�g=p�       �	=�pXc�A�*

lossxa�<�:,�       �	���pXc�A�*

lossnΡ<?m�       �	�g�pXc�A�*

lossV^�<Ƒ)�       �	L��pXc�A�*

loss/p>O;L       �	��pXc�A�*

loss�\ > �       �	-w�pXc�A�*

lossC�*>Qc�       �	�	�pXc�A�*

loss�N�=����       �	t��pXc�A�*

loss��	= G.       �	<.�pXc�A�*

loss�ʤ=M���       �	��pXc�A�*

loss�HW=TS�C       �	�e�pXc�A�*

lossJ�?=%8�       �	V��pXc�A�*

loss���=?x�       �	���pXc�A�*

loss�=�=��x�       �	�&�pXc�A�*

loss��>t���       �	��pXc�A�*

loss�->���       �	{I�pXc�A�*

losslg�=}�<       �	���pXc�A�*

loss->b�L       �	��pXc�A�*

losse�=r �       �	�pXc�A�*

loss�u�=�       �	H��pXc�A�*

loss-��=lƧ       �	�>�pXc�A�*

loss���=��à       �	���pXc�A�*

loss  �=�S�       �	�q�pXc�A�*

lossMa>%��M       �	��pXc�A�*

loss��=LC�       �	s��pXc�A�*

lossw$= %       �	���pXc�A�*

loss%�=���	       �	!;�pXc�A�*

loss��S>�;       �	���pXc�A�*

loss*��=B虔       �	���pXc�A�*

lossc�j=L���       �	�O�pXc�A�*

loss��=*s�       �	���pXc�A�*

lossw=��iL       �	h��pXc�A�*

lossr�=m�f       �	ji�pXc�A�*

loss�=m5>�       �	���pXc�A�*

loss#�=C^��       �	��pXc�A�*

loss��=G�h�       �	Q��pXc�A�*

loss6�=Z���       �	�C�pXc�A�*

loss�X�=Aם       �	>��pXc�A�*

loss!�=D��       �	a��pXc�A�*

loss��=���        �	�-�pXc�A�*

loss�0>K܍       �	^��pXc�A�*

lossS�p=�J��       �	_{�pXc�A�*

loss��=O0��       �	��pXc�A�*

loss�X=�1�H       �	��pXc�A�*

loss�Z>���#       �	�]�pXc�A�*

loss�;�=����       �	��pXc�A�*

lossVA0=��.�       �	4�pXc�A�*

lossXa<F�E$       �	.��pXc�A�*

loss�m�=�Ί�       �	�D�pXc�A�*

loss#��=�2�       �	?��pXc�A�*

losse�>�sR�       �	�{�pXc�A�*

lossH#>�|��       �	��pXc�A�*

lossߖL<�w	�       �	#��pXc�A�*

loss��=Ќ       �	���pXc�A�*

loss�B�=']��       �	��pXc�A�*

loss\�=A:I�       �	A��pXc�A�*

losse]�=>l�x       �	�P�pXc�A�*

loss�VN=z%,       �	���pXc�A�*

loss��=x0y�       �	���pXc�A�*

loss��<nH��       �	/�pXc�A�*

loss�z=����       �	���pXc�A�*

loss��C<N:��       �	h[�pXc�A�*

loss�8�=?�J       �	��pXc�A�*

loss]�>���$       �	(��pXc�A�*

loss*��=��N       �	L8�pXc�A�*

loss�O_=W�]�       �	Q��pXc�A�*

loss�ǫ=�2       �	�y�pXc�A�*

loss���=�m��       �	��pXc�A�*

loss3�P={���       �	P��pXc�A�*

lossx��<3���       �	lv�pXc�A�*

loss��<J��&       �	��pXc�A�*

loss֊{=m���       �	0��pXc�A�*

lossS��=+tO�       �	�{�pXc�A�*

loss��=1��       �	S"�pXc�A�*

lossB�>���]       �	K�pXc�A�*

lossl�4=fw�       �	1��pXc�A�*

loss��}=l'��       �	��pXc�A�*

loss;�=|�3       �	���pXc�A�*

loss�_�<��	�       �	�n�pXc�A�*

loss��"=E�w       �	{/�pXc�A�*

loss��	>^9�       �	���pXc�A�*

loss.7m=�q��       �	�� qXc�A�*

losss��=+Y��       �	�qXc�A�*

lossS��<j'��       �	�(qXc�A�*

lossW��<��&=       �	SXqXc�A�*

loss<�;��~M       �	�YqXc�A�*

loss��Q<�P_       �	�qXc�A�*

losseDb<l�N       �	N�qXc�A�*

loss���=&�ht       �	�iqXc�A�*

lossD.�=gx�.       �	�CqXc�A�*

lossތ>|r�b       �	[$qXc�A�*

loss�=�d��       �	��qXc�A�*

loss_�7>G��'       �	f�	qXc�A�*

loss�n0<jE��       �	s�
qXc�A�*

loss�gK=� �       �	;qXc�A�*

loss���<��       �	��qXc�A�*

loss!��<1 4�       �	ڑqXc�A�*

loss��=�~"_       �	(+qXc�A�*

loss��>���g       �	
fqXc�A�*

loss%O�=��       �	rqXc�A�*

lossԾ6=g�Z       �	_�qXc�A�*

lossJy*<��j�       �	n�qXc�A�*

loss���=7_#v       �	!=qXc�A�*

lossX�=�{��       �	��qXc�A�*

lossdi�=v���       �	(�qXc�A�*

loss��n=�9�       �	!tqXc�A�*

loss�V�=�;�d       �	�qXc�A�*

loss�s+=a�8       �	8�qXc�A�*

loss�)�=@��       �	��qXc�A�*

loss�<�%�A       �	�jqXc�A�*

loss*��=�ѳ#       �	V�qXc�A�*

loss1�}=p׳�       �	b,qXc�A�*

loss֙ =��        �	7�qXc�A�*

lossֵa=��c       �	fkqXc�A�*

loss�y�=���.       �	�`qXc�A�*

lossЦ=b�x       �	�qXc�A�*

loss<�=ٖb�       �	W�qXc�A�*

loss���<\�J       �	v�qXc�A�*

loss�z�<@���       �	_�qXc�A�*

loss��>1��       �	5yqXc�A�*

lossx��<u�Vz       �	+PqXc�A�*

loss�  =|"�       �	�F qXc�A�*

loss2�>��x       �	�!qXc�A�*

loss�9�=# 7;       �	��!qXc�A�*

loss�*=w*�       �	��"qXc�A�*

loss��/=��       �	a�#qXc�A�*

lossd�B;-��>       �	7P$qXc�A�*

loss�t�=,��I       �	��$qXc�A�*

loss_�=�Ⱥ�       �	1&&qXc�A�*

lossc=�Y&%       �	�l'qXc�A�*

loss���<AVul       �	&:(qXc�A�*

lossv��=���L       �	��(qXc�A�*

loss��>U��       �	[�)qXc�A�*

loss��=�p��       �	N)+qXc�A�*

loss�	=�׾       �	{�+qXc�A�*

loss�I >�,e       �	#�,qXc�A�*

loss��=A�2�       �	?-qXc�A�*

loss��A=΂��       �	m�-qXc�A�*

losso=��Q       �	i�.qXc�A�*

loss��=��       �	Q3/qXc�A�*

loss�w
<i7n       �	��/qXc�A�*

loss��}=F�?�       �	g|0qXc�A�*

loss��=I5N       �	�1qXc�A�*

loss���=z/#�       �	'�1qXc�A�*

loss16>���H       �	�c2qXc�A�*

lossc[=���       �	�@3qXc�A�*

loss;	�=^���       �	D�3qXc�A�*

loss(A�<�Z�@       �	�y4qXc�A�*

loss8��=���S       �	�5qXc�A�*

loss8�=&�8       �	O�5qXc�A�*

loss�O=[���       �	J6qXc�A�*

loss��=��5�       �	��6qXc�A�*

loss�,>+p83       �	A~7qXc�A�*

loss��=�+��       �	-%8qXc�A�*

loss��=5?�       �	��8qXc�A�*

loss�	>Ű$       �	�>:qXc�A�*

lossa��=C�       �	�v;qXc�A�*

lossCb=��       �	�&<qXc�A�*

loss�a=tY��       �	�=qXc�A�*

losse=�2@       �	�I>qXc�A�*

loss3�?=�:��       �	��>qXc�A�*

loss�/>���       �	�o?qXc�A�*

loss�Rb>��E       �	�@qXc�A�*

loss��2>��B       �	��@qXc�A�*

loss��+>,��<       �	5DAqXc�A�*

loss>Q��k       �	��AqXc�A�*

loss�Z=��q�       �	�kBqXc�A�*

loss��=���7       �	SCqXc�A�*

loss@�=T��W       �	��CqXc�A�*

loss�C!>x#8u       �	Y4DqXc�A�*

loss�4
=/]�w       �	*�DqXc�A�*

lossx�+=�S�       �	�`EqXc�A�*

lossE)6=P7       �	�FqXc�A�*

loss�=}0~       �	}�FqXc�A�*

lossx45=�zL�       �	oEGqXc�A�*

loss]��=���-       �	{�GqXc�A�*

lossA|�<�݄?       �	jkHqXc�A�*

lossL��< �8�       �	C IqXc�A�*

lossϛ�=�*C�       �	��IqXc�A�*

loss�X�=�b6`       �	�,JqXc�A�*

loss F
>���'       �	P�JqXc�A�*

loss���<=@$       �	��KqXc�A�*

lossdO�=K��       �	�%LqXc�A�*

loss!%>�<�       �	w�LqXc�A�*

loss�E=QC��       �	nNMqXc�A�*

lossQҧ<2��       �	��MqXc�A�*

loss��=W��       �	��NqXc�A�*

loss�X�=��!r       �	�mOqXc�A�*

loss7�M=|x�       �	#PqXc�A�*

loss~�=9J       �	ΦPqXc�A�*

loss�'�=<�(       �	�IQqXc�A�*

lossf�=x��       �	�QqXc�A�*

loss�]=�m�       �	<�RqXc�A�*

loss��1=��،       �	�SqXc�A�*

loss��<r��       �	��SqXc�A�*

loss��=���0       �	�cTqXc�A�*

loss�Ή=�0sZ       �	\ UqXc�A�*

loss��>UX�>       �	h�UqXc�A�*

lossZ��=SR       �	�GVqXc�A�*

lossب<z��F       �	�WqXc�A�*

loss$<���       �	y�WqXc�A�*

lossM�X<�}�F       �	�DXqXc�A�*

lossr�<J5       �	�JYqXc�A�*

lossD�=> �       �	'�YqXc�A�*

lossR�=g���       �	sZqXc�A�*

loss}^>Y%       �	h[qXc�A�*

loss4�#=��       �	��[qXc�A�*

loss�%=N��       �	A\qXc�A�*

loss�/�=ܫ�E       �	K�\qXc�A�*

loss��c=����       �	��]qXc�A�*

loss��R=q��       �	�(^qXc�A�*

lossu=�"�       �	��^qXc�A�*

loss�:�= ���       �	p^_qXc�A�*

loss�l�=���       �	V,`qXc�A�*

loss�}
>	�H�       �	~�`qXc�A�*

loss��=k���       �	saqXc�A�*

loss�=I�t       �	�bqXc�A�*

loss&,�=�o�b       �	h�bqXc�A�*

loss�Ʉ=��O�       �	d�cqXc�A�*

loss�<�>`       �	�_dqXc�A�*

lossZ�>ٽ=�       �	��dqXc�A�*

lossf->s�-?       �	P�eqXc�A�*

loss��<�ǃ�       �	� fqXc�A�*

loss[�;=�PM       �	^�fqXc�A�*

loss-H�=��       �	UQgqXc�A�*

loss�IW=ϲ�*       �	c�gqXc�A�*

loss��i<�O��       �	L�hqXc�A�*

loss���<���       �	�iqXc�A�*

lossT�=��}�       �	�#jqXc�A�*

loss}�F=u�!�       �	�jqXc�A�*

loss���=�K��       �	t^kqXc�A�*

loss��=�qv�       �	Y�kqXc�A�*

loss��j=|��r       �	 �lqXc�A�*

loss�K�<j*�       �	(+mqXc�A�*

loss�T=�PL�       �	�nqXc�A�*

lossz�<Ѫ5|       �	�nqXc�A�*

loss��=`6j       �	�EoqXc�A�*

loss�-l=��	k       �	'�oqXc�A�*

loss�x�<kE��       �	ƈpqXc�A�*

loss���<b�       �	�rqqXc�A�*

loss�S�<v��       �	1rqXc�A�*

loss���=y ��       �	��rqXc�A�*

loss=��<�       �	ʋsqXc�A�*

loss��=UFR�       �	� tqXc�A�*

loss���=e���       �	��tqXc�A�*

loss�V�=5��       �	mUuqXc�A�*

loss�)�=sB�q       �	�uqXc�A�*

loss��U<T�Q       �		qwqXc�A�*

loss��L=���$       �	�<xqXc�A�*

loss�/�=cR��       �	��xqXc�A�*

lossL��=^NM�       �	]�yqXc�A�*

loss��s>�Z�       �	�%zqXc�A�*

lossg%�<��Io       �	=�zqXc�A�*

loss]�=n���       �	ap{qXc�A�*

loss/rV=#}Q       �	t|qXc�A�*

loss�X+=�h�       �	[}qXc�A�*

loss��e= ��       �	��}qXc�A�*

loss�t=�!�       �	G~qXc�A�*

loss&��=�Y��       �	?�~qXc�A�*

loss��=��bO       �	�qXc�A�*

loss���<�º�       �	��qXc�A�*

loss;Ej=y���       �	N��qXc�A�*

loss�w�<a ��       �	�W�qXc�A�*

loss�E=L�#       �	��qXc�A�*

lossT��=k�?       �	 ��qXc�A�*

loss�/=���	       �	t$�qXc�A�*

loss�y�=_}��       �	z��qXc�A�*

loss/�>��       �	k}�qXc�A�*

loss�={�vS       �	��qXc�A�*

loss׻1<� �       �	���qXc�A�*

lossa��=��&       �	ޒ�qXc�A�*

loss��	>����       �	�Z�qXc�A�*

lossRR=� �D       �	���qXc�A�*

loss���=�=��       �	���qXc�A�*

lossn�E=8H_       �	/�qXc�A�*

lossc�=7��       �	GˉqXc�A�*

loss�C�<�Um�       �	�`�qXc�A�*

loss1dn= ��       �	���qXc�A�*

loss��=�H|       �	���qXc�A�*

loss��1=.��       �	(�qXc�A�*

loss��>FºX       �	?ƌqXc�A�*

lossvf<=�1��       �	__�qXc�A�*

loss�G�=R�J�       �	O�qXc�A�*

losse�=�)x�       �	R��qXc�A�*

loss��=|q+�       �	'2�qXc�A�*

loss��o;�M       �	��qXc�A�*

loss�S�<�m��       �	���qXc�A�*

loss���<ƹ�5       �	#,�qXc�A�*

loss!fU=�nb+       �	qXc�A�*

lossQ~�=�*RC       �	Gu�qXc�A�*

loss�x|>��       �	�)�qXc�A�*

loss��t=�n��       �	4ؓqXc�A�*

loss�-=�fYj       �	���qXc�A�*

loss���=n#�       �	�)�qXc�A�*

loss�!=Q�       �	LŕqXc�A�*

loss�z�<E���       �	h\�qXc�A�*

lossi}�=v�Q�       �	� �qXc�A�*

loss�=!72b       �	���qXc�A�*

loss�<=�D�       �	隘qXc�A�*

lossd�?=�)       �	:�qXc�A�*

loss��=f��       �	�ЙqXc�A�*

loss��'<٘�       �	h��qXc�A�*

loss��=����       �	T�qXc�A�*

lossA*�=�[,       �	�e�qXc�A�*

loss�>2@C�       �	���qXc�A�*

lossMsY=*�       �	ݙ�qXc�A�*

losss�q=3���       �	�A�qXc�A�*

loss$�<Tp��       �	~�qXc�A�*

loss�@�=�Zt=       �	�w�qXc�A�*

loss��0>(�h       �	��qXc�A�*

loss�S�=��A\       �	�7�qXc�A�*

loss6{l=?� �       �	�͢qXc�A�*

loss�M>[�       �	�d�qXc�A�*

loss�G=s�U       �	��qXc�A�*

loss���<���m       �	ʥ�qXc�A�*

loss��<F��w       �	:y�qXc�A�*

loss{jp=��Qw       �	=�qXc�A�*

loss#^�<�G��       �	��qXc�A�*

loss-	�=D�^i       �	��qXc�A�*

loss�m�<4D�       �	��qXc�A�*

loss�$�=�5�l       �	���qXc�A�*

loss��=��R       �	G�qXc�A�*

loss�x
>���;       �	�۩qXc�A�*

loss�c=p�EH       �	�p�qXc�A�*

lossNA�=��+       �	%�qXc�A�*

loss: �=_��       �	�ݫqXc�A�*

loss-�=vWT       �	s��qXc�A�*

loss�~�=���G       �	K�qXc�A�*

lossN �=�$�u       �	6��qXc�A�*

loss�3=G���       �	�F�qXc�A�*

loss��<���       �	�ٮqXc�A�*

loss�%>��       �	n�qXc�A�*

loss���=֢��       �	6�qXc�A�*

lossM��=��0*       �	�qXc�A�*

lossW�@=�,        �	�Y�qXc�A�*

lossvb=���       �	�qXc�A�*

loss8�9=�5�       �	h��qXc�A�*

loss�`�=۹)�       �	P:�qXc�A�*

lossࢎ<M�L,       �	�سqXc�A�*

loss�=L�M�       �	���qXc�A�*

lossd�O=��        �	aU�qXc�A�*

losslm�=Ҵ�       �	��qXc�A�*

loss��;��â       �	���qXc�A�*

lossȉ�<�iUW       �	�M�qXc�A�*

loss�gT=RSGH       �		�qXc�A�*

loss��=��H[       �	�u�qXc�A�*

loss��<C��       �	.�qXc�A�*

lossr�g=���       �	�qXc�A�*

loss|�=���L       �	R��qXc�A�*

loss���<),bd       �	�1�qXc�A�*

loss��^=�;��       �	 ɻqXc�A�*

loss���<e��R       �	�p�qXc�A�*

loss�W=r(       �	��qXc�A�*

lossC!B=�Y;�       �	g��qXc�A�*

loss}ͧ=O�Z       �	]2�qXc�A�*

loss�6�=�͐�       �	���qXc�A�*

loss���=���       �	L��qXc�A�*

loss2%�=92��       �	� �qXc�A�*

loss�Ց=�Vy       �	z��qXc�A�*

loss��>��       �	*��qXc�A�*

loss�(=b�       �	�$�qXc�A�*

losss��=��'       �	���qXc�A�*

loss��<d���       �	�i�qXc�A�*

loss-\�=�sz       �	���qXc�A�*

lossx�=p؄�       �	��qXc�A�*

loss��T=n��       �	�*�qXc�A�*

loss=��<��$       �	j��qXc�A�*

loss���=0�{       �	;V�qXc�A�*

lossCX�=H�       �	���qXc�A�*

loss���=��n�       �	���qXc�A�*

loss<��=�:��       �	��qXc�A�*

loss��>�z/<       �	��qXc�A�*

lossz9�=�d�       �	J�qXc�A�*

loss���=�x�       �	���qXc�A�*

loss�=Ω��       �	U��qXc�A�*

loss���<�蹞       �	�qXc�A�*

loss��V=r�;<       �	���qXc�A�*

loss&,�<% ��       �	�L�qXc�A�*

loss:ȝ<��j�       �	u��qXc�A�*

lossGV=߲{�       �	σ�qXc�A�*

lossS=�H��       �	I*�qXc�A�*

lossJ1=����       �	!��qXc�A�*

loss��r=7���       �	�j�qXc�A�*

loss:r
=��u       �	O�qXc�A�*

loss.�d<㉃1       �	d��qXc�A�*

lossv+W=�8L�       �	�N�qXc�A�*

lossX�
>��=P       �	n��qXc�A�*

loss���=�B	�       �	���qXc�A�*

lossԿ=���       �	�(�qXc�A�*

loss�j�=7#�       �	���qXc�A�*

loss6+<O�m       �	&U�qXc�A�*

loss�=5L9       �	���qXc�A�*

loss��->�i�       �	/��qXc�A�*

lossJ:�<\�B       �	�*�qXc�A�*

loss�5=�t        �	���qXc�A�*

loss'>�{p       �	�W�qXc�A�*

loss���=oZ       �	���qXc�A�*

loss��g=��J       �	M��qXc�A�*

loss��4=��m       �	��qXc�A�*

loss�ӵ=���        �	��qXc�A�*

loss
�>����       �	,F�qXc�A�*

loss\�j=Z�        �	���qXc�A�*

lossW��=��       �	-w�qXc�A�*

lossM(F=a��F       �	s�qXc�A�*

loss�=��e       �	Y��qXc�A�*

loss�(�=NM�       �	�=�qXc�A�*

loss9=�ޜ�       �	���qXc�A�*

loss=,F<�IU�       �	�k�qXc�A�*

loss`�>=^D�       �	T�qXc�A�*

lossL$>���       �	 ��qXc�A�*

loss�W�=Z�       �	�L�qXc�A�*

loss�7�<�m&       �	��qXc�A�*

loss��=ޖx5       �	_��qXc�A�*

lossT�<ztk       �	p_�qXc�A�*

lossd��=��y
       �	� �qXc�A�*

loss.�<(��       �	��qXc�A�*

loss��=�8^       �	HO�qXc�A�*

loss8k=��O       �	� �qXc�A�*

loss���=7&��       �	Z��qXc�A�*

lossq�C>��Y�       �	&7�qXc�A�*

lossD�a<ë�       �	��qXc�A�*

losst&=����       �	P��qXc�A�*

lossa��=�$_5       �	Yl�qXc�A�*

loss� �=���       �	��qXc�A�*

loss~��=��)H       �	&��qXc�A�*

loss��{>�T|(       �	�W�qXc�A�*

loss5
>s}߰       �	8��qXc�A�*

lossIZ�=���\       �	j��qXc�A�*

loss��+=�#6�       �	�@�qXc�A�*

lossd�>���       �	!��qXc�A�*

lossnh�<x���       �	k��qXc�A�*

loss x�<��g       �	g_�qXc�A�*

loss�[�<��tM       �	���qXc�A�*

lossԭ�=��͕       �	8��qXc�A�*

loss���<�Tp       �	�B�qXc�A�*

loss�խ=v�(       �	m��qXc�A�*

lossj��=��]/       �	1��qXc�A�*

loss�6�=�M��       �	��qXc�A�*

loss�f�<G�L       �	�'�qXc�A�*

lossWG=��x{       �	���qXc�A�*

loss�/�<
]0�       �	qr�qXc�A�*

loss(E�<D}�       �	��qXc�A�*

loss�WR=-�:       �	���qXc�A�*

lossT��<D�`X       �	BA�qXc�A�*

loss_�|=D�[�       �	��qXc�A�*

lossc,�=����       �	�s�qXc�A�*

loss�!>�r��       �	l�qXc�A�*

loss<M�<]<       �	f��qXc�A�*

loss�Db<��|�       �	�M�qXc�A�*

loss_��<�<�Y       �	e�qXc�A�*

loss���=�)�       �	��qXc�A�*

loss,��<x�	�       �	}A�qXc�A�*

lossJ>��       �	���qXc�A�*

lossqP`=���=       �	�t�qXc�A�*

loss3�=
D5       �	|)�qXc�A�*

lossM>�=Ѭ��       �	���qXc�A�*

loss?X�=!��       �	X�qXc�A�*

loss2�_=5�>       �	���qXc�A�*

loss�dn=Xh��       �	߈�qXc�A�*

loss���<����       �	�. rXc�A�*

loss���<1��L       �	(� rXc�A�*

loss]}�=���       �	znrXc�A�*

losst=sƙ�       �	9rXc�A�*

loss��H=ReY�       �	�rXc�A�*

loss�1!<6X��       �	�RrXc�A�*

loss��=e�Z       �	rXc�A�*

lossA�<G�'�       �	n�rXc�A�*

loss�`�<m�W       �	�ErXc�A�*

loss�?;�:�|       �	��rXc�A�*

loss_P=&���       �	�rXc�A�*

loss�=NxK@       �	�.rXc�A�*

loss�D�<�Z+�       �	W�rXc�A�*

loss�S�<QEO�       �	6vrXc�A�*

loss�=�R&0       �	�	rXc�A�*

loss�&$>6+�       �	��	rXc�A�*

lossD�<��(�       �	�F
rXc�A�*

loss���;quI�       �	H�
rXc�A�*

loss�G�=]?�6       �	5yrXc�A�*

loss�)=ƪ:�       �	4rXc�A�*

loss�E�:(�D�       �	f�rXc�A�*

lossjQ<��/       �	�9rXc�A�*

lossd��;K7�        �	V�rXc�A�*

loss<��;R��       �	nrXc�A�*

lossS�==NY       �	�rXc�A�*

lossE�X;����       �	]�rXc�A�*

loss[~�=�"�       �	�vrXc�A�*

loss3v
<h>2       �	�rXc�A�*

loss���:��       �	|�rXc�A�*

loss_®:���q       �	�rXc�A�*

lossQd=��,z       �	{0rXc�A�*

loss�8
>AY       �	 �rXc�A�*

loss�L=��d�       �	i�rXc�A�*

loss- �:u��V       �	�rXc�A�*

loss ��<�#Z�       �	�>rXc�A�*

loss>�>�΀.       �	��rXc�A�*

lossD:I;��C       �	vorXc�A�*

lossk>����       �	SrXc�A�*

loss�U�=�; �       �	��rXc�A�	*

loss(�Y=&�Hj       �	�UrXc�A�	*

loss��0=��x&       �	9�rXc�A�	*

loss�=��S       �	d�rXc�A�	*

loss��=�4M       �	�|rXc�A�	*

loss�l=R��>       �	'rXc�A�	*

lossa�&=��0       �	]�rXc�A�	*

loss�5�=�e�*       �	�xrXc�A�	*

loss�o=F���       �	�rXc�A�	*

loss��=g�       �	n�rXc�A�	*

loss���=i1�       �	�>rXc�A�	*

loss��N<FX�       �	 rXc�A�	*

loss�t=��       �	�3!rXc�A�	*

loss$3�=^�X       �	_�!rXc�A�	*

loss6R4=$:��       �	t�"rXc�A�	*

loss��v=��e       �	�5#rXc�A�	*

loss��}=ޣ��       �	�#rXc�A�	*

loss��=%t�       �	<�$rXc�A�	*

loss�q�<�h;)       �	R+%rXc�A�	*

loss�^�=�$S�       �	��%rXc�A�	*

lossT6\=��.�       �	�d&rXc�A�	*

loss��<�e�       �	(*'rXc�A�	*

loss��=����       �	d�'rXc�A�	*

loss�<�J��       �	�g(rXc�A�	*

loss1�=�B{       �	.)rXc�A�	*

loss3`�<P�j       �	��)rXc�A�	*

loss���=L4j�       �	�H*rXc�A�	*

loss�|n=�;�       �	�+rXc�A�	*

loss�T7=T>��       �	ܞ+rXc�A�	*

loss��<�j�       �	C8,rXc�A�	*

loss��<�W�       �	��,rXc�A�	*

lossL�<>{�O       �	Ym-rXc�A�	*

loss,�>=b��       �	�.rXc�A�	*

lossDM�<�22�       �	��.rXc�A�	*

lossR+�<]x�       �	�0/rXc�A�	*

loss�G=�)�%       �	t�/rXc�A�	*

loss^d�=n�y       �	Yj0rXc�A�	*

loss{N�=��=�       �	�1rXc�A�	*

loss:��<�W��       �	հ1rXc�A�	*

loss��]=�.x       �	P2rXc�A�	*

lossn�<�M��       �	��2rXc�A�	*

loss�
=O�
*       �	��3rXc�A�	*

loss&2=@��       �	�j4rXc�A�	*

lossc_�=mJ�n       �	65rXc�A�	*

loss!Ī=�9(-       �	1�5rXc�A�	*

lossiC+=����       �	�i6rXc�A�	*

lossJS�=rn�       �	-7rXc�A�	*

loss6��<���       �	�7rXc�A�	*

lossw:�=H%�       �	C:8rXc�A�	*

loss�`=,�]G       �	��PrXc�A�	*

loss6ݒ=t�1W       �	:yQrXc�A�	*

loss�>�$�       �	�RrXc�A�	*

loss��>��       �	�RrXc�A�	*

loss�c�<P-�       �	�JSrXc�A�	*

loss��-=�%f�       �	��SrXc�A�	*

lossr�a=�;       �	w�TrXc�A�	*

loss��X=o�       �	�$UrXc�A�	*

loss%�(>Y"�        �	V�UrXc�A�	*

loss�4>98-       �	moVrXc�A�	*

loss�1�<�¦�       �	WrXc�A�	*

loss�w�=�Ɋ�       �	��WrXc�A�	*

loss/��=AC        �	�0XrXc�A�	*

loss�-�=R�_       �	iYrXc�A�	*

loss)$2=[_5       �	��YrXc�A�	*

loss!�l=1"       �	]jZrXc�A�	*

loss|=˪F�       �	-[rXc�A�	*

loss@��<�/#�       �	��[rXc�A�	*

loss�=�Q�       �	�e\rXc�A�	*

loss.L6>]�U�       �	?�\rXc�A�	*

loss��=�)a�       �	K�]rXc�A�	*

lossֶ�=sNgR       �	�Q^rXc�A�	*

loss�+g=C7r       �	��^rXc�A�	*

loss��=u��       �	��_rXc�A�	*

loss"J=iSs1       �	�'`rXc�A�	*

loss!��<�?*�       �	��`rXc�A�	*

loss&Z�<��A>       �	harXc�A�	*

loss��u=��C@       �	CbrXc�A�	*

loss�N|=2��X       �	F�brXc�A�	*

loss�*�=;       �	a2crXc�A�	*

lossԵ�<A''E       �	Q�crXc�A�	*

lossr��<S�i�       �	{drXc�A�	*

lossx�%=��H       �	�erXc�A�	*

lossE�>�6��       �	��erXc�A�	*

loss�,?=��'       �	ZHfrXc�A�	*

loss�8=�2y       �	��frXc�A�	*

loss�;<u��0       �	ׅgrXc�A�	*

loss8�4=���       �	hrXc�A�	*

loss�n;>Kb��       �	��hrXc�A�	*

loss�ep=�*�       �	�\irXc�A�	*

loss���=s4�i       �	rjrXc�A�	*

loss� �=�?�       �	P�jrXc�A�	*

loss� &<O�M�       �	nkrXc�A�	*

loss	�,=[pLb       �	2lrXc�A�	*

loss�_&=���_       �	��lrXc�A�	*

loss8z�=s��       �	R�mrXc�A�	*

lossf�$=ǩ�       �	�(nrXc�A�	*

loss�MN=�9�       �	��nrXc�A�	*

loss���=�ltj       �	�vorXc�A�	*

lossB��<=�R       �	� prXc�A�	*

loss��;�1�!       �	��prXc�A�	*

lossE!	>�d��       �	\qrXc�A�	*

loss̒�<��(�       �	]�qrXc�A�	*

loss�'>�2��       �	�rrXc�A�	*

lossJ>;=�ᔞ       �	�.srXc�A�	*

lossF�<&![       �	'�srXc�A�	*

loss�E�<��`       �	�}trXc�A�	*

loss���;rs��       �	urXc�A�	*

loss��C=^���       �	��urXc�A�	*

loss�H]=�P�       �	�^vrXc�A�	*

loss��>5c�4       �	z�vrXc�A�	*

lossq�=ء��       �	O�wrXc�A�	*

loss���<�-��       �	+hxrXc�A�	*

lossw�D=W/ $       �	�yrXc�A�	*

loss��|=�-$,       �	��yrXc�A�	*

loss��X=�2 �       �	�({rXc�A�	*

loss(�< N       �	��{rXc�A�	*

loss��R=7j^P       �	�a|rXc�A�	*

loss8"�=U٘h       �	2 }rXc�A�	*

lossvdY=m$�       �	Z�}rXc�A�	*

lossm��<�C�C       �	P7~rXc�A�	*

lossH��=����       �	��~rXc�A�	*

loss�/�=Y��/       �	�wrXc�A�	*

lossS��<�;LW       �	U�rXc�A�	*

loss׈>=�Tl       �	\��rXc�A�	*

loss��B<~�S?       �	E�rXc�A�	*

lossh}*<"`�&       �	nہrXc�A�	*

loss:y	=&=�[       �	?q�rXc�A�	*

loss�s-=� �       �	��rXc�A�	*

loss��/=>C��       �	ޮ�rXc�A�
*

loss�jM==L��       �	k�rXc�A�
*

lossq��=�
�       �	��rXc�A�
*

lossV��=2���       �	���rXc�A�
*

loss��<��_�       �	�2�rXc�A�
*

loss@w]<_��i       �	�ʆrXc�A�
*

loss-�=����       �	�b�rXc�A�
*

loss@jI=���z       �	u�rXc�A�
*

lossM=�3"       �	4��rXc�A�
*

loss�u�<n��       �	�C�rXc�A�
*

loss�T�<gݚ       �	QىrXc�A�
*

loss��z=Znm       �	�q�rXc�A�
*

loss��>��       �	��rXc�A�
*

loss��<18q�       �	���rXc�A�
*

loss/=9��j       �	�6�rXc�A�
*

loss�	>��O�       �	�یrXc�A�
*

lossI0�;ҁ�/       �	倍rXc�A�
*

loss�E�=ܲ�       �	D�rXc�A�
*

loss�s<Mʁ       �	氎rXc�A�
*

loss��w>D�       �	�G�rXc�A�
*

lossI�2<���>       �	X�rXc�A�
*

loss/�O=��3       �	/��rXc�A�
*

loss��
<�'!       �	��rXc�A�
*

lossl��=yHO       �	�rXc�A�
*

loss��~=��       �	P��rXc�A�
*

loss���=�J
�       �	�<�rXc�A�
*

loss�_3>ξj�       �	RғrXc�A�
*

lossV%<:$�       �	�n�rXc�A�
*

loss���<:�e       �	��rXc�A�
*

loss���=�s#       �	zŕrXc�A�
*

loss�+�<�-B�       �	�\�rXc�A�
*

lossN�W=��       �	���rXc�A�
*

loss�g=U��7       �	-��rXc�A�
*

loss�c<�No�       �	�2�rXc�A�
*

lossH�.<hi}�       �	�̘rXc�A�
*

loss��;=g�B       �	�m�rXc�A�
*

loss�=���j       �	x
�rXc�A�
*

loss��<����       �	���rXc�A�
*

lossX��=��o       �	�F�rXc�A�
*

lossW�=���       �	�ܛrXc�A�
*

lossƚ=�gI�       �	w��rXc�A�
*

loss�/q=>�       �	 �rXc�A�
*

loss�='q       �	̘�rXc�A�
*

loss���=u(�       �	M.�rXc�A�
*

lossA*>{9�       �	�rXc�A�
*

loss{Ѣ=�/�F       �	�^�rXc�A�
*

lossĶ<=���       �	^��rXc�A�
*

loss�W=z�.�       �	f��rXc�A�
*

loss��V<=��U       �	\9�rXc�A�
*

loss�@F=ը        �	-�rXc�A�
*

loss��=6��       �	]��rXc�A�
*

loss�#=���       �	�1�rXc�A�
*

loss���=�`*�       �	�̤rXc�A�
*

lossND�=�'�e       �	!v�rXc�A�
*

loss��<;:��       �	��rXc�A�
*

loss$\>c:�c       �	���rXc�A�
*

loss�\�;�hR       �	5��rXc�A�
*

loss�L=ೌT       �	UK�rXc�A�
*

loss��y<}
n       �	?�rXc�A�
*

loss��;�^��       �	���rXc�A�
*

loss��*<7���       �	O�rXc�A�
*

loss���;��w�       �	���rXc�A�
*

lossF�2=�#       �	XU�rXc�A�
*

lossX,�=�/       �	[�rXc�A�
*

lossT�(>�]       �	=��rXc�A�
*

loss�[{=���       �	)$�rXc�A�
*

loss��<�l       �	4J�rXc�A�
*

loss&��<���       �	��rXc�A�
*

lossa��;#0x       �	˿�rXc�A�
*

losszw=�ɩ       �	�Z�rXc�A�
*

loss�<x=��       �	���rXc�A�
*

loss\�a=�,��       �	I��rXc�A�
*

lossG��=�|�       �	���rXc�A�
*

lossQ�@=�o�       �	��rXc�A�
*

loss���<n�M       �	�ֳrXc�A�
*

loss�qq=q��       �	�m�rXc�A�
*

loss�8<.��4       �	;�rXc�A�
*

loss��=3�TD       �	غ�rXc�A�
*

loss�=�L�       �	�Y�rXc�A�
*

loss��>>���       �	��rXc�A�
*

loss�ȃ=��`e       �	��rXc�A�
*

loss�3�=�$a       �	cD�rXc�A�
*

lossJ�<�	3       �	+ݸrXc�A�
*

loss��V=����       �	��rXc�A�
*

loss�;��nC       �	lZ�rXc�A�
*

lossI�0=���       �	�)�rXc�A�
*

loss}� =�6�%       �	��rXc�A�
*

loss�F=dR��       �	���rXc�A�
*

loss�"�<�?�w       �	�j�rXc�A�
*

loss�(�<�J5       �	y��rXc�A�
*

lossʷ�<$ "
       �	���rXc�A�
*

loss2��<\��       �	�b�rXc�A�
*

lossw=;F=]�       �	/��rXc�A�
*

lossIY=���       �	�X�rXc�A�
*

loss�%=:���       �	.��rXc�A�
*

loss���<n       �	���rXc�A�
*

loss@%�<ʹ�       �	��rXc�A�
*

loss�c�={��       �	���rXc�A�
*

loss_�
=d�j       �	[]�rXc�A�
*

loss�)�=� ک       �	�!�rXc�A�
*

loss��<U�\�       �	��rXc�A�
*

loss
HZ;�16�       �	8��rXc�A�
*

loss��=ec�i       �	֌�rXc�A�
*

loss��=F\k       �	�T�rXc�A�
*

loss��;���c       �	���rXc�A�
*

lossj�<�W�r       �	���rXc�A�
*

loss!��; ���       �	n��rXc�A�
*

lossߧ�<���       �	���rXc�A�
*

loss,_K<���       �	���rXc�A�
*

lossC�5=}<p�       �	�X�rXc�A�
*

loss��=q�s       �	��rXc�A�
*

loss��4>w<7       �	A�rXc�A�
*

loss��L=s��       �	��rXc�A�
*

loss�m�<	���       �	�U�rXc�A�
*

loss��=Q�!]       �	^��rXc�A�
*

loss��4;�f�)       �	%�rXc�A�
*

loss媝<B��       �	"��rXc�A�
*

loss��=��q+       �	�i�rXc�A�
*

loss�_=�g4�       �	/��rXc�A�
*

loss�v5=QYh       �	Gu�rXc�A�
*

loss�<��$�       �	w��rXc�A�
*

loss�o=����       �	�Y�rXc�A�
*

loss�H�=�`Y       �	Y��rXc�A�
*

lossD��=����       �	���rXc�A�
*

loss
�<]!�C       �	��rXc�A�
*

loss�=;#s�       �	���rXc�A�
*

loss�@�<��7+       �	�d�rXc�A�
*

loss�k=m b       �	��rXc�A�
*

lossf��=�=       �	���rXc�A�*

loss� =��1       �	9G�rXc�A�*

lossj�=Kr�F       �	&S�rXc�A�*

loss0̂<�q?=       �	J��rXc�A�*

loss��;��:       �	i��rXc�A�*

loss�yX<"�       �	�(�rXc�A�*

loss�<�2       �	���rXc�A�*

loss���=�k�       �	-`�rXc�A�*

loss���=y��       �	���rXc�A�*

lossv�>�oܸ       �	͏�rXc�A�*

lossO(*=J4�{       �	�&�rXc�A�*

losssh=���       �	H��rXc�A�*

loss�{=���       �	(`�rXc�A�*

loss
�<��%�       �	���rXc�A�*

loss�̴<���       �	���rXc�A�*

loss�Qa=�!       �	9�rXc�A�*

loss��"=�T       �	���rXc�A�*

loss���;�z��       �	��rXc�A�*

loss�N=�^�       �	*�rXc�A�*

loss�=�=�B��       �	&��rXc�A�*

loss��=�e�D       �	�m�rXc�A�*

loss�=�i<       �	��rXc�A�*

lossC9=�~�T       �	z��rXc�A�*

loss_�4<zC�       �	6>�rXc�A�*

loss:R?=|H0       �	3��rXc�A�*

loss�JQ=XW�       �	u��rXc�A�*

loss�V�<��       �	�1�rXc�A�*

loss�W=�U�       �	e�rXc�A�*

loss��U=��X       �	���rXc�A�*

loss7��<J��       �	�X�rXc�A�*

loss1y�=mn�A       �	���rXc�A�*

lossY_>��f�       �	ٔ�rXc�A�*

loss"�<��       �	���rXc�A�*

loss|��<�$       �	U0�rXc�A�*

loss�=ذp       �	���rXc�A�*

loss	��<t�"�       �	'l�rXc�A�*

lossj<-�E"       �	c	�rXc�A�*

loss�A[=�m^X       �	���rXc�A�*

loss��<�h|�       �	�>�rXc�A�*

loss��<����       �	\��rXc�A�*

loss�1;��7       �	ǂ�rXc�A�*

losss'd=�4��       �	��rXc�A�*

loss�=�	�@       �	��rXc�A�*

lossct�;e�H�       �	�J�rXc�A�*

loss��=h(��       �	�N�rXc�A�*

loss�0�=�       �	QN�rXc�A�*

loss�F�<�;�F       �	�2�rXc�A�*

loss��]<s+-       �	���rXc�A�*

loss�(�;͵{       �	bj�rXc�A�*

lossx<n=â��       �	ˡ�rXc�A�*

loss|��<�%�p       �	&U sXc�A�*

loss��<�r�       �	�
sXc�A�*

loss�[=�kb�       �	�sXc�A�*

loss���=��G-       �	�=sXc�A�*

loss? =����       �	��sXc�A�*

lossq)�=�/��       �	�osXc�A�*

loss
��<+F       �	)sXc�A�*

loss�� =�4�       �	M�sXc�A�*

loss���<�A�       �	lAsXc�A�*

loss(�=3}��       �	��sXc�A�*

loss�a==4A�       �	Q�sXc�A�*

loss�ִ=�C       �	�3sXc�A�*

loss�3�<�1�       �	`�sXc�A�*

lossF�=絖�       �	SvsXc�A�*

lossi�=���       �	/	sXc�A�*

loss	=�m��       �	"�	sXc�A�*

loss��$='*%�       �	�\
sXc�A�*

loss�2E>do�       �	/�
sXc�A�*

lossZV�=Ï[       �	t�sXc�A�*

loss��=�\t       �	PsXc�A�*

lossq_[=�C��       �	��sXc�A�*

loss�	�=�b�S       �	/�sXc�A�*

lossf@/=�=5�       �	)sXc�A�*

loss<=�?�       �	#�sXc�A�*

loss�_�;,�~       �	OvsXc�A�*

lossOr�<��K�       �	{sXc�A�*

loss�P'=�l��       �	�sXc�A�*

lossƍ�=OM       �	�rsXc�A�*

loss�:�=Dg��       �	l
sXc�A�*

lossf�N=>26       �	��sXc�A�*

loss�`�;�{z       �	>sXc�A�*

loss��=�Da�       �	�sXc�A�*

loss�vA<v���       �	��sXc�A�*

loss��`<��]       �	�\sXc�A�*

loss ��=��       �	f�sXc�A�*

lossκ�;��P�       �	��sXc�A�*

loss��;<�^��       �	�1sXc�A�*

lossCs�<���n       �	�sXc�A�*

loss�̈=�I!       �	odsXc�A�*

loss�D<{Y�y       �	sXc�A�*

loss�=c�#�       �	еsXc�A�*

loss҇O=�͋�       �	@PsXc�A�*

loss���=㸵       �	E�sXc�A�*

loss�&)=�5��       �	��sXc�A�*

losst=�:�       �	|(sXc�A�*

loss/��<�̻       �	��sXc�A�*

loss�l�=RAu       �	fksXc�A�*

lossLP�=@ү�       �	�	sXc�A�*

loss �=xRg�       �	]�sXc�A�*

lossl��<�#�       �	�XsXc�A�*

lossz�=��       �	Z�sXc�A�*

loss͘>1��4       �	l� sXc�A�*

loss���=�t�       �	�j!sXc�A�*

loss�d*=_N��       �	�"sXc�A�*

loss��n=�\��       �	��"sXc�A�*

lossq�)=1¿l       �	��#sXc�A�*

loss�?�<�6�       �	�l$sXc�A�*

loss�W<��:       �	t'%sXc�A�*

loss�E>v���       �	 �%sXc�A�*

loss��L<��	�       �	�&sXc�A�*

loss�q�=8
c�       �	�1'sXc�A�*

loss 8}<�%C7       �	�(sXc�A�*

lossm8�<���S       �	ӥ(sXc�A�*

losst'=k3.�       �	�?)sXc�A�*

loss44>,�       �	,*sXc�A�*

loss��!=��       �	�*sXc�A�*

losssD�<��Ø       �	�+sXc�A�*

loss��P<U�[5       �	*,sXc�A�*

loss)Ŀ=���#       �	�,sXc�A�*

loss`�5=y!G       �	By-sXc�A�*

loss=W=���       �	�'.sXc�A�*

loss��x=}��       �	��.sXc�A�*

loss$��=�`o       �	J{/sXc�A�*

loss*+=�H��       �	*0sXc�A�*

loss_9�<�Ï�       �	Z�0sXc�A�*

loss�ܲ<ARn       �	d�1sXc�A�*

loss�<$F�T       �	B\2sXc�A�*

loss�	=v/'�       �	��3sXc�A�*

loss�5<���r       �	�i4sXc�A�*

loss��=f4
!       �	�5sXc�A�*

loss�H�<}��g       �	��5sXc�A�*

lossl)�=$�H       �	z6sXc�A�*

lossOg�;�vaF       �	}$7sXc�A�*

lossD<M�n       �	n�7sXc�A�*

loss��<ݖC       �	�8sXc�A�*

loss��=a�       �	�.9sXc�A�*

loss?��<4�,       �	��9sXc�A�*

lossl!H>���       �	n:sXc�A�*

loss�ޟ=�ˠ9       �	;sXc�A�*

loss�P�;~I��       �	�;sXc�A�*

loss�G=�jL&       �	rQ<sXc�A�*

loss��=O �
       �	��<sXc�A�*

loss&<��7�       �	�=sXc�A�*

losso�=��       �	=)>sXc�A�*

loss$�h=��       �	��>sXc�A�*

loss�g=��!�       �	>\?sXc�A�*

loss[?q=N� �       �	��?sXc�A�*

loss�l�=B��`       �	��@sXc�A�*

lossOt;<�Ծu       �	I+AsXc�A�*

loss Y�<Y�)i       �	��AsXc�A�*

loss6��<�4vE       �	W[BsXc�A�*

loss
�=�Ƿ�       �	jCsXc�A�*

loss���=;I�)       �	SDsXc�A�*

loss$U�<o��2       �	פDsXc�A�*

lossC=1d1�       �	EEsXc�A�*

loss��=H	       �	J�FsXc�A�*

loss�&>i7Yv       �	a�GsXc�A�*

loss.3�=�l}�       �	HsXc�A�*

loss_p�<�8�       �	!IsXc�A�*

loss���=&3       �	�IsXc�A�*

lossI��<��o       �	�NJsXc�A�*

loss@�=ڶ��       �	�KsXc�A�*

loss�=�9�       �	��KsXc�A�*

loss$=�UH�       �	�?LsXc�A�*

loss��=�r]       �	/�LsXc�A�*

losss��<��k       �	|MsXc�A�*

loss��=��[�       �	�NsXc�A�*

loss�k�<���       �	y�NsXc�A�*

loss��>!�a       �	��OsXc�A�*

loss���=�*7�       �		�PsXc�A�*

loss���<���       �	N&QsXc�A�*

lossYG�=��       �	;�QsXc�A�*

lossx�z=OQ�       �	�cRsXc�A�*

loss=_
.�       �	�SsXc�A�*

loss��}=C��       �	��SsXc�A�*

loss�� >d�'�       �	�PTsXc�A�*

loss��\<�$��       �	2�TsXc�A�*

loss���;r       �	ǂUsXc�A�*

lossL1�=!�"/       �	HVsXc�A�*

loss��f=��#�       �	�VsXc�A�*

loss�0=��"       �	�DWsXc�A�*

loss醓=ku$�       �	m�WsXc�A�*

lossŋ�=Xn!:       �	yXsXc�A�*

loss���<���g       �	�YsXc�A�*

loss�=-�       �	��YsXc�A�*

lossz� =�i�       �	�<ZsXc�A�*

lossw�K<F�2�       �	$�ZsXc�A�*

loss@�#=���       �	֏[sXc�A�*

loss	��=5�       �	&\sXc�A�*

loss�K;����       �	��\sXc�A�*

lossH՟;���f       �	�g]sXc�A�*

loss�=�
?       �	�]sXc�A�*

loss�=)�)\       �	ԙ^sXc�A�*

loss�D�<����       �	1`_sXc�A�*

loss�C�<�(ڐ       �	9�`sXc�A�*

lossT==��Z       �	�6asXc�A�*

loss|��; cx       �	��asXc�A�*

lossnb�=��R       �	E�bsXc�A�*

loss�f<ⲩ3       �	'3csXc�A�*

loss�8=|�?       �	��csXc�A�*

lossȝY=�L5@       �	MjdsXc�A�*

loss��#=�+#       �	esXc�A�*

loss��=c��K       �	~�esXc�A�*

lossE۱=�p�S       �	�@fsXc�A�*

loss���=�|�M       �	#�fsXc�A�*

loss�=3$��       �	N|gsXc�A�*

loss۵4=-(�       �	�phsXc�A�*

loss9��=(�#�       �	isXc�A�*

loss��=C	�       �	�isXc�A�*

loss��K;M	T       �	�DjsXc�A�*

lossM|"<���-       �	�jsXc�A�*

loss%Ǐ=�7�       �	t~ksXc�A�*

loss!Cq=��M�       �	�lsXc�A�*

loss<ṁ>       �	gdmsXc�A�*

loss޽�=3��s       �	r�msXc�A�*

loss���;�CD       �	��nsXc�A�*

loss���=�+y       �	�4osXc�A�*

loss�d=�&�q       �	=�osXc�A�*

loss@�k=���W       �	LopsXc�A�*

loss��:=�F	G       �	)qsXc�A�*

loss���=�9l       �	P�qsXc�A�*

loss�7y<c���       �	K�rsXc�A�*

loss#��;�:p       �	A(ssXc�A�*

loss?��<�ڐ(       �	R�ssXc�A�*

loss�;<~       �	�mtsXc�A�*

loss�n�<mm:�       �	�usXc�A�*

loss���<f��%       �	��usXc�A�*

loss<�g=��4`       �	b0vsXc�A�*

loss�=��"�       �	��vsXc�A�*

loss��,=��=�       �	�YwsXc�A�*

loss�F�=�S$F       �	9�wsXc�A�*

loss��<�:�       �	��xsXc�A�*

loss�=�H��       �	�ysXc�A�*

loss���=��(       �	��ysXc�A�*

lossA<�=J�9�       �	�zsXc�A�*

lossm��=�$ߌ       �	�{sXc�A�*

lossI��<o2g�       �	R(|sXc�A�*

loss�<��@�       �	(�|sXc�A�*

loss�a=,>8       �	l�}sXc�A�*

lossi�.=x�N�       �	�&~sXc�A�*

lossc��<n��       �	XsXc�A�*

loss�#=R�"H       �	V�sXc�A�*

lossq>��N       �	�ŀsXc�A�*

lossa��=1 cz       �	m��sXc�A�*

loss�t<��Է       �	�)�sXc�A�*

loss�c�;�1�k       �	2ǂsXc�A�*

loss�)>�i|�       �	h\�sXc�A�*

lossz1�=7V�       �	���sXc�A�*

lossM D=צ�q       �	���sXc�A�*

loss ��<7�o3       �	
1�sXc�A�*

lossC�<>Y�;       �	څsXc�A�*

loss���<����       �	p�sXc�A�*

lossZr�=����       �	�sXc�A�*

loss�-<Nߴ�       �	��sXc�A�*

loss�K<�Yn<       �	�>�sXc�A�*

loss�O;��S       �	fوsXc�A�*

loss�=�=����       �	Tq�sXc�A�*

loss��)=�D��       �	F�sXc�A�*

loss,��<��av       �	���sXc�A�*

loss4
=��       �	�3�sXc�A�*

loss��=o�t       �	�̋sXc�A�*

loss�
=
X�       �	4d�sXc�A�*

lossnZ�:�j�       �	K �sXc�A�*

lossd�?>�#��       �	���sXc�A�*

loss��;=ⓖ�       �	N_�sXc�A�*

lossa:�=�y�       �	��sXc�A�*

loss�P=��(�       �	ͫ�sXc�A�*

loss�F+<��{�       �	�U�sXc�A�*

losse��=F�}       �	b��sXc�A�*

loss���<i���       �	���sXc�A�*

loss��<��Lh       �	*7�sXc�A�*

loss��V<D[:       �	גsXc�A�*

lossB	>,��       �	�o�sXc�A�*

loss�v�=�d��       �	��sXc�A�*

loss�=2�       �	�sXc�A�*

lossi�$<���i       �	�O�sXc�A�*

lossW�=/�o       �	w�sXc�A�*

loss�F�=���       �	>��sXc�A�*

loss�'<�ٸ~       �	&V�sXc�A�*

loss���</��i       �	9�sXc�A�*

loss,s-=�gϫ       �	z��sXc�A�*

losss��<��Z       �	�:�sXc�A�*

loss�I�=+�@;       �	��sXc�A�*

loss���=�Mw       �	ڏ�sXc�A�*

loss�J	>ʍY       �	gE�sXc�A�*

loss���<�'��       �	��sXc�A�*

lossR1f<����       �	@ޜsXc�A�*

lossG.�<b��P       �	W��sXc�A�*

lossA�= �4�       �	�U�sXc�A�*

lossN&�=��x       �	-�sXc�A�*

loss��<�z�x       �	�ԟsXc�A�*

loss_�=˔V|       �	��sXc�A�*

loss���=zZ|�       �	E��sXc�A�*

lossʇ_=$��_       �	J�sXc�A�*

loss�$�<0��       �	���sXc�A�*

loss=��<����       �	g��sXc�A�*

loss@Z�<����       �	�K�sXc�A�*

loss��i==&�=       �		��sXc�A�*

loss`�=�Ս\       �	7��sXc�A�*

lossɭ�=�o�Y       �	YO�sXc�A�*

losshm�<KlAB       �	r��sXc�A�*

lossm:�=�s�@       �	���sXc�A�*

lossB�!=9��$       �	<��sXc�A�*

losso��;R��4       �	{N�sXc�A�*

loss$1=#�B�       �		��sXc�A�*

losslO=�P�       �	ؚ�sXc�A�*

loss�C#=Z�)�       �	�<�sXc�A�*

loss���<w pQ       �	X�sXc�A�*

loss�=��&       �	���sXc�A�*

loss1�=�       �	S �sXc�A�*

loss� �=]��G       �	��sXc�A�*

loss���=ˑו       �	5b�sXc�A�*

loss:P@=b�       �	` �sXc�A�*

loss�j<���       �	��sXc�A�*

loss�.
=�D��       �	%X�sXc�A�*

lossd;r�;�       �	��sXc�A�*

loss�Y�;��,       �	y��sXc�A�*

loss��i=�M�       �	�.�sXc�A�*

loss��G<T�u�       �	�ǲsXc�A�*

loss<�]=�50       �	�]�sXc�A�*

loss�h%=GW҇       �	�8�sXc�A�*

loss�=}�.�       �	�дsXc�A�*

loss�.=Rfj�       �	4h�sXc�A�*

loss?CM<c�       �	w�sXc�A�*

loss��B=��J       �	l��sXc�A�*

lossHP<s��       �	YM�sXc�A�*

loss��
=c`�       �	��sXc�A�*

lossI��;>G@�       �	���sXc�A�*

lossߑ�:;�R       �	�.�sXc�A�*

loss�1^;�Fv�       �	!̹sXc�A�*

loss�<�;>!�       �	i�sXc�A�*

lossl�F:zoJ�       �	G�sXc�A�*

loss�BG<#^\�       �	ü�sXc�A�*

loss.�9�wZ       �	�r�sXc�A�*

loss=��:c�       �	�9�sXc�A�*

loss$�O:�H%       �	�ӽsXc�A�*

loss��<ǲ�]       �	pw�sXc�A�*

loss�B�=�	]       �	|�sXc�A�*

lossW��<:��        �	@��sXc�A�*

lossa}�:\ct�       �	9E�sXc�A�*

loss�%G;e��       �	���sXc�A�*

lossPE�>2W+P       �	b��sXc�A�*

loss��:�u�       �	z�sXc�A�*

loss{9e>C;=n       �	���sXc�A�*

loss ��<�MB       �	1[�sXc�A�*

loss���=���       �	��sXc�A�*

loss�VH=N�       �	���sXc�A�*

loss*{w<Zt��       �	��sXc�A�*

lossWm>��       �	9�sXc�A�*

lossr�=�b4       �	�+�sXc�A�*

loss.ެ=bs,       �	���sXc�A�*

lossf�+=�r:�       �	���sXc�A�*

loss�}
<.�.�       �	KX�sXc�A�*

loss�=��)�       �	���sXc�A�*

loss��	=�ւ       �	ݖ�sXc�A�*

loss�B�=y\{�       �	�`�sXc�A�*

loss�6m=f*N�       �		��sXc�A�*

loss��=���&       �	��sXc�A�*

loss�5U=�@�       �	ZK�sXc�A�*

loss��=���       �	)��sXc�A�*

losst\�=s���       �	܄�sXc�A�*

loss)V<L��       �	X�sXc�A�*

lossad<T ~�       �	o��sXc�A�*

lossla�<w��+       �	S\�sXc�A�*

loss�S�=��W
       �	���sXc�A�*

lossz�<��qI       �	ڎ�sXc�A�*

loss��<�`�5       �	�-�sXc�A�*

loss�D<���v       �	%��sXc�A�*

loss�L=֎{       �	�v�sXc�A�*

loss� =@�:�       �	�sXc�A�*

loss�k�=��I       �	(��sXc�A�*

loss3>���>       �	�c�sXc�A�*

loss6 F;�(�l       �	���sXc�A�*

loss�bC=!�       �	r��sXc�A�*

loss�G<�       �	[�sXc�A�*

loss.��;-�1�       �	�sXc�A�*

loss1#�<���C       �	[��sXc�A�*

lossWk6<���7       �	�0�sXc�A�*

loss�Ͳ<o�Q�       �	y��sXc�A�*

loss]��=ó��       �	�i�sXc�A�*

loss�0N= ��       �	5
�sXc�A�*

loss��=����       �	��sXc�A�*

loss�E�<���       �	�X�sXc�A�*

loss�=�)B�       �	�	�sXc�A�*

loss�u�<<��       �	���sXc�A�*

loss��A=6�O�       �	�G�sXc�A�*

loss<Ů<�Q%       �	l��sXc�A�*

loss�Χ=W�Tx       �	���sXc�A�*

loss�au=6�       �	���sXc�A�*

lossQ5�<f��       �	ff�sXc�A�*

loss�e�<��       �	�
�sXc�A�*

loss�"1<x6ß       �	�P�sXc�A�*

loss*��<a���       �	���sXc�A�*

loss}�1=J�m        �	�.�sXc�A�*

lossK�=���C       �	���sXc�A�*

lossd�y=( &Y       �	y?�sXc�A�*

loss%S"<h��       �	� tXc�A�*

lossx6=1x�Q       �	�� tXc�A�*

loss��==�9N       �	7OtXc�A�*

lossT��<��؊       �	#�tXc�A�*

loss��<a� (       �	.�tXc�A�*

loss4y=d�0B       �	r1tXc�A�*

losswA�=#"c"       �	��tXc�A�*

loss��<w6E�       �	;ntXc�A�*

loss͵<�i�       �	}tXc�A�*

lossR!�<�I��       �	�wtXc�A�*

loss�C=}Mry       �	`utXc�A�*

lossT�)=N��-       �	itXc�A�*

lossh�=�7�       �	��tXc�A�*

loss���:G߻X       �	�_	tXc�A�*

loss��6<&�C-       �	H�	tXc�A�*

loss�1<<�B��       �	Û
tXc�A�*

loss�
�=��       �	�itXc�A�*

loss��=s��       �	tXc�A�*

lossQ��=G�F       �	��tXc�A�*

lossD!9<ʼu�       �	CtXc�A�*

lossͣ_=~E�m       �	�tXc�A�*

loss��;��'�       �	�mtXc�A�*

lossep�<�-       �	tXc�A�*

loss��=�Z�       �	�tXc�A�*

loss�:�;��)       �	�<tXc�A�*

loss�J�<�t       �	|�tXc�A�*

lossR�#=���0       �	W�tXc�A�*

loss�;Q�c       �	��tXc�A�*

loss�K=~�(�       �	'2tXc�A�*

loss=*�<j��4       �	��tXc�A�*

loss�=�a�q       �	 �tXc�A�*

loss.�A<�i��       �	Z)tXc�A�*

loss�k.=̑�       �	��tXc�A�*

lossi"*<�*�       �	@jtXc�A�*

losse=�
�q       �	StXc�A�*

loss;Y�=��z�       �	�tXc�A�*

loss�=:=���       �	HtXc�A�*

loss�!/=��U       �	��tXc�A�*

lossc�<L	��       �	i�tXc�A�*

loss��<���       �	�ttXc�A�*

loss��y;���       �	RtXc�A�*

loss�K=���       �	�tXc�A�*

loss�#=��P�       �	�DtXc�A�*

lossC��<u��d       �	��tXc�A�*

loss�=��       �	WztXc�A�*

loss���<�t�       �	�tXc�A�*

lossr	+<��TF       �	*�tXc�A�*

loss���<��Z       �	\>tXc�A�*

lossH�a=�uG       �	��tXc�A�*

loss�.<X�8       �	�j tXc�A�*

loss_��>�|�;       �	�!tXc�A�*

lossv�"=4��       �	��!tXc�A�*

loss��;g���       �	��"tXc�A�*

lossV�J<7��%       �	�/#tXc�A�*

loss�d�<�q�|       �	.$tXc�A�*

loss�]�<�Yi(       �	̷$tXc�A�*

loss<v�=l\�       �	�N%tXc�A�*

loss�QX>k��       �	o�%tXc�A�*

loss�l�=9(%       �	+�&tXc�A�*

loss�z�<��)�       �	�'tXc�A�*

lossR�D=��y�       �	��'tXc�A�*

loss�7�<��^�       �	[(tXc�A�*

lossr�Q=���.       �	��(tXc�A�*

loss[�=�`�A       �	�)tXc�A�*

lossȩ=��R�       �	-*tXc�A�*

lossɿ�=[�H�       �	V�*tXc�A�*

lossQ;=�L,�       �	9|+tXc�A�*

lossム<��"       �	�,tXc�A�*

loss�$�<� ?+       �	_�,tXc�A�*

loss_i+<3k�n       �	t{-tXc�A�*

loss�'[=d���       �	 .tXc�A�*

loss1pD=��_N       �	�.tXc�A�*

loss�F�=�rMi       �	6w/tXc�A�*

loss\�
=���	       �	�30tXc�A�*

loss��<�D       �	��0tXc�A�*

loss�5=�       �	@�1tXc�A�*

lossM*=	/��       �	�52tXc�A�*

loss#�=q��       �	��2tXc�A�*

loss�N�=(�;R       �	Ƈ3tXc�A�*

loss@\Q=܁LA       �	GT4tXc�A�*

loss8��<@��       �	�b6tXc�A�*

loss�L�<��H�       �	��6tXc�A�*

loss�)+<�x+       �	�7tXc�A�*

lossDx�<�6v       �	g)8tXc�A�*

loss/�<�9       �	I�8tXc�A�*

lossD2�<����       �	�P9tXc�A�*

loss{ <����       �	&�9tXc�A�*

loss�N<�i       �	΍:tXc�A�*

loss�ђ=�,̯       �	ܝ;tXc�A�*

loss(k!< 9�       �	��<tXc�A�*

loss�S<��=�       �	�4=tXc�A�*

loss���<S���       �	G8>tXc�A�*

lossju�<o�:�       �	��>tXc�A�*

loss�c�=�>�K       �	��?tXc�A�*

loss�أ<gX       �	�@tXc�A�*

loss؁�=�hת       �	m�AtXc�A�*

lossIM�;p&�D       �	ĖBtXc�A�*

lossÓ<v��;       �	I0CtXc�A�*

losss�; B�       �	l�CtXc�A�*

loss>j=(��_       �	�dDtXc�A�*

loss4��<(��W       �	/3EtXc�A�*

lossn��<C1�       �	B�EtXc�A�*

lossS-�=��       �	&oFtXc�A�*

loss�ȏ;�(��       �	�GtXc�A�*

lossRw;$���       �	e�GtXc�A�*

loss*�?=ȷL�       �	�WHtXc�A�*

lossZ4#=�eئ       �	^�HtXc�A�*

lossd��=�(�-       �	ÝItXc�A�*

loss�=�F       �	�9JtXc�A�*

loss�T$=NV!�       �	�JtXc�A�*

loss�U<��O�       �	F�KtXc�A�*

loss�&�<WB(~       �	�2LtXc�A�*

lossE.><-��       �	��LtXc�A�*

loss���<,�?       �	�aMtXc�A�*

loss/�=���       �	��MtXc�A�*

loss���=���k       �	*�NtXc�A�*

loss��=C��r       �	�OtXc�A�*

loss���=0:<[       �	��OtXc�A�*

loss�͵=H�	C       �	0LPtXc�A�*

loss��#=s�7       �	a�PtXc�A�*

loss�F�=�P        �	�QtXc�A�*

loss
�<�l@�       �	�RtXc�A�*

loss��}=b���       �	K�RtXc�A�*

lossz��<�.��       �	NEStXc�A�*

loss�F<��Q�       �	Q�StXc�A�*

loss�X=뫴t       �	ytTtXc�A�*

loss�Z=^�!�       �	EUtXc�A�*

loss.v=���       �	��UtXc�A�*

loss���<-G       �	m;VtXc�A�*

lossM��<�zK�       �	��VtXc�A�*

loss�\<�eё       �	<fWtXc�A�*

loss�q>��d�       �	�'XtXc�A�*

loss���<��0�       �	�XtXc�A�*

loss2�(<���d       �	2XYtXc�A�*

lossmwL<K�       �	R�YtXc�A�*

lossc��;��4       �	'�ZtXc�A�*

loss�=1��       �	%[tXc�A�*

loss�1n;dSh�       �	p�[tXc�A�*

loss��=�M`k       �	m\tXc�A�*

loss��=錟�       �	�]tXc�A�*

loss���<�!�       �	Ý]tXc�A�*

lossi�k=t1       �	O>^tXc�A�*

lossJK�<�Z�       �	��^tXc�A�*

lossW��=�B�x       �	7o_tXc�A�*

loss�<c4��       �	[	`tXc�A�*

loss���<��jn       �	z�`tXc�A�*

loss��<Jr�       �	�<atXc�A�*

loss��<<�>�       �	��atXc�A�*

lossX�=�^F�       �	��btXc�A�*

lossu�=ڞ�=       �	�GctXc�A�*

loss�_�=��}       �	�odtXc�A�*

loss���<�ǺW       �	wIetXc�A�*

loss��:9i]�       �	9�etXc�A�*

lossIk�<���P       �	p�ftXc�A�*

loss�+<E�Xs       �	TRgtXc�A�*

lossJc=^�)       �	�gtXc�A�*

loss�0<I�       �	��htXc�A�*

loss���<�|'"       �	b/itXc�A�*

loss�=z=uQ�=       �	��itXc�A�*

lossƝK=ҥ��       �	�]jtXc�A�*

lossՉ;r��@       �	k�jtXc�A�*

loss>3�<1��b       �	��ktXc�A�*

lossSuv<�xb       �	�ltXc�A�*

loss���;��p        �	HmtXc�A�*

losshT�=HeD�       �	�mtXc�A�*

lossT��=�J��       �	��ntXc�A�*

loss��e<�/��       �	K�otXc�A�*

loss�K�=��/       �	�JptXc�A�*

loss�||;�E�       �	��ptXc�A�*

loss��a<a��       �	 �qtXc�A�*

loss.�d=�gg�       �	�<rtXc�A�*

loss7�f;,��       �	%�rtXc�A�*

loss�X�;ܿ&n       �	�stXc�A�*

loss�H|=��f4       �	g)ttXc�A�*

loss�B�=�V[�       �	��ttXc�A�*

loss?w<H��       �	DjutXc�A�*

losse� <h@       �	J	vtXc�A�*

lossB"<�ѯ�       �	�vtXc�A�*

loss��6=9��R       �	��wtXc�A�*

loss�ZY=��        �		4xtXc�A�*

loss(j�;�4�7       �	��xtXc�A�*

loss;'�;���1       �	�tytXc�A�*

loss%X�;c�E       �	 ztXc�A�*

loss��=����       �	��ztXc�A�*

lossc<�I�I       �	�I{tXc�A�*

loss�N|=ȿ�v       �	LT|tXc�A�*

loss��=ݫr       �	�(}tXc�A�*

lossrtP=D��       �	��}tXc�A�*

loss�V =�       �	'h~tXc�A�*

lossLУ;�sz&       �	Y�~tXc�A�*

lossdA�<���4       �	��tXc�A�*

loss�8j; �R�       �	31�tXc�A�*

loss���<���       �	�̀tXc�A�*

loss�=��/       �	hv�tXc�A�*

loss���=�x/       �	�"�tXc�A�*

lossF3�=h_�       �	D��tXc�A�*

loss�<��[       �	�]�tXc�A�*

loss�+=P��       �	T��tXc�A�*

loss��F<�� �       �	p��tXc�A�*

loss!�==Qt��       �	Y4�tXc�A�*

loss/<.��       �	9҅tXc�A�*

loss�%�<y:�       �	�k�tXc�A�*

loss��;���T       �	!�tXc�A�*

loss��=;j�       �	R��tXc�A�*

lossH,L=P�~       �	31�tXc�A�*

lossz�2<Cx�       �	��tXc�A�*

loss�/=��       �	���tXc�A�*

loss�c'=t$       �	�<�tXc�A�*

loss׬�;��S�       �	�֊tXc�A�*

loss�?�;5���       �	nm�tXc�A�*

loss8k�;8(%�       �	C�tXc�A�*

loss?��<�{X�       �	��tXc�A�*

loss_�8=β>       �	HR�tXc�A�*

loss�)>�c��       �	��tXc�A�*

losscח=����       �	���tXc�A�*

loss�=��        �	��tXc�A�*

loss2�/=j���       �	\��tXc�A�*

loss�y�=�B=L       �	>��tXc�A�*

loss�;<+�       �	)�tXc�A�*

loss��y=���       �	û�tXc�A�*

loss�֬=���c       �	�S�tXc�A�*

loss��<�dX�       �	��tXc�A�*

loss��<���Y       �	G��tXc�A�*

loss�)=uW�       �	 $�tXc�A�*

loss���=QG       �	V��tXc�A�*

loss��?<i��       �	�K�tXc�A�*

loss)�[=����       �	&��tXc�A�*

lossѿ<��tF       �	�v�tXc�A�*

lossR��<�䄴       �	��tXc�A�*

lossV�*=k7��       �	���tXc�A�*

loss�)�<.��       �	�_�tXc�A�*

loss}��<�9�       �	���tXc�A�*

loss�1=N�&0       �	���tXc�A�*

loss��<k3�       �	�"�tXc�A�*

loss��~=�E
       �	0��tXc�A�*

loss�Z�<�aq9       �	�L�tXc�A�*

lossl�<>�h}       �	��tXc�A�*

lossW��<r�#S       �	���tXc�A�*

lossMͫ<���       �	��tXc�A�*

loss}=z���       �	���tXc�A�*

loss$%=��       �	�W�tXc�A�*

loss��<�V��       �	p�tXc�A�*

loss��\<ł��       �	#��tXc�A�*

loss�?<\�*�       �	~�tXc�A�*

loss�E=vf|�       �	h��tXc�A�*

lossW��<0��       �	EF�tXc�A�*

lossR�]=�oj�       �	�ޡtXc�A�*

loss�=��c�       �	���tXc�A�*

loss�+e=�q�       �	m �tXc�A�*

loss1!=����       �	T�tXc�A�*

loss���:E��        �	�}�tXc�A�*

loss�g<�=�       �	�Y�tXc�A�*

loss��;l�o�       �	�tXc�A�*

lossx<� j�       �	덦tXc�A�*

loss�×<���       �	�7�tXc�A�*

lossL�=e2��       �	RҧtXc�A�*

loss��S=�tK@       �	�t�tXc�A�*

loss]�;M���       �	<�tXc�A�*

loss!�(=i��Q       �	竩tXc�A�*

loss���<��)�       �	B�tXc�A�*

lossc�<9�E        �	QتtXc�A�*

loss�
�<e�YA       �	�m�tXc�A�*

loss�O�;TW       �	��tXc�A�*

loss2�<=��       �	�άtXc�A�*

loss��=��Wi       �	�s�tXc�A�*

loss�@r=~K�d       �	L�tXc�A�*

loss�t	=��       �	���tXc�A�*

lossӆ�<�'"       �	�M�tXc�A�*

loss���<c��V       �	��tXc�A�*

loss��;�Y�       �	���tXc�A�*

loss�<���       �	0H�tXc�A�*

loss�m$>Wk�       �	��tXc�A�*

lossђ�=���       �	?��tXc�A�*

loss%�<Ay�       �	�%�tXc�A�*

loss��6:#"�$       �	���tXc�A�*

lossvb~=jJ2       �	V��tXc�A�*

loss�<�=�v�       �	"�tXc�A�*

loss�;a�ٍ       �	��tXc�A�*

loss:vk;���j       �	jM�tXc�A�*

loss'O<��       �	��tXc�A�*

loss��/=�v86       �	���tXc�A�*

lossO�=w#��       �	W>�tXc�A�*

loss�z<@��x       �	"�tXc�A�*

lossa1==�c�       �	�~�tXc�A�*

loss�'=vS��       �	O#�tXc�A�*

loss`/E<����       �	TĺtXc�A�*

lossfw�<,9�        �	4d�tXc�A�*

loss[;ǰ�3       �	e4�tXc�A�*

loss<D=4��       �	dͼtXc�A�*

loss�'=�4�       �	��tXc�A�*

loss=�;u�#�       �	hξtXc�A�*

loss��;�E�       �	#f�tXc�A�*

loss��q=W�       �	vO�tXc�A�*

loss̣�;�V��       �	r�tXc�A�*

lossKԃ=z�1*       �	)��tXc�A�*

lossL�=��d       �	CX�tXc�A�*

lossNb=B�m�       �	/��tXc�A�*

loss���<I�c(       �	e��tXc�A�*

loss̷ =�2��       �	N)�tXc�A�*

lossݲ&<(�Ķ       �	��tXc�A�*

lossspj=���       �	���tXc�A�*

lossE�z=>��       �	�"�tXc�A�*

lossO%>�r�       �	���tXc�A�*

lossi�*<��S       �	TW�tXc�A�*

loss��=-/�I       �	���tXc�A�*

lossҿ�<����       �	ԝ�tXc�A�*

loss{�&=ƻ�        �	{M�tXc�A�*

loss�2<*��       �	��tXc�A�*

loss��*<P͛0       �	�{�tXc�A�*

loss�T�<�胀       �	��tXc�A�*

loss�=�t �       �	���tXc�A�*

lossV��<��8       �	Ӄ�tXc�A�*

loss���=�=�0       �	(*�tXc�A�*

loss�"$=Tw�       �	���tXc�A�*

loss���<uΪ       �	kd�tXc�A�*

lossHV=��_�       �	���tXc�A�*

loss� �<���c       �	���tXc�A�*

loss�_ =�       �	/P�tXc�A�*

loss�q�=���       �	���tXc�A�*

loss�jp=�Ϭ       �	���tXc�A�*

lossł�<�֊]       �	IL�tXc�A�*

lossuߒ<6�I�       �	���tXc�A�*

lossZB =�խ�       �	t�tXc�A�*

loss��+<I���       �		�tXc�A�*

losso%�<�߄�       �	��tXc�A�*

lossi�U=����       �	3p�tXc�A�*

lossᶤ=�[ :       �	W�tXc�A�*

loss��'<�o�       �	��tXc�A�*

lossK)"<ѴN)       �	#0�tXc�A�*

loss��v<!0�k       �	r��tXc�A�*

loss�jk<�8       �	[�tXc�A�*

lossv��=KW��       �	��tXc�A�*

lossL��<(��:       �	/��tXc�A�*

loss]͗=U��3       �	��tXc�A�*

loss�&<����       �	��tXc�A�*

loss�$G=l��x       �	�B�tXc�A�*

loss�7;�#&3       �	.��tXc�A�*

loss�p�;k56!       �	x��tXc�A�*

loss�Ɨ=��Q�       �	�tXc�A�*

loss9��=����       �	���tXc�A�*

loss-�=��
       �	#��tXc�A�*

loss�݁=��_�       �	��tXc�A�*

loss�ӯ=q�:       �	=��tXc�A�*

loss�v=ȢR)       �	�T�tXc�A�*

loss)}{=��L       �	�2�tXc�A�*

lossO�F=�h       �	q��tXc�A�*

loss���<P��       �	�]�tXc�A�*

loss��%;\�nq       �	D��tXc�A�*

loss�<v�n       �	[��tXc�A�*

loss�U�=`AC�       �	�1�tXc�A�*

lossጩ<�5f_       �	���tXc�A�*

loss�ę=.*��       �	ۉ�tXc�A�*

loss��=j���       �	�%�tXc�A�*

loss�~L<̆��       �	��tXc�A�*

lossfY<b�_�       �	\X�tXc�A�*

loss��=3�       �	9��tXc�A�*

loss�0<=:�       �	φ�tXc�A�*

loss=�m�       �	i5�tXc�A�*

loss��Z<���d       �	c��tXc�A�*

loss��<is�       �	6��tXc�A�*

loss�5�=�	�       �	oF�tXc�A�*

loss�=�芨       �	m��tXc�A�*

lossM�=��       �	�|�tXc�A�*

lossi6m=��       �	��tXc�A�*

loss�<#Y��       �	���tXc�A�*

loss��<n�Z�       �	�{�tXc�A�*

lossf��<B޳       �	��tXc�A�*

loss��<����       �	���tXc�A�*

loss~8=�v�r       �	S@�tXc�A�*

loss쀓<��4       �	I��tXc�A�*

loss��9<��W�       �	&p�tXc�A�*

loss�8=�s	       �	��tXc�A�*

lossJ��=�ڛ       �	��tXc�A�*

loss�0�=A%�^       �	�?�tXc�A�*

loss���<���e       �	��tXc�A�*

lossN=���       �	�q�tXc�A�*

loss��\=��I�       �	�tXc�A�*

losso��;P���       �	`��tXc�A�*

loss�/�<�wy�       �	�G�tXc�A�*

loss}�o=��)�       �	���tXc�A�*

loss�=�<�
��       �	l{�tXc�A�*

lossA��<���       �	��tXc�A�*

loss��<G6�       �	=��tXc�A�*

loss`��=��d�       �	�U�tXc�A�*

loss���<u4;       �	��tXc�A�*

loss�G=��F       �	,��tXc�A�*

loss�%)<V��w       �	#/�tXc�A�*

loss��`=���2       �	~��tXc�A�*

loss���< �Q�       �	{j�tXc�A�*

loss�m�<��K       �	81�tXc�A�*

loss�	=�tD5       �	��tXc�A�*

lossz��<U�       �	$��tXc�A�*

loss�3	=�n��       �	ۋ uXc�A�*

loss��;;#(��       �	29uXc�A�*

lossτD=��f�       �	��uXc�A�*

lossC\=�-�       �	�uXc�A�*

lossk�<6D�       �	�uXc�A�*

loss���;��
       �	��uXc�A�*

loss�<M��       �	[�uXc�A�*

loss�Po=,�]�       �	Q.uXc�A�*

lossX�;���r       �	��uXc�A�*

loss�[/=n��       �	�duXc�A�*

loss��/=�L��       �	�uXc�A�*

loss��=Fb�       �	+�uXc�A�*

loss2`�<���       �	:uXc�A�*

loss�8�<U�       �	��uXc�A�*

loss�<Ub (       �	�n	uXc�A�*

loss�Ah=*J�       �	O
uXc�A�*

loss(�=2���       �	��
uXc�A�*

loss;\-<,�ɺ       �	3uXc�A�*

loss�~�<�1��       �	p�uXc�A�*

loss|o�<�с       �	ZfuXc�A�*

loss�6�=~c       �	��uXc�A�*

loss�7�;���       �	�uXc�A�*

loss�;mW�       �	4+uXc�A�*

loss#of=�}�       �	��uXc�A�*

loss&�<=��+=       �	��uXc�A�*

lossCo<3�B�       �	�uXc�A�*

loss��<�ĵ2       �	_�uXc�A�*

loss��;�,��       �	�IuXc�A�*

loss�� >Ei�       �	��uXc�A�*

lossR`�;z���       �	ΌuXc�A�*

lossog�<���       �	H2uXc�A�*

lossr�=� ~�       �	`�uXc�A�*

loss�=��       �	TsuXc�A�*

lossIr�<�#��       �	�uXc�A�*

loss?�I=��^?       �	y�uXc�A�*

loss��V=����       �	¥uXc�A�*

loss��{<�8P�       �	�BuXc�A�*

lossƪe<��ʐ       �	B�uXc�A�*

lossk�;��U       �	ޯuXc�A�*

loss4D[<���v       �	�GuXc�A�*

loss�z_=C��       �	��uXc�A�*

loss�l�<	��       �	g}uXc�A�*

loss���<jR6       �	<uXc�A�*

loss��:���5       �	�uXc�A�*

lossҀ�<&�u       �	bJuXc�A�*

loss��=×�       �	o�uXc�A�*

loss���=3C�[       �	`�uXc�A�*

losse)<=<���       �	X7uXc�A�*

loss��>�-�       �	��uXc�A�*

loss�9�= �       �	e�uXc�A�*

loss��<e�v�       �	B% uXc�A�*

loss���=k�       �	�)!uXc�A�*

loss|��=�os       �	��!uXc�A�*

loss0�=�;�       �	9"uXc�A�*

loss?-�<�ԅ       �	��#uXc�A�*

loss���=�Z%o       �	�5$uXc�A�*

loss#�=܎�       �	u�$uXc�A�*

loss���;��$!       �	Fx%uXc�A�*

loss�X_=(J J       �	b&uXc�A�*

loss��>]3t       �	+�&uXc�A�*

loss��=�	�       �	��'uXc�A�*

loss�$�<pygh       �	1(uXc�A�*

loss���=�<�       �	��)uXc�A�*

loss�E�<
���       �	��*uXc�A�*

loss1��<����       �	3+uXc�A�*

loss�h�=$��       �	l�+uXc�A�*

loss�d=��;a       �	 |,uXc�A�*

lossnգ;!�*       �	�j-uXc�A�*

loss�Sh=��%�       �	p#.uXc�A�*

lossX4_=�#̱       �	��.uXc�A�*

lossC#6<����       �	�e/uXc�A�*

lossIbJ<�~�       �	�0uXc�A�*

loss���<�2m       �	�/1uXc�A�*

loss�N�<��[�       �	��1uXc�A�*

loss*u;TA��       �	Tn2uXc�A�*

loss%�:=�ȉ?       �	Q3uXc�A�*

loss�ld<��t�       �	6�3uXc�A�*

loss�nV=�1s[       �	�K4uXc�A�*

loss8$�=	AL$       �	��4uXc�A�*

loss�5<�h       �	\�5uXc�A�*

loss
��<����       �	)6uXc�A�*

loss���<�	IP       �	�6uXc�A�*

loss,�=aQ�:       �	�d7uXc�A�*

loss[m�;wN=       �	��7uXc�A�*

lossE��=TV;       �	�8uXc�A�*

loss�+->�_{       �	i:9uXc�A�*

lossl0=Uj\N       �	��9uXc�A�*

lossO�<D!b       �	�k:uXc�A�*

loss��"=^N��       �	�;uXc�A�*

loss�*>Z#Ri       �	��;uXc�A�*

loss���<���       �	�i<uXc�A�*

loss�&<x"?�       �	i =uXc�A�*

lossa��=刺�       �	��=uXc�A�*

loss���<S:T       �	!>>uXc�A�*

loss��= ��       �	oF?uXc�A�*

loss)�=�r��       �	��?uXc�A�*

loss�M�<��:N       �	$�@uXc�A�*

loss�.8<�3	B       �	7UAuXc�A�*

lossn9<�u       �	��BuXc�A�*

loss��;%P       �	K�CuXc�A�*

loss,<��n�       �	��DuXc�A�*

loss,�?=�E�
       �	�EuXc�A�*

lossd}�<��       �	XpFuXc�A�*

loss�<�3L�       �	�GuXc�A�*

loss8=C0D�       �		�GuXc�A�*

lossJ0M=�C^�       �	#�HuXc�A�*

loss4R=�`?�       �	�JuXc�A�*

lossB�<�f�g       �	H�JuXc�A�*

loss�A�;a�hi       �	ʋKuXc�A�*

loss��<����       �	ocLuXc�A�*

loss��q<�e�       �	LMuXc�A�*

loss��<xyp�       �	��MuXc�A�*

loss��)<�I:�       �	/kNuXc�A�*

loss�q�<�%�       �	.OuXc�A�*

lossi��<��+�       �	��OuXc�A�*

lossؠ	<�       �	.PuXc�A�*

lossr;�<ȳ�       �	�PuXc�A�*

loss���<a�O�       �	�bQuXc�A�*

loss7�=� ��       �	hRuXc�A�*

lossH��<� +�       �	��RuXc�A�*

loss���<��G"       �	�SSuXc�A�*

loss�0=��E�       �	�SuXc�A�*

loss��P="z�L       �	��TuXc�A�*

lossS��;j�       �	�,UuXc�A�*

lossH=(�Q        �	�VuXc�A�*

loss��< �N�       �	�UWuXc�A�*

loss��}<�M��       �	��WuXc�A�*

loss� �:���       �	O�XuXc�A�*

loss2�%<���       �	�-YuXc�A�*

loss�<�       �	"�YuXc�A�*

loss�;�K�G       �	�[ZuXc�A�*

loss�&A<6�?�       �	��ZuXc�A�*

loss)1�;��v\       �	��[uXc�A�*

lossqs�=%pUC       �	�"\uXc�A�*

loss}ʜ:ɗ�z       �	�\uXc�A�*

lossq׭;Ndk]       �	iU]uXc�A�*

losst^==�knA       �	��]uXc�A�*

loss��T;��B       �	�z^uXc�A�*

loss��:n4~m       �	_uXc�A�*

loss��#:��H�       �	{�_uXc�A�*

loss/��;��J       �	�@`uXc�A�*

lossHR�;�D�       �	�`uXc�A�*

loss6H�;�K       �	%�auXc�A�*

loss҆O:M�P�       �	Z)buXc�A�*

loss<Bj<sۆ�       �	�"cuXc�A�*

loss�]�80��       �	�cuXc�A�*

loss;�{9�N       �	F�duXc�A�*

loss�_::έԩ       �	�MeuXc�A�*

loss3Oi; �       �	'�euXc�A�*

loss.aV=��-       �	��guXc�A�*

loss\h<���       �	��huXc�A�*

loss��9�~S�       �	P�iuXc�A�*

loss�*;���       �	SjuXc�A�*

loss�)7>��܆       �	2�juXc�A�*

loss��<?��A       �	ՑkuXc�A�*

loss��M>[Ʃ2       �	�-luXc�A�*

loss�d=�k�       �	r�luXc�A�*

loss��>��
�       �	�]muXc�A�*

lossL��;��5�       �	+�muXc�A�*

loss��O;B�       �	1�nuXc�A�*

lossњ�=?]�J       �	�,ouXc�A�*

loss���<$T"�       �	��ouXc�A�*

lossS�=c��7       �	mWpuXc�A�*

lossr�<0r�A       �	��puXc�A�*

loss���=;�S       �	m�quXc�A�*

loss���;��n�       �	�/ruXc�A�*

lossŧ�=����       �	q�ruXc�A�*

loss&�r<�Fod       �	�]suXc�A�*

loss/�=�6�O       �	�suXc�A�*

loss�zH=&�       �	�tuXc�A�*

loss��=�ʳh       �	�.uuXc�A�*

loss1��<�Z�       �	�vuXc�A�*

loss�א=\�b�       �	B�vuXc�A�*

lossI�;<&�R�       �	uUwuXc�A�*

loss�$9<.�       �	#�wuXc�A�*

loss��d=A,&       �	��xuXc�A�*

loss�g
<6L�       �	S]yuXc�A�*

lossfj�<1�       �	�yuXc�A�*

lossC]<b��       �	�zuXc�A�*

loss���;Y��g       �	+2{uXc�A�*

loss�=I`؂       �	h�{uXc�A�*

loss���<WP�       �	�i|uXc�A�*

loss��}=>���       �	�}uXc�A�*

loss�.=�1       �	�}uXc�A�*

loss��<��~�       �	�^~uXc�A�*

loss#=Ӈ��       �	ouXc�A�*

loss><D��       �	�uXc�A�*

lossri�;c�$b       �	�L�uXc�A�*

loss�֬<����       �	��uXc�A�*

loss|~<�8��       �	�y�uXc�A�*

loss	��<
�!_       �	��uXc�A�*

lossR�=����       �	���uXc�A�*

lossJ{�=��c�       �	�@�uXc�A�*

lossR#<ί�^       �	�ԃuXc�A�*

lossm�;�Q7       �	�j�uXc�A�*

loss 31<�]��       �	� �uXc�A�*

lossL�<����       �	|��uXc�A�*

loss�?�;����       �	�/�uXc�A�*

lossTW<��S�       �	�uXc�A�*

loss	Q<�ұ       �	�ÇuXc�A�*

loss���=V�QU       �	GZ�uXc�A�*

loss�1�;�O�       �	��uXc�A�*

loss��B=�3[�       �	���uXc�A�*

loss�
�;W�Ҁ       �	l!�uXc�A�*

lossZ,�;N>"�       �	���uXc�A�*

loss�'�<��       �	��uXc�A�*

loss���<h�l       �	��uXc�A�*

loss7G<�Ǆ       �	��uXc�A�*

loss�9�<O׮�       �	}�uXc�A�*

loss��=��>       �	r6�uXc�A�*

loss���<,�
       �	��uXc�A�*

lossZ߈=p���       �	 ǦuXc�A�*

lossJ s<��_�       �	�ͧuXc�A�*

loss.��=ƾ       �	3o�uXc�A�*

loss��,=`�       �	��uXc�A�*

loss�GL;JE�       �	�ةuXc�A�*

lossax)=�xt�       �	vk�uXc�A�*

loss&$�<((	       �	���uXc�A�*

loss-�<j�n�       �	殫uXc�A�*

loss�O<1<��       �	���uXc�A�*

loss�J�=��N�       �	�0�uXc�A�*

loss���:Q���       �	m�uXc�A�*

loss��<����       �	�-�uXc�A�*

loss��<�(!d       �	įuXc�A�*

loss
P�='�<�       �	bf�uXc�A�*

lossڂ�;��3       �	d�uXc�A�*

loss`Ҭ=����       �	��uXc�A�*

lossmz*<��l�       �	�b�uXc�A�*

loss�:�=��S       �	�W�uXc�A�*

loss\+J<F��       �	���uXc�A�*

loss
=����       �	�дuXc�A�*

losst=�<�       �	���uXc�A�*

loss�N;b(�       �	RѶuXc�A�*

lossze�<����       �	���uXc�A�*

loss���<� �       �	am�uXc�A�*

lossx�<h�vy       �	V�uXc�A�*

loss��A;���A       �	���uXc�A�*

loss3��<ʐ�       �	;V�uXc�A�*

loss��q=O��!       �	���uXc�A�*

loss�9=�0|6       �	���uXc�A�*

loss��=��3~       �	�7�uXc�A�*

lossw4�;{9I       �	RѼuXc�A�*

lossdL<Sg�p       �	)u�uXc�A�*

lossfk�=�	0       �	��uXc�A�*

loss�<�Q��       �	-��uXc�A�*

loss�{�<��&�       �	�G�uXc�A�*

loss��n=W�*       �	C�uXc�A�*

loss �+<�Xt       �	g�uXc�A�*

lossH�<�":�       �	 E�uXc�A�*

loss���=G�       �	���uXc�A�*

losshm�=�o1�       �	�y�uXc�A�*

lossn�j;�X�       �	��uXc�A�*

loss�`�<��D       �	j��uXc�A�*

loss�CF<Y!p]       �	�Y�uXc�A�*

loss�ۑ:Fe��       �	��uXc�A�*

loss�ј;��       �	���uXc�A�*

loss��=�|��       �	�D�uXc�A�*

loss�W�;z���       �	m��uXc�A�*

loss@�x>=1'�       �	���uXc�A�*

loss�֪<��{       �	}#�uXc�A�*

lossυ�:��2�       �	��uXc�A�*

lossvD�;x��M       �	�i�uXc�A�*

loss�&r;D$�       �	��uXc�A�*

loss��<E;q�       �	M��uXc�A�*

loss�Ԗ=�Ps       �	6:�uXc�A�*

loss�6>c (       �	��uXc�A�*

losss]R;��A       �	�v�uXc�A�*

lossC-n:AHq       �	7�uXc�A�*

loss��<�(Ϯ       �	Y��uXc�A�*

loss"<��X�       �	Ze�uXc�A�*

loss�@<�R�|       �	y�uXc�A�*

loss-t?=yڝ�       �	ˢ�uXc�A�*

lossl'#=�{�       �	6?�uXc�A�*

loss��~=z��{       �	��uXc�A�*

loss��L=d��       �	]��uXc�A�*

loss�I�<L�H       �	�F�uXc�A�*

loss��=��<\       �	���uXc�A�*

lossS��<%��P       �	Wy�uXc�A�*

loss�ڄ=�H�M       �	�-�uXc�A�*

loss!�V=:�t       �	��uXc�A�*

loss��#<���1       �	%v�uXc�A�*

loss��<�{˸       �	�	�uXc�A�*

lossQ�k<t�&�       �	��uXc�A�*

loss.��<J���       �	2q�uXc�A�*

loss��:<����       �	� �uXc�A�*

loss��<���n       �	:��uXc�A�*

lossW��<�rx       �	�-�uXc�A�*

lossRI=<�%       �	��uXc�A�*

loss��=�ԍ       �	h�uXc�A�*

loss���:d:6       �	���uXc�A�*

loss/� =�^��       �	"��uXc�A�*

loss�N�;L}j       �	�=�uXc�A�*

loss� w<wM��       �	_��uXc�A�*

loss]�=�c7       �	�b�uXc�A�*

loss�;�*�       �	���uXc�A�*

loss6��<�+:�       �	-��uXc�A�*

lossL�X=�*�       �	^��uXc�A�*

loss�K7=R��       �	M��uXc�A�*

loss���<�,L9       �	��uXc�A�*

loss�m�<9B�B       �	u��uXc�A�*

lossq��;�l�       �	Lq�uXc�A�*

loss��=�Ȟ       �	p�uXc�A�*

loss:f�:K`-G       �	t��uXc�A�*

loss+�<p*�o       �	�N�uXc�A�*

loss7��<��N       �	��uXc�A�*

loss�]�<ϋ�=       �	��uXc�A�*

loss�:h�@       �	�=�uXc�A�*

lossF�=Qܿ       �	���uXc�A�*

loss��<j�>G       �	��uXc�A�*

loss�>4<>���       �	��uXc�A�*

loss��=��W�       �	���uXc�A�*

loss�r=s�r       �	R�uXc�A�*

lossѱ;,�l       �	��uXc�A�*

loss*��=p�       �	a��uXc�A�*

loss]�=Acu       �	�9�uXc�A�*

loss�6+=�04       �	c��uXc�A�*

lossnK<�%�       �	ob�uXc�A�*

loss17I=ɉ �       �	�5�uXc�A�*

loss|(�:[U<�       �	���uXc�A�*

loss/Y=���A       �	v�uXc�A�*

loss��;��%       �	L��uXc�A�*

loss%�C=��7e       �	��uXc�A�*

lossʀ=�$ܾ       �	�$�uXc�A�*

loss�R�<)�P       �	���uXc�A�*

loss��<��3T       �	j��uXc�A�*

loss���<�i	H       �	B"�uXc�A�*

loss�Y`<�:��       �	ü�uXc�A�*

loss��=� �       �	�X�uXc�A�*

losshyj=�N��       �	$��uXc�A�*

loss=�u:C�P�       �	���uXc�A�*

loss,L�;�T        �	�!�uXc�A�*

loss�o=�S��       �	��uXc�A�*

loss��n;�0 �       �	Lq�uXc�A�*

loss=��=�Ŀ�       �	��uXc�A�*

loss�@<�lI       �	f��uXc�A�*

loss]�P=��rt       �	EK�uXc�A�*

loss��<��Kw       �	|��uXc�A�*

loss�.�<$
�       �	0��uXc�A�*

lossX5�;z�u�       �	�#�uXc�A�*

loss��=�       �	X�uXc�A�*

loss}�<�oy�       �	,��uXc�A�*

loss��=���?       �	֋�uXc�A�*

losss <ӓ��       �	�%�uXc�A�*

loss�3=�ZT�       �	��uXc�A�*

loss�w<R#�Q       �	x�uXc�A�*

loss]_�;��       �	a vXc�A�*

loss_Q�<���       �	\� vXc�A�*

lossۥ�<�j>       �	~SvXc�A�*

loss�B	=�ò>       �	��vXc�A�*

loss�k =�0�       �	d�vXc�A�*

loss|�.=)�Dg       �	�-vXc�A�*

loss�w"<����       �	��vXc�A�*

loss���:*��R       �	hXvXc�A�*

lossz�r<%���       �	��vXc�A�*

loss�g�;7�       �	��vXc�A�*

loss�R<w�Ш       �	vXc�A�*

loss5�<�,f;       �	�vXc�A�*

loss���;�a�#       �	JEvXc�A�*

loss���=���\       �	g�vXc�A�*

loss �O=�*       �	^�vXc�A�*

lossܯ<��2�       �	�	vXc�A�*

lossn�<:`L�       �	i�	vXc�A�*

loss��=��MZ       �	�C
vXc�A�*

loss?�F=����       �	��
vXc�A�*

lossn��<���       �	�jvXc�A�*

loss\�
=4��c       �	�vXc�A�*

loss���:�/"       �	�vXc�A�*

loss���<�q�       �	 8vXc�A�*

lossF?<����       �	}�vXc�A�*

loss��1=��M       �	��vXc�A�*

loss���<BUu�       �	|&vXc�A�*

loss�� ;n/oo       �	�vXc�A�*

lossf��=¾)       �	XvXc�A�*

loss�f�;3U��       �	V�vXc�A�*

lossc�c=:2Q0       �	׈vXc�A�*

loss�9=3�q�       �	+vXc�A�*

loss��8<y�v�       �	{�vXc�A�*

loss��<?�p       �	bvXc�A�*

loss�=�Tr^       �	��vXc�A�*

loss:ܘ;�w�       �	A�vXc�A�*

loss�	�<Ӕ=       �	�:vXc�A�*

loss!�=����       �	 �vXc�A�*

loss~�<����       �	�svXc�A�*

loss�k�<�OX�       �	0vXc�A�*

loss�1;K��       �	P�vXc�A�*

loss�<O�;       �	�@vXc�A�*

loss�hk=᠏:       �	f�vXc�A�*

loss]*)=�H       �	rovXc�A�*

loss`ck<��)�       �	�2vXc�A�*

loss���:6&-f       �	��vXc�A�*

lossm2�<��$       �	�`vXc�A�*

lossH�=� �       �	'�vXc�A�*

lossn�=���Z       �	ΌvXc�A�*

loss��=B_       �	W$vXc�A�*

loss{ =~$:�       �	�vXc�A�*

loss��$=�1Ե       �	��vXc�A�*

loss��$=�TY       �	�>vXc�A�*

loss��;���%       �	��vXc�A�*

loss�=㬚       �	� vXc�A�*

loss�`k;�:       �	�-!vXc�A�*

loss��<�G(       �	��!vXc�A�*

loss�ئ<D�{�       �	�Y"vXc�A�*

lossC�-<	�x�       �	N�"vXc�A�*

loss���;�U�;       �	�#vXc�A�*

lossT�;'��       �	W&$vXc�A�*

loss�� =��s       �	�$vXc�A�*

loss:R�<��X\       �	�Q%vXc�A�*

lossgE�;��       �	��%vXc�A�*

losso�<]��       �	��&vXc�A�*

loss1��;��F�       �	x('vXc�A�*

loss�~<S��<       �	?�'vXc�A�*

lossx =���       �	3l(vXc�A�*

loss*�<���i       �	)vXc�A�*

loss
,=h��       �	ӣ)vXc�A�*

loss-6�;�NR&       �	�A*vXc�A�*

loss�X�=Tv       �	X�*vXc�A�*

loss��;G�j+       �	Ǆ+vXc�A�*

loss=��<t�m       �	K,vXc�A�*

lossC�<k���       �	�,vXc�A�*

loss���<���       �	*�-vXc�A�*

loss�s@=���       �	�5.vXc�A�*

loss'C>+Y��       �	��.vXc�A�*

loss�Me=0�F       �	ӄ/vXc�A�*

loss\A>�<�N       �	) 0vXc�A�*

loss�A=c�       �	��0vXc�A�*

loss��b=�Q��       �	�1vXc�A�*

loss���;���       �	2"2vXc�A�*

loss:Đ;b��s       �	�2vXc�A�*

loss���;D���       �	�d3vXc�A�*

loss�1�;��%X       �	�4vXc�A�*

loss܏=<���       �	d5vXc�A�*

loss���=/6�	       �	 6vXc�A�*

loss<AF=�U'       �	�6vXc�A�*

loss[=ϼ4B       �	^M7vXc�A�*

loss*�W=[N��       �	E�7vXc�A�*

loss��?<g���       �	�8vXc�A�*

lossc�K<�{\/       �	V,9vXc�A�*

loss�b<ye2!       �	��9vXc�A�*

loss�T�<B�o�       �	�g:vXc�A�*

loss�Z@=�#��       �	J	;vXc�A�*

loss��=���       �	D�;vXc�A�*

loss��=��>|       �	�?<vXc�A�*

loss�i�<Ar�       �	
�<vXc�A�*

loss���<���       �	px=vXc�A�*

loss��<ʗ)�       �	�>vXc�A�*

loss���;�81?       �	;�>vXc�A�*

loss�Z<f��       �	5D?vXc�A�*

loss�'=�cp       �	�)@vXc�A�*

loss�l�<���H       �	��@vXc�A�*

loss�X�<�$�       �	�AvXc�A�*

loss���<g�o       �	�BBvXc�A�*

lossM��:��Ȅ       �	�CvXc�A�*

loss�|�<��О       �	�DvXc�A�*

loss�#�<B��       �	�DvXc�A�*

loss�?�;~.       �	�PEvXc�A�*

loss;�3=[v�       �	��EvXc�A�*

loss��=\�'       �	O�FvXc�A�*

loss��<ͤ�       �	%HvXc�A�*

lossxI�;�sm\       �	ͬHvXc�A�*

loss���<�       �	WIvXc�A�*

loss��W<���       �	;�IvXc�A�*

loss�a;twW       �	�JvXc�A�*

loss���<!,�#       �	�xKvXc�A�*

lossa�<�4^       �	@�LvXc�A�*

loss-��<��I�       �	�fMvXc�A�*

loss.�@=��0       �	�rNvXc�A�*

lossv�=�iWK       �	-!OvXc�A�*

lossl��<��d�       �	��OvXc�A�*

loss��i=��&       �	�<QvXc�A�*

lossᚳ;����       �	�QvXc�A�*

loss�Tg=�~�o       �	R~RvXc�A�*

loss
�
=v��       �	� SvXc�A�*

lossT�=���       �	��SvXc�A�*

loss@F=ČH       �	�`TvXc�A�*

lossL% <f�'       �	�*UvXc�A�*

loss�:�<o�0H       �	��UvXc�A�*

loss��G=���        �	wjVvXc�A�*

loss��@=�	       �	�WvXc�A�*

loss��<CrJ       �	��WvXc�A�*

loss?�=�hڮ       �	�CXvXc�A�*

lossfm=<���q       �	'�XvXc�A�*

loss��;޾!�       �	͐YvXc�A�*

loss��;×�       �	h�[vXc�A�*

lossB=Y�Ѽ       �	��]vXc�A�*

loss�=<�?č       �	D^vXc�A�*

loss}'F;�ނ�       �	��^vXc�A�*

lossZoQ;í`       �	��_vXc�A�*

lossx�h<�:       �	�S`vXc�A�*

loss�̝<�\]�       �	!�avXc�A�*

loss���<��       �	�+bvXc�A�*

loss��f=�	�V       �	��bvXc�A�*

lossmf�<���       �	]cvXc�A�*

lossa٥<˝       �	�dvXc�A�*

loss�W�<�[�       �	G�dvXc�A�*

loss�ʉ:�w�       �	�KevXc�A�*

lossHm;��6�       �	]�evXc�A�*

loss� �<yW       �	��fvXc�A�*

lossd��<� �F       �	k�gvXc�A�*

loss�o
<��x�       �	�UhvXc�A�*

lossUW#=:Y�C       �	Q�hvXc�A�*

loss��=P!�       �	�ivXc�A�*

loss�=?)       �	�1jvXc�A�*

losse`=�n       �	
�jvXc�A�*

loss�s=�;�X       �	_�kvXc�A�*

loss�>�<��       �	�QlvXc�A�*

loss�=����       �	N�lvXc�A�*

loss���;ޤ�a       �	G�mvXc�A�*

loss,6R:�F�       �	�+nvXc�A�*

lossܸ=Sy��       �	n�nvXc�A�*

loss���=�bx       �	�^ovXc�A�*

loss�k�=μ       �	�
pvXc�A�*

loss\C�<pvs�       �	w�pvXc�A�*

loss��6=%$.�       �	!:qvXc�A�*

lossy�<���       �	x�qvXc�A�*

lossj�:�z�       �	 prvXc�A�*

lossa�=�d��       �	�svXc�A�*

loss�@o=.��        �	��svXc�A�*

loss�{=\�.       �	8�tvXc�A�*

lossEUG<���       �	�uvXc�A�*

loss�}=�8z       �	,�uvXc�A�*

lossa�=��v#       �	�qvvXc�A�*

lossI��<4���       �	�wvXc�A�*

loss�Jt=��HN       �	�wvXc�A�*

lossȄ�<jZ�x       �	�`xvXc�A�*

loss�?C<2DXS       �	�yvXc�A�*

lossQ{=�Uo       �	J�yvXc�A�*

loss��=�V�       �	�SzvXc�A�*

loss�oe=��
       �	R�zvXc�A�*

loss��F;��	�       �	�{vXc�A�*

loss���<�yY�       �	�#|vXc�A�*

lossD=��ߙ       �	��|vXc�A�*

loss裶<�*)�       �	h[}vXc�A�*

loss�9=��
       �	%#~vXc�A�*

loss�7K=ЪM       �	��~vXc�A�*

loss�<�=�,'�       �	�zvXc�A�*

loss?�<���       �	)$�vXc�A�*

loss���:�ep�       �	���vXc�A�*

loss��;1 ��       �	9��vXc�A�*

loss��;芧�       �	ZE�vXc�A�*

lossE�X=�^x�       �	���vXc�A�*

loss#�<���       �	6�vXc�A�*

loss�3�<����       �	D��vXc�A�*

loss��<b�2       �	Y��vXc�A�*

loss�H=վ��       �	�&�vXc�A�*

loss�8�;��Ǔ       �	f-�vXc�A�*

loss/Y;q�|�       �	�̇vXc�A�*

loss��^<d,9�       �	\t�vXc�A�*

loss��=��a       �	9�vXc�A�*

lossEc�;�J��       �	���vXc�A�*

loss��*=�%�|       �	)��vXc�A�*

loss\��<I	       �	�b�vXc�A�*

loss��!<� �       �	��vXc�A�*

lossϏ�=%l�-       �	��vXc�A�*

loss?�<#���       �	F`�vXc�A�*

loss\�=s��w       �	���vXc�A�*

loss�E< z�       �	f��vXc�A�*

loss@�;�W�       �	B�vXc�A�*

lossc�=��s�       �	��vXc�A�*

loss�(F=�8       �	��vXc�A�*

loss;�e=�5�       �	:Y�vXc�A�*

lossV�<�p�       �	��vXc�A�*

loss|��;����       �	���vXc�A�*

loss���;q��/       �	�.�vXc�A�*

loss�<�3"       �	�ȓvXc�A�*

loss�<�U�       �	kc�vXc�A�*

lossź=ig�       �	�D�vXc�A�*

loss�k;���D       �	nޕvXc�A�*

loss,�D=��       �	uw�vXc�A�*

loss�=��6       �	� �vXc�A�*

lossN��=4�K       �	-�vXc�A�*

loss,��=����       �	b��vXc�A�*

lossd�N=���R       �	m�vXc�A�*

loss�|�<l��       �	B��vXc�A�*

loss�q�;ܒ�       �	���vXc�A�*

loss��6<po��       �	?�vXc�A�*

loss�<��$�       �	9�vXc�A�*

loss2e�<R>e       �	���vXc�A�*

lossJl�=!�_�       �	2;�vXc�A�*

loss�e�<��O       �	q�vXc�A�*

loss��>=
g��       �	�|�vXc�A�*

lossd��<���{       �	��vXc�A�*

loss{��=�dZ       �	��vXc�A�*

lossQI|=����       �	���vXc�A�*

loss�j�<j&	�       �	o+�vXc�A�*

lossm�1=F 6x       �	t�vXc�A�*

loss@Vb=�w��       �	���vXc�A�*

loss�p=�vV       �	�G�vXc�A�*

losss��=��Ӱ       �	X�vXc�A�*

loss.*�;V�U�       �	ǃ�vXc�A�*

lossmB"<�O��       �	�"�vXc�A�*

loss�a>�h��       �	���vXc�A�*

lossx��<�;*       �	�a�vXc�A�*

loss�<r�       �	� �vXc�A�*

loss�MB<���       �	���vXc�A�*

loss�b�<�^�       �	O=�vXc�A�*

loss3S�<���       �	=רvXc�A�*

lossj�=;D�       �	�w�vXc�A�*

loss_|k<,��       �	v�vXc�A�*

loss�Z�<�2d�       �	pϪvXc�A�*

lossV�=8�<U       �	�y�vXc�A�*

loss��=��       �	��vXc�A�*

loss��9;�[[�       �	:��vXc�A�*

loss��A<p�0�       �	�Y�vXc�A�*

loss�x�;��1�       �	�vXc�A�*

lossx�=t���       �		��vXc�A�*

loss/Y<ϛ�       �	�+�vXc�A�*

loss܎x<�|�       �	�ѯvXc�A�*

lossl5�<&�`       �	h�vXc�A�*

loss�H�=���       �	���vXc�A�*

loss,x�=9�]       �	�vXc�A�*

loss���;ސ��       �	G��vXc�A�*

lossr��<���H       �	�D�vXc�A�*

loss�=��*       �	Q۳vXc�A�*

loss�8�<n���       �	��vXc�A�*

loss1�=�7G�       �	AH�vXc�A�*

lossy#<&�$m       �	�ݵvXc�A�*

loss��=�Q�       �	Wz�vXc�A�*

lossu�=��&       �	��vXc�A�*

loss���=�_��       �	���vXc�A�*

loss.C�=΂pv       �	YL�vXc�A�*

loss�R=P>�3       �	d�vXc�A�*

loss��K=���       �	���vXc�A�*

loss��)<���       �	nR�vXc�A�*

loss��=���       �	$�vXc�A�*

loss���<KM'�       �	ƻvXc�A�*

loss�^G<j��       �	�}�vXc�A�*

loss�ߐ=�"iC       �	l$�vXc�A�*

loss�U<_       �	�˽vXc�A�*

loss���=���       �	�h�vXc�A�*

loss
��=�i�       �	y�vXc�A�*

lossz�.=�s7�       �	���vXc�A�*

loss�x=��C�       �	 �vXc�A�*

loss0��<ö#m       �	��vXc�A�*

lossD��<a��2       �	H��vXc�A�*

lossZ��<j�>       �	���vXc�A�*

lossi��<}U��       �	�+�vXc�A�*

loss�=hXc�       �	���vXc�A�*

loss:�#<H�K�       �	��vXc�A�*

lossT�M=M� �       �	5�vXc�A�*

loss��d<�� h       �	6�vXc�A�*

loss�=%8�       �	�#�vXc�A�*

loss�8�;�{       �	S	�vXc�A�*

loss�g�;k�H�       �	�@�vXc�A�*

loss$��<�Y)�       �	'��vXc�A�*

lossYW�<P$c�       �	���vXc�A�*

loss��J=�kU�       �	ׄ�vXc�A�*

loss���<m%n;       �	8I�vXc�A�*

loss��=V���       �	%<�vXc�A�*

loss&��=:��       �	^��vXc�A�*

lossA��<���       �	�v�vXc�A�*

loss��"=]�>       �	��vXc�A�*

loss�t�=��       �	���vXc�A�*

loss(N�;�&J�       �	�8�vXc�A�*

loss�A�<F{��       �	q �vXc�A�*

loss6s�=�"qb       �	|��vXc�A�*

loss�V�=��P�       �	�I�vXc�A�*

loss䩿<t�NP       �	���vXc�A�*

lossWi<�	�       �	�r�vXc�A�*

losshg=�:`�       �	p
�vXc�A�*

loss��=�+v       �	Q��vXc�A�*

loss��=o��i       �	=�vXc�A�*

loss�H=�/�       �	}��vXc�A�*

lossјV<�B�       �	�a�vXc�A�*

loss�L�<G�'       �	��vXc�A�*

loss��c=�T�G       �	��vXc�A�*

lossx�J=8�fu       �	�(�vXc�A�*

loss��;��N       �	[��vXc�A�*

loss�2U;��-�       �	j�vXc�A�*

lossh~�=8��       �	�vXc�A�*

lossO�	=I..�       �	���vXc�A�*

lossk�<Z�9       �	F@�vXc�A�*

loss�N�<�-\       �	���vXc�A�*

loss)B/<�xL       �	k�vXc�A�*

loss[4<��       �	K�vXc�A�*

lossz��:1/�)       �	B��vXc�A�*

loss�G�= ���       �	"8�vXc�A�*

loss2A�;�ꊈ       �	���vXc�A�*

loss���<u       �	@l�vXc�A�*

loss�E=���       �	/�vXc�A�*

lossc|;�~c       �	&��vXc�A�*

loss��;=�3       �	t}�vXc�A�*

loss�4<*���       �	9&�vXc�A�*

loss�a�<���R       �	&�vXc�A�*

loss2~�<����       �	.��vXc�A�*

loss�~=�j�       �	j��vXc�A�*

loss���<42�       �	�e�vXc�A�*

loss�:A;}��       �	�8�vXc�A�*

loss]��;7O22       �	)��vXc�A�*

loss��=�8l       �	j�vXc�A�*

loss��=����       �	 �vXc�A�*

lossぜ:�A��       �	5��vXc�A�*

loss$h�:J�o�       �	�8�vXc�A�*

loss\?�=v�N�       �	�@�vXc�A�*

loss��$<�a��       �	���vXc�A�*

loss��<���z       �	j�vXc�A�*

lossO��<�i�J       �	uU�vXc�A�*

loss�5a<W��       �	���vXc�A�*

loss�)%<x<��       �	R|�vXc�A�*

loss�:�=t��       �	�Z�vXc�A�*

loss�)�;C2]Y       �	���vXc�A�*

loss��<�Ρ�       �	���vXc�A�*

lossnV?;6TV       �	{3�vXc�A�*

loss���;Ud       �	�v�vXc�A�*

loss4�d<P��       �	�vXc�A�*

loss��W=�Oc=       �	b��vXc�A�*

loss�]{=�A�       �	Y�vXc�A�*

lossvV<-A՟       �	���vXc�A�*

lossf�<Na       �	��vXc�A�*

loss3&<PXۼ       �	�5�vXc�A�*

loss���=Du�k       �	o��vXc�A�*

loss��=Sk��       �	n�vXc�A�*

loss叝<_���       �	'�vXc�A�*

lossz�<C�TU       �	��vXc�A�*

loss�D�=3K��       �	�w�vXc�A�*

loss���=�֎       �	M�vXc�A�*

loss�?�;h��       �	���vXc�A�*

loss�f{<�7
       �	m<�vXc�A�*

lossj��<R	��       �	��vXc�A�*

lossQ��<��Q@       �	}v�vXc�A�*

loss,�"<��       �	}$�vXc�A�*

loss� <6�)�       �	��vXc�A�*

loss3
�=�c�       �	�[�vXc�A�*

lossJ�<_4%�       �	@��vXc�A�*

loss=��r�       �	���vXc�A�*

loss���<�ٳ       �	^M wXc�A�*

losst�];w�ff       �	�� wXc�A�*

loss�n<�*��       �	c�wXc�A�*

loss�\ ;�6�       �	�wXc�A�*

loss��;ǻ/A       �	��wXc�A�*

loss�=��8�       �	�QwXc�A�*

loss��,<�9%�       �	�wXc�A�*

loss�<���       �	ςwXc�A�*

losse��=����       �	�wXc�A�*

lossd�+<�>>       �	��wXc�A�*

loss��;��       �	6XwXc�A�*

loss�2=���       �	��wXc�A�*

loss�#=�Q��       �	|�wXc�A�*

loss�;2�:�       �	�WwXc�A�*

loss�@l9��       �	��wXc�A�*

loss��=� U       �	(�	wXc�A�*

loss�'�<L9v       �	�<
wXc�A�*

loss�u>;� =�       �	��
wXc�A�*

lossoa9=GS�       �	�twXc�A�*

loss���<�<e       �	�`wXc�A�*

lossj�,<�f%�       �	�wXc�A�*

lossc�:�/��       �	��wXc�A�*

loss3-�:X��7       �	�RwXc�A�*

loss��(;L���       �	|�wXc�A�*

loss�7�;��[}       �	��wXc�A�*

loss���<�"       �	�,wXc�A�*

loss�J<�;o       �	��wXc�A�*

loss)F�9o�vu       �	o�wXc�A�*

loss�ž<{��\       �	wXc�A�*

lossf��=��2�       �	l�wXc�A�*

loss���;���       �	�FwXc�A�*

loss7�v>���       �	��wXc�A�*

loss��=����       �	��wXc�A�*

loss��<����       �	7wXc�A�*

loss���<!he_       �	��wXc�A�*

loss��4=b��       �	�nwXc�A�*

loss��M=��/@       �	�wXc�A�*

lossO� >����       �	��wXc�A�*

loss:)�=	��       �	�owXc�A�*

loss+�<��~*       �	�wXc�A�*

lossϚ�;��L       �	��wXc�A�*

loss(Ӯ<HV!�       �		7wXc�A�*

loss��=�EP       �	�wXc�A�*

loss���;�c��       �	�ywXc�A�*

lossu�
=�f�       �	wXc�A�*

lossl�<��Z4       �	X�wXc�A�*

loss�4=�W�       �	�bwXc�A�*

loss�U�<)��       �	�wXc�A�*

loss�i�=��`       �	��wXc�A�*

loss_�W=`^G�       �	9wXc�A�*

loss4K=�g�       �	��!wXc�A�*

loss.O<
��       �	�K"wXc�A�*

loss	�<�=��       �	��"wXc�A�*

loss��<�^�       �	�#wXc�A�*

loss�ы<{9��       �	�T$wXc�A�*

loss�m><_0B       �	�%wXc�A�*

loss�2�<D��       �	�%wXc�A�*

loss�c=�%
       �	�>&wXc�A�*

loss�;i<�:�       �	��&wXc�A�*

lossl=;���       �	��'wXc�A�*

loss�^�<<�Q�       �	�d(wXc�A�*

loss6ԓ=����       �	�/)wXc�A�*

loss��F<�W�Y       �	��)wXc�A�*

loss���;ʥ,�       �	@k*wXc�A�*

losslq�;E       �	
�+wXc�A�*

loss�Z<�6��       �	W,wXc�A�*

lossS��<���V       �	�/-wXc�A�*

loss�#�<A��       �	��-wXc�A�*

lossO21=gt�       �	{i.wXc�A�*

loss�'B=��?�       �		/wXc�A�*

loss��z;��'       �	¤/wXc�A�*

loss@C�<v2�-       �	tA0wXc�A�*

loss�cv<�D[&       �	M�0wXc�A�*

lossJc�<�m�P       �	Cq1wXc�A�*

loss&�A;���X       �	.2wXc�A�*

loss
�<�!wL       �	~3wXc�A�*

lossQ��<���       �	��3wXc�A�*

lossfv�;�U       �	fJ4wXc�A�*

loss���<��n�       �	+�4wXc�A�*

lossƤ�;U��z       �	��5wXc�A�*

loss��,=V�W       �	D6wXc�A�*

loss�x<�S��       �	K�QwXc�A�*

loss]5=����       �	M�RwXc�A�*

lossld=�I+       �	�"SwXc�A�*

loss��r<����       �	��SwXc�A�*

loss(�=���5       �	�PTwXc�A�*

lossݱ�;5�V0       �	l�TwXc�A�*

loss��;8H��       �	l�UwXc�A�*

loss��=�;(�       �	�3VwXc�A�*

loss|�<����       �	B�VwXc�A�*

lossS�U<G{�v       �	?oWwXc�A�*

loss�&g<�-Q�       �	&XwXc�A�*

lossx�;K�a�       �	�XwXc�A�*

loss� =Avs       �	ZdYwXc�A�*

loss��<^b�D       �	oZwXc�A�*

loss�p?;j�       �	̸ZwXc�A�*

loss���;�K��       �	W[wXc�A�*

lossl<�:?�W�       �	��[wXc�A�*

loss?��;��)=       �	��\wXc�A�*

loss���<�#?       �	�A]wXc�A�*

loss|� =1�(�       �	v�]wXc�A�*

lossM�<!�ɽ       �	y^wXc�A�*

loss��>3�*       �	~_wXc�A�*

loss_Q�;�.��       �	�_wXc�A�*

loss!
>=��`m       �	�Y`wXc�A�*

lossat;y准       �	��`wXc�A�*

losss{[;��       �	J�awXc�A�*

lossTV=~�       �	lAbwXc�A�*

loss�Wj=���4       �	��bwXc�A�*

loss-4%=����       �	ÂcwXc�A�*

loss��=����       �	�dwXc�A�*

lossv]�; �mH       �	��dwXc�A�*

loss�(<���       �	�ewXc�A�*

loss�W�;�W�       �	� fwXc�A�*

lossڠ�=�U�P       �	]�fwXc�A�*

loss��<����       �	�\gwXc�A�*

loss�i6=Dͣ�       �	m�gwXc�A�*

loss=��<���       �	�hwXc�A�*

loss�r�<6g�       �	�CiwXc�A�*

loss
?�<?q��       �	D�iwXc�A�*

loss
}�<J��7       �	�ujwXc�A�*

loss�(:<G�B       �	��kwXc�A�*

loss
_�=�7}       �	�4lwXc�A�*

lossH�=;�l/�       �	��lwXc�A�*

lossV_�<�7�b       �	z�mwXc�A�*

losss,�<���       �	�,nwXc�A�*

loss���=;�!5       �	�nwXc�A�*

loss	{T=B?�       �	�|owXc�A�*

lossJ=d��       �	�*pwXc�A�*

loss�C�<���       �	��pwXc�A�*

lossO��:��       �	GrqwXc�A�*

loss%n�;܌�-       �	UrwXc�A�*

losshn`=���(       �	/�rwXc�A�*

loss��;\i�       �	_zswXc�A�*

loss(>�W�       �	�ptwXc�A�*

lossOT2<Q��       �	9uwXc�A�*

loss	6�;�ᚬ       �	�uwXc�A�*

lossA�:e&�       �	��vwXc�A�*

lossld:��       �	�'wwXc�A�*

loss=��;��@g       �	��wwXc�A�*

loss�$<�䗖       �	jxwXc�A�*

loss�ۊ=%���       �	<ywXc�A�*

loss���;(a�`       �	{�ywXc�A�*

lossLx�:�l��       �	=ezwXc�A�*

lossIA=?�z�       �	�"{wXc�A�*

loss�<�;e1V�       �	z�{wXc�A�*

loss�h=���       �	`v|wXc�A�*

loss .6=���       �	�}wXc�A�*

loss\aX=Oҟ�       �	ͫ}wXc�A�*

loss��p= '�$       �	bL~wXc�A�*

loss�_G=�W8�       �	M�~wXc�A�*

lossqI;�@�       �	]�wXc�A�*

loss�/<�{A�       �	wJ�wXc�A�*

loss��;|���       �	��wXc�A�*

loss�=�G��       �	h��wXc�A�*

loss��8;�~�W       �	`u�wXc�A�*

lossyP;�TR       �	��wXc�A�*

losslsE<�5"�       �	���wXc�A�*

loss�W�<�ѓ       �	J҄wXc�A�*

loss�n<�{��       �	�wXc�A�*

loss��<<��M3       �	U��wXc�A�*

loss�cF<��H       �	�r�wXc�A�*

loss6b�<4I�c       �	~;�wXc�A�*

lossLk=���       �	�!�wXc�A�*

loss�/<��n@       �	��wXc�A�*

loss��<F��       �	�X�wXc�A�*

lossa��<�i>       �	�wXc�A�*

loss��:=ة�}       �	t��wXc�A�*

loss
��;���+       �	�U�wXc�A�*

loss�@V<>���       �	p�wXc�A�*

loss�Q<�۞       �	]��wXc�A�*

lossTK�=����       �	q�wXc�A�*

lossc�=.�.U       �	���wXc�A�*

loss��<����       �	'K�wXc�A�*

loss�l�<?ւ�       �	��wXc�A�*

loss���<B��       �	į�wXc�A�*

loss1�<��
�       �	�N�wXc�A�*

lossS��=Զ�       �	��wXc�A�*

loss)[#=��(       �	���wXc�A�*

loss��=�cP       �	J��wXc�A�*

loss�9�;aj!�       �	;�wXc�A�*

lossw��;a�Ǟ       �	�ߕwXc�A�*

loss�;�;
�S!       �	���wXc�A�*

loss8��<�d�       �	��wXc�A�*

lossh[�:���e       �	鸗wXc�A�*

loss��<2,       �	�P�wXc�A�*

lossp�=.^!       �	�wXc�A�*

lossR�d<�L�|       �	$~�wXc�A�*

loss�h�:���       �	��wXc�A�*

loss�L~=̨RI       �	��wXc�A�*

loss��<UC73       �	9E�wXc�A�*

loss���;I��       �	ޛwXc�A�*

loss�1�=Aұ        �	\w�wXc�A�*

loss��=(�J       �	�
�wXc�A�*

loss��J;%�v�       �	k��wXc�A�*

loss&(�<*�'       �	}A�wXc�A�*

loss�y<>|�       �	<ޞwXc�A�*

loss#��<,��       �	~�wXc�A�*

loss�n<�d-�       �	@�wXc�A�*

loss�t�<�R0       �	��wXc�A�*

loss$&n;��a       �	�P�wXc�A�*

loss�uC=,I6�       �	���wXc�A�*

lossZ��<���       �	҉�wXc�A�*

lossܯ�;cj�       �	>&�wXc�A�*

loss�;�<ͷ�[       �	~ţwXc�A�*

lossR:��T�       �	�j�wXc�A�*

loss�f�=3O�       �	��wXc�A�*

lossa��<U���       �	���wXc�A�*

loss��<�|�       �	P8�wXc�A�*

lossŽF<��Yq       �	NզwXc�A�*

loss��<�>��       �	Uk�wXc�A�*

loss'�=~)k       �	:�wXc�A�*

loss�H=�{WY       �	�wXc�A�*

lossAS�<P::       �	P9�wXc�A�*

loss�3�;R��       �	�ҩwXc�A�*

loss*��<z�
       �	�i�wXc�A�*

lossd��<ul       �	A�wXc�A�*

loss�=|��S       �	�ګwXc�A�*

loss�,<�!՝       �	ߋ�wXc�A�*

loss#)%<��N�       �	H2�wXc�A�*

loss���;`�٬       �	�ԭwXc�A�*

loss�iO:;�zF       �	]m�wXc�A�*

loss�|g<���8       �	�wXc�A�*

lossoeW=l�lH       �	ᛯwXc�A�*

loss��<_�8       �	�5�wXc�A�*

lossن"=�`�       �	q̰wXc�A�*

loss��H=�Nc(       �	}w�wXc�A�*

lossF��<te��       �	��wXc�A�*

lossW��;�j       �	v��wXc�A�*

loss�Ho<��C�       �	�>�wXc�A�*

loss�#<=�cE       �	�ԳwXc�A�*

loss���<�f��       �	�k�wXc�A�*

loss�9�<f��       �	��wXc�A�*

lossse�<7��       �	�wXc�A�*

loss��u=��       �	2>�wXc�A�*

lossRӖ;)/�       �	YܶwXc�A�*

loss���:��'       �	mp�wXc�A�*

losst�:�xÉ       �	~�wXc�A�*

loss�K�=Y�|       �	��wXc�A�*

loss��k<�~��       �	I�wXc�A�*

loss���:z�z8       �	�wXc�A�*

loss���;�f@8       �	�|�wXc�A�*

loss�	�;ӟ��       �	��wXc�A�*

loss�O2<G�{U       �	Ӽ�wXc�A�*

lossf�;�H��       �	�S�wXc�A�*

lossz��<       �	��wXc�A�*

lossN�g;���       �	,��wXc�A�*

loss=��:A7�       �	��wXc�A�*

lossh6�=3zJ�       �	���wXc�A�*

loss\u<���       �	�K�wXc�A�*

loss� �<���i       �	��wXc�A�*

losswK�=��E       �	~�wXc�A�*

loss}��:���       �	w�wXc�A�*

loss�jv;Ҹ�       �	G��wXc�A�*

loss�1<����       �	_C�wXc�A�*

lossO�o:5nhC       �	h��wXc�A�*

lossn5i<:cJ9       �	g��wXc�A�*

loss��]=�1�J       �	�/�wXc�A�*

lossL=�<���       �	1	�wXc�A�*

loss�8.<�#70       �	Z��wXc�A�*

loss���<x�TD       �	U��wXc�A�*

loss���:��ic       �	���wXc�A�*

lossW/�<���;       �	T��wXc�A�*

loss��;����       �	x^�wXc�A�*

loss�z�;[r�       �	IG�wXc�A�*

lossx�7<��S       �	��wXc�A�*

loss��;G�+t       �	m��wXc�A�*

lossx =]��       �	%Z�wXc�A�*

loss�	�<.�k�       �	���wXc�A�*

lossq��<]�TP       �	ђ�wXc�A�*

loss� +=�ペ       �	N*�wXc�A�*

loss}>�|        �	���wXc�A�*

loss*[;7�XW       �	�l�wXc�A�*

loss�:C�d�       �	�wXc�A�*

loss��-<F�Z�       �	=��wXc�A�*

loss2�<��       �	/��wXc�A�*

loss�?�=��^+       �	>�wXc�A�*

loss�=��       �	0��wXc�A�*

loss B=��~       �	Yi�wXc�A�*

loss},�=g2��       �	�wXc�A�*

loss�9d<*�c       �	Ů�wXc�A�*

loss�Z==�ҿk       �	�F�wXc�A�*

lossC}=���       �	���wXc�A�*

loss�c<��(w       �	�s�wXc�A�*

loss�~�<���       �	�wXc�A�*

lossҘ|=`�%A       �	���wXc�A�*

loss�L�<�<�       �	�G�wXc�A�*

loss�@d=�Ͷ�       �	��wXc�A�*

loss���<}���       �	lz�wXc�A�*

loss�8�<��       �	o�wXc�A�*

lossy<öG�       �	ҩ�wXc�A�*

loss8y<�נ       �	�?�wXc�A�*

loss�>�<��!       �	��wXc�A�*

loss�@�:�U��       �	"p�wXc�A�*

loss&�<���       �	�wXc�A�*

loss��7=b��1       �	���wXc�A�*

lossѦ�<v��       �	�0�wXc�A�*

lossOt>%�a.       �	\��wXc�A�*

loss��=���R       �	9a�wXc�A�*

loss��g=�A��       �	���wXc�A�*

loss�+�=r2��       �	��wXc�A�*

loss�=:��       �	A.�wXc�A�*

lossT�<Ol�       �	���wXc�A�*

loss|�<��jD       �	Xo�wXc�A�*

loss
X�<��Y       �	��wXc�A�*

loss��:K착       �	���wXc�A�*

loss�;k<����       �	�3�wXc�A�*

loss��G<��       �	��wXc�A�*

loss:��;����       �	p}�wXc�A�*

loss�<��u�       �	�I�wXc�A�*

loss6�<|A�       �	&��wXc�A�*

loss�<�E�       �	��wXc�A�*

loss�c2<�<D�       �	� �wXc�A�*

loss1�=��l       �	��wXc�A�*

loss���;h�       �	�V�wXc�A�*

lossʅ�;��F�       �	���wXc�A�*

loss���;k+5�       �	���wXc�A�*

loss��<$�y�       �	�C�wXc�A�*

loss�&�=E`��       �	���wXc�A�*

lossV��<<�'       �	�}�wXc�A�*

loss��s;|а�       �	�wXc�A�*

lossR�"<;��9       �	O��wXc�A�*

loss��=��^?       �	$C�wXc�A�*

lossd��<A��       �	���wXc�A�*

loss�R1=��       �	���wXc�A�*

loss:�<�       �	&�wXc�A�*

loss��);^��R       �	���wXc�A�*

loss.�}<��3       �	�L�wXc�A�*

loss�Ё=��sx       �	���wXc�A�*

loss6=<��6       �	g~�wXc�A�*

loss8�O<G�v�       �	M�wXc�A�*

loss�i~<��A       �	���wXc�A�*

loss��=	���       �	�>�wXc�A�*

loss�ϔ=��       �	V��wXc�A�*

loss$&\;�R�~       �	�f�wXc�A�*

lossP�;f:QR       �	 �wXc�A�*

loss�oq=ާ��       �	 ��wXc�A�*

loss{�p<`w��       �	�-�wXc�A�*

loss�P{;=�;\       �	.��wXc�A�*

loss�`�:��Ā       �	�[�wXc�A�*

lossOw<S��       �	H��wXc�A�*

loss�w�<����       �	A��wXc�A�*

lossDSf<��G       �	l]�wXc�A�*

loss�<��T       �	���wXc�A�*

losseg�<�<�       �	v��wXc�A�*

loss���<�ۏ       �	9F�wXc�A�*

loss�أ;����       �	��wXc�A�*

loss��x<2�v       �	���wXc�A�*

loss��<f�       �	!�wXc�A�*

loss`<ѽ��       �	D��wXc�A�*

loss�z@<n&       �	>Y xXc�A�*

loss��;�ד.       �	#2xXc�A�*

lossW׍<���       �	�xXc�A�*

lossn�;�=&       �	��xXc�A�*

lossDy�=�d��       �	��xXc�A�*

loss�G]=�� C       �	�kxXc�A�*

loss�<d���       �	G<xXc�A�*

loss�\F=��_       �	d�xXc�A�*

loss�z@=��u2       �	��xXc�A�*

loss;=<p�       �	�YxXc�A�*

loss���;�#+v       �	p�xXc�A�*

lossj�;���       �	S=	xXc�A�*

loss�=�X�       �	��	xXc�A�*

loss�+<�$��       �	��
xXc�A�*

loss#�F;��w�       �	�xXc�A�*

loss؋�<4��4       �	�RxXc�A�*

loss�*�=K�	       �	}�xXc�A�*

loss��<\@�       �	H�xXc�A�*

loss�V�<r��C       �	:ZxXc�A�*

loss�a4;_�D       �	=�xXc�A�*

loss)%�;�<ZD       �	e�xXc�A�*

loss��9���       �	6!xXc�A�*

loss�s$=�+,�       �	U�xXc�A�*

loss���;
��       �	F`xXc�A�*

loss/�U9�s�:       �	�xXc�A�*

loss���;K�i       �	X�xXc�A�*

loss.�=Z�@�       �	�DxXc�A�*

loss��o<�&�e       �	o�xXc�A�*

loss�e<=a3       �	��xXc�A�*

loss���=�"*�       �	N%xXc�A�*

loss,f=�S'       �	A�xXc�A�*

loss��M<y5-       �	�RxXc�A�*

loss���<�ԫl       �	�(xXc�A�*

lossip�;��	8       �	�8xXc�A�*

loss�lc={a       �	��xXc�A�*

loss�2�<>(��       �	�oxXc�A�*

loss_=����       �	�xXc�A�*

lossӻ9����       �	u�xXc�A�*

loss���;,}�$       �	�CxXc�A�*

loss�]�<�ީ�       �	��xXc�A�*

loss�'<s�       �	�xXc�A�*

loss�K�<t5�       �	d xXc�A�*

losst��;�\       �	��xXc�A�*

loss�@<��F�       �	NxXc�A�*

lossL�m<�@��       �	7�xXc�A�*

loss�S<��O�       �	{xXc�A�*

lossFː<x�l5       �	� xXc�A�*

loss3R�;z��       �	\� xXc�A�*

loss�j�<���V       �	B!xXc�A�*

loss|�=}m�       �	��!xXc�A�*

loss�`�<t`cb       �	�o"xXc�A�*

loss׬t<��>�       �	A+#xXc�A�*

loss���=��(Y       �	'�#xXc�A�*

lossU�<*��Z       �	�V$xXc�A�*

loss�-<�G�       �	`�$xXc�A�*

losse��;���(       �	%xXc�A�*

lossT�d=�T�       �	�&xXc�A�*

loss��:;��'/       �	��&xXc�A�*

loss�=ݵ'�       �	S?'xXc�A�*

loss:�<��?       �	��'xXc�A�*

lossq;<�0��       �	T�(xXc�A�*

loss
P^<��8�       �	�Y)xXc�A�*

loss!D�;��b`       �	��*xXc�A�*

loss�̗:�N�L       �	�+xXc�A�*

loss�G�;0~{w       �	S,xXc�A�*

loss�j�;G�&`       �	��,xXc�A�*

loss>M<��d�       �	j�-xXc�A�*

loss.�%=��|       �	z.xXc�A�*

loss�3�<�Ab,       �	ĳ.xXc�A�*

loss-��=�S^       �	UL/xXc�A�*

lossm&:Zg�@       �	��/xXc�A�*

lossz1�<���       �	5z0xXc�A�*

lossW�]<�L�       �	k1xXc�A�*

loss��:
�o-       �	��1xXc�A�*

loss�ڪ<�]       �	82xXc�A�*

loss��>�MI�       �	��2xXc�A�*

loss��<�}�       �	�k3xXc�A�*

loss�(<���       �	[4xXc�A�*

loss��/=I��       �	��4xXc�A�*

lossJh�=��n�       �	CV5xXc�A�*

loss�·;��ܩ       �	�5xXc�A�*

loss:=��o3       �	Z�6xXc�A�*

loss�F�;�J       �	n37xXc�A�*

loss�x�=X��       �	��7xXc�A�*

lossI�<!�#�       �	�y8xXc�A�*

loss���<s���       �	9xXc�A�*

loss�0$<�A�       �	��9xXc�A�*

loss>�<<�'z       �	A:xXc�A�*

loss��<�&�       �	=�:xXc�A�*

loss90;/O�J       �	Q�;xXc�A�*

loss��;GJo       �	i<xXc�A�*

loss���<��Xa       �	��<xXc�A�*

loss�<<�F�       �	nN=xXc�A�*

loss݋Q<���       �	��=xXc�A�*

loss
g-<��       �	ԁ>xXc�A�*

loss �<5��       �	�?xXc�A�*

loss�<􃖺       �	�?xXc�A�*

lossE��=���       �	EK@xXc�A�*

loss��<5&�;       �	��@xXc�A�*

loss̺w;�d��       �	qvAxXc�A�*

loss�=�O       �	�EBxXc�A�*

lossq<�<�u�       �	<�BxXc�A�*

lossJ��<IT4       �	�DxXc�A�*

lossu)=긣X       �	6�DxXc�A�*

loss��S;໳       �	��ExXc�A�*

lossA�	=Z@V       �	6 FxXc�A�*

loss��<�Ɔ�       �	-\GxXc�A�*

lossDg=둥M       �	2VHxXc�A�*

loss���=i�H       �	`IxXc�A�*

lossE-Y< �D       �	�$JxXc�A�*

loss�3�;҃��       �	;�JxXc�A�*

loss]A�;ܧ;�       �	��KxXc�A�*

lossf?�<�%}       �	��LxXc�A�*

loss/>��<�       �	o-MxXc�A�*

lossɛ�:�e�       �	��MxXc�A�*

loss��(<]�       �	��NxXc�A�*

loss�eI=���       �	z�OxXc�A�*

lossM<<,�       �	�lPxXc�A�*

loss��F=ｑ7       �	QxXc�A�*

loss��)=��'       �	!�QxXc�A�*

loss_�W;��       �	�^RxXc�A�*

loss���<�jȄ       �	�SxXc�A�*

lossܑ=9T��       �	��SxXc�A�*

loss�c);;g	T       �	�TTxXc�A�*

loss���<,��       �	wUxXc�A�*

loss\qM<C�l       �	�UxXc�A�*

loss��F<P�K�       �	_AVxXc�A�*

loss���:�d�       �	\WxXc�A�*

loss�ȑ;���,       �	S�WxXc�A�*

loss:�;՚��       �	�'XxXc�A�*

loss�-< �\.       �	`�XxXc�A�*

loss��U<k�^       �	EgYxXc�A�*

loss[<ǂ�       �	�ZxXc�A�*

loss�`�;�DA�       �	^�ZxXc�A�*

loss� �:/,�        �	q[xXc�A�*

losss�b=��       �	�\xXc�A�*

loss��7=e��       �	�\xXc�A�*

loss�6q<iz       �	Q2]xXc�A�*

loss���<C[�       �	S�]xXc�A�*

loss=W�;,���       �	xa^xXc�A�*

loss��5=_�w�       �	��^xXc�A�*

loss0<�u�]       �	m�_xXc�A�*

loss�\�=�°�       �	H�`xXc�A�*

loss�a<�d�p       �	��axXc�A�*

loss�<}i�       �	4bxXc�A�*

loss!o="��M       �	F�bxXc�A�*

loss��<y�n       �	pcxXc�A�*

lossm<��"       �	|GdxXc�A�*

lossi�F<��       �	<�dxXc�A�*

loss�:B<hM�       �	��exXc�A�*

loss�b�=�RZ       �	G<fxXc�A�*

loss4^�<�_       �	��fxXc�A�*

lossMo(=�6�       �		ogxXc�A�*

loss)�,;;v�*       �	QhhxXc�A�*

loss��W=�|�Z       �	�ixXc�A�*

loss��n<y��1       �	[BjxXc�A�*

loss..Z<�g       �	/�jxXc�A�*

loss �Q=�Ks�       �	@�kxXc�A�*

loss3��<����       �	�lxXc�A�*

lossD�<az��       �	�$mxXc�A�*

loss�jA<ז       �	a�mxXc�A�*

loss��<�wlQ       �	�[nxXc�A�*

lossVM<L���       �	}�oxXc�A�*

loss@��;A���       �	alpxXc�A�*

loss��=�`h�       �	�qxXc�A�*

loss��;~N�       �	��qxXc�A�*

lossʉ�<+��       �	qVrxXc�A�*

loss�9	<�D�       �	n�rxXc�A�*

loss�JH:�7�	       �	��sxXc�A�*

loss��;� �"       �	+5txXc�A�*

loss�l�;,2�       �	��txXc�A�*

loss.AA<��z       �	�ruxXc�A�*

loss���=�8<�       �	�vxXc�A�*

lossةj<U���       �	�vxXc�A�*

loss�$|=�&ۛ       �	�:wxXc�A�*

loss�[H;+�<�       �	k�wxXc�A�*

lossv7<���       �	`txxXc�A�*

loss�G>��ߟ       �	�yxXc�A�*

loss�:�;ۖa       �	کyxXc�A�*

loss.�=|���       �	�HzxXc�A�*

loss;�o<7���       �	��zxXc�A�*

loss�Ы=8�.4       �	1x{xXc�A�*

loss}_;�~       �	�|xXc�A�*

loss-��:�i�       �	h�|xXc�A�*

loss��d=Ζ�       �	K}xXc�A�*

losso+�=P��0       �	H�}xXc�A�*

lossS<*��8       �	6�~xXc�A�*

loss,� <�f��       �	y$xXc�A�*

lossD�*=^?�       �	��xXc�A�*

loss���<�y:       �	3S�xXc�A�*

lossߑ�=a��       �	���xXc�A�*

loss/�=ȣ       �	���xXc�A�*

loss�m<���       �	ׂxXc�A�*

loss��n;OR�g       �	���xXc�A�*

loss�ʌ=v�ˀ       �	zV�xXc�A�*

loss3��;�\}�       �	�xXc�A�*

lossl;<�䁲       �	��xXc�A�*

lossM��;�X2       �	�9�xXc�A�*

loss<��:�]m�       �	��xXc�A�*

loss��
<���       �	"ǈxXc�A�*

lossi�:�-�b       �	��xXc�A�*

loss�'0=�?_�       �		l�xXc�A�*

lossXW`<p))�       �	x�xXc�A�*

loss8u
=�6       �	��xXc�A�*

loss��=`�k       �	ٲ�xXc�A�*

loss�ݧ:�闟       �	Kx�xXc�A�*

loss�<;���I       �	�M�xXc�A�*

lossx	=�n�N       �	-�xXc�A�*

lossJ��=�5
�       �	y�xXc�A�*

lossּ,<�$��       �	NՐxXc�A�*

lossx��=L�J       �	䁑xXc�A�*

lossD�Q=]�ҥ       �	���xXc�A�*

loss�=�fr�       �	��xXc�A�*

loss;V�<���/       �	�R�xXc�A�*

loss��M<���       �	-�xXc�A�*

loss3��<s�F       �	=��xXc�A�*

loss=?=b'�*       �	j3�xXc�A�*

loss�)<�G��       �	���xXc�A�*

loss��R=�p�       �	!��xXc�A�*

lossF�<ǰ�a       �	zm�xXc�A�*

loss��;�fSr       �	G�xXc�A�*

loss�+F=��*       �	���xXc�A�*

lossR�=P8�       �	A,�xXc�A�*

lossT�]:��:0       �	�ϚxXc�A�*

loss)�2=A�&s       �	6w�xXc�A�*

lossX؜;��V�       �	�H�xXc�A�*

lossZ�a<��&9       �	��xXc�A�*

loss�=�<#�       �	|�xXc�A�*

loss���<LQ�       �	:ʞxXc�A�*

loss�]E=z-<m       �	Y��xXc�A�*

loss[�<<�B�       �	�'�xXc�A�*

loss��<�N��       �	�ϠxXc�A�*

loss;B;"��)       �	'g�xXc�A�*

loss�j
<H�-       �	��xXc�A�*

lossz�*=���       �	���xXc�A�*

lossd+�<�!Ȋ       �	IG�xXc�A�*

lossD-j<5�M       �	��xXc�A�*

loss��;=SՈ       �	I��xXc�A�*

loss���;v6�       �	�x�xXc�A�*

lossō�=�ټ�       �	��xXc�A�*

loss!	<�"��       �	i��xXc�A�*

lossg�:u{       �	�Q�xXc�A�*

loss!I<ʖr�       �	]�xXc�A�*

lossP�<��ڀ       �	���xXc�A�*

lossM��<�D*U       �	=a�xXc�A�*

lossc�;N�       �	��xXc�A�*

loss���<-�\�       �	D��xXc�A�*

loss�?=Y�u       �	!=�xXc�A�*

loss|$�;�l�q       �	S�xXc�A�*

loss
��:��>       �	xXc�A�*

lossG =Uck�       �	nl�xXc�A�*

lossx�K:���       �	M-�xXc�A�*

loss�B�<�n�G       �	`ǰxXc�A�*

lossVT==�x?�       �	ca�xXc�A�*

loss�w�<�H�5       �	j��xXc�A�*

loss�� ='72       �	���xXc�A�*

loss�/;ɚE�       �	P�xXc�A�*

loss�P;��=       �	��xXc�A�*

lossct<��       �	N��xXc�A�*

loss�.=9��       �	��xXc�A�*

loss���;4�r9       �	��xXc�A�*

loss���<�^�       �	hY�xXc�A�*

lossl�=!jIP       �	y�xXc�A�*

lossE�=;{X�Q       �	o��xXc�A�*

loss��:��r[       �	�E�xXc�A�*

losso�:�F��       �	(
�xXc�A�*

lossנ$;o�;       �	��xXc�A�*

loss�׋:$n9       �	�/�xXc�A�*

loss2 �;��       �	_
�xXc�A�*

lossav�8�@�y       �	���xXc�A�*

loss�i"<�nFf       �	~8�xXc�A�*

loss���:��Q�       �	�̼xXc�A�*

losscq�:J��       �	�]�xXc�A�*

lossPw�9J��       �	��xXc�A�*

loss{�C;v�&r       �	���xXc�A�*

losshq<��)�       �	c*�xXc�A�*

loss}�=<l�A�       �	ϿxXc�A�*

lossk��:)��       �	�b�xXc�A�*

loss4`�<թ�:       �	���xXc�A�*

loss�P�=,1d�       �	���xXc�A�*

loss_�:��!       �	M0�xXc�A�*

loss���=��}�       �	e��xXc�A�*

loss��=�W�3       �	b�xXc�A�*

loss鸢<i�)�       �	 �xXc�A�*

lossm�;#��       �	���xXc�A�*

lossB�=�޼T       �	w�xXc�A�*

loss��`=#'`�       �	��xXc�A�*

loss��!=�׶N       �	��xXc�A�*

loss�5$<�Vy�       �	>�xXc�A�*

lossM:�<i�V�       �	���xXc�A�*

loss���;b�Z�       �	qv�xXc�A�*

lossJ��;Z��       �	4�xXc�A�*

lossv!=O�f       �	���xXc�A�*

loss�<5�p       �	@k�xXc�A�*

loss3�<�h�       �	�xXc�A�*

lossθ�<%�p�       �	���xXc�A�*

loss�B=aٿ.       �	�8�xXc�A�*

loss�V{=��'       �	Q��xXc�A�*

loss��;=\�5�       �	6x�xXc�A�*

lossw�=�|��       �	��xXc�A�*

lossC�s:�G��       �	�F�xXc�A�*

lossj2�<9��       �	���xXc�A�*

loss%P�;ܙhy       �	��xXc�A�*

lossL<$#�       �	�2�xXc�A�*

loss�;�C�0       �	���xXc�A�*

loss�' :�/�       �	6\�xXc�A�*

loss���< Rr�       �	���xXc�A�*

lossJ��:%H=S       �	���xXc�A�*

loss@O`;̓�       �	��xXc�A�*

loss��=���       �	���xXc�A�*

loss\�<�g�       �	5�xXc�A�*

lossL��<.��       �	X�xXc�A�*

loss6�!=tԸ�       �	��xXc�A�*

lossh�D<备�       �	x�xXc�A�*

loss_��;EB��       �	I�xXc�A�*

lossI��;�.O�       �	���xXc�A�*

loss�?<�K�       �	t�xXc�A�*

loss� �<�2P       �	_�xXc�A�*

loss�U�=��'�       �	J��xXc�A�*

loss�T�<��i�       �	Q�xXc�A�*

lossJ�:��_�       �	��xXc�A�*

loss���<��i       �	�~�xXc�A�*

loss�I=�u       �	:"�xXc�A�*

loss��;c�'       �	(��xXc�A�*

loss�Ճ=�%!       �	Q�xXc�A�*

loss8��<}ѭ,       �	���xXc�A�*

lossr��=b~��       �	��xXc�A�*

loss�P�<�:       �	m�xXc�A�*

loss=�<X�b�       �	; �xXc�A�*

loss�g�;�d^5       �	���xXc�A�*

loss;a��        �	�A�xXc�A�*

loss�̬<A�9       �	K��xXc�A�*

lossN��<����       �	�`�xXc�A�*

loss$N=Iw��       �	D��xXc�A�*

loss$-�<�eL       �	}��xXc�A�*

lossDP4=\s�       �	|&�xXc�A�*

lossèQ<f�,�       �	N(�xXc�A�*

loss�ѽ;�Q��       �	��xXc�A�*

loss��<��L8       �	�X yXc�A�*

loss�S:<�R��       �	%yXc�A�*

loss�J=W���       �	I�yXc�A�*

loss\(�;H?�_       �	P5yXc�A�*

loss���<}뭳       �	P�yXc�A�*

loss�<�1f       �	2YyXc�A�*

loss�&�;iH��       �	��yXc�A�*

loss��;Q�l/       �	ΈyXc�A�*

loss_�E=����       �	W%yXc�A�*

loss�V�;V�'_       �	s�yXc�A�*

lossߌ�;�Pr       �	m�yXc�A�*

lossQS=/{       �	#yXc�A�*

loss@��<��2       �	f�yXc�A�*

loss[ӥ<��%�       �	�YyXc�A�*

lossO��=�k�       �	�yXc�A�*

lossؠ�:x��       �	#�	yXc�A�*

lossb!=9�}X       �	.
yXc�A�*

lossJk:��/�       �	J�
yXc�A�*

loss�=��>       �	�DyXc�A�*

loss X�<,ϫ       �	��yXc�A�*

losst�i<��z�       �	�yyXc�A�*

loss#Y$<Uz�$       �	oyXc�A�*

lossT�:=�^y;       �	��yXc�A�*

loss���<E�ή       �	4IyXc�A�*

lossE�:7��{       �	��yXc�A�*

loss��<��"       �	��yXc�A�*

loss �\<����       �	�,yXc�A�*

loss��5<
��^       �	�yXc�A�*

loss��<	'�       �	�fyXc�A�*

loss�C�;h`�       �	�yXc�A�*

lossp�<]��       �	�hyXc�A�*

loss�;>�D       �	�yXc�A�*

loss�S=��       �	��yXc�A�*

loss��<���       �	0cyXc�A�*

lossH4�;q~M       �	�byXc�A�*

lossð:��4       �	n�yXc�A�*

lossovu<H��       �	��yXc�A�*

loss�;<i;�       �	�]yXc�A�*

loss��=���       �	�yXc�A�*

loss*��;Ra��       �	<�yXc�A�*

loss2:q@J3       �	�iyXc�A�*

loss<�'�%       �	�yXc�A�*

loss��:�]       �	%�yXc�A�*

loss�m;�_}�       �	�LyXc�A�*

loss��&<x&�M       �	��yXc�A�*

loss�}<���       �	��yXc�A�*

lossL�=.� 
       �	!:yXc�A�*

loss�y�<�{u�       �	��yXc�A�*

lossj�<Zse-       �	�� yXc�A�*

loss9�
;]�p       �	�#!yXc�A�*

loss�D�:�^�       �	��!yXc�A�*

loss�>�;�;G       �	�s"yXc�A�*

loss�@.<갆f       �	�#yXc�A�*

loss��=ǟ
�       �	��#yXc�A�*

loss�&:8���       �	0J$yXc�A�*

loss�z�9��~�       �	��$yXc�A�*

loss�ء<tC�       �	4�%yXc�A�*

lossC�<\�iW       �	�3&yXc�A�*

loss&4;��K�       �	��&yXc�A�*

loss��l=a�       �	Ii'yXc�A�*

loss�y�;;�Q8       �	
(yXc�A�*

loss"�	>7�@�       �	�(yXc�A�*

lossC�>l��       �	 S)yXc�A�*

loss4c<�,��       �	�H*yXc�A�*

loss��;=u;F       �	P+yXc�A�*

loss�;��i       �	ö+yXc�A�*

loss9�<u��       �	�s,yXc�A�*

loss�3=w���       �	�-yXc�A�*

loss���<~J��       �	��-yXc�A�*

loss�st;��7       �	�N.yXc�A�*

loss�+p;�6l       �	��.yXc�A�*

lossϩ<"�B$       �	h�/yXc�A�*

loss��;=k{v5       �	b-0yXc�A�*

losscІ<��a�       �	��0yXc�A�*

loss�@=I��       �	ji1yXc�A�*

loss�<�Zl       �	2yXc�A�*

loss��E;i�'       �	�2yXc�A�*

loss�;�(+�       �	�;3yXc�A�*

loss$W=*�-       �	��3yXc�A�*

loss2=��       �	�q4yXc�A�*

loss��*<m��       �	�
5yXc�A�*

loss��;D�0�       �	z�5yXc�A�*

loss���< �~�       �	d>6yXc�A�*

loss4�;K�l�       �	��6yXc�A�*

loss:�;~$��       �	�q7yXc�A�*

loss��;F��-       �	�8yXc�A�*

loss��O<'�M�       �	̶8yXc�A�*

loss�P�=����       �	iS9yXc�A�*

lossJ"�:���	       �	(�9yXc�A�*

loss��;E~�@       �	C�:yXc�A�*

lossq>�<"�#l       �	�4;yXc�A�*

lossIq�=���h       �	�G<yXc�A�*

lossʊ<�d��       �	h�<yXc�A�*

loss��3<�.'       �	�=yXc�A�*

loss�j�:>���       �	O!>yXc�A�*

loss�e�;|h�0       �	��>yXc�A�*

loss]&<����       �	f?yXc�A�*

loss�k�=��       �	��?yXc�A�*

losse�_; x�       �	à@yXc�A�*

loss�y:�M#       �	kIAyXc�A�*

loss<��<��5       �	(�AyXc�A�*

loss�=�:X�       �	��ByXc�A�*

lossf��:�       �	�TCyXc�A�*

loss�k�<զ��       �	�0DyXc�A�*

loss))=��       �	�EyXc�A�*

loss�{,=�j�       �	f�EyXc�A�*

loss{T�;u�"       �	lzFyXc�A�*

lossn��<4Ya�       �	R�GyXc�A�*

loss��6<�H9       �	NaHyXc�A�*

loss��:<�}��       �	`IyXc�A�*

loss�=H+u       �	k�IyXc�A�*

loss@�<]��K       �	zUJyXc�A�*

loss�oh<�ֺ       �	��JyXc�A�*

loss3�;�P�/       �	ɐKyXc�A�*

loss��r<�hI�       �	�qLyXc�A�*

loss�R�;�&�       �	�MyXc�A�*

loss��p<3p��       �	��MyXc�A�*

loss/�:bI-�       �	��NyXc�A�*

lossf�^<�'�       �	b.OyXc�A�*

loss���<-�t=       �	$�OyXc�A�*

loss��<�&>�       �	s�PyXc�A�*

loss��a<�?f       �	�xQyXc�A�*

loss6<0���       �	TRyXc�A�*

loss���<=�A       �	T�RyXc�A�*

loss�z\<)��4       �	;oSyXc�A�*

loss�Z�<ۺx�       �	(TyXc�A�*

loss�|�;!Ч:       �	��TyXc�A�*

loss��6=��&       �	�wUyXc�A�*

loss��a;a�9�       �	�VyXc�A�*

loss�G<t!��       �	�VyXc�A�*

loss�:;2g�       �	�MWyXc�A�*

lossv�;�7(�       �	��WyXc�A�*

loss��z<�ݑ�       �	6�XyXc�A�*

loss��: p�       �	�(YyXc�A�*

loss��-=�BB�       �	e�YyXc�A�*

loss_1�<v��1       �	�ZyXc�A�*

loss-��<��S       �	�$[yXc�A�*

loss�u=����       �	^�[yXc�A�*

lossS�<�=�       �	�m\yXc�A�*

loss�?B=,.�;       �	�]yXc�A�*

loss�1';��o       �	u�]yXc�A�*

loss�߱:p���       �	�L^yXc�A�*

loss�x�<�w�~       �	g�^yXc�A�*

loss�h�;�@       �	��_yXc�A�*

loss�]�<�xaW       �	�7`yXc�A�*

loss�Wg=�~)�       �	��`yXc�A�*

loss�<�`��       �	g�ayXc�A�*

lossT�=��R        �	$byXc�A�*

loss߷�:x�.       �	��byXc�A�*

loss�&];���       �	�ecyXc�A�*

loss��9=�s�       �	�dyXc�A�*

lossDD�=lP"�       �	8�dyXc�A�*

lossJ�;�RH       �		ReyXc�A�*

loss�ϡ<�f       �	��eyXc�A�*

loss6K;�^��       �		�fyXc�A�*

loss�`�<�~�       �	�gyXc�A�*

lossi��9�3cW       �	hyXc�A�*

loss{�(<%�Y�       �	�-iyXc�A�*

lossR�U;VZ^+       �	S�iyXc�A�*

loss��:��]�       �	�ljyXc�A�*

lossʌ�<�gDN       �	=kyXc�A�*

loss���<MR
�       �	��kyXc�A�*

loss�A<1��       �	�UlyXc�A�*

loss��?;�=�H       �	k�lyXc�A�*

loss?�%=�e9       �	a�myXc�A�*

loss�3<��Vj       �	�0nyXc�A�*

loss_�<�]       �	��nyXc�A�*

losstD�<��7�       �	c}oyXc�A�*

loss�<7=3���       �	�pyXc�A�*

loss�׶=��p�       �	 �pyXc�A�*

loss���<���       �	lqyXc�A�*

loss���<d�h0       �	h!ryXc�A�*

loss 3;7}�       �	.�ryXc�A�*

loss��r:v2�        �	nsyXc�A�*

loss��=
˚l       �	otyXc�A�*

lossIU=/�Y�       �	5�tyXc�A�*

loss17<��:       �	�NuyXc�A�*

loss���;�p       �	��uyXc�A�*

loss�)�:�\yK       �	,�vyXc�A�*

loss�=�X],       �	pBwyXc�A�*

lossAJm<�z       �	'�wyXc�A�*

loss|�==VY9�       �	�uxyXc�A�*

loss��;We@       �	gyyXc�A�*

loss4� =�ҥ�       �	��yyXc�A�*

loss�	�;z��@       �	�*{yXc�A�*

loss� <�j?,       �	��{yXc�A�*

loss@J^;����       �	Ie|yXc�A�*

loss�1';P��       �	K}yXc�A�*

lossH�w=��Yh       �	ݗ}yXc�A�*

lossXy};��       �	�-~yXc�A�*

lossm�=�ڂ�       �	��~yXc�A�*

loss�9t=:LW�       �	`yXc�A�*

losse��;0���       �	4�yXc�A�*

losss*	>��       �	H��yXc�A�*

lossd H=(�i#       �	�4�yXc�A�*

loss�d�<�a       �	��yXc�A�*

loss�{�;�D�       �	���yXc�A�*

loss���:E�/�       �	�=�yXc�A�*

lossM8	=	M�       �	EՃyXc�A�*

loss���<��       �	&q�yXc�A�*

loss��U=��       �	��yXc�A�*

losshi=��&�       �	s�yXc�A�*

lossk�=�s$/       �	���yXc�A�*

loss.{�<I�~�       �	�݇yXc�A�*

loss��;�_��       �	���yXc�A�*

loss/g<*
4       �	��yXc�A�*

loss��;�Z�_       �	&�yXc�A�*

loss�a=�QV�       �	,ՋyXc�A�*

lossHZz=��|       �	*ʌyXc�A�*

loss�$#>�1��       �	�j�yXc�A�*

loss��<�W!K       �	7 �yXc�A�*

loss��:<�,zw       �	+��yXc�A�*

loss�7�=��j"       �	�[�yXc�A�*

loss! =�zBN       �	U��yXc�A�*

loss1%�;N �G       �	⭐yXc�A�*

loss�_<��g       �	_^�yXc�A�*

loss,�<\�~       �	�y�yXc�A�*

loss�l=<&d��       �	1�yXc�A�*

loss�K<�07       �	%ϓyXc�A�*

lossM�b;S�|�       �	�ΔyXc�A�*

loss�#�<4.�^       �	q�yXc�A�*

loss�q�;4%5�       �	L�yXc�A�*

loss DP<��       �	��yXc�A�*

loss���;0&d       �	Z��yXc�A�*

lossÙ�;1��       �	�1�yXc�A�*

loss�$=��=       �	�yXc�A�*

loss�d-=5���       �	���yXc�A�*

loss�_�<��!�       �	�%�yXc�A�*

loss%��<r�       �	s��yXc�A�*

loss;@E=*6R       �	�o�yXc�A�*

loss�%B<ʣ�P       �	�yXc�A�*

loss
�=U[1       �	E��yXc�A�*

lossf��:��,F       �	�Q�yXc�A�*

loss��;�'�       �	��yXc�A�*

loss��<(6�       �	M��yXc�A�*

lossQ�=��       �	�8�yXc�A�*

lossR�A<}��\       �	FϟyXc�A�*

lossâ�=bl�8       �	a��yXc�A�*

loss��<���
       �	�$�yXc�A�*

loss���<�_       �	աyXc�A�*

loss���<�l       �	���yXc�A�*

loss3<fvS�       �	� �yXc�A�*

loss��;p3��       �	��yXc�A�*

loss��,=���       �	�m�yXc�A�*

loss!�'<�h͗       �	��yXc�A�*

loss�<���F       �	��yXc�A�*

loss�]�:�9       �	�>�yXc�A�*

loss2��:��       �	'��yXc�A�*

loss1_l:zH�       �	���yXc�A�*

loss���<���       �	�;�yXc�A�*

lossNph;0SY7       �	��yXc�A�*

loss6�h=���&       �	��yXc�A�*

loss�=�)b       �	O�yXc�A�*

loss���;n�7       �	���yXc�A�*

loss�^�;��3       �	~R�yXc�A�*

lossf�&<��       �	��yXc�A�*

lossj?s<�
q�       �	i�yXc�A�*

loss�1=g�       �	ᶭyXc�A�*

loss��A;��;       �	eR�yXc�A�*

lossv0=���       �	��yXc�A�*

losswi<�_��       �	0��yXc�A�*

lossБ=}�3�       �	�O�yXc�A�*

loss��d=|��       �	��yXc�A�*

loss(�;��%       �	�±yXc�A�*

loss��<A&�       �	��yXc�A�*

loss�D<�\       �	ZG�yXc�A�*

lossx��;�#{       �	 �yXc�A�*

loss��=��       �	클yXc�A�*

loss1�<���       �	.�yXc�A�*

loss.��;cY��       �	���yXc�A�*

loss�:���       �	件yXc�A�*

loss)�p<���       �	9��yXc�A�*

loss���<�         �	Y1�yXc�A�*

loss{��;�ee       �	���yXc�A�*

loss��;�?��       �	���yXc�A�*

lossA�I;N�\w       �	�3�yXc�A�*

loss��<�ŜG       �	OκyXc�A�*

loss{�=�2 �       �	nh�yXc�A�*

loss�M�;�=l�       �	��yXc�A�*

loss-E�;�h��       �	���yXc�A�*

lossz�<;�0d       �	i6�yXc�A�*

loss]f�;�ZDX       �	ֽyXc�A�*

loss�t></*z�       �	�m�yXc�A�*

loss�Ǖ9ƒ��       �	��yXc�A�*

loss��<���       �	���yXc�A�*

loss�1<�P�       �	y=�yXc�A�*

loss��M;Z�y:       �	���yXc�A�*

lossO�,=���       �	]��yXc�A�*

lossʮ=/�$	       �	X7�yXc�A�*

lossO�=��v�       �	��yXc�A�*

loss�E=ȩy<       �	��yXc�A�*

loss�%k<jGQ�       �	���yXc�A�*

loss���<z0�N       �	��yXc�A�*

loss케<�2��       �	��yXc�A�*

loss��S<�q.       �	�W�yXc�A�*

loss#m;���       �	0��yXc�A�*

loss�} =��       �	���yXc�A�*

loss]�U<� -       �	�5�yXc�A�*

loss�(C=���       �	���yXc�A�*

loss��J:J�S�       �	�t�yXc�A�*

loss7�;��v:       �	Q�yXc�A�*

lossl��;�xw       �	��yXc�A�*

loss7-<<�S       �	Na�yXc�A�*

lossH":PST       �	\;�yXc�A�*

loss+�=Cn2       �	���yXc�A�*

loss�Y�<��       �	�p�yXc�A�*

loss��';�9��       �	��yXc�A�*

loss�n#;�F       �	��yXc�A�*

loss��=
��t       �	�x�yXc�A�*

loss��;W��       �	�yXc�A�*

loss��.=�ꞛ       �	R��yXc�A�*

loss��<�?��       �	Y�yXc�A�*

loss���=ɑ?�       �	&��yXc�A�*

loss�2�<�́       �	+��yXc�A�*

loss;��<hnU�       �	xG�yXc�A�*

loss[0�<'�q       �	B��yXc�A�*

lossE�;<-K��       �	���yXc�A�*

loss2;�<��^�       �	'2�yXc�A�*

loss �=�D��       �	��yXc�A�*

loss��u=K%f-       �	<��yXc�A�*

loss��:{��       �	�?�yXc�A�*

lossV��< g�       �	���yXc�A�*

loss(�!=�~�.       �	�x�yXc�A�*

lossa�;I�kv       �	g�yXc�A�*

loss���;�i�+       �	2��yXc�A�*

loss��z<M��       �	@N�yXc�A�*

lossV�4<tzt       �	���yXc�A�*

loss�\;=�\�       �	��yXc�A�*

loss�^<�L0       �	?�yXc�A�*

loss*�d<�\U       �	���yXc�A�*

lossn´=��[       �	�Y�yXc�A�*

loss4c\=6��a       �	���yXc�A�*

lossa|s:���\       �	y��yXc�A�*

loss!8�<.�       �	 (�yXc�A�*

loss�T;	�u       �	ǻ�yXc�A�*

lossvx�;^o�[       �	O�yXc�A�*

lossR�m=V��#       �	&4�yXc�A�*

loss\��<�u{       �	���yXc�A�*

lossV�<i��       �	�w�yXc�A�*

loss2;;�ж!       �	]�yXc�A�*

loss\��<��w�       �	e��yXc�A�*

lossqnY<�*L�       �	/��yXc�A�*

loss
&�:U�       �	�]�yXc�A�*

loss���:�+/       �	��yXc�A�*

loss(Z�;����       �	���yXc�A�*

loss�v<��	�       �	&�yXc�A�*

lossxA�<>�Є       �	���yXc�A�*

loss���<�B7�       �	�l�yXc�A�*

lossd��:�|�       �	l�yXc�A�*

loss~;�R�       �	(��yXc�A�*

lossͫ�;X�J       �	�5�yXc�A�*

loss��<'���       �	�>�yXc�A�*

loss�.�< }�i       �	E��yXc�A�*

lossH�3=+7%�       �	�v�yXc�A�*

loss��j<>���       �	��yXc�A�*

loss��B=���       �	��yXc�A�*

loss}&�=@��       �	�^�yXc�A�*

losst�Q<�8F       �	�
�yXc�A�*

loss��<�iJ(       �	���yXc�A�*

loss`�N=E��       �	�\�yXc�A�*

loss<
<��       �	���yXc�A�*

lossQ�<qp�       �	ܞ�yXc�A�*

loss�#�<�       �	�@�yXc�A�*

lossƙ|=�y�U       �	l�yXc�A�*

loss_�j=� m~       �	Ͱ�yXc�A�*

lossi7�;��#       �	CS�yXc�A�*

loss��D=hb��       �	���yXc�A�*

loss��;�&M;       �	��yXc�A�*

loss�r<��4       �	�Z�yXc�A�*

loss��='Tb       �	���yXc�A�*

loss뇝<� �       �	��yXc�A�*

lossoA�<�q��       �	gH�yXc�A�*

loss1��;��(�       �	i��yXc�A�*

loss��8;�땅       �	s��yXc�A�*

loss�@6<'\�B       �	�(�yXc�A�*

loss���=��d       �	���yXc�A�*

loss���<�e�       �	�i�yXc�A�*

loss�<9-�^       �	��yXc�A�*

loss�(>ma�       �	צ�yXc�A�*

loss�a�<-. �       �	9G�yXc�A�*

loss&3=�v�       �	���yXc�A�*

loss���;<�D       �	I� zXc�A�*

lossO!<��       �	"zXc�A�*

loss�;{"3}       �	��zXc�A�*

loss��=���       �	MfzXc�A�*

loss��#<���h       �	?�zXc�A�*

loss��<oyuI       �	ݙzXc�A�*

loss�v�;�D%|       �	}?zXc�A�*

lossm��<�k�a       �	�EzXc�A�*

loss���8z�ƶ       �	��zXc�A�*

loss ��<��       �	O�zXc�A�*

loss%�0<��a       �	�OzXc�A�*

loss��<���       �	�zXc�A�*

loss���;,|�       �	+�zXc�A�*

loss��O;s7;S       �	�<	zXc�A�*

loss�/F=�%H�       �	�	zXc�A�*

lossU]=��       �	��
zXc�A�*

lossmJ5=�<��       �	&zXc�A�*

loss��<)�R�       �	��zXc�A�*

loss�3e=�_+       �	VzXc�A�*

lossD�"=��w�       �	zXc�A�*

loss	�<\G�;       �	DzXc�A�*

lossH�=F)0       �	�zXc�A�*

lossc�=y�F"       �	�SzXc�A�*

lossnQ�<x��       �	s�zXc�A�*

loss �U<���\       �	��zXc�A�*

loss<n�<�qǋ       �	�=zXc�A�*

loss��\=�B[�       �	Z�zXc�A�*

lossҩ�;r3�       �	XpzXc�A�*

loss�b5<�hu�       �	�zXc�A�*

loss�	><'�2       �	n�zXc�A�*

losslr	=O��       �	>zXc�A�*

loss$�=L�3�       �	�zXc�A�*

lossv��<!��       �	��zXc�A�*

loss��<5�?�       �	�=zXc�A�*

loss�FE;��^�       �	��zXc�A�*

loss���</13�       �	�}zXc�A�*

lossl!�<��v       �	zXc�A�*

loss�<��{�       �	��zXc�A�*

loss��};�K�       �	�zXc�A�*

loss�8E<���       �	q9zXc�A�*

lossn'�<1��       �	k�zXc�A�*

loss��:R��(       �	�zzXc�A�*

loss|}�;��_"       �	�zXc�A�*

lossC�f<�N�       �	�zXc�A�*

loss�5�<��/       �	$BzXc�A�*

loss�x�<3B�T       �	XzXc�A�*

lossh<�;�F       �	 �zXc�A�*

loss�<�65;       �	oKzXc�A�*

loss�ȍ;S�;       �	��zXc�A�*

lossT��<N�0       �	%y zXc�A�*

loss/�:�mӥ       �	�!zXc�A�*

loss�<�|�z       �	��!zXc�A�*

loss��=l�y�       �	�;"zXc�A�*

lossW�<B�<       �	��"zXc�A�*

loss{N0=�w�D       �	�#zXc�A�*

loss_�W<cId       �	�$zXc�A�*

lossTFq:�ҔB       �	�$zXc�A�*

lossrx=��2       �	�i%zXc�A�*

lossD�<�Q�       �	�&zXc�A�*

loss1Ȳ<���?       �	�&zXc�A�*

loss�=E<fj�:       �	w'zXc�A�*

lossl�<��       �	(zXc�A�*

loss���="�'       �	W�(zXc�A�*

loss�Y�:W'2�       �	9(*zXc�A�*

loss��|;�}�!       �	m�*zXc�A�*

loss��<(��       �	�]+zXc�A�*

loss	��;͈9�       �	,zXc�A�*

loss�V=X��       �	�,zXc�A�*

loss2"D<"C�       �	'h-zXc�A�*

loss �"<���       �	V.zXc�A�*

loss��:�z-E       �	��.zXc�A�*

loss�a�<ֳ�.       �	@K/zXc�A�*

loss�t�<.C�        �	90zXc�A�*

loss1�;�X�       �	}�0zXc�A�*

loss9�;��N�       �	�H1zXc�A�*

loss7�8=_       �	��1zXc�A�*

loss�ch;ޱ�       �	�q2zXc�A�*

lossT&[:�A�       �	�3zXc�A�*

loss�}�<:��       �	j�3zXc�A�*

loss��E:�1m9       �	�E4zXc�A�*

loss!S3;�u,�       �	H�4zXc�A�*

lossTA�:�	�       �	t5zXc�A�*

loss{�I=�#�       �	�6zXc�A�*

losse��<�Up       �	X�6zXc�A�*

loss�Q;;���       �	�E7zXc�A�*

loss��=m~A       �	�7zXc�A�*

loss�N:ԓ��       �	.v8zXc�A�*

loss�hW<��`       �	z9zXc�A�*

lossQ�<�Z��       �	�9zXc�A�*

loss�7�<�wg       �	+K:zXc�A�*

loss<0<b�9       �	h�:zXc�A�*

lossꪌ<��       �	��;zXc�A�*

lossZ�H=���       �	�<zXc�A�*

losspx=���       �	��<zXc�A�*

loss�R�;K��%       �	�I=zXc�A�*

loss�� ;�º       �	��=zXc�A�*

loss�v�=7X�Z       �	7�>zXc�A�*

loss#�I:ۧ"�       �	�"?zXc�A�*

lossc�;�w�       �	Ӽ?zXc�A�*

loss�8=��)       �	S@zXc�A�*

loss3�'<e��[       �	��@zXc�A�*

lossؓ<��/       �	�AzXc�A�*

loss	��;@%2       �	�BzXc�A�*

loss��^<Q�x       �	�BzXc�A�*

lossj�Z:�S�
       �	obCzXc�A�*

loss��<�o�       �	DzXc�A�*

loss巐:�+I       �	�EzXc�A�*

loss!��<���5       �	��EzXc�A� *

loss5<LC��       �	�FzXc�A� *

loss&zh;X�       �	]HzXc�A� *

loss��<���k       �	̴HzXc�A� *

loss�h=!x\p       �	�oIzXc�A� *

loss��;�w�       �	�JzXc�A� *

loss�=�<�       �	2�JzXc�A� *

lossO��;3�R       �	�BKzXc�A� *

loss*��:b5�5       �	 �KzXc�A� *

loss�*�<��J       �	��LzXc�A� *

lossma�:�o��       �	8�MzXc�A� *

loss��+<lt�       �	rNzXc�A� *

loss�k:r�       �	��NzXc�A� *

losssr�<k�       �	�FOzXc�A� *

loss�%�:��)�       �	nPzXc�A� *

loss/��;֖�       �	��PzXc�A� *

loss�Um=��>�       �	�QzXc�A� *

loss#۬;ْ��       �	\!RzXc�A� *

loss�fS<��]       �	��RzXc�A� *

loss�|�<�0��       �	PSzXc�A� *

loss:c�;`C"       �	��SzXc�A� *

losse��=5~8\       �	ɎTzXc�A� *

loss�=H8�`       �	�0UzXc�A� *

lossIt;s��a       �	��UzXc�A� *

loss��2<�Bwt       �	�VzXc�A� *

loss��;�(�       �	XWzXc�A� *

lossEIQ;�x.<       �	%�WzXc�A� *

loss�G�9s��#       �	�LXzXc�A� *

loss�Ӕ=;3�       �	�	YzXc�A� *

loss�';�a       �	��YzXc�A� *

lossh);e�       �	�;ZzXc�A� *

lossI� <.� .       �	��ZzXc�A� *

loss<K9<�P��       �	lx[zXc�A� *

loss���<C���       �	�\zXc�A� *

loss���;9�;.       �	 �\zXc�A� *

loss���;}�k�       �	<L]zXc�A� *

loss���=�$6�       �	W%^zXc�A� *

lossNr�<����       �	T�^zXc�A� *

loss
��;er}�       �	Mi_zXc�A� *

lossX=�<��Is       �	�`zXc�A� *

loss���<0S       �	ϣ`zXc�A� *

losst�	<%\��       �	;azXc�A� *

loss��[:�p       �	��azXc�A� *

loss,*L9Fɣ8       �	ubzXc�A� *

loss�D�;����       �	�czXc�A� *

loss�R%9�'��       �	ٴczXc�A� *

loss��8�N       �	~SdzXc�A� *

losse��8�2�       �	��dzXc�A� *

losso;D���       �	]�ezXc�A� *

loss3��;���U       �	k'fzXc�A� *

lossH�<<�՗�       �	�fzXc�A� *

loss�-;*O{       �	�[gzXc�A� *

lossTK>����       �	�gzXc�A� *

loss�צ=Xkp       �	ȚhzXc�A� *

loss-��:���i       �	�@izXc�A� *

loss�ݐ=R���       �	�izXc�A� *

loss�Y=mg�       �	��jzXc�A� *

loss2]�=͂��       �	I+kzXc�A� *

loss@�;Z7�>       �	 ^lzXc�A� *

loss��=L��       �		mzXc�A� *

lossڶ-=ToN       �	��mzXc�A� *

loss3�<@Q#�       �	�VnzXc�A� *

loss��;�b�       �	#�nzXc�A� *

loss���<#��p       �	��ozXc�A� *

lossL�<<�:��       �	zQpzXc�A� *

loss�3>�v#       �	��pzXc�A� *

loss�c`=��       �	��qzXc�A� *

lossp��<��7�       �	�;rzXc�A� *

lossW,R=R�!Q       �	H�rzXc�A� *

lossFc=5��       �	��szXc�A� *

lossA^q=]���       �	� tzXc�A� *

loss�J�;�k�       �	&�tzXc�A� *

loss+=��n�       �	�ruzXc�A� *

lossQ�<��"       �	�vzXc�A� *

losst�;�� �       �	��vzXc�A� *

loss�k<Y�_       �	�[wzXc�A� *

loss7��;`:�^       �	�xzXc�A� *

loss̮�:��W       �	�xzXc�A� *

loss:��;u�#�       �	JyzXc�A� *

loss�)<��,&       �	��yzXc�A� *

loss_F�=VB@�       �	��zzXc�A� *

losss��<�=��       �	�&{zXc�A� *

loss�<`�J�       �	x�{zXc�A� *

loss@��<ϝ��       �	�r|zXc�A� *

loss"��;�f��       �	ߊ}zXc�A� *

loss�C�<�qr,       �	�%~zXc�A� *

loss&�=t�g       �	z�~zXc�A� *

loss�Ծ:V�E       �	�dzXc�A� *

loss�f�<�lƑ       �	S�zXc�A� *

lossS�<PVM#       �	
��zXc�A� *

loss�m�<�d�;       �	�;�zXc�A� *

lossAL�;%ҷ�       �	/߁zXc�A� *

loss��^<h�ώ       �	���zXc�A� *

loss���<�2       �	C�zXc�A� *

loss�%:���       �	��zXc�A� *

loss��
<v�;�       �	��zXc�A� *

loss�.�<��K9       �	�zXc�A� *

loss�y=4��j       �	��zXc�A� *

loss��<�G1       �	��zXc�A� *

loss]*;�l��       �	�zXc�A� *

loss�P)=�=�       �	gC�zXc�A� *

loss\`�;�-�u       �	x^�zXc�A� *

loss �=@�T�       �	���zXc�A� *

loss���;�gVe       �	j�zXc�A� *

loss���;�rZ       �	.�zXc�A� *

loss؄s=��_�       �	��zXc�A� *

loss�W�<I�Y       �	>�zXc�A� *

loss�=<�X�       �	�ަzXc�A� *

loss]��:xL�N       �	���zXc�A� *

loss��<�=|�       �	bL�zXc�A� *

lossC5�;���       �	���zXc�A� *

loss��w<�w&       �	N)�zXc�A� *

loss��<|(�}       �	�˪zXc�A� *

lossC2==���       �	ڮ�zXc�A� *

lossH�Z=�l�       �	!��zXc�A� *

loss��<:�%o~       �	���zXc�A� *

lossTJ;�7�W       �	�8�zXc�A� *

loss��P<���2       �	��zXc�A� *

lossx<W��       �	S{�zXc�A� *

loss2�<;��       �	L�zXc�A� *

loss�^`<l�{       �	W��zXc�A� *

loss��<��x�       �	�J�zXc�A� *

lossϳ�<DS�q       �	+�zXc�A� *

loss�sx<�f��       �	��zXc�A� *

lossΧ=�X�       �	�X�zXc�A� *

loss�z:=es��       �	6�zXc�A� *

lossʥ�=�-�       �	ǝ�zXc�A� *

lossd
I<'�"       �	RC�zXc�A� *

loss��=觖       �	�ֵzXc�A�!*

loss�8�:!?�       �	�u�zXc�A�!*

lossH�;��       �	��zXc�A�!*

loss�<��#~       �	�ݷzXc�A�!*

lossi@
<y�u�       �	�o�zXc�A�!*

loss�7�<�|z       �	�	�zXc�A�!*

loss(d?<f�T       �	=��zXc�A�!*

loss���;|�       �	q:�zXc�A�!*

loss���<�\�       �	)ϺzXc�A�!*

loss/�:7Et�       �	sd�zXc�A�!*

loss���<���       �	Q��zXc�A�!*

loss��a<�s        �	N��zXc�A�!*

loss��<��a�       �	/�zXc�A�!*

loss�[;+��       �	�ŽzXc�A�!*

lossC�
=�A�       �	�Y�zXc�A�!*

loss��<���       �	5��zXc�A�!*

lossr�=��^O       �	���zXc�A�!*

lossZ�u<N�7l       �	37�zXc�A�!*

loss��q<��2�       �	���zXc�A�!*

lossc=�;�`yO       �	o�zXc�A�!*

lossJ�;Q�       �	S�zXc�A�!*

loss��;<�P Q       �	w��zXc�A�!*

loss��<Ws�Q       �	mX�zXc�A�!*

lossE�1=�N�t       �	��zXc�A�!*

loss:߃<���       �	���zXc�A�!*

loss��+=}�$%       �	'O�zXc�A�!*

lossDo�<0W�       �	`;�zXc�A�!*

loss�t%<mj�R       �	
�zXc�A�!*

loss��<.�L�       �	�zXc�A�!*

lossҌ�;x>       �	���zXc�A�!*

lossdU�<�y9�       �	�j�zXc�A�!*

loss�}<�ܪ�       �	�q�zXc�A�!*

losss��;���       �	��zXc�A�!*

loss�;-?�       �	��zXc�A�!*

lossE��9֊Sq       �	0�zXc�A�!*

lossk6;��)       �	���zXc�A�!*

loss�c<��j{       �	}Z�zXc�A�!*

lossC�Q=���       �	,��zXc�A�!*

loss�l�:���       �	��zXc�A�!*

loss�x;fO�X       �	A�zXc�A�!*

loss�<���       �	���zXc�A�!*

loss1;.�9       �	��zXc�A�!*

loss�A$;��       �	��zXc�A�!*

loss��y=�       �	��zXc�A�!*

lossds	<��*       �	�k�zXc�A�!*

loss��<S�*       �	5	�zXc�A�!*

loss�(U<�ȗZ       �	��zXc�A�!*

loss��=����       �	�;�zXc�A�!*

loss��#;�<"�       �	���zXc�A�!*

loss�BC<U�@Q       �	|d�zXc�A�!*

loss�0;F$��       �	��zXc�A�!*

loss�3<�.�       �	&��zXc�A�!*

loss F�=Ep�       �	~6�zXc�A�!*

loss-�v;��b       �	���zXc�A�!*

loss�;Ҽ�       �	�j�zXc�A�!*

loss(E�:-�`�       �	a6�zXc�A�!*

lossf��;(�       �	���zXc�A�!*

loss�G<����       �	�v�zXc�A�!*

losss��;>�ZP       �	�(�zXc�A�!*

loss��<�!�       �	
��zXc�A�!*

loss�~(:)��       �	|��zXc�A�!*

loss���9_���       �	�s�zXc�A�!*

loss��;!���       �	�#�zXc�A�!*

loss2(�;XX=       �	c��zXc�A�!*

loss\!= P2�       �	�r�zXc�A�!*

loss��<�D�B       �	^�zXc�A�!*

loss�i=��K~       �	���zXc�A�!*

loss�Y�=T�"E       �	�_�zXc�A�!*

loss
m�<+�M%       �	Z�zXc�A�!*

loss*�.=f�A0       �	 �zXc�A�!*

loss�=_g       �	{��zXc�A�!*

losss��<<[0�       �	�B�zXc�A�!*

loss� �9-(��       �	���zXc�A�!*

lossTa3=�m��       �	���zXc�A�!*

loss�q;9ac�       �	\r�zXc�A�!*

losszم<9�9�       �	��zXc�A�!*

loss�4=d�u�       �	��zXc�A�!*

loss���;݁��       �	�c�zXc�A�!*

loss���:GoT8       �	���zXc�A�!*

loss���;6ј�       �	���zXc�A�!*

loss�-�::<��       �	t%�zXc�A�!*

loss3�5;$��;       �	��zXc�A�!*

lossy<��ʵ       �	_�zXc�A�!*

lossZ�:p�v       �	���zXc�A�!*

loss6�<j-(�       �	V��zXc�A�!*

loss=<�G��       �	YM�zXc�A�!*

loss��;���       �	c�zXc�A�!*

loss�c7<,       �	���zXc�A�!*

lossd��;���       �	��zXc�A�!*

loss
K=�OW�       �	���zXc�A�!*

lossu�
;��5�       �	�_�zXc�A�!*

loss�^=��       �	��zXc�A�!*

loss4��97X��       �	Ͼ�zXc�A�!*

loss�Z�:P�6�       �	i�zXc�A�!*

loss@Z�<SI�       �	��zXc�A�!*

lossD7�;E��8       �	���zXc�A�!*

loss$��;��c�       �	;��zXc�A�!*

loss�χ:�5+H       �	�*�zXc�A�!*

loss$�;=��rb       �	���zXc�A�!*

loss�k;�@�       �	|�zXc�A�!*

loss��N;���       �	�'�zXc�A�!*

loss\��:#&��       �	���zXc�A�!*

loss��2;{�K       �	Ã�zXc�A�!*

loss��=@��       �	73�zXc�A�!*

loss�S�<`I�       �	+��zXc�A�!*

lossZ��<�|       �	��zXc�A�!*

loss)&�:�QI�       �	YL�zXc�A�!*

loss}��;�/.h       �	���zXc�A�!*

loss`Z<��6�       �	B��zXc�A�!*

loss��=T�Fv       �	�g�zXc�A�!*

loss��U:r4��       �	'�zXc�A�!*

loss#^T;(L��       �	Ի�zXc�A�!*

loss�A<�~�=       �	�Y {Xc�A�!*

lossv-;,�$"       �	(� {Xc�A�!*

loss���<�K��       �	C�{Xc�A�!*

loss���<QO�       �	�<{Xc�A�!*

loss��;��;       �	��{Xc�A�!*

lossR#
:��1       �	nk{Xc�A�!*

loss)�<��       �	�{Xc�A�!*

loss�K�<�
ǩ       �	L�{Xc�A�!*

loss8�%<�8�       �	�S{Xc�A�!*

loss�<��       �	 �{Xc�A�!*

lossfK<j���       �	w,{Xc�A�!*

loss��X=S�,�       �	��{Xc�A�!*

loss=$H:�D+�       �	��{Xc�A�!*

lossW�9o?.�       �	�<	{Xc�A�!*

loss��1;���H       �	w�
{Xc�A�!*

loss۟:`�8       �	��{Xc�A�!*

loss��M=O�V5       �	�{Xc�A�"*

loss��N<73l/       �	VD{Xc�A�"*

lossV�=�G�       �	��{Xc�A�"*

losslZ :���       �	ݔ{Xc�A�"*

lossUm74���       �	I/{Xc�A�"*

loss��9Ke �       �	��{Xc�A�"*

loss��=�t��       �	�{Xc�A�"*

lossF�<o5�       �	H�{Xc�A�"*

loss�5U;B       �	�f{Xc�A�"*

loss��:=�g|M       �	C{Xc�A�"*

loss��<�
	       �	%�{Xc�A�"*

loss�^�<�'vu       �	S�{Xc�A�"*

loss�p�:+�       �	bJ{Xc�A�"*

lossN��<�f�       �	fK{Xc�A�"*

loss��:`�0       �	E�{Xc�A�"*

loss8B:;w(!       �	{Xc�A�"*

loss���;�>�       �	)�{Xc�A�"*

loss1��;r�q�       �	��{Xc�A�"*

loss�<��&�       �	RC{Xc�A�"*

loss��1=t��e       �	nR{Xc�A�"*

losseL;�r h       �	i{Xc�A�"*

loss<�=�޲�       �	T{Xc�A�"*

loss)��;�G�       �	a�{Xc�A�"*

lossJ1�9�<B6       �	/{Xc�A�"*

losse��;[N       �	��{Xc�A�"*

loss��<�R�3       �	D� {Xc�A�"*

loss4[7=}6�%       �	� "{Xc�A�"*

lossqWW< ���       �	V�"{Xc�A�"*

loss	�*9z,I�       �	}�#{Xc�A�"*

loss3�:��}X       �	t`${Xc�A�"*

loss7�V=UE�        �	��${Xc�A�"*

loss���;�gy�       �	��%{Xc�A�"*

loss䋉;���8       �	7p&{Xc�A�"*

lossx?a:���|       �	�;'{Xc�A�"*

loss(��9|/�t       �	�(({Xc�A�"*

lossIn_;�t�       �	��({Xc�A�"*

lossܿ�;Ja��       �	��){Xc�A�"*

loss�d =���       �	��*{Xc�A�"*

loss#.�=�RK~       �	6\+{Xc�A�"*

lossRR=���       �	%,{Xc�A�"*

loss���<m��       �	��,{Xc�A�"*

lossn�;��       �	1'.{Xc�A�"*

lossl2<�`�Y       �	��.{Xc�A�"*

loss]�
:%W	       �	��/{Xc�A�"*

loss��:���h       �	߇0{Xc�A�"*

loss�͊:�0�x       �	Y�1{Xc�A�"*

loss[>h;>+�       �	�p2{Xc�A�"*

loss��<�^b       �	3{Xc�A�"*

lossM�<�Z��       �	(�3{Xc�A�"*

loss�u�;⼔�       �	�4{Xc�A�"*

loss&�L;Dw�&       �	&�5{Xc�A�"*

loss2�=l�T!       �	}x6{Xc�A�"*

lossf�
=�}�y       �	V)7{Xc�A�"*

lossz6�< @ۄ       �	k+8{Xc�A�"*

loss-z�:_��*       �	`9{Xc�A�"*

loss��"=��"�       �	2�9{Xc�A�"*

loss�%�<��       �	�Z:{Xc�A�"*

loss���<�<�       �	�;{Xc�A�"*

loss�I<~�<O       �	�<{Xc�A�"*

loss�8�;T��@       �	l	={Xc�A�"*

loss���;��       �	�G>{Xc�A�"*

loss@N�;��       �	��>{Xc�A�"*

loss�7U<߈�       �	��?{Xc�A�"*

lossm��<���       �	?@{Xc�A�"*

loss���;����       �	�A{Xc�A�"*

loss8l�<�Ǐ�       �	��A{Xc�A�"*

lossD9�<4�7�       �	�B{Xc�A�"*

loss � =mW�       �	�}C{Xc�A�"*

loss�"�<�U��       �	��D{Xc�A�"*

loss���<re�$       �	}xE{Xc�A�"*

loss1�;*Z�?       �	�LF{Xc�A�"*

loss}҉<�`8�       �	0KG{Xc�A�"*

loss��;��5�       �	�;H{Xc�A�"*

lossȅ�:�~�       �	�I{Xc�A�"*

lossZ�=���       �	m�I{Xc�A�"*

lossp��<��U       �	�fK{Xc�A�"*

loss��;}L,�       �	L{Xc�A�"*

lossJ�)=��[�       �	,�L{Xc�A�"*

loss!\&=���       �	�3M{Xc�A�"*

losss��:��@       �	x	N{Xc�A�"*

loss�r�9����       �	�N{Xc�A�"*

loss+Y<D0̛       �	cFO{Xc�A�"*

lossF!=�G�H       �	��O{Xc�A�"*

loss�3�<���)       �	g}P{Xc�A�"*

loss��:���       �	�"Q{Xc�A�"*

lossnO�<�VO�       �	��Q{Xc�A�"*

loss	%<��v       �	�kR{Xc�A�"*

loss�,~<�cHq       �	�S{Xc�A�"*

lossܒ:����       �	��S{Xc�A�"*

loss���:���       �	DRT{Xc�A�"*

loss�VD<�?�       �	w�T{Xc�A�"*

loss��<�<�       �	e�U{Xc�A�"*

loss�Al=���       �	|+V{Xc�A�"*

loss�Z<�]!       �	��V{Xc�A�"*

lossR <�z(w       �	�^W{Xc�A�"*

loss�1�:�h0/       �	��W{Xc�A�"*

loss��=%�0       �	��X{Xc�A�"*

losso݉<;���       �	�hY{Xc�A�"*

loss���;��L       �	Z{Xc�A�"*

loss�e<����       �	=�Z{Xc�A�"*

loss]-�<�ƾ�       �	�6[{Xc�A�"*

loss#$W;���       �	F�[{Xc�A�"*

loss[�k;s�e       �	2t\{Xc�A�"*

loss1�[:y��       �	�]{Xc�A�"*

loss�e�;ȶz       �	4�]{Xc�A�"*

loss���;��p       �	�~_{Xc�A�"*

loss�P�;R|_�       �	�"`{Xc�A�"*

loss[b5=F��       �	Ը`{Xc�A�"*

losswv�<H�       �	�Ra{Xc�A�"*

loss&;�S��       �	�a{Xc�A�"*

loss�|;�X}�       �	v�b{Xc�A�"*

loss۔�<3a�U       �	O#c{Xc�A�"*

loss�=䞛l       �	��c{Xc�A�"*

loss��=�p�9       �	ׇd{Xc�A�"*

lossؗ#=���       �	F"e{Xc�A�"*

loss�<=ﶿ       �	�f{Xc�A�"*

lossDE�;8o�K       �	�f{Xc�A�"*

lossT =8�W�       �	Ph{Xc�A�"*

lossՃ=戧�       �	�h{Xc�A�"*

loss��1<V���       �	�i{Xc�A�"*

loss�9W<��g6       �	�Dj{Xc�A�"*

loss�N�;��       �	��j{Xc�A�"*

loss���;��G       �	��k{Xc�A�"*

loss_p�=����       �	_(l{Xc�A�"*

losshs=G��W       �	am{Xc�A�"*

lossݚ;-��r       �	^�m{Xc�A�"*

loss#92;��_1       �	Vn{Xc�A�"*

lossH��<}:�       �	�o{Xc�A�"*

loss���<�k�       �	İo{Xc�A�#*

loss� :.ŅW       �	�{p{Xc�A�#*

loss��<w�
W       �	�q{Xc�A�#*

loss
�}<\G
       �	)�q{Xc�A�#*

loss:Y�;�i��       �	Rdr{Xc�A�#*

losst��;>(c       �	� s{Xc�A�#*

lossxr�<�lP�       �	$�s{Xc�A�#*

loss���<;ѽH       �	~6t{Xc�A�#*

lossf��;d�-       �	t&u{Xc�A�#*

loss�<G $�       �	��u{Xc�A�#*

lossA��<��b       �	�Rv{Xc�A�#*

losss� ;㌶�       �	��v{Xc�A�#*

loss�#�;hz�       �	f�w{Xc�A�#*

loss�W�;���E       �	+x{Xc�A�#*

lossZ�U:�y��       �	L�x{Xc�A�#*

loss�$N<r�?       �	/ny{Xc�A�#*

loss���<��b       �	�z{Xc�A�#*

lossi��:B>7�       �	]�z{Xc�A�#*

loss<�D</�X�       �	d={{Xc�A�#*

loss���<��D�       �	�{{Xc�A�#*

loss�N�<&3�       �	"l|{Xc�A�#*

losset�;Q]Ԛ       �	I}{Xc�A�#*

loss�v<�|�O       �	��}{Xc�A�#*

loss�)<�[��       �	
�~{Xc�A�#*

loss#1=�;�       �	'{Xc�A�#*

loss��=�-4�       �	��{Xc�A�#*

lossɓ�<R��S       �	�O�{Xc�A�#*

loss��n<����       �	��{Xc�A�#*

loss%d�;7rk       �	ލ�{Xc�A�#*

loss��<O,�       �	�1�{Xc�A�#*

loss�:�5�       �	Iւ{Xc�A�#*

loss4O�;��or       �	9{�{Xc�A�#*

loss��e;�G½       �	�{Xc�A�#*

lossl�W<q�L       �	�2�{Xc�A�#*

lossJ�"<��M�       �	2˅{Xc�A�#*

loss�U:h���       �	Mh�{Xc�A�#*

loss�+�=����       �	�{Xc�A�#*

loss&�':*ܗ       �	(F�{Xc�A�#*

lossS$A;~R/U       �	y@�{Xc�A�#*

loss��?=����       �	�m�{Xc�A�#*

loss���:5:k�       �	?�{Xc�A�#*

loss��!=%L�       �	d�{Xc�A�#*

lossë�;��[       �	,��{Xc�A�#*

lossR�:򈛇       �	L��{Xc�A�#*

lossk<ͥ�I       �	%͎{Xc�A�#*

lossq*�:0y�       �	�r�{Xc�A�#*

lossx�<�j#<       �	y>�{Xc�A�#*

loss�;���l       �	ِ{Xc�A�#*

loss��=q%�]       �	���{Xc�A�#*

losse��:�Z       �	�N�{Xc�A�#*

loss���=���1       �	[�{Xc�A�#*

lossrH;o�W�       �	�{Xc�A�#*

lossJh�:6h��       �	�ǔ{Xc�A�#*

lossIM�9Ñ��       �	5b�{Xc�A�#*

lossA/S<�s �       �	��{Xc�A�#*

lossk�=��D       �	���{Xc�A�#*

loss��	:Y��       �	V)�{Xc�A�#*

loss�z�<�B��       �	�{Xc�A�#*

loss%�y<��3       �	�V�{Xc�A�#*

loss)�)=�]%       �	Y�{Xc�A�#*

lossV�9o �p       �	��{Xc�A�#*

loss�c=�l �       �	�H�{Xc�A�#*

loss�Y;�NL[       �	���{Xc�A�#*

loss�@N<,U�       �	�}�{Xc�A�#*

loss��D9w)�G       �	��{Xc�A�#*

loss8�=�r6       �	���{Xc�A�#*

lossF��<S{۰       �	~Q�{Xc�A�#*

lossz)�<N&;�       �	V�{Xc�A�#*

lossaȕ<���       �	m��{Xc�A�#*

loss�Y�<nYl3       �	�"�{Xc�A�#*

loss��=s�C�       �	���{Xc�A�#*

loss���9Xm�t       �	�`�{Xc�A�#*

lossJ>;:V��       �	���{Xc�A�#*

loss䌂<H,%�       �	���{Xc�A�#*

loss�=���h       �	�E�{Xc�A�#*

lossב=�o�X       �	��{Xc�A�#*

loss=�;�5��       �	�{Xc�A�#*

lossU;�e`�       �	!\�{Xc�A�#*

loss���:Q���       �	��{Xc�A�#*

loss��<t&z�       �	&��{Xc�A�#*

lossÅ�;�+�       �	�$�{Xc�A�#*

lossL(=I�e#       �	���{Xc�A�#*

lossM��;�/�       �	�W�{Xc�A�#*

lossc�=ݣM�       �	�{Xc�A�#*

loss-n�=���-       �	4��{Xc�A�#*

loss���=���       �	N'�{Xc�A�#*

loss�n}=�Ij       �	���{Xc�A�#*

loss�[<��X       �	��{Xc�A�#*

loss��<�B�A       �	ȷ�{Xc�A�#*

loss3��:�m��       �	�Q�{Xc�A�#*

loss&>W=�Ƙd       �	��{Xc�A�#*

lossF|�;�|�       �	Z��{Xc�A�#*

loss�\;��x�       �	Է�{Xc�A�#*

loss:c�<�(�D       �	�]�{Xc�A�#*

loss�F<�E�a       �	v��{Xc�A�#*

loss$D�<���       �	 ��{Xc�A�#*

loss$��=0MA�       �	�X�{Xc�A�#*

loss=�*<) ^       �	*�{Xc�A�#*

loss�1�=���E       �	r�{Xc�A�#*

loss6k�:w��g       �	��{Xc�A�#*

lossZ�1=�(�       �	U�{Xc�A�#*

loss|ë<����       �	5��{Xc�A�#*

loss�<f���       �	�W�{Xc�A�#*

loss�!�=w�       �	��{Xc�A�#*

loss�H`;�]NQ       �	���{Xc�A�#*

loss�:���#       �	aR�{Xc�A�#*

loss�R�=���       �	h�{Xc�A�#*

loss���<«       �	Tɹ{Xc�A�#*

loss�u�;_U�       �	m��{Xc�A�#*

loss��
<C&�       �	n0�{Xc�A�#*

lossB|;o���       �	�ϻ{Xc�A�#*

loss��:���       �	+i�{Xc�A�#*

loss��;g��       �	�{Xc�A�#*

lossO�`=QG�       �	��{Xc�A�#*

loss|d<��;       �	9D�{Xc�A�#*

loss\�h<�o+       �	�{Xc�A�#*

loss�
�=�H�       �	�z�{Xc�A�#*

loss"Q<,�5�       �	�{Xc�A�#*

loss�#;:;�6       �	Ѱ�{Xc�A�#*

loss1��<D�W\       �	�G�{Xc�A�#*

lossT��<yz�       �	���{Xc�A�#*

loss���:��       �	���{Xc�A�#*

lossW/;y�""       �	i�{Xc�A�#*

lossԒ�<�	�j       �	��{Xc�A�#*

loss�1<M�9�       �	�H�{Xc�A�#*

lossAj&=���       �	��{Xc�A�#*

loss��2=�5�       �	�w�{Xc�A�#*

losst�<�u��       �	<J�{Xc�A�#*

loss/�2=�,G       �	P��{Xc�A�$*

loss��M<�	�       �	�K�{Xc�A�$*

lossNo/<�c:�       �	���{Xc�A�$*

loss�L�<	��       �	J{�{Xc�A�$*

loss)��=Kk��       �	1��{Xc�A�$*

loss��;n��	       �	�+�{Xc�A�$*

loss��<BN�       �	_��{Xc�A�$*

loss���;%;       �	���{Xc�A�$*

loss��;�/��       �	y�{Xc�A�$*

loss�D=�A�       �	��{Xc�A�$*

loss�<X���       �	���{Xc�A�$*

lossW��;��       �	y��{Xc�A�$*

lossX�V=o�r       �	bi�{Xc�A�$*

loss�/;F�|�       �	? �{Xc�A�$*

loss��e<�p��       �	��{Xc�A�$*

loss��=��K       �	+0�{Xc�A�$*

lossafQ=��M�       �	���{Xc�A�$*

loss�[=L�W       �	l]�{Xc�A�$*

lossL��<��	.       �	I��{Xc�A�$*

loss87�;���?       �	���{Xc�A�$*

lossm��=�Ud�       �	�'�{Xc�A�$*

loss�IC<��]A       �	n��{Xc�A�$*

lossC�;��       �	�Z�{Xc�A�$*

loss�=�F�a       �	=��{Xc�A�$*

lossɾE=Ȕ�@       �	@��{Xc�A�$*

loss1�;�D��       �	p[�{Xc�A�$*

loss���;o
��       �	g��{Xc�A�$*

loss��8=7]e       �	���{Xc�A�$*

lossk�;0\aA       �	�'�{Xc�A�$*

loss���;����       �	���{Xc�A�$*

lossOR=�-�       �	�]�{Xc�A�$*

loss*(�;k*Do       �	N�{Xc�A�$*

loss8=�;�Y3       �	��{Xc�A�$*

loss.&�=�xo�       �	+N�{Xc�A�$*

loss�Q�<���s       �	K��{Xc�A�$*

lossN!o<.Ҹ       �	]��{Xc�A�$*

loss��=��       �	=)�{Xc�A�$*

loss��a:��       �	���{Xc�A�$*

lossLM�<"fc�       �	sf�{Xc�A�$*

lossʙ�<��OH       �	��{Xc�A�$*

loss�p$;��.{       �	���{Xc�A�$*

loss��+<��;�       �	<f�{Xc�A�$*

loss�'�=����       �	h�{Xc�A�$*

lossȣ�<��t       �	E��{Xc�A�$*

loss�<�ww�       �	y=�{Xc�A�$*

loss��t<ݿ��       �	���{Xc�A�$*

loss��=M��)       �	��{Xc�A�$*

loss�z=� ��       �	j/�{Xc�A�$*

loss��<����       �	m��{Xc�A�$*

loss���<l�]       �	<g�{Xc�A�$*

loss�d=���^       �	��{Xc�A�$*

loss\�3<�Q؈       �	%��{Xc�A�$*

loss(&<n=�x       �	UL�{Xc�A�$*

loss�X�;T$�A       �	�O�{Xc�A�$*

loss�&;��P       �	1��{Xc�A�$*

loss=�;�m��       �	z��{Xc�A�$*

loss�$=�&c       �	�$�{Xc�A�$*

lossh�<9���       �	��{Xc�A�$*

loss!L!;���o       �	�M�{Xc�A�$*

loss�o=>��5       �	���{Xc�A�$*

loss��;�甦       �	���{Xc�A�$*

loss��;[.�       �	�o�{Xc�A�$*

loss8�~:,�&�       �	��{Xc�A�$*

loss@�=���X       �	s*�{Xc�A�$*

loss*��:H���       �	���{Xc�A�$*

loss�Ұ;���       �	m�{Xc�A�$*

loss�9�=����       �	�u�{Xc�A�$*

losso��;�'�4       �	R�{Xc�A�$*

loss�#g<�X�a       �	���{Xc�A�$*

loss6J;��l�       �	���{Xc�A�$*

loss�x;[���       �	nO�{Xc�A�$*

lossJVh;�2��       �	���{Xc�A�$*

loss�>�T��       �	��{Xc�A�$*

loss�q<=탚�       �	�*�{Xc�A�$*

lossS��<ˤe�       �	���{Xc�A�$*

lossQ��:���       �	��{Xc�A�$*

loss�2�:��-�       �	�S�{Xc�A�$*

loss��=|�`       �	���{Xc�A�$*

lossO"e;���       �	[}�{Xc�A�$*

loss��j;�	�       �	��{Xc�A�$*

loss<^Y<]G��       �	P��{Xc�A�$*

loss:�<�$��       �	 F�{Xc�A�$*

loss�(�;��S       �	d |Xc�A�$*

loss7�g<�[yP       �	-� |Xc�A�$*

loss��<1�P7       �	�5|Xc�A�$*

loss�;��${       �	��|Xc�A�$*

loss�J=�f-       �	�q|Xc�A�$*

lossc��9�0S       �	S|Xc�A�$*

loss�I1<=��7       �	{�|Xc�A�$*

loss@�;�\�       �	m<|Xc�A�$*

loss%��<���       �	x�|Xc�A�$*

loss6͠<��4I       �	�b|Xc�A�$*

lossq�:=�%\�       �	3�|Xc�A�$*

lossf)=dk�R       �	��|Xc�A�$*

loss�*;z�A       �	�%|Xc�A�$*

lossA�<���|       �	#|Xc�A�$*

loss�+<e��       �	Ӽ|Xc�A�$*

lossA�M=�W@"       �	�_	|Xc�A�$*

loss}y�:sF*g       �	�	|Xc�A�$*

lossh@�;��O       �	6�
|Xc�A�$*

loss�'<�u]       �	+3|Xc�A�$*

lossq9<{ v�       �	��|Xc�A�$*

loss�џ:P�W       �	�c|Xc�A�$*

loss�r�<���A       �	��|Xc�A�$*

lossJ��<��'       �	Ѯ|Xc�A�$*

lossDك<�LR       �	q|Xc�A�$*

loss#�r<bq�/       �	�
|Xc�A�$*

loss�@=����       �	��|Xc�A�$*

loss���="��\       �	!=|Xc�A�$*

loss�=����       �	�|Xc�A�$*

loss��:�i�       �	ʤ|Xc�A�$*

loss}�g< B$       �	�c|Xc�A�$*

loss�	�<nNbh       �	1|Xc�A�$*

loss�&<14��       �	1�|Xc�A�$*

loss��2;�q��       �	�w|Xc�A�$*

lossJ��9����       �	�|Xc�A�$*

lossV�G<c5�       �	��|Xc�A�$*

loss@<�k$       �	�h|Xc�A�$*

lossitQ=��       �	;�|Xc�A�$*

lossH��:��       �	��|Xc�A�$*

losso]�<UQ[u       �	 _|Xc�A�$*

loss$��<E��9       �	M�|Xc�A�$*

loss�"?<ҏL$       �	C�|Xc�A�$*

loss��_;�'�       �	�<|Xc�A�$*

lossE�h<��=       �	(�|Xc�A�$*

lossό�;0�e�       �	en|Xc�A�$*

loss�hX9z^S�       �	6|Xc�A�$*

loss@2=���       �	J�|Xc�A�$*

lossH��;��~7       �	s/|Xc�A�%*

loss�l;M7K�       �	�|Xc�A�%*

loss���;�2|>       �	�_|Xc�A�%*

loss��;�;��       �	��|Xc�A�%*

loss��<+��       �	r� |Xc�A�%*

loss/��9��       �	�1!|Xc�A�%*

loss�?:� �       �	��!|Xc�A�%*

loss�9@8hX��       �	�c"|Xc�A�%*

loss-��:ai�       �	��"|Xc�A�%*

loss���;���;       �	��#|Xc�A�%*

loss7��<1H�        �	�M$|Xc�A�%*

loss؏�98�
a       �	o�$|Xc�A�%*

lossER�9�`�        �	z�%|Xc�A�%*

loss�E�<	�t       �	�#&|Xc�A�%*

loss��:u�C�       �	0�&|Xc�A�%*

loss�i>ADH*       �	�T'|Xc�A�%*

loss.�=H�a       �	��'|Xc�A�%*

loss�P:��       �	�(|Xc�A�%*

lossq �;Q2�       �	�)|Xc�A�%*

lossƭ�<R��       �	W�)|Xc�A�%*

loss��]=s��       �	Y�*|Xc�A�%*

loss��<�x-       �	l=+|Xc�A�%*

loss� �=�hwv       �	=�+|Xc�A�%*

loss�x<mN�       �	��,|Xc�A�%*

loss��L<r%t       �	�M-|Xc�A�%*

loss���<&�       �	��-|Xc�A�%*

loss��6=�؏�       �	�.|Xc�A�%*

loss�W*<��L�       �	~/|Xc�A�%*

loss���;^G��       �	0|Xc�A�%*

loss���;��0�       �	ʦ0|Xc�A�%*

losss��;c�t       �	��1|Xc�A�%*

loss���<�       �	��2|Xc�A�%*

loss�mA<���       �	�X3|Xc�A�%*

lossT�<9���       �	��3|Xc�A�%*

loss�;�PҒ       �	��4|Xc�A�%*

loss�<]%N       �	�45|Xc�A�%*

loss��<J|�       �	�[6|Xc�A�%*

lossf<K��       �	��6|Xc�A�%*

lossm/<$�<�       �	�7|Xc�A�%*

loss�2�;)0�       �	�?8|Xc�A�%*

loss�/X<a��       �	��8|Xc�A�%*

loss�l�;>�k       �	ץ9|Xc�A�%*

lossqb�<f��       �	�<:|Xc�A�%*

loss!��<���       �	R�:|Xc�A�%*

loss�e;)�d       �	�v;|Xc�A�%*

loss�<�-�]       �	/<|Xc�A�%*

lossJ�;Z=�A       �	G�<|Xc�A�%*

loss�ӂ;���       �	Y�=|Xc�A�%*

loss��;�$�J       �	z>|Xc�A�%*

loss6��;Fd;       �	O�>|Xc�A�%*

lossoh�;<n�       �	�C?|Xc�A�%*

loss�Bn=Q��       �	$�?|Xc�A�%*

lossщ3=3�f�       �	A(A|Xc�A�%*

lossIY�;;��o       �	f�A|Xc�A�%*

loss��:�e       �	_`B|Xc�A�%*

lossj4d<�&�.       �	b�B|Xc�A�%*

loss4�;7ev[       �	-�C|Xc�A�%*

loss̤6=�ژ6       �	�4D|Xc�A�%*

loss6(5;}�Ӭ       �	�D|Xc�A�%*

loss�E�<��v:       �	6ZE|Xc�A�%*

loss2��<�9       �	9�E|Xc�A�%*

lossRL+<8��       �	�F|Xc�A�%*

lossx@�<d�J       �	M-G|Xc�A�%*

lossWK;�(~       �	�G|Xc�A�%*

loss�"�;��J#       �	��H|Xc�A�%*

loss ߢ;�:�       �	��b|Xc�A�%*

lossx3�<Lp�_       �	��c|Xc�A�%*

loss�/�;×e�       �	�{d|Xc�A�%*

loss��;$�       �	h"e|Xc�A�%*

loss#�<�e�       �	r�e|Xc�A�%*

loss��:t��       �	ǝf|Xc�A�%*

loss]��<�G�>       �	�Hg|Xc�A�%*

lossf��<�7Y       �	6�g|Xc�A�%*

loss6�W<'8�       �	��h|Xc�A�%*

lossfT�:��6�       �	z9i|Xc�A�%*

lossQ��:�č       �	��i|Xc�A�%*

lossT�0<5�?�       �	k|Xc�A�%*

loss��<��L       �	�k|Xc�A�%*

loss� |<~k�c       �	,Jl|Xc�A�%*

loss_%e;���       �	��l|Xc�A�%*

loss@>i:��A�       �	X�m|Xc�A�%*

loss�y�99�OP       �	�=n|Xc�A�%*

loss]hb;�~       �	�n|Xc�A�%*

loss��-<�.��       �	J�o|Xc�A�%*

loss��;{��t       �	�Qp|Xc�A�%*

loss3L�;�}JO       �	*�q|Xc�A�%*

loss�Z0=��?/       �	�1r|Xc�A�%*

lossVIZ:��jT       �	?9s|Xc�A�%*

lossJT =�G�s       �	� t|Xc�A�%*

lossb)�<R�H       �	k�t|Xc�A�%*

lossT�<g�       �	8u|Xc�A�%*

loss�;��       �	��u|Xc�A�%*

loss�i2=s�l       �	V�v|Xc�A�%*

lossf�T<���       �	6sw|Xc�A�%*

loss8�<���       �	�x|Xc�A�%*

lossv"<��a       �	$�x|Xc�A�%*

lossZ�2:�/�       �	mSy|Xc�A�%*

loss$�-=�LXf       �	��y|Xc�A�%*

loss0 =�J�       �	��z|Xc�A�%*

loss�7M;��]�       �	4�{|Xc�A�%*

loss�j4=��P�       �	�||Xc�A�%*

loss6��;N�;�       �	��||Xc�A�%*

lossL	<D��       �	�
~|Xc�A�%*

lossL��=��wz       �	J�~|Xc�A�%*

losss@ =_�B�       �	yX|Xc�A�%*

loss�n�:���i       �	J�|Xc�A�%*

loss�_:_��       �	���|Xc�A�%*

lossO!�:�3Ư       �	�^�|Xc�A�%*

loss�67;��W�       �	� �|Xc�A�%*

loss]o'=�F��       �	��|Xc�A�%*

lossZ� <����       �	�?�|Xc�A�%*

lossʁ�<����       �	��|Xc�A�%*

loss�&	;%�O       �	̴�|Xc�A�%*

loss1(q;R�'�       �	�S�|Xc�A�%*

lossA�{9�c�       �	��|Xc�A�%*

loss��:A�P1       �	|��|Xc�A�%*

loss�\*< �%       �	��|Xc�A�%*

loss���9���       �	���|Xc�A�%*

loss�2�=�߮       �	�P�|Xc�A�%*

loss8��;�Ab       �	��|Xc�A�%*

loss���:�,�       �	���|Xc�A�%*

loss3�<�}$       �	�*�|Xc�A�%*

loss���8'aZ�       �	�Ŋ|Xc�A�%*

loss.�h<�[�       �	p\�|Xc�A�%*

loss2��<�       �	��|Xc�A�%*

loss��<�T��       �	Û�|Xc�A�%*

loss�;�ь       �	?5�|Xc�A�%*

lossw�=�Ebc       �	!�|Xc�A�%*

loss��K:��Qy       �	D��|Xc�A�&*

loss� �<���       �	�G�|Xc�A�&*

loss�_�<� ��       �	C�|Xc�A�&*

lossS�<h1       �	���|Xc�A�&*

loss#�o;�$�       �	�.�|Xc�A�&*

loss�5<xbn       �	ّ|Xc�A�&*

loss�4<��I       �	|��|Xc�A�&*

loss��<�{       �	@3�|Xc�A�&*

loss�6<�e6�       �	^��|Xc�A�&*

lossJ�;h��&       �	͔�|Xc�A�&*

loss%� =/��S       �	 ;�|Xc�A�&*

loss�;#�׻       �	vݖ|Xc�A�&*

loss�=�w��       �	[��|Xc�A�&*

loss=��:�I0�       �	�^�|Xc�A�&*

lossFv<J&�)       �	���|Xc�A�&*

loss��@::{n�       �	���|Xc�A�&*

lossH<��z�       �	QJ�|Xc�A�&*

lossc��<�m�       �	N�|Xc�A�&*

loss�&<�1�9       �	;��|Xc�A�&*

lossռ#;�3#       �	�2�|Xc�A�&*

loss,K:?�Q�       �	R՜|Xc�A�&*

loss�_<�5J�       �	!��|Xc�A�&*

lossnp;F���       �	�V�|Xc�A�&*

loss=$�       �	���|Xc�A�&*

losskA<>��       �	���|Xc�A�&*

loss�;nBW�       �	�:�|Xc�A�&*

loss�;m1��       �	�נ|Xc�A�&*

loss4�=g��       �	�r�|Xc�A�&*

lossᡉ<�(��       �	j�|Xc�A�&*

loss�S<��S       �	���|Xc�A�&*

loss4.<C�;�       �	�c�|Xc�A�&*

loss`��=���g       �	D��|Xc�A�&*

lossjK<��֪       �	���|Xc�A�&*

loss��<���1       �	�5�|Xc�A�&*

loss'x�=/r��       �	�ԥ|Xc�A�&*

loss�,="n/L       �	Wv�|Xc�A�&*

loss��<i���       �	O �|Xc�A�&*

loss���8s��q       �	9ӧ|Xc�A�&*

loss࣊=��       �	>y�|Xc�A�&*

lossԀ�;�*L0       �	��|Xc�A�&*

loss�K=���       �	���|Xc�A�&*

loss��A<��       �	���|Xc�A�&*

loss�D<�(��       �	�)�|Xc�A�&*

loss���<`�~       �	�ګ|Xc�A�&*

loss<�	:0Ė       �	�|�|Xc�A�&*

loss ?�=ZB|       �	��|Xc�A�&*

loss��u;�bT�       �	ᴭ|Xc�A�&*

loss��<]��A       �	�R�|Xc�A�&*

lossJ�w;xEA1       �	�'�|Xc�A�&*

lossO�<�i&       �	߿�|Xc�A�&*

losss_x;H�       �	{k�|Xc�A�&*

loss&s�;M8t�       �	q�|Xc�A�&*

loss$T*<x��H       �	b��|Xc�A�&*

loss��</}:       �	���|Xc�A�&*

loss��N=N-��       �	�'�|Xc�A�&*

loss%f�;��O	       �	4۳|Xc�A�&*

lossc�=ş�@       �	o�|Xc�A�&*

loss�A�<�&F�       �	�	�|Xc�A�&*

lossS�d=�-
       �	a��|Xc�A�&*

loss�A�<o�y<       �	�I�|Xc�A�&*

loss���<�K)>       �	��|Xc�A�&*

lossdRD;T��H       �	|�|Xc�A�&*

loss�7�;>�a       �	��|Xc�A�&*

loss�=�3�       �	#��|Xc�A�&*

loss͘<�'��       �	�9�|Xc�A�&*

loss��=,�B�       �	��|Xc�A�&*

loss�I`=�?9(       �	���|Xc�A�&*

lossʣ�;՗a�       �	�|Xc�A�&*

loss.��9j�L       �	V��|Xc�A�&*

loss�!N=��0       �	�e�|Xc�A�&*

lossy/;�+�s       �	��|Xc�A�&*

lossX?�<��E       �	���|Xc�A�&*

lossw}.=��D�       �	b/�|Xc�A�&*

loss��<8��7       �	VӾ|Xc�A�&*

loss���;#0�       �	�i�|Xc�A�&*

loss�"�;\���       �	���|Xc�A�&*

lossV�=fz�       �	���|Xc�A�&*

loss�<�;j~��       �	c(�|Xc�A�&*

loss8��<!�a�       �	���|Xc�A�&*

loss;<��A       �	�g�|Xc�A�&*

lossX�<��9�       �	��|Xc�A�&*

lossO��<�?�{       �	M��|Xc�A�&*

loss:b=􈽖       �	d;�|Xc�A�&*

loss%�w<d$��       �	���|Xc�A�&*

loss��8:��_�       �	�l�|Xc�A�&*

lossgw;n]�y       �	��|Xc�A�&*

loss�nZ:;f��       �	Ú�|Xc�A�&*

loss�8;H9;j       �	@/�|Xc�A�&*

lossLR"=a�H       �	B��|Xc�A�&*

loss�!�;����       �	0f�|Xc�A�&*

lossLo=��c        �	VJ�|Xc�A�&*

lossV��<���       �	*��|Xc�A�&*

lossM�-9��m�       �	,��|Xc�A�&*

loss���:��       �	\!�|Xc�A�&*

lossl�y=-��d       �	��|Xc�A�&*

loss���;JMe       �	���|Xc�A�&*

loss�M=U���       �	�:�|Xc�A�&*

loss�n�<z)v�       �	���|Xc�A�&*

lossiN�;nx³       �	�}�|Xc�A�&*

lossZW=
�P       �	�F�|Xc�A�&*

loss]��9w�       �	��|Xc�A�&*

loss�#�<�Ǳ�       �	��|Xc�A�&*

loss�l�9]E9H       �	��|Xc�A�&*

loss,;�;�y��       �	��|Xc�A�&*

lossi�:���7       �	X�|Xc�A�&*

loss<S���       �	 �|Xc�A�&*

losssR';���       �	/��|Xc�A�&*

loss-�X<9��u       �	;T�|Xc�A�&*

loss{-:�KY       �	���|Xc�A�&*

loss��@9 �|       �	���|Xc�A�&*

loss��8<ye        �	 )�|Xc�A�&*

loss��9�LP�       �	 ��|Xc�A�&*

loss#k<��       �	�_�|Xc�A�&*

loss:��<,g�U       �	���|Xc�A�&*

loss/UJ;�y��       �	ٓ�|Xc�A�&*

loss.��<ӌa|       �	
/�|Xc�A�&*

loss`�:���       �	���|Xc�A�&*

loss�"<�	"�       �	w��|Xc�A�&*

loss�V =L��#       �	O#�|Xc�A�&*

lossj�<�eV�       �	���|Xc�A�&*

loss�Α;��u       �	�`�|Xc�A�&*

loss��:�4ۜ       �	���|Xc�A�&*

loss�W�:�c       �	��|Xc�A�&*

loss�< ���       �	f-�|Xc�A�&*

lossoi�<+F�       �	C��|Xc�A�&*

lossl�<�3A�       �	4e�|Xc�A�&*

loss�ؘ;�8�_       �	� �|Xc�A�&*

loss�r;Ί��       �	p��|Xc�A�&*

loss��;=?��       �	*6�|Xc�A�'*

loss���:mfW        �	��|Xc�A�'*

loss��:�	�       �	�`�|Xc�A�'*

loss�(�<R���       �	���|Xc�A�'*

loss]�;;:-ޥ       �	���|Xc�A�'*

loss�z�<`y       �	���|Xc�A�'*

loss��=t���       �	���|Xc�A�'*

loss���:�h��       �	�5�|Xc�A�'*

loss;Z�<y}i       �	���|Xc�A�'*

loss�^�<����       �	�l�|Xc�A�'*

lossN@$:ނ҄       �	��|Xc�A�'*

loss���<F�IS       �	���|Xc�A�'*

loss<�}<-!l       �	�*�|Xc�A�'*

loss��;�b�       �	��|Xc�A�'*

loss39P���       �	gb�|Xc�A�'*

loss�, =T]�       �	K�|Xc�A�'*

loss�K�<u��       �	6��|Xc�A�'*

loss�)<�y�       �	V)�|Xc�A�'*

loss���<̙L3       �	��|Xc�A�'*

loss!:��m       �	�^�|Xc�A�'*

loss��;��r<       �	���|Xc�A�'*

lossS;�I       �	���|Xc�A�'*

loss3;�:9[�       �	�.�|Xc�A�'*

loss\'1=�o�j       �	k��|Xc�A�'*

lossԠ�<܄�       �	��|Xc�A�'*

lossV}�=L���       �	cG�|Xc�A�'*

loss�q=�       �	���|Xc�A�'*

loss��^=�9U       �	��|Xc�A�'*

loss��<�~~       �	�-�|Xc�A�'*

lossNw2;m��       �		��|Xc�A�'*

loss��;=�aY       �	'��|Xc�A�'*

loss��:���"       �	D��|Xc�A�'*

lossb�='%��       �	���|Xc�A�'*

loss&Q�92;�       �	��|Xc�A�'*

loss e;xg޵       �	�.�|Xc�A�'*

loss<;C<����       �	�)�|Xc�A�'*

loss��	<��k�       �	��|Xc�A�'*

loss��;R��       �	���|Xc�A�'*

loss�OW;S�h�       �	)?�|Xc�A�'*

loss�L�;X/       �	+��|Xc�A�'*

loss�M:!���       �	�s�|Xc�A�'*

loss]p<o�Y�       �	P�|Xc�A�'*

loss��x;$�       �	˹�|Xc�A�'*

loss��=K���       �	�Q�|Xc�A�'*

loss�5�;�?&       �	c��|Xc�A�'*

loss��:�f��       �	T� }Xc�A�'*

loss�Y
=hwsh       �	�#}Xc�A�'*

lossǩ�<e��       �	(�}Xc�A�'*

lossx@Q;��xj       �	�P}Xc�A�'*

loss���;J\Y�       �	5
}Xc�A�'*

loss��<K�̳       �	�}Xc�A�'*

loss
�4:�q��       �	X8}Xc�A�'*

loss���<r�"       �	��}Xc�A�'*

loss���<�E�       �	6u}Xc�A�'*

loss��5=L�j�       �	}Xc�A�'*

loss;�g<�qeM       �	v�}Xc�A�'*

lossl�9�6�       �	�7}Xc�A�'*

loss?��<la�f       �	�"}Xc�A�'*

lossa4y;���5       �	�}Xc�A�'*

lossE=�;�pH�       �	�K	}Xc�A�'*

loss��`:W b8       �		�	}Xc�A�'*

loss$�n<p�)'       �	.u
}Xc�A�'*

loss���9t�       �	|}Xc�A�'*

loss<f^:���       �	¤}Xc�A�'*

losse��:�|�@       �	%;}Xc�A�'*

loss%R.<�WFr       �	
�}Xc�A�'*

lossD�9��{�       �	Ts}Xc�A�'*

loss��
=ˣ       �	�}Xc�A�'*

loss��S<���6       �	v�}Xc�A�'*

lossF��;o�       �	�D}Xc�A�'*

loss<�5;-�3       �	��}Xc�A�'*

loss��?;�� "       �	Tn}Xc�A�'*

loss��q9�v��       �	#}Xc�A�'*

losshrp;�x�6       �	�}Xc�A�'*

loss�SP:Я;       �	�p}Xc�A�'*

lossm�=	?_�       �	k}Xc�A�'*

loss!��;�f�       �	7�}Xc�A�'*

loss���;r�n       �	$E}Xc�A�'*

loss���<�[��       �	Q�}Xc�A�'*

loss|9;q���       �	�y}Xc�A�'*

loss���;<��j       �	}Xc�A�'*

lossT-;$��H       �	q�}Xc�A�'*

loss��;H_��       �	�S}Xc�A�'*

loss�ó=����       �	O�}Xc�A�'*

losso��<<R)       �	�v}Xc�A�'*

lossS�.=���X       �	}Xc�A�'*

lossO��9G��]       �	{�}Xc�A�'*

lossܧ=f�DV       �	�5}Xc�A�'*

loss�u�<�ٜ�       �	��}Xc�A�'*

loss��<���B       �	�_}Xc�A�'*

lossG<Z��c       �	3�}Xc�A�'*

lossL�;1Ț       �	c�}Xc�A�'*

lossO�=|�N       �	�1}Xc�A�'*

loss�5<em       �	*�}Xc�A�'*

loss�<;B-�I       �	�`}Xc�A�'*

loss��=9�5�       �	/}Xc�A�'*

loss��I;�+¹       �	��}Xc�A�'*

loss�:yQ�       �	�V }Xc�A�'*

loss��<���       �	_� }Xc�A�'*

loss�!;���       �	ӣ!}Xc�A�'*

loss�*2<@��       �	�;"}Xc�A�'*

loss��r:�Q'�       �	Y�"}Xc�A�'*

loss���:?�nX       �	ӆ#}Xc�A�'*

lossc�;�P       �	�H$}Xc�A�'*

loss�\b<T~��       �	-`%}Xc�A�'*

loss�)V;9W��       �	��%}Xc�A�'*

loss��w<�iy9       �	��&}Xc�A�'*

loss�'=*9       �	�>'}Xc�A�'*

loss�c�;
���       �	)�'}Xc�A�'*

loss��;��.�       �	�(}Xc�A�'*

loss���:�L�)       �	3)}Xc�A�'*

loss
��<ӗD        �	ms*}Xc�A�'*

loss�&f=�]�X       �	�
+}Xc�A�'*

loss�/�<և�!       �	��+}Xc�A�'*

loss��=��       �	��,}Xc�A�'*

lossT�;���(       �	A-}Xc�A�'*

loss��=
��       �	��-}Xc�A�'*

loss}�<&:��       �	Wy.}Xc�A�'*

loss��;�%       �	�H/}Xc�A�'*

loss���:k�A�       �	3�/}Xc�A�'*

loss��5;�[=�       �	:w0}Xc�A�'*

lossƺ,<3fU�       �	a51}Xc�A�'*

loss�:{��       �	�1}Xc�A�'*

loss�a�9�+�n       �	�}2}Xc�A�'*

loss[W=w�~       �	/M3}Xc�A�'*

loss��b=kY�       �	�3}Xc�A�'*

loss)�;��v�       �	J|4}Xc�A�'*

loss[N�<���;       �	�5}Xc�A�'*

loss�E�;�6       �	��5}Xc�A�(*

loss8 �9N�       �	φ6}Xc�A�(*

loss�	6<��HZ       �	7}Xc�A�(*

lossC,�<�mu-       �	�7}Xc�A�(*

loss4�9Rw �       �	�o8}Xc�A�(*

losso�=�pwj       �	�9}Xc�A�(*

loss���;{p�-       �	7�9}Xc�A�(*

loss_��<���       �	qW:}Xc�A�(*

lossaS<�[       �	�;}Xc�A�(*

loss�k�;�-4)       �	<}Xc�A�(*

loss(}�=25�#       �	�<}Xc�A�(*

loss���;@3       �	mU=}Xc�A�(*

lossEV�;oope       �	��=}Xc�A�(*

loss���:r	�U       �	C�>}Xc�A�(*

loss%0�:�5       �	�H?}Xc�A�(*

loss6؅<�$�       �	+0@}Xc�A�(*

loss#vi<�\��       �	��@}Xc�A�(*

loss{P=�       �	5}A}Xc�A�(*

loss��<$ t�       �	B}Xc�A�(*

lossx8A=/�ko       �	��B}Xc�A�(*

loss�;���       �	��C}Xc�A�(*

loss���<q��       �	��D}Xc�A�(*

lossHl<�1��       �	�.E}Xc�A�(*

loss��=kz�s       �	TWF}Xc�A�(*

loss��:��2       �	)�F}Xc�A�(*

loss��=w�8       �	��G}Xc�A�(*

loss!�<y�a       �	�H}Xc�A�(*

loss貳<��4       �	p�H}Xc�A�(*

loss�+�<lR�       �	�TI}Xc�A�(*

loss�=�<����       �	4J}Xc�A�(*

loss���<f�=�       �	�)K}Xc�A�(*

loss�P;�0�g       �	�L}Xc�A�(*

loss��!<J5ǡ       �	�&M}Xc�A�(*

lossAY�<�wWP       �	׿M}Xc�A�(*

loss��1=|        �	=bN}Xc�A�(*

loss���<���W       �	z�N}Xc�A�(*

loss��<.C<V       �	+�O}Xc�A�(*

loss���:����       �	OtP}Xc�A�(*

losst>�;�͆       �	�Q}Xc�A�(*

loss��2<h H       �	5�Q}Xc�A�(*

loss�8�;�_U�       �	vR}Xc�A�(*

loss�I�;�Wx�       �	�
S}Xc�A�(*

loss|o�<�l6�       �	��S}Xc�A�(*

loss:��;x��=       �	�4T}Xc�A�(*

loss��<"���       �	:U}Xc�A�(*

lossF�;|�       �	f�U}Xc�A�(*

loss��;��`�       �	!=V}Xc�A�(*

loss�8=���       �	�V}Xc�A�(*

loss��;&^�9       �	�eW}Xc�A�(*

loss���;�t��       �	7�W}Xc�A�(*

lossd"e:��CC       �	àX}Xc�A�(*

loss�>�<X��       �	�4Y}Xc�A�(*

lossF�(:�4�       �	��Y}Xc�A�(*

lossQ��:��bL       �	�fZ}Xc�A�(*

lossc68<F�R       �	K[}Xc�A�(*

loss���;��       �	��[}Xc�A�(*

loss�
[;0��       �	<\}Xc�A�(*

loss3=��       �	o]}Xc�A�(*

loss�m�;b��       �	�]}Xc�A�(*

loss�U;t�8�       �	�B^}Xc�A�(*

loss�H�<��!�       �	��^}Xc�A�(*

loss�8�;�$�       �	�r_}Xc�A�(*

loss��;}���       �	`}Xc�A�(*

loss\C>�_��       �	�`}Xc�A�(*

lossV�3;9�@       �	Ja}Xc�A�(*

loss�G�9��       �	.�a}Xc�A�(*

loss��L<a���       �	V�b}Xc�A�(*

loss۠�<�Ν�       �	B"c}Xc�A�(*

loss�=:��       �	Q�c}Xc�A�(*

loss-��<C.�       �	�^d}Xc�A�(*

lossm�b;�i��       �	
�d}Xc�A�(*

loss��V;�Y�`       �	b�e}Xc�A�(*

lossd0l;�Q)�       �	�5f}Xc�A�(*

lossu_!<�P�n       �	��f}Xc�A�(*

loss�;膎�       �	r�g}Xc�A�(*

loss�S<%���       �	6=h}Xc�A�(*

losss*�;m�6�       �	N�h}Xc�A�(*

loss�Pt:�c�       �	�hi}Xc�A�(*

loss��(;p5�       �		�i}Xc�A�(*

loss =)<U�q�       �	�j}Xc�A�(*

lossq��;�?�       �	5k}Xc�A�(*

loss��}:qˌ       �	�l}Xc�A�(*

lossK�<�-�        �	��l}Xc�A�(*

loss�w:<F�       �	�em}Xc�A�(*

loss���<��0       �	3�m}Xc�A�(*

loss!�<ڑ�       �	��n}Xc�A�(*

loss�><A�f�       �	9'o}Xc�A�(*

loss-P;3       �	J�o}Xc�A�(*

lossX��;訚�       �	Ҍp}Xc�A�(*

lossi	�<��g�       �	�+q}Xc�A�(*

loss))�;��J�       �	��q}Xc�A�(*

loss-��<��       �	��r}Xc�A�(*

loss)��<�i�       �	�ms}Xc�A�(*

loss�S�;�6��       �	dt}Xc�A�(*

loss�@�;��s$       �	Υt}Xc�A�(*

loss���<R�,{       �	�
v}Xc�A�(*

loss�[�;e���       �	�w}Xc�A�(*

loss��1;��	       �	V�w}Xc�A�(*

loss���:�a��       �	h[x}Xc�A�(*

loss��[=���#       �	��x}Xc�A�(*

loss2�]<Y3�       �	+�y}Xc�A�(*

lossEܺ;¸[       �	:z}Xc�A�(*

loss�x�<G�>�       �	��z}Xc�A�(*

lossi�;b��3       �	�n{}Xc�A�(*

loss��<7�h4       �	�|}Xc�A�(*

loss���;�\��       �	0�|}Xc�A�(*

lossm]�;-��-       �	�|}}Xc�A�(*

loss�8U<�՞�       �	#~}Xc�A�(*

loss�
�;��H       �	-�~}Xc�A�(*

loss�% ;-�       �	U}Xc�A�(*

loss!�;Qh       �	��}Xc�A�(*

loss�8G=�&�9       �	���}Xc�A�(*

lossyo;+�*�       �	�'�}Xc�A�(*

loss�|<�tu�       �	{��}Xc�A�(*

loss6,;9O1�       �	�^�}Xc�A�(*

loss ܣ<�dN�       �	���}Xc�A�(*

loss�L�<�2�3       �	���}Xc�A�(*

loss[C�;n܍�       �	!"�}Xc�A�(*

loss��;&0       �	���}Xc�A�(*

lossw<�:ؽ�       �	F[�}Xc�A�(*

loss�p�:���b       �	���}Xc�A�(*

loss��^=��3       �	���}Xc�A�(*

loss)<e1�       �	o+�}Xc�A�(*

loss)p�:�i0H       �	CŇ}Xc�A�(*

loss/��<�:L       �	�[�}Xc�A�(*

loss!:�|�       �	~��}Xc�A�(*

lossb�;O��       �	Q�}Xc�A�(*

loss�:A<N��       �	;��}Xc�A�(*

loss���<��       �	�֋}Xc�A�)*

loss��<�`�$       �	�Ō}Xc�A�)*

lossN�Q;�i�f       �	|a�}Xc�A�)*

loss=(�S       �	O�}Xc�A�)*

lossD�m:SǤ       �	�?�}Xc�A�)*

loss�z�:ʁ�       �	��}Xc�A�)*

lossp�:kӤ       �	���}Xc�A�)*

loss�7;��j       �	�&�}Xc�A�)*

loss�(�<���8       �	f��}Xc�A�)*

loss9�:��q3       �	Q��}Xc�A�)*

loss�Ρ:Ǫ��       �	��}Xc�A�)*

lossʐ�;B�q       �	Ի�}Xc�A�)*

loss!��<Ւ�h       �	dZ�}Xc�A�)*

loss��4:��w       �	L�}Xc�A�)*

loss�ڣ9>�t       �	���}Xc�A�)*

loss�N:���       �	�]�}Xc�A�)*

loss���="C�       �	��}Xc�A�)*

loss7|�9
���       �	ӣ�}Xc�A�)*

loss�L�;�@D       �	.:�}Xc�A�)*

loss�KS:��       �	Rә}Xc�A�)*

loss:�F;Q��"       �	�}Xc�A�)*

loss�n;���\       �	�%�}Xc�A�)*

loss��8N��=       �	���}Xc�A�)*

loss�\<s�α       �	T�}Xc�A�)*

loss�O�;�~��       �	 �}Xc�A�)*

loss�:��       �	A��}Xc�A�)*

loss��/=�c�       �	}?�}Xc�A�)*

loss���:��~       �	�}Xc�A�)*

loss���;�X<       �	2Y�}Xc�A�)*

lossӆ2;4�ǧ       �	���}Xc�A�)*

loss�x<����       �	a��}Xc�A�)*

loss�E=�;$       �	I֢}Xc�A�)*

loss�Hr;X�$       �	Tt�}Xc�A�)*

loss��>C���       �	��}Xc�A�)*

loss:�<C��       �	;��}Xc�A�)*

lossDĊ9���       �	�>�}Xc�A�)*

lossB�;am��       �	�ߥ}Xc�A�)*

loss͏={!Ϝ       �	�u�}Xc�A�)*

loss70c;���       �	X�}Xc�A�)*

loss8-;�G�'       �	5��}Xc�A�)*

lossؔ;�'&�       �	&S�}Xc�A�)*

lossڗ_9Ե�       �	���}Xc�A�)*

loss^<�C       �	���}Xc�A�)*

loss��o;��w       �	��}Xc�A�)*

loss�
=O{0�       �	3��}Xc�A�)*

loss��:���       �	X�}Xc�A�)*

loss�J�<:o�       �	��}Xc�A�)*

lossH]�=�ci       �	ٕ�}Xc�A�)*

loss?��=��0n       �	�/�}Xc�A�)*

loss�,�;i�Z       �	ѭ}Xc�A�)*

lossm\	<��       �	�r�}Xc�A�)*

lossE�6=�X�n       �	'�}Xc�A�)*

loss�d<S~��       �	!��}Xc�A�)*

loss/��;w/�.       �	oF�}Xc�A�)*

lossrR
:�'       �	��}Xc�A�)*

loss��{;�2~�       �	)w�}Xc�A�)*

loss�u(<27��       �	<�}Xc�A�)*

loss�m�<j�\       �	S��}Xc�A�)*

lossJq�<�0��       �	�O�}Xc�A�)*

loss��2;�Ƥ<       �	=�}Xc�A�)*

loss�Q<S�(       �	���}Xc�A�)*

loss��=�mI/       �	NG�}Xc�A�)*

loss�=�;3�H�       �	�}Xc�A�)*

loss�T�;�ٌ       �	��}Xc�A�)*

loss���=YX>:       �	+5�}Xc�A�)*

loss�-s;U��       �	�θ}Xc�A�)*

lossc��<�A�       �	[}�}Xc�A�)*

loss��y=���       �	#�}Xc�A�)*

loss4�S<���W       �	��}Xc�A�)*

loss���=_X�6       �	�C�}Xc�A�)*

loss�;�)3�       �	�ۻ}Xc�A�)*

loss���;��       �	Bv�}Xc�A�)*

lossB2<��       �	,�}Xc�A�)*

loss�8;�h��       �	ࡽ}Xc�A�)*

loss���:���       �	?�}Xc�A�)*

loss�xs9c���       �	BѾ}Xc�A�)*

lossnp�<�c�       �	�f�}Xc�A�)*

loss7<5��       �	���}Xc�A�)*

loss�ǡ<3�rm       �	d��}Xc�A�)*

loss4[<�$"�       �	$�}Xc�A�)*

loss6;�_.!       �	���}Xc�A�)*

lossZۚ<��       �	�T�}Xc�A�)*

losshk<fH�V       �	^��}Xc�A�)*

loss��:��'       �	~��}Xc�A�)*

loss��:+h�
       �	��}Xc�A�)*

loss}�8���6       �	B��}Xc�A�)*

lossV��:��Md       �	^H�}Xc�A�)*

loss��:�*��       �	��}Xc�A�)*

lossh�;��V�       �	�z�}Xc�A�)*

loss�=DE�       �	&�}Xc�A�)*

lossE��:K^�L       �	��}Xc�A�)*

loss��:�{|�       �	�W�}Xc�A�)*

lossq�:�]�7       �	���}Xc�A�)*

loss��E9���)       �	�C�}Xc�A�)*

loss2�q8z���       �	^��}Xc�A�)*

loss.�_9���7       �	{�}Xc�A�)*

loss_��<�O       �	��}Xc�A�)*

loss��K=��?�       �	֪�}Xc�A�)*

loss�n�;����       �	�E�}Xc�A�)*

loss��9nM�       �	Q��}Xc�A�)*

lossDݱ:g��       �	3��}Xc�A�)*

loss���=��B�       �	2=�}Xc�A�)*

loss�vS:rT+�       �	�m�}Xc�A�)*

lossT;�=�I�n       �	
�}Xc�A�)*

loss� _<��kI       �	���}Xc�A�)*

losss5�<�'�"       �	E�}Xc�A�)*

losse�q:�܆�       �	��}Xc�A�)*

loss�;m:bvl�       �	~t�}Xc�A�)*

loss��>�uZ       �	0�}Xc�A�)*

lossR1=���       �	6��}Xc�A�)*

loss���:e�?       �	 `�}Xc�A�)*

loss�c<����       �	���}Xc�A�)*

loss���<(3�C       �	���}Xc�A�)*

loss��=?_֦       �	l]�}Xc�A�)*

loss��=���b       �	]��}Xc�A�)*

loss��;;O�ud       �	���}Xc�A�)*

loss}p�=�h�       �	�5�}Xc�A�)*

loss�iQ=�q3       �	F��}Xc�A�)*

losss:=���P       �	){�}Xc�A�)*

lossd� =]�ʆ       �	P�}Xc�A�)*

loss�@�=�       �	���}Xc�A�)*

loss���:zM       �	OZ�}Xc�A�)*

lossz�-<�I�h       �	H��}Xc�A�)*

lossȫ1<}0       �	��}Xc�A�)*

lossV�=.�Gu       �	�?�}Xc�A�)*

lossL;����       �	A��}Xc�A�)*

loss���:��)       �	 p�}Xc�A�)*

loss�F�:����       �	��}Xc�A�)*

loss�<G��%       �	v��}Xc�A�**

loss���:�`�       �	�h�}Xc�A�**

loss3��=S��l       �	��}Xc�A�**

lossM�;8��N       �	M��}Xc�A�**

lossS��:��h}       �	�M�}Xc�A�**

lossF;6��j       �	���}Xc�A�**

loss�q�;Q@�       �	��}Xc�A�**

loss��;�=�       �	u"�}Xc�A�**

loss��<�o�S       �	Z��}Xc�A�**

loss�G/<9�T�       �	���}Xc�A�**

loss^�;�?��       �	���}Xc�A�**

lossۢ�<<��J       �	�#�}Xc�A�**

loss�8M<x�"�       �	��}Xc�A�**

loss�é<H~_�       �	�`�}Xc�A�**

loss�;q�$       �	S�}Xc�A�**

loss��\;���7       �	��}Xc�A�**

loss[�B=�g}       �	�C�}Xc�A�**

lossQ��;�&{       �	���}Xc�A�**

loss�H<\�Ց       �	{��}Xc�A�**

loss��<*Q       �	�#�}Xc�A�**

loss�]6;��       �	ϼ�}Xc�A�**

loss�C;�3�       �	iV�}Xc�A�**

lossHI�;-i��       �	���}Xc�A�**

lossn+j;� �       �	ޑ�}Xc�A�**

loss߫;��q       �	Z-�}Xc�A�**

loss7t�;����       �	��~Xc�A�**

loss�J�;�S��       �	;�~Xc�A�**

lossc�=9�I�       �	(~Xc�A�**

loss_�<�2��       �	�~Xc�A�**

lossD��<q�w�       �	�_	~Xc�A�**

lossx��;ĆHb       �	��	~Xc�A�**

loss�g;�9�       �	��
~Xc�A�**

lossP�;@bB�       �	�X~Xc�A�**

loss�ʑ; �5�       �	c(~Xc�A�**

lossj��;F]�~       �	��~Xc�A�**

loss�T;���       �	A�~Xc�A�**

lossߠ�<t��       �	g_~Xc�A�**

loss�ډ<��	J       �	~Xc�A�**

loss͟�<����       �	֏~Xc�A�**

loss�A�<�O5c       �	wJ~Xc�A�**

loss��=�d�U       �	�~Xc�A�**

loss*ˎ9l���       �	1�~Xc�A�**

loss��6:AJ�       �	r1~Xc�A�**

loss���:qcF�       �	?�~Xc�A�**

lossJΆ=L��*       �	�{~Xc�A�**

loss��:u��}       �	>&~Xc�A�**

losst$<*O       �	��~Xc�A�**

loss��|;ڙ��       �	�~~Xc�A�**

loss��<=`���       �	T~Xc�A�**

loss���9�,��       �	�~Xc�A�**

lossI3�<�?��       �	�O~Xc�A�**

loss�/<oX       �	q�~Xc�A�**

lossͬ0;�:�o       �	Z�~Xc�A�**

loss@_�<�&0       �	�~Xc�A�**

loss�<���       �	��~Xc�A�**

loss�n�;o�c       �	��~Xc�A�**

loss,J9����       �	�5~Xc�A�**

loss8`�:a��       �	��~Xc�A�**

losslEG<ݼ��       �	k~Xc�A�**

lossl�;���;       �	pB~Xc�A�**

loss)�_< ;�       �	�~Xc�A�**

loss���:{z~�       �	��~Xc�A�**

loss*�<H\�       �	�1 ~Xc�A�**

loss�Vq= [       �	�� ~Xc�A�**

loss�#�<�V�       �	yu!~Xc�A�**

lossCE�:��F       �	H"~Xc�A�**

loss�SV;���       �	��"~Xc�A�**

loss�<�sT�       �	`t#~Xc�A�**

loss��;�؉�       �	X$~Xc�A�**

lossq��;�+o�       �	v�$~Xc�A�**

loss.��;�|�       �	s�%~Xc�A�**

loss��;e�f       �	w&~Xc�A�**

loss �_;>��9       �	e'~Xc�A�**

loss���9κ��       �	 �'~Xc�A�**

lossO�,:֙�       �	�h(~Xc�A�**

loss�J:^*�       �	�)~Xc�A�**

loss|@�=��       �	��)~Xc�A�**

lossQ:Y�v       �	.V*~Xc�A�**

loss�9�<$+��       �	��*~Xc�A�**

losss�o=ô��       �	W�+~Xc�A�**

loss?y
:�sXu       �	L6,~Xc�A�**

loss���:_@ǰ       �	31-~Xc�A�**

lossә	;.��       �	l�-~Xc�A�**

loss�I;w���       �	h.~Xc�A�**

loss��Y=��       �	�/~Xc�A�**

loss.�=�w�Z       �	Ę/~Xc�A�**

loss�U<GF��       �	�K0~Xc�A�**

lossܞ�;�jC       �	��0~Xc�A�**

loss�^<�$L       �	�1~Xc�A�**

lossq��;��       �	l�2~Xc�A�**

loss8��:U��       �	�I3~Xc�A�**

loss
ڽ;h���       �	��3~Xc�A�**

loss��j;�]�k       �	��4~Xc�A�**

loss��=�^e�       �	��5~Xc�A�**

lossA�;@���       �	I�6~Xc�A�**

loss)2(:j���       �	�T7~Xc�A�**

loss���:}��       �	��7~Xc�A�**

loss=�	<x�       �	o9~Xc�A�**

lossTz=< ���       �	�9~Xc�A�**

loss���;].t       �	~V:~Xc�A�**

loss܇�:�։H       �	�;~Xc�A�**

loss��|;��        �	ü;~Xc�A�**

loss��:X��B       �	<�<~Xc�A�**

loss�::�I*�       �	�v=~Xc�A�**

loss�0=��       �	e>~Xc�A�**

loss�d<��9�       �	��>~Xc�A�**

lossP�;<=Z	       �	�L?~Xc�A�**

loss�E�:�=�       �	i�?~Xc�A�**

loss� 6<�to�       �	��@~Xc�A�**

loss�w<����       �	�(A~Xc�A�**

lossI�<W�       �	5�A~Xc�A�**

loss��;��b       �	iB~Xc�A�**

lossz��;��       �	X�B~Xc�A�**

loss�g�;|5�       �	��C~Xc�A�**

loss���:�x��       �	D0D~Xc�A�**

loss0;<�VB       �	��D~Xc�A�**

loss?��=���@       �	[E~Xc�A�**

loss��r:yD��       �	��E~Xc�A�**

loss:�1<��!       �	`�F~Xc�A�**

lossz(<�i��       �	p(G~Xc�A�**

loss"�;`�l       �	8�G~Xc�A�**

loss��=V�       �	}XH~Xc�A�**

loss�(;�Y�       �	Z�H~Xc�A�**

loss��<{�
�       �	��I~Xc�A�**

loss;�;|q>�       �	��J~Xc�A�**

lossP�:(�P       �	I+K~Xc�A�**

loss;n���       �	��K~Xc�A�**

loss�Dp<����       �	|
M~Xc�A�**

loss�h�:RK�       �	�M~Xc�A�+*

lossJ߫:��3       �	�iN~Xc�A�+*

loss�-:�h =       �	O~Xc�A�+*

loss3��<�Z�       �	s�O~Xc�A�+*

loss/��=�@�%       �	pP~Xc�A�+*

lossnǑ;�C�       �	�Q~Xc�A�+*

loss�4/:��u�       �	�Q~Xc�A�+*

losshz�;�K�       �	LTR~Xc�A�+*

lossZ�^;����       �	U�R~Xc�A�+*

loss]� =�;o�       �	��S~Xc�A�+*

loss��;C	       �	Q3T~Xc�A�+*

loss�Q<�W@       �	��T~Xc�A�+*

loss�Fi9s}%�       �	[{U~Xc�A�+*

loss?��;�2�W       �	V~Xc�A�+*

loss���=H%�       �	7�V~Xc�A�+*

loss�)W<h�a�       �	�AW~Xc�A�+*

loss��S=e4�y       �	�X~Xc�A�+*

losshy�;Q��        �	��X~Xc�A�+*

loss`~�<�;�       �	�aY~Xc�A�+*

losso��;��H�       �	��Y~Xc�A�+*

lossf��<�u�P       �	�[~Xc�A�+*

loss-q;��3�       �	�[~Xc�A�+*

loss���;�H�       �	q\~Xc�A�+*

loss��;�@q�       �	j]~Xc�A�+*

loss$/:W�c�       �	;�]~Xc�A�+*

lossQ�%<�+76       �	`v^~Xc�A�+*

loss��):K|K       �	�_~Xc�A�+*

loss�ԑ<�,��       �	j�_~Xc�A�+*

lossc��<~;��       �	dX`~Xc�A�+*

loss?��;���       �	��`~Xc�A�+*

lossZ�3<�t#       �	'�a~Xc�A�+*

loss)�q;�C�       �	HNb~Xc�A�+*

lossc@�;�^!0       �	��b~Xc�A�+*

loss?2�:-&�?       �	��c~Xc�A�+*

loss*�J:4x0L       �	F[d~Xc�A�+*

loss�{�;��S       �	��d~Xc�A�+*

loss�D6<���r       �	��e~Xc�A�+*

loss��;��ĕ       �	.f~Xc�A�+*

loss(�<��       �	��f~Xc�A�+*

loss��7<��N       �	�vg~Xc�A�+*

loss
��:�j       �	Zh~Xc�A�+*

loss�S�=4�c�       �	֪h~Xc�A�+*

lossn(�:��}5       �	�Bi~Xc�A�+*

loss���<�k�e       �	��i~Xc�A�+*

loss���;*�
       �	\wj~Xc�A�+*

loss�:�T.       �	�m~Xc�A�+*

loss��X=��R       �	F�m~Xc�A�+*

loss_c�;ۑg�       �	ɒn~Xc�A�+*

lossp-�=T�2�       �	�)o~Xc�A�+*

loss��1<մ��       �	��o~Xc�A�+*

loss�ү<�Z�J       �	��p~Xc�A�+*

loss�3�;����       �	��q~Xc�A�+*

lossDZ�8%"��       �	n�r~Xc�A�+*

loss@�9-�@�       �	Ps~Xc�A�+*

loss	�2=�P       �	�
t~Xc�A�+*

loss��=m���       �	��t~Xc�A�+*

loss�ǲ:p��       �	�Qu~Xc�A�+*

loss���<"J       �	��u~Xc�A�+*

lossI��;�A`d       �	�v~Xc�A�+*

loss!�	=���        �	nOw~Xc�A�+*

loss��:�Ůe       �	�x~Xc�A�+*

loss�s=��a9       �	+Oy~Xc�A�+*

loss�H=n=��       �	�y~Xc�A�+*

losst,�:�t9       �	��z~Xc�A�+*

loss��5=���       �	�t{~Xc�A�+*

lossER�<���       �	Q|~Xc�A�+*

lossJ�<���       �	`�|~Xc�A�+*

loss�|;���       �	I�}~Xc�A�+*

loss&�;���       �	JD~~Xc�A�+*

loss��1<��O       �	a�~~Xc�A�+*

loss�x�<����       �	�z~Xc�A�+*

loss60�;-�z�       �	��~Xc�A�+*

loss��:9���       �	1��~Xc�A�+*

loss�|<�=B�       �	rQ�~Xc�A�+*

loss)9�=hxE       �	��~Xc�A�+*

loss[Q;#rm       �	�~Xc�A�+*

loss��;�_��       �	<�~Xc�A�+*

loss\�:���:       �	�߃~Xc�A�+*

loss�_*:%��P       �	�x�~Xc�A�+*

loss:�2:�
��       �	� �~Xc�A�+*

loss���:�,��       �	鹅~Xc�A�+*

loss�`
=]��       �	6X�~Xc�A�+*

lossѴ�:WI�-       �	��~Xc�A�+*

losso�A<�S�       �	֐�~Xc�A�+*

loss��Z="׃       �	�8�~Xc�A�+*

loss���;{HA�       �	���~Xc�A�+*

loss��;��_�       �	9%�~Xc�A�+*

loss�)<"�4�       �	��~Xc�A�+*

lossL��=0�f       �	DP�~Xc�A�+*

lossLs:�LNc       �	��~Xc�A�+*

loss���;��M       �	�B�~Xc�A�+*

loss�Kf:հ��       �	�~Xc�A�+*

loss��:�ܟ       �	Kȏ~Xc�A�+*

loss;c;����       �	/��~Xc�A�+*

loss_ ;��i�       �	�`�~Xc�A�+*

loss���;73%�       �	S$�~Xc�A�+*

loss�qd9�VL`       �	�Ē~Xc�A�+*

loss�U�:P�ͪ       �	�a�~Xc�A�+*

lossCA:���       �	�A�~Xc�A�+*

lossm	;Է�6       �	e�~Xc�A�+*

losst�<��0       �	|�~Xc�A�+*

lossꋽ<��       �	a�~Xc�A�+*

lossEa8:[C9X       �	y��~Xc�A�+*

lossJ(<Ku�       �	EK�~Xc�A�+*

loss�S4;���       �	7�~Xc�A�+*

loss���;��/       �	���~Xc�A�+*

lossR<䟡�       �	�I�~Xc�A�+*

loss�7;o���       �	C�~Xc�A�+*

lossX?�;í�       �	�x�~Xc�A�+*

loss�~:e��       �	�7�~Xc�A�+*

loss��<;&�t       �	�ț~Xc�A�+*

loss
I�;��{       �	�]�~Xc�A�+*

loss�E=x�s�       �	���~Xc�A�+*

loss=��=[ԧ       �	��~Xc�A�+*

lossTy�=�d�       �	"�~Xc�A�+*

loss���<�	�K       �	o��~Xc�A�+*

loss�	�;��Wi       �	�K�~Xc�A�+*

loss�8s<5��       �	���~Xc�A�+*

loss,��<<e��       �	ɏ�~Xc�A�+*

lossV�<O��       �	L4�~Xc�A�+*

lossF��<���       �	�ԡ~Xc�A�+*

loss�@<�+       �	΋�~Xc�A�+*

losso��<%'��       �	?9�~Xc�A�+*

lossX��;��       �	�~Xc�A�+*

loss���:P��g       �	͑�~Xc�A�+*

loss$r9�I       �	<�~Xc�A�+*

lossw=���/       �	�~Xc�A�+*

loss�65:��^�       �	l��~Xc�A�+*

loss���<�W��       �	0�~Xc�A�,*

loss�9;l��       �	�ʧ~Xc�A�,*

loss��;�(�       �	t�~Xc�A�,*

loss�<�;�R��       �	��~Xc�A�,*

loss_@<�yV       �	m��~Xc�A�,*

loss���=��L       �	�F�~Xc�A�,*

loss�=%L�       �	nߪ~Xc�A�,*

loss���<&��       �	5|�~Xc�A�,*

loss�&�:��       �	u�~Xc�A�,*

lossma�;��)       �	г�~Xc�A�,*

lossA8`<2�r)       �	�W�~Xc�A�,*

lossV!�:���2       �	�P�~Xc�A�,*

loss��Q=:m��       �	Z��~Xc�A�,*

loss,&�;Y+ }       �	�~Xc�A�,*

loss��:fHxF       �	�L�~Xc�A�,*

loss��;T�u       �	���~Xc�A�,*

loss��\<{n       �	���~Xc�A�,*

loss%�;".�5       �	�W�~Xc�A�,*

loss[��;�k��       �	 �~Xc�A�,*

lossew3;���I       �	ʧ�~Xc�A�,*

loss�bt<�x<       �	�R�~Xc�A�,*

lossW9>;88O       �	�"�~Xc�A�,*

loss���9�{.       �	ѵ~Xc�A�,*

loss��<��w       �	�z�~Xc�A�,*

loss/\�:�Ⱦ�       �	�%�~Xc�A�,*

loss�P9��5;       �	�ҷ~Xc�A�,*

loss�u;�Ǜ�       �	���~Xc�A�,*

loss#9�[-�       �	e�~Xc�A�,*

loss��<q��       �	<�~Xc�A�,*

loss��Q<.&�       �	~�~Xc�A�,*

loss]0<��l�       �	@��~Xc�A�,*

loss��=�s�       �	V)�~Xc�A�,*

loss��;o�(D       �	P¼~Xc�A�,*

loss�2=}8y       �	���~Xc�A�,*

lossx��:W;+�       �	�&�~Xc�A�,*

loss��<Å�       �	Y¾~Xc�A�,*

loss���;��Å       �	Vd�~Xc�A�,*

loss�-<WD�       �	��~Xc�A�,*

loss�	�<�)��       �	���~Xc�A�,*

loss��:�;s�       �	�J�~Xc�A�,*

lossSa�;)X�       �	z��~Xc�A�,*

loss`ۭ:��       �	|�~Xc�A�,*

lossӥ:;��       �	/�~Xc�A�,*

lossM�G=�Q��       �		��~Xc�A�,*

lossLn	=�Cڟ       �	P��~Xc�A�,*

loss�|�:4���       �	�Q�~Xc�A�,*

loss)@�:Ģ�[       �	Ad�~Xc�A�,*

lossV��<U���       �	ob�~Xc�A�,*

loss�a<����       �	���~Xc�A�,*

loss�@:V��L       �	1��~Xc�A�,*

loss{[�:�*       �	UL�~Xc�A�,*

loss�<�7y�       �	��~Xc�A�,*

loss�N�<�ew       �	��~Xc�A�,*

loss[�,=���       �	 p�~Xc�A�,*

loss��=��       �	�`�~Xc�A�,*

loss6�w;���4       �	m��~Xc�A�,*

loss���;��       �	ٴ�~Xc�A�,*

loss�%�:r-Ux       �	�O�~Xc�A�,*

lossE��9i|       �	���~Xc�A�,*

lossh�];؏7�       �	���~Xc�A�,*

loss�o<%���       �	�6�~Xc�A�,*

loss��99�x�w       �	V��~Xc�A�,*

loss���:�`�       �	Sw�~Xc�A�,*

loss|�<�c�       �	f�~Xc�A�,*

losss�.<�Wv�       �	��~Xc�A�,*

loss���<����       �	ND�~Xc�A�,*

loss�+�<�gk       �	r��~Xc�A�,*

loss0��;� �^       �	��~Xc�A�,*

lossxi=���       �	�&�~Xc�A�,*

lossC;+;�vn�       �	���~Xc�A�,*

lossψ9:�yE�       �	Yk�~Xc�A�,*

loss�0==��       �	]�~Xc�A�,*

loss�7,=�ߚ3       �	g��~Xc�A�,*

loss૜<�Kh       �	�O�~Xc�A�,*

lossoo!<�/�       �	y�~Xc�A�,*

loss��::46&�       �	,��~Xc�A�,*

loss[��:��       �	{1�~Xc�A�,*

loss�ݎ<�ur       �	��~Xc�A�,*

lossZ�:��{       �	�i�~Xc�A�,*

lossG <흯;       �	��~Xc�A�,*

loss#��;%8��       �	���~Xc�A�,*

lossԢ�<�n��       �	:\�~Xc�A�,*

loss_M�:f��x       �	Y��~Xc�A�,*

loss�r=B> �       �	5��~Xc�A�,*

lossF��=w'P       �	IH�~Xc�A�,*

losst�+;Fý�       �	��~Xc�A�,*

loss��<A�P       �	���~Xc�A�,*

loss8F�<�^�       �	�(�~Xc�A�,*

lossep9<�i��       �	���~Xc�A�,*

loss��<��e	       �	Ul�~Xc�A�,*

loss���<[RF       �	�	�~Xc�A�,*

lossWY�<�8i       �	"��~Xc�A�,*

loss�S�9^��G       �	�_�~Xc�A�,*

loss��<yC�       �	h�~Xc�A�,*

loss<����       �	���~Xc�A�,*

lossq]?;�g�       �	�;�~Xc�A�,*

loss�,O:��%Q       �	���~Xc�A�,*

losst��;G*8s       �	���~Xc�A�,*

lossJ��<�^�        �	g*�~Xc�A�,*

loss�<���       �	���~Xc�A�,*

loss�5�;w�       �	�[�~Xc�A�,*

loss�@`;vxb       �	|��~Xc�A�,*

loss6Jg:)��7       �	֌�~Xc�A�,*

loss�'�<��Ľ       �	�,�~Xc�A�,*

loss�~ ;=�)       �	c��~Xc�A�,*

loss�.�;�EA�       �	u�~Xc�A�,*

loss�Ǝ<�~       �	6�~Xc�A�,*

loss}%<(��v       �	m��~Xc�A�,*

lossM��92�Y`       �	6t�~Xc�A�,*

loss�in<�8�@       �	��~Xc�A�,*

loss�": ��       �	��~Xc�A�,*

loss�D�:�I|�       �	�k�~Xc�A�,*

loss�4=�x��       �	��~Xc�A�,*

losst�<���       �	L��~Xc�A�,*

loss�<�$��       �	NC�~Xc�A�,*

loss�p�<!y�X       �	���~Xc�A�,*

loss_:=�1e       �	��~Xc�A�,*

loss�V;�eC�       �	�2�~Xc�A�,*

lossO�<F�[       �	���~Xc�A�,*

loss�9,]       �	}�~Xc�A�,*

loss3��8���       �	��~Xc�A�,*

loss�T<Yۢ       �	{��~Xc�A�,*

loss �b:�>~^       �	�[�~Xc�A�,*

loss�=��۩       �	��~Xc�A�,*

loss�L~:��W�       �	g��~Xc�A�,*

lossl|�:v�       �	�4�~Xc�A�,*

lossL��;(��       �	���~Xc�A�,*

loss�s�<�tzF       �	�q�~Xc�A�,*

lossҖk8��\       �	a�~Xc�A�-*

loss�9�<��)       �	<��~Xc�A�-*

lossa��<��       �	~T�~Xc�A�-*

loss���96(w�       �	���~Xc�A�-*

loss�?�;�!�~       �	Ӆ�~Xc�A�-*

loss�~=9�N	       �	��~Xc�A�-*

loss7;;G��       �	���~Xc�A�-*

lossMy�<9):       �	�H�~Xc�A�-*

loss�A<:��:       �	���~Xc�A�-*

loss�9�>��       �	K� Xc�A�-*

lossmP;\�i       �	�9Xc�A�-*

lossK�9��G�       �	Q�Xc�A�-*

loss��S:`L=       �	�|Xc�A�-*

loss�b�;�H/�       �	�Xc�A�-*

loss 	'<eNϝ       �	l�Xc�A�-*

lossh2�<F�i       �	(�Xc�A�-*

loss�6�:F�D       �	v�Xc�A�-*

loss��=N�%�       �	"PXc�A�-*

loss� �<#�Vl       �	��Xc�A�-*

loss��:��o�       �	��Xc�A�-*

loss�Q<k`Ss       �	�Xc�A�-*

loss��<%       �	��Xc�A�-*

losstj]=���       �	�	Xc�A�-*

loss:M�=��       �	u"
Xc�A�-*

loss���<D�^6       �	�
Xc�A�-*

loss}n�:��?�       �	YlXc�A�-*

loss�4"<��K(       �	QXc�A�-*

lossad�9�s�       �	�Xc�A�-*

loss�<Sn�       �	υXc�A�-*

lossL6-<�N�D       �	�4Xc�A�-*

loss�'�:�s!m       �	T�Xc�A�-*

lossx�5<'0{k       �	d�Xc�A�-*

loss��:�CX�       �	 BXc�A�-*

loss%1�:5�\z       �	�Xc�A�-*

loss%1�;�s�       �	X�Xc�A�-*

loss�r�<��5�       �	(GXc�A�-*

loss1 I=̃�       �	9�Xc�A�-*

loss4��8�D�U       �	��Xc�A�-*

loss���=s��       �	�?Xc�A�-*

loss��:$=�X       �	��Xc�A�-*

loss�"=4�$�       �	t�Xc�A�-*

lossf~�:P�q       �	�>Xc�A�-*

lossL��;�W]i       �	��Xc�A�-*

loss��<�Z��       �	��Xc�A�-*

loss6�^;{��       �	�YXc�A�-*

loss�{�;"G�       �	�Xc�A�-*

loss��d<�*-       �	��Xc�A�-*

loss��U<�օ       �	=`Xc�A�-*

lossL4�<�l|�       �	OXc�A�-*

lossv;�>��       �	��Xc�A�-*

loss���<�u��       �	�]Xc�A�-*

loss��G<�*0       �	.Xc�A�-*

loss�5=�J�(       �	�Xc�A�-*

loss�z<B��9       �	�YXc�A�-*

loss��S=�P�       �	
Xc�A�-*

loss�=���+       �	��Xc�A�-*

lossG;�"��       �	!v Xc�A�-*

loss���9/az+       �	=*!Xc�A�-*

loss�;ư*�       �	��!Xc�A�-*

loss�w�:aŚ       �	,"Xc�A�-*

loss�m<�)��       �	J&#Xc�A�-*

loss���:.��       �	W�#Xc�A�-*

loss��<`d�       �	�p$Xc�A�-*

loss��&;��(       �	#%Xc�A�-*

loss�==Jv�       �	��%Xc�A�-*

loss��::~V9%       �	J{&Xc�A�-*

loss��<:���       �	5('Xc�A�-*

loss�2!<.���       �	��'Xc�A�-*

lossok<C�W�       �	�o(Xc�A�-*

loss���;^��K       �	�)Xc�A�-*

loss��;l���       �	��)Xc�A�-*

lossX��<���       �	d*Xc�A�-*

lossj��;[���       �	+Xc�A�-*

loss���<J���       �	0�+Xc�A�-*

loss�b;."ݜ       �	�5,Xc�A�-*

loss��M<n�K       �	I�,Xc�A�-*

loss�#�= ���       �	�}-Xc�A�-*

loss�l�<� �       �	�.Xc�A�-*

loss2!y;�e:       �	��.Xc�A�-*

loss��8:�ay+       �	(H/Xc�A�-*

loss/�A<��-       �	~�/Xc�A�-*

loss�ƙ;Ses       �	�0Xc�A�-*

loss���;{�O�       �	x%1Xc�A�-*

loss�q]=��-       �	��1Xc�A�-*

loss�9�<=>�       �	�\2Xc�A�-*

loss!L: H�       �	Z3Xc�A�-*

lossi=��B       �	ٵ3Xc�A�-*

loss�=i�c�       �	YP4Xc�A�-*

loss��:<��6       �	��4Xc�A�-*

lossd>C<t�y�       �	k�5Xc�A�-*

loss[r�;#5�
       �	K<6Xc�A�-*

loss�M�=B��/       �	��6Xc�A�-*

loss�<���D       �	�~7Xc�A�-*

loss��<��~�       �	�+8Xc�A�-*

loss��}:��       �	(�8Xc�A�-*

loss�=�/,�       �	Ϡ9Xc�A�-*

loss�$;(M�z       �	5D:Xc�A�-*

lossCՅ;�L�j       �	6�:Xc�A�-*

loss�=?���       �	��;Xc�A�-*

loss��:�j��       �	"4<Xc�A�-*

loss�.
=He��       �	��<Xc�A�-*

loss�6;M�d�       �	l=Xc�A�-*

loss�?�;{,:^       �	�>Xc�A�-*

loss!�?;�.       �	�>Xc�A�-*

loss��=���       �	�>?Xc�A�-*

lossv��;$��       �	P�?Xc�A�-*

loss�9K<�P�       �	@Xc�A�-*

loss�-=�(X�       �	��AXc�A�-*

loss�9S;;�u�       �	vnBXc�A�-*

loss�!<^A�b       �	�CXc�A�-*

loss���9W}S       �	k�CXc�A�-*

loss��a=���       �	�\DXc�A�-*

lossO�H;����       �	�EXc�A�-*

lossT<;�ha       �	ߨEXc�A�-*

loss%�<F�       �	*�FXc�A�-*

lossS�y:	�.�       �	p&GXc�A�-*

loss�n;�B�       �	��GXc�A�-*

loss�N�:�!E�       �	ZgHXc�A�-*

loss�]`<(���       �	�	IXc�A�-*

loss&�8��       �	��JXc�A�-*

loss
F�=�`t       �	�EKXc�A�-*

loss;>�;[��"       �	2�KXc�A�-*

loss��<.m>�       �	ˢLXc�A�-*

lossLS=�D��       �	�XMXc�A�-*

loss�K�8���o       �	��NXc�A�-*

loss};=�i<       �	��OXc�A�-*

loss��;,lCX       �	�PXc�A�-*

loss��0<�0��       �	�|QXc�A�-*

loss�<W:��        �	#RXc�A�.*

lossV:4}c       �	��RXc�A�.*

loss<��;���       �	�GSXc�A�.*

loss��K<
��       �	`�SXc�A�.*

loss&Q�:,u�       �	��TXc�A�.*

loss���9�� �       �	�&UXc�A�.*

losst|�<�s�       �	L�UXc�A�.*

loss�:�T�       �	�[VXc�A�.*

lossv�=��=       �	x�VXc�A�.*

loss�Z�:�Ʌq       �	��WXc�A�.*

lossQ�L;��~Z       �	�XXc�A�.*

loss��<1�u�       �	��XXc�A�.*

loss.'<�C�       �	=GYXc�A�.*

loss�_<53�       �	��YXc�A�.*

loss-Ȧ;}��{       �	lwZXc�A�.*

loss�%;��V�       �	[Xc�A�.*

loss�=R�/       �	]�[Xc�A�.*

loss�3�=�/       �	gD\Xc�A�.*

loss�[*;,#��       �	O�\Xc�A�.*

loss���;��
       �	�}]Xc�A�.*

loss7��;��;�       �	f^Xc�A�.*

loss6��;��Q�       �	2�^Xc�A�.*

lossMQ�;Un3I       �	5F_Xc�A�.*

loss�q&;Ǿ�       �	��_Xc�A�.*

loss1_�<n�R       �	z`Xc�A�.*

lossr|<4��       �	"aXc�A�.*

loss:y�:.POS       �	t�aXc�A�.*

loss�y<?�m�       �	BZbXc�A�.*

loss6��<S���       �	��bXc�A�.*

loss ��=�(0       �	T�cXc�A�.*

loss�R�:�㿿       �	�,dXc�A�.*

loss��;��V�       �	�dXc�A�.*

loss24;��J       �	GqeXc�A�.*

lossOx<A[1�       �	�fXc�A�.*

loss�B:��2�       �	��fXc�A�.*

loss_�+:6��       �	f2gXc�A�.*

loss$�B:f�!P       �	��gXc�A�.*

loss �+=��A�       �	{jhXc�A�.*

lossdr4<���       �	� iXc�A�.*

loss�:��L       �	0�iXc�A�.*

loss��(<^�v�       �	74jXc�A�.*

lossQ��=�D�&       �	!�jXc�A�.*

loss$�:��Ȣ       �	��kXc�A�.*

loss�h99X�z       �	=lXc�A�.*

lossn�b<X�J�       �	��lXc�A�.*

loss���:���       �	HomXc�A�.*

loss�Z�6~��B       �	�	nXc�A�.*

loss1;(��R       �	�nXc�A�.*

lossjD�9+�N       �	hBoXc�A�.*

loss��0:�f��       �	��oXc�A�.*

loss�ƹ<
_��       �	�qpXc�A�.*

loss(�`9���       �	BqXc�A�.*

loss��B:2��K       �	.�qXc�A�.*

lossQ�R:�v�       �	�>rXc�A�.*

loss1�(9���       �	1�rXc�A�.*

loss�L8+�J2       �	�bsXc�A�.*

loss�;�;o�l       �	t`tXc�A�.*

loss]v�<mæ�       �	��tXc�A�.*

lossD��;���       �	�uXc�A�.*

lossSZ�9����       �	ivXc�A�.*

loss�BO;4�       �	wXc�A�.*

loss���;��v=       �	ݘwXc�A�.*

loss�;K>�       �	�KxXc�A�.*

loss��1>C�t       �	��xXc�A�.*

lossl��<�Ӌ       �	��yXc�A�.*

lossR��:�Q�{       �	�:zXc�A�.*

loss�=�/�       �	��zXc�A�.*

lossn��;
�'J       �	\r{Xc�A�.*

loss�5�<�kl�       �	�G|Xc�A�.*

lossJ�<�?�|       �	a�|Xc�A�.*

losstB�<�'d�       �	Z�}Xc�A�.*

lossc��<����       �	C~Xc�A�.*

lossMt<
v��       �	��~Xc�A�.*

loss��=��)       �	
MXc�A�.*

loss�n�<�C�       �	��Xc�A�.*

loss��H;�:�       �	퀀Xc�A�.*

loss�j<AA�B       �	�*�Xc�A�.*

losshk�;[���       �	�ƁXc�A�.*

loss�7!=⽛N       �	*p�Xc�A�.*

loss�8<Īߝ       �	��Xc�A�.*

loss�w�<�f�       �	ࢃXc�A�.*

loss���;�%r        �	�7�Xc�A�.*

loss/4�:��[�       �	�τXc�A�.*

lossu=I��       �	�e�Xc�A�.*

loss�׆<��U       �	��Xc�A�.*

loss׼;9ωy       �	��Xc�A�.*

loss�}�:b,       �	�8�Xc�A�.*

loss߽:z+f�       �	̇Xc�A�.*

loss��<񉃂       �	d�Xc�A�.*

lossW��:W+\       �	��Xc�A�.*

loss4��<߆S�       �	稉Xc�A�.*

loss|3�;J\3�       �	)?�Xc�A�.*

loss�m�<�^��       �	�ӊXc�A�.*

loss�.<�e�       �	�w�Xc�A�.*

loss�y/:t¶`       �	��Xc�A�.*

loss��;�R^�       �	: �Xc�A�.*

lossl&�;'�W       �	�Xc�A�.*

loss�i�:��S�       �	��Xc�A�.*

loss:�2<@E�       �	n4�Xc�A�.*

loss�s;�ڎB       �	NЏXc�A�.*

lossJb:<[~��       �	j�Xc�A�.*

loss_��<�;��       �	�Xc�A�.*

loss�<Ƌ�]       �	1��Xc�A�.*

loss��F<�Y�O       �	-�Xc�A�.*

loss���;�)�       �	�ĒXc�A�.*

loss3�p;'\γ       �	�\�Xc�A�.*

loss��;�DH�       �	E�Xc�A�.*

loss�;mJ*       �	"��Xc�A�.*

loss��=�D�       �	�!�Xc�A�.*

loss�.A:�i��       �		ŕXc�A�.*

loss���;
�$       �	�[�Xc�A�.*

lossR�$;��/�       �	2�Xc�A�.*

lossr�9c-		       �	N��Xc�A�.*

loss3��;��D�       �	�O�Xc�A�.*

loss[Y;Nz+�       �	��Xc�A�.*

loss���<qzn�       �	�s�Xc�A�.*

loss�9��m�       �	�Xc�A�.*

loss/N<<g�J       �	8��Xc�A�.*

lossF�;Kͧ       �	�o�Xc�A�.*

loss]A;Q�A/       �	p�Xc�A�.*

lossV7F<�u��       �	O$�Xc�A�.*

loss��<~]�H       �	�ŵXc�A�.*

loss�v�<*�8�       �	��Xc�A�.*

loss@p~;�yx:       �	᳷Xc�A�.*

loss��<�.1b       �		S�Xc�A�.*

loss\�1;�]��       �	��Xc�A�.*

loss�K<�D��       �	��Xc�A�.*

lossC�$<$���       �	<1�Xc�A�.*

loss恲9�G       �	��Xc�A�/*

loss��V74�Q�       �	}��Xc�A�/*

lossW�;.33M       �	iT�Xc�A�/*

loss�g�;�<        �	�U�Xc�A�/*

lossW>Q;D֊       �	y�Xc�A�/*

loss��<&N�$       �	�z�Xc�A�/*

loss��<�Z��       �	�P�Xc�A�/*

loss���:a)��       �	i;�Xc�A�/*

lossۦt<�D��       �	���Xc�A�/*

loss|vA;��z�       �	=��Xc�A�/*

loss?<�^F�       �		�Xc�A�/*

loss�5S;{A�       �	F��Xc�A�/*

loss{��<EtM       �	��Xc�A�/*

loss�վ<"��       �	9�Xc�A�/*

lossr��:οҮ       �	���Xc�A�/*

lossZog;S)��       �	y�Xc�A�/*

loss[Y=;x���       �	�Xc�A�/*

loss�+�:��T�       �	ݴ�Xc�A�/*

lossP�<l��       �	�J�Xc�A�/*

loss�!�9�y��       �	���Xc�A�/*

lossf��:����       �	Ō�Xc�A�/*

loss��K<֨��       �	)"�Xc�A�/*

lossQ�</1��       �	k��Xc�A�/*

lossΈ=�dA       �	.X�Xc�A�/*

loss�>Y:�g\       �	���Xc�A�/*

loss4��:���G       �	���Xc�A�/*

lossW��9 ���       �	2=�Xc�A�/*

lossF$n:V�7C       �	���Xc�A�/*

lossc}�<��F       �	��Xc�A�/*

loss�B;�>�B       �	�2�Xc�A�/*

lossc�=�;�       �	���Xc�A�/*

loss��;d^SC       �	�D�Xc�A�/*

lossOG,=s�.       �	��Xc�A�/*

loss��:���p       �	��Xc�A�/*

loss�8�F��       �	LS�Xc�A�/*

loss�8<G�x       �	���Xc�A�/*

lossxA<��R�       �	ǀ�Xc�A�/*

loss�ܴ;��p�       �	��Xc�A�/*

loss��;o���       �	û�Xc�A�/*

loss�*�;'�Ho       �	�Y�Xc�A�/*

loss��<��4s       �	$��Xc�A�/*

loss�a�:�K�@       �	��Xc�A�/*

loss�wb9^H�I       �	�'�Xc�A�/*

loss]�L<�5       �	u��Xc�A�/*

loss��<5at�       �	�k�Xc�A�/*

lossT�<�i+(       �	��Xc�A�/*

lossZ�:�i R       �	���Xc�A�/*

loss���;�<�Y       �	�K�Xc�A�/*

lossZx�8.ˬ[       �	��Xc�A�/*

loss�0:�+'�       �	���Xc�A�/*

lossW�L:�4�       �	Y6�Xc�A�/*

loss�T:���S       �	��Xc�A�/*

loss?0�:�m�s       �	��Xc�A�/*

loss.C�<c.1�       �	�#�Xc�A�/*

lossA�;���       �	���Xc�A�/*

loss`�<]%��       �	�p�Xc�A�/*

loss/h;8X�}       �	v�Xc�A�/*

losst��;#2�       �	V��Xc�A�/*

loss�m;��h�       �	�[�Xc�A�/*

loss^ȍ;u��       �	���Xc�A�/*

loss���<���p       �	���Xc�A�/*

loss��t9����       �	�8�Xc�A�/*

loss(��9I�$       �	[��Xc�A�/*

loss�p�<1�f       �	�i�Xc�A�/*

loss��<�I8       �	��Xc�A�/*

loss�WJ:��w9       �	���Xc�A�/*

loss4�<��_       �	���Xc�A�/*

lossh�e<��gf       �	�-�Xc�A�/*

losso��8����       �	���Xc�A�/*

lossjХ80���       �	Xp�Xc�A�/*

loss!��9l[a/       �	�&�Xc�A�/*

loss�ר8J�3�       �	���Xc�A�/*

loss`h�;$�x�       �	^�Xc�A�/*

loss1:h�ߺ       �	Y��Xc�A�/*

loss���<��)       �	5��Xc�A�/*

loss.��:뢙�       �	w�Xc�A�/*

loss��</�	�       �	*�Xc�A�/*

lossJZ8��       �	Y��Xc�A�/*

lossf( <s�K�       �	$^�Xc�A�/*

lossj"4<��m       �	3��Xc�A�/*

loss���8G�E|       �	��Xc�A�/*

lossz�@<��        �	�I�Xc�A�/*

lossU]�:zr}�       �	:��Xc�A�/*

lossה)<M��G       �	���Xc�A�/*

lossqܶ9-��       �	'0�Xc�A�/*

loss!�:��       �	:��Xc�A�/*

loss?�<�^��       �	�o�Xc�A�/*

loss�v=�ڕ       �	��Xc�A�/*

loss��J9��,{       �	N��Xc�A�/*

loss&;:��*       �	m;�Xc�A�/*

loss�1�<��       �	��Xc�A�/*

loss�_;��B�       �	%��Xc�A�/*

loss�D4<=%�       �	�N�Xc�A�/*

loss��>�x��       �	���Xc�A�/*

loss��<g%�!       �	3��Xc�A�/*

loss1F/=��u�       �	�)�Xc�A�/*

loss�e<x糿       �	��Xc�A�/*

lossg� <f#�       �	��Xc�A�/*

loss���9'��       �	R|�Xc�A�/*

lossq�V7_�       �	�-�Xc�A�/*

loss��k9��J�       �	��Xc�A�/*

loss�:a7�G       �	�p�Xc�A�/*

loss?�7=�i�       �	?�Xc�A�/*

lossݝ�:�9�       �	���Xc�A�/*

loss>܀;��'C       �	�y�Xc�A�/*

loss�<CwL       �	\  �Xc�A�/*

loss�2�<"D"H       �	�� �Xc�A�/*

loss��=,�Ǩ       �	.q�Xc�A�/*

loss���;J~��       �	L�Xc�A�/*

loss<.:d>�{       �	���Xc�A�/*

loss{?p=O>º       �	ga�Xc�A�/*

lossl'=:�u       �	��Xc�A�/*

loss��};�(�2       �	���Xc�A�/*

loss��l=�<,
       �	eU�Xc�A�/*

loss�="�	K       �	��Xc�A�/*

loss�r�;��;S       �	���Xc�A�/*

loss�FH<�u��       �	gC�Xc�A�/*

loss	J<��       �	{�Xc�A�/*

loss���= ިQ       �	��Xc�A�/*

loss�GJ:�X��       �	�d	�Xc�A�/*

lossW@�;<�0�       �	�
�Xc�A�/*

loss`:#�l}       �	J�
�Xc�A�/*

loss��;����       �	#L�Xc�A�/*

loss��:(�]Z       �	���Xc�A�/*

loss.��<���v       �	�~�Xc�A�/*

loss��=;UN�       �	�+�Xc�A�/*

loss���<��bN       �	��Xc�A�/*

lossD�<��        �	]3�Xc�A�/*

lossMn�;�@A�       �	ú�Xc�A�0*

lossԦ=�z       �	�b�Xc�A�0*

loss�!2=�q$�       �	N�Xc�A�0*

loss�q;L%�       �	��Xc�A�0*

loss�Q;T]�3       �	�B�Xc�A�0*

loss脒;/��8       �	���Xc�A�0*

lossh;�b�       �	>v�Xc�A�0*

loss��;�S�       �	�
�Xc�A�0*

loss@s8<�w�Q       �	H��Xc�A�0*

loss#�<�6�	       �	�B�Xc�A�0*

loss���;wj��       �	���Xc�A�0*

loss��<���#       �	���Xc�A�0*

lossc�:���       �	�7�Xc�A�0*

loss� =<*@�       �	���Xc�A�0*

losse|U<�f�_       �	8g�Xc�A�0*

loss��j<����       �	r��Xc�A�0*

loss���9g��       �	���Xc�A�0*

lossl�<��Ē       �	r1�Xc�A�0*

loss�k�:���       �	K��Xc�A�0*

lossԳ=� ��       �	���Xc�A�0*

losszc�;A�+�       �	��Xc�A�0*

loss�\�=���       �	��Xc�A�0*

lossذm<)�<^       �	���Xc�A�0*

loss���9�D*       �	�H�Xc�A�0*

lossk� <v��       �	i��Xc�A�0*

loss=��;�6��       �	�x �Xc�A�0*

loss��6:�B^G       �	!�Xc�A�0*

loss���=?�^       �	��!�Xc�A�0*

loss���;{e��       �	�K"�Xc�A�0*

loss:M:��       �	�#�Xc�A�0*

loss�܀<�.%�       �	�)$�Xc�A�0*

loss�pb:�o�       �	�9%�Xc�A�0*

lossZ;o<K���       �	.�%�Xc�A�0*

loss� =0i�       �	�\&�Xc�A�0*

loss��:<
��L       �	r�&�Xc�A�0*

loss���<���+       �	y�'�Xc�A�0*

loss�F�8�Hf7       �	82(�Xc�A�0*

lossӬ_9�5�U       �	h�(�Xc�A�0*

loss�)\=s6       �	Rb)�Xc�A�0*

lossSZ�;��C�       �	T�)�Xc�A�0*

loss�E�<)m˫       �	��*�Xc�A�0*

loss=W-:�hn       �	Lp+�Xc�A�0*

loss�hS;�GVq       �	�,�Xc�A�0*

loss��<�`z8       �	9�,�Xc�A�0*

lossE��;͓��       �	^J-�Xc�A�0*

loss2��;>ol�       �	;�-�Xc�A�0*

lossx�;Nv4�       �	z�.�Xc�A�0*

lossC��<��       �	tA/�Xc�A�0*

loss��:���       �	��/�Xc�A�0*

loss?=ρCe       �	�{0�Xc�A�0*

lossrp�:�i�       �	O;1�Xc�A�0*

lossA<�<��       �	!2�Xc�A�0*

loss�&�;����       �	�2�Xc�A�0*

loss���;S)�a       �	E*3�Xc�A�0*

loss��1=��       �	�3�Xc�A�0*

loss�&<J�       �	��4�Xc�A�0*

loss���9֖�       �	!5�Xc�A�0*

lossl��;o�(H       �	��5�Xc�A�0*

loss,5�:���       �	bM6�Xc�A�0*

loss�&O:���       �	a�6�Xc�A�0*

loss[:�<@��C       �	�y7�Xc�A�0*

loss�D�9h��       �	8�Xc�A�0*

lossg`<��	       �	�8�Xc�A�0*

lossv��<|j�       �	�=9�Xc�A�0*

loss��<8���       �	�W:�Xc�A�0*

losssh#;���       �	n�:�Xc�A�0*

loss���=�g�       �	ԙ;�Xc�A�0*

loss��v;��P
       �	=<�Xc�A�0*

loss_&i<|��       �	Cu=�Xc�A�0*

loss���<�1C�       �	E>�Xc�A�0*

loss�]�;����       �	&�>�Xc�A�0*

lossȡ�<���       �	�D?�Xc�A�0*

loss��5<�<��       �	��?�Xc�A�0*

loss���;�d�D       �	4�@�Xc�A�0*

loss$��=��,�       �	5(A�Xc�A�0*

lossӪ&=2�S8       �	x�A�Xc�A�0*

lossؖD;q.��       �	�gB�Xc�A�0*

loss�J�9 �`�       �	��B�Xc�A�0*

loss���9ek�^       �	5�C�Xc�A�0*

loss�A;�M�       �	p@D�Xc�A�0*

loss�P<�9��       �	@�D�Xc�A�0*

lossd:�ߨ       �	�oE�Xc�A�0*

loss���:~l?f       �	F�Xc�A�0*

loss	-<fp�h       �	��F�Xc�A�0*

loss�A<+&KS       �	�bG�Xc�A�0*

loss�ފ:[Gŗ       �	-"H�Xc�A�0*

loss�`.<C��       �	�H�Xc�A�0*

loss��;�+b�       �	VI�Xc�A�0*

lossJ=F��       �	��I�Xc�A�0*

lossa <��9T       �	��J�Xc�A�0*

loss�*=�!��       �	�K�Xc�A�0*

loss-<�Ko       �	��K�Xc�A�0*

lossF̐;����       �	��L�Xc�A�0*

loss��
:A1�       �	g&M�Xc�A�0*

loss/{F=��X       �	��M�Xc�A�0*

loss�v=w=6�       �	1_N�Xc�A�0*

loss��;�yLk       �	�
O�Xc�A�0*

loss:K<�hX�       �	U�O�Xc�A�0*

loss�ʖ<Ɯx       �	��P�Xc�A�0*

lossOF:��P�       �	znQ�Xc�A�0*

lossØ�:(��       �	!VR�Xc�A�0*

loss�L;�]�M       �	��R�Xc�A�0*

loss:�<b�C�       �	��S�Xc�A�0*

loss�)E9�˜       �	�T�Xc�A�0*

loss�ұ9p�y       �	��T�Xc�A�0*

loss�Wn;�?V�       �	t`U�Xc�A�0*

lossl�:FL��       �	DV�Xc�A�0*

loss� =B7��       �	��V�Xc�A�0*

loss�}�:���       �	gGW�Xc�A�0*

loss�&=�3�&       �	��W�Xc�A�0*

lossY��<���       �	hwX�Xc�A�0*

loss�6"9V�"       �	�Y�Xc�A�0*

loss���;����       �	��Y�Xc�A�0*

loss|��;n�g       �	z8Z�Xc�A�0*

lossk�;�       �	b�Z�Xc�A�0*

loss���:N\U0       �	�s[�Xc�A�0*

loss�-�;&D�0       �	W
\�Xc�A�0*

lossg�;��,       �	��\�Xc�A�0*

loss3�L;�34�       �	�d]�Xc�A�0*

loss�:�;�/��       �	+�]�Xc�A�0*

loss��K:A�je       �	��^�Xc�A�0*

loss`n�;<��       �	�)_�Xc�A�0*

loss8$u:8�       �	P�_�Xc�A�0*

losscyC=�       �	�\`�Xc�A�0*

lossl��:�xW�       �	Q�`�Xc�A�0*

loss�F�:�M.�       �	�a�Xc�A�0*

loss
��<v��       �	�(b�Xc�A�0*

loss��;eK�       �	�b�Xc�A�0*

lossZF-:8�5       �	�Xc�Xc�A�1*

loss��;�qr�       �	��c�Xc�A�1*

loss��<�и       �	��d�Xc�A�1*

loss�<��ݕ       �	Q3e�Xc�A�1*

lossw:�:W.׽       �	�e�Xc�A�1*

loss���;Ja;       �	�jf�Xc�A�1*

loss䒦<R�Z�       �	Gg�Xc�A�1*

loss��G<�<�       �	��g�Xc�A�1*

loss<��;���       �	�>h�Xc�A�1*

loss�:*:���       �	x�h�Xc�A�1*

loss׳�;\��       �	ii�Xc�A�1*

lossL�<M���       �	��i�Xc�A�1*

loss&�e<���       �	d�j�Xc�A�1*

lossWI�<���       �	�k�Xc�A�1*

loss��=%)�	       �	2rl�Xc�A�1*

loss<��j       �	!m�Xc�A�1*

loss���;)o��       �	��m�Xc�A�1*

loss�=�:)�       �	#jn�Xc�A�1*

loss�׊:�͊�       �	g
o�Xc�A�1*

loss���:�%E       �	�o�Xc�A�1*

lossT��<b���       �	��p�Xc�A�1*

loss,�<p���       �	S�q�Xc�A�1*

loss$#�;�'�u       �	8jr�Xc�A�1*

loss�oB;%%,>       �	� s�Xc�A�1*

loss���<V4��       �	�s�Xc�A�1*

loss`�<�ʹ       �	�0t�Xc�A�1*

loss�.<Ɔ��       �	�u�Xc�A�1*

loss�=1ɬ       �	U�u�Xc�A�1*

loss&@�=j       �	�Ww�Xc�A�1*

loss� =�B�&       �	��w�Xc�A�1*

loss�t�;&�F�       �	ёx�Xc�A�1*

loss�q�:J��       �	.y�Xc�A�1*

loss`��<�5g       �	enz�Xc�A�1*

lossn��<�t�       �	{�Xc�A�1*

loss�V=d�/z       �	�{�Xc�A�1*

loss<85:B�68       �	(I|�Xc�A�1*

lossq";�f5P       �	��|�Xc�A�1*

lossO�R<�ø�       �	χ}�Xc�A�1*

loss��:٩�O       �	6~�Xc�A�1*

loss0|;���       �	c�Xc�A�1*

lossI�u:)>��       �	*��Xc�A�1*

lossL��;%K\�       �	ۇ��Xc�A�1*

losst�<6��       �	���Xc�A�1*

loss(��:\���       �	����Xc�A�1*

loss��<�}4b       �	�\��Xc�A�1*

loss��;��_�       �	�Xc�A�1*

lossT�+;'�P       �	����Xc�A�1*

loss�|�<v��       �	`?��Xc�A�1*

lossL��<�s�/       �	�ل�Xc�A�1*

loss�*;��	�       �	t���Xc�A�1*

loss)�%=���       �	�X��Xc�A�1*

loss�?;t���       �	FXc�A�1*

lossT�<�9"       �	Y���Xc�A�1*

loss~x: _(�       �	�%��Xc�A�1*

loss��;�f��       �	����Xc�A�1*

lossT9�:Rl�c       �	�U��Xc�A�1*

loss��;�/�       �	#�Xc�A�1*

lossX��:�[       �	׉��Xc�A�1*

lossmD<���       �	M,��Xc�A�1*

lossÑ];?�       �	Pċ�Xc�A�1*

loss$�<���        �	�k��Xc�A�1*

lossƷ�9��       �	���Xc�A�1*

loss���=e[�E       �	����Xc�A�1*

loss{��;��J       �	xA��Xc�A�1*

loss�_�;�ek       �	Uێ�Xc�A�1*

lossi��<����       �	��Xc�A�1*

loss�U�<�F6       �	7ސ�Xc�A�1*

loss�(�<eç�       �	����Xc�A�1*

loss�/&;�#0       �	f���Xc�A�1*

loss�d:/��       �	X���Xc�A�1*

loss��=l��       �	�W��Xc�A�1*

loss��<�&R�       �	��Xc�A�1*

lossł�9��+X       �	����Xc�A�1*

loss�¯<5��       �	f���Xc�A�1*

lossi��<K1�^       �	���Xc�A�1*

lossn֒;���       �	@j��Xc�A�1*

loss��;: R�       �	 ��Xc�A�1*

loss���;���U       �		p��Xc�A�1*

loss,�;6\�       �	���Xc�A�1*

loss���:.�&       �	����Xc�A�1*

lossW��:ͼ��       �	H��Xc�A�1*

lossK��<�}:       �	��Xc�A�1*

lossN^M<O<=�       �	㨝�Xc�A�1*

loss�<s�N�       �	���Xc�A�1*

loss��Y;,]�p       �	z��Xc�A�1*

loss1y�:��|       �	^���Xc�A�1*

lossx�=�9       �	)Z��Xc�A�1*

loss�y;���       �	�
��Xc�A�1*

loss��E:� ��       �	����Xc�A�1*

lossJ<
��v       �	�<��Xc�A�1*

lossa��:J�       �	5Т�Xc�A�1*

loss;�S;�Ct�       �	&m��Xc�A�1*

loss̈́z=r��       �	���Xc�A�1*

lossE��<A��I       �	F뤀Xc�A�1*

loss�E�<���       �	p���Xc�A�1*

loss��<ER�       �	]4��Xc�A�1*

loss���<��       �	Iۦ�Xc�A�1*

loss�6�:V���       �	���Xc�A�1*

lossO�J;�]#�       �	�+��Xc�A�1*

lossr};���       �	�è�Xc�A�1*

loss� :
^�       �	�_��Xc�A�1*

lossO��;��       �	2���Xc�A�1*

loss�<&<�͈�       �	����Xc�A�1*

loss��D:%���       �	7��Xc�A�1*

loss���;� @       �	5Ы�Xc�A�1*

loss&0�;WM�p       �	e��Xc�A�1*

loss��:F̋h       �	����Xc�A�1*

losseD=GM       �	����Xc�A�1*

loss�R<���       �	
/��Xc�A�1*

loss��;�H�@       �	Ȯ�Xc�A�1*

lossvÜ<>T��       �	t]��Xc�A�1*

loss��2=�T�*       �	����Xc�A�1*

loss�;�9ӧĺ       �	����Xc�A�1*

loss�4�9���       �	�&��Xc�A�1*

lossz:Q=��q�       �	n±�Xc�A�1*

lossx�i<�s       �	q���Xc�A�1*

loss��<[.�o       �	kI��Xc�A�1*

loss'�;����       �	鳀Xc�A�1*

loss�j�<��1x       �	��Xc�A�1*

loss�T�:��'       �	�E��Xc�A�1*

lossѶ�;�xHr       �	�޵�Xc�A�1*

loss�K:qic       �	�~��Xc�A�1*

loss-�;�(j}       �	y"��Xc�A�1*

loss�E
:l��?       �	xշ�Xc�A�1*

loss��"=OC��       �	hz��Xc�A�1*

loss$��9Fm��       �	7��Xc�A�1*

loss�m90�'       �	<ݹ�Xc�A�1*

loss�g:�J3�       �	1{��Xc�A�1*

loss3T�;�s�       �	�e��Xc�A�2*

loss��;+�|	       �	��Xc�A�2*

loss�B�<��       �	���Xc�A�2*

loss�p�<i�       �	�[��Xc�A�2*

lossc�:9��       �	}"��Xc�A�2*

loss��g<8���       �	���Xc�A�2*

loss_rY;��O       �	�Y��Xc�A�2*

loss�(�<5n       �	^�Xc�A�2*

lossa<8EH�       �	���Xc�A�2*

loss�<g�_       �	1#��Xc�A�2*

loss:�;XVg       �	����Xc�A�2*

loss��~<���       �	�lXc�A�2*

loss�l�=(b       �	�ÀXc�A�2*

lossȽ;n�y�       �	�ÀXc�A�2*

loss��<���F       �	JbĀXc�A�2*

loss��X<l��;       �	�ŀXc�A�2*

loss�c�<�x       �	ԛŀXc�A�2*

loss���;���N       �	�2ƀXc�A�2*

loss�Y<��Z"       �	,�ƀXc�A�2*

loss(e�<�ç�       �	�nǀXc�A�2*

loss=�i:�@��       �	�ȀXc�A�2*

loss}:q:|��       �	�ȀXc�A�2*

loss��=^�       �	WCɀXc�A�2*

loss� <3��       �	��ɀXc�A�2*

loss�`�=�j��       �	WwʀXc�A�2*

loss�ȋ<��xO       �	�ˀXc�A�2*

loss�J<C�U       �	ԹˀXc�A�2*

loss?�<
�h%       �	�ỲXc�A�2*

lossr+=$ٌ�       �	��̀Xc�A�2*

loss\֝=fH"�       �	�̀Xc�A�2*

loss��9�ֶ�       �	�#΀Xc�A�2*

loss�ˁ=�V�       �	��΀Xc�A�2*

loss���=�6i�       �	iVπXc�A�2*

lossH��:_��       �	��ЀXc�A�2*

loss��;S��'       �	�IрXc�A�2*

lossׄ<��}.       �	��рXc�A�2*

loss��:q��       �	�ҀXc�A�2*

loss��:���       �	�!ӀXc�A�2*

lossH�<��P�       �	�ӀXc�A�2*

loss���;��Z       �	�ԀXc�A�2*

loss̙�;4<�,       �	c&ՀXc�A�2*

lossi.�;=�!�       �	�ՀXc�A�2*

loss\�<��]�       �	��րXc�A�2*

loss~0�<�#�       �	Н׀Xc�A�2*

loss6�<���       �	�U؀Xc�A�2*

loss�Մ<>%~       �	�؀Xc�A�2*

loss9�=u�`       �	�ـXc�A�2*

loss���<��q�       �	73ڀXc�A�2*

loss��E<)!7       �	��ڀXc�A�2*

lossX;��"�       �	ZۀXc�A�2*

loss��"=8�>       �	
�ۀXc�A�2*

loss��<X���       �	N�܀Xc�A�2*

lossZ��;&ԋ       �	I݀Xc�A�2*

loss�;�<�h       �	��݀Xc�A�2*

loss..<d��A       �	�sހXc�A�2*

loss�H=$ͤ�       �	�
߀Xc�A�2*

lossA�';���"       �	��߀Xc�A�2*

loss��^=lk�!       �	�8��Xc�A�2*

loss�_J;t��p       �	t���Xc�A�2*

loss�%Q<���i       �	�q�Xc�A�2*

loss��H<_���       �	��Xc�A�2*

lossj��:��J�       �	���Xc�A�2*

loss�;�0�       �	�X�Xc�A�2*

loss�RY:}��       �	���Xc�A�2*

loss�T�<����       �	&��Xc�A�2*

lossY�;<`�c       �	� �Xc�A�2*

lossq;e���       �	���Xc�A�2*

loss6��<��fa       �	8J�Xc�A�2*

lossS��;�τ       �	��Xc�A�2*

loss_Ͽ<����       �	���Xc�A�2*

loss�d�8)�b       �	#�Xc�A�2*

loss�N�=�8�       �	R��Xc�A�2*

loss[Y�<�4       �	�L�Xc�A�2*

lossf�:E�$`       �	�Xc�A�2*

loss�t�=�+�       �	��Xc�A�2*

lossrm�;�F�2       �	^H�Xc�A�2*

losss�9	�m�       �	/��Xc�A�2*

loss|X�:O��j       �	;r�Xc�A�2*

loss{ �;d�a       �	��Xc�A�2*

loss��=���S       �	��Xc�A�2*

losssM>Bk�       �	zS�Xc�A�2*

loss��*=���       �	���Xc�A�2*

loss�;\;�༃       �	b���Xc�A�2*

lossH�/<#�%       �	\U�Xc�A�2*

loss၃;��-       �	���Xc�A�2*

loss�)=t3       �	���Xc�A�2*

lossty9+S       �	*W�Xc�A�2*

loss��s:MYL�       �	���Xc�A�2*

loss�U^<~ z�       �	*��Xc�A�2*

loss,R�<��       �	�7��Xc�A�2*

loss۸i<�jg       �	!���Xc�A�2*

loss�5�<�W       �	�f��Xc�A�2*

loss9�;یk�       �	����Xc�A�2*

loss� w;���9       �	����Xc�A�2*

lossq��:9Ʋf       �	M��Xc�A�2*

loss�=�:��{�       �	����Xc�A�2*

lossf�.=�x�       �	�~��Xc�A�2*

loss�k<^k�       �	'��Xc�A�2*

loss�˅<w�m�       �	n��Xc�A�2*

loss��<I7�v       �	x���Xc�A�2*

lossҳd<Ky�5       �	�v��Xc�A�2*

lossؼ=��M       �	vQ��Xc�A�2*

loss��9���       �	����Xc�A�2*

loss��;A�8�       �	��Xc�A�2*

loss*�8;��MQ       �	C��Xc�A�2*

lossC4<[�W�       �	� �Xc�A�2*

lossnBw;��V3       �	*� �Xc�A�2*

loss�)�<N!c�       �	�@�Xc�A�2*

loss���;�+�       �	k��Xc�A�2*

loss���;Vnt�       �	��Xc�A�2*

lossF�<��w        �	�;�Xc�A�2*

loss3q<���X       �	���Xc�A�2*

loss)��;~�       �	�w�Xc�A�2*

lossN{C;|��n       �	|�Xc�A�2*

loss~�<j��J       �	���Xc�A�2*

lossnW:��       �	��Xc�A�2*

loss�& ;|k��       �	|+�Xc�A�2*

loss�8�<���       �	���Xc�A�2*

loss��&<o�S       �	h�Xc�A�2*

lossW�o;���       �	�
	�Xc�A�2*

loss�Ϥ<��h�       �	?�	�Xc�A�2*

loss�IW9PQ��       �	�E
�Xc�A�2*

lossN��;�mA4       �	��
�Xc�A�2*

loss��9"v��       �	�{�Xc�A�2*

loss��:�d��       �	�Xc�A�2*

losswV;�       �	���Xc�A�2*

loss��;ϭ�u       �	1@�Xc�A�2*

loss�W�:
V�3       �	o��Xc�A�2*

loss�4$<
�       �	�u�Xc�A�3*

loss�p�<�_�v       �	��Xc�A�3*

loss�;��?d       �	���Xc�A�3*

loss��:{�+6       �	���Xc�A�3*

loss3�5=3��       �	g��Xc�A�3*

loss͌=<�M��       �	40�Xc�A�3*

losspk8kǽu       �	u��Xc�A�3*

loss$V;'g�D       �	]�Xc�A�3*

loss
�9��Q�       �	�!�Xc�A�3*

loss�(S;ʐ�       �	��Xc�A�3*

loss�R�<t9�       �	,c�Xc�A�3*

loss(@�9ӼP�       �	�
�Xc�A�3*

loss��=���       �	O��Xc�A�3*

loss�F�:<��       �	XV�Xc�A�3*

loss��{5���       �	���Xc�A�3*

loss��8_r:,       �	r��Xc�A�3*

loss��9U�7       �	�>�Xc�A�3*

lossd(�;�5X�       �	^��Xc�A�3*

lossDGg;��x�       �	��Xc�A�3*

loss�}^9eY�       �	�Xc�A�3*

lossX��=���       �	���Xc�A�3*

loss�U�<W��       �	i�Xc�A�3*

loss9�";��Z       �	���Xc�A�3*

loss�8�=A���       �	e��Xc�A�3*

loss�Fq<7\�"       �	�b�Xc�A�3*

loss��
=�ڊ^       �	���Xc�A�3*

loss6\:F�<"       �	v��Xc�A�3*

lossƶ�9د�S       �	X9 �Xc�A�3*

lossT��=z�J4       �	:� �Xc�A�3*

loss���9�I�       �	�e!�Xc�A�3*

lossr+<[Q��       �	��!�Xc�A�3*

loss��~<ֈ��       �	1�"�Xc�A�3*

loss�=:�jC       �	w+#�Xc�A�3*

lossH�M=��ƨ       �	��#�Xc�A�3*

loss��=	HW       �	]$�Xc�A�3*

lossE[�;6/�6       �	V�$�Xc�A�3*

loss��1<|'rY       �	]�%�Xc�A�3*

loss��	=k�0Y       �	�(&�Xc�A�3*

loss�L�;�i��       �	G�&�Xc�A�3*

loss��<�t��       �	�n'�Xc�A�3*

losss��<��y�       �	F	(�Xc�A�3*

lossp�;(��i       �	Ҩ(�Xc�A�3*

loss��	:/�|a       �	�\)�Xc�A�3*

lossz�;H&=       �	YR*�Xc�A�3*

lossdu�<�a+}       �	�*�Xc�A�3*

loss-H@;����       �	�+�Xc�A�3*

loss_%:;�U�       �	,�Xc�A�3*

loss�f�:eJuH       �	�,�Xc�A�3*

lossE�=;D��       �	��-�Xc�A�3*

loss:Z=�"       �	�.�Xc�A�3*

lossfˊ<D�@�       �	{�.�Xc�A�3*

lossc`;J[^       �		R/�Xc�A�3*

loss q<��gD       �	G�/�Xc�A�3*

loss��=F�       �	R}0�Xc�A�3*

lossӯ�<�>u!       �	{1�Xc�A�3*

loss�u<��%|       �	��1�Xc�A�3*

loss�͠=q)��       �	3P2�Xc�A�3*

loss�Z�;����       �	��2�Xc�A�3*

loss��&;[�*{       �	 |3�Xc�A�3*

loss�"�9x��       �	'4�Xc�A�3*

lossO�<u	        �	��4�Xc�A�3*

loss,��:��L|       �	�5�Xc�A�3*

loss�h:$�:~       �	�6�Xc�A�3*

loss���;�K�       �	O�6�Xc�A�3*

lossQK�<Ң�#       �	�F7�Xc�A�3*

loss���;���       �	��7�Xc�A�3*

lossi��;e[m)       �	8�Xc�A�3*

loss�]�;N��       �		9�Xc�A�3*

loss�p=����       �	w�9�Xc�A�3*

loss#~D9�o�_       �	�m:�Xc�A�3*

loss)��;�E�       �	�
;�Xc�A�3*

loss2�:��xd       �	��;�Xc�A�3*

lossi��:�V6       �	VE<�Xc�A�3*

loss66�<j���       �	�&U�Xc�A�3*

lossq
�<�ј       �	�U�Xc�A�3*

loss��.<j.�       �	�YV�Xc�A�3*

loss]��<�D�       �	P�V�Xc�A�3*

lossC�<O��>       �	��W�Xc�A�3*

loss\�<c:��       �	g+X�Xc�A�3*

loss�5;�*�n       �	n�X�Xc�A�3*

loss���<�v�!       �	kY�Xc�A�3*

loss�=A��~       �	-Z�Xc�A�3*

loss�0=<�V�#       �	��Z�Xc�A�3*

loss0�:}Ed�       �	2<[�Xc�A�3*

loss-��;^�#�       �	��[�Xc�A�3*

loss/�}<�	��       �	*]�Xc�A�3*

loss'<<�	�4       �	ȶ]�Xc�A�3*

loss�<����       �	:W^�Xc�A�3*

loss!�(;�d�       �	c�^�Xc�A�3*

lossgX�:��Ů       �	U�_�Xc�A�3*

loss��:2؈
       �	d$`�Xc�A�3*

loss�<�!~       �	��`�Xc�A�3*

loss���;
>o       �	Yna�Xc�A�3*

loss{�I:KZmD       �	|b�Xc�A�3*

lossLK=��o/       �	Ͱb�Xc�A�3*

loss���<GbB       �	�Sc�Xc�A�3*

lossc|=Jv       �	��c�Xc�A�3*

loss�S_;���N       �	��d�Xc�A�3*

loss8�:�5��       �	�"e�Xc�A�3*

loss�N�<8(�
       �	�
f�Xc�A�3*

loss퀛:şC�       �	�f�Xc�A�3*

loss��<;.@y�       �	XXg�Xc�A�3*

loss�\;,r�       �	��g�Xc�A�3*

loss!��:m��S       �	��h�Xc�A�3*

loss/r�<NP�       �	�i�Xc�A�3*

loss�=�;6���       �	{�i�Xc�A�3*

loss�<�^�       �	O�j�Xc�A�3*

loss���<>s~3       �	�k�Xc�A�3*

loss]��<�u4�       �	�l�Xc�A�3*

lossS�;s�
�       �	K�l�Xc�A�3*

lossm�A=M���       �	}Am�Xc�A�3*

loss��<��-       �	��m�Xc�A�3*

loss���;&T�E       �	kn�Xc�A�3*

lossF�.;F~b�       �	so�Xc�A�3*

loss�Xh<��7       �	6Yp�Xc�A�3*

lossx��;S GV       �	��p�Xc�A�3*

loss@e<qƌ�       �	U�q�Xc�A�3*

loss���=��s�       �	��r�Xc�A�3*

loss �H=���       �	�Zs�Xc�A�3*

loss\�;V&D       �	��s�Xc�A�3*

loss�U1=��9       �	F�t�Xc�A�3*

loss.�<����       �	�*u�Xc�A�3*

loss��T;���       �	x�u�Xc�A�3*

loss-:���       �	S�v�Xc�A�3*

lossN��;<j��       �	&5w�Xc�A�3*

loss���9�Kg�       �	��w�Xc�A�3*

loss�~G<,8��       �	pwx�Xc�A�3*

loss{9���7       �	�y�Xc�A�3*

loss��:a"�       �	W�y�Xc�A�4*

loss��~;��l       �	�Oz�Xc�A�4*

loss���9�I�       �	7{�Xc�A�4*

loss�G�:Y��|       �	P�{�Xc�A�4*

loss�q=���6       �	[�|�Xc�A�4*

lossx��=�m��       �	b1}�Xc�A�4*

lossvk:�'��       �	y�}�Xc�A�4*

loss�:�;��$`       �	Zf~�Xc�A�4*

loss
)<�B�       �	��~�Xc�A�4*

loss�)�:���i       �	��Xc�A�4*

lossw�x<�;��       �	�`��Xc�A�4*

lossf~p;��       �	����Xc�A�4*

loss�U<�
��       �	㍁�Xc�A�4*

loss���;�э       �	�#��Xc�A�4*

lossCa�=���v       �	c���Xc�A�4*

lossld^=Rܦq       �	`��Xc�A�4*

loss\��;��L       �	o���Xc�A�4*

loss<X�<G(`�       �	����Xc�A�4*

loss���:�:�/       �	)��Xc�A�4*

loss�<�_;D       �	�ȅ�Xc�A�4*

lossJɘ:�v|       �	h\��Xc�A�4*

loss͢M<�B       �	����Xc�A�4*

lossq�C;�I�       �	Û��Xc�A�4*

loss��<�z��       �	�5��Xc�A�4*

loss(��;�,~�       �	Ԉ�Xc�A�4*

loss�_=��,5       �	�u��Xc�A�4*

loss�<$?�4       �	���Xc�A�4*

loss�;�ͼ       �	����Xc�A�4*

loss���:����       �	'M��Xc�A�4*

loss�9�:E?�9       �	���Xc�A�4*

loss�ۈ:~=:�       �		���Xc�A�4*

lossTz�;\���       �	`:��Xc�A�4*

loss��E<����       �	�ۍ�Xc�A�4*

loss]/;�kZ�       �	>v��Xc�A�4*

loss�:s�f       �	���Xc�A�4*

loss2@<���       �	����Xc�A�4*

loss�d=�6Q       �	�E��Xc�A�4*

loss�s�<����       �	ې�Xc�A�4*

lossq�<�QTe       �	=|��Xc�A�4*

loss2a8=�rE�       �	D��Xc�A�4*

loss3�I9A٥�       �	���Xc�A�4*

lossT�
<���       �	ڭ��Xc�A�4*

lossx��:���       �	D��Xc�A�4*

loss��;��$       �	�ؔ�Xc�A�4*

loss��!;�0|       �	an��Xc�A�4*

loss�I�:lϾ�       �	���Xc�A�4*

loss×c:���       �	
���Xc�A�4*

loss?[)<�ct�       �	�9��Xc�A�4*

loss�Ђ:�D�       �	,��Xc�A�4*

lossv, =
��j       �	����Xc�A�4*

lossL��9��]G       �	%>��Xc�A�4*

loss4c�9��8       �	=ԙ�Xc�A�4*

loss��:H�Q(       �	8���Xc�A�4*

loss��;��z7       �	�c��Xc�A�4*

loss&��:���H       �	���Xc�A�4*

lossO#o=϶�#       �	򵜁Xc�A�4*

loss�s=\�`�       �	�Z��Xc�A�4*

loss/͈:�Z��       �	;��Xc�A�4*

loss���<k�>�       �	���Xc�A�4*

loss#�{:96{       �	�:��Xc�A�4*

loss�SW:�dG]       �	ڟ�Xc�A�4*

loss�˃;{|       �	V��Xc�A�4*

lossT�=
���       �	���Xc�A�4*

lossQ�:�E��       �	c���Xc�A�4*

loss�-&=���       �	�T��Xc�A�4*

lossn�9���       �	k�Xc�A�4*

loss��<��)<       �	`���Xc�A�4*

loss�6s;��G       �	3��Xc�A�4*

loss�><ь,�       �	�פ�Xc�A�4*

loss� �:�(Ɠ       �	2w��Xc�A�4*

lossv�=��I:       �	���Xc�A�4*

loss�-�<8��       �	����Xc�A�4*

loss�<�Q�       �	dv��Xc�A�4*

lossJ4h<�6�       �	ao��Xc�A�4*

lossE��<_���       �	���Xc�A�4*

loss��F<&�jr       �	革�Xc�A�4*

loss�h%=���       �	 9��Xc�A�4*

lossE�9�D�       �	�ժ�Xc�A�4*

loss	�@=01�       �	ۇ��Xc�A�4*

lossq�C<j�'       �	kb��Xc�A�4*

loss(��<�>(       �	����Xc�A�4*

lossV<;9       �	S���Xc�A�4*

loss��:���       �	5��Xc�A�4*

lossM�<�~��       �	�㮁Xc�A�4*

loss��;�-Rn       �	@���Xc�A�4*

loss��m:��!�       �	6��Xc�A�4*

loss�x,< �[~       �	q㰁Xc�A�4*

loss��<�6$�       �	����Xc�A�4*

loss-W<��pG       �	�/��Xc�A�4*

lossq� =��q�       �	[Ҳ�Xc�A�4*

lossl��9��       �	mp��Xc�A�4*

lossCMK<� ��       �	�K��Xc�A�4*

loss���8�N�       �	촁Xc�A�4*

lossk�:uaTT       �	d���Xc�A�4*

loss�;r;�A�a       �	�7��Xc�A�4*

lossEϜ:�S�       �	eඁXc�A�4*

loss�ߝ<!0��       �	숷�Xc�A�4*

loss��=�bV       �	�x��Xc�A�4*

loss�Q�=���       �	���Xc�A�4*

lossA��<4[��       �	佹�Xc�A�4*

lossa�%;����       �	�`��Xc�A�4*

lossec:��P�       �	M���Xc�A�4*

loss/Y�<Eb       �	>��Xc�A�4*

loss^;K#�       �	�ݼ�Xc�A�4*

loss<@<ᨑ       �	1���Xc�A�4*

loss3��=�N�8       �	v���Xc�A�4*

loss8	�:D�#3       �	�1��Xc�A�4*

loss�,�<�	
�       �	!��Xc�A�4*

loss��:�u�z       �	����Xc�A�4*

loss�Ee;@ n       �	 |��Xc�A�4*

loss7�);��I       �	p#Xc�A�4*

loss�R<�w��       �	7ÁXc�A�4*

loss] <$��       �	}�ÁXc�A�4*

losswj�:5m�       �	��āXc�A�4*

loss�G;�H�        �	H1ŁXc�A�4*

loss#D5;�ښT       �	�ŁXc�A�4*

loss_�~8�č       �	�|ƁXc�A�4*

loss�e�8��J�       �	mǁXc�A�4*

loss�$�:(�J�       �	�ǁXc�A�4*

loss�>#���       �	�lȁXc�A�4*

lossR'�=�7�       �	�ɁXc�A�4*

loss�	=u\D       �	N�ɁXc�A�4*

loss��	<��!       �	�VʁXc�A�4*

loss�F�:��       �	��ʁXc�A�4*

lossUQ�:��O�       �	��ˁXc�A�4*

loss���7�K�Q       �	�*́Xc�A�4*

loss�C<Q��       �	O�́Xc�A�4*

loss���8 ��a       �	úXc�A�4*

loss�P9&�_       �	�΁Xc�A�5*

loss�49�`V�       �	{�΁Xc�A�5*

loss�8�9��Ph       �	�eρXc�A�5*

loss�D=��       �	��ρXc�A�5*

loss%)<�G�       �	R�ЁXc�A�5*

lossHJ?<�;�P       �	�9сXc�A�5*

loss�= ;�s7�       �	8�ҁXc�A�5*

loss��b;��       �	�qӁXc�A�5*

loss�h�:_��x       �	�/ԁXc�A�5*

loss�=h9���F       �	�ԁXc�A�5*

loss�=�Z�	       �	�iՁXc�A�5*

loss��|9�S�       �	�ցXc�A�5*

loss�M'=h��       �	e�ցXc�A�5*

loss�]D:�T��       �	�MׁXc�A�5*

loss�/n<D��       �	��ׁXc�A�5*

loss��<�yk<       �	u�؁Xc�A�5*

lossѣ�;�	�       �	)%فXc�A�5*

loss�J�:G�M�       �	��فXc�A�5*

loss	��9F*v�       �	iSځXc�A�5*

lossF�V8EL��       �	!�ځXc�A�5*

loss�;��U?       �	؂ہXc�A�5*

loss[B�;�n�p       �	6܁Xc�A�5*

loss�v�:���       �	A�܁Xc�A�5*

loss֎�<��       �	&V݁Xc�A�5*

loss��;@1�       �	��݁Xc�A�5*

lossJO�<B���       �	}�ށXc�A�5*

loss�N�;��ђ       �	P:߁Xc�A�5*

lossR��:�ѹ6       �	��߁Xc�A�5*

loss�/;=�0�b       �	h��Xc�A�5*

loss�,�86�`       �	 �Xc�A�5*

lossq;lޙH       �	��Xc�A�5*

loss���:9c��       �	L��Xc�A�5*

loss�U�<�m       �	pA�Xc�A�5*

lossJ�<ʔ�6       �	 ��Xc�A�5*

loss�� =3�˪       �	"��Xc�A�5*

loss�*<�y�(       �	9%�Xc�A�5*

loss���<�H�       �	��Xc�A�5*

lossD��<�DO�       �	@j�Xc�A�5*

lossCm�8��       �	V�Xc�A�5*

lossc|;�F��       �	���Xc�A�5*

loss�C;@3{       �	�F�Xc�A�5*

lossS&;@X�       �	��Xc�A�5*

loss$� <^h��       �	���Xc�A�5*

loss�v:kk�       �	�^�Xc�A�5*

loss���;���       �	Y��Xc�A�5*

loss(�6=K�y*       �	�/�Xc�A�5*

loss�X%<9��t       �	���Xc�A�5*

loss0;#�Mn       �	�|�Xc�A�5*

losslJ�8�s�e       �	��Xc�A�5*

loss{�;�${)       �	ݶ�Xc�A�5*

loss$4�:�T�       �	SY�Xc�A�5*

lossX�1</��6       �	���Xc�A�5*

lossԽ�<�UІ       �	����Xc�A�5*

loss��<�b�-       �	R,�Xc�A�5*

lossƒ�<_o1       �	���Xc�A�5*

loss�=�Eb�       �	[�Xc�A�5*

loss�y�;�/I       �	���Xc�A�5*

loss�&u<j�S_       �	��Xc�A�5*

loss|�(;x}�       �	�`�Xc�A�5*

loss�!K:C��       �	D��Xc�A�5*

loss�~a=/8W       �	%���Xc�A�5*

loss���:[؅       �	�.��Xc�A�5*

loss
T<(�F       �	����Xc�A�5*

losso8[:���v       �	�g��Xc�A�5*

lossW�M=���       �	�5��Xc�A�5*

loss�^�;z�j�       �	����Xc�A�5*

loss�%�;YY�Z       �	�d��Xc�A�5*

loss��t9�Q�       �	"���Xc�A�5*

loss��:Eb.       �	����Xc�A�5*

loss��<#5�       �	����Xc�A�5*

lossM�j:j 0�       �	�m��Xc�A�5*

loss_�.;D/�|       �	�F��Xc�A�5*

loss ��:1�.       �	�<��Xc�A�5*

loss?�	<΂m�       �	~��Xc�A�5*

loss:��<Hz��       �	[���Xc�A�5*

loss>U�:{31       �	�L �Xc�A�5*

loss�~6=rԓ�       �	� �Xc�A�5*

loss�u/;�Gb�       �	$��Xc�A�5*

loss1�1:�\�+       �	\>�Xc�A�5*

loss|�x=�v       �	w��Xc�A�5*

loss�M=NLԕ       �	~t�Xc�A�5*

loss��=~KD|       �	��Xc�A�5*

loss�O�9<�;       �	
��Xc�A�5*

loss��4=~��Q       �	$B�Xc�A�5*

loss�;�:�E�K       �	
��Xc�A�5*

loss�f�;w�y�       �	ms�Xc�A�5*

loss{M<�o(�       �	F�Xc�A�5*

loss��;��       �	Q��Xc�A�5*

loss��;��>�       �	�?�Xc�A�5*

loss*�X:]�e�       �	���Xc�A�5*

loss���=F2�\       �	�n	�Xc�A�5*

loss$؜;���       �	�
�Xc�A�5*

loss��:�q�       �	��
�Xc�A�5*

loss9 �9���       �	�=�Xc�A�5*

loss�;�9q�*�       �	���Xc�A�5*

loss<��;�@��       �	���Xc�A�5*

loss?��<����       �	�I�Xc�A�5*

loss8CY9I�b       �	.��Xc�A�5*

loss��>9$��'       �	��Xc�A�5*

loss�ht<I���       �	���Xc�A�5*

loss�4�;���       �	�H�Xc�A�5*

lossag=	rM�       �	]��Xc�A�5*

loss���:�I1       �	Ҧ�Xc�A�5*

loss��:<�u!       �	]�Xc�A�5*

loss׍�9���+       �	�o�Xc�A�5*

loss���:�Wj       �	(+�Xc�A�5*

loss��/:��       �	���Xc�A�5*

lossdyA:�%>
       �	���Xc�A�5*

lossV�;;Kd�|       �	��Xc�A�5*

loss,\~9�N;       �	���Xc�A�5*

loss�ɂ: OZ�       �	��Xc�A�5*

lossW��:t��       �	.��Xc�A�5*

loss��G;p�       �	]m�Xc�A�5*

lossT��<�o$C       �	;��Xc�A�5*

lossƈZ;�1       �	#1�Xc�A�5*

lossNָ<z��       �	���Xc�A�5*

loss�k�<t���       �	8��Xc�A�5*

loss3��<#�       �	h�Xc�A�5*

loss3$5=<���       �	H1�Xc�A�5*

lossr�><
�       �	�{ �Xc�A�5*

loss)��=�O�?       �	A)!�Xc�A�5*

lossi0�<�oj:       �	7�!�Xc�A�5*

loss��p;���       �	��"�Xc�A�5*

lossÌ�=��dB       �	��#�Xc�A�5*

lossG��=�Q��       �	�$�Xc�A�5*

loss��=-       �	m�%�Xc�A�5*

loss6�f:;��       �	J�&�Xc�A�5*

lossQ7<yu�u       �	>='�Xc�A�5*

lossI7,;�Z�:       �	��'�Xc�A�6*

loss���<�3j       �	�)�Xc�A�6*

loss���9�-�       �	5*�Xc�A�6*

loss�J=5��       �	�*�Xc�A�6*

loss��;��Ma       �	��+�Xc�A�6*

loss���<�k��       �	��,�Xc�A�6*

loss�H�<Z\��       �	+P-�Xc�A�6*

loss���;JYG8       �	�.�Xc�A�6*

loss�;�P��       �	�.�Xc�A�6*

loss���:�$�        �	&9/�Xc�A�6*

loss�և;y�       �	��/�Xc�A�6*

loss�NF<0���       �	3k0�Xc�A�6*

loss�A�92i��       �	M1�Xc�A�6*

loss�)�;)r��       �	��1�Xc�A�6*

loss1�<࿈       �	�2�Xc�A�6*

loss��<�       �	{�3�Xc�A�6*

loss4+y<�W�       �	p4�Xc�A�6*

loss��;��f�       �	�
5�Xc�A�6*

loss��,=�x�       �	�6�Xc�A�6*

loss*�<�7%       �	�6�Xc�A�6*

loss���<�>l       �	 _7�Xc�A�6*

loss�2;�J�h       �	p	8�Xc�A�6*

loss�1+=j�<       �	9�8�Xc�A�6*

lossv��<��*�       �	�39�Xc�A�6*

lossz�#<[�!L       �	:�9�Xc�A�6*

lossrE<���       �	U�:�Xc�A�6*

loss[gE:�7!�       �	�7;�Xc�A�6*

loss��=J��,       �	K�;�Xc�A�6*

loss��;����       �	�`<�Xc�A�6*

loss�d�8���       �	��<�Xc�A�6*

lossᤡ;h��|       �	>�=�Xc�A�6*

loss���:��P�       �	8f>�Xc�A�6*

loss� ;�       �	`?�Xc�A�6*

lossaD=W�|       �	,�?�Xc�A�6*

loss ߹<߿3�       �	3@�Xc�A�6*

loss��<+���       �	.�@�Xc�A�6*

loss�R�<��Ψ       �	�cA�Xc�A�6*

loss�u
=��       �	��A�Xc�A�6*

lossDG�;�:F       �	G�B�Xc�A�6*

loss�c;}�{       �	�%C�Xc�A�6*

loss�q,<{�b�       �	\�C�Xc�A�6*

loss��<�       �	�bD�Xc�A�6*

lossS�/;:�{u       �	f�D�Xc�A�6*

loss��P;��؍       �	ڏE�Xc�A�6*

loss�_]:���       �	�ZF�Xc�A�6*

loss�6>:�$�p       �	^�F�Xc�A�6*

loss�8�=�Ѧ]       �	3�G�Xc�A�6*

loss;�q:���       �	6 H�Xc�A�6*

lossC�<��        �	�H�Xc�A�6*

loss�)=Vfw'       �	�_I�Xc�A�6*

loss��:F�,)       �	��I�Xc�A�6*

loss���:���       �	A�J�Xc�A�6*

loss���;_�       �	�xK�Xc�A�6*

loss�+;g���       �	�L�Xc�A�6*

loss��=pɈ       �	��L�Xc�A�6*

loss��<j�"o       �	S]M�Xc�A�6*

loss���;���P       �	��M�Xc�A�6*

loss�T�<���       �	
�N�Xc�A�6*

loss��8<���       �	�8O�Xc�A�6*

loss�{g;�b=       �	%�O�Xc�A�6*

lossP <�Q%�       �	��P�Xc�A�6*

loss��;޷�T       �	�8Q�Xc�A�6*

lossl.�:�0��       �	��Q�Xc�A�6*

loss�^V=��/       �	��R�Xc�A�6*

loss�T1<�c�}       �	rS�Xc�A�6*

lossn��=2(�       �	�S�Xc�A�6*

loss�S==��^t       �	�T�Xc�A�6*

loss�<b�+Q       �	)YU�Xc�A�6*

lossH8�;����       �	A�U�Xc�A�6*

loss�t*<@�!       �	/�V�Xc�A�6*

loss!��:Ҭ       �	g(W�Xc�A�6*

loss���=�D+       �	��W�Xc�A�6*

loss��v:)_��       �	��X�Xc�A�6*

loss�D�8�u9�       �	�/Y�Xc�A�6*

loss =����       �	�Y�Xc�A�6*

lossԆ=�j@�       �	�aZ�Xc�A�6*

loss�^<�nB$       �	�[�Xc�A�6*

loss� s9DiT�       �	ϟ[�Xc�A�6*

lossV�:ο<�       �	�=\�Xc�A�6*

loss���<_l�       �	2�\�Xc�A�6*

loss�<겐�       �	/�]�Xc�A�6*

lossl݊:,M��       �	['^�Xc�A�6*

loss��;�ե       �	��^�Xc�A�6*

loss�!�:�{&       �	�[_�Xc�A�6*

lossh�b<E�2}       �	
�_�Xc�A�6*

loss哃8a\4�       �	��`�Xc�A�6*

loss.�<;|��p       �	�a�Xc�A�6*

lossX<x:��*       �	��a�Xc�A�6*

lossH��;��h7       �	DMb�Xc�A�6*

loss-�:zHǷ       �	6�b�Xc�A�6*

lossu�< ��       �	Fyc�Xc�A�6*

loss���;Q��j       �	d�Xc�A�6*

loss��:ʬ�       �	�d�Xc�A�6*

lossRc<�j?�       �	WAe�Xc�A�6*

loss[�J=�ȇ       �	�e�Xc�A�6*

loss�=<wN       �	dxf�Xc�A�6*

loss\��:��       �	�g�Xc�A�6*

loss��R<���
       �	�g�Xc�A�6*

loss�{=�jH       �	�Qh�Xc�A�6*

loss�f/;�V��       �	l�h�Xc�A�6*

loss)�<��Tf       �	�i�Xc�A�6*

loss�L><����       �	0,j�Xc�A�6*

lossZE <{IN�       �	��j�Xc�A�6*

loss��/<�a��       �		pk�Xc�A�6*

lossҀ9=�t�       �	nl�Xc�A�6*

loss���:[X��       �	ݳl�Xc�A�6*

loss�=�:��G       �	�Ro�Xc�A�6*

loss���=p��Q       �	m p�Xc�A�6*

lossM[<gH��       �	�p�Xc�A�6*

loss��9V��z       �	fJq�Xc�A�6*

lossz@�;҅|�       �	�Dr�Xc�A�6*

loss��b:~.U�       �	!�r�Xc�A�6*

loss��=P���       �	ܸs�Xc�A�6*

loss�[$=��NA       �	~St�Xc�A�6*

loss��`<�i�o       �	��t�Xc�A�6*

loss��I<B8       �	t�u�Xc�A�6*

loss���;��p       �	�9v�Xc�A�6*

loss{�P:��       �	��v�Xc�A�6*

loss�R�:�L �       �	��w�Xc�A�6*

lossA�=`�R�       �	�y�Xc�A�6*

loss 9;�#�       �	=�y�Xc�A�6*

lossf��;#(�       �	�>z�Xc�A�6*

loss�V(<}`�o       �	k{�Xc�A�6*

loss�p;S���       �	��{�Xc�A�6*

loss>�=�ۺ       �	�|�Xc�A�6*

lossA�r;e�,�       �	-@}�Xc�A�6*

loss��=)M��       �	��}�Xc�A�6*

loss�A�;.m�       �	z�~�Xc�A�6*

loss�<D��       �	F�Xc�A�7*

loss�T�<�\�V       �	u��Xc�A�7*

loss�k;<�i       �	g{��Xc�A�7*

lossǫ;i�N�       �	���Xc�A�7*

loss(�K<��s4       �	�ȁ�Xc�A�7*

lossQ[;%���       �	�f��Xc�A�7*

lossHy<B�)F       �	���Xc�A�7*

loss+��;l��       �	'���Xc�A�7*

lossʗ;�E=P       �	�߄�Xc�A�7*

lossv�i<����       �	�v��Xc�A�7*

loss��<��       �	���Xc�A�7*

loss���<�qź       �	����Xc�A�7*

lossxRa:��       �	�?��Xc�A�7*

loss���:B�Tj       �	|Ӈ�Xc�A�7*

lossx�;~��C       �	�n��Xc�A�7*

loss�S<l��d       �	v��Xc�A�7*

loss��<;�m�       �	����Xc�A�7*

loss(9;���       �	�R��Xc�A�7*

loss��^=uw�       �	�슂Xc�A�7*

loss�ʝ;Ď�U       �	Ǆ��Xc�A�7*

loss���<.'�       �	���Xc�A�7*

loss}�H<�RFk       �	���Xc�A�7*

lossr�T:��@       �	�E��Xc�A�7*

loss�2�:4v�S       �	�ꍂXc�A�7*

loss@ii=�
��       �	���Xc�A�7*

loss��t;�,_       �	�%��Xc�A�7*

loss�:ζm       �	ܻ��Xc�A�7*

loss��:;u�       �	QM��Xc�A�7*

loss��<;6%!�       �	Pߐ�Xc�A�7*

lossx1�:sv�       �	u��Xc�A�7*

losss�9��M�       �	/��Xc�A�7*

lossR�}<N!d'       �	9���Xc�A�7*

loss��<��       �	IL��Xc�A�7*

loss=\?;^��Q       �	�ޓ�Xc�A�7*

loss@3L=��       �	s��Xc�A�7*

loss�U�9e�sj       �	���Xc�A�7*

loss���:m�aR       �	����Xc�A�7*

lossP��:��9       �	�/��Xc�A�7*

loss�a=�|�       �	��Xc�A�7*

loss��#<��"       �	�Z��Xc�A�7*

loss��B<��       �	^���Xc�A�7*

loss�:#;X(ԩ       �	撘�Xc�A�7*

loss���<���       �	.:��Xc�A�7*

loss���;���*       �	�ޙ�Xc�A�7*

loss�r=J��       �	r��Xc�A�7*

loss�<D��       �	�	��Xc�A�7*

loss�'�:�Q�k       �	����Xc�A�7*

loss��
;����       �	�A��Xc�A�7*

lossX�X<3*M4       �	x휂Xc�A�7*

loss���:�r       �	����Xc�A�7*

loss���96o�	       �	�A��Xc�A�7*

loss��s<0�@       �	�ꞂXc�A�7*

losst1�:*_R       �	͐��Xc�A�7*

loss�3�9�2GX       �	7��Xc�A�7*

lossn-'9^��       �	NҠ�Xc�A�7*

loss�t�9���X       �	vq��Xc�A�7*

loss�v	<�:52       �	���Xc�A�7*

lossO�=���       �	o���Xc�A�7*

loss<];�
Jg       �	wg��Xc�A�7*

loss��V<��I+       �	W��Xc�A�7*

loss���;�u�s       �	����Xc�A�7*

loss%<U5�       �	�R��Xc�A�7*

loss���;2(�~       �	BB��Xc�A�7*

loss��<���       �	���Xc�A�7*

loss؆</��       �	鸧�Xc�A�7*

loss��C;�~�.       �	�Z��Xc�A�7*

loss��<%Ӣ�       �	 7��Xc�A�7*

loss{�;�v3X       �	�ک�Xc�A�7*

loss
,;��Z       �	M���Xc�A�7*

loss�:p��F       �	)��Xc�A�7*

loss��<U�#>       �	�ϫ�Xc�A�7*

loss��:�y6�       �	�d��Xc�A�7*

loss/�: ��       �	���Xc�A�7*

loss�'P<�b�       �	/���Xc�A�7*

loss\E�;���U       �	�?��Xc�A�7*

lossh.�<�m�       �	|Ү�Xc�A�7*

lossq2�<c%�       �	����Xc�A�7*

lossX�+=�4@       �	*��Xc�A�7*

loss�<ui,       �	W���Xc�A�7*

loss$�Q:]�^       �	'K��Xc�A�7*

loss�<��p�       �	���Xc�A�7*

lossܒ�9�=       �	*���Xc�A�7*

lossSΟ<.Sn�       �	�X��Xc�A�7*

loss)�8���q       �	��Xc�A�7*

loss;�I}v       �	ׇ��Xc�A�7*

loss<�	z�       �	���Xc�A�7*

loss��;�N=�       �	﵂Xc�A�7*

loss���9jf�       �	����Xc�A�7*

lossϪ�;�+�,       �		4��Xc�A�7*

loss�3<�*]       �	Kͷ�Xc�A�7*

loss��j<row�       �	qq��Xc�A�7*

lossr|�;x���       �	O��Xc�A�7*

loss�U�;����       �	����Xc�A�7*

lossZ�O8y��|       �	쾺�Xc�A�7*

loss���5�(       �	KW��Xc�A�7*

loss;a;>�b       �	Z���Xc�A�7*

loss��:3�:        �	y���Xc�A�7*

loss�(�9c�|%       �	6:��Xc�A�7*

lossfL�9�b)�       �	|ֽ�Xc�A�7*

loss�F�8��48       �	;�Xc�A�7*

loss^9<�0i       �	ep��Xc�A�7*

loss�'i95ښ�       �	c��Xc�A�7*

loss��u9�p�Z       �	����Xc�A�7*

loss�#5:[(�       �	�C��Xc�A�7*

loss|XR;�i%Y       �	����Xc�A�7*

loss�:�NvW       �	��Xc�A�7*

loss��:�T�       �	m�ÂXc�A�7*

loss��7�2�       �	�łXc�A�7*

loss� ;@�4�       �	ӢłXc�A�7*

loss���=�,+       �	�HƂXc�A�7*

loss���:Q�,]       �	L�ƂXc�A�7*

loss���=<))�       �	ȚǂXc�A�7*

loss�{d<��G       �	�1ȂXc�A�7*

loss��<�P6�       �	�ȂXc�A�7*

lossmX;�Z�X       �	�[ɂXc�A�7*

lossTy;t/'       �	�ʂXc�A�7*

loss�?�9���%       �	ʥʂXc�A�7*

lossjd-<<���       �	fK˂Xc�A�7*

loss���<fCd�       �	��˂Xc�A�7*

lossQ$u8���P       �	3�̂Xc�A�7*

lossN;��)       �	�&͂Xc�A�7*

loss��9Y�L       �	/�͂Xc�A�7*

loss(�<�2L2       �	TV΂Xc�A�7*

loss�L:;�^��       �	��΂Xc�A�7*

lossuq!;zҘ�       �	'�ςXc�A�7*

loss,y,=��:       �	HЂXc�A�7*

loss�=+�F�       �	�ЂXc�A�7*

loss{��9�A��       �	Y�тXc�A�7*

loss��;��g       �	�*҂Xc�A�8*

loss,�F:���r       �	bӂXc�A�8*

loss���9��`       �	�ԂXc�A�8*

loss<F;����       �	/ՂXc�A�8*

loss�a�=
��       �	h�ՂXc�A�8*

loss�I;����       �	,HւXc�A�8*

lossb�:)^�G       �	��ׂXc�A�8*

loss�ޅ:1�*m       �	�؂Xc�A�8*

loss�^�;��p�       �	��؂Xc�A�8*

loss�v�<Q"�
       �	�YقXc�A�8*

loss��	<�i7g       �	�0ڂXc�A�8*

loss��<�=�       �	�ڂXc�A�8*

lossO&;�7o       �	�{ۂXc�A�8*

loss��:�-�q       �	�܂Xc�A�8*

loss-a[9�.       �	[�܂Xc�A�8*

loss;::a�~�       �	�]݂Xc�A�8*

loss
L�<W�1�       �	��݂Xc�A�8*

loss�n�:W[ |       �	��ނXc�A�8*

lossd=�i�T       �	B�߂Xc�A�8*

lossW�;�T,       �	)=��Xc�A�8*

loss��K=����       �	���Xc�A�8*

lossd/�<��       �	V�Xc�A�8*

loss�C~:�e�g       �	��Xc�A�8*

loss���:���       �	���Xc�A�8*

loss��b:�/��       �	���Xc�A�8*

lossC�";��lf       �	�p�Xc�A�8*

loss.��<�-ҡ       �	��Xc�A�8*

loss��=�l"       �	ȱ�Xc�A�8*

loss?�<�QS�       �	���Xc�A�8*

loss��;z�       �	�(�Xc�A�8*

loss�8;��x       �	7��Xc�A�8*

loss�&38U��       �	fh�Xc�A�8*

lossڶ :����       �	T�Xc�A�8*

losso�:h�