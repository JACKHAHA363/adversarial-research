       �K"	  @"fc�Abrain.Event:2�O4"�     �])�	�dm"fc�A"��
^
dataPlaceholder*
dtype0*
shape: */
_output_shapes
:���������
W
labelPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

h
conv2d_1_inputPlaceholder*
dtype0*
shape: */
_output_shapes
:���������
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
`
conv2d_1/random_uniform/minConst*
valueB
 *�x�*
dtype0*
_output_shapes
: 
`
conv2d_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�x=
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
dtype0*
seed���)*
T0*&
_output_shapes
:@*
seed2��L
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
	container *
dtype0*
shared_name *
shape:@
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
VariableV2*
shared_name *
dtype0*
shape:@*
_output_shapes
:@*
	container 
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
t
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
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
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
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
seed2���*
dtype0*
T0*
seed���)
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
	container *
dtype0*
shared_name *
shape:@@
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
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
t
conv2d_2/bias/readIdentityconv2d_2/bias*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
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
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:���������@
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
a
dropout_1/keras_learning_phasePlaceholder*
_output_shapes
:*
dtype0
*
shape: 
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
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
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
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2���
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
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*
T0*/
_output_shapes
:���������@
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
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
T0*
new_axis_mask *
_output_shapes
:*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
end_mask
Y
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
\
flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
_output_shapes
:*
T0*
N
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
T0
m
dense_1/random_uniform/shapeConst*
valueB" d  �   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�3z�
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
seed2��#*
dtype0*
T0*
seed���)
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
VariableV2*
shared_name *
dtype0*
shape:���*!
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
dense_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
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
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_output_shapes	
:�*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
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
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
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
 *  �?*
_output_shapes
: *
dtype0
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
T0*
N
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   
   
_
dense_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U�
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *̈́U>
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
VariableV2*
shared_name *
dtype0*
shape:	�
*
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
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
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
VariableV2*
shared_name *
dtype0*
shape:
*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
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
/sequential_1/conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_2/BiasAddBiasAdd!sequential_1/conv2d_2/convolutionconv2d_2/bias/read*/
_output_shapes
:���������@*
data_formatNHWC*
T0
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
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu*
T0
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@
�
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2���
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
,sequential_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
sequential_1/flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
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
&sequential_1/dropout_2/cond/mul/SwitchSwitchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
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
seed2���
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
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
T0
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
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
*
data_formatNHWC*
T0
b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*'
_output_shapes
:���������
*
T0
[
num_inst/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
num_inst
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
a
num_inst/readIdentitynum_inst*
T0*
_output_shapes
: *
_class
loc:@num_inst
^
num_correct/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
o
num_correct
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
�
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
use_locking(
j
num_correct/readIdentitynum_correct*
T0*
_output_shapes
: *
_class
loc:@num_correct
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
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
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
�
AssignAssignnum_instConst_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_inst
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
softmax_cross_entropy_loss/RankConst*
_output_shapes
: *
dtype0*
value	B :
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
%softmax_cross_entropy_loss/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
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
&softmax_cross_entropy_loss/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
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
!softmax_cross_entropy_loss/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_2Shapelabel*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*
T0*
_output_shapes
:*

axis 
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
j
(softmax_cross_entropy_loss/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
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
"softmax_cross_entropy_loss/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
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
T0*#
_output_shapes
:���������*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
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
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
_output_shapes
: *
T0
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
T0*
_output_shapes
:*
out_type0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
T0*
_output_shapes
:*
out_type0
d
"softmax_cross_entropy_loss_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
�
 softmax_cross_entropy_loss_1/SubSub#softmax_cross_entropy_loss_1/Rank_1"softmax_cross_entropy_loss_1/Sub/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*
N*
T0*
_output_shapes
:*

axis 
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
,softmax_cross_entropy_loss_1/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
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
T0*
_output_shapes
:*
out_type0
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
*softmax_cross_entropy_loss_1/Slice_1/beginPack"softmax_cross_entropy_loss_1/Sub_1*
N*
T0*
_output_shapes
:*

axis 
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
.softmax_cross_entropy_loss_1/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
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
$softmax_cross_entropy_loss_1/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
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
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
N*
T0*
_output_shapes
:*

axis 
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
value	B : *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2*
T0*
_output_shapes
:*
out_type0

=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
0softmax_cross_entropy_loss_1/num_present/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
8softmax_cross_entropy_loss_1/num_present/ones_like/ShapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
�
ksoftmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ShapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Jsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successl^softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
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
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
"softmax_cross_entropy_loss_1/valueSelect$softmax_cross_entropy_loss_1/Greater softmax_cross_entropy_loss_1/div'softmax_cross_entropy_loss_1/zeros_like*
_output_shapes
: *
T0
P
Placeholder_1Placeholder*
dtype0*
shape: *
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
7gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
_output_shapes
: *
Tshape0*
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
3gradients/softmax_cross_entropy_loss_1/div_grad/mulMulJgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss_1/div_grad/RealDiv_2*
T0*
_output_shapes
: 
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
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape*
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
9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectSelect"softmax_cross_entropy_loss_1/EqualJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
Cgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss_1/Select_grad/Select<^gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
�
Kgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss_1/Select_grad/SelectD^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select*
T0
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
_output_shapes
: *N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
T0
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
dtype0*
_output_shapes
: *
valueB 
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
Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
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
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape*#
_output_shapes
:���������
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
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
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1SumOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/mul_1agradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
�
Zgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_depsNoOpR^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ReshapeT^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
bgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityQgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape[^gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/group_deps*
T0*
_output_shapes
: *d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape
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
=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeReshapeHgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
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
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mulMul?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDimsDgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradient*
T0*0
_output_shapes
:������������������
~
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
out_type0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
gradients/div_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
T0*'
_output_shapes
:���������
*
Tshape0
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
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape
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
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_2_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape
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
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
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
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
�
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
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
gradients/zeros/ConstConst*
dtype0*
_output_shapes
: *
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
_output_shapes
:*
out_type0*
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1Sum<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/mul_1Ngradients/sequential_1/dropout_2/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:����������*
Tshape0*
T0
�
Ggradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape
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
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*
T0
�
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1
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
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*(
_output_shapes
:����������*
T0
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1*
T0
�
gradients/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
_output_shapes
:*
out_type0
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
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
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
Fgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:�*J
_class@
><loc:@gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
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
T0*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul
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
@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1*/
_output_shapes
:���������@*
Tshape0*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*/
_output_shapes
:���������@
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
T0*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
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
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape
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
gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
out_type0*
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
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
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
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad
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
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"      @   @   
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
T0*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput
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
:@*
data_formatNHWC*
T0
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
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4������������������������������������
�
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         @   
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
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
shared_name *
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
dtype0*
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
beta2_power/readIdentitybeta2_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
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
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
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
shared_name *
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
dtype0*
	container 
�
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
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
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
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
zeros_5Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
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
conv2d_2/bias/Adam/AssignAssignconv2d_2/bias/Adamzeros_6*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
~
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@
T
zeros_7Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
conv2d_2/bias/Adam_1
VariableV2*
_output_shapes
:@*
dtype0*
shape:@*
	container * 
_class
loc:@conv2d_2/bias*
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
zeros_8Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
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
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*!
_output_shapes
:���*
T0
b
zeros_9Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
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
dense_1/bias/Adam/AssignAssigndense_1/bias/Adamzeros_10*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel*
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
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
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
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
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
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "��,     nKqz	4�p"fc�AJ��
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
dtype0*
seed���)*
T0*&
_output_shapes
:@*
seed2��L
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
	container *
dtype0*
shared_name *
shape:@
�
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@*
T0
[
conv2d_1/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
y
conv2d_1/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
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
conv2d_1/bias/readIdentityconv2d_1/bias*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
T0
s
conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
s
"conv2d_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
paddingVALID*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
T0*/
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
conv2d_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2���*
dtype0*
T0*
seed���)
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
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
:���������@*
data_formatNHWC*
T0
e
activation_2/ReluReluconv2d_2/BiasAdd*/
_output_shapes
:���������@*
T0
a
dropout_1/keras_learning_phasePlaceholder*
dtype0
*
shape: *
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
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2���
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
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
T0
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
g
flatten_1/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask 
Y
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
:���*
seed2��#
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
:����������*
seed2���*
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
seed2���
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
valueB:*
_output_shapes
:*
dtype0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
 *   @*
dtype0*
_output_shapes
: 
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
���������*
dtype0*
_output_shapes
:
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
value	B :*
_output_shapes
: *
dtype0
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
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*/
_output_shapes
:���������@*
T0
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
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
T
zeros_3Const*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
l
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
conv2d_2/kernel/Adam_1/readIdentityconv2d_2/kernel/Adam_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
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
	container *
dtype0* 
_class
loc:@conv2d_2/bias*
shared_name *
_output_shapes
:@*
shape:@
�
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0
b
zeros_8Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
�
dense_1/kernel/Adam
VariableV2*
	container *
dtype0*!
_class
loc:@dense_1/kernel*
shared_name *!
_output_shapes
:���*
shape:���
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
zeros_9Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
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
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
W
zeros_11Const*
valueB�*    *
dtype0*
_output_shapes	
:�
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
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
shared_name *
shape:	�
*
_output_shapes
:	�
*!
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
dtype0*
_output_shapes
:	�
*
valueB	�
*    
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
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_output_shapes
:
*
_class
loc:@dense_2/bias*
T0
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

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
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
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking( 
�
#Adam/update_conv2d_2/bias/ApplyAdam	ApplyAdamconv2d_2/biasconv2d_2/bias/Adamconv2d_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
Adam/beta1
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
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
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_1
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
Adam/beta1&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking( 
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
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0"V
lossesL
J
"softmax_cross_entropy_loss/value:0
$softmax_cross_entropy_loss_1/value:0��w;       ��-	ak�"fc�A*

loss]@`�n       ��-	v�"fc�A*

loss��@���g       ��-	�x�"fc�A*

loss��@����       ��-	CX�"fc�A*

loss�p�?�p�       ��-	���"fc�A*

loss���?t��A       ��-	��"fc�A*

loss߮�?�i�       ��-	��"fc�A*

loss4��?���K       ��-	c��"fc�A*

loss2�?�#֞       ��-	D��"fc�A	*

loss��z?���_       ��-	�˫"fc�A
*

loss?}i?.���       ��-	8��"fc�A*

lossfS?v��       ��-	�s�"fc�A*

lossE�6?�L��       ��-	��"fc�A*

loss�QJ?�'J�       ��-	�ʰ"fc�A*

loss��"?���       ��-	���"fc�A*

loss��H?�Z�5       ��-	iT�"fc�A*

loss��>���(       ��-	*�"fc�A*

lossߚ0?���       ��-	�ܳ"fc�A*

loss8�?meR        ��-	A��"fc�A*

loss�;$?gڛ�       ��-	p�"fc�A*

loss<~P?�LcE       ��-	���"fc�A*

loss
-;?��2�       ��-	�P�"fc�A*

lossM�K?婺�       ��-	���"fc�A*

loss�Hw?��d       ��-	���"fc�A*

loss�z-?���g       ��-	u:�"fc�A*

loss6+
?~6       ��-	|�"fc�A*

loss�=�>0zWx       ��-	 ��"fc�A*

lossNZ!?D��       ��-	�1�"fc�A*

loss�W-?O���       ��-	λ"fc�A*

loss��9?Đ�       ��-	�z�"fc�A*

lossJN?P�~       ��-	a�"fc�A*

loss� �>ۥ�N       ��-	���"fc�A *

lossLI�>�v�z       ��-	�\�"fc�A!*

loss� ?mS"7       ��-	+��"fc�A"*

lossײַ>ź�       ��-	5��"fc�A#*

loss���>g��D       ��-	�>�"fc�A$*

loss5�>4�נ       ��-	X��"fc�A%*

loss��L?JW݉       ��-	��"fc�A&*

loss`�9?�       ��-	���"fc�A'*

loss�?8L��       ��-	;o�"fc�A(*

loss>@?Pi`       ��-	W�"fc�A)*

lossw��>��ѷ       ��-	��"fc�A**

lossn��>��n&       ��-	 �"fc�A+*

lossn��>�7�       ��-	���"fc�A,*

loss��?X>��       ��-	�r�"fc�A-*

loss"-?@�v�       ��-	U�"fc�A.*

loss1��>�l�       ��-	k��"fc�A/*

loss�}�>�       ��-	?W�"fc�A0*

loss��>��k�       ��-	��"fc�A1*

loss!�>��}�       ��-	v�"fc�A2*

loss�r�>>;-�       ��-	��"fc�A3*

loss�}�>T)��       ��-	ܻ�"fc�A4*

loss"�>�tD.       ��-	�X�"fc�A5*

loss��n?XRS       ��-	���"fc�A6*

losszQ>���       ��-	ɓ�"fc�A7*

loss�)>�d��       ��-	*�"fc�A8*

loss@�P>��       ��-	%�"fc�A9*

loss8T	>,.(s       ��-	<��"fc�A:*

loss�x�>L�x       ��-	�@�"fc�A;*

loss,B�>��C�       ��-	��"fc�A<*

lossF�>,�{"       ��-	���"fc�A=*

loss_�S>A�M       ��-	�g�"fc�A>*

loss�1>1;&�       ��-	w�"fc�A?*

loss�:�>k��(       ��-	���"fc�A@*

lossmS�>���       ��-	�]�"fc�AA*

lossr�D>���?       ��-	���"fc�AB*

loss#�>�c(       ��-	{��"fc�AC*

lossƩ�>`��F       ��-	`>�"fc�AD*

lossd�>lK��       ��-	��"fc�AE*

loss���>���       ��-	LS�"fc�AF*

loss���>nrΊ       ��-	���"fc�AG*

lossZ�Y>����       ��-	���"fc�AH*

loss}�>�C0Z       ��-	1'�"fc�AI*

lossO��>�*�e       ��-	���"fc�AJ*

lossl~�>�5�       ��-	�U�"fc�AK*

loss���>UIkR       ��-	���"fc�AL*

loss���>����       ��-	Y��"fc�AM*

loss܆�>��~H       ��-	� �"fc�AN*

loss��~>�p:�       ��-	��"fc�AO*

lossC�n>�oC       ��-	�i�"fc�AP*

lossI��>��i�       ��-	>�"fc�AQ*

lossO^�>G/��       ��-	��"fc�AR*

lossv��>K᫨       ��-	�>�"fc�AS*

loss�c>y��       ��-	���"fc�AT*

loss���>�Lȓ       ��-	�}�"fc�AU*

lossdEe>�R�       ��-	��"fc�AV*

loss�Ϩ>�2��       ��-	,��"fc�AW*

loss�B�>��w�       ��-	px�"fc�AX*

loss�,�>1��2       ��-	�,�"fc�AY*

lossz�!>��s       ��-	+�"fc�AZ*

lossk�>��˴       ��-	���"fc�A[*

lossa�S>	$��       ��-	�a�"fc�A\*

losseC�>7`��       ��-	Z��"fc�A]*

loss#Za>&�!9       ��-	���"fc�A^*

lossw��>~35       ��-	�l�"fc�A_*

loss��>����       ��-	��"fc�A`*

lossqT�>��M�       ��-	���"fc�Aa*

loss� �>{�E�       ��-	A��"fc�Ab*

loss��>�T<=       ��-	�(�"fc�Ac*

loss�+l>,��       ��-	��"fc�Ad*

loss��>0q       ��-	�|�"fc�Ae*

lossD��= ܌a       ��-	m�"fc�Af*

loss[��>uG��       ��-	���"fc�Ag*

loss;�L>8��       ��-	Vb�"fc�Ah*

loss��w>5O�       ��-	���"fc�Ai*

loss�o>)�8       ��-	���"fc�Aj*

loss)�->f}�G       ��-	}�"fc�Ak*

loss1�>��       ��-	u �"fc�Al*

loss���>t��       ��-	P��"fc�Am*

loss|Ε>9]�       ��-	�f�"fc�An*

loss:��>9�E       ��-	k�"fc�Ao*

loss㻙>�d��       ��-	���"fc�Ap*

loss��>�Z�       ��-	���"fc�Aq*

loss4b>�{}'       ��-	�.�"fc�Ar*

loss~S>_���       ��-	8��"fc�As*

lossW=>���       ��-	���"fc�At*

losst�">��-�       ��-	O=�"fc�Au*

loss4y>&�S       ��-	t��"fc�Av*

loss(��>k֦�       ��-	��"fc�Aw*

lossy��>>�`       ��-	�6�"fc�Ax*

loss:h->�9�       ��-	���"fc�Ay*

loss��>���       ��-	Ҏ�"fc�Az*

loss��{>��bB       ��-	6�"fc�A{*

loss���>�@�       ��-	z��"fc�A|*

loss���=����       ��-	�� #fc�A}*

loss�4!>
J�o       ��-	b1#fc�A~*

lossݣ�>�~�       ��-	��#fc�A*

lossɬK>۠~       �	�|#fc�A�*

loss.[�>V��a       �	)#fc�A�*

loss���>�C��       �	�#fc�A�*

loss��T>���       �	6�#fc�A�*

loss�?O>ʦ�]       �	�:#fc�A�*

loss�h�=��l^       �	��#fc�A�*

loss���=�k�%       �	�#fc�A�*

loss=k�=Q�       �	
.#fc�A�*

lossH�F>,� �       �	��#fc�A�*

lossRl�>?�/       �	~#fc�A�*

loss
۬=�#C       �	�"	#fc�A�*

loss� l>Rj]       �	4�	#fc�A�*

lossd}�=k�M       �	�k
#fc�A�*

loss��=`#�       �	�#fc�A�*

loss3�,>`o�~       �	G�#fc�A�*

lossۺ>�="s       �	4H#fc�A�*

loss��>1��       �	��#fc�A�*

loss��>s��       �	��#fc�A�*

loss�>@�k       �	�J#fc�A�*

loss|I�>��wO       �	}#fc�A�*

loss��P={��       �	��#fc�A�*

loss�r>]^��       �	�V#fc�A�*

loss��h>r�K       �	�#fc�A�*

loss=X�=��1(       �	(�#fc�A�*

loss4�>6®�       �	q9#fc�A�*

loss�ǋ>� f       �	r�#fc�A�*

lossj:>���       �	��#fc�A�*

loss� >�W�K       �	T#fc�A�*

loss�c>'!�       �	��#fc�A�*

lossȸ�=Jv��       �	�N#fc�A�*

loss�V>����       �	��#fc�A�*

loss/��>Bf�       �	�~#fc�A�*

loss�>[v       �	�#fc�A�*

loss�S@>A �       �	��#fc�A�*

loss_�:>.��h       �	i#fc�A�*

loss>	�]D       �	R#fc�A�*

loss�!�=�1+�       �	��#fc�A�*

loss�>�W       �	�b#fc�A�*

loss�>�.�	       �	� #fc�A�*

loss��=*ߟ�       �	�#fc�A�*

loss&��=�I�       �	4K#fc�A�*

loss��>
���       �	��#fc�A�*

loss���=����       �	ҍ#fc�A�*

losso>��       �	�A#fc�A�*

lossq��=;���       �	=�#fc�A�*

loss�`L>�z�       �	�#fc�A�*

loss#4>�H�I       �	y= #fc�A�*

loss&�#=/���       �	�� #fc�A�*

loss��=.���       �	��!#fc�A�*

loss�V�>$���       �	p%"#fc�A�*

loss&�T>�[�=       �	��"#fc�A�*

lossx_�=��F�       �	\�##fc�A�*

loss���=eF�       �	$#fc�A�*

loss�h�<�m&�       �	�%#fc�A�*

loss@�[>�M�       �	W&#fc�A�*

loss� ">�c�       �	�l'#fc�A�*

loss6?y=1�cm       �	l%(#fc�A�*

loss��=l��|       �	s�(#fc�A�*

loss�o>���       �	�{)#fc�A�*

loss���=h䲺       �	�*#fc�A�*

loss�p�>Ш0�       �	1�*#fc�A�*

loss�
�=�s�       �	,#fc�A�*

lossr�>~��       �	�,#fc�A�*

loss>1>��       �	,b-#fc�A�*

loss*S>(�       �	A+.#fc�A�*

loss;��=/���       �	�.#fc�A�*

loss�	>�í       �	��/#fc�A�*

lossR�`=�,�       �	�0#fc�A�*

lossT;$>@��       �	r1#fc�A�*

loss4[�>�e�U       �	�2#fc�A�*

lossth>R��       �	��2#fc�A�*

loss�/�>ΰ�I       �	�m3#fc�A�*

losso�>Hdp       �	�.4#fc�A�*

loss�q�>� �       �	,�4#fc�A�*

loss=m�=[��       �	ܻ5#fc�A�*

loss�W>��$k       �	�~6#fc�A�*

lossH|�=ן��       �	 7#fc�A�*

lossa�B>�@�t       �	�7#fc�A�*

loss���=$R!|       �	��8#fc�A�*

loss��>�-��       �	�d9#fc�A�*

lossA��=;W�T       �	��9#fc�A�*

lossr�>-��       �	�:#fc�A�*

loss��>�:F�       �	�;#fc�A�*

loss �>6�       �	�j<#fc�A�*

loss���="��       �	�=#fc�A�*

loss�}x=!�       �	��=#fc�A�*

loss��=Mv_       �	I>#fc�A�*

loss��y>�=i-       �	��>#fc�A�*

loss�v:>�'�       �	ō?#fc�A�*

loss�>�໯       �	�*@#fc�A�*

lossݛ�>C��       �	9�@#fc�A�*

loss�,;>���+       �	�qA#fc�A�*

loss�e�>}ST       �	B#fc�A�*

loss�[�=���,       �	��B#fc�A�*

loss�z>���C       �	EbC#fc�A�*

loss�W�>i�7�       �	�
D#fc�A�*

lossE�t>�BB       �	дD#fc�A�*

loss�k�=\�q8       �	�YE#fc�A�*

lossIV�=yȶ_       �	^�E#fc�A�*

lossJ�2>��#�       �	:�F#fc�A�*

loss1>���d       �	JG#fc�A�*

loss��>�S|       �	�G#fc�A�*

loss�H>��G       �	��H#fc�A�*

lossM�=)�^       �	-I#fc�A�*

lossHkj=�B�9       �	��I#fc�A�*

loss���=�r�O       �	�qJ#fc�A�*

loss��e>�A��       �	�K#fc�A�*

loss\�h>�00O       �	��K#fc�A�*

loss�l5>�l�$       �	<NL#fc�A�*

loss�j3>��;       �	}�L#fc�A�*

loss��?>�4A�       �	��M#fc�A�*

lossn�8>EB��       �	$(N#fc�A�*

loss�>;���       �	��N#fc�A�*

loss���=�d4�       �	�]O#fc�A�*

loss��K>�ǧ�       �	��O#fc�A�*

loss�;l>��       �	��P#fc�A�*

loss1m1>����       �	F?Q#fc�A�*

loss��:>��	�       �	��Q#fc�A�*

loss�(l>�k       �	<�R#fc�A�*

loss�	>��ܢ       �	�sS#fc�A�*

loss���=y��       �		T#fc�A�*

lossi�>�5a       �	�T#fc�A�*

lossx~G>(n��       �	>wU#fc�A�*

loss�Kv>��1�       �	�V#fc�A�*

loss��>�md       �	��V#fc�A�*

lossũ!>�à       �	VW#fc�A�*

lossAϰ=�.й       �	Q�W#fc�A�*

lossۥ�=�X��       �	�X#fc�A�*

loss�X�=>Dj       �	LoY#fc�A�*

lossᨊ> !�d       �	ZZ#fc�A�*

loss�I=>J^�       �	��Z#fc�A�*

loss:��=�'�       �	�A[#fc�A�*

lossR�=���4       �	�[#fc�A�*

loss�)>��-�       �	{\#fc�A�*

loss��=���       �	�]#fc�A�*

lossh0�>���       �	.�]#fc�A�*

loss�>I���       �	�H^#fc�A�*

loss#�>9��       �	;�^#fc�A�*

loss���==$G�       �	_#fc�A�*

loss�>l�_�       �	�`#fc�A�*

loss�v1>���       �	s�`#fc�A�*

loss[��>�mO_       �	�Qa#fc�A�*

loss#D>(?��       �	K�a#fc�A�*

lossT��=2�X�       �	��b#fc�A�*

lossA	:>Ⱦb       �	�Mc#fc�A�*

loss}8�= �a�       �	R�c#fc�A�*

loss3��= �T0       �	�e#fc�A�*

lossN>X��       �	$f#fc�A�*

loss*�C>#�+       �	D�f#fc�A�*

loss���=��%�       �	��g#fc�A�*

lossl\q=��n        �	��h#fc�A�*

lossCm >�p       �	�ci#fc�A�*

loss��>�ZmL       �	�j#fc�A�*

loss�F=��t�       �	4�j#fc�A�*

lossZ�=7y�       �	&:k#fc�A�*

loss@�[>���       �	��k#fc�A�*

lossJM�=�\�       �	c}l#fc�A�*

loss}MF>���       �	{m#fc�A�*

losss�5>�#/�       �	��m#fc�A�*

loss`��=�^H�       �	�en#fc�A�*

loss���=���       �	�o#fc�A�*

loss��\=��l�       �	t�o#fc�A�*

loss��!=ox�       �	Sp#fc�A�*

loss|�6=���       �	��p#fc�A�*

lossu>7��       �	�q#fc�A�*

loss��=��H       �	IJr#fc�A�*

lossa�=�4�t       �	��r#fc�A�*

loss=�=0�rw       �	D�s#fc�A�*

loss�>`q�3       �	�#t#fc�A�*

lossc�>N���       �	��t#fc�A�*

loss�]>�U�       �	oeu#fc�A�*

lossX.9>��J�       �	��u#fc�A�*

loss#A>�*       �	�v#fc�A�*

loss�ݱ=�u��       �	Tx#fc�A�*

loss�41=@vuJ       �	��x#fc�A�*

loss�.>�y�5       �	�y#fc�A�*

lossN�>�7�
       �	N+z#fc�A�*

loss�&�=ѧ
�       �	��z#fc�A�*

loss}@d>�l)O       �	�a{#fc�A�*

loss|6=���G       �	& |#fc�A�*

loss�L>�?Y       �	��|#fc�A�*

loss��O=��@�       �	ڎ}#fc�A�*

loss�I>���       �	o*~#fc�A�*

loss��=+\%       �	!#fc�A�*

lossq�>�o��       �	]�#fc�A�*

loss��>cqj       �	�Z�#fc�A�*

loss�=)�}X       �	��#fc�A�*

loss�U'=k �       �	8��#fc�A�*

lossE�=�<��       �	�#�#fc�A�*

loss׋5=���/       �	΂#fc�A�*

lossí�<G@��       �	ir�#fc�A�*

loss��=��j       �	X�#fc�A�*

lossHF�=ca2�       �	k��#fc�A�*

loss;�p=l�>       �	@h�#fc�A�*

loss��>XN~�       �	��#fc�A�*

lossf�e=��|�       �	���#fc�A�*

losse�=���       �	 ^�#fc�A�*

losse
>���       �	�G�#fc�A�*

loss,�b>�\(       �	r�#fc�A�*

loss*�.>� A^       �	�}�#fc�A�*

loss�B�=;���       �	�-�#fc�A�*

loss��>����       �	CŊ#fc�A�*

loss�1>�_       �	�[�#fc�A�*

loss��6> �+       �	��#fc�A�*

loss�r�=,�!�       �	 ��#fc�A�*

loss&XU=�th�       �	�'�#fc�A�*

loss	nU=i�>       �	���#fc�A�*

loss&�B>J{)M       �	Na�#fc�A�*

loss�@�=N	�W       �	���#fc�A�*

loss�>G���       �	q��#fc�A�*

loss��>@i��       �	�,�#fc�A�*

loss�2�=�d�       �	XȐ#fc�A�*

loss�O=rJ       �	�a�#fc�A�*

loss$�]=%i��       �	���#fc�A�*

loss�@=��2)       �	���#fc�A�*

loss�x�=��       �	9�#fc�A�*

lossnb�=+���       �	ٓ#fc�A�*

loss P�>DV��       �	�y�#fc�A�*

lossp6>�۸�       �	�#fc�A�*

loss}'s=�X       �	Q��#fc�A�*

loss!�5>;L�       �	J`�#fc�A�*

lossO��=���       �	��#fc�A�*

loss��S>��c"       �	��#fc�A�*

loss��=v���       �	G�#fc�A�*

lossݶ�==�1�       �	*�#fc�A�*

loss��
>�7       �	���#fc�A�*

loss<��=�YZ       �	|)�#fc�A�*

loss� 0>��B`       �	7Ś#fc�A�*

loss'�>��T�       �	p_�#fc�A�*

loss%*>%��       �		��#fc�A�*

loss�dx=0E�g       �	l��#fc�A�*

loss�=���       �	�5�#fc�A�*

loss�]>����       �	�ߝ#fc�A�*

lossؾ=9�       �	H��#fc�A�*

loss�P�=H��       �	=(�#fc�A�*

loss�x=7+Ƀ       �	�ş#fc�A�*

loss�@V>5��       �	�g�#fc�A�*

loss��9>�B�@       �	��#fc�A�*

loss��=8ڻ       �	6��#fc�A�*

loss�[>M�{w       �	�G�#fc�A�*

loss��>=��9�       �	��#fc�A�*

loss(�>JG       �	��#fc�A�*

loss��=��       �	E�#fc�A�*

lossQ�>��C       �	���#fc�A�*

loss]ܓ=��/�       �	k��#fc�A�*

loss��=���       �	�A�#fc�A�*

lossf��=Ʒ�r       �	�P�#fc�A�*

loss: >�8�i       �	���#fc�A�*

loss�,�=C�       �	�ө#fc�A�*

loss�-[>��H       �	f��#fc�A�*

loss�v�=����       �	��#fc�A�*

loss�֨=N_Ye       �	 �#fc�A�*

loss��!>M�a�       �	���#fc�A�*

loss�G>>ƴ]6       �	|�#fc�A�*

loss)�S>���       �	#J�#fc�A�*

loss.��=�7v3       �	�[�#fc�A�*

loss<�=�y1<       �	�1�#fc�A�*

lossI�=��B       �	�R�#fc�A�*

loss_�>`�os       �	7��#fc�A�*

lossH��=���u       �	{��#fc�A�*

loss�N>��?       �	�#fc�A�*

loss4�>a�q�       �	`��#fc�A�*

lossx�#>�Y��       �	�ɷ#fc�A�*

loss��>�/f�       �	���#fc�A�*

lossr��=4DS       �	���#fc�A�*

lossj:�=�M�       �	�i�#fc�A�*

loss] >����       �	��#fc�A�*

loss�7?>"@�'       �	��#fc�A�*

loss�g�>���*       �	GX�#fc�A�*

loss �R=��@c       �	e��#fc�A�*

loss��=�pp�       �	���#fc�A�*

loss��>t�+�       �	0F�#fc�A�*

loss:�=%�֞       �	���#fc�A�*

loss��4=xtE�       �	X��#fc�A�*

loss��=��       �	�R�#fc�A�*

loss�>�eɊ       �	��#fc�A�*

lossJg�=��f�       �	���#fc�A�*

loss�)�=��+6       �	C:�#fc�A�*

loss��=>�*�`       �	���#fc�A�*

lossE|=>eX�n       �	u�#fc�A�*

loss�(>�7�D       �	{�#fc�A�*

loss�>��U�       �	E��#fc�A�*

lossJ�_>�N�?       �	�U�#fc�A�*

loss">HzX+       �	D��#fc�A�*

loss��'>� 2�       �	,��#fc�A�*

loss�=��v       �	��#fc�A�*

loss]�\>�! �       �	t}�#fc�A�*

loss�'>���       �	I-�#fc�A�*

loss��(>֒a(       �	���#fc�A�*

loss���<�e�       �	{��#fc�A�*

loss8�=�¨�       �	�0�#fc�A�*

loss�{�=��"       �	��#fc�A�*

lossvd�=^֒       �	u��#fc�A�*

loss�'=&+��       �	l�#fc�A�*

loss���=���7       �	�
�#fc�A�*

lossiv�="�0       �	q��#fc�A�*

lossv1@>�WK(       �	�O�#fc�A�*

loss-}C>]" ,       �	���#fc�A�*

loss�a>^���       �	���#fc�A�*

loss�S�=�Khw       �	�1�#fc�A�*

loss�
>��r4       �	c��#fc�A�*

loss@_�=H��       �	�r�#fc�A�*

loss
��=)       �	0�#fc�A�*

loss�
�="Z1�       �	���#fc�A�*

lossr��=���       �	 _�#fc�A�*

loss�3�=>�=(       �	���#fc�A�*

loss ;�=��1�       �	���#fc�A�*

loss�>�_M�       �	���#fc�A�*

lossz�'=냥�       �	�/�#fc�A�*

loss���=`5Dy       �	���#fc�A�*

loss!k�<��;�       �	{g�#fc�A�*

loss�<qE�       �	��#fc�A�*

loss��=pP�h       �	Q��#fc�A�*

loss��F>$��       �	a6�#fc�A�*

loss�n>�\�       �	���#fc�A�*

lossRߩ=�ߎ�       �	of�#fc�A�*

losse��=���       �	@�#fc�A�*

lossք`=�?a�       �	���#fc�A�*

lossZ�>"���       �	Do�#fc�A�*

loss}5�>�E��       �	�T�#fc�A�*

losss��=��3       �	cC�#fc�A�*

loss��|=���L       �	��#fc�A�*

loss&�=���       �	��#fc�A�*

lossW��=�n�       �	��#fc�A�*

loss%R�=�F��       �	b��#fc�A�*

loss8�=s+�:       �	�]�#fc�A�*

loss�&'>���]       �	��#fc�A�*

lossK~>�cg�       �	]��#fc�A�*

lossQ�=�/�       �	�Y�#fc�A�*

loss�- >���       �	��#fc�A�*

loss���=)�-�       �	���#fc�A�*

lossR�6>�Ĭ�       �	\=�#fc�A�*

lossu�>�U�       �	 E�#fc�A�*

loss �=���5       �	�#fc�A�*

loss��=U�f�       �	���#fc�A�*

loss��<r}��       �	���#fc�A�*

lossE�l>%1R-       �	���#fc�A�*

lossJ��=e$       �	���#fc�A�*

loss���=����       �	�C�#fc�A�*

lossZ��=dﶧ       �	��#fc�A�*

lossO
�=(���       �	�v�#fc�A�*

loss���=ѢT�       �	���#fc�A�*

loss�1=[�)       �	�=�#fc�A�*

loss:n>��'       �	���#fc�A�*

lossI�=��/       �	���#fc�A�*

loss�Zm>:jSM       �	�/�#fc�A�*

lossnI=>T��       �	���#fc�A�*

loss{�O=�	"�       �	(}�#fc�A�*

loss���=�!�       �	�#�#fc�A�*

lossZe�=��       �	.��#fc�A�*

losso��=�Ƀ       �	�m�#fc�A�*

lossDҽ=��QV       �	��#fc�A�*

loss}�3>Oc
       �	���#fc�A�*

loss��{>;��       �	iV�#fc�A�*

loss1��=Y�@�       �	���#fc�A�*

loss?3>�)��       �	K��#fc�A�*

loss� O>n@�       �	5�#fc�A�*

loss;��=)Ӄ�       �	�N�#fc�A�*

loss���=���       �	���#fc�A�*

loss�A&>�4�       �	y��#fc�A�*

loss��,>��|�       �	�2�#fc�A�*

lossMZ�=�Oԕ       �	 ��#fc�A�*

lossT�%>ڰkU       �	
g�#fc�A�*

loss;�1>����       �	� $fc�A�*

loss�N>�tE       �	I� $fc�A�*

lossc7=��&�       �	�<$fc�A�*

lossh�$=�v�       �	w�$fc�A�*

loss��i=�>�       �	�r$fc�A�*

loss׼�=�&n       �	�$fc�A�*

loss�N>��S=       �	�$fc�A�*

lossؖ=һ�       �	ZG$fc�A�*

loss{ �=���.       �	��$fc�A�*

lossM�8>qփ�       �	{�$fc�A�*

lossL('>	�y:       �	��$fc�A�*

loss*�E=z�;�       �	xA$fc�A�*

loss1�*=HĎ       �	��$fc�A�*

loss�J=�:�       �	�s$fc�A�*

loss��=�xm�       �	A	$fc�A�*

lossù�=`(��       �	u�	$fc�A�*

loss>�^&       �	�R
$fc�A�*

loss&;	=�<�       �	�($fc�A�*

loss��?>Or6k       �	��$fc�A�*

loss�>����       �	|_$fc�A�*

loss�}�=(�V�       �	��$fc�A�*

losss��=���       �	��$fc�A�*

loss��<(DVj       �	�:$fc�A�*

loss���=
�x        �	��$fc�A�*

loss�l�=�39u       �	��$fc�A�*

loss�<>|➲       �	T$fc�A�*

lossā�=��
+       �	=�$fc�A�*

loss�>RĴ       �	|$fc�A�*

lossS��=ZeL       �	AG$fc�A�*

loss���=-%�       �	 $fc�A�*

loss�x=�k��       �	��$fc�A�*

loss�R�=�N>       �	J{$fc�A�*

lossEKZ<t�g�       �	'$fc�A�*

loss�7�=���j       �	[�$fc�A�*

loss���=����       �	>z$fc�A�*

loss=iO= ��       �	�$fc�A�*

lossL��=0&��       �	ۿ$fc�A�*

loss\#>��l�       �	
h$fc�A�*

lossx�D>��       �	�m$fc�A�*

lossX�=?�       �	$fc�A�*

lossZ�=p��       �	��$fc�A�*

loss*��=M^DZ       �	fI$fc�A�*

loss��<�/�	       �	�$fc�A�*

loss f=����       �	d�$fc�A�*

loss�I=��K�       �	�$fc�A�*

loss��M<�Â       �	�($fc�A�*

loss\��<����       �	m�$fc�A�*

loss�J�<�=WI       �	t$fc�A�*

loss��/=�7�#       �	o $fc�A�*

losse��=��       �	� $fc�A�*

loss�"=�oC       �	�L!$fc�A�*

loss�G<�1�R       �	��!$fc�A�*

loss�˞;|8%       �	"$fc�A�*

lossա�=�l       �	�#$fc�A�*

loss��\=u�n       �	_�#$fc�A�*

lossͨ�<�W�>       �	��$$fc�A�*

loss=�<���<       �	K9%$fc�A�*

loss��<N       �	�&$fc�A�*

loss8>�>�t�       �	�3'$fc�A�*

lossr� <[��>       �	�($fc�A�*

loss�{&=��=       �	�:)$fc�A�*

loss@n�=��       �	E�)$fc�A�*

loss���=�=I>       �	d�*$fc�A�*

loss��<=Cnۮ       �	?+$fc�A�*

lossL�=��O       �	�,$fc�A�*

loss��>�/ս       �	�-$fc�A�*

loss4H}>A�#�       �	��-$fc�A�*

loss��=0�U       �	jh.$fc�A�*

loss�՘=Z�8       �	�t/$fc�A�*

loss���=|3��       �	`0$fc�A�*

loss�n_=���       �	��0$fc�A�*

loss��=�#7       �	��1$fc�A�*

loss�p>���y       �	@L2$fc�A�*

loss��e>�(�;       �	G�2$fc�A�*

lossɁ>��(�       �	��3$fc�A�*

loss��=k���       �	�)4$fc�A�*

loss�8>B��       �	��4$fc�A�*

loss��>n�R�       �	VI6$fc�A�*

loss f�=a��>       �	��6$fc�A�*

loss<�<[�]       �	@�7$fc�A�*

loss�=��       �	 8$fc�A�*

loss���=U,xy       �	|9$fc�A�*

lossJ�%=�P�       �	�:$fc�A�*

lossQ�-=ytD*       �	u�:$fc�A�*

loss�Q=��       �	bJ;$fc�A�*

loss_F">cG�       �	��;$fc�A�*

loss�إ=��#�       �	ʍ<$fc�A�*

loss
6�=X,�       �	x&=$fc�A�*

loss��>       �	��=$fc�A�*

loss��==r{�       �	�g>$fc�A�*

lossN��=)6�#       �	C?$fc�A�*

loss���=G}f       �	�?$fc�A�*

lossT��<"4d�       �	O@@$fc�A�*

loss���=歊C       �	�@$fc�A�*

loss��=�޳       �	�vA$fc�A�*

lossH�=J,�       �	�B$fc�A�*

lossqB�=��       �	��B$fc�A�*

loss�>�%�g       �	B>C$fc�A�*

lossy >_Jf�       �	��C$fc�A�*

loss� =o��       �	'lD$fc�A�*

loss�=׬�       �	�E$fc�A�*

loss�*1=Q       �	��E$fc�A�*

lossVI�=���N       �	CTF$fc�A�*

loss=�X�       �	8�F$fc�A�*

loss�m�=�Iz       �	��G$fc�A�*

losss3>�[c       �	79H$fc�A�*

loss�oH=�1V�       �	r�H$fc�A�*

loss$҇=)=��       �	�yI$fc�A�*

loss���=�~}W       �	�J$fc�A�*

lossx%�=/��       �	��J$fc�A�*

loss��=�1       �	�Xg$fc�A�*

lossh��=s7       �	�g$fc�A�*

loss��>=�!n       �	K�h$fc�A�*

lossͥ#>�Mte       �	�%i$fc�A�*

lossR�=}��       �	L�i$fc�A�*

loss:�$=@gw       �	P�j$fc�A�*

loss��1>z��       �	�!k$fc�A�*

loss��=O��       �	�k$fc�A�*

loss���=�+w�       �	HRl$fc�A�*

loss$�c=�[
       �	��l$fc�A�*

loss�G=2���       �	U�m$fc�A�*

loss�)=�Z��       �	�Gn$fc�A�*

lossHK�=/�       �	d�n$fc�A�*

loss1��=q��       �	��o$fc�A�*

loss/�=�{       �	[{p$fc�A�*

loss�1}>��L�       �	�q$fc�A�*

loss�(�=�p�       �	s�q$fc�A�*

loss�W= 5��       �	B`r$fc�A�*

loss �>���2       �	.s$fc�A�*

loss��>{Is       �	�s$fc�A�*

lossP�>hZs�       �	�Qt$fc�A�*

loss�>�       �	u$fc�A�*

losst�`=m_|�       �	w�u$fc�A�*

loss�	 >��7       �	Tnv$fc�A�*

loss�J�=K�fF       �	aw$fc�A�*

lossi1�=��       �	5x$fc�A�*

loss�>';       �	��x$fc�A�*

loss$L�<��        �	Sy$fc�A�*

lossC��=�       �	�z$fc�A�*

loss��=^�$�       �	��z$fc�A�*

lossqE=���d       �	y?{$fc�A�*

loss�7=!7L       �	��{$fc�A�*

loss�b"=F�=G       �	L�|$fc�A�*

lossm'>B��       �	D5}$fc�A�*

loss�۬=?��       �	^�}$fc�A�*

lossQ�>z�       �	J~~$fc�A�*

lossWf"=�ލ       �	�$fc�A�*

lossH�>x���       �	��$fc�A�*

loss��>*։       �	�`�$fc�A�*

lossn_>�Faf       �	���$fc�A�*

loss[O�=���       �	c��$fc�A�*

lossJ�=��l�       �	Y0�$fc�A�*

loss��$=��`�       �	�ς$fc�A�*

loss��>-���       �	Kr�$fc�A�*

lossȅx=g��       �	��$fc�A�*

loss�ƻ=��       �	�$fc�A�*

lossqk=��׃       �	�Y�$fc�A�*

loss�Zs=�\       �	��$fc�A�*

loss:�8>ϧ14       �	9��$fc�A�*

loss��<�	9       �	�B�$fc�A�*

loss�R�<�I`       �	P�$fc�A�*

loss#	">�U��       �	Sy�$fc�A�*

lossX̍=�V�       �	x�$fc�A�*

loss�	�>!��       �	���$fc�A�*

loss{%4=���       �	�C�$fc�A�*

loss�g�<��       �	�݊$fc�A�*

lossO�=G��b       �	�|�$fc�A�*

loss��A<k�:�       �	��$fc�A�*

loss\(>�T       �	2��$fc�A�*

loss�@ >6�M�       �	�R�$fc�A�*

lossm�>ދ�r       �	)�$fc�A�*

loss���=L
WK       �	���$fc�A�*

loss�&=Q���       �	u!�$fc�A�*

loss մ=L#�^       �	i�$fc�A�*

loss���=��       �	M��$fc�A�*

loss`�,=̢�       �	�#�$fc�A�*

loss �->݂�       �	3��$fc�A�*

lossa=�=�:�U       �	~�$fc�A�*

loss~I>�W       �	�$fc�A�*

loss�tM=F�
L       �	�#�$fc�A�*

loss<�3=�       �	��$fc�A�*

lossv|�=�M��       �	�S�$fc�A�*

loss�"$=�}f       �	Q��$fc�A�*

loss45�=�
`D       �	s��$fc�A�*

loss��=���+       �	�B�$fc�A�*

lossT�X=���       �	��$fc�A�*

loss�>i       �	��$fc�A�*

loss���=0Hi       �	?r�$fc�A�*

loss���=�a3       �	�b�$fc�A�*

loss{��=��ޯ       �	���$fc�A�*

loss%3�=`�O�       �	f��$fc�A�*

loss���=��k�       �	�I�$fc�A�*

lossH5>�Ү       �	K�$fc�A�*

loss2�b=k4zT       �	��$fc�A�*

loss���=o�M�       �	� �$fc�A�*

loss�=S��       �	o��$fc�A�*

loss-�=�c�       �	&U�$fc�A�*

losswl�=� H       �	���$fc�A�*

loss��#=V���       �	��$fc�A�*

loss�L�=@�г       �	�=�$fc�A�*

loss���=�:�       �	��$fc�A�*

lossaY�=^F�       �	���$fc�A�*

loss��!=0�       �	�7�$fc�A�*

loss�ED>!+5R       �	.�$fc�A�*

loss,f�=�Z��       �	e�$fc�A�*

loss2�<��       �	���$fc�A�*

loss;6>č��       �	��$fc�A�*

lossa��=� ��       �	���$fc�A�*

lossͺg>��`V       �	H7�$fc�A�*

loss��=[��       �	�ͨ$fc�A�*

loss�?8=�gnb       �	�b�$fc�A�*

loss��=���       �	��$fc�A�*

loss���=��dQ       �	��$fc�A�*

lossDi�=�)��       �	�K�$fc�A�*

loss���=��h       �	��$fc�A�*

lossA�H>�edF       �	"��$fc�A�*

lossq�w=�! m       �	F#�$fc�A�*

loss�=�< T       �	�ƭ$fc�A�*

loss�~I>�S��       �	Eb�$fc�A�*

loss�6q=����       �	�P�$fc�A�*

loss:��=Q�A�       �	���$fc�A�*

loss�# >]; �       �	W�$fc�A�*

loss�[4=��"       �	��$fc�A�*

loss{0�<F�K       �	���$fc�A�*

lossC�#=u>G�       �	�/�$fc�A�*

loss8^�<�c        �	rų$fc�A�*

loss��F=VW�       �	�r�$fc�A�*

loss� �=�1�       �	m;�$fc�A�*

lossO.=�Z:�       �	�Ե$fc�A�*

loss��Z=q�f�       �	7o�$fc�A�*

loss8˴=P���       �	��$fc�A�*

loss�C�=�I�_       �	Z��$fc�A�*

loss�#=�v,       �	�X�$fc�A�*

losss��=��/�       �	k~�$fc�A�*

loss��D<�@��       �	�!�$fc�A�*

loss��=��!       �	仺$fc�A�*

loss��=�Z       �	~W�$fc�A�*

loss�?=���       �	�-�$fc�A�*

loss.�=�s       �	Gȼ$fc�A�*

loss%[�=%�       �	c�$fc�A�*

loss�=v=`N�T       �	�$fc�A�*

loss�=1�^f       �	���$fc�A�*

loss&�=�ކ�       �	<1�$fc�A�*

loss:�=��`n       �	ʿ$fc�A�*

lossm�>�ç�       �	���$fc�A�*

lossn=�2@s       �	�$fc�A�*

lossc��=9V��       �	���$fc�A�*

loss=�?=:Gc�       �	gG�$fc�A�*

loss7=��ȅ       �	��$fc�A�*

loss�ۖ<	��       �	�w�$fc�A�*

loss�l2=ί"       �	�.�$fc�A�*

loss"^�=�X�       �	6�$fc�A�*

loss��=[�Z       �	���$fc�A�*

loss��=�l�g       �	Ow�$fc�A�*

loss/�>�T��       �	��$fc�A�*

loss��&=+Q�y       �	���$fc�A�*

loss�T}=Ŋ��       �	�E�$fc�A�*

loss�\=��       �	P��$fc�A�*

loss(˨<E-U       �	+��$fc�A�*

lossW�>M�       �	�&�$fc�A�*

loss�K�<��       �	���$fc�A�*

loss�>k��       �	���$fc�A�*

loss� J=x��}       �	'M�$fc�A�*

loss���=Nn�       �	���$fc�A�*

lossCc>u�(�       �	q��$fc�A�*

lossC�;=|�O       �	f�$fc�A�*

loss6,�=�]D       �	d�$fc�A�*

loss�>S虃       �	D��$fc�A�*

loss
e>(�        �	AE�$fc�A�*

loss��=��#�       �	���$fc�A�*

loss�pf=��       �	yv�$fc�A�*

lossr��<F�u       �	�$fc�A�*

lossA�P=��z�       �	£�$fc�A�*

loss��=n�       �	pA�$fc�A�*

loss��=����       �	S��$fc�A�*

loss�=^       �	�v�$fc�A�*

loss�@�<�o��       �	:\�$fc�A�*

lossSY>�lu$       �	^��$fc�A�*

loss�ƃ=�v-�       �	���$fc�A�*

loss�z�<^R��       �	w-�$fc�A�*

loss�s�=v�)       �	9��$fc�A�*

loss���< �0       �	i�$fc�A�*

loss!�=��       �	9'�$fc�A�*

lossM��=��DQ       �	��$fc�A�*

lossU<��k�       �	��$fc�A�*

loss�q�=��{b       �	�F�$fc�A�*

loss�p�=���3       �	���$fc�A�*

loss���=�ZA       �	�H�$fc�A�*

lossjl�=���6       �	q��$fc�A�*

losst<P�       �	=��$fc�A�*

loss(C�<��U       �	)!�$fc�A�*

losstЫ=���W       �	ݴ�$fc�A�*

loss�)i=����       �	�G�$fc�A�*

lossr��<�˹�       �	���$fc�A�*

lossS�=RC�       �	0�$fc�A�*

loss�K�=b���       �	��$fc�A�*

lossxl�=��5       �	Ƨ�$fc�A�*

loss�w2=���       �	lA�$fc�A�*

loss���<����       �	���$fc�A�*

lossok�=>��       �	]n�$fc�A�*

loss�!W=n�m       �	�S�$fc�A�*

loss)�=$��       �	���$fc�A�*

loss�v�=���       �	���$fc�A�*

loss��=�0       �	'��$fc�A�*

loss��U<����       �	�2�$fc�A�*

loss�'�<ǯ       �	��$fc�A�*

lossʀs=��       �	�g�$fc�A�*

loss�ʲ=�,�       �	��$fc�A�*

loss���=��a       �	��$fc�A�*

loss��=�Q��       �	�;�$fc�A�*

loss��P=�YG       �	7�$fc�A�*

loss���<�1�       �	���$fc�A�*

loss~۔=4�w       �	k�$fc�A�*

lossM/=�G��       �	�$fc�A�*

losss)=)��       �	���$fc�A�*

losst6=8r��       �	�W�$fc�A�*

loss�ms=s���       �	���$fc�A�*

loss�p�=g�       �	��$fc�A�*

loss�"�=���       �	�#�$fc�A�*

loss6�>�Ӫ�       �	���$fc�A�*

loss�?S=���       �	Z�$fc�A�*

loss@��<��c       �	���$fc�A�*

lossx�<�ރ       �	���$fc�A�*

loss?&=�X�       �	�0�$fc�A�*

loss���=���       �	A��$fc�A�*

lossl>S.l�       �	�o�$fc�A�*

loss��=b��       �	��$fc�A�*

loss�.>>����       �	}��$fc�A�*

loss��>� ��       �	�T�$fc�A�*

lossh�=��9       �	B��$fc�A�*

lossZL%=j.�h       �	o��$fc�A�*

lossE��<� ܉       �	v�$fc�A�*

loss��=ބ�K       �	o+�$fc�A�*

loss�>o�O�       �	Y��$fc�A�*

losst��<>zZ6       �	{f�$fc�A�*

lossʇ5=\?+p       �	e �$fc�A�*

loss���=b��Z       �	���$fc�A�*

lossK�=I�U       �	g`�$fc�A�*

loss�M=Z�;�       �	��$fc�A�*

loss���=���       �	��$fc�A�*

loss�a�<�'��       �	1\ %fc�A�*

loss%�=���       �	3� %fc�A�*

loss�>IA�)       �	ٙ%fc�A�*

lossȿ�=���q       �	VE%fc�A�*

loss.K�=�!)�       �	��%fc�A�*

loss�t =&��       �	Ή%fc�A�*

loss�:1=�y�       �	�@%fc�A�*

loss�6>=�U�       �	.�%fc�A�*

loss�L�=ܭ��       �	ׇ%fc�A�*

lossR�i=У�>       �	�(%fc�A�*

loss)N.=��W�       �	��%fc�A�*

loss�t_=�4;�       �	{f%fc�A�*

loss\��=����       �	S%fc�A�*

loss7��=u���       �	�%fc�A�*

loss�R�=���       �	e7	%fc�A�*

loss��U=�V7Z       �	��	%fc�A�*

loss�Za=`       �	�k
%fc�A�*

loss@�=�#�       �	�%fc�A�*

lossl'�=C��g       �	��%fc�A�*

loss#R=��Z       �	Hm%fc�A�*

loss�C�=H �       �	%fc�A�*

loss
;�=n�g�       �	�%fc�A�*

loss��^=p�\�       �	q<%fc�A�*

loss�;=�sKF       �	��%fc�A�*

loss���;o,hK       �	�%fc�A�*

loss}Z�=���       �	�%fc�A�*

lossڠ�=���       �	�%fc�A�*

loss!��=fs�K       �	K%fc�A�*

loss��*=D���       �	�%fc�A�*

lossM�=?���       �	+�%fc�A�*

loss�ǚ=���       �	&%fc�A�*

loss�t=Gؽ       �	��%fc�A�*

loss���=���       �	@j%fc�A�*

loss�s	=W���       �	%fc�A�*

loss!�r=J4�       �	̲%fc�A�*

loss:��<Ʉ�l       �	�^%fc�A�*

losst/�=^ ;�       �	�%fc�A�*

loss;��=�SL�       �	]�%fc�A�*

loss �#>����       �	O?%fc�A�*

loss�v�=H]�2       �	�-%fc�A�*

loss8�f=���       �	��%fc�A�*

lossc��=:��       �	�z%fc�A�*

lossς=�i��       �	�%fc�A�*

loss�w�=��:x       �	�j%fc�A�*

loss=z>��/7       �	�%fc�A�*

lossj?o>��       �	�%fc�A�*

loss�^Z=��       �	 B %fc�A�*

loss�a=�ˑ�       �	� %fc�A�*

loss��=�^>       �	֌!%fc�A�*

loss��^=�<��       �	�*"%fc�A�*

loss��;~s��       �	`�"%fc�A�*

loss�6=��z�       �	Sw#%fc�A�*

loss��=����       �	*$%fc�A�*

loss@ =�x5�       �	<�$%fc�A�*

loss7ԕ=�`[�       �	'�%%fc�A�*

loss��=��       �	�d&%fc�A�*

loss���=�{��       �	��&%fc�A�*

loss�6�=+X��       �	�'%fc�A�*

loss���=Dv O       �	�a(%fc�A�*

loss�$�<��       �	$�)%fc�A�*

loss��=�z��       �	��*%fc�A�*

loss��=C��[       �	3�+%fc�A�*

loss��h="H_�       �	ǻ,%fc�A�*

loss��=����       �	Yj-%fc�A�*

lossȁ�<'        �	u[.%fc�A�*

loss�A�=�ŗ�       �	5/%fc�A�*

lossn��<��       �	`�/%fc�A�*

loss�|!>=�Dr       �	�0%fc�A�*

loss�>��Q>       �	_E1%fc�A�*

loss�e�=麧�       �	��1%fc�A�*

loss�GI=(oz       �	4�2%fc�A�*

loss]�=
;��       �	c+3%fc�A�*

loss���=��
       �	��3%fc�A�*

lossʫ=d�       �	0�4%fc�A�*

loss|��=<�       �	(D5%fc�A�*

loss҇>TFXr       �	��5%fc�A�*

loss�#0<��       �	��7%fc�A�*

loss{�=Ib�h       �	z8%fc�A�*

loss�AU=��       �	�!9%fc�A�*

loss�=�y��       �	��9%fc�A�*

loss�l�=��D       �	=_:%fc�A�*

loss\'�=�}�0       �	L�:%fc�A�*

lossx�=� ��       �	��;%fc�A�*

losse^�<���       �	/2<%fc�A�*

lossZi�<���8       �	�<%fc�A�*

loss���=��]       �	�k=%fc�A�*

lossrI<,��       �	�>%fc�A�*

loss�N=J�|       �	��>%fc�A�*

loss6��=�+�       �	ZK?%fc�A�*

loss@�1=p��       �	>�?%fc�A�*

loss�z(=� Z       �	�@%fc�A�*

lossHb>���       �	#A%fc�A�*

losspH�=�Ρ       �	f�A%fc�A�*

lossX^�=�eH�       �	\B%fc�A�*

loss���=��       �	��B%fc�A�*

lossW�>�Fs       �	��C%fc�A�*

loss7�=��       �	�@D%fc�A�*

loss��;=�EG       �	��D%fc�A�*

lossl?�=�=��       �	˃E%fc�A�*

loss��=��p       �	�#F%fc�A�*

loss��<��>�       �	��F%fc�A�*

loss��g=��h�       �	�lG%fc�A�*

loss3�B<i���       �	�H%fc�A�*

loss��L=4,�       �	A�H%fc�A�*

loss:��=fb�=       �	yZI%fc�A�*

lossJon=�$ .       �	I�I%fc�A�*

loss�0==QĔ       �	}�J%fc�A�*

loss}a>�2��       �	q9K%fc�A�*

loss�e=�I       �	"6L%fc�A�*

loss�g�;ۇ|       �	��L%fc�A�*

loss�=��       �	�nM%fc�A�*

loss#��<=�~       �	/�N%fc�A�*

loss/��=�07�       �	[[O%fc�A�*

loss�=�B�       �	n�O%fc�A�*

lossh}3>�l��       �	4�P%fc�A�*

loss��=%�g       �	�:Q%fc�A�*

loss��=D���       �	/�Q%fc�A�*

loss/�>�6�        �	{R%fc�A�*

loss��=C?       �	@S%fc�A�*

loss��.=eP��       �	_�S%fc�A�*

loss�'=�`jJ       �	��T%fc�A�*

lossR��<�~�       �	�5U%fc�A�*

loss7�B=��        �	Z�U%fc�A�*

loss8�i=�dE�       �	0�V%fc�A�*

loss
>w��       �	@W%fc�A�*

lossM�O=�E�       �	�W%fc�A�*

loss� <+�       �	RX%fc�A�*

loss2{<=
�N       �	�Y%fc�A�*

loss�a>�@�J       �	�VZ%fc�A�*

loss6\_=�Z�=       �	�Z%fc�A�*

lossLu=,��       �	�[%fc�A�*

loss�n�<�;�       �	�&\%fc�A�*

loss��1=���S       �	�\%fc�A�*

loss755>�       �	i]%fc�A�*

losss�/>{��       �	�/_%fc�A�*

loss.�H=�Ux�       �	��_%fc�A�*

loss;� >�@�8       �	Tq`%fc�A�*

loss1��<��hF       �	�a%fc�A�*

loss��!=�g��       �	�b%fc�A�*

losst�=�;2       �	L�b%fc�A�*

loss��=ƶ&�       �	�=c%fc�A�*

loss�=/�c�       �	��c%fc�A�*

loss��p=�g�       �	h�d%fc�A�*

loss�Z=h��       �	�oe%fc�A�*

lossuS�=�x�       �	Byf%fc�A�*

loss��'>�A       �	�cg%fc�A�*

loss��=�T��       �	�h%fc�A�*

lossF��=|o�       �	u�h%fc�A�*

loss��d=�&�_       �	�Vi%fc�A�*

loss�>�e�O       �	]�i%fc�A�*

loss���=���       �	o�j%fc�A�*

loss�c=��       �	:\k%fc�A�*

loss�Y>�}p       �	��k%fc�A�*

loss��<fj�=       �	Ϡl%fc�A�*

loss$�2=<�C       �	�:m%fc�A�*

lossr{=�*�       �	x�m%fc�A�*

loss�ʂ=����       �	�on%fc�A�*

loss� �=H�Af       �	�o%fc�A�*

loss�9K=�yD       �	�o%fc�A�*

lossC%�=Z�Gk       �	?p%fc�A�*

loss"��=G"H       �	4�p%fc�A�*

lossx��=^��       �	��q%fc�A�*

loss��=�f�       �	�Tr%fc�A�*

loss �=u�Ǜ       �	��r%fc�A�*

lossw+�<!�B       �	��s%fc�A�*

loss�P>���       �	�7t%fc�A�*

loss��i;��C       �	b�t%fc�A�*

loss|^=��K]       �	vu%fc�A�*

loss��=��       �	�v%fc�A�*

lossK =᳒s       �	��v%fc�A�*

loss�%�<�ǵ�       �	�Zw%fc�A�*

loss_=��c       �	��w%fc�A�*

loss�>)�?�       �	��x%fc�A�*

loss�A�<&B��       �	+Ky%fc�A�*

loss�G	>v�	       �	:�y%fc�A�*

loss_ؕ=�0>       �	w�z%fc�A�*

loss.��=�C       �	)#{%fc�A�*

loss~� >�h       �	��{%fc�A�*

loss�=V��       �	}^|%fc�A�*

loss�=����       �	I�|%fc�A�*

loss�{�=�K       �	��}%fc�A�*

loss.^�=�b�"       �	�+~%fc�A�*

lossqA�<_���       �	��~%fc�A�*

loss���=#O�x       �	x%fc�A�*

lossƞ�=�2��       �	�%fc�A�*

loss�Y�=ؿ��       �	ݱ�%fc�A�*

lossƱ~<s&�<       �	eQ�%fc�A�*

loss��=�DD`       �	�%fc�A�*

loss��=�       �	���%fc�A�*

loss��=0�_�       �	/�%fc�A�*

loss�æ<�Rr�       �	B��%fc�A�*

loss���=G�M"       �	(I�%fc�A�*

loss:�_<�]L4       �	��%fc�A�*

loss�E>�9)x       �	�w�%fc�A�*

loss���=���&       �	=�%fc�A�*

lossf6�=!�i       �	���%fc�A�*

loss)��<�	�'       �	Pq�%fc�A�*

loss�F=	u:�       �	�ǈ%fc�A�*

loss��=���C       �	6r�%fc�A�*

loss*F�=$� �       �	�&�%fc�A�*

loss�@=a�       �	�Ȋ%fc�A�*

lossl}=�#!       �	xc�%fc�A�*

loss���<Xğ�       �	?��%fc�A�*

lossCo�=�C��       �	���%fc�A�*

loss��V=
�%       �	��%fc�A�*

loss[kC=�;$       �	�@�%fc�A�*

loss�\�=�.�       �	��%fc�A�*

loss
�*=�F��       �	���%fc�A�*

loss��<��{�       �	^/�%fc�A�*

lossc�@<�k4�       �	Gɐ%fc�A�*

lossԅ>�ZH       �	�g�%fc�A�*

loss�A�=��2       �	K�%fc�A�*

loss��=m���       �	^��%fc�A�*

loss�1�=J�e�       �	�5�%fc�A�*

loss�
,=���       �	�Г%fc�A�*

lossw�~=l�       �	�h�%fc�A�*

loss>��&�       �	jm�%fc�A�*

lossL��=���_       �	��%fc�A�*

lossɨ�=�6ܙ       �	��%fc�A�*

lossQ��=���       �	�A�%fc�A�*

loss���=��"�       �	
ۗ%fc�A�*

loss(�t=J���       �	�u�%fc�A�*

loss&$=RW�C       �	o�%fc�A�*

loss��>L��       �	?��%fc�A�*

loss$�">��+       �	�H�%fc�A�*

loss8�=���       �	�f�%fc�A�*

loss��e="-�
       �	a��%fc�A�*

loss��-=�!��       �	���%fc�A�*

loss���<��n       �	�e�%fc�A�*

loss�=�jh�       �	��%fc�A�*

loss�%�=v��       �	���%fc�A�*

lossa)�<�g�u       �	=I�%fc�A�*

loss�W�<ıR�       �	�%fc�A�*

lossM\�=��M�       �	���%fc�A�*

loss1�=XN�       �	P�%fc�A�*

lossf=!~I�       �	�ɢ%fc�A�*

loss*�=���       �	rm�%fc�A�*

loss�&=�/�d       �	U�%fc�A�*

loss�I�=�,<       �	%��%fc�A�*

loss�gI<7��}       �	���%fc�A�*

lossvI`>����       �	E��%fc�A�*

lossҎ�=?��       �	z�%fc�A�*

loss{Q�=�ژ1       �	�̨%fc�A�*

loss�u�=Y`�       �	
ש%fc�A�*

loss�=IH�       �	5��%fc�A�*

loss
�=��lY       �	�d�%fc�A�*

loss��<���J       �	A�%fc�A�*

loss2$=wm       �	Ҩ�%fc�A�*

loss$�<�v�u       �	B�%fc�A�*

loss�>/��       �	�٭%fc�A�*

loss�<�=T�E       �	`t�%fc�A�*

loss%�b=��@�       �	n0�%fc�A�*

loss�RC=4e/�       �	�ʯ%fc�A�*

loss��>F�zj       �	Rd�%fc�A�*

loss���=�}~�       �	��%fc�A�*

loss�e�<�f��       �	鞱%fc�A�*

loss�MN=a.u�       �	�7�%fc�A�*

loss���=��Y�       �	RӲ%fc�A�*

loss̳<'��       �	�p�%fc�A�*

lossL�=z)�8       �	�M�%fc�A�*

loss-�<�@�       �	���%fc�A�*

loss�4�=l��R       �	U1�%fc�A�*

loss��=��       �	>ζ%fc�A�*

loss�d=*���       �	�m�%fc�A�*

loss~S=!:<       �	�%fc�A�*

loss�=�!       �	��%fc�A�*

loss*�=5�i       �	�J�%fc�A�*

loss[�=�Ű�       �	��%fc�A�*

loss��|=�W       �	���%fc�A�*

lossq�f=ݢ�m       �	0�%fc�A�*

loss�~G=7��n       �	���%fc�A�*

loss�O<�ћ       �	D��%fc�A�*

loss	�<�+       �	�x�%fc�A�*

loss&��<�P2       �	�'�%fc�A�*

loss��=�EO       �	�տ%fc�A�*

loss��<��Ɠ       �	��%fc�A�*

lossU=_�       �	�8�%fc�A�*

loss��<l��       �	���%fc�A�*

loss�#�=�S|�       �	���%fc�A�*

lossq=T�=       �	78�%fc�A�*

loss���==��       �	+��%fc�A�*

lossv��=\���       �	Z�%fc�A�*

loss�7�<�&��       �	� �%fc�A�*

loss=�=[=��       �	��%fc�A�*

loss�YH=3گ�       �	y[�%fc�A�*

lossEL8=��?�       �	'��%fc�A�*

loss̍}=Ic��       �	��%fc�A�*

loss[�=��0`       �	j��%fc�A�*

loss�]�<Ա��       �	uv�%fc�A�*

loss�b6=�:��       �	�%fc�A�*

lossz$�=J��C       �	���%fc�A�*

loss��"=;��       �	�\�%fc�A�*

loss�<!�Y        �	��%fc�A�*

loss@�<c�H�       �	���%fc�A�*

loss�?3=k�q       �	�L�%fc�A�*

loss���=���       �	���%fc�A�*

lossKˉ=��       �	1��%fc�A�*

loss�_�=M<h       �	�8�%fc�A�*

loss�v�=w
��       �	���%fc�A�*

lossZ�<\Y�       �	ƨ�%fc�A�*

lossG�<I^�B       �	EF�%fc�A�*

loss��=;��*       �	���%fc�A�*

loss��w< #��       �	m��%fc�A�*

loss��<�D�E       �	X7�%fc�A�*

loss�%?<��r       �	r��%fc�A�*

loss�G�<��       �	���%fc�A�*

loss�B�:.Xi9       �	��%fc�A�*

loss�w�<q�7S       �	5D�%fc�A�*

loss�50=�*Oc       �	��%fc�A�*

loss��o=����       �	]��%fc�A�*

lossj%�;lb��       �	�K�%fc�A�*

loss�r�9^:�       �	���%fc�A�*

lossHЗ:;Ԉl       �	���%fc�A�*

lossX�<D�&{       �	�8�%fc�A�*

loss��]=݀9�       �	#��%fc�A�*

lossc�=����       �	���%fc�A�*

loss��f<�~�       �	�X�%fc�A�*

loss�A�<ޜ!       �	���%fc�A�*

lossm��>S��0       �	���%fc�A�*

lossfE�;VNؙ       �	�j�%fc�A�*

loss��+=��       �	A�%fc�A�*

loss�=�
       �	T��%fc�A�	*

lossrw*>��Q       �	,~�%fc�A�	*

loss�<@y�K       �	;�%fc�A�	*

lossE��<�Z�z       �	0��%fc�A�	*

loss,>�h<p       �	\�%fc�A�	*

loss�}�=�N�b       �	z��%fc�A�	*

loss�S�=v�<       �	��%fc�A�	*

loss�>����       �	��%fc�A�	*

loss~:�=��x       �	�?�%fc�A�	*

loss &�=���       �	V�%fc�A�	*

loss�y�=�	&�       �	<N�%fc�A�	*

loss��&=�ES       �	���%fc�A�	*

loss���=�I�       �	���%fc�A�	*

loss���=�˦�       �	~q�%fc�A�	*

loss��^=��'       �	:"�%fc�A�	*

lossl�8=?WL#       �	���%fc�A�	*

lossC��=�}�       �	��%fc�A�	*

loss�=�rP�       �	yx�%fc�A�	*

loss�=0�J�       �	3�%fc�A�	*

loss�1=ꙮ�       �	|C�%fc�A�	*

losss^=閍�       �	$%�%fc�A�	*

loss�z<��       �	�m�%fc�A�	*

loss�~/=�c��       �	Gs�%fc�A�	*

loss���<nd��       �		�%fc�A�	*

loss�1�=L�       �	7��%fc�A�	*

loss��=y�|       �	�b�%fc�A�	*

loss�I�=_�Xg       �	��%fc�A�	*

lossL��=��,s       �	b��%fc�A�	*

loss:�4<�»�       �	9�%fc�A�	*

loss8%W=��Ĝ       �	���%fc�A�	*

loss@�V=�c�       �	:v�%fc�A�	*

loss�hp<A�@       �	�%fc�A�	*

loss}�>=�ȁ-       �	$��%fc�A�	*

loss�m�<�N�v       �	?S�%fc�A�	*

loss(D=�Đ       �	���%fc�A�	*

lossC��=�d       �	͒�%fc�A�	*

loss�u�=�婂       �	�*�%fc�A�	*

loss<��=li9�       �	.��%fc�A�	*

loss�g<.kL       �	Ie�%fc�A�	*

loss��<oD.u       �	��%fc�A�	*

loss8��=G�.       �	���%fc�A�	*

lossַT=8j�       �	�D�%fc�A�	*

loss꽔=^�r�       �	x��%fc�A�	*

lossFJ[=��.M       �	�%fc�A�	*

loss��=(�c�       �	y" &fc�A�	*

loss���<�W�X       �	]� &fc�A�	*

loss ؁=	P��       �	Ac&fc�A�	*

loss���<��OR       �	y&fc�A�	*

lossH��<��       �	��&fc�A�	*

loss���=d�h�       �	 ]&fc�A�	*

loss���=M�E       �	M�&fc�A�	*

lossS>�ӧ       �	$�&fc�A�	*

loss#qA=���       �	e7&fc�A�	*

lossc�C=ǻ44       �	��&fc�A�	*

lossS�8=Q}�       �	�&fc�A�	*

loss�Z	=����       �	5&fc�A�	*

loss$��=�In       �	
�&fc�A�	*

loss�>'�w�       �	�v&fc�A�	*

loss�=����       �	�&fc�A�	*

loss< @< /�       �	�&fc�A�	*

lossnT>lV�:       �	=_ &fc�A�	*

loss�f=o~       �	z� &fc�A�	*

loss�z=J\��       �	ٗ!&fc�A�	*

loss�8I=�s��       �	G8"&fc�A�	*

loss?��=�P       �	��"&fc�A�	*

loss���<��*       �	Po#&fc�A�	*

loss\�<A��*       �	�$&fc�A�	*

loss6��<a�d       �	��$&fc�A�	*

loss���=��f       �	9B%&fc�A�	*

loss��r=T���       �	��%&fc�A�	*

loss�>����       �	�&&fc�A�	*

lossRݰ<O.�       �	.�'&fc�A�	*

loss���=w�E       �	Q(&fc�A�	*

loss�&=k���       �	�-)&fc�A�	*

loss�7=F��       �	�r*&fc�A�	*

loss��=�|�7       �	�+&fc�A�	*

loss��<���       �	�,&fc�A�	*

loss�ӌ=��C@       �		-&fc�A�	*

lossfp�=0���       �	+�-&fc�A�	*

loss҅1=��B       �	�m.&fc�A�	*

loss�#�<˔	I       �	'P/&fc�A�	*

loss;T�=j:�d       �	50&fc�A�	*

lossL�>�6.A       �	W�0&fc�A�	*

lossP#=�6ς       �	�g1&fc�A�	*

lossDO= se       �	i2&fc�A�	*

loss<��<n�6�       �	��2&fc�A�	*

loss!��=��`E       �	9*3&fc�A�	*

loss	��=��d       �	��3&fc�A�	*

loss8M0=)��q       �	"l4&fc�A�	*

loss&$q=�/�=       �	r5&fc�A�	*

loss�a=��       �	��5&fc�A�	*

lossɒ<.f�       �	?T6&fc�A�	*

lossD�=��:N       �	��6&fc�A�	*

loss/��<
\B�       �	t�7&fc�A�	*

loss[�=�p�       �	Ύ8&fc�A�	*

loss�u=xE.-       �	�*9&fc�A�	*

loss*�=F�       �	�9&fc�A�	*

loss7d�<9ɭ       �	�^:&fc�A�	*

loss�L�<	�B[       �	�;&fc�A�	*

loss#v�<���       �	k�;&fc�A�	*

loss���<n�m       �	q9<&fc�A�	*

loss�<�<        �	��<&fc�A�	*

loss�\�>���       �	�k=&fc�A�	*

loss�F�=�iX       �	!>&fc�A�	*

loss�h�;C��       �	r�>&fc�A�	*

loss�-O<���       �	sK?&fc�A�	*

loss�%�<;��       �	��?&fc�A�	*

loss[�=X�7       �	Ԙ@&fc�A�	*

loss�v�=�]�       �	&�A&fc�A�	*

loss�a�=	G̳       �	�)B&fc�A�	*

loss�??=���0       �	��B&fc�A�	*

loss!-\<��H       �	p`C&fc�A�	*

loss2(t=��Z       �	a�C&fc�A�	*

loss�HI<~��       �	{�D&fc�A�	*

loss@�=U��       �	(FE&fc�A�	*

loss6)t=(��k       �	nNF&fc�A�	*

loss}q�=uim       �	{�F&fc�A�	*

lossAq�=]K8       �	�G&fc�A�	*

lossI��=��	�       �	T;H&fc�A�	*

loss�3=[���       �	��H&fc�A�	*

loss��G=}�A�       �	yxI&fc�A�	*

loss�6,=�.��       �	J&fc�A�	*

loss:��=�,|�       �	��J&fc�A�	*

loss*=�DC=       �	a�K&fc�A�	*

loss�c�<ڪ#�       �	?6L&fc�A�	*

loss���=�3       �	<�L&fc�A�	*

loss���=��$&       �	ۆM&fc�A�	*

lossn#=����       �	O&fc�A�	*

loss�1=�u��       �	;�O&fc�A�
*

loss�ֆ=�#�       �	�pP&fc�A�
*

loss���=Q�܀       �	�)Q&fc�A�
*

loss��=��       �	8�Q&fc�A�
*

lossv��<���       �	�~R&fc�A�
*

loss��<����       �	I/S&fc�A�
*

loss+=��l�       �	��S&fc�A�
*

loss�=���       �	��T&fc�A�
*

lossH�\<�C�f       �	w0U&fc�A�
*

loss���=T��e       �	�U&fc�A�
*

loss���<AZ;	       �	+�V&fc�A�
*

lossrS�<$�&       �	�W&fc�A�
*

loss��='�       �	��X&fc�A�
*

loss�}=�*G�       �	FY&fc�A�
*

lossE�A=�5i�       �	�Y&fc�A�
*

loss; 0=c��       �	V�Z&fc�A�
*

loss��<���r       �	$�[&fc�A�
*

loss,[�=�n�       �	y>\&fc�A�
*

loss�j�<���       �	K�\&fc�A�
*

loss8x>��@        �	��]&fc�A�
*

loss�(�<A�       �	�k^&fc�A�
*

loss���<9�       �	q_&fc�A�
*

loss7�"=���*       �	u�_&fc�A�
*

loss��=<׎       �	X�`&fc�A�
*

loss<BB=E��       �	�+a&fc�A�
*

loss sc<�Gm�       �	��a&fc�A�
*

loss4O�=�hxu       �	Ɗb&fc�A�
*

loss1�<DJ��       �	,.c&fc�A�
*

loss!=��i       �	�c&fc�A�
*

loss�>P�Ԙ       �	�d&fc�A�
*

lossNAu<i�w1       �	�"e&fc�A�
*

loss�zs=����       �	��e&fc�A�
*

loss3�=kQ�       �	B[f&fc�A�
*

loss��<��$J       �	o�f&fc�A�
*

loss�5�<���m       �	2�g&fc�A�
*

loss;=K��       �	�ah&fc�A�
*

loss��h=�휸       �	�:i&fc�A�
*

loss.�=&�       �	mj&fc�A�
*

loss��=E�X�       �	��j&fc�A�
*

loss�N�=ç       �	!tk&fc�A�
*

loss�<�=��g       �	ml&fc�A�
*

lossa/=2�0       �	��l&fc�A�
*

lossd�<+�*�       �	qm&fc�A�
*

loss���=����       �	^n&fc�A�
*

loss���=0��g       �	��n&fc�A�
*

loss���;]lH7       �	��o&fc�A�
*

loss�=��$�       �	�Bp&fc�A�
*

loss~$�<�~:l       �	"�p&fc�A�
*

loss��=ٸ�#       �	��q&fc�A�
*

loss6ژ=�^       �	�8r&fc�A�
*

loss�FM=?�rl       �	��r&fc�A�
*

lossx(�<�x��       �	�s&fc�A�
*

loss)`=u�6       �	�Et&fc�A�
*

lossV�=���       �	��t&fc�A�
*

loss�/d=d���       �	�u&fc�A�
*

lossW�=W�       �	U0v&fc�A�
*

loss��\<}`�I       �	�v&fc�A�
*

loss�kj=`8k"       �	�ww&fc�A�
*

lossM�<��}g       �	5$x&fc�A�
*

loss��<O\B       �	�x&fc�A�
*

loss.�<����       �	^�y&fc�A�
*

lossvQ3<�0vp       �	-#z&fc�A�
*

loss]i=�u^�       �	�z&fc�A�
*

lossA��<���       �	�{&fc�A�
*

loss�4>�\�e       �	z9|&fc�A�
*

loss::=��j�       �	�|&fc�A�
*

loss�!={y0Z       �	w}&fc�A�
*

loss&��<ʆ�v       �	�~&fc�A�
*

loss��<��       �	d�~&fc�A�
*

loss
��;�u�>       �	�h&fc�A�
*

loss�ä<�A9/       �	}�&fc�A�
*

loss�W<�d��       �	˝�&fc�A�
*

lossW�=ԺM       �	"7�&fc�A�
*

lossD�=t�T�       �	��&fc�A�
*

lossSɃ=��       �	��&fc�A�
*

lossA�b=�MYV       �	���&fc�A�
*

lossD�a<+'I�       �	�O�&fc�A�
*

loss���=:��p       �	=�&fc�A�
*

loss��=��a�       �	���&fc�A�
*

loss�ÿ=�sذ       �	�M�&fc�A�
*

lossWd<P��       �	��&fc�A�
*

lossdY�=m��       �	���&fc�A�
*

lossWT�<��]       �	�;�&fc�A�
*

loss�L�='�       �	Uڈ&fc�A�
*

lossQ�<<顠j       �	~q�&fc�A�
*

loss�!�=@�D       �	�!�&fc�A�
*

loss)�r=��       �	ʊ&fc�A�
*

loss�j<�<[�       �	(d�&fc�A�
*

lossxT�=e��<       �	��&fc�A�
*

loss�d=��X       �	ծ�&fc�A�
*

lossQƔ=>�       �	�M�&fc�A�
*

lossl:�=G�'       �	��&fc�A�
*

lossIKT<�;�N       �	��&fc�A�
*

loss=r�<���       �	^H�&fc�A�
*

loss8�S=~DF�       �	��&fc�A�
*

loss�<"X�       �	͐�&fc�A�
*

loss�m=Z��A       �	1�&fc�A�
*

loss�o!>:%�       �	Jϑ&fc�A�
*

loss;��<?�2       �	em�&fc�A�
*

loss��=&��       �	��&fc�A�
*

losshug<4�fJ       �	6��&fc�A�
*

lossn�(;�� �       �	�F�&fc�A�
*

loss���<uE�i       �	G�&fc�A�
*

loss�2=j A       �	倕&fc�A�
*

loss>%�<&�e       �	z�&fc�A�
*

lossO��<�       �	@��&fc�A�
*

loss8�<nzg�       �	�i�&fc�A�
*

loss��=~���       �	��&fc�A�
*

lossh]=X!)?       �	#��&fc�A�
*

loss�H>==C�       �	�:�&fc�A�
*

lossD3I=�0�       �	��&fc�A�
*

loss�<=�u�       �	���&fc�A�
*

loss�Fx=(h��       �	�:�&fc�A�
*

loss�=B�j�       �	�қ&fc�A�
*

loss��%<����       �	m�&fc�A�
*

loss)��:Σ�b       �	h	�&fc�A�
*

loss?=�1i�       �	�J�&fc�A�
*

loss&��=���h       �	;�&fc�A�
*

loss,�L=$���       �	�a�&fc�A�
*

loss،�=�`��       �	F�&fc�A�
*

losss��<hվ       �	���&fc�A�
*

lossh=�=�̵�       �	��&fc�A�
*

loss�$6=�1R[       �	�D�&fc�A�
*

lossmʇ<��       �	D�&fc�A�
*

loss��w=���       �	�~�&fc�A�
*

loss�r<11j!       �	&�&fc�A�
*

loss��<���       �	�ƥ&fc�A�
*

loss���=�KJ       �	5^�&fc�A�
*

loss��>       �	���&fc�A�*

losst=�/�       �	獧&fc�A�*

lossM�|<�}A       �	n5�&fc�A�*

loss�c�=�;(�       �	�Ψ&fc�A�*

loss�)=��p       �	�x�&fc�A�*

loss��<����       �	+�&fc�A�*

loss��<[���       �	;��&fc�A�*

lossx��=�
�`       �	�B�&fc�A�*

loss���=��*       �	�1�&fc�A�*

loss-߬=�z&�       �	�ޭ&fc�A�*

lossr�=��P�       �	솮&fc�A�*

loss��=P5#�       �	a3�&fc�A�*

lossn�*>%_�       �	ܯ&fc�A�*

loss�p�=��J�       �	���&fc�A�*

lossXc<�ח       �	�0�&fc�A�*

loss�Q�=
J�       �	�˱&fc�A�*

loss�`�=�X��       �	�i�&fc�A�*

loss��<=���       �	�&fc�A�*

loss��I=�:�       �	���&fc�A�*

loss�ހ=���       �	]O�&fc�A�*

loss��='�H�       �	K�&fc�A�*

lossQ�<KXDH       �	%��&fc�A�*

lossF]=�'�       �	�8�&fc�A�*

loss�<�uZ!       �	a޶&fc�A�*

loss<��<��b�       �	yy�&fc�A�*

loss�2�<�j�z       �	�'�&fc�A�*

loss���<��`�       �	�¸&fc�A�*

loss��=w/<�       �	�[�&fc�A�*

loss-:<N��       �	S�&fc�A�*

loss洁<���       �	���&fc�A�*

lossd�=W��*       �	P�&fc�A�*

loss)}�=����       �	R�&fc�A�*

loss�QU<�N�O       �	%��&fc�A�*

lossD��<�ա;       �	�&�&fc�A�*

loss蝰<X= �       �	/ý&fc�A�*

lossZ�.=z���       �	uZ�&fc�A�*

loss�4�=�9��       �	��&fc�A�*

lossW��<X�n�       �	C��&fc�A�*

loss�3�<A+�       �	�#�&fc�A�*

loss3[<�U��       �	���&fc�A�*

loss
�<�4��       �	�P�&fc�A�*

loss��=��x       �	 ��&fc�A�*

loss/��=�4�b       �	���&fc�A�*

lossf�<e0       �	Z+�&fc�A�*

lossJ��=�)%&       �	��&fc�A�*

lossn/�=u���       �	�^�&fc�A�*

loss�<ˎ�p       �	.��&fc�A�*

loss(�<��       �	o��&fc�A�*

loss�j�<V$�       �	Y4�&fc�A�*

loss�1�=� �        �	���&fc�A�*

loss��-=��-       �	�b�&fc�A�*

lossk�=�Դ       �	��&fc�A�*

loss��=E��       �	|��&fc�A�*

loss�.�<�k�       �	2:�&fc�A�*

loss(i=�b�       �	73�&fc�A�*

lossw�S=w\8�       �	���&fc�A�*

loss��x<���       �	�&fc�A�*

lossR!.=n��i       �	�&fc�A�*

lossm�c;ݷ�K       �	���&fc�A�*

loss�X�=A(g�       �	�W�&fc�A�*

losswv�<���       �	���&fc�A�*

loss<�>����       �	���&fc�A�*

loss�4�<Z��+       �	xA�&fc�A�*

lossL�X=�"6�       �	���&fc�A�*

lossH�X;����       �	k��&fc�A�*

loss,D6=rk�       �	��&fc�A�*

loss��G=�=b}       �	��&fc�A�*

loss��=%�.�       �	W�&fc�A�*

loss�=�Q�       �	. �&fc�A�*

loss =z�#�       �	���&fc�A�*

lossj�<l��       �	�F�&fc�A�*

loss(M=�<x�       �	�L�&fc�A�*

loss��:=1�#�       �	~��&fc�A�*

loss��<�A�       �	s��&fc�A�*

lossR�(=��o�       �	�O�&fc�A�*

loss6�f=�ǟp       �	/��&fc�A�*

loss�{ =8-X?       �	S��&fc�A�*

loss\��=w�ڛ       �	%t�&fc�A�*

loss��>�P,       �	W&�&fc�A�*

loss��^=Q�)       �	>��&fc�A�*

loss�)�<q�       �	K��&fc�A�*

lossyv�<��Jg       �	�^�&fc�A�*

lossܶ�<Q"��       �	���&fc�A�*

lossc��;o6       �	��&fc�A�*

lossS8=v�I       �	�:�&fc�A�*

loss���;f��       �	��&fc�A�*

loss�`�;0.x�       �	{��&fc�A�*

loss���<�a�+       �	�"�&fc�A�*

loss�'�=�?�       �	m��&fc�A�*

loss/�1=Y��       �	O��&fc�A�*

loss�)�=f��/       �	o.�&fc�A�*

loss3�>o�J       �	���&fc�A�*

loss�*�=L�4�       �	 b�&fc�A�*

loss��<�        �	���&fc�A�*

loss.��<�a�       �	���&fc�A�*

loss�_�=���0       �	(+�&fc�A�*

loss֬�=�#�       �	g��&fc�A�*

loss��=�x�       �	׉�&fc�A�*

lossL�>|B�       �	�1�&fc�A�*

loss���;�a˕       �	�W�&fc�A�*

loss=B�= �       �	%�&fc�A�*

loss���<��W       �	t��&fc�A�*

lossoП<��a       �	��&fc�A�*

loss��<��)       �	>]�&fc�A�*

loss���<��";       �	�,�&fc�A�*

loss�$W=#���       �	m��&fc�A�*

loss���<����       �	E��&fc�A�*

loss�&=����       �	׆�&fc�A�*

loss�vI>軸A       �	ސ�&fc�A�*

loss�_h<5P�       �	�-�&fc�A�*

loss��H<lY
T       �	��&fc�A�*

loss�=�>X�       �	�u�&fc�A�*

lossn��<��       �	C �&fc�A�*

loss#=���B       �	,��&fc�A�*

lossJ"�=�@�       �	[]�&fc�A�*

lossD9r=�Sx       �	���&fc�A�*

loss�e\<EE��       �	d��&fc�A�*

loss�K=J	EE       �	�-�&fc�A�*

loss���=�=��       �	��&fc�A�*

lossz?�<��W@       �	n�&fc�A�*

losss=;/y�       �	�
�&fc�A�*

loss�݀=���       �	S��&fc�A�*

loss�6�='_�       �	@L�&fc�A�*

lossA�<�~s       �	d��&fc�A�*

loss��=<L��       �	�~�&fc�A�*

loss-:<��l�       �	��&fc�A�*

loss��<���       �	y��&fc�A�*

loss�p�=z�ɶ       �	�W�&fc�A�*

loss�&)=���       �	��&fc�A�*

loss���=G���       �	2��&fc�A�*

lossv=�a�D       �	�*�&fc�A�*

loss�կ=�Z�       �	i�&fc�A�*

loss�3�;QE�       �	S@�&fc�A�*

loss�b�<{�m       �	��&fc�A�*

loss]�<D��O       �	� 'fc�A�*

loss�k=�7-       �	�L'fc�A�*

loss4X�=�C�       �	�'fc�A�*

loss1u>R���       �	3�'fc�A�*

lossۂ=�1�       �	y<'fc�A�*

loss)<��!T       �	��'fc�A�*

loss��J=V��)       �	E�'fc�A�*

loss���<߭~�       �	1#'fc�A�*

loss{�p=
bӿ       �	��'fc�A�*

loss�r<ܳ        �	��'fc�A�*

loss3�=I�>O       �	jg'fc�A�*

loss��=Ó�e       �	'fc�A�*

loss�5=��_       �	��'fc�A�*

loss<�Z=p@��       �	C	'fc�A�*

loss�j6=J��m       �	��	'fc�A�*

loss�%�<E�
$       �	��
'fc�A�*

loss�h_<�j�1       �	�'fc�A�*

loss�� =J�       �	q�'fc�A�*

lossa�W=�}��       �	vO'fc�A�*

loss'�='�       �	��'fc�A�*

loss���<�<��       �	^�'fc�A�*

loss�sM=%jP       �	\'fc�A�*

loss��=$�k�       �	�'fc�A�*

loss�)�=����       �	~�'fc�A�*

loss ��=YG�       �	[['fc�A�*

loss��=�o6;       �	~�'fc�A�*

lossE�=���M       �	��'fc�A�*

lossP/=��[       �	JE'fc�A�*

lossq0�=[�       �	��'fc�A�*

lossD�?="��       �	��'fc�A�*

loss_0�<�C�       �	h'fc�A�*

lossͽ8=M���       �	
�'fc�A�*

loss�b�<�4R       �	�Q'fc�A�*

loss.Q�<��H       �	��'fc�A�*

lossÊ�=�q[       �	�~'fc�A�*

losshi�=H�;       �	�d'fc�A�*

loss�	"=���       �	a�'fc�A�*

loss��J=J��t       �	��'fc�A�*

lossX��=hs�(       �	�:'fc�A�*

loss�^�<ל�       �	��'fc�A�*

lossZ}�<�s�       �	�h'fc�A�*

loss���=��N       �	'fc�A�*

loss���<!���       �	E�'fc�A�*

loss��
=F0�       �	�B'fc�A�*

loss��m=L\�       �	F�'fc�A�*

losssҭ<��       �	��'fc�A�*

loss�%�=���J       �	�2'fc�A�*

losso��<��0       �	�'fc�A�*

loss�P�<`��       �	�x'fc�A�*

loss�p==>~G       �	W[ 'fc�A�*

loss:Vt<� �       �	� !'fc�A�*

loss?'�<���       �	��!'fc�A�*

loss7Ά=2��%       �	^f"'fc�A�*

loss8I=t��-       �	p{#'fc�A�*

loss�Կ=���~       �	x&$'fc�A�*

loss�0t<�w:       �	0�$'fc�A�*

loss\�k<��       �	׉%'fc�A�*

loss�Ĥ<�Xd       �	8&'fc�A�*

lossEsb=�s��       �	/�&'fc�A�*

loss@.�<�h��       �	Ց''fc�A�*

lossR�0<8��,       �	)?('fc�A�*

lossL�^=*j+�       �	%�('fc�A�*

loss�iq=����       �	��)'fc�A�*

loss��=]1�       �	n�*'fc�A�*

loss��_<��ʾ       �	�o+'fc�A�*

lossA !=���       �	�,'fc�A�*

loss�i=�L�S       �	�,'fc�A�*

lossk�	=s�M�       �	Eg-'fc�A�*

loss%;Q=T5��       �	#.'fc�A�*

loss��=j�       �	¿.'fc�A�*

lossџ8=�H/       �	�a/'fc�A�*

loss8�F=Ҵ�       �	�0'fc�A�*

loss�>�ӈ�       �	@�0'fc�A�*

loss�m�=�u       �	�Q1'fc�A�*

loss3�>��T�       �	d2'fc�A�*

loss/�<a��L       �	�2'fc�A�*

loss�s�<?��l       �	vS3'fc�A�*

lossbL=���	       �	�4'fc�A�*

loss.@U=���       �	ɰ4'fc�A�*

loss�a_<B޿}       �	�a5'fc�A�*

loss<�<^U)R       �	�6'fc�A�*

loss��<7V\       �	/�6'fc�A�*

loss�K>��a�       �	�q7'fc�A�*

lossr5�=��NS       �	�&8'fc�A�*

lossxMq=��T]       �	P�8'fc�A�*

loss @�<"�aO       �	��9'fc�A�*

loss��=#jn       �	�3:'fc�A�*

loss?=�2Q=       �	��:'fc�A�*

loss�Y=��l-       �	
�;'fc�A�*

loss �C=���       �	�5<'fc�A�*

lossSy�<��)w       �	��<'fc�A�*

loss�E"<�}�       �	�='fc�A�*

loss��g=�h��       �	T6>'fc�A�*

loss��4=�<r/       �	��>'fc�A�*

loss�J�=bD��       �	y�?'fc�A�*

loss��=K\CC       �	^@'fc�A�*

loss���<8�T�       �	�A'fc�A�*

lossc߯<���       �	=�A'fc�A�*

loss|_�<��%       �	t_B'fc�A�*

loss�j�=e�D       �	�C'fc�A�*

loss�ڏ=78�       �	�C'fc�A�*

loss�L=?�v�       �	�]D'fc�A�*

loss�]�=N{ƹ       �	�qE'fc�A�*

loss�n�;ŷ�A       �	F'fc�A�*

loss
u<����       �	x�F'fc�A�*

loss�=k7/f       �	��G'fc�A�*

loss�B�<D�4       �	7H'fc�A�*

loss��<����       �	q�H'fc�A�*

loss�ۣ<io��       �	��I'fc�A�*

loss���=�Û�       �	�8J'fc�A�*

loss�^<���v       �	�J'fc�A�*

loss���<��7       �	��K'fc�A�*

losso��=��P�       �	��L'fc�A�*

loss���=N�V�       �	�PM'fc�A�*

loss��l=ick�       �	Q�M'fc�A�*

loss"Y�<��o       �	(�N'fc�A�*

loss�>�</�       �	;7O'fc�A�*

loss/�< �f,       �	t�O'fc�A�*

loss���<���       �	#�P'fc�A�*

loss�0=�_�       �	�1Q'fc�A�*

loss#Y�<^�s�       �	��Q'fc�A�*

loss�"<򡀏       �	�hR'fc�A�*

loss�)>���       �	CS'fc�A�*

loss���<Ʌ2       �	��S'fc�A�*

lossZΝ<F�v�       �	�?T'fc�A�*

loss#=_3�?       �	��T'fc�A�*

loss�K�<��j�       �	�zU'fc�A�*

loss�v�<��Q�       �	V'fc�A�*

lossm9K<#��0       �	�V'fc�A�*

loss���=�"q�       �	QOW'fc�A�*

lossM<�5       �	��W'fc�A�*

loss� �=k�e       �	�X'fc�A�*

lossJ�="̛�       �	Z+Y'fc�A�*

lossv��<����       �	��Y'fc�A�*

loss�H�<b	��       �	��Z'fc�A�*

loss#�&=���       �	M�['fc�A�*

lossj��<(��       �	�\'fc�A�*

loss}�<��n�       �	D�\'fc�A�*

loss
�5>�g$       �	�X]'fc�A�*

loss�:>hp�y       �	�^'fc�A�*

loss�X�=��~�       �	�^'fc�A�*

loss3�<y��       �	�8_'fc�A�*

loss���=!7       �	=�_'fc�A�*

loss�}s=�,1�       �	~�`'fc�A�*

losse!+<�o�g       �	Z*a'fc�A�*

loss/�=��@�       �	6�a'fc�A�*

loss�;�=LI�Z       �	Zgb'fc�A�*

loss���<���       �	�c'fc�A�*

loss�K�=����       �	c�c'fc�A�*

loss���<4���       �	5`d'fc�A�*

loss��<�x�       �	�e'fc�A�*

loss2��<� =       �	e�e'fc�A�*

loss�Na=�z�       �	�Ef'fc�A�*

loss�s�;��s�       �	�f'fc�A�*

loss�d
=͙|�       �	{�g'fc�A�*

lossRc�=
�D        �	�(h'fc�A�*

lossܑ�<f�r       �	YLi'fc�A�*

loss�B=��H       �	\j'fc�A�*

loss��m=��       �	P�j'fc�A�*

loss9=�a�       �	 {k'fc�A�*

loss,��<�h�(       �	�l'fc�A�*

loss\A�<��E�       �	�Sm'fc�A�*

loss3�<�#l       �	��m'fc�A�*

loss�P�=RgO       �	3o'fc�A�*

lossԵ�<���       �	�p'fc�A�*

loss���<+�9(       �	�/q'fc�A�*

loss�ݳ<I�U�       �	(Cr'fc�A�*

loss�(�=<`�       �	�7s'fc�A�*

loss]T�<�F�C       �	��s'fc�A�*

lossY�<�[��       �	C�t'fc�A�*

loss+=)�d[       �	�Kv'fc�A�*

loss�==]5��       �	�qw'fc�A�*

losswZ�=��       �	4x'fc�A�*

loss�=�HJ       �	��x'fc�A�*

lossnGE=��x�       �	�2z'fc�A�*

loss@�{=�M��       �	W[{'fc�A�*

loss��<��	       �	�|'fc�A�*

loss��S<`tЎ       �	9F}'fc�A�*

loss�}=_f	�       �	~'fc�A�*

loss��<* l       �	�;'fc�A�*

loss�=�U��       �	�u�'fc�A�*

lossg<��       �	�ρ'fc�A�*

loss��*=�Ą       �	z�'fc�A�*

loss
�,=#��       �	�k�'fc�A�*

loss�u�<���       �	BC�'fc�A�*

lossU5�=Bkq       �	��'fc�A�*

lossrS�=��m�       �	���'fc�A�*

loss*Q;>��&q       �	��'fc�A�*

lossSO�<�gw       �	���'fc�A�*

loss%7=H���       �	G�'fc�A�*

lossҡ3<rf��       �	N�'fc�A�*

loss��<t�Ox       �	f��'fc�A�*

loss#�:ҿ+       �	�J�'fc�A�*

loss��}<{΂       �	��'fc�A�*

loss��.<k2       �	���'fc�A�*

lossj��;n�C�       �	GZ�'fc�A�*

lossz��<+.^h       �	��'fc�A�*

loss~^;��>�       �	���'fc�A�*

loss,�<p��V       �	aU�'fc�A�*

losst{<�=�       �	K �'fc�A�*

loss�ű:�ĝA       �	h��'fc�A�*

loss�m91��7       �	�b�'fc�A�*

loss`��<)�(       �	L�'fc�A�*

lossl7=�xE       �	�ɑ'fc�A�*

loss��<?�?       �	a��'fc�A�*

loss�pu:�6/       �	�A�'fc�A�*

loss��=���       �	O�'fc�A�*

loss-o3>�0�-       �	ۆ�'fc�A�*

loss�%=v�v       �	m<�'fc�A�*

lossh,�<gX       �	��'fc�A�*

loss��=�k�N       �	U��'fc�A�*

loss\��=ҕO[       �	"U�'fc�A�*

loss3��<#J��       �	��'fc�A�*

lossEc<��
�       �	+��'fc�A�*

loss���=D��       �	�L�'fc�A�*

loss|�Q=���U       �	��'fc�A�*

loss��N<��[,       �	s��'fc�A�*

loss�a�<x ��       �	�M�'fc�A�*

lossf	 =.ʣ�       �	]��'fc�A�*

lossQ4�=y��	       �	ݲ�'fc�A�*

loss�"n=�6P�       �	�m�'fc�A�*

loss�2!=�/Q       �	��'fc�A�*

loss`d�<)���       �	˺�'fc�A�*

loss�=# �       �	� 'fc�A�*

loss�=�M�       �	�s�'fc�A�*

loss`i8=5�Q�       �	J$�'fc�A�*

lossљ�=C��       �	�Ԣ'fc�A�*

loss��=�`Qh       �	-y�'fc�A�*

loss�X�<�z��       �	0�'fc�A�*

lossF|I=j�k       �	n�'fc�A�*

loss�`�<E�K�       �	ς�'fc�A�*

loss���<���       �	�=�'fc�A�*

loss�=��Q�       �	w�'fc�A�*

lossq+�;���       �	�ħ'fc�A�*

losswN�=�ȅ!       �	�{�'fc�A�*

loss<��< 5|       �	:[�'fc�A�*

loss6��=���       �	�#�'fc�A�*

lossp�=���       �	�7�'fc�A�*

loss�<�<��       �	N^�'fc�A�*

loss�X=�� �       �	g��'fc�A�*

losstXr=F�{�       �	"o�'fc�A�*

loss:�I<��9�       �	��'fc�A�*

loss֥�<�$̧       �	�M�'fc�A�*

lossvu%<x~d�       �	`�'fc�A�*

lossџ="�9�       �	!��'fc�A�*

loss(@=��c�       �	�_�'fc�A�*

lossieo=w�o�       �	��'fc�A�*

loss]`�=XU^�       �	��'fc�A�*

loss:=�3��       �	�Դ'fc�A�*

lossm� <5-�o       �	x�'fc�A�*

loss���<�xb�       �	�*�'fc�A�*

loss�~2=��ק       �	�˶'fc�A�*

loss�[ =~�[�       �	p
�'fc�A�*

loss�ö=����       �	73�'fc�A�*

loss�ُ=Mʰ       �	!#�'fc�A�*

loss�F=���       �	�պ'fc�A�*

loss���=z�y�       �	���'fc�A�*

loss@J�;?��       �	�j�'fc�A�*

lossZ�=���i       �	5^�'fc�A�*

loss}KT={�I       �	l��'fc�A�*

lossQf=��       �	�c�'fc�A�*

loss�=�*@       �	��'fc�A�*

loss�Դ=�x�       �	���'fc�A�*

loss�UR=z���       �	�U�'fc�A�*

loss&�|<b�       �	x�'fc�A�*

loss�ml=~��'       �	׿�'fc�A�*

loss_*<=�a��       �	Xq�'fc�A�*

loss(�>[�l5       �	��'fc�A�*

loss�_=}���       �	i��'fc�A�*

loss�g<�Qqj       �	ao�'fc�A�*

lossRF=�b"�       �	�%�'fc�A�*

loss,�={*C�       �	���'fc�A�*

loss�B�<���@       �	�}�'fc�A�*

lossw�G=���       �	A+�'fc�A�*

loss\&�=��w       �	���'fc�A�*

loss�c�;��l@       �	){�'fc�A�*

loss�<�n�%       �	�(�'fc�A�*

loss���<:$=�       �	���'fc�A�*

lossC% =�(�_       �	j�'fc�A�*

lossw%=Nh��       �	��'fc�A�*

loss/h>];��       �	'��'fc�A�*

loss�@�<"<�       �	~q�'fc�A�*

lossm�$=�$��       �	��'fc�A�*

loss���<��[r       �	�$�'fc�A�*

lossu=��4p       �	���'fc�A�*

loss[�<6�Z�       �	�k�'fc�A�*

loss�~�<5A��       �	.;�'fc�A�*

loss�S�<g��       �	�X�'fc�A�*

lossc��<[x��       �	C�'fc�A�*

loss�b2=�u�       �	��'fc�A�*

lossw�;%Mz       �	ӿ�'fc�A�*

loss��k=iMA�       �	y��'fc�A�*

lossC� =�R��       �	ݲ�'fc�A�*

loss��n<�RL       �	.��'fc�A�*

loss�U�=�adV       �	���'fc�A�*

loss���<tp�R       �	���'fc�A�*

loss���<�(c�       �	��'fc�A�*

loss?2�=@	��       �	�m�'fc�A�*

loss�$�=�bc�       �	\�'fc�A�*

loss[`�<{��       �	*��'fc�A�*

lossZf�<�I(j       �	�'fc�A�*

loss���;�;��       �	���'fc�A�*

loss��<TS��       �	;T�'fc�A�*

loss�%==Bb�       �	U��'fc�A�*

loss��.=EZ`X       �	��'fc�A�*

loss���;�-�'       �	~V�'fc�A�*

loss��	=��SY       �	��'fc�A�*

lossxI�=��       �	C��'fc�A�*

lossv2�;�v<       �	�W�'fc�A�*

loss\4�;_9       �	���'fc�A�*

loss׵S=l�r       �	|��'fc�A�*

loss�9<��Q�       �	�:�'fc�A�*

loss�=&Yb       �	5��'fc�A�*

loss��>=~s��       �	^��'fc�A�*

loss[3<���<       �	�X (fc�A�*

loss�Q�;�1{G       �	�
(fc�A�*

loss�a�;����       �	��(fc�A�*

losss�<pm�       �	!u(fc�A�*

loss��=�{�
       �	�(fc�A�*

loss{�Z=� o�       �	E�(fc�A�*

loss���<8�       �	6[(fc�A�*

loss���;h��f       �	Q�(fc�A�*

loss=���       �	W�(fc�A�*

lossr�<��kS       �	z8(fc�A�*

loss{��<�4       �	�(fc�A�*

loss�fH=%�5-       �	�k(fc�A�*

lossN��<$19       �	(fc�A�*

loss�J�=)��       �	��(fc�A�*

lossA!r=%��       �	�H	(fc�A�*

loss�c�<�su�       �	G�	(fc�A�*

lossrq6<r[n       �	8�
(fc�A�*

loss�=���;       �	/4(fc�A�*

loss�$=����       �	`�(fc�A�*

loss���<L�kS       �	�c(fc�A�*

loss3"�;��0       �	{�(fc�A�*

loss]�<�A��       �	[�(fc�A�*

lossa �<"9�       �	,,(fc�A�*

loss�6�<��4       �	��(fc�A�*

lossҭ=��e       �	@�(fc�A�*

loss�J=�V�       �	(fc�A�*

losskq>�1��       �	ڮ(fc�A�*

loss:�=��|       �	�U(fc�A�*

loss�]�=Dv�       �	v�(fc�A�*

lossS�=�ۜ�       �	��(fc�A�*

loss(�<�<�H       �	�b(fc�A�*

loss�U�=��~       �	�	(fc�A�*

loss\6:=�L�       �	)�(fc�A�*

lossb�<�0�       �	�V(fc�A�*

loss�L$=�v�@       �	��(fc�A�*

lossX{<Nl�       �	͑(fc�A�*

lossi�5=��&�       �	�7(fc�A�*

losshI�<s��       �	5�(fc�A�*

loss/��<�� h       �	u(fc�A�*

losse�=l0��       �	�(fc�A�*

loss�6<uP�       �	��(fc�A�*

loss�F�=3Ⱥ       �	�O(fc�A�*

loss��A<��%       �	��(fc�A�*

loss�{=��Fr       �	��(fc�A�*

loss���<�d��       �	�(fc�A�*

loss�S=���       �	��(fc�A�*

loss�S<�h3�       �	
d(fc�A�*

loss<8�<�v��       �	.(fc�A�*

loss��=�Rs       �	A�(fc�A�*

loss 	=�2Γ       �	H2(fc�A�*

loss��=󪚃       �	�(fc�A�*

loss�*�<���       �	,} (fc�A�*

lossQ�J=���~       �	�$!(fc�A�*

loss�jW=Yqv;       �	9�!(fc�A�*

lossWK�<��i       �	�k"(fc�A�*

loss��=�}       �	�#(fc�A�*

loss�j�<���       �	m�#(fc�A�*

loss���<r�       �	�F$(fc�A�*

loss=�)<)��       �	-�$(fc�A�*

lossD� ={'�       �	��%(fc�A�*

loss�`�<�"y�       �	�/&(fc�A�*

loss�ڀ=�F�       �	 �&(fc�A�*

loss���=
I��       �	�d'(fc�A�*

loss))=��o�       �	x	((fc�A�*

loss4��=�WN       �	�((fc�A�*

loss1V�<�S�       �	�U)(fc�A�*

loss��<�$##       �	�U*(fc�A�*

loss�ET<�(;�       �	�*(fc�A�*

lossq�S=6:7       �	A�+(fc�A�*

losshv<#��*       �	�],(fc�A�*

loss8v=�,`       �	�&-(fc�A�*

loss)}P<�"��       �	��-(fc�A�*

loss=�<K�;�       �	�h.(fc�A�*

loss�-�<�u�       �	r/(fc�A�*

loss��=��dL       �	��/(fc�A�*

loss�N�;�F2=       �	�F0(fc�A�*

loss�S�<@��       �	P�0(fc�A�*

loss�n$=��j       �	�{1(fc�A�*

lossT@�<L(��       �	�+2(fc�A�*

loss��d=tºT       �	7�2(fc�A�*

loss7%�</<f       �	5c3(fc�A�*

loss��:<��d       �	��3(fc�A�*

lossJ�<�R��       �	��4(fc�A�*

loss)o~<m��p       �	�05(fc�A�*

loss��;���G       �	��5(fc�A�*

loss�p<��c�       �	�o6(fc�A�*

loss�;5=V���       �	9	7(fc�A�*

loss��<�)��       �	�7(fc�A�*

loss���=�#5S       �	G>8(fc�A�*

loss��>V8       �	M�8(fc�A�*

lossV��<�l�
       �	is9(fc�A�*

lossx�=�9ڑ       �	�:(fc�A�*

lossP<�ڸ       �	��:(fc�A�*

loss6��;��&(       �	$B;(fc�A�*

loss�b�=���       �	��;(fc�A�*

lossqt,<���       �	�t<(fc�A�*

loss��m=
=؈       �	�=(fc�A�*

loss�A�<C��       �	e�=(fc�A�*

lossf߲=�aT�       �	?>(fc�A�*

lossfq�<\�ä       �	#�>(fc�A�*

lossAs|<�K�       �	ǂ?(fc�A�*

loss}�<T�F       �	t'@(fc�A�*

loss1L=�a       �	V�@(fc�A�*

lossy�="�h       �	(~A(fc�A�*

loss�Q<S�F       �	YB(fc�A�*

loss���<��s�       �	8�B(fc�A�*

loss��!=�;T       �	��C(fc�A�*

loss%1�<��"�       �	��D(fc�A�*

loss�g<]���       �	�4E(fc�A�*

loss]�<<a�       �	�E(fc�A�*

loss��<����       �	�uF(fc�A�*

lossrv@<(���       �	h$G(fc�A�*

loss.�
=��$       �	�]H(fc�A�*

loss���<�h�[       �	�I(fc�A�*

loss/̵<-�$       �	h�I(fc�A�*

lossT��=��A       �	�]J(fc�A�*

loss ?�;���       �	�K(fc�A�*

loss�1<�b       �	�K(fc�A�*

loss�=��y4       �	�OL(fc�A�*

loss���;���       �	��L(fc�A�*

loss��<�g�A       �	�M(fc�A�*

loss6��<�
l�       �	�!N(fc�A�*

lossﴏ=��D�       �	��N(fc�A�*

loss�o�=�N �       �	�vO(fc�A�*

lossH*�;�Y>�       �	*P(fc�A�*

lossL�B;n���       �	ݵP(fc�A�*

loss�ɝ<��g�       �	�QQ(fc�A�*

lossj�<]T       �	k�Q(fc�A�*

loss�F<i�g�       �		�R(fc�A�*

loss���<�
|�       �	�S(fc�A�*

loss�J�<�{�       �	K�S(fc�A�*

loss!g�=K�qQ       �	�nT(fc�A�*

loss���<M���       �	fU(fc�A�*

loss�=<�۠�       �	�U(fc�A�*

loss&E�=��d       �	 EV(fc�A�*

loss.з<l�@�       �	m�V(fc�A�*

loss���<��h�       �	�W(fc�A�*

loss�4=��cx       �	�X(fc�A�*

loss�<��%       �	��X(fc�A�*

loss�:TaY�       �	�lY(fc�A�*

loss��<� �       �	@Z(fc�A�*

lossJ��<?>?       �	��Z(fc�A�*

lossT�B<K�~       �	[[(fc�A�*

loss�^O>�!�        �	~�[(fc�A�*

loss��8<�t�       �	V�\(fc�A�*

loss�D�==��
       �	�=](fc�A�*

loss}<�s�f       �	��](fc�A�*

loss}�< �(5       �	9^(fc�A�*

loss%r=>���       �	�_(fc�A�*

lossh�O=㔹�       �	�_(fc�A�*

loss#�<F�y�       �	�^`(fc�A�*

lossA��=d�$t       �	z�`(fc�A�*

loss�x�=�h��       �	��a(fc�A�*

loss<��<�~z�       �	:b(fc�A�*

loss��=��_~       �	x�b(fc�A�*

loss��=׏[�       �	znc(fc�A�*

loss��2=�S�       �	�d(fc�A�*

lossC �<k��       �	P�d(fc�A�*

loss��;�g��       �	�?e(fc�A�*

loss4.�<���       �	��e(fc�A�*

loss�T=U�       �	�mf(fc�A�*

loss<��=廴H       �	�Fg(fc�A�*

loss���=�v*�       �	'�g(fc�A�*

lossڲ�=Nj�        �	E�h(fc�A�*

lossw��=����       �	Qi(fc�A�*

lossZ)�<6�@
       �	m�i(fc�A�*

lossA��<j���       �	�|j(fc�A�*

loss�/q<��f�       �	�k(fc�A�*

loss�X�=W��       �	��k(fc�A�*

loss4��;�$�Y       �	T;l(fc�A�*

loss_��<矖\       �	��l(fc�A�*

loss�p=��S�       �	�rm(fc�A�*

lossE�Q<�_x       �	yn(fc�A�*

loss]�<�*v       �	�n(fc�A�*

loss��{=��K       �	EFo(fc�A�*

loss@��<E��       �	c�o(fc�A�*

loss%H)<7��G       �	��p(fc�A�*

lossߥ�<�       �	�$q(fc�A�*

lossἕ<'�O       �	׾q(fc�A�*

loss�q=8<�       �	Yr(fc�A�*

loss��<���       �	��r(fc�A�*

loss=7�<v��       �	`�s(fc�A�*

lossfڒ=+�6�       �	6;t(fc�A�*

lossό�=���       �	~�t(fc�A�*

loss1�;�r       �	{u(fc�A�*

lossí�;cl��       �	�v(fc�A�*

loss�/,=j���       �	��v(fc�A�*

loss��Q=�.�       �	�dw(fc�A�*

loss��<1���       �	wx(fc�A�*

loss�׏=z��       �	��x(fc�A�*

lossH�<��B�       �	_^y(fc�A�*

lossƁ<Z*Cb       �	��y(fc�A�*

loss���<�y       �	��z(fc�A�*

loss3�;� �%       �	>A{(fc�A�*

loss{<�Ш�       �	Y�{(fc�A�*

lossH��<�=�W       �	uv|(fc�A�*

loss��	>\�       �	U}(fc�A�*

loss��<'4�       �	�}(fc�A�*

loss�;P<�5S�       �	�W~(fc�A�*

loss%m_;B\�(       �	��~(fc�A�*

loss��q</�       �	˜(fc�A�*

loss$==��a       �	gH�(fc�A�*

loss�c=ppH       �	��(fc�A�*

loss�u=����       �	���(fc�A�*

loss!=D��       �	�5�(fc�A�*

lossD&�=�&��       �	>Ђ(fc�A�*

lossL��<� 0�       �	/m�(fc�A�*

loss:d<ų��       �	��(fc�A�*

loss��;�q �       �	s��(fc�A�*

loss�v=�朎       �	.<�(fc�A�*

lossC�;l!�       �	�Ӆ(fc�A�*

loss��=𢀥       �	Lp�(fc�A�*

loss�=:kb�       �	p	�(fc�A�*

lossv��=�c��       �	���(fc�A�*

loss
�s=���)       �	|G�(fc�A�*

lossY+=��z       �	\�(fc�A�*

loss���<�Kp       �	��(fc�A�*

loss8�N=Z��       �	��(fc�A�*

loss��=-{`�       �	���(fc�A�*

loss=� &�       �	�\�(fc�A�*

loss8�=�p��       �	/��(fc�A�*

loss�<GG�       �	0��(fc�A�*

loss���;Pv�       �	�H�(fc�A�*

loss;[�=$D�H       �	���(fc�A�*

loss�G�<4݄�       �	0��(fc�A�*

loss2��;��       �	B�(fc�A�*

loss�;���]       �	c�(fc�A�*

loss�Y�;�yeb       �	x��(fc�A�*

loss1ƣ<�{�D       �	B�(fc�A�*

loss�h�<*�ms       �	c�(fc�A�*

loss=�&=�82�       �	1��(fc�A�*

loss���<~�7       �	F�(fc�A�*

loss�</�_]       �	��(fc�A�*

lossC�>=���       �	�(fc�A�*

lossr<<2Z��       �	�+�(fc�A�*

lossj;�(	�       �	ƕ(fc�A�*

lossnwV=��T�       �	p_�(fc�A�*

loss��G;\�<       �	p�(fc�A�*

loss �<�b�       �	���(fc�A�*

loss.��<J��       �	�N�(fc�A�*

loss߰z=�ׂ       �	��(fc�A�*

loss��=<zθo       �	Ȗ�(fc�A�*

loss�M.=�3��       �	�5�(fc�A�*

lossqW#=jx�D       �	��(fc�A�*

lossz�=�]�        �	կ�(fc�A�*

loss8C�<T��       �	,G�(fc�A�*

loss�-�<��g       �	��(fc�A�*

loss�==�t7       �	tz�(fc�A�*

loss�"{=�@�       �	��(fc�A�*

lossA8p==B:�       �	���(fc�A�*

lossn>�=�       �	ץ�(fc�A�*

lossA�F;*�iN       �	�K�(fc�A�*

loss�=A=���H       �	z�(fc�A�*

loss�5e=�-       �	XƢ(fc�A�*

loss��=˃Ҳ       �	�a�(fc�A�*

lossHX=�)ܪ       �	���(fc�A�*

losse�="ُ�       �	ʥ�(fc�A�*

loss��I=�k��       �	0J�(fc�A�*

lossO�=��0]       �	�ݥ(fc�A�*

loss�o=���@       �	�Φ(fc�A�*

loss=�ʝo       �	�i�(fc�A�*

lossa֑; 5�       �	��(fc�A�*

loss(��<�]�9       �	⫨(fc�A�*

loss��-=7       �	�V�(fc�A�*

lossX��<��H�       �	���(fc�A�*

loss�=V��o       �	��(fc�A�*

loss_r	>x�?       �	�G�(fc�A�*

lossE�9=�       �	��(fc�A�*

loss���;>��2       �	���(fc�A�*

lossaҔ<c5�       �	0�(fc�A�*

loss`oH=�8�@       �	K̭(fc�A�*

loss���<����       �	�b�(fc�A�*

lossO~=)z��       �	a��(fc�A�*

loss�oH=W%S�       �	_��(fc�A�*

loss-I=�g>�       �	��(fc�A�*

lossq�U=&��       �	?R�(fc�A�*

lossZ͂<�c�       �	d�(fc�A�*

loss�i<0��       �	*��(fc�A�*

lossE=1@�I       �	C�(fc�A�*

lossZjv=���       �	�̷(fc�A�*

lossֲ�<��&�       �	���(fc�A�*

loss��=#��]       �	���(fc�A�*

lossxN#<
�u�       �	81�(fc�A�*

loss�<u�!_       �	�к(fc�A�*

loss#�;��w       �	�u�(fc�A�*

lossTP;.�Y7       �	��(fc�A�*

loss��b=u(Rb       �	�¼(fc�A�*

loss�\=� �>       �	�^�(fc�A�*

losst�>=�}       �	��(fc�A�*

loss���=YAM�       �	(fc�A�*

lossv7=;��       �	�8�(fc�A�*

lossw�=��A�       �	���(fc�A�*

loss��=��3       �	���(fc�A�*

lossW��<��X�       �	Y0�(fc�A�*

loss<�<�ܬ�       �	���(fc�A�*

lossϠ�<��       �	���(fc�A�*

loss��;)�Y�       �	~�(fc�A�*

loss��+<�+ì       �	߿�(fc�A�*

loss��<G �       �	�[�(fc�A�*

lossp�=��
�       �	���(fc�A�*

loss�<��v�       �	b��(fc�A�*

lossJA<W���       �	�D�(fc�A�*

loss�-=�]�       �	�P�(fc�A�*

loss�o=�[Sr       �	���(fc�A�*

loss�
0=���        �	��(fc�A�*

loss��=S���       �	r7�(fc�A�*

loss<��<1��       �	���(fc�A�*

loss1�s=�LT�       �	z�(fc�A�*

loss�L�=�a,�       �		�(fc�A�*

loss���<�[       �	���(fc�A�*

loss�nN=���N       �	�L�(fc�A�*

loss7��<rD�       �	`��(fc�A�*

lossq�r<�a�k       �	��(fc�A�*

loss	�<QF�/       �	�'�(fc�A�*

lossT��=@�f�       �	���(fc�A�*

loss��3=�t&�       �	]S�(fc�A�*

lossaə<�ۋ       �	���(fc�A�*

loss#ʽ<���'       �	���(fc�A�*

loss�G=W�f       �	E*�(fc�A�*

loss�ފ=�F       �	���(fc�A�*

lossD@=��ba       �	e�(fc�A�*

lossײ)>��       �	��(fc�A�*

loss��D=�
�p       �	[��(fc�A�*

loss��,=4���       �	*6�(fc�A�*

loss�r$=aԩ       �	���(fc�A�*

lossҹ`=��]�       �	mp�(fc�A�*

loss��a=K�0u       �	c�(fc�A�*

lossj�>���       �	���(fc�A�*

loss�6<Y�       �	�X�(fc�A�*

loss��< �&�       �	���(fc�A�*

lossE=ҩ�E       �	U��(fc�A�*

loss��<=f��6       �	z5�(fc�A�*

lossW�=��Ew       �	W��(fc�A�*

loss���;J���       �	�j�(fc�A�*

loss��<�*��       �	��(fc�A�*

loss��<;-�       �	���(fc�A�*

lossO8D=N�*3       �	�F�(fc�A�*

lossv��<ү��       �	B��(fc�A�*

loss΃M=�X;       �	��(fc�A�*

loss���<��[n       �	�$�(fc�A�*

loss�}�=��ڰ       �	���(fc�A�*

loss��=�5        �	�k�(fc�A�*

loss�{-;g7t       �	_�(fc�A�*

loss��6<V.�T       �	`��(fc�A�*

loss�`!=���_       �	�L�(fc�A�*

loss �8<�Y��       �	y��(fc�A�*

loss�G�;�W       �	��(fc�A�*

lossF==$>��       �	i�(fc�A�*

loss�=�CJ       �	��(fc�A�*

lossG-=�Ģ{       �	R�(fc�A�*

loss-b8=܄k�       �	}��(fc�A�*

loss�n/=�9       �	��(fc�A�*

loss�T=J!�}       �	.�(fc�A�*

lossC��<��       �	���(fc�A�*

loss{$=���       �	<h�(fc�A�*

lossi2�=�<�,       �	��(fc�A�*

loss�RH=�ta4       �	ӡ�(fc�A�*

loss��<�]�$       �	.;�(fc�A�*

loss߂�<��eY       �	���(fc�A�*

loss\88=��E�       �	7q�(fc�A�*

lossr)=��b       �	o�(fc�A�*

loss�({;�4x�       �	��(fc�A�*

loss��<H�lB       �	�s�(fc�A�*

lossܘC=���       �	��(fc�A�*

lossk�=����       �	v��(fc�A�*

loss}�<s<q�       �	#K�(fc�A�*

loss�>Y=ޘp�       �	��(fc�A�*

loss���<�`��       �	�v�(fc�A�*

loss���=�o��       �	��(fc�A�*

lossS�=$���       �	���(fc�A�*

loss�s�<��5       �	�F�(fc�A�*

loss�_�<�%�       �	��(fc�A�*

lossX1U=��Q       �	���(fc�A�*

loss{N�;�M       �	�(fc�A�*

loss��R<�׊[       �	���(fc�A�*

loss@��<ub��       �	AJ�(fc�A�*

loss�1=�E�P       �	���(fc�A�*

loss��<��.�       �	��(fc�A�*

loss
�v=�(M�       �	&�(fc�A�*

losshp�<��h�       �	@��(fc�A�*

loss�f�<��k�       �	 T�(fc�A�*

loss
��<'�c       �	��(fc�A�*

loss\�=�,_?       �	7��(fc�A�*

lossr@=NTJ�       �	��(fc�A�*

loss�ϡ<�W��       �	A��(fc�A�*

loss���=��a       �	]�(fc�A�*

loss��W=�m�       �	8��(fc�A�*

loss�	�<g��       �	=��(fc�A�*

loss:W=q�V       �	JD�(fc�A�*

loss��<�/$a       �	���(fc�A�*

loss�"4<B���       �	��(fc�A�*

loss*�
>b�       �	$�(fc�A�*

loss���<s��       �	��(fc�A�*

loss˝�=��=~       �	m�(fc�A�*

loss��r=nFI,       �	�)fc�A�*

lossᇛ=\�E#       �	Ҩ)fc�A�*

loss�-{<V|6
       �	f�)fc�A�*

loss��=�X�C       �	�%)fc�A�*

loss��<�e��       �	2�)fc�A�*

loss��=��H       �	�)fc�A�*

loss�'Y=\K�0       �	�')fc�A�*

loss�=�f�s       �	v�)fc�A�*

lossXy�<B�EX       �	cc)fc�A�*

lossQ��<��17       �	i)fc�A�*

losss�R=0C�*       �	�	)fc�A�*

loss<";=���A       �	��	)fc�A�*

loss&,c<O�O       �	@
)fc�A�*

loss�=�]��       �	=�
)fc�A�*

lossTC�=uc�       �	\�)fc�A�*

loss�%�=�
Q       �	JD)fc�A�*

loss��<C93�       �	��)fc�A�*

loss�.�<���       �	��)fc�A�*

loss�A�;�4       �	�.)fc�A�*

lossc�<�"�       �	=�)fc�A�*

loss�i�:ӑ�       �	��)fc�A�*

loss%��=���       �	��)fc�A�*

loss�q<�]�r       �	�9)fc�A�*

lossA,=�(�       �	��)fc�A�*

loss���=���z       �	q)fc�A�*

losso�<��       �	К)fc�A�*

loss��=|{��       �	�5)fc�A�*

loss��:�Ǹ�       �	B�)fc�A�*

loss?y�=�>]       �	�e)fc�A�*

loss���<l�x?       �	�)fc�A�*

loss��=ӇU       �	�)fc�A�*

loss&D�=�P��       �	{3)fc�A�*

lossOT=���       �	��)fc�A�*

loss���<�ϝ1       �	Bx)fc�A�*

loss�b>¯�8       �	{)fc�A�*

loss��d=i��i       �	G�)fc�A�*

loss��<h�R�       �	�Y)fc�A�*

loss�̸<��       �	�&)fc�A�*

losss6'=��c�       �	ӿ)fc�A�*

lossv0�;	Ʃ       �	!v)fc�A�*

loss���=�X�       �	0))fc�A�*

lossGQ�=p��       �	�)fc�A�*

loss��0=�?�       �	��)fc�A�*

lossb �;ϛ�       �	�)fc�A�*

loss�!=���n       �	.7 )fc�A�*

loss�j<��Ŋ       �	�� )fc�A�*

loss���<k�.-       �	7�!)fc�A�*

lossߖ�<7�q       �	�%")fc�A�*

loss�#=	�O^       �	�")fc�A�*

loss� =c��       �	�f#)fc�A�*

loss�4=�Ѕ       �	k$)fc�A�*

loss��<�N@       �	�$)fc�A�*

loss�$�;ț�X       �	D%)fc�A�*

loss�c%=d��       �	T�%)fc�A�*

loss���<���       �	�{&)fc�A�*

loss.#�<m�E�       �	�')fc�A�*

loss{��<�u�=       �	P�')fc�A�*

loss1�<4�Z       �	�^()fc�A�*

loss�6G<�N       �	))fc�A�*

lossP�<t��       �	{�))fc�A�*

loss�Yj=��Rw       �	�;*)fc�A�*

loss�s�:I��m       �	��*)fc�A�*

loss�\=g�       �	��+)fc�A�*

loss�g�;�b��       �	K�,)fc�A�*

loss�ݖ=H�       �	�J-)fc�A�*

loss�V"<VdJ>       �	��-)fc�A�*

loss3I�<?\N       �	i�.)fc�A�*

lossEOZ=���       �	W'/)fc�A�*

loss>=�lf       �	��/)fc�A�*

loss�a#<VL       �	�w0)fc�A�*

losst�S=ļ�       �	�1)fc�A�*

lossC!�;��m       �	��1)fc�A�*

loss�=_=K�j%       �	�F2)fc�A�*

loss�/K<3���       �	]�2)fc�A�*

loss.��;P���       �	��3)fc�A�*

losst��<����       �	s�4)fc�A�*

lossX� <����       �	��5)fc�A�*

loss})L=��-�       �	�56)fc�A�*

loss��=%���       �	�6)fc�A�*

lossڥ�=�(B�       �	!x7)fc�A�*

loss��<����       �	�?8)fc�A�*

loss�5Z<T��       �	�8)fc�A�*

loss�=�<��       �	܂9)fc�A�*

loss���<��B       �	:)fc�A�*

loss��;��       �	֭:)fc�A�*

lossU
;��(�       �	�O;)fc�A�*

loss��r=�+1L       �	��;)fc�A�*

loss
)];���.       �	��<)fc�A�*

loss��<��H�       �	~=)fc�A�*

loss��%;�"�       �	k�=)fc�A�*

loss(��;�R_X       �	qV>)fc�A�*

losscZn=U�y       �	�>)fc�A�*

loss��9��t       �	�?)fc�A�*

lossv��9wo�        �	 ^@)fc�A�*

loss���;}��       �	�A)fc�A�*

loss�<����       �	Y�A)fc�A�*

losss�<���       �	n�B)fc�A�*

loss���9v�8�       �	'0C)fc�A�*

loss��==�1D       �	�C)fc�A�*

lossNBg>�U�O       �	6xD)fc�A�*

lossX�<��#�       �	F"E)fc�A�*

lossT�<�ڣ       �	�E)fc�A�*

loss�<��Q�       �	l�F)fc�A�*

lossO]=E��2       �	z9G)fc�A�*

lossn7�<�5��       �	O�G)fc�A�*

loss�X�<y�ڛ       �	�H)fc�A�*

loss��(=��8       �	�?I)fc�A�*

lossv�</�>�       �	üJ)fc�A�*

loss��X<���       �	�]K)fc�A�*

loss��T=��0
       �	a�K)fc�A�*

loss��=���n       �	3�L)fc�A�*

lossŰN=g�Z�       �	|BM)fc�A�*

loss؄A=0�       �	h�M)fc�A�*

loss(��<k��(       �	��N)fc�A�*

loss�Di=�G�I       �	�4O)fc�A�*

loss��[<*�7       �	��O)fc�A�*

loss�h|=�$�       �	)vP)fc�A�*

lossY��<9Ι       �	�$Q)fc�A�*

lossO9�=/6��       �	��Q)fc�A�*

loss���<��\       �	^fR)fc�A�*

loss� �<j�m�       �	�S)fc�A�*

lossM�<���e       �	z�S)fc�A�*

loss�W[<ݎS       �	�QT)fc�A�*

lossI"<E5�       �	��T)fc�A�*

loss�< '��       �	�U)fc�A�*

loss�*<?c�#       �	�7V)fc�A�*

lossƭ�<Dԓ:       �	c�V)fc�A�*

loss<%�<K��       �	�lW)fc�A�*

loss[O�=\/�E       �	5
X)fc�A�*

lossC�j=k�       �	��X)fc�A�*

loss �;R��       �	>Y)fc�A�*

loss3��<>[D       �	2Z)fc�A�*

loss~<�ӹ       �	3�Z)fc�A�*

lossyU<�?
       �	�C[)fc�A�*

loss��|=���       �	D�[)fc�A�*

loss��;׶�       �	#�\)fc�A�*

loss�n};�i �       �	y])fc�A�*

loss�<̆       �	��])fc�A�*

loss&�=~��       �	�c^)fc�A�*

lossf
=A��       �	��^)fc�A�*

losslN;��g       �	�_)fc�A�*

lossC%�<�O��       �	+`)fc�A�*

losssm�<}**�       �	�`)fc�A�*

loss��x=�|�       �	5`a)fc�A�*

loss]�;0�.       �	7�a)fc�A�*

lossm��<���8       �	ϟb)fc�A�*

loss�Z+=�1       �	�9c)fc�A�*

loss�p�<���       �	��c)fc�A�*

lossz�<��f       �	�kd)fc�A�*

loss��><ƌ��       �	Se)fc�A�*

loss$.<y���       �	
�e)fc�A�*

loss���<���       �	+�)fc�A�*

loss��=�$�       �	��)fc�A�*

loss�o�=v���       �	E��)fc�A�*

loss��=�`s�       �	7S�)fc�A�*

lossZ�B=ԕ�#       �	�)fc�A�*

loss�'.<�}?�       �	���)fc�A�*

loss|/�<��{       �	��)fc�A�*

loss�ď<�*�       �	���)fc�A�*

loss�$�<4㻢       �	^L�)fc�A�*

lossv��<_>d       �	q�)fc�A�*

loss��T<�Ϣ�       �	p|�)fc�A�*

loss!��=!v�       �	}w�)fc�A�*

lossm�<B(+       �	z��)fc�A�*

losss\V=��׏       �	:"�)fc�A�*

loss��
=��)a       �	���)fc�A�*

loss�m=Qn̯       �	�^�)fc�A�*

loss�IT;3I}�       �	��)fc�A�*

loss'��<��       �	Ҫ�)fc�A�*

loss��p<��(       �	�C�)fc�A�*

loss�:=�>6�       �	�=�)fc�A�*

loss��<>�8       �	ٌ)fc�A�*

loss��>� �a       �	��)fc�A�*

loss���;�pQ       �	�(�)fc�A�*

loss)�e=�LL�       �	ˎ)fc�A�*

loss��<#�7(       �	^��)fc�A�*

loss�ۖ=��i:       �	p$�)fc�A�*

lossV�#=?JRO       �	�А)fc�A�*

loss�"=�A       �	�v�)fc�A�*

loss)-Q=��ߍ       �	Q�)fc�A�*

loss��X<�vn       �	���)fc�A�*

loss;�];�-L       �	|b�)fc�A�*

lossX�<�f       �	�	�)fc�A�*

loss-J�<E6�       �	d��)fc�A�*

loss���<��!�       �	?W�)fc�A�*

loss-:�<sh?�       �	���)fc�A�*

loss-��=�$�)       �	���)fc�A�*

loss�I=)`�       �	�%�)fc�A�*

loss��=�	�       �	�ȗ)fc�A�*

lossa��=�czz       �	)^�)fc�A�*

losssYv=Jŷ       �	���)fc�A�*

loss��<����       �	i��)fc�A�*

loss���=�F,       �	�)�)fc�A�*

loss��;O��       �	S˚)fc�A�*

loss�	�<�w_r       �	
��)fc�A�*

loss<�=-�       �	)��)fc�A�*

loss��<�Y!       �	0�)fc�A�*

loss*#�;��v       �	ם)fc�A�*

loss�MA=��A�       �	�|�)fc�A�*

lossn�8<��5T       �	��)fc�A�*

lossn�V;!$j       �	ܻ�)fc�A�*

loss��<5��       �	XW�)fc�A�*

loss��=[���       �	�)fc�A�*

loss�[<��j       �	��)fc�A�*

loss��>���Z       �	�#�)fc�A�*

loss�3 >q���       �	ϼ�)fc�A�*

loss�
�;Nܰz       �	�W�)fc�A�*

loss���;G�7       �	C��)fc�A�*

loss���:Ko�|       �	���)fc�A�*

loss�,<0�?�       �	�<�)fc�A�*

loss��=�(�       �	Nӥ)fc�A�*

loss��2=���J       �	#h�)fc�A�*

loss,�0<R��n       �	�)fc�A�*

lossq�;� 1       �	'��)fc�A�*

loss�o(=���       �	e9�)fc�A�*

loss$�%<�}�7       �	_Ψ)fc�A�*

losse<�S�H       �	l�)fc�A�*

loss<M�=(�O�       �	��)fc�A�*

loss��=ѷ�B       �	е�)fc�A�*

loss��<��\       �	�L�)fc�A�*

lossR�=:��]       �	ݫ)fc�A�*

loss�~=S�B�       �	w�)fc�A�*

lossd�B<8��C       �	r��)fc�A�*

loss��<`2e5       �	{1�)fc�A�*

loss8�<�/�       �	�̮)fc�A�*

loss��p<T��       �	uw�)fc�A�*

loss�W=Yp(&       �	�#�)fc�A�*

loss�,=��       �	ư)fc�A�*

loss#�d;$���       �	<k�)fc�A�*

loss�Y<Bc5e       �	��)fc�A�*

loss��s=�ZRM       �	"��)fc�A�*

loss��3=���       �	�N�)fc�A�*

lossַ=�]L�       �	��)fc�A�*

lossfg�<,�/�       �	���)fc�A�*

loss���:��v
       �	�$�)fc�A�*

lossF?<��5M       �	�õ)fc�A�*

loss.4;O�Ԉ       �	h^�)fc�A�*

losse�<�&��       �	��)fc�A�*

lossx�&=$�ֺ       �	j��)fc�A�*

loss��V<�4ȟ       �	"6�)fc�A�*

lossz�d=�cA|       �	cи)fc�A�*

loss�Be=���       �	~�)fc�A�*

loss��=�;d       �	N%�)fc�A�*

lossH44<Q'       �	f��)fc�A�*

lossMˁ<�(K       �	�d�)fc�A�*

lossI�}=��$       �	\�)fc�A�*

loss�{<[��       �	Ϡ�)fc�A�*

loss���=��'       �	�>�)fc�A�*

loss�%�;��}%       �	�ٽ)fc�A�*

loss+I=��d�       �	\��)fc�A�*

losssT<�4       �	�.�)fc�A�*

loss�x0=�K�       �	1�)fc�A�*

loss<��	�       �	w��)fc�A�*

loss�>�<�       �	9��)fc�A�*

loss���;PF={       �	5'�)fc�A�*

loss;`�;�Wg       �	'��)fc�A�*

loss��=���       �	OW�)fc�A�*

loss�9;�+�       �	���)fc�A�*

loss�$<|�[       �	p��)fc�A�*

lossv`=s� �       �	�^�)fc�A�*

loss�K�<��[       �	��)fc�A�*

loss� i=So1�       �	��)fc�A�*

loss�;e��e       �	���)fc�A�*

loss��P</5�#       �	L5�)fc�A�*

loss3�:�1T�       �	_a�)fc�A�*

loss���<�?       �	j�)fc�A�*

lossq~K;���       �	=��)fc�A�*

loss!�=�>�W       �	+h�)fc�A�*

loss*.�<��,�       �	��)fc�A�*

loss-u<�� �       �	���)fc�A�*

loss��=XI:%       �	���)fc�A�*

loss���=�?	�       �	���)fc�A�*

lossF'�<�f�9       �	�5�)fc�A�*

loss��<�!��       �	���)fc�A�*

loss���=j��       �	���)fc�A�*

losscu�;��1�       �	d"�)fc�A�*

loss���<At��       �	���)fc�A�*

loss��=��I�       �	�j�)fc�A�*

losss��;���       �	V�)fc�A�*

lossď�=�6�       �	���)fc�A�*

lossj�<φ��       �	"T�)fc�A�*

lossl�<�Yc�       �	V��)fc�A�*

loss�I�=�S       �	��)fc�A�*

loss�<a��g       �	��)fc�A�*

loss�=�T/       �	Q��)fc�A�*

loss��1=΅�       �	;S�)fc�A�*

loss��<��X       �	���)fc�A�*

lossƸ�<v�dT       �	V��)fc�A�*

lossW/�<��J�       �	;�)fc�A�*

loss|�;[�5�       �	l��)fc�A�*

lossBٓ;!"��       �	eT�)fc�A�*

loss��<��#       �	h�)fc�A�*

loss4̎<H[Q       �	 ��)fc�A�*

loss�̬<,8e        �	���)fc�A�*

loss�L>��J       �	.8�)fc�A�*

loss���<7/�(       �	I��)fc�A�*

loss�Ї;�t{       �	�r�)fc�A�*

lossx��<���       �	��)fc�A�*

lossV&T<�Lɹ       �	���)fc�A�*

loss�}�;�	�       �	4H�)fc�A�*

loss���;7[|B       �	�B�)fc�A�*

loss��<����       �	���)fc�A�*

loss��=@�8�       �	�y�)fc�A�*

loss1�=�XJ�       �	��)fc�A�*

loss� >�{�       �	:��)fc�A�*

loss�y=Q.��       �	�P�)fc�A�*

lossd:�<���^       �	��)fc�A�*

loss�^S<"�R       �	ޑ�)fc�A�*

loss�]=?�"       �	�.�)fc�A�*

loss�D=/g2�       �	��)fc�A�*

loss_�<J��       �	�k�)fc�A�*

loss�y=����       �	}�)fc�A�*

loss	H-;��+w       �	w��)fc�A�*

lossM�=
��^       �	�;�)fc�A�*

loss��;�ٳ       �	4��)fc�A�*

lossW�Z<4�.�       �	�r�)fc�A�*

loss���<J�Y       �	��)fc�A�*

loss�=��U�       �	���)fc�A�*

loss8D�<�2��       �	_�)fc�A�*

lossM�"<{d�       �	kF�)fc�A�*

loss�!�<2��       �	�z�)fc�A�*

loss�p�<>*br       �	1%�)fc�A�*

loss��Z<��C]       �	 �)fc�A�*

lossOȨ;�؛�       �		��)fc�A�*

loss�+=��%X       �	�_�)fc�A�*

loss
B�;/�#�       �		��)fc�A�*

loss�{=��R%       �	ٗ�)fc�A�*

loss3��<c�'       �	�8�)fc�A�*

loss�)�=�r�N       �	9��)fc�A�*

loss�)�<��ڮ       �	�w�)fc�A�*

loss� H<平�       �	��)fc�A�*

loss��k:oq��       �	���)fc�A�*

loss�0=�a�       �	�i�)fc�A�*

loss�b<{�       �	���)fc�A�*

loss�F0<��       �	і�)fc�A�*

loss�7<t6#�       �	6\�)fc�A�*

loss�'<�a<       �	{��)fc�A�*

loss^k=kE�A       �	��)fc�A�*

loss�M�<y�.�       �	w-�)fc�A�*

loss	�[=gG        �	���)fc�A�*

loss�P�<����       �	Z�)fc�A�*

loss�=����       �	��)fc�A�*

loss2��=OL	       �	;��)fc�A�*

loss&�;�xT       �	�0�)fc�A�*

losstY�<;f�}       �	i;�)fc�A�*

loss.E;g���       �	(��)fc�A�*

loss�h�<靟r       �	,��)fc�A�*

lossF�B=�
�       �	� *fc�A�*

loss�O�<hc�;       �	� *fc�A�*

loss�K3=�t�       �	u[*fc�A�*

loss��<�Z��       �	b�*fc�A�*

loss��=�&       �	A�*fc�A�*

loss���<��$       �	ڪ*fc�A�*

losseA�;l�4       �	U*fc�A�*

loss��w<?�;       �	�1*fc�A�*

loss�M�;�'��       �	f�*fc�A�*

loss���<��V1       �	��*fc�A�*

lossr,�<w���       �	�k*fc�A�*

lossj��=�6=�       �	�h*fc�A�*

loss�;=�Z       �	6	*fc�A�*

loss�ٲ<�n�       �	ˠ	*fc�A�*

lossj�:=���~       �	tB
*fc�A�*

loss6�<��       �	��
*fc�A�*

loss��*<kV'Z       �	�~*fc�A�*

loss"c�;'⛻       �	~*fc�A�*

loss�<}yL�       �	��*fc�A�*

losso��=zM��       �	�\*fc�A�*

lossIV�=,�C�       �	�*fc�A�*

loss9J=B�3       �	ؼ*fc�A�*

lossl̽=�g�	       �	�Y*fc�A�*

loss^v=����       �	U�*fc�A�*

loss�8<̠k�       �	�*fc�A�*

loss٢;�&K�       �	�A*fc�A�*

loss�=�<�x�       �	��*fc�A�*

loss=���h       �	�t*fc�A�*

loss��;	5�       �	[*fc�A�*

loss�W=<�0iU       �	u�*fc�A�*

loss��=��IN       �	�@*fc�A�*

lossN5�=�L�       �	
�*fc�A�*

loss�iJ<I��       �	k*fc�A�*

lossM��<�Nw       �	� *fc�A�*

loss߳"<�}��       �	��*fc�A�*

loss��;�	H�       �	�7*fc�A�*

loss�q<d\Co       �	��*fc�A�*

loss��<%�Q�       �	k�*fc�A�*

loss)��<{?�       �	�`*fc�A�*

lossM4)<>^�2       �	K*fc�A�*

loss�V�<4<�       �	��*fc�A�*

lossV�=)`��       �	P4*fc�A�*

lossy>�mw       �	d�*fc�A�*

loss���<^�P       �	�k*fc�A�*

loss�ߞ<�5[�       �	m�*fc�A�*

lossQ�=��W       �	��*fc�A�*

loss <14�       �	?9*fc�A�*

lossM�<��>e       �	�*fc�A�*

loss=�<;��       �	?q *fc�A�*

loss�~�<*z�       �	�!*fc�A�*

loss ��<F�H^       �	�"*fc�A�*

lossw�q<f�p       �	��"*fc�A�*

lossap>J�       �	8J#*fc�A�*

loss�ߟ<�,U�       �	i�#*fc�A�*

loss�D�=�,=M       �	ׄ$*fc�A�*

loss��=?�       �	5E%*fc�A�*

loss27�<)|�       �	��%*fc�A�*

loss��;.a�$       �	��&*fc�A�*

loss��W:@��%       �	5%'*fc�A�*

loss)�;Ǜ!C       �	�'*fc�A�*

lossU�=�F�       �	qY(*fc�A�*

loss��<i���       �	��(*fc�A�*

loss/�G=�Y�~       �	z�)*fc�A�*

loss�k�<�E�E       �	'**fc�A�*

loss��=S�%       �	8�**fc�A�*

loss���<bCn�       �	:X+*fc�A�*

loss��=[���       �	;�+*fc�A�*

loss�|=���       �	�=-*fc�A�*

loss#2=�-�       �	F{.*fc�A�*

loss
�<��Fs       �	�,/*fc�A�*

loss��l=�m�k       �	Cr0*fc�A�*

losslp�=��R�       �	��1*fc�A�*

losss�<w�l
       �	�F2*fc�A�*

loss	<=!���       �	Nz3*fc�A�*

loss1L�;��       �	 W4*fc�A�*

loss�d=��       �	N~5*fc�A�*

loss��=�-       �	��6*fc�A�*

loss��=��       �	 �7*fc�A�*

loss+><��J       �	��8*fc�A�*

lossn��=���       �	�I9*fc�A�*

loss��<��.�       �	�9*fc�A�*

lossԓ�;)�#       �	�%;*fc�A�*

loss�́=�a?       �	5�;*fc�A�*

loss���<ȯ�/       �	[�<*fc�A�*

lossm�<T`�       �	�4=*fc�A�*

loss ��<<�ɷ       �	�q>*fc�A�*

loss��;�_�       �	�?*fc�A�*

loss���<�A��       �	�P@*fc�A�*

loss1�=�fo       �	�A*fc�A�*

loss��,<kOiL       �	]�A*fc�A�*

lossaq=T?<       �	�JB*fc�A�*

loss6��<c"�K       �	��B*fc�A�*

loss&ƿ<H�ǔ       �	�C*fc�A�*

lossF�c;�wȘ       �	[(D*fc�A�*

loss:|'=�>ċ       �	��D*fc�A�*

loss
==!>       �	�bE*fc�A�*

losstF1<G+Þ       �	��E*fc�A�*

lossT��;|�       �	�F*fc�A�*

loss��<X(�       �	+3G*fc�A�*

loss=W=���       �	]�G*fc�A�*

loss�e�<?7�       �	[xH*fc�A�*

loss���=�k8�       �	��I*fc�A�*

loss�>��Rv       �	H�J*fc�A�*

loss_��<R�&�       �	y!K*fc�A�*

loss�i�<X+uv       �	��K*fc�A�*

loss�3<�[�       �	SXL*fc�A�*

loss;�f=����       �	�L*fc�A�*

lossz1:=��?�       �	��M*fc�A�*

lossv��<X��]       �	P�N*fc�A�*

loss�Jg=u��       �	t'O*fc�A�*

lossZ�<����       �	2�O*fc�A�*

loss��=�u6�       �	iP*fc�A�*

lossV9<���2       �	 �P*fc�A�*

lossWt�;)ؔM       �	��Q*fc�A�*

loss8��<FDY       �	�3R*fc�A�*

loss��<�j��       �	1�R*fc�A�*

loss�V�<Cw�       �	�rS*fc�A�*

loss �Q<����       �	PT*fc�A�*

loss:(<��$&       �	�T*fc�A�*

loss*�?=�2W2       �	�nU*fc�A�*

loss!"�<�ɗ�       �	mV*fc�A�*

lossb�<O��c       �	��V*fc�A�*

loss�{
=��       �	?rW*fc�A�*

loss��V=�}Y       �	�$X*fc�A�*

loss���<aZ��       �	&�X*fc�A�*

loss&�=N��       �	�pY*fc�A�*

loss#9�<4@/'       �	mZ*fc�A�*

loss�M<���q       �	!�Z*fc�A�*

loss�=�Q�V       �	�h[*fc�A�*

loss�r�=9:/�       �	�\*fc�A�*

lossE	a=�G�&       �	X�\*fc�A�*

loss��n=1u-�       �	�N]*fc�A�*

loss@�0=̇��       �	>�]*fc�A�*

loss��G>��       �	H�^*fc�A�*

loss��<�H:�       �	F"_*fc�A�*

loss� 0<�B�)       �	׽_*fc�A�*

loss��:-R�7       �	KW`*fc�A�*

loss��;D X5       �	J�`*fc�A�*

loss홂=~�\e       �	f�a*fc�A�*

lossc|�<`���       �	"b*fc�A�*

loss�t=/J�       �	��b*fc�A�*

lossr˩<*��Y       �	�cc*fc�A�*

lossN��<��#1       �	d*fc�A�*

loss@��;,g�       �	��d*fc�A�*

loss��o;ҝt�       �	�<e*fc�A�*

loss̴=����       �	��e*fc�A�*

lossZ�5;��I       �	vpf*fc�A�*

loss��=��v       �	_g*fc�A�*

loss�,�=�0��       �	Ԟg*fc�A�*

loss��V=B{]�       �	�5h*fc�A�*

loss�G�;��(       �	f�h*fc�A�*

loss)��<bA(/       �	/�i*fc�A�*

lossCY=;��       �	�;j*fc�A�*

loss��Z=��Q       �	C�k*fc�A�*

lossߦV;�H%       �	�Pl*fc�A�*

loss�q3<����       �	Dm*fc�A�*

loss|_�=H2�c       �	�Yn*fc�A�*

loss���=��       �	��n*fc�A�*

loss��5=�ۭu       �	cp*fc�A�*

lossqO<}
J       �	�p*fc�A�*

loss��I=A��       �	��q*fc�A�*

loss#h=�O�w       �	��r*fc�A�*

loss/<�Љ5       �	�s*fc�A�*

loss��m<-e�       �	,�t*fc�A�*

loss��s=�f��       �	9Gu*fc�A�*

lossT�;�M�       �	��u*fc�A�*

losswʴ<��Y       �	��v*fc�A�*

loss�>�=�ž�       �	�+w*fc�A�*

lossl׏<_�4       �	8�w*fc�A�*

loss�*=c�6L       �	�{x*fc�A�*

loss��_<���x       �	y*fc�A�*

loss��S<���       �	ܷy*fc�A�*

loss|�}=#��       �	�Uz*fc�A�*

loss?�G=H7       �	_
{*fc�A�*

loss��}<��=X       �	t�{*fc�A�*

lossm �<B���       �	"U|*fc�A�*

loss1��<��~�       �	��|*fc�A�*

losse,:=�G�       �	�}*fc�A�*

loss:��;/�       �	�2~*fc�A�*

loss(��=3s��       �	��~*fc�A�*

loss@�=P5hm       �	�*fc�A�*

loss�:=#��$       �	u=�*fc�A�*

loss�5=��M^       �	��*fc�A�*

loss��Y=��_       �	���*fc�A�*

lossA�<���       �	�!�*fc�A�*

losslb�<�<��       �	�*fc�A�*

lossQ��=�J�       �	�R�*fc�A�*

lossz��='�%]       �	��*fc�A�*

loss3`�<mʥ       �	ƅ*fc�A�*

loss��;=j��	       �	'�*fc�A�*

loss��&=�v̀       �	���*fc�A�*

loss�Pl=���       �	�]�*fc�A�*

loss���<�g�       �	4�*fc�A�*

loss]�<��"�       �	���*fc�A�*

loss���<�=��       �	T�*fc�A�*

loss�1�<ϓ�3       �	6��*fc�A�*

loss6j><����       �	���*fc�A�*

loss�n=Y��       �	�O�*fc�A�*

loss�,.<c��u       �	��*fc�A�*

loss*-=���X       �	Oy�*fc�A�*

loss{��</�v�       �	 W�*fc�A�*

loss��{;i�û       �	��*fc�A�*

loss�<=��V       �	Y��*fc�A�*

loss'��<�
/       �	�#�*fc�A�*

loss׌&<�H�-       �	���*fc�A�*

loss��s<�S�-       �	\�*fc�A�*

lossF�<,�)       �	y�*fc�A�*

loss���<��<�       �	���*fc�A�*

loss̿4=wJ�       �	�C�*fc�A�*

loss@��;�M�       �	��*fc�A�*

loss(^�=5$�       �	���*fc�A�*

loss�c�=�]       �	�#�*fc�A�*

loss��i</`�       �	;ĕ*fc�A�*

loss�b�<��A�       �	e�*fc�A�*

loss��,=�U%�       �	E�*fc�A�*

lossv�"=O.ل       �	K��*fc�A�*

lossf.�=4�D       �	"O�*fc�A�*

loss�=`��[       �	��*fc�A�*

loss���<�d�       �	���*fc�A�*

loss�Ad=��       �	�!�*fc�A�*

loss�C1;�\Њ       �	�Ě*fc�A�*

lossJ�Q<�}��       �	�[�*fc�A�*

loss�z�<�{B       �	(�*fc�A�*

loss<�?=�)�       �	O��*fc�A�*

loss���<�d;       �	�)�*fc�A�*

lossl6=j��       �	{��*fc�A�*

lossᩀ<l�       �	GU�*fc�A�*

lossu�=�7:*       �	M��*fc�A�*

lossqD�=[�OL       �	���*fc�A�*

loss��=�Wf�       �	_$�*fc�A�*

loss�8�<�>�       �	Z��*fc�A�*

lossʐ/=��t       �	�R�*fc�A�*

lossh�V<vt�_       �	���*fc�A�*

loss�j<	Z�       �	���*fc�A�*

loss�H�<���5       �	~9�*fc�A�*

loss���<U	V       �	�ң*fc�A�*

loss)�)=9��       �	�t�*fc�A�*

loss�ؑ<�']       �	M�*fc�A�*

loss$�<sƻ�       �	;��*fc�A�*

lossE/=ח�|       �	�Q�*fc�A�*

loss3��<85,�       �	��*fc�A�*

losss�	<�Mt(       �	O��*fc�A�*

loss�8<����       �	�0�*fc�A�*

lossC|�<���L       �	�ͨ*fc�A�*

loss`p�=���       �	i�*fc�A�*

lossGF�=p��       �	��*fc�A�*

loss�'=_�J�       �	]��*fc�A�*

loss�X�=�Q�       �	�J�*fc�A�*

loss�`I<z@J       �	`�*fc�A�*

loss���<�q/�       �	Z��*fc�A�*

lossD��=t-�       �	�R�*fc�A�*

loss�yD<n���       �	��*fc�A�*

lossô�=�=9       �	�ܮ*fc�A�*

loss�]<G���       �	,��*fc�A�*

loss�,�=���       �	�c�*fc�A�*

loss��4="G�"       �	���*fc�A�*

loss�|�<�/�       �	�8�*fc�A�*

loss�=`��d       �	pѲ*fc�A�*

lossߐ�<��D�       �	�ȳ*fc�A�*

loss��C=F��x       �	�x�*fc�A�*

loss��P<��X6       �	<�*fc�A�*

lossMU3<�H7       �	|��*fc�A�*

loss��<f���       �	|�*fc�A�*

lossJ��<�k�8       �	�!�*fc�A�*

loss���<�E�       �	���*fc�A�*

loss@q�;�Z�       �	�U�*fc�A�*

loss8G�;S�/�       �		6�*fc�A�*

loss�'^=lXD       �	�q�*fc�A�*

loss��Y<A�3       �	L�*fc�A�*

loss��E<���       �	���*fc�A�*

loss��;��]3       �	�]�*fc�A�*

lossQ�; ݸ       �	7��*fc�A�*

loss�P�<M
�       �	��*fc�A�*

lossS�:�e�       �	�3�*fc�A�*

lossr7�=c�       �	�Ҿ*fc�A�*

lossO%�;�f�       �	.p�*fc�A�*

lossc&0=#��       �	'�*fc�A�*

loss� �=76��       �	F��*fc�A�*

loss��=В�       �	Q�*fc�A�*

loss�lf<��'       �	��*fc�A�*

loss���;ާ�w       �	ۈ�*fc�A�*

lossݷ�<�#`�       �	p%�*fc�A�*

loss�<ڐ�       �	��*fc�A�*

loss$lr=r�'       �	���*fc�A�*

loss,�=.�0        �	TT�*fc�A�*

loss�h'=^\�       �	���*fc�A�*

loss�c<���       �	��*fc�A�*

loss,~,<���z       �	�*�*fc�A�*

loss8l4=�?�       �	@��*fc�A�*

loss��1<��S�       �	�w�*fc�A�*

loss�G<�gw       �	z�*fc�A�*

lossOBw<���e       �	���*fc�A�*

loss���<���       �	ޓ�*fc�A�*

loss��=H8�       �	R+�*fc�A�*

lossF�=]dD�       �	@��*fc�A�*

loss��~<Q�4       �	KZ�*fc�A�*

loss���<�8��       �	_��*fc�A�*

loss�ơ<:9�?       �	���*fc�A�*

loss-<�eX�       �	�(�*fc�A�*

loss�?�<�q!       �	���*fc�A�*

loss���<�3��       �	�V�*fc�A�*

loss�/O<z'W7       �	���*fc�A�*

lossD�<a2��       �	���*fc�A�*

lossR�e;UC�(       �	4�*fc�A�*

loss�<*�       �	\��*fc�A�*

lossvc<[0�       �	�@�*fc�A�*

loss��N<��a       �	��*fc�A�*

loss-�Z;���       �	�j�*fc�A�*

loss��=
|�l       �	��*fc�A�*

losso/�<��       �	;��*fc�A�*

loss(�<�)�       �	�@�*fc�A�*

lossF�;vkT�       �	���*fc�A�*

loss7#�<��y       �	���*fc�A�*

loss�W<�į�       �	-`�*fc�A�*

loss��;(�       �	s��*fc�A�*

loss/m=��3�       �	P��*fc�A�*

loss�M�<!ޙ�       �	p�*fc�A�*

loss�$p=�"�b       �	]�*fc�A�*

loss���<K��&       �	ƾ�*fc�A�*

lossb<��յ       �	6[�*fc�A�*

loss;z	=N!O       �	��*fc�A�*

loss6g�:��7t       �	���*fc�A�*

loss��;��Rv       �	�J�*fc�A�*

lossc�P<��z       �	���*fc�A�*

loss�x;*6.       �	^��*fc�A�*

loss��2<x���       �	 �*fc�A�*

lossʵ7:e1́       �	���*fc�A�*

loss(J�;�_��       �	2V�*fc�A�*

lossf,v=bf�       �	��*fc�A�*

lossr�!<�.�       �	Q��*fc�A�*

loss���<@��       �	e8�*fc�A�*

lossŦ�<x���       �	)��*fc�A�*

lossF��<����       �	�o�*fc�A�*

lossX��<�t�y       �	x�*fc�A�*

loss�A;+��       �	��*fc�A�*

lossxz�<Ǟ3�       �	͔�*fc�A�*

loss](�:%���       �	EK�*fc�A�*

loss�f:��       �	`��*fc�A�*

lossI��:��       �	�~�*fc�A�*

loss葬;���       �	~�*fc�A�*

lossD��:sg�E       �	���*fc�A�*

loss�h<>�       �	wL�*fc�A�*

loss�w�:����       �	��*fc�A�*

loss���<�'�       �	H��*fc�A�*

loss�H=��u       �	�+�*fc�A�*

loss�Ȇ8?NyB       �	��*fc�A�*

loss���:m
�       �	�e�*fc�A�*

loss�.<K���       �	z��*fc�A�*

loss�D�<|GD[       �	n��*fc�A�*

loss�w<�@��       �	O;�*fc�A�*

loss)��8���       �	B��*fc�A�*

lossj�<C���       �	o��*fc�A�*

lossNn�=V�z�       �	�O�*fc�A�*

loss�z"=���~       �	8��*fc�A�*

loss��<#b��       �	D��*fc�A�*

lossL�p=��n�       �	>�*fc�A�*

loss��o<I/��       �	9�*fc�A�*

loss3�<k��J       �	��*fc�A�*

loss{]z<B4�       �	`<�*fc�A�*

loss?��=}�T       �	���*fc�A�*

loss<!=���       �	���*fc�A�*

loss&��:���C       �	���*fc�A�*

loss� 0=+��       �	a��*fc�A�*

loss�ˆ<�Ӕ�       �	�q�*fc�A�*

loss�K4=&%Q       �	��*fc�A�*

lossa��=T�.       �	���*fc�A�*

loss7�;��&       �	�7�*fc�A�*

lossNq�=�P�f       �	[��*fc�A�*

loss�=_�
�       �	i�*fc�A�*

loss�2i< ���       �	g�*fc�A�*

loss�}�<Zy�F       �	^��*fc�A�*

lossm��=to�P       �	WZ�*fc�A�*

loss�R�<�W=Y       �	+��*fc�A�*

loss&�D<c��       �	� +fc�A�*

loss/<�\WR       �	J{+fc�A�*

loss�*K<&K�       �	%!+fc�A�*

loss*��:}��       �	�+fc�A�*

loss�2�:��{G       �	B	+fc�A�*

loss=jU<?�T9       �	��+fc�A�*

loss��=i :V       �	�a+fc�A�*

lossݓ:=��       �	�+fc�A�*

loss8J=m�v�       �	�+fc�A�*

loss�=�1:       �	�A+fc�A�*

lossA��;@��]       �	r�+fc�A�*

loss��;	RJc       �	�9	+fc�A�*

loss �;@�$�       �	R�	+fc�A�*

lossZ:o;��:       �	$~
+fc�A�*

loss� �<ã.       �	�+fc�A�*

loss�-�;����       �	��+fc�A�*

loss��<m���       �	�V+fc�A�*

lossa��=l�N�       �	�+fc�A�*

loss��l=Fs       �	��+fc�A�*

lossx�(=�Æ       �	�4+fc�A�*

loss)��;L���       �	��+fc�A�*

loss,�=+�ϑ       �	x+fc�A�*

loss��<���       �	�+fc�A�*

lossi��<���y       �	��+fc�A�*

lossa2s<�k�w       �	�]+fc�A�*

loss�Hl==��g       �	+�+fc�A�*

loss��=��       �	�+fc�A�*

lossQ��;�h��       �	�3+fc�A�*

loss	g}=U�J       �	.+fc�A�*

lossN��<NK��       �	l�+fc�A�*

loss��9=}e�w       �	�+fc�A�*

loss=�<�R�,       �	�.-+fc�A�*

loss-<=!1/n       �	&�-+fc�A�*

loss�ɾ=�       �	�.+fc�A�*

loss�=L��M       �	��/+fc�A�*

loss��<A|ZH       �	�o0+fc�A�*

loss�u�<ǻ��       �	��1+fc�A�*

loss&�=ɑM�       �	1A2+fc�A�*

loss/#�<��       �	�&3+fc�A�*

loss��h=��        �	��3+fc�A�*

lossh�=�6]�       �	�4+fc�A�*

lossv��;�C�;       �	��5+fc�A�*

loss�=��ˍ       �	e�6+fc�A�*

lossn�)<�}	       �	�O7+fc�A�*

loss�!�<���       �	i�7+fc�A�*

loss�p�<����       �	C�8+fc�A�*

lossʖ�<1o5n       �	Ow9+fc�A�*

loss���:[wr       �	r:+fc�A�*

loss�$=�e�       �	ݶ:+fc�A�*

loss��<���       �	l];+fc�A�*

loss �h=��bo       �	�<+fc�A�*

loss
R<����       �	*�<+fc�A�*

loss`z�=)W�j       �	�K=+fc�A�*

lossJ�
;.A�{       �	��=+fc�A�*

loss���=Kw��       �	ލ>+fc�A�*

loss-T�;_̰�       �	j3?+fc�A�*

loss�7<��@       �	��?+fc�A�*

loss%/d<M�w�       �	ap@+fc�A�*

loss�)W<�{�b       �	(
A+fc�A�*

loss{=G|�       �	z�A+fc�A�*

loss��<73�r       �	p@B+fc�A�*

loss֎�<�ߢE       �	g�B+fc�A�*

lossX�~<�(       �	�oC+fc�A�*

lossx%<�.�V       �	(D+fc�A�*

loss8��<[��       �	D�D+fc�A�*

loss�b<h�+H       �	�DE+fc�A�*

loss���<}�^�       �	� F+fc�A�*

lossD�;����       �	��F+fc�A�*

loss�<�<i�h)       �	G9G+fc�A�*

loss.s�=,��8       �	g�G+fc�A�*

loss{u�<�[˹       �	�gH+fc�A�*

loss�.=��Y�       �	� I+fc�A�*

loss�ٞ<�(-       �	A�I+fc�A�*

lossf{�<����       �	�0J+fc�A�*

loss�}�<���       �	��J+fc�A�*

loss��<��c�       �	bjK+fc�A�*

loss��<���=       �	�L+fc�A�*

loss+5=�       �	��L+fc�A�*

lossmJ�<���       �	�:M+fc�A�*

loss�,�<K�^a       �	^�M+fc�A�*

loss*�[<T�D       �	CsN+fc�A�*

loss&��;��T       �	_O+fc�A�*

loss��=�Q��       �	צO+fc�A�*

loss6ő<3|�       �	aPP+fc�A�*

loss�D=��V       �	��P+fc�A�*

loss�y
<��HE       �	&�Q+fc�A�*

loss�B�:U�}�       �	�@R+fc�A�*

loss��u<|DF!       �	��R+fc�A�*

loss���;�>��       �	�S+fc�A�*

lossJ�s=-��H       �	T+fc�A�*

loss���=��r�       �	�T+fc�A�*

loss6�B=ЖL�       �	�WU+fc�A�*

lossr�;�p�g       �	��U+fc�A�*

loss���:J�O�       �	{�V+fc�A�*

lossa�+=TP6       �	s-W+fc�A�*

lossh<6<� '�       �	O�W+fc�A�*

lossxô=E,XI       �	TpX+fc�A�*

loss��;��J�       �	gY+fc�A�*

loss Fq<��ȳ       �	:�Y+fc�A�*

lossM}=���       �	HZ+fc�A�*

loss`=}N       �	�[+fc�A�*

losssT�;�r��       �	r\+fc�A�*

loss���<f��]       �	�!]+fc�A�*

loss'd=+[�       �	-�]+fc�A�*

loss��I<h��       �	�x^+fc�A�*

loss	YY=)4�       �	�\`+fc�A�*

lossHD�=�G��       �	n�`+fc�A�*

lossE(w;`Y�       �	9�a+fc�A�*

loss��<0���       �	�@b+fc�A�*

loss�+<��2b       �	8�b+fc�A�*

loss�L�<�j/       �	Àc+fc�A�*

loss��=�M�       �	�d+fc�A�*

loss���<;��]       �	A�d+fc�A�*

lossr*�<3�@^       �	'Pe+fc�A�*

loss�6=�K��       �	��e+fc�A�*

loss��'=h��       �	��f+fc�A�*

loss�p?<#"�       �	!g+fc�A�*

loss�A�<G�9&       �	#�g+fc�A�*

lossK��<�3�       �	S^h+fc�A�*

loss�F =���       �	��h+fc�A�*

loss=��<�%�       �	ői+fc�A�*

lossC�U=���       �	;7j+fc�A�*

loss� =Rb       �	��j+fc�A�*

lossf��<[ZD�       �	quk+fc�A�*

loss��A<�RF�       �	Ml+fc�A�*

loss�E=��o       �	�l+fc�A�*

loss��;�/�       �	�dm+fc�A�*

loss�^H=����       �	�n+fc�A�*

lossȨQ=υ�       �	��n+fc�A�*

losswgT=6��(       �	D�o+fc�A�*

loss�q<����       �	obp+fc�A�*

loss�<����       �	�p+fc�A�*

loss|�:'�)I       �	'�q+fc�A�*

lossXE<��w3       �	<r+fc�A�*

loss�RZ<!��       �	<s+fc�A�*

loss-=�;rF�       �	��s+fc�A�*

loss�b<�lYr       �	�[t+fc�A�*

loss�q�<�ct       �	<�t+fc�A�*

loss�X�<�+m#       �	�u+fc�A�*

loss\�=�v��       �	 7v+fc�A�*

lossR�<H��       �	O�v+fc�A�*

loss���<"n՚       �	�gw+fc�A�*

lossǍ=�$�       �	x+fc�A�*

lossx^Z<�ɰY       �	��x+fc�A�*

loss��8;_Ϯ       �	�;y+fc�A�*

lossң�<V��       �	Mz+fc�A�*

lossO�i=sq��       �	��z+fc�A�*

loss���<�/�       �	p}{+fc�A�*

lossv=ٿ�       �	�|+fc�A�*

loss(�<|Ֆb       �	�|+fc�A�*

loss��V=飧�       �	S^}+fc�A�*

loss!�)<@��3       �	h~+fc�A�*

losseY�=��#`       �	m�~+fc�A�*

lossz�;��߽       �	�J+fc�A�*

loss���=W�#�       �	��+fc�A�*

loss��;/$�       �	/��+fc�A�*

loss�
u<��c       �	%"�+fc�A�*

lossc3�=���       �	M��+fc�A�*

loss}iN;�WV	       �	�Z�+fc�A�*

loss�'[=��Q3       �	N�+fc�A�*

loss�b{=WR��       �	D��+fc�A�*

loss���<�՘       �	�+fc�A�*

loss}�<�֖       �	���+fc�A�*

loss���;�<��       �	_�+fc�A�*

lossoI=f�O�       �	�+fc�A�*

loss�=��N       �	��+fc�A�*

lossl$?=r�\(       �	�B�+fc�A�*

loss�M=�|b%       �	��+fc�A�*

loss"�=����       �	���+fc�A�*

lossE3�:��       �	�,�+fc�A�*

loss��<�t�l       �	ԉ+fc�A�*

lossJW=�hRk       �	}�+fc�A�*

loss�c�=!�ڴ       �	�G�+fc�A�*

loss��1<�Q�       �	��+fc�A�*

lossr�<<�       �	K��+fc�A�*

loss��=}���       �	Z+�+fc�A�*

loss)� <}h�       �	�ō+fc�A�*

loss�~�<3���       �	�j�+fc�A�*

loss8�N<@*�       �	h�+fc�A�*

loss�J�;���       �	雏+fc�A�*

loss]}</���       �	O<�+fc�A�*

loss&�9<3���       �	Aؐ+fc�A�*

loss���<���a       �	�x�+fc�A�*

loss��=\�P�       �	��+fc�A�*

loss�M=����       �	]��+fc�A�*

loss!J	=_x       �	�H�+fc�A�*

loss4;=�7l       �	��+fc�A�*

loss���;g8��       �	_Ӕ+fc�A�*

loss&X�<v�F_       �	(��+fc�A�*

loss�^=�GBa       �	!�+fc�A�*

loss��;��U       �	���+fc�A�*

loss���<?pO/       �	�^�+fc�A�*

lossz2�;]K{�       �	���+fc�A�*

lossqP�<qJ5       �	"��+fc�A�*

loss1ھ;�$��       �	|E�+fc�A�*

loss�5<<� ��       �	��+fc�A�*

loss8F�<���U       �		��+fc�A�*

loss��n<aĒ�       �	@�+fc�A�*

loss٠=o�M       �	ܛ+fc�A�*

loss���<���       �	F|�+fc�A�*

loss�m<��?�       �	a�+fc�A�*

loss|8�=��Y�       �	���+fc�A�*

loss}�6<�9       �	�T�+fc�A�*

loss)k<��eg       �	�+fc�A�*

loss��E=W�&F       �	@��+fc�A�*

loss�(�:<:?p       �	�+fc�A�*

lossc�<��,2       �	|��+fc�A�*

loss��<E�N�       �	�U�+fc�A�*

losso�<�	       �	��+fc�A�*

loss��<r�b       �	���+fc�A�*

loss�F�;~��|       �	�1�+fc�A�*

loss�a:Ϊ�       �	Σ+fc�A�*

loss$��=���       �	/l�+fc�A�*

loss���<u��d       �	��+fc�A�*

lossN�/;5sY       �	U��+fc�A�*

loss�>�;W�0       �	�8�+fc�A�*

loss`#=;�h       �	�Ѧ+fc�A�*

loss8�=C7�       �	�n�+fc�A�*

losss/�<��.�       �	:�+fc�A�*

loss�!�<�+�5       �	
��+fc�A�*

loss61={�O        �	!>�+fc�A�*

loss(��<b�f�       �	R֩+fc�A�*

loss�]�<�w\       �	�t�+fc�A�*

loss%�<��X�       �	�+fc�A�*

loss���<��[�       �	@��+fc�A�*

loss|�;ֿ�       �	G�+fc�A�*

loss勈<��o       �	�߬+fc�A�*

loss�T�<:-'�       �	�v�+fc�A�*

loss}�9=Jk�`       �	#��+fc�A�*

loss�?�<5��       �	¯+fc�A�*

loss�]�;vA��       �	��+fc�A�*

loss��==�Ήb       �	�<�+fc�A�*

loss�_-=1TI       �	g�+fc�A�*

loss)��<���c       �	�7�+fc�A�*

loss�^<t��       �	��+fc�A�*

loss[�/<��YI       �	�޴+fc�A�*

loss,�#<�\�       �	��+fc�A�*

loss�t[=���       �	�O�+fc�A�*

loss"=Z�~       �	��+fc�A�*

loss&��<Q�cS       �	%��+fc�A�*

loss���;�a(�       �	�*�+fc�A�*

loss��<���[       �	�ǹ+fc�A�*

loss���=X���       �	i�+fc�A�*

loss��<xX��       �	5�+fc�A�*

loss�W==��Z       �	��+fc�A�*

loss�^*=�d4n       �	�L�+fc�A�*

loss��R=��E       �	���+fc�A�*

loss`)q=�Z��       �	���+fc�A�*

loss���=5���       �	�L�+fc�A�*

loss�V=D�
       �	�+fc�A�*

loss���<��7*       �	u��+fc�A�*

lossW	r<[g�       �	�P�+fc�A�*

loss��s<<��       �	.��+fc�A�*

lossaE�<Ph       �	o��+fc�A�*

loss_��<��g�       �	�?�+fc�A�*

lossw��;#%.�       �	A��+fc�A�*

lossD�<O���       �	hw�+fc�A�*

lossV�=�C�       �	��+fc�A�*

lossl�1<~�4       �	�0�+fc�A�*

loss\nC<��J�       �	_��+fc�A�*

loss�.�=�4��       �	�p�+fc�A�*

loss�R<gn       �	F
�+fc�A�*

lossw��;���       �	���+fc�A�*

loss��`<�1�       �	3O�+fc�A�*

loss��!<��z@       �	[��+fc�A�*

loss �C<49.�       �	��+fc�A�*

loss��<�Е�       �	�8�+fc�A�*

loss�P�<�5�h       �	���+fc�A�*

loss��<�x	�       �	�o�+fc�A�*

loss�=���       �	��+fc�A�*

loss��: �h�       �	���+fc�A�*

loss*�<p �v       �	�B�+fc�A�*

lossf�=&�       �	L��+fc�A�*

lossZ�<�0PH       �	�z�+fc�A�*

loss�?�<���|       �	��+fc�A�*

lossl#Z=�M��       �	a��+fc�A�*

loss?�H=�֚�       �	�=�+fc�A�*

loss_�;�q)�       �	���+fc�A�*

loss�?=�5��       �	���+fc�A�*

lossL�*<�4+d       �	40�+fc�A�*

lossQu�<�`M!       �	-��+fc�A�*

lossT�;��       �	+n�+fc�A�*

loss�K�<�<��       �	��+fc�A�*

lossrCM<�ܷ       �	��+fc�A�*

loss�ǰ<946G       �	�J�+fc�A�*

loss��0;9NH2       �	���+fc�A�*

loss�&n<g��       �	ۇ�+fc�A�*

lossO�<�VǪ       �	�$�+fc�A�*

loss��l<N�5       �	w��+fc�A�*

lossdIv<�F$�       �	�a�+fc�A�*

lossA��;!�<       �	���+fc�A�*

loss�3�<k�Y       �	K��+fc�A�*

loss��;�.5
       �	�E�+fc�A�*

loss�c =;=�0       �	���+fc�A�*

loss�<�cx�       �	,}�+fc�A�*

loss=�@0       �	��+fc�A�*

loss�\<��       �	F��+fc�A�*

loss�u�<��d       �	IK�+fc�A�*

loss�%=uR]t       �	 ��+fc�A�*

loss���<�ˈ�       �	g{�+fc�A�*

loss�}4=���	       �	��+fc�A�*

loss�>�<�>.       �	��+fc�A�*

loss �<ٛ��       �	���+fc�A�*

loss���<l�Һ       �	YO�+fc�A�*

loss�y�<Qֿ�       �	���+fc�A�*

lossd=����       �	��+fc�A�*

loss�F='��}       �	N'�+fc�A�*

lossq��:H-��       �	��+fc�A�*

loss�>D:!x�       �	���+fc�A�*

loss�%�;��n       �	���+fc�A�*

loss-�=_�UO       �	$�+fc�A�*

loss���;5�}f       �	6��+fc�A�*

loss_�i;�_��       �	�o�+fc�A�*

loss�=<�v=�       �	@�+fc�A�*

lossh�='ur       �	���+fc�A�*

loss�'=��r       �	�N�+fc�A�*

lossͽ�=�(�       �	q��+fc�A�*

loss:�<қ�d       �	���+fc�A�*

loss�m<�       �	�!�+fc�A�*

lossù�;����       �	M��+fc�A�*

loss�|�;���       �	CV�+fc�A�*

loss1΂;2Y�       �	���+fc�A�*

loss�LW;nצj       �	v��+fc�A�*

loss��\<�y�7       �	�~�+fc�A�*

loss\8�:\��E       �	zP�+fc�A�*

loss�]%<.:q       �	��+fc�A�*

loss�]�<6ӭ       �	�i�+fc�A�*

loss!�<�BM3       �	�C�+fc�A�*

lossnC�=��<>       �	hz�+fc�A�*

loss�=��       �	m��+fc�A�*

loss��9=�՜�       �	L��+fc�A�*

lossEN�<�G�q       �	@k�+fc�A�*

lossx� <>�8�       �	Ů�+fc�A�*

loss�U<�Z�       �	|F�+fc�A�*

loss�g=P�L�       �	���+fc�A�*

loss�N=���       �	���+fc�A�*

loss��=���1       �	�%�+fc�A�*

loss|!	:uu        �	\��+fc�A�*

loss_�.<��s�       �	��+fc�A�*

loss��<��m#       �	j ,fc�A�*

loss��.=���       �	i,fc�A�*

loss�l;�i�       �	D�,fc�A�*

loss��=wb�       �	1x,fc�A�*

loss�
�=��W�       �	�,fc�A�*

lossiڰ;O�Q�       �	��,fc�A�*

loss�6�; �j�       �	�W,fc�A�*

loss�N�=>i�       �	s�,fc�A�*

loss�:)��v       �	ō,fc�A�*

losst�A<M���       �	ޫ,fc�A�*

lossEi>H�e�       �	�F,fc�A�*

loss�h�<�c�       �	��,fc�A�*

loss��=��k       �	�,fc�A�*

loss}�U=�x�       �	�	,fc�A�*

loss�r�=b�`R       �	��	,fc�A�*

lossq'�<NM�       �	t_
,fc�A�*

loss��<�|�Z       �	H�
,fc�A�*

lossA��=��       �	t�,fc�A�*

loss��<R�;       �	�5,fc�A�*

lossGR<����       �	��,fc�A�*

loss�<�צ�       �	�v,fc�A�*

lossw�j= �8�       �	�,fc�A�*

loss�S�<��       �	�,fc�A�*

loss긎<)���       �	sc,fc�A�*

lossOxL;����       �	
,fc�A�*

loss/e�<��       �	a�,fc�A�*

loss�g�<6FD+       �	�J,fc�A�*

loss��=��i�       �	��,fc�A�*

loss:F=���       �	�,fc�A�*

loss�-�<�4*/       �	(),fc�A�*

loss�®<@N�       �	��,fc�A�*

loss�լ; +	O       �	X,fc�A�*

loss��<���       �	�,fc�A�*

loss;<d_       �	z�,fc�A�*

loss�D�;y�j       �	�R,fc�A�*

loss���<r�[/       �	��,fc�A�*

losss�r=��/�       �	��,fc�A�*

loss�@=aW�       �	�<,fc�A�*

lossE��:tQ�       �	��,fc�A�*

lossa%=E��       �	׈,fc�A�*

loss;AD=F�Λ       �	A.,fc�A�*

lossT�<�5Ц       �	�,fc�A�*

loss�&�<���       �	n,fc�A�*

loss�@B;���       �	E,fc�A�*

loss�1=!�+       �	m�,fc�A�*

loss�{8<����       �	5E,fc�A�*

lossd!�=�df       �	��,fc�A�*

loss��;4�R'       �	e�,fc�A�*

lossB�<j���       �	2>,fc�A�*

loss�<���       �	R�,fc�A�*

loss���;�e�.       �	s ,fc�A�*

lossA`w; ;�3       �	�!,fc�A�*

lossD�<Q
��       �	�!,fc�A�*

loss�H�<�q�       �	NG",fc�A�*

loss���;��mo       �	��",fc�A�*

loss
4f=)Ҡ       �	�u#,fc�A�*

lossv�=�s �       �	$$,fc�A�*

loss�>>j�t�       �	��$,fc�A�*

loss�;�=��v�       �	[D%,fc�A�*

loss���<[K8       �	z�%,fc�A�*

loss�h�<��       �	�~&,fc�A�*

loss���<[u��       �	�',fc�A�*

loss�F<.*�       �	g�',fc�A�*

loss�;�a~�       �	�^(,fc�A�*

loss�< *��       �	��(,fc�A�*

loss��<���       �	��),fc�A�*

loss�+<���       �	81*,fc�A�*

loss/�c<��7�       �	X�*,fc�A�*

loss7*$=#r��       �	Sz+,fc�A�*

loss�p�<^E��       �	^,,fc�A�*

loss��=V#0�       �	?�,,fc�A�*

loss�]*=;�P       �	�B-,fc�A�*

loss��<v#       �	��-,fc�A�*

loss�Gm=!���       �	�.,fc�A�*

loss���=VϽ       �	F/,fc�A�*

loss|��;�	�       �	P�/,fc�A�*

lossA��:��        �	�{0,fc�A�*

loss���<��       �	�1,fc�A�*

loss���<r��       �	��1,fc�A�*

lossm�	=�?�       �	�W2,fc�A�*

loss���;*nN       �	�2,fc�A�*

loss�e_<��|D       �	{�3,fc�A�*

loss[�=���l       �	9&4,fc�A�*

loss��<(��       �	��4,fc�A�*

loss��<,.`       �	O\5,fc�A�*

loss2�J=����       �	��5,fc�A�*

loss��=U߬�       �	��6,fc�A�*

loss{ж=6��]       �	U17,fc�A�*

loss�0;懍�       �	%�7,fc�A�*

loss�J�<QY:�       �	�g8,fc�A�*

lossL<6�D�       �	�9,fc�A�*

lossf�;��і       �	*�9,fc�A�*

loss�c=a�@K       �	�F:,fc�A�*

loss|. </���       �	;�:,fc�A�*

loss�=<��3�       �	��;,fc�A�*

lossb;�#��       �	 %<,fc�A�*

loss`��=��93       �	��<,fc�A�*

loss%�v;3}Aw       �	�g=,fc�A�*

lossZW�<�߭       �	�	>,fc�A�*

loss���<	���       �	��>,fc�A�*

loss��Z<Δ�       �	�E?,fc�A�*

loss�j�=�eM        �	2�?,fc�A�*

loss���<(:       �	4�@,fc�A�*

loss�	=�U1�       �	_&A,fc�A�*

loss�G=�А�       �	<�A,fc�A�*

loss�<Q�y�       �	�XB,fc�A�*

loss�Z�<Q���       �	�C,fc�A�*

lossj=�_�       �	V�C,fc�A�*

lossH��;	�D�       �	u;D,fc�A�*

loss��W<�;k?       �	;�D,fc�A�*

loss��_<�]       �	�}E,fc�A�*

lossS��<<,��       �	�&F,fc�A�*

loss�B<G=�       �	��F,fc�A�*

loss6ͳ<8��E       �	\G,fc�A�*

loss��:I'}�       �	��G,fc�A�*

loss�z=����       �	�H,fc�A�*

lossu�<1��       �	0I,fc�A�*

loss{�F<(���       �	L�I,fc�A�*

lossC=<���       �	�bJ,fc�A�*

loss�O~<@�I[       �	��J,fc�A�*

loss��k;?�p       �	�K,fc�A�*

loss�=�<,�(       �	��L,fc�A�*

loss��=ˤ�       �	�?M,fc�A�*

lossu�<.�>       �	��M,fc�A�*

lossh�<:�"*       �	ƇN,fc�A�*

loss}�<*tD�       �	�3O,fc�A�*

loss��<��a       �	�O,fc�A�*

loss�k�<���       �	�P,fc�A�*

lossH �<iv�       �	�Q,fc�A�*

loss��<� Z�       �	�Q,fc�A�*

lossJ�&<׌E�       �	�SR,fc�A�*

losss��<��8       �	��R,fc�A�*

loss �j=�'!�       �	��S,fc�A�*

loss�D<:��       �	qT,fc�A�*

loss���<��#       �	g�T,fc�A�*

loss�#�<Qj�N       �	�QU,fc�A�*

lossfg<@��5       �	�U,fc�A�*

loss�ۋ<�c��       �	�QW,fc�A�*

loss�1�<�ϓ�       �	!X,fc�A�*

lossA=�!��       �	[�X,fc�A�*

loss(��<�>c�       �	�[Y,fc�A�*

loss�e�=
(X�       �	��Y,fc�A�*

loss�N�<�S_       �	}�Z,fc�A�*

loss��=�R       �	a3[,fc�A�*

loss�è<�܃�       �	[�[,fc�A�*

loss��R;i���       �	%;],fc�A�*

lossvt=�0�       �	��],fc�A�*

lossHi=^8��       �	q^,fc�A�*

loss�G�;���>       �	�_,fc�A�*

loss^�<�8��       �	K�_,fc�A�*

loss�q�;��Z       �	�K`,fc�A�*

lossM�<����       �	��`,fc�A�*

loss1=6=��2�       �	P�a,fc�A�*

loss枷;ǁ	       �	�*b,fc�A�*

loss�F�:c!�       �	��b,fc�A�*

loss�Ù<'>1       �	_zc,fc�A�*

loss ��;M��2       �	�d,fc�A�*

loss(x�;!ߝr       �	>�d,fc�A�*

loss*.A<R6�9       �	��e,fc�A�*

loss@D�<�iO       �	5�f,fc�A�*

loss!/ :T���       �	zog,fc�A�*

loss$?�:sf�        �	fh,fc�A�*

loss�s=���       �	�h,fc�A�*

loss��;`��       �	�Mi,fc�A�*

lossV�=����       �	
�i,fc�A�*

loss!EV=� �i       �	|�j,fc�A�*

lossҎ�<Ph��       �	�Bk,fc�A�*

lossv��<��       �		�k,fc�A�*

loss���;��       �	�l,fc�A�*

loss��<�鞦       �	@/m,fc�A�*

loss�=�;p M�       �	^�m,fc�A�*

lossd�>�\��       �	(|n,fc�A�*

loss��o=��J�       �	�o,fc�A�*

loss_�|<�Kw       �	�	p,fc�A�*

loss]�9;I�       �	�p,fc�A�*

loss;X�:#�ct       �	'�q,fc�A�*

loss�Ī=,i�       �	�Gr,fc�A�*

loss>�#<+0Ь       �	cGs,fc�A�*

lossum�;`@c�       �	Rt,fc�A�*

loss�g<=7��       �	��t,fc�A�*

lossE��:)�u@       �	��u,fc�A�*

loss�8�;�r�t       �	��v,fc�A�*

lossl�O<f'l�       �	��w,fc�A�*

loss��)<*��
       �	�,x,fc�A�*

loss6��;���       �	t�x,fc�A�*

lossN�<�x�&       �	Ӿy,fc�A�*

loss��<�L�       �	�z,fc�A�*

loss�SA=ӂ�       �	C{,fc�A�*

loss��*=�R       �	|,fc�A�*

loss�p9<����       �	��|,fc�A�*

loss��W=��       �	g�},fc�A�*

loss���<nV@       �	yw~,fc�A�*

loss��<\���       �	u=,fc�A�*

lossS:�;���       �	(�,fc�A�*

lossT0�<�e%       �	�؀,fc�A�*

loss� D<=��p       �	ڭ�,fc�A�*

loss�>D�V`       �	\t�,fc�A�*

loss�=�/CK       �	B#�,fc�A�*

loss�=;w�u       �	��,fc�A�*

lossS��;by8-       �	�߄,fc�A�*

loss8=<�       �	���,fc�A�*

loss��O<����       �	^ֆ,fc�A�*

loss��v;�v       �	�4�,fc�A�*

losse�Z=���W       �	=,�,fc�A�*

loss��$<�H߻       �	�M�,fc�A�*

loss2=�<|��C       �	"��,fc�A�*

loss�`�;+b?       �	x�,fc�A�*

lossi��;�%�       �	�D�,fc�A�*

loss�j�<��"�       �	nۍ,fc�A�*

loss��<�fT#       �	E��,fc�A�*

loss�}�<��=       �	oc�,fc�A�*

loss!�B<p}�N       �	��,fc�A�*

loss�-<�{b       �	�.�,fc�A�*

lossW5<2�[�       �	]��,fc�A�*

loss㗼<�}       �	ly�,fc�A�*

loss�N�;<�       �	(�,fc�A�*

lossع=��`       �	�ԓ,fc�A�*

lossD0 <U��       �	�w�,fc�A�*

loss��^=47n       �	��,fc�A�*

losss��<ӆ�       �	��,fc�A�*

loss3�X=�>a       �	ZF�,fc�A�*

loss�0<�b�       �	�,fc�A�*

loss��h;Sg�4       �	���,fc�A�*

loss��=���       �	�<�,fc�A�*

loss��1<��       �	�ؘ,fc�A�*

lossN�:/�&
       �	��,fc�A�*

loss�o:D��       �	�#�,fc�A�*

loss6�:��L4       �	���,fc�A�*

loss!��;�W       �	)Z�,fc�A�*

loss9�<��/       �	���,fc�A�*

loss�ӻ:�,�       �	{M�,fc�A�*

loss��<>|�       �	��,fc�A�*

loss(�<Ӻ/�       �	&��,fc�A�*

loss��;�E�       �	�1�,fc�A�*

loss���9�Dw�       �	Ơ,fc�A�*

loss�D�;�ʧ       �	__�,fc�A�*

loss�I=S$<       �	0��,fc�A�*

loss�P�;���       �	�֣,fc�A�*

loss���;���u       �	Ho�,fc�A�*

loss��i<�wu�       �	 �,fc�A�*

lossf��=xcڝ       �	죥,fc�A�*

loss}��:Uh       �	D�,fc�A�*

loss�b�<]0�       �	��,fc�A�*

loss\'�;�ݥ       �	��,fc�A�*

loss6S<ֱ�       �	���,fc�A�*

loss3�t=N;��       �	�V�,fc�A�*

loss �;�"�       �	��,fc�A�*

loss���=��@S       �	p��,fc�A�*

loss|�}=V�ʍ       �	]R�,fc�A�*

loss8�<:��f       �	��,fc�A�*

lossvBo=t��       �	���,fc�A�*

lossxV�;W�@       �	S?�,fc�A�*

loss�o:=�@	�       �	E��,fc�A�*

loss�<zx       �	?��,fc�A�*

loss(u;q��       �	K�,fc�A�*

losst��<�S�       �	-��,fc�A�*

lossਡ<JB�_       �	7�,fc�A�*

loss3��<�*%d       �	e�,fc�A�*

lossG�<�U       �	�,fc�A�*

lossf�>�Tt       �	��,fc�A�*

loss���<V��f       �	��,fc�A�*

loss=6*<1\       �	���,fc�A�*

loss<�?=��	�       �	RI�,fc�A�*

loss���<#�w       �	��,fc�A�*

loss�\�: BR�       �	'غ,fc�A�*

lossV�;��٨       �	˻,fc�A�*

lossb<�`@�       �	5|�,fc�A�*

loss!\4=3��       �	��,fc�A�*

loss��<��*       �	�
�,fc�A�*

loss�^2=}g��       �	���,fc�A�*

loss�wx=)�       �	��,fc�A�*

loss�،<�'�       �	���,fc�A�*

lossϨ�<�뭜       �	E�,fc�A�*

losss�<:�!,       �	[��,fc�A�*

lossh��;vR8�       �	�g�,fc�A�*

loss�4?<{� �       �	]�,fc�A�*

lossE��<6-�5       �	��,fc�A�*

loss=? <t�       �	�f�,fc�A�*

loss��<�N�       �	��,fc�A�*

loss�=<a��N       �	F��,fc�A�*

loss��<��Ɩ       �	�X�,fc�A�*

loss�h;t��       �	r��,fc�A�*

loss3CB<����       �	���,fc�A�*

loss��<:x�       �	>�,fc�A�*

lossC�=���       �	n��,fc�A�*

loss��<&Ѯ�       �	�x�,fc�A�*

loss	�;c|	�       �	D�,fc�A�*

lossh�=j��       �	@��,fc�A�*

lossJ��<�,�       �	qY�,fc�A�*

loss�)�<��c       �	V��,fc�A�*

loss���:����       �	��,fc�A�*

loss���:����       �	�(�,fc�A�*

loss���;�ב       �	�)�,fc�A�*

loss��/=���       �	���,fc�A�*

lossMύ={�       �	x_�,fc�A�*

loss1I�;o��       �	���,fc�A�*

loss魒<x�ޠ       �	���,fc�A�*

lossS.�<���w       �	�;�,fc�A�*

loss߅�<����       �	���,fc�A�*

loss��.<�?t       �	m�,fc�A�*

lossב<��Y"       �	��,fc�A�*

lossj��<�Q��       �	4��,fc�A�*

loss\Uy<�' f       �	�=�,fc�A�*

loss��D<y���       �	e��,fc�A�*

loss��O<�Q       �	�}�,fc�A�*

lossq�b=�c߸       �	}#�,fc�A�*

loss�+=�ħ       �	��,fc�A�*

loss�ǆ<��=�       �	Yj�,fc�A�*

loss|�Z:[       �	b,�,fc�A�*

loss3�=L�<,       �	O��,fc�A�*

loss�3�<5�B       �	�a�,fc�A�*

loss�(
=�!8�       �	��,fc�A�*

loss��=@!(�       �	6��,fc�A�*

lossjv�=�$n       �	w��,fc�A�*

loss�i�;T��O       �	���,fc�A�*

loss�>�<�H9       �	�;�,fc�A�*

lossȨ�<�~a�       �	Y��,fc�A�*

loss�=m*��       �	w�,fc�A�*

loss��<i���       �	#�,fc�A�*

loss�<�<���       �	ҩ�,fc�A�*

lossѲd<��v       �	FD�,fc�A�*

loss��%<Y�̍       �	V��,fc�A�*

lossMP�=�f<�       �	k�,fc�A�*

loss��1=穳       �	`�,fc�A�*

loss��<ͳ�F       �	0��,fc�A�*

loss �B=6�.       �	/4�,fc�A�*

loss�d=��/D       �	,��,fc�A�*

lossq=���       �	m�,fc�A�*

loss��T;���R       �	u�,fc�A�*

loss��;9b�a       �	V��,fc�A�*

loss?m=T�6s       �	u:�,fc�A�*

loss��!=�nn
       �	@��,fc�A�*

loss\;C;��w       �	�,fc�A�*

lossz��<y��       �	 -fc�A�*

loss� r;��j,       �	N� -fc�A�*

loss��<` �$       �	vQ-fc�A�*

lossno>lچ       �	�*-fc�A�*

loss��j<1C�[       �	��-fc�A�*

loss�3=�^�"       �	u-fc�A�*

lossWP�;R��P       �	�-fc�A�*

lossZҴ;"C�       �	��-fc�A�*

loss��;�0Pl       �	��-fc�A�*

loss��;��@       �	�?-fc�A�*

loss=̩�       �	;�-fc�A�*

lossc��;Ga�-       �	�u-fc�A�*

loss���=u��@       �	�-fc�A�*

loss���<�K�       �	4�-fc�A�*

lossJ��;@��X       �	Eb	-fc�A�*

lossӨ<=_	w�       �	
-fc�A�*

loss.�:�Q�{       �	��
-fc�A�*

loss{L<����       �	aP-fc�A�*

loss
��= OI�       �	��-fc�A�*

loss���<��N�       �	��-fc�A�*

loss�<��O�       �	#-fc�A�*

loss�r�:"�$       �	n�-fc�A�*

loss�<�h��       �	�`-fc�A�*

loss��a<�n�       �	��-fc�A�*

loss�ߛ:�ǝ       �	�-fc�A�*

loss�=�g�Q       �	�#-fc�A�*

loss�'D<DJ�       �	Q�-fc�A�*

lossm~=& �=       �	\Z-fc�A�*

loss%9=c�c"       �	��-fc�A�*

loss6=d"       �	��-fc�A�*

loss��<��c�       �	�!-fc�A�*

losseE�<B��       �	�-fc�A�*

loss2�=+�Um       �	�T-fc�A�*

lossSK�<�:        �	��-fc�A�*

loss_�,;8 =�       �	��-fc�A�*

loss�<P��>       �	-fc�A�*

loss7;�:�>       �	��-fc�A�*

lossh�<y��       �	QM-fc�A�*

lossp�<�'�       �	��-fc�A�*

losse��<�V<       �	x~-fc�A�*

loss�n=�+�n       �	�-fc�A�*

lossy<i��P       �	�%-fc�A�*

loss�?�<\&�$       �	��-fc�A�*

lossM9�;�/ u       �	�-fc�A�*

loss
}�;d �       �	/-fc�A�*

loss V�<"�       �	��-fc�A�*

loss�;ְ       �	s�-fc�A�*

loss?��<�\D�       �	�(-fc�A�*

loss��=��Wb       �	��-fc�A�*

loss��T<����       �	�p-fc�A�*

loss��<���       �	� -fc�A�*

loss�:K:!w�^       �	�� -fc�A�*

loss��:=���       �	�l!-fc�A�*

lossX.<M�6K       �	]"-fc�A�*

loss�ö;\�9       �	��"-fc�A�*

lossh�=I�1       �	�a#-fc�A�*

loss�)�<�6J�       �	G$-fc�A�*

loss9'<JB�       �	��$-fc�A�*

loss��;�v¥       �	�4%-fc�A�*

loss�5=�|�Y       �	�%-fc�A�*

loss��0<ey       �	ca&-fc�A�*

loss��\=��       �	��&-fc�A�*

loss�IW<���3       �	B�'-fc�A�*

loss�<D���       �	KY(-fc�A�*

loss���<��n�       �	f2)-fc�A�*

loss�'�;���D       �	c�)-fc�A�*

lossܗ<;�Rv�       �	�*-fc�A�*

lossC��<S��	       �	��+-fc�A�*

lossF�<E��       �	�?,-fc�A�*

lossa*=]r`       �	��,-fc�A�*

lossQ\=T�+}       �	is--fc�A�*

loss��D=��4       �	�.-fc�A�*

loss{��:`f       �	j�.-fc�A�*

loss���<=���       �	!;/-fc�A�*

loss�W�;?kV�       �	1�/-fc�A�*

loss�3�:NS       �	��0-fc�A�*

loss4m_=���6       �	�[1-fc�A�*

losslT�<��,       �	o�1-fc�A�*

loss4^[=ogQ       �	j�2-fc�A�*

loss��k=$ �(       �	�3-fc�A�*

loss�Ә<i3w�       �	��4-fc�A�*

loss.R>=���	       �	�O5-fc�A�*

loss�}�;,J       �	6-fc�A�*

loss��y:�̩       �	�6-fc�A�*

loss�n�;x��       �	"�7-fc�A�*

loss�n<���       �	��8-fc�A�*

loss��B;K�?�       �	g}9-fc�A�*

loss�i=��3�       �	�:-fc�A�*

losso�<S��       �	��:-fc�A�*

lossf�T<���o       �	�=;-fc�A�*

loss��<4ևi       �	��;-fc�A�*

loss<�<�2�H       �	E�<-fc�A�*

lossɩ =Q�'�       �	�R=-fc�A�*

lossG�<y��       �	��=-fc�A�*

lossX��; �f�       �	l�>-fc�A�*

lossE�7;F��X       �	[?-fc�A�*

losso3<��<�       �	�@-fc�A�*

loss��w;���\       �	��@-fc�A�*

lossKc<�)�2       �	�aA-fc�A�*

loss�<����       �	�B-fc�A�*

loss���<0h�O       �	��B-fc�A�*

loss�EQ=�j�       �	�C-fc�A�*

loss�=L�-�       �	2D-fc�A�*

loss[��<�S;�       �	u�D-fc�A�*

loss���<��S�       �	�]E-fc�A�*

lossŝJ<O�s�       �	-F-fc�A�*

loss?��<��       �	"�F-fc�A�*

loss�<�o��       �	K?G-fc�A�*

loss
J�;4���       �	M�G-fc�A�*

loss=��:2;C�       �	!uH-fc�A�*

loss��=��G�       �	p
I-fc�A�*

loss���;K��       �	+�I-fc�A�*

loss �<�/8       �	�8J-fc�A�*

lossR��<���t       �	��J-fc�A�*

loss��c<�y       �	iK-fc�A�*

lossX�;�r�8       �	�L-fc�A�*

loss|��=/{2�       �	3�L-fc�A�*

loss��\<�Gu<       �	)ZM-fc�A�*

lossH��<h!a�       �	E�M-fc�A�*

lossM�]<{�J�       �	�N-fc�A�*

loss���;�M!       �	�O-fc�A�*

loss7ߍ<���       �	��O-fc�A�*

loss�b�;�g��       �	�JP-fc�A�*

loss/��<��۫       �	X�P-fc�A�*

loss}�2<T�m       �	ۉQ-fc�A�*

loss(��9�jr�       �	�1S-fc�A�*

loss��%=DL u       �	��S-fc�A�*

lossl�;:B)�       �	��T-fc�A�*

loss�|4=����       �	|*U-fc�A�*

loss�;`=�'�       �	��U-fc�A�*

lossAY�:���       �	�\V-fc�A�*

loss �<hѹ�       �	��V-fc�A�*

loss�K�;��vM       �	��W-fc�A�*

lossۅ|:G�V0       �	|'X-fc�A�*

loss��==](��       �	��X-fc�A�*

loss�P�<��       �	�fY-fc�A�*

loss�+<ԟ�S       �	tZ-fc�A�*

loss?�O<Tc�       �	��Z-fc�A�*

lossJh; �+�       �	P[-fc�A�*

losstHt;�~H       �	��[-fc�A�*

lossxTF<�A<c       �	��\-fc�A�*

loss8?L<�g��       �	&]-fc�A�*

lossl�Q;d� �       �	G�]-fc�A�*

loss��=E�wI       �	Vb^-fc�A�*

lossXX�;�:+       �	z�^-fc�A�*

loss�=��P�       �	}�_-fc�A�*

loss��H=�ݏ       �	g+`-fc�A�*

lossЗ<qc�       �	��`-fc�A�*

loss��m<�^x�       �	�ba-fc�A�*

loss���<��EE       �	�a-fc�A�*

loss�L�<����       �	.�b-fc�A�*

loss$J�;�Cġ       �	�%c-fc�A�*

loss��<�       �	!�c-fc�A�*

loss!�&:増q       �	=ad-fc�A�*

loss)\�;N,v       �	4�d-fc�A�*

lossF��<Y��[       �	;f-fc�A�*

loss�A�=�]       �	��f-fc�A�*

loss�t�<b�Pi       �	+�g-fc�A�*

loss��<MYy       �	�!h-fc�A�*

loss D�<w��       �	�h-fc�A�*

loss� <\
S�       �	GYi-fc�A�*

loss)�;�F��       �	<�i-fc�A�*

loss�" =0z�f       �	��j-fc�A�*

loss���;�w6       �	��k-fc�A�*

lossOG=R�v�       �	W&l-fc�A�*

loss���<FXM�       �	n�l-fc�A�*

lossp�<4�HD       �	�Xm-fc�A�*

lossH�]<�kb       �	�m-fc�A�*

lossM��;�}:P       �	z�n-fc�A�*

loss�$>����       �	#o-fc�A�*

loss�z{<k`��       �	��o-fc�A�*

loss���;��@       �	��p-fc�A�*

lossE�;_ݛ�       �	��q-fc�A�*

loss#��<����       �	v�r-fc�A�*

loss���=�{��       �	�zs-fc�A�*

lossT�=�*��       �	d t-fc�A�*

loss?Z�=�8�.       �	�!u-fc�A�*

loss|�&=�.ę       �	AHv-fc�A�*

loss[=��       �	Xw-fc�A�*

losst�P=�ki�       �	�	x-fc�A�*

loss;��;��       �	��x-fc�A�*

lossve<�hg�       �	��y-fc�A�*

loss�o�<`J�       �	��z-fc�A�*

losshb=[^�       �	��{-fc�A�*

loss�1�;���       �	��|-fc�A�*

loss�v?=��\�       �	�h}-fc�A�*

loss��d=�r�P       �	w.~-fc�A�*

loss̦a<�FB�       �	qq-fc�A�*

loss��;]sd       �	^L�-fc�A�*

loss3#<�,�       �	{O�-fc�A�*

loss1e<\h��       �	��-fc�A�*

lossS�;��       �	��-fc�A�*

loss�}D<�}��       �	��-fc�A�*

loss�R�<w���       �	��-fc�A�*

losst�;e�jM       �	�7�-fc�A�*

loss�G;6A�=       �	k�-fc�A�*

lossa��<���       �	nĆ-fc�A�*

loss��=N�[�       �	R��-fc�A�*

loss�#<��ߑ       �	i6�-fc�A�*

lossZ�<�D�       �	Iڈ-fc�A�*

loss��<U���       �	5�-fc�A�*

loss,��=�#��       �	ٳ�-fc�A�*

lossR�<C~
       �	���-fc�A�*

loss&�"=,+*�       �	�Y�-fc�A�*

loss��<*��       �	 �-fc�A�*

loss��=䶽�       �	N��-fc�A�*

loss`�!=�s3	       �	��-fc�A�*

loss@^M=�i��       �	
��-fc�A�*

lossWd<Ae+       �	�ِ-fc�A�*

loss��;jT�c       �	8��-fc�A�*

loss��7<��v]       �	�R�-fc�A�*

loss�y�</O>5       �	4�-fc�A�*

loss;��;{�R       �	S��-fc�A�*

lossc��;~?��       �	5�-fc�A�*

loss$~	;O,�       �	z��-fc�A�*

loss1>�<M:�       �	A,�-fc�A�*

loss�V�<R��       �	q̖-fc�A�*

loss��u=&�       �	�e�-fc�A�*

loss��=S�QN       �	��-fc�A�*

loss�Y�<nF�       �	��-fc�A�*

loss�=-��       �	SZ�-fc�A�*

loss�3
=�#��       �	���-fc�A�*

loss��.<I��       �	0��-fc�A�*

lossU�=��j       �	r5�-fc�A�*

loss,(�:P;y=       �	�Λ-fc�A�*

loss�h	=ު`       �	�h�-fc�A�*

lossx�<��.       �	%�-fc�A�*

loss��=+ *       �	�-fc�A�*

loss<D�<e���       �	�6�-fc�A�*

loss��=��~       �	�͞-fc�A�*

loss�4�<�">�       �	��-fc�A�*

loss�=��V       �	;��-fc�A�*

loss��<���       �	�3�-fc�A�*

loss���=qAA�       �	��-fc�A�*

loss���<�.�x       �	�b�-fc�A�*

lossPh	<��       �	��-fc�A�*

loss�	�:�z"Y       �	\��-fc�A�*

loss�Ϊ<YĲ       �	�O�-fc�A�*

lossCL�;mP�:       �	��-fc�A�*

loss�;����       �	u��-fc�A�*

loss��<���       �	�5�-fc�A�*

loss�<��`�       �	Q٧-fc�A�*

loss�2{<��as       �	eq�-fc�A�*

loss\�V=���       �	��-fc�A�*

loss=�$=v�       �	��-fc�A�*

loss��<��E�       �	���-fc�A�*

loss�;�;f�%x       �	�8�-fc�A�*

lossCp;b��6       �	��-fc�A�*

loss�*�:�kܞ       �	��-fc�A�*

loss��;��b       �	i�-fc�A�*

loss;�1;���1       �	�\�-fc�A�*

loss1�<�wUb       �	�K�-fc�A�*

losss��<}���       �	,�-fc�A�*

loss8�P< �J�       �	���-fc�A�*

loss��=�m�-       �	�Y�-fc�A�*

loss��|<���       �	���-fc�A�*

loss�)�<[ű:       �	؞�-fc�A�*

loss{M�;�^�x       �	�>�-fc�A�*

loss�?2=
j�       �	&�-fc�A�*

lossL�%='Փ�       �	��-fc�A�*

loss �;Y�       �	���-fc�A�*

loss�<�A�       �	�?�-fc�A�*

lossv�=)Ĩ�       �	��-fc�A�*

loss%�<�X_i       �	֏�-fc�A�*

loss�
=T!X�       �	2�-fc�A�*

loss��<:u}z�       �	 Ѽ-fc�A�*

loss�JJ;wt�       �	�g�-fc�A�*

loss�O<Q��n       �	�	�-fc�A�*

loss��D;^��       �	m��-fc�A�*

loss(�<�j��       �	�E�-fc�A�*

loss�=-�P       �	��-fc�A�*

loss���;ϣ       �	�}�-fc�A�*

loss�=��%       �	�-fc�A�*

loss�=N��       �	���-fc�A�*

loss���<2��'       �	�J�-fc�A�*

loss�S�;����       �	���-fc�A�*

lossWߤ<ܡ��       �	��-fc�A�*

loss�{=�        �	�(�-fc�A�*

loss��[<=��]       �	��-fc�A�*

loss67a<eُs       �	���-fc�A�*

loss�j%=�&l       �	>&�-fc�A�*

lossX�s<�ly�       �	v��-fc�A�*

loss�z�<!�       �	Ac�-fc�A�*

lossO�\=�zvg       �	���-fc�A�*

lossv�<��@       �	ʧ�-fc�A�*

lossv<6��.       �	F�-fc�A�*

loss�+=��/@       �	.��-fc�A�*

loss�k�;��^�       �	7��-fc�A�*

loss��<>��T       �	�)�-fc�A�*

loss�+<�X�h       �	s��-fc�A�*

loss]�<N��       �	�s�-fc�A�*

lossRe�;q�(       �	��-fc�A�*

loss'$<���J       �	���-fc�A�*

loss�z= �g�       �	�G�-fc�A�*

lossA]9;/r/�       �	��-fc�A�*

loss��<��8       �	�{�-fc�A�*

loss6��<��	�       �	��-fc�A�*

loss�D=o�       �	��-fc�A�*

loss�J�:��ʥ       �	�O�-fc�A�*

loss��:d�>       �	���-fc�A�*

loss��Q<��?a       �	-��-fc�A�*

loss��8<.��;       �	F�-fc�A�*

lossH�C<�Il�       �	���-fc�A�*

loss��<���       �	��-fc�A�*

loss�=([dn       �	�5�-fc�A�*

loss��;Ĉ��       �	p��-fc�A�*

loss�^M=�3<       �	��-fc�A�*

loss�t�<�U�       �	U2�-fc�A�*

loss�5�<��b�       �	��-fc�A�*

lossž�;?7�i       �	�q�-fc�A�*

lossW� ;v�N�       �	j�-fc�A�*

lossC9<��E.       �	��-fc�A�*

loss%1;m�S       �	a�-fc�A�*

loss6�=��J6       �	��-fc�A�*

loss�<�0��       �	:��-fc�A�*

loss(�<�l�       �	�P�-fc�A�*

lossx�<�D��       �	���-fc�A�*

loss���;���       �	���-fc�A�*

loss��<5��       �	e8�-fc�A�*

lossio<�a��       �	���-fc�A�*

loss��<����       �	Wy�-fc�A�*

loss3��=�22�       �	��-fc�A�*

loss7ˉ=CK�       �	)��-fc�A�*

loss��<̙(�       �	�k�-fc�A�*

loss��;��
       �	��-fc�A�*

loss 3�=r�h       �	��-fc�A�*

loss���<��':       �	�@�-fc�A�*

loss%�Q<ݲ3�       �	\��-fc�A�*

loss,<�&�       �	#��-fc�A�*

loss3��<�'	       �	 �-fc�A�*

loss�{X<��'b       �	���-fc�A�*

loss�p'<_A��       �	���-fc�A�*

loss��=���       �	G��-fc�A�*

loss[�<����       �	�e�-fc�A�*

loss�<"�~       �	Rb�-fc�A�*

loss�-b<KZ��       �	0�-fc�A�*

loss���<iu       �	���-fc�A�*

loss��=��[�       �	^�-fc�A�*

loss�7�<gP�       �	%�-fc�A�*

loss�9>�vty       �	¢�-fc�A�*

loss}��;+��7       �	�F�-fc�A�*

lossN�=�&]�       �	`��-fc�A�*

loss��;7��R       �	w��-fc�A�*

loss��#;����       �	q!�-fc�A�*

lossDc=���G       �	��-fc�A�*

loss���;�j       �	�[�-fc�A�*

loss��*<d5m       �	_F�-fc�A�*

loss��
=�p�f       �	���-fc�A�*

loss��=���W       �	��-fc�A�*

loss��=��L~       �	�6�-fc�A�*

lossc��<b/�!       �	>��-fc�A�*

loss���<5�9�       �	/i�-fc�A�*

loss�M=-�.{       �	��-fc�A�*

lossVŦ;`�DA       �	��-fc�A�*

lossC�=i       �	EH�-fc�A�*

loss=�:����       �	���-fc�A�*

loss��;�$r�       �	�u�-fc�A�*

loss}�;�Wb       �	��-fc�A�*

lossto_=�	՚       �	���-fc�A�*

loss�
�<�f��       �	bH�-fc�A�*

lossL�r<R�Ώ       �	���-fc�A�*

loss�	#<c�tK       �	F}�-fc�A�*

loss��&<�D_       �	��-fc�A�*

loss�%Y<f���       �	��-fc�A�*

loss�]=�7�       �	�Z�-fc�A�*

loss�$=��9       �	I��-fc�A�*

loss$-<Z;$       �	���-fc�A�*

lossk4=AG�       �	)% .fc�A�*

lossRY�<G��:       �	P� .fc�A�*

loss�U=OO|�       �	ak.fc�A�*

loss�V�<b��       �	�.fc�A�*

loss��<R��       �	¤.fc�A�*

loss��0=��t<       �	G:.fc�A�*

lossύ@=�Ci�       �	Z�.fc�A�*

loss��L;��H�       �	6s.fc�A�*

loss�x;I���       �	.fc�A�*

loss��;��?�       �	q�.fc�A�*

loss��<E�       �	�F.fc�A�*

loss�}<�]$e       �	:�.fc�A�*

loss	��;��gE       �	ʉ.fc�A�*

lossQc�<��       �	=+.fc�A�*

loss��<<:Y       �	2�.fc�A�*

lossL,�<ۮ��       �	�h	.fc�A�*

loss�n�<Gu�
       �	�

.fc�A�*

loss�0m<�,�v       �	��
.fc�A�*

lossJ�<F~�       �	�{.fc�A�*

lossI�?=�Rg       �	Q.fc�A�*

loss��=�N]@       �	�.fc�A�*

lossI�;&�-H       �	�O.fc�A�*

loss��8</�d�       �	��.fc�A�*

loss
��;_3�       �	�.fc�A�*

lossZ;^��       �	�.fc�A�*

loss �<L&TU       �	o�.fc�A�*

loss{��<���       �	4�.fc�A�*

loss�2=��B       �	F}.fc�A�*

loss�MT<�;e�       �	.fc�A�*

loss��:fU�5       �	8�.fc�A�*

loss@�<@;�       �	b.fc�A�*

loss n,<Q?�       �	��.fc�A�*

lossIb�<�¬�       �	��.fc�A�*

loss��D=�K.B       �	A,.fc�A�*

loss�ە<C       �	��.fc�A�*

loss�đ=�T>9       �	��.fc�A�*

loss��;����       �	Z/.fc�A�*

loss ��<I3�       �	�.fc�A�*

loss܏�=��8�       �	�d.fc�A�*

lossC�=�:�&       �	��.fc�A�*

loss�9<��        �	�.fc�A�*

loss��<&@Վ       �	A.fc�A�*

lossW�	=M:�       �	��.fc�A�*

lossb
<#�$�       �	Ԃ.fc�A�*

loss�($<�u�       �	�.fc�A�*

lossL�'=!K��       �	R�.fc�A�*

loss��e=       �		R.fc�A�*

loss��A=}R�       �	��.fc�A�*

loss;�;P�o       �	�}.fc�A�*

losst�;�f��       �	.fc�A�*

loss��;���       �	��.fc�A�*

loss@��<T"�       �	�X .fc�A�*

loss��;4�OO       �	f� .fc�A�*

loss �(<�=r       �	h�!.fc�A�*

loss��Y<�|2�       �	�K".fc�A�*

loss�d�<�C�       �	�".fc�A�*

loss��<Z�;�       �	��#.fc�A�*

loss��;� CV       �	0�$.fc�A�*

loss�LJ<���       �	� %.fc�A�*

loss���; ��<       �	g�%.fc�A�*

loss�N;�Rl       �	+k&.fc�A�*

lossq.�:PǠ9       �	u'.fc�A�*

lossɻ�= �1%       �	o�'.fc�A�*

lossJ;yC��       �	�G(.fc�A�*

loss�N=<h�N�       �	��(.fc�A�*

loss,l=/i��       �	�w).fc�A�*

loss��6<ӓs{       �	�*.fc�A�*

loss(r�;����       �	��*.fc�A�*

loss�O<3�@�       �	tb+.fc�A�*

loss��c<"S*�       �	�+.fc�A�*

lossXo!<W�a�       �	2�,.fc�A�*

lossl8=]ُ�       �	�--.fc�A�*

loss}7=����       �	��-.fc�A�*

losswu�<Rh+�       �	,c..fc�A�*

lossH�{;KiQ       �	�..fc�A�*

loss���<���       �	��/.fc�A�*

loss*e=��       �	40.fc�A�*

loss��<l���       �	m�0.fc�A�*

loss{�W;�ۭ       �	Dl1.fc�A�*

lossd��<���       �	�2.fc�A�*

lossK ;�xG       �	�73.fc�A�*

lossl�;Tl�;       �	�-4.fc�A�*

loss��D=�጗       �	��4.fc�A�*

loss�=?�j       �	¿5.fc�A�*

loss���;�8�       �	.�6.fc�A�*

loss��=1 ��       �	(c7.fc�A�*

loss%"0<4k�       �	S"8.fc�A�*

loss���<�Hm       �	��8.fc�A� *

loss1x=�,�R       �	��9.fc�A� *

loss߄�; ��       �	Nd:.fc�A� *

loss��<���{       �	�:.fc�A� *

lossD�k;h޿        �	B�;.fc�A� *

lossN��<��       �	<=.fc�A� *

loss��a;����       �	��=.fc�A� *

loss��I;�K~�       �	Ƌ>.fc�A� *

loss1�o;D2l       �	-�?.fc�A� *

loss�� <$��       �	�P@.fc�A� *

lossD�<��2       �	e�@.fc�A� *

loss%0�;2��[       �	��A.fc�A� *

loss$Z�;���       �	d�B.fc�A� *

loss��	<3o       �	U�C.fc�A� *

loss�=vg�G       �	�qD.fc�A� *

loss#�1<]e��       �	^E.fc�A� *

loss�Y�<w�39       �	��E.fc�A� *

lossq}�;Ē�       �	�cF.fc�A� *

loss3��;�1�       �	,G.fc�A� *

loss�<�]��       �	��G.fc�A� *

loss�?=SgD       �	^MH.fc�A� *

loss�<�<�pX       �	)�H.fc�A� *

loss�<���7       �	e�I.fc�A� *

loss��n:�h�       �	G=J.fc�A� *

loss�q.<���c       �	��J.fc�A� *

loss�?;�e�       �	�tK.fc�A� *

loss�:�<{el�       �	�L.fc�A� *

loss<��:���7       �	h�L.fc�A� *

loss�P�<����       �	LTM.fc�A� *

lossi, =�ϵ�       �	��M.fc�A� *

lossiz#<�hM       �	��N.fc�A� *

loss��<�F�;       �	iO.fc�A� *

loss�<��F�       �	)P.fc�A� *

losst��<�(ľ       �	0�P.fc�A� *

loss�b\<��1       �	e9Q.fc�A� *

lossT�4;�X�       �	��Q.fc�A� *

loss�={y��       �	�tR.fc�A� *

loss<����       �	�S.fc�A� *

loss\h8C�m       �	��S.fc�A� *

loss�K;�5Uy       �	!VT.fc�A� *

loss�G�:�?:       �	y�T.fc�A� *

lossQ�r;/�3       �	_�W.fc�A� *

loss�;.�:       �	{MX.fc�A� *

lossI��:-��U       �	q�X.fc�A� *

loss�+�<?�v�       �	�Y.fc�A� *

loss�g�;�M�       �	�*Z.fc�A� *

loss��9G�c7       �	��Z.fc�A� *

loss�Q<9 �pS       �	�e[.fc�A� *

lossH>�:&�9�       �	*\.fc�A� *

loss#{�;�C��       �	-�\.fc�A� *

loss�Ye;-���       �	�0].fc�A� *

loss-�95r�       �	��].fc�A� *

loss��<cA��       �	1{^.fc�A� *

loss%E2>m8��       �	�_.fc�A� *

loss�m7:���9       �	ղ_.fc�A� *

loss67�<���       �	uY`.fc�A� *

loss�<�1�       �	��`.fc�A� *

loss�=[�8�       �	�a.fc�A� *

loss�C�<9 �f       �	�!b.fc�A� *

lossD��;�c       �	4�b.fc�A� *

loss|�<���       �	~Uc.fc�A� *

lossŨ�<���^       �	��c.fc�A� *

loss�R�;55�       �	�d.fc�A� *

loss��;�~V       �	C e.fc�A� *

lossc<ߺ�       �	��e.fc�A� *

lossr$�<H�~       �	��f.fc�A� *

loss���<p�{�       �	!Wg.fc�A� *

loss���=�gU\       �	�g.fc�A� *

loss�P5<���|       �	�h.fc�A� *

loss���<"u�       �	�i.fc�A� *

lossS�;#ա       �	��i.fc�A� *

loss�Y�<��m       �	��j.fc�A� *

lossJ��<�ǎ�       �	֐k.fc�A� *

loss�7<T]��       �	$Bl.fc�A� *

loss��:��       �	z�l.fc�A� *

loss���;�nq�       �	�~m.fc�A� *

loss�=ܴט       �	L3n.fc�A� *

loss:#�<]@_�       �	X�n.fc�A� *

loss|I";,��       �	co.fc�A� *

loss���;ކ��       �	�o.fc�A� *

loss]�]<Q��8       �	��p.fc�A� *

loss���<�U`�       �	��q.fc�A� *

loss�l�=?��       �	]�r.fc�A� *

loss��4=/��       �	)�s.fc�A� *

loss&�!<=H�       �	��t.fc�A� *

loss�6}<���K       �	�u.fc�A� *

lossWX<[~Z�       �	y�v.fc�A� *

loss!�X;-9xk       �	��w.fc�A� *

loss�7=�Tl�       �	��x.fc�A� *

lossr6@<�VB�       �	��y.fc�A� *

loss6l�;�w       �	�z.fc�A� *

loss�g<����       �	�X{.fc�A� *

losstDI=*�       �	��|.fc�A� *

loss,�<<��y       �	~9}.fc�A� *

lossjs�;�[U'       �	�}.fc�A� *

losseo�;���       �	�~.fc�A� *

loss&��<���       �	_@.fc�A� *

loss�n�<�%�       �	��.fc�A� *

loss�.�;�.�       �	��.fc�A� *

loss��<�)�2       �	��.fc�A� *

loss���<�f?       �	���.fc�A� *

loss��(<C�o       �	u<�.fc�A� *

loss.l�<[4�       �	��.fc�A� *

loss��;��i       �	�ф.fc�A� *

loss{��<<Q;       �	6s�.fc�A� *

loss���;��<       �	V�.fc�A� *

lossѰ�<E       �	�.fc�A� *

loss��5=aƬq       �	ǂ�.fc�A� *

loss��0<Đ��       �	.�.fc�A� *

lossvR=�N_�       �	�ş.fc�A� *

lossvU<2��D       �	�b�.fc�A� *

loss (=��RI       �	���.fc�A� *

lossï�;���B       �	���.fc�A� *

loss_�=��s       �	�'�.fc�A� *

loss�w=u�H       �	Ģ.fc�A� *

loss���;=�[�       �	F^�.fc�A� *

loss詗<P;V       �	#��.fc�A� *

loss��B;�8��       �	e��.fc�A� *

loss�p=w���       �	|'�.fc�A� *

loss��&=�!�       �	ü�.fc�A� *

lossmo�; :Ky       �	Q�.fc�A� *

loss�;��x@       �	X�.fc�A� *

loss��<H��       �	�.fc�A� *

loss��7<g/�       �	��.fc�A� *

loss�Ã<�T�       �	7�.fc�A� *

loss��<��|       �	���.fc�A� *

loss#�&=C3K�       �	�f�.fc�A� *

loss��+<& w�       �	���.fc�A� *

loss�)=�-i       �	���.fc�A�!*

loss8��<����       �	9�.fc�A�!*

loss=��       �	fݮ.fc�A�!*

loss�Y�<$^       �	�}�.fc�A�!*

lossz`�:�y��       �	� �.fc�A�!*

loss�\�<��       �	eİ.fc�A�!*

loss/�)=+�R       �	�b�.fc�A�!*

lossS��<�.��       �	X�.fc�A�!*

loss��<�x*       �	(��.fc�A�!*

loss��v<�;�4       �	̴.fc�A�!*

loss�J�;�IP�       �	�.fc�A�!*

loss�;�8�$       �	���.fc�A�!*

loss��=��       �	� �.fc�A�!*

loss�;�)J�       �	U¸.fc�A�!*

losse[9<����       �	���.fc�A�!*

loss�n�=��W       �	yX�.fc�A�!*

losss�=:4+?       �	���.fc�A�!*

loss�m9=�J��       �	D��.fc�A�!*

loss|�=찱y       �	+��.fc�A�!*

loss��;�7�       �	�F�.fc�A�!*

loss���;�g�       �	C �.fc�A�!*

loss���;ZU�y       �	��.fc�A�!*

lossR�<x5C       �	ˡ�.fc�A�!*

lossg;iP�B       �	B�.fc�A�!*

loss��<t��g       �	�-�.fc�A�!*

loss��;��-       �	���.fc�A�!*

loss��7;m�fP       �	�d�.fc�A�!*

lossx��;�~��       �	-%�.fc�A�!*

loss�7�<�;�       �	��.fc�A�!*

loss�;V�ʮ       �	-��.fc�A�!*

loss��<�I=N       �	�(�.fc�A�!*

loss��-<�kj�       �	���.fc�A�!*

loss!$�;i�Z�       �	�`�.fc�A�!*

loss�]�;��(       �	��.fc�A�!*

lossa�:�?       �	��.fc�A�!*

loss��;�t5�       �	�D�.fc�A�!*

loss���<��}�       �	=��.fc�A�!*

lossfP=[<�       �	��.fc�A�!*

lossRJL;���       �	K#�.fc�A�!*

loss.�:����       �	���.fc�A�!*

loss/<:<�-�       �	4e�.fc�A�!*

loss�:X�       �	~�.fc�A�!*

loss+R<���I       �	Q��.fc�A�!*

loss]E<j\�t       �	}<�.fc�A�!*

loss�(<��G�       �	��.fc�A�!*

lossiCa<�se       �	���.fc�A�!*

lossW�<:s�       �	�O�.fc�A�!*

loss�I	<�.�O       �	���.fc�A�!*

loss���<�ǅ�       �	"��.fc�A�!*

lossQ[=pc�       �	�I�.fc�A�!*

loss\��<j�Q(       �	n��.fc�A�!*

lossA^;{ř        �	ut�.fc�A�!*

lossA��<ld�       �	m�.fc�A�!*

losshq;f[C�       �	��.fc�A�!*

loss-޷;F�nH       �	�_�.fc�A�!*

loss�><ttk       �	j��.fc�A�!*

loss�!�<�9��       �	Ւ�.fc�A�!*

loss.9=�{       �	�,�.fc�A�!*

loss���<��4)       �	��.fc�A�!*

loss�cS<�`��       �	�_�.fc�A�!*

loss�v�9[�q       �	���.fc�A�!*

loss�U�;���a       �	~��.fc�A�!*

loss�o�;�B��       �	�"�.fc�A�!*

loss��X< <z�       �	M��.fc�A�!*

lossF�<�)��       �	:W�.fc�A�!*

loss.�;Hlv       �	���.fc�A�!*

lossR�=Rp�K       �	��.fc�A�!*

loss�?z;എJ       �	�#�.fc�A�!*

loss��=��_       �	���.fc�A�!*

lossë�=7���       �	�Q�.fc�A�!*

loss�/b<�x�d       �	���.fc�A�!*

loss`D�=���       �	���.fc�A�!*

loss���;�ڻ       �	�,�.fc�A�!*

loss]r�=Mli�       �	~�.fc�A�!*

lossʝ;�@�w       �	���.fc�A�!*

lossj�<��#�       �	�p�.fc�A�!*

loss6(<I�V�       �	��.fc�A�!*

loss�(<K��       �	9��.fc�A�!*

loss�<�G��       �	�h�.fc�A�!*

loss;�;�э       �	O�.fc�A�!*

loss�M<;��,       �	g��.fc�A�!*

lossv s<��r�       �	5�.fc�A�!*

lossɘ=$¯�       �	��.fc�A�!*

loss�܎<�x�        �	�o�.fc�A�!*

loss��<g���       �	9�.fc�A�!*

loss�z=F�k�       �	���.fc�A�!*

loss͆L;φ"       �	�7�.fc�A�!*

loss�8<Ƣ�       �	�;�.fc�A�!*

lossC�c;��k       �	,��.fc�A�!*

loss�T >@�(�       �	�q�.fc�A�!*

loss�:�<�Z�;       �	!�.fc�A�!*

lossj=<)�k       �	��.fc�A�!*

lossn<��#       �	�?�.fc�A�!*

loss�9%<�վm       �	���.fc�A�!*

losss��<�KK�       �	1x�.fc�A�!*

loss}ՙ<�c�       �	j�.fc�A�!*

loss�"=�`�       �	���.fc�A�!*

loss�ߋ<���       �	L�.fc�A�!*

loss�s/=A���       �	���.fc�A�!*

loss�x<s��       �	3��.fc�A�!*

loss?4V<SB7       �	�#�.fc�A�!*

lossl�:�@Uz       �	��.fc�A�!*

loss�{f=�a�U       �	���.fc�A�!*

lossuې; }��       �	E��.fc�A�!*

loss��?;!���       �	,)�.fc�A�!*

loss��w<�c��       �	t_�.fc�A�!*

loss���<r�F       �	�&�.fc�A�!*

loss<�s<��B(       �	��.fc�A�!*

losstߝ<f�˜       �	�P�.fc�A�!*

loss��;Η!�       �	\�.fc�A�!*

loss�ܽ<���       �	���.fc�A�!*

loss���<u�/�       �	�L�.fc�A�!*

lossw�< EP       �	�Z�.fc�A�!*

loss�<|��       �	�
�.fc�A�!*

loss��=ڻ(       �	��.fc�A�!*

loss��B<�_��       �	�n /fc�A�!*

loss9j
<�1f       �	8/fc�A�!*

loss��< �1�       �	��/fc�A�!*

loss唟<׃=	       �	�k/fc�A�!*

lossƒ <W�c�       �	#/fc�A�!*

loss;�<�a{j       �	��/fc�A�!*

loss��=N�m�       �	�Y/fc�A�!*

loss�*�<���       �	�/fc�A�!*

lossj<8��       �	��/fc�A�!*

loss:�Q=la1�       �	�Y/fc�A�!*

loss� �:o87_       �	�/fc�A�!*

lossÕ�;���       �	��/fc�A�!*

lossrc;
;�s       �	al/fc�A�!*

loss� �;�{�j       �	�	/fc�A�"*

loss�=PM�       �	��	/fc�A�"*

loss��<��H�       �	�u
/fc�A�"*

loss`i�;p2�       �	�/fc�A�"*

loss�:�<��?b       �	��/fc�A�"*

loss�;h�d�       �	�[/fc�A�"*

losscH4=�jM�       �	��/fc�A�"*

lossI?d<r�       �	�/fc�A�"*

loss�5�;8X��       �	Ǜ/fc�A�"*

loss}Ns<�*Dh       �	Y5/fc�A�"*

loss���;���       �	��/fc�A�"*

loss��1=��}�       �	zm/fc�A�"*

loss��:�D�G       �	�/fc�A�"*

loss�&�;��b�       �	Թ/fc�A�"*

lossS�<��(       �	�V/fc�A�"*

loss��5:eG*�       �	�/fc�A�"*

loss�?=���       �	�/fc�A�"*

loss�g�:���       �	9*/fc�A�"*

loss�;�<��P�       �	��/fc�A�"*

loss�_Z<U�R       �	�j/fc�A�"*

lossnW=����       �	C/fc�A�"*

loss��<ԟ��       �	��/fc�A�"*

loss oX=X��M       �	P/fc�A�"*

loss��:��Ԣ       �	��/fc�A�"*

loss<�?=�e       �	B�/fc�A�"*

loss�"�<Y���       �	E)/fc�A�"*

loss��=��?       �	��/fc�A�"*

lossA�=��fZ       �	ge/fc�A�"*

loss"�<YՆ       �	& /fc�A�"*

lossj�H9T�B       �	��/fc�A�"*

loss�_^<H�]       �	</fc�A�"*

loss�/!=���#       �	��/fc�A�"*

loss}�);�d        �	Gq/fc�A�"*

loss�(2<���       �	/fc�A�"*

loss���;�+B�       �	�/fc�A�"*

loss�=����       �	�N/fc�A�"*

loss�կ<Lo       �	�/fc�A�"*

loss�K�<h�~�       �	1} /fc�A�"*

lossE}�<pg       �	0*!/fc�A�"*

loss��=�C�       �	"�!/fc�A�"*

losso:�<	���       �	�S"/fc�A�"*

loss���<6���       �	�"/fc�A�"*

lossj�j;a�{G       �	��#/fc�A�"*

lossR��;�r'�       �	$/fc�A�"*

loss?8%<���       �	�$/fc�A�"*

loss
Y�<!{�       �	f%/fc�A�"*

lossMlP<&�F8       �	�&/fc�A�"*

loss1V�<���}       �	k�&/fc�A�"*

loss8�b<>�       �	H4'/fc�A�"*

loss3��<+��       �	��'/fc�A�"*

loss�t�<
�I�       �	9a(/fc�A�"*

loss�$�<SOt�       �	��(/fc�A�"*

loss�'�:6ʧ�       �	��)/fc�A�"*

losszm<]y��       �	O[*/fc�A�"*

lossŵ=Xy��       �	��*/fc�A�"*

loss��$=X�I       �	�+/fc�A�"*

loss:�F=�-��       �	�g,/fc�A�"*

loss�K�<j���       �	�-/fc�A�"*

loss�;<
0Lg       �	r3./fc�A�"*

loss��2=��       �	�./fc�A�"*

loss���;4a�       �	xc//fc�A�"*

loss�8<d�[Q       �	20/fc�A�"*

loss҃�;��kZ       �	2�0/fc�A�"*

loss��	<�r�.       �	B^1/fc�A�"*

loss��1=��97       �	@�1/fc�A�"*

loss1�P=�:@�       �	��2/fc�A�"*

loss�|=B��       �	x)3/fc�A�"*

loss�=���       �	�3/fc�A�"*

loss�7A=��       �	$�4/fc�A�"*

lossz��<%V>�       �	I�5/fc�A�"*

loss��;�]!?       �	�!6/fc�A�"*

lossF
r;}C4$       �	o�6/fc�A�"*

loss��<���       �	�V7/fc�A�"*

lossJ)H<�PL]       �	��7/fc�A�"*

loss�C�;E3�       �	�8/fc�A�"*

loss��<K���       �	^.9/fc�A�"*

loss�9D<��
C       �	�9/fc�A�"*

loss6;!;"B�	       �	0b:/fc�A�"*

loss��{</
��       �	��:/fc�A�"*

loss�Y(<o�,       �	a�;/fc�A�"*

loss�	=���       �	J~</fc�A�"*

loss��?=H��       �	=/fc�A�"*

loss	�;��cx       �	b�=/fc�A�"*

losscu=��ߤ       �	p\>/fc�A�"*

loss��<�[��       �	\?/fc�A�"*

lossO.�<����       �	ߤ?/fc�A�"*

loss��>=�%       �	1C@/fc�A�"*

loss��<�:�<       �	��@/fc�A�"*

loss���:ٿ       �	s�A/fc�A�"*

loss�;��o       �	�&B/fc�A�"*

loss=��<,�M�       �	��B/fc�A�"*

loss&�;HyS       �	�\C/fc�A�"*

loss��/<B��       �	��C/fc�A�"*

loss΋�<'7c       �	�D/fc�A�"*

lossc=;��m       �	�9E/fc�A�"*

loss�4<>a n       �	,�E/fc�A�"*

lossﭭ;�(�$       �	�|F/fc�A�"*

loss�-<�3:       �	�G/fc�A�"*

loss̷;X��       �	�G/fc�A�"*

lossC8�;nj0_       �	�OH/fc�A�"*

loss�Ȇ=t�        �	l�H/fc�A�"*

loss��;]B7�       �	�I/fc�A�"*

loss(��;���	       �	&J/fc�A�"*

loss85:��ċ       �	հJ/fc�A�"*

loss��;`��       �	oIK/fc�A�"*

loss�o=Տ_-       �	?�K/fc�A�"*

loss��G<�u�       �	@�L/fc�A�"*

loss�>�<�ؘ�       �	Y�M/fc�A�"*

loss�[�=*WGn       �	�N/fc�A�"*

loss-s5<{Ke�       �	>O/fc�A�"*

loss%+5<|)�       �	��P/fc�A�"*

loss\@�<�/`0       �	6uQ/fc�A�"*

loss�3)<eV�G       �	�R/fc�A�"*

loss&1�<��D3       �	_�R/fc�A�"*

loss��3;��O6       �	�PS/fc�A�"*

loss�u�<��.       �	k�S/fc�A�"*

loss�U<�'��       �	}�T/fc�A�"*

loss�LJ<'JO       �	�|U/fc�A�"*

loss��T<5�\�       �	�V/fc�A�"*

loss��;h�'�       �	1CW/fc�A�"*

loss�6<o�u�       �	C�W/fc�A�"*

loss��<7K�       �	�X/fc�A�"*

loss�+<h]       �	JY/fc�A�"*

loss���<��       �	��Y/fc�A�"*

loss$��=�Ր       �	/�Z/fc�A�"*

loss�&�:]~�       �	74[/fc�A�"*

loss��9��:7       �	D\/fc�A�"*

loss�&Z<Ju#{       �	,�\/fc�A�"*

loss���;�#�       �	9_]/fc�A�#*

loss���;	��       �	�^/fc�A�#*

lossLE�:'&X       �	��^/fc�A�#*

lossO;�qY�       �	�K_/fc�A�#*

loss�|�<ϩi       �	�_/fc�A�#*

losst��</9t       �	ڍ`/fc�A�#*

loss&p�;O���       �	�+a/fc�A�#*

loss�n�=M���       �	q�a/fc�A�#*

lossx�P;аk?       �	�bb/fc�A�#*

lossV
�;�L1.       �	2�b/fc�A�#*

loss��=��&X       �	��c/fc�A�#*

loss��P;+���       �	EHd/fc�A�#*

loss7';=�q�       �	��d/fc�A�#*

loss�N�:sCV=       �	�}e/fc�A�#*

loss�zK:�y�!       �	�f/fc�A�#*

loss۠�<L!�D       �	�f/fc�A�#*

lossrS�<�ER�       �	�Hg/fc�A�#*

loss�W�;��$�       �	��g/fc�A�#*

loss�x=6C��       �	 ~h/fc�A�#*

loss�J�<J��       �	ai/fc�A�#*

loss�g<���       �	�i/fc�A�#*

loss��<1w
�       �	�Pj/fc�A�#*

loss��-=��.       �	��j/fc�A�#*

loss� �<FI�       �	"�k/fc�A�#*

lossC�<����       �	%!l/fc�A�#*

loss��y;1cK       �	�l/fc�A�#*

loss`��<���       �	em/fc�A�#*

loss�!�:
���       �	mn/fc�A�#*

loss=�<���}       �	ԛn/fc�A�#*

loss\?<y�~�       �	?6o/fc�A�#*

loss�H<���       �	��o/fc�A�#*

loss�ĥ:�VQ       �	�p/fc�A�#*

loss��<���k       �	�>q/fc�A�#*

loss�2�<�x��       �	��q/fc�A�#*

loss�ɝ:R�-�       �	$~r/fc�A�#*

loss�y<�][�       �	�s/fc�A�#*

lossV�z=��C       �	Ѱs/fc�A�#*

loss�;(+��       �	��t/fc�A�#*

loss]N�:zo�|       �	�u/fc�A�#*

lossTM=�lzN       �	r�v/fc�A�#*

loss*�i<���       �	�'x/fc�A�#*

loss��<H'N�       �	[�x/fc�A�#*

loss�b>���       �	!z/fc�A�#*

loss�5L=����       �	"�z/fc�A�#*

loss?��;����       �	<i{/fc�A�#*

loss]��<���       �	� |/fc�A�#*

loss�,Q=V��R       �	��|/fc�A�#*

loss���<�ʄk       �	/�}/fc�A�#*

loss�\�<��Ĵ       �	(~~/fc�A�#*

loss��G=
e�       �	6/fc�A�#*

loss�7�=`�Q       �	n�/fc�A�#*

loss��<)���       �	&��/fc�A�#*

lossD�S=�p       �	�J�/fc�A�#*

loss���;���       �	�!�/fc�A�#*

loss(s<�l�7       �	��/fc�A�#*

loss��=����       �	�)�/fc�A�#*

loss g�<�;0�       �	Ö́/fc�A�#*

loss�A<�A,�       �	%ʅ/fc�A�#*

loss�˴;͙k�       �	�k�/fc�A�#*

loss�6=��[       �	�	�/fc�A�#*

lossU�;FU        �	窇/fc�A�#*

lossTl�;_+��       �	�G�/fc�A�#*

loss�b<,�       �	߈/fc�A�#*

loss���;l��       �	�z�/fc�A�#*

loss-i�:�k�'       �	��/fc�A�#*

loss.%=j��       �	-��/fc�A�#*

loss2"=�$�       �	�Y�/fc�A�#*

loss��;U�0�       �	���/fc�A�#*

loss�E"=�(       �	��/fc�A�#*

lossz��=��       �	�2�/fc�A�#*

loss�	=�5�0       �	5Ѝ/fc�A�#*

loss��.;�H]       �	㍎/fc�A�#*

loss���<���       �	�5�/fc�A�#*

loss�Y8=�r       �	�Ϗ/fc�A�#*

loss�`#;��Y       �	�h�/fc�A�#*

loss��;���       �	���/fc�A�#*

loss�k=���       �	�L�/fc�A�#*

lossD3�;Gt�       �	o�/fc�A�#*

loss�E;g��v       �	:��/fc�A�#*

lossP<���q       �	�;�/fc�A�#*

loss�3 =-��       �	��/fc�A�#*

lossL��<�"�       �	��/fc�A�#*

loss@=!��'       �	2"�/fc�A�#*

lossY�<խ��       �	v��/fc�A�#*

loss�P=���       �	$`�/fc�A�#*

loss3z�<����       �	���/fc�A�#*

lossQm�<_�G<       �	A��/fc�A�#*

loss�H�=IL��       �	\;�/fc�A�#*

loss�*=�}��       �	י/fc�A�#*

loss-(<��X=       �	Ww�/fc�A�#*

loss��&<���	       �	:[�/fc�A�#*

loss��B<�"U       �	���/fc�A�#*

loss�й:[,��       �	(��/fc�A�#*

loss]�1;(��       �	�D�/fc�A�#*

loss1S]=P��a       �	���/fc�A�#*

loss�{�;Ag       �	@��/fc�A�#*

loss���;X��       �	�T�/fc�A�#*

loss�(>l�s�       �	���/fc�A�#*

loss�֤=��%�       �	���/fc�A�#*

loss�:<�H�e       �	;�/fc�A�#*

loss��n<�g	�       �	�֡/fc�A�#*

loss�)7<
�|�       �	Gr�/fc�A�#*

loss@�1=���       �	/�/fc�A�#*

loss��<r��       �	W��/fc�A�#*

loss�=�<�j��       �	�\�/fc�A�#*

loss�<^��       �	'��/fc�A�#*

loss�~=�X,       �	��/fc�A�#*

loss�6;��[       �	0.�/fc�A�#*

loss㈰<s+�M       �	�צ/fc�A�#*

loss�S<��8	       �	{��/fc�A�#*

loss�Z�<��W       �	�E�/fc�A�#*

loss�<b �       �	W�/fc�A�#*

loss|3�<) �7       �	��/fc�A�#*

loss�<E��<       �	N(�/fc�A�#*

loss��x=q[ ;       �	���/fc�A�#*

lossW��<@�d�       �	xc�/fc�A�#*

lossL2�=�^�8       �	a��/fc�A�#*

loss���;����       �	��/fc�A�#*

lossv"�;�`-       �	+N�/fc�A�#*

lossId�<�Fz       �	��/fc�A�#*

loss('�<Onֵ       �	�/fc�A�#*

loss�xI<�ZI�       �	�E�/fc�A�#*

lossE��;n��        �	=�/fc�A�#*

loss<%k<����       �	���/fc�A�#*

loss/OC<���       �	� �/fc�A�#*

loss�<����       �	���/fc�A�#*

loss�X�:��;�       �	�j�/fc�A�#*

loss���=��-�       �	��/fc�A�#*

loss[8=9��
       �	ԙ�/fc�A�$*

losszL�<�zO�       �	�D�/fc�A�$*

loss���;��d       �	��/fc�A�$*

loss�{�<�=�c       �	���/fc�A�$*

loss`�=b���       �	OY�/fc�A�$*

loss�x<����       �	>��/fc�A�$*

loss�T<[C�       �	���/fc�A�$*

loss|`<C��       �	��/fc�A�$*

loss=ȋ;���       �	ĵ�/fc�A�$*

loss�Hm:-Ų�       �	�I�/fc�A�$*

loss:��;��q       �	z߹/fc�A�$*

lossb�=0w{       �	`t�/fc�A�$*

losss4=Zݑ�       �	��/fc�A�$*

loss#��;���       �	5��/fc�A�$*

loss��=<�Y       �	Q��/fc�A�$*

loss���;�ό�       �	�7�/fc�A�$*

lossE�:=��A       �	ͽ/fc�A�$*

lossO�=�/�S       �	�`�/fc�A�$*

loss�p =&՝�       �	0��/fc�A�$*

lossQB
=<g��       �	���/fc�A�$*

loss�=}`��       �	�T�/fc�A�$*

loss�ְ;��       �	5��/fc�A�$*

loss�%u<�-S�       �	���/fc�A�$*

loss��1<M��t       �	*�/fc�A�$*

loss��<�J.}       �	���/fc�A�$*

losss��;�į+       �	sI�/fc�A�$*

losssTj<�+?j       �	��/fc�A�$*

lossJջ;� �       �	/��/fc�A�$*

loss���<]٧�       �	�"�/fc�A�$*

loss���<�y�       �	^��/fc�A�$*

loss�'�;P �       �	�O�/fc�A�$*

loss���<7�p�       �	#��/fc�A�$*

loss�5�;���       �	���/fc�A�$*

lossu[=cF�c       �	�4�/fc�A�$*

loss��E=w7L0       �	���/fc�A�$*

loss��V=o[�       �	|��/fc�A�$*

loss�$=\��@       �	�,�/fc�A�$*

loss�<)"       �	 ��/fc�A�$*

loss�J�<�j       �	=��/fc�A�$*

loss�$�=J       �	\!�/fc�A�$*

loss��:<���       �	��/fc�A�$*

loss�=�|V�       �	)]�/fc�A�$*

loss�V<�|�       �	��/fc�A�$*

loss-�
=��d�       �		��/fc�A�$*

loss<:�o       �	j��/fc�A�$*

lossd�?={��       �	�!�/fc�A�$*

loss��:=�h       �	�J�/fc�A�$*

loss�@h=
@_�       �	���/fc�A�$*

loss�v<.��N       �	I��/fc�A�$*

loss6��<�*aj       �	
I�/fc�A�$*

loss�u=}��$       �	���/fc�A�$*

loss�$<�D�       �	���/fc�A�$*

loss�ھ<4B�       �		��/fc�A�$*

losst<��n       �	�5�/fc�A�$*

loss�1�;���"       �	���/fc�A�$*

loss��;�ˣ       �	2��/fc�A�$*

loss��G=sl�J       �		o�/fc�A�$*

lossX��;��l#       �	P�/fc�A�$*

loss��;|笻       �	A��/fc�A�$*

loss尮;~!9       �	�f�/fc�A�$*

loss7=<��f       �	h�/fc�A�$*

loss]/;���       �	B��/fc�A�$*

loss�I�;��ъ       �	xG�/fc�A�$*

lossv=N~       �	;��/fc�A�$*

loss�*;�c�       �	Ή�/fc�A�$*

loss�`�<+�JF       �	�'�/fc�A�$*

loss�v=R�f�       �	L��/fc�A�$*

loss�
<��6       �	�^�/fc�A�$*

lossϖq;S���       �	���/fc�A�$*

loss�>"=�@P�       �	���/fc�A�$*

loss/��<'%n�       �	u=�/fc�A�$*

loss��H<nQ%       �	%]�/fc�A�$*

loss�Ş<JCep       �	w��/fc�A�$*

loss[�K<C�h       �	6��/fc�A�$*

loss�)	<��
h       �	�,�/fc�A�$*

loss�?<��B       �	��/fc�A�$*

loss�6<b� �       �	Gu�/fc�A�$*

lossq}<��d       �	j�/fc�A�$*

loss�VP<H?d       �	���/fc�A�$*

loss� �<�y       �	�I�/fc�A�$*

loss_Ք=$2P       �	���/fc�A�$*

lossH�l<���i       �	�{�/fc�A�$*

loss��<��}z       �	��/fc�A�$*

lossx)u=Z]-�       �	*��/fc�A�$*

loss@u�=�)��       �	�B�/fc�A�$*

loss�>J<�4�f       �	���/fc�A�$*

loss)��<�oYi       �	ap�/fc�A�$*

loss1�<�       �	��/fc�A�$*

lossz�z=^�       �	���/fc�A�$*

lossc\7<���}       �	�5�/fc�A�$*

loss��;�	W       �	��/fc�A�$*

loss���<^���       �	nj�/fc�A�$*

loss2z;=�É�       �	��/fc�A�$*

loss�/�<�f�       �	ܜ�/fc�A�$*

loss���:�ѥ       �	�;�/fc�A�$*

loss���;���0       �	���/fc�A�$*

lossW��;�Cɵ       �	�{�/fc�A�$*

loss$�<[#�       �	!#�/fc�A�$*

loss�Z�;���       �	T��/fc�A�$*

loss�Ϫ;I��^       �	��/fc�A�$*

loss㞨<
��       �	��/fc�A�$*

loss���;�=�        �	���/fc�A�$*

losse5�<�R`       �	�F�/fc�A�$*

loss]�;Do��       �	&��/fc�A�$*

loss�e�;PDu       �	�y�/fc�A�$*

loss��=H�P       �	���/fc�A�$*

loss1��;G|'�       �	'��/fc�A�$*

lossc��;'�~�       �	�T�/fc�A�$*

loss�.�;�b       �	��/fc�A�$*

loss8R�<;�       �	-��/fc�A�$*

loss�h<��e<       �	i8�/fc�A�$*

lossq;�;���       �	���/fc�A�$*

loss�cJ<��       �	�}�/fc�A�$*

loss\�\; �¦       �	8,�/fc�A�$*

loss1,<��1I       �	2:�/fc�A�$*

loss|�:�c��       �	���/fc�A�$*

loss�_�;�k��       �	�q 0fc�A�$*

loss���<�=�       �	0fc�A�$*

loss�r�<q�W.       �	]�0fc�A�$*

loss
J�=�ޘ`       �	�B0fc�A�$*

loss(<INW�       �	��0fc�A�$*

loss��5<��i�       �	�0fc�A�$*

loss�ھ;@���       �	q0fc�A�$*

loss}��;I��       �	��0fc�A�$*

loss��<ԭ�t       �	�^0fc�A�$*

loss܄<�& r       �	��0fc�A�$*

lossZ�8$D��       �	�0fc�A�$*

lossŒd<���C       �	�)0fc�A�$*

loss)�n<j�Ĺ       �	�0fc�A�%*

loss�&};�p)       �	�Z0fc�A�%*

loss$�*;�Ђ�       �	��0fc�A�%*

loss�D�:~ǎ       �	��	0fc�A�%*

loss7L�;�:kX       �	�
0fc�A�%*

lossWt�;�ݟb       �	J�
0fc�A�%*

loss��8C�)�       �	�d0fc�A�%*

loss�8�?�       �	�0fc�A�%*

loss�5�;# Y�       �	0�0fc�A�%*

loss�M�<�ސ       �	�0fc�A�%*

loss�D�;��!       �	eU0fc�A�%*

loss���8�U��       �	�0fc�A�%*

loss*<���       �		�0fc�A�%*

loss��X=��"       �	�,0fc�A�%*

loss��~:��}       �	��0fc�A�%*

loss���<��LH       �	_�0fc�A�%*

loss�sY<4]�       �	�N0fc�A�%*

lossfu�<$�G       �	�0fc�A�%*

loss!3�;rj�G       �	L�0fc�A�%*

loss,�0=cM5       �	�T0fc�A�%*

loss��;ny��       �	��0fc�A�%*

loss��=(�D�       �	x�0fc�A�%*

loss��<�=�D       �	�0fc�A�%*

loss�
�;\1�       �	��0fc�A�%*

loss��<-���       �	E0fc�A�%*

loss�p�<�J��       �	��0fc�A�%*

loss,�<Ӄ�       �	��0fc�A�%*

lossOT<��       �	X:0fc�A�%*

loss�U�<bD'�       �	U�0fc�A�%*

loss(�;$�|       �	�x0fc�A�%*

loss�_�<!�s�       �	0fc�A�%*

loss�h�;���       �		�0fc�A�%*

loss%�*=I�{�       �	5D0fc�A�%*

loss�b;��:       �	v�0fc�A�%*

loss�7�<����       �	�s0fc�A�%*

lossR]�;�u��       �	0fc�A�%*

loss��<��r       �	Z�0fc�A�%*

loss��:.�R       �	10fc�A�%*

loss $a=��       �	`�0fc�A�%*

loss��V;��i       �	�_ 0fc�A�%*

loss!)�<$���       �	Z� 0fc�A�%*

loss�\<��5       �	@�!0fc�A�%*

lossT�!=�2�       �	�"0fc�A�%*

lossf�=��Q�       �	9�"0fc�A�%*

lossH2<m�AO       �	�M#0fc�A�%*

loss 
�<�r       �	�#0fc�A�%*

loss�E�;L��       �	��$0fc�A�%*

loss�^H;dBl8       �	ۊ%0fc�A�%*

lossҒ�<Guj       �	�#&0fc�A�%*

loss1پ<UuV�       �	��&0fc�A�%*

lossl�<��Y       �	�'0fc�A�%*

loss���<��Ô       �	�i(0fc�A�%*

loss�K�<Qrc�       �	�)0fc�A�%*

loss�_"<��=_       �	s�)0fc�A�%*

loss�*%<
^�       �	 U*0fc�A�%*

loss�,*;w-l�       �	��*0fc�A�%*

lossL|�;�       �	,0fc�A�%*

loss��<R��q       �	f�,0fc�A�%*

losss]�;�j�       �	Nb-0fc�A�%*

loss���;�SIp       �	S.0fc�A�%*

loss  =j~��       �	8�.0fc�A�%*

loss�o�;�r�s       �	�A/0fc�A�%*

loss"J=��O       �	U�/0fc�A�%*

loss�c�;���#       �	�v00fc�A�%*

loss��v;^O7�       �	,a10fc�A�%*

loss���;�`Ӱ       �	��L0fc�A�%*

lossN`$=b�!�       �	��M0fc�A�%*

loss�dB=�Aa5       �	o,N0fc�A�%*

loss*u;�_��       �	��N0fc�A�%*

loss�_�;��ua       �	$
P0fc�A�%*

lossH<��\.       �	��P0fc�A�%*

loss�ɥ<5��       �	�R0fc�A�%*

lossR/<YL�u       �	�bS0fc�A�%*

loss.�u;@aW�       �	��S0fc�A�%*

loss�P�;�0��       �	�T0fc�A�%*

loss_��<��l�       �		V0fc�A�%*

loss�v<�n       �	�2W0fc�A�%*

loss >=y��       �	��W0fc�A�%*

loss��<�XE+       �	�kX0fc�A�%*

loss��A=��X(       �	�;Y0fc�A�%*

lossO�<�o�	       �	�Y0fc�A�%*

loss���9�|-#       �	I�Z0fc�A�%*

loss�$g<\wΓ       �	X[0fc�A�%*

loss���<~@�       �	�q\0fc�A�%*

loss�?=�+.�       �	�]0fc�A�%*

losso��:d _       �	s.^0fc�A�%*

loss���=r�`h       �	C�^0fc�A�%*

lossX&�:UGF�       �	�k_0fc�A�%*

loss֮�<��uN       �	u`0fc�A�%*

loss)?=��&!       �	��`0fc�A�%*

lossd�A<�5Е       �	�Ka0fc�A�%*

loss���<�       �	8�a0fc�A�%*

lossf%;�
RX       �	�"c0fc�A�%*

loss��<�       �	�d0fc�A�%*

loss=���I       �	!�d0fc�A�%*

loss�t;̐:�       �	�'f0fc�A�%*

loss��;����       �	��f0fc�A�%*

loss>T <��       �	yg0fc�A�%*

losss�4=��z       �	{h0fc�A�%*

loss�I=���H       �	�h0fc�A�%*

lossv*�<�k�       �	�Vi0fc�A�%*

loss���;��       �	�j0fc�A�%*

loss�6<�~       �	�Ik0fc�A�%*

loss��K=00݄       �	��k0fc�A�%*

loss�,<�躀       �	�l0fc�A�%*

loss��<?E       �	Y6m0fc�A�%*

losss�<e8�!       �	e�m0fc�A�%*

lossil�:�G1f       �	~n0fc�A�%*

lossmh�;b��       �	�o0fc�A�%*

loss���<WO^�       �	"p0fc�A�%*

loss��=�]�       �	��p0fc�A�%*

loss�c2<c���       �	K[q0fc�A�%*

loss�g�<�� �       �	��q0fc�A�%*

loss�	:��Du       �	!�r0fc�A�%*

loss�;[fdx       �	��s0fc�A�%*

loss�ǲ:�"��       �	4-t0fc�A�%*

loss�"�=���S       �	��t0fc�A�%*

loss1�:��       �	,�u0fc�A�%*

lossSn=j��       �	�v0fc�A�%*

loss�1<�q`v       �	��w0fc�A�%*

loss�,8;Eҁ;       �	qx0fc�A�%*

loss�.2<�+�       �	�
y0fc�A�%*

lossϤB:���)       �	��y0fc�A�%*

loss1`�;�F�       �	FAz0fc�A�%*

lossX,�<lxe       �	%�z0fc�A�%*

loss펛=��(#       �	��{0fc�A�%*

loss�}�<�@m�       �	�&|0fc�A�%*

loss\�;;���       �	x�|0fc�A�%*

loss�lf<�l�       �	~}0fc�A�&*

loss��;̈e       �	u ~0fc�A�&*

loss��@;7�7Y       �	`�~0fc�A�&*

loss{��;݀Kl       �	/k0fc�A�&*

loss���;x�/       �	�0fc�A�&*

loss��]=
�B       �	ǡ�0fc�A�&*

lossh�P<<��       �	�=�0fc�A�&*

lossM�S=�\�       �	Zԁ0fc�A�&*

loss�n�;k���       �	ji�0fc�A�&*

loss�;!<5=�       �	�
�0fc�A�&*

loss ��<���z       �	���0fc�A�&*

loss�L�<�d�       �	l@�0fc�A�&*

loss�+<@fL�       �	*��0fc�A�&*

loss��;Z_�       �	�9�0fc�A�&*

loss�b�;�C�$       �	��0fc�A�&*

loss?(�;��p`       �	��0fc�A�&*

lossV%=YY�       �	(��0fc�A�&*

lossL�|<m6^�       �	&Q�0fc�A�&*

loss�f=��T�       �	��0fc�A�&*

loss%";���       �	K��0fc�A�&*

loss��7:�x�       �	xE�0fc�A�&*

loss)v�:��<�       �	�݌0fc�A�&*

loss�*�;��?k       �	�~�0fc�A�&*

loss\��<�[�       �	)%�0fc�A�&*

lossO�;�YH       �	�Ǝ0fc�A�&*

loss�uy<�7p$       �	F_�0fc�A�&*

lossq`�<�H�V       �	�S�0fc�A�&*

loss4�=S���       �	ⱑ0fc�A�&*

loss�"�=��~)       �	�Q�0fc�A�&*

loss� �<t'0       �	��0fc�A�&*

loss��=e�9       �	*��0fc�A�&*

loss���<�r�       �	�*�0fc�A�&*

loss=A�<�m�0       �	�ϔ0fc�A�&*

loss�=Y��       �	lv�0fc�A�&*

loss�_�=�c�       �	[%�0fc�A�&*

loss���<;d��       �	�͖0fc�A�&*

loss`<�� �       �	k�0fc�A�&*

loss�N�<8A�a       �	��0fc�A�&*

loss	<�90n�6       �	ӥ�0fc�A�&*

loss��<��        �	�@�0fc�A�&*

lossY��;j9�P       �	�ٙ0fc�A�&*

loss�<'��       �	1}�0fc�A�&*

lossq�=�`L�       �	H�0fc�A�&*

lossr;�X�8       �	E��0fc�A�&*

loss�9(U�       �	R�0fc�A�&*

lossF�i=�Q�x       �	��0fc�A�&*

loss�Z<D�T       �	�|�0fc�A�&*

loss���;G��d       �	�"�0fc�A�&*

loss��x<CɁ       �	�Ϟ0fc�A�&*

loss@SA< �[M       �	Ae�0fc�A�&*

loss�>�;|�r�       �	���0fc�A�&*

loss+�:�/	       �	���0fc�A�&*

loss��-;*�J�       �	�7�0fc�A�&*

loss��;m       �	�ѡ0fc�A�&*

loss�:�<�y�*       �	�f�0fc�A�&*

loss���<���       �	���0fc�A�&*

lossM�=���       �	̘�0fc�A�&*

loss3�9;;|�"       �	v5�0fc�A�&*

loss�D�;	�B�       �	�Ϥ0fc�A�&*

loss*� =���=       �	�o�0fc�A�&*

loss�x�<`�ԗ       �	��0fc�A�&*

loss�؉<N�r�       �	���0fc�A�&*

loss�[o<Ǟ)�       �	�=�0fc�A�&*

loss��;�0��       �	kҧ0fc�A�&*

loss/�=��       �	<i�0fc�A�&*

loss*=��\�       �	%��0fc�A�&*

loss^N<�%�       �	rP�0fc�A�&*

loss��?=�Ba�       �	��0fc�A�&*

loss��<�r�       �	u��0fc�A�&*

loss9c#=�v��       �	�<�0fc�A�&*

loss��=p<7�       �	V�0fc�A�&*

loss�lR<��Ż       �	W��0fc�A�&*

lossO�;f%��       �	y:�0fc�A�&*

loss�c�<�b       �	D4�0fc�A�&*

loss�L�<�h�       �	K �0fc�A�&*

loss�><Is��       �	���0fc�A�&*

loss��;�       �	,a�0fc�A�&*

lossbq<m8��       �	�
�0fc�A�&*

loss��=E�Ģ       �	�<�0fc�A�&*

lossx��<P��Q       �	�޴0fc�A�&*

loss�Cf<�t�       �	D��0fc�A�&*

loss|z�<�P�\       �	�i�0fc�A�&*

loss-��;��b�       �	�v�0fc�A�&*

loss��"<w��       �	G�0fc�A�&*

loss]�<X?dn       �	��0fc�A�&*

loss��;���       �	��0fc�A�&*

lossa ~<��Y�       �	a��0fc�A�&*

loss!�<�*�       �	젼0fc�A�&*

loss�t�<�ԋ�       �	��0fc�A�&*

lossf��<�h�       �	��0fc�A�&*

lossv�<��       �	�:�0fc�A�&*

loss���=L�f4       �	\�0fc�A�&*

loss��q<�+ø       �	2��0fc�A�&*

loss$�:�
&       �	�>�0fc�A�&*

loss�\�=)��       �	/��0fc�A�&*

loss�^�<�Ѿ       �	���0fc�A�&*

lossf��<���       �	�(�0fc�A�&*

loss��i=w�>�       �	���0fc�A�&*

loss��;�Ko(       �	�s�0fc�A�&*

loss���<|��       �	��0fc�A�&*

loss�R�9w�3�       �	��0fc�A�&*

lossR�<Vr�       �	�D�0fc�A�&*

loss��M<m�i       �	3��0fc�A�&*

loss�{\:J&[�       �	mr�0fc�A�&*

loss(�=֠/�       �		�0fc�A�&*

loss@j�:�Nq�       �	��0fc�A�&*

loss-��<d�I�       �	?�0fc�A�&*

loss*�<m�-�       �	���0fc�A�&*

loss�9�!       �	q�0fc�A�&*

lossO(;�}�e       �	2�0fc�A�&*

loss\��;˦��       �	��0fc�A�&*

lossI��;�r��       �	�I�0fc�A�&*

loss�k�=\?S       �	:��0fc�A�&*

loss*4%=�vW4       �	���0fc�A�&*

lossS�/<h�       �	{�0fc�A�&*

lossi�y<ζ��       �	:��0fc�A�&*

loss��:W�       �	L�0fc�A�&*

loss�29��p|       �	���0fc�A�&*

loss3ܚ<AN�O       �	���0fc�A�&*

lossA	�;��Q�       �	#N�0fc�A�&*

loss��,;V�r       �	���0fc�A�&*

lossk�< ��}       �	|�0fc�A�&*

loss��:����       �	��0fc�A�&*

lossZ�<&J/       �	o��0fc�A�&*

loss�F�<��a^       �	�]�0fc�A�&*

loss�_==&�gJ       �	!�0fc�A�&*

loss�-=)U�        �	-��0fc�A�&*

loss��<�x�6       �	eQ�0fc�A�&*

loss�R	=Xt9z       �	r��0fc�A�'*

loss21�<X�n       �	��0fc�A�'*

loss��;4��       �	�h�0fc�A�'*

loss��:6��:       �	��0fc�A�'*

lossQ�;�rR       �	+��0fc�A�'*

lossI�=>���       �	��0fc�A�'*

loss{t6<���       �	{��0fc�A�'*

loss�)�<�l�F       �	�e�0fc�A�'*

lossS��;>ƛp       �	��0fc�A�'*

loss�Sx<�$ΐ       �	3��0fc�A�'*

lossN�=4�       �	�B�0fc�A�'*

loss���;�q�       �	���0fc�A�'*

loss#D�;�oR�       �	v��0fc�A�'*

loss��;�+1K       �	9(�0fc�A�'*

loss�R";m%?}       �	]��0fc�A�'*

loss���<e��E       �	a�0fc�A�'*

loss�0&<Z�S       �	��0fc�A�'*

lossl)�<,�zR       �	Z��0fc�A�'*

lossl|�<2�?h       �	�<�0fc�A�'*

loss�d�=����       �	^��0fc�A�'*

loss��M;CӚD       �	���0fc�A�'*

loss3�O<�n"p       �	u�0fc�A�'*

losssA;����       �	���0fc�A�'*

loss�D!=-A�       �	}��0fc�A�'*

loss��A=��}       �	a7�0fc�A�'*

lossn��=��.�       �	���0fc�A�'*

loss�%Z<�.p�       �	T��0fc�A�'*

loss�\=�⃢       �	B�0fc�A�'*

loss���<>�-       �	���0fc�A�'*

lossf��;�Z4�       �	��0fc�A�'*

lossZ�H;*a�       �	r5�0fc�A�'*

loss��;�VM       �	l��0fc�A�'*

loss,J�<��#       �	�s�0fc�A�'*

loss'e�;�rhF       �	��0fc�A�'*

loss�p�;v�L#       �	��0fc�A�'*

loss�>=\��       �	�N�0fc�A�'*

lossfz8=���       �	���0fc�A�'*

loss�N�:E� �       �	y��0fc�A�'*

loss��;h��g       �	�5�0fc�A�'*

loss���;���       �	��0fc�A�'*

loss�N�;$��       �	W��0fc�A�'*

loss%�;�ʽ       �	�R�0fc�A�'*

loss�<�!r       �	���0fc�A�'*

loss�x�<��       �	N��0fc�A�'*

loss�׭<e��       �	�{�0fc�A�'*

lossox:;%��       �	% �0fc�A�'*

loss�>
=)���       �	���0fc�A�'*

loss���<}�va       �	���0fc�A�'*

loss�;�:�6�C       �	jK�0fc�A�'*

loss�M�<�E��       �	G!�0fc�A�'*

lossz =��<       �	���0fc�A�'*

loss[��;¤�;       �	�c�0fc�A�'*

loss��D=8)JQ       �	���0fc�A�'*

loss㬫<s<]�       �	FD�0fc�A�'*

loss���:t�Ƙ       �	G��0fc�A�'*

loss4)�<a�$�       �	���0fc�A�'*

loss#��;��       �	�� 1fc�A�'*

loss�=���       �	$�1fc�A�'*

lossh�"<F	�       �	6:1fc�A�'*

loss#<��P       �	��1fc�A�'*

loss���<ٕ[�       �	��1fc�A�'*

lossh��;X�O8       �	i61fc�A�'*

loss�Q�;!�d�       �		�1fc�A�'*

loss :Ѩ�1       �	~�1fc�A�'*

loss.�-<!&R�       �	 W1fc�A�'*

loss���;�8�&       �	��1fc�A�'*

lossd�o<᫥�       �	"�	1fc�A�'*

loss��;͚;�       �	^*
1fc�A�'*

lossCr;���m       �	z�
1fc�A�'*

loss���;G^��       �	nm1fc�A�'*

loss���<3t��       �	W1fc�A�'*

lossW�<m��       �	��1fc�A�'*

loss��<���       �	cA1fc�A�'*

lossl{ =*�       �	��1fc�A�'*

loss�X< ^ޕ       �	ǝ1fc�A�'*

loss�"�<�nNe       �	fJ1fc�A�'*

loss�1=Lf��       �	��1fc�A�'*

loss�[<��       �	
�1fc�A�'*

lossڠ�=)��M       �	^H1fc�A�'*

loss4�I;����       �	
�1fc�A�'*

lossD�<�m�       �	}�1fc�A�'*

lossT�Z<�\1       �	U21fc�A�'*

loss5�<w�       �	��1fc�A�'*

lossv��=�i]       �	�p1fc�A�'*

loss��h=k�       �	�1fc�A�'*

lossߣD;����       �	h�1fc�A�'*

loss߂:�0�       �	�J1fc�A�'*

loss�~)<%��|       �	O�1fc�A�'*

loss��=���q       �	V�1fc�A�'*

loss,��9�~^{       �	!1fc�A�'*

loss��c:����       �	��1fc�A�'*

loss\W�:�̦6       �	KZ1fc�A�'*

loss�:�<�G�       �	f�1fc�A�'*

loss$`J=_�Mr       �	�1fc�A�'*

loss#�<}�=�       �	�>1fc�A�'*

lossL=<��       �	�1fc�A�'*

lossȈ<8Eq�       �	p{1fc�A�'*

loss3��<��G�       �	)"1fc�A�'*

lossZ�;1e�q       �	��1fc�A�'*

loss�"!;@��F       �	[\1fc�A�'*

lossA�;�Z��       �	D�1fc�A�'*

lossYx�:�#�S       �	ǜ1fc�A�'*

loss�4�97��       �	�8 1fc�A�'*

loss��;ʔ��       �	�� 1fc�A�'*

lossi�<J��       �	�o!1fc�A�'*

loss�$=<M>�       �	
"1fc�A�'*

loss�ZS=ӱ��       �	b�"1fc�A�'*

loss�Ҫ<�C       �	�E#1fc�A�'*

loss�3�<w{�.       �	`�#1fc�A�'*

loss���<�ل       �	ǃ$1fc�A�'*

loss�D�;����       �	�$%1fc�A�'*

lossE%)<i3A       �	�%1fc�A�'*

loss�mm=���I       �	tb&1fc�A�'*

loss,r<52>�       �	��&1fc�A�'*

loss��=��mm       �	פ'1fc�A�'*

loss��4;Z��b       �	(G(1fc�A�'*

lossê�;�{�       �	1)1fc�A�'*

loss8mS<�� �       �	x�)1fc�A�'*

loss��g;���2       �	�s*1fc�A�'*

loss&X};�V�6       �	+1fc�A�'*

lossqѭ<��bW       �	�+1fc�A�'*

loss�
�<D	       �	#I,1fc�A�'*

lossT�5;��_W       �	�,1fc�A�'*

loss7'F:��.       �	V�-1fc�A�'*

losse�R<���       �	�4.1fc�A�'*

loss�ߨ:N%�=       �	��.1fc�A�'*

loss��<��?       �	�c/1fc�A�'*

loss��<��NJ       �	R01fc�A�'*

lossCA<<JN�       �	�01fc�A�(*

loss��:<�Ki�       �	}[11fc�A�(*

loss�P>s71       �	��11fc�A�(*

loss��Q<츧T       �	�21fc�A�(*

lossc�<�V       �	�(31fc�A�(*

loss��I<�&Hv       �	��31fc�A�(*

loss�Z�='~�        �	�`41fc�A�(*

lossS(J;�F^       �	"�41fc�A�(*

loss�7t<��ʆ       �	5�51fc�A�(*

loss�<L�v       �	��61fc�A�(*

lossf��<r�h       �	V71fc�A�(*

loss8��;�`C       �	H281fc�A�(*

loss��H<%�       �	�91fc�A�(*

loss��9=Җ�       �	8:1fc�A�(*

loss!�;\��u       �	�;1fc�A�(*

loss3r�=w(�       �	!�;1fc�A�(*

lossX�(=���       �	��<1fc�A�(*

loss�pW<���       �	]�=1fc�A�(*

loss��;��V       �	�>1fc�A�(*

loss`�U<b�       �	#�>1fc�A�(*

losskn:���       �	��?1fc�A�(*

loss���:�jR        �	��@1fc�A�(*

loss*��:��       �	7nA1fc�A�(*

lossh��:�6E       �	bB1fc�A�(*

lossj��;�n�       �	�^C1fc�A�(*

loss͉=yf_�       �	�'D1fc�A�(*

loss�< .�       �	E1fc�A�(*

lossLA;�ȧ�       �	�JF1fc�A�(*

lossl��<�4F       �	�F1fc�A�(*

loss��(=�q�       �	q�G1fc�A�(*

loss��<Jq�       �	=H1fc�A�(*

loss��:x!:�       �	,~I1fc�A�(*

loss�q�<�>       �	l[J1fc�A�(*

lossE��=ӵ��       �	�K1fc�A�(*

loss*��<���       �	��K1fc�A�(*

loss2N�;U� r       �	LTM1fc�A�(*

loss��;Z/��       �	p�M1fc�A�(*

loss]J<�;IH       �	ÃN1fc�A�(*

loss=��;��t-       �	�$O1fc�A�(*

loss��;,���       �	u�O1fc�A�(*

loss��7;��*J       �	^�P1fc�A�(*

loss�~<Y��       �	�|Q1fc�A�(*

loss�2�;��Q       �	&R1fc�A�(*

loss��<�R�S       �	s�R1fc�A�(*

loss��<,�@       �	M.T1fc�A�(*

lossM �;XU       �	��T1fc�A�(*

loss�V<�u�;       �	qsU1fc�A�(*

loss_�(=J$C�       �	
V1fc�A�(*

loss�<<(x��       �	{�V1fc�A�(*

loss��<ך7E       �	�9W1fc�A�(*

loss3�c<v��       �	S�W1fc�A�(*

loss���<I)g       �	}yX1fc�A�(*

loss�E�<[�       �	bY1fc�A�(*

loss<<��7       �	կY1fc�A�(*

lossD+=v�)�       �	�QZ1fc�A�(*

lossM��;�0/       �	f�Z1fc�A�(*

loss�4u;t��}       �	�[1fc�A�(*

loss*�2=���       �	M-\1fc�A�(*

loss�$�<9e�n       �	��\1fc�A�(*

lossP�<���       �	�j]1fc�A�(*

loss��<s���       �	�^1fc�A�(*

loss��;�ٴ�       �	��^1fc�A�(*

losse�-;Ā�       �	�S_1fc�A�(*

loss�B�<C/e       �	��_1fc�A�(*

lossۮ;�5V       �	��`1fc�A�(*

loss���<8_k�       �	�da1fc�A�(*

loss3g%=��q�       �	��a1fc�A�(*

loss�R8<r
K�       �	�b1fc�A�(*

lossϴ�:(�       �	DMc1fc�A�(*

lossM��;l���       �	��c1fc�A�(*

loss:��<^��       �	��d1fc�A�(*

loss��=��p       �	�e1fc�A�(*

loss�^;�s�       �	�e1fc�A�(*

loss�\A<c��       �	�[f1fc�A�(*

lossD �<@ε       �	'2g1fc�A�(*

loss��;�.=�       �	��g1fc�A�(*

lossw1=�IP5       �	W�h1fc�A�(*

loss��2:# 
       �	�Fi1fc�A�(*

loss�ٜ;/ ۯ       �	.�i1fc�A�(*

loss��<��DP       �	��j1fc�A�(*

loss��<���       �	�'k1fc�A�(*

loss�<l<H8�       �	��k1fc�A�(*

loss�<�y�F       �	�`l1fc�A�(*

loss�:< ���       �	�l1fc�A�(*

loss�~�;>V=       �	8�m1fc�A�(*

loss�� <'6�       �	�:n1fc�A�(*

loss��<�Q�?       �	I�n1fc�A�(*

loss�8�<6g{~       �	\wo1fc�A�(*

loss}��;��!�       �	�p1fc�A�(*

loss*�<�L}^       �	Զp1fc�A�(*

loss�x�<���       �	�lq1fc�A�(*

loss���<JI��       �	�r1fc�A�(*

losssW=��-�       �	�r1fc�A�(*

loss��[=j�c       �	Ls1fc�A�(*

loss|]<�u�       �	m�s1fc�A�(*

loss�<���       �	�t1fc�A�(*

loss�&2<�	Q       �	y!u1fc�A�(*

lossZ�;zY#=       �	�u1fc�A�(*

loss���<��T�       �	w�v1fc�A�(*

loss:͊<�Ǜ�       �	�6w1fc�A�(*

loss��i< ���       �	�w1fc�A�(*

loss�-�<���2       �	�;y1fc�A�(*

loss��<Ý �       �	oz1fc�A�(*

loss��E:)QGH       �	d�z1fc�A�(*

loss�T�=KK{       �	I{1fc�A�(*

loss�@Q=�� h       �	�|1fc�A�(*

lossV�o<З�       �	 �}1fc�A�(*

loss��<t��R       �	#�~1fc�A�(*

loss���<ކL       �	Ӄ1fc�A�(*

loss��J=G�t|       �	��1fc�A�(*

loss�+H={:�       �	���1fc�A�(*

loss���; �'       �	j2�1fc�A�(*

loss��<�c�       �	t(�1fc�A�(*

loss	��;�0&�       �	��1fc�A�(*

loss�F�<��*       �	PS�1fc�A�(*

loss(��<��       �	�{�1fc�A�(*

loss�6v<ʧ|s       �	�@�1fc�A�(*

loss�k�<��g       �	C�1fc�A�(*

loss6̚;�<X       �	��1fc�A�(*

loss-�=�U�       �	 '�1fc�A�(*

loss<��p       �	,׉1fc�A�(*

loss���=~��        �	���1fc�A�(*

loss���<!bm�       �	�%�1fc�A�(*

lossnH�;�k�#       �	�1fc�A�(*

lossDT�< �_l       �	uv�1fc�A�(*

lossj�<�       �	�1fc�A�(*

lossl=<+        �	���1fc�A�(*

loss}S=={znh       �	�O�1fc�A�(*

loss:p<|Z       �	�1fc�A�)*

loss��<6g�^       �	��1fc�A�)*

loss3@<��m�       �	�1fc�A�)*

loss� �<�U�       �	>��1fc�A�)*

loss�w<�M��       �	N�1fc�A�)*

loss�/:< ��0       �	�1fc�A�)*

loss��K<�ؕ�       �	c~�1fc�A�)*

loss��< �       �	�"�1fc�A�)*

loss}5�;D�       �	���1fc�A�)*

lossdGZ;6¼�       �	]Q�1fc�A�)*

loss��<_K��       �	��1fc�A�)*

loss�C);�#Y       �	���1fc�A�)*

loss�/!<�0<�       �	;�1fc�A�)*

lossX�;�(:       �	���1fc�A�)*

lossm�:^8u       �	N�1fc�A�)*

loss<�(<�|�       �	G�1fc�A�)*

loss�=^�oQ       �	؀�1fc�A�)*

loss�$�;Ϊ��       �	���1fc�A�)*

losstr�;a�~�       �	�D�1fc�A�)*

loss8o <�j�       �	��1fc�A�)*

loss=��:���       �	_��1fc�A�)*

loss��;�I=       �	G:�1fc�A�)*

lossME%:�k��       �	8ٝ1fc�A�)*

lossM��<��P       �	yy�1fc�A�)*

lossS:�;���T       �	]�1fc�A�)*

loss�s4<n���       �	F��1fc�A�)*

loss�=�[�       �	wg�1fc�A�)*

lossn��;髓�       �	���1fc�A�)*

loss��<#�̪       �	���1fc�A�)*

loss��i:�}�1       �	�2�1fc�A�)*

lossc�*=>u��       �	�ڢ1fc�A�)*

loss�I�<d��g       �	>v�1fc�A�)*

lossm��<ׯ�       �	��1fc�A�)*

loss�=<,}       �	֧�1fc�A�)*

loss�I�;4C�        �	kH�1fc�A�)*

loss �;gx       �	v�1fc�A�)*

loss�Y�<�
��       �	5y�1fc�A�)*

loss�'<)�դ       �	��1fc�A�)*

loss��X;H2A�       �	峧1fc�A�)*

loss���:ː��       �	8H�1fc�A�)*

loss�|�<���       �	�;�1fc�A�)*

loss�~�;T�ǭ       �	�ϩ1fc�A�)*

loss�4�<|VS�       �	d�1fc�A�)*

lossoO<gM}W       �	�1fc�A�)*

loss�:�:K��1       �	���1fc�A�)*

losso�$;#�Bc       �	:;�1fc�A�)*

loss���;0��       �	Ѭ1fc�A�)*

loss��k=K	yu       �	�e�1fc�A�)*

loss	x�<�s\       �	�	�1fc�A�)*

loss� X=�%z       �	&��1fc�A�)*

lossT�H;�.z       �	�=�1fc�A�)*

lossŔ<P�ef       �	�ۯ1fc�A�)*

loss��<�(�2       �	�v�1fc�A�)*

loss���=�ɒ�       �	�5�1fc�A�)*

loss�^:(�@�       �	k�1fc�A�)*

loss�e	<@���       �	셲1fc�A�)*

loss�'>=�C       �	��1fc�A�)*

lossŸ<��M�       �	J��1fc�A�)*

lossC5�=U�a/       �	@L�1fc�A�)*

loss�I�;�ie       �	r�1fc�A�)*

loss��;��e�       �	x�1fc�A�)*

loss�K�=%9?�       �	��1fc�A�)*

lossl#�;p�[�       �	��1fc�A�)*

loss;�Q:��T8       �	�F�1fc�A�)*

loss��<_�pd       �	�ݷ1fc�A�)*

loss8�=��}*       �	���1fc�A�)*

lossv&�<��J       �	-�1fc�A�)*

loss�';v�l�       �	6ʹ1fc�A�)*

loss��<�rH�       �	Έ�1fc�A�)*

loss8'�<����       �	� �1fc�A�)*

lossj~;�6��       �	���1fc�A�)*

lossf5$:���       �	�Ѽ1fc�A�)*

losse��<H�1{       �	���1fc�A�)*

loss�&.=�       �	�k�1fc�A�)*

loss��m<���       �	#�1fc�A�)*

lossx��9�a�       �	GV�1fc�A�)*

lossV�=��P�       �	�Q�1fc�A�)*

loss31�<oЬ�       �	ob�1fc�A�)*

loss��	;�2-       �	���1fc�A�)*

loss�/=)#�       �	��1fc�A�)*

loss!�$=3ߜ�       �	���1fc�A�)*

loss��<�>�       �	x��1fc�A�)*

loss}n�;�g�       �	3�1fc�A�)*

lossbU:[2m�       �	�[�1fc�A�)*

loss��<S�7       �	��1fc�A�)*

loss��F<i�/       �	���1fc�A�)*

loss#��;�'D:       �	z��1fc�A�)*

loss�P�;K�g       �	���1fc�A�)*

loss={�;����       �	n5�1fc�A�)*

loss�>';9)Fk       �	��1fc�A�)*

loss1x�;U3��       �	�D�1fc�A�)*

lossA��:��m�       �	�#�1fc�A�)*

lossФ�;�u       �	7��1fc�A�)*

loss�!�<r��       �	+4�1fc�A�)*

loss��D9xHY�       �	���1fc�A�)*

loss���8/)>C       �	'��1fc�A�)*

loss��g;$��>       �	�n�1fc�A�)*

lossL�><+G       �	׈�1fc�A�)*

lossZ�;w�O       �	�0�1fc�A�)*

loss�Ց9�_��       �	��1fc�A�)*

loss=pH<��       �	S��1fc�A�)*

loss�+ =&�-       �	4��1fc�A�)*

loss=��:(@�       �	�f�1fc�A�)*

loss��<y��       �	�)�1fc�A�)*

loss3��;��w       �	���1fc�A�)*

loss	�< @b�       �	U0�1fc�A�)*

loss��</���       �	�E�1fc�A�)*

loss��<��       �	��1fc�A�)*

loss�@2<� �Z       �	5D�1fc�A�)*

lossA��</�;�       �	�y�1fc�A�)*

lossM;<���       �	`#�1fc�A�)*

lossx^�<���       �	Rd�1fc�A�)*

loss���<J���       �	���1fc�A�)*

lossf%�<8�)       �	[��1fc�A�)*

lossܚ+<�.H�       �	W��1fc�A�)*

loss"V<�5d       �	i��1fc�A�)*

loss��<��d.       �	�*�1fc�A�)*

loss�cJ<3��W       �	J��1fc�A�)*

loss3�n<n��       �	x}�1fc�A�)*

lossI&8;�IQ       �	b�1fc�A�)*

loss��9=@˶       �	2��1fc�A�)*

loss��=+��O       �	�S�1fc�A�)*

lossd�:ƻ��       �	���1fc�A�)*

loss E;��s�       �	u��1fc�A�)*

loss=���       �	31�1fc�A�)*

loss��Q;�r	       �	C��1fc�A�)*

lossF&;�a3�       �	=_�1fc�A�)*

lossE��;��\�       �	���1fc�A�)*

lossj�7<E9�       �	���1fc�A�**

loss3E�;���?       �	*�1fc�A�**

loss�u�<3�E�       �	x��1fc�A�**

loss�+�;�`��       �	,J�1fc�A�**

loss��<�1       �	���1fc�A�**

loss��7<N)��       �	I��1fc�A�**

loss��;�.�       �	��1fc�A�**

loss�ʀ;_�z       �	j��1fc�A�**

loss!�<s��P       �	�S�1fc�A�**

loss�W�;��G
       �	��1fc�A�**

loss��N<��ϟ       �	2��1fc�A�**

loss(��;��A+       �	W'�1fc�A�**

loss6��<����       �	���1fc�A�**

lossx;u�w       �	iS�1fc�A�**

lossj�%:[�/       �	���1fc�A�**

lossԠ�;���7       �	e��1fc�A�**

loss�;4���       �	�\�1fc�A�**

loss�ө<�j�|       �	E��1fc�A�**

lossd%>�o��       �	���1fc�A�**

loss2c�<��g�       �	�v�1fc�A�**

loss���<�@{       �	�4�1fc�A�**

loss㞢;l
.       �	���1fc�A�**

lossŷ�;�        �	�~�1fc�A�**

loss�3;����       �	.�1fc�A�**

loss�):ɿ��       �	��1fc�A�**

loss^�=�u��       �	�m2fc�A�**

loss��=A�k       �	�2fc�A�**

lossJ�D="��       �	��2fc�A�**

loss��<�5       �	f2fc�A�**

loss��<�>;O       �	p2fc�A�**

lossIG<%��       �	�2fc�A�**

lossZ3�<1��S       �	�>2fc�A�**

loss@��<�I       �	2fc�A�**

losslV]<G
�;       �	�2fc�A�**

loss�<4���       �	��2fc�A�**

loss���;��-�       �	�2fc�A�**

loss:.�<�q\�       �	 2fc�A�**

lossm�<�uʻ       �	ĵ2fc�A�**

loss�3�<��J       �	>^2fc�A�**

loss�3=�       �	��2fc�A�**

loss�5=YH�t       �	٘2fc�A�**

loss�U�9FF]�       �	<2fc�A�**

loss!<e�y       �	9�2fc�A�**

losso y<�et       �	uv2fc�A�**

losse;�<�r/\       �	!2fc�A�**

loss=�&<���{       �	*�2fc�A�**

loss͵�<�_R�       �	k2fc�A�**

loss��:�IU       �	$2fc�A�**

loss��l=�.��       �	�2fc�A�**

loss�$9<�G�       �	YQ 2fc�A�**

lossu��<����       �	�� 2fc�A�**

lossȥ�<�(O       �	}�!2fc�A�**

loss��L:ǵ�R       �	�;"2fc�A�**

loss��<X0�i       �	A�"2fc�A�**

loss�9�;�9       �	�o#2fc�A�**

loss��	<i�b9       �	p$2fc�A�**

loss�f�<���       �	�$2fc�A�**

loss�I�<$���       �	�>%2fc�A�**

loss�o�<���       �	��%2fc�A�**

lossPw<u9��       �	Cs&2fc�A�**

loss H<���       �	1
'2fc�A�**

loss�$);B]�       �	^�'2fc�A�**

loss��<qݡ%       �	r4(2fc�A�**

loss���<�Q�*       �	��(2fc�A�**

losso��<%-�       �	,c)2fc�A�**

loss��<-���       �	��)2fc�A�**

loss�>Q<.�-=       �	-�*2fc�A�**

lossjJ<KN��       �	I-+2fc�A�**

loss#�:<����       �	��+2fc�A�**

loss��;���)       �	d],2fc�A�**

loss��;
��       �	i�,2fc�A�**

loss�Xa<��YS       �	��-2fc�A�**

loss���;CK":       �	�@.2fc�A�**

loss��:tG��       �	v�.2fc�A�**

loss��;L:xM       �	�/2fc�A�**

loss���:~��i       �	002fc�A�**

lossxɯ<�F�       �	�02fc�A�**

loss�|�;��y�       �	�12fc�A�**

loss[D=h|�}       �	M/22fc�A�**

losse��;q�?�       �	6�22fc�A�**

loss�8:��(_       �	�t32fc�A�**

loss��<�B�9       �	�42fc�A�**

lossT��:A�!*       �	�42fc�A�**

loss��;�+�       �	�852fc�A�**

loss�|=�W\@       �	B�52fc�A�**

lossh�"=�a`v       �	�h62fc�A�**

loss=�;"u�M       �	�62fc�A�**

lossݔ�:�"I       �	R&82fc�A�**

lossA<����       �	��82fc�A�**

loss��;ы��       �	Do92fc�A�**

loss�:��{�       �	�:2fc�A�**

loss���<}ʹv       �	�:2fc�A�**

lossRc=uc>�       �	GT;2fc�A�**

loss���=c��h       �	��;2fc�A�**

loss��<�:2R       �	��<2fc�A�**

losscJ;�)��       �	�'=2fc�A�**

loss%�m=�^�       �	<�=2fc�A�**

loss�H<t�8�       �	�X>2fc�A�**

loss� �;�м�       �	�>2fc�A�**

loss���;c��       �	B�?2fc�A�**

loss�M�;��L�       �	�F@2fc�A�**

lossø	< ��+       �	Q�@2fc�A�**

lossU�<���       �	wA2fc�A�**

lossj�;�c˟       �	�B2fc�A�**

lossJ_�;�t�)       �	ƥB2fc�A�**

loss�=[c*]       �	*:C2fc�A�**

loss�i<�9>�       �	T9D2fc�A�**

loss�Dk<��fQ       �	��D2fc�A�**

lossH`�9:�       �	�{E2fc�A�**

loss.M;��!l       �	+F2fc�A�**

loss���<�;�       �	��F2fc�A�**

lossDlk<���       �	YPG2fc�A�**

loss��;�̇       �	��G2fc�A�**

loss��:�k��       �	ʋH2fc�A�**

loss׫�<TD*�       �	_)I2fc�A�**

lossv��<���       �	?�I2fc�A�**

loss8��<җ�       �	ݗK2fc�A�**

loss�ä;�BU�       �	W=L2fc�A�**

loss��<��       �	��L2fc�A�**

loss%�<�g2       �	S�M2fc�A�**

loss\Y�;.߼�       �	j2N2fc�A�**

loss!�n=ع��       �	��N2fc�A�**

loss�v�;<��       �	lO2fc�A�**

loss�=1�       �	�P2fc�A�**

loss�'-:6��       �	4�P2fc�A�**

loss�r<\U��       �	P7Q2fc�A�**

loss��+:�RR2       �	��Q2fc�A�**

losss�+<[ �       �	��R2fc�A�**

loss��;�       �	�>S2fc�A�+*

loss֧-<��X       �	o/T2fc�A�+*

loss=I=V.�       �	�7U2fc�A�+*

lossѻ�;C�K        �	W�U2fc�A�+*

lossF�:ŋ��       �	�rV2fc�A�+*

loss�֌<�KQ�       �	�W2fc�A�+*

loss�w�<�:M       �	��W2fc�A�+*

loss�f;�yXP       �	�FX2fc�A�+*

loss6_�;4�       �	�X2fc�A�+*

lossM�+=��       �	�{Y2fc�A�+*

loss/��<C��       �	+Z2fc�A�+*

lossnW&<��w       �	��Z2fc�A�+*

loss��x:�z�       �	�[2fc�A�+*

loss��;�L�|       �	�.\2fc�A�+*

lossH�k=�@K       �	B�\2fc�A�+*

loss�S[<%�       �	#k]2fc�A�+*

lossZ�}<Mxe       �	5^2fc�A�+*

lossD".<���E       �	ܺ^2fc�A�+*

losstO;0���       �	S_2fc�A�+*

loss6c<��x�       �	��_2fc�A�+*

loss��5;�+$       �	*�`2fc�A�+*

loss���;��       �	0)a2fc�A�+*

loss�ȥ<���       �	`�b2fc�A�+*

loss.�j<�Vd       �	
-c2fc�A�+*

loss�!w<��       �	��c2fc�A�+*

loss��G<���       �	Vfd2fc�A�+*

lossK��<�L!       �	f2fc�A�+*

loss�;f�\       �	?�f2fc�A�+*

loss4�<�f�Z       �	�Cg2fc�A�+*

lossC�;H��       �	��g2fc�A�+*

lossQK�<�wt�       �	�:i2fc�A�+*

loss���=�gT+       �	��i2fc�A�+*

loss��;u7       �	jij2fc�A�+*

lossl�;��H/       �	)k2fc�A�+*

losssW<�V       �	��k2fc�A�+*

loss<��:�i*�       �	�Cl2fc�A�+*

loss�<b��       �	��l2fc�A�+*

loss��d<���       �	�~m2fc�A�+*

loss�Y�<��       �	en2fc�A�+*

loss�+�<#��       �	дn2fc�A�+*

loss�j�;�-�"       �	Po2fc�A�+*

loss�X�=�S        �	��o2fc�A�+*

lossLR ;�E��       �	3�p2fc�A�+*

lossf�^<G��       �	a3q2fc�A�+*

loss��<��dt       �	l�q2fc�A�+*

loss7U�;�V�U       �	?rr2fc�A�+*

loss-�Q=�pEn       �	s2fc�A�+*

loss=��;�c�k       �	��s2fc�A�+*

loss�	=Y��       �	�Ot2fc�A�+*

loss�.�<�X       �	�Ru2fc�A�+*

loss�r�<˹n       �	��u2fc�A�+*

loss$��;8��M       �	2�v2fc�A�+*

loss�<�<̱ܞ       �	�Kw2fc�A�+*

loss���;Ze�       �	��w2fc�A�+*

loss�հ=�XS?       �	�x2fc�A�+*

loss��6<ׯ
=       �	Ty2fc�A�+*

lossI��;�4�       �	S�z2fc�A�+*

loss�5=��9�       �	��{2fc�A�+*

loss�6�;�JL       �	<�|2fc�A�+*

lossT�;O�       �	��}2fc�A�+*

lossWU=���       �	 �~2fc�A�+*

loss�*=^q��       �	��2fc�A�+*

loss1�<Y�F�       �	��2fc�A�+*

loss�;�dx       �	˞�2fc�A�+*

loss?R�<u���       �	$ւ2fc�A�+*

loss�>#<n��r       �	�z�2fc�A�+*

loss��s<�       �	0�2fc�A�+*

loss�J<;�       �	O��2fc�A�+*

loss�b;���G       �	r��2fc�A�+*

loss��X:����       �	�1�2fc�A�+*

lossV��;��jE       �	�Ն2fc�A�+*

lossԄ:Wi+Y       �	r�2fc�A�+*

lossq�=����       �	�X�2fc�A�+*

loss�{�=kՑ       �	L��2fc�A�+*

loss���<ټ�x       �	��2fc�A�+*

loss�&=�5[       �	1��2fc�A�+*

loss�B}<�t�       �	�2�2fc�A�+*

loss�:;�m�       �	�ϋ2fc�A�+*

loss��W<�F�       �	To�2fc�A�+*

loss��<��j�       �	,�2fc�A�+*

loss5�;QzsT       �	W��2fc�A�+*

loss���<�$P�       �	eT�2fc�A�+*

loss,7P<ݤŴ       �	��2fc�A�+*

loss�u=՘Z       �	�2fc�A�+*

loss���<:u1�       �	kؐ2fc�A�+*

loss��;�@��       �	�x�2fc�A�+*

loss�d�<�3<�       �	��2fc�A�+*

lossm��</-h�       �	s��2fc�A�+*

lossOY�<j�       �	�Q�2fc�A�+*

lossz<�:G�_       �	��2fc�A�+*

loss�m�;A�       �	=��2fc�A�+*

lossA/<+e�       �	w�2fc�A�+*

loss�-f; C�R       �	��2fc�A�+*

loss���<b�       �	�Y�2fc�A�+*

loss=݊=XX�5       �	��2fc�A�+*

lossH>=б0�       �	!ȗ2fc�A�+*

loss۹�<bQž       �	�3�2fc�A�+*

lossQ��;ƍ�       �	�љ2fc�A�+*

loss�D�<Q�x       �	�p�2fc�A�+*

loss�w�;��       �	�2fc�A�+*

loss�kp;���N       �	Ӥ�2fc�A�+*

loss=ˍ<c��Z       �	�:�2fc�A�+*

loss?�<�o��       �	5Ӝ2fc�A�+*

loss��<+z��       �	q�2fc�A�+*

loss#=d��       �	V�2fc�A�+*

lossJPw=��u/       �	���2fc�A�+*

loss�7)<��g       �	�\�2fc�A�+*

loss��^=�j�       �	.�2fc�A�+*

loss���=�h       �	霠2fc�A�+*

loss�g�:4��	       �	T5�2fc�A�+*

lossPl�;D���       �	6ˡ2fc�A�+*

loss4=,i#�       �	bi�2fc�A�+*

loss�G>^ɧ       �	l�2fc�A�+*

loss}=� ��       �	&T�2fc�A�+*

lossnI�<��h       �	n�2fc�A�+*

lossd�=�y*       �	Ե�2fc�A�+*

loss�R�<��|[       �	iQ�2fc�A�+*

loss�ϊ;��       �	�2fc�A�+*

loss�K!;���?       �	���2fc�A�+*

loss2��;��N6       �	��2fc�A�+*

loss��<T���       �	Ů�2fc�A�+*

loss���:��       �	�D�2fc�A�+*

loss�ӆ<�:       �	���2fc�A�+*

loss�ޏ<� �n       �	���2fc�A�+*

lossFa<l��_       �	��2fc�A�+*

loss���;�7�       �	c��2fc�A�+*

loss���;�	fc       �	�O�2fc�A�+*

loss���;y"Qu       �	��2fc�A�+*

loss���;���x       �	��2fc�A�,*

loss��<�� 9       �	�2fc�A�,*

lossn�<Y{֡       �	��2fc�A�,*

loss��=>���       �	]P�2fc�A�,*

loss�C�;Um�       �	��2fc�A�,*

loss�=a<�5L       �	 ��2fc�A�,*

lossf=lt;�       �	u?�2fc�A�,*

loss�C/=��,       �	�ӱ2fc�A�,*

loss�N�;���       �	rn�2fc�A�,*

lossJ�D<�1'�       �	}�2fc�A�,*

lossvN�<�T��       �	�ϳ2fc�A�,*

loss���;d�#�       �	�m�2fc�A�,*

loss�)�<�Ċ�       �	��2fc�A�,*

loss�B?=n�$�       �	���2fc�A�,*

loss��N=�S�+       �	\9�2fc�A�,*

lossř�;U���       �	�Ͷ2fc�A�,*

loss䦥<��M       �	>�2fc�A�,*

loss���;2�ߜ       �	Ҹ2fc�A�,*

lossV<�+V�       �	J|�2fc�A�,*

loss|S<<�ϓ�       �	;�2fc�A�,*

loss�ܛ={�!       �	�7�2fc�A�,*

loss	^<^�~�       �	�2fc�A�,*

loss7��;��ږ       �	���2fc�A�,*

loss6P�93&�       �	鷽2fc�A�,*

loss�uT<��Z       �	�S�2fc�A�,*

loss���<��C�       �	�<�2fc�A�,*

loss�<W;       �	nܿ2fc�A�,*

loss�_;�iU       �	e��2fc�A�,*

loss6t[;��(       �	M��2fc�A�,*

lossf@4=+���       �	�g�2fc�A�,*

loss�/�;X�m�       �	��2fc�A�,*

loss&�H<��;B       �	-?�2fc�A�,*

loss�U;%�ݨ       �	���2fc�A�,*

loss��{=rR[0       �	I��2fc�A�,*

loss�.:tC�       �	�B�2fc�A�,*

loss�Д<X��       �	���2fc�A�,*

loss(y$=>�n�       �	!��2fc�A�,*

loss�N=� 9�       �	z7�2fc�A�,*

lossf��;"4w�       �	6��2fc�A�,*

loss]��:	~�_       �	���2fc�A�,*

loss
��:ͽ�       �	�#�2fc�A�,*

loss���<J�	�       �	��2fc�A�,*

lossf/�<�n�v       �	uZ�2fc�A�,*

lossm:=�cv�       �	8��2fc�A�,*

loss�N�=�[M       �	���2fc�A�,*

lossf4�;W�       �	�2�2fc�A�,*

loss��;��       �	���2fc�A�,*

loss�2�<[o�u       �	e�2fc�A�,*

loss0=�
       �	2�2fc�A�,*

loss}�<��0       �	���2fc�A�,*

loss�ƞ;ڈ�       �	�D�2fc�A�,*

loss�?�;�$H�       �	F��2fc�A�,*

loss�G4=�B��       �	[	�2fc�A�,*

loss�3T<)�Ň       �	��2fc�A�,*

lossa=�}�t       �	���2fc�A�,*

lossN��<�g��       �	�R�2fc�A�,*

lossZ�I<�Cx,       �	0��2fc�A�,*

loss���:��7       �	��2fc�A�,*

loss���;'&��       �	IK�2fc�A�,*

loss2xw;�N'       �	���2fc�A�,*

loss�x*;�~d�       �	��2fc�A�,*

loss ;��v/       �	�"�2fc�A�,*

loss�*�9�]�       �	���2fc�A�,*

lossĻu=��<�       �	?U�2fc�A�,*

loss?J=��J�       �	���2fc�A�,*

loss��;���       �	�>�2fc�A�,*

loss�#<���       �	���2fc�A�,*

loss�n;1*�       �	��2fc�A�,*

loss�c<v��L       �	l=�2fc�A�,*

loss�E-<yw�       �	e��2fc�A�,*

loss.$<�h       �	�z�2fc�A�,*

loss#�=���|       �	�2fc�A�,*

loss���<b��_       �	���2fc�A�,*

lossԿ�;���       �	���2fc�A�,*

loss`ۙ<�Q�R       �	�D�2fc�A�,*

lossw-R:l�b�       �	��2fc�A�,*

loss���;#.R       �	���2fc�A�,*

lossL�<iS       �	78�2fc�A�,*

loss��R:����       �	���2fc�A�,*

loss�9�:����       �	�r�2fc�A�,*

loss�1;��J       �	��2fc�A�,*

loss���<@�Q       �	���2fc�A�,*

loss�r�;!C       �	+��2fc�A�,*

loss�(�;�I ;       �	��2fc�A�,*

loss�R�<P���       �	���2fc�A�,*

loss }�:U1q;       �	Jy�2fc�A�,*

loss�k@;�Y�       �	(�2fc�A�,*

loss�=І$       �	���2fc�A�,*

loss׾�<Rʦ�       �	�=�2fc�A�,*

lossD�B<#�Y�       �	l��2fc�A�,*

loss���<pV       �	�j�2fc�A�,*

loss[Е<j��       �	��2fc�A�,*

loss��;��       �	���2fc�A�,*

loss@.�<4#8       �	j4�2fc�A�,*

loss1AL=�Z�       �	���2fc�A�,*

loss|��<���       �	\�2fc�A�,*

loss��=����       �	���2fc�A�,*

loss���;O���       �	���2fc�A�,*

lossE.J=��G       �	�&�2fc�A�,*

loss��<e�?       �	���2fc�A�,*

lossXG�<�7[       �	�T�2fc�A�,*

lossW?�;�;k       �	���2fc�A�,*

lossJNr<7�G       �	Q��2fc�A�,*

loss�j<Bw?       �	i�2fc�A�,*

loss�q_;@i       �	ޮ�2fc�A�,*

loss2Bs<��t       �	xF�2fc�A�,*

lossa��;���       �	8��2fc�A�,*

lossv=�<1�C       �	qs�2fc�A�,*

loss"�:�Q�T       �	=�2fc�A�,*

loss��:E�|�       �	��2fc�A�,*

loss$�&;�"�       �	I�2fc�A�,*

loss!�E=\��       �	6Z�2fc�A�,*

lossȰ�<��A       �	���2fc�A�,*

loss�>y7�       �	�2fc�A�,*

loss�=���       �	?�2fc�A�,*

loss�K;�$�       �	���2fc�A�,*

loss�+&=���       �	�_�2fc�A�,*

lossR�<c}"�       �	��2fc�A�,*

loss,�l<�aY�       �	j��2fc�A�,*

loss��|:�$�Z       �	P8�2fc�A�,*

loss��:�/Δ       �	���2fc�A�,*

loss$�4=zW¥       �	)y 3fc�A�,*

lossj�=���       �	g3fc�A�,*

loss�Q<K&|�       �	��3fc�A�,*

loss��@<�ϫ�       �	�;3fc�A�,*

lossM�8<�h       �	�3fc�A�,*

loss�^�<����       �	rl3fc�A�,*

loss�i(=��D�       �	�?3fc�A�,*

loss�?l=���_       �	�C3fc�A�-*

loss�;*<�8       �		3fc�A�-*

loss�7R<xk�d       �	4�3fc�A�-*

loss1�=��       �	i53fc�A�-*

loss��=�b�       �	��3fc�A�-*

losss��<ZSA       �	�^3fc�A�-*

loss,(<@%sU       �	��3fc�A�-*

loss�;�T�A       �	�	3fc�A�-*

loss� 2<���        �	� 
3fc�A�-*

losse��<CPV�       �	��
3fc�A�-*

loss
��<�>k�       �	vP3fc�A�-*

loss��I=���V       �	��3fc�A�-*

loss�Lf:�^�Q       �	��3fc�A�-*

loss~ɓ=)�       �	�3fc�A�-*

lossM��<'8`Y       �	��3fc�A�-*

loss��<D1��       �	kH3fc�A�-*

lossmC$<�\�[       �	�3fc�A�-*

lossj8];ț�       �	Ԃ3fc�A�-*

lossj�=!*��       �	�3fc�A�-*

loss��=�� �       �	��3fc�A�-*

lossܷ=j��j       �	�\3fc�A�-*

lossY<��h�       �	��3fc�A�-*

loss]P�:�8?�       �	'�3fc�A�-*

loss���=�ˬ`       �	`3fc�A�-*

lossm3;T�k       �	ݱ3fc�A�-*

lossVg:%��       �	J3fc�A�-*

loss)�=�9%�       �	��3fc�A�-*

loss�^�<���       �	�u3fc�A�-*

lossA �;[O%       �	1	3fc�A�-*

loss�>�;�OA�       �	¡3fc�A�-*

loss��<��e       �	 C3fc�A�-*

loss�H�<o12       �	'�3fc�A�-*

loss�N=��       �	��3fc�A�-*

loss<���;       �	�J3fc�A�-*

lossM�J=��r       �	��3fc�A�-*

loss*��;��7F       �	s3fc�A�-*

lossM��<�ak       �	�3fc�A�-*

lossa5�9wJ�b       �	��3fc�A�-*

lossx4;c�j       �	e73fc�A�-*

loss��=J�ܸ       �	��3fc�A�-*

lossI�;�p�-       �	�c3fc�A�-*

loss���;��q       �	?�3fc�A�-*

loss���<8�       �	��3fc�A�-*

loss�V�<peF�       �	�-3fc�A�-*

loss.�[<~] +       �	��3fc�A�-*

loss� �<��>       �	�Y 3fc�A�-*

lossԶ�<?�^�       �	�� 3fc�A�-*

loss�X<z2Y�       �	L�!3fc�A�-*

loss�<AK�l       �	�'"3fc�A�-*

lossص,<�q�j       �	��"3fc�A�-*

loss�>�<���       �	�_#3fc�A�-*

loss�aw<�a4       �	b�#3fc�A�-*

loss��<$�ۄ       �	�$3fc�A�-*

loss�1	=�W-       �	�%%3fc�A�-*

loss��'=���v       �	j�%3fc�A�-*

losss��<�4=�       �	�V&3fc�A�-*

loss�BM<vXg�       �	W�&3fc�A�-*

loss�M�:fg˓       �	#�'3fc�A�-*

loss&�;z�C�       �	r(3fc�A�-*

loss�20<���z       �	O�(3fc�A�-*

loss@yz=�ZUl       �	�C)3fc�A�-*

loss��<Xa�       �	��)3fc�A�-*

loss��<��`�       �	�t*3fc�A�-*

loss�=9<`^w       �	�+3fc�A�-*

loss��Y=r��A       �	��+3fc�A�-*

loss��<�2       �	�:,3fc�A�-*

loss�+J<L�T�       �	��,3fc�A�-*

loss�B�;>˼�       �	�e-3fc�A�-*

lossϐ�<�Q�P       �	2.3fc�A�-*

lossإ<7�F       �	F�.3fc�A�-*

lossͬ�<Uٕ1       �	�//3fc�A�-*

loss��I<���       �	K�/3fc�A�-*

loss��;Q̚�       �	�l03fc�A�-*

lossӧ�;���J       �	d13fc�A�-*

lossw��=��2       �	��13fc�A�-*

loss��=���y       �	`923fc�A�-*

lossM��<���       �	J�23fc�A�-*

loss86�<���       �	8g33fc�A�-*

loss\p<�m�%       �	�43fc�A�-*

loss^m�<��e�       �	,�43fc�A�-*

loss1�;�L�       �	QN53fc�A�-*

loss�I�<�%�.       �	�53fc�A�-*

loss��=f���       �	�{63fc�A�-*

loss}�<��[]       �	�73fc�A�-*

loss&_�<�y#       �	Z�73fc�A�-*

loss�/<���       �	�683fc�A�-*

loss�ä;�R       �	@h93fc�A�-*

lossJM8=	Z�       �	Y�:3fc�A�-*

loss,�b<�D       �	�l;3fc�A�-*

loss(=G\��       �	Ԛ<3fc�A�-*

loss7�q<В�>       �	5�=3fc�A�-*

loss���<��P�       �	)Z>3fc�A�-*

loss6�+<��       �	F�>3fc�A�-*

loss�,X<�|ɱ       �	��?3fc�A�-*

lossX�:��<�       �	��@3fc�A�-*

loss�,=ly�       �	�bA3fc�A�-*

loss.��;7��V       �	[B3fc�A�-*

loss�#u<&��v       �	,�B3fc�A�-*

loss��H<��       �	�\C3fc�A�-*

loss&n/<�o�$       �	NAD3fc�A�-*

lossW&\<���       �	��E3fc�A�-*

loss��H;�^C�       �	�9F3fc�A�-*

loss���;�Cx       �	��F3fc�A�-*

losss�< X9�       �	�G3fc�A�-*

lossq��<-+%�       �	�8H3fc�A�-*

loss��=��4       �	��H3fc�A�-*

loss��D<�K�u       �	:�I3fc�A�-*

losso@�<VG;	       �	?SJ3fc�A�-*

loss���;��I�       �	��J3fc�A�-*

loss�Q�;$/�o       �	7�K3fc�A�-*

loss�);��       �	g�L3fc�A�-*

loss�Wf=Aogw       �	�1M3fc�A�-*

loss�� <�VI       �	��M3fc�A�-*

loss�<���       �	��N3fc�A�-*

lossl��<b�eD       �	 )O3fc�A�-*

loss�;��       �	�O3fc�A�-*

loss�@�;?��       �	CYP3fc�A�-*

loss�=E��q       �	��P3fc�A�-*

lossS�<��3r       �	'R3fc�A�-*

loss��L<:�ܠ       �	��R3fc�A�-*

loss<��<�!&       �	US3fc�A�-*

loss:�M<Pk>�       �	�ST3fc�A�-*

loss�`;��Ͱ       �	��T3fc�A�-*

loss��0<(<�L       �	9�U3fc�A�-*

loss{��;���       �	i6V3fc�A�-*

lossTm=�/�7       �	�W3fc�A�-*

loss��;V~�>       �	=X3fc�A�-*

lossm6Q<o΋�       �	��X3fc�A�-*

loss��;r�D'       �	]�Y3fc�A�.*

loss���;:cT2       �	"Z3fc�A�.*

loss���<�}��       �	̸Z3fc�A�.*

lossm`�<�       �	�Z[3fc�A�.*

loss)�;}R��       �	��[3fc�A�.*

lossE;���       �	Ɏ\3fc�A�.*

lossqM<�Ȣ�       �	�$]3fc�A�.*

loss*<�;�໹       �	/�]3fc�A�.*

loss��Y=D��       �	�X^3fc�A�.*

loss�ɉ<dx�       �	��^3fc�A�.*

lossm��;Y�M       �	�_3fc�A�.*

loss��;��       �	�6`3fc�A�.*

loss�Q�<t�7       �	��`3fc�A�.*

loss�W�<���.       �	�ka3fc�A�.*

lossJԟ<c��       �	Ib3fc�A�.*

loss��<�,<4       �	��b3fc�A�.*

loss��9<#�       �	Nc3fc�A�.*

loss���;�ԇ       �	��c3fc�A�.*

lossz�;�Z�;       �	�d3fc�A�.*

loss���<���       �	�e3fc�A�.*

loss�|E<�\bl       �	�e3fc�A�.*

loss���;p� �       �	�Tf3fc�A�.*

loss,C�;��1       �	N�f3fc�A�.*

loss���;�i�H       �	�g3fc�A�.*

loss�	�;j�T       �	5'h3fc�A�.*

lossȼ<�qW�       �	S�h3fc�A�.*

loss\X�<���       �	{ji3fc�A�.*

loss�ғ;��B�       �	.j3fc�A�.*

loss�=뮬0       �	ǻj3fc�A�.*

lossZ�<��       �	�bk3fc�A�.*

loss>�:Þ��       �	j�k3fc�A�.*

loss{�:1Gl       �	��l3fc�A�.*

loss|��;�̖       �	-#m3fc�A�.*

loss��;���       �	p�m3fc�A�.*

loss�h�<�#�N       �	�pn3fc�A�.*

lossa�9;�ʺ�       �	�^o3fc�A�.*

loss}3;J���       �	�o3fc�A�.*

loss�7=�xu/       �	��p3fc�A�.*

lossi�;�]�t       �	-q3fc�A�.*

losst�=�       �	t�q3fc�A�.*

loss��	<��6�       �	;mr3fc�A�.*

loss��/<{�X�       �	�s3fc�A�.*

loss���;o��`       �	L�s3fc�A�.*

losssoN:�7�       �	�bt3fc�A�.*

lossSU<|�!5       �	Au3fc�A�.*

loss��v;bb+�       �	�u3fc�A�.*

loss�t�9xmx       �	Vv3fc�A�.*

loss���<�K��       �	8�v3fc�A�.*

loss%i�8"�       �	�w3fc�A�.*

loss�q�:����       �	�$x3fc�A�.*

loss	!�;�֦       �	�y3fc�A�.*

loss3�:��L$       �	5�y3fc�A�.*

loss���:cAE       �	rRz3fc�A�.*

loss��;6�M       �	��z3fc�A�.*

lossV��8�(�       �	w�{3fc�A�.*

lossIo�7�iѯ       �	\|3fc�A�.*

loss��;�}��       �	Z�|3fc�A�.*

loss�/;�9�       �	�M}3fc�A�.*

loss� |;��Tr       �	��}3fc�A�.*

lossb�8Lx�G       �	��~3fc�A�.*

loss
�k<>��R       �	�$3fc�A�.*

lossة�<�Tr       �	F�3fc�A�.*

loss[�:����       �	�h�3fc�A�.*

loss �<����       �		�3fc�A�.*

lossq�<qP0       �	Ƣ�3fc�A�.*

loss�x�=��do       �	�>�3fc�A�.*

loss&��;�       �	�Ԃ3fc�A�.*

loss��3<ɋ��       �	3k�3fc�A�.*

loss���<%��6       �	� �3fc�A�.*

loss�|2;�(|       �	���3fc�A�.*

loss�V2:�L�D       �	�1�3fc�A�.*

loss���;)gψ       �	Tǅ3fc�A�.*

lossT��;͜'        �	#f�3fc�A�.*

loss�+?<����       �	���3fc�A�.*

lossԶ<ƐH�       �	���3fc�A�.*

loss�9�; �s@       �	0/�3fc�A�.*

loss�,=7_oA       �	�ʈ3fc�A�.*

loss���< �       �	g�3fc�A�.*

lossH��;��/       �	� �3fc�A�.*

loss]>�:]��d       �	���3fc�A�.*

loss?%<�ʔ        �	_)�3fc�A�.*

lossNI0;!��       �	Oʋ3fc�A�.*

losse9K<��h~       �	n�3fc�A�.*

loss���;��A�       �	��3fc�A�.*

loss��.<�b        �	���3fc�A�.*

loss��r:t��c       �	E�3fc�A�.*

loss�0;��       �	P�3fc�A�.*

loss@@P;>��i       �	cz�3fc�A�.*

lossOI�;62Tn       �	��3fc�A�.*

lossf��; ��       �	P��3fc�A�.*

lossG�<��D�       �	)B�3fc�A�.*

loss��<�~��       �	+ޑ3fc�A�.*

loss�;����       �	-w�3fc�A�.*

loss�F:�P�       �	��3fc�A�.*

lossr�x<r6΍       �	��3fc�A�.*

loss� #;���2       �	�L�3fc�A�.*

loss4�7<�N�6       �	��3fc�A�.*

loss$�; B       �	��3fc�A�.*

loss^��;jP       �	r��3fc�A�.*

loss�`}<��G       �	 W�3fc�A�.*

lossD��;Y�g       �	1D�3fc�A�.*

lossM��=�L       �	��3fc�A�.*

loss�a4:���[       �	��3fc�A�.*

lossX��:+�z�       �	���3fc�A�.*

loss�8<��>�       �	^+�3fc�A�.*

lossEs�<1��       �	�˜3fc�A�.*

loss ��;y�b@       �	Ym�3fc�A�.*

loss?ck;�8��       �	��3fc�A�.*

lossĠ=0��_       �	Ԟ�3fc�A�.*

loss�>'<��       �	���3fc�A�.*

lossIە<��j�       �	/4�3fc�A�.*

lossZO:�J�       �	Sʠ3fc�A�.*

lossr#;c�m       �	�_�3fc�A�.*

lossA�=n�1�       �	ߌ�3fc�A�.*

loss��=7��x       �	�ý3fc�A�.*

loss�\�<kߕ       �	�X�3fc�A�.*

lossT'�:#0y       �	a�3fc�A�.*

loss,(�<�k)       �	P4�3fc�A�.*

loss�G�;v��*       �	rS�3fc�A�.*

loss#j�<�B��       �	�-�3fc�A�.*

loss���<����       �	���3fc�A�.*

loss
j<E�4�       �	.r�3fc�A�.*

loss��J<��.�       �	�<�3fc�A�.*

loss��';^��x       �	?s�3fc�A�.*

loss�ܞ;ƃ�       �	�+�3fc�A�.*

loss�@+<���5       �	�l�3fc�A�.*

loss��< *4�       �	�6�3fc�A�.*

loss���<�祁       �	�M�3fc�A�.*

lossc��<z��       �	��3fc�A�/*

loss?;�9��       �	���3fc�A�/*

loss�^<{�       �	���3fc�A�/*

loss�S�<5TS        �	�s�3fc�A�/*

loss�c<��4G       �	��3fc�A�/*

lossë<�/S       �	!��3fc�A�/*

loss�4=��b       �	�D�3fc�A�/*

loss=�N:IXZN       �	���3fc�A�/*

loss[fn<�I�6       �	Sx�3fc�A�/*

loss� j<��Ɂ       �	s�3fc�A�/*

lossj��<3<�       �	ߥ�3fc�A�/*

lossM�v<���       �	T8�3fc�A�/*

loss�5#;�<�       �	k��3fc�A�/*

lossM&<���       �	�j�3fc�A�/*

loss�ۯ<��L       �	��3fc�A�/*

lossuu;
���       �	Û�3fc�A�/*

loss���;{�T�       �	]2�3fc�A�/*

lossE%,=��Hh       �	�i�3fc�A�/*

loss:��<t��       �	i �3fc�A�/*

loss�d�;�ћ       �	���3fc�A�/*

loss���="���       �	�7�3fc�A�/*

loss���:�\�s       �	���3fc�A�/*

lossK>F7�       �	�p�3fc�A�/*

loss!�g<�+N�       �	�	�3fc�A�/*

loss��<�5��       �	w��3fc�A�/*

lossf,K;U���       �	6�3fc�A�/*

loss��/=�c��       �	���3fc�A�/*

loss��;�߄<       �	�k�3fc�A�/*

lossnU�;i���       �	t�3fc�A�/*

lossfҁ;�h�       �	Ǜ�3fc�A�/*

losss(�;��@,       �	{0�3fc�A�/*

loss��=� d�       �	���3fc�A�/*

lossRA�<s��       �	\Y�3fc�A�/*

loss��:��گ       �	���3fc�A�/*

lossַ�<D@�        �	���3fc�A�/*

loss��;g���       �	�3fc�A�/*

loss�v�<*�<�       �	 ��3fc�A�/*

loss-�<�!�       �	�J�3fc�A�/*

lossq#�=?VL       �	��3fc�A�/*

loss:�<'�       �	�u�3fc�A�/*

loss�� ;0R �       �	��3fc�A�/*

lossp�;�1�       �	���3fc�A�/*

loss$�9��HA       �	�:�3fc�A�/*

loss�{;���        �	���3fc�A�/*

loss��$=�}       �	o�3fc�A�/*

loss �<�V*E       �	�3fc�A�/*

loss�ѥ;f�Z       �	|��3fc�A�/*

loss�۟:�:�_       �	o/�3fc�A�/*

loss�j�<��       �	/��3fc�A�/*

loss Z|;7�w�       �	F\�3fc�A�/*

loss�aB:�Bc       �	���3fc�A�/*

loss׫�;y�Ҹ       �	+��3fc�A�/*

lossJ==ؾ,h       �	2"�3fc�A�/*

loss�s�<��e       �	���3fc�A�/*

loss�c}=g�;�       �	�o�3fc�A�/*

loss!p�;z�       �	��3fc�A�/*

loss�W<�Θ�       �	)��3fc�A�/*

loss�\<��K�       �	�T�3fc�A�/*

loss���<yP��       �	E��3fc�A�/*

loss�Hp<���+       �	}��3fc�A�/*

loss�;��       �	f0�3fc�A�/*

loss��":��=       �	���3fc�A�/*

loss|�{<���^       �	�`�3fc�A�/*

loss}�<Iq?L       �	��3fc�A�/*

losss��<��v       �	%��3fc�A�/*

loss�G<-�}�       �	F�3fc�A�/*

loss�:`=�wm1       �	s��3fc�A�/*

loss��U<��       �	q�3fc�A�/*

loss`f�9"McZ       �	4�3fc�A�/*

loss�<P;�O�       �	G��3fc�A�/*

loss
�2;�ば       �	�I�3fc�A�/*

loss=��<��I�       �	~��3fc�A�/*

lossT;��%       �	2��3fc�A�/*

loss��|<���       �	�0�3fc�A�/*

loss���<1Μ       �	&��3fc�A�/*

loss���<�f       �	���3fc�A�/*

loss���<�|��       �	|�3fc�A�/*

losst��:�]
       �	G �3fc�A�/*

loss�(<�]C       �	��3fc�A�/*

loss@-!<���4       �	�U�3fc�A�/*

loss(�&;�H�       �	���3fc�A�/*

loss��=ptp       �	��3fc�A�/*

lossgX	:�y�~       �	�n�3fc�A�/*

loss�K�<�g,�       �	' 4fc�A�/*

loss�<IP�q       �	� 4fc�A�/*

loss�{<�`�u       �	'g4fc�A�/*

loss�_Q:�s�B       �	� 4fc�A�/*

loss[�b;�L�       �	*�4fc�A�/*

loss�˟;�<+�       �	�M4fc�A�/*

loss_�<aO/j       �	s�4fc�A�/*

loss�?=͍�3       �	�4fc�A�/*

lossE��:CL7       �	u:4fc�A�/*

lossm�<+���       �	_�4fc�A�/*

lossCQ-<��s       �	Ii4fc�A�/*

loss���;�O<�       �	� 4fc�A�/*

loss�9L=�+U�       �	Ș4fc�A�/*

loss��2;��       �	94fc�A�/*

lossss:<pI�C       �	�4fc�A�/*

lossH�:��f�       �	Hj	4fc�A�/*

lossW�5=�_�       �	�
4fc�A�/*

loss�k;���?       �	9�
4fc�A�/*

losseX�:9n9c       �	1C4fc�A�/*

loss�T&=CC�       �	��4fc�A�/*

loss�+<���       �	�4fc�A�/*

loss�#=���=       �	H4fc�A�/*

loss]�<��1�       �	��4fc�A�/*

loss�s=�uI`       �	MJ4fc�A�/*

loss���:�w��       �	�4fc�A�/*

loss.��;���       �	?q4fc�A�/*

lossq��:��6       �	�4fc�A�/*

loss�R�;.)�Z       �	��4fc�A�/*

loss֌f;���       �	�E4fc�A�/*

loss:'�<��h       �	A+4fc�A�/*

loss�~�;��d�       �	�4fc�A�/*

loss���<�Q,�       �	
J4fc�A�/*

loss��s<WP=       �	��4fc�A�/*

loss���<i~       �	�m4fc�A�/*

loss�%o:�,L       �	X 4fc�A�/*

lossT
�<A,�       �	�4fc�A�/*

loss}6<S� t       �	�04fc�A�/*

loss���:�/��       �	��4fc�A�/*

lossS�T;�1       �	�b4fc�A�/*

loss�l%=��}P       �	f�4fc�A�/*

loss�L�:ףb       �	��4fc�A�/*

loss��<P�f       �	814fc�A�/*

loss�<~<D��$       �	l�4fc�A�/*

loss��l=o�i�       �	�b4fc�A�/*

loss�<91       �	��4fc�A�/*

lossa �;�1%�       �	��4fc�A�0*

lossї"<�R��       �	O4fc�A�0*

loss�+<�%&_       �	k�4fc�A�0*

lossO�<v��       �	��4fc�A�0*

loss=��<�J�       �	F(4fc�A�0*

lossɹ2<�T�       �	��4fc�A�0*

loss�e�;��8�       �	l^4fc�A�0*

loss@<����       �	> 4fc�A�0*

losstD{;Z�       �	�� 4fc�A�0*

loss���;�o:d       �	BA!4fc�A�0*

loss6g�;Z@)�       �	��!4fc�A�0*

loss��/<~+6T       �	Ou"4fc�A�0*

lossl�f<�]�       �	�#4fc�A�0*

loss���;ږ��       �	��#4fc�A�0*

loss��I=�'8       �	^L$4fc�A�0*

lossJ=�!       �	m�$4fc�A�0*

loss.��;�r�       �	�%4fc�A�0*

loss�\=��@�       �	+&4fc�A�0*

loss��:26�       �	��&4fc�A�0*

loss��J<��       �	Qf'4fc�A�0*

loss�)�8��y       �	��'4fc�A�0*

lossڄ�;�T��       �	Q�(4fc�A�0*

loss̦}<��6       �	xB)4fc�A�0*

loss��z;�0�       �	��)4fc�A�0*

loss�/=}���       �	�p*4fc�A�0*

lossi�<�8V/       �	�	+4fc�A�0*

loss��<6���       �	��+4fc�A�0*

loss�-�<����       �	^0-4fc�A�0*

lossf�:9y��       �	��-4fc�A�0*

loss��;��}�       �	�a.4fc�A�0*

loss7�;��M�       �	��.4fc�A�0*

loss;92<n)�       �	)�/4fc�A�0*

loss���<��)5       �	�304fc�A�0*

loss�_�<-Bn�       �	�04fc�A�0*

loss@4P<��P       �	�o14fc�A�0*

loss��S<�H�       �	Z24fc�A�0*

loss拾;��'�       �	H�24fc�A�0*

loss��m9b���       �	u=34fc�A�0*

losst��<{�E�       �	��34fc�A�0*

loss�=;�w/�       �	�s44fc�A�0*

loss)�*<��       �	�54fc�A�0*

loss���;890y       �	m�54fc�A�0*

loss9l�;�ʬW       �	|G64fc�A�0*

loss.�=ܛ�       �	��64fc�A�0*

loss�;=(I,       �	�v74fc�A�0*

losspU�;���i       �	n84fc�A�0*

loss�<�<2!x       �	\�84fc�A�0*

loss���<��8`       �	�F94fc�A�0*

lossD�<�n�       �	��94fc�A�0*

loss��<'O��       �	�;4fc�A�0*

lossT0m;�K�       �	6�;4fc�A�0*

loss�?J;�kH       �	C=4fc�A�0*

loss��<Y�X'       �	uY>4fc�A�0*

loss��|=$B/�       �	?4fc�A�0*

losso�Y<ζ       �	�%@4fc�A�0*

loss�u�=j�/�       �	~�@4fc�A�0*

loss??�;m��Q       �	�B4fc�A�0*

loss�V1;���       �	�FC4fc�A�0*

loss�3�<Y��       �	�FD4fc�A�0*

loss�H�;C	�       �	h�D4fc�A�0*

loss�k�<����       �	��E4fc�A�0*

loss��<���0       �	p[F4fc�A�0*

lossQ}Y<��@/       �	dG4fc�A�0*

loss�
=�ʝ�       �	�FH4fc�A�0*

loss�Q�<��@       �	��H4fc�A�0*

loss�ǌ<�KA       �	�I4fc�A�0*

loss�l;�r       �	��J4fc�A�0*

loss�
=q�v�       �	��K4fc�A�0*

loss�j�;�\       �	�M4fc�A�0*

loss��M;���       �	G�M4fc�A�0*

loss��;C�eg       �	B�N4fc�A�0*

loss��<K�O�       �	,�O4fc�A�0*

loss��=���H       �	~�P4fc�A�0*

lossT�L=Б*�       �	{LQ4fc�A�0*

loss�M=���p       �	��Q4fc�A�0*

loss��]=x���       �	1�R4fc�A�0*

loss�Z<�<��       �	D6S4fc�A�0*

loss�R;E�@y       �	��S4fc�A�0*

loss��:)��+       �	�T4fc�A�0*

lossZ��;Y��       �	?�U4fc�A�0*

loss�B<O���       �	rSV4fc�A�0*

loss�b�;Љ<       �	��V4fc�A�0*

loss7u=5!ZM       �	��Y4fc�A�0*

lossK?"=u�KK       �	M�Z4fc�A�0*

loss��Q;�Ε       �	^0[4fc�A�0*

loss堯;fVbR       �	k�[4fc�A�0*

lossd1;��>)       �	�\4fc�A�0*

lossJ��:)~+       �	)!]4fc�A�0*

lossC�;b%��       �	v�]4fc�A�0*

lossD�!=`�U'       �	8h^4fc�A�0*

loss���;�9?       �	W_4fc�A�0*

loss�3�;q���       �	�_4fc�A�0*

lossSS�;����       �	�B`4fc�A�0*

loss!�<����       �	A�`4fc�A�0*

loss�T*={CSM       �	��a4fc�A�0*

loss�s=���       �	g,b4fc�A�0*

loss���:"�7       �	�b4fc�A�0*

loss;A;�t��       �	�fc4fc�A�0*

lossk��<���@       �	� d4fc�A�0*

loss�-\<8)-e       �	x�d4fc�A�0*

loss�(<�]       �	<1e4fc�A�0*

loss!c�<nN�D       �	��e4fc�A�0*

loss|��;�ò�       �	S[f4fc�A�0*

lossо;�W��       �	��f4fc�A�0*

loss�v�;�{�s       �	�g4fc�A�0*

loss���:D��Z       �	�-h4fc�A�0*

loss_Λ<w�"�       �	 �h4fc�A�0*

loss�s+<���       �	O]i4fc�A�0*

loss���<�c+o       �	��i4fc�A�0*

loss��<�       �	��j4fc�A�0*

lossh^�;����       �	�+k4fc�A�0*

lossq�>:�[G)       �	��k4fc�A�0*

losseʯ;��!       �	�>m4fc�A�0*

losspU�;Mܶ<       �	��m4fc�A�0*

loss2�=��v�       �	n�n4fc�A�0*

loss
>s;��W�       �	R*o4fc�A�0*

lossj��<1��       �	.�o4fc�A�0*

lossT�N:�       �	�mp4fc�A�0*

loss��<���!       �	�q4fc�A�0*

loss���;,h�       �	ȴq4fc�A�0*

lossX�=eK93       �	�Tr4fc�A�0*

loss�n6=:cG       �	��r4fc�A�0*

loss!�;�_�       �	��s4fc�A�0*

loss��;��4�       �	�3t4fc�A�0*

loss�a�<B��       �	��t4fc�A�0*

loss�7<`�g�       �	�eu4fc�A�0*

lossdu�;� 0#       �	C�u4fc�A�0*

loss�;5��d       �	��v4fc�A�0*

loss���=j��       �	�4w4fc�A�1*

loss��<�Я�       �	��w4fc�A�1*

loss�Q�<#|�       �	sx4fc�A�1*

loss�a=g�`       �	�y4fc�A�1*

loss*�<�Z��       �	T�y4fc�A�1*

loss��G<�Q�       �	[]z4fc�A�1*

loss��:H8��       �	ȗ{4fc�A�1*

loss:�q;���~       �	�8|4fc�A�1*

loss��(<3K�f       �	l�|4fc�A�1*

loss�::�^Q�       �	t�}4fc�A�1*

loss�ή<�8�	       �	80~4fc�A�1*

loss�`h<ͥ�[       �	�~4fc�A�1*

lossȉ�;P\��       �	�s4fc�A�1*

loss��#=jY[>       �	{�4fc�A�1*

loss��W=�냖       �	��4fc�A�1*

lossr�t<��       �	s��4fc�A�1*

loss��;���#       �	&6�4fc�A�1*

loss��K:Gs�I       �	�.�4fc�A�1*

lossA4;"b@e       �		Ń4fc�A�1*

loss��;-	�@       �	�]�4fc�A�1*

lossN�;Y��^       �	�4fc�A�1*

loss�IX:��O�       �	���4fc�A�1*

loss�;�:I�t�       �	L�4fc�A�1*

lossH0_;�P�       �	�z�4fc�A�1*

loss��><b�       �	%�4fc�A�1*

lossJ�<��       �	ȷ�4fc�A�1*

loss�'<����       �	U�4fc�A�1*

loss��<��       �	���4fc�A�1*

losszmG<���S       �		��4fc�A�1*

loss�SN;akS       �	��4fc�A�1*

loss]D!<�b�*       �	�4fc�A�1*

lossC/W<n�"�       �	��4fc�A�1*

loss��<!��1       �	���4fc�A�1*

loss��<�3�       �	�I�4fc�A�1*

loss>�<�_��       �	ߎ4fc�A�1*

loss��:_g.�       �	 s�4fc�A�1*

loss3�<��       �	1�4fc�A�1*

loss�,+=/;ZR       �	Q��4fc�A�1*

loss8I[;�$�       �	�<�4fc�A�1*

loss�r;g��6       �	5ӑ4fc�A�1*

loss�r;����       �	���4fc�A�1*

loss��<�͢       �	2�4fc�A�1*

lossHu�:�g�t       �	w֓4fc�A�1*

loss�9-��       �	0�4fc�A�1*

lossa}�;�e�       �	?��4fc�A�1*

loss-:Q�`       �	Z/�4fc�A�1*

loss�O�:�d&       �	�͖4fc�A�1*

loss�n<��5       �	�i�4fc�A�1*

lossߚ�;nIK�       �	q�4fc�A�1*

loss�=im�       �	�4fc�A�1*

loss�gp=���       �	���4fc�A�1*

loss�8=o�o�       �	9(�4fc�A�1*

loss�On;w�]�       �	���4fc�A�1*

loss��;���       �	�b�4fc�A�1*

loss�5�<n��       �	���4fc�A�1*

loss�4�;�
�       �	Ҩ�4fc�A�1*

loss`Y�<{�-y       �	TV�4fc�A�1*

loss�x�:�4]       �	���4fc�A�1*

loss_=9T)�       �	���4fc�A�1*

loss��#<ߚ�0       �	r5�4fc�A�1*

loss&\<��~       �	�ҟ4fc�A�1*

loss��:X�:       �	���4fc�A�1*

lossCճ:��G1       �	|�4fc�A�1*

loss���<I�A       �	!�4fc�A�1*

loss�t<d��&       �	���4fc�A�1*

loss��"=����       �	�V�4fc�A�1*

loss�;���       �	O�4fc�A�1*

loss�M<>T�V       �	c�4fc�A�1*

loss��Q:D�ʹ       �	s�4fc�A�1*

lossE��<�N��       �	۾�4fc�A�1*

loss_�;ɾF       �	.W�4fc�A�1*

loss�=ݗnH       �	j��4fc�A�1*

loss]�v:l���       �	獫4fc�A�1*

lossE��<��n�       �	�6�4fc�A�1*

loss�=��d�       �	{ܬ4fc�A�1*

loss�<dT�       �	�s�4fc�A�1*

loss���<{�]       �	��4fc�A�1*

lossʒ�<ğ�       �	ȱ�4fc�A�1*

loss#$=9�W       �	�L�4fc�A�1*

loss7]<�ܙ�       �	��4fc�A�1*

loss���;0��       �	k�4fc�A�1*

loss�(!=s��       �	U�4fc�A�1*

loss��<�+�       �	 ��4fc�A�1*

lossÃ�<����       �	�K�4fc�A�1*

loss�?=-�Uy       �	��4fc�A�1*

loss�^o;H       �	�y�4fc�A�1*

losso��;ܐ�       �	vP�4fc�A�1*

loss�<����       �	!�4fc�A�1*

loss��;�Vz       �	~�4fc�A�1*

loss�0�<�e;?       �	��4fc�A�1*

loss�NF<����       �	)��4fc�A�1*

loss�՜<N}v       �	�N�4fc�A�1*

lossL�$=�]��       �	.�4fc�A�1*

loss��<\�]�       �	�~�4fc�A�1*

lossl��;�l       �	��4fc�A�1*

loss�c�<�ꈶ       �	K��4fc�A�1*

lossl9�;���       �	�E�4fc�A�1*

loss�@�<|oJ�       �	e޺4fc�A�1*

loss�\�<�ŧ2       �	��4fc�A�1*

lossR�'<��       �	��4fc�A�1*

lossdg";����       �	8k�4fc�A�1*

loss�$�:�vK       �	���4fc�A�1*

lossg<yWQ       �	uV�4fc�A�1*

loss(�;��+       �	��4fc�A�1*

lossNKA=D6z�       �	ٔ�4fc�A�1*

loss[�I<�K\       �	t)�4fc�A�1*

loss��J=�:��       �	L��4fc�A�1*

loss#�<c��       �	f��4fc�A�1*

loss}(�<Z�\�       �	���4fc�A�1*

loss`x<|,k,       �	4/�4fc�A�1*

loss�#<-&�       �	w�4fc�A�1*

loss�%�<��       �	)��4fc�A�1*

loss�Xj;����       �	֫�4fc�A�1*

loss��:/,�j       �	�e�4fc�A�1*

loss�A"=���       �	�I�4fc�A�1*

loss�96;��B�       �	���4fc�A�1*

loss:�=�c��       �	���4fc�A�1*

loss��<*�0       �	k)�4fc�A�1*

loss�fB=�D�       �	���4fc�A�1*

loss|�m<����       �	]o�4fc�A�1*

lossM�J<.j�       �	K�4fc�A�1*

lossR�L<T��       �	���4fc�A�1*

loss/R	=_M��       �	�d�4fc�A�1*

loss=�<�U��       �	�C�4fc�A�1*

loss�D�<za|       �	���4fc�A�1*

lossiY;h"m�       �	׉�4fc�A�1*

loss��)<9M��       �	w1�4fc�A�1*

loss&<`��       �	���4fc�A�1*

loss�D<����       �	6w�4fc�A�2*

loss��<�	       �	��4fc�A�2*

losso��;�SR       �	��4fc�A�2*

lossoɅ;�ɯ�       �	�b�4fc�A�2*

loss��C;J^       �	X��4fc�A�2*

loss���<���c       �	��4fc�A�2*

loss��;�&j�       �	�>�4fc�A�2*

loss���;�O�!       �	��4fc�A�2*

loss#�<uS       �	z��4fc�A�2*

loss�ب;V
�=       �	!:�4fc�A�2*

lossM�X<��,Z       �	���4fc�A�2*

loss��<#���       �	��4fc�A�2*

lossl\=c���       �	�!�4fc�A�2*

loss  ;,6�"       �	^��4fc�A�2*

losss��<*irC       �	�]�4fc�A�2*

loss�'=E�        �	C��4fc�A�2*

loss�jC<�a;       �	�&�4fc�A�2*

lossù;YYW�       �	�.�4fc�A�2*

loss�)�:����       �	!��4fc�A�2*

lossT�V<�>�4       �	�d�4fc�A�2*

lossc��<{���       �	���4fc�A�2*

loss�Z<ź�       �	w��4fc�A�2*

loss�E�=��w_       �	VG�4fc�A�2*

loss ��:�7~$       �	��4fc�A�2*

loss��=���k       �	v��4fc�A�2*

loss8�=�.SW       �	�2�4fc�A�2*

lossC
�;��}       �	i��4fc�A�2*

lossf �;�;�       �	-�4fc�A�2*

loss3�'<lH��       �	���4fc�A�2*

lossZ�><���       �	v�4fc�A�2*

loss_��<��Q       �	�4fc�A�2*

loss�P=ťmb       �	��4fc�A�2*

loss`&<����       �	gG�4fc�A�2*

losst�V; Ӂ       �	z��4fc�A�2*

loss��T=�4�       �	'��4fc�A�2*

loss�Ա;�cn
       �	b-�4fc�A�2*

loss��@<c��       �	6��4fc�A�2*

loss��<��Ы       �	��4fc�A�2*

loss��Z;=��       �	�h�4fc�A�2*

loss_��<��̤       �	.�4fc�A�2*

loss�̻;K -�       �	x��4fc�A�2*

lossJ��<�vK       �	�0�4fc�A�2*

loss[�<َf�       �	G��4fc�A�2*

loss��!=�
��       �	h[�4fc�A�2*

loss�S=��       �	Q��4fc�A�2*

loss��<ٱ�       �		��4fc�A�2*

losst��;_��O       �	(�4fc�A�2*

loss���=5�       �	>��4fc�A�2*

loss$�"<��ى       �	Gr�4fc�A�2*

loss3�C=U���       �	��4fc�A�2*

loss�
E;�|�       �	���4fc�A�2*

lossa2=��"Y       �	�h�4fc�A�2*

loss��&<�!*       �	h�4fc�A�2*

loss�j�<J�L        �	��4fc�A�2*

loss��;;v	f�       �	�:�4fc�A�2*

loss-p�<�R7	       �	���4fc�A�2*

lossH�=�~5�       �	�z�4fc�A�2*

lossi@�:����       �	9a�4fc�A�2*

loss�;�}�       �	%�4fc�A�2*

loss��;2uĿ       �	~��4fc�A�2*

loss$�<����       �	G�4fc�A�2*

loss��:�L|�       �	��4fc�A�2*

loss��;q�]       �	���4fc�A�2*

loss ԅ;ʰX       �	m6�4fc�A�2*

lossֈc<~�i       �	]��4fc�A�2*

loss��=���       �	?��4fc�A�2*

loss�C�<;j       �	.�4fc�A�2*

loss��@<��,E       �	)��4fc�A�2*

lossQO�;6!x�       �	>y�4fc�A�2*

loss��L=3�,�       �	�  5fc�A�2*

loss���:%Ξ       �	u� 5fc�A�2*

loss�N=R1�       �	�p5fc�A�2*

loss�;�H�4       �	�5fc�A�2*

loss�;a�_       �	C�5fc�A�2*

loss�S=�ʪ*       �	wN5fc�A�2*

lossT��;���&       �	�5fc�A�2*

losssu;�ld       �	��5fc�A�2*

lossR�<�K��       �	|+5fc�A�2*

lossXi<	�X*       �	}�5fc�A�2*

loss_p�;�&E�       �	d5fc�A�2*

lossϴ�=���F       �	X�5fc�A�2*

loss��g<P��       �	��5fc�A�2*

loss�.F;^D�       �	65fc�A�2*

loss��;霊�       �	�5fc�A�2*

loss���:�B�        �	cd	5fc�A�2*

loss�A;�/.       �	��	5fc�A�2*

lossR&|<㣛       �	�
5fc�A�2*

loss�
�;Ò��       �	�65fc�A�2*

loss�o�<D,[�       �	��5fc�A�2*

loss�$2<��AK       �	5c5fc�A�2*

loss��;��G       �	H�5fc�A�2*

loss&2=�jQC       �	b�5fc�A�2*

lossa��;\.�       �	.95fc�A�2*

loss��;W�"C       �	U�5fc�A�2*

loss���<�`       �	�}5fc�A�2*

loss o�;�mS       �	405fc�A�2*

loss}�=Şi�       �	��5fc�A�2*

lossO�@=3���       �	I�5fc�A�2*

loss�w�;f��       �	�)5fc�A�2*

lossœ%<e6M       �	��5fc�A�2*

lossA"=hJWX       �	�j5fc�A�2*

lossۗ�<���       �	�5fc�A�2*

loss�T�;�o�       �	M�5fc�A�2*

loss&�g;K�߮       �	8M5fc�A�2*

lossP�";?�'�       �	|�5fc�A�2*

lossC͢;�       �	��5fc�A�2*

loss�fM<a��U       �	D15fc�A�2*

loss	<�h�       �	��5fc�A�2*

losso��:;8��       �	q5fc�A�2*

loss&!<�C�       �		5fc�A�2*

lossi��<� P�       �	��5fc�A�2*

lossG:��v       �	�L5fc�A�2*

lossM�f<��{�       �	p�5fc�A�2*

loss)�;,�{�       �	��5fc�A�2*

loss�0�=a�9�       �	�.5fc�A�2*

lossv�<}K̾       �	��5fc�A�2*

loss�?;�ZN�       �		m5fc�A�2*

lossJ$=���       �	�5fc�A�2*

loss޿;��	       �	Ǽ5fc�A�2*

lossrN;��k�       �	T5fc�A�2*

loss� �;�ȏV       �	��5fc�A�2*

loss���;�8P       �	Z� 5fc�A�2*

loss1X�<�!       �	�<!5fc�A�2*

loss�Ŧ:&3o       �	�!5fc�A�2*

loss�n<s�       �	�}"5fc�A�2*

lossL��<�q��       �	�$#5fc�A�2*

loss�x;��<       �	�#5fc�A�2*

lossMi>cb        �	�l$5fc�A�2*

loss�E=>
��       �	�
%5fc�A�3*

loss0!=<�!       �	�%5fc�A�3*

lossJKM<Q�N�       �	�P&5fc�A�3*

loss�x�;�v�       �	��&5fc�A�3*

loss�'�<�z�       �	��'5fc�A�3*

loss$�<!F}�       �	�*(5fc�A�3*

loss�S�8�u��       �	1�(5fc�A�3*

lossv�W;=�x!       �	�m)5fc�A�3*

loss��w9���"       �	k*5fc�A�3*

loss ��;H^*       �	��*5fc�A�3*

loss�;uϙ0       �	
K+5fc�A�3*

loss-r;���       �	��+5fc�A�3*

loss��<�eU       �	F},5fc�A�3*

loss�6�<�X��       �	z-5fc�A�3*

loss���9�^�T       �	J�-5fc�A�3*

loss�Q29j���       �	�J.5fc�A�3*

loss8�^;{F�       �	��.5fc�A�3*

loss��<<��       �	��/5fc�A�3*

loss��<0v�x       �	x%05fc�A�3*

loss�bK9�X       �	H�05fc�A�3*

loss��<�m�       �	p_15fc�A�3*

loss���=ϡ�       �	3�15fc�A�3*

loss��h;���       �	)�25fc�A�3*

loss���<iW�       �	Q/35fc�A�3*

loss_=~�        �	�35fc�A�3*

loss�3<eE.�       �	Ll45fc�A�3*

lossd'&<\P>       �	c55fc�A�3*

loss�)<�jY�       �	&�55fc�A�3*

loss�=�Q��       �	OX65fc�A�3*

lossH=�/�K       �	b�65fc�A�3*

loss?c�;0���       �	��75fc�A�3*

loss�"<Ҁ(�       �	}=85fc�A�3*

lossC� =56��       �	��85fc�A�3*

loss��<^��       �	��95fc�A�3*

loss1�j<Α<E       �	 %:5fc�A�3*

lossq=L<��%0       �	�:5fc�A�3*

lossX��<5���       �	)�;5fc�A�3*

loss�4(=��       �	i5<5fc�A�3*

loss�Y<
�0       �	��<5fc�A�3*

loss���;�}�k       �	�k=5fc�A�3*

loss��<6ݙ�       �	�>5fc�A�3*

lossk<;�k�       �	r�>5fc�A�3*

lossC��<�'��       �	�??5fc�A�3*

lossm7�;����       �	E�?5fc�A�3*

loss��(<byH|       �	�t@5fc�A�3*

loss��*:�f�       �	"�A5fc�A�3*

lossh�;U�Z�       �	�_B5fc�A�3*

loss�g;8��K       �	&�B5fc�A�3*

loss��#<Aj��       �	��C5fc�A�3*

loss@ɤ;��       �	m<D5fc�A�3*

loss���<H�y~       �	��D5fc�A�3*

lossHJ�;nx3�       �	YjF5fc�A�3*

loss�M�<�}�       �	`G5fc�A�3*

lossj`;���       �	��G5fc�A�3*

lossc(];x�O�       �	�7H5fc�A�3*

loss��;�q�       �	��H5fc�A�3*

losszZ�<("a�       �	jjI5fc�A�3*

loss��;�       �	+J5fc�A�3*

loss�A�<��        �	1�J5fc�A�3*

loss��-=��K�       �	�NK5fc�A�3*

loss6�<Lڥ�       �	(�K5fc�A�3*

loss�v�;\c�       �	��L5fc�A�3*

lossND;Yf�\       �	S%M5fc�A�3*

lossc��;��"       �	˿M5fc�A�3*

lossMzm<�n�=       �	SZN5fc�A�3*

loss�&N<�{�       �	J�N5fc�A�3*

loss���;����       �	a�O5fc�A�3*

loss��H=�r       �	�3P5fc�A�3*

loss;��<;�k       �	��P5fc�A�3*

lossy�;���Y       �	�Q5fc�A�3*

loss�5�;'�eU       �	�WR5fc�A�3*

lossx�6<ŝ��       �	��R5fc�A�3*

loss�Y:r�       �	f�S5fc�A�3*

loss�E�;�(��       �	��k5fc�A�3*

lossZwQ<�J;�       �	yl5fc�A�3*

loss�X=K��S       �	�tm5fc�A�3*

lossZ_{;0yF�       �	�n5fc�A�3*

loss��<~b�q       �	�n5fc�A�3*

loss��;�<��       �	h?o5fc�A�3*

loss��\<���       �	W]p5fc�A�3*

loss��Q<ީTK       �	�p5fc�A�3*

loss�q(=j`um       �	'�q5fc�A�3*

losszA1=��       �	�r5fc�A�3*

lossi�;����       �	^�r5fc�A�3*

loss��i=Aj�       �	��s5fc�A�3*

loss8N�;�ԅ       �	�Vt5fc�A�3*

loss�^N<B��       �	/�t5fc�A�3*

loss�^�<����       �	�u5fc�A�3*

loss�ZT<�E�%       �	I-v5fc�A�3*

lossW��:�l*�       �	C�v5fc�A�3*

loss�ϧ;���s       �	"�w5fc�A�3*

loss��<D�Q       �	A+x5fc�A�3*

loss ��<.�M�       �	/�x5fc�A�3*

loss�n;aan       �	Aay5fc�A�3*

loss}U}<T��       �	@�y5fc�A�3*

loss��d:��!�       �	Ŏz5fc�A�3*

loss���<�qX�       �	�+{5fc�A�3*

loss#�*<'A|       �	�|5fc�A�3*

loss��<q��%       �	�v}5fc�A�3*

lossE�l<.ua       �	�~5fc�A�3*

loss,�=[�       �	�~5fc�A�3*

lossL�<W�Zd       �	�`5fc�A�3*

loss���;݁       �	7�5fc�A�3*

loss��l<ͤb       �	~��5fc�A�3*

loss��U=�>Q:       �	�I�5fc�A�3*

lossd4�;�@��       �	���5fc�A�3*

lossA!�<ݡ
!       �	ę�5fc�A�3*

loss*��;�ͭ       �	:;�5fc�A�3*

loss���<��       �	�	�5fc�A�3*

lossW��;�ѩ9       �	צ�5fc�A�3*

loss3E�<��r�       �	F@�5fc�A�3*

loss��<�ݷ=       �	P�5fc�A�3*

loss�"�<�oJ       �	
��5fc�A�3*

lossd��;���       �	�"�5fc�A�3*

loss�fB<�ԁ�       �	�5fc�A�3*

loss�;8�|       �	�d�5fc�A�3*

loss�E<��Z�       �	� �5fc�A�3*

lossA�<6 <k       �	���5fc�A�3*

loss`X3;��_1       �	�7�5fc�A�3*

loss^Q<7y��       �	Bҋ5fc�A�3*

loss�1�;��       �	�n�5fc�A�3*

lossmp;��է       �	�	�5fc�A�3*

loss�(s<נ#p       �	좍5fc�A�3*

loss�X;��Ƭ       �	>>�5fc�A�3*

loss�GD<�R�       �	܎5fc�A�3*

losso�;�TNH       �	`s�5fc�A�3*

loss�?B={�aT       �	��5fc�A�3*

loss��+<���       �	O��5fc�A�3*

loss,�$;��nd       �	J�5fc�A�4*

loss��;Fg��       �	�Y�5fc�A�4*

loss!#i:�9�       �	���5fc�A�4*

lossߝ�;6��=       �	���5fc�A�4*

lossl�#=i��       �	b�5fc�A�4*

lossٚ�<����       �	�
�5fc�A�4*

loss�S�:���       �	��5fc�A�4*

loss?>:�5��       �	�>�5fc�A�4*

loss��5<#">7       �	�֖5fc�A�4*

loss���:�6�U       �	�j�5fc�A�4*

loss�Y<ˬ�P       �	�E�5fc�A�4*

loss	��<|�N       �	��5fc�A�4*

lossQ�P;n���       �	>u�5fc�A�4*

lossX�R<�Il       �	�s�5fc�A�4*

loss�#�;I���       �	�
�5fc�A�4*

loss6��:��^       �	��5fc�A�4*

loss-�-<	�o       �	 ^�5fc�A�4*

loss�5�=�Z�a       �	� �5fc�A�4*

loss��;P�{�       �	Y��5fc�A�4*

loss�;/��       �	WC�5fc�A�4*

lossJ��9��L       �	�ޞ5fc�A�4*

loss�[�:lH       �	ꕟ5fc�A�4*

loss�O:;�$�t       �	O<�5fc�A�4*

loss	̥;�� �       �	
֠5fc�A�4*

loss���;�e�       �	5�5fc�A�4*

loss>><��(       �	�%�5fc�A�4*

lossz��<�~�h       �	"¢5fc�A�4*

loss�"<�`�       �	�i�5fc�A�4*

loss_�<���       �	�5fc�A�4*

loss` ;�,�&       �	㩤5fc�A�4*

loss9h:\ -       �	WB�5fc�A�4*

lossX��<Cj��       �	nۥ5fc�A�4*

loss�T=;��       �	?q�5fc�A�4*

lossfA5;�rT       �	�5fc�A�4*

loss?-�<	��       �	+��5fc�A�4*

loss�z<]ш�       �	�:�5fc�A�4*

loss��;=M��       �	�Ԩ5fc�A�4*

lossr�?<h�*�       �	�s�5fc�A�4*

loss��K<�x=3       �	��5fc�A�4*

loss@�<]C�       �	\��5fc�A�4*

loss��:���,       �	�O�5fc�A�4*

lossj2=�F       �	��5fc�A�4*

lossN�{<n
��       �	֏�5fc�A�4*

loss��<[�0       �	���5fc�A�4*

lossj� ; rP�       �	�%�5fc�A�4*

loss�;=�x       �	r®5fc�A�4*

lossh%;z�f       �	 a�5fc�A�4*

loss���; �]�       �	���5fc�A�4*

loss
<�;���       �	ܹ�5fc�A�4*

lossE;$�ߝ       �	�b�5fc�A�4*

lossm�;dU1]       �	� �5fc�A�4*

loss�:p@��       �	���5fc�A�4*

loss;��: F|       �	+N�5fc�A�4*

loss�Yg=%��#       �	��5fc�A�4*

lossj׮:�jlE       �	D��5fc�A�4*

loss4��<�=�       �	C�5fc�A�4*

loss��;���       �	g��5fc�A�4*

loss��<Nl       �	��5fc�A�4*

lossfk�;Z��)       �	�A�5fc�A�4*

loss"F;��Y       �	�ٷ5fc�A�4*

lossdEd<��\       �	�r�5fc�A�4*

loss��G:�AL       �	W	�5fc�A�4*

loss?D�<�=�h       �	נ�5fc�A�4*

loss�B;��6       �	.8�5fc�A�4*

loss#\N<q���       �	�к5fc�A�4*

lossjT0<?Bğ       �	jg�5fc�A�4*

loss8�e;Sک+       �	���5fc�A�4*

loss��J:�k       �	��5fc�A�4*

loss��D;�       �	���5fc�A�4*

loss�8G:���       �	�S�5fc�A�4*

loss�iO<v���       �	�C�5fc�A�4*

loss
@n:��c!       �	&�5fc�A�4*

loss���<�L�M       �	l��5fc�A�4*

loss�<e��O       �	+L�5fc�A�4*

lossz-�<N���       �	���5fc�A�4*

loss~1�;Ȝ�       �	��5fc�A�4*

loss@�<$���       �	���5fc�A�4*

lossx�#:����       �	O�5fc�A�4*

lossq�N<nH1�       �	s��5fc�A�4*

loss{��<4)�       �	3��5fc�A�4*

loss�V=",�       �	H6�5fc�A�4*

lossE�;�c�B       �	���5fc�A�4*

losse?L<����       �	t{�5fc�A�4*

loss��"<\4�R       �	��5fc�A�4*

loss�uQ;98h       �	E��5fc�A�4*

loss�L/<��m�       �	:W�5fc�A�4*

lossMZn<q��@       �	���5fc�A�4*

loss G�;��
�       �	���5fc�A�4*

loss�؛;<�|       �	�9�5fc�A�4*

loss��C=(���       �	L��5fc�A�4*

loss�?.<g�1>       �	tz�5fc�A�4*

lossI�<����       �	7�5fc�A�4*

loss��<q}       �	���5fc�A�4*

loss�C<�L��       �	�a�5fc�A�4*

loss J<J�E       �	���5fc�A�4*

loss�=;E]VN       �	ɒ�5fc�A�4*

loss@C"<���       �	�5�5fc�A�4*

loss,�;=�r�       �	���5fc�A�4*

loss�=���       �	q �5fc�A�4*

loss�m<E�       �	���5fc�A�4*

loss��<��!       �	�O�5fc�A�4*

loss��&:�B�'       �	C��5fc�A�4*

loss��5=d�c�       �	φ�5fc�A�4*

loss�;O��U       �	��5fc�A�4*

losslL<���*       �	#��5fc�A�4*

loss��=p��       �	�W�5fc�A�4*

loss��<"R�r       �	R��5fc�A�4*

loss�ڶ;:�e       �	��5fc�A�4*

lossE��9֩��       �	DL�5fc�A�4*

loss���=���f       �	���5fc�A�4*

loss��<.��       �	 :�5fc�A�4*

loss�/;��o       �	�.�5fc�A�4*

loss�-=7�       �	q��5fc�A�4*

loss��;��       �	 `�5fc�A�4*

loss�X
<��[       �	��5fc�A�4*

loss,z:<i�b       �	��5fc�A�4*

loss�I:�l�       �	I�5fc�A�4*

loss���:� E�       �	D��5fc�A�4*

loss\4n;�>�]       �	Ks�5fc�A�4*

loss�+;��)       �	%�5fc�A�4*

losstА<m�A       �	���5fc�A�4*

loss�3�<e��A       �	�.�5fc�A�4*

loss��&<ks��       �	l�5fc�A�4*

loss�+�<e2�3       �	���5fc�A�4*

loss�^o9�5<_       �	�2�5fc�A�4*

loss�gH8���       �	���5fc�A�4*

lossRԥ;�jw�       �	c��5fc�A�4*

lossߛ�:���       �	��5fc�A�4*

lossT�f:���       �	8��5fc�A�5*

lossl��;ђ�       �	}[�5fc�A�5*

lossC��;r�       �	���5fc�A�5*

lossZC�<��Sp       �	Ւ�5fc�A�5*

loss�l�<�K\V       �	j4�5fc�A�5*

loss��<o�       �	�j�5fc�A�5*

loss:5=J��$       �	.��5fc�A�5*

loss1��<����       �	���5fc�A�5*

loss�Ơ<V��O       �	�8�5fc�A�5*

loss�Y�:!       �	H��5fc�A�5*

loss=�<�=�       �	��5fc�A�5*

lossy(�:�r       �	� �5fc�A�5*

loss!�<MG:U       �	Z��5fc�A�5*

loss�.a<��i�       �	Id�5fc�A�5*

loss�m{<~kO�       �	 �5fc�A�5*

loss��<����       �	e��5fc�A�5*

loss�"=(D(       �	}z�5fc�A�5*

lossE4�:9��       �	��5fc�A�5*

loss<���       �	:��5fc�A�5*

lossЧ=��       �	AG�5fc�A�5*

loss;`<Zr��       �	���5fc�A�5*

loss���;�vc       �	���5fc�A�5*

lossL%}:�jK       �	R&�5fc�A�5*

lossvK=��ؒ       �	s��5fc�A�5*

loss�}�<� x�       �	9^�5fc�A�5*

loss��<��v�       �	�B�5fc�A�5*

lossz�;��]       �	]��5fc�A�5*

loss��<\��h       �	�x�5fc�A�5*

loss���;���/       �	L�5fc�A�5*

loss��:�ϊ�       �	���5fc�A�5*

loss�I�<#`)3       �	 R�5fc�A�5*

lossĴ#<�+v       �	���5fc�A�5*

loss4�{=Kg�A       �	���5fc�A�5*

loss�h=��B,       �	�"�5fc�A�5*

loss.�=��v       �	n��5fc�A�5*

lossZlE=Ii��       �		��5fc�A�5*

loss�3�<dl:       �	Y��5fc�A�5*

loss�<L��       �	)Z�5fc�A�5*

loss��i:�)^       �	�@�5fc�A�5*

loss���:���       �	� 6fc�A�5*

loss1&<��؜       �	u� 6fc�A�5*

loss���;[}��       �	/m6fc�A�5*

loss�M�;�x|       �	�6fc�A�5*

loss���=�ׄS       �	#�6fc�A�5*

loss�h4<���e       �	T76fc�A�5*

lossos�:#��d       �	��6fc�A�5*

loss��Q;��#G       �	�g6fc�A�5*

loss��Y;�%4       �	? 6fc�A�5*

loss�1	<8G!       �	n�6fc�A�5*

lossH�r;n�       �	�I6fc�A�5*

lossS;m.�d       �	��6fc�A�5*

loss�: <Sc[�       �	�q6fc�A�5*

loss`	G;� E       �	%6fc�A�5*

loss���:��Hi       �	��6fc�A�5*

lossܵ>��v�       �	�9	6fc�A�5*

loss�G�<V��       �	��	6fc�A�5*

loss{�^:J��       �	��
6fc�A�5*

loss���;|$E`       �	�6fc�A�5*

loss�q[<��'       �	N_6fc�A�5*

loss�M<��ۼ       �	�6fc�A�5*

loss�=�2�       �	K�6fc�A�5*

loss��%<HK�       �	�X6fc�A�5*

loss<�c;��G�       �	� 6fc�A�5*

loss�ݕ;�|h�       �	��6fc�A�5*

loss�D;�4��       �	�86fc�A�5*

loss=��;�jR       �	��6fc�A�5*

loss�"<�,B�       �	��6fc�A�5*

loss���<�o�>       �	;R6fc�A�5*

loss�<�22K       �	S�6fc�A�5*

loss�2<��خ       �	��6fc�A�5*

lossn��;)�4       �	_�6fc�A�5*

loss�9�:��#k       �	�-6fc�A�5*

loss\�m;C�Q
       �	O�6fc�A�5*

loss)y�< 9�       �	8h6fc�A�5*

loss)E}<|��       �	6fc�A�5*

loss4�;etK�       �	��6fc�A�5*

loss�#<u�       �	@Q6fc�A�5*

loss&�_;�n��       �	Z�6fc�A�5*

loss2�);!�       �	Q06fc�A�5*

loss!�<W���       �	��6fc�A�5*

loss��B;�0��       �	V6fc�A�5*

loss_�&=��"�       �	�26fc�A�5*

lossXUw:%=�       �	��6fc�A�5*

loss�_�<�*m       �	~6fc�A�5*

loss;T1�I       �	�6fc�A�5*

loss� =�z�       �	��6fc�A�5*

lossC�;A�#q       �	W\ 6fc�A�5*

loss{ ;Pњ       �	�� 6fc�A�5*

lossLڞ:`[+T       �	F�!6fc�A�5*

loss[�7<�7�       �	�?"6fc�A�5*

lossIOV<\�       �	n�"6fc�A�5*

lossw��<�"��       �	�y#6fc�A�5*

loss߷;='_N�       �	�$6fc�A�5*

loss��;62)       �	��$6fc�A�5*

loss�!2;����       �	�U%6fc�A�5*

loss��;S�_       �	��%6fc�A�5*

lossh`h=J�!       �	f�&6fc�A�5*

lossio�;Ϟ�       �	�'6fc�A�5*

loss�A<ZT�M       �	��'6fc�A�5*

loss�G>;E^��       �	zV(6fc�A�5*

loss t<У��       �	�(6fc�A�5*

loss�v�<���       �	.�)6fc�A�5*

loss�L<�]~�       �	�;*6fc�A�5*

lossl�;s�]�       �		�*6fc�A�5*

loss�%=Qm�W       �	o�+6fc�A�5*

loss*�;ͧ�       �	7,6fc�A�5*

loss��:����       �	�,6fc�A�5*

lossrW�:�&�^       �	�j-6fc�A�5*

loss/�G<�R1�       �	,.6fc�A�5*

loss��0<�{       �	��.6fc�A�5*

lossC�;�i��       �	F/6fc�A�5*

loss�g=;Y@3�       �	��/6fc�A�5*

lossJ�</�y*       �	mt06fc�A�5*

loss|̣:�ˊ�       �	!16fc�A�5*

loss&�P<��?       �	Z�16fc�A�5*

loss�==�7�G       �	+M26fc�A�5*

loss�kw<#\��       �	��26fc�A�5*

loss��9=:>8�       �	7�36fc�A�5*

lossR;�;����       �	�"46fc�A�5*

loss� �<i�yY       �	��46fc�A�5*

lossJrN<p��       �	�T56fc�A�5*

lossb�=���*       �	o�56fc�A�5*

loss��w<���       �	҉66fc�A�5*

loss�=�;�=�       �	 76fc�A�5*

lossa!�<	�w       �	x�76fc�A�5*

loss���;�۶�       �	�U86fc�A�5*

loss_U:J��       �	W�86fc�A�5*

lossM�:�{o�       �	��96fc�A�5*

loss)�;�.��       �	t':6fc�A�6*

lossR�W<<U=d       �	�:6fc�A�6*

loss��9��[�       �	ms;6fc�A�6*

loss��p<d8�       �	?<6fc�A�6*

lossh@M;/�|       �	�<6fc�A�6*

loss��I;_�I�       �	kb=6fc�A�6*

lossvK=��|       �	�>6fc�A�6*

loss ��<y߮6       �	��>6fc�A�6*

loss��t<.Le�       �	%A?6fc�A�6*

loss�E�<ۊ��       �	��?6fc�A�6*

loss���<��!~       �	�~@6fc�A�6*

loss��;�Dȍ       �	A6fc�A�6*

loss�2;�T��       �	O�A6fc�A�6*

lossX�$<�;�       �	�FB6fc�A�6*

loss���=�%��       �	K�B6fc�A�6*

lossl�<e�)p       �	a�C6fc�A�6*

loss���<���.       �	>%D6fc�A�6*

loss	�;��pc       �	�D6fc�A�6*

loss�M�<��-z       �	zUE6fc�A�6*

loss�;��#�       �	��E6fc�A�6*

losss� =xhI       �	p|F6fc�A�6*

loss"�:�C�;       �	�G6fc�A�6*

loss�0�;��@       �	�G6fc�A�6*

lossMOC=e6.       �	ܝH6fc�A�6*

loss�y;��[       �	�7I6fc�A�6*

loss��<��.�       �	�I6fc�A�6*

loss�ߌ<2�       �	 tJ6fc�A�6*

loss�vK<��~f       �	t	K6fc�A�6*

loss`�i;�E8$       �	]�K6fc�A�6*

loss�<W:ν��       �	�<L6fc�A�6*

loss�"~;�!{       �	
�L6fc�A�6*

loss
L*;moϑ       �	�sM6fc�A�6*

loss�˺:���       �	�N6fc�A�6*

loss�PS=K�QO       �	��N6fc�A�6*

loss,��<�s�J       �	K:O6fc�A�6*

lossV��:.8}�       �	�O6fc�A�6*

lossi�N<eN�       �	CqP6fc�A�6*

loss�9<�x��       �	�Q6fc�A�6*

loss��c=��U       �	E�Q6fc�A�6*

loss<g!;B,��       �	�:R6fc�A�6*

loss6�R;M�       �	��R6fc�A�6*

loss�?;3�9J       �	>�S6fc�A�6*

losszC�:����       �	�aT6fc�A�6*

loss�';��       �	n�T6fc�A�6*

loss�<�fCo       �	\�U6fc�A�6*

loss\';�,�W       �	p(V6fc�A�6*

loss�6�;賄�       �	�V6fc�A�6*

loss�>�;�;       �	�VW6fc�A�6*

loss�~�<��       �	�X6fc�A�6*

losslB<����       �	��X6fc�A�6*

loss��s;)�       �	�XY6fc�A�6*

loss*&�<,�|       �	!Z6fc�A�6*

lossM>\=�ތ�       �	��Z6fc�A�6*

loss6G=bϖ	       �	�5[6fc�A�6*

loss�g�;	��n       �	+�[6fc�A�6*

loss쫂<aɚ�       �	�x\6fc�A�6*

lossJ�f<N�       �	�]6fc�A�6*

loss1��<�h��       �	�X^6fc�A�6*

loss�}�<6�.       �	��^6fc�A�6*

loss�M)<\�Z�       �	��_6fc�A�6*

loss#�$;����       �	A`6fc�A�6*

loss��x;���       �	P�`6fc�A�6*

loss��0=�<B       �	P�a6fc�A�6*

loss|�;�       �	B%b6fc�A�6*

loss;�c       �	t�b6fc�A�6*

loss�Z�<0�t�       �	isc6fc�A�6*

lossZ��<�5�,       �	�d6fc�A�6*

loss�ES<M�;�       �	h�d6fc�A�6*

loss�d <��rS       �	Oe6fc�A�6*

loss[��;x�|�       �	-�e6fc�A�6*

loss���;[X��       �	n�f6fc�A�6*

loss�z=��       �	�g6fc�A�6*

lossV�";�n�J       �	��g6fc�A�6*

loss�j9;�j       �	�`h6fc�A�6*

loss�O	</�b       �	X�h6fc�A�6*

lossNW<�%T       �	�i6fc�A�6*

loss�
�:aӒx       �	�7j6fc�A�6*

loss��G; �֣       �	��j6fc�A�6*

losshp�<��W*       �	�nk6fc�A�6*

loss���;Y���       �	$
l6fc�A�6*

loss��;ނ4       �	7�l6fc�A�6*

loss��;��{       �	>>m6fc�A�6*

loss�{�<G*'       �	�n6fc�A�6*

loss�s<�䌟       �	��n6fc�A�6*

loss���<�%�1       �	�oo6fc�A�6*

lossMa)9Z6��       �	�
p6fc�A�6*

lossn��:N�נ       �	��p6fc�A�6*

loss���;ѧ�       �	vOq6fc�A�6*

loss&�;;|:�       �	��q6fc�A�6*

loss$�.;�/c�       �	~r6fc�A�6*

loss@�$=Rք�       �	js6fc�A�6*

loss1�;�w�q       �	5�s6fc�A�6*

loss���;����       �	]St6fc�A�6*

loss��<�E��       �	��t6fc�A�6*

lossJ�c<�}�       �	�u6fc�A�6*

losswb<�#�,       �	�=v6fc�A�6*

loss��;J�u       �	�v6fc�A�6*

lossQ��;���       �	�w6fc�A�6*

loss�q�;-��       �	.x6fc�A�6*

loss�QR=(�4�       �	7�x6fc�A�6*

lossܑ<�n�6       �	�gy6fc�A�6*

lossҏ�<�L��       �	�z6fc�A�6*

loss��	<���k       �	жz6fc�A�6*

loss��<T��       �	V{6fc�A�6*

loss#}k=�/��       �	 �{6fc�A�6*

loss��[;��uD       �	�Q}6fc�A�6*

lossc��;�IP�       �	��}6fc�A�6*

loss9́<[���       �	��~6fc�A�6*

loss���=J�       �	�C6fc�A�6*

loss��<h�L       �	��6fc�A�6*

loss���</�h       �	��6fc�A�6*

loss�3�<Y�R       �	�D�6fc�A�6*

loss��j=�P?�       �	�ށ6fc�A�6*

loss>)�<vC�H       �	㋂6fc�A�6*

loss]	1=��       �	s,�6fc�A�6*

loss�̴;�B       �	�Ã6fc�A�6*

lossi,Y<f�g       �	yX�6fc�A�6*

lossЭ�:��a�       �	��6fc�A�6*

loss�O�;_��a       �	���6fc�A�6*

loss���<��'       �	��6fc�A�6*

loss!�<�*�H       �	J��6fc�A�6*

lossX+�:�I��       �	OZ�6fc�A�6*

loss�L<k<*       �	U��6fc�A�6*

loss��<�<%8       �	���6fc�A�6*

loss� �<�G�       �	�0�6fc�A�6*

lossj��<�F7�       �	!�6fc�A�6*

loss[jA;���       �	��6fc�A�6*

loss��<Քd�       �	�W�6fc�A�6*

lossi�9<_h       �	g�6fc�A�7*

loss̍<2 2m       �	i��6fc�A�7*

loss�Қ<�A\        �	�-�6fc�A�7*

loss�l�<��       �	p͍6fc�A�7*

loss?�a<9$+       �	�m�6fc�A�7*

loss�L<�Om�       �	[�6fc�A�7*

loss�@�;ɯ|d       �	8��6fc�A�7*

loss���<%��K       �	]6�6fc�A�7*

loss���;Qr�k       �	VҐ6fc�A�7*

lossWx�<�:�k       �	���6fc�A�7*

lossڶ�<�)��       �	S@�6fc�A�7*

loss�_�<���       �	Dے6fc�A�7*

losso�<�Y*       �	�w�6fc�A�7*

lossyP<ۗ�W       �	��6fc�A�7*

loss��;�~�       �	�ޔ6fc�A�7*

loss8qx=�ɞ       �	���6fc�A�7*

loss!��;Q���       �	�9�6fc�A�7*

loss�;CO�       �	�ݖ6fc�A�7*

lossC��;|ֆ-       �	.��6fc�A�7*

loss�&;�D�0       �	V��6fc�A�7*

loss�$�;*Tk�       �	�I�6fc�A�7*

loss{�<-�       �	0�6fc�A�7*

lossq�T<L N�       �	���6fc�A�7*

loss��:@(�       �	 C�6fc�A�7*

loss�*=:��%       �	W�6fc�A�7*

loss|,;VG       �	�$�6fc�A�7*

loss/<��U       �	�ڝ6fc�A�7*

loss�;)�7�       �	Ou�6fc�A�7*

loss��;y        �	�6fc�A�7*

loss4gB=��       �	�6fc�A�7*

lossxg�9�t       �	w�6fc�A�7*

loss��<�N�h       �	�!�6fc�A�7*

loss�x;<JR6�       �	�â6fc�A�7*

loss�&�:�T0       �	)]�6fc�A�7*

lossr�;<�M�Z       �	���6fc�A�7*

loss��!<��3       �	 ��6fc�A�7*

lossH3;�ƘZ       �	a7�6fc�A�7*

lossh�;Ac��       �	_ҥ6fc�A�7*

loss��[<�w�       �	ܛ�6fc�A�7*

loss�;<ߜo�       �	�7�6fc�A�7*

lossh��;��A�       �	�ѧ6fc�A�7*

loss�=���       �	Dl�6fc�A�7*

loss���;Pn�       �	��6fc�A�7*

loss�X�;%l��       �	A��6fc�A�7*

lossh�:��q       �	�<�6fc�A�7*

loss^�<%�	       �	٪6fc�A�7*

loss E$<��ă       �	�s�6fc�A�7*

loss|$�;��Ρ       �	��6fc�A�7*

lossQ]�<���       �	蠬6fc�A�7*

loss��7<k�<|       �	L5�6fc�A�7*

lossU�;����       �	�έ6fc�A�7*

loss���<�s�:       �	d�6fc�A�7*

loss��<B���       �	��6fc�A�7*

loss���;íe�       �	��6fc�A�7*

loss&�;��?       �	5�6fc�A�7*

losst�;�-*w       �	�Ӱ6fc�A�7*

loss�?=$�	       �	�j�6fc�A�7*

lossܕ�<)=t�       �	1�6fc�A�7*

lossF�b<�T�
       �	���6fc�A�7*

loss��N<��d�       �	Jϳ6fc�A�7*

lossV�;��A�       �	-&�6fc�A�7*

lossl42<�:�       �	�6fc�A�7*

loss�6�:���       �	���6fc�A�7*

loss�i�;�W�       �	Ae�6fc�A�7*

loss#2�;�@�       �	�6fc�A�7*

lossIq�;�j       �	�<�6fc�A�7*

loss��<{`%�       �	��6fc�A�7*

loss|��<��       �	���6fc�A�7*

lossE��<����       �	O�6fc�A�7*

loss}��<�Z`Z       �	���6fc�A�7*

loss=A\<��b\       �	�b�6fc�A�7*

loss�;r7Y�       �	v��6fc�A�7*

loss*�x<�6v�       �	�0�6fc�A�7*

lossċ`;��q       �	�߿6fc�A�7*

lossc<s��       �	��6fc�A�7*

loss�t�<���       �	T��6fc�A�7*

lossRQ�;��       �	�G�6fc�A�7*

loss��<�d��       �	��6fc�A�7*

loss��b;�v�S       �	&:�6fc�A�7*

lossȡS:-LQ�       �	���6fc�A�7*

loss(";��q�       �	%x�6fc�A�7*

loss�7;v��       �	��6fc�A�7*

loss���<48
(       �	nN�6fc�A�7*

loss��9�Yʫ       �	'�6fc�A�7*

loss��v<c��       �	L��6fc�A�7*

lossc �<���       �	F��6fc�A�7*

loss��};B�M/       �	���6fc�A�7*

loss��<s{�       �	^��6fc�A�7*

loss?hg<�+�'       �	_`�6fc�A�7*

loss`�<���       �	�6fc�A�7*

loss�);���       �	ZH�6fc�A�7*

loss���;�/��       �	���6fc�A�7*

lossQ�<��o7       �	���6fc�A�7*

lossR%�9k�p.       �	�|�6fc�A�7*

loss�y�8PyV       �	�6fc�A�7*

loss&O�:u��        �	���6fc�A�7*

lossie;�Xz       �	���6fc�A�7*

loss�g�;S�y�       �	�n�6fc�A�7*

loss#��;0���       �	�9�6fc�A�7*

loss@�:eի�       �	�W�6fc�A�7*

loss���:�ժ�       �	f-�6fc�A�7*

loss.4n<���       �	>��6fc�A�7*

lossj��8n& �       �	Qj�6fc�A�7*

loss��7��G       �	�%�6fc�A�7*

loss̰T;e>��       �	���6fc�A�7*

loss�9�;��Yb       �	#��6fc�A�7*

lossJa;{���       �	{��6fc�A�7*

lossz�-9���       �	���6fc�A�7*

loss�L<Dk7�       �	@��6fc�A�7*

loss��Z=���d       �	�\�6fc�A�7*

loss�;�       �	N��6fc�A�7*

loss4�<3k�h       �	l=�6fc�A�7*

loss�>�;Y���       �	�A�6fc�A�7*

loss,j�<�)6�       �	@��6fc�A�7*

loss��y<���       �	��6fc�A�7*

loss@�;`��       �	�M�6fc�A�7*

loss���;���       �	~�6fc�A�7*

lossXi�:E�Q�       �	.��6fc�A�7*

loss_m;g]�       �	�i�6fc�A�7*

loss�Q�<�[��       �	�a�6fc�A�7*

loss�[b;�D�       �	��6fc�A�7*

loss_S�=�6g�       �	���6fc�A�7*

loss��<�h��       �	5��6fc�A�7*

losst�+:���       �	;��6fc�A�7*

loss�e=�-G       �	�i�6fc�A�7*

loss�&<��2S       �	�1�6fc�A�7*

loss`�N;&�*�       �	X��6fc�A�7*

loss�k;S��"       �	���6fc�A�7*

loss���:�`�       �	���6fc�A�8*

loss��&=cv       �	{��6fc�A�8*

loss��;���       �	*��6fc�A�8*

loss�K�:WM�6       �	G��6fc�A�8*

loss��<���       �	���6fc�A�8*

lossv�N:��D       �	�7�6fc�A�8*

loss�1d:p��g       �	���6fc�A�8*

loss�j�;�9       �	oc�6fc�A�8*

loss��;�P��       �	X��6fc�A�8*

loss%�E<��s       �	o��6fc�A�8*

loss��?<h�bF       �	j3�6fc�A�8*

loss���;�Q�       �	���6fc�A�8*

loss��;��K       �	�f�6fc�A�8*

loss�[<�xq�       �	  �6fc�A�8*

lossq=/䩰       �	_��6fc�A�8*

lossѡ�<���       �	�:�6fc�A�8*

losst%<�*_       �	���6fc�A�8*

loss>�;�L��       �	}x�6fc�A�8*

loss�>�;��ܵ       �	.�6fc�A�8*

loss���<�b��       �	���6fc�A�8*

loss���;Vs�!       �	�c�6fc�A�8*

loss�g�<�I��       �	�6fc�A�8*

loss��j:��       �	��6fc�A�8*

loss(�;КAf       �	��6fc�A�8*

losst<����       �	j��6fc�A�8*

loss�<�`+       �	|e 7fc�A�8*

loss_A<��X�       �	G7fc�A�8*

lossg;��>�       �	hB7fc�A�8*

loss���;k�d�       �	X�7fc�A�8*

lossVF�;)jx�       �	�7fc�A�8*

lossA�<ʟ�%       �	�7fc�A�8*

lossB��:��#I       �	�B7fc�A�8*

loss�1F;����       �	�L7fc�A�8*

loss��G;�A��