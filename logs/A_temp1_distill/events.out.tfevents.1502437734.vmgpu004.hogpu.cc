       �K"	  �YXc�Abrain.Event:2�dn.#�     h�k�	d�YXc�A"��
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
dtype0*
shape: 
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
conv2d_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�x�
`
conv2d_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�x=
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@*
seed2���
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
conv2d_1/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_1/bias*
T0*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
s
conv2d_1/convolution/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�\1�
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2���
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
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
[
conv2d_2/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_2/bias
VariableV2*
shared_name *
dtype0*
shape:@*
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
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
s
conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
paddingVALID*
T0*
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
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
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
seed2��*
dtype0*
T0*
seed���)
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
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
_output_shapes
:*
out_type0*
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
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
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
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2���
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
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
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
VariableV2*
shared_name *
dtype0*
shape:�*
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*(
_output_shapes
:����������*
T0

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
_output_shapes
:*
out_type0*
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2��
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
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
T0*
N**
_output_shapes
:����������: 
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"�   
   
_
dense_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U�
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
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

q
dense_2/bias/readIdentitydense_2/bias*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
:���������@*
data_formatNHWC*
T0
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
!sequential_1/dropout_1/cond/mul/yConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu*
T0
�
sequential_1/dropout_1/cond/mulMul(sequential_1/dropout_1/cond/mul/Switch:1!sequential_1/dropout_1/cond/mul/y*/
_output_shapes
:���������@*
T0
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
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2���
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
*sequential_1/flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
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
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
transpose_b( *(
_output_shapes
:����������*
transpose_a( *
T0
�
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
data_formatNHWC*
T0
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
sequential_1/dropout_2/cond/mulMul(sequential_1/dropout_2/cond/mul/Switch:1!sequential_1/dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������
�
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
�
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
T0*
out_type0*
_output_shapes
:
�
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
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
seed2�Ȇ*
T0*
seed���)*
dtype0
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
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
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
VariableV2*
shared_name *
dtype0*
shape: *
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
num_inst/readIdentitynum_inst*
_output_shapes
: *
_class
loc:@num_inst*
T0
^
num_correct/initial_valueConst*
_output_shapes
: *
dtype0*
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
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
T0*
_output_shapes
: *
_class
loc:@num_correct
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
 *    *
dtype0*
_output_shapes
: 
�
Assign_1Assignnum_correctConst_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_correct
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
softmax_cross_entropy_loss/RankConst*
_output_shapes
: *
dtype0*
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
_output_shapes
:*
out_type0*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
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
dtype0*
_output_shapes
:*
valueB:
���������
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
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
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
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
N*
T0*
_output_shapes
:*

axis 
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
"softmax_cross_entropy_loss/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
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
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
_output_shapes
:*
N*

axis *
T0
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
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
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
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
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
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
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
]
PlaceholderPlaceholder*'
_output_shapes
:���������
*
dtype0*
shape: 
L
div_2/yConst*
valueB
 *  �?*
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
"softmax_cross_entropy_loss_1/ShapeShapediv_2*
_output_shapes
:*
out_type0*
T0
e
#softmax_cross_entropy_loss_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
i
$softmax_cross_entropy_loss_1/Shape_1Shapediv_2*
_output_shapes
:*
out_type0*
T0
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
(softmax_cross_entropy_loss_1/Slice/beginPack softmax_cross_entropy_loss_1/Sub*

axis *
_output_shapes
:*
T0*
N
q
'softmax_cross_entropy_loss_1/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
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
value	B : *
_output_shapes
: *
dtype0
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
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
$softmax_cross_entropy_loss_1/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
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
)softmax_cross_entropy_loss_1/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
$softmax_cross_entropy_loss_1/Slice_1Slice$softmax_cross_entropy_loss_1/Shape_2*softmax_cross_entropy_loss_1/Slice_1/begin)softmax_cross_entropy_loss_1/Slice_1/size*
_output_shapes
:*
Index0*
T0
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
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
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
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
)softmax_cross_entropy_loss_1/Slice_2/sizePack"softmax_cross_entropy_loss_1/Sub_2*
_output_shapes
:*
N*

axis *
T0
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
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
=softmax_cross_entropy_loss_1/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
U
Msoftmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_successNoOp
�
(softmax_cross_entropy_loss_1/ToFloat_1/xConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
 softmax_cross_entropy_loss_1/MulMul&softmax_cross_entropy_loss_1/Reshape_2(softmax_cross_entropy_loss_1/ToFloat_1/x*#
_output_shapes
:���������*
T0
�
"softmax_cross_entropy_loss_1/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
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
8softmax_cross_entropy_loss_1/num_present/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
]softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShape&softmax_cross_entropy_loss_1/Reshape_2N^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
[softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
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
:softmax_cross_entropy_loss_1/num_present/broadcast_weightsMul/softmax_cross_entropy_loss_1/num_present/SelectDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
.softmax_cross_entropy_loss_1/num_present/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
(softmax_cross_entropy_loss_1/num_presentSum:softmax_cross_entropy_loss_1/num_present/broadcast_weights.softmax_cross_entropy_loss_1/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
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
,softmax_cross_entropy_loss_1/ones_like/ConstConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
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
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
Bgradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_depsNoOp9^gradients/softmax_cross_entropy_loss_1/value_grad/Select;^gradients/softmax_cross_entropy_loss_1/value_grad/Select_1
�
Jgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependencyIdentity8gradients/softmax_cross_entropy_loss_1/value_grad/SelectC^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select
�
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
_output_shapes
: *M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
@gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/div_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1*
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
T0*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select
�
Mgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Identity;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1D^gradients/softmax_cross_entropy_loss_1/Select_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1*
_output_shapes
: 
�
?gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
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
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*
_output_shapes
: *
T0*

Tmultiples0
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
T0*
_output_shapes
:*
out_type0
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
?gradients/softmax_cross_entropy_loss_1/num_present_grad/ReshapeReshapeMgradients/softmax_cross_entropy_loss_1/Select_grad/tuple/control_dependency_1Egradients/softmax_cross_entropy_loss_1/num_present_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
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
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
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
9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum_17gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: *
T0
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
T0*
_output_shapes
: *
Tshape0
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
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
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
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
_output_shapes
:*
out_type0*
T0
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
Dgradients/softmax_cross_entropy_loss_1/xentropy_grad/PreventGradientPreventGradient'softmax_cross_entropy_loss_1/xentropy:1*
T0*0
_output_shapes
:������������������
�
Cgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
?gradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims
ExpandDims=gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_loss_1/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
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
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_2_grad/ShapeShapesequential_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
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
-gradients/div_2_grad/tuple/control_dependencyIdentitygradients/div_2_grad/Reshape&^gradients/div_2_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_2_grad/Reshape*
T0
�
/gradients/div_2_grad/tuple/control_dependency_1Identitygradients/div_2_grad/Reshape_1&^gradients/div_2_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/div_2_grad/Reshape_1
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_2_grad/tuple/control_dependency*
data_formatNHWC*
T0*
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
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
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
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
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
T0*
N**
_output_shapes
:����������: 
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_2/cond/dropout/Floor*
T0*
_output_shapes
:*
out_type0
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_2/cond/mul_grad/ShapeShape(sequential_1/dropout_2/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0
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
8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1Reshape4gradients/sequential_1/dropout_2/cond/mul_grad/Sum_16gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_depsNoOp7^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape9^gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
�
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape
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
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*(
_output_shapes
:����������*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad*
T0
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
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad*(
_output_shapes
:����������*
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
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
Qgradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
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
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
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
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
T0*
_output_shapes
: *
Tshape0
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
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
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
Egradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltersequential_1/activation_1/Relu8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@@*
data_formatNHWC*
strides
*
T0*
paddingVALID
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
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
out_type0*
_output_shapes
:*
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
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta2_power/readIdentitybeta2_power*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
T0
j
zerosConst*&
_output_shapes
:@*
dtype0*%
valueB@*    
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
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
	container *
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
T
zeros_7Const*
_output_shapes
:@*
dtype0*
valueB@*    
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
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
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
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
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
	container *
shared_name *
dtype0*
shape:�*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
Adam/mul_1*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
: "4�A��,     ��#�	ka�YXc�AJ��
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
dtype0*
shape: 
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
dtype0*
shape: 
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
seed2���*
dtype0*
T0*
seed���)
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
VariableV2*
shared_name *
dtype0*
shape:@*&
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
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*    
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
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
:���������@*
data_formatNHWC*
T0
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
:@@*
seed2���*
T0*
seed���)*
dtype0
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
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
_output_shapes
:@*
validate_shape(* 
_class
loc:@conv2d_2/bias*
T0*
use_locking(
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
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
n
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
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
:���������@*
seed2��*
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
dropout_1/cond/dropout/mulMuldropout_1/cond/dropout/divdropout_1/cond/dropout/Floor*/
_output_shapes
:���������@*
T0
�
dropout_1/cond/Switch_1Switchactivation_2/Reludropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu
�
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*
T0*
N*1
_output_shapes
:���������@: 
c
flatten_1/ShapeShapedropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
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
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
_output_shapes
:*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
Index0*
T0
Y
flatten_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
T0
m
dense_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB" d  �   
_
dense_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�3z�
_
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
dtype0*
T0*
seed���)
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0

dropout_2/cond/mulMuldropout_2/cond/mul/Switch:1dropout_2/cond/mul/y*
T0*(
_output_shapes
:����������

 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
n
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
out_type0*
_output_shapes
:*
T0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
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
seed2��*
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
N*
T0
m
dense_2/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   
   
_
dense_2/random_uniform/minConst*
valueB
 *̈́U�*
dtype0*
_output_shapes
: 
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
seed2���*
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
	container *
dtype0*
shared_name *
shape:	�

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
dense_2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_2/bias
VariableV2*
_output_shapes
:
*
	container *
dtype0*
shared_name *
shape:

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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
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
dtype0*
_output_shapes
:*
valueB"      
�
!sequential_1/conv2d_1/convolutionConv2Ddataconv2d_1/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
T0
�
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*
T0*/
_output_shapes
:���������@
�
'sequential_1/conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
�
/sequential_1/conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
seed2���*
dtype0*
T0*
seed���)
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
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
_output_shapes
:*
out_type0*
T0
t
*sequential_1/flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
sequential_1/flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 
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
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
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
#sequential_1/dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
�
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
-sequential_1/dropout_2/cond/dropout/keep_probConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
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
6sequential_1/dropout_2/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2�Ȇ
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
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul*
N*
T0**
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
*
data_formatNHWC*
T0
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
VariableV2*
shared_name *
dtype0*
shape: *
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
T0*
_output_shapes
: *
_class
loc:@num_correct
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
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@num_inst
~
AssignAdd_1	AssignAddnum_correctSum*
_output_shapes
: *
_class
loc:@num_correct*
T0*
use_locking( 
L
Const_2Const*
dtype0*
_output_shapes
: *
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
 *  �?*
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
value	B :*
_output_shapes
: *
dtype0
g
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
T0*
_output_shapes
:*
out_type0
b
 softmax_cross_entropy_loss/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
_output_shapes
:*
N*

axis *
T0
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
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
c
!softmax_cross_entropy_loss/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
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
N*
T0*
_output_shapes
:*

axis 
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
(softmax_cross_entropy_loss/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
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
"softmax_cross_entropy_loss/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
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
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
_output_shapes
:*
N*

axis *
T0
�
"softmax_cross_entropy_loss/Slice_2Slice softmax_cross_entropy_loss/Shape(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:���������
�
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*#
_output_shapes
:���������*
Tshape0*
T0
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
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
_output_shapes
:*
out_type0*
T0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:���������
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB: 
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
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
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
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
dtype0*
shape: 
L
div_2/yConst*
valueB
 *  �?*
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
'softmax_cross_entropy_loss_1/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
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
(softmax_cross_entropy_loss_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
#softmax_cross_entropy_loss_1/concatConcatV2,softmax_cross_entropy_loss_1/concat/values_0"softmax_cross_entropy_loss_1/Slice(softmax_cross_entropy_loss_1/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
$softmax_cross_entropy_loss_1/ReshapeReshapediv_2#softmax_cross_entropy_loss_1/concat*
T0*0
_output_shapes
:������������������*
Tshape0
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
$softmax_cross_entropy_loss_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
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
���������*
_output_shapes
:*
dtype0
l
*softmax_cross_entropy_loss_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
%softmax_cross_entropy_loss_1/concat_1ConcatV2.softmax_cross_entropy_loss_1/concat_1/values_0$softmax_cross_entropy_loss_1/Slice_1*softmax_cross_entropy_loss_1/concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
&softmax_cross_entropy_loss_1/Reshape_1ReshapePlaceholder%softmax_cross_entropy_loss_1/concat_1*
Tshape0*0
_output_shapes
:������������������*
T0
�
%softmax_cross_entropy_loss_1/xentropySoftmaxCrossEntropyWithLogits$softmax_cross_entropy_loss_1/Reshape&softmax_cross_entropy_loss_1/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
f
$softmax_cross_entropy_loss_1/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
"softmax_cross_entropy_loss_1/Sub_2Sub!softmax_cross_entropy_loss_1/Rank$softmax_cross_entropy_loss_1/Sub_2/y*
_output_shapes
: *
T0
t
*softmax_cross_entropy_loss_1/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
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
9softmax_cross_entropy_loss_1/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
?softmax_cross_entropy_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
>softmax_cross_entropy_loss_1/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
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
 softmax_cross_entropy_loss_1/SumSum softmax_cross_entropy_loss_1/Mul"softmax_cross_entropy_loss_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
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
valueB *
_output_shapes
: *
dtype0
�
\softmax_cross_entropy_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
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
&softmax_cross_entropy_loss_1/Greater/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
$softmax_cross_entropy_loss_1/GreaterGreater(softmax_cross_entropy_loss_1/num_present&softmax_cross_entropy_loss_1/Greater/y*
_output_shapes
: *
T0
�
$softmax_cross_entropy_loss_1/Equal/yConstN^softmax_cross_entropy_loss_1/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1Select$softmax_cross_entropy_loss_1/Greater<gradients/softmax_cross_entropy_loss_1/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
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
Lgradients/softmax_cross_entropy_loss_1/value_grad/tuple/control_dependency_1Identity:gradients/softmax_cross_entropy_loss_1/value_grad/Select_1C^gradients/softmax_cross_entropy_loss_1/value_grad/tuple/group_deps*
_output_shapes
: *M
_classC
A?loc:@gradients/softmax_cross_entropy_loss_1/value_grad/Select_1*
T0
x
5gradients/softmax_cross_entropy_loss_1/div_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
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
3gradients/softmax_cross_entropy_loss_1/div_grad/SumSum7gradients/softmax_cross_entropy_loss_1/div_grad/RealDivEgradients/softmax_cross_entropy_loss_1/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
7gradients/softmax_cross_entropy_loss_1/div_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/div_grad/Sum5gradients/softmax_cross_entropy_loss_1/div_grad/Shape*
T0*
_output_shapes
: *
Tshape0
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
9gradients/softmax_cross_entropy_loss_1/div_grad/Reshape_1Reshape5gradients/softmax_cross_entropy_loss_1/div_grad/Sum_17gradients/softmax_cross_entropy_loss_1/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
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
;gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1Select"softmax_cross_entropy_loss_1/Equal=gradients/softmax_cross_entropy_loss_1/Select_grad/zeros_likeJgradients/softmax_cross_entropy_loss_1/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
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
T0*
_output_shapes
: *N
_classD
B@loc:@gradients/softmax_cross_entropy_loss_1/Select_grad/Select_1
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
@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiplesConst*
dtype0*
_output_shapes
: *
valueB 
�
6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/TileTile9gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Reshape@gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
7gradients/softmax_cross_entropy_loss_1/Sum_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss_1/Sum_1_grad/Tile=gradients/softmax_cross_entropy_loss_1/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
5gradients/softmax_cross_entropy_loss_1/Sum_grad/ShapeShape softmax_cross_entropy_loss_1/Mul*
T0*
_output_shapes
:*
out_type0
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
T0*
_output_shapes
:*
Tshape0
�
=gradients/softmax_cross_entropy_loss_1/num_present_grad/ShapeShape:softmax_cross_entropy_loss_1/num_present/broadcast_weights*
T0*
_output_shapes
:*
out_type0
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
7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeReshape3gradients/softmax_cross_entropy_loss_1/Mul_grad/Sum5gradients/softmax_cross_entropy_loss_1/Mul_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
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
T0*
_output_shapes
: *
Tshape0
�
@gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape:^gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1
�
Hgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss_1/Mul_grad/ReshapeA^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*#
_output_shapes
:���������*J
_class@
><loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape*
T0
�
Jgradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1A^gradients/softmax_cross_entropy_loss_1/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss_1/Mul_grad/Reshape_1*
_output_shapes
: 
�
Ogradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
�
Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1ShapeDsoftmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like*
T0*
_output_shapes
:*
out_type0
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
T0*
_output_shapes
: *
Tshape0
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
Sgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1ReshapeOgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Sum_1Qgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
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
T0*#
_output_shapes
:���������*f
_class\
ZXloc:@gradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/Reshape_1
�
Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
Wgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/SumSumdgradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights_grad/tuple/control_dependency_1Ygradients/softmax_cross_entropy_loss_1/num_present/broadcast_weights/ones_like_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss_1/Reshape_2_grad/ShapeShape%softmax_cross_entropy_loss_1/xentropy*
_output_shapes
:*
out_type0*
T0
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
9gradients/softmax_cross_entropy_loss_1/Reshape_grad/ShapeShapediv_2*
T0*
_output_shapes
:*
out_type0
�
;gradients/softmax_cross_entropy_loss_1/Reshape_grad/ReshapeReshape8gradients/softmax_cross_entropy_loss_1/xentropy_grad/mul9gradients/softmax_cross_entropy_loss_1/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
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
gradients/div_2_grad/ReshapeReshapegradients/div_2_grad/Sumgradients/div_2_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
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
T0*
_output_shapes
: *
Tshape0
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
T0*/
_class%
#!loc:@gradients/div_2_grad/Reshape*'
_output_shapes
:���������

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
transpose_b( *
_output_shapes
:	�
*
transpose_a(*
T0
�
;gradients/sequential_1/dense_2/MatMul_grad/tuple/group_depsNoOp2^gradients/sequential_1/dense_2/MatMul_grad/MatMul4^gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
�
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
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
gradients/Shape_1Shapegradients/Switch:1*
out_type0*
_output_shapes
:*
T0
Z
gradients/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
_output_shapes
:*
out_type0*
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
T0*
out_type0*
_output_shapes
:
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
Ogradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape*
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
T0*
_output_shapes
:*
out_type0
y
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape*
T0
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
N*
T0**
_output_shapes
:����������: 
�
gradients/AddNAddN=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_grad*
N*
T0*(
_output_shapes
:����������*P
_classF
DBloc:@gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_grad
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
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1*
T0
�
3gradients/sequential_1/flatten_1/Reshape_grad/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
�
5gradients/sequential_1/flatten_1/Reshape_grad/ReshapeReshapeCgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency3gradients/sequential_1/flatten_1/Reshape_grad/Shape*
T0*/
_output_shapes
:���������@*
Tshape0
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
gradients/Shape_3Shapegradients/Switch_2:1*
_output_shapes
:*
out_type0*
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
N*
T0
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape_1Shape)sequential_1/dropout_1/cond/dropout/Floor*
_output_shapes
:*
out_type0*
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
Ggradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_depsNoOp?^gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeA^gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1
�
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
T0
y
6gradients/sequential_1/dropout_1/cond/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
T0*
_output_shapes
: *
Tshape0
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
T0*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*&
_output_shapes
:@@*X
_classN
LJloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
6gradients/sequential_1/activation_1/Relu_grad/ReluGradReluGradIgradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencysequential_1/activation_1/Relu*/
_output_shapes
:���������@*
T0
�
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
�
=gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_1/Relu_grad/ReluGrad9^gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
�
Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_1/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
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
Egradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdata8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Egradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*&
_output_shapes
:@*
data_formatNHWC*
strides
*
T0*
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta2_power/readIdentitybeta2_power*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
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
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adamzeros*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
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
shared_name *
shape:@*&
_output_shapes
:@*"
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
	container *
dtype0* 
_class
loc:@conv2d_1/bias*
shared_name *
_output_shapes
:@*
shape:@
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
zeros_3Const*
dtype0*
_output_shapes
:@*
valueB@*    
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
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam
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
conv2d_2/kernel/Adam/AssignAssignconv2d_2/kernel/Adamzeros_4*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
�
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
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
shared_name *
shape:@@*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
dtype0*
	container 
�
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
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
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
dtype0*
	container 
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
conv2d_2/bias/Adam_1/AssignAssignconv2d_2/bias/Adam_1zeros_7* 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0
b
zeros_8Const*
dtype0*!
_output_shapes
:���* 
valueB���*    
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
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
zeros_10Const*
dtype0*
_output_shapes	
:�*
valueB�*    
�
dense_1/bias/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
shared_name *
_output_shapes	
:�*
shape:�
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
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1zeros_11*
_output_shapes	
:�*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
use_locking(
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
_output_shapes
:	�
*
dtype0
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
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

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
dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
_output_shapes
:
*
_class
loc:@dense_2/bias*
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
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
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
_class
loc:@dense_1/kernel*!
_output_shapes
:���
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
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

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
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
e
lossScalarSummary	loss/tags"softmax_cross_entropy_loss_1/value*
T0*
_output_shapes
: 
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
$softmax_cross_entropy_loss_1/value:0֚�       ��-	���YXc�A*

loss�X@��       ��-	�:�YXc�A*

loss�@�lmZ       ��-	��YXc�A*

loss��@��       ��-	V��YXc�A*

loss9�?��`�       ��-	GT�YXc�A*

loss���?is�4       ��-	�,�YXc�A*

loss��?�Z�       ��-	���YXc�A*

loss�Ǚ?���       ��-	�l�YXc�A*

loss�L�?�D       ��-	N�YXc�A	*

lossh:�?��yx       ��-	���YXc�A
*

loss&I?�i=�       ��-	�W�YXc�A*

loss@�"?q_R_       ��-	���YXc�A*

loss �0?J�       ��-	���YXc�A*

loss��:?yI�+       ��-	#�YXc�A*

loss�?�Z;q       ��-	H��YXc�A*

loss�=?�H�7       ��-	!Z�YXc�A*

loss��>���o       ��-	w��YXc�A*

loss�M ?͇ߊ       ��-	r��YXc�A*

loss���>���       ��-	��YXc�A*

loss�.W?��C�       ��-	���YXc�A*

loss��?��-1       ��-	a�YXc�A*

loss�r#?�[��       ��-	e��YXc�A*

loss�?�9m       ��-	I��YXc�A*

loss�'1?�U?�       ��-	5D�YXc�A*

lossG�?���$       ��-	E��YXc�A*

loss�b�>k���       ��-	�%�YXc�A*

loss���>V��r       ��-	@��YXc�A*

lossޝ
?]���       ��-	�[�YXc�A*

lossD?#;�       ��-	Y��YXc�A*

loss�?���       ��-	��YXc�A*

loss��?���       ��-	)�YXc�A*

losss��>3�S�       ��-	>��YXc�A *

lossT��>R׋�       ��-	���YXc�A!*

loss��(?�]�       ��-	�N�YXc�A"*

lossZ�>&�z,       ��-	E��YXc�A#*

loss<e?F_�       ��-	�X�YXc�A$*

loss���>68��       ��-	��YXc�A%*

loss��?�W�       ��-	l��YXc�A&*

loss��2?���u       ��-	+�YXc�A'*

loss�.?���B       ��-	��YXc�A(*

loss�f?�'-       ��-	�]�YXc�A)*

loss���>���1       ��-	;��YXc�A**

loss=��>�+�(       ��-	ٖ�YXc�A+*

loss��>�R�o       ��-	���YXc�A,*

loss�K�>`�~�       ��-	.��YXc�A-*

lossY�?B.��       ��-	=(�YXc�A.*

loss���>���3       ��-	���YXc�A/*

lossآ�>��J       ��-	�Z�YXc�A0*

loss�>���Z       ��-	�~�YXc�A1*

loss�q>���g       ��-	`"�YXc�A2*

loss�Y>�bB       ��-	���YXc�A3*

loss��>k6��       ��-	,b�YXc�A4*

loss�D�>�er       ��-	��YXc�A5*

loss �?Ohԇ       ��-	G��YXc�A6*

loss{;>���       ��-	�P�YXc�A7*

lossW5+>_�-s       ��-	���YXc�A8*

loss*w>�mZ�       ��-	I��YXc�A9*

lossf�O>�xϯ       ��-	�E�YXc�A:*

loss�<�>eby"       ��-	!��YXc�A;*

loss?�>���       ��-	��YXc�A<*

loss���>i���       ��-	
1�YXc�A=*

loss��H>�n�$       ��-	s��YXc�A>*

loss��{>,�|       ��-	t�YXc�A?*

loss7v�>hO�<       ��-	��YXc�A@*

loss�)>�Q       ��-	��YXc�AA*

loss��|>3���       ��-	B�YXc�AB*

loss�l?���&       ��-	���YXc�AC*

loss*^�>��       ��-	�~�YXc�AD*

loss�?�>P��       ��-	�YXc�AE*

loss�{D>�J�       ��-	B��YXc�AF*

lossvU>M���       ��-	�L�YXc�AG*

loss�<�>ܭ�       ��-	}��YXc�AH*

loss[jD>��}�       ��-	�~�YXc�AI*

lossĭ�>�Km|       ��-	��YXc�AJ*

loss���>ҟ�       ��-	��YXc�AK*

lossDӌ>\y�)       ��-	qU�YXc�AL*

lossϢ�>�NXR       ��-	���YXc�AM*

loss|g�>C�ϗ       ��-	ڐ�YXc�AN*

losszk	>k��X       ��-	2�YXc�AO*

loss��W> ��7       ��-	���YXc�AP*

loss�>��*�       ��-	�e�YXc�AQ*

loss�>���2       ��-	e  ZXc�AR*

loss�2�>F}W       ��-	� ZXc�AS*

loss�s>��dc       ��-	�GZXc�AT*

loss�v�>�~��       ��-	��ZXc�AU*

loss�S;>�Ob       ��-	'�ZXc�AV*

loss�>���       ��-	�%ZXc�AW*

lossQ��>��0�       ��-	�ZXc�AX*

loss4Z�>��9I       ��-	�SZXc�AY*

loss܅O>|���       ��-	��ZXc�AZ*

loss�u�>�0<       ��-	��ZXc�A[*

loss1�Z>�W�]       ��-	�"ZXc�A\*

loss�G>�%��       ��-	b�ZXc�A]*

loss=�>��;       ��-	�RZXc�A^*

loss�Ƴ>Ye��       ��-	x�ZXc�A_*

lossв>�>�       ��-	&�ZXc�A`*

loss�n>��       ��-	l$	ZXc�Aa*

loss��|>��q�       ��-	��	ZXc�Ab*

loss�?�j�       ��-	P�
ZXc�Ac*

loss_b>L��H       ��-	)ZXc�Ad*

losst̷=��       ��-	��ZXc�Ae*

loss�W>5x�       ��-	.sZXc�Af*

loss��i>���l       ��-	�DZXc�Ag*

lossd��=��       ��-	�ZXc�Ah*

lossq�h>3�K�       ��-	>�ZXc�Ai*

loss��>��}       ��-	��ZXc�Aj*

loss�?>]�ؚ       ��-	�WZXc�Ak*

loss��=C�~8       ��-	4�ZXc�Al*

loss���>GG��       ��-	�ZXc�Am*

loss4��>ms�       ��-	32ZXc�An*

lossN�>vT��       ��-	d�ZXc�Ao*

lossiA=>�S��       ��-	IhZXc�Ap*

loss{M�=V�       ��-	OZXc�Aq*

lossLx�=��|       ��-	��ZXc�Ar*

lossFH=	�QP       ��-	�6ZXc�As*

loss���=) )�       ��-	�ZXc�At*

lossM	>�0"5       ��-	CpZXc�Au*

lossϦ3>�.1�       ��-		ZXc�Av*

loss�y�>,7 �       ��-	�ZXc�Aw*

loss�2>rR)       ��-	d>ZXc�Ax*

loss�=R>���       ��-	��ZXc�Ay*

losswE>s�O�       ��-	X�ZXc�Az*

lossX>��m�       ��-	=*ZXc�A{*

loss*d�>�� 1       ��-	�ZXc�A|*

loss�Ƹ=��B       ��-	�cZXc�A}*

loss��>s�       ��-	'ZXc�A~*

loss�ob>�lj�       ��-	��ZXc�A*

loss-�]>c&�       �	=GZXc�A�*

loss�P�>*�[�       �	��ZXc�A�*

loss�]�>D�*       �	-wZXc�A�*

lossQ�Q>��S       �	�ZXc�A�*

loss8��=��g�       �	֬ZXc�A�*

lossl�>��t       �	,H ZXc�A�*

loss-V>;�i�       �	�� ZXc�A�*

loss��;>�&!�       �	��!ZXc�A�*

loss.]�>2��u       �	 "ZXc�A�*

loss���>z�3       �	��"ZXc�A�*

lossA�>a;$       �	IK#ZXc�A�*

loss!�F>ݳm�       �	��#ZXc�A�*

loss��h=A;x       �	�$ZXc�A�*

loss���=��       �	�>%ZXc�A�*

losssW6>V/�       �	r�%ZXc�A�*

loss��g>�4       �	�r&ZXc�A�*

lossM�5>��O&       �	�	'ZXc�A�*

lossa�7>c��W       �	ӣ'ZXc�A�*

loss$�q=ό��       �	G>(ZXc�A�*

lossx>�u�       �	�(ZXc�A�*

lossO�F=o�!       �	Ku)ZXc�A�*

loss�y�=��2       �	0*ZXc�A�*

loss�>�V��       �	 (+ZXc�A�*

loss^�=����       �	/�+ZXc�A�*

loss�!>>w�       �	�W,ZXc�A�*

lossEJ>�f�       �	1�,ZXc�A�*

loss=Ć>�fK       �	?�-ZXc�A�*

lossH�?>���       �	1%.ZXc�A�*

loss$*=��I       �	"�.ZXc�A�*

loss�1�=��&       �	rn/ZXc�A�*

lossn�z>�L�       �	�0ZXc�A�*

loss��`>G(�       �	�0ZXc�A�*

lossH~�=�2d       �	F[1ZXc�A�*

loss&|f>��v�       �	��1ZXc�A�*

loss;�>�Lu�       �	ѓ2ZXc�A�*

lossFYh>5��       �	(*3ZXc�A�*

loss[I0=�r�       �	Q�3ZXc�A�*

loss��D>EW�       �	F]4ZXc�A�*

loss	w�=�Z=       �	��4ZXc�A�*

loss�6�=�M�       �	��5ZXc�A�*

loss)Z>����       �	46ZXc�A�*

loss^�
>��^       �	)�6ZXc�A�*

loss��$>��K       �	��7ZXc�A�*

lossL��=��%]       �	�B8ZXc�A�*

loss�_}=q�\�       �	��8ZXc�A�*

loss�x.>��1m       �	�~9ZXc�A�*

lossj�>h4j�       �	0:ZXc�A�*

loss�\V=��       �	O�:ZXc�A�*

loss(w�=���       �	0I;ZXc�A�*

loss�ɏ>��t       �	��;ZXc�A�*

lossF�[>7��i       �	Z�<ZXc�A�*

loss��>x4�:       �	�3=ZXc�A�*

loss!��=K
��       �	��=ZXc�A�*

loss��z=	�"�       �	6s>ZXc�A�*

loss�j>+{�t       �	?ZXc�A�*

loss\G">.�Ik       �	j�?ZXc�A�*

loss�\�=��       �	�<@ZXc�A�*

loss�\>�|��       �	�@ZXc�A�*

loss�U�=>�Ԃ       �	vmAZXc�A�*

loss�J>�'       �	FBZXc�A�*

loss�9>�Ř       �	��BZXc�A�*

loss���=��       �	�DCZXc�A�*

loss�E>g��6       �	j�CZXc�A�*

loss9k>��       �	F�DZXc�A�*

lossQ��=�w��       �	o.EZXc�A�*

lossƄb=�'�       �	��EZXc�A�*

loss�k=q�c       �	�aFZXc�A�*

loss��=���       �	��FZXc�A�*

loss�(�>m��       �	}�GZXc�A�*

loss��w>��E�       �	Y0HZXc�A�*

loss�6�=���       �	��HZXc�A�*

loss�ԯ>�X�`       �	��IZXc�A�*

loss�۠=��\�       �	JZXc�A�*

loss��G>ǈ��       �	��JZXc�A�*

loss���=Ğ��       �	shKZXc�A�*

lossyI>�O       �	LZXc�A�*

lossr� >�|#        �	�LZXc�A�*

loss��+>ژl       �	�5MZXc�A�*

loss �=�˟�       �	%�MZXc�A�*

lossP=>���       �	QfNZXc�A�*

loss�	>�П       �	T�NZXc�A�*

loss}n!>����       �	S�OZXc�A�*

loss��=�1J�       �	�4PZXc�A�*

loss�RM>-��       �	U�PZXc�A�*

loss��=]��       �	�{QZXc�A�*

loss�Z�=]�F�       �	RZXc�A�*

loss:�	>jϰ�       �	�RZXc�A�*

lossS�>�aE       �	�CSZXc�A�*

lossJ�\>�I^�       �	��SZXc�A�*

lossa��=��g       �	�TZXc�A�*

loss�Ԉ>�,       �	�UZXc�A�*

loss�R>�3;�       �	�UZXc�A�*

loss���>���       �	pAVZXc�A�*

loss,�=��n       �	�VZXc�A�*

loss��>���       �	�pWZXc�A�*

loss��7>�.a�       �	�XZXc�A�*

loss<�>���       �	��XZXc�A�*

loss���=�H�       �	�.YZXc�A�*

loss�_\=�"!u       �	@�YZXc�A�*

lossI	�="�Q�       �	�VZZXc�A�*

loss	%J>TY�       �	[�ZZXc�A�*

lossM��=� �       �	g�[ZXc�A�*

loss�9.>|��c       �	\ZXc�A�*

loss5>;?3�       �	T�\ZXc�A�*

loss�J�=SG       �	9G]ZXc�A�*

lossOǐ>��N       �	��]ZXc�A�*

lossn�%>�f��       �	1~^ZXc�A�*

loss�!>�	է       �	�_ZXc�A�*

losswc�=���'       �	��_ZXc�A�*

loss�C>����       �	�W`ZXc�A�*

loss,�>�曚       �	�XaZXc�A�*

losse�t>�I�       �	4/bZXc�A�*

loss{Qp=���       �	��bZXc�A�*

loss��=q㥏       �	�lcZXc�A�*

lossԟ�=��Ш       �	�dZXc�A�*

loss��	>E��       �	?�dZXc�A�*

loss3��=U�n�       �	^GeZXc�A�*

loss���=��9       �	0�fZXc�A�*

loss?�I>��ѥ       �		QgZXc�A�*

loss���=�wrH       �	�gZXc�A�*

loss��=g���       �	��iZXc�A�*

loss!��=l��f       �	jjZXc�A�*

loss���=N��       �	akZXc�A�*

lossV�#>���       �	29lZXc�A�*

loss}�U>�TQ       �	��lZXc�A�*

losso�>�U�       �	��mZXc�A�*

loss�8�=鐞&       �	emnZXc�A�*

loss�4�=�&��       �	oZXc�A�*

loss��=�@sk       �	��oZXc�A�*

lossS>�=�       �	�RpZXc�A�*

loss�fA>R�       �	��pZXc�A�*

loss�J�=Pru\       �	ޒqZXc�A�*

loss_s�=xD��       �	 *rZXc�A�*

loss;�=I�ԯ       �	W�rZXc�A�*

loss �=����       �	�ksZXc�A�*

loss65)>:㘯       �	l
tZXc�A�*

lossÐ�={X2       �	v�tZXc�A�*

loss���>^&       �	�CuZXc�A�*

loss�J�=(�q       �	��uZXc�A�*

loss���=]��       �	zrvZXc�A�*

lossW�&>�B*V       �	�	wZXc�A�*

loss6v�>��}       �	W�wZXc�A�*

lossN{�=�RB�       �	LOxZXc�A�*

lossD��=�LS%       �	��xZXc�A�*

loss��8>Ӕ%       �	�}yZXc�A�*

loss�	>��       �	�zZXc�A�*

loss�k�=�Zsb       �	X�zZXc�A�*

loss $�=�ڪ       �	�U{ZXc�A�*

loss��0>J�5       �	��{ZXc�A�*

loss���=]2       �	k�|ZXc�A�*

lossH��=��       �	�}ZXc�A�*

loss1�4>����       �	��}ZXc�A�*

loss���=��?       �	�O~ZXc�A�*

loss�#�<����       �	��~ZXc�A�*

loss��.=�:       �	n�ZXc�A�*

lossi/�=1�0       �	%�ZXc�A�*

loss(�v=���       �	���ZXc�A�*

loss-�K>�$�       �	LT�ZXc�A�*

losst�>ӳF�       �	��ZXc�A�*

losssJ�=�6��       �	���ZXc�A�*

lossT0>&6	�       �	}%�ZXc�A�*

loss�kt=�       �	��ZXc�A�*

loss�w=��L�       �	Rb�ZXc�A�*

loss�g=RCC�       �	���ZXc�A�*

loss�'�=㺰�       �	���ZXc�A�*

loss��<�y��       �	�6�ZXc�A�*

loss�1�=����       �	�ՆZXc�A�*

loss�UX=T"=�       �	%v�ZXc�A�*

loss���=u��o       �	W%�ZXc�A�*

loss]>�ۡ       �	�ψZXc�A�*

loss��t>��4�       �	�j�ZXc�A�*

loss�vM>�e=*       �	S�ZXc�A�*

lossn�>2`��       �	Q��ZXc�A�*

loss��r=ޘC        �	y@�ZXc�A�*

loss���=�ւW       �	a��ZXc�A�*

loss7��=I�K#       �	���ZXc�A�*

loss��Q>X��       �	���ZXc�A�*

lossԛ�=|(�^       �	�h�ZXc�A�*

lossD�>-IE�       �	�j�ZXc�A�*

loss_�<=ϼ�       �	�%�ZXc�A�*

loss7 �=�j�       �	�ƐZXc�A�*

loss��!=�ؒ       �	���ZXc�A�*

loss̗�=���B       �	���ZXc�A�*

loss�@�=����       �	���ZXc�A�*

loss*+!>ND�       �	A*�ZXc�A�*

loss>w��y       �	P��ZXc�A�*

loss��O=�1��       �	TǕZXc�A�*

loss�ޘ=5��       �	���ZXc�A�*

loss�@�=kz*�       �	��ZXc�A�*

lossT~�<6,�<       �	���ZXc�A�*

loss_[�=p�*�       �	���ZXc�A�*

loss��3>q�F       �	\��ZXc�A�*

loss���=w#
�       �	L4�ZXc�A�*

lossS}�<4lJ)       �	�͛ZXc�A�*

loss2m>���X       �	�e�ZXc�A�*

lossV��=�مD       �	� �ZXc�A�*

loss�G�=�d�       �	R��ZXc�A�*

loss�$>��?       �	�3�ZXc�A�*

loss��z>�O       �	�ОZXc�A�*

loss�(�=���       �	�k�ZXc�A�*

loss'��=��
       �	}�ZXc�A�*

loss&WM>�g�O       �	z��ZXc�A�*

loss$��=a��       �	S?�ZXc�A�*

loss���=G��       �	�֡ZXc�A�*

loss���=��       �	Yl�ZXc�A�*

lossЙ=a�MW       �	}�ZXc�A�*

loss�:#=pGw�       �	���ZXc�A�*

loss��">�1       �	�7�ZXc�A�*

loss�"�=J��       �	g�ZXc�A�*

loss���=n���       �	a��ZXc�A�*

loss/Vp>z�1       �	o��ZXc�A�*

loss�P�=F��       �	��ZXc�A�*

loss�s=�X       �	t�ZXc�A�*

lossq�=}�       �	��ZXc�A�*

loss�q=��       �	��ZXc�A�*

loss?�
>[,8�       �	�9�ZXc�A�*

loss��=�F�+       �	VҪZXc�A�*

loss�Ƶ>+�(Z       �	�h�ZXc�A�*

loss�q,>/��       �		��ZXc�A�*

loss�C�=����       �	S��ZXc�A�*

loss���=}+��       �	�&�ZXc�A�*

losssa�=|�`�       �	���ZXc�A�*

lossF#>�k       �	_�ZXc�A�*

loss-�W=J��       �	7��ZXc�A�*

loss6e�=���!       �	���ZXc�A�*

lossAi=
A�       �	�,�ZXc�A�*

lossD>k��       �	zưZXc�A�*

loss���>��       �	Zf�ZXc�A�*

loss���=�h�:       �	���ZXc�A�*

loss��=�|a1       �	���ZXc�A�*

loss�="6t	       �	NG�ZXc�A�*

lossl<�=�(A_       �	?�ZXc�A�*

lossʒ�=JS�       �	x�ZXc�A�*

loss�� >$9�       �	��ZXc�A�*

lossa��=V��       �	Y��ZXc�A�*

lossĂ�<��+8       �	�@�ZXc�A�*

loss��u>�/R       �	e�ZXc�A�*

lossݣn=��=�       �	>x�ZXc�A�*

loss�>>��!�       �	��ZXc�A�*

loss� >�(�       �	?ƸZXc�A�*

loss@� >�m�m       �	}^�ZXc�A�*

loss���=na�W       �	���ZXc�A�*

loss���=����       �	��ZXc�A�*

lossH
�=yk�	       �	�)�ZXc�A�*

loss�P�=Jm       �	�ûZXc�A�*

loss���=�-b�       �	�]�ZXc�A�*

loss��=ĸ��       �	b��ZXc�A�*

lossR��=Q�?�       �	ʌ�ZXc�A�*

loss���>TP�I       �	&�ZXc�A�*

loss%��=&��       �	½�ZXc�A�*

loss�l�=�\t       �	�T�ZXc�A�*

loss���=Z��	       �	��ZXc�A�*

loss��&>��W�       �	��ZXc�A�*

loss��=��       �	^-�ZXc�A�*

loss��J>���       �	)��ZXc�A�*

loss�M�=�6�       �	m�ZXc�A�*

loss��=�'�       �	w�ZXc�A�*

loss���=Д�:       �	���ZXc�A�*

lossD�>6�M�       �	�B�ZXc�A�*

loss��=��&       �	���ZXc�A�*

loss`�>�8g|       �	Û�ZXc�A�*

lossO�=�%~       �	2�ZXc�A�*

loss	Cd>�t&       �	���ZXc�A�*

loss�pD>�w�Z       �	Hk�ZXc�A�*

lossL�>	�       �	��ZXc�A�*

lossN��=��"       �	��ZXc�A�*

loss��=���       �	�B�ZXc�A�*

loss�$�=��F�       �	���ZXc�A�*

loss�� >fu��       �	���ZXc�A�*

loss_)=Ǧ7�       �	\V�ZXc�A�*

loss3Yq=�L&       �	���ZXc�A�*

loss�{�=��v�       �	�$�ZXc�A�*

loss���=�Q�       �	���ZXc�A�*

loss�=�=�dsG       �	a��ZXc�A�*

loss��=Ʀ��       �	'2�ZXc�A�*

loss���=���R       �	@j�ZXc�A�*

loss�׸<M4y�       �	ɬ�ZXc�A�*

loss�n=r�h!       �	G�ZXc�A�*

loss�t>(ݧ       �	:��ZXc�A�*

loss:�6>��^�       �	׆�ZXc�A�*

lossܩ�=��       �	�!�ZXc�A�*

lossz}�=�
2�       �	��ZXc�A�*

loss'+>s)|�       �	�S�ZXc�A�*

lossH��=Z�{}       �	1��ZXc�A�*

loss�;�>�$	       �	z��ZXc�A�*

loss�z�=
Э�       �	! �ZXc�A�*

loss��\>`��       �	��ZXc�A�*

lossRI�=c�K       �	�W�ZXc�A�*

lossO5r>�c�1       �	���ZXc�A�*

loss���<�Y�       �	m��ZXc�A�*

loss6br=�E��       �	�$�ZXc�A�*

loss���=���/       �	ؼ�ZXc�A�*

loss��=�S�       �	\U�ZXc�A�*

loss��=�V*       �	���ZXc�A�*

loss��>$?[       �	���ZXc�A�*

lossR#�=�Y��       �	�"�ZXc�A�*

lossP>��q�       �	<��ZXc�A�*

loss�J>>���       �	:]�ZXc�A�*

loss��B>�}6�       �	���ZXc�A�*

loss��=�{W       �	���ZXc�A�*

lossS>F9m       �	�6�ZXc�A�*

lossQb�=jW�1       �	��ZXc�A�*

loss�Z�=:s�       �	Ab�ZXc�A�*

loss��=��C       �	� �ZXc�A�*

lossc�o=E��       �	4��ZXc�A�*

loss�ɫ=?Lc�       �	0G�ZXc�A�*

loss��>L�Ӳ       �	���ZXc�A�*

loss8�D>->�E       �	��ZXc�A�*

loss��4=�Lq�       �	%@�ZXc�A�*

loss�?=N'�       �	e��ZXc�A�*

loss*-=[���       �	���ZXc�A�*

loss�=J��       �	�v�ZXc�A�*

loss%݄=�6�7       �	�3�ZXc�A�*

loss�->�|��       �	7�ZXc�A�*

loss��;>�	�       �	���ZXc�A�*

loss=e�=V�u#       �	���ZXc�A�*

lossg�>�5t�       �	�)�ZXc�A�*

loss�K�<\�5�       �	v��ZXc�A�*

loss���=���G       �	nm�ZXc�A�*

loss��>w���       �	w�ZXc�A�*

lossҧ�=߾��       �	��ZXc�A�*

loss<r=vSSU       �	H��ZXc�A�*

loss�.`>��w�       �	�i�ZXc�A�*

loss.�8>��C=       �	��ZXc�A�*

loss�g�=��>t       �	���ZXc�A�*

loss�L=6��F       �	�<�ZXc�A�*

loss�v�=��<       �	���ZXc�A�*

loss��>R�sd       �	it�ZXc�A�*

loss��=I�o(       �	��ZXc�A�*

loss�= m��       �	���ZXc�A�*

lossC�=���       �	TR�ZXc�A�*

loss��W=G���       �	���ZXc�A�*

loss�z=�c�       �	ʌ�ZXc�A�*

loss}T4=*<F*       �	�*�ZXc�A�*

loss�B=�H       �	e��ZXc�A�*

loss6�,=#�"       �	�a�ZXc�A�*

loss�N=�4�|       �	���ZXc�A�*

losss�j=���       �	B��ZXc�A�*

loss���=�`k�       �	I-�ZXc�A�*

lossʏ�=K�	       �	2��ZXc�A�*

loss��=>�'       �	�k�ZXc�A�*

loss��=����       �	_�ZXc�A�*

loss��=�Pک       �	��ZXc�A�*

loss\��>D��3       �	>�ZXc�A�*

loss��$>���       �	x��ZXc�A�*

lossI��=5A��       �	�|�ZXc�A�*

loss"�>B ��       �	n�ZXc�A�*

loss�5�=�A��       �	ٴ�ZXc�A�*

lossk'�=��C�       �	N�ZXc�A�*

lossI��<8!�       �	���ZXc�A�*

loss|��=kt�       �	�� [Xc�A�*

loss=ӓA       �	-[Xc�A�*

lossq�@>�l)T       �	~�[Xc�A�*

loss}�>�@d�       �	�g[Xc�A�*

loss�D�=����       �	�[Xc�A�*

lossv�=Ou�       �	�[Xc�A�*

loss�Y�=/�&�       �	�N[Xc�A�*

loss���=&=Hu       �	��[Xc�A�*

loss��=����       �	Œ[Xc�A�*

lossd�v=`�[       �	�,[Xc�A�*

lossH>I���       �	��[Xc�A�*

loss�"�=����       �	�d[Xc�A�*

loss��=��[       �	�[Xc�A�*

loss?m�=.T�!       �	9�[Xc�A�*

loss�d�=�f9�       �	�=	[Xc�A�*

loss���;Վ��       �	k�	[Xc�A�*

loss;��<��F       �	v
[Xc�A�*

loss�Ʊ=�1�       �	�[Xc�A�*

loss���<�H�       �	��[Xc�A�*

loss6��=�KK�       �	j[Xc�A�*

loss?:=Zs�       �	�D[Xc�A�*

loss�Gr=o��w       �	��[Xc�A�*

lossQ�=��h�       �	P�[Xc�A�*

loss���=�@��       �	v�[Xc�A�*

loss�+=k�Ҷ       �	i�[Xc�A�*

loss�=A%       �	�+[Xc�A�*

loss���<�^�       �	��[Xc�A�*

loss;;�=���       �	�%[Xc�A�*

loss�rE=�Y�       �	U�[Xc�A�*

lossڐ�=����       �	tb[Xc�A�*

lossmf�<��       �	K [Xc�A�*

loss6�=f�j$       �	�[Xc�A�*

loss.u�=4��<       �	UQ[Xc�A�*

loss��9=͒:3       �	��[Xc�A�*

loss��>+�t       �	�[Xc�A�*

lossCY�<V��n       �	�x[Xc�A�*

loss�=���       �	�[Xc�A�*

loss=��=���       �	+[Xc�A�*

loss�l>��+@       �	��[Xc�A�*

loss�%�=M�,T       �	�h[Xc�A�*

loss��=z FS       �	oe[Xc�A�*

loss�� =��r       �	��[Xc�A�*

lossh�=v�P       �	͕[Xc�A�*

loss� =�0       �	�/[Xc�A�*

loss�h�=�8x�       �	W�[Xc�A�*

loss,�T<r��       �	Mj[Xc�A�*

loss2��=��F�       �	 [Xc�A�*

loss��=o�f�       �	�� [Xc�A�*

lossC�=+��a       �	�T![Xc�A�*

loss��=�c�       �	4i"[Xc�A�*

lossC�">�R$       �	2#[Xc�A�*

loss�Z�=����       �	'N$[Xc�A�*

loss$f=�@��       �	�$[Xc�A�*

lossȭ=͡{�       �	g�%[Xc�A�*

loss�=j(�       �	z&[Xc�A�*

lossn�(=�g��       �	� '[Xc�A�*

loss�=a�+       �	��'[Xc�A�*

lossqb=�       �	 A([Xc�A�*

loss�T<���       �	��([Xc�A�*

loss]z�<��R       �	ׄ)[Xc�A�*

lossvB<�J       �	�'*[Xc�A�*

loss�G�<G75z       �	Y�*[Xc�A�*

loss1>�Ȉ       �	 c+[Xc�A�*

loss��7;�YC/       �	G�+[Xc�A�*

loss��;�m6�       �	�,[Xc�A�*

lossV��;���D       �	/6-[Xc�A�*

loss�G==��       �	=�-[Xc�A�*

loss,��=̏9       �	�m.[Xc�A�*

loss�I�=�.�       �	�/[Xc�A�*

lossԎ�;�׹�       �	�/[Xc�A�*

loss�H�=Z��       �	!?0[Xc�A�*

loss ��>KWv       �	w�0[Xc�A�*

loss-i;��j�       �	eq1[Xc�A�*

loss/ �=A���       �	$	2[Xc�A�*

lossD��=�#       �	�2[Xc�A�*

lossG�=��       �	�A3[Xc�A�*

loss$9�=Y���       �	�3[Xc�A�*

loss���<���       �	��4[Xc�A�*

loss��;=0Su       �	�15[Xc�A�*

lossX >ߌ��       �	��5[Xc�A�*

loss�%=�6��       �	�}6[Xc�A�*

loss (�=Ļ	�       �	�7[Xc�A�*

loss��:=��&       �	�*8[Xc�A�*

loss?�f>J9��       �	.�8[Xc�A�*

loss�b>�th       �	|_9[Xc�A�*

lossء�=��       �	��9[Xc�A�*

loss���=q��       �	��:[Xc�A�*

lossje>�ܖ       �	#/;[Xc�A�*

lossXTO=��d        �	X�;[Xc�A�*

loss�{>�g�       �	�h<[Xc�A�*

loss��>g�       �	�=[Xc�A�*

lossP!>(w�#       �	��=[Xc�A�*

loss�Y=����       �	A>[Xc�A�*

loss��<=���       �	��>[Xc�A�*

loss���=&�       �	�o?[Xc�A�*

lossh~�<x7M�       �	�@[Xc�A�*

lossCF=�ܿ       �	��@[Xc�A�*

loss��:<�=8w       �	�7A[Xc�A�*

loss;7�=k�j�       �	[�A[Xc�A�*

loss�HR=JM�       �	�fB[Xc�A�*

loss�g�=1�g�       �	�C[Xc�A�*

loss�>H9�       �	��C[Xc�A�*

loss|QO=��o�       �	�5D[Xc�A�*

loss:k�=��u       �	(�D[Xc�A�*

loss=�(=ey6v       �	0hE[Xc�A�*

lossXvf=؅r�       �	�F[Xc�A�*

loss�J=p�܎       �	I�F[Xc�A�*

loss�>=M���       �	*8G[Xc�A�*

loss��_=p~�P       �	l�G[Xc�A�*

loss���=�}�       �	�oH[Xc�A�*

loss�/>�G�       �	�I[Xc�A�*

loss�h>
��N       �	��I[Xc�A�*

lossv*�<N���       �	�[J[Xc�A�*

loss��=ӈ�d       �	@�J[Xc�A�*

loss쵳=�n�       �	v�K[Xc�A�*

loss��m=խ��       �	�0L[Xc�A�*

lossq{�=D��h       �	i�L[Xc�A�*

loss��=v+�@       �	��M[Xc�A�*

loss�'>�'       �	$DN[Xc�A�*

lossTT�<��Ni       �	��N[Xc�A�*

loss+�=����       �	��O[Xc�A�*

loss,8n=\��p       �	�dP[Xc�A�*

loss�KI=�rj"       �	�Q[Xc�A�*

loss�P�=��       �	d�n[Xc�A�*

loss|z�=����       �	'o[Xc�A�*

loss���=,�'       �	"�o[Xc�A�*

lossXd.=��/       �	�dp[Xc�A�*

loss�U=#"�       �	��p[Xc�A�*

loss��p=gL��       �	��q[Xc�A�*

loss,�=��i+       �	=)r[Xc�A�*

lossh֦<��&\       �	U�r[Xc�A�*

losst��=��k)       �	)Ys[Xc�A�*

loss#G#>8�)       �	�s[Xc�A�*

loss�g=b��M       �	ۈt[Xc�A�*

loss<>�x       �	�)u[Xc�A�*

loss�w#>�{�.       �	��u[Xc�A�*

loss6�D=��b�       �	�dv[Xc�A�*

loss�T�<�G��       �	�w[Xc�A�*

lossqe�>����       �	V�w[Xc�A�*

loss᥾<$��       �	�4x[Xc�A�*

loss)�=cq�       �	I�x[Xc�A�*

lossی�=��͔       �	�{y[Xc�A�*

loss\[>4��       �	mz[Xc�A�*

loss�X1>����       �	N�z[Xc�A�*

loss�b>�Ž�       �	6Z{[Xc�A�*

loss$=��d       �	��{[Xc�A�*

lossDJ�=�x�       �	Z�|[Xc�A�*

lossn�=��Gl       �	�7}[Xc�A�*

lossH̻<G�        �	|�}[Xc�A�*

losscU>I鑂       �	Ui~[Xc�A�*

loss�"=�܇�       �	��~[Xc�A�*

loss���=xq �       �	h�[Xc�A�*

loss��=�xm�       �	�-�[Xc�A�*

lossۿ�=L�       �		�[Xc�A�*

lossT�=\�,�       �	:z�[Xc�A�*

loss=j>�w�a       �	�C�[Xc�A�*

loss�	%>�E�       �	wق[Xc�A�*

lossd��=87��       �	t�[Xc�A�*

loss�>>�u4       �	��[Xc�A�*

loss\�c==��       �	��[Xc�A�*

loss%�e>�y�        �	Q3�[Xc�A�*

loss
7>]��       �	Sυ[Xc�A�*

loss�=�=���       �	�g�[Xc�A�*

loss���=�*q+       �	� �[Xc�A�*

loss�e=��<       �	؟�[Xc�A�*

loss�ʩ<�=       �	5A�[Xc�A�*

lossq|�=}��       �	��[Xc�A�*

losskv">0Ӣk       �	���[Xc�A�*

loss��=>��P�       �	C �[Xc�A�*

loss�r>f�n       �	���[Xc�A�*

loss<ڽ=���       �	�]�[Xc�A�*

loss@b=3|�       �	~��[Xc�A�*

loss@wn<���       �	Z��[Xc�A�*

loss�Z�<]��=       �	=��[Xc�A�*

loss��==��q       �	E��[Xc�A�*

loss��d=ޠX1       �	���[Xc�A�*

lossEj�>���u       �	B�[Xc�A�*

loss��2=�       �	��[Xc�A�*

loss�C)=���       �	/��[Xc�A�*

lossj#=�o|�       �	{J�[Xc�A�*

loss��Q<�$��       �	��[Xc�A�*

loss�>���       �	�Ǔ[Xc�A�*

losseϝ=�Z�%       �	Tt�[Xc�A�*

lossHFR=~r�        �	�=�[Xc�A�*

lossr�z=�c��       �	�ѕ[Xc�A�*

loss���<}�ަ       �	�m�[Xc�A�*

loss�!6=5�Q^       �	��[Xc�A�*

loss�=��
�       �	���[Xc�A�*

loss�<�<ﲠ�       �	�A�[Xc�A�*

lossn��=�jf�       �	Eژ[Xc�A�*

loss��K=L���       �	�l�[Xc�A�*

loss?>��~?       �	6�[Xc�A�*

lossAB�=1��       �	̖�[Xc�A�*

loss��<�g��       �	�V�[Xc�A�*

lossM,=�
�{       �	{��[Xc�A�*

loss#��=��:�       �	/��[Xc�A�*

loss���=��k       �	�!�[Xc�A�*

loss}C�=I���       �	���[Xc�A�*

loss�߀=��N       �	;V�[Xc�A�*

loss�Y�=��Z�       �	��[Xc�A�*

loss���=<1g       �	/��[Xc�A�*

loss=��f�       �	�[Xc�A�*

lossc�{=��[       �	���[Xc�A�*

lossA�{=��%t       �	�S�[Xc�A�*

loss</>�%D       �	��[Xc�A�*

loss�p�=�j�       �	���[Xc�A�*

lossH!�=�N�       �	�!�[Xc�A�*

loss��i=��x[       �	���[Xc�A�*

loss�τ=\S�       �	���[Xc�A�*

loss��6>>�       �	5�[Xc�A�*

lossd3�=�ɳ�       �	���[Xc�A�*

loss�H�=8g:       �	$��[Xc�A�*

loss!c=<S�       �	��[Xc�A�*

loss
ɭ=4��0       �	���[Xc�A�*

loss�[�=)=4       �	���[Xc�A�*

loss�=�E�       �	�J�[Xc�A�*

loss@�3=�?�       �	�3�[Xc�A�*

loss���=5��       �	�ܫ[Xc�A�*

lossP�	=*��       �	I��[Xc�A�*

lossZ�=T��       �	!�[Xc�A�*

loss�=�o�       �	���[Xc�A�*

lossj�>,{lL       �	:\�[Xc�A�*

loss$#L=R5�7       �	z �[Xc�A�*

loss�g<����       �	l��[Xc�A�*

loss�F=ם��       �	�O�[Xc�A�*

lossȭ+>w���       �	x�[Xc�A�*

loss�OZ=��ܸ       �	�[Xc�A�*

loss���=_��       �	�#�[Xc�A�*

loss�h>T��s       �	���[Xc�A�*

loss�]�<�qp�       �	CW�[Xc�A�*

loss�w<|�;2       �	��[Xc�A�*

loss�,�==���       �	O��[Xc�A�*

lossIu=��/       �	�3�[Xc�A�*

loss(�=�Ry|       �	dε[Xc�A�*

loss#� =\q       �	�k�[Xc�A�*

loss���=��(�       �	��[Xc�A�*

lossŀ+<Ь�       �	���[Xc�A�*

loss���<���       �	i6�[Xc�A�*

lossz	�=���        �	�˸[Xc�A�*

loss�L	=��P�       �	`�[Xc�A�*

lossER�=�΍Z       �	��[Xc�A�*

loss�V�=�*�       �	]��[Xc�A�*

loss�J1=��XZ       �	?9�[Xc�A�*

loss�	�=�o�z       �	�л[Xc�A�*

loss�Ɛ=�K�       �	l�[Xc�A�*

loss��=J��T       �	�[Xc�A�*

loss��=��2�       �	��[Xc�A�*

loss�Ʊ<M�B�       �	e9�[Xc�A�*

loss���=_	$�       �	Sξ[Xc�A�*

loss�R>>b3,       �	�e�[Xc�A�*

loss�T=&O`       �	;��[Xc�A�*

lossh�	>��0�       �	��[Xc�A�*

loss��=7[��       �	�8�[Xc�A�*

loss��L=ou�       �	f��[Xc�A�*

loss���=��R       �	�w�[Xc�A�*

loss%'=ً?#       �	��[Xc�A�*

lossaf=�`��       �	��[Xc�A�*

loss)��=��        �	�A�[Xc�A�*

loss�>xc��       �	���[Xc�A�*

lossʷ%>[�1       �	-x�[Xc�A�*

loss�$
=ʥ72       �	a�[Xc�A�*

loss��<��O[       �	V��[Xc�A�*

lossS��<+6R�       �	3S�[Xc�A�*

loss�\<���       �	���[Xc�A�*

lossýT=�!       �	%��[Xc�A�*

loss�F�=b"ga       �	�f�[Xc�A�*

loss�߂=���7       �	Z�[Xc�A�*

loss�>S�
�       �	 ��[Xc�A�*

loss8�v=�� x       �		o�[Xc�A�*

lossש�=�#w�       �	S�[Xc�A�*

lossN:�<�z��       �	Q��[Xc�A�*

lossf%r<��       �	,��[Xc�A�*

loss�=/䆄       �	-[�[Xc�A�*

loss�$=����       �	�1�[Xc�A�*

loss�A=����       �	���[Xc�A�*

lossj��=|A�       �	Mj�[Xc�A�*

loss�:b>��CU       �	�-�[Xc�A�*

loss�P�==��       �	���[Xc�A�*

loss\vA<�iq�       �	���[Xc�A�*

loss�Kk=�
 �       �	&U�[Xc�A�*

loss��=OмK       �		��[Xc�A�*

loss�k.>�&,�       �	���[Xc�A�*

loss�=�aG       �	�W�[Xc�A�*

loss�w�=c��       �	N)�[Xc�A�*

loss��@=	�       �	���[Xc�A�*

loss7��=}D_�       �	8f�[Xc�A�*

lossf1�<n�sZ       �	h$�[Xc�A�*

loss�O�=��d       �	�]�[Xc�A�*

loss
I=�.,       �	���[Xc�A�*

lossWKt= �Y       �	�V�[Xc�A�*

lossY�=�߼Q       �	Q�[Xc�A�*

loss`��=Wm�1       �	0��[Xc�A�*

lossn@�=��       �	m��[Xc�A�*

loss�y�=.k       �	>�[Xc�A�*

loss�!<�Y��       �	K�[Xc�A�*

loss�O�<q=G�       �	���[Xc�A�*

loss�i�=&o��       �	���[Xc�A�*

loss&�=<���       �	��[Xc�A�*

losss�/=@�G       �	�[Xc�A�*

loss[�=?�"a       �	�{�[Xc�A�*

loss;6p=�xI       �	Y�[Xc�A�*

lossH�
=�O�C       �	���[Xc�A�*

loss�r<F]�       �	 c�[Xc�A�*

lossY <�Ѳ�       �		7�[Xc�A�*

loss���=<{ԕ       �	��[Xc�A�*

loss���=��       �	o,�[Xc�A�*

loss?=l�gY       �	u��[Xc�A�*

loss�]�<�L�       �	 ��[Xc�A�*

loss�C=mkI]       �	5�[Xc�A�*

loss7�=����       �	r�[Xc�A�*

lossNB=�.��       �	�K�[Xc�A�*

loss�0V=?�y       �	p��[Xc�A�*

loss�\�=��>       �	���[Xc�A�*

lossύN>ր��       �	I*�[Xc�A�*

loss���=N�q�       �	t��[Xc�A�*

loss%��<��q       �	u�[Xc�A�*

loss��o=e@Dl       �	���[Xc�A�*

lossE�=dB��       �	���[Xc�A�*

loss�{�<ʦi       �	zV�[Xc�A�*

loss�q�<(���       �	�0�[Xc�A�*

loss�{=��Rq       �	���[Xc�A�*

loss�>G�A       �	�t�[Xc�A�*

loss �!=_̉D       �	n�[Xc�A�*

loss��=f~�s       �	ȳ�[Xc�A�*

loss�N<�OJH       �	�S�[Xc�A�*

loss�l<��       �	^��[Xc�A�*

loss,��<���       �	���[Xc�A�*

lossL N=��       �	!;�[Xc�A�*

loss5	�=KVE       �	���[Xc�A�*

lossjY >E(       �	h��[Xc�A�*

loss�>�c��       �	2;�[Xc�A�*

loss=��=�)�       �	U��[Xc�A�*

loss}>x)3�       �	��[Xc�A�*

loss\�=aKk�       �	��[Xc�A�*

loss*$}=�X�U       �	��[Xc�A�*

lossΫI<Ec       �	zP \Xc�A�*

losss�<Z��       �	�� \Xc�A�*

lossr/�=Xұ	       �	��\Xc�A�*

lossXj=8��m       �	�2\Xc�A�*

loss�=�e�{       �	��\Xc�A�*

loss�(�>TH��       �	 s\Xc�A�*

loss�y�=���       �	�\Xc�A�*

loss`r�=��E~       �	d�\Xc�A�*

loss�=���       �	HP\Xc�A�*

loss�f=��Cw       �	_�\Xc�A�*

loss���=��d}       �	��\Xc�A�*

loss���=O�C�       �	|,\Xc�A�*

loss(=*��       �	2�\Xc�A�*

lossF�\=X�M       �	_\Xc�A�*

lossi�r=\�       �	X 	\Xc�A�*

lossS�=�?�       �	�	\Xc�A�*

loss3�|=ujr�       �	E
\Xc�A�*

lossz4�=l�95       �	��
\Xc�A�*

loss˳<mb�\       �	�\Xc�A�*

lossi�Y=	 �       �	c�\Xc�A�*

loss�>{�w       �	:�\Xc�A�*

loss\C=�y8�       �	3�\Xc�A�*

loss�n�=��       �	�^\Xc�A�*

loss�z�=e+�       �	Ӈ\Xc�A�*

loss��a=��       �	�@\Xc�A�*

loss}j�=1g�       �	�'\Xc�A�*

lossd�E=]�3=       �	C�\Xc�A�*

loss�l�<�WGm       �	J�\Xc�A�*

loss� f=��'o       �	{g\Xc�A�*

lossޫ=�d#]       �	�\Xc�A�*

loss#�+=�>'�       �	��\Xc�A�*

loss�&>7��       �	+k\Xc�A�*

loss�=��       �	z\Xc�A�*

loss��v=
��K       �	��\Xc�A�*

loss�I;=X[�       �	S{\Xc�A�*

loss�~k=+v�        �	\Xc�A�*

loss���=9�]       �	��\Xc�A�*

losss�&=W�Ǒ       �	x\Xc�A�*

loss8?�=���       �	-#\Xc�A�*

loss�]�=���)       �	��\Xc�A�*

loss��=�a��       �	7�\Xc�A�*

loss�2<�T�^       �	%Y\Xc�A�*

lossOѧ<��_�       �	g\Xc�A�*

lossI'�<*T�       �	��\Xc�A�*

lossy��=?�.       �	�� \Xc�A�*

lossD�=�OzF       �	4!\Xc�A�*

lossIS=f4!~       �	e�!\Xc�A�*

loss��=k��       �	�$\Xc�A�*

loss�I=7���       �	�_%\Xc�A�*

loss�<EH�       �	/&\Xc�A�*

loss�!�=H];O       �	��&\Xc�A�*

lossdA�=1&/�       �	\w'\Xc�A�*

loss���=0o�#       �	+P(\Xc�A�*

loss��<���       �	��(\Xc�A�*

loss?��=L+�b       �	x�)\Xc�A�*

loss][=�>��       �	��*\Xc�A�*

lossE�=b,{       �	r+\Xc�A�*

loss�ƽ=R�R�       �	-�,\Xc�A�*

lossH�<�       �	7o-\Xc�A�*

loss`Cb=�	~       �	V.\Xc�A�*

loss�@�=���       �	��.\Xc�A�*

loss�3�=�\wu       �	�_/\Xc�A�*

loss�@�=��%O       �	D�/\Xc�A�*

loss?��=��l        �	ɪ0\Xc�A�*

lossL��<�\��       �	Z1\Xc�A�*

loss��<&K�z       �	2\Xc�A�*

loss��>�)k�       �	��2\Xc�A�*

lossZ��=6\ 3       �	�z3\Xc�A�*

loss�3=	��       �	�4\Xc�A�*

loss���<k�؆       �	��4\Xc�A�*

loss��<q�r       �	ak5\Xc�A�*

loss�p<��k       �	6\Xc�A�*

lossj[>��[�       �	{�6\Xc�A�*

loss�E�=lM�       �	�a7\Xc�A�*

lossO�=Vy�Z       �		8\Xc�A�*

loss�  =ʋ>f       �	��8\Xc�A�*

loss�%=���       �	;R9\Xc�A�*

loss�*=����       �	�9\Xc�A�*

loss*��<C��7       �	ڐ:\Xc�A�*

loss�-�=�3�       �	-;\Xc�A�*

loss�<�j�p       �	��;\Xc�A�*

loss��; �C�       �	vm<\Xc�A�*

loss�&=P�l       �	�=\Xc�A�*

loss�=���       �	��=\Xc�A�*

lossE�<��i�       �	�C>\Xc�A�*

loss�n�==��>       �	$�>\Xc�A�*

loss�=Ce��       �	P�?\Xc�A�*

loss!��=���       �	I.@\Xc�A�*

loss߿0=�~<        �	��@\Xc�A�*

loss�h7=�S��       �	�A\Xc�A�*

loss�Kw="��=       �	<B\Xc�A�*

loss��>X       �	�B\Xc�A�*

loss��*=�=�D       �	�C\Xc�A�*

loss��=5�%       �	�9D\Xc�A�*

loss���;��G�       �	��D\Xc�A�*

loss��=0��.       �	��E\Xc�A�*

lossv�=��=       �	$&F\Xc�A�*

loss�A�=����       �	��F\Xc�A�*

loss��=5��       �	b�G\Xc�A�*

loss���=��       �	VaH\Xc�A�*

loss}w�=8V       �	�I\Xc�A�*

lossa�*<���       �	�TJ\Xc�A�*

lossw��<�9�T       �	
�J\Xc�A�*

loss)9�=�j1       �	șK\Xc�A�*

loss�w�<(�?�       �	�6L\Xc�A�*

loss� =8Vvh       �	�tM\Xc�A�*

loss�� >��+       �	4N\Xc�A�*

loss}6�=h���       �	B�N\Xc�A�*

loss�Q�=� ��       �	MjO\Xc�A�*

lossJMO>��c�       �	�P\Xc�A�*

lossT�=^�.)       �	3�P\Xc�A�*

loss��<�^�       �	cCQ\Xc�A�*

loss.�)=կ       �	<�Q\Xc�A�*

loss��>�8?�       �	�wR\Xc�A�*

loss�=����       �	
S\Xc�A�*

loss��v=1�z       �	��S\Xc�A�*

lossָ�=Y�       �	UT\Xc�A�*

loss<ĸ=�@��       �	��T\Xc�A�*

loss�O�<Y�       �	��U\Xc�A�*

loss�<�!>,       �	(-V\Xc�A�*

lossE'<<n+
�       �	u�V\Xc�A�*

loss��5=��Q�       �	]pW\Xc�A�*

loss�(�=#�5       �	tX\Xc�A�*

loss��=D>z�       �	m�X\Xc�A�*

loss3o~=m�r�       �	�MY\Xc�A�*

loss�T#>�p�a       �	k�Y\Xc�A�*

loss{=�=0�G       �	�Z\Xc�A�*

lossvp<���       �	C<[\Xc�A�*

loss���<�-       �	"�[\Xc�A�*

loss��=J`��       �	��\\Xc�A�*

loss�g�=2F��       �	�2]\Xc�A�*

lossl��<��	�       �	��]\Xc�A�*

lossU��=�޴�       �	�^\Xc�A�*

loss���=3��       �	�C_\Xc�A�*

loss��=n�        �	��_\Xc�A�*

lossl��=�4�M       �	��`\Xc�A�*

loss䛇=��]�       �	[Da\Xc�A�*

lossF!=�3�       �	�a\Xc�A�*

loss�=�O^       �	�b\Xc�A�*

lossAr>=���h       �	�Fc\Xc�A�*

lossx�r<��T�       �	!<d\Xc�A�*

loss��=�T       �	@�d\Xc�A�*

loss�
>���       �	Df\Xc�A�*

loss�:=Xl       �	�f\Xc�A�*

loss�U=�X܂       �	�rg\Xc�A�*

loss�>=���       �	�2h\Xc�A�*

loss�e<H.��       �	�2i\Xc�A�*

loss-��=���       �	3�i\Xc�A�*

loss�N�=� 8       �	�j\Xc�A�*

loss�w�=V|J�       �	�Zk\Xc�A�*

loss[ľ=��_�       �	�k\Xc�A�*

loss�U�=J�       �	�l\Xc�A�*

loss�T=���       �	�+m\Xc�A�*

loss��=.�p       �	��m\Xc�A�*

loss���=}�       �	7�n\Xc�A�*

lossV��<P�vc       �	�&o\Xc�A�*

lossΎ-=1���       �	<�o\Xc�A�*

loss���<��U       �	�q\Xc�A�*

loss&}�=��       �	��q\Xc�A�*

loss_��<�#��       �	oGr\Xc�A�*

loss���=����       �	q�r\Xc�A�*

loss��<~�-�       �	��s\Xc�A�*

loss!%=��`       �	�+t\Xc�A�*

lossg}=k^��       �	H�t\Xc�A�*

loss��4=�       �	>Yu\Xc�A�*

loss)��=HZ��       �	5^v\Xc�A�*

loss
ث=�fDM       �	��v\Xc�A�*

loss��=����       �	��w\Xc�A�*

loss[�u=݆�       �	J'x\Xc�A�*

loss2=#`��       �	��x\Xc�A�*

lossh=�r�.       �	�cy\Xc�A�*

loss���<S���       �	�z\Xc�A�*

lossH��<��m�       �	��z\Xc�A�*

loss��c=bB�v       �	�8{\Xc�A�*

lossR�R=��       �	0�{\Xc�A�*

loss�\M=`���       �	��|\Xc�A�*

loss͖�<p>��       �	d�}\Xc�A�*

lossq�=A��`       �	�-~\Xc�A�*

loss=�>=���C       �	��~\Xc�A�*

lossd6=�       �	/n\Xc�A�*

loss�D=���       �	`�\Xc�A�*

loss�W#=u���       �	Z��\Xc�A�*

loss�4�=p��       �	6�\Xc�A�*

loss�Y0>#�#o       �	gׁ\Xc�A�*

loss���;ǴrS       �	*o�\Xc�A�*

lossP=˥��       �	�\Xc�A�*

loss�u�<�(X       �	���\Xc�A�*

loss�>�=v(00       �	�J�\Xc�A�*

lossˁ=�.<�       �	-�\Xc�A�*

loss�P=aT�
       �	D��\Xc�A�*

lossE��=�m�       �	 '�\Xc�A�*

loss�^%=�ȝ�       �	Ȇ\Xc�A�*

loss��=� 7�       �	{�\Xc�A�*

lossR7�<���9       �	��\Xc�A�*

loss�6�=�~_�       �	
��\Xc�A�*

loss6pj=��}       �	�X�\Xc�A�*

lossC�)=�t*       �	<��\Xc�A�*

losso��=�m?3       �	|��\Xc�A�*

loss\��=�q��       �	S?�\Xc�A�*

lossn��=����       �	ZՌ\Xc�A�*

loss�J)=3#&�       �	o��\Xc�A�*

loss�^�=\?       �	�+�\Xc�A�*

loss��f=>��:       �	�Ǝ\Xc�A�*

loss,��=4F�L       �	Ȳ�\Xc�A�*

loss��<֠�       �	3��\Xc�A�*

loss�k�<� �       �	1��\Xc�A�*

loss@9�=���{       �	�Ԓ\Xc�A�*

lossI��=����       �	�m�\Xc�A�*

loss��<RU�       �	Ė�\Xc�A�*

loss`ߣ=0bB       �	�`�\Xc�A�*

loss�NK=��!B       �	Ii�\Xc�A�*

loss�%9>M���       �	(
�\Xc�A�*

lossW�=`��       �	!ʗ\Xc�A�*

loss�%>:���       �	C��\Xc�A�*

loss��9= K�!       �	�,�\Xc�A�*

loss(��=ɷ�       �	�ڙ\Xc�A�*

lossO �=�ظ        �	���\Xc�A�*

loss�uJ=�c��       �	b��\Xc�A�*

loss�zC=�0��       �	� �\Xc�A�*

loss�T
=y�%�       �	��\Xc�A�*

lossú3=�i&/       �	�;�\Xc�A�*

loss��>�z��       �	U��\Xc�A�*

lossD?�=J��       �	�\Xc�A�*

loss̬�=��S�       �	#��\Xc�A�*

loss�|9=��A       �	�V�\Xc�A�*

lossCe$=7�,       �	��\Xc�A�*

loss�C�<񯬋       �	���\Xc�A�*

loss�C�<٭�R       �	|+�\Xc�A�*

loss#E>	.�       �	`ʣ\Xc�A�*

lossJG�=�ҮS       �	Po�\Xc�A�*

loss� �=T��       �	��\Xc�A�*

loss�fN=��       �	;��\Xc�A�*

loss�n=m��       �	fJ�\Xc�A�*

lossL�=4�a\       �	��\Xc�A�*

lossa�$>��"       �	�*�\Xc�A�*

loss�0=��QZ       �	���\Xc�A�*

loss��Q=���       �	[�\Xc�A�*

lossm�=�Z�       �	g�\Xc�A�*

loss��=9���       �	��\Xc�A�*

loss��P=h�Մ       �	���\Xc�A�*

loss���;�F�       �	�9�\Xc�A�*

loss<�=���       �	jݬ\Xc�A�*

loss�g->ؠ�       �	U��\Xc�A�*

lossϏQ=�>Y       �	��\Xc�A�*

loss��=YXu       �	��\Xc�A�*

loss��'=�s�       �	ƨ�\Xc�A�*

loss��=�Q�       �	�K�\Xc�A�*

loss�ԟ=�PfQ       �	:�\Xc�A�*

loss���<h��       �	!��\Xc�A�*

loss�;�<��Y       �	n3�\Xc�A�*

lossv�<�k       �	*ɲ\Xc�A�*

lossA�=�܆�       �	�h�\Xc�A�*

loss���<z��R       �	��\Xc�A�*

lossat�<19S       �	���\Xc�A�*

lossqAU=�o��       �	~V�\Xc�A�*

loss(=�C/!       �	��\Xc�A�*

loss�RA=.���       �	��\Xc�A�*

lossm3�;�Jf�       �	�9�\Xc�A�*

loss��=�oU�       �	�ڷ\Xc�A�*

loss}�=���       �	dw�\Xc�A�*

loss���=Ŀ:`       �	��\Xc�A�*

loss�R�=�O�j       �	���\Xc�A�*

loss�A�<0k�*       �	O�\Xc�A�*

loss&ny<8�Wv       �	�\Xc�A�*

loss��!=���       �	��\Xc�A�*

lossZ��<䬤J       �	+2�\Xc�A�*

loss,�E<;��       �	�Ӽ\Xc�A�*

loss��>�zoj       �	Ox�\Xc�A�*

lossZG(>,��       �	��\Xc�A�*

loss^��=�y�       �	v��\Xc�A�*

lossn�=iI��       �	C�\Xc�A�*

loss9ޒ=�c�       �	d�\Xc�A�*

loss���=Y>�       �	��\Xc�A�*

loss�,=&+R       �	�,�\Xc�A�*

lossy=�<!�       �	��\Xc�A�*

lossv�=����       �	�\Xc�A�*

lossv*m<Q[~�       �	�\Xc�A�*

lossld,>�I��       �		��\Xc�A�*

lossH��=��jQ       �	�c�\Xc�A�*

loss:�=_�R       �	��\Xc�A�*

loss��~<�5U       �	Թ�\Xc�A�*

loss �=I�|       �	�e�\Xc�A�*

lossب=CC�       �	t�\Xc�A�*

loss�$=Ŝ��       �	���\Xc�A�*

loss��{=���p       �	�\�\Xc�A�*

loss�=�65       �	�	�\Xc�A�*

loss6�L=�       �	��\Xc�A�*

loss�	�=%�*k       �	7��\Xc�A�*

loss�'�=d0�       �	"O�\Xc�A�*

lossA�~<l�,�       �	���\Xc�A�*

lossJ?=�_�       �	���\Xc�A�*

loss4�<Ĵt.       �	�<�\Xc�A�*

loss��=<p��       �	Q3�\Xc�A�*

loss��<#��T       �	n�\Xc�A�*

loss�Ku=�|�X       �	���\Xc�A�*

loss�P�<��s       �	���\Xc�A�*

loss�A�=�<       �	w��\Xc�A�*

losseN�=�r62       �	_F�\Xc�A�*

loss*�l<���G       �	3��\Xc�A�*

loss��(=�+��       �	���\Xc�A�*

losstu,<8��       �	d��\Xc�A�*

loss�r�=C�Y�       �	�o�\Xc�A�*

loss�$=�4Z`       �	�
�\Xc�A�*

lossԈ�=ޚ��       �	P��\Xc�A�*

loss��
=ʙ��       �	5b�\Xc�A�*

loss�@=#~�       �	S�\Xc�A�*

loss@	�<8�       �	C��\Xc�A�*

loss��-=w�=       �	xD�\Xc�A�*

loss�_�<:ɱ�       �	C��\Xc�A�*

losssu3=�)]H       �	||�\Xc�A�*

lossnI�:QX��       �	��\Xc�A�*

loss�.�<䘁�       �	;��\Xc�A�*

loss7�\=�O       �	�`�\Xc�A�*

lossW��<M^ڔ       �	���\Xc�A�*

loss��<<���       �	���\Xc�A�*

losshL�=۳+       �	:�\Xc�A�*

lossh�s=@�F       �	���\Xc�A�*

loss2��<��       �	Xo�\Xc�A�*

loss;Rw=��       �	,�\Xc�A�*

loss\�K=ڗst       �	7��\Xc�A�*

loss�HW=y�R.       �	xG�\Xc�A�*

loss�Ѣ;O�S�       �	)��\Xc�A�*

loss-��<t3��       �	���\Xc�A�*

loss��;Ҡ�       �	%!�\Xc�A�*

loss:��;y0 2       �	��\Xc�A�*

loss�Ka=N        �	y]�\Xc�A�*

loss�f�<���}       �	�1�\Xc�A�*

loss/.�<�q3f       �	���\Xc�A�*

loss�%�<ǹo6       �	�x�\Xc�A�*

loss��	:��J       �	\�\Xc�A�*

lossݫ:��ć       �	_��\Xc�A�*

lossqL<����       �	[�\Xc�A�*

loss�|=�ެ�       �	���\Xc�A�*

loss���<E��       �	>��\Xc�A�*

loss��)<(Y��       �	�7�\Xc�A�*

loss���<W�       �	���\Xc�A�*

loss���>��݋       �	�q�\Xc�A�*

loss�6m<̪c�       �	��\Xc�A�*

loss��=a�h       �	���\Xc�A�*

lossSߜ=�4�(       �	_�\Xc�A�	*

lossx�>�J��       �	��\Xc�A�	*

loss��=�˧       �	(��\Xc�A�	*

loss7֦<�T       �	�_�\Xc�A�	*

lossc�=��{       �	:�\Xc�A�	*

loss �>>z9`�       �	���\Xc�A�	*

lossR07=���       �	�<�\Xc�A�	*

loss1vF=�Z�-       �	P��\Xc�A�	*

loss<[=��Ȁ       �	���\Xc�A�	*

loss���=<�7�       �	�.�\Xc�A�	*

loss��>���       �	���\Xc�A�	*

lossn0k=D�Cj       �	Mj�\Xc�A�	*

loss;G=Ë�       �	�\Xc�A�	*

loss��=6mB       �	���\Xc�A�	*

loss�K=.X;1       �	�d�\Xc�A�	*

lossl+�=$�UC       �	� �\Xc�A�	*

loss�>=���       �	4��\Xc�A�	*

loss��C=&��       �	�C�\Xc�A�	*

loss�M<X2.�       �	��\Xc�A�	*

loss�T|=aֵ�       �	z�\Xc�A�	*

loss��>hc=       �	��\Xc�A�	*

lossq�0=��bA       �	���\Xc�A�	*

lossd�<4�	�       �	�H�\Xc�A�	*

loss	�;�_��       �	���\Xc�A�	*

loss��\=�Mi�       �	�z�\Xc�A�	*

lossz%=p�R�       �	��\Xc�A�	*

loss��<=����       �	��\Xc�A�	*

loss;R�=�]�2       �	EK�\Xc�A�	*

lossV�x=LU��       �	q��\Xc�A�	*

loss� >܃1�       �	��\Xc�A�	*

loss�
�<�GB�       �	� ]Xc�A�	*

loss_��;����       �	4� ]Xc�A�	*

loss�Gq=����       �	�`]Xc�A�	*

lossW+<4�֢       �	�]Xc�A�	*

loss�<)R�       �	ɓ]Xc�A�	*

loss�Z�=����       �	�']Xc�A�	*

loss���=�p�B       �	��]Xc�A�	*

loss�sz=I��U       �	�Y]Xc�A�	*

loss��<��]"       �	��]Xc�A�	*

loss
�U=���       �	��]Xc�A�	*

lossQ=�u       �	�$]Xc�A�	*

loss�P�=�U�1       �	��]Xc�A�	*

loss8*K<*KOS       �	�h]Xc�A�	*

loss)Ѭ=���]       �	�]Xc�A�	*

lossd�>��m�       �	ɭ]Xc�A�	*

loss��-=b       �	�L	]Xc�A�	*

loss�X=}�       �	��	]Xc�A�	*

lossck�<U�̀       �	j�
]Xc�A�	*

lossC&=�D6       �	K=]Xc�A�	*

loss�	�=����       �	�+]Xc�A�	*

lossf��=�׿       �	�+]Xc�A�	*

loss�N>pF�h       �	\<,]Xc�A�	*

loss�=vN��       �	��,]Xc�A�	*

loss��`=P��       �	
�-]Xc�A�	*

loss�ҭ<�,       �	�.]Xc�A�	*

loss3�h=�f�G       �	�.]Xc�A�	*

loss&��=���a       �	�?/]Xc�A�	*

loss��=�q       �	1�/]Xc�A�	*

loss� |=�#�       �	�l0]Xc�A�	*

loss��<~NL       �	t1]Xc�A�	*

loss�=�雰       �	��1]Xc�A�	*

loss��4=�{       �	(I2]Xc�A�	*

loss�r=�� �       �	��2]Xc�A�	*

loss�$H=�r�g       �	Z�3]Xc�A�	*

loss��=�jT+       �	f/4]Xc�A�	*

lossn[=���.       �	��4]Xc�A�	*

loss��.=S�       �	N}5]Xc�A�	*

loss�)*=�H9       �	�*6]Xc�A�	*

loss�	?>=�֛       �	��6]Xc�A�	*

loss��=��S       �	��7]Xc�A�	*

loss��=���/       �	�#8]Xc�A�	*

loss;��<�+i       �	�8]Xc�A�	*

loss `=�B�q       �	+j9]Xc�A�	*

loss\L�<&j�W       �		T:]Xc�A�	*

loss���<�5Z       �	{�:]Xc�A�	*

loss�2=����       �	��;]Xc�A�	*

loss #�<�7f       �	�U<]Xc�A�	*

lossJv=���
       �	��<]Xc�A�	*

loss�I~=��(       �	\�=]Xc�A�	*

loss
�=�D�       �	�T>]Xc�A�	*

loss�CC=�E�       �	��>]Xc�A�	*

loss|Db=�Y�       �	�?]Xc�A�	*

loss�n�=��=�       �	�A@]Xc�A�	*

loss.�t=�%��       �	��@]Xc�A�	*

loss�b�=�4Q       �	�A]Xc�A�	*

loss�I+<i�ZY       �	)%B]Xc�A�	*

loss�i�=�L��       �	;�B]Xc�A�	*

lossD�>#��
       �	�`C]Xc�A�	*

lossD��=���       �	&�C]Xc�A�	*

lossa�R=(��       �	l�D]Xc�A�	*

loss�f=9�2Z       �	�3E]Xc�A�	*

lossh�<2ê       �	%�E]Xc�A�	*

lossI�<-*�k       �	�iF]Xc�A�	*

loss�3t<��       �	G]Xc�A�	*

loss%k}=���P       �	s�G]Xc�A�	*

loss�e�=Q�T�       �	WAH]Xc�A�	*

lossQ��<xls�       �	��H]Xc�A�	*

lossZ��<��2�       �	�rI]Xc�A�	*

lossN<��ջ       �	rJ]Xc�A�	*

loss�H�;D�4�       �	B�J]Xc�A�	*

lossl�G=��3       �	�QK]Xc�A�	*

loss�],<�k�       �	��K]Xc�A�	*

loss�>�;&       �	�L]Xc�A�	*

loss���<�I��       �	]M]Xc�A�	*

loss ��;��%D       �	�N]Xc�A�	*

loss�S�<]B��       �	q�N]Xc�A�	*

lossŠ�;;�D       �	�^O]Xc�A�	*

loss��=�|��       �	R�P]Xc�A�	*

loss���=;�<       �	̖Q]Xc�A�	*

lossz̝=��U`       �	��R]Xc�A�	*

loss�ّ=�0       �	��S]Xc�A�	*

loss���<r.�       �	&�T]Xc�A�	*

loss�5�=}s}�       �	g�U]Xc�A�	*

loss[��=�ݛ5       �	ܡV]Xc�A�	*

lossS�<y��       �	]OW]Xc�A�	*

lossEf�<'}t       �	�/X]Xc�A�	*

loss��<|��       �	o�X]Xc�A�	*

lossh'�=�!�       �	�Z]Xc�A�	*

loss /�='@�2       �	[]Xc�A�	*

loss�wV=_Zv       �	\]Xc�A�	*

loss܉�=>\�       �	9�\]Xc�A�	*

loss[Lj<�E*p       �	 c]]Xc�A�	*

loss���=5��r       �	�^]Xc�A�	*

loss;�=�h�       �	�^]Xc�A�	*

lossN� <}8!.       �	U_]Xc�A�	*

lossA��=�� m       �	�`]Xc�A�	*

loss�,�<}'8A       �		�`]Xc�A�	*

loss1�=>�J%       �	�Ya]Xc�A�	*

loss��Q=\(�       �	�b]Xc�A�
*

lossc�p=�>��       �	A�b]Xc�A�
*

loss1�=pA��       �	XYc]Xc�A�
*

lossq�p='P�       �	d]Xc�A�
*

loss�I_<��$       �	/�d]Xc�A�
*

loss��<��+       �	�He]Xc�A�
*

loss3�
=U45�       �	��e]Xc�A�
*

lossv�h=��b       �	��f]Xc�A�
*

lossfMV=DC�8       �	32g]Xc�A�
*

lossɉ)=4��       �	��g]Xc�A�
*

loss���;ID�}       �	�wh]Xc�A�
*

loss�=E0�7       �	�i]Xc�A�
*

loss��=��N�       �	��i]Xc�A�
*

lossl�n=>щ6       �	_`j]Xc�A�
*

loss2V=���#       �	�k]Xc�A�
*

lossݤ�=����       �	q�k]Xc�A�
*

loss���;��k�       �	�Ll]Xc�A�
*

loss��\=-       �	��l]Xc�A�
*

loss�E�;}!v       �	��m]Xc�A�
*

lossϫ�=��8�       �	�'n]Xc�A�
*

loss��=`>(       �	�n]Xc�A�
*

loss2�W<�"�9       �	^co]Xc�A�
*

loss;d}<���s       �	Vp]Xc�A�
*

loss���=9��       �	�p]Xc�A�
*

loss�j<���i       �	�Bq]Xc�A�
*

loss�(�<Wz�`       �	��q]Xc�A�
*

loss�hc=��d       �	�zr]Xc�A�
*

lossԬ/<A��a       �	fs]Xc�A�
*

loss���<~��       �	\�s]Xc�A�
*

losso3>f�       �	{Kt]Xc�A�
*

loss<9=�To        �	��t]Xc�A�
*

loss6b�=��ײ       �	~u]Xc�A�
*

loss`+=���       �	�v]Xc�A�
*

loss?��=��)�       �	��v]Xc�A�
*

loss��<��[#       �	�Nw]Xc�A�
*

loss5�<�pp�       �	��w]Xc�A�
*

loss�d:=�Hk�       �	��x]Xc�A�
*

loss��< @�       �	�-y]Xc�A�
*

loss�]!=,��       �	V�z]Xc�A�
*

loss�*�<l3       �	�1{]Xc�A�
*

loss$Y�=�~       �	c�{]Xc�A�
*

lossʙ=��       �	Sy|]Xc�A�
*

lossQ��=&�.       �	?}]Xc�A�
*

losso��=zހD       �	A�}]Xc�A�
*

loss��=�s��       �	�c~]Xc�A�
*

losslC;=^�X�       �	�]Xc�A�
*

loss�ݬ<#U��       �	��]Xc�A�
*

lossS��<.�o�       �	�P�]Xc�A�
*

loss_�<��z�       �	���]Xc�A�
*

loss#��=@=�       �	m��]Xc�A�
*

lossjȄ=�BN�       �	�Y�]Xc�A�
*

loss��2=P���       �	  �]Xc�A�
*

loss#�E=,ܾ�       �	���]Xc�A�
*

lossN�-=��y       �	�H�]Xc�A�
*

loss^w=�{0       �	p�]Xc�A�
*

lossv��=���       �	\��]Xc�A�
*

loss ^�<C���       �	2>�]Xc�A�
*

lossN�%=}�W        �	;�]Xc�A�
*

loss�+�;�]�       �	ρ�]Xc�A�
*

lossA�J=Z?.       �	>&�]Xc�A�
*

loss#-�<�|z�       �	�Έ]Xc�A�
*

lossnT";��       �	���]Xc�A�
*

lossWC�<>��       �	�\�]Xc�A�
*

lossB7�=���       �	�	�]Xc�A�
*

loss���=�f�J       �	;�]Xc�A�
*

loss��U=��k-       �	俍]Xc�A�
*

loss�֪<�;�       �	p�]Xc�A�
*

loss;��=�'�       �	�
�]Xc�A�
*

loss��<��E       �	"��]Xc�A�
*

lossOo�<΍�t       �	��]Xc�A�
*

loss.J=�@V       �	f��]Xc�A�
*

loss�5�<9hu�       �	�_�]Xc�A�
*

loss��F=����       �	@i�]Xc�A�
*

loss��=d�A�       �	B�]Xc�A�
*

lossZ�=O#I       �	4�]Xc�A�
*

loss���<�
��       �	���]Xc�A�
*

loss�e�<w�^�       �	��]Xc�A�
*

loss.�<r��       �	�c�]Xc�A�
*

loss�#�=
�9�       �	���]Xc�A�
*

loss*$>��.�       �	$��]Xc�A�
*

lossݢ�<��m       �	Nc�]Xc�A�
*

loss3�=)�v�       �	A�]Xc�A�
*

loss�e=a�        �	��]Xc�A�
*

loss!�>=S��       �	#��]Xc�A�
*

lossAi�;XJf*       �	�i�]Xc�A�
*

loss�8=ܷT�       �	R|�]Xc�A�
*

lossJhB=	D�.       �	��]Xc�A�
*

lossr��;��{w       �	̳�]Xc�A�
*

loss1�j=X�1       �	�u�]Xc�A�
*

loss�J�<��Ǒ       �	�S�]Xc�A�
*

lossN��=p^Qb       �	��]Xc�A�
*

lossC&�=�Lp       �	>��]Xc�A�
*

loss�$�;ţ�       �	ຣ]Xc�A�
*

lossRxK<��4�       �	P�]Xc�A�
*

loss�j=�V/�       �	5�]Xc�A�
*

loss8�k;�	F�       �	٥]Xc�A�
*

loss�7�<���       �	}�]Xc�A�
*

loss1��<�"~�       �	��]Xc�A�
*

lossL�=�h��       �	�§]Xc�A�
*

loss��9=� t�       �	�h�]Xc�A�
*

loss���<���       �	��]Xc�A�
*

loss��;���       �	���]Xc�A�
*

lossD�=���v       �	fO�]Xc�A�
*

lossN�o=pT�2       �	J�]Xc�A�
*

loss��<�Kt       �	?��]Xc�A�
*

loss���<�       �	�,�]Xc�A�
*

loss�<�J��       �	�ͬ]Xc�A�
*

loss��<���       �	�l�]Xc�A�
*

loss[h�<�� G       �	&�]Xc�A�
*

lossm	�='Ԑ{       �	 Ů]Xc�A�
*

loss�=�tD       �	b�]Xc�A�
*

loss))�=�+0@       �	��]Xc�A�
*

loss$�<�8Jp       �	�Z�]Xc�A�
*

loss��:=sGW       �	���]Xc�A�
*

losslB�<�F��       �	[��]Xc�A�
*

loss���<L��z       �	g,�]Xc�A�
*

loss���<��       �	H³]Xc�A�
*

loss���<��       �	W[�]Xc�A�
*

loss]"=TD�       �	��]Xc�A�
*

loss=�g=���?       �	���]Xc�A�
*

lossc�-=��'�       �	��]Xc�A�
*

loss�ܕ==��       �	'��]Xc�A�
*

loss��}=��>:       �	>[�]Xc�A�
*

loss���;�-;�       �	|�]Xc�A�
*

lossE��<�k�%       �	���]Xc�A�
*

lossw��<]`�-       �	�)�]Xc�A�
*

loss��<*1��       �	�ù]Xc�A�
*

loss�=�<�t       �	�_�]Xc�A�
*

loss��5=����       �	v��]Xc�A�*

loss�F==�v�       �	9��]Xc�A�*

loss�==}~��       �	�/�]Xc�A�*

loss��<}�2=       �	&ȼ]Xc�A�*

loss��<��z�       �	�_�]Xc�A�*

loss�>e<�ݮ�       �	��]Xc�A�*

loss�b<)��       �	���]Xc�A�*

loss��>�ʙ�       �	R'�]Xc�A�*

lossóO=�3�       �	�ǿ]Xc�A�*

loss��=^���       �	�^�]Xc�A�*

loss[˖>WT��       �	
��]Xc�A�*

loss`k�=?�]�       �	*��]Xc�A�*

loss2�E> |2/       �	v4�]Xc�A�*

loss�==u�       �	���]Xc�A�*

lossH��;Q0�       �	f�]Xc�A�*

loss�m"<z{�K       �	�	�]Xc�A�*

loss�F�=���       �	h��]Xc�A�*

lossX�<�s7�       �	PU�]Xc�A�*

loss�S=�)�6       �	��]Xc�A�*

lossHK�<���       �	͑�]Xc�A�*

loss]sS=��_]       �	�,�]Xc�A�*

lossH�C=�؟�       �	��]Xc�A�*

loss��t=��n,       �	k�]Xc�A�*

lossf��<���L       �	-	�]Xc�A�*

loss�V�<vN�       �	��]Xc�A�*

lossj}*=h���       �	�?�]Xc�A�*

loss�8�<���$       �	���]Xc�A�*

loss��<����       �	*r�]Xc�A�*

loss�=����       �	�	�]Xc�A�*

loss�>�<���g       �	g��]Xc�A�*

loss!B=����       �	y>�]Xc�A�*

loss�$�=�,�       �	��]Xc�A�*

losss�<~�       �	m�]Xc�A�*

loss��=dƚ       �	R�]Xc�A�*

loss��	=�C�       �	���]Xc�A�*

loss�iQ=��       �	���]Xc�A�*

lossi"�<s~"�       �	���]Xc�A�*

loss��$=�Nz       �	JB�]Xc�A�*

loss��=S��-       �	6x�]Xc�A�*

loss���=�7��       �	B?�]Xc�A�*

lossN�G=�h       �	i��]Xc�A�*

lossG�=!w�m       �	�;�]Xc�A�*

loss��<���       �	(��]Xc�A�*

loss$�=JT��       �	�s�]Xc�A�*

lossm>X���       �	iW�]Xc�A�*

loss�ֈ=�;��       �	N	�]Xc�A�*

loss�<����       �	e��]Xc�A�*

loss�G�<[�J"       �	��]Xc�A�*

loss�u�<~Js       �	���]Xc�A�*

loss̤�<���t       �	��]Xc�A�*

loss�b�<t�C       �	�5�]Xc�A�*

loss�@=̶ >       �	j�]Xc�A�*

loss�x=zU��       �	�4�]Xc�A�*

loss�1G=�:�D       �	p`�]Xc�A�*

loss�5_=e�1�       �	=H�]Xc�A�*

lossC3=~�z       �	� �]Xc�A�*

loss!�=��}       �	���]Xc�A�*

loss�F�=�-�       �	R~�]Xc�A�*

loss��A<� e       �	k�]Xc�A�*

loss�>=��.       �	��]Xc�A�*

loss�a)=��f8       �	���]Xc�A�*

loss��=��v�       �	�X�]Xc�A�*

loss�I,=����       �	���]Xc�A�*

loss=��=       �	���]Xc�A�*

loss���=!��       �	���]Xc�A�*

loss�a�=���       �	���]Xc�A�*

lossc�y<<U�E       �	G�]Xc�A�*

loss%�=�`2       �	���]Xc�A�*

lossyR>�l�       �	X�]Xc�A�*

lossj�;�~�       �	���]Xc�A�*

loss?�<�mv�       �	��]Xc�A�*

loss�r�=����       �	�$�]Xc�A�*

lossZH�=��Z       �	���]Xc�A�*

loss��$<+y�       �	�\�]Xc�A�*

lossj9�;8�H�       �	���]Xc�A�*

lossqF<8[�N       �	N��]Xc�A�*

lossMB�<"�f�       �	�.�]Xc�A�*

loss�˩=��F`       �	���]Xc�A�*

loss�j�=�}#�       �	9|�]Xc�A�*

loss)
k=��e�       �	��]Xc�A�*

loss!�=.���       �	���]Xc�A�*

loss�%=ĮÎ       �	�!�]Xc�A�*

lossA_�;�_X       �	���]Xc�A�*

loss�6d;Uc�e       �	�$�]Xc�A�*

loss�Q>=&Ub�       �	T��]Xc�A�*

loss<�<��*       �	�p�]Xc�A�*

loss)T<C���       �	��]Xc�A�*

lossE�< J.�       �	��]Xc�A�*

loss1��<���       �	�`�]Xc�A�*

lossd�-<FȐ|       �	.��]Xc�A�*

lossM�=�{X�       �	���]Xc�A�*

loss���=~�ʪ       �	/0�]Xc�A�*

loss�A%=�^�q       �	���]Xc�A�*

loss�j=���5       �	�d ^Xc�A�*

loss&-,=��e�       �	`^Xc�A�*

loss��;�ѩc       �	�^Xc�A�*

loss�M�=�ڙ       �	�<^Xc�A�*

lossH�9=-!       �	��^Xc�A�*

loss���=����       �	�z^Xc�A�*

loss$��<�T�       �	P^Xc�A�*

loss�R�<���       �	V�^Xc�A�*

lossA�$=�ǒ�       �	]O^Xc�A�*

loss�+%<���n       �	�^Xc�A�*

loss��M=��r6       �	N�^Xc�A�*

loss�!=�I\Q       �	�^Xc�A�*

loss (3=q��9       �	�*^Xc�A�*

loss7Փ<U̳�       �	��^Xc�A�*

lossoȔ;w-�N       �	�c	^Xc�A�*

loss4=��       �	��	^Xc�A�*

loss��<�9	       �	l�
^Xc�A�*

loss@մ<��       �	7^Xc�A�*

loss^g#=33��       �	J�^Xc�A�*

lossVR(<�       �	�u^Xc�A�*

loss���<�А`       �	^Xc�A�*

loss7��=��GS       �	�^Xc�A�*

lossh7=_(�       �	b^Xc�A�*

loss�+�<!�@       �	M^Xc�A�*

lossse+=� )       �	/^Xc�A�*

loss��h=�<,       �	��^Xc�A�*

loss�gR=�$R       �	qq^Xc�A�*

loss�*�;�q�       �	^Xc�A�*

loss�e�<d�@�       �	��^Xc�A�*

loss��=�Y�       �	�^Xc�A�*

lossW�J<)�       �	�b^Xc�A�*

loss��	=�Z�       �	C^Xc�A�*

loss}�'<�C<       �	�^Xc�A�*

loss��)=Ե@y       �	�7^Xc�A�*

loss�b>%��       �	}�^Xc�A�*

loss�=�ZS       �	d^Xc�A�*

loss��=<���       �	�^Xc�A�*

loss�W�<F�d       �	f�^Xc�A�*

loss_}�=��f       �	�;^Xc�A�*

lossv$J;��d       �	5�^Xc�A�*

lossf2<���        �	��^Xc�A�*

loss!��<�e �       �	�%^Xc�A�*

loss�̢=o�       �	Ժ^Xc�A�*

loss��z=�@
=       �	�t^Xc�A�*

losss��=�(u       �	F^Xc�A�*

lossd �=��/I       �	Χ^Xc�A�*

lossRh<q�
�       �	:^Xc�A�*

lossq�=;���       �	��^Xc�A�*

loss�UP=�L�E       �	Zf^Xc�A�*

lossqP�=lc�       �	��^Xc�A�*

lossw�;��}O       �	G� ^Xc�A�*

loss	^*=��p       �	}!!^Xc�A�*

loss�=x�"       �	7�!^Xc�A�*

lossa�1=R�6       �	�^"^Xc�A�*

loss}^�=���       �	!#^Xc�A�*

lossO��;�.�       �	��#^Xc�A�*

loss�a�;���       �	5D$^Xc�A�*

loss��2<l�H�       �	��$^Xc�A�*

loss$W�=Њ�x       �	�r%^Xc�A�*

loss�Q=���       �	&^Xc�A�*

loss=�==�{y       �	��&^Xc�A�*

lossWO�;����       �	K\(^Xc�A�*

loss�e�<9QȰ       �	t)^Xc�A�*

loss��&>����       �	J�)^Xc�A�*

loss�(=�\l�       �	�f*^Xc�A�*

loss[�=�q       �	�+^Xc�A�*

lossذL=߯�|       �	ӽ+^Xc�A�*

lossՅ<Փ�A       �	d,^Xc�A�*

lossz�R=�/�       �	�-^Xc�A�*

lossT�<\�D       �	R�-^Xc�A�*

loss��C<�A�w       �	�`.^Xc�A�*

lossW��;���       �	
/^Xc�A�*

lossC.<��/       �	J�/^Xc�A�*

loss�3�</�{m       �	�`0^Xc�A�*

loss��@=����       �	�1^Xc�A�*

loss�ɰ=�|       �	I2^Xc�A�*

loss	֊<%;�l       �	y�2^Xc�A�*

loss{�=Q��       �	�3^Xc�A�*

loss?�R=���       �	�*4^Xc�A�*

loss(x,=wIR       �	��4^Xc�A�*

loss�=��*E       �	�v5^Xc�A�*

lossěl=��+V       �	�7^Xc�A�*

loss�j{=S��       �	2�7^Xc�A�*

loss�1<%�C       �	X8^Xc�A�*

loss�U�;�       �	��8^Xc�A�*

loss���=�M       �	^�9^Xc�A�*

lossTA�=��Ƶ       �	[?:^Xc�A�*

loss�\6=��F       �	��:^Xc�A�*

lossmH=f��       �	��;^Xc�A�*

loss:=�B0�       �	�o<^Xc�A�*

lossQU�<��Z�       �	�=^Xc�A�*

loss�l-=��u!       �	$|>^Xc�A�*

loss�~%=XA#       �	�?^Xc�A�*

loss�V�<3��5       �	J�?^Xc�A�*

loss��<���       �	"O@^Xc�A�*

loss�^�=�m��       �	k,A^Xc�A�*

loss���:�g7       �	��A^Xc�A�*

lossc�<E˭       �	N~B^Xc�A�*

lossY<�CC       �	C^Xc�A�*

loss�Y/=���       �	��C^Xc�A�*

loss\�=!3�       �	�UD^Xc�A�*

lossj�B=�aP�       �	y�D^Xc�A�*

loss��a=��n       �	��E^Xc�A�*

loss��<M�T       �	�#F^Xc�A�*

loss���=�C�       �	��F^Xc�A�*

loss���=��p       �	�\G^Xc�A�*

loss�D�<8�ME       �	HH^Xc�A�*

loss7e�=$ ��       �	�H^Xc�A�*

loss���;�.�       �	�]I^Xc�A�*

loss6=�D�       �	;�I^Xc�A�*

loss�{=�r��       �	_�J^Xc�A�*

loss�>=9��I       �	j0K^Xc�A�*

loss\W�<��       �	��K^Xc�A�*

loss�t >���       �	�cL^Xc�A�*

lossP�=;ֺ       �	�L^Xc�A�*

loss��<A���       �	�M^Xc�A�*

lossR�;�#T�       �	�.N^Xc�A�*

loss#>�<���       �	&�N^Xc�A�*

loss�={=YA��       �	t]O^Xc�A�*

loss1V�<p��       �	��O^Xc�A�*

losshй<�n+�       �	��P^Xc�A�*

loss���=��sL       �	�Q^Xc�A�*

loss�� <�p>*       �	�R^Xc�A�*

loss��|=k��q       �	�R^Xc�A�*

loss%E�=���       �	p[S^Xc�A�*

loss��=lF!       �	
�S^Xc�A�*

loss$��=N��E       �	ɓT^Xc�A�*

loss=�"b       �	�)U^Xc�A�*

loss�~�<.F       �	��U^Xc�A�*

loss�B�;��$(       �	0dV^Xc�A�*

lossc�<�H�       �	W^Xc�A�*

loss�5�<��c       �	��W^Xc�A�*

loss���;���       �	�)X^Xc�A�*

loss�=fޏ=       �	�X^Xc�A�*

loss��=��.|       �	^�Y^Xc�A�*

lossv�
=<H       �	&Z^Xc�A�*

loss�:=)�U!       �	��Z^Xc�A�*

loss��c=�d2�       �	)^[^Xc�A�*

loss��;5�9       �	��[^Xc�A�*

loss��<W/�       �	9�\^Xc�A�*

loss6�h=:Y�       �	�H]^Xc�A�*

lossh~�=�h�       �	��]^Xc�A�*

losskc�=���f       �	.s^^Xc�A�*

loss{,�=V� 7       �	�_^Xc�A�*

loss�1.<���&       �	c�_^Xc�A�*

loss�ӣ<��       �	�\`^Xc�A�*

loss}�>[w�       �	U�`^Xc�A�*

loss�Y�<�M��       �	�a^Xc�A�*

loss�N�=v���       �	>Bb^Xc�A�*

loss���=1E�6       �	��b^Xc�A�*

loss�5�=6�C       �	��c^Xc�A�*

loss8�H<��Ǖ       �	�,d^Xc�A�*

loss�,�<�)�       �	��d^Xc�A�*

lossI�x=�F�w       �	ue^Xc�A�*

loss���=)G       �	=f^Xc�A�*

loss�{R=�EU�       �	��f^Xc�A�*

lossŲ#=�6��       �	BBg^Xc�A�*

lossz��<ה�=       �	�Yh^Xc�A�*

loss|�<�l9       �	��h^Xc�A�*

loss�X=j��       �	��i^Xc�A�*

loss2�<�aP�       �	�aj^Xc�A�*

lossh��;��&       �	]�j^Xc�A�*

lossz?�<�hX        �	��k^Xc�A�*

loss-w='�>       �	�pl^Xc�A�*

lossd��<�8w2       �	h	m^Xc�A�*

loss1M=����       �	�Gn^Xc�A�*

loss�#�<�[�       �	@�n^Xc�A�*

loss�N?;�~�       �	so^Xc�A�*

lossI<u|B       �	�p^Xc�A�*

lossع<��0|       �	��p^Xc�A�*

lossi�=Uޒ�       �	\Tq^Xc�A�*

loss�6<���       �	+�q^Xc�A�*

loss�!=	jv�       �	�r^Xc�A�*

loss�G�=���I       �	�=s^Xc�A�*

lossc�=3�m       �	�s^Xc�A�*

lossm-)<\�?       �	Dnt^Xc�A�*

loss�w�;���       �	�u^Xc�A�*

loss��=�D�z       �	��u^Xc�A�*

loss
<���       �	�Dv^Xc�A�*

loss��=G���       �	m�v^Xc�A�*

loss�05>��,       �	@�w^Xc�A�*

loss�J�=�t�1       �	(x^Xc�A�*

lossF�<�vK\       �	�x^Xc�A�*

loss~>[Q       �	�ey^Xc�A�*

loss�=9��       �	:#z^Xc�A�*

loss���<��8J       �	��z^Xc�A�*

loss�x(<sx�X       �	|b{^Xc�A�*

loss�m+=S�`�       �	N|^Xc�A�*

loss�^"=���:       �	v�|^Xc�A�*

loss���<L��       �	�:}^Xc�A�*

loss܏=�Ko       �	 �}^Xc�A�*

loss2_=�2TE       �	bh~^Xc�A�*

loss{�; �kl       �	G^Xc�A�*

loss_P=�/+       �	0�^Xc�A�*

loss_0�<�Zf       �	~8�^Xc�A�*

loss�/=i���       �	pр^Xc�A�*

loss��@=sx�[       �	Pn�^Xc�A�*

loss�iO<}f�       �	��^Xc�A�*

loss���<��:�       �	��^Xc�A�*

loss��G=C��O       �	/�^Xc�A�*

loss��`=��n       �		ă^Xc�A�*

loss��<B0D       �	>Y�^Xc�A�*

loss�5= ���       �	��^Xc�A�*

loss%�=<' I-       �	���^Xc�A�*

loss��O=�Rvj       �	%?�^Xc�A�*

loss�ǎ<���       �	�چ^Xc�A�*

loss�>*=h�       �	��^Xc�A�*

loss��<	�c,       �	��^Xc�A�*

loss�n=�A'       �	+�^Xc�A�*

lossJUP=0|��       �	���^Xc�A�*

loss�<{=�       �	�R�^Xc�A�*

loss�=_��       �	s�^Xc�A�*

loss�A�<��+�       �	���^Xc�A�*

loss��=|�       �	�+�^Xc�A�*

lossd��<���}       �	
e�^Xc�A�*

loss�`�=Yb�a       �	���^Xc�A�*

losstS(=�"�       �	���^Xc�A�*

loss�o=O!8�       �	B��^Xc�A�*

lossL��<"�N       �	�M�^Xc�A�*

loss�9o=���       �	!�^Xc�A�*

loss�4y<i�+}       �	3�^Xc�A�*

loss{&�<�Mx       �	!Β^Xc�A�*

lossn�;��{P       �	�b�^Xc�A�*

loss)��=��`       �	v��^Xc�A�*

loss�x=%W��       �	?��^Xc�A�*

lossS�<6f�       �	�,�^Xc�A�*

lossV��<��'       �	LǕ^Xc�A�*

loss�+O=�=p�       �	Ja�^Xc�A�*

loss��}=\��       �	H��^Xc�A�*

lossCp�;����       �	O��^Xc�A�*

lossS�<��;�       �	�0�^Xc�A�*

loss!��=��       �	�Ϙ^Xc�A�*

loss8�H<T�7I       �	e�^Xc�A�*

loss�|-:�!       �	~��^Xc�A�*

loss�;yzj�       �	X��^Xc�A�*

lossJ �<����       �	�.�^Xc�A�*

loss]n�;/�U�       �	'؛^Xc�A�*

loss��;���       �	y�^Xc�A�*

losss�x:b+�D       �	�^Xc�A�*

loss�b<�R�       �	4��^Xc�A�*

lossA�P:*.�       �	�k�^Xc�A�*

loss��);��rU       �	E�^Xc�A�*

loss�y�:��$       �	q��^Xc�A�*

loss���<�5~�       �	�K�^Xc�A�*

loss���<f�ɣ       �	�^Xc�A�*

loss�H#=U%��       �	ǂ�^Xc�A�*

loss'�;��u�       �	�$�^Xc�A�*

loss{6=�]�B       �	�Ǣ^Xc�A�*

loss{�>�}�?       �	�s�^Xc�A�*

loss|��<iO       �	<�^Xc�A�*

lossӁ=�.��       �	ͮ�^Xc�A�*

loss6��=z�J�       �	�E�^Xc�A�*

lossw�=��L       �	�ܥ^Xc�A�*

loss�$�<�?F�       �	�v�^Xc�A�*

loss�7�<�T�h       �	'�^Xc�A�*

loss1�:>�v�v       �	��^Xc�A�*

loss#�+=v�       �	�T�^Xc�A�*

loss�ϭ=�/�       �	�^Xc�A�*

loss���=7ږ       �	��^Xc�A�*

loss�<x=��       �	ˆ�^Xc�A�*

loss���=�z^+       �	Ϋ^Xc�A�*

loss��j=�/�       �	i�^Xc�A�*

lossXY�<�F��       �	nk�^Xc�A�*

loss�o=�~)       �	.S�^Xc�A�*

loss��<=4ʕ�       �	��^Xc�A�*

loss��=�)
�       �	ݚ�^Xc�A�*

lossrϭ<����       �	e�^Xc�A�*

loss��
>�>�       �	�<�^Xc�A�*

loss
m�<pm�M       �	%�^Xc�A�*

lossA��<�~�%       �	AE�^Xc�A�*

loss�<1^܁       �	z�^Xc�A�*

loss�W�=gG��       �	ˁ�^Xc�A�*

loss���;�C(�       �	<�^Xc�A�*

loss.]M<jʼ�       �	F��^Xc�A�*

lossݒ;ψ��       �	U�^Xc�A�*

loss��<�k�       �	I�^Xc�A�*

loss�[:=����       �	Ɗ�^Xc�A�*

lossn�a=
X�e       �	�4�^Xc�A�*

losss�=�gT�       �	�ٺ^Xc�A�*

lossY�<�Y�&       �	�|�^Xc�A�*

loss#<�<U���       �	��^Xc�A�*

loss�)<��G       �	J��^Xc�A�*

loss<�<��o       �	QK�^Xc�A�*

loss���<�<H       �	���^Xc�A�*

lossF
�;+H:5       �	���^Xc�A�*

loss��
=6'�x       �	 $�^Xc�A�*

loss@D&=�d��       �	E��^Xc�A�*

losslU`=���       �	�T�^Xc�A�*

loss�t=n�[�       �	��^Xc�A�*

loss�?`;fs�D       �	>��^Xc�A�*

loss�
:=\���       �	5'�^Xc�A�*

loss�i�='\�}       �	ؼ�^Xc�A�*

lossd|�=�p��       �	�Q�^Xc�A�*

loss=�4<
�@       �	���^Xc�A�*

loss3��<���       �	��^Xc�A�*

losswj�=PIl�       �	��^Xc�A�*

loss��<�!@       �	��^Xc�A�*

loss���<�	 �       �		N�^Xc�A�*

loss��J<��4T       �	�)�^Xc�A�*

loss�3�<$�2'       �	]��^Xc�A�*

loss���<j�o       �	�.�^Xc�A�*

loss�=�XB       �	� �^Xc�A�*

loss��8=����       �	���^Xc�A�*

loss�9�=4���       �	�O�^Xc�A�*

loss���<*�G�       �	�^Xc�A�*

loss��r<�Yr       �	��^Xc�A�*

loss��=z�Y       �	� �^Xc�A�*

lossL��<��.       �	���^Xc�A�*

lossе=&Q�       �	)��^Xc�A�*

loss��9=��       �	�+�^Xc�A�*

lossd�z<!�l       �	Y��^Xc�A�*

loss^�<���_       �	�Y�^Xc�A�*

lossVA�=�)��       �	 �^Xc�A�*

loss�=�<�n       �	���^Xc�A�*

loss!�U=7`�	       �	�b�^Xc�A�*

loss#=9���       �	���^Xc�A�*

lossM�K<�/��       �	��^Xc�A�*

loss�gC==�       �	?6�^Xc�A�*

loss��<��       �	���^Xc�A�*

lossșE=�H҇       �	bi�^Xc�A�*

loss��<ˇ�       �	��^Xc�A�*

loss���=�Kmm       �	��^Xc�A�*

loss�"=I�c       �	a5�^Xc�A�*

loss�=�Vg�       �	���^Xc�A�*

lossF��<%�       �	�c _Xc�A�*

lossc�0=�fr�       �	D� _Xc�A�*

loss��<:a��       �	1�_Xc�A�*

lossF��<�e��       �	s,_Xc�A�*

loss(�H==�0�       �	��_Xc�A�*

loss�="؏       �	CY_Xc�A�*

lossl=���       �	�_Xc�A�*

lossa�<.2x       �	��_Xc�A�*

loss�!<�5�N       �	X_Xc�A�*

loss�#�=2Wp%       �	O�_Xc�A�*

loss���<Y�o       �	�C_Xc�A�*

loss��{=7>�7       �	B�_Xc�A�*

loss�׵;@���       �	V_Xc�A�*

loss���<[�|       �	p"_Xc�A�*

loss��=1�)�       �	o�_Xc�A�*

lossò�=�^�       �	�R	_Xc�A�*

loss�ao;?<F-       �	t�	_Xc�A�*

loss�T�=c�E       �	�
_Xc�A�*

loss��<���s       �	�_Xc�A�*

loss;�.<TD08       �	i�_Xc�A�*

lossd��= zqy       �	�D_Xc�A�*

loss}Ͷ=�q�l       �	��_Xc�A�*

loss���<>�b       �	��_Xc�A�*

loss�)=<��       �	?�_Xc�A�*

loss4�:=�1Q�       �	b0_Xc�A�*

loss�و;}\8       �	G�_Xc�A�*

loss�<H���       �	�e_Xc�A�*

lossR#�=��       �	z�_Xc�A�*

loss��<���       �	�_Xc�A�*

loss���=#�1       �	��_Xc�A�*

loss2�<~Éw       �	�\_Xc�A�*

loss3A�;��$G       �	��_Xc�A�*

lossM�V;�m�        �	�_Xc�A�*

lossl��;�,�[       �	�-_Xc�A�*

lossĂ�<?5��       �	��_Xc�A�*

loss�q�=�&�&       �	=d_Xc�A�*

loss�ш<}g�       �	�_Xc�A�*

loss�2= G��       �	u�_Xc�A�*

loss	<���       �	)B_Xc�A�*

loss ��<|���       �	��_Xc�A�*

loss/?<���       �	Ɔ_Xc�A�*

lossx�*=4y6:       �	�%_Xc�A�*

lossD�=�F�       �	�_Xc�A�*

loss��< Qtp       �	�`_Xc�A�*

loss�>Y���       �	H�_Xc�A�*

loss�G<�0y6       �	B�_Xc�A�*

loss��<t�D       �	]6_Xc�A�*

lossv��<���       �	�_Xc�A�*

loss�?=�*�       �	<j_Xc�A�*

lossߞ<7f       �	�	_Xc�A�*

loss��C=�N��       �	n�_Xc�A�*

loss��3<��       �	; _Xc�A�*

loss᱾<$��       �	B� _Xc�A�*

loss
u}<*3�       �	h!_Xc�A�*

loss�˺;?�	       �	H�!_Xc�A�*

losscN�=Y�I       �	�"_Xc�A�*

lossD�<�e^�       �	�@#_Xc�A�*

loss��<Rjy       �	��#_Xc�A�*

loss&jo=R�ͼ       �	(|$_Xc�A�*

loss׈J;S�"�       �	]%_Xc�A�*

loss��<�t�       �	��%_Xc�A�*

loss)��;��       �	JB&_Xc�A�*

loss?��=��n       �	f�&_Xc�A�*

loss@�<z���       �	t~'_Xc�A�*

loss`1�<�m�'       �	<(_Xc�A�*

loss��e<|�,       �	L�(_Xc�A�*

loss�7m=�_!�       �	�C)_Xc�A�*

loss*W�=ɤҴ       �	��)_Xc�A�*

loss�y�<ܥ       �	�u*_Xc�A�*

loss!��<���       �	�+_Xc�A�*

loss.��<!xDA       �	a�+_Xc�A�*

loss���;�o�       �	lB,_Xc�A�*

losswD:<��       �	�,_Xc�A�*

lossA5�<9_;s       �	1x-_Xc�A�*

loss!/�=��       �	��._Xc�A�*

loss�,3<���V       �	�S/_Xc�A�*

loss�x�<�F       �	8�/_Xc�A�*

loss�ӗ<���~       �	�0_Xc�A�*

loss��==)5�       �	�.1_Xc�A�*

loss?��<�b�N       �	��1_Xc�A�*

loss	E�=�Ea�       �	=~2_Xc�A�*

loss3 �=�t]�       �	m3_Xc�A�*

loss$�l<I;A       �	>�3_Xc�A�*

loss;B1<���*       �	&S4_Xc�A�*

losss.�=��o       �	t�4_Xc�A�*

loss`l�<��*       �	�5_Xc�A�*

loss_�=�z}       �	T76_Xc�A�*

loss6� =?J       �	+�6_Xc�A�*

loss̀�=[Μ5       �	�}7_Xc�A�*

loss�h<\@8�       �	�8_Xc�A�*

loss�<6i�       �	��8_Xc�A�*

loss�{2<vj�       �	-Z9_Xc�A�*

loss\8=-�       �	P�9_Xc�A�*

loss[=6>�       �	��:_Xc�A�*

loss,�=ؽ�3       �	�/;_Xc�A�*

loss6G.=����       �	�;_Xc�A�*

loss�!==g+�,       �	wd<_Xc�A�*

lossF�G=��Ӎ       �	��<_Xc�A�*

loss=��<G�i�       �	 �=_Xc�A�*

loss('o<�H�       �	U0>_Xc�A�*

lossX��;X��       �	d�>_Xc�A�*

loss�p8=�]�       �	Eg?_Xc�A�*

lossMX=u�x       �	 @_Xc�A�*

lossy��<�Lx|       �	��@_Xc�A�*

loss�e�=;�p       �	�.A_Xc�A�*

loss��<��T       �	J�A_Xc�A�*

loss��l;%��m       �	`rB_Xc�A�*

lossT5w<"�0B       �	kC_Xc�A�*

loss抎=�ɟ�       �	-�C_Xc�A�*

loss�6=�z��       �	�KD_Xc�A�*

loss��>P�>`       �	�D_Xc�A�*

loss T�=ut�}       �	�xE_Xc�A�*

loss2݋=��       �	sF_Xc�A�*

loss �$<�Ժ�       �	�F_Xc�A�*

loss�r�<@�C	       �	IG_Xc�A�*

loss�3=եٴ       �	��G_Xc�A�*

lossCPG;��       �	:wH_Xc�A�*

loss�K=!�c�       �	EI_Xc�A�*

loss�=�
�       �	��I_Xc�A�*

loss1�<�HAN       �	(CJ_Xc�A�*

loss�<���       �	�J_Xc�A�*

lossӠw<�
�       �	�pK_Xc�A�*

loss͇�=+�K�       �	�L_Xc�A�*

loss���<N��       �	��L_Xc�A�*

loss-��=^z�A       �	�TN_Xc�A�*

loss��h<0�l4       �	.O_Xc�A�*

loss&�l=�d59       �	�O_Xc�A�*

loss�~J=�8��       �	�GP_Xc�A�*

lossz�=�z�       �	�P_Xc�A�*

loss���=�Izi       �	~Q_Xc�A�*

loss��=�K�q       �	 R_Xc�A�*

loss�rG<��̙       �	4�R_Xc�A�*

lossfI�;3��       �	6xS_Xc�A�*

loss��w=*�p^       �	 )T_Xc�A�*

lossҰ�=0�k�       �	�T_Xc�A�*

loss�v=W�       �	9_U_Xc�A�*

loss���<��Od       �	��U_Xc�A�*

loss��q=��	G       �	%�V_Xc�A�*

loss�z=��F�       �	�0W_Xc�A�*

loss�e5;��4       �	��W_Xc�A�*

loss�| =K��       �	�YX_Xc�A�*

loss�<l'j       �	R�X_Xc�A�*

loss)��<�7k�       �	�Y_Xc�A�*

loss���<>";�       �	�Z_Xc�A�*

loss���<�ջ7       �	R�Z_Xc�A�*

loss��=a��       �	�K[_Xc�A�*

loss�,=+�       �	�[_Xc�A�*

loss�d<�Tu}       �	ѓ\_Xc�A�*

lossVF�<
�P       �	B']_Xc�A�*

loss�27=r��       �	�]_Xc�A�*

loss�)�<u�/�       �	�U^_Xc�A�*

loss_�<X�p       �	��^_Xc�A�*

loss��=�*�>       �	��__Xc�A�*

loss/�Y=nޫ       �	U.`_Xc�A�*

lossX��<��k       �	L�`_Xc�A�*

lossd8�;�/��       �	�ia_Xc�A�*

loss�Δ;�R��       �	Cb_Xc�A�*

loss��<��T
       �	��b_Xc�A�*

loss(^=q�%�       �	�Cc_Xc�A�*

loss?y�;.8ca       �	��c_Xc�A�*

loss�7<aB       �	�d_Xc�A�*

loss7��<�-�'       �	e_Xc�A�*

lossOV=�J�       �	;�e_Xc�A�*

lossW=0�u�       �	�Ef_Xc�A�*

loss$�q=X�y1       �	;�f_Xc�A�*

loss��<TP<       �	�yg_Xc�A�*

loss���=uEI       �	�h_Xc�A�*

loss 	<Sz�        �	c�h_Xc�A�*

loss���<>=       �	�[i_Xc�A�*

loss�~�<=��       �	��i_Xc�A�*

loss?��<�:�[       �	��j_Xc�A�*

losss�`=�Ơ�       �	cCk_Xc�A�*

lossٔ=��<�       �	Vl_Xc�A�*

loss,6Q=�r�j       �	^�l_Xc�A�*

loss�0�=�D��       �	�n_Xc�A�*

loss*s�;�jآ       �	��n_Xc�A�*

lossmN�<�d��       �	B�o_Xc�A�*

lossf�8<�]e�       �	Hp_Xc�A�*

lossH H=dI�       �	D�p_Xc�A�*

loss�;�=U6KU       �	�;r_Xc�A�*

lossc��<���       �	 �r_Xc�A�*

loss�3+<�\�       �	�es_Xc�A�*

loss���=?�Jh       �	Mt_Xc�A�*

lossج�=��^       �	Ψt_Xc�A�*

loss}�<�n�       �	ePu_Xc�A�*

loss L=���       �	'�u_Xc�A�*

loss�SP<�H��       �	6�v_Xc�A�*

lossD[=�USR       �	�)w_Xc�A�*

loss8��;uӀ�       �	]�w_Xc�A�*

loss�+�<���O       �	�Yx_Xc�A�*

loss�ix=n�       �	|�x_Xc�A�*

loss�ԅ=%r5�       �	`�y_Xc�A�*

loss�0�=-$6       �	U.z_Xc�A�*

loss永=Z�       �	;�z_Xc�A�*

loss�=�ٽg       �	�b{_Xc�A�*

loss���=X	y       �	��{_Xc�A�*

lossDǢ<f��       �	��|_Xc�A�*

loss�7=щz�       �	c'}_Xc�A�*

loss��$<�ڗ�       �	b�}_Xc�A�*

loss�o�=hW�9       �	uW~_Xc�A�*

lossJP4<h~T�       �	��~_Xc�A�*

lossf=�|�       �	6�_Xc�A�*

lossqw�=�ӵ�       �	e4�_Xc�A�*

loss�F<����       �	^؀_Xc�A�*

loss%�<���/       �	<2�_Xc�A�*

loss�~0=���       �	�҂_Xc�A�*

loss@u�;ZN        �	�t�_Xc�A�*

loss��z<�BJ@       �	��_Xc�A�*

loss�A=�       �	���_Xc�A�*

lossR4�<��:j       �	�V�_Xc�A�*

lossջ<� (W       �	��_Xc�A�*

loss�e�<����       �	���_Xc�A�*

lossn�=��x�       �	+�_Xc�A�*

lossN+=	��       �	�ȇ_Xc�A�*

loss�x�<tR6v       �	�b�_Xc�A�*

loss/��<R:V       �	� �_Xc�A�*

loss���<1�ʦ       �	]��_Xc�A�*

loss
I_=5wO       �	���_Xc�A�*

loss�>�<�O�       �	~9�_Xc�A�*

loss�=Hk�       �	�ԋ_Xc�A�*

lossR�<P�;       �	:x�_Xc�A�*

lossT�<�$��       �	*�_Xc�A�*

loss���<�sY       �	��_Xc�A�*

loss��=���B       �	���_Xc�A�*

loss�P=���       �	?W�_Xc�A�*

loss�DY<�k��       �	��_Xc�A�*

loss�@o=5��       �	!��_Xc�A�*

loss
g�=��%       �	'0�_Xc�A�*

loss�@=��@8       �	��_Xc�A�*

loss�^<A׮�       �	�%�_Xc�A�*

loss4��:��       �	xϓ_Xc�A�*

loss<��<X
c       �	�i�_Xc�A�*

loss���<�a       �	� �_Xc�A�*

loss�ܩ<�):�       �	<��_Xc�A�*

loss��=��       �	
i�_Xc�A�*

loss�q=iI��       �	�_Xc�A�*

loss7�<X�i0       �	���_Xc�A�*

lossW-�=�Y�7       �	�@�_Xc�A�*

loss4��=�R�x       �	�ߘ_Xc�A�*

loss]k,=!YA�       �	[}�_Xc�A�*

loss��L=)Mn       �	0�_Xc�A�*

loss��;s$b{       �	կ�_Xc�A�*

loss8�=�}
�       �	�E�_Xc�A�*

lossLş<���       �	ڜ_Xc�A�*

lossVؕ=���       �	�o�_Xc�A�*

loss��U=�G��       �	�&�_Xc�A�*

loss8b�<Zǚ*       �	<��_Xc�A�*

loss��=�M�x       �	�\�_Xc�A�*

loss��<���       �	���_Xc�A�*

lossd�+=�E-{       �	Н�_Xc�A�*

loss��=�Ծ&       �	�4�_Xc�A�*

lossӳ=��!       �	9С_Xc�A�*

lossxT)=�D�       �	�j�_Xc�A�*

loss�F�;�A~       �	5�_Xc�A�*

lossz�=>d-�       �	���_Xc�A�*

loss�/r=Y��N       �	�;�_Xc�A�*

loss���;J�Q@       �	��_Xc�A�*

loss�\<�R<Q       �	�z�_Xc�A�*

lossĿ=�5��       �	��_Xc�A�*

lossI~�<�ȋ       �	ͫ�_Xc�A�*

loss�a�=T֤       �	AI�_Xc�A�*

lossVB=�k�M       �	�_Xc�A�*

loss*�"=[���       �	j��_Xc�A�*

loss�2>=���       �	)#�_Xc�A�*

lossA:�<�[)�       �	�Ʃ_Xc�A�*

loss���;�<�       �	�c�_Xc�A�*

loss��4;@X�       �	�	�_Xc�A�*

lossH��<�9�       �	(��_Xc�A�*

lossM�y;N�rJ       �	gb�_Xc�A�*

loss�8�;ZV�       �	p�_Xc�A�*

loss�q�<��Dz       �	+�_Xc�A�*

loss`v�<T��       �	W��_Xc�A�*

loss-��=�&c       �	�ѯ_Xc�A�*

loss䨪=h�	�       �	�p�_Xc�A�*

lossM�=��        �	��_Xc�A�*

loss��`=5f:k       �	J��_Xc�A�*

loss�D=���+       �	�M�_Xc�A�*

lossX�S;��L       �	��_Xc�A�*

loss�K�<bU�"       �	f��_Xc�A�*

loss�π=�/�:       �	��_Xc�A�*

loss԰#=�T��       �	���_Xc�A�*

lossx�+=@�Fq       �	�G�_Xc�A�*

lossFs�<���@       �	��_Xc�A�*

lossh �<(��U       �	��_Xc�A�*

lossÂ%=�d$h       �	��_Xc�A�*

loss���<�oEo       �	}��_Xc�A�*

loss��<4�9D       �	�J�_Xc�A�*

loss(W=�j�       �	��_Xc�A�*

loss*# =���       �	���_Xc�A�*

loss�p3<Ć�       �	82�_Xc�A�*

loss�_�=V��r       �	^պ_Xc�A�*

loss���=���       �	�|�_Xc�A�*

loss�v�<O�m�       �	�_Xc�A�*

loss�r6<�ބl       �	�ü_Xc�A�*

loss��T=���       �	bj�_Xc�A�*

loss)�=ʄ�       �	�_Xc�A�*

lossڄ�= ��       �	���_Xc�A�*

loss�w�=�$�       �	VJ�_Xc�A�*

lossMN�=�=B       �	'��_Xc�A�*

loss�ğ=�л�       �	A��_Xc�A�*

loss�=��6       �	�A�_Xc�A�*

lossS=	��"       �	���_Xc�A�*

loss5�<��]�       �	l��_Xc�A�*

loss�?=���       �	�?�_Xc�A�*

loss��	=�.��       �	]��_Xc�A�*

loss[x=���       �	}��_Xc�A�*

lossź�<]Dh       �	 8�_Xc�A�*

loss�V=%+��       �	I��_Xc�A�*

loss�# <�,�       �	qu�_Xc�A�*

losshڣ<<���       �	��_Xc�A�*

lossp�=�"       �	���_Xc�A�*

loss	�p;)A8       �	6w�_Xc�A�*

loss�l<��h       �	� �_Xc�A�*

lossv^;=��N       �	���_Xc�A�*

loss��=�O�       �	�|�_Xc�A�*

lossb��;w���       �	\�_Xc�A�*

loss�N=X3��       �	R��_Xc�A�*

lossm�)<����       �	HR�_Xc�A�*

lossl˙=b;�@       �	���_Xc�A�*

loss��=6�?}       �	r�_Xc�A�*

loss��->��!�       �	5��_Xc�A�*

loss�)=��k�       �	�`�_Xc�A�*

lossD��<ڟb	       �	���_Xc�A�*

loss|�|=w��Q       �	��_Xc�A�*

loss��==_R�       �	B�_Xc�A�*

loss��=�S�\       �	���_Xc�A�*

loss��;� ��       �	���_Xc�A�*

loss餣<�L8�       �	 ��_Xc�A�*

lossW��<>�d�       �	���_Xc�A�*

loss�¤<���       �	�k�_Xc�A�*

loss��=�1�       �	O�_Xc�A�*

lossfߏ<�v`       �	��_Xc�A�*

lossaڕ=�^�\       �	P9�_Xc�A�*

lossC?�<�O�       �	0��_Xc�A�*

losss��<�I�       �	~r�_Xc�A�*

loss���;j��       �	S�_Xc�A�*

loss�7&<l�"       �	��_Xc�A�*

loss(��<�t�       �	���_Xc�A�*

lossf�=H��E       �	ܸ�_Xc�A�*

loss蹡=9R0�       �	΋�_Xc�A�*

loss}�5=��MC       �	@��_Xc�A�*

loss$m={�       �	�U�_Xc�A�*

loss��>H�L       �	|��_Xc�A�*

loss�7<@���       �	���_Xc�A�*

loss8<�<��`0       �	�R�_Xc�A�*

loss���<2R�       �	���_Xc�A�*

loss��Z<�mzh       �	\��_Xc�A�*

loss�=!���       �	�0�_Xc�A�*

loss��-<��:       �	���_Xc�A�*

loss)3<PBV2       �	l�_Xc�A�*

loss�w?<ĥ��       �	�_Xc�A�*

loss�5�=��U       �	d��_Xc�A�*

loss o'=��ڜ       �	K�_Xc�A�*

lossr1?=n��       �	���_Xc�A�*

loss��<.8{       �	���_Xc�A�*

loss�	=6a�*       �	40�_Xc�A�*

lossE5%<�b��       �	���_Xc�A�*

loss�a�<�       �	Ŏ�_Xc�A�*

lossG��=��,       �	r6�_Xc�A�*

lossq�8<����       �	�v�_Xc�A�*

loss�2�;�>�q       �	!�_Xc�A�*

loss�1j=g���       �	���_Xc�A�*

loss*_=����       �	��_Xc�A�*

loss%��=f�O       �	8�_Xc�A�*

lossk<t[��       �	���_Xc�A�*

loss�g<(`_       �	�m�_Xc�A�*

loss��<x!��       �	1#�_Xc�A�*

loss\�T<��        �	���_Xc�A�*

loss�,V<Mx�       �	&T�_Xc�A�*

lossc=d�ʃ       �	G�_Xc�A�*

lossz #=ө�       �	r��_Xc�A�*

loss]�=��PC       �	�r�_Xc�A�*

lossmc�:�%�0       �	��_Xc�A�*

lossJ�;`;       �	���_Xc�A�*

loss?�(<o�       �	tD�_Xc�A�*

lossT�;Po��       �	G��_Xc�A�*

loss�ע;a*       �	~�_Xc�A�*

lossȇ-=R��        �	��_Xc�A�*

loss��>�	��       �	���_Xc�A�*

loss�F=��^�       �	iV�_Xc�A�*

loss���=���       �	���_Xc�A�*

lossD�<�@�m       �	���_Xc�A�*

loss��^<5S�m       �	o.�_Xc�A�*

loss�Ѣ<�S�       �	��_Xc�A�*

lossŀ=p*       �	�s�_Xc�A�*

lossf_�=)�       �	o�_Xc�A�*

loss��D=��%�       �	���_Xc�A�*

loss�}�=�6�T       �	5E�_Xc�A�*

lossn�><ϲ/�       �	���_Xc�A�*

loss,�=�̷       �	{�_Xc�A�*

loss1@�<�jx�       �	��_Xc�A�*

lossJ;
=��-       �	���_Xc�A�*

lossFD~<\��       �	�M `Xc�A�*

loss� =����       �	�� `Xc�A�*

loss$zZ=~�j       �	\�`Xc�A�*

loss��=[��       �	�,`Xc�A�*

loss�2w<G(ؽ       �	�`Xc�A�*

loss�=��g�       �	�^`Xc�A�*

loss� i;��u�       �	��`Xc�A�*

loss�V=����       �	��`Xc�A�*

loss�Vd=�       �	�0`Xc�A�*

lossN�=�U       �	��`Xc�A�*

loss�E=�I3_       �	vn`Xc�A�*

loss��7=����       �	�`Xc�A�*

losse�
=i�)       �	n�`Xc�A�*

loss@�\;Nb�-       �	�6`Xc�A�*

loss���=���       �	��`Xc�A�*

loss� *<S�!l       �	�g	`Xc�A�*

lossT;]��       �	�
`Xc�A�*

loss��=ǰG       �	ٗ
`Xc�A�*

loss��=I�p       �	�C`Xc�A�*

loss��5<4;Q       �	�`Xc�A�*

loss6m<�Lɟ       �	J{`Xc�A�*

loss�<���       �	`Xc�A�*

loss��?=�}��       �	ծ`Xc�A�*

lossjp=B�zL       �	�E`Xc�A�*

loss��~=�iSi       �	�`Xc�A�*

loss���=2��       �	��`Xc�A�*

loss�?�=f��2       �	&`Xc�A�*

loss�"�=jc       �	��`Xc�A�*

loss��u<oy�B       �	�a`Xc�A�*

loss���< ���       �	��`Xc�A�*

loss��W=|�a�       �	��`Xc�A�*

loss��=��=7       �	3`Xc�A�*

lossW��<B��}       �	��`Xc�A�*

loss��<�4�       �	�i`Xc�A�*

loss�&=zx�       �	`Xc�A�*

lossh�g=:��~       �	��`Xc�A�*

loss�.v<�_o       �	�M`Xc�A�*

loss�c�<iA��       �	��`Xc�A�*

lossm��<,       �	��`Xc�A�*

lossI�<���8       �	 `Xc�A�*

loss
�<�p_�       �	ȵ`Xc�A�*

loss�~<���       �	�_`Xc�A�*

losss_<�R1       �	;�`Xc�A�*

loss8'�=��'�       �	��`Xc�A�*

loss4:<l�0�       �	�3`Xc�A�*

lossa��;�       �	`�`Xc�A�*

loss�W�;Elp�       �	h`Xc�A�*

lossuC=zw7       �	�
`Xc�A�*

loss==�       �	P�`Xc�A�*

loss؏�<�A�       �	�>`Xc�A�*

lossד�=��       �	b�`Xc�A�*

lossqH<#k"I       �	}u`Xc�A�*

lossk��=Q��       �	Z `Xc�A�*

lossD\�:���}       �	4� `Xc�A�*

loss�8=�F��       �	.U!`Xc�A�*

loss/�x<���       �	x�!`Xc�A�*

lossf��<�8wW       �	?�"`Xc�A�*

loss���<eN�       �	GZ#`Xc�A�*

loss4q<K��#       �	��#`Xc�A�*

lossN�;l��y       �	��$`Xc�A�*

loss��<ۄ�$       �	�:%`Xc�A�*

loss|�;]��       �	��%`Xc�A�*

loss��5=�ai�       �	�&`Xc�A�*

lossȪ�=="<       �	H'`Xc�A�*

loss�>�I��       �	t�'`Xc�A�*

loss�m=Ղ�       �	�M(`Xc�A�*

lossaON<���       �	��(`Xc�A�*

lossc�=F�{`       �	4�)`Xc�A�*

losswo=@YF�       �	�*`Xc�A�*

loss`-$<�s\V       �	B�*`Xc�A�*

loss
,�<{�E       �	�`+`Xc�A�*

lossfo�=Th�       �	�,`Xc�A�*

loss#<�K0�       �	�,`Xc�A�*

loss{OQ=L�3�       �	I-`Xc�A�*

loss$O=�Dd�       �	��-`Xc�A�*

loss.~�<���       �	��.`Xc�A�*

loss��;�;��       �	h$/`Xc�A�*

loss��U=�(A�       �	��/`Xc�A�*

loss=�;V ��       �	XY0`Xc�A�*

loss��;���       �	A�0`Xc�A�*

loss-��<�>�m       �	��1`Xc�A�*

loss�k4<��ݓ       �	�2`Xc�A�*

loss�t<F�       �	iR3`Xc�A�*

loss���<�.�       �	��3`Xc�A�*

loss��:=R�21       �	͏4`Xc�A�*

loss�wk<.       �	�=5`Xc�A�*

loss��=��       �	�5`Xc�A�*

loss��	<s�4~       �	`v6`Xc�A�*

loss�s=���       �	j7`Xc�A�*

loss��;|v�       �	ƾ7`Xc�A�*

loss7�d<���       �	0d8`Xc�A�*

loss���<��       �	�9`Xc�A�*

lossQ�Z=?$�       �	�9`Xc�A�*

lossF�7=�+�       �	�Y:`Xc�A�*

loss��<?�       �	�1<`Xc�A�*

loss�t=LD��       �	|�<`Xc�A�*

loss�C0<I|�!       �	 t=`Xc�A�*

loss��6=��UI       �	� >`Xc�A�*

loss�X�;M�0       �	�>`Xc�A�*

losst�<p�:#       �	�[?`Xc�A�*

lossR�@=j�7�       �	��?`Xc�A�*

loss��;b���       �	\�@`Xc�A�*

loss�	�<㼃E       �	,A`Xc�A�*

loss=ë<ݏL�       �	P�A`Xc�A�*

loss^t=j�w       �	�kB`Xc�A�*

loss��*=�k�       �	�C`Xc�A�*

loss�կ:0�       �	��C`Xc�A�*

loss1m�;�S!�       �	�LD`Xc�A�*

losss��<\�B�       �	��D`Xc�A�*

lossx�<Œ7       �	�}E`Xc�A�*

loss���<Pֵ       �	�F`Xc�A�*

lossҕ
<�&��       �	c�F`Xc�A�*

loss�]&=
=�       �	<MG`Xc�A�*

loss��I;Q��L       �	�G`Xc�A�*

loss��<a�C       �	��H`Xc�A�*

loss��=Fa�b       �	0*I`Xc�A�*

loss� �:p���       �	4�I`Xc�A�*

lossU�<^�ۥ       �	�kJ`Xc�A�*

loss�l<��Q       �	K`Xc�A�*

loss�y<~�SO       �	h�K`Xc�A�*

lossZ��<?Y�t       �	�XL`Xc�A�*

loss�:Q���       �	�L`Xc�A�*

loss��:�?ϻ       �	؝M`Xc�A�*

loss���<X��       �	X8N`Xc�A�*

loss��;OX�       �	��N`Xc�A�*

loss=�t8Į       �	�O`Xc�A�*

loss�\x8���m       �	$P`Xc�A�*

lossb|:F��S       �	��P`Xc�A�*

loss��<���       �	�`Q`Xc�A�*

loss{5;��       �	3�Q`Xc�A�*

loss@�<_f�\       �	`�R`Xc�A�*

loss�>�<^�o#       �	�YS`Xc�A�*

loss,�)>D���       �	77T`Xc�A�*

loss1�";���;       �	l
U`Xc�A�*

loss��3=���       �	��U`Xc�A�*

loss�48<�X�y       �	bfV`Xc�A�*

loss�&�=�w��       �	W`Xc�A�*

lossZ[�<�@�       �	��W`Xc�A�*

loss��<=�cG       �	$�X`Xc�A�*

loss3�<���       �	$cY`Xc�A�*

lossۡ�<�^L       �	}Z`Xc�A�*

loss�~)=�7��       �	z�Z`Xc�A�*

loss�ԑ<��#       �	@P[`Xc�A�*

lossT�G<�p�       �	��[`Xc�A�*

loss�}�=���       �	\�\`Xc�A�*

lossA�H=[ݡ       �	]1]`Xc�A�*

losss��<���K       �	w�]`Xc�A�*

loss�O=�ߎ       �	��^`Xc�A�*

loss�.�<���       �	C_`Xc�A�*

loss�N=��Qt       �	��_`Xc�A�*

losszK�=R*��       �		�``Xc�A�*

loss�/{=����       �	9}a`Xc�A�*

loss%]�<ATT�       �	*b`Xc�A�*

loss��==oV�       �	+�b`Xc�A�*

lossO�;"�j�       �	�c`Xc�A�*

loss�< �+       �	(d`Xc�A�*

lossV�B:��ӎ       �	��d`Xc�A�*

loss�N[<t       �	>xe`Xc�A�*

loss�J�;Ea�A       �	�!f`Xc�A�*

lossAL <�;|�       �	y�f`Xc�A�*

lossJ#=���       �	rg`Xc�A�*

loss�%�<�?��       �	�%h`Xc�A�*

lossL��=+�       �	��h`Xc�A�*

loss?��; ���       �	Έi`Xc�A�*

loss�`<&��       �	�4j`Xc�A�*

lossNR<�.r�       �	��j`Xc�A�*

loss��=��{�       �	ߋk`Xc�A�*

loss�7<1pH3       �	�7l`Xc�A�*

loss�Ξ<s,�       �	;�l`Xc�A�*

loss� �;v/�       �	�m`Xc�A�*

lossI��<.�C       �	Nn`Xc�A�*

loss�$�=>J��       �	��n`Xc�A�*

loss�u�;���       �	��o`Xc�A�*

lossE��;n1�       �	B]p`Xc�A�*

lossc>=���w       �	=q`Xc�A�*

lossa�=����       �	�q`Xc�A�*

loss��<�2�       �	�Zr`Xc�A�*

loss�Z�<�Z�       �	#s`Xc�A�*

loss$�W=�H       �	��s`Xc�A�*

loss��u=�̏�       �	zt`Xc�A�*

lossd�,<�.�1       �	@0u`Xc�A�*

loss�=�<���e       �	��u`Xc�A�*

loss���;��)�       �	Trv`Xc�A�*

loss��2;J=��       �	�w`Xc�A�*

loss�-�<+ӟ,       �	��`Xc�A�*

loss�_=�ޢ       �	y��`Xc�A�*

loss�>��e       �	�-�`Xc�A�*

lossa^�=�|w�       �	Kɏ`Xc�A�*

loss]��<V�=       �	�g�`Xc�A�*

loss{� =�n�       �	p	�`Xc�A�*

loss�<��ݪ       �	���`Xc�A�*

loss�?=�\j       �	(b�`Xc�A�*

lossg�=�/�       �	�`Xc�A�*

loss�N�<��B       �	���`Xc�A�*

loss�=4m       �	�Y�`Xc�A�*

loss�g=3���       �	'/�`Xc�A�*

loss�4=+�       �	͕`Xc�A�*

loss1�=3ݶ       �	���`Xc�A�*

lossqYT<�_�5       �	DP�`Xc�A�*

loss��(=9�/       �	w�`Xc�A�*

loss�h;m�N"       �	���`Xc�A�*

loss�R�<�p1�       �	�@�`Xc�A�*

loss��
=sG	�       �	sٙ`Xc�A�*

loss�4=�O0�       �	�n�`Xc�A�*

loss6�w<�L�       �		�`Xc�A�*

loss=X˓       �	���`Xc�A�*

loss��K;���V       �	�8�`Xc�A�*

loss^�=�d�       �	�ܜ`Xc�A�*

loss�*�<ۊ`       �	7q�`Xc�A�*

lossv��<i��>       �	J�`Xc�A�*

loss:"[=u�{P       �	�d�`Xc�A�*

loss(��<��       �	���`Xc�A�*

loss�>=>e*�       �	��`Xc�A�*

loss�M=�mI]       �	R�`Xc�A�*

losszQ�;U��       �	1�`Xc�A�*

loss_]�<=�2L       �	��`Xc�A�*

loss|=A��       �	�8�`Xc�A�*

loss6J�=Gt��       �	<ڣ`Xc�A�*

loss�;�RB       �	���`Xc�A�*

loss;Qv=��       �	W!�`Xc�A�*

lossA�!<p-��       �	ƥ`Xc�A�*

losse�"=ķ��       �	���`Xc�A�*

loss��=��k       �	�J�`Xc�A�*

loss�+�<�\�       �	O�`Xc�A�*

loss��P<'���       �	I��`Xc�A�*

loss�H=q?-�       �	�*�`Xc�A�*

loss:�;���.       �	2̩`Xc�A�*

lossOg�<CHZx       �	Ll�`Xc�A�*

lossX"r=M��       �	�	�`Xc�A�*

loss�=�E�       �	���`Xc�A�*

loss�k<%Z:�       �	2=�`Xc�A�*

loss6�~<5�C�       �	�խ`Xc�A�*

loss:�<���       �	��`Xc�A�*

loss�<�?��       �	���`Xc�A�*

loss=0<c 5�       �	ܺ�`Xc�A�*

lossOi=gu��       �	��`Xc�A�*

loss���<��>       �	�\�`Xc�A�*

loss��=�Yv       �	M�`Xc�A�*

loss�=����       �	��`Xc�A�*

loss1`�;�$��       �	M��`Xc�A�*

lossn�;���       �	�'�`Xc�A�*

loss��M<���+       �	�ص`Xc�A�*

loss���<�X%       �	�y�`Xc�A�*

loss~~�<*��       �	�'�`Xc�A�*

loss?k=����       �	�˷`Xc�A�*

loss�i;�d�       �	�b�`Xc�A�*

loss�J;�L]       �	<��`Xc�A�*

loss-0�<R�       �	C��`Xc�A�*

lossA�<�O�       �	�3�`Xc�A�*

lossT�:�K�y       �	�Ӻ`Xc�A�*

loss{W=����       �	Ps�`Xc�A�*

lossl};���       �	��`Xc�A�*

lossl��<%�Ut       �	ߨ�`Xc�A�*

loss$;�T��       �	�A�`Xc�A�*

loss��=ނ�       �	��`Xc�A�*

lossv��<��XV       �	9|�`Xc�A�*

lossOY�=�	�       �	E�`Xc�A�*

loss�pE;�c�       �	\��`Xc�A�*

loss��<�zP       �	lC�`Xc�A�*

loss�=�;�l       �	���`Xc�A�*

loss���<�O:       �	Gr�`Xc�A�*

loss��.;}U�!       �	��`Xc�A�*

loss�y7<�\��       �	��`Xc�A�*

loss@W=�F�       �	m<�`Xc�A�*

lossad�<A`�x       �	'��`Xc�A�*

lossnG={���       �	�t�`Xc�A�*

loss'�<�Oղ       �	_�`Xc�A�*

loss���:�}[�       �	U��`Xc�A�*

loss\��;���F       �	1?�`Xc�A�*

loss�E;O-�=       �	���`Xc�A�*

loss�>g<��       �	k��`Xc�A�*

loss�>ԶO4       �	)$�`Xc�A�*

lossdP�<�[       �	��`Xc�A�*

loss��E;^�k�       �	�j�`Xc�A�*

loss���<�\�~       �	0�`Xc�A�*

lossn�P=.*4_       �	��`Xc�A�*

lossS�<���       �	zR�`Xc�A�*

loss]v�<�(�^       �	��`Xc�A�*

loss�D=�s       �	ׅ�`Xc�A�*

loss�{�<��>       �	B%�`Xc�A�*

loss�F�=�@�}       �	#��`Xc�A�*

lossC]<�7��       �	�\�`Xc�A�*

loss)�">KT��       �	/��`Xc�A�*

lossb�=��       �	��`Xc�A�*

loss�;f<L       �	<�`Xc�A�*

lossrӫ;��/�       �	���`Xc�A�*

loss��;��R       �	�o�`Xc�A�*

loss
��<��        �	t	�`Xc�A�*

loss��7<~���       �	��`Xc�A�*

lossܙ=����       �	n��`Xc�A�*

loss&(;HO�       �	�/�`Xc�A�*

loss�X<Di[       �	\��`Xc�A�*

loss&)1=Ynf_       �	fi�`Xc�A�*

loss
�;��+>       �	�
�`Xc�A�*

loss�߁=� ��       �	���`Xc�A�*

loss��=���       �	eS�`Xc�A�*

loss���<�w I       �	���`Xc�A�*

lossol�;;�W       �	ԝ�`Xc�A�*

loss�b<�)��       �	�D�`Xc�A�*

loss��<��{       �	��`Xc�A�*

loss8�n;��IS       �	^��`Xc�A�*

losss�=d2eW       �	��`Xc�A�*

lossz��;�ܐ~       �	_��`Xc�A�*

loss)�<@�$6       �	3R�`Xc�A�*

loss�Ī;QP��       �	b��`Xc�A�*

losst%=4(�       �	��`Xc�A�*

lossrUI=Y7�       �	+�`Xc�A�*

loss���<�ݍV       �	���`Xc�A�*

loss4D <,�"}       �	Ef�`Xc�A�*

loss��!=�p       �	��`Xc�A�*

loss��l=�Ѽ&       �	���`Xc�A�*

loss~�<���       �	�D�`Xc�A�*

loss��=	��       �	9��`Xc�A�*

loss�<�1�       �	u��`Xc�A�*

loss!F=�+       �	n0�`Xc�A�*

loss#�=~�rj       �	9��`Xc�A�*

loss�\�= f/t       �	�m�`Xc�A�*

loss��= $��       �	��`Xc�A�*

loss���<��'q       �	q��`Xc�A�*

lossIj=.ڰi       �	|G�`Xc�A�*

loss���=�[       �	��`Xc�A�*

loss�ۆ;�r8�       �	׈�`Xc�A�*

loss��<\w�       �	�3�`Xc�A�*

loss�=t��       �	b��`Xc�A�*

loss�;Ĕ�<       �	�t�`Xc�A�*

loss�;[��       �	��`Xc�A�*

loss�<�<̂       �	��`Xc�A�*

loss6�/=�G       �	sJ�`Xc�A�*

loss���<q�g�       �	}��`Xc�A�*

loss��<�):L       �	V��`Xc�A�*

loss�}	=��       �	��`Xc�A�*

lossܥ\;=4e       �	
��`Xc�A�*

loss�e-=O�|�       �	q�`Xc�A�*

lossd�4=�r       �	��`Xc�A�*

lossz��;cGev       �	��`Xc�A�*

lossi�8<Q�d       �	?�`Xc�A�*

loss[À=���        �	���`Xc�A�*

loss�8�=�0z�       �	���`Xc�A�*

lossN�f=���       �	i��`Xc�A�*

loss�";�QR�       �	�4�`Xc�A�*

loss�Z�;a\��       �	s��`Xc�A�*

lossH�X=U��"       �	`u�`Xc�A�*

loss6��=�qh       �	/�`Xc�A�*

lossWc
=eX�       �	���`Xc�A�*

lossW�	<�>��       �	V�`Xc�A�*

loss���;�xI       �	��`Xc�A�*

loss3��<	��       �	n��`Xc�A�*

loss�J:;7�`       �	S>�`Xc�A�*

loss@%}=�       �	j��`Xc�A�*

lossq�<f�#       �	,}�`Xc�A�*

loss(NJ;g��       �	C�`Xc�A�*

loss�bV=&��/       �	c��`Xc�A�*

loss��=C
�?       �	�Q�`Xc�A�*

lossje�<*��       �	��`Xc�A�*

loss�Ƀ=S��       �	��`Xc�A�*

loss���;�	@       �	�6�`Xc�A�*

loss�F�;���       �	h��`Xc�A�*

loss��=��nT       �	�b�`Xc�A�*

lossJ4<�>f�       �	��`Xc�A�*

lossL��< ��O       �	�� aXc�A�*

loss�V=��       �	('aXc�A�*

loss4�=�tb�       �	�aXc�A�*

loss�Q=G8j       �	-ZaXc�A�*

lossSf�;G�;       �	��aXc�A�*

loss��0<@!�m       �	�aXc�A�*

loss��;E��d       �	_$aXc�A�*

loss�T<k]       �	��aXc�A�*

loss^�<U�ԃ       �	�QaXc�A�*

loss,�<:H=-�       �	��aXc�A�*

lossV��9Ѓ~\       �	�}aXc�A�*

loss�-{=���       �	aXc�A�*

loss�ٙ;��       �	��aXc�A�*

loss^��<��       �	@KaXc�A�*

loss��==?�@p       �	.�aXc�A�*

loss�7=�t0       �	�~	aXc�A�*

lossI�<x�DW       �	
aXc�A�*

loss�<<t�d       �	˺
aXc�A�*

loss!��<�ʧx       �	�SaXc�A�*

loss�y:�x�"       �	l�aXc�A�*

loss�`=�k�       �	Q�aXc�A�*

loss��R<�s�       �	aXc�A�*

loss��=��       �	��aXc�A�*

lossm}={e�       �	�VaXc�A�*

loss�6;���W       �	E�aXc�A�*

losso��=eQ\�       �	X�aXc�A�*

lossg�<Xũ�       �	V(aXc�A�*

loss<��;Ҡ       �	��aXc�A�*

loss쿋<���[       �	��aXc�A�*

lossn��<���       �	�XaXc�A�*

loss\O�;��)�       �	x�aXc�A�*

loss#l�=f��       �	aXc�A�*

loss�}==�}N)       �	�4aXc�A�*

loss��V<��#       �	z8aXc�A�*

loss�KL=h�n       �	�>aXc�A�*

loss��=�kj       �	�aXc�A�*

lossmu�<�m_�       �	U�aXc�A�*

loss?# <�K       �	��aXc�A�*

loss�h<�|D�       �	DRaXc�A�*

lossA�}=r�
8       �	^�aXc�A�*

loss�">=	�       �	�aXc�A�*

loss��%=���       �	��aXc�A�*

losst�=Lc��       �	�aXc�A�*

loss�$=6�
�       �	haXc�A�*

loss%V|<o�~.       �	<aXc�A�*

loss��<��ܢ       �	� aXc�A�*

loss:_<��c       �	�� aXc�A�*

loss��=�@��       �	�b!aXc�A�*

loss��<6���       �	��!aXc�A�*

loss�}
=���       �	I�"aXc�A�*

loss���<��[        �	�<#aXc�A�*

loss���<h�       �	��#aXc�A�*

loss�.�<ʌt�       �	\s$aXc�A�*

lossE�<7�X�       �	�%aXc�A�*

loss��'=�Z]       �	ɫ%aXc�A�*

loss�	�;{�?�       �	cE&aXc�A�*

lossIG�:�f��       �	�A'aXc�A�*

loss��<�#�A       �	��'aXc�A�*

losso�=���        �	��(aXc�A�*

loss^��=��%       �	0-)aXc�A�*

loss���<��WB       �	�)aXc�A�*

lossJ�>�fJ�       �	Um*aXc�A�*

loss/�<G� �       �	�+aXc�A�*

loss\՛<�K�       �	��+aXc�A�*

loss�=�<�2��       �	GU,aXc�A�*

loss��<#T��       �	��,aXc�A�*

loss��=����       �	��-aXc�A�*

loss�<ɦ�       �	�.aXc�A�*

loss|3�<�       �	vP/aXc�A�*

lossϜ�<J�~       �	Q�/aXc�A�*

loss@|�;s\�y       �	ę0aXc�A�*

loss��;(bd       �	+11aXc�A�*

lossԳm=5�m       �	��1aXc�A�*

loss!i�<de"       �	4h2aXc�A�*

lossp4<&-��       �	�3aXc�A�*

loss�R�=��l       �	��3aXc�A�*

loss���=�ؠQ       �	�R4aXc�A�*

loss�7=�PT        �	��4aXc�A�*

lossbl <�-yt       �	�5aXc�A�*

lossė�<��       �	�46aXc�A�*

loss�m-<;��4       �	��6aXc�A�*

loss+�=�v��       �	Bv7aXc�A�*

loss1�7=urn7       �	f8aXc�A�*

loss�n2<]ȆQ       �	;�8aXc�A�*

lossI<�_��       �	eV9aXc�A�*

loss�8~=��}�       �	�:aXc�A�*

loss;�<c��E       �	F%<aXc�A�*

lossҫh<��Η       �	H�<aXc�A�*

loss&�=��'_       �	�c=aXc�A�*

lossy�<g;l�       �	>aXc�A�*

loss�5A;E��^       �	��>aXc�A�*

loss�A�=��~�       �	!??aXc�A�*

loss� �=�m$�       �	`�?aXc�A�*

losshp�=��T       �	j�@aXc�A�*

loss�F<��Ld       �	i6AaXc�A�*

loss�E<L�:�       �	��AaXc�A�*

loss�{�<P:n�       �	[yBaXc�A�*

loss�t<�ѽD       �	�!CaXc�A�*

lossN�<o˷,       �	��CaXc�A�*

loss��=��X       �	TrDaXc�A�*

loss�,�=��=       �	?EaXc�A�*

loss-v};�        �	0�EaXc�A�*

loss�jf<��W       �	�QFaXc�A�*

loss*��=�x       �	��FaXc�A�*

loss7��< <�#       �	r�GaXc�A�*

loss>�;��A�       �	5%HaXc�A�*

lossѸ�<m�       �	�HaXc�A�*

loss��;���m       �	`WIaXc�A�*

loss��=<�8g�       �	��IaXc�A�*

losstӆ=��       �	��JaXc�A�*

loss�=Izɍ       �	$KaXc�A�*

loss׶:=iKw       �	��KaXc�A�*

loss��<�       �	�PLaXc�A�*

losse�	=�|�       �	��LaXc�A�*

loss&G4<?�/^       �	ׇMaXc�A�*

losst-Y<̯j       �	�NaXc�A�*

loss��7<���       �	��NaXc�A�*

loss�E;=��G       �	�MOaXc�A�*

loss�;��Z�       �	��OaXc�A�*

lossV��<�2$�       �	�PaXc�A�*

loss��<:��O       �	7QaXc�A�*

losstV;
m"�       �	I�QaXc�A�*

loss�u<�<E       �	�oRaXc�A�*

lossH��<� ��       �	�SaXc�A�*

lossZ�<{?�       �	�TaXc�A�*

loss��h<+D�       �	��TaXc�A�*

loss`�;=Lv��       �	 �UaXc�A�*

loss	=;�+�       �	-�VaXc�A�*

loss*�	=�=̅       �	�@WaXc�A�*

lossi�}=�;<�       �	{�WaXc�A�*

lossg7=��c       �	��XaXc�A�*

loss�0�<e4�       �	!ZaXc�A�*

loss�Ȟ<N#       �	h�ZaXc�A�*

loss�/<���#       �	o�[aXc�A�*

lossSj�<���W       �	�/\aXc�A�*

loss�I�<�        �	.�\aXc�A�*

loss�i�<-�=(       �	�^]aXc�A�*

lossO�o<�&l       �	�]aXc�A�*

loss���;d	�       �	Ő^aXc�A�*

loss��u;l>��       �	(_aXc�A�*

loss���<��S       �	Q�_aXc�A�*

loss�K�:I�	�       �	6W`aXc�A�*

loss�K�<��?N       �	��`aXc�A�*

loss:�<�r�       �	0�aaXc�A�*

loss�!=B���       �	�baXc�A�*

loss��8<b��       �	d�baXc�A�*

loss8X�=�%�k       �	�]caXc�A�*

loss$��<�K��       �	�daXc�A�*

lossS9<�'Sv       �	N�daXc�A�*

loss�ϣ<�Q`       �	�6eaXc�A�*

loss�<���       �		�eaXc�A�*

loss�I�;���B       �	SwfaXc�A�*

loss�[<t�[�       �	6"gaXc�A�*

loss�
i<�s��       �	h�gaXc�A�*

lossxb!=_y.       �	khaXc�A�*

loss:�;}�ǚ       �	�iaXc�A�*

loss*�<J���       �	��iaXc�A�*

loss4y;���       �	iRjaXc�A�*

loss���<yHC_       �	�jaXc�A�*

loss��=�d       �	W�kaXc�A�*

loss{,d;H=�       �	q�laXc�A�*

loss�nu=$A��       �	�6maXc�A�*

loss�:<.�       �	��maXc�A�*

loss�3�=�#G       �	�qnaXc�A�*

loss�	�;͆�       �	�ApaXc�A�*

loss]f�;�?       �	��paXc�A�*

loss��<�gڸ       �	�qaXc�A�*

loss��f=,�DD       �	�7saXc�A�*

loss��;+e�!       �	��saXc�A�*

loss��q>XW��       �	V�taXc�A�*

loss9=�-��       �	J%uaXc�A�*

loss��q<<��       �	9�uaXc�A�*

loss�=U��       �	�ovaXc�A�*

loss��*=�{       �	NwaXc�A�*

loss���;b�       �	�waXc�A�*

loss�J<u�%�       �	~RxaXc�A�*

loss�{�<�       �	��xaXc�A�*

loss�S�<Yrw       �	�yaXc�A�*

loss�E<R���       �	�(zaXc�A�*

loss�X�=�ܶ�       �	��zaXc�A�*

loss�� <�N~       �	�{aXc�A�*

loss�T<�u�m       �	�|aXc�A�*

lossA�<�к       �	�|aXc�A�*

lossM1�<�}�       �	%w}aXc�A�*

loss�=��-=       �	�(~aXc�A�*

loss��<n|       �	�~aXc�A�*

loss
V=��eq       �	ˆaXc�A�*

loss��<#��       �	C7�aXc�A�*

lossQ5?=�h�       �	��aXc�A�*

lossq��<�'_n       �	���aXc�A�*

loss�=����       �	B?�aXc�A�*

loss��#=�0��       �	��aXc�A�*

loss�P-<�U�       �	-��aXc�A�*

loss6�0<WQ[l       �	]7�aXc�A�*

loss��
<7B3�       �	/�aXc�A�*

loss�]<7��/       �	���aXc�A�*

loss�=Y<764       �	} �aXc�A�*

lossT��<��Ǥ       �	�p�aXc�A�*

lossϖL<1CLW       �	W[�aXc�A�*

lossϗ�;�#       �	O�aXc�A�*

loss�?�= ��A       �	䟉aXc�A�*

loss*�|=?n7�       �	�:�aXc�A�*

loss�,�=f��       �	��aXc�A�*

lossw#'=]v>2       �	���aXc�A�*

loss�5<˳       �	/N�aXc�A�*

loss���<-��       �	��aXc�A�*

loss~<�	�       �	Y��aXc�A�*

lossSn�<Wo5}       �	a��aXc�A�*

loss��f<aǹ       �	�h�aXc�A�*

loss��
;�O��       �	%�aXc�A�*

loss��=�#�F       �	���aXc�A�*

loss(`=�)��       �	B>�aXc�A�*

loss��<�f�e       �	�ّaXc�A�*

loss�;�Rx       �	\t�aXc�A�*

loss��b=
^��       �		�aXc�A�*

loss�/�<��ig       �	]��aXc�A�*

loss�a=j�       �	`?�aXc�A�*

loss��j<|�Ea       �	�+�aXc�A�*

loss���<��:       �	%[�aXc�A�*

lossf��<g�V�       �	N�aXc�A�*

loss�u<�9k       �	��aXc�A�*

loss2.p<M��E       �	 &�aXc�A�*

lossDX�<��g       �	2ǘaXc�A�*

loss�s�;�c        �	Uh�aXc�A�*

loss�S�<�|=�       �	��aXc�A�*

loss�5X<�v��       �	{��aXc�A�*

loss���<�\�       �	�L�aXc�A�*

loss�v�=N�	       �	0��aXc�A�*

loss��j;�Ax�       �	��aXc�A�*

lossXu�<XT:       �	�-�aXc�A�*

lossr=��[       �	��aXc�A�*

losstF�<��e�       �	Z��aXc�A�*

loss=L-=�TY�       �	%$�aXc�A�*

lossb݇<-r��       �	vßaXc�A�*

lossOF�<UY       �	�]�aXc�A�*

loss{�G=Ux�       �	�r�aXc�A�*

lossi:�==W�M       �	��aXc�A�*

loss��m<�
V�       �	v��aXc�A�*

loss0 =��L�       �	AG�aXc�A�*

lossԟq=��|�       �	��aXc�A�*

loss滷<Z3%       �	�|�aXc�A�*

loss�9<=GA�       �	Y�aXc�A�*

loss�><q:M       �	���aXc�A�*

loss�P=��K�       �	�H�aXc�A�*

loss��M<r�8�       �	��aXc�A�*

loss8x<n�I�       �	+��aXc�A�*

loss�l�<����       �	.�aXc�A�*

lossrǨ;�`       �	д�aXc�A�*

loss�|-=���[       �	/P�aXc�A�*

loss��n=q��       �	�aXc�A�*

loss,��<�KS�       �	���aXc�A�*

loss�=�	��       �	~6�aXc�A�*

loss���<��A�       �	N��aXc�A�*

loss�K
=��Y�       �	Y��aXc�A�*

loss鬪;��_�       �	�aXc�A�*

loss�N</�,       �	˭aXc�A�*

lossp��<�U~       �	�e�aXc�A�*

loss� �;�`�       �	���aXc�A�*

loss��k<�M�       �	�m�aXc�A�*

loss��=9��       �	�aXc�A�*

loss��<Æ��       �	[�aXc�A�*

loss�l�<�a�l       �	��aXc�A�*

loss�r<�=W       �	P��aXc�A�*

loss���;�6�       �	Z/�aXc�A�*

loss��e<}���       �	\ɴaXc�A�*

loss�V=�
�G       �	S��aXc�A�*

lossL�K<XA��       �	bK�aXc�A�*

lossRE2=¥f�       �	�\�aXc�A�*

losse�8= ��       �	Y2�aXc�A�*

loss�-h;���       �	ZظaXc�A�*

loss��<w��'       �	J}�aXc�A�*

loss4LK=@�3�       �	d#�aXc�A�*

lossQ+<���6       �	�ǺaXc�A�*

loss��=d�R�       �	�`�aXc�A�*

loss��
<�_��       �	Q��aXc�A�*

loss�h�=�u��       �	���aXc�A�*

loss�S;�m�a       �	?�aXc�A�*

loss���:\Ӿ�       �	��aXc�A�*

loss�<i�{_       �	D��aXc�A�*

loss��<^�WA       �	� �aXc�A�*

losst�m<�B!       �	�ɿaXc�A�*

losscN=*��       �	zl�aXc�A�*

loss�=�<^�b�       �	W�aXc�A�*

loss�t�;��ů       �	��aXc�A�*

lossht�<�v�       �	�C�aXc�A�*

loss�Ɂ<;���       �	z��aXc�A�*

loss�Y�;S쁕       �	���aXc�A�*

loss��;���       �	��aXc�A�*

loss�T|=�c��       �	-%�aXc�A�*

loss~��<6f       �	¾�aXc�A�*

loss|�<�ַ�       �	GV�aXc�A�*

loss �i<��       �	J��aXc�A�*

loss:��;�v_f       �	߉�aXc�A�*

loss�J=���       �	�2�aXc�A�*

loss-;W7q       �	t��aXc�A�*

loss���=�?N�       �	��aXc�A�*

loss�F:<��lm       �	,)�aXc�A�*

loss��D=�8��       �	���aXc�A�*

lossIϸ=�e��       �	�h�aXc�A�*

loss�<|���       �	��aXc�A�*

lossą�;�2       �	���aXc�A�*

loss�eS;�:/       �	T7�aXc�A�*

loss��<1�       �	���aXc�A�*

lossa�=|��       �	���aXc�A�*

loss
k=�)��       �	�,�aXc�A�*

loss ��=&��       �	���aXc�A�*

loss�z�<�1�G       �	�g�aXc�A�*

loss�~<�M��       �	
�aXc�A�*

lossh�<F-�       �	��aXc�A�*

lossq3�<ä7�       �	�i�aXc�A�*

loss�#<6^�       �	��aXc�A�*

loss���<�@�r       �	Q��aXc�A�*

lossɸ�<��k'       �	@m�aXc�A�*

loss�%�<zNӈ       �	��aXc�A�*

loss3@^<�}��       �	D�aXc�A�*

loss��y=b�`R       �	��aXc�A�*

loss4��;^}�(       �	/��aXc�A�*

loss�34;� �x       �	��aXc�A�*

loss�:�<����       �	��aXc�A�*

lossRkh=��r       �	|B�aXc�A�*

loss�;=���r       �	H��aXc�A�*

loss��/=E	��       �	b��aXc�A�*

loss�μ<	       �	;�aXc�A�*

loss|I*=@<       �	���aXc�A�*

loss���<Y�       �	�S�aXc�A�*

losse8�<V9�       �	���aXc�A�*

losslb<j^��       �	i��aXc�A�*

lossa��<��3       �	W%�aXc�A�*

loss?��:d�y�       �	���aXc�A�*

loss#<�=Z&�       �	�d�aXc�A�*

lossRkM=� �       �	*��aXc�A�*

loss}�=�gϒ       �	ݘ�aXc�A�*

loss�S6<���       �	S<�aXc�A�*

loss9��;��`       �	���aXc�A�*

lossO��=�pX�       �	e��aXc�A�*

loss��;�,�       �	�(�aXc�A�*

loss#��<㟅A       �	���aXc�A�*

loss�H<~���       �	k�aXc�A�*

losshYE<k�.�       �	�aXc�A�*

loss @<�B]J       �	X��aXc�A�*

loss4�=u�>       �	uU�aXc�A�*

loss��<ƪd8       �	���aXc�A�*

loss@UV=��       �	I��aXc�A�*

loss237<8��       �	yY�aXc�A�*

loss��w<���       �	
1�aXc�A�*

loss��_=v�E       �	��aXc�A�*

loss���<Q�%�       �	f�aXc�A�*

losseLG;��#       �	��aXc�A�*

lossS�q;)W��       �	7��aXc�A�*

lossiX5=)��       �	�O�aXc�A�*

loss��*=��       �	�L�aXc�A�*

loss�T=�@X�       �	���aXc�A�*

loss��t=;�       �	�{�aXc�A�*

loss�]�=�[�2       �	�$�aXc�A�*

lossL;l�       �	i��aXc�A�*

loss�/�<�,�       �	;n�aXc�A�*

lossfSL<�@u<       �	��aXc�A�*

loss���;
��Q       �	Q��aXc�A�*

lossD�:~       �	΍�aXc�A�*

loss6�<���t       �	%�aXc�A�*

loss�z�<ָ�-       �	��aXc�A�*

lossH	�<6�U�       �	Mh�aXc�A�*

lossw�;�M��       �		�aXc�A�*

lossN0:�BW�       �	}��aXc�A�*

loss=f�<;Z!^       �	�[�aXc�A�*

loss?:K�       �	���aXc�A�*

loss��:h�r�       �	���aXc�A�*

loss�P8ݱo�       �	�0�aXc�A�*

loss<l       �	���aXc�A�*

loss�ހ<A�ծ       �	Ef�aXc�A�*

loss,�R=�h�       �	z��aXc�A�*

lossO��;����       �	���aXc�A�*

lossh1<���M       �	�4�aXc�A�*

loss��>TJ��       �	���aXc�A�*

loss��a<���       �	Oy�aXc�A�*

loss}5=_ژ�       �	8�aXc�A�*

lossq�m=(�B|       �	;��aXc�A�*

loss�q@;nW       �	F bXc�A�*

loss��]<�z       �	� bXc�A�*

loss�K�<�Ä       �	�|bXc�A�*

loss�{�<�Ŏ       �	�bXc�A�*

lossS?�;�$J�       �	ͬbXc�A�*

loss�a;��4       �	�LbXc�A�*

lossR٥<���N       �	��bXc�A�*

loss��<Y�^�       �	T�bXc�A�*

loss/:>=��:       �	�$bXc�A�*

lossrT>=�$�       �	]�bXc�A�*

loss�XP=��nV       �	cbXc�A�*

loss�Ή=��!�       �	bXc�A�*

loss.�)= �&       �	x�bXc�A�*

loss�|@<��       �	'0bXc�A�*

loss�*x;!َ       �	6�bXc�A�*

lossBY=����       �	�_	bXc�A�*

lossf�='���       �	��	bXc�A�*

loss�V;��       �	6�
bXc�A�*

loss�r�<\l��       �	/0bXc�A�*

loss[!<�X%i       �	^�bXc�A�*

loss2��;�g'�       �	�vbXc�A�*

lossp��<���       �	�bXc�A�*

lossɘ�<��       �	i�bXc�A�*

loss���=���       �	{MbXc�A�*

loss��;<��       �	�bXc�A�*

loss_*F=c��=       �	B�bXc�A�*

losss<=5m/�       �	w1bXc�A�*

lossF�4<�Y߈       �	��bXc�A�*

loss&#<�JW�       �	xbXc�A�*

loss
K;�       �	�bXc�A�*

loss��<Oº        �	"�bXc�A�*

loss�@r<N�J�       �	�CbXc�A�*

loss�V<##��       �	��bXc�A�*

lossz�o;�uV       �	jbXc�A�*

loss�W�<Ŋ�w       �	o�bXc�A�*

loss��>=1���       �	�ZbXc�A�*

loss��6<J'(�       �	�2bXc�A�*

loss�8;���       �	��bXc�A�*

lossS=D��       �	v�bXc�A�*

loss鎽<xG�!       �	.bXc�A�*

loss|ߋ<*rX�       �	bXc�A�*

loss6�<z~       �	:"bXc�A�*

lossN|$<�>�@       �	��bXc�A�*

loss4nk=��g�       �	�ibXc�A�*

loss��=b�[       �	bXc�A�*

loss/�<�-�       �	��bXc�A�*

loss/�<��l       �	)vbXc�A�*

loss��<����       �	vbXc�A�*

loss=a�<qt�@       �	4<bXc�A�*

loss/{ =W^�       �	-�<bXc�A�*

loss�Y[=,h&s       �	�J=bXc�A�*

losss�5<��l�       �	`�=bXc�A�*

loss��=��˒       �	ԁ>bXc�A�*

loss���;2���       �	?bXc�A�*

loss��E<�$��       �	o�?bXc�A�*

lossm�=�)�"       �	mT@bXc�A�*

loss5�=��&�       �	k�@bXc�A�*

loss�*�<�Ӟ       �	ۉAbXc�A�*

loss6��;���       �	W#BbXc�A�*

losse�<%�3(       �	��BbXc�A�*

lossn%=���       �	PUCbXc�A�*

loss���<�ل       �	J�CbXc�A�*

loss�m<�       �	<�DbXc�A�*

loss���; g�&       �	�EbXc�A�*

loss�ր:�y�N       �	�FbXc�A�*

loss��<A�L�       �	�FbXc�A�*

lossT�<��t�       �	�bGbXc�A�*

loss�D�=���{       �	�HbXc�A�*

loss���;��8.       �	��HbXc�A�*

loss,��<�Q��       �	t�IbXc�A�*

lossS��<n���       �	�CJbXc�A�*

loss�l=�K��       �	��JbXc�A�*

loss��V<�/�o       �	<�KbXc�A�*

loss$�:K}��       �	�.LbXc�A�*

loss�<��       �	��LbXc�A�*

loss��;��,�       �	șMbXc�A�*

loss��M=���I       �	�>NbXc�A�*

loss�xb=Mx��       �	��NbXc�A�*

loss1r�<��       �	�}ObXc�A�*

loss��<�4ێ       �	UPbXc�A�*

lossq�d<��}{       �	��PbXc�A�*

loss[�=�g�v       �	�GQbXc�A�*

loss"�;�       �	*�QbXc�A�*

loss���=/�yd       �	ҨRbXc�A�*

loss�p�;����       �	�bSbXc�A�*

loss��	=l��       �	
TbXc�A�*

loss�P=�!�       �	հTbXc�A�*

loss�Q=����       �	��UbXc�A�*

loss��<0
h�       �	[�VbXc�A�*

loss��<3�2       �	�XbXc�A�*

lossw�<d��       �	o�XbXc�A�*

lossZ�}<�n�       �	^�YbXc�A�*

loss�J�<	i��       �	��ZbXc�A�*

lossĊq=���       �	:�[bXc�A�*

loss��;ZubB       �	w1\bXc�A�*

lossD��=��bV       �	�]bXc�A�*

loss.^><�"�}       �		T^bXc�A�*

loss�ޥ;ͼ�       �	�_bXc�A�*

lossz�u;��        �	��_bXc�A�*

loss���<�W�       �	n`bXc�A�*

loss8�K<L�>�       �	�abXc�A�*

loss��=���       �	D�abXc�A�*

loss&��<ﱷ       �	��bbXc�A�*

loss���;͑��       �	�HcbXc�A�*

loss���9�R2y       �	X�cbXc�A�*

loss��-;��Ж       �	 }dbXc�A�*

loss�e�;�;U       �		ebXc�A�*

lossÙ�=i�       �	8�ebXc�A�*

loss
�'<�3E�       �	�SfbXc�A�*

lossx~�;�Y��       �	��fbXc�A�*

loss�P�;���Q       �	�gbXc�A�*

loss���<U��       �	�ChbXc�A�*

loss���<����       �	��hbXc�A�*

losstx|;�<       �	F�ibXc�A�*

loss
A<t3��       �	�LjbXc�A�*

lossD��;���       �	��jbXc�A�*

loss �=_w�       �	�kbXc�A�*

lossv��=1�	       �	U3lbXc�A�*

loss%�I=�QTH       �	��lbXc�A�*

loss)��<_-�       �	�\mbXc�A�*

loss4-�;�j��       �	9�mbXc�A�*

loss�[~<'��       �	H�nbXc�A�*

lossL��=۪g       �	�)obXc�A�*

loss���;t�t�       �	U�obXc�A�*

lossm��<��ב       �	�`pbXc�A�*

loss�T?<RΌN       �	��pbXc�A�*

loss�nZ;���       �	��qbXc�A�*

loss��=��B�       �	*;rbXc�A�*

loss�8�<�ð       �	�sbXc�A�*

loss��<ð�]       �	zubXc�A�*

loss��f<���       �	#�ubXc�A�*

loss�I;;���       �	�ZvbXc�A�*

loss�m;6�&       �	�vbXc�A�*

lossn�=l��       �	t_xbXc�A�*

loss���<(�g�       �	ybXc�A�*

loss}=<��       �	��ybXc�A�*

loss4=�-j       �	ƧzbXc�A�*

lossS�;��9       �	b�{bXc�A�*

loss0ƀ<�a��       �	!;|bXc�A�*

lossM�=b��O       �	k�|bXc�A�*

loss*%�;��0}       �	t}bXc�A�*

loss^�;�j�h       �	�~bXc�A�*

lossش=7���       �	�~bXc�A�*

loss�;d-�       �	VFbXc�A�*

loss�m�<���3       �	K�bXc�A�*

loss*��<9��       �	��bXc�A�*

loss4��<¢x       �	��bXc�A�*

loss�[p;N�r�       �	n�bXc�A�*

loss$�7;DKOB       �	᷂bXc�A�*

loss��:�L��       �	]�bXc�A�*

loss&8z;ͷ��       �	���bXc�A�*

loss�(=6��       �	ǜ�bXc�A�*

loss�ϣ<����       �	T9�bXc�A�*

loss�`/={�*�       �	�܅bXc�A�*

loss ^:/�0�       �	g{�bXc�A�*

loss��.;5�x       �	��bXc�A�*

loss���<��]C       �	��bXc�A�*

lossׇq<[�/       �	���bXc�A�*

lossw�<��}N       �	�^�bXc�A�*

loss��	<gc��       �	]��bXc�A�*

loss�D�<�;V       �	���bXc�A�*

loss�^%;��?[       �	�-�bXc�A�*

loss��<��A       �	KȋbXc�A�*

loss3��9(7��       �	�h�bXc�A�*

loss|.<t~�_       �	8I�bXc�A�*

lossh�<x��       �	��bXc�A�*

lossx��;I.�       �	Q��bXc�A�*

loss�H;�&73       �	)�bXc�A�*

lossĠ�<^���       �	<��bXc�A�*

loss�';݂u       �	���bXc�A�*

loss��<!�F       �	9|�bXc�A�*

loss&2�<T�       �	��bXc�A�*

loss�Ӄ<�.6r       �	<��bXc�A�*

loss?�K=2G��       �	Z�bXc�A�*

loss79k<o�D       �	DN�bXc�A�*

loss��:�        �	�bXc�A�*

loss�J=:E�       �	��bXc�A�*

loss�Jm;����       �	I�bXc�A�*

loss�F�<�o�8       �	�bXc�A�*

loss\��<�Yx       �	�bXc�A�*

loss.&h<�ؖq       �	�T�bXc�A�*

loss�2�<�P�S       �	'��bXc�A�*

loss��1=iQ.       �	�Q�bXc�A�*

loss�2Z<�F�       �	4I�bXc�A�*

loss�)�<�K       �	��bXc�A�*

loss�_;�8       �	%�bXc�A�*

loss�4#=�[�       �	$�bXc�A�*

loss��;h�,X       �	{��bXc�A�*

loss�q:^{:�       �	Z*�bXc�A�*

loss��L<�F��       �	fk�bXc�A�*

lossl��<��       �	��bXc�A�*

lossjJ�="D`       �	w��bXc�A�*

loss�W=|�A(       �	���bXc�A�*

loss�'�<*Ty�       �	cӦbXc�A�*

loss(�<����       �	#�bXc�A�*

loss��h;瑧�       �	��bXc�A�*

loss;��;�� f       �	���bXc�A�*

loss��<m��       �	� �bXc�A�*

lossNcf;���       �	f��bXc�A�*

lossj�B<6��       �	W^�bXc�A�*

lossTf�<���       �	m�bXc�A�*

loss�>�=R��       �	~ĭbXc�A�*

loss�G�<��V       �	qW�bXc�A�*

loss�:"S:       �	p�bXc�A�*

loss�z<��N       �	���bXc�A�*

loss�Ш<���       �	Sv�bXc�A�*

loss{�k=%p��       �	��bXc�A�*

lossl5';d�}�       �	zȱbXc�A�*

loss$[�<�V{�       �	}�bXc�A�*

loss���<��Q�       �	\��bXc�A�*

losspT<���       �	��bXc�A�*

loss}��:̱��       �	��bXc�A�*

loss��<�[�2       �	�y�bXc�A�*

lossAm<SB��       �	��bXc�A�*

lossV:���s       �		�bXc�A�*

lossi<�j��       �	��bXc�A�*

loss��=^��       �	�,�bXc�A�*

loss���;�M��       �	Zh�bXc�A�*

lossAf5=B��        �	)�bXc�A�*

lossZ}�:� �       �	��bXc�A�*

lossQ,	:6Yc�       �	�8�bXc�A�*

lossC�<
��'       �	�ҽbXc�A�*

loss�hB;&��       �	�i�bXc�A�*

lossq��;d4��       �	� �bXc�A�*

loss��=�k�       �	���bXc�A�*

loss�q�=�9�       �	�2�bXc�A�*

loss�0�<�?�0       �	S��bXc�A�*

loss�5@;;V'8       �	Uk�bXc�A�*

loss�V;�#�4       �	��bXc�A�*

loss��X<}��       �	���bXc�A�*

loss���;pCu       �	|F�bXc�A�*

loss���:D��c       �	��bXc�A�*

loss��=�N��       �	��bXc�A�*

loss8`�< ��v       �	��bXc�A�*

loss3=����       �	���bXc�A�*

loss���;�q�       �	HQ�bXc�A�*

lossZ�=��       �	B��bXc�A�*

loss,>�<�(�y       �	ˁ�bXc�A�*

loss��<�q[�       �	r�bXc�A�*

loss��<Gk/N       �	ѱ�bXc�A�*

lossd;�i�       �	g�bXc�A�*

loss�/P=�Ń�       �	��bXc�A�*

loss�c6:�U�%       �	/��bXc�A�*

loss��<֓n�       �	J�bXc�A�*

loss��O=%,)       �	4��bXc�A�*

loss�O�<��       �	X��bXc�A�*

loss׫;�A�       �	L5�bXc�A�*

loss��;��q       �	���bXc�A�*

lossJ�,<y��       �	Gu�bXc�A�*

losscS{<˫_       �	;�bXc�A�*

loss��G=.�z       �	F��bXc�A�*

loss��p<���7       �	1`�bXc�A�*

loss�c�;$đ       �	� �bXc�A�*

loss�U <�nfx       �	v��bXc�A�*

loss��Z=v�x�       �	�j�bXc�A�*

lossރ�;�^jz       �	c�bXc�A�*

loss�&I=.�	�       �	`��bXc�A�*

loss�<��E       �	,J�bXc�A�*

loss�V=q�(h       �	~��bXc�A�*

loss��<r^�        �	Û�bXc�A�*

lossm^=<�$       �	�L�bXc�A�*

loss&\�;H�$`       �	���bXc�A�*

loss�hK=����       �	+��bXc�A�*

loss(� =,��       �	t��bXc�A�*

lossόl=���       �	��bXc�A�*

loss~�<�czq       �	�l�bXc�A�*

loss�.�=�7�       �	��bXc�A�*

loss��P=��2�       �	���bXc�A�*

lossL�<�}s       �	=��bXc�A�*

loss�*�<BT�       �	�@�bXc�A�*

loss߾�;�-�K       �	*��bXc�A�*

lossUe�<���       �	j��bXc�A�*

loss	�K<����       �	,-�bXc�A�*

lossʁd<���       �	P��bXc�A�*

loss�y~<���       �	�f�bXc�A�*

loss��n;ѝ��       �	Q�bXc�A�*

loss�S�<�<L(       �	���bXc�A�*

loss3�:e�D@       �	�F�bXc�A�*

lossS�:��       �	���bXc�A�*

loss�1<	�$       �	��bXc�A�*

loss%~�;��(�       �	$�bXc�A�*

losseF�;F�       �	��bXc�A�*

lossz�<r�(       �	0��bXc�A�*

loss�Ǌ<E�K       �	\�bXc�A�*

loss�g$<�*��       �	
H�bXc�A�*

lossڄ<v��=       �	���bXc�A�*

loss;��<D��       �	���bXc�A�*

loss�U�:|�XM       �	���bXc�A�*

loss���<�e��       �	�y�bXc�A�*

lossFO�<"���       �	��bXc�A�*

lossO9;[1�       �	e��bXc�A�*

lossZ�y<���       �	�D�bXc�A�*

loss�M<s@�W       �	���bXc�A�*

lossӤ;".*�       �	��bXc�A�*

loss�F<.WL�       �	�#�bXc�A�*

loss�}<$��       �	5�bXc�A�*

lossO�3=��.       �	5��bXc�A�*

loss�=��4       �	��bXc�A�*

lossr3C=�?��       �	�F�bXc�A�*

loss�f\=���       �	O��bXc�A�*

lossj�<
$mf       �	�}�bXc�A�*

lossz.<d��-       �	�D�bXc�A�*

loss��9��I�       �	H��bXc�A�*

loss1��<���       �	�u�bXc�A�*

lossv��<B�"       �	��bXc�A�*

lossf��<7�2        �	x(�bXc�A�*

loss�g�<3@/       �	#��bXc�A�*

loss�<b��       �	�m�bXc�A�*

lossn�z<��d�       �	�
�bXc�A�*

lossŘ�; ��$       �	��bXc�A�*

loss�(<M�       �	�f�bXc�A�*

losss^A=1�       �	��bXc�A�*

loss`^�<v���       �	��bXc�A�*

loss�uB:���       �	�H�bXc�A�*

loss���=>LR
       �	6��bXc�A�*

loss�n;=Gc�       �	���bXc�A�*

loss��I=&=�i       �	�T�bXc�A�*

loss��<�+Tf       �	���bXc�A�*

loss(ǒ<�̿�       �	�� cXc�A�*

loss���<ʊau       �	�,cXc�A�*

lossQ�r<TF;z       �	�cXc�A�*

loss8��;ÚRL       �	�[cXc�A�*

loss֨X=�^�       �	w�cXc�A�*

loss���<̡f�       �	��cXc�A�*

loss#<�I��       �	�cXc�A�*

loss��:���U       �	'�cXc�A�*

loss�t�<�_<�       �	qcXc�A�*

loss�B�<��K       �	�	cXc�A�*

loss��;f��;       �	f�cXc�A�*

lossCe�;ҵ       �	7cXc�A�*

loss@�u<c�       �	�cXc�A�*

lossi�8=����       �	�gcXc�A�*

loss�ݜ=l�{       �	�	cXc�A�*

loss��<%9�       �	��	cXc�A�*

loss��<��/       �	�=
cXc�A�*

loss��=A��A       �	�
cXc�A�*

loss^<�;��F       �	TtcXc�A�*

loss&Ա9�c       �	�cXc�A�*

loss�":��9�       �	��cXc�A�*

losst/�;�p�       �	CcXc�A�*

loss�%�:X�G�       �	��cXc�A�*

loss3?�:u=:       �	�ycXc�A�*

loss��=��       �	$cXc�A�*

lossȒ�;�\��       �	�cXc�A�*

lossh��<��L       �	c`cXc�A�*

loss��=��       �	BcXc�A�*

loss�%=[��M       �	d�cXc�A�*

loss�=C�s       �	KXcXc�A�*

loss�J�<���I       �	5�cXc�A�*

lossD��<�>�3       �	l�cXc�A�*

lossa�w;%g       �	mWcXc�A�*

loss�
>��       �	�1cXc�A�*

loss�^�<���G       �	�	cXc�A�*

loss;��<��+�       �	ԻcXc�A�*

loss�];��       �	�vcXc�A�*

loss�@�=ԒC�       �	9DcXc�A�*

loss��=����       �	�cXc�A�*

lossn�;���       �	��cXc�A�*

loss��;��G�       �	3cXc�A�*

loss]�<	�G       �	T�cXc�A�*

loss��9<�5�       �	�ecXc�A�*

lossi��;��)B       �	/�cXc�A�*

loss�$<�ƈ#       �	q�cXc�A�*

loss��<�~�       �	�&cXc�A�*

lossL΅;!�       �	Q�cXc�A�*

lossq��;c� N       �	�OcXc�A�*

loss>�=*��       �	��cXc�A�*

loss:�~=���@       �	${cXc�A�*

lossE�=enF�       �	�` cXc�A�*

lossm�=��       �	�� cXc�A�*

loss���<�!:        �	��!cXc�A�*

lossN9�;�7       �	(E"cXc�A�*

loss��I=6�\�       �	��"cXc�A�*

loss`1=�޹r       �	׈#cXc�A�*

loss�;�S�       �	�&$cXc�A�*

lossx��;�H�       �	��$cXc�A�*

loss?��<��
       �	)]%cXc�A�*

lossZ��<����       �	��%cXc�A�*

loss��;u5	8       �	��&cXc�A�*

loss��a<��"/       �	y"'cXc�A�*

loss��	<�ڒ�       �	_�'cXc�A�*

loss��;S*��       �	�P(cXc�A�*

loss�)�=$'�       �	u�(cXc�A�*

loss��/;wX��       �	s)cXc�A�*

loss^��<<cV�       �	�*cXc�A�*

loss�y�<��       �	��*cXc�A�*

loss��D=h;�\       �	�F+cXc�A�*

loss�f<��6�       �	��+cXc�A�*

loss���;7�       �	�t,cXc�A�*

loss��~;`T��       �	�
-cXc�A�*

loss-Ք<��       �	�-cXc�A�*

loss���<�ɕ�       �	�6.cXc�A�*

loss_y	=~�       �	S�.cXc�A�*

lossW��<#=Ή       �	Di/cXc�A�*

loss�F�;�%:�       �	�0cXc�A�*

loss�=�pRW       �	|�0cXc�A�*

loss;�1=�c`�       �	&41cXc�A�*

loss��r<q       �	4�1cXc�A�*

lossz(;�/R#       �	��2cXc�A�*

loss=Ѵ<r406       �	�*3cXc�A�*

loss��;V�       �	 �3cXc�A�*

loss���<�҆�       �	˄4cXc�A�*

lossØ6=�%3�       �	G5cXc�A�*

lossvc�<(hW�       �	S�5cXc�A�*

loss��:��V_       �	i6cXc�A�*

loss{�k<�K�B       �	�7cXc�A�*

loss_J�<��c�       �	!�7cXc�A�*

loss���<-��       �	�G8cXc�A�*

loss�#=��Tv       �	��8cXc�A�*

lossT5;�       �	'�9cXc�A�*

loss�%=�)ӱ       �	q :cXc�A�*

loss]=���p       �	��:cXc�A�*

loss��f;�k��       �	�P;cXc�A�*

loss*�<�$�       �	��;cXc�A�*

loss��<��       �	��<cXc�A�*

loss&�,;����       �	�=cXc�A�*

loss�]d<U�Щ       �	��=cXc�A�*

loss�Z�;���       �	�>cXc�A�*

lossXT1=I��k       �	nP?cXc�A�*

loss]r=,o�        �	�6@cXc�A�*

loss���;#:�0       �	#�@cXc�A�*

lossF~m=��C       �	U�BcXc�A�*

losss��=U]�       �	�iCcXc�A�*

loss.�=���       �	� DcXc�A�*

loss�;�<D�F       �	��DcXc�A�*

loss��<��X�       �	�:EcXc�A�*

loss�&�<��h�       �	-�EcXc�A�*

loss�rS=0dD       �	@kFcXc�A�*

loss{�]<(⭾       �	%GcXc�A�*

loss��l;$�        �	�GcXc�A�*

loss��={�w       �	.HcXc�A�*

loss,�j<���       �	�*IcXc�A�*

loss[�k;}�1�       �	��IcXc�A�*

loss?��=�>��       �	�sJcXc�A�*

loss%=TG�       �	�KcXc�A�*

loss��<va:�       �	��KcXc�A�*

loss��<�y�'       �	l?LcXc�A�*

loss	6=<��dI       �	E�LcXc�A�*

loss.y�;���       �	�tMcXc�A�*

lossa�b;���       �	�NcXc�A�*

loss��4<7g$�       �	t�NcXc�A�*

lossF]=�y�n       �	I.OcXc�A�*

loss��a=�|�       �	�OcXc�A�*

lossv�=�u        �	:WPcXc�A�*

lossnK�:I��,       �	��PcXc�A�*

loss<�N<%�Yv       �	��QcXc�A�*

loss�z�<���       �	�RcXc�A�*

loss�V�<9���       �	1�RcXc�A�*

losshe<Y��       �	�PScXc�A�*

loss��<}���       �	��ScXc�A�*

lossmq1=�T�M       �	��TcXc�A�*

loss��=�ү�       �	�UcXc�A�*

loss�8�=z�M�       �	�UcXc�A�*

loss���<`���       �	�HVcXc�A�*

lossA�<t�o�       �	�VcXc�A�*

lossT��;��>       �	�{WcXc�A�*

loss ��<�<�       �	�XcXc�A�*

loss4�;���       �	4�XcXc�A�*

lossȇ�<0q       �	�7YcXc�A�*

lossN7=˚ƞ       �	l�YcXc�A�*

loss�s3=[�<       �	�fZcXc�A�*

loss!v�=�)/F       �	?�ZcXc�A�*

loss��<�R�       �	w�[cXc�A�*

lossQ��< �+F       �	�6\cXc�A�*

loss���<҂��       �	��\cXc�A�*

loss��<�2F       �	�l]cXc�A�*

loss��A=	9m�       �	Y^cXc�A�*

loss})h=!r�P       �	g�^cXc�A�*

losse8$<=�y/       �	�N_cXc�A�*

loss!W<c,��       �	1�_cXc�A�*

loss���;���P       �	o�`cXc�A�*

lossj�	=�q�       �	�acXc�A�*

loss�3�=�n��       �	��acXc�A�*

loss���=3�       �	*sbcXc�A�*

loss���=t-v�       �	�ccXc�A�*

loss{'�<�F|<       �	s�dcXc�A�*

loss >�;��#�       �	�6ecXc�A�*

loss�);=I�S       �	��ecXc�A�*

lossF��<$b�       �	�hfcXc�A�*

lossb�<xV�p       �	gcXc�A�*

loss��:R�-�       �	��gcXc�A�*

loss�9;<��7�       �	RHhcXc�A�*

loss,�I<���       �	�hcXc�A�*

loss��;3�t       �	E�icXc�A�*

loss@�a<T!�u       �	PjcXc�A�*

lossL(�<p���       �	%�jcXc�A�*

lossߨ�;��1�       �	$HkcXc�A�*

loss�+�<�nv�       �	?�kcXc�A�*

loss�G	=� W       �	�zlcXc�A�*

loss4<�<�K�b       �	 mcXc�A�*

loss|�=��.n       �	8�mcXc�A�*

loss�y<�(3�       �	�SncXc�A�*

loss�n!<����       �	��ncXc�A�*

lossF�=��       �	�ocXc�A�*

loss�8=&� z       �	�HpcXc�A�*

loss��C<�$       �	�pcXc�A�*

loss�Gy;��:�       �	"rqcXc�A�*

lossm.�<]��       �	�
rcXc�A�*

loss���=�t�       �	b�rcXc�A�*

loss�ǣ;��o{       �	AscXc�A�*

loss͸D<��:�       �	��scXc�A�*

lossstK<��c�       �	�xtcXc�A�*

loss��=J�$       �	�!ucXc�A�*

loss��=o�       �	��ucXc�A�*

loss벉<�IO       �	�WvcXc�A�*

loss/�=ʫ�#       �	�wcXc�A�*

loss��<RN       �	^�xcXc�A�*

loss�޲<���       �	w,ycXc�A�*

loss �<"C��       �	��ycXc�A�*

loss��$=U(�       �	�gzcXc�A�*

lossTld;(���       �	��zcXc�A�*

loss4��<f;�        �	
�{cXc�A�*

loss��;�#��       �	�O|cXc�A�*

loss��{<��A       �	��|cXc�A�*

loss��:���       �	[�}cXc�A�*

lossa;�M�       �	�4~cXc�A�*

lossɌD<�|��       �	��~cXc�A�*

loss�)<��q�       �	icXc�A�*

loss�מ=Iw       �	�N�cXc�A�*

loss\	�<��       �	��cXc�A�*

loss�$=�5�       �	=�cXc�A�*

lossᶞ;�o�q       �	P�cXc�A�*

lossr��;���       �	ͬ�cXc�A�*

lossQD=���       �	�f�cXc�A�*

loss��;�$iI       �	���cXc�A�*

loss{��<%��m       �	2��cXc�A�*

loss`A)=�˛�       �	C�cXc�A�*

loss(õ= =ȟ       �	��cXc�A�*

lossP��=ī�       �	���cXc�A�*

loss�Z�<�Iٳ       �	�2�cXc�A�*

loss���;{��       �	ȇcXc�A�*

loss!�M<��|#       �	}w�cXc�A�*

lossV�5;�.#�       �	�!�cXc�A�*

loss�ep=@�kH       �	�̉cXc�A�*

loss|;�\�       �	�n�cXc�A�*

loss,�%=�O �       �	x�cXc�A�*

loss�ə;���Q       �	n��cXc�A�*

loss��4=�er�       �	�B�cXc�A�*

loss��<5PO�       �	�ԌcXc�A�*

lossҔ,=�&Y�       �	i�cXc�A�*

lossii�:�       �	i��cXc�A�*

lossȬ=��VU       �	���cXc�A�*

loss�
<�%��       �	=C�cXc�A�*

loss���:���       �	[�cXc�A�*

loss=Q=�s�#       �	���cXc�A�*

loss�NJ=
��       �	nk�cXc�A�*

loss�eh<o���       �	v��cXc�A�*

loss�:=���X       �	oגcXc�A�*

lossϞ�<�!y�       �	��cXc�A�*

lossɔ;a{!2       �	D��cXc�A�*

loss�%=K���       �	i�cXc�A�*

loss�*�;�s�       �	9��cXc�A�*

loss@ �<�2�       �	E��cXc�A�*

lossw;��G       �	)��cXc�A�*

lossWO^=F �       �	�W�cXc�A�*

loss���;�kt       �	D�cXc�A�*

loss4B�<�[o       �	�/�cXc�A�*

loss��=�$QD       �	9�cXc�A�*

lossJ��:�O�       �	ߌ�cXc�A�*

loss3=���0       �	�%�cXc�A�*

loss���;��       �	~ȜcXc�A�*

lossa�i< �^w       �	F`�cXc�A�*

loss�V<��       �	���cXc�A�*

loss��<��l�       �	-��cXc�A�*

loss�&=�;�       �	#.�cXc�A�*

loss?�;M�o       �	4֟cXc�A�*

loss�b_<S&�       �	t��cXc�A�*

loss �K<��5�       �	C:�cXc�A�*

loss��S;kb       �	�ԡcXc�A�*

loss:k<Մ�;       �	c
�cXc�A�*

loss%�9E���       �	f��cXc�A�*

lossa�C<�t~�       �	�9�cXc�A�*

loss|��<epy       �	�֤cXc�A�*

loss���;�k�       �	Cs�cXc�A�*

loss7��<x��^       �	��cXc�A�*

loss�YD=ϽZ       �	靦cXc�A�*

lossr�=��xx       �	�9�cXc�A�*

loss�!�;\K��       �	<ާcXc�A�*

loss���<̀�0       �	Z��cXc�A�*

loss�W�<c�       �	z4�cXc�A�*

loss��<�ou       �	�өcXc�A�*

loss��93�ws       �	�r�cXc�A�*

loss�ϗ:4��~       �	�h�cXc�A�*

loss*~N;7�yi       �	��cXc�A�*

lossH��;��J       �	���cXc�A�*

loss[oa<��9k       �	P�cXc�A�*

loss^�;�~�       �	��cXc�A�*

loss9�;�筜       �	���cXc�A�*

lossd��9���       �	pA�cXc�A�*

loss6t:��       �	g��cXc�A�*

loss�!u8����       �	u��cXc�A�*

loss;6�;�"�I       �	�;�cXc�A�*

lossm�<�+G�       �	�t�cXc�A�*

loss��;�\�       �	��cXc�A�*

loss#:�;�:W       �	鸳cXc�A�*

loss�]n:�� �       �	^�cXc�A�*

loss�D=�#       �	쥵cXc�A�*

loss�}�;h�g]       �	F�cXc�A�*

loss@O=�ϰ�       �	:�cXc�A�*

loss&�{=�1\       �	���cXc�A�*

loss<�<�?qg       �	�L�cXc�A�*

loss:c:=��f       �	��cXc�A�*

loss|��;H��       �	���cXc�A�*

loss��%<��       �	�I�cXc�A�*

loss���<�+�
       �	r��cXc�A�*

loss��<��R       �	���cXc�A�*

loss�i�<���       �	�`�cXc�A�*

loss�T,;�j�       �	
�cXc�A�*

loss@!A<���/       �	ýcXc�A�*

loss�#�<D�'       �	sf�cXc�A�*

loss,��<�_c       �	�cXc�A�*

loss,c<�a#�       �	�ɿcXc�A�*

loss"��<ޯb�       �	�w�cXc�A�*

loss��<B��       �	�v�cXc�A�*

loss�=�̐�       �	��cXc�A�*

lossz�	=��/       �	��cXc�A�*

lossq��:"��u       �	�b�cXc�A�*

loss�;���
       �	���cXc�A�*

loss�5<��C       �	ӥ�cXc�A�*

lossd�<~pO�       �	�<�cXc�A�*

loss��m;ʹ�B       �	>��cXc�A�*

loss)}�<�b       �	���cXc�A�*

loss�ѳ;�(�\       �	(�cXc�A�*

loss8>i<t@       �	��cXc�A�*

losst�<��Q�       �	�b�cXc�A�*

loss_W�<����       �	���cXc�A�*

lossVRS=q&d�       �	���cXc�A�*

lossl9=���	       �	�,�cXc�A�*

loss<-�L       �	���cXc�A�*

loss��Z=R4��       �	Ku�cXc�A�*

loss>��;�0��       �	��cXc�A�*

loss�#�<�*L       �	~��cXc�A�*

loss���:��       �	�F�cXc�A�*

loss�ޛ<j��C       �	���cXc�A�*

loss�C�<�?��       �	9{�cXc�A�*

loss�^�<��       �	0�cXc�A�*

losso*�<�Q%.       �	��cXc�A�*

lossSp�<u��P       �	rO�cXc�A�*

loss�D�<0ja�       �	-��cXc�A�*

loss�i=��Mk       �	���cXc�A�*

loss�4�<�,b�       �	K#�cXc�A�*

lossN��:3S;	       �	<��cXc�A�*

loss�+\<9+       �	Y�cXc�A�*

loss�#�<`'G       �	��cXc�A�*

loss�$v;H;�       �	v��cXc�A�*

loss(��<�
�       �	H��cXc�A�*

lossR�<$E�       �	���cXc�A�*

loss��><�֔       �	�F�cXc�A�*

loss仅<�.m'       �	���cXc�A�*

loss��='A{�       �		o�cXc�A�*

loss��V<� u       �	�cXc�A�*

loss�}<�;��       �	��cXc�A�*

loss7<�$~�       �	 B�cXc�A�*

loss!�;9M��       �	���cXc�A�*

loss��;)-F�       �	9�cXc�A�*

loss�4
=v��z       �	��cXc�A�*

loss.U+>�v\�       �	���cXc�A�*

loss-<���       �	8��cXc�A�*

loss���;��V       �	nn�cXc�A�*

loss��<U)|�       �	��cXc�A�*

loss;�\<f�E�       �	���cXc�A�*

loss%S�<�h��       �	�U�cXc�A�*

loss�T�<�U�       �	���cXc�A�*

loss4B�<%l�       �	��cXc�A�*

lossAM';y��       �	i�cXc�A�*

loss]�<~X�       �	O��cXc�A�*

loss��<]�_�       �	�I�cXc�A�*

loss|0<�fw�       �	���cXc�A�*

loss��6<��_�       �	!t�cXc�A�*

loss�-R=/�l�       �	��cXc�A�*

loss�7�;h�$       �	���cXc�A�*

loss���<�N�       �	�H�cXc�A�*

loss�pp<�qj�       �	���cXc�A�*

loss�U�;D�8�       �	�w�cXc�A�*

loss��j=�k&       �	� dXc�A�*

loss��J<����       �	� dXc�A�*

loss���;d�]�       �	�VdXc�A�*

lossſ8<1S~�       �	n�dXc�A�*

lossr��;݀)2       �	�dXc�A�*

loss�}�<�Y�>       �	0dXc�A�*

loss�P&<��       �	�dXc�A�*

loss�,=|�G#       �	�bdXc�A�*

loss�M�;���9       �	dXc�A�*

loss�^i=����       �	9�dXc�A�*

loss��C;�O       �	3dXc�A�*

loss��;��?       �	��dXc�A�*

loss� �=#�       �	$bdXc�A�*

loss��=���n       �	�dXc�A�*

losso�j;���       �	�dXc�A�*

loss��<@��       �	pB	dXc�A�*

loss��C;8y��       �	��	dXc�A�*

loss)M7<ɡO�       �	Xp
dXc�A�*

loss,�N<F��       �	VdXc�A�*

loss#��;�I        �	Q�dXc�A�*

loss�9�<��       �	>dXc�A�*

loss܉}<����       �	�dXc�A�*

loss���;�h�H       �	�zdXc�A�*

loss���9z��       �	�dXc�A�*

lossս;e��9       �	.�dXc�A�*

loss9t=��       �	�DdXc�A�*

loss)�:{Ĝ       �	��dXc�A�*

loss�V�=a	`       �	�tdXc�A�*

loss�i<�c�        �	K;dXc�A�*

lossiU
;��&       �	o�dXc�A�*

loss��<;D�
�       �	odXc�A�*

loss��9�da       �	�dXc�A�*

loss�ږ=5       �	ۢdXc�A�*

losstG=L�f       �	�@dXc�A�*

loss��j<n�i#       �	w�dXc�A�*

loss2x�:�s�        �	�qdXc�A�*

loss{��:��Z       �	N~dXc�A�*

loss�=���7       �	S=dXc�A�*

lossN�0;�i       �	�dXc�A�*

lossc��=mq�       �	TtdXc�A�*

loss�|=Zh��       �	dXc�A�*

loss�7�<�qA�       �	��dXc�A�*

lossO�?=�kH�       �	�<dXc�A�*

lossf�s=#N}       �	1�dXc�A�*

loss^�= 
��       �	8�dXc�A�*

loss�- =��:^       �	]dXc�A�*

loss4��;�A�       �	��dXc�A�*

loss!�<OQ{)       �	|GdXc�A�*

lossF�=�ПE       �	?�dXc�A�*

loss� <$��       �	��dXc�A�*

loss���<s}Lx       �	�dXc�A�*

loss@��;ܗp5       �	Q�dXc�A�*

loss@�c;�)       �	]S dXc�A�*

loss �z<��v       �	�� dXc�A�*

loss��<�U��       �	*�!dXc�A�*

loss46�<�b       �	#"dXc�A�*

loss���;�a��       �	��"dXc�A�*

lossZ�;`��"       �	�Y#dXc�A�*

loss4<���*       �	�#dXc�A�*

loss^N<ۑ�       �	v�$dXc�A�*

loss�=1�ii       �	=+%dXc�A�*

lossv|<<{�d�       �	��%dXc�A�*

lossF]�;�Ƃ9       �	�^&dXc�A�*

loss��<SW �       �	I�&dXc�A�*

loss�E-<�g�l       �		�'dXc�A�*

loss�E�<�ā�       �	p&(dXc�A�*

loss4�I;]�o�       �	��(dXc�A�*

lossi`�;oJ�$       �	�Y)dXc�A�*

lossx8R=v�!�       �	,�)dXc�A�*

lossW��;J�Fh       �	��*dXc�A�*

loss���=�T9}       �	�!+dXc�A�*

loss���;f��       �	��+dXc�A�*

loss��9=G�       �	�d,dXc�A�*

loss�ΐ<'�       �	a�,dXc�A�*

loss��;����       �	l�-dXc�A�*

lossI�h::|��       �	V+.dXc�A�*

loss|:<#��       �	:�.dXc�A�*

loss��H<l�       �	��/dXc�A�*

loss)Y6= #q       �	�60dXc�A�*

loss�s�<���       �	q�0dXc�A�*

lossZ�:���S       �	�y1dXc�A�*

loss?W�<���       �	e2dXc�A�*

loss?I>3\�p       �	�2dXc�A�*

lossTY�<����       �	�M3dXc�A�*

loss<c���       �	z�3dXc�A�*

lossF�;�zH       �	��4dXc�A�*

loss2@�<�߃�       �	[]5dXc�A�*

loss�j;�	�M       �	��5dXc�A�*

loss4��;�f��       �	N�6dXc�A�*

loss��;���o       �	27dXc�A�*

loss�<�4�s       �	e�7dXc�A�*

loss�=�֠�       �	|�8dXc�A�*

loss���<�z�       �	�89dXc�A�*

loss\= �gG       �	��9dXc�A�*

lossn��<��G�       �	Um:dXc�A�*

loss`�<#x       �	:;dXc�A�*

loss,��<��ā       �	�;dXc�A�*

loss,X<��1U       �	ڐ<dXc�A�*

lossw��<��M�       �	)=dXc�A�*

loss��2=���       �	y�=dXc�A�*

lossJƗ=L[:]       �	be>dXc�A�*

lossf�;6��t       �	a�>dXc�A�*

loss���<�{@       �	��?dXc�A�*

loss�~;&�L�       �	&8@dXc�A�*

loss��<�zY9       �	
BdXc�A�*

lossH�<ξ.       �	��BdXc�A�*

loss�G�<J�"6       �	�fCdXc�A�*

loss���<��I|       �	�CdXc�A�*

loss�5 =.��       �	�DdXc�A�*

loss�X�;����       �	`=EdXc�A�*

lossF��;���       �	x�EdXc�A�*

loss��B;E�9+       �	[zFdXc�A�*

loss���<�~��       �	�"GdXc�A�*

loss���;���       �	��GdXc�A�*

loss���:�b	.       �	�lHdXc�A�*

loss@R=�V�
       �	IdXc�A�*

loss���=�O��       �	v�IdXc�A�*

loss_��;Bp/>       �	�HJdXc�A�*

loss
�=ʙ�\       �	@�JdXc�A�*

loss�OP<CiJ       �	�yKdXc�A�*

loss�s<>�%�       �	�LdXc�A�*

loss�A�:�i6$       �	�LdXc�A�*

loss��f;�&�       �	� OdXc�A�*

loss4F;U��       �	,�OdXc�A�*

loss��<-��}       �	)<PdXc�A�*

loss=F�<\�[c       �	��PdXc�A�*

lossX��<��]       �	�rQdXc�A�*

loss?�4=�1��       �	dRdXc�A�*

lossyۚ<�DO)       �	�RdXc�A�*

loss�=:����       �	�/SdXc�A�*

loss��:�h�       �	3�SdXc�A�*

loss�>�P�       �	tbTdXc�A�*

loss/,E=^���       �	7�TdXc�A�*

loss��;��M�       �	��UdXc�A�*

loss'�	=���       �	�@VdXc�A�*

loss���:[7P�       �	l�VdXc�A�*

loss�֝<�3�       �	O�WdXc�A�*

loss=�U:iό�       �	�SXdXc�A�*

loss�-=�«       �	�XdXc�A�*

loss�\�<�0%       �	��YdXc�A�*

loss���:��I�       �	�QZdXc�A�*

loss��T<�
�_       �	��ZdXc�A�*

loss�s!; m       �	S�[dXc�A�*

lossF��;sQ       �	G�\dXc�A�*

loss6��<�n7       �	]�]dXc�A�*

loss=l:�ܪ�       �	A�^dXc�A�*

loss)D;��8       �	�`dXc�A�*

loss��Q<sN��       �		�`dXc�A�*

loss�-�;XuA       �	��adXc�A�*

losss
�<Zr�]       �	�]bdXc�A�*

loss,@=#��_       �	ucdXc�A�*

loss��=B���       �	�cdXc�A�*

loss�<�Y=       �	@ddXc�A�*

lossl�:\�j~       �	z�ddXc�A�*

lossY :!^�       �	��edXc�A�*

loss%��<�J�       �	�fdXc�A�*

lossD��<���[       �	��fdXc�A�*

lossq.<���       �	aQgdXc�A�*

lossd��:�V�       �	��gdXc�A�*

lossHp�<�i=�       �	Y�hdXc�A�*

lossLtb<T�֣       �	<idXc�A�*

losst�;�c`�       �	��idXc�A�*

lossZ��<��       �	�pjdXc�A�*

lossJ�P<y�       �	-	kdXc�A�*

loss8� =]��       �	0�kdXc�A�*

loss��;Y�~       �	/4ldXc�A�*

loss
�l;s�`u       �	��ldXc�A�*

loss�!<gU       �	kemdXc�A�*

lossf�;����       �	� ndXc�A�*

loss��<�S^       �	�ndXc�A�*

loss���;����       �	#,odXc�A�*

lossa�<w6�       �	a�odXc�A�*

lossd<�;ѻI�       �	�apdXc�A�*

lossԈ;��0�       �	��pdXc�A�*

lossڒ�<�S�       �	�qdXc�A�*

loss��<)��       �	�VrdXc�A�*

loss��(;��X       �	[�rdXc�A�*

lossʞY;�o�       �	@�sdXc�A�*

loss��;��5�       �	�-tdXc�A�*

loss�4�;���       �	��tdXc�A�*

loss/��<ڪ��       �	Y�udXc�A�*

loss\$=0�       �	_wdXc�A�*

lossE&<��`�       �	�xdXc�A�*

lossݍ�<�i�N       �	}�xdXc�A�*

losss��;��       �	?sydXc�A�*

lossԁ�;�F�       �	��zdXc�A�*

loss�@q;���)       �	�Y{dXc�A�*

loss8��<&@�s       �	��{dXc�A�*

loss4�c>F/       �	��|dXc�A�*

loss3��={�mz       �	q<}dXc�A�*

loss[�^<�c:�       �	��}dXc�A�*

loss�%"=_���       �	ۅ~dXc�A�*

loss�%�=eY�       �	�2dXc�A�*

loss���<Ŗ"�       �	��dXc�A�*

loss��\<]��b       �	�z�dXc�A�*

lossn�;4�)+       �	@�dXc�A�*

lossJ��<A��       �	�dXc�A�*

lossy~=�&�e       �	Q�dXc�A�*

loss�4�<V9�5       �	xz�dXc�A�*

loss���;�(       �	U�dXc�A�*

loss��.=�l�       �	T��dXc�A�*

loss�<1W�       �	ZI�dXc�A�*

loss��F= ��       �	?�dXc�A�*

lossi�<F���       �	-z�dXc�A�*

lossrQ�<qI��       �	\�dXc�A�*

lossm��;B��       �	��dXc�A�*

loss}C�=^�6       �	�l�dXc�A�*

lossF��<���       �	@�dXc�A�*

loss U�<��p�       �	��dXc�A�*

loss�o]=9v.�       �	�G�dXc�A�*

lossv�<2D��       �	5�dXc�A�*

loss��=!�Q�       �	���dXc�A�*

loss:-=�ag�       �	�$�dXc�A�*

loss2�:x1�       �	L��dXc�A�*

loss��<~��d       �	�\�dXc�A�*

lossi`�;��
�       �	��dXc�A�*

loss�@T=؄�       �	L��dXc�A�*

loss��^<�� 1       �	�L�dXc�A�*

loss��|<N�Q�       �	"�dXc�A�*

loss�n?<�m       �	s��dXc�A�*

loss��<�G1=       �	=+�dXc�A�*

loss<��<䯲�       �	���dXc�A�*

loss��<Q�       �	CV�dXc�A�*

loss���<�1�       �	,�dXc�A�*

lossd�<=���       �	߇�dXc�A�*

loss!�Y=Y/ f       �	G�dXc�A�*

lossϐ�<Hv       �	�ڕdXc�A�*

loss��^<��       �	���dXc�A�*

lossI-�<��Ԃ       �	4M�dXc�A�*

loss��p<��Y&       �	��dXc�A�*

loss�{�:��G�       �	���dXc�A�*

loss�:�<�3��       �	n0�dXc�A�*

loss�}=��d�       �	�˙dXc�A�*

loss�{=��]�       �	�d�dXc�A�*

loss^p�<s�b       �	q �dXc�A�*

loss���<�Y�b       �	��dXc�A�*

loss<ߴ;ܦ�6       �	o-�dXc�A�*

loss(*�=:�D�       �	NќdXc�A�*

loss�M=�Lӌ       �	h�dXc�A�*

losss�e:��       �	���dXc�A�*

lossx�%=��a       �	���dXc�A�*

loss���;vB�B       �	 7�dXc�A�*

lossQ=�^�2       �	ԟdXc�A�*

loss���<q~       �	m�dXc�A�*

losst<7���       �	F	�dXc�A�*

loss�F<p]G�       �	�ޡdXc�A�*

lossT��<e��       �	w�dXc�A�*

loss�q�<&��G       �	D�dXc�A�*

losszI�=&Q�       �	:��dXc�A�*

lossj%�<:IX       �	=C�dXc�A�*

loss�<i��r       �	K�dXc�A�*

loss��:]e       �	V�dXc�A�*

loss�֝<�u�       �	��dXc�A�*

loss�=�LI�       �	0��dXc�A�*

loss�X�:��k       �	S�dXc�A�*

loss��J<��2r       �	���dXc�A�*

lossw�;��*@       �	���dXc�A�*

loss�v�;W�       �	0,�dXc�A�*

losso�=����       �	�̩dXc�A�*

loss���;�Ĺ        �	�f�dXc�A�*

loss��;&�       �	�dXc�A�*

loss(ś<��       �	��dXc�A�*

loss�^;q�
5       �	�5�dXc�A�*

loss���;z8[B       �	�̬dXc�A�*

loss,o�;8)�b       �	$c�dXc�A�*

loss�'<��G       �	���dXc�A�*

loss��&<�S>�       �	P��dXc�A�*

loss��:<r�	       �	B&�dXc�A�*

lossL��<#�:�       �	HįdXc�A�*

loss�|�<T�=�       �	KZ�dXc�A�*

loss�O�;Qt       �	9�dXc�A�*

lossSyZ<ո��       �	Ҍ�dXc�A�*

loss��2=�v��       �	g(�dXc�A�*

lossm}=��ٰ       �	.ǲdXc�A�*

lossD��<�+�,       �	�o�dXc�A�*

loss-�m;��*�       �	��dXc�A�*

loss�ޞ;���       �	���dXc�A�*

loss���=�c�       �	�`�dXc�A�*

lossS��<q�O�       �	 
�dXc�A�*

loss�=�հ�       �	-��dXc�A�*

loss�7�:����       �	,E�dXc�A�*

losso��=�5       �	u�dXc�A�*

loss�
<uU�b       �	I��dXc�A�*

loss���;���f       �	Q1�dXc�A�*

loss���;M$ m       �	�͹dXc�A�*

lossS��;�?Ѳ       �	m�dXc�A�*

loss�==<��       �	1�dXc�A�*

lossQrm:$ê�       �	X��dXc�A�*

loss�Z;nO^       �	
H�dXc�A�*

lossģ�<�-       �	]߼dXc�A�*

lossec<G�	�       �	pz�dXc�A�*

losszN�<���/       �	��dXc�A�*

loss�<��Z       �	���dXc�A�*

loss�ܶ<I��]       �	�L�dXc�A�*

loss��+<j�z       �	d�dXc�A�*

loss�+e=1e�       �	A�dXc�A�*

loss�%�<%<�       �	��dXc�A�*

loss<�;
�       �	c��dXc�A�*

loss�<���p       �	H�dXc�A�*

loss��t<��6       �	q��dXc�A�*

loss�n�<���2       �	���dXc�A�*

loss��=Vb       �	[&�dXc�A�*

loss��=K��       �	׽�dXc�A�*

loss؉�<�#       �	�]�dXc�A�*

loss/&;X"Z=       �	���dXc�A�*

loss�ّ<��yG       �	��dXc�A�*

loss��=+�       �	%�dXc�A�*

loss QG;��l       �	6��dXc�A�*

loss&��<����       �	�o�dXc�A�*

loss׆8;���       �	��dXc�A�*

loss�&�<��       �	��dXc�A�*

loss�i3=w'�       �	�M�dXc�A�*

loss�}=�˰       �	g��dXc�A�*

lossZ��;Ti�f       �	���dXc�A�*

loss��:�v]�       �	#.�dXc�A�*

loss�� ;��}�       �	���dXc�A�*

loss���<����       �	9b�dXc�A�*

loss���;q��%       �	���dXc�A�*

loss�k�=��       �	���dXc�A�*

loss3�N=�F?\       �	k�dXc�A�*

loss$�;�%I       �	�#�dXc�A�*

lossn�<N���       �	
��dXc�A�*

loss�W�<#���       �	3T�dXc�A�*

loss�r	<�(�       �	���dXc�A�*

loss�1=;��p�       �	��dXc�A�*

loss�T~<)0z       �	�%�dXc�A�*

loss��Z<���       �	
��dXc�A�*

loss���;�欋       �	
0�dXc�A�*

loss��6=��?�       �	D��dXc�A�*

loss%�W=���!       �	Y��dXc�A�*

lossO�;�'�j       �	N�dXc�A�*

loss��&=�\�a       �	|��dXc�A�*

loss�ws;�}       �	�^�dXc�A�*

loss��m<�R:       �	M�dXc�A�*

loss��;�31       �	�X�dXc�A�*

lossP�;��0       �	���dXc�A�*

lossm<��N�       �	gH�dXc�A�*

loss-i"=�c#�       �	;��dXc�A�*

loss9=�6       �	@��dXc�A�*

loss�=|���       �	[{�dXc�A�*

loss�=DS8�       �	���dXc�A�*

loss��;�Z]�       �	2v�dXc�A�*

loss�+�;�,T&       �	l$�dXc�A�*

loss���<4(`�       �	��dXc�A�*

lossfxN;���       �	em�dXc�A�*

lossc+<�8��       �	��dXc�A�*

loss��;=���j       �	��dXc�A�*

lossF;L�fz       �	�U�dXc�A�*

loss��<���       �	3��dXc�A�*

lossZ�=LB       �	���dXc�A�*

loss���<�*`[       �	�:�dXc�A�*

loss�XU<Oy�       �	���dXc�A�*

loss��<�w�       �	]o�dXc�A�*

loss;fK<(Ǡh       �	~�dXc�A�*

loss�)g;�߅A       �	��dXc�A�*

lossZ��;��u       �	HQ�dXc�A�*

losseFO=�M}       �	w��dXc�A�*

loss��
<�T�       �	���dXc�A�*

lossF�e<�˅�       �	�+�dXc�A�*

loss �4<!�       �	B��dXc�A�*

loss؄�<���       �	���dXc�A�*

loss�'o<�b�       �	%x�dXc�A�*

loss� �<�a�E       �	�dXc�A�*

loss7g�;�ҝ�       �	��dXc�A�*

loss�:��5�       �	q�dXc�A�*

loss��;�β       �	G��dXc�A�*

lossC"�<0ӿ<       �	�j�dXc�A�*

loss���;��i�       �	 �dXc�A�*

loss�<�;��(       �	���dXc�A�*

loss.�<�H�;       �		O�dXc�A�*

lossm�};���       �	A��dXc�A�*

loss�;�M(n       �	e��dXc�A�*

loss�% =�p�       �	8,�dXc�A�*

loss��;{�       �	 ��dXc�A�*

loss�޺:��{       �	`�dXc�A�*

lossM8�;K8��       �	���dXc�A�*

lossE4�<�<��       �	,��dXc�A�*

loss�I:N��       �	3�dXc�A�*

loss���=��%       �	`��dXc�A�*

loss���<��n       �	F_�dXc�A�*

loss�=h�q+       �	��dXc�A�*

loss��=F"x       �	���dXc�A�*

loss��;�֑�       �	�3�dXc�A�*

loss�=Փ�       �	c��dXc�A�*

loss�z;<&m`       �	'i�dXc�A�*

loss�)5=�0       �	 eXc�A�*

loss��<�G��       �	+� eXc�A�*

loss�b^=��B       �	:eXc�A�*

loss�w�;o���       �	��eXc�A�*

loss�b=�/��       �	ReXc�A�*

loss�˙;W�^�       �	�eXc�A�*

loss�2<"��       �	�eXc�A�*

loss��<pG(       �	sLeXc�A�*

loss&��<����       �	!�eXc�A�*

loss1�<E흈       �	��eXc�A�*

loss���;�E�Z       �	�3eXc�A�*

loss�.;�{4`       �	��eXc�A�*

loss��8<��"       �	��eXc�A�*

loss�=���       �	k(eXc�A�*

loss�H<��I       �	O�eXc�A�*

loss�K=X        �	�e	eXc�A�*

loss��M;I��       �	W
eXc�A�*

lossz <�-       �	��
eXc�A�*

loss8Ϡ<���       �	82eXc�A�*

lossx=0��       �	p�eXc�A�*

loss�ݍ=<���       �	PneXc�A�*

lossX*;۝$       �	�eXc�A�*

loss�o�<'Р�       �	ÝeXc�A�*

lossn�</d	       �	�6eXc�A�*

loss���=����       �	3�eXc�A�*

loss���;a��       �	#�eXc�A�*

loss[�~<�N�E       �	�.eXc�A�*

loss*��<t�~       �	&�eXc�A�*

loss��=�gj�       �	keXc�A�*

loss�I�<W�+�       �	6eXc�A�*

loss��5=���i       �	 *eXc�A�*

loss���=`���       �	��eXc�A�*

lossB��=�E]u       �	�]eXc�A�*

loss��;\I�c       �	��eXc�A�*

loss?�u<�u��       �	T�eXc�A�*

loss�s<,o�J       �	(*eXc�A�*

lossSI<+�tJ       �	��eXc�A�*

lossͭ�<|e��       �	�aeXc�A�*

loss�&/<E��       �	�eXc�A�*

loss
&d=s�6�       �	)�eXc�A�*

lossN��;��s�       �	�^eXc�A�*

loss\{<#(:f       �	�eXc�A�*

loss��{<'�2�       �	+�eXc�A�*

loss<�K=uj;       �	�FeXc�A�*

lossIE�<'$�Z       �	�eXc�A�*

lossl�<�]Mn       �	(~eXc�A�*

loss�ڨ<�5^       �	IeXc�A�*

loss,��;I9	2       �	C�eXc�A�*

lossQ�q=ɹ+�       �	�DeXc�A�*

loss���;
X       �	�eXc�A�*

lossT=�:�MΡ       �	�|eXc�A�*

loss�a�;�vrt       �	� eXc�A�*

loss��=!#       �	i� eXc�A�*

loss@�)=�[�       �	�@!eXc�A�*

loss�<v�0       �	{�!eXc�A�*

loss�aH;r�       �	*q"eXc�A�*

loss�H[<��       �	�#eXc�A�*

loss��<<#Kg       �	�#eXc�A�*

loss�r�;��q3       �	T$eXc�A�*

losst�=�6�D       �	;%eXc�A�*

loss�М;�F�z       �	��%eXc�A�*

loss)�=�4>       �	@l&eXc�A�*

loss���<���T       �	e'eXc�A�*

loss�R�;,*�       �	��'eXc�A�*

loss� 5<�A�r       �	0+(eXc�A�*

loss4m"<�whO       �	��(eXc�A�*

loss��:��>       �	�Z)eXc�A�*

losstS�<�!��       �	��)eXc�A�*

lossF��=�P��       �	*eXc�A�*

loss�q=5\W�       �	�!+eXc�A�*

loss6˃;�d��       �	9�+eXc�A�*

loss�է;*#��       �	�P,eXc�A�*

loss�S�<�٩�       �	��,eXc�A�*

loss E�;(��L       �	��-eXc�A�*

lossC"<�!��       �	�.eXc�A�*

lossȳ�;���n       �	��.eXc�A�*

lossT�<8�޶       �	�M/eXc�A�*

loss:�c;��       �	��/eXc�A�*

loss�4<@�R{       �	a�0eXc�A�*

lossh��<Ƚ��       �	�2eXc�A�*

loss_!<��G       �	]�2eXc�A�*

loss�;�ĕ$       �	�Z3eXc�A�*

loss�)<D*c�       �	z�3eXc�A�*

loss$-�;��S#       �	��4eXc�A�*

loss-z0=�U       �	2V5eXc�A� *

loss��;h��u       �	��5eXc�A� *

lossh�0<*��       �	��6eXc�A� *

loss��<���Y       �	f17eXc�A� *

lossn�=���       �	p�7eXc�A� *

loss1��=�Eo|       �	�8eXc�A� *

loss�}Y<
�g�       �	��9eXc�A� *

loss*̑<�r�       �	�L:eXc�A� *

loss��;�
|       �	^�:eXc�A� *

lossL
=�/�+       �	0�;eXc�A� *

loss߬�<^(AB       �	�8<eXc�A� *

losswA<��`       �	t�>eXc�A� *

loss�,�;[pbQ       �	�s?eXc�A� *

loss�=�,g.       �	k@eXc�A� *

loss��=\��       �	v�@eXc�A� *

loss�&f:�P��       �	fMAeXc�A� *

loss��;�,       �	�AeXc�A� *

loss%<���u       �	'�BeXc�A� *

loss�XC=w��y       �	�%CeXc�A� *

loss��=�.       �		�CeXc�A� *

loss�[<�Ц       �	�bDeXc�A� *

loss��M<�Bb       �	��DeXc�A� *

lossإ;�S�       �	�EeXc�A� *

loss�<ꢪ       �	r1FeXc�A� *

lossq�u<���       �	i�FeXc�A� *

lossڤ;C�       �	�^GeXc�A� *

loss���;h�ψ       �	��GeXc�A� *

losss��;ʠ�4       �	q�HeXc�A� *

lossݒ<��L�       �	�)IeXc�A� *

losst[=Ψ��       �	�IeXc�A� *

loss�;Eq6L       �	:WJeXc�A� *

loss�%"<�u       �	��KeXc�A� *

loss�4�<8D��       �	�LeXc�A� *

loss
>�<j�|�       �	GwMeXc�A� *

loss?i�<� �y       �	NeXc�A� *

loss��:�2[       �	��NeXc�A� *

loss!�;l��C       �	YOeXc�A� *

loss���9���`       �	L�OeXc�A� *

loss."�9b�L�       �	�PeXc�A� *

loss��:LQu�       �	�+QeXc�A� *

loss��$<��       �	P�QeXc�A� *

loss���;��b       �	alReXc�A� *

loss�:C;
�/�       �	�SeXc�A� *

loss��8dަ�       �	��SeXc�A� *

loss��7<��B@       �	_aTeXc�A� *

loss��9/�       �	UeXc�A� *

loss?"�7�E�       �	��UeXc�A� *

loss�7�(�-       �	v5VeXc�A� *

lossU�<Ơ�        �	�WeXc�A� *

lossA$<қY�       �	-XeXc�A� *

lossi�T<�̍�       �	��XeXc�A� *

loss��<��(       �	HmYeXc�A� *

lossw0�;��g�       �	,.ZeXc�A� *

loss0�=�i<�       �	�g[eXc�A� *

lossFh�:�v�       �	�	\eXc�A� *

loss1��<e�̘       �	�]eXc�A� *

losst?<�:�       �	,�]eXc�A� *

loss\̍<?,�O       �	��^eXc�A� *

lossH'
;��e�       �	+N_eXc�A� *

lossz�>=���       �	Kt`eXc�A� *

loss�p�=v��       �	oaeXc�A� *

loss��8<-�7       �	��aeXc�A� *

loss�=�{�       �	�PbeXc�A� *

loss�c�<{��       �	��beXc�A� *

loss��</�       �	��ceXc�A� *

lossl�^<�%�       �	�deXc�A� *

lossB}='�S�       �	%�deXc�A� *

loss=6�;@�       �	�FeeXc�A� *

loss6�Q<�h�~       �	��eeXc�A� *

loss��[=��}�       �	�|feXc�A� *

loss��:)PM       �	�geXc�A� *

loss���<�f�b       �	��geXc�A� *

loss�=jH��       �	�VheXc�A� *

loss��b:۰C�       �	��heXc�A� *

loss��:�7߁       �	P�ieXc�A� *

loss���<���       �	�@jeXc�A� *

loss��|<ؔ��       �	��keXc�A� *

loss��:n��S       �	�6leXc�A� *

loss&�x<��u�       �	Q�leXc�A� *

loss
 _:��'4       �	��meXc�A� *

loss�$�<����       �	�'neXc�A� *

loss���;o���       �	!�neXc�A� *

loss��w<�[�       �	�goeXc�A� *

lossO�=CP��       �	�peXc�A� *

lossX��;r���       �	�peXc�A� *

loss�["=Ə9�       �	�JqeXc�A� *

lossT�D<��2�       �	;�qeXc�A� *

loss%�:le-       �	�xreXc�A� *

lossS@;s�U       �	�wseXc�A� *

loss`�G:��v4       �	�teXc�A� *

loss5�<Qx�       �	\�teXc�A� *

lossyU!=��"�       �	�BueXc�A� *

loss8�;E{U�       �	��ueXc�A� *

loss`��;(>)       �	vveXc�A� *

loss�b�;=3Z       �	�weXc�A� *

loss��*<���       �	�0xeXc�A� *

loss���<���q       �	��xeXc�A� *

lossw�&;�v4       �	�fyeXc�A� *

loss��:��       �	P�yeXc�A� *

lossJ'K<J`��       �	��zeXc�A� *

loss�m< ,$�       �	Z*{eXc�A� *

loss�8v;��L^       �	C�{eXc�A� *

loss��<?Ȍ�       �	)_|eXc�A� *

loss�$;7�7.       �	*}eXc�A� *

losstrl<�W�        �	h�}eXc�A� *

loss�3<�Q�
       �	?V�eXc�A� *

loss�U�<�/�       �	��eXc�A� *

loss
h<�!A       �	���eXc�A� *

lossx�=eϴ�       �	��eXc�A� *

loss�u�<9�{m       �	CŕeXc�A� *

loss�C�<�=�       �	�Z�eXc�A� *

loss��<�H��       �	��eXc�A� *

loss�j<_��P       �	J~�eXc�A� *

loss,��<��       �	�:�eXc�A� *

loss�<=%q/�       �	r��eXc�A� *

loss�\k=?�[X       �	�eXc�A� *

lossC�K=<J~�       �	���eXc�A� *

losstZ+;�|�       �	CT�eXc�A� *

lossj�=�gX       �	�-�eXc�A� *

loss���<D�;       �	�K�eXc�A� *

loss��=܌r�       �	��eXc�A� *

loss���:�ڵ�       �	�eXc�A� *

loss��<��(�       �	fN�eXc�A� *

loss��=k�       �	g�eXc�A� *

loss\��<����       �	���eXc�A� *

loss���<-�       �	��eXc�A� *

loss�=:=�V1       �	��eXc�A� *

loss`ߦ:�ф�       �	�?�eXc�A� *

lossX�<WC��       �	�עeXc�A�!*

loss��5<��#       �	���eXc�A�!*

loss���=K�J(       �	iQ�eXc�A�!*

loss�-�=��<�       �	'��eXc�A�!*

loss��/<:+       �	R��eXc�A�!*

lossL�<���       �	y@�eXc�A�!*

loss��;U<�       �	`�eXc�A�!*

loss$v�<���       �	߈�eXc�A�!*

loss*@�;�R       �	D2�eXc�A�!*

loss��N<]Ln�       �	�̩eXc�A�!*

loss�D=d٭�       �	.�eXc�A�!*

lossN�=
b�       �	n��eXc�A�!*

lossN��<�Jz       �	�D�eXc�A�!*

loss�;;���       �	֬eXc�A�!*

loss��<�0�        �	<��eXc�A�!*

loss��<���       �	:�eXc�A�!*

loss�=���       �	��eXc�A�!*

loss;/�;$�<�       �	�K�eXc�A�!*

lossDl;���7       �	/��eXc�A�!*

loss�tF;߼f�       �	a4�eXc�A�!*

loss�R<�pۮ       �	�زeXc�A�!*

loss]ɖ;��m       �	!Y�eXc�A�!*

loss��?=Y��       �	��eXc�A�!*

lossg=��       �	��eXc�A�!*

loss{=ъ/       �	.=�eXc�A�!*

loss���<�Eq�       �	�@�eXc�A�!*

lossϺ49�/H       �	��eXc�A�!*

loss8c <i��       �	���eXc�A�!*

loss�<�ۅ*       �	mW�eXc�A�!*

loss�2<<Oӧ       �	b��eXc�A�!*

loss��>��w�       �	R��eXc�A�!*

lossE3�<(�_�       �	�4�eXc�A�!*

loss�W;g���       �	GȼeXc�A�!*

loss�I,=mo�       �	�^�eXc�A�!*

lossV�9&�;�       �	�Z�eXc�A�!*

lossPh<#�B       �	���eXc�A�!*

lossߌ�<9m�       �	?��eXc�A�!*

loss��</e7       �	s-�eXc�A�!*

loss���:0�c       �	���eXc�A�!*

loss�uE<f*�       �	�c�eXc�A�!*

loss7�<B�E�       �	���eXc�A�!*

loss��9=��B�       �	��eXc�A�!*

lossw&;�R]�       �	t)�eXc�A�!*

loss:$=��{1       �	/��eXc�A�!*

loss-;�W�       �	^�eXc�A�!*

loss�<.Ol�       �	���eXc�A�!*

losst�U=�w       �	���eXc�A�!*

loss/��<,�x       �	s�eXc�A�!*

loss���<��ت       �	d�eXc�A�!*

losshf,=���       �	f��eXc�A�!*

loss$��<y�n       �	�F�eXc�A�!*

loss2V�=Kd�F       �	h��eXc�A�!*

loss�̷<����       �	|��eXc�A�!*

loss4}<z�0       �	W?�eXc�A�!*

loss��;�eG       �	���eXc�A�!*

loss虊;JO�       �	�s�eXc�A�!*

loss3�M=���       �	U�eXc�A�!*

loss�lS=��       �	T��eXc�A�!*

loss�o�<�:��       �	�>�eXc�A�!*

loss|(<%�#       �	8��eXc�A�!*

loss�>h;�nܢ       �	�n�eXc�A�!*

lossq�u<�?x       �	�!�eXc�A�!*

loss{;�r�       �	���eXc�A�!*

loss�
g<@�v�       �	\��eXc�A�!*

lossQ�$<�       �	n3�eXc�A�!*

loss.e<^���       �	���eXc�A�!*

loss|P�< p�A       �	h�eXc�A�!*

loss���<�� }       �	��eXc�A�!*

loss3�n=�f�       �	Ѳ�eXc�A�!*

loss���<�8K*       �	�P�eXc�A�!*

loss�<�1       �	���eXc�A�!*

loss���=m�۠       �	A��eXc�A�!*

loss(|/<�       �	�C�eXc�A�!*

loss���<<��       �	�4�eXc�A�!*

loss1#X;��?�       �	#��eXc�A�!*

lossȿ*=�ڡ�       �	�!�eXc�A�!*

loss.Q< {�@       �	���eXc�A�!*

loss��: X<       �	�l�eXc�A�!*

loss�9=m�t�       �	�)�eXc�A�!*

loss��n<̹�       �	�=�eXc�A�!*

loss ��;]�E�       �	 ��eXc�A�!*

loss_=e<E��<       �	 �eXc�A�!*

loss��\=����       �	~�eXc�A�!*

losso6�:@��       �	��eXc�A�!*

loss
�":c^m       �	\�eXc�A�!*

loss�.�<6��F       �	���eXc�A�!*

lossb.�<b<�n       �	��eXc�A�!*

loss���='S��       �	�6�eXc�A�!*

lossE��<�M�d       �	���eXc�A�!*

loss�T;Ol��       �	o�eXc�A�!*

loss�{�:�x�       �	��eXc�A�!*

loss 	J<���       �	��eXc�A�!*

loss�:��گ       �	>Y�eXc�A�!*

loss@�K;�       �	���eXc�A�!*

loss3��<��t       �	��eXc�A�!*

loss!��<^=L�       �	�8�eXc�A�!*

loss�jg<�8�       �	��eXc�A�!*

loss`1><�dJ       �	@j�eXc�A�!*

loss6^;7�1�       �	K�eXc�A�!*

loss�{=��       �	x��eXc�A�!*

loss�l<J��5       �	�7�eXc�A�!*

loss7�<:��       �	g��eXc�A�!*

lossAMx<�З�       �	yy�eXc�A�!*

loss.X:<�� �       �	��eXc�A�!*

lossn�<\z�R       �	t��eXc�A�!*

lossC�=w˺4       �	�Q�eXc�A�!*

lossh=����       �	���eXc�A�!*

loss,��<6_�       �	���eXc�A�!*

loss��<��       �	�Z�eXc�A�!*

loss��<Y*uw       �	���eXc�A�!*

loss��A<��       �	�H�eXc�A�!*

losso�i<���       �	��eXc�A�!*

loss�E�; 3C       �	���eXc�A�!*

loss�,=       �	T8�eXc�A�!*

loss�;��l       �	���eXc�A�!*

loss1�=6�       �	Sv�eXc�A�!*

loss��;2�[       �	�eXc�A�!*

loss6^:;.�x�       �	ƿ�eXc�A�!*

lossŴ<}�vs       �	 ��eXc�A�!*

loss�@�<�[�       �	1A�eXc�A�!*

lossԆ�;F��+       �	��eXc�A�!*

loss\�=e���       �	8��eXc�A�!*

loss��;��L       �	���eXc�A�!*

loss`��;�:J�       �	I�eXc�A�!*

loss��:���       �	���eXc�A�!*

loss��J<,�?�       �	���eXc�A�!*

loss�=>=0F�       �	nO�eXc�A�!*

lossA�7<Loɥ       �	@��eXc�A�!*

loss��=���       �	��eXc�A�"*

loss�d=��1U       �	�>�eXc�A�"*

loss, =L��       �	���eXc�A�"*

loss}?<����       �	A��eXc�A�"*

losslX�:3o�       �	�"�eXc�A�"*

loss���;���       �	���eXc�A�"*

losso+=SVR�       �	�m fXc�A�"*

loss��_=6��>       �	hfXc�A�"*

loss��I<�(�&       �	àfXc�A�"*

loss_6�<��       �	.=fXc�A�"*

loss,�H:����       �	��fXc�A�"*

loss�Ύ=�y�       �	�ufXc�A�"*

loss2X<V��       �	xfXc�A�"*

loss�r�<�wo       �	C�fXc�A�"*

lossBE�<���       �	MfXc�A�"*

lossmk�::�2       �	��fXc�A�"*

losssG�<��_�       �	�|fXc�A�"*

loss�=ܭ�        �	�fXc�A�"*

loss�j�<�(�       �	��fXc�A�"*

loss3�<��b       �	jKfXc�A�"*

lossm:=�*'       �	��fXc�A�"*

loss�[�9D�(        �	�u	fXc�A�"*

loss�`�<����       �	�
fXc�A�"*

loss��<��Bh       �	��
fXc�A�"*

loss��;�
s�       �	�2fXc�A�"*

lossD=_=���%       �	��fXc�A�"*

loss!$�<�S�t       �	�wfXc�A�"*

lossᓍ<�F��       �	�fXc�A�"*

loss�_�;��ӊ       �	.�fXc�A�"*

loss�.�:�~Z�       �	�BfXc�A�"*

lossq�=>R��       �	��fXc�A�"*

loss�¦=K�˹       �	7qfXc�A�"*

loss�3�:���Z       �	nfXc�A�"*

loss7r�:��o�       �	��fXc�A�"*

loss�P;< �%�       �	;fXc�A�"*

lossb�;c1�       �	��fXc�A�"*

loss#�;^�_�       �	lfXc�A�"*

loss��=�*Bx       �	�fXc�A�"*

lossW0;<>wl�       �	N�fXc�A�"*

loss���=���       �	�5fXc�A�"*

loss#�*;��z�       �	?:fXc�A�"*

lossV�:Xs       �	9�fXc�A�"*

loss��;�Q��       �	jfXc�A�"*

lossH��:i�o       �	�ffXc�A�"*

loss�"=NЭ_       �	]�fXc�A�"*

lossRB�;$	q       �	�fXc�A�"*

loss��<< ʡ       �	�2fXc�A�"*

loss�$�<k۲       �	��fXc�A�"*

lossJ:�:�)S�       �	�ifXc�A�"*

loss���<E)G^       �	�fXc�A�"*

loss�'�;w�Vb       �	��fXc�A�"*

loss���:�p�       �	2=fXc�A�"*

lossȢD;�?��       �	��fXc�A�"*

loss��;�7       �	6tfXc�A�"*

loss��2<n��       �	fXc�A�"*

lossl�=;�@       �	�fXc�A�"*

loss�o=��"p       �	�@fXc�A�"*

loss���<�R��       �	�fXc�A�"*

loss��k;�NaB       �	�y fXc�A�"*

loss��;���y       �	�!fXc�A�"*

lossZs4<�b�       �	a�!fXc�A�"*

loss��;6}_�       �	NF"fXc�A�"*

loss�	�;.r�x       �	-�"fXc�A�"*

lossׅ!=��       �	�#fXc�A�"*

lossC= iK�       �	�)$fXc�A�"*

loss��<�y��       �	�$fXc�A�"*

lossmn�=v$�       �	�]%fXc�A�"*

loss}��=Eժ�       �	E&fXc�A�"*

loss䌰<c�+j       �	��&fXc�A�"*

lossF14=��Õ       �	Bw'fXc�A�"*

lossc�;���R       �	�(fXc�A�"*

loss�y<') �       �	�(fXc�A�"*

lossS��<o�t�       �	�K)fXc�A�"*

loss��m<;��(       �	��)fXc�A�"*

lossNA=�       �	��*fXc�A�"*

loss�1<�w��       �	�+fXc�A�"*

loss��C;Tl       �	�+fXc�A�"*

loss��;=��       �	MI,fXc�A�"*

lossG<<�[�       �	��,fXc�A�"*

loss�h<;�	]       �	�{-fXc�A�"*

loss74�:�y`       �	�.fXc�A�"*

loss�]S=7g/       �	�.fXc�A�"*

loss��:�D       �	�D/fXc�A�"*

loss�z<KB��       �	�/fXc�A�"*

lossXg'=���       �	er0fXc�A�"*

loss���;�s�?       �	�1fXc�A�"*

loss흝<\�2       �	�1fXc�A�"*

loss���:��/�       �	?;2fXc�A�"*

loss��!;��!       �	|�2fXc�A�"*

loss l�<��Up       �	'j3fXc�A�"*

loss0��<�f�       �	y4fXc�A�"*

loss��;}H�       �	Û4fXc�A�"*

lossD�<9DWq       �	�35fXc�A�"*

loss��<%{��       �	��5fXc�A�"*

lossר�;(I9�       �	; 7fXc�A�"*

loss���<}�3�       �	�7fXc�A�"*

lossV��;�k�       �	#,8fXc�A�"*

loss6�<��I       �	0F9fXc�A�"*

loss��D;o�ٯ       �	�6:fXc�A�"*

loss{�B= ���       �	�:fXc�A�"*

lossMނ<#Nx�       �	8�;fXc�A�"*

loss���<ݰ��       �	'�<fXc�A�"*

loss}<8��'       �	c�=fXc�A�"*

losss�;hR��       �	��>fXc�A�"*

loss�U�<�A�       �	A�?fXc�A�"*

loss��<-A/�       �	�m@fXc�A�"*

loss|�r<�       �	�AfXc�A�"*

lossF�<���K       �	.:BfXc�A�"*

loss/�<�J�       �	y�BfXc�A�"*

lossi4�;���       �	�bCfXc�A�"*

loss,_�;p�Z�       �	8�CfXc�A�"*

loss@$;6�[[       �	��DfXc�A�"*

lossj�4=W��       �	((EfXc�A�"*

loss@BH=e�oR       �	!?FfXc�A�"*

loss��:�d��       �	w�FfXc�A�"*

loss;{�<!=ɍ       �	"rGfXc�A�"*

loss]G�;sMEO       �	�HfXc�A�"*

lossgo�<�8lR       �	q�HfXc�A�"*

lossuc�<yQ:6       �	�>IfXc�A�"*

loss��h<d�~�       �	�IfXc�A�"*

loss��.<	�Yn       �	&mJfXc�A�"*

lossY=��y       �	GKfXc�A�"*

loss��;S)       �	؜KfXc�A�"*

loss��=�^�V       �	�5LfXc�A�"*

loss̯"=��R:       �	��LfXc�A�"*

loss΍\;F���       �	�cMfXc�A�"*

loss?��;�,�z       �	[NfXc�A�"*

loss��<v��n       �	.�NfXc�A�"*

loss�Y�<w,       �	�LOfXc�A�#*

loss}9�Z��       �	��OfXc�A�#*

loss�� <Pe�Q       �	A~PfXc�A�#*

loss��_;�,�       �	�QfXc�A�#*

loss}KR<�@d�       �	�QfXc�A�#*

loss�>E=��}       �	%\RfXc�A�#*

loss�e�<�e�       �	KSfXc�A�#*

loss�=�UE       �	w�SfXc�A�#*

lossO
x=Ա�d       �	a5TfXc�A�#*

lossn��;�B�       �	K�TfXc�A�#*

lossV=<6P�       �	CrUfXc�A�#*

loss�F?:���       �	�VfXc�A�#*

loss�J<�Z�       �	4�VfXc�A�#*

loss��g;�b�       �	75WfXc�A�#*

loss_�:����       �	h�WfXc�A�#*

lossd�6=���       �	�eXfXc�A�#*

loss��,=
�       �	��XfXc�A�#*

loss�	;�>�;       �	�YfXc�A�#*

loss	(<���'       �	��ZfXc�A�#*

loss��=9�T       �	Ϥ[fXc�A�#*

lossr&�<��]       �	>\fXc�A�#*

loss=�P<B�+�       �	�]fXc�A�#*

loss��.=1�9*       �	JA^fXc�A�#*

lossN�2;7�cd       �	��^fXc�A�#*

loss�/a=%�       �	��_fXc�A�#*

loss�h�<J ��       �	�afXc�A�#*

lossN"�<횱Z       �	�afXc�A�#*

loss��:�K��       �	�bfXc�A�#*

lossn�=R'xG       �	e6cfXc�A�#*

lossj6<�W�-       �	��cfXc�A�#*

lossC�9��p�       �	.�dfXc�A�#*

loss�<&1s       �	�CefXc�A�#*

loss�%�;z�z       �	�NffXc�A�#*

lossFT=��u�       �	�xgfXc�A�#*

loss��?;�S^�       �	0�hfXc�A�#*

loss�)';-�zi       �	}<ifXc�A�#*

lossoG�<��*       �	;�ifXc�A�#*

loss��P;�rH       �	�kfXc�A�#*

lossx��;
��       �	U�kfXc�A�#*

loss��=x�W       �	��lfXc�A�#*

loss�y5;�C�       �	��mfXc�A�#*

loss8X)<�#��       �	E�nfXc�A�#*

loss|1=��x       �	��ofXc�A�#*

loss2F=0\�       �	8�pfXc�A�#*

loss6��<j�K       �	܃qfXc�A�#*

loss
�q=,~��       �	c&rfXc�A�#*

loss�;	��y       �	�$sfXc�A�#*

loss�X�:X~�       �	FatfXc�A�#*

loss�(<#iR@       �	�]ufXc�A�#*

loss�Y�;�)�w       �	vfXc�A�#*

loss���<�F=I       �	-`wfXc�A�#*

loss�Q�<��
       �	ڌxfXc�A�#*

lossM�Q<sR��       �	-[yfXc�A�#*

loss�j�;��en       �	�FzfXc�A�#*

loss<<-si�       �	�{fXc�A�#*

lossqō<���       �	߿{fXc�A�#*

loss���;��;�       �	�s|fXc�A�#*

lossC�1<�k\<       �	w}fXc�A�#*

lossaƒ<v��       �	��}fXc�A�#*

loss���<C�z       �	X�~fXc�A�#*

loss.�;�޽       �	UfXc�A�#*

loss�5�;8�7|       �	��fXc�A�#*

loss���<Q�DG       �	���fXc�A�#*

loss��>=_K-       �	�=�fXc�A�#*

loss���:�oc?       �	&4�fXc�A�#*

loss��=�ի�       �	�(�fXc�A�#*

lossy��<	p�d       �	�̓fXc�A�#*

lossT_=<A���       �	�fXc�A�#*

loss�=��.       �	�e�fXc�A�#*

lossf��<g�0D       �	x	�fXc�A�#*

loss�B8<��W       �	;��fXc�A�#*

loss�q,;�I��       �	�Q�fXc�A�#*

loss�"�<���       �	���fXc�A�#*

loss���<�B2m       �	>��fXc�A�#*

lossT�.;B1\�       �	o/�fXc�A�#*

losscP�=�7��       �	R҉fXc�A�#*

lossa�;��Y6       �	-��fXc�A�#*

lossV��<K5��       �	�ɋfXc�A�#*

loss!;�;�h�e       �	\�fXc�A�#*

loss�<o�z�       �	,��fXc�A�#*

loss�56<x���       �	QO�fXc�A�#*

loss�O�;�Ď�       �	���fXc�A�#*

loss�ȑ;	G       �	��fXc�A�#*

loss.�=��[       �	�+�fXc�A�#*

lossa�<0�S�       �	���fXc�A�#*

lossҷ�="��       �	n3�fXc�A�#*

lossd^b=��|       �	�˒fXc�A�#*

losscG�;�O�8       �	�c�fXc�A�#*

lossrHC<�^�       �	���fXc�A�#*

loss�ط:�E�       �	��fXc�A�#*

loss��<<�W�b       �	�5�fXc�A�#*

loss��O<��9D       �	�ΕfXc�A�#*

loss._�;� �       �	�f�fXc�A�#*

lossA7;
�x:       �	��fXc�A�#*

loss�I=��Э       �	��fXc�A�#*

loss���<����       �	7�fXc�A�#*

loss��<�U�       �	9^�fXc�A�#*

loss%�2<�,P       �	
0�fXc�A�#*

loss\�<$��       �	F#�fXc�A�#*

lossʋ(<'̢s       �	�ӛfXc�A�#*

loss�nV<����       �	�ќfXc�A�#*

lossep4;^ �       �	�͝fXc�A�#*

lossS*�=7�       �	>�fXc�A�#*

loss���<�#       �	��fXc�A�#*

loss	]0;>��       �	jN�fXc�A�#*

lossX80:t=d       �	>�fXc�A�#*

loss}3>� �       �	t��fXc�A�#*

loss�<��1�       �	�i�fXc�A�#*

loss���<�c�       �	�%�fXc�A�#*

lossL�B;�H�       �	Jb�fXc�A�#*

loss&��; ݛ�       �	Kw�fXc�A�#*

loss���<@���       �	0�fXc�A�#*

loss���<��В       �	�٧fXc�A�#*

loss)k�;bP�       �	q��fXc�A�#*

losss�;�Fx       �	��fXc�A�#*

loss���<�d��       �	t��fXc�A�#*

lossM�N<l�H�       �	���fXc�A�#*

loss���:^''x       �	�ӬfXc�A�#*

lossr�<a:δ       �	��fXc�A�#*

lossã	;)��       �	d��fXc�A�#*

lossSY<���f       �	��fXc�A�#*

loss{��;�s�=       �	۾�fXc�A�#*

lossoY�;�	       �	��fXc�A�#*

loss��<��~�       �	���fXc�A�#*

loss���<�E�B       �	�M�fXc�A�#*

lossH�9=0��       �	���fXc�A�#*

lossz��<6̽�       �	�z�fXc�A�#*

loss��<%���       �	E��fXc�A�#*

loss&��<-�       �	�ķfXc�A�$*

loss�ɐ<�ǟK       �	Sv�fXc�A�$*

lossG,=F@fD       �	�j�fXc�A�$*

loss�~�<D�z[       �	2 �fXc�A�$*

loss�y�=sB�       �	H��fXc�A�$*

loss)`�<���       �	��fXc�A�$*

loss-��<<��       �	��fXc�A�$*

loss�Ր;�F�V       �	ؽfXc�A�$*

loss_g =��o�       �	Ow�fXc�A�$*

lossCE";o���       �	�fXc�A�$*

loss���;+qw�       �	ı�fXc�A�$*

loss�a�<�0�       �	,J�fXc�A�$*

lossZ��:����       �	r��fXc�A�$*

loss2��;`�       �	�~�fXc�A�$*

loss�'�:fA{       �	j�fXc�A�$*

loss���:[��        �	X��fXc�A�$*

loss��=��p@       �	�G�fXc�A�$*

loss H�=7ZD       �	��fXc�A�$*

loss[��<����       �	!x�fXc�A�$*

loss�C�<"j�       �	�fXc�A�$*

loss�wt<Y��w       �	K��fXc�A�$*

loss��O=E�f8       �	�E�fXc�A�$*

loss��:�       �	.��fXc�A�$*

lossH��<��tb       �	���fXc�A�$*

loss=Q�<u�L       �	�$�fXc�A�$*

loss/�A=���       �	���fXc�A�$*

loss�V=�Y       �	�a�fXc�A�$*

loss{=��       �	3��fXc�A�$*

loss��<v��[       �	��fXc�A�$*

loss�S4<���}       �	�)�fXc�A�$*

loss�Y�</C"�       �	���fXc�A�$*

loss�5�<�n�5       �	��fXc�A�$*

lossԜ<���H       �	���fXc�A�$*

loss�5�;n,�_       �	���fXc�A�$*

loss�Cm=J���       �	33�fXc�A�$*

loss�<R�`4       �	���fXc�A�$*

loss�^�=ՓD       �	m�fXc�A�$*

loss.�;�C�       �	��fXc�A�$*

loss�g]<��       �	��fXc�A�$*

loss�f�<7��       �	>�fXc�A�$*

loss�E<�s��       �	���fXc�A�$*

loss6��;�f�H       �	>v�fXc�A�$*

lossEr<�$7�       �	��fXc�A�$*

loss�+=v��       �	��fXc�A�$*

losss��;�ߗr       �	�e�fXc�A�$*

lossڐy;'�=       �	L��fXc�A�$*

loss�f<#rh       �	vT�fXc�A�$*

lossI�=�s�z       �	��fXc�A�$*

lossz��=��T       �	`��fXc�A�$*

lossT@V;VVG       �	�I�fXc�A�$*

lossW��;�:��       �	~�fXc�A�$*

loss���;�a��       �	��fXc�A�$*

loss�c7<��"�       �	vQ�fXc�A�$*

loss�<���V       �	�.�fXc�A�$*

lossx��<Լ��       �	���fXc�A�$*

loss��q;����       �	�f�fXc�A�$*

loss�42=��ǯ       �	��fXc�A�$*

loss�x<���       �	���fXc�A�$*

loss]�R<��       �	�C�fXc�A�$*

loss�=eSˈ       �	m��fXc�A�$*

loss)�p;��*8       �	��fXc�A�$*

losshO;>��       �	��fXc�A�$*

loss�!�<��{�       �	���fXc�A�$*

lossL�=!���       �	�|�fXc�A�$*

loss�}<o�       �	�fXc�A�$*

loss�;}T�       �	���fXc�A�$*

lossZ&G=�]Ik       �	sG�fXc�A�$*

lossoQ<��]�       �	���fXc�A�$*

loss��;s�&�       �	�|�fXc�A�$*

lossd(�;�9Vq       �	<�fXc�A�$*

loss�K;X��       �	���fXc�A�$*

loss�0< �3�       �	�T�fXc�A�$*

loss�Қ<��T�       �	���fXc�A�$*

loss��=(��       �	4��fXc�A�$*

loss�I<���&       �	J��fXc�A�$*

loss�e<d�2E       �	��fXc�A�$*

loss�T�<�~0       �	��fXc�A�$*

loss��M;�P��       �	�k�fXc�A�$*

loss�'(=��[C       �	��fXc�A�$*

loss6��;�j��       �	i��fXc�A�$*

loss�8$==�$�       �	JF�fXc�A�$*

loss�TF<�C}Q       �	3��fXc�A�$*

loss*�W;�2�y       �	Bz�fXc�A�$*

lossz��<�O(}       �	X�fXc�A�$*

lossg5�<AxE       �	���fXc�A�$*

loss�:I��       �	EF�fXc�A�$*

loss��<t:       �	���fXc�A�$*

loss\�:���i       �	Sy�fXc�A�$*

loss���;�<��       �	0�fXc�A�$*

loss*f�<�2��       �	���fXc�A�$*

loss�YM<q�K�       �	�N�fXc�A�$*

loss|<����       �	z��fXc�A�$*

lossq�<��؄       �	~5�fXc�A�$*

lossqS�<\��       �	��fXc�A�$*

lossRF:       �	B?�fXc�A�$*

loss֫<d�       �	���fXc�A�$*

loss%��;F�       �	���fXc�A�$*

loss�#�<;E^a       �	�"�fXc�A�$*

loss��;��P       �	/��fXc�A�$*

loss�`�<Q..�       �	�Y�fXc�A�$*

loss:��;��)R       �	���fXc�A�$*

lossrӧ<�^       �	d��fXc�A�$*

loss��=��q       �	a6 gXc�A�$*

loss!<�;j�b�       �	�� gXc�A�$*

loss`D�:����       �	�jgXc�A�$*

loss!\�;�*�^       �	�gXc�A�$*

lossS�<� ��       �	�fgXc�A�$*

loss�8;r"�       �	��gXc�A�$*

lossHit=`��l       �	ēgXc�A�$*

lossҝz=�Qy�       �	 (gXc�A�$*

lossl;<Gs@�       �	�gXc�A�$*

lossZ�V<ì��       �	YgXc�A�$*

lossS�":��~n       �	E�gXc�A�$*

lossTP:����       �	'�gXc�A�$*

loss�;��U       �	O"gXc�A�$*

loss.��:�{�w       �	��gXc�A�$*

loss���;l�K8       �	�W	gXc�A�$*

loss� �<�.��       �	=�	gXc�A�$*

loss���:���       �	G�
gXc�A�$*

loss�<P�"       �	BAgXc�A�$*

lossI�D;�1u       �	��gXc�A�$*

lossӈ�={�       �	�xgXc�A�$*

loss��<��A�       �	wgXc�A�$*

loss�w�;5g�6       �	C�gXc�A�$*

loss�J�<�a�u       �	eVgXc�A�$*

loss��9pcj�       �	��gXc�A�$*

lossnYi9Ox��       �	͓gXc�A�$*

loss��<ؠ��       �	b.gXc�A�$*

loss�O�:HZ�i       �	�gXc�A�%*

loss���;xmm       �	�egXc�A�%*

loss�]
<�U�|       �	�gXc�A�%*

lossl�99)�#}       �	��gXc�A�%*

lossL�;]���       �	�=gXc�A�%*

loss3Uq;|x0�       �	|�gXc�A�%*

loss@�:g��       �	vgXc�A�%*

loss�o�9���       �	�gXc�A�%*

loss�Av:��G�       �	�gXc�A�%*

loss�*h=~�r       �	kDgXc�A�%*

losss�O<���       �	�gXc�A�%*

lossX�;��{u       �	BygXc�A�%*

lossq�M;1 ��       �	�gXc�A�%*

loss�D=���       �	ҦgXc�A�%*

loss�h-;���E       �	lCgXc�A�%*

loss��W=��F�       �	��gXc�A�%*

losszf_<���       �	:gXc�A�%*

lossO��;�-ظ       �	�gXc�A�%*

loss��h;�t��       �	l�gXc�A�%*

loss��!=���       �	��gXc�A�%*

loss�ď=���)       �	�gXc�A�%*

lossS�=Bl�T       �	t# gXc�A�%*

loss��x;E=E       �	� !gXc�A�%*

lossd�,<s���       �	�!gXc�A�%*

lossX=��       �	Eb"gXc�A�%*

lossv&�<c@��       �	d=#gXc�A�%*

loss�Ǚ;Q@��       �	(`$gXc�A�%*

loss�l�<�e�^       �	��$gXc�A�%*

loss���;��ɤ       �	6�%gXc�A�%*

loss�q�< �!o       �	�#&gXc�A�%*

loss*~<�1�1       �	��&gXc�A�%*

loss	X�;By�P       �	Xq'gXc�A�%*

lossヶ<�2       �	�(gXc�A�%*

lossň9���J       �	�(gXc�A�%*

loss�j�:�s�~       �	$C)gXc�A�%*

loss-�<*k�       �	H�)gXc�A�%*

lossp�=�A�M       �	d�*gXc�A�%*

loss�O�:��r�       �	�-+gXc�A�%*

loss,��<��p�       �	��+gXc�A�%*

lossL.o:�x       �	Rb,gXc�A�%*

loss.ܖ<�&B�       �	 -gXc�A�%*

loss�x<����       �	��-gXc�A�%*

loss^#=B#�       �	cE.gXc�A�%*

loss��!=��<b       �	B�.gXc�A�%*

loss��:9%       �	3�/gXc�A�%*

loss���;���       �	�#0gXc�A�%*

lossM'�:8�$�       �	��0gXc�A�%*

loss j:��       �	�c1gXc�A�%*

loss\�<�2��       �	a�1gXc�A�%*

lossF�<�j       �	V�2gXc�A�%*

lossAŀ:���e       �	<03gXc�A�%*

loss
 �<>��w       �	�3gXc�A�%*

lossϓx<�&��       �	\4gXc�A�%*

loss_؂<�mv�       �	o�4gXc�A�%*

loss]�9ݿN>       �	e�5gXc�A�%*

lossñ[<��Î       �	�"6gXc�A�%*

loss�y;����       �	{�6gXc�A�%*

loss�ߐ=�5��       �	rR7gXc�A�%*

loss-�*=Gz�       �	�7gXc�A�%*

loss�/<�+a�       �	΍8gXc�A�%*

loss��i=��g�       �	 9gXc�A�%*

loss1 ;Z�Q�       �	�:gXc�A�%*

loss
��<�'�9       �	 �:gXc�A�%*

loss�@�<78��       �	�Y;gXc�A�%*

lossыM<6.%�       �	x�;gXc�A�%*

loss�;<�۫2       �	�wTgXc�A�%*

loss�m-<���       �	<UgXc�A�%*

loss(�=WNB       �	��UgXc�A�%*

loss��=�p       �	�KVgXc�A�%*

loss�X�<��PG       �	�3YgXc�A�%*

loss�^);��I�       �	��YgXc�A�%*

loss�p= f�6       �	�[gXc�A�%*

loss���<DG�U       �	�\gXc�A�%*

loss�=�<�BϤ       �	k�\gXc�A�%*

loss�h9<���       �	X8]gXc�A�%*

loss��j;wIx       �	��]gXc�A�%*

loss t�<���       �	l]^gXc�A�%*

loss��<��       �	p>_gXc�A�%*

loss��=�`I�       �	��_gXc�A�%*

loss_\ <a�@       �	~t`gXc�A�%*

lossXp<���X       �	WagXc�A�%*

loss�KC<w���       �	.�agXc�A�%*

loss�F=��f       �	oIbgXc�A�%*

lossnv�<P�       �	��bgXc�A�%*

loss�t�<�#�n       �	�~cgXc�A�%*

lossT� ;H�)~       �	dgXc�A�%*

loss(��=#���       �	N�dgXc�A�%*

losst�
: �E�       �	�HegXc�A�%*

losstnD=����       �	��egXc�A�%*

loss8�B<|(a�       �	�{fgXc�A�%*

loss�D:Qܴ�       �	��ggXc�A�%*

loss&��<-)��       �	��hgXc�A�%*

loss4J<;��       �	��igXc�A�%*

loss��=�,=w       �	�^jgXc�A�%*

loss֫�9�;�       �	*�jgXc�A�%*

loss���;�S6       �	��kgXc�A�%*

loss8�;���L       �	n5lgXc�A�%*

loss$[�<c��       �	��lgXc�A�%*

loss �=���       �	[ymgXc�A�%*

loss:��<��;�       �	�ngXc�A�%*

loss/��=��)�       �	��ngXc�A�%*

loss���<���       �	�QogXc�A�%*

loss��<��G       �	��ogXc�A�%*

loss�=�=A�(       �	��pgXc�A�%*

losszQZ= �a�       �	�rgXc�A�%*

loss�;��94       �	��rgXc�A�%*

lossM4�;��R       �	�CsgXc�A�%*

lossx�=;1c9K       �	+�sgXc�A�%*

loss�6<��)W       �	ltgXc�A�%*

loss|�O=b.	       �	� ugXc�A�%*

loss�k�<=�z�       �	F�ugXc�A�%*

loss�݈;�Rt7       �	)vgXc�A�%*

loss2ɧ<B���       �	k�vgXc�A�%*

lossފ;���e       �	RIwgXc�A�%*

lossI�9���x       �	��wgXc�A�%*

lossC�;�F?|       �	b�xgXc�A�%*

loss(��;�!t�       �	�ygXc�A�%*

loss=
=f��       �	��ygXc�A�%*

loss�n�=H�т       �	WBzgXc�A�%*

loss�X�<��       �	��zgXc�A�%*

loss7ۺ;�;�       �	�w{gXc�A�%*

loss�	;ƽ`       �	|gXc�A�%*

lossժ9��^�       �	��|gXc�A�%*

lossAL�;��       �		p}gXc�A�%*

lossd?g<^+<�       �	�~gXc�A�%*

loss
(E<���n       �	t�~gXc�A�%*

lossh�c;�&       �	�UgXc�A�%*

loss!ba;�pS       �	0�gXc�A�%*

loss��<@       �	\��gXc�A�&*

loss�h6;,]�       �	�+�gXc�A�&*

loss\;�:�׿w       �	�ÁgXc�A�&*

loss#f�;�F<X       �	cb�gXc�A�&*

lossn �;|�o       �	;�gXc�A�&*

loss\~=��7O       �	���gXc�A�&*

loss�#G;��j       �	�N�gXc�A�&*

loss
�E<�R�       �	��gXc�A�&*

loss�3=$u�        �	���gXc�A�&*

loss���<o�	�       �	��gXc�A�&*

loss�V�;m�       �	f��gXc�A�&*

loss��=0�^�       �	�W�gXc�A�&*

loss�Q�;aT�       �	��gXc�A�&*

lossO<TA��       �	3��gXc�A�&*

loss��;{ߙ       �	F%�gXc�A�&*

lossc��;�E	       �	˿�gXc�A�&*

loss�U�<Ϋ8       �	vS�gXc�A�&*

loss!<�^�       �	��gXc�A�&*

loss�P<�*ӷ       �	n��gXc�A�&*

lossȇ�<��t       �	�;�gXc�A�&*

lossGz9���       �	 ՍgXc�A�&*

loss8�<z �       �	�q�gXc�A�&*

loss3��<��]       �	��gXc�A�&*

lossC;�<f�k       �	3��gXc�A�&*

loss3&;ӴY       �	�@�gXc�A�&*

loss$ɝ;�$e       �	fؐgXc�A�&*

lossT��;N���       �	p�gXc�A�&*

lossf<��?�       �	��gXc�A�&*

loss�%�<����       �	'��gXc�A�&*

lossl�U;�)B       �	�6�gXc�A�&*

loss��;iȢ:       �	�͓gXc�A�&*

lossm,�=�RD       �	�h�gXc�A�&*

loss�Ye<is       �	��gXc�A�&*

loss��(<�x��       �	N��gXc�A�&*

loss҈�<d��       �	v4�gXc�A�&*

loss�V�<�8��       �	�ʖgXc�A�&*

lossi�<:�:�3       �	�a�gXc�A�&*

loss��Q;���       �	"�gXc�A�&*

loss�1�;�n�       �	�ĘgXc�A�&*

loss��<�8x�       �	���gXc�A�&*

lossh��;��#�       �	�J�gXc�A�&*

loss�=x�c�       �	#��gXc�A�&*

loss1Z<-,X       �	~�gXc�A�&*

loss��g;P��c       �	���gXc�A�&*

lossVJ�:�oE�       �	$H�gXc�A�&*

loss�)<���
       �	��gXc�A�&*

loss\�;����       �	ׄ�gXc�A�&*

loss� <���#       �	?�gXc�A�&*

loss���;C���       �	[��gXc�A�&*

lossZ�={@�       �	�F�gXc�A�&*

loss��A:�+F�       �	��gXc�A�&*

lossQ�<��+�       �	��gXc�A�&*

loss��;�y��       �	L��gXc�A�&*

loss��G;t_�       �	+�gXc�A�&*

loss	<F=���Y       �	�ģgXc�A�&*

lossH|A;�V�       �	g�gXc�A�&*

lossZ��;U�C       �	`�gXc�A�&*

loss8�=;��e�       �	��gXc�A�&*

loss�4�;CX��       �	㪦gXc�A�&*

loss��<-*       �	l?�gXc�A�&*

loss���;�◚       �	:�gXc�A�&*

loss��h<�W/i       �	�{�gXc�A�&*

loss%kb=P]��       �	4�gXc�A�&*

loss�>n=$��       �	e��gXc�A�&*

loss�1R<��D)       �	D�gXc�A�&*

lossl=!6Eh       �	YݪgXc�A�&*

loss,�<��       �	:t�gXc�A�&*

lossҨa;`ׄ       �	��gXc�A�&*

loss2�;MW�       �	���gXc�A�&*

lossF�=�&��       �	\�gXc�A�&*

loss��F=�u�       �	��gXc�A�&*

loss� =Ɍ�;       �	gXc�A�&*

loss�a�<���~       �	�*�gXc�A�&*

loss'B�;F���       �	�̯gXc�A�&*

loss���:hEh       �	�a�gXc�A�&*

lossł<�A�       �	V��gXc�A�&*

loss-��<��i       �	��gXc�A�&*

loss+n<��k�       �	�gXc�A�&*

loss�<Q<F��       �	p$�gXc�A�&*

loss��+<r ��       �	ཱིgXc�A�&*

losso�;�pI�       �	�Y�gXc�A�&*

loss.Q=SȐ�       �	w��gXc�A�&*

losseN<���d       �	됵gXc�A�&*

loss�X�;%s�y       �	8�gXc�A�&*

loss(�@<��:L       �	���gXc�A�&*

loss�v�:a��       �	d��gXc�A�&*

loss?5�<F#�       �	
/�gXc�A�&*

loss�jc<�(A�       �	͸gXc�A�&*

loss��=h�x{       �	Zh�gXc�A�&*

loss���;]K�!       �	��gXc�A�&*

loss�mA=3��?       �	���gXc�A�&*

loss�3C<%`�.       �	S��gXc�A�&*

loss��:0��i       �	p\�gXc�A�&*

loss;<|:��H       �	�g�gXc�A�&*

loss�h�<U�N       �	A�gXc�A�&*

loss�>=|Fy�       �	���gXc�A�&*

loss���;���       �	;U�gXc�A�&*

lossĪ<�Ф,       �	`r�gXc�A�&*

loss?�~;~ڙ       �	+/�gXc�A�&*

loss1a==ݐs�       �	$��gXc�A�&*

loss�`�:�:�       �	�q�gXc�A�&*

loss:�<R��       �	4�gXc�A�&*

loss��F<ю؃       �	$��gXc�A�&*

loss�Z^;��       �	b�gXc�A�&*

loss]�="�]o       �	��gXc�A�&*

loss�1t:˴�       �	���gXc�A�&*

loss�}:<�;?8       �	;S�gXc�A�&*

loss��=e $O       �	���gXc�A�&*

loss1�;�L$y       �	u��gXc�A�&*

lossq�n<Q��L       �	�3�gXc�A�&*

loss&��=KH�i       �	���gXc�A�&*

loss|:/ʤ{       �	/l�gXc�A�&*

loss:��<���$       �	� �gXc�A�&*

loss��=�
~7       �	��gXc�A�&*

loss颫= ��       �	w+�gXc�A�&*

lossf�a<<��9       �	���gXc�A�&*

loss�T	<��       �	yt�gXc�A�&*

loss��<sk�       �	��gXc�A�&*

loss��><{v��       �	j��gXc�A�&*

loss-�<!ۯ       �	�\�gXc�A�&*

loss}�r;&�       �	 �gXc�A�&*

loss��<���       �	��gXc�A�&*

loss��;l3�       �	�G�gXc�A�&*

loss��<�~�       �	`��gXc�A�&*

lossV�'<��k�       �	+��gXc�A�&*

loss-m=n:       �	"�gXc�A�&*

lossף�<W�7=       �	��gXc�A�&*

lossMj�<��<R       �	�G�gXc�A�&*

lossN��<�A4�       �	e��gXc�A�'*

loss7��;a%�       �	���gXc�A�'*

loss$�<i�3        �	��gXc�A�'*

lossG�:���       �	$��gXc�A�'*

loss3yO<!'�       �	���gXc�A�'*

loss�N�=��/&       �	�(�gXc�A�'*

loss!�=�&�+       �	{��gXc�A�'*

loss��;���       �	Mg�gXc�A�'*

lossx�A<7�~3       �	 ��gXc�A�'*

loss:<U2H       �	}��gXc�A�'*

loss�j�<��{�       �	-�gXc�A�'*

loss��;�{͒       �	���gXc�A�'*

loss��<UvY�       �	���gXc�A�'*

loss��<�o�O       �	a��gXc�A�'*

loss��;�xA       �	���gXc�A�'*

lossë\=�g/       �		o�gXc�A�'*

loss��<_Q�       �	d�gXc�A�'*

loss�<Ix�m       �	G��gXc�A�'*

loss�M�<��M       �	YR�gXc�A�'*

loss_�U<�B�       �	l��gXc�A�'*

loss�?�;rrO�       �	��gXc�A�'*

loss��:���       �	d"�gXc�A�'*

loss�<�       �	Ǹ�gXc�A�'*

lossDږ='0�       �	�M�gXc�A�'*

loss��	=�>46       �	v��gXc�A�'*

loss��"=��D/       �	�z�gXc�A�'*

loss�O >���       �	`�gXc�A�'*

loss�g%=��zR       �	и�gXc�A�'*

loss�b!=Zu�       �	�W�gXc�A�'*

loss�<�<a���       �	N��gXc�A�'*

loss�l-<��-       �	Y2�gXc�A�'*

lossf�<dU�y       �	��gXc�A�'*

loss���<���       �	�w�gXc�A�'*

loss\W�;'0       �	
�gXc�A�'*

lossc��<�I�       �	Ͻ�gXc�A�'*

loss�;�L�D       �	XW�gXc�A�'*

lossq�<=�$�       �	p
�gXc�A�'*

lossϩ<���       �	���gXc�A�'*

loss�<��l�       �	?s�gXc�A�'*

loss��<�So       �	o�gXc�A�'*

lossR��:)���       �	��gXc�A�'*

loss���<+V�^       �	�b�gXc�A�'*

loss%y�;w�T`       �	�	�gXc�A�'*

loss���<ʘ�J       �	<��gXc�A�'*

loss�@�<���       �	9C�gXc�A�'*

lossᗢ;��C�       �	��gXc�A�'*

loss�V<���       �	E��gXc�A�'*

lossc�-;�	-M       �	�!�gXc�A�'*

loss�{�=�C�O       �	��gXc�A�'*

loss�9<t���       �	^e�gXc�A�'*

loss��6<�:�       �	Y�gXc�A�'*

lossX<�9�-       �	A��gXc�A�'*

loss:T<&Jl�       �	V��gXc�A�'*

lossl�=�i       �	���gXc�A�'*

loss�Za<ԝ��       �	 '�gXc�A�'*

lossWH�<� d�       �	��gXc�A�'*

lossCB<���l       �	�q�gXc�A�'*

lossi�B<ły�       �	�#�gXc�A�'*

loss)&�;/�ߵ       �	��gXc�A�'*

lossS��<�WE       �	P��gXc�A�'*

loss.j!<$*�       �	�C�gXc�A�'*

loss&��<�R       �	t��gXc�A�'*

lossU�<ݩ       �	܂�gXc�A�'*

loss�&:t�^o       �	�v�gXc�A�'*

loss�6g<H3=l       �	��gXc�A�'*

loss���;%8h       �	���gXc�A�'*

loss�Ň<B��       �	�a hXc�A�'*

loss3=�c�       �	�� hXc�A�'*

losst,=f���       �	�hXc�A�'*

loss�s�<BF��       �	@1hXc�A�'*

loss��;N<@       �	�hXc�A�'*

lossV�<#��       �	pahXc�A�'*

loss8�A=��mv       �	��hXc�A�'*

loss��\<_��       �	ǛhXc�A�'*

lossډ�:��=�       �	r5hXc�A�'*

loss�2<ɬ;�       �	)�hXc�A�'*

loss���<[�       �	�ahXc�A�'*

loss)�.=�	��       �	��hXc�A�'*

loss���;l��       �	��hXc�A�'*

loss�V�;���       �	77hXc�A�'*

lossJ�<ǃ�       �	��hXc�A�'*

loss�V�<���i       �	�g	hXc�A�'*

loss���;L4q�       �	��	hXc�A�'*

loss,=nΕ�       �	ٔ
hXc�A�'*

loss�k�<LͱZ       �	�,hXc�A�'*

lossnt�;)/�#       �	��hXc�A�'*

loss�\=-w�       �	��hXc�A�'*

loss�IL<O�z       �	�<hXc�A�'*

loss��<�˂�       �	��hXc�A�'*

loss��O;��b�       �	RehXc�A�'*

lossx<�eW=       �	!hXc�A�'*

loss`�X:���       �	r�hXc�A�'*

losse�&<f#-�       �	�DhXc�A�'*

lossd� =޹        �	*;hXc�A�'*

loss�;b�+�       �	��hXc�A�'*

loss��<���       �	4�hXc�A�'*

loss�d=Vm       �	YhXc�A�'*

loss���<���       �	>�hXc�A�'*

loss=��       �	LRhXc�A�'*

loss�x{:ಝ�       �	��hXc�A�'*

loss��;+���       �	��hXc�A�'*

losssE�;�g�$       �	�'hXc�A�'*

loss��c9��dA       �	�hXc�A�'*

loss�-<0h4�       �	�UhXc�A�'*

loss�,�;3�(�       �	1�hXc�A�'*

loss_��:��R       �	s�hXc�A�'*

loss�Ma=�@0       �	.hXc�A�'*

loss؈V=��"       �	��hXc�A�'*

loss�9�<�$�       �	�]hXc�A�'*

loss�Y;=���       �	.qhXc�A�'*

loss�G�<��g;       �	�AhXc�A�'*

loss�/:����       �	��hXc�A�'*

lossMw�<s�J[       �	�hXc�A�'*

lossR8<��>�       �	OwhXc�A�'*

loss$��<.U�       �	
f hXc�A�'*

losslrF:V�       �	�� hXc�A�'*

loss8?<��yc       �	6�!hXc�A�'*

loss�~v<��}       �	B$"hXc�A�'*

loss��:�̤       �	��"hXc�A�'*

loss���;��I       �	e#hXc�A�'*

loss��=��       �	�#hXc�A�'*

loss��=4�       �	Ԛ$hXc�A�'*

loss߲�:r���       �	�8%hXc�A�'*

loss �q;C�
q       �	��%hXc�A�'*

loss�b=f�       �	�c&hXc�A�'*

lossv@<���       �	%'hXc�A�'*

loss�f�<�&��       �	f�'hXc�A�'*

loss�'�<���       �	�b(hXc�A�'*

lossz��<��       �	�(hXc�A�(*

loss���<��V]       �	>�)hXc�A�(*

loss��>�gY�       �	�4*hXc�A�(*

loss��='	 �       �	��*hXc�A�(*

loss��A;����       �	i+hXc�A�(*

loss`��;�;(�       �	b,hXc�A�(*

loss/�0<�_�       �	��,hXc�A�(*

loss �$;7�4p       �	_C-hXc�A�(*

loss�
O<*�̻       �	��-hXc�A�(*

loss�B=)�8       �	p|.hXc�A�(*

loss7�=Ԋa�       �	"/hXc�A�(*

loss1��;�O?9       �	��/hXc�A�(*

loss��<��}t       �	oJ0hXc�A�(*

lossr�9��i       �	�0hXc�A�(*

lossq2�;�>�       �	NA2hXc�A�(*

lossS�=��`L       �	f�2hXc�A�(*

loss�6Z;��       �	l{3hXc�A�(*

loss�|�<��P       �	<4hXc�A�(*

loss*�<�13�       �	�Y5hXc�A�(*

loss��=�3��       �	D�5hXc�A�(*

loss#��;�Bہ       �	!�6hXc�A�(*

lossS�;g�p       �	�-7hXc�A�(*

lossH8G;��?       �	W�7hXc�A�(*

loss`'M<�?f�       �	�g8hXc�A�(*

loss$"�<5.�       �	T�8hXc�A�(*

loss{CE=L��B       �	̗9hXc�A�(*

loss��'=�{�       �	;9:hXc�A�(*

loss�B�;d���       �	��:hXc�A�(*

lossd��<��4�       �	w;hXc�A�(*

loss)lR<��.       �	�<hXc�A�(*

loss�"=�WG]       �	��<hXc�A�(*

loss���:�\6       �	�h=hXc�A�(*

loss�� <M,�N       �	>hXc�A�(*

loss(��<�a�       �	]�>hXc�A�(*

loss�v�;��(       �	�N?hXc�A�(*

loss��<�R�       �	�?hXc�A�(*

loss�*K;�6�       �	s�@hXc�A�(*

loss<�O<b��       �	$AhXc�A�(*

loss�;S;�}�K       �	z�AhXc�A�(*

loss���:3�L�       �	:]BhXc�A�(*

lossS%�=2)�       �	��BhXc�A�(*

loss���<�4A�       �	��ChXc�A�(*

loss�A�<؜       �	�/DhXc�A�(*

loss�j�<���9       �	��DhXc�A�(*

loss��=��"       �	kEhXc�A�(*

loss{��;���n       �	WFhXc�A�(*

loss��;���       �	¤FhXc�A�(*

loss!;R=a��       �	oJGhXc�A�(*

loss��:� 6~       �	m�GhXc�A�(*

lossl)�;�f�       �	��HhXc�A�(*

loss��<c�D�       �	V)IhXc�A�(*

loss��K<�m�       �	�IhXc�A�(*

loss��[:��b�       �	�JhXc�A�(*

loss�$�;���a       �	)%KhXc�A�(*

lossZ3m<���       �	��KhXc�A�(*

loss2��;��       �	�hLhXc�A�(*

lossq�=��=       �	�LhXc�A�(*

lossQɍ<�]�J       �	/�MhXc�A�(*

lossN�;��       �	�9NhXc�A�(*

loss�W<�V�       �	)�NhXc�A�(*

loss�<N��       �	3�OhXc�A�(*

lossL�;]�,�       �	�@PhXc�A�(*

lossQ>Y;U�       �	��PhXc�A�(*

loss1�<�UX�       �	�}QhXc�A�(*

loss�o�<*
Z       �	qRhXc�A�(*

loss/�:���f       �	(�RhXc�A�(*

lossIj�;,�Jq       �	�WShXc�A�(*

loss���<z+�       �	�ShXc�A�(*

loss�E;4o       �	��ThXc�A�(*

loss��F;>M�       �	3UhXc�A�(*

lossn<=��       �	��UhXc�A�(*

loss&d<��       �	T�VhXc�A�(*

loss�4.<�˼�       �	�GWhXc�A�(*

loss�^l;��MK       �	��WhXc�A�(*

loss��_=�/�T       �	~XhXc�A�(*

losscޮ<mJ�        �	;YhXc�A�(*

loss��;�U       �	>�YhXc�A�(*

loss��9(�I       �	nhZhXc�A�(*

loss4�:���       �	�[hXc�A�(*

loss��;1� �       �	V�[hXc�A�(*

loss�J);�J\       �	�K\hXc�A�(*

loss�4�;u�K`       �	��\hXc�A�(*

loss[VG:\�       �	�^hXc�A�(*

loss��#<k2F       �	ϡ^hXc�A�(*

lossta`;�        �	�p_hXc�A�(*

loss�s�;��W�       �	�"`hXc�A�(*

loss�&�;}��       �	7RahXc�A�(*

loss2<<FPC       �	��ahXc�A�(*

loss�w<��        �	��bhXc�A�(*

loss�ҧ;M:��       �	HlchXc�A�(*

loss��;N� �       �	<�dhXc�A�(*

loss�x<���       �	�AehXc�A�(*

lossI�	=K'L�       �	�fhXc�A�(*

loss���=x��       �	]ghXc�A�(*

loss�y�<4�\�       �	`�ghXc�A�(*

lossS�}<[ē\       �	��hhXc�A�(*

losss'V=�o�       �	�KihXc�A�(*

loss���;T�{       �	��ihXc�A�(*

loss���;�/�       �	��jhXc�A�(*

loss�T�<��;i       �	v2khXc�A�(*

loss�xJ<��0w       �	��khXc�A�(*

loss?��;��ٹ       �	,dlhXc�A�(*

lossQ�
=��       �	3�lhXc�A�(*

loss�p�8O���       �	*�mhXc�A�(*

loss�E�<'^d�       �	qYnhXc�A�(*

loss;W6=��*�       �	Z�nhXc�A�(*

lossq�v;j�r       �	U�ohXc�A�(*

loss��$<�峣       �	�PphXc�A�(*

loss�MY;2�d�       �	p�phXc�A�(*

loss���;^���       �	��qhXc�A�(*

loss�]�;���C       �	�<rhXc�A�(*

lossV8=�F�       �	��rhXc�A�(*

loss��>=���?       �	�mshXc�A�(*

lossډ�;�N�       �	�thXc�A�(*

loss�3�<#�'       �	��thXc�A�(*

loss"b<VZ��       �	&muhXc�A�(*

loss&�;�U�       �	5vhXc�A�(*

loss
�;>kGv       �	ȲwhXc�A�(*

loss�a�;��       �	9FxhXc�A�(*

loss��;$Hy       �	��xhXc�A�(*

lossQcs=�)U�       �	�yhXc�A�(*

loss���==L��       �	�,zhXc�A�(*

lossz��:����       �	��zhXc�A�(*

lossH.<�>�       �	�_{hXc�A�(*

loss��=��ud       �	G|hXc�A�(*

loss<�)<,��       �	�|hXc�A�(*

loss$��;���{       �	�B}hXc�A�(*

loss�<��z�       �	��}hXc�A�(*

loss�Vt;����       �	_y~hXc�A�)*

loss`��:a�A       �	�hXc�A�)*

loss߫;;��k       �	G�hXc�A�)*

lossI¼<�1��       �	�C�hXc�A�)*

loss��<����       �	�ڀhXc�A�)*

loss�� <�	է       �	�q�hXc�A�)*

lossJD}<A�5�       �	��hXc�A�)*

loss�?�=}�j�       �	坂hXc�A�)*

loss��;<���D       �	G>�hXc�A�)*

loss��;=1�       �	=уhXc�A�)*

loss�aQ<�� %       �	�g�hXc�A�)*

loss��;����       �	���hXc�A�)*

lossnd=���p       �	:��hXc�A�)*

loss*&;���       �	+�hXc�A�)*

loss��;OG-9       �	���hXc�A�)*

loss�Z;(�R�       �	�X�hXc�A�)*

loss��(=���       �	��hXc�A�)*

loss<9�<#
��       �	>{�hXc�A�)*

loss��H<Gx��       �	��hXc�A�)*

loss�:{�K       �	��hXc�A�)*

loss�<"�na       �	A�hXc�A�)*

lossta�;"<�M       �	��hXc�A�)*

loss���<�s7�       �	r��hXc�A�)*

loss�pn<���b       �	J�hXc�A�)*

loss�5�;��q�       �	Y��hXc�A�)*

lossQJ�<i
	       �	Gu�hXc�A�)*

loss�rQ<�vl�       �	 �hXc�A�)*

loss��;����       �	b��hXc�A�)*

loss��6:       �	�:�hXc�A�)*

lossT(;-CD�       �	�ҏhXc�A�)*

loss�3�;�w�7       �	�g�hXc�A�)*

loss�ڔ:T$�       �		��hXc�A�)*

loss�==�9K�       �	>��hXc�A�)*

loss��<w�ܔ       �	�'�hXc�A�)*

loss!X=�IN(       �	vhXc�A�)*

loss�O�<��Ӭ       �	�R�hXc�A�)*

lossq��;�KI�       �	��hXc�A�)*

loss÷;��ue       �	z��hXc�A�)*

loss�(k<�^��       �	/�hXc�A�)*

loss}�;6Y�       �	_ӕhXc�A�)*

lossL�;���       �	�l�hXc�A�)*

loss�LG;ێm�       �	��hXc�A�)*

lossT�;�F��       �	 ��hXc�A�)*

loss2=�       �	�8�hXc�A�)*

lossZ<����       �	o՘hXc�A�)*

losskU:�*w       �	�q�hXc�A�)*

loss�Ň;އ��       �	��hXc�A�)*

loss/;<CS       �	��hXc�A�)*

lossVY�;!�u       �	z��hXc�A�)*

loss2�<��e       �	=e�hXc�A�)*

loss�M�;��*       �	7�hXc�A�)*

loss\�j<4
�       �	�)�hXc�A�)*

loss`�<��q?       �	��hXc�A�)*

lossw#�<���       �	�͟hXc�A�)*

loss���:+�0       �	|a�hXc�A�)*

loss	EG<uEB       �	�s�hXc�A�)*

lossz�:�~�       �	]�hXc�A�)*

loss";��<�       �	5��hXc�A�)*

loss�u;�x       �	���hXc�A�)*

lossF0�<?       �	�&�hXc�A�)*

loss/��<���       �	���hXc�A�)*

lossr�I;jzF       �	p\�hXc�A�)*

lossc,�<d���       �	�8�hXc�A�)*

loss���;%j�|       �	{٦hXc�A�)*

loss�u;;7LC�       �	�s�hXc�A�)*

loss��?;XB�_       �	�	�hXc�A�)*

loss�<��7�       �	]��hXc�A�)*

loss��:�ᜈ       �	G<�hXc�A�)*

lossZ��;z���       �	��hXc�A�)*

loss���;ڶn�       �	��hXc�A�)*

loss�u;��O       �	��hXc�A�)*

loss�<�M�Y       �	?T�hXc�A�)*

loss�.�;{3B�       �	��hXc�A�)*

lossU�
;��       �	���hXc�A�)*

loss)��;���L       �	@1�hXc�A�)*

loss�0�:S��       �	i�hXc�A�)*

loss"؂<f`�       �	|�hXc�A�)*

lossC�=�9       �	��hXc�A�)*

loss��D<0�"�       �	�hXc�A�)*

loss�Ѐ<��"       �	U�hXc�A�)*

loss��<��;       �	��hXc�A�)*

losssR�<>��       �	���hXc�A�)*

loss%�;Q���       �	�2�hXc�A�)*

loss��d:H�4       �	�سhXc�A�)*

loss\x=��qK       �	�w�hXc�A�)*

loss<Љ9Wx�       �	�hXc�A�)*

lossvR9Nj]�       �	���hXc�A�)*

lossm/,<�<�       �	EJ�hXc�A�)*

loss �:c��j       �	��hXc�A�)*

lossq�<�@T�       �	���hXc�A�)*

loss	�;+F/       �	��hXc�A�)*

lossWb�;��'       �	���hXc�A�)*

loss��<gc��       �	�I�hXc�A�)*

loss��:��Y       �	e�hXc�A�)*

lossO�8[ā       �	���hXc�A�)*

loss�Js8�C�J       �	r�hXc�A�)*

lossj��9�^�?       �	���hXc�A�)*

loss.��<s��       �	�E�hXc�A�)*

loss�p�;�eyx       �	�ܼhXc�A�)*

loss��<\�(5       �	���hXc�A�)*

lossz+{=Y3       �	�1�hXc�A�)*

lossZ��=���       �	��hXc�A�)*

loss�
:90&~       �	���hXc�A�)*

loss�"=����       �	�j�hXc�A�)*

loss6�;<�x+       �	�hXc�A�)*

lossw��=ɹ��       �	W��hXc�A�)*

loss�:�[�5       �	rM�hXc�A�)*

loss?�< �8       �	���hXc�A�)*

losscL�:lP?�       �	|~�hXc�A�)*

loss��:��?       �	�hXc�A�)*

lossA�(:��"       �	���hXc�A�)*

loss.�<6�       �	�c�hXc�A�)*

lossc\J<Zt��       �	���hXc�A�)*

lossS$�;;��       �	��hXc�A�)*

loss���<��W�       �	=,�hXc�A�)*

loss-�: A�q       �	���hXc�A�)*

loss��<��.Z       �	9a�hXc�A�)*

lossJ=�D�7       �	���hXc�A�)*

loss�9<v���       �	���hXc�A�)*

loss�:V<��0       �	f-�hXc�A�)*

lossW�D;dh�       �	i��hXc�A�)*

loss�f:hYfc       �	�Y�hXc�A�)*

lossd�3:�q��       �	(��hXc�A�)*

loss\�H<�A�Z       �	z��hXc�A�)*

loss�$�=�2\X       �	K�hXc�A�)*

lossȖ�;B���       �	���hXc�A�)*

loss4}=n���       �	�k�hXc�A�)*

loss�<ݮ��       �	��hXc�A�)*

loss3v
<�?�       �	���hXc�A�**

loss�T�=��x       �	?�hXc�A�**

lossQT=7���       �	l�hXc�A�**

lossI�<>�y�       �	��hXc�A�**

loss��;^��G       �	@��hXc�A�**

loss��;���       �	�C�hXc�A�**

loss_��<y��       �	���hXc�A�**

loss ��;���	       �	�}�hXc�A�**

loss��;,pX       �	e�hXc�A�**

loss̗�:����       �	���hXc�A�**

loss��<j?��       �	�N�hXc�A�**

lossm�=���I       �	���hXc�A�**

loss�3=lɬ       �	���hXc�A�**

loss�71<���       �	. �hXc�A�**

loss㬏;��?2       �	w��hXc�A�**

loss�L�< �%       �	hY�hXc�A�**

loss	�<�_$       �	@��hXc�A�**

loss�a�;�ü       �	}��hXc�A�**

loss1O�;�u�       �	c}�hXc�A�**

loss���;<�2       �	�w�hXc�A�**

loss2s�<�m�       �	��hXc�A�**

loss|�;Cv��       �	���hXc�A�**

loss�d~<��       �	cc�hXc�A�**

lossq(�;=���       �	`�hXc�A�**

lossd�:Q`�       �	���hXc�A�**

loss�G=w�X�       �	|c�hXc�A�**

loss�ӂ<\��^       �	b��hXc�A�**

loss��;Iem�       �	T��hXc�A�**

loss!��;SԊ       �	�.�hXc�A�**

loss{��<��g�       �	6�hXc�A�**

loss"�"= �ɜ       �	���hXc�A�**

loss�P{;2 Y�       �	C<�hXc�A�**

lossԗN<Q��{       �	J��hXc�A�**

lossة<I�)       �	h�hXc�A�**

lossq(=~��R       �	h�hXc�A�**

loss��:=���       �	%>�hXc�A�**

loss��=�h�       �	���hXc�A�**

loss@�[=2��Y       �	Jy�hXc�A�**

loss�V�;��π       �	� iXc�A�**

losss��;�^/�       �	�� iXc�A�**

loss�y[;O؎d       �	n�iXc�A�**

loss#5 9��va       �	@iXc�A�**

loss}\"=Y>�       �	/�iXc�A�**

lossg�;qY �       �	2viXc�A�**

loss��1=`��       �	�iXc�A�**

loss0�<�Q�       �	O�iXc�A�**

loss$J=jʆ'       �	�GiXc�A�**

loss�4=Y���       �	��iXc�A�**

loss!=}n�	       �	�iXc�A�**

loss�_/<�KY�       �	ԜiXc�A�**

loss��N:��.g       �	'1iXc�A�**

loss�t<U�=�       �	��iXc�A�**

losso��;3}UQ       �	+i	iXc�A�**

loss.�<w��n       �	M
iXc�A�**

loss��/;�rdd       �	��
iXc�A�**

loss�?;�N�       �	�iXc�A�**

lossj�<*pu       �	�iXc�A�**

loss!x<	}�y       �	��iXc�A�**

loss0d=aG,       �	<PiXc�A�**

loss��!:P��       �	B�iXc�A�**

lossE=���       �	�-iXc�A�**

loss;���n       �	Y�iXc�A�**

loss[��<濹�       �	�YiXc�A�**

loss}�R=�̿x       �	�=iXc�A�**

loss| =��       �	��iXc�A�**

loss��a<u�       �	6riXc�A�**

lossV��:a��       �	
iXc�A�**

loss��;9�LZ       �	9�iXc�A�**

loss�Ù<��+�       �	�5iXc�A�**

lossR=ORs       �	1�iXc�A�**

lossj�;QG�       �	�siXc�A�**

loss�g�;W���       �	�iXc�A�**

lossg�<��{\       �	��iXc�A�**

lossȇ�:}<E       �	J@iXc�A�**

loss���:9̀       �	��iXc�A�**

loss3�;8�{�       �	UhiXc�A�**

loss��f<�B*       �	e iXc�A�**

lossHց:�o��       �	<�iXc�A�**

lossefW=��S�       �	�2iXc�A�**

loss���<r�gL       �	��iXc�A�**

loss�w6:�I
�       �	\iXc�A�**

loss��:�u��       �	�iXc�A�**

loss���:'��       �	0iXc�A�**

loss<U�;1x^�       �	�PiXc�A�**

loss=�<�!       �	,iXc�A�**

loss���<5��       �	D iXc�A�**

loss#�$={C\7       �	6�!iXc�A�**

loss�0P:��VS       �	��"iXc�A�**

loss��2<+ �       �	O>#iXc�A�**

loss@DX;Ւ�s       �	g�#iXc�A�**

loss|\�9˥:9       �	�$iXc�A�**

lossF<���       �	gH%iXc�A�**

loss:�:��TJ       �	�%iXc�A�**

loss2N
=Na>�       �	��&iXc�A�**

loss#�
<;���       �	�:'iXc�A�**

loss� <vV�       �	E�'iXc�A�**

loss��Q<4�z�       �	�l(iXc�A�**

loss_mU<�<�       �	O)iXc�A�**

loss4W�:H{Ff       �	&�)iXc�A�**

loss��<�k��       �	t@*iXc�A�**

loss��;mDC�       �	/�*iXc�A�**

loss��U<=�w�       �	p+iXc�A�**

lossW�<���       �	�,iXc�A�**

lossHw�;$�g       �	|�,iXc�A�**

loss�MI<֞<�       �	�/-iXc�A�**

lossz��<����       �	�-iXc�A�**

loss��=�O�       �	�.iXc�A�**

lossd�;�$�o       �	vO/iXc�A�**

loss�ߎ<���        �	�X0iXc�A�**

lossƽ&;���       �	5�0iXc�A�**

loss��;��p%       �	^�1iXc�A�**

loss/�!<A(j�       �	T92iXc�A�**

lossjp�;p�1n       �	�2iXc�A�**

loss�]<Dy 0       �	�3iXc�A�**

loss s]<�{       �	j4iXc�A�**

loss}��<jް�       �	��4iXc�A�**

loss�y1=�3�       �	1�5iXc�A�**

loss��;ă��       �	�/6iXc�A�**

loss�?<�1�(       �	��6iXc�A�**

loss�}�=�zk       �	l^7iXc�A�**

loss�!�;��S�       �	�8iXc�A�**

loss@&<�gv�       �	��8iXc�A�**

loss\��;֠>�       �	��9iXc�A�**

loss�<%�       �	v:iXc�A�**

lossJZ);�PD@       �	Ů:iXc�A�**

loss���;P�Y       �	,D;iXc�A�**

loss��Q:�t^�       �	��;iXc�A�**

lossL��:�2*       �	p<iXc�A�**

loss=Z<�B�%       �	�=iXc�A�+*

loss�H=*���       �	S�=iXc�A�+*

lossL��<�-X       �	�j>iXc�A�+*

loss�f�9���       �	K ?iXc�A�+*

loss�r:zZ/�       �	�?iXc�A�+*

loss�x"<3�ڟ       �	Y@iXc�A�+*

loss 79;�)�       �	�/AiXc�A�+*

loss$y�<Òh�       �	*�AiXc�A�+*

loss�� <�05       �	
�BiXc�A�+*

loss�2�=N6q�       �	*9CiXc�A�+*

lossv�D:��G�       �	ToDiXc�A�+*

loss{�w<�r�       �	tEiXc�A�+*

loss�,�:7䁋       �	3�EiXc�A�+*

loss�C�<
�x�       �	d:FiXc�A�+*

lossM��=>ܴ�       �	��FiXc�A�+*

loss"�;0!V�       �	��GiXc�A�+*

lossAh	;q�n       �	�HiXc�A�+*

loss�m<<t�6�       �	)�HiXc�A�+*

loss�ym;�]3N       �	�KIiXc�A�+*

loss�I�<@ $       �	L�IiXc�A�+*

loss��j;g*s�       �	�pJiXc�A�+*

loss�|�9�']       �	�KiXc�A�+*

losse�=��~       �	&�KiXc�A�+*

loss��;}R2�       �	�>LiXc�A�+*

loss@��:�+       �	q�LiXc�A�+*

lossv�=�l <       �	��MiXc�A�+*

loss��D<&�_�       �	�NiXc�A�+*

loss|�/<�=X       �	j�NiXc�A�+*

lossׂ�<;ta       �	�BOiXc�A�+*

lossm�P<8�R�       �	��OiXc�A�+*

loss�o�<�']�       �	5�PiXc�A�+*

loss)d-=�=4       �	\QiXc�A�+*

loss��<�=w�       �	,�QiXc�A�+*

loss�#�<3o=B       �	ΎRiXc�A�+*

loss��0<����       �	�%SiXc�A�+*

loss�<�SG)       �	�SiXc�A�+*

loss���;3\]�       �	�TTiXc�A�+*

lossx��:�u�       �	�UiXc�A�+*

loss�)<m=�       �	5ViXc�A�+*

loss�>&=x��       �	��ViXc�A�+*

loss��S;�	m�       �	�RWiXc�A�+*

lossI�=<��P       �	��WiXc�A�+*

loss�^;I��       �	`�XiXc�A�+*

loss�<1;1��s       �	J)YiXc�A�+*

loss�fW:>*��       �	�YiXc�A�+*

loss�ܛ;�9q       �	�[ZiXc�A�+*

loss�d
<={�       �	��ZiXc�A�+*

loss��w;���w       �	2�[iXc�A�+*

loss��,=�e��       �	�9\iXc�A�+*

lossx=Ym       �	��\iXc�A�+*

loss��>=���       �	�]iXc�A�+*

loss&�c<��K,       �	�'^iXc�A�+*

loss��o;p�g       �	��^iXc�A�+*

loss�<���       �	��_iXc�A�+*

loss��s<z]_       �	
j`iXc�A�+*

loss̠)=w���       �	5$aiXc�A�+*

loss�:�;q&ȁ       �	.�aiXc�A�+*

loss2c�<S�<n       �	�biXc�A�+*

loss2k�;_�'       �	h$ciXc�A�+*

loss�R?<+�Z3       �	��ciXc�A�+*

loss��:)�}Y       �	J�diXc�A�+*

loss�?<���       �	�MeiXc�A�+*

lossї�<��?       �	��eiXc�A�+*

loss��;�[=       �	J~fiXc�A�+*

loss{��<����       �	�giXc�A�+*

loss5
�;�F�n       �	)�giXc�A�+*

lossi�<	�?�       �	�OhiXc�A�+*

loss���<u �       �	h�hiXc�A�+*

loss� ;n�?a       �	��iiXc�A�+*

loss���9�ܥ�       �	�5jiXc�A�+*

loss[!3<r3�p       �	`�jiXc�A�+*

lossJ�:��ӓ       �	�dkiXc�A�+*

lossl̥<f,�y       �	.liXc�A�+*

loss��=��       �	/�liXc�A�+*

loss��4<ɆT�       �	1DmiXc�A�+*

loss�iH<w�B\       �	�miXc�A�+*

loss��S;<g�F       �	�niXc�A�+*

loss`�b:���       �	N'oiXc�A�+*

lossd@=��5-       �	�oiXc�A�+*

loss��2;�;&j       �	��piXc�A�+*

loss�1�<Qŧ�       �	)<qiXc�A�+*

loss5�:���M       �	��qiXc�A�+*

loss��:Oǆ       �	�jriXc�A�+*

loss���<U�       �	�siXc�A�+*

loss��T;����       �	.�siXc�A�+*

loss3��<B	�y       �	=FtiXc�A�+*

loss�<�+j�       �	w�tiXc�A�+*

loss��o;Z���       �	*suiXc�A�+*

loss���=��       �	�viXc�A�+*

loss�܄;��       �	��viXc�A�+*

loss��=_�\       �	�<wiXc�A�+*

lossH?+:B��       �	�wiXc�A�+*

loss�\<���       �	{ixiXc�A�+*

lossĤ�<t�S       �	��xiXc�A�+*

lossE��;�(2*       �	K�yiXc�A�+*

loss:d;~��       �	=+ziXc�A�+*

loss'p;���=       �	��ziXc�A�+*

loss�g�;`��       �	�Y{iXc�A�+*

loss�ӏ<(��       �	Z+|iXc�A�+*

lossԠ�;v�˸       �	��|iXc�A�+*

loss�-=���       �	m�}iXc�A�+*

loss�5<VDϖ       �	N~iXc�A�+*

loss��;y���       �	��~iXc�A�+*

losss�W=���       �	r�iXc�A�+*

loss�]�<>4�       �	xa�iXc�A�+*

loss{c=N!h       �	��iXc�A�+*

loss�%,=zI	A       �	��iXc�A�+*

loss:�=�R�       �	�=�iXc�A�+*

loss�@)<U��       �	؂iXc�A�+*

loss�j�:$q�       �	�u�iXc�A�+*

losss!_<�*�       �	��iXc�A�+*

loss�u�<��       �	���iXc�A�+*

loss�0=���       �	�D�iXc�A�+*

loss��=l���       �	�܅iXc�A�+*

loss
��;;�H       �	[z�iXc�A�+*

loss�u�=����       �	f�iXc�A�+*

lossL<8��       �	���iXc�A�+*

loss~<�VD�       �	�D�iXc�A�+*

loss��;Q�       �	݈iXc�A�+*

loss_��;	Ȍ�       �	�r�iXc�A�+*

lossT�8<;�M       �	N�iXc�A�+*

loss[�3<rIN1       �	{��iXc�A�+*

loss���;/tӷ       �	"4�iXc�A�+*

loss���<<��       �	�ȋiXc�A�+*

lossX�;Yi       �	�_�iXc�A�+*

loss�e<���1       �	8�iXc�A�+*

loss��<�q�       �	?�iXc�A�+*

lossQ�?<74a�       �	sَiXc�A�+*

lossM��:ڳ>       �	�t�iXc�A�,*

loss2<=;ug       �	T5�iXc�A�,*

lossdb];��4       �	��iXc�A�,*

loss\?�;��G�       �	N~�iXc�A�,*

loss|�<ge"o       �	��iXc�A�,*

loss�bG<# �U       �	��iXc�A�,*

loss�==;��_       �	�E�iXc�A�,*

loss,1�;d=�       �	��iXc�A�,*

lossa�;�].       �	���iXc�A�,*

loss ~"=��J       �	�'�iXc�A�,*

loss>[�<��j       �	qʕiXc�A�,*

loss4Yg<�2K       �	�d�iXc�A�,*

loss���<g�R\       �	%�iXc�A�,*

loss|C[=u�~�       �	u��iXc�A�,*

loss� v;���V       �	�J�iXc�A�,*

loss�xY<1��       �	}�iXc�A�,*

lossL
?;8*�       �	���iXc�A�,*

loss�:�;�%��       �	�0�iXc�A�,*

lossJ��;����       �	d˚iXc�A�,*

lossa�5=�-9n       �	�b�iXc�A�,*

loss$�$=zfM"       �	���iXc�A�,*

lossf�F=Z�&       �	�:�iXc�A�,*

loss�͌<x��       �	(՝iXc�A�,*

loss���<�/#/       �	q�iXc�A�,*

loss�̚<[��       �	d�iXc�A�,*

loss�;��s       �	a��iXc�A�,*

loss���<�wq�       �	�>�iXc�A�,*

loss��<;���       �	�"�iXc�A�,*

loss2"<���J       �	'��iXc�A�,*

loss@A�<у<�       �	�`�iXc�A�,*

lossE��;�/�!       �	u�iXc�A�,*

loss#;�e�       �	֣iXc�A�,*

loss3<h=@�t�       �	�m�iXc�A�,*

loss�w�<J��       �	�iXc�A�,*

loss�KQ:9+��       �	���iXc�A�,*

lossi8�=쌢g       �	X<�iXc�A�,*

loss&��;j�<       �	c��iXc�A�,*

loss-�Y<�E�       �	��iXc�A�,*

loss]<�X�:       �	��iXc�A�,*

loss�#�;-nG�       �	¨iXc�A�,*

loss�X�;	���       �	�U�iXc�A�,*

loss���;hD<�       �	�&�iXc�A�,*

loss��N;_+       �	+��iXc�A�,*

loss��=���       �	�V�iXc�A�,*

lossʀ�;�ݞU       �	��iXc�A�,*

loss$�;�_f       �	f��iXc�A�,*

loss%�<��
V       �	-A�iXc�A�,*

loss��<h�qv       �	O�iXc�A�,*

loss��<%�       �	���iXc�A�,*

loss
n�;Ώ1
       �	�J�iXc�A�,*

loss��;��Q�       �	��iXc�A�,*

lossJ:3;�I\       �	X��iXc�A�,*

loss��W=�(/r       �	]5�iXc�A�,*

loss˪=���       �	L޲iXc�A�,*

lossW߭<p��w       �	�{�iXc�A�,*

loss���<�K��       �	'�iXc�A�,*

loss��:=1RD�       �	���iXc�A�,*

loss��0;��;       �	�c�iXc�A�,*

loss��B:U]�M       �	)�iXc�A�,*

lossvoq9]��       �	���iXc�A�,*

loss�j�;�.�C       �	�C�iXc�A�,*

lossW��;���       �	DݷiXc�A�,*

loss���:`:        �	n��iXc�A�,*

lossF2*<��*       �	�-�iXc�A�,*

loss�A<`~��       �	S̹iXc�A�,*

loss�%<v�W�       �	'�iXc�A�,*

loss�N<=z���       �	\��iXc�A�,*

loss�><{�(�       �	F�iXc�A�,*

loss��<��	�       �	@�iXc�A�,*

loss}�6<�N�       �	Y½iXc�A�,*

loss��t<���       �	c�iXc�A�,*

lossA�;d�q|       �	���iXc�A�,*

loss;�C<�Ӎd       �	���iXc�A�,*

loss���<}��       �	�M�iXc�A�,*

loss�/�<���       �	qU�iXc�A�,*

loss~D:��$3       �	��iXc�A�,*

loss{<��˸       �	���iXc�A�,*

lossJ6:<Ur\       �	�a�iXc�A�,*

loss��	;`��       �	��iXc�A�,*

loss�<�R�^       �	���iXc�A�,*

loss�w�<�`s�       �	���iXc�A�,*

loss�:<�Tb�       �	�m�iXc�A�,*

loss�~�<���       �	F�iXc�A�,*

loss�Ҕ:�Vs       �	��iXc�A�,*

loss�B=>A�s       �	 z�iXc�A�,*

loss�4�;V�h%       �	��iXc�A�,*

loss���<��?       �	S��iXc�A�,*

loss\�+<"t�       �	rR�iXc�A�,*

loss�w>;M �       �	���iXc�A�,*

loss �I<�cP       �	��iXc�A�,*

loss�(^=���       �	<2�iXc�A�,*

loss���;1�'       �	���iXc�A�,*

loss��?<�T$�       �	Ct�iXc�A�,*

loss���<�f6{       �	/�iXc�A�,*

loss��2<�(y�       �	���iXc�A�,*

losssA;�)�       �	�Q�iXc�A�,*

loss���;�l�       �	��iXc�A�,*

loss��<H��	       �	���iXc�A�,*

loss�^�<c���       �	f��iXc�A�,*

loss[ʚ:�/��       �	�[�iXc�A�,*

loss��<��       �	���iXc�A�,*

lossq�[:�~�       �	��iXc�A�,*

loss�B;o�w�       �	�V�iXc�A�,*

loss�(=���       �	D��iXc�A�,*

loss�V=����       �	��iXc�A�,*

lossx�E<tJ��       �	�d�iXc�A�,*

loss*�; ���       �	��iXc�A�,*

loss�TL<M���       �	���iXc�A�,*

loss8��<hv�=       �	�?�iXc�A�,*

loss�� ;#Gi�       �	��iXc�A�,*

lossA��:��P�       �	�p�iXc�A�,*

loss��<�D�e       �	��iXc�A�,*

lossv�;�s       �	ޫ�iXc�A�,*

lossL�=�k%       �	�K�iXc�A�,*

loss�i=!r       �	���iXc�A�,*

loss�В;&�ʽ       �	!�iXc�A�,*

lossN�<�G�^       �	��iXc�A�,*

loss�	�<?�	5       �	r�iXc�A�,*

lossJ{c<|�P�       �	��iXc�A�,*

lossH:�;��	       �	*�iXc�A�,*

loss��;/\��       �	�U�iXc�A�,*

loss�<0\-       �	CT�iXc�A�,*

lossƄ�:��=z       �		�iXc�A�,*

lossii�;!��f       �	S��iXc�A�,*

lossQ��:e�:       �	I��iXc�A�,*

loss��;�h��       �	�#�iXc�A�,*

lossZ,�;Ɔ�v       �	�/�iXc�A�,*

lossMcz;A��{       �	
�iXc�A�,*

loss�wW<�g��       �	ߨ�iXc�A�-*

lossĆI<@/k       �	dX�iXc�A�-*

loss��V<����       �	���iXc�A�-*

lossdk=G��G       �	^��iXc�A�-*

loss�=\��       �	���iXc�A�-*

loss<S�;��&=       �	O@�iXc�A�-*

loss�?<6�*       �	���iXc�A�-*

loss�L�;ل�<       �	���iXc�A�-*

loss�x�:�d`�       �	\r�iXc�A�-*

loss[��=���       �	��iXc�A�-*

loss���;'��       �	ū�iXc�A�-*

losss��<���       �	֌�iXc�A�-*

lossv�:׭:�       �	V(�iXc�A�-*

loss� ;�i�       �	��iXc�A�-*

loss���:����       �	fh�iXc�A�-*

loss`Y=W��       �	*�iXc�A�-*

lossEqy;�L�       �	���iXc�A�-*

loss(�<�m�B       �	�2�iXc�A�-*

loss�&<����       �	���iXc�A�-*

loss�X<F%l�       �	�p�iXc�A�-*

loss��;_^�       �	��iXc�A�-*

loss�=<���       �	���iXc�A�-*

loss%}�=j�R       �	y[�iXc�A�-*

loss��1=*9�Y       �	
��iXc�A�-*

loss�)<tG)r       �	ޏ�iXc�A�-*

loss]��;ij^�       �	F(�iXc�A�-*

loss�WM<5��       �	'��iXc�A�-*

loss�n=%�j       �	Zd�iXc�A�-*

loss���;{}$o       �	��iXc�A�-*

loss.j<�T�       �	a��iXc�A�-*

loss3�J;��Ơ       �	CU�iXc�A�-*

lossҌ<?Z��       �	^��iXc�A�-*

loss,p�<�-�E       �	m��iXc�A�-*

loss�=G;@�       �	.�iXc�A�-*

loss�c�;Cr�N       �	���iXc�A�-*

loss�_><��<�       �	�s�iXc�A�-*

loss(R�<����       �	b�iXc�A�-*

loss��:��C       �	���iXc�A�-*

loss���;� `f       �	�o jXc�A�-*

loss.J�;s��       �	$jXc�A�-*

loss�� =P�@�       �	l�jXc�A�-*

loss�7�:Q"|       �	fLjXc�A�-*

loss�79;�       �	��jXc�A�-*

loss�~�<Z�       �	ծjXc�A�-*

loss��<t��n       �	�NjXc�A�-*

loss�߅<��V       �	p�jXc�A�-*

loss�f<�o�6       �	'�jXc�A�-*

loss<ݡ;��       �	�jXc�A�-*

loss��^;�C�       �	��jXc�A�-*

loss�<67d`       �	�NjXc�A�-*

loss1^�<Pgh-       �	��jXc�A�-*

loss6v�<�ooq       �	&�jXc�A�-*

loss� >Eo�o       �	�,	jXc�A�-*

loss8�<��.C       �	��	jXc�A�-*

lossP}<#'��       �	'k
jXc�A�-*

loss���<�6��       �	jXc�A�-*

loss�%y;��i       �	��jXc�A�-*

loss/��:aOW       �	��jXc�A�-*

loss���;�n&O       �	�'jXc�A�-*

lossSD�<w�-C       �	f�jXc�A�-*

lossOf="�       �	�UjXc�A�-*

loss"�;rU�>       �	�jXc�A�-*

loss�-=��9�       �	��jXc�A�-*

lossH��:�B9i       �	�-jXc�A�-*

lossv0+<�K5�       �	��jXc�A�-*

loss? =  ��       �	MhjXc�A�-*

loss��<�       �	Q3jXc�A�-*

loss
=���       �	��jXc�A�-*

lossH�<@߾'       �	��jXc�A�-*

lossJ�;�Z�       �	 jXc�A�-*

loss^;��q:       �		�jXc�A�-*

loss�A:+X�       �	�_jXc�A�-*

loss,�E;G:,       �	��jXc�A�-*

lossWD�;�2�x       �	Q�jXc�A�-*

loss��/<C�4s       �	�@jXc�A�-*

loss��<��       �	��jXc�A�-*

loss�F<+�7       �	5}jXc�A�-*

lossd ;n�߀       �	�jXc�A�-*

loss�}2<!/>�       �	�jXc�A�-*

loss��x<l:Q       �	BZjXc�A�-*

loss@ >;�#��       �	��jXc�A�-*

loss�	�<���       �	��jXc�A�-*

loss?OS=~��       �	.jXc�A�-*

loss�<e�b�       �	��jXc�A�-*

lossƚG<-9       �	�pjXc�A�-*

loss6b;�P�       �	4fjXc�A�-*

lossr�+<�R�n       �	jXc�A�-*

loss/�<�G)L       �	��jXc�A�-*

loss��%<PTD�       �	\: jXc�A�-*

lossDד<U��'       �	�� jXc�A�-*

loss:-�<o&i�       �	��!jXc�A�-*

lossQ�5=DD�       �	RD"jXc�A�-*

loss ��:IS��       �	'#jXc�A�-*

loss���:Aw�a       �	��#jXc�A�-*

loss�\�;���       �	��$jXc�A�-*

loss���<{lKZ       �	�v%jXc�A�-*

loss಩<b�       �	�e&jXc�A�-*

loss�pR<���       �	)'jXc�A�-*

loss�s�<;�7       �	l(jXc�A�-*

loss]О;PR��       �	ɬ(jXc�A�-*

loss��<]��F       �	_a)jXc�A�-*

loss�$�<�R}       �	m�)jXc�A�-*

loss��<�Ѕ{       �	c�*jXc�A�-*

loss�'�:c9U       �	�b+jXc�A�-*

loss}P-=�X��       �	-	,jXc�A�-*

lossQJ�<�2�       �	B�,jXc�A�-*

lossR�<<_D�       �	6�-jXc�A�-*

loss��=�t�       �	/n.jXc�A�-*

lossO�;�+        �	�U/jXc�A�-*

lossT"�;NM�e       �	��/jXc�A�-*

loss̑$;(w�d       �	ũ0jXc�A�-*

loss%/�<�kV�       �	w1jXc�A�-*

loss7�;��'       �	�I2jXc�A�-*

loss!�s;�J��       �	>�2jXc�A�-*

loss�s=�1��       �	r�3jXc�A�-*

loss��1<j`2       �	u4jXc�A�-*

loss��=:G��e       �	�y5jXc�A�-*

loss�A):�e.�       �	� 6jXc�A�-*

loss츏;����       �	�6jXc�A�-*

lossRK�;�N�       �	p`7jXc�A�-*

loss�Qs=BQ       �	 8jXc�A�-*

lossϡ�=�%ϡ       �	��8jXc�A�-*

loss_)�:5��T       �	��9jXc�A�-*

loss�:@"g�       �	�7:jXc�A�-*

loss)�,;	D��       �	��:jXc�A�-*

loss�8;���       �	�o;jXc�A�-*

loss̮t<���8       �	S<jXc�A�-*

loss�j�;J	O�       �	t�<jXc�A�-*

lossi�<.��N       �	o/=jXc�A�.*

losst�=���L       �	/�=jXc�A�.*

loss��=d|�       �	�S>jXc�A�.*

losse\s<)?�	       �	�>jXc�A�.*

loss���;q�_k       �	�?jXc�A�.*

lossj� ;cHl�       �	-$@jXc�A�.*

loss��k=c7�       �	��@jXc�A�.*

loss?e2<,�I       �	iVAjXc�A�.*

loss%�D:kZO       �	y�AjXc�A�.*

lossQ<�;F+�       �	�~BjXc�A�.*

loss�6)<-:�G       �	�CjXc�A�.*

loss�9�;A�Q�       �	��CjXc�A�.*

lossOV?;5��       �	K>DjXc�A�.*

loss��=�� �       �	��DjXc�A�.*

loss��#<��*�       �	&mEjXc�A�.*

lossG�<��       �	�FjXc�A�.*

loss<� :zk�m       �	��FjXc�A�.*

loss�<�ɿ       �	A.GjXc�A�.*

loss�;�~�}       �	z�GjXc�A�.*

lossv��:���       �	�aHjXc�A�.*

lossM��:*���       �	y:IjXc�A�.*

loss�b;�$�       �	��IjXc�A�.*

lossƦ<��x:       �	lJjXc�A�.*

loss�{�;6��)       �	�KjXc�A�.*

loss�z_<0ر       �	{�KjXc�A�.*

losswa{<R��       �	(FLjXc�A�.*

loss�.�;0���       �	��LjXc�A�.*

lossq?M;��SG       �	�yMjXc�A�.*

loss)�:�4�       �	 NjXc�A�.*

loss��=��&p       �	_�NjXc�A�.*

loss��x<�'�       �	LROjXc�A�.*

loss��<4�B       �	1�OjXc�A�.*

loss�j�:{n+       �	�PjXc�A�.*

lossC�q;���       �	1$QjXc�A�.*

loss�~�:���J       �	��QjXc�A�.*

loss���:FT��       �	zSRjXc�A�.*

loss[��;/�       �	��RjXc�A�.*

loss�==�*       �	��SjXc�A�.*

lossO��<�l       �	)<TjXc�A�.*

loss1�:=��I�       �	L�TjXc�A�.*

lossH�-=�E'       �	�wUjXc�A�.*

loss�;��@)       �	VjXc�A�.*

loss���;"��       �	� WjXc�A�.*

loss��<b�8�       �	v�WjXc�A�.*

loss�S�<�e�       �	�^XjXc�A�.*

loss�|;Mv:       �	~�XjXc�A�.*

lossm�:L��i       �	�YjXc�A�.*

lossQ�.<�G��       �	�\ZjXc�A�.*

loss�.�;�Ҙ       �	��ZjXc�A�.*

loss��D<��       �	|�[jXc�A�.*

loss��;�}       �	�5\jXc�A�.*

loss�N$;P�2�       �	K�\jXc�A�.*

loss���;����       �	�z]jXc�A�.*

loss��v<�       �	�_jXc�A�.*

lossQ�f:�Q       �	��_jXc�A�.*

lossx�8*M(�       �	H`jXc�A�.*

loss�F%:�Dl�       �	��`jXc�A�.*

loss�=e���       �	ēajXc�A�.*

loss�o<�&J�       �	�3bjXc�A�.*

losszi�;���       �	��bjXc�A�.*

loss�Og:��O       �	kecjXc�A�.*

lossRab=��       �	��cjXc�A�.*

loss�]�:QR=�       �	J�djXc�A�.*

lossü=��	       �	BejXc�A�.*

loss�{"<מ=       �	��ejXc�A�.*

lossa�;@��       �	e�fjXc�A�.*

lossW�<;K,�       �	N(gjXc�A�.*

loss�ܟ=�L�       �	�hjXc�A�.*

lossS1�:�cA�       �	)�hjXc�A�.*

loss#0Y<P��-       �	�0ijXc�A�.*

lossq��:��g       �	��ijXc�A�.*

loss��_;bn/       �	�_jjXc�A�.*

loss$z\;r�/�       �	b�jjXc�A�.*

loss;�<���       �	S�kjXc�A�.*

loss}u;q)�       �	LljXc�A�.*

lossI�I;�^��       �	 �ljXc�A�.*

loss��a<S�W       �	f�mjXc�A�.*

lossC
<6��:       �	�njXc�A�.*

loss���;V���       �	*�njXc�A�.*

loss��<�q       �	&mojXc�A�.*

loss�)P<0��%       �	
pjXc�A�.*

loss��;yN.       �	��pjXc�A�.*

loss��9�K͂       �	�JqjXc�A�.*

loss2�5;��       �	)�qjXc�A�.*

loss���;�7�       �	��rjXc�A�.*

loss%�;���       �	8,sjXc�A�.*

loss�U�<���       �	i�sjXc�A�.*

loss��;�(��       �	�\tjXc�A�.*

loss� ^<���0       �	�:ujXc�A�.*

loss볖;c5�L       �	��ujXc�A�.*

lossV�)<��
�       �	vmvjXc�A�.*

loss[/�;�["       �	wjXc�A�.*

loss�?%;~Wu       �	��wjXc�A�.*

loss%0@=կ�?       �	�OxjXc�A�.*

loss���;�Q�E       �	)�xjXc�A�.*

lossX�:$E�_       �	��yjXc�A�.*

loss Yk<6Q25       �	�zjXc�A�.*

loss�D<�G[�       �	�zjXc�A�.*

loss�9�*�{       �	�S{jXc�A�.*

lossxݍ;%3��       �	5�{jXc�A�.*

lossg�=���       �	��|jXc�A�.*

loss���=X<Z$       �	�@}jXc�A�.*

lossp<%9       �	y~jXc�A�.*

loss _f<)��       �	��~jXc�A�.*

loss�)<ޗ=H       �	!;jXc�A�.*

loss�0�<���       �	5�jXc�A�.*

loss�H{;��H}       �	�n�jXc�A�.*

loss�2=�,�n       �	�k�jXc�A�.*

loss(\�<7,M       �	��jXc�A�.*

lossOO�<8��       �	g��jXc�A�.*

loss���;�EU       �	�0�jXc�A�.*

lossFns;�:�{       �	dʃjXc�A�.*

loss���:X��       �	�e�jXc�A�.*

loss�{�<�4��       �	���jXc�A�.*

lossGV�=K�O       �	�'�jXc�A�.*

loss;"�<���       �	pϝjXc�A�.*

loss�,�<D��       �	{�jXc�A�.*

loss�:�<�*��       �	� �jXc�A�.*

lossғ�:��#Y       �	f؟jXc�A�.*

loss�K�;�$�       �	6�jXc�A�.*

loss�=:R��       �	���jXc�A�.*

lossA��<��0       �	䇢jXc�A�.*

loss�U�;�oY�       �	wh�jXc�A�.*

loss��q:K�       �	j�jXc�A�.*

loss��B=ތ��       �	'�jXc�A�.*

loss|�C<qLS�       �	�M�jXc�A�.*

lossa�;M��6       �	� �jXc�A�.*

loss|ݞ<���)       �	�jXc�A�.*

loss:�v;       �	[�jXc�A�/*

loss�8���       �	�*�jXc�A�/*

loss&�<�b�       �	ҩjXc�A�/*

loss~/=�Pg       �	�w�jXc�A�/*

loss��B<I�q�       �	�R�jXc�A�/*

loss�;袔�       �	S�jXc�A�/*

loss���=�k��       �	 �jXc�A�/*

loss���9P���       �	��jXc�A�/*

loss% =R;!�       �	DQ�jXc�A�/*

loss��y;�i	r       �	��jXc�A�/*

lossQ:=j+�q       �		ůjXc�A�/*

loss4�J<��       �	�^�jXc�A�/*

loss=��;��b�       �	���jXc�A�/*

loss��<����       �	k�jXc�A�/*

loss��;��0?       �	���jXc�A�/*

loss��:���       �	�$�jXc�A�/*

loss�R�;����       �	���jXc�A�/*

loss���<b���       �	 W�jXc�A�/*

loss4=sg#       �	0�jXc�A�/*

loss��p:����       �	B��jXc�A�/*

loss�5�=���	       �	�0�jXc�A�/*

lossM�I;3Q��       �	IնjXc�A�/*

lossȂN<����       �	�o�jXc�A�/*

loss�۳=D��       �	��jXc�A�/*

loss�<��       �	�øjXc�A�/*

loss��l:_6(       �	�Z�jXc�A�/*

loss]Fw;q�f�       �	��jXc�A�/*

lossq<5G*�       �	���jXc�A�/*

lossu�<R�       �	�1�jXc�A�/*

lossб<����       �	�̻jXc�A�/*

loss���;��y]       �	�jXc�A�/*

lossD@�;T\��       �	�5�jXc�A�/*

loss^T�<���C       �	�ͽjXc�A�/*

loss���;�^�*       �	�o�jXc�A�/*

loss�߫9����       �	K#�jXc�A�/*

lossE��:�km       �	�8�jXc�A�/*

loss��;N\��       �	I��jXc�A�/*

loss��m<�1��       �	��jXc�A�/*

losso �=u��       �	� �jXc�A�/*

losse��<�a�Q       �	z��jXc�A�/*

loss��$:y	�       �	�b�jXc�A�/*

loss�;9<&�       �	7��jXc�A�/*

loss��:i4�       �	%��jXc�A�/*

loss���:К@       �	~;�jXc�A�/*

lossӱx<�֥�       �	e��jXc�A�/*

loss� =��v�       �	4��jXc�A�/*

loss��f:��6       �	y!�jXc�A�/*

lossQ�:/N       �	��jXc�A�/*

loss���<�0l�       �	�Q�jXc�A�/*

losss�;]���       �	g��jXc�A�/*

loss_�:l �\       �	L��jXc�A�/*

loss�(e<�ϛ%       �	��jXc�A�/*

lossj{_<M�T�       �	��jXc�A�/*

losshz�<0yB�       �	R�jXc�A�/*

lossF��<��t       �	0��jXc�A�/*

loss@�;�l�       �	n��jXc�A�/*

loss�8�<f���       �	tF�jXc�A�/*

lossv�1;vH.       �	q��jXc�A�/*

loss��=)�LM       �	9�jXc�A�/*

loss�q+=��       �	3�jXc�A�/*

lossA\�;%JN�       �	��jXc�A�/*

lossS�<���       �	UL�jXc�A�/*

loss�!;*%e�       �	5��jXc�A�/*

loss}��;m{-W       �	���jXc�A�/*

loss�==M��y       �	fh�jXc�A�/*

loss���<��ن       �	��jXc�A�/*

lossx�q<��ݒ       �	��jXc�A�/*

lossF\=��y       �	�5�jXc�A�/*

loss��&9�HK^       �	i��jXc�A�/*

lossZ�I; E�#       �	d[�jXc�A�/*

loss[Po<���
       �	-�jXc�A�/*

loss��<�Q/�       �	ע�jXc�A�/*

loss�]�;��;       �		��jXc�A�/*

losstZ;0.��       �	dW�jXc�A�/*

lossE`;��96       �	1��jXc�A�/*

lossO=gi��       �	��jXc�A�/*

lossΆ�=����       �	'�jXc�A�/*

loss;���B       �	+��jXc�A�/*

lossľ�<�%��       �	-[�jXc�A�/*

loss��=jWT�       �	,��jXc�A�/*

lossF��;���       �	ݳ�jXc�A�/*

loss��{<F�c�       �	Gu�jXc�A�/*

loss�<�5��       �	B'�jXc�A�/*

loss� <��e�       �	j�jXc�A�/*

lossꅤ:��       �	��jXc�A�/*

loss�<J�;�       �	c|�jXc�A�/*

loss�r9c��       �	�^�jXc�A�/*

loss��:���>       �	�M�jXc�A�/*

loss��;޽�R       �	'3�jXc�A�/*

lossh�I=��0       �	��jXc�A�/*

loss ��<�A%       �	?��jXc�A�/*

loss�5:Q�!�       �	'/�jXc�A�/*

lossε=<�ꊪ       �	���jXc�A�/*

loss���<&)�N       �	&r�jXc�A�/*

loss�m<�B��       �	7�jXc�A�/*

loss.�~<��ܓ       �	���jXc�A�/*

loss��<M��       �	f�jXc�A�/*

loss�w
<�y5	       �	�	�jXc�A�/*

lossIK�:d���       �	���jXc�A�/*

lossP<�9tR       �	@P�jXc�A�/*

loss�a�8'�p�       �	���jXc�A�/*

loss��Q;��       �	���jXc�A�/*

lossӰ=��m�       �	H7�jXc�A�/*

loss1��;֐��       �	���jXc�A�/*

loss(=���        �	�y�jXc�A�/*

loss?�9;{��       �	� �jXc�A�/*

loss��u;,�       �	6��jXc�A�/*

lossCp<p�c       �	v�jXc�A�/*

loss���<�#r       �	�)�jXc�A�/*

loss�;^:�;�F       �	���jXc�A�/*

loss��D<҅�b       �	�q�jXc�A�/*

loss��	<A�F�       �	E�jXc�A�/*

loss��;2�~       �	K��jXc�A�/*

loss���<��5(       �	�P�jXc�A�/*

loss��>�/��       �	���jXc�A�/*

lossf>�:�c�l       �	S��jXc�A�/*

lossh�P<�3�b       �	ݖ�jXc�A�/*

loss�i<5m:�       �	�^�jXc�A�/*

loss�ߓ<':�       �	��jXc�A�/*

loss2EG=:L��       �	���jXc�A�/*

loss��;���       �	Ug�jXc�A�/*

lossI;B^Ы       �	���jXc�A�/*

loss�g4:�7�       �	��jXc�A�/*

lossr&�<���t       �	�2�jXc�A�/*

loss��;��       �	���jXc�A�/*

loss̺�:|���       �	���jXc�A�/*

loss-#$<��       �	�.�jXc�A�/*

lossI�)=����       �	��jXc�A�/*

loss�~+=l��&       �	[`�jXc�A�0*

lossD��<�C       �	w�jXc�A�0*

loss^o<-�)C       �	O��jXc�A�0*

loss���;���       �	�F kXc�A�0*

loss���:ȲU�       �	�� kXc�A�0*

loss|5:\��       �	e�kXc�A�0*

loss��P<5Jn       �	+2kXc�A�0*

lossh�F;,��Y       �	-�kXc�A�0*

loss=��{       �	gkXc�A�0*

lossq9}<���       �	�kXc�A�0*

lossLm4=�%H       �	�CkXc�A�0*

loss��7<x��V       �	%�kXc�A�0*

loss�ȝ:]O�d       �	t~kXc�A�0*

loss��<���       �	�kXc�A�0*

loss�@k=��e�       �	X�kXc�A�0*

loss���<5�g�       �	5AkXc�A�0*

lossO'�::0�l       �	�kXc�A�0*

loss��;��x�       �	*�	kXc�A�0*

loss#�=y       �	�3
kXc�A�0*

loss�?)=��       �	}�
kXc�A�0*

loss2�G;��t�       �	hkXc�A�0*

loss�`=z ��       �	�kXc�A�0*

loss��=m7@}       �	��kXc�A�0*

loss���:}��       �	<JkXc�A�0*

loss��<n�"       �	��kXc�A�0*

lossV��:�0#       �	:�kXc�A�0*

lossT��<7�#       �	r�kXc�A�0*

lossmt@<�
��       �	�^kXc�A�0*

loss_�;q/�       �	-kXc�A�0*

loss`�0:�ê       �	h�kXc�A�0*

loss�+<c�S�       �	�IkXc�A�0*

loss��:�J"�       �	 �kXc�A�0*

loss�g0=H��       �	L�kXc�A�0*

loss*��<Bc�8       �	�'kXc�A�0*

losszP�<�^b       �	��kXc�A�0*

loss�|�;�^M       �	�WkXc�A�0*

lossc��<��/�       �	�IkXc�A�0*

loss��;�BqD       �	�kXc�A�0*

loss���<3Y�v       �	=|kXc�A�0*

loss�܅;VR       �	�kXc�A�0*

loss�V;߮\       �	!�kXc�A�0*

lossu��:�BU�       �	��kXc�A�0*

loss���9��P       �	$&kXc�A�0*

loss�NI<r��m       �	%�kXc�A�0*

lossx�y<dH�       �	xbkXc�A�0*

lossr�;�g�\       �	hkXc�A�0*

loss�ak<T�[�       �	ߥkXc�A�0*

loss_q+<s��<       �	�LkXc�A�0*

loss!��;�uc�       �	9�kXc�A�0*

loss�k<1���       �	��kXc�A�0*

lossdf�;�Z�       �	dWkXc�A�0*

lossO��9�!       �	�( kXc�A�0*

lossl+<B��       �	�� kXc�A�0*

loss��7=�ٵ       �	I�!kXc�A�0*

loss&�H<��2�       �	��"kXc�A�0*

loss@>=�U.       �	��#kXc�A�0*

loss��:[�^o       �	W�$kXc�A�0*

loss��<�(;W       �	
d%kXc�A�0*

lossɶ�;�E)�       �	/�&kXc�A�0*

loss��Z;�7ɳ       �	#M'kXc�A�0*

lossQ��:�mqD       �	A�'kXc�A�0*

lossMq<>4��       �	Ό(kXc�A�0*

loss�Hy<���       �	}Z)kXc�A�0*

loss�ϻ<t��       �	�U*kXc�A�0*

loss�;��z�       �	+kXc�A�0*

loss�]�;��b~       �	��+kXc�A�0*

lossX*�; ���       �	2W,kXc�A�0*

loss� <�j�       �	� -kXc�A�0*

loss���;I���       �	X�-kXc�A�0*

lossŨ|;��p       �	nN.kXc�A�0*

loss��%<�#G       �	3�.kXc�A�0*

loss��=�6       �	��/kXc�A�0*

loss\�<^䯚       �	�70kXc�A�0*

loss���<.�j       �	0�0kXc�A�0*

loss��=�@�N       �	�u1kXc�A�0*

loss�{~=�CRd       �	�2kXc�A�0*

loss%r{;\��       �	�2kXc�A�0*

loss�V�<�+k       �	IL3kXc�A�0*

loss�3�;%P�       �	��3kXc�A�0*

loss�q�<��       �	D�4kXc�A�0*

loss͙�;�xX       �	�<5kXc�A�0*

loss&��<��_       �	��5kXc�A�0*

loss��;�E9t       �	ٙ6kXc�A�0*

loss�Z=�vIh       �	�A7kXc�A�0*

loss0;Qγ�       �	;�7kXc�A�0*

lossw��<@��y       �	��8kXc�A�0*

lossQ};�	��       �	�J9kXc�A�0*

loss߆�:F��       �	w�9kXc�A�0*

loss)�J<#�7       �	
�:kXc�A�0*

loss�<�;tF�d       �	Na;kXc�A�0*

lossz�;��V�       �	E<kXc�A�0*

loss�;�<�-T       �	��<kXc�A�0*

lossn.�<��M�       �	�p=kXc�A�0*

loss"�;�T�       �	&>kXc�A�0*

loss��+;7{��       �	J&?kXc�A�0*

loss���=�>@       �	�;@kXc�A�0*

loss&i�<��       �	��@kXc�A�0*

lossM�<B�$�       �	��AkXc�A�0*

loss �$=�d       �	~SBkXc�A�0*

lossi$3;�9�}       �	�	CkXc�A�0*

loss���;�m��       �	� DkXc�A�0*

loss��9<J��       �	��DkXc�A�0*

lossG�<ӌ�b       �	o�EkXc�A�0*

losswׁ<2y��       �	��FkXc�A�0*

loss_�;�VT�       �	]�GkXc�A�0*

loss��= ��       �	�tHkXc�A�0*

lossk�;�K{�       �	�IkXc�A�0*

lossF�9=Y���       �	Y�IkXc�A�0*

loss��;b�}       �	qJkXc�A�0*

lossڐ�<��i�       �	XKkXc�A�0*

lossS<���       �	��KkXc�A�0*

loss,��:"��[       �	�XLkXc�A�0*

loss�=�+�H       �	�3MkXc�A�0*

loss�m�<�W       �	��MkXc�A�0*

loss��=�ڣ�       �	�qNkXc�A�0*

loss{k-=�X'       �	�OkXc�A�0*

loss~z=���       �	�OkXc�A�0*

loss��<�P:m       �	5`PkXc�A�0*

loss
ǥ<�*F�       �	�#QkXc�A�0*

loss`@j=#�h       �	E�QkXc�A�0*

loss� <�I��       �	�gRkXc�A�0*

loss�&�="�       �	נSkXc�A�0*

loss��;̢N       �	�7TkXc�A�0*

loss�R�<��       �	��TkXc�A�0*

loss�o�;pz�-       �	'gUkXc�A�0*

loss%M<dz       �	��UkXc�A�0*

loss�'t=���       �	ՑVkXc�A�0*

loss��X<�b��       �	�(WkXc�A�0*

loss w�<GYK�       �	��WkXc�A�1*

loss��<t!�       �	��XkXc�A�1*

loss1	�=.[       �	uuYkXc�A�1*

loss��z=�`��       �	@ZkXc�A�1*

loss��=��       �	�ZkXc�A�1*

loss�Za;�+��       �	�N[kXc�A�1*

lossͨ&<�)�"       �	��[kXc�A�1*

loss��<�+�x       �	 �\kXc�A�1*

loss�D�<V�T�       �	E+]kXc�A�1*

loss�L:��z�       �	��]kXc�A�1*

loss4<y���       �	9�^kXc�A�1*

loss��;�>*�       �	D6_kXc�A�1*

loss��><�3�       �	��_kXc�A�1*

loss?�=�Ei�       �	�b`kXc�A�1*

lossfa	;K�?       �	�akXc�A�1*

loss�$�<�Ƌ�       �	�bkXc�A�1*

loss�(<#Q�       �	�bkXc�A�1*

lossH�<��       �	�jckXc�A�1*

loss��X<-��m       �	:dkXc�A�1*

loss� :�_��       �	L�dkXc�A�1*

loss�<�Þ�       �	�ekXc�A�1*

loss��;�:Be       �	�@fkXc�A�1*

loss���:�P9�       �	��fkXc�A�1*

loss栍<�o�       �	ɎgkXc�A�1*

loss��x<�\��       �	�1hkXc�A�1*

loss�;��       �	_�hkXc�A�1*

loss*��<�t�	       �	��ikXc�A�1*

lossX��<��ZX       �	$jkXc�A�1*

loss�/�<�Cu       �	��jkXc�A�1*

lossgU�;9�#�       �	�]kkXc�A�1*

lossC��<2z�       �	�lkXc�A�1*

loss!$�9J���       �	n�lkXc�A�1*

loss�Q<���       �	�JmkXc�A�1*

loss=RC�!       �	��mkXc�A�1*

loss���<��o       �	SznkXc�A�1*

lossը�:��C       �	okXc�A�1*

lossE��;(Vy       �	�okXc�A�1*

lossQF<��n�       �	LpkXc�A�1*

lossI*-;���       �	�pkXc�A�1*

loss,��;��T�       �	/�qkXc�A�1*

loss'̏;9�2       �	(+rkXc�A�1*

loss\�<e��&       �	;�rkXc�A�1*

loss�`;8��       �	N^skXc�A�1*

loss��;`�ݲ       �	z�skXc�A�1*

loss�߄<n��       �	��tkXc�A�1*

loss�nE< E�       �	=HukXc�A�1*

lossѹL;%�M7       �	�ukXc�A�1*

loss�]�<k��\       �	��vkXc�A�1*

loss���;�56       �	�$wkXc�A�1*

loss6��<���       �	��wkXc�A�1*

loss,p�<�w��       �	ͱxkXc�A�1*

lossO��<�/�       �	�FykXc�A�1*

loss�}T;"ז       �	��ykXc�A�1*

loss=��;�       �	1yzkXc�A�1*

lossWX ;�j��       �	8{kXc�A�1*

loss�ʛ<��       �	m�{kXc�A�1*

loss��Z;&|,       �	�H|kXc�A�1*

loss�w�;5���       �	O�|kXc�A�1*

loss���<��A#       �	�}}kXc�A�1*

loss8��:.�,        �	1&~kXc�A�1*

loss�X\<Y���       �	n�~kXc�A�1*

loss��9u	�       �	�ekXc�A�1*

lossC��:zzG       �	�w�kXc�A�1*

loss��S<�W�x       �	
�kXc�A�1*

loss��;Ml��       �	���kXc�A�1*

loss�_P=�1       �	4d�kXc�A�1*

loss:�<$OEK       �	? �kXc�A�1*

loss�y6=z���       �	b��kXc�A�1*

loss�Q8<0$��       �	���kXc�A�1*

lossnH;O���       �	NC�kXc�A�1*

loss�A]<Ճ�       �	?�kXc�A�1*

loss%]�;���       �	|�kXc�A�1*

loss�E�;(�!       �	��kXc�A�1*

lossT*>��>V       �	z��kXc�A�1*

lossA��<�T��       �	�m�kXc�A�1*

loss;�X;�q	w       �	��kXc�A�1*

loss�=�O�:       �	>��kXc�A�1*

lossŐ�<��I       �	ࢊkXc�A�1*

loss(>2=���X       �	�G�kXc�A�1*

loss�I�<r(�t       �	�݋kXc�A�1*

loss��<i�Nq       �	v�kXc�A�1*

loss��o<n|M�       �	e�kXc�A�1*

lossAr�<0�cf       �	%��kXc�A�1*

loss?�I;���       �	=G�kXc�A�1*

loss�pP;�_�n       �	DߎkXc�A�1*

loss*C};��K~       �	�v�kXc�A�1*

lossZ��;;�K�       �	��kXc�A�1*

loss��6<����       �	q��kXc�A�1*

losse�Z;z�
       �	Id�kXc�A�1*

loss�E�=i��L       �	2 �kXc�A�1*

loss�u;H��       �	���kXc�A�1*

loss^k;L�wZ       �	�>�kXc�A�1*

loss�I=N��n       �	�ؓkXc�A�1*

loss�~?=�!��       �	�q�kXc�A�1*

losss��<�t`�       �	��kXc�A�1*

loss�԰<Z��       �	5��kXc�A�1*

lossh`�<r��       �	�U�kXc�A�1*

loss��;Œ�       �	s�kXc�A�1*

loss��<�IF�       �	Փ�kXc�A�1*

loss�|<L�N�       �	�'�kXc�A�1*

lossi�c<�{�4       �	��kXc�A�1*

loss{<�B�       �	rN�kXc�A�1*

lossp�:�]YR       �	��kXc�A�1*

loss�m�;;�B�       �	�x�kXc�A�1*

loss�J�<ZSV       �	��kXc�A�1*

loss�C�<,�}       �	ҫ�kXc�A�1*

loss�<��       �	�D�kXc�A�1*

lossݻ!<O�"�       �	eߜkXc�A�1*

loss�x<��*�       �	z�kXc�A�1*

loss-�4;�f��       �	�kXc�A�1*

loss�p�;��pN       �	���kXc�A�1*

loss�
="7R�       �	m�kXc�A�1*

loss�A�;�*2       �	���kXc�A�1*

loss���:3޴�       �	�l�kXc�A�1*

lossؤa=��l(       �	�w�kXc�A�1*

loss�V<�ٻ�       �	�kXc�A�1*

lossf��<}�!       �	~V�kXc�A�1*

loss�z<�ϑ       �	
��kXc�A�1*

loss��<0{A       �	Z�kXc�A�1*

loss���:���       �	!�kXc�A�1*

lossU�;FQd�       �	��kXc�A�1*

loss�qD<$�       �	�9�kXc�A�1*

loss�*;����       �	ΨkXc�A�1*

loss�
�;4���       �	:��kXc�A�1*

loss֎><�}�       �	���kXc�A�1*

loss�}�:��zq       �	���kXc�A�1*

loss��p<}��       �	���kXc�A�1*

loss�Q�<#6��       �	�I�kXc�A�1*

loss��;	�C�       �	9F�kXc�A�2*

lossn��: b�       �	Y�kXc�A�2*

loss��:��׸       �	*ƯkXc�A�2*

loss��={a�       �	}�kXc�A�2*

lossJ��9��Q       �	�A�kXc�A�2*

loss�E�<C��       �	8�kXc�A�2*

loss�)�:�֛�       �	���kXc�A�2*

loss���;~F�       �	y��kXc�A�2*

loss��?<&й       �	Kx�kXc�A�2*

losstԝ;
��:       �	<L�kXc�A�2*

loss�p7=�v|       �	���kXc�A�2*

loss��=-z�`       �	k��kXc�A�2*

loss�B=ϼ�       �	��kXc�A�2*

loss3�A<3�m[       �	ޮ�kXc�A�2*

loss���<)���       �	
d�kXc�A�2*

lossM�x<0       �	�;�kXc�A�2*

lossfۚ<���       �	��kXc�A�2*

loss��:��J�       �	8��kXc�A�2*

lossg�;���       �	s.�kXc�A�2*

lossn�`<�)�e       �	�ükXc�A�2*

loss�,�;�g��       �	KY�kXc�A�2*

loss%(�;��)       �	���kXc�A�2*

loss�<�]�<       �	΋�kXc�A�2*

loss�c�:X|�"       �	%!�kXc�A�2*

loss��Z<I��       �	��kXc�A�2*

losscȹ<o��       �	uU�kXc�A�2*

loss1a�;��	�       �	���kXc�A�2*

loss��<3f��       �	��kXc�A�2*

loss���:��'       �	k'�kXc�A�2*

loss��;�B��       �	y��kXc�A�2*

loss[�0;<�i       �	�_�kXc�A�2*

lossp�;C1�       �	k��kXc�A�2*

losso3�;�r�R       �	Y��kXc�A�2*

loss�Ҕ:uA�[       �	A,�kXc�A�2*

loss�{<|o`�       �	%��kXc�A�2*

lossz�A<"�͒       �	�b�kXc�A�2*

lossʣE<�37       �	��kXc�A�2*

loss�-�<��W       �	R��kXc�A�2*

loss��j=��l       �	%?�kXc�A�2*

lossi�:+KO�       �	<��kXc�A�2*

lossڳ�:��C       �	��kXc�A�2*

loss
�=&���       �	:�kXc�A�2*

loss�nx<Z��?       �	ȵ�kXc�A�2*

loss:�y<L���       �	'P�kXc�A�2*

loss�Y�<��I       �	���kXc�A�2*

loss�m;,�m       �	�}�kXc�A�2*

lossqq�;c�E{       �	�kXc�A�2*

loss�d<t���       �	���kXc�A�2*

loss�@�<b>H�       �	B��kXc�A�2*

loss�Sy;�"�{       �	,.�kXc�A�2*

loss�%�<����       �	r��kXc�A�2*

lossA!={��T       �	�Y�kXc�A�2*

loss���;pZ�u       �	���kXc�A�2*

lossc7;H�       �	���kXc�A�2*

loss��<�N�       �	�(�kXc�A�2*

loss�<<<@�       �	���kXc�A�2*

lossH�=�R��       �	7S�kXc�A�2*

loss�d�;V�_       �	^�kXc�A�2*

loss� ?<���y       �	���kXc�A�2*

loss1�<��"�       �	=_�kXc�A�2*

lossOz�<�sF       �	���kXc�A�2*

loss��<x�޲       �	ӈ�kXc�A�2*

lossp#<lq��       �	X �kXc�A�2*

loss�#�9�,�D       �	��kXc�A�2*

loss=)�<K�a�       �	�I�kXc�A�2*

lossV�<���3       �	U��kXc�A�2*

lossZd�<�S�
       �	it�kXc�A�2*

loss��:9�B|       �	�	�kXc�A�2*

loss�PE;zd2N       �	Ӣ�kXc�A�2*

losseP]=gH�j       �	�W�kXc�A�2*

lossd0S;4�       �	2�kXc�A�2*

loss�a�<����       �	���kXc�A�2*

loss%<�٥       �	@�kXc�A�2*

lossFbY<��<�       �	R+�kXc�A�2*

loss�~>�/�       �	G��kXc�A�2*

loss�,Q;���       �	e��kXc�A�2*

loss�Z;:Y�2Q       �	&��kXc�A�2*

loss�;�s}       �	w��kXc�A�2*

loss��;�i�       �	���kXc�A�2*

loss^N;}bZ?       �	y�kXc�A�2*

loss+) =$&�(       �	z��kXc�A�2*

lossoE
=#Z�H       �	<O�kXc�A�2*

loss��U<�p�       �	��kXc�A�2*

loss���<e9�N       �	s0�kXc�A�2*

loss�!�=�-       �	m��kXc�A�2*

loss�=�8��       �	6u�kXc�A�2*

lossӗ�<��E�       �	x�kXc�A�2*

loss���;G�(�       �	t��kXc�A�2*

loss=�%o[       �	�P�kXc�A�2*

loss=\;v�@       �	'��kXc�A�2*

loss'%=�KU�       �	��kXc�A�2*

loss��<^O�       �	�/�kXc�A�2*

loss��O;ַj�       �	���kXc�A�2*

lossM�>;���;       �	�e�kXc�A�2*

loss�m;N��       �	�kXc�A�2*

lossi��:���W       �	��kXc�A�2*

loss��\;𚝥       �	�V�kXc�A�2*

loss��;��t       �	I��kXc�A�2*

loss�l�<NNy�       �	G��kXc�A�2*

losslS7<���       �	Z-�kXc�A�2*

lossZ�;�gW�       �	���kXc�A�2*

loss��=��nG       �	�b�kXc�A�2*

loss�; 9��       �	�kXc�A�2*

lossiv�;��H�       �	Q��kXc�A�2*

lossۣ<s>{�       �	�7�kXc�A�2*

lossڤ�<l5w�       �	��kXc�A�2*

lossvU<+9��       �	�m�kXc�A�2*

loss��:&p-�       �	2�kXc�A�2*

loss�j�;�u�       �	���kXc�A�2*

loss�<x\��       �	-�kXc�A�2*

loss��<�7��       �	?��kXc�A�2*

loss�qz;Zэ       �	[^�kXc�A�2*

loss��<ꑊ�       �	���kXc�A�2*

loss4��:u�g       �	v��kXc�A�2*

lossm�<iX       �	�(�kXc�A�2*

loss���;��k2       �	��kXc�A�2*

loss�%;�z�       �	�X�kXc�A�2*

loss�5=���       �	5��kXc�A�2*

loss��3;wV^�       �	_��kXc�A�2*

lossF��<�� �       �	*9�kXc�A�2*

loss���9:S�       �	���kXc�A�2*

loss�Ć;���       �	e�kXc�A�2*

loss�~�;}׭�       �	���kXc�A�2*

lossw�P<R)I       �	d� lXc�A�2*

loss��:�p       �	�9lXc�A�2*

lossZ=f=IO�       �	d�lXc�A�2*

losss�9��@5       �	x�lXc�A�2*

lossww<_
L	       �	-wlXc�A�2*

loss�%;r�)�       �	�mlXc�A�3*

loss���;13       �	}lXc�A�3*

loss�i;�@M       �	?�lXc�A�3*

lossj[�<3qh       �	KlXc�A�3*

losst�<���       �	��lXc�A�3*

lossZX�<��       �	.�lXc�A�3*

lossݴz92r�       �	R&	lXc�A�3*

loss&x�<N4h�       �	�	lXc�A�3*

lossCO�;:TGt       �	a
lXc�A�3*

lossH^�;՚9�       �	�lXc�A�3*

lossԑ<��&�       �	��lXc�A�3*

loss��7k�$�       �	�xlXc�A�3*

lossnW\<i�Y       �	�#lXc�A�3*

loss;�9 ѶN       �	J�lXc�A�3*

loss8��9ħ��       �	{lXc�A�3*

loss��9�X��       �	-#lXc�A�3*

loss�y�:[��
       �	��lXc�A�3*

loss�lZ<ն�R       �	ulXc�A�3*

lossv��<��       �	�lXc�A�3*

lossm�;�f4�       �	��lXc�A�3*

loss=<��~[       �	\YlXc�A�3*

loss�7=�s�-       �	��lXc�A�3*

lossfk�:O�J       �	�lXc�A�3*

loss,=O�D�       �	�5lXc�A�3*

loss�I?<�K       �	c�lXc�A�3*

loss�=;��]       �	�tlXc�A�3*

loss��<� s       �	�lXc�A�3*

loss���;s�_�       �	ݱlXc�A�3*

lossڍ@:r"^       �	"OlXc�A�3*

loss{iX<1���       �	��lXc�A�3*

loss݊A:�EK�       �	�lXc�A�3*

loss�s=�N�=       �	p%lXc�A�3*

loss���;����       �	n�lXc�A�3*

loss���<6��       �	�XlXc�A�3*

loss���<Jy�       �	��lXc�A�3*

lossm�U:�	*�       �	l�lXc�A�3*

lossJN�;V��W       �	�1lXc�A�3*

lossz$�;{�q       �	��lXc�A�3*

lossÄ:�w�       �	�lXc�A�3*

loss�;f�       �	�'lXc�A�3*

loss��!=Ƹl       �	��lXc�A�3*

lossã#=,%�N       �	עlXc�A�3*

loss�-;~��       �	�� lXc�A�3*

loss�~<D��,       �	t_!lXc�A�3*

lossj͍<G�y�       �	�"lXc�A�3*

loss��B;Y�K       �	��"lXc�A�3*

lossQd�;He;�       �	}w#lXc�A�3*

loss0;��7�       �	H$lXc�A�3*

loss�a�<���       �	_�$lXc�A�3*

loss���<3K�~       �	}�%lXc�A�3*

lossc��<ZTu�       �	\8&lXc�A�3*

loss�o�<�~7"       �	��&lXc�A�3*

lossW�;*\j�       �	̚'lXc�A�3*

loss��=:s��       �	y>(lXc�A�3*

loss{�<�"�       �	-�(lXc�A�3*

loss��%;�(@�       �	��)lXc�A�3*

loss�y{;Vb�       �	#-*lXc�A�3*

loss��V:�t       �	\�*lXc�A�3*

loss��\:]\8�       �	bf+lXc�A�3*

loss���;�\��       �	�,lXc�A�3*

loss���;�]�       �	��,lXc�A�3*

loss�;B���       �	�F-lXc�A�3*

lossM��9�o       �	��-lXc�A�3*

loss�o<���       �	�.lXc�A�3*

loss�B�<�[4�       �	�%/lXc�A�3*

loss]�8;��w�       �	��/lXc�A�3*

loss�n;�]�x       �	KY0lXc�A�3*

losst)<x�Th       �	��0lXc�A�3*

loss��<�<�       �	-�1lXc�A�3*

loss)��<F�Lv       �	:2lXc�A�3*

loss2z�<����       �	f�2lXc�A�3*

loss�U,9 �a       �	Kw3lXc�A�3*

loss��5;��4�       �	�:4lXc�A�3*

loss��<Yb�       �	a�KlXc�A�3*

loss0��<�ǃ#       �	
eLlXc�A�3*

loss�	 <+�       �	�LlXc�A�3*

loss�9k<I��       �	�NlXc�A�3*

lossoU=��=�       �	ϿNlXc�A�3*

lossj�<<�h�       �	�]OlXc�A�3*

loss��<���       �	7�OlXc�A�3*

loss��=G���       �	�PlXc�A�3*

loss���<�k�       �	�8QlXc�A�3*

loss�|�;�2�       �	��QlXc�A�3*

loss��0;���       �	�wRlXc�A�3*

loss6A�<��$       �	�SlXc�A�3*

loss���=!p��       �	J�SlXc�A�3*

lossfA=���       �	F\TlXc�A�3*

loss/S�;BU�       �	��TlXc�A�3*

loss�D1=,o��       �	ۢUlXc�A�3*

loss��R9��       �	�;VlXc�A�3*

lossx�[=ʥp       �	�VlXc�A�3*

lossZZ"<��%�       �	�gWlXc�A�3*

lossa�<��D�       �	�WlXc�A�3*

lossf�;��!�       �	�XlXc�A�3*

lossX�<�(?�       �	$+YlXc�A�3*

loss[��9��%\       �	��YlXc�A�3*

lossWu�<T��       �	rSZlXc�A�3*

loss�<��       �	��ZlXc�A�3*

loss���;��`�       �	[lXc�A�3*

loss���;��       �	0\lXc�A�3*

loss�%;�ym       �	d�\lXc�A�3*

loss�r�;�N�R       �	`]lXc�A�3*

losss�<�J�4       �	��]lXc�A�3*

loss&?�:��S�       �	Ύ^lXc�A�3*

loss��";&"       �	�,_lXc�A�3*

lossjۻ;1!R       �	�`lXc�A�3*

loss2�<�f#�       �	e�`lXc�A�3*

loss���9_E��       �	�blXc�A�3*

loss�r<Z��       �	�clXc�A�3*

lossťA;�\�'       �	��clXc�A�3*

lossh�/<��       �	�FdlXc�A�3*

lossR�<�x       �	��dlXc�A�3*

loss�x�<@��       �	elXc�A�3*

loss�g<����       �	flXc�A�3*

loss�kA:�˫	       �	x�flXc�A�3*

lossF�v<fd�       �	EGglXc�A�3*

loss�m�<�)�       �	��glXc�A�3*

loss���:�?ʌ       �	�phlXc�A�3*

loss�]�;�w�       �	1ilXc�A�3*

loss�u<����       �	�ilXc�A�3*

loss<.�<9��u       �	=jlXc�A�3*

loss�K=\�       �	��jlXc�A�3*

lossAN9�D�       �	nklXc�A�3*

loss�#�<Wz~       �	hllXc�A�3*

loss!�z;�ʗZ       �	ϠllXc�A�3*

loss!�:��i7       �	%;mlXc�A�3*

loss[�=�Y�s       �	/�mlXc�A�3*

loss�@�<ukD       �	�wnlXc�A�3*

lossb&�:<�#u       �	�"olXc�A�4*

loss��2:K$F       �	��olXc�A�4*

loss��9�w       �	�fplXc�A�4*

lossL"�;;7��       �	lqlXc�A�4*

loss�״=}ʥF       �	�qlXc�A�4*

loss�;�<��y�       �	�CrlXc�A�4*

loss)!&;�)       �	e�rlXc�A�4*

loss ��;O�       �	k�slXc�A�4*

loss�3�<GǸ�       �	|CtlXc�A�4*

loss�"<�ފ�       �	�tlXc�A�4*

loss�O;N���       �	�ulXc�A�4*

loss^�<
h       �	FvlXc�A�4*

loss�< ��       �	�vlXc�A�4*

loss��;=�PvD       �	��wlXc�A�4*

loss ��;���       �	:?xlXc�A�4*

loss�c;�uk�       �	Y�xlXc�A�4*

loss��3<ﴏ�       �	�vylXc�A�4*

loss�R�;��Ҫ       �	MzlXc�A�4*

lossk6�:6��       �	�zlXc�A�4*

lossj�<O�<       �	�F{lXc�A�4*

loss�R^<��S�       �	��{lXc�A�4*

lossR"�<"��>       �	��|lXc�A�4*

loss#�;jk�       �	�?}lXc�A�4*

losss7�;Y�W       �	1�}lXc�A�4*

lossolK<�[�       �	��~lXc�A�4*

loss �g<gFi       �	�lXc�A�4*

loss�R<��2       �	_�lXc�A�4*

loss�� <V�,       �	�L�lXc�A�4*

loss��9���       �	-�lXc�A�4*

loss�w�:��       �	���lXc�A�4*

loss<�m=x\�       �	�$�lXc�A�4*

loss��<G��)       �	ܷ�lXc�A�4*

loss��<����       �	�r�lXc�A�4*

loss��;�"\�       �	��lXc�A�4*

lossS�R;Jt��       �	gӅlXc�A�4*

lossƂ�:�y�       �	v�lXc�A�4*

loss�-=w?�y       �	��lXc�A�4*

lossW�;���       �	��lXc�A�4*

loss�^<=�0�       �	�A�lXc�A�4*

loss��*=��       �	U܈lXc�A�4*

loss.�;C�       �	�u�lXc�A�4*

lossM�<���       �	��lXc�A�4*

loss��;��f>       �	Ϥ�lXc�A�4*

loss�t�<���       �	?�lXc�A�4*

loss1��;��Υ       �	�׋lXc�A�4*

loss�W;����       �	�p�lXc�A�4*

loss|�9 0)�       �	p�lXc�A�4*

loss���:���       �	��lXc�A�4*

lossi�|<D�Y       �	@��lXc�A�4*

loss�T�<ء�a       �	c�lXc�A�4*

lossfW�<����       �	+��lXc�A�4*

loss#w�;���       �	���lXc�A�4*

lossCq:_���       �	�5�lXc�A�4*

loss��<���       �	�ԑlXc�A�4*

lossT�1;�[�r       �	�o�lXc�A�4*

lossn�=C:H�       �	N
�lXc�A�4*

loss�:<�/4*       �	���lXc�A�4*

lossm.:<�V��       �	AJ�lXc�A�4*

loss�F]:�i4�       �	��lXc�A�4*

loss�&�;ߧ       �	���lXc�A�4*

loss��90��G       �	y#�lXc�A�4*

loss��v<�o�)       �	]��lXc�A�4*

loss��I=1(�]       �	O[�lXc�A�4*

loss�_$<�E       �	���lXc�A�4*

lossh�;صAH       �	lXc�A�4*

loss��;C�C       �	�*�lXc�A�4*

loss=�_=y���       �	U��lXc�A�4*

lossf9a<o8��       �	�\�lXc�A�4*

loss�7\;_��{       �	���lXc�A�4*

loss�xU<�v8�       �	���lXc�A�4*

loss�ՙ<Ue�!       �	^/�lXc�A�4*

loss�<��h�       �	iŜlXc�A�4*

loss:Q�[�       �	�^�lXc�A�4*

loss��<�o��       �	��lXc�A�4*

loss�O�=:J,       �	Ȕ�lXc�A�4*

loss�<�)�       �	,*�lXc�A�4*

lossŸg<���       �	���lXc�A�4*

loss3��<���       �	OX�lXc�A�4*

loss��<1ր       �	N�lXc�A�4*

loss}pu<�?@       �	+��lXc�A�4*

loss�g=��v1       �	g�lXc�A�4*

loss�${;ӉJ�       �	2!�lXc�A�4*

loss�_#:a���       �	J]�lXc�A�4*

loss�Z'<�g�       �	��lXc�A�4*

loss,�<��       �	��lXc�A�4*

loss�:�)��       �	坦lXc�A�4*

loss�ȓ<����       �	�F�lXc�A�4*

lossi&8<ā��       �	-@�lXc�A�4*

loss��</��       �	���lXc�A�4*

loss�S�<��9,       �	�ЩlXc�A�4*

loss�;���A       �	u�lXc�A�4*

loss��
;�V�@       �	�ϫlXc�A�4*

loss(G/;5�NS       �	�x�lXc�A�4*

loss2�:t�"u       �	��lXc�A�4*

loss�P=_�7>       �	_��lXc�A�4*

loss��m;��       �	�I�lXc�A�4*

lossE,<���       �	��lXc�A�4*

loss�S�<sR�       �	L��lXc�A�4*

loss�/+=��a       �	��lXc�A�4*

loss�B<}�i9       �	ŰlXc�A�4*

loss�QP:i�T       �	�m�lXc�A�4*

loss��;.Ri=       �	��lXc�A�4*

loss`V<5���       �	~��lXc�A�4*

lossp�<��(       �	�F�lXc�A�4*

losso��:<��       �	��lXc�A�4*

lossd��;
�	R       �	j��lXc�A�4*

loss�5�;)+,�       �	 �lXc�A�4*

loss�%=�Y(�       �	p��lXc�A�4*

loss
��<H<�       �	�I�lXc�A�4*

loss��.<A�?       �	:�lXc�A�4*

loss83'<
;W�       �	A��lXc�A�4*

loss_�8:`{�       �	6X�lXc�A�4*

lossw=W=� `       �	���lXc�A�4*

loss�]M:S��w       �	Ș�lXc�A�4*

lossTR<��}�       �	6�lXc�A�4*

loss@�= �l       �	׺lXc�A�4*

lossh�:	G�o       �	ds�lXc�A�4*

lossO�e;)U       �	��lXc�A�4*

lossI�=<h��|       �	��lXc�A�4*

lossO|�9Y]�       �	�J�lXc�A�4*

loss��P<@�       �	=�lXc�A�4*

lossjw�<�#�       �	���lXc�A�4*

loss���;�$�       �	z4�lXc�A�4*

loss�}<��n�       �	�пlXc�A�4*

loss���;|=`�       �	�o�lXc�A�4*

loss��:���       �	��lXc�A�4*

lossw��<T���       �	���lXc�A�4*

loss���:��_�       �	v�lXc�A�4*

lossrK�:0�I       �	0�lXc�A�5*

loss�P�;�I�}       �	��lXc�A�5*

loss)Z:�/`�       �	uZ�lXc�A�5*

loss�D=͕Tc       �	���lXc�A�5*

loss��<��O       �	\��lXc�A�5*

loss�u;����       �	8�lXc�A�5*

loss��0<���       �	M��lXc�A�5*

loss�T	=O��       �	Pr�lXc�A�5*

loss���;�o��       �	+�lXc�A�5*

loss`n;U�X�       �	��lXc�A�5*

loss��w;
�N       �	�Q�lXc�A�5*

loss�m�9�f       �	�"�lXc�A�5*

lossW�O<�q�&       �	4��lXc�A�5*

lossctI=�={       �	|_�lXc�A�5*

lossR��<�m�       �	�lXc�A�5*

loss��<{c       �	���lXc�A�5*

loss^��<�#/�       �	�^�lXc�A�5*

losso�=�?�}       �	�lXc�A�5*

loss�Ч;F�       �	D��lXc�A�5*

loss��g<��Gb       �	�C�lXc�A�5*

loss���:����       �	��lXc�A�5*

loss��;�~Z"       �	��lXc�A�5*

loss��<)���       �	�,�lXc�A�5*

loss��<�`��       �	���lXc�A�5*

loss���<v�}�       �	=c�lXc�A�5*

loss6�r;�{Ϗ       �	.��lXc�A�5*

loss�A�;�OI$       �	ǜ�lXc�A�5*

loss?W�<���       �	L6�lXc�A�5*

loss$J
<'͹Q       �	>��lXc�A�5*

losso;���       �	Hp�lXc�A�5*

loss{�;�C:.       �	��lXc�A�5*

loss}(#=��       �	���lXc�A�5*

loss� )=]��       �	0G�lXc�A�5*

loss"��<	j#       �	>��lXc�A�5*

lossz��<����       �	�~�lXc�A�5*

loss8r=���       �	� �lXc�A�5*

loss�wl<���       �	Z��lXc�A�5*

loss��<��f       �	[�lXc�A�5*

lossQ�S;��)       �	��lXc�A�5*

lossHi�<o��H       �	D��lXc�A�5*

loss��;��eR       �	�G�lXc�A�5*

loss��;��²       �	���lXc�A�5*

loss���<�Y��       �	ˁ�lXc�A�5*

loss]hT<s���       �	��lXc�A�5*

loss�)�;&%�       �	k*�lXc�A�5*

loss�
<2��       �	|��lXc�A�5*

loss��<���A       �	f��lXc�A�5*

loss���;�o%x       �	�7�lXc�A�5*

loss�|�:/?�|       �	���lXc�A�5*

loss���;t�-�       �	
g�lXc�A�5*

loss��=;�ڟq       �	��lXc�A�5*

lossVl6<~��Q       �	���lXc�A�5*

loss!�<*�c       �	!Y�lXc�A�5*

loss��;:�2j       �	Z��lXc�A�5*

lossd��;��       �	F��lXc�A�5*

loss�*=3���       �	8/�lXc�A�5*

loss�EB9�,�t       �	.��lXc�A�5*

loss���<��_       �	�_�lXc�A�5*

loss���;��Ā       �	���lXc�A�5*

loss�~"<ẋ�       �	���lXc�A�5*

lossY�;<�֘       �	� �lXc�A�5*

loss/�<Zj�6       �	��lXc�A�5*

lossi��;��>       �	�M�lXc�A�5*

loss���;We       �	���lXc�A�5*

lossO�2=f�w       �	��lXc�A�5*

loss�͞;!��Y       �	�a�lXc�A�5*

lossVn�;2&�F       �	��lXc�A�5*

loss�~�<���       �	��lXc�A�5*

losse5<ey"�       �	���lXc�A�5*

loss��<�X�       �	'J�lXc�A�5*

lossc�)<�@EA       �	5��lXc�A�5*

loss/A�9��_�       �	;��lXc�A�5*

loss4�<�e�       �	�1�lXc�A�5*

loss���:��W?       �	���lXc�A�5*

loss�^�<�c�       �	��lXc�A�5*

loss���=ݱ|p       �	N��lXc�A�5*

loss]�;�8�!       �	�(�lXc�A�5*

loss��<R-�H       �	���lXc�A�5*

loss|d�;�%�K       �	6s�lXc�A�5*

loss��;����       �		�lXc�A�5*

loss��@=�x �       �	���lXc�A�5*

loss�<��}1       �	R�lXc�A�5*

lossW��:�v-c       �	Q��lXc�A�5*

losszF<<Gv       �	���lXc�A�5*

loss�Wp;���       �	(H�lXc�A�5*

loss��+<<�+P       �	���lXc�A�5*

lossH�=��/       �	|��lXc�A�5*

loss��O<q�#*       �	�U�lXc�A�5*

loss�h<Sy��       �	���lXc�A�5*

loss���;R�0�       �	���lXc�A�5*

loss2{�;���       �	�e mXc�A�5*

loss��=e��       �	�mXc�A�5*

loss̒<J�@       �	�mXc�A�5*

loss1�<pO��       �	bmXc�A�5*

lossI�<G��       �	rNmXc�A�5*

lossϳ[<Ǯ��       �	`�mXc�A�5*

loss\ۅ<�&X�       �	�zmXc�A�5*

lossEА:bž�       �	�+mXc�A�5*

loss���<�%��       �	��mXc�A�5*

loss6A<<��       �	�mXc�A�5*

loss��5=J
�       �	)!mXc�A�5*

loss��<�"q       �	GmXc�A�5*

lossj�~<�!�       �	0�mXc�A�5*

loss�m<�dJ�       �	6Z	mXc�A�5*

loss႓<*G�       �	a�	mXc�A�5*

loss:��;M�	�       �	x�
mXc�A�5*

loss���:�-ǐ       �	�]mXc�A�5*

loss˳:��x�       �	�LmXc�A�5*

loss3/�<�-�       �	��mXc�A�5*

lossQ:��)       �	&�mXc�A�5*

loss��y9�,2       �	VFmXc�A�5*

loss���<0kN       �	�mXc�A�5*

loss��<���       �	)�mXc�A�5*

loss�G<��       �	�gmXc�A�5*

lossb��=���       �	�mXc�A�5*

loss�׍<       �	��mXc�A�5*

loss�%�<D�Z�       �	1�mXc�A�5*

loss���;5���       �	w+mXc�A�5*

loss�Eq:M�b�       �	4�mXc�A�5*

lossӇ�9�iO�       �		OmXc�A�5*

loss�1^<�Y�       �	��mXc�A�5*

loss Ԙ<:�f       �	j�mXc�A�5*

lossۤ�<w���       �	y"mXc�A�5*

lossv��9���       �	o�mXc�A�5*

losss��;�Z�L       �	 UmXc�A�5*

lossp�;;�0�       �	��mXc�A�5*

loss#G�:�B�}       �	ӆmXc�A�5*

loss��;0y�1       �	�mXc�A�5*

loss��<Hu�       �	��mXc�A�6*

loss���<��       �	�JmXc�A�6*

loss��:�
E]       �	��mXc�A�6*

loss�=��       �	�rmXc�A�6*

lossĂ�<h�k�       �	JmXc�A�6*

loss3�<�B�"       �	��mXc�A�6*

loss��E<3��j       �	�:mXc�A�6*

loss��<
�j�       �	-�mXc�A�6*

loss�0;�]Zn       �	^imXc�A�6*

loss�0�<�X�       �	?�mXc�A�6*

loss^=X���       �	6�mXc�A�6*

loss�z-<
�C�       �	3 mXc�A�6*

loss;�<�kag       �	� mXc�A�6*

losss��;����       �	�!mXc�A�6*

loss�W<��h       �	�S"mXc�A�6*

lossI�$<�pd�       �	:#mXc�A�6*

loss�V =���       �	�#mXc�A�6*

loss:�"<�0F       �	$�$mXc�A�6*

loss6-�<��L       �	|�%mXc�A�6*

loss4��;OK�       �	�&mXc�A�6*

loss_��<K��       �	0'mXc�A�6*

loss��9����       �	�(mXc�A�6*

loss��W<�H|       �	�(mXc�A�6*

loss�BE<�X�       �	�O+mXc�A�6*

lossR٠:�)-T       �	��+mXc�A�6*

loss��;v�)p       �	��,mXc�A�6*

loss���<�8��       �	^J-mXc�A�6*

loss&�=����       �	a�-mXc�A�6*

loss��i;^�       �	��.mXc�A�6*

loss�!�9�/(|       �	uZ/mXc�A�6*

lossC��;��i       �	K0mXc�A�6*

losssه<���       �	 �0mXc�A�6*

loss�o=޸^�       �	LR1mXc�A�6*

lossE��=�谧       �	��1mXc�A�6*

loss���<i̓x       �	f�2mXc�A�6*

losszE�<:�aC       �	E3mXc�A�6*

loss|`=_��j       �	B�3mXc�A�6*

loss�(�<��%       �	��4mXc�A�6*

loss��?;�%�1       �	�15mXc�A�6*

loss��;����       �	��5mXc�A�6*

loss3,(<��If       �	�}6mXc�A�6*

loss��<;�G�       �	�7mXc�A�6*

loss{��;dL#       �	9�7mXc�A�6*

loss->=��	�       �	�X8mXc�A�6*

loss�P�;�]       �	��8mXc�A�6*

loss��=�Q��       �	��9mXc�A�6*

loss0�;�!H�       �	�9:mXc�A�6*

loss�/�=�N��       �	��:mXc�A�6*

loss-~�;cﴪ       �	�y;mXc�A�6*

loss_	H;o\u�       �	8<mXc�A�6*

loss���<���       �	h�<mXc�A�6*

loss��I=I�|       �	e=mXc�A�6*

lossJ0[=���0       �	�>mXc�A�6*

loss��<��v�       �	��>mXc�A�6*

lossh^;?���       �	L4?mXc�A�6*

loss-`�<�˝�       �	j�?mXc�A�6*

loss��;��s       �	�s@mXc�A�6*

loss��;��x�       �		AmXc�A�6*

loss��;TtD�       �	��AmXc�A�6*

lossSFs;�7��       �	_ABmXc�A�6*

loss�:چ$�       �	�BmXc�A�6*

loss֦�=���       �	zCmXc�A�6*

loss�:jC9F       �	�DmXc�A�6*

loss��;��       �	��DmXc�A�6*

loss�	�;b8�m       �	\EmXc�A�6*

loss���<��4       �	�EmXc�A�6*

lossm�<4�!       �	v�FmXc�A�6*

loss7�y<�~�V       �	�#GmXc�A�6*

lossRh
<�x       �	�GmXc�A�6*

loss���:༿�       �	�EHmXc�A�6*

lossRk�;�[��       �	��HmXc�A�6*

losslD[<�M       �	lImXc�A�6*

lossBn�;Z       �	�JmXc�A�6*

loss���</�e       �	.�JmXc�A�6*

loss�K!=���       �	lAKmXc�A�6*

loss��=�q�<       �	k�KmXc�A�6*

loss���:2r\       �	�kLmXc�A�6*

loss���;���       �	i�LmXc�A�6*

loss�\_;�4�       �	��MmXc�A�6*

loss�=�;�I�p       �	V)NmXc�A�6*

lossf�5;5��       �	�NmXc�A�6*

loss.�;��G       �	UOmXc�A�6*

loss��b:O�E       �	$�OmXc�A�6*

loss�e5;A���       �	��PmXc�A�6*

loss�^<�F}�       �	�QmXc�A�6*

lossCp8�L�W       �	�QmXc�A�6*

loss��;�{       �	BRmXc�A�6*

lossw�4;       �	t�RmXc�A�6*

loss۟!;���       �	QgSmXc�A�6*

loss|a�:%wq       �	<TmXc�A�6*

loss@zs:� w       �	�UmXc�A�6*

loss8��<Rd`Y       �	��UmXc�A�6*

loss�/:s���       �	/2VmXc�A�6*

loss��<��A       �	��VmXc�A�6*

lossIt-;�`�       �	aWmXc�A�6*

loss� ;���?       �	��WmXc�A�6*

lossۭf;�\�5       �	�YmXc�A�6*

loss�_<F��&       �	�YmXc�A�6*

loss���<W�4       �	�GZmXc�A�6*

lossd5k<���       �	��ZmXc�A�6*

loss!?=b��v       �	v[mXc�A�6*

loss��.<Q
8e       �	�\mXc�A�6*

loss�[�<&+       �	��\mXc�A�6*

loss��;m�*�       �	�T]mXc�A�6*

loss=c�;G	�b       �	:�]mXc�A�6*

loss�9t<���]       �	�z^mXc�A�6*

loss��X;�o(       �	I_mXc�A�6*

loss�ϻ;k6u       �	��_mXc�A�6*

loss��; �c�       �	;`mXc�A�6*

loss��<����       �	d�`mXc�A�6*

lossIAg;i�l�       �	eamXc�A�6*

loss�k�;��ܪ       �	bmXc�A�6*

loss�9L<;�V"       �	ҧbmXc�A�6*

lossR-�<�y��       �	�KcmXc�A�6*

loss[^=�m       �	)$dmXc�A�6*

loss1a�=\EO0       �	|�dmXc�A�6*

loss�-<<?|!       �	�[emXc�A�6*

loss��;)��/       �	/MfmXc�A�6*

loss�d�<̀��       �	�]gmXc�A�6*

loss���;�1�g       �	"8hmXc�A�6*

losso�;aPk"       �	cimXc�A�6*

loss�S�;tج       �	�jmXc�A�6*

loss���<_n��       �	�jmXc�A�6*

loss�|<��k       �	z�kmXc�A�6*

lossa�;�J�_       �	B@lmXc�A�6*

loss��;�%R$       �	mmXc�A�6*

loss�0=��       �	¾mmXc�A�6*

loss��N<8&�       �	?�nmXc�A�6*

losstw�;G� �       �	�GomXc�A�7*

lossa>i>_�       �	3pmXc�A�7*

lossM'Q<S��F       �	�pmXc�A�7*

loss)�=�÷       �	�PqmXc�A�7*

loss�J<R�Q       �	rmXc�A�7*

loss�<��B       �	��rmXc�A�7*

loss��"<���       �	��smXc�A�7*

loss�ܤ<	$L       �	o/tmXc�A�7*

lossfX�;e���       �	��tmXc�A�7*

loss�e;�N��       �	�\umXc�A�7*

loss��;[��       �	�vmXc�A�7*

loss�d=B�/Q       �	X�vmXc�A�7*

loss&�:��{5       �	MNwmXc�A�7*

lossO	;��       �	q�wmXc�A�7*

loss��<&JK_       �	�xmXc�A�7*

losss)�<�w��       �	:�ymXc�A�7*

loss��<<�8       �	�+zmXc�A�7*

loss��;��g6       �	��zmXc�A�7*

loss�Hh<�A:4       �	�{{mXc�A�7*

loss��w<  �C       �	�|mXc�A�7*

loss���<p��       �	ѱ|mXc�A�7*

loss�WD;�mvV       �	VI}mXc�A�7*

loss��c9�k`       �	�L~mXc�A�7*

loss�bT<��F2       �	:�~mXc�A�7*

loss�L�<ZJY       �	��mXc�A�7*

loss���;"�r~       �	�"�mXc�A�7*

loss��=���       �	A��mXc�A�7*

lossUَ:��~       �	L�mXc�A�7*

losst3:�_3w       �	��mXc�A�7*

loss�r�;8���       �	Y��mXc�A�7*

loss���;<��       �	>"�mXc�A�7*

loss�<��:�       �	��mXc�A�7*

losst��;Sߞ�       �	⭄mXc�A�7*

loss�?N<��=/       �	J�mXc�A�7*

loss��=����       �	H݅mXc�A�7*

loss_��:\�v�       �	t��mXc�A�7*

loss�/�<*��       �	�5�mXc�A�7*

loss�N
;lb|�       �	*�mXc�A�7*

loss J�<wS��       �	}x�mXc�A�7*

loss1�n:���       �	,�mXc�A�7*

loss\�"=��a�       �	��mXc�A�7*

loss
P�<aQI4       �	>=�mXc�A�7*

loss�,;�n�;       �	�׊mXc�A�7*

lossMT�:�`�       �	fl�mXc�A�7*

loss��;#�ja       �	&�mXc�A�7*

lossF�q<���       �	�mXc�A�7*

loss�dH<*[j       �	)�mXc�A�7*

loss�-<2h�       �	8��mXc�A�7*

loss2W�;��E�       �	�S�mXc�A�7*

loss�S�:���9       �	t�mXc�A�7*

loss? O<aU��       �	Q��mXc�A�7*

loss�	a<7�2       �	q�mXc�A�7*

loss�j�;���       �	�ҐmXc�A�7*

loss2ji;�H�^       �	s�mXc�A�7*

lossr>�;[9��       �	� �mXc�A�7*

loss�{�;?���       �	̒mXc�A�7*

loss�c=�+�       �	�x�mXc�A�7*

loss(�<�Ccl       �	� �mXc�A�7*

loss�X�<��`$       �	4��mXc�A�7*

loss7��;�v�P       �	
h�mXc�A�7*

loss#ˑ;�=�       �	�
�mXc�A�7*

loss[er<�[P�       �	ĖmXc�A�7*

loss�:���       �	_y�mXc�A�7*

lossͺ�<!���       �	�mXc�A�7*

loss��;dtk�       �	�ʘmXc�A�7*

lossx��9�a]�       �	j�mXc�A�7*

loss��;.��+       �	R
�mXc�A�7*

lossŏ%;���       �	H��mXc�A�7*

loss���:�{3�       �	�>�mXc�A�7*

loss��;��5       �	�ٛmXc�A�7*

loss��<�I��       �	}v�mXc�A�7*

lossM�,;@�	       �	��mXc�A�7*

loss�>O=jUac       �	|��mXc�A�7*

lossΛ#:ךo       �	�a�mXc�A�7*

loss�ԭ;k���       �	��mXc�A�7*

loss�T;�4"
       �	���mXc�A�7*

loss���;wB�       �	v7�mXc�A�7*

loss��=�֛       �	�#�mXc�A�7*

loss�0;w.QZ       �	j��mXc�A�7*

lossiS�;�kj       �	m7�mXc�A�7*

loss��;�\3#       �	٣mXc�A�7*

losss(�:���G       �	��mXc�A�7*

loss�#=���       �	�mXc�A�7*

loss��z;�wA�       �	�O�mXc�A�7*

loss���;Q�"I       �	  �mXc�A�7*

lossע�<Zԣ       �	���mXc�A�7*

loss7a�;:�
�       �	�J�mXc�A�7*

loss�<����       �	���mXc�A�7*

loss��:�rC0       �	`��mXc�A�7*

loss��2<^��       �	'�mXc�A�7*

lossD�;��ѩ       �	�~�mXc�A�7*

loss���;@]%�       �	��mXc�A�7*

loss�CH<�s�       �	�ˬmXc�A�7*

loss���9q1�       �	i�mXc�A�7*

loss���9�A��       �	I�mXc�A�7*

loss)�9F��       �	-��mXc�A�7*

loss7-�;49t       �	�W�mXc�A�7*

loss�{�<���G       �	��mXc�A�7*

loss<�<�        �	��mXc�A�7*

lossMS�;�c%       �	�"�mXc�A�7*

loss��<KE*       �	��mXc�A�7*

loss���9�Պ�       �	�Q�mXc�A�7*

loss8; "�{       �	-�mXc�A�7*

loss�Z�7���       �	0�mXc�A�7*

loss<�K}       �	�P�mXc�A�7*

loss!�<ﶰg       �	�mXc�A�7*

loss{F4<D       �	��mXc�A�7*

loss�X�;�@�@       �	x_�mXc�A�7*

lossJ��9���       �	J�mXc�A�7*

loss�]C=�\z       �	젷mXc�A�7*

loss1��9@�u       �	�8�mXc�A�7*

loss�Ę<S"�       �	ӸmXc�A�7*

loss�J<��       �	
i�mXc�A�7*

lossM^=)�mS       �	��mXc�A�7*

loss�r;�^3�       �	^��mXc�A�7*

lossn�;4�l       �	�2�mXc�A�7*

lossN�;e�wR       �	SлmXc�A�7*

loss�bm;�%�       �	�h�mXc�A�7*

loss��;?��       �	 �mXc�A�7*

loss?	O:>���       �	��mXc�A�7*

loss���;�֖       �	�Y�mXc�A�7*

loss�z<G|�       �	h�mXc�A�7*

loss��<W�O       �	즿mXc�A�7*

losso��;�R       �	$D�mXc�A�7*

loss�a�<�*�       �	.��mXc�A�7*

lossJXM<K�       �	���mXc�A�7*

loss\1�:���v       �	�0�mXc�A�7*

loss�_�<���B       �	��mXc�A�7*

loss��x<�w.&       �	yw�mXc�A�8*

loss�eG;��<       �	��mXc�A�8*

loss('�:�{��       �	��mXc�A�8*

loss�=�	)       �	�i�mXc�A�8*

loss���;��C�       �	��mXc�A�8*

loss��;�� O       �	L��mXc�A�8*

lossX=�<K��       �	Ǡ�mXc�A�8*

loss���;�>��       �	�H�mXc�A�8*

loss�cg;I�փ       �	���mXc�A�8*

loss��;\���       �	�9�mXc�A�8*

loss��/=v�c�       �	x�mXc�A�8*

loss�ܴ<�!�       �	��mXc�A�8*

loss:�:r�N       �	.��mXc�A�8*

lossƞ=���I       �	${�mXc�A�8*

losslyY;M��c       �	�)�mXc�A�8*

loss���;���       �	;��mXc�A�8*

loss��*;�T,�       �	�j�mXc�A�8*

loss��@:���6       �	��mXc�A�8*

loss���:2T�       �	���mXc�A�8*

loss�ž<e}�Y       �	!:�mXc�A�8*

loss��;1��       �	c��mXc�A�8*

loss���<T�5+       �	nk�mXc�A�8*

loss�i9;��)I       �	��mXc�A�8*

loss��	=c�u�       �	}��mXc�A�8*

loss�6;��'       �	�r�mXc�A�8*

loss��<0\u       �	�mXc�A�8*

loss��	<Y�Z       �	���mXc�A�8*

loss8��;�Z�       �	)Y�mXc�A�8*

loss�<v���       �	��mXc�A�8*

losso�#<����       �	��mXc�A�8*

loss�=��L       �	~:�mXc�A�8*

losse�;���       �	���mXc�A�8*

loss��9#�E       �	)u�mXc�A�8*

loss�<���