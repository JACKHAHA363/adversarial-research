       �K"	  ��Xc�Abrain.Event:2��M6�     �	�КXc�A"��
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
dtype0*
shape: */
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
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
	container *
dtype0*
shared_name *
shape:@
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
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
dtype0*
seed���)*
T0*&
_output_shapes
:@@*
seed2�ɡ
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
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
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
	container *
dtype0*
shared_name *
shape:@
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
conv2d_2/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   
s
"conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*/
_output_shapes
:���������@*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0
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
 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  @?
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
seed2��
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
dropout_1/cond/MergeMergedropout_1/cond/Switch_1dropout_1/cond/dropout/mul*1
_output_shapes
:���������@: *
N*
T0
c
flatten_1/ShapeShapedropout_1/cond/Merge*
out_type0*
_output_shapes
:*
T0
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
i
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
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
new_axis_mask *
ellipsis_mask *

begin_mask *
shrink_axis_mask *
T0*
Index0
Y
flatten_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
\
flatten_1/stack/0Const*
_output_shapes
: *
dtype0*
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
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
dense_1/random_uniform/minConst*
valueB
 *�3z�*
_output_shapes
: *
dtype0
_
dense_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*!
_output_shapes
:���*
seed2���*
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*!
_output_shapes
:���*
T0
�
dense_1/kernel
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
shape:���
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
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
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
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
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
data_formatNHWC*
T0
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
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
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
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*
seed���)*
T0*
dtype0*(
_output_shapes
:����������*
seed2���
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
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
�
dropout_2/cond/MergeMergedropout_2/cond/Switch_1dropout_2/cond/dropout/mul*
N*
T0**
_output_shapes
:����������: 
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
dtype0*
seed���)*
T0*
_output_shapes
:	�
*
seed2㔚
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
dense_2/ConstConst*
_output_shapes
:
*
dtype0*
valueB
*    
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
'sequential_1/conv2d_1/convolution/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
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
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
�
'sequential_1/conv2d_2/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
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
&sequential_1/dropout_1/cond/mul/SwitchSwitchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
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
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
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
seed2ڊ�
�
6sequential_1/dropout_1/cond/dropout/random_uniform/subSub6sequential_1/dropout_1/cond/dropout/random_uniform/max6sequential_1/dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
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
N*
T0
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
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
,sequential_1/flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
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
sequential_1/flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
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
sequential_1/flatten_1/ReshapeReshape!sequential_1/dropout_1/cond/Mergesequential_1/flatten_1/stack*0
_output_shapes
:������������������*
Tshape0*
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
)sequential_1/dropout_2/cond/dropout/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
T0
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
@sequential_1/dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_2/cond/dropout/Shape*
dtype0*
seed���)*
T0*(
_output_shapes
:����������*
seed2�ӂ
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
*
T0*
data_formatNHWC
b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

[
num_inst/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_correct
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
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
X
SumSumToFloatConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
z
	AssignAdd	AssignAddnum_instConst_1*
_output_shapes
: *
_class
loc:@num_inst*
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
 *    *
dtype0*
_output_shapes
: 
�
Assign_1Assignnum_correctConst_3*
_output_shapes
: *
validate_shape(*
_class
loc:@num_correct*
T0*
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
softmax_cross_entropy_loss/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
out_type0*
_output_shapes
:*
T0
c
!softmax_cross_entropy_loss/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
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
%softmax_cross_entropy_loss/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
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
N*
T0*

Tidx0
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
'softmax_cross_entropy_loss/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_2(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
Index0*
T0*
_output_shapes
:
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
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
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
valueB: *
_output_shapes
:*
dtype0
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
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
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
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
out_type0*
T0
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
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
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
dtype0*
shape: *
_output_shapes
:
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
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
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
�
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
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
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
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
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
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
T0*
_output_shapes
: *
Tshape0
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
T0*#
_output_shapes
:���������*
Tshape0
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
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0
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
7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapediv_1*
_output_shapes
:*
out_type0*
T0
�
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
v
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
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
T0*
_output_shapes
:	�
*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1
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
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
N*
T0
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
@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Shape_1*
T0*(
_output_shapes
:����������*
Tshape0
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
T0*
_output_shapes
:*
out_type0
�
>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
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
6gradients/sequential_1/dropout_2/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_2/cond/mul_grad/Sum4gradients/sequential_1/dropout_2/cond/mul_grad/Shape*(
_output_shapes
:����������*
Tshape0*
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
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape_1
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
6gradients/sequential_1/activation_3/Relu_grad/ReluGradReluGradgradients/AddNsequential_1/activation_3/Relu*(
_output_shapes
:����������*
T0
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
T0*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
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
Kgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_1/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
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
T0*
_output_shapes
:*
out_type0
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
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape
�
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
T0*
_output_shapes
:*
out_type0
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
6gradients/sequential_1/dropout_1/cond/mul_grad/ReshapeReshape2gradients/sequential_1/dropout_1/cond/mul_grad/Sum4gradients/sequential_1/dropout_1/cond/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape
�
Igradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependency_1Identity8gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape_1
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
N*
T0
�
gradients/AddN_1AddN=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_grad*/
_output_shapes
:���������@*
N*P
_classF
DBloc:@gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_grad*
T0
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
T0*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad
�
Ggradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
6gradients/sequential_1/conv2d_2/convolution_grad/ShapeShapesequential_1/activation_1/Relu*
T0*
_output_shapes
:*
out_type0
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
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"      @   @   
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
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput*
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
8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
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
Dgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_1/convolution_grad/Shapeconv2d_1/kernel/readEgradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
strides
*
T0*
paddingVALID
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
T0*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput
�
Kgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1IdentityEgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilterB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*X
_classN
LJloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@*
T0
�
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?*"
_class
loc:@conv2d_1/kernel
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
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
n
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
j
zerosConst*
dtype0*&
_output_shapes
:@*%
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
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1zeros_1*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
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
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adamzeros_2*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
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
loc:@conv2d_2/kernel*
shared_name *&
_output_shapes
:@@*
shape:@@
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
conv2d_2/kernel/Adam/readIdentityconv2d_2/kernel/Adam*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
l
zeros_5Const*&
_output_shapes
:@@*
dtype0*%
valueB@@*    
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
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel
b
zeros_9Const*!
_output_shapes
:���*
dtype0* 
valueB���*    
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
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_output_shapes	
:�*
_class
loc:@dense_1/bias*
T0
W
zeros_11Const*
_output_shapes	
:�*
dtype0*
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
_output_shapes
:	�
*
dtype0*
valueB	�
*    
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
T0*
_output_shapes
:	�
*!
_class
loc:@dense_2/kernel
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
 *w�+2*
dtype0*
_output_shapes
: 
�
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonKgradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependency_1*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
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
Adam/beta2Adam/epsilonGgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
T0*
use_locking( 
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
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
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
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
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
N"D�����     >�	�ҚXc�AJ��
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
dtype0*
shape: */
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
dtype0*
shape: 
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
`
conv2d_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�x�
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
seed2���*
dtype0*
T0*
seed���)
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
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
[
conv2d_1/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
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
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2�ɡ
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
VariableV2*
shared_name *
dtype0*
shape:@@*&
_output_shapes
:@@*
	container 
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
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
[
conv2d_2/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*    
y
conv2d_2/bias
VariableV2*
_output_shapes
:@*
	container *
dtype0*
shared_name *
shape:@
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
conv2d_2/convolutionConv2Dactivation_1/Reluconv2d_2/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
data_formatNHWC*
strides
*
T0*
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
dropout_1/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_1/cond/mul/yConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
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
T0*
_output_shapes
:*
out_type0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
seed���)*
T0*
dtype0*/
_output_shapes
:���������@*
seed2��
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
dense_1/random_uniform/maxConst*
valueB
 *�3z<*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed���)*
T0*!
_output_shapes
:���*
seed2���
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
VariableV2*
shared_name *
dtype0*
shape:���*!
_output_shapes
:���*
	container 
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
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
shape:�
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
dropout_2/cond/mul/SwitchSwitchactivation_3/Reludropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu
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
dropout_2/cond/dropout/ShapeShapedropout_2/cond/mul*
T0*
_output_shapes
:*
out_type0
�
)dropout_2/cond/dropout/random_uniform/minConst^dropout_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
�
)dropout_2/cond/dropout/random_uniform/maxConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2���*
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
dropout_2/cond/Switch_1Switchactivation_3/Reludropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*$
_class
loc:@activation_3/Relu*
T0
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
dense_2/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U�
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
seed2㔚*
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
dense_2/ConstConst*
_output_shapes
:
*
dtype0*
valueB
*    
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
dense_2/bias/readIdentitydense_2/bias*
_output_shapes
:
*
_class
loc:@dense_2/bias*
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������
*
data_formatNHWC*
T0
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
sequential_1/conv2d_1/BiasAddBiasAdd!sequential_1/conv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@

sequential_1/activation_1/ReluRelusequential_1/conv2d_1/BiasAdd*/
_output_shapes
:���������@*
T0
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
!sequential_1/conv2d_2/convolutionConv2Dsequential_1/activation_1/Reluconv2d_2/kernel/read*
paddingVALID*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
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
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
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
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*/
_output_shapes
:���������@*
seed2ڊ�*
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
*sequential_1/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
v
,sequential_1/flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
6sequential_1/dropout_2/cond/dropout/random_uniform/minConst%^sequential_1/dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
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
seed2�ӂ*
dtype0*
T0*
seed���)
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
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu*
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
sequential_1/dense_2/BiasAddBiasAddsequential_1/dense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������

b
SoftmaxSoftmaxsequential_1/dense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
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
ConstConst*
_output_shapes
:*
dtype0*
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
T0*
_output_shapes
: *
_class
loc:@num_correct
L
Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
�
AssignAssignnum_instConst_2*
_output_shapes
: *
validate_shape(*
_class
loc:@num_inst*
T0*
use_locking(
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *    
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
add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���.
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
div_1RealDivsequential_1/dense_2/BiasAdddiv_1/y*
T0*'
_output_shapes
:���������

a
softmax_cross_entropy_loss/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
 softmax_cross_entropy_loss/ShapeShapediv_1*
T0*
_output_shapes
:*
out_type0
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
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
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
T0*
_output_shapes
: 
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
N*
T0*
_output_shapes
:*

axis 
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
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
N*
T0*

Tidx0
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
"softmax_cross_entropy_loss/Shape_2Shapelabel*
_output_shapes
:*
out_type0*
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
(softmax_cross_entropy_loss/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
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
(softmax_cross_entropy_loss/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
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
T0*#
_output_shapes
:���������*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
_output_shapes
:*
out_type0
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
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
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *    
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
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
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
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
�
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
�
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
_output_shapes
: *I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
T0
�
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
T0
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
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
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
_output_shapes
: *
Tshape0*
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
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
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
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
�
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
T0
�
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
_output_shapes
: *L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
T0
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
�
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
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
Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
_output_shapes
:*
out_type0*
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
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
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
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape
�
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
T0
�
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
Tshape0*
T0
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
T0*
_output_shapes
: *b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape
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
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
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
T0*'
_output_shapes
:���������
*
Tshape0
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
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
/gradients/div_1_grad/tuple/control_dependency_1Identitygradients/div_1_grad/Reshape_1&^gradients/div_1_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/div_1_grad/Reshape_1
�
7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/div_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:

�
<gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/div_1_grad/tuple/control_dependency8^gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/div_1_grad/tuple/control_dependency=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/div_1_grad/Reshape*
T0
�
Fgradients/sequential_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad=^gradients/sequential_1/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*J
_class@
><loc:@gradients/sequential_1/dense_2/BiasAdd_grad/BiasAddGrad*
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
Cgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_2/MatMul_grad/MatMul<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
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
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*
T0*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*<
_output_shapes*
(:����������:����������
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
Kgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependency_1Identity<gradients/sequential_1/dropout_2/cond/Merge_grad/cond_grad:1B^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul*
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
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
N*
T0
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
Qgradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*(
_output_shapes
:����������*S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape_1*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/div_grad/ShapeShapesequential_1/dropout_2/cond/mul*
_output_shapes
:*
out_type0*
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
T0*
_output_shapes
: *
Tshape0
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
6gradients/sequential_1/dropout_2/cond/mul_grad/Shape_1Const*
dtype0*
_output_shapes
: *
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
T0*
_output_shapes
: *
Tshape0
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
gradients/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
x
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*(
_output_shapes
:����������
�
?gradients/sequential_1/dropout_2/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencygradients/zeros_1**
_output_shapes
:����������: *
N*
T0
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
:�*
T0*
data_formatNHWC
�
<gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/sequential_1/activation_3/Relu_grad/ReluGrad8^gradients/sequential_1/dense_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/sequential_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_3/Relu_grad/ReluGrad=^gradients/sequential_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/activation_3/Relu_grad/ReluGrad
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
Igradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_1/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_1/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������@*H
_class>
<:loc:@gradients/sequential_1/flatten_1/Reshape_grad/Reshape*
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
gradients/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*/
_output_shapes
:���������@
�
=gradients/sequential_1/dropout_1/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_1/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
N*
T0*1
_output_shapes
:���������@: 
�
<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_1/cond/dropout/div*
_output_shapes
:*
out_type0*
T0
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
>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeReshape:gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Sum<gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Shape*/
_output_shapes
:���������@*
Tshape0*
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
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
Ogradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/div_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape
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
Ggradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_1/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_1/cond/mul_grad/tuple/group_deps*/
_output_shapes
:���������@*I
_class?
=;loc:@gradients/sequential_1/dropout_1/cond/mul_grad/Reshape*
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
gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
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
Egradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/sequential_1/activation_2/Relu_grad/ReluGrad>^gradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/sequential_1/activation_2/Relu_grad/ReluGrad*/
_output_shapes
:���������@*
T0
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
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"      @   @   
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
Ggradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad>^gradients/sequential_1/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
T0*
_output_shapes
:*
out_type0
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
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
	container *
shared_name *
dtype0*
shape:@*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1zeros_3*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
l
zeros_4Const*
dtype0*&
_output_shapes
:@@*%
valueB@@*    
�
conv2d_2/kernel/Adam
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
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
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
conv2d_2/kernel/Adam_1/AssignAssignconv2d_2/kernel/Adam_1zeros_5*&
_output_shapes
:@@*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
T0*
use_locking(
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
shape:@*
_output_shapes
:@*
shared_name * 
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
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
dtype0*
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
dtype0*!
_output_shapes
:���* 
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
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1zeros_9*!
_output_shapes
:���*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
�
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
W
zeros_10Const*
valueB�*    *
dtype0*
_output_shapes	
:�
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
dtype0*
_output_shapes
:	�
*
valueB	�
*    
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
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�
*
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
zeros_14Const*
dtype0*
_output_shapes
:
*
valueB
*    
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
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
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
Adam/beta2&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_2/kernel/ApplyAdam$^Adam/update_conv2d_2/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0��'�       ��-	�j��Xc�A*

loss�K@T�       ��-	�, �Xc�A*

loss�@��1       ��-	�� �Xc�A*

lossWN@D|4i       ��-	�}�Xc�A*

loss�/@%ԗI       ��-	Y2�Xc�A*

loss��?���Q       ��-	���Xc�A*

loss�;�?Y�uE       ��-	���Xc�A*

loss��?��       ��-	�-�Xc�A*

loss�S�?�5�       ��-	�I�Xc�A	*

lossl�?���       ��-	O�Xc�A
*

loss��?y9D       ��-	ß�Xc�A*

lossN�k?���       ��-	�=�Xc�A*

lossśD?�S�       ��-	��Xc�A*

loss/�?3"$�       ��-	~�Xc�A*

loss7Q�?$ɕ        ��-	�	�Xc�A*

loss���?���H       ��-	��	�Xc�A*

loss�	?�=$�       ��-	<O
�Xc�A*

losss�V?���       ��-	s�
�Xc�A*

loss��A?	E��       ��-	���Xc�A*

loss��?>���       ��-	�,�Xc�A*

loss�Yz?.�b       ��-	���Xc�A*

loss�=3?7�,#       ��-	}^�Xc�A*

lossm+?��0       ��-	���Xc�A*

loss�q?��k�       ��-	��Xc�A*

loss�q3?AIH       ��-	�%�Xc�A*

lossΟM?��       ��-	/��Xc�A*

loss#\?���       ��-	GY�Xc�A*

lossZ1?d���       ��-	���Xc�A*

lossl+8?�?�       ��-	��Xc�A*

loss�"?��X�       ��-	!�Xc�A*

loss(A%?nbO       ��-	R��Xc�A*

loss�{?gƹ       ��-	�|�Xc�A *

lossc?��,       ��-	`�Xc�A!*

lossj�H?]1�       ��-	ND�Xc�A"*

loss�"?HP�       ��-	v��Xc�A#*

loss'z?'�i�       ��-	�y�Xc�A$*

loss	*�>~#�w       ��-	K#�Xc�A%*

losst0?ݎ�-       ��-	���Xc�A&*

loss{Z?.7       ��-	=d�Xc�A'*

lossw/?���       ��-	��Xc�A(*

loss6!?���       ��-	�W�Xc�A)*

loss���>�D=�       ��-	f��Xc�A**

lossc8?�%�C       ��-	P�Xc�A+*

loss�� ?F��M       ��-	�5�Xc�A,*

loss%�?{��       ��-	(��Xc�A-*

loss�I?��w       ��-	�z�Xc�A.*

loss?!1;�       ��-	 �Xc�A/*

loss�>�>�s�       ��-	�� �Xc�A0*

loss��>|�hJ       ��-	�]!�Xc�A1*

loss4h�>BL�       ��-	��!�Xc�A2*

loss�$u>Nx�0       ��-	�"�Xc�A3*

lossw��>i�|       ��-	M�#�Xc�A4*

loss�,�>VP܋       ��-	%$�Xc�A5*

lossAp?IZf�       ��-	��$�Xc�A6*

loss|x>Qx�X       ��-	c`%�Xc�A7*

loss9�>�.�       ��-	(
&�Xc�A8*

lossq��>Cs��       ��-	�&�Xc�A9*

loss�J�>�dhA       ��-	�E'�Xc�A:*

losst�!?G�1       ��-	��'�Xc�A;*

loss���>R�k�       ��-	�}(�Xc�A<*

loss!�>�@-�       ��-	�)�Xc�A=*

loss��>a�)       ��-	��)�Xc�A>*

loss��>�*ȇ       ��-	E*�Xc�A?*

loss$��>�i��       ��-	��*�Xc�A@*

loss<��>�
�       ��-	�t+�Xc�AA*

loss�!�>h�S�       ��-	�,�Xc�AB*

loss��>����       ��-	��,�Xc�AC*

lossEm�>W�*�       ��-	'N-�Xc�AD*

lossb?<l=       ��-	�.�Xc�AE*

loss���>��       ��-	ס.�Xc�AF*

loss��>]�~[       ��-	�9/�Xc�AG*

loss�͟>�O�       ��-	�/�Xc�AH*

loss�Z�>�J�N       ��-	.p0�Xc�AI*

loss��>��d       ��-	1�Xc�AJ*

lossxR�>�>�       ��-	��1�Xc�AK*

loss�	�>�qI       ��-	{�2�Xc�AL*

loss��?�?��       ��-	K!3�Xc�AM*

loss�G?o�s       ��-	�3�Xc�AN*

loss�&X>��       ��-	�Y4�Xc�AO*

loss?��>Sʴ       ��-	��4�Xc�AP*

loss�<�>���       ��-	i�5�Xc�AQ*

lossí ?���       ��-	�+6�Xc�AR*

loss,�?m���       ��-	��6�Xc�AS*

lossE�D>�d
�       ��-	B]7�Xc�AT*

loss���>B��2       ��-	��7�Xc�AU*

lossI"�>�)�/       ��-	��8�Xc�AV*

lossh,q>|��       ��-	w+9�Xc�AW*

lossGI�>2��       ��-	,d:�Xc�AX*

loss�0�>���\       ��-	� ;�Xc�AY*

loss]ި>��       ��-	ˢ;�Xc�AZ*

loss���>�-2�       ��-	2u=�Xc�A[*

loss�&�>����       ��-	�>�Xc�A\*

loss���>�/       ��-	�>�Xc�A]*

loss���>���       ��-	g?�Xc�A^*

loss��>���       ��-	}�@�Xc�A_*

loss��X>��`~       ��-	 BA�Xc�A`*

lossL�y>0h=       ��-	��A�Xc�Aa*

lossQ�>'�}�       ��-	��B�Xc�Ab*

losso!?����       ��-		�C�Xc�Ac*

lossŤo>��w       ��-	�D�Xc�Ad*

loss�i>Q�       ��-	��E�Xc�Ae*

loss���=��?        ��-	�G�Xc�Af*

loss��>`�b       ��-	��G�Xc�Ag*

lossw2Q>�`��       ��-	.�H�Xc�Ah*

loss��W>�\��       ��-	_~I�Xc�Ai*

loss�].>I��|       ��-	4HJ�Xc�Aj*

losst+>�h�       ��-	�;K�Xc�Ak*

loss�	>.5�t       ��-	�qL�Xc�Al*

loss۵�>J�]x       ��-	�M�Xc�Am*

lossw�>T��       ��-	6uN�Xc�An*

loss���>��3�       ��-	h�O�Xc�Ao*

loss��>�J�       ��-	%�P�Xc�Ap*

loss�3>�laU       ��-	ۤQ�Xc�Aq*

loss-R&>*��       ��-	τR�Xc�Ar*

loss� �=yQO       ��-	�%S�Xc�As*

lossz��>C��       ��-	��S�Xc�At*

lossϰP>�ʩ	       ��-	�uT�Xc�Au*

loss��1>���$       ��-	�U�Xc�Av*

loss �>��       ��-	D�U�Xc�Aw*

lossN>C�T4       ��-	�7V�Xc�Ax*

loss�p>�Z��       ��-	6�V�Xc�Ay*

loss�k>���       ��-	�cW�Xc�Az*

loss[�>��z       ��-	��W�Xc�A{*

loss *�>���(       ��-	_�X�Xc�A|*

loss�E�=�\�`       ��-	FAY�Xc�A}*

loss8E�>�Ҋ       ��-	Q�Y�Xc�A~*

lossn�>�f�       ��-	�{Z�Xc�A*

loss��*>"�F       �	�[�Xc�A�*

loss�ݑ>�*�       �	��[�Xc�A�*

lossR��>�T�=       �	�A\�Xc�A�*

loss�vf>�6��       �	|�\�Xc�A�*

loss�r>)��9       �	�h]�Xc�A�*

loss`�#>��       �	�^�Xc�A�*

lossN6�=�u�       �	l�^�Xc�A�*

lossX��>21��       �	�/_�Xc�A�*

loss1�|>�� �       �	��_�Xc�A�*

loss`&(>�J.       �	�i`�Xc�A�*

loss)��=�m�       �	�a�Xc�A�*

loss9w>2�D�       �	Ûa�Xc�A�*

loss���=>&<�       �	4b�Xc�A�*

loss.4�=d�b       �	K�b�Xc�A�*

loss�S|>>K��       �	�kc�Xc�A�*

loss_c�>a�fz       �	�d�Xc�A�*

loss� T>&pU_       �	Șd�Xc�A�*

loss�t�>�+��       �	�4e�Xc�A�*

loss��=��&       �	��e�Xc�A�*

lossыX>�!�       �	�jf�Xc�A�*

loss���=��H�       �	Bg�Xc�A�*

loss8�*>���       �	��g�Xc�A�*

loss3F>^���       �	Eh�Xc�A�*

lossV�>1i9j       �	
�h�Xc�A�*

loss�|�>{�V       �	�i�Xc�A�*

loss��{>q�       �	�j�Xc�A�*

lossQB6>{�@j       �	}�j�Xc�A�*

loss >/Wհ       �	�Mk�Xc�A�*

loss��=��x       �	K�k�Xc�A�*

loss��>���       �	�l�Xc�A�*

lossz�>��       �	m�Xc�A�*

loss�ɕ>��6       �	�m�Xc�A�*

loss _>�wPG       �	�Hn�Xc�A�*

loss-�>��zX       �	&�n�Xc�A�*

lossC?>�z�?       �	�zo�Xc�A�*

loss<�>6xf       �	F�p�Xc�A�*

loss�b�=ʣ��       �	��q�Xc�A�*

loss�7M>��L�       �	! r�Xc�A�*

lossN��=�k{h       �	�
s�Xc�A�*

loss��= �+       �	y�s�Xc�A�*

loss�V,>��`0       �	�Wt�Xc�A�*

loss��>�W�e       �	��t�Xc�A�*

loss:�=�Ma       �	ۆu�Xc�A�*

loss���=Z�پ       �	iv�Xc�A�*

loss!3�=��K       �	��v�Xc�A�*

losszA(>�T�#       �	QNw�Xc�A�*

loss�;R>�       �	��w�Xc�A�*

losst��=��1�       �	�ux�Xc�A�*

loss?��>��y       �	Fy�Xc�A�*

loss��>��|�       �	��y�Xc�A�*

loss�JO>&Zf�       �	�Az�Xc�A�*

losso4�=A滔       �	 �z�Xc�A�*

lossm3>l@�       �	F�{�Xc�A�*

loss���=7%J       �	7|�Xc�A�*

loss��>���G       �	��|�Xc�A�*

loss7�>���       �	��}�Xc�A�*

loss�T�=��	       �	3~�Xc�A�*

loss ��=��9�       �	9�~�Xc�A�*

loss, />F��]       �	?T�Xc�A�*

lossڼ�=���       �	(���Xc�A�*

lossN��>�I�3       �	�[��Xc�A�*

lossM�Y>D��       �	���Xc�A�*

loss�O->�CZ�       �	D��Xc�A�*

loss�*�>e2��       �	�惛Xc�A�*

loss� Z>2�9{       �	qƄ�Xc�A�*

loss/��=p       �	H���Xc�A�*

loss�w>bƕ       �	���Xc�A�*

lossfw�=�e�4       �	�Ň�Xc�A�*

loss&�]>�C       �	x��Xc�A�*

loss�`0>?!��       �	~��Xc�A�*

loss�R>��\       �	�Q��Xc�A�*

loss�\�>���       �	�h��Xc�A�*

lossf�;>�?       �	A���Xc�A�*

loss�'>�/��       �	����Xc�A�*

lossd� >�~�       �	�&��Xc�A�*

loss���=�7%u       �	0���Xc�A�*

loss���=�       �	Q���Xc�A�*

loss��>&Q�       �	�Б�Xc�A�*

loss��= �W       �	����Xc�A�*

loss��>)9�       �	B'��Xc�A�*

loss�d)>ڿ       �	��Xc�A�*

loss�z\> �e+       �	!���Xc�A�*

loss
]F>�a[       �	'3��Xc�A�*

loss6f>�o��       �	S���Xc�A�*

lossM>|�-       �	T9��Xc�A�*

loss��>j<J�       �	ݗ�Xc�A�*

loss���=�ٻ�       �	�z��Xc�A�*

loss�&q>h��       �	s���Xc�A�*

lossЈ>"��_       �	�;��Xc�A�*

loss4J>l�       �	v���Xc�A�*

loss>t>2t�G       �	)���Xc�A�*

loss��>bb͚       �	�.��Xc�A�*

loss��`>��        �	�Ӝ�Xc�A�*

loss�:n=�di�       �	e���Xc�A�*

loss�ɉ=�oh       �	�4��Xc�A�*

loss*{�> 	��       �	6̞�Xc�A�*

loss��>���       �	_��Xc�A�*

loss��>��b�       �	%��Xc�A�*

loss���=��*       �	����Xc�A�*

lossl}v>��@       �	C��Xc�A�*

loss�`>�lf       �	�㡛Xc�A�*

lossb>�Ƿa       �	�~��Xc�A�*

loss��V>Q��       �	��Xc�A�*

loss;)�=S���       �	���Xc�A�*

loss�k>��       �	6Y��Xc�A�*

loss�]O>v!       �	����Xc�A�*

loss`V>h�Ô       �	;���Xc�A�*

loss��a>��m�       �	�*��Xc�A�*

loss#�f>�-�       �	Kʦ�Xc�A�*

loss��3>��b'       �	h��Xc�A�*

lossi��>Ȋ�)       �	-��Xc�A�*

loss�Ey>���       �	o���Xc�A�*

loss-�=V��?       �	`:��Xc�A�*

loss�/>���       �	�ة�Xc�A�*

lossw�&>���)       �	.p��Xc�A�*

loss���>��ӂ       �	��Xc�A�*

loss,�A>��g       �	����Xc�A�*

loss�-D>���W       �	�*��Xc�A�*

loss�0�=����       �	{���Xc�A�*

lossQ��=��       �	�Q��Xc�A�*

loss��>f`�       �	�孛Xc�A�*

loss�	�>�JA       �	pz��Xc�A�*

lossfo\>ˁ��       �	{��Xc�A�*

loss�o9>�Im�       �	����Xc�A�*

loss*�.>u��       �	K=��Xc�A�*

losse=>Rme       �	TⱛXc�A�*

loss�)�="ZF�       �	���Xc�A�*

loss�T>ԡ2       �	_$��Xc�A�*

lossrϻ=�0        �	�ų�Xc�A�*

lossTMa>��V       �	Z��Xc�A�*

lossN�>Q#}�       �	4�Xc�A�*

loss�{�=@+�       �	⏵�Xc�A�*

loss��0>Y�"|       �	�$��Xc�A�*

loss�t>�]�y       �	����Xc�A�*

loss��Q>�m�       �	�O��Xc�A�*

loss���>�]�       �	?㷛Xc�A�*

loss[�>���       �	�׸�Xc�A�*

loss�_�>X�[       �	�h��Xc�A�*

loss���=m��       �	z���Xc�A�*

loss_4[>H"��       �	����Xc�A�*

loss�>�d��       �	(-��Xc�A�*

loss�6�>��GO       �	"���Xc�A�*

loss�rO>#Ax       �	�Y��Xc�A�*

loss�4#>$+47       �	����Xc�A�*

loss��k>�G)       �	֍��Xc�A�*

loss#1>?��0       �	G!��Xc�A�*

lossz�>`���       �	���Xc�A�*

loss��>��U       �	�X��Xc�A�*

lossE�m>,q��       �	A�Xc�A�*

loss�>H#       �	f���Xc�A�*

loss�>!w       �	%u��Xc�A�*

loss�F%>m��       �	@PÛXc�A�*

loss=J>�ܹ�       �	��ÛXc�A�*

loss�=���&       �	�ěXc�A�*

losse��=��       �	R&śXc�A�*

loss59>��O�       �	>�śXc�A�*

lossa>��\       �	�uƛXc�A�*

loss�h>�?�M       �	�ǛXc�A�*

loss�>6<z�       �	IțXc�A�*

loss�u�=R�6m       �	��țXc�A�*

loss���=�-��       �	�ɛXc�A�*

loss��=���       �	yʛXc�A�*

loss\��<GK       �	�˛Xc�A�*

lossS��=�>       �	��˛Xc�A�*

loss�,
>��       �	�Q̛Xc�A�*

lossAT!=���       �	��̛Xc�A�*

lossn5�=t�	N       �	��͛Xc�A�*

loss	8�=]�=       �	�,ΛXc�A�*

loss�3K>��wT       �	i�ΛXc�A�*

loss �/>�iw�       �	OyϛXc�A�*

loss��]>�*��       �	�ЛXc�A�*

loss�;>���
       �	&�ЛXc�A�*

loss�K>�#�       �	�PћXc�A�*

loss��=}�R       �	��ћXc�A�*

loss&�L=�M0�       �	��қXc�A�*

loss�-�=s��       �	�ӛXc�A�*

loss
�Z>>�{�       �	P�ӛXc�A�*

lossM�=�dfH       �	�AԛXc�A�*

loss�s�>��5       �	��ԛXc�A�*

loss�0/=�\�       �	�q՛Xc�A�*

loss�l >�l�       �	֛Xc�A�*

lossr�E=L:ƛ       �	Ӣ֛Xc�A�*

loss̘�=e
|       �	_EכXc�A�*

loss �=vA��       �	-�כXc�A�*

loss�Ym>�t       �	��؛Xc�A�*

loss�=>�(��       �	�5ٛXc�A�*

loss��=f��       �	��ٛXc�A�*

loss\�=��'�       �		�ڛXc�A�*

loss�j>��Ҝ       �	#/ۛXc�A�*

loss��=P0��       �	B�ۛXc�A�*

loss$��=4��\       �	��ܛXc�A�*

loss�*>7�       �	t&ݛXc�A�*

loss��6>��̍       �	x�ݛXc�A�*

loss���=�$�]       �	�zޛXc�A�*

loss��~>���       �	c+ߛXc�A�*

loss�x1><���       �	�ߛXc�A�*

loss\,>���       �	����Xc�A�*

loss%��=� �6       �	-�Xc�A�*

lossa��>@�;�       �	���Xc�A�*

lossf�D>�&'V       �	�y�Xc�A�*

loss��=�k��       �	��Xc�A�*

loss�h>�O5       �	!��Xc�A�*

loss�P<>�7 �       �	�u�Xc�A�*

loss��!>�a�k       �	�Xc�A�*

loss�;�=��R�       �	���Xc�A�*

lossa��=#�h�       �	�s�Xc�A�*

loss�t�=�80       �	��Xc�A�*

loss��P>z���       �	P��Xc�A�*

loss,G�=g#�       �	�x�Xc�A�*

lossS�=�َ�       �	�Xc�A�*

loss�{>��       �	/��Xc�A�*

loss_��=���       �	&o�Xc�A�*

loss}n=k<��       �	<N�Xc�A�*

loss�r>j�z�       �	��Xc�A�*

lossS��=u�	       �	��Xc�A�*

loss}�>���       �	�B�Xc�A�*

loss�?�=��d+       �	���Xc�A�*

lossf��>�Sb       �	Y��Xc�A�*

lossA>c{�6       �	N+�Xc�A�*

loss�|=t��       �	���Xc�A�*

loss|=>�A�       �	^e�Xc�A�*

loss`>Рi�       �	 �Xc�A�*

loss�>>�N�       �	N��Xc�A�*

loss?̯=m�N       �	���Xc�A�*

loss��=6Gۂ       �	�-�Xc�A�*

loss%�U>YS��       �	e��Xc�A�*

loss�S�>@	��       �	����Xc�A�*

lossAF>V�       �	0G��Xc�A�*

loss�܏=㦏�       �	 ���Xc�A�*

loss@��=\�WX       �	����Xc�A�*

lossS+�=,v�       �	"p��Xc�A�*

loss*�>�E2       �	���Xc�A�*

loss~	>���       �	$���Xc�A�*

loss6��=4[l�       �	�]��Xc�A�*

loss.��=�mA       �	���Xc�A�*

loss�D�=�I�       �	>���Xc�A�*

lossO��>JR��       �	�U��Xc�A�*

loss���>m0��       �	���Xc�A�*

loss�H>��.�       �	؜��Xc�A�*

loss��>�	H       �	�@��Xc�A�*

lossM�=��V       �	w���Xc�A�*

loss�}>�Z�       �	x���Xc�A�*

loss���=1��z       �	�� �Xc�A�*

lossa��=Ƞfw       �	�L�Xc�A�*

lossy>K���       �	S��Xc�A�*

lossn��=�0S       �	���Xc�A�*

loss4�v=�m�       �	��Xc�A�*

loss3�>���       �	�h�Xc�A�*

loss�8P>_��       �	�Xc�A�*

loss�A9>�э       �	��Xc�A�*

loss��=B�=�       �	�h�Xc�A�*

loss� �=�(�|       �	��Xc�A�*

loss�G>\?��       �	@��Xc�A�*

loss`�N>5e�V       �	�h�Xc�A�*

loss���=Nt�Z       �		�Xc�A�*

loss��(>T�J�       �	c�	�Xc�A�*

loss�F�=7�7g       �	b
�Xc�A�*

lossy�=�>��       �	��Xc�A�*

lossd?>�>�       �	=��Xc�A�*

loss?h>�D��       �	F`�Xc�A�*

loss��>�պ�       �	��Xc�A�*

loss�S�=O��       �	t��Xc�A�*

loss Q.>�ѡ�       �	p^�Xc�A�*

lossnA>t�'�       �	�F�Xc�A�*

loss<n�=�7m       �	r��Xc�A�*

loss�*P>k*t       �	���Xc�A�*

lossx��=	�us       �	�g�Xc�A�*

loss?�=�"�       �	�Xc�A�*

loss˂>���E       �	��Xc�A�*

loss
�=^A�       �	wi�Xc�A�*

loss0�=�""�       �	1
�Xc�A�*

loss~�=vQ��       �	ް�Xc�A�*

lossJ->�Lf�       �	�^�Xc�A�*

loss1@�=CM��       �	�	�Xc�A�*

loss��>�n�u       �	#��Xc�A�*

loss��>�$��       �	�d�Xc�A�*

loss��=THH�       �	�Xc�A�*

lossj��=�ֽ       �	���Xc�A�*

lossʾ�=�<�       �	fI�Xc�A�*

loss8�a>aee�       �	:�Xc�A�*

loss��>�#f       �	Ϣ�Xc�A�*

loss\�>�C�2       �	lA�Xc�A�*

loss�&�>~p�       �	���Xc�A�*

loss�`Y>u���       �	��Xc�A�*

lossQ�>
*�       �	�"�Xc�A�*

loss���=,�X�       �	x��Xc�A�*

loss�Pe>���       �	�m�Xc�A�*

lossϭo>�@~�       �	B �Xc�A�*

loss���>͛       �	Q� �Xc�A�*

loss=`f=�c��       �	:!�Xc�A�*

lossݨ�=i3b/       �	�|"�Xc�A�*

loss�->쓲R       �	�%#�Xc�A�*

losso>�)�4       �	v�#�Xc�A�*

loss��<=2rg       �	�d$�Xc�A�*

loss]=*>���       �	�%�Xc�A�*

loss�� >��bS       �	�%�Xc�A�*

lossl�j>���n       �	�w&�Xc�A�*

loss��">p���       �	�'�Xc�A�*

losskr�>���       �	��'�Xc�A�*

loss��(>lӬ�       �	\(�Xc�A�*

loss�-?>�ܪ�       �	 )�Xc�A�*

lossҕ&>ݖ
�       �	��)�Xc�A�*

lossء�=+��       �	UO*�Xc�A�*

loss���=�E�U       �	)�*�Xc�A�*

loss�C�=W��       �	͕+�Xc�A�*

losst�z=gA0       �	�2,�Xc�A�*

loss�->tP5z       �	4�,�Xc�A�*

losse�=��Qr       �	��-�Xc�A�*

loss�n�=���B       �	>#.�Xc�A�*

lossr�=b��8       �	2�.�Xc�A�*

loss�pb=���       �	k�/�Xc�A�*

loss͍,=�M�,       �	y!0�Xc�A�*

loss�"I= �%�       �	��0�Xc�A�*

lossal>M�ob       �	��1�Xc�A�*

loss�z >֪�O       �	�"2�Xc�A�*

loss3�=��.       �	��2�Xc�A�*

loss�B>5�rh       �	Na3�Xc�A�*

loss7�<<k�       �	24�Xc�A�*

lossj2>�a�       �	�4�Xc�A�*

lossS��>P�p       �	�B5�Xc�A�*

loss��>        �	��5�Xc�A�*

lossV>z��
       �	�|6�Xc�A�*

loss�K>.p&�       �	�7�Xc�A�*

loss�e>�@��       �	A�7�Xc�A�*

lossS��=a��       �	�Z8�Xc�A�*

loss�I=(E��       �	��8�Xc�A�*

loss�7>+;�       �	��9�Xc�A�*

loss�x>���@       �	�):�Xc�A�*

loss�U9>�cÐ       �	H�:�Xc�A�*

loss�q�=�r��       �	�`;�Xc�A�*

loss<��=K�J�       �	�*<�Xc�A�*

loss���=�A�^       �	}�<�Xc�A�*

loss�W�=�0       �	�n=�Xc�A�*

loss�d�=�V       �	�	>�Xc�A�*

loss�cY="~m�       �	Φ>�Xc�A�*

lossp��=/��       �	�O?�Xc�A�*

loss>�>{P/       �	�%@�Xc�A�*

loss��R=��       �	��@�Xc�A�*

loss'5=O�F       �	�A�Xc�A�*

loss�w~=re�        �	
JB�Xc�A�*

loss���=��`�       �	�mC�Xc�A�*

losseQ�='L�@       �	�ED�Xc�A�*

loss�~=�v       �	��D�Xc�A�*

loss��0>���        �	őE�Xc�A�*

loss��>��m�       �	�DF�Xc�A�*

lossc�9>�(S�       �	��F�Xc�A�*

loss�n=>se(       �	"�G�Xc�A�*

lossl�K=�#Oc       �	�UH�Xc�A�*

loss�0=z��       �	CI�Xc�A�*

loss��#=��Y       �	|�I�Xc�A�*

loss�(�=��q       �	�[J�Xc�A�*

loss��_=��       �	S	K�Xc�A�*

lossjx�>��.k       �	��K�Xc�A�*

loss�?�>$S(       �	�gL�Xc�A�*

loss��=p��       �	�M�Xc�A�*

lossv��=���       �	>�M�Xc�A�*

loss
X�=e��       �	�}N�Xc�A�*

loss�a�=T�h       �	/4O�Xc�A�*

loss*��=�Ǽ�       �	y�O�Xc�A�*

loss33>��^f       �	c�P�Xc�A�*

loss�3�=���A       �	��Q�Xc�A�*

loss�Ə=w9X�       �	�S�Xc�A�*

lossHn[>�*i5       �	�T�Xc�A�*

loss$z�=]�?       �	AU�Xc�A�*

lossÜ'>�I4�       �	y�U�Xc�A�*

lossR|�<�å       �	�V�Xc�A�*

loss�?@=�M��       �	�>W�Xc�A�*

lossN�#=���P       �	Q�W�Xc�A�*

loss�x�<�	?!       �	�X�Xc�A�*

loss#�V>��0�       �	�WY�Xc�A�*

loss��!=j;�%       �	�Z�Xc�A�*

loss�4�=���       �	D�Z�Xc�A�*

loss!�>P��       �	mt[�Xc�A�*

lossC�>50�/       �	eR\�Xc�A�*

loss�)
=��-       �	D�\�Xc�A�*

loss�U�=�       �	Φ]�Xc�A�*

lossMs�=ڢt       �	�^�Xc�A�*

loss!>���(       �	�/_�Xc�A�*

lossvq�=�k��       �	��_�Xc�A�*

loss;}�>��6K       �	�n`�Xc�A�*

lossa�4=dN�       �	Ea�Xc�A�*

lossi-<>�4��       �	ݳa�Xc�A�*

loss��=��       �	Ab�Xc�A�*

loss#E=�p�       �	�+c�Xc�A�*

loss�lu=F�M       �	��c�Xc�A�*

loss�+�=�[ƨ       �	�ld�Xc�A�*

lossM�&> O=�       �	Ve�Xc�A�*

loss�=�;�       �	S�e�Xc�A�*

loss�ce>�9u       �	�Rf�Xc�A�*

lossZ��=q�       �	L�f�Xc�A�*

lossg>�Ve�       �	�g�Xc�A�*

loss�'�=+BǾ       �	�=h�Xc�A�*

loss6T>?L\\       �	��h�Xc�A�*

lossx��=�r�       �	Ȕi�Xc�A�*

lossє=�=�Y       �	�Aj�Xc�A�*

loss���<�W�d       �	��j�Xc�A�*

loss4��=����       �	��k�Xc�A�*

lossX�>��{�       �	�/l�Xc�A�*

loss��=�cu�       �	-�l�Xc�A�*

loss�(�=�)ؽ       �	�lm�Xc�A�*

loss&�=`�1M       �	n�Xc�A�*

loss_�'>�@:�       �	�n�Xc�A�*

lossr1=c���       �	Wo�Xc�A�*

lossA�=�T�U       �	�p�Xc�A�*

lossOn=�J�"       �	C�p�Xc�A�*

losst�<ɚ��       �	�Yq�Xc�A�*

losss�<L�i       �	%r�Xc�A�*

loss��=��       �	��r�Xc�A�*

lossuG=Y`        �	
js�Xc�A�*

loss�n<= E       �	t�Xc�A�*

loss7P�<����       �	�t�Xc�A�*

loss�<���i       �	�Uu�Xc�A�*

loss���=�F�l       �	<�u�Xc�A�*

loss�vm;JkV       �	c�v�Xc�A�*

lossjc<�&~E       �	G=w�Xc�A�*

loss�<��\       �	��w�Xc�A�*

lossؒ�=����       �	}xx�Xc�A�*

loss�� >�d��       �	"y�Xc�A�*

loss+��=��       �	��y�Xc�A�*

loss㊀<f� ~       �	vSz�Xc�A�*

loss��=���       �	��z�Xc�A�*

lossV��>{'L?       �	��{�Xc�A�*

lossc�<�+Z       �	\v|�Xc�A�*

lossH�>�*       �	�%}�Xc�A�*

lossj�=5+��       �	�}�Xc�A�*

loss�G>̟��       �	4h~�Xc�A�*

lossZ�=�6�       �	T�Xc�A�*

loss��	=KA�c       �	J��Xc�A�*

lossRmD>�Ą�       �	V-��Xc�A�*

loss��2>�2%       �	
ڀ�Xc�A�*

loss�?>k�M       �	x���Xc�A�*

loss$��=�I��       �	���Xc�A�*

loss%��=�p8$       �	K���Xc�A�*

losssb{>�lY       �	�K��Xc�A�*

lossh&>X#c       �	�ꃜXc�A�*

loss�w>�e@       �	����Xc�A�*

loss�b>��"D       �	���Xc�A�*

lossԿI>�ݪ       �	���Xc�A�*

lossd��=ך��       �	�E��Xc�A�*

loss���=؃��       �	���Xc�A�*

loss�1>�8�       �	3���Xc�A�*

loss@��=��4       �	u;��Xc�A�*

loss@�=��,       �	g҈�Xc�A�*

lossM9�=�V       �	�~��Xc�A�*

loss!q(>R���       �	���Xc�A�*

loss�[=�E9�       �	����Xc�A�*

loss�y�=���       �	>@��Xc�A�*

lossC�$=x��0       �	G䋜Xc�A�*

lossĩ�=�AZa       �	����Xc�A�*

lossh�=K�
�       �	�!��Xc�A�*

loss3�+>��̐       �	����Xc�A�*

lossM�!>~�f       �	�\��Xc�A�*

loss_�={��       �	4���Xc�A�*

loss�.�=c �       �	~���Xc�A�*

loss��=���       �	��Xc�A�*

loss��<��Z       �	c`��Xc�A�*

lossd6�=oj߅       �	���Xc�A�*

loss��<��7       �	����Xc�A�*

lossxCB=?qm       �	�6��Xc�A�*

loss �=c�G�       �	擜Xc�A�*

loss�>���	       �	�~��Xc�A�*

lossL�/>?`@       �	���Xc�A�*

loss���=F��       �	󪕜Xc�A�*

loss�v(=� T       �	�H��Xc�A�*

loss���=�L��       �	?▜Xc�A�*

loss�A�=��/�       �	hx��Xc�A�*

loss5�=�r�.       �	���Xc�A�*

lossQ��=Hl��       �	m���Xc�A�*

losss��=�,bZ       �	?��Xc�A�*

loss�G�<�aUx       �	�ڙ�Xc�A�*

loss���=#��       �	+n��Xc�A�*

loss	�=T�V       �	e5��Xc�A�*

loss���=%��(       �	Dߛ�Xc�A�*

loss�D�=ʌ��       �	]���Xc�A�*

loss��>��[�       �	���Xc�A�*

loss�>��W"       �	*��Xc�A�*

loss��7>�z��       �	9ն�Xc�A�*

lossʉ=-�)       �	#i��Xc�A�*

loss�)@=D՗       �	���Xc�A�*

lossu=>EĚ�       �	����Xc�A�*

loss��X>��B�       �	@��Xc�A�*

lossv+�=�<�)       �	Թ�Xc�A�*

loss��*>���       �	Tq��Xc�A�*

loss��<��^       �	���Xc�A�*

loss3_b=ۦ*       �	ǻ��Xc�A�*

lossL��=���       �	�P��Xc�A�*

loss�.�=�x�       �	�Xc�A�*

loss���=���2       �	!���Xc�A�*

loss��2>y��,       �	�3��Xc�A�*

loss@rN=�R�       �	=վ�Xc�A�*

loss�`�=����       �	l��Xc�A�*

lossdf=�ј�       �	Z��Xc�A�*

loss��>I�y1       �	���Xc�A�*

lossC$>���       �	����Xc�A�*

lossls+>�ѰA       �	J&Xc�A�*

lossAͰ=e�<�       �	��Xc�A�*

loss=�>)�q�       �	��ÜXc�A�*

loss���=�b�"       �	/ĜXc�A�*

loss=`�w       �	��ĜXc�A�*

loss�3}=k��       �	x�ŜXc�A�*

loss]O=Yg       �	��ƜXc�A�*

loss1o�=��_�       �	��ǜXc�A�*

loss���=�x�       �	�}ȜXc�A�*

loss�=lsnh       �	�2ɜXc�A�*

loss�xV=C��       �	��ɜXc�A�*

loss�[�=��oh       �	��ʜXc�A�*

losso6u>V�U       �	p'˜Xc�A�*

loss��(=RA��       �	G ̜Xc�A�*

loss~>���       �	U�̜Xc�A�*

loss�d(=C)ʑ       �	�c͜Xc�A�*

lossx�=9ZCz       �	 XΜXc�A�*

loss���>=�       �	�ΜXc�A�*

loss�/>���       �	��ϜXc�A�*

loss���=��I       �	3�МXc�A�*

loss���= է�       �	7ќXc�A�*

lossT;t=s1�       �	��ќXc�A�*

loss�Y>��7�       �	�pҜXc�A�*

loss�&x=�%�q       �	jӜXc�A�*

lossb�>��I�       �	�ӜXc�A�*

lossM�t=r��       �	4IԜXc�A�*

loss��=]i`�       �	�ԜXc�A�*

loss���=����       �	j�՜Xc�A�*

loss<��<�4�U       �	�[֜Xc�A�*

loss��<�*U�       �	��֜Xc�A�*

loss%@�={<4       �	N�לXc�A�*

lossŠ�=�'m�       �	�.؜Xc�A�*

loss|�>���       �	��؜Xc�A�*

loss$ax=���       �	�ٜXc�A�*

loss�<���R       �	�/ڜXc�A�*

loss�F�<��m       �	��ڜXc�A�*

lossE�2=w��       �	�ZۜXc�A�*

loss�e|=򣙹       �	 �ۜXc�A�*

loss��>s�       �	'�ܜXc�A�*

loss�O�=:�@^       �	ǃݜXc�A�*

lossZ��= ū�       �	\ޜXc�A�*

losstS=D%�!       �	��ޜXc�A�*

loss���=�U��       �	�OߜXc�A�*

loss���=/��b       �	z�ߜXc�A�*

lossF��=]��       �	ˁ��Xc�A�*

lossfh�=��%       �	��Xc�A�*

loss��>%��       �	{��Xc�A�*

loss��=d�Ί       �	�^�Xc�A�*

loss��1>v$�m       �	E��Xc�A�*

loss/�C=�@*       �	��Xc�A�*

loss�=�(��       �	��Xc�A�*

lossT�=�G�       �	���Xc�A�*

lossd�>4">a       �	A�Xc�A�*

loss��=E��       �	���Xc�A�*

loss��=sԏ�       �	<j�Xc�A�*

lossod�=�"Q�       �	
�Xc�A�*

loss��=d�       �	 ��Xc�A�*

loss�1L=����       �	Ց�Xc�A�*

loss!�c=�8�       �	�(�Xc�A�*

lossC�>���       �	׿�Xc�A�*

loss[�=�L       �	�T�Xc�A�*

loss��=��;       �	���Xc�A�*

loss#A6=z���       �	]��Xc�A�*

loss��=���       �	��Xc�A�*

loss%�Z=���       �	p��Xc�A�*

loss�dd=��2B       �	zP�Xc�A�*

loss��r=�`��       �	���Xc�A�*

lossa_�=s���       �	ʋ�Xc�A�*

lossH��=:�r�       �	�0�Xc�A�*

loss�B>�=�-       �	��Xc�A�*

loss��=�g2       �	S�Xc�A�*

loss�-�<�U�       �	���Xc�A�*

lossx�>��J�       �	&R�Xc�A�*

loss�U�=�w4�       �	z��Xc�A�*

lossl��<��5�       �	���Xc�A�*

lossR��=^5;       �	@M��Xc�A�*

loss���=.ךj       �	C���Xc�A�*

loss��>��       �	�}��Xc�A�*

loss�AX>�d!J       �	f��Xc�A�*

loss��=��B       �	���Xc�A�*

loss��<{��       �	N���Xc�A�*

loss���=�%�       �	�"��Xc�A�*

loss�N�=sX�       �	u���Xc�A�*

loss�'�=�X u       �	�m��Xc�A�*

loss�->��j<       �	���Xc�A�*

lossZ�2=r�@       �	ͱ��Xc�A�*

losst$]=��	       �	IK��Xc�A�*

lossFs>�1j       �	f���Xc�A�*

lossqv=�]       �	
���Xc�A�*

lossJ/R>��t�       �	8���Xc�A�*

loss�>u���       �	�%��Xc�A�*

loss�cK=��i       �	����Xc�A�*

loss��<�q��       �	SZ��Xc�A�*

loss6I=\t�       �	c���Xc�A�*

loss�р=��4�       �	;� �Xc�A�*

loss�Oo=���       �	�&�Xc�A�*

loss�z�=#�ã       �	�4�Xc�A�*

loss�>�<�       �	�h�Xc�A�*

loss���<���       �	VF�Xc�A�*

lossum�=b�}       �	*�Xc�A�*

loss��=A��L       �	�2�Xc�A�*

loss�!>5B8       �	���Xc�A�*

loss{��=(xdt       �	��Xc�A�*

loss���<��y$       �	=_�Xc�A�*

lossq�H>�݀�       �	�;	�Xc�A�*

loss�L�=���       �	�
�Xc�A�*

lossR�<x�=�       �	��
�Xc�A�*

lossX'5>Z�5�       �	�@�Xc�A�*

loss���=:UE       �	���Xc�A�*

loss�9�=�oC�       �	ʣ�Xc�A�*

loss�B >ܿ!�       �	�5�Xc�A�*

loss:|�=�{�       �	���Xc�A�*

loss͡=/}�O       �	�z�Xc�A�*

lossh'9>t�q�       �	��Xc�A�*

loss��=�2H�       �	m��Xc�A�*

loss9>�ݰ�       �	�C�Xc�A�*

lossv�<=��       �	i��Xc�A�*

loss�[=Z|��       �	���Xc�A�*

loss�(<(��       �	2X�Xc�A�*

lossU�<Ȃc       �	���Xc�A�*

loss�GD=�� P       �	n��Xc�A�*

loss
� >V�;'       �	�T�Xc�A�*

loss�Ub=��%       �	���Xc�A�*

lossI�>�b�i       �	��Xc�A�*

loss��=� ��       �	+�Xc�A�*

loss�2�=a�`�       �	��Xc�A�*

loss� <�M�       �	?��Xc�A�*

lossf�=�gC       �	�D�Xc�A�*

loss�L=����       �	3��Xc�A�*

lossnǻ<*��       �	�{�Xc�A�*

loss�O�=��p(       �	3�Xc�A�*

loss�g�=
��V       �	��Xc�A�*

loss���=�H       �	�K�Xc�A�*

loss�,�=��ۛ       �	���Xc�A�*

loss���<`=1       �	_�Xc�A�*

loss�g�<����       �	��Xc�A�*

loss�>*�;       �	F��Xc�A�*

loss��->z͛�       �	EH�Xc�A�*

loss��<�js2       �	���Xc�A�*

loss�}=��5       �	�}�Xc�A�*

lossSeA=��       �	; �Xc�A�*

loss.��=�       �	� �Xc�A�*

lossa=a��w       �	�P!�Xc�A�*

loss��>` ��       �	�!�Xc�A�*

loss�{=�P�H       �	��"�Xc�A�*

loss��.=�R}!       �	#�Xc�A�*

loss�z*>�w�S       �	�#�Xc�A�*

loss�t�=Su�)       �	D�$�Xc�A�*

loss��=�L�       �	�%�Xc�A�*

losso>~Ŭ       �	A�%�Xc�A�*

loss���<Ƃ:�       �	�W&�Xc�A�*

loss|{=�Q�       �	W�&�Xc�A�*

lossc��=�7}       �	S�'�Xc�A�*

loss3b�<V�Em       �	+(�Xc�A�*

loss}M�=��zn       �	-x)�Xc�A�*

lossa�=<��       �	*�Xc�A�*

loss�N�=H>�       �	��*�Xc�A�*

loss튦=w��^       �	�P+�Xc�A�*

loss���<>��       �	��+�Xc�A�*

loss�<��3�       �	g},�Xc�A�*

loss�9,>���       �	-�Xc�A�*

loss��H=����       �	�-�Xc�A�*

loss�x�<�ӧ       �	0H.�Xc�A�*

loss�x�<�X�       �		�.�Xc�A�*

loss@�o=�X	�       �	�z/�Xc�A�*

loss�S�=&S�       �	�0�Xc�A�*

loss�@=�N��       �	�0�Xc�A�*

loss�k=��	       �	_D1�Xc�A�*

lossQ!�=(w�k       �	$2�Xc�A�*

loss�l*>� ��       �	Ƥ2�Xc�A�*

loss3�<�|�Y       �	T3�Xc�A�*

loss��=P2��       �	��3�Xc�A�*

loss�H*=�0�       �	p}4�Xc�A�*

lossj<�y��       �	�5�Xc�A�*

lossqz�=�YF�       �	��5�Xc�A�*

loss��y>��l       �	iW6�Xc�A�*

lossd_�=5�       �	E�6�Xc�A�*

loss�'>���p       �	�7�Xc�A�*

loss��=���;       �	=+8�Xc�A�*

loss['=�^M^       �	��8�Xc�A�*

loss�#?=�Gb       �	[\9�Xc�A�*

loss&��=ʆ�o       �	�9�Xc�A�*

loss�JN=��/�       �	��:�Xc�A�*

loss�ȁ=��L       �	}@;�Xc�A�*

loss��B=!y|       �	,�;�Xc�A�*

loss�=�o�&       �	�<�Xc�A�*

lossV�
>1�Y�       �	l!=�Xc�A�*

loss4Z�=ݛ�       �	'�=�Xc�A�*

loss�gf= Ò�       �	]R>�Xc�A�*

loss�k�=���       �	,�>�Xc�A�*

loss,)==���       �	~�?�Xc�A�*

loss���=�4E       �	X:@�Xc�A�*

loss�=�rP[       �	��@�Xc�A�*

loss@�='��D       �	fB�Xc�A�*

lossp�=y+D       �	h<C�Xc�A�*

loss��;>{v=       �	��C�Xc�A�*

loss��E>��=       �	Q�D�Xc�A�*

loss��%>�^:       �	�AE�Xc�A�*

loss�r">ٍ,b       �	�E�Xc�A�*

loss�́<,p       �	$aG�Xc�A�*

loss�ȉ=��       �	\H�Xc�A�*

loss6 >�J5       �	��H�Xc�A�*

loss���=R�Y       �	O<I�Xc�A�*

lossi�;=�Q�       �	o�I�Xc�A�*

loss�l=����       �	�}J�Xc�A�*

loss��=j�M�       �	�K�Xc�A�*

loss�F�=�1J�       �	GL�Xc�A�*

loss)PH=��0Q       �	4�L�Xc�A�*

lossnݮ=�4�m       �	gHM�Xc�A�*

loss<A�< yo.       �	��M�Xc�A�*

losss�=�`@       �	xN�Xc�A�*

loss	��=��T1       �	�O�Xc�A�*

losshi1=�E�       �	~�O�Xc�A�*

lossȳ�=l�sC       �	6rP�Xc�A�*

lossc7�=�L$       �	"Q�Xc�A�*

loss=>>�A�f       �	��Q�Xc�A�*

lossw�>J�$j       �	�fR�Xc�A�*

loss��>{�m       �	yS�Xc�A�*

lossD�=7)|�       �	��S�Xc�A�*

loss)~m=z���       �	��T�Xc�A�*

loss��>4�V       �	i6U�Xc�A�*

loss�C�=��\c       �	$V�Xc�A�*

lossI�g=H���       �	��V�Xc�A�*

loss���=�C��       �	�FW�Xc�A�*

lossmr�=0g�X       �	z�W�Xc�A�*

loss/�D<���O       �	�zX�Xc�A�*

loss� �=�ϱ�       �	Y�Xc�A�*

loss]�>Z�}$       �	��Y�Xc�A�*

lossT@=����       �	 zZ�Xc�A�*

loss�P�=x�~       �	�[�Xc�A�*

loss�+
>T��+       �	]\�Xc�A�*

lossI��=��%:       �	ձ\�Xc�A�*

loss�9==��P�       �	�]�Xc�A�*

lossd�=�n��       �	k,^�Xc�A�*

loss:�=�J�       �	#�^�Xc�A�*

loss\��=4�b2       �	�k_�Xc�A�*

loss��=�c_       �	xA`�Xc�A�*

loss6��<-���       �	��`�Xc�A�*

lossY��=+�w�       �	�a�Xc�A�*

lossd"�=�EI       �	\<b�Xc�A�*

loss�<V��F       �	%uc�Xc�A�*

loss���=ꎎ�       �	�d�Xc�A�*

lossx�=�)q�       �	<�d�Xc�A�*

loss\��=�?�<       �	�e�Xc�A�*

loss�}�<��        �	�f�Xc�A�*

loss($>k��       �	-�g�Xc�A�*

loss1�=fdܖ       �	K�h�Xc�A�*

loss�Rw>���       �	P4i�Xc�A�*

lossA/>�1��       �	�i�Xc�A�*

loss`^=�_       �	��j�Xc�A�*

loss6��=�j��       �	[(k�Xc�A�*

loss��p=U��       �	 �k�Xc�A�*

loss��=l�Zk       �	�el�Xc�A�*

lossT9�=g�       �	�m�Xc�A�*

loss�>�R       �	ˢm�Xc�A�*

loss��\=�r�(       �	��n�Xc�A�*

loss��=A���       �	+/o�Xc�A�*

loss��=.�+       �	�Rp�Xc�A�*

loss�y�=[ݱ�       �	��p�Xc�A�*

loss@��;Tք�       �	��q�Xc�A�*

loss���<u�Ju       �	�Wr�Xc�A�*

lossW!7>��       �	��r�Xc�A�*

loss��-=����       �	L�s�Xc�A�*

lossi1�=���o       �	�$t�Xc�A�*

loss��=�2��       �	�t�Xc�A�*

loss=�M=���       �	�Vu�Xc�A�*

loss�R==�l       �	�u�Xc�A�*

losss�=����       �	~�v�Xc�A�*

loss�h=I�       �	dw�Xc�A�*

lossv=�*є       �	��w�Xc�A�*

loss��=3�|       �	J�x�Xc�A�*

lossn�z;��@       �	V,y�Xc�A�*

loss�\=�i�       �	!�y�Xc�A�*

loss�m@=Mȵ       �	�}z�Xc�A�*

loss3e�=��y       �	{�Xc�A�*

lossZO�=�hk       �	��{�Xc�A�*

lossX�>��       �	��|�Xc�A�*

loss��>ʙ��       �	.}�Xc�A�*

loss�5=>�g*       �	�}�Xc�A�*

loss\]^=y�       �	l~�Xc�A�*

lossm�=�
|�       �	D�Xc�A�*

loss�O�=w�2S       �	���Xc�A�*

loss� >����       �	�Q��Xc�A�*

lossn�>eJc�       �	N�Xc�A�*

loss��->�g�B       �	����Xc�A�*

loss$<z�Z�       �	�s��Xc�A�*

lossDm>ZZ�       �	!@��Xc�A�*

lossc�=	�       �	~��Xc�A�*

lossO�j=uV<       �	���Xc�A�*

loss��<��*       �	1C��Xc�A�*

loss_��=��$       �	�ۅ�Xc�A�*

loss��=X]A�       �	פ��Xc�A�*

losstQ=�fA	       �	)@��Xc�A�*

loss���=�Լ>       �	Z�Xc�A�*

lossm��=�iq�       �	n���Xc�A�*

lossi]1=<��$       �	!!��Xc�A�*

loss�ۑ=I;?       �	����Xc�A�*

loss�.>e~�x       �	�h��Xc�A�*

loss���=J��l       �	���Xc�A�*

loss,?=M�`�       �	����Xc�A�*

loss�{]>M��q       �	tA��Xc�A�*

loss��=�>��       �	�⌝Xc�A�*

lossMaC=5���       �	�~��Xc�A�*

loss%��=�z�W       �	3��Xc�A�*

loss�|,>�L�'       �	l���Xc�A�*

loss
�=�u1       �	vO��Xc�A�*

loss�3 =@��       �	�ꏝXc�A�*

lossj1>���       �	#���Xc�A�*

loss�d�=, �       �	&��Xc�A�*

loss��1=����       �	����Xc�A�*

lossO)(=�e+       �	sG��Xc�A�*

loss�b =�T�       �	Uޒ�Xc�A�*

loss�=�=����       �	�v��Xc�A�*

loss[B�=�t�       �	8��Xc�A�*

lossBG=�?i�       �	3���Xc�A�*

lossEC�=���9       �	5D��Xc�A�*

loss
�=[ȷ       �	G��Xc�A�*

lossΈ	>⹺\       �	旝Xc�A�*

loss��Z<8�O�       �	����Xc�A�*

loss�<��W&       �	>A��Xc�A�*

loss�{6=8�Fd       �	�癝Xc�A�*

loss��>f��^       �	����Xc�A�*

loss2]�=���       �	&��Xc�A�*

lossi�w><M�,       �	w���Xc�A�*

loss�m:>���v       �	�T��Xc�A�*

loss�pZ=�#��       �	�-��Xc�A�*

loss�>�?�       �	6͝�Xc�A�*

loss��=�"�       �	�v��Xc�A�*

loss��=��       �	���Xc�A�*

loss���<�E��       �	:���Xc�A�*

lossT�|=��=        �	�s��Xc�A�*

loss���=�K�       �	1}��Xc�A�*

loss�t=��/�       �	����Xc�A�*

loss�j>�+�v       �	�壝Xc�A�*

loss��<��H       �	���Xc�A�*

loss�B=zx��       �	�V��Xc�A�*

loss��=Ti��       �	��Xc�A�*

lossxQ=�<�{       �	����Xc�A�*

loss�_=����       �	]4��Xc�A�*

lossx�G=k��       �	i��Xc�A�*

loss�s�=i}�       �	����Xc�A�*

loss���<X24       �	�g��Xc�A�*

loss�c6>���x       �	G:��Xc�A�*

losss�>x\��       �	�ت�Xc�A�*

loss)̱=���}       �	�n��Xc�A�*

loss�Q�=�ʁ�       �	"��Xc�A�*

loss�t9=�h��       �	����Xc�A�*

loss� 1=�m��       �	�Z��Xc�A�*

loss*p=%Lh       �	8���Xc�A�*

loss�x�= 1s�       �	Ҏ��Xc�A�*

loss,D�=1KB�       �	N(��Xc�A�*

lossԂ=ȭ�>       �	ү�Xc�A�*

loss �K=H�       �	�h��Xc�A�*

loss= �=$��^       �	��Xc�A�*

loss��>?��y       �	ȴ��Xc�A�*

loss@6>�6�       �	�K��Xc�A�*

loss��=u���       �	�貝Xc�A�*

loss:�=w�=       �	q���Xc�A�*

lossփ>~�3<       �	�)��Xc�A�*

loss ck=�3�       �	ô�Xc�A�*

lossY�=�"}H       �	__��Xc�A�*

loss��=e�~       �	���Xc�A�*

loss&ơ<4�#       �	5���Xc�A�*

lossf�=ۯs>       �	i;��Xc�A�*

loss���=[��       �	�ѷ�Xc�A�*

lossO��="�c�       �	�g��Xc�A�*

lossv�=��H�       �	����Xc�A�*

loss�y==�9}       �	 ���Xc�A�*

lossr.S=�<el       �	|,��Xc�A�*

lossf �=M��3       �	�ĺ�Xc�A�*

loss���=�c �       �	�\��Xc�A�*

lossCy=<�q       �	����Xc�A�*

loss�`�=^}�       �	����Xc�A�*

lossǕ�=ZGH�       �	�*��Xc�A�*

loss�aY>V��I       �	�ͽ�Xc�A�*

lossߑ%=Uǘ�       �	�p��Xc�A�*

loss��=p4(       �	}	��Xc�A�*

loss���<�Y_       �	���Xc�A�*

loss\$�=���2       �	O=��Xc�A�*

lossT�<��0       �		���Xc�A�*

loss�VW=mR�,       �	6x��Xc�A�*

loss=Tb>���       �	�Xc�A�*

loss��<�LH       �	��Xc�A�*

loss�{�=�LG       �	��ÝXc�A�*

loss��=����       �	3�ĝXc�A�*

lossy�=#�       �	{�ŝXc�A�*

loss T=w��?       �	d�ƝXc�A�*

loss
�=?h�\       �	��ǝXc�A�*

loss���= 04`       �	1ȝXc�A�*

loss�G�=�o�       �	��ȝXc�A�*

loss&�=x�hg       �	bɝXc�A�*

loss�=`Ri?       �	L�ɝXc�A�*

loss=�N>7�r       �	x�ʝXc�A�*

loss	�>*قX       �	+0˝Xc�A�*

losstE�=*�-�       �	m�˝Xc�A�*

loss�Cl<�w�       �	c̝Xc�A�*

loss�A�<9��E       �	X ͝Xc�A�*

loss�_F=�O6�       �	9�͝Xc�A�*

loss(��=��}!       �	<ΝXc�A�*

loss�)�<�Ǟ       �	��ΝXc�A�*

lossh�=r\ܙ       �	rkϝXc�A�*

loss"=�SX       �	�НXc�A�*

lossT!�=|       �	��НXc�A�*

loss�L=�>�       �	'gѝXc�A�*

lossL�=�M��       �	�ҝXc�A�*

loss��=�=�P       �	��ҝXc�A�*

lossf
)>�˜P       �	�>ӝXc�A�*

losst)�<9S��       �	1�ӝXc�A�*

loss�c=V_�       �	;sԝXc�A�*

loss��=�TZ�       �	�՝Xc�A�*

loss͜j=S>]       �	��՝Xc�A�*

lossP�=N�+d       �	�<֝Xc�A�*

loss�;\>�pl]       �	��֝Xc�A�*

loss�l�=m�y       �	�gםXc�A�*

lossڐ�<5�%       �	��םXc�A�*

loss��y<7�~       �	��؝Xc�A�*

loss1A3=�:M[       �	xFٝXc�A�*

lossN��<���v       �	Y�ٝXc�A�*

loss:�R<5�       �	?rڝXc�A�*

loss��1>80       �	Q۝Xc�A�*

loss�r�=��       �	:�۝Xc�A�*

losse�f=����       �	�SܝXc�A�*

lossko>4�T       �	��ܝXc�A�*

loss]i�<���       �	��ݝXc�A�*

loss���=� �.       �	�3ޝXc�A�*

loss�Z1>sC�       �	S�ޝXc�A�*

loss�+�=�N�       �	}^ߝXc�A�*

loss�=F}V&       �	M�ߝXc�A�*

loss���=m��t       �	����Xc�A�*

loss��=��<c       �	(�Xc�A�*

loss�$=V�       �	׾�Xc�A�*

lossF�<�/�2       �	�X�Xc�A�*

loss%��=]'ǫ       �	���Xc�A�*

loss�p2>���A       �	���Xc�A�*

lossɒ�=�       �	a�Xc�A�*

loss���=��E(       �	��Xc�A�*

loss��s=%`�+       �	(��Xc�A�*

loss�=0�H       �	�2�Xc�A�*

loss�i=�%��       �	���Xc�A�*

loss���=��       �	�_�Xc�A�*

loss���<���       �	v6�Xc�A�*

loss�<x�       �	��Xc�A�*

loss��>Rw�       �	]�Xc�A�*

loss���<�n��       �	3��Xc�A�*

loss_��<l�       �	���Xc�A�*

loss��<=���       �	.S�Xc�A�*

loss��<�v߻       �	���Xc�A�*

lossJ>�=qS��       �	���Xc�A�*

loss��C<o%�|       �	��Xc�A�*

loss�Q>â       �	��Xc�A�*

lossÍ==��O�       �	bN�Xc�A�*

loss��=��.T       �	���Xc�A�*

loss�y>3/Z       �	�v�Xc�A�*

loss���<���       �	w�Xc�A�*

loss�{=�9��       �	D��Xc�A�*

loss�2F=Jm�R       �	YO�Xc�A�*

loss\�-=Q�[�       �	���Xc�A�*

loss�=Z��       �	�z�Xc�A�*

loss�]>�e       �	��Xc�A�*

loss1�k>�5�       �	p��Xc�A�*

loss�w�=���       �	E��Xc�A�*

loss��=�WF       �	����Xc�A�*

loss��==>"f       �	�s��Xc�A�*

loss�X�=,�Q       �	��Xc�A�*

loss��=af       �	ٱ��Xc�A�*

loss7=�#�       �	gH��Xc�A�*

loss���=��C       �	����Xc�A�*

losse=,EΙ       �	?t��Xc�A�*

loss���=��       �	���Xc�A�*

loss��=C.	       �	����Xc�A�*

loss���=�>�@       �	�H��Xc�A�*

loss���<�'2�       �	����Xc�A�*

loss[�<=w�z�       �	�o��Xc�A�*

lossQa�<y*6       �	c	��Xc�A�*

lossOL<�d�       �	���Xc�A�*

lossq�=$���       �	2>��Xc�A�*

loss���<�dq       �	����Xc�A�*

loss�}i=]�%~       �	n��Xc�A�*

loss��=�?Ć       �	��Xc�A�*

loss�q=y��3       �	ף��Xc�A�*

loss���<�Y       �	PR �Xc�A�*

loss��z=��       �	[� �Xc�A�*

loss݅�<�o.�       �	���Xc�A�*

loss�̲=9�jJ       �	�+�Xc�A�*

lossx:=��P�       �	���Xc�A�*

loss/��=�%�g       �	���Xc�A�*

loss9#=���       �	�w�Xc�A�*

lossC+�=,��       �	�t�Xc�A�*

loss:̒=79�       �	P�Xc�A�*

loss�Q�<�|�       �	��Xc�A�*

loss(��=V�       �	*U�Xc�A�*

loss,��<���T       �	T�Xc�A�*

loss�c�=�pr!       �	j��Xc�A�*

loss@O\=rNH�       �	4�	�Xc�A�*

loss��=v�G       �	t@
�Xc�A�*

lossj��=]cY1       �	��
�Xc�A�*

loss��=�鹕       �	O��Xc�A�*

lossS��<x�       �	I��Xc�A�*

loss�߄=��F       �	��Xc�A�*

lossS�=�x��       �	���Xc�A�*

loss�=�       �	�M�Xc�A�*

lossW��<7jm       �	#��Xc�A�*

loss���=�#(       �	ϡ�Xc�A�*

losso��<K*�       �	p@�Xc�A�*

lossX�}=�Zr�       �	m��Xc�A�*

lossd��<�l>�       �	E��Xc�A�*

loss��q<k]��       �	&�Xc�A�*

loss (�=RF.�       �	���Xc�A�*

lossj��<O=]�       �	�_�Xc�A�*

loss�	=���+       �	��Xc�A�*

loss,,�=r�l�       �	5��Xc�A�*

loss-�<4�e�       �	qT�Xc�A�*

loss��B;�A��       �	���Xc�A�*

loss�%�<��v       �	���Xc�A�*

lossw�<���       �	i:�Xc�A�*

loss1�<i        �	Y��Xc�A�*

lossjg�<-���       �	$}�Xc�A�*

loss���;%�:       �	["�Xc�A�*

lossx\g=��B       �	P��Xc�A�*

loss%�::RQ       �	9^�Xc�A�*

loss3%�9Z;֗       �	c��Xc�A�*

loss�b�:{�`       �	}��Xc�A�*

loss��[=�oк       �	�/�Xc�A�*

lossWq,=`{w3       �	��Xc�A�*

loss� =D�X       �	�n�Xc�A�*

loss���;E�N       �	Q�Xc�A�*

lossW�`<�M��       �	O��Xc�A�*

loss���>r<       �	���Xc�A�*

loss���;�$�        �	�� �Xc�A�*

loss{q>��>       �	vn!�Xc�A�*

loss�!j=��V       �	w"�Xc�A�	*

loss�r�=[���       �	��"�Xc�A�	*

loss��U=��4�       �	#�Xc�A�	*

loss�ΐ<��       �	E$�Xc�A�	*

loss�O=o�       �	X�$�Xc�A�	*

loss1	�=�J�s       �	�D%�Xc�A�	*

lossn>�=,|       �	��%�Xc�A�	*

loss�=��9       �	�t&�Xc�A�	*

loss�P�=V�y�       �	'�Xc�A�	*

lossj�2>b?�       �	e�'�Xc�A�	*

lossr��=�0F|       �	�i(�Xc�A�	*

loss�g�=�틕       �	�)�Xc�A�	*

loss��=MX��       �	b�)�Xc�A�	*

loss�=�ֺ�       �	=*�Xc�A�	*

loss�6d=�9f        �	{�*�Xc�A�	*

loss��=��T       �	�x+�Xc�A�	*

loss�="�       �	�,�Xc�A�	*

loss�If=-S��       �	,�,�Xc�A�	*

loss A'=?2�       �	�X-�Xc�A�	*

loss�#�=x��-       �	b�-�Xc�A�	*

loss턠=�x�u       �	��.�Xc�A�	*

loss7lX<|2�       �	//�Xc�A�	*

loss���;�V�       �	��/�Xc�A�	*

lossOj�<K�@�       �	�i0�Xc�A�	*

lossI�=���       �	�01�Xc�A�	*

loss���=z\�y       �	��1�Xc�A�	*

lossi��=��*       �	�z2�Xc�A�	*

loss��=�:3!       �	&s3�Xc�A�	*

loss=��<Y}�P       �	L4�Xc�A�	*

loss��=#7s:       �	&�4�Xc�A�	*

loss�#6=�Q}8       �	�m5�Xc�A�	*

loss�P;��r�       �	�6�Xc�A�	*

loss�2=����       �	,�6�Xc�A�	*

loss#�g<�^g9       �	�a7�Xc�A�	*

loss��q=��l�       �	f�7�Xc�A�	*

lossl|�=	�N�       �	�8�Xc�A�	*

loss�/#>X�ڞ       �	�*9�Xc�A�	*

loss8k=)Џ�       �	S�9�Xc�A�	*

loss��</5`9       �	wi:�Xc�A�	*

loss(=b�au       �	$	;�Xc�A�	*

loss�W�=7)�c       �	M�;�Xc�A�	*

loss�.�=�V        �	��<�Xc�A�	*

loss�,=��       �	�.=�Xc�A�	*

loss�J�=8l��       �	�=�Xc�A�	*

loss���=ڲ>       �	�u>�Xc�A�	*

loss"{=��
�       �	{?�Xc�A�	*

loss}��=5Q       �	�?�Xc�A�	*

loss&kK=�A>       �	�W@�Xc�A�	*

loss�[�=�f��       �	w�@�Xc�A�	*

loss��=I�g[       �	�Y�Xc�A�	*

loss���=ʇ       �	��Y�Xc�A�	*

loss�f�=��M       �	-@Z�Xc�A�	*

loss妣=��Z       �	��Z�Xc�A�	*

lossJ�=��y       �	�|[�Xc�A�	*

loss���=�I�       �	\�Xc�A�	*

loss�=�R��       �	��\�Xc�A�	*

loss
uS=.Zw	       �	�L]�Xc�A�	*

loss.>_��       �	a�]�Xc�A�	*

loss7�@>��(       �	�u^�Xc�A�	*

loss!}�<�N8�       �	h_�Xc�A�	*

loss�
=�I�       �	��_�Xc�A�	*

loss(�=��y�       �	R`�Xc�A�	*

loss!3"=.���       �	}�`�Xc�A�	*

lossw�,=���       �	��a�Xc�A�	*

loss�6v=�V�!       �	P4b�Xc�A�	*

loss#Y�;M��       �	g�b�Xc�A�	*

loss�aP=��c       �	�xc�Xc�A�	*

loss�*=<F       �	nd�Xc�A�	*

loss�E>���       �	/�d�Xc�A�	*

loss���=%�V       �	�re�Xc�A�	*

loss��C>V��N       �	~f�Xc�A�	*

loss�pe=�^��       �	��f�Xc�A�	*

lossS��=����       �	LTg�Xc�A�	*

lossJ�~=�:��       �	��g�Xc�A�	*

loss��=aK��       �	"�h�Xc�A�	*

loss��=��       �	&i�Xc�A�	*

lossa�#=�VUU       �	��i�Xc�A�	*

loss�E=�7       �	�Nj�Xc�A�	*

loss�+�=��K       �	��j�Xc�A�	*

loss�v=&�M       �	�{k�Xc�A�	*

loss��<���       �	Zl�Xc�A�	*

loss�=�=M       �	¡l�Xc�A�	*

loss��=�i�       �	�Zm�Xc�A�	*

loss��=�x�5       �	��m�Xc�A�	*

lossnBy=�=2       �	��n�Xc�A�	*

loss�xR<P��       �	�$o�Xc�A�	*

loss�2=H�0H       �	A�o�Xc�A�	*

loss܅;>�[�j       �	�Kp�Xc�A�	*

loss*��=�c*Z       �	?�p�Xc�A�	*

lossh�=��ָ       �	-yq�Xc�A�	*

lossQr�=���       �	�r�Xc�A�	*

loss(�<'.       �	�r�Xc�A�	*

loss4�w=(;�       �	kHs�Xc�A�	*

loss��=�
��       �	H�s�Xc�A�	*

lossit>�`��       �	hvt�Xc�A�	*

lossM�>=���       �	u�Xc�A�	*

loss+ҕ=�49Z       �	 �u�Xc�A�	*

losst�=	~3�       �	G>v�Xc�A�	*

loss(��;�8F       �	j�v�Xc�A�	*

lossl^< t�|       �	�pw�Xc�A�	*

lossz�=��β       �	:x�Xc�A�	*

lossOұ<TJ�8       �	
�x�Xc�A�	*

lossoH�>�s��       �	�<y�Xc�A�	*

loss'
==k"       �	��y�Xc�A�	*

loss�=���       �	lxz�Xc�A�	*

loss*�,<I��       �	�
{�Xc�A�	*

loss0�;�5�        �	��{�Xc�A�	*

loss��=Y$�5       �	�3|�Xc�A�	*

loss[W�=1�ɑ       �	��|�Xc�A�	*

loss� e>n57w       �	%[}�Xc�A�	*

loss��{=U��       �	�~�Xc�A�	*

loss,^�;dS�E       �	4�~�Xc�A�	*

loss���=�9��       �	�B�Xc�A�	*

loss��<�Јk       �	���Xc�A�	*

loss)u�;�ρ�       �	6r��Xc�A�	*

loss�i=���       �	-��Xc�A�	*

loss���<�9�       �	�܁�Xc�A�	*

loss�>���       �	���Xc�A�	*

lossԿ>Q17�       �	��Xc�A�	*

loss���<���       �	e���Xc�A�	*

loss�G�=�.�<       �	BB��Xc�A�	*

loss��<��h�       �	�ׄ�Xc�A�	*

loss�؈=m�e       �	Ln��Xc�A�	*

loss���=d�       �	��Xc�A�	*

loss��<U�~       �	u���Xc�A�	*

loss�'C=���R       �	�_��Xc�A�	*

loss�=qt�V       �	*��Xc�A�	*

loss���<�o�~       �	b���Xc�A�	*

loss�9�=#��       �	B���Xc�A�
*

lossּ�=����       �	xc��Xc�A�
*

loss��=bT�K       �	��Xc�A�
*

loss��=*�%�       �	k���Xc�A�
*

loss�_�;
N�       �	3��Xc�A�
*

loss��w<DY��       �	=Ҍ�Xc�A�
*

loss�i==�ߊ�       �	�n��Xc�A�
*

loss���=p��       �	���Xc�A�
*

loss���<�X�       �	����Xc�A�
*

loss3X�<G��o       �	�I��Xc�A�
*

loss�o?<K>��       �	;⏞Xc�A�
*

loss;R=%J�       �	��Xc�A�
*

lossAd�=�>��       �	:"��Xc�A�
*

lossWmn=�Id       �	Ǽ��Xc�A�
*

loss��h=Y�7b       �	�k��Xc�A�
*

loss���=ޅ��       �	���Xc�A�
*

lossʥd<�9��       �	���Xc�A�
*

loss�K�=MA@e       �	O��Xc�A�
*

loss��=:��       �	7��Xc�A�
*

loss��">춷Z       �	X˕�Xc�A�
*

loss�W�<�5��       �	Tp��Xc�A�
*

loss<�j=�[]       �	���Xc�A�
*

loss,]�;И�       �	����Xc�A�
*

loss�[=�Dl�       �	'P��Xc�A�
*

lossۅH<��p       �	�蘞Xc�A�
*

loss�<=���=       �	o���Xc�A�
*

loss):i=O��       �	���Xc�A�
*

loss]��<E�R+       �	����Xc�A�
*

loss��=��W�       �	�O��Xc�A�
*

loss�>B�(       �	XV��Xc�A�
*

loss�r�<�*       �	����Xc�A�
*

lossc�=��X       �	���Xc�A�
*

loss)��<ah�       �	l���Xc�A�
*

loss�Y={~       �	�F��Xc�A�
*

loss��<(�       �	Y���Xc�A�
*

loss;�;=L�Y3       �	�Ѡ�Xc�A�
*

loss\�<�׎P       �	�u��Xc�A�
*

loss̬�=A�       �	^��Xc�A�
*

loss�M=�Sr�       �	x���Xc�A�
*

loss1�*=s��3       �	�|��Xc�A�
*

loss_:<����       �	/��Xc�A�
*

loss�[�=��       �	]���Xc�A�
*

loss���=s�V�       �	&W��Xc�A�
*

loss=��=l�       �	���Xc�A�
*

lossO��=<�<V       �	����Xc�A�
*

loss�: < >�m       �	oH��Xc�A�
*

loss�Q=jn݄       �	�槞Xc�A�
*

loss�"={�       �	w���Xc�A�
*

loss�0h=�?�       �	L��Xc�A�
*

loss\��=� ?       �	�Xc�A�
*

loss�G�=���       �	����Xc�A�
*

loss��)=|�X       �	���Xc�A�
*

lossd��<6�@       �	���Xc�A�
*

loss-=j�       �	����Xc�A�
*

loss�Q=��R�       �	&��Xc�A�
*

loss��>��7H       �	�έ�Xc�A�
*

losss(=0Xz�       �	o��Xc�A�
*

loss�_=A�s�       �	O��Xc�A�
*

lossJr�<��       �	p���Xc�A�
*

loss�<���s       �	kF��Xc�A�
*

loss�w/<�>$)       �	�ᰞXc�A�
*

loss�`<�҄       �	,��Xc�A�
*

loss[�=�#��       �	m��Xc�A�
*

loss���=�ɮ�       �	����Xc�A�
*

loss�>_U��       �	�L��Xc�A�
*

loss�I�=�WԔ       �	�ೞXc�A�
*

lossxu�<:�        �	Gw��Xc�A�
*

lossl�/=��w       �	�3��Xc�A�
*

loss��<�Pe�       �	�ʵ�Xc�A�
*

lossj�.<�A#'       �	�_��Xc�A�
*

loss�=0�G�       �	���Xc�A�
*

loss�=�l       �	����Xc�A�
*

loss�==�PO       �	)��Xc�A�
*

lossI��<�FP�       �	����Xc�A�
*

loss���=)���       �	�W��Xc�A�
*

loss3v4=��m�       �	��Xc�A�
*

lossnF<�Q��       �	7���Xc�A�
*

loss�<f�(�       �	���Xc�A�
*

lossLg�=��1       �	t���Xc�A�
*

loss(/�=5��5       �	�L��Xc�A�
*

lossmR=�d�?       �	�༞Xc�A�
*

loss6.�<�ؑ�       �	���Xc�A�
*

lossIY=j��'       �	�<��Xc�A�
*

lossM��<��,       �	�Ծ�Xc�A�
*

loss�Q}<��r       �	�o��Xc�A�
*

loss�]}=	}�       �	F��Xc�A�
*

lossO(�<���V       �	����Xc�A�
*

lossJ�k<�O       �	A��Xc�A�
*

lossϐ�<�E��       �	����Xc�A�
*

lossH�]=��	       �	�pXc�A�
*

loss���=��       �	�ÞXc�A�
*

loss.�;=)ǘ       �	�ĞXc�A�
*

loss��<���       �	�ŞXc�A�
*

loss�<~���       �	<�ŞXc�A�
*

loss� >��*#       �	�sƞXc�A�
*

loss׿\<��ܣ       �	�PǞXc�A�
*

loss�)-=	M�       �	xCȞXc�A�
*

loss�=��-�       �	��ȞXc�A�
*

loss`B�=�F       �	��ɞXc�A�
*

loss�=(��       �	�TʞXc�A�
*

loss��< h��       �	B	˞Xc�A�
*

lossy�<�       �	�˞Xc�A�
*

loss�)=?Gq�       �	i9̞Xc�A�
*

loss�%={h��       �	�̞Xc�A�
*

loss;M=�       �	�u͞Xc�A�
*

loss��_<�&u`       �	ΞXc�A�
*

loss%*�=�),�       �	ۤΞXc�A�
*

loss ��<�|O7       �	�<ϞXc�A�
*

lossm2�=�Ӵ       �	{�ϞXc�A�
*

loss��<q���       �	@�ОXc�A�
*

lossc��=[�/       �	�VўXc�A�
*

loss�}�=:��N       �	��ўXc�A�
*

loss6S�<L��       �	��ҞXc�A�
*

loss��=(���       �	�ӞXc�A�
*

loss�B<�k       �	��ӞXc�A�
*

loss_��:�`�       �	�QԞXc�A�
*

loss��=��]A       �	O�ԞXc�A�
*

loss
B�<���       �	�՞Xc�A�
*

losszԅ<]��*       �	֞Xc�A�
*

loss��>�q�D       �	l�֞Xc�A�
*

loss5��<�q�j       �	nlמXc�A�
*

lossp2�=���~       �	�؞Xc�A�
*

losssLP<�`.W       �	��؞Xc�A�
*

loss��>=m�       �	�8ٞXc�A�
*

loss��~=i��       �	_�ٞXc�A�
*

loss�E=K       �	��ڞXc�A�
*

loss?H�<_u6�       �	n۞Xc�A�
*

loss}��=���       �	��۞Xc�A�
*

loss6j�=R��       �	�DܞXc�A�*

loss1�4=��ru       �	�ܞXc�A�*

loss�l�=W��       �	6vݞXc�A�*

loss���=�==-       �	�ޞXc�A�*

lossV�g=�R       �	��ޞXc�A�*

loss�4�<�y�       �	l�ߞXc�A�*

loss���<��       �	�0��Xc�A�*

loss q�<�2X       �	����Xc�A�*

lossa�_=TD�N       �	Lo�Xc�A�*

loss��*>E��`       �	)�Xc�A�*

loss��=�ϡ)       �	ū�Xc�A�*

loss$��=��)l       �	�W�Xc�A�*

loss��_=�G�       �	&�Xc�A�*

loss-�}<���       �	ʦ�Xc�A�*

loss��=��       �	�X�Xc�A�*

loss��1=�ռ       �	�Xc�A�*

lossu?�=�˄       �	��Xc�A�*

loss��;
�Qz       �	�j�Xc�A�*

lossV�=7��       �	
�Xc�A�*

loss%�v=a_e~       �	��Xc�A�*

loss���=�Q��       �	�r�Xc�A�*

loss(w=�RW3       �	�Xc�A�*

loss2b=�oX       �	A��Xc�A�*

loss3Qt<�94�       �	*T�Xc�A�*

loss���<��dc       �	���Xc�A�*

loss巹<i��       �	��Xc�A�*

loss�2�=}�K�       �	</�Xc�A�*

loss��=��       �	c��Xc�A�*

loss�=ֈ�       �	Tr�Xc�A�*

loss��=��!       �	��Xc�A�*

loss��>ν��       �	���Xc�A�*

loss���=�       �	Di�Xc�A�*

loss`�-=��/�       �	��Xc�A�*

lossa��<�{F�       �	���Xc�A�*

loss;�j=���       �	���Xc�A�*

loss���=���       �	H6�Xc�A�*

loss��<%�{       �	���Xc�A�*

loss��=@�8       �	;���Xc�A�*

loss�Ί=w�I.       �	\��Xc�A�*

loss�y=$ȶ-       �	���Xc�A�*

lossҊ�<e�       �	���Xc�A�*

loss#)�<%�uk       �	ka��Xc�A�*

loss}_	=��#�       �	���Xc�A�*

loss禊<4�)�       �	����Xc�A�*

lossR��=>���       �	Z��Xc�A�*

lossO6=X~�#       �	]���Xc�A�*

loss�k$<����       �	���Xc�A�*

loss��<��z       �	����Xc�A�*

lossT#m<Y���       �	$��Xc�A�*

loss�2#=��o�       �	����Xc�A�*

lossÑ�<c�t�       �	@���Xc�A�*

loss͢�<����       �	�G��Xc�A�*

loss*��<MM�       �	����Xc�A�*

loss�=���;       �	\���Xc�A�*

loss߇<:�       �	M2 �Xc�A�*

loss�>z}/*       �	�� �Xc�A�*

loss�wo<�@�       �	Gs�Xc�A�*

loss�8�=�Չ       �	��Xc�A�*

loss��-=�Mv�       �	1��Xc�A�*

loss�`�=���A       �	�S�Xc�A�*

loss]`2=�m�       �	��Xc�A�*

loss7��=n��       �	���Xc�A�*

loss���=�b.P       �	���Xc�A�*

loss㲵=,-       �	���Xc�A�*

loss4[=���.       �	���Xc�A�*

losst[�<�u�       �	�\�Xc�A�*

loss��=�-��       �	@	�Xc�A�*

loss��=Y�       �	]�	�Xc�A�*

loss%�>!:��       �	@�
�Xc�A�*

lossm�<O!       �	Yj�Xc�A�*

loss2H%=+8:�       �	��Xc�A�*

loss��>�u�       �	[$�Xc�A�*

loss���=�|:�       �	���Xc�A�*

loss�r<YĎ       �	�]�Xc�A�*

lossIsb<��       �	L��Xc�A�*

loss(]?=����       �	l��Xc�A�*

loss#I�<WS|       �	�,�Xc�A�*

loss��=j��<       �	���Xc�A�*

lossV=�<�;�       �	�c�Xc�A�*

loss�1�=8n��       �	+��Xc�A�*

lossmҥ<�z�p       �	��Xc�A�*

loss{�r<���       �	�;�Xc�A�*

loss�
�;�I�+       �	)��Xc�A�*

loss(3<�j��       �	�d�Xc�A�*

loss�b=����       �	�\�Xc�A�*

loss|�K<�g�\       �	��Xc�A�*

loss���<-)�       �	/��Xc�A�*

loss�y<���b       �	��Xc�A�*

loss��=n&C       �	#��Xc�A�*

loss�q�<P&x       �	HQ�Xc�A�*

loss��>Z�
�       �	�j�Xc�A�*

lossD��=�V�       �	h�Xc�A�*

loss��=�ᣩ       �	��Xc�A�*

loss�R=�q[�       �	v6�Xc�A�*

loss\}�<�M�       �	���Xc�A�*

loss�H<(���       �	l_�Xc�A�*

loss�O�=���       �	���Xc�A�*

lossJ
='l�       �	%��Xc�A�*

loss4�>��Q       �	�'�Xc�A�*

lossq�;c�R9       �	]��Xc�A�*

loss�q>["�P       �	�W�Xc�A�*

loss��<�?��       �	��Xc�A�*

loss�*T=��M       �	�� �Xc�A�*

loss��=K)�       �	�-!�Xc�A�*

loss[t,=7�       �	�!�Xc�A�*

loss��=9	�       �	��"�Xc�A�*

loss��)=�:       �	�F#�Xc�A�*

loss��d=N�n6       �	�#�Xc�A�*

loss�=��       �	�$�Xc�A�*

lossX��<�9�       �	��%�Xc�A�*

loss��<��v       �	rl&�Xc�A�*

loss蚒=_f       �	�'�Xc�A�*

loss�x�<`{��       �	��'�Xc�A�*

loss_=�%�       �	A(�Xc�A�*

loss�=-�-v       �	�)�Xc�A�*

loss�V=
6�       �	~�)�Xc�A�*

loss�<Df�q       �	�C*�Xc�A�*

lossvȖ=b��K       �	t�*�Xc�A�*

loss�B�=�Tt�       �	�+�Xc�A�*

losso��<�q~       �	�,�Xc�A�*

loss�">�~�       �	!�,�Xc�A�*

lossM�k=�+��       �	�L-�Xc�A�*

loss?G�=�;�T       �	��-�Xc�A�*

lossn�)=AM�       �	j�.�Xc�A�*

loss�L<Y��r       �	� /�Xc�A�*

loss��M< ���       �	�/�Xc�A�*

loss�^�<H��U       �	�Y0�Xc�A�*

lossC�={*�       �	��0�Xc�A�*

loss�`�=��'�       �	��1�Xc�A�*

lossCΑ=��	Y       �	g*2�Xc�A�*

loss�Q=߼�       �	�2�Xc�A�*

loss6��=�^'�       �	6u3�Xc�A�*

lossK��;v̎�       �	�4�Xc�A�*

loss85�<N�O�       �	�4�Xc�A�*

lossĕQ=�RDB       �	�b5�Xc�A�*

loss!b�=<G?=       �	� 6�Xc�A�*

lossω>.��       �	��6�Xc�A�*

lossI��=�i|�       �	Q7�Xc�A�*

loss�x�=���       �	5�7�Xc�A�*

loss �4=�Ge�       �	ӄ8�Xc�A�*

loss�� =/�"       �	�9�Xc�A�*

loss71f=�z��       �	6�9�Xc�A�*

loss��=�D��       �	`Y:�Xc�A�*

loss��O<2�       �	�:�Xc�A�*

loss�8�<d�}�       �	ӡ;�Xc�A�*

loss�K >��?�       �	1A<�Xc�A�*

lossd1Y=�	(�       �	^�<�Xc�A�*

loss�Wl=�(1C       �	'k=�Xc�A�*

lossH#<�T�       �	>�Xc�A�*

loss	א<�M�       �	˜>�Xc�A�*

loss�=g�       �	<0?�Xc�A�*

loss�`T=H,̬       �	`�?�Xc�A�*

loss�!=���       �	�a@�Xc�A�*

loss<=G�       �	��@�Xc�A�*

loss�=��n�       �	�A�Xc�A�*

loss��<e��@       �	Z*B�Xc�A�*

loss�A`>��S�       �	��B�Xc�A�*

loss�FL=����       �	��C�Xc�A�*

loss�P�<��;       �	�SD�Xc�A�*

loss�7�=�R       �	�E�Xc�A�*

loss���=��8       �	`�E�Xc�A�*

lossI�=n�FV       �	�GF�Xc�A�*

loss���<��}�       �	"�F�Xc�A�*

losszA=�%&[       �	_}G�Xc�A�*

loss�=��ٿ       �	�'H�Xc�A�*

loss�==s�$�       �	��H�Xc�A�*

loss F=.T       �	�fI�Xc�A�*

loss3 �<���`       �	�J�Xc�A�*

loss�<>���        �	��J�Xc�A�*

loss�4�<)W=�       �	4HK�Xc�A�*

loss)G�=��O�       �	��K�Xc�A�*

lossxx$=�$"y       �	8�L�Xc�A�*

lossсn=A�7       �	M�Xc�A�*

loss6�<��       �	��M�Xc�A�*

loss�"�<n���       �	�lN�Xc�A�*

loss�b�=��)�       �	#O�Xc�A�*

lossq� =KS"�       �	q�O�Xc�A�*

loss3�<�ȵ       �	EP�Xc�A�*

loss���=��OP       �	Z�P�Xc�A�*

lossHu�=�+�u       �	�oQ�Xc�A�*

loss�ns=˓[�       �	� R�Xc�A�*

loss`�<M�5       �	O�R�Xc�A�*

lossƧ�<�w��       �	�+S�Xc�A�*

loss�Bn=���       �	U�S�Xc�A�*

loss��E=TN�       �	�VT�Xc�A�*

lossE��=�-�       �	��T�Xc�A�*

loss��s=q/�r       �	�U�Xc�A�*

loss���<�7}�       �	GV�Xc�A�*

loss���=�Pے       �	��V�Xc�A�*

loss!�p<��~�       �	CuW�Xc�A�*

loss��< E��       �	�X�Xc�A�*

loss�=��KY       �	��X�Xc�A�*

loss���<�.       �	AY�Xc�A�*

loss\�A<��k       �	Q�Y�Xc�A�*

loss�n)=u�{       �	�Z�Xc�A�*

lossX�=�(r       �	P4[�Xc�A�*

loss�O=�8xj       �	��[�Xc�A�*

loss�<�=�&��       �	�f\�Xc�A�*

loss<WL<ir       �	� ]�Xc�A�*

loss���=�%`       �		�]�Xc�A�*

loss���=���&       �	�@^�Xc�A�*

loss6��=Ӏ�i       �	��^�Xc�A�*

lossO��=��}       �	@�_�Xc�A�*

loss�/r=8��W       �	� `�Xc�A�*

loss��>��'�       �	sa�Xc�A�*

loss��=g�       �	ʩa�Xc�A�*

loss�G=J�V�       �	RGb�Xc�A�*

loss�=�l|       �	��b�Xc�A�*

lossۜ=U�m       �	]�c�Xc�A�*

loss�+<�>�       �	x)d�Xc�A�*

loss�y<X�,L       �	��d�Xc�A�*

loss2�=L���       �	OYe�Xc�A�*

lossr�=��       �	3�e�Xc�A�*

loss\�=> �       �	�f�Xc�A�*

lossW��=	ԕ       �	�\g�Xc�A�*

loss)��;�^�&       �	.�g�Xc�A�*

loss`�>'H0       �	N�h�Xc�A�*

loss��=��Ƕ       �	q:i�Xc�A�*

loss%X%>���       �	��i�Xc�A�*

loss��)=�6U�       �	��j�Xc�A�*

loss�ш=�H��       �	�$k�Xc�A�*

loss	;�<^L�       �	��k�Xc�A�*

loss7��<e�       �	�Zl�Xc�A�*

loss���=����       �	��l�Xc�A�*

loss&6+=ۚЦ       �	X�m�Xc�A�*

loss�{<�$�       �	�Pn�Xc�A�*

loss���<��K       �	��n�Xc�A�*

loss&iR=��Z=       �	��o�Xc�A�*

losso�~=��LX       �	Zp�Xc�A�*

lossԖ:<@ug�       �	�oq�Xc�A�*

loss�Q;[�*>       �	�	r�Xc�A�*

lossc��<�4��       �	^�r�Xc�A�*

loss�+2=个T       �	�7s�Xc�A�*

loss��=.�|i       �	F�s�Xc�A�*

loss���=�ࢤ       �	�gt�Xc�A�*

lossC2=L��k       �	�u�Xc�A�*

loss_(�=���       �	Лu�Xc�A�*

loss�(X<���       �	�0v�Xc�A�*

loss��=���N       �	��v�Xc�A�*

loss��>y4�       �	�cw�Xc�A�*

loss���<O��       �	��w�Xc�A�*

lossI�=�N%       �	��x�Xc�A�*

lossX�{<���"       �	c^y�Xc�A�*

loss��=O�c       �	A�y�Xc�A�*

loss���<Wg7       �	;�z�Xc�A�*

lossn�<	a�       �	`!{�Xc�A�*

loss�U�=��!�       �	��{�Xc�A�*

lossA�B>RF�       �	*V|�Xc�A�*

lossiQ<=����       �	��|�Xc�A�*

lossE��<ˊ~�       �	٘}�Xc�A�*

loss6zr=̏�       �	�G~�Xc�A�*

loss,��;m<�       �	��~�Xc�A�*

loss��=�)2�       �	�Xc�A�*

loss D<���t       �	���Xc�A�*

loss��!<���e       �	[���Xc�A�*

loss�4�;�W�I       �	�X��Xc�A�*

loss$&>c?z�       �	/���Xc�A�*

loss�#V=H���       �	Ŏ��Xc�A�*

lossw��<����       �	�6��Xc�A�*

lossh7�<�	�       �	�у�Xc�A�*

loss
`�<s0)0       �	����Xc�A�*

loss1�G=%L�Z       �	�J��Xc�A�*

loss�,*<��       �	����Xc�A�*

loss� >�B�e       �	�Ɔ�Xc�A�*

loss��S=�Rj       �	�s��Xc�A�*

loss���=��       �	/o��Xc�A�*

loss/=��s       �	�D��Xc�A�*

loss���<���6       �	�݉�Xc�A�*

lossi�.=Rn�'       �	I���Xc�A�*

loss�"0<qR�t       �	�'��Xc�A�*

lossr��=��*�       �	����Xc�A�*

loss��<���       �	^��Xc�A�*

lossG�>�f{�       �	����Xc�A�*

loss���=z��       �	<���Xc�A�*

loss�d�=�ja       �	F��Xc�A�*

loss�<=��h�       �	v㎟Xc�A�*

lossX��=<."       �	N~��Xc�A�*

loss���=��X        �	�!��Xc�A�*

loss��=�Y�       �	����Xc�A�*

loss:��<*�C�       �	�T��Xc�A�*

loss��m=�:[       �	푟Xc�A�*

loss<�R<�
_�       �	����Xc�A�*

lossF�=kR;�       �	�$��Xc�A�*

losszcX=��1�       �	�2��Xc�A�*

loss��=�E       �	Ҕ�Xc�A�*

loss��<�s)a       �	�|��Xc�A�*

losstv!=����       �	. ��Xc�A�*

loss� >=�<�       �	����Xc�A�*

loss�<��V       �	�]��Xc�A�*

loss�N�=�c�       �	����Xc�A�*

loss���<����       �	G���Xc�A�*

loss��="09       �	pA��Xc�A�*

lossn=�x�R       �	�ޙ�Xc�A�*

loss�+�=�pC#       �	K ��Xc�A�*

loss�<��n       �	ߩ��Xc�A�*

lossx��<7$J�       �	�H��Xc�A�*

loss�<�<���       �	�휟Xc�A�*

loss�!-=�9��       �	����Xc�A�*

loss�U�<�*C       �	�-��Xc�A�*

lossĔ<b�=       �	h̞�Xc�A�*

loss�p�<Im1e       �	�l��Xc�A�*

losss&6=\��_       �	�	��Xc�A�*

loss!&�=�q�F       �	���Xc�A�*

lossҴ<U�i       �	C��Xc�A�*

loss��1=|��B       �	�ۡ�Xc�A�*

loss��\=�`�       �	Cs��Xc�A�*

loss~@=��\|       �	�
��Xc�A�*

lossT2o=U���       �	���Xc�A�*

loss���<rh��       �	7��Xc�A�*

loss�+@=}[)�       �	ڤ�Xc�A�*

loss�=��E5       �	\q��Xc�A�*

lossC�J<�˥�       �	S	��Xc�A�*

lossԕ�=E�;X       �	����Xc�A�*

loss7�<���@       �	�F��Xc�A�*

lossqݿ<��{�       �	ާ�Xc�A�*

loss?<[;��       �	�t��Xc�A�*

lossl�=<md       �	T��Xc�A�*

lossL��=�n�       �	,���Xc�A�*

loss��=��p       �	PV��Xc�A�*

lossNÜ=��5       �	s�Xc�A�*

loss���=��E�       �	����Xc�A�*

loss���=R��1       �	�5��Xc�A�*

loss|��;DkJ       �	֬�Xc�A�*

lossn"=�V�       �	�{��Xc�A�*

lossȤ-=�6	       �	O��Xc�A�*

lossJb�;��i       �	����Xc�A�*

loss��<��Ԕ       �	�f��Xc�A�*

lossl�]<���       �	>��Xc�A�*

losso�:FK�$       �	����Xc�A�*

loss��n<o�=       �	$E��Xc�A�*

losso"�<8o�!       �	6江Xc�A�*

loss��&:��k       �	����Xc�A�*

lossM|=;�2        �	6��Xc�A�*

loss1��;a:0       �	޳�Xc�A�*

loss�h`:���       �	����Xc�A�*

loss��:.jN�       �	�4��Xc�A�*

loss��=a��R       �	�׵�Xc�A�*

loss�-�<f{K       �	t��Xc�A�*

loss2]<�(s�       �	z��Xc�A�*

loss�T;S�+�       �	����Xc�A�*

loss�Z>]Yۋ       �	 ]��Xc�A�*

loss�Ui>GǪ       �	@���Xc�A�*

loss4�;��>�       �	����Xc�A�*

lossX>�N        �	3��Xc�A�*

loss��o=��'k       �	5Ӻ�Xc�A�*

lossSk�=��YO       �	�v��Xc�A�*

loss�P<��1       �	���Xc�A�*

loss.�/=_Q�       �	ȱ��Xc�A�*

loss��>W��?       �	�I��Xc�A�*

loss�lT=�>=�       �	�㽟Xc�A�*

lossHDQ<�t��       �	�~��Xc�A�*

loss��|=�u~�       �	���Xc�A�*

lossL��=���       �	����Xc�A�*

loss_U�=���-       �	�`��Xc�A�*

loss���=��=       �	��Xc�A�*

loss��=j	C�       �	t���Xc�A�*

loss�3�=��^�       �	�VXc�A�*

lossx�	>v�bq       �	��Xc�A�*

losse=d.�       �	��ßXc�A�*

lossJF=��       �	
�ğXc�A�*

loss#��=�!f       �	�vşXc�A�*

loss&�=��a]       �	vƟXc�A�*

loss�!=)���       �	.7ǟXc�A�*

lossô�</��       �	�ǟXc�A�*

lossv_=.���       �	�ȟXc�A�*

loss3�<��8"       �	�lɟXc�A�*

loss��<?��       �	�ʟXc�A�*

lossaP<�!��       �	��ʟXc�A�*

lossJ=����       �	�~˟Xc�A�*

loss��={UQ       �	̟Xc�A�*

lossD��=~�       �	y�̟Xc�A�*

loss���=�@.       �	��͟Xc�A�*

lossR�;ag}n       �	�ZΟXc�A�*

loss��=�i       �	H�ΟXc�A�*

loss���<��3�       �	iПXc�A�*

lossd';L�9       �	��ПXc�A�*

loss�]=�l       �	ΌџXc�A�*

loss��/=̳��       �	ILҟXc�A�*

loss��=��`�       �	�ҟXc�A�*

loss��<���       �	YԟXc�A�*

loss?��=X\?�       �	�՟Xc�A�*

lossid$=)�79       �	��՟Xc�A�*

loss�B�<m�g+       �	dX֟Xc�A�*

lossM��=�Z�Q       �	��֟Xc�A�*

loss���<��<�       �	��ןXc�A�*

loss�Fp<M}�*       �	�<؟Xc�A�*

lossj� =:w�       �	3mٟXc�A�*

loss6F�=Q���       �	RڟXc�A�*

loss׭
>{F�       �	h�ڟXc�A�*

loss�L�;����       �	�H۟Xc�A�*

loss�$k=��d       �	��۟Xc�A�*

loss-y=�B�       �	j�ܟXc�A�*

lossFi�<J��       �	�$ݟXc�A�*

lossۿ9=\ׁ       �	+��Xc�A�*

loss���=��h�       �	^J��Xc�A�*

loss�Q:=���+       �	����Xc�A�*

loss@X�<S�       �	=���Xc�A�*

loss�1/=!�Ι       �	�9��Xc�A�*

losst��<+�Be       �	����Xc�A�*

loss
=ħ�       �	���Xc�A�*

loss�\�<RAf       �	:��Xc�A�*

loss�>qOI�       �	4���Xc�A�*

loss�q�<g	ܬ       �	LT��Xc�A�*

loss�&�<5��P       �	����Xc�A�*

lossI�=��^�       �	)	��Xc�A�*

loss��	=�Z�       �	����Xc�A�*

loss��=(�2�       �	�i��Xc�A�*

loss��*<�q�       �	}��Xc�A�*

loss;W�=��^�       �	e���Xc�A�*

loss��M;>:�       �	�I��Xc�A�*

loss8��;��@       �	?���Xc�A�*

loss���<��       �	|��Xc�A�*

loss�H:=0���       �	] �Xc�A�*

loss���<XBS�       �	� �Xc�A�*

lossM�P>T��       �	 ��Xc�A�*

lossg�<��(�       �	��Xc�A�*

loss���=e�F0       �	�2�Xc�A�*

lossi�m<�wkL       �	���Xc�A�*

loss���<�K��       �	���Xc�A�*

lossn�e<Z�       �	sJ�Xc�A�*

loss�`<R�N0       �	���Xc�A�*

loss\=,���       �	)z�Xc�A�*

loss�II=
��       �	��Xc�A�*

lossʯ�<��v       �	��Xc�A�*

loss���<���       �	$E�Xc�A�*

loss���<r�       �	���Xc�A�*

losss%O=�G�       �	��	�Xc�A�*

loss���</�
�       �	.T
�Xc�A�*

loss,�=�XP       �	��
�Xc�A�*

loss��;�T�       �	H��Xc�A�*

loss�s=�/0�       �	!�Xc�A�*

loss{�>��       �	���Xc�A�*

loss'}�=.	Q�       �	A}�Xc�A�*

loss���<~��       �	�Xc�A�*

lossd_7=��t�       �	d��Xc�A�*

lossq)�;�L�e       �	�C�Xc�A�*

loss�=Z"��       �	���Xc�A�*

loss�f�=�~�D       �	�n�Xc�A�*

lossn��=�:5�       �	;�Xc�A�*

lossN8?=���a       �	���Xc�A�*

loss��K="���       �	�;�Xc�A�*

loss,!5=4eZ�       �	H��Xc�A�*

loss<T"<�@�       �	�v�Xc�A�*

loss�J�<�5       �	��Xc�A�*

lossĄ�=L���       �	���Xc�A�*

loss[�3<�3*�       �	�Y�Xc�A�*

loss֊�=�G�       �	V��Xc�A�*

loss�3w=����       �	]��Xc�A�*

loss���;��       �	$�Xc�A�*

loss�E�;�!��       �	f��Xc�A�*

loss=��;����       �	�e�Xc�A�*

loss��+=1*�       �		��Xc�A�*

loss��3=d�A�       �	��Xc�A�*

lossI��=
��       �	P��Xc�A�*

loss�"l=����       �	)<�Xc�A�*

lossJ��<�e*�       �	'��Xc�A�*

lossq7_=[�\�       �	x�Xc�A�*

loss���<�4H�       �	H�Xc�A�*

loss�h=2�O       �	B��Xc�A�*

loss<'=iV��       �	�i�Xc�A�*

lossS�<֛�       �	� �Xc�A�*

loss��+>š�       �	���Xc�A�*

loss3j=M���       �	3 �Xc�A�*

loss%Ġ<�xd�       �	�� �Xc�A�*

loss�2�=��       �	[!�Xc�A�*

loss�ڛ=Ę�       �	��!�Xc�A�*

lossFޥ<�F�       �	��"�Xc�A�*

loss���<2<2�       �	 #�Xc�A�*

losse��<^��Z       �	%�#�Xc�A�*

loss`�<�_
,       �	1�$�Xc�A�*

loss@ &=;0�       �	p�&�Xc�A�*

lossI�9=g���       �	o'�Xc�A�*

lossE�<���       �	h$(�Xc�A�*

loss[�=ׁ�#       �	� *�Xc�A�*

loss��=�d1�       �	��*�Xc�A�*

loss���=Y��       �	�'+�Xc�A�*

lossQ��;j�       �	x_,�Xc�A�*

loss$�=J1�       �	� -�Xc�A�*

lossE��<��g�       �	a�-�Xc�A�*

loss;ӕ<����       �	T�.�Xc�A�*

lossC�\<m�!       �	_/�Xc�A�*

loss�:�<t2ג       �	��/�Xc�A�*

loss�x�;Eà       �	3�0�Xc�A�*

loss}�R<��Om       �	xF1�Xc�A�*

loss|\=��        �	��1�Xc�A�*

lossN <��       �	'�2�Xc�A�*

loss�&<���       �	1#3�Xc�A�*

lossd�=JNl�       �	,�3�Xc�A�*

lossQ�<�|d       �	>u4�Xc�A�*

loss��P=$�W�       �	�5�Xc�A�*

loss.��<�%�       �	w�5�Xc�A�*

lossP�=~ì�       �	�P6�Xc�A�*

loss�s<X�FW       �	1�6�Xc�A�*

loss/r�<X=��       �	�7�Xc�A�*

loss�y�<B~1I       �	o)8�Xc�A�*

lossN�=Q��       �	�A9�Xc�A�*

lossW6)=lnj       �	��9�Xc�A�*

lossl\y=��       �	��:�Xc�A�*

lossP�=����       �	G:;�Xc�A�*

loss	��<��8�       �	�;�Xc�A�*

lossR=P��       �	
�<�Xc�A�*

loss��>>��+N       �	� >�Xc�A�*

lossϢ�<'�,v       �	V�>�Xc�A�*

loss�	=:c:       �	ML?�Xc�A�*

loss�H
=���=       �	��?�Xc�A�*

lossa��=�27K       �	�w@�Xc�A�*

loss�/�;�8�       �	�@A�Xc�A�*

loss�k=\�U�       �	B�Xc�A�*

loss;=��k�       �	c�B�Xc�A�*

loss�<�i�        �	'LC�Xc�A�*

lossVױ="C�       �	)�C�Xc�A�*

loss�D:<���       �	 �D�Xc�A�*

loss�(�<Χ�       �	�+E�Xc�A�*

lossnڠ<\��       �	��E�Xc�A�*

loss�<<�بW       �	+iF�Xc�A�*

loss�p�<���       �	��F�Xc�A�*

loss�pR=��       �	�G�Xc�A�*

loss�<}�8v       �	�<H�Xc�A�*

losss=���n       �	)�H�Xc�A�*

loss�E=.�s       �	��I�Xc�A�*

loss�Ό<k;!�       �	�7J�Xc�A�*

lossZ�g=��       �	��J�Xc�A�*

loss{��=n��       �	c|K�Xc�A�*

loss�=l9W�       �	mL�Xc�A�*

loss��<��`�       �	��L�Xc�A�*

loss��<����       �	�YM�Xc�A�*

loss�=��g       �	��M�Xc�A�*

loss���=�Q1�       �	ҋN�Xc�A�*

loss�"=��F	       �	 7O�Xc�A�*

loss-D=b��*       �	��O�Xc�A�*

loss8�<<X�I�       �	�uP�Xc�A�*

lossc<���`       �	�Q�Xc�A�*

lossR4�;! Y�       �	�Q�Xc�A�*

loss�;T0zu       �	&QR�Xc�A�*

loss�m�<b��6       �	_�R�Xc�A�*

loss���=;~�       �	=�S�Xc�A�*

loss@�=�ɛW       �	�T�Xc�A�*

lossN��<��WF       �	�T�Xc�A�*

lossC��<K��       �	.VU�Xc�A�*

loss�z;=YR,X       �	�U�Xc�A�*

losso5<g �       �	��V�Xc�A�*

loss�=EI��       �	�W�Xc�A�*

loss�{�<	Ӓ       �	G�W�Xc�A�*

loss��*<׭Z       �	�=X�Xc�A�*

loss:�<���u       �	��X�Xc�A�*

loss���=v�c�       �	�nY�Xc�A�*

loss�}@=�K�|       �	�Z�Xc�A�*

loss��=פ��       �	ղZ�Xc�A�*

loss(��;���K       �	�V[�Xc�A�*

loss%;H<�I�_       �	��[�Xc�A�*

loss��=���       �	��\�Xc�A�*

loss͖�=CM�       �	�C]�Xc�A�*

lossMC^=�'{�       �	��]�Xc�A�*

loss���=P��       �	mr^�Xc�A�*

loss��N<�$��       �	
_�Xc�A�*

loss�^=�K�       �	ۦ_�Xc�A�*

loss��N;���       �	M`�Xc�A�*

loss���=[��$       �	��`�Xc�A�*

loss6�>=����       �	ݖa�Xc�A�*

loss�#�;,�       �	�8b�Xc�A�*

lossSZ�=�\�S       �	��b�Xc�A�*

lossO�x<�y�       �	�qc�Xc�A�*

lossCj2<h��       �	x	d�Xc�A�*

loss_�=��06       �	��d�Xc�A�*

loss�v\<��9       �	`?e�Xc�A�*

loss��Z<Χ�%       �	b�e�Xc�A�*

loss�F�<U��       �	-zf�Xc�A�*

loss�=�x�       �	�g�Xc�A�*

lossM��=s�d0       �	��g�Xc�A�*

loss4�=�ϒg       �	�Mh�Xc�A�*

loss�4=��it       �	��h�Xc�A�*

lossH�:=ְ�7       �	w�i�Xc�A�*

lossS-<���U       �	�Ij�Xc�A�*

loss��<�iֱ       �	��j�Xc�A�*

loss��V=�?��       �	��k�Xc�A�*

loss�@=��2�       �	�+l�Xc�A�*

loss�E�;� O       �	V�l�Xc�A�*

loss)c<�
e�       �	�sm�Xc�A�*

loss���;�I��       �	xn�Xc�A�*

lossV`=��b�       �	]�n�Xc�A�*

lossv<�J�       �		No�Xc�A�*

loss�|�<��d�       �	�o�Xc�A�*

loss;xU<,r�I       �	z�p�Xc�A�*

loss�:>�c�       �	&q�Xc�A�*

loss�I�<]ذ       �	��q�Xc�A�*

lossh��;;��       �	Z�r�Xc�A�*

lossC��;�55       �	�Cs�Xc�A�*

loss�f&<����       �	��s�Xc�A�*

loss�<�-.�       �	��t�Xc�A�*

loss��T<Xe��       �	>u�Xc�A�*

loss-M�<~�.       �	��u�Xc�A�*

loss�c=��       �	��v�Xc�A�*

loss��(<0��p       �	�@w�Xc�A�*

loss�;x=��	�       �	�w�Xc�A�*

loss�ɟ<�$�+       �	{x�Xc�A�*

loss��H=my�       �	F#y�Xc�A�*

loss�R�<o�3       �	�y�Xc�A�*

lossQ��<E~��       �	cz�Xc�A�*

loss4\}<��Q�       �	!{�Xc�A�*

losst��=��v�       �	�{�Xc�A�*

lossV��<��.       �	~U|�Xc�A�*

loss�ל<�"��       �	Q�|�Xc�A�*

loss; �<����       �	�}�Xc�A�*

lossE/w<��$y       �	�4~�Xc�A�*

loss��<��k�       �	��~�Xc�A�*

loss�`@<�X�       �	Mh�Xc�A�*

loss�C�<���       �	���Xc�A�*

loss�ʂ=�}k4       �	���Xc�A�*

loss7��=��A       �	�:��Xc�A�*

loss�pO>��r�       �	NЁ�Xc�A�*

loss��>_g��       �	<f��Xc�A�*

loss��1=q�һ       �	���Xc�A�*

lossDt=[�N�       �	����Xc�A�*

loss�(�<1"       �	3��Xc�A�*

loss�"=��?       �	�ʄ�Xc�A�*

loss�<��v       �	0���Xc�A�*

lossf0�<��s       �	��Xc�A�*

loss�T�:gB*       �	&���Xc�A�*

loss� =����       �	+/��Xc�A�*

loss��<p� <       �	2v��Xc�A�*

loss᜹<2��,       �	�L��Xc�A�*

lossn�<��T       �	����Xc�A�*

loss��5=��h       �	����Xc�A�*

loss�Y�;h�:`       �	q8��Xc�A�*

loss<��< I�E       �	ލ�Xc�A�*

loss�s�=�V��       �	v��Xc�A�*

lossvʯ<��G�       �	���Xc�A�*

loss�n2=�a�K       �	W���Xc�A�*

losss}�<����       �	gH��Xc�A�*

loss��X=N��\       �	`䐠Xc�A�*

loss���=��L       �	�|��Xc�A�*

loss���=�#J	       �	a��Xc�A�*

loss��<��)4       �	T���Xc�A�*

loss��]<��2�       �	G��Xc�A�*

loss,��<����       �	�䓠Xc�A�*

loss��=���       �	����Xc�A�*

loss��=%q       �	�e��Xc�A�*

loss�Е=�qT�       �	?��Xc�A�*

loss��4<6��6       �	o���Xc�A�*

loss 1�<}��       �	�<��Xc�A�*

loss� �<+��$       �	/ܗ�Xc�A�*

loss�9�=�RNF       �	�}��Xc�A�*

loss�ܻ<��       �	���Xc�A�*

lossl�l=�@�S       �	%���Xc�A�*

loss���<K���       �	zT��Xc�A�*

loss�o�=E)��       �	�횠Xc�A�*

loss��:0�J       �	���Xc�A�*

loss�7<<�z>       �	���Xc�A�*

loss�s�<y���       �	^���Xc�A�*

loss��<�
��       �	�O��Xc�A�*

loss��%=W�;�       �	��Xc�A�*

loss\/j<m��h       �	A���Xc�A�*

loss6D1=�;*       �	�0��Xc�A�*

loss6��=��j       �	~埠Xc�A�*

loss{&B<��d�       �	����Xc�A�*

loss�܂=����       �	�=��Xc�A�*

lossP=<���       �	�ء�Xc�A�*

loss��T=�ϽE       �	Fy��Xc�A�*

loss�_<F�7       �	��Xc�A�*

loss�%/=Ճ=�       �	�ģ�Xc�A�*

loss�=�ք�       �	�a��Xc�A�*

loss`��=���       �	���Xc�A�*

loss��j=�"��       �	褥�Xc�A�*

losshV=�sџ       �	X��Xc�A�*

lossJ�<��O       �	��Xc�A�*

loss$�q=���       �	���Xc�A�*

loss�.�;z>^�       �	y먠Xc�A�*

loss-m/=<��T       �	'���Xc�A�*

lossfp�=l�       �	i6��Xc�A�*

loss$9<�B�)       �	1��Xc�A�*

lossRܧ;q�KM       �	9���Xc�A�*

lossؗ=֕��       �	�Z��Xc�A�*

lossQ�@=Z��*       �	���Xc�A�*

loss&�<����       �	���Xc�A�*

loss/�J<�@e       �	hZ��Xc�A�*

loss���<m��       �	 �Xc�A�*

lossĎ�<�V�       �	Ș��Xc�A�*

loss�i=�?�       �	<��Xc�A�*

loss��<gN��       �	�ᰠXc�A�*

loss��x=�X�d       �	z��Xc�A�*

loss?c<��Q�       �	/��Xc�A�*

lossU=zZ+�       �	G���Xc�A�*

lossK::�       �	�I��Xc�A�*

loss��;���+       �	��Xc�A�*

lossN^T=AR��       �	Ŏ��Xc�A�*

loss���:H#�T       �	�4��Xc�A�*

loss�v�;��%�       �	����Xc�A�*

loss��d;�{�       �	`���Xc�A�*

losso)6=3 -6       �	�8��Xc�A�*

loss�"=��io       �	�ٷ�Xc�A�*

loss�>�õ       �	u��Xc�A�*

loss��=�7-�       �	���Xc�A�*

losseN�<;[�       �	����Xc�A�*

loss�,<xe@K       �	NF��Xc�A�*

loss�E=Ǵ	       �	v޺�Xc�A�*

loss��p<I�       �	���Xc�A�*

loss�i�=F���       �	&���Xc�A�*

loss�9�=a�       �	7���Xc�A�*

loss��L>� �       �	cE��Xc�A�*

loss��<���       �	羠Xc�A�*

loss�=���       �	ǟ��Xc�A�*

loss�$E;.�:[       �	0J��Xc�A�*

lossBA=����       �	4���Xc�A�*

loss��0<���       �	j���Xc�A�*

lossn�3='�L       �	< Xc�A�*

lossQ�?=�d�       �	Q� Xc�A�*

loss�J�<s��Y       �	}àXc�A�*

loss;B�<޶       �	h%ĠXc�A�*

loss;�=>���       �	��ĠXc�A�*

loss��;#q�|       �	jlŠXc�A�*

loss�ң<�f�       �	=ƠXc�A�*

loss�R*=�0!u       �	H�ƠXc�A�*

lossIΡ<���q       �	��ǠXc�A�*

loss�4<H�|�       �	x(ȠXc�A�*

loss�>�\x�       �	.�ȠXc�A�*

lossxd&=i
�=       �	�fɠXc�A�*

lossnؤ<)u��       �	NʠXc�A�*

loss��<�*y       �	�ʠXc�A�*

lossh�=xm�       �	�>ˠXc�A�*

loss�<����       �	��ˠXc�A�*

lossi4i=��S       �	/o̠Xc�A�*

loss���<����       �	͠Xc�A�*

loss�	�=k��       �	s�͠Xc�A�*

loss��<)lY2       �	�7ΠXc�A�*

loss. �:@}w       �	��ΠXc�A�*

loss�W�<N��Q       �	\qϠXc�A�*

lossw5==�E�,       �	�РXc�A�*

loss	��=��zH       �	"�РXc�A�*

loss��=**S�       �	�CѠXc�A�*

lossXE=u��(       �	y�ѠXc�A�*

lossR�8<7�       �	%zҠXc�A�*

loss1=.�8       �	�ӠXc�A�*

lossX�O;��3       �	�ӠXc�A�*

loss]q<� c       �	�JԠXc�A�*

loss�� =9��       �	�ԠXc�A�*

loss]�/<���       �	�ՠXc�A�*

loss��.=��0F       �	-`֠Xc�A�*

lossLP>
��       �	l
נXc�A�*

loss\;=sC3�       �	�נXc�A�*

loss.;�<s�       �	�MؠXc�A�*

loss3��<��2E       �	0�ؠXc�A�*

lossʍF=u�,       �	L�٠Xc�A�*

lossR5=-g�z       �	�+ڠXc�A�*

lossC��<B��o       �	��ڠXc�A�*

loss�O={.Q�       �	�^۠Xc�A�*

loss�m�=�㻷       �	��۠Xc�A�*

lossN��<+�(�       �	ٙܠXc�A�*

loss�[�=Y���       �	^.ݠXc�A�*

loss���<�	��       �	��ݠXc�A�*

loss��=q
lS       �	�_ޠXc�A�*

loss��<��Pw       �	&�ޠXc�A�*

loss$=�j��       �	K�ߠXc�A�*

lossN��<���       �	�D�Xc�A�*

loss&�=VVl�       �	|��Xc�A�*

lossiӕ<>��       �	t��Xc�A�*

lossi�}<VQ�l       �	�A�Xc�A�*

lossgӡ=�os�       �	��Xc�A�*

loss�=l�o       �	e��Xc�A�*

loss2ł=��       �	0.�Xc�A�*

loss�=����       �	9��Xc�A�*

loss3��<����       �	F|�Xc�A�*

lossCdn<��T       �	�#�Xc�A�*

loss�yk=�aNa       �	��Xc�A�*

lossz4>4�^       �	�s�Xc�A�*

loss;�=�[�       �	Q�Xc�A�*

loss�z�=
��;       �	p��Xc�A�*

loss
��;Y�       �	���Xc�A�*

lossq��<Udu�       �	���Xc�A�*

loss\�F=�Ǥ       �	�#�Xc�A�*

lossg=Ri       �	���Xc�A�*

loss���;`��       �	���Xc�A�*

loss&t�<�ʑ       �	����Xc�A�*

lossx��=��       �	�H�Xc�A�*

loss��H<h
:j       �	}��Xc�A�*

loss}��<��       �	���Xc�A�*

loss�O>�       �	�N�Xc�A�*

lossMQ=$D�e       �	���Xc�A�*

lossc�;��%       �	h��Xc�A�*

loss�zF=2Ҋ�       �	</�Xc�A�*

lossП=�Zp�       �	=��Xc�A�*

loss��Y=執j       �	f��Xc�A�*

lossZ��<�}       �	40��Xc�A�*

loss;��;��       �	9���Xc�A�*

loss�$=МƬ       �	�s��Xc�A�*

loss/��<�ss�       �	L��Xc�A�*

loss�Ċ<vIǭ       �	����Xc�A�*

lossg�<f���       �	�f��Xc�A�*

loss�)=�)u       �	
��Xc�A�*

lossO��<٨"�       �	Q���Xc�A�*

loss�а;x���       �	�D��Xc�A�*

loss+�<�'�       �	;��Xc�A�*

losso��<�/)�       �	,���Xc�A�*

loss��=�l?a       �	8L��Xc�A�*

lossM�<�\�G       �	����Xc�A�*

lossLw�<ءNq       �	z���Xc�A�*

lossʓx=���       �	s+��Xc�A�*

loss6 �;Ř;s       �	����Xc�A�*

losse] >l��       �	�g��Xc�A�*

loss�ū<�uJ       �	#��Xc�A�*

losssr&=ڃVb       �	����Xc�A�*

lossvW�=x���       �	�b �Xc�A�*

loss���<�r0�       �	Q� �Xc�A�*

loss�S=D`�       �	[��Xc�A�*

loss+�=�]�       �	�.�Xc�A�*

loss�M�=߀��       �	���Xc�A�*

loss��<��w�       �	p`�Xc�A�*

loss���=a
�V       �	3�Xc�A�*

loss7�=|t�       �	���Xc�A�*

loss �I=�'%       �	�f�Xc�A�*

loss�< �˔       �	���Xc�A�*

loss;B�:;܊       �	5��Xc�A�*

loss��@=����       �	�3�Xc�A�*

loss3�T=Ӟy�       �	�Xc�A�*

loss<��;����       �	c��Xc�A�*

loss���=��g       �	ka	�Xc�A�*

loss���;"�"       �	#�	�Xc�A�*

loss\�=�4k       �	2�
�Xc�A�*

loss��=�s�L       �	
-�Xc�A�*

loss���<R�       �	���Xc�A�*

loss,�E=�0�?       �	$d�Xc�A�*

loss���=H���       �	���Xc�A�*

loss|f<H��       �	2��Xc�A�*

loss�o=�y�m       �	]O�Xc�A�*

lossF=v!	       �	%��Xc�A�*

loss���<���       �	���Xc�A�*

loss'ܟ<���       �	u"�Xc�A�*

lossH&�<ۍ��       �	���Xc�A�*

lossj�=^��       �	_�Xc�A�*

lossW��<�lp|       �	��Xc�A�*

lossR]<��       �	��Xc�A�*

loss��<�R�D       �	9C�Xc�A�*

lossqԄ=)P��       �	L��Xc�A�*

loss��P<���       �	�y�Xc�A�*

loss%~=))b�       �	��Xc�A�*

lossfG>KQ�       �	d��Xc�A�*

loss�=&[2�       �	@Q�Xc�A�*

loss�v�=���       �	���Xc�A�*

loss��<D�i�       �	��Xc�A�*

lossڢ<>~>s       �	�#�Xc�A�*

loss�:�=1�!�       �	T��Xc�A�*

loss�ћ<y^R�       �	�a�Xc�A�*

loss�}Z=���f       �	��Xc�A�*

loss�}�=$��y       �	̛�Xc�A�*

lossc�<G��       �	�3�Xc�A�*

loss抓<�y8@       �	���Xc�A�*

loss��<!��       �	o�Xc�A�*

loss<]R=�wF       �	��Xc�A�*

loss}�@=ڃ��       �	���Xc�A�*

loss#=��
       �	A��Xc�A�*

loss�KJ=$��       �	��Xc�A�*

loss��=�AN>       �	.��Xc�A�*

lossa	s<"�       �	� �Xc�A�*

lossӋ;=ܚ�       �	w0!�Xc�A�*

loss�V�;��S       �	��!�Xc�A�*

loss�~<�7�@       �	�p"�Xc�A�*

loss��<��R@       �	�#�Xc�A�*

lossV��=�M�       �	T�#�Xc�A�*

loss�?<vT2m       �	_C$�Xc�A�*

lossT+=��h�       �	��$�Xc�A�*

loss[HH<�r�i       �	��%�Xc�A�*

lossop�;�c(X       �	E*&�Xc�A�*

loss�=�"N'       �	�'�Xc�A�*

loss�,�<(|�Q       �	�'�Xc�A�*

lossJ��=*6oJ       �	|D(�Xc�A�*

loss�H<���       �	��(�Xc�A�*

loss�=7�       �	؀)�Xc�A�*

loss�Y�=
K       �	*�Xc�A�*

loss�p�;6i       �	��*�Xc�A�*

loss�L;���       �	�`+�Xc�A�*

loss���;�D�       �	{�+�Xc�A�*

lossM�{<�Ż�       �	��,�Xc�A�*

loss�<�#�;       �	s*-�Xc�A�*

loss&҄=�p�N       �	�-�Xc�A�*

loss<�=w�:       �	��.�Xc�A�*

loss�W=F��M       �	�,/�Xc�A�*

loss�<����       �	��/�Xc�A�*

loss�g�<5��       �	c0�Xc�A�*

loss��l=(�       �	�	1�Xc�A�*

lossNaK<���]       �	4�1�Xc�A�*

loss��v<@�B�       �	�J2�Xc�A�*

loss�H�=�sv�       �	��2�Xc�A�*

loss/_:<��       �	a�3�Xc�A�*

lossa�=핹       �	1'4�Xc�A�*

lossM<<:��       �	1�4�Xc�A�*

lossO�=��       �	Hm5�Xc�A�*

loss�+�;��       �	�6�Xc�A�*

loss�h4=��       �	N�6�Xc�A�*

loss�oh;no�U       �	'k7�Xc�A�*

loss���=���T       �	�8�Xc�A�*

loss�s)=h��=       �	T�8�Xc�A�*

loss�
�;��jh       �	�@9�Xc�A�*

loss	_�<�-�       �	��9�Xc�A�*

loss���=ǻ�f       �	q:�Xc�A�*

loss���<�b��       �	|
;�Xc�A�*

loss �Y;�ז�       �	*�;�Xc�A�*

lossU}�;E�       �	-B<�Xc�A�*

loss�2=D<+�       �	��<�Xc�A�*

lossڄ�=B`:       �	y=�Xc�A�*

loss�<?�ji       �	>�Xc�A�*

loss��F=!���       �	o�>�Xc�A�*

loss��y;�a��       �	�U?�Xc�A�*

loss�s�=Ldz        �	^�?�Xc�A�*

loss�̑=!��       �	��@�Xc�A�*

lossU��:�7{'       �	&7A�Xc�A�*

loss*m=���       �	#�A�Xc�A�*

loss��=);6       �	msB�Xc�A�*

lossI<S,l       �	}!C�Xc�A�*

loss�6�<1S��       �	N�C�Xc�A�*

loss_��=׈j       �	�rD�Xc�A�*

lossl��=�`;       �	E�Xc�A�*

lossO��<��20       �	M�E�Xc�A�*

loss#�=0�'       �	`�F�Xc�A�*

loss7�=���       �	�FG�Xc�A�*

loss?�=%�I       �	�	H�Xc�A�*

loss6q<8�_�       �	%�H�Xc�A�*

lossf��;5��       �	?UI�Xc�A�*

loss1�b<�a�       �	�ZJ�Xc�A�*

loss��<�C       �	�J�Xc�A�*

loss��k=o��e       �	��K�Xc�A�*

loss�c�<��ȃ       �	�7L�Xc�A�*

losst��<q3S       �	�L�Xc�A�*

loss���<f|;       �	�M�Xc�A�*

loss�v�<;��       �	�O�Xc�A�*

loss�\�<c���       �	 �O�Xc�A�*

loss��w=ˉML       �	PP�Xc�A�*

loss.<�       �	� Q�Xc�A�*

loss��:@s�       �	��Q�Xc�A�*

loss�^�<��       �	7RR�Xc�A�*

loss|�<қJ       �	�R�Xc�A�*

loss�;� |�       �	�S�Xc�A�*

loss��r<�+E�       �	�rT�Xc�A�*

loss�<�Hգ       �	wU�Xc�A�*

lossʀ�;�E       �	��U�Xc�A�*

lossh$;�Խ       �	 �V�Xc�A�*

loss_�o9h��*       �	��W�Xc�A�*

loss/G�9�p��       �	�@X�Xc�A�*

loss�L7=�C�       �	��X�Xc�A�*

lossέ>0�X�       �	m�Y�Xc�A�*

loss��=ʑ��       �	~UZ�Xc�A�*

loss���9N�ͩ       �	s�Z�Xc�A�*

loss��;��L       �	��[�Xc�A�*

losssL>H$��       �	1�\�Xc�A�*

loss6��;m��       �	J]�Xc�A�*

lossdn=x )�       �	#�]�Xc�A�*

lossN�5<:       �	4�^�Xc�A�*

lossl�`<k3�       �	XT_�Xc�A�*

loss�K<��G       �	�Z`�Xc�A�*

loss���;��!       �	�Fa�Xc�A�*

loss]3�<�hU�       �	O�a�Xc�A�*

loss�VD=�A1       �	��b�Xc�A�*

loss�$3=����       �	F?c�Xc�A�*

loss�݊<B��       �	W�c�Xc�A�*

loss���<���       �	��d�Xc�A�*

loss8b�<?��       �	�>e�Xc�A�*

loss�G=ʛ]�       �	)�e�Xc�A�*

loss��L<�vY       �	*�f�Xc�A�*

loss�3=hXK�       �	M.g�Xc�A�*

loss��!=�L�       �	��g�Xc�A�*

lossRi&=��-       �	p�h�Xc�A�*

loss�\:<(�       �	�yi�Xc�A�*

loss�s�=�:�       �	Cj�Xc�A�*

loss��Y<�i��       �	7�j�Xc�A�*

loss� <N��       �	�nk�Xc�A�*

loss��J=ůQ�       �	l�Xc�A�*

loss�h�<�X��       �	
dm�Xc�A�*

loss� �<���+       �	<n�Xc�A�*

loss@��;��j       �	�n�Xc�A�*

lossŌ<���Z       �	�`o�Xc�A�*

loss���=a���       �	p�Xc�A�*

loss\:N<f�ć       �	+�p�Xc�A�*

lossrK�=�a$�       �	nLq�Xc�A�*

loss6�=(�       �	S�q�Xc�A�*

loss,�$;"�       �	��r�Xc�A�*

lossR>�=�ٮ�       �	�3s�Xc�A�*

loss�|	;���       �	��s�Xc�A�*

loss�(�<�i9|       �	mt�Xc�A�*

loss�=)�y�       �	�!u�Xc�A�*

loss͋V<���       �	ػu�Xc�A�*

loss<�m<�#F       �	%]v�Xc�A�*

lossw%�=���       �	qw�Xc�A�*

losszu�<��/�       �	?�w�Xc�A�*

loss�_�=��       �	�Jx�Xc�A�*

loss��<���       �	��x�Xc�A�*

loss�=G7��       �	h�y�Xc�A�*

lossa�*=�o{�       �	�<z�Xc�A�*

loss�ǃ<�K�       �	�z�Xc�A�*

lossI�.=���       �	R�{�Xc�A�*

loss��T=��)a       �	�R|�Xc�A�*

loss��=���N       �	U�|�Xc�A�*

loss�']<3�dJ       �	��}�Xc�A�*

loss��<W���       �	�J~�Xc�A�*

loss�F<�w�       �	��~�Xc�A�*

loss�C=_f       �	���Xc�A�*

loss4=k�fv       �	�6��Xc�A�*

loss�o�<�y�2       �	�̞�Xc�A�*

loss}vg=��X�       �	�؟�Xc�A�*

lossv1=/��.       �	����Xc�A�*

loss	�<�\n       �	ԙ��Xc�A�*

loss@˽<�	       �	����Xc�A�*

loss��=����       �	��Xc�A�*

loss��\=D#��       �	.���Xc�A�*

lossI��=2y�(       �	M��Xc�A�*

loss]<w=q���       �	m⤡Xc�A�*

loss�*<�d�       �	�w��Xc�A�*

loss�>��&       �	���Xc�A�*

loss@�_<�^��       �	���Xc�A�*

loss�"�=4�!       �	\���Xc�A�*

loss�Y=� !       �	�1��Xc�A�*

loss�f�<��^m       �	�ͨ�Xc�A�*

loss�4:j��       �	�n��Xc�A�*

loss�fc<;q       �	���Xc�A�*

loss��<Բ�5       �	S!��Xc�A�*

loss�b�=�3�/       �	U���Xc�A�*

lossA�2<{�       �	�W��Xc�A�*

loss7͑=��&c       �	�Xc�A�*

loss(!�;R��       �	]���Xc�A�*

lossZ�>t���       �	>&��Xc�A�*

loss�b=�A
�       �	�ʮ�Xc�A�*

loss8ֹ;S���       �	Ic��Xc�A�*

loss_�x<�T&�       �	����Xc�A�*

loss	n%=o�>       �	����Xc�A�*

loss
�v=z��       �	�S��Xc�A�*

lossT�q=�lŇ       �	�Xc�A�*

lossb
=����       �	섲�Xc�A�*

loss��D=FH       �	��Xc�A�*

lossZ��=�¦       �	�ȳ�Xc�A�*

lossF5�<��^�       �	�^��Xc�A�*

loss!�=z�&�       �	����Xc�A�*

lossF��=���       �	!���Xc�A�*

loss�9<�p
5       �	A+��Xc�A�*

lossi��<Z,       �	¶�Xc�A�*

loss "�=�Ϊ       �	yZ��Xc�A�*

loss�Q=�:�       �	���Xc�A�*

loss�X=�2��       �	���Xc�A�*

lossE�b<nN�G       �	T;��Xc�A�*

loss	�; ��0       �	Թ�Xc�A�*

loss�}T<�i�       �	����Xc�A�*

loss���<0ՠ�       �	�^��Xc�A�*

lossM�=���A       �	����Xc�A�*

loss�3�<t�       �	����Xc�A�*

lossM=���/       �	�'��Xc�A�*

lossq�F=��V?       �	�Ž�Xc�A�*

loss�)�;-�iB       �	�o��Xc�A�*

loss@L<�(V        �	��Xc�A�*

loss���=�l��       �	����Xc�A�*

loss�{�<�W�i       �	�d��Xc�A�*

loss%N�=���5       �	N��Xc�A�*

loss�=I_^�       �	���Xc�A�*

loss���;-
��       �	�O¡Xc�A�*

loss�}<���       �	��¡Xc�A�*

loss?Q�;@�ܢ       �	ɒáXc�A�*

lossN5�<,�w|       �	�-ġXc�A�*

loss_g�<��W       �	O�ġXc�A�*

loss1��=��       �	�sšXc�A�*

loss$.�<��dc       �	�!ơXc�A�*

lossta<4�"�       �	��ơXc�A�*

loss�!�<���1       �	�VǡXc�A�*

losso@<��       �	��ǡXc�A�*

loss�H�=pu4       �	 �ȡXc�A�*

loss.�I=�0�Y       �	�)ɡXc�A�*

loss��<��AP       �	��ɡXc�A�*

loss��}=R��       �	�lʡXc�A�*

loss��W<���`       �	NˡXc�A�*

lossq�h<�/Il       �	�ˡXc�A�*

lossO|�=~�t       �	kJ̡Xc�A�*

loss�T<��V       �	��̡Xc�A�*

loss�V�<��dD       �	�͡Xc�A�*

loss� =�pHB       �	v4ΡXc�A�*

loss�(�<L��       �	��ΡXc�A�*

loss��<�I�B       �	rϡXc�A�*

loss�ۜ<����       �	�СXc�A�*

losso��<��t�       �	��СXc�A�*

loss���=���o       �	�DѡXc�A�*

loss�G�=j���       �	w�ѡXc�A�*

loss�=��Rs       �	ÁҡXc�A�*

loss�}<q[2�       �	zӡXc�A�*

loss�k�<Oŵ&       �	/�ӡXc�A�*

lossE�,<G�       �	!YԡXc�A�*

loss�� <u��       �	�dաXc�A�*

losse�=���?       �	��աXc�A�*

lossiV=2�=       �	K�֡Xc�A�*

loss�<�<���q       �	�;סXc�A�*

loss�.=b�4       �	��סXc�A�*

loss�B�<�\�       �	%uءXc�A�*

lossR2=jL��       �	�١Xc�A�*

losstF�<-F�u       �	ݴ١Xc�A�*

loss2E�=�u$       �	UڡXc�A�*

loss���=�T       �	��ڡXc�A�*

loss`<yQ)       �	&�ۡXc�A�*

loss��<h�\�       �	^+ܡXc�A�*

loss�j9<�?       �	#�ܡXc�A�*

loss�<�<���b       �	��ݡXc�A�*

loss�L=��`       �	�%ޡXc�A�*

loss�e;�kn       �	�ޡXc�A�*

lossX[<��N�       �	��ߡXc�A�*

loss<�=Vݔ       �	h$�Xc�A�*

lossz��<��1       �	f��Xc�A�*

lossq*�<�ً�       �	�W�Xc�A�*

loss��;o�v�       �	�'�Xc�A�*

loss��0<y���       �	���Xc�A�*

loss���<ҫ��       �	�h�Xc�A�*

lossQd�=���~       �	��Xc�A�*

loss��[<���`       �	���Xc�A�*

loss�6=��m�       �	�H�Xc�A�*

loss�<��       �	���Xc�A�*

loss#]<=���       �	�~�Xc�A�*

lossN~5<
��       �	Q�Xc�A�*

loss�nc<b���       �	p�Xc�A�*

loss`��<����       �	Ϡ�Xc�A�*

loss 0J=���       �	?�Xc�A�*

lossq�U=UM��       �	H��Xc�A�*

loss��=8 I�       �	C��Xc�A�*

loss��{<��A       �	���Xc�A�*

loss��;n���       �	G��Xc�A�*

loss��*=��       �	�"��Xc�A�*

loss��=�jO�       �	y���Xc�A�*

loss.�<�H��       �	�u�Xc�A�*

loss��;�HD�       �	+�Xc�A�*

loss{��<r>
n       �	R��Xc�A�*

loss�-="�E�       �	\Z�Xc�A�*

lossM �<*dY�       �	
�Xc�A�*

losst�3=�JNi       �	���Xc�A�*

loss��<��$�       �	A�Xc�A�*

loss!΋<� Qj       �	���Xc�A�*

loss-,+<�z��       �	�y�Xc�A�*

loss�gd=�Y       �	n��Xc�A�*

loss��&;��       �	����Xc�A�*

loss(�4<E��       �	�]��Xc�A�*

loss�t	=�_K�       �	����Xc�A�*

loss߫3='g�       �	h���Xc�A�*

lossZW=�Y��       �	i���Xc�A�*

loss�S;��(        �	zS��Xc�A�*

losshR�<=}��       �	����Xc�A�*

loss��<�O       �	Օ��Xc�A�*

loss���<��dJ       �	�3��Xc�A�*

loss�\=>j       �	����Xc�A�*

loss�6�<�=�P       �	ލ��Xc�A�*

loss+�=׹�S       �	O@��Xc�A�*

lossc�C=���j       �	M���Xc�A�*

losse�!<�C�I       �	�q��Xc�A�*

loss�4v;ǀ��       �	���Xc�A�*

loss/�<�:�       �	O���Xc�A�*

loss{ά;���
       �	F_��Xc�A�*

loss1<͠��       �	~ �Xc�A�*

loss���<3�K-       �	� �Xc�A�*

loss<MJ=�p��       �	9_�Xc�A�*

loss�=��׈       �	��Xc�A�*

loss�<�Q�8       �	�0�Xc�A�*

lossߏ`;��>�       �	P��Xc�A�*

loss���;��Rr       �	=`�Xc�A�*

loss��<=n_�W       �	��Xc�A�*

loss]� ='L�N       �	@��Xc�A�*

loss�`;�j�[       �	2=�Xc�A�*

loss���<ry�        �	���Xc�A�*

loss7;�<9X��       �	�|�Xc�A�*

lossZ�<a>       �	�W�Xc�A�*

loss�QS;Ga       �	p?	�Xc�A�*

loss_�M=d�       �	�
�Xc�A�*

loss3��;\�(       �	��
�Xc�A�*

loss���:���#       �	a�Xc�A�*

lossܳ=1z��       �	w+�Xc�A�*

loss�|�=`��       �	`��Xc�A�*

loss���;O-{       �	�(�Xc�A�*

loss��=���       �	s��Xc�A�*

loss��;�-       �	I��Xc�A�*

loss
H-<2�5�       �	d�Xc�A�*

loss��<#=�6       �	4�Xc�A�*

loss�w<�r6�       �	c��Xc�A�*

lossDB<���U       �	�W�Xc�A�*

lossEq�<��b�       �	'��Xc�A�*

loss���=�#�f       �	,~�Xc�A�*

lossX]<���E       �	��Xc�A�*

loss�<U�%�       �	۾�Xc�A�*

loss�z�:��       �	�f�Xc�A�*

loss`��<hN�       �	B�Xc�A�*

lossW�=�m$       �	���Xc�A�*

loss~}�<lS^       �	ǀ�Xc�A�*

lossJϩ<y�[S       �	�-�Xc�A�*

loss/Ҽ;}���       �	!��Xc�A�*

loss���=�]�       �	Ra�Xc�A�*

loss�4<k<��       �	�Xc�A�*

losss��<A�5�       �	#��Xc�A�*

lossVFC=t��       �	u<�Xc�A�*

loss�^/=�}       �	���Xc�A�*

loss��<��&5       �	R��Xc�A�*

loss�9=�Hv/       �	j�Xc�A�*

loss�n�;��H�       �	�Xc�A�*

loss�zG;��}       �	���Xc�A�*

loss��<���       �	�> �Xc�A�*

loss�z�<U]��       �	V� �Xc�A�*

lossÒ�<����       �	w!�Xc�A�*

loss+�=� �       �	�"�Xc�A�*

loss:K1;+�4�       �	��"�Xc�A�*

loss�\�=L�k       �	�D#�Xc�A�*

loss�/�:�2       �	P�#�Xc�A�*

lossWyf;6�       �	Z�$�Xc�A�*

loss���;@���       �	�~%�Xc�A�*

loss!�q<�Ê�       �	�&&�Xc�A�*

loss�U2<)j	�       �	��&�Xc�A�*

lossʍ�=��6       �	|d'�Xc�A�*

loss��9=���       �	(�Xc�A�*

loss.b�<�!�)       �	�(�Xc�A�*

losso�%=�y�H       �	|G)�Xc�A�*

loss���;Ni�       �	��)�Xc�A�*

loss��c;��Ï       �	�*�Xc�A�*

loss���;P>�       �	�.+�Xc�A�*

loss��<d[��       �	\�+�Xc�A�*

lossY/�=➿#       �	�d,�Xc�A�*

loss�8\=`��       �	C-�Xc�A�*

lossH>�jrY       �	/�-�Xc�A�*

losss�4>$P*       �	].�Xc�A�*

losskr=��       �	� /�Xc�A�*

loss��i=Q �       �	[�/�Xc�A�*

loss��)<��       �	_F1�Xc�A�*

loss$�<l��       �	��1�Xc�A�*

loss�S=�R��       �	��2�Xc�A�*

lossR��=}�q�       �	�83�Xc�A�*

loss���;)�       �	��3�Xc�A�*

lossSQ1=�e
Q       �	�o4�Xc�A�*

lossif�<aF\       �	�5�Xc�A�*

loss���<��-1       �	��5�Xc�A�*

loss1��;����       �	�>6�Xc�A�*

loss�G<E��:       �	o�6�Xc�A�*

loss��<&B        �	Gw7�Xc�A�*

loss�b�<GA�j       �	#8�Xc�A�*

loss{�%<Y���       �	e�8�Xc�A�*

losst�r<��i       �	^I9�Xc�A�*

loss�R�<Ie�       �	��9�Xc�A�*

lossё�<�V�r       �	�:�Xc�A�*

loss��+<��6       �	;�Xc�A�*

lossy%=�b]       �	��;�Xc�A�*

loss�s=
I>�       �	R<�Xc�A�*

loss�4<�*�?       �	n=�Xc�A�*

loss�#�;sJ8�       �	�=�Xc�A�*

loss�]-<5�{�       �	bH>�Xc�A�*

loss�<,^G       �	X�>�Xc�A�*

lossI,�<~V"{       �	�?�Xc�A�*

loss.AP<чi       �	K>@�Xc�A�*

lossD�;�p��       �	��@�Xc�A�*

loss
&P=$?/       �	�mA�Xc�A�*

loss���<Q$�f       �	�
B�Xc�A�*

loss4�1=��       �	"�B�Xc�A�*

lossa֭;7D�-       �	8HC�Xc�A�*

loss�U�<t�;       �	��C�Xc�A�*

loss�|u<�K�p       �	�D�Xc�A�*

loss�a�<��I       �	�'E�Xc�A�*

loss�J<.�m[       �	��E�Xc�A�*

loss�7;]#�*       �	zUF�Xc�A�*

loss��<((       �	��F�Xc�A�*

loss=�<">�B       �	�G�Xc�A�*

loss��:<M5u�       �	)%H�Xc�A�*

loss��<�X��       �	>vI�Xc�A�*

loss��9=�_u�       �	tJ�Xc�A�*

loss��<�E�       �	�J�Xc�A�*

lossF��<���       �	�K�Xc�A�*

loss�9>�f�       �	�L�Xc�A�*

lossZ3�<s�)+       �	J�L�Xc�A�*

loss��=i�z�       �	�KM�Xc�A�*

loss�<��e�       �	C�M�Xc�A�*

loss-�m=��{�       �	|�N�Xc�A�*

loss�<��׫       �	�O�Xc�A�*

loss��=b�w       �	��O�Xc�A�*

lossA1%=^��$       �	5CP�Xc�A�*

loss��<���B       �	#�P�Xc�A�*

loss�G<��&�       �	�oQ�Xc�A�*

lossz�<��       �	�R�Xc�A�*

loss�q�;鿦       �	2�R�Xc�A�*

loss��?>a�JK       �	�ES�Xc�A�*

loss��=:u��       �	�S�Xc�A�*

loss�=�;�	�       �	�sT�Xc�A�*

loss@B<��]       �	AU�Xc�A�*

lossI�=�h��       �	ʧU�Xc�A�*

loss��<���m       �	kJV�Xc�A�*

loss��:��q       �	-�W�Xc�A�*

lossO=�.�       �	
IX�Xc�A�*

loss�9_=x�1�       �	��X�Xc�A�*

loss�S=X��       �	sY�Xc�A�*

lossm�8<�1��       �	�Z�Xc�A�*

loss��=�*l�       �	_�Z�Xc�A�*

loss��\=�3f�       �	�O[�Xc�A�*

loss�փ<�9!�       �	~�[�Xc�A�*

loss\TQ<�{�       �	��\�Xc�A�*

loss=C)<\pu�       �	�]�Xc�A�*

loss�<��B�       �	�]�Xc�A�*

lossn&|<��1       �	�J^�Xc�A�*

loss�Y;�u�       �	i�^�Xc�A�*

loss{��;6C��       �	<�_�Xc�A�*

lossx+�<p��S       �	�`�Xc�A�*

loss�T�="it�       �	��`�Xc�A�*

loss׵V<U��       �	�sa�Xc�A�*

loss��=D�P4       �	�
b�Xc�A�*

loss�k�=cL       �	��b�Xc�A�*

loss�JY=�� �       �	�Ec�Xc�A�*

loss��<���y       �	��c�Xc�A�*

loss��Q<��}       �		�d�Xc�A�*

loss���;�<u<       �	WCe�Xc�A�*

loss�~�<���       �	P�e�Xc�A�*

loss��=��6       �	�xf�Xc�A�*

loss�4>��Z�       �	bg�Xc�A�*

loss�y;�/��       �	��g�Xc�A�*

lossl3
=}#       �	�@h�Xc�A�*

loss}�	<�͢�       �	��h�Xc�A�*

loss�sy<���       �	��i�Xc�A�*

lossj��<n=��       �	�"j�Xc�A�*

loss6�x<ޜu�       �	�k�Xc�A�*

loss��q=Dc�`       �	��k�Xc�A�*

loss�>�<��-       �	YRl�Xc�A�*

losss��=�==A       �	�2m�Xc�A�*

loss;�=9�+�       �	��m�Xc�A�*

loss�y�;%�       �	m�n�Xc�A�*

loss<<���       �	f-o�Xc�A�*

lossHD�=EW;       �	��o�Xc�A�*

lossZ�u<m��H       �	Z�p�Xc�A�*

lossu7<��       �	�Wq�Xc�A�*

loss� �=vѹ       �	�q�Xc�A�*

loss�<{br�       �	��r�Xc�A�*

loss��<�J       �	�1s�Xc�A�*

lossx�6=*���       �	�t�Xc�A�*

loss�^�<#_h       �	/�t�Xc�A�*

loss�4�<dّ�       �	�Ou�Xc�A�*

loss_��<��p4       �	��u�Xc�A�*

loss4� =� ]�       �	8�v�Xc�A�*

lossm�=��m�       �	P8w�Xc�A�*

loss/�<s�N�       �	n�w�Xc�A�*

loss�M<s�vk       �	�xx�Xc�A�*

lossW�;+Ǐ        �	�y�Xc�A�*

loss2�5<;��       �	�y�Xc�A�*

lossl�P=Fl       �	�\z�Xc�A�*

losss��<&� X       �	r�z�Xc�A�*

loss(==D�a       �	n�{�Xc�A�*

lossD��<�(�!       �	�<|�Xc�A�*

losso�D="��R       �	��|�Xc�A�*

loss��;�3�       �	�w}�Xc�A�*

loss#�<�wze       �	~�Xc�A�*

loss!e&<xU3       �	��~�Xc�A�*

loss�t�=y�a�       �	���Xc�A�*

loss�^T<��)�       �	�z��Xc�A�*

loss��=�WJ�       �	?��Xc�A�*

loss��`<)��       �	ۿ��Xc�A�*

loss�$<J8�       �	�c��Xc�A�*

loss�D!=�m��       �	���Xc�A�*

loss��<YK~       �	����Xc�A�*

loss���<{6u       �	;��Xc�A�*

loss��;��J�       �	�ք�Xc�A�*

loss�M�;KY'�       �	m��Xc�A�*

loss��B=1GK!       �	�!��Xc�A�*

loss�A<���       �	XY��Xc�A�*

loss6%�<̾�       �	~���Xc�A�*

lossƝ�<"�؂       �	�ሢXc�A�*

loss�Mv<�?�Y       �	���Xc�A�*

loss_��=�Pz7       �	�y��Xc�A�*

loss���;#۠L       �	)!��Xc�A�*

losst=�r�       �	w���Xc�A�*

loss��=$��r       �	���Xc�A�*

loss_I�<M�#       �	�v��Xc�A�*

loss���<s��       �	����Xc�A�*

loss��|=㮤�       �	�K��Xc�A�*

loss�n�=��Q       �	�⏢Xc�A�*

loss3�<��,�       �	Ț��Xc�A�*

lossMf;=Ƚ��       �	p}��Xc�A�*

loss,3�<4�X       �	�"��Xc�A�*

loss��<���       �	���Xc�A�*

loss�<��!^       �	�O��Xc�A�*

lossHa#=��       �	� ��Xc�A�*

loss�<y%�|       �	+���Xc�A�*

lossT��=�i�       �	�<��Xc�A�*

loss&��<�=Rk       �	ߕ�Xc�A�*

loss�i=���       �	]���Xc�A�*

lossl?="v       �	�Z��Xc�A�*

loss<_=��_�       �	:��Xc�A�*

lossL�;��       �	����Xc�A�*

loss�#V=���       �	�H��Xc�A�*

loss���=���       �	z♢Xc�A�*

loss���<Q\�       �	㉚�Xc�A�*

loss�4 =���       �	'/��Xc�A�*

loss��=�ym�       �	dɛ�Xc�A�*

loss�Z�<o��       �	�e��Xc�A�*

loss��';p�!       �	"3��Xc�A�*

loss=U=ɂQ�       �	ӝ�Xc�A�*

loss�'�<xI��       �	�r��Xc�A�*

loss��=�q�?       �	���Xc�A�*

loss�x<ǔ�t       �	ʤ��Xc�A�*

loss׬�;�!�       �	:A��Xc�A�*

loss��B<�`6l       �	����Xc�A�*

loss�|!=a>�       �	u���Xc�A�*

loss3��;#�       �	,(��Xc�A�*

loss���<ÿ�r       �	�¢�Xc�A�*

lossL޲<!��`       �	g��Xc�A�*

loss�@=m��       �	���Xc�A�*

lossw5X:�{        �	�夢Xc�A�*

loss?�<�\r�       �	���Xc�A�*

loss�Q�;.�w       �	t%��Xc�A�*

loss��(=kc�y       �	����Xc�A�*

loss���;����       �	T��Xc�A�*

lossɄ=�HT       �	�駢Xc�A�*

loss�M�<�u�#       �	Ӆ��Xc�A�*

lossD�;�i.       �	?��Xc�A�*

loss��=4=Ah       �	¿��Xc�A�*

lossH��;q��9       �	hX��Xc�A�*

loss`�<9�#       �	��Xc�A�*

loss��;ì�       �	1���Xc�A�*

loss�N<��P       �	�]��Xc�A�*

loss��z<j	       �	����Xc�A�*

loss�r�<�>l       �	3���Xc�A�*

lossJ�"=�8�v       �	d>��Xc�A�*

loss���<#�Z&       �	S#��Xc�A�*

loss�&>���O       �	����Xc�A�*

lossڨ�<ؖP�       �	eT��Xc�A�*

loss��=Hx�       �	1밢Xc�A�*

lossQ��:ߺ��       �	���Xc�A�*

loss�z;��-       �	�"��Xc�A�*

loss���<�4d�       �	q̲�Xc�A�*

loss�\�<�h��       �	�`��Xc�A�*

lossQ�\<ˮ%�       �	���Xc�A�*

loss`��<zy�       �	B���Xc�A�*

loss.;��7       �	'0��Xc�A�*

loss=F=�{[�       �	.Ƶ�Xc�A�*

lossקּ=����       �	���Xc�A�*

lossX�=�[�       �	mU��Xc�A�*

loss��K={[l�       �	c���Xc�A�*

loss�{^=�e��       �	����Xc�A�*

lossH~�<LGb�       �	�(��Xc�A�*

lossģK<���       �	aƹ�Xc�A�*

loss�:=���?       �	"l��Xc�A�*

loss���;�j&C       �	d��Xc�A�*

loss�7;gK�       �	A���Xc�A�*

loss4��;���?       �	�2��Xc�A�*

loss��<W�E�       �	~ż�Xc�A�*

loss��\<!�]       �	Id��Xc�A�*

loss #�<Ҡ��       �	����Xc�A�*

lossMĺ<��       �	ᕾ�Xc�A�*

lossIM�;'cҫ       �	�0��Xc�A�*

loss(��;�h5�       �	!˿�Xc�A�*

lossMos=�/��       �	�b��Xc�A�*

loss�:=��0�       �	b���Xc�A�*

loss�'=>�l�       �	`���Xc�A�*

lossv�6=����       �	k+¢Xc�A�*

lossX��;5��R       �	��¢Xc�A�*

lossoA�<�!��       �	�hâXc�A�*

loss}�=��d�       �	" ĢXc�A�*

loss�6�;���       �	g�ĢXc�A�*

loss�<�EM       �	�3ŢXc�A�*

loss:�;���       �	��ŢXc�A�*

loss�)=����       �	�bƢXc�A�*

loss76Q<oJ       �	{�ƢXc�A�*

loss���:�,��       �	`�ǢXc�A�*

loss�Dk<�DS�       �	�&ȢXc�A�*

lossM\=����       �	˽ȢXc�A�*

lossL�=���       �	��ɢXc�A�*

lossd��<D�A       �	j2ʢXc�A�*

loss���<�M��       �	B�ʢXc�A�*

loss�;��%       �	gˢXc�A�*

lossmZ<��%�       �	��ˢXc�A�*

loss�7><x!       �	B�̢Xc�A�*

loss�	;<��       �	�'͢Xc�A�*

loss���;��       �	��͢Xc�A�*

loss�F�=H0�c       �	�W΢Xc�A�*

loss���:Vp7�       �	��΢Xc�A�*

loss��:<�.n[       �	G�ϢXc�A�*

loss��X=�	       �	$%ТXc�A�*

loss�=H:       �	J�ТXc�A�*

lossl	>G[�$       �	/MѢXc�A�*

losswx�<�	y       �	��ѢXc�A�*

loss}�>��7R       �	{�ҢXc�A�*

loss-=�M       �	�ӢXc�A�*

loss%��;�l��       �	˻ӢXc�A�*

loss��D=�Aet       �	WԢXc�A�*

loss� ;!��       �	��ԢXc�A�*

loss6�|<W@�       �	`�բXc�A�*

loss4H;�_�       �	�/֢Xc�A�*

loss��=~��       �	�֢Xc�A�*

loss�)<l|]�       �	fעXc�A�*

losszy�=�~/       �	�آXc�A�*

loss(�=ď]�       �	ۢآXc�A�*

loss�j*=l��       �	2�٢Xc�A�*

loss���;KSg       �	xDڢXc�A�*

loss�/�=���       �	��ڢXc�A�*

loss)��<#��u       �	�~ۢXc�A�*

loss�H=�6O       �	�ܢXc�A�*

loss�e[<?�n�       �	m�ܢXc�A�*

loss]>�=>���       �	_`ݢXc�A�*

loss�O/=���2       �	��ݢXc�A�*

loss�:�<WR�       �	�ޢXc�A�*

loss�`_=��:�       �	T:ߢXc�A�*

loss} =*�       �	�ߢXc�A�*

loss�e�:���       �	o�Xc�A�*

loss�:=��D       �	�Xc�A�*

losssX�<�>�f       �	���Xc�A�*

lossEg�<Ҟ�        �	ƨ�Xc�A�*

loss�<�̋8       �	�D�Xc�A�*

loss�y,<���       �	��Xc�A�*

lossȘr<⤏�       �	qu�Xc�A�*

lossc}�<����       �	��Xc�A�*

loss��<��X�       �	���Xc�A�*

loss@��;g       �	�A�Xc�A�*

lossH�p<��M       �	b��Xc�A�*

loss-�;���       �	"p�Xc�A�*

loss>>�7x       �	d�Xc�A�*

loss��@=k��       �	)��Xc�A�*

loss��I=�J	P       �	If�Xc�A�*

loss�%�<�á�       �		��Xc�A�*

loss�T=fi�]       �	ˠ�Xc�A�*

loss=(=	�^�       �	1D�Xc�A�*

loss&�;\xT       �	H��Xc�A�*

loss���<x|'       �	�}�Xc�A�*

loss�S=v��       �	NG��Xc�A�*

loss�l=�Z�       �	 ���Xc�A�*

loss�m�<\��       �	��Xc�A�*

lossX�=�9]�       �	7�Xc�A�*

lossp0=���8       �	-��Xc�A�*

loss�GL<�%�       �	�l�Xc�A�*

loss��j<�|��       �	t�Xc�A�*

lossƸ`<ɥ;       �	���Xc�A�*

loss�p<My�,       �	V�Xc�A�*

loss�0�;e��{       �	��Xc�A�*

loss�&<���<       �	ϟ�Xc�A�*

loss[R=\6�_       �	+4��Xc�A�*

lossݎ�<���Q       �	���Xc�A�*

loss7m<�x�_       �	"���Xc�A�*

loss}D0=)Ǉ�       �	�D��Xc�A�*

loss�:<���       �	@���Xc�A�*

lossZ"�<��3�       �	l���Xc�A�*

loss��<�M�       �	�.��Xc�A�*

lossF�<�'��       �	����Xc�A�*

loss6��=q���       �	yu��Xc�A�*

losss_<��       �	4��Xc�A�*

loss	8=<fd��       �	���Xc�A�*

loss��<���       �	y@��Xc�A�*

loss�;B� �       �	c��Xc�A�*

loss�;%���       �	}���Xc�A�*

lossy�<D�       �	�Q��Xc�A�*

loss��Z;��^
       �	����Xc�A�*

loss/Z=�M_�       �	���Xc�A�*

loss0�:1��       �	*7��Xc�A�*

loss��g9@�M�       �	\���Xc�A�*

loss�Y9���       �	�� �Xc�A�*

loss���;�4�       �	!"�Xc�A�*

loss�HC=��       �	���Xc�A�*

loss�X8<x��       �	]�Xc�A�*

lossm��9��       �	���Xc�A�*

loss�� =�-�       �	S��Xc�A�*

loss��=�$�       �	�Q�Xc�A�*

loss�Y�:&���       �	k��Xc�A�*

losstTr>��,       �	Y��Xc�A�*

lossx�<.Ҙh       �	R&�Xc�A�*

loss��=e_��       �	ܽ�Xc�A�*

loss(GM<�Wœ       �	=c�Xc�A�*

loss7�:�O�1       �	"��Xc�A�*

loss��>��9�       �	��Xc�A�*

lossb�=ި5,       �	9*	�Xc�A�*

lossJ��<�Z#�       �	��	�Xc�A�*

lossh02=k��I       �	+�
�Xc�A�*

loss��.<�e��       �	��Xc�A�*

lossd�=[x�f       �	���Xc�A�*

loss��+=T�        �	���Xc�A�*

loss�Y9<s       �	�M�Xc�A�*

loss��=���       �	���Xc�A�*

loss$8=�M,Q       �	�|�Xc�A�*

loss��<�_�       �	�Xc�A�*

loss�� ="��       �	���Xc�A�*

loss쉂=�P+�       �	�Y�Xc�A�*

loss*q�<�,�       �	R��Xc�A�*

lossf<cf��       �	w��Xc�A�*

loss�n=+EL�       �	� �Xc�A�*

loss�5<)Ǜ;       �	�Xc�A�*

loss.�;��O�       �	-��Xc�A�*

losszb�<M/��       �	t]�Xc�A�*

loss�w;mt��       �	�Xc�A�*

loss�cq=�Zc�       �	�3�Xc�A�*

lossۋD<dz@       �	���Xc�A�*

losss,>c�!�       �	wj�Xc�A�*

loss_��=��V�       �	��Xc�A�*

loss��<�GC}       �	z��Xc�A�*

loss_��<����       �	t@�Xc�A�*

loss�5"=�       �	���Xc�A�*

loss�	t<�E�       �	��Xc�A�*

loss�SK=�&       �	�#�Xc�A�*

loss�$<�I       �	#��Xc�A�*

loss3 �<oB       �	^�Xc�A�*

loss���<�{��       �	���Xc�A�*

loss*�=�>A       �	���Xc�A�*

loss*K�<wX�       �	�;�Xc�A�*

lossCN;m�ƹ       �	g��Xc�A�*

loss���;"Եr       �	7l �Xc�A�*

loss�މ<Į}1       �	d!�Xc�A�*

loss$��<�	'�       �	ϡ!�Xc�A�*

loss���<�_��       �	�x"�Xc�A�*

lossl&%=ٛRX       �	4#�Xc�A�*

loss\�4= 6.�       �	�#�Xc�A�*

loss�b�;�=.�       �	-@$�Xc�A�*

loss��;5��       �	3�$�Xc�A�*

loss�e�;����       �	�u%�Xc�A�*

loss�ȹ;�Q
�       �	�	&�Xc�A�*

loss���<Oڶ/       �	x�>�Xc�A�*

loss���<�%��       �	.X?�Xc�A�*

loss
q�=DȽ4       �	�?�Xc�A�*

loss��=�,{Y       �	 �@�Xc�A�*

loss/B�;#�,       �	�FA�Xc�A�*

lossŰ6<�w�       �	T�A�Xc�A�*

loss�	U<�a9�       �	��B�Xc�A�*

loss ��<���       �	c&C�Xc�A�*

loss��<W9�Y       �	��C�Xc�A�*

lossR�6=|��]       �	Q�D�Xc�A�*

lossS�;��6       �	'E�Xc�A�*

loss���<���I       �	��E�Xc�A�*

loss�-T<([�       �	(dF�Xc�A�*

loss;F�=��r       �	G�Xc�A�*

loss�=߄       �	R�G�Xc�A�*

loss�<xZ�-       �	P4H�Xc�A�*

loss�
�:�,��       �	"�H�Xc�A�*

loss�E)<`��       �	RJ�Xc�A�*

lossI<rgS       �	'�J�Xc�A�*

loss���<t�ܚ       �	SYK�Xc�A�*

loss��< P�j       �	��K�Xc�A�*

lossHh�=� >z       �	��L�Xc�A�*

loss�?;�.6.       �	�M�Xc�A�*

loss�N=1i��       �	\UN�Xc�A�*

lossו�<���       �	��N�Xc�A�*

loss�<�:�       �	3�O�Xc�A�*

loss��*<���       �	)P�Xc�A�*

lossqU�;�ì       �	��P�Xc�A�*

lossnƳ<Y%L�       �	>YQ�Xc�A�*

loss7�/=��D�       �	��Q�Xc�A�*

loss�< vj�       �	�R�Xc�A�*

loss)#�<'��+       �	T�Xc�A�*

loss��;	��       �	�T�Xc�A�*

loss1y�=��w       �	�bU�Xc�A�*

loss��;x_(       �	i�U�Xc�A�*

loss̅<�`��       �	¢V�Xc�A�*

loss}��;�VO@       �	?:W�Xc�A�*

lossW+�<���       �	[�W�Xc�A�*

loss�y�=܇��       �	%tX�Xc�A�*

loss�=-v       �	Y�Xc�A�*

lossVRW;q&_       �	,�Y�Xc�A�*

loss�4�<��Y�       �	�SZ�Xc�A�*

loss��v;.��R       �	��Z�Xc�A�*

loss!�<z|BR       �	ˆ[�Xc�A�*

loss	*	=Sr*�       �	.\�Xc�A�*

loss}��<�ݴy       �	��\�Xc�A�*

loss���<'�x       �	�a]�Xc�A�*

loss,Kw<3       �	��]�Xc�A�*

losswJ%=�r �       �	�^�Xc�A�*

loss$l:�~�L       �	9d_�Xc�A�*

loss�<�� $       �	��_�Xc�A�*

loss���;�4#B       �	�`�Xc�A�*

loss2-�;�HI*       �	�`a�Xc�A�*

loss�O�=�`^�       �	D�a�Xc�A�*

loss��c<�ͮ       �	�b�Xc�A�*

loss�J�:j��       �	�Cc�Xc�A�*

loss��<i;a       �	�c�Xc�A�*

loss���;��6g       �	f�d�Xc�A�*

loss�=�<(�cz       �	%$e�Xc�A�*

losst��<<�0-       �	+�e�Xc�A�*

loss(�=>��y�       �	�bf�Xc�A�*

loss�S <yw+�       �	D�f�Xc�A�*

lossAv�<��~       �	p�g�Xc�A�*

loss� W=��       �	#0h�Xc�A�*

loss.b<���       �	ii�Xc�A�*

loss�G�;c       �	�i�Xc�A�*

loss��:�'       �	m�j�Xc�A�*

loss�g<C��       �	TVk�Xc�A�*

loss|I=bH��       �	
gl�Xc�A�*

lossEf�<�6Fn       �	z�m�Xc�A�*

loss\7�<�TmM       �	�Cn�Xc�A�*

loss�M<>%�       �	3�n�Xc�A�*

loss��_=K���       �	�zo�Xc�A�*

loss�i�=��        �	Yp�Xc�A�*

loss3�5<R�)�       �	4�p�Xc�A�*

loss5�;�lP�       �	OXq�Xc�A�*

lossw� ;�'4�       �	��q�Xc�A�*

loss��<��(�       �	��r�Xc�A�*

loss��<I�a       �	� s�Xc�A�*

losso�=i:H�       �	��s�Xc�A�*

loss�f�<k&P        �	�Yt�Xc�A�*

loss_�<�N�       �	��t�Xc�A�*

loss�><<y��       �	a�u�Xc�A�*

loss�]:�y|�       �	ٕw�Xc�A�*

loss�\=G�       �	-x�Xc�A�*

loss�;�V�e       �	5�x�Xc�A�*

lossjJ�;��z       �	�py�Xc�A�*

loss�C�;�X5�       �	�z�Xc�A�*

lossԔS=/��       �	�z�Xc�A�*

loss��<j�e       �	"O{�Xc�A�*

loss�5=����       �	��{�Xc�A�*

loss}HA<G��~       �	��|�Xc�A�*

lossѢk;�|       �	�"}�Xc�A�*

loss�Nb<���3       �	��}�Xc�A�*

loss�n�<|O{�       �	�k~�Xc�A�*

lossIq�:�o�       �	��~�Xc�A�*

lossc�=	im       �	x��Xc�A�*

loss�S�< y       �	0��Xc�A�*

loss��=��       �	oـ�Xc�A�*

loss���<o�       �	�~��Xc�A�*

loss�9<�v��       �	�*��Xc�A�*

loss>΁;��       �		Ă�Xc�A�*

lossEN�<r>6       �	���Xc�A�*

loss�[�<^Ƙw       �	�<��Xc�A�*

loss��r<�ko�       �	N҄�Xc�A�*

loss\�<6�I�       �	zn��Xc�A�*

lossL+;s5��       �	R��Xc�A�*

loss�C6:s$�       �	ͭ��Xc�A�*

loss̊I=r�8)       �	N��Xc�A�*

loss�*�<��:�       �	�釣Xc�A�*

losstd&=�̾	       �	w���Xc�A�*

loss�CQ<��D       �	���Xc�A�*

loss�m�=�ͤB       �	҉�Xc�A�*

lossN*=��       �	<���Xc�A�*

loss�4=��o       �	=*��Xc�A�*

loss&'j=#�7^       �	�ǋ�Xc�A�*

loss�t�<g��       �	�ጣXc�A�*

loss=��=-B�k       �	㉍�Xc�A�*

loss!_<��ۆ       �	E��Xc�A�*

loss�ґ<���       �	�:��Xc�A�*

loss�t<�z��       �	7��Xc�A�*

loss}/�;�z��       �	RXc�A�*

lossӌ2<;��       �	����Xc�A�*

lossf�N=�*$�       �	�@��Xc�A�*

loss��<�c�s       �	��Xc�A�*

lossT۬<���u       �	�"��Xc�A�*

lossO�{<��?       �	�Xc�A�*

lossi�;�}�!       �	����Xc�A�*

loss���<'7G�       �	xb��Xc�A�*

loss�� <h�[       �	&��Xc�A�*

loss��-; ]i       �	G㗣Xc�A�*

loss��?<Y6�l       �	�瘣Xc�A�*

lossm��;+M`�       �	:��Xc�A�*

loss�L�;.V��       �	���Xc�A�*

lossm�p=�+�G       �	����Xc�A�*

loss��<�l%�       �	�䜣Xc�A�*

loss���<ׯ��       �	�ĝ�Xc�A�*

loss��/<����       �	�y��Xc�A�*

loss��=��       �	-C��Xc�A�*

lossS�<���       �	�ߟ�Xc�A�*

loss�T:��-=       �	���Xc�A�*

loss�S<F��       �	{ܡ�Xc�A�*

loss�=7Z�       �	Y��Xc�A�*

losss*>���       �	�̣�Xc�A�*

losseĬ=5�       �	||��Xc�A�*

loss��<��[�       �	AD��Xc�A�*

loss}g�;�f�       �	�h��Xc�A�*

loss�d;���       �	�d��Xc�A�*

loss���<1�7       �	c	��Xc�A�*

lossܪ�;>��\       �	g���Xc�A�*

loss�0�;�9�l       �	����Xc�A�*

loss<r<%��E       �	&᪣Xc�A�*

loss�$=�[�       �	����Xc�A�*

lossd,+=��mF       �	:��Xc�A�*

loss9d=�E��       �	�㭣Xc�A�*

loss��<X.�!       �	�ή�Xc�A�*

loss<�+=�>�       �	�t��Xc�A�*

loss9�<��ɽ       �	��Xc�A�*

loss���<�Ͽ�       �	s۰�Xc�A�*

loss�d<_�[r       �	Ku��Xc�A�*

loss���=���       �	4��Xc�A�*

lossE��:Ǎٮ       �	����Xc�A�*

loss�2�=-���       �	�G��Xc�A�*

loss8j�9�5��       �	�ⳣXc�A�*

loss��<��       �	8���Xc�A�*

loss_�;����       �	C9��Xc�A�*

lossn�:@�7j       �	�絣Xc�A�*

losse��<�N�       �	Ѐ��Xc�A�*

loss!��;�F��       �	O?��Xc�A�*

loss\(<�� �       �	ط�Xc�A�*

loss��;<?üT       �	�~��Xc�A�*

loss�!�<4�$       �	\��Xc�A�*

loss`�<��'       �	����Xc�A�*

loss��;ğؠ       �	�S��Xc�A�*

loss��;<W}/�       �	���Xc�A�*

lossZ��<�_�        �	���Xc�A�*

loss/%�<��PW       �	\:��Xc�A�*

lossvܲ<���?       �	aT��Xc�A�*

loss��;$~�Y       �	�轣Xc�A�*

lossu�;���       �	_~��Xc�A�*

loss@#�;pw�       �	f��Xc�A�*

loss=�=c�N�       �	�Ϳ�Xc�A�*

loss�!�:��;       �	�e��Xc�A�*

loss�3<{x       �	����Xc�A�*

loss�:4;s��       �	���Xc�A�*

loss��E<���       �	�@£Xc�A�*

loss�B=�RQJ       �	��£Xc�A�*

loss��+<k�>       �	�zãXc�A�*

lossx�<=��~�       �	"ģXc�A�*

loss���;"�       �	]�ģXc�A�*

loss�J=�̔M       �	cţXc�A�*

loss@�n=����       �	��ţXc�A�*

lossr�;���       �	��ƣXc�A�*

loss�<���       �	�0ǣXc�A�*

lossr��<��Y       �	X�ǣXc�A�*

loss6<�:%{�       �	�\ȣXc�A�*

loss.�K;%�/       �	�ɣXc�A�*

lossX�<!r��       �	B�ɣXc�A�*

loss�Ħ=�V�<       �	YNʣXc�A�*

lossQim:���       �	u�ʣXc�A�*

loss��,=�G�       �	/�ˣXc�A�*

loss��<�|/E       �	�"̣Xc�A�*

loss��=�ik        �	A�̣Xc�A�*

lossm��<����       �	HRͣXc�A�*

loss���;�N��       �	�8ΣXc�A�*

lossZ��<�~       �	��ΣXc�A�*

loss��<u��       �	�vϣXc�A�*

lossQǓ=vn��       �	�УXc�A�*

loss���<4���       �	R�УXc�A�*

loss��;>� �       �	�RѣXc�A�*

loss~�=�iT       �	��ѣXc�A�*

loss���=��u5       �	��ңXc�A�*

loss\A8<��B       �	#2ӣXc�A�*

loss��/<�ʥ       �	��ӣXc�A�*

loss.2B=%��c       �	pգXc�A�*

lossV1�<�BE[       �	
֣Xc�A�*

loss�V�=�sI       �	��֣Xc�A�*

loss��=��l       �	_ףXc�A�*

loss���=L]	�       �	�أXc�A�*

loss-�`=k���       �	1�أXc�A�*

loss���=gHI,       �	�X٣Xc�A�*

lossd�A:���       �	��٣Xc�A�*

loss\[	=} ��       �	נڣXc�A�*

lossq��=jҏ       �	CۣXc�A�*

loss�7A;���       �	{ܣXc�A�*

loss�=m���       �	ݣXc�A�*

lossE�=�E'�       �	��ݣXc�A�*

loss�im;@F�       �	hZޣXc�A�*

loss�I�<۾       �	
ߣXc�A�*

lossȢ9=m��M       �	�ߣXc�A�*

lossS��<��,b       �	�\�Xc�A�*

loss��<�Lv       �	E��Xc�A�*

loss|�;鶠�       �	s��Xc�A�*

lossA�I=H��|       �	�@�Xc�A�*

loss_�[<�L��       �	�{�Xc�A�*

loss݉�<�L!�       �	K"�Xc�A�*

losshD<�de       �	]��Xc�A�*

loss<�=��R�       �	�a�Xc�A�*

lossfo=���S       �	�Xc�A�*

lossR�o<�Vf       �	@��Xc�A�*

loss���=�Cc�       �	JD�Xc�A�*

loss=��<�"�       �	&��Xc�A�*

lossA�=9T�-       �	N�Xc�A�*

loss1d=q�ڐ       �	��Xc�A�*

loss�D=�0�       �	��Xc�A�*

loss�C<Bc\�       �	��Xc�A�*

loss��:{�js       �	�F�Xc�A�*

loss��$<X��       �	@��Xc�A�*

loss��=Ԙ       �	u��Xc�A�*

loss#C�<6���       �	����Xc�A�*

loss�K=N��       �	�3�Xc�A�*

lossX]d=��+       �	���Xc�A�*

loss���;>�'       �	}�Xc�A�*

loss|B�:r       �	dZ�Xc�A�*

loss �<�]�       �	���Xc�A�*

loss�<��       �	d��Xc�A�*

loss��#<�o��       �	�W�Xc�A�*

loss���;_-�       �	���Xc�A�*

loss�c4<��Po       �	Z��Xc�A�*

lossZ�<OE�2       �	W\��Xc�A�*

loss��<H���       �	���Xc�A�*

lossO�=yn5       �	���Xc�A�*

lossVvx<�%�       �	����Xc�A�*

loss�PU=~���       �	Ք��Xc�A�*

loss逓=��`       �	�*��Xc�A�*

loss��^<�X��       �	����Xc�A�*

loss�4R=t�'R       �	`��Xc�A�*

lossw�><TC��       �	����Xc�A�*

lossֹ^=×�       �	ס��Xc�A�*

loss��;���       �	�G��Xc�A�*

lossvol<Y�       �	���Xc�A�*

loss,h0<�w��       �	����Xc�A�*

loss*Y�<a��       �	�\��Xc�A�*

loss�<v�m       �	����Xc�A�*

lossm�!>��۷       �	����Xc�A�*

loss�9�=mkz�       �	f2��Xc�A�*

lossz�y<�K��       �	����Xc�A�*

loss��n;>�7�       �	�i �Xc�A�*

loss�<{�
U       �		�Xc�A�*

loss�B�<ԛ(F       �	"��Xc�A�*

loss��;Q��e       �	9G�Xc�A�*

loss�Q�<k�8f       �	`��Xc�A�*

loss�W�;p8��       �	@��Xc�A�*

loss���=EL�3       �	�(�Xc�A�*

loss�2=	D�       �	��Xc�A�*

loss璇<�_       �	nk�Xc�A�*

loss��=�.�       �	��Xc�A�*

loss�i=G�i�       �	*��Xc�A�*

lossƁ�<�~       �	F�Xc�A�*

loss�jv;e[wE       �	���Xc�A�*

lossa:;3�g
       �	�Xc�A�*

loss�s�<+j8�       �	0,	�Xc�A�*

loss͌;�s       �	��	�Xc�A�*

loss��@;�h�3       �	�
�Xc�A�*

loss��;��       �	�r�Xc�A�*

lossf��<��O       �	0�Xc�A�*

loss�Đ<�W�       �	c�Xc�A�*

loss�Y=��ͮ       �	���Xc�A�*

loss���=��[       �	�U�Xc�A�*

loss�i�<2-��       �	�k�Xc�A�*

lossJ�B=E��)       �	k�Xc�A�*

loss;�<�a�       �	��Xc�A�*

lossJ9�;��x       �	��Xc�A�*

lossej=��.f       �	r�Xc�A�*

loss�N=���       �	m�Xc�A�*

loss�$W=���       �	���Xc�A�*

lossdܶ:��	       �	E��Xc�A�*

lossD�<�)�"       �	�G�Xc�A�*

lossdst;�c       �	���Xc�A�*

lossT;�=[��Z       �	���Xc�A�*

loss�b�=�Y��       �	˄�Xc�A�*

loss���<��q       �	d$�Xc�A�*

lossPX�=�5p�       �	���Xc�A�*

loss�?3<�>T�       �	�z�Xc�A�*

loss)�:#��c       �	�4�Xc�A�*

loss�V=yh��       �	���Xc�A�*

loss
<~<ʨ��       �	c|�Xc�A�*

lossƭ�;Y�F�       �	�,�Xc�A�*

loss�z�<�l\�       �	~��Xc�A�*

loss��;c�'�       �	ҍ�Xc�A�*

loss��<��?       �	� �Xc�A�*

lossQ�Z<hh}       �	F�Xc�A�*

loss��=��O       �	���Xc�A�*

loss��(;w���       �	Eg �Xc�A�*

lossX)c<b��       �	�!�Xc�A�*

loss!9V=�}��       �	�!�Xc�A�*

loss��=�:       �	��"�Xc�A�*

loss�X�;P��;       �	=#�Xc�A�*

loss�~=���       �	D�#�Xc�A�*

lossaA8=-2�       �	��$�Xc�A�*

loss��(<���       �	��%�Xc�A�*

loss�n=Q)�P       �	�p&�Xc�A�*

loss���<��]       �	�'�Xc�A�*

lossI"`<[��       �	M(�Xc�A�*

loss�l5<��b       �	��(�Xc�A�*

loss��<t�/       �	��)�Xc�A�*

loss��3=��JE       �	�W*�Xc�A�*

loss��;�<��       �	��*�Xc�A�*

loss�P=�8_�       �	��+�Xc�A�*

loss�y;|] r       �	�,�Xc�A�*

loss�]�:t�(�       �	2W-�Xc�A�*

loss�M<a��       �	�-�Xc�A�*

loss�܊;��(       �	�.�Xc�A�*

lossx%�;۽�d       �	�/�Xc�A�*

loss��>��7�       �	�/�Xc�A�*

loss��=�~�       �	�J0�Xc�A�*

loss�h�;Iƒ�       �	��0�Xc�A�*

loss.�=x�       �	��1�Xc�A�*

lossZ��<�L�`       �	�'2�Xc�A�*

loss=̍<l�rR       �	��2�Xc�A�*

loss�2�;jj4�       �	�R3�Xc�A�*

loss���<t�$�       �	��3�Xc�A�*

loss|j6=��       �	��4�Xc�A�*

loss��;����       �	�5�Xc�A�*

loss䖳=���       �	��5�Xc�A�*

loss݌�:Q�p       �	L6�Xc�A�*

loss���;_YW�       �	��6�Xc�A�*

loss�r!=���R       �	j�7�Xc�A�*

loss�O=�<       �	�8�Xc�A�*

lossI�;�G�
       �	9�8�Xc�A�*

loss���<�<0       �	�L9�Xc�A�*

loss���<�7       �	u�9�Xc�A�*

loss=�6$       �	(|:�Xc�A�*

lossmH=?���       �	�3;�Xc�A�*

loss�Ww=��hJ       �	��;�Xc�A�*

loss[^=5�w       �	l_<�Xc�A�*

lossM0<+       �	�=�Xc�A�*

lossR��; ;��       �	��=�Xc�A�*

losscC�;���       �	�->�Xc�A�*

loss}�.<�#8"       �	��>�Xc�A�*

loss��<=��       �	'h?�Xc�A�*

lossj7;���G       �	2@�Xc�A�*

loss��<ů*�       �	W�@�Xc�A�*

loss� <��֔       �	�<A�Xc�A�*

loss��e=�       �	�A�Xc�A�*

loss6Ӝ<��7w       �	anB�Xc�A�*

loss׏=�<`       �	kC�Xc�A�*

loss&��<�rQ       �	/�C�Xc�A�*

loss�-�<ڀ3w       �	>AD�Xc�A�*

loss���<�ZK8       �	*�D�Xc�A�*

loss���<�z�0       �	�|E�Xc�A�*

loss�7�<Fus       �	+F�Xc�A�*

lossJ>T�Ը       �	 �F�Xc�A�*

loss�b�;���       �	NG�Xc�A�*

loss�L�:��EV       �	��G�Xc�A�*

loss-W<u?�       �	�{H�Xc�A�*

loss�+C=�!�N       �	� I�Xc�A�*

loss��=K�7(       �	a�I�Xc�A�*

lossU�:���       �	K\J�Xc�A�*

lossf�E;[��W       �	�
K�Xc�A�*

loss��>=����       �	�K�Xc�A�*

lossH]!=h�yf       �	��L�Xc�A�*

loss�h�<�2\�       �	 XM�Xc�A�*

loss��M=�V�       �	i�M�Xc�A�*

loss%��<�I       �	`�N�Xc�A�*

lossL-=1��<       �	Y�O�Xc�A�*

lossl��:I7ś       �	FP�Xc�A�*

loss|�<����       �	j�P�Xc�A�*

loss
r<�	\�       �	vQ�Xc�A�*

loss��=ߙ�       �	�R�Xc�A�*

lossa��;�C��       �	N�R�Xc�A�*

loss���;�<�       �	�S�Xc�A�*

loss�_=A��:       �	�4T�Xc�A�*

loss)o<��a       �	��T�Xc�A�*

loss仾=����       �	h�U�Xc�A�*

lossπ$<l���       �	dV�Xc�A�*

lossn��<���       �	@�V�Xc�A�*

lossF��<��e       �	C�W�Xc�A�*

loss_��<ʗu       �	A*X�Xc�A�*

loss$a�<�-��       �	��X�Xc�A�*

loss�.<�o[�       �	SvY�Xc�A�*

loss�٪<�Ԧ       �	�Z�Xc�A�*

loss1��;h��_       �		�Z�Xc�A�*

loss�G�<��       �	�h[�Xc�A�*

loss�F
=� x�       �	�\�Xc�A�*

loss�=����       �	¦\�Xc�A�*

loss��;��;       �	EK]�Xc�A�*

lossXV\<qpA       �	�]�Xc�A�*

loss��<�!=       �	ܜ^�Xc�A�*

loss(��=�{�}       �	�B_�Xc�A�*

loss�H	=B;{�       �	J�_�Xc�A�*

loss��=�d�       �	��`�Xc�A�*

loss$E<s_�       �	�@a�Xc�A�*

loss�=ٟ
�       �	��a�Xc�A�*

loss�<�	��       �	r�b�Xc�A�*

loss��+=i�`)       �	��c�Xc�A�*

loss�S�<9��       �	�4d�Xc�A�*

loss��<��=8       �	��d�Xc�A�*

lossck=�J�J       �	�e�Xc�A�*

loss{ٖ<É�       �	�sf�Xc�A�*

loss��=����       �	�g�Xc�A�*

loss]�V<���       �	&�g�Xc�A�*

lossL��<�p��       �	�ch�Xc�A�*

loss ֘<�X�       �	:i�Xc�A�*

loss���;�WuT       �	֧i�Xc�A�*

loss�G�<��j�       �	�Cj�Xc�A�*

loss�D;�~K�       �	��j�Xc�A�*

lossf�R<�ixC       �	��k�Xc�A�*

loss/�j<e��       �	>$m�Xc�A�*

loss �;��<�       �	x�n�Xc�A�*

lossâ=��9�       �	sLp�Xc�A�*

loss�0�<xH��       �	O�p�Xc�A�*

lossQ�3=t�<j       �	�q�Xc�A�*

loss]o>M���       �	>#r�Xc�A�*

loss��:�5ݐ       �	��r�Xc�A�*

loss�C=�Ҥ�       �	3js�Xc�A�*

loss\��<v$p       �	Xt�Xc�A�*

loss��<4��       �	�t�Xc�A�*

loss*�;�؉       �	iRu�Xc�A�*

losst[�=c�0       �	f�u�Xc�A�*

lossL�r<=�;       �	$�v�Xc�A�*

loss�<���       �	AJw�Xc�A�*

loss�"=��U{       �	9�w�Xc�A�*

lossC�<t��       �	�x�Xc�A�*

lossz5�<�YI       �	B\y�Xc�A�*

loss��;���       �	�z�Xc�A�*

losst�*<�Ǯ       �	�z�Xc�A�*

loss���<�Ѭ       �	�L{�Xc�A�*

loss�[�<�2��       �	x�{�Xc�A�*

loss
ׂ=�#�>       �	�|�Xc�A�*

loss�Y�;�qi       �	O!}�Xc�A�*

lossJΗ<+]M�       �	Y�}�Xc�A�*

losst$;��4       �	�W~�Xc�A�*

loss���=4�P�       �	�%�Xc�A�*

loss��f;֤(=       �	���Xc�A�*

lossR�;ti4       �	�Y��Xc�A�*

loss�,�<��s       �	��Xc�A�*

lossS+�;4�       �	����Xc�A�*

loss�_�<�/�:       �	k(��Xc�A�*

loss�W�:�:kx       �	��Xc�A�*

loss��=�:J       �	򲃤Xc�A�*

lossC�`<�`(       �	M��Xc�A�*

loss{#�=�/       �	;儤Xc�A�*

losszk<ʷ'�       �	-{��Xc�A�*

loss(�<�!�}       �	0��Xc�A�*

loss�D�;~��       �	���Xc�A�*

loss�CM;=a�       �	 U��Xc�A�*

lossCLH=�q�n       �	�쇤Xc�A�*

loss���:�v3S       �	����Xc�A�*

loss�=�nq       �	�j��Xc�A�*

losscV�<��q�       �	��Xc�A�*

loss��=�q^       �	����Xc�A�*

loss�
=�9�       �	����Xc�A�*

loss$;��L       �	J���Xc�A�*

loss�h�=E)b       �	�t��Xc�A�*

lossX��;�*ǂ       �	8��Xc�A�*

losss"\;�P��       �	�r��Xc�A�*

loss�E�<�	�Q       �	�u��Xc�A�*

loss��y;�vi       �	�n��Xc�A�*

loss�)�<Ja2�       �	73��Xc�A�*

loss�92<B�º       �	���Xc�A�*

loss�=�;~l       �	����Xc�A�*

loss#`�:]�t       �	�n��Xc�A�*

loss�=�W��       �	=���Xc�A�*

loss�x#<�7�       �	If��Xc�A�*

loss�ˆ=.�=�       �	4���Xc�A�*

loss���<����       �	�W��Xc�A�*

loss�7<��       �	�*��Xc�A�*

loss!��<S�\�       �	��Xc�A�*

loss�RJ=��s       �	���Xc�A�*

loss�5�<���u       �	�c��Xc�A�*

loss��<־G        �	X��Xc�A�*

lossJ�<��|7       �	i���Xc�A�*

loss�z<eC��       �	P��Xc�A�*

lossT=<�Ӈ       �	,�Xc�A�*

lossH}.<�n       �	����Xc�A�*

loss�kU;��       �	q=��Xc�A�*

lossA��;��\T       �	`䟤Xc�A�*

lossf�=����       �	����Xc�A�*

loss��;l��       �	�/��Xc�A�*

loss�^�:ڮN       �	ѡ�Xc�A�*

lossx8C=<�r       �	�h��Xc�A�*

loss��<#��       �	5*��Xc�A�*

loss��W=�~�       �	.ţ�Xc�A�*

loss�4�<�H�"       �	�f��Xc�A�*

lossvT�<�1��       �	���Xc�A�*

loss�Dl=V��       �	:���Xc�A�*

loss<��%       �	+���Xc�A�*

lossO#^;�u��       �	1��Xc�A�*

lossi�g<!��       �	�Ч�Xc�A�*

loss~^<��3�       �	�k��Xc�A�*

lossC<���       �	+��Xc�A�*

loss�9�9u�/       �	����Xc�A�*

loss�'
;�VN~       �	2Y��Xc�A�*

lossF��<�9��       �	n���Xc�A�*

lossJ<P��|       �	ݕ��Xc�A�*

loss��:���       �	�5��Xc�A�*

loss,C�=?n�       �	x�Xc�A�*

loss1��=r��       �	ߊ��Xc�A�*

loss�w;�g��       �	�'��Xc�A�*

loss��<R�P       �	6��Xc�A�*

loss�9�<��+�       �	����Xc�A�*

loss���;����       �	o���Xc�A�*

lossMs�9��E       �	m8��Xc�A�*

loss|�#;���v       �	xӱ�Xc�A�*

loss�·;:�H       �	�z��Xc�A�*

loss"D<�*.       �	���Xc�A�*

lossv<�:�-�       �	Z���Xc�A�*

loss4#	9�D�}       �	�Y��Xc�A�*

loss�=�<� �       �	����Xc�A�*

loss-H<�y�?       �	P���Xc�A�*

loss��:F�v�       �	1(��Xc�A�*

lossj`�9�dܺ       �	�¶�Xc�A�*

lossfF�:�lź       �	4e��Xc�A�*

loss|�<�$�.       �	���Xc�A�*

loss&�=~���       �	����Xc�A�*

lossԬ%9��.�       �	K��Xc�A�*

loss���;9��       �	�﹤Xc�A�*

loss�}�=�Ú       �	Ӈ��Xc�A�*

loss���;��{       �	E)��Xc�A�*

lossF�>B
;�       �	�»�Xc�A�*

loss[��=f9�       �	2Y��Xc�A�*

lossV=�/:       �	����Xc�A�*

lossR��<|��#       �	����Xc�A�*

lossᏑ<���       �	�9��Xc�A�*

loss��=����       �	�۾�Xc�A�*

loss��J<
<�V       �	Ԁ��Xc�A�*

loss�j'<�W��       �	�%��Xc�A�*

loss�r=�D       �	`���Xc�A�*

loss���;ɒ�       �	+h��Xc�A�*

lossT	=���1       �	�¤Xc�A�*

loss�2�<��R       �	��¤Xc�A�*

loss�y-<�F�       �	y=äXc�A�*

loss�~�<���J       �	��äXc�A�*

loss�*�=���       �	�tĤXc�A�*

loss�D�=�D�i       �	
ŤXc�A�*

loss��~<��J�       �	�ŤXc�A�*

lossaYk=u)f       �	�EƤXc�A�*

loss��;�ҽ�       �	��ƤXc�A�*

lossx�<���       �	SwǤXc�A�*

loss%|<
IEF       �	ZȤXc�A�*

loss2�:=�?i       �	ߣȤXc�A�*

loss��;��,       �	8MɤXc�A�*

lossn��;;��       �	��ɤXc�A�*

loss���:��       �	�ʤXc�A�*

lossVI<����       �	]ˤXc�A�*

lossI=��8       �	g�ˤXc�A�*

lossj�=�7�?       �	��̤Xc�A�*

loss���<�Km7       �	�iͤXc�A�*

losssEx<L�8       �	eΤXc�A�*

loss�v<�ꋌ       �	�ϤXc�A�*

loss��d;��*       �	��ϤXc�A�*

loss8�<����       �	�ФXc�A�*

loss�(�;��C       �	�YҤXc�A�*

loss�3<ʕ/�       �	�ӤXc�A�*

loss��I<V�;       �	��ӤXc�A�*

loss���=�hvU       �	��ԤXc�A�*

loss�<,J��       �	�gդXc�A�*

loss�l�<Уq8       �	�֤Xc�A�*

loss��v;�#��       �	KפXc�A�*

loss���:U���       �	�EؤXc�A�*

lossǉ<���       �	�٤Xc�A�*

loss�=�m%       �	r�٤Xc�A�*

loss��	=�`4�       �	�ڤXc�A�*

lossH�<����       �	r6ۤXc�A�*

lossN[@=����       �	��ۤXc�A�*

loss��:7H��       �	�ܤXc�A�*

loss�`=�j�j       �	�RݤXc�A�*

lossƧ�;-�^r       �	-\ޤXc�A�*

loss�< x�       �	��ޤXc�A�*

loss��<$�y        �	/���Xc�A�*

loss��<K�       �	s��Xc�A�*

lossi�u<��.`       �	���Xc�A�*

loss�7=��Ή       �	m���Xc�A�*

loss9�=��R       �	�>��Xc�A�*

loss�؝<�z)       �	����Xc�A�*

loss�<�Zʉ       �	�z��Xc�A�*

loss�4�:�k|       �	��Xc�A�*

lossf
=H
�       �	����Xc�A�*

loss�;�f�`       �	H��Xc�A�*

loss-�t<�嚋       �	���Xc�A�*

lossvG;���       �	ʈ��Xc�A�*

lossڿ7=��e�       �	�*��Xc�A�*

loss D<߳�l       �	����Xc�A�*

loss�=�;v�p       �	�^��Xc�A�*

loss% �<ڢ@k       �	����Xc�A�*

loss���:�$�       �	�� �Xc�A�*

loss� l<-]       �	��Xc�A�*

losss��:=T�g       �	��Xc�A�*

loss�*�<Qqh�       �	��Xc�A�*

loss���;7}(N       �	 �Xc�A�*

lossĐ�=I�[       �	Z��Xc�A�*

lossdQ\<�b�)       �	�V�Xc�A�*

loss���=�@�       �	���Xc�A�*

loss�?�;��k�       �	ǹ�Xc�A�*

loss��<��N       �	�T�Xc�A�*

loss{s�<�&��       �	���Xc�A�*

lossV�<����       �	'��Xc�A�*

loss*�<���0       �	��Xc�A�*

lossm��=@�,�       �	x��Xc�A�*

losso�<5���       �	�O	�Xc�A�*

loss�~�<� Ò       �	��	�Xc�A�*

loss$�8=�U��       �	�
�Xc�A�*

loss���<��       �	|'�Xc�A�*

lossd�%;׻K�       �	���Xc�A�*

loss�q�;�&@	       �	>��Xc�A�*

loss �1:�q��       �	9��Xc�A�*

loss6pT=$q��       �	�9�Xc�A�*

loss�4p=Q�z&       �	���Xc�A�*

loss�;�<����       �	
��Xc�A�*

lossR��<j&u�       �	"��Xc�A�*

lossx<&"       �	1��Xc�A�*

loss/;;tQ��       �	�c�Xc�A�*

lossX:z�4       �	��Xc�A�*

loss��?=���C       �	��Xc�A�*

loss���=�A��       �	�P�Xc�A�*

lossnG�<����       �	��Xc�A�*

lossa"r=����       �	��Xc�A�*

loss#��<��4       �	&o�Xc�A�*

loss�gg<j۶�       �	�Xc�A�*

loss, �:�c��       �	ǽ�Xc�A�*

loss�w�<y�z�       �	l^�Xc�A�*

loss�)e<�x	s       �	$�Xc�A�*

loss~�=�L�       �	��Xc�A�*

lossQP�;�&��       �	�Q�Xc�A�*

loss��:��Kl       �	���Xc�A�*

lossq_Z:�K�f       �	؟�Xc�A�*

loss��z9f�kW       �	���Xc�A�*

loss�U�<��Pc       �	+��Xc�A�*

lossNX_<+0       �	I��Xc�A�*

lossCc>C���       �	{M�Xc�A�*

loss�!H;�f�       �	���Xc�A�*

lossJ'G<�p�{       �	�� �Xc�A�*

loss�q�<�jj�       �	cC!�Xc�A�*

loss4�;n3դ       �	��!�Xc�A�*

loss�y!;�:��       �	�"�Xc�A�*

lossS�=���       �	�+#�Xc�A�*

loss��<2       �	��#�Xc�A�*

loss�\�=�*Ɠ       �	�q$�Xc�A�*

losse�
<��       �	�%�Xc�A�*

loss8�V<#b>	       �	-�%�Xc�A�*

lossc]<M���       �	fK&�Xc�A�*

loss�#;8��_       �	 %'�Xc�A�*

loss��a<fMb       �	��'�Xc�A�*

loss��i< $�       �	(�(�Xc�A�*

losswb�<d��M       �	:!)�Xc�A�*

loss�X�<�P       �	e�)�Xc�A�*

loss��;��#�       �	�g*�Xc�A�*

lossO��;J�r�       �	�+�Xc�A�*

loss�D'=i��       �	��+�Xc�A�*

loss�`/<{��q       �	_`,�Xc�A�*

losss�=NQ��       �	� -�Xc�A�*

lossW ]<$��       �	��-�Xc�A�*

loss6fZ;��       �	N.�Xc�A�*

lossI';��9       �	��.�Xc�A�*

loss�O;���       �	�/�Xc�A�*

loss�'�<�0r�       �	_(0�Xc�A�*

loss\=�<s[�e       �	v�0�Xc�A�*

loss��<�b��       �	Fa1�Xc�A�*

lossƐt=�(�G       �	7�1�Xc�A�*

loss�c\<z�E�       �	�2�Xc�A�*

loss��<����       �	"33�Xc�A�*

lossq>�;5�1*       �	C�3�Xc�A�*

loss���;֔       �	ni4�Xc�A�*

loss��<��#*       �	.5�Xc�A�*

loss�S�;��8�       �	��5�Xc�A�*

loss��<rW={       �	�k6�Xc�A�*

loss-��;����       �	�7�Xc�A�*

loss��%=���       �	,�7�Xc�A�*

lossO�<3���       �	>8�Xc�A�*

loss8~Y;�tP�       �	��8�Xc�A�*

loss	��:��       �	ѐ9�Xc�A�*

lossAE�;$d*�       �	Y0:�Xc�A�*

loss�1;v�       �	��:�Xc�A�*

loss���=�d�k       �	;n;�Xc�A�*

loss�Sz=��s       �	�<�Xc�A�*

loss�9Ϧ(�       �	��<�Xc�A�*

loss�dZ<5<��       �	�Q=�Xc�A�*

lossd�<,�-       �	i�=�Xc�A�*

loss/K= ��c       �	x�>�Xc�A�*

loss��=d        �	28?�Xc�A�*

lossVQ<UЋ       �	=�?�Xc�A�*

loss�,=窻�       �	Uk@�Xc�A�*

loss�< ���       �	�A�Xc�A�*

loss�&f<���^       �	i�A�Xc�A�*

loss.C;`߉       �	
MB�Xc�A�*

lossۜ�;�.�       �	��B�Xc�A�*

loss@��=��Ҕ       �	ٲC�Xc�A�*

lossmQ=H�A{       �	�bD�Xc�A�*

lossC}�<N�N�       �	`E�Xc�A�*

loss?MC;�"�       �	�E�Xc�A�*

loss�[<���       �	��F�Xc�A�*

loss�
Z<��[       �	�G�Xc�A�*

loss`�6<����       �	�;H�Xc�A�*

lossfc�:@�`�       �	J�H�Xc�A�*

loss��;����       �	פI�Xc�A�*

loss@��<0��)       �	�FJ�Xc�A�*

lossg�<A���       �	��J�Xc�A�*

loss�S@=Rw�       �	>�K�Xc�A�*

loss u*;���       �	�oL�Xc�A�*

loss�H�<l���       �	sLM�Xc�A�*

loss�]V<����       �	��M�Xc�A�*

loss��=��`1       �	�O�Xc�A�*

loss!��;��?q       �	��O�Xc�A�*

loss-()=����       �	8�P�Xc�A�*

loss��<=���       �	��Q�Xc�A�*

loss��r<�D�8       �	�jR�Xc�A�*

loss)��;�l��       �	6S�Xc�A�*

loss#�N<Q�       �	�S�Xc�A�*

loss��d<�?[�       �	�IT�Xc�A�*

loss�=/;�e��       �	nU�Xc�A�*

loss�i9=㍤       �	�	V�Xc�A�*

lossr&�<^��       �	]�V�Xc�A�*

loss/S;LX�       �	�PW�Xc�A�*

loss���;��ߙ       �	��W�Xc�A�*

loss�2�<����       �	��X�Xc�A�*

loss7|;[��       �	�Y�Xc�A�*

loss�%�:s�te       �	�Z�Xc�A�*

loss�%;0T�       �	i�Z�Xc�A�*

losst��<�3       �	+K[�Xc�A�*

lossX�m<���c       �	�[�Xc�A�*

lossv��=tP@       �	ލ\�Xc�A�*

loss�(=̎�       �	�8]�Xc�A�*

lossN�|;1�s�       �	�]�Xc�A�*

loss��<|�       �	�x^�Xc�A�*

lossq��<v��       �	�_�Xc�A�*

loss�+�<T�       �	��_�Xc�A�*

loss�h=M��       �	@O`�Xc�A�*

losshO;��Mn       �	�`�Xc�A�*

loss���;w?�       �	l�a�Xc�A�*

loss�L?<qօ}       �	1b�Xc�A�*

loss�<a��       �	R�b�Xc�A�*

loss���;6K~�       �	>wc�Xc�A�*

loss�y:�}�G       �	+d�Xc�A�*

loss��<����       �	еd�Xc�A�*

lossQ\�;���X       �	zVe�Xc�A�*

loss%��;�@       �	��e�Xc�A�*

loss =/�       �	��f�Xc�A�*

lossp%;ᢚI       �		6g�Xc�A�*

losss.0:��Tp       �	gh�Xc�A�*

loss�F�<�G��       �	;�h�Xc�A�*

loss�!�;�0�-       �	�Ai�Xc�A�*

loss_�:�t       �	
�i�Xc�A�*

loss���;Di��       �	�ij�Xc�A�*

loss��m<��       �	�k�Xc�A�*

loss�K�:K3       �	Ƨk�Xc�A�*

lossa�E<���       �	0Kl�Xc�A�*

lossO8<�l}B       �	��l�Xc�A�*

loss�3=bB�S       �	�~m�Xc�A�*

loss���;�l�       �	�n�Xc�A�*

losse�b9�'�       �	5�n�Xc�A�*

loss��<F�)9       �	�No�Xc�A�*

loss�8�;�0g�       �	��o�Xc�A�*

loss�|�:��I�       �	R�p�Xc�A�*

loss��><�\�       �	�:q�Xc�A�*

loss��9
j7�       �	��q�Xc�A�*

loss�;�<aW�E       �	{�r�Xc�A�*

lossĺ!;��       �	+5s�Xc�A�*

loss�N�<�da�       �	��s�Xc�A�*

loss�R�<1��       �	�t�Xc�A�*

loss;)C= 8�       �	#u�Xc�A�*

lossv�<���z       �	"�u�Xc�A�*

lossȌ);v��       �	�av�Xc�A�*

loss�j:u}�2       �	�w�Xc�A�*

lossHR�:�n|�       �	ϡw�Xc�A�*

losslT<��rB       �	Bx�Xc�A�*

loss{�;@L��       �	x�x�Xc�A�*

loss�P=U�=       �	��y�Xc�A�*

loss�O�<�§       �	��z�Xc�A�*

loss���<UQg�       �	�U{�Xc�A�*

loss��W<+ܕt       �	{�{�Xc�A�*

loss�5�;E��B       �	ѯ|�Xc�A�*

loss�7�:�C/8       �	QN}�Xc�A�*

loss���;��7       �	��}�Xc�A�*

loss>Y <o��       �	�~�Xc�A�*

lossO#�<�f[!       �	<N�Xc�A�*

loss[�=B5^       �	O��Xc�A�*

lossqWc=Gl�+       �	����Xc�A�*

loss8r�<4�<       �	�3��Xc�A�*

lossw�<ľy�       �	3ށ�Xc�A�*

loss��;���       �	�z��Xc�A�*

loss�+	:z~�       �	�Q��Xc�A�*

loss���:,ق�       �	�(��Xc�A�*

loss-�V:�LA       �	�Є�Xc�A�*

lossc=v X       �	�o��Xc�A�*

lossx)T<�Y��       �	Y��Xc�A�*

lossjɈ=UF(       �	 ���Xc�A�*

loss�=FP       �	Ԟ��Xc�A�*

lossc4�<s�q       �	�K��Xc�A�*

lossj�=p���       �	�ꈥXc�A�*

losst�U=Zh'$       �	����Xc�A�*

loss�_�;����       �	C<��Xc�A�*

loss�4�=QR<H       �	e㊥Xc�A�*

loss˫�=˫a�       �	����Xc�A�*

loss&\:&M�B       �	�,��Xc�A�*

loss���<@"�       �	.䌥Xc�A�*

lossΏ=e�4�       �	���Xc�A�*

loss�ŏ<�D[f       �	����Xc�A�*

loss�I|;Dute       �	�ŏ�Xc�A�*

lossn�^=����       �	ߊ��Xc�A�*

loss�2	<O���       �	y?��Xc�A�*

lossF��<�.�       �	�U��Xc�A�*

loss�.�;~AV       �	����Xc�A�*

loss3:<��h�       �	�E��Xc�A�*

lossn�@=g�Vu       �	䔥Xc�A�*

loss?�<�)��       �	9{��Xc�A�*

loss�Qk;�"�       �	���Xc�A�*

loss���<6�e�       �	����Xc�A�*

lossI��<T��{       �	D��Xc�A�*

loss}�4<k)9�       �	�ݗ�Xc�A�*

loss��<{�M       �	�ט�Xc�A�*

loss)ě<g��W       �	ro��Xc�A�*

loss�/%<��-H       �	���Xc�A�*

loss!��<Iu�R       �	̘��Xc�A�*

loss�d~;��       �	�?��Xc�A�*

loss.3�<NW$�       �	�֛�Xc�A�*

loss�� =��^z       �	܀��Xc�A�*

loss��g<\�v*       �	���Xc�A�*

loss�*�;�{       �	3�Xc�A�*

loss~;rw��       �	�\��Xc�A�*

loss'=P8�       �	A�Xc�A�*

loss�lf<	yX=       �	0���Xc�A�*

lossQ�<���,       �	z6��Xc�A�*

loss��;h͊       �	JѠ�Xc�A�*

loss�:wt"�       �	g��Xc�A�*

loss̵�;Qd>9       �	���Xc�A�*

loss]�0<U�5       �	z���Xc�A�*

loss�d';��&U       �	�D��Xc�A�*

loss��d;c���       �	�ޣ�Xc�A�*

loss ��<�FD1       �	)y��Xc�A�*

loss�yn<�44�       �	���Xc�A�*

loss�%�=���       �	����Xc�A�*

loss�W;lI�<       �	�`��Xc�A�*

loss��q;rh�{       �	����Xc�A�*

loss�T�<����       �	4���Xc�A�*

loss�X;����       �	�~��Xc�A�*

loss��=���%       �	���Xc�A�*

loss���<X�?�       �	 ũ�Xc�A�*

loss4<�<1��H       �	�a��Xc�A�*

loss�5=BM       �	���Xc�A�*

loss�?�<q'#       �	(���Xc�A�*

loss�&.=;H�$       �	KY��Xc�A�*

losst�<{�7^       �	z���Xc�A�*

loss�3=���       �	����Xc�A�*

loss͡>j        �	v8��Xc�A�*

lossC�<�I       �	S该Xc�A�*

loss��;��b�       �	ҋ��Xc�A�*

loss��1;HM�       �	-��Xc�A�*

loss���=J=��       �	$ұ�Xc�A�*

loss�$0<�W�"       �	�u��Xc�A�*

loss���;���>       �	��Xc�A�*

loss�,�<�+i       �	����Xc�A�*

loss���:}�{�       �	RG��Xc�A�*

lossI<#�ű       �	�ﴥXc�A�*

loss*��<��7       �	����Xc�A�*

loss��W=]���       �	Yi��Xc�A�*

loss�1=��       �	��Xc�A�*

loss�iG;3��       �	�÷�Xc�A�*

loss]b�<���       �	zl��Xc�A�*

lossaJe;,�o       �	�8��Xc�A�*

loss��=�ɷ        �	�ݹ�Xc�A�*

loss4�<VE�^       �	����Xc�A�*

loss}�:��B       �	*��Xc�A�*

loss�@<e˥�       �	�̻�Xc�A�*

loss��4;(p�v       �	Qj��Xc�A�*

loss�=7=%C%�       �	��Xc�A�*

lossY�<mo�P       �	����Xc�A�*

loss?��=DD�%       �	
L��Xc�A�*

loss��C=��5       �	h쾥Xc�A�*

loss�� =2]o
       �	6���Xc�A�*

loss�x�;��7b       �	-��Xc�A�*

lossx�^;�)c�       �	 ���Xc�A�*

loss�\<I���       �	7m��Xc�A�*

loss�_<�#s       �	0¥Xc�A�*

lossN�<S\�       �	i�¥Xc�A�*

losså�=��}7       �	ZEåXc�A�*

lossN)T<.78�       �	e�åXc�A�*

lossߓ4=Lr��       �	�wĥXc�A�*

loss���;��,2       �	�ťXc�A�*

loss<�e=���V       �	��ťXc�A�*

loss��I;���4       �	.=ƥXc�A�*

loss�$=�8q�       �	��ƥXc�A�*

loss��;=J#fI       �	��ǥXc�A�*

lossĶ�;1d��       �	�SȥXc�A�*

lossz��;
��M       �	��ȥXc�A�*

loss���<'l�       �	6�ɥXc�A�*

lossh}@;�ڏ�       �	4ʥXc�A�*

loss.i7=��       �	��ʥXc�A�*

lossMc�<�|��       �	�e˥Xc�A�*

loss@�E;t���       �	�˥Xc�A�*

lossֆ�;VdK       �	��̥Xc�A�*

lossr��= �Ѣ       �	L6ͥXc�A�*

loss77q<Aot�       �	�ΥXc�A�*

loss;;=J�"�       �	O�ΥXc�A�*

loss��3<���V       �	��ϥXc�A�*

lossv*C=&�       �	��ХXc�A�*

loss���;�o��       �	&SѥXc�A�*

lossz�< R�       �	��ѥXc�A�*

loss��;$	��       �	��ҥXc�A�*

loss��=~�=       �	�5ӥXc�A�*

loss�)<��~c       �	��ӥXc�A�*

loss�e<W#       �	�zԥXc�A�*

loss�:�%�R       �	MեXc�A�*

loss���:��h       �	ɯեXc�A�*

loss�Ŏ=�m��       �	C֥Xc�A�*

loss�<�:��       �	j�֥Xc�A�*

loss�<=B       �	�xץXc�A�*

loss�!�;%�S*       �	�إXc�A�*

loss��= ˻|       �	5٥Xc�A�*

loss6�:��HZ       �	�٥Xc�A�*

loss�m�<O�&       �	iSڥXc�A�*

loss4�c<;lf�       �	
�ڥXc�A�*

loss6k<7���       �	ЛۥXc�A�*

loss��<��K       �	ʣܥXc�A�*

lossS��=MU�       �	2<ݥXc�A�*

loss[�3=�Nܶ       �	,�ݥXc�A�*

lossM�\=���       �	�mޥXc�A�*

lossx�=ڱ�       �	�ߥXc�A�*

loss���<Y�h�       �	N�ߥXc�A�*

loss<q;��6       �	%[�Xc�A�*

loss�;��       �	���Xc�A�*

loss��;G�-       �	>��Xc�A�*

loss@ F=  ��       �	�2�Xc�A�*

loss�ry=V"�       �	��Xc�A�*

loss;�	=]       �	�h�Xc�A�*

loss�7=�sV       �	i�Xc�A�*

loss�Ԝ;�m	�       �	��Xc�A�*

lossO�o<���       �	�9�Xc�A�*

loss�<њ&�       �	���Xc�A�*

lossh"�<��       �	qs�Xc�A�*

loss�iG=��#�       �	��Xc�A�*

loss,A�;@6W8       �	���Xc�A�*

loss�M;�ݥz       �	��Xc�A�*

loss�H�=�O�       �	D5�Xc�A�*

loss`�k;�ˁ�       �	���Xc�A�*

loss���<�ɘ       �	#h�Xc�A�*

lossĭ=�{��       �	�Xc�A�*

loss�L&<�-DI       �	���Xc�A�*

loss��Y<A�        �	����Xc�A�*

lossj�k;l|��       �	�8�Xc�A�*

losst�H<�*       �	��Xc�A�*

lossԨg<v���       �	ӆ�Xc�A�*

loss�	�;��_       �	6�Xc�A�*

loss�x@;$qH�       �	N��Xc�A�*

loss%��;Wt��       �	�T�Xc�A�*

lossC��;���       �	�U�Xc�A�*

loss�l=�H�L       �	'�Xc�A�*

loss$g=M���       �	,��Xc�A�*

lossc�H<���M       �	"T��Xc�A�*

loss�Ù<���:       �	�,��Xc�A�*

loss�[0=�S��       �	����Xc�A�*

loss:�P;ΎDR       �	���Xc�A�*

loss\��=���       �	�'��Xc�A�*

loss�,;;�N�.       �	���Xc�A�*

loss�W;��9�       �	,a��Xc�A�*

loss ��=ت��       �	c'��Xc�A�*

loss�H<-�A�       �	����Xc�A�*

loss�xd=Ih&�       �	W_��Xc�A�*

loss�rq;Z�x       �	׊��Xc�A�*

loss�^�;�q�       �	�&��Xc�A�*

loss\��;+] �       �	����Xc�A�*

loss}��;x5M       �	4f��Xc�A�*

loss��5;���       �	���Xc�A�*

loss]�#<�sT2       �	����Xc�A�*

loss}��;�*�       �		7��Xc�A�*

losssms=9�C       �	���Xc�A�*

loss�:8�       �	rp �Xc�A�*

loss�n:)-j       �	��Xc�A�*

loss!Y9<K��<       �	���Xc�A�*

loss?
<�j�%       �	�@�Xc�A�*

loss��~;��%�       �	
��Xc�A�*

loss���;6n	R       �	;n�Xc�A�*

losso�<��       �	� �Xc�A�*

loss�y<�͙       �	���Xc�A�*

loss�I%=�E�%       �	[�Xc�A�*

loss�Ɋ<e��S       �	���Xc�A�*

loss��<�i�       �	��Xc�A�*

lossk$�<���       �	�1�Xc�A�*

lossl�;�"&       �	���Xc�A�*

loss��#=Eo3�       �	�l�Xc�A�*

lossid�<й��       �	 	�Xc�A�*

loss��=W�y�       �	}�	�Xc�A�*

loss�(�=,���       �	0+
�Xc�A�*

lossD��; ��       �	��
�Xc�A�*

loss��n<�7s       �	<h�Xc�A�*

loss��<=����       �	��Xc�A�*

loss�OE;���       �	���Xc�A�*

loss,��;,F�3       �	[B�Xc�A�*

lossm�;Ww�       �	N
�Xc�A�*

loss|��=�JS       �	i��Xc�A�*

lossQ9�<�h
�       �	�`�Xc�A�*

lossυ�<�c�       �	���Xc�A�*

loss���;l��       �	h��Xc�A�*

lossV �=� �       �	�4�Xc�A�*

loss���:����       �	���Xc�A�*

lossa:�<%��@       �	u��Xc�A�*

loss�A�<	�       �	(�Xc�A�*

loss,�k=�J-.       �	�,�Xc�A�*

loss=��<���q       �	���Xc�A�*

lossܲ�<_�R�       �	�I�Xc�A�*

loss��-<�ޠ�       �	���Xc�A�*

loss[�<FC�       �	�z�Xc�A�*

lossW��;�{�       �	z�Xc�A�*

lossv;��m�       �	��Xc�A�*

loss(� <ꘝ       �	3S�Xc�A�*

loss�aa;�d��       �	p��Xc�A�*

lossO�C;Y��       �	S��Xc�A�*

loss0�<G�w�       �	�.�Xc�A�*

loss��m<����       �	nM�Xc�A�*

loss�!
=;6>       �	&�Xc�A�*

loss�̖=h�Ҥ       �	���Xc�A�*

loss��==N�)x       �	i�Xc�A�*

loss���<��?�       �	w �Xc�A�*

loss6�<m�7�       �	x� �Xc�A�*

loss��;@��q       �	�Z!�Xc�A�*

lossź<�MP7       �	��!�Xc�A�*

loss��#=x�8�       �	נ"�Xc�A�*

loss��;���G       �	K=#�Xc�A�*

loss�h�<�%̜       �	=�#�Xc�A�*

lossd9�<���s       �	�~$�Xc�A�*

loss?�<4̈́,       �	�%�Xc�A�*

loss�+:�Kx       �	8�%�Xc�A�*

loss�¼9��]C       �	�Y&�Xc�A�*

loss(ѣ=k�\�       �	
�&�Xc�A�*

loss�o;���S       �	b�'�Xc�A�*

lossE��;��m�       �	�;(�Xc�A�*

loss(�:<V|hL       �	��(�Xc�A�*

lossJ��<��ݲ       �	�)�Xc�A�*

loss��;�6e�       �	�*�Xc�A�*

loss��b<�Jy�       �	F�*�Xc�A�*

loss�ޛ<^�]�       �	&S+�Xc�A�*

loss�'5;��       �	G�+�Xc�A�*

loss�e<�JD       �	k�,�Xc�A�*

loss���=u�sb       �	!-�Xc�A�*

loss[};�       �	ܻ-�Xc�A�*

loss>��<���       �	C�.�Xc�A�*

loss�ь=�8��       �	�w/�Xc�A�*

loss�eH;�� �       �	#0�Xc�A�*

loss�';���       �	��0�Xc�A�*

loss`��;��       �	�j1�Xc�A�*

loss=6�=��       �	�2�Xc�A�*

loss�(Y;y���       �	��2�Xc�A�*

lossh_x=����       �	M3�Xc�A�*

loss_�I=s��"       �	o/4�Xc�A�*

loss��<����       �	%�4�Xc�A�*

loss�)&;U7'G       �	�a5�Xc�A�*

loss�n;;h�XN       �	�5�Xc�A�*

lossa1�<4��       �	��6�Xc�A�*

loss�z�;P@dm       �	�$7�Xc�A�*

loss4��=I��       �	�7�Xc�A�*

loss(J|=RR��       �	\8�Xc�A�*

loss*i=���w       �	n49�Xc�A�*

lossE��;�D��       �	��9�Xc�A�*

loss�#�<��2�       �	�\:�Xc�A�*

lossD4=+\�d       �	 ;�Xc�A�*

loss�/H<$(�       �	��;�Xc�A�*

loss��c;~P�       �	�-<�Xc�A�*

loss��<>�m�       �	��<�Xc�A�*

loss��:���       �	PT=�Xc�A�*

loss�mS=L�Ϲ       �	\�=�Xc�A�*

loss_��<#%#       �	�}>�Xc�A�*

loss�><����       �	e?�Xc�A�*

loss��:��\       �	��?�Xc�A�*

loss�;�<$��Y       �	rO@�Xc�A�*

lossoj<�4��       �	B�@�Xc�A�*

loss�;8<��+       �	�~A�Xc�A� *

loss\ǝ;EũG       �	�B�Xc�A� *

loss�8 =�n�       �	��B�Xc�A� *

lossW�<����       �	{MC�Xc�A� *

loss/B;�j�       �	��C�Xc�A� *

loss���<�{�       �	�D�Xc�A� *

loss��;�Z��       �	yWE�Xc�A� *

loss�7�<Y��G       �	��E�Xc�A� *

loss�`<wݤ�       �	��F�Xc�A� *

loss�`=��>f       �	�,G�Xc�A� *

loss�֡;]�#       �	 �G�Xc�A� *

loss�2�;@|��       �	�vH�Xc�A� *

loss��;SK9       �	fI�Xc�A� *

loss�"�<�Ola       �	��I�Xc�A� *

loss��<+��J       �	xcJ�Xc�A� *

loss�0�:�kF       �	G9K�Xc�A� *

loss�$�<�3��       �	fML�Xc�A� *

loss�a�;��,�       �	��L�Xc�A� *

loss4Vf<)�)       �	��M�Xc�A� *

loss@��;��g       �	M�N�Xc�A� *

loss��*<�؛       �	�O�Xc�A� *

lossEn�<y4       �	$P�Xc�A� *

loss�r�<�uN�       �	�P�Xc�A� *

loss��<W�YT       �	�nQ�Xc�A� *

loss(K�<|��y       �	�VR�Xc�A� *

loss$X;LP��       �	�R�Xc�A� *

loss��<(�n�       �	��S�Xc�A� *

lossQ��:MW�       �	ڐT�Xc�A� *

loss
�q;F
$U       �	!<U�Xc�A� *

loss)AU<����       �	S�U�Xc�A� *

losst�r;��       �	��V�Xc�A� *

loss(8�<���K       �	/3W�Xc�A� *

loss�݀;GW5       �	g�W�Xc�A� *

loss�d�=���       �	LnX�Xc�A� *

loss��=o�IO       �	EY�Xc�A� *

loss��;�6��       �	[�Y�Xc�A� *

lossB�;�h@�       �	�ZZ�Xc�A� *

loss��\;�?|�       �	�Z�Xc�A� *

loss�:���M       �	ߧ[�Xc�A� *

loss4��:���       �	B?\�Xc�A� *

loss�:�ǭ       �	��\�Xc�A� *

loss3;���       �	�]�Xc�A� *

loss���;(�'<       �	�/^�Xc�A� *

loss��;��g       �	1�^�Xc�A� *

loss|��;�lG�       �	w_�Xc�A� *

loss^��9���2       �	�`�Xc�A� *

loss�92:�67       �	��`�Xc�A� *

loss/\:��       �	Ia�Xc�A� *

loss�=��Y�       �	;�a�Xc�A� *

loss11h<�*s�       �	��b�Xc�A� *

loss�0�;"�u�       �	Cc�Xc�A� *

loss��m:g       �	ضc�Xc�A� *

loss��;:�rh       �	�Wd�Xc�A� *

loss�SP=Ϣ�       �	��d�Xc�A� *

loss�;��#*       �	��e�Xc�A� *

loss;�9>KN�       �	zf�Xc�A� *

loss�<+���       �	��f�Xc�A� *

loss�o�<E��'       �	�Ng�Xc�A� *

lossw�<�}�)       �	l�g�Xc�A� *

lossC�y<Ʃ�I       �	9}h�Xc�A� *

loss�7�<G��       �	+i�Xc�A� *

loss6f;dJ       �	��i�Xc�A� *

loss�/l<U:x�       �	�]j�Xc�A� *

loss��";]�Z       �	�6k�Xc�A� *

loss��.<%K       �	��k�Xc�A� *

loss��=���       �	fl�Xc�A� *

loss��==�`"       �	�m�Xc�A� *

loss���<l{F       �	��m�Xc�A� *

loss��<Fr�       �	��n�Xc�A� *

loss�c�=��z       �	Go�Xc�A� *

loss�h=v.�       �	��o�Xc�A� *

lossN]<�˹�       �	Z�p�Xc�A� *

lossO��=S���       �	m q�Xc�A� *

loss��<�ZU�       �	@�q�Xc�A� *

loss�@<{��       �	�ar�Xc�A� *

lossp۞<Z���       �	~s�Xc�A� *

loss��1= @�R       �	Ut�Xc�A� *

loss!�U;.#<{       �	��t�Xc�A� *

loss���:�U       �	+lu�Xc�A� *

loss��Q<.�	�       �	ov�Xc�A� *

lossc�=0]��       �	̵v�Xc�A� *

loss���<�э       �	�Yw�Xc�A� *

lossڳ|<��O�       �	�x�Xc�A� *

lossѺ=<�]�       �	��x�Xc�A� *

loss��;�ZB       �	REy�Xc�A� *

lossÞS=��H       �	�y�Xc�A� *

lossm��<���       �	�z�Xc�A� *

loss\l9<��o       �	:@{�Xc�A� *

loss:�;x�]       �	6�{�Xc�A� *

losszpA;�� 0       �	��|�Xc�A� *

lossc��<��{       �	Y6}�Xc�A� *

loss��=@�M�       �	|�}�Xc�A� *

loss\�_=����       �	mq~�Xc�A� *

loss#��=�ɉ�       �	��Xc�A� *

loss�{�;|^<       �	ع�Xc�A� *

loss1z<����       �	�Q��Xc�A� *

loss��y<��P       �	�ꀦXc�A� *

loss�Y=<��d�       �	;���Xc�A� *

lossS�<����       �	�l��Xc�A� *

loss���<��       �	5��Xc�A� *

loss$kr=h�ƕ       �	����Xc�A� *

lossip�<�r�       �	����Xc�A� *

loss���<�Z�       �	Fz��Xc�A� *

loss\w8<��       �	���Xc�A� *

loss�p�;D�P7       �	����Xc�A� *

loss�<<6�a(       �	ƞ�Xc�A� *

loss���<�7�       �	�]��Xc�A� *

loss1�<��`�       �	��Xc�A� *

lossǩ#;5�\&       �	����Xc�A� *

loss 1<!g��       �	���Xc�A� *

loss<=5�C       �	2���Xc�A� *

loss+�;��       �	�C��Xc�A� *

loss��;��3       �	�բ�Xc�A� *

loss#��<��pR       �	_}��Xc�A� *

lossD<�;���       �	���Xc�A� *

loss�4<'F��       �	���Xc�A� *

loss�?�;�i_�       �	�M��Xc�A� *

loss�b�=���       �	�륦Xc�A� *

loss���<�{       �	����Xc�A� *

loss�m�<��       �	w-��Xc�A� *

loss:�=���       �	����Xc�A� *

loss�C�;�R�       �	�W��Xc�A� *

lossvt;9�:       �	����Xc�A� *

losss�W;?���       �	%���Xc�A� *

loss�Ș<�S�f       �	80��Xc�A� *

loss�ύ<�jl�       �	�ͪ�Xc�A� *

lossZm�=���P       �	�f��Xc�A� *

loss��<ȋ��       �	����Xc�A� *

loss�Y(=�/^k       �	-���Xc�A�!*

loss��_;p΀&       �	w-��Xc�A�!*

loss�{�;�0��       �	4֭�Xc�A�!*

lossXH�:Nc��       �	Q���Xc�A�!*

loss�1�;"��       �	u��Xc�A�!*

lossa��=�cDo       �	����Xc�A�!*

lossJ�7;P8{       �	�d��Xc�A�!*

loss6.<3��e       �	���Xc�A�!*

loss���:�:Gz       �	����Xc�A�!*

lossD8;8�E�       �	�B��Xc�A�!*

losszu;6�@D       �	A��Xc�A�!*

loss4�;?r�_       �	���Xc�A�!*

loss�.=_:M       �	�D��Xc�A�!*

loss{0O:H��       �	Qڴ�Xc�A�!*

loss�]!=���       �	�{��Xc�A�!*

loss�t=�sn�       �	R��Xc�A�!*

loss$��<(�bc       �	£��Xc�A�!*

loss�̓<�$��       �	�N��Xc�A�!*

loss��J=X�       �	��Xc�A�!*

loss���;�_�	       �	R���Xc�A�!*

loss/��<��o       �	�G��Xc�A�!*

loss���<�	y�       �	�鹦Xc�A�!*

loss%�;��p�       �	^ֺ�Xc�A�!*

loss���;\g       �	�t��Xc�A�!*

loss�<	�|       �	��Xc�A�!*

loss*=�L��       �	����Xc�A�!*

lossx�:��@9       �	H��Xc�A�!*

loss5�:�;C       �	�Xc�A�!*

loss(��;��q�       �	ʊ��Xc�A�!*

loss��;H���       �	�#��Xc�A�!*

loss�F==�+��       �	'���Xc�A�!*

loss��l<Z�)       �	\U��Xc�A�!*

lossfH�;'��       �	���Xc�A�!*

loss��;J�wi       �	����Xc�A�!*

loss���:~��3       �	�<¦Xc�A�!*

loss=�{<<���       �	�¦Xc�A�!*

loss@b�;��x!       �	�æXc�A�!*

loss:��=GP��       �	�+ĦXc�A�!*

loss�,�<�^��       �	E�ĦXc�A�!*

loss�<��!�       �	qŦXc�A�!*

loss:T�<��2�       �	�ƦXc�A�!*

loss� �;ȱ�       �	��ƦXc�A�!*

loss��<I}�       �	FCǦXc�A�!*

loss�k
=� H�       �	��ǦXc�A�!*

lossd�:���       �	:wȦXc�A�!*

lossһs=c��       �	0ɦXc�A�!*

loss�F�</k�       �	b�ɦXc�A�!*

loss���;�XӋ       �	CUʦXc�A�!*

loss�I0=D���       �	�˦Xc�A�!*

lossc�Z;Yu�       �	Ϡ˦Xc�A�!*

loss�e;�k�I       �	"6̦Xc�A�!*

loss�P=E�"@       �	�̦Xc�A�!*

loss�L�;E:�       �	jͦXc�A�!*

lossߓa:��i       �	�ΦXc�A�!*

loss��<t��(       �	�ϦXc�A�!*

loss�R:�X�'       �	�ϦXc�A�!*

loss&X�<�6�B       �	XЦXc�A�!*

loss��L<L�nf       �	ѦXc�A�!*

lossq<�       �	4�ѦXc�A�!*

loss(k�;u��        �	�1ҦXc�A�!*

loss��:<�v�       �	��ҦXc�A�!*

loss��;�� w       �	_\ӦXc�A�!*

loss�5*<��#       �	��ӦXc�A�!*

loss�+=J��&       �	��ԦXc�A�!*

lossu=�^��       �	�(զXc�A�!*

loss��;%��:       �	�զXc�A�!*

lossƒ<�8e       �	�]֦Xc�A�!*

lossh�<)b       �	�֦Xc�A�!*

lossH��<�X       �	ƋצXc�A�!*

loss�<���       �	-@ئXc�A�!*

lossa��:�)�s       �	٦Xc�A�!*

loss̆";���       �	�٦Xc�A�!*

loss�H�</��       �	�6ڦXc�A�!*

loss=���       �	]ۦXc�A�!*

losso(<�&�       �	Q�ۦXc�A�!*

lossn[�:#$��       �	�`ܦXc�A�!*

lossm�;,�        �	[
ݦXc�A�!*

loss�%<�       �	��ݦXc�A�!*

loss���9]�)       �	VHަXc�A�!*

lossLe�<�R�       �	��ަXc�A�!*

lossB;�L�       �	�xߦXc�A�!*

loss,<|;�m3K       �	��Xc�A�!*

loss��I=� I@       �	��Xc�A�!*

loss� :��       �	ZF�Xc�A�!*

lossH��:wV(       �	��Xc�A�!*

loss4O�<Q��       �	Kr�Xc�A�!*

loss��;Iܼ^       �	��Xc�A�!*

loss��\<�U.<       �	V��Xc�A�!*

lossxM"<���       �	�V�Xc�A�!*

loss� �<
��l       �	k��Xc�A�!*

loss�;�u\�       �	/��Xc�A�!*

loss�0<s*�A       �	�c�Xc�A�!*

loss%�w;M�       �	���Xc�A�!*

loss�;��W       �	���Xc�A�!*

loss}��<O�UU       �	(C�Xc�A�!*

losscD>�r�b       �	L��Xc�A�!*

loss��B=D�G�       �	�y�Xc�A�!*

loss��<�V$e       �	�Xc�A�!*

lossp�<���       �	i��Xc�A�!*

lossz��;�\�{       �	���Xc�A�!*

loss\ug;��s8       �	�Xc�A�!*

loss��I;7ﱼ       �	��Xc�A�!*

loss�˸;	�;%       �	eR��Xc�A�!*

loss�o>�1�L       �	���Xc�A�!*

loss���<iXV�       �	�}�Xc�A�!*

loss�z<R ��       �	�2�Xc�A�!*

loss��	=+lV       �	��Xc�A�!*

loss���;�2�       �	)��Xc�A�!*

loss�(
=F50r       �	e��Xc�A�!*

loss�_�<�z��       �	�0�Xc�A�!*

loss8��:�iR7       �	��Xc�A�!*

loss���<98�       �	jh�Xc�A�!*

lossB�	<);�
       �	q��Xc�A�!*

losso�=ƞr�       �	����Xc�A�!*

loss��<;N�̯       �	~T��Xc�A�!*

lossJ~�:G�dR       �	����Xc�A�!*

loss�ž:|NMe       �	���Xc�A�!*

loss[,<xh3g       �	$��Xc�A�!*

lossnaN=��z�       �	���Xc�A�!*

loss�Ø=B �       �	�S��Xc�A�!*

lossE�j<�Q)       �	����Xc�A�!*

loss�+�<�Jq�       �	E���Xc�A�!*

loss}�	<��       �	��Xc�A�!*

loss��8;UR��       �	���Xc�A�!*

loss��;����       �	�T��Xc�A�!*

loss��;6Gj�       �	4���Xc�A�!*

lossJs<`�G       �	����Xc�A�!*

lossM΁<r���       �	�$��Xc�A�!*

loss��b<v�       �	����Xc�A�"*

loss��=߿�       �	�_��Xc�A�"*

loss��=����       �	����Xc�A�"*

loss�Ŏ<�%P"       �	����Xc�A�"*

loss��9uٴ�       �	" �Xc�A�"*

loss��;�i��       �	�� �Xc�A�"*

loss��==�Ǌ       �	�Z�Xc�A�"*

lossM�<���       �	n��Xc�A�"*

loss-�;0��       �	���Xc�A�"*

loss�L�<���5       �	2�Xc�A�"*

losss�<���       �	���Xc�A�"*

lossm3�;3���       �	�g�Xc�A�"*

lossf�;�[��       �	O�Xc�A�"*

loss�|D<|�Go       �	���Xc�A�"*

loss�l;J�       �	�q�Xc�A�"*

loss��<#t�       �	_�Xc�A�"*

loss�1�:���[       �	��Xc�A�"*

loss㮒<T��       �	�C�Xc�A�"*

loss���:�w�(       �	���Xc�A�"*

loss�|.:�B��       �	�m	�Xc�A�"*

loss�6�:�S��       �	l
�Xc�A�"*

loss_��;.\�       �	b�
�Xc�A�"*

loss���=K�        �	�;�Xc�A�"*

lossjY�:���       �	���Xc�A�"*

lossɒ�=����       �	���Xc�A�"*

loss�^�<\E��       �	���Xc�A�"*

loss��<rɨ"       �	���Xc�A�"*

loss&�<����       �	p?�Xc�A�"*

lossh>;��Kv       �	C8�Xc�A�"*

loss��9�
�       �	���Xc�A�"*

loss]��<��[*       �	@��Xc�A�"*

loss��:~��       �	�`�Xc�A�"*

loss�:��Ĭ       �	�	�Xc�A�"*

loss{�z:�:*       �	9E�Xc�A�"*

lossd]=Q �       �	�3�Xc�A�"*

lossu�=�rn�       �	!�Xc�A�"*

loss��<ӊ�,       �	���Xc�A�"*

loss�R�<�\�       �	Tn�Xc�A�"*

loss�p�<��Q       �	y�Xc�A�"*

losseu�<�)�Z       �	�<�Xc�A�"*

loss�v;�٨�       �	���Xc�A�"*

loss@~Q<}k�T       �	��Xc�A�"*

loss��=@��0       �	�6�Xc�A�"*

loss��n;T�
�       �	0�Xc�A�"*

lossZ�<���_       �	��Xc�A�"*

loss�RK<l/��       �	���Xc�A�"*

losse�=�Љ�       �	�q�Xc�A�"*

loss�t�=�.       �	�G�Xc�A�"*

loss���:��Y       �	��Xc�A�"*

loss^=�e,       �	� �Xc�A�"*

loss��*;#�ri       �	��!�Xc�A�"*

loss��P<4J��       �	�p"�Xc�A�"*

loss��"<�*U       �	�#�Xc�A�"*

lossU� =��       �	P�#�Xc�A�"*

loss�<K)y�       �	�p$�Xc�A�"*

loss,�<e�*6       �	�	%�Xc�A�"*

loss\��<���       �	b�%�Xc�A�"*

loss���;Ol�6       �	ut&�Xc�A�"*

loss��<��       �	x'�Xc�A�"*

loss���<g�n       �	��'�Xc�A�"*

loss���=la[R       �	<K(�Xc�A�"*

loss��<[;{Y       �	�(�Xc�A�"*

loss�x<	��X       �	��)�Xc�A�"*

loss
��;���l       �	S*�Xc�A�"*

loss�uZ=�i��       �	��*�Xc�A�"*

loss<�=�et�       �	�+�Xc�A�"*

loss|��<�CQ       �	1,�Xc�A�"*

loss%��<KE�       �	��,�Xc�A�"*

loss,�<��*�       �	n-�Xc�A�"*

loss�ې<�c�       �	g.�Xc�A�"*

loss�K;�}u�       �	�.�Xc�A�"*

lossH�(<��l       �	�u/�Xc�A�"*

loss�z<��       �	�	0�Xc�A�"*

lossԹ�9Rm�       �	¡0�Xc�A�"*

loss�=�`a       �	9D1�Xc�A�"*

lossv�V<���W       �	��1�Xc�A�"*

loss�!W; �a       �	��2�Xc�A�"*

loss�T<�2�[       �	�(3�Xc�A�"*

loss��a<�afy       �	�3�Xc�A�"*

lossxϳ:�/�       �	4f4�Xc�A�"*

lossjQ(<M�       �	5�Xc�A�"*

loss�ƿ=L[6i       �	v�5�Xc�A�"*

loss�~<]�`       �	1D6�Xc�A�"*

lossD�5<[��       �	��6�Xc�A�"*

loss��#;���       �	�7�Xc�A�"*

loss�U�<>�q       �	�8�Xc�A�"*

loss��<P�O       �	s�8�Xc�A�"*

loss�֣<5��       �	UO9�Xc�A�"*

loss%�':��%       �	��9�Xc�A�"*

loss��E<���       �	�~:�Xc�A�"*

loss.�C<��8D       �	�;�Xc�A�"*

loss�F�<���9       �	W�;�Xc�A�"*

loss��;��.       �	4H<�Xc�A�"*

loss�l=��:�       �	��<�Xc�A�"*

loss�Q�<�&~       �	�y=�Xc�A�"*

loss�>�;X���       �	<>�Xc�A�"*

lossG��;�<��       �	��>�Xc�A�"*

loss��a;�,       �	�J?�Xc�A�"*

lossH˛<��.�       �	��?�Xc�A�"*

loss��<r�zN       �	��@�Xc�A�"*

loss6�=�uzd       �	�*A�Xc�A�"*

loss���;�R�a       �	r�A�Xc�A�"*

loss���<��C�       �	geB�Xc�A�"*

lossY�9F��       �	W
C�Xc�A�"*

loss�o�<{�       �	v�C�Xc�A�"*

lossJĵ;P��K       �	�>D�Xc�A�"*

lossD@<��       �	��D�Xc�A�"*

loss1*<
n       �	�~E�Xc�A�"*

loss���;����       �	�F�Xc�A�"*

loss�<|�|       �	ܺF�Xc�A�"*

lossZ͆<:�!�       �	�\G�Xc�A�"*

loss��:=_��       �	��G�Xc�A�"*

loss�ъ<��`       �	�H�Xc�A�"*

loss�%V;�=X]       �	�=I�Xc�A�"*

loss�q;2�8       �	�I�Xc�A�"*

lossτA<����       �	1zJ�Xc�A�"*

lossSb<��`I       �	K�Xc�A�"*

loss�Tj<��3       �	��K�Xc�A�"*

loss���;t��       �	kHL�Xc�A�"*

loss�<k᫱       �	��L�Xc�A�"*

lossV6�;���       �	�pM�Xc�A�"*

lossƦz<����       �	9
N�Xc�A�"*

loss{�8=�d�       �	6�N�Xc�A�"*

loss�!�<���       �	
KO�Xc�A�"*

losst?O=���       �	2uP�Xc�A�"*

loss�+^<�x.       �	�)Q�Xc�A�"*

loss	�*:e��       �	��Q�Xc�A�"*

loss��=���       �	�R�Xc�A�"*

loss-O�<�r[       �	֪S�Xc�A�#*

loss8Q�9����       �	�eT�Xc�A�#*

loss;�i;��.h       �	�WU�Xc�A�#*

loss�#;�3�       �	+V�Xc�A�#*

lossty�<�;S�       �	��V�Xc�A�#*

loss8�W<�ۜ{       �	W�W�Xc�A�#*

loss��<fE*.       �	6=X�Xc�A�#*

loss��<�v�       �	0�X�Xc�A�#*

loss�E=;8u��       �	ԀY�Xc�A�#*

lossT9=����       �	{Z�Xc�A�#*

loss���:��x�       �	J�Z�Xc�A�#*

lossxT�:{t��       �	K[�Xc�A�#*

loss �;��1�       �	�[�Xc�A�#*

loss�{;��9A       �	1y\�Xc�A�#*

loss�~j:�x�w       �	�]�Xc�A�#*

loss-�<.�I�       �	 �]�Xc�A�#*

loss�C;g�Vx       �	W^^�Xc�A�#*

loss��<����       �	�^�Xc�A�#*

loss�z=����       �	C�_�Xc�A�#*

loss�b^=���Z       �	�-`�Xc�A�#*

loss  J=�(u       �	��`�Xc�A�#*

loss�ێ=�'A�       �	�oa�Xc�A�#*

loss��!=���q       �	�b�Xc�A�#*

loss�W�:=��       �	��b�Xc�A�#*

lossE�R=ڟI�       �	/2c�Xc�A�#*

loss累<��V       �	I�c�Xc�A�#*

lossw��<�&�:       �	Zd�Xc�A�#*

loss�C�:�͈�       �	�e�Xc�A�#*

loss@>5��'       �	��e�Xc�A�#*

lossN��;�B��       �	�bf�Xc�A�#*

loss��<�w       �	�Gh�Xc�A�#*

loss2s�; ���       �	��h�Xc�A�#*

loss���:�P�I       �	�i�Xc�A�#*

loss?5<�6�       �	t�j�Xc�A�#*

lossm�;<A��.       �	�[k�Xc�A�#*

loss&!<�ZC�       �	�l�Xc�A�#*

losso*P<J��       �	�&m�Xc�A�#*

loss��9�q_p       �	?�m�Xc�A�#*

loss�n�;'��F       �	�on�Xc�A�#*

loss|�<���[       �	Do�Xc�A�#*

loss��,=z�	�       �	ٰo�Xc�A�#*

loss55�<w�L�       �	?Sp�Xc�A�#*

loss�M�<�s�(       �	W�p�Xc�A�#*

loss���<I��r       �	8�q�Xc�A�#*

loss�~;�n       �	�r�Xc�A�#*

loss3W5; i        �	Ӿr�Xc�A�#*

loss�U=��%       �	�\s�Xc�A�#*

loss!O;�v�       �	x't�Xc�A�#*

loss@/�<�G�F       �	#�t�Xc�A�#*

loss���;}�.       �	�du�Xc�A�#*

loss@�=n��2       �	�v�Xc�A�#*

loss�><��X�       �	�v�Xc�A�#*

loss	�<9��Y       �	H2w�Xc�A�#*

loss=@:id�       �	��w�Xc�A�#*

loss6��<_�rB       �	^x�Xc�A�#*

loss��<!c�S       �	�x�Xc�A�#*

lossh�;��/�       �	A�y�Xc�A�#*

losszn=���       �	zSz�Xc�A�#*

lossJ,�<�7�L       �	�z�Xc�A�#*

loss6�<	Co       �	��{�Xc�A�#*

loss!�(;N��Z       �	.|�Xc�A�#*

loss�}<f��       �	��|�Xc�A�#*

lossh�;�� Y       �	�n}�Xc�A�#*

loss�I�=8�o�       �	�	~�Xc�A�#*

loss4�)=^���       �	Ӥ~�Xc�A�#*

loss�.=�M�u       �	�=�Xc�A�#*

loss���=L-[�       �	���Xc�A�#*

loss��:<B�       �	�{��Xc�A�#*

loss  �;3���       �	f��Xc�A�#*

loss�w<ݍ4D       �	x���Xc�A�#*

lossM/_<���       �	^��Xc�A�#*

loss #�:��AQ       �	���Xc�A�#*

loss��;�9�6       �	K���Xc�A�#*

lossDn�<�K�       �	�S��Xc�A�#*

loss��s:H��       �	{��Xc�A�#*

loss�"�<@	�       �	[���Xc�A�#*

loss!��;��G       �	�j��Xc�A�#*

loss�&�;2P�F       �	���Xc�A�#*

lossqc=�ցJ       �	G���Xc�A�#*

loss�<�=��R       �	�k��Xc�A�#*

loss���<��{S       �	K��Xc�A�#*

losso��;A��       �	ϟ��Xc�A�#*

loss�B;���       �	�8��Xc�A�#*

lossR�A<X�N�       �	�ӊ�Xc�A�#*

loss��<�	�t       �	�y��Xc�A�#*

lossl�e<�a�       �	���Xc�A�#*

lossbG=�M       �	Y���Xc�A�#*

loss ��<�xܹ       �	W��Xc�A�#*

lossa�;ad�       �	w�Xc�A�#*

loss�g�;z;<�       �	뎎�Xc�A�#*

loss��<��       �	|���Xc�A�#*

loss��<����       �	*��Xc�A�#*

loss��;��K�       �	:ː�Xc�A�#*

loss�t�<䖧�       �	�d��Xc�A�#*

loss�[;TI]�       �	����Xc�A�#*

loss���;��ݠ       �	����Xc�A�#*

lossRn�<�x!�       �	�E��Xc�A�#*

lossW�=���       �	 ᓧXc�A�#*

loss\�g<�I�B       �	����Xc�A�#*

loss蒢<�4��       �	2��Xc�A�#*

loss�j�<����       �	𿕧Xc�A�#*

loss��;�H4       �	�Z��Xc�A�#*

loss\J�:۔�3       �	����Xc�A�#*

loss<�=��       �	ٖ��Xc�A�#*

lossyf�;8��       �	�4��Xc�A�#*

loss���:r�	�       �	d̘�Xc�A�#*

loss���<R!z       �	{i��Xc�A�#*

loss
�<wX�       �	���Xc�A�#*

loss7v�<&ɑ�       �	���Xc�A�#*

loss=�M=m�<�       �	>?��Xc�A�#*

loss�@=�o/t       �	�뛧Xc�A�#*

loss�u�:����       �	U���Xc�A�#*

loss���;�~�       �	�j��Xc�A�#*

loss���:���       �	"���Xc�A�#*

loss��=��>       �	���Xc�A�#*

lossZ�=��d       �	+4��Xc�A�#*

loss!
<�-�U       �	Oϟ�Xc�A�#*

lossі�9x޴n       �	�h��Xc�A�#*

loss���<bi��       �	����Xc�A�#*

losse��;��       �	����Xc�A�#*

loss���<�[T�       �	�1��Xc�A�#*

loss���;��       �	Z٢�Xc�A�#*

lossqؒ<���"       �	Ḳ�Xc�A�#*

loss�2S<F��       �	Va��Xc�A�#*

loss�{<9C       �	����Xc�A�#*

loss���=d��       �	B���Xc�A�#*

lossS<�I�       �	1��Xc�A�#*

loss9�<��a�       �	�˦�Xc�A�#*

loss�I�;��I       �	�m��Xc�A�$*

loss_��;#\ 0       �	-
��Xc�A�$*

lossC;q^5       �	;���Xc�A�$*

lossD�W==��       �	?��Xc�A�$*

loss�}}="a�       �	"ߩ�Xc�A�$*

lossM�5<,wZ       �	x~��Xc�A�$*

lossZ+�;���       �	p(��Xc�A�$*

loss���<�'Z       �	����Xc�A�$*

lossȹ�;�Gw/       �	P���Xc�A�$*

loss�$K<扠�       �	tD��Xc�A�$*

lossi=��t�       �	rޭ�Xc�A�$*

loss)�<�8o�       �	R~��Xc�A�$*

loss&�:<�{�       �	c%��Xc�A�$*

loss��_:�%j�       �	���Xc�A�$*

loss:ҥ<ӈ�       �	���Xc�A�$*

loss!ao;b��	       �	�*��Xc�A�$*

loss#�L=��W       �	aƱ�Xc�A�$*

loss9�;�9�       �	�c��Xc�A�$*

lossO;�<;��       �	l
��Xc�A�$*

loss��b;�kD�       �	����Xc�A�$*

loss�+[:��       �	jg��Xc�A�$*

loss�?8<S�b�       �	����Xc�A�$*

lossF+U;)���       �	�`��Xc�A�$*

loss���;<�"�       �	����Xc�A�$*

loss��<1]$�       �	I���Xc�A�$*

loss���;v	1�       �	9��Xc�A�$*

loss��;AI(�       �	���Xc�A�$*

lossj�<r��       �	*���Xc�A�$*

lossS�;4]V�       �	�3��Xc�A�$*

loss��<���E       �	㺧Xc�A�$*

loss�$�<�1x�       �	�x��Xc�A�$*

loss�٪<c�=�       �	�a��Xc�A�$*

lossܯ�:DPM       �	���Xc�A�$*

loss(G;�x�\       �	����Xc�A�$*

loss�=�A�       �	�U��Xc�A�$*

loss%��:�8��       �	~���Xc�A�$*

loss<j�<�j4�       �	����Xc�A�$*

loss@�;�t�.       �	����Xc�A�$*

loss�Ʃ<}# 0       �	���Xc�A�$*

loss��=;bc��       �	�a§Xc�A�$*

loss�}�<Ouc�       �	�çXc�A�$*

loss�H=���       �	�lħXc�A�$*

loss�@B<^�aV       �	#2ŧXc�A�$*

loss�=�P�       �	\�ŧXc�A�$*

loss=��; l84       �	c�ƧXc�A�$*

loss
g:س�}       �	�BǧXc�A�$*

loss��:b	3       �	^�ǧXc�A�$*

loss���=��       �	��ȧXc�A�$*

loss{S�;��F       �	Z.ɧXc�A�$*

loss77F<��г       �	��ɧXc�A�$*

loss��z;�t�       �	l|ʧXc�A�$*

loss&.=�',�       �	X˧Xc�A�$*

loss��4=�V��       �	�˧Xc�A�$*

loss& I=a�z       �	%\̧Xc�A�$*

lossd�=d_@{       �	~ͧXc�A�$*

loss�]�;�:V       �	ٙͧXc�A�$*

loss���=�ޯm       �	�?ΧXc�A�$*

loss��<�5pQ       �	Q3ϧXc�A�$*

loss�=;�ô�       �	5�ϧXc�A�$*

loss�B;�p��       �	@kЧXc�A�$*

lossú�:*���       �	ѧXc�A�$*

losse<?	�A       �	|�ѧXc�A�$*

lossW�7:-�k�       �	2ҧXc�A�$*

lossj�Z=#х�       �	$�ҧXc�A�$*

loss6�<a�2�       �	vӧXc�A�$*

loss���<���A       �	}!ԧXc�A�$*

loss���<��${       �	��ԧXc�A�$*

loss��V<!���       �	�jէXc�A�$*

loss�r<;%8\4       �	>֧Xc�A�$*

loss���:ǹ��       �	m�֧Xc�A�$*

loss��p<���       �	�FקXc�A�$*

loss���:�v9�       �	��קXc�A�$*

lossrU�=O�       �	A}اXc�A�$*

losshs�=�ڣ}       �	�٧Xc�A�$*

loss̫�<��	�       �	��٧Xc�A�$*

loss�b�<%"^       �	�sڧXc�A�$*

loss�܀=ЕS�       �	|(ۧXc�A�$*

loss�*�=�T��       �	��ۧXc�A�$*

loss��:�[iu       �	�\ܧXc�A�$*

loss��B<݀lo       �	SݧXc�A�$*

loss�" =���j       �	��ݧXc�A�$*

loss!�:�4�       �	ZFާXc�A�$*

loss�tn;+�8l       �	��ާXc�A�$*

lossU�:��U�       �	_�ߧXc�A�$*

loss�F�<��1j       �	�;�Xc�A�$*

loss��F<�B��       �	P��Xc�A�$*

loss���<X��       �	8��Xc�A�$*

loss��:VgU�       �	�.�Xc�A�$*

loss�E:j>�B       �	���Xc�A�$*

loss��</���       �	}�Xc�A�$*

loss�b<�a)       �	��Xc�A�$*

loss.� <ˡz�       �	D��Xc�A�$*

loss�Ӛ<x���       �	�\�Xc�A�$*

loss�"�<��g       �	��Xc�A�$*

loss@V�;���       �	b��Xc�A�$*

loss_��;�R�       �	TT�Xc�A�$*

loss�!�;$P��       �	���Xc�A�$*

loss7C�<9*�       �	���Xc�A�$*

loss�<<m�~       �	�E�Xc�A�$*

loss��<ǰ�       �	f��Xc�A�$*

loss��n;��ǅ       �	�7�Xc�A�$*

loss�Û;��{�       �	���Xc�A�$*

loss�p<��jh       �	�f�Xc�A�$*

loss��;<ߐ�       �	���Xc�A�$*

loss��!<���0       �	�(�Xc�A�$*

loss�U	;NY$       �	���Xc�A�$*

loss�'�<�w       �	.V�Xc�A�$*

loss}��;�->�       �	���Xc�A�$*

loss֞�<�Q;       �	υ�Xc�A�$*

loss�N�<G7        �	��Xc�A�$*

loss�p�:���5       �	��Xc�A�$*

loss�%G=+�N�       �	�K�Xc�A�$*

loss��<��       �	S��Xc�A�$*

loss���8X?�       �	���Xc�A�$*

loss���;�5       �	Id��Xc�A�$*

loss�: m1�       �	����Xc�A�$*

loss��:� ��       �	���Xc�A�$*

lossmĊ<�rڶ       �	�4��Xc�A�$*

loss!1I<lE9b       �	����Xc�A�$*

lossV?C;��       �	er��Xc�A�$*

loss�<�ut       �	�r��Xc�A�$*

lossc >��       �	�6��Xc�A�$*

loss��9.@d�       �	���Xc�A�$*

loss�jW<�.:       �	jg��Xc�A�$*

loss&#�<�B       �	2���Xc�A�$*

loss*��:QN��       �	����Xc�A�$*

lossN0w92���       �	UK��Xc�A�$*

loss");*�s       �	h���Xc�A�$*

loss��:�c�;       �	���Xc�A�%*

loss��G:�[�I       �	�V��Xc�A�%*

loss#;�ޖ�       �	����Xc�A�%*

loss�*P:�$�,       �	r���Xc�A�%*

loss�><jy�o       �	? �Xc�A�%*

loss�7�9��>�       �	�� �Xc�A�%*

loss�e<��H�       �	(c�Xc�A�%*

lossE@�7��{�       �	z��Xc�A�%*

loss�R;�W       �	m��Xc�A�%*

loss�'#=�f�@       �	�H�Xc�A�%*

loss8PS;����       �	���Xc�A�%*

lossJΈ94�       �	���Xc�A�%*

lossh�Z<u�v       �	�Xc�A�%*

loss��=W�       �	d��Xc�A�%*

lossH��:��       �	IM�Xc�A�%*

loss/��=�­       �	y��Xc�A�%*

loss�d�<]*=       �	q��Xc�A�%*

loss}��<W&`�       �	i��Xc�A�%*

loss}�;<��H<       �	�,	�Xc�A�%*

lossaJ</��       �	%�	�Xc�A�%*

lossy$=��I       �	4i
�Xc�A�%*

loss�<�-�       �	W�Xc�A�%*

loss�:�=�Ro�       �	ߣ�Xc�A�%*

lossF�.<� �       �	�9�Xc�A�%*

loss3q�;��7#       �	��Xc�A�%*

lossH�:M|       �	/��Xc�A�%*

loss�ov=f��n       �	}<�Xc�A�%*

lossn�;���z       �	���Xc�A�%*

loss�j�;=ok�       �	/��Xc�A�%*

loss��	=��M�       �	V��Xc�A�%*

lossd�<>�ZN       �	.:�Xc�A�%*

lossf��;���J       �	B
�Xc�A�%*

loss���<f'��       �	��Xc�A�%*

lossrW;I��       �	l��Xc�A�%*

loss�#;�BU�       �	�h�Xc�A�%*

loss<��<�'�L       �	���Xc�A�%*

loss��x<J��3       �	)�Xc�A�%*

loss#o9��       �	:z�Xc�A�%*

loss�Q:~T�c       �	�$�Xc�A�%*

loss u3;�/�4       �	5��Xc�A�%*

loss��;��       �	��Xc�A�%*

loss*�6<��       �	�$�Xc�A�%*

loss���:���       �	��Xc�A�%*

losst2�;a��       �	�c�Xc�A�%*

loss�˘<U��g       �	`�Xc�A�%*

loss#�C=�XG�       �	��Xc�A�%*

loss��j<vA��       �	�K�Xc�A�%*

loss�D�;��q       �	)��Xc�A�%*

loss��p;��^�       �	8��Xc�A�%*

loss�p;2�T�       �	%�Xc�A�%*

loss(�<��p       �	n2 �Xc�A�%*

lossA3S=���       �	�� �Xc�A�%*

loss��<��W�       �	@l!�Xc�A�%*

losss�:h�       �	2U"�Xc�A�%*

lossk�:�O�       �	��"�Xc�A�%*

lossfT�;��       �	7�#�Xc�A�%*

loss#�3<�>�d       �	uu$�Xc�A�%*

loss\��;eTå       �	zl%�Xc�A�%*

loss �;\�>       �	P&�Xc�A�%*

lossa��;�yه       �	\�&�Xc�A�%*

loss���<����       �	�f'�Xc�A�%*

lossw��;9�i       �	�(�Xc�A�%*

loss��;5btN       �	U�(�Xc�A�%*

loss�7,;���N       �	�^)�Xc�A�%*

loss��H9}�u@       �	�)�Xc�A�%*

loss���;F�)       �	>�D�Xc�A�%*

lossv�u<��de       �	^JE�Xc�A�%*

lossF�<�� �       �	��E�Xc�A�%*

loss\�<��T       �	��F�Xc�A�%*

loss�=26§       �	�G�Xc�A�%*

loss�a�<����       �	��G�Xc�A�%*

loss6�b;��H�       �	kbH�Xc�A�%*

loss��;�A��       �	#I�Xc�A�%*

lossl�Q<"���       �	��I�Xc�A�%*

loss�	<`1}�       �	�gJ�Xc�A�%*

loss��:�m��       �	qK�Xc�A�%*

loss��<�u       �	��K�Xc�A�%*

loss:��<d��5       �	�8L�Xc�A�%*

lossO�=�m�       �	 �L�Xc�A�%*

loss�=ŽϦ       �	�rM�Xc�A�%*

lossĢi<@q��       �	aN�Xc�A�%*

loss���9��       �	�N�Xc�A�%*

loss
�==��0z       �	�PO�Xc�A�%*

loss�æ<��]`       �	(�O�Xc�A�%*

loss}E<)y       �	��P�Xc�A�%*

losss�(<&�`       �	N%Q�Xc�A�%*

loss�8�=��cy       �	m�Q�Xc�A�%*

loss��v:�{\�       �	�^R�Xc�A�%*

loss|�=��       �	}S�Xc�A�%*

lossqS�<���       �	q�S�Xc�A�%*

loss�|�;�W�       �	MLT�Xc�A�%*

loss��{<�_	�       �	�T�Xc�A�%*

loss(8?;7�gN       �	��U�Xc�A�%*

loss΂�<�9C�       �	�#V�Xc�A�%*

loss��J<҂��       �	��V�Xc�A�%*

loss�&=�c^�       �	5^W�Xc�A�%*

loss�:	iT       �	��W�Xc�A�%*

loss�<ǵ�       �	x�X�Xc�A�%*

loss8��<@p>       �	�8Y�Xc�A�%*

loss�5=�+��       �	V�Y�Xc�A�%*

loss��;)/       �	�jZ�Xc�A�%*

loss�;yA]'       �	�`[�Xc�A�%*

lossc�<�d�       �	Q�[�Xc�A�%*

loss�F�<�L�       �	�\�Xc�A�%*

loss�9�<s�4!       �	^*]�Xc�A�%*

loss
#<��(       �	��]�Xc�A�%*

lossV�<=+�Z       �	�^�Xc�A�%*

lossZ#:��>�       �	�)_�Xc�A�%*

loss��<�;�       �	��_�Xc�A�%*

lossR!A<o\Z       �	uW`�Xc�A�%*

loss��=�3       �	E�`�Xc�A�%*

loss-��:|M�       �	��a�Xc�A�%*

loss��Q=�
�       �	p'b�Xc�A�%*

lossU��:nVb�       �	8�b�Xc�A�%*

lossw��9�f`3       �	vTc�Xc�A�%*

lossS�(;f��       �	_�c�Xc�A�%*

loss�<B=J<8�       �	��d�Xc�A�%*

loss���;�       �	�e�Xc�A�%*

loss�ho=�Ge       �	�3f�Xc�A�%*

loss�;=o��       �	��f�Xc�A�%*

loss|�~;@��L       �	�ag�Xc�A�%*

lossRJ�:~[&       �	��g�Xc�A�%*

loss�b�:�~0�       �	�h�Xc�A�%*

loss4�;����       �	(*i�Xc�A�%*

loss��=;F+       �	��i�Xc�A�%*

loss�E>t��       �	�_j�Xc�A�%*

loss��;[���       �	2k�Xc�A�%*

loss��%<&<J�       �	��l�Xc�A�%*

loss� �;��.	       �	9Fn�Xc�A�&*

loss�Le;=oge       �	ōo�Xc�A�&*

loss��::�U�       �	 *p�Xc�A�&*

loss*�y;�-�V       �	F�p�Xc�A�&*

loss1�M;e       �	��q�Xc�A�&*

loss�&=/-J       �	]�r�Xc�A�&*

loss<L�<$��       �	�Xs�Xc�A�&*

loss��-;����       �	�	t�Xc�A�&*

loss�6<6)ZN       �	��t�Xc�A�&*

loss#�}=�}I       �	~Wu�Xc�A�&*

loss���;?�Y       �	��u�Xc�A�&*

loss�Z<DF�       �	��v�Xc�A�&*

loss{:<l�B�       �	!Yw�Xc�A�&*

loss�Ln;
ۈR       �	��w�Xc�A�&*

loss��;��       �	Έx�Xc�A�&*

loss�g�<";��       �	�[y�Xc�A�&*

loss=o���       �	!z�Xc�A�&*

lossù;1��       �	q�z�Xc�A�&*

loss�=�.s(       �	�~{�Xc�A�&*

loss��l;W�p�       �	�@|�Xc�A�&*

losstЇ;����       �	o�|�Xc�A�&*

lossage9$=,L       �	��}�Xc�A�&*

loss�/f;�\E"       �	~�Xc�A�&*

loss|�<B�n       �	9�~�Xc�A�&*

loss�;�=A       �	�Q�Xc�A�&*

loss��%<Q7W�       �	��Xc�A�&*

loss<�C<Wۗ.       �	����Xc�A�&*

loss��y</�o�       �	���Xc�A�&*

loss?�+=8�K�       �	���Xc�A�&*

loss]f=Yx       �	����Xc�A�&*

loss!p8=ǽ�L       �	\V��Xc�A�&*

loss��<�n.       �	��Xc�A�&*

loss�}:@�8       �	*���Xc�A�&*

lossN�<���       �	�(��Xc�A�&*

lossN<<©��       �	����Xc�A�&*

loss�]=���       �	aS��Xc�A�&*

lossEN;o�|N       �	T���Xc�A�&*

loss#�9�o��       �	����Xc�A�&*

loss��:���]       �	�6��Xc�A�&*

losst��<��J�       �	�Ȉ�Xc�A�&*

loss�3�<M���       �	_^��Xc�A�&*

loss�D=zV��       �	���Xc�A�&*

loss@A<��P       �	���Xc�A�&*

loss#��:��j       �	"P��Xc�A�&*

lossD�:(��S       �	틨Xc�A�&*

loss!׍<!Dvs       �	���Xc�A�&*

loss۰�;���(       �	{0��Xc�A�&*

loss��/<�yG*       �	:ɍ�Xc�A�&*

loss���:V�       �	^���Xc�A�&*

loss}�:���t       �	��Xc�A�&*

loss��:}�2�       �	����Xc�A�&*

loss�I�;���       �	#K��Xc�A�&*

loss.�7=v��I       �	*��Xc�A�&*

loss��:0��       �	b���Xc�A�&*

lossL�:;Y��       �	"R��Xc�A�&*

loss�x;��#`       �	�蒨Xc�A�&*

loss��W=X�J       �	X���Xc�A�&*

loss���;�.:�       �	$(��Xc�A�&*

loss]a�<�R�       �	�Ҕ�Xc�A�&*

loss��:�N)�       �	�h��Xc�A�&*

loss��R<Xs�       �	 ��Xc�A�&*

loss}�&:�<       �	0���Xc�A�&*

loss�x�;��ޥ       �	�3��Xc�A�&*

loss���;�?�       �	}̗�Xc�A�&*

loss<j=:��s�       �	Lp��Xc�A�&*

loss�S�;��"�       �	F��Xc�A�&*

lossm7=��,�       �	*ə�Xc�A�&*

loss�Y�<u��)       �	�a��Xc�A�&*

loss_�;�^��       �	�	��Xc�A�&*

loss��K<�O"�       �	𤛨Xc�A�&*

loss�;;��h9       �	A��Xc�A�&*

loss�<N�A       �	�՜�Xc�A�&*

loss��<�g&�       �	Ɔ��Xc�A�&*

loss�C=fF       �	�"��Xc�A�&*

loss,�;x��2       �	8���Xc�A�&*

lossC�:ɓ�[       �	}Y��Xc�A�&*

lossh�&:ߵ+�       �	��Xc�A�&*

lossH��;��       �	R��Xc�A�&*

loss�_<CC�!       �	���Xc�A�&*

loss�"<E+�X       �	H7��Xc�A�&*

loss���<I8�       �	�΢�Xc�A�&*

loss\?D=����       �	�b��Xc�A�&*

loss﭅;��p�       �	����Xc�A�&*

losst��;"�P�       �	�E��Xc�A�&*

loss�9z$��       �	ޥ�Xc�A�&*

lossNm�9ﰯ       �	2s��Xc�A�&*

lossc�v;^�	�       �	
��Xc�A�&*

loss���;��       �	Ԟ��Xc�A�&*

loss5�<4�T�       �	L8��Xc�A�&*

loss	�;h���       �	/ܨ�Xc�A�&*

loss��;/-}       �	�p��Xc�A�&*

loss�p6;��f       �	���Xc�A�&*

loss�S/:S��       �	����Xc�A�&*

loss7�;u���       �	�.��Xc�A�&*

losse�:����       �	vë�Xc�A�&*

loss� C<���Z       �	�T��Xc�A�&*

loss���<81�       �	ﬨXc�A�&*

loss߆#<:xrF       �	8���Xc�A�&*

losstA�;��       �	�2��Xc�A�&*

loss/�;4�       �	�֮�Xc�A�&*

lossÉ'8}� �       �	�u��Xc�A�&*

loss�ϴ:@�V�       �	�U��Xc�A�&*

lossj/n;ۜ�       �	����Xc�A�&*

loss�� :GqB�       �	����Xc�A�&*

lossR��<�N�       �	oH��Xc�A�&*

loss��=֟�5       �	nݲ�Xc�A�&*

lossM<��È       �	�t��Xc�A�&*

losstr=�i�       �	���Xc�A�&*

loss
�~:�       �	"���Xc�A�&*

lossk�;�;��       �	+���Xc�A�&*

loss1�==֎��       �	���Xc�A�&*

loss��~9���       �	be��Xc�A�&*

loss�cN;yD��       �	����Xc�A�&*

loss��<�X��       �	���Xc�A�&*

loss4."<m�w       �	�a��Xc�A�&*

loss=��=2G,�       �	����Xc�A�&*

loss�W1;�_,i       �	(���Xc�A�&*

lossO��:��`�       �	X8��Xc�A�&*

loss1P]={       �	WXc�A�&*

loss�=K��       �	����Xc�A�&*

loss-ݰ;�!��       �	�3��Xc�A�&*

loss���<��RQ       �	����Xc�A�&*

loss��;}��V       �	����Xc�A�&*

lossaͦ:��       �	#/��Xc�A�&*

loss���:PCd       �	SϿ�Xc�A�&*

losswN@<%,"       �	�b��Xc�A�&*

loss�Z�<I��       �	v���Xc�A�&*

lossE/=B��       �	h���Xc�A�&*

loss6�#<nXg�       �	oc¨Xc�A�'*

loss-�;�q�       �	��¨Xc�A�'*

loss���;�P/�       �	��èXc�A�'*

lossN�:l��?       �	�0ĨXc�A�'*

loss3�%;��*       �	D�ĨXc�A�'*

loss��;}Q/�       �	�XŨXc�A�'*

loss��=�m�       �	_�ŨXc�A�'*

loss�o�=�@�q       �	��ƨXc�A�'*

lossf�:�z�T       �	�>ǨXc�A�'*

lossӴS=��O.       �	��ǨXc�A�'*

loss�4�9��'�       �	LnȨXc�A�'*

loss��3<�`�       �	XɨXc�A�'*

loss7)�<�&Rb       �	��ɨXc�A�'*

loss{�:�M       �	�RʨXc�A�'*

loss:�=���       �	��ʨXc�A�'*

lossn٧<t��X       �	K�˨Xc�A�'*

loss �*=
'|       �	�2̨Xc�A�'*

lossSs�<�y        �	"�̨Xc�A�'*

loss�2N<nt��       �	�dͨXc�A�'*

loss,C�;��X�       �	��ͨXc�A�'*

loss[ͻ:��L�       �	؛ΨXc�A�'*

loss�5<�v
w       �	�7ϨXc�A�'*

loss�M<?���       �	��ϨXc�A�'*

loss/_=��j       �	Q�ШXc�A�'*

lossAg�<�u2r       �	.WѨXc�A�'*

loss?� =1q�       �	,�ѨXc�A�'*

loss !|<6.K       �	�ҨXc�A�'*

loss�zU<-u?<       �	ӨXc�A�'*

loss�ka;��IL       �	��ӨXc�A�'*

loss���;֚�x       �	TԨXc�A�'*

loss�p;��l�       �	��ԨXc�A�'*

loss�y�;�c:v       �	)�ըXc�A�'*

lossF�0<п�       �	l]֨Xc�A�'*

loss��<܄�	       �	��֨Xc�A�'*

loss��&<"Iy�       �	C�רXc�A�'*

loss�);#�~�       �	l#بXc�A�'*

loss��\<�σk       �	�بXc�A�'*

loss�_<���$       �	�]٨Xc�A�'*

loss��:W���       �	ڨXc�A�'*

loss���;,\       �	�ڨXc�A�'*

lossn((<����       �	�MۨXc�A�'*

loss���:`���       �	��ۨXc�A�'*

loss��=jw�Y       �	j�ܨXc�A�'*

lossë�;�8d5       �	PݨXc�A�'*

loss�1�;qy7       �	�ݨXc�A�'*

loss���;��       �	ӃިXc�A�'*

loss�;D=�fQ8       �	p'ߨXc�A�'*

loss�Qe<���+       �	{�ߨXc�A�'*

loss�o";��b       �	�f�Xc�A�'*

loss�w�:��i^       �	H��Xc�A�'*

loss�}<����       �	���Xc�A�'*

loss-��<�V��       �	�'�Xc�A�'*

loss�0]<H�:�       �	���Xc�A�'*

loss��<Ϊ��       �	�y�Xc�A�'*

loss׊�<�I�       �	�Xc�A�'*

loss_�p;4���       �	Ʀ�Xc�A�'*

loss_�Z;�%9�       �	>Y�Xc�A�'*

lossd>�; 1�       �	z��Xc�A�'*

lossA�>;�L       �	-��Xc�A�'*

loss�4�;��/�       �	
1�Xc�A�'*

loss+ϖ<�'m�       �	���Xc�A�'*

loss_]<r�       �	�h�Xc�A�'*

lossr��:�fF       �	�Xc�A�'*

loss�y�9G]��       �	!��Xc�A�'*

lossz~5=�aH       �	�A�Xc�A�'*

loss@|-<�zQ       �	���Xc�A�'*

loss�	<ڒ�       �	x�Xc�A�'*

loss��:	]       �	v�Xc�A�'*

loss�h4=�]M       �	D��Xc�A�'*

lossg=���%       �	W`��Xc�A�'*

loss���:�vk       �	����Xc�A�'*

lossL�<�;�       �	u��Xc�A�'*

lossX'P<oگ�       �	�)�Xc�A�'*

loss�\x=�n�u       �	���Xc�A�'*

lossK�;��y       �	t|�Xc�A�'*

loss�<i17�       �	��Xc�A�'*

loss)�=}'��       �	
��Xc�A�'*

lossͨ�;���       �	j�Xc�A�'*

loss�ng<V�oq       �	�Xc�A�'*

loss��B=��R       �	ƿ�Xc�A�'*

loss�A;�B�7       �	~���Xc�A�'*

loss�̊;�댇       �	z���Xc�A�'*

loss��:��@�       �	`[��Xc�A�'*

loss�G=2A��       �	V���Xc�A�'*

loss��<�$�)       �	?���Xc�A�'*

loss/�C;��{�       �		o��Xc�A�'*

loss���9k��       �	���Xc�A�'*

loss��<PM       �	���Xc�A�'*

loss�ɫ<�ˍ�       �	ak��Xc�A�'*

loss]��8�A�;       �	�	��Xc�A�'*

losss).=c�ծ       �	���Xc�A�'*

loss1��:����       �	l��Xc�A�'*

lossV�+<��l�       �	���Xc�A�'*

loss�M�;��\       �	����Xc�A�'*

loss�L=u���       �	G��Xc�A�'*

lossn�W;����       �	<���Xc�A�'*

lossD :xEJ�       �	�z��Xc�A�'*

loss���:�)�       �	� �Xc�A�'*

loss���:Cz8�       �	2� �Xc�A�'*

loss�s,:����       �	�H�Xc�A�'*

loss�Ҳ=Izz9       �	��Xc�A�'*

loss
��9�o	�       �	�~�Xc�A�'*

loss �;H�       �	�<�Xc�A�'*

loss�	G<}��L       �	s��Xc�A�'*

loss�n<���>       �	1��Xc�A�'*

loss��d<kR�n       �	�E�Xc�A�'*

loss-+�<'Ӗ�       �	b��Xc�A�'*

loss �%<�"�?       �	&m�Xc�A�'*

loss��s=C�k�       �	P�Xc�A�'*

lossf�<҈x�       �	p��Xc�A�'*

loss�w�;e>�       �	�\�Xc�A�'*

loss*�=��3       �	��Xc�A�'*

loss�}o=0p       �	��	�Xc�A�'*

loss��;���       �	�7
�Xc�A�'*

loss_*�=ɬ��       �	��
�Xc�A�'*

lossi�:<����       �	�i�Xc�A�'*

lossC=�=c�j       �	��Xc�A�'*

loss�V�:s)b       �	H��Xc�A�'*

lossa��:k`�e       �	�J�Xc�A�'*

lossR�;�$V?       �	���Xc�A�'*

losss}*<r9д       �	h��Xc�A�'*

loss��';�4��       �	�0�Xc�A�'*

lossA�F;��~       �	��Xc�A�'*

lossyn�:�X�#       �	��Xc�A�'*

loss�I�:���       �	)Z�Xc�A�'*

loss=;TQ�       �	.�Xc�A�'*

loss
�<l	(       �	��Xc�A�'*

loss�=2�       �	�;�Xc�A�'*

loss-DK<ty3       �	E��Xc�A�(*

loss �u;���       �	���Xc�A�(*

loss��==+�       �	�Q�Xc�A�(*

lossm�<��ڙ       �	���Xc�A�(*

lossA�9)�       �	��Xc�A�(*

loss�rU;�B�       �	�P�Xc�A�(*

loss� 	;cY!       �	���Xc�A�(*

loss�<áƦ       �	���Xc�A�(*

lossV�<�!h       �	e�Xc�A�(*

loss_�<��.0       �	���Xc�A�(*

loss�Z=�T       �	�G�Xc�A�(*

loss}��;$®�       �	���Xc�A�(*

loss��:����       �		��Xc�A�(*

loss��|:GH��       �	:�Xc�A�(*

losst�=�i       �	��Xc�A�(*

loss��<�]#�       �	�i�Xc�A�(*

loss���<��       �	�2�Xc�A�(*

loss�;�=��c�       �	2��Xc�A�(*

loss�(�=���H       �	5a�Xc�A�(*

loss�M=n���       �	��Xc�A�(*

loss�.�:�Խ       �	G� �Xc�A�(*

loss�Q�;lU��       �	ԛ!�Xc�A�(*

loss��I;�p�)       �	H2"�Xc�A�(*

loss�-=�r�f       �	��"�Xc�A�(*

lossZY3<�*_X       �	�]#�Xc�A�(*

loss���<#��X       �	��#�Xc�A�(*

loss��'=��       �	G�$�Xc�A�(*

loss��A;�w�b       �	.%�Xc�A�(*

loss�t=n��        �	��%�Xc�A�(*

loss��1=�CO�       �	τ&�Xc�A�(*

loss;�;��=T       �	�'�Xc�A�(*

loss�%�:��~�       �	��'�Xc�A�(*

loss�L�:v��       �	�e(�Xc�A�(*

loss]��<�N��       �	�(�Xc�A�(*

lossvȆ9����       �	6�)�Xc�A�(*

loss]}<MNF       �	�+*�Xc�A�(*

lossڐZ;G(       �	�*�Xc�A�(*

lossS�><0�=�       �	��+�Xc�A�(*

loss�?�<� �       �	�',�Xc�A�(*

loss�7�;8"�g       �	�.-�Xc�A�(*

loss��2<@,')       �	e�-�Xc�A�(*

loss��3<>Jg#       �	-\.�Xc�A�(*

loss�%j=��6       �	��.�Xc�A�(*

lossHљ<��Y�       �	��/�Xc�A�(*

loss�[�<�(Sa       �	�/0�Xc�A�(*

lossӅ�=���x       �	��0�Xc�A�(*

lossey�<�o��       �	�i1�Xc�A�(*

lossv�1<p!c�       �	��1�Xc�A�(*

loss���<�ovQ       �	��2�Xc�A�(*

loss�t�;6ȥk       �	�43�Xc�A�(*

loss9r=��       �	"�3�Xc�A�(*

loss�Ug<lX�       �	_�4�Xc�A�(*

loss�"�:ո�       �	�;6�Xc�A�(*

lossH6!<��c�       �	@�6�Xc�A�(*

loss�O;�=��       �	��7�Xc�A�(*

loss��8=��H�       �	�E8�Xc�A�(*

loss��7< �5�       �	W�8�Xc�A�(*

loss��<O{�"       �	��9�Xc�A�(*

loss�K;�OF       �	<0:�Xc�A�(*

loss7&�<��<�       �	��:�Xc�A�(*

lossa�W<����       �	y�;�Xc�A�(*

loss�ڤ;�&�:       �	-<�Xc�A�(*

loss��;�M�<       �	4�<�Xc�A�(*

loss>��=�wK       �	�=�Xc�A�(*

loss�|�;N��c       �	�)>�Xc�A�(*

loss��;T-P       �	�>�Xc�A�(*

lossj�;=���       �	�~?�Xc�A�(*

loss�&;���V       �	
@�Xc�A�(*

loss�Pb<b~��       �	y�@�Xc�A�(*

loss�;�;�P�       �	xCA�Xc�A�(*

loss��l;��Ȝ       �	��A�Xc�A�(*

loss�J�<+e��       �	�mB�Xc�A�(*

loss��=Р�       �	��B�Xc�A�(*

lossߍ;����       �	�C�Xc�A�(*

loss
Q�<���       �	�,D�Xc�A�(*

loss�<jHc�       �	@�D�Xc�A�(*

loss�A|<4v�       �	�VE�Xc�A�(*

loss#F�:'[I�       �	��F�Xc�A�(*

loss�"�;���       �	�tG�Xc�A�(*

loss_� ;i��~       �	H�Xc�A�(*

lossJZc;���       �	Q�H�Xc�A�(*

loss�6�<�D       �	�2I�Xc�A�(*

loss�q�;���       �	�I�Xc�A�(*

loss���;!���       �	kJ�Xc�A�(*

loss@��:� A�       �	RK�Xc�A�(*

loss��<��D�       �	 �K�Xc�A�(*

loss�<αA       �	�CL�Xc�A�(*

loss3TU<�g�v       �	��L�Xc�A�(*

loss!��<;Ч�       �	�mM�Xc�A�(*

loss�b^;*J�A       �	N�Xc�A�(*

loss��Z;c��       �	�N�Xc�A�(*

loss@��<B�G4       �	�aO�Xc�A�(*

loss��<}�f       �	�P�Xc�A�(*

loss���<ٔ�R       �	��P�Xc�A�(*

loss�X�;�M��       �	�nQ�Xc�A�(*

lossɇY<:N��       �	�	R�Xc�A�(*

loss�,=���       �	i�R�Xc�A�(*

loss#2�:�<W3       �	��S�Xc�A�(*

lossˮ<,:׽       �	]ST�Xc�A�(*

loss�f<dwt�       �	C�T�Xc�A�(*

loss4m�;W}       �	x�U�Xc�A�(*

loss=�<�+k�       �	BAV�Xc�A�(*

loss��6=�_��       �	�V�Xc�A�(*

loss���;Ĕ0       �	��W�Xc�A�(*

losshhK<��q_       �	��X�Xc�A�(*

loss�(�<�t       �	1yY�Xc�A�(*

loss�D;{�<       �	�(Z�Xc�A�(*

loss�
<&De       �	��Z�Xc�A�(*

lossiĄ;^�       �	��[�Xc�A�(*

loss�R�=hO��       �	P\�Xc�A�(*

loss���<�F��       �	�]�Xc�A�(*

loss��;<��{       �	�]�Xc�A�(*

loss��<ȆV       �	,^�Xc�A�(*

loss���:A�!�       �	�^�Xc�A�(*

loss�`@<W��$       �	�V_�Xc�A�(*

lossH�<vq?l       �	��_�Xc�A�(*

loss��;1RKH       �	�}`�Xc�A�(*

lossps�<�W�       �	�a�Xc�A�(*

loss��=P!       �	��a�Xc�A�(*

loss6:��y�       �	h@b�Xc�A�(*

loss�r�<�B�*       �	�b�Xc�A�(*

loss�Ƃ:$��@       �	�pc�Xc�A�(*

loss;{)<��       �	=d�Xc�A�(*

loss2vp<�X��       �	��d�Xc�A�(*

loss,=�8�@       �	5@e�Xc�A�(*

loss`*<k��       �	�e�Xc�A�(*

loss�= ��F       �	�jf�Xc�A�(*

loss��0=��       �	�g�Xc�A�(*

loss�ա<21��       �	.�g�Xc�A�)*

loss���:" 1�       �	�Kh�Xc�A�)*

loss��h;:��       �	��h�Xc�A�)*

loss�=m  �       �	c}i�Xc�A�)*

lossX��;����       �	�j�Xc�A�)*

loss[{;��~       �	O�j�Xc�A�)*

lossy��<�:�0       �	�Gk�Xc�A�)*

loss�ć;3<�       �	��k�Xc�A�)*

lossDe�;l��       �	c|l�Xc�A�)*

loss���:����       �	}$m�Xc�A�)*

loss枔<ܧLx       �	�m�Xc�A�)*

loss�.�:
\��       �	~Vn�Xc�A�)*

lossa8�;�u~�       �	��n�Xc�A�)*

loss��:w��9       �	�o�Xc�A�)*

lossr�;�ⓩ       �	K>p�Xc�A�)*

loss�;���       �	��p�Xc�A�)*

lossab�<���9       �	O�q�Xc�A�)*

loss��=�a��       �	\<r�Xc�A�)*

loss��:}�'       �	X�r�Xc�A�)*

loss��<㘄       �	��s�Xc�A�)*

loss`��:U.��       �	�'t�Xc�A�)*

loss�g9=L�s       �	��t�Xc�A�)*

loss��F<�Iؼ       �	�iu�Xc�A�)*

loss|�>S\�       �	Bv�Xc�A�)*

loss��:JE       �	�v�Xc�A�)*

loss
G�;���&       �	�>w�Xc�A�)*

loss���<�"       �	��w�Xc�A�)*

loss���:J��       �	�x�Xc�A�)*

loss64;H+h9       �	$Dy�Xc�A�)*

loss�*#:h�Z�       �	��y�Xc�A�)*

loss��2<<�2v       �	�{z�Xc�A�)*

loss��!:
5�|       �	�{�Xc�A�)*

loss[�=���       �	��{�Xc�A�)*

loss��2<���       �	h^|�Xc�A�)*

loss���<x8�       �	0�|�Xc�A�)*

loss��;Fv�$       �	r�}�Xc�A�)*

lossGc<����       �	�L~�Xc�A�)*

loss�==IU��       �	�~�Xc�A�)*

loss�r<�|�&       �	�z�Xc�A�)*

loss��<&��       �	���Xc�A�)*

loss�2�<Ʌ�       �	ٲ��Xc�A�)*

loss��;��B       �	�K��Xc�A�)*

loss���=zk�       �	9�Xc�A�)*

lossl��<W�m�       �	���Xc�A�)*

lossv��<zSY&       �	q8��Xc�A�)*

lossW)�:�[&       �	�܃�Xc�A�)*

lossdZ:�52�       �	ۅ��Xc�A�)*

loss��<��#]       �	�4��Xc�A�)*

loss��J=�S�       �	BЅ�Xc�A�)*

loss� 5;l��x       �	�e��Xc�A�)*

loss��;&i��       �	~���Xc�A�)*

losst�f<�ي       �	ĕ��Xc�A�)*

lossTE=�ev�       �	�-��Xc�A�)*

lossSYi<k
��       �	����Xc�A�)*

lossM&<i��{       �	�Y��Xc�A�)*

loss�<�AV�       �	w���Xc�A�)*

loss[�$<�-#_       �	����Xc�A�)*

loss�.Z<��*�       �	6:��Xc�A�)*

loss��=��uy       �	�؋�Xc�A�)*

loss<(Π�       �	}z��Xc�A�)*

loss��(<'�us       �	LP��Xc�A�)*

loss�:ߪ�f       �	;��Xc�A�)*

lossŕ>;�rf       �	����Xc�A�)*

lossXKM<\3�q       �	����Xc�A�)*

loss�4;�X�       �	UO��Xc�A�)*

loss-�S<��[L       �	���Xc�A�)*

loss�@u=rxS       �	ޏ��Xc�A�)*

loss�X<�紶       �	�)��Xc�A�)*

loss��&<S2�       �	�뒩Xc�A�)*

loss�l=��T�       �	����Xc�A�)*

lossX0�<�݆       �	�t��Xc�A�)*

lossc:X��6       �	� ��Xc�A�)*

loss���:!�j       �	���Xc�A�)*

loss	�9�L�(       �	?㖩Xc�A�)*

lossd�;:�9�       �	����Xc�A�)*

loss�xm:�/�       �	ظ��Xc�A�)*

loss��;؟       �	eQ��Xc�A�)*

loss�<���       �	��Xc�A�)*

loss�Z�:M� c       �	u���Xc�A�)*

lossa�;1�:�       �	�(��Xc�A�)*

lossv�:�E�       �	=��Xc�A�)*

loss��<=I��       �	a���Xc�A�)*

lossW��:��k       �	;��Xc�A�)*

loss1!�:��       �	|ҝ�Xc�A�)*

lossz��;:�̚       �	�d��Xc�A�)*

loss/:`�%�       �	����Xc�A�)*

loss;e�8�)�:       �	#���Xc�A�)*

loss�,
9�!r       �	�8��Xc�A�)*

loss�*I=�"�       �	�Ϡ�Xc�A�)*

losszvE94Ǥ-       �	,g��Xc�A�)*

lossLc:���       �	z���Xc�A�)*

loss���9p�M�       �	l���Xc�A�)*

lossP6<,)�       �	�,��Xc�A�)*

loss ��:�3U�       �	�ȣ�Xc�A�)*

loss���69���       �	�e��Xc�A�)*

loss	��7fbq       �	,��Xc�A�)*

loss��N:�zυ       �	P���Xc�A�)*

loss��:-
��       �	BB��Xc�A�)*

loss}�;�`[�       �	�ަ�Xc�A�)*

loss-yy:�b�e       �	�y��Xc�A�)*

lossw[�;<��	       �	���Xc�A�)*

lossS$=S�zI       �	����Xc�A�)*

loss/�9��%�       �	aS��Xc�A�)*

lossf"Z=tJR�       �	����Xc�A�)*

loss��U<N�B       �	d���Xc�A�)*

loss���<i��c       �	}=��Xc�A�)*

loss�̰<���       �	R׫�Xc�A�)*

loss��C9�᱀       �	`r��Xc�A�)*

loss�{
<����       �		��Xc�A�)*

loss��=Ǽڼ       �	���Xc�A�)*

loss��W;L�T�       �	vS��Xc�A�)*

lossv�<<2���       �	#�Xc�A�)*

loss��9j<2n       �	s���Xc�A�)*

loss�=�D9       �	5E��Xc�A�)*

loss��1<�К       �	�㰩Xc�A�)*

loss�+�;R��`       �	��Xc�A�)*

loss�B<�|��       �	�]��Xc�A�)*

lossE�=��Ϣ       �	��Xc�A�)*

loss��<�?�       �	U���Xc�A�)*

loss�`<�}�f       �	#,��Xc�A�)*

loss��:��Q�       �	����Xc�A�)*

lossO�r<���       �	_��Xc�A�)*

loss��<j
�g       �	f���Xc�A�)*

loss��0=�yK       �	�Xc�A�)*

lossM��:�f^�       �	N+��Xc�A�)*

loss�]�=��3       �	w���Xc�A�)*

loss��;��J�       �	�̸�Xc�A�)*

loss���;(       �	Nd��Xc�A�)*

loss�=�=T܀�       �	����Xc�A�**

loss��<���;       �	ŏ��Xc�A�**

loss��3=e��3       �	�*��Xc�A�**

loss���<㴀^       �	�ػ�Xc�A�**

loss@x�;Ԏ�       �	����Xc�A�**

loss}�{;CN��       �	mV��Xc�A�**

loss%�:��       �	]6��Xc�A�**

lossvO;F�       �	�C��Xc�A�**

loss$�n<�8��       �	[鿩Xc�A�**

lossDؑ;�e\f       �	����Xc�A�**

losso;��O�       �	�/��Xc�A�**

lossl��;���       �	i���Xc�A�**

loss�eg<Z���       �	�h©Xc�A�**

loss�><���       �	t
éXc�A�**

loss�*�;�o	       �	��éXc�A�**

loss4|b<x�T�       �	�6ĩXc�A�**

loss*[�;$�+�       �	 �ĩXc�A�**

loss�D�;j�(I       �	�nũXc�A�**

loss�V;��       �	�/ƩXc�A�**

loss���;�2&|       �	u9ǩXc�A�**

lossƣ�<��L<       �	��ǩXc�A�**

loss�s�:˔~�       �	m�ȩXc�A�**

loss��=!w/       �	9+ɩXc�A�**

loss[:�@       �	��ɩXc�A�**

loss�s�;��`       �	:yʩXc�A�**

losso�;;�� '       �	�v�Xc�A�**

loss��7<:Ho�       �	��Xc�A�**

loss�p=3gb       �	���Xc�A�**

loss��_<w�       �	[B�Xc�A�**

lossc{^<�Ӻ1       �	��Xc�A�**

loss��;m�(e       �	���Xc�A�**

loss��<a�j       �	O�Xc�A�**

loss��;�4       �	���Xc�A�**

loss�!=���       �	7O�Xc�A�**

loss��
=�;       �	h��Xc�A�**

loss*-;1?Ɨ       �	{�Xc�A�**

loss��<�v�       �	��Xc�A�**

loss�<��_�       �	���Xc�A�**

loss���:e3�3       �	�8�Xc�A�**

losse7*=��g        �	���Xc�A�**

loss���<$��@       �	6��Xc�A�**

loss�ɠ8��!�       �	/1��Xc�A�**

loss(<lvQ�       �	u���Xc�A�**

loss�*�<����       �	`�Xc�A�**

lossq�=��o)       �	u�Xc�A�**

loss[3c;�RJ       �	���Xc�A�**

loss.i�=�,a       �	�K�Xc�A�**

loss�;;%0�!       �	&��Xc�A�**

loss�G�<��1�       �	x}�Xc�A�**

loss���;O�+I       �	�Xc�A�**

loss��9<D��       �	���Xc�A�**

loss��<�|�       �	O\�Xc�A�**

loss��;@��       �	��Xc�A�**

loss��;Dˬ       �	���Xc�A�**

lossF�X;�d��       �	~7��Xc�A�**

loss��:���%       �	���Xc�A�**

loss�G6<�"�.       �	��Xc�A�**

lossiO<��N�       �	����Xc�A�**

lossme<ƹ�&       �	�u��Xc�A�**

lossL��:�Gh       �	���Xc�A�**

lossn�<����       �	?���Xc�A�**

loss�v�<�4�c       �	�o��Xc�A�**

loss�<���Z       �	W��Xc�A�**

loss_n^<�΃~       �	{���Xc�A�**

lossQB<����       �	x���Xc�A�**

loss���<syK�       �	���Xc�A�**

loss�>Y;_���       �	����Xc�A�**

lossx��<���?       �	j3��Xc�A�**

loss�g
;�L�       �	����Xc�A�**

losshuy<1�a�       �	�� �Xc�A�**

losso��<���5       �	�)�Xc�A�**

loss@p�;�U!       �	��Xc�A�**

lossB��;1&h&       �	�z�Xc�A�**

loss��;�Ѿ       �	��Xc�A�**

loss�r�:J	��       �	���Xc�A�**

loss�0;���]       �	�T�Xc�A�**

loss�h<�)ڥ       �	V��Xc�A�**

loss��z;��_�       �	���Xc�A�**

loss��=%�1�       �	P�Xc�A�**

losss��;��g       �	O��Xc�A�**

loss8��:��       �	��Xc�A�**

lossq�;�ҹX       �	+�Xc�A�**

loss7�9$D�"       �	w��Xc�A�**

loss_9<h��%       �	?W	�Xc�A�**

loss�(�;@���       �	=�	�Xc�A�**

lossW��<t���       �	"�
�Xc�A�**

loss���<���       �	.�Xc�A�**

loss��}9u��<       �	���Xc�A�**

loss��H;���<       �	y��Xc�A�**

loss<f�9r���       �	/�Xc�A�**

lossU�<�T��       �	T��Xc�A�**

loss�ig;�R\       �	a��Xc�A�**

lossFW�<?{�_       �	�2�Xc�A�**

loss�&=��p�       �	���Xc�A�**

loss�y=���       �	h��Xc�A�**

loss|�<Y�ב       �	��Xc�A�**

loss.�[<vS;�       �	��Xc�A�**

loss�:W@d8       �	���Xc�A�**

lossx=�;?x�b       �	�n�Xc�A�**

loss�(�:�@�{       �	�Xc�A�**

loss�"�:��       �	ø�Xc�A�**

loss���9!�	       �	�Z�Xc�A�**

loss�f^;��+u       �	���Xc�A�**

loss�v8:�a8�       �	��Xc�A�**

loss��H<���       �	�%�Xc�A�**

loss� :*�0       �	I��Xc�A�**

lossZ��;�)_       �	�_�Xc�A�**

lossVC}:H�       �	���Xc�A�**

loss�Ȅ:e��       �	Z��Xc�A�**

loss�E;]�8"       �	^�Xc�A�**

lossnV�:����       �	���Xc�A�**

lossA�r;�|�$       �	��Xc�A�**

loss<!���       �	�;�Xc�A�**

loss�;T�7       �	���Xc�A�**

loss��:�<��       �	�t�Xc�A�**

loss��9;�_��       �	U�Xc�A�**

loss���:�MS�       �	-��Xc�A�**

loss�}�:êa       �	jO�Xc�A�**

loss{;ۀ�]       �	���Xc�A�**

loss���< w�       �	�� �Xc�A�**

loss�t�:=R{       �	�G!�Xc�A�**

loss��$=�       �	3�!�Xc�A�**

loss���<� <       �	t|"�Xc�A�**

loss�<l��       �	�#�Xc�A�**

loss��Q<�D��       �	��#�Xc�A�**

loss޸;w�
�       �	/O$�Xc�A�**

loss�е9��j       �	2�$�Xc�A�**

loss�<i`K�       �	�%�Xc�A�**

loss�j�:DUA�       �	|,&�Xc�A�+*

lossJ4:l�       �	��&�Xc�A�+*

loss3}!<O�       �	+h'�Xc�A�+*

loss��<*��       �	��'�Xc�A�+*

lossm��9��D       �	��(�Xc�A�+*

loss�:�;?��5       �	80)�Xc�A�+*

loss�uJ:��<�       �	�)�Xc�A�+*

loss$��;�)�       �	n*�Xc�A�+*

loss2�:��t�       �	7+�Xc�A�+*

loss+�;����       �	D�+�Xc�A�+*

loss|�;�\q       �	Qf,�Xc�A�+*

loss�<~v�       �	�-�Xc�A�+*

lossy;:��Ӆ       �	��-�Xc�A�+*

loss�D�9+A��       �	TS.�Xc�A�+*

loss�<4�       �	@�.�Xc�A�+*

loss�g;!D)        �	O�/�Xc�A�+*

loss��;>�>�       �	�70�Xc�A�+*

loss��b<�	?       �	�0�Xc�A�+*

loss��=wB�        �	�g1�Xc�A�+*

loss�	:&��       �	�2�Xc�A�+*

loss��<��`F       �	��2�Xc�A�+*

loss�*�9�ر       �	�53�Xc�A�+*

loss�9�<~)�       �	�3�Xc�A�+*

loss���<�(\F       �	$C5�Xc�A�+*

loss龿:X<�       �	��5�Xc�A�+*

loss��=�<��       �	tz6�Xc�A�+*

loss_3;}�h+       �	7�Xc�A�+*

loss��9�gj       �	)�7�Xc�A�+*

loss��=���       �	ML8�Xc�A�+*

loss��<��       �	�8�Xc�A�+*

lossc�,=" .�       �	��9�Xc�A�+*

loss�Z;��>+       �	-:�Xc�A�+*

loss�;��       �	 u;�Xc�A�+*

lossΦ�;���L       �	�<�Xc�A�+*

lossJ�<�
ό       �	F�<�Xc�A�+*

lossc�:>68�       �	%\=�Xc�A�+*

loss2�;�TW       �	�~>�Xc�A�+*

loss$�N:�9�       �	�?�Xc�A�+*

loss�/=k�{       �	�:@�Xc�A�+*

loss�9<;��R       �	 �@�Xc�A�+*

loss:&=��m       �	�A�Xc�A�+*

lossE��<��/�       �	
�B�Xc�A�+*

loss 	�;�=�       �	�~C�Xc�A�+*

lossEv�<J��"       �	�$D�Xc�A�+*

loss�c`;���Z       �	�E�Xc�A�+*

loss(�q9��N       �	��E�Xc�A�+*

loss:v�;�2L       �	7�F�Xc�A�+*

loss��z:��]�       �	�-G�Xc�A�+*

loss��=G�M�       �	�]H�Xc�A�+*

losso}	<vIN�       �	i�H�Xc�A�+*

loss�'�;���       �	��I�Xc�A�+*

loss�ѕ<�-��       �	�9J�Xc�A�+*

loss�i�:��\       �	��J�Xc�A�+*

loss�!c<E��       �	,~K�Xc�A�+*

loss�S�=ʕ�       �	mL�Xc�A�+*

lossՌ=�j%       �	M�L�Xc�A�+*

loss<�<,��1       �	$M�Xc�A�+*

loss)�;̤�       �	�#N�Xc�A�+*

loss�d�9dI��       �	�N�Xc�A�+*

loss�k;=2	       �	9_O�Xc�A�+*

loss�&�;�/�*       �	��O�Xc�A�+*

loss{=���       �	�P�Xc�A�+*

lossI5J<��       �	�8Q�Xc�A�+*

loss��>:4�r�       �	��Q�Xc�A�+*

loss}&g;&�V       �	�{R�Xc�A�+*

loss�P<���       �	�!S�Xc�A�+*

lossf�=�㖖       �	��S�Xc�A�+*

lossB =��p�       �	�QT�Xc�A�+*

loss�E;�XB�       �	��T�Xc�A�+*

lossQ�:�'�Y       �	;�U�Xc�A�+*

loss�Zf<���j       �	:V�Xc�A�+*

loss�9<�&��       �	N�V�Xc�A�+*

loss}<쫠d       �	oIW�Xc�A�+*

loss�=����       �	��W�Xc�A�+*

loss�Y;��       �	c~X�Xc�A�+*

loss�S�<
	@�       �	
Y�Xc�A�+*

loss�y�;�&W       �	��Y�Xc�A�+*

loss�e.:=n       �	5AZ�Xc�A�+*

lossH�<jUZ�       �	��Z�Xc�A�+*

losslj�;���       �	+h[�Xc�A�+*

loss���:L��j       �	\�Xc�A�+*

lossߥ':ꛯ       �	j�\�Xc�A�+*

loss�!�;�!�I       �	5]�Xc�A�+*

loss��I=xZ��       �	�]�Xc�A�+*

loss�<>T�       �	Q�^�Xc�A�+*

lossd�s;���       �	PP_�Xc�A�+*

loss�[=;c��       �	G�_�Xc�A�+*

lossH-�=I?H�       �	)y`�Xc�A�+*

lossa�:�ý|       �	Aa�Xc�A�+*

loss(;J8�Y       �	��a�Xc�A�+*

loss��S<,�/Z       �	:b�Xc�A�+*

loss�ћ;��o       �	S�b�Xc�A�+*

lossp�;&���       �	?rc�Xc�A�+*

loss��e:b UQ       �	�d�Xc�A�+*

loss]R=�:       �	��d�Xc�A�+*

loss���;(إ�       �	�@e�Xc�A�+*

loss��0=�U�       �	��e�Xc�A�+*

loss#��<J�        �	�qf�Xc�A�+*

loss�so<�%
       �	�	g�Xc�A�+*

loss��;2�h�       �	0�g�Xc�A�+*

lossD��:��D}       �	q<h�Xc�A�+*

loss�&<G:6q       �	k�h�Xc�A�+*

lossӅ<�3ݐ       �	��i�Xc�A�+*

lossX�<��g�       �	�}j�Xc�A�+*

loss�*=6�8�       �	�k�Xc�A�+*

loss�=֑9       �	��k�Xc�A�+*

loss���;��a^       �	 bl�Xc�A�+*

loss�/<{{       �	�bm�Xc�A�+*

loss�z;T=<       �	=)n�Xc�A�+*

loss��:�f�        �	)�n�Xc�A�+*

lossth�;��1       �	�bo�Xc�A�+*

loss8�	=v.h�       �	r�o�Xc�A�+*

loss!D_=1�       �	�p�Xc�A�+*

loss�w=�kv�       �	p'q�Xc�A�+*

loss$V<>Tޠ       �	��q�Xc�A�+*

loss���<�֮       �	�yr�Xc�A�+*

lossVUE;;@       �	�s�Xc�A�+*

lossF�*;���       �	��s�Xc�A�+*

lossIw
=�C�       �	�Wt�Xc�A�+*

loss���;*�       �	��t�Xc�A�+*

loss�S�<OOS       �	|�u�Xc�A�+*

loss4�;&��       �	`;v�Xc�A�+*

lossO� ;���       �	��v�Xc�A�+*

loss�S�:���       �	dyw�Xc�A�+*

loss���=G��       �	ax�Xc�A�+*

lossN��:8�{n       �	�x�Xc�A�+*

loss�:���X       �	_]y�Xc�A�+*

loss��;�[$       �	�y�Xc�A�+*

loss#��:Ԡz       �	��z�Xc�A�,*

loss��<4Q}       �	+0{�Xc�A�,*

loss=~�9�       �	�|�Xc�A�,*

lossi>;��(       �	k�|�Xc�A�,*

loss���;�N�       �	j3}�Xc�A�,*

loss�4Y;����       �	�Y~�Xc�A�,*

loss�|<�N9�       �	��~�Xc�A�,*

loss�j�;4���       �	���Xc�A�,*

loss��9�l��       �	�)��Xc�A�,*

loss��:�'�       �	`ˀ�Xc�A�,*

loss��<IPE       �	S\��Xc�A�,*

loss���:�I�<       �	��Xc�A�,*

loss�z�:OO��       �	����Xc�A�,*

loss,�(;���       �	a��Xc�A�,*

lossV*=L,�       �	M���Xc�A�,*

lossԂ�;xm       �	�R��Xc�A�,*

loss�N+>t�x       �	d焪Xc�A�,*

loss�fs=3�
�       �	;���Xc�A�,*

loss�o`;)���       �	�K��Xc�A�,*

loss�a:�Iɫ       �	Xc�A�,*

loss�q<&�ҵ       �	���Xc�A�,*

lossS��<��y�       �	N(��Xc�A�,*

loss?;�w��       �	����Xc�A�,*

lossD7 :�lէ       �	�Y��Xc�A�,*

lossۯ�:�X|6       �	��Xc�A�,*

loss&��<�� �       �	����Xc�A�,*

loss��:���       �	xE��Xc�A�,*

loss� E;v=C       �	�ۋ�Xc�A�,*

lossa��;�>��       �	�ǌ�Xc�A�,*

loss�[;�f�'       �	�h��Xc�A�,*

loss#M�=�Թ#       �	[��Xc�A�,*

loss���<ec��       �	���Xc�A�,*

loss�?K9���;       �	����Xc�A�,*

loss�L�:ŀi�       �	m7��Xc�A�,*

loss��;��2       �	ΐ�Xc�A�,*

loss���:�;��       �	�f��Xc�A�,*

loss�*<	<�       �	����Xc�A�,*

loss*��=���       �	)���Xc�A�,*

lossZ�1= |�5       �	/3��Xc�A�,*

lossr<�@\7       �	�Γ�Xc�A�,*

loss3/�<Z��n       �	{f��Xc�A�,*

loss�Z�=�       �	����Xc�A�,*

loss=�t=ð˶       �	����Xc�A�,*

losss�=N�\       �	/��Xc�A�,*

lossqt<��N�       �	,Ֆ�Xc�A�,*

loss y�;���       �	�k��Xc�A�,*

lossS-";���<       �	���Xc�A�,*

loss�8h;U�]       �	?���Xc�A�,*

loss3�7;y��       �	eS��Xc�A�,*

lossCh	9Х5�       �	[陪Xc�A�,*

loss�)3<�X��       �	����Xc�A�,*

loss�50<,�X|       �	]��Xc�A�,*

losskv�<�       �	����Xc�A�,*

loss�ٟ<���       �	5C��Xc�A�,*

loss�Ƅ<��9�       �	�ٜ�Xc�A�,*

loss�M�<*��D       �	Ho��Xc�A�,*

loss��n;�ş�       �	���Xc�A�,*

lossj��96���       �	����Xc�A�,*

loss�_�;�o       �	�2��Xc�A�,*

loss��:�+�]       �	*ɟ�Xc�A�,*

loss���<��E       �	�]��Xc�A�,*

lossLY;��R       �	���Xc�A�,*

loss'�:*W��       �	g���Xc�A�,*

loss"X�<���       �	<3��Xc�A�,*

loss�Q�;r��s       �	 Ȣ�Xc�A�,*

loss#5�;Z�m�       �	ߣ�Xc�A�,*

lossϣ<`�@�       �	�u��Xc�A�,*

loss���=/%u�       �	 	��Xc�A�,*

lossE(�<I���       �	+���Xc�A�,*

lossNN�;���       �	�<��Xc�A�,*

loss���;iј\       �	��Xc�A�,*

loss�[�;���g       �	����Xc�A�,*

loss�B<#���       �	K=��Xc�A�,*

loss3�+=��0       �	�飯Xc�A�,*

loss|U�=� .�       �	e���Xc�A�,*

losszI];����       �	�*��Xc�A�,*

loss"$;Ԓ�x       �	�Ȫ�Xc�A�,*

loss�L�<B�s�       �	c}��Xc�A�,*

loss_�n=�J       �	"��Xc�A�,*

lossq�;U�߮       �	,���Xc�A�,*

lossn6Z<��̲       �	]O��Xc�A�,*

lossA��;2��(       �	���Xc�A�,*

loss���:l���       �	ᳮ�Xc�A�,*

loss�)�;��l       �	*S��Xc�A�,*

loss㡖;�P��       �	(�Xc�A�,*

lossI>
<�'       �	D���Xc�A�,*

loss�7O<�q}g       �	K!��Xc�A�,*

loss���<Q'�       �	�뱪Xc�A�,*

lossrÃ<	��       �	*���Xc�A�,*

loss��;B�P       �	�'��Xc�A�,*

loss�sv=�;       �	����Xc�A�,*

loss��o;W^T�       �	.X��Xc�A�,*

loss_8�;�/�U       �	�Xc�A�,*

loss��I=j�       �	����Xc�A�,*

loss*��<J���       �	A)��Xc�A�,*

lossq�<���       �	½��Xc�A�,*

loss�O�;�P�V       �	:Z��Xc�A�,*

lossؓ�;?�+�       �	��Xc�A�,*

loss̶�<��~       �	����Xc�A�,*

loss�6=�`Yo       �	�z��Xc�A�,*

loss�B�;��G       �	���Xc�A�,*

lossr�i=����       �	���Xc�A�,*

loss��5;¼�i       �	<��Xc�A�,*

loss���= ��       �	�л�Xc�A�,*

loss�Ґ<�a��       �	���Xc�A�,*

loss��=0g�       �	Ɖ��Xc�A�,*

lossQ9�;z��h       �	���Xc�A�,*

loss��<Vl\�       �	L¾�Xc�A�,*

loss&F�;�ރ�       �	qY��Xc�A�,*

lossfK�9�;�7       �	�F��Xc�A�,*

loss�w@=m7�       �	���Xc�A�,*

loss�:�<���_       �	����Xc�A�,*

loss.�d:��.�       �	W=ªXc�A�,*

lossN�;��A6       �	
êXc�A�,*

lossM��<�t��       �	%�êXc�A�,*

lossz�o<�|-       �	�KĪXc�A�,*

loss�Z=n:D       �	�ĪXc�A�,*

loss��m;��R2       �	��ŪXc�A�,*

loss�<͝�0       �	�;ƪXc�A�,*

loss}�/:Y-�Z       �	t�ƪXc�A�,*

loss��;S+��       �	�kǪXc�A�,*

loss��V<ө       �	YȪXc�A�,*

lossnV<�Ñ5       �	�ȪXc�A�,*

loss
��:�z�       �	�NɪXc�A�,*

lossT�-<gµ�       �	)�ɪXc�A�,*

loss6a�;ͬ��       �	�ʪXc�A�,*

lossx�:cKU1       �	%˪Xc�A�,*

loss�k�:�gl       �	E�˪Xc�A�,*

loss��;��$�       �	k̪Xc�A�-*

loss	R<0Ǜ�       �	1ͪXc�A�-*

lossns�<��       �	�ͪXc�A�-*

loss(�<���+       �	KΪXc�A�-*

loss���<o"�       �	��ΪXc�A�-*

lossxt�=���       �	.�ϪXc�A�-*

loss[�8=>���       �	&UЪXc�A�-*

loss�C�<�mN       �	n�ЪXc�A�-*

loss&��:��Zw       �	�ѪXc�A�-*

lossp��;&S�|       �	+MҪXc�A�-*

loss��<3���       �	IӪXc�A�-*

lossq��;)���       �	1�ӪXc�A�-*

loss��[;���z       �	�zԪXc�A�-*

lossW��<�{�       �	�)ժXc�A�-*

loss��f=�m�       �	 �ժXc�A�-*

loss;d�;�)�       �	`r֪Xc�A�-*

loss)�Q<is�`       �	�aתXc�A�-*

loss��=-�3�       �	uتXc�A�-*

loss�+F=ԉ.       �	&�تXc�A�-*

lossa��<;��s       �	�C٪Xc�A�-*

loss��<k���       �	��٪Xc�A�-*

loss�^�<H��       �	+�ڪXc�A�-*

loss�̥=��U       �	�%۪Xc�A�-*

loss��=v��>       �	Z�۪Xc�A�-*

lossq,6;!)��       �	�ܪXc�A�-*

lossZ�9��       �	�7ݪXc�A�-*

lossw`=%�>�       �	ǛުXc�A�-*

loss��l<��?�       �	AߪXc�A�-*

lossV�m<.��c       �	*�ߪXc�A�-*

loss��;]�g}       �	9}�Xc�A�-*

loss��5:3��       �	1%�Xc�A�-*

loss[M<��       �	~��Xc�A�-*

loss]X�<Ă��       �	�p�Xc�A�-*

loss��A<�Z��       �	��Xc�A�-*

lossnn�<��=f       �	��Xc�A�-*

loss��=��/r       �	�I�Xc�A�-*

loss�h�<kD%v       �	A��Xc�A�-*

loss�m�9 ���       �	���Xc�A�-*

loss�|�;� �D       �	w.�Xc�A�-*

lossw�;�w(d       �	%��Xc�A�-*

loss!��:��-$       �	fi�Xc�A�-*

loss!��:��xd       �	 �Xc�A�-*

loss_�@:]rS�       �	ͮ�Xc�A�-*

loss8��<a��a       �	^K�Xc�A�-*

loss�<T�X�       �	t��Xc�A�-*

lossK?=:K�       �	i��Xc�A�-*

lossxp<�J       �	X7�Xc�A�-*

lossr�2<趧       �	)��Xc�A�-*

lossJ��;�Q@       �	?o�Xc�A�-*

lossᕄ:xպ       �	���Xc�A�-*

lossr�;�Z �       �	h���Xc�A�-*

lossD�=}���       �	�N�Xc�A�-*

loss�m�<����       �	N��Xc�A�-*

loss�y�;�+�@       �	���Xc�A�-*

loss���:�0%K       �	�5�Xc�A�-*

loss2�=�[��       �	1��Xc�A�-*

lossYw�;{4��       �	}x�Xc�A�-*

loss��S;�́x       �	T�Xc�A�-*

lossҺ�;�ED       �	ĳ�Xc�A�-*

lossH�]<�Q       �	�Y�Xc�A�-*

loss�4�;���       �	���Xc�A�-*

lossHWI<�Ŧ�       �	����Xc�A�-*

loss�7�<��V       �	�<��Xc�A�-*

loss**�:�P�       �	s���Xc�A�-*

loss�#=�=�       �	�x��Xc�A�-*

loss���=���       �	d��Xc�A�-*

loss�B<"�
=       �	���Xc�A�-*

lossq�;$��[       �	eS��Xc�A�-*

loss
��<sc�b       �	���Xc�A�-*

loss[{B<��B�       �	���Xc�A�-*

lossXz�9nZ9�       �	,��Xc�A�-*

loss�k�;���       �	����Xc�A�-*

loss<X�:����       �	ݖ��Xc�A�-*

loss�n�:�x       �	`:��Xc�A�-*

lossvq;�*)       �	����Xc�A�-*

loss��<�y�       �	c���Xc�A�-*

loss�C<WI��       �	�v��Xc�A�-*

loss�	�:�e��       �	�X��Xc�A�-*

lossKL<�fxv       �	���Xc�A�-*

loss�<;"ސ       �	�� �Xc�A�-*

loss�1q;���2       �	u��Xc�A�-*

loss^Z">ZE�r       �	�(�Xc�A�-*

loss(�=�0�Q       �	½�Xc�A�-*

loss�i<�AxP       �	�U�Xc�A�-*

loss��>=�h0N       �	-��Xc�A�-*

loss��^;0X8       �	�~�Xc�A�-*

lossd�:<?�P       �	��Xc�A�-*

loss�-�<��P�       �	a��Xc�A�-*

loss���<Cd�       �	�^�Xc�A�-*

loss��e<�I_b       �	���Xc�A�-*

loss�L�<$���       �	d��Xc�A�-*

loss�$=��+�       �	�*�Xc�A�-*

loss�ӭ:n��[       �	���Xc�A�-*

loss��;J1�       �	�	�Xc�A�-*

loss�s�:�(u�       �	�
�Xc�A�-*

loss�I�=�D�<       �	$�
�Xc�A�-*

loss��N<�M~q       �	�O�Xc�A�-*

loss���=Crk�       �	X��Xc�A�-*

loss5f;��s       �	��Xc�A�-*

loss�ˎ:_�,�       �	K!�Xc�A�-*

loss��g<̡\       �	I��Xc�A�-*

loss��x<w��       �	%[�Xc�A�-*

loss�X]:����       �	Q��Xc�A�-*

loss�:���o       �	Ք�Xc�A�-*

lossqs-=WR�)       �	 6�Xc�A�-*

lossA�
;v�n-       �	��Xc�A�-*

losso�<w���       �	B��Xc�A�-*

lossq�O:&�	       �	�_�Xc�A�-*

lossH<����       �	@��Xc�A�-*

loss���:৾�       �	
��Xc�A�-*

lossR?<����       �	Y�Xc�A�-*

lossn!=��Y�       �	!�Xc�A�-*

lossd��;���o       �	ס�Xc�A�-*

loss��<D�R�       �	JC�Xc�A�-*

loss!��<���       �	D��Xc�A�-*

loss��;.�       �	���Xc�A�-*

loss��=��5       �	;�Xc�A�-*

lossу�9+       �	���Xc�A�-*

loss�\�;�Qa       �	�O�Xc�A�-*

loss���; �4=       �	B��Xc�A�-*

loss2�_<�#��       �	���Xc�A�-*

lossTH�<�Ƞ�       �	��Xc�A�-*

loss<0       �	=��Xc�A�-*

loss�� ;W��/       �	eP�Xc�A�-*

loss2��<�J0�       �	���Xc�A�-*

loss���<j��       �	(��Xc�A�-*

loss� �:/߷W       �	]�Xc�A�-*

lossj//=x��       �	���Xc�A�-*

lossa2�<8L�       �	J�Xc�A�.*

loss�ȥ:��m�       �	:��Xc�A�.*

loss6]�<F�*�       �	k~ �Xc�A�.*

loss�;�;��Ua       �	S !�Xc�A�.*

loss��;^:˒       �	��!�Xc�A�.*

loss�S:�� '       �	CU"�Xc�A�.*

loss��g=�
�       �	l�"�Xc�A�.*

lossR�<��?       �	>�#�Xc�A�.*

lossŖ�<�G��       �	�0$�Xc�A�.*

loss�=3[oS       �	l�$�Xc�A�.*

loss��;�+�       �	�w%�Xc�A�.*

lossh.<Z�1%       �	�&�Xc�A�.*

loss�=Qv��       �	b�&�Xc�A�.*

loss�K�<.P�       �	�['�Xc�A�.*

losss'�<QCA�       �	��'�Xc�A�.*

loss��&;q�8�       �	*�(�Xc�A�.*

loss�A:�I�W       �	FE)�Xc�A�.*

loss}�<g$�o       �	��)�Xc�A�.*

loss:Gv=?i�2       �	��*�Xc�A�.*

loss��I=r        �	V~+�Xc�A�.*

loss-�:AI+2       �	@K,�Xc�A�.*

loss���:m>��       �	!�,�Xc�A�.*

loss�9�;d�xS       �	��-�Xc�A�.*

loss��	:�Oq       �	�K.�Xc�A�.*

loss���;	x\       �	(|/�Xc�A�.*

loss��J<A\�!       �	20�Xc�A�.*

loss�"><\#N       �	��0�Xc�A�.*

loss\;�       �	�W1�Xc�A�.*

loss��;�^^       �	FC2�Xc�A�.*

loss<c�=���       �	��2�Xc�A�.*

loss�<��h�       �	^i3�Xc�A�.*

loss��;%c�       �	7 4�Xc�A�.*

loss���;_R]       �	h�4�Xc�A�.*

loss�$ 9	Z�       �	}x5�Xc�A�.*

loss��3< ��       �	�K6�Xc�A�.*

loss�#�;��5k       �	��6�Xc�A�.*

lossV�K=�$��       �	ty7�Xc�A�.*

loss��<�U        �	�8�Xc�A�.*

loss{��;�IF       �	ū8�Xc�A�.*

loss��?<��[�       �	(H9�Xc�A�.*

loss�C�<��!n       �	H�9�Xc�A�.*

loss@(�<n�;       �	�w:�Xc�A�.*

lossd�:d���       �	�;�Xc�A�.*

loss���9�E��       �	ߦ;�Xc�A�.*

loss��;y�^       �	�b<�Xc�A�.*

loss�;:�Z8A       �	��<�Xc�A�.*

loss��t8���
       �	y�=�Xc�A�.*

lossr�;��       �	�?�Xc�A�.*

loss H:�Z?�       �	��?�Xc�A�.*

loss�C�<LO��       �	��@�Xc�A�.*

loss���<"���       �	K\A�Xc�A�.*

loss�/�;���       �	'�A�Xc�A�.*

loss��<��ԓ       �	��B�Xc�A�.*

loss��k9t��#       �	-{C�Xc�A�.*

loss�*�9��~�       �	�$D�Xc�A�.*

loss�[�9C^��       �	/�D�Xc�A�.*

lossq�<9�	�"       �	��E�Xc�A�.*

loss���;��[w       �	 'F�Xc�A�.*

lossӗ0<_�δ       �	��F�Xc�A�.*

loss��P7
[(`       �	hG�Xc�A�.*

loss�x�<�9�       �	�H�Xc�A�.*

lossu�<G4l�       �	ۢH�Xc�A�.*

lossJU:��       �		QI�Xc�A�.*

loss�r>VY2@       �	�I�Xc�A�.*

loss�x#=m�Q�       �	+K�Xc�A�.*

loss1�\;��<�       �	��K�Xc�A�.*

loss�b�:�f#�       �	�dL�Xc�A�.*

lossa�;�t�       �	��L�Xc�A�.*

loss���<w��       �	��M�Xc�A�.*

loss]��:�**B       �	/2N�Xc�A�.*

loss�9�<hG       �	c�N�Xc�A�.*

loss���;�Z�       �	XuO�Xc�A�.*

lossMź<�]<x       �	]P�Xc�A�.*

loss
�=?ꜧ       �	�P�Xc�A�.*

loss7D<2t�*       �	�LQ�Xc�A�.*

lossvy�<tmL       �	\�Q�Xc�A�.*

loss�Nq=4:��       �	�yR�Xc�A�.*

loss<�;���       �	
S�Xc�A�.*

loss�lz:��(       �	�S�Xc�A�.*

lossD�;�S�u       �	LT�Xc�A�.*

loss��<D��       �	i�T�Xc�A�.*

loss��*;AZk�       �	xzU�Xc�A�.*

losse�:pa&�       �	'V�Xc�A�.*

lossVw1:U�<�       �	��V�Xc�A�.*

loss���:��       �	�NW�Xc�A�.*

loss�9(*
�       �	�W�Xc�A�.*

lossS6;�D�*       �	q�X�Xc�A�.*

lossV�<G*	�       �	82Y�Xc�A�.*

loss���<���       �	��Y�Xc�A�.*

loss *�<�!3�       �	
jZ�Xc�A�.*

loss {�:��B�       �	$*[�Xc�A�.*

loss�n|=���       �	�[�Xc�A�.*

loss��;m5��       �	�n\�Xc�A�.*

lossz��<?�6I       �	�]�Xc�A�.*

loss%�<�
�Y       �	��]�Xc�A�.*

loss�!;��17       �	�U^�Xc�A�.*

loss7R;jc�c       �	��^�Xc�A�.*

lossJ��:X�K�       �	�_�Xc�A�.*

loss\S�;�⇖       �	 `�Xc�A�.*

loss�.�<��ަ       �	�`�Xc�A�.*

loss��g=	���       �	]a�Xc�A�.*

lossT =Ŵ9H       �	�b�Xc�A�.*

lossQq9�KB�       �	D�b�Xc�A�.*

loss�"T<<V�       �	�;c�Xc�A�.*

loss���;�ՙ,       �	�c�Xc�A�.*

loss�Z�<����       �	ld�Xc�A�.*

lossUQ=�=��       �	�e�Xc�A�.*

loss!!)=y,�       �	j�e�Xc�A�.*

loss&6<=9���       �	SAf�Xc�A�.*

loss|b�:���       �	�f�Xc�A�.*

loss�Z�<К.�       �	��g�Xc�A�.*

loss���;���       �	�|h�Xc�A�.*

loss�\6<ۿ?	       �	�,i�Xc�A�.*

loss�w�;m�5t       �	E���Xc�A�.*

loss���;p*�       �	���Xc�A�.*

loss��
=#��W       �	rŃ�Xc�A�.*

loss�;};���       �	�a��Xc�A�.*

loss�B<.       �	����Xc�A�.*

loss��<�Np       �	E���Xc�A�.*

lossA*�<ÜW�       �	�@��Xc�A�.*

lossm(�;"�R       �	w؆�Xc�A�.*

lossd�<��F�       �	�n��Xc�A�.*

loss���:t�g       �	��Xc�A�.*

loss��9��8       �	����Xc�A�.*

loss�h=Wz~       �	�L��Xc�A�.*

loss�x�<T�-d       �	��Xc�A�.*

loss
�<G��/       �	����Xc�A�.*

loss���:��}�       �	0)��Xc�A�.*

lossɪ�<�%�       �	v���Xc�A�/*

loss�K�;{62       �	�V��Xc�A�/*

loss�Qf<�2CV       �	���Xc�A�/*

loss��:c��       �	;���Xc�A�/*

lossN*;��&       �	�W��Xc�A�/*

loss
�>;�]�|       �	��Xc�A�/*

lossxkw<�k�       �	Ō��Xc�A�/*

loss���9���&       �	F"��Xc�A�/*

loss�}=Q��2       �	����Xc�A�/*

loss&�<�P?       �	�J��Xc�A�/*

losslL=��L�       �	�⑫Xc�A�/*

lossbY<v�I       �	���Xc�A�/*

loss,P;Z��       �	�I��Xc�A�/*

loss��;j�       �	����Xc�A�/*

loss&΍;�}�       �	�Ɣ�Xc�A�/*

lossdi<�,�|       �	����Xc�A�/*

lossי�9�H��       �	�5��Xc�A�/*

loss.6f;��       �	4Ֆ�Xc�A�/*

loss]�;�q��       �	�o��Xc�A�/*

loss�׉<G       �	���Xc�A�/*

loss��;���       �	����Xc�A�/*

loss�":_��       �	�F��Xc�A�/*

loss���<��;       �	0ؙ�Xc�A�/*

loss$v�<�n       �	�m��Xc�A�/*

lossF>�;0���       �	u��Xc�A�/*

loss�l�;���       �	�Û�Xc�A�/*

loss�T!</]�       �	 W��Xc�A�/*

lossO��;+d/Z       �	�$��Xc�A�/*

loss��G;P��       �	ܹ��Xc�A�/*

lossϐ"=� ��       �	Z��Xc�A�/*

loss���;�ܚ       �	�)��Xc�A�/*

loss߯�:2�Q�       �	����Xc�A�/*

loss��;7��z       �	�X��Xc�A�/*

lossyh�:�,�}       �	3���Xc�A�/*

loss8�t9ښx^       �	����Xc�A�/*

loss }c<`*       �	73��Xc�A�/*

lossi�;=��       �	�Ȣ�Xc�A�/*

loss��:,.��       �	�r��Xc�A�/*

loss` �<&L@�       �	6��Xc�A�/*

loss�p<F�k�       �	ퟤ�Xc�A�/*

loss��e:\��       �	�J��Xc�A�/*

loss)�:iu��       �	�ꥫXc�A�/*

lossfo|9�y��       �	z���Xc�A�/*

loss=�b;��`�       �	�.��Xc�A�/*

loss�B�<����       �	K��Xc�A�/*

loss��<��_�       �	ࠨ�Xc�A�/*

loss6;���       �	N��Xc�A�/*

loss�|9#4M�       �	����Xc�A�/*

loss,==�09       �	����Xc�A�/*

lossl��;(|�       �	W?��Xc�A�/*

loss�|�;��h       �	�ӫ�Xc�A�/*

loss�;D��       �	N��Xc�A�/*

loss7c�:CQ��       �	.��Xc�A�/*

loss6;�x��       �	����Xc�A�/*

loss[}�;J:��       �	�R��Xc�A�/*

loss !n=/ҏt       �	���Xc�A�/*

loss�6�<�>)�       �	b���Xc�A�/*

loss?�:(�YS       �	E��Xc�A�/*

loss��;�>��       �	����Xc�A�/*

loss!�;��i�       �	����Xc�A�/*

loss4��;��ە       �	�6��Xc�A�/*

loss��K<#�@?       �	�в�Xc�A�/*

lossA��9�L {       �	Ǟ��Xc�A�/*

loss
�;�~+       �	E��Xc�A�/*

lossr-h<�9`�       �	�봫Xc�A�/*

lossi�;ͨ�*       �	����Xc�A�/*

loss��<��-�       �	z9��Xc�A�/*

loss��;���       �	�۶�Xc�A�/*

lossb�<���       �	c~��Xc�A�/*

loss�#$=Gi       �	:��Xc�A�/*

lossi��<���       �	񻸫Xc�A�/*

loss8�g9�1�<       �	�R��Xc�A�/*

loss��=��y�       �	﹫Xc�A�/*

losszz<��       �	����Xc�A�/*

loss��;��?       �	d���Xc�A�/*

lossS�O;��       �	*T��Xc�A�/*

loss��;[@B       �	�輫Xc�A�/*

loss��;k4�       �	����Xc�A�/*

loss��f<x� �       �	t)��Xc�A�/*

loss�Б=ed��       �	zȾ�Xc�A�/*

loss��N:����       �	mo��Xc�A�/*

lossǦ�<����       �	���Xc�A�/*

loss���<ؠ�7       �	����Xc�A�/*

loss��=�
4�       �	PP��Xc�A�/*

loss��;h�[       �	����Xc�A�/*

loss��h9��%�       �	o�«Xc�A�/*

loss.��8���}       �	�ëXc�A�/*

loss���<�~�       �	��ëXc�A�/*

loss�{0<I,�       �	�[īXc�A�/*

loss[��;R�bL       �	�īXc�A�/*

loss��;��&m       �	u�ūXc�A�/*

lossE��:����       �	R)ƫXc�A�/*

loss|��;]�O�       �	��ƫXc�A�/*

loss���=��u�       �	<gǫXc�A�/*

loss��g:��kM       �	��ǫXc�A�/*

loss/��=���       �	W�ȫXc�A�/*

loss��:J�R�       �	�)ɫXc�A�/*

loss���<4�       �	M�ɫXc�A�/*

lossQ�*;�qi�       �	�SʫXc�A�/*

lossԪ�<d�       �	F�ʫXc�A�/*

loss���:Y���       �	=�˫Xc�A�/*

lossh)<Ϸu&       �	̫Xc�A�/*

loss�<1=���       �	 �̫Xc�A�/*

loss�A�=� BC       �	hͫXc�A�/*

loss�S:	:�0       �	�ΫXc�A�/*

lossΤb;'6��       �	��ΫXc�A�/*

lossa��<~cJ�       �	�DϫXc�A�/*

loss��;p��       �	��ϫXc�A�/*

loss�_%<4��       �	��ЫXc�A�/*

loss�М;�a��       �	-!ѫXc�A�/*

lossn�<e�       �	��ѫXc�A�/*

loss	��:���       �	~�ҫXc�A�/*

loss#l	=��/J       �	#IӫXc�A�/*

loss�'<���       �	�2ԫXc�A�/*

lossO�;V��>       �	�fիXc�A�/*

loss�s�;��pf       �	N{֫Xc�A�/*

loss�Ǔ<l���       �	�/׫Xc�A�/*

loss8�=uq��       �	��׫Xc�A�/*

loss�:�<�39       �	�ثXc�A�/*

loss��<�a��       �	�?٫Xc�A�/*

loss��:��r       �	��٫Xc�A�/*

loss�Y<�"p=       �		nګXc�A�/*

loss��c:��̈       �	�۫Xc�A�/*

loss< )<㹖�       �	t�۫Xc�A�/*

loss���:t�
       �	DNܫXc�A�/*

loss��F;-��f       �	�ݫXc�A�/*

loss�G<	�       �	E�ݫXc�A�/*

loss�Q/;�1K       �	�TޫXc�A�/*

lossOV;9=�       �	�ޫXc�A�0*

loss<r�;�=�}       �	��߫Xc�A�0*

loss��:%��       �	�S�Xc�A�0*

loss�t;�m��       �	��Xc�A�0*

loss�;'� �       �	o��Xc�A�0*

loss`K�:�׿M       �	�T�Xc�A�0*

loss��;�0�       �	���Xc�A�0*

lossH��:�z`       �	Q��Xc�A�0*

loss�ؖ<hTn�       �	�G�Xc�A�0*

loss�<��G       �	��Xc�A�0*

loss� �<�X
       �	4��Xc�A�0*

loss1}=��{�       �	q�Xc�A�0*

loss��h:a=f       �	��Xc�A�0*

loss���:�D�       �	�E�Xc�A�0*

loss�>2<�3�       �	���Xc�A�0*

lossl#=��N#       �	�s�Xc�A�0*

loss�c�;]���       �	��Xc�A�0*

lossh��<�9c�       �	��Xc�A�0*

loss�ڬ:�{�"       �	�b�Xc�A�0*

loss)CP=��{4       �	��Xc�A�0*

loss�5�<����       �	B��Xc�A�0*

loss�8;���       �	OY�Xc�A�0*

loss��;|?��       �	�
��Xc�A�0*

lossmL+<��       �	���Xc�A�0*

loss}�n=,��       �	CY�Xc�A�0*

loss�<���.       �	j��Xc�A�0*

loss�zw;��g       �	=��Xc�A�0*

lossSl�;�-2&       �	�<�Xc�A�0*

loss ��9��       �	��Xc�A�0*

loss��,98�^s       �	O��Xc�A�0*

loss�:�<m�Q�       �	�Xc�A�0*

lossN�;���d       �	��Xc�A�0*

lossIAv:`S��       �	��Xc�A�0*

lossQh=9�d       �	9G��Xc�A�0*

lossͼ�<�-�S       �	����Xc�A�0*

loss�<����       �	w��Xc�A�0*

loss���:��W�       �	�J��Xc�A�0*

lossO:\\��       �	����Xc�A�0*

loss�#$=�1�D       �	\���Xc�A�0*

loss�m�;͖Ls       �	�2��Xc�A�0*

loss�YX:F�       �	����Xc�A�0*

lossiO�:?r��       �	�h��Xc�A�0*

loss�f+:�V �       �	{��Xc�A�0*

loss�|�<����       �	,���Xc�A�0*

loss�39u���       �	�a��Xc�A�0*

loss��&<w�ɭ       �	���Xc�A�0*

loss�E:�g�m       �	���Xc�A�0*

loss�';ͭ�       �	���Xc�A�0*

loss3��<g���       �	=+��Xc�A�0*

lossf�;!�}h       �	����Xc�A�0*

lossSӽ;�`$0       �	^h��Xc�A�0*

loss�9��3       �	� �Xc�A�0*

loss�G:�z��       �	ŭ �Xc�A�0*

loss���;a�B       �	&W�Xc�A�0*

loss���:�O�$       �	���Xc�A�0*

lossO�;ӣ        �	Й�Xc�A�0*

lossaE :;��8       �	�9�Xc�A�0*

loss���;фܡ       �	���Xc�A�0*

loss]Fw;�ȃ�       �		��Xc�A�0*

loss�mN<oY       �	���Xc�A�0*

loss�T�:":�       �	�=�Xc�A�0*

loss�̲;�R��       �	���Xc�A�0*

loss7�:��[       �	���Xc�A�0*

loss�=��OR       �	5_�Xc�A�0*

loss./(;���       �	q	�Xc�A�0*

loss�*-;���}       �	]�	�Xc�A�0*

loss�/=���O       �	�R
�Xc�A�0*

loss1��<%�p�       �	��
�Xc�A�0*

lossC �<�M��       �	���Xc�A�0*

losssAQ=�s       �	�>�Xc�A�0*

lossr��;���       �	:��Xc�A�0*

loss�0�;ܧ^�       �	��Xc�A�0*

loss ]<Bu       �	�:�Xc�A�0*

loss�]=yG�       �	U��Xc�A�0*

loss���<�?��       �	���Xc�A�0*

loss�� =L'vD       �	�,�Xc�A�0*

loss62�<ùI       �	0��Xc�A�0*

loss�/�<iZm�       �	%v�Xc�A�0*

loss��<;;5       �	��Xc�A�0*

loss��;b��       �	�_�Xc�A�0*

loss� �:U�x       �	�C�Xc�A�0*

lossp<���W       �	&�Xc�A�0*

loss��_<%�gf       �	���Xc�A�0*

loss�2�=�x       �	Ve�Xc�A�0*

loss�!=���       �	m�Xc�A�0*

loss <� �)       �	���Xc�A�0*

loss3�:=��^       �	?o�Xc�A�0*

loss�&<�dr       �	�(�Xc�A�0*

loss�:p��x       �	 ��Xc�A�0*

lossM�>*�       �	�r�Xc�A�0*

losswb�;���;       �	��Xc�A�0*

loss;�S<v�       �	���Xc�A�0*

lossQU�:��       �	$b�Xc�A�0*

loss&��;�2��       �	:�Xc�A�0*

loss��I:����       �	^��Xc�A�0*

loss��<k�       �	�; �Xc�A�0*

loss�֖9a�&w       �	Y� �Xc�A�0*

loss��#=u �r       �	.t!�Xc�A�0*

lossL��<i-H�       �	�r"�Xc�A�0*

loss� $=P��j       �	�#�Xc�A�0*

lossåe<���T       �	��#�Xc�A�0*

loss��}=��Ax       �	�M$�Xc�A�0*

loss6H<?���       �	:�$�Xc�A�0*

loss���;��.       �	Z�%�Xc�A�0*

loss㣺8u�       �	vT&�Xc�A�0*

lossO/�:�@�N       �	��&�Xc�A�0*

loss,�S;����       �	�'�Xc�A�0*

loss-%�:�j�       �	�K(�Xc�A�0*

loss�P:�'��       �	��(�Xc�A�0*

loss��h;����       �	�z)�Xc�A�0*

loss沝;:���       �	<*�Xc�A�0*

loss$!K;| �{       �	�*�Xc�A�0*

loss'��:)�       �	�y+�Xc�A�0*

lossD<Y
Bd       �	s,�Xc�A�0*

loss�<�՘       �	t�,�Xc�A�0*

lossڋ=��J�       �	�Z-�Xc�A�0*

loss�Y!=7l�       �	 �-�Xc�A�0*

loss�B;PL!E       �	�.�Xc�A�0*

loss
�l;2���       �	UK/�Xc�A�0*

lossj�{=�Hcx       �	�l0�Xc�A�0*

loss�o|;!���       �	�1�Xc�A�0*

loss*��<��       �	/�1�Xc�A�0*

loss1P;
LS       �	�L2�Xc�A�0*

lossZ2<l��       �	�2�Xc�A�0*

loss�v:�
��       �	�3�Xc�A�0*

loss��=�UD�       �	a64�Xc�A�0*

lossWE�=Pk�       �	��4�Xc�A�0*

loss=_�:~��       �	�5�Xc�A�0*

loss1��<:�ָ       �	F_6�Xc�A�1*

loss��T<��       �	3�6�Xc�A�1*

loss?/;p|3       �	��7�Xc�A�1*

loss&]�=�6��       �	'�8�Xc�A�1*

loss��<`       �	�9�Xc�A�1*

lossp�=��X       �	k�9�Xc�A�1*

lossZ�)9͏�       �	�;�Xc�A�1*

loss�f4<��'�       �	ٵ;�Xc�A�1*

loss=���       �	;R<�Xc�A�1*

lossTd�8�G��       �	�5=�Xc�A�1*

lossC��:�էb       �	�=�Xc�A�1*

loss�a;e`)z       �	3�>�Xc�A�1*

lossơ;ksz       �	-`?�Xc�A�1*

loss	.4<���       �	@�Xc�A�1*

losszr�;��D       �	��@�Xc�A�1*

loss=�=��       �	ׄA�Xc�A�1*

loss��='�f$       �	�B�Xc�A�1*

loss���:���1       �	��B�Xc�A�1*

lossצ9��.�       �	CsC�Xc�A�1*

lossƻ�:�.B       �	1D�Xc�A�1*

loss}��<���G       �	U�D�Xc�A�1*

loss���94Tg�       �	�CE�Xc�A�1*

loss��9���H       �	�E�Xc�A�1*

lossu�;�R�       �	,}F�Xc�A�1*

loss,&�<�;L�       �	�G�Xc�A�1*

losss�~9�P��       �	l�G�Xc�A�1*

loss��=Q4̻       �	�H�Xc�A�1*

loss�Q�<�~��       �	2!I�Xc�A�1*

loss,T =o��/       �	�I�Xc�A�1*

loss�X�<63��       �	�NJ�Xc�A�1*

loss2<0<����       �	�J�Xc�A�1*

lossx%;��/�       �	,�K�Xc�A�1*

loss�8l<�ϕ�       �	�L�Xc�A�1*

lossH�8<�E��       �	��L�Xc�A�1*

lossD%<
D\�       �	l[M�Xc�A�1*

loss��:���       �	��M�Xc�A�1*

lossx$�=!v�       �	�N�Xc�A�1*

loss��<���       �	�;O�Xc�A�1*

loss���;@`�       �	��O�Xc�A�1*

loss}�V;��F       �	pP�Xc�A�1*

loss/�k=;��.       �	�Q�Xc�A�1*

loss�<G�U�       �	ݳQ�Xc�A�1*

loss}��:cY�K       �	.VR�Xc�A�1*

loss:��<	�<       �	$�R�Xc�A�1*

lossr=!��       �	˝S�Xc�A�1*

lossd��:���Z       �	&�T�Xc�A�1*

loss���:Z��       �	�U�Xc�A�1*

loss��;wgG�       �	~V�Xc�A�1*

loss�%�;-�       �	!�W�Xc�A�1*

loss=e�:��       �	�fX�Xc�A�1*

lossI��<���&       �	[_Y�Xc�A�1*

lossv~�<�o�W       �	��Y�Xc�A�1*

loss��!:v��       �	ЗZ�Xc�A�1*

loss]$|;Ѝ��       �	�E[�Xc�A�1*

loss��=�s�       �	l�[�Xc�A�1*

losslH:;� �M       �	��\�Xc�A�1*

loss���<��3�       �	
�]�Xc�A�1*

loss��<�W_�       �	�R^�Xc�A�1*

loss�}�<�PY       �	��^�Xc�A�1*

loss�D�:C��>       �	�_�Xc�A�1*

loss�X�<��ȣ       �	�6`�Xc�A�1*

loss���<���_       �	��`�Xc�A�1*

losspk9�M��       �	%wa�Xc�A�1*

lossZׯ=2 �       �	b�Xc�A�1*

loss�.�<�J�       �	x�b�Xc�A�1*

loss�%�<����       �	҉c�Xc�A�1*

loss��1<M���       �	�-d�Xc�A�1*

loss�[=zs�E       �	��d�Xc�A�1*

loss�lD:�ts�       �	�he�Xc�A�1*

loss;&d�       �	�f�Xc�A�1*

loss��;ꢅ       �	��f�Xc�A�1*

loss�V�;N��E       �	Ig�Xc�A�1*

loss	��;��T�       �	7�g�Xc�A�1*

loss�K�=�l�       �	M�h�Xc�A�1*

lossd�<ҹq�       �	E)i�Xc�A�1*

loss�Ø:����       �	��i�Xc�A�1*

loss.�X<��?       �	:uj�Xc�A�1*

loss�|<D��       �	�k�Xc�A�1*

lossz��<Wи�       �	��k�Xc�A�1*

loss�|:�{г       �	<Jl�Xc�A�1*

loss��:�k�       �	��l�Xc�A�1*

lossD>�<mF��       �	=�m�Xc�A�1*

loss1y�<oZ�       �	K9n�Xc�A�1*

loss�k<�Km       �	��n�Xc�A�1*

loss�;��^�       �	�po�Xc�A�1*

lossD;��M       �	�p�Xc�A�1*

loss�7+<���       �	Q�p�Xc�A�1*

lossƸ�:[���       �	t@q�Xc�A�1*

loss���<���       �	��q�Xc�A�1*

loss��= �X       �	�zr�Xc�A�1*

loss��\:q0��       �	s�Xc�A�1*

lossA+�<Y��2       �	��s�Xc�A�1*

lossLL=��P�       �	��t�Xc�A�1*

lossXx=��3       �	}$u�Xc�A�1*

loss���<�v��       �	�Ov�Xc�A�1*

loss'<:u�m       �	�v�Xc�A�1*

loss�x�<���!       �	�w�Xc�A�1*

lossQ*l;��J       �	�x�Xc�A�1*

lossn��:#M؜       �	��x�Xc�A�1*

lossc�;�(��       �	Qy�Xc�A�1*

loss�G�;hM0       �	d�y�Xc�A�1*

loss���<�*I�       �	��z�Xc�A�1*

loss�0�<Ҿ��       �	2>{�Xc�A�1*

loss)��<z)       �	(�{�Xc�A�1*

loss��
;��L       �	�s|�Xc�A�1*

loss�'�;�E�$       �	�}�Xc�A�1*

loss���;Б{�       �	~�Xc�A�1*

loss�XH<��p�       �	<�~�Xc�A�1*

loss�4#=�c       �	�E�Xc�A�1*

loss� <��Y�       �	��Xc�A�1*

loss�Z<N��       �	G���Xc�A�1*

lossԳ�=��k�       �	�*��Xc�A�1*

lossM�:����       �	Mׁ�Xc�A�1*

lossTFD<B�6d       �	�w��Xc�A�1*

loss�=��^�       �	%!��Xc�A�1*

losszI=r6�       �	�܄�Xc�A�1*

loss
�Y;�b�       �	�~��Xc�A�1*

losse(�<g��       �	���Xc�A�1*

losst�;���       �	=
��Xc�A�1*

lossb�;u:K�       �	����Xc�A�1*

loss�c�;n�       �	�|��Xc�A�1*

loss�<m ��       �	���Xc�A�1*

lossrV�;�ة�       �	ެ��Xc�A�1*

loss�o�<�l>�       �	�@��Xc�A�1*

loss���<��L       �	@ۋ�Xc�A�1*

loss_)�8 ��F       �	�|��Xc�A�1*

loss;N<f���       �	P��Xc�A�1*

loss��:���       �	Ѳ��Xc�A�1*

lossl]�:�g0�       �	�n��Xc�A�2*

loss��<JZQ       �	='��Xc�A�2*

loss��9OQ�       �	�ґ�Xc�A�2*

loss�,�;���       �	����Xc�A�2*

lossx�:YB��       �	�?��Xc�A�2*

loss�j�<H�O�       �	����Xc�A�2*

loss�d�;dqS       �	w���Xc�A�2*

loss�J�;j��       �	YR��Xc�A�2*

loss:U="��l       �	���Xc�A�2*

lossv�=�
�       �	��Xc�A�2*

loss=b<��1�       �	G˗�Xc�A�2*

loss_�<��Kf       �	vq��Xc�A�2*

loss�M�<�p��       �	�=��Xc�A�2*

lossm��=uO�       �	 ��Xc�A�2*

loss��x<JDo�       �	�	��Xc�A�2*

loss�R'=�<r       �	G���Xc�A�2*

loss�$#=�fO�       �	����Xc�A�2*

losssɘ;7Ps6       �	���Xc�A�2*

lossܴ�;�6{       �	�:��Xc�A�2*

lossPJ=�(am       �	�۞�Xc�A�2*

lossq�:i��       �	�~��Xc�A�2*

loss��=$��c       �	�)��Xc�A�2*

loss�c=@-;�       �	^���Xc�A�2*

loss_\/;��       �	]k��Xc�A�2*

lossxQz<e'0�       �	8,��Xc�A�2*

loss�0=Z�K       �	UN��Xc�A�2*

loss/��=��l       �	�⣬Xc�A�2*

loss�T/<c~C�       �	ߦ��Xc�A�2*

lossn&N=p�[       �	�9��Xc�A�2*

loss��:LQ��       �	pΥ�Xc�A�2*

loss_��8T��~       �	&p��Xc�A�2*

loss�\#=���       �	���Xc�A�2*

lossʖ<�+Jr       �	U���Xc�A�2*

loss��:wl[       �	����Xc�A�2*

lossf5�;}��       �	*��Xc�A�2*

lossa$L<���S       �	���Xc�A�2*

lossf�#;p�       �	�R��Xc�A�2*

loss��:5Td�       �	����Xc�A�2*

lossa%z=�!t�       �	K���Xc�A�2*

losswB�;��G       �	�+��Xc�A�2*

loss�lS;���       �	~Ŭ�Xc�A�2*

loss8~�<�R�^       �	Y��Xc�A�2*

loss�<c���       �	�S��Xc�A�2*

loss=�<�	�       �	箬Xc�A�2*

lossA>�[�       �	����Xc�A�2*

loss2a<�L>       �	"��Xc�A�2*

loss�ǝ<7�6_       �	I���Xc�A�2*

lossh�x=� �       �	�U��Xc�A�2*

lossA�y;ڠ��       �	�걬Xc�A�2*

loss[	�<C�Dr       �	o���Xc�A�2*

loss6�;���       �	�7��Xc�A�2*

loss���=K8       �	Ѵ�Xc�A�2*

lossF�<B��       �	w��Xc�A�2*

loss�T�;���       �	V��Xc�A�2*

loss��<����       �	M�Xc�A�2*

loss��:�M�%       �	����Xc�A�2*

loss$f�=�?�^       �	�}��Xc�A�2*

lossSɒ=C��       �	�T��Xc�A�2*

lossX��<D*�       �	5�Xc�A�2*

loss��:j��       �	����Xc�A�2*

loss,C�<A�Wv       �	�$��Xc�A�2*

loss|��;�<��       �	����Xc�A�2*

loss��&:�f�       �	�W��Xc�A�2*

loss���:��7~       �	��Xc�A�2*

loss���<~�v�       �	뎽�Xc�A�2*

loss��Z;���       �	�;��Xc�A�2*

losse�^; ���       �	پ�Xc�A�2*

lossR��9�y|�       �	�z��Xc�A�2*

loss���;"a�       �	���Xc�A�2*

lossp=2O�       �	����Xc�A�2*

lossӛ:�P�       �	N���Xc�A�2*

loss|�G=�F?       �	�x¬Xc�A�2*

loss��V<�}#       �	�^ìXc�A�2*

loss��i<J�z�       �	KĬXc�A�2*

loss�);�R�       �	Y�ĬXc�A�2*

loss�3t<�q�       �	�?ŬXc�A�2*

loss�l�:E6c.       �	S"ƬXc�A�2*

loss�1!;�	��       �	H�ƬXc�A�2*

loss�>�;�ҭ       �	JbǬXc�A�2*

loss3��;8c=-       �	qȬXc�A�2*

lossو<�%<�       �	 �ȬXc�A�2*

loss��u<�Ү�       �	�8ɬXc�A�2*

loss�<<ń�u       �	l�ɬXc�A�2*

loss�`t<�	�       �	oʬXc�A�2*

loss�"�:\#�A       �	�ˬXc�A�2*

loss���<�J��       �	��ˬXc�A�2*

loss�>;�ׯ       �	��̬Xc�A�2*

loss�w':�w�       �	[&ͬXc�A�2*

loss��<뢊       �	��ͬXc�A�2*

lossD�N;��t�       �	��άXc�A�2*

loss��=щ��       �	a�ϬXc�A�2*

loss&��<rŠ=       �	�eЬXc�A�2*

lossj�U:ڔ��       �	kѬXc�A�2*

loss	�j9�K��       �	��ѬXc�A�2*

loss�J<�W�       �	JҬXc�A�2*

loss���9 L�       �	
�ҬXc�A�2*

loss�2�:Z��       �	S�ӬXc�A�2*

lossf��;��o3       �	~RԬXc�A�2*

loss�(�;�5M       �	!�ԬXc�A�2*

loss�M�;0pa�       �	��լXc�A�2*

loss=�|<ܿ�W       �	1(֬Xc�A�2*

loss��;z�/�       �	��֬Xc�A�2*

loss�U;�G��       �	=b׬Xc�A�2*

loss|�:<�       �	A)جXc�A�2*

loss�?i;��K�       �	��جXc�A�2*

lossͫv;�A�Y       �	�j٬Xc�A�2*

loss��:OC9�       �	�ڬXc�A�2*

loss��C<����       �	�<۬Xc�A�2*

lossVc:8~?�       �	��۬Xc�A�2*

lossG�<e��6       �	(�ܬXc�A�2*

loss��;յ�       �	(ݬXc�A�2*

lossW��9�0��       �	��ݬXc�A�2*

loss*' <S*m"       �	EeެXc�A�2*

loss�s�:����       �	)߬Xc�A�2*

loss�d<I���       �	Q�߬Xc�A�2*

loss=�f<��       �	S<�Xc�A�2*

lossOђ<�{       �	4��Xc�A�2*

lossc��<��'�       �	Hj�Xc�A�2*

loss���:2��%       �	��Xc�A�2*

loss#�;:V�2       �	���Xc�A�2*

losst;Z~�       �	>�Xc�A�2*

loss�V;��̊       �	���Xc�A�2*

loss6��:$`p5       �	��Xc�A�2*

lossI69�ͫ       �	�L�Xc�A�2*

lossv!r:����       �	��Xc�A�2*

loss�;�;!�{)       �		��Xc�A�2*

loss6�4;N%�       �	�%�Xc�A�2*

loss��;��       �	��Xc�A�2*

loss��'<0zBD       �	Pr�Xc�A�3*

loss��Z<V..�       �	�Xc�A�3*

loss͸<4���       �	r��Xc�A�3*

loss�<X�Vp       �	���Xc�A�3*

lossԲ;�[�       �	�&�Xc�A�3*

loss<i�9c�       �	���Xc�A�3*

lossQ��9��@       �	�a�Xc�A�3*

loss�K�:R��       �	���Xc�A�3*

loss�ֆ:4r]9       �	V���Xc�A�3*

loss�;���s       �	�2�Xc�A�3*

loss#l<�h��       �	��Xc�A�3*

loss�ԅ8�W��       �	A��Xc�A�3*

lossc:{:3ؚ       �	�"�Xc�A�3*

loss��H9��       �	o��Xc�A�3*

loss�e8Nf       �	�W�Xc�A�3*

lossY�8u!�)       �	���Xc�A�3*

loss̰<���O       �	���Xc�A�3*

loss���;blV�       �	���Xc�A�3*

loss�U�:u�5       �	� ��Xc�A�3*

loss3�9�2��       �	����Xc�A�3*

loss��a; 3��       �	2Z��Xc�A�3*

lossj1y=W��       �	t���Xc�A�3*

loss�a�8�hvE       �	b���Xc�A�3*

loss<�=5��       �	!��Xc�A�3*

loss�m<_���       �	���Xc�A�3*

loss&.<��       �	�R��Xc�A�3*

loss�V39�|�       �	,���Xc�A�3*

lossC��;Vk�       �	����Xc�A�3*

lossC�g<�`       �	�/��Xc�A�3*

loss��;o���       �	z���Xc�A�3*

loss�x�:��7F       �	SZ��Xc�A�3*

losszU:×��       �	5���Xc�A�3*

loss��;A���       �	���Xc�A�3*

loss=��<�}��       �	u"��Xc�A�3*

loss,t;}�q       �	k���Xc�A�3*

loss9!;�'��       �	T��Xc�A�3*

loss��<�M�n       �	����Xc�A�3*

loss�mL=�)       �	 ���Xc�A�3*

loss[�8<�!�U       �	I+ �Xc�A�3*

loss��:]��h       �	r� �Xc�A�3*

loss�`�<�8o       �	@��Xc�A�3*

lossx�i;�T)       �	�j�Xc�A�3*

loss�.<��b       �	*�Xc�A�3*

lossW�;�"       �	��Xc�A�3*

loss��;�r\�       �	�b�Xc�A�3*

loss;�9f�4D       �	{��Xc�A�3*

loss��6<b2�0       �	��Xc�A�3*

loss�[h:�Gٜ       �	�3�Xc�A�3*

loss�ؙ;�?]�       �	���Xc�A�3*

lossy�=沒s       �	�x�Xc�A�3*

loss�*�;j�"�       �	��Xc�A�3*

loss�T;~�       �	���Xc�A�3*

loss���;&L]       �	��	�Xc�A�3*

loss�+�;���       �	|)
�Xc�A�3*

loss� t9�L3�       �	��
�Xc�A�3*

loss�O;tq"�       �	���Xc�A�3*

lossN]=<��8�       �	()�Xc�A�3*

loss�3e<�8�       �	"��Xc�A�3*

lossP�=�~B�       �	;p�Xc�A�3*

lossV�p;T�D       �	o�Xc�A�3*

loss��<Y��       �	���Xc�A�3*

loss��q:�~�       �	C��Xc�A�3*

loss�Y:1���       �	1�Xc�A�3*

loss*K�9��uX       �	���Xc�A�3*

loss�S�<7fGk       �	�b�Xc�A�3*

loss�];Gy��       �	2�Xc�A�3*

lossd"C:Fzk       �	ʤ�Xc�A�3*

loss�Ȼ;WW�       �	�@�Xc�A�3*

lossJ��<U+�t       �	{��Xc�A�3*

loss�U�;uC       �	�v�Xc�A�3*

loss�&;��9       �	R*�Xc�A�3*

loss`�9���0       �	P��Xc�A�3*

loss��:@���       �	l�Xc�A�3*

loss���;me�       �	�-4�Xc�A�3*

loss���;�f�j       �	d�4�Xc�A�3*

lossX��:v�m       �	ŭ5�Xc�A�3*

loss��<�d��       �	�I6�Xc�A�3*

loss@�/;��^�       �	�6�Xc�A�3*

lossfQ�<�g�       �	��7�Xc�A�3*

loss���;X芩       �	{8�Xc�A�3*

loss���<�}�       �	#I9�Xc�A�3*

loss?��=|��H       �	��9�Xc�A�3*

loss<�;/�8�       �	hz:�Xc�A�3*

lossnFK:ma�       �	n;�Xc�A�3*

loss!��;���       �	��;�Xc�A�3*

loss:�Y<A=       �	��<�Xc�A�3*

loss;:i��d       �	�d=�Xc�A�3*

loss��:ģ��       �	��=�Xc�A�3*

loss��(=yX�G       �	��>�Xc�A�3*

loss�͇;Qp�z       �	L�?�Xc�A�3*

lossh!�:N�M       �	��@�Xc�A�3*

loss7[�:�Y�;       �	�DA�Xc�A�3*

loss_4�=�.[j       �	��A�Xc�A�3*

lossE|d:ctw       �	�|B�Xc�A�3*

loss�Wh<�DHl       �	K"C�Xc�A�3*

loss�O�9�	�       �	��C�Xc�A�3*

loss��<�U�       �	WwD�Xc�A�3*

loss�M�;��-�       �	�E�Xc�A�3*

loss�-Q<��;�       �	F�E�Xc�A�3*

loss}V5; ��       �	��F�Xc�A�3*

loss�ǟ;W^��       �	�KG�Xc�A�3*

loss|�<�	�       �	�1H�Xc�A�3*

lossFB�:�g�       �	R�H�Xc�A�3*

lossE��;�]��       �	ƅI�Xc�A�3*

loss_�:����       �	:$J�Xc�A�3*

loss��;���/       �	r�J�Xc�A�3*

loss	׎<SQ��       �	[]K�Xc�A�3*

lossja�:���       �	��K�Xc�A�3*

loss&��<~ґ:       �	��L�Xc�A�3*

lossW�X;��       �	�>M�Xc�A�3*

lossH��<���       �	,�M�Xc�A�3*

lossJ¹<�?��       �	znN�Xc�A�3*

loss�=��       �	�O�Xc�A�3*

loss�i<���\       �	��O�Xc�A�3*

loss�a�::p\N       �	�?P�Xc�A�3*

loss&L=j�       �	�P�Xc�A�3*

lossa��:��I       �	1xQ�Xc�A�3*

loss���:)��       �	�R�Xc�A�3*

loss�l�<�t�       �	ɯR�Xc�A�3*

loss���:�i��       �	�MS�Xc�A�3*

loss��
;�6�d       �	��S�Xc�A�3*

loss,��:;��       �	b�T�Xc�A�3*

loss���8j���       �	�\U�Xc�A�3*

lossS��;��        �	
V�Xc�A�3*

lossR�x<�g@u       �	3W�Xc�A�3*

loss]A�<����       �	H�W�Xc�A�3*

loss�`�<�t�       �	�}X�Xc�A�3*

loss���<��d       �	:"Y�Xc�A�3*

loss��9�Qɉ       �	,�Y�Xc�A�4*

loss1��:4Q�       �	nQZ�Xc�A�4*

lossi�9S��       �	G�Z�Xc�A�4*

loss7j�:��h       �	�|[�Xc�A�4*

loss�M�=���       �	�\�Xc�A�4*

lossݯ�=(m       �	E�\�Xc�A�4*

lossțh;v��       �	�Z]�Xc�A�4*

loss�{;�o�       �	��]�Xc�A�4*

loss�E�9�ܤ�       �	 �^�Xc�A�4*

loss�a=m�u       �	�0_�Xc�A�4*

loss�c�:���       �	��_�Xc�A�4*

loss��<	�.       �	|a`�Xc�A�4*

lossXV<��       �	Q�`�Xc�A�4*

loss�^�;$SN       �	?�a�Xc�A�4*

lossf�<DNx�       �	%$b�Xc�A�4*

loss{M=`�z(       �	c�b�Xc�A�4*

loss'�=����       �	<Pc�Xc�A�4*

loss�@<i>�       �	1�c�Xc�A�4*

loss�V:�
       �	 �d�Xc�A�4*

loss�%�;z��z       �	�;e�Xc�A�4*

loss�z�;'5ד       �	��e�Xc�A�4*

lossj�:d苑       �	��f�Xc�A�4*

loss��;�C��       �	�<g�Xc�A�4*

loss8 �<�J>\       �	Rh�Xc�A�4*

loss`; }�       �	d�h�Xc�A�4*

lossOx�;[/��       �	>�i�Xc�A�4*

loss��;n\��       �	�7j�Xc�A�4*

loss�(�;��$4       �	�j�Xc�A�4*

lossk��;ӣ�       �	~qk�Xc�A�4*

lossJ[;J�DU       �	�l�Xc�A�4*

loss1��:M-��       �	Úl�Xc�A�4*

loss*=���       �	s0m�Xc�A�4*

loss�1m<��\�       �	��m�Xc�A�4*

lossԏ;�ga,       �	>Yn�Xc�A�4*

loss�3�:�a�       �	��n�Xc�A�4*

loss�i;;��       �	V�o�Xc�A�4*

loss��<- .H       �	p�Xc�A�4*

loss�y:dֆU       �	&�p�Xc�A�4*

loss��=n1S�       �	�<q�Xc�A�4*

loss��=?��5       �	�q�Xc�A�4*

loss�0�:A��       �	�Zr�Xc�A�4*

lossh�<��       �	��r�Xc�A�4*

lossf��;�\K�       �	�}s�Xc�A�4*

loss:!<\��       �	t�Xc�A�4*

loss�#;3S͍       �	q�t�Xc�A�4*

loss�;��2�       �	Ou�Xc�A�4*

losst�x<�2�u       �	��u�Xc�A�4*

lossRG�;k��Z       �	�zv�Xc�A�4*

loss��o;�-�       �	Gw�Xc�A�4*

loss�;�;]��d       �	��w�Xc�A�4*

lossi��9�L51       �	%yx�Xc�A�4*

loss�:�Gl1       �	�y�Xc�A�4*

loss��I:��(       �	*�y�Xc�A�4*

loss��N=��n       �	kEz�Xc�A�4*

lossFYR;���(       �	#�z�Xc�A�4*

loss?��<�ʮl       �	vq{�Xc�A�4*

loss�}l;D1��       �	B|�Xc�A�4*

loss2�"=h��       �	��|�Xc�A�4*

loss\��8��*�       �	�.}�Xc�A�4*

loss6J�:��e       �	��}�Xc�A�4*

lossZΔ;�n4�       �	��~�Xc�A�4*

loss��=��~�       �	73�Xc�A�4*

loss��o<��       �	���Xc�A�4*

loss*��;Y�       �	�c��Xc�A�4*

loss8�	;U �       �		���Xc�A�4*

loss�4�<�
��       �	����Xc�A�4*

loss���;K�0       �	)��Xc�A�4*

loss���:�$5       �	Â�Xc�A�4*

loss��:�qM�       �	c��Xc�A�4*

loss�H�:@#�       �	����Xc�A�4*

loss>�<G,W�       �	0���Xc�A�4*

lossc��:�oA       �	1D��Xc�A�4*

lossq�-<��M�       �	:ꅭXc�A�4*

loss��<��C�       �	W���Xc�A�4*

loss��;A�K       �	&7��Xc�A�4*

loss�Y�;)�`�       �	a݇�Xc�A�4*

loss��=J�j�       �	C���Xc�A�4*

lossI��;�@�u       �	�N��Xc�A�4*

loss��D;��X�       �	V�Xc�A�4*

lossjG�;#Hb;       �	���Xc�A�4*

lossߦ�:y�       �	o/��Xc�A�4*

loss(�=�H��       �	�勭Xc�A�4*

loss��:A>R5       �	J���Xc�A�4*

lossz��:���;       �	"5��Xc�A�4*

loss�q:�Vd�       �	�ҍ�Xc�A�4*

loss*U�; kin       �	0���Xc�A�4*

lossxʱ<Xa       �	$��Xc�A�4*

loss��&;\'�X       �	�Ə�Xc�A�4*

loss-�:�Ex       �	vo��Xc�A�4*

loss��^=�y�-       �	0��Xc�A�4*

loss�n�;��T       �	Sϑ�Xc�A�4*

loss�.+:�ǟ�       �	�w��Xc�A�4*

loss$�;�7U�       �	J	��Xc�A�4*

loss�B[;�`s�       �	����Xc�A�4*

loss�_�;"c�       �	"4��Xc�A�4*

loss $~:��4�       �	�P��Xc�A�4*

lossCW�;\(�       �	`畭Xc�A�4*

lossw��<�E)       �	���Xc�A�4*

lossN{<��"S       �	���Xc�A�4*

losscjA<�� C       �	_���Xc�A�4*

loss܍�8��       �	�P��Xc�A�4*

loss��;
���       �	2瘭Xc�A�4*

lossf�;�`��       �	�}��Xc�A�4*

loss(�;�u��       �	���Xc�A�4*

lossJ��:ad]�       �	i���Xc�A�4*

loss�K�<
��       �	�D��Xc�A�4*

loss��o;ԕ�       �	8ٛ�Xc�A�4*

loss��6<0�       �	�l��Xc�A�4*

loss\�:���       �	���Xc�A�4*

loss��;�^��       �	����Xc�A�4*

lossք7;jM�	       �	񂞭Xc�A�4*

lossԋ5;�       �	*��Xc�A�4*

lossvk�;pps�       �	����Xc�A�4*

loss��[:h�*�       �	o���Xc�A�4*

loss�ݕ<fؓ       �	� ��Xc�A�4*

loss-j�:]Q       �	����Xc�A�4*

loss�=b:i��       �	�V��Xc�A�4*

loss��S9���       �	�Xc�A�4*

loss�<��m&       �	����Xc�A�4*

loss\��9��=(       �	�-��Xc�A�4*

loss;w|��       �	7Ǥ�Xc�A�4*

loss��\<���       �	n��Xc�A�4*

lossw��=��i       �	B	��Xc�A�4*

loss��:�Vy       �	ݦ�Xc�A�4*

loss�m[8�Џh       �	hv��Xc�A�4*

lossN�`9����       �	���Xc�A�4*

loss�};�R��       �	���Xc�A�4*

lossH��<���T       �	@m��Xc�A�4*

loss�uC:g�b�       �	���Xc�A�5*

lossW�!:����       �	����Xc�A�5*

loss�|�:��       �	�0��Xc�A�5*

lossL��<E���       �	�ī�Xc�A�5*

loss��C;�[�       �	�V��Xc�A�5*

loss���;��5�       �	)묭Xc�A�5*

loss4�7<
i6�       �	�~��Xc�A�5*

loss�zl<0�27       �	��Xc�A�5*

loss�Tp:����       �	����Xc�A�5*

loss�Ɣ:&��       �	`>��Xc�A�5*

loss���9�`       �	�ӯ�Xc�A�5*

loss;��<zv�       �	�e��Xc�A�5*

lossHn�<,�~�       �	���Xc�A�5*

loss�D#:�z��       �	�Ʊ�Xc�A�5*

lossS��<�4��       �	�\��Xc�A�5*

loss�̰;z��`       �	yY��Xc�A�5*

loss&��<��wW       �	�곭Xc�A�5*

loss��:[�J�       �	���Xc�A�5*

lossb;�Z��       �	1Ҷ�Xc�A�5*

loss�<=CBE       �	f��Xc�A�5*

loss��a;&8�<       �	2��Xc�A�5*

loss��9�6��       �	E���Xc�A�5*

losstZ;
'y�       �	L6��Xc�A�5*

loss�C�<�'�<       �	mɹ�Xc�A�5*

lossZ�<�އ~       �	d��Xc�A�5*

loss칌;8�"[       �	����Xc�A�5*

loss�:��       �	����Xc�A�5*

lossE	i:ɰ'       �	.9��Xc�A�5*

lossR�;p��<       �	h鼭Xc�A�5*

loss&��9�fC`       �	/���Xc�A�5*

loss��:b2x'       �	�#��Xc�A�5*

loss+(;� ��       �	?ྭXc�A�5*

loss`3C;�9��       �	�r��Xc�A�5*

loss)B=��j       �	���Xc�A�5*

loss3"�;X_p�       �	���Xc�A�5*

loss��;*W�       �	�U��Xc�A�5*

loss�x�=��,�       �	oH­Xc�A�5*

loss�Y8r�&�       �	��­Xc�A�5*

lossrUE9a       �	a�íXc�A�5*

loss�'x:��'       �	HNĭXc�A�5*

loss�"�<O.��       �	.ŭXc�A�5*

lossO�;2�\       �	��ŭXc�A�5*

loss$ z;�!��       �	NEǭXc�A�5*

loss��<���       �	]�ǭXc�A�5*

loss��e<���       �	6�ȭXc�A�5*

loss&:�Um,       �	
�ɭXc�A�5*

loss��<9~_L#       �	�XʭXc�A�5*

loss��:%��&       �	k�ʭXc�A�5*

loss�9U�W       �	͐˭Xc�A�5*

loss�!:��f       �	�$̭Xc�A�5*

loss���<���M       �	w�̭Xc�A�5*

loss�֌<8'�       �	NcͭXc�A�5*

loss<l�<�Ұ$       �	xέXc�A�5*

loss�}a;R��       �	עέXc�A�5*

loss��<3��       �	=GϭXc�A�5*

lossQ>T:nO�*       �	��ϭXc�A�5*

losszڅ9�/�0       �	�yЭXc�A�5*

lossd�;��>       �	�ѭXc�A�5*

loss�J�:�z�o       �	��ѭXc�A�5*

lossH̄;,U�X       �	�gҭXc�A�5*

lossڋ6='��       �	�ӭXc�A�5*

loss,�<���       �	�ӭXc�A�5*

loss�%9�K��       �	O<ԭXc�A�5*

loss�N;t�f�       �	4�ԭXc�A�5*

loss*	=+8       �	yvխXc�A�5*

loss�=+       �	��֭Xc�A�5*

lossa��:�       �	��׭Xc�A�5*

loss�H<��3[       �	x`حXc�A�5*

loss�j�;��c�       �	�"٭Xc�A�5*

lossq�;s���       �	��٭Xc�A�5*

losss�f:�q       �	��ڭXc�A�5*

loss$�<���        �	��ۭXc�A�5*

loss� F9G�.�       �	W	ݭXc�A�5*

lossmN:R�N�       �	�PޭXc�A�5*

loss{^�:�i��       �	N�ޭXc�A�5*

loss�X;eF@�       �	֏߭Xc�A�5*

lossQ='���       �	�4�Xc�A�5*

loss�@�=��3m       �	Y��Xc�A�5*

loss��A;��r       �	hy�Xc�A�5*

lossve"<k���       �	��Xc�A�5*

loss\��8�.^�       �	[��Xc�A�5*

loss�6=QvP�       �	�N�Xc�A�5*

loss1e�9�a̼       �	G�Xc�A�5*

loss��x=���       �	|��Xc�A�5*

loss���<}�1#       �	2U�Xc�A�5*

loss�;<I-)6       �	=��Xc�A�5*

loss�#=C��       �	���Xc�A�5*

loss�=�rz       �	*�Xc�A�5*

loss�� <��{�       �	���Xc�A�5*

lossf�y<��ҹ       �	�]�Xc�A�5*

loss̦@<�Y�       �	v��Xc�A�5*

loss�@<�S       �	��Xc�A�5*

loss�TD;�|V       �	�w�Xc�A�5*

loss�P�:*cѲ       �	�Xc�A�5*

loss�y�<p��       �	��Xc�A�5*

loss��M=���T       �	=�Xc�A�5*

loss���9/i��       �	s��Xc�A�5*

loss�R8�=[       �	io��Xc�A�5*

loss��
;g�@s       �	h�Xc�A�5*

loss���;���       �	���Xc�A�5*

loss/w:�Ӳc       �	;5�Xc�A�5*

loss��;/dԻ       �	���Xc�A�5*

loss�V�;��J6       �	&q�Xc�A�5*

loss��l;�"�       �	��Xc�A�5*

loss-�:-XB6       �	���Xc�A�5*

loss�v�::�\6       �	�M�Xc�A�5*

loss%<��[5       �	���Xc�A�5*

lossC*9�u��       �	��Xc�A�5*

loss*�;,!�       �	zr��Xc�A�5*

loss�QV9�Cە       �	g��Xc�A�5*

loss�F�8��Xq       �	����Xc�A�5*

lossS��;'[�       �	rM��Xc�A�5*

loss�];��e�       �	����Xc�A�5*

loss,-P<�k";       �	ҋ��Xc�A�5*

loss�b�<��i       �	+���Xc�A�5*

loss�P�:�Q��       �	2v��Xc�A�5*

lossp"<��N]       �	^0��Xc�A�5*

loss�BO:��U       �	����Xc�A�5*

loss���:����       �	����Xc�A�5*

loss 
�:�z!v       �	=G��Xc�A�5*

lossN �;R�:�       �	����Xc�A�5*

loss[2J=�8��       �	L��Xc�A�5*

loss��G=���
       �	A���Xc�A�5*

loss��U9O��       �	����Xc�A�5*

loss�r�;�D�C       �	& �Xc�A�5*

loss���:��       �	Ǻ �Xc�A�5*

lossD�:'�        �	]N�Xc�A�5*

loss��:�9̠       �	���Xc�A�5*

loss��K9c�y�       �	���Xc�A�6*

lossf15;�r'       �	gF�Xc�A�6*

loss���;��Ko       �	���Xc�A�6*

loss��9���       �	>u�Xc�A�6*

loss��<&��-       �	9E�Xc�A�6*

loss��-:o���       �	]��Xc�A�6*

lossJ&�9� {�       �	���Xc�A�6*

loss欄;��$       �	Oy�Xc�A�6*

loss��-<��g�       �	�"�Xc�A�6*

loss��U;���;       �	���Xc�A�6*

loss,�K:}R�       �	�^	�Xc�A�6*

lossZY;ߌC�       �	��	�Xc�A�6*

lossT::��a       �	/�
�Xc�A�6*

lossh:�;o�i�       �	aP�Xc�A�6*

loss���:��Q�       �	���Xc�A�6*

loss��&<�-
�       �	��Xc�A�6*

loss�k�:}n��       �	xE�Xc�A�6*

losshOW=��o*       �	���Xc�A�6*

loss=�=��Hl       �	5~�Xc�A�6*

loss.��;=\�       �	E,�Xc�A�6*

loss{�z:���       �	���Xc�A�6*

loss�bU:bCi�       �	�`�Xc�A�6*

loss��:�X       �	L��Xc�A�6*

loss�ȡ<��s       �	u��Xc�A�6*

loss�{N:q��       �	8-�Xc�A�6*

loss\h�<;��       �	��Xc�A�6*

loss��;�W�O       �	�g�Xc�A�6*

lossl�=���q       �	���Xc�A�6*

loss��9#|v       �	p��Xc�A�6*

loss��9
M�       �	�+�Xc�A�6*

loss�_y:�ո�       �	���Xc�A�6*

loss�U:����       �	e��Xc�A�6*

loss�{<�XN       �	l@�Xc�A�6*

loss�]�=��(       �	�Xc�A�6*

loss�Y�<r�X�       �	h��Xc�A�6*

lossX)<��^O       �	���Xc�A�6*

loss��=<�.�       �	�V�Xc�A�6*

lossJ(=L�N�       �	��Xc�A�6*

loss�;�;��       �	���Xc�A�6*

loss�(<���       �	�P�Xc�A�6*

losszT<c�t       �	w�Xc�A�6*

loss�=݇4f       �	�,�Xc�A�6*

loss*�o9AT(�       �	B��Xc�A�6*

loss��;�P/       �	���Xc�A�6*

loss1;�{��       �	�� �Xc�A�6*

loss�;�Q�s       �	�|!�Xc�A�6*

lossJ�:D	       �	K["�Xc�A�6*

loss.�;$6�,       �	(�#�Xc�A�6*

lossCs�:=���       �	9$�Xc�A�6*

loss[�/;?��       �	�$�Xc�A�6*

loss\�3<r��|       �	�%�Xc�A�6*

lossME�9p]p�       �	�\&�Xc�A�6*

loss$�6<o�I       �	��&�Xc�A�6*

loss?��<�M`�       �	��'�Xc�A�6*

loss]�;~ѪO       �	l%(�Xc�A�6*

loss��<	��       �	��(�Xc�A�6*

loss���;�N       �	�j)�Xc�A�6*

loss:��;|�3�       �	*�Xc�A�6*

lossң�;o=��       �	<�*�Xc�A�6*

lossX��<��u�       �	�C+�Xc�A�6*

loss�%/;J���       �	V�+�Xc�A�6*

loss}�;K�q5       �	�n,�Xc�A�6*

loss��\=y���       �	�-�Xc�A�6*

loss/f;��:�       �	�-�Xc�A�6*

lossrgd<�L#*       �	JF.�Xc�A�6*

loss� k<{^�^       �	��.�Xc�A�6*

loss���;�ɸ�       �	Sx/�Xc�A�6*

loss]�e<*x�?       �	�0�Xc�A�6*

loss�ڨ<�K��       �	��0�Xc�A�6*

loss(��:2=�       �	�V1�Xc�A�6*

loss�]�;�Xْ       �	�2�Xc�A�6*

lossIk=�	��       �	� 3�Xc�A�6*

loss���<!�
       �	��3�Xc�A�6*

lossA�:I�j       �	^K4�Xc�A�6*

loss.4t;SGe�       �	��4�Xc�A�6*

loss��%=��{'       �	T�5�Xc�A�6*

lossf�<±:       �	'.6�Xc�A�6*

loss Y9,���       �	P�6�Xc�A�6*

loss���:���       �	r8�Xc�A�6*

loss�?#=�u�,       �	��8�Xc�A�6*

loss�:e��       �	_\9�Xc�A�6*

lossNB�:���       �	3�9�Xc�A�6*

lossE��;��jv       �	B�:�Xc�A�6*

lossP;6$#       �	0;�Xc�A�6*

loss��<ny
{       �	�;�Xc�A�6*

loss���9�&)        �	�i<�Xc�A�6*

loss��:I�WE       �	��<�Xc�A�6*

loss��9���       �	��=�Xc�A�6*

loss�Z<��}        �	�I>�Xc�A�6*

lossH�<bl�       �	�2?�Xc�A�6*

loss��9�O�       �	b�?�Xc�A�6*

lossA�h;$'M       �	v@�Xc�A�6*

lossBԟ9�m�       �	�A�Xc�A�6*

lossM0�<��       �	�A�Xc�A�6*

loss��9���       �	/PB�Xc�A�6*

loss�ؒ;;�4       �	y�B�Xc�A�6*

loss��&=jN       �	�C�Xc�A�6*

loss���:���       �	PD�Xc�A�6*

lossM� :k<��       �	��D�Xc�A�6*

loss,5<o��       �	e�E�Xc�A�6*

loss��<���       �	�(F�Xc�A�6*

loss�6;��Kp       �	`�F�Xc�A�6*

loss���=�m�       �	;�G�Xc�A�6*

loss�<�<!���       �	)"H�Xc�A�6*

loss��:3�d       �	/�H�Xc�A�6*

loss�m�:%��       �	�bI�Xc�A�6*

loss|Q>;��i       �	�J�Xc�A�6*

loss���<��}       �	��J�Xc�A�6*

loss2<�Ҿ�       �	�K�Xc�A�6*

loss�F;k�       �	?L�Xc�A�6*

loss��;8ׂ�       �	��L�Xc�A�6*

loss��W:3G*�       �	��M�Xc�A�6*

lossq�<"T�       �	�+N�Xc�A�6*

lossn4;���       �	K�N�Xc�A�6*

loss\;:���       �	g{O�Xc�A�6*

loss��~;���[       �	 P�Xc�A�6*

loss�<n�       �	��P�Xc�A�6*

loss�`<�+1G       �	�SQ�Xc�A�6*

loss2�e;��x�       �	�Q�Xc�A�6*

loss�ɇ=S���       �	�R�Xc�A�6*

loss/;�!V�       �	b-S�Xc�A�6*

loss���<z�       �	g�T�Xc�A�6*

loss3P�;�e1�       �	b/U�Xc�A�6*

loss�&L;y�R�       �	��U�Xc�A�6*

loss���<�:6       �	hV�Xc�A�6*

loss�4�;��#?       �	�W�Xc�A�6*

lossȱ�;�&�       �	tEX�Xc�A�6*

lossJ�;돆&       �	�Y�Xc�A�6*

loss��	=��y�       �	�Y�Xc�A�7*

loss�V;�mR       �	�Z�Xc�A�7*

loss��;<�t�       �	(~[�Xc�A�7*

lossc<�`�       �	\�Xc�A�7*

loss���<-��       �	��\�Xc�A�7*

loss�$�<��       �	@h]�Xc�A�7*

loss
�H;�&�k       �	^�Xc�A�7*

lossl9<a�L�       �	0�^�Xc�A�7*

loss��:g��       �	�_�Xc�A�7*

loss�H}:S5        �	|+`�Xc�A�7*

loss��(<�'��       �	aQa�Xc�A�7*

loss�#�<��,       �	��a�Xc�A�7*

loss���<�U�       �	P�b�Xc�A�7*

lossL>;GR�       �	
Hc�Xc�A�7*

lossl�,;��.       �	�c�Xc�A�7*

lossf�M=*s�]       �	I�d�Xc�A�7*

lossq�:NX       �	Kse�Xc�A�7*

loss;K^=LRI       �	Nf�Xc�A�7*

loss��
;�$^�       �	ͬf�Xc�A�7*

lossR2�;��f�       �	Bg�Xc�A�7*

loss��;�*�       �	��g�Xc�A�7*

lossH"<Ji$	       �	Owh�Xc�A�7*

loss�?:�x�G       �	Yi�Xc�A�7*

loss�@;[�	�       �	�i�Xc�A�7*

loss��V=�R5o       �	6xj�Xc�A�7*

lossAjl:��U       �	.k�Xc�A�7*

loss.ؽ;b��       �	��k�Xc�A�7*

lossX�f9Jtg�       �	�pl�Xc�A�7*

loss�;I;��9�       �	�m�Xc�A�7*

loss�Y�;Ǩ��       �	$�m�Xc�A�7*

loss�P�8eZ�       �	6\n�Xc�A�7*

lossv)�=d]S       �	o�Xc�A�7*

loss�kt<�       �	�o�Xc�A�7*

loss�:+p��       �	&Up�Xc�A�7*

loss;�v<��o�       �	Z�p�Xc�A�7*

lossT_c;w�[       �	��q�Xc�A�7*

loss��;���       �	_~r�Xc�A�7*

lossDF<2���       �	�*s�Xc�A�7*

loss���<j�       �	��s�Xc�A�7*

loss*F�;_~�j       �	؟t�Xc�A�7*

loss$B5=V��       �	�vu�Xc�A�7*

loss�4=K�f$       �	�v�Xc�A�7*

lossT��:{��       �	<�v�Xc�A�7*

lossj��;�F��       �	�?w�Xc�A�7*

loss;�O;�'w�       �	��w�Xc�A�7*

lossP�:~�sr       �	Lnx�Xc�A�7*

loss���;����       �	�y�Xc�A�7*

lossJ�#<�qIP       �	��y�Xc�A�7*

lossqص<t�*�       �	�Hz�Xc�A�7*

loss��<\k5�       �	��z�Xc�A�7*

loss�B<���       �	�z{�Xc�A�7*

loss3��:��
       �	:!|�Xc�A�7*

loss�]5;�3�%       �	b�|�Xc�A�7*

loss�ۘ9I�]�       �	�l}�Xc�A�7*

loss\<T�T       �	>~�Xc�A�7*

lossVF=�G�       �	��~�Xc�A�7*

loss��=k#��       �	C��Xc�A�7*

loss��!=3�Z�       �	�#��Xc�A�7*

loss�S�<���       �	񽀮Xc�A�7*

loss�=�9ė       �	�U��Xc�A�7*

loss�
;J�U       �	9Xc�A�7*

loss�;�=V
��       �	&���Xc�A�7*

loss��,;�UM       �	T9��Xc�A�7*

loss+;=Ҝ�       �	�샮Xc�A�7*

loss�O<:aTT       �	̄�Xc�A�7*

loss��<~�r}       �	�p��Xc�A�7*

loss��<Ҋc�       �	)��Xc�A�7*

loss�m<
yj       �	φ�Xc�A�7*

loss�;��Da       �	Qڇ�Xc�A�7*

loss��9�y�?       �	���Xc�A�7*

loss3/�;\CA       �	�`��Xc�A�7*

loss]>";Zj6�       �	P��Xc�A�7*

loss(?�<sW.       �	����Xc�A�7*

lossi�U=�       �	����Xc�A�7*

lossV-=�k/�       �	�4��Xc�A�7*

lossR�(;>Ӄ�       �	�ߌ�Xc�A�7*

loss@9�9�
�       �	φ��Xc�A�7*

loss��<}�~       �	{1��Xc�A�7*

loss#ǥ<J^�       �	I؎�Xc�A�7*

loss�o�:�C�       �	�z��Xc�A�7*

lossA�<t��       �	f��Xc�A�7*

loss�]�:
���       �	����Xc�A�7*

lossXF�;��|m       �	�q��Xc�A�7*

lossDq9GKg�       �	�	��Xc�A�7*

lossT�:�*��       �	����Xc�A�7*

loss�r<�*!�       �	0H��Xc�A�7*

loss�s;s��       �	擮Xc�A�7*

lossj�:�~n�       �		Ŕ�Xc�A�7*

loss�x <ӝ       �	�Y��Xc�A�7*

loss�z;)�t�       �	Z�Xc�A�7*

loss�͟;T��       �	4���Xc�A�7*

loss_:XL�L       �	ԗ�Xc�A�7*

loss��=9��#       �	l��Xc�A�7*

loss�m:#Q�       �	����Xc�A�7*

lossD�<:�m/S       �	�X��Xc�A�7*

loss��9��       �	����Xc�A�7*

loss,�2;�ww       �	���Xc�A�7*

loss��H:�AB�       �	�E��Xc�A�7*

loss�ѡ<��T       �	�K��Xc�A�7*

lossC,�7:��       �	�{��Xc�A�7*

loss�(<����       �	7��Xc�A�7*

loss���<�H�       �	����Xc�A�7*

loss�J�7##°       �	�2��Xc�A�7*

loss�$9�/w       �	�⢮Xc�A�7*

loss�)b8n�x�       �	����Xc�A�7*

loss���;�>��       �	T��Xc�A�7*

loss�ڨ;�A1�       �	���Xc�A�7*

lossV;vb       �	����Xc�A�7*

lossc�J<$��       �	�Q��Xc�A�7*

loss�56=�[v       �	rp��Xc�A�7*

lossqi9�.�H       �	o��Xc�A�7*

loss�Q=���       �	3ݨ�Xc�A�7*

lossi�<M��       �	���Xc�A�7*

loss�ce=�ќu       �	T7��Xc�A�7*

loss,�<:��|        �	�ݪ�Xc�A�7*

loss2#�9���       �	Wv��Xc�A�7*

loss�-�=ɽ�f       �	�!��Xc�A�7*

loss���;�({�       �	Fέ�Xc�A�7*

loss�z:C�       �	ut��Xc�A�7*

loss���;Ue�       �	���Xc�A�7*

loss� n;����       �	ȷ��Xc�A�7*

loss�s<l�k       �	�W��Xc�A�7*

loss��<<X�Z       �	�Xc�A�7*

loss�:zҫ       �	v���Xc�A�7*

loss�a�=��T       �	)"��Xc�A�7*

loss@T<*�       �	ǲ�Xc�A�7*

loss���<���d       �	'k��Xc�A�7*

loss�u�<�P(�       �	O<��Xc�A�7*

loss�%<(�_       �	ܴ�Xc�A�8*

lossD�:>�L       �	w���Xc�A�8*

loss< 9;����       �	���Xc�A�8*

loss�I';,#��       �	Ǻ��Xc�A�8*

loss)ĥ=��       �	�X��Xc�A�8*

loss8<z;Hi��       �	���Xc�A�8*

loss��:H��       �	����Xc�A�8*

loss�~�:��R       �	�*��Xc�A�8*

loss#�w;q�       �	\ɹ�Xc�A�8*

loss��s<_�       �	�_��Xc�A�8*

loss<��<[ �S       �	����Xc�A�8*

loss�,0=����       �	/���Xc�A�8*

loss���<Q�Vp       �	"O��Xc�A�8*

loss��;��ߨ       �	l|��Xc�A�8*

lossV��:ĉVs       �	�+��Xc�A�8*

loss��=���0       �	����Xc�A�8*

loss�;��66       �	�l��Xc�A�8*

loss$�9��x�       �	h�®Xc�A�8*

loss�M�;��d       �	oFîXc�A�8*

loss]�d;��Ȏ       �	��îXc�A�8*

loss�Er;�\m\       �	�tĮXc�A�8*

loss9<LY�N       �	+ŮXc�A�8*

loss|��8;�&U       �	ѲŮXc�A�8*

loss!�;mȺ`       �	�kƮXc�A�8*

loss���<N��h       �	��ǮXc�A�8*

loss�W�9�#ƍ       �	��ȮXc�A�8*

lossѝ�;��7�       �	eoɮXc�A�8*

loss��Y=�*b�       �	�ʮXc�A�8*

lossɁ$;'�N       �	ٱʮXc�A�8*

lossj�m=�q�_       �	%YˮXc�A�8*

loss./==d��       �	A�ˮXc�A�8*

loss��:�1��       �	��̮Xc�A�8*

losslp�9.�v       �	J&ͮXc�A�8*

lossE�.<{���