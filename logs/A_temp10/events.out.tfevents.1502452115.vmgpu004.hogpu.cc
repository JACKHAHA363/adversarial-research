       �K"	  �dfc�Abrain.Event:2R_�:5�     Or�	� �dfc�A"��
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
conv2d_1/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0
`
conv2d_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *�x�
`
conv2d_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�x=
�
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
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
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
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
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������@*
paddingVALID*
T0*
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
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   
`
conv2d_2/random_uniform/minConst*
valueB
 *�\1�*
_output_shapes
: *
dtype0
`
conv2d_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�\1=
�
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:@@*
seed2��z*
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
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
:@@*
T0
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
conv2d_2/ConstConst*
_output_shapes
:@*
dtype0*
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
use_locking(*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
dtype0*
_output_shapes
:*
valueB"      
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
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:���������@
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
dropout_1/cond/mulMuldropout_1/cond/mul/Switch:1dropout_1/cond/mul/y*
T0*/
_output_shapes
:���������@

 dropout_1/cond/dropout/keep_probConst^dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  @?
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
:���������@*
seed2��*
dtype0*
T0*
seed���)
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
flatten_1/ShapeShapedropout_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0
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
N*
T0*
_output_shapes
:*

axis 
�
flatten_1/ReshapeReshapedropout_1/cond/Mergeflatten_1/stack*
T0*0
_output_shapes
:������������������*
Tshape0
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
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*!
_output_shapes
:���*
seed2�ʶ
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
shared_name *
dtype0*
shape:���*!
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
dense_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
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
dense_1/bias/readIdentitydense_1/bias*
T0*
_output_shapes	
:�*
_class
loc:@dense_1/bias
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
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
3dropout_2/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_2/cond/dropout/Shape*(
_output_shapes
:����������*
seed2ף�*
dtype0*
T0*
seed���)
�
)dropout_2/cond/dropout/random_uniform/subSub)dropout_2/cond/dropout/random_uniform/max)dropout_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
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
dense_2/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
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
 *̈́U>*
_output_shapes
: *
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed���)*
T0*
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
	container *
dtype0*
shared_name *
shape:	�

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
'sequential_1/conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
/sequential_1/conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
:���������@*
data_formatNHWC*
T0
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
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
out_type0*
_output_shapes
:*
T0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
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
seed2���*
dtype0*
T0*
seed���)
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
$sequential_1/flatten_1/strided_sliceStridedSlicesequential_1/flatten_1/Shape*sequential_1/flatten_1/strided_slice/stack,sequential_1/flatten_1/strided_slice/stack_1,sequential_1/flatten_1/strided_slice/stack_2*
end_mask*
ellipsis_mask *

begin_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
T0*
Index0
f
sequential_1/flatten_1/ConstConst*
dtype0*
_output_shapes
:*
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
N*
T0*
_output_shapes
:*

axis 
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
sequential_1/dense_1/BiasAddBiasAddsequential_1/dense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:����������
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
!sequential_1/dropout_2/cond/mul/yConst%^sequential_1/dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
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
seed2ڊ�*
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
$sequential_1/dropout_2/cond/Switch_1Switchsequential_1/activation_3/Relu#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*1
_class'
%#loc:@sequential_1/activation_3/Relu
�
!sequential_1/dropout_2/cond/MergeMerge$sequential_1/dropout_2/cond/Switch_1'sequential_1/dropout_2/cond/dropout/mul**
_output_shapes
:����������: *
N*
T0
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
num_inst/AssignAssignnum_instnum_inst/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@num_inst
a
num_inst/readIdentitynum_inst*
T0*
_output_shapes
: *
_class
loc:@num_inst
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
shape: *
dtype0*
shared_name 
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
Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
div_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
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
T0*
_output_shapes
:*
out_type0
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
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
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
&softmax_cross_entropy_loss/concat/axisConst*
_output_shapes
: *
dtype0*
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
value	B :*
_output_shapes
: *
dtype0
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
value	B : *
dtype0*
_output_shapes
: 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
T0*
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
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
dtype0*
valueB 
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
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
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
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:*
out_type0
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
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
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
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
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
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
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
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
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
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
T0*
_output_shapes
: *H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape
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
9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
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
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
dtype0*
_output_shapes
: *
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
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
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
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
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
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
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
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
_output_shapes
: *
Tshape0*
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
T0*
_output_shapes
: *J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
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
`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
�
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1
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
9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
_output_shapes
:*
out_type0*
T0
�
;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
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
Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
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
gradients/div_1_grad/ShapeShapesequential_1/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
_
gradients/div_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
*
data_formatNHWC*
T0
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
Egradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_2/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_2/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	�
*
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
Igradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencyIdentity:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradB^gradients/sequential_1/dropout_2/cond/Merge_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
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
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*(
_output_shapes
:����������
�
=gradients/sequential_1/dropout_2/cond/Switch_1_grad/cond_gradMergeIgradients/sequential_1/dropout_2/cond/Merge_grad/tuple/control_dependencygradients/zeros**
_output_shapes
:����������: *
N*
T0
�
<gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ShapeShape'sequential_1/dropout_2/cond/dropout/div*
_output_shapes
:*
out_type0*
T0
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
@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_2/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_2/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
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
Qgradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_2/cond/dropout/div_grad/tuple/group_deps*
_output_shapes
: *S
_classI
GEloc:@gradients/sequential_1/dropout_2/cond/dropout/div_grad/Reshape_1*
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
gradients/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
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
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
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
gradients/Switch_2Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@
e
gradients/Shape_3Shapegradients/Switch_2:1*
_output_shapes
:*
out_type0*
T0
\
gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
<gradients/sequential_1/dropout_1/cond/dropout/div_grad/ShapeShapesequential_1/dropout_1/cond/mul*
_output_shapes
:*
out_type0*
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
@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1Reshape<gradients/sequential_1/dropout_1/cond/dropout/div_grad/Sum_1>gradients/sequential_1/dropout_1/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
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
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������@*
T0
�
?gradients/sequential_1/dropout_1/cond/mul/Switch_grad/cond_gradMergeGgradients/sequential_1/dropout_1/cond/mul_grad/tuple/control_dependencygradients/zeros_3*1
_output_shapes
:���������@: *
N*
T0
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
:@*
data_formatNHWC*
T0
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
T0*I
_class?
=;loc:@gradients/sequential_1/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:���������@
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@
T
zeros_2Const*
_output_shapes
:@*
dtype0*
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
zeros_4Const*&
_output_shapes
:@@*
dtype0*%
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
zeros_6Const*
_output_shapes
:@*
dtype0*
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
conv2d_2/bias/Adam/readIdentityconv2d_2/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:@
�
conv2d_2/bias/Adam_1/readIdentityconv2d_2/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
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
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
_output_shapes
:���*
dtype0*
shape:���*
	container *!
_class
loc:@dense_1/kernel*
shared_name 
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
dtype0*
_output_shapes
:	�

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
dtype0*
_output_shapes
:	�
*
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
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_output_shapes
:
*
_class
loc:@dense_2/bias*
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
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
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N"��o_��     �>�	4��dfc�AJ��
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
conv2d_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
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
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
:@*
seed2���*
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
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*&
_output_shapes
:@*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel
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
conv2d_1/convolutionConv2Dconv2d_1_inputconv2d_1/kernel/read*
use_cudnn_on_gpu(*/
_output_shapes
:���������@*
data_formatNHWC*
strides
*
T0*
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
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
seed���)*
T0*
dtype0*&
_output_shapes
:@@*
seed2��z
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
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel*
T0
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
	container *
dtype0*
shared_name *
shape:@
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
conv2d_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*
T0*/
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
dropout_1/cond/mul/SwitchSwitchactivation_2/Reludropout_1/cond/pred_id*J
_output_shapes8
6:���������@:���������@*$
_class
loc:@activation_2/Relu*
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
dropout_1/cond/dropout/ShapeShapedropout_1/cond/mul*
_output_shapes
:*
out_type0*
T0
�
)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
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
seed2��
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
flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
_output_shapes
:*
new_axis_mask *
Index0*
T0
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
 *�3z�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *�3z<
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed���)*
T0*!
_output_shapes
:���*
seed2�ʶ
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
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
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel
~
dense_1/kernel/readIdentitydense_1/kernel*!
_output_shapes
:���*!
_class
loc:@dense_1/kernel*
T0
\
dense_1/ConstConst*
_output_shapes	
:�*
dtype0*
valueB�*    
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
dropout_2/cond/pred_idIdentitydropout_1/keras_learning_phase*
T0
*
_output_shapes
:
s
dropout_2/cond/mul/yConst^dropout_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
 dropout_2/cond/dropout/keep_probConst^dropout_2/cond/switch_t*
_output_shapes
: *
dtype0*
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
seed2ף�*
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
dense_2/random_uniform/minConst*
valueB
 *̈́U�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *̈́U>
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

�
initNoOp^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^conv2d_2/kernel/Assign^conv2d_2/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
�
'sequential_1/conv2d_1/convolution/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
/sequential_1/conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
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
-sequential_1/dropout_1/cond/dropout/keep_probConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *  @?*
dtype0*
_output_shapes
: 
�
)sequential_1/dropout_1/cond/dropout/ShapeShapesequential_1/dropout_1/cond/mul*
T0*
_output_shapes
:*
out_type0
�
6sequential_1/dropout_1/cond/dropout/random_uniform/minConst%^sequential_1/dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6sequential_1/dropout_1/cond/dropout/random_uniform/maxConst%^sequential_1/dropout_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
@sequential_1/dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniform)sequential_1/dropout_1/cond/dropout/Shape*
dtype0*
seed���)*
T0*/
_output_shapes
:���������@*
seed2���
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
$sequential_1/dropout_1/cond/Switch_1Switchsequential_1/activation_2/Relu#sequential_1/dropout_1/cond/pred_id*
T0*J
_output_shapes8
6:���������@:���������@*1
_class'
%#loc:@sequential_1/activation_2/Relu
�
!sequential_1/dropout_1/cond/MergeMerge$sequential_1/dropout_1/cond/Switch_1'sequential_1/dropout_1/cond/dropout/mul*
N*
T0*1
_output_shapes
:���������@: 
}
sequential_1/flatten_1/ShapeShape!sequential_1/dropout_1/cond/Merge*
T0*
_output_shapes
:*
out_type0
t
*sequential_1/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
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
sequential_1/flatten_1/stackPacksequential_1/flatten_1/stack/0sequential_1/flatten_1/Prod*
_output_shapes
:*
N*

axis *
T0
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
:����������*
data_formatNHWC*
T0
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
seed2ڊ�*
dtype0*
T0*
seed���)
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
N*
T0
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
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: 
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
num_correct/AssignAssignnum_correctnum_correct/initial_value*
_class
loc:@num_correct*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
num_correct/readIdentitynum_correct*
_output_shapes
: *
_class
loc:@num_correct*
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
	AssignAdd	AssignAddnum_instConst_1*
use_locking( *
T0*
_class
loc:@num_inst*
_output_shapes
: 
~
AssignAdd_1	AssignAddnum_correctSum*
_output_shapes
: *
_class
loc:@num_correct*
T0*
use_locking( 
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
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
Const_3Const*
dtype0*
_output_shapes
: *
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
 *   A*
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
"softmax_cross_entropy_loss/Shape_1Shapediv_1*
out_type0*
_output_shapes
:*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
�
softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0
�
&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*
_output_shapes
:*
N*

axis *
T0
o
%softmax_cross_entropy_loss/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
�
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_1&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
}
*softmax_cross_entropy_loss/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
h
&softmax_cross_entropy_loss/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
"softmax_cross_entropy_loss/ReshapeReshapediv_1!softmax_cross_entropy_loss/concat*
T0*0
_output_shapes
:������������������*
Tshape0
c
!softmax_cross_entropy_loss/Rank_2Const*
dtype0*
_output_shapes
: *
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
 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 
�
(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*
_output_shapes
:*
N*

axis *
T0
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
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
(softmax_cross_entropy_loss/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
$softmax_cross_entropy_loss/Reshape_1Reshapelabel#softmax_cross_entropy_loss/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
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
'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*
N*
T0*
_output_shapes
:*

axis 
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
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
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
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
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
T0*
_output_shapes
:*
out_type0
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
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
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
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
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
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
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
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1
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
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
�
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*
_output_shapes
: *H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
T0
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
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
out_type0*
_output_shapes
:*
T0
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
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*#
_output_shapes
:���������*
Tshape0*
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
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
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
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

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
gradients/div_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

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
:gradients/sequential_1/dropout_2/cond/Merge_grad/cond_gradSwitchCgradients/sequential_1/dense_2/MatMul_grad/tuple/control_dependency#sequential_1/dropout_2/cond/pred_id*
T0*<
_output_shapes*
(:����������:����������*D
_class:
86loc:@gradients/sequential_1/dense_2/MatMul_grad/MatMul
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
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
out_type0*
T0
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
Ogradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_2/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_2/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_2/cond/dropout/mul_grad/Reshape*(
_output_shapes
:����������*
T0
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
Ggradients/sequential_1/dropout_2/cond/mul_grad/tuple/control_dependencyIdentity6gradients/sequential_1/dropout_2/cond/mul_grad/Reshape@^gradients/sequential_1/dropout_2/cond/mul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*I
_class?
=;loc:@gradients/sequential_1/dropout_2/cond/mul_grad/Reshape
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
gradients/Shape_2Shapegradients/Switch_1*
_output_shapes
:*
out_type0*
T0
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
Cgradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/sequential_1/dense_1/MatMul_grad/MatMul<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*)
_output_shapes
:�����������*D
_class:
86loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul*
T0
�
Egradients/sequential_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/sequential_1/dense_1/MatMul_grad/MatMul_1<^gradients/sequential_1/dense_1/MatMul_grad/tuple/group_deps*
T0*!
_output_shapes
:���*F
_class<
:8loc:@gradients/sequential_1/dense_1/MatMul_grad/MatMul_1
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
Ogradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/control_dependencyIdentity>gradients/sequential_1/dropout_1/cond/dropout/mul_grad/ReshapeH^gradients/sequential_1/dropout_1/cond/dropout/mul_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/sequential_1/dropout_1/cond/dropout/mul_grad/Reshape*/
_output_shapes
:���������@*
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
Qgradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/control_dependency_1Identity@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1H^gradients/sequential_1/dropout_1/cond/dropout/div_grad/tuple/group_deps*S
_classI
GEloc:@gradients/sequential_1/dropout_1/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0
�
4gradients/sequential_1/dropout_1/cond/mul_grad/ShapeShape(sequential_1/dropout_1/cond/mul/Switch:1*
_output_shapes
:*
out_type0*
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
T0*
_output_shapes
:*
out_type0
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
N*
T0
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
8gradients/sequential_1/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/sequential_1/activation_2/Relu_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
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
Dgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput6gradients/sequential_1/conv2d_2/convolution_grad/Shapeconv2d_2/kernel/readEgradients/sequential_1/conv2d_2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(
�
8gradients/sequential_1/conv2d_2/convolution_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
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
Igradients/sequential_1/conv2d_2/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_2/convolution_grad/tuple/group_deps*
T0*/
_output_shapes
:���������@*W
_classM
KIloc:@gradients/sequential_1/conv2d_2/convolution_grad/Conv2DBackpropInput
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
T0*
_output_shapes
:@*K
_classA
?=loc:@gradients/sequential_1/conv2d_1/BiasAdd_grad/BiasAddGrad
z
6gradients/sequential_1/conv2d_1/convolution_grad/ShapeShapedata*
_output_shapes
:*
out_type0*
T0
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
8gradients/sequential_1/conv2d_1/convolution_grad/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"         @   
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
Igradients/sequential_1/conv2d_1/convolution_grad/tuple/control_dependencyIdentityDgradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInputB^gradients/sequential_1/conv2d_1/convolution_grad/tuple/group_deps*/
_output_shapes
:���������*W
_classM
KIloc:@gradients/sequential_1/conv2d_1/convolution_grad/Conv2DBackpropInput*
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*"
_class
loc:@conv2d_1/kernel*
T0*
use_locking(
n
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
j
zerosConst*%
valueB@*    *
dtype0*&
_output_shapes
:@
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
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:@
�
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*&
_output_shapes
:@*"
_class
loc:@conv2d_1/kernel*
T0
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
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
T0
T
zeros_3Const*
_output_shapes
:@*
dtype0*
valueB@*    
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
loc:@conv2d_2/bias*
shared_name *
_output_shapes
:@*
shape:@
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
zeros_8Const* 
valueB���*    *!
_output_shapes
:���*
dtype0
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
zeros_10Const*
_output_shapes	
:�*
dtype0*
valueB�*    
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
valueB	�
*    *
dtype0*
_output_shapes
:	�

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
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*
_class
loc:@dense_2/bias
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
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
_output_shapes
:���*!
_class
loc:@dense_1/kernel
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
	loss/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
c
lossScalarSummary	loss/tags softmax_cross_entropy_loss/value*
_output_shapes
: *
T0
I
Merge/MergeSummaryMergeSummaryloss*
_output_shapes
: *
N""
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
 sequential_1/activation_3/Relu:0&sequential_1/dropout_2/cond/Switch_1:0znБ       ��-	z�efc�A*

loss�`@M80�       ��-	�{efc�A*

losscC@�b�       ��-	U/efc�A*

loss��@���       ��-	��efc�A*

loss)	@ɂj       ��-	��efc�A*

loss��@M�ͪ       ��-	�4 efc�A*

loss�t�?���       ��-	� efc�A*

loss1K�?���V       ��-	��!efc�A*

loss���?s�/       ��-	g)"efc�A	*

loss��?�Xg�       ��-	3#efc�A
*

loss��?�a�       ��-	]�#efc�A*

loss�`y?�E�t       ��-	#g$efc�A*

loss���?�k�C       ��-	%efc�A*

loss$d�?��X�       ��-	u�%efc�A*

lossS��?y;O�       ��-	5}&efc�A*

loss@�o?yR�       ��-	�"'efc�A*

loss�2?�� �       ��-	��'efc�A*

loss/�Y?�Y�j       ��-	�t(efc�A*

loss1uE?V�2�       ��-	)efc�A*

loss��p?X��       ��-	j*efc�A*

loss!_�?J\�       ��-	�+efc�A*

loss�F?�(u�       ��-	W�+efc�A*

loss� ;?z}>3       ��-	F[,efc�A*

loss�	�?�	=       ��-	7�,efc�A*

lossI�S?tUp-       ��-	��-efc�A*

loss��R?IDv       ��-	ӆ.efc�A*

loss��??HK�       ��-	N'/efc�A*

loss��5?���\       ��-	��/efc�A*

lossM�Q?>�j%       ��-	�b0efc�A*

loss)V?;��       ��-	01efc�A*

loss�e?�1��       ��-	O�1efc�A*

loss(�?4y��       ��-	�h2efc�A *

loss� ?��       ��-	3efc�A!*

loss� P??�Y       ��-	��3efc�A"*

lossE?!._       ��-		P4efc�A#*

lossɷ>?La�M       ��-	��4efc�A$*

loss��>�n<N       ��-	#�6efc�A%*

loss�f!?�� '       ��-	�j7efc�A&*

loss�F?3       ��-	�8efc�A'*

lossV=C?ӹ�       ��-	K?9efc�A(*

lossa�+?���       ��-	��9efc�A)*

loss�!�>�(��       ��-	�:efc�A**

loss�W?��`�       ��-	�';efc�A+*

lossJ��>�R/N       ��-	��;efc�A,*

loss"#?M��
       ��-	x^<efc�A-*

lossSu?��       ��-	�<efc�A.*

loss;h?!u0I       ��-	O�=efc�A/*

loss;�?j���       ��-	U3>efc�A0*

loss� ?�f\       ��-	O�>efc�A1*

lossߪ>�1�       ��-	Ll?efc�A2*

loss�/�>2���       ��-	�	@efc�A3*

loss���>�J��       ��-	
�@efc�A4*

loss���>����       ��-	C;Aefc�A5*

loss� A?G�*S       ��-	��Aefc�A6*

loss\A`>���       ��-	jlBefc�A7*

lossc��>|J�       ��-	�Cefc�A8*

loss���>]П�       ��-	��Cefc�A9*

loss($l>e�Ӈ       ��-	NFDefc�A:*

loss7>�>!�o       ��-	7�Defc�A;*

loss��>�9X�       ��-	�|Eefc�A<*

loss	�>Q
��       ��-	]Fefc�A=*

loss�ʼ>� 7q       ��-	n�Fefc�A>*

loss�)�>Bzp�       ��-	�\Gefc�A?*

lossU�> ��Y       ��-	n�Gefc�A@*

loss;�>Q(��       ��-	K�Hefc�AA*

lossJ�>�d�        ��-	f-Iefc�AB*

lossq�?�,w�       ��-	q�Iefc�AC*

loss!S�>M�i�       ��-	=aJefc�AD*

lossS*?5��-       ��-	z�Jefc�AE*

loss)��>]��)       ��-	�Kefc�AF*

loss��q>�O`       ��-	0GLefc�AG*

loss �>�T�       ��-	��Lefc�AH*

loss ��>�6�       ��-	�Mefc�AI*

lossJ�>��*       ��-	z4Nefc�AJ*

loss�{?��M�       ��-	��Nefc�AK*

loss���>��       ��-	��Oefc�AL*

lossB�?U��"       ��-	bPefc�AM*

loss�(?w��       ��-	r�Pefc�AN*

loss��>����       ��-	=�Qefc�AO*

loss,�>w%\A       ��-	36Refc�AP*

loss�<�>�cs       ��-	��Refc�AQ*

loss=�?��c       ��-	;sSefc�AR*

lossJ��>�:�       ��-	�Tefc�AS*

loss��f>Y5�\       ��-	 �Tefc�AT*

loss7��>J�B�       ��-	�MUefc�AU*

loss�Q�>��ؑ       ��-	�Uefc�AV*

lossO�>�K       ��-	�Vefc�AW*

lossv��>�ڳo       ��-	2=Wefc�AX*

losse?�>r�m�       ��-	�Wefc�AY*

loss���>� ��       ��-	��Xefc�AZ*

losst��>#%�&       ��-	�\Yefc�A[*

loss��>♵'       ��-	vZefc�A\*

loss� �>L�^�       ��-	��Zefc�A]*

lossa�>c`       ��-	��[efc�A^*

lossZ�>Z�w�       ��-	�G\efc�A_*

lossF�">��       ��-	��]efc�A`*

loss�@�>1Q��       ��-	��^efc�Aa*

loss3��>��x       ��-	&_efc�Ab*

loss��?��k�       ��-	1�_efc�Ac*

loss��>K�q�       ��-	�J`efc�Ad*

loss-��>(rRm       ��-	a�`efc�Ae*

loss�$>2nf�       ��-	taefc�Af*

loss���>/v�2       ��-	0befc�Ag*

loss�ɀ>�,1       ��-	/�befc�Ah*

loss�R>�IG       ��-	;cefc�Ai*

lossq�>>ឮ       ��-	��cefc�Aj*

loss{�f>ͩ_�       ��-	�fdefc�Ak*

loss�w>>(�x       ��-	C�defc�Al*

loss�-�>/��w       ��-	1�eefc�Am*

loss�֠>�u�       ��-	U/fefc�An*

loss��>�4��       ��-	��fefc�Ao*

loss	�>dR�d       ��-	J`gefc�Ap*

loss1�;>f~t�       ��-	�hefc�Aq*

loss��K>C���       ��-	��hefc�Ar*

loss�j)>m�P�       ��-	l=iefc�As*

loss��8>J3�!       ��-	
�iefc�At*

lossx�V>ʦڼ       ��-	U�jefc�Au*

loss�v�>+�       ��-	&kefc�Av*

loss���>TЄ       ��-	�kefc�Aw*

lossȦ]>�e%       ��-	Ihlefc�Ax*

lossV^X>��H@       ��-	�mefc�Ay*

loss���>���       ��-	�mefc�Az*

loss޾�>~�Y�       ��-	�Tnefc�A{*

lossD��>�q*2       ��-	��nefc�A|*

loss�/>�qW�       ��-	:�oefc�A}*

lossl��>�K#       ��-	�,pefc�A~*

lossO�>g�`~       ��-	�pefc�A*

loss��S>P�w�       �	��qefc�A�*

loss)_�>sd       �	�(refc�A�*

lossmƦ>GŰM       �	��refc�A�*

loss��|>����       �	�tefc�A�*

loss,O7>�g�       �	��tefc�A�*

loss��>?�Bh       �	��uefc�A�*

loss�6�=P*��       �	8�vefc�A�*

lossx'<>6h&�       �	 wefc�A�*

loss}W�>�U*       �	��wefc�A�*

loss;A>`���       �	�Txefc�A�*

lossϺ>�F�'       �	�xefc�A�*

lossz�$>b�       �	b�yefc�A�*

loss&�=�EƼ       �	�Jzefc�A�*

loss@E�=�Ǆs       �	�{efc�A�*

loss]��>K��R       �	��{efc�A�*

loss X�>5|�K       �	v|efc�A�*

loss��i>?��       �	v}efc�A�*

loss�υ>_6�       �	��}efc�A�*

loss
+>-�UG       �	�`~efc�A�*

loss�_>���.       �	�efc�A�*

loss�>0a,       �	�efc�A�*

lossW��=;��       �	�h�efc�A�*

loss�#>�ؿ�       �	��efc�A�*

lossf:�=}��!       �	'��efc�A�*

lossE�z>a�       �	}>�efc�A�*

lossd�>�{-B       �	�؂efc�A�*

loss�|>���       �	z�efc�A�*

lossj�!>�,�       �	�.�efc�A�*

loss�1�=�˔       �	:̈́efc�A�*

loss._>�Bw�       �	�q�efc�A�*

loss֒�>f�0       �	��efc�A�*

loss�ӛ>�f��       �	崆efc�A�*

lossβb>��]�       �	�S�efc�A�*

lossLC�>�{m�       �	���efc�A�*

loss�\>;-a       �	g��efc�A�*

lossb<>9�:       �	WA�efc�A�*

loss���=;Pj2       �	�܉efc�A�*

lossH�>m�Nq       �	�}�efc�A�*

loss.�3>�A�       �	��efc�A�*

lossi�>����       �	���efc�A�*

loss

>�L��       �	�J�efc�A�*

loss�i>�2�       �	���efc�A�*

loss�>Ae       �	~�efc�A�*

loss�C>�i]�       �	�efc�A�*

lossW�=2M�       �	T��efc�A�*

loss�]>Kq�k       �	Џefc�A�*

loss�Q�>�z'       �	"q�efc�A�*

loss�Y�=���m       �	��efc�A�*

loss�|>' ?8       �	U��efc�A�*

loss��?�h�:       �	eP�efc�A�*

loss�M�>�C�!       �	!�efc�A�*

loss�>�l�       �	�{�efc�A�*

lossR�3>��       �	�efc�A�*

loss�J�=-W�       �	��efc�A�*

loss��C>��2�       �	�T�efc�A�*

loss�>��       �	���efc�A�*

lossZ'	>���5       �	٘�efc�A�*

loss��>K�c       �	)@�efc�A�*

lossJ�>gIX�       �	m�efc�A�*

loss�&>}Pn]       �	ڐ�efc�A�*

loss{	Y>�ع�       �	�3�efc�A�*

loss�#*>�d¦       �	�יefc�A�*

loss�T�>�$�       �	���efc�A�*

loss�/>� �       �	�.�efc�A�*

losso�>�ڌ�       �	˛efc�A�*

lossh(�=pd�       �	��efc�A�*

loss��R>#.G0       �	е�efc�A�*

lossc��=]F!�       �	 R�efc�A�*

loss��>��       �	"�efc�A�*

loss��e>��=       �	ٟefc�A�*

loss��\>���e       �	*t�efc�A�*

lossa
e>�N       �	5��efc�A�*

loss��)>�0M       �	��efc�A�*

loss���>X�       �	�g�efc�A�*

loss=�=r�E�       �	$&�efc�A�*

loss�`�=��WG       �	�efc�A�*

loss��*>U�&R       �	�efc�A�*

loss�S2>��pC       �	��efc�A�*

loss��=�u>�       �	>\�efc�A�*

losst�F>^�xc       �	��efc�A�*

loss�'�=����       �	N�efc�A�*

lossuC>½T       �	�ҩefc�A�*

lossZ4>��Bv       �	!��efc�A�*

loss��=�&       �	�8�efc�A�*

loss���=S6�Z       �	k�efc�A�*

loss�v�=j���       �	��efc�A�*

loss�Q�=���|       �	�O�efc�A�*

loss��d>z4�       �	5�efc�A�*

loss�/]>k-(f       �	�0�efc�A�*

loss�b�>��V7       �	n�efc�A�*

loss��>Ɵ��       �	x�efc�A�*

loss-f�>�Z�       �	,�efc�A�*

loss)@�>y7�       �	稱efc�A�*

loss��:=b�q�       �	�@�efc�A�*

loss:]>����       �	�ܲefc�A�*

loss��>��7       �	2v�efc�A�*

lossd�c>��T       �	��efc�A�*

loss�V�=sJ�       �	���efc�A�*

losse��=,�F�       �	2=�efc�A�*

loss��@>׆@�       �	��efc�A�*

loss�>���U       �	 ~�efc�A�*

loss\E1>j�t       �	> �efc�A�*

loss7Y>)��       �	Nѷefc�A�*

lossT-8>�W\.       �	c�efc�A�*

loss�N�=����       �	*�efc�A�*

loss�G�>�!�       �	�¹efc�A�*

loss���>2gT       �	Qk�efc�A�*

loss��>��-        �	_�efc�A�*

loss�Sj>�?�^       �	>��efc�A�*

loss�:J>�9f�       �	�R�efc�A�*

loss���>���       �	3��efc�A�*

loss.�E>���w       �	(��efc�A�*

lossW�=�-O       �	�<�efc�A�*

loss�	>ˢ��       �	�۾efc�A�*

loss��>���l       �	�w�efc�A�*

loss�C�>���9       �	o�efc�A�*

loss!�>��9�       �	̷�efc�A�*

lossX,:>[�dA       �	�\�efc�A�*

loss��|>!c"�       �	)�efc�A�*

loss��$>q<��       �	���efc�A�*

loss�l+>m;<-       �	�J�efc�A�*

loss�hg>�y2       �	���efc�A�*

loss[<^>^C��       �	|��efc�A�*

loss��~>�
�G       �	�7�efc�A�*

loss�3H>[�А       �	b��efc�A�*

loss|�!>��n�       �	}y�efc�A�*

loss���=�.1       �	�efc�A�*

lossE'L>.rq       �	���efc�A�*

loss[�W>)a�       �	�J�efc�A�*

lossTd>c�eA       �	\��efc�A�*

lossD�>l�%       �	��efc�A�*

lossi^�=�+`       �	��efc�A�*

loss� />��Y       �	W��efc�A�*

lossv��=���       �	O�efc�A�*

loss��=?l       �	���efc�A�*

loss��>�H�	       �	̘�efc�A�*

loss��:>�t�       �	T:�efc�A�*

losslD�>3�O       �	v��efc�A�*

loss{�->w�T       �	F~�efc�A�*

loss��*>k'K       �	y!�efc�A�*

loss��6>���       �	<��efc�A�*

loss���>��l       �	�f�efc�A�*

loss�?>�t7W       �	h�efc�A�*

loss�>`�[       �	T��efc�A�*

loss��>:��       �	J�efc�A�*

loss�,>y��       �	���efc�A�*

loss�
>��       �	���efc�A�*

loss�[ >�«�       �	�;�efc�A�*

loss3�>�T�I       �	/��efc�A�*

loss�6>��"Q       �	Xu�efc�A�*

lossaT�=���d       �	��efc�A�*

loss/�>�D|�       �	L��efc�A�*

lossYK>!	C�       �	�B�efc�A�*

loss�V�=ḷ       �	�p�efc�A�*

loss��=�2�       �	$�efc�A�*

loss2>�#4       �	3��efc�A�*

lossR��=L�       �	�=�efc�A�*

loss���>��&       �	��efc�A�*

loss�>@���       �	�q�efc�A�*

loss�U>ռ       �	�
�efc�A�*

loss|8�=�F       �	U��efc�A�*

loss v>!:F       �	��efc�A�*

loss�=��       �	���efc�A�*

loss>�=�h�       �	vl�efc�A�*

lossD5;>��J       �	^�efc�A�*

loss���=���       �	{��efc�A�*

loss��=���~       �	�&�efc�A�*

loss�F�=X%       �	X��efc�A�*

lossݵ�>i�n�       �	;n�efc�A�*

loss��u>�;0�       �	��efc�A�*

loss��F>�|�       �	n��efc�A�*

loss�^>�Z�       �	�\�efc�A�*

lossڐ>���       �	@��efc�A�*

loss�>V�{�       �	Ϥ�efc�A�*

loss��j=����       �	�A�efc�A�*

loss"9>H���       �	w��efc�A�*

loss�cd>���       �	Bx�efc�A�*

loss�(>O��       �	a�efc�A�*

loss��Q>1��       �	���efc�A�*

loss,�=BnD       �	Vf�efc�A�*

lossS_>�IaO       �	��efc�A�*

loss��=�>       �	��efc�A�*

loss�w�=K�"       �	�X�efc�A�*

lossC[>�"��       �	���efc�A�*

loss�EW>/���       �	��efc�A�*

lossh�">��       �		5�efc�A�*

loss�E�=Nd�-       �	���efc�A�*

loss�z�=S[T       �	�m�efc�A�*

loss�ux>}��U       �	��efc�A�*

loss���= v�       �		��efc�A�*

loss��=&��W       �	h@�efc�A�*

loss<�e>���'       �	+��efc�A�*

loss}	�=���N       �	�q�efc�A�*

loss��=g�Aw       �	��efc�A�*

loss�(|>+g#�       �	M��efc�A�*

lossEyM>
|�       �	;6�efc�A�*

loss��	>�=       �	���efc�A�*

loss��e>E%Я       �	Mj�efc�A�*

lossv�>r��       �	��efc�A�*

loss;�=��vy       �	��efc�A�*

lossh�=_�m1       �	�D�efc�A�*

lossckj>Z�iU       �	Q��efc�A�*

loss�q>8�       �	;s�efc�A�*

loss�:>3צ�       �	��efc�A�*

loss1��=� m       �	��efc�A�*

loss㊛=]%�       �	Dk�efc�A�*

lossn��=r@uV       �	��efc�A�*

losso)>�@�       �	���efc�A�*

lossx_=Pd�       �	e6�efc�A�*

loss6�+>V�y�       �	���efc�A�*

loss�G>�|�       �	o�efc�A�*

loss8a
>�M�       �	ɭ�efc�A�*

loss�D=�5.       �	�L ffc�A�*

lossP?>�z(�       �	1� ffc�A�*

loss*0�=���w       �	��ffc�A�*

loss%�'>=�/3       �	*�ffc�A�*

loss7I	>'��       �	ׇffc�A�*

loss�{c>+��       �	&ffc�A�*

lossd>�Y�_       �	6ffc�A�*

loss��=��,�       �	N�ffc�A�*

loss�`R>&�{       �	2Uffc�A�*

loss��<>_�]       �	��ffc�A�*

loss�(>w�Mz       �	��ffc�A�*

loss��=�^��       �	$ffc�A�*

loss��=�'�       �	�	ffc�A�*

lossM �=Q"s       �	д	ffc�A�*

lossq��=~�#       �	2X
ffc�A�*

lossY>����       �	^�
ffc�A�*

losso:=R;�$       �	ѓffc�A�*

loss/>7���       �	U/ffc�A�*

loss&��=a.�s       �	v�ffc�A�*

loss�~=���       �	�]ffc�A�*

lossD)�=�Gf0       �	�xffc�A�*

loss<�=���       �	�ffc�A�*

lossr�>8(n�       �	��ffc�A�*

loss�x=�Z�       �	��ffc�A�*

loss��>�I�       �	ffc�A�*

lossz"E>vb1o       �	J�ffc�A�*

loss��>Yi�       �	]Rffc�A�*

loss�2>Ph+�       �	�ffc�A�*

loss�;�=�V       �	�ffc�A�*

loss$��=�ʔ�       �	�'ffc�A�*

loss�X�=}�       �	n�ffc�A�*

lossqg>���       �	GWffc�A�*

lossN��=��)X       �	��ffc�A�*

loss	�>#wjG       �	�ffc�A�*

loss[.%>q`$       �	�9ffc�A�*

loss�N>�)w       �	��ffc�A�*

loss[�M>n� s       �	ρffc�A�*

loss��=��        �	"ffc�A�*

loss]��=%d��       �	|�ffc�A�*

loss\q�=rFr       �	TSffc�A�*

loss.nU>��ƈ       �	��ffc�A�*

loss�e>9uM       �	�ffc�A�*

loss8*�>�
2�       �	�"ffc�A�*

lossÿ�=W��       �	=�ffc�A�*

lossM��=Kx8y       �	�effc�A�*

loss���=�a=�       �	�~ffc�A�*

loss���=��+�       �	�ffc�A�*

loss�S>�:~�       �	�ffc�A�*

lossdt�=���       �	�L ffc�A�*

loss|C�=��3V       �	�� ffc�A�*

loss�<>��eR       �	L�!ffc�A�*

loss�>>�߀       �	o*"ffc�A�*

loss�0�=`��b       �	J�"ffc�A�*

loss\�#>p	�       �	Dj#ffc�A�*

loss_�(>��;       �	W
$ffc�A�*

loss�E3>�B��       �	��$ffc�A�*

loss��>�|�'       �	�\%ffc�A�*

loss�h=]C*�       �	��%ffc�A�*

loss �=�s0(       �	�&ffc�A�*

lossl��=�=*       �	�G'ffc�A�*

lossLa>w���       �	��'ffc�A�*

loss��->�3�       �	�(ffc�A�*

lossi]>Wǆ�       �	.)ffc�A�*

losssuK>S[��       �	�)ffc�A�*

loss�ۇ=a䑒       �	�Q*ffc�A�*

lossq�=(��       �	c�*ffc�A�*

lossoW$>�L��       �	@�+ffc�A�*

lossD�@>��h       �	_�,ffc�A�*

loss�ʌ>�Vnu       �	:-ffc�A�*

lossT�%>U9�       �	��-ffc�A�*

loss: w>P�B�       �	�m.ffc�A�*

loss �->�r��       �	�/ffc�A�*

loss��>NI�       �	�/ffc�A�*

loss��=����       �	X0ffc�A�*

loss���>��:!       �	L�0ffc�A�*

loss��E>zU_�       �	�1ffc�A�*

loss��E>"�       �	�B2ffc�A�*

lossV��<0:�o       �	/�2ffc�A�*

loss(k�=@��       �	-4ffc�A�*

lossa�'>��@       �	��4ffc�A�*

loss�J>(��       �	g(6ffc�A�*

loss���=Bo-d       �	�`7ffc�A�*

loss$d>F�6k       �	�!8ffc�A�*

loss�Vx=^�i       �	M�8ffc�A�*

loss��>�z9�       �	T�9ffc�A�*

loss�,>�6��       �	�/:ffc�A�*

loss�C�>�0��       �	�X;ffc�A�*

loss���=��	�       �	�
<ffc�A�*

lossE�B>�k6�       �	��<ffc�A�*

loss�r�=����       �	&U=ffc�A�*

loss/N=2��Y       �	�=ffc�A�*

lossq�
>��       �	�>ffc�A�*

loss�;�=�z�       �	��?ffc�A�*

loss=��=�ol�       �	]@ffc�A�*

loss	�>?�6{       �	��@ffc�A�*

loss�I>���       �	�Affc�A�*

lossa�s=U�       �	<Bffc�A�*

loss��= �x+       �	��Bffc�A�*

loss=�+=�Ƅl       �	g|Cffc�A�*

loss��`=��       �	k�Dffc�A�*

loss>��=��m       �	|�Effc�A�*

lossm��>�2~       �	�Fffc�A�*

loss�J>��_�       �	l_Gffc�A�*

loss�d�=�S�       �	& Hffc�A�*

lossS�>�k��       �	{�Hffc�A�*

loss��<FL       �	�<Iffc�A�*

loss$�">�e�       �	��Iffc�A�*

losso2�>��ݬ       �	�{Jffc�A�*

loss�(%>l0z       �	Kffc�A�*

loss���=�3Hv       �	��Kffc�A�*

losslA?>*�f�       �	�`Lffc�A�*

loss�):>c��       �	~ Mffc�A�*

loss�1�=��*       �	L�Mffc�A�*

lossa��=x�J0       �	�BNffc�A�*

loss��>5��       �	r�Nffc�A�*

loss��S>�~�|       �	�}Offc�A�*

loss�D>��V       �	{Pffc�A�*

loss�u>8�vl       �	l�Pffc�A�*

loss%�=��@       �	LQffc�A�*

loss�/>��ܮ       �	J�Qffc�A�*

loss>%
>-�       �	j�Rffc�A�*

loss!�= p�       �	S$Sffc�A�*

loss-qH=�N\       �	�Sffc�A�*

loss��>K��       �	�lTffc�A�*

losse�&>rs`^       �	�Uffc�A�*

loss�%�=("�       �	�Uffc�A�*

loss7KQ=��B�       �	�YVffc�A�*

loss�?=��y       �	��Vffc�A�*

loss�O=��3       �	�Wffc�A�*

loss�J
>�+       �	�7Xffc�A�*

loss�}4=�>�       �	I�Xffc�A�*

lossV�>`���       �	4�Yffc�A�*

lossW�>��]L       �	�1Zffc�A�*

lossc>�"Z       �	��Zffc�A�*

loss��>ψ�[       �	w[ffc�A�*

loss���=�~�Y       �	P\ffc�A�*

loss�^B=��&#       �	�\ffc�A�*

loss���=�5�       �	�/^ffc�A�*

loss���=��7�       �	�_ffc�A�*

loss;�=M��       �	d�_ffc�A�*

lossT
>�^�       �	#L`ffc�A�*

loss�6�>��       �	1�`ffc�A�*

loss�c�=�C�       �	��affc�A�*

lossse�=����       �	p#bffc�A�*

loss=7>~��;       �	��bffc�A�*

lossї�=�!j�       �	.qcffc�A�*

loss�=ǣcE       �	Fdffc�A�*

loss��R>6�g�       �	��dffc�A�*

loss��I>	*��       �	�Geffc�A�*

loss8�=�� w       �	��effc�A�*

loss�$>'�7h       �	��fffc�A�*

loss!2>b�98       �	�gffc�A�*

loss��f>�9R#       �	��hffc�A�*

loss��_<d��]       �	}Yiffc�A�*

lossqg�=a�       �	R�iffc�A�*

loss��K==���       �		�jffc�A�*

loss��l=�6��       �	p"kffc�A�*

loss�/�=ϫfC       �	Qlffc�A�*

loss�@|=��=       �	k�lffc�A�*

lossw��=�E��       �	�Xmffc�A�*

loss�s>X��       �	��mffc�A�*

loss(:�=>M��       �	}�nffc�A�*

lossXY�=�u�<       �	P4offc�A�*

loss��e=��Ԓ       �	�offc�A�*

loss�F�=>�a       �	jpffc�A�*

loss��
>Hm��       �	;qffc�A�*

loss�Σ=�U�       �	U�qffc�A�*

loss�($>͛�       �	.Trffc�A�*

loss<.�=C۞�       �	��rffc�A�*

loss�)<>e��       �	A�sffc�A�*

loss���=�ݦ�       �	luffc�A�*

lossz��=�%"�       �	!:vffc�A�*

loss�0�=ZC��       �	.pwffc�A�*

loss_Z=K�"|       �	�xffc�A�*

lossD��=ݡ-       �	�xffc�A�*

loss375=��i       �	�[yffc�A�*

loss��>��ђ       �	Z�yffc�A�*

loss�>����       �	y�zffc�A�*

loss���=/RO�       �	�/{ffc�A�*

loss��=��g       �	�{ffc�A�*

loss���=�o       �	Eh|ffc�A�*

loss�2�=���       �	q}ffc�A�*

loss1�>$mF       �	&�}ffc�A�*

lossm�{<̧�s       �	�g~ffc�A�*

loss�sM=[��       �	9_ffc�A�*

loss8)�=�ׄ�       �	��ffc�A�*

lossL�=SYZv       �	���ffc�A�*

lossxI�=��8�       �	J�ffc�A�*

lossQQ�=b�       �	f��ffc�A�*

lossA]>K��`       �	d�ffc�A�*

loss�%o=��       �	���ffc�A�*

loss��Y=��5       �	�W�ffc�A�*

lossU�=l���       �	��ffc�A�*

loss��>=Sm��       �	h��ffc�A�*

loss1H=#��       �	;5�ffc�A�*

loss�@�<17�t       �	'܆ffc�A�*

loss�X=��Kr       �	���ffc�A�*

loss��'=	>��       �	�Y�ffc�A�*

loss̀�<&       �	��ffc�A�*

lossių<�@=v       �	���ffc�A�*

loss�>?:��       �	���ffc�A�*

loss�b<�!�       �	�8�ffc�A�*

loss_�
<	��       �	�ߋffc�A�*

loss�jC<���       �	Q��ffc�A�*

loss�X�<u��       �	R'�ffc�A�*

loss��=��1       �	Ǎffc�A�*

lossQ3=�jJb       �	�n�ffc�A�*

lossNU{;���u       �	
�ffc�A�*

loss��=GP�       �	,��ffc�A�*

losss�?����       �	qY�ffc�A�*

lossu)�;&��       �	��ffc�A�*

loss��#>]R#       �	���ffc�A�*

loss_�>	ܡ�       �	'�ffc�A�*

loss2_V>=��+       �	��ffc�A�*

loss/�=�-s       �	Q�ffc�A�*

loss�`�<���       �	��ffc�A�*

lossT�=�\�:       �	��ffc�A�*

loss1]~=QC��       �	��ffc�A�*

loss��F>��!       �	��ffc�A�*

lossj��=R��       �	+P�ffc�A�*

loss"l>��_       �	��ffc�A�*

losssϔ>�&�"       �	8��ffc�A�*

loss^,�>+#�       �	��ffc�A�*

lossׯ�=�Ut       �	F��ffc�A�*

loss<�Q>J[k       �	�T�ffc�A�*

losssF1>���A       �	��ffc�A�*

loss���=r�>       �	@��ffc�A�*

loss�+ >w�Ҫ       �	�"�ffc�A�*

lossF�=�cy        �	�ěffc�A�*

loss�=��       �	�d�ffc�A�*

lossͱ}=a��       �	��ffc�A�*

lossZ�=o�O       �	k��ffc�A�*

loss��V>c���       �	@4�ffc�A�*

loss�?=WW?        �	�A�ffc�A�*

lossRm�=�yT}       �	��ffc�A�*

loss�jt=��       �	��ffc�A�*

loss���=<�m       �	}�ffc�A�*

loss�z=_�o?       �	%\�ffc�A�*

loss\�e>��@       �	Wz�ffc�A�*

lossc�=�l8�       �	9F�ffc�A�*

lossc�=�?�       �	��ffc�A�*

loss�հ=�wzz       �	Y��ffc�A�*

loss���=��r�       �	�Y�ffc�A�*

loss&K�<a���       �	���ffc�A�*

lossZ��=�8w
       �	'��ffc�A�*

loss�A=��0�       �	���ffc�A�*

loss���<p��       �	,�ffc�A�*

loss&��=Q���       �	�ͩffc�A�*

loss��!>�td       �	rj�ffc�A�*

loss\X >@'/�       �	%�ffc�A�*

loss��=�`)       �	���ffc�A�*

loss{�=-�l�       �	AE�ffc�A�*

loss�>B��       �	߬ffc�A�*

lossO1�=CD�       �	�~�ffc�A�*

lossQ�=1G5�       �	!!�ffc�A�*

loss��=���       �	8,�ffc�A�*

loss�Q>RI�       �	ůffc�A�*

loss�*=�}       �	�`�ffc�A�*

loss���=̛$       �	��ffc�A�*

loss��=�S�       �	���ffc�A�*

loss�{�=�V=       �	v3�ffc�A�*

loss��>�|i�       �	�:�ffc�A�*

loss���=�x��       �	���ffc�A�*

loss�:>d��H       �	���ffc�A�*

lossOF�=�=�       �	e�ffc�A�*

lossW�>�+i?       �	���ffc�A�*

lossi=>�R�'       �	�f�ffc�A�*

loss��=U:=&       �	� �ffc�A�*

losse�*>��       �	���ffc�A�*

loss�>��w=       �	@�ffc�A�*

loss*;>NVLM       �	���ffc�A�*

loss<��=}(��       �	�|�ffc�A�*

loss�`=�&e       �	��ffc�A�*

loss���==��       �	_��ffc�A�*

loss���=���       �	�g�ffc�A�*

loss���=9���       �	p�ffc�A�*

loss:�>:��m       �	ʦ�ffc�A�*

loss��y=��       �	�J�ffc�A�*

loss<�=MP^       �	1��ffc�A�*

lossn��=�O�       �	;��ffc�A�*

lossS��>��+       �	� �ffc�A�*

lossd�?=�l*�       �	���ffc�A�*

loss��0>��0�       �	kc�ffc�A�*

loss���=�:��       �	a��ffc�A�*

lossO�R>u]#p       �	���ffc�A�*

loss�>�A       �	o-�ffc�A�*

loss��<=;o�       �	
��ffc�A�*

loss�E�=3	h�       �	���ffc�A�*

loss8*n=%àJ       �	y@�ffc�A�*

loss@&>�$��       �	xF�ffc�A�*

lossz�*>?Y4�       �	���ffc�A�*

loss���=t<�       �	���ffc�A�*

lossӱ�=�       �	�l�ffc�A�*

loss�J�=�~Y       �	�x�ffc�A�*

loss{�>�l�S       �	U��ffc�A�*

loss汯=��j�       �	 ��ffc�A�*

lossQk>��       �	ӆ�ffc�A�*

losse�o=~[ҝ       �	�]�ffc�A�*

lossW�D>4aH�       �	T�ffc�A�*

loss���>E��       �	>�ffc�A�*

lossx�$>۟��       �	v��ffc�A�*

loss��1>�5P�       �	1��ffc�A�*

lossN��=o�bV       �	�O�ffc�A�*

loss�˵=+�s�       �	��ffc�A�*

loss��<>��       �	Ֆ�ffc�A�*

lossΑ�='�z       �	75�ffc�A�*

loss�>��       �	���ffc�A�*

lossj�=���W       �	���ffc�A�*

lossXx�=Ё{�       �	�%�ffc�A�*

loss��=x��	       �	���ffc�A�*

loss}\"=�#{9       �	n�ffc�A�*

lossZ�J=����       �	�ffc�A�*

loss{�3>�?�       �	��ffc�A�*

loss�Ү=E�H�       �	�W�ffc�A�*

loss�"?�W�q       �	5(�ffc�A�*

loss��=MhW3       �	���ffc�A�*

loss��<       �	�s�ffc�A�*

loss6�<r2ja       �	#�ffc�A�*

lossv�=M�z�       �	T��ffc�A�*

loss���=��s0       �	8I�ffc�A�*

loss��>Iɷ�       �	��ffc�A�*

loss�<>��        �	7��ffc�A�*

loss���=�?�       �	�s�ffc�A�*

loss@�=�4+       �	��ffc�A�*

lossE�>sW�       �	+��ffc�A�*

loss�Z>cμa       �	S=�ffc�A�*

lossء=�       �	[��ffc�A�*

loss��=|�V�       �	q��ffc�A�*

loss*$>⸕`       �	k,�ffc�A�*

loss7�I>c�)B       �	���ffc�A�*

lossX)(>�Ǩ�       �	�h�ffc�A�*

loss��=�N�       �	�ffc�A�*

loss9u>���T       �	.��ffc�A�*

loss(��=!P]       �	~S�ffc�A�*

loss��=���       �	8��ffc�A�*

lossyZ�=׶�       �	ѕ gfc�A�*

lossEh�=E���       �	�2gfc�A�*

loss]�M>.��
       �	��gfc�A�*

loss�*->o_+�       �	�gfc�A�*

loss��u=ؙ=       �	% gfc�A�*

losso�U=�4�       �	Z�gfc�A�*

loss0�=Ӎ�_       �	?Wgfc�A�*

loss�$>���       �	��gfc�A�*

loss�>�K�       �	0�gfc�A�*

loss�!=��[U       �	?:gfc�A�*

loss��=.�~       �	�gfc�A�*

loss.[�=4W�u       �	rgfc�A�*

losssU>(�X�       �	�gfc�A�*

loss8/�=�u_"       �	W�gfc�A�*

loss��=ٱ��       �	}Y	gfc�A�*

loss�!>��z       �	� 
gfc�A�*

loss
�=ʈ��       �	��
gfc�A�*

lossss�=���       �	�Fgfc�A�*

loss��>=6�       �	��gfc�A�*

loss
>>��c�       �	�zgfc�A�*

lossX�R>\�       �	Tgfc�A�*

lossaH=4��       �	�gfc�A�*

loss�-�=�TkV       �	gcgfc�A�*

loss�^�=I�0�       �	�gfc�A�*

lossr�|>H�       �	ǝgfc�A�*

loss�-�=����       �	�6gfc�A�*

loss�*={��V       �	(�gfc�A�*

lossr)F=�7       �	�kgfc�A�*

losst2>mE[       �	�gfc�A�*

loss�ž=yb��       �	�gfc�A�*

loss�Z�=�)��       �	|Bgfc�A�*

loss\}�>��p�       �	��gfc�A�*

lossG�<�b5       �	I�gfc�A�*

loss��=j���       �	�;gfc�A�*

loss�}->�o�       �	�gfc�A�*

loss�J�=Si{	       �	'jgfc�A�*

lossf�%>n3`�       �	ugfc�A�*

loss ��=��9�       �	Йgfc�A�*

loss�Q�=u⟜       �	.9gfc�A�*

losscD=^�]       �	��gfc�A�*

lossX�#=Ӎh�       �	Iegfc�A�*

loss[Q�=g�y       �	a�gfc�A�*

lossi=��/       �	�gfc�A�*

loss���=��I�       �	�*gfc�A�*

lossi�'>��J�       �	��gfc�A�*

loss\�i=�U�       �	�]gfc�A�*

loss[��=�>�0       �	Z�gfc�A�*

loss�7=���       �	�gfc�A�*

loss ]�=��]�       �	�.gfc�A�*

loss���=6�       �	> gfc�A�*

lossLTG<���G       �	�gfc�A�*

loss8��=d#       �	� gfc�A�*

loss�4�=�d��       �	fh!gfc�A�*

loss���=H��@       �	ҏ"gfc�A�*

loss�(0>���l       �	 y#gfc�A�*

loss�.>���i       �	�$gfc�A�*

lossc�=��nr       �	Ĳ%gfc�A�*

loss��>f��Q       �	Ҭ&gfc�A�*

loss}0�=d۞l       �	|�'gfc�A�*

loss�_=F_��       �	�(gfc�A�*

loss/�	>+�#v       �	O�)gfc�A�*

loss
]g=��6       �	��*gfc�A�*

loss�w�=���;       �	\q+gfc�A�*

loss��m=j�޶       �	*,gfc�A�*

loss�3�=;�!	       �	:�-gfc�A�*

loss�Jf=Nξ�       �	JB.gfc�A�*

loss�_U<u�1       �	��.gfc�A�*

loss7�j=��       �	0�/gfc�A�*

loss/�=>A&H       �	�)0gfc�A�*

loss��->�ھ�       �	��0gfc�A�*

loss��=h|��       �	�a1gfc�A�*

loss]��=�w��       �	J2gfc�A�*

loss?0�=#N �       �	�2gfc�A�*

loss?"=x?��       �	*U3gfc�A�*

lossZ��<
+��       �	D�3gfc�A�*

loss81�=}�7�       �	��4gfc�A�*

loss�<U��D       �	�?5gfc�A�*

lossC��=::�       �	8�5gfc�A�*

lossq�J>j =&       �	�6gfc�A�*

lossv�>�P|       �	�b7gfc�A�*

loss�D�=�Zu       �	��7gfc�A�*

loss�<؇��       �	��8gfc�A�*

loss�*=���i       �	uw9gfc�A�*

losst >�ޔ�       �	�:gfc�A�*

loss�Z�=����       �	��:gfc�A�*

loss$Ӑ=]�       �	�b;gfc�A�*

loss�R�=:S�<       �	��;gfc�A�*

loss�F�=z�"       �	[�<gfc�A�*

loss C�=�͍�       �	�'=gfc�A�*

loss�)1<��       �	I�=gfc�A�*

loss�l�=���       �	�L>gfc�A�*

lossϨ�=� "&       �	�_?gfc�A�*

loss��=���       �	F�?gfc�A�*

lossVv>���       �	��@gfc�A�*

loss�=�!ga       �	H7Agfc�A�*

loss�<�=pw�       �	6�Agfc�A�*

loss�̷=)qIO       �	�`Bgfc�A�*

loss&t�<�*�?       �	%Cgfc�A�*

loss۬~=��!�       �	�Cgfc�A�*

loss�+>Q�	       �	fDgfc�A�*

lossW<[=����       �	/�Dgfc�A�*

loss��:=޴EJ       �	��Egfc�A�*

loss�>��Ʉ       �	~nFgfc�A�*

loss���=Z���       �	�Ggfc�A�*

lossR��=,��o       �	�Ggfc�A�*

loss�L�< �Z�       �	%AHgfc�A�*

lossHl�=�bl�       �	��Hgfc�A�*

loss�� =�̤�       �	�Igfc�A�*

lossR,�=Bv��       �	 Jgfc�A�*

lossq2�<���       �	��Jgfc�A�*

loss���<����       �	�QKgfc�A�*

loss$��=�N$�       �	��Kgfc�A�*

loss>��0�       �	��Lgfc�A�*

lossFW�=qIߙ       �	B!Mgfc�A�*

loss.��=�l�       �	3�Mgfc�A�*

loss2��=L.h�       �	�ZNgfc�A�*

loss6�=�R       �	��Ngfc�A�*

lossZ��=xк�       �	՗Ogfc�A�*

loss,=0K�       �	o,Pgfc�A�*

lossD�H=a$F\       �	��Pgfc�A�*

loss�@=DQ%       �	![Qgfc�A�*

loss��=��}�       �	��Qgfc�A�*

losse�=o��       �	O�Rgfc�A�*

loss1�=q�_�       �	�-Sgfc�A�*

loss���=�I�        �	�Sgfc�A�*

loss�t�=����       �	Y�Tgfc�A�*

loss�V�=3�+       �	6"Ugfc�A�*

loss��2=�"��       �	��Ugfc�A�*

loss�U�=#f��       �	�`Vgfc�A�*

loss|l$=�d�       �	��Vgfc�A�*

loss�`3=�X.�       �	��Wgfc�A�*

loss.��='f�       �	�IXgfc�A�*

loss�
>m��       �	��Xgfc�A�*

loss@=W��x       �	��Ygfc�A�*

loss�8�=��BF       �	5@Zgfc�A�*

loss��=�|}       �	�Zgfc�A�*

loss,8
=VE��       �	�[gfc�A�*

lossr3�=!���       �	�;\gfc�A�*

loss��d=����       �	��\gfc�A�*

loss��4=�k�2       �	��]gfc�A�*

lossrN�=1�Ok       �	�7^gfc�A�*

loss5o�= �'       �	/�^gfc�A�*

lossJ�4>Ie5�       �	R�_gfc�A�*

loss��>Af	�       �	�w`gfc�A�*

loss�>�y�       �	3�agfc�A�*

loss��=�+:       �	ٔbgfc�A�*

loss�J=�&�       �	m�cgfc�A�*

lossʫ�=W��       �	��dgfc�A�*

loss��=�jj�       �	ƅegfc�A�*

loss�}�=Y       �	�?fgfc�A�*

loss�]�<)6Vr       �	��fgfc�A�*

loss�gd=���       �	'hgfc�A�*

lossX[�=x�:M       �	ɮhgfc�A�*

lossJ��=k�fu       �	�Nigfc�A�*

loss� �=�v�~       �	#�igfc�A�*

loss��=��}       �	��jgfc�A�*

loss��H=*[�       �	�Dkgfc�A�*

loss �=(s�       �	P�kgfc�A�*

loss�z�=�2w�       �	�lgfc�A�*

loss�=�=�5��       �	8-mgfc�A�*

lossX%>@��A       �	��mgfc�A�*

lossߍ�=�I�2       �	��ngfc�A�*

loss�`�<c��       �	A(ogfc�A�*

loss���=�b�       �	��ogfc�A�*

loss�x�=�F       �	ipgfc�A�*

loss2P/=o�w6       �	�qgfc�A�*

loss��Z=��^\       �	{�qgfc�A�*

loss/D�=���^       �	5@rgfc�A�*

loss��=-��       �	��rgfc�A�*

loss��=diam       �	Szsgfc�A�*

lossh��="�\�       �	tgfc�A�*

loss�Ĭ=b,�       �	̳tgfc�A�*

lossal=K�n�       �	�Uugfc�A�*

loss	շ=r��       �	��ugfc�A�*

loss�k�=�d}�       �	)�vgfc�A�*

loss�ؒ=�Ҏ�       �	o/wgfc�A�*

loss���=Di�Y       �	�wgfc�A�*

loss��=���       �	��xgfc�A�*

loss���=q�H�       �	t#ygfc�A�*

loss
^<,��       �	Ǻygfc�A�*

loss��5=6�(�       �	|zgfc�A�*

losspg=j~       �	z{gfc�A�*

lossl��=�{!�       �	ͭ{gfc�A�*

loss�y>��       �	G�|gfc�A�*

loss���=�Y7�       �	�4}gfc�A�*

loss��r=v@�       �	N�}gfc�A�*

loss�N�=���       �	t~gfc�A�*

losse$�<��[�       �	gfc�A�*

loss��>O��       �	�gfc�A�*

lossq�=���m       �	N�gfc�A�*

loss���=�r'�       �	��gfc�A�*

loss��?=iv��       �	��gfc�A�*

loss�$C>ғ'�       �	�1�gfc�A�*

loss*y�=p]��       �	��gfc�A�*

lossZ�a>���       �	���gfc�A�*

loss��=��Ph       �	�M�gfc�A�*

loss���=*t�       �	���gfc�A�*

loss��w=p�^�       �	���gfc�A�*

lossq�^=�I��       �	2<�gfc�A�*

loss��>y�;�       �	6�gfc�A�*

loss���=죩�       �	돇gfc�A�*

loss�h>����       �	gfc�A�*

loss~�=;���       �	z9�gfc�A�*

loss�+=����       �	L�gfc�A�*

loss�z>a���       �		��gfc�A�*

loss��=��R	       �	�[�gfc�A�*

loss�.<zO��       �	���gfc�A�*

lossM=�qs�       �	p��gfc�A�*

loss� >�3�       �	�4�gfc�A�*

loss�#<�V       �	|Սgfc�A�*

loss��>۽��       �	|�gfc�A�*

loss�OD>
��a       �	"�gfc�A�*

loss7��=@IǷ       �	�͏gfc�A�*

loss؂{=���       �	�s�gfc�A�*

loss3`�=bp�       �	3�gfc�A�*

loss��<1��_       �	��gfc�A�*

loss�=C͗4       �	ʌ�gfc�A�*

loss���=��       �	�%�gfc�A�*

loss�	�<��`~       �	'��gfc�A�*

loss}��<�=<�       �	F^�gfc�A�*

loss��<8�y�       �	���gfc�A�*

loss�r�=���9       �	A��gfc�A�*

loss*�|=_V6       �	`;�gfc�A�*

loss��|>��s       �	�Ֆgfc�A�*

loss�=S"��       �	vo�gfc�A�*

loss	s">z�B       �	�	�gfc�A�*

loss4ZO=颬�       �	��gfc�A�*

loss��a=s��D       �	�Q�gfc�A�*

loss ��=�]�       �	���gfc�A�*

loss��=��       �	��gfc�A�*

lossC��=X��^       �	�3�gfc�A�*

loss|�L>S���       �	ޛgfc�A�*

loss��L<�0       �	p}�gfc�A�*

loss�E>����       �	��gfc�A�*

losst�=8O�{       �	}��gfc�A�*

loss��_=0�4�       �	$F�gfc�A�*

loss8N�=	�}       �	�ߞgfc�A�*

loss��]=( v�       �	܁�gfc�A�*

loss�R�=(�T       �	�gfc�A�*

lossw�N=V'*>       �	���gfc�A�*

loss��c=9�-
       �	i�gfc�A�*

loss���=��       �	� �gfc�A�*

lossب=�h�6       �	k��gfc�A�*

loss���<����       �	C;�gfc�A�*

loss*zw=�N       �	ۣgfc�A�*

loss���=��       �	���gfc�A�*

loss�Z
=Xc�       �	�(�gfc�A�*

lossi�G>�u        �	�Хgfc�A�*

loss.E>��d�       �	an�gfc�A�*

loss��!=�@��       �	d!�gfc�A�*

loss���=�k       �	���gfc�A�*

loss�s�=�(��       �	&W�gfc�A�*

loss��/=�ȼ�       �	��gfc�A�*

loss3��<�at       �	m��gfc�A�*

loss���=؞}�       �	t��gfc�A�*

loss�b >�6J�       �	31�gfc�A�*

loss��=��O�       �	٫gfc�A�*

loss�Ka=b�Y       �	���gfc�A�*

loss�0=�;Z�       �	� �gfc�A�*

losssO=l��&       �	��gfc�A�*

loss��>6�;       �	�Y�gfc�A�*

loss��=.��Y       �	��gfc�A�*

loss�U=�[��       �	̴�gfc�A�*

loss��>�'ɬ       �	�a�gfc�A�*

loss� =^��       �	R
�gfc�A�*

losst�<d@�       �	���gfc�A�*

lossT'=�KX�       �	3R�gfc�A�*

loss|R�<�Kb       �	���gfc�A�*

loss@Ƀ=ɾ��       �	���gfc�A�*

lossD�=�籘       �	rO�gfc�A�*

loss �_>��p�       �	!�gfc�A�*

loss�`9>�rka       �	x��gfc�A�*

loss@�==h!�       �	�d�gfc�A�*

lossHE�=n�'       �	��gfc�A�*

loss�?�=�4t�       �	 W�gfc�A�*

loss��=���@       �	��gfc�A�*

loss�=�ˡ�       �	O�gfc�A�*

loss,�i=�ɋ       �	���gfc�A�*

lossZ�[=�U�       �	�E�gfc�A�*

loss��]=�M�^       �	��gfc�A�*

loss@�=Yd�E       �	䇼gfc�A�*

loss�=)lu�       �	(�gfc�A�*

loss�'9<w!n�       �	�нgfc�A�*

loss�
T=V��y       �	��gfc�A�*

lossW��=V2��       �	˿gfc�A�*

loss���=Ȥ       �	�i�gfc�A�*

loss��=��!�       �	B�gfc�A�*

loss��<Ha�       �	^��gfc�A�*

loss�Ӧ<-t�.       �	MH�gfc�A�*

losst�R>߭�4       �	���gfc�A�*

loss��9>��j9       �	w��gfc�A�*

loss�(*=ʴ��       �	�=�gfc�A�*

loss!Ѹ=xYQn       �	���gfc�A�*

lossRְ=����       �	�y�gfc�A�*

loss�CI=O��       �	4�gfc�A�*

lossF!�=��V�       �	��gfc�A�*

lossEM�=���<       �	 ;�gfc�A�*

loss�vp=��?       �	���gfc�A�*

loss�%�=���       �	�m�gfc�A�*

lossc��=���       �	��gfc�A�*

loss�{�=���L       �	6��gfc�A�*

lossA��=�mU�       �	�T�gfc�A�*

loss��>�y`C       �	��gfc�A�*

loss`А=��v<       �	���gfc�A�*

loss]��=ӎ��       �	W&�gfc�A�*

lossX(>���       �	��gfc�A�*

loss��=7�J�       �	�t�gfc�A�*

loss�6	>[0       �	��gfc�A�*

loss�E�=��n�       �	���gfc�A�*

loss��`=���8       �	�Q�gfc�A�*

loss���<�UcT       �	���gfc�A�*

lossY�=�<�       �	���gfc�A�*

lossJ��=��^       �	%!�gfc�A�*

loss���=@��       �	��gfc�A�*

loss�[J=9Z�       �	i��gfc�A�*

loss��=xo��       �	g_�gfc�A�*

loss�>�?0c       �	r��gfc�A�*

loss�݂=+Ӹ       �	���gfc�A�*

loss���=�bUo       �	�.�gfc�A�*

loss-��=�s��       �	���gfc�A�*

lossh�p=�S?[       �	�p�gfc�A�*

loss="=>�6       �	f�gfc�A�*

loss�G=� �       �	��gfc�A�*

loss�G{=�� 7       �	Q�gfc�A�*

loss�6�<����       �	0��gfc�A�*

loss��=b��       �	���gfc�A�*

lossOd8=���       �	$�gfc�A�*

loss���=6�+|       �	���gfc�A�*

loss�->��       �	���gfc�A�*

lossz�<�$z       �	���gfc�A�*

loss�-=k1ܦ       �	 )�gfc�A�*

losstc�=�R��       �	*��gfc�A�*

loss��=s�-       �	�z�gfc�A�*

loss�{�=uN��       �	��gfc�A�*

loss�8�=�GZQ       �	���gfc�A�*

loss�>H�V       �	6t�gfc�A�*

lossl�=d5O!       �	M�gfc�A�*

loss���=Es�       �	
+�gfc�A�*

loss�nf=W�       �	���gfc�A�*

loss�`�>�9ao       �	���gfc�A�*

lossXA�=���       �	i��gfc�A�*

loss/�>&\�?       �	k��gfc�A�*

loss�N�<1Cb       �	À�gfc�A�*

loss
�V=�v֝       �	cB�gfc�A�*

loss{��=�ܠ�       �	���gfc�A�*

loss]�j=�}��       �	���gfc�A�*

lossM�4=#��       �	=�gfc�A�*

loss7�=���N       �	��gfc�A�*

lossY��=]�v       �	6��gfc�A�*

loss%�=�ĴY       �	(G�gfc�A�*

loss�a�=��8o       �	���gfc�A�*

lossht]>U���       �	v��gfc�A�*

loss�^�=��G�       �	�O�gfc�A�*

loss��=��       �	���gfc�A�*

loss�f=K\�T       �	N��gfc�A�*

lossCGm=���r       �	$C�gfc�A�*

loss:��=,��       �	���gfc�A�*

loss�u�=��	       �	|�gfc�A�*

loss��=���       �	�gfc�A�*

lossL�=dvW       �	���gfc�A�*

loss�\=$W��       �	�U�gfc�A�*

loss
�/=1���       �	��gfc�A�*

loss�|=�B�       �	z��gfc�A�*

lossZˮ<�i6       �	�)�gfc�A�*

loss�ku<z^d�       �	?��gfc�A�*

losssK=Q^       �	;s�gfc�A�*

loss�i�>V'�}       �	t�gfc�A�*

loss�g >��b�       �	���gfc�A�*

loss훠=��       �	�Y�gfc�A�*

loss@��=|N       �	H��gfc�A�*

losst?�<���:       �	Z��gfc�A�*

lossc{=TXY*       �	;�gfc�A�*

loss$�u>Mt!�       �	���gfc�A�*

loss��`=v��       �	
j�gfc�A�*

losso>C=�� �       �	��gfc�A�*

lossO8>[5       �	\<�gfc�A�*

loss���=�ŭC       �	��gfc�A�*

lossȻ�<~���       �	z�gfc�A�*

loss��7=�J��       �	� hfc�A�*

loss�>�E�       �	�� hfc�A�*

loss �=G<�       �	�Zhfc�A�*

loss�5s=���_       �	z�hfc�A�*

loss�>h��       �	��hfc�A�*

loss��>=Ù�+       �	��hfc�A�*

loss��i=S$�       �	�[hfc�A�*

loss���=f_H       �	��hfc�A�*

lossl2�<�;��       �	l�hfc�A�*

loss��<~�Z       �	:hfc�A�*

lossؽ�<V~t�       �	[�hfc�A�*

loss	>�1g�       �	?ohfc�A�*

lossJ=I)~�       �	�hfc�A�*

loss�<��       �	�hfc�A�*

lossÃO=��7�       �	�E	hfc�A�*

loss��4=���2       �	��	hfc�A�*

loss=�>VuP�       �	J|
hfc�A�*

loss�9�;�v��       �	�hfc�A�*

lossRt%>OÙK       �	��hfc�A�*

loss��=��{�       �	t`hfc�A�*

loss,��=.�_       �	`hfc�A�*

loss��>��A�       �	s�hfc�A�*

lossM�W=j       �	&5hfc�A�*

loss���=9��{       �	_�hfc�A�*

loss��S<��E       �	�mhfc�A�*

loss롘=�@&       �	�hfc�A�*

lossJ��=Jj#�       �	��hfc�A�*

loss�>�q��       �	�<hfc�A�*

loss:�^>Z�L�       �	��hfc�A�*

loss�ˢ=���       �	�rhfc�A�*

losslt�<���L       �	�hfc�A�*

loss�xS>�>`t       �	�hfc�A�*

loss}��=|���       �	c~hfc�A�*

lossQL3<��1R       �	qhfc�A�*

loss$�=p�V�       �	:�hfc�A�*

loss��E>g�,�       �	�8hfc�A�*

lossOxJ="ح�       �	(�hfc�A�*

loss!�=/6І       �	;shfc�A�*

loss��5=�:�x       �	hfc�A�*

loss��>)�j�       �	��hfc�A�*

loss��<\��       �	/mhfc�A�*

lossZ=-�8�       �	F	hfc�A�*

loss���=��ډ       �	ͯhfc�A�*

loss�!�<�c�#       �	�Zhfc�A�*

loss|��=���       �	�hfc�A�*

loss���<�dEJ       �	��hfc�A�*

loss�e2=Z-�       �	�8hfc�A�*

loss��+>`IeB       �	g�hfc�A�*

loss�)>��*�       �	7�hfc�A�*

lossJ��<5��W       �	/4 hfc�A�*

loss,W�<[���       �	� hfc�A�*

lossn�<�k}t       �	��"hfc�A�*

loss�$>s��       �	�o#hfc�A�*

lossn�@=�%��       �	� $hfc�A�*

loss�� >$L��       �	�$hfc�A�*

loss�Kl=K���       �	�w%hfc�A�*

loss�1B>��h�       �	 &hfc�A�*

loss\,�=��0�       �	��&hfc�A�*

loss	ٿ<@&_       �	�'hfc�A�*

loss��-=u�|�       �	�h(hfc�A�*

loss��<�?8�       �	{)hfc�A�*

lossz��=%J$d       �	n�)hfc�A�*

loss,Q=��ۗ       �	�i*hfc�A�*

loss���=��C�       �	F+hfc�A�*

loss��=	�Sw       �	�+hfc�A�*

loss��=3�o�       �	�E,hfc�A�*

lossJ3�=Wh��       �	��,hfc�A�*

loss�Y�=�ņ       �	��-hfc�A�*

loss��
=��       �	S#.hfc�A�*

loss�*4=	�>N       �	��.hfc�A�*

loss��<c�-�       �	�X/hfc�A�*

loss��= ��;       �	��/hfc�A�*

loss�:U=\�K       �	\�0hfc�A�*

losssį='S�m       �	?51hfc�A�*

loss��=��b       �	��1hfc�A�*

loss/�=��J       �	�h2hfc�A�*

loss���=l56       �	� 3hfc�A�*

loss��=n��c       �	I�3hfc�A�*

loss?Q<���{       �	H54hfc�A�*

loss���=Ff�
       �	O�4hfc�A�*

lossػE=�r       �	�r5hfc�A�*

loss1B�<e�L�       �	,6hfc�A�*

loss̆7=����       �	�7hfc�A�*

loss�|L<h�\`       �	ߨ7hfc�A�*

loss�{<UԀ        �	,E8hfc�A�*

lossof�<=߆�       �	S9hfc�A�*

lossj�;G@ո       �	g�9hfc�A�*

lossv��<p��I       �	�7:hfc�A�*

loss�9�;�&       �	0�:hfc�A�*

loss`j:
U0       �	A<hfc�A�*

lossD�	;	�5�       �	BC=hfc�A�*

loss�D=�O��       �	{�=hfc�A�*

loss{��=='�       �	Ow>hfc�A�*

lossEtM=b9��       �	5?hfc�A�*

lossi'<;��       �	Ϥ?hfc�A�*

loss@3�=���1       �	�8@hfc�A�*

loss�>[�w�       �	��@hfc�A�*

lossaL�<a���       �	�`Ahfc�A�*

loss�n{>�q5       �	��Ahfc�A�*

loss?�U=�3�       �	��Bhfc�A�	*

loss�=��=R       �	�Chfc�A�	*

lossHs/=�C��       �	��Chfc�A�	*

loss��!<)��       �	LDhfc�A�	*

loss�<0>�;v       �	��Dhfc�A�	*

lossEN�=�4Be       �	�tEhfc�A�	*

loss��=��u       �	=Fhfc�A�	*

loss샂=ǝ&�       �	Q�Fhfc�A�	*

loss��=&ρ&       �	�<Ghfc�A�	*

lossu^>üc�       �	��Ghfc�A�	*

lossL�>*g:�       �	�wHhfc�A�	*

loss3�>�P�       �	 Ihfc�A�	*

loss�F�=���       �	�Ihfc�A�	*

loss69>��|       �	`9Jhfc�A�	*

loss�q�=�\�b       �	�Khfc�A�	*

loss�K�=�	�       �	U�Khfc�A�	*

lossZ��=Z{�0       �	5Lhfc�A�	*

loss�R=�aM�       �	u�Lhfc�A�	*

loss�	�<��z�       �	�cMhfc�A�	*

lossd=�/��       �	��Mhfc�A�	*

loss�>`N�       �	=�Nhfc�A�	*

loss(�
=v��       �	z4Ohfc�A�	*

loss��8<�5f_       �	�Ohfc�A�	*

loss�r�<��yN       �	�bPhfc�A�	*

lossIu=E��	       �	w�Phfc�A�	*

loss��<P8<V       �	ۇQhfc�A�	*

loss<�	>H׫t       �	LRhfc�A�	*

lossU�>/B�       �	��Rhfc�A�	*

loss�!<�Z�b       �	�JShfc�A�	*

lossa"�=�i�       �	D�Shfc�A�	*

loss��=�F�       �	{Thfc�A�	*

loss��;㥈s       �	1Uhfc�A�	*

lossIU�<��&       �	"�Uhfc�A�	*

lossL2=��       �	�[Vhfc�A�	*

lossj=>`x�       �	�Whfc�A�	*

lossn�Z=���       �	פWhfc�A�	*

loss��>U��       �	FEXhfc�A�	*

loss�}J=f�Yi       �	��Xhfc�A�	*

loss�E�<��(       �	x�Yhfc�A�	*

loss�}�<�w�A       �	KXZhfc�A�	*

loss���<Z&X�       �	�Zhfc�A�	*

loss��=B��       �	߇[hfc�A�	*

loss��<U��K       �	y \hfc�A�	*

loss �=X&�q       �	��\hfc�A�	*

loss���=U��       �	�\]hfc�A�	*

lossb��<��,�       �	��]hfc�A�	*

loss��=�5�       �	��^hfc�A�	*

loss���<k�G�       �	W?_hfc�A�	*

loss�d=��"       �	�`hfc�A�	*

loss�A>�,       �	��|hfc�A�	*

loss�j�=H�       �	�}}hfc�A�	*

loss���=����       �	�~hfc�A�	*

loss�̧=��Z�       �	��~hfc�A�	*

loss��<��n�       �	 Rhfc�A�	*

loss�%�={��A       �	R,�hfc�A�	*

loss���<�N^�       �	�ˀhfc�A�	*

loss��=�3       �	�l�hfc�A�	*

loss�T/>��/�       �	��hfc�A�	*

loss��=jk�       �	���hfc�A�	*

loss���<���       �	�D�hfc�A�	*

loss61�=���D       �	�܃hfc�A�	*

lossl��=��#       �	�y�hfc�A�	*

loss�͗=�H��       �	?�hfc�A�	*

loss	�d='�       �	��hfc�A�	*

lossGA>��'�       �	l[�hfc�A�	*

loss���<�`       �	���hfc�A�	*

loss�9=���w       �	���hfc�A�	*

loss���=��5       �	��hfc�A�	*

loss�Ga>�
�A       �	r�hfc�A�	*

lossw��=�]P       �	���hfc�A�	*

losszG>�4'�       �	�=�hfc�A�	*

lossSt�=���       �	�Ӌhfc�A�	*

loss��=9v��       �	�z�hfc�A�	*

losse�	=�,�       �	4�hfc�A�	*

loss�=��v       �	^��hfc�A�	*

loss)�e=w���       �	�A�hfc�A�	*

loss�$=7��s       �		ߎhfc�A�	*

loss�=�Ͱ=       �	'��hfc�A�	*

loss��8=�j��       �	y!�hfc�A�	*

lossQ "=p*H�       �	k��hfc�A�	*

lossL�=��j       �	:X�hfc�A�	*

loss�'=���       �	9�hfc�A�	*

loss&gG>P�       �	w��hfc�A�	*

loss�b=�1       �	��hfc�A�	*

loss���=�",]       �	}��hfc�A�	*

lossJ��<�P;�       �	RI�hfc�A�	*

losscg�=W��       �	�hfc�A�	*

loss�>V���       �	t{�hfc�A�	*

loss�Ś=�Y��       �	��hfc�A�	*

lossa.�=�l�=       �	q��hfc�A�	*

loss���=�ؿ�       �	SB�hfc�A�	*

loss�C�<j���       �	0חhfc�A�	*

loss���=����       �	��hfc�A�	*

lossۣ�=��7       �	���hfc�A�	*

loss��=a�t       �	�M�hfc�A�	*

loss�;=Ů$~       �	���hfc�A�	*

loss��G=���W       �	֋�hfc�A�	*

loss���=���	       �	�)�hfc�A�	*

loss�=;q-       �	�Ĝhfc�A�	*

loss�0�<Ir\`       �	�d�hfc�A�	*

lossO�=
���       �	G�hfc�A�	*

loss�N_=ֶ�       �	࢞hfc�A�	*

loss�ϝ>E���       �	:>�hfc�A�	*

lossm�$=@���       �	�ߟhfc�A�	*

loss.6�<fc&�       �	��hfc�A�	*

loss7�;�8�Z       �	Fҡhfc�A�	*

loss�c"<�S_/       �	��hfc�A�	*

loss�;(=:�       �	P�hfc�A�	*

lossvG�=�i       �	�¤hfc�A�	*

lossW��=�$       �	p`�hfc�A�	*

lossl�=Y]�       �	���hfc�A�	*

loss��%<��&"       �	Ό�hfc�A�	*

loss�p�=C���       �	p$�hfc�A�	*

loss�g�=�l�       �	n��hfc�A�	*

loss��=X���       �	 R�hfc�A�	*

loss�(=�:�       �	��hfc�A�	*

lossἌ=��i�       �	���hfc�A�	*

losst>o��       �	%$�hfc�A�	*

loss�>U7j       �	�êhfc�A�	*

loss��F=�Ϩ�       �	si�hfc�A�	*

losss}�=S���       �	t
�hfc�A�	*

losss��=���>       �	���hfc�A�	*

loss#hP=x�O       �	M�hfc�A�	*

lossOޗ=���.       �	G�hfc�A�	*

loss&�.=?-�d       �	�z�hfc�A�	*

lossY��=��P       �	��hfc�A�	*

loss�k>��H       �	��hfc�A�	*

lossrO�=�|�       �	[]�hfc�A�	*

lossO�-=��m�       �	���hfc�A�
*

losswG=YbP       �	ȗ�hfc�A�
*

lossR��=1C�)       �	�*�hfc�A�
*

lossؠ�=�ߞy       �	Ĳhfc�A�
*

lossL��<�e�       �	*W�hfc�A�
*

loss��=���u       �	=�hfc�A�
*

loss�BC=p'�t       �	���hfc�A�
*

loss�J>"ٌ�       �	�1�hfc�A�
*

lossx��<T�&�       �	�Ƶhfc�A�
*

loss�h�<� Q;       �	�d�hfc�A�
*

loss���<x�]$       �	@��hfc�A�
*

loss�R�=e��       �	��hfc�A�
*

loss�;}=��(       �	k*�hfc�A�
*

loss��J=�%6       �	n��hfc�A�
*

loss�p[=�W�~       �	L��hfc�A�
*

loss��=��o       �	�$�hfc�A�
*

loss��=��       �	#ܺhfc�A�
*

loss��=�y<�       �	V�hfc�A�
*

loss`?:=� 	       �	�hfc�A�
*

lossQw>��       �	���hfc�A�
*

loss3��=1�s       �	9E�hfc�A�
*

loss3��<w_�V       �	q�hfc�A�
*

lossVhc=,>       �	�z�hfc�A�
*

loss(�%>�.�       �	,�hfc�A�
*

loss�q9=S�!�       �	]��hfc�A�
*

loss��=g$D       �	�D�hfc�A�
*

loss�$?>�}փ       �	Z��hfc�A�
*

loss|�<��ǀ       �	[{�hfc�A�
*

loss�2=�Ns       �	4��hfc�A�
*

loss!-�=�S�       �	�#�hfc�A�
*

loss��Q=���{       �	���hfc�A�
*

loss�=+�F{       �	&q�hfc�A�
*

loss�ao=#V�       �	L�hfc�A�
*

loss���<UQG?       �	���hfc�A�
*

loss�=�~�       �	r�hfc�A�
*

loss:�=�Q       �	Q�hfc�A�
*

loss�(g=nA<       �	���hfc�A�
*

loss`�=�hV�       �	<K�hfc�A�
*

loss�e�=����       �	���hfc�A�
*

loss�L�=1 $�       �	�}�hfc�A�
*

loss���=��)C       �	��hfc�A�
*

loss$��<��B�       �	���hfc�A�
*

lossiZ=!\�^       �	��hfc�A�
*

loss���=����       �	>\�hfc�A�
*

loss�I{=;�!�       �	���hfc�A�
*

lossCW<��_       �	��hfc�A�
*

loss��=[���       �	�6�hfc�A�
*

loss��>��]A       �	���hfc�A�
*

loss���<'��       �	}y�hfc�A�
*

loss���=�..       �	O�hfc�A�
*

loss�s�=D���       �	w��hfc�A�
*

loss�X�<ve�       �	(��hfc�A�
*

loss�=��*       �	;�hfc�A�
*

loss���=��Z       �	
��hfc�A�
*

loss8O<�C       �	�o�hfc�A�
*

loss�ɳ=����       �	��hfc�A�
*

loss��<���9       �	���hfc�A�
*

loss���=�)C~       �	:�hfc�A�
*

loss���<�a�w       �	O��hfc�A�
*

lossŁ-=�蚔       �	�g�hfc�A�
*

loss�;=��5}       �	�hfc�A�
*

loss��<;7&�       �	���hfc�A�
*

loss��<#T{       �	gC�hfc�A�
*

loss��f= W},       �	{��hfc�A�
*

lossh#=��/       �	T��hfc�A�
*

lossF�N>GtP�       �	�[�hfc�A�
*

lossj8=�{�~       �	e��hfc�A�
*

loss��=4QO       �	~��hfc�A�
*

losss�<���       �	�(�hfc�A�
*

loss�=�b       �	¾�hfc�A�
*

lossD�=c��       �	�X�hfc�A�
*

lossӘ�;@���       �	���hfc�A�
*

loss�|=$M��       �	%��hfc�A�
*

loss�=�z�       �	)�hfc�A�
*

loss7' =o���       �	���hfc�A�
*

lossQd�=�Nn-       �	Kw�hfc�A�
*

loss.�@;2@�       �	���hfc�A�
*

loss
�<����       �	Q�hfc�A�
*

loss�j�=��>       �	���hfc�A�
*

loss���=��       �	�}�hfc�A�
*

loss��<�$��       �	��hfc�A�
*

loss���=s��>       �	���hfc�A�
*

loss܀,=�>�I       �	}]�hfc�A�
*

loss�4=S��       �	% �hfc�A�
*

loss�M�<��       �	���hfc�A�
*

loss�ԙ=l�g       �	���hfc�A�
*

loss#o~<7X.       �	_&�hfc�A�
*

loss�c�<˽�       �	��hfc�A�
*

loss�|>����       �	�f�hfc�A�
*

loss��<�N��       �	S�hfc�A�
*

losst�=ڻ�+       �	��hfc�A�
*

loss��=T�^�       �	�N�hfc�A�
*

loss݋<���       �	m��hfc�A�
*

loss���<�*       �	3��hfc�A�
*

loss\,K=�A�5       �	'1�hfc�A�
*

loss�f�;>k׃       �	h��hfc�A�
*

loss�.�=!��+       �	!w�hfc�A�
*

loss��=rĸ       �	B&�hfc�A�
*

loss�N�=��&W       �	���hfc�A�
*

loss���=��X�       �	+n�hfc�A�
*

loss���<�`�       �	Z�hfc�A�
*

loss�,<e<�       �	!��hfc�A�
*

loss��=���       �	AG�hfc�A�
*

loss�1�=�&��       �	���hfc�A�
*

loss��<r��       �	���hfc�A�
*

loss�G=�c &       �	D3�hfc�A�
*

loss}�B=P��a       �	���hfc�A�
*

loss�B�=@��       �	`��hfc�A�
*

loss��g=ۮ-       �	=_�hfc�A�
*

loss���<�zO)       �	=a�hfc�A�
*

loss���=���       �	K=�hfc�A�
*

loss�0�=.��       �	�3�hfc�A�
*

lossM�d=E/       �	���hfc�A�
*

loss|HB=դi�       �	��hfc�A�
*

loss! S<K�o�       �	���hfc�A�
*

loss܌<����       �	Ή�hfc�A�
*

loss��<(��       �	<�hfc�A�
*

loss!�=�!Ze       �	���hfc�A�
*

loss-y=��^�       �	�� ifc�A�
*

lossmB:>��-�       �	{0ifc�A�
*

loss1��<%ޤ       �	��ifc�A�
*

loss�Ȣ=���       �	-#ifc�A�
*

loss[ҷ<����       �	U�ifc�A�
*

loss��<����       �	�]ifc�A�
*

loss�v=���       �	Q�ifc�A�
*

loss���<Q��d       �	[�ifc�A�
*

lossQ��=#�[       �	�-ifc�A�
*

loss�փ=3iN�       �	��ifc�A�
*

loss�j�=O��]       �	#iifc�A�*

losssN0=!c-�       �	Kifc�A�*

loss��O=���       �	ۢifc�A�*

loss�&�=�\��       �	 D	ifc�A�*

loss���<��       �	��	ifc�A�*

lossa!8<S��       �	
ifc�A�*

loss��G=��a�       �	�ifc�A�*

loss�׌=P��       �	��ifc�A�*

lossӦ=���$       �	�Tifc�A�*

lossM��>VH6�       �	��ifc�A�*

loss`|(>�_�       �	��ifc�A�*

loss��T>�r?�       �	[#ifc�A�*

loss	��<��       �	Y�ifc�A�*

loss��L=HMc�       �		Sifc�A�*

loss�B�<���0       �	Ψifc�A�*

loss���=�6       �	!?ifc�A�*

loss�=�H2�       �	n�ifc�A�*

loss��<DE�       �	N|ifc�A�*

loss�=��M�       �	�ifc�A�*

lossM�=���       �	�ifc�A�*

loss�"�=�4��       �	�bifc�A�*

lossO�o=�n�g       �	ifc�A�*

loss�}=M��       �	��ifc�A�*

losse�M<':��       �	CTifc�A�*

lossK7=:�-J       �	�ifc�A�*

lossX�<:8Cm       �	?�ifc�A�*

loss�}E=��?       �	/ifc�A�*

loss�̜=���       �	3�ifc�A�*

loss�xL=k�\       �	�difc�A�*

loss�Z<=n��       �	��ifc�A�*

loss��e='��       �	��ifc�A�*

loss�R�=/W�4       �	LTifc�A�*

loss���<��Te       �	��ifc�A�*

loss�=�9�}       �	f�ifc�A�*

loss=�a=�{`=       �	Kifc�A�*

lossEt�=?V3       �	J�ifc�A�*

loss�I^=��       �	jLifc�A�*

lossߏ�=��:       �	��ifc�A�*

loss�=�y��       �	�{ifc�A�*

loss6c�</c       �	b ifc�A�*

lossت�=qlN       �	�� ifc�A�*

lossb=�8       �	){!ifc�A�*

loss߾)=xT�       �	�L"ifc�A�*

loss�LD=˴B       �	�#ifc�A�*

loss��=K��+       �	��#ifc�A�*

loss!�O=�R�       �	p[$ifc�A�*

loss1q�<�[u�       �	�A%ifc�A�*

losse@<2nrX       �	<�%ifc�A�*

loss8�c<��V       �	Ǹ&ifc�A�*

loss��=$��       �	$`'ifc�A�*

loss���<AK��       �	S?(ifc�A�*

loss, [=��9�       �	�I)ifc�A�*

loss��=���       �	�n*ifc�A�*

loss^J�=����       �	a+ifc�A�*

loss��T=���%       �	Y�+ifc�A�*

loss#��=SP��       �	N},ifc�A�*

loss8O�<ʸ!a       �	�-ifc�A�*

loss���<�l�Z       �	|`.ifc�A�*

loss_5�<:��J       �	_/ifc�A�*

loss{�=؟ĥ       �	k�/ifc�A�*

loss&<=�,ȩ       �	Sw0ifc�A�*

lossF�6>+/�~       �	V1ifc�A�*

lossG�<Up�       �	�1ifc�A�*

loss�4=��4�       �	�O2ifc�A�*

loss�~�<%� �       �	%�2ifc�A�*

loss?�=��L       �	��3ifc�A�*

loss�K�=S�5       �	r4ifc�A�*

loss��=t���       �	��4ifc�A�*

loss�>_CH       �	RF5ifc�A�*

loss��<<�r       �	�5ifc�A�*

loss�U^;PI }       �	�s6ifc�A�*

loss�.�=�5       �	�7ifc�A�*

loss׃B=���       �	��7ifc�A�*

loss�<9J��       �	�.8ifc�A�*

loss�<�<b�	       �	ۿ8ifc�A�*

loss��=��w�       �	�o9ifc�A�*

lossv"�<��!        �	�:ifc�A�*

loss}��=e�rt       �	��:ifc�A�*

loss�X>�+�^       �	I;ifc�A�*

loss���=o�e       �	;�;ifc�A�*

loss��=�X�       �	-|<ifc�A�*

loss�a�<v$�       �	;=ifc�A�*

loss��}<�,�j       �	f�=ifc�A�*

loss��$<�ZI       �	�X>ifc�A�*

loss*+�=
ML       �	��>ifc�A�*

loss�?�;�P��       �	�?ifc�A�*

lossԭ.<($X       �	�3@ifc�A�*

lossl��;��       �	!�@ifc�A�*

lossxF�=�+�[       �	fAifc�A�*

loss��<�νF       �	�Bifc�A�*

lossJZ�=b�/        �	��Bifc�A�*

loss(z>e��       �	�yCifc�A�*

loss�1>71{'       �	IDifc�A�*

loss���<z?       �	F�Difc�A�*

loss��=��5�       �	fiEifc�A�*

loss��=�E�D       �	�Fifc�A�*

loss	O�=fʡ�       �	��Fifc�A�*

loss�%=78�       �	�dGifc�A�*

loss�`9>^��       �	L�Gifc�A�*

loss���<B�       �	��Hifc�A�*

loss��=Z&�_       �	�DIifc�A�*

loss��x=�{\i       �	��Iifc�A�*

loss���=x���       �	�uJifc�A�*

loss4�E=���r       �	Kifc�A�*

lossI�M=(X�@       �	�Kifc�A�*

loss.��=��,"       �	�dLifc�A�*

loss�'=�o�`       �	��Mifc�A�*

loss�E�<w�Ps       �	�6Nifc�A�*

loss�d�=�%Q       �	�Nifc�A�*

lossE�==�I-�       �	kaOifc�A�*

lossm=t��S       �	��Oifc�A�*

loss��=2��       �	ŐPifc�A�*

loss�h�<���1       �	�%Qifc�A�*

loss�@=v�o       �	��Qifc�A�*

lossO�>�Ԭ�       �	��Rifc�A�*

loss�er=i��
       �	�dSifc�A�*

loss#!8=�H�c       �	qTifc�A�*

lossJ֯=�!GM       �	Q�Tifc�A�*

loss)Ѷ=��k       �	QMUifc�A�*

loss9w�<V�       �	�Uifc�A�*

loss�}[=s���       �	d?Wifc�A�*

loss��=��"�       �	��Wifc�A�*

loss��>�K5       �	{�Xifc�A�*

loss��e=	C�       �	�!Yifc�A�*

loss�kc<�2�|       �	k�Yifc�A�*

loss��
=+T��       �	dXZifc�A�*

loss�ٕ=����       �	�4[ifc�A�*

loss��->���        �	2�[ifc�A�*

lossW�=4��j       �	Zg\ifc�A�*

loss��=��       �	��\ifc�A�*

lossO�$=�Ie~       �	��]ifc�A�*

lossCė=�m�a       �	�+^ifc�A�*

lossJB[<:��V       �	�B`ifc�A�*

loss	�<�~��       �	S�`ifc�A�*

loss#+6=�Y;       �	��aifc�A�*

loss�k=P�P       �	obbifc�A�*

loss�tB=�*AC       �	�cifc�A�*

loss��=�       �	difc�A�*

loss��=RjE^       �	Heifc�A�*

loss�.6<Z��       �	��eifc�A�*

loss��=`�       �	M�fifc�A�*

loss��=����       �	cBgifc�A�*

loss�^=�I5�       �	#�gifc�A�*

loss�<�Ңo       �	�vhifc�A�*

lossL��<�Bi�       �	�)iifc�A�*

loss\B�=��mE       �	�iifc�A�*

loss碋=>�oB       �	Xjifc�A�*

lossי7>pu/�       �	��jifc�A�*

loss9t=rw�       �	ۊkifc�A�*

loss�/�<:�<       �	�(lifc�A�*

lossdjO=��t�       �	v�lifc�A�*

lossv�R=�t�       �	c`mifc�A�*

lossdq�<\~k       �	P�mifc�A�*

lossʈq=�w4       �	��nifc�A�*

loss.0F=h?�N       �	`;oifc�A�*

lossV;g=�)7|       �	��oifc�A�*

loss�X>-#       �	�upifc�A�*

loss=��=�P-       �	�qifc�A�*

loss�|�=ʠ       �	�qifc�A�*

loss�Z=��v       �	�Xrifc�A�*

loss\ic=�yk�       �	��rifc�A�*

lossV�=+�>       �	Ҏsifc�A�*

loss�>>0�O�       �	�.tifc�A�*

loss���<Lj��       �	��tifc�A�*

loss�ug=�:<       �	�cuifc�A�*

loss?�=��w       �	vifc�A�*

loss�5=�F��       �	��vifc�A�*

loss��i=$���       �	�Hwifc�A�*

loss{��=�
�       �	v�wifc�A�*

lossC>��E       �	 rxifc�A�*

loss��<��s�       �	Qyifc�A�*

loss`ZG=b-1/       �	q�yifc�A�*

loss춡=�4�        �	BAzifc�A�*

loss�U=�[�       �	��zifc�A�*

loss��=���       �	+l{ifc�A�*

lossM;�=B�       �	y|ifc�A�*

loss��<w��       �	=�|ifc�A�*

loss�Y�;�$�       �	n�}ifc�A�*

loss.��=�;s�       �	JC~ifc�A�*

loss��}=�?��       �	��~ifc�A�*

loss ��=��B�       �	��ifc�A�*

loss=|q<RA�u       �	�"�ifc�A�*

loss���<�i�       �	8��ifc�A�*

loss8[�=;�{       �	}]�ifc�A�*

lossL� >�6��       �	j��ifc�A�*

loss4�<�-�       �	J��ifc�A�*

loss�u�=�&d�       �	�5�ifc�A�*

loss6��<6��       �	׃ifc�A�*

loss�?�=���       �	�o�ifc�A�*

loss4<�X��       �	5�ifc�A�*

loss�<J��?       �	Ӥ�ifc�A�*

loss$=ڭ11       �	|H�ifc�A�*

loss��&=��F       �	q�ifc�A�*

loss���<�\�       �	Fx�ifc�A�*

lossF=��Vi       �	J�ifc�A�*

loss��=��e       �	I��ifc�A�*

losst�$<����       �	5�ifc�A�*

loss�^=����       �	�͉ifc�A�*

lossbM=`�	�       �	xb�ifc�A�*

loss Z�=�H��       �	4��ifc�A�*

lossN��=���       �	֍�ifc�A�*

loss��=�C       �	K"�ifc�A�*

loss<��=���       �	[��ifc�A�*

loss��=O�Ij       �	�L�ifc�A�*

loss M�=��f�       �	C�ifc�A�*

loss2�b=_��       �	�{�ifc�A�*

lossŨ�=*8qM       �	0�ifc�A�*

loss(�x=4�)�       �	���ifc�A�*

loss��&>G$�       �		S�ifc�A�*

lossE�#<zF�       �	��ifc�A�*

lossR�=tU�       �	���ifc�A�*

lossq�=4���       �	S%�ifc�A�*

loss��=4�^�       �	/ifc�A�*

loss���<y%oy       �	B]�ifc�A�*

loss}Dk=oB�       �	Y��ifc�A�*

loss���=а͢       �	=��ifc�A�*

loss%>R){F       �	q9�ifc�A�*

loss4g�=�	>       �	�וifc�A�*

loss�~�=��q:       �	�q�ifc�A�*

loss�=�g*t       �	9	�ifc�A�*

loss��d=�mc�       �	ۥ�ifc�A�*

lossWH�<�)��       �	�B�ifc�A�*

loss�J�<���       �	;ߘifc�A�*

loss���=; n�       �	hz�ifc�A�*

loss�5=e?x>       �	��ifc�A�*

lossxZ�<X�X�       �	���ifc�A�*

lossX�<��%       �	D3�ifc�A�*

lossI1=Đ�x       �	țifc�A�*

lossq/N<�ڈ5       �	>\�ifc�A�*

loss?��<0�       �	��ifc�A�*

lossD�<��	       �	I��ifc�A�*

loss��9<�A�       �	?�ifc�A�*

loss=+=nF�l       �	�ifc�A�*

loss�=rI       �	�I�ifc�A�*

loss��>��#       �	��ifc�A�*

lossܗ=���       �	�Šifc�A�*

loss�>��\N       �	sg�ifc�A�*

loss��2<�XƗ       �	u�ifc�A�*

loss�t�<�㸏       �	$��ifc�A�*

loss��>�/�R       �	�9�ifc�A�*

lossל=�1<       �	>�ifc�A�*

loss�4�<�n��       �	i�ifc�A�*

loss�Ӣ=X��       �	~�ifc�A�*

loss�=h3�       �	0�ifc�A�*

losso��<�jV       �	J��ifc�A�*

loss1��;�M[�       �	�O�ifc�A�*

loss�C�=v��&       �	�ifc�A�*

loss�C�=m�3�       �	�ifc�A�*

lossm�=�먽       �	�R�ifc�A�*

loss��v=_��       �	���ifc�A�*

lossaM=M�NT       �	���ifc�A�*

loss�_=�A�)       �	e8�ifc�A�*

loss�hG=\�Ǉ       �	,ӫifc�A�*

loss{0_=M�X       �	ds�ifc�A�*

loss2y�; �Yz       �	
�ifc�A�*

loss&/�<Yǧ       �	4��ifc�A�*

loss��K>�g�t       �	�7�ifc�A�*

loss�.<�`c�       �	��ifc�A�*

lossq�O=��       �	܂�ifc�A�*

loss��=ttn#       �	�"�ifc�A�*

lossZ_�<��C       �	4��ifc�A�*

lossW�h=#,�       �	y[�ifc�A�*

lossȽV<0�H�       �	U��ifc�A�*

loss�`	>�iy�       �	닲ifc�A�*

loss���<D�͈       �	$&�ifc�A�*

loss2�=���       �	�³ifc�A�*

loss���=L[��       �	�^�ifc�A�*

lossz�;��h�       �	���ifc�A�*

loss|=�<t.��       �	{��ifc�A�*

lossΌ�<��'�       �	�G�ifc�A�*

loss�-<BDLB       �	�ifc�A�*

loss)�=�&c�       �	���ifc�A�*

lossƸ�=,b��       �	��ifc�A�*

lossQ{J>�38�       �	���ifc�A�*

loss���=�1�       �	�Z�ifc�A�*

loss�R�<�H0n       �	��ifc�A�*

loss�8>l��~       �	���ifc�A�*

loss@Y�=b7ޢ       �	qT�ifc�A�*

loss᪷<O"i�       �	j��ifc�A�*

loss҄1<H�`3       �	��ifc�A�*

loss���=qXXG       �	�(�ifc�A�*

loss��/<�h�9       �	�½ifc�A�*

loss��(=U���       �	v��ifc�A�*

loss��\=<�p       �	i�ifc�A�*

lossVr{=�*��       �	`x�ifc�A�*

lossv��<�%�3       �	=�ifc�A�*

loss��=��>G       �	��ifc�A�*

lossҲ�=��       �	f��ifc�A�*

loss�pG=�k�       �	���ifc�A�*

loss���=B*l       �	C;�ifc�A�*

loss�y�<�B�       �	���ifc�A�*

loss��=z�ɞ       �	 y�ifc�A�*

loss(�>RoiO       �	��ifc�A�*

lossC��=��D�       �	j��ifc�A�*

loss�xD<3(�       �	�\�ifc�A�*

loss�X�<�O��       �	Q��ifc�A�*

lossї<�W�       �	��ifc�A�*

lossHG�<���       �	�+�ifc�A�*

loss��;��{       �	;��ifc�A�*

lossQ��=��?�       �	�j�ifc�A�*

lossTG�<�<\�       �	T�ifc�A�*

lossN��={@\       �	Ƨ�ifc�A�*

loss}=�@ �       �	8J�ifc�A�*

loss�̑<��       �	���ifc�A�*

loss��=��0h       �		��ifc�A�*

loss�[{<u��       �	�#�ifc�A�*

loss}�=Q���       �	f��ifc�A�*

loss4x=���Y       �	GV�ifc�A�*

loss��=��:       �	F��ifc�A�*

loss�v�=��sl       �	���ifc�A�*

lossD-=�E       �	��ifc�A�*

loss���<K5��       �	 ��ifc�A�*

loss��=��H       �	�P�ifc�A�*

loss�gR<K^0       �	y��ifc�A�*

loss�[=�*(�       �	���ifc�A�*

lossf+<��K�       �	�+�ifc�A�*

loss�#=N��       �	H��ifc�A�*

lossw}�<�<��       �	Zb�ifc�A�*

losslΰ<��u       �	���ifc�A�*

lossi=�	۞       �	���ifc�A�*

loss;p3=���       �	|(�ifc�A�*

loss�G=����       �	���ifc�A�*

loss��;���y       �	l^�ifc�A�*

loss�ѷ<��       �	3��ifc�A�*

loss���=�ޠ�       �	��ifc�A�*

loss`��<h�+F       �	2�ifc�A�*

loss���9\3��       �	���ifc�A�*

loss��S<O       �	�r�ifc�A�*

loss�Su<iY)       �	�	�ifc�A�*

loss��<�R!Q       �	���ifc�A�*

loss�O�;�Ei       �	\;�ifc�A�*

loss15$;W�       �	���ifc�A�*

lossXμ=�c�<       �	�q�ifc�A�*

loss��r=
�9       �	J�ifc�A�*

loss��;:�i��       �	��ifc�A�*

lossR�:#r�       �	�H�ifc�A�*

loss��0<|i[       �	d��ifc�A�*

loss$#V=:�X       �	���ifc�A�*

loss�X<�o��       �	���ifc�A�*

loss��9�NX!       �	7�ifc�A�*

loss-��<P�(�       �	��ifc�A�*

loss��>�A�{       �	V��ifc�A�*

loss��<g��       �	� �ifc�A�*

lossi>��f�       �	ȵ�ifc�A�*

loss���<Ugaf       �	rS�ifc�A�*

lossK>>���       �	���ifc�A�*

lossB&#=�F��       �	���ifc�A�*

loss���<[s)       �	��ifc�A�*

lossX�=1�       �	6��ifc�A�*

lossV��=�W�/       �	�E�ifc�A�*

lossh�=;�V       �	���ifc�A�*

loss��}=x�vq       �	\r�ifc�A�*

loss�4=d�V       �	��ifc�A�*

loss��>����       �	ߣ�ifc�A�*

lossQR=�A��       �	�;�ifc�A�*

loss=�R=r�CH       �	c��ifc�A�*

loss��=��=B       �	g�ifc�A�*

loss��B=�&�k       �	e��ifc�A�*

loss��b=��       �	���ifc�A�*

loss��t<Dob       �	�g�ifc�A�*

loss�5�=��       �	\ �ifc�A�*

loss$p�<T���       �	؝�ifc�A�*

loss�!�<\���       �	b1�ifc�A�*

lossI��<�/��       �	���ifc�A�*

lossJ�=bJ�y       �	B\�ifc�A�*

loss�<��*       �	3��ifc�A�*

loss�۸<�S       �	T��ifc�A�*

loss�]<�$-�       �	V*�ifc�A�*

lossR�D=�\�       �	@��ifc�A�*

loss��==�*�F       �	�Y�ifc�A�*

lossd�=���9       �	<��ifc�A�*

loss!�=FzM�       �	Ҋ�ifc�A�*

lossm�)<$��s       �	y �ifc�A�*

loss��B=�p       �	÷�ifc�A�*

loss��<�J�n       �	�l�ifc�A�*

loss2s�;�C�       �	S�ifc�A�*

loss�#^=�nD       �	���ifc�A�*

lossJʶ<c���       �	J{�ifc�A�*

loss6h!=��T�       �	��ifc�A�*

loss�z�=��@�       �	k��ifc�A�*

lossʏ�=L׏[       �	�U�ifc�A�*

loss��=5>�       �	��ifc�A�*

loss�=�.��       �	��ifc�A�*

loss ��<��#N       �	2��ifc�A�*

lossWo�<�5       �	�A jfc�A�*

loss�˃=�:�*       �	�� jfc�A�*

loss�~�<�ː-       �	6�jfc�A�*

loss�=�
܋       �	cajfc�A�*

loss)l�=Iu�P       �	\ jfc�A�*

loss�=P�Ve       �	 �jfc�A�*

loss�Ւ<*��d       �	Q-jfc�A�*

loss��;w���       �	�jfc�A�*

lossZ��<2��       �	�sjfc�A�*

loss���=�@h�       �	D�jfc�A�*

losso`>.BUo       �	9jfc�A�*

loss���=����       �	��jfc�A�*

loss���=�ٌ�       �	�}jfc�A�*

loss�m�=Մ�:       �	l!jfc�A�*

lossV=�o:�       �	��jfc�A�*

loss���<9S�       �	�Njfc�A�*

lossj>W=P       �	w�jfc�A�*

lossߴ	>����       �	�� jfc�A�*

loss��v<��?       �	�)!jfc�A�*

lossj�<~�j�       �	��!jfc�A�*

loss��=��ޑ       �	e"jfc�A�*

lossaUd=�V"       �	�#jfc�A�*

loss$y�=|�       �	��#jfc�A�*

lossq��<+6�&       �	ݕ$jfc�A�*

loss���=x���       �	7T%jfc�A�*

loss���;�E       �	T�&jfc�A�*

loss��= L�;       �	��'jfc�A�*

loss�?=;�^       �	��(jfc�A�*

lossύ�=���       �	 �)jfc�A�*

loss� h=B��       �	��*jfc�A�*

loss�Z�=���       �	��+jfc�A�*

loss�)=�0�       �	�-jfc�A�*

lossX>= 5h:       �	R
.jfc�A�*

lossD�:=��ai       �	_�.jfc�A�*

lossF~=f�W8       �	I�/jfc�A�*

loss�e�<r��       �	:!1jfc�A�*

lossq�9<?���       �	��1jfc�A�*

loss��=��,       �	m<3jfc�A�*

loss{n�=i0�       �	��3jfc�A�*

loss	X=���6       �	��4jfc�A�*

losst�1<`r��       �	+25jfc�A�*

loss��=s:{�       �	�5jfc�A�*

loss:��=3�b�       �	��6jfc�A�*

loss��0=��       �	�/7jfc�A�*

loss�=�<|�[       �	��7jfc�A�*

loss��a<N�5W       �	i�8jfc�A�*

loss#��=��6       �	��9jfc�A�*

loss[�=(�       �	��:jfc�A�*

loss�@!=�ɽ       �	�%;jfc�A�*

loss��>�eq       �	\�;jfc�A�*

loss!̴<�N)�       �	�x<jfc�A�*

lossj1A<�?K�       �	X=jfc�A�*

lossJ��<�4��       �	��>jfc�A�*

loss$ه<����       �	yu?jfc�A�*

loss@(b=g�Fh       �	�
@jfc�A�*

loss��=�ED�       �	��@jfc�A�*

loss���=�x�       �	@�Ajfc�A�*

loss}�O=ws�c       �	�'Bjfc�A�*

loss�-;�0N)       �	�Bjfc�A�*

loss��^<c�       �	��Cjfc�A�*

loss�r=v&�       �	�RDjfc�A�*

loss��<P,@       �	��Djfc�A�*

loss]��=�Lr       �	n�Ejfc�A�*

loss��=4��+       �	�7Fjfc�A�*

loss��;Or�b       �	-�Fjfc�A�*

loss�g�;Nt	r       �	\wGjfc�A�*

loss S;�z�D       �	�Hjfc�A�*

loss�)={��       �	ٵHjfc�A�*

loss��=dDޔ       �	�LIjfc�A�*

loss��=����       �	��Ijfc�A�*

loss�D2<k	�       �	�yJjfc�A�*

lossۻ�<��
�       �	�Kjfc�A�*

loss�_=�tc�       �	�Kjfc�A�*

lossj��<�T�       �	DLjfc�A�*

loss&�<���       �	 �Ljfc�A�*

loss��<cG�       �	�Mjfc�A�*

losss T=�ΥA       �	Njfc�A�*

losse�=���       �	3�Njfc�A�*

loss�B = Az�       �	�Ojfc�A�*

loss?��=�B       �	�Pjfc�A�*

loss�J�=#o�       �	XQjfc�A�*

loss�<Zz=       �	3�Qjfc�A�*

loss��=�ߘ�       �	�\Rjfc�A�*

lossפ�<$���       �	hSjfc�A�*

loss��<7���       �	w�Sjfc�A�*

loss��=�=W�       �	xETjfc�A�*

loss��=n�s       �	j�Tjfc�A�*

loss�<K�L       �	��Ujfc�A�*

lossN��=2��	       �	�NVjfc�A�*

loss� =�#]_       �	��Vjfc�A�*

loss�)=V�"
       �	R�Wjfc�A�*

loss{h�<�A�?       �	uXjfc�A�*

loss�Q<���       �	5�Xjfc�A�*

loss\v<�_�       �	'KYjfc�A�*

loss��<���       �	�Yjfc�A�*

loss;��<���H       �	JzZjfc�A�*

loss�<!=-�L        �	[jfc�A�*

losst&�<j�;u       �	�[jfc�A�*

lossx�6=�=�c       �	�2\jfc�A�*

loss��>m�D,       �	L�\jfc�A�*

lossݚ==��W_       �	XW]jfc�A�*

lossy/�=�El�       �	��]jfc�A�*

loss^"�=�U       �	��^jfc�A�*

loss
��=.a�       �	�_jfc�A�*

loss2K<(A.�       �	v�_jfc�A�*

lossS�=���       �	<M`jfc�A�*

loss,�<b���       �	B�`jfc�A�*

loss�R>�;��       �	{�ajfc�A�*

loss]JI=��G�       �	!bjfc�A�*

loss(��;Qٓ�       �	v�bjfc�A�*

loss�5<N�Rn       �	t\cjfc�A�*

loss�;�=SQ       �	�djfc�A�*

loss�I[<����       �	{�djfc�A�*

loss�tc=WycZ       �	�<ejfc�A�*

loss�&�=0�.       �	{�ejfc�A�*

loss��<y��i       �	�ufjfc�A�*

loss��t=w�       �	�gjfc�A�*

loss��>v)�       �	��gjfc�A�*

loss/�=���       �		6hjfc�A�*

lossj�=��h�       �	��hjfc�A�*

loss]^v=)J�r       �	�aijfc�A�*

loss�<T=�ג�       �	��ijfc�A�*

loss��<$�5       �	��jjfc�A�*

lossd��<o�.�       �	-'kjfc�A�*

loss��<�0��       �	Ͼkjfc�A�*

loss6GF=yH0       �	$^ljfc�A�*

loss(�d=�D7�       �	��ljfc�A�*

loss��l=FJ"�       �	�mjfc�A�*

loss���<�J{�       �	�8njfc�A�*

lossQ�=�8|~       �	��njfc�A�*

loss&S~=��t�       �	]pojfc�A�*

loss�[=����       �	-pjfc�A�*

lossAA�<��$�       �	ƣpjfc�A�*

loss���<K!�       �	�Aqjfc�A�*

loss��R=ȹѡ       �	4�qjfc�A�*

lossN�%=��J       �	�wrjfc�A�*

loss��=:lpZ       �	�sjfc�A�*

loss�>Z=%���       �	f�sjfc�A�*

loss!�r=����       �	��tjfc�A�*

loss�Y�<tY�       �	�6ujfc�A�*

loss�(�=X���       �	F�ujfc�A�*

lossd�0=r�f        �	��vjfc�A�*

loss�=�#�        �	�(wjfc�A�*

loss�;>�D�       �	��wjfc�A�*

losshY=Xa��       �	�xjfc�A�*

loss p�="���       �	�"yjfc�A�*

lossꢡ<� V       �	�yjfc�A�*

loss#�<��.       �	tzjfc�A�*

loss�:�;f ��       �	X{jfc�A�*

losss�F<	�M       �	�{jfc�A�*

lossW��;W�%�       �	�a|jfc�A�*

loss���=z��       �	Tt}jfc�A�*

loss~>�g�i       �	P~jfc�A�*

loss�n>84�       �	^�~jfc�A�*

loss_�U=��6       �	-]jfc�A�*

loss�_2=�QG�       �	��jfc�A�*

loss��=<��       �	��jfc�A�*

loss�9�<Gga       �	�9�jfc�A�*

loss,y�<j�f       �	��jfc�A�*

loss�>�<��       �	���jfc�A�*

loss
�8=?�D�       �	�/�jfc�A�*

loss_1Z=�qO�       �	�˃jfc�A�*

loss�=�_8       �	#g�jfc�A�*

loss�QG=$s�       �	d�jfc�A�*

loss�k�<r       �	5��jfc�A�*

loss�v<�k�       �	G=�jfc�A�*

loss1��<��       �	�߆jfc�A�*

lossx��=t�w�       �	��jfc�A�*

loss1y=�uSJ       �	�#�jfc�A�*

loss�ŕ<k6�2       �	Ȉjfc�A�*

loss�_=�k�       �	rm�jfc�A�*

loss�1f=��N�       �	��jfc�A�*

loss�Ʉ;�H7�       �	�Ɋjfc�A�*

loss#�<�"��       �	�w�jfc�A�*

loss#;�<�P�t       �	q�jfc�A�*

loss�%�;B���       �	�Ìjfc�A�*

loss��<�#�       �	�l�jfc�A�*

lossA�;d9:?       �	��jfc�A�*

loss`b�<-"/|       �	I��jfc�A�*

lossaNN=m� �       �	�c�jfc�A�*

loss��]<����       �	��jfc�A�*

loss3�F=��]       �	��jfc�A�*

loss��m=�v`       �	2Y�jfc�A�*

lossԂ�;���       �	��jfc�A�*

lossEy�=^_�&       �	���jfc�A�*

loss��=dz��       �	.T�jfc�A�*

loss\��=�&N       �	���jfc�A�*

loss=e}<&0D       �	���jfc�A�*

lossqB[;}Ag�       �	�?�jfc�A�*

loss��@;��       �	O�jfc�A�*

loss/Un=O.�       �	���jfc�A�*

loss��#=e�7       �	�7�jfc�A�*

loss:�<!9�
       �	ۗjfc�A�*

lossH��;@�Y       �	�r�jfc�A�*

loss�3�<�[S       �	��jfc�A�*

loss�W;=�[�       �	���jfc�A�*

lossԿ=���       �	B^�jfc�A�*

lossޖ=b[A�       �	?�jfc�A�*

loss���=�8�k       �	�jfc�A�*

lossʙ�=;�B       �	�?�jfc�A�*

loss�\=�*�       �	.�jfc�A�*

loss��</y,       �	�{�jfc�A�*

loss�3�<�q�N       �	��jfc�A�*

loss��L;�u�       �	i��jfc�A�*

loss��<���       �	O@�jfc�A�*

lossܩ]<���       �	���jfc�A�*

loss!��<��g       �	؂�jfc�A�*

lossו�=q�Ni       �	��jfc�A�*

loss�=�O�       �	Y��jfc�A�*

loss�|X="���       �	J^�jfc�A�*

loss��
<���)       �	���jfc�A�*

lossc��<�Fp�       �	u��jfc�A�*

loss�Kc=h��       �	$_�jfc�A�*

loss���<�a��       �		��jfc�A�*

loss�'<=��w�       �	%�jfc�A�*

loss���=绨       �	��jfc�A�*

loss�-�=����       �	���jfc�A�*

lossQ�<��<       �	�ʨjfc�A�*

lossmf�=�g{       �	4��jfc�A�*

loss펦<ֳ0       �	YQ�jfc�A�*

lossV�<Go�>       �	��jfc�A�*

loss(]=C��Q       �	M��jfc�A�*

lossE=�<y��       �	�`�jfc�A�*

loss��L=	��M       �	�-�jfc�A�*

loss���=Q{       �	�ŭjfc�A�*

lossZbW>:kH�       �	�a�jfc�A�*

loss��=�q       �	D��jfc�A�*

loss#��=�
       �	u��jfc�A�*

loss]3=�ǁ�       �	�9�jfc�A�*

loss^��=C�H       �	�аjfc�A�*

loss���<���       �	k�jfc�A�*

loss8Yt=���y       �	��jfc�A�*

loss��4=Rq�       �	��jfc�A�*

loss���<��V       �	~8�jfc�A�*

loss#
=t֛L       �	γjfc�A�*

loss�j�<���       �	,e�jfc�A�*

loss��4=���       �	���jfc�A�*

loss��W=L,P9       �	���jfc�A�*

lossm��=�-��       �	pB�jfc�A�*

loss���<��+r       �	�Ӹjfc�A�*

loss�r.=,S�y       �	���jfc�A�*

loss�6�=���I       �	�'�jfc�A�*

lossӎ=�.��       �	%˺jfc�A�*

loss((�=�6�Y       �	h�jfc�A�*

loss���=^�O]       �	�jfc�A�*

lossݗ�<읚       �	E��jfc�A�*

loss#��<���F       �	l>�jfc�A�*

loss`�>b�5       �	�޽jfc�A�*

loss��o=��       �		��jfc�A�*

loss�f�<���       �	j��jfc�A�*

loss/�=�{�`       �	�H�jfc�A�*

loss���<d�$       �	���jfc�A�*

loss�T=E���       �	|��jfc�A�*

loss�B'=Ut̨       �	�@�jfc�A�*

loss���<�f7�       �	���jfc�A�*

loss[�= �u       �	���jfc�A�*

loss�}�<�A�       �	r5�jfc�A�*

loss#C=Pf.       �	5��jfc�A�*

loss�{=��,R       �	�x�jfc�A�*

loss��<��b       �	�!�jfc�A�*

loss�m�=����       �	z��jfc�A�*

loss
X�<����       �	kc�jfc�A�*

loss=7<�4A5       �	.�jfc�A�*

lossg�;�,�k       �	,��jfc�A�*

loss7:�;�B       �	�M�jfc�A�*

loss�R�<wDŠ       �	���jfc�A�*

loss��<i\XO       �	���jfc�A�*

loss�C�<= �a       �	�)�jfc�A�*

loss�w=��I       �	\��jfc�A�*

loss�= ȣ�       �	�s�jfc�A�*

lossQ��<0Dk!       �	��jfc�A�*

loss0�=o4�~       �	���jfc�A�*

loss�5�=+C*9       �	>[�jfc�A�*

loss4`�<��Da       �	c�jfc�A�*

loss�=�k�l       �	��jfc�A�*

loss\*�=;Ee`       �	���jfc�A�*

loss���<��       �	nL�jfc�A�*

lossr��=/��       �	? �jfc�A�*

loss��<_3F       �	���jfc�A�*

lossͺ�=X
��       �	L�jfc�A�*

loss)K:=��]�       �	Z��jfc�A�*

loss��=ƥ�       �	���jfc�A�*

loss=��<Iȥ�       �	8�jfc�A�*

loss���=N;�       �	��jfc�A�*

loss��<�p�o       �	���jfc�A�*

lossf��<�v��       �	
.�jfc�A�*

loss2[<ͪY�       �	���jfc�A�*

loss�=�]�       �	�p�jfc�A�*

lossqud={       �	�jfc�A�*

loss�a�;�՟       �	P��jfc�A�*

loss���;_%�       �	x_�jfc�A�*

loss;��<T��D       �	 �jfc�A�*

loss�-a=��L	       �	(��jfc�A�*

lossSv�=Z�$       �	36�jfc�A�*

lossj��<G�w�       �	@��jfc�A�*

loss�Pj=��]$       �	���jfc�A�*

loss��=�k�       �	!�jfc�A�*

loss�_=z�
�       �	^��jfc�A�*

loss�[�<\g�       �	�W�jfc�A�*

loss�˿;kc�       �	Q��jfc�A�*

loss�Z=U8�       �	Z��jfc�A�*

loss8�:2a��       �	<�jfc�A�*

loss���:�W�y       �	U��jfc�A�*

loss���<��       �	6w�jfc�A�*

loss0>�$�        �	��jfc�A�*

loss���;^]�d       �	I��jfc�A�*

loss���=�*�       �	#��jfc�A�*

loss��>F	�       �	�P�jfc�A�*

lossQG�=?��       �	���jfc�A�*

loss���=%�       �	���jfc�A�*

loss+Ǎ=0<�       �	�3�jfc�A�*

lossW7�<Xi@       �	���jfc�A�*

loss��&<@Lo�       �	p^�jfc�A�*

loss]��=����       �	(��jfc�A�*

loss��>u�       �	u��jfc�A�*

loss�!;�>�-       �	�)�jfc�A�*

loss���=�J��       �	���jfc�A�*

loss��:=���'       �	�^�jfc�A�*

losss�^=�X(�       �	���jfc�A�*

lossh1=��\       �	���jfc�A�*

loss��=m�       �	�*�jfc�A�*

loss{=����       �	"��jfc�A�*

lossa"1<��?�       �	]�jfc�A�*

loss�l�<�� q       �	I��jfc�A�*

lossj��=� ��       �	���jfc�A�*

lossWW=w2�       �	�0�jfc�A�*

loss��U=�Jj       �	���jfc�A�*

loss�^�=��F+       �	�\�jfc�A�*

lossoT=�	�	       �	���jfc�A�*

loss��<���t       �	ס�jfc�A�*

loss��>r�%�       �	?7�jfc�A�*

lossV=�5�d       �	.��jfc�A�*

lossV��<��8�       �	_�jfc�A�*

lossŐH=Y+Q�       �	���jfc�A�*

loss��>=2�\W       �	'��jfc�A�*

loss�C=��
       �	�R�jfc�A�*

loss�	=����       �	���jfc�A�*

lossE�=�!�       �	��jfc�A�*

loss�R�=�T�       �	d#�jfc�A�*

loss��<����       �	f��jfc�A�*

loss�Z�<����       �	���jfc�A�*

loss	<2��#       �	[y�jfc�A�*

losse�/=|�ē       �	��jfc�A�*

loss)0P=�|�^       �	jM�jfc�A�*

loss�~=�'�'       �	��jfc�A�*

loss�%�=K�ܤ       �	���jfc�A�*

loss{�<jx�       �	JB�jfc�A�*

loss��O=f+�       �	k��jfc�A�*

loss}x�<#���       �	4i�jfc�A�*

loss�`I=E(       �	� kfc�A�*

lossךo=���       �	�� kfc�A�*

lossh=�%       �	�/kfc�A�*

lossW!=�N�C       �	��kfc�A�*

loss��>=,�h       �	ٔkfc�A�*

lossї�=����       �	�qkfc�A�*

loss\�V<w�w       �	^kfc�A�*

lossH��=iζ       �	�kfc�A�*

loss��=����       �	�Kkfc�A�*

lossܤ�<��e       �	�kfc�A�*

loss2,�<̢�1       �	͔kfc�A�*

loss�P/=��3�       �	+kfc�A�*

loss#�<�e5       �	f�kfc�A�*

loss&=��{�       �	~Rkfc�A�*

loss�,�=�~g*       �	��kfc�A�*

loss,+�<:�       �	��	kfc�A�*

lossA!<6�s       �	?
kfc�A�*

loss�7B=\0Ic       �	%�
kfc�A�*

lossb=�k\�       �	�Dkfc�A�*

loss�o�=^�       �	��kfc�A�*

loss .X=��#�       �	~rkfc�A�*

loss�l�<y��I       �	�	kfc�A�*

loss��-<� Ą       �	a�kfc�A�*

lossz>Oôx       �	�Ckfc�A�*

loss6,�<�`e�       �	��kfc�A�*

losso��<r�J�       �	ipkfc�A�*

loss��=��       �	�kfc�A�*

lossq�L<����       �	C�kfc�A�*

losssd}<�"N       �	�;kfc�A�*

lossW�9=�6��       �	��kfc�A�*

loss�<���V       �	�okfc�A�*

loss&��<�u�       �	�kfc�A�*

loss:@�<�{��       �	��kfc�A�*

loss��<��'       �	32kfc�A�*

lossn�<!��/       �	�kfc�A�*

loss�2v=��%       �	kfc�A�*

loss�>Μ!�       �	��kfc�A�*

loss���<�#�       �	Ikfc�A�*

lossI3=���k       �	�kfc�A�*

loss���=횷       �	�~kfc�A�*

lossW�<�L��       �	�kfc�A�*

lossQ+�<�K��       �	�kfc�A�*

lossm=��,       �	�ekfc�A�*

loss���=:"4�       �	�kfc�A�*

loss�*<k&       �	*�kfc�A�*

loss*�~=��e�       �	G>kfc�A�*

loss�	�=DFnW       �	��kfc�A�*

lossq�<��ъ       �	#�kfc�A�*

loss�ɥ<�x��       �	�)kfc�A�*

loss�{=5�8       �	��kfc�A�*

loss�'=��h       �	�dkfc�A�*

loss�B=�&�       �	m�kfc�A�*

lossZ�<���       �	S� kfc�A�*

loss��e=�t
�       �	�+!kfc�A�*

loss$�1=U�,�       �	��!kfc�A�*

lossr�8>��        �	g"kfc�A�*

lossf<@;*�,a       �	8#kfc�A�*

loss�E�;+�
       �	F�#kfc�A�*

loss���<�x�       �	�q$kfc�A�*

loss��[<�|�       �	�z%kfc�A�*

loss��'<ǯ��       �	&kfc�A�*

loss=�l@       �	3�&kfc�A�*

loss[��=/N�       �	h'kfc�A�*

loss��><<���       �	�(kfc�A�*

loss�%V=��,B       �	��(kfc�A�*

loss��o=�c�z       �	#2)kfc�A�*

loss���=��5       �	�)kfc�A�*

loss�ٚ=8���       �	ka*kfc�A�*

lossr�<�;�,       �	]�*kfc�A�*

loss;�=�KR�       �	�+kfc�A�*

lossZ`p<(���       �	k-,kfc�A�*

loss�h�=O�E�       �	g�,kfc�A�*

loss��<Lw�       �	"q-kfc�A�*

lossx03=l�       �	:.kfc�A�*

lossNI�=K|��       �	//kfc�A�*

loss���=H�^�       �	��/kfc�A�*

lossxV<)|       �	�h0kfc�A�*

loss��;�Z       �	�1kfc�A�*

loss��\=3)��       �	h�1kfc�A�*

lossk��=��D       �	��2kfc�A�*

loss��<\��       �	��3kfc�A�*

lossE�=���       �	є4kfc�A�*

losss�=��       �	�-5kfc�A�*

loss;�)>�W��       �	%�5kfc�A�*

loss.5�=��7       �	�d6kfc�A�*

losseZ�=���       �	u7kfc�A�*

loss�[�=�+�%       �	Ý7kfc�A�*

lossZ�`=�]}/       �	)<8kfc�A�*

loss=�<���       �	U�8kfc�A�*

lossvSj=kk�G       �	ԁ9kfc�A�*

loss&Ou=ۧKu       �	�:kfc�A�*

loss��<�H��       �	��:kfc�A�*

loss- 3=S#t*       �	TR<kfc�A�*

loss�!�< �F�       �	:�<kfc�A�*

lossԵ=��q       �	�=kfc�A�*

loss��<|d�       �	�>kfc�A�*

lossx�.<1���       �	[�>kfc�A�*

loss4¶<���       �	J^?kfc�A�*

loss�b<5��`       �	�Akfc�A�*

loss��<"���       �	��Akfc�A�*

lossA�h=ܨ)       �	�6Bkfc�A�*

loss�7�=��1       �	F�Bkfc�A�*

loss��3=���       �	�fCkfc�A�*

loss�9�=k��       �	|
Dkfc�A�*

loss�L<F�X       �	C�Dkfc�A�*

loss	�<�b7�       �	�>Ekfc�A�*

loss��=���J       �	{�Ekfc�A�*

loss��r=e1*       �	�wFkfc�A�*

loss��<�\�u       �	@Gkfc�A�*

loss:�f<(7�'       �	��Gkfc�A�*

loss��=���       �	EHkfc�A�*

lossi��<���       �	�4Ikfc�A�*

loss4�R<�!Ui       �	�Ikfc�A�*

loss{D�=�d$�       �	ڑJkfc�A�*

lossWcR=�*��       �	�(Kkfc�A�*

loss�t=ԊpV       �	�Kkfc�A�*

loss�0
=�FQ       �	�WLkfc�A�*

lossi<<�`]�       �	�Lkfc�A�*

loss�Z�<���t       �	)�Mkfc�A�*

lossS�<=�32       �	0Nkfc�A�*

loss	�.<�M{]       �	`�Nkfc�A�*

lossq
�<I�U       �	�nOkfc�A�*

loss�5< �T       �	Pkfc�A�*

loss���<z~&       �	F�Pkfc�A�*

loss|U�<�$�       �	~QQkfc�A�*

losso�$=^�k�       �	��Qkfc�A�*

loss�8@<�uv�       �	�Rkfc�A�*

lossI��;u�ʤ       �	�Skfc�A�*

loss�?-=G��       �	��Skfc�A�*

loss�/-;�"��       �	1[Tkfc�A�*

loss�>��L	       �	��Tkfc�A�*

loss�P<�H��       �	7�Ukfc�A�*

loss-�=��       �	t#Vkfc�A�*

loss���=}W*�       �	øVkfc�A�*

loss7=�i�X       �	�MWkfc�A�*

loss�]);%�b       �	K�Wkfc�A�*

loss_��;�v��       �	9|Xkfc�A�*

loss�4=!�F�       �	�Ykfc�A�*

lossd�;i-(D       �	6�Ykfc�A�*

loss�S3>��J�       �	�DZkfc�A�*

lossT>n=N�_(       �	�Zkfc�A�*

loss��=�yP       �	�t[kfc�A�*

loss�io<��Z�       �	�\kfc�A�*

loss�M�=�A�<       �	ܡ\kfc�A�*

loss��=&�g�       �	�7]kfc�A�*

loss�}�<����       �	��]kfc�A�*

loss��=�
p       �	�d^kfc�A�*

lossf�=ui�       �	��^kfc�A�*

loss�K�;Y��       �	��_kfc�A�*

loss�q�<$vz       �	�*`kfc�A�*

loss�f�<���       �	%�`kfc�A�*

loss�=3�˻       �	�bakfc�A�*

loss�ZJ;?�{       �	z�akfc�A�*

lossz8�= w       �	��bkfc�A�*

loss<(Kg       �	 Eckfc�A�*

loss\~�<��	       �	��ckfc�A�*

loss��=�E       �	c|dkfc�A�*

loss��L=Gp:�       �	�}ekfc�A�*

losss�=���       �	�fkfc�A�*

loss��=:z\       �	�_gkfc�A�*

loss[��=2q�*       �	�hkfc�A�*

losswz;o�S#       �	�hkfc�A�*

loss�c-=���       �	�ikfc�A�*

losse0/<��F8       �	c(jkfc�A�*

loss��=�<s       �	�jkfc�A�*

loss�a�;��p       �	fgkkfc�A�*

loss���<��QA       �	�
lkfc�A�*

loss���<n��6       �	 �lkfc�A�*

loss&�<s�w)       �	$Bmkfc�A�*

loss��<\�7`       �	��mkfc�A�*

loss;��;~ l7       �	�vnkfc�A�*

loss&=_�	@       �	�okfc�A�*

loss�`<(^s!       �	�)pkfc�A�*

lossO��<þC!       �	<�pkfc�A�*

loss2=g=       �	�Xqkfc�A�*

lossxl$=\�x�       �	��qkfc�A�*

loss�R�=7��       �	a�rkfc�A�*

loss#Bg=��#+       �	�#skfc�A�*

loss���<�M�       �	��skfc�A�*

loss�T-=ڈ�p       �	�Stkfc�A�*

lossi�=�?k�       �	f�tkfc�A�*

loss�x<��
�       �	Ԙukfc�A�*

loss*$�;�z��       �	1vkfc�A�*

loss��<;�       �	��vkfc�A�*

lossJ�z=S�tD       �	�~wkfc�A�*

lossҙ�<	��       �	�xkfc�A�*

loss�.k<���t       �	`�xkfc�A�*

loss_>�<���       �	zSykfc�A�*

lossnZ�=���?       �	9�ykfc�A�*

losse.=U�$Z       �		�zkfc�A�*

loss�._=�3Z�       �	�&{kfc�A�*

loss�,�=Tx��       �	��{kfc�A�*

lossZ6�;W�/v       �	�W|kfc�A�*

loss,q�:X>�Q       �	}�|kfc�A�*

loss��< #�       �	��}kfc�A�*

loss<���       �	o�~kfc�A�*

lossG�;T�vX       �	M1kfc�A�*

lossŰp=��ە       �	��kfc�A�*

loss� C<h��       �	�\�kfc�A�*

lossή=�nO�       �	k�kfc�A�*

lossx��:B�m       �	lρkfc�A�*

loss��8:F$��       �	X��kfc�A�*

lossi��:�-��       �	rM�kfc�A�*

loss;�q:����       �	9�kfc�A�*

loss�d�<kV"�       �	Ϣ�kfc�A�*

lossl^�;{�P       �	3�kfc�A�*

loss�?;��(       �	2ǅkfc�A�*

loss��s<Į��       �	��kfc�A�*

loss7M>�A�       �	��kfc�A�*

loss!3;T��       �	��kfc�A�*

lossv5�>y*�       �	1C�kfc�A�*

loss��;��d�       �	n݈kfc�A�*

loss$�F=muf       �	���kfc�A�*

loss#��<��6       �	mS�kfc�A�*

losso�<��e�       �	�kfc�A�*

loss*�=$c��       �	E��kfc�A�*

loss�9.=%��       �	Z��kfc�A�*

lossa��=#��q       �	j2�kfc�A�*

loss���<���       �	�͍kfc�A�*

loss&�<�^��       �	���kfc�A�*

loss��=���       �	�4�kfc�A�*

losst��=�+�       �	�֏kfc�A�*

loss�#�<�j�       �	_y�kfc�A�*

loss�jS=�`��       �	L�kfc�A�*

loss��=Á�A       �	P�kfc�A�*

loss�-=�m�       �	N�kfc�A�*

loss��/=�z�       �	{g�kfc�A�*

losst�m=*���       �	�	�kfc�A�*

loss�D�;�}�       �	e��kfc�A�*

loss��<8ee{       �	�>�kfc�A�*

loss踬<y���       �	oזkfc�A�*

lossd>=0y       �	_{�kfc�A�*

loss`��:^Gd�       �	��kfc�A�*

lossLA<mC�j       �	��kfc�A�*

loss�f;�(       �	�a�kfc�A�*

loss�=�>�       �	�kfc�A�*

loss��+<&y�       �	���kfc�A�*

loss��=:��       �	=�kfc�A�*

loss[@�=�ۖ|       �	�כkfc�A�*

loss͜[<���       �	Kr�kfc�A�*

loss�(<-_%       �	p�kfc�A�*

loss� <e�%K       �	]��kfc�A�*

loss#Q;}�Z�       �	�@�kfc�A�*

lossl�<��fR       �	�۞kfc�A�*

lossc�N<����       �	: �kfc�A�*

loss��;<�L	       �	���kfc�A�*

loss($�=3�l�       �	Pm�kfc�A�*

loss;ȓ=���g       �	�
�kfc�A�*

lossɑ<NR�       �	#�kfc�A�*

lossܼ/<	�       �	��kfc�A�*

loss�=a=-���       �	6��kfc�A�*

lossNB<U�ZV       �	6��kfc�A�*

loss:��=[��       �	AH�kfc�A�*

loss"�<�s��       �	r��kfc�A�*

loss��=��}       �	�U�kfc�A�*

loss�w>7e"�       �	E�kfc�A�*

loss��;"oA!       �	�,�kfc�A�*

loss��=H��=       �	��kfc�A�*

loss:�g<�͵�       �	l�kfc�A�*

loss���<6xV       �	��kfc�A�*

loss���<�!'       �	��kfc�A�*

loss�A�=�+y       �	
��kfc�A�*

loss`�={J��       �	�A�kfc�A�*

loss&S]==Ջ�       �	*��kfc�A�*

lossO�=��z       �	1~�kfc�A�*

loss.T�<Ȯ�       �	�kfc�A�*

loss�y�=}t�       �	[��kfc�A�*

loss�\<gv�       �	UP�kfc�A�*

loss�~�=�Cދ       �	���kfc�A�*

loss�f�<
�m�       �	��kfc�A�*

loss�'�;�bd       �	6�kfc�A�*

lossR�&=����       �	ǹ�kfc�A�*

loss`�/=s��       �	&m�kfc�A�*

loss�2�=�x5b       �	F�kfc�A�*

loss%!<�'��       �	��kfc�A�*

loss�Db=1��       �	�L�kfc�A�*

loss�y�;$�7       �	h��kfc�A�*

lossw'v=��'�       �	���kfc�A�*

lossSÐ<�'��       �	3�kfc�A�*

loss���=1�t       �	U��kfc�A�*

loss��<��       �	�t�kfc�A�*

lossG�=����       �	��kfc�A�*

loss!��<w�       �	���kfc�A�*

loss,΄=�@R       �	�p�kfc�A�*

loss��;���       �	��kfc�A�*

lossR'�<!RF       �	4��kfc�A�*

loss
@=�[ܔ       �	M��kfc�A�*

lossܩ3=�6n       �	���kfc�A�*

loss�h�<pı�       �	Q1�kfc�A�*

loss���<�~��       �	���kfc�A�*

loss���=�R,�       �	L��kfc�A�*

loss�>�<&���       �	�;�kfc�A�*

loss�=؂&       �	���kfc�A�*

loss��<:���       �	�{�kfc�A�*

lossUF<1�       �	���kfc�A�*

losso�=0m��       �	��kfc�A�*

lossݤ�<)ĽF       �	��kfc�A�*

loss)�T>��I�       �	�P�kfc�A�*

loss%f�=��c�       �	���kfc�A�*

lossOx�=[���       �	��kfc�A�*

lossDְ=�j��       �	5$�kfc�A�*

loss���<|�       �	o��kfc�A�*

loss3��;��       �	�e�kfc�A�*

lossJ��<�}�c       �	i��kfc�A�*

loss�Zb=�:�       �	��kfc�A�*

loss,2`=z^u	       �	>�kfc�A�*

lossdD�<����       �	q�kfc�A�*

loss�1�<�r!E       �	�:�kfc�A�*

lossEY{=����       �	��kfc�A�*

loss2�;.�֊       �	�p�kfc�A�*

lossl<N��       �	(�kfc�A�*

loss�<�)b�       �	���kfc�A�*

lossu��;��tV       �	\r�kfc�A�*

loss�=��q9       �	��kfc�A�*

lossj�<Ƿ       �	��kfc�A�*

loss��<�Ұ       �	�F�kfc�A�*

loss�F;�L+       �	���kfc�A�*

loss*KX;E_j       �	���kfc�A�*

loss,�,<T	�       �	"�kfc�A�*

loss�֯=�Jr       �	���kfc�A�*

loss�v�=�|*=       �	�p�kfc�A�*

loss�ϥ<yEy       �	��kfc�A�*

lossvYQ< �w�       �	ͫ�kfc�A�*

loss�0=n��i       �	��kfc�A�*

loss�W_=n�*I       �	s*�kfc�A�*

loss\��=�O��       �	b��kfc�A�*

loss���<E?��       �	�S�kfc�A�*

loss�}�=�0o)       �	���kfc�A�*

loss, �=3��       �	c{�kfc�A�*

loss��<C��        �	��kfc�A�*

loss��m=*�d       �	v��kfc�A�*

loss�:I<v�t�       �	�=�kfc�A�*

loss�C<M�N       �	���kfc�A�*

loss��<���3       �	~��kfc�A�*

loss�u<ႃ~       �	�(�kfc�A�*

loss� 8<8\!�       �	z��kfc�A�*

loss	a=f�}�       �	c�kfc�A�*

lossػl=. �       �	��kfc�A�*

loss���<��D�       �	q��kfc�A�*

loss��~=��ߜ       �	ܷ lfc�A�*

loss�=�+(�       �	G�lfc�A�*

loss�?c=!R�Z       �	Dlfc�A�*

loss���<�Aa�       �	hxlfc�A�*

loss�J=���       �	�Llfc�A�*

loss]��<��C�       �	��lfc�A�*

loss �<�E%�       �	ܸlfc�A�*

lossq�<
[�e       �	Vlfc�A�*

loss��#=�x
�       �	@�lfc�A�*

loss��E=��       �	��lfc�A�*

loss�'}=�dX�       �	�.lfc�A�*

loss�8=��am       �	��lfc�A�*

losst�=��J�       �	�j	lfc�A�*

loss��q<g�d       �	�
lfc�A�*

loss�<I=����       �	A�
lfc�A�*

loss�D=�< �       �	5@lfc�A�*

loss�W<'�?@       �	�Dlfc�A�*

lossXg=,�i,       �	��lfc�A�*

loss�UM=��       �	�lfc�A�*

loss6�>�6n�       �	lfc�A�*

loss��T<�oJ       �	J�lfc�A�*

loss6F�;���S       �	�Nlfc�A�*

loss�j�<M.       �	��lfc�A�*

loss�S�<E�r       �	��lfc�A�*

lossŠj;+��       �	�lfc�A�*

loss���=���d       �	�lfc�A�*

lossa��=}@[P       �	�[lfc�A�*

loss�!<�8>�       �	A�lfc�A�*

loss/U�<��Q�       �	��lfc�A�*

loss��=�)��       �	"7lfc�A�*

loss��;O`��       �	>�lfc�A�*

loss�\=f�       �	elfc�A�*

lossW�c=)��        �	6lfc�A�*

loss�y~=�~|       �	�lfc�A�*

loss�
�;��$�       �	�Qlfc�A�*

loss)�'=����       �	Y�lfc�A�*

loss�<�;\��       �	�lfc�A�*

loss3&�<�$�b       �	7lfc�A�*

lossc�T=ؕS       �	��lfc�A�*

loss��<�똜       �	~plfc�A�*

loss��Q=� p�       �	�
lfc�A�*

loss)DF=���       �	&�lfc�A�*

lossEf<�窐       �	�Blfc�A�*

lossF�=�eV}       �	��lfc�A�*

loss�*=5�j�       �	�mlfc�A�*

loss���<j+�S       �	�lfc�A�*

loss���=�>�       �	�lfc�A�*

lossF��=��f       �	G9lfc�A�*

loss�	�<y_       �	�lfc�A�*

loss�$U=S�]       �	�n lfc�A�*

lossa��<��!4       �	�
!lfc�A�*

loss�ԛ=E���       �	��!lfc�A�*

loss	�R=s���       �	�D"lfc�A�*

loss��=R��       �	r�"lfc�A�*

loss�ė;����       �	�x#lfc�A�*

loss�O >�"�       �	$lfc�A�*

lossM^	<����       �	Ҩ$lfc�A�*

loss̦�=�@?       �	�%lfc�A�*

loss��;x^6       �	A�&lfc�A�*

loss��<�}��       �	4h'lfc�A�*

loss�"`<V��S       �	t((lfc�A�*

loss��
=}�2?       �	�)lfc�A�*

loss� �<CF}       �	�)lfc�A�*

loss�C=�r�       �	
�*lfc�A�*

lossf=h��       �	�+lfc�A�*

lossΊW=�Vg�       �	u,lfc�A�*

loss�N[<�       �	�A-lfc�A�*

loss��<;���       �	��-lfc�A�*

loss��<�8�J       �	P�.lfc�A�*

loss��7="�/�       �	a�/lfc�A�*

loss�O=��C�       �	E�0lfc�A�*

loss+V�<6��[       �	׽1lfc�A�*

loss��<�N7�       �	�2lfc�A�*

loss�� =�       �	&�3lfc�A�*

loss:��=k#��       �	�"4lfc�A�*

loss�8=L���       �	ؼ4lfc�A�*

lossW�;4T�       �	nl5lfc�A�*

lossZ��<�j4�       �	�6lfc�A�*

loss��=
��       �	�6lfc�A�*

lossc�=&nO�       �	�e7lfc�A�*

loss�G�<�a�       �	�8lfc�A�*

loss�E=>�       �	 �8lfc�A�*

loss7�<<ڲ�       �	aO9lfc�A�*

loss/Sg<s�m       �	� :lfc�A�*

loss-&i;���       �	;�:lfc�A�*

lossJ=g��       �	�`;lfc�A�*

loss%�h<k�t       �	��;lfc�A�*

lossi�$<h�˹       �	Z�<lfc�A�*

loss��=	ZM>       �	@=lfc�A�*

loss�Un<���n       �	i�=lfc�A�*

loss�/;<d��       �	{>lfc�A�*

loss�T=k3I       �	�?lfc�A�*

loss�H�;��       �	p�?lfc�A�*

loss��;�zp�       �	'K@lfc�A�*

loss�2<L�       �	��@lfc�A�*

loss���<�Q��       �	؀Alfc�A�*

loss�r<[6�       �	�%Blfc�A�*

loss�/�<t���       �	��Blfc�A�*

lossI#.=�v��       �	�xClfc�A�*

loss�L`=@4D�       �	>>Dlfc�A�*

lossߘL<^�Fo       �	��Dlfc�A�*

loss�7�; ʱ�       �	9{Elfc�A�*

loss�}<��[�       �	�Flfc�A�*

loss�Q<����       �	��Flfc�A�*

loss�l�<)m�       �	�SGlfc�A�*

lossD�N<�c/       �	0�Glfc�A�*

loss�]<�Px       �	�Hlfc�A�*

loss6��=^8       �	�6Ilfc�A�*

loss�`�<�B�W       �	-�Klfc�A�*

loss� �<��)       �	M,Llfc�A�*

loss��+=�rO       �	��Llfc�A�*

loss�w/=݌�       �	�XMlfc�A�*

loss�vz<$�       �	uNlfc�A�*

loss�@w<��{<       �	�Nlfc�A�*

loss<1<����       �	fLOlfc�A�*

loss}�;�F�       �	��Olfc�A�*

lossn��<>�c�       �	t�Plfc�A�*

lossU��<ݼ�S       �	!<Qlfc�A�*

lossK�<�-�       �	��Qlfc�A�*

loss#2=9�x�       �	�xRlfc�A�*

lossx�6=�{~�       �	jSlfc�A�*

loss��=K~.       �	ٯSlfc�A�*

loss���=A���       �	�VTlfc�A�*

loss��g<&�       �	��Tlfc�A�*

loss$=�e�       �	�Ulfc�A�*

lossF�;҂
�       �	�5Vlfc�A�*

loss*[�;�XU       �	�Vlfc�A�*

loss�"�=W`��       �	;qWlfc�A�*

loss!3�=���       �	�Xlfc�A�*

loss�*�<xc       �	�Xlfc�A�*

loss�8;ь�m       �	7Ylfc�A�*

loss��w<�-       �	��Ylfc�A�*

loss.�=g��       �	2uZlfc�A�*

loss�]�<g�;       �	L[lfc�A�*

loss- <4�r       �	A�[lfc�A�*

loss��=-���       �	�N\lfc�A�*

loss-0m=<<Y�       �	�\lfc�A�*

loss�*0>��"       �	��]lfc�A�*

loss��R=q���       �	((^lfc�A�*

loss���=��[       �	�^lfc�A�*

lossd	�<�!�       �	/m_lfc�A�*

loss��e=i���       �	)`lfc�A�*

loss��=��       �	ߤ`lfc�A�*

loss:�Z=&���       �	�Balfc�A�*

loss�d�=M�       �	P�alfc�A�*

loss�H�;Կ��       �	zblfc�A�*

loss2�<��8       �	fclfc�A�*

loss���=4�zt       �	i�clfc�A�*

loss6��<g3x       �	SAdlfc�A�*

loss�=��}       �	�dlfc�A�*

loss��=5%       �	Xtelfc�A�*

lossFߕ<�i�       �	�flfc�A�*

loss�j�<6�2       �	�Jglfc�A�*

loss&,�<[n�7       �	T;hlfc�A�*

loss���=Zp�G       �	�!ilfc�A�*

loss�=�q�<       �	+�ilfc�A�*

loss��x<_��m       �	�Wjlfc�A�*

loss��=�~-W       �	#klfc�A�*

lossr��<^�S^       �	i�klfc�A�*

lossj��<o�x�       �	O>llfc�A�*

lossN��;q�K       �	��llfc�A�*

loss?�=t��       �	�rmlfc�A�*

loss���<�l��       �	nnlfc�A�*

lossNh=��       �	4�nlfc�A�*

loss:x�<�(��       �	�Tolfc�A�*

loss���<b�A�       �	:�olfc�A�*

loss:�<h       �	�xplfc�A�*

loss�x�<o Pn       �	�
qlfc�A�*

loss��Y<��?/       �	ūqlfc�A�*

loss
��<ȱeM       �	Jrlfc�A�*

loss!=!(��       �	�rlfc�A�*

loss��`=�J6�       �	{slfc�A�*

loss��<* �R       �	ztlfc�A�*

losspc=@��@       �	�tlfc�A�*

loss�ȧ<HB       �	�Mulfc�A�*

lossffe<���Z       �	��ulfc�A�*

loss��=���       �	Nzvlfc�A�*

lossI�<�EA       �	swlfc�A�*

loss�{�<�⏷       �	��wlfc�A�*

loss���<@���       �	@Oxlfc�A�*

lossa�e=b��s       �	}�xlfc�A�*

loss�=���a       �	�ylfc�A�*

lossׁ�;{�k       �	U.zlfc�A�*

lossi��<]ܞ       �	��zlfc�A�*

loss���<���o       �	0g{lfc�A�*

loss)��<IP�       �	��}lfc�A�*

loss�;8@�
       �	J�~lfc�A�*

loss�|=wAf�       �	�olfc�A�*

loss���<4q��       �	��lfc�A�*

loss�_�=�=�       �	J�lfc�A�*

loss�<UY�       �	���lfc�A�*

loss��A<�ѥ�       �	�U�lfc�A�*

loss�`=knW       �	ᗃlfc�A�*

loss��6=�9�2       �	V�lfc�A�*

loss	T"<<�>�       �	�3�lfc�A�*

loss�)O>�*c       �	�Ņlfc�A�*

losso�=��       �	?ǆlfc�A�*

loss@��<���       �	<��lfc�A�*

loss��<�ì       �	=e�lfc�A�*

loss�R�<'Ov;       �	_\�lfc�A�*

loss*�s=$���       �	���lfc�A�*

loss�Q�:8	��       �	T5�lfc�A�*

loss�"y<Wk       �	�؋lfc�A�*

loss` =^�'M       �	~�lfc�A�*

loss��<���o       �	���lfc�A�*

lossj�x=�       �	&T�lfc�A�*

lossC��=W��r       �	f��lfc�A�*

loss!�s=c��       �	M��lfc�A�*

loss�W�<�1�\       �	E�lfc�A�*

lossW>=�Hm       �	���lfc�A�*

loss��1<�@�       �	5��lfc�A�*

loss��6<j"��       �	=�lfc�A�*

loss/�=�g0�       �	Yڒlfc�A�*

loss��K;��K       �	�u�lfc�A�*

lossSn�;*:n�       �	��lfc�A�*

loss�Ay<<���       �	9�lfc�A�*

loss&��=��V       �	1��lfc�A�*

loss�G<� rn       �	?R�lfc�A�*

loss�C?=�YJP       �	;�lfc�A�*

loss�W=.v��       �	���lfc�A�*

lossp=��       �	�7�lfc�A�*

loss��]<6.��       �	�Ҙlfc�A�*

loss��;���       �	�n�lfc�A�*

loss@��<t�,       �	��lfc�A�*

loss��>}�}       �	���lfc�A�*

loss.\:=}k�       �	�O�lfc�A�*

loss�>�V$�       �	���lfc�A�*

loss��i<�q9E       �	���lfc�A�*

loss��b=C"xe       �	�.�lfc�A�*

lossJ��<N�p�       �	nĝlfc�A�*

loss��;nȈn       �	<h�lfc�A�*

loss� W=l�&h       �	���lfc�A�*

loss�&@=Hwn�       �	��lfc�A�*

loss�TM=q��       �	)�lfc�A�*

loss�X<�l=       �	i��lfc�A�*

loss�)=��       �	M�lfc�A�*

loss]��=��H�       �	S�lfc�A�*

loss���;�l[�       �	=��lfc�A�*

loss��8=�Y       �	�lfc�A�*

loss�
�=��c       �	��lfc�A�*

loss |�<���       �	�E�lfc�A�*

loss�F�<X[#t       �	{�lfc�A�*

lossX!=�+*�       �	+m�lfc�A�*

loss`1=t:��       �	N�lfc�A�*

loss%G(=��`�       �	�f�lfc�A�*

lossNuQ<BDk       �	�lfc�A�*

loss�r�=0��.       �	���lfc�A�*

lossi�<!���       �	�z�lfc�A�*

loss��;�I�       �	� �lfc�A�*

loss@R=�q       �	I��lfc�A�*

lossRE�=;^�       �	�z�lfc�A�*

loss���;��y3       �	=�lfc�A�*

losse�j<�Rw�       �	���lfc�A�*

loss��;ا�_       �	 ]�lfc�A�*

loss}�I<�       �	l
�lfc�A�*

lossAr{=ֶ       �	��lfc�A�*

loss�}K=�k�i       �	+K�lfc�A�*

lossf�>=��|       �	N�lfc�A�*

loss6��<CK�~       �	���lfc�A�*

loss�KC=��       �	�%�lfc�A�*

lossd,�;J)��       �	PĴlfc�A�*

loss,,,=�!3       �	�c�lfc�A�*

loss���<<A       �	��lfc�A�*

loss��>=�'�       �	㩶lfc�A�*

loss��=Z�W5       �	�T�lfc�A�*

loss���=�u��       �	��lfc�A�*

loss�� >
�3�       �	���lfc�A�*

loss��h<�r��       �	<1�lfc�A�*

lossV	�=�Z\�       �	5ӹlfc�A�*

loss�GL=�ǭ       �	�w�lfc�A�*

loss��/=.�k�       �	��lfc�A�*

loss$"<8+�       �	ȳ�lfc�A�*

loss�v<N�HB       �	�L�lfc�A�*

loss=ġ�}       �	��lfc�A�*

loss�~�<���       �	c��lfc�A�*

lossԒ�=s-�       �	`��lfc�A�*

loss�\/<u��M       �	e4�lfc�A�*

loss���<O�z(       �	���lfc�A�*

loss(G<C�(�       �	�k�lfc�A�*

loss��=&P_=       �	|�lfc�A�*

loss��=�k~�       �	��lfc�A�*

loss3� <�2�       �	rM�lfc�A�*

loss|=�@�       �	���lfc�A�*

loss_u�<��@�       �	_�lfc�A�*

loss=��=�       �	{/�lfc�A�*

loss&��<�:Gd       �	���lfc�A�*

loss)�:=��1       �	�m�lfc�A�*

loss��<^�ض       �	n�lfc�A�*

loss��\<~�Y       �	,��lfc�A�*

loss�T3<�U��       �	�Z�lfc�A�*

lossW�<}�-s       �	��lfc�A�*

lossX��<4]�l       �	���lfc�A�*

loss;�)<��4       �	:;�lfc�A�*

loss�ܚ=؎g�       �	���lfc�A�*

lossR�U=I<v�       �	fi�lfc�A�*

loss(�=5�oL       �	�v�lfc�A�*

lossq(&=ye�z       �	�%�lfc�A�*

loss|> t�9       �	��lfc�A�*

loss��=��@       �	���lfc�A�*

loss_@}<]���       �	Ih�lfc�A�*

loss,O%=w�       �	�
�lfc�A�*

loss���<��T       �	G��lfc�A�*

loss��=V}E�       �	�P�lfc�A�*

loss/��=�=       �	a��lfc�A�*

loss=Z=�/�       �	�L�lfc�A�*

lossiDY;�M��       �	=��lfc�A�*

loss�+�=       �	Ԙ�lfc�A�*

loss�T==�a       �	p>�lfc�A�*

loss,t=��59       �	n��lfc�A�*

loss�b<(��       �	���lfc�A�*

loss�{<F-m       �	[z�lfc�A�*

loss���<���a       �	�!�lfc�A�*

loss=�~<��t�       �	j��lfc�A�*

lossR�<�z$�       �	�a�lfc�A�*

loss�A�<*�c�       �	q�lfc�A�*

loss�k <T#�       �	t��lfc�A�*

loss^=��X�       �	�U�lfc�A�*

loss��6<��       �	��lfc�A�*

loss=�<Rm��       �	'��lfc�A�*

lossi�E<�9V       �	!�lfc�A�*

loss
=�H�8       �	���lfc�A�*

lossү�;�q        �	QN�lfc�A�*

loss���<WMJ�       �	��lfc�A�*

loss��=ʂ�4       �	I��lfc�A�*

loss�m�<��)�       �	��lfc�A�*

lossxZ�=6�1       �	��lfc�A�*

loss�,L<�ዑ       �	�H�lfc�A�*

loss�#=3B       �	i��lfc�A�*

loss��;|�G       �	�v�lfc�A�*

lossT�<��       �	��lfc�A�*

lossS&�<;}        �	���lfc�A�*

loss7]N=fI��       �	Ie�lfc�A�*

loss��=�Qi�       �	i��lfc�A�*

lossWq�=�~o       �	��lfc�A�*

loss.��=�Fr       �	�,�lfc�A�*

losse�=��&�       �	��lfc�A�*

loss� �=$PH�       �	SY�lfc�A�*

loss�:k�^�       �	���lfc�A�*

loss,�<�0�       �	"��lfc�A�*

loss��o<���       �	2 �lfc�A�*

loss�= 퉽       �	}��lfc�A�*

loss:6�<��       �	<J�lfc�A�*

loss��=��u�       �	%��lfc�A�*

lossZ��<��C,       �	�g�lfc�A�*

loss��?=�4��       �	z��lfc�A�*

loss�e�=;�       �	���lfc�A�*

loss6=�=L�2J       �	cD�lfc�A�*

loss
��<�Y�2       �	'��lfc�A�*

loss�X=��x       �	@k�lfc�A�*

loss�H�;jP       �	��lfc�A�*

lossER*=�H�5       �	��lfc�A�*

loss���<֚y       �	77�lfc�A�*

loss��F=]5��       �	G��lfc�A�*

loss�%]<��{       �	J_�lfc�A�*

lossd�L=�%4�       �	/��lfc�A�*

loss�45=�M�p       �	2��lfc�A�*

loss�=O<�r       �	)&�lfc�A�*

loss�7u<���4       �	��lfc�A�*

loss
R�<0猎       �	�N�lfc�A�*

loss��;���       �	���lfc�A�*

loss�H�;;C�       �	�{�lfc�A�*

loss|H�=�G
       �	��lfc�A�*

lossh?�=���       �	p��lfc�A�*

loss+�=�ϴ       �	�I�lfc�A�*

loss�}>�r��       �	���lfc�A�*

loss �T<��OT       �	|�lfc�A�*

loss�*<�3o       �	w�lfc�A�*

loss_!=�v��       �	;��lfc�A�*

loss`�=�x~�       �	.<�lfc�A�*

loss�1V=$�f       �	;�lfc�A�*

loss-�8=�d�       �	d��lfc�A�*

loss�L'=��u       �	�K�lfc�A�*

loss��=��+�       �	r��lfc�A�*

loss}�<�*�       �	 mfc�A�*

lossa�=Q�       �	�mfc�A�*

lossT�F=�x�       �	�[mfc�A�*

loss��<��gV       �	Q�mfc�A�*

loss��?=ϥ�       �	ٙmfc�A�*

loss<J=4��       �	=`mfc�A�*

loss[��<d�2`       �	��mfc�A�*

loss
�v=�&       �	طmfc�A�*

loss�#�=W��       �	.Ymfc�A�*

lossȡ�<^��O       �	��mfc�A�*

lossƉ><�W7       �	mfc�A�*

lossݭ>=����       �	v6mfc�A�*

loss13�<Q{
*       �	S�mfc�A�*

losse�I<�W�	       �	�m	mfc�A�*

loss�x<�f(Z       �	@
mfc�A�*

loss�G�<K��       �	��
mfc�A�*

loss�=��l       �	Omfc�A�*

lossoZ�;��ut       �	O�mfc�A�*

loss7�=7�5       �	ҋmfc�A�*

loss�P�<~wE       �	,mfc�A�*

loss4=�       �	��mfc�A�*

loss�;�=��U�       �	�`mfc�A�*

loss��;q7       �	��mfc�A�*

loss힐<��R�       �	��mfc�A�*

loss���:\���       �	 �mfc�A�*

loss=�g<�v�u       �	�Smfc�A�*

lossI<��K       �	Q�mfc�A�*

loss�ø=��d&       �	N�mfc�A�*

lossVm�=�;��       �	�;mfc�A�*

loss�+n=-�X       �	��mfc�A�*

loss�Ơ;KM�       �	�rmfc�A�*

loss�R:;Ŗ�       �	Zmfc�A�*

loss��=�U�Z       �	Թmfc�A�*

loss%V=�^˵       �	�\mfc�A�*

loss���:0�)3       �	dmfc�A�*

loss�;=AM�       �	��mfc�A�*

lossH6<qG       �	(Dmfc�A�*

loss�'=Bh�       �	^�mfc�A�*

lossU}�<�{G       �	"qmfc�A�*

loss}�<�G�L       �	kmfc�A�*

loss;6�;d�g�       �	��mfc�A�*

lossۧ�=�=�       �	�>mfc�A�*

loss%��<1���       �	g�mfc�A�*

losss�<`+�       �	�hmfc�A�*

loss%��=w<�       �	Omfc�A�*

loss��<�m�       �	�mfc�A�*

loss�C�<q���       �	u?mfc�A�*

loss��=�S��       �	X� mfc�A�*

loss�n�< ���       �	6!mfc�A�*

loss���:d�+�       �	��!mfc�A�*

loss]�:<l�/B       �	ut"mfc�A�*

lossE"k<�w[`       �	�#mfc�A�*

loss�C=O���       �	8�#mfc�A�*

loss,*w=R�c�       �	�i$mfc�A�*

loss.7�=��7       �	�%mfc�A�*

loss�@<�*��       �	��%mfc�A�*

lossE�<�$��       �	Lk&mfc�A�*

lossE86=��{@       �	f�'mfc�A�*

loss�(�<3192       �	)>(mfc�A�*

loss�3=.�+�       �	��(mfc�A�*

loss��U<\�       �	k�)mfc�A�*

loss��=+��-       �	�*mfc�A�*

loss ��<כ�R       �	0�*mfc�A�*

loss1��<_��       �	�a+mfc�A�*

loss ߹<a̭V       �	��+mfc�A�*

loss��i=���:       �	N�,mfc�A�*

loss��<���       �	~9-mfc�A�*

loss1��<�	��       �	��-mfc�A�*

loss;�;L���       �	�2/mfc�A�*

loss��<6�ް       �	G�/mfc�A�*

loss2=K;�4�]       �	/i0mfc�A�*

loss� !<�`
U       �	
1mfc�A�*

loss��;��Z�       �	��1mfc�A�*

loss�%<��=p       �	d[2mfc�A�*

lossW��<D-��       �	��2mfc�A�*

lossm�=C2�"       �	k�3mfc�A�*

loss3>�4�^       �	U4mfc�A�*

loss���;sٞ       �	��4mfc�A�*

loss�@�<(s��       �	WB5mfc�A�*

loss���<KE��       �	��5mfc�A�*

loss�[�:BL�       �	vn6mfc�A�*

loss:�8����       �	�7mfc�A�*

lossl�:1O��       �	<�7mfc�A�*

loss{��:�N       �	�88mfc�A�*

loss���;8/�K       �	A�8mfc�A�*

loss�;$�"�       �	Ct9mfc�A�*

loss˟9Yp�L       �	�:mfc�A�*

lossW�g<�CI       �	��:mfc�A�*

loss���;2��?       �	�X;mfc�A�*

loss��8:rB�R       �	��;mfc�A�*

loss��;9�:�       �	z�<mfc�A�*

lossæ~;x�d�       �	�-=mfc�A�*

losscM={�3T       �	!�=mfc�A�*

loss��<���K       �	R`>mfc�A�*

lossW�4:��r`       �	a�>mfc�A�*

lossx��=����       �	��?mfc�A�*

loss�T=>� �       �	b.@mfc�A�*

loss體;�J=       �	��@mfc�A�*

loss6��=M8_3       �	ʋAmfc�A�*

lossh�=z�m�       �	�$Bmfc�A�*

lossX�=��`       �	��Bmfc�A�*

loss�= ��       �	�eCmfc�A�*

lossG�<I���       �	��Cmfc�A�*

loss�6O=y��       �	K�Dmfc�A�*

lossz��<�i�:       �	Q�Emfc�A�*

loss��H<��X�       �	�WFmfc�A�*

loss�,�<��       �	��Gmfc�A�*

loss��+=>H��       �	�'Hmfc�A�*

loss�=o3D       �	��Hmfc�A�*

loss]�j=9��D       �	�[Imfc�A�*

lossۻ�<B$�
       �	|)Kmfc�A�*

lossiL<��Uk       �	,Lmfc�A�*

lossW=2"6       �	�Lmfc�A�*

loss Kb<'�`5       �	jKMmfc�A�*

loss�i=�B�       �	��Mmfc�A�*

loss�wc=���       �	z�Nmfc�A�*

loss�4�<��-�       �	�*Omfc�A�*

loss�n(<�#c       �	��Omfc�A�*

loss]�<��,M       �	pPmfc�A�*

loss��<�%�j       �	�Qmfc�A�*

loss���<8_�9       �	.�Qmfc�A�*

loss䕤;���       �	�BRmfc�A�*

loss$��;۫Ӯ       �	�Smfc�A�*

loss!DR=&�t       �	ͭSmfc�A�*

loss}�{<��;2       �	�ETmfc�A�*

loss�Y|=o��/       �	��Tmfc�A�*

loss4��=�s[       �	�rUmfc�A�*

loss�9;<�`��       �	�Vmfc�A�*

loss�r(=.���       �	��Vmfc�A�*

loss� �=�g
�       �	�GWmfc�A�*

loss�V<qL��       �	�Wmfc�A�*

loss�=I<r�9�       �	�Xmfc�A�*

loss�h~<�Iqw       �	eYmfc�A�*

loss�{<g�<       �	��Ymfc�A�*

loss�x=A��\       �	~RZmfc�A�*

loss��m=2��
       �	R�Zmfc�A�*

loss���=�%a�       �	��[mfc�A�*

loss մ<G�:       �	�\mfc�A�*

loss�G�;	���       �	�\mfc�A�*

loss��=��y�       �	�F]mfc�A�*

loss��#=Z��       �	��]mfc�A�*

lossQ'M<[��       �	Q�^mfc�A�*

loss$�p=�R3}       �	P_mfc�A�*

loss�=�7[l       �	��_mfc�A�*

loss
J<�q�       �	�R`mfc�A�*

loss��4=Շ�       �	J�`mfc�A�*

lossT(:35o       �	M�amfc�A�*

loss���<�x?�       �	ebmfc�A�*

loss	O<�પ       �	�}mfc�A�*

loss/�=�\J�       �	B!~mfc�A�*

loss�M=�Q��       �	��~mfc�A�*

loss��=,l
g       �	Amfc�A�*

loss��<\�b�       �	��mfc�A�*

lossr�<��)       �	�c�mfc�A�*

lossqtA<ɉd�       �	c	�mfc�A�*

lossl�l=����       �	���mfc�A�*

loss)�-=�ښ0       �	w��mfc�A�*

loss)�=���       �	�7�mfc�A�*

loss��;^ṋ       �	cԃmfc�A�*

lossA�;T�s�       �	y�mfc�A�*

loss���<.��       �	F�mfc�A�*

lossxQ�=	"D       �	Ҩ�mfc�A�*

loss�7p<)~��       �	�@�mfc�A�*

loss3�<�L       �	Dކmfc�A�*

loss��f:�7�       �	���mfc�A�*

loss�<$��       �	]4�mfc�A�*

loss'�<P*a*       �	1E�mfc�A�*

loss7(�=&�Y       �	�މmfc�A�*

lossF=�}�       �	c~�mfc�A�*

loss��/=�d./       �	{�mfc�A�*

loss��%=��       �	V�mfc�A�*

loss]��=���       �	���mfc�A�*

loss��0<�:�       �	X�mfc�A�*

loss��<no?I       �	� �mfc�A�*

loss�d�<U�       �	HÏmfc�A�*

loss:�<�5�       �	W̐mfc�A�*

loss)v*=�b�       �	�e�mfc�A�*

loss��)<d'�       �	���mfc�A�*

lossV� <:�`       �	���mfc�A�*

loss�(�;����       �	�@�mfc�A�*

lossX��<��/       �	�mfc�A�*

loss�<=@�H�       �	���mfc�A�*

loss�P<
�m       �	-`�mfc�A�*

loss�ex=����       �	���mfc�A�*

lossD1<X��       �	A��mfc�A�*

loss8�<�53�       �	�4�mfc�A�*

loss�$>�K0       �	�ٗmfc�A�*

lossJ�=�,+�       �	�u�mfc�A�*

loss|��<w4��       �	�mfc�A�*

loss.��=[oOV       �	���mfc�A�*

loss��;��8       �	P�mfc�A�*

loss\�/=��"�       �	W�mfc�A�*

loss��;S���       �	���mfc�A�*

loss�B=��^       �	�mfc�A�*

loss�<�=Mn       �	���mfc�A�*

loss)_=|-��       �	�J�mfc�A�*

loss�1�;9'1�       �	*�mfc�A�*

loss\�<�[�       �	��mfc�A�*

loss��;M;ß       �	���mfc�A�*

lossL�y<y���       �	�0�mfc�A�*

loss?:�;�({�       �	��mfc�A�*

loss9�=�(��       �	Ό�mfc�A�*

lossT�=H�%       �	c*�mfc�A�*

loss=<��1]       �	�΢mfc�A�*

losslmx<���       �	�n�mfc�A�*

lossCs�:���       �	s*�mfc�A�*

lossO�=�1�       �	yʤmfc�A�*

loss$�k=�P1y       �	�q�mfc�A�*

loss��=���       �	��mfc�A�*

loss2��<o��       �	��mfc�A�*

lossN!B;�'�       �	�^�mfc�A�*

loss:�=(��}       �	��mfc�A�*

loss2,#==�+       �	��mfc�A�*

loss�S�<����       �	5c�mfc�A�*

lossxǤ<��$�       �	��mfc�A�*

lossf1-<�\�:       �	t��mfc�A�*

loss��f=�qe       �	�Q�mfc�A�*

loss|h=���       �	]��mfc�A�*

loss�s[<g��c       �	ᗬmfc�A�*

lossWa< �k�       �	�1�mfc�A�*

loss�P�;e��?       �	|խmfc�A�*

loss��=�x�	       �	3n�mfc�A�*

loss�ֿ<�f       �	q�mfc�A�*

loss��<�+�>       �	���mfc�A�*

loss�!e<oz��       �	�@�mfc�A�*

loss�)�;����       �	�mfc�A�*

loss�4�;�GZ       �	Ov�mfc�A�*

lossfP�<��	       �	��mfc�A�*

lossA�<<       �	���mfc�A�*

loss�T<>	w       �	#N�mfc�A�*

lossA
�<��LN       �	��mfc�A�*

loss�J�:v��j       �	]��mfc�A�*

loss�O=����       �	+��mfc�A�*

loss(U";5�-       �	CV�mfc�A�*

loss� *=N���       �	��mfc�A�*

loss�'�=�)p�       �	��mfc�A�*

loss�`�;�C$       �	�#�mfc�A�*

loss]�+=t�5       �	o��mfc�A�*

loss��=����       �	wd�mfc�A�*

lossLl�<��%       �	��mfc�A�*

loss��F;:n3       �	8��mfc�A�*

loss*`�<ʑ�H       �	�F�mfc�A�*

loss�t=���       �	��mfc�A�*

loss@�;t�6U       �	v��mfc�A�*

loss]��=Q��       �	S �mfc�A�*

loss�I�;ň�)       �	�ýmfc�A�*

lossÎ2=�V�       �	�˾mfc�A�*

loss��<��_q       �	�o�mfc�A�*

loss��b;�
�       �	s�mfc�A�*

lossw��;���       �	�mfc�A�*

lossx��;�$��       �	J��mfc�A�*

loss���<]O\       �	�R�mfc�A�*

lossL+�<�^�N       �	^-�mfc�A�*

loss�1�<G�l       �	��mfc�A�*

loss܏�;�]�J       �	�O�mfc�A�*

loss�@!<G���       �	�$�mfc�A�*

lossf�=S�1       �	��mfc�A�*

loss%T.<���       �	��mfc�A�*

losscW==#       �	���mfc�A�*

loss=4�<��$�       �	���mfc�A�*

loss��3=^�       �	cA�mfc�A�*

loss�c�:���O       �	��mfc�A�*

lossoQ�=c�[       �	3��mfc�A�*

loss���=�x�r       �	�h�mfc�A�*

loss��Y='��       �	�n�mfc�A�*

loss1�C=$%2�       �	}�mfc�A�*

loss�%�<E8�       �	���mfc�A�*

loss� D= C#       �	ӡ�mfc�A�*

loss��<���       �	�P�mfc�A�*

loss�nb=ό=_       �	���mfc�A�*

lossuv<�)7       �	Y��mfc�A�*

losss"(=�z�       �	�L�mfc�A�*

loss�-�;v{��       �	4H�mfc�A�*

loss.e_;p9�2       �	���mfc�A�*

loss�l8=��       �	|��mfc�A�*

loss�C�;�%�       �	'O�mfc�A�*

loss�e�<�z �       �	[�mfc�A�*

loss�l�<����       �	���mfc�A�*

loss[�t<{��6       �	=a�mfc�A�*

loss��=PjBw       �	{�mfc�A�*

lossA�!=k�$       �	���mfc�A�*

loss�a;=%�       �	`�mfc�A�*

loss�q�<�       �	��mfc�A�*

loss�%=ρz�       �	3��mfc�A�*

loss3%�;�^k       �	�n�mfc�A�*

loss*s	<�8�       �	��mfc�A�*

loss/�=�^�       �	���mfc�A�*

loss�H<t�Z       �	6r�mfc�A�*

loss}/�;J�#�       �	��mfc�A�*

lossT��<O;q�       �	���mfc�A�*

loss�(�=p)F       �	�8�mfc�A�*

loss`6�;�^��       �	n��mfc�A�*

loss��)=�i��       �	���mfc�A�*

loss&S�<T��
       �	/��mfc�A�*

loss��<�?�       �	*�mfc�A�*

lossXI�;[h�       �	k��mfc�A�*

loss_(�<4�%       �	���mfc�A�*

loss�4=ؓ�s       �	�(�mfc�A�*

loss�a_<$��y       �	���mfc�A�*

loss�<$�       �	�i�mfc�A�*

loss�#<O��       �	��mfc�A�*

loss>�R[�       �	��mfc�A�*

lossiȢ=tO�/       �	RC�mfc�A�*

loss���:Q�*�       �	>��mfc�A�*

loss\��<�n       �	���mfc�A�*

loss��<&�V       �	�(�mfc�A�*

loss��E<g���       �	��mfc�A�*

lossC��=|T��       �	V��mfc�A�*

loss
�(<�S�       �	:;�mfc�A�*

lossn�o<@�_3       �	��mfc�A�*

loss�w�<N�|�       �	���mfc�A�*

lossM\.<D       �	�I�mfc�A�*

loss�s�=`��       �	���mfc�A�*

lossf	<�h��       �	���mfc�A�*

loss��=�dO�       �	�$�mfc�A�*

loss��<a%5       �	���mfc�A�*

loss���<��P       �	�T�mfc�A�*

lossE�|<���z       �	���mfc�A�*

loss칪<&�&       �	���mfc�A�*

loss���;�DP�       �	�,�mfc�A�*

loss��;l���       �	���mfc�A�*

loss�KI=��k&       �	�d�mfc�A�*

lossB��;H#�       �	J
�mfc�A�*

loss=�=0&k@       �	��mfc�A�*

loss�$&=�p��       �	�]�mfc�A�*

lossV��=�>T<       �	��mfc�A�*

loss��=��Q�       �	)��mfc�A�*

loss6�;$�e       �	_'�mfc�A�*

loss��:&w�       �	���mfc�A�*

lossA4=?z�       �	��mfc�A�*

loss��;�j:       �	���mfc�A�*

loss��<�+�       �	�X�mfc�A�*

loss�w�:u���       �	��mfc�A�*

loss��<n̕       �	V��mfc�A�*

loss��=��{�       �	�\ nfc�A�*

lossa��;���	       �	s� nfc�A�*

loss��<���       �	.�nfc�A�*

lossZ�3=��T�       �	WAnfc�A�*

loss��(=Z7�{       �	��nfc�A�*

loss6%<bV�       �	=�nfc�A�*

loss��G<"��       �	�nfc�A�*

loss��<y�       �	,�nfc�A�*

lossI�9<'��\       �	�nfc�A�*

loss��<c��       �	\Ynfc�A�*

loss�c�=�U2f       �	
�nfc�A�*

loss�$=J�-       �	��nfc�A�*

loss��=]��y       �	��nfc�A�*

loss׿.;4��       �	��	nfc�A�*

loss7i�<U�I<       �	v�
nfc�A�*

loss���;���       �	�[nfc�A�*

loss��l<gRs.       �	0�nfc�A�*

lossNaM= M��       �	߈nfc�A�*

lossҩ�<��       �	�!nfc�A�*

lossN�<��*       �	<�nfc�A�*

loss�5=��TP       �	�Unfc�A�*

loss�Y�=�       �	J�nfc�A�*

loss�vo<f��       �	ςnfc�A�*

loss���<5�R�       �	�nfc�A�*

lossA3�<b$��       �	!�nfc�A�*

loss�i<��T�       �	�Cnfc�A�*

lossvk;Pm��       �	��nfc�A�*

loss��H=Z���       �	^�nfc�A�*

loss7z�<n��       �	�6nfc�A�*

loss���=�_�       �	��nfc�A�*

loss�H]>yk��       �	gnfc�A�*

lossh��=�~i       �	c	nfc�A�*

loss���=k�@       �	��nfc�A�*

loss*Б=���'       �	"8nfc�A�*

loss|�<�(p       �	B�nfc�A�*

lossp�=�       �	Qknfc�A�*

lossEmA<3j:|       �	�nfc�A�*

loss% M=�%��       �	�nfc�A�*

lossy\<%�M�       �	�`nfc�A�*

lossZ��;-�p       �	nfc�A�*

loss\
%=0��       �	�nfc�A�*

loss�W�<��0�       �	�Vnfc�A�*

loss��<��p�       �	Z�nfc�A�*

loss���<Tb�7       �	~�nfc�A�*

loss�G�<Q�W�       �	*nfc�A�*

loss�1=;�%�       �	�nfc�A�*

lossϩ�=_4�       �	�Wnfc�A�*

loss��X<��Ds       �	��nfc�A�*

lossk��=��f       �	Ŏnfc�A�*

loss���<�A       �	o+ nfc�A�*

loss�_J<�CFE       �	u� nfc�A�*

loss_�><P���       �		o!nfc�A�*

loss ��<|'�       �	^"nfc�A�*

loss��;�|6�       �	�^#nfc�A�*

lossn�=�
ޞ       �	n�#nfc�A�*

loss�t,=Wk�l       �	w�$nfc�A�*

loss�9�<C�г       �	�?%nfc�A�*

loss;�	=9���       �	��%nfc�A�*

loss8Lr<�`�       �	�j&nfc�A�*

loss�A�<�J       �	
�'nfc�A�*

loss1��;��T�       �	
M(nfc�A�*

lossN8�<���       �	")nfc�A�*

loss߃�=}�H       �	w�)nfc�A�*

loss|��<�bUH       �	{�*nfc�A�*

loss6�t<�!I�       �	!+nfc�A�*

loss�c=y=8�       �	�L,nfc�A�*

loss�=��$�       �	��,nfc�A�*

loss,�;��p�       �	�-nfc�A�*

loss���;�e;�       �	�C.nfc�A�*

loss���<�_B�       �	�/nfc�A�*

lossM�\=��W       �	k�/nfc�A�*

loss���<��       �	�{0nfc�A�*

loss݌=V�1       �	I+1nfc�A�*

lossȐ�;��i       �	u�1nfc�A�*

loss��;P��       �	�^2nfc�A�*

loss�(�;�JcE       �	3nfc�A�*

lossx�k=�#�       �	��3nfc�A�*

loss�w;�N       �	l^4nfc�A�*

loss��$=��P       �	�4nfc�A�*

lossq9<�j�       �	�G6nfc�A�*

loss�q/<}��       �	��6nfc�A�*

loss^�=��4       �	�7nfc�A�*

lossiY5=/�e�       �	��8nfc�A�*

loss�?�=n�f?       �	�:nfc�A�*

loss=��<����       �	g�:nfc�A�*

loss�X�<~���       �	�`;nfc�A�*

loss*�=��M\       �	* <nfc�A�*

lossiR?=L��~       �	��<nfc�A�*

loss�f�='ga"       �	]>nfc�A�*

lossmY5=x�N       �	+�>nfc�A�*

loss��o;O'�`       �	�o?nfc�A�*

loss׫<٤PD       �	�@nfc�A�*

loss�t�=`��[       �	��@nfc�A�*

lossVI<Zl�       �	��Anfc�A�*

loss	�;�X       �	��Bnfc�A�*

loss(�w<�a�       �	�Cnfc�A�*

loss$;�X       �	��Dnfc�A�*

loss�_Y=�5�>       �	:�Enfc�A�*

losszI=·N       �	�Fnfc�A�*

loss==�og�       �	��Gnfc�A�*

loss��=ą��       �	��Hnfc�A�*

loss�
<�z:�       �	W�Infc�A�*

lossZb<:L�       �	.�Jnfc�A�*

loss1�<�Ec       �	'NLnfc�A�*

loss�`v<\�       �	#Mnfc�A�*

loss���;eb��       �	�$Nnfc�A�*

loss���;.r�       �	�Nnfc�A�*

loss�5,;��       �	1]Onfc�A�*

losssT<j5^V       �	��Onfc�A�*

lossS��<��s�       �	��Pnfc�A�*

loss0�;��e�       �	�0Qnfc�A�*

loss�Ia=�q�;       �	u�Qnfc�A�*

loss��_=�S       �	NbRnfc�A�*

loss�u�=��'�       �	��Rnfc�A�*

loss
�_=�ק�       �	V�Snfc�A�*

lossc9�<o�$|       �	@�Tnfc�A�*

lossd��;V4�*       �	R`Unfc�A�*

loss��s<����       �	%Vnfc�A�*

loss*nL=@�p�       �	�Vnfc�A�*

loss�+>:���       �	PpWnfc�A�*

loss�:�(N/       �	lXnfc�A�*

loss�Ɓ=�s)       �	��Xnfc�A�*

lossN8�<��       �	#1Ynfc�A�*

loss���<LUÂ       �	.�Ynfc�A�*

lossL�<��R       �	f�Znfc�A�*

loss3�;�x��       �	;o[nfc�A�*

lossʳ�=X�Ga       �	e\nfc�A�*

loss��Z;2���       �	�\nfc�A�*

loss�|<�YP       �	�r]nfc�A�*

lossC� =�p��       �	[C^nfc�A�*

loss�SV<���"       �	\�^nfc�A�*

loss��=��Dd       �	I�_nfc�A�*

loss��O<Ǚ�       �	(*`nfc�A�*

loss�GT<�&�       �	��`nfc�A�*

loss�m�<k*�&       �	danfc�A�*

lossn�i=v�!s       �	��anfc�A�*

loss��=�Ԓ       �	�Gcnfc�A�*

loss�&<0�`f       �	P�cnfc�A�*

lossd/=R:!       �	��dnfc�A�*

loss-�=�hc|       �	�Sfnfc�A�*

loss$e�<g��!       �	�gnfc�A�*

loss{<r=�        �	D�gnfc�A�*

loss�=9QV�       �	R�hnfc�A�*

lossQ1=���c       �	��infc�A�*

loss-��<8J��       �	{Njnfc�A�*

lossч�<��Wl       �		5knfc�A�*

lossX/<nF+�       �	Ullnfc�A�*

loss�Υ<P��!       �	mmnfc�A�*

loss��=�&�f       �	�8nnfc�A�*

loss���=�z69       �	�Nonfc�A�*

lossN��<o��t       �	�pnfc�A�*

loss�~�<���=       �	��qnfc�A�*

loss$��=�^�       �	�Qrnfc�A�*

loss/�;��       �	�ltnfc�A�*

loss���<]o��       �	�unfc�A�*

loss���<�[��       �	`�unfc�A�*

loss&V�=�E       �	�Pvnfc�A�*

loss΂&<�|��       �	*wnfc�A�*

loss�*(>?"V       �	ʩwnfc�A�*

lossQ�n<��n�       �	�Lxnfc�A�*

loss���;&�       �	K�xnfc�A�*

loss��<+�g�       �	+�ynfc�A�*

loss��k=L-ё       �	iznfc�A�*

lossqDc=W�P�       �	p�znfc�A�*

loss���;�ڠ\       �	vO{nfc�A�*

loss&��;���       �	��{nfc�A�*

loss��=���       �	�|nfc�A�*

loss���<�\�       �	v}nfc�A�*

loss��=���       �	��}nfc�A�*

loss��:6��       �	�Y~nfc�A�*

loss�#�<���       �	�%nfc�A�*

loss?M�<�;�       �	��nfc�A�*

lossn��<P�!       �	�f�nfc�A�*

lossRW=T���       �	���nfc�A�*

lossW��<���@       �	���nfc�A�*

loss��<T��       �	�1�nfc�A�*

lossqTx<����       �	(ւnfc�A�*

loss�D>igt�       �	�r�nfc�A�*

lossǞ<�9d�       �	9�nfc�A�*

lossH�=�W�E       �	ҩ�nfc�A�*

loss�Y=kb�@       �	gF�nfc�A�*

loss�U%<s�       �	!�nfc�A�*

loss;{=Β]�       �	E��nfc�A�*

loss��='�(       �	G�nfc�A�*

loss�p�<����       �	���nfc�A�*

loss�{x<��       �	O�nfc�A�*

loss�6�;Tt��       �	��nfc�A�*

loss�3�;#��J       �	N{�nfc�A�*

loss��=Ġ�       �	��nfc�A�*

loss�(=;g�@       �	��nfc�A�*

lossJ��=��G1       �	5C�nfc�A�*

loss5�=Am
�       �	�֋nfc�A�*

loss��w<�3!       �	�k�nfc�A�*

loss���<(5�D       �	S�nfc�A�*

lossrj;ʫ�L       �	���nfc�A�*

lossh`�<s�n       �	2�nfc�A�*

loss4�><��l       �	�Ďnfc�A�*

loss��?<�}�"       �	�X�nfc�A�*

loss<?Q<8��       �	��nfc�A�*

loss�F9=O��       �	���nfc�A�*

loss���<�%E       �	�)�nfc�A�*

lossfL�<Ż�       �	���nfc�A�*

loss���<���       �	�Z�nfc�A�*

loss	8�;�       �	��nfc�A�*

loss���;��W�       �	\��nfc�A�*

lossf�q=�%�       �	�E�nfc�A�*

loss
P
=�1�!       �	C�nfc�A�*

loss߈�<��R       �	�nfc�A�*

loss���<5��[       �	��nfc�A�*

loss�a�=�ć       �	"��nfc�A�*

loss��D<���       �	�D�nfc�A�*

loss}��=3���       �	�ڗnfc�A�*

loss�^�<���"       �	%w�nfc�A�*

loss��e<NI�       �	K#�nfc�A�*

lossZ[F<�,�       �	ęnfc�A�*

loss.��<���       �	�_�nfc�A�*

lossӤ�;&���       �	���nfc�A�*

loss�d�<X�{[       �	���nfc�A�*

lossE|L=�J�k       �	�&�nfc�A�*

loss��L=�֛       �	Ի�nfc�A�*

loss=�^<V�       �	^�nfc�A�*

loss�3/<琱�       �	~�nfc�A�*

loss�g/=$ߐ�       �	��nfc�A�*

lossJEH=��?n       �	�@�nfc�A�*

lossE=8=ǥI[       �	H�nfc�A�*

lossŦV=2���       �	���nfc�A�*

lossQ�<=&ؚ�       �	�*�nfc�A�*

lossa�f<["9       �	n¡nfc�A�*

loss\o�<�j^       �	�m�nfc�A�*

loss��=��2�       �	��nfc�A�*

loss��~;�bM�       �	nfc�A�*

lossx��;�R��       �	N^�nfc�A�*

loss�=�'��       �	���nfc�A�*

loss�?�=�W�       �	O��nfc�A�*

lossFg;
�<_       �	v7�nfc�A�*

loss&�=}�/�       �	Uܦnfc�A�*

lossf7<5���       �	V�nfc�A�*

loss��=+�	�       �	y˨nfc�A�*

loss�&=��]j       �	g�nfc�A�*

loss�a�<�f'+       �	nfc�A�*

loss��)<ӝ}{       �	�U�nfc�A�*

lossX'0=�X�       �	���nfc�A�*

loss�X�<5d�       �	:��nfc�A�*

loss�-�;B�k       �	<0�nfc�A�*

lossദ<��,~       �	�ϭnfc�A�*

lossE�l<��.m       �	^f�nfc�A�*

lossxY#=A:�^       �	��nfc�A�*

loss��<[���       �	໯nfc�A�*

loss�a�<��/       �	�_�nfc�A�*

loss���<�ؚR       �	��nfc�A�*

loss`_`<�7�       �	+��nfc�A�*

lossg�<r���       �	v��nfc�A�*

loss�s�;I)       �	P�nfc�A�*

lossTV;<���&       �	�nfc�A�*

loss�Y\=�hl       �	���nfc�A�*

lossrǈ=3�3       �	��nfc�A�*

loss��=[��       �	���nfc�A�*

loss��=)�       �	�H�nfc�A�*

loss1^�:1�%       �	��nfc�A�*

lossU�=�B       �	@��nfc�A�*

loss��>�J�       �	b1�nfc�A�*

loss@Y<�Pp�       �	�Թnfc�A�*

loss�>�<GM�Z       �	s�nfc�A�*

loss&0=�D�Y       �	*�nfc�A�*

lossO��=�vl       �	j»nfc�A�*

lossI3<;N�x�       �	�f�nfc�A�*

losshڨ<T��       �	��nfc�A�*

lossS&�=Ps       �	a��nfc�A�*

loss	�=3�[Y       �	,G�nfc�A�*

lossL.�<�@       �	:�nfc�A�*

lossK��=0�$        �	�ÿnfc�A�*

loss#�	=%=M(       �	ep�nfc�A�*

loss�/=A��;       �	i�nfc�A�*

lossi,`=��       �	,��nfc�A�*

loss�D�<�a1�       �	�^�nfc�A�*

loss=0<��)`       �	U��nfc�A�*

loss�#<�?�       �	B��nfc�A�*

loss���<�>S�       �	�.�nfc�A�*

lossX�`<Q��       �	h��nfc�A�*

loss6�;3�
�       �	oc�nfc�A�*

losss��<ｧ5       �	@��nfc�A�*

loss\�=hf�       �	K��nfc�A�*

loss	b�;�Qs       �	t'�nfc�A�*

lossƦ�;
��       �	���nfc�A�*

loss��=���q       �	�T�nfc�A�*

lossL�<�=��       �	c��nfc�A�*

lossj�=�K�       �	A�nfc�A�*

lossJE�<�4<�       �	�j�nfc�A�*

loss!��:CD��       �	��nfc�A�*

lossa;�zΗ       �	��nfc�A�*

loss��!<���       �	?�nfc�A�*

loss2�p=D�S�       �	���nfc�A�*

loss>��<���       �	
��nfc�A�*

lossl��=
�       �	�.�nfc�A�*

loss�1=���       �	���nfc�A�*

loss�2=���G       �	�h�nfc�A�*

lossl<�o�%       �	g
�nfc�A�*

lossA��=7��       �	0��nfc�A�*

loss�h�=�
��       �	�>�nfc�A�*

loss�;��/�       �	���nfc�A�*

loss$�<&@       �	o�nfc�A�*

loss�1z=�AiK       �	2�nfc�A�*

loss�ɨ:����       �	ݶ�nfc�A�*

loss<Fo=����       �	�]�nfc�A�*

lossw��<����       �	���nfc�A�*

lossC��<��A       �	���nfc�A�*

loss�;�       �	�w�nfc�A�*

lossf�<�q?!       �	��nfc�A�*

lossh,�:�]w       �	{��nfc�A�*

loss]�;y�+�       �	�a�nfc�A�*

loss���<�bAA       �	��nfc�A�*

loss6h<�h�v       �	Ҫ�nfc�A�*

loss�lc=�5�       �	sL�nfc�A�*

loss#��<���       �	!��nfc�A�*

lossZS�<�}�9       �	!��nfc�A�*

loss�[�;�U)       �	��nfc�A�*

loss�+�:�g��       �	�]�nfc�A�*

loss�ٌ;��A�       �	3�nfc�A�*

loss�8�<z{�R       �	���nfc�A�*

loss��<6���       �	���nfc�A�*

loss���<���       �	I��nfc�A�*

loss�F<+1�1       �	+�nfc�A�*

lossQ"=����       �	��nfc�A�*

loss��<9��o       �	f�nfc�A�*

loss���<�!�       �	#�nfc�A�*

loss��=$���       �	`��nfc�A�*

loss�Ѵ<���       �	pC�nfc�A�*

lossFh�<GQ�       �	���nfc�A�*

loss��;��\       �	�u�nfc�A�*

lossR�K=��B�       �	��nfc�A�*

loss᰼=.*"       �	��nfc�A�*

lossr*;I�#       �	EG�nfc�A�*

loss�u3<'��h       �	���nfc�A�*

losss�e<wmO       �	2�nfc�A�*

loss�;>�>b       �	���nfc�A�*

lossf�<���       �	5��nfc�A�*

loss��"; %�y       �	�P�nfc�A�*

loss��6<�倗       �	���nfc�A�*

lossE�<e�        �	���nfc�A�*

loss��;ˑk       �	d��nfc�A�*

loss*Z;:`�*       �	�1�nfc�A�*

loss7��<��u�       �	���nfc�A�*

losseV�=JРV       �	�h�nfc�A�*

lossn��<˒z       �	\�nfc�A�*

lossJ�<�=�>       �	,��nfc�A�*

lossM��<��}o       �	�1�nfc�A�*

loss6��:
h�       �	A��nfc�A�*

loss,}3<z��       �	�l�nfc�A�*

loss��;���<       �	��nfc�A�*

loss��<�#E       �	m�nfc�A�*

loss��m<IABP       �	���nfc�A�*

loss�eG:`��5       �	aT�nfc�A�*

loss��:�'��       �	���nfc�A�*

lossA;U�=       �	Ė�nfc�A�*

loss;��9~RK       �	j��nfc�A�*

loss� �9s]h#       �	c%�nfc�A�*

loss��o9�Gr�       �	���nfc�A�*

loss���;�Z�       �	?V�nfc�A�*

lossZ��;F���       �	w��nfc�A�*

lossM��;i��       �	e��nfc�A�*

lossa��9��mR       �	&�nfc�A�*

lossZL4;ZA�       �	*��nfc�A�*

loss][�=�٠w       �	�g�nfc�A�*

loss��<�G��       �	G �nfc�A�*

loss�>K�ٳ       �	9��nfc�A�*

loss��C<�s�c       �	]4�nfc�A�*

lossLS=�C��       �	���nfc�A�*

loss�Ϣ<��>�       �	�� ofc�A�*

lossJsw=7T:�       �	�#ofc�A�*

loss��d=ZYJ       �	�ofc�A�*

loss�̋=&j       �	�\ofc�A�*

loss��<��SI       �	��ofc�A�*

loss�<�-+L       �	��ofc�A�*

loss�<���K       �	�Cofc�A�*

loss�ò=�3�       �	�ofc�A�*

lossA�d=����       �	��ofc�A�*

loss�R�;;��       �	Oofc�A�*

lossO�<�6�       �	E�ofc�A�*

loss)/,=ۡQ       �	.�ofc�A�*

loss�31=٨        �	^-	ofc�A�*

loss#��<{ͫH       �	��	ofc�A�*

loss��G=����       �	$`
ofc�A�*

loss�AE<�M:�       �	�Hofc�A�*

loss<t4.d       �	]�ofc�A�*

loss�� =���       �	�xofc�A�*

loss�;�4�       �	�Gofc�A�*

lossd�"<$�y       �	�ofc�A�*

loss�u;.��       �	��ofc�A�*

loss�<��}       �	+�ofc�A�*

loss�U<c=8�       �	�Oofc�A�*

loss�q�<�nP�       �	��ofc�A�*

loss�=���       �	��ofc�A�*

lossj�=�G��       �	�.ofc�A�*

loss֣Y<1��       �	2�ofc�A�*

lossY5=pfda       �	Eeofc�A�*

loss@�l;�[Zc       �	?�ofc�A�*

lossdW�<����       �	;�ofc�A�*

loss�ݞ<��×       �	�Gofc�A�*

loss�zy<y�Z�       �	��ofc�A�*

losslZ�;n�D`       �	B�ofc�A�*

loss�p<r%t       �	�-ofc�A�*

loss��A=�-�       �	�
ofc�A�*

loss���<�W�       �		�ofc�A�*

lossT'v;��Z       �	�Eofc�A�*

loss$w:_n[�       �	��ofc�A�*

loss�'�<����       �	&�ofc�A�*

lossiWa<?άR       �	J�ofc�A�*

loss��u<܍��       �	�kofc�A�*

loss�!�=O6�       �	U2ofc�A�*

loss1��<�Q!0       �	��ofc�A�*

loss��<�`�       �	W\ofc�A�*

loss��<��W       �	�ofc�A�*

lossH�Y;K�       �	b�ofc�A�*

loss!��;����       �	7 ofc�A�*

loss�4<6&(       �	�0Bofc�A�*

lossW��<�9[       �	5Cofc�A�*

loss�}�=��y       �	��Cofc�A�*

lossֻ�;��a       �	�0Dofc�A�*

lossE�=���%       �	i�Dofc�A�*

lossߍ;��[�       �	A�Eofc�A�*

loss��E<4       �	�JFofc�A�*

losst��<N�%�       �	��Fofc�A�*

loss���<�q�       �	D�Gofc�A�*

loss�=�^L�       �	*Hofc�A�*

loss���:��       �	��Hofc�A�*

lossϣL<.��       �	
JIofc�A�*

loss�.<�jK       �	e�Iofc�A�*

lossڅ�<;] �       �	[zJofc�A�*

loss�A@=�윷       �	UKofc�A�*

loss(�;�*mL       �	\�Kofc�A�*

loss�5�;�O�       �	�ALofc�A�*

loss�n<ާ�       �	X�Lofc�A�*

loss��<n��       �	Mofc�A�*

loss��v=��a       �	�Nofc�A�*

lossG�<H{|       �	`�Nofc�A�*

lossd�K=��a       �	ROofc�A�*

lossȵ=�.�       �	!�Oofc�A�*

loss�,�=�{G�       �	�|Pofc�A�*

loss�2�<7��       �	QQofc�A�*

loss;Ӡ;����       �	ԹQofc�A�*

lossO��=���       �	�QRofc�A�*

loss�F<�߬Q       �	��Rofc�A�*

loss�	=D}��       �	BxSofc�A�*

loss%�<	h;       �	Tofc�A�*

loss�4<���       �	M�Tofc�A�*

lossXp<����       �	PrUofc�A�*

loss.0<�+��       �	�
Vofc�A�*

loss�%�<]�>�       �	3�Vofc�A�*

loss�=_
؀       �	�?Wofc�A�*

loss<3�<⧾$       �	��Wofc�A�*

loss��<�|       �	jXofc�A�*

loss%r5=��9a       �	KYofc�A�*

loss���=��h�       �	x�Yofc�A�*

lossr��<��7X       �	�.Zofc�A�*

lossd�;���p       �	��Zofc�A�*

loss���<pL��       �	�g[ofc�A�*

lossx�;��v       �	�U\ofc�A�*

loss�{<]Cۜ       �	��\ofc�A�*

loss/�'=����       �	�z]ofc�A�*

loss��<���v       �	*�^ofc�A�*

lossRm�<�1�       �	�[_ofc�A�*

loss��<��x0       �	6#`ofc�A�*

loss�2V=\L�|       �	ü`ofc�A�*

lossE+�;ڬ��       �	HPaofc�A�*

losso<G;��U�       �	?bofc�A�*

loss[�|=�w       �	 �bofc�A�*

loss�#<��T       �	�/cofc�A�*

loss@��<;IL�       �	��cofc�A�*

loss��<�d       �	�\dofc�A�*

loss��;t~#       �	��dofc�A�*

loss*r;Κn       �	�eofc�A�*

loss�D�;	H�(       �	mfofc�A�*

loss�o�<�֖U       �	�fofc�A�*

loss#��<�
       �	<Lgofc�A�*

loss*�>>�w�       �	��gofc�A�*

loss2^7<��z�       �	�hofc�A�*

loss鑖;E+�       �	Kiofc�A�*

loss�,=�XT       �	'jofc�A�*

loss���:6�0       �	��jofc�A�*

loss�.=wյ       �	��kofc�A�*

loss<��B�       �	��lofc�A�*

loss)� =z�@�       �	�-mofc�A�*

loss4�<{�b5       �	��mofc�A�*

loss�t<<_=�K       �	�dnofc�A�*

loss� r<l�q       �	��nofc�A�*

loss�lo=C�       �	t�oofc�A�*

loss���<�-�H       �	�4pofc�A�*

loss��<c���       �	�pofc�A�*

loss/�<�#q�       �	.rqofc�A�*

loss^B�< �'       �	crofc�A�*

loss�/;�\��       �	f�rofc�A�*

loss�o<(1       �	�7sofc�A�*

loss���;g*z�       �	S�sofc�A�*

loss��P<�=%m       �	�`tofc�A�*

loss�&�;�Q       �	j�tofc�A�*

loss�]&<��c�       �	N�uofc�A�*

loss���<h�P0       �	$dvofc�A�*

loss$�{;�)�       �	#�vofc�A�*

loss��@<�I/�       �	֍wofc�A�*

loss��~;1M0�       �	%xofc�A�*

lossi�;3�b�       �	!yofc�A�*

lossM��<�X�z       �	p�yofc�A�*

loss���<n3�       �	3Rzofc�A�*

lossw�E;GJ�v       �	D�zofc�A�*

loss��<.��j       �	+�{ofc�A�*

loss�|m=f�g�       �	oE|ofc�A�*

loss�:P;�'g       �	j�|ofc�A�*

loss���;[)��       �	<�}ofc�A�*

loss�%;=h��       �	P~ofc�A�*

lossW��<�Xp       �	�~ofc�A�*

lossc�8<��1       �	�Yofc�A�*

loss��8<��'       �	~5�ofc�A�*

lossWܱ;&��_       �	�̀ofc�A�*

loss�_<�7E"       �	w��ofc�A�*

loss���<78X       �	�4�ofc�A�*

loss��;Y2��       �	���ofc�A�*

lossR�P=�sq�       �	ߣ�ofc�A�*

loss��&;�:�       �	WC�ofc�A�*

loss��<ș�@       �	:�ofc�A�*

loss�<�9��       �	ލ�ofc�A�*

loss(�";b�z       �	�4�ofc�A�*

lossq�&=�       �	d̆ofc�A�*

loss��q=�a
�       �	�e�ofc�A�*

loss��<+��j       �	P��ofc�A�*

loss��P=7��       �	1��ofc�A�*

lossf%�:z�P�       �	6�ofc�A�*

loss-�Q=�i�       �	F͉ofc�A�*

loss�@�:0l�*       �	�c�ofc�A�*

loss�<�;W�m       �	���ofc�A�*

loss<��;��@       �	:>�ofc�A�*

lossZ>�)�^       �	֌ofc�A�*

lossi��=]�(�       �	�k�ofc�A�*

loss���<]��       �	��ofc�A�*

loss/}<���       �	��ofc�A�*

loss\��;�L��       �	|F�ofc�A�*

loss�5/=~�	�       �	��ofc�A�*

loss[<��m�       �	���ofc�A�*

lossTd�<
�       �	隑ofc�A�*

loss�O;���}       �	f0�ofc�A�*

loss�X�<���i       �	0g�ofc�A�*

loss�b�;����       �	a��ofc�A�*

loss��;���D       �	��ofc�A�*

loss�O=7B��       �	�0�ofc�A�*

loss�P;�JW�       �	Hofc�A�*

lossn��<�[�X       �	.U�ofc�A�*

loss�A=� *       �	�ofc�A�*

loss�d=����       �	�{�ofc�A�*

loss|��;h?��       �	��ofc�A�*

lossX��<����       �	P��ofc�A�*

loss2�;���       �	�;�ofc�A�*

loss��<�K!       �	�Йofc�A�*

loss�M<�V_�       �	 o�ofc�A�*

lossV�C;����       �	�N�ofc�A�*

loss��H<�i       �	 �ofc�A�*

loss��<�<cv       �	�u�ofc�A�*

lossT+�<`�       �	��ofc�A�*

lossࣘ<����       �	W��ofc�A�*

lossX:�<�(��       �	S\�ofc�A�*

loss�#<�<�@       �	c�ofc�A�*

lossj]<�#D       �	u��ofc�A�*

loss͵�;��&z       �	�1�ofc�A�*

loss�(�;D?t�       �	�Ġofc�A�*

loss���:t_��       �	ca�ofc�A�*

loss�HH<��4       �	H��ofc�A�*

loss�2�;4�       �	}��ofc�A�*

losswy>=�s��       �	.�ofc�A�*

lossz3�<�-�&       �	�ƣofc�A�*

loss��N<�#,�       �	8f�ofc�A�*

loss=�<g���       �	��ofc�A�*

loss��m;t=o�       �	䢥ofc�A�*

lossAB�<�@�       �	?�ofc�A�*

lossʜb=ү�O       �	��ofc�A�*

loss�f�<���j       �	��ofc�A�*

loss��;��c�       �	N*�ofc�A�*

loss#��<�z��       �	��ofc�A�*

lossc@=A��       �	���ofc�A�*

loss﹨=&xG       �	^J�ofc�A�*

lossx��:bkN        �	e�ofc�A�*

loss�}�=��       �	�v�ofc�A�*

loss��};���       �	�
�ofc�A�*

loss�~�<� A       �	���ofc�A�*

loss��<Q+N       �	�T�ofc�A�*

lossOK*;}<��       �	>��ofc�A�*

lossH'$;�?��       �	υ�ofc�A�*

loss��<�v��       �	!�ofc�A�*

loss�*;L�X?       �	U�ofc�A�*

losseH};)):�       �	�5�ofc�A�*

lossq�<@��Y       �	]ܱofc�A�*

loss�_<KAپ       �	�|�ofc�A�*

loss_��<Q���       �	j�ofc�A�*

lossFJR=zB��       �	���ofc�A�*

loss�/�;f��       �	=C�ofc�A�*

lossH��<1��       �	�شofc�A�*

loss��;�֩       �	�~�ofc�A�*

lossΫr</4�       �	G�ofc�A�*

lossD4�;jE�       �	0�ofc�A�*

loss��>; 
<3       �	�ʷofc�A�*

loss�2<��       �	Eg�ofc�A�*

loss�z;)��       �	y��ofc�A�*

loss�þ<#�M�       �	3O�ofc�A�*

loss�.c=�Y�&       �	k�ofc�A�*

loss?;<��       �	5��ofc�A�*

loss��;u!�       �	}>�ofc�A�*

loss7�;��f�       �	!�ofc�A�*

losst�#=kȏA       �	셽ofc�A�*

loss��v<	l�u       �	%�ofc�A�*

lossϳ<��
�       �	<��ofc�A�*

loss7hm;K�@W       �	W]�ofc�A�*

loss)�:i~]�       �	�ofc�A�*

loss�w�<��}�       �	h��ofc�A�*

losswx:=/?       �		2�ofc�A�*

loss#��<mߝ       �	���ofc�A�*

loss!�+<�'�       �	l�ofc�A�*

loss7z�;�8�e       �	^�ofc�A�*

loss�r=��E       �	���ofc�A�*

loss�Ǝ:����       �	�O�ofc�A�*

lossmB�=��К       �	!�ofc�A�*

loss�pJ;m�       �	��ofc�A�*

loss��<V_C       �	}@�ofc�A�*

lossU.<��&�       �	N��ofc�A�*

lossq:f=F9�       �	�n�ofc�A�*

loss"o=js��       �	�2�ofc�A�*

loss���;�gPf       �	��ofc�A�*

loss�xQ=iĻC       �	�{�ofc�A�*

lossj�;���       �	?�ofc�A�*

loss�O�;*L�       �	��ofc�A�*

lossR�{;����       �	6W�ofc�A�*

loss�[�<n�b       �	��ofc�A�*

loss�<LP��       �	i��ofc�A�*

loss�Rv=z�	       �	 6�ofc�A�*

loss��3>��       �	���ofc�A�*

loss�S�<��EN       �	dw�ofc�A�*

loss!�=�?��       �	~�ofc�A�*

loss�f�=q�?       �	G=�ofc�A�*

lossc�p=�L��       �	>��ofc�A�*

loss�f]<n�q�       �	Œ�ofc�A�*

loss[|>=�~`       �	m9�ofc�A�*

loss��=Lz
L       �	6��ofc�A�*

lossq�3<���       �	���ofc�A�*

lossN�==�/|�       �	�2�ofc�A�*

loss�|�:\b��       �	��ofc�A�*

loss��<AG�W       �	�o�ofc�A�*

loss]6+<���       �	��ofc�A�*

lossX�h<<h>�       �	���ofc�A�*

loss�1�;[;�       �	�V�ofc�A�*

loss�t�;#L��       �	���ofc�A�*

loss�A�<��]�       �	J��ofc�A�*

loss��a=đ��       �	+1�ofc�A�*

loss�R<=d"       �	h��ofc�A�*

lossDZ�<�\�       �	�o�ofc�A�*

lossj~�<��@       �	��ofc�A�*

loss�d=����       �	��ofc�A�*

loss��</��       �	-A�ofc�A�*

lossoe�<] �       �		��ofc�A�*

loss�O;`�       �	�t�ofc�A�*

loss���<�F&}       �	�ofc�A�*

loss)�I;��,�       �	���ofc�A�*

loss\�<�X�M       �	�9�ofc�A�*

losso��<7(5�       �	�ofc�A�*

loss)bS=d���       �	��ofc�A�*

loss��<�q�}       �	r7�ofc�A�*

loss6B6=N×2       �	���ofc�A�*

loss? 	=gQM�       �	Uk�ofc�A�*

loss6<�<��s�       �	)�ofc�A�*

lossS�<G7�       �	���ofc�A�*

loss�=a*�       �	͒�ofc�A�*

loss&��;wr��       �	�E�ofc�A�*

lossa=;���       �	��ofc�A�*

loss���:m��       �	*��ofc�A�*

loss�V=�vcC       �	]�ofc�A�*

lossH�<\0�       �	,-�ofc�A�*

loss��6=�~       �	���ofc�A�*

loss�oJ=����       �	��ofc�A�*

loss�3=˓7�       �	ɐ�ofc�A�*

loss��<o��       �	x�ofc�A�*

loss�f%;��       �	%$�ofc�A�*

lossIK�<�6 �       �	���ofc�A�*

loss�i�<pPy�       �	�]�ofc�A�*

loss�fL=��U       �	���ofc�A�*

loss��h<f��       �	��ofc�A�*

loss$��<��1�       �	�+�ofc�A�*

lossR��<�1��       �	6��ofc�A�*

loss(��=[˶       �	=_�ofc�A�*

loss��<����       �	���ofc�A�*

loss��[<���       �	��ofc�A�*

loss���;AL�       �	�0�ofc�A�*

loss��<8�       �	��ofc�A�*

loss��<&@D�       �	�m�ofc�A�*

loss_j�=�;|c       �	[
�ofc�A�*

loss�}�=�d6�       �	ҫ�ofc�A�*

loss�=ƀ�       �	F�ofc�A�*

loss3��9��       �	���ofc�A�*

loss��<=.�       �	���ofc�A�*

loss���<&^�f       �	�.�ofc�A�*

loss�`�<�֘�       �	^��ofc�A�*

loss�s�<?��       �	؀�ofc�A�*

loss0��=�S��       �	� �ofc�A�*

loss���<!�Tf       �	���ofc�A�*

loss\�=��       �	g�ofc�A�*

loss���<k;��       �	`�ofc�A�*

lossP�<��       �	��ofc�A�*

loss��=&P       �	lC�ofc�A�*

loss?u<���E       �	@��ofc�A�*

loss(J�;O�RI       �	hw�ofc�A�*

loss��;��s�       �	� pfc�A�*

loss�8�<��S       �	� pfc�A�*

loss�%e<���       �	\>pfc�A�*

loss��];����       �	��pfc�A�*

lossJ�q<��3       �	fpfc�A�*

loss� =�#3       �	��pfc�A�*

loss�Dj<f� �       �	_�pfc�A�*

lossj�k=�p��       �	�7pfc�A�*

loss���=�Ù       �	pfc�A�*

loss�{=���[       �	�:pfc�A�*

loss�,~=0N�       �	�pfc�A�*

loss��<Y��       �	�qpfc�A�*

loss�Q<y�c�       �	pfc�A�*

lossl�~=Qv       �	��pfc�A�*

loss(�=�/�       �	�|	pfc�A�*

loss?{>/J&�       �	ݔ
pfc�A�*

lossL��;L�>       �	ADpfc�A�*

loss�=���|       �	�~pfc�A�*

loss��;?�e�       �	�)pfc�A�*

lossR�;��n�       �	��pfc�A�*

lossΎ<($��       �	mppfc�A�*

loss!��<�.`�       �	�pfc�A�*

loss�K=�դ       �	c�pfc�A�*

loss�u�:���M       �	�\pfc�A�*

lossd��;���       �	Xpfc�A�*

loss1�=��r0       �	Z�pfc�A�*

loss�1l;!���       �	�@pfc�A�*

loss�<�]d       �	<�pfc�A�*

loss�=|Zl5       �	=}pfc�A�*

lossl �=�/ZA       �	�!pfc�A�*

loss_'�;n��       �	��pfc�A�*

loss��>C��       �	�tpfc�A�*

loss%��<vB       �	�jpfc�A�*

lossQ܁<��3<       �	�pfc�A�*

loss#�=���h       �	�pfc�A�*

loss�u�;�OJ       �	�tpfc�A�*

loss��=����       �	��pfc�A�*

lossO�q=^�C#       �	�Fpfc�A�*

loss�]�<�?�       �	�pfc�A�*

loss��O<U�       �	]�pfc�A�*

lossD��<�h�       �	�pfc�A�*

loss�+=�O       �	V�pfc�A�*

loss4,<XWD       �	�\pfc�A�*

loss�f�;K'��       �	��pfc�A�*

loss�r8=╱       �	��pfc�A�*

loss�B�<ސ'       �	�|pfc�A�*

loss��E<�)�       �	� pfc�A�*

losss=qF�       �	�� pfc�A�*

loss�=�ĵ       �	�R!pfc�A�*

loss�n�;�۽       �	 �!pfc�A�*

loss�3�<���Z       �	b�"pfc�A�*

loss���<���       �	m#pfc�A�*

loss�;%=�6?'       �	��#pfc�A�*

loss-"f=+��       �	�K$pfc�A�*

lossR�l=(�J,       �	\�$pfc�A�*

lossc�(=��       �	�|%pfc�A�*

loss5�<�.�       �	�&&pfc�A�*

lossH�`<MWi�       �	^�&pfc�A�*

loss��F==L��       �	&Q'pfc�A�*

loss╖;	2�       �	��'pfc�A�*

loss�z�;�E+       �	�|(pfc�A�*

loss)�(<��B       �	b)pfc�A�*

loss��=��l"       �	�*pfc�A�*

lossچ�;1R��       �	]�*pfc�A�*

lossO�A=�ض5       �	�+pfc�A�*

loss;}�;�S�^       �	��,pfc�A�*

loss�C=��C       �	H�-pfc�A�*

loss)�$<����       �	RH.pfc�A�*

loss
�<� >�       �	�.pfc�A�*

loss.^:=₰�       �	�M0pfc�A�*

lossvX/<Hɉ�       �	2�0pfc�A�*

loss��
<j��c       �	��1pfc�A�*

loss
��<)&�       �	��2pfc�A�*

loss26=�@�f       �	�X3pfc�A�*

loss�љ;}0��       �	/�3pfc�A�*

lossI�P=�2q"       �	Ֆ4pfc�A�*

loss�t=�/��       �	pC5pfc�A�*

loss�<��        �	�5pfc�A�*

loss0K�<70�y       �	֌6pfc�A�*

loss*�<��^/       �	�/7pfc�A�*

loss{�]=2Ϛ�       �	�7pfc�A�*

loss���;Z��       �	�p8pfc�A�*

loss�y�;V��       �	39pfc�A�*

loss��;�Y�       �	�9pfc�A�*

loss�Y<"���       �	�{:pfc�A�*

loss�t�<W���       �	�;pfc�A�*

lossb�=���       �	ӿ;pfc�A�*

loss3�]<���       �	<�<pfc�A�*

loss�v.<>c~       �	$�=pfc�A�*

loss���<���       �	�8>pfc�A�*

loss()2<�7�       �	H�>pfc�A�*

loss�x�<�#X]       �	s�?pfc�A�*

loss�"�=(��n       �	�*@pfc�A�*

loss�k<}0�U       �	O�@pfc�A�*

loss�H;�?       �	�zApfc�A�*

lossQ́<�fq�       �	�/Bpfc�A�*

loss�=�b��       �	I�Bpfc�A�*

loss!o�=�̫~       �	K�Cpfc�A�*

loss��<�       �	�WDpfc�A�*

loss�_<�Ú       �	�:Epfc�A�*

lossᙚ<M�@       �	�Epfc�A�*

lossM <ؿ��       �	$|Fpfc�A�*

loss<Z�;��       �	�Gpfc�A�*

lossn�%<�q��       �	ȱGpfc�A�*

loss�4=���       �	TSHpfc�A�*

lossA�<̷cr       �	(�Hpfc�A�*

loss��;ED�       �	�Ipfc�A�*

loss�Z�;�:m       �	*Jpfc�A�*

loss�ܑ;q�0G       �	��Jpfc�A�*

loss	<�<��h'       �	��Kpfc�A�*

loss�PH<�k!;       �	�cLpfc�A�*

lossj��;��1       �	m�Lpfc�A�*

loss:�#=P�       �	�Mpfc�A�*

loss̇�<֧�J       �	�5Npfc�A�*

loss��*=t�       �	��Npfc�A�*

loss�<?a�[       �	"qOpfc�A�*

loss�(�<�G��       �	�Ppfc�A�*

lossMn�<�D�3       �	s�Ppfc�A�*

loss�K=�I�W       �	�`Qpfc�A�*

loss�V-<Xl       �	Rpfc�A�*

loss�X=�G       �	j�Rpfc�A�*

loss#=�er�       �	�<Spfc�A�*

loss/=<�sE       �	Q�Spfc�A�*

loss���<�ޏ�       �	�oTpfc�A�*

loss�0=ҫf       �	�
Upfc�A�*

loss��=<&�       �	��Upfc�A�*

lossL�;��G       �	�>Vpfc�A�*

loss: �<8QI�       �	D�Vpfc�A�*

lossZt�<k9�       �	�pWpfc�A�*

loss1(
=��KR       �	�Xpfc�A�*

lossI�<�3u       �	��Xpfc�A�*

loss�6�;�N�+       �	�pYpfc�A�*

loss�P;��)*       �	�Zpfc�A�*

loss���<��{       �	ʤZpfc�A�*

loss�e�<#�0       �	LS[pfc�A�*

loss�D�=���       �	��[pfc�A�*

loss��;zK�       �	��\pfc�A�*

loss��q<�q�@       �	�%]pfc�A�*

loss*��<��SP       �	+�]pfc�A�*

loss!<����       �	�a^pfc�A�*

loss�b�<�$�w       �	��^pfc�A�*

loss�LT<7�rG       �	ß_pfc�A�*

loss���;��I�       �	%;`pfc�A�*

lossA��<��       �	��`pfc�A�*

loss�0�<��iW       �	!xapfc�A�*

loss/.�<���r       �	�bpfc�A�*

loss�D�:8�Y'       �	��bpfc�A�*

loss�\;����       �	�Fcpfc�A�*

loss	��;'Z�       �	��cpfc�A�*

loss~t�<�x�       �	�vdpfc�A�*

lossd �<�rq�       �	�
epfc�A�*

loss^�=ș��       �	�epfc�A�*

loss�<�]       �	�7fpfc�A�*

loss<I�=?E|�       �	��fpfc�A�*

lossC��;��:�       �	��gpfc�A�*

lossP�<�'��       �	?hpfc�A�*

loss]��=��'       �	ͭhpfc�A�*

loss܇Y<�3�       �	@ipfc�A�*

lossã8;k��       �	�jpfc�A�*

loss�˕;�P�a       �	��jpfc�A�*

loss֨�<�)��       �	GUkpfc�A�*

losszp�<�$�       �	�lpfc�A�*

loss��8;��       �	�lpfc�A�*

loss�=�;��       �	Vmpfc�A�*

loss�<@=s�)       �	��mpfc�A�*

loss�lC<3L�       �	��npfc�A�*

lossF	<]QQ�       �	@opfc�A�*

loss)vJ;�2t       �	�opfc�A�*

loss� �<����       �	GYppfc�A�*

loss�y =�lxk       �	��ppfc�A�*

loss6�;P�>�       �	�qpfc�A�*

loss	{c<����       �	80rpfc�A�*

loss��<F���       �	g�rpfc�A�*

loss[�/>Eb��       �	g~spfc�A�*

losswy;^��3       �	atpfc�A�*

loss���:{7��       �	,�tpfc�A�*

loss���<���       �	�\upfc�A�*

loss��:�,�       �	_(vpfc�A�*

losst ;Xf�G       �	j�vpfc�A�*

lossn�N:��g�       �	�Wwpfc�A�*

loss1�%=�f��       �	��wpfc�A�*

lossj��<lj�       �	��xpfc�A�*

loss��@<(�I       �	�4ypfc�A�*

lossO�=.}n       �	��ypfc�A�*

lossx��;�9��       �	w�zpfc�A�*

loss��6;�h!g       �	 {pfc�A�*

lossfC�9vZT�       �	��{pfc�A�*

lossR�*;�y �       �	�\|pfc�A�*

loss=I!<�a/�       �	|�|pfc�A�*

loss̋�=� 6       �	ۆ}pfc�A�*

loss�I<w��       �	�~pfc�A�*

loss#�`=öе       �	��~pfc�A�*

lossSY�;����       �	ypfc�A�*

losscE=X^H       �	��pfc�A�*

loss��=���=       �	֨�pfc�A�*

loss�V;(:rK       �	Â�pfc�A�*

loss\�:I��       �	�&�pfc�A�*

loss4AR=��e�       �	��pfc�A�*

loss�A�<�!�'       �	$|�pfc�A�*

loss�ɝ=�[��       �	�=�pfc�A�*

lossq3<!6��       �	i�pfc�A�*

lossa�2=8'�5       �	H��pfc�A�*

loss���:)Z       �	��pfc�A�*

loss1W=��k�       �	��pfc�A�*

loss��<���w       �	�2�pfc�A�*

loss_gI;��       �	�̈pfc�A� *

loss��{=��x�       �	㋉pfc�A� *

lossʱv;�Ş�       �	!�pfc�A� *

lossT�P=?7��       �	k��pfc�A� *

loss��=���       �	rn�pfc�A� *

loss�H�<�N�       �	��pfc�A� *

lossX��;��!       �	 ��pfc�A� *

loss�|�;�Ui       �	s/�pfc�A� *

lossFK^<x�&�       �	�ōpfc�A� *

loss�<&[`       �	�~�pfc�A� *

loss��X;:��Y       �	��pfc�A� *

loss,��<���       �	���pfc�A� *

lossm�<��$�       �		��pfc�A� *

loss<==椾�       �	6<�pfc�A� *

lossʯ]=:�(       �	�ёpfc�A� *

loss��';Տ�B       �	��pfc�A� *

loss�ڙ<�o�       �	�M�pfc�A� *

lossp�<p���       �	��pfc�A� *

loss���<v�       �	��pfc�A� *

loss�=1g       �	�S�pfc�A� *

loss�ԇ=�?M       �	�y�pfc�A� *

loss��<�e�       �	��pfc�A� *

losss��<J���       �	噘pfc�A� *

loss�<{[A_       �	'3�pfc�A� *

loss��f<����       �	ߙpfc�A� *

lossR�<c���       �	���pfc�A� *

lossCC�;�F��       �	��pfc�A� *

loss�c�9pxX�       �	���pfc�A� *

loss$�=��q       �	%Y�pfc�A� *

loss���=(P5�       �	���pfc�A� *

loss�<��ά       �	ˆ�pfc�A� *

loss )=ͧ��       �	��pfc�A� *

loss-�F<uG�       �	{��pfc�A� *

loss��<�X�       �	lZ�pfc�A� *

loss�,�<祯�       �	b��pfc�A� *

loss��A:���}       �	g`�pfc�A� *

loss��q<��       �	���pfc�A� *

loss�K<�(�\       �	y��pfc�A� *

lossiF8:��o       �	�0�pfc�A� *

loss3X�<�*       �	RԤpfc�A� *

loss���;NzNk       �	t�pfc�A� *

loss��;_�       �	X�pfc�A� *

loss!ܶ;�̰F       �	�(�pfc�A� *

loss��;�p�       �	C˧pfc�A� *

loss�V(;�;��       �	�j�pfc�A� *

loss���:`�{J       �	u�pfc�A� *

loss���9C탅       �	攩pfc�A� *

loss�u9�o       �	�\�pfc�A� *

loss�z�;ٙ?       �	,�pfc�A� *

loss�{;<MxY       �	��pfc�A� *

loss0
=���!       �	��pfc�A� *

lossFl:��6�       �	�K�pfc�A� *

loss{{<:+�=<       �	]�pfc�A� *

lossrҿ=O���       �	~�pfc�A� *

loss6�L:�]�       �	[$�pfc�A� *

loss��E>�w�       �	�įpfc�A� *

lossw��<�B�3       �	�z�pfc�A� *

loss%�=�-U�       �	W&�pfc�A� *

lossFgJ<�;?�       �	��pfc�A� *

loss� -;s�       �	~��pfc�A� *

lossA�=%?��       �	�9�pfc�A� *

loss
=w(e>       �	e��pfc�A� *

loss��=4��       �	[��pfc�A� *

loss�I�;GR+�       �	
+�pfc�A� *

lossT�6<oEd�       �	��pfc�A� *

loss-��<e!Q       �	yW�pfc�A� *

loss;��<�%%�       �	E�pfc�A� *

loss3��<�E�       �	`��pfc�A� *

lossR�*=)*}�       �	t)�pfc�A� *

loss�<=�*��       �	H��pfc�A� *

loss&!�<��@%       �	Q�pfc�A� *

loss��<�MR       �	��pfc�A� *

loss�}"<C��       �	큺pfc�A� *

loss�2�;�:��       �	��pfc�A� *

loss�!<�׀�       �	i��pfc�A� *

loss�q=�>B�       �	8�pfc�A� *

loss�Lr<%N�Z       �	�˼pfc�A� *

loss2�;;���       �	�^�pfc�A� *

lossѾ;@?��       �	��pfc�A� *

loss �;���       �	Q��pfc�A� *

loss�w�=G�W�       �	7�pfc�A� *

loss��	=G&��       �	���pfc�A� *

loss*2_=ԇ:�       �	h=�pfc�A� *

lossq�2=%v       �	g��pfc�A� *

loss3��;���       �	�u�pfc�A� *

loss�t=�`�       �	W�pfc�A� *

loss�W�;i@��       �	��pfc�A� *

loss/c�: G��       �	��pfc�A� *

lossw=�\�R       �	�pfc�A� *

lossxK�<�?H       �	1��pfc�A� *

lossQ��;Z�G�       �	c�pfc�A� *

loss��=B�3E       �	���pfc�A� *

loss?�&=)	��       �	ú�pfc�A� *

loss�0�;�/��       �	fN�pfc�A� *

loss�R3<Z�Am       �	���pfc�A� *

loss�R�<f�       �	���pfc�A� *

loss<X1��       �	�D�pfc�A� *

loss��.=$Ί]       �	���pfc�A� *

loss�T�;���W       �	t~�pfc�A� *

loss#]	=�S�       �	!�pfc�A� *

loss<ϙ=��q       �	ܹ�pfc�A� *

loss8L=0�W       �	�L�pfc�A� *

lossz��;�֓�       �	���pfc�A� *

loss���;�$�       �	�{�pfc�A� *

loss��V;��w       �	��pfc�A� *

loss�֩<��       �	I��pfc�A� *

losss� =��?       �	�!�pfc�A� *

lossf:�=��K�       �	��pfc�A� *

lossHQ�;[���       �	�Q�pfc�A� *

loss:	P<�}�b       �	%��pfc�A� *

loss�[?<'��!       �	4��pfc�A� *

loss���;]�{v       �	j�pfc�A� *

lossO�<�ڷ�       �	���pfc�A� *

loss�P7=9�fx       �	�>�pfc�A� *

loss��;�FӲ       �	���pfc�A� *

loss PK;\�95       �	~�pfc�A� *

loss
<'k}       �	2�pfc�A� *

loss�(�<�ᖁ       �	T��pfc�A� *

loss��Y=��U       �	���pfc�A� *

loss�eD;oЯc       �	�j�pfc�A� *

loss��=m���       �	,�pfc�A� *

loss,�:y�^�       �	ܠ�pfc�A� *

loss)�6<���       �	�j�pfc�A� *

lossD�<	yQ}       �	��pfc�A� *

loss���<��6�       �	m��pfc�A� *

loss)�<8���       �	�N�pfc�A� *

loss�.�=!���       �	�j�pfc�A� *

loss�<��V       �	�pfc�A� *

lossR�*=�w��       �	Q��pfc�A�!*

lossQ�E<[ �~       �	�:�pfc�A�!*

loss�ҹ=J9       �	���pfc�A�!*

loss���=F��I       �	�t�pfc�A�!*

lossso�<��I�       �	r�pfc�A�!*

loss�#<{��       �	O��pfc�A�!*

loss�W;��l�       �	`��pfc�A�!*

lossd�;�z+       �	T7�pfc�A�!*

lossi�=X��Y       �	���pfc�A�!*

loss\K.<��Ov       �	_|�pfc�A�!*

lossFޏ<�9�       �	��pfc�A�!*

loss�wh;=�       �	h��pfc�A�!*

loss�=�z�       �	�C�pfc�A�!*

lossL��;���       �	���pfc�A�!*

loss�W<W�8)       �	u�pfc�A�!*

loss^ �=�7	       �	E�pfc�A�!*

loss�a =��×       �	o��pfc�A�!*

loss=%�;]h��       �	c�pfc�A�!*

loss7j�<��7       �	H��pfc�A�!*

loss��H=N�$�       �	Z� qfc�A�!*

loss�D\;��g�       �	<qfc�A�!*

loss�=5]�\       �	!�qfc�A�!*

loss:'�<�B��       �	C�qfc�A�!*

loss!�<��,       �	�'qfc�A�!*

loss0}�<5�       �	;�qfc�A�!*

loss��=��y�       �	hqfc�A�!*

loss��<�b�       �	fqfc�A�!*

loss-}?;�U��       �	��qfc�A�!*

lossځ=���       �	Xqfc�A�!*

loss:��;Q       �	��qfc�A�!*

loss��$>hKp�       �	�rqfc�A�!*

lossAw�<����       �	k	qfc�A�!*

loss@E�;� ��       �	�	qfc�A�!*

loss8��;h��       �	�D
qfc�A�!*

loss�?<�j�#       �	��
qfc�A�!*

loss(��<y|��       �	�xqfc�A�!*

loss
;�<��@       �	Eqfc�A�!*

loss�&�=�o�       �	�.qfc�A�!*

loss4�;��	       �	��qfc�A�!*

lossZtQ<�o�       �	�lqfc�A�!*

loss�U�<���X       �	}#qfc�A�!*

loss��<��k       �	�qfc�A�!*

lossG3!<à�       �	�Xqfc�A�!*

lossoT�;��u       �	�qfc�A�!*

loss%-�<���+       �	�8qfc�A�!*

lossv�v=O�       �	9�qfc�A�!*

loss�N=(��o       �	Qiqfc�A�!*

loss&߯<_"+U       �	� qfc�A�!*

lossr��<�R��       �	�qfc�A�!*

loss�_<�Y*�       �	l�qfc�A�!*

loss��0=}tJ       �	�iqfc�A�!*

loss���<���        �	x	qfc�A�!*

loss�z ;'u       �	��qfc�A�!*

lossw�;�U�       �	�Aqfc�A�!*

loss�Z<Ā�j       �	��qfc�A�!*

loss���;,��       �	-qfc�A�!*

lossdQ�<�us       �	Ϊqfc�A�!*

lossw2�<�/�`       �	Iqfc�A�!*

loss�S+=_�       �	��qfc�A�!*

loss�Ǚ<�6�W       �	�qfc�A�!*

loss#��:,.��       �	<.qfc�A�!*

loss2��<W�?`       �	��qfc�A�!*

loss�j[<��@�       �	�hqfc�A�!*

loss3<<�*͑       �	��qfc�A�!*

lossd�	=a�       �	�qfc�A�!*

loss�p-<����       �	9% qfc�A�!*

loss�/d<�WZ�       �	f� qfc�A�!*

loss#��<��P1       �	KX!qfc�A�!*

loss:e�<�|6b       �	��!qfc�A�!*

lossjB=Z�M�       �	P�"qfc�A�!*

losssW�;D�U3       �	�$#qfc�A�!*

loss��<���l       �	�#qfc�A�!*

loss�M;_^�       �		Q$qfc�A�!*

loss�_=t%^       �	��$qfc�A�!*

lossuw<o�[       �	��%qfc�A�!*

loss��<�9w�       �	D1&qfc�A�!*

loss�=YOJ       �	��&qfc�A�!*

lossa;�~c�       �	-|'qfc�A�!*

loss�zu<�\�       �	�(qfc�A�!*

loss��<���       �	N�(qfc�A�!*

lossw�;�F�       �	�P)qfc�A�!*

loss�;��I�       �	��)qfc�A�!*

loss�ӑ;	ĥ       �	H�*qfc�A�!*

loss:�0;��?�       �	�+qfc�A�!*

loss�lO<D6�1       �	�-qfc�A�!*

loss���<��i"       �	
�-qfc�A�!*

loss���;���{       �	��.qfc�A�!*

loss1xl<�{ܝ       �	P�/qfc�A�!*

loss]T<(��j       �	�x0qfc�A�!*

loss�ݎ;J?       �	@�1qfc�A�!*

lossn�F<��sy       �	@�2qfc�A�!*

loss��a<��X8       �	��3qfc�A�!*

loss�N ;��       �	Z�4qfc�A�!*

loss��s<;�Y       �	T;5qfc�A�!*

loss��=RJ�M       �	��5qfc�A�!*

loss��:D��       �	qq6qfc�A�!*

loss̈e<�c��       �	�7qfc�A�!*

loss��;<�Tɉ       �	s�7qfc�A�!*

loss�7�<�P��       �	�<8qfc�A�!*

loss��;�L�       �	�8qfc�A�!*

lossd��<%��       �	q9qfc�A�!*

loss2a;bi	       �	�:qfc�A�!*

loss��=C,�2       �		�:qfc�A�!*

lossr%<��       �	�@;qfc�A�!*

loss�Ti=_��       �	��;qfc�A�!*

loss���<7���       �	�o<qfc�A�!*

loss=s�=��/?       �	1	=qfc�A�!*

loss�Y<_��T       �	�=qfc�A�!*

loss	�f=��G       �	�Q>qfc�A�!*

lossZ�<���       �	6�>qfc�A�!*

loss��<�$^�       �	I�?qfc�A�!*

loss���;�X��       �	% @qfc�A�!*

lossi�<�� I       �	
�@qfc�A�!*

loss��(=ڦ}       �	�VAqfc�A�!*

loss��:���       �	��Aqfc�A�!*

loss���;�N}       �	��Bqfc�A�!*

loss#�= j��       �	�+Cqfc�A�!*

loss��V9��ׯ       �	��Cqfc�A�!*

loss�?�;(;.�       �	�]Dqfc�A�!*

lossj�X<���G       �	�Eqfc�A�!*

loss�R<�I�       �	5�Eqfc�A�!*

loss4��<q���       �	`>Fqfc�A�!*

loss�k�<~�$s       �	��Fqfc�A�!*

lossD=ze       �	~;Hqfc�A�!*

loss<Y���       �	Z�Hqfc�A�!*

losshX�<y9"       �	�xIqfc�A�!*

loss�<C�c!       �	 Jqfc�A�!*

loss�a�<6JJD       �	�Jqfc�A�!*

loss`�<����       �	�cKqfc�A�"*

loss��<��[�       �	�
Lqfc�A�"*

loss[~==ښ       �	X�Lqfc�A�"*

lossg�=Z /.       �	!XMqfc�A�"*

lossJ�<����       �	I�Nqfc�A�"*

loss�/�:q�,       �	�Oqfc�A�"*

lossl�D=�߫�       �	�bPqfc�A�"*

loss��/=��<M       �	�Qqfc�A�"*

loss� j<a��3       �	��Qqfc�A�"*

loss3�Y=��        �	URqfc�A�"*

loss�؅:�_�       �	r�Rqfc�A�"*

loss�9I<��#       �	c�Sqfc�A�"*

losssr{;��U�       �	�8Tqfc�A�"*

lossAs�<�_�       �	��Tqfc�A�"*

loss�n2=���       �	}zUqfc�A�"*

lossk�9տV       �	LVqfc�A�"*

loss��;m?-�       �	)�Vqfc�A�"*

loss�6_<ţ�       �	YQWqfc�A�"*

loss@�;5c.Z       �	`�Wqfc�A�"*

losslP=ۣ3c       �	r�Xqfc�A�"*

loss��<#%       �	�Yqfc�A�"*

loss!�'<��}       �	��Yqfc�A�"*

loss��<9��       �	DMZqfc�A�"*

loss@�Q:��<       �	��Zqfc�A�"*

losso��<8�H       �	Nz[qfc�A�"*

loss�oR<��M�       �	�\qfc�A�"*

loss��<�c/       �	`�\qfc�A�"*

loss��I;�|u�       �	kF]qfc�A�"*

loss1*;�J       �	��]qfc�A�"*

loss�l�:H�Q�       �	�^qfc�A�"*

loss�?�9���       �	]3_qfc�A�"*

loss��A='Wz�       �	�_qfc�A�"*

loss�V�;)� �       �	�x`qfc�A�"*

loss�<>/Έ       �	�Aaqfc�A�"*

loss���;<��%       �	L�aqfc�A�"*

loss�6;|@H�       �	�ybqfc�A�"*

loss��=�C�       �	:cqfc�A�"*

loss��Z<{p�       �	��cqfc�A�"*

lossP�=��a�       �	oddqfc�A�"*

loss�@=��       �	C eqfc�A�"*

loss� =���e       �	Йeqfc�A�"*

loss.��;=/x       �	G=fqfc�A�"*

loss��~:�Ls�       �	]�fqfc�A�"*

losstz-<JT       �	�zgqfc�A�"*

loss[u�;%B��       �	�hqfc�A�"*

loss�v;�E��       �	2�hqfc�A�"*

lossH�=X�-       �	pCiqfc�A�"*

loss�D}=/J�       �	�iqfc�A�"*

loss���<Ťf       �	�|jqfc�A�"*

loss��;��IT       �	�&kqfc�A�"*

losscNd<i#�       �	`>lqfc�A�"*

loss�m= c�P       �	��lqfc�A�"*

loss�7r=`�D�       �	�rmqfc�A�"*

loss��I<��w6       �	
nqfc�A�"*

loss6�<�|p�       �	ͯnqfc�A�"*

loss�ڡ=���       �	Ioqfc�A�"*

loss!Γ=5{R       �	;�oqfc�A�"*

lossmO�;��l&       �	q�pqfc�A�"*

lossw�%<����       �	zqqfc�A�"*

loss6��:�h��       �	�rqfc�A�"*

lossH�W=Uj/       �	¥rqfc�A�"*

loss�b<s�       �	9sqfc�A�"*

loss�t;L��G       �	S�sqfc�A�"*

losse$�;�M(n       �	wetqfc�A�"*

lossɩ�<��Dr       �	�Huqfc�A�"*

loss��>TtX�       �	}�uqfc�A�"*

loss�}+=��ő       �	7�vqfc�A�"*

lossSc=�F�S       �	� wqfc�A�"*

lossr�<F�       �	��wqfc�A�"*

loss�;��iS       �	�axqfc�A�"*

loss!%�;#^�       �	� yqfc�A�"*

lossF<K��       �	J�yqfc�A�"*

loss�
`=�-��       �	0zqfc�A�"*

loss���:p��       �	��zqfc�A�"*

loss�M%=x�V�       �	�n{qfc�A�"*

loss[�<-�oX       �	E|qfc�A�"*

loss,~<E��u       �	ۣ|qfc�A�"*

loss�@=���T       �	*8}qfc�A�"*

loss.H[<��.�       �	��}qfc�A�"*

loss�m�<�D\�       �	vn~qfc�A�"*

loss��a<����       �	dqfc�A�"*

lossdP�<H2�       �	̙qfc�A�"*

loss�e�;���       �	.�qfc�A�"*

loss�9=xU�       �	ƀqfc�A�"*

loss���<��m�       �	KY�qfc�A�"*

loss�ӄ<��J       �	��qfc�A�"*

loss6��<94�       �	σ�qfc�A�"*

loss�Y�=�l�       �	��qfc�A�"*

loss��:���G       �	�Ãqfc�A�"*

loss�1 <Y
9       �	}Y�qfc�A�"*

lossѵ#<��6f       �	���qfc�A�"*

loss��f<��7(       �	�\�qfc�A�"*

lossXh=yde�       �	3��qfc�A�"*

loss&�=X���       �	���qfc�A�"*

loss �<�x�       �	NB�qfc�A�"*

loss��;�E�%       �	|�qfc�A�"*

lossX}�:���>       �	%��qfc�A�"*

loss
�G<zA�       �	G=�qfc�A�"*

loss���<]N�]       �	�qfc�A�"*

loss�H�<	�ԯ       �	�qfc�A�"*

loss��=��B       �	�6�qfc�A�"*

loss��v=����       �	�Ԏqfc�A�"*

lossƴF;�uq�       �	;n�qfc�A�"*

loss,��:�U�       �	|e�qfc�A�"*

loss��w;�ϏT       �	嘑qfc�A�"*

loss4�O=��f       �	�f�qfc�A�"*

loss��;��,�       �	� �qfc�A�"*

loss�;�N6�       �	(��qfc�A�"*

lossa�<����       �	�6�qfc�A�"*

loss�=�G5q       �	Iڕqfc�A�"*

lossY�
<�V��       �	dx�qfc�A�"*

lossQf<ȼ7�       �	��qfc�A�"*

lossr�-=Ϟ       �	%��qfc�A�"*

lossHh1=�_       �	i��qfc�A�"*

lossN5\;?��       �	R��qfc�A�"*

loss�7=,�!f       �	�@�qfc�A�"*

loss�<oE�       �	Uߛqfc�A�"*

lossL��<�m�       �	���qfc�A�"*

loss�I�;{@M�       �	"�qfc�A�"*

loss�$a<f�<t       �	�Ýqfc�A�"*

loss?"�<�mb�       �	�c�qfc�A�"*

loss�;M<{y"       �	s�qfc�A�"*

loss��<ɔ�0       �	:��qfc�A�"*

lossS�=�W��       �	�G�qfc�A�"*

loss?��=�v�       �	=�qfc�A�"*

loss1%�;V�J�       �	���qfc�A�"*

loss��m;���T       �	�A�qfc�A�"*

loss���;3� ]       �	�ߢqfc�A�"*

lossM(y<!��5       �	��qfc�A�#*

loss�̟:&q{�       �	��qfc�A�#*

loss�8�<�r�       �	���qfc�A�#*

loss, ;I1       �	�X�qfc�A�#*

loss�0�<����       �	���qfc�A�#*

loss��k=J8W       �	B��qfc�A�#*

loss�*�;f���       �	>�qfc�A�#*

lossJ��<h*ϭ       �	@ڧqfc�A�#*

loss�`�;dzV�       �	�v�qfc�A�#*

loss�H�:��B�       �	��qfc�A�#*

loss�a-;v��I       �	㤩qfc�A�#*

lossֲ�:���       �	xB�qfc�A�#*

loss��x=���       �	+ުqfc�A�#*

loss��
<|�]}       �	v�qfc�A�#*

loss)��:�rF�       �	X�qfc�A�#*

loss�$;er2�       �	h��qfc�A�#*

loss���<�m��       �	v�qfc�A�#*

loss��: �[�       �	e��qfc�A�#*

loss�I�=�A��       �	�"�qfc�A�#*

loss �<R�       �	¯qfc�A�#*

loss<)6<���       �	�h�qfc�A�#*

loss��E=��<       �	�	�qfc�A�#*

loss���;'I.9       �	ȱ�qfc�A�#*

lossH}�;���       �	nN�qfc�A�#*

lossMZ=&       �	-�qfc�A�#*

lossc�<4���       �	m��qfc�A�#*

loss�i�=1���       �	-�qfc�A�#*

loss��:;!Iym       �	дqfc�A�#*

lossx�Z<#\�       �	�t�qfc�A�#*

loss��;����       �	��qfc�A�#*

loss˹<��.R       �	*��qfc�A�#*

loss��;<N!�       �	׷qfc�A�#*

lossf��<��XR       �	�p�qfc�A�#*

loss��<Mh�       �	1%�qfc�A�#*

loss��;�C�O       �	���qfc�A�#*

loss��;��;       �	ٰ�qfc�A�#*

lossc�n=��f       �	�G�qfc�A�#*

loss�l;�n�"       �	ܻqfc�A�#*

loss�;�O�M       �	�q�qfc�A�#*

loss�v;�P       �	�qfc�A�#*

lossc:<�m��       �	ެ�qfc�A�#*

loss
Y<�|�+       �	�C�qfc�A�#*

loss c=w��       �	־qfc�A�#*

loss��<�jh7       �	�j�qfc�A�#*

loss��/=�a��       �	G�qfc�A�#*

loss1Ͽ;�ؽO       �	���qfc�A�#*

loss@T�<���n       �	�8�qfc�A�#*

loss�(t<i��       �	���qfc�A�#*

loss,u�;!�O       �	Qi�qfc�A�#*

loss�<Aa�E       �	��qfc�A�#*

loss)K=w{t       �	��qfc�A�#*

loss(<d��       �	�9�qfc�A�#*

loss;��;�*M       �	���qfc�A�#*

loss�<�� �       �	n�qfc�A�#*

loss8��;N=��       �	��qfc�A�#*

loss�M=��4�       �	��qfc�A�#*

loss�k;
�Ǫ       �	�=�qfc�A�#*

loss�ɛ<���)       �	g��qfc�A�#*

loss���<�ϼ       �	"l�qfc�A�#*

lossA�=�E��       �	��qfc�A�#*

loss�U6;DpH       �	G��qfc�A�#*

lossRX,=ur7       �	*V�qfc�A�#*

loss)��<�?       �	��qfc�A�#*

loss���:����       �	{��qfc�A�#*

lossoDi;�qH�       �	vQ�qfc�A�#*

loss�_p=�ǉ�       �	���qfc�A�#*

loss���<�dT�       �	6Z�qfc�A�#*

loss���;g"�       �	�/�qfc�A�#*

loss�=cפ�       �	��qfc�A�#*

loss�G0=p�>i       �	ʧ�qfc�A�#*

loss��W;�"       �	�H�qfc�A�#*

loss��:���:       �	y��qfc�A�#*

loss��;�f2R       �	���qfc�A�#*

loss���<����       �	�V�qfc�A�#*

lossL�;yy�       �	���qfc�A�#*

loss��<}Р�       �	���qfc�A�#*

lossE/F;E��z       �	/0�qfc�A�#*

loss>��<�V�       �	h��qfc�A�#*

loss�KZ<q�S       �	�_�qfc�A�#*

loss	�=�?��       �	��qfc�A�#*

lossW��:��d       �	ӟ�qfc�A�#*

loss�!<]�D       �	�5�qfc�A�#*

loss��w=�*f       �	X��qfc�A�#*

loss���<7�N�       �	�Z�qfc�A�#*

loss-��<�VJc       �	g��qfc�A�#*

loss��<ڦ�F       �	�k�qfc�A�#*

loss�&�=���       �	���qfc�A�#*

losslǠ=�e�       �	�+�qfc�A�#*

loss7>V<���`       �	���qfc�A�#*

loss�(m:Wi�       �	^c�qfc�A�#*

loss-�:�t!a       �	G8�qfc�A�#*

loss��=:�g�       �	*�qfc�A�#*

lossT��:�ޟ       �	��qfc�A�#*

losss�=*o�       �	�D�qfc�A�#*

lossiȹ:�h�       �	���qfc�A�#*

loss<��:KDh�       �	ms�qfc�A�#*

loss��K;B���       �	m�qfc�A�#*

lossX$�=�X v       �	��qfc�A�#*

loss��<��       �	�E�qfc�A�#*

lossL�=�>̵       �	���qfc�A�#*

lossQLD;�6Q�       �	c{�qfc�A�#*

loss�=<j��       �	��qfc�A�#*

loss��L=����       �	��qfc�A�#*

lossE[=��       �	�L�qfc�A�#*

loss(�;-L`�       �	n��qfc�A�#*

loss��'<۫ҝ       �	�q�qfc�A�#*

loss��<Hix       �	B
�qfc�A�#*

loss���<��b�       �	Ԟ�qfc�A�#*

loss}$R=��O       �	�/�qfc�A�#*

lossL�o;�ʦ       �	�n�qfc�A�#*

loss��}<Li�R       �	Nd�qfc�A�#*

loss��k;�@�       �	��qfc�A�#*

loss�fO<���       �	]Q�qfc�A�#*

lossj��;9k�k       �	��qfc�A�#*

loss6�/<a:       �	_��qfc�A�#*

lossa��;r�R�       �	�v�qfc�A�#*

loss =�~�S       �	6��qfc�A�#*

loss�3:��_'       �	F(�qfc�A�#*

loss2ی;9�A       �	��qfc�A�#*

lossE|�<�q
       �	���qfc�A�#*

loss��!<�W�Y       �	[��qfc�A�#*

loss3��;)Yx�       �	�9�qfc�A�#*

loss�;<C�I       �	I��qfc�A�#*

loss�4�<Y�       �	>{�qfc�A�#*

loss��<z:S       �	�qfc�A�#*

loss�(�=G\$�       �	M��qfc�A�#*

loss���<��u�       �	\�qfc�A�#*

lossy��<��       �	��qfc�A�#*

loss��;�ރF       �	O��qfc�A�$*

loss�4 <�q$,       �	)Y�qfc�A�$*

lossŎU=�       �	b��qfc�A�$*

loss�;=i2��       �	c��qfc�A�$*

loss��>����       �	�=�qfc�A�$*

loss��h;�J��       �	���qfc�A�$*

loss�x�<�M*       �	]l�qfc�A�$*

loss�\=A,�       �	�qfc�A�$*

lossa�=<m���       �	��qfc�A�$*

loss$@�;4�r2       �	R rfc�A�$*

loss_�;Ɏ�F       �	�� rfc�A�$*

lossf��<��       �	�rfc�A�$*

loss�jk=C�޵       �	O#rfc�A�$*

loss�;=:�n       �	0�rfc�A�$*

loss�U�<��       �	Nrfc�A�$*

loss���;��#       �	��rfc�A�$*

loss�3�<�F�2       �	��rfc�A�$*

loss���<]���       �	rfc�A�$*

loss#'P<��1z       �	��rfc�A�$*

lossQ99<�̶�       �	�Krfc�A�$*

lossf�<h7`       �	Y�rfc�A�$*

loss)�
<�       �	X�rfc�A�$*

lossf��;c_�       �	]7rfc�A�$*

lossZm<\��       �	c�rfc�A�$*

loss$�<�t�       �	w	rfc�A�$*

lossX�;�aoe       �	6
rfc�A�$*

lossoþ<v 8J       �	��
rfc�A�$*

lossM+�<3��5       �	W#rfc�A�$*

loss��W=�k�       �	~�rfc�A�$*

loss�t�;�X�n       �	�hrfc�A�$*

loss6E�<iN8"       �	��rfc�A�$*

loss�!!;�*�       �	�rfc�A�$*

loss��<���       �	]7rfc�A�$*

loss/�<��       �	��rfc�A�$*

loss�&= c�       �	D�rfc�A�$*

loss.�<��,D       �	�"rfc�A�$*

loss��=�ۢ       �	�[rfc�A�$*

loss��/=��W$       �	& rfc�A�$*

loss�H'<,�       �	��rfc�A�$*

lossJԀ=�i�Z       �	q;rfc�A�$*

loss��;�%V=       �	��rfc�A�$*

loss�1�<QS�        �	�rrfc�A�$*

loss�|�;���       �	�
rfc�A�$*

loss\Y}<f�׏       �	v�rfc�A�$*

lossϮ�;�rr�       �	<rfc�A�$*

loss��=<� �8       �	��rfc�A�$*

loss�L�=)5�<       �	werfc�A�$*

loss}��=�i�M       �	�rfc�A�$*

loss�"w=�˦       �	őrfc�A�$*

loss�~�:��\�       �	(,rfc�A�$*

loss���;2tg�       �	+�rfc�A�$*

lossJ�b<Ѫ�       �	h]rfc�A�$*

loss:�<O�H�       �	;�rfc�A�$*

loss�o<���       �	�rfc�A�$*

lossș�;N�       �	�1rfc�A�$*

loss��<Qy�k       �	�rfc�A�$*

losssX�<���       �	�N rfc�A�$*

loss��
<6���       �	
� rfc�A�$*

loss�)�:�ߜ�       �	V�!rfc�A�$*

lossS��<��}$       �	~8"rfc�A�$*

loss�= ;vJd�       �	��"rfc�A�$*

loss�U{<�PD       �	�d#rfc�A�$*

lossW�<�q:O       �	z�#rfc�A�$*

loss��=�       �	p�$rfc�A�$*

loss:��<D�j       �	1%rfc�A�$*

loss�s<!� �       �	��%rfc�A�$*

lossT/v<�@m�       �	c&rfc�A�$*

loss5��;m�l       �	�&rfc�A�$*

loss!H^=�9!       �	#�(rfc�A�$*

loss�K�;Y/�F       �	�{)rfc�A�$*

lossj�<�2u       �	�*rfc�A�$*

lossE<�-�       �	Ͱ*rfc�A�$*

loss!з=0�C\       �	�k+rfc�A�$*

lossz��=�8o       �	��,rfc�A�$*

loss�B�<o�R       �	��-rfc�A�$*

lossZ=:����       �	�.rfc�A�$*

loss��K=��֘       �	Kv/rfc�A�$*

loss�S�=�G��       �	q0rfc�A�$*

lossVI�:�3�i       �	��0rfc�A�$*

losss��;���       �	0c1rfc�A�$*

loss� �=M�K       �	�2rfc�A�$*

losss=�Hyh       �	^�2rfc�A�$*

lossLs�;1��D       �	�?3rfc�A�$*

loss*�<����       �	�3rfc�A�$*

loss�Vi<g�f�       �	�q4rfc�A�$*

loss��9���       �	-�5rfc�A�$*

loss\�<���       �	VH6rfc�A�$*

lossa�W;����       �	4�6rfc�A�$*

loss�A=���2       �	V�7rfc�A�$*

loss�K<=��       �	�K8rfc�A�$*

loss�э;A�P       �	B�8rfc�A�$*

loss$�5<�}�       �	��9rfc�A�$*

loss��<6�       �	�7:rfc�A�$*

loss���;���       �	��:rfc�A�$*

loss1�H<M9�	       �	.v;rfc�A�$*

losss�?<u�K$       �	~<rfc�A�$*

lossvU�;�Lz       �	S�<rfc�A�$*

loss;1�<��H�       �	�J=rfc�A�$*

loss]�=GW       �	�=rfc�A�$*

loss��<'��R       �	x�>rfc�A�$*

loss�Ij<M��       �	d=?rfc�A�$*

lossa��<*��w       �	8�?rfc�A�$*

loss)�<�
       �	�{@rfc�A�$*

loss*�<���L       �	wArfc�A�$*

lossJ�<�IX�       �	ŭArfc�A�$*

loss��<����       �	$DBrfc�A�$*

loss/��<Т �       �	w�Brfc�A�$*

loss���;��4       �	�tCrfc�A�$*

loss�LL<����       �	hDrfc�A�$*

loss�X�=��4       �	$�Drfc�A�$*

loss���;�H�       �	z6Erfc�A�$*

loss�w<M2�       �	��Erfc�A�$*

loss�g2<+�X       �	/�Frfc�A�$*

lossQq�:�A�U       �	W'Grfc�A�$*

loss�=ʓ��       �	�Grfc�A�$*

loss�|�;�S       �	�QHrfc�A�$*

lossc:�<^1       �	��Hrfc�A�$*

loss�H�<��~       �	SxIrfc�A�$*

loss�@p<<F�       �	Jrfc�A�$*

losssQ�<8��w       �	H�Jrfc�A�$*

loss�d�;Z�X�       �	c~Krfc�A�$*

lossT��;#��N       �	�Lrfc�A�$*

lossʶ]<��g       �	"Mrfc�A�$*

loss���:�W-       �	�*Nrfc�A�$*

loss�	?<�E��       �	Y�Nrfc�A�$*

loss��!;����       �	�Orfc�A�$*

lossHt89�FkN       �	�Prfc�A�$*

loss��);A)�[       �	5DQrfc�A�$*

loss�<��3       �	�Qrfc�A�%*

loss�?�<#2!       �	�Rrfc�A�%*

loss��H;B\��       �	;Srfc�A�%*

loss,=�9� �K       �	��Srfc�A�%*

loss��O;#u�       �	�}Trfc�A�%*

lossS�<�{�a       �	UUrfc�A�%*

lossS669ߚ�h       �	�Urfc�A�%*

loss���9�f       �	�BVrfc�A�%*

loss!��:m��t       �	�Vrfc�A�%*

loss���;��        �	BxWrfc�A�%*

loss` <�R�T       �	3Xrfc�A�%*

lossr��:���q       �	ϽXrfc�A�%*

loss ��<�㬯       �	�Q[rfc�A�%*

loss��<��       �	F�[rfc�A�%*

loss�g9�p�       �	�\rfc�A�%*

loss��">��aU       �	y]rfc�A�%*

loss{�|=ph�       �	_%^rfc�A�%*

loss�>9=����       �	��^rfc�A�%*

lossO�<׀�       �	\q_rfc�A�%*

lossצ�:��ؠ       �	`rfc�A�%*

loss���<��/m       �	�`rfc�A�%*

loss�R<����       �	Oarfc�A�%*

lossT71;|��0       �	F�arfc�A�%*

losso�<`��       �	��brfc�A�%*

loss�-<t��&       �	�$crfc�A�%*

loss��<�{�U       �	�crfc�A�%*

loss��=�Q       �	�Udrfc�A�%*

lossH��;��xY       �	'�drfc�A�%*

lossZ�/<�C�w       �	A�erfc�A�%*

lossR�!=��P       �	�Ffrfc�A�%*

loss��<��%�       �	p�frfc�A�%*

loss��;)~�       �	ѐgrfc�A�%*

loss��=���       �	�3hrfc�A�%*

loss�
Y;	���       �	�hrfc�A�%*

loss��<72_�       �	:xirfc�A�%*

loss��y<�Δ�       �	�jrfc�A�%*

loss�<F<���       �	7�jrfc�A�%*

lossG�;t�s�       �	v�krfc�A�%*

loss,��;+��       �	�5lrfc�A�%*

loss���;ni��       �	F�lrfc�A�%*

lossң<�M�f       �	\vmrfc�A�%*

loss��3<Ŝr�       �	�nrfc�A�%*

loss�^=<���       �	иnrfc�A�%*

loss�P�<\#�>       �	�Yorfc�A�%*

lossz߈<I�       �	3�orfc�A�%*

loss�n�;�:$�       �	͐prfc�A�%*

loss:;�Ҳ       �	�3qrfc�A�%*

lossS�n;B�L9       �	��qrfc�A�%*

loss�m�<�V`       �	�errfc�A�%*

loss�u�:���       �	�rrfc�A�%*

loss	�;�A:       �	Ĕsrfc�A�%*

loss�%1<�K�       �	(,trfc�A�%*

lossF�$=�3:V       �	��trfc�A�%*

lossOֈ;�x2       �	�]urfc�A�%*

loss�#j;�-5?       �	��urfc�A�%*

loss��;W��e       �	�vrfc�A�%*

loss$p{<%��       �	y"wrfc�A�%*

loss9P�;v6�       �	@�wrfc�A�%*

loss�;��1$       �	\xrfc�A�%*

loss�S=Ͽʔ       �	>yrfc�A�%*

lossO��=�P@       �	��yrfc�A�%*

loss͑�:F��       �	O;zrfc�A�%*

loss}W�;��eM       �	8�zrfc�A�%*

lossC(�:��2z       �	dy{rfc�A�%*

loss�QI<����       �	�|rfc�A�%*

lossJ6�:	g       �	���rfc�A�%*

loss���;?�O       �	T5�rfc�A�%*

loss��;��i�       �	�Ɠrfc�A�%*

loss�R\=�/�       �	�]�rfc�A�%*

loss�)8<,�+       �	��rfc�A�%*

loss7�*=�ew       �	�Օrfc�A�%*

loss\[=oo¯       �	*s�rfc�A�%*

loss��;��e^       �	��rfc�A�%*

loss�=K��       �	ߩ�rfc�A�%*

loss�Qb=]"�       �	NF�rfc�A�%*

loss��&<��o       �	�rfc�A�%*

loss��;��vj       �	ǝ�rfc�A�%*

loss(g<ϊ�       �	|B�rfc�A�%*

loss���:�_�/       �	2�rfc�A�%*

lossC	�;��\K       �	���rfc�A�%*

loss�i;�?       �	�"�rfc�A�%*

lossG:_.��       �	>̜rfc�A�%*

loss[U=;�5	       �	�d�rfc�A�%*

loss ck;���       �	u�rfc�A�%*

loss�,=��       �	x��rfc�A�%*

loss�ؑ<�@�       �	�2�rfc�A�%*

loss��=��       �	-ҟrfc�A�%*

loss���:���o       �	�v�rfc�A�%*

loss���<h1n       �	
�rfc�A�%*

loss��;�j�       �	E��rfc�A�%*

loss��f;Ţ+@       �	�?�rfc�A�%*

loss��;R�r       �	s֢rfc�A�%*

loss�:d;\j��       �	�v�rfc�A�%*

loss�??=?��       �	��rfc�A�%*

loss`�=Yc�        �	s��rfc�A�%*

loss���<vǥ�       �	LQ�rfc�A�%*

loss1!,;��v�       �	[�rfc�A�%*

loss�:Ͻ[6       �		��rfc�A�%*

loss:9�<J	�       �	�$�rfc�A�%*

loss]��;
#��       �	"��rfc�A�%*

loss[><lsX       �	X�rfc�A�%*

loss��#<�ܾ�       �	7��rfc�A�%*

loss�Ey<�xz       �	'��rfc�A�%*

loss���<�ĀU       �	$G�rfc�A�%*

loss��3<�q�L       �	ܪrfc�A�%*

loss-k:'�       �	Ww�rfc�A�%*

loss��<��.       �	rO�rfc�A�%*

loss�]<��4&       �	���rfc�A�%*

loss���;�t�o       �	�ݮrfc�A�%*

loss�|�<>�f�       �	�~�rfc�A�%*

lossrZ<��       �	O$�rfc�A�%*

loss��<�3��       �	]��rfc�A�%*

loss}�<��;       �	�b�rfc�A�%*

lossQ�;�Ǒ�       �	���rfc�A�%*

loss��9=bI4       �	_��rfc�A�%*

loss?�n:�t��       �	�4�rfc�A�%*

loss��#<(zB`       �	�γrfc�A�%*

lossi�<ʎ3       �	�g�rfc�A�%*

loss��8=���       �	���rfc�A�%*

lossdT�<�Q�}       �	���rfc�A�%*

loss6::N�=�       �	�+�rfc�A�%*

loss���;5�|�       �	�ƶrfc�A�%*

loss�=j8�%�?       �	�Z�rfc�A�%*

loss�3=�fcc       �	=�rfc�A�%*

loss�F�<P%7u       �	��rfc�A�%*

lossa>�!�       �	v�rfc�A�%*

loss�9�<��D       �	ѭ�rfc�A�%*

lossW��:j��       �	xD�rfc�A�%*

loss�*�<C'�       �	Qںrfc�A�&*

loss(Wu;'U�       �	�q�rfc�A�&*

loss�Z:RV�V       �	_�rfc�A�&*

loss5=3�j�       �	d��rfc�A�&*

loss��<�E"�       �	"Q�rfc�A�&*

loss���=0�       �	�rfc�A�&*

lossx�V<�:�       �	s��rfc�A�&*

loss;�n<�x�       �	m�rfc�A�&*

loss�|�<��       �	�rfc�A�&*

lossRR�;�ܓ       �	KX�rfc�A�&*

lossl��<�k=�       �	0��rfc�A�&*

loss��<�P7�       �	���rfc�A�&*

loss,͋:z>�       �	�/�rfc�A�&*

lossE��<��"^       �	t��rfc�A�&*

loss��<����       �	{h�rfc�A�&*

loss�%?;� �       �	�|�rfc�A�&*

loss.�;	�       �	��rfc�A�&*

loss���;g�D[       �	մ�rfc�A�&*

loss6�<��Ȕ       �	�W�rfc�A�&*

loss:x5<֍k       �	s��rfc�A�&*

loss�I�;���       �	���rfc�A�&*

loss)��;���H       �	�.�rfc�A�&*

loss�=@Hx�       �	T�rfc�A�&*

lossx��<�]4y       �	k��rfc�A�&*

loss�n�<\�3�       �	Ja�rfc�A�&*

lossYc�<�$a�       �	�rfc�A�&*

loss�%�;y_$       �	���rfc�A�&*

loss� �<ً	       �	˅�rfc�A�&*

lossa�+=�(��       �	�rfc�A�&*

loss���:(��       �	0d�rfc�A�&*

lossj�E=�1Yl       �	$�rfc�A�&*

loss�m=��`�       �	��rfc�A�&*

loss�:���       �	�Y�rfc�A�&*

loss��<�SP       �	���rfc�A�&*

loss	� <�B1       �	�X�rfc�A�&*

loss�`�<ۆZ       �	��rfc�A�&*

loss7Qb;O��       �	ɮ�rfc�A�&*

loss�=# b�       �	S�rfc�A�&*

loss.a;��5�       �	��rfc�A�&*

loss�
�;��       �	��rfc�A�&*

lossr�<ߣϺ       �	�M�rfc�A�&*

loss��=*�0'       �	��rfc�A�&*

loss �<ŢG       �	<��rfc�A�&*

loss4�?;w֩�       �	|b�rfc�A�&*

lossy�;ɋ+�       �	��rfc�A�&*

lossm�_<�R�       �	ɮ�rfc�A�&*

loss�J�<��gt       �	�O�rfc�A�&*

loss��>J��9       �	1��rfc�A�&*

loss*ޛ<�       �	��rfc�A�&*

loss�e�<&1��       �	�.�rfc�A�&*

loss�M;�	�	       �	���rfc�A�&*

loss@)<6��a       �	w�rfc�A�&*

loss4�3<���       �	��rfc�A�&*

loss ��:x��R       �	x��rfc�A�&*

loss���<WAB�       �	�c�rfc�A�&*

loss�k=����       �	��rfc�A�&*

lossF�;N�Z<       �	���rfc�A�&*

lossi?�;(ͫ       �	er�rfc�A�&*

loss��I<CSZ�       �	��rfc�A�&*

loss��l=C��       �	l��rfc�A�&*

loss��P;kNp�       �	fO�rfc�A�&*

lossa: �{�       �	,��rfc�A�&*

loss��^:�֏�       �	ő�rfc�A�&*

loss�2U<�H��       �	{3�rfc�A�&*

loss�o[<��QV       �	���rfc�A�&*

loss���<��T       �	�`�rfc�A�&*

lossA�`;�6{       �	���rfc�A�&*

loss5<��0       �	l��rfc�A�&*

loss�j�<�	��       �	�/�rfc�A�&*

loss�D<���       �	H��rfc�A�&*

loss�=K�Z       �	Rb�rfc�A�&*

loss%$m<�Icd       �	:�rfc�A�&*

loss��.<�=S�       �	(��rfc�A�&*

loss-H =r4��       �	E�rfc�A�&*

loss���:5t��       �	��rfc�A�&*

loss�?;�gβ       �	�N�rfc�A�&*

loss��;��U       �	s��rfc�A�&*

loss	�S;�"'�       �	��rfc�A�&*

lossL
=��       �	uU�rfc�A�&*

lossGu<��[l       �	���rfc�A�&*

loss�W�<�zz       �	���rfc�A�&*

loss{��;���h       �	:<�rfc�A�&*

losso�;�*�       �	v��rfc�A�&*

loss�1<���h       �	��rfc�A�&*

loss�T;���       �	�2�rfc�A�&*

loss���;k���       �	Q��rfc�A�&*

lossj!<�N�o       �	0��rfc�A�&*

loss��t=�g��       �	#.�rfc�A�&*

loss$͕<�s`?       �	���rfc�A�&*

loss{�<cpR       �	���rfc�A�&*

loss�k�<�#{       �	/O�rfc�A�&*

lossm={~��       �	���rfc�A�&*

loss�>�:K�W�       �	�~�rfc�A�&*

lossx_p:�:��       �	<�rfc�A�&*

lossC
H<xw       �	+��rfc�A�&*

loss�se<�@_�       �	 S�rfc�A�&*

loss��W<�M�       �	��rfc�A�&*

loss��<cJ��       �	Z��rfc�A�&*

loss T�;7���       �	�Y�rfc�A�&*

loss�`<��C�       �	4��rfc�A�&*

loss��\;6�G�       �	j��rfc�A�&*

lossj\V<��03       �	kI sfc�A�&*

loss�Q_<�с4       �	G� sfc�A�&*

loss��;�v�|       �	��sfc�A�&*

losslO�<��=
       �	asfc�A�&*

loss�`<Am��       �	}�sfc�A�&*

lossl�<�E6�       �	�Fsfc�A�&*

loss&Ѽ<ޮ>�       �	2�sfc�A�&*

loss��;Bl��       �	g�sfc�A�&*

loss4bM;-A�8       �	�sfc�A�&*

loss�=�S��       �	��sfc�A�&*

loss�d�8Ec�       �	�Osfc�A�&*

loss���<��ce       �	��sfc�A�&*

lossB<c�;�       �	��sfc�A�&*

loss��=I�       �	_)sfc�A�&*

loss�s�<(�o�       �	L�sfc�A�&*

lossؘ�:;Y6       �	d\	sfc�A�&*

loss֮�<�
S       �	c�	sfc�A�&*

loss�5=����       �	�
sfc�A�&*

loss��<��e       �	�sfc�A�&*

loss%u;ֲ�m       �	N�sfc�A�&*

lossS�<�E,�       �	kJsfc�A�&*

loss���:�j�       �	��sfc�A�&*

lossL/{=�6       �	|�sfc�A�&*

loss%T=�F�       �	+sfc�A�&*

loss-�==�	�       �	V�sfc�A�&*

lossvxT<�
X�       �	7Psfc�A�&*

loss���;Y���       �	��sfc�A�&*

loss��V=�S�7       �	�~sfc�A�'*

loss��<S�       �	�sfc�A�'*

loss�V <��       �	X�sfc�A�'*

loss;B�:ڥ\�       �	��sfc�A�'*

lossw4 ;*^�       �	7sfc�A�'*

loss;f
:"h�v       �	_�sfc�A�'*

loss@8�<\�ʁ       �	DLsfc�A�'*

loss��Q;��h�       �	7�sfc�A�'*

losstt�<I�Y       �	��sfc�A�'*

lossm<�;�q��       �	 ;sfc�A�'*

lossT��<B��!       �	��sfc�A�'*

loss��/<�,z       �	�ysfc�A�'*

loss$g�;Q��L       �	�sfc�A�'*

lossi%;����       �	ĳsfc�A�'*

loss��<�׼�       �		Osfc�A�'*

loss"�=7}��       �	5�sfc�A�'*

loss&��<
x       �	Q�sfc�A�'*

loss���;~��t       �	�(sfc�A�'*

lossh�7=D9kv       �	�sfc�A�'*

loss�j<� ~�       �	vpsfc�A�'*

lossew=V
4�       �	sfc�A�'*

loss��s:�#�t       �	��sfc�A�'*

loss���<�.�       �	�Csfc�A�'*

loss�2_=α�       �	��sfc�A�'*

loss��=Q�ޫ       �	p� sfc�A�'*

loss/9�=8�g�       �	3!sfc�A�'*

loss��<h�7e       �	��!sfc�A�'*

loss�4n=��^�       �	�i"sfc�A�'*

loss}��<�7t       �	�"sfc�A�'*

lossZ<l;ids^       �	��#sfc�A�'*

loss���;���       �	0+$sfc�A�'*

loss&r�<�Crj       �	׿$sfc�A�'*

lossH�<<�\       �	T%sfc�A�'*

loss�78;Ǡ@|       �	\�%sfc�A�'*

loss:;��$       �	�'sfc�A�'*

loss�Y=����       �	��'sfc�A�'*

loss�<=��       �	 z(sfc�A�'*

loss��;�ӫF       �	�#)sfc�A�'*

loss|�S=��       �	��)sfc�A�'*

loss�3<���Z       �	��*sfc�A�'*

lossƄ�:�;       �	
-+sfc�A�'*

loss3�=��l~       �	�j,sfc�A�'*

loss!�=����       �	-sfc�A�'*

losslv=�s       �	R�-sfc�A�'*

loss�;f�       �	�X.sfc�A�'*

loss���<�J�       �	�/sfc�A�'*

loss�~<	�,m       �	$�/sfc�A�'*

loss�#=���       �	P�0sfc�A�'*

lossC�r;r�|�       �	X1sfc�A�'*

loss�bg;��yD       �	
�1sfc�A�'*

loss���<�f��       �	iQ2sfc�A�'*

loss� /=B��       �	%�2sfc�A�'*

loss��;1m�=       �	��3sfc�A�'*

loss�c;�u��       �	�`4sfc�A�'*

loss�>�<Ts6�       �	�E5sfc�A�'*

loss��<�/%       �	��5sfc�A�'*

lossxg�;�=�       �	�l6sfc�A�'*

loss�C!=���       �	�7sfc�A�'*

loss`�<�7��       �	��7sfc�A�'*

loss�B,<�K�       �	4�8sfc�A�'*

lossI��<y.��       �	�<9sfc�A�'*

lossV{�<D���       �	[�9sfc�A�'*

loss,�;cT;�       �	�q:sfc�A�'*

lossL��;9L,       �	J	;sfc�A�'*

loss�<Ȳ�       �	z�;sfc�A�'*

loss���< �­       �	�U<sfc�A�'*

lossأ�;^��       �	=sfc�A�'*

loss�:�!�       �	D�=sfc�A�'*

loss(�A=$4gi       �	�?>sfc�A�'*

loss80�;�3��       �	~�>sfc�A�'*

loss���<Z/A       �	��?sfc�A�'*

loss�r^=P̔�       �	�0@sfc�A�'*

loss��=���       �	_�@sfc�A�'*

loss��r<�s�       �	lAsfc�A�'*

loss�zM<���       �	KBsfc�A�'*

lossm�<p"�       �	�Bsfc�A�'*

lossn��;�6�       �	�8Csfc�A�'*

lossć�<s��E       �	M�Csfc�A�'*

lossϷ$<�w       �	�mDsfc�A�'*

loss
p<v���       �	WEsfc�A�'*

losst��;�Q�       �	b�Esfc�A�'*

loss�)=��iA       �	�;Fsfc�A�'*

loss��=�X�<       �	��Fsfc�A�'*

loss�A=�-K       �	�|Gsfc�A�'*

lossm`<�,�       �	]Hsfc�A�'*

lossxy�;��yu       �	�Hsfc�A�'*

loss��;=��J       �	�DIsfc�A�'*

loss�<{�&~       �	s�Isfc�A�'*

lossDI<B]q       �	�qJsfc�A�'*

loss��; �h       �	U.Ksfc�A�'*

loss��*=���       �	;�Ksfc�A�'*

loss���;^fos       �	AcLsfc�A�'*

lossC-S=}��        �	& Msfc�A�'*

loss�s�<@��       �	`�Msfc�A�'*

loss��;H֪=       �	,Nsfc�A�'*

loss�Z�<��(v       �	��Nsfc�A�'*

loss�>�:���       �	(cOsfc�A�'*

loss�Z�<��       �	s�Osfc�A�'*

lossS��;9�bZ       �	�Psfc�A�'*

loss��\:��-�       �	� Qsfc�A�'*

lossk��:�]�       �	4�Qsfc�A�'*

loss =�q]�       �	}Ssfc�A�'*

lossn�v;�+��       �	�Ssfc�A�'*

loss��=*�-�       �	l=Tsfc�A�'*

lossjNB<�       �	��Tsfc�A�'*

loss��<N�       �	�Usfc�A�'*

lossZ�O<�H�       �	Y1Vsfc�A�'*

loss��<�ᒄ       �	A�Vsfc�A�'*

loss:�=����       �	Xsfc�A�'*

loss�}x<�U       �	-�Xsfc�A�'*

loss_U�<y��Y       �	75Ysfc�A�'*

loss��<�ɗe       �	��Ysfc�A�'*

loss7�;Lv�U       �	�oZsfc�A�'*

lossb�<�rs5       �	�;[sfc�A�'*

loss�N�="X�       �	�[sfc�A�'*

loss�&B<F�y$       �	(�\sfc�A�'*

lossf<�0�       �	g]sfc�A�'*

loss�.B;0[��       �	.�]sfc�A�'*

loss/L�<��d       �	�^sfc�A�'*

loss�Y�;^ �/       �	�G_sfc�A�'*

lossi�&;|�_       �	H�_sfc�A�'*

lossZ��<�F       �	�q`sfc�A�'*

lossZ+4:��.�       �	N+asfc�A�'*

loss�Ճ8ߑ#C       �	��asfc�A�'*

loss��V=���       �	CTbsfc�A�'*

loss���;e���       �	��bsfc�A�'*

loss���;j}z       �	Ácsfc�A�'*

loss��"<��a�       �	�+dsfc�A�'*

lossdH�<O��       �	a�dsfc�A�(*

loss�<Xt       �	�aesfc�A�(*

lossoK�<��f5       �	��esfc�A�(*

loss8X�<bwm|       �	֌fsfc�A�(*

loss���;����       �	�gsfc�A�(*

loss���<�z��       �	��gsfc�A�(*

loss}[H<��@�       �	�Thsfc�A�(*

loss��;���F       �	M�hsfc�A�(*

lossI<�1��       �	��isfc�A�(*

loss(��<3"�$       �	֌jsfc�A�(*

loss�U=�!��       �	�7ksfc�A�(*

loss1Y�<'���       �	'�ksfc�A�(*

loss��^<!3       �	�lsfc�A�(*

loss��<��b       �	��msfc�A�(*

loss:H�;}#�       �	�mnsfc�A�(*

loss��=�[ �       �	I�osfc�A�(*

lossJ�9@��       �	QKpsfc�A�(*

loss�`+=\i��       �	b�psfc�A�(*

loss�4<���       �	d�qsfc�A�(*

loss���<�|&       �	;�rsfc�A�(*

loss]�;�]��       �	-tsfc�A�(*

loss�F�:2`�       �	m�tsfc�A�(*

loss�<۩��       �	�usfc�A�(*

loss�;6��B       �	�7vsfc�A�(*

lossF˚:
�6�       �	��vsfc�A�(*

loss֓�=�g��       �	��wsfc�A�(*

loss�{<+C��       �	|�xsfc�A�(*

loss�<���       �	��ysfc�A�(*

loss�k<kzh       �	)�zsfc�A�(*

loss��=`�       �		Q{sfc�A�(*

loss7]�=:U�r       �	��|sfc�A�(*

loss�e7:R�       �	c}sfc�A�(*

loss�O�<���       �	j�~sfc�A�(*

loss�1;-U*       �	��sfc�A�(*

lossg�#:�g^       �	
��sfc�A�(*

loss��=��}�       �	힁sfc�A�(*

loss��<��ӡ       �	�<�sfc�A�(*

loss�
�;Ldp       �	�O�sfc�A�(*

loss���;4��       �	�/�sfc�A�(*

loss�o�<���b       �	@ڄsfc�A�(*

lossm�>)D5�       �	�v�sfc�A�(*

lossf��=����       �	@�sfc�A�(*

losso�=����       �	hB�sfc�A�(*

loss�`�<����       �	�%�sfc�A�(*

lossT��<�v2�       �	8/�sfc�A�(*

loss�u=��	�       �	��sfc�A�(*

loss�ȟ<-�0�       �	q��sfc�A�(*

loss�O=Z��       �	81�sfc�A�(*

loss��&="~�       �	
ڋsfc�A�(*

loss�d�:��       �	1�sfc�A�(*

loss�)J;A�a       �	�ƍsfc�A�(*

loss���;����       �	!x�sfc�A�(*

loss��J=�,�
       �	���sfc�A�(*

loss��;<�o��       �	R�sfc�A�(*

lossv-*<�4�       �	Y��sfc�A�(*

lossX�@<{��"       �	eősfc�A�(*

loss���=2��       �	u�sfc�A�(*

loss,/�<�t9�       �	��sfc�A�(*

loss�gR=�T�       �	D��sfc�A�(*

losssv�<%�b       �	�9�sfc�A�(*

loss(��;G�wj       �	CU�sfc�A�(*

loss��<�Rm       �	|�sfc�A�(*

loss\^+<`���       �	���sfc�A�(*

loss2��=6��=       �	&řsfc�A�(*

loss�V=;��^�       �	�{�sfc�A�(*

loss���;	,�       �	Nc�sfc�A�(*

loss�o=jDzD       �	
��sfc�A�(*

loss���<��m       �	<��sfc�A�(*

loss�q=�y�       �	c��sfc�A�(*

loss��;Н�       �	ԟsfc�A�(*

lossZK;�ݍq       �	���sfc�A�(*

loss���<֛s�       �	X��sfc�A�(*

loss�nk<��'       �	��sfc�A�(*

loss3�R=��1V       �	���sfc�A�(*

lossN�<f_#�       �	���sfc�A�(*

loss{Y�;,���       �	�T�sfc�A�(*

loss
�3=v0       �	0��sfc�A�(*

lossaF:q�{       �	$^�sfc�A�(*

lossLTN<�q�       �	���sfc�A�(*

loss%��;��       �	D��sfc�A�(*

loss�7�<'|;       �	�Q�sfc�A�(*

loss�|;D"iF       �	V�sfc�A�(*

loss<��;�O_�       �	+��sfc�A�(*

lossz>�<��       �	�6�sfc�A�(*

lossI`Y:���{       �	�۫sfc�A�(*

loss�$4=���       �	.��sfc�A�(*

loss7DZ<{u{       �	 ��sfc�A�(*

loss]��;Y;       �	s��sfc�A�(*

lossɱ�<\`��       �	���sfc�A�(*

lossq�;���-       �	�%�sfc�A�(*

loss��*=��4N       �	���sfc�A�(*

lossq��;�J�       �	S[�sfc�A�(*

lossֱ=�J��       �	��sfc�A�(*

lossi.�=���{       �	d��sfc�A�(*

lossiRn<
��       �	E.�sfc�A�(*

loss$>=5��	       �	�ɳsfc�A�(*

loss?��<F�@       �	sg�sfc�A�(*

loss�(
:�]_�       �	h�sfc�A�(*

loss�2:=�0       �	L��sfc�A�(*

loss�2�;�5�       �	tC�sfc�A�(*

loss��U<r��@       �	��sfc�A�(*

losslm<z:~�       �	�z�sfc�A�(*

lossJ�=�*�       �	j�sfc�A�(*

lossa�e;3�e�       �	��sfc�A�(*

loss�O�=�*��       �	ML�sfc�A�(*

lossv��;@_8       �	��sfc�A�(*

lossغ<��G�       �	뎺sfc�A�(*

loss�]�:�ͬ       �	0�sfc�A�(*

losst��<AV��       �	?Ȼsfc�A�(*

loss���;u��A       �	�^�sfc�A�(*

lossmC�:��(       �	��sfc�A�(*

loss]��<ꤖ�       �	���sfc�A�(*

loss���<Sf�<       �	��sfc�A�(*

loss� m;��]�       �	y��sfc�A�(*

loss��=�r�       �	�G�sfc�A�(*

loss��s<mz��       �	��sfc�A�(*

loss�S�<�Re       �	���sfc�A�(*

lossC�<�hٳ       �	2�sfc�A�(*

lossi��;��       �	��sfc�A�(*

loss.%9<�OP�       �	it�sfc�A�(*

lossW>;���       �	K�sfc�A�(*

loss�=��|       �	��sfc�A�(*

lossÔ�<��3f       �	�b�sfc�A�(*

loss7��<��U       �	8��sfc�A�(*

loss�O-=n<R       �	`��sfc�A�(*

lossi�(; !       �	R(�sfc�A�(*

loss��)<�<x�       �	
��sfc�A�(*

loss��<�.��       �	<j�sfc�A�(*

lossvJ�;��       �		�sfc�A�)*

loss���<�JH*       �	/��sfc�A�)*

loss�~�;���       �	�[�sfc�A�)*

loss�
 >L��       �	��sfc�A�)*

lossDE;A�Sb       �	��sfc�A�)*

lossz9�2=       �	�~�sfc�A�)*

losss�<Nw2       �	b��sfc�A�)*

loss��{<ْe�       �	�Y�sfc�A�)*

loss-Y%<�43^       �	���sfc�A�)*

loss���;9�       �	ߊ�sfc�A�)*

loss���<�np�       �	/4�sfc�A�)*

loss��;-�       �	���sfc�A�)*

lossa`=v��       �	��sfc�A�)*

loss�NA<��֛       �	^,�sfc�A�)*

loss�f1;��z�       �	7��sfc�A�)*

loss!�:l�7       �	�'�sfc�A�)*

loss��3=�4��       �	���sfc�A�)*

loss\
�<Kڬ       �	yv�sfc�A�)*

loss�h#<����       �	�&�sfc�A�)*

lossC<��k       �	���sfc�A�)*

loss�/G;_�       �	,��sfc�A�)*

loss.�<��       �	ؚ�sfc�A�)*

loss�D_:�-��       �	&T�sfc�A�)*

lossXg�=�õ       �	&��sfc�A�)*

lossDe<��@       �	>�sfc�A�)*

loss:��;�u�       �	���sfc�A�)*

lossۖ;����       �	ʍ�sfc�A�)*

lossQ{�;�m��       �	/6�sfc�A�)*

loss�
�:���N       �	��sfc�A�)*

loss�ב:�}%       �	��sfc�A�)*

losstX
<�+��       �	B�sfc�A�)*

lossj΀;�#�       �	��sfc�A�)*

loss1�E=�<�       �	���sfc�A�)*

lossܠ�=�b!       �	�_�sfc�A�)*

loss��,<K	v�       �	5*�sfc�A�)*

loss��
<�Y�3       �	���sfc�A�)*

lossA :A�|       �	�b�sfc�A�)*

loss�W�=-���       �	d�sfc�A�)*

loss���<#|�,       �	���sfc�A�)*

loss�a.<F��?       �	�Z�sfc�A�)*

lossVP�<���       �	���sfc�A�)*

loss�;]��       �	`��sfc�A�)*

loss�k�=����       �	�=�sfc�A�)*

lossN�;P���       �	��sfc�A�)*

lossC^<��       �	�u�sfc�A�)*

loss�m�:J�       �	>!�sfc�A�)*

loss 1�<�R�       �		��sfc�A�)*

loss!��:4͒z       �	Ll�sfc�A�)*

loss֛<\7�       �	��sfc�A�)*

lossO�<�J	       �	I��sfc�A�)*

loss�;��U�       �	�e�sfc�A�)*

loss閴<˞��       �	��sfc�A�)*

loss���;�} �       �	��sfc�A�)*

loss�eG<'OF�       �	q��sfc�A�)*

loss-��:f       �	���sfc�A�)*

loss��j;���       �	]�sfc�A�)*

loss�g; �u�       �	���sfc�A�)*

loss��<#�x       �	��sfc�A�)*

loss��=Fw       �	�d�sfc�A�)*

loss�X<�G@�       �	?:�sfc�A�)*

loss���;�z�&       �	1�sfc�A�)*

loss��;��       �	TV�sfc�A�)*

lossܵ/<G���       �	�^�sfc�A�)*

loss<|�:�ׅL       �	1`�sfc�A�)*

loss�O=�+6       �	�s�sfc�A�)*

loss��;�Z       �		O�sfc�A�)*

loss�R<.��       �	� �sfc�A�)*

loss�o�;.       �	��sfc�A�)*

lossH��<�Q��       �	A��sfc�A�)*

loss��=�en       �	6�sfc�A�)*

loss��:�LΘ       �	���sfc�A�)*

loss���;� 
       �	k�sfc�A�)*

loss�Ao<�:��       �	�sfc�A�)*

lossMW>;}��:       �	W��sfc�A�)*

lossO�;�ox_       �	>� tfc�A�)*

loss	�;V       �	��tfc�A�)*

loss�*u;2��2       �	T�tfc�A�)*

loss��<R/��       �	��tfc�A�)*

loss�g=K7��       �	��tfc�A�)*

loss�!<�s�       �	�Atfc�A�)*

loss��W=��;       �	?�tfc�A�)*

lossݳ�=g��       �	6�tfc�A�)*

lossLM�;Q��n       �	@htfc�A�)*

loss`�%<hJ~       �	J
tfc�A�)*

loss��=+b�L       �	H�tfc�A�)*

loss�:;��He       �	�<	tfc�A�)*

loss�C39[�P�       �	,�	tfc�A�)*

loss�g<�O       �	�
tfc�A�)*

loss�<i��Q       �	�)tfc�A�)*

loss�p;���,       �	S�tfc�A�)*

loss{m�;��       �	�jtfc�A�)*

loss�/b:k��#       �	Atfc�A�)*

loss�;*f��       �	��tfc�A�)*

loss��Y9�m�       �	�gtfc�A�)*

loss�9�sǭ       �	�tfc�A�)*

lossH��8��x       �	��tfc�A�)*

loss�?Q:mX�r       �	catfc�A�)*

loss�";�@��       �	�tfc�A�)*

lossv��<�$/       �	8�tfc�A�)*

loss�3�;�e.       �	atfc�A�)*

lossH&@:��       �	��tfc�A�)*

loss.��=��4q       �	��tfc�A�)*

lossĤ�:	bJK       �	L6tfc�A�)*

lossl<�=��       �	�tfc�A�)*

loss�Σ<��n�       �	Ҍtfc�A�)*

loss�4=f��q       �	 8tfc�A�)*

loss��:]x��       �	I�tfc�A�)*

loss��9;��q�       �	ptfc�A�)*

loss���<�W�       �	Rtfc�A�)*

loss=��<J���       �	��tfc�A�)*

loss-��<���       �	�Ctfc�A�)*

loss�˙<ֹ�O       �	8�tfc�A�)*

loss{Y=�-cf       �	�rtfc�A�)*

lossx�+<Pۼ�       �	�tfc�A�)*

loss��<��       �	D�tfc�A�)*

loss_hs;�P(z       �	�<tfc�A�)*

loss�s=��       �	#�tfc�A�)*

loss6�]=]��       �	�mtfc�A�)*

loss��\<��M       �	�tfc�A�)*

loss�f<2�17       �	Ĳtfc�A�)*

loss@�@=�"-�       �	,Htfc�A�)*

loss��i<]w�       �	7�tfc�A�)*

loss4��;�'�       �	�w tfc�A�)*

loss���;1��p       �	�!tfc�A�)*

loss��p;Tu�       �	�!tfc�A�)*

loss�
�:���       �	�N"tfc�A�)*

lossn�0;���       �	��"tfc�A�)*

loss��p:�L<!       �	7�#tfc�A�)*

loss�N�<]�       �	�)$tfc�A�**

lossUF<% '       �	��$tfc�A�**

loss1�V:�4�w       �	�Z%tfc�A�**

loss�14=��!<       �	�%tfc�A�**

loss���;޳�       �	Օ&tfc�A�**

loss>�=���4       �	�5'tfc�A�**

lossl�O<Fy�       �	��'tfc�A�**

loss!|8=��q�       �	�_(tfc�A�**

loss���<?���       �	��(tfc�A�**

loss�A_;�I�       �	��)tfc�A�**

loss�F:<�v�o       �	5'*tfc�A�**

loss\�<�rV�       �	�*tfc�A�**

loss�v�<mқp       �	�Z+tfc�A�**

loss�n�=��3       �	�+tfc�A�**

loss��9-	�       �	-�,tfc�A�**

lossqF�:��a       �	�-tfc�A�**

loss��?<�r�       �	0/tfc�A�**

lossL�;B��P       �	m�/tfc�A�**

loss��<��9       �	�0tfc�A�**

lossL�'=s�m�       �	�s2tfc�A�**

loss�==i�JF       �	R3tfc�A�**

lossF�-;�l��       �	(�3tfc�A�**

loss�f|<AۮI       �	}Z4tfc�A�**

lossc0�<����       �	n�4tfc�A�**

losso�[=3�J�       �	��5tfc�A�**

loss�: =�       �	�Mtfc�A�**

lossѧ�<�Ʌ�       �	�;Ntfc�A�**

loss��<�-�       �	(�Ntfc�A�**

loss��< b*4       �	/�Otfc�A�**

lossA�&=z�%       �	8LPtfc�A�**

loss�r":���Z       �	;�Ptfc�A�**

loss.[�<I'�        �	_zQtfc�A�**

loss#�Z<Q��       �	aRtfc�A�**

lossFa<N��       �	iVStfc�A�**

loss�;=���       �	��Stfc�A�**

lossJ��:���g       �	��Ttfc�A�**

lossdʚ=o_       �	��Utfc�A�**

losss"=�=R:       �	�Vtfc�A�**

loss��;:	t       �	�/Wtfc�A�**

loss�7�;���       �	��Wtfc�A�**

loss�~U=���P       �	'kXtfc�A�**

lossWW�9��       �	%Ytfc�A�**

loss���;�0�       �	�Ytfc�A�**

loss�D�<L��       �	i8Ztfc�A�**

loss�y<̏��       �	ͭ[tfc�A�**

lossN�\:��&       �	XX\tfc�A�**

loss,�c=7槽       �	��\tfc�A�**

loss+�:�<[�       �	�y]tfc�A�**

loss��=��]=       �	(
^tfc�A�**

loss��:����       �	:_tfc�A�**

lossn�m:�� u       �	��_tfc�A�**

loss��:;�ǩA       �	�U`tfc�A�**

loss�x�:����       �	�`tfc�A�**

loss�=�1w       �	��atfc�A�**

loss���:�
�       �	Dbtfc�A�**

loss�5;V���       �	u�btfc�A�**

loss�K�<,tA       �	~ctfc�A�**

lossQ�%;[=��       �	�=dtfc�A�**

lossz�=|��       �	��dtfc�A�**

loss�&t:� ��       �	bietfc�A�**

loss�Ke<',��       �	pftfc�A�**

loss#%<N#��       �	Χftfc�A�**

loss��.=1\��       �	Lgtfc�A�**

loss"=��       �	��gtfc�A�**

loss��=w��Q       �	��htfc�A�**

loss��<�A_       �	^*itfc�A�**

loss���;�{]       �	��itfc�A�**

loss2 
<�D<�       �	
ijtfc�A�**

loss~(�<l\�       �	$ktfc�A�**

loss,�J=B�O       �	�ktfc�A�**

loss�ߩ<�&(�       �	�Fltfc�A�**

loss��:�S�       �	��ltfc�A�**

losszt;8�r       �	M�mtfc�A�**

loss���;ϕ�       �	��ntfc�A�**

lossN�/: -?       �	�Motfc�A�**

loss��:��R4       �	��otfc�A�**

lossT�R=�xs       �	r�ptfc�A�**

loss��;Sa��       �	(Fqtfc�A�**

loss_�>�ԅ�       �	j�qtfc�A�**

lossa?;��0T       �	Ӣrtfc�A�**

losssi�;随�       �	Cstfc�A�**

loss���9��ֳ       �	��stfc�A�**

lossf!:+��       �	W�utfc�A�**

loss8�^;�b�       �	8jvtfc�A�**

lossX�!>H�N       �	�wtfc�A�**

losssm==M       �	��wtfc�A�**

lossF�<��H       �	�@xtfc�A�**

loss�j<ΘW"       �	��xtfc�A�**

loss�'=�{
?       �	�xytfc�A�**

loss�f<��       �	4ztfc�A�**

lossTե;\�
j       �	%�ztfc�A�**

loss�mK=s@��       �	�A{tfc�A�**

loss7��; I�+       �	3�{tfc�A�**

loss��<e��       �	�{|tfc�A�**

lossf��;u6
       �	�}tfc�A�**

loss4�<��k       �	V�}tfc�A�**

loss�	=���       �	�M~tfc�A�**

loss�Z&<���       �	��~tfc�A�**

loss�;�'t-       �	�wtfc�A�**

loss4�<���:       �	p�tfc�A�**

loss4{z<0�WN       �	n��tfc�A�**

loss^R�<Fyk       �	�6�tfc�A�**

loss龧<���       �	�ԁtfc�A�**

loss�B{;}�2       �	!x�tfc�A�**

loss�Y�<���       �	�tfc�A�**

loss�7�<�kcT       �	ϣ�tfc�A�**

loss�R�=�B��       �	��tfc�A�**

lossi��<�~       �	!�tfc�A�**

lossƢ�;��L�       �	[��tfc�A�**

loss�-�:��S�       �	�X�tfc�A�**

loss�":Ve^4       �	&��tfc�A�**

loss6{�<�0��       �	��tfc�A�**

loss�q=3�6       �	�^�tfc�A�**

loss�u�=mѡG       �	X �tfc�A�**

loss_�&;�!��       �	9	�tfc�A�**

loss�O<I�.Y       �	��tfc�A�**

loss��A;ٚ�       �	�}�tfc�A�**

losslQ=�&�       �	��tfc�A�**

lossf}�;�6       �	���tfc�A�**

loss�<�o�P       �	���tfc�A�**

loss �I;��<       �	W=�tfc�A�**

loss~0�=��1�       �	^ۏtfc�A�**

loss���;�#��       �	dt�tfc�A�**

loss���<�ńd       �	��tfc�A�**

loss���;����       �	䣑tfc�A�**

loss8˗;B%�(       �	?�tfc�A�**

loss �-=ߘ۔       �	�֒tfc�A�**

loss�ڍ<�S��       �	�v�tfc�A�**

loss�*z;H_�E       �	�
�tfc�A�+*

loss��z;_H3w       �	>B�tfc�A�+*

loss��3=Q~
�       �	�y�tfc�A�+*

lossֺ�<nˡ6       �	�>�tfc�A�+*

loss]tm;hf�_       �	;S�tfc�A�+*

loss�m�<���f       �	�tfc�A�+*

loss���;|�a       �	��tfc�A�+*

lossv#�<���       �	 �tfc�A�+*

loss�2<�B��       �	ѱ�tfc�A�+*

loss���;;��       �	�ߜtfc�A�+*

loss��<GD[       �	֭�tfc�A�+*

loss� "=�M6�       �	X��tfc�A�+*

losso��;�E3f       �	i�tfc�A�+*

loss�<���       �	�tfc�A�+*

loss�P><�[       �	K�tfc�A�+*

loss[v<c�lX       �	��tfc�A�+*

loss�4�<>P��       �	Xˣtfc�A�+*

loss���<��.Q       �	`�tfc�A�+*

loss.�9=/��       �	��tfc�A�+*

loss{"�<vݚ       �	�7�tfc�A�+*

loss���<0Ah       �	-�tfc�A�+*

loss��<;���       �	���tfc�A�+*

loss�=�1�2       �	�B�tfc�A�+*

lossw.;��$       �	�Өtfc�A�+*

loss�g�;��OZ       �	�q�tfc�A�+*

loss0(<�J��       �	�
�tfc�A�+*

loss��a=gl�9       �	���tfc�A�+*

lossM�~<�R��       �	vO�tfc�A�+*

loss.;�;zr?       �	��tfc�A�+*

loss\��;�;0       �	���tfc�A�+*

loss��f<3�xL       �	�)�tfc�A�+*

loss�#�<]���       �	��tfc�A�+*

lossdi;�}��       �	���tfc�A�+*

loss��<E��M       �	�k�tfc�A�+*

loss�<XKʄ       �	�K�tfc�A�+*

loss;x<z�1�       �	��tfc�A�+*

lossĪ�;����       �	���tfc�A�+*

loss��N;���S       �	��tfc�A�+*

lossn�^=c���       �	���tfc�A�+*

loss��<�u='       �	f��tfc�A�+*

lossO�)=����       �	�1�tfc�A�+*

loss�0^=���l       �	@õtfc�A�+*

lossW]<�U�       �	�Z�tfc�A�+*

loss�§<�<zQ       �	N�tfc�A�+*

loss�j�;I9�       �	G��tfc�A�+*

lossփ�:��
       �	�2�tfc�A�+*

loss";=���       �	Ǹtfc�A�+*

loss�j:a��       �	e�tfc�A�+*

loss�'`<{�ײ       �	���tfc�A�+*

lossj��<��6       �	���tfc�A�+*

loss�y�;��+�       �	鵻tfc�A�+*

loss��i<&C#�       �	HP�tfc�A�+*

lossܜ�;L��       �	�O�tfc�A�+*

lossH;�d       �	W�tfc�A�+*

loss�g:��/       �	�|�tfc�A�+*

loss��k=�L��       �	E�tfc�A�+*

loss4{D<�+!x       �	?��tfc�A�+*

loss/�=2ؐ       �	=�tfc�A�+*

loss�e{;*
��       �	/m�tfc�A�+*

loss���<�ZA�       �	��tfc�A�+*

loss��9����       �	ݙ�tfc�A�+*

loss���=��Kt       �	�7�tfc�A�+*

loss
��:kC       �	���tfc�A�+*

loss��;����       �	�n�tfc�A�+*

losso�:�'X       �	��tfc�A�+*

lossw�}:~���       �	���tfc�A�+*

loss=M:2Z#       �	HQ�tfc�A�+*

loss[��;XrM       �	���tfc�A�+*

loss��?:�*H       �	���tfc�A�+*

lossv`�:+�S       �	s+�tfc�A�+*

loss���;�Yݵ       �	���tfc�A�+*

loss��^<=�]       �	[�tfc�A�+*

loss��4=)��       �	s��tfc�A�+*

loss�(<��<e       �	��tfc�A�+*

loss�g=�xmO       �	6!�tfc�A�+*

lossx;=Ã��       �	k��tfc�A�+*

loss�&�:�O�       �	HR�tfc�A�+*

loss5��<RQnx       �	���tfc�A�+*

loss<�<
s       �	��tfc�A�+*

lossF�P<���        �	)�tfc�A�+*

loss��5;`e1�       �	���tfc�A�+*

loss�L;��}�       �	GT�tfc�A�+*

loss�|�;�"�       �	w��tfc�A�+*

lossV
=}!Q       �	���tfc�A�+*

lossq.=���h       �	#�tfc�A�+*

lossQ�/=pZ~       �	���tfc�A�+*

loss��;�^ʠ       �	@Q�tfc�A�+*

loss|օ=�:��       �	6��tfc�A�+*

loss��<ہ�o       �	��tfc�A�+*

lossA��:�ue       �	A-�tfc�A�+*

lossMʨ<1���       �	9��tfc�A�+*

loss_;��9       �	�m�tfc�A�+*

loss�� <���       �	��tfc�A�+*

lossL;���       �	��tfc�A�+*

lossE(�;���       �	UO�tfc�A�+*

loss�bk<�u�u       �	���tfc�A�+*

loss�e�:�ٹU       �	D��tfc�A�+*

loss1��<[\A<       �	�#�tfc�A�+*

loss\��=1�	       �	/��tfc�A�+*

loss8ޞ<� �[       �	!Y�tfc�A�+*

loss,�*<iɶ(       �	&��tfc�A�+*

loss),.<�.2       �	:��tfc�A�+*

loss���:6�       �	�0�tfc�A�+*

loss[È=1��       �	h��tfc�A�+*

loss�~�=�g��       �	�_�tfc�A�+*

loss_=�;�z�       �	���tfc�A�+*

loss��/<Ӊ       �	���tfc�A�+*

loss�t<�`��       �	1(�tfc�A�+*

loss�#�;�bG2       �	���tfc�A�+*

loss�?;���       �	�Y�tfc�A�+*

loss:�O;�t�       �	���tfc�A�+*

loss,�L=Q��       �	lz�tfc�A�+*

lossITd<��       �	M�tfc�A�+*

loss25�=~��e       �	���tfc�A�+*

loss��<f�u�       �	�@�tfc�A�+*

loss`�=M�1�       �	c��tfc�A�+*

loss�jK=�H��       �	Hm�tfc�A�+*

loss��<&��       �	��tfc�A�+*

lossi�6<�y�       �	���tfc�A�+*

loss��U; �+H       �	y;�tfc�A�+*

loss� �=�;'�       �	���tfc�A�+*

lossw��:�W�       �	k�tfc�A�+*

loss�v�<��P�       �	S�tfc�A�+*

loss�D�<�,Z�       �	��tfc�A�+*

loss�VE<h�       �	/2�tfc�A�+*

loss��q;}���       �	���tfc�A�+*

loss���<@�1       �	=d�tfc�A�+*

lossz ;��       �	���tfc�A�+*

loss�?�;��       �	"��tfc�A�,*

lossCs2=t��       �	�-�tfc�A�,*

loss�9<��%�       �	v��tfc�A�,*

loss�g�<�]       �	1a�tfc�A�,*

loss�i;�v�       �	�6�tfc�A�,*

loss��$<ҝ�e       �	q�tfc�A�,*

loss_ �=L�[       �	��tfc�A�,*

loss�g<����       �	���tfc�A�,*

loss!)�;P��       �	-Z�tfc�A�,*

lossʞ;p|�       �	� �tfc�A�,*

lossU�<_%#�       �	���tfc�A�,*

lossڳ<���       �	*7�tfc�A�,*

loss�@�=.��       �	j��tfc�A�,*

lossq#<�       �	_z�tfc�A�,*

loss�;�Y�       �	��tfc�A�,*

lossm�;Z��       �	���tfc�A�,*

loss�T<�v��       �	V�tfc�A�,*

loss%��;B��N       �	���tfc�A�,*

loss.J�;F�O�       �	��tfc�A�,*

loss�q�;��       �	E*�tfc�A�,*

loss��;��       �	7��tfc�A�,*

loss�?�<^�b       �	]�tfc�A�,*

loss.5/;����       �	���tfc�A�,*

loss$C�<Ņ�       �	�tfc�A�,*

loss�_:$��       �	Y��tfc�A�,*

loss4
<�1�       �	kD�tfc�A�,*

lossh�<	��H       �	���tfc�A�,*

loss���<���       �	\v�tfc�A�,*

lossR_�<�?�       �	��tfc�A�,*

loss�<����       �	ͫ�tfc�A�,*

loss�7<v��       �	pA�tfc�A�,*

loss��<oH�K       �	���tfc�A�,*

loss1<�bS       �	�m ufc�A�,*

loss���<� ]�       �	m� ufc�A�,*

losslq=����       �	!�ufc�A�,*

loss��<�Z�~       �	�%ufc�A�,*

loss$�<�6�        �	��ufc�A�,*

loss8��<V<|�       �	t�ufc�A�,*

loss
�<f3��       �	�;ufc�A�,*

loss�N<Η˺       �	�oufc�A�,*

loss5�;b�.       �	�ufc�A�,*

loss$�g<[�       �	v�ufc�A�,*

loss�Z�;c׬       �	7Oufc�A�,*

loss�E= L��       �	N�ufc�A�,*

lossQ�r=�i       �	�ufc�A�,*

loss�t$<�;�       �	�I	ufc�A�,*

loss�W3;��       �	g�	ufc�A�,*

loss���<t�ww       �	r�
ufc�A�,*

loss�u=��[       �	�;ufc�A�,*

loss�h":���       �	�zufc�A�,*

lossj|;���       �	�Jufc�A�,*

loss�G
;շK5       �	D�ufc�A�,*

loss�<`BO0       �	N�ufc�A�,*

loss�7;ko�6       �	<ufc�A�,*

lossw+z=�2�       �	P�ufc�A�,*

loss�a<U�\       �	�ufc�A�,*

loss��<��        �	�ufc�A�,*

loss��:�u�B       �	��ufc�A�,*

loss��:zۓ�       �	LOufc�A�,*

loss���;�&�       �	6�ufc�A�,*

lossT�=&��h       �	|ufc�A�,*

lossZ��;��!�       �	�ufc�A�,*

lossO��9��       �	V�ufc�A�,*

loss�|<��x       �	�pufc�A�,*

loss��=D�`       �	{ufc�A�,*

loss�$;�`��       �	��ufc�A�,*

loss�6c=�D�i       �	�=ufc�A�,*

lossa��<��       �	��ufc�A�,*

loss<��p       �	amufc�A�,*

loss��;�5)�       �	��ufc�A�,*

loss/�;0�       �	�ufc�A�,*

loss�`j;�-=�       �	$)ufc�A�,*

loss��<ރH�       �	{�ufc�A�,*

loss�"�<!|�Z       �	��ufc�A�,*

loss���=�Y�       �	quufc�A�,*

loss�	j:��       �	]ufc�A�,*

loss]�b=�9�K       �	i�ufc�A�,*

loss��;tӯ       �	�ufc�A�,*

loss
��;h �?       �	*ufc�A�,*

loss�|�:����       �	��ufc�A�,*

loss��=��.�       �	�� ufc�A�,*

loss =w��       �	�K!ufc�A�,*

lossM1<S7+f       �	��!ufc�A�,*

loss��:�3�j       �	s�"ufc�A�,*

lossڐ�<����       �	#ufc�A�,*

losslp=��       �	A�#ufc�A�,*

loss�g;%rJN       �	�L$ufc�A�,*

lossy�=�p�       �	��$ufc�A�,*

lossv�=~_�       �	N|%ufc�A�,*

lossۧH;o�       �	�&ufc�A�,*

lossd��=�p�       �	��&ufc�A�,*

loss�8<��j<       �	��'ufc�A�,*

loss�!�:��Y�       �	�+(ufc�A�,*

loss<�<(�]Z       �	*�(ufc�A�,*

lossz�j<'�<       �	��)ufc�A�,*

loss�;�~�       �	@�*ufc�A�,*

loss�<ʸ�       �	z9+ufc�A�,*

loss�0�<NY�       �	��+ufc�A�,*

lossvޠ=���       �	�r,ufc�A�,*

losse&<X��7       �	-ufc�A�,*

loss�`�;���R       �	q�-ufc�A�,*

loss�:x:k�(       �	L.ufc�A�,*

loss�Ex<���       �	y�/ufc�A�,*

lossad�;��Q       �	��0ufc�A�,*

lossɤ�<��~       �	~n1ufc�A�,*

loss1��<{MYO       �	+�2ufc�A�,*

lossE3�<��b�       �	�G3ufc�A�,*

lossd��<,հ\       �	φ4ufc�A�,*

loss]��;�6�       �	=(5ufc�A�,*

loss��;p;si       �	��5ufc�A�,*

loss��9鶀