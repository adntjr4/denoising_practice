
loss_types = {}

def regist_loss(loss_class):
    loss_name = loss_class.__name__.lower()
    assert not loss_name in loss_types, 'there is already registered loss name: %s in loss_types.' % loss_name
    loss_types[loss_name] = loss_class

    return loss_class

'''
default format of loss

    @regist_loss
    class ():
        def __call__(self, input_data, model_output, data, model):

example of loss: L1 loss

    @regist_loss
    class L1():
        def __call__(self, input_data, model_output, data, model):
            if type(model_output) is tuple: output = model_output[0]
            else: output = model_output
            return F.l1_loss(output, data['clean'])
'''

