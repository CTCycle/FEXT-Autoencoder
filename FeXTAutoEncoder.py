import sys
import art
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('FeXT AE')
print(ascii_art)

# [MAIN MENU]
#==============================================================================
# Starting DITK analyzer, checking for dictionary presence and perform conditional
# import of modules
#==============================================================================
user_operations = UserOperations()
operations_menu = {'1' : 'Pretrain FeXTAutoEncoder model',
                   '2' : 'Evaluate FeXTAutoEncoder model',
                   '3' : 'Extract features from images',                   
                   '4' : 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()
    if op_sel == 1:
        import modules.model_training
        del sys.modules['modules.model_training']        
    elif op_sel == 2:
        import modules.model_evaluation
        del sys.modules['modules.model_evaluation']
    elif op_sel == 3:
        import modules.features_extraction
        del sys.modules['modules.features_extraction']
    elif op_sel == 4:
        break
