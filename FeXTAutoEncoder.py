import os
import sys
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
# =============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.components.data_assets import UserOperations

# [MAIN MENU]
#==============================================================================
# Starting DITK analyzer, checking for dictionary presence and perform conditional
# import of modules
#==============================================================================
print('''
-------------------------------------------------------------------------------
Features Extraction FeatEXT
-------------------------------------------------------------------------------
... 
''')
user_operations = UserOperations()
operations_menu = {'1' : 'Pretrain FeXTAutoEncoder model',
                   '2' : 'Evaluate FEXT-AutoEncoder model',
                   '3' : 'Extract features from images',                   
                   '4' : 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()

    if op_sel == 1:
        import modules.FEXT_training
        del sys.modules['modules.FEXT_training']
    
    elif op_sel == 2:
        import modules.FEXT_evaluation
        del sys.modules['modules.FEXT_evaluation']

    elif op_sel == 3:
        import modules.FEXT_extraction
        del sys.modules['modules.FEXT_features_extraction']


    elif op_sel == 4:
        break

