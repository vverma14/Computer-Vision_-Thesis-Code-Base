# classic
import getopt, os, json
# perso
from utils.training_module import trainModel

def start_with_json(argv, P):
    # A) Get information about a possible jsonfolder
    #-----------------------------------------------
    # initialization
    jsonfolder = ''
    P_original = None
    try:
        opts, args = getopt.getopt(argv,"hf",["jsonfolder=","reset"])
    except getopt.GetoptError:
        print ('Usage: train_launcher.py --jsonfolder=<jsonfolder> --reset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Usage: train_launcher.py --jsonfolder=<jsonfolder> --reset')
            sys.exit()
        elif opt in ("-f", "--jsonfolder"):
            jsonfolder = arg
        elif opt in ("--reset"):
            P_original = P.copy()
    # B) Start
    #---------
    if (jsonfolder == ''):
        # no jsonfolder: use P
        trainModel(myModel, P)
    else:
        # jsonfolder: go through each file
        if os.path.exists(jsonfolder):
            dirs = sorted( os.listdir(jsonfolder) )
            # only take json file and not temporary file
            matching = [f for f in dirs if (".json" in f) and (not f.startswith('.'))]
            if (matching == None):
                print ('No Json file in '+jsonfolder)
                sys.exit(2)
            else:
                for one_file in matching:
                    with open(os.path.join(jsonfolder, one_file),'r') as jsonFile:
                        jsonFile = jsonFile.read()
                        P_new = json.loads(jsonFile)
                        P.update(P_new)
                        print('\nNew Parameters from '+one_file + ' with the model: '+ P['model']+'\n')
                        trainModel(myModel)
                        if (P_original != None):
                            P = P_original.copy()
        else:
            print('Error: '+jsonfolder+' do not exist')
