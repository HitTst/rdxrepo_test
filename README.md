USAGE:

1. paste your detector model files along with labels.txt in 'models' folder inside cloned directory.
2. update the pgie config with your config parameters.
3. update the tracker config with your config parameters.
4. for production environment:
   - change ip address and port of the socket server inside deepstream_template.py
   - change the value of environment in debug_config.json to production
   for test environment:
   - change the value of environment in debug_config.json to test
5. add the cameras in debug_config.json for test environment with dummy id and its rtsp link as a new key valu pair.
6. to display the inference, set display value to true otherwise false
7. run the file.

Note: Do not change the name of any config text file. To update the parameters make the changes in existing text files.

