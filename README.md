# Starting
If you want to mess around with the code in this repo the dependencies are a bit messy so I have provided a docker file and a few ways to work with it.
### Option 1 Use VScode
Once you have vs code installed you will need to add the Remote - Container extension and docker. With that you should simple be able to open up this work space by click the Open remote window button at the bottom left of the screen and selecting open in container. Once inside the container you can run `./start_inside.sh` to launch jupiter server. Then simple click the link provided by jupiter to view the notebooks.
I recommend using vs-code because I have included all the relevant extensions and setting in order to edit the rust code used in many of the exercises.

### Option 2 Use Pure docker
If you don't plan to edit the underlying rust and just want to run the note-books this should work fine. Make sure you have docker installed and are able to access it from bash i.e.
```bash
# you can do this
$ docker ps
```
Then use the start.sh script as follows
```bash
$ ./start.sh --build
# this might take a bit
$ ./start.sh --run
```
Follow the link provided by jupiter and your off to the races!