var express =   require("express");
var multer  =   require('multer');
var app     =   express();
var spawn 	= require("child_process").spawn;
var process = spawn('python',["./testMain.py"]);
var storage =   multer.diskStorage({
	destination: function (req, file, callback) 
		{
    		callback(null, './uploads'); 
    	},
  		filename: function (req, file, callback) 
  		{
    		callback(null, file.fieldname + '-' + Date.now()); 
		}
	}
);

var upload = multer({ storage : storage}).single('mysteryPokemon');

app.use(express.static(__dirname) )

app.get('/',function(req,res){
      res.sendFile(__dirname + "/index.html");
});

app.post('/api/photo',function(req,res){
    upload(req,res,function(err) {
        if(err) {
            return res.end("Error uploading file.");
        }
        //res.end("File is uploaded");
    });
});

app.listen(3000,function(){
    console.log("Working on port 3000");
});