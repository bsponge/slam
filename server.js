var express = require('express');
var cors = require('cors')
var app = express();

app.use(cors())
app.use(express.static(__dirname)); //Serves resources from public folder


var server = app.listen(5000);
