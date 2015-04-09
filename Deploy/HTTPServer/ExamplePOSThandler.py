import zmq

class a_POST_handler:
  def __init__(self,a_number):
    self.number=a_number

  def post(self,request_handler):
	# check request headers
        if not request_handler.headers.has_key('content-length'):
          request_handler.send_error(550,"No content-length given")
        try:
          content_length = int(request_handler.headers['content-length'])
        except ValueError:
          content_length = 0
        if content_length<=0:
          request_handler.send_error(551,"invalid content-length given")

        # if text is given, response appropriately
        if request_handler.headers.has_key('content-type') and request_handler.headers['content-type']=='text':
          text = request_handler.rfile.read(content_length)

          # retrieve the result from zmq and return
          resp = self.retrieveMatches(text)

          self.sendResponse(resp,request_handler)
        else:
          request_handler.send_error(552, "No or unrecognized content-type")  

  def sendResponse(self,body,request_handler):
        request_handler.send_response( 200 )
        request_handler.send_header( "content-type", "text" )
        request_handler.send_header( "content-length", str(len(body)) )
        request_handler.end_headers()
        request_handler.wfile.write( body )
  
  def retrieveMatches(self, text):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        # strings must be null-delimited (C style)
        socket.send(text + "\0")
        message = socket.recv()
        return message

