package app.termora

import okhttp3.Response

class ResponseException : RuntimeException {
    val code: Int
    val response: Response

    constructor(code: Int, response: Response) : this(code, "Response code: $code", response)
    constructor(code: Int, message: String, response: Response) : super(message) {
        this.code = code
        this.response = response
    }

}
