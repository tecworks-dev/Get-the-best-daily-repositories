package ru.blackfan.bfscan.parsing.httprequests;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class Constants {

    private Constants() {
    }

    public static final class MicroProfileOpenApi {

        private MicroProfileOpenApi() {
        }
        
        public static final String PARAMETERS = "Lorg/eclipse/microprofile/openapi/annotations/parameters/Parameters;";
        public static final String PARAMETER = "Lorg/eclipse/microprofile/openapi/annotations/parameters/Parameter;";
        public static final String REQUEST_BODY = "Lorg/eclipse/microprofile/openapi/annotations/parameters/RequestBody;";
        public static final String OPERATION = "Lorg/eclipse/microprofile/openapi/annotations/Operation;";
        public static final String OPENAPID_DEFINITION = "Lorg/eclipse/microprofile/openapi/annotations/OpenAPIDefinition;";
        public static final String SERVERS = "Lorg/eclipse/microprofile/openapi/annotations/servers/Servers;";
        public static final String SERVER = "Lorg/eclipse/microprofile/openapi/annotations/servers/Server;";
        public static final String SCHEMA = "Lorg/eclipse/microprofile/openapi/annotations/media/Schema;";
        public static final String CONTENT = "Lorg/eclipse/microprofile/openapi/annotations/media/Content;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                OPERATION
        ));
    }

    public static final class OpenApi {

        private OpenApi() {
        }
        
        public static final String PARAMETERS = "Lio/swagger/v3/oas/annotations/Parameters;";
        public static final String PARAMETER = "Lio/swagger/v3/oas/annotations/Parameter;";
        public static final String REQUEST_BODY = "Lio/swagger/v3/oas/annotations/parameters/RequestBody;";
        public static final String OPERATION = "Lio/swagger/v3/oas/annotations/Operation;";
        public static final String OPENAPID_DEFINITION = "Lio/swagger/v3/oas/annotations/OpenAPIDefinition;";
        public static final String SERVERS = "Lio/swagger/v3/oas/annotations/servers/Servers;";
        public static final String SERVER = "Lio/swagger/v3/oas/annotations/servers/Server;";
        public static final String SCHEMA = "Lio/swagger/v3/oas/annotations/media/Schema;";
        public static final String CONTENT = "Lio/swagger/v3/oas/annotations/media/Content;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                OPERATION
        ));
    }

    public static final class Swagger {

        private Swagger() {
        }

        public static final String API = "Lio/swagger/annotations/Api;";
        public static final String API_PARAM = "Lio/swagger/annotations/ApiParam;";
        public static final String API_IMPLICIT_PARAM = "Lio/swagger/annotations/ApiImplicitParam;";
        public static final String API_IMPLICIT_PARAMS = "Lio/swagger/annotations/ApiImplicitParams;";
        public static final String API_OPERATION = "Lio/swagger/annotations/ApiOperation;";
        public static final String API_MODEL_PROPERTY = "Lio/swagger/annotations/ApiModelProperty;";
        public static final String API_MODEL = "Lio/swagger/annotations/ApiModel;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                API_OPERATION
        ));
    }

    public static final class Struts {

        private Struts() {
        }

        public static final String ACTION = "Lorg/apache/struts2/convention/annotation/Action;";
        public static final String ACTIONS = "Lorg/apache/struts2/convention/annotation/Actions;";
        public static final String NAMESPACE = "Lorg/apache/struts2/convention/annotation/Namespace;";
        public static final String NAMESPACES = "Lorg/apache/struts2/convention/annotation/Namespaces;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                ACTION, ACTIONS
        ));

        public static final String VALID_FORM = "org.apache.struts.validator.ValidatorForm";

        public static final List<String> DYNA_FORM = Collections.unmodifiableList(Arrays.asList(
                "org.apache.struts.action.DynaActionForm",
                "org.apache.struts.validator.DynaValidatorForm"
        ));
    }

    public static final class Micronaut {

        private Micronaut() {
        }

        public static final String CLIENT = "Lio/micronaut/http/client/annotation/Client;";
        public static final String CONTROLLER = "Lio/micronaut/http/annotation/Controller;";
        public static final String TRACE = "Lio/micronaut/http/annotation/Trace;";
        public static final String PATCH = "Lio/micronaut/http/annotation/Patch;";
        public static final String OPTIONS = "Lio/micronaut/http/annotation/Options;";
        public static final String HEAD = "Lio/micronaut/http/annotation/Head;";
        public static final String GET = "Lio/micronaut/http/annotation/Get;";
        public static final String POST = "Lio/micronaut/http/annotation/Post;";
        public static final String PUT = "Lio/micronaut/http/annotation/Put;";
        public static final String DELETE = "Lio/micronaut/http/annotation/Delete;";
        public static final String CUSTOM_HTTP_METHOD = "Lio/micronaut/http/annotation/CustomHttpMethod;";
        public static final String CONSUMES = "Lio/micronaut/http/annotation/Consumes;";
        public static final String QUERY_VALUE = "Lio/micronaut/http/annotation/QueryValue;";
        public static final String PATH_VARIABLE = "Lio/micronaut/http/annotation/PathVariable;";
        public static final String BODY = "Lio/micronaut/http/annotation/Body;";
        public static final String HEADER = "Lio/micronaut/http/annotation/Header;";
        public static final String HEADERS = "Lio/micronaut/http/annotation/Headers;";
        public static final String COOKIE_VALUE = "Lio/micronaut/http/annotation/CookieValue;";
        public static final String PART = "Lio/micronaut/http/annotation/Part;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                CLIENT, CONTROLLER, TRACE, PATCH, OPTIONS, HEAD,
                GET, POST, PUT, DELETE, CUSTOM_HTTP_METHOD
        ));
    }

    public static final class Feign {

        private Feign() {
        }

        public static final String CLIENT = "Lorg/springframework/cloud/netflix/feign/FeignClient;";
        public static final String REQUESTLINE = "Lfeign/RequestLine;";
        public static final String HEADERS = "Lfeign/Headers;";
        public static final String QUERYMAP = "Lfeign/QueryMap;";
        public static final String HEADERMAP = "Lfeign/HeaderMap;";
        public static final String PARAM = "Lfeign/Param;";
        public static final String BODY = "Lfeign/Body;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                REQUESTLINE
        ));
    }

    public static final class Retrofit {

        private Retrofit() {
        }

        public static final String HTTP = "Lretrofit2/http/HTTP;";
        public static final String HEAD = "Lretrofit2/http/HEAD;";
        public static final String GET = "Lretrofit2/http/GET;";
        public static final String POST = "Lretrofit2/http/POST;";
        public static final String PUT = "Lretrofit2/http/PUT;";
        public static final String DELETE = "Lretrofit2/http/DELETE;";
        public static final String PATCH = "Lretrofit2/http/PATCH;";
        public static final String OPTIONS = "Lretrofit2/http/OPTIONS;";
        public static final String HEADERS = "Lretrofit2/http/Headers;";
        public static final String MULTIPART = "Lretrofit2/http/Multipart;";
        public static final String FORM = "Lretrofit2/http/FormUrlEncoded;";
        public static final String PARAM_HEADER = "Lretrofit2/http/Header;";
        public static final String PARAM_PART = "Lretrofit2/http/Part;";
        public static final String PARAM_FIELD = "Lretrofit2/http/Field;";
        public static final String PARAM_BODY = "Lretrofit2/http/Body;";
        public static final String PARAM_QUERY = "Lretrofit2/http/Query;";
        public static final String PARAM_PATH = "Lretrofit2/http/Path;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                HTTP, HEAD, GET, POST, PUT, DELETE, PATCH, OPTIONS
        ));
    }

    public static final class Ktorfit {

        private Ktorfit() {
        }

        public static final String HTTP = "Lde/jensklingenberg/ktorfit/http/HTTP;";
        public static final String HEAD = "Lde/jensklingenberg/ktorfit/http/HEAD;";
        public static final String GET = "Lde/jensklingenberg/ktorfit/http/GET;";
        public static final String POST = "Lde/jensklingenberg/ktorfit/http/POST;";
        public static final String PUT = "Lde/jensklingenberg/ktorfit/http/PUT;";
        public static final String DELETE = "Lde/jensklingenberg/ktorfit/http/DELETE;";
        public static final String PATCH = "Lde/jensklingenberg/ktorfit/http/PATCH;";
        public static final String OPTIONS = "Lde/jensklingenberg/ktorfit/http/OPTIONS;";
        public static final String HEADERS = "Lde/jensklingenberg/ktorfit/http/Headers;";
        public static final String MULTIPART = "Lde/jensklingenberg/ktorfit/http/Multipart;";
        public static final String FORM = "Lde/jensklingenberg/ktorfit/http/FormUrlEncoded;";
        public static final String PARAM_HEADER = "Lde/jensklingenberg/ktorfit/http/Header;";
        public static final String PARAM_PART = "Lde/jensklingenberg/ktorfit/http/Part;";
        public static final String PARAM_FIELD = "Lde/jensklingenberg/ktorfit/http/Field;";
        public static final String PARAM_BODY = "Lde/jensklingenberg/ktorfit/http/Body;";
        public static final String PARAM_QUERY = "Lde/jensklingenberg/ktorfit/http/Query;";
        public static final String PARAM_PATH = "Lde/jensklingenberg/ktorfit/http/Path;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                HTTP, HEAD, GET, POST, PUT, DELETE, PATCH, OPTIONS
        ));
    }

    public static final class JaxRs {

        private JaxRs() {
        }

        public static final String WEBSERVLET = "Ljavax/servlet/annotation/WebServlet;";
        public static final String WEBFILTER = "Ljavax/servlet/annotation/WebFilter;";
        public static final String HEAD = "Ljavax/ws/rs/HEAD;";
        public static final String GET = "Ljavax/ws/rs/GET;";
        public static final String POST = "Ljavax/ws/rs/POST;";
        public static final String PUT = "Ljavax/ws/rs/PUT;";
        public static final String DELETE = "Ljavax/ws/rs/DELETE;";
        public static final String OPTIONS = "Ljavax/ws/rs/OPTIONS;";
        public static final String PATH = "Ljavax/ws/rs/Path;";
        public static final String METHOD = "Ljavax/ws/rs/HttpMethod;";
        public static final String CONSUMES = "Ljavax/ws/rs/Consumes;";
        public static final String PRODUCES = "Ljavax/ws/rs/Produces;";
        public static final String PARAM_PATH = "Ljavax/ws/rs/PathParam;";
        public static final String PARAM_QUERY = "Ljavax/ws/rs/QueryParam;";
        public static final String PARAM_HEADER = "Ljavax/ws/rs/HeaderParam;";
        public static final String PARAM_FORM = "Ljavax/ws/rs/FormParam;";
        public static final String PARAM_COOKIE = "Ljavax/ws/rs/CookieParam;";
        public static final String PARAM_BEAN = "Ljavax/ws/rs/BeanParam;";
        public static final String PARAM_MATRIX = "Ljavax/ws/rs/MatrixParam;";
        public static final String DEFAULT_VALUE = "Ljavax/ws/rs/DefaultValue;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                HEAD, GET, POST, PUT, DELETE, OPTIONS, METHOD, WEBSERVLET, WEBFILTER
        ));
    }

    public static final class Jakarta {

        private Jakarta() {
        }

        public static final String WEBSERVLET = "Ljakarta/servlet/annotation/WebServlet;";
        public static final String WEBFILTER = "Ljakarta/servlet/annotation/WebFilter;";
        public static final String HEAD = "Ljakarta/ws/rs/HEAD;";
        public static final String GET = "Ljakarta/ws/rs/GET;";
        public static final String POST = "Ljakarta/ws/rs/POST;";
        public static final String PUT = "Ljakarta/ws/rs/PUT;";
        public static final String DELETE = "Ljakarta/ws/rs/DELETE;";
        public static final String OPTIONS = "Ljakarta/ws/rs/OPTIONS;";
        public static final String PATH = "Ljakarta/ws/rs/Path;";
        public static final String METHOD = "Ljakarta/ws/rs/HttpMethod;";
        public static final String CONSUMES = "Ljakarta/ws/rs/Consumes;";
        public static final String PRODUCES = "Ljakarta/ws/rs/Produces;";
        public static final String PARAM_PATH = "Ljakarta/ws/rs/PathParam;";
        public static final String PARAM_QUERY = "Ljakarta/ws/rs/QueryParam;";
        public static final String PARAM_HEADER = "Ljakarta/ws/rs/HeaderParam;";
        public static final String PARAM_FORM = "Ljakarta/ws/rs/FormParam;";
        public static final String PARAM_COOKIE = "Ljakarta/ws/rs/CookieParam;";
        public static final String PARAM_DEFAULT_VALUE = "Ljakarta/ws/rs/DefaultValue;";
        public static final String PARAM_BEAN = "Ljakarta/ws/rs/BeanParam;";
        public static final String PARAM_MATRIX = "Ljakarta/ws/rs/MatrixParam;";
        public static final String DEFAULT_VALUE = "Ljakarta/ws/rs/DefaultValue;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                HEAD, GET, POST, PUT, DELETE, OPTIONS, METHOD, WEBSERVLET, WEBFILTER
        ));
    }

    public static final class Spring {

        private Spring() {
        }

        public static final String REQUEST_MAPPING = "Lorg/springframework/web/bind/annotation/RequestMapping;";
        public static final String DELETE_MAPPING = "Lorg/springframework/web/bind/annotation/DeleteMapping;";
        public static final String GET_MAPPING = "Lorg/springframework/web/bind/annotation/GetMapping;";
        public static final String PATCH_MAPPING = "Lorg/springframework/web/bind/annotation/PatchMapping;";
        public static final String POST_MAPPING = "Lorg/springframework/web/bind/annotation/PostMapping;";
        public static final String PUT_MAPPING = "Lorg/springframework/web/bind/annotation/PutMapping;";
        public static final String REQUEST_PARAMETER = "Lorg/springframework/web/bind/annotation/RequestParam;";
        public static final String PARAM_BODY = "Lorg/springframework/web/bind/annotation/RequestBody;";
        public static final String PARAM_PART = "Lorg/springframework/web/bind/annotation/RequestPart;";
        public static final String PARAM_HEADER = "Lorg/springframework/web/bind/annotation/RequestHeader;";
        public static final String PARAM_COOKIE = "Lorg/springframework/web/bind/annotation/CookieValue;";
        public static final String PARAM_MODEL = "Lorg/springframework/web/bind/annotation/ModelAttribute;";
        public static final String PARAM_PATH = "Lorg/springframework/web/bind/annotation/PathVariable;";
        public static final String BIND_PARAM = "Lorg/springframework/web/bind/annotation/BindParam;";

        public static final List<String> REQUEST_METHODS = Collections.unmodifiableList(Arrays.asList(
                REQUEST_MAPPING, DELETE_MAPPING, GET_MAPPING,
                PATCH_MAPPING, POST_MAPPING, PUT_MAPPING
        ));
    }

    public static final class Serialization {

        private Serialization() {
        }

        public static final String KOTLINX_SERIALNAME = "Lkotlinx/serialization/SerialName;";
        public static final String GSON_SERIALIZED_NAME = "Lcom/google/gson/annotations/SerializedName;";
        public static final String JAX_XML_ELEMENT = "Ljavax/xml/bind/annotation/XmlElement;";
        public static final String JAKARTA_XML_ELEMENT = "Ljakarta/xml/bind/annotation/XmlElement;";
        public static final String JACKSON_JSON_PROPERTY = "Lcom/fasterxml/jackson/annotation/JsonProperty;";
        public static final String CODEHAUS_JSON_PROPERTY = "Lorg/codehaus/jackson/annotate/JsonProperty;";
        public static final String SQUAREUP_MOSHI_JSON = "Lcom/squareup/moshi/Json;";
        public static final String JAKARTA_JSONBPROPERTY = "Ljakarta/json/bind/annotation/JsonbProperty;";
        public static final String JAX_JSONBPROPERTY = "Ljavax/json/bind/annotation/JsonbProperty;";
        public static final String APACHE_JOHNZON_PROPERTY = "Lorg/apache/johnzon/mapper/JohnzonProperty;";
        public static final String FASTJSON_JSONFIELD = "Lcom/alibaba/fastjson/annotation/JSONField;";
        public static final String FASTJSON2_JSONFIELD = "Lcom/alibaba/fastjson2/annotation/JSONField;";
        public static final String XSTREAM_ALIAS = "Lcom/thoughtworks/xstream/annotations/XStreamAlias;";
        public static final String JAX_COLUMN = "Ljavax/persistence/Column;";

        public static final List<String> SERIALIZATION_ANNOTATIONS = Collections.unmodifiableList(
                Arrays.asList(KOTLINX_SERIALNAME, GSON_SERIALIZED_NAME, JAX_XML_ELEMENT,
                        JAKARTA_XML_ELEMENT, JACKSON_JSON_PROPERTY, SQUAREUP_MOSHI_JSON,
                        JAKARTA_JSONBPROPERTY, JAX_JSONBPROPERTY, APACHE_JOHNZON_PROPERTY,
                        FASTJSON_JSONFIELD, FASTJSON2_JSONFIELD, CODEHAUS_JSON_PROPERTY,
                        XSTREAM_ALIAS, JAX_COLUMN)
        );
    }

    public static final class Ignore {

        private Ignore() {
        }

        public static final String HIBERNATE_GENERIC_GENERATOR = "Lorg/hibernate/annotations/GenericGenerator;";
        public static final String JETBRAINS_NOTNULL = "Lorg/jetbrains/annotations/NotNull;";
        public static final String JAX_NOTNULL = "Ljavax/validation/constraints/NotNull;";
        public static final String JAX_SIZE = "Ljavax/validation/constraints/Size;";
        public static final String JAX_PATTERN = "Ljavax/validation/constraints/Pattern;";
        public static final String JAX_MIN = "Ljavax/validation/constraints/Min;";
        public static final String JAX_MAX = "Ljavax/validation/constraints/Max;";
        public static final String JAX_DIGITS = "Ljavax/validation/constraints/Digits;";
        public static final String JAX_VALID = "Ljavax/validation/Valid;";
        public static final String JAX_XML_SCHEMA_TYPE = "Ljavax/xml/bind/annotation/XmlSchemaType;";
        public static final String JAKARTA_NOTNULL = "Ljakarta/validation/constraints/NotNull;";
        public static final String JAKARTA_SIZE = "Ljakarta/validation/constraints/Size;";
        public static final String JAKARTA_PATTERN = "Ljakarta/validation/constraints/Pattern;";
        public static final String JAKARTA_MIN = "Ljakarta/validation/constraints/Min;";
        public static final String JAKARTA_MAX = "Ljakarta/validation/constraints/Max;";
        public static final String JAKARTA_DIGITS = "Ljakarta/validation/constraints/Digits;";
        public static final String JAKARTA_VALID = "Ljakarta/validation/Valid;";
        public static final String JAKARTA_XML_SCHEMA_TYPE = "Ljakarta/xml/bind/annotation/XmlSchemaType;";
        
        
        public static final List<String> IGNORE_ANNOTATIONS = Collections.unmodifiableList(
                Arrays.asList(
                        HIBERNATE_GENERIC_GENERATOR,
                        JETBRAINS_NOTNULL,
                        JAX_NOTNULL, JAX_SIZE, JAX_PATTERN, JAX_MIN, JAX_MAX, JAX_DIGITS, JAX_XML_SCHEMA_TYPE, JAX_VALID,
                        JAKARTA_NOTNULL, JAKARTA_SIZE, JAKARTA_PATTERN, JAKARTA_MIN, JAKARTA_MAX, JAKARTA_DIGITS, JAKARTA_XML_SCHEMA_TYPE, JAKARTA_VALID
                )
        );
    }

    public static final List<String> BUILD_REQUEST_ANNOTATIONS = Collections.unmodifiableList(
            Stream.of(
                    Struts.REQUEST_METHODS, Retrofit.REQUEST_METHODS, Ktorfit.REQUEST_METHODS,
                    JaxRs.REQUEST_METHODS, Jakarta.REQUEST_METHODS, Micronaut.REQUEST_METHODS,
                    Spring.REQUEST_METHODS, Feign.REQUEST_METHODS, Swagger.REQUEST_METHODS,
                    OpenApi.REQUEST_METHODS, MicroProfileOpenApi.REQUEST_METHODS
            ).flatMap(List::stream).collect(Collectors.toList())
    );

    public static final String KOTLIN_METADATA = "Lkotlin/";
}
