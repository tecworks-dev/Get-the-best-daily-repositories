package ru.blackfan.bfscan.parsing.httprequests;

import io.swagger.v3.core.util.Yaml;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.servers.Server;
import io.swagger.v3.oas.models.tags.Tag;
import jadx.api.JadxDecompiler;
import jadx.api.JavaClass;
import jadx.api.JavaMethod;
import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.IAnnotation;
import jadx.api.plugins.input.data.attributes.JadxAttrType;
import jadx.api.plugins.input.data.attributes.types.AnnotationMethodParamsAttr;
import jadx.api.plugins.input.data.attributes.types.AnnotationsAttr;
import jadx.api.plugins.input.data.IDebugInfo;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.CodeVar;
import jadx.core.dex.instructions.args.RegisterArg;
import jadx.core.dex.instructions.args.SSAVar;
import jadx.core.dex.nodes.ClassNode;
import jadx.core.dex.nodes.MethodNode;
import jadx.core.dex.nodes.RootNode;
import jadx.core.utils.exceptions.DecodeException;
import java.io.PrintWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.blackfan.bfscan.config.ConfigLoader;
import ru.blackfan.bfscan.config.ProcessorConfig;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.processors.AnnotationProcessor;
import ru.blackfan.bfscan.parsing.httprequests.processors.AnnotationProcessorFactory;
import ru.blackfan.bfscan.parsing.httprequests.processors.AnnotationUtils;
import ru.blackfan.bfscan.parsing.httprequests.processors.ArgProcessingState;
import ru.blackfan.bfscan.parsing.httprequests.processors.MinifiedProcessor;

public class HTTPRequestProcessor {

    private static final Logger logger = LoggerFactory.getLogger(HTTPRequestProcessor.class);
    public static final List<String> EXCLUDED_PACKAGES = ConfigLoader.getExcludedPackages();

    private final JadxDecompiler jadx;
    private final PrintWriter methodsWriter;
    private final PrintWriter openApiWriter;
    private final OpenAPI openApi;
    private final String apiBasePath;
    private final String apiHost;
    private final ResourceProcessor resourceProcessor;

    public HTTPRequestProcessor(JadxDecompiler jadx, URI apiUrl, boolean minifiedAnnotationsSupport, PrintWriter methodsWriter, PrintWriter openApiWriter) {
        this.jadx = jadx;
        this.apiBasePath = apiUrl.getPath();
        this.apiHost = apiUrl.getHost();
        this.methodsWriter = methodsWriter;
        this.openApiWriter = openApiWriter;
        this.openApi = initializeOpenApi(apiUrl);
        this.resourceProcessor = new ResourceProcessor(jadx, apiUrl);
        ProcessorConfig.getInstance().setMinifiedAnnotationsSupport(minifiedAnnotationsSupport);
    }

    private OpenAPI initializeOpenApi(URI apiUrl) {
        return new OpenAPI()
                .info(new Info().title("API Methods").version("1.0"))
                .addServersItem(createOpenApiServer(apiUrl));
    }

    private Server createOpenApiServer(URI apiUrl) {
        String port = apiUrl.getPort() != -1 ? ":" + apiUrl.getPort() : "";
        String serverUrl = String.format("%s://%s%s", apiUrl.getScheme(), apiUrl.getHost(), port);
        return new Server().url(serverUrl);
    }

    public void processHttpMethods() {
        writeHeader();
        processClasses();
        processResources();
        writeOutput();
    }

    private void writeHeader() {
        methodsWriter.println("**HTTP Methods**\r\n");
    }

    private void buildRequests(MultiHTTPRequest request) {
        request.getRequests().forEach(req -> {
            req.addOpenApiPath(request.getClassName(), request.getMethodName(), openApi);
            writeRequest(req, request);
        });
    }

    private void writeRequest(HTTPRequest req, MultiHTTPRequest request) {
        methodsWriter.println("**Method**: " + request.getClassName() + "->" + request.getMethodName() + "\r\n");
        if (!req.additionalInformation.isEmpty()) {
            req.additionalInformation.forEach(info
                    -> methodsWriter.println("* " + info)
            );
        }
        try {
            methodsWriter.println("```\r\n" + req.format() + "\r\n```\r\n\r\n");
        } catch (Exception ex) {
            logger.error("Error request format " + request.getClassName() + "->" + request.getMethodName(), ex);
        }
        methodsWriter.flush();
    }

    private void writeOutput() {
        openApiWriter.print(Yaml.pretty(openApi));
        openApiWriter.flush();
        methodsWriter.flush();
    }

    private void processClasses() {
        jadx.getClasses().stream()
                .filter(this::shouldProcessClass)
                .forEach(this::processClass);
    }

    private boolean shouldProcessClass(JavaClass cls) {
        return EXCLUDED_PACKAGES.stream()
                .noneMatch(pkg -> cls.getFullName().startsWith(pkg));
    }

    private void processResources() {
        List<MultiHTTPRequest> requests = resourceProcessor.processResources();
        if (requests != null) {
            requests.forEach(this::processRequestsFromRes);
        }
    }

    private void processRequestsFromRes(MultiHTTPRequest multiRequest) {
        addOpenApiTag(multiRequest.getClassName());
        buildRequests(multiRequest);
    }

    private void processClass(JavaClass cls) {
        MultiHTTPRequest classRequest = new MultiHTTPRequest(apiHost, apiBasePath, cls.getFullName(), "class");
        processClassAnnotations(classRequest, cls, apiBasePath);
        processClassMethods(cls, classRequest);
    }

    private void processClassAnnotations(MultiHTTPRequest request, JavaClass cls, String apiBasePath) {
        boolean buildRequest = false;
        AnnotationsAttr classAList = cls.getClassNode().get(JadxAttrType.ANNOTATION_LIST);
        if (classAList != null && !classAList.isEmpty()) {
            for (IAnnotation a : classAList.getAll()) {
                String aCls = AnnotationUtils.getAnnotationClass(a, cls.getClassNode().root());
                Map<String, EncodedValue> annotationValues = a.getValues();

                for (AnnotationProcessor processor : AnnotationProcessorFactory.getProcessors()) {
                    if (processor.processClassAnnotations(request, aCls, annotationValues, apiBasePath)) {
                        buildRequest = true;
                    }
                }
            }
        }
        if (buildRequest) {
            addOpenApiTag(cls.getFullName());
            buildRequests(request);
        }
    }

    public void addOpenApiTag(String className) {
        if (openApi.getTags() == null) {
            openApi.addTagsItem(new Tag().name(className));
        } else if (openApi.getTags().stream().noneMatch(tag -> className.equals(tag.getName()))) {
            openApi.addTagsItem(new Tag().name(className));
        }
    }

    private void processClassMethods(JavaClass cls, MultiHTTPRequest classRequest) {
        for (JavaMethod mth : cls.getMethods()) {
            try {
                processClassMethod(classRequest, cls, mth);
            } catch (Exception ex) {
                logger.error("Error processing method " + mth.getFullName(), ex);
            }
        }
    }

    private void processClassMethod(MultiHTTPRequest classRequest, JavaClass cls, JavaMethod mth) {
        MultiHTTPRequest methodRequest = processMethod(
                new MultiHTTPRequest(classRequest, cls.getFullName(), mth.getName()),
                mth,
                cls
        );

        if (methodRequest != null) {
            addOpenApiTag(cls.getFullName());
            buildRequests(methodRequest);
        }
    }

    private MultiHTTPRequest processMethod(MultiHTTPRequest request, JavaMethod mth, JavaClass cls) {
        RootNode rootNode = cls.getClassNode().root();
        MethodNode methodNode = mth.getMethodNode();
        AnnotationsAttr aList = methodNode.get(JadxAttrType.ANNOTATION_LIST);
        if (aList == null || aList.isEmpty()) {
            return null;
        }
        if (processMethodAnnotations(request, aList, rootNode)) {
            processMethodParameters(request, methodNode, rootNode);
            return request;
        }
        return null;
    }

    private boolean processMethodAnnotations(MultiHTTPRequest request, AnnotationsAttr aList, RootNode rn) {
        boolean buildRequest = false;
        for (IAnnotation annotation : aList.getAll()) {
            String aCls = AnnotationUtils.getAnnotationClass(annotation, rn);
            Map<String, EncodedValue> annotationValues = annotation.getValues();

            if (Constants.BUILD_REQUEST_ANNOTATIONS.contains(aCls)) {
                buildRequest = true;
            }

            boolean processed = false;
            for (AnnotationProcessor processor : AnnotationProcessorFactory.getProcessors()) {
                try {
                    processed |= processor.processMethodAnnotations(request, aCls, annotationValues, rn);
                } catch (Exception ex) {
                    logger.error("Error in processMethodAnnotations: " + request.getClassName() + "->" + request.getMethodName());
                }
            }
            if (!processed && ProcessorConfig.getInstance().isMinifiedAnnotationsSupport()) {
                if (MinifiedProcessor.processMethodAnnotations(request, aCls, annotationValues, rn)) {
                    buildRequest = true;
                }
            }
        }
        return buildRequest;
    }

    private void processMethodParameters(MultiHTTPRequest request, MethodNode mn, RootNode rn) {
        AnnotationMethodParamsAttr paramsAnnotations = mn.get(JadxAttrType.ANNOTATION_MTH_PARAMETERS);
        try {
            mn.load();
            if ((mn.getInstructions() == null) && mn.getInsnsCount() != 0) {
                mn.reload();
            }
        } catch (DecodeException e) {
            logger.error("Failed to load method " + mn.getName(), e);
            return;
        }

        List<ILocalVar> localVars = getLocalVars(mn);
        processMethodArgs(request, mn, rn, paramsAnnotations, localVars);
    }

    private List<ILocalVar> getLocalVars(MethodNode mn) {
        List<ILocalVar> localVars = new ArrayList<>();
        IDebugInfo debugInfo = mn.getDebugInfo();
        if (debugInfo != null && debugInfo.getLocalVars() != null) {
            localVars = debugInfo.getLocalVars();
        }
        return localVars;
    }

    private void processMethodArgs(MultiHTTPRequest request, MethodNode mn, RootNode rn,
            AnnotationMethodParamsAttr paramsAnnotations, List<ILocalVar> localVars) {
        List<RegisterArg> methodArgs = mn.getArgRegs();
        int annNum = 0;
        for (RegisterArg mthArg : methodArgs) {
            CodeVar var = getCodeVar(mthArg);
            processMethodArg(request, var, paramsAnnotations, localVars, rn, mthArg.getRegNum(), annNum++);
        }
    }

    private CodeVar getCodeVar(RegisterArg mthArg) {
        SSAVar ssaVar = mthArg.getSVar();
        return ssaVar == null ? CodeVar.fromMthArg(mthArg, true) : ssaVar.getCodeVar();
    }

    private void processMethodArg(MultiHTTPRequest request, CodeVar var,
            AnnotationMethodParamsAttr paramsAnnotations,
            List<ILocalVar> localVars, RootNode rn, int regNum, int annNum) {
        if (paramsAnnotations != null) {
            processAnnotatedArg(request, var, paramsAnnotations, localVars, rn, regNum, annNum);
        } else {
            processUnannotatedArg(request, var, localVars, regNum, rn);
        }
    }

    private void processAnnotatedArg(MultiHTTPRequest request, CodeVar var,
            AnnotationMethodParamsAttr paramsAnnotations,
            List<ILocalVar> localVars, RootNode rootNode, int regNum, int annNum) {
        List<AnnotationsAttr> paramList = paramsAnnotations.getParamList();
        AnnotationsAttr argAnnList = paramList.get(annNum);

        if (argAnnList == null || argAnnList.isEmpty()) {
            processUnannotatedArg(request, var, localVars, regNum, rootNode);
            return;
        }

        ParameterInfo paramInfo = new ParameterInfo();
        ArgProcessingState state = ArgProcessingState.NOT_PROCESSED;
        Map<String,Map<String, EncodedValue>> minifiedAnnotations = new HashMap();

        for (IAnnotation annotation : argAnnList.getAll()) {
            String aCls = AnnotationUtils.getAnnotationClass(annotation, rootNode);
            Map<String, EncodedValue> annotationValues = annotation.getValues();
            
            ArgProcessingState currentAnnotationState = ArgProcessingState.NOT_PROCESSED;
            for (AnnotationProcessor processor : AnnotationProcessorFactory.getProcessors()) {
                try {
                    ArgProcessingState currentProcessorState = processor.processParameterAnnotations(request, paramInfo, aCls, annotationValues, localVars, regNum, var.getType(), rootNode);
                    if(currentProcessorState != ArgProcessingState.NOT_PROCESSED) {
                        currentAnnotationState = currentProcessorState;
                        if(state != ArgProcessingState.PARAMETER_CREATED) {
                            state = currentProcessorState;
                        }
                    }
                } catch (Exception ex) {
                    logger.error("Error in processAnnotatedArg: " + request.getClassName() + "->" + request.getMethodName());
                }
            }
            if ((currentAnnotationState == ArgProcessingState.NOT_PROCESSED) && ProcessorConfig.getInstance().isMinifiedAnnotationsSupport()) {
                minifiedAnnotations.put(aCls, annotationValues);
            }
        }
        if(state != ArgProcessingState.PARAMETER_CREATED) {
            for (Map.Entry<String, Map<String, EncodedValue>> entry : minifiedAnnotations.entrySet()) {
                ArgProcessingState currentState = MinifiedProcessor.processParameterAnnotations(request, paramInfo, entry.getKey(), entry.getValue(), localVars, regNum, var.getType(), rootNode);
                if(currentState == ArgProcessingState.PARAMETER_CREATED) {
                    state = currentState;
                    break;
                }
            } 
        }
        if(state != ArgProcessingState.PARAMETER_CREATED) {
            processUnannotatedArg(request, var, localVars, regNum, rootNode);
        }
    }

    private void processUnannotatedArg(MultiHTTPRequest request, CodeVar var, List<ILocalVar> localVars, int regNum, RootNode rootNode) {
        if (AnnotationUtils.isSimpleObject(var.getType())) {
            String paramName = AnnotationUtils.getParamName(null, null, localVars, regNum);
            AnnotationUtils.processQueryParameter(request, paramName, "", var.getType(), rootNode);
        } else {
            if (!var.getType().isPrimitive() && var.getType() != null) {
                try {
                    request.putBodyParameters((Map) AnnotationUtils.argTypeToValue("value", var.getType(), rootNode, new HashSet<>(), true));
                } catch (Exception ex) {
                    logger.error("Error in processUnannotatedArg: " + request.getClassName() + "->" + request.getMethodName());
                }
            }
        }
    }
}
