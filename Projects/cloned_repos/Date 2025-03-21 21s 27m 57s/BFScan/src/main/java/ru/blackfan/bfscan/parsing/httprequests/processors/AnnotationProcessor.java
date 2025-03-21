package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.RootNode;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public interface AnnotationProcessor {

    ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int regNum,
            ArgType var,
            RootNode rn) throws Exception;

    boolean processMethodAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            RootNode rn)
            throws Exception;

    boolean processClassAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            String globalBasePath);
}
