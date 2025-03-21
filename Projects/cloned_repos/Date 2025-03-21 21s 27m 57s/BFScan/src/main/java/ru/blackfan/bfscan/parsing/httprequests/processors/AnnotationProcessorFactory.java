package ru.blackfan.bfscan.parsing.httprequests.processors;

import java.util.ArrayList;
import java.util.List;

public class AnnotationProcessorFactory {
    private static final List<AnnotationProcessor> processors = new ArrayList<>();

    static {
        processors.add(new SwaggerProcessor());
        processors.add(new SpringProcessor());
        processors.add(new JaxJakartaProcessor());
        processors.add(new MicronautProcessor());
        processors.add(new FeignProcessor());
        processors.add(new Struts2Processor());
        processors.add(new RetrofitProcessor());
        processors.add(new CommonProcessor());
        processors.add(new OpenApiProcessor());
    }

    public static List<AnnotationProcessor> getProcessors() {
        return processors;
    }
} 