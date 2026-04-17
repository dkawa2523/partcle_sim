import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DumpIcpModel {
    public static void main(String[] args) throws Exception {
        Path mph = Paths.get(firstNonEmpty(argValue(args, "--mph"), System.getenv("COMSOL_ICP_MPH"), "data/icp_rf_bias_cf4_o2_si_etching (2).mph"));
        Path out = Paths.get(firstNonEmpty(argValue(args, "--outdir"), System.getenv("COMSOL_ICP_OUTDIR"), "_external_exports/icp_model_dump"));
        Files.createDirectories(out);
        Model model = ModelUtil.load("icp_model_dump", mph.toString());
        try {
            writeMethods(out.resolve("model_param_methods.txt"), call(model, "param"));
            writeStringArray(out.resolve("model_param_varnames.txt"), stringArrayOrEmpty(call(call(model, "param"), "varnames")));
            writeStringArray(out.resolve("physics_tags.txt"), stringArrayOrEmpty(call(call(model, "physics"), "tags")));
            writeStringArray(out.resolve("study_tags.txt"), stringArrayOrEmpty(call(call(model, "study"), "tags")));
            writeStringArray(out.resolve("solver_tags.txt"), stringArrayOrEmpty(call(call(model, "sol"), "tags")));
            writeStringArray(out.resolve("dataset_tags.txt"), stringArrayOrEmpty(call(call(model, "result"), "dataset"), "tags"));
            writeParameters(out.resolve("model_parameters.csv"), call(model, "param"));
            writeModelInfo(out.resolve("model_info.txt"), model);
        } finally {
            try {
                ModelUtil.disconnect();
            } catch (Throwable ignored) {
            }
        }
    }

    private static void writeModelInfo(Path path, Model model) throws Exception {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            w.println("comsol_version," + csv(String.valueOf(call(model, "getComsolVersion"))));
            w.println("title," + csv(String.valueOf(call(model, "title"))));
            w.println("description," + csv(String.valueOf(call(model, "description"))));
            w.println("file_path," + csv(String.valueOf(call(model, "getFilePath"))));
        }
    }

    private static void writeParameters(Path path, Object param) throws Exception {
        String[] names = stringArrayOrEmpty(call(param, "varnames"));
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            w.println("name,expression,value,description");
            for (String name : names) {
                String expr = callString(param, "get", name);
                String value = callString(param, "evaluate", name);
                String descr = callString(param, "descr", name);
                w.println(csv(name) + "," + csv(expr) + "," + csv(value) + "," + csv(descr));
            }
        }
    }

    private static void writeMethods(Path path, Object target) throws Exception {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            if (target == null) {
                return;
            }
            for (Method m : target.getClass().getMethods()) {
                w.print(m.getName());
                w.print("(");
                Class<?>[] params = m.getParameterTypes();
                for (int i = 0; i < params.length; i++) {
                    if (i > 0) {
                        w.print(",");
                    }
                    w.print(params[i].getName());
                }
                w.print(") -> ");
                w.println(m.getReturnType().getName());
            }
        }
    }

    private static void writeStringArray(Path path, String[] values) throws Exception {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            for (String value : values) {
                w.println(value);
            }
        }
    }

    private static String callString(Object target, String name, String arg) {
        try {
            Object out = call(target, name, arg);
            return out == null ? "" : String.valueOf(out);
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static Object call(Object target, String name, Object... args) throws Exception {
        if (target == null) {
            throw new NullPointerException("target");
        }
        Method found = null;
        for (Method method : target.getClass().getMethods()) {
            if (!method.getName().equals(name) || method.getParameterCount() != args.length) {
                continue;
            }
            found = method;
            break;
        }
        if (found == null) {
            throw new NoSuchMethodException(name);
        }
        return found.invoke(target, args);
    }

    private static String[] stringArrayOrEmpty(Object raw) {
        if (raw instanceof String[]) {
            return (String[]) raw;
        }
        return new String[0];
    }

    private static String[] stringArrayOrEmpty(Object target, String method) {
        try {
            return stringArrayOrEmpty(call(target, method));
        } catch (Throwable ignored) {
            return new String[0];
        }
    }

    private static String csv(String raw) {
        if (raw == null) {
            raw = "";
        }
        return "\"" + raw.replace("\"", "\"\"").replace("\r", " ").replace("\n", " ") + "\"";
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                return value;
            }
        }
        return "";
    }

    private static String argValue(String[] args, String key) {
        if (args == null) {
            return "";
        }
        for (int i = 0; i + 1 < args.length; i++) {
            if (key.equals(args[i])) {
                return args[i + 1];
            }
        }
        return "";
    }
}
