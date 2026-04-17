import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class IcpCf4O2SiEtchExporter {
    private static final String[] DEFAULT_REQUIRED = {"ux", "uy", "mu", "E_x", "E_y"};

    private static Path mphPath;
    private static Path configPath;
    private static Path outDir;
    private static String dataset = "dset1";
    private static String meshTag = "mesh1";
    private static String geometryModelUnit = "cm";
    private static double geometryScaleMPerModelUnit = 0.01;
    private static double rfBiasV = 20.0;
    private static double qRefC = 1.602176634e-19;
    private static double mRefKg = 6.64215627e-26;
    private static double rMin = 0.0;
    private static double rMax = 0.16;
    private static double zMin = 0.0;
    private static double zMax = 0.20;
    private static int rCount = 241;
    private static int zCount = 301;
    private static Map<String, List<String>> expressions = new LinkedHashMap<>();
    private static List<String> required = Arrays.asList(DEFAULT_REQUIRED);
    private static Map<String, String> selectedDatasets = new LinkedHashMap<>();

    public static void main(String[] args) throws Exception {
        loadOptions(args);
        loadConfig(configPath);
        Files.createDirectories(outDir);

        Model model = ModelUtil.load("icp_model", mphPath.toString());
        trySetModelParameter(model, "Vrf", rfBiasV + "[V]");
        tryWriteMethodList(outDir.resolve("result_methods.txt"), safeCall(model, "result"));
        Object datasetList = safeCall(safeCall(model, "result"), "dataset");
        tryWriteMethodList(outDir.resolve("dataset_list_methods.txt"), datasetList);
        tryWriteTags(outDir.resolve("dataset_tags.txt"), datasetList);

        exportMesh(model, outDir.resolve("mesh.mphtxt"));
        tryWriteMaterialInventory(model, outDir.resolve("material_inventory.json"));
        try {
            call(call(model, "result"), "run");
        } catch (Throwable ignored) {
        }

        double[] rAxis = linspace(rMin, rMax, rCount);
        double[] zAxis = linspace(zMin, zMax, zCount);
        String[] datasetTags = readDatasetTags(model);
        Map<String, String> selected = selectExpressions(model, datasetTags, probeCoordinates(rAxis, zAxis));

        writeInventory(outDir.resolve("expression_inventory.json"), selected);
        writeFieldSamples(outDir.resolve("field_samples.csv"), model, selected, rAxis, zAxis);
        writeManifest(outDir.resolve("export_manifest.json"), selected, rAxis, zAxis);

        try {
            ModelUtil.disconnect();
        } catch (Throwable ignored) {
        }
    }

    private static void loadOptions(String[] args) {
        if (args == null) {
            args = new String[0];
        }
        Map<String, String> map = new LinkedHashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            map.put(args[i].replaceFirst("^-+", ""), args[i + 1]);
        }
        String mph = firstNonEmpty(map.get("mph"), System.getenv("COMSOL_ICP_MPH"), "data/icp_rf_bias_cf4_o2_si_etching (2).mph");
        String config = firstNonEmpty(
            map.get("config"),
            System.getenv("COMSOL_ICP_CONFIG"),
            "external/comsol_icp_export/config/icp_cf4_o2_v20.json"
        );
        String out = firstNonEmpty(map.get("outdir"), System.getenv("COMSOL_ICP_OUTDIR"), "_external_exports/icp_cf4_o2_v20");
        mphPath = Paths.get(mph);
        configPath = Paths.get(config);
        outDir = Paths.get(out);
    }

    private static void loadConfig(Path path) throws IOException {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        expressions = parseExpressionLists(text);
        if (expressions.containsKey("required")) {
            required = new ArrayList<>(expressions.get("required"));
            expressions.remove("required");
        }
        dataset = jsonString(text, "dataset", dataset);
        meshTag = jsonString(text, "mesh_tag", meshTag);
        geometryModelUnit = jsonString(text, "geometry_model_unit", geometryModelUnit);
        geometryScaleMPerModelUnit = jsonDouble(text, "geometry_scale_m_per_model_unit", geometryScaleMPerModelUnit);
        rfBiasV = jsonDouble(text, "rf_bias_v", rfBiasV);
        qRefC = jsonDouble(text, "q_ref_c", qRefC);
        mRefKg = jsonDouble(text, "m_ref_kg", mRefKg);
        rMin = jsonDouble(text, "r_min", rMin);
        rMax = jsonDouble(text, "r_max", rMax);
        zMin = jsonDouble(text, "z_min", zMin);
        zMax = jsonDouble(text, "z_max", zMax);
        rCount = (int) jsonDouble(text, "r_count", rCount);
        zCount = (int) jsonDouble(text, "z_count", zCount);
    }

    private static void trySetModelParameter(Model model, String name, String value) {
        try {
            Object param = call(model, "param");
            call(param, "set", name, value);
        } catch (Throwable ignored) {
        }
    }

    private static void exportMesh(Model model, Path out) {
        List<String> errors = new ArrayList<>();
        try {
            Object mesh = call(model, "mesh", meshTag);
            writeMphtxtFromMeshSequence(mesh, out);
            return;
        } catch (Throwable t) {
            errors.add(t.toString());
            tryWriteMethodList(outDir.resolve("mesh_methods.txt"), safeCall(model, "mesh", meshTag));
        }
        for (String compTag : new String[]{"comp1", "comp2"}) {
            try {
                Object comp = call(model, "component", compTag);
                Object mesh = call(comp, "mesh", meshTag);
                call(mesh, "export", out.toString());
                return;
            } catch (Throwable t) {
                errors.add(t.toString());
            }
        }
        tryWriteMethodList(outDir.resolve("model_methods.txt"), model);
        throw new RuntimeException("Could not export mesh.mphtxt through COMSOL Java API. Errors: " + errors);
    }

    private static void writeMphtxtFromMeshSequence(Object mesh, Path out) throws IOException {
        double[][] vertices = normalizeVertices((double[][]) call(mesh, "getVertex"));
        String[] types = (String[]) call(mesh, "getTypes");
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            int sdim = vertices.length == 0 ? 0 : vertices[0].length;
            writer.println(sdim + " # sdim");
            writer.println(vertices.length + " # number of mesh vertices");
            writer.println("# Mesh vertex coordinates");
            for (int i = 0; i < vertices.length; i++) {
                for (int d = 0; d < sdim; d++) {
                    if (d > 0) {
                        writer.print(" ");
                    }
                    writer.print(String.format(Locale.US, "%.17g", vertices[i][d]));
                }
                writer.println();
            }
            writer.println(types.length + " # number of element types");
            int typeIndex = 0;
            for (String type : types) {
                int[] entity = (int[]) call(mesh, "getElemEntity", type);
                int[][] elems = normalizeElements((int[][]) call(mesh, "getElem", type), entity.length);
                int nvp = elems.length == 0 ? 0 : elems[0].length;
                writer.println(typeIndex + " " + type + " # type name");
                writer.println(nvp + " # number of vertices per element");
                writer.println(elems.length + " # number of elements");
                writer.println("# Elements");
                for (int i = 0; i < elems.length; i++) {
                    for (int j = 0; j < nvp; j++) {
                        if (j > 0) {
                            writer.print(" ");
                        }
                        writer.print(elems[i][j]);
                    }
                    writer.println();
                }
                writer.println(entity.length + " # number of geometric entity indices");
                writer.println("# Geometric entity indices");
                for (int i = 0; i < entity.length; i++) {
                    writer.println(entity[i]);
                }
                typeIndex++;
            }
        }
    }

    private static double[][] normalizeVertices(double[][] raw) {
        if (raw.length == 0) {
            return raw;
        }
        if (raw.length <= 3 && raw[0].length > raw.length) {
            int sdim = raw.length;
            int n = raw[0].length;
            double[][] out = new double[n][sdim];
            for (int d = 0; d < sdim; d++) {
                for (int i = 0; i < n; i++) {
                    out[i][d] = raw[d][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static int[][] normalizeElements(int[][] raw, int elementCount) {
        if (raw.length == 0 || elementCount == raw.length) {
            return raw;
        }
        if (raw[0].length == elementCount) {
            int nvp = raw.length;
            int[][] out = new int[elementCount][nvp];
            for (int j = 0; j < nvp; j++) {
                for (int i = 0; i < elementCount; i++) {
                    out[i][j] = raw[j][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static String[] readDatasetTags(Model model) {
        return new String[]{dataset};
    }

    private static Map<String, String> selectExpressions(Model model, String[] datasetTags, double[][] probeCoords) {
        Map<String, String> selected = new LinkedHashMap<>();
        Map<String, String> failures = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : expressions.entrySet()) {
            String key = entry.getKey();
            List<String> fail = new ArrayList<>();
            for (String expr : entry.getValue()) {
                for (String datasetTag : datasetTags) {
                    String tag = "inv_" + sanitize(key);
                    try {
                        Object interp = createInterp(model, tag, datasetTag, expr);
                        double value = evalFirstFinite(interp, probeCoords);
                        if (Double.isFinite(value)) {
                            selected.put(key, expr);
                            selectedDatasets.put(key, datasetTag);
                            failures.put(key, "");
                            break;
                        }
                        fail.add(datasetTag + "/" + expr + ": non-finite");
                    } catch (Throwable t) {
                        fail.add(datasetTag + "/" + expr + ": " + t.getClass().getSimpleName() + ": " + t.getMessage());
                    } finally {
                        removeNumerical(model, tag);
                    }
                    if (selected.containsKey(key)) {
                        break;
                    }
                }
                if (selected.containsKey(key)) {
                    break;
                }
            }
            if (!selected.containsKey(key)) {
                failures.put(key, String.join("; ", fail));
            }
        }
        for (String key : required) {
            if (!selected.containsKey(key)) {
                throw new RuntimeException("Required COMSOL expression not found for " + key + ": " + failures.get(key));
            }
        }
        return selected;
    }

    private static Object createInterp(Model model, String tag, String datasetTag, String expr) {
        Object result = call(model, "result");
        Object numerical = call(result, "numerical");
        try {
            call(numerical, "remove", tag);
        } catch (Throwable ignored) {
        }
        call(numerical, "create", tag, "Interp");
        Object interp = call(result, "numerical", tag);
        call(interp, "set", "data", datasetTag);
        call(interp, "set", "solnum", new int[]{17});
        call(interp, "set", "expr", new String[]{expr});
        tryWriteMethodList(outDir.resolve("numerical_feature_methods.txt"), interp);
        return interp;
    }

    private static void removeNumerical(Model model, String tag) {
        try {
            call(call(call(model, "result"), "numerical"), "remove", tag);
        } catch (Throwable ignored) {
        }
    }

    private static double evalFirstFinite(Object interp, double[][] coords) {
        double[] values = evalMany(interp, coords);
        for (int i = 0; i < values.length; i++) {
            if (Double.isFinite(values[i])) {
                return values[i];
            }
        }
        return Double.NaN;
    }

    private static double[] evalMany(Object interp, double[][] coords) {
        call(interp, "setInterpolationCoordinates", coords);
        Object data = call(interp, "getData");
        return firstVector(data, coords[0].length);
    }

    private static void writeFieldSamples(Path out, Model model, Map<String, String> selected, double[] rAxis, double[] zAxis) throws IOException {
        Map<String, Object> features = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : selected.entrySet()) {
            features.put(entry.getKey(), createInterp(model, "grid_" + sanitize(entry.getKey()), selectedDatasets.get(entry.getKey()), entry.getValue()));
        }
        double[][] coords = gridCoordinates(rAxis, zAxis);
        Map<String, double[]> valuesByKey = new LinkedHashMap<>();
        for (String key : selected.keySet()) {
            valuesByKey.put(key, evalMany(features.get(key), coords));
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            writer.print("r,z,valid_mask");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();
            int idx = 0;
            for (double r : rAxis) {
                for (double z : zAxis) {
                    boolean valid = true;
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[idx];
                        if (required.contains(key) && !Double.isFinite(value)) {
                            valid = false;
                        }
                    }
                    writer.printf(Locale.US, "%.17g,%.17g,%d", r, z, valid ? 1 : 0);
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[idx];
                        writer.print(",");
                        writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                    }
                    writer.println();
                    idx++;
                }
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "grid_" + sanitize(key));
            }
        }
    }

    private static void writeInventory(Path out, Map<String, String> selected) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"required\": " + jsonArray(required) + ",\n");
            w.write("  \"selected\": {\n");
            int i = 0;
            for (Map.Entry<String, List<String>> entry : expressions.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                String key = entry.getKey();
                w.write("    " + json(key) + ": {");
                w.write("\"expression\": " + json(selected.containsKey(key) ? selected.get(key) : "") + ", ");
                w.write("\"dataset\": " + json(selectedDatasets.containsKey(key) ? selectedDatasets.get(key) : "") + ", ");
                w.write("\"available\": " + selected.containsKey(key));
                w.write("}");
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void writeManifest(Path out, Map<String, String> selected, double[] rAxis, double[] zAxis) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"comsol_java_api_external_export\",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"comsol_version\": " + json("") + ",\n");
            w.write("  \"rf_bias_v\": " + jsonNumber(rfBiasV) + ",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"mesh_tag\": " + json(meshTag) + ",\n");
            w.write("  \"geometry_model_unit\": " + json(geometryModelUnit) + ",\n");
            w.write("  \"geometry_scale_m_per_model_unit\": " + jsonNumber(geometryScaleMPerModelUnit) + ",\n");
            w.write("  \"solver_coordinate_unit\": \"m\",\n");
            w.write("  \"q_ref_c\": " + jsonNumber(qRefC) + ",\n");
            w.write("  \"m_ref_kg\": " + jsonNumber(mRefKg) + ",\n");
            w.write("  \"electric_acceleration_formula\": \"ax=q_ref_c/m_ref_kg*E_x, ay=q_ref_c/m_ref_kg*E_y\",\n");
            w.write("  \"grid_shape\": [" + rAxis.length + ", " + zAxis.length + "],\n");
            w.write("  \"r_bounds\": [" + jsonNumber(rAxis[0]) + ", " + jsonNumber(rAxis[rAxis.length - 1]) + "],\n");
            w.write("  \"z_bounds\": [" + jsonNumber(zAxis[0]) + ", " + jsonNumber(zAxis[zAxis.length - 1]) + "],\n");
            w.write("  \"r_bounds_model_units\": [" + jsonNumber(rAxis[0]) + ", " + jsonNumber(rAxis[rAxis.length - 1]) + "],\n");
            w.write("  \"z_bounds_model_units\": [" + jsonNumber(zAxis[0]) + ", " + jsonNumber(zAxis[zAxis.length - 1]) + "],\n");
            w.write("  \"r_bounds_m\": [" + jsonNumber(rAxis[0] * geometryScaleMPerModelUnit) + ", " + jsonNumber(rAxis[rAxis.length - 1] * geometryScaleMPerModelUnit) + "],\n");
            w.write("  \"z_bounds_m\": [" + jsonNumber(zAxis[0] * geometryScaleMPerModelUnit) + ", " + jsonNumber(zAxis[zAxis.length - 1] * geometryScaleMPerModelUnit) + "],\n");
            w.write("  \"expression_mapping\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : selected.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  },\n");
            w.write("  \"expression_dataset\": {\n");
            int j = 0;
            for (Map.Entry<String, String> entry : selectedDatasets.entrySet()) {
                if (j++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void tryWriteMaterialInventory(Model model, Path out) {
        try {
            writeMaterialInventory(model, out);
        } catch (Throwable t) {
            try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
                w.write("{\n");
                w.write("  \"source_kind\": \"comsol_material_inventory\",\n");
                w.write("  \"status\": \"unavailable\",\n");
                w.write("  \"error\": " + json(t.getClass().getSimpleName() + ": " + t.getMessage()) + ",\n");
                w.write("  \"materials\": []\n");
                w.write("}\n");
            } catch (Throwable ignored) {
            }
        }
    }

    private static void writeMaterialInventory(Model model, Path out) throws IOException {
        Object materialList = call(model, "material");
        tryWriteMethodList(outDir.resolve("material_list_methods.txt"), materialList);
        String[] tags = (String[]) call(materialList, "tags");
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"comsol_material_inventory\",\n");
            w.write("  \"status\": \"ok\",\n");
            w.write("  \"entity_id_base\": \"comsol_selection_entities_as_reported\",\n");
            w.write("  \"materials\": [\n");
            for (int i = 0; i < tags.length; i++) {
                String tag = tags[i];
                Object material = call(model, "material", tag);
                if (i == 0) {
                    tryWriteMethodList(outDir.resolve("material_methods.txt"), material);
                }
                Object selection = safeCall(material, "selection");
                if (i == 0 && selection != null) {
                    tryWriteMethodList(outDir.resolve("material_selection_methods.txt"), selection);
                }
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(tag) + ",\n");
                w.write("      \"label\": " + json(stringOrEmpty(safeCall(material, "label"))) + ",\n");
                w.write("      \"name\": " + json(stringOrEmpty(safeCall(material, "name"))) + ",\n");
                w.write("      \"selection_entities\": " + jsonIntArray(selectionEntities(selection)) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static int[] selectionEntities(Object selection) {
        if (selection == null) {
            return new int[0];
        }
        for (String method : new String[]{"entities", "inputEntities"}) {
            try {
                int[] out = normalizeIntArray(call(selection, method));
                if (out != null) {
                    return out;
                }
            } catch (Throwable ignored) {
            }
        }
        return new int[0];
    }

    private static int[] normalizeIntArray(Object value) {
        if (value instanceof int[]) {
            return (int[]) value;
        }
        if (value instanceof Integer[]) {
            Integer[] raw = (Integer[]) value;
            int[] out = new int[raw.length];
            for (int i = 0; i < raw.length; i++) {
                out[i] = raw[i] == null ? 0 : raw[i];
            }
            return out;
        }
        return null;
    }

    private static String stringOrEmpty(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static Object call(Object target, String name, Object... args) {
        Throwable last = null;
        for (Method method : target.getClass().getMethods()) {
            if (!method.getName().equals(name) || method.getParameterCount() != args.length) {
                continue;
            }
            try {
                return method.invoke(target, args);
            } catch (InvocationTargetException t) {
                Throwable cause = t.getCause() == null ? t : t.getCause();
                throw new RuntimeException("Method " + name + " threw: " + cause.getMessage(), cause);
            } catch (Throwable t) {
                last = t;
            }
        }
        throw new RuntimeException("No callable method " + name + " with " + args.length + " args on " + target.getClass(), last);
    }

    private static Object safeCall(Object target, String name, Object... args) {
        try {
            return call(target, name, args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static void tryWriteMethodList(Path path, Object target) {
        if (target == null) {
            return;
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            writer.println(target.getClass().getName());
            for (Method method : target.getClass().getMethods()) {
                Class[] params = method.getParameterTypes();
                List<String> names = new ArrayList<>();
                for (int i = 0; i < params.length; i++) {
                    names.add(params[i].getName());
                }
                writer.println(method.getName() + "(" + String.join(",", names) + ") -> " + method.getReturnType().getName());
            }
        } catch (Throwable ignored) {
        }
    }

    private static void tryWriteTags(Path path, Object list) {
        if (list == null) {
            return;
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            String[] tags = (String[]) call(list, "tags");
            for (int i = 0; i < tags.length; i++) {
                writer.println(tags[i]);
            }
        } catch (Throwable ignored) {
        }
    }

    private static double firstDouble(Object data) {
        if (data instanceof Double) {
            return (Double) data;
        }
        if (data instanceof double[]) {
            double[] a = (double[]) data;
            return a.length == 0 ? Double.NaN : a[0];
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            return a.length == 0 || a[0].length == 0 ? Double.NaN : a[0][0];
        }
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            return a.length == 0 || a[0].length == 0 || a[0][0].length == 0 ? Double.NaN : a[0][0][0];
        }
        throw new RuntimeException("Cannot extract double from " + data.getClass());
    }

    private static double[] firstVector(Object data, int expected) {
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            if (a.length == 0 || a[0].length == 0) {
                return filledNaN(expected);
            }
            return padded(a[0][0], expected);
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            if (a.length == 0) {
                return filledNaN(expected);
            }
            return padded(a[0], expected);
        }
        if (data instanceof double[]) {
            return padded((double[]) data, expected);
        }
        return new double[]{firstDouble(data)};
    }

    private static double[] padded(double[] values, int expected) {
        if (values.length == expected) {
            return values;
        }
        double[] out = filledNaN(expected);
        int n = Math.min(values.length, expected);
        for (int i = 0; i < n; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static double[] filledNaN(int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = Double.NaN;
        }
        return out;
    }

    private static double[][] gridCoordinates(double[] rAxis, double[] zAxis) {
        int n = rAxis.length * zAxis.length;
        double[][] coords = new double[2][n];
        int idx = 0;
        for (int i = 0; i < rAxis.length; i++) {
            for (int j = 0; j < zAxis.length; j++) {
                coords[0][idx] = rAxis[i];
                coords[1][idx] = zAxis[j];
                idx++;
            }
        }
        return coords;
    }

    private static double[][] probeCoordinates(double[] rAxis, double[] zAxis) {
        int[] frac = new int[]{1, 2, 3};
        double[][] coords = new double[2][frac.length * frac.length];
        int idx = 0;
        for (int a = 0; a < frac.length; a++) {
            for (int b = 0; b < frac.length; b++) {
                coords[0][idx] = rAxis[(rAxis.length - 1) * frac[a] / 4];
                coords[1][idx] = zAxis[(zAxis.length - 1) * frac[b] / 4];
                idx++;
            }
        }
        return coords;
    }

    private static double[] linspace(double min, double max, int count) {
        double[] out = new double[count];
        double step = (max - min) / (count - 1);
        for (int i = 0; i < count; i++) {
            out[i] = min + step * i;
        }
        return out;
    }

    private static Map<String, List<String>> parseExpressionLists(String text) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        Matcher matcher = Pattern.compile("\"([A-Za-z0-9_]+)\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        while (matcher.find()) {
            String key = matcher.group(1);
            String body = matcher.group(2);
            List<String> values = new ArrayList<>();
            Matcher strings = Pattern.compile("\"(.*?)\"").matcher(body);
            while (strings.find()) {
                values.add(strings.group(1));
            }
            if (!values.isEmpty()) {
                out.put(key, values);
            }
        }
        return out;
    }

    private static double jsonDouble(String text, String key, double fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*([-+0-9.eE]+)").matcher(text);
        return matcher.find() ? Double.parseDouble(matcher.group(1)) : fallback;
    }

    private static String jsonString(String text, String key, String fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"").matcher(text);
        return matcher.find() ? matcher.group(1) : fallback;
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    private static String sanitize(String value) {
        return value.replaceAll("[^A-Za-z0-9_]", "_");
    }

    private static String json(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String jsonNumber(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "null";
    }

    private static String jsonArray(List<String> values) {
        List<String> quoted = new ArrayList<>();
        for (String value : values) {
            quoted.add(json(value));
        }
        return "[" + String.join(", ", quoted) + "]";
    }

    private static String jsonIntArray(int[] values) {
        List<String> formatted = new ArrayList<>();
        for (int value : values) {
            formatted.add(String.valueOf(value));
        }
        return "[" + String.join(", ", formatted) + "]";
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = Files.readAllBytes(path);
        byte[] hash = digest.digest(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : hash) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
