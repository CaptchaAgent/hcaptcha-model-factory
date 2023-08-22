## workflow

1. collect dataset

    ```bash
    python whist_binary_collector.py
    ```

2. unpack cache
    
    ```bash
    python whist_unpack.py
    ```

3. Classify the dataset

    AI automatic labeling in the workspace `/database2023/binary_backup/<focus_flag>`

4. Mini workflow

    Edit `factory_mini_workflow:focus_flags` and run the following script

    ```bash
    python factory_mini_workflow.py
    ```
    - copy from: `/database2023/binary_backup/<focus_flag>`
    - paste to: `/data/<focus_flag>`
    - output to: `/model/<focus_flag>/<focus_flag>.onnx`
5. Test challenges
    
    Edit `challenge_with_selenium` and run the following script

    ```bash
    python challenge_with_selenium.py
    ```
   
6. Upload onnx model to GitHub releases

    By `hcaptcha_whistleblower.plugins.github_issues` CI/CD

7. Sign GitHub Issues with challenge-tag
8. Update and push `objects.yaml` of the hcaptcha-challenger