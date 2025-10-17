"""
Kubernetes Operations Tools for Gitcontainer application.

This module provides utilities for interacting with Kubernetes clusters,
including creating pods, streaming logs, and monitoring pod status.
"""

import asyncio
import logging
import os
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
from kubernetes import client, config
from kubernetes.client import V1Pod, V1PodList
from kubernetes.stream import stream
from kubernetes.watch import Watch

logger = logging.getLogger(__name__)


class K8sOperations:
    """
    Kubernetes operations wrapper with async support.
    """
    
    def __init__(self):
        """
        Initialize Kubernetes client using configuration from environment variable.
        """
        try:
            # 尝试从.env文件加载环境变量
            k8s_config_path = None
            try:
                from dotenv import load_dotenv
                load_dotenv()
                k8s_config_path = os.getenv('k8s_config_path')
            except ImportError as import_err:
                logger.warning(f"dotenv module not found or import failed: {import_err}, using default kubeconfig loading")
            
            # 尝试加载Kubernetes配置
            try:
                if k8s_config_path:
                    logger.info(f"Loading Kubernetes config from: {k8s_config_path}")
                    config.load_kube_config(config_file=k8s_config_path)
                else:
                    logger.info("No k8s_config_path found in environment, using default kubeconfig")
                    config.load_kube_config()
            except Exception as outer_e:
                logger.warning(f"Failed to load kubeconfig: {outer_e}, trying in-cluster config")
                try:
                    config.load_incluster_config()
                    logger.info("Successfully loaded in-cluster config")
                except Exception as inner_e:
                    logger.error(f"Failed to load in-cluster config: {inner_e}")
                    raise RuntimeError("Could not load Kubernetes configuration")
            
            # 创建API客户端
            self.core_v1_api = client.CoreV1Api()
            self.batch_v1_api = client.BatchV1Api()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
            raise
    
    async def create_pod(self, name: str, image: str, command: Optional[list] = None,
                        namespace: str = "default", labels: Optional[Dict[str, str]] = None,
                        env_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a pod in Kubernetes.
        
        Args:
            name: Name of the pod
            image: Docker image to use
            command: Command to run in the pod
            namespace: Kubernetes namespace
            labels: Labels to apply to the pod
            env_vars: Environment variables to set
            
        Returns:
            Dict with pod creation result
        """
        try:
            # Create pod spec
            container = client.V1Container(
                name=name,
                image=image,
                command=command or ["/bin/sh", "-c", "echo 'Pod started'"],
                env=[client.V1EnvVar(name=k, value=v) for k, v in env_vars.items()] if env_vars else None,
                image_pull_policy="IfNotPresent"
            )
            
            # Create pod spec with tolerations for potential resource constraints
            pod_spec = client.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                tolerations=[
                    client.V1Toleration(
                        key="node.kubernetes.io/not-ready",
                        operator="Exists",
                        effect="NoSchedule"
                    ),
                    client.V1Toleration(
                        key="node.kubernetes.io/unreachable",
                        operator="Exists",
                        effect="NoSchedule"
                    )
                ]
            )
            
            # Create pod metadata
            metadata = client.V1ObjectMeta(
                name=name,
                labels=labels or {}
            )
            
            # Create pod
            pod = client.V1Pod(
                api_version="v1",
                kind="Pod",
                metadata=metadata,
                spec=pod_spec
            )
            
            # Create pod in Kubernetes
            result = await asyncio.to_thread(
                self.core_v1_api.create_namespaced_pod,
                namespace=namespace,
                body=pod
            )
            
            return {
                "success": True,
                "pod_name": result.metadata.name,
                "namespace": result.metadata.namespace
            }
        except Exception as e:
            logger.error(f"Failed to create pod {name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_pod(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Delete a pod from Kubernetes.
        
        Args:
            name: Name of the pod to delete
            namespace: Kubernetes namespace
            
        Returns:
            Dict with deletion result
        """
        try:
            await asyncio.to_thread(
                self.core_v1_api.delete_namespaced_pod,
                name=name,
                namespace=namespace,
                body=client.V1DeleteOptions(grace_period_seconds=5)
            )
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to delete pod {name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stream_pod_logs(self, name: str, namespace: str = "default",
                            container: Optional[str] = None,
                            follow: bool = True) -> AsyncGenerator[str, None]:
        """
        Stream logs from a pod.
        
        Args:
            name: Name of the pod
            namespace: Kubernetes namespace
            container: Container name
            follow: Whether to follow the logs
            
        Yields:
            Log lines as they are produced
        """
        try:
            # First, wait for the pod to be running
            await self.wait_for_pod_running(name, namespace)
            
            # Get log stream (移除了无限循环)
            logs_stream = await asyncio.to_thread(
                self.core_v1_api.read_namespaced_pod_log,
                name=name,
                namespace=namespace,
                container=container,
                follow=follow,
                _preload_content=False
            )
            
            # Process log chunks
            for line in logs_stream:
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                yield line.rstrip('\n')
                
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {name} not found")
            raise
        except Exception as e:
            logger.error(f"Error streaming logs from pod {name}: {e}")
            yield f"ERROR: {str(e)}"
    
    async def wait_for_pod_completion(self, name: str, namespace: str = "default",
                                    timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for a pod to complete.
        
        Args:
            name: Name of the pod
            namespace: Kubernetes namespace
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict with pod completion status
        """
        try:
            watch = Watch()
            field_selector = f"metadata.name={name}"
            
            # Start a timer
            start_time = asyncio.get_event_loop().time()
            
            # Watch for pod events
            while True:
                # Check if timeout has been reached
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    return {
                        "success": False,
                        "error": f"Timeout waiting for pod {name} to complete"
                    }
                
                # Get pod status
                pod = await asyncio.to_thread(
                    self.core_v1_api.read_namespaced_pod,
                    name=name,
                    namespace=namespace
                )
                
                # Check pod phase
                if pod.status.phase == "Succeeded":
                    return {
                        "success": True,
                        "status": "Succeeded",
                        "exit_code": pod.status.container_statuses[0].state.terminated.exit_code
                    }
                elif pod.status.phase == "Failed":
                    return {
                        "success": False,
                        "status": "Failed",
                        "exit_code": pod.status.container_statuses[0].state.terminated.exit_code,
                        "message": pod.status.container_statuses[0].state.terminated.message
                    }
                
                # Wait before checking again
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error waiting for pod {name} completion: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def wait_for_pod_running(self, name: str, namespace: str = "default",
                                timeout: int = 120) -> bool:
        """
        Wait for a pod to be in Running state.
        
        Args:
            name: Name of the pod
            namespace: Kubernetes namespace
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if pod is running, False otherwise
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            while True:
                # Check if timeout has been reached
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.warning(f"Timeout waiting for pod {name} to run")
                    return False
                
                # Get pod status
                pod = await asyncio.to_thread(
                    self.core_v1_api.read_namespaced_pod,
                    name=name,
                    namespace=namespace
                )
                
                # Check pod status
                if pod.status.phase == "Running":
                    return True
                elif pod.status.phase == "Failed":
                    logger.error(f"Pod {name} failed to start")
                    return False
                
                # Wait before checking again
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error waiting for pod {name} to run: {e}")
            return False


def create_test_pod_name(base_name: str) -> str:
    """
    Create a unique pod name for testing.
    
    Args:
        base_name: Base name for the pod
        
    Returns:
        Unique pod name
    """
    # Generate a unique suffix
    suffix = str(uuid.uuid4())[:8]
    # Remove invalid characters for Kubernetes resource names
    safe_name = base_name.lower().replace('_', '-').replace('.', '-')
    # Truncate if necessary and add suffix
    max_length = 63 - len(suffix) - 1  # Leave room for the hyphen
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    return f"{safe_name}-test-{suffix}"


async def run_test_in_k8s(image: str, test_command: Optional[list] = None,
                        ws_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run a test in Kubernetes.
    
    Args:
        image: Docker image to use for testing
        test_command: Command to run for testing
        ws_manager: WebSocket manager for sending logs
        
    Returns:
        Dict with test result
    """
    # Extract image name for pod naming
    image_name = image.split('/')[-1].split(':')[0]
    pod_name = create_test_pod_name(image_name)
    
    try:
        # Initialize K8s client
        k8s = K8sOperations()
        
        # Create pod
        if ws_manager:
            await ws_manager.send_build_log(f"🚀 创建测试Pod: {pod_name}\n")
        
        create_result = await k8s.create_pod(
            name=pod_name,
            image=image,
            command=test_command or ["/bin/sh", "-c", "echo 'Test completed successfully'"],
            labels={"app": "gitcontainer-test", "component": "test"}
        )
        
        if not create_result["success"]:
            if ws_manager:
                await ws_manager.send_error(f"❌ 创建Pod失败: {create_result['error']}\n")
            return {
                "success": False,
                "error": f"创建Pod失败: {create_result['error']}",
                "pod_name": pod_name
            }
        
        # Stream logs
        if ws_manager:
            await ws_manager.send_build_log(f"📝 开始收集测试日志...\n")
        
        log_lines = []
        async for line in k8s.stream_pod_logs(pod_name):
            log_lines.append(line)
            if ws_manager:
                await ws_manager.send_build_log(f"{line}\n")
        
        # Wait for pod completion
        if ws_manager:
            await ws_manager.send_build_log(f"⏳ 等待测试完成...\n")
        
        completion_result = await k8s.wait_for_pod_completion(pod_name)
        
        # Clean up pod
        if ws_manager:
            await ws_manager.send_build_log(f"🧹 清理测试Pod...\n")
        
        delete_result = await k8s.delete_pod(pod_name)
        if not delete_result["success"]:
            logger.warning(f"Failed to delete pod {pod_name}: {delete_result['error']}")
        
        # Combine logs into output
        output = "\n".join(log_lines)
        
        # Return test result
        if completion_result["success"]:
            if ws_manager:
                await ws_manager.send_build_log(f"✅ 测试成功完成!\n")
            return {
                "success": True,
                "pod_name": pod_name,
                "logs": output,
                "output": output,
                "exit_code": completion_result.get("exit_code", 0)
            }
        else:
            error_msg = completion_result.get("message", "测试失败")
            if ws_manager:
                await ws_manager.send_error(f"❌ 测试失败: {error_msg}\n")
            return {
                "success": False,
                "error": error_msg,
                "pod_name": pod_name,
                "logs": output,
                "output": output,
                "exit_code": completion_result.get("exit_code", 1)
            }
    
    except Exception as e:
        # Try to clean up pod if it exists
        try:
            k8s = K8sOperations()
            await k8s.delete_pod(pod_name)
        except:
            pass
        
        error_msg = str(e)
        if ws_manager:
            await ws_manager.send_error(f"❌ 测试过程中出错: {error_msg}\n")
        return {
            "success": False,
            "error": error_msg,
            "pod_name": pod_name
        }