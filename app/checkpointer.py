"""Checkpointer async sobre Redis estandar; no requiere RedisJSON/RediSearch.

Implementa el contrato BaseCheckpointSaver. Conserva checkpoints completos,
metadatos, parent_config y pending_writes (incluyendo interrupt y resume).
No usa pickle. No implementa borrado/pruning: se preserva el historial.
"""
import base64
import json
from langgraph.checkpoint.base import BaseCheckpointSaver,CheckpointTuple,WRITES_IDX_MAP


class RedisCheckpointSaver(BaseCheckpointSaver):
    def __init__(self,redis,prefix="intelligence:cp"):
        super().__init__()
        self.redis,self.prefix=redis,prefix

    def _key(self,config):
        c=config["configurable"]
        identity=json.dumps([c["thread_id"],c.get("checkpoint_ns","")])
        return self.prefix+":"+base64.urlsafe_b64encode(identity.encode()).decode()

    @staticmethod
    def _config(config,cid=None):
        values=config["configurable"]
        result={"thread_id":values["thread_id"],"checkpoint_ns":values.get("checkpoint_ns","")}
        checkpoint_id=cid or values.get("checkpoint_id")
        if checkpoint_id:
            result["checkpoint_id"]=checkpoint_id
        return {"configurable":result}

    def _dump(self,value):
        kind,data=self.serde.dumps_typed(value)
        return json.dumps([kind,base64.b64encode(data).decode()])

    def _load(self,value):
        kind,data=json.loads(value)
        return self.serde.loads_typed((kind,base64.b64decode(data)))

    async def aput(self,config,checkpoint,metadata,new_versions):
        key=self._key(config)
        cid=checkpoint["id"]
        current=self._config(config,cid)
        parent=self._config(config) if config["configurable"].get("checkpoint_id") else None
        packed=self._dump({"config":current,"checkpoint":checkpoint,"metadata":metadata,"parent":parent})
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(f"{key}:{cid}",packed)
            pipe.zadd(key+":ids",{cid:0})  # UUID checkpoint sortable lexicograficamente.
            await pipe.execute()
        return current

    async def aget_tuple(self,config):
        key=self._key(config)
        cid=config["configurable"].get("checkpoint_id")
        if not cid:
            ids=await self.redis.zrevrange(key+":ids",0,0)
            if not ids:
                return None
            cid=ids[0].decode()
        packed=await self.redis.get(f"{key}:{cid}")
        if packed is None:
            return None
        row=self._load(packed)
        writes=await self.redis.hgetall(f"{key}:{cid}:writes")
        return CheckpointTuple(config=row["config"],checkpoint=row["checkpoint"],
            metadata=row["metadata"],parent_config=row["parent"],
            pending_writes=[self._load(v) for _,v in sorted(writes.items())])

    async def aput_writes(self,config,writes,task_id,task_path=""):
        key=self._key(config)+":"+config["configurable"]["checkpoint_id"]+":writes"
        async with self.redis.pipeline(transaction=True) as pipe:
            for i,(channel,value) in enumerate(writes):
                idx=WRITES_IDX_MAP.get(channel,i)
                field=f"{task_id}:{idx}"
                data=self._dump((task_id,channel,value))
                if idx<0:
                    pipe.hset(key,field,data)
                else:
                    pipe.hsetnx(key,field,data)
            await pipe.execute()

    async def alist(self,config,*,filter=None,before=None,limit=None):
        if config is None:
            raise ValueError("alist requiere thread_id para limitar el alcance")
        key=self._key(config)
        ids=await self.redis.zrevrange(key+":ids",0,-1)
        before_id=(before or {}).get("configurable",{}).get("checkpoint_id")
        count=0
        for raw in ids:
            cid=raw.decode()
            if before_id and cid>=before_id:
                continue
            row=await self.aget_tuple({"configurable":{**config["configurable"],"checkpoint_id":cid}})
            if row and (not filter or all(row.metadata.get(k)==v for k,v in filter.items())):
                yield row
                count+=1
                if limit is not None and count>=limit:
                    return
